#!/usr/bin/env python3
"""Offline field replay gate for the recall-map polish.

Every manifest case is replayed through the real retrieval service and the
real :class:`living_memory.recall_map.RecallMapBuilder` against the manifest's
pinned SQLite snapshot.  The complete replay is run twice from fresh working
copies so ordering determinism is measured rather than inferred.

The output is aggregate-only by split.  In particular, holdout queries,
labels, medoids, payloads, and case identifiers never leave this process.

Usage::

    python3 scripts/recall_map_polish_eval.py

The default report is ``artifacts/recall-map/polish-eval.json``.  Exit status
is zero only when every P2 threshold passes independently on both ``eval`` and
``holdout``, and the source contains no diagnosed field label hardcoded as a
label constant.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from living_memory.config import MemoryConfig  # noqa: E402
from living_memory.recall_map import (  # noqa: E402
    MAX_RESPONSE_CHARS,
    MIN_MEDOID_EXAMPLE_CHARS,
    RecallMap,
    RecallMapBuilder,
    _collapse,
    _node_subsystem,
    _terms,
    normalize_key,
)
from living_memory.retrieval import MemoryRecallService  # noqa: E402
from living_memory.retrieval_harness import sha256_file, working_copy  # noqa: E402
from living_memory.storage import MemoryStore  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "artifacts/recall-map/field/manifest.json"
DEFAULT_DIAGNOSIS = REPO_ROOT / "artifacts/recall-map/field/diagnosis.md"
DEFAULT_SOURCE = REPO_ROOT / "src/living_memory/recall_map.py"
DEFAULT_OUT = REPO_ROOT / "artifacts/recall-map/polish-eval.json"

EXPECTED_SPLITS = ("train", "eval", "holdout")
ENFORCED_SPLITS = ("eval", "holdout")
MIN_MULTIWORD_CONTENT_RATE = 0.90
REPLAY_RUNS = 2


class EvalError(RuntimeError):
    """The inputs cannot support a valid replay evaluation."""


@dataclass(frozen=True, slots=True)
class CaseObservation:
    """Aggregate-safe facts retained from one case.

    ``case_id`` is used only to join the two in-memory runs.  Reports contain
    counts, never identifiers or semantic material from individual cases.
    """

    case_id: str
    split: str
    source: str
    head_reproduced: bool
    head_signature: tuple[str, ...]
    cache_hit: bool
    outcome: str
    order_signature: tuple[tuple[str, str], ...]
    payload_signature: str
    delivered_clusters: int
    degenerate_labels: int
    multiword_content_labels: int
    gist_floor_met: int
    gist_distinct_from_label: int
    filtering_occurred: bool
    journal_complete: bool
    payload_within_budget: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _depth(raw: Any) -> Any:
    if raw is None:
        return 1
    try:
        return int(str(raw))
    except ValueError:
        return str(raw)


def _read_object(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise EvalError(f"{name} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvalError(f"cannot read {name}: {error}") from error
    if not isinstance(value, dict):
        raise EvalError(f"{name} must be a JSON object")
    return value


def _validate_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_cases = manifest.get("cases")
    snapshots = manifest.get("snapshots")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise EvalError("manifest cases must be a non-empty list")
    if not isinstance(snapshots, dict) or not snapshots:
        raise EvalError("manifest snapshots must be a non-empty object")

    required = {
        "case_id",
        "split",
        "source",
        "query",
        "requested_scope",
        "scope",
        "depth",
        "max_results",
        "ambient_context",
        "task",
        "task_pattern",
        "head_node_ids",
        "created_at",
    }
    seen: set[str] = set()
    cases: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    for ordinal, raw in enumerate(raw_cases, 1):
        if not isinstance(raw, dict):
            raise EvalError(f"manifest case {ordinal} must be an object")
        missing = sorted(required - set(raw))
        if missing:
            raise EvalError(
                f"manifest case {ordinal} lacks required keys: {', '.join(missing)}"
            )
        case_id = raw["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise EvalError(f"manifest case {ordinal} has an invalid or duplicate case_id")
        seen.add(case_id)
        split = raw["split"]
        source = raw["source"]
        if split not in EXPECTED_SPLITS:
            raise EvalError(f"manifest case {ordinal} has unknown split {split!r}")
        if not isinstance(source, str) or source not in snapshots:
            raise EvalError(f"manifest case {ordinal} names an unknown snapshot source")
        if not isinstance(raw["query"], str) or not raw["query"].strip():
            raise EvalError(f"manifest case {ordinal} has an empty query")
        if not isinstance(raw["max_results"], int) or raw["max_results"] <= 0:
            raise EvalError(f"manifest case {ordinal} has invalid max_results")
        if raw["ambient_context"] is not None and not isinstance(
            raw["ambient_context"], dict
        ):
            raise EvalError(f"manifest case {ordinal} has invalid ambient_context")
        if not isinstance(raw["head_node_ids"], list):
            raise EvalError(f"manifest case {ordinal} has invalid head_node_ids")
        split_counts[split] += 1
        cases.append(dict(raw))

    missing_splits = [split for split in EXPECTED_SPLITS if split_counts[split] <= 0]
    if missing_splits:
        raise EvalError(f"manifest has no cases for split(s): {', '.join(missing_splits)}")
    return cases


def _snapshot_paths(
    manifest: Mapping[str, Any], overrides: Mapping[str, Path]
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    snapshots = manifest["snapshots"]
    resolved: dict[str, Path] = {}
    attestations: dict[str, dict[str, Any]] = {}
    for source, raw in sorted(snapshots.items()):
        if not isinstance(raw, dict):
            raise EvalError(f"snapshot entry {source!r} must be an object")
        path = overrides.get(source, Path(str(raw.get("path") or ""))).expanduser()
        if not path.is_file():
            raise EvalError(
                f"snapshot {source!r} not found at {path}; use --snapshot {source}=PATH"
            )
        expected = str((raw.get("manifest") or {}).get("snapshot_sha256") or "")
        actual = sha256_file(path)
        if not expected:
            raise EvalError(f"snapshot {source!r} has no pinned sha256")
        if actual != expected:
            raise EvalError(
                f"snapshot {source!r} sha256 mismatch: expected {expected}, got {actual}"
            )
        resolved[source] = path.resolve()
        attestations[source] = {
            "sha256": actual,
            "bytes": path.stat().st_size,
            "verified": True,
        }
    unknown = sorted(set(overrides) - set(resolved))
    if unknown:
        raise EvalError(f"snapshot override names unknown source(s): {', '.join(unknown)}")
    return resolved, attestations


_DIAGNOSIS_ROW = re.compile(
    r"^\|\s*`(?P<label>[^`]+)`\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(?P<terms>\d+)\s*\|"
)


def _diagnosed_degenerate_labels(path: Path) -> frozenset[str]:
    """Read the diagnosis frequency table's single-content-term labels.

    These labels are used only to audit source hardcoding.  Replay quality is
    scored from corpus statistics and medoid structure, so the diagnosis is
    never an eval/holdout answer key.
    """

    if not path.is_file():
        raise EvalError(f"diagnosis not found: {path}")
    labels: set[str] = set()
    in_frequency_table = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## 3. "):
            in_frequency_table = True
            continue
        if in_frequency_table and line.startswith("## "):
            break
        if not in_frequency_table:
            continue
        match = _DIAGNOSIS_ROW.match(line)
        if match and int(match.group("terms")) == 1:
            label = _collapse(match.group("label")).casefold()
            if label:
                labels.add(label)
    if not labels:
        raise EvalError("diagnosis yielded no single-term field labels for hardcoding audit")
    return frozenset(labels)


def _docstring_nodes(tree: ast.AST) -> set[ast.Constant]:
    found: set[ast.Constant] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            found.add(body[0].value)
    return found


def _assignment_targets(node: ast.AST) -> set[str]:
    targets: Sequence[ast.AST]
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = (node.target,)
    else:
        return set()
    return {part.id for target in targets for part in ast.walk(target) if isinstance(part, ast.Name)}


def _hardcoded_label_constants(
    source_path: Path, diagnosed: frozenset[str]
) -> list[dict[str, Any]]:
    """Find diagnosed terms embedded in executable label constants.

    Comments and docstrings are evidence and are deliberately ignored.  The
    pre-existing path-root vocabulary is also not a label constant: it tells
    ``_subsystem`` where a repository root ends.  Every other executable
    literal containing a diagnosed term is a violation, which catches direct
    assignments, comparisons, return values, and constructor arguments rather
    than checking one preferred spelling of a blacklist variable.
    """

    if not source_path.is_file():
        raise EvalError(f"recall-map source not found: {source_path}")
    source = source_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(source_path))
    except SyntaxError as error:
        raise EvalError(f"cannot parse recall-map source for hardcoding audit: {error}") from error
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    docstrings = _docstring_nodes(tree)
    violations: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Constant)
            or not isinstance(node.value, str)
            or node in docstrings
        ):
            continue
        literal_terms = set(_terms(node.value))
        matched = sorted(literal_terms & diagnosed)
        if not matched:
            continue

        ancestor = parents.get(node)
        allowed_path_vocabulary = False
        while ancestor is not None:
            targets = _assignment_targets(ancestor)
            if "_PATH_ROOT_SEGMENTS" in targets:
                allowed_path_vocabulary = True
                break
            if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                break
            ancestor = parents.get(ancestor)
        if allowed_path_vocabulary:
            continue
        violations.append({"line": node.lineno, "terms": matched})
    return sorted(violations, key=lambda item: (item["line"], item["terms"]))


def _normal(text: str) -> str:
    return _collapse(text).casefold()


def _journal_complete(built: RecallMap, payload: Mapping[str, Any]) -> bool:
    filtering = built.withheld > 0 or built.dropped > 0
    block = payload.get("filtered")
    if not filtering:
        return block is None
    if not isinstance(block, dict):
        return False
    if set(("withheld", "dropped")) - set(block):
        return False
    if block.get("withheld") != built.withheld or block.get("dropped") != built.dropped:
        return False
    if built.dropped > 0 and payload.get("more") != built.dropped:
        return False

    names = block.get("names", [])
    omitted = block.get("names_omitted", 0)
    if not isinstance(names, list) or not isinstance(omitted, int) or omitted < 0:
        return False
    if any(
        not isinstance(entry, list)
        or len(entry) != 3
        or entry[0] not in ("w", "d")
        or not isinstance(entry[1], str)
        or not isinstance(entry[2], int)
        or entry[2] <= 0
        for entry in names
    ):
        return False
    return len(names) + omitted == built.withheld + built.dropped


def _observe_case(
    *,
    case: Mapping[str, Any],
    head_ids: tuple[str, ...],
    builder: RecallMapBuilder,
    built: RecallMap | None,
    store: MemoryStore,
) -> CaseObservation:
    expected_head = tuple(str(value) for value in case["head_node_ids"])
    if built is None:
        return CaseObservation(
            case_id=str(case["case_id"]),
            split=str(case["split"]),
            source=str(case["source"]),
            head_reproduced=head_ids == expected_head,
            head_signature=head_ids,
            cache_hit=builder.last_cache_hit,
            outcome="none",
            order_signature=(),
            payload_signature="null",
            delivered_clusters=0,
            degenerate_labels=0,
            multiword_content_labels=0,
            gist_floor_met=0,
            gist_distinct_from_label=0,
            filtering_occurred=False,
            journal_complete=True,
            payload_within_budget=True,
        )

    payload = built.to_dict()
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    outcome = "curtailed" if built.curtailed else "delivered"
    order = tuple((cluster.label, cluster.medoid.node_id) for cluster in built.clusters)
    degenerate = 0
    multiword = 0
    gist_floor = 0
    gist_distinct = 0
    for cluster in built.clusters:
        terms = set(_terms(cluster.label))
        if len(terms) >= 2:
            multiword += 1

        node = store.get_node(cluster.medoid.node_id)
        subsystem = _node_subsystem(node) if node is not None else None
        bare_path = subsystem is not None and _normal(cluster.label) == _normal(
            normalize_key(subsystem)
        )
        single_generic_or_stop = len(terms) <= 1 and not builder._deliverable(
            cluster.label
        )
        if bare_path or single_generic_or_stop:
            degenerate += 1

        gist = _collapse(cluster.medoid.example)
        source_text = _collapse(node.content) if node is not None else ""
        required = min(MIN_MEDOID_EXAMPLE_CHARS, len(source_text))
        if required > 0 and len(gist) >= required:
            gist_floor += 1
        if gist and _normal(gist) != _normal(cluster.label):
            gist_distinct += 1

    filtering = built.withheld > 0 or built.dropped > 0
    return CaseObservation(
        case_id=str(case["case_id"]),
        split=str(case["split"]),
        source=str(case["source"]),
        head_reproduced=head_ids == expected_head,
        head_signature=head_ids,
        cache_hit=builder.last_cache_hit,
        outcome=outcome,
        order_signature=order,
        payload_signature=payload_json,
        delivered_clusters=len(built.clusters),
        degenerate_labels=degenerate,
        multiword_content_labels=multiword,
        gist_floor_met=gist_floor,
        gist_distinct_from_label=gist_distinct,
        filtering_occurred=filtering,
        journal_complete=_journal_complete(built, payload),
        payload_within_budget=len(payload_json) <= MAX_RESPONSE_CHARS,
    )


def _replay_once(
    cases: Sequence[Mapping[str, Any]],
    snapshots: Mapping[str, Path],
    *,
    run_number: int,
    progress_every: int,
) -> tuple[dict[str, CaseObservation], dict[str, float]]:
    observations: dict[str, CaseObservation] = {}
    timings: dict[str, float] = {}
    completed = 0
    total = len(cases)
    for source, snapshot in sorted(snapshots.items()):
        source_cases = sorted(
            (case for case in cases if case["source"] == source),
            key=lambda case: (str(case["created_at"]), str(case["case_id"])),
        )
        started = time.monotonic()
        with working_copy(snapshot) as working:
            with MemoryStore(MemoryConfig(db_path=str(working))) as store:
                service = MemoryRecallService(store)
                builder = RecallMapBuilder(store)
                for case in source_cases:
                    head = service.memory_recall(
                        query=str(case["query"]),
                        scope=case["requested_scope"],
                        depth=_depth(case["depth"]),
                        max_results=int(case["max_results"]),
                        ambient_context=case["ambient_context"] or None,
                        log_access=False,
                        log_event=False,
                    )
                    built = builder.build(
                        service.last_residual,
                        scope=case["scope"],
                        task=case["task"],
                        task_pattern=case["task_pattern"],
                    )
                    observation = _observe_case(
                        case=case,
                        head_ids=tuple(result.node.id for result in head),
                        builder=builder,
                        built=built,
                        store=store,
                    )
                    observations[observation.case_id] = observation
                    completed += 1
                    if progress_every > 0 and (
                        completed % progress_every == 0 or completed == total
                    ):
                        print(
                            f"replay run {run_number}/{REPLAY_RUNS}: {completed}/{total} cases",
                            file=sys.stderr,
                            flush=True,
                        )
        timings[source] = round(time.monotonic() - started, 3)
    if len(observations) != len(cases):
        raise EvalError(
            f"replay run {run_number} observed {len(observations)} of {len(cases)} cases"
        )
    return observations, timings


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator > 0 else None


def _threshold(
    *, actual: float | int | None, comparator: str, target: float | int, passed: bool
) -> dict[str, Any]:
    return {
        "actual": actual,
        "comparator": comparator,
        "target": target,
        "passed": passed,
    }


def _split_report(
    split: str,
    first: Sequence[CaseObservation],
    second_by_id: Mapping[str, CaseObservation],
) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    for item in first:
        other = second_by_id[item.case_id]
        counters["cases"] += 1
        counters[f"outcome_{item.outcome}"] += 1
        counters["head_reproduced_cases"] += int(item.head_reproduced)
        counters["cache_hit_cases"] += int(item.cache_hit)
        counters["delivered_clusters"] += item.delivered_clusters
        counters["degenerate_label_clusters"] += item.degenerate_labels
        counters["multiword_content_clusters"] += item.multiword_content_labels
        counters["gist_floor_clusters"] += item.gist_floor_met
        counters["gist_distinct_clusters"] += item.gist_distinct_from_label
        counters["filtering_maps"] += int(item.filtering_occurred)
        counters["complete_filter_journals"] += int(
            item.filtering_occurred and item.journal_complete
        )
        counters["journal_failures"] += int(
            item.filtering_occurred and not item.journal_complete
        )
        counters["payload_budget_failures"] += int(not item.payload_within_budget)
        counters["ordering_mismatch_cases"] += int(
            item.outcome != other.outcome
            or item.order_signature != other.order_signature
        )
        counters["payload_mismatch_cases"] += int(
            item.payload_signature != other.payload_signature
        )
        counters["head_mismatch_between_runs"] += int(
            item.head_signature != other.head_signature
        )

    clusters = counters["delivered_clusters"]
    filtering_maps = counters["filtering_maps"]
    multiword_rate = _rate(counters["multiword_content_clusters"], clusters)
    gist_floor_rate = _rate(counters["gist_floor_clusters"], clusters)
    gist_distinct_rate = _rate(counters["gist_distinct_clusters"], clusters)
    journal_rate = _rate(counters["complete_filter_journals"], filtering_maps)
    deterministic_rate = _rate(
        counters["cases"] - counters["ordering_mismatch_cases"], counters["cases"]
    )

    evaluable = clusters > 0
    thresholds = {
        "no_degenerate_labels": _threshold(
            actual=counters["degenerate_label_clusters"],
            comparator="<=",
            target=0,
            passed=evaluable and counters["degenerate_label_clusters"] == 0,
        ),
        "multiword_content_rate": _threshold(
            actual=multiword_rate,
            comparator=">=",
            target=MIN_MULTIWORD_CONTENT_RATE,
            passed=evaluable
            and multiword_rate is not None
            and multiword_rate >= MIN_MULTIWORD_CONTENT_RATE,
        ),
        "gist_design_floor_rate": _threshold(
            actual=gist_floor_rate,
            comparator=">=",
            target=1.0,
            passed=evaluable and gist_floor_rate == 1.0,
        ),
        "gist_distinct_from_label_rate": _threshold(
            actual=gist_distinct_rate,
            comparator=">=",
            target=1.0,
            passed=evaluable and gist_distinct_rate == 1.0,
        ),
        "filter_journaling_rate": _threshold(
            actual=journal_rate if filtering_maps else 1.0,
            comparator=">=",
            target=1.0,
            passed=counters["journal_failures"] == 0,
        ),
        "ordering_mismatch_cases": _threshold(
            actual=counters["ordering_mismatch_cases"],
            comparator="<=",
            target=0,
            passed=counters["cases"] > 0
            and counters["ordering_mismatch_cases"] == 0,
        ),
    }
    enforced = split in ENFORCED_SPLITS
    thresholds_met = all(check["passed"] for check in thresholds.values())
    return {
        "enforced": enforced,
        "evaluable": evaluable,
        "cases": counters["cases"],
        "map_outcomes": {
            "delivered": counters["outcome_delivered"],
            "curtailed": counters["outcome_curtailed"],
            "none": counters["outcome_none"],
        },
        "head_replay": {
            "reproduced_cases": counters["head_reproduced_cases"],
            "rate": _rate(counters["head_reproduced_cases"], counters["cases"]),
            "mismatches_between_runs": counters["head_mismatch_between_runs"],
        },
        "cache_hit_cases": counters["cache_hit_cases"],
        "delivered_clusters": clusters,
        "label_quality": {
            "degenerate_clusters": counters["degenerate_label_clusters"],
            "multiword_content_clusters": counters["multiword_content_clusters"],
            "multiword_content_rate": multiword_rate,
        },
        "gist_quality": {
            "design_floor_chars": MIN_MEDOID_EXAMPLE_CHARS,
            "floor_met_clusters": counters["gist_floor_clusters"],
            "floor_met_rate": gist_floor_rate,
            "distinct_from_label_clusters": counters["gist_distinct_clusters"],
            "distinct_from_label_rate": gist_distinct_rate,
        },
        "filter_journaling": {
            "maps_with_filtering": filtering_maps,
            "complete_maps": counters["complete_filter_journals"],
            "complete_rate": journal_rate if filtering_maps else 1.0,
            "failures": counters["journal_failures"],
        },
        "determinism": {
            "runs": REPLAY_RUNS,
            "cases_compared": counters["cases"],
            "ordering_mismatch_cases": counters["ordering_mismatch_cases"],
            "ordering_deterministic_rate": deterministic_rate,
            "payload_mismatch_cases": counters["payload_mismatch_cases"],
        },
        "payload_budget_failures": counters["payload_budget_failures"],
        "thresholds": thresholds,
        "thresholds_met": thresholds_met,
        "passed": thresholds_met if enforced else None,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    diagnosis_path = Path(args.diagnosis).resolve()
    source_path = Path(args.source).resolve()
    manifest = _read_object(manifest_path, name="field manifest")
    cases = _validate_manifest(manifest)
    overrides = _parse_snapshot_overrides(args.snapshot)
    snapshots, snapshot_attestations = _snapshot_paths(manifest, overrides)

    diagnosed = _diagnosed_degenerate_labels(diagnosis_path)
    hardcoded = _hardcoded_label_constants(source_path, diagnosed)

    started = time.monotonic()
    first, first_timing = _replay_once(
        cases,
        snapshots,
        run_number=1,
        progress_every=args.progress_every,
    )
    second, second_timing = _replay_once(
        cases,
        snapshots,
        run_number=2,
        progress_every=args.progress_every,
    )
    split_reports = {
        split: _split_report(
            split,
            [item for item in first.values() if item.split == split],
            second,
        )
        for split in EXPECTED_SPLITS
    }
    hardcoding_check = {
        "diagnosed_single_term_labels": len(diagnosed),
        "executable_label_literal_violations": hardcoded,
        "passed": not hardcoded,
    }
    passed = hardcoding_check["passed"] and all(
        split_reports[split]["thresholds_met"] for split in ENFORCED_SPLITS
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "purpose": "Offline replay gate for recall-map polish P2 thresholds.",
        "privacy": {
            "holdout_output": "aggregate_only",
            "case_level_material_emitted": False,
        },
        "inputs": {
            "manifest": str(manifest_path.relative_to(REPO_ROOT)),
            "manifest_sha256": _sha256(manifest_path),
            "diagnosis": str(diagnosis_path.relative_to(REPO_ROOT)),
            "diagnosis_sha256": _sha256(diagnosis_path),
            "snapshots": snapshot_attestations,
        },
        "code": {
            "commit": _git("rev-parse", "HEAD"),
            "worktree_dirty": bool(_git("status", "--porcelain")),
            "measured_source_sha256": {
                str(source_path.relative_to(REPO_ROOT)): _sha256(source_path),
                str(Path(__file__).resolve().relative_to(REPO_ROOT)): _sha256(
                    Path(__file__).resolve()
                ),
            },
        },
        "protocol": {
            "runs": REPLAY_RUNS,
            "fresh_working_copy_per_snapshot_per_run": True,
            "replay_order": "created_at_then_case_id_within_snapshot",
            "real_retrieval_service": True,
            "real_recall_map_builder": True,
            "quality_scoring_uses_diagnosed_label_lookup": False,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "run_seconds_by_snapshot": {
                "run_1": first_timing,
                "run_2": second_timing,
            },
        },
        "thresholds": {
            "enforced_splits": list(ENFORCED_SPLITS),
            "maximum_degenerate_label_clusters": 0,
            "minimum_multiword_content_rate": MIN_MULTIWORD_CONTENT_RATE,
            "minimum_gist_design_floor_rate": 1.0,
            "minimum_gist_distinct_from_label_rate": 1.0,
            "minimum_filter_journaling_rate_when_filtering_occurs": 1.0,
            "maximum_ordering_mismatch_cases": 0,
        },
        "hardcoding_check": hardcoding_check,
        "splits": split_reports,
        "passed": passed,
    }


def _parse_snapshot_overrides(values: Iterable[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for raw in values:
        source, separator, path = raw.partition("=")
        if not separator or not source or not path:
            raise EvalError(f"invalid --snapshot {raw!r}; expected SOURCE=PATH")
        if source in overrides:
            raise EvalError(f"duplicate --snapshot override for {source!r}")
        overrides[source] = Path(path)
    return overrides


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--diagnosis", default=str(DEFAULT_DIAGNOSIS))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--snapshot",
        action="append",
        default=[],
        metavar="SOURCE=PATH",
        help="override one manifest snapshot path while retaining its pinned sha256",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="print aggregate progress every N cases; 0 disables progress",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.progress_every < 0:
        print("ERROR: --progress-every must be non-negative", file=sys.stderr)
        return 2
    try:
        report = evaluate(args)
        _write_report(Path(args.out), report)
    except EvalError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        f"recall-map polish replay: {'PASS' if report['passed'] else 'FAIL'}; "
        f"report={Path(args.out)}",
        file=sys.stderr,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
