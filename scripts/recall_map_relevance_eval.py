#!/usr/bin/env python3
"""Deterministic, read-only historical relevance evaluation.

The evaluator intentionally has no item-level output.  Node ids, event ids,
queries, labels, task names, cache keys, sessions, and host paths exist only in
memory while cohorts are constructed.  The two JSON artifacts contain source
hashes, opaque component digests, denominators, feature aggregates, and the one
organic-train-fitted model.

The frozen effect implementation is executable protocol, not documentation.
Before importing it, :func:`load_frozen_bindings` verifies the byte hashes of
both it and its preregistration.  Only then are ``Protocol``, ``ConsumerIndex``,
``organic_items``, ``map_items``, ``AnchorStratum``, and ``score_pairs`` used.

Typical use::

    PYTHONPATH=src python3 scripts/recall_map_relevance_eval.py run \
      --source local=artifacts/recall-map/relevance/snapshots/local.sqlite3 \
      --manifest-out artifacts/recall-map/relevance/dataset-manifest.json \
      --analysis-out artifacts/recall-map/relevance/feature-analysis.json

The candidate-only holdout directory is a fail-closed forbidden input and
output in every command in this script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = REPO_ROOT / "artifacts" / "recall-map" / "prereg.json"
DEFAULT_EFFECT_TOOL = REPO_ROOT / "scripts" / "recall_map_effect.py"
DEFAULT_BASELINE = REPO_ROOT / "artifacts" / "recall-map" / "baseline.json"
DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "recall-map" / "relevance" / "dataset-manifest.json"
DEFAULT_ANALYSIS = REPO_ROOT / "artifacts" / "recall-map" / "relevance" / "feature-analysis.json"

AS_OF = "2026-08-22T18:00:00Z"
EXPECTED_PREREG_SHA256 = "9eb6a170459bb68138a972b4ba76b76abb044cbb00024cdb04e38e038a85b1b0"
EXPECTED_EFFECT_SHA256 = "e92aceeafb27ae736378f1b1866886cf8446b1cf17c1269a7cd6ad61a039eeea"
SPLIT_SEED = "recall-map-relevance-components-v1"
ORGANIC_TRAIN_FRACTION = 0.80
TRANSFER_FLOOR = 0.233
MODEL_ITERATIONS = 1_200
MODEL_LEARNING_RATE = 0.05
MODEL_L2 = 0.01
MIN_THRESHOLD_TRAIN_FRACTION = 0.20

HOLDOUT_MARKERS = (
    "artifacts/recall-map/relevance/field",
    "candidate_holdout",
    "candidate-holdout",
)
SOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
STRATEGY_STAGNATION_RE = re.compile(
    r"^Strategy stagnation detected on [^\r\n]{1,160}(?:\r?\n"
    r"(?:attempts|strategy|window|reason):[^\r\n]{1,160}){0,4}\s*$"
)
FILE_CHUNK_INDEX_RE = re.compile(r"^[1-9][0-9]*/[1-9][0-9]*$")
FILE_CHUNK_LINES_RE = re.compile(r"^[1-9][0-9]*-[1-9][0-9]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OPAQUE_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
SUPERVISION_KINDS = frozenset(
    {"supervision_journal", "monitoring_journal", "execution_journal", "goal_tree_journal"}
)
SCORE_FIELDS = ("score", "bm25_score", "vector_score", "graph_score", "trigger_score")
BALLAST_CONTROL_SEED = 20260823
BALLAST_CONTROL_VARIANTS_PER_CLASS = 32
FEATURE_FAMILIES: tuple[str, ...] = (
    "snapshot_content_form",
    "historical_context_provenance",
    "node_age",
    "level",
    "snapshot_context_shape",
    "cascade_stage",
    "recorded_delivery_scores",
    "past_nonconsumption",
)

# This is the complete feature surface handed to the primary model.  An
# explicit allowlist makes accidental metadata leakage fail closed.
PRIMARY_MODEL_FEATURES: tuple[str, ...] = (
    "node_age_log_days",
    "level_trace",
    "level_concept",
    "level_schema",
    "prior_matured_log_count",
    "prior_nonconsumed_log_count",
    "prior_nonconsumption_streak_log",
    "prior_consumption_rate",
)
ANALYSIS_ONLY_FEATURES: tuple[str, ...] = (
    # Node content, context and provenance are absent from the delivery
    # envelope and mutable in the live schema.  These snapshot-only
    # diagnostics may describe the frozen corpus, but cannot enter fitting.
    "form_file_chunk_envelope",
    "form_strategy_stagnation",
    "form_supervision_journal",
    "form_machine_ballast",
    "content_log_chars",
    "content_log_lines",
    "content_json_envelope",
    "content_code_fence",
    "provenance_log_key_count",
    "provenance_log_source_trace_count",
    "context_log_key_count",
    "context_log_nested_count",
    "context_log_sequence_count",
    "context_log_scalar_count",
    "cascade_anchor",
    *SCORE_FIELDS,
)
FORBIDDEN_MODEL_NAME_PARTS = (
    "node_id",
    "event_id",
    "label",
    "task",
    "host",
    "source_identity",
    "source_name",
    "cache_key",
    "session",
    "access_count",
    "usefulness_score",
    "last_accessed",
    "updated_at",
)


class EvaluationError(SystemExit):
    """A pinned-input, privacy, split, or reproducibility invariant failed."""


def canonical_json(value: Any) -> str:
    """The only JSON encoding written or hashed by this evaluator."""

    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _assert_not_holdout(path: Path | str) -> None:
    raw = Path(path)
    raw_text = str(raw).replace("\\", "/").lower()
    # Reject the named path before resolving it: even symlink resolution would
    # be unnecessary filesystem contact with the prohibited subtree.
    if any(marker in raw_text for marker in HOLDOUT_MARKERS):
        raise EvaluationError(f"candidate-holdout access is prohibited: {path}")
    resolved_text = (
        str(raw.expanduser().resolve(strict=False)).replace("\\", "/").lower()
    )
    if any(marker in resolved_text for marker in HOLDOUT_MARKERS):
        raise EvaluationError(f"candidate-holdout access is prohibited: {path}")


@dataclass(frozen=True, slots=True)
class FrozenBindings:
    effect: ModuleType
    protocol: Any
    prereg: dict[str, Any]
    prereg_sha256: str
    effect_sha256: str
    baseline: dict[str, Any]
    baseline_sha256: str
    cache_key: Any


_BINDINGS_CACHE: dict[tuple[str, str, str], FrozenBindings] = {}


def load_frozen_bindings(
    *,
    prereg_path: Path = DEFAULT_PREREG,
    effect_path: Path = DEFAULT_EFFECT_TOOL,
    baseline_path: Path = DEFAULT_BASELINE,
) -> FrozenBindings:
    """Verify both sealed bytes *before* importing their implementation.

    The exact expected hashes are deliberately not configurable.  Tests that
    pass copied files exercise the same bytes; a modified copy is refused
    before any Python in it can execute.
    """

    for path in (prereg_path, effect_path, baseline_path):
        _assert_not_holdout(path)
        if not path.is_file():
            raise EvaluationError(f"required frozen input not found: {path}")
    prereg_sha = sha256_file(prereg_path)
    effect_sha = sha256_file(effect_path)
    if prereg_sha != EXPECTED_PREREG_SHA256:
        raise EvaluationError(
            f"sealed prereg hash mismatch: expected {EXPECTED_PREREG_SHA256}, got {prereg_sha}"
        )
    if effect_sha != EXPECTED_EFFECT_SHA256:
        raise EvaluationError(
            f"sealed effect-tool hash mismatch: expected {EXPECTED_EFFECT_SHA256}, got {effect_sha}"
        )

    baseline_sha = sha256_file(baseline_path)
    cache_key = (str(prereg_path.resolve()), str(effect_path.resolve()), baseline_sha)
    cached = _BINDINGS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # No project import appears above this point.  The sealed files have now
    # passed their byte checks, so executing the frozen tool is permitted.
    module_name = "_living_memory_frozen_recall_map_effect"
    spec = importlib.util.spec_from_file_location(module_name, effect_path)
    if spec is None or spec.loader is None:
        raise EvaluationError(f"cannot import sealed effect tool: {effect_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    prereg, protocol, recomputed = module.load_prereg(prereg_path)
    if recomputed != protocol.plan_sha256:
        raise EvaluationError("sealed prereg internal plan hash did not survive import")
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"invalid frozen baseline: {baseline_path}: {exc}") from exc
    if not isinstance(baseline, dict) or baseline.get("plan_sha256") != protocol.plan_sha256:
        raise EvaluationError("baseline does not belong to the verified sealed plan")
    baseline_rate = (
        ((baseline.get("arms") or {}).get("organic") or {})
        .get("primary_endpoint", {})
        .get("rate")
    )
    if not isinstance(baseline_rate, (int, float)) or isinstance(baseline_rate, bool):
        raise EvaluationError("baseline has no measured organic primary rate")
    if module.normalize_instant(str(baseline.get("as_of") or "")) != protocol.organic_as_of:
        raise EvaluationError("baseline as_of differs from the sealed organic as_of")
    derived_transfer_floor = round(float(baseline_rate) * protocol.relative_floor, 3)
    if derived_transfer_floor != TRANSFER_FLOOR:
        raise EvaluationError(
            "frozen baseline and sealed relative floor do not derive the registered transfer floor"
        )

    from living_memory.recall_map import cache_key as recall_map_cache_key

    bound = FrozenBindings(
        effect=module,
        protocol=protocol,
        prereg=prereg,
        prereg_sha256=prereg_sha,
        effect_sha256=effect_sha,
        baseline=baseline,
        baseline_sha256=baseline_sha,
        cache_key=recall_map_cache_key,
    )
    _BINDINGS_CACHE[cache_key] = bound
    return bound


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    path: Path

    @classmethod
    def parse(cls, value: str) -> "SourceSpec":
        if "=" not in value:
            raise EvaluationError("--source must be LOGICAL_NAME=SQLITE_PATH")
        name, raw_path = value.split("=", 1)
        if not SOURCE_NAME_RE.fullmatch(name):
            raise EvaluationError(f"invalid logical source name: {name!r}")
        path = Path(raw_path).expanduser()
        _assert_not_holdout(path)
        if not path.is_file():
            raise EvaluationError(f"source snapshot not found: {path}")
        return cls(name=name, path=path)


def _snapshot_parts(source: SourceSpec) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    candidates = (("database", source.path),) + tuple(
        (suffix[1:], Path(str(source.path) + suffix))
        for suffix in ("-wal", "-shm", "-journal")
    )
    for kind, path in candidates:
        if not path.exists():
            continue
        if not path.is_file():
            raise EvaluationError(f"snapshot component is not a file: {path}")
        parts.append({"kind": kind, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if not parts or parts[0]["kind"] != "database":
        raise EvaluationError(f"source snapshot database is missing: {source.path}")
    return parts


def snapshot_receipt(source: SourceSpec) -> dict[str, Any]:
    parts = _snapshot_parts(source)
    digest = _sha256_text(canonical_json(parts))
    return {
        "logical_name": source.name,
        "redacted_path": f"$SNAPSHOT/{source.name}.sqlite3",
        "parts": parts,
        "snapshot_sha256": digest,
    }


def _augment_sqlite_receipt(
    receipt: dict[str, Any], connection: sqlite3.Connection
) -> None:
    schema_rows = [
        [str(row[0]), str(row[1]), str(row[2]), str(row[3] or "")]
        for row in connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        )
    ]
    bounds = connection.execute(
        "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM recall_events "
        "WHERE created_at <= ?",
        (AS_OF,),
    ).fetchone()
    receipt["sqlite"] = {
        "schema_sha256": _sha256_text(canonical_json(schema_rows)),
        "schema_objects": len(schema_rows),
        "user_version": int(connection.execute("PRAGMA user_version").fetchone()[0]),
        "pre_candidate_event_rows": int(bounds[0]),
        "pre_candidate_event_min": bounds[1],
        "pre_candidate_event_max": bounds[2],
    }


def _json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(str(raw or "{}"))
    except ValueError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _json_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return list(raw)
    try:
        parsed = json.loads(str(raw or "[]"))
    except ValueError:
        return []
    return list(parsed) if isinstance(parsed, list) else []


def _is_file_chunk_envelope(envelope: Any) -> bool:
    if not isinstance(envelope, dict):
        return False
    chunk = envelope.get("chunk")
    lines = envelope.get("lines")
    chunk_match = FILE_CHUNK_INDEX_RE.fullmatch(chunk) if isinstance(chunk, str) else None
    lines_match = FILE_CHUNK_LINES_RE.fullmatch(lines) if isinstance(lines, str) else None
    if chunk_match is None or lines_match is None:
        return False
    part, total = (int(value) for value in chunk.split("/"))
    first_line, last_line = (int(value) for value in lines.split("-"))
    return (
        part <= total
        and first_line <= last_line
        and isinstance(envelope.get("path"), str)
        and bool(str(envelope["path"]).strip())
        and isinstance(envelope.get("kind"), str)
        and bool(str(envelope["kind"]).strip())
        and isinstance(envelope.get("language"), str)
        and isinstance(envelope.get("sha256"), str)
        and SHA256_RE.fullmatch(str(envelope["sha256"])) is not None
    )


def classify_content_form(
    content: str,
    context: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> str:
    """Return an emitter-independent structural content class.

    The rules are intentionally narrow.  Free prose merely discussing one of
    these emissions is a near miss, not ballast.
    """

    context = dict(context or {})
    provenance = dict(provenance or {})
    if content.startswith("[file-chunk]"):
        suffix = content[len("[file-chunk]") :]
        separator = suffix[:1]
        header = suffix.lstrip(" \t").splitlines()[0].strip() if separator in {" ", "\t"} else ""
        try:
            envelope = json.loads(header)
        except ValueError:
            envelope = None
        if _is_file_chunk_envelope(envelope):
            return "file_chunk_envelope"

    if STRATEGY_STAGNATION_RE.fullmatch(content):
        return "strategy_stagnation"

    kind_values = {
        str(mapping.get(key) or "").strip().lower().replace("-", "_")
        for mapping in (context, provenance)
        for key in ("kind", "type", "lesson_kind", "record_kind")
    }
    structured_marker = next((line for line in content.splitlines()[:1]), "")
    marker_kind = structured_marker.strip().lower().replace("-", "_")
    marker_match = marker_kind in {"[supervision_journal]", "[monitoring_journal]"}
    if kind_values & SUPERVISION_KINDS or marker_match:
        return "supervision_journal"
    return "eligible"


USER_AUTHORED_NEAR_MISSES: tuple[tuple[str, Mapping[str, Any], Mapping[str, Any]], ...] = (
    (
        'I pasted the text [file-chunk] {"path":"notes.py","chunk":"1/1"} in my report.',
        {},
        {},
    ),
    ('[file-chunk] {"chunk":"1/1","note":"my hand-written outline"}', {}, {}),
    (
        '[file-chunk] {"path":"notes.py","kind":"source","language":"python",'
        '"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"chunk":"9/1","lines":"20-10"}',
        {},
        {},
    ),
    ("[file-chunk] this is prose, not an indexing envelope", {}, {}),
    ("A user noted: Strategy stagnation detected on retry-loop", {}, {}),
    ("Strategy stagnation was detected on my own drafting process", {}, {}),
    (
        "Strategy stagnation detected on retry-loop\nThis is my diagnosis, not a watchdog row.",
        {},
        {},
    ),
    ("I keep a supervision_journal for my own project notes.", {}, {}),
    ("[supervision_journal] is the literal token I am documenting.", {}, {}),
    ("My monitoring journal is user-authored prose.", {"topic": "monitoring_journal"}, {}),
)


def generated_ballast_controls(
    *,
    seed: int = BALLAST_CONTROL_SEED,
    variants_per_class: int = BALLAST_CONTROL_VARIANTS_PER_CLASS,
) -> list[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    """Generate fixed-seed positive variants plus hand-authored near misses.

    Only aggregate confusion counts leave this function.  The generator is
    deliberately separate from :func:`classify_content_form`, and the tests
    use a second generator so classifier and controls cannot drift together
    unnoticed.
    """

    if variants_per_class < 1:
        raise EvaluationError("ballast controls require at least one variant per class")
    rng = random.Random(seed)
    controls: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    languages = ("python", "rust", "typescript", "markdown", "text")
    kinds = ("source", "test", "documentation", "configuration")
    detail_keys = ("attempts", "strategy", "window", "reason")
    journal_kinds = tuple(sorted(SUPERVISION_KINDS))
    metadata_keys = ("kind", "type", "lesson_kind", "record_kind")
    for index in range(variants_per_class):
        part = rng.randint(1, 12)
        total = rng.randint(part, part + 12)
        start = rng.randint(1, 900)
        end = start + rng.randint(0, 120)
        header = {
            "path": (
                f"src/generated_{rng.randrange(1_000_000):06d}."
                f"{rng.choice(('py', 'rs', 'ts', 'md'))}"
            ),
            "kind": rng.choice(kinds),
            "language": rng.choice(languages),
            "sha256": f"{rng.getrandbits(256):064x}",
            "chunk": f"{part}/{total}",
            "lines": f"{start}-{end}",
        }
        if rng.randrange(2):
            header["heading"] = f"section-{rng.randrange(10_000)}"
        separator = rng.choice((" ", "  ", "\t"))
        controls.append(
            (
                "file_chunk_envelope",
                "[file-chunk]"
                + separator
                + json.dumps(header, sort_keys=bool(index % 2))
                + "\n```\nopaque\n```",
                {},
                {},
            )
        )

        target = f"strategy-{rng.randrange(1_000_000):06d}"
        details = rng.sample(detail_keys, rng.randrange(len(detail_keys) + 1))
        stagnation = "Strategy stagnation detected on " + target
        if details:
            stagnation += "".join(
                f"\n{name}:{rng.choice((' retry budget exhausted', ' unchanged', ' 3', ' stalled'))}"
                for name in details
            )
        controls.append(("strategy_stagnation", stagnation, {}, {}))

        journal_kind = rng.choice(journal_kinds)
        if rng.randrange(2):
            mapping = {
                rng.choice(metadata_keys): journal_kind.replace(
                    "_", rng.choice(("_", "-"))
                )
            }
            context, provenance = (mapping, {}) if rng.randrange(2) else ({}, mapping)
            content = f"structured journal payload {index}"
        else:
            marker = rng.choice(("supervision_journal", "monitoring-journal"))
            context, provenance = {}, {}
            content = f"[{marker}]\nrun={index}"
        controls.append(("supervision_journal", content, context, provenance))

    controls.extend(
        ("eligible", content, dict(context), dict(provenance))
        for content, context, provenance in USER_AUTHORED_NEAR_MISSES
    )
    return controls


def ballast_control_audit() -> dict[str, Any]:
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for expected, content, context, provenance in generated_ballast_controls():
        confusion[expected][classify_content_form(content, context, provenance)] += 1
    ballast_classes = {
        "file_chunk_envelope", "strategy_stagnation", "supervision_journal"
    }
    expected_ballast = sum(
        sum(confusion[expected].values()) for expected in ballast_classes
    )
    expected_eligible = sum(confusion["eligible"].values())
    false_negatives = sum(
        count
        for expected in ballast_classes
        for predicted, count in confusion[expected].items()
        if predicted != expected
    )
    false_positives = sum(
        count for predicted, count in confusion["eligible"].items() if predicted != "eligible"
    )
    correct_ballast = sum(confusion[expected][expected] for expected in ballast_classes)
    return {
        "seed": BALLAST_CONTROL_SEED,
        "variants_per_ballast_class": BALLAST_CONTROL_VARIANTS_PER_CLASS,
        "cases": expected_ballast + expected_eligible,
        "expected_ballast": expected_ballast,
        "expected_eligible": expected_eligible,
        "true_positives": correct_ballast,
        "true_negatives": confusion["eligible"]["eligible"],
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "confusion": {
            expected: dict(sorted(predicted.items()))
            for expected, predicted in sorted(confusion.items())
        },
        "case_text_published": False,
    }


def _shape(value: Any, *, depth: int = 0) -> tuple[int, int, int, int]:
    """Return (keys, nested containers, sequences, scalar leaves)."""

    if isinstance(value, Mapping):
        keys = len(value)
        nested = int(depth > 0)
        seqs = scalars = 0
        for child in value.values():
            ck, cn, cs, cv = _shape(child, depth=depth + 1)
            keys += ck
            nested += cn
            seqs += cs
            scalars += cv
        return keys, nested, seqs, scalars
    if isinstance(value, (list, tuple)):
        keys = nested = 0
        seqs = 1
        scalars = 0
        for child in value:
            ck, cn, cs, cv = _shape(child, depth=depth + 1)
            keys += ck
            nested += cn
            seqs += cs
            scalars += cv
        return keys, nested, seqs, scalars
    return 0, 0, 0, 1


def _parse_instant(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _shift_hours(value: str, hours: int) -> str:
    return (_parse_instant(value) + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log_count(value: int | float) -> float:
    return round(math.log1p(max(0.0, float(value))), 8)


@dataclass(slots=True)
class EventMeta:
    source: str
    event: Any
    raw_results: tuple[dict[str, Any], ...]
    session_id: str | None
    task_pattern: str | None
    qualified_cache_key: str
    qualified_sessions: tuple[str, ...]

    def result_for(self, node_id: str) -> dict[str, Any] | None:
        return next((entry for entry in self.raw_results if str(entry.get("node_id") or "") == node_id), None)


@dataclass(slots=True)
class ItemRecord:
    source: str
    arm: str
    event_id: str
    created_at: str
    transport_session_id: str | None = None
    component_id: str = ""
    split: str = ""
    node_id: str = ""
    consumed: bool = False
    primary_stratum: bool = True
    features: dict[str, float | None] = field(default_factory=dict)
    family_reasons: dict[str, str] = field(default_factory=dict)
    form_class: str = "unavailable"


@dataclass(slots=True)
class SourceState:
    spec: SourceSpec
    connection: sqlite3.Connection
    receipt_before: dict[str, Any]
    events: list[Any]
    metas: dict[str, EventMeta]
    consumer_index: Any
    gate_stratum: Any
    node_rows: dict[str, sqlite3.Row]
    observed_map_event_ids: set[str]


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            keep, move = min(a, b), max(a, b)
            self.parent[move] = keep


def _event_digest(source: str, event_id: str) -> str:
    return _sha256_text(f"recall-map-relevance-event-v1\0{source}\0{event_id}")


def assign_component_splits(
    records: Sequence[ItemRecord],
    metas: Mapping[tuple[str, str], EventMeta],
    *,
    seed: str = SPLIT_SEED,
    component_event_keys: Iterable[tuple[str, str]] | None = None,
    forced_eval_event_keys: Iterable[tuple[str, str]] = (),
) -> dict[str, dict[str, Any]]:
    """Union event components and mutate records with deterministic splits.

    Component membership is event-level, not item-level.  A journal-only map
    delivery still forces its entire component to eval, and an event carrying
    no scorable item can still be the cache/session bridge between two events.
    """

    record_event_keys = {(record.source, record.event_id) for record in records}
    event_keys = sorted(set(component_event_keys or record_event_keys) | record_event_keys)
    missing_meta = [key for key in event_keys if key not in metas]
    if missing_meta:
        raise EvaluationError("component event has no loaded metadata")
    opaque_by_event = {key: _event_digest(*key) for key in event_keys}
    union = _UnionFind(opaque_by_event.values())
    owners: dict[str, str] = {}
    for key in event_keys:
        meta = metas[key]
        opaque = opaque_by_event[key]
        identities = (meta.qualified_cache_key, *meta.qualified_sessions)
        for identity in identities:
            token = _sha256_text(f"component-link-v1\0{identity}")
            previous = owners.setdefault(token, opaque)
            union.union(opaque, previous)

    event_groups: dict[str, list[str]] = defaultdict(list)
    for opaque in opaque_by_event.values():
        event_groups[union.find(opaque)].append(opaque)
    component_for_root = {
        root: _sha256_text("component-v1\0" + "\0".join(sorted(event_digests)))
        for root, event_digests in event_groups.items()
    }
    forced_events = set(forced_eval_event_keys)
    forced_events.update(
        (record.source, record.event_id) for record in records if record.arm == "observed_map"
    )
    unknown_forced = forced_events - set(event_keys)
    if unknown_forced:
        raise EvaluationError("forced-eval event is absent from the component universe")
    component_has_map: Counter[str] = Counter()
    for key in forced_events:
        root = union.find(opaque_by_event[key])
        component_has_map[component_for_root[root]] += 1
    for record in records:
        root = union.find(opaque_by_event[(record.source, record.event_id)])
        component_id = component_for_root[root]
        record.component_id = component_id

    components: dict[str, dict[str, Any]] = {}
    for component_id in sorted(set(component_for_root.values())):
        if component_has_map[component_id]:
            split = "eval"
            reason = "touches_observed_map_delivery"
        else:
            draw = int(_sha256_text(f"{seed}\0{component_id}")[:16], 16) / float(16**16)
            split = "train" if draw < ORGANIC_TRAIN_FRACTION else "eval"
            reason = "seeded_organic_component"
        components[component_id] = {"split": split, "reason": reason}
    for record in records:
        record.split = components[record.component_id]["split"]
        if record.arm == "observed_map" and record.split != "eval":  # pragma: no cover
            raise EvaluationError("observed map item escaped forced eval")
    return components


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _load_event_meta(
    source: SourceSpec, connection: sqlite3.Connection, events: Sequence[Any], cache_key_fn: Any
) -> dict[str, EventMeta]:
    columns = _table_columns(connection, "recall_events")
    wanted = [name for name in ("id", "results", "session_id", "ambient_context") if name in columns]
    rows = connection.execute(
        f"SELECT {', '.join(wanted)} FROM recall_events "
        "WHERE created_at <= ? ORDER BY created_at, id",
        (AS_OF,),
    ).fetchall()
    raw_by_id = {str(row["id"]): row for row in rows}
    metas: dict[str, EventMeta] = {}
    for event in events:
        raw = raw_by_id.get(event.id)
        if raw is None:
            continue
        ambient = _json_object(raw["ambient_context"] if "ambient_context" in columns else None)
        task_pattern = str(ambient.get("task_pattern") or "").strip() or None
        local_cache = cache_key_fn(
            event.scope or "global", task=event.task, task_pattern=task_pattern
        )
        sessions: list[str] = []
        if event.transport_session_id:
            sessions.append(f"{source.name}\0transport\0{event.transport_session_id}")
        session_id = None
        if "session_id" in columns:
            session_id = str(raw["session_id"] or "").strip() or None
        if session_id:
            sessions.append(f"{source.name}\0session\0{session_id}")
        metas[event.id] = EventMeta(
            source=source.name,
            event=event,
            raw_results=tuple(
                dict(entry) for entry in _json_list(raw["results"]) if isinstance(entry, dict)
            ),
            session_id=session_id,
            task_pattern=task_pattern,
            qualified_cache_key=f"{source.name}\0cache\0{local_cache}",
            qualified_sessions=tuple(sessions),
        )
    return metas


def _load_nodes(connection: sqlite3.Connection, node_ids: set[str]) -> dict[str, sqlite3.Row]:
    if not node_ids:
        return {}
    result: dict[str, sqlite3.Row] = {}
    ordered = sorted(node_ids)
    for offset in range(0, len(ordered), 500):
        batch = ordered[offset : offset + 500]
        marks = ",".join("?" for _ in batch)
        rows = connection.execute(
            "SELECT id, level, content, context, provenance, source_traces, "
            "created_at, timestamp FROM nodes WHERE id IN (" + marks + ")",
            batch,
        ).fetchall()
        result.update({str(row["id"]): row for row in rows})
    return result


def _score_one(bindings: FrozenBindings, state: SourceState, event: Any, item: Any) -> tuple[bool, bool]:
    scored = bindings.effect.score_pairs(
        [(event, item)],
        state.consumer_index,
        min_token_overlap=bindings.protocol.min_token_overlap,
        stratum=state.gate_stratum,
    )
    primary = scored["primary_endpoint"]
    anchor = scored["secondary_anchor_stratum"]
    is_primary = bool(primary["items"])
    consumed = bool(primary["consumed"] + anchor["consumed"])
    return consumed, is_primary


def _candidate_records(bindings: FrozenBindings, state: SourceState) -> list[tuple[ItemRecord, Any]]:
    protocol = bindings.protocol
    organic_start, organic_end = protocol.organic_window
    map_end = _shift_hours(AS_OF, -protocol.horizon_hours)
    pairs: list[tuple[ItemRecord, Any]] = []
    for event in state.events:
        meta = state.metas[event.id]
        if not state.consumer_index.qualifying(event):
            continue
        if organic_start <= event.created_at < organic_end:
            for item in bindings.effect.organic_items(
                event, head_cut=protocol.head_cut, cap=protocol.organic_cap
            ):
                pairs.append(
                    (
                        ItemRecord(
                            source=state.spec.name,
                            arm="organic",
                            event_id=event.id,
                            created_at=event.created_at,
                            transport_session_id=event.transport_session_id,
                            node_id=item.node_id,
                        ),
                        item,
                    )
                )
        if event.id in state.observed_map_event_ids and event.created_at < map_end:
            for item in bindings.effect.map_items(event, cap=protocol.map_cap):
                pairs.append(
                    (
                        ItemRecord(
                            source=state.spec.name,
                            arm="observed_map",
                            event_id=event.id,
                            created_at=event.created_at,
                            transport_session_id=event.transport_session_id,
                            node_id=item.node_id,
                        ),
                        item,
                    )
                )
    return pairs


def _history_index(
    bindings: FrozenBindings,
    state: SourceState,
    target_ids: set[str],
) -> dict[str, list[tuple[str, str, bool]]]:
    """Matured-delivery history; no outcome can cross the current delivery."""

    histories: dict[str, list[tuple[str, str, bool]]] = defaultdict(list)
    delivered_cache: dict[str, frozenset[str]] = {}
    for event in state.events:
        if event.created_at >= AS_OF:
            continue
        candidates = list(
            bindings.effect.organic_items(
                event,
                head_cut=bindings.protocol.head_cut,
                cap=bindings.protocol.organic_cap,
            )
        )
        if event.recall_map:
            candidates.extend(bindings.effect.map_items(event, cap=bindings.protocol.map_cap))
        seen: set[str] = set()
        for item in candidates:
            if item.node_id in seen or item.node_id not in target_ids:
                continue
            seen.add(item.node_id)
            delivered = delivered_cache.get(event.id)
            if delivered is None:
                ids: set[str] = set()
                for consumer in state.consumer_index.qualifying(event):
                    ids.update(consumer.result_ids)
                delivered = frozenset(ids)
                delivered_cache[event.id] = delivered
            histories[item.node_id].append(
                (
                    _shift_hours(event.created_at, bindings.protocol.horizon_hours),
                    event.created_at,
                    item.node_id in delivered,
                )
            )
    for entries in histories.values():
        entries.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    return histories


def matured_past_features(
    histories: Mapping[str, Sequence[tuple[str, str, bool]]], node_id: str, at: str
) -> dict[str, float]:
    eligible = [
        entry
        for entry in histories.get(node_id, ())
        if entry[1] < at and entry[0] <= at
    ]
    consumed = sum(int(entry[2]) for entry in eligible)
    nonconsumed = len(eligible) - consumed
    streak = 0
    for entry in reversed(eligible):
        if entry[2]:
            break
        streak += 1
    return {
        "prior_matured_log_count": _log_count(len(eligible)),
        "prior_nonconsumed_log_count": _log_count(nonconsumed),
        "prior_nonconsumption_streak_log": _log_count(streak),
        "prior_consumption_rate": round(consumed / len(eligible), 8) if eligible else 0.0,
    }


def _extract_features(
    bindings: FrozenBindings,
    state: SourceState,
    record: ItemRecord,
    histories: Mapping[str, Sequence[tuple[str, str, bool]]],
) -> None:
    node = state.node_rows.get(record.node_id)
    features: dict[str, float | None] = {
        name: None for name in (*PRIMARY_MODEL_FEATURES, *ANALYSIS_ONLY_FEATURES)
    }
    reasons: dict[str, str] = {}
    meta = state.metas[record.event_id]
    result = meta.result_for(record.node_id) if record.arm == "organic" else None
    if node is None:
        for family in (
            "snapshot_content_form",
            "historical_context_provenance",
            "node_age",
            "level",
            "snapshot_context_shape",
        ):
            reasons[family] = "node_snapshot_row_missing"
    else:
        content = str(node["content"] or "")
        context = _json_object(node["context"])
        provenance = _json_object(node["provenance"])
        source_traces = _json_list(node["source_traces"])
        form = classify_content_form(content, context, provenance)
        record.form_class = form
        features.update(
            {
                "form_file_chunk_envelope": float(form == "file_chunk_envelope"),
                "form_strategy_stagnation": float(form == "strategy_stagnation"),
                "form_supervision_journal": float(form == "supervision_journal"),
                "form_machine_ballast": float(form != "eligible"),
                "content_log_chars": _log_count(len(content)),
                "content_log_lines": _log_count(max(1, len(content.splitlines()))),
                "content_json_envelope": float(content.lstrip().startswith(("{", "["))),
                "content_code_fence": float("```" in content),
                "provenance_log_key_count": _log_count(len(provenance)),
                "provenance_log_source_trace_count": _log_count(len(source_traces)),
            }
        )
        reasons["snapshot_content_form"] = "snapshot_only_not_recorded_at_delivery"
        reasons["historical_context_provenance"] = (
            "not_versioned_snapshot_diagnostics_excluded_from_fitting"
        )

        try:
            age_seconds = (
                _parse_instant(record.created_at) - _parse_instant(str(node["created_at"]))
            ).total_seconds()
            if age_seconds < 0:
                reasons["node_age"] = "node_created_after_delivery"
            else:
                age_days = age_seconds / 86_400.0
                features["node_age_log_days"] = round(math.log1p(age_days), 8)
                reasons["node_age"] = "available_from_immutable_created_at"
        except (TypeError, ValueError):
            reasons["node_age"] = "node_created_at_unparseable"

        recorded_level = result.get("level") if result else None
        level = str(recorded_level or node["level"] or "")
        if level in {"trace", "concept", "schema"}:
            for choice in ("trace", "concept", "schema"):
                features[f"level_{choice}"] = float(level == choice)
            reasons["level"] = (
                "available_from_recorded_organic_result"
                if recorded_level
                else "available_from_immutable_node_level"
            )
        else:
            reasons["level"] = "recorded_level_invalid"

        keys, nested, sequences, scalars = _shape(context)
        features.update(
            {
                "context_log_key_count": _log_count(keys),
                "context_log_nested_count": _log_count(nested),
                "context_log_sequence_count": _log_count(sequences),
                "context_log_scalar_count": _log_count(scalars),
            }
        )
        reasons["snapshot_context_shape"] = "snapshot_only_not_recorded_at_delivery"

    # The delivery envelopes carry no cascade stage.  Anchor rows are mutable
    # and have no temporal version of their decay state, so even an as-of SQL
    # cut would reconstruct history from later state.  The frozen effect tool's
    # AnchorStratum remains the scoring stratum; it is never promoted into a
    # decision-time feature here.
    reasons["cascade_stage"] = "not_recorded_at_delivery_not_reconstructed"

    score_available = result is not None
    for name in SCORE_FIELDS:
        raw = result.get(name) if result else None
        features[name] = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
        score_available = score_available and features[name] is not None
    reasons["recorded_delivery_scores"] = (
        "available_from_recorded_organic_result"
        if score_available
        else (
            "not_recorded_in_recall_map_payload"
            if record.arm == "observed_map"
            else "recorded_result_score_missing"
        )
    )

    features.update(matured_past_features(histories, record.node_id, record.created_at))
    reasons["past_nonconsumption"] = "available_strictly_matured_before_delivery"
    record.features = features
    record.family_reasons = reasons


def validate_model_feature_names(names: Sequence[str]) -> None:
    if tuple(names) != PRIMARY_MODEL_FEATURES:
        raise EvaluationError("primary model feature surface differs from the sealed allowlist")
    for name in names:
        lowered = name.lower()
        forbidden = [part for part in FORBIDDEN_MODEL_NAME_PARTS if part in lowered]
        if forbidden:
            raise EvaluationError(f"forbidden model feature {name!r}: {forbidden[0]}")


@dataclass(frozen=True, slots=True)
class FittedModel:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    threshold: float

    def probability(self, features: Mapping[str, float | None]) -> float:
        total = self.intercept
        for index, name in enumerate(self.feature_names):
            raw = features.get(name)
            value = self.means[index] if raw is None else float(raw)
            total += self.coefficients[index] * ((value - self.means[index]) / self.scales[index])
        return _sigmoid(total)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 700.0)))
    exp_value = math.exp(max(value, -700.0))
    return exp_value / (1.0 + exp_value)


def _fit_threshold(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    if not probabilities:
        raise EvaluationError("cannot fit a threshold without organic train items")
    ordered = sorted(zip(probabilities, labels, strict=True), key=lambda pair: (-pair[0], -pair[1]))
    positives = sum(labels)
    min_selected = max(1, math.ceil(MIN_THRESHOLD_TRAIN_FRACTION * len(labels)))
    tp = fp = 0
    best: tuple[float, float, float] | None = None
    index = 0
    while index < len(ordered):
        threshold = ordered[index][0]
        while index < len(ordered) and ordered[index][0] == threshold:
            if ordered[index][1]:
                tp += 1
            else:
                fp += 1
            index += 1
        selected = tp + fp
        if selected < min_selected:
            continue
        fn = positives - tp
        f1 = 2.0 * tp / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        candidate = (f1, selected / len(labels), threshold)
        if best is None or candidate > best:
            best = candidate
    return float(best[2] if best else min(probabilities))


def fit_primary_model(records: Sequence[ItemRecord]) -> FittedModel:
    """Fit exactly once, on primary-stratum organic train records."""

    validate_model_feature_names(PRIMARY_MODEL_FEATURES)
    train = [
        record
        for record in records
        if record.arm == "organic" and record.split == "train" and record.primary_stratum
    ]
    if not train:
        raise EvaluationError("organic train cohort is empty")
    labels = [int(record.consumed) for record in train]
    means: list[float] = []
    scales: list[float] = []
    for name in PRIMARY_MODEL_FEATURES:
        values = [float(record.features[name]) for record in train if record.features.get(name) is not None]
        mean = sum(values) / len(values) if values else 0.0
        variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
        means.append(mean)
        scales.append(math.sqrt(variance) if variance > 1e-12 else 1.0)
    matrix = [
        [
            ((means[j] if record.features.get(name) is None else float(record.features[name])) - means[j])
            / scales[j]
            for j, name in enumerate(PRIMARY_MODEL_FEATURES)
        ]
        for record in train
    ]
    prevalence = min(1.0 - 1e-6, max(1e-6, sum(labels) / len(labels)))
    intercept = math.log(prevalence / (1.0 - prevalence))
    coefficients = [0.0] * len(PRIMARY_MODEL_FEATURES)
    for _ in range(MODEL_ITERATIONS):
        grad_intercept = 0.0
        gradients = [0.0] * len(coefficients)
        for row, label in zip(matrix, labels, strict=True):
            prediction = _sigmoid(intercept + sum(w * x for w, x in zip(coefficients, row, strict=True)))
            error = prediction - label
            grad_intercept += error
            for index, value in enumerate(row):
                gradients[index] += error * value
        count = float(len(matrix))
        intercept -= MODEL_LEARNING_RATE * grad_intercept / count
        for index in range(len(coefficients)):
            gradient = gradients[index] / count + MODEL_L2 * coefficients[index]
            coefficients[index] -= MODEL_LEARNING_RATE * gradient
    probabilities = [
        _sigmoid(intercept + sum(w * x for w, x in zip(coefficients, row, strict=True)))
        for row in matrix
    ]
    threshold = _fit_threshold(probabilities, labels)
    return FittedModel(
        feature_names=PRIMARY_MODEL_FEATURES,
        means=tuple(means),
        scales=tuple(scales),
        coefficients=tuple(coefficients),
        intercept=intercept,
        threshold=threshold,
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _cohort_summary(records: Sequence[ItemRecord], *, include_sources: bool = True) -> dict[str, Any]:
    events = {(record.source, record.event_id) for record in records}
    transports = {
        (record.source, record.transport_session_id)
        for record in records
        if record.transport_session_id
    }
    consumed = sum(int(record.consumed) for record in records)
    payload: dict[str, Any] = {
        "events": len(events),
        "items": len(records),
        "consumed": consumed,
        "rate": _ratio(consumed, len(records)),
        "transport_sessions": len(transports),
        "components": sorted({record.component_id for record in records}),
    }
    if include_sources:
        payload["by_source"] = {
            source: {
                "events": len({record.event_id for record in records if record.source == source}),
                "items": sum(record.source == source for record in records),
                "consumed": sum(
                    int(record.consumed) for record in records if record.source == source
                ),
            }
            for source in sorted({record.source for record in records})
        }
    return payload


def _selected_summary(records: Sequence[ItemRecord], model: FittedModel) -> dict[str, Any]:
    selected = [record for record in records if model.probability(record.features) >= model.threshold]
    summary = _cohort_summary(selected, include_sources=False)
    summary["total_items"] = len(records)
    summary["selection_rate"] = _ratio(len(selected), len(records))
    return summary


def _association(records: Sequence[ItemRecord], feature: str) -> dict[str, Any]:
    available = [record for record in records if record.features.get(feature) is not None]
    used = [float(record.features[feature]) for record in available if record.consumed]
    unused = [float(record.features[feature]) for record in available if not record.consumed]
    mean_used = sum(used) / len(used) if used else None
    mean_unused = sum(unused) / len(unused) if unused else None
    difference = None if mean_used is None or mean_unused is None else mean_used - mean_unused
    direction = (
        "unavailable"
        if difference is None
        else "positive"
        if difference > 1e-12
        else "negative"
        if difference < -1e-12
        else "zero"
    )
    return {
        "available": len(available),
        "missing": len(records) - len(available),
        "consumed": len(used),
        "not_consumed": len(unused),
        "mean_consumed": None if mean_used is None else round(mean_used, 8),
        "mean_not_consumed": None if mean_unused is None else round(mean_unused, 8),
        "difference": None if difference is None else round(difference, 8),
        "direction": direction,
    }


def _availability(records: Sequence[ItemRecord]) -> dict[str, Any]:
    return {
        family: {
            "items": len(records),
            "reasons": dict(
                sorted(Counter(record.family_reasons.get(family, "not_attempted") for record in records).items())
            ),
        }
        for family in FEATURE_FAMILIES
    }


def _form_summary(records: Sequence[ItemRecord]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for form in sorted({record.form_class for record in records}):
        selected = [record for record in records if record.form_class == form]
        consumed = sum(int(record.consumed) for record in selected)
        result[form] = {
            "items": len(selected),
            "consumed": consumed,
            "rate": _ratio(consumed, len(selected)),
        }
    return result


def _real_form_audit(records: Sequence[ItemRecord]) -> dict[str, Any]:
    classified = Counter(record.form_class for record in records)
    ballast = sum(
        count
        for form, count in classified.items()
        if form not in {"eligible", "unavailable"}
    )
    return {
        "items": len(records),
        "classified_ballast": ballast,
        "classified_eligible": classified["eligible"],
        "classification_unavailable": classified["unavailable"],
        "false_positives": None,
        "false_negatives": None,
        "error_rate_reason": "real_corpus_has_no_independent_item_level_form_ground_truth",
    }


def _feature_analysis(
    records: Sequence[ItemRecord],
    model: FittedModel,
    protocol: Any,
    *,
    baseline_rate: float,
) -> dict[str, Any]:
    cohorts = {
        "organic_train": [
            r for r in records if r.arm == "organic" and r.split == "train" and r.primary_stratum
        ],
        "organic_eval": [
            r for r in records if r.arm == "organic" and r.split == "eval" and r.primary_stratum
        ],
        "observed_map_eval": [
            r for r in records if r.arm == "observed_map" and r.split == "eval" and r.primary_stratum
        ],
        "secondary_anchor": [r for r in records if not r.primary_stratum],
    }
    associations = {
        feature: {cohort: _association(items, feature) for cohort, items in cohorts.items()}
        for feature in (*PRIMARY_MODEL_FEATURES, *ANALYSIS_ONLY_FEATURES)
    }
    transferability: dict[str, Any] = {}
    for feature in PRIMARY_MODEL_FEATURES:
        directions = {
            cohort: associations[feature][cohort]["direction"]
            for cohort in ("organic_train", "organic_eval", "observed_map_eval")
        }
        informative = [value for value in directions.values() if value not in {"zero", "unavailable"}]
        transferability[feature] = {
            "directions": directions,
            "same_sign": len(informative) == 3 and len(set(informative)) == 1,
        }

    evaluations = {
        cohort: _selected_summary(items, model)
        for cohort, items in cohorts.items()
        if cohort != "secondary_anchor"
    }
    observed = evaluations["observed_map_eval"]
    minimums = {
        "events": protocol.min_map_events,
        "items": protocol.min_map_items,
        "transport_sessions": protocol.min_map_transports,
    }
    pass_floor = observed["rate"] is not None and observed["rate"] >= TRANSFER_FLOOR
    pass_minimums = all(observed[name] >= minimum for name, minimum in minimums.items())
    return {
        "schema_version": 1,
        "artifact": "recall-map-relevance-feature-analysis",
        "as_of": AS_OF,
        "feature_policy": {
            "primary_model_features": list(PRIMARY_MODEL_FEATURES),
            "analysis_only_features": list(ANALYSIS_ONLY_FEATURES),
            "decision_time_feature_source": (
                "immutable node creation fields, recorded delivery envelope, "
                "and strictly matured history"
            ),
            "snapshot_only_features": [
                "content form and size",
                "provenance shape",
                "context shape",
            ],
            "snapshot_only_role": "aggregate sensitivity diagnostics; excluded from fitting",
            "unavailable_not_reconstructed": [
                "historical context and provenance values",
                "historical cascade stage",
                "observed-map residual scores",
            ],
            "forbidden_as_features": [
                "node ids",
                "labels",
                "task names",
                "hosts",
                "source identity",
                "cache keys",
                "sessions",
                "current access_count",
                "current usefulness_score",
                "current last_accessed",
                "updated_at-derived usage",
            ],
            "missing_value_policy": "organic-train mean; no missingness indicator",
        },
        "availability": {cohort: _availability(items) for cohort, items in cohorts.items()},
        "associations": associations,
        "transferability": transferability,
        "content_forms": {cohort: _form_summary(items) for cohort, items in cohorts.items()},
        "form_classification_audit": {
            "classifier": "emitter_invariant_form_schema_provenance_v1",
            "real_corpus_role": "snapshot-only aggregate diagnostic; never a fitting input",
            "synthetic": ballast_control_audit(),
            "real": {cohort: _real_form_audit(items) for cohort, items in cohorts.items()},
            "aggregate_only": True,
        },
        "primary_model": {
            "kind": "deterministic_l2_logistic_regression",
            "fit_cohort": "organic_train_only",
            "fit_items": len(cohorts["organic_train"]),
            "fit_consumed": sum(int(item.consumed) for item in cohorts["organic_train"]),
            "feature_schema_sha256": _sha256_text(canonical_json(list(model.feature_names))),
            "iterations": MODEL_ITERATIONS,
            "learning_rate": MODEL_LEARNING_RATE,
            "l2": MODEL_L2,
            "intercept": round(model.intercept, 10),
            "coefficients": {
                name: round(value, 10)
                for name, value in zip(model.feature_names, model.coefficients, strict=True)
            },
            "standardization": {
                name: {"mean": round(model.means[i], 10), "scale": round(model.scales[i], 10)}
                for i, name in enumerate(model.feature_names)
            },
            "threshold": round(model.threshold, 10),
            "threshold_fit": "maximum train F1 with at least 20 percent train selection",
            "evaluations": evaluations,
        },
        "transfer_verdict": {
            "threshold": TRANSFER_FLOOR,
            "threshold_derivation": {
                "frozen_organic_rate": baseline_rate,
                "sealed_relative_floor": protocol.relative_floor,
                "rounded_decision_floor": TRANSFER_FLOOR,
            },
            "cohort_minimums": minimums,
            "floor_pass": pass_floor,
            "minimums_pass": pass_minimums,
            "verdict": "PASS" if pass_floor and pass_minimums else "FAIL",
            "refit_on_eval": False,
        },
        "leakage_audit": {
            "fit_population": "organic train primary-stratum items only",
            "map_rows_in_fit": 0,
            "observed_map_role": "transfer eval only",
            "past_history_cut": "prior delivery outcome_end <= current delivery created_at",
            "future_mutated_node_stats_read": False,
            "mutable_node_columns_read": [],
            "historical_context_provenance": (
                "not versioned; snapshot-only diagnostics excluded from fitting"
            ),
            "historical_cascade_stage": "not recorded; unavailable and not reconstructed",
            "observed_map_residual_scores": "not recorded; unavailable and not reconstructed",
            "candidate_holdout_accessed": False,
        },
    }


def _manifest(
    records: Sequence[ItemRecord],
    components: Mapping[str, Mapping[str, Any]],
    receipts: Sequence[dict[str, Any]],
    bindings: FrozenBindings,
) -> dict[str, Any]:
    primary = [record for record in records if record.primary_stratum]
    organic_train = [r for r in primary if r.arm == "organic" and r.split == "train"]
    organic_eval = [r for r in primary if r.arm == "organic" and r.split == "eval"]
    map_eval = [r for r in primary if r.arm == "observed_map" and r.split == "eval"]
    anchor = [r for r in records if not r.primary_stratum]
    forced = sorted(
        component_id
        for component_id, assignment in components.items()
        if assignment["reason"] == "touches_observed_map_delivery"
    )
    assigned_train = sorted(
        component_id
        for component_id, assignment in components.items()
        if assignment["split"] == "train"
    )
    assigned_eval = sorted(
        component_id
        for component_id, assignment in components.items()
        if assignment["split"] == "eval"
    )
    return {
        "schema_version": 1,
        "artifact": "recall-map-relevance-dataset-manifest",
        "as_of": AS_OF,
        "evaluator_sha256": sha256_file(Path(__file__)),
        "sealed_inputs": {
            "prereg": {"logical_path": "artifacts/recall-map/prereg.json", "sha256": bindings.prereg_sha256},
            "effect_tool": {"logical_path": "scripts/recall_map_effect.py", "sha256": bindings.effect_sha256},
            "baseline": {"logical_path": "artifacts/recall-map/baseline.json", "sha256": bindings.baseline_sha256},
            "plan_sha256": bindings.protocol.plan_sha256,
        },
        "sources": list(receipts),
        "split": {
            "algorithm": "connected_components(source-qualified cache key, transport session, session)",
            "seed": SPLIT_SEED,
            "organic_train_fraction": ORGANIC_TRAIN_FRACTION,
            "components": len(components),
            "forced_eval_components": forced,
            "train_eval_component_disjoint": True,
        },
        "train": {
            "cohort": "pre-feature organic primary-stratum opportunity-bearing items",
            **_cohort_summary(organic_train),
            "components": assigned_train,
        },
        "eval": {
            "organic": _cohort_summary(organic_eval),
            "observed_map": _cohort_summary(map_eval),
            "components": assigned_eval,
        },
        "secondary_anchor": _cohort_summary(anchor),
        "protocol": {
            "horizon_hours": bindings.protocol.horizon_hours,
            "organic_window": list(bindings.protocol.organic_window),
            "organic_head_cut": bindings.protocol.head_cut,
            "organic_cap": bindings.protocol.organic_cap,
            "map_cap": bindings.protocol.map_cap,
            "endpoint": bindings.protocol.endpoint,
            "stratum_rule": bindings.protocol.stratum_rule,
            "unit_of_analysis": bindings.protocol.unit_of_analysis,
        },
        "privacy": {
            "aggregate_only": True,
            "opaque_component_digests_only": True,
            "item_level_rows": False,
            "corpus_text": False,
            "candidate_holdout_accessed": False,
        },
    }


def _leaf_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _leaf_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _leaf_strings(child)


def _privacy_audit(
    states: Sequence[SourceState], manifest: Mapping[str, Any], analysis: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail if a raw corpus string is copied into an aggregate artifact.

    The sealed privacy guard rejects unsafe encodings, while this independent
    scan rejects exact disclosure of the printable identifiers and semantic
    strings that the guard intentionally cannot distinguish from safe prose.
    Timestamps and source logical names are public manifest metadata and are
    not part of the confidential comparison set.
    """

    confidential: set[str] = set()
    for state in states:
        columns = _table_columns(state.connection, "recall_events")
        wanted = [
            name
            for name in (
                "id",
                "query",
                "task",
                "session_id",
                "transport_session_id",
                "ambient_context",
                "results",
                "recall_map",
            )
            if name in columns
        ]
        for row in state.connection.execute(
            f"SELECT {', '.join(wanted)} FROM recall_events WHERE created_at <= ?",
            (AS_OF,),
        ):
            for name in wanted:
                raw = row[name]
                if raw is None:
                    continue
                value: Any = str(raw)
                if name in {"ambient_context", "results", "recall_map"}:
                    try:
                        value = json.loads(str(raw))
                    except ValueError:
                        pass
                confidential.update(text for text in _leaf_strings(value) if len(text) >= 8)
        for row in state.node_rows.values():
            confidential.add(str(row["id"]))
            confidential.add(str(row["content"] or ""))
            for name in ("context", "provenance", "source_traces"):
                try:
                    value = json.loads(str(row[name] or "null"))
                except ValueError:
                    value = str(row[name] or "")
                confidential.update(text for text in _leaf_strings(value) if len(text) >= 8)
    confidential = {value for value in confidential if len(value) >= 8}
    published = set(_leaf_strings(manifest)) | set(_leaf_strings(analysis))
    matches = confidential & published
    if matches:
        raise EvaluationError(
            f"aggregate privacy audit found {len(matches)} exact raw corpus string match(es)"
        )
    return {
        "raw_strings_compared": len(confidential),
        "published_string_cells_compared": len(published),
        "exact_raw_string_matches": 0,
        "corpus_text_published": False,
    }


def evaluate_sources(
    sources: Sequence[SourceSpec],
    *,
    prereg_path: Path = DEFAULT_PREREG,
    effect_path: Path = DEFAULT_EFFECT_TOOL,
    baseline_path: Path = DEFAULT_BASELINE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not sources:
        raise EvaluationError("at least one source snapshot is required")
    if len({source.name for source in sources}) != len(sources):
        raise EvaluationError("logical source names must be unique")
    bindings = load_frozen_bindings(
        prereg_path=prereg_path, effect_path=effect_path, baseline_path=baseline_path
    )
    states: list[SourceState] = []
    all_pairs: list[tuple[ItemRecord, Any, SourceState]] = []
    receipts: list[dict[str, Any]] = []
    opened_connections: list[sqlite3.Connection] = []
    try:
        for spec in sorted(sources, key=lambda item: item.name):
            receipt = snapshot_receipt(spec)
            connection = bindings.effect.open_readonly(spec.path)
            opened_connections.append(connection)
            connection.execute("PRAGMA query_only = ON")
            connection.execute("BEGIN")
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            missing = {"recall_events", "nodes"} - tables
            if missing:
                raise EvaluationError(
                    f"source {spec.name!r} lacks required table(s): {', '.join(sorted(missing))}"
                )
            _augment_sqlite_receipt(receipt, connection)
            earliest = connection.execute("SELECT MIN(created_at) FROM recall_events").fetchone()[0]
            if earliest is None:
                raise EvaluationError(f"source {spec.name!r} has no recall events")
            has_map = bindings.effect._has_recall_map_column(connection)
            events = bindings.effect.load_events(
                connection, start=str(earliest), end=AS_OF, with_map=has_map
            )
            metas = _load_event_meta(spec, connection, events, bindings.cache_key)
            events = [event for event in events if event.id in metas]
            consumer_index = bindings.effect.ConsumerIndex(
                events, horizon_hours=bindings.protocol.horizon_hours
            )
            gate_stratum = bindings.effect.AnchorStratum(connection)
            deploy = bindings.effect.feature_deploy_instant(connection, as_of=AS_OF)
            observed_map_ids = {
                event.id
                for event in events
                if deploy is not None
                and event.recall_map
                and deploy <= event.created_at < _shift_hours(AS_OF, -bindings.protocol.horizon_hours)
            }
            state = SourceState(
                spec=spec,
                connection=connection,
                receipt_before=receipt,
                events=events,
                metas=metas,
                consumer_index=consumer_index,
                gate_stratum=gate_stratum,
                node_rows={},
                observed_map_event_ids=observed_map_ids,
            )
            pairs = _candidate_records(bindings, state)
            node_ids = {record.node_id for record, _item in pairs}
            state.node_rows = _load_nodes(connection, node_ids)
            history = _history_index(bindings, state, node_ids)
            for record, item in pairs:
                record.consumed, record.primary_stratum = _score_one(
                    bindings, state, state.metas[record.event_id].event, item
                )
                _extract_features(bindings, state, record, history)
                all_pairs.append((record, item, state))
            states.append(state)
            receipts.append(receipt)

        duplicate_snapshots = [
            digest
            for digest, count in Counter(
                str(receipt["snapshot_sha256"]) for receipt in receipts
            ).items()
            if count > 1
        ]
        if duplicate_snapshots:
            raise EvaluationError(
                "the same source snapshot was supplied under multiple logical names"
            )

        records = [record for record, _item, _state in all_pairs]
        if not records:
            raise EvaluationError("pinned sources produced no opportunity-bearing items")
        all_metas = {
            (state.spec.name, event_id): meta
            for state in states
            for event_id, meta in state.metas.items()
        }
        organic_start, organic_end = bindings.protocol.organic_window
        component_event_keys = {
            (state.spec.name, event.id)
            for state in states
            for event in state.events
            if (
                organic_start <= event.created_at < organic_end
                and bool(state.consumer_index.qualifying(event))
            )
            or event.id in state.observed_map_event_ids
        }
        forced_eval_event_keys = {
            (state.spec.name, event_id)
            for state in states
            for event_id in state.observed_map_event_ids
        }
        components = assign_component_splits(
            records,
            all_metas,
            component_event_keys=component_event_keys,
            forced_eval_event_keys=forced_eval_event_keys,
        )
        model = fit_primary_model(records)
        manifest = _manifest(records, components, receipts, bindings)
        baseline_rate = float(
            bindings.baseline["arms"]["organic"]["primary_endpoint"]["rate"]
        )
        analysis = _feature_analysis(
            records, model, bindings.protocol, baseline_rate=baseline_rate
        )
        analysis["dataset_manifest_sha256"] = _sha256_text(canonical_json(manifest))
        analysis["privacy_audit"] = _privacy_audit(states, manifest, analysis)
        try:
            bindings.effect.check_privacy(manifest)
            bindings.effect.check_privacy(analysis)
        except bindings.effect.PrivacyGuardError as exc:
            raise EvaluationError(str(exc)) from exc
        return manifest, analysis
    finally:
        for connection in opened_connections:
            try:
                connection.rollback()
            finally:
                connection.close()
        # Detect snapshot drift, including a WAL transition, across the read.
        for spec, before in zip(sorted(sources, key=lambda item: item.name), receipts, strict=False):
            after = snapshot_receipt(spec)
            before_files = {
                key: before[key]
                for key in ("logical_name", "redacted_path", "parts", "snapshot_sha256")
            }
            if before_files != after:
                raise EvaluationError(f"source snapshot changed during evaluation: {spec.name}")


SENSITIVE_ARTIFACT_KEYS = frozenset(
    {
        "node_id",
        "event_id",
        "query",
        "content",
        "label",
        "task",
        "session_id",
        "cache_key",
        "host",
        "item_rows",
        "event_rows",
    }
)


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def verify_artifacts(
    manifest: Mapping[str, Any],
    analysis: Mapping[str, Any],
    *,
    check_evaluator_hash: bool = True,
) -> list[str]:
    problems: list[str] = []
    if manifest.get("artifact") != "recall-map-relevance-dataset-manifest":
        problems.append("manifest artifact kind mismatch")
    if analysis.get("artifact") != "recall-map-relevance-feature-analysis":
        problems.append("analysis artifact kind mismatch")
    if manifest.get("as_of") != AS_OF or analysis.get("as_of") != AS_OF:
        problems.append("as_of differs from the pinned pre-candidate instant")
    sealed = manifest.get("sealed_inputs", {})
    if sealed.get("prereg", {}).get("sha256") != EXPECTED_PREREG_SHA256:
        problems.append("manifest prereg hash mismatch")
    if sealed.get("effect_tool", {}).get("sha256") != EXPECTED_EFFECT_SHA256:
        problems.append("manifest effect-tool hash mismatch")
    source_receipts = manifest.get("sources", [])
    if not isinstance(source_receipts, list) or not source_receipts:
        problems.append("source snapshot receipts missing")
    else:
        for receipt in source_receipts:
            if not isinstance(receipt, Mapping) or not isinstance(receipt.get("parts"), list):
                problems.append("malformed source snapshot receipt")
                continue
            logical_name = receipt.get("logical_name")
            if not isinstance(logical_name, str) or SOURCE_NAME_RE.fullmatch(logical_name) is None:
                problems.append("source snapshot logical name is invalid")
            elif receipt.get("redacted_path") != f"$SNAPSHOT/{logical_name}.sqlite3":
                problems.append("source snapshot path is not canonically redacted")
            expected = _sha256_text(canonical_json(receipt["parts"]))
            if receipt.get("snapshot_sha256") != expected:
                problems.append("source snapshot receipt digest mismatch")
    if check_evaluator_hash and manifest.get("evaluator_sha256") != sha256_file(Path(__file__)):
        problems.append("manifest evaluator hash mismatch")
    expected_manifest = _sha256_text(canonical_json(dict(manifest)))
    if analysis.get("dataset_manifest_sha256") != expected_manifest:
        problems.append("analysis does not bind the canonical manifest")
    train_components = set(manifest.get("train", {}).get("components", []))
    eval_components = set(manifest.get("eval", {}).get("components", []))
    if train_components & eval_components:
        problems.append("train/eval component overlap")
    all_components = train_components | eval_components
    forced_components = set(manifest.get("split", {}).get("forced_eval_components", []))
    if any(
        not isinstance(value, str) or OPAQUE_DIGEST_RE.fullmatch(value) is None
        for value in all_components
    ):
        problems.append("component assignment contains a non-opaque digest")
    if any(
        not isinstance(value, str) or OPAQUE_DIGEST_RE.fullmatch(value) is None
        for value in forced_components
    ):
        problems.append("forced-eval component contains a non-opaque digest")
    if not forced_components <= eval_components:
        problems.append("forced-eval component is absent from eval")
    if manifest.get("split", {}).get("train_eval_component_disjoint") is not True:
        problems.append("split disjointness assertion missing")
    features = tuple(analysis.get("feature_policy", {}).get("primary_model_features", []))
    try:
        validate_model_feature_names(features)
    except EvaluationError as exc:
        problems.append(str(exc))
    if analysis.get("primary_model", {}).get("fit_cohort") != "organic_train_only":
        problems.append("primary model fit cohort is not organic train only")
    model = analysis.get("primary_model", {})
    if model.get("fit_items") != manifest.get("train", {}).get("items"):
        problems.append("primary model fit denominator differs from organic train")
    if model.get("fit_consumed") != manifest.get("train", {}).get("consumed"):
        problems.append("primary model fit labels differ from organic train")
    map_evaluation = model.get("evaluations", {}).get("observed_map_eval")
    if map_evaluation is None:
        problems.append("observed-map transfer evaluation missing")
    elif map_evaluation.get("total_items") != (
        manifest.get("eval", {}).get("observed_map", {}).get("items")
    ):
        problems.append("observed-map transfer denominator mismatch")
    leakage = analysis.get("leakage_audit", {})
    if leakage.get("candidate_holdout_accessed") is not False:
        problems.append("candidate-holdout non-access assertion missing")
    if leakage.get("future_mutated_node_stats_read") is not False:
        problems.append("future-mutated node-stat rejection assertion missing")
    if leakage.get("mutable_node_columns_read") != []:
        problems.append("mutable node columns entered evaluation")
    if leakage.get("map_rows_in_fit") != 0:
        problems.append("map fitting-row count is not zero")
    synthetic = analysis.get("form_classification_audit", {}).get("synthetic", {})
    expected_cases = 3 * BALLAST_CONTROL_VARIANTS_PER_CLASS + len(USER_AUTHORED_NEAR_MISSES)
    if synthetic.get("seed") != BALLAST_CONTROL_SEED or synthetic.get("cases") != expected_cases:
        problems.append("synthetic form-control seed or denominator mismatch")
    if synthetic.get("false_positives") != 0 or synthetic.get("false_negatives") != 0:
        problems.append("synthetic form controls have classification errors")
    if synthetic.get("case_text_published") is not False:
        problems.append("synthetic form-control text publication assertion missing")
    map_items = manifest.get("eval", {}).get("observed_map", {}).get("items")
    map_score_reasons = (
        analysis.get("availability", {})
        .get("observed_map_eval", {})
        .get("recorded_delivery_scores", {})
        .get("reasons", {})
    )
    if isinstance(map_items, int) and map_items > 0:
        if map_score_reasons.get("not_recorded_in_recall_map_payload") != map_items:
            problems.append("unavailable map residual scores were not fully reported")
    for cohort, summary in analysis.get("availability", {}).items():
        if not isinstance(summary, Mapping):
            continue
        cascade = summary.get("cascade_stage", {})
        if (
            cascade.get("items", 0) > 0
            and cascade.get("reasons", {}).get("not_recorded_at_delivery_not_reconstructed")
            != cascade.get("items")
        ):
            problems.append(f"historical cascade availability reconstructed in {cohort}")
        historical = summary.get("historical_context_provenance", {})
        historical_reasons = historical.get("reasons", {})
        unavailable_count = sum(
            int(historical_reasons.get(reason, 0))
            for reason in (
                "not_versioned_snapshot_diagnostics_excluded_from_fitting",
                "node_snapshot_row_missing",
            )
        )
        if historical.get("items", 0) > 0 and unavailable_count != historical.get("items"):
            problems.append(f"historical context/provenance reconstructed in {cohort}")
    privacy = analysis.get("privacy_audit", {})
    if (
        privacy.get("exact_raw_string_matches") != 0
        or privacy.get("corpus_text_published") is not False
    ):
        problems.append("aggregate raw-string privacy audit missing or failed")
    artifact_keys = set(_walk_keys(manifest)) | set(_walk_keys(analysis))
    leaked = sorted(artifact_keys & SENSITIVE_ARTIFACT_KEYS)
    if leaked:
        problems.append("item-level sensitive artifact key(s): " + ", ".join(leaked))
    return problems


def render_markdown(manifest: Mapping[str, Any], analysis: Mapping[str, Any]) -> str:
    train = manifest["train"]
    organic = manifest["eval"]["organic"]
    observed = manifest["eval"]["observed_map"]
    transfer = analysis["transfer_verdict"]
    model = analysis["primary_model"]
    lines = [
        "# Historical recall-map relevance evidence",
        "",
        f"Pinned as-of: `{manifest['as_of']}`. The candidate holdout was not accessed.",
        "",
        "| cohort | events | items | consumed | rate | components |",
        "|---|---:|---:|---:|---:|---:|",
        f"| organic train | {train['events']} | {train['items']} | {train['consumed']} | {train['rate']} | {len(train['components'])} |",
        f"| organic eval | {organic['events']} | {organic['items']} | {organic['consumed']} | {organic['rate']} | {len(organic['components'])} |",
        f"| observed map eval | {observed['events']} | {observed['items']} | {observed['consumed']} | {observed['rate']} | {len(observed['components'])} |",
        "",
        "The primary deterministic logistic model was fitted once on organic train only. "
        "Recorded delivery scores are analysis-only because map payloads do not record them; "
        "missingness indicators are not model features.",
        "",
        f"Train-fitted threshold: `{model['threshold']}`. Transfer floor: `{transfer['threshold']}`. "
        f"Transfer verdict: **{transfer['verdict']}** (no eval refit).",
        "",
        "Current `access_count`, `usefulness_score`, `last_accessed`, and `updated_at`-derived "
        "usage are rejected. Identity fields and corpus text never enter the model or artifacts.",
        "",
    ]
    return "\n".join(lines)


def write_canonical(path: Path, payload: Mapping[str, Any]) -> None:
    _assert_not_holdout(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(dict(payload)) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    _assert_not_holdout(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"JSON artifact must be an object: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", aliases=["extract"], help="extract, split, fit, and evaluate")
    run.add_argument("--source", action="append", required=True, metavar="NAME=PATH")
    run.add_argument("--as-of", default=AS_OF)
    run.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    run.add_argument("--analysis-out", type=Path, default=DEFAULT_ANALYSIS)
    run.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    run.add_argument("--effect-tool", type=Path, default=DEFAULT_EFFECT_TOOL)
    run.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)

    verify = subparsers.add_parser("verify", help="verify aggregate artifacts")
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    verify.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    verify.add_argument(
        "--source",
        action="append",
        metavar="NAME=PATH",
        help="also re-hash a pinned source snapshot (repeatable)",
    )

    render = subparsers.add_parser("render", help="render deterministic Markdown")
    render.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    render.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    render.add_argument("--out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"run", "extract"}:
        if args.as_of != AS_OF:
            raise EvaluationError(f"--as-of is sealed at {AS_OF}; got {args.as_of}")
        for path in (args.manifest_out, args.analysis_out):
            _assert_not_holdout(path)
        sources = [SourceSpec.parse(value) for value in args.source]
        manifest, analysis = evaluate_sources(
            sources,
            prereg_path=args.prereg,
            effect_path=args.effect_tool,
            baseline_path=args.baseline,
        )
        problems = verify_artifacts(manifest, analysis)
        if problems:
            raise EvaluationError("artifact verification failed: " + "; ".join(problems))
        write_canonical(args.manifest_out, manifest)
        write_canonical(args.analysis_out, analysis)
        print(
            f"wrote aggregate evidence: train={manifest['train']['items']} "
            f"organic_eval={manifest['eval']['organic']['items']} "
            f"map_eval={manifest['eval']['observed_map']['items']}"
        )
        return 0
    if args.command == "verify":
        # Verification of produced evidence still starts by checking the live
        # sealed inputs, so a modified effect script can never be normalized by
        # trusting hashes copied out of an artifact.
        bindings = load_frozen_bindings()
        manifest = _read_json(args.manifest)
        analysis = _read_json(args.analysis)
        problems = verify_artifacts(manifest, analysis)
        try:
            bindings.effect.check_privacy(manifest)
            bindings.effect.check_privacy(analysis)
        except bindings.effect.PrivacyGuardError as exc:
            problems.append(str(exc))
        sealed = manifest.get("sealed_inputs", {})
        if sealed.get("baseline", {}).get("sha256") != bindings.baseline_sha256:
            problems.append("manifest baseline hash mismatch")
        if sealed.get("plan_sha256") != bindings.protocol.plan_sha256:
            problems.append("manifest plan hash mismatch")
        if args.source:
            actual: dict[str, dict[str, Any]] = {}
            for value in args.source:
                spec = SourceSpec.parse(value)
                receipt = snapshot_receipt(spec)
                connection = bindings.effect.open_readonly(spec.path)
                try:
                    connection.execute("PRAGMA query_only = ON")
                    _augment_sqlite_receipt(receipt, connection)
                finally:
                    connection.close()
                actual[spec.name] = receipt
            recorded = {
                str(receipt.get("logical_name")): receipt
                for receipt in manifest.get("sources", [])
                if isinstance(receipt, Mapping)
            }
            if set(actual) != set(recorded):
                problems.append("verified source-name set differs from manifest")
            for name in sorted(set(actual) & set(recorded)):
                if actual[name] != recorded[name]:
                    problems.append(f"source snapshot hash mismatch: {name}")
        if problems:
            for problem in problems:
                print(f"FAIL: {problem}", file=sys.stderr)
            return 1
        print("OK: aggregate artifacts, split, fit provenance, privacy, and sealed hashes verify")
        return 0
    if args.command == "render":
        manifest = _read_json(args.manifest)
        analysis = _read_json(args.analysis)
        problems = verify_artifacts(manifest, analysis)
        if problems:
            raise EvaluationError("refusing to render invalid artifacts: " + "; ".join(problems))
        rendered = render_markdown(manifest, analysis)
        if args.out is None:
            print(rendered, end="")
        else:
            _assert_not_holdout(args.out)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered, encoding="utf-8")
        return 0
    raise AssertionError(args.command)  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
