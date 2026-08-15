#!/usr/bin/env python3
"""ap_baseline.py — the single baseline calculator of the animal-planet audit packet.

Computes every baseline metric mechanically from the tracked de-identified
replay corpus (`artifacts/animal-planet/corpus/`), never from stored
constants: the cited audit numbers appear only in the expected-value/tolerance
table used for comparison (`EXPECTED`), and every deviation is reported as an
explicit discrepancy entry instead of being patched.

Subcommands
-----------
report   Compute the frozen baseline and write baseline-report.json.
verify   Recompute everything and exit 0 iff (a) the recomputation equals the
         stored baseline-report.json, (b) every metric matches its cited value
         within the documented tolerance or is covered by a recorded
         discrepancy entry, and (c) the SHA-256 of
         corpus/{dev,eval,holdout}.jsonl + splits.json match
         artifacts/animal-planet/manifest.json (written by the freeze step).
compare  Before/after JSON on stdout for one split: BEFORE is the recorded
         behavior in the corpus, AFTER is the current code applied to the same
         events (living_memory.delivery.shape_recall_results with current
         defaults, plus a living_memory.replay-style recorded-candidate rerank
         through MemoryRecallService.rank_candidates). The auto_recall family
         additionally replays the server's online repeat-gating rule through a
         real in-memory MemoryStore (metrics.auto_recall.repeat_gating). Its
         additive unseen_in_dev aggregate uses the complete dev split by
         default, or --dev-fingerprint-index for privacy-safe identity across
         independently de-identified builds.

Replay fidelity limits (compare)
--------------------------------
* BM25/vector/graph re-execution is out of scope: the corpus carries the
  recorded per-method candidate scores only; reranks re-mix those recorded
  scores through the current ranking code, they never re-retrieve. Candidates
  the historical ranking excluded can never (re)appear.
* Node stats (confidence/usefulness/access/decayed) and supersedes edges are
  snapshot-time (extraction capture), not event-time.
* Content fields are length/whitespace/equality-class-preserving surrogates;
  snippet truncation may cut at a different clean boundary and causal-query
  markers are not detectable (replay derives causal/decision mode from the
  recorded depth enum only, with a neutral query).
* BEFORE payload chars are the historical transcript wire text
  (ASCII-escaped originals); AFTER payload chars measure the replayed
  response (envelope reconstructed as the current server ships it, with
  `scope`=recorded requested scope and `auto_decay`=null) serialized with
  ensure_ascii=False. Provenance content is not in the corpus; replayed
  provenance is a shape-filler padded to the recorded serialized_chars
  (escape-related residuals of a few chars are counted and reported).
* Session-dedup replay sees only the selected split's slice of each transport
  session (splits are hash-partitioned by event id), applying equally to
  BEFORE and AFTER counters; the delivered-id horizon mirrors
  storage.delivered_node_ids (last 200 session events).

The holdout split is SEALED (corpus/POLICY.md): report/verify aggregate it
only through the same generic code path as dev/eval; `compare --split
holdout` is reserved for the post-freeze shadow-eval stage.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_CORPUS_ROOT = REPO_ROOT / "artifacts" / "animal-planet" / "corpus"

SPLIT_NAMES = ("dev", "eval", "holdout")
CORPUS_FILES = ("dev.jsonl", "eval.jsonl", "holdout.jsonl", "splits.json")
SCHEMA_VERSION = 1

# Verbatim caveat required by the evaluation contract.
FEEDBACK_APPLIED_CAVEAT = "lower-bound proxy for observed use, not ground-truth relevance"

# Mirrors storage.delivered_node_ids(max_events=200): the session dedup horizon.
DELIVERED_HORIZON_EVENTS = 200

# Cross-build repeat identities are keyed HMACs created by the separately
# frozen P6 packet.  They are intentionally accepted only as opaque equality
# tokens: the evaluator never serializes them or derives an unkeyed query
# fingerprint from private text.
DEV_FINGERPRINT_INDEX_KIND = "frozen-dev-automatic-fingerprint-token-index"
DEV_FINGERPRINT_NORMALIZATION = '" ".join(query.split()) + "\\n" + requested_scope'
OPAQUE_REPLAY_SCOPE = "internal:opaque-recall-fingerprint-v1"
_LOWER_HEX64 = re.compile(r"^[0-9a-f]{64}$")

AUDIT_NODE = "audit node 01KZK2WMP07CNTYDQXFTS23F39 (scope project:ae)"
PAYLOAD_TRACE = "payload follow-up trace 01KZV6VCVGXVKPF6PSFXBH00EM"
LINKAGE_TRACE = "net-value trace 01KZVZ5ZTE1WSXFHCK6091H0RE / root-goal field baseline"

# ---------------------------------------------------------------------------
# EXPECTED-VALUE / TOLERANCE TABLE — the ONLY place cited constants may live.
# Every value under "cited" is a citation from the audit/goal record, present
# strictly for comparison; nothing here is ever emitted as a computed metric.
# Tolerance kinds: pp (± percentage points), pct (± relative percent),
# exact (integer equality), not_derivable (cannot be computed from the
# de-identified corpus; requires a covering discrepancy entry).
# ---------------------------------------------------------------------------
EXPECTED: dict[str, dict[str, Any]] = {
    "organic_linkage_pct": {
        "cited": 42.4,
        "tolerance": {"kind": "pp", "value": 0.5},
        "source": LINKAGE_TRACE,
    },
    "automatic_linkage_pct": {
        "cited": 17.6,
        "tolerance": {"kind": "pp", "value": 0.5},
        "source": LINKAGE_TRACE,
    },
    "payload_median_chars": {
        "cited": 26496,
        "tolerance": {"kind": "pct", "value": 1.0},
        "source": PAYLOAD_TRACE,
    },
    "payload_p90_chars": {
        "cited": 35237,
        "tolerance": {"kind": "pct", "value": 1.0},
        "source": PAYLOAD_TRACE,
    },
    "payload_mean_chars": {
        "cited": 26543,
        "tolerance": {"kind": "pct", "value": 1.0},
        "source": PAYLOAD_TRACE,
    },
    "payload_n": {
        "cited": 278,
        "tolerance": {"kind": "exact"},
        "source": PAYLOAD_TRACE,
    },
    "w1_recall_events": {
        "cited": 727,
        "tolerance": {"kind": "exact"},
        "source": AUDIT_NODE,
    },
    "w1_supersedes_edges": {
        "cited": 9,
        "tolerance": {"kind": "exact"},
        "source": AUDIT_NODE,
    },
    "w1_reopen_lesson_events": {
        "cited": 152,
        "tolerance": {"kind": "exact"},
        "source": AUDIT_NODE,
    },
    "w1_architectural_decision_events": {
        "cited": 152,
        "tolerance": {"kind": "exact"},
        "source": AUDIT_NODE,
    },
    "w1_deterministic_template_share_pct": {
        "cited": 42.0,
        "tolerance": {"kind": "pp", "value": 0.5},
        "source": AUDIT_NODE + " ('42% deterministic pre-recalls')",
    },
    "w2_game_events": {
        "cited": 1349,
        "tolerance": {"kind": "exact"},
        "source": LINKAGE_TRACE,
    },
    "w2_game_automatic_events": {
        "cited": 1188,
        "tolerance": {"kind": "exact"},
        "source": LINKAGE_TRACE,
    },
    "w2_game_automatic_feedback": {
        "cited": 209,
        "tolerance": {"kind": "exact"},
        "source": LINKAGE_TRACE,
    },
    "w2_game_organic_events": {
        "cited": 165,
        "tolerance": {"kind": "exact"},
        "source": LINKAGE_TRACE,
    },
    "w2_game_organic_feedback": {
        "cited": 70,
        "tolerance": {"kind": "exact"},
        "source": LINKAGE_TRACE,
    },
    "w1_auto_outcome_traces": {
        "cited": 105,
        "tolerance": {"kind": "not_derivable"},
        "source": AUDIT_NODE,
    },
    "w1_avg_recall_kb": {
        "cited": 12.6,
        "tolerance": {"kind": "not_derivable"},
        "source": AUDIT_NODE,
    },
}

# Known, documented deviations — copied/extended from the packet metadata
# (recipe/01-extract.md "Cross-check outcome", mirrored from the private
# staging METADATA.json.discrepancies[]). Computed/cited numbers are filled in
# dynamically at report time; the explanations are the recorded findings.
DISCREPANCY_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "d1-organic-linkage-population",
        "metrics": [
            "organic_linkage_pct",
            "w2_game_organic_events",
            "w2_game_organic_feedback",
        ],
        "explanation": (
            "The cited organic population is internally inconsistent: cited organic"
            " events + cited automatic events do not sum to the W2 project:game"
            " window total, and cited feedback counts do not sum to the window's"
            " total feedback either. No column-level predicate reproduces the cited"
            " organic split; the corpus pins the class predicate 'automatic iff"
            " agent IS NULL' (splits.json event_class_predicate), under which the"
            " automatic side is exact and the organic side differs as recorded"
            " here. The qualitative baseline gap (organic linkage far above"
            " automatic) is preserved."
        ),
        "source": "recipe/01-extract.md discrepancy 1; staging METADATA.json.discrepancies[]",
    },
    {
        "id": "d2-auto-outcome-not-derivable",
        "metrics": ["w1_auto_outcome_traces"],
        "explanation": (
            "Auto-OUTCOME traces are identified by a content template"
            " (node.sh _node_lm_record_outcome, 'OUTCOME pass/fail: ...' with"
            " agent='ae'); corpus content and agent fields are de-identified"
            " surrogates, so the marker cannot be recomputed from the tracked"
            " corpus at all. At extract time the pinned template matched 78 W1"
            " traces (89 under a broad substring match) against the cited 105;"
            " the snapshot is a lower bound because identical retry contents"
            " dedup at remember time and traces can decay or be forgotten"
            " between the audit and the snapshot."
        ),
        "source": "recipe/01-extract.md discrepancy 2; staging METADATA.json.discrepancies[]",
    },
    {
        "id": "d3-w3-payload-population",
        "metrics": [
            "payload_median_chars",
            "payload_p90_chars",
            "payload_mean_chars",
            "payload_n",
        ],
        "explanation": (
            "The cited W3 payload population decomposes exactly as the surviving"
            " transcript-matched recalls (carried in the corpus) plus 114 W3 DB"
            " events from /root-agent manual sessions whose transcripts are"
            " unreadable (permission denied) or removed on the evidence host; the"
            " serialized sizes of those tool-results cannot be re-measured from"
            " surviving evidence. The corpus reproduces the survivor population's"
            " statistics exactly; the cited median/p90/mean/n cover the larger,"
            " partly unrecoverable population."
        ),
        "source": "recipe/01-extract.md discrepancy 3; staging METADATA.json.discrepancies[]",
    },
    {
        "id": "d4-avg-recall-kb-measure-basis",
        "metrics": ["w1_avg_recall_kb"],
        "explanation": (
            "The audit's 'avg 12.6KB/recall' (and its 18.8% share of tool-result"
            " volume) is measured on the len_text basis — the sum of text-block"
            " lengths over ALL W1 LM tool-results, including tools other than"
            " memory_recall. The corpus carries only the len_json_content measure"
            " for transcript-matched recall events, so this figure is not"
            " derivable from the tracked corpus; it was reproduced exactly at the"
            " extract stage from the raw transcripts."
        ),
        "source": "recipe/01-extract.md 'Serialized payload measure'",
    },
]


# ---------------------------------------------------------------------------
# Corpus loading (pure stdlib)
# ---------------------------------------------------------------------------


def load_split(corpus_root: Path, split: str) -> tuple[dict[str, dict], list[dict]]:
    """Load one split file into (nodes by id, events ordered as stored)."""

    nodes: dict[str, dict] = {}
    events: list[dict] = []
    path = corpus_root / f"{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["type"] == "node":
                nodes[record["id"]] = record
            elif record["type"] == "event":
                events.append(record)
            else:
                raise ValueError(f"{path}: unknown record type {record['type']!r}")
    return nodes, events


def load_splits_meta(corpus_root: Path) -> dict[str, Any]:
    return json.loads((corpus_root / "splits.json").read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def in_window(created_at: str, window: dict[str, str]) -> bool:
    """Inclusive-both-ends ISO-Z comparison (recipe: 'Inclusive both ends')."""

    return window["start"] <= created_at <= window["end"]


def supersedes_edges(nodes: dict[str, dict]) -> dict[tuple[str, str, str], None]:
    """Deduped supersedes edges (source=superseding, target=superseded, created_at)."""

    edges: dict[tuple[str, str, str], None] = {}
    for node in nodes.values():
        for relation in node.get("relations") or []:
            if relation.get("type") != "supersedes":
                continue
            if relation["direction"] == "out":
                source, target = node["id"], relation["other_id"]
            else:
                source, target = relation["other_id"], node["id"]
            edges[(source, target, relation.get("created_at") or "")] = None
    return edges


def percentile_nearest_rank(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile (matches the extract-stage measurement)."""

    if not sorted_values:
        raise ValueError("empty population")
    index = max(0, math.ceil(fraction * len(sorted_values)) - 1)
    return sorted_values[index]


# ---------------------------------------------------------------------------
# report — compute the frozen baseline from the corpus
# ---------------------------------------------------------------------------


def compute_report(corpus_root: Path) -> dict[str, Any]:
    splits_meta = load_splits_meta(corpus_root)
    windows = splits_meta["windows"]

    file_info: dict[str, dict[str, Any]] = {}
    all_events: list[dict] = []
    all_edges: dict[tuple[str, str, str], None] = {}
    per_split_counts: dict[str, dict[str, int]] = {}

    # One generic pass per split — identical mechanical aggregation for
    # dev/eval/holdout (holdout is never inspected per-case, POLICY.md).
    for split in SPLIT_NAMES:
        nodes, events = load_split(corpus_root, split)
        all_events.extend(events)
        all_edges.update(supersedes_edges(nodes))
        per_split_counts[split] = {"events": len(events), "nodes": len(nodes)}
        file_info[f"{split}.jsonl"] = {
            "sha256": sha256_file(corpus_root / f"{split}.jsonl"),
            "events": len(events),
            "nodes": len(nodes),
        }
    file_info["splits.json"] = {"sha256": sha256_file(corpus_root / "splits.json")}

    alt_events = [event for event in all_events if event.get("source") == "alt"]

    # --- W1 cross-checks -------------------------------------------------
    w1_events = [e for e in alt_events if in_window(e["created_at"], windows["W1"])]
    w1_templates: dict[str, int] = defaultdict(int)
    for event in w1_events:
        w1_templates[str(event.get("template_id"))] += 1
    w1_deterministic = w1_templates.get("reopen_lesson", 0) + w1_templates.get(
        "architectural_decision", 0
    )
    w1_edge_count = sum(
        1 for (_s, _t, created) in all_edges if in_window(created, windows["W1"])
    )
    w1_class: dict[str, int] = defaultdict(int)
    for event in w1_events:
        w1_class[event["class"]] += 1

    # --- W2 linkage (scope project:game) ---------------------------------
    w2_game = [
        e
        for e in alt_events
        if e.get("scope") == "project:game" and in_window(e["created_at"], windows["W2"])
    ]
    organic = [e for e in w2_game if e["class"] == "organic"]
    automatic = [e for e in w2_game if e["class"] == "automatic"]
    organic_fb = sum(1 for e in organic if e.get("feedback_applied"))
    automatic_fb = sum(1 for e in automatic if e.get("feedback_applied"))

    # --- W3 payload (transcript-matched tool-result chars) ----------------
    w3_chars = sorted(
        event["transcript_serialized_chars"]
        for event in alt_events
        if event.get("transcript_matched")
        and event.get("transcript_serialized_chars") is not None
        and in_window(event["created_at"], windows["W3"])
    )

    metrics: dict[str, Any] = {
        "organic_linkage_pct": round(100.0 * organic_fb / len(organic), 4) if organic else None,
        "automatic_linkage_pct": (
            round(100.0 * automatic_fb / len(automatic), 4) if automatic else None
        ),
        "payload_median_chars": round(statistics.median(w3_chars), 1) if w3_chars else None,
        "payload_p90_chars": percentile_nearest_rank(w3_chars, 0.90) if w3_chars else None,
        "payload_mean_chars": round(sum(w3_chars) / len(w3_chars), 2) if w3_chars else None,
        "payload_n": len(w3_chars),
        "cross_checks": {
            "w1_recall_events": len(w1_events),
            "w1_supersedes_edges": w1_edge_count,
            "w1_reopen_lesson_events": w1_templates.get("reopen_lesson", 0),
            "w1_architectural_decision_events": w1_templates.get("architectural_decision", 0),
            "w1_deterministic_template_share_pct": (
                round(100.0 * w1_deterministic / len(w1_events), 4) if w1_events else None
            ),
            "w1_class_counts": dict(sorted(w1_class.items())),
            "w2_game_events": len(w2_game),
            "w2_game_organic_events": len(organic),
            "w2_game_organic_feedback": organic_fb,
            "w2_game_automatic_events": len(automatic),
            "w2_game_automatic_feedback": automatic_fb,
            # Content/agent fields are surrogates: the OUTCOME content template
            # is not recomputable from the corpus (discrepancy d2).
            "w1_auto_outcome_traces": None,
            # len_text basis over all W1 tool-results is not carried in the
            # corpus (discrepancy d4).
            "w1_avg_recall_kb": None,
        },
    }

    flat = flatten_metrics(metrics)
    comparison = []
    failing_uncovered = []
    covering = {
        metric: template["id"]
        for template in DISCREPANCY_TEMPLATES
        for metric in template["metrics"]
    }
    for name, expectation in EXPECTED.items():
        computed = flat.get(name)
        verdict = compare_to_cited(computed, expectation)
        row = {
            "metric": name,
            "computed": computed,
            "cited": expectation["cited"],
            "tolerance": expectation["tolerance"],
            "source": expectation["source"],
            "within_tolerance": verdict,
            "covered_by_discrepancy": covering.get(name),
        }
        comparison.append(row)
        if verdict is not True and covering.get(name) is None:
            failing_uncovered.append(name)

    discrepancies = []
    for template in DISCREPANCY_TEMPLATES:
        discrepancies.append(
            {
                "id": template["id"],
                "metrics": template["metrics"],
                "computed": {name: flat.get(name) for name in template["metrics"]},
                "cited": {name: EXPECTED[name]["cited"] for name in template["metrics"]},
                "explanation": template["explanation"],
                "source": template["source"],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "calculator": "scripts/ap_baseline.py",
        "corpus": {
            "root": "artifacts/animal-planet/corpus",
            "files": dict(sorted(file_info.items())),
            "splits": per_split_counts,
            "alt_events": len(alt_events),
            "total_events": len(all_events),
        },
        "windows": {name: windows[name] for name in sorted(windows)},
        "window_convention": "created_at within [start, end], inclusive both ends",
        "populations": {
            "w1_alt_events": len(w1_events),
            "w2_game_events": len(w2_game),
            "w3_transcript_matched_events": len(w3_chars),
            "supersedes_edges_in_corpus": len(all_edges),
        },
        "metrics": metrics,
        "definitions": {
            "organic_linkage_pct": (
                "share of W2 project:game events with class=organic (splits.json"
                " event_class_predicate: automatic iff agent IS NULL) that have"
                " feedback_applied set"
            ),
            "automatic_linkage_pct": (
                "share of W2 project:game events with class=automatic that have"
                " feedback_applied set"
            ),
            "payload_*_chars": (
                "median/nearest-rank-p90/mean of transcript_serialized_chars"
                " (len_json_content of the matched MCP recall tool-result) over"
                " transcript-matched events created in W3"
            ),
            "w1_supersedes_edges": (
                "supersedes relations (deduped by source,target,created_at across"
                " split files) created within W1"
            ),
        },
        "tolerances": {
            "linkage_pct": "±0.5 percentage points vs the rounded cited values",
            "payload_chars": "±1% vs the cited values (covers the 26.5k/35.2k roundings)",
            "counts": "exact",
        },
        "caveats": {
            "feedback_applied": FEEDBACK_APPLIED_CAVEAT,
        },
        "expected": EXPECTED,
        "comparison": comparison,
        "discrepancies": discrepancies,
        "failing_uncovered_metrics": failing_uncovered,
    }


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flat = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    flat.update(metrics.get("cross_checks", {}))
    return flat


def compare_to_cited(computed: Any, expectation: dict[str, Any]) -> bool | None:
    """True/False when checkable; None when not derivable from the corpus."""

    tolerance = expectation["tolerance"]
    if tolerance["kind"] == "not_derivable" or computed is None:
        return None
    cited = expectation["cited"]
    if tolerance["kind"] == "exact":
        return computed == cited
    if tolerance["kind"] == "pp":
        return abs(float(computed) - float(cited)) <= tolerance["value"]
    if tolerance["kind"] == "pct":
        return abs(float(computed) - float(cited)) <= float(cited) * tolerance["value"] / 100.0
    raise ValueError(f"unknown tolerance kind: {tolerance['kind']}")


def dump_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    corpus_root = args.corpus_root
    report = compute_report(corpus_root)
    out_path = args.out or (corpus_root.parent / "baseline-report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dump_json(report), encoding="utf-8")

    ok = sum(1 for row in report["comparison"] if row["within_tolerance"] is True)
    covered = sum(
        1
        for row in report["comparison"]
        if row["within_tolerance"] is not True and row["covered_by_discrepancy"]
    )
    print(f"wrote {out_path}")
    print(
        f"comparison: {ok}/{len(report['comparison'])} within tolerance, "
        f"{covered} deviations covered by recorded discrepancies"
    )
    if report["failing_uncovered_metrics"]:
        print(
            "WARNING: deviations NOT covered by any discrepancy entry: "
            + ", ".join(report["failing_uncovered_metrics"])
        )
    return 0


# ---------------------------------------------------------------------------
# verify — gate the frozen packet
# ---------------------------------------------------------------------------

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def find_manifest_hashes(manifest: Any) -> dict[str, set[str]]:
    """Collect sha256 values associated with each corpus file in the manifest.

    Tolerant of manifest shape: a hash binds to a corpus file when the dict key
    holding it, an ancestor key, or a sibling string value names that file
    (e.g. {"corpus/dev.jsonl": {"sha256": ...}} or
    {"path": "corpus/dev.jsonl", "sha256": ...}).
    """

    found: dict[str, set[str]] = {name: set() for name in CORPUS_FILES}

    def walk(obj: Any, context: frozenset[str]) -> None:
        if isinstance(obj, dict):
            names = set(context)
            for key, value in obj.items():
                for target in CORPUS_FILES:
                    if isinstance(key, str) and key.endswith(target):
                        names.add(target)
                    if isinstance(value, str) and value.endswith(target):
                        names.add(target)
            for key, value in obj.items():
                if isinstance(value, str) and _HEX64.match(value):
                    key_names = {t for t in CORPUS_FILES if key.endswith(t)}
                    for target in key_names or names:
                        found[target].add(value)
                elif isinstance(value, (dict, list)):
                    child = frozenset(
                        {t for t in CORPUS_FILES if isinstance(key, str) and key.endswith(t)}
                        or names
                    )
                    walk(value, child)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, context)

    walk(manifest, frozenset())
    return found


def cmd_verify(args: argparse.Namespace) -> int:
    corpus_root = args.corpus_root
    packet_root = corpus_root.parent
    failures: list[str] = []

    manifest_path = packet_root / "manifest.json"
    if not manifest_path.exists():
        print(
            f"verify: manifest missing at {manifest_path} — the immutable manifest is"
            " written by the freeze step (freeze-manifest); run it before verify.",
            file=sys.stderr,
        )
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hashes = find_manifest_hashes(manifest)
    for name in CORPUS_FILES:
        actual = sha256_file(corpus_root / name)
        expected_set = manifest_hashes[name]
        if not expected_set:
            failures.append(f"manifest lists no sha256 for corpus file {name}")
        elif len(expected_set) > 1:
            failures.append(f"manifest lists conflicting sha256 values for {name}")
        elif actual not in expected_set:
            failures.append(
                f"sha256 mismatch for {name}: corpus {actual}, manifest {next(iter(expected_set))}"
            )

    report_path = packet_root / "baseline-report.json"
    if not report_path.exists():
        failures.append(f"baseline report missing at {report_path} — run `report` first")
        recomputed = None
    else:
        stored = json.loads(report_path.read_text(encoding="utf-8"))
        recomputed = compute_report(corpus_root)
        if recomputed != stored:
            for diff in dict_diff_paths(stored, recomputed)[:8]:
                failures.append(f"baseline-report.json mismatch at {diff}")

    active = recomputed if recomputed is not None else None
    if active is not None:
        for row in active["comparison"]:
            if row["within_tolerance"] is True:
                continue
            if row["covered_by_discrepancy"]:
                continue
            failures.append(
                f"metric {row['metric']} deviates from cited {row['cited']}"
                f" (computed {row['computed']}) with no covering discrepancy entry"
            )

    if failures:
        for failure in failures:
            print(f"verify: FAIL: {failure}", file=sys.stderr)
        return 1
    print(
        "verify: OK — corpus hashes match the manifest, the baseline report"
        " reproduces from the corpus, and every cited metric is within tolerance"
        " or covered by a recorded discrepancy"
    )
    return 0


def dict_diff_paths(a: Any, b: Any, prefix: str = "$") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        paths: list[str] = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                paths.append(f"{prefix}.{key} (missing on one side)")
            elif a[key] != b[key]:
                paths.extend(dict_diff_paths(a[key], b[key], f"{prefix}.{key}"))
        return paths
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{prefix} (list length {len(a)} != {len(b)})"]
        paths = []
        for index, (item_a, item_b) in enumerate(zip(a, b)):
            if item_a != item_b:
                paths.extend(dict_diff_paths(item_a, item_b, f"{prefix}[{index}]"))
        return paths
    return [f"{prefix} ({a!r} != {b!r})"]


# ---------------------------------------------------------------------------
# compare — BEFORE (recorded) vs AFTER (current code) on one split
# ---------------------------------------------------------------------------

METRIC_FAMILIES = ("payload", "cross_scope", "correction_dominance", "auto_recall")

FIDELITY_LIMITS = [
    "BM25/vector/graph re-execution out of scope: reranks re-mix recorded candidate scores only",
    "node stats and supersedes edges are snapshot-time, not event-time",
    "content fields are surrogates: snippet boundaries and causal-query detection are approximate",
    "AFTER payload reconstructs the current server envelope (scope=requested_scope, auto_decay=null), ensure_ascii=False",
    "provenance is a shape-filler padded to recorded serialized_chars; small escape residuals counted",
    "session replay sees only this split's slice of each transport session (before and after equally)",
]


def _import_living_memory() -> dict[str, Any]:
    sys.path.insert(0, str(SRC_ROOT))
    from living_memory.config import MemoryConfig
    from living_memory.delivery import (
        context_value_max_chars_from_env,
        full_node_diet_enabled_from_env,
        provenance_value_max_chars_from_env,
        session_dedup_enabled_from_env,
        shape_recall_results,
        snippet_ladder_from_env,
        snippet_max_chars_from_env,
        sparse_entries_enabled_from_env,
        stats_compaction_enabled_from_env,
    )
    from living_memory.models import Node
    from living_memory.replay import assumed_evidence, floor_default_weights, make_ranking_service
    from living_memory.resources import node_to_dict
    from living_memory.retrieval import (
        RecallResult,
        _Candidate,
        _is_decision_depth,
        _parse_depth,
    )
    from living_memory.scope import ScopePlan
    from living_memory.storage import (
        FingerprintGatePolicy,
        MemoryStore,
        recall_fingerprint,
        should_gate_fingerprint,
    )

    return {
        "MemoryConfig": MemoryConfig,
        "MemoryStore": MemoryStore,
        "FingerprintGatePolicy": FingerprintGatePolicy,
        "recall_fingerprint": recall_fingerprint,
        "should_gate_fingerprint": should_gate_fingerprint,
        "node_to_dict": node_to_dict,
        "shape_recall_results": shape_recall_results,
        "snippet_max_chars_from_env": snippet_max_chars_from_env,
        "context_value_max_chars_from_env": context_value_max_chars_from_env,
        "session_dedup_enabled_from_env": session_dedup_enabled_from_env,
        "snippet_ladder_from_env": snippet_ladder_from_env,
        "full_node_diet_enabled_from_env": full_node_diet_enabled_from_env,
        "provenance_value_max_chars_from_env": provenance_value_max_chars_from_env,
        "stats_compaction_enabled_from_env": stats_compaction_enabled_from_env,
        "sparse_entries_enabled_from_env": sparse_entries_enabled_from_env,
        "Node": Node,
        "assumed_evidence": assumed_evidence,
        "floor_default_weights": floor_default_weights,
        "make_ranking_service": make_ranking_service,
        "RecallResult": RecallResult,
        "_Candidate": _Candidate,
        "_is_decision_depth": _is_decision_depth,
        "_parse_depth": _parse_depth,
        "ScopePlan": ScopePlan,
    }


def _json_chars(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str))


def build_provenance_filler(record: dict) -> tuple[dict[str, Any], int]:
    """Reconstruct a provenance dict whose delivery footprint matches the record.

    prior_recalls keeps its exact element count (the summarization branch in
    delivery reduces it to {"count": n}); every other key becomes a filler of
    the recorded per-key char size; a final pad absorbs the remaining delta up
    to the recorded serialized_chars of the merged provenance dict. Returns
    (provenance_without_source_traces_corrections, residual_chars).
    """

    shape = record.get("provenance_shape") or {}
    keys: dict[str, Any] = shape.get("keys") or {}
    provenance: dict[str, Any] = {}
    for key in sorted(keys):
        meta = keys[key]
        kind = meta.get("kind")
        chars = int(meta.get("chars") or 0)
        count = int(meta.get("count") or 0)
        if key == "prior_recalls" and kind == "list":
            provenance[key] = _list_filler(count, chars)
        elif kind == "list":
            provenance[key] = _list_filler(min(count, 1) if chars < 6 else 1, chars)
        elif kind == "int":
            provenance[key] = int("1" + "0" * max(0, chars - 1))
        else:  # str and anything else: a string with the same delivery chars
            provenance[key] = "a" * chars

    target = shape.get("serialized_chars")
    residual = 0
    if target is not None:
        merged = {
            **provenance,
            "source_traces": record.get("source_traces") or [],
            "corrections": record.get("corrections") or [],
        }
        delta = int(target) - _json_chars(merged)
        residual = _absorb_delta(provenance, delta)
    return provenance, residual


def _list_filler(count: int, chars: int) -> list[str]:
    if count <= 0:
        return []
    budget = chars - 2 - 2 * count - 2 * (count - 1)
    if budget < 0:
        return ["" for _ in range(count)]
    base, remainder = divmod(budget, count)
    return ["a" * (base + 1) if i < remainder else "a" * base for i in range(count)]


def _absorb_delta(provenance: dict[str, Any], delta: int) -> int:
    """Adjust fillers so the merged provenance hits its recorded size; return residual."""

    if delta == 0:
        return 0
    prior = provenance.get("prior_recalls")
    if isinstance(prior, list) and prior and isinstance(prior[0], str):
        grown = len(prior[0]) + delta
        if grown >= 0:
            prior[0] = "a" * grown
            return 0
    for key, value in provenance.items():
        if isinstance(value, str):
            grown = len(value) + delta
            if grown >= 0:
                provenance[key] = "a" * grown
                return 0
    if delta >= 12:
        provenance["_pad"] = "a" * (delta - 12)
        return 0
    return delta


class ReplayCorpus:
    """One split loaded for replay: node objects, recall results, sessions."""

    def __init__(self, corpus_root: Path, split: str, lm: dict[str, Any]) -> None:
        self.corpus_root = corpus_root
        self.split = split
        self.lm = lm
        self.node_records, self.events = load_split(corpus_root, split)
        self.events.sort(key=lambda event: (event["created_at"], event["id"]))
        self.edges = supersedes_edges(self.node_records)
        self.provenance_residuals = 0
        self._nodes: dict[str, Any] = {}

    def node(self, node_id: str) -> Any:
        cached = self._nodes.get(node_id)
        if cached is not None:
            return cached
        record = self.node_records[node_id]
        provenance, residual = build_provenance_filler(record)
        self.provenance_residuals += abs(residual)
        stats = record.get("stats") or {}
        node = self.lm["Node"](
            id=record["id"],
            level=record["level"],
            content=record.get("content_surrogate") or "",
            context=record.get("context_surrogate") or {},
            scope=record["scope"],
            agent=record.get("agent_surrogate"),
            task=record.get("task_surrogate"),
            timestamp=record.get("timestamp") or "",
            decayed=bool(record.get("decayed")),
            decay_reason=record.get("decay_reason_surrogate"),
            access_count=int(stats.get("access_count") or 0),
            last_accessed=stats.get("last_accessed"),
            usefulness_score=float(stats.get("usefulness_score") or 0.0),
            confidence=float(stats.get("confidence") if stats.get("confidence") is not None else 0.5),
            unique_agents=int(stats.get("unique_agents") or 1),
            temporal_hint=stats.get("temporal_hint"),
            source_traces=list(record.get("source_traces") or []),
            corrections=[dict(c) for c in record.get("corrections") or []],
            provenance=provenance,
            created_at=record.get("created_at") or "",
            updated_at=record.get("updated_at") or "",
        )
        self._nodes[node_id] = node
        return node

    def plan(self, event: dict) -> Any:
        resolved = tuple(event.get("resolved_scopes") or [event["scope"]])
        return self.lm["ScopePlan"](
            requested_scope=event.get("requested_scope") or event["scope"],
            scopes=resolved,
        )

    def recall_results(self, event: dict) -> list[Any]:
        plan = self.plan(event)
        results = []
        for candidate in sorted(event["results"], key=lambda c: c.get("rank") or 0):
            node = self.node(candidate["node_id"])
            results.append(
                self.lm["RecallResult"](
                    node=node,
                    score=float(candidate.get("score") or 0.0),
                    bm25_score=float(candidate.get("bm25_score") or 0.0),
                    vector_score=float(candidate.get("vector_score") or 0.0),
                    graph_score=float(candidate.get("graph_score") or 0.0),
                    trigger_score=float(candidate.get("trigger_score") or 0.0),
                    scope_rank=plan.rank(node.scope),
                    methods=tuple(candidate.get("methods") or ()),
                    path=tuple(candidate.get("path") or ()),
                    recall_event_id=event["id"],
                )
            )
        return results

    def sessions(self) -> list[list[dict]]:
        """Events grouped by transport session (session-less events stand alone)."""

        grouped: dict[str, list[dict]] = defaultdict(list)
        singletons: list[list[dict]] = []
        for event in self.events:
            transport = event.get("transport_session_id")
            if transport:
                grouped[transport].append(event)
            else:
                singletons.append([event])
        ordered = [grouped[key] for key in sorted(grouped)]
        ordered.extend(singletons)
        return ordered


def replay_delivery(corpus: ReplayCorpus) -> dict[str, dict[str, Any]]:
    """Shape every event through current delivery defaults, per transport session.

    Returns per-event: delivery class counts, replayed serialized chars, which
    result node ids were session-repeats at delivery time, and the replayed
    retention flags — ``top_retained`` (the top-ranked entry bears complete
    content, or is a session-duplicate of a node whose content this session
    already delivered inline) and ``content_access_retained`` (every result is
    content-bearing inline, a twin of an in-response bearer, or a
    session-duplicate with prior in-session content delivery). The BEFORE
    renderer ships every result's complete content inline, so both flags are
    true there by construction.
    """

    lm = corpus.lm
    shape = lm["shape_recall_results"]
    snippet_chars = lm["snippet_max_chars_from_env"]()
    context_chars = lm["context_value_max_chars_from_env"]()
    dedup_default = lm["session_dedup_enabled_from_env"]()
    snippet_ladder = lm["snippet_ladder_from_env"]()
    full_node_diet = lm["full_node_diet_enabled_from_env"]()
    provenance_chars = lm["provenance_value_max_chars_from_env"]()
    stats_compaction = lm["stats_compaction_enabled_from_env"]()
    sparse_entries = lm["sparse_entries_enabled_from_env"]()

    outcome: dict[str, dict[str, Any]] = {}
    for session in corpus.sessions():
        history: list[set[str]] = []
        content_delivered: set[str] = set()
        for event in session:
            transport = event.get("transport_session_id")
            session_dedup = bool(transport) and dedup_default
            delivered: set[str] = set()
            for previous in history[-DELIVERED_HORIZON_EVENTS:]:
                delivered |= previous
            results = corpus.recall_results(event)
            shaped = shape(
                results,
                already_delivered_ids=delivered,
                snippet_max_chars=snippet_chars,
                context_value_max_chars=context_chars,
                session_dedup=session_dedup,
                snippet_ladder=snippet_ladder,
                full_node_diet=full_node_diet,
                provenance_value_max_chars=provenance_chars,
                stats_compaction=stats_compaction,
                sparse_entries=sparse_entries,
            )
            envelope = {
                "query": event.get("query_surrogate") or "",
                "scope": event.get("requested_scope") or event.get("scope"),
                "recall_event_id": event["id"],
                "count": len(shaped),
                "results": shaped,
                "auto_decay": None,
            }
            # Legacy-renderer emulation (pre-diet: every result full, provenance
            # verbatim, no delivery/content_ref keys) — the reconstruction-bias
            # anchor: unshaped-vs-recorded isolates snapshot/envelope bias,
            # after-vs-unshaped isolates the current shaping effect.
            unshaped_entries = [
                {
                    "node": lm["node_to_dict"](result.node),
                    "score": result.score,
                    "bm25_score": result.bm25_score,
                    "vector_score": result.vector_score,
                    "graph_score": result.graph_score,
                    "trigger_score": result.trigger_score,
                    "scope_rank": result.scope_rank,
                    "methods": list(result.methods),
                    "path": list(result.path),
                    "recall_event_id": result.recall_event_id,
                }
                for result in results
            ]
            unshaped_envelope = {**envelope, "count": len(unshaped_entries), "results": unshaped_entries}
            result_ids = [c["node_id"] for c in event["results"] if c.get("node_id")]
            outcome[event["id"]] = {
                "chars": len(json.dumps(envelope, ensure_ascii=False)),
                "unshaped_chars": len(json.dumps(unshaped_envelope, ensure_ascii=False)),
                "classes": [entry["delivery"] for entry in shaped],
                "repeat_ids": [nid for nid in result_ids if nid in delivered],
                "result_ids": result_ids,
                "top_retained": _top_content_retained(shaped, content_delivered),
                "content_access_retained": _content_access_retained(shaped, content_delivered),
            }
            content_delivered.update(
                entry["node"]["id"]
                for entry in shaped
                if entry["delivery"] in ("full", "snippet")
            )
            history.append(set(result_ids))
    return outcome


def _top_content_retained(shaped: list[dict[str, Any]], content_delivered: set[str]) -> bool | None:
    """Top-ranked content survives delivery (complete inline, or already
    delivered inline earlier this session). None when the event had no results."""

    if not shaped:
        return None
    top = shaped[0]
    if top["delivery"] == "full":
        return True
    if top["delivery"] == "session_duplicate":
        return top["node"]["id"] in content_delivered
    return False


def _content_access_retained(
    shaped: list[dict[str, Any]], content_delivered: set[str]
) -> bool | None:
    """Every result stays content-bearing inline, a twin of an in-response
    bearer, or a session-duplicate of prior in-session content delivery."""

    if not shaped:
        return None
    for entry in shaped:
        if entry["delivery"] in ("full", "snippet", "twin_duplicate"):
            continue
        if entry["node"]["id"] not in content_delivered:
            return False
    return True


def _response_content_access_retained(
    result_ids: list[str],
    shaped: list[dict[str, Any]],
    content_delivered: set[str],
    dropped_twin_sources: set[str] | None = None,
) -> bool | None:
    """Whether all recorded results remain accessible from one response.

    Unlike ``_content_access_retained``, this helper also receives the
    unshaped result ids.  A surviving duplicate stub retains access through its
    ``content_ref``; repeat gating can remove a trailing run of those stubs
    entirely, and an omitted result is retained only when the same transport
    session already received its content.  This keeps aggregate retention from
    silently dropping compacted, non-empty events from its denominator.
    """

    if not result_ids:
        return None
    by_id = {entry["node"]["id"]: entry for entry in shaped}
    twin_sources = dropped_twin_sources or set()
    for node_id in result_ids:
        entry = by_id.get(node_id)
        if entry is None:
            if node_id not in content_delivered and node_id not in twin_sources:
                return False
            continue
        if entry["delivery"] in ("full", "snippet", "twin_duplicate"):
            continue
        if entry.get("has_content_ref"):
            continue
        if node_id not in content_delivered:
            return False
    return True


def _distribution(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"n": 0, "median": None, "p90": None, "mean": None}
    return {
        "n": len(ordered),
        "median": round(statistics.median(ordered), 1),
        "p90": round(percentile_nearest_rank(ordered, 0.90), 1),
        "mean": round(sum(ordered) / len(ordered), 2),
    }


def family_payload(corpus: ReplayCorpus, delivery: dict[str, dict[str, Any]]) -> dict[str, Any]:
    matched = [
        event
        for event in corpus.events
        if event.get("transcript_matched") and event.get("transcript_serialized_chars") is not None
    ]
    before = _distribution([e["transcript_serialized_chars"] for e in matched])
    after = _distribution([delivery[e["id"]]["chars"] for e in matched])
    unshaped = _distribution([delivery[e["id"]]["unshaped_chars"] for e in matched])
    after_all = _distribution([delivery[e["id"]]["chars"] for e in corpus.events])
    unshaped_all = _distribution([delivery[e["id"]]["unshaped_chars"] for e in corpus.events])

    def deltas(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        return {
            key: (
                round(a[key] - b[key], 2)
                if isinstance(a.get(key), (int, float)) and isinstance(b.get(key), (int, float))
                else None
            )
            for key in ("median", "p90", "mean")
        }

    feedback_matched = [e for e in matched if e.get("feedback_applied")]
    feedback_all = [e for e in corpus.events if e.get("feedback_applied")]

    return {
        "population": {
            "events_with_recorded_chars": before["n"],
            "all_split_events_replayed": after_all["n"],
            "note": (
                "before = recorded transcript tool-result chars; after = replayed"
                " current-code response chars on the same transcript-matched"
                " events; replayed_unshaped emulates the legacy full-delivery"
                " renderer on the reconstructed events, so unshaped-vs-before"
                " quantifies reconstruction bias (snapshot-time provenance/stats,"
                " envelope approximation, escaping) and after-vs-unshaped"
                " isolates the current shaping effect; *_all_events extend the"
                " replayed measures to every event in the split"
            ),
        },
        "before": before,
        "after": after,
        "replayed_unshaped": unshaped,
        "after_all_events": after_all,
        "replayed_unshaped_all_events": unshaped_all,
        "delta": deltas(after, before),
        "delta_shaping_only": deltas(after, unshaped),
        "retention": {
            "note": (
                "before = the legacy renderer, which ships every result's"
                " complete content inline, so both retention shares are 1.0"
                " there by construction; after = replayed current delivery."
                " top_result_content counts events whose top-ranked entry"
                " bears complete content inline or is a session-duplicate of"
                " a node whose content this session already delivered inline;"
                " useful_feedback counts feedback_applied events"
                " (" + FEEDBACK_APPLIED_CAVEAT + ") where every result stays"
                " content-bearing inline, a twin of an in-response bearer, or"
                " a session-duplicate with prior in-session content delivery."
                " Populations exclude zero-result events."
            ),
            "top_result_content": {
                "matched_events": _retention_shares(matched, delivery, "top_retained"),
                "all_events": _retention_shares(corpus.events, delivery, "top_retained"),
            },
            "useful_feedback": {
                "matched_events": _retention_shares(
                    feedback_matched, delivery, "content_access_retained"
                ),
                "all_events": _retention_shares(
                    feedback_all, delivery, "content_access_retained"
                ),
            },
        },
        "provenance_char_residuals_total": corpus.provenance_residuals,
    }


def _retention_shares(
    events: list[dict], delivery: dict[str, dict[str, Any]], flag: str
) -> dict[str, Any]:
    populated = [event for event in events if delivery[event["id"]][flag] is not None]
    retained = sum(1 for event in populated if delivery[event["id"]][flag])
    n = len(populated)
    if not n:
        return {"n": 0, "before": None, "after": None, "ratio": None}
    after_share = round(retained / n, 6)
    return {"n": n, "before": 1.0, "after": after_share, "ratio": after_share}


def family_auto_recall(corpus: ReplayCorpus, delivery: dict[str, dict[str, Any]]) -> dict[str, Any]:
    automatic = [
        event
        for event in corpus.events
        if event["class"] == "automatic" and event.get("transport_session_id")
    ]
    sessions = {event["transport_session_id"] for event in automatic}
    fingerprints_seen: dict[tuple[str, str], int] = defaultdict(int)
    repeated_fingerprint_events = 0
    deliveries = 0
    duplicate_deliveries = 0
    after_classes: dict[str, int] = defaultdict(int)
    after_duplicate_content = 0
    for event in automatic:
        key = (event["transport_session_id"], event.get("query_surrogate") or "")
        if fingerprints_seen[key]:
            repeated_fingerprint_events += 1
        fingerprints_seen[key] += 1
        replayed = delivery[event["id"]]
        repeats = set(replayed["repeat_ids"])
        deliveries += len(replayed["result_ids"])
        duplicate_deliveries += len(replayed["repeat_ids"])
        for node_id, delivery_class in zip(replayed["result_ids"], replayed["classes"]):
            after_classes[delivery_class] += 1
            if node_id in repeats and delivery_class in ("full", "snippet"):
                after_duplicate_content += 1
    return {
        "population": {
            "automatic_events_with_transport_session": len(automatic),
            "transport_sessions": len(sessions),
            "repeated_fingerprint_events": repeated_fingerprint_events,
            "note": (
                "duplicates = result node ids already delivered by an earlier"
                " event of the same transport session (200-event horizon,"
                " mirroring storage.delivered_node_ids); before ships every"
                " result full, after classifies repeats via current delivery"
            ),
        },
        "before": {
            "deliveries": deliveries,
            "duplicate_deliveries": duplicate_deliveries,
            "duplicate_share": round(duplicate_deliveries / deliveries, 6) if deliveries else None,
            "content_bearing_duplicates": duplicate_deliveries,
        },
        "after": {
            "deliveries": deliveries,
            "delivery_classes": dict(sorted(after_classes.items())),
            "duplicate_deliveries": duplicate_deliveries,
            "content_bearing_duplicates": after_duplicate_content,
            "content_bearing_duplicate_share": (
                round(after_duplicate_content / deliveries, 6) if deliveries else None
            ),
        },
        "delta": {
            "content_bearing_duplicates": after_duplicate_content - duplicate_deliveries,
        },
    }


def _frozen_dev_reference_contract() -> dict[str, Any]:
    """Return the hash/count binding for the original frozen dev split."""

    manifest_path = DEFAULT_CORPUS_ROOT.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - repository damage
        raise ValueError("the frozen dev manifest is unavailable or invalid") from exc

    hashes = find_manifest_hashes(manifest)["dev.jsonl"]
    if len(hashes) != 1:
        raise ValueError("the frozen dev manifest has no unique split hash")
    split_sha256 = next(iter(hashes))
    if sha256_file(DEFAULT_CORPUS_ROOT / "dev.jsonl") != split_sha256:
        raise ValueError("the frozen dev corpus is not hash-valid")
    try:
        automatic_events = int(manifest["corpus"]["counts"]["dev"]["by_class"]["automatic"])
    except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - repository damage
        raise ValueError("the frozen dev manifest has no automatic-event count") from exc
    _nodes, dev_events = load_split(DEFAULT_CORPUS_ROOT, "dev")
    automatic_identities = {
        hashlib.sha256(
            (
                " ".join(str(event.get("query_surrogate") or "").split())
                + "\n"
                + str(event.get("requested_scope") or event["scope"])
            ).encode("utf-8")
        ).hexdigest()
        for event in dev_events
        if event.get("class") == "automatic"
    }
    return {
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_split_sha256": split_sha256,
        "automatic_event_count": automatic_events,
        "unique_automatic_fingerprint_count": len(automatic_identities),
    }


def _manifest_file_sha256(manifest: Any, relative_path: str) -> str:
    """Read one exact file binding from a packet manifest."""

    if not isinstance(manifest, dict):
        raise ValueError("the packet manifest must be a JSON object")
    files = manifest.get("files")
    entry = files.get(relative_path) if isinstance(files, dict) else None
    digest = entry.get("sha256") if isinstance(entry, dict) else None
    if not isinstance(digest, str) or not _LOWER_HEX64.fullmatch(digest):
        raise ValueError("the packet manifest is missing a required file binding")
    return digest


def _load_packet_manifest(corpus_root: Path) -> dict[str, Any]:
    try:
        document = json.loads(
            (corpus_root.parent / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("the packet manifest is unavailable or invalid") from exc
    if not isinstance(document, dict):
        raise ValueError("the packet manifest must be a JSON object")
    return document


def _packet_verifier_is_hash_valid(
    packet_root: Path, manifest: dict[str, Any]
) -> bool:
    """Verify the independently reviewed verifier named by a sealed packet."""

    verifier = manifest.get("implementation", {}).get("verifier")
    if not isinstance(verifier, dict) or verifier.get("path") != "recipe/verify.py":
        return False
    expected = verifier.get("sha256")
    if not isinstance(expected, str) or not _LOWER_HEX64.fullmatch(expected):
        return False
    try:
        return sha256_file(packet_root / "recipe" / "verify.py") == expected
    except OSError:
        return False


def _require_exact_keys(
    value: dict[str, Any], expected: frozenset[str], description: str
) -> None:
    if frozenset(value) != expected:
        raise ValueError(f"the dev fingerprint index has an invalid {description} schema")


def load_dev_fingerprint_index(path: Path) -> dict[str, Any]:
    """Load a complete, privacy-safe cross-build dev identity index.

    Tokens are ephemeral-key HMAC-SHA-256 outputs over the production-normalized
    query/scope input.  The index itself is equality-only; semantic completeness,
    shared-key provenance, and target binding live in the separately generated
    sealed packet manifest and its independent keyed-validation receipt.  The
    loader deliberately returns no source path or token-derived digest suitable
    for later serialization.
    """

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError("the dev fingerprint index could not be read") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "the dev fingerprint index is invalid JSON"
            f" (line {exc.lineno}, column {exc.colno})"
        ) from None
    if not isinstance(document, dict):
        raise ValueError("the dev fingerprint index must be a JSON object")
    if document.get("schema_version") != 1:
        raise ValueError("the dev fingerprint index has an unsupported schema version")
    if document.get("kind") != DEV_FINGERPRINT_INDEX_KIND:
        raise ValueError("the dev fingerprint index has an unsupported kind")
    _require_exact_keys(
        document,
        frozenset(
            {
                "schema_version",
                "kind",
                "algorithm",
                "normalization",
                "population_events",
                "unique_tokens",
                "tokens",
            }
        ),
        "top-level",
    )
    tokens = document.get("tokens")
    if (
        document.get("algorithm") != "HMAC-SHA256"
        or document.get("normalization") != DEV_FINGERPRINT_NORMALIZATION
    ):
        raise ValueError("the dev fingerprint index has an unsupported opaque identity contract")

    automatic_count = document.get("population_events")
    unique_count = document.get("unique_tokens")
    if (
        type(automatic_count) is not int
        or type(unique_count) is not int
        or automatic_count < 0
        or unique_count < 0
        or automatic_count < unique_count
    ):
        raise ValueError("the dev fingerprint index has invalid aggregate counts")
    if not isinstance(tokens, list) or any(
        not isinstance(token, str) or not _LOWER_HEX64.fullmatch(token) for token in tokens
    ):
        raise ValueError("the dev fingerprint index contains malformed opaque identities")
    if len(tokens) != len(set(tokens)):
        raise ValueError("the dev fingerprint index contains duplicate opaque identities")
    if tokens != sorted(tokens):
        raise ValueError("the dev fingerprint index opaque identities are not sorted")
    if unique_count != len(tokens):
        raise ValueError("the dev fingerprint index aggregate counts do not match its identities")

    frozen = _frozen_dev_reference_contract()
    if (
        automatic_count != frozen["automatic_event_count"]
        or unique_count != frozen["unique_automatic_fingerprint_count"]
    ):
        raise ValueError("the dev fingerprint index is not bound to the frozen dev reference")
    return {
        "mode": "opaque",
        "identities": frozenset(tokens),
        "automatic_event_count": automatic_count,
    }


def dev_fingerprint_reference(
    corpus: ReplayCorpus,
    index_path: Path | None,
) -> dict[str, Any]:
    """Build the complete dev membership set and validate target identity mode."""

    lm = corpus.lm
    if index_path is not None:
        reference = load_dev_fingerprint_index(index_path)
        try:
            same_directory = index_path.resolve().parent == corpus.corpus_root.resolve()
        except OSError:
            same_directory = False
        if not same_directory:
            raise ValueError(
                "the dev fingerprint index and target corpus are not one sealed packet"
            )
        target_path = corpus.corpus_root / f"{corpus.split}.jsonl"
        target_sha256 = sha256_file(target_path)
        manifest = _load_packet_manifest(corpus.corpus_root)
        frozen = _frozen_dev_reference_contract()
        identity = manifest.get("identity", {})
        counts = manifest.get("counts", {})
        supplement_counts = counts.get("supplement", {})
        dev_counts = counts.get("frozen_dev", {})
        repeated_counts = counts.get("repeated_automatic", {})
        receipt = manifest.get("validation", {}).get("keyed_preseal", {})
        target_automatic = sum(
            event.get("class") == "automatic" for event in corpus.events
        )
        target_organic = sum(event.get("class") == "organic" for event in corpus.events)
        if (
            manifest.get("packet")
            != "animal-planet P6 replacement temporal supplement"
            or manifest.get("namespace") != "replacement-holdout"
            or manifest.get("frozen") is not True
            or manifest.get("semantic_reads") != 0
            or identity.get("algorithm") != "HMAC-SHA256"
            or identity.get("one_key_for_dev_and_supplement") is not True
            or identity.get("key_persisted") is not False
            or identity.get("unkeyed_query_fingerprints_persisted") is not False
            or identity.get("normalization") != DEV_FINGERPRINT_NORMALIZATION
            or manifest.get("sources", {})
            .get("original_packet_manifest", {})
            .get("sha256")
            != frozen["source_manifest_sha256"]
            or not _packet_verifier_is_hash_valid(corpus.corpus_root.parent, manifest)
            or receipt.get("schema_version") != 1
            or receipt.get("status") != "pass"
            or receipt.get("mode") != "keyed-preseal"
            or receipt.get("mismatches") != 0
            or receipt.get("token_collisions") != 0
            or receipt.get("semantic_reads") != 0
            or receipt.get("supplement_events") != len(corpus.events)
            or receipt.get("automatic_events") != target_automatic
            or receipt.get("organic_events") != target_organic
            or receipt.get("dev_automatic_events")
            != reference["automatic_event_count"]
            or receipt.get("dev_unique_tokens") != len(reference["identities"])
            or receipt.get("repeated_automatic_families")
            != repeated_counts.get("families")
            or receipt.get("repeated_automatic_events")
            != repeated_counts.get("events")
            or receipt.get("unseen_in_dev_families")
            != repeated_counts.get("unseen_in_dev_families")
            or receipt.get("unseen_in_dev_events")
            != repeated_counts.get("unseen_in_dev_events")
            or manifest.get("validation", {}).get(
                "candidate_unchanged_after_validation"
            )
            is not True
            or supplement_counts.get("events") != len(corpus.events)
            or supplement_counts.get("automatic") != target_automatic
            or supplement_counts.get("organic") != target_organic
            or dev_counts.get("automatic") != reference["automatic_event_count"]
            or dev_counts.get("unique_automatic_tokens")
            != len(reference["identities"])
        ):
            raise ValueError("the replacement packet lacks a valid sealed receipt")
        if (
            _manifest_file_sha256(
                manifest, f"corpus/{corpus.split}.jsonl"
            )
            != target_sha256
            or _manifest_file_sha256(
                manifest, "corpus/dev-fingerprint-index.json"
            )
            != sha256_file(index_path)
        ):
            raise ValueError("the target corpus or dev index is not hash-valid")
        invalid = sum(
            1
            for event in corpus.events
            if not isinstance(event.get("fingerprint_token"), str)
            or not _LOWER_HEX64.fullmatch(event["fingerprint_token"])
        )
        if invalid:
            raise ValueError(
                "the target corpus does not have complete valid opaque identity coverage"
                f" ({invalid} events)"
            )
        return reference

    if any("fingerprint_token" in event for event in corpus.events):
        raise ValueError(
            "the target corpus carries opaque identities but no dev fingerprint index"
        )
    try:
        _nodes, events = load_split(corpus.corpus_root, "dev")
    except OSError:
        return {
            "mode": "native",
            "identities": None,
            "automatic_event_count": None,
            "unavailable_reason": "complete_dev_reference_not_attested",
        }
    if any("fingerprint_token" in event for event in events):
        raise ValueError("the dev split mixes opaque and native fingerprint identities")
    automatic = [event for event in events if event.get("class") == "automatic"]
    try:
        manifest = _load_packet_manifest(corpus.corpus_root)
        attested_dev_sha256 = _manifest_file_sha256(manifest, "corpus/dev.jsonl")
    except ValueError:
        return {
            "mode": "native",
            "identities": None,
            "automatic_event_count": None,
            "unavailable_reason": "complete_dev_reference_not_attested",
        }
    dev_path = corpus.corpus_root / "dev.jsonl"
    if attested_dev_sha256 != sha256_file(dev_path):
        raise ValueError("the dev reference is not hash-valid")
    try:
        attested_automatic = manifest["corpus"]["counts"]["dev"]["by_class"][
            "automatic"
        ]
    except (KeyError, TypeError) as exc:
        return {
            "mode": "native",
            "identities": None,
            "automatic_event_count": None,
            "unavailable_reason": "complete_dev_reference_not_attested",
        }
    if type(attested_automatic) is not int or attested_automatic != len(automatic):
        raise ValueError("the dev reference automatic population is incomplete")
    identities = {
        lm["recall_fingerprint"](
            event.get("query_surrogate") or "",
            event.get("requested_scope") or event["scope"],
        )
        for event in automatic
    }
    return {
        "mode": "native",
        "identities": frozenset(identities),
        "automatic_event_count": attested_automatic,
    }


def _event_repeat_identity(
    event: dict[str, Any], mode: str, lm: dict[str, Any]
) -> tuple[str, str, str]:
    """Return (family identity, store-only query, store requested scope)."""

    if mode == "opaque":
        token = event["fingerprint_token"]
        return token, token, OPAQUE_REPLAY_SCOPE
    query = event.get("query_surrogate") or ""
    requested_scope = event.get("requested_scope") or event["scope"]
    return lm["recall_fingerprint"](query, requested_scope), query, requested_scope


def family_repeat_gating(
    corpus: ReplayCorpus,
    delivery: dict[str, dict[str, Any]],
    *,
    dev_fingerprint_index: Path | None = None,
) -> dict[str, Any]:
    """Replay the server's online repeat-gating rule through a real store.

    Drives an in-memory ``MemoryStore`` through every split event in
    (created_at, id) order, mirroring the ``memory_recall`` handler: before
    each event is recorded, the gate decision comes from
    ``should_gate_fingerprint`` over the store's accumulated per-fingerprint
    signal. Native corpora use production ``recall_fingerprint`` identity.
    Cross-build corpora use an opaque HMAC token as the store-only query under
    a fixed sentinel scope; their independently de-identified query surrogate
    remains in the response envelope, so privacy identity never changes the
    payload measurement or enters output.
    Each event is shaped twice — ungated (the ``replay_delivery`` outcome,
    reused untouched) and gated (when the rule fires, the fingerprint's
    delivered ids union into ``already_delivered_ids`` and ``session_dedup``
    is forced on, exactly like the handler). The event is then recorded
    (``record_recall_event`` with the event's query/scope/transport/session
    surrogates and result node ids), and gated events are flagged post-record.
    Feedback links are interleaved at ``feedback_applied_at`` (deliveries
    before links on ties), matching storage's authoritative aggregate rebuild.
    The store never sees corpus class/template labels; those labels and dev
    absence slice aggregate measurement only after the full replay.
    """

    lm = corpus.lm
    reference = dev_fingerprint_reference(corpus, dev_fingerprint_index)
    identity_mode = reference["mode"]
    policy = lm["FingerprintGatePolicy"].from_env()
    shape = lm["shape_recall_results"]
    snippet_chars = lm["snippet_max_chars_from_env"]()
    context_chars = lm["context_value_max_chars_from_env"]()
    snippet_ladder = lm["snippet_ladder_from_env"]()
    full_node_diet = lm["full_node_diet_enabled_from_env"]()
    provenance_chars = lm["provenance_value_max_chars_from_env"]()
    stats_compaction = lm["stats_compaction_enabled_from_env"]()
    sparse_entries = lm["sparse_entries_enabled_from_env"]()

    gated_chars: dict[str, int] = {}
    gated_retained: dict[str, bool | None] = {}
    gated_event_ids: set[str] = set()
    event_identities: dict[str, str] = {}

    store = lm["MemoryStore"](":memory:")
    try:
        anchor = store.create_node(
            level="trace", content="repeat-gating replay feedback anchor"
        )
        session_history: dict[Any, list[set[str]]] = defaultdict(list)
        session_content: dict[Any, set[str]] = defaultdict(set)
        pending_feedback: list[tuple[str, int, str]] = []
        for index, event in enumerate(corpus.events):
            created_at = event["created_at"]
            # The aggregate rebuild orders all deliveries before feedback on
            # an equal timestamp, so only strictly earlier links are visible
            # to this pre-delivery gate decision.
            while pending_feedback and pending_feedback[0][0] < created_at:
                _at, _order, recorded_id = heapq.heappop(pending_feedback)
                store.mark_recall_event_feedback(recorded_id, anchor.id)

            transport = event.get("transport_session_id")
            session_key: Any = transport if transport else ("solo", index)
            history = session_history[session_key]
            delivered: set[str] = set()
            for previous in history[-DELIVERED_HORIZON_EVENTS:]:
                delivered |= previous

            family_identity, store_query, store_scope = _event_repeat_identity(
                event, identity_mode, lm
            )
            event_identities[event["id"]] = family_identity
            # Stats are read BEFORE this event is recorded, so the decision
            # sees only the accounting accumulated by prior events — the same
            # ordering as the handler.
            fingerprint = lm["recall_fingerprint"](store_query, store_scope)
            gated = policy.enabled and lm["should_gate_fingerprint"](
                store.get_recall_fingerprint_stats(fingerprint), policy
            )

            replayed = delivery[event["id"]]
            result_ids = replayed["result_ids"]
            if gated:
                gated_event_ids.add(event["id"])
                shaped = shape(
                    corpus.recall_results(event),
                    already_delivered_ids=(
                        delivered | store.fingerprint_delivered_node_ids(fingerprint)
                    ),
                    snippet_max_chars=snippet_chars,
                    context_value_max_chars=context_chars,
                    session_dedup=True,
                    snippet_ladder=snippet_ladder,
                    full_node_diet=full_node_diet,
                    provenance_value_max_chars=provenance_chars,
                    stats_compaction=stats_compaction,
                    sparse_entries=sparse_entries,
                )
                dropped_twin_sources: set[str] = set()
                if policy.drop_trailing_stubs:
                    # Mirror the handler: a gated delivery drops its trailing
                    # all-stub run (reduced-result-count compaction).
                    while shaped and shaped[-1]["delivery"] in (
                        "session_duplicate",
                        "twin_duplicate",
                    ):
                        dropped = shaped.pop()
                        duplicate_of = (dropped.get("content_ref") or {}).get(
                            "duplicate_of"
                        )
                        if (
                            dropped["delivery"] == "twin_duplicate"
                            and isinstance(duplicate_of, str)
                            and any(
                                item["node"]["id"] == duplicate_of
                                and (
                                    item["delivery"]
                                    in ("full", "snippet", "twin_duplicate")
                                    or "content_ref" in item
                                )
                                for item in shaped
                            )
                        ):
                            dropped_twin_sources.add(dropped["node"]["id"])
                envelope = {
                    "query": event.get("query_surrogate") or "",
                    "scope": event.get("requested_scope") or event.get("scope"),
                    "recall_event_id": event["id"],
                    "count": len(shaped),
                    "results": shaped,
                    "auto_decay": None,
                }
                chars = len(json.dumps(envelope, ensure_ascii=False))
                entries = [
                    {
                        "delivery": item["delivery"],
                        "node": {"id": item["node"]["id"]},
                        "has_content_ref": "content_ref" in item,
                    }
                    for item in shaped
                ]
            else:
                # Rule idle: shaping inputs equal the ungated replay's (the
                # recorded results, and with them the session horizon, are
                # identical in both worlds), so its outcome is reused as-is.
                chars = replayed["chars"]
                entries = [
                    {"delivery": delivery_class, "node": {"id": node_id}}
                    for node_id, delivery_class in zip(result_ids, replayed["classes"])
                ]
            gated_chars[event["id"]] = chars
            gated_retained[event["id"]] = (
                _response_content_access_retained(
                    result_ids,
                    entries,
                    session_content[session_key],
                    dropped_twin_sources,
                )
                if gated
                else replayed["content_access_retained"]
            )
            session_content[session_key].update(
                item["node"]["id"]
                for item in entries
                if item["delivery"] in ("full", "snippet")
            )
            history.append(set(result_ids))

            recorded = store.record_recall_event(
                query=store_query,
                scope=event["scope"] if identity_mode == "native" else store_scope,
                requested_scope=store_scope,
                resolved_scopes=(
                    event.get("resolved_scopes") or [event["scope"]]
                    if identity_mode == "native"
                    else [store_scope]
                ),
                ambient_context=(
                    {"transport_session_id": transport} if transport is not None else {}
                ),
                depth=event.get("depth"),
                max_results=int(event.get("max_results") or 10),
                results=[{"node_id": node_id} for node_id in result_ids],
            )
            if gated:
                store.mark_recall_event_gated(recorded.id)
            if event.get("feedback_applied"):
                applied_at = str(event.get("feedback_applied_at") or created_at)
                heapq.heappush(
                    pending_feedback,
                    (max(applied_at, created_at), index, recorded.id),
                )
    finally:
        store.close()

    # --- measurement (corpus class labels slice only from here on) --------
    automatic = [event for event in corpus.events if event["class"] == "automatic"]
    organic = [event for event in corpus.events if event["class"] == "organic"]
    by_key: dict[str, list[dict]] = defaultdict(list)
    for event in automatic:
        by_key[event_identities[event["id"]]].append(event)
    repeated_keys = {
        key: events
        for key, events in by_key.items()
        if len(events) >= 3
        and len({event.get("transport_session_id") for event in events}) >= 2
    }
    repeated_events = [event for events in repeated_keys.values() for event in events]
    repeated_ids = {event["id"] for event in repeated_events}
    nonrepeated = [event for event in automatic if event["id"] not in repeated_ids]
    reference_identities = reference["identities"]
    unseen_keys = (
        {
            key: events
            for key, events in repeated_keys.items()
            if key not in reference_identities
        }
        if reference_identities is not None
        else {}
    )
    unseen_events = [event for events in unseen_keys.values() for event in events]
    unseen_ids = {event["id"] for event in unseen_events}

    def char_totals(events: list[dict]) -> tuple[int, int, list[int], list[int]]:
        ungated = [delivery[event["id"]]["chars"] for event in events]
        gated = [gated_chars[event["id"]] for event in events]
        return sum(ungated), sum(gated), ungated, gated

    rep_ungated_total, rep_gated_total, rep_ungated, rep_gated = char_totals(repeated_events)
    org_ungated_total, org_gated_total, _, _ = char_totals(organic)
    non_ungated_total, non_gated_total, _, _ = char_totals(nonrepeated)
    unseen_ungated_total, unseen_gated_total, _unseen_ungated, _unseen_gated = char_totals(
        unseen_events
    )

    def relative_delta_pct(ungated_total: float | None, gated_total: float | None) -> float | None:
        if not ungated_total or gated_total is None:
            return None
        return round(100.0 * (gated_total - ungated_total) / ungated_total, 4)

    organic_populated = [
        event
        for event in organic
        if delivery[event["id"]]["content_access_retained"] is not None
    ]
    ungated_share = gated_share = None
    if organic_populated:
        ungated_share = round(
            sum(
                1
                for event in organic_populated
                if delivery[event["id"]]["content_access_retained"]
            )
            / len(organic_populated),
            6,
        )
        gated_share = round(
            sum(1 for event in organic_populated if gated_retained[event["id"]])
            / len(organic_populated),
            6,
        )

    unseen_populated = [
        event
        for event in unseen_events
        if delivery[event["id"]]["content_access_retained"] is not None
    ]
    unseen_ungated_retained = sum(
        1
        for event in unseen_populated
        if delivery[event["id"]]["content_access_retained"]
    )
    unseen_gated_retained = sum(
        1 for event in unseen_populated if gated_retained[event["id"]]
    )
    unseen_ungated_share = unseen_gated_share = None
    if unseen_populated:
        unseen_ungated_share = round(
            unseen_ungated_retained / len(unseen_populated), 6
        )
        unseen_gated_share = round(unseen_gated_retained / len(unseen_populated), 6)

    if reference_identities is None:
        # Preserve legacy/custom native compare behavior without treating an
        # unattested or absent dev population as evidence of novelty.
        unseen_in_dev = {
            "available": False,
            "reason": reference["unavailable_reason"],
        }
    else:
        unseen_in_dev = {
            "reference_split": "dev",
            "population": {
                "automatic_events_in_dev": reference["automatic_event_count"],
                "automatic_fingerprints_in_dev": len(reference_identities),
                "repeated_fingerprints": len(unseen_keys),
                "repeated_fingerprint_events": len(unseen_events),
                "gated_events": len(gated_event_ids & unseen_ids),
            },
            "chars": {
                "repeated_automatic": {
                    "ungated_total": unseen_ungated_total,
                    "gated_total": unseen_gated_total,
                    "reduction_ratio": (
                        round(1.0 - unseen_gated_total / unseen_ungated_total, 6)
                        if unseen_ungated_total
                        else None
                    ),
                }
            },
            "retention": {
                "repeated_automatic_content_access": {
                    "events": len(unseen_populated),
                    "ungated_retained": unseen_ungated_retained,
                    "gated_retained": unseen_gated_retained,
                    "ungated_share": unseen_ungated_share,
                    "gated_share": unseen_gated_share,
                    "delta_pct": relative_delta_pct(
                        unseen_ungated_share, unseen_gated_share
                    ),
                }
            },
        }

    return {
        "policy": {
            "enabled": policy.enabled,
            "min_unlinked": policy.min_unlinked,
            "max_link_rate": policy.max_link_rate,
            "min_sessions": policy.min_sessions,
            "probe_every": policy.probe_every,
            "drop_trailing_stubs": policy.drop_trailing_stubs,
        },
        "population": {
            "repeated_fingerprints": len(repeated_keys),
            "repeated_fingerprint_events": len(repeated_events),
            "gated_events": len(gated_event_ids),
            "note": (
                "a repeated fingerprint is a production-normalized native or"
                " opaque cross-build identity with >=3 automatic events in the"
                " split (>=2 distinct transport sessions); gated_events counts replay"
                " events where the class-blind online rule fired. chars cover"
                " ALL events of repeated fingerprints — warmup deliveries made"
                " before the rule can fire dilute the reduction honestly"
            ),
        },
        "unseen_in_dev": unseen_in_dev,
        "chars": {
            "repeated_automatic": {
                "ungated_total": rep_ungated_total,
                "gated_total": rep_gated_total,
                "reduction_ratio": (
                    round(1.0 - rep_gated_total / rep_ungated_total, 6)
                    if rep_ungated_total
                    else None
                ),
                "ungated_median": (
                    round(statistics.median(rep_ungated), 1) if rep_ungated else None
                ),
                "gated_median": (
                    round(statistics.median(rep_gated), 1) if rep_gated else None
                ),
            },
            "organic": {
                "ungated_total": org_ungated_total,
                "gated_total": org_gated_total,
                "delta_pct": relative_delta_pct(org_ungated_total, org_gated_total),
            },
            "automatic_nonrepeated": {
                "delta_pct": relative_delta_pct(non_ungated_total, non_gated_total),
            },
        },
        "retention": {
            "organic_content_access": {
                "ungated_share": ungated_share,
                "gated_share": gated_share,
                "delta_pct": relative_delta_pct(ungated_share, gated_share),
            },
        },
    }


def build_rerank_service(corpus: ReplayCorpus) -> Any:
    lm = corpus.lm
    resolver = lm["floor_default_weights"](lm["MemoryConfig"](), lm["assumed_evidence"])
    pairs = sorted({(source, target) for (source, target, _created) in corpus.edges})
    return lm["make_ranking_service"](resolver, pairs)


def rerank_event(corpus: ReplayCorpus, service: Any, event: dict) -> list[str]:
    lm = corpus.lm
    candidates: dict[str, Any] = {}
    for raw in sorted(event["results"], key=lambda c: c.get("rank") or 0):
        node = corpus.node(raw["node_id"])
        candidates[raw["node_id"]] = lm["_Candidate"](
            node=node,
            bm25_score=float(raw.get("bm25_score") or 0.0),
            vector_score=float(raw.get("vector_score") or 0.0),
            graph_score=float(raw.get("graph_score") or 0.0),
            trigger_score=float(raw.get("trigger_score") or 0.0),
            path=tuple(raw.get("path") or ()),
        )
    # Surrogate queries cannot carry causal markers; mode comes from the
    # recorded depth enum alone (documented fidelity limit).
    _graph_depth, causal_mode = lm["_parse_depth"](event.get("depth"), "")
    decision_mode = lm["_is_decision_depth"](event.get("depth"))
    ranked = service.rank_candidates(
        candidates,
        corpus.plan(event),
        causal_mode=causal_mode,
        decision_mode=decision_mode,
    )
    limited = ranked[: int(event.get("max_results") or len(ranked))]
    return [result.node.id for result in limited]


def family_cross_scope(corpus: ReplayCorpus, reranks: dict[str, list[str]]) -> dict[str, Any]:
    before_total = before_cross = 0
    after_total = after_cross = 0
    events_with_cross_before = events_with_cross_after = 0
    for event in corpus.events:
        requested = event.get("requested_scope") or event["scope"]
        recorded = [c["node_id"] for c in sorted(event["results"], key=lambda c: c.get("rank") or 0)]
        scopes = {c["node_id"]: c.get("scope") for c in event["results"]}
        cross_before = [nid for nid in recorded if scopes.get(nid) != requested]
        before_total += len(recorded)
        before_cross += len(cross_before)
        events_with_cross_before += bool(cross_before)
        after_ids = reranks[event["id"]]
        cross_after = [nid for nid in after_ids if scopes.get(nid) != requested]
        after_total += len(after_ids)
        after_cross += len(cross_after)
        events_with_cross_after += bool(cross_after)
    return {
        "population": {
            "events": len(corpus.events),
            "note": (
                "cross-scope = result node scope differs from the event's"
                " requested scope; after = current rank_candidates over the"
                " recorded candidates (snapshot node stats, corpus supersedes"
                " pairs, current default weights), truncated to max_results"
            ),
        },
        "before": {
            "results": before_total,
            "cross_scope_results": before_cross,
            "cross_scope_share": round(before_cross / before_total, 6) if before_total else None,
            "events_with_cross_scope": events_with_cross_before,
        },
        "after": {
            "results": after_total,
            "cross_scope_results": after_cross,
            "cross_scope_share": round(after_cross / after_total, 6) if after_total else None,
            "events_with_cross_scope": events_with_cross_after,
            "results_dropped_by_rerank": before_total - after_total,
        },
        "delta": {
            "cross_scope_results": after_cross - before_cross,
            "cross_scope_share": (
                round(after_cross / after_total - before_cross / before_total, 6)
                if before_total and after_total
                else None
            ),
        },
    }


def family_correction_dominance(
    corpus: ReplayCorpus, reranks: dict[str, list[str]]
) -> dict[str, Any]:
    pair_by_nodes = {(source, target) for (source, target, _created) in corpus.edges}
    superseded_ids = {target for (_source, target) in pair_by_nodes}
    events_with_pairs = 0
    pairs_total = 0
    before_violations = 0
    after_violations = 0
    corrections_dropped = 0
    stale_dropped = 0
    before_results = after_results = 0
    before_superseded = after_superseded = 0
    before_superseded_top1 = after_superseded_top1 = 0
    for event in corpus.events:
        recorded = sorted(event["results"], key=lambda c: c.get("rank") or 0)
        position_before = {c["node_id"]: index for index, c in enumerate(recorded)}
        recorded_ids = [c["node_id"] for c in recorded]
        after_ids = reranks[event["id"]]

        before_results += len(recorded_ids)
        after_results += len(after_ids)
        before_superseded += sum(1 for nid in recorded_ids if nid in superseded_ids)
        after_superseded += sum(1 for nid in after_ids if nid in superseded_ids)
        before_superseded_top1 += bool(recorded_ids and recorded_ids[0] in superseded_ids)
        after_superseded_top1 += bool(after_ids and after_ids[0] in superseded_ids)

        present = set(position_before)
        event_pairs = [
            (superseding, superseded)
            for (superseding, superseded) in pair_by_nodes
            if superseding in present and superseded in present
        ]
        if not event_pairs:
            continue
        events_with_pairs += 1
        pairs_total += len(event_pairs)
        position_after = {nid: index for index, nid in enumerate(after_ids)}
        for superseding, superseded in sorted(event_pairs):
            if position_before[superseded] < position_before[superseding]:
                before_violations += 1
            stale_pos = position_after.get(superseded)
            corr_pos = position_after.get(superseding)
            if stale_pos is None:
                stale_dropped += 1
            elif corr_pos is None:
                corrections_dropped += 1
            elif stale_pos < corr_pos:
                after_violations += 1
    return {
        "population": {
            "events": len(corpus.events),
            "events_with_pairs": events_with_pairs,
            "pairs": pairs_total,
            "note": (
                "pairs = supersedes edges (correction supersedes stale) with both"
                " endpoints among one event's recorded candidates; violation ="
                " stale ranked above its correction. Supersedes edges are"
                " snapshot-time and mostly post-date the recorded recalls, so the"
                " pairs population can legitimately be 0; superseded_deliveries"
                " tracks the stale-delivery pressure regardless (results whose"
                " node is a supersedes target, i.e. known-superseded at snapshot"
                " time)"
            ),
        },
        "before": {
            "violations": before_violations,
            "violation_rate": round(before_violations / pairs_total, 6) if pairs_total else None,
            "superseded_deliveries": before_superseded,
            "superseded_share": (
                round(before_superseded / before_results, 6) if before_results else None
            ),
            "superseded_top1_events": before_superseded_top1,
        },
        "after": {
            "violations": after_violations,
            "violation_rate": round(after_violations / pairs_total, 6) if pairs_total else None,
            "corrections_dropped": corrections_dropped,
            "stale_dropped": stale_dropped,
            "superseded_deliveries": after_superseded,
            "superseded_share": (
                round(after_superseded / after_results, 6) if after_results else None
            ),
            "superseded_top1_events": after_superseded_top1,
        },
        "delta": {
            "violations": after_violations - before_violations,
            "superseded_deliveries": after_superseded - before_superseded,
            "superseded_top1_events": after_superseded_top1 - before_superseded_top1,
        },
    }


def cmd_compare(args: argparse.Namespace) -> int:
    requested = [name.strip() for name in args.metrics.split(",") if name.strip()]
    unknown = [name for name in requested if name not in METRIC_FAMILIES]
    if unknown:
        print(f"compare: unknown metric families: {', '.join(unknown)}", file=sys.stderr)
        return 2
    privacy_safe_cross_build = args.dev_fingerprint_index is not None
    if privacy_safe_cross_build and set(requested) != {"auto_recall"}:
        print(
            "compare: --dev-fingerprint-index requires only the auto_recall metric",
            file=sys.stderr,
        )
        return 2
    if args.split == "holdout":
        print(
            "compare: WARNING — holdout is sealed during development"
            " (corpus/POLICY.md); this run is only legitimate post-freeze.",
            file=sys.stderr,
        )

    lm = _import_living_memory()
    try:
        corpus = ReplayCorpus(args.corpus_root, args.split, lm)
    except Exception:
        if not privacy_safe_cross_build:
            raise
        print(
            "compare: the sealed cross-build corpus failed privacy-safe validation",
            file=sys.stderr,
        )
        return 2

    output: dict[str, Any] = {
        "split": args.split,
        "corpus_root": (
            "sealed-cross-build-packet"
            if privacy_safe_cross_build
            else str(args.corpus_root)
        ),
        "delivery_defaults": {
            "snippet_max_chars": lm["snippet_max_chars_from_env"](),
            "context_value_max_chars": lm["context_value_max_chars_from_env"](),
            "session_dedup": lm["session_dedup_enabled_from_env"](),
            "snippet_ladder": (
                list(ladder) if (ladder := lm["snippet_ladder_from_env"]()) is not None else None
            ),
            "full_node_diet": lm["full_node_diet_enabled_from_env"](),
            "provenance_value_max_chars": lm["provenance_value_max_chars_from_env"](),
            "stats_compaction": lm["stats_compaction_enabled_from_env"](),
            "sparse_entries": lm["sparse_entries_enabled_from_env"](),
        },
        "metrics": {},
        "fidelity_limits": FIDELITY_LIMITS,
    }

    delivery = None
    if "payload" in requested or "auto_recall" in requested:
        try:
            delivery = replay_delivery(corpus)
        except Exception:
            if not privacy_safe_cross_build:
                raise
            print(
                "compare: the sealed cross-build corpus failed privacy-safe validation",
                file=sys.stderr,
            )
            return 2
    reranks = None
    if "cross_scope" in requested or "correction_dominance" in requested:
        service = build_rerank_service(corpus)
        reranks = {event["id"]: rerank_event(corpus, service, event) for event in corpus.events}

    if "payload" in requested:
        output["metrics"]["payload"] = family_payload(corpus, delivery)
    if "auto_recall" in requested:
        try:
            auto_recall = family_auto_recall(corpus, delivery)
            # Additive section: the online repeat-gating replay (real MemoryStore).
            auto_recall["repeat_gating"] = family_repeat_gating(
                corpus,
                delivery,
                dev_fingerprint_index=args.dev_fingerprint_index,
            )
        except Exception as exc:
            if not privacy_safe_cross_build and not isinstance(exc, ValueError):
                raise
            message = (
                "the sealed cross-build packet failed privacy-safe validation"
                if privacy_safe_cross_build
                else str(exc)
            )
            print(f"compare: {message}", file=sys.stderr)
            return 2
        output["metrics"]["auto_recall"] = auto_recall
    if "cross_scope" in requested:
        output["metrics"]["cross_scope"] = family_cross_scope(corpus, reranks)
    if "correction_dominance" in requested:
        output["metrics"]["correction_dominance"] = family_correction_dominance(corpus, reranks)

    print(dump_json(output), end="")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 scripts/ap_baseline.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_corpus_root(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--corpus-root",
            type=Path,
            default=DEFAULT_CORPUS_ROOT,
            help="corpus directory (default: artifacts/animal-planet/corpus;"
            " override for tamper tests)",
        )

    report = subparsers.add_parser(
        "report", help="compute the frozen baseline and write baseline-report.json"
    )
    add_corpus_root(report)
    report.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default: <corpus-root>/../baseline-report.json)",
    )
    report.set_defaults(func=cmd_report)

    verify = subparsers.add_parser(
        "verify",
        help="recompute metrics, compare against baseline-report.json + cited"
        " tolerances, check corpus hashes against manifest.json",
    )
    add_corpus_root(verify)
    verify.set_defaults(func=cmd_verify)

    compare = subparsers.add_parser(
        "compare",
        help="before/after JSON for one split via current delivery shaping and"
        " a recorded-candidate rerank",
        description="See the module docstring (--help of the program) for"
        " replay fidelity limits.",
    )
    add_corpus_root(compare)
    compare.add_argument("--split", choices=SPLIT_NAMES, required=True)
    compare.add_argument(
        "--metrics",
        default=",".join(METRIC_FAMILIES),
        help="comma-separated subset of: " + ", ".join(METRIC_FAMILIES),
    )
    compare.add_argument(
        "--dev-fingerprint-index",
        type=Path,
        default=None,
        help=(
            "independently validated opaque dev identity index, hash-bound with"
            " its target by <corpus-root>/../manifest.json; omitted uses the"
            " manifest-attested automatic identities in <corpus-root>/dev.jsonl"
        ),
    )
    compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
