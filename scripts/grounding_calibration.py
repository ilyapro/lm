#!/usr/bin/env python3
"""Calibrate the live grounding gate against the offline replay label.

The replay harness grades containment with an IDF built over the *whole*
labeled corpus. The live write path (``feedback.apply_pending_recall_feedback``)
cannot: when a ``memory_remember`` consumes a pending recall event it holds
only the consuming trace and that event's result nodes, so it builds a
per-event IDF from those documents alone.

Both views run the same arithmetic (``living_memory.grounding``); only the
corpus differs. This script quantifies that difference on recorded history so
the shared ``min_containment`` default is a measured choice rather than an
assumption: it sweeps the live threshold, and at each value reports precision,
recall and F1 of the live verdict against the whole-corpus verdict at the
replay default 0.25.

Read the output as agreement between two proxies, not as accuracy against
truth: neither view is ground truth for "the agent used this result".

Usage::

    python3 scripts/grounding_calibration.py \
        --db ~/.cache/living-memory-harness/snapshot.sqlite3 \
        --report artifacts/grounding/calibration.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from living_memory.grounding import (  # noqa: E402
    DEFAULT_MIN_CONTAINMENT,
    build_idf,
    containment,
    ground_token_sets,
)
from living_memory.replay import (  # noqa: E402
    build_label_data,
    load_replay_events,
    open_readonly,
)

SWEEP = [round(0.05 + 0.005 * step, 3) for step in range(71)]


def _rates(pairs: list[tuple[float, bool]], threshold: float) -> dict[str, Any]:
    true_positive = false_positive = false_negative = true_negative = 0
    for live_value, reference in pairs:
        predicted = live_value >= threshold
        if predicted and reference:
            true_positive += 1
        elif predicted and not reference:
            false_positive += 1
        elif not predicted and reference:
            false_negative += 1
        else:
            true_negative += 1
    total = len(pairs) or 1
    precision = true_positive / (true_positive + false_positive or 1)
    recall = true_positive / (true_positive + false_negative or 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "positives": true_positive + false_positive,
        "positive_rate": round((true_positive + false_positive) / total, 6),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "agreement": round((true_positive + true_negative) / total, 6),
    }


def calibrate(db_path: str, *, reference_threshold: float) -> dict[str, Any]:
    connection = open_readonly(db_path)
    try:
        events, load_stats = load_replay_events(connection)
        data = build_label_data(connection, events)
    finally:
        connection.close()

    labeled = [event for event in events if event.labeled]
    pairs: list[tuple[float, bool]] = []
    live_values: list[float] = []
    corpus_values: list[float] = []
    graded_events = 0

    for event in labeled:
        trace_tokens = data.token_sets.get(event.feedback_trace_id or "")
        if not trace_tokens:
            continue
        result_tokens = {
            result.node_id: tokens
            for result in event.results
            if (tokens := data.token_sets.get(result.node_id)) is not None
        }
        if not result_tokens:
            continue
        graded_events += 1
        # Live view: per-event IDF, exactly what the write path can build.
        verdicts = ground_token_sets(trace_tokens, result_tokens, min_containment=1e9)
        for node_id, tokens in result_tokens.items():
            live = verdicts[node_id].containment
            # Reference view: whole-corpus IDF, what the harness labels with.
            corpus = containment(tokens, trace_tokens, data.idf)
            pairs.append((live, corpus >= reference_threshold))
            live_values.append(live)
            corpus_values.append(corpus)

    sweep = [_rates(pairs, threshold) for threshold in SWEEP]
    best_f1 = max(sweep, key=lambda row: row["f1"])
    at_default = next(
        row for row in sweep if abs(row["threshold"] - DEFAULT_MIN_CONTAINMENT) < 1e-9
    )
    reference_positive = sum(1 for _value, ref in pairs if ref)

    return {
        "db_path": str(db_path),
        "corpus": {
            **load_stats,
            "labeled_events": len(labeled),
            "graded_events": graded_events,
            "graded_pairs": len(pairs),
            "corpus_documents": len(data.token_sets),
        },
        "reference": {
            "view": "whole-corpus IDF containment (replay label)",
            "threshold": reference_threshold,
            "positives": reference_positive,
            "positive_rate": round(reference_positive / (len(pairs) or 1), 6),
        },
        "correlation": {
            "pearson_live_vs_corpus": round(
                statistics.correlation(live_values, corpus_values), 6
            ),
            "mean_live": round(statistics.fmean(live_values), 6),
            "mean_corpus": round(statistics.fmean(corpus_values), 6),
        },
        "sweep": sweep,
        "at_default": at_default,
        "best_f1": best_f1,
        "adopted": {
            "min_containment": DEFAULT_MIN_CONTAINMENT,
            "rationale": (
                "Held at the replay default. The live view at 0.25 is the "
                "precision-favouring point (precision "
                f"{at_default['precision']:.3f}, recall {at_default['recall']:.3f}); "
                f"F1 peaks at {best_f1['threshold']} (precision "
                f"{best_f1['precision']:.3f}, recall {best_f1['recall']:.3f}). For a "
                "credit rule a false positive is the defect being fixed — noise "
                "re-entering the learned signal — while a false negative only "
                "withholds a signal that recurs on the next consumption, so the "
                "asymmetry favours precision and no recalibration was adopted."
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    corpus = report["corpus"]
    reference = report["reference"]
    adopted = report["adopted"]
    lines = [
        "# Live grounding calibration",
        "",
        f"DB: `{report['db_path']}`",
        "",
        f"- {corpus['graded_pairs']} result/trace pairs over {corpus['graded_events']} "
        f"consumed events ({corpus['labeled_events']} labeled of "
        f"{corpus['usable_events']} usable, {corpus['corpus_documents']} corpus documents)",
        f"- Reference (whole-corpus IDF at {reference['threshold']}): "
        f"{reference['positives']} positives ({reference['positive_rate']:.4f} of pairs)",
        f"- Pearson correlation live vs corpus containment: "
        f"{report['correlation']['pearson_live_vs_corpus']:.4f}",
        "",
        "## Threshold sweep (live per-event IDF vs the corpus-IDF reference)",
        "",
        "| threshold | positives | pos.rate | precision | recall | F1 | agreement |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["sweep"]:
        if round(row["threshold"] * 1000) % 25:
            continue
        lines.append(
            f"| {row['threshold']:.3f} | {row['positives']} | {row['positive_rate']:.4f} "
            f"| {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} "
            f"| {row['agreement']:.4f} |"
        )
    lines += [
        "",
        f"F1 optimum: {report['best_f1']['threshold']} "
        f"(F1 {report['best_f1']['f1']:.4f}, agreement {report['best_f1']['agreement']:.4f}).",
        f"Adopted: **{adopted['min_containment']}**.",
        "",
        adopted["rationale"],
        "",
        "Both columns are proxies for \"the agent used this result\"; the sweep "
        "measures how far the live view can drift from the offline label, not "
        "accuracy against truth.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="snapshot SQLite DB (opened read-only)")
    parser.add_argument("--report", required=True, help="output JSON path")
    parser.add_argument("--markdown", help="output markdown path (default: report with .md)")
    parser.add_argument(
        "--reference-threshold",
        type=float,
        default=DEFAULT_MIN_CONTAINMENT,
        help="corpus-IDF threshold defining the reference label",
    )
    args = parser.parse_args(argv)

    report = calibrate(args.db, reference_threshold=args.reference_threshold)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = Path(args.markdown) if args.markdown else report_path.with_suffix(".md")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"calibrated on {report['corpus']['graded_pairs']} pairs -> {report_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
