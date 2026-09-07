#!/usr/bin/env python3
"""Replay the usage signal over a closure corpus: grounding, lookups, coverage.

Consumes the JSONL written by ``scripts/usage_signal_corpus.py`` and re-runs
the live grounding arithmetic (``living_memory.grounding``) over every
result/trace pair, under the tokenizer and threshold of the *current* checkout.
Running it before and after a change to the tokenizer or the threshold is the
before/after measurement each usage-signal vector has to publish.

What it reports, per threshold and per host (plus pooled):

* **pairs** — result/trace pairs grounded, overall and split by script
  (``lat/lat`` vs ``cyr-any``) and by *pair relatedness*: the encoder cosine
  between the delivered node and the closing trace, ``related`` at or above
  ``--related-cosine``, ``unrelated`` at or below ``--unrelated-cosine``. The
  grounded share among unrelated pairs is the lexical false-positive rate the
  threshold has to hold down; the share among related pairs is the signal it
  has to let through.
* **closures** — events with at least one grounded result, split by *closure
  relatedness*: the encoder cosine between the event's query and the closing
  trace. This is the "share of topically related closures that ground
  anything" figure the calibration criterion is stated in.
* **lookups** — results fetched by id through ``memory_lookup`` after the
  event, from the same transport session (``--any-transport`` to relax),
  within ``--lookup-window`` seconds; the overlap with grounding (the pairs a
  lookup credit would double-count if it were not deduplicated) and the
  coverage a grounded-or-looked-up credit rule reaches.

The relatedness measure comes from the production multilingual encoder and
shares nothing with the lexical tokenizer under test, which is what makes it
usable as an external reference for a tokenizer or threshold change.

Usage::

    python3 scripts/usage_signal_replay.py \
        --corpus ~/.cache/living-memory-harness/usage-signal/sfx-closures.jsonl \
        --corpus ~/.cache/living-memory-harness/usage-signal/alt-closures.jsonl \
        --sweep 0.10,0.15,0.20,0.25 --report artifacts/grounding/usage-signal-before.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from living_memory.grounding import (  # noqa: E402
    DEFAULT_MIN_CONTAINMENT,
    ground_token_sets,
    token_set,
)

DEFAULT_RELATED_COSINE = 0.5
DEFAULT_UNRELATED_COSINE = 0.3
DEFAULT_LOOKUP_WINDOW_SECONDS = 24 * 3600.0


def load_corpus(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def relatedness(cosine: float, *, related: float, unrelated: float) -> str:
    if cosine >= related:
        return "related"
    if cosine <= unrelated:
        return "unrelated"
    return "middle"


def grade_record(record: dict[str, Any]) -> dict[str, float]:
    """Containment of every result of one closure under the current tokenizer."""

    results = record["results"]
    if not results:
        return {}
    graded = ground_token_sets(
        token_set(record["trace"]["content"]),
        {item["node_id"]: token_set(item["content"]) for item in results},
        min_containment=0.0,
    )
    return {node_id: verdict.containment for node_id, verdict in graded.items()}


def looked_up(item: dict[str, Any], *, window: float, same_transport: bool) -> bool:
    for lookup in item.get("lookups", []):
        if same_transport and not lookup.get("same_transport"):
            continue
        lag = float(lookup.get("lag_seconds", 0.0))
        if lag < 0.0 or (window > 0.0 and lag > window):
            continue
        return True
    return False


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


class _Tally:
    def __init__(self) -> None:
        self.total = 0
        self.hit = 0

    def add(self, hit: bool) -> None:
        self.total += 1
        self.hit += int(hit)

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.total, "hit": self.hit, "share": _ratio(self.hit, self.total)}


def measure(
    records: Sequence[dict[str, Any]],
    *,
    threshold: float,
    related: float,
    unrelated: float,
    lookup_window: float,
    same_transport: bool,
    containments: dict[str, dict[str, float]],
) -> dict[str, Any]:
    pairs = _Tally()
    pairs_by_script: dict[str, _Tally] = defaultdict(_Tally)
    pairs_by_relatedness: dict[str, _Tally] = defaultdict(_Tally)
    pairs_by_script_relatedness: dict[str, _Tally] = defaultdict(_Tally)
    closures = _Tally()
    closures_by_relatedness: dict[str, _Tally] = defaultdict(_Tally)
    closures_by_trace_script: dict[str, _Tally] = defaultdict(_Tally)
    lookup_pairs = _Tally()
    lookup_closures = _Tally()
    union_closures = _Tally()
    overlap_pairs = 0
    lookup_only_pairs = 0
    lookup_by_relatedness: dict[str, _Tally] = defaultdict(_Tally)

    for record in records:
        event_id = record["event"]["id"]
        graded = containments.get(event_id, {})
        closure_class = relatedness(
            float(record["query_trace_cosine"]), related=related, unrelated=unrelated
        )
        any_grounded = False
        any_lookup = False
        for item in record["results"]:
            value = graded.get(item["node_id"], 0.0)
            grounded = value >= threshold
            fetched = looked_up(item, window=lookup_window, same_transport=same_transport)
            pair_rel = relatedness(
                float(item["node_trace_cosine"]), related=related, unrelated=unrelated
            )
            pairs.add(grounded)
            pairs_by_script[item["pair_script"]].add(grounded)
            pairs_by_relatedness[pair_rel].add(grounded)
            pairs_by_script_relatedness[f"{item['pair_script']}|{pair_rel}"].add(grounded)
            lookup_pairs.add(fetched)
            lookup_by_relatedness[pair_rel].add(fetched)
            overlap_pairs += int(grounded and fetched)
            lookup_only_pairs += int(fetched and not grounded)
            any_grounded = any_grounded or grounded
            any_lookup = any_lookup or fetched
        closures.add(any_grounded)
        closures_by_relatedness[closure_class].add(any_grounded)
        closures_by_trace_script[record["trace"]["script"]].add(any_grounded)
        lookup_closures.add(any_lookup)
        union_closures.add(any_grounded or any_lookup)

    related_pairs = pairs_by_relatedness["related"].to_dict()
    unrelated_pairs = pairs_by_relatedness["unrelated"].to_dict()
    related_closures = closures_by_relatedness["related"].to_dict()
    unrelated_closures = closures_by_relatedness["unrelated"].to_dict()
    return {
        "threshold": threshold,
        "pairs": pairs.to_dict(),
        "pairs_by_script": {key: tally.to_dict() for key, tally in sorted(pairs_by_script.items())},
        "pairs_by_relatedness": {
            key: tally.to_dict() for key, tally in sorted(pairs_by_relatedness.items())
        },
        "pairs_by_script_relatedness": {
            key: tally.to_dict() for key, tally in sorted(pairs_by_script_relatedness.items())
        },
        "pair_signal_to_noise": _snr(related_pairs, unrelated_pairs),
        "closures": closures.to_dict(),
        "closures_by_relatedness": {
            key: tally.to_dict() for key, tally in sorted(closures_by_relatedness.items())
        },
        "closures_by_trace_script": {
            key: tally.to_dict() for key, tally in sorted(closures_by_trace_script.items())
        },
        "closure_signal_to_noise": _snr(related_closures, unrelated_closures),
        "lookups": {
            "pairs": lookup_pairs.to_dict(),
            "closures": lookup_closures.to_dict(),
            "pairs_by_relatedness": {
                key: tally.to_dict() for key, tally in sorted(lookup_by_relatedness.items())
            },
            "overlap_pairs_grounded_and_looked_up": overlap_pairs,
            "lookup_only_pairs": lookup_only_pairs,
            "closures_grounded_or_looked_up": union_closures.to_dict(),
        },
    }


def _snr(signal: dict[str, Any], noise: dict[str, Any]) -> dict[str, Any]:
    ratio = None
    if noise["share"] > 0.0:
        ratio = round(signal["share"] / noise["share"], 3)
    elif signal["share"] > 0.0:
        ratio = float("inf")
    return {
        "signal_share": signal["share"],
        "noise_share": noise["share"],
        "ratio": ratio,
        "difference": round(signal["share"] - noise["share"], 4),
    }


def replay(
    records: Sequence[dict[str, Any]],
    *,
    thresholds: Sequence[float],
    related: float,
    unrelated: float,
    lookup_window: float,
    same_transport: bool,
) -> dict[str, Any]:
    containments = {record["event"]["id"]: grade_record(record) for record in records}
    hosts = sorted({record["host"] for record in records})
    report: dict[str, Any] = {
        "records": len(records),
        "pairs": sum(len(record["results"]) for record in records),
        "hosts": hosts,
        "relatedness": {"related_cosine": related, "unrelated_cosine": unrelated},
        "lookup": {"window_seconds": lookup_window, "same_transport": same_transport},
        "default_min_containment": DEFAULT_MIN_CONTAINMENT,
        "per_host": {},
        "pooled": [],
        "pair_containment_percentiles": _percentiles(
            [value for graded in containments.values() for value in graded.values()]
        ),
    }
    for host in hosts:
        subset = [record for record in records if record["host"] == host]
        report["per_host"][host] = [
            measure(
                subset,
                threshold=threshold,
                related=related,
                unrelated=unrelated,
                lookup_window=lookup_window,
                same_transport=same_transport,
                containments=containments,
            )
            for threshold in thresholds
        ]
    report["pooled"] = [
        measure(
            records,
            threshold=threshold,
            related=related,
            unrelated=unrelated,
            lookup_window=lookup_window,
            same_transport=same_transport,
            containments=containments,
        )
        for threshold in thresholds
    ]
    return report


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pick(share: float) -> float:
        index = min(len(ordered) - 1, int(share * (len(ordered) - 1)))
        return round(ordered[index], 4)

    return {"p50": pick(0.5), "p75": pick(0.75), "p90": pick(0.9), "p95": pick(0.95)}


def render_markdown(report: dict[str, Any], title: str) -> str:
    lines = [f"# {title}", ""]
    lines.append(
        f"- Records: {report['records']} closures / {report['pairs']} pairs over hosts "
        f"{', '.join(report['hosts'])}; relatedness by encoder cosine "
        f"(related ≥ {report['relatedness']['related_cosine']}, "
        f"unrelated ≤ {report['relatedness']['unrelated_cosine']}); lookups "
        f"{'same-transport' if report['lookup']['same_transport'] else 'any transport'} "
        f"within {report['lookup']['window_seconds']:.0f} s."
    )
    lines.append(
        f"- Pair containment percentiles: {report['pair_containment_percentiles']}"
    )
    lines.append("")
    for label, rows in [("pooled", report["pooled"])] + sorted(report["per_host"].items()):
        lines.append(f"## {label}")
        lines.append("")
        lines.append(
            "| thr | pairs grounded | lat/lat | cyr-any | pairs related | pairs unrelated | "
            "pair S/N | closures grounded | closures related | closures unrelated | "
            "closure S/N | lookup pairs | lookup closures | overlap | union closures |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for row in rows:
            scripts = row["pairs_by_script"]
            lookups = row["lookups"]
            lines.append(
                "| {thr:.3f} | {pairs} | {lat} | {cyr} | {prel} | {punrel} | {psnr} | "
                "{closures} | {crel} | {cunrel} | {csnr} | {lpairs} | {lclos} | {overlap} | {union} |".format(
                    thr=row["threshold"],
                    pairs=_cell(row["pairs"]),
                    lat=_cell(scripts.get("lat/lat")),
                    cyr=_cell(scripts.get("cyr-any")),
                    prel=_cell(row["pairs_by_relatedness"].get("related")),
                    punrel=_cell(row["pairs_by_relatedness"].get("unrelated")),
                    psnr=row["pair_signal_to_noise"]["ratio"],
                    closures=_cell(row["closures"]),
                    crel=_cell(row["closures_by_relatedness"].get("related")),
                    cunrel=_cell(row["closures_by_relatedness"].get("unrelated")),
                    csnr=row["closure_signal_to_noise"]["ratio"],
                    lpairs=_cell(lookups["pairs"]),
                    lclos=_cell(lookups["closures"]),
                    overlap=lookups["overlap_pairs_grounded_and_looked_up"],
                    union=_cell(lookups["closures_grounded_or_looked_up"]),
                )
            )
        lines.append("")
    return "\n".join(lines)


def _cell(tally: dict[str, Any] | None) -> str:
    if not tally:
        return "–"
    return f"{tally['hit']}/{tally['n']} ({tally['share']:.1%})"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", action="append", required=True, help="JSONL corpus (repeatable)")
    parser.add_argument(
        "--sweep",
        default=None,
        help="comma-separated thresholds; default: the checkout's DEFAULT_MIN_CONTAINMENT",
    )
    parser.add_argument("--related-cosine", type=float, default=DEFAULT_RELATED_COSINE)
    parser.add_argument("--unrelated-cosine", type=float, default=DEFAULT_UNRELATED_COSINE)
    parser.add_argument(
        "--lookup-window", type=float, default=DEFAULT_LOOKUP_WINDOW_SECONDS,
        help="seconds after the event within which a lookup counts (0 = unbounded)",
    )
    parser.add_argument("--any-transport", action="store_true", help="count cross-transport lookups")
    parser.add_argument("--report", default=None, help="JSON report path")
    parser.add_argument("--markdown", default=None, help="Markdown report path")
    parser.add_argument("--title", default="Usage-signal replay")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    thresholds = (
        [float(value) for value in args.sweep.split(",") if value.strip()]
        if args.sweep
        else [DEFAULT_MIN_CONTAINMENT]
    )
    records = load_corpus(args.corpus)
    report = replay(
        records,
        thresholds=thresholds,
        related=args.related_cosine,
        unrelated=args.unrelated_cosine,
        lookup_window=args.lookup_window,
        same_transport=not args.any_transport,
    )
    report["corpus"] = [str(Path(path).resolve()) for path in args.corpus]
    markdown = render_markdown(report, args.title)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(markdown + "\n")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
