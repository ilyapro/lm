#!/usr/bin/env python3
"""Paired ``memory_recall`` latency with and without the query-anchor entry.

What this measures, and why it is shaped the way it is
------------------------------------------------------

The query-anchor entry adds two things to the read path: a **scan** of the
live anchor vectors on every recall (:meth:`MemoryRecallService._collect_anchor_seeds`),
and, when something clears the match floor, a **hop** — a second graph walk
opened from the matched anchors' edge targets. The scan is paid always; the
hop is paid only on a match. Any honest figure has to separate them, because
the operator's typical recall pays the first and not the second.

The protocol below is the one the previous (throwaway) benchmark used, kept
verbatim so the numbers stay comparable; it lives in the repo now so the
figure is reproducible:

* **One snapshot, one process, one store.** Both arms are the *same*
  ``MemoryRecallService`` with ``anchor_seeding`` toggled between iterations.
  Two services would each build their own chunk index and scope caches, and
  the difference between two cold caches is not the difference the anchor
  entry makes. Toggling one attribute isolates exactly the scan and the hop.
* **Arms alternated every iteration.** A fixed arm order gives the second arm
  the first arm's warm page cache; when the original benchmark did that it
  produced a spurious +5.6 ms. Alternating cancels drift and cache warming.
* **Paired, per query.** Per-query spread is 80–400 ms depending on scope size
  and depth, which dwarfs a single-digit-millisecond effect. Each query's two
  arm medians are differenced, and the *differences* are summarised. The
  pooled percentiles are reported too, because that is what an operator's
  dashboard would show, but the paired number is the one that answers "what
  did anchors cost".
* **Reads only.** ``log_access=False, log_event=False``, so neither arm
  mutates the working copy and neither arm's writes perturb the other's timing.
* **Real queries.** ``holdout`` = distinct queries from ``recall_events``
  strictly at/after the cutoff — traffic the anchor corpus never saw.
  ``repeat`` = distinct queries from strictly before it, i.e. the very events
  the anchors were built from. ``repeat`` is a deliberate worst case: every
  one of those queries has an anchor sitting on top of it.

Usage
-----

    python3 scripts/anchor_latency_bench.py \\
        --snapshot /tmp/anchor-eval/snap-anchored.sqlite3 \\
        --cutoff 2026-07-15T00:00:00Z \\
        --out artifacts/anchors/latency.json

The snapshot must already carry the anchor tables (build one with
``scripts/backfill_query_anchors.py backfill``). The live database is never
opened: ``--snapshot`` is copied to a scratch working copy first, because
``MemoryStore`` opens its path read-write.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from living_memory.config import MemoryConfig  # noqa: E402
from living_memory.query_anchors import (  # noqa: E402
    ANCHOR_MATCH_COSINE_THRESHOLD,
    ANCHOR_MATCH_LIMIT,
    match_anchors,
)
from living_memory.retrieval import MemoryRecallService  # noqa: E402
from living_memory.retrieval_harness import (  # noqa: E402
    git_commit,
    load_manifest,
    sha256_file,
    working_copy,
)
from living_memory.storage import MemoryStore  # noqa: E402

BUDGET_MS = 5.0
"""The acceptance bar: p50 ``memory_recall`` no worse than +5 ms."""


# ----------------------------------------------------------------------
# Query selection
# ----------------------------------------------------------------------


def _distinct_events(
    connection: sqlite3.Connection, *, cutoff: str, after: bool, limit: int, seed: int
) -> list[dict[str, Any]]:
    """Distinct real queries from one side of the cutoff.

    Distinct on the query text: replaying the same string twice measures the
    page cache, not the anchor entry. Scope and depth are replayed as recorded,
    so the scan sees the scope partition it would really see.
    """

    comparison = ">=" if after else "<"
    rows = connection.execute(
        f"""
        SELECT query, requested_scope, depth, MAX(created_at) AS created_at
        FROM recall_events
        WHERE created_at {comparison} ?
          AND LENGTH(TRIM(query)) > 0
        GROUP BY query
        ORDER BY created_at
        """,
        (cutoff,),
    ).fetchall()
    events = [
        {
            "query": row[0],
            "scope": row[1] or "global",
            "depth": _parse_depth(row[2]),
            "created_at": row[3],
        }
        for row in rows
    ]
    if len(events) <= limit:
        return events
    return random.Random(seed).sample(events, limit)


def _parse_depth(raw: Any) -> int | str:
    """``recall_events.depth`` is TEXT and holds either an int or a keyword."""

    if raw is None:
        return 1
    text = str(raw)
    try:
        return int(text)
    except ValueError:
        return text


# ----------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------


class _AnchorStageTimer:
    """Wraps ``_collect_anchor_seeds`` to bill the scan separately.

    The wrapper is installed for the whole run and records nothing while the
    anchors-off arm is active, because that arm short-circuits before the scan.
    """

    def __init__(self, service: MemoryRecallService) -> None:
        self._inner = service._collect_anchor_seeds
        self.elapsed_ms: float = 0.0
        self.seeds: int = 0
        service._collect_anchor_seeds = self  # type: ignore[method-assign]

    def __call__(self, plan: Any, query_embedding: Any) -> dict[str, float]:
        start = time.perf_counter()
        seeds = self._inner(plan, query_embedding)
        self.elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.seeds = len(seeds)
        return seeds


def _time_recall(
    service: MemoryRecallService, event: dict[str, Any], timer: _AnchorStageTimer
) -> tuple[float, float, int]:
    """One recall. Returns (wall ms, anchor stage ms, seed count)."""

    timer.elapsed_ms = 0.0
    timer.seeds = 0
    start = time.perf_counter()
    service.memory_recall(
        event["query"],
        scope=event["scope"],
        depth=event["depth"],
        max_results=10,
        log_access=False,
        log_event=False,
    )
    wall_ms = (time.perf_counter() - start) * 1000.0
    return wall_ms, timer.elapsed_ms, timer.seeds


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile; the sample sizes here are too small for interpolation."""

    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "median": round(statistics.median(values), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "max": round(max(values), 3),
        "n": len(values),
    }


# ----------------------------------------------------------------------
# Match rate (the diagnostic that decides which cost row is paid)
# ----------------------------------------------------------------------


def _match_rate(
    service: MemoryRecallService, events: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Unfloored nearest-anchor cosine per holdout query.

    Separates "no anchor exists" from "the floor rejected it". Computed with
    ``min_similarity=0.0`` and ``include_targets=False`` so it is a pure
    similarity probe and costs no edge lookups.
    """

    best: list[float] = []
    cleared = 0
    # The same cached matrix the read path uses, so the probe is a scan and
    # not 384 re-reads of every anchor BLOB.
    source = service._anchor_vector_source() or service.store
    for event in events:
        plan = service.scope_resolver.resolve(
            scope=event["scope"], store=service.store
        )
        embedding = service.embedder.embed(event["query"])
        if not embedding:
            continue
        matches = match_anchors(
            source, embedding, plan, limit=1, min_similarity=0.0,
            include_targets=False,
        )
        similarity = matches[0].similarity if matches else 0.0
        best.append(similarity)
        if similarity >= ANCHOR_MATCH_COSINE_THRESHOLD:
            cleared += 1
    return {
        "holdout_distinct_queries": len(best),
        "holdout_queries_that_seeded": cleared,
        "share": round(cleared / len(best), 4) if best else 0.0,
        "match_floor": ANCHOR_MATCH_COSINE_THRESHOLD,
        "match_limit": ANCHOR_MATCH_LIMIT,
        "best_anchor_cosine_percentiles": {
            "p50": round(_percentile(best, 0.50), 3),
            "p75": round(_percentile(best, 0.75), 3),
            "p90": round(_percentile(best, 0.90), 3),
            "p95": round(_percentile(best, 0.95), 3),
            "p99": round(_percentile(best, 0.99), 3),
            "max": round(max(best), 3) if best else 0.0,
        },
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="anchored snapshot (never the live db)")
    parser.add_argument("--cutoff", required=True, help="ISO cutoff separating repeat from holdout")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--iters", type=int, default=13, help="timed iterations per query per arm")
    parser.add_argument("--warmup", type=int, default=2, help="untimed iterations per query per arm")
    parser.add_argument("--holdout-queries", type=int, default=22)
    parser.add_argument("--repeat-queries", type=int, default=20)
    parser.add_argument("--match-rate-sample", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args(argv)

    snapshot = Path(args.snapshot)
    manifest = load_manifest(snapshot)

    with working_copy(snapshot) as working_db:
        store = MemoryStore(MemoryConfig(db_path=working_db))
        try:
            service = MemoryRecallService(store)
            timer = _AnchorStageTimer(service)
            connection = sqlite3.connect(f"file:{working_db}?mode=ro", uri=True)

            holdout = _distinct_events(
                connection, cutoff=args.cutoff, after=True,
                limit=args.holdout_queries, seed=args.seed,
            )
            repeat = _distinct_events(
                connection, cutoff=args.cutoff, after=False,
                limit=args.repeat_queries, seed=args.seed,
            )
            match_rate_events = _distinct_events(
                connection, cutoff=args.cutoff, after=True,
                limit=args.match_rate_sample, seed=args.seed,
            )
            corpus = {
                "anchors": store.count_query_anchors(),
                "live_anchors": store.count_query_anchors(include_decayed=False),
                "edges": store.count_query_anchor_edges(),
                "active_nodes": connection.execute(
                    "SELECT COUNT(*) FROM nodes WHERE decayed = 0"
                ).fetchone()[0],
                "connections": connection.execute(
                    "SELECT COUNT(*) FROM connections"
                ).fetchone()[0],
                "recall_events": connection.execute(
                    "SELECT COUNT(*) FROM recall_events"
                ).fetchone()[0],
                "cutoff": args.cutoff,
            }
            corpus["edges_per_anchor"] = (
                round(corpus["edges"] / corpus["anchors"], 2) if corpus["anchors"] else 0.0
            )

            # The scan caches its matrix on first use; warm it outside the
            # timed region so query #1 does not carry the whole corpus load.
            service.anchor_seeding = True
            if holdout:
                _time_recall(service, holdout[0], timer)

            per_query: list[dict[str, Any]] = []
            pooled_with: list[float] = []
            pooled_without: list[float] = []

            plan = [(event, "holdout") for event in holdout]
            plan += [(event, "repeat") for event in repeat]

            for event, stratum in plan:
                with_samples: list[float] = []
                without_samples: list[float] = []
                stage_samples: list[float] = []
                seed_counts: list[int] = []

                for _ in range(args.warmup):
                    service.anchor_seeding = True
                    _time_recall(service, event, timer)
                    service.anchor_seeding = False
                    _time_recall(service, event, timer)

                for iteration in range(args.iters):
                    # Alternate which arm goes first, every iteration.
                    arms = [True, False] if iteration % 2 == 0 else [False, True]
                    for anchors_on in arms:
                        service.anchor_seeding = anchors_on
                        wall, stage, seeds = _time_recall(service, event, timer)
                        if anchors_on:
                            with_samples.append(wall)
                            stage_samples.append(stage)
                            seed_counts.append(seeds)
                        else:
                            without_samples.append(wall)

                pooled_with.extend(with_samples)
                pooled_without.extend(without_samples)
                anchor_seeds = max(seed_counts) if seed_counts else 0
                matched = anchor_seeds > 0
                per_query.append(
                    {
                        "scope": event["scope"],
                        "depth": event["depth"],
                        "stratum": (
                            "repeat" if stratum == "repeat"
                            else ("matched_holdout" if matched else "unmatched_holdout")
                        ),
                        "matched": matched,
                        "anchor_seeds": anchor_seeds,
                        "query_chars": len(event["query"]),
                        "with_anchors_median_ms": round(statistics.median(with_samples), 3),
                        "without_anchors_median_ms": round(statistics.median(without_samples), 3),
                        "anchor_stage_median_ms": round(statistics.median(stage_samples), 3),
                        "paired_delta_ms": round(
                            statistics.median(with_samples) - statistics.median(without_samples), 3
                        ),
                    }
                )

            service.anchor_seeding = True
            match_rate = _match_rate(service, match_rate_events)

            def _deltas(predicate: Any) -> list[float]:
                return [row["paired_delta_ms"] for row in per_query if predicate(row)]

            def _stages(predicate: Any) -> dict[str, Any]:
                values = [row["anchor_stage_median_ms"] for row in per_query if predicate(row)]
                if not values:
                    return {"n": 0}
                return {
                    "p50": round(statistics.median(values), 3),
                    "p95": round(_percentile(values, 0.95), 3),
                    "max": round(max(values), 3),
                    "n": len(values),
                }

            report: dict[str, Any] = {
                "artifact": str(Path(args.out)),
                "what": "memory_recall p50/p95 with and without the query-anchor graph entry",
                "budget": {"rule": f"p50 memory_recall no worse than +{BUDGET_MS:g} ms", "unit": "ms"},
                "reproduce": (
                    f"python3 scripts/anchor_latency_bench.py --snapshot {args.snapshot} "
                    f"--cutoff {args.cutoff} --out {args.out} --iters {args.iters} "
                    f"--warmup {args.warmup} --holdout-queries {args.holdout_queries} "
                    f"--repeat-queries {args.repeat_queries} "
                    f"--match-rate-sample {args.match_rate_sample} --seed {args.seed}"
                ),
                "method": {
                    "script": "scripts/anchor_latency_bench.py",
                    "database": (
                        "sqlite backup-API snapshot, copied to a scratch working copy; "
                        "the live database is never opened"
                    ),
                    "snapshot": str(snapshot),
                    "snapshot_sha256": sha256_file(snapshot),
                    "snapshot_manifest": manifest.get("derivation"),
                    "anchor_population": (
                        "whatever the snapshot carries -- built by "
                        "scripts/backfill_query_anchors.py from real consumed recall_events "
                        "strictly before the cutoff, with the real encoder"
                    ),
                    "eval_queries": (
                        "distinct real recall_events queries strictly at/after the cutoff "
                        "(holdout), plus a worst-case repeat stratum of distinct queries from "
                        "strictly before it -- the events the anchors were built from"
                    ),
                    "arms": (
                        "one MemoryRecallService with anchor_seeding toggled, so caches and "
                        "chunk index are shared and only the scan and hop differ"
                    ),
                    "arm_order": (
                        "alternated every iteration; a fixed order gave the second arm the "
                        "first arm's warm page cache and produced a spurious +5.6 ms"
                    ),
                    "reads": "log_access=False, log_event=False, so neither arm mutates the snapshot",
                    "paired": (
                        "per-query medians differenced, because per-query spread (80-400 ms, "
                        "driven by scope size and depth) dwarfs the effect"
                    ),
                    "iters_per_query": args.iters,
                    "warmup_per_query": args.warmup,
                    "seed": args.seed,
                    "machine": "same host as the live server; numpy present",
                },
                "corpus": corpus,
                "match_rate": match_rate,
                "with_anchors": {
                    "p50_ms": round(statistics.median(pooled_with), 3),
                    "p95_ms": round(_percentile(pooled_with, 0.95), 3),
                    "mean_ms": round(statistics.fmean(pooled_with), 3),
                    "samples": len(pooled_with),
                },
                "without_anchors": {
                    "p50_ms": round(statistics.median(pooled_without), 3),
                    "p95_ms": round(_percentile(pooled_without, 0.95), 3),
                    "mean_ms": round(statistics.fmean(pooled_without), 3),
                    "samples": len(pooled_without),
                },
                "pooled_delta_ms": {
                    "p50": round(
                        statistics.median(pooled_with) - statistics.median(pooled_without), 3
                    ),
                    "p95": round(
                        _percentile(pooled_with, 0.95) - _percentile(pooled_without, 0.95), 3
                    ),
                },
                "paired_delta_ms": {
                    "all": _summary(_deltas(lambda row: True)),
                    "unmatched_holdout": _summary(
                        _deltas(lambda row: row["stratum"] == "unmatched_holdout")
                    ),
                    "matched_holdout": _summary(
                        _deltas(lambda row: row["stratum"] == "matched_holdout")
                    ),
                    "repeat_worst_case": _summary(_deltas(lambda row: row["stratum"] == "repeat")),
                },
                "anchor_stage_ms": {
                    "all": _stages(lambda row: True),
                    "matched_holdout": _stages(lambda row: row["stratum"] == "matched_holdout"),
                    "unmatched_holdout": _stages(lambda row: row["stratum"] == "unmatched_holdout"),
                    "repeat": _stages(lambda row: row["stratum"] == "repeat"),
                },
                "per_query": per_query,
                "baseline_commit": git_commit(),
            }

            paired_all = report["paired_delta_ms"]["all"].get("median", 0.0)

            # The measured query mix is deliberately adversarial: the repeat
            # stratum is every-query-has-an-anchor by construction, so matched
            # queries are over-represented against real traffic. Re-weight the
            # two holdout strata by the match rate actually observed on holdout
            # traffic to get the figure an operator would see.
            matched_share = match_rate["share"]
            matched_median = report["paired_delta_ms"]["matched_holdout"].get("median")
            unmatched_median = report["paired_delta_ms"]["unmatched_holdout"].get("median")
            traffic_weighted: float | None = None
            if matched_median is not None and unmatched_median is not None:
                traffic_weighted = round(
                    matched_share * matched_median + (1.0 - matched_share) * unmatched_median, 3
                )
            report["traffic_weighted_delta_ms"] = {
                "p50": traffic_weighted,
                "matched_share": matched_share,
                "definition": (
                    "matched_share * paired matched_holdout p50 + "
                    "(1 - matched_share) * paired unmatched_holdout p50, where matched_share "
                    "is the fraction of distinct holdout queries that clear the match floor. "
                    "The measured mix over-represents matched queries; this re-weights to it."
                ),
            }

            report["verdict"] = {
                "budget_ms": BUDGET_MS,
                "budget_basis": "paired_delta_ms.all.median",
                "p50_delta_paired_ms": paired_all,
                "p50_delta_pooled_ms": report["pooled_delta_ms"]["p50"],
                "p50_delta_traffic_weighted_ms": traffic_weighted,
                "within_budget": bool(paired_all <= BUDGET_MS),
                "pooled_within_budget": bool(report["pooled_delta_ms"]["p50"] <= BUDGET_MS),
                "traffic_weighted_within_budget": (
                    None if traffic_weighted is None else bool(traffic_weighted <= BUDGET_MS)
                ),
                "mix_note": (
                    f"{sum(1 for r in per_query if r['stratum'] != 'unmatched_holdout')} of "
                    f"{len(per_query)} measured queries carry a matching anchor, against "
                    f"{matched_share:.1%} of real distinct holdout queries. The pooled p50 is "
                    "therefore an upper bound on this mix, not a traffic estimate; the paired "
                    "and traffic-weighted rows are the ones that describe the operator's recall."
                ),
            }

            Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(
                f"paired p50 delta {paired_all:+.3f} ms "
                f"(budget +{BUDGET_MS:g} ms, within={report['verdict']['within_budget']}), "
                f"pooled p50 delta {report['pooled_delta_ms']['p50']:+.3f} ms, "
                f"anchor stage p50 {report['anchor_stage_ms']['all']['p50']:.3f} ms "
                f"-> {args.out}",
                file=sys.stderr,
            )
        finally:
            store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
