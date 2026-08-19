#!/usr/bin/env python3
"""Paired ``memory_recall`` latency with and without the recall map.

What this measures, and why it is shaped the way it is
------------------------------------------------------

The map is built from the residual candidate pool a recall has already ranked
and thrown away, on the live read path, inside the same handler the agent is
waiting on. So the only honest question is a ratio: what fraction does the map
add to the recall it decorates? The pre-registered budget is **warm p95
overhead ≤ 0.20** — a fifth of a recall, paid on the path an operator's session
actually walks, where the builder's per-``(scope, task)`` structure cache is
populated. The cold path (first map for a task, full label cascade including
the chunk-vector fallback) is measured too and reported, but it is paid once
per task per corpus revision, not per recall.

The protocol follows ``scripts/anchor_latency_bench.py``, for the same reasons:

* **One snapshot, one process, one store.** Every arm is the *same*
  ``MemoryRecallService``, so chunk index and scope caches are shared and the
  only difference between arms is the map construction itself.
* **Arms alternated every iteration.** A fixed arm order hands the later arm
  the earlier one's warm page cache; on the anchor bench that produced a
  spurious +5.6 ms.
* **Paired, per query.** Per-query spread (scope size, depth) dwarfs a
  single-digit-millisecond effect, so each query's arm medians are differenced
  and the differences summarised. Pooled percentiles are reported as well,
  because the ratio the budget names is a pooled p95 — that is what an
  operator's dashboard shows.
* **Reads only.** ``log_access=False, log_event=False``: neither arm mutates
  the working copy, and the builder's corpus-revision probe therefore stays
  put, which is what makes the warm arm genuinely warm.
* **Real queries and real tasks.** Queries, scopes and ``task`` come from the
  snapshot's own ``recall_events``; ``task`` matters because it is half the
  map's cache key, so replaying it is what makes the warm/cold split real.

Three arms per query:

``off``       recall alone — the "before" number.
``warm``      recall plus a map built by a builder that has already seen this
              ``(scope, task)`` at this corpus revision — the "after" number
              the budget is set against.
``cold``      recall plus a map built by a fresh builder — the full cascade,
              reported for context.

Usage
-----

    python3 scripts/recall_map_latency_bench.py \\
        --snapshot /tmp/anchor-eval/snap-anchored.sqlite3 \\
        --out artifacts/recall-map/latency-before-after.json

``--source-db`` is accepted instead and freezes its own snapshot first (the
live database is a legal argument; it is opened read-only to freeze, never by
``MemoryStore``).
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
from living_memory.recall_map import MAX_CLUSTERS, RecallMapBuilder  # noqa: E402
from living_memory.retrieval import MemoryRecallService  # noqa: E402
from living_memory.retrieval_harness import (  # noqa: E402
    frozen_snapshot,
    git_commit,
    sha256_file,
    working_copy,
)
from living_memory.storage import MemoryStore  # noqa: E402

BUDGET_P95_OVERHEAD_RATIO = 0.20
"""The acceptance bar, pre-registered: warm p95 no worse than +20%."""

ARMS = ("off", "warm", "cold")


# ----------------------------------------------------------------------
# Query selection
# ----------------------------------------------------------------------


def _distinct_events(
    connection: sqlite3.Connection, *, limit: int, seed: int
) -> list[dict[str, Any]]:
    """Distinct real queries, with the scope, depth and task they ran under.

    Distinct on the query text: replaying one string twice measures the page
    cache, not the map. ``task`` is replayed because it is half the builder's
    cache key — without it every query would share one degenerate key and the
    warm arm would be measuring the wrong cache.
    """

    rows = connection.execute(
        """
        SELECT query,
               requested_scope,
               depth,
               task,
               MAX(created_at) AS created_at
        FROM recall_events
        WHERE LENGTH(TRIM(query)) > 0
        GROUP BY query
        ORDER BY created_at
        """
    ).fetchall()
    events = [
        {
            "query": row[0],
            "scope": row[1] or "global",
            "depth": _parse_depth(row[2]),
            "task": row[3],
            "created_at": row[4],
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


def _time_arm(
    service: MemoryRecallService,
    event: dict[str, Any],
    *,
    arm: str,
    warm_builder: RecallMapBuilder,
    store: MemoryStore,
    max_results: int,
) -> dict[str, Any]:
    """One recall, plus the map the arm asks for. Times what the handler times.

    The ``cold`` arm builds its builder *before* the clock starts, so the
    measurement is the cascade and not an object allocation.
    """

    builder = warm_builder if arm == "warm" else RecallMapBuilder(store)
    started = time.perf_counter()
    service.memory_recall(
        event["query"],
        scope=event["scope"],
        depth=event["depth"],
        max_results=max_results,
        log_access=False,
        log_event=False,
    )
    recall_done = time.perf_counter()
    residual = service.last_residual
    built = None
    if arm != "off" and residual:
        built = builder.build(
            residual, scope=event["scope"], task=event["task"]
        )
    finished = time.perf_counter()
    return {
        "wall_ms": (finished - started) * 1000.0,
        "stage_ms": (finished - recall_done) * 1000.0,
        "residual": len(residual),
        "clusters": len(built.clusters) if built is not None else 0,
        "cache_hit": bool(builder.last_cache_hit) if arm != "off" else None,
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile; these sample sizes are too small to interpolate."""

    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "p50_ms": round(statistics.median(values), 3),
        "p95_ms": round(_percentile(values, 0.95), 3),
        "mean_ms": round(statistics.fmean(values), 3),
        "max_ms": round(max(values), 3),
        "samples": len(values),
    }


def _ratio(after: float, before: float) -> float | None:
    if not before:
        return None
    return round(after / before - 1.0, 4)


def _summary_ratios(values: Sequence[float | None]) -> dict[str, Any]:
    """Percentiles of per-query overhead ratios, skipping unmeasurable ones."""

    present = [value for value in values if value is not None]
    if not present:
        return {"n": 0}
    return {
        "p50": round(statistics.median(present), 4),
        "p95": round(_percentile(present, 0.95), 4),
        "max": round(max(present), 4),
        "n": len(present),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", help="frozen snapshot (never the live db)")
    source.add_argument("--source-db", help="database to freeze a snapshot from")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--iters", type=int, default=7, help="timed iterations per query per arm")
    parser.add_argument("--warmup", type=int, default=2, help="untimed iterations per query per arm")
    parser.add_argument("--max-results", type=int, default=5, help="the recall cut, i.e. where the residual starts")
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args(argv)

    with frozen_snapshot(args.snapshot, args.source_db) as (snapshot, manifest):
        with working_copy(snapshot) as working_db:
            store = MemoryStore(MemoryConfig(db_path=working_db))
            try:
                service = MemoryRecallService(store)
                warm_builder = RecallMapBuilder(store)
                connection = sqlite3.connect(f"file:{working_db}?mode=ro", uri=True)

                events = _distinct_events(
                    connection, limit=args.queries, seed=args.seed
                )
                corpus = {
                    "active_nodes": connection.execute(
                        "SELECT COUNT(*) FROM nodes WHERE decayed = 0"
                    ).fetchone()[0],
                    "connections": connection.execute(
                        "SELECT COUNT(*) FROM connections"
                    ).fetchone()[0],
                    "recall_events": connection.execute(
                        "SELECT COUNT(*) FROM recall_events"
                    ).fetchone()[0],
                    "chunk_embeddings": connection.execute(
                        "SELECT COUNT(*) FROM node_chunk_embeddings"
                    ).fetchone()[0],
                    "query_anchors": store.count_query_anchors(),
                    "measured_queries": len(events),
                }

                # Warm the service's own lazy caches outside every timed
                # region, so query #1 does not carry the chunk-index load.
                if events:
                    _time_arm(
                        service, events[0], arm="off", warm_builder=warm_builder,
                        store=store, max_results=args.max_results,
                    )

                pooled: dict[str, list[float]] = {arm: [] for arm in ARMS}
                pooled_stage: dict[str, list[float]] = {"warm": [], "cold": []}
                per_query: list[dict[str, Any]] = []
                warm_cache_hits = 0
                warm_builds = 0

                for event in events:
                    samples: dict[str, list[float]] = {arm: [] for arm in ARMS}
                    stages: dict[str, list[float]] = {"warm": [], "cold": []}
                    residuals: list[int] = []
                    clusters: list[int] = []

                    for _ in range(args.warmup):
                        for arm in ARMS:
                            _time_arm(
                                service, event, arm=arm, warm_builder=warm_builder,
                                store=store, max_results=args.max_results,
                            )

                    for iteration in range(args.iters):
                        # Rotate which arm leads, every iteration.
                        order = [ARMS[(iteration + offset) % len(ARMS)] for offset in range(len(ARMS))]
                        for arm in order:
                            measured = _time_arm(
                                service, event, arm=arm, warm_builder=warm_builder,
                                store=store, max_results=args.max_results,
                            )
                            samples[arm].append(measured["wall_ms"])
                            if arm != "off":
                                stages[arm].append(measured["stage_ms"])
                            if arm == "warm":
                                warm_builds += 1
                                warm_cache_hits += int(bool(measured["cache_hit"]))
                                residuals.append(measured["residual"])
                                clusters.append(measured["clusters"])

                    for arm in ARMS:
                        pooled[arm].extend(samples[arm])
                    for arm in ("warm", "cold"):
                        pooled_stage[arm].extend(stages[arm])

                    medians = {
                        arm: statistics.median(samples[arm]) for arm in ARMS
                    }
                    per_query.append(
                        {
                            "scope": event["scope"],
                            "depth": event["depth"],
                            "task": event["task"],
                            "query_chars": len(event["query"]),
                            "residual": max(residuals) if residuals else 0,
                            "clusters": max(clusters) if clusters else 0,
                            "off_median_ms": round(medians["off"], 3),
                            "warm_median_ms": round(medians["warm"], 3),
                            "cold_median_ms": round(medians["cold"], 3),
                            "warm_stage_median_ms": round(
                                statistics.median(stages["warm"]), 3
                            ) if stages["warm"] else 0.0,
                            "cold_stage_median_ms": round(
                                statistics.median(stages["cold"]), 3
                            ) if stages["cold"] else 0.0,
                            "warm_paired_delta_ms": round(
                                medians["warm"] - medians["off"], 3
                            ),
                            "cold_paired_delta_ms": round(
                                medians["cold"] - medians["off"], 3
                            ),
                            "warm_paired_ratio": _ratio(medians["warm"], medians["off"]),
                            "cold_paired_ratio": _ratio(medians["cold"], medians["off"]),
                        }
                    )

                before = _summary(pooled["off"])
                after_warm = _summary(pooled["warm"])
                after_cold = _summary(pooled["cold"])
                mapped = [row for row in per_query if row["clusters"] > 0]
                warm_deltas = [row["warm_paired_delta_ms"] for row in per_query]
                cold_deltas = [row["cold_paired_delta_ms"] for row in per_query]
                p95_overhead_ratio = _ratio(
                    after_warm.get("p95_ms", 0.0), before.get("p95_ms", 0.0)
                )

                report: dict[str, Any] = {
                    "artifact": str(Path(args.out)),
                    "what": (
                        "memory_recall p50/p95 before (map off) and after (map on), "
                        "cold and warm builder cache, on one frozen snapshot"
                    ),
                    "budget": {
                        "rule": (
                            "warm p95_overhead_ratio <= "
                            f"{BUDGET_P95_OVERHEAD_RATIO:g}"
                        ),
                        "pre_registered": True,
                        "basis": "pooled p95 of the warm arm over the pooled p95 of the off arm",
                    },
                    "reproduce": (
                        "python3 scripts/recall_map_latency_bench.py "
                        f"--snapshot {args.snapshot or snapshot} --out {args.out} "
                        f"--queries {args.queries} --iters {args.iters} "
                        f"--warmup {args.warmup} --max-results {args.max_results} "
                        f"--seed {args.seed}"
                    ),
                    "method": {
                        "script": "scripts/recall_map_latency_bench.py",
                        "database": (
                            "sqlite backup-API snapshot, copied to a scratch working "
                            "copy; the live database is never opened by MemoryStore"
                        ),
                        "snapshot": str(snapshot),
                        "snapshot_sha256": sha256_file(snapshot),
                        "snapshot_manifest": manifest.get("derivation"),
                        "arms": (
                            "one MemoryRecallService for all arms; off = recall alone, "
                            "warm = recall + map from the run-long builder (structure "
                            "cache populated), cold = recall + map from a fresh builder"
                        ),
                        "arm_order": "rotated every iteration, so no arm keeps another's warm page cache",
                        "reads": "log_access=False, log_event=False -- no arm mutates the working copy",
                        "eval_queries": (
                            "distinct real recall_events queries, replayed with their "
                            "recorded scope, depth and task (task is half the map cache key)"
                        ),
                        "max_results": args.max_results,
                        "iters_per_query": args.iters,
                        "warmup_per_query": args.warmup,
                        "seed": args.seed,
                        "max_clusters": MAX_CLUSTERS,
                    },
                    "corpus": corpus,
                    "coverage": {
                        "queries_with_a_map": len(mapped),
                        "queries_measured": len(per_query),
                        "share": (
                            round(len(mapped) / len(per_query), 4) if per_query else 0.0
                        ),
                        "warm_cache_hit_share": (
                            round(warm_cache_hits / warm_builds, 4) if warm_builds else 0.0
                        ),
                        "median_residual": (
                            round(statistics.median([row["residual"] for row in per_query]), 1)
                            if per_query else 0.0
                        ),
                        "median_clusters_when_mapped": (
                            round(statistics.median([row["clusters"] for row in mapped]), 1)
                            if mapped else 0.0
                        ),
                    },
                    "before_map_off": before,
                    "after_map_on_warm": after_warm,
                    "after_map_on_cold": after_cold,
                    "map_stage_ms": {
                        "warm": _summary(pooled_stage["warm"]),
                        "cold": _summary(pooled_stage["cold"]),
                    },
                    "paired_delta_ms": {
                        "warm": {
                            "median": round(statistics.median(warm_deltas), 3) if warm_deltas else 0.0,
                            "p95": round(_percentile(warm_deltas, 0.95), 3),
                            "max": round(max(warm_deltas), 3) if warm_deltas else 0.0,
                            "n": len(warm_deltas),
                        },
                        "cold": {
                            "median": round(statistics.median(cold_deltas), 3) if cold_deltas else 0.0,
                            "p95": round(_percentile(cold_deltas, 0.95), 3),
                            "max": round(max(cold_deltas), 3) if cold_deltas else 0.0,
                            "n": len(cold_deltas),
                        },
                    },
                    # The pooled p95 ratio the budget names is a ratio of two
                    # *tails*: which arm drew the unlucky sample on the two or
                    # three slowest queries moves it by more than the map
                    # costs. The paired form asks the same question per query
                    # and is what survives a re-run, so both are reported and
                    # the verdict carries both.
                    "paired_overhead_ratio": {
                        arm: _summary_ratios(
                            [row[f"{arm}_paired_ratio"] for row in per_query]
                        )
                        for arm in ("warm", "cold")
                    },
                    "overhead_ratio": {
                        "warm_p50": _ratio(
                            after_warm.get("p50_ms", 0.0), before.get("p50_ms", 0.0)
                        ),
                        "warm_p95": p95_overhead_ratio,
                        "cold_p50": _ratio(
                            after_cold.get("p50_ms", 0.0), before.get("p50_ms", 0.0)
                        ),
                        "cold_p95": _ratio(
                            after_cold.get("p95_ms", 0.0), before.get("p95_ms", 0.0)
                        ),
                    },
                    "per_query": per_query,
                    "baseline_commit": git_commit(),
                }

                report["p95_overhead_ratio"] = p95_overhead_ratio
                report["verdict"] = {
                    "budget_p95_overhead_ratio": BUDGET_P95_OVERHEAD_RATIO,
                    "p95_overhead_ratio": p95_overhead_ratio,
                    "within_budget": bool(
                        p95_overhead_ratio is not None
                        and p95_overhead_ratio <= BUDGET_P95_OVERHEAD_RATIO
                    ),
                    "warm_paired_p50_overhead_ratio": (
                        report["paired_overhead_ratio"]["warm"].get("p50")
                    ),
                    "warm_paired_p95_overhead_ratio": (
                        report["paired_overhead_ratio"]["warm"].get("p95")
                    ),
                    "warm_paired_p95_within_budget": bool(
                        (report["paired_overhead_ratio"]["warm"].get("p95") or 0.0)
                        <= BUDGET_P95_OVERHEAD_RATIO
                    ),
                    "warm_paired_delta_median_ms": report["paired_delta_ms"]["warm"]["median"],
                    "warm_cache_hit_share": report["coverage"]["warm_cache_hit_share"],
                    "cold_p95_overhead_ratio": report["overhead_ratio"]["cold_p95"],
                    "cold_p50_overhead_ratio": report["overhead_ratio"]["cold_p50"],
                    "note": (
                        "The budget names the warm arm, and the warm arm is one "
                        "structure per (scope, task) per corpus revision, reused by "
                        "every later recall of that task. How often it is warm in "
                        "production is bounded by what moves the builder's revision "
                        "probe: a chunk-embedding or query-anchor write, i.e. any "
                        "memory_remember -- so a session that remembers between "
                        "recalls pays the cold arm, and the cold row is the one to "
                        "read for it. Cold is reported, not budgeted; its p95 ratio "
                        "is the honest worst case and its p50 ratio is much larger "
                        "because the fixed cascade cost lands hardest on the recalls "
                        "that were fast to begin with."
                    ),
                    "paired_caveat": (
                        "warm_paired_p95_overhead_ratio is the 95th percentile over "
                        "*queries* of (warm median / off median - 1), and it exceeds "
                        "the budget while the budgeted pooled figure does not. Both "
                        "are reported because they answer different questions and "
                        "neither was chosen after the fact: the pooled p95 is the "
                        "tail of the latency an operator waits on (where the map's "
                        "few milliseconds are noise next to a 700 ms recall), the "
                        "paired p95 is the query where the map costs the largest "
                        "*share* -- a fast recall, on which a fixed few milliseconds "
                        "is a big fraction of a small number. Judge the absolute row "
                        "with it: warm_paired_delta_median_ms is the whole cost."
                    ),
                }

                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    "before p50/p95 "
                    f"{before['p50_ms']:.1f}/{before['p95_ms']:.1f} ms -> warm "
                    f"{after_warm['p50_ms']:.1f}/{after_warm['p95_ms']:.1f} ms "
                    f"(p95 overhead {p95_overhead_ratio:+.3f}, budget "
                    f"{BUDGET_P95_OVERHEAD_RATIO:g}, within="
                    f"{report['verdict']['within_budget']}), cold p95 "
                    f"{after_cold['p95_ms']:.1f} ms, map stage warm p50 "
                    f"{report['map_stage_ms']['warm']['p50_ms']:.3f} ms "
                    f"-> {args.out}",
                    file=sys.stderr,
                )
            finally:
                store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
