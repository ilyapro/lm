#!/usr/bin/env python3
"""BM25 (``nodes_fts``) query latency with and without Cyrillic prefix terms.

``retrieval._expanded_query`` appends one FTS5 prefix term (``"стем"*``) per
Russian content word when ``LM_TOKENIZE_CYRILLIC_STEM`` is on. A prefix term
makes FTS5 scan every index term that starts with the stem instead of seeking
one term, so the cost has to be measured on a real index. This script replays
recorded Russian queries from a read-only snapshot through
``MemoryStore.search_content`` exactly as ``_collect_bm25`` calls it (the
expanded query, one call per scope of the recorded event's plan, the same
per-scope limit) and reports wall-clock percentiles for both switch states.

The snapshot is never opened for writing: ``retrieval_harness.working_copy``
makes a temporary copy through the SQLite backup API first, because
``MemoryStore`` runs schema migrations on open.

Usage::

    python3 scripts/bm25_cyrillic_latency.py \\
        --snapshot ~/.cache/living-memory-harness/usage-signal/sfx-2026-09-07.sqlite3 \\
        --since 2026-08-01 --queries 400 --rounds 3 --report /tmp/bm25-latency.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from living_memory.config import MemoryConfig  # noqa: E402
from living_memory.embeddings import (  # noqa: E402
    CYRILLIC_STEM_ENV_VAR,
    cyrillic_prefix_terms,
    reset_cyrillic_stem_cache,
)
from living_memory.retrieval import _expanded_query  # noqa: E402
from living_memory.retrieval_harness import working_copy  # noqa: E402
from living_memory.scope import ScopeResolver  # noqa: E402
from living_memory.storage import MemoryStore, _fts_query  # noqa: E402

CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")


def recorded_queries(snapshot: Path, *, since: str, limit: int) -> list[tuple[str, str | None]]:
    connection = sqlite3.connect(f"file:{snapshot.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT query, scope FROM recall_events WHERE created_at >= ? ORDER BY created_at DESC",
            (since,),
        ).fetchall()
    finally:
        connection.close()
    picked: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for query, scope in rows:
        if not query or not CYRILLIC_RE.search(query) or query in seen:
            continue
        seen.add(query)
        picked.append((str(query), str(scope) if scope else None))
        if len(picked) >= limit:
            break
    return picked


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def pick(share: float) -> float:
        index = min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))
        return round(ordered[index] * 1000.0, 3)

    return {
        "p50_ms": pick(0.5),
        "p90_ms": pick(0.9),
        "p95_ms": pick(0.95),
        "p99_ms": pick(0.99),
        "max_ms": round(ordered[-1] * 1000.0, 3),
        "mean_ms": round(statistics.fmean(ordered) * 1000.0, 3),
    }


def measure(
    store: MemoryStore,
    queries: list[tuple[str, str | None]],
    *,
    switch: str,
    rounds: int,
    max_results: int,
) -> dict[str, object]:
    os.environ[CYRILLIC_STEM_ENV_VAR] = switch
    reset_cyrillic_stem_cache()
    per_scope_limit = max(25, max_results * 8)
    resolver = ScopeResolver()
    per_query: list[float] = []
    per_call: list[float] = []
    prefix_counts: list[int] = []
    term_counts: list[int] = []
    result_counts: list[int] = []
    for _ in range(rounds):
        for query, scope in queries:
            plan = resolver.resolve(query=query, scope=scope, store=store)
            expanded = _expanded_query(query)
            fts = _fts_query(expanded)
            prefix_counts.append(len(cyrillic_prefix_terms(query)))
            term_counts.append(fts.count(" OR ") + 1 if fts else 0)
            started = time.perf_counter()
            hits = 0
            for plan_scope in plan.scopes:
                call_started = time.perf_counter()
                rows = store.search_content(expanded, scope=plan_scope, limit=per_scope_limit)
                per_call.append(time.perf_counter() - call_started)
                hits += len(rows)
            per_query.append(time.perf_counter() - started)
            result_counts.append(hits)
    return {
        "switch": switch,
        "queries": len(queries),
        "rounds": rounds,
        "per_query": _percentiles(per_query),
        "per_search_content_call": _percentiles(per_call),
        "prefix_terms_per_query": {
            "mean": round(statistics.fmean(prefix_counts), 3),
            "max": max(prefix_counts),
        },
        "fts_terms_per_query": {
            "mean": round(statistics.fmean(term_counts), 3),
            "max": max(term_counts),
        },
        "results_per_query_mean": round(statistics.fmean(result_counts), 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--snapshot", required=True, help="read-only SQLite snapshot")
    parser.add_argument("--since", default="2026-08-01", help="recall_events.created_at floor")
    parser.add_argument("--queries", type=int, default=400, help="distinct Russian queries to replay")
    parser.add_argument("--rounds", type=int, default=3, help="passes over the query list")
    parser.add_argument("--max-results", type=int, default=5, help="recall max_results (sets the per-scope limit)")
    parser.add_argument("--report", help="JSON report path")
    args = parser.parse_args(argv)

    snapshot = Path(args.snapshot).expanduser()
    queries = recorded_queries(snapshot, since=args.since, limit=args.queries)
    if not queries:
        print("no Russian queries found", file=sys.stderr)
        return 1
    previous = os.environ.get(CYRILLIC_STEM_ENV_VAR)
    report: dict[str, object] = {
        "snapshot": str(snapshot),
        "since": args.since,
        "queries": len(queries),
        "rounds": args.rounds,
        "max_results": args.max_results,
        "arms": [],
    }
    try:
        with working_copy(snapshot) as working_db:
            store = MemoryStore(MemoryConfig(db_path=working_db))
            try:
                # Warm the page cache with the arm that touches more pages, then
                # measure off/on/off/on so cache drift cannot favour one arm.
                measure(store, queries[:50], switch="on", rounds=1, max_results=args.max_results)
                arms = []
                for switch in ("off", "on", "off", "on"):
                    arms.append(
                        measure(
                            store,
                            queries,
                            switch=switch,
                            rounds=args.rounds,
                            max_results=args.max_results,
                        )
                    )
                report["arms"] = arms
            finally:
                store.close()
    finally:
        if previous is None:
            os.environ.pop(CYRILLIC_STEM_ENV_VAR, None)
        else:
            os.environ[CYRILLIC_STEM_ENV_VAR] = previous
        reset_cyrillic_stem_cache()

    for arm in report["arms"]:  # type: ignore[union-attr]
        print(
            f"{arm['switch']:>3}: per query p50 {arm['per_query']['p50_ms']} ms "
            f"p95 {arm['per_query']['p95_ms']} ms max {arm['per_query']['max_ms']} ms; "
            f"per call p50 {arm['per_search_content_call']['p50_ms']} ms "
            f"p95 {arm['per_search_content_call']['p95_ms']} ms; "
            f"prefix terms/query mean {arm['prefix_terms_per_query']['mean']} "
            f"max {arm['prefix_terms_per_query']['max']}; results/query {arm['results_per_query_mean']}"
        )
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
