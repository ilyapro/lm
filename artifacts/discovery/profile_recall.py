"""Per-phase profiler for MemoryRecallService.memory_recall.

Decomposes hot/median recall latency into stages, on a copy of the live DB.
Run read-only (no-write) for the read path, then write-enabled for the
access/event-logging cost. Uses the SAME embedding backend as the live server
(auto -> real model when cached) unless LIVING_MEMORY_EMBEDDING_BACKEND is set.

Usage:
    PYTHONPATH=src python3 artifacts/discovery/profile_recall.py /tmp/lm-profile-copy.sqlite3
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore

DB = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/lm-profile-copy.sqlite3")

# (scope, query, depth). Queries are representative of real recalls per scope.
MATRIX = [
    ("project:lm", "validate recall latency bottleneck profiling storage vector", 1),
    ("project:octopus", "OCPA bilingual lifecycle canary evaluation trainer planner", 1),
    ("project:ae", "bootstrap file chunk duplicate density feedback applied goal tree", 1),
    ("global", "living memory architecture retrieval scope embeddings consolidation", 1),
    # depth=0 isolates graph cost for the heaviest scope:
    ("project:octopus", "OCPA bilingual lifecycle canary evaluation trainer planner", 0),
]
MAX_RESULTS = 10
WARMUP = 2
ITERS = 9

_current: dict[str, float] = {}
_counts: dict[str, int] = {}


def _wrap(obj, name, key, accumulate=False, count_arg=None):
    orig = getattr(obj, name)

    def w(*a, **k):
        pre = None
        if count_arg is not None:
            try:
                pre = len(a[count_arg])
            except Exception:
                pre = None
        t0 = time.perf_counter()
        r = orig(*a, **k)
        dt = (time.perf_counter() - t0) * 1000.0
        if accumulate:
            _current[key] = _current.get(key, 0.0) + dt
        else:
            _current[key] = dt
        if count_arg is not None:
            if pre is not None:
                _counts[key + "_cands_pre"] = pre
            try:
                _counts[key + "_cands_post"] = len(a[count_arg])
            except Exception:
                pass
        return r

    setattr(obj, name, w)


def instrument(service: MemoryRecallService) -> None:
    _wrap(service, "_collect_bm25", "bm25")
    _wrap(service, "_collect_vector", "vector_total")
    _wrap(service, "_collect_schema_triggers", "schema")
    _wrap(service, "_collect_graph", "graph", count_arg=1)
    _wrap(service, "rank_candidates", "rank_total")
    _wrap(service, "_supersedes_sets", "supersedes")
    _wrap(service, "_record_result_access", "record_access", accumulate=True)
    # embed_query is called inside _collect_vector once; record separately.
    oe = service.embedder.embed

    def we(text):
        t0 = time.perf_counter()
        r = oe(text)
        _current["embed_query"] = (time.perf_counter() - t0) * 1000.0
        return r

    service.embedder.embed = we
    ore = service.store.record_recall_event

    def wre(*a, **k):
        t0 = time.perf_counter()
        r = ore(*a, **k)
        _current["record_event"] = (time.perf_counter() - t0) * 1000.0
        return r

    service.store.record_recall_event = wre


def run(service, scope, query, depth, *, log_access, log_event):
    global _current
    _current = {}
    t0 = time.perf_counter()
    res = service.memory_recall(
        query, scope=scope, depth=depth, max_results=MAX_RESULTS,
        ambient_context={"task": "optimization/profile-latency", "agent": "claude"},
        log_access=log_access, log_event=log_event,
    )
    total = (time.perf_counter() - t0) * 1000.0
    snap = dict(_current)
    snap["total"] = total
    snap["_nresults"] = len(res)
    # vector_scan = vector_total - embed_query (embed runs inside _collect_vector)
    if "vector_total" in snap and "embed_query" in snap:
        snap["vector_scan"] = max(0.0, snap["vector_total"] - snap["embed_query"])
    return snap


def med(xs):
    return round(statistics.median(xs), 2)


def p95(xs):
    o = sorted(xs)
    return round(o[int(0.95 * (len(o) - 1))], 2)


def profile(write: bool):
    label = "WRITE (log_access+log_event)" if write else "NO-WRITE (read path)"
    print(f"\n{'='*78}\n{label}\n{'='*78}")
    with MemoryStore(DB) as store:
        service = MemoryRecallService(store)
        instrument(service)
        for scope, query, depth in MATRIX:
            for _ in range(WARMUP):
                run(service, scope, query, depth, log_access=write, log_event=write)
            samples: list[dict] = []
            for _ in range(ITERS):
                samples.append(run(service, scope, query, depth, log_access=write, log_event=write))
            keys = ["bm25", "embed_query", "vector_scan", "schema", "graph",
                    "supersedes", "rank_total", "record_access", "record_event", "total"]
            agg = {}
            for k in keys:
                vals = [s.get(k, 0.0) for s in samples]
                agg[k] = (med(vals), p95(vals))
            nres = samples[-1].get("_nresults")
            cpre = _counts.get("graph_cands_pre")
            cpost = _counts.get("graph_cands_post")
            print(f"\n[{scope}] depth={depth} results={nres} graph_cands {cpre}->{cpost}")
            print(f"  {'phase':14s} {'median':>9s} {'p95':>9s}   %of-total")
            tot_med = agg["total"][0] or 1.0
            for k in keys:
                m, p = agg[k]
                bar = "#" * int(40 * m / tot_med) if k != "total" else ""
                pct = f"{100*m/tot_med:4.0f}%" if k != "total" else ""
                print(f"  {k:14s} {m:9.2f} {p:9.2f}   {pct:>5s} {bar}")


def main():
    print(f"DB: {DB}  size={DB.stat().st_size/1e6:.1f} MB")
    import os
    print(f"backend env LIVING_MEMORY_EMBEDDING_BACKEND={os.environ.get('LIVING_MEMORY_EMBEDDING_BACKEND','(unset->auto/real)')}")
    profile(write=False)
    profile(write=True)


if __name__ == "__main__":
    main()
