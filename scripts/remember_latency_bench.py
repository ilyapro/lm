#!/usr/bin/env python3
"""Measure what content grounding costs the ``memory_remember`` write path.

Grounding runs inside ``apply_pending_recall_feedback``, so it lands on every
remember that consumes a pending recall event: the consuming trace and each
delivered result node get tokenized and scored. This benchmark isolates that
cost by running one identical seeded workload twice, changing only
``LM_RECALL_CREDIT_POLICY`` — ``all`` skips grading entirely (the pre-grounding
path), ``grounded`` performs it.

Realism matters for the number to mean anything, so node contents are sampled
from a real snapshot rather than invented: tokenization cost scales with
content length, and Living Memory traces are long (mean ~1.1 KB, tail past
100 KB).

Both embedding backends are reported. The ``hash`` arm isolates the grounding
delta with a large sample and no model noise; the real-model arm is the
production-representative p50 the +10 ms budget is written against.

Usage::

    python3 scripts/remember_latency_bench.py \
        --snapshot ~/.cache/living-memory-harness/snapshot.sqlite3 \
        --report artifacts/grounding/remember-latency.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SEED = 20260818
CORPUS_NODES = 400
MAX_RESULTS = 10


class _FakeMCP:
    def __init__(self, name: str, instructions: str | None = None) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.tools[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        return decorate if func is None else decorate(func)

    def resource(self, uri: str, **_kwargs: Any) -> Any:
        return lambda func: func

    def prompt(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            return inner

        return decorate if func is None else decorate(func)


def sample_contents(snapshot: str, count: int) -> list[str]:
    """Real trace contents, so tokenization sees production-sized documents."""

    connection = sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT content FROM nodes WHERE decayed = 0 AND LENGTH(content) > 200"
            " ORDER BY id LIMIT ?",
            (count * 4,),
        ).fetchall()
    finally:
        connection.close()
    contents = [str(row[0]) for row in rows]
    rng = random.Random(SEED)
    rng.shuffle(contents)
    return contents[:count]


def _percentiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "p50_ms": round(statistics.median(ordered) * 1000, 4),
        "p90_ms": round(ordered[int(0.90 * len(ordered))] * 1000, 4),
        "p99_ms": round(ordered[min(int(0.99 * len(ordered)), len(ordered) - 1)] * 1000, 4),
        "mean_ms": round(statistics.fmean(ordered) * 1000, 4),
    }


def run_arm(contents: list[str], *, policy: str, iterations: int) -> dict[str, Any]:
    """One seeded recall->remember workload under a fixed credit policy."""

    import living_memory.feedback as feedback_module
    from living_memory.server import create_mcp_server

    os.environ["LM_RECALL_CREDIT_POLICY"] = policy
    scope = "project:bench"
    rng = random.Random(SEED)

    # A grounding gate that grounds nothing would be trivially cheap and the
    # measured delta would be meaningless, so count what it actually grades.
    grounded_per_event: list[int] = []
    original_ground = feedback_module.ground_results

    def counting_ground(trace_content, result_contents, **kwargs):  # type: ignore[no-untyped-def]
        graded = original_ground(trace_content, result_contents, **kwargs)
        grounded_per_event.append(sum(1 for item in graded.values() if item.grounded))
        return graded

    feedback_module.ground_results = counting_ground  # type: ignore[assignment]

    try:
        with tempfile.TemporaryDirectory() as workdir:
            mcp = create_mcp_server(Path(workdir) / "bench.sqlite3", mcp_factory=_FakeMCP)
            for content in contents:
                mcp.tools["memory_remember"](
                    content, {"scope": scope, "agent": "seed", "session_id": "seed"}
                )

            ambient = {"agent": "bench", "task": "bench", "session_id": "bench-session"}
            grounded_per_event.clear()
            durations: list[float] = []
            delivered: list[int] = []
            for index in range(iterations):
                source = contents[rng.randrange(len(contents))]
                query = " ".join(source.split()[:12])
                recalled = mcp.tools["memory_recall"](
                    query,
                    scope=scope,
                    max_results=MAX_RESULTS,
                    depth=0,
                    ambient_context=dict(ambient),
                )
                delivered.append(len(recalled["results"]))
                # The consuming trace quotes the query, as a real follow-up
                # note would, so grounding has real work to do rather than
                # bailing out on an empty overlap.
                note = f"follow-up {index}: {source[:600]}"
                start = time.perf_counter()
                mcp.tools["memory_remember"](note, {"scope": scope, **ambient})
                durations.append(time.perf_counter() - start)
    finally:
        feedback_module.ground_results = original_ground  # type: ignore[assignment]

    return {
        "policy": policy,
        "mean_results_delivered": round(statistics.fmean(delivered), 3)
        if delivered
        else 0.0,
        "graded_events": len(grounded_per_event),
        "mean_grounded_per_event": round(statistics.fmean(grounded_per_event), 3)
        if grounded_per_event
        else 0.0,
        "events_with_a_grounded_result": sum(1 for count in grounded_per_event if count),
        **_percentiles(durations),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="snapshot to sample contents from")
    parser.add_argument("--report", required=True)
    parser.add_argument("--hash-iterations", type=int, default=400)
    parser.add_argument("--model-iterations", type=int, default=150)
    parser.add_argument("--budget-ms", type=float, default=10.0)
    args = parser.parse_args(argv)

    contents = sample_contents(args.snapshot, CORPUS_NODES)
    lengths = sorted(len(content) for content in contents)
    backends: dict[str, Any] = {}

    for backend, iterations in (
        ("hash", args.hash_iterations),
        ("model", args.model_iterations),
    ):
        if backend == "hash":
            os.environ["LIVING_MEMORY_EMBEDDING_BACKEND"] = "hash"
        else:
            os.environ.pop("LIVING_MEMORY_EMBEDDING_BACKEND", None)
        arms = {
            policy: run_arm(contents, policy=policy, iterations=iterations)
            # "all" first so the grounded arm never benefits from a warm cache
            # the other arm paid for.
            for policy in ("all", "grounded")
        }
        delta = arms["grounded"]["p50_ms"] - arms["all"]["p50_ms"]
        backends[backend] = {
            "arms": arms,
            "p50_delta_ms": round(delta, 4),
            "budget_ms": args.budget_ms,
            "within_budget": delta <= args.budget_ms,
        }

    report = {
        "snapshot": args.snapshot,
        "seed": SEED,
        "corpus_nodes": len(contents),
        "max_results": MAX_RESULTS,
        "content_length_bytes": {
            "min": lengths[0],
            "p50": lengths[len(lengths) // 2],
            "p90": lengths[int(0.9 * len(lengths))],
            "max": lengths[-1],
            "mean": round(statistics.fmean(lengths), 1),
        },
        "backends": backends,
        "verdict": {
            "budget_ms": args.budget_ms,
            "pass": all(block["within_budget"] for block in backends.values()),
            "worst_p50_delta_ms": round(
                max(block["p50_delta_ms"] for block in backends.values()), 4
            ),
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["backends"], indent=2), file=sys.stderr)
    print(f"verdict: {report['verdict']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
