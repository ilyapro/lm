"""Aggregate before/after measure of scope-less recall over the real MCP path.

Runs ``memory_recall`` through an in-process FastMCP client against a *copy*
of a store (never the live file: every recall writes an event) and prints
aggregates only -- no query text, node content or scope names leave the run.

Cases, all drawn from the copy itself with a fixed seed:

* ``project`` -- recorded recalls that named a ``project:*`` scope explicitly
  and whose top result was a live node of that scope. The same query is
  re-asked with no scope; the question is whether that node is still reached.
  Asked twice: with no ambient context, and with only an ambient
  ``session_id`` (the metadata a client stamps without meaning to restrict).
* ``negative`` -- seeded nonsense queries; whatever they return is noise a
  wider search could add.
* ``explicit_target_scope`` -- the same queries with the target's own scope
  named: the restricted upper bound the scope-less arms are compared with.
* ``--cases`` -- an optional JSON list of ``{"query", "targets", "scope"}`` cases kept
  outside the repository (e.g. the reported misses).

Usage::

    python scripts/scopeless_recall_measure.py --db /tmp/copy.sqlite3 [--cases cases.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any

SEED = 20260929
PROJECT_CASES = 40
NEGATIVE_CASES = 10
MAX_RESULTS = 5


def _project_cases(db: Path, limit: int) -> list[dict[str, str]]:
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT query, requested_scope, results FROM recall_events
        WHERE requested_scope LIKE 'project:%' AND results <> '[]'
        ORDER BY created_at DESC LIMIT 4000
        """
    ).fetchall()
    live = {
        row["id"]: row["scope"]
        for row in connection.execute("SELECT id, scope FROM nodes WHERE decayed = 0")
    }
    connection.close()
    seen: set[str] = set()
    pool: list[dict[str, str]] = []
    for row in rows:
        query = " ".join(str(row["query"]).split())
        if not query or query in seen:
            continue
        top = (json.loads(row["results"]) or [{}])[0]
        target = str(top.get("node_id") or "")
        if live.get(target) != row["requested_scope"]:
            continue
        seen.add(query)
        pool.append({"query": query, "target": target})
    random.Random(SEED).shuffle(pool)
    return pool[:limit]


def _negative_cases(count: int) -> list[str]:
    rng = random.Random(SEED)
    letters = "bcdfghjklmnpqrstvwxz"
    return [
        " ".join("".join(rng.choice(letters) for _ in range(7)) for _ in range(3))
        for _ in range(count)
    ]


async def _run(db: Path, cases: list[dict[str, str]]) -> dict[str, Any]:
    from fastmcp import Client

    from living_memory.server import create_mcp_server

    mcp = create_mcp_server(db)

    async def recall(
        client: Any, query: str, ambient: dict[str, Any] | None, scope: str | None = None
    ) -> tuple[dict[str, Any], float, int]:
        arguments: dict[str, Any] = {"query": query, "max_results": MAX_RESULTS}
        if scope:
            arguments["scope"] = scope
        if ambient:
            arguments["ambient_context"] = ambient
        started = time.perf_counter()
        result = await client.call_tool("memory_recall", arguments)
        elapsed = (time.perf_counter() - started) * 1000.0
        payload = result.structured_content or json.loads(result.content[0].text)
        return payload, elapsed, len(json.dumps(payload, ensure_ascii=False))

    report: dict[str, Any] = {}
    arms = (
        ("no_ambient", None, False),
        ("session_id_only", {"session_id": "measure"}, False),
        ("explicit_target_scope", None, True),
    )
    # One MCP session per arm: session dedup would otherwise stub, in a later
    # arm, whatever an earlier arm already delivered.
    for label, ambient, explicit in arms:
        async with Client(mcp) as client:
            await recall(client, "warm up the encoder and the chunk index", None)
            reached = 0
            outside = 0
            delivered = 0
            latencies: list[float] = []
            volume: list[int] = []
            for case in cases:
                scope = case.get("scope") if explicit else None
                if explicit and not scope:
                    continue
                payload, elapsed, size = await recall(client, case["query"], ambient, scope)
                ids = [entry["node"]["id"] for entry in payload["results"]]
                reached += any(target in ids for target in case.get("targets") or [case["target"]])
                delivered += len(ids)
                latencies.append(elapsed)
                volume.append(size)
                target_scope = case.get("scope")
                if target_scope:
                    outside += sum(
                        1
                        for entry in payload["results"]
                        if entry["node"]["scope"] not in {target_scope, "global"}
                    )
        report[label] = {
            "cases": len(latencies),
            "target_reached": reached,
            "results_delivered": delivered,
            "results_outside_target_scope_and_global": outside,
            "latency_ms_p50": round(statistics.median(latencies), 1),
            "latency_ms_p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1),
            "response_chars_mean": round(statistics.mean(volume)),
        }
    negatives = _negative_cases(NEGATIVE_CASES)
    noise = 0
    async with Client(mcp) as client:
        for query in negatives:
            payload, _elapsed, _size = await recall(client, query, None)
            noise += len(payload["results"])
    report["negative"] = {"cases": len(negatives), "results_delivered": noise}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, required=True, help="a disposable copy of the store")
    parser.add_argument("--cases", type=Path, help="extra JSON cases kept outside the repo")
    args = parser.parse_args()
    cases = _project_cases(args.db, PROJECT_CASES)
    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    scopes = dict(connection.execute("SELECT id, scope FROM nodes"))
    connection.close()
    for case in cases:
        case["scope"] = scopes[case["target"]]
    report = {"project": asyncio.run(_run(args.db, cases))}
    if args.cases:
        extra = json.loads(args.cases.read_text())
        reported = asyncio.run(_run(args.db, extra))
        report["reported"] = {label: reported[label] for label in ("no_ambient", "explicit_target_scope")}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
