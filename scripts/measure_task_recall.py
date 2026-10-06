#!/usr/bin/env python3
"""Bounded, synthetic, public-tool task recall measurement.

Run from the repository root: python3 scripts/measure_task_recall.py --output PATH
Each invocation creates a fresh SQLite database in a temporary directory. The
fixture is development evidence; it is intentionally not a held-out sample.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fastmcp import Client  # noqa: E402
from living_memory.server import create_mcp_server  # noqa: E402

TASK = "mqr-78164-sable"
OTHER_TASK = "mqr-78165-sable"
SCOPE = "project:marble"
OTHER_SCOPE = "project:quartz"
LIMIT = 6
TARGET_CONTENT = (
    "The integrated comparison found that a native revision carries its wait "
    "through landing. Supervisor review should use the landed comparison "
    "record before deciding whether the release is ready. "
    + "The staged review observed the same result after the landing checkpoint. " * 32
)


def payload(result: Any) -> dict[str, Any]:
    value = result.structured_content
    if set(value) == {"result"} and isinstance(value["result"], dict):
        return value["result"]
    return value


def wire_bytes(value: dict[str, Any]) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def seed(store: Any) -> dict[str, str]:
    """Return node-id to fixture-label map; never persist random node IDs."""
    labels: dict[str, str] = {}

    def add(label: str, content: str, task: str | None, scope: str = SCOPE) -> str:
        context = {"scope": scope, "agent": "synthetic-fixture"}
        if task is not None:
            context["task"] = task
        node = store.append_trace(content, context)
        labels[node.id] = label
        return node.id

    # Identifier appears in context.task only. The body is deliberately long
    # enough to exercise snippet -> lookup when this fact is reached.
    add("target", TARGET_CONTENT, TASK)
    add("similar_task", "The archive export checksum was verified after staging.", OTHER_TASK)
    add("same_task_irrelevant", "The cafeteria inventory was counted on Friday.", TASK)
    add("same_name_other_project", "The quartz project measured packaging labels.", TASK, OTHER_SCOPE)
    old = add(
        "obsolete_instruction",
        "For landing review, the old procedure said to skip the integrated wait comparison.",
        "landing-review-note",
    )
    add(
        "cross_project_lesson",
        "Across projects, a landing decision should compare recorded outcomes "
        "and retain the review evidence before changing a release checklist.",
        "shared-landing-lesson",
        "global",
    )
    return labels | {"_old_id": old}


CASES = [
    {"name": "short_task", "query": TASK, "scope": None, "relevant": ["target"]},
    {
        "name": "contextual_task",
        "query": f"{TASK} supervisor landing integrated comparison",
        "scope": None,
        "relevant": ["target"],
    },
    {
        "name": "explicit_scope",
        "query": f"{TASK} supervisor landing integrated comparison",
        "scope": SCOPE,
        "relevant": ["target"],
    },
    {
        "name": "correction",
        "query": "landing review integrated wait comparison correction",
        "scope": None,
        "relevant": ["correction"],
    },
    {
        "name": "cross_project",
        "query": "cross project landing decision review evidence checklist",
        "scope": None,
        "relevant": ["cross_project_lesson"],
    },
    {
        "name": "missing_knowledge",
        "query": "violet comet submarine oxygen ledger 917",
        "scope": None,
        "relevant": [],
    },
]


async def measure() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lm-task-recall-") as temp:
        mcp = create_mcp_server(Path(temp) / "fixture.sqlite3")
        labels = seed(mcp.memory_store)
        old_id = labels.pop("_old_id")
        async with Client(mcp) as client:
            taught = payload(
                await client.call_tool(
                    "memory_teach",
                    {
                        "trace_id": old_id,
                        "correction": "For landing review, include the integrated wait comparison before the release decision.",
                        "context": {"scope": SCOPE, "task": "landing-review-note", "agent": "synthetic-fixture"},
                    },
                )
            )
        correction_id = taught["corrective_trace"]["id"]
        labels[correction_id] = "correction"

        rows = []
        for case in CASES:
            args = {"query": case["query"], "max_results": LIMIT}
            if case["scope"] is not None:
                args["scope"] = case["scope"]
            start = perf_counter()
            async with Client(mcp) as client:
                recall = payload(await client.call_tool("memory_recall", args))
                found = []
                lookups = []
                for rank, item in enumerate(recall["results"], 1):
                    label = labels.get(item["node"]["id"], "unknown")
                    found.append({
                        "rank": rank,
                        "fact": label,
                        "delivery": item.get("delivery"),
                        "methods": item.get("methods", []),
                    })
                    if label in case["relevant"] and item.get("delivery") != "full":
                        ref = item.get("content_ref", {})
                        if ref.get("node_id"):
                            full = payload(await client.call_tool("memory_lookup", {"node_id": ref["node_id"]}))
                            lookups.append({"fact": label, "bytes": wire_bytes(full), "complete": any(
                                entry["id"] == ref["node_id"] for entry in full.get("results", [])
                            )})
            elapsed_ms = round((perf_counter() - start) * 1000, 3)
            reached = [entry["fact"] for entry in found if entry["fact"] in case["relevant"]]
            rows.append({
                "name": case["name"],
                "query": case["query"],
                "scope": case["scope"],
                "limit": LIMIT,
                "oracle_relevant": case["relevant"],
                "returned_facts": found,
                "reached_relevant": reached,
                "recall_calls": 1,
                "lookup_calls": len(lookups),
                "lookups": lookups,
                "recall_bytes": wire_bytes(recall),
                "recall_plus_lookup_bytes": wire_bytes(recall) + sum(item["bytes"] for item in lookups),
                "elapsed_ms": elapsed_ms,
            })
        return {
            "schema": "synthetic-task-recall-v1",
            "evidence": "development synthetic fixture; not independent holdout",
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "seed": {
                "task_identifier": TASK,
                "similar_task_identifier": OTHER_TASK,
                "scopes": [SCOPE, OTHER_SCOPE, "global"],
                "facts": [
                    {"label": "target", "task": TASK, "scope": SCOPE, "content": TARGET_CONTENT},
                    {"label": "similar_task", "task": OTHER_TASK, "scope": SCOPE, "content": "The archive export checksum was verified after staging."},
                    {"label": "same_task_irrelevant", "task": TASK, "scope": SCOPE, "content": "The cafeteria inventory was counted on Friday."},
                    {"label": "same_name_other_project", "task": TASK, "scope": OTHER_SCOPE, "content": "The quartz project measured packaging labels."},
                    {"label": "obsolete_instruction", "task": "landing-review-note", "scope": SCOPE, "content": "For landing review, the old procedure said to skip the integrated wait comparison."},
                    {"label": "correction", "task": "landing-review-note", "scope": SCOPE, "content": "For landing review, include the integrated wait comparison before the release decision.", "supersedes": "obsolete_instruction"},
                    {"label": "cross_project_lesson", "task": "shared-landing-lesson", "scope": "global", "content": "Across projects, a landing decision should compare recorded outcomes and retain the review evidence before changing a release checklist."},
                ],
            },
            "oracle": {case["name"]: case["relevant"] for case in CASES},
            "cases": rows,
            "totals": {
                "recall_calls": len(rows),
                "lookup_calls": sum(row["lookup_calls"] for row in rows),
                "recall_plus_lookup_bytes": sum(row["recall_plus_lookup_bytes"] for row in rows),
                "elapsed_ms": round(sum(row["elapsed_ms"] for row in rows), 3),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    # Avoid optional periodic jobs changing the isolated fixture during reads.
    os.environ.setdefault("LM_AUTO_CONSOLIDATE_POLICY", "off")
    report = asyncio.run(measure())
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
