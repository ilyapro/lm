"""Invented paired MCP reading chains against rejected production and this tree."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from living_memory.server import create_mcp_server


REJECTED = "917836a479763fe102be647475210d4500a9d7a1"
QUERY = "opal valve calibration verified pressure"
FACT = "For opal valve calibration, the verified pressure is 31 psi."
SOURCE = "01SYNTH00000000000019"


WORKER = r'''
import asyncio, json, sys
sys.path.insert(0, sys.argv[1])
from fastmcp import Client
import living_memory.delivery as delivery_module
from living_memory.server import create_mcp_server

def structured(result):
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)

async def run():
    mcp = create_mcp_server(sys.argv[2])
    async with Client(mcp) as client:
        recall = structured(await client.call_tool("memory_recall", {
            "query": sys.argv[3], "scope": "project:synthetic-opal", "max_results": 5
        }))
        responses = [recall]
        required = json.loads(sys.argv[4])
        def sufficient():
            content = "\n".join(
                item.get("node", item)["content"] for response in responses
                for item in response.get("results", [])
            )
            return all(fact in content for fact in required)
        recall_sufficient = sufficient()
        if not recall_sufficient:
            carrier = next(item for item in recall["results"] if item["node"]["level"] == "concept")
            assert carrier["content_ref"]["node_id"] == carrier["node"]["id"]
            responses.append(structured(await client.call_tool("memory_lookup", {
                "node_id": carrier["node"]["id"]
            })))
        assert sufficient()
        print(json.dumps({
            "delivery_path": delivery_module.__file__,
            "ranked_ids": [item["node"]["id"] for item in recall["results"]],
            "ranked_scores": [
                [item[key] for key in ("score", "bm25_score", "vector_score", "graph_score", "trigger_score")]
                for item in recall["results"]
            ],
            "recall_sufficient": recall_sufficient,
            "response_chars": [len(json.dumps(r, sort_keys=True, ensure_ascii=False)) for r in responses],
            "calls": len(responses), "responses": responses,
        }, ensure_ascii=False))

asyncio.run(run())
'''


def _run(src: Path, database: Path, required: tuple[str, ...]) -> dict:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("LM_DELIVERY_") or key.startswith("LM_RECALL_NEAR_DUP_"):
            env.pop(key)
    process = subprocess.run(
        [sys.executable, "-c", WORKER, str(src), str(database), QUERY, json.dumps(required)],
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout.strip().splitlines()[-1])


def test_rejected_revision_counterexample(tmp_path: Path) -> None:
    """Both production versions read the same stored records in fresh sessions."""
    root = Path(__file__).resolve().parents[1]
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    archive = subprocess.run(
        ["git", "archive", REJECTED, "src"], cwd=root, capture_output=True, check=True,
    )
    subprocess.run(["tar", "-x", "-C", str(rejected)], input=archive.stdout, check=True)

    database = tmp_path / "source.sqlite3"
    mcp = create_mcp_server(database)
    decoy = (
        "Opal records discuss the valve inventory and calibration schedule. "
        "A verified inspector logged a pressure question without an answer. "
        + "Routine equipment notes. " * 24
    )
    records = [f"- 01SYNTH{i:020d}: {decoy}" for i in range(19)]
    records.append(f"- {SOURCE}: {FACT} " + "Applicable test context. " * 12)
    records.extend(f"- 01SYNTH{i:020d}: unrelated archive notes." for i in range(20, 24))
    content = "Evidence: invented equipment case records, not procedures\n" + "\n".join(records)
    carrier = mcp.memory_store.create_node(
        level="concept", content=content, context={"scope": "project:synthetic-opal"}
    )
    assert f"- {SOURCE}: {FACT}" in carrier.content
    rejected_db = tmp_path / "rejected.sqlite3"
    candidate_db = tmp_path / "candidate.sqlite3"
    with sqlite3.connect(database) as source:
        for path in (rejected_db, candidate_db):
            with sqlite3.connect(path) as destination:
                source.backup(destination)
    required = (f"- {SOURCE}: {FACT}",)
    before = _run(rejected / "src", rejected_db, required)
    after = _run(root / "src", candidate_db, required)

    assert Path(before["delivery_path"]).is_relative_to(rejected / "src")
    assert Path(after["delivery_path"]).is_relative_to(root / "src")
    assert before["ranked_ids"] == after["ranked_ids"]
    assert before["ranked_scores"] == after["ranked_scores"]
    assert before["ranked_ids"][0] == carrier.id
    for reading in (before, after):
        assert reading["response_chars"] == [
            len(json.dumps(response, sort_keys=True, ensure_ascii=False))
            for response in reading["responses"]
        ]
    assert not before["recall_sufficient"]
    assert before["calls"] == len(before["response_chars"]) == 2
    assert before["responses"][1]["results"][0]["content"] == carrier.content
    assert after["recall_sufficient"]
    assert after["calls"] == len(after["response_chars"]) == 1
    assert after["responses"][0]["results"][0]["delivery"] == "snippet"
    assert len(after["responses"][0]["results"][0]["node"]["content"]) <= 2400
    assert sum(after["response_chars"]) < sum(before["response_chars"])
