"""Invented source-record counterexample and development-checker guards."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from living_memory.server import create_mcp_server

REJECTED = "273f5a9e1385e292f20a3dde284fa9356047baab"
QUERY = "cedar vault relay current archival rotation audit runtime ledger"
FACT = "keep signed receipts in the local ledger before any archive rotation"
SOURCE = "01SYNTH00000000000000008"

WORKER = r'''
import asyncio, json, sys
sys.path.insert(0, sys.argv[1])
from fastmcp import Client
import living_memory.delivery as delivery
from living_memory.server import create_mcp_server

def structured(value):
    return value.structured_content if value.structured_content is not None else json.loads(value.content[0].text)

async def run():
    mcp = create_mcp_server(sys.argv[2])
    async with Client(mcp) as client:
        recall = structured(await client.call_tool("memory_recall", {
            "query": sys.argv[3], "scope": "project:invented-cedar", "max_results": 5
        }))
        responses = [recall]
        required = json.loads(sys.argv[4])
        def sufficient():
            return all(any(needle in item.get("node", item)["content"]
                           for response in responses for item in response["results"])
                       for needle in required)
        recall_sufficient = sufficient()
        if not recall_sufficient:
            carrier = next(item for item in recall["results"] if item["node"]["level"] == "concept")
            assert carrier["content_ref"]["node_id"] == carrier["node"]["id"]
            responses.append(structured(await client.call_tool("memory_lookup", {
                "node_id": carrier["node"]["id"]
            })))
        assert sufficient()
        print(json.dumps({
            "delivery_path": delivery.__file__,
            "ranked_ids": [item["node"]["id"] for item in recall["results"]],
            "recall_sufficient": recall_sufficient,
            "responses": responses,
            "response_chars": [len(json.dumps(r, ensure_ascii=False, separators=(",", ":"))) for r in responses],
            "calls": len(responses),
        }, ensure_ascii=False))

asyncio.run(run())
'''


def _read(src: Path, db: Path, required: tuple[str, ...]) -> dict:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("LM_DELIVERY_") or key.startswith("LM_RECALL_NEAR_DUP_"):
            env.pop(key)
    result = subprocess.run(
        [sys.executable, "-c", WORKER, str(src), str(db), QUERY, json.dumps(required)],
        env=env, text=True, capture_output=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_latest_rejected_revision_counterexample(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    archive = subprocess.run(["git", "archive", REJECTED, "src"], cwd=root,
                             capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(rejected)], input=archive.stdout, check=True)

    db = tmp_path / "source.sqlite3"
    mcp = create_mcp_server(db)
    target = (
        "Cedar vault relay rotation audit ledger covers the current archival runtime workflow. "
        + "Historical dates and context for the cedar vault relay. " * 12
        + " The active rule for cedar vault relay is: " + FACT + ". "
        + "Further background. " * 16
    )
    decoy = (
        "Cedar vault relay current archival rotation audit runtime ledger had several unrelated scheduling notes. "
        + "General meeting notes. " * 33
    )
    other = (
        "Cedar vault relay current archival rotation audit was discussed without a rule. "
        + "Old meeting notes. " * 33
    )
    records = [f"- 01SYNTH{i:020d}: {decoy if i == 0 else other}" for i in range(8)]
    records.append(f"- {SOURCE}: {target}")
    content = "Evidence: invented archive records, not procedures\n" + "\n".join(records)
    carrier = mcp.memory_store.create_node(
        level="concept", content=content, context={"scope": "project:invented-cedar"}
    )
    assert f"- {SOURCE}: " in carrier.content and FACT in carrier.content
    databases = [tmp_path / "rejected.sqlite3", tmp_path / "candidate.sqlite3"]
    with sqlite3.connect(db) as source:
        for path in databases:
            with sqlite3.connect(path) as destination:
                source.backup(destination)
    required = (f"- {SOURCE}:", FACT)
    before = _read(rejected / "src", databases[0], required)
    after = _read(root / "src", databases[1], required)
    assert Path(before["delivery_path"]).is_relative_to(rejected / "src")
    assert Path(after["delivery_path"]).is_relative_to(root / "src")
    assert before["ranked_ids"] == after["ranked_ids"]
    assert before["ranked_ids"][0] == carrier.id
    assert not before["recall_sufficient"]
    assert before["calls"] == 2
    assert before["responses"][1]["results"][0]["content"] == carrier.content
    assert after["recall_sufficient"]
    assert after["calls"] == 1
    assert after["responses"][0]["results"][0]["delivery"] == "snippet"
    assert len(after["responses"][0]["results"][0]["node"]["content"]) <= 2400
    for arm in (before, after):
        assert arm["response_chars"] == [
            len(json.dumps(r, ensure_ascii=False, separators=(",", ":")))
            for r in arm["responses"]
        ]
    assert sum(after["response_chars"]) < sum(before["response_chars"])


def _checker():
    path = Path(__file__).resolve().parents[1] / "scripts/check_carrier_development.py"
    spec = importlib.util.spec_from_file_location("check_carrier_development", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _report() -> dict:
    def row(chars: int, lookup: bool = False) -> dict:
        responses = [{"tool": "memory_recall", "chars": chars}]
        if lookup:
            responses.append({"tool": "memory_lookup", "chars": 4000})
        return {"ordered_retrieval_ids": [["invented-id"]], "oracle_pass": True,
                "sufficient": True, "remaining_requirements": 0,
                "responses": responses, "calls": len(responses), "recall_calls": 1,
                "lookup_calls": int(lookup), "response_chars": sum(x["chars"] for x in responses)}
    return {"revisions": {"baseline": "base", "candidate": "candidate"},
            "cases": [{"split": "development", "ranking_changed": False,
                       "baseline": row(9000), "candidate": row(3000)}]}


def test_development_checker_excludes_holdout_and_requires_one_case() -> None:
    checker = _checker()
    packet = {"schema_version": 2, "cases": [
        {"split": "holdout", "family": "known_large_carrier", "query": "sealed"},
        {"split": "development", "family": "known_large_carrier", "query": "selected"},
        {"split": "development", "family": "ordinary_short_fact", "query": "different"},
    ]}
    subset = checker.selected_packet(packet)
    assert len(subset["cases"]) == 1
    assert subset["cases"][0]["query"] == "selected"
    with pytest.raises(ValueError, match="exactly one"):
        checker.selected_packet({**packet, "cases": packet["cases"][:1]})
    with pytest.raises(ValueError, match="exactly one"):
        checker.selected_packet({**packet, "cases": packet["cases"] + [packet["cases"][1]]})


def test_development_checker_rejects_invalid_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checker = _checker()
    fixture = tmp_path / "fixture"
    fixture.write_bytes(b"fixture")
    prereg = tmp_path / "prereg.md"
    prereg.write_text("| Synthetic fixtures | `fixture` | `" + "0" * 64 + "` |\n")
    monkeypatch.setattr(checker, "PREREG", prereg)
    monkeypatch.setattr(checker, "HASH_LABELS", {"Synthetic fixtures": fixture})
    with pytest.raises(ValueError, match="frozen hash mismatch"):
        checker.verify_inputs()
    report = _report()
    with pytest.raises(ValueError, match="revision provenance"):
        checker.check_report(report, baseline="wrong", candidate="candidate", rejected=False)
    report["cases"][0]["ranking_changed"] = True
    with pytest.raises(ValueError, match="retrieval IDs"):
        checker.check_report(report, baseline="base", candidate="candidate", rejected=False)


def test_development_checker_rejects_insufficiency_and_added_calls() -> None:
    checker = _checker()
    report = _report()
    checker.check_report(report, baseline="base", candidate="candidate", rejected=False)
    insufficient = copy.deepcopy(report)
    insufficient["cases"][0]["candidate"]["sufficient"] = False
    with pytest.raises(ValueError, match="insufficient knowledge"):
        checker.check_report(insufficient, baseline="base", candidate="candidate", rejected=False)
    extra = copy.deepcopy(report)
    row = extra["cases"][0]["candidate"]
    row["responses"].append({"tool": "memory_lookup", "chars": 10000})
    row["calls"] += 1
    row["lookup_calls"] += 1
    row["response_chars"] += 10000
    with pytest.raises(ValueError, match="adds a call"):
        checker.check_report(extra, baseline="base", candidate="candidate", rejected=False)
    extra["cases"][0]["candidate"]["responses"] = []
    with pytest.raises(ValueError, match="skipped execution"):
        checker.check_report(extra, baseline="base", candidate="candidate", rejected=False)
