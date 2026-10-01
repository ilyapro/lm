"""Task-conditioned delivery and optional reading at the MCP boundary."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("fastmcp")
from fastmcp import Client

from living_memory.delivery import DEFAULT_SNIPPET_LADDER
from living_memory.server import (
    _RECALL_DESCRIPTION,
    _RECALL_DESCRIPTION_MANDATORY,
    create_mcp_server,
)
from test_transport_identity import _structured


def test_both_recall_arms_make_lookup_conditional() -> None:
    for description in (_RECALL_DESCRIPTION, _RECALL_DESCRIPTION_MANDATORY):
        assert "If delivered knowledge is insufficient" in description
        assert "content_ref via memory_lookup" in description
        assert "level:schema result is a binding procedure" in description
        assert "Read broad: omit scope" in description
        assert "Refetch a non-full" not in description
    assert "used ids as used" in _RECALL_DESCRIPTION_MANDATORY
    assert "off-topic ids as irrelevant" in _RECALL_DESCRIPTION_MANDATORY


def test_mcp_sufficient_short_and_buried_instruction_need_one_recall(tmp_path: Path) -> None:
    scope = "project:needed-delivery"
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    short = store.append_trace("Amber compass requires seven turns.", {"scope": scope})
    source_ids = [f"01TESTSOURCE{i:015d}" for i in range(20)]
    records = [
        f"- {source_id}: Archive case {index}. " + "Unrelated ledger material. " * 35
        for index, source_id in enumerate(source_ids)
    ]
    records[14] = (
        f"- {source_ids[14]}: For amber compass calibration, apply the "
        "cobalt latch before turning the spindle. " + "Detail. " * 20
    )
    carrier_content = "Evidence: compass cases (case records, not steps)\n" + "\n".join(records)
    carrier = store.append_trace(carrier_content, {"scope": scope})

    async def scenario() -> tuple[dict, dict]:
        async with Client(mcp) as client:
            short_result = _structured(await client.call_tool("memory_recall", {
                "query": "amber compass seven turns", "scope": scope,
            }))
        async with Client(mcp) as client:
            buried_result = _structured(await client.call_tool("memory_recall", {
                "query": "amber compass calibration cobalt latch spindle", "scope": scope,
            }))
        return short_result, buried_result

    short_result, buried_result = asyncio.run(scenario())
    short_entry = next(e for e in short_result["results"] if e["node"]["id"] == short.id)
    assert short_entry["delivery"] == "full"
    assert short_entry["node"]["content"] == short.content
    carrier_entry = next(e for e in buried_result["results"] if e["node"]["id"] == carrier.id)
    assert carrier_entry["delivery"] == "snippet"
    excerpt = carrier_entry["node"]["content"]
    assert len(excerpt) <= DEFAULT_SNIPPET_LADDER[0]
    assert source_ids[14] in excerpt
    assert "apply the cobalt latch before turning the spindle" in excerpt
    assert excerpt.startswith("Evidence: compass cases (case records, not steps)")
    assert carrier_entry["content_ref"]["node_id"] == carrier.id


def test_mcp_missing_detail_uses_byte_complete_lookup_and_correction_survives(
    tmp_path: Path,
) -> None:
    scope = "project:needed-detail"
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    content = (
        "Violet rotor handbook.\n\n"
        + "General rotor background. " * 150
        + "\n\nThe exception code is ORBIT-73; use the revised stop rule."
    )
    node = store.create_node(
        level="concept",
        content=content,
        context={"scope": scope},
        provenance={"corrections": [{
            "by": "reviewer", "supersedes": "01OLDROTORRULE", "text": "Use the revised stop rule",
        }]},
    )

    async def scenario() -> tuple[dict, dict, dict]:
        async with Client(mcp) as client:
            recall = _structured(await client.call_tool("memory_recall", {
                "query": "violet rotor handbook", "scope": scope,
            }))
            lookup = _structured(await client.call_tool("memory_lookup", {"node_id": node.id}))
            missing = _structured(await client.call_tool("memory_recall", {
                "query": "quartz sail absent protocol", "scope": "project:empty-needed-detail",
            }))
        return recall, lookup, missing

    recall, lookup, missing = asyncio.run(scenario())
    entry = next(e for e in recall["results"] if e["node"]["id"] == node.id)
    assert entry["delivery"] == "snippet"
    assert "ORBIT-73" not in entry["node"]["content"]
    assert entry["node"]["provenance"]["corrections"][0]["supersedes"] == "01OLDROTORRULE"
    assert entry["content_ref"]["node_id"] == node.id
    assert lookup["results"][0]["content"] == content
    assert lookup["results"][0]["provenance"]["corrections"] == node.provenance["corrections"]
    assert missing["count"] == 0


def test_mcp_taught_rule_is_current_after_superseding_old_rule(tmp_path: Path) -> None:
    scope = "project:needed-correction"
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    old = mcp.memory_store.append_trace(
        "Cerulean gate rule: open before pressure release.", {"scope": scope}
    )

    async def scenario() -> tuple[dict, dict]:
        async with Client(mcp) as client:
            taught = _structured(await client.call_tool("memory_teach", {
                "trace_id": old.id,
                "correction": "Cerulean gate rule: release pressure before opening.",
                "context": {"scope": scope},
            }))
            recalled = _structured(await client.call_tool("memory_recall", {
                "query": "cerulean gate pressure release opening rule", "scope": scope,
            }))
        return taught, recalled

    taught, recalled = asyncio.run(scenario())
    assert taught["supersedes"]["type"] == "supersedes"
    contents = [entry["node"]["content"] for entry in recalled["results"]]
    assert any("release pressure before opening" in content for content in contents)
    assert all("open before pressure release" not in content for content in contents)
