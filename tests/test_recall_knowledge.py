"""Agent reading and metadata consumers across the public recall views."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path

import pytest
from fastmcp import Client

from living_memory.delivery import shape_recall_results
from test_delivery_shaping import delivered_history
from living_memory.models import Node
from living_memory.postsession.client import LiveMemoryClient, McpMemoryIndex
from living_memory.retrieval import RecallResult
from living_memory.server import create_mcp_server
from test_transport_identity import _structured


def test_knowledge_preserves_delivery_and_validity_without_diagnostics() -> None:
    correction = {"by": "reviewer", "timestamp": "2026-10-20", "text": "Use the safe rule. " * 40}
    nodes = [
        Node(id="schema", level="schema", content="Binding step. " * 400),
        Node(id="snippet", level="trace", content="Long passage. " * 300),
        Node(id="twin", level="trace", content="Long passage. " * 300),
        Node(id="session", level="trace", content="Already read."),
        Node(id="near", level="trace", content="A paraphrase of the long passage."),
        Node(id="withheld", level="trace", content="Below score threshold."),
    ]
    for node in nodes:
        node.context = {"task_pattern": "source", "procedure": "metadata " * 100}
        node.corrections = [deepcopy(correction)]
        node.provenance = {"supersedes": "older-node", "unrelated": ["history"] * 100}
    nodes[1].decayed = True
    nodes[1].decay_reason = "superseded"
    results = [
        RecallResult(node, 0.123456789, vector_score=0.234567891,
                     superseded=True,
                     withheld="below_threshold" if node.id == "withheld" else None)
        for node in nodes
    ]
    original = deepcopy(results)
    options = dict(
        delivery_history=delivered_history(results, {"session"}), snippet_max_chars=100,
        context_value_max_chars=160, session_dedup=True,
        snippet_ladder=(100,),
        correction_refs={node.id: ["replacement"] for node in nodes},
    )
    diagnostic = shape_recall_results(results, **options, view="full")
    knowledge = shape_recall_results(results, **options, view="knowledge")
    assert results == original
    assert [entry["delivery"] for entry in knowledge] == [
        "full", "snippet", "twin_duplicate", "session_duplicate", "full", "below_threshold",
    ]
    for full, item, node in zip(diagnostic, knowledge, nodes):
        assert item["id"] == full["node"]["id"] == node.id
        assert item["superseded"] is full["superseded"] is True
        assert item["superseded_by"] == full["superseded_by"] == ["replacement"]
        assert item["content"] == full["node"]["content"]
        assert item["delivery"] == full["delivery"]
        assert item["content_ref"]["node_id"] == node.id
        assert item["provenance"]["corrections"] == full["node"]["provenance"]["corrections"]
        assert item["provenance"]["supersedes"] == "older-node"
        assert item["decayed"] == node.decayed
        assert item["decay_reason"] == node.decay_reason
        assert not {"node", "context", "stats", "score", "created_at"} & item.keys()
        assert full["bm25_score"] == 0.0
        assert full["vector_score"] == pytest.approx(0.234567891, abs=1e-6)
    assert knowledge[0]["content"] == nodes[0].content
    assert knowledge[2]["content_ref"]["duplicate_of"] == "snippet"
    assert "duplicate_of" not in knowledge[4]["content_ref"]
    assert knowledge[5]["content"] == ""


def test_mcp_read_lookup_feedback_and_diagnostic_access(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    scope = "project:knowledge-flow"
    original = store.append_trace(
        "Quartz rotor calibration. " * 180 + "Exception code ORBIT-73.",
        {"scope": scope, "task_pattern": "quartz-rotor", "procedure": "Original metadata. " * 100},
    )

    async def scenario() -> None:
        async with Client(mcp) as client:
            recall_tool = next(t for t in await client.list_tools() if t.name == "memory_recall")
            assert recall_tool.inputSchema["properties"]["view"]["default"] == "knowledge"
            assert recall_tool.inputSchema["properties"]["view"]["enum"] == ["knowledge", "full"]
            recall = _structured(await client.call_tool("memory_recall", {
                "query": "quartz rotor calibration", "scope": scope,
            }))
            entry = next(r for r in recall["results"] if r["id"] == original.id)
            assert entry["delivery"] == "snippet"
            assert "node" not in entry
            lookup = _structured(await client.call_tool("memory_lookup", {
                "node_id": entry["content_ref"]["node_id"],
            }))["results"][0]
            assert lookup["content"] == original.content
            assert lookup["provenance"] == entry["provenance"]
            assert not {"context", "stats", "created_at"} & lookup.keys()
            marked = _structured(await client.call_tool("memory_recall", {
                "query": "quartz rotor calibration", "scope": scope,
                "used": [entry["id"]], "view": "full",
            }))
            assert marked["feedback_marks"] == {"accepted": 1, "dropped": 0, "by_reason": {}}
            diagnostic = next(r for r in marked["results"] if r["node"]["id"] == original.id)
            assert diagnostic["delivery"] == "session_duplicate"
            assert diagnostic["node"]["context"]["task_pattern"] == "quartz-rotor"
            assert {"score", "bm25_score", "vector_score", "graph_score", "trigger_score", "scope_rank", "methods"} <= diagnostic.keys()
            context = _structured(await client.call_tool("memory_lookup", {
                "node_id": original.id, "context_key": "procedure",
            }))["results"][0]
            assert context["context"] == {"procedure": original.context["procedure"]}
            full = _structured(await client.call_tool("memory_lookup", {
                "node_id": original.id, "view": "full",
            }))["results"][0]
            assert full["context"]["procedure"] == original.context["procedure"]
            assert full["created_at"] == original.created_at
            exact = _structured(await client.call_tool("memory_lookup", {
                "scope": scope, "task_pattern": "quartz-rotor",
            }))["results"][0]
            assert exact["context"]["task_pattern"] == "quartz-rotor"
            assert store.get_recall_event(recall["recall_event_id"]).result_ids == [original.id]
            assert len(store.list_recall_events()) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("long_content", [False, True])
def test_postsession_keeps_original_span_sha_and_created_at(
    tmp_path: Path, long_content: bool,
) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    context = {
        "scope": "project:extraction-metadata",
        "source_span": "session/" + "long-source-path/" * 25 + ":1-5",
        "source_transcript_sha256": "ab" * 32,
    }
    node = mcp.memory_store.append_trace(
        "Quartz extraction source. " * (200 if long_content else 1), context,
    )

    class InProcessClient(LiveMemoryClient):
        async def _call_async(self, tool: str, arguments: dict) -> dict:
            async with Client(mcp) as client:
                return _structured(await client.call_tool(tool, arguments))

    client = InProcessClient()
    index = McpMemoryIndex(client)
    snapshots = index.candidates("Quartz extraction source", scope=context["scope"])
    assert len(snapshots) == 1
    assert snapshots[0].content == node.content
    assert snapshots[0].created_at == node.created_at
    assert index.last_contexts[node.id]["source_span"] == context["source_span"]
    assert index.last_contexts[node.id]["source_transcript_sha256"] == context["source_transcript_sha256"]
    assert client.calls == ["memory_recall", "memory_lookup"]


def test_invalid_view_does_not_record_recall(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")

    async def scenario() -> None:
        async with Client(mcp) as client:
            with pytest.raises(Exception, match="view|knowledge"):
                await client.call_tool("memory_recall", {"query": "quartz", "view": "unknown"})
        assert mcp.memory_store.list_recall_events() == []

    asyncio.run(scenario())
