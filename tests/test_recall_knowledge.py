"""Agent reading and metadata consumers across the public recall views."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import pytest
from fastmcp import Client

from living_memory import delivery
from living_memory.delivery import shape_recall_results
from test_delivery_shaping import delivered_history
from living_memory.models import Node
from living_memory.postsession.client import LiveMemoryClient, McpMemoryIndex
from living_memory.retrieval import RecallResult
from living_memory.server import create_mcp_server
from test_transport_identity import _structured


@pytest.mark.parametrize("metadata_size", [0, 1000])
def test_knowledge_does_not_materialize_diagnostic_metadata(monkeypatch, metadata_size):
    class UncopiedMapping(Mapping):
        def __init__(self, values):
            self.values = values

        def __getitem__(self, key):
            return self.values[key]

        def __len__(self):
            return len(self.values)

        def __iter__(self):
            raise AssertionError("knowledge copied diagnostic metadata")

    class UncopiedSources(list):
        def __iter__(self):
            raise AssertionError("knowledge copied source ids")

    def diagnostic_read(*args):
        raise AssertionError("knowledge constructed diagnostics")

    nodes = [
        Node(id="full", level="trace", content="Complete knowledge."),
        Node(id="snippet", level="trace", content="Source passage. " * 300),
        Node(id="schema", level="schema", content="Required step. " * 300),
        Node(id="withheld", level="trace", content="Unqualified knowledge."),
    ]
    # Distinct content does not require source comparison for twin identity.
    for node in nodes:
        node.context = UncopiedMapping({str(i): "context" for i in range(metadata_size)})
        node.source_traces = UncopiedSources(str(i) for i in range(metadata_size))
        node.corrections = [{"id": "corrector", "text": "Use current knowledge."}]
        node.provenance = UncopiedMapping({
            **{str(i): "history" for i in range(metadata_size)},
            "supersedes": "prior-node", "corrections": [{"wrong": "not authoritative"}],
        })
    monkeypatch.setattr(delivery, "node_to_dict", diagnostic_read)
    monkeypatch.setattr(Node, "stats", property(diagnostic_read))
    shaped = shape_recall_results(
        [RecallResult(node, 1., withheld="below_threshold" if node.id == "withheld" else None)
         for node in nodes],
        delivery_history=delivery.DeliveryHistory(), snippet_max_chars=100,
        context_value_max_chars=160, session_dedup=True, view="knowledge",
    )
    assert [item["delivery"] for item in shaped] == ["full", "snippet", "full", "below_threshold"]
    assert [item["id"] for item in shaped] == [node.id for node in nodes]
    assert shaped[0]["content"] == nodes[0].content
    assert len(shaped[1]["content"]) < len(nodes[1].content)
    assert shaped[2]["content"] == nodes[2].content
    assert shaped[3]["content"] == ""
    for item in shaped:
        assert item["provenance"] == {
            "corrections": [{"id": "corrector", "text": "Use current knowledge."}],
            "supersedes": "prior-node",
        }


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


@pytest.mark.parametrize("view", ["knowledge", "full"])
@pytest.mark.parametrize("first_level", ["trace", "schema"])
@pytest.mark.parametrize("replace_twin", [False, True])
def test_mcp_schema_twins_share_one_complete_offer(tmp_path, view, first_level, replace_twin):
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    text = (
        "Calibrate the silver dial in sequence.\n"
        + "\n".join(f"Inspect each of the individual gauge teeth and log its reading, stage {i}."
                    for i in range(100))
        + "\nStop if the final pin is absent."
    )
    scope = "project:gauges"
    levels = [first_level, "trace" if first_level == "schema" else "schema"]
    strong = {"confidence": 1., "usefulness_score": 1., "access_count": 20}
    weak = {"confidence": .85, "usefulness_score": 0., "access_count": 14}
    ids = [store.create_node(level=level, content=text, context={"scope": scope}, stats=stats).id
           for level, stats in zip(levels, (strong, weak), strict=True)]
    fact = "Maintenance records mention the gauge with the serial number."
    extra = store.create_node(level="trace", content=fact, context={"scope": scope},
                              stats={"confidence": .02}) if replace_twin else None
    query = {"query": "silver dial calibration sequence gauge reading", "scope": scope,
             "depth": 0, "view": view, "max_results": 2}

    async def scenario():
        async with Client(mcp) as client:
            if first_level == "trace":
                # A prior excerpt for this bearer cannot suppress the complete recipe.
                single = _structured(await client.call_tool("memory_recall", {**query, "max_results": 1}))
                assert single["results"][0]["delivery"] == "snippet"
            response = _structured(await client.call_tool("memory_recall", query))
            bearer, second = response["results"]
            nodes = [item.get("node", item) for item in (bearer, second)]
            expected_ids = [ids[0], extra.id] if extra else ids
            assert [node["id"] for node in nodes] == expected_ids
            assert [node["level"] for node in nodes] == ([levels[0], "trace"] if extra else levels)
            assert bearer["delivery"] == "full" and nodes[0]["content"] == text
            if extra:
                assert second["delivery"] == "full" and nodes[1]["content"] == fact
            else:
                assert second["delivery"] == "twin_duplicate" and nodes[1]["content"] != text
                assert second["content_ref"]["duplicate_of"] == ids[0]
                assert second["content_ref"]["node_id"] == ids[1]

            def full_record(body):
                digest = hashlib.sha256(body.encode()).hexdigest()
                return {"kind": "full", "content_sha256": digest, "delivered_sha256": digest}

            event = store.get_recall_event(response["recall_event_id"])
            assert event.result_ids == expected_ids
            assert [r["delivery"] for r in event.results] == [
                full_record(text), full_record(fact) if extra else {"kind": "twin_duplicate"},
            ]
            repeat = _structured(await client.call_tool("memory_recall", {**query, "used": expected_ids}))
            assert repeat["feedback_marks"]["accepted"] == 2
            assert [item["delivery"] for item in repeat["results"]] == [
                "session_duplicate", "session_duplicate" if extra else "twin_duplicate",
            ]
            lookup = _structured(await client.call_tool("memory_lookup", {"node_ids": ids}))
            assert {item["id"]: item["content"] for item in lookup["results"]} == dict.fromkeys(ids, text)

    try:
        asyncio.run(scenario())
    finally:
        store.close()


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
            guidance = recall_tool.inputSchema["properties"]["query"]["description"]
            assert "current role or stage" in guidance and "fact sought" in guidance
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
    observation = index.candidates("Quartz extraction source", scope=context["scope"])
    assert observation.status == "success"
    snapshots = observation.nodes
    assert len(snapshots) == 1
    assert snapshots[0].content == node.content
    assert snapshots[0].created_at == node.created_at
    assert snapshots[0].context["source_span"] == context["source_span"]
    assert snapshots[0].context["source_transcript_sha256"] == context["source_transcript_sha256"]
    assert snapshots[0].source_identity == (context["source_transcript_sha256"], context["source_span"])
    assert client.calls == ["memory_recall", "memory_lookup"]


def test_invalid_view_does_not_record_recall(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")

    async def scenario() -> None:
        async with Client(mcp) as client:
            with pytest.raises(Exception, match="view|knowledge"):
                await client.call_tool("memory_recall", {"query": "quartz", "view": "unknown"})
        assert mcp.memory_store.list_recall_events() == []

    asyncio.run(scenario())
