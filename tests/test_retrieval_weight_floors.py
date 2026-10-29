from __future__ import annotations

import asyncio
from pathlib import Path
import sqlite3
from unittest.mock import Mock

import pytest

from living_memory.config import MemoryConfig
from living_memory.models import RetrievalPolicyFloors, RetrievalWeights
from living_memory.resources import retrieval_skew_metrics
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore


@pytest.mark.parametrize("channel", ["vector", "graph"])
@pytest.mark.parametrize("minimum", [0.0, 0.1])
@pytest.mark.parametrize("consumer", ["evaluate", "update", "diagnostic"])
def test_evidence_errors_only_affect_enabled_minima(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    channel: str,
    minimum: float,
    consumer: str,
) -> None:
    scope = "session:errors"
    floors = RetrievalPolicyFloors(
        bm25_max=0.8, bm25_min=0.1,
        vector_min=minimum if channel == "vector" else 0,
        graph_min=minimum if channel == "graph" else 0,
    )
    config = MemoryConfig(
        db_path=tmp_path / "memory.sqlite3", retrieval_policy_floors={"session": floors},
    )
    with MemoryStore(config) as store:
        weights = store.set_retrieval_weights(scope, bm25=0, vector=0, graph=1)
        failed_read = Mock(side_effect=sqlite3.OperationalError("evidence unavailable"))
        other_read = Mock(side_effect=AssertionError("disabled evidence read"))
        monkeypatch.setattr(store, f"_has_{channel}_evidence", failed_read)
        other = "graph" if channel == "vector" else "vector"
        monkeypatch.setattr(store, f"_has_{other}_evidence", other_read)

        def evaluate() -> RetrievalWeights:
            if consumer == "evaluate":
                applicable, result = store.evaluate_retrieval_weight_floors(scope, weights)
                assert applicable == RetrievalPolicyFloors(1.0, 0, 0, 0.1)
                return result
            if consumer == "update":
                return store.update_retrieval_weights(scope, graph_signal=1)
            risk, = retrieval_skew_metrics(store, scope=scope)["scopes_at_risk"]
            assert risk["applicable_floors"] == {
                "bm25_max": 1.0, "vector_min": 0, "graph_min": 0, "bm25_min": 0.1,
            }
            return RetrievalWeights(scope=scope, **risk["effective_after_floor"])

        if minimum:
            with pytest.raises(sqlite3.OperationalError, match="evidence unavailable"):
                evaluate()
            failed_read.assert_called_once_with(scope)
            assert store.get_retrieval_weights(scope) == weights
        else:
            result = evaluate()
            assert (result.bm25, result.vector, result.graph) == (0.1, 0, 0.9)
            failed_read.assert_not_called()
        other_read.assert_not_called()


@pytest.mark.parametrize("vector_min,graph_min,expected", [
    (0, 0, (1, 0, 0)),
    (0.1, 0, (0.8, 0.2, 0)),
    (0, 0.05, (0.8, 0, 0.2)),
    (0.1, 0.05, (0.8, 0.15, 0.05)),
])
def test_native_positive_feedback_reads_only_enabled_floor_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    vector_min: float,
    graph_min: float,
    expected: tuple[float, float, float],
) -> None:
    from fastmcp import Client
    import living_memory.server as server_module

    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setattr(server_module, "_transport_session_id", lambda: "floor-transport")
    scope = "session:feedback"
    config = MemoryConfig(
        db_path=tmp_path / "native.sqlite3",
        retrieval_policy_floors={"session": RetrievalPolicyFloors(
            bm25_max=0.8, vector_min=vector_min, graph_min=graph_min, bm25_min=0.1,
        )},
    )
    mcp = create_mcp_server(config=config)
    store = mcp.memory_store

    async def scenario() -> None:
        async with Client(mcp) as client:
            response = await client.call_tool("memory_remember", {
                "content": "cobalt floor feedback fixture", "context": {"scope": scope},
            })
            node_id = response.data["node"]["id"]
            target = store.append_trace("cobalt floor evidence target", {"scope": scope})
            store.create_connection(node_id, target.id, "caused")
            assert store._has_vector_evidence(scope)
            assert store._has_graph_evidence(scope)
            event = store.record_recall_event(
                query="cobalt floor feedback fixture", scope=scope,
                ambient_context={"transport_session_id": "floor-transport"},
                results=[{
                    "node_id": node_id, "bm25_score": 1.0,
                    "vector_score": 0.0, "graph_score": 0.0, "methods": ["bm25"],
                }],
            )
            store.set_retrieval_weights(scope, bm25=1, vector=0, graph=0)
            before = store.get_node(node_id).usefulness_score
            reads = {}
            for channel in ("vector", "graph"):
                name = f"_has_{channel}_evidence"
                reads[channel] = Mock(wraps=getattr(store, name))
                monkeypatch.setattr(store, name, reads[channel])

            response = await client.call_tool("memory_recall", {
                "query": "cobalt floor feedback fixture", "scope": scope, "used": [node_id],
            })
            assert not response.is_error
            assert response.data["feedback_marks"] == {
                "accepted": 1, "dropped": 0, "by_reason": {},
            }
            assert store.credited_node_ids(event.id) == {node_id}
            assert store.get_node(node_id).usefulness_score > before
            weights = store.get_retrieval_weights(scope)
            assert (weights.bm25, weights.vector, weights.graph) == pytest.approx(expected)
            for channel, minimum in (("vector", vector_min), ("graph", graph_min)):
                assert reads[channel].call_count == int(minimum > 0)
                if minimum:
                    reads[channel].assert_called_with(scope)

    try:
        asyncio.run(scenario())
    finally:
        mcp.auto_consolidation.close()
        store.close()
