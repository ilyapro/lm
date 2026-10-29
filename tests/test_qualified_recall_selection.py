"""Exact-twin replacement uses the recall's current qualification evidence."""

import asyncio

from fastmcp import Client
import pytest

from living_memory.retrieval import MemoryRecallService
from living_memory.server import create_mcp_server


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
        "LM_RECALL_MIN_SCORE": "0.35",
        "LM_RECALL_SCHEMA_TRIGGER": "off",
        "LM_RECALL_MAP": "1",
        "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
        "LM_AUTO_CONSOLIDATE_POLICY": "off",
        "LM_DELIVERY_SESSION_DEDUP": "1",
        "LM_DELIVERY_SNIPPET_LADDER": "off",
    }.items():
        monkeypatch.setenv(key, value)


@pytest.mark.parametrize("view", ["knowledge", "full"])
@pytest.mark.parametrize("form", ["drop", "stub"])
@pytest.mark.parametrize("threshold", ["0.35", "0.7"])
def test_native_qualified_replacement_and_accounting(tmp_path, monkeypatch, view, form, threshold):
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", threshold)
    monkeypatch.setenv("LM_RECALL_GATE_FORM", form)
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    nodes = [store.create_node(level=level, content=content, context={"scope": "project:cobalt"})
             for level, content in [
                 ("trace", "Cobalt routing requires approval."),
                 ("concept", "Cobalt routing requires approval."),
                 ("trace", "Cobalt routing requires inventory."),
                 ("trace", "Cobalt porcelain kiln temperatures."),
             ]]
    observations = []
    rank = MemoryRecallService.rank_candidates

    def observe(service, *args, **kwargs):
        rows = rank(service, *args, **kwargs)
        observations.append((service, rows))
        return rows

    monkeypatch.setattr(MemoryRecallService, "rank_candidates", observe)
    selected = [nodes[0].id, nodes[2 if threshold == "0.35" else 1].id]

    async def scenario():
        async with Client(mcp) as client:
            args = {"query": "cobalt routing approval", "scope": "project:cobalt",
                    "depth": 0, "max_results": 2, "view": view}
            first = (await client.call_tool("memory_recall", args)).structured_content
            service, ranked = observations[-1]
            # Observe ordinary ranking; no ordering or channel score is replaced.
            assert [r.node.id for r in ranked] == [node.id for node in nodes]
            evidence = service.last_query_relevance
            assert evidence.passes(ranked[2]) == (threshold == "0.35")
            assert not evidence.passes(ranked[3])
            assert [entry.get("node", entry)["id"] for entry in first["results"]] == selected
            expected_deliveries = ["full", "full" if threshold == "0.35" else "twin_duplicate"]
            assert [entry["delivery"] for entry in first["results"]] == expected_deliveries
            residual = [node.id for node in nodes if node.id not in selected]
            assert [r.node.id for r in service.last_residual] == residual
            event = store.get_recall_event(first["recall_event_id"])
            assert event.result_ids == selected
            assert [store.get_node(node.id).access_count for node in nodes] == [
                int(node.id in selected) for node in nodes
            ]
            recorded_map = store.recent_recall_map_history()[0]["recall_map"]
            assert recorded_map["sel"]["n"] == len(residual)
            assert first["recall_map"] == {
                key: value for key, value in recorded_map.items()
                if view == "full" or key != "sel"
            }
            full_ids = [node_id for node_id, delivery in zip(selected, expected_deliveries)
                        if delivery == "full"]
            second = (await client.call_tool("memory_recall", {
                **args, "used": full_ids, "irrelevant": residual,
            })).structured_content
            assert second["feedback_marks"] == {
                "accepted": len(full_ids), "dropped": len(residual),
                "by_reason": {"not_delivered": len(residual)},
            }
            assert [entry.get("node", entry)["id"] for entry in second["results"]] == selected
            assert [entry["delivery"] for entry in second["results"]] == [
                "session_duplicate" if delivery == "full" else delivery
                for delivery in expected_deliveries
            ]
            assert not store.connection.in_transaction

    try:
        asyncio.run(scenario())
    finally:
        mcp.auto_consolidation.close()
        store.close()
