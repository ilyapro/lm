"""The MCP projection preserves ranked delivery while direct recall exposes vectors."""

import asyncio
from dataclasses import asdict
from itertools import count
from shutil import copyfile

from fastmcp import Client

from living_memory import storage
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.retrieval import MemoryRecallService
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore


def test_direct_and_mcp_candidate_embeddings(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
    monkeypatch.setenv("LM_RECALL_MAP", "1")
    monkeypatch.setattr(storage, "_utc_now", lambda: "2026-01-01T00:00:00Z")
    query = "harbor calibration"
    seed = tmp_path / "seed.db"
    vector = LocalEmbeddingModel().embed(query)
    with MemoryStore(seed) as store:
        for i in range(12):
            store.create_node(level="trace", node_id=f"record-{i}", content=f"{query} {i}",
                              embedding=vector, context={"scope": "project:harbor"})
        store.create_connection("record-0", "record-1", "caused")
        direct = MemoryRecallService(store).memory_recall(query, log_access=False)
        assert direct and all(item.to_dict()["node"]["embedding"] == vector for item in direct)

    async def recall(projected):
        path = tmp_path / f"mcp-{projected}.db"
        copyfile(seed, path)
        mcp = create_mcp_server(path)
        store = mcp.memory_store
        read = store.get_recall_nodes
        hydrated = []

        def observe(ids, plan, *, include_embedding=True):
            nodes = read(ids, plan, include_embedding=include_embedding if projected else True)
            hydrated.extend(node.embedding is not None for node in nodes.values())
            return nodes

        monkeypatch.setattr(store, "get_recall_nodes", observe)
        event_ids = count()
        monkeypatch.setattr(storage, "new_ulid", lambda: f"event-{next(event_ids)}")
        outcomes = []
        try:
            async with Client(mcp) as client:
                for view in ("full", "knowledge"):
                    response = await client.call_tool("memory_recall", {
                        "query": query, "scope": "project:harbor", "depth": 1, "max_results": 3,
                        "view": view, "ambient_context": {"transport_session_id": "same-client"},
                    })
                    payload = dict(response.data)
                    assert payload["count"] == 3 and payload["recall_map"]["pool"] > 0
                    event = store.get_recall_event(payload["recall_event_id"])
                    outcomes.append((payload, asdict(event)))
            assert set(hydrated) == {not projected}
            return outcomes
        finally:
            mcp.auto_consolidation.close()
            store.close()

    assert asyncio.run(recall(True)) == asyncio.run(recall(False))
