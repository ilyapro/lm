"""Ordinary recall maps use the coordinate system recorded on each current chunk."""

import asyncio

import pytest
from fastmcp import Client

from living_memory.embeddings import EncodedVector
from living_memory.retrieval import LAZY_CHUNK_NODE_LIMIT
from living_memory.server import create_mcp_server


@pytest.mark.parametrize("rotate", [False, True], ids=["shared-space", "permuted-space"])
def test_mcp_map_keeps_topics_and_medoids_after_reopen(tmp_path, monkeypatch, rotate):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
    path = tmp_path / "map.db"
    scope = "project:space-fixture"
    query = "benchmemo"
    topics = ("canary rollout deployment", "baking sourdough loaves")

    def vector(axis, encoder):
        return EncodedVector([float(i == axis) for i in range(384)], encoder)

    def seed(store):
        for i in range(80):
            store.append_trace(f"quarterly ledger reconciliation {i}")
        # Keep both ordinary lazy repair passes on older sources.
        for i in range(2 * LAZY_CHUNK_NODE_LIMIT + 1):
            store.create_node(level="trace", content=f"historical marigolds near beehives {i}",
                              context={"scope": scope}, timestamp="2025-01-01T00:00:00Z",
                              embedding=vector(2, "fixture:guard"))
        for topic, text in enumerate(topics):
            swapped = rotate and topic == 1
            for i in range(3):
                store.create_node(level="trace", node_id=f"note-{topic}-{i}",
                                  content=f"{query} {text} record {i}", context={"scope": scope},
                                  timestamp="2025-02-01T00:00:00Z",
                                  embedding=vector(0 if swapped else topic,
                                                   "fixture:rotated" if swapped else "fixture:base"))

    async def recall(reopen):
        server = create_mcp_server(path)
        store = server.memory_store
        try:
            if not reopen:
                seed(store)
            sql = """SELECT id, encoder_id FROM node_chunk_embeddings
                     WHERE node_id LIKE 'note-%' ORDER BY node_id"""
            before = list(store.connection.execute(sql))
            async with Client(server) as client:
                raw = await client.call_tool("memory_recall", {
                    "scope": scope, "query": query, "max_results": 1,
                })
            assert len(before) == 6 and list(store.connection.execute(sql)) == before
            assert store.connection.execute("SELECT count(*) FROM query_anchors").fetchone()[0] == 0
            response = raw.structured_content
            assert response["results"][0]["id"] == "note-0-0"
            nav = response["recall_map"]
            assert nav["covered"] == nav["pool"] == 5
            groups = {group["count"]: group for group in nav["clusters"]}
            assert len(nav["clusters"]) == 2
            for topic, size in enumerate((2, 3)):
                group = groups[size]
                assert group["medoid"]["node_id"].startswith(f"note-{topic}-")
                assert set(group["ask_hint"].split()) == set(topics[topic].split())
            return nav
        finally:
            server.auto_consolidation.close()
            store.close()

    cold = asyncio.run(recall(False))
    assert asyncio.run(recall(True)) == cold
