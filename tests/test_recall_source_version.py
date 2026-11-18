"""Source-version membership at the MCP boundary and in cached scope blocks."""

import asyncio
import hashlib

import pytest
from fastmcp import Client

from living_memory.chunking import chunk_text
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.health_audit import ReadOnlyAuditStore
from living_memory.retrieval import LAZY_CHUNK_NODE_LIMIT, MemoryRecallService
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore


QUERY = "asynchronous telemetry calibration"
NEW = "violet porcelain teacup"


@pytest.fixture(autouse=True)
def local_policy(monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setenv("LM_RECALL_MAP", "0")


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("late_windows", [False, True])
def test_recall_across_source_edit(warm, late_windows):
    model = LocalEmbeddingModel(backend="hash")
    server = create_mcp_server(":memory:")
    store = server.memory_store
    target = store.create_node(level="concept", content=QUERY, embedding=model.embed(QUERY))
    old_window, = store.list_node_chunks(target.id)

    async def scenario():
        async with Client(server) as client:
            async def recall(query):
                response = await client.call_tool("memory_recall", {
                    "query": query, "depth": 0, "view": "full",
                })
                return [r for r in response.data["results"] if r["node"]["id"] == target.id]

            if warm:
                before, = await recall(QUERY)
                assert before["vector_score"] == pytest.approx(1)
            for i in range(40):
                store.create_node(level="trace", content=f"quiet river {i}",
                                  timestamp="2000-01-01T00:00:00Z")
            store.update_node(target.id, content=NEW)
            assert not store.list_node_chunks(target.id)
            if late_windows:
                store.replace_node_chunks(target.id, [(chunk_text(QUERY)[0], model.embed(QUERY))],
                                          content_fingerprint=old_window.content_fingerprint)
            assert await recall(QUERY) == []
            assert not store.node_chunks_are_current(target.id)
            assert store.count_node_chunks() == LAZY_CHUNK_NODE_LIMIT + int(late_windows)
            # Normal bounded maintenance eventually reaches the edited source.
            for _ in range(40 // LAZY_CHUNK_NODE_LIMIT + 1):
                assert await recall(QUERY) == []
            after, = await recall(NEW)
            assert after["node"]["content"] == NEW
            assert after["vector_score"] == pytest.approx(1)
            assert store.node_chunks_are_current(target.id, encoder_id=model.encoder_id())

    try:
        asyncio.run(scenario())
    finally:
        server.auto_consolidation.close()
        store.close()


@pytest.mark.parametrize("side", ["source", "chunk"])
@pytest.mark.parametrize("old_format", [False, True])
def test_fingerprint_only_transitions_on_live_reader(tmp_path, side, old_format):
    model = LocalEmbeddingModel(backend="hash")
    path = tmp_path / "history.sqlite3"
    with MemoryStore(path) as writer:
        node = writer.create_node(level="concept", content=NEW, embedding=model.embed(QUERY))
        # A matching version can provide semantic evidence without coordinates.
        with writer.transaction():
            writer.connection.execute("UPDATE node_chunk_embeddings SET char_start=NULL, char_end=NULL")
            if old_format:
                writer.connection.executescript("""
                    DROP TRIGGER chunk_corpus_update;
                    DROP TRIGGER chunk_corpus_membership;
                    UPDATE metadata SET value='2' WHERE key='chunk_scope_revision_format';
                """)
        with ReadOnlyAuditStore.open_path(path) as reader:
            service = MemoryRecallService(reader, embedder=model)

            def recall(query):
                return service.memory_recall(query, depth=0, log_event=False, log_access=False)

            assert recall(QUERY)[0].vector_score == pytest.approx(1)
            table = "nodes" if side == "source" else "node_chunk_embeddings"
            for fingerprint in ("older", hashlib.sha256(NEW.encode()).hexdigest()):
                with writer.transaction():
                    writer.connection.execute(f"UPDATE {table} SET content_fingerprint=?", (fingerprint,))
                if fingerprint == "older":
                    assert not recall(QUERY)
                    lexical, = recall(NEW)
                    assert lexical.node_id == node.id and lexical.bm25_score > 0 and lexical.vector_score == 0
                else:
                    assert recall(QUERY)[0].vector_score == pytest.approx(1)
            if old_format:
                assert reader.chunk_corpus_revisions() is None and not service._chunk_index
