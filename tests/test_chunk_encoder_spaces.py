"""A chunk contributes a cosine only in its producer's coordinate space."""

import pytest

from living_memory import embeddings, retrieval
from living_memory.chunking import chunk_text
from living_memory.embeddings import EncodedVector, LocalEmbeddingModel
from living_memory.health_audit import ReadOnlyAuditStore
from living_memory.retrieval import LAZY_CHUNK_NODE_LIMIT, MemoryRecallService
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore
from test_mcp_server import FakeMCP


SOURCE = "PostgreSQL migration failed"
QUERY = "database schema failure"


def read(service, query, *, repair=True):
    return service.memory_recall(query, depth=0, max_results=50, repair_pending=repair,
                                 log_event=False, log_access=False)


@pytest.mark.parametrize("numpy", [True, False])
def test_same_width_mixed_windows_and_prepared_query(tmp_path, monkeypatch, numpy):
    if not numpy:
        monkeypatch.setattr(retrieval, "_np", None)
    model = LocalEmbeddingModel(backend="hash")
    with MemoryStore(tmp_path / "chunks.db") as store:
        service = MemoryRecallService(store, embedder=model)
        prepared = service.prepare_query(QUERY)
        vector = model.embed(QUERY)
        node = store.append_trace(QUERY + ". " + "A long technical note. " * 200)
        chunks = chunk_text(node.content)
        assert len(chunks) > 1
        store.replace_node_chunks(node.id, [
            (chunk, vector if i else EncodedVector(vector, "another-space"))
            for i, chunk in enumerate(chunks)
        ])
        # Both sets point in the query's numerical direction. Only a
        # compatible row may win, even if the producer changes after prepare.
        monkeypatch.setattr(model, "embed", lambda _: pytest.fail("query re-encoded"))
        scored, = read(service, prepared, repair=False)
        assert scored.vector_score > 0
        assert scored.semantic_passage.char_start in {chunk.char_start for chunk in chunks[1:]}
        assert store.list_unchunked_node_positions(encoder_id=vector.encoder_id)
        with store.transaction():
            store.connection.execute("UPDATE node_chunk_embeddings SET encoder_id=NULL")
        lexical, = read(service, prepared, repair=False)
        assert lexical.vector_score == 0 and lexical.bm25_score > 0
        assert lexical.semantic_passage is None


def test_unknown_history_opens_unchanged_then_repairs_only_bounded_sources(tmp_path):
    path = tmp_path / "history.db"
    model = LocalEmbeddingModel(backend="hash")
    with MemoryStore(path) as store:
        nodes = [store.create_node(level="trace", content=f"{SOURCE} {i}",
                                   embedding=list(model.embed(QUERY)))
                 for i in range(LAZY_CHUNK_NODE_LIMIT + 2)]
        conn = store.connection
        original = list(conn.execute("SELECT id, embedding FROM node_chunk_embeddings ORDER BY id"))
        # Old snapshots lack both the column and the revision tracking it.
        conn.executescript("""
            DROP TRIGGER chunk_corpus_update;
            DROP INDEX idx_chunk_source_space;
            ALTER TABLE node_chunk_embeddings DROP COLUMN encoder_id;
            UPDATE metadata SET value='1' WHERE key='chunk_scope_revision_format';
        """)
    with ReadOnlyAuditStore.open_path(path) as audit:
        assert all(r.vector_score == 0 for r in read(MemoryRecallService(audit, embedder=model), SOURCE))
    with MemoryStore(path) as store:
        assert original == list(store.connection.execute(
            "SELECT id, embedding FROM node_chunk_embeddings ORDER BY id"))
        service = MemoryRecallService(store, embedder=model)
        assert read(service, QUERY, repair=False) == []
        results = read(service, QUERY)
        assert results and all(r.vector_score > 0 for r in results)
        changed = store.connection.execute(
            "SELECT count(*) FROM node_chunk_embeddings WHERE encoder_id IS NOT NULL"
        ).fetchone()[0]
        assert changed == LAZY_CHUNK_NODE_LIMIT
        for node in nodes:
            assert store.get_node(node.id).embedding == node.embedding
        read(service, QUERY)
        assert not store.list_unchunked_node_positions(encoder_id=model.encoder_id())


def test_prompt_tool_and_subsequent_remember_share_fallback(tmp_path, monkeypatch):
    class Backing:
        def get_sentence_embedding_dimension(self):
            return 384

        def encode(self, texts, **kwargs):
            if texts == QUERY:
                raise ValueError("query encoder failure")
            row = [1.0] + [0.0] * 383
            return [row for _ in texts] if isinstance(texts, list) else row

    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "offline")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setitem(embeddings._MODEL_CACHE, (embeddings.DEFAULT_EMBEDDING_MODEL, "offline"), Backing())
    server = create_mcp_server(tmp_path / "server.db", mcp_factory=FakeMCP)
    store = server.memory_store
    try:
        remember = server.tools["memory_remember"]
        node_id = remember(SOURCE)["node"]["id"]
        before, = store.list_node_chunks(node_id)
        assert before.encoder_id.startswith("st:")
        recall = server.tools["memory_recall"]
        target, = recall(QUERY, view="full", depth=0)["results"]
        assert target["node"]["id"] == node_id
        assert target["bm25_score"] == 0 and target["vector_score"] > 0
        after, = store.list_node_chunks(node_id)
        assert after.encoder_id == "hash:384" and after.id != before.id
        server.prompts["memory://prompt/retrieval_context"](task=QUERY)
        again, = recall(QUERY, view="full", depth=0)["results"]
        assert again["vector_score"] == target["vector_score"]
        assert store.list_node_chunks(node_id) == [after]
        new = remember("A second source")["node"]["id"]
        assert store.list_node_chunks(new)[0].encoder_id == "hash:384"
    finally:
        server.auto_consolidation.close()
        store.close()


@pytest.mark.parametrize("numpy", [True, False])
def test_first_semantic_recall_confirms_off_page_unknown_sources(tmp_path, monkeypatch, numpy):
    if not numpy:
        monkeypatch.setattr(retrieval, "_np", None)
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    path = tmp_path / "history.db"
    model = LocalEmbeddingModel(backend="hash")
    # No exact FTS hit or synonym expansion. Plural and singular have the same hash features.
    query, content = "orchestrator", "orchestrators"
    with MemoryStore(path) as store:
        for i in range(4 * LAZY_CHUNK_NODE_LIMIT):
            store.create_node(level="trace", content=f"blue pottery {i}",
                              embedding=list(model.embed(f"blue pottery {i}")))
        decoy = store.create_node(level="trace", content="violin", embedding=list(model.embed(query)))
        target = store.create_node(level="trace", content=content, embedding=list(model.embed(content)))
        assert len(store.list_unchunked_node_positions(encoder_id=model.encoder_id())) == 4 * LAZY_CHUNK_NODE_LIMIT + 2
        assert not store.search_content_ids(query)
    for first_call in (True, False):
        server = create_mcp_server(path, mcp_factory=FakeMCP)
        store = server.memory_store
        try:
            before = {chunk.id for chunk in store.list_node_chunks(target.id)}
            result = server.tools["memory_recall"](query)
            hit = result["results"][0]
            assert hit["id"] == target.id and hit["content"] == content
            event = store.get_recall_event(result["recall_event_id"])
            assert event.results[0]["bm25_score"] == 0 and event.results[0]["methods"] == ["vector"]
            assert not any(r["id"] == decoy.id and r["content"] for r in result["results"])
            chunk, = store.list_node_chunks(target.id)
            assert chunk.encoder_id == model.encoder_id()
            if first_call:
                assert chunk.id not in before
                assert store.node_chunks_are_current(decoy.id, encoder_id=model.encoder_id())
                assert store.connection.execute(
                    "SELECT COUNT(*) FROM node_chunk_embeddings WHERE encoder_id IS NOT NULL"
                ).fetchone()[0] == LAZY_CHUNK_NODE_LIMIT
            else:
                assert chunk.id in before
        finally:
            server.auto_consolidation.close()
            store.close()
