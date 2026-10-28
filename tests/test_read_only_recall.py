"""Read-only retrieval needs no writer preparation; pending writes still repair."""

from threading import RLock

import pytest

from living_memory import retrieval
from living_memory.chunking import chunk_text
from living_memory.health_audit import ReadOnlyAuditStore, hot_recall_latency
from living_memory.resources import memory_health
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore


QUERY = "orbital telemetry drift"
SCOPE = "project:telemetry"
VECTOR = [1.0, 0.0, 0.0]
WRITE_OPERATIONS = {
    "INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE", "DROP", "ALTER",
    "REINDEX", "VACUUM",
}


class Encoder:
    def __init__(self):
        self.texts = []

    def embed(self, text):
        self.texts.append(text)
        return list(VECTOR)

    def embed_batch(self, texts):
        return [self.embed(text) for text in texts]


@pytest.fixture
def encoder(monkeypatch):
    model = Encoder()
    monkeypatch.setattr(retrieval, "LocalEmbeddingModel", lambda **kwargs: model)
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    return model


def recall(service):
    return service.memory_recall(
        QUERY, scope=SCOPE, log_access=False, log_event=False,
    )


def assert_no_writes(connection, before, changes, statements):
    assert connection.total_changes == changes
    assert connection.serialize() == before
    assert not [sql for sql in statements if sql.split()[0].upper() in WRITE_OPERATIONS]


@pytest.mark.parametrize("schema", ["current", "without-revisions", "without-chunks"])
@pytest.mark.parametrize("population", ["empty", "indexed", "pending"])
@pytest.mark.parametrize("connection_kind", ["borrowed", "path"])
def test_read_only_recall_and_real_latency_do_not_write(
    tmp_path, encoder, schema, population, connection_kind,
):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path, chunk_embedder=encoder.embed_batch) as store:
        node = None
        if population != "empty":
            node = store.create_node(
                level="trace", content=QUERY, context={"scope": SCOPE},
                embedding=VECTOR if population == "indexed" else None,
            )
        if schema != "current":
            with store.connection:
                for suffix in ("insert", "delete", "update", "membership", "node_insert", "node_delete"):
                    store.connection.execute(f"DROP TRIGGER chunk_corpus_{suffix}")
                store.connection.execute("DELETE FROM metadata WHERE key LIKE 'chunk_scope_revision%'")
                if schema == "without-chunks":
                    store.connection.execute("DROP TABLE node_chunk_embeddings")
        reader = (
            ReadOnlyAuditStore.from_store(store) if connection_kind == "borrowed"
            else ReadOnlyAuditStore.open_path(path)
        )
        with reader:
            connection = reader.connection
            before, changes = connection.serialize(), connection.total_changes
            statements = []
            connection.set_trace_callback(statements.append)
            try:
                service = MemoryRecallService(reader, embedder=encoder)
                for _ in range(2):
                    results = recall(service)
                    assert [result.node_id for result in results] == ([node.id] if node else [])
                if schema != "current":
                    assert reader.chunk_corpus_revisions() is None
                    assert service._chunk_index == {}
                latency = hot_recall_latency(reader, query=QUERY, scope=SCOPE, samples=1)
                assert latency["samples"] == len(latency["samples_ms"]) == 1
                assert latency["warmup_ms"] >= 0
                assert latency["samples_ms"][0] >= 0
                assert_no_writes(connection, before, changes, statements)
            finally:
                connection.set_trace_callback(None)
        # Closing an adapter must leave the borrowed owner's connection usable.
        assert store.connection.execute("SELECT 1").fetchone()[0] == 1
        if node:
            assert store.get_node(node.id) == node


def test_memory_health_latency_leaves_writable_pending_work_untouched(tmp_path, encoder):
    with MemoryStore(tmp_path / "memory.sqlite3", chunk_embedder=encoder.embed_batch) as store:
        node = store.append_trace(QUERY, context={"scope": SCOPE})
        before, changes = store.connection.serialize(), store.connection.total_changes
        statements = []
        store.connection.set_trace_callback(statements.append)
        try:
            report = memory_health(store, latency_samples=1, latency_query=QUERY)
            assert report["latency"]["samples"] == 1
            assert_no_writes(store.connection, before, changes, statements)
        finally:
            store.connection.set_trace_callback(None)
        assert store.get_node(node.id) == node
        assert [item.id for item in store.list_unchunked_nodes()] == [node.id]
        assert store.count_node_chunks(node.id) == 0


@pytest.mark.parametrize("drop_embedding_column", [False, True])
def test_writer_preparation_waits_for_work_and_preserves_guard(
    tmp_path, encoder, monkeypatch, drop_embedding_column,
):
    guard = RLock()
    with MemoryStore(tmp_path / "memory.sqlite3", chunk_embedder=encoder.embed_batch) as store:
        if drop_embedding_column:
            store.drop_node_embedding_column()
        shares = []
        original_share = store.set_chunk_embedder

        def share(batch):
            assert guard._is_owned()
            shares.append(batch)
            original_share(batch)

        original_encode = encoder.embed

        def encode(text):
            assert not guard._is_owned()
            assert not store.connection.in_transaction
            return original_encode(text)

        monkeypatch.setattr(store, "set_chunk_embedder", share)
        monkeypatch.setattr(encoder, "embed", encode)
        service = MemoryRecallService(store, embedder=encoder)
        service.refresh_pending_chunks(scope=SCOPE, guard=lambda: guard)
        assert shares == []
        assert encoder.texts == []

        content = (QUERY + " describes the satellite measurements. ") * 100
        node = store.append_trace(content, context={"scope": SCOPE})
        assert store.count_node_chunks(node.id) == 0
        service.refresh_pending_chunks(scope=SCOPE, guard=lambda: guard)
        assert shares == [encoder.embed_batch]
        assert store._resolve_chunk_embedder() == encoder.embed_batch
        chunks = chunk_text(content)
        assert len(chunks) > 1
        assert store.count_node_chunks(node.id) == len(chunks)
        assert store.node_chunks_are_current(node.id)
        assert encoder.texts == [content, *(chunk.text for chunk in chunks)]

        before, changes = store.connection.serialize(), store.connection.total_changes
        encoder.texts.clear()
        service.refresh_pending_chunks(scope=SCOPE, guard=lambda: guard)
        assert encoder.texts == []
        assert [result.node_id for result in recall(service)] == [node.id]
        assert shares == [encoder.embed_batch]
        assert store.connection.total_changes == changes
        assert store.connection.serialize() == before
