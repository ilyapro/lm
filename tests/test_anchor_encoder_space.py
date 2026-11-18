"""Learned graph and negative evidence follow the encoder that produced a query."""

import sqlite3
from threading import RLock

import pytest

from living_memory import embeddings, retrieval, server
from living_memory.embeddings import EncodedVector, LocalEmbeddingModel, cosine_similarity
from living_memory.health_audit import ReadOnlyAuditStore
from living_memory.query_anchors import match_anchors
from living_memory.retrieval import MemoryRecallService
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore, recall_fingerprint
from test_mcp_server import FakeMCP


def test_feedback_after_encoder_fallback(tmp_path, monkeypatch):
    previous, unrelated = "amber sundial", "redis eviction"
    hash_model = LocalEmbeddingModel(backend="hash")
    collision = hash_model.embed(unrelated)
    assert cosine_similarity(collision, hash_model.embed(previous)) < 0.1

    class Backing:
        def get_sentence_embedding_dimension(self):
            return len(collision)

        def encode(self, texts, **kwargs):
            if texts == unrelated:
                raise ValueError("model unavailable")
            return [list(collision) for _ in texts] if isinstance(texts, list) else list(collision)

    monkeypatch.setitem(embeddings._MODEL_CACHE, (embeddings.DEFAULT_EMBEDDING_MODEL, "offline"), Backing())
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "offline")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    monkeypatch.setattr(server, "_transport_session_id", lambda: "test-client")
    mcp = create_mcp_server(tmp_path / "feedback.db", mcp_factory=FakeMCP)
    store = mcp.memory_store
    try:
        remember = mcp.tools["memory_remember"]
        recall = mcp.tools["memory_recall"]
        used = remember(previous)["node"]["id"]
        marked = remember(previous + " " + unrelated)["node"]["id"]
        recall(previous, depth=0)
        recall(previous, used=[used], irrelevant=[marked], depth=0)
        anchor, = store.list_query_anchors()
        assert anchor.encoder_id.startswith("st:")
        assert cosine_similarity(anchor.embedding, collision) == pytest.approx(1.0)
        assert [(e.target_id, e.hits) for e in store.list_query_anchor_edges()] == [(used, 1)]
        result = recall(unrelated, view="full")
        delivered = {r["node"]["id"]: r for r in result["results"]}
        chunk, = store.list_node_chunks(used)
        assert chunk.encoder_id == collision.encoder_id
        assert cosine_similarity(chunk.embedding, collision) < 0.1
        service = MemoryRecallService(store, embedder=hash_model, anchor_seeding=False)

        def score(text):
            return next(r.score for r in service.memory_recall(text, depth=0, log_access=False)
                        if r.node_id == marked)

        active = [score(q) for q in (unrelated, previous)]
        monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "audit")
        control = [score(q) for q in (unrelated, previous)]
        assert used not in delivered
        assert active[0] == pytest.approx(control[0])
        assert active[1] < control[1]
    finally:
        mcp.auto_consolidation.close()
        store.close()


@pytest.mark.parametrize("reader", ["current", "old-read-only", "upgraded"])
def test_first_historical_match_rescores_text_and_reuses_encoder_work(tmp_path, reader, monkeypatch):
    model = LocalEmbeddingModel(backend="hash")
    query = "database schema failure"
    old_text = "database migration failed"
    path = tmp_path / "anchors.db"
    with MemoryStore(path) as store:
        def learn(text, vector, target, weight, scope="global"):
            anchor = store.insert_query_anchor(
                query=text, embedding=vector, fingerprint=recall_fingerprint(text, scope), scope=scope,
            )
            store.upsert_query_anchor_edge(anchor.id, target, weight=weight)
            return anchor

        for name in ("false", "historical", "current", "other"):
            store.create_node(node_id=name, level="trace", content="opaque target " + name)
        for index in range(12):
            learn(f"pottery workshop {index}", list(model.embed(query)), "false", 1)
        # This row cannot even be nominated by its historical width.
        historical = learn(old_text, [1.0, 0.0], "historical", 0.75)
        learn("schema failure", model.embed("schema failure"), "current", 1)
        learn(old_text, list(model.embed(query)), "other", 1, "project:outside")
        if reader != "current":
            store.connection.executescript("""
                DROP TRIGGER anchor_text_insert;
                DROP TRIGGER anchor_text_update;
                DROP TRIGGER anchor_text_delete;
                DELETE FROM metadata WHERE key LIKE 'anchor_text_revision:%';
                DROP TRIGGER query_anchor_corpus_update;
                ALTER TABLE query_anchors DROP COLUMN encoder_id;
            """)

    open_store = ReadOnlyAuditStore.open_path if reader == "old-read-only" else MemoryStore
    with open_store(path) as store:
        calls = []
        encode = model.embed_batch

        def batch(texts):
            calls.append(list(texts))
            return encode(texts)

        monkeypatch.setattr(model, "embed_batch", batch)
        service = MemoryRecallService(store, embedder=model)
        snapshot = list(store.connection.iterdump())
        for _ in range(2):
            results = service.memory_recall(query, scope="global", depth=1, repair_pending=False,
                                            log_event=False, log_access=False)
            assert {r.node_id for r in results} == {"historical", "current"}
            hit = next(r for r in results if r.node_id == "historical")
            assert hit.bm25_score == hit.vector_score == 0
            assert hit.graph_score == pytest.approx(
                0.75 * cosine_similarity(model.embed(query), model.embed(old_text)))
        assert len(calls) == 1 and 1 <= len(calls[0]) <= 10
        assert old_text in calls[0]
        assert list(store.connection.iterdump()) == snapshot
        assert store.get_query_anchor(historical.id) == historical
        if reader == "old-read-only":
            # Its remaining revision triggers cannot see a query-only edit.
            revision = store.query_anchor_revision()
            with sqlite3.connect(path) as writer:
                writer.execute("UPDATE query_anchors SET query='unrelated text' WHERE id=?", (historical.id,))
            assert store.query_anchor_revision() == revision
            results = service.memory_recall(query, scope="global", repair_pending=False,
                                            log_event=False, log_access=False)
            assert "historical" not in {r.node_id for r in results}


def test_verification_admits_only_rows_from_the_prepared_space(tmp_path):
    guard = RLock()

    class Encoder:
        def embed(self, text):
            return EncodedVector([1.0, 0.0], "first")

        def embed_batch(self, texts):
            assert not guard._is_owned()
            return [{"valid": EncodedVector([1.0, 0.0], "first"),
                     "fallback": EncodedVector([1.0, 0.0], "second"),
                     "missing": [1.0, 0.0],
                     "short": EncodedVector([1.0], "first")}[text] for text in texts]

    with MemoryStore(tmp_path / "mixed.db") as store:
        for text in ("valid", "fallback", "missing", "short"):
            anchor = store.insert_query_anchor(query=text, embedding=[1.0, 0.0],
                                               fingerprint=text, scope="global")
            node = store.create_node(node_id=text, level="trace", content="opaque artifact " + text)
            store.upsert_query_anchor_edge(anchor.id, node.id, weight=1)
        assert match_anchors(store, [1.0, 0.0]) == []
        service = MemoryRecallService(store, embedder=Encoder())
        prepared = service.prepare_anchor_history(service.prepare_query("question"), scope=None,
                                                  ambient_context=None, guard=lambda: guard)
        # Selection under the guard consumes only checked projections, then
        # revalidates their text, identity and liveness in the store.
        def read():
            with guard:
                return [r.node_id for r in service.memory_recall(
                    prepared, repair_pending=False, log_access=False, log_event=False)]
        assert read() == ["valid"]
        with store.transaction():
            store.connection.execute("UPDATE query_anchors SET query='changed' WHERE query='valid'")
        assert read() == []


def test_known_foreign_collisions_cannot_displace_unlabelled_semantic_nominee(tmp_path):
    class Model:
        def embed(self, text):
            direction = [1.0, 0.0] if text in ("needle", "lasso") else [0.0, 1.0]
            return EncodedVector(direction, "current")

    with MemoryStore(tmp_path / "mixed.db") as store:
        node = store.create_node(level="trace", content="opaque target")
        learned = store.insert_query_anchor(query="lasso", fingerprint="lasso", scope="global",
                                            embedding=[0.8, 0.6])
        store.upsert_query_anchor_edge(learned.id, node.id, weight=0.75)
        for i in range(10):
            store.insert_query_anchor(query=f"unrelated {i}", fingerprint=str(i), scope="global",
                                      embedding=EncodedVector([1.0, 0.0], "foreign"))
        service = MemoryRecallService(store, embedder=Model())
        hit, = service.memory_recall("needle", repair_pending=False, log_access=False)
        assert hit.node_id == node.id and hit.graph_score == pytest.approx(0.75)
        assert hit.vector_score == hit.bm25_score == 0


def test_historical_index_reads_and_nominees_match_full_scan(tmp_path, monkeypatch):
    class Model:
        encoder = "active"

        def embed(self, text):
            return EncodedVector([1.0, 0.0], self.encoder)

    path = tmp_path / "anchors.db"
    with MemoryStore(path) as reader, MemoryStore(path) as writer:
        model = Model()
        warm = MemoryRecallService(reader, embedder=model)
        reference = MemoryRecallService(reader, embedder=model)
        # Stream vectors and rebuild postings from every admitted SQL row.
        monkeypatch.setattr(reference, "_anchor_vector_source", lambda *args: reader)
        observed = []
        for service in (warm, reference):
            history = service._historical_anchor_queries

            def capture(*args, history=history):
                selected = history(*args)
                observed.append(list(selected.items()))
                return selected

            monkeypatch.setattr(service, "_historical_anchor_queries", capture)

        def insert(i, encoder, scope):
            vector = [1., 0.] if encoder is None else EncodedVector([1., 0.], encoder)
            anchor = writer.insert_query_anchor(
                anchor_id=str(i), query=f"archived pattern {i}", embedding=vector,
                fingerprint=str(i), scope=scope,
            )
            node = writer.create_node(node_id=f"n{i}", level="trace", content=f"opaque object {i}",
                                      context={"scope": scope})
            writer.upsert_query_anchor_edge(anchor.id, node.id, weight=0.8)

        for i in range(12):
            insert(i, None if i % 3 else "previous", "global" if i % 2 else "project:other")

        scans, tokens = [], []
        read, tokenize = reader.iter_foreign_anchor_queries, retrieval.tokenize

        def scan(*args):
            for row in read(*args):
                scans.append(row[0])
                yield row

        def count_tokens(text, **kwargs):
            if text.startswith("archived "):
                tokens.append(text)
            return tokenize(text, **kwargs)

        monkeypatch.setattr(reader, "iter_foreign_anchor_queries", scan)
        monkeypatch.setattr(retrieval, "tokenize", count_tokens)

        def check(scope, processed, admitted):
            observed.clear()
            for service, expected in ((warm, processed), (reference, admitted)):
                scans.clear()
                tokens.clear()
                result = service.memory_recall("pattern", scope=scope, log_access=False, log_event=False)
                assert len(scans) == len(tokens) == expected
                if service is warm:
                    actual = result
                else:
                    assert result and result == actual
            assert observed[0] and observed[0] == observed[1]

        check(None, 12, 12)
        insert(12, "active", "global")
        check(None, 0, 12)
        check("global", 6, 6)
        check(None, 12, 12)
        check(None, 0, 12)
        model.encoder = "previous"
        check(None, 9, 9)
        check(None, 0, 9)

        # A pinned reader keeps its own membership across a concurrent commit.
        with reader.read_snapshot():
            check(None, 0, 9)
            writer.decay_query_anchor("1", "retired")
            check(None, 0, 9)
        check(None, 8, 8)
        reader.connection.execute("UPDATE query_anchors SET query='archived replacement' WHERE id='2'")
        check(None, 8, 8)
        reader.connection.rollback()
        check(None, 0, 8)
        with writer.transaction():
            writer.connection.execute("UPDATE query_anchors SET query='archived pattern changed' WHERE id='2'")
        check(None, 8, 8)

        # A fingerprint conflict can displace an anchor in a different space.
        with writer.transaction():
            writer.connection.execute("PRAGMA recursive_triggers=OFF")
            writer.connection.execute("DELETE FROM query_anchor_edges WHERE anchor_id='2'")
            writer.connection.execute(
                "UPDATE OR REPLACE query_anchors SET fingerprint='2' WHERE id='0'"
            )
        check(None, 7, 7)
        with writer.transaction():
            writer.connection.execute(
                "UPDATE query_anchors SET encoder_id='previous' WHERE id='4'"
            )
        check(None, 6, 6)
