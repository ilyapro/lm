"""Learned graph and negative evidence follow the encoder that produced a query."""

import pytest

from living_memory import embeddings, server
from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity
from living_memory.retrieval import MemoryRecallService
from living_memory.server import create_mcp_server
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
        assert (used in delivered, active, control) == (False, [control[0], active[1]], control)
        assert active[0] == pytest.approx(control[0])
        assert active[1] < control[1]
    finally:
        mcp.auto_consolidation.close()
        store.close()
