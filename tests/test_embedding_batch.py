"""Batch inference preserves the scalar encoder and consumer mapping contracts."""

from contextlib import nullcontext
from pathlib import Path
from typing import Any
import zlib

import pytest

from living_memory import consolidation, storage
from living_memory.chunking import chunk_text
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.feedback import _apply_query_feedback
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore


TEXTS = ["", "   ", "English deployment note", "Русская заметка", "English deployment note"]


class BatchEncoder:
    def __init__(self) -> None:
        self.calls: list[str | list[str]] = []

    @staticmethod
    def vector(text: str) -> list[float]:
        return [float(zlib.crc32(text.encode()) % 100 + 1), float(len(text)), 2.0]

    def encode(self, texts: str | list[str], **kwargs: Any) -> Any:
        assert kwargs == {"normalize_embeddings": True}
        self.calls.append(texts)
        if isinstance(texts, str):
            return self.vector(texts)
        return [self.vector(text) for text in texts]


def model_with(encoder: Any) -> LocalEmbeddingModel:
    model = LocalEmbeddingModel(model_name="batch-contract-fixture", backend="auto")
    model._model = encoder
    model._load_attempted = True
    return model


@pytest.mark.parametrize("texts", [[], [""], ["one"], TEXTS])
@pytest.mark.parametrize("array_output", [False, True])
def test_batch_matches_scalar_in_order_and_uses_one_encode(texts, array_output) -> None:
    class Encoder(BatchEncoder):
        def encode(self, texts, **kwargs):
            result = super().encode(texts, **kwargs)
            if array_output:
                import numpy as np
                return np.asarray(result, dtype=np.float32)
            return result

    encoder = Encoder()
    model = model_with(encoder)
    expected = [model.embed(text) for text in texts]
    encoder.calls.clear()
    assert model.embed_batch(tuple(texts)) == expected
    assert encoder.calls == ([texts] if texts else [])
    assert all(len(vector) == 3 for vector in expected)


def test_empty_batch_does_not_resolve_model(monkeypatch) -> None:
    model = LocalEmbeddingModel()
    monkeypatch.setattr(model, "_sentence_transformer", lambda: pytest.fail("model resolved"))
    assert model.embed_batch([]) == []


@pytest.mark.parametrize("backend", ["hash", "fallback", "local-hash"])
def test_hash_batch_is_exactly_scalar(backend) -> None:
    model = LocalEmbeddingModel(dimensions=7, backend=backend)
    assert model.embed_batch(TEXTS) == [model.embed(text) for text in TEXTS]
    assert model.embed_batch(["", "  "]) == [[0.0] * 7, [0.0] * 7]


def test_missing_model_uses_scalar_hash_fallback(monkeypatch) -> None:
    model = LocalEmbeddingModel(dimensions=7, backend="auto")
    monkeypatch.setattr(model, "_sentence_transformer", lambda: None)
    expected = LocalEmbeddingModel(dimensions=7, backend="hash")
    assert model.embed_batch(TEXTS) == [expected.embed(text) for text in TEXTS]


def test_backend_without_normalize_keyword_still_batches() -> None:
    class Encoder(BatchEncoder):
        def encode(self, texts):
            return super().encode(texts, normalize_embeddings=True)

    encoder = Encoder()
    model = model_with(encoder)
    expected = [model.embed(text) for text in TEXTS]
    encoder.calls.clear()
    assert model.embed_batch(TEXTS) == expected
    assert encoder.calls == [TEXTS]


@pytest.mark.parametrize("failure", ["unsupported", "wrong_count", "flat", "invalid_value"])
def test_unavailable_batch_retries_scalar_without_switching_encoder(failure) -> None:
    class Encoder(BatchEncoder):
        def encode(self, texts, **kwargs):
            if isinstance(texts, list):
                self.calls.append(texts)
                if failure == "unsupported":
                    raise TypeError("scalar input only")
                if failure == "wrong_count":
                    return [[1.0, 2.0]]
                if failure == "flat":
                    return [1.0] * len(texts)
                return [[object()]] * len(texts)
            return super().encode(texts, **kwargs)

    encoder = Encoder()
    model = model_with(encoder)
    expected = [model.embed(text) for text in TEXTS]
    encoder.calls.clear()
    assert model.embed_batch(TEXTS) == expected
    assert [call for call in encoder.calls if isinstance(call, str)] == TEXTS
    assert model.encoder_id().startswith("st:")


def test_encode_failure_preserves_scalar_prefix_then_hash_transition() -> None:
    class Encoder(BatchEncoder):
        def encode(self, texts, **kwargs):
            if isinstance(texts, list) or texts == "fails":
                raise RuntimeError("encode failed")
            return super().encode(texts, **kwargs)

    texts = ["works", "fails", "after failure"]
    scalar = model_with(Encoder())
    batched = model_with(Encoder())
    assert batched.embed_batch(texts) == [scalar.embed(text) for text in texts]
    assert batched.encoder_id() == scalar.encoder_id() == "hash:3"
    assert batched.embed_batch(["later"]) == [scalar.embed("later")]


def test_empty_backend_vectors_keep_current_dimension() -> None:
    class Encoder:
        def encode(self, texts, **kwargs):
            return [[] for _ in texts] if isinstance(texts, list) else []

    model = model_with(Encoder())
    assert model.embed_batch(TEXTS) == [model.embed(text) for text in TEXTS]
    assert model.dimensions == 384


@pytest.mark.parametrize("wiring", ["store", "retrieval", "consolidation"])
def test_multichunk_write_reaches_one_model_batch(tmp_path: Path, monkeypatch, wiring) -> None:
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    encoder = BatchEncoder()
    model = model_with(encoder)
    monkeypatch.setattr(storage, "LocalEmbeddingModel", lambda **kwargs: model)
    monkeypatch.setattr(consolidation, "LocalEmbeddingModel", lambda **kwargs: model)
    content = " ".join(f"Section {i} explains a distinct deployment step." for i in range(35))
    chunks = chunk_text(content)
    assert len(chunks) > 1
    expected = [model.embed(chunk.text) for chunk in chunks]
    encoder.calls.clear()
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        if wiring == "retrieval":
            MemoryRecallService(store, embedder=model)._share_embedder_with_store()
        elif wiring == "consolidation":
            consolidation.memory_consolidate(store, force=True)
        node = store.create_node(level="trace", content=content, embedding=[1.0, 0.0, 0.0])
        rows = store.list_node_chunks(node.id)
        assert encoder.calls == [[chunk.text for chunk in chunks]]
        assert [row.chunk_index for row in rows] == list(range(len(chunks)))
        for row, vector in zip(rows, expected, strict=True):
            assert row.embedding == pytest.approx(vector, abs=1e-7)


def test_retrieval_accepts_scalar_only_injected_encoder(tmp_path: Path) -> None:
    class ScalarEncoder:
        def embed(self, text):
            return BatchEncoder.vector(text)

    encoder = ScalarEncoder()
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        MemoryRecallService(store, embedder=encoder)._share_embedder_with_store()
        assert store._resolve_chunk_embedder()(TEXTS) == [encoder.embed(text) for text in TEXTS]


def test_consolidation_uses_existing_bounded_batches_and_cache(tmp_path: Path) -> None:
    encoder = BatchEncoder()
    model = model_with(encoder)
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        nodes = [store.append_trace(f"Synthetic trace {i}") for i in range(19)]
        expected = {node.id: model.embed(node.content) for node in nodes}
        encoder.calls.clear()
        assert consolidation._trace_embeddings(store, nodes, model, nullcontext) == expected
        assert encoder.calls == [[n.content for n in nodes[i:i + 8]] for i in (0, 8, 16)]
        encoder.calls.clear()
        assert consolidation._trace_embeddings(store, nodes, model, nullcontext) == expected
        assert encoder.calls == []


def test_feedback_queries_reach_model_in_one_batch(tmp_path: Path, monkeypatch) -> None:
    encoder = BatchEncoder()
    model = model_with(encoder)
    monkeypatch.setattr(storage, "LocalEmbeddingModel", lambda **kwargs: model)
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        nodes = [store.append_trace(f"Synthetic memory {i}") for i in range(2)]
        queries = ["deployment timeout", "исправление конфигурации"]
        events = [store.record_recall_event(query=query, scope="global", results=[{"node_id": node.id}])
                  for query, node in zip(queries, nodes, strict=True)]
        anchors = _apply_query_feedback(store, [
            (event, (node.id,), ()) for event, node in zip(events, nodes, strict=True)
        ])
        assert encoder.calls == [queries]
        assert len(anchors) == 2
        for anchor_id, query, node in zip(anchors, queries, nodes, strict=True):
            anchor = store.get_query_anchor(anchor_id)
            assert anchor.query == query
            assert anchor.embedding == pytest.approx(model.embed(query))
            assert [edge.target_id for edge in store.list_query_anchor_edges(anchor_id=anchor_id)] == [node.id]
