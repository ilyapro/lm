"""Batch inference preserves the scalar encoder and consumer mapping contracts."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path
from threading import Event, RLock
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


@pytest.mark.parametrize("output", ["aliased", "numpy", "ragged", "nonfinite"])
def test_batch_detaches_retained_backend_rows_and_matches_scalar_exactly(output) -> None:
    raw = [[3.0, 4.0]] * 3
    if output == "numpy":
        import numpy as np
        raw = np.asarray(raw, dtype=np.float32)
    elif output == "ragged":
        raw = [[], [3.0, 4.0], [], [1.0, 2.0, 2.0], []]
    elif output == "nonfinite":
        raw = [[0.0, -0.0], [float("inf"), 1.0], [float("nan"), 0.0]]
    retained = repr(raw)
    texts = [str(index) for index in range(len(raw))]

    class Encoder:
        def encode(self, texts, **kwargs):
            return raw if isinstance(texts, list) else raw[int(texts)]

    scalar = model_with(Encoder())
    expected = [[value.hex() for value in scalar.embed(text)] for text in texts]
    model = model_with(Encoder())
    initial_dimensions = model.dimensions
    result = model.embed_batch(texts)
    assert [[value.hex() for value in row] for row in result] == expected
    assert model.dimensions == scalar.dimensions
    assert result is not raw
    assert len({id(row) for row in result}) == len(result)
    assert all(row is not source for row in result for source in raw)
    result[0][0] = 999.0
    assert [[value.hex() for value in row] for row in result[1:]] == expected[1:]
    assert repr(raw) == retained
    # Empty rows use the dimension reached at that point in this call.
    model.dimensions = initial_dimensions
    again = model.embed_batch(texts)
    assert [[value.hex() for value in row] for row in again] == expected
    assert all(row is not previous for row in again for previous in result)
    assert repr(raw) == retained
    assert texts == [str(index) for index in range(len(raw))]


def test_batch_normalization_error_does_not_retry_scalar_or_switch_encoder(monkeypatch) -> None:
    encoder = BatchEncoder()
    model = model_with(encoder)
    normalize = model._normalize_embedding
    calls = 0

    def fail_second(vector):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ArithmeticError("normalization failed")
        return normalize(vector)

    monkeypatch.setattr(model, "_normalize_embedding", fail_second)
    with pytest.raises(ArithmeticError, match="normalization failed"):
        model.embed_batch(TEXTS)
    assert calls == 2
    assert encoder.calls == [TEXTS]
    assert model.dimensions == 3
    assert model.encoder_id().startswith("st:")


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


@pytest.mark.parametrize("transition", [
    "stable", "scalar", "before_batch", "during_batch", "before_identity", "before_write",
])
def test_consolidation_cache_keeps_vector_identity_across_preparation(tmp_path, monkeypatch, transition):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")

    class Encoder(BatchEncoder):
        def encode(self, texts, **kwargs):
            if isinstance(texts, list) and transition in ("scalar", "during_batch"):
                raise TypeError("scalar backend")
            if texts == "fail" or (transition == "during_batch" and texts == "Harbor record 2"):
                raise RuntimeError("encoder unavailable")
            return super().encode(texts, **kwargs)

    backend = Encoder()
    model = model_with(backend)
    model.dimensions = 3
    initial_identity = model.encoder_id()
    reference = model_with(BatchEncoder())
    hashed = LocalEmbeddingModel(dimensions=3, backend="hash")
    guard = RLock()

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        nodes = [store.append_trace(f"Harbor record {i}") for i in range(10)]
        expected = {node.id: reference.embed(node.content) for node in nodes}
        store.put_consolidation_embeddings([(nodes[0], expected[nodes[0].id])], initial_identity)
        steps = 0
        batch = model.embed_batch

        def encode(texts):
            assert not guard._is_owned()
            assert not store.connection.in_transaction
            result = batch(texts)
            if transition == "before_identity" and texts[-1] == nodes[-1].content:
                model.embed("fail")
            return result

        monkeypatch.setattr(model, "embed_batch", encode)

        @contextmanager
        def step():
            nonlocal steps
            steps += 1
            if steps == 2 and transition == "before_write":
                model.embed("fail")
            with guard:
                yield
            if steps == 1 and transition == "before_batch":
                model.embed("fail")

        vectors = consolidation._trace_embeddings(store, nodes, model, step)
        first_hash = {"before_batch": 1, "during_batch": 2}.get(transition, len(nodes))
        for node in nodes[first_hash:]:
            expected[node.id] = hashed.embed(node.content)
        assert vectors == expected
        cached = store.get_consolidation_embeddings(nodes, initial_identity)
        if transition in ("stable", "scalar", "before_write"):
            assert cached == vectors
        else:
            assert cached == {nodes[0].id: vectors[nodes[0].id]}
        if transition in ("stable", "scalar"):
            assert model.encoder_id() == initial_identity
            calls = list(backend.calls)
            assert consolidation._trace_embeddings(store, nodes, model, lambda: guard) == vectors
            assert backend.calls == calls
        else:
            assert model.encoder_id() == "hash:3"
            assert store.get_consolidation_embeddings(nodes, model.encoder_id()) == {}
            expected_hash = {node.id: hashed.embed(node.content) for node in nodes}
            assert consolidation._trace_embeddings(store, nodes, model, lambda: guard) == expected_hash
            assert store.get_consolidation_embeddings(nodes, model.encoder_id()) == expected_hash


def test_consolidation_cache_rejects_changed_content_after_encoding_gap(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    entered, release = Event(), Event()

    class Encoder(BatchEncoder):
        def encode(self, texts, **kwargs):
            entered.set()
            assert release.wait(30)
            return super().encode(texts, **kwargs)

    model = model_with(Encoder())
    model.dimensions = 3
    identity = model.encoder_id()
    guard = RLock()
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        # Isolate content indexing from the whole-content consolidation encoder.
        store.set_chunk_embedder(lambda texts: [[0.0, 1.0, 0.0] for _ in texts])
        source = store.append_trace("Harbor checks the original manifest.")
        with ThreadPoolExecutor(max_workers=2) as workers:
            work = workers.submit(consolidation._trace_embeddings, store, [source], model, lambda: guard)

            def edit():
                with guard:
                    # Derived sources may change content while retaining their id.
                    store.update_node(source.id, level="concept")
                    return store.update_node(source.id, content="Harbor checks the revised manifest.")

            try:
                assert entered.wait(10)
                current = workers.submit(edit).result(timeout=10)
                assert not work.done()
            finally:
                release.set()
            vectors = work.result(timeout=15)
        assert store.get_consolidation_embeddings([source], identity) == vectors
        assert store.get_consolidation_embeddings([current], identity) == {}
        refreshed = consolidation._trace_embeddings(store, [current], model, lambda: guard)
        assert refreshed == {current.id: model.embed(current.content)}
        assert refreshed != vectors
        assert store.get_consolidation_embeddings([current], identity) == refreshed
        assert store.get_consolidation_embeddings([source], identity) == {}


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


@pytest.mark.parametrize("stage", ["load", "encode", "fallback"])
def test_shared_encoder_serializes_loading_inference_and_fallback(tmp_path, monkeypatch, stage):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, RLock
    from types import SimpleNamespace
    import os
    import sys

    from living_memory import embeddings

    entered, release, attempted = Event(), Event(), Event()

    class ObservedLock:
        def __init__(self):
            self.lock = RLock()

        def __enter__(self):
            if not self.lock.acquire(blocking=False):
                attempted.set()
                self.lock.acquire()
            return self

        def __exit__(self, *args):
            self.lock.release()

    loads = []
    calls = []

    class Encoder:
        def __init__(self, source):
            loads.append(source)
            if stage == "load":
                entered.set()
                assert release.wait(10)

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, texts, **kwargs):
            calls.append(texts)
            if stage != "load" and not entered.is_set():
                entered.set()
                assert release.wait(10)
            if stage == "fallback":
                raise RuntimeError("encoder unavailable")
            return [[1.0, 2.0, 2.0] for _ in texts] if isinstance(texts, list) else [1.0, 2.0, 2.0]

    model_dir = tmp_path / "encoder"
    model_dir.mkdir()
    monkeypatch.setattr(embeddings, "_ENCODER_LOCK", ObservedLock())
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Encoder))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    first = LocalEmbeddingModel(model_name=str(model_dir), backend="auto")
    # Separate instances share the cached backend; fallback is instance state.
    second = first if stage == "fallback" else LocalEmbeddingModel(model_name=str(model_dir), backend="auto")
    with ThreadPoolExecutor(max_workers=2) as workers:
        batch = workers.submit(first.embed_batch, ["first", "second"])
        try:
            assert entered.wait(5)
            scalar = workers.submit(second.embed, "next")
            assert attempted.wait(5)
            assert not scalar.done()
            if stage == "load":
                assert calls == []
            else:
                assert calls == [["first", "second"]]
        finally:
            release.set()
        vectors, vector = batch.result(timeout=5), scalar.result(timeout=5)
    assert loads == [str(model_dir)]
    assert first.dimensions == second.dimensions == 3
    if stage == "fallback":
        reference = LocalEmbeddingModel(dimensions=3, backend="hash")
        assert vectors == reference.embed_batch(["first", "second"])
        assert vector == reference.embed("next")
        assert first.encoder_id() == "hash:3"
    else:
        assert vectors == [[1 / 3, 2 / 3, 2 / 3]] * 2
        assert vector == [1 / 3, 2 / 3, 2 / 3]
        assert first.encoder_id() == second.encoder_id()
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
