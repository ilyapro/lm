from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from living_memory.embeddings import EMBEDDING_DIMENSIONS, LocalEmbeddingModel, cosine_similarity
from living_memory.retrieval import memory_recall
from living_memory.storage import MemoryStore


def test_default_embedding_uses_hash_fallback_when_model_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[Any] = []

    class FakeSentenceTransformer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))
            raise AssertionError("uncached default model must not be loaded")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hf-hub"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-hub-alt"))
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(tmp_path / "sentence-transformers"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "transformers"))
    monkeypatch.delenv("LIVING_MEMORY_EMBEDDING_BACKEND", raising=False)

    model = LocalEmbeddingModel()
    first = model.embed("authentication timeout in production")
    second = model.embed("authentication timeout in production")

    assert calls == []
    assert len(first) == EMBEDDING_DIMENSIONS
    assert first == second
    assert cosine_similarity(first, second) == pytest.approx(1.0)


def test_local_model_path_loads_under_offline_huggingface_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    calls: list[tuple[str, str | None, str | None]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            calls.append(
                (
                    model_name,
                    os.environ.get("HF_HUB_OFFLINE"),
                    os.environ.get("TRANSFORMERS_OFFLINE"),
                )
            )

        def get_sentence_embedding_dimension(self) -> int:
            return 3

        def encode(self, _text: str, normalize_embeddings: bool = True) -> list[float]:
            assert normalize_embeddings is True
            return [1.0, 2.0, 2.0]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    model = LocalEmbeddingModel(model_name=str(model_dir))
    vector = model.embed("anything")

    assert calls == [(str(model_dir), "1", "1")]
    assert vector == pytest.approx([1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])
    assert os.environ.get("HF_HUB_OFFLINE") is None
    assert os.environ.get("TRANSFORMERS_OFFLINE") is None


def test_default_model_loads_existing_huggingface_cache_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = (
        tmp_path
        / "hf"
        / "hub"
        / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
        / "snapshots"
        / "abc123"
    )
    snapshot.mkdir(parents=True)
    calls: list[str] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            calls.append(model_name)

        def get_sentence_embedding_dimension(self) -> int:
            return 3

        def encode(self, _text: str, normalize_embeddings: bool = True) -> list[float]:
            assert normalize_embeddings is True
            return [0.0, 3.0, 4.0]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)

    model = LocalEmbeddingModel()
    vector = model.embed("cached model text")

    assert calls == [str(snapshot)]
    assert vector == pytest.approx([0.0, 0.6, 0.8])


def test_default_recall_embedding_works_offline_when_dependency_is_installed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[Any] = []

    class FakeSentenceTransformer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))
            raise AssertionError("recall should use fallback instead of downloading")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hf-hub"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-hub-alt"))
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(tmp_path / "sentence-transformers"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "transformers"))
    monkeypatch.delenv("LIVING_MEMORY_EMBEDDING_BACKEND", raising=False)

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        trace = store.append_trace(
            "Authentication slowdown in production login",
            {"scope": "project:alpha", "agent": "agent-a"},
        )

        results = memory_recall(store, "signin slow prod", scope="project:alpha", max_results=3)

        assert results
        assert results[0].node.id == trace.id
        assert store.get_node(trace.id).embedding is not None

    assert calls == []
