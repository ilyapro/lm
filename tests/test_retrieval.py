from pathlib import Path

import pytest

from living_memory.retrieval import MemoryRecallService, memory_recall
from living_memory.storage import MemoryStore


def test_recall_uses_bm25_vector_reranking_and_access_logging(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        relevant = store.append_trace(
            "Authentication timeout in production login flow",
            {
                "scope": "project:alpha",
                "agent": "agent-a",
                "task": "incident",
            },
            feedback={"confidence": 0.5, "usefulness_score": 0.2},
        )
        store.append_trace(
            "CSS spacing changed on the billing settings page",
            {"scope": "project:alpha", "agent": "agent-b"},
        )

        for index in range(250):
            store.append_trace(
                f"background trace {index} about exports and reports",
                {"scope": "project:alpha", "agent": "agent-c"},
            )

        results = memory_recall(
            store,
            "login auth slow in prod",
            scope="project:alpha",
            max_results=5,
        )

        assert results
        assert results[0].node.id == relevant.id
        assert {"bm25", "vector"} & set(results[0].methods)
        assert store.get_node(relevant.id).access_count == 1
        assert store.get_node(relevant.id).embedding is not None


def test_feedback_and_confidence_affect_ranking(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        weak = store.append_trace(
            "Database migration fixes deploy failure",
            {"scope": "project:alpha", "agent": "agent-a"},
            feedback={"confidence": 0.3, "usefulness_score": -0.4},
        )
        strong = store.create_node(
            level="concept",
            content="Schema migration resolves deployment failure",
            context={"scope": "project:alpha", "agent": "agent-b"},
            stats={"confidence": 0.95, "unique_agents": 2, "usefulness_score": 0.9},
            provenance={"source_traces": [weak.id]},
        )

        service = MemoryRecallService(store)
        results = service.memory_recall(
            "why did deploy fail after db change",
            scope="project:alpha",
            max_results=3,
        )

        assert results[0].node.id == strong.id
        assert results[0].score > results[-1].score


def test_empty_recall_is_fast_and_empty(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        assert memory_recall(store, "anything", max_results=5) == []


def test_recall_rejects_zero_result_request(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.append_trace("some trace", {"scope": "global"})
        assert memory_recall(store, "trace", max_results=0) == []


def test_embedding_similarity_handles_semantic_aliases() -> None:
    from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity

    model = LocalEmbeddingModel()
    similarity = cosine_similarity(
        model.embed("auth timeout in prod"),
        model.embed("authentication slow in production"),
    )
    assert similarity > 0.2


def test_embedding_similarity_handles_cross_language_deployment_failure() -> None:
    from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity

    model = LocalEmbeddingModel()
    similarity = cosine_similarity(
        model.embed("deployment failed because database migration was missing"),
        model.embed("развертывание упало потому что отсутствовала миграция базы данных"),
    )

    assert similarity > 0.5


def test_cross_language_recall_finds_russian_and_english_traces(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        russian = store.append_trace(
            "Развертывание упало из-за ошибки миграции базы данных",
            {"scope": "project:alpha", "agent": "agent-ru"},
            feedback={"confidence": 0.5, "usefulness_score": 0.8},
        )
        english = store.append_trace(
            "Configuration error broke authentication login",
            {"scope": "project:alpha", "agent": "agent-en"},
            feedback={"confidence": 0.5, "usefulness_score": 0.7},
        )

        english_query = memory_recall(
            store,
            "deployment failed database migration error",
            scope="project:alpha",
            max_results=3,
        )
        russian_query = memory_recall(
            store,
            "ошибка конфигурации авторизации входа",
            scope="project:alpha",
            max_results=3,
        )

        assert english_query
        assert english_query[0].node.id == russian.id
        assert "vector" in english_query[0].methods
        assert russian_query
        assert russian_query[0].node.id == english.id
