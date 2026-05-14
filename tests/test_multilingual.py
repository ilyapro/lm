from __future__ import annotations

from pathlib import Path

from living_memory.config import MemoryConfig
from living_memory.consolidation import memory_consolidate
from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity, tokenize
from living_memory.retrieval import memory_recall
from living_memory.storage import MemoryStore


def test_unicode_tokenizer_keeps_non_latin_terms_and_maps_known_aliases() -> None:
    tokens = tokenize("Развёртывание упало: пользовательский индекс базы")

    assert "deployment" in tokens
    assert "failure" in tokens
    assert "database" in tokens
    assert "пользовательский" in tokens
    assert "индекс" in tokens


def test_cross_language_embedding_similarity_for_deployment_failure() -> None:
    model = LocalEmbeddingModel()

    similarity = cosine_similarity(
        model.embed("deployment failed"),
        model.embed("развёртывание упало"),
    )

    assert similarity > 0.5


def test_cross_language_recall_finds_russian_from_english_and_english_from_russian(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        russian = store.append_trace(
            "Развёртывание упало из-за ошибки миграции базы данных",
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
        assert "vector" in russian_query[0].methods


def test_cross_language_consolidation_merges_same_topic_into_one_concept(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        traces = []
        for index in range(50):
            traces.append(
                store.append_trace(
                    f"deployment failed because database migration was missing {index}",
                    {"scope": "project:alpha", "agent": f"agent-en-{index % 2}"},
                )
            )
            traces.append(
                store.append_trace(
                    f"развёртывание упало потому что отсутствовала миграция базы данных {index}",
                    {"scope": "project:alpha", "agent": f"agent-ru-{index % 2}"},
                )
            )

        result = memory_consolidate(store, scope="project:alpha")

        assert len(result.concepts_created) == 1
        concept = result.concepts_created[0]
        assert concept.scope == "project:alpha"
        assert concept.provenance["strategy"] == "embedding-cosine"
        assert concept.source_traces == sorted(trace.id for trace in traces)
        assert store.list_nodes(level="concept", scope="project:alpha") == [concept]


def test_embedding_model_is_configurable_from_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "memory.toml"
    config_path.write_text(
        """
[storage]
db_path = "memory.sqlite3"

[embeddings]
model = "custom/local-multilingual-model"
""".strip(),
        encoding="utf-8",
    )

    config = MemoryConfig.from_toml(config_path)

    assert config.embedding_model == "custom/local-multilingual-model"
