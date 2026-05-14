from pathlib import Path

import pytest

from living_memory.config import MemoryConfig, load_config
from living_memory.storage import MemoryStore


def test_storage_schema_has_uniform_nodes_fts_edges_indexes_and_weights(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    try:
        tables = {
            row["name"]
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
            )
        }
        assert "nodes" in tables
        assert "nodes_fts" in tables
        assert "connections" in tables
        assert "recall_events" in tables
        assert "retrieval_weights" in tables

        node_columns = {
            row["name"] for row in store.connection.execute("PRAGMA table_info(nodes)").fetchall()
        }
        assert {
            "level",
            "content",
            "embedding",
            "scope",
            "context",
            "access_count",
            "usefulness_score",
            "confidence",
            "unique_agents",
            "temporal_hint",
            "source_traces",
            "corrections",
            "provenance",
            "decayed",
        }.issubset(node_columns)

        indexes = {
            row["name"]: row["sql"]
            for row in store.connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL"
            )
        }
        assert "idx_nodes_trace_time" in indexes
        assert "WHERE level = 'trace' AND decayed = 0" in indexes["idx_nodes_trace_time"]
        assert "idx_nodes_concepts_content" in indexes
        assert "level IN ('concept', 'schema') AND decayed = 0" in indexes[
            "idx_nodes_concepts_content"
        ]

        project_weights = store.get_retrieval_weights("project:alpha")
        assert project_weights.scope == "project"
        assert project_weights.bm25 == pytest.approx(0.7)
        assert project_weights.vector == pytest.approx(0.3)
        assert project_weights.graph == pytest.approx(0.0)
    finally:
        store.close()


def test_memory_store_crud_search_connections_and_append_only_traces(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        trace = store.append_trace(
            "deploy failed because migration 42 was missing",
            {"scope": "project:alpha", "agent": "agent-a", "task": "deploy"},
        )
        assert len(trace.id) == 26
        assert trace.level == "trace"
        assert trace.scope == "project:alpha"
        assert trace.confidence <= 0.5
        assert trace.context["agent"] == "agent-a"

        concept = store.create_node(
            level="concept",
            content="missing migration files cause deploy failures",
            context={"scope": "project:alpha", "agent": "agent-b"},
            provenance={"source_traces": [trace.id]},
            stats={"confidence": 0.9, "unique_agents": 2, "usefulness_score": 0.8},
        )
        assert concept.source_traces == [trace.id]
        assert concept.provenance["source_traces"] == [trace.id]
        assert concept.confidence == pytest.approx(0.9)
        assert concept.to_dict()["stats"]["confidence"] == pytest.approx(0.9)

        connection = store.create_connection(trace.id, concept.id, "caused", weight=0.75)
        assert connection.source_id == trace.id
        assert connection.target_id == concept.id
        assert connection.type == "caused"
        assert store.get_connection(connection.id) == connection

        updated_connection = store.update_connection(
            connection.id,
            weight=0.9,
            metadata={"source": "test"},
        )
        assert updated_connection.weight == pytest.approx(0.9)
        assert updated_connection.metadata == {"source": "test"}

        results = store.search_content("migration", scope="project:alpha")
        assert [node.id for node, _score in results] == [concept.id, trace.id]

        accessed = store.record_access(trace.id)
        assert accessed.access_count == 1
        assert accessed.last_accessed is not None

        corrected = store.add_correction(
            trace.id,
            old="migration 42",
            new="migration 43",
            by="agent-c",
        )
        assert corrected.corrections[-1]["new"] == "migration 43"

        with pytest.raises(ValueError, match="append-only"):
            store.update_node(trace.id, content="mutated raw trace")

        decayed = store.soft_delete_node(trace.id, "ttl")
        assert decayed.decayed is True
        assert decayed.decay_reason == "ttl"
        assert store.get_node(trace.id) is not None

        active_ids = [node.id for node in store.list_nodes(level="trace")]
        assert trace.id not in active_ids
        all_trace_ids = [node.id for node in store.list_nodes(level="trace", include_decayed=True)]
        assert trace.id in all_trace_ids

        store.delete_connection(connection.id)
        assert store.get_connection(connection.id) is None


def test_toml_config_loads_storage_phase_and_retrieval_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "memory.toml"
    config_path.write_text(
        """
[storage]
db_path = "custom.sqlite3"
default_scope = "project:test"
trace_ttl_days = 30
embedding_model = "custom-multilingual-model"

[phases]
2 = 50

[retrieval_weights."project:test"]
bm25 = 0.6
vector = 0.2
graph = 0.2
learning_rate = 0.1
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.db_path == Path("custom.sqlite3")
    assert config.default_scope == "project:test"
    assert config.trace_ttl_days == 30
    assert config.embedding_model == "custom-multilingual-model"
    assert config.phase_thresholds[2] == 50
    assert config.retrieval_weights["project:test"].graph == pytest.approx(0.2)

    with MemoryStore(MemoryConfig(db_path=tmp_path / "configured.sqlite3", default_scope="project:test")) as store:
        trace = store.append_trace("stored under configured scope")
        assert trace.scope == "project:test"
