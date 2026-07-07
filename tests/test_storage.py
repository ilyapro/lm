import hashlib
import sqlite3
from pathlib import Path

import pytest

from living_memory.config import MemoryConfig, load_config
from living_memory.storage import SCHEMA_VERSION, MemoryStore, _content_fingerprint


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
        assert "idx_nodes_embedded_active_scope" in indexes
        assert "embedding IS NOT NULL" in indexes["idx_nodes_embedded_active_scope"]
        assert "idx_nodes_dedup" in indexes
        dedup_sql = indexes["idx_nodes_dedup"]
        assert "level" in dedup_sql
        assert "scope" in dedup_sql
        assert "content_fingerprint" in dedup_sql
        assert "decayed = 0" in dedup_sql
        assert "content_fingerprint IS NOT NULL" in dedup_sql

        assert "content_fingerprint" in node_columns

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


def test_find_similar_by_embedding_filters_scope_prefix_and_threshold(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        alpha = store.create_node(
            level="concept",
            content="alpha database timeout pattern",
            context={"scope": "project:alpha"},
            embedding=[1.0, 0.0],
        )
        beta = store.create_node(
            level="concept",
            content="beta database timeout pattern",
            context={"scope": "project:beta"},
            embedding=[0.8, 0.6],
        )
        store.create_node(
            level="concept",
            content="global database timeout pattern",
            context={"scope": "global"},
            embedding=[0.9, 0.1],
        )
        store.create_node(
            level="concept",
            content="gamma unrelated pattern",
            context={"scope": "project:gamma"},
            embedding=[0.0, 1.0],
        )

        matches = store.find_similar_by_embedding(
            alpha.embedding or [],
            level="concept",
            exclude_scope="project:alpha",
            scope_prefix="project:",
            threshold=0.75,
        )

        assert [node.id for node, _score in matches] == [beta.id]
        assert matches[0][1] == pytest.approx(0.8)


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


def test_schema_version_is_four_after_initialize(tmp_path: Path) -> None:
    assert SCHEMA_VERSION == 4
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        row = store.connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        assert row is not None
        assert row["value"] == "4"


def test_schema_v2_database_migrates_to_v3_with_backfill(tmp_path: Path) -> None:
    """A pre-v3 fixture DB must gain content_fingerprint + idx_nodes_dedup on open.

    The version stamp lands on the current SCHEMA_VERSION because every
    migration in the chain runs on open.
    """

    db = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('schema_version', '2')"
        )
        conn.execute(
            """
            CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                level TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                scope TEXT NOT NULL DEFAULT 'global',
                agent TEXT,
                task TEXT,
                context TEXT NOT NULL DEFAULT '{}',
                timestamp TEXT NOT NULL,
                decayed INTEGER NOT NULL DEFAULT 0,
                decay_reason TEXT,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed TEXT,
                usefulness_score REAL NOT NULL DEFAULT 0.0,
                confidence REAL NOT NULL DEFAULT 0.5,
                unique_agents INTEGER NOT NULL DEFAULT 0,
                temporal_hint TEXT,
                source_traces TEXT NOT NULL DEFAULT '[]',
                corrections TEXT NOT NULL DEFAULT '[]',
                provenance TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        seeded = []
        for index in range(50):
            content = f"legacy trace number {index}"
            node_id = f"LEGACY{index:020d}"
            conn.execute(
                """
                INSERT INTO nodes (id, level, content, scope, timestamp, created_at, updated_at)
                VALUES (?, 'trace', ?, 'project:legacy', '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z')
                """,
                (node_id, content),
            )
            seeded.append((node_id, content))
        conn.commit()
    finally:
        conn.close()

    with MemoryStore(db) as store:
        row = store.connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        assert row["value"] == str(SCHEMA_VERSION)

        columns = {
            r["name"]
            for r in store.connection.execute("PRAGMA table_info(nodes)").fetchall()
        }
        assert "content_fingerprint" in columns

        indexes = {
            r["name"]
            for r in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_nodes_dedup" in indexes

        for node_id, content in seeded:
            row = store.connection.execute(
                "SELECT content_fingerprint FROM nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
            assert row is not None
            assert row["content_fingerprint"] == hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest()
            assert row["content_fingerprint"] == _content_fingerprint(content)


def test_schema_v3_database_migrates_to_v4_with_transport_column(tmp_path: Path) -> None:
    """A pre-v4 fixture DB must gain recall_events.transport_session_id on open."""

    db = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('schema_version', '3')"
        )
        conn.execute(
            """
            CREATE TABLE recall_events (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                scope TEXT NOT NULL DEFAULT 'global',
                requested_scope TEXT NOT NULL DEFAULT 'global',
                resolved_scopes TEXT NOT NULL DEFAULT '[]',
                ambient_context TEXT NOT NULL DEFAULT '{}',
                depth TEXT,
                max_results INTEGER NOT NULL DEFAULT 10,
                results TEXT NOT NULL DEFAULT '[]',
                agent TEXT,
                task TEXT,
                session_id TEXT,
                feedback_applied INTEGER NOT NULL DEFAULT 0 CHECK (feedback_applied IN (0, 1)),
                feedback_trace_id TEXT REFERENCES nodes(id),
                feedback_applied_at TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        seeded = (
            ("LEGACYEVENT01", "alpha bravo charlie delta", 0, None),
            ("LEGACYEVENT02", "echo foxtrot golf hotel", 1, "2026-01-01T00:00:01Z"),
            ("LEGACYEVENT03", "india juliett kilo lima", 0, None),
        )
        for index, (event_id, query, feedback_applied, applied_at) in enumerate(seeded):
            conn.execute(
                """
                INSERT INTO recall_events (
                    id, query, scope, requested_scope, resolved_scopes,
                    ambient_context, feedback_applied, feedback_applied_at, created_at
                )
                VALUES (?, ?, 'project:legacy', 'project:legacy',
                        '["project:legacy"]', '{"note":"legacy"}', ?, ?, ?)
                """,
                (event_id, query, feedback_applied, applied_at, f"2026-01-01T00:00:0{index}Z"),
            )
        conn.commit()
    finally:
        conn.close()

    with MemoryStore(db) as store:
        row = store.connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        assert row["value"] == "4"

        columns = {
            r["name"]
            for r in store.connection.execute(
                "PRAGMA table_info(recall_events)"
            ).fetchall()
        }
        assert "transport_session_id" in columns

        by_id = {event.id: event for event in store.list_recall_events(scope="project:legacy")}
        assert set(by_id) == {"LEGACYEVENT01", "LEGACYEVENT02", "LEGACYEVENT03"}
        assert by_id["LEGACYEVENT01"].feedback_applied is False
        assert by_id["LEGACYEVENT02"].feedback_applied is True
        assert by_id["LEGACYEVENT03"].feedback_applied is False
        assert all(event.transport_session_id is None for event in by_id.values())
        assert by_id["LEGACYEVENT02"].ambient_context == {"note": "legacy"}

        recorded = store.record_recall_event(
            query="mike november oscar papa",
            scope="project:fresh",
            ambient_context={"transport_session_id": "transport-1"},
        )
        fetched = store.get_recall_event(recorded.id)
        assert fetched is not None
        assert fetched.transport_session_id == "transport-1"
        assert fetched.ambient_context == {"transport_session_id": "transport-1"}

        # Equal transport ids match strongly even with zero text overlap...
        pending = store.pending_recall_events(
            scope="project:fresh",
            context={"transport_session_id": "transport-1"},
            content="quebec romeo sierra tango",
        )
        assert [event.id for event in pending] == [recorded.id]

        # ...while differing transport ids reject despite full text overlap.
        mismatched = store.pending_recall_events(
            scope="project:fresh",
            context={"transport_session_id": "transport-2"},
            content="mike november oscar papa exact overlap",
        )
        assert mismatched == []


def test_repeated_initialize_does_not_rewrite_existing_fingerprints(
    tmp_path: Path,
) -> None:
    db = tmp_path / "stable.sqlite3"
    with MemoryStore(db) as store:
        trace = store.append_trace(
            "stable content fixture", {"scope": "project:demo"}
        )
        original_fp = store.connection.execute(
            "SELECT content_fingerprint FROM nodes WHERE id = ?", (trace.id,)
        ).fetchone()["content_fingerprint"]

    # Reopening triggers _initialize_schema again; backfill must be a no-op.
    with MemoryStore(db) as store:
        row = store.connection.execute(
            "SELECT content_fingerprint FROM nodes WHERE id = ?", (trace.id,)
        ).fetchone()
        assert row["content_fingerprint"] == original_fp
