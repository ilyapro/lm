import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

import living_memory.storage as storage_module
from living_memory.config import MemoryConfig, load_config
from living_memory.storage import (
    MAX_RECALL_HISTORY_CANDIDATES,
    RECALL_DELIVERY_HISTORY_TABLE,
    RECALL_DELIVERY_HISTORY_STATE_TABLE,
    RECALL_HISTORY_EVENT_TABLE,
    RECALL_HISTORY_RESULT_TABLE,
    RECALL_LOOKUP_EVENT_TABLE,
    SCHEMA_VERSION,
    MemoryStore,
    _content_fingerprint,
    recall_fingerprint,
)


def test_node_scopes_read_authoritative_facts_in_batches_including_decayed(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        active = store.append_trace("Inspect the hatch.")
        decayed = store.append_trace("Release pressure first.")
        store.soft_delete_node(decayed.id)
        with store.transaction():
            store.connection.execute(
                "UPDATE nodes SET scope = ?, content = CAST(x'ff' AS TEXT) WHERE id = ?",
                ("project:hatch", decayed.id),
            )
        ids = [f"missing-{i}" for i in range(1100)] + [active.id, decayed.id, active.id]
        assert store.get_node_scopes(iter(ids)) == {active.id: "global", decayed.id: "project:hatch"}
        assert store.get_node_scopes([]) == {}
        with pytest.raises(sqlite3.OperationalError, match="Could not decode to UTF-8"):
            store.get_node(decayed.id)


def test_node_contents_read_current_sources_without_unrelated_fields(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        active = store.create_node(level="concept", content="Inspect the hatch.")
        decayed = store.append_trace("Release pressure first.")
        store.soft_delete_node(decayed.id)
        with store.transaction():
            store.connection.execute(
                "UPDATE nodes SET provenance = CAST(x'ff' AS TEXT) WHERE id = ?", (decayed.id,),
            )
        ids = [f"missing-{i}" for i in range(1100)] + [active.id, decayed.id, active.id]
        facts = store.get_node_contents(iter(ids))
        assert {key: (value.id, value.content, value.level) for key, value in facts.items()} == {
            active.id: (active.id, active.content, "concept"),
            decayed.id: (decayed.id, decayed.content, "trace"),
        }
        assert store.get_node_contents([]) == {}
        store.update_node(active.id, content="Open the current hatch.")
        assert store.get_node_contents([active.id])[active.id].content == "Open the current hatch."
        with store.transaction():
            store.connection.execute(
                "UPDATE nodes SET content = CAST(x'ff' AS TEXT) WHERE id = ?", (decayed.id,),
            )
        with pytest.raises(sqlite3.OperationalError, match="Could not decode to UTF-8"):
            store.get_node_contents([decayed.id])


def test_list_nodes_filters_source_citations_with_scope_level_and_pagination(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        source = store.append_trace("Harbor inspection evidence.")
        trace_owner = store.create_node(
            level="concept", content="Harbor inspection digest.", context={"scope": "project:harbor"},
            provenance={"source_traces": [source.id]},
        )
        concept_owner = store.create_node(
            level="concept", content="Harbor project synthesis.", context={"scope": "project:harbor"},
            provenance={"source_concepts": [source.id]},
        )
        edge_only = store.create_node(
            level="concept", content="Harbor related policy.", context={"scope": "project:harbor"},
        )
        store.create_connection(edge_only.id, source.id, "related", metadata={"kind": "derived_from"})
        for scope, level in (("project:other", "concept"), ("project:harbor", "schema")):
            store.create_node(level=level, content=f"Other citation {scope} {level}.",
                              context={"scope": scope}, provenance={"source_traces": [source.id]})
        retired = store.create_node(
            level="concept", content="Retired harbor digest.", context={"scope": "project:harbor"},
            provenance={"source_traces": [source.id]},
        )
        store.soft_delete_node(retired.id)
        selection = {"scope": "project:harbor", "level": "concept"}
        expected = sorted([trace_owner.id, concept_owner.id], reverse=True)
        # Source sets can include long replacement chains without SQL variable limits.
        ids = [f"missing-{i}" for i in range(1100)] + [source.id, source.id]
        assert [node.id for node in store.list_nodes(**selection, source_ids=iter(ids))] == expected
        assert [node.id for node in store.list_nodes(**selection, source_ids=ids, limit=1, offset=1)] == expected[1:]
        assert store.list_nodes(**selection, source_ids=[]) == []
        assert {node.id for node in store.list_nodes(**selection, source_ids=ids, include_decayed=True)} == {
            *expected, retired.id,
        }
        assert edge_only.id in {node.id for node in store.list_nodes(**selection)}


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
        assert RECALL_HISTORY_EVENT_TABLE in tables
        assert RECALL_HISTORY_RESULT_TABLE in tables
        assert RECALL_DELIVERY_HISTORY_TABLE in tables
        assert RECALL_DELIVERY_HISTORY_STATE_TABLE in tables
        assert RECALL_LOOKUP_EVENT_TABLE in tables
        assert "recall_fingerprints" not in tables
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


def test_task_fts_rebuilds_old_index_and_tracks_new_and_edited_tasks(tmp_path: Path) -> None:
    db = tmp_path / "old-index.sqlite3"
    with MemoryStore(db) as store:
        old = store.append_trace(
            "The release comparison was recorded.",
            {"scope": "project:alpha", "task": "prism673old", "procedure": "hidden581"},
        )

    # A committed old-format index and triggers beside current nodes emulate an
    # existing database without rewriting its source rows.
    raw = sqlite3.connect(db)
    try:
        raw.executescript(
            """
            DROP TRIGGER nodes_fts_insert;
            DROP TRIGGER nodes_fts_delete;
            DROP TRIGGER nodes_fts_update;
            DROP TABLE nodes_fts_vocab;
            DROP TABLE nodes_fts;
            CREATE VIRTUAL TABLE nodes_fts USING fts5(
                node_id UNINDEXED, content, level UNINDEXED,
                scope UNINDEXED, tokenize = 'unicode61'
            );
            INSERT INTO nodes_fts(rowid, node_id, content, level, scope)
            SELECT rowid, id, content, level, scope FROM nodes;
            CREATE TRIGGER nodes_fts_insert AFTER INSERT ON nodes BEGIN
                INSERT INTO nodes_fts(rowid, node_id, content, level, scope)
                VALUES (new.rowid, new.id, new.content, new.level, new.scope);
            END;
            CREATE TRIGGER nodes_fts_delete AFTER DELETE ON nodes BEGIN
                DELETE FROM nodes_fts WHERE rowid = old.rowid;
            END;
            CREATE TRIGGER nodes_fts_update
            AFTER UPDATE OF content, level, scope ON nodes BEGIN
                DELETE FROM nodes_fts WHERE rowid = old.rowid;
                INSERT INTO nodes_fts(rowid, node_id, content, level, scope)
                VALUES (new.rowid, new.id, new.content, new.level, new.scope);
            END;
            CREATE VIRTUAL TABLE nodes_fts_vocab USING fts5vocab('nodes_fts', 'row');
            """
        )
        raw.commit()
    finally:
        raw.close()

    with MemoryStore(db) as store:
        assert [node.id for node, _ in store.search_content("prism673old")] == [old.id]
        assert store.search_content("hidden581") == []
        assert store.term_document_frequencies(["prism", "release"]) == {
            "prism": 0,
            "release": 1,
        }
        new = store.append_trace(
            "A second comparison was recorded.",
            {"scope": "project:alpha", "task": "prism674new"},
        )
        assert [node.id for node, _ in store.search_content("prism674new")] == [new.id]
        store.update_node(old.id, context={"scope": "project:alpha", "task": "prism675edited"})
        assert store.search_content("prism673old") == []
        assert [node.id for node, _ in store.search_content("prism675edited")] == [old.id]
        assert store.get_node(old.id).content == "The release comparison was recorded."

    # Reopening a migrated database must keep the index and row identities.
    with MemoryStore(db) as store:
        assert [node.id for node, _ in store.search_content("prism675edited")] == [old.id]


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


def test_schema_version_is_eight_after_initialize(tmp_path: Path) -> None:
    assert SCHEMA_VERSION == 8
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        row = store.connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        assert row is not None
        assert row["value"] == "8"


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
        assert row["value"] == str(SCHEMA_VERSION)

        columns = {
            r["name"]
            for r in store.connection.execute(
                "PRAGMA table_info(recall_events)"
            ).fetchall()
        }
        assert "transport_session_id" in columns
        assert {"fingerprint", "gated"}.isdisjoint(columns)

        # The transport index lands on the same open: the pre-v4 column
        # migration must run before the unconditional CREATE INDEX.
        index_names = {
            r["name"]
            for r in store.connection.execute(
                "PRAGMA index_list('recall_events')"
            ).fetchall()
        }
        assert "idx_recall_events_transport_created" in index_names

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

        # Equal transport ids match...
        pending = store.pending_recall_events(
            scope="project:fresh",
            context={"transport_session_id": "transport-1"},
        )
        assert [event.id for event in pending] == [recorded.id]

        # ...while differing transport ids reject.
        mismatched = store.pending_recall_events(
            scope="project:fresh",
            context={"transport_session_id": "transport-2"},
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


def _record_event_at(
    store: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    at: str,
    *,
    results: list[dict[str, object]],
    scope: str = "project:history",
    task: str | None = "task-a",
    transport: str | None = None,
):
    monkeypatch.setattr(storage_module, "_utc_now", lambda: at)
    ambient: dict[str, str] = {}
    if task is not None:
        ambient["task"] = task
    if transport is not None:
        ambient["transport_session_id"] = transport
    return store.record_recall_event(
        query=f"query at {at}",
        scope=scope,
        ambient_context=ambient,
        results=results,
    )


def _organic_tail(node_id: str, prefix: str) -> list[dict[str, object]]:
    return [
        {"node_id": f"{prefix}-HEAD-0"},
        {"node_id": f"{prefix}-HEAD-1"},
        {"node_id": f"{prefix}-HEAD-2"},
        {"node_id": node_id},
    ]


def test_matured_recall_history_matches_frozen_delivery_and_consumption_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        # D1: a scope/task fallback repeats A, but a later same-transport event
        # exists without A. Transport-first therefore makes D1 nonconsumed.
        _record_event_at(
            store,
            monkeypatch,
            "2026-01-01T00:00:00Z",
            results=_organic_tail("NODE-A", "D1"),
            transport="transport-1",
        )
        _record_event_at(
            store,
            monkeypatch,
            "2026-01-01T01:00:00Z",
            results=[{"node_id": "NODE-A"}],
            transport="different-transport",
        )
        _record_event_at(
            store,
            monkeypatch,
            "2026-01-01T02:00:00Z",
            results=[{"node_id": "OTHER"}],
            transport="transport-1",
        )

        # D2 is consumed through its transport; D3 has no consumer.  At the
        # exact D3 maturity boundary the expected history is M=3,C=1,K=1.
        _record_event_at(
            store,
            monkeypatch,
            "2026-01-02T06:00:00Z",
            results=_organic_tail("NODE-A", "D2"),
            transport="transport-2",
        )
        _record_event_at(
            store,
            monkeypatch,
            "2026-01-02T07:00:00Z",
            results=[{"node_id": "NODE-A"}],
            transport="transport-2",
        )
        _record_event_at(
            store,
            monkeypatch,
            "2026-01-03T12:00:00Z",
            results=_organic_tail("NODE-A", "D3"),
            transport=None,
        )

        before_boundary = store.matured_recall_history(
            ["NODE-A", "NEVER-DELIVERED"], "2026-01-04T11:59:59Z"
        )
        assert before_boundary["NODE-A"].available is True
        assert (
            before_boundary["NODE-A"].m,
            before_boundary["NODE-A"].c,
            before_boundary["NODE-A"].k,
        ) == (2, 1, 0)
        assert before_boundary["NEVER-DELIVERED"].available is True
        assert (
            before_boundary["NEVER-DELIVERED"].m,
            before_boundary["NEVER-DELIVERED"].c,
            before_boundary["NEVER-DELIVERED"].k,
        ) == (0, 0, 0)

        at_boundary = store.matured_recall_history(
            ["NODE-A"], "2026-01-04T12:00:00Z"
        )["NODE-A"]
        assert (at_boundary.m, at_boundary.c, at_boundary.k) == (3, 1, 1)

        # A delivery at the decision instant and a later delivery are both
        # present in the ledger, but neither has a matured outcome.
        _record_event_at(
            store,
            monkeypatch,
            "2026-01-04T12:00:00Z",
            results=_organic_tail("NODE-A", "CURRENT"),
            task="other-task",
        )
        _record_event_at(
            store,
            monkeypatch,
            "2026-01-04T13:00:00Z",
            results=_organic_tail("NODE-A", "FUTURE"),
            task="other-task",
        )
        unchanged = store.matured_recall_history(
            ["NODE-A"], "2026-01-04T12:00:00Z"
        )["NODE-A"]
        assert (unchanged.m, unchanged.c, unchanged.k) == (3, 1, 1)


def test_attach_recall_map_adds_medoid_and_resolves_existing_later_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        delivery = _record_event_at(
            store,
            monkeypatch,
            "2026-02-01T00:00:00Z",
            results=[{"node_id": "ONLY-A-HEAD-RESULT"}],
            transport="map-transport",
        )
        _record_event_at(
            store,
            monkeypatch,
            "2026-02-01T01:00:00Z",
            results=[{"node_id": "MAP-MEDOID"}],
            transport="map-transport",
        )
        store.attach_recall_map(
            delivery.id,
            {
                "clusters": [
                    {"medoid": {"node_id": "MAP-MEDOID"}},
                    # The organic/map union de-duplicates within one event.
                    {"medoid": {"node_id": "MAP-MEDOID"}},
                ]
            },
        )

        history = store.matured_recall_history(
            ["MAP-MEDOID"], "2026-02-02T00:00:00Z"
        )["MAP-MEDOID"]
        assert history.available is True
        assert (history.m, history.c, history.k) == (1, 1, 0)


def test_matured_recall_history_is_indexed_bounded_and_select_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for index, at in enumerate(
            (
                "2026-03-01T00:00:00Z",
                "2026-03-03T00:00:00Z",
                "2026-03-05T00:00:00Z",
            )
        ):
            _record_event_at(
                store,
                monkeypatch,
                at,
                results=_organic_tail("FREQUENT", f"F{index}"),
                task=None,
            )

        plan = " ".join(
            row["detail"]
            for row in store.connection.execute(
                f"""
                EXPLAIN QUERY PLAN
                SELECT transport_matched, transport_consumed,
                       fallback_consumed, lookup_consumed
                FROM {RECALL_DELIVERY_HISTORY_TABLE}
                WHERE node_id = 'FREQUENT'
                  AND delivered_at < '2026-03-07T00:00:00.000000Z'
                  AND outcome_end <= '2026-03-07T00:00:00.000000Z'
                ORDER BY outcome_end DESC, delivered_at DESC,
                    (CASE WHEN transport_matched = 1
                        THEN transport_consumed ELSE fallback_consumed END) DESC,
                    delivery_event_id DESC
                LIMIT 1025
                """
            ).fetchall()
        )
        assert "idx_recall_delivery_history_node_matured" in plan

        statements: list[str] = []
        before_changes = store.connection.total_changes
        store.connection.set_trace_callback(statements.append)
        history = store.matured_recall_history(
            ["FREQUENT", "ABSENT"], "2026-03-07T00:00:00Z"
        )
        store.connection.set_trace_callback(None)
        assert history["FREQUENT"].m == 3
        assert store.connection.total_changes == before_changes
        assert not any(
            statement.lstrip().upper().startswith(
                ("INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE", "DROP", "ALTER")
            )
            for statement in statements
        )

        monkeypatch.setattr(
            storage_module, "MAX_RECALL_HISTORY_DELIVERIES_PER_NODE", 2
        )
        truncated = store.matured_recall_history(
            ["FREQUENT"], "2026-03-07T00:00:00Z"
        )["FREQUENT"]
        assert truncated.available is False
        assert truncated.unavailable_reason == "history_truncated"
        assert (truncated.m, truncated.c, truncated.k) == (None, None, None)

        too_many = store.matured_recall_history(
            [f"NODE-{index}" for index in range(MAX_RECALL_HISTORY_CANDIDATES + 1)],
            "2026-03-07T00:00:00Z",
        )
        assert {item.unavailable_reason for item in too_many.values()} == {
            "candidate_batch_truncated"
        }


def test_recall_history_backfill_is_idempotent_and_malformed_legacy_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "legacy.sqlite3"
    with MemoryStore(db) as store:
        _record_event_at(
            store,
            monkeypatch,
            "2026-04-01T00:00:00Z",
            results=_organic_tail("LEGACY-NODE", "LEGACY"),
        )
        for table in (
            RECALL_DELIVERY_HISTORY_TABLE,
            RECALL_HISTORY_RESULT_TABLE,
            RECALL_HISTORY_EVENT_TABLE,
            RECALL_DELIVERY_HISTORY_STATE_TABLE,
        ):
            store.connection.execute(f"DELETE FROM {table}")
        store.connection.commit()

    with MemoryStore(db) as store:
        rebuilt = store.matured_recall_history(
            ["LEGACY-NODE"], "2026-04-02T00:00:00Z"
        )["LEGACY-NODE"]
        assert (rebuilt.m, rebuilt.c, rebuilt.k) == (1, 0, 1)
        row_count = store.connection.execute(
            f"SELECT COUNT(*) FROM {RECALL_DELIVERY_HISTORY_TABLE}"
        ).fetchone()[0]

    with MemoryStore(db) as store:
        assert store.connection.execute(
            f"SELECT COUNT(*) FROM {RECALL_DELIVERY_HISTORY_TABLE}"
        ).fetchone()[0] == row_count
        store.connection.execute("UPDATE recall_events SET results = '{'")
        for table in (
            RECALL_DELIVERY_HISTORY_TABLE,
            RECALL_HISTORY_RESULT_TABLE,
            RECALL_HISTORY_EVENT_TABLE,
            RECALL_DELIVERY_HISTORY_STATE_TABLE,
        ):
            store.connection.execute(f"DELETE FROM {table}")
        store.connection.commit()

    store = MemoryStore(db)
    malformed = store.matured_recall_history(
        ["LEGACY-NODE"], "2026-04-02T00:00:00Z"
    )["LEGACY-NODE"]
    assert malformed.available is False
    assert malformed.unavailable_reason == "malformed_legacy_results"
    store.close()

    failed = store.matured_recall_history(
        ["LEGACY-NODE"], "2026-04-02T00:00:00Z"
    )["LEGACY-NODE"]
    assert failed.available is False
    assert failed.unavailable_reason == "history_read_failed"


def _lookup_column(store: MemoryStore, node_id: str) -> list:
    return [
        row["lookup_consumed"]
        for row in store.connection.execute(
            f"""
            SELECT lookup_consumed FROM {RECALL_DELIVERY_HISTORY_TABLE}
            WHERE node_id = ? ORDER BY delivered_at
            """,
            (node_id,),
        )
    ]


def test_record_lookup_event_stores_exactly_what_was_requested(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        event_id = store.record_lookup_event(
            # A duplicate, a blank, and an id that resolves to nothing.
            ["ASKED-A", "ASKED-A", "  ", "ASKED-GONE"],
            transport_session_id="agent-transport",
            occurred_at="2026-08-01T00:00:00Z",
        )
        assert event_id
        rows = [
            dict(row)
            for row in store.connection.execute(
                f"SELECT * FROM {RECALL_LOOKUP_EVENT_TABLE} ORDER BY node_id"
            )
        ]
        assert [row["node_id"] for row in rows] == ["ASKED-A", "ASKED-GONE"]
        assert {row["lookup_event_id"] for row in rows} == {event_id}
        assert {row["occurred_at"] for row in rows} == {
            "2026-08-01T00:00:00.000000Z"
        }
        assert {row["transport_session_id"] for row in rows} == {"agent-transport"}

        # Nothing to record is not an event.
        assert store.record_lookup_event([]) is None
        assert store.record_lookup_event(["", None]) is None  # type: ignore[list-item]
        assert (
            store.connection.execute(
                f"SELECT COUNT(*) FROM {RECALL_LOOKUP_EVENT_TABLE}"
            ).fetchone()[0]
            == 2
        )

        with pytest.raises(ValueError):
            store.record_lookup_event(["ASKED-A"], occurred_at="not a timestamp")


def test_lookup_consumption_is_global_and_bounded_by_the_open_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        # An unrelated lookup first, so the delivery below lands inside the
        # observable era and its windows resolve to a known 0 rather than NULL.
        store.record_lookup_event(
            ["EPOCH-ONLY"], occurred_at="2026-05-31T00:00:00Z"
        )
        _record_event_at(
            store,
            monkeypatch,
            "2026-06-01T00:00:00Z",
            results=[
                {"node_id": f"HEAD-{index}"} for index in range(3)
            ] + [
                {"node_id": "BOUND-OPEN"},
                {"node_id": "BOUND-MID"},
                {"node_id": "BOUND-END"},
                {"node_id": "BOUND-LATE"},
            ],
            transport="delivering-transport",
        )
        assert _lookup_column(store, "BOUND-MID") == [0]

        # (delivered_at, outcome_end]: open at the left, closed at the right.
        store.record_lookup_event(
            ["BOUND-OPEN"], occurred_at="2026-06-01T00:00:00Z"
        )
        store.record_lookup_event(
            ["BOUND-END"], occurred_at="2026-06-02T00:00:00Z"
        )
        store.record_lookup_event(
            ["BOUND-LATE"], occurred_at="2026-06-02T00:00:00.000001Z"
        )
        # Mid-window, from a transport that never saw the delivery — the
        # dashboard-probe-vs-agent split the global correlation exists for.
        store.record_lookup_event(
            ["BOUND-MID"],
            transport_session_id="a-completely-different-transport",
            occurred_at="2026-06-01T12:00:00Z",
        )

        assert _lookup_column(store, "BOUND-OPEN") == [0]
        assert _lookup_column(store, "BOUND-END") == [1]
        assert _lookup_column(store, "BOUND-LATE") == [0]
        assert _lookup_column(store, "BOUND-MID") == [1]

        # Both directions of the correlation are index range probes, not
        # scans: this runs on the live lookup path and on every delivery.
        def plan(sql: str) -> str:
            return " ".join(
                row["detail"]
                for row in store.connection.execute(f"EXPLAIN QUERY PLAN {sql}")
            )

        assert "idx_recall_lookup_events_node_time" in plan(
            f"""
            SELECT DISTINCT node_id FROM {RECALL_LOOKUP_EVENT_TABLE}
            WHERE node_id IN ('BOUND-MID')
              AND occurred_at > '2026-06-01T00:00:00.000000Z'
              AND occurred_at <= '2026-06-02T00:00:00.000000Z'
            """
        )
        assert "idx_recall_lookup_events_time" in plan(
            f"SELECT MIN(occurred_at) FROM {RECALL_LOOKUP_EVENT_TABLE}"
        )
        assert "idx_recall_delivery_history_node_matured" in plan(
            f"""
            UPDATE {RECALL_DELIVERY_HISTORY_TABLE} SET lookup_consumed = 1
            WHERE node_id = 'BOUND-MID'
              AND delivered_at < '2026-06-01T12:00:00.000000Z'
              AND outcome_end >= '2026-06-01T12:00:00.000000Z'
            """
        )

        # The lookup path touches the lookup column and nothing else.
        frozen = store.connection.execute(
            f"""
            SELECT DISTINCT transport_matched, transport_consumed,
                            fallback_consumed
            FROM {RECALL_DELIVERY_HISTORY_TABLE}
            """
        ).fetchall()
        assert [tuple(row) for row in frozen] == [(0, 0, 0)]


def test_rebuilt_windows_predating_the_first_lookup_are_unknown_not_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "coldstart.sqlite3"

    def rebuild() -> None:
        """Force the reconstruction the format bump performs on a live file."""

        with MemoryStore(db) as store:
            for table in (
                RECALL_DELIVERY_HISTORY_TABLE,
                RECALL_HISTORY_RESULT_TABLE,
                RECALL_HISTORY_EVENT_TABLE,
                RECALL_DELIVERY_HISTORY_STATE_TABLE,
            ):
                store.connection.execute(f"DELETE FROM {table}")
            store.connection.commit()

    with MemoryStore(db) as store:
        _record_event_at(
            store,
            monkeypatch,
            "2026-07-01T00:00:00Z",
            results=_organic_tail("OLD-NODE", "OLD"),
        )

    # (a) No lookup was ever recorded: day one, and every window is unknown.
    rebuild()
    with MemoryStore(db) as store:
        assert _lookup_column(store, "OLD-NODE") == [None]
        history = store.matured_recall_history(
            ["OLD-NODE"], "2026-07-20T00:00:00Z"
        )["OLD-NODE"]
        assert (history.m, history.c, history.k) == (1, 0, 1)
        assert history.lookup_known == 0
        assert history.lookup_consumed == 0

        # (b) The first lookup arrives long after this window closed. It still
        # says nothing about the window, so the window stays unknown — the
        # false-absence this rule exists to prevent.
        store.record_lookup_event(
            ["UNRELATED"], occurred_at="2026-07-10T00:00:00Z"
        )
    rebuild()
    with MemoryStore(db) as store:
        assert _lookup_column(store, "OLD-NODE") == [None]
        assert store.matured_recall_history(
            ["OLD-NODE"], "2026-07-20T00:00:00Z"
        )["OLD-NODE"].lookup_known == 0

        # (c) A lookup predating the window makes it observable. It is outside
        # the window, so the answer is a known absence: 0, not NULL.
        store.record_lookup_event(
            ["OLD-NODE"], occurred_at="2026-06-30T12:00:00Z"
        )
    rebuild()
    with MemoryStore(db) as store:
        assert _lookup_column(store, "OLD-NODE") == [0]
        history = store.matured_recall_history(
            ["OLD-NODE"], "2026-07-20T00:00:00Z"
        )["OLD-NODE"]
        assert (history.lookup_known, history.lookup_consumed) == (1, 0)
        # Re-derivation never invents the frozen triple either.
        assert (history.m, history.c, history.k) == (1, 0, 1)
        # The primary record survives its own rebuild.
        assert store.connection.execute(
            f"SELECT COUNT(*) FROM {RECALL_LOOKUP_EVENT_TABLE}"
        ).fetchone()[0] == 2


def test_matured_recall_history_reports_the_tristate_skipping_unknown_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for index, at in enumerate(
            (
                "2026-09-01T00:00:00Z",
                "2026-09-03T00:00:00Z",
                "2026-09-05T00:00:00Z",
                "2026-09-07T00:00:00Z",
            )
        ):
            _record_event_at(
                store, monkeypatch, at, results=_organic_tail("TRI", f"T{index}")
            )
        # Followed inside the second window only.
        store.record_lookup_event(["TRI"], occurred_at="2026-09-03T06:00:00Z")

        # The reader's contract, stated on a ledger written by hand because the
        # online rule cannot produce an unknown window newer than a known one:
        # newest-first the outcomes are 0, NULL, 0, 1, so the run counts two
        # known absences, steps over the unknown, and stops at the follow.
        store.connection.execute(
            f"""
            UPDATE {RECALL_DELIVERY_HISTORY_TABLE}
            SET lookup_consumed = CASE delivered_at
                WHEN '2026-09-01T00:00:00.000000Z' THEN 1
                WHEN '2026-09-03T00:00:00.000000Z' THEN 0
                WHEN '2026-09-05T00:00:00.000000Z' THEN NULL
                ELSE 0 END
            WHERE node_id = 'TRI'
            """
        )
        store.connection.commit()

        history = store.matured_recall_history(["TRI"], "2026-09-09T00:00:00Z")["TRI"]
        assert (history.m, history.c, history.k) == (4, 0, 4)
        assert history.lookup_known == 3
        assert history.lookup_consumed == 1
        assert history.lookup_trailing_absent == 2

        # Unavailability is tri-state-aware: no count is a stand-in for zero.
        monkeypatch.setattr(
            storage_module, "MAX_RECALL_HISTORY_DELIVERIES_PER_NODE", 2
        )
        truncated = store.matured_recall_history(["TRI"], "2026-09-09T00:00:00Z")["TRI"]
        assert truncated.available is False
        assert (
            truncated.lookup_known,
            truncated.lookup_consumed,
            truncated.lookup_trailing_absent,
        ) == (None, None, None)


def test_lookup_column_is_added_to_a_ledger_that_predates_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "prelookup.sqlite3"
    with MemoryStore(db) as store:
        _record_event_at(
            store,
            monkeypatch,
            "2026-10-01T00:00:00Z",
            results=_organic_tail("PRE-NODE", "PRE"),
        )

    # A file written by the release before this one: no column, and a state
    # row still claiming the previous ledger format.
    connection = sqlite3.connect(db)
    connection.execute(
        f"ALTER TABLE {RECALL_DELIVERY_HISTORY_TABLE} DROP COLUMN lookup_consumed"
    )
    connection.execute(
        f"UPDATE {RECALL_DELIVERY_HISTORY_STATE_TABLE} SET format_version = 1"
    )
    connection.commit()
    connection.close()

    def columns(store: MemoryStore) -> set[str]:
        return {
            row["name"]
            for row in store.connection.execute(
                f"PRAGMA table_info({RECALL_DELIVERY_HISTORY_TABLE})"
            )
        }

    # Opening migrates the column and the format bump re-derives the rows.
    for _ in range(2):
        with MemoryStore(db) as store:
            assert "lookup_consumed" in columns(store)
            assert store.connection.execute(
                f"SELECT format_version FROM {RECALL_DELIVERY_HISTORY_STATE_TABLE}"
            ).fetchone()["format_version"] == storage_module._RECALL_DELIVERY_HISTORY_FORMAT
            assert _lookup_column(store, "PRE-NODE") == [None]
            history = store.matured_recall_history(
                ["PRE-NODE"], "2026-10-03T00:00:00Z"
            )["PRE-NODE"]
            assert history.available is True
            assert (history.m, history.c, history.k) == (1, 0, 1)
            assert history.lookup_known == 0


#: One replay that exercises every frozen mechanic at once: transport-first
#: invalidation of a fallback hit, a transport consumer, a scope/task consumer,
#: a late-attached map medoid resolved against an already-recorded consumer,
#: and a delivery nobody ever answers.  Kept free of pytest fixtures on purpose
#: — ``scripts`` reproducing the golden below runs it against a checkout of the
#: pre-lookup code, where no fixture of this module exists.
FROZEN_SCENARIO_SCOPE = "project:frozen"
FROZEN_SCENARIO_NODES = ("FROZEN-A", "FROZEN-B", "FROZEN-MEDOID", "FROZEN-NEVER")
FROZEN_SCENARIO_INSTANTS = (
    "2026-05-02T00:00:00Z",
    "2026-05-04T00:00:00Z",
    "2026-05-07T00:00:00Z",
)


def _replay_frozen_scenario(store, set_now, *, record_lookups: bool) -> None:
    def event(at, results, *, task, transport=None):
        set_now(at)
        ambient: dict[str, str] = {}
        if task is not None:
            ambient["task"] = task
        if transport is not None:
            ambient["transport_session_id"] = transport
        return store.record_recall_event(
            query=f"query at {at}",
            scope=FROZEN_SCENARIO_SCOPE,
            ambient_context=ambient,
            results=results,
        )

    # D1: a scope/task fallback repeats A, but a later same-transport event
    # exists without A, so transport-first leaves D1 nonconsumed.
    event(
        "2026-05-01T00:00:00Z",
        _organic_tail("FROZEN-A", "D1"),
        task="alpha",
        transport="T1",
    )
    event(
        "2026-05-01T01:00:00Z",
        [{"node_id": "FROZEN-A"}],
        task="alpha",
        transport="T2",
    )
    if record_lookups:
        # Inside D1's window, from a transport that never delivered it: the
        # case the whole tri-state exists for, and the one the three frozen
        # bits must keep calling nonconsumed.
        store.record_lookup_event(
            ["FROZEN-A"],
            transport_session_id="dashboard-probe",
            occurred_at="2026-05-01T03:00:00Z",
        )
    event("2026-05-01T02:00:00Z", [{"node_id": "OTHER"}], task="beta", transport="T1")

    # D2 is consumed through its own transport.
    event(
        "2026-05-02T00:00:00Z",
        _organic_tail("FROZEN-B", "D2"),
        task="gamma",
        transport="T3",
    )
    event(
        "2026-05-02T03:00:00Z",
        [{"node_id": "FROZEN-B"}],
        task="gamma",
        transport="T3",
    )

    # D3 has no transport at all and is consumed through scope/task.
    event("2026-05-03T00:00:00Z", _organic_tail("FROZEN-A", "D3"), task="delta")
    event("2026-05-03T04:00:00Z", [{"node_id": "FROZEN-A"}], task="delta")

    # A map medoid attached after its consumer was already recorded.
    delivery = event(
        "2026-05-04T00:00:00Z",
        [{"node_id": "HEAD-ONLY"}],
        task="epsilon",
        transport="T4",
    )
    event(
        "2026-05-04T06:00:00Z",
        [{"node_id": "FROZEN-MEDOID"}],
        task="epsilon",
        transport="T4",
    )
    store.attach_recall_map(
        delivery.id, {"clusters": [{"medoid": {"node_id": "FROZEN-MEDOID"}}]}
    )

    # D5: delivered into silence.
    event("2026-05-05T00:00:00Z", _organic_tail("FROZEN-B", "D5"), task=None)
    if record_lookups:
        # A fetch of something never delivered: recorded as an event, matching
        # no window, changing no outcome.
        store.record_lookup_event(
            ["FROZEN-NEVER"], occurred_at="2026-05-05T12:00:00Z"
        )


def _frozen_ledger_dump(store) -> dict:
    """Every frozen field of the ledger, plus the M/C/K the evaluator reads.

    ``delivery_event_id`` is excluded because it is a fresh ULID per run; the
    remaining key ``(node_id, delivered_at)`` is unique within this scenario.
    """

    rows = [
        [
            row["node_id"],
            row["delivered_at"],
            row["outcome_end"],
            row["transport_session_id"],
            row["scope"],
            row["task"],
            row["transport_matched"],
            row["transport_consumed"],
            row["fallback_consumed"],
        ]
        for row in store.connection.execute(
            f"""
            SELECT node_id, delivered_at, outcome_end, transport_session_id,
                   scope, task, transport_matched, transport_consumed,
                   fallback_consumed
            FROM {RECALL_DELIVERY_HISTORY_TABLE}
            ORDER BY node_id, delivered_at, delivery_event_id
            """
        )
    ]
    mck = {
        instant: {
            node_id: [history.available, history.m, history.c, history.k]
            for node_id, history in sorted(
                store.matured_recall_history(
                    list(FROZEN_SCENARIO_NODES), instant
                ).items()
            )
        }
        for instant in FROZEN_SCENARIO_INSTANTS
    }
    return {"rows": rows, "mck": mck}


#: Captured by replaying ``_replay_frozen_scenario`` against the pre-lookup
#: code (``git archive HEAD`` of the commit that introduced the column), where
#: ``lookup_consumed`` and ``recall_lookup_events`` do not exist.  It is the
#: evidence for the claim that the tri-state is additive: not an assertion that
#: nothing moved, but the previous release's own answer, kept verbatim.
FROZEN_LEDGER_GOLDEN: dict = {
    "rows": [
        # D1: transport-matched, transport-nonconsumed, and a fallback hit that
        # transport-first overrides. The run that records lookups scores this
        # same window lookup_consumed = 1 — which is exactly the divergence the
        # goal is about, and it moves none of these three bits.
        ["FROZEN-A", "2026-05-01T00:00:00.000000Z", "2026-05-02T00:00:00.000000Z", "T1", "project:frozen", "alpha", 1, 0, 1],
        ["FROZEN-A", "2026-05-03T00:00:00.000000Z", "2026-05-04T00:00:00.000000Z", None, "project:frozen", "delta", 0, 0, 1],
        ["FROZEN-B", "2026-05-02T00:00:00.000000Z", "2026-05-03T00:00:00.000000Z", "T3", "project:frozen", "gamma", 1, 1, 1],
        ["FROZEN-B", "2026-05-05T00:00:00.000000Z", "2026-05-06T00:00:00.000000Z", None, "project:frozen", None, 0, 0, 0],
        ["FROZEN-MEDOID", "2026-05-04T00:00:00.000000Z", "2026-05-05T00:00:00.000000Z", "T4", "project:frozen", "epsilon", 1, 1, 1],
    ],
    "mck": {
        "2026-05-02T00:00:00Z": {
            "FROZEN-A": [True, 1, 0, 1],
            "FROZEN-B": [True, 0, 0, 0],
            "FROZEN-MEDOID": [True, 0, 0, 0],
            "FROZEN-NEVER": [True, 0, 0, 0],
        },
        "2026-05-04T00:00:00Z": {
            "FROZEN-A": [True, 2, 1, 0],
            "FROZEN-B": [True, 1, 1, 0],
            "FROZEN-MEDOID": [True, 0, 0, 0],
            "FROZEN-NEVER": [True, 0, 0, 0],
        },
        "2026-05-07T00:00:00Z": {
            "FROZEN-A": [True, 2, 1, 0],
            "FROZEN-B": [True, 2, 1, 1],
            "FROZEN-MEDOID": [True, 1, 1, 0],
            "FROZEN-NEVER": [True, 0, 0, 0],
        },
    },
}


def test_lookup_column_leaves_the_frozen_ledger_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def set_now(at: str) -> None:
        monkeypatch.setattr(storage_module, "_utc_now", lambda: at)

    dumps = []
    for index, record_lookups in enumerate((False, True)):
        with MemoryStore(tmp_path / f"frozen-{index}.sqlite3") as store:
            _replay_frozen_scenario(store, set_now, record_lookups=record_lookups)
            dumps.append(_frozen_ledger_dump(store))

    without_lookups, with_lookups = dumps
    # The golden is the pre-change release's answer; both of today's runs must
    # reproduce it, including the run where lookups were recorded throughout.
    assert without_lookups == FROZEN_LEDGER_GOLDEN
    assert with_lookups == FROZEN_LEDGER_GOLDEN


def test_recall_delivery_history_is_bounded_and_isolated(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        events = []
        for index in range(5):
            event = store.record_recall_event(
                query="alpha", scope="project:alpha",
                ambient_context={"transport_session_id": "transport-1"},
                results=[{"node_id": f"NODE-{index}", "score": 0.5}],
            )
            store.record_recall_delivery(event.id, {
                f"NODE-{index}": {"kind": "full", "content_sha256": "source",
                                  "delivered_sha256": "source"},
            })
            events.append(event)
        store.record_recall_event(
            query="other", scope="project:other",
            ambient_context={"transport_session_id": "transport-2"},
            results=[{"node_id": "OTHER"}],
        )
        args = {"transport_session_id": "transport-1"}
        history = store.recall_delivery_history(**args)
        assert [row["node_id"] for row in history] == [f"NODE-{i}" for i in reversed(range(5))]
        assert all(row["score"] == 0.5 for row in history)
        assert len(store.recall_delivery_history(**args, max_events=2)) == 2
        assert store.recall_delivery_history(**args, max_events=0) == []
        assert store.recall_delivery_history() == []
        assert store.recall_delivery_history("missing") == []
        assert store.get_recall_event(events[0].id).result_ids == ["NODE-0"]


def test_recall_events_transport_index_exists_and_serves_the_probe(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        index_names = {
            row["name"]
            for row in store.connection.execute(
                "PRAGMA index_list('recall_events')"
            ).fetchall()
        }
        assert "idx_recall_events_transport_created" in index_names

        plan = " ".join(
            row["detail"]
            for row in store.connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT results FROM recall_events
                WHERE transport_session_id = 'transport-1'
                ORDER BY created_at DESC, rowid DESC
                LIMIT 200
                """
            ).fetchall()
        )
        assert "idx_recall_events_transport_created" in plan


def test_recall_fingerprint_collapses_whitespace_and_binds_scope() -> None:
    base = recall_fingerprint("goal api recipe", "project:alpha")
    assert base == recall_fingerprint("  goal   api\trecipe \n", "project:alpha")
    assert base != recall_fingerprint("goal api recipe", "project:beta")
    assert base != recall_fingerprint("goal api recipes", "project:alpha")
    assert base != recall_fingerprint("goal api recipe", None)
    # The payload encoding is a stability contract: stored fingerprints must
    # keep matching recomputed ones across releases.
    assert base == hashlib.sha256(b"goal api recipe\nproject:alpha").hexdigest()


# ----------------------------------------------------------------------
# recall_events.task_pattern — the map's cache label, as a column
# ----------------------------------------------------------------------


def _recall_event_columns(store: MemoryStore) -> list[str]:
    return [
        str(row["name"])
        for row in store.connection.execute("PRAGMA table_info(recall_events)")
    ]


def test_recall_events_lifts_task_pattern_out_of_the_ambient_context(
    tmp_path: Path,
) -> None:
    """Written on the same pass as ``task``, from the same place.

    ``task_pattern`` is the label the recall map keys its cache on, and until
    it was a column the delivery history could not be filtered by it — so a
    client whose ``task`` changes every turn had an empty history under every
    key it ever used. The value was never missing, only unqueryable: it has
    always been in the ambient JSON.
    """

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        event = store.record_recall_event(
            query="what is nearby",
            scope="project:lm",
            ambient_context={
                "task": "chat:5f3a91/turn-7",
                "task_pattern": "lm/recall-map",
            },
        )
        row = store.connection.execute(
            "SELECT task, task_pattern, ambient_context FROM recall_events WHERE id = ?",
            (event.id,),
        ).fetchone()

        assert row["task"] == "chat:5f3a91/turn-7"
        assert row["task_pattern"] == "lm/recall-map"
        # The column is a lift, not a move: the ambient JSON is unchanged, so
        # every existing reader of it keeps working.
        assert json.loads(row["ambient_context"])["task_pattern"] == "lm/recall-map"

        # A caller that sends none leaves NULL, which is what the read's
        # fallback to `task` keys on.
        bare = store.record_recall_event(query="q", scope="project:lm")
        assert (
            store.connection.execute(
                "SELECT task_pattern FROM recall_events WHERE id = ?", (bare.id,)
            ).fetchone()["task_pattern"]
            is None
        )


def test_recent_recall_map_history_filters_on_the_map_key_label(
    tmp_path: Path,
) -> None:
    """The read the curtail rule needs: one pattern, whatever the turn said.

    Three deliveries of one tree-goal, each under a task string no other
    delivery uses. Filtering by ``task`` returns one row per turn — which is
    why the streak never accumulated. Filtering by the pattern returns the
    history that actually belongs to the key.
    """

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        ids = [
            store.record_recall_event(
                query=f"turn {index}",
                scope="project:lm",
                ambient_context={
                    "task": f"chat:5f3a91/turn-{index}",
                    "task_pattern": "lm/recall-map",
                },
                recall_map={"clusters": [{"label": "alpha", "count": 2}], "pool": 9},
            ).id
            for index in range(3)
        ]
        other = store.record_recall_event(
            query="a different tree goal",
            scope="project:lm",
            ambient_context={
                "task": "chat:5f3a91/turn-9",
                "task_pattern": "lm/decay-sweep",
            },
            recall_map={"clusters": [{"label": "beta", "count": 1}], "pool": 4},
        ).id

        by_pattern = store.recent_recall_map_history(
            scope="project:lm", task="chat:5f3a91/turn-3", task_pattern="lm/recall-map"
        )
        assert [row["id"] for row in by_pattern] == list(reversed(ids))

        # ...and the neighbouring pattern in the same scope is not in it.
        assert other not in {row["id"] for row in by_pattern}
        assert [
            row["id"]
            for row in store.recent_recall_map_history(
                scope="project:lm", task_pattern="lm/decay-sweep"
            )
        ] == [other]

        # The task-only read is untouched: same call, same answer as before
        # the column existed.
        assert [
            row["id"]
            for row in store.recent_recall_map_history(
                scope="project:lm", task="chat:5f3a91/turn-1"
            )
        ] == [ids[1]]


def test_recent_recall_map_history_falls_back_to_task_for_patternless_rows(
    tmp_path: Path,
) -> None:
    """A row with no pattern is offered its only other identity, and no more.

    Every row written before the column existed is this shape. Without the
    fallback, deploying the column would blank every accrued delivery history
    in the database; with a fallback that ignored ``task`` it would merge every
    legacy delivery in the scope into whichever pattern asked first.
    """

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        payload = {"clusters": [{"label": "alpha", "count": 2}], "pool": 9}
        legacy = store.record_recall_event(
            query="before the column",
            scope="project:lm",
            ambient_context={"task": "recall-map/ae-contract-doc"},
            recall_map=payload,
        ).id
        stranger = store.record_recall_event(
            query="somebody else's legacy row",
            scope="project:lm",
            ambient_context={"task": "decay/ttl-sweep"},
            recall_map=payload,
        ).id

        matched = store.recent_recall_map_history(
            scope="project:lm",
            task="recall-map/ae-contract-doc",
            task_pattern="lm/recall-map",
        )
        assert [row["id"] for row in matched] == [legacy]
        assert stranger not in {row["id"] for row in matched}

        # A task-less pattern key sees only task-less legacy rows.
        assert (
            store.recent_recall_map_history(
                scope="project:lm", task_pattern="lm/recall-map"
            )
            == []
        )


def test_task_pattern_column_migrates_additively_and_keeps_column_order(
    tmp_path: Path,
) -> None:
    """A migrated file and a fresh one must be indistinguishable.

    Column *order* and not merely membership, because the DDL text is what
    SQLite reconstructs on ``ALTER TABLE ... DROP COLUMN`` and what every
    positional read of this table is written against. The two append-migrations
    run in DDL order — ``recall_map`` then ``task_pattern`` — so a database
    that has neither, one that has only the first, and a fresh one all converge.
    """

    fresh = tmp_path / "fresh.sqlite3"
    with MemoryStore(fresh) as store:
        expected = _recall_event_columns(store)
    assert expected[-2:] == ["recall_map", "task_pattern"]

    migrated = tmp_path / "migrated.sqlite3"
    with MemoryStore(migrated) as store:
        store.record_recall_event(query="seed", scope="project:lm")

    connection = sqlite3.connect(migrated)
    try:
        # A database predating the column predates the index that names it,
        # and SQLite refuses to drop a column any index still mentions.
        connection.execute("DROP INDEX idx_recall_events_scope_pattern_created")
        connection.execute("ALTER TABLE recall_events DROP COLUMN task_pattern")
        connection.execute("ALTER TABLE recall_events DROP COLUMN recall_map")
        connection.commit()
    finally:
        connection.close()

    # Reopening reconciles both, in order, and reopening again changes nothing.
    for _ in range(2):
        with MemoryStore(migrated) as store:
            assert _recall_event_columns(store) == expected
            index_names = {
                str(row["name"])
                for row in store.connection.execute(
                    "PRAGMA index_list('recall_events')"
                )
            }
            assert "idx_recall_events_scope_pattern_created" in index_names


def test_task_pattern_backfills_from_the_ambient_json_of_legacy_rows(
    tmp_path: Path,
) -> None:
    """The column starts as complete as the history it describes.

    Backfilled rather than left NULL because the value was never lost — every
    recall event stores its whole ambient context — and a curtail rule that
    began counting only at deploy time would be blind to exactly the accrued
    silence it exists to notice.
    """

    db = tmp_path / "legacy.sqlite3"
    with MemoryStore(db) as store:
        store.record_recall_event(
            query="carried a pattern",
            scope="project:lm",
            ambient_context={"task": "t", "task_pattern": "lm/recall-map"},
        )
        store.record_recall_event(
            query="carried none",
            scope="project:lm",
            ambient_context={"task": "t"},
        )

    connection = sqlite3.connect(db)
    try:
        # The shape of a database written before the column existed: the JSON
        # still has the value, the column does not exist yet.
        connection.execute("DROP INDEX idx_recall_events_scope_pattern_created")
        connection.execute("ALTER TABLE recall_events DROP COLUMN task_pattern")
        # ...plus one row no json_* function may be pointed at. A single
        # malformed legacy document must not make the database unopenable.
        connection.execute(
            """
            INSERT INTO recall_events (
                id, query, scope, requested_scope, resolved_scopes,
                ambient_context, created_at
            ) VALUES ('LEGACYBROKEN', 'q', 'project:lm', 'project:lm',
                      '["project:lm"]', 'not json at all{"task_pattern"', '2026-01-01T00:00:00Z')
            """
        )
        connection.commit()
    finally:
        connection.close()

    with MemoryStore(db) as store:
        lifted = {
            str(row["query"]): row["task_pattern"]
            for row in store.connection.execute(
                "SELECT query, task_pattern FROM recall_events"
            )
        }
        assert lifted["carried a pattern"] == "lm/recall-map"
        assert lifted["carried none"] is None
        assert lifted["q"] is None

    # Idempotent, and terminal: a second open finds nothing left to do.
    with MemoryStore(db) as store:
        before = store.connection.total_changes
        store._backfill_recall_events_task_pattern()
        assert store.connection.total_changes == before


def test_task_pattern_backfill_ignores_non_string_ambient_values(
    tmp_path: Path,
) -> None:
    """Only what the write path itself would have stored gets lifted.

    ``_optional_str`` on the write path yields a string or nothing, so a caller
    that put an object under the key never had a column; the backfill must not
    invent one for it by stringifying JSON.
    """

    db = tmp_path / "odd.sqlite3"
    with MemoryStore(db) as store:
        store.record_recall_event(query="seed", scope="project:lm")

    connection = sqlite3.connect(db)
    try:
        connection.execute("DROP INDEX idx_recall_events_scope_pattern_created")
        connection.execute("ALTER TABLE recall_events DROP COLUMN task_pattern")
        connection.execute(
            """
            UPDATE recall_events
            SET ambient_context = '{"task_pattern":{"nested":"object"}}'
            WHERE query = 'seed'
            """
        )
        connection.commit()
    finally:
        connection.close()

    with MemoryStore(db) as store:
        assert (
            store.connection.execute(
                "SELECT task_pattern FROM recall_events WHERE query = 'seed'"
            ).fetchone()["task_pattern"]
            is None
        )
