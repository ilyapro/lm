import hashlib
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
    SCHEMA_VERSION,
    FingerprintGatePolicy,
    MemoryStore,
    RecallFingerprintStats,
    _content_fingerprint,
    recall_fingerprint,
    should_gate_fingerprint,
)


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
        assert "recall_fingerprints" in tables
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
        # The whole migration chain runs on one open: the pre-v5 columns
        # land as well.
        assert {"fingerprint", "gated"}.issubset(columns)

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
                SELECT transport_matched, transport_consumed, fallback_consumed
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


def test_delivered_node_ids_accumulates_per_transport_session(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.record_recall_event(
            query="alpha query",
            scope="project:alpha",
            ambient_context={"transport_session_id": "transport-1"},
            results=[
                {"rank": 1, "node_id": "NODE-A"},
                {"rank": 2, "node_id": "NODE-B"},
            ],
        )
        store.record_recall_event(
            query="bravo query",
            scope="project:alpha",
            ambient_context={"transport_session_id": "transport-1"},
            results=[
                {"rank": 1, "node_id": "NODE-B"},
                {"rank": 2, "node_id": "NODE-C"},
                {"rank": 3, "note": "entry without a node id is skipped"},
            ],
        )
        # Same transport on another scope still accumulates: delivery history
        # follows the connection, not the scope.
        store.record_recall_event(
            query="charlie query",
            scope="project:beta",
            ambient_context={"transport_session_id": "transport-1"},
            results=[{"rank": 1, "node_id": "NODE-D"}],
        )
        store.record_recall_event(
            query="delta query",
            scope="project:alpha",
            ambient_context={"transport_session_id": "transport-2"},
            results=[{"rank": 1, "node_id": "NODE-E"}],
        )
        # Unstamped event: transport_session_id stays NULL.
        store.record_recall_event(
            query="echo query",
            scope="project:alpha",
            results=[{"rank": 1, "node_id": "NODE-F"}],
        )

        assert store.delivered_node_ids("transport-1") == {
            "NODE-A",
            "NODE-B",
            "NODE-C",
            "NODE-D",
        }
        assert store.delivered_node_ids("transport-2") == {"NODE-E"}
        assert store.delivered_node_ids("transport-unknown") == set()


def test_delivered_node_ids_none_or_empty_transport_returns_empty(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.record_recall_event(
            query="alpha query",
            scope="project:alpha",
            ambient_context={"transport_session_id": "transport-1"},
            results=[{"rank": 1, "node_id": "NODE-A"}],
        )
        assert store.delivered_node_ids(None) == set()
        assert store.delivered_node_ids("") == set()


def test_delivered_node_ids_caps_window_to_most_recent_events(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for index in range(5):
            store.record_recall_event(
                query=f"query {index}",
                scope="project:alpha",
                ambient_context={"transport_session_id": "transport-1"},
                results=[{"rank": 1, "node_id": f"NODE-{index}"}],
            )

        assert store.delivered_node_ids("transport-1") == {
            f"NODE-{index}" for index in range(5)
        }
        # created_at ties break by rowid, matching list_recall_events
        # ordering: the window keeps the newest inserts.
        assert store.delivered_node_ids("transport-1", max_events=2) == {
            "NODE-3",
            "NODE-4",
        }
        assert store.delivered_node_ids("transport-1", max_events=0) == set()


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


def _stats(
    *,
    delivery_count: int,
    linked_count: int = 0,
    deliveries_since_link: int = 0,
    transport_session_count: int = 2,
) -> RecallFingerprintStats:
    return RecallFingerprintStats(
        fingerprint="f" * 64,
        first_seen="2026-01-01T00:00:00Z",
        last_seen="2026-01-01T00:10:00Z",
        delivery_count=delivery_count,
        linked_count=linked_count,
        deliveries_since_link=deliveries_since_link,
        last_linked_at=None,
        transport_session_count=transport_session_count,
        last_transport_session_id="t-1",
        updated_at="2026-01-01T00:10:00Z",
    )


def test_recall_fingerprint_collapses_whitespace_and_binds_scope() -> None:
    base = recall_fingerprint("goal api recipe", "project:alpha")
    assert base == recall_fingerprint("  goal   api\trecipe \n", "project:alpha")
    assert base != recall_fingerprint("goal api recipe", "project:beta")
    assert base != recall_fingerprint("goal api recipes", "project:alpha")
    assert base != recall_fingerprint("goal api recipe", None)
    # The payload encoding is a stability contract: stored fingerprints must
    # keep matching recomputed ones across releases.
    assert base == hashlib.sha256(b"goal api recipe\nproject:alpha").hexdigest()


def test_schema_v4_database_migrates_to_v5_with_fingerprint_signal(tmp_path: Path) -> None:
    """A pre-v5 fixture DB must gain the fingerprint/gated columns, stamped
    fingerprints, and rebuilt recall_fingerprints aggregates on open."""

    db = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('schema_version', '4')"
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
                transport_session_id TEXT,
                feedback_applied INTEGER NOT NULL DEFAULT 0 CHECK (feedback_applied IN (0, 1)),
                feedback_trace_id TEXT REFERENCES nodes(id),
                feedback_applied_at TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        seeded = (
            # (id, query, transport, feedback_applied, applied_at, created_at, results_json)
            ("REPEAT01", "repeat  query", "t-1", 0, None, "2026-01-01T00:00:00Z", '[{"node_id":"NODE-A","rank":1}]'),
            ("REPEAT02", "repeat query", "t-1", 1, "2026-01-01T00:00:05Z", "2026-01-01T00:00:01Z", '[{"node_id":"NODE-B","rank":1}]'),
            ("REPEAT03", "repeat query", "t-2", 0, None, "2026-01-01T00:00:02Z", "[]"),
            ("REPEAT04", "repeat query", None, 0, None, "2026-01-01T00:00:03Z", "[]"),
            ("UNIQUE01", "unique query", "t-1", 0, None, "2026-01-01T00:00:04Z", "[]"),
            ("REPEAT05", "repeat query", "t-2", 0, None, "2026-01-01T00:00:06Z", '[{"node_id":"NODE-C","rank":1}]'),
        )
        for event_id, query, transport, applied, applied_at, created_at, results_json in seeded:
            conn.execute(
                """
                INSERT INTO recall_events (
                    id, query, scope, requested_scope, resolved_scopes,
                    transport_session_id, results, feedback_applied,
                    feedback_applied_at, created_at
                )
                VALUES (?, ?, 'project:legacy', 'project:legacy',
                        '["project:legacy"]', ?, ?, ?, ?, ?)
                """,
                (event_id, query, transport, results_json, applied, applied_at, created_at),
            )
        conn.commit()
    finally:
        conn.close()

    repeat_fp = recall_fingerprint("repeat query", "project:legacy")
    unique_fp = recall_fingerprint("unique query", "project:legacy")

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
        assert {"fingerprint", "gated"}.issubset(columns)

        index_names = {
            r["name"]
            for r in store.connection.execute(
                "PRAGMA index_list('recall_events')"
            ).fetchall()
        }
        assert "idx_recall_events_fingerprint_created" in index_names

        stamped = {
            str(r["id"]): (r["fingerprint"], r["gated"])
            for r in store.connection.execute(
                "SELECT id, fingerprint, gated FROM recall_events"
            )
        }
        # The whitespace-variant REPEAT01 collapses onto the same fingerprint.
        for event_id in ("REPEAT01", "REPEAT02", "REPEAT03", "REPEAT04", "REPEAT05"):
            assert stamped[event_id] == (repeat_fp, 0)
        assert stamped["UNIQUE01"] == (unique_fp, 0)

        stats = store.get_recall_fingerprint_stats(repeat_fp)
        assert stats is not None
        assert stats.delivery_count == 5
        assert stats.linked_count == 1
        # The link applied at 00:00:05 lands after the 00:00:03 delivery and
        # before the 00:00:06 one: only REPEAT05 counts as since-link.
        assert stats.deliveries_since_link == 1
        # t-1, t-1 (same), t-2 (new), NULL (distinct), t-2 (after NULL: new).
        assert stats.transport_session_count == 4
        assert stats.first_seen == "2026-01-01T00:00:00Z"
        assert stats.last_seen == "2026-01-01T00:00:06Z"
        assert stats.last_linked_at == "2026-01-01T00:00:05Z"
        assert stats.last_transport_session_id == "t-2"

        unique_stats = store.get_recall_fingerprint_stats(unique_fp)
        assert unique_stats is not None
        assert unique_stats.delivery_count == 1
        assert unique_stats.linked_count == 0
        assert unique_stats.deliveries_since_link == 1
        assert unique_stats.transport_session_count == 1

        # Legacy reads keep working over migrated rows.
        by_id = {
            event.id: event
            for event in store.list_recall_events(scope="project:legacy")
        }
        assert set(by_id) == {entry[0] for entry in seeded}
        assert by_id["REPEAT02"].feedback_applied is True
        assert by_id["REPEAT02"].results == [{"node_id": "NODE-B", "rank": 1}]
        assert by_id["REPEAT04"].transport_session_id is None

        assert store.fingerprint_delivered_node_ids(repeat_fp) == {
            "NODE-A",
            "NODE-B",
            "NODE-C",
        }
        assert store.fingerprint_delivered_node_ids(repeat_fp, max_events=1) == {"NODE-C"}

        # The migrated aggregates feed the gate rule directly (explicit
        # policy values; the default constants are not pinned here).
        policy = FingerprintGatePolicy(
            enabled=True,
            min_unlinked=1,
            max_link_rate=0.5,
            min_sessions=2,
            probe_every=0,
        )
        assert should_gate_fingerprint(stats, policy) is True
        # The unique fingerprint never repeated across sessions: no gate.
        assert should_gate_fingerprint(unique_stats, policy) is False


def test_v5_backfill_is_idempotent_and_preserves_online_aggregates(tmp_path: Path) -> None:
    db = tmp_path / "memory.sqlite3"
    fp = recall_fingerprint("hotel query", "project:alpha")
    with MemoryStore(db) as store:
        store.record_recall_event(
            query="hotel query",
            scope="project:alpha",
            ambient_context={"transport_session_id": "t-1"},
        )
        store.record_recall_event(
            query="hotel query",
            scope="project:alpha",
            ambient_context={"transport_session_id": "t-2"},
        )
        before = store.get_recall_fingerprint_stats(fp)
        assert before is not None
        assert before.delivery_count == 2

    # Reopening runs _initialize_schema again: with every event already
    # stamped, the backfill must not touch the aggregates.
    with MemoryStore(db) as store:
        assert store.get_recall_fingerprint_stats(fp) == before
        store.record_recall_event(
            query="hotel query",
            scope="project:alpha",
            ambient_context={"transport_session_id": "t-2"},
        )
        after = store.get_recall_fingerprint_stats(fp)
        assert after is not None
        assert after.delivery_count == 3
        # Same transport as the persisted last stamp: no new session.
        assert after.transport_session_count == 2

    with MemoryStore(db) as store:
        assert store.get_recall_fingerprint_stats(fp) == after


def test_record_recall_event_stamps_fingerprint_and_counts_sessions(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        first = store.record_recall_event(
            query="alpha   beta",
            scope="project:alpha",
            ambient_context={"transport_session_id": "t-1"},
        )
        fp = recall_fingerprint("alpha beta", "project:alpha")
        row = store.connection.execute(
            "SELECT fingerprint, gated FROM recall_events WHERE id = ?",
            (first.id,),
        ).fetchone()
        assert row["fingerprint"] == fp
        assert row["gated"] == 0

        def counters() -> tuple[int, int, int, int]:
            stats = store.get_recall_fingerprint_stats(fp)
            assert stats is not None
            return (
                stats.delivery_count,
                stats.linked_count,
                stats.deliveries_since_link,
                stats.transport_session_count,
            )

        assert counters() == (1, 0, 1, 1)

        store.record_recall_event(
            query="alpha beta",
            scope="project:alpha",
            ambient_context={"transport_session_id": "t-1"},
        )
        assert counters() == (2, 0, 2, 1)

        store.record_recall_event(
            query="alpha beta",
            scope="project:alpha",
            ambient_context={"transport_session_id": "t-2"},
        )
        assert counters() == (3, 0, 3, 2)

        # A missing transport id always counts as a distinct session...
        store.record_recall_event(query="alpha beta", scope="project:alpha")
        assert counters() == (4, 0, 4, 3)
        # ...including consecutively missing ones.
        store.record_recall_event(query="alpha beta", scope="project:alpha")
        assert counters() == (5, 0, 5, 4)
        # And a stamped session after a missing one is new again.
        store.record_recall_event(
            query="alpha beta",
            scope="project:alpha",
            ambient_context={"transport_session_id": "t-2"},
        )
        assert counters() == (6, 0, 6, 5)

        # The fingerprint binds the requested scope: same query elsewhere is
        # a separate aggregate.
        store.record_recall_event(query="alpha beta", scope="project:beta")
        other = store.get_recall_fingerprint_stats(
            recall_fingerprint("alpha beta", "project:beta")
        )
        assert other is not None
        assert other.delivery_count == 1
        assert counters() == (6, 0, 6, 5)

        # An explicit requested_scope wins over the resolved scope, matching
        # the stamped requested_scope column.
        recorded = store.record_recall_event(
            query="alpha beta",
            scope="resolved:z",
            requested_scope="asked:w",
        )
        row = store.connection.execute(
            "SELECT fingerprint FROM recall_events WHERE id = ?",
            (recorded.id,),
        ).fetchone()
        assert row["fingerprint"] == recall_fingerprint("alpha beta", "asked:w")

        assert store.get_recall_fingerprint_stats(None) is None
        assert store.get_recall_fingerprint_stats("") is None
        assert store.get_recall_fingerprint_stats(
            recall_fingerprint("never seen", "global")
        ) is None


def test_mark_recall_event_feedback_credits_fingerprint_only_on_first_link(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        trace = store.append_trace("closure trace", {"scope": "project:alpha"})
        other_trace = store.append_trace("second closure trace", {"scope": "project:alpha"})
        fp = recall_fingerprint("gamma delta", "project:alpha")
        first = store.record_recall_event(
            query="gamma delta",
            scope="project:alpha",
            ambient_context={"transport_session_id": "t-1"},
        )
        second = store.record_recall_event(
            query="gamma delta",
            scope="project:alpha",
            ambient_context={"transport_session_id": "t-2"},
        )

        stats = store.get_recall_fingerprint_stats(fp)
        assert stats is not None
        assert (stats.delivery_count, stats.linked_count, stats.deliveries_since_link) == (2, 0, 2)
        assert stats.last_linked_at is None

        marked = store.mark_recall_event_feedback(first.id, trace.id)
        assert marked.feedback_applied is True
        stats = store.get_recall_fingerprint_stats(fp)
        assert (stats.linked_count, stats.deliveries_since_link) == (1, 0)
        assert stats.last_linked_at is not None

        # Fresh deliveries after the link re-accumulate the unlinked counter.
        store.record_recall_event(
            query="gamma delta",
            scope="project:alpha",
            ambient_context={"transport_session_id": "t-1"},
        )
        stats = store.get_recall_fingerprint_stats(fp)
        assert (stats.delivery_count, stats.linked_count, stats.deliveries_since_link) == (3, 1, 1)

        # Re-marking the same event keeps the legacy trace-pointer overwrite
        # but must not credit the aggregate again.
        remarked = store.mark_recall_event_feedback(first.id, other_trace.id)
        assert remarked.feedback_trace_id == other_trace.id
        stats = store.get_recall_fingerprint_stats(fp)
        assert (stats.linked_count, stats.deliveries_since_link) == (1, 1)

        # A different event's 0 -> 1 flip credits again and resets the counter.
        store.mark_recall_event_feedback(second.id, trace.id)
        stats = store.get_recall_fingerprint_stats(fp)
        assert (stats.linked_count, stats.deliveries_since_link) == (2, 0)


def test_mark_recall_event_gated_flags_only_that_event(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        first = store.record_recall_event(query="epsilon", scope="project:alpha")
        second = store.record_recall_event(query="epsilon", scope="project:alpha")
        store.mark_recall_event_gated(second.id)
        gated = {
            str(row["id"]): int(row["gated"])
            for row in store.connection.execute(
                "SELECT id, gated FROM recall_events"
            )
        }
        assert gated == {first.id: 0, second.id: 1}
        with pytest.raises(KeyError):
            store.mark_recall_event_gated("MISSING-EVENT-ID")


def test_fingerprint_delivered_node_ids_unions_and_caps_window(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        fp = recall_fingerprint("zeta query", "project:alpha")
        for index in range(5):
            store.record_recall_event(
                query="zeta query",
                scope="project:alpha",
                ambient_context={"transport_session_id": f"t-{index}"},
                results=[
                    {"rank": 1, "node_id": f"NODE-{index}"},
                    {"rank": 2, "note": "entry without a node id is skipped"},
                ],
            )
        # A different fingerprint's deliveries stay invisible.
        store.record_recall_event(
            query="other query",
            scope="project:alpha",
            results=[{"rank": 1, "node_id": "NODE-X"}],
        )

        assert store.fingerprint_delivered_node_ids(fp) == {
            f"NODE-{index}" for index in range(5)
        }
        # created_at ties break by rowid, matching delivered_node_ids: the
        # window keeps the newest inserts.
        assert store.fingerprint_delivered_node_ids(fp, max_events=2) == {
            "NODE-3",
            "NODE-4",
        }
        assert store.fingerprint_delivered_node_ids(fp, max_events=0) == set()
        assert store.fingerprint_delivered_node_ids(None) == set()
        assert store.fingerprint_delivered_node_ids("") == set()
        assert store.fingerprint_delivered_node_ids(
            recall_fingerprint("unseen", "global")
        ) == set()

        plan = " ".join(
            row["detail"]
            for row in store.connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT results FROM recall_events
                WHERE fingerprint = 'aaaa'
                ORDER BY created_at DESC, rowid DESC
                LIMIT 200
                """
            ).fetchall()
        )
        assert "idx_recall_events_fingerprint_created" in plan


@pytest.mark.parametrize(
    "policy",
    [
        FingerprintGatePolicy(
            enabled=True, min_unlinked=3, max_link_rate=0.34, min_sessions=2, probe_every=0
        ),
        FingerprintGatePolicy(
            enabled=True, min_unlinked=6, max_link_rate=0.15, min_sessions=3, probe_every=0
        ),
    ],
)
def test_should_gate_fingerprint_threshold_properties(policy: FingerprintGatePolicy) -> None:
    # Below the unlinked threshold nothing gates, however heavy the repetition.
    for since in range(policy.min_unlinked):
        assert not should_gate_fingerprint(
            _stats(
                delivery_count=40,
                deliveries_since_link=since,
                transport_session_count=policy.min_sessions + 3,
            ),
            policy,
        )

    # Long unlinked repetition across enough sessions gates.
    assert should_gate_fingerprint(
        _stats(
            delivery_count=40,
            deliveries_since_link=40,
            transport_session_count=policy.min_sessions,
        ),
        policy,
    )

    # Any link resets the counter: the next min_unlinked deliveries stay full.
    for since in range(policy.min_unlinked):
        assert not should_gate_fingerprint(
            _stats(
                delivery_count=41,
                linked_count=1,
                deliveries_since_link=since,
                transport_session_count=policy.min_sessions + 3,
            ),
            policy,
        )

    # High-linkage fingerprints never gate, even under heavy repetition.
    assert not should_gate_fingerprint(
        _stats(
            delivery_count=40,
            linked_count=20,
            deliveries_since_link=40,
            transport_session_count=policy.min_sessions + 3,
        ),
        policy,
    )

    # Too few distinct transport sessions never gates.
    assert not should_gate_fingerprint(
        _stats(
            delivery_count=40,
            deliveries_since_link=40,
            transport_session_count=policy.min_sessions - 1,
        ),
        policy,
    )

    # Master valve off: nothing gates.
    assert not should_gate_fingerprint(
        _stats(
            delivery_count=40,
            deliveries_since_link=40,
            transport_session_count=policy.min_sessions,
        ),
        replace(policy, enabled=False),
    )

    # Unknown fingerprints never gate.
    assert not should_gate_fingerprint(None, policy)


def test_should_gate_fingerprint_uses_laplace_smoothed_link_rate() -> None:
    stats = _stats(
        delivery_count=5, linked_count=1, deliveries_since_link=3, transport_session_count=2
    )
    base = FingerprintGatePolicy(
        enabled=True, min_unlinked=1, max_link_rate=0.28, min_sessions=1, probe_every=0
    )
    # The raw rate is 1/5 = 0.20 <= 0.28, but the Laplace-smoothed rate
    # (1+1)/(5+2) ~= 0.286 exceeds the ceiling: no gate.
    assert stats.smoothed_link_rate == pytest.approx(2 / 7)
    assert not should_gate_fingerprint(stats, base)
    assert should_gate_fingerprint(stats, replace(base, max_link_rate=0.29))


def test_should_gate_fingerprint_probe_cadence_forces_periodic_full_delivery() -> None:
    policy = FingerprintGatePolicy(
        enabled=True, min_unlinked=4, max_link_rate=0.5, min_sessions=1, probe_every=6
    )

    def decide(since: int, active: FingerprintGatePolicy) -> bool:
        return should_gate_fingerprint(
            _stats(
                delivery_count=since,
                deliveries_since_link=since,
                transport_session_count=2,
            ),
            active,
        )

    decisions = [
        decide(since, policy)
        for since in range(policy.min_unlinked, policy.min_unlinked + 40)
    ]
    # Gating engages between probes...
    assert any(decisions)
    # ...but every probe_every consecutive gate-eligible deliveries include
    # at least one full (non-gated) probe delivery.
    for start in range(len(decisions) - policy.probe_every + 1):
        assert False in decisions[start : start + policy.probe_every]
    # The probe is deterministic: the same stats always decide the same way.
    assert decisions == [
        decide(since, policy)
        for since in range(policy.min_unlinked, policy.min_unlinked + 40)
    ]

    # probe_every=0 disables probing: the whole eligible run gates.
    no_probe = replace(policy, probe_every=0)
    assert all(
        decide(since, no_probe)
        for since in range(no_probe.min_unlinked, no_probe.min_unlinked + 40)
    )


def test_fingerprint_gate_policy_from_env_reads_pinned_knobs() -> None:
    defaults = FingerprintGatePolicy()
    assert FingerprintGatePolicy.from_env(env={}) == defaults
    assert defaults.enabled is False
    assert defaults.drop_trailing_stubs is False

    custom = FingerprintGatePolicy.from_env(
        env={
            "LM_RECALL_REPEAT_GATING": " yes ",
            "LM_RECALL_REPEAT_DROP_TRAILING_STUBS": "ON",
            "LM_RECALL_REPEAT_MIN_UNLINKED": "7",
            "LM_RECALL_REPEAT_MAX_LINK_RATE": "0.35",
            "LM_RECALL_REPEAT_MIN_SESSIONS": "4",
            "LM_RECALL_REPEAT_PROBE_EVERY": "11",
        }
    )
    assert custom == FingerprintGatePolicy(
        enabled=True,
        min_unlinked=7,
        max_link_rate=0.35,
        min_sessions=4,
        probe_every=11,
        drop_trailing_stubs=True,
    )

    # Numeric parsing remains independent: invalid values use the threshold
    # defaults, rates clamp into [0, 1], and counters clamp at zero.
    malformed = FingerprintGatePolicy.from_env(
        env={
            "LM_RECALL_REPEAT_MIN_UNLINKED": "not-a-number",
            "LM_RECALL_REPEAT_MAX_LINK_RATE": "2.5",
            "LM_RECALL_REPEAT_MIN_SESSIONS": "",
            "LM_RECALL_REPEAT_PROBE_EVERY": "-3",
        }
    )
    assert malformed.enabled is False
    assert malformed.drop_trailing_stubs is False
    assert malformed.min_unlinked == defaults.min_unlinked
    assert malformed.max_link_rate == pytest.approx(1.0)
    assert malformed.min_sessions == defaults.min_sessions
    assert malformed.probe_every == 0


@pytest.mark.parametrize(
    ("env_name", "attribute", "other_attribute"),
    [
        ("LM_RECALL_REPEAT_GATING", "enabled", "drop_trailing_stubs"),
        (
            "LM_RECALL_REPEAT_DROP_TRAILING_STUBS",
            "drop_trailing_stubs",
            "enabled",
        ),
    ],
)
@pytest.mark.parametrize("value", ["1", " true ", "YES", "\tOn\n"])
def test_fingerprint_gate_policy_repeat_flags_accept_only_explicit_true_values(
    env_name: str,
    attribute: str,
    other_attribute: str,
    value: str,
) -> None:
    policy = FingerprintGatePolicy.from_env(env={env_name: value})

    assert getattr(policy, attribute) is True
    assert getattr(policy, other_attribute) is False


@pytest.mark.parametrize(
    ("env_name", "attribute", "other_env_name", "other_attribute"),
    [
        (
            "LM_RECALL_REPEAT_GATING",
            "enabled",
            "LM_RECALL_REPEAT_DROP_TRAILING_STUBS",
            "drop_trailing_stubs",
        ),
        (
            "LM_RECALL_REPEAT_DROP_TRAILING_STUBS",
            "drop_trailing_stubs",
            "LM_RECALL_REPEAT_GATING",
            "enabled",
        ),
    ],
)
@pytest.mark.parametrize(
    "value",
    [None, "", "   ", "0", "false", "NO", "Off", "2", "enabled", "tru"],
)
def test_fingerprint_gate_policy_repeat_flags_reject_non_true_values(
    env_name: str,
    attribute: str,
    other_env_name: str,
    other_attribute: str,
    value: str | None,
) -> None:
    env = {other_env_name: "1"}
    if value is not None:
        env[env_name] = value
    policy = FingerprintGatePolicy.from_env(env=env)

    assert getattr(policy, attribute) is False
    assert getattr(policy, other_attribute) is True


def test_fingerprint_gate_policy_from_env_defaults_to_process_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LM_RECALL_REPEAT_GATING", raising=False)
    monkeypatch.delenv("LM_RECALL_REPEAT_DROP_TRAILING_STUBS", raising=False)
    defaults = FingerprintGatePolicy.from_env()
    assert defaults.enabled is False
    assert defaults.drop_trailing_stubs is False

    monkeypatch.setenv("LM_RECALL_REPEAT_GATING", " true ")
    monkeypatch.setenv("LM_RECALL_REPEAT_DROP_TRAILING_STUBS", "yes")
    monkeypatch.setenv("LM_RECALL_REPEAT_MIN_UNLINKED", "9")
    policy = FingerprintGatePolicy.from_env()
    assert policy.enabled is True
    assert policy.drop_trailing_stubs is True
    assert policy.min_unlinked == 9
