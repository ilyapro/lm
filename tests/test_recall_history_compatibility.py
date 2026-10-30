"""Ordinary recall and feedback work with inert legacy database objects."""

import asyncio
from dataclasses import replace
import sqlite3

import pytest
from fastmcp import Client

from living_memory import storage
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore

SCOPE = "project:history-compatibility"
QUERY = "harbour survey procedure"
CONTENT = "Harbour survey procedure requires the pilot to countersign on Tuesday."


def legacy_signal(store):
    """Populate the schema-v5 objects, including an unstamped legacy event."""
    event = store.record_recall_event(query=QUERY, scope=SCOPE)
    unstamped = store.record_recall_event(query="older query", scope=SCOPE)
    store.connection.executescript("""
        ALTER TABLE recall_events ADD COLUMN fingerprint TEXT;
        ALTER TABLE recall_events ADD COLUMN gated INTEGER NOT NULL DEFAULT 0;
        CREATE TABLE recall_fingerprints (
            fingerprint TEXT PRIMARY KEY,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            delivery_count INTEGER NOT NULL DEFAULT 0,
            linked_count INTEGER NOT NULL DEFAULT 0,
            deliveries_since_link INTEGER NOT NULL DEFAULT 0,
            last_linked_at TEXT,
            transport_session_count INTEGER NOT NULL DEFAULT 0,
            last_transport_session_id TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_recall_events_fingerprint_created
            ON recall_events(fingerprint, created_at DESC);
        INSERT INTO recall_fingerprints VALUES (
            'historical', '2026-01-01', '2026-01-02', 7, 2, 3,
            '2026-01-02', 4, 'old-session', '2026-01-02'
        );
        CREATE TRIGGER reject_fingerprint_insert BEFORE INSERT ON recall_fingerprints
            BEGIN SELECT RAISE(ABORT, 'unavailable aggregate'); END;
        CREATE TRIGGER reject_fingerprint_update BEFORE UPDATE ON recall_fingerprints
            BEGIN SELECT RAISE(ABORT, 'unavailable aggregate'); END;
        CREATE TRIGGER reject_fingerprint_delete BEFORE DELETE ON recall_fingerprints
            BEGIN SELECT RAISE(ABORT, 'unavailable aggregate'); END;
    """)
    store.connection.execute(
        "UPDATE recall_events SET fingerprint = 'historical', gated = 1 WHERE id = ?",
        (event.id,),
    )
    store.connection.commit()
    return event.id, unstamped.id


def deny_aggregate(accesses):
    def authorize(action, table, column, database, source):
        if table == "recall_fingerprints":
            accesses.append((action, table, column))
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return authorize


def test_open_preserves_legacy_rows_without_reading_or_rebuilding_signal(tmp_path, monkeypatch):
    db = tmp_path / "legacy.sqlite3"
    with MemoryStore(db) as store:
        ids = legacy_signal(store)
        events = [tuple(row) for row in store.connection.execute("SELECT * FROM recall_events")]
        aggregate = [tuple(row) for row in store.connection.execute("SELECT * FROM recall_fingerprints")]

    connect = sqlite3.connect
    accesses = []

    def guarded_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connection.set_authorizer(deny_aggregate(accesses))
        return connection

    monkeypatch.setattr(storage.sqlite3, "connect", guarded_connect)
    for _ in range(2):
        with MemoryStore(db) as store:
            assert [tuple(row) for row in store.connection.execute("SELECT * FROM recall_events")] == events
            assert {event.id for event in store.list_recall_events()} == set(ids)
    assert accesses == []
    with connect(db) as connection:
        assert connection.execute("SELECT * FROM recall_fingerprints").fetchall() == aggregate


@pytest.mark.parametrize("view", ["knowledge", "full"])
def test_mcp_recall_delivery_and_feedback_ignore_unavailable_aggregate(tmp_path, monkeypatch, view):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_PROMPT", "mandatory")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    legacy_signal(store)
    node = store.create_node(level="concept", content=CONTENT, context={"scope": SCOPE})
    accesses = []
    store.connection.set_authorizer(deny_aggregate(accesses))

    async def call(client, tool, **arguments):
        result = await client.call_tool(tool, arguments)
        assert not result.is_error
        return result.structured_content

    async def scenario():
        async with Client(mcp) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            assert "REQUIRED after a recall" in tools["memory_recall"].inputSchema["properties"]["used"]["description"]
            first = await call(client, "memory_recall", query=QUERY, scope=SCOPE,
                               depth=0, max_results=1, view=view, used=[], irrelevant=[])
            assert first["results"][0]["delivery"] == "full"
            repeated = await call(client, "memory_recall", query=QUERY, scope=SCOPE,
                                  depth=0, max_results=1, view=view, used=[node.id], irrelevant=[])
            assert repeated["results"][0]["delivery"] == "session_duplicate"
            assert repeated["feedback_marks"]["accepted"] == 1
            remembered = await call(client, "memory_remember", content=CONTENT + " Confirmed.",
                                    context={"scope": SCOPE}, used=[node.id], irrelevant=[])
            trace_id = remembered["node"]["id"]
            for response in (first, repeated):
                event = store.get_recall_event(response["recall_event_id"])
                assert event.result_ids == [node.id]
                assert event.feedback_applied and event.feedback_trace_id == trace_id
            assert store.get_node(node.id).usefulness_score > 0
            assert store.connection.execute("SELECT COUNT(*) FROM recall_explicit_credit").fetchone()[0] == 2
            assert store.connection.execute("SELECT COUNT(*) FROM recall_feedback_marks WHERE accepted = 1").fetchone()[0] == 2
        async with Client(mcp) as client:
            fresh = await call(client, "memory_recall", query=QUERY, scope=SCOPE,
                               depth=0, max_results=1, view=view, used=[], irrelevant=[])
            assert fresh["results"][0]["delivery"] == "full"

    try:
        asyncio.run(scenario())
        assert accesses == []
    finally:
        store.close()


@pytest.mark.parametrize("decayed", [False, True])
@pytest.mark.parametrize("node_id, reference", [(" Trace ", " Trace "), ("1", 1)])
def test_feedback_closure_preserves_attribution_and_rejects_missing_references(
    monkeypatch, decayed, node_id, reference,
):
    with MemoryStore(":memory:") as store:
        event = store.record_recall_event(
            query=QUERY, scope=SCOPE, context={"session_id": "history-session"},
            results=[{"node_id": node_id, "bm25_score": 0.75}],
        )
        first = store.append_trace("first consumer", node_id=node_id)
        second = store.append_trace("second consumer")
        if decayed:
            store.soft_delete_node(first.id)

        def no_hydration(row):
            pytest.fail("feedback closure must not hydrate a Node")

        monkeypatch.setattr(storage, "_node_from_row", no_hydration)
        closed = store.mark_recall_event_feedback(event.id, reference)
        assert closed.feedback_applied_at
        assert closed == replace(
            event, feedback_applied=True, feedback_trace_id=first.id,
            feedback_applied_at=closed.feedback_applied_at,
        )
        assert store.mark_recall_event_feedback(event.id, second.id).feedback_trace_id == second.id
        before = list(store.connection.iterdump())
        for event_id, trace_id, missing in [
            ("absent-event", first.id, "absent-event"),
            (event.id, "absent-trace", "absent-trace"),
            ("absent-event", "absent-trace", "absent-trace"),
            (event.id, "Trace", "Trace"),
            (event.id, " trace ", " trace "),
        ]:
            with pytest.raises(KeyError) as error:
                store.mark_recall_event_feedback(event_id, trace_id)
            assert error.value.args == (missing,)
            assert not store.connection.in_transaction
            assert list(store.connection.iterdump()) == before
        assert store.get_recall_event(event.id).feedback_trace_id == second.id


@pytest.mark.parametrize("column", ["id", "decayed", "content", "context", "provenance"])
def test_feedback_closure_requires_only_endpoint_columns(column):
    with MemoryStore(":memory:") as store:
        trace = store.append_trace("consumer")
        event = store.record_recall_event(query=QUERY, scope=SCOPE)
        before = list(store.connection.iterdump())

        def authorize(action, table, name, *_):
            if action == sqlite3.SQLITE_READ and table == "nodes" and name == column:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        store.connection.set_authorizer(authorize)
        try:
            if column in {"id", "decayed"}:
                with pytest.raises(sqlite3.DatabaseError, match=f"nodes.{column}"):
                    store.mark_recall_event_feedback(event.id, trace.id)
            else:
                assert store.mark_recall_event_feedback(event.id, trace.id).feedback_applied
        finally:
            store.connection.set_authorizer(None)
        assert not store.connection.in_transaction
        if column in {"id", "decayed"}:
            assert list(store.connection.iterdump()) == before
