"""Preflight reads current query inputs; required feedback still reads full events."""

import asyncio
import itertools
import json
import re
import sqlite3
from dataclasses import asdict

from fastmcp import Client
from fastmcp.exceptions import ToolError
import pytest

from living_memory import feedback, server, storage
from living_memory.models import RecallEvent
from living_memory.storage import MemoryStore, recall_fingerprint
from test_derived_read_preparation import runtime
from test_lookup import FakeMCP
from test_recall_query_preparation import CONTENT, SCOPE, call


CONTEXT = {"scope": SCOPE, "task": "pending-inputs", "session_id": "pending-inputs"}
BODY = "unused-delivery-body-" + "d" * 16000
AMBIENT = "unused-ambient-context-" + "a" * 16000


def queries(store):
    return feedback._feedback_queries(
        store, used=None, irrelevant=None, transport_session_id=None,
        occurred_at=None, lookup_ids=None, trace_content=CONTENT,
        trace_context=CONTEXT, context=CONTEXT,
    )


@pytest.mark.parametrize("raw", [
    json.dumps([{"node_id": value} for value in [
        "same", "same", None, False, True, 0, -0.0, 1.5, 2**100, "", [], {},
        [1, False], {"b": 2, "a": True}, "雪",
    ]] + [{"other": "missing"}]),
    '[{"node_id":"first","node_id":"last"}]',
    '[[["node_id","first"],["node_id","last"]]]',
    '[{"node_id":{"key":1,"key":2}}]',
    '[{"\\u006eode_id":"escaped"}]',
    '[{"node_id":NaN},{"node_id":Infinity},{"node_id":-Infinity}]',
    '[{"node_id":1e999}]', '[]', '[[]]', '[{}]', '{}', '""', '',
])
def test_result_ids_match_full_reader_including_legacy_shapes(raw):
    with MemoryStore(":memory:") as store:
        store.record_recall_event(query="query", scope=SCOPE, ambient_context=CONTEXT)
        with store.transaction():
            store.connection.execute("UPDATE recall_events SET results = ?", (raw,))
        full, = store.pending_recall_events(scope=SCOPE, context=CONTEXT)
        narrow, = store.pending_recall_query_inputs(scope=SCOPE, context=CONTEXT)
        assert (narrow.id, narrow.query, narrow.scope, narrow.result_ids) == (
            full.id, full.query, full.scope, full.result_ids,
        )


@pytest.mark.parametrize("raw,error", [
    ('{', json.JSONDecodeError), ('null', TypeError), ('42', TypeError),
    ('[null]', TypeError), ('[3]', TypeError), ('["abc"]', ValueError),
    ('[[["node_id"]]]', ValueError),
])
@pytest.mark.parametrize("existing", [False, True])
def test_selected_results_errors_propagate_before_preflight_yields(raw, error, existing):
    with MemoryStore(":memory:") as store:
        event = store.record_recall_event(query="query", scope=SCOPE, ambient_context=CONTEXT)
        if existing:
            store.insert_query_anchor(scope=SCOPE, query=event.query,
                                      fingerprint=recall_fingerprint(event.query, SCOPE),
                                      embedding=[1.0, 0.0])
        with store.transaction():
            store.connection.execute("UPDATE recall_events SET results = ?", (raw,))
        with pytest.raises(error) as full:
            store.pending_recall_events(scope=SCOPE, context=CONTEXT)
        with pytest.raises(error) as narrow:
            queries(store)
        assert str(narrow.value) == str(full.value)
        assert not store.connection.in_transaction


def test_projection_happens_after_cap_and_uses_current_results_without_history():
    with MemoryStore(":memory:") as store:
        old = store.record_recall_event(query="old", scope=SCOPE, ambient_context=CONTEXT)
        current = store.record_recall_event(query="current", scope=SCOPE, ambient_context=CONTEXT,
                                            results=[{"node_id": "frozen"}])
        store.connection.execute("UPDATE recall_events SET results = '{' WHERE id = ?", (old.id,))
        store.connection.execute("UPDATE recall_events SET results = ? WHERE id = ?",
                                 ('[{"node_id":"current"},{"node_id":"current"}]', current.id))
        store.connection.execute("DROP TABLE recall_history_result_nodes")
        for reader in (store.pending_recall_events, store.pending_recall_query_inputs):
            selected, = reader(scope=SCOPE, context=CONTEXT, limit=1)
            assert selected.id == current.id and selected.result_ids == ["current", "current"]
            with pytest.raises(json.JSONDecodeError):
                reader(scope=SCOPE, context=CONTEXT, limit=2)


def test_storage_failure_is_not_empty_success():
    with MemoryStore(":memory:") as store:
        store.record_recall_event(query="query", scope=SCOPE, ambient_context=CONTEXT)

        def deny(action, table, column, db, trigger):
            return (sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ and table == "recall_events"
                    else sqlite3.SQLITE_OK)

        store.connection.set_authorizer(deny)
        try:
            with pytest.raises(sqlite3.DatabaseError):
                queries(store)
        finally:
            store.connection.set_authorizer(None)


@pytest.mark.parametrize("explicit", [False, True])
def test_actual_remember_matches_full_preflight_without_materializing_bodies(tmp_path, monkeypatch, explicit):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", "grounded")
    monkeypatch.setattr(server, "_transport_session_id", lambda: "pending-inputs-transport")
    monkeypatch.setattr(storage, "_utc_now", lambda: "2026-01-01T00:00:00Z")
    monkeypatch.setattr(feedback, "_anchor_embedder", lambda store: lambda texts: [[1.0, 0.0] for _ in texts])
    original_queries, original_decode = feedback._feedback_queries, storage._json_loads
    snapshots, query_outputs, full_outcomes = [], [], []

    for projected in (False, True):
        ids = itertools.count()
        monkeypatch.setattr(storage, "new_ulid", lambda: f"test-{next(ids):04d}")
        mcp = server.create_mcp_server(tmp_path / f"{projected}.sqlite3", mcp_factory=FakeMCP)
        store = mcp.memory_store
        try:
            source = store.append_trace(CONTENT, CONTEXT)
            other = store.append_trace("Independent amber observation.", CONTEXT)
            events = [store.record_recall_event(
                query=f"query {i}", scope=SCOPE, ambient_context={**CONTEXT, "body": AMBIENT},
                results=[{"node_id": node_id, "bm25_score": 0.7, "vector_score": 0.3,
                          "delivery": {"kind": "full", "content": BODY}}
                         for node_id in [source.id, "missing", other.id, source.id, other.id, source.id]],
            ) for i in range(5)]
            store.insert_query_anchor(scope=SCOPE, query=events[0].query,
                                      fingerprint=recall_fingerprint(events[0].query, SCOPE),
                                      embedding=[1.0, 0.0])
            if not projected:
                monkeypatch.setattr(store, "pending_recall_query_inputs", store.pending_recall_events)
            observed_queries = []
            original_rows = store._pending_recall_rows

            def preflight(*args, **kwargs):
                def decode(value, default):
                    if projected:
                        assert BODY not in str(value) and AMBIENT not in str(value)
                    return original_decode(value, default)

                def rows(**kwargs):
                    result = original_rows(**kwargs)
                    for row in result:
                        assert set(row.keys()) == {"id", "query", "scope", "results"}
                        assert BODY not in str(tuple(row)) and AMBIENT not in str(tuple(row))
                    return result

                with monkeypatch.context() as check:
                    check.setattr(storage, "_json_loads", decode)
                    if projected:
                        def no_full_event(row):
                            pytest.fail("preflight materialized a full RecallEvent")
                        check.setattr(storage, "_recall_event_from_row", no_full_event)
                        check.setattr(store, "_pending_recall_rows", rows)
                    result = original_queries(*args, **kwargs)
                    observed_queries.append(result)
                    return result

            monkeypatch.setattr(feedback, "_feedback_queries", preflight)
            original_apply = feedback.apply_pending_recall_feedback

            def apply(*args, **kwargs):
                result = original_apply(*args, **kwargs)
                assert all(type(event) is RecallEvent for event in result.events)
                assert all(event.ambient_context["body"] == AMBIENT for event in result.events)
                assert all(event.results[0]["delivery"]["content"] == BODY for event in result.events)
                full_outcomes.append(asdict(result))
                return result

            monkeypatch.setattr(server, "apply_pending_recall_feedback", apply)
            result = mcp.tools["memory_remember"](CONTENT, CONTEXT, used=[other.id] if explicit else None)
            assert result["implicit_feedback"]["recall_event_ids"] == [event.id for event in reversed(events)]
            assert store.get_node(source.id).usefulness_score > 0
            if explicit:
                assert result["feedback_marks"] == {"accepted": 1, "dropped": 0, "by_reason": {}}
                assert other.id in store.credited_node_ids(events[-1].id)
            assert len(store.list_query_anchors()) == 5
            assert all(store.get_recall_event(event.id).feedback_applied for event in events)
            query_outputs.append(observed_queries)
            snapshots.append({table: [tuple(row) for row in store.connection.execute(f"SELECT * FROM {table}")]
                              for table in ("nodes", "recall_events", "recall_credit_ledger", "retrieval_weights",
                                            "query_anchors", "query_anchor_edges", "recall_feedback_marks")})
        finally:
            mcp.auto_consolidation.close()
            store.close()
    assert query_outputs[0] == query_outputs[1]
    assert full_outcomes[0] == full_outcomes[1]
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize("change", ["inputs", "identity", "closed", "malformed"])
def test_pending_inputs_are_reread_after_unlocked_encoding(runtime, change):
    r = runtime
    with r.guard:
        other = r.store.create_node(level="concept", content=CONTENT + " Verified.",
                                    context=CONTEXT, embedding=[1.0, 0.0, 0.0])
        event = r.store.record_recall_event(query="first query", scope=SCOPE, ambient_context=CONTEXT,
                                            results=[{"node_id": r.node.id}])

    async def scenario():
        async with Client(r.mcp) as client:
            r.feedback_hold = True
            pending = asyncio.create_task(call(client, "memory_remember", content=CONTENT, context=CONTEXT))
            try:
                assert await asyncio.to_thread(r.entered.wait, 4)
                with MemoryStore(r.store.db_path) as writer, writer.transaction():
                    if change == "inputs":
                        writer.connection.execute("UPDATE recall_events SET query = ?, results = ? WHERE id = ?",
                                                  ("current query", json.dumps([{"node_id": other.id}]), event.id))
                    elif change == "identity":
                        writer.connection.execute("UPDATE recall_events SET task = 'foreign' WHERE id = ?", (event.id,))
                    elif change == "closed":
                        writer.connection.execute("UPDATE recall_events SET feedback_applied = 1 WHERE id = ?", (event.id,))
                    else:
                        writer.connection.execute("UPDATE recall_events SET results = '{' WHERE id = ?", (event.id,))
            finally:
                r.release.set()
            if change == "malformed":
                with pytest.raises(ToolError, match="Expecting property name"):
                    await asyncio.wait_for(pending, 5)
            else:
                response = await asyncio.wait_for(pending, 5)
                assert response["implicit_feedback"]["recall_event_ids"] == ([event.id] if change == "inputs" else [])
            with r.guard:
                assert r.store.credited_node_ids(event.id) == ({other.id} if change == "inputs" else set())
                anchors = r.store.list_query_anchors()
                assert [anchor.query for anchor in anchors] == (["current query"] if change == "inputs" else [])
                assert r.store.get_node(r.node.id).usefulness_score == 0
                assert r.store.connection.execute("SELECT count(*) FROM nodes").fetchone()[0] == (
                    2 if change == "malformed" else 3
                )
                assert not r.store.connection.in_transaction
            assert r.batches == [("first query",)] + ([("current query",)] if change == "inputs" else [])

    asyncio.run(scenario())


@pytest.mark.parametrize("field,value,error", [
    ("ambient_context", "{", json.JSONDecodeError),
    ("resolved_scopes", "{", json.JSONDecodeError),
    ("resolved_scopes", "null", TypeError),
    ("max_results", "invalid", ValueError),
])
@pytest.mark.parametrize("route", ["remember", "teach"])
def test_full_metadata_errors_remain_required_and_roll_back_native_write(runtime, field, value, error, route):
    r = runtime
    with r.guard:
        target = r.store.append_trace("Correction target.", CONTEXT)
        event = r.store.record_recall_event(query="query", scope=SCOPE, ambient_context=CONTEXT,
                                            results=[{"node_id": r.node.id}])
        with r.store.transaction():
            r.store.connection.execute(f"UPDATE recall_events SET {field} = ? WHERE id = ?", (value, event.id))
        # Unneeded metadata is checked by the unchanged authoritative reader.
        assert queries(r.store)
        with pytest.raises(error) as failure:
            r.store.pending_recall_events(scope=SCOPE, context=CONTEXT)
        before = list(r.store.connection.iterdump())

    async def scenario():
        async with Client(r.mcp) as client:
            with pytest.raises(ToolError, match=re.escape(str(failure.value))):
                if route == "remember":
                    await call(client, "memory_remember", content=CONTENT, context=CONTEXT)
                else:
                    await call(client, "memory_teach", trace_id=target.id, correction=CONTENT, context=CONTEXT)
    asyncio.run(scenario())
    with r.guard:
        assert list(r.store.connection.iterdump()) == before
        assert not r.store.connection.in_transaction
