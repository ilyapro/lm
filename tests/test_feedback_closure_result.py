"""Closure consumers read saved attribution without decoding the event again."""

from dataclasses import replace
import sqlite3

import pytest

from living_memory import server, storage
from living_memory.health_audit import ReadOnlyAuditStore
from living_memory.storage import MemoryStore
from test_lookup import FakeMCP


CONTEXT = {"scope": "project:closure-result", "session_id": "closure-session", "agent": "fixture"}
CONTENT = "Cobalt certificate renewal requires preserving the trusted issuer chain."


@pytest.fixture
def mcp(tmp_path, monkeypatch):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
        "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic",
        "LM_AUTO_CONSOLIDATE_POLICY": "off",
        "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
        "LM_RETRIEVAL_TUNING_POLICY": "fixed",
        "LM_RECALL_CREDIT_POLICY": "grounded",
        "LM_EXPLICIT_FEEDBACK_POLICY": "credit",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(server, "_transport_session_id", lambda: "fixture-transport")
    instance = server.create_mcp_server(
        tmp_path / "memory.sqlite3", mcp_factory=FakeMCP, expose_attest=True,
    )
    try:
        yield instance
    finally:
        instance.auto_consolidation.close()
        instance.memory_store.close()


def seed(mcp):
    store = mcp.memory_store
    source = store.append_trace(CONTENT, CONTEXT)
    event = store.record_recall_event(
        query="cobalt certificate renewal", scope=CONTEXT["scope"],
        ambient_context={**CONTEXT, "context_data": "z" * 131072},
        results=[{"node_id": source.id, "bm25_score": 1.0,
                  "delivery": {"kind": "full"}, "extra": {"preserved": [1, 2, 3]}}],
    )
    return source, event


def consume(mcp, route, source, event):
    if route == "remember":
        result = mcp.tools["memory_remember"](CONTENT + " Verified.", CONTEXT, used=[source.id])
        return result["node"]["id"], result
    if route == "teach":
        result = mcp.tools["memory_teach"](
            source.id, CONTENT + " Verified.", context=CONTEXT, used=[source.id],
        )
        return result["corrective_trace"]["id"], result
    result = mcp.tools["memory_attest"](event.id, [CONTENT], trace_id=source.id)
    return source.id, result


@pytest.mark.parametrize("route", ["remember", "teach", "attest"])
def test_actual_consumers_preserve_payload_without_full_closure_readback(mcp, monkeypatch, route):
    store = mcp.memory_store
    source, event = seed(mcp)
    original_close = store.mark_recall_event_feedback
    original_decode = storage._recall_event_from_row
    closing = False
    closure_calls, decoded, closure_sql = [], [], []

    def decode(row):
        decoded.append((closing, row["id"], len(row["ambient_context"])))
        return original_decode(row)

    def close(*args, **kwargs):
        nonlocal closing
        closure_calls.append((args, kwargs))
        closing = True
        try:
            return original_close(*args, **kwargs)
        finally:
            closing = False

    # Keep the ordinary writer/override dispatch, and observe only closure's
    # work. Full authoritative reads before closure are still required.
    with monkeypatch.context() as observe:
        observe.setattr(store, "mark_recall_event_feedback", close)
        observe.setattr(storage, "_recall_event_from_row", decode)
        store.connection.set_trace_callback(lambda sql: closure_sql.append(sql) if closing else None)
        try:
            trace_id, result = consume(mcp, route, source, event)
        finally:
            store.connection.set_trace_callback(None)
    assert closure_calls == [((event.id, trace_id), {"trace_id_only": True})]
    assert decoded and all(not inside and identity == event.id and size > 131072
                           for inside, identity, size in decoded)
    assert not any("SELECT * FROM recall_events" in sql for sql in closure_sql)
    closed = store.get_recall_event(event.id)
    assert closed.feedback_applied_at
    assert closed == replace(event, feedback_applied=True, feedback_trace_id=trace_id,
                             feedback_applied_at=closed.feedback_applied_at)
    assert not store.connection.in_transaction
    if route == "attest":
        assert result["feedback_trace_id"] == trace_id
        assert result["closed"] and result["closed_by_attestation"] and result["credited"]
        ledger, = store.list_recall_attestations(recall_event_id=event.id)
        assert ledger.feedback_trace_id == trace_id
        committed = list(store.connection.iterdump())
        assert consume(mcp, route, source, event)[1] == dict(result, replay=True)
        assert list(store.connection.iterdump()) == committed
    else:
        trace = store.get_node(trace_id)
        assert trace.provenance["prior_recalls"][0]["id"] == event.id
        assert trace.provenance["recalled_nodes"] == [source.id]
        assert trace.source_traces == [source.id]
        assert result["feedback_marks"]["accepted"] == 1
        assert store.credited_node_ids(event.id) == {source.id}


def test_attestation_uses_saved_attribution_in_completion_and_replay(mcp):
    store = mcp.memory_store
    source, event = seed(mcp)
    saved = store.append_trace("Persisted closure attribution.", CONTEXT, node_id="saved-attribution")
    # A synthetic storage transform distinguishes readback from an input echo.
    store.connection.execute("""CREATE TRIGGER stored_attribution
        AFTER UPDATE OF feedback_applied_at ON recall_events
        BEGIN UPDATE recall_events SET feedback_trace_id = 'saved-attribution'
            WHERE id = NEW.id; END""")
    _, result = consume(mcp, "attest", source, event)
    assert result["feedback_trace_id"] == saved.id != source.id
    assert store.get_recall_event(event.id).feedback_trace_id == saved.id
    ledger, = store.list_recall_attestations(recall_event_id=event.id)
    assert ledger.feedback_trace_id == saved.id
    assert consume(mcp, "attest", source, event)[1] == dict(result, replay=True)


@pytest.mark.parametrize("trace_id_only", [False, True])
def test_closure_result_and_timestamp_keep_sqlite_affinity(monkeypatch, trace_id_only):
    with MemoryStore(":memory:") as store:
        trace = store.append_trace("Consumer", node_id="1")
        event = store.record_recall_event(query="query", scope=CONTEXT["scope"])
        monkeypatch.setattr(storage, "_utc_now", lambda: 1234567890)
        result = store.mark_recall_event_feedback(event.id, 1, trace_id_only=trace_id_only)
        assert (result if trace_id_only else result.feedback_trace_id) == trace.id == "1"
        closed = store.get_recall_event(event.id)
        assert closed.feedback_applied_at == "1234567890"
        if not trace_id_only:
            assert result == closed


@pytest.mark.parametrize("trace_id_only", [False, True])
def test_readonly_override_rejects_closure_on_writable_borrowed_connection(trace_id_only):
    with MemoryStore(":memory:") as store:
        trace = store.append_trace("Consumer")
        event = store.record_recall_event(query="query", scope=CONTEXT["scope"])
        before = list(store.connection.iterdump())
        with pytest.raises(RuntimeError, match="read-only"):
            ReadOnlyAuditStore.from_store(store).mark_recall_event_feedback(
                event.id, trace.id, trace_id_only=trace_id_only,
            )
        assert list(store.connection.iterdump()) == before


@pytest.mark.parametrize("route", ["remember", "teach", "attest"])
def test_required_projection_failure_rolls_back_actual_consumer(mcp, monkeypatch, route):
    store = mcp.memory_store
    source, event = seed(mcp)
    before = list(store.connection.iterdump())
    original = store.mark_recall_event_feedback

    def close(*args, **kwargs):
        def authorize(action, table, column, *_):
            if action == sqlite3.SQLITE_READ and table == "recall_events" and column == "feedback_trace_id":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        store.connection.set_authorizer(authorize)
        try:
            return original(*args, **kwargs)
        finally:
            store.connection.set_authorizer(None)

    with monkeypatch.context() as fault:
        fault.setattr(store, "mark_recall_event_feedback", close)
        with pytest.raises(sqlite3.DatabaseError, match="recall_events.feedback_trace_id"):
            consume(mcp, route, source, event)
    assert not store.connection.in_transaction
    assert list(store.connection.iterdump()) == before
    trace_id, _ = consume(mcp, route, source, event)
    assert store.get_recall_event(event.id).feedback_trace_id == trace_id
