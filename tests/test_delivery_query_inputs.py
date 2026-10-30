"""Explicit/lookup preparation projects IDs; authoritative consumers retain results."""

import itertools
import json
import sqlite3

import pytest

from living_memory import consolidation, feedback, irrelevance, query_anchors, server, storage
from living_memory.storage import MemoryStore
from test_lookup import FakeMCP


NOW = "2026-10-30T12:00:00Z"
SCOPE = "project:delivery-inputs"
CONTEXT = {"scope": SCOPE, "session_id": "session", "task": "task", "agent": "agent"}
BODY = "unused-delivery-" + "x" * 17600


def event(store, *, raw=None, query="query", transport="transport", context=None,
          at=NOW, ids=("a",)):
    item = store.record_recall_event(
        query=query, scope=SCOPE,
        ambient_context={**(context or CONTEXT), "transport_session_id": transport},
        results=[{"node_id": node_id, "bm25_score": .7, "vector_score": .3,
                  "graph_score": .2, "delivery": {"content": BODY}} for node_id in ids],
    )
    with store.transaction():
        store.connection.execute("UPDATE recall_events SET created_at = ? WHERE id = ?", (at, item.id))
        if raw is not None:
            store.connection.execute("UPDATE recall_events SET results = ? WHERE id = ?", (raw, item.id))
    return item


def queries(store, **kwargs):
    return feedback._feedback_queries(store, **{
        "used": ["a"], "irrelevant": None, "lookup_ids": None,
        "transport_session_id": "transport", "occurred_at": NOW,
        "trace_content": None, "trace_context": None, "context": CONTEXT, **kwargs,
    })


def compare_queries(store, monkeypatch, **kwargs):
    reader = store.recall_deliveries
    before = store.connection.total_changes
    outcomes = []
    for projected in (False, True):
        with monkeypatch.context() as patch:
            if not projected:
                def full(*args, **options):
                    return reader(*args, **{**options, "query_inputs_only": False})
                patch.setattr(store, "recall_deliveries", full)
            try:
                outcomes.append(queries(store, **kwargs))
            except (sqlite3.Error, ValueError, TypeError) as error:
                outcomes.append((type(error), str(error)))
        assert not store.connection.in_transaction
    assert outcomes[0] == outcomes[1]
    assert before == store.connection.total_changes
    return outcomes[1]


@pytest.fixture(autouse=True)
def policies(monkeypatch):
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    monkeypatch.setenv("LM_LOOKUP_CREDIT_POLICY", "delivered")


@pytest.mark.parametrize("raw,requested,expected", [
    ('[{"node_id":"a","node_id":"b"}]', ["a", "b"], {"b"}),
    ('[{"node_id":"a","node_id":"b"}]', ["b"], set()),
    ('[{"node_id":"a","node_id":"b"}]', ["a"], set()),
    ('[{"node_id":"a","node_id":42}]', ["a", "42"], {"42"}),
    ('[{"node_id":"a","node_id":false}]', ["a", "False"], set()),
    ('[{"node_id":"a","node_id":{"x":1,"x":2}}]', ["a", "{'x': 2}"], {"{'x': 2}"}),
    ('[{"node_id":"a","node_id":[1,false]}]', ["a", "[1, False]"], {"[1, False]"}),
    ('[{"\\u006eode_id":"a"}]', ["a"], {"a"}),
    ('[{"node_id":"a","\\u006eode_id":"b"}]', ["a", "b"], {"b"}),
    ('[{"node_id":"a","body":NaN}]', ["a"], {"a"}),
    ('[{"node_id":"a","body":1e999}]', ["a"], {"a"}),
    ('[null,"a",4,true,{},[1],{"node_id":"a"}]', ["a"], {"a"}),
    ('[[["node_id","a"]]]', ["a"], set()),
    ('[{"node_id":42}]', ["42"], set()),
    ('null', ["a"], set()), ('{}', ["a"], set()), ('"a"', ["a"], set()),
])
def test_raw_id_membership_and_hit_insertion_match_full(monkeypatch, raw, requested, expected):
    with MemoryStore(":memory:") as store:
        event(store, raw=raw)
        options = dict(earliest=feedback._lookup_instant("2026-10-29"), latest=feedback._lookup_instant(NOW))
        full = store.recall_deliveries("transport", requested, **options)
        narrow = store.recall_deliveries("transport", requested, query_inputs_only=True, **options)
        assert set(full) == set(narrow) == expected
        for node_id in expected:
            assert {k: v for k, v in full[node_id].items() if k != "result"} == {
                k: v for k, v in narrow[node_id].items() if k != "result"
            }
        assert bool(compare_queries(store, monkeypatch, used=requested)) == bool(expected)


@pytest.mark.parametrize("case", [
    "used", "negative", "lookup", "marks", "reconnect", "conflict", "lookup-reconnect",
    "no-transport", "lookup-credited", "used-credited", "skip-old", "need-old", "time-excluded",
])
def test_actual_preflight_matches_full_and_is_read_only(monkeypatch, case):
    with MemoryStore(":memory:") as store:
        options = {}
        if case in {"skip-old", "need-old", "time-excluded"}:
            event(store, raw="{", at="2026-10-28" if case == "time-excluded" else "2026-10-30T10:00:00Z")
        item = event(store, query="  query  ",
                     transport="old" if case in {"reconnect", "conflict", "lookup-reconnect"} else "transport",
                     context={**CONTEXT, "agent": "other"} if case == "conflict" else CONTEXT)
        if case in {"lookup", "lookup-reconnect", "lookup-credited"}:
            options["lookup_ids"] = ["a"]
        if case == "negative":
            options.update(used=None, irrelevant=["a"])
        if case == "marks":
            options.update(used=["a", "a", "b", None, ""], irrelevant=["b"])
        if case == "no-transport":
            options["transport_session_id"] = None
        if case.endswith("credited"):
            store.claim_recall_credit(item.id, "a", basis="lookup", source_id="lookup")
        if case == "need-old":
            options["used"] = ["a", "missing"]
        result = compare_queries(store, monkeypatch, **options)
        if case == "need-old":
            assert result == (sqlite3.OperationalError, "malformed JSON")
        elif case in {"conflict", "lookup-reconnect", "lookup-credited", "no-transport"}:
            assert result == {}
        else:
            assert list(result.values()) == ["query" if case == "negative" else "  query  "]


@pytest.mark.parametrize("table,column", [
    ("recall_events", "results"), ("recall_events", "created_at"),
    ("recall_explicit_credit", "node_id"), ("recall_credit_ledger", "node_id"),
])
def test_preflight_storage_errors_match_full(monkeypatch, table, column):
    with MemoryStore(":memory:") as store:
        event(store)
        store.connection.set_authorizer(lambda action, tab, col, *_:
            sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ and (tab, col) == (table, column)
            else sqlite3.SQLITE_OK)
        try:
            error, message = compare_queries(store, monkeypatch)
            assert error is sqlite3.DatabaseError and "prohibited" in message
        finally:
            store.connection.set_authorizer(None)


@pytest.mark.parametrize("credited_id", ["a", "b"])
def test_duplicate_id_credit_stays_bound_to_sql_membership(monkeypatch, credited_id):
    with MemoryStore(":memory:") as store:
        item = event(store, raw='[{"node_id":"a","node_id":"b"}]')
        store.claim_recall_credit(item.id, credited_id, basis="lookup", source_id="lookup")
        result = compare_queries(store, monkeypatch, lookup_ids=["a", "b"])
        assert bool(result) == (credited_id != "a")


@pytest.mark.parametrize("fail_decode", [False, True])
def test_narrow_rows_exclude_bodies_and_close_cursor_on_decode_failure(monkeypatch, fail_decode):
    with MemoryStore(":memory:") as store:
        event(store)
        original_conn = store.connection
        cursors = []

        class Rows:
            closed = False

            def __init__(self, cursor):
                self.cursor = cursor

            def __iter__(self):
                for row in self.cursor:
                    assert BODY not in str(tuple(row))
                    assert json.loads(row["result"]) == {"node_id": "a"}
                    yield row

            def close(self):
                self.closed = True
                self.cursor.close()

        class Connection:
            def __getattr__(self, name):
                return getattr(original_conn, name)

            def execute(self, sql, *args):
                cursor = original_conn.execute(sql, *args)
                if "AS result," in sql:
                    cursor = Rows(cursor)
                    cursors.append(cursor)
                return cursor

        with monkeypatch.context() as patch:
            patch.setattr(store, "_conn", Connection())
            if fail_decode:
                def decode(*args):
                    raise ValueError("decode failed")
                patch.setattr(storage, "_json_loads", decode)
                with pytest.raises(ValueError, match="decode failed"):
                    queries(store)
            else:
                assert queries(store)
        assert len(cursors) == 1 and cursors[0].closed
        assert not store.connection.in_transaction


@pytest.mark.parametrize("route", ["lookup", "recall", "remember", "teach"])
def test_consumers_match_full_preflight_outputs_feedback_and_tables(tmp_path, monkeypatch, route):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash", "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic",
        "LM_AUTO_CONSOLIDATE_POLICY": "off", "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
        "LM_RECALL_CREDIT_POLICY": "grounded",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(server, "_transport_session_id", lambda: "transport")
    monkeypatch.setattr(storage, "_utc_now", lambda: NOW)
    monkeypatch.setattr(consolidation, "_utc_now", lambda: NOW)
    monkeypatch.setattr(query_anchors, "_timestamp", lambda value: NOW)
    monkeypatch.setattr(irrelevance, "_now", lambda: NOW)
    monkeypatch.setattr(feedback, "_anchor_embedder", lambda store: lambda texts: [[1.0, 0.0] for _ in texts])
    original_queries, original_decode = feedback._feedback_queries, storage._json_loads
    outcomes = []
    for projected in (False, True):
        ids = itertools.count()
        monkeypatch.setattr(storage, "new_ulid", lambda: f"test-{next(ids):04d}")
        monkeypatch.setattr(consolidation, "new_ulid", storage.new_ulid)
        mcp = server.create_mcp_server(tmp_path / f"{projected}.sqlite3", mcp_factory=FakeMCP)
        store = mcp.memory_store
        try:
            used = store.append_trace("Amber source observation.", CONTEXT)
            negative = store.append_trace("Independent violet observation.", CONTEXT)
            target = store.append_trace("Separate correction target.", CONTEXT)
            item = event(store, ids=[used.id, negative.id])
            with store.transaction():
                store.connection.execute("UPDATE recall_events SET feedback_applied = 1")
            query_outputs, decoded = [], []
            reader = store.recall_deliveries

            def preflight(*args, **kwargs):
                with monkeypatch.context() as patch:
                    if projected:
                        def decode(value, default):
                            assert BODY not in str(value)
                            parsed = original_decode(value, default)
                            if isinstance(parsed, dict) and "node_id" in parsed:
                                decoded.append(parsed)
                                assert set(parsed) == {"node_id"}
                            return parsed
                        patch.setattr(storage, "_json_loads", decode)
                    else:
                        def full(*args, **options):
                            return reader(*args, **{**options, "query_inputs_only": False})
                        patch.setattr(store, "recall_deliveries", full)
                    result = original_queries(*args, **kwargs)
                    query_outputs.append(result)
                    return result

            monkeypatch.setattr(feedback, "_feedback_queries", preflight)
            responses = []
            for _ in range(2):
                fields = dict(used=[used.id], irrelevant=[negative.id])
                if route == "lookup":
                    response = mcp.tools["memory_lookup"](node_ids=[used.id])
                elif route == "recall":
                    response = mcp.tools["memory_recall"](query="", scope=SCOPE, **fields)
                elif route == "remember":
                    response = mcp.tools["memory_remember"]("New independent observation.", CONTEXT, **fields)
                else:
                    response = mcp.tools["memory_teach"](target.id, "Revised independent observation.", context=CONTEXT, **fields)
                responses.append(response)
            assert query_outputs and (not projected or decoded)
            assert used.id in store.credited_node_ids(item.id)
            assert store.get_node(used.id).usefulness_score > 0
            # The public default and accepted ExplicitMark.result stay full,
            # including scores needed by the authoritative feedback consumers.
            marks = feedback.resolve_explicit_marks(store, [used.id], None, "transport", NOW)
            assert marks[0].accepted and marks[0].result == item.results[0]
            full = reader("transport", [used.id], earliest=feedback._lookup_instant("2026-10-29"),
                          latest=feedback._lookup_instant(NOW))
            assert full[used.id]["result"] == item.results[0]
            tables = [row[0] for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )]
            snapshot = {table: [tuple(row) for row in store.connection.execute(f'SELECT * FROM "{table}"')]
                        for table in tables}
            outcomes.append((responses, query_outputs, snapshot))
        finally:
            mcp.auto_consolidation.close()
            store.close()
    assert outcomes[0] == outcomes[1]
