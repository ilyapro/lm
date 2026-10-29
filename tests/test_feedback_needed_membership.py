"""Delivery selection reads membership only while a requested fact is needed."""

from datetime import datetime, timezone
import json
import sqlite3

from fastmcp.exceptions import ToolError
import pytest

from living_memory import storage
from living_memory.feedback import resolve_explicit_marks
from living_memory.storage import MemoryStore
from test_recall_response_atomicity import CONTENT, CONTEXT, runtime, snapshot


NOW = "2026-10-29T12:00:00Z"
EARLIEST = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
LATEST = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)


def replace_results(store, event_id, results):
    with store.transaction():
        store.connection.execute("UPDATE recall_events SET results = ? WHERE id = ?",
                                 (results, event_id))


@pytest.mark.parametrize("corrupt", [False, True])
def test_current_recall_marks_use_healthy_newest_delivery(runtime, corrupt):
    store, node, _, call = runtime
    older = call()["recall_event_id"]
    if corrupt:
        replace_results(store, older, "{")
    newer = call()["recall_event_id"]
    response = call(used=[node.id])
    assert response["feedback_marks"] == {"accepted": 1, "dropped": 0, "by_reason": {}}
    mark, = store.connection.execute("SELECT * FROM recall_feedback_marks").fetchall()
    assert mark["recall_event_id"] == newer
    assert mark["source_id"] == response["recall_event_id"]
    credit, = store.connection.execute("SELECT * FROM recall_explicit_credit").fetchall()
    assert (credit["recall_event_id"], credit["node_id"]) == (newer, node.id)
    assert credit["source_id"] == f"mark:{mark['id']}"
    assert store.get_node(node.id).access_count == 3


def test_repeated_explicit_marks_claim_newest_delivery_once(runtime):
    store, node, _, call = runtime
    older = call()["recall_event_id"]
    replace_results(store, older, "{")
    newer = call()["recall_event_id"]
    # Close events without touching their bodies: implicit closure has a
    # separate contract; explicit feedback still accepts closed deliveries.
    with store.transaction():
        store.connection.execute("UPDATE recall_events SET feedback_applied = 1")
    responses = [call("memory_remember", content=CONTENT, context=CONTEXT, used=[node.id])
                 for _ in range(2)]
    assert all(response["feedback_marks"]["accepted"] == 1 for response in responses)
    marks = store.connection.execute("SELECT * FROM recall_feedback_marks ORDER BY id").fetchall()
    assert [mark["recall_event_id"] for mark in marks] == [newer, newer]
    assert [mark["source_id"] for mark in marks] == [response["node"]["id"] for response in responses]
    credit, = store.connection.execute("SELECT * FROM recall_explicit_credit").fetchall()
    assert (credit["recall_event_id"], credit["node_id"]) == (newer, node.id)
    assert credit["source_id"] == f"mark:{marks[0]['id']}"


def test_needed_malformed_delivery_refuses_without_credit_or_consumption(runtime):
    store, node, _, call = runtime
    event = call()["recall_event_id"]
    original = store.connection.execute("SELECT results FROM recall_events WHERE id = ?",
                                        (event,)).fetchone()[0]
    replace_results(store, event, "{")
    before = snapshot(store)
    with pytest.raises((sqlite3.DatabaseError, ToolError), match="malformed JSON"):
        call(used=[node.id])
    assert snapshot(store) == before
    replace_results(store, event, original)
    assert call(used=[node.id])["feedback_marks"]["accepted"] == 1


def test_no_current_membership_cannot_use_history_or_foreign_delivery(runtime):
    store, node, _, call = runtime
    event = call()["recall_event_id"]
    replace_results(store, event, "[]")
    delivery(store, [node.id], transport="foreign")
    response = call(used=[node.id])
    assert response["feedback_marks"] == {
        "accepted": 0, "dropped": 1, "by_reason": {"not_delivered": 1},
    }
    mark, = store.connection.execute("SELECT * FROM recall_feedback_marks").fetchall()
    assert not mark["accepted"] and mark["recall_event_id"] == ""
    assert mark["source_id"] == response["recall_event_id"]
    assert not store.connection.execute("SELECT * FROM recall_explicit_credit").fetchall()


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "selection.sqlite3") as store:
        yield store


def delivery(store, ids, *, at=NOW, transport="transport", context=None, scope="global"):
    event = store.record_recall_event(
        query="selection", scope=scope,
        ambient_context={**(context or {}), "transport_session_id": transport},
        results=[{"node_id": node_id} for node_id in ids],
    )
    with store.transaction():
        store.connection.execute("UPDATE recall_events SET created_at = ? WHERE id = ?",
                                 (at, event.id))
    return event.id


def select(store, ids, **kwargs):
    return store.recall_deliveries("transport", ids, earliest=EARLIEST, latest=LATEST, **kwargs)


def test_selection_stops_after_all_requested_ids_and_never_writes(store):
    older = delivery(store, ["a", "b"])
    replace_results(store, older, "{")
    middle = delivery(store, ["b"])
    newest = delivery(store, ["other", "a", "a"])
    before = snapshot(store)
    changes = store.connection.total_changes
    store.connection.execute("PRAGMA query_only = ON")
    try:
        hits = select(store, ["a", "b"])
    finally:
        store.connection.execute("PRAGMA query_only = OFF")
    assert {key: (hit["event_id"], hit["rank"]) for key, hit in hits.items()} == {
        "a": (newest, 1), "b": (middle, 0),
    }
    assert snapshot(store) == before
    assert store.connection.total_changes == changes
    # A still-unresolved id makes the old body necessary. Do not return
    # partial evidence as a successful resolution of the whole request.
    with pytest.raises(sqlite3.OperationalError, match="malformed JSON"):
        select(store, ["a", "missing"])


@pytest.mark.parametrize("at", ["2026-10-28T11:59:59Z", "9999-01-01", "invalid"])
def test_time_excludes_body_reads_but_still_spends_cap(store, at):
    healthy = delivery(store, ["a"])
    malformed = delivery(store, ["a"], at=at)
    replace_results(store, malformed, "{")
    assert select(store, ["a"])["a"]["event_id"] == healthy
    expected = {} if at > NOW else {"a"}
    assert set(select(store, ["a"], limit=1)) == set(expected)


def test_cap_excludes_old_malformed_body_before_membership(store):
    older = delivery(store, ["a"])
    replace_results(store, older, "{")
    delivery(store, ["other"])
    assert select(store, ["a"], limit=1) == {}
    with pytest.raises(sqlite3.OperationalError, match="malformed JSON"):
        select(store, ["a"], limit=2)


@pytest.mark.parametrize("field", ["session_id", "task", "agent", "scope"])
def test_incompatible_reconnect_does_not_read_foreign_body(store, field):
    context = {"session_id": "session", "task": "task", "agent": "agent", "scope": "project:one"}
    healthy = delivery(store, ["a"], context=context, scope=context["scope"])
    foreign_context = {**context, field: "project:foreign" if field == "scope" else "foreign"}
    foreign = delivery(store, ["a"], transport="reconnected",
                       context=foreign_context, scope=foreign_context["scope"])
    replace_results(store, foreign, "{")
    assert select(store, ["a"], context=context, limit=1)["a"]["event_id"] == healthy
    assert select(store, ["missing"], context=context) == {}


@pytest.mark.parametrize("body", [
    "null", '"a"', '{"0": {"node_id": "a"}}',
    '[null, "a", 1, true, [ {"node_id": "a"} ], {}, {"node_id": ["a"]}]',
])
def test_invalid_shapes_never_prove_delivery(store, body):
    event = delivery(store, ["a"])
    replace_results(store, event, body)
    assert select(store, ["a"]) == {}
    mark, = resolve_explicit_marks(store, ["a"], None, "transport", NOW)
    assert not mark.accepted and mark.reject_reason == "not_delivered"


def test_invalid_elements_do_not_mask_valid_result_position(store):
    event = delivery(store, ["a"])
    replace_results(store, event, json.dumps(["not JSON", None, {"node_id": "a"}]))
    hit = select(store, ["a"])["a"]
    assert (hit["event_id"], hit["rank"], hit["result"]) == (event, 2, {"node_id": "a"})


def test_unreadable_newest_cannot_attribute_to_older_healthy_event(store):
    delivery(store, ["a"])
    newest = delivery(store, ["a"])
    replace_results(store, newest, "{")
    with pytest.raises(sqlite3.OperationalError, match="malformed JSON"):
        select(store, ["a"])


def test_identity_and_membership_share_one_read_snapshot(store, monkeypatch):
    event = delivery(store, [])
    parse = storage.parse_timestamp
    changed = False

    def concurrent_change(value):
        nonlocal changed
        if not changed:
            changed = True
            # Neither committed state delivers a to this caller: before the
            # write membership is empty; after it the transport is foreign.
            with sqlite3.connect(store.db_path) as writer:
                writer.execute("""UPDATE recall_events SET transport_session_id = 'foreign',
                                  results = ? WHERE id = ?""",
                               (json.dumps([{"node_id": "a"}]), event))
        return parse(value)

    monkeypatch.setattr(storage, "parse_timestamp", concurrent_change)
    assert select(store, ["a"]) == {}
    assert changed
    assert select(store, ["a"]) == {}
    assert not store.connection.in_transaction


@pytest.mark.parametrize("table,column", [
    ("recall_events", "results"), ("recall_events", "created_at"),
    ("recall_explicit_credit", "node_id"), ("recall_credit_ledger", "node_id"),
])
def test_required_storage_errors_propagate(store, table, column):
    delivery(store, ["a"])

    def authorizer(action, actual_table, actual_column, *rest):
        if action == sqlite3.SQLITE_READ and (actual_table, actual_column) == (table, column):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    store.connection.set_authorizer(authorizer)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="prohibited"):
            select(store, ["a"])
    finally:
        store.connection.set_authorizer(None)
