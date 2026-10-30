"""Anchor consumers validate fresh endpoints without discarded node hydration."""

import sqlite3

import pytest

from living_memory import feedback, query_anchors as qa, storage
from living_memory.storage import MemoryStore, recall_fingerprint


SCOPE = "project:anchors"
QUERY = "current endpoint"
STAMP = "2026-01-01T00:00:00Z"


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as opened:
        yield opened


def prepare(store, case):
    target = " target " if case == "padded" else "target"
    for node_id in ("source", target, "replacement"):
        content = "exact copy" if case == "exact_copy" and node_id != "source" else node_id
        store.create_node(
            node_id=node_id, level="trace", content=content,
            context={"scope": SCOPE, "task": "anchor validity"},
            provenance={"origin": "fixture"},
        )
    if case == "decayed":
        store.soft_delete_node(target, "fixture")
    if case in ("correction", "cycle"):
        store.create_connection("replacement", target, "supersedes")
    if case == "cycle":
        store.create_connection(target, "replacement", "supersedes")
    if case == "backcycle":
        store.create_connection("source", target, "supersedes")
    if case == "exact_copy":
        store.create_connection(
            "replacement", target, "related", metadata={"kind": "duplicate_content"},
        )
    store.insert_query_anchor(
        scope=SCOPE, query=QUERY, fingerprint=recall_fingerprint(QUERY, SCOPE),
        embedding=[1.0, 0.0], anchor_id="anchor", now=STAMP,
    )
    return "missing" if case == "missing" else target


def operation(store, consumer, target):
    if consumer == "liveness":
        return lambda: qa._live_target(store, target)
    if consumer == "migration":
        return lambda: qa.migrate_anchor_edges(store, "source", target, now=STAMP)
    event = store.record_recall_event(
        query=QUERY, scope=SCOPE, results=[{"node_id": target}],
    )
    return lambda: feedback._apply_query_feedback(store, [(event, (target,), ())])


def edges(store):
    return {
        edge.target_id: (edge.weight, edge.hits)
        for edge in store.list_query_anchor_edges(anchor_id="anchor")
    }


@pytest.mark.parametrize("consumer", ["liveness", "feedback", "migration"])
@pytest.mark.parametrize(
    "case,resolved,decodes",
    [
        ("live", "target", 0),
        ("padded", " target ", 0),
        ("decayed", None, 0),
        ("missing", None, 0),
        ("correction", "replacement", 1),
        ("exact_copy", "replacement", 2),
        ("cycle", "replacement", 2),
        ("backcycle", "source", 1),
    ],
)
def test_anchor_consumers_resolve_and_merge_without_final_node_decode(
    store, monkeypatch, consumer, case, resolved, decodes,
):
    target = prepare(store, case)
    if consumer == "migration":
        store.upsert_query_anchor_edge("anchor", "source", weight=0.5, now=STAMP)
        if case != "missing":
            store.upsert_query_anchor_edge("anchor", target, weight=0.25, now=STAMP)
    elif consumer == "feedback" and resolved is not None:
        for _ in range(2):
            store.upsert_query_anchor_edge("anchor", resolved, weight=0.25, now=STAMP)
    invoke = operation(store, consumer, target)
    before = edges(store)
    changes = store.connection.total_changes
    decoded = []
    decode = storage._node_from_row

    def record_decode(row):
        decoded.append(row["id"])
        return decode(row)

    monkeypatch.setattr(storage, "_node_from_row", record_decode)
    if consumer == "liveness":
        store.connection.execute("PRAGMA query_only=ON")
    result = invoke()
    assert len(decoded) == decodes
    anchor = store.get_query_anchor("anchor")
    if consumer == "liveness":
        assert result == resolved
        assert store.connection.total_changes == changes
        assert edges(store) == before
    elif consumer == "feedback":
        assert result == ["anchor"]
        assert anchor.reinforcement_count == 1
        assert edges(store) == ({} if resolved is None else {resolved: (0.75, 3)})
    elif case in ("decayed", "missing", "backcycle"):
        assert result == 0
        assert edges(store) == before
    else:
        assert result == 1
        assert edges(store) == {target: (0.75, 2)}
    if consumer != "feedback":
        assert anchor.reinforcement_count == 0


@pytest.mark.parametrize("consumer", ["liveness", "feedback", "migration"])
@pytest.mark.parametrize("column", ["id", "decayed"])
def test_final_endpoint_sql_failure_preserves_consumer_error_boundary(
    store, monkeypatch, caplog, consumer, column,
):
    target = prepare(store, "live")
    store.upsert_query_anchor_edge("anchor", "source", weight=0.5, now=STAMP)
    invoke = operation(store, consumer, target)
    read = store.get_node_endpoints
    denied = []

    def authorize(action, table, name, *_):
        if action == sqlite3.SQLITE_READ and table == "nodes" and name == column:
            denied.append(name)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def read_with_denied_column(node_ids):
        store.connection.set_authorizer(authorize)
        try:
            return read(node_ids)
        finally:
            store.connection.set_authorizer(None)

    monkeypatch.setattr(store, "get_node_endpoints", read_with_denied_column)
    if consumer == "feedback":
        assert invoke() == []
        assert "query anchor write failed" in caplog.text
        assert store.get_query_anchor("anchor").reinforcement_count == 1
    else:
        with pytest.raises(sqlite3.DatabaseError):
            invoke()
    assert denied == [column]
    assert edges(store) == {"source": (0.5, 1)}


@pytest.mark.parametrize("consumer", ["liveness", "feedback", "migration"])
def test_final_endpoint_reads_peer_decay_before_edge_write(
    store, tmp_path, monkeypatch, consumer,
):
    target = prepare(store, "correction" if consumer != "migration" else "live")
    final_target = "replacement" if consumer != "migration" else target
    assert qa._live_target(store, target) == final_target
    store.upsert_query_anchor_edge("anchor", "source", weight=0.5, now=STAMP)
    invoke = operation(store, consumer, target)
    read = store.get_node_endpoints
    observed = []

    def decay_then_read(node_ids):
        # The replacement walk has finished; a separate committed writer can
        # invalidate its candidate before the final liveness check.
        with sqlite3.connect(tmp_path / "memory.sqlite3") as peer:
            peer.execute("UPDATE nodes SET decayed = 1 WHERE id = ?", (final_target,))
        endpoints = read(node_ids)
        observed.append(endpoints[final_target].decayed)
        return endpoints

    monkeypatch.setattr(store, "get_node_endpoints", decay_then_read)
    result = invoke()
    assert observed == [True]
    assert result == {"liveness": None, "feedback": ["anchor"], "migration": 0}[consumer]
    assert edges(store) == {"source": (0.5, 1)}
