from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from living_memory.resources import retrieval_skew_metrics
from living_memory.storage import MemoryStore


@pytest.fixture(params=["project:graph-evidence", "global"])
def scoped_store(tmp_path: Path, request):
    scope = request.param
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        # A prefix match and another scope family must not supply evidence.
        unrelated_scope = "project:global" if scope == "global" else scope + ":child"
        a = store.append_trace("unrelated source", {"scope": unrelated_scope})
        b = store.append_trace("unrelated target", {"scope": "session:other"})
        store.create_connection(a.id, b.id, "related")
        yield store, scope


@pytest.mark.parametrize("incoming", [False, True])
@pytest.mark.parametrize("relation,metadata,expected", [
    ("related", {}, True),
    ("related", {"kind": "unknown", "basis": "unknown"}, True),
    ("related", {"basis": "implicit_recall_feedback"}, False),
    ("related", {"basis": "implicit_recall_feedback", "kind": "content_reference"}, True),
    ("related", {"basis": "implicit_recall_feedback", "kind": "derived_from"}, True),
    ("requires", {"basis": "implicit_recall_feedback"}, True),
    ("supersedes", {}, True),
    ("supersedes", {"kind": "duplicate_content"}, False),
    ("supersedes", {"kind": "content_correction"}, False),
    ("supersedes", {"rule": "R1c"}, False),
    ("supersedes", {"kind": "duplicate_content", "correction": "verified"}, True),
    ("caused", {}, True),
    ("caused", {"rule": "R2a"}, False),
    ("caused", {"kind": "failure_resolution"}, False),
    ("caused", {"rule": "R2a", "kind": "failure_resolution", "basis": "explicit"}, True),
])
def test_graph_evidence_uses_either_scoped_endpoint(
    scoped_store, incoming: bool, relation: str, metadata: dict, expected: bool,
) -> None:
    store, scope = scoped_store
    selected = store.append_trace("selected endpoint", {"scope": scope})
    other = store.append_trace("cross-scope endpoint", {"scope": "session:counterpart"})
    source, target = (other, selected) if incoming else (selected, other)
    store.create_connection(source.id, target.id, relation, weight=0, metadata=metadata)

    assert store._has_graph_evidence(scope) is expected
    weights = store.set_retrieval_weights(scope, bm25=1, vector=0, graph=0)
    applicable, effective = store.evaluate_retrieval_weight_floors(scope, weights)
    assert applicable.graph_min == (0.05 if expected else 0)
    maximum = (0.75 if scope == "global" else 0.85) if expected else 1
    assert (effective.bm25, effective.vector, effective.graph) == pytest.approx(
        (maximum, 0, 1 - maximum),
    )


@pytest.mark.parametrize("incoming", [False, True])
@pytest.mark.parametrize("decayed", ["selected", "other", "both"])
def test_graph_evidence_requires_both_endpoints_active(scoped_store, incoming, decayed):
    store, scope = scoped_store
    selected = store.append_trace("selected active endpoint", {"scope": scope})
    other = store.append_trace("other active endpoint", {"scope": "session:counterpart"})
    source, target = (other, selected) if incoming else (selected, other)
    store.create_connection(source.id, target.id, "related")
    assert store._has_graph_evidence(scope)
    with store.transaction():
        for name, node in (("selected", selected), ("other", other)):
            if decayed in (name, "both"):
                store.connection.execute("UPDATE nodes SET decayed = 1 WHERE id = ?", (node.id,))
    assert not store._has_graph_evidence(scope)


def test_graph_evidence_needs_an_edge_in_the_exact_scope(scoped_store):
    store, scope = scoped_store
    assert not store._has_graph_evidence(scope)
    store.append_trace("isolated scoped node", {"scope": scope})
    assert not store._has_graph_evidence(scope)


def test_graph_evidence_accepts_an_active_self_loop(scoped_store):
    store, scope = scoped_store
    node = store.append_trace("self-loop endpoint", {"scope": scope})
    store.create_connection(node.id, node.id, "related")
    assert store._has_graph_evidence(scope)
    with store.transaction():
        store.connection.execute("UPDATE nodes SET decayed = 1 WHERE id = ?", (node.id,))
    assert not store._has_graph_evidence(scope)


@pytest.mark.parametrize("consumer", ["evaluate", "update", "diagnostic"])
def test_required_graph_read_failure_is_explicit(scoped_store, consumer):
    store, scope = scoped_store
    weights = store.set_retrieval_weights(scope, bm25=1, vector=0, graph=0)

    def deny_graph_read(action, table, column, database, trigger):
        if action == sqlite3.SQLITE_READ and table == "connections":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    store.connection.set_authorizer(deny_graph_read)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="prohibited"):
            if consumer == "evaluate":
                store.evaluate_retrieval_weight_floors(scope, weights)
            elif consumer == "update":
                store.update_retrieval_weights(scope, bm25_signal=1)
            else:
                retrieval_skew_metrics(store, scope=scope)
    finally:
        store.connection.set_authorizer(None)
    assert store.get_retrieval_weights(scope) == weights
