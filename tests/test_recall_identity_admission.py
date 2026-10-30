"""Source traversal reads only identities under the full reader's admission policy."""

import sqlite3

import pytest

from living_memory.scope import ScopePlan, resolve_scope
from living_memory.storage import MemoryStore
from test_recall_graph_preparation import node


@pytest.mark.parametrize("plan", [
    resolve_scope(scope="project:alpha"),
    resolve_scope(scope="session:review", ambient_context={"project": "alpha"}),
    resolve_scope(ambient_context={"project": "beta"}),
    ScopePlan("global", ()),
])
def test_identity_admission_matches_bodies_across_batches_and_fresh_changes(tmp_path, plan):
    path = tmp_path / "identities.sqlite3"
    with MemoryStore(path) as store, MemoryStore(path) as writer:
        for name, scope in (
            ("global", "global"), ("alpha", "project:alpha"),
            ("beta", "project:beta"), ("session", "session:review"),
        ):
            node(store, name, scope=scope)
        ids = [f"missing-{i}" for i in range(1100)] + ["global", "alpha", "beta", "session", "alpha"]

        def check():
            assert store.get_recall_node_ids(iter(ids), plan) == set(store.get_recall_nodes(ids, plan))

        check()
        assert store.get_recall_node_ids([], plan) == set()
        writer.update_node("alpha", context={"scope": "project:beta"})
        writer.soft_delete_node("global", "expired")
        writer.delete_node("session")
        check()
        with pytest.raises(RuntimeError, match="rollback"):
            with store.transaction():
                store.update_node("alpha", context={"scope": "project:alpha"})
                store.soft_delete_node("beta", "expired")
                check()
                raise RuntimeError("rollback")
        check()


def test_identity_projection_does_not_read_body_columns(tmp_path):
    with MemoryStore(tmp_path / "projection.sqlite3") as store:
        node(store, "source", scope="project:alpha")
        columns = set()

        def authorize(action, table, column, database, trigger):
            if action == sqlite3.SQLITE_READ and table == "nodes":
                columns.add(column)
                if column not in {"id", "scope", "decayed"}:
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        store.connection.set_authorizer(authorize)
        try:
            plan = resolve_scope(scope="project:alpha")
            assert store.get_recall_node_ids(["source"], plan) == {"source"}
            assert columns == {"id", "scope", "decayed"}
            with pytest.raises(sqlite3.DatabaseError, match="prohibited"):
                store.get_recall_nodes(["source"], plan)
        finally:
            store.connection.set_authorizer(None)


@pytest.mark.parametrize("body", ["content", "context", "provenance", "access_count"])
def test_unused_corrupt_body_is_not_decoded_but_required_body_stays_strict(tmp_path, body):
    with MemoryStore(tmp_path / "corrupt.sqlite3") as store:
        node(store, "source", scope="project:alpha")
        invalid_text = "x'22ff22'" if body == "context" else "x'ff'"
        with store.transaction():
            store.connection.execute(f"UPDATE nodes SET {body} = CAST({invalid_text} AS TEXT) WHERE id = 'source'")
        plan = resolve_scope(scope="project:alpha")
        assert store.get_recall_node_ids(["source"], plan) == {"source"}
        with pytest.raises(sqlite3.OperationalError, match="Could not decode to UTF-8"):
            store.get_recall_nodes(["source"], plan)


@pytest.mark.parametrize("required", ["id", "scope", "decayed"])
def test_required_admission_read_errors_propagate(tmp_path, required):
    with MemoryStore(tmp_path / "denied.sqlite3") as store:
        node(store, "source", scope="project:alpha")

        def authorize(action, table, column, database, trigger):
            if action == sqlite3.SQLITE_READ and table == "nodes" and column == required:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        store.connection.set_authorizer(authorize)
        try:
            with pytest.raises(sqlite3.DatabaseError, match="prohibited"):
                store.get_recall_node_ids(["source"], resolve_scope(scope="project:alpha"))
        finally:
            store.connection.set_authorizer(None)
