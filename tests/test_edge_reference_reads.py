"""Reference derivation reads current endpoint facts through the ordinary writer."""

import asyncio
import sqlite3
import threading

from fastmcp import Client
import pytest

from living_memory import server, storage
from living_memory.edge_backfill import backfill_edges
from living_memory.edge_derivation import derive_edges_for_new_trace


MISSING = "01" + "Z" * 24
SELF = "01" + "Y" * 24


def edge_tuples(edges):
    return [(edge.source_id, edge.target_id, edge.type, edge.weight, edge.metadata) for edge in edges]


def expected_edges(source, targets):
    return [(source, target, "related", 0.9, {"rule": "R6", "kind": "content_reference"})
            for target in targets]


def test_endpoint_batches_read_current_decay_and_obey_rollback(tmp_path):
    with storage.MemoryStore(tmp_path / "memory.sqlite3") as store:
        active = store.append_trace("Current source")
        stale = store.append_trace("Retired source")
        store.soft_delete_node(stale.id)
        ids = [f"missing-{i}" for i in range(1200)] + [active.id, stale.id, active.id]
        assert store.get_node_endpoints(iter(ids)) == {
            active.id: storage.NodeEndpoint(active.id, False),
            stale.id: storage.NodeEndpoint(stale.id, True),
        }
        with pytest.raises(RuntimeError, match="rollback"):
            with store.transaction():
                store.soft_delete_node(active.id)
                assert store.get_node_endpoints([active.id])[active.id].decayed
                raise RuntimeError("rollback")
        assert not store.get_node_endpoints([active.id])[active.id].decayed
        assert store.get_node_endpoints([]) == {}


@pytest.mark.parametrize("column", ["id", "decayed", "content", "context", "provenance"])
def test_endpoint_read_failures_are_not_missing_nodes(tmp_path, column):
    with storage.MemoryStore(tmp_path / "memory.sqlite3") as store:
        target = store.append_trace("Source")
        note = store.append_trace(f"Reference {target.id}")

        def authorize(action, table, name, *_args):
            if action == sqlite3.SQLITE_READ and table == "nodes" and name == column:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        store.connection.set_authorizer(authorize)
        try:
            if column in {"id", "decayed"}:
                with pytest.raises(sqlite3.DatabaseError, match="prohibited"):
                    derive_edges_for_new_trace(store, note)
            else:
                assert edge_tuples(derive_edges_for_new_trace(store, note)) == expected_edges(
                    note.id, [target.id],
                )
        finally:
            store.connection.set_authorizer(None)


def test_reference_hook_reads_only_endpoint_columns_and_matches_backfill(tmp_path, monkeypatch):
    with storage.MemoryStore(tmp_path / "memory.sqlite3") as store:
        targets = [store.create_node(
            level=("trace", "concept", "schema")[i % 3],
            content=f"Source {i}: " + "b" * 131072,
            context={"procedure": "p" * 131072},
            provenance={"detail": "h" * 65536},
        ).id for i in range(20)]
        stale = store.append_trace("Retired source")
        store.soft_delete_node(stale.id)
        # An undecayed superseded endpoint remains a valid historical reference.
        store.create_connection(targets[0], targets[1], "supersedes")
        note = store.create_node(
            node_id=SELF, level="trace",
            content="Reference " + " ".join([*targets, stale.id, targets[0], MISSING, SELF]),
        )
        materialized, columns = [], set()
        original = storage._node_from_row

        def materialize(row):
            materialized.append(row["id"])
            return original(row)

        def authorize(action, table, column, *_args):
            if action == sqlite3.SQLITE_READ and table == "nodes":
                columns.add(column)
                if column not in {"id", "decayed"}:
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        with monkeypatch.context() as capture:
            capture.setattr(storage, "_node_from_row", materialize)
            store.connection.set_authorizer(authorize)
            try:
                with pytest.raises(RuntimeError, match="rollback"):
                    with store.transaction():
                        assert edge_tuples(derive_edges_for_new_trace(store, note)) == expected_edges(
                            note.id, targets,
                        )
                        raise RuntimeError("rollback")
            finally:
                store.connection.set_authorizer(None)
        assert materialized == []
        assert columns == {"id", "decayed"}
        assert store.list_connections(source_id=note.id) == []
        backfill_edges(store, apply=True)
        assert sorted(edge_tuples(store.list_connections(source_id=note.id))) == sorted(
            (source, target, relation, weight, {**metadata, "basis": "provenance_derivation"})
            for source, target, relation, weight, metadata in expected_edges(note.id, targets)
        )


def test_reference_writer_failure_rolls_back_enclosing_transaction(tmp_path):
    with storage.MemoryStore(tmp_path / "memory.sqlite3") as store:
        first = store.append_trace("First source")
        second = store.append_trace("Second source")
        note = store.append_trace(f"Reference {first.id} {second.id}")
        store.connection.execute(f"""CREATE TRIGGER reject_edge BEFORE INSERT ON connections
            WHEN NEW.target_id = '{second.id}'
            BEGIN SELECT RAISE(ABORT, 'edge fault'); END""")
        with pytest.raises(sqlite3.IntegrityError, match="edge fault"):
            with store.transaction():
                derive_edges_for_new_trace(store, note)
        assert store.list_connections() == []
        assert store.get_node(note.id) == note


def test_reference_writer_still_validates_selected_endpoints(tmp_path, monkeypatch):
    with storage.MemoryStore(tmp_path / "memory.sqlite3") as store:
        target = store.append_trace("Source")
        note = store.append_trace(f"Reference {target.id}")
        read = store.get_node_endpoints

        def removed(ids):
            facts = read(ids)
            store.connection.execute("DELETE FROM nodes WHERE id = ?", (target.id,))
            return facts

        monkeypatch.setattr(store, "get_node_endpoints", removed)
        with pytest.raises(KeyError, match=target.id):
            with store.transaction():
                derive_edges_for_new_trace(store, note)
        assert store.get_node(target.id) == target
        assert store.list_connections() == []


@pytest.mark.parametrize("fault", [None, "read", "write"])
def test_native_remember_reads_endpoints_under_guard_after_required_commit(
    tmp_path, monkeypatch, fault,
):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    guard = threading.RLock()
    monkeypatch.setattr(server, "RLock", lambda: guard)
    mcp = server.create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    try:
        target = store.append_trace("Source " + "b" * 131072, {"procedure": "p" * 131072})
        stale = store.append_trace("Retired source")
        store.soft_delete_node(stale.id)
        read, materialize = store.get_node_endpoints, storage._node_from_row
        observations, materialized = [], []

        def observed(ids):
            observations.append((guard._is_owned(), store.connection.in_transaction))
            if fault == "read":
                store.connection.set_authorizer(lambda action, table, column, *_args:
                    sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ
                    and table == "nodes" and column == "decayed" else sqlite3.SQLITE_OK)
            try:
                return read(ids)
            finally:
                store.connection.set_authorizer(None)

        def counted(row):
            materialized.append(row["id"])
            return materialize(row)

        monkeypatch.setattr(store, "get_node_endpoints", observed)
        monkeypatch.setattr(storage, "_node_from_row", counted)
        if fault == "write":
            store.connection.execute("""CREATE TRIGGER reject_edge BEFORE INSERT ON connections
                BEGIN SELECT RAISE(ABORT, 'edge fault'); END""")

        async def scenario():
            async with Client(mcp) as client:
                result = await client.call_tool("memory_remember", {
                    "content": f"Reference {target.id} {stale.id} {target.id} {MISSING}",
                })
                assert not result.is_error
                return result.data["node"]["id"]

        node_id = asyncio.run(scenario())
        assert observations == [(True, False)]
        assert target.id not in materialized and stale.id not in materialized
        assert edge_tuples(store.list_connections(source_id=node_id)) == (
            expected_edges(node_id, [target.id]) if fault is None else []
        )
        assert store.get_node(node_id) is not None
        assert not store.connection.in_transaction
    finally:
        mcp.auto_consolidation.close()
        store.close()
