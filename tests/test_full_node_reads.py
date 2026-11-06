"""Full node readers finish validation and release SQLite reads before writers."""

from contextlib import nullcontext
import json
import sqlite3

import pytest

from living_memory.storage import MemoryStore, _node_from_row


def read_nodes(store, method, ids):
    if method == "list":
        return store.list_nodes(include_decayed=True, limit=-1)
    return list(store.get_nodes(iter(ids)).values())


def assert_closed(cursors):
    assert cursors
    for cursor in cursors:
        with pytest.raises(sqlite3.ProgrammingError, match="closed cursor"):
            cursor.fetchone()


def capture_cursors(store):
    cursors = []

    def rows(cursor, values):
        if not cursors or cursor is not cursors[-1]:
            if cursors:
                assert_closed(cursors)
            cursors.append(cursor)
        return sqlite3.Row(cursor, values)

    store.connection.row_factory = rows
    return cursors


@pytest.mark.parametrize("method", ["list", "get"])
def test_full_fields_order_and_batch_closure(tmp_path, method):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        with store.transaction():
            nodes = [store.create_node(
                node_id=f"node-{i:04}", level=("trace", "concept", "schema")[i % 3],
                content=f"Harbor record {i}.",
                context={"scope": "project:harbor", "agent": "inspector", "task": "survey",
                         "nested": {"values": [None, True, 2**60, "проверка"]}},
                embedding=[0.25, -0.5], timestamp="2026-01-02T00:00:00Z",
                stats={"access_count": 3, "last_accessed": "2026-01-03T00:00:00Z",
                       "usefulness_score": 0.6, "confidence": 0.8, "unique_agents": 2,
                       "temporal_hint": "during survey"},
                provenance={"source_traces": ["source"], "source_concepts": ["concept"],
                            "corrections": [{"old": "before", "new": "after"}],
                            "custom": {"value": i}},
            ) for i in range(1003)]
        nodes[7] = store.soft_delete_node(nodes[7].id, "retired")
        by_id = {node.id: node for node in nodes}
        requested = [*reversed(by_id), nodes[0].id, "missing"]
        if method == "list":
            expected_ids = sorted(by_id, reverse=True)
        else:
            unique = list(set(requested))
            expected_ids = [node_id for start in range(0, len(unique), 999)
                            for node_id in sorted(unique[start:start + 999]) if node_id in by_id]
        before = store.connection.serialize()
        cursors = capture_cursors(store)
        try:
            result = read_nodes(store, method, requested)
        finally:
            store.connection.row_factory = sqlite3.Row
        assert result == [by_id[node_id] for node_id in expected_ids]
        assert_closed(cursors)
        assert store.connection.serialize() == before
        assert not store.connection.in_transaction
        assert store.get_nodes([]) == {}


@pytest.mark.parametrize("method", ["list", "get"])
@pytest.mark.parametrize("transaction", [False, True])
@pytest.mark.parametrize("failure", [None, "context", "provenance", "type", "utf8"])
def test_late_failure_is_not_partial_and_closes_cursor(tmp_path, method, transaction, failure):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        nodes = [store.append_trace(f"Harbor record {i}.") for i in range(5)]
        ids = [node.id for node in nodes]
        last = sorted(ids, reverse=method == "list")[-1]
        if failure:
            field = "provenance" if failure in {"provenance", "type"} else "context"
            value = "[]" if failure == "type" else "{malformed"
            with store.transaction():
                if field == "context":
                    # Model a legacy malformed row without the newer JSON
                    # expression indexes rejecting the fixture's insertion.
                    store.connection.execute("DROP INDEX idx_nodes_commit_prefix")
                    store.connection.execute("DROP INDEX idx_nodes_files_present")
                if failure == "utf8":
                    store.connection.execute(
                        f"UPDATE nodes SET {field} = CAST(x'ff' AS TEXT) WHERE id = ?", (last,),
                    )
                else:
                    store.connection.execute(f"UPDATE nodes SET {field} = ? WHERE id = ?", (value, last))
        before = store.connection.serialize()
        error = {"utf8": sqlite3.OperationalError, "type": TypeError}.get(failure, json.JSONDecodeError)
        cursors = capture_cursors(store)
        with store.transaction() if transaction else nullcontext():
            try:
                if failure:
                    # Keep the traceback alive: cleanup must not depend on GC.
                    with pytest.raises(error) as caught:
                        read_nodes(store, method, ids)
                    assert caught.value is not None
                else:
                    assert len(read_nodes(store, method, ids)) == 5
            finally:
                store.connection.row_factory = sqlite3.Row
            assert_closed(cursors)
            assert store.connection.in_transaction is transaction
            assert store.connection.serialize() == before
        # No active statement may block the next schema writer, even while the
        # caller retains a failed read's traceback.
        store.connection.execute("ALTER TABLE nodes ADD COLUMN reader_released INTEGER")


@pytest.mark.parametrize("method", ["list", "get"])
@pytest.mark.parametrize("transaction", [False, True])
def test_snapshot_ends_at_call_or_existing_transaction(tmp_path, method, transaction):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store, sqlite3.connect(path) as writer:
        nodes = [store.append_trace(f"Harbor record {i}.") for i in range(5)]
        ids = [node.id for node in nodes]
        changed = False
        cursors = []

        def rows(cursor, values):
            nonlocal changed
            if not changed:
                cursors.append(cursor)
                writer.execute("UPDATE nodes SET confidence = 0.25")
                writer.commit()
                changed = True
            return sqlite3.Row(cursor, values)

        with store.transaction() if transaction else nullcontext():
            store.connection.row_factory = rows
            try:
                result = read_nodes(store, method, ids)
            finally:
                store.connection.row_factory = sqlite3.Row
            assert {node.confidence for node in result} == {0.5}
            assert_closed(cursors)
            assert store.connection.in_transaction is transaction
            expected = 0.5 if transaction else 0.25
            assert {node.confidence for node in read_nodes(store, method, ids)} == {expected}
        assert {node.confidence for node in read_nodes(store, method, ids)} == {0.25}


@pytest.mark.parametrize("method", ["list", "get"])
def test_query_errors_propagate_without_writes(tmp_path, method):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = store.append_trace("Harbor record.")
        before = store.connection.serialize()

        def authorize(action, table, column, database, trigger):
            if action == sqlite3.SQLITE_READ and table == "nodes" and column == "content":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        store.connection.set_authorizer(authorize)
        try:
            with pytest.raises(sqlite3.DatabaseError, match="prohibited"):
                read_nodes(store, method, [node.id])
        finally:
            store.connection.set_authorizer(None)
        assert store.connection.serialize() == before
        assert not store.connection.in_transaction


def test_get_nodes_late_batch_failure_releases_all_cursors(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        with store.transaction():
            ids = [store.append_trace(f"Harbor record {i}.").id for i in range(1003)]
        # Match only the documented set/batch order to corrupt the last row
        # of the second query, after an entire batch has already hydrated.
        last = sorted(list(set(ids))[999:])[-1]
        with store.transaction():
            store.connection.execute("UPDATE nodes SET provenance = '[]' WHERE id = ?", (last,))
        before = store.connection.serialize()
        cursors = capture_cursors(store)
        try:
            with pytest.raises(TypeError) as caught:
                store.get_nodes(iter(ids))
        finally:
            store.connection.row_factory = sqlite3.Row
        assert caught.value is not None
        assert len(cursors) == 2
        assert_closed(cursors)
        assert store.connection.serialize() == before
        assert not store.connection.in_transaction
        store.connection.execute("ALTER TABLE nodes ADD COLUMN reader_released INTEGER")


def test_decoder_only_hydrates_owned_values(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = store.append_trace("Harbor record.", {"nested": {"value": [1, 2]}})
        row = store.connection.execute("SELECT * FROM nodes WHERE id = ?", (node.id,)).fetchone()
        original_row = tuple(row)
        queries = []
        before = store.connection.serialize()
        store.connection.set_trace_callback(queries.append)
        try:
            decoded = _node_from_row(row)
            assert decoded == node
            decoded.context["nested"]["value"].append(3)
            decoded.provenance["corrections"].append({"new": "local"})
            assert _node_from_row(row) == node
        finally:
            store.connection.set_trace_callback(None)
        assert not queries
        assert tuple(row) == original_row
        assert store.connection.serialize() == before
