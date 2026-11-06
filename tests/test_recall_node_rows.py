"""Recall admission releases raw rows and owns every batch until consumption ends."""

from contextlib import nullcontext
import json
import sqlite3
import weakref

import pytest

from living_memory.scope import ScopePlan, resolve_scope
from living_memory.storage import MemoryStore
from test_full_node_reads import assert_closed, capture_cursors


PLAN = resolve_scope(scope="project:harbor")


def read(store, method, ids, plan=PLAN):
    reader = store.get_recall_nodes if method == "nodes" else store.get_recall_node_ids
    return reader(iter(ids), plan)


def seed(store, count, *, mixed_scopes=False):
    with store.transaction():
        return [store.create_node(
            node_id=f"node-{i:04}", level=("trace", "concept", "schema")[i % 3],
            content=f"Harbor record {i}.",
            context={"scope": ("global", "project:harbor", "project:other", "session:survey")[i % 4]
                     if mixed_scopes else "project:harbor",
                     "agent": "inspector", "task": "survey",
                     "nested": {"values": [None, True, 2**60, "проверка"]}},
            embedding=[0.25, -0.5], timestamp="2026-01-02T00:00:00Z",
            stats={"access_count": 3, "last_accessed": "2026-01-03T00:00:00Z",
                   "usefulness_score": 0.6, "confidence": 0.8, "unique_agents": 2,
                   "temporal_hint": "during survey"},
            provenance={"source_traces": ["source"], "source_concepts": ["concept"],
                        "corrections": [{"old": "before", "new": "after"}], "custom": i},
        ) for i in range(count)]


@pytest.mark.parametrize("method", ["nodes", "ids"])
@pytest.mark.parametrize("plan", [
    PLAN,
    resolve_scope(scope="session:survey", ambient_context={"project": "harbor"}),
    resolve_scope(ambient_context={"project": "other"}),
    ScopePlan("global", ()),
])
def test_stream_preserves_full_values_admission_order_and_one_query_per_batch(tmp_path, method, plan):
    with MemoryStore(tmp_path / "recall.sqlite3") as store:
        nodes = seed(store, 1003, mixed_scopes=True)
        nodes[7] = store.soft_delete_node(nodes[7].id, "retired")
        requested = [node.id for node in reversed(nodes)] + [nodes[0].id, "missing"]
        unique = list(set(requested))
        admitted = {n.id: n for n in nodes if not n.decayed and plan.allows(n.scope)}
        order = [nid for start in range(0, len(unique), 999)
                 for nid in sorted(unique[start:start + 999]) if nid in admitted]
        before = store.connection.serialize()
        queries, cursors, refs = [], [], []

        class Row(sqlite3.Row):
            __slots__ = ("__weakref__",)

        def rows(cursor, values):
            # Previous raw rows may not accumulate behind decoded Nodes.
            # Allow the loop's one previous row while obtaining the next.
            assert sum(ref() is not None for ref in refs) <= 1
            if not cursors or cursor is not cursors[-1]:
                if cursors:
                    assert_closed(cursors)
                cursors.append(cursor)
            row = Row(cursor, values)
            refs.append(weakref.ref(row))
            return row

        store.connection.row_factory = rows
        store.connection.set_trace_callback(queries.append)
        try:
            result = read(store, method, requested, plan)
            assert read(store, method, [], plan) == ({} if method == "nodes" else set())
        finally:
            store.connection.row_factory = sqlite3.Row
            store.connection.set_trace_callback(None)
        assert result == (admitted if method == "nodes" else set(admitted))
        if method == "nodes":
            assert list(result) == order
        if admitted:
            assert_closed(cursors)
        assert all(ref() is None for ref in refs)
        expected_queries = 0 if plan.restricted and not plan.scopes else 2
        assert len(queries) == expected_queries
        projection = "*" if method == "nodes" else "id"
        assert all(sql.startswith(f"SELECT {projection} FROM nodes WHERE id IN (")
                   and "AND decayed = 0" in sql and sql.endswith("ORDER BY id") for sql in queries)
        assert store.connection.serialize() == before
        assert not store.connection.in_transaction


@pytest.mark.parametrize("transaction", [False, True])
@pytest.mark.parametrize("position", [0, 4])
@pytest.mark.parametrize("failure", ["json", "type", "numeric", "utf8"])
def test_required_decode_failure_closes_cursor_with_retained_traceback(tmp_path, transaction, position, failure):
    with MemoryStore(tmp_path / "recall.sqlite3") as store:
        ids = [node.id for node in seed(store, 5)]
        with store.transaction():
            field, value = {
                "json": ("provenance", "{malformed"), "type": ("provenance", "[]"),
                "numeric": ("access_count", "bad"), "utf8": ("content", None),
            }[failure]
            if failure == "utf8":
                store.connection.execute(
                    "UPDATE nodes SET content = CAST(x'ff' AS TEXT) WHERE id = ?", (ids[position],),
                )
            else:
                store.connection.execute(f"UPDATE nodes SET {field} = ? WHERE id = ?", (value, ids[position]))
        before = store.connection.serialize()
        error = {"json": json.JSONDecodeError, "type": TypeError,
                 "numeric": ValueError, "utf8": sqlite3.OperationalError}[failure]
        cursors = []
        # Capture even a cursor whose very first SQLite text conversion fails.
        class Connection:
            def execute(self, *args):
                cursor = connection.execute(*args)
                cursors.append(cursor)
                return cursor

        with store.transaction() if transaction else nullcontext():
            connection = store.connection
            store._conn = Connection()
            try:
                result = None
                with pytest.raises(error) as caught:
                    result = read(store, "nodes", ids)
                assert result is None
                assert caught.value.__traceback__ is not None
                assert_closed(cursors)
            finally:
                store._conn = connection
            assert store.connection.in_transaction is transaction
            assert store.connection.serialize() == before
        store.connection.execute("ALTER TABLE nodes ADD COLUMN reader_released INTEGER")


@pytest.mark.parametrize("failure", ["conversion", "sqlite"])
def test_identity_failure_closes_cursor_without_partial_admission(tmp_path, failure):
    with MemoryStore(tmp_path / "recall.sqlite3") as store:
        ids = [node.id for node in seed(store, 5)]
        cursors = []

        class BadID:
            def __str__(self):
                raise ValueError("identity conversion failed")

        def rows(cursor, values):
            if not cursors:
                cursors.append(cursor)
            if values[0] == ids[2]:
                if failure == "sqlite":
                    raise sqlite3.OperationalError("late read failed")
                return {"id": BadID()}
            return sqlite3.Row(cursor, values)

        before = store.connection.serialize()
        store.connection.row_factory = rows
        try:
            result = None
            with pytest.raises(ValueError if failure == "conversion" else sqlite3.OperationalError) as caught:
                result = read(store, "ids", ids)
            assert result is None
            assert caught.value.__traceback__ is not None
            assert_closed(cursors)
        finally:
            store.connection.row_factory = sqlite3.Row
        assert store.connection.serialize() == before
        store.connection.execute("ALTER TABLE nodes ADD COLUMN reader_released INTEGER")


@pytest.mark.parametrize("method", ["nodes", "ids"])
def test_empty_admission_still_closes_its_cursor(tmp_path, method):
    with MemoryStore(tmp_path / "recall.sqlite3") as store:
        nodes = seed(store, 2)
        store.soft_delete_node(nodes[0].id)
        store.update_node(nodes[1].id, context={"scope": "project:other"})
        connection, cursors = store.connection, []

        class Connection:
            def execute(self, *args):
                cursor = connection.execute(*args)
                cursors.append(cursor)
                return cursor

        store._conn = Connection()
        try:
            assert not read(store, method, [n.id for n in nodes] + ["missing"])
            assert len(cursors) == 1
            assert_closed(cursors)
        finally:
            store._conn = connection


@pytest.mark.parametrize("method", ["nodes", "ids"])
@pytest.mark.parametrize("failure", ["execute", "read"])
def test_next_batch_failure_closes_previous_and_current_cursors(tmp_path, method, failure):
    with MemoryStore(tmp_path / "recall.sqlite3") as store:
        ids = [node.id for node in seed(store, 1003)]
        last = sorted(list(set(ids))[999:])[-1]
        cursors = capture_cursors(store)
        factory = store.connection.row_factory

        def deny(action, table, column, *args):
            return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ else sqlite3.SQLITE_OK

        def rows(cursor, values):
            row = factory(cursor, values)
            if failure == "execute":
                store.connection.set_authorizer(deny)
            elif row["id"] == last:
                raise sqlite3.OperationalError("second batch read failed")
            return row

        before = store.connection.serialize()
        store.connection.row_factory = rows
        try:
            result = None
            with pytest.raises(sqlite3.DatabaseError) as caught:
                result = read(store, method, ids)
            assert result is None
            assert caught.value.__traceback__ is not None
            assert_closed(cursors)
            assert len(cursors) == (1 if failure == "execute" else 2)
        finally:
            store.connection.row_factory = sqlite3.Row
            store.connection.set_authorizer(None)
        assert store.connection.serialize() == before
        assert not store.connection.in_transaction
        store.connection.execute("ALTER TABLE nodes ADD COLUMN reader_released INTEGER")


@pytest.mark.parametrize("method", ["nodes", "ids"])
@pytest.mark.parametrize("transaction", [False, True])
def test_batch_and_caller_snapshot_boundaries(tmp_path, method, transaction):
    path = tmp_path / "recall.sqlite3"
    with MemoryStore(path) as store, sqlite3.connect(path) as writer:
        ids = [node.id for node in seed(store, 1003)]
        first_batch = set(list(set(ids))[:999])
        cursors = capture_cursors(store)
        factory = store.connection.row_factory
        changed = False

        def rows(cursor, values):
            nonlocal changed
            row = factory(cursor, values)
            if not changed:
                writer.execute("UPDATE nodes SET confidence = 0.25, decayed = 1")
                writer.commit()
                changed = True
            return row

        with store.transaction() if transaction else nullcontext():
            store.connection.row_factory = rows
            try:
                result = read(store, method, ids)
            finally:
                store.connection.row_factory = sqlite3.Row
            assert set(result) == (set(ids) if transaction else first_batch)
            if method == "nodes":
                assert {node.confidence for node in result.values()} == {0.8}
            assert_closed(cursors)
            assert store.connection.in_transaction is transaction
            assert set(read(store, method, ids)) == (set(ids) if transaction else set())
        assert not read(store, method, ids)
