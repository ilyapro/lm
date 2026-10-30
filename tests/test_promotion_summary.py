"""Promotion observation reads bounded provenance without loading knowledge bodies."""

import asyncio
import json
import sqlite3
from threading import RLock
from weakref import WeakValueDictionary

from fastmcp import Client
import pytest

from living_memory import resources, server, storage
from living_memory.storage import MemoryStore


EMPTY = {"promoted_concepts": 0, "promotion_events": 0, "source_scopes": []}


class Retention:
    def __init__(self):
        self.live = WeakValueDictionary()
        self.peak = 0

    def track(self, value):
        self.live[id(value)] = value
        self.peak = max(self.peak, len(self.live))
        return value


class RawProvenance(str):
    pass


class DecodedProvenance(dict):
    pass


class ProvenanceCursor:
    def __init__(self, cursor, watch):
        self.cursor, self.watch = cursor, watch
        self.reads = 0
        self.closed = self.exhausted = False

    def __iter__(self):
        return self

    def __next__(self):
        assert self.watch.guard._is_owned()
        if self.watch.failure == "read" and self.reads == 1:
            raise sqlite3.OperationalError("late provenance read")
        try:
            row = next(self.cursor)
        except StopIteration:
            self.exhausted = True
            raise
        self.reads += 1
        return {"id": row["id"], "provenance": self.watch.raw.track(RawProvenance(row["provenance"]))}

    def fetchall(self):
        pytest.fail("promotion traversal must not collect encoded rows")

    def close(self):
        assert self.watch.guard._is_owned()
        self.cursor.close()
        self.closed = True
        if self.watch.failure == "close":
            raise sqlite3.OperationalError("late provenance close")


class ProvenanceConnection:
    def __init__(self, connection, guard):
        self.connection, self.guard = connection, guard
        self.cursors = []
        self.failure = None
        self.raw, self.decoded = Retention(), Retention()

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def execute(self, sql, *args):
        cursor = self.connection.execute(sql, *args)
        if " ".join(sql.split()).startswith("SELECT id, provenance FROM nodes"):
            assert self.guard._is_owned()
            cursor = ProvenanceCursor(cursor, self)
            self.cursors.append(cursor)
        return cursor


@pytest.fixture
def promotion_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
    guard = RLock()
    monkeypatch.setattr(server, "RLock", lambda: guard)
    mcp = server.create_mcp_server(tmp_path / "native.sqlite3")
    store = mcp.memory_store
    watch = ProvenanceConnection(store.connection, guard)
    monkeypatch.setattr(store, "_conn", watch)
    decode = storage._json_loads

    def tracked(raw, default):
        value = decode(raw, default)
        if isinstance(raw, RawProvenance) and isinstance(value, dict):
            return watch.decoded.track(DecodedProvenance(value))
        return value

    monkeypatch.setattr(storage, "_json_loads", tracked)
    try:
        yield mcp, guard, watch
    finally:
        mcp.auto_consolidation.close()
        store.close()


def padded_provenances(store, count=96):
    for index in range(count):
        store.create_node(
            node_id=f"metadata-{index:04}", level="concept", content=f"Metadata fixture {index}.",
            provenance={"padding": "x" * 16384, "promoted_from": ["project:a", "project:b"],
                        "promotion_events": [1, 2]},
        )


@pytest.mark.parametrize("consumer", ["status", "stats"])
def test_native_summary_releases_previous_provenance(promotion_runtime, consumer):
    mcp, guard, watch = promotion_runtime
    store = mcp.memory_store
    padded_provenances(store)
    before, changes = list(store.connection.iterdump()), store.connection.total_changes
    store.connection.execute("PRAGMA query_only=ON")

    async def exercise():
        async with Client(mcp) as client:
            if consumer == "status":
                result = (await client.call_tool("memory_status", {"scope": "project:a"})).data
            else:
                result = json.loads((await client.read_resource("memory://stats"))[0].text)
            assert result["promotions"] == {
                "promoted_concepts": 96, "promotion_events": 192,
                "source_scopes": ["project:a", "project:b"],
            }

    asyncio.run(exercise())
    cursor, = watch.cursors
    assert cursor.reads == 96 and cursor.exhausted and cursor.closed
    assert watch.raw.peak <= 2 and watch.decoded.peak <= 2
    assert not watch.raw.live and not watch.decoded.live
    assert store.connection.total_changes == changes
    assert list(store.connection.iterdump()) == before


@pytest.mark.parametrize("failure", [None, "read", "close", "parse", "object", "reduce", "cancel"])
@pytest.mark.parametrize("consumer", ["status", "stats"])
def test_summary_late_failures_close_cursor(promotion_runtime, monkeypatch, failure, consumer):
    mcp, guard, watch = promotion_runtime
    store = mcp.memory_store
    for node_id in ("a-unrelated", "z-promotion"):
        store.create_node(
            node_id=node_id, level="concept", content=node_id,
            context={"timestamp": "2026-01-01T00:00:00Z"},
            provenance={"promoted_from": ["project:a"]} if node_id.startswith("z") else {},
        )
    if failure in ("parse", "object"):
        with store.transaction():
            store.connection.execute("UPDATE nodes SET provenance=? WHERE id='a-unrelated'",
                                     ("{broken" if failure == "parse" else "[]",))
    watch.failure = failure
    if failure in ("reduce", "cancel"):
        original = resources.string_list

        def fail(value):
            if value is None:
                if failure == "cancel":
                    raise asyncio.CancelledError()
                raise ValueError("late reducer failure")
            return original(value)

        monkeypatch.setattr(resources, "string_list", fail)
    before, changes = list(store.connection.iterdump()), store.connection.total_changes
    with guard:
        call = (lambda: resources.memory_status(store, "project:a")) if consumer == "status" else (
            lambda: resources.memory_stats(store))
        errors = {"parse": json.JSONDecodeError, "object": TypeError, "reduce": ValueError,
                  "cancel": asyncio.CancelledError, "read": sqlite3.OperationalError,
                  "close": sqlite3.OperationalError}
        if failure:
            with pytest.raises(errors[failure]):
                call()
        else:
            assert call()["promotions"]["promoted_concepts"] == 1
    cursor, = watch.cursors
    assert cursor.reads == (1 if failure == "read" else 2) and cursor.closed
    assert store.connection.total_changes == changes
    assert list(store.connection.iterdump()) == before


@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError])
def test_traversal_early_exit_releases_row_and_cursor(promotion_runtime, failure):
    mcp, guard, watch = promotion_runtime
    padded_provenances(mcp.memory_store, count=2)

    def consume():
        with guard, mcp.memory_store.stream_global_concept_provenance(limit=2) as rows:
            assert next(rows)[1]["promoted_from"] == ["project:a", "project:b"]
            if failure:
                raise failure()
        return rows

    if failure:
        with pytest.raises(failure):
            consume()
    else:
        assert list(consume()) == []
    cursor, = watch.cursors
    assert cursor.reads == 1 and cursor.closed and not cursor.exhausted
    assert not watch.raw.live and not watch.decoded.live
    with pytest.raises(sqlite3.ProgrammingError):
        next(cursor.cursor)


@pytest.mark.parametrize(
    "raw,sources,events",
    [
        ("", [], 0),
        ("{}", [], 0),
        ('{"promoted_from":null}', [], 0),
        ('{"promoted_from":"project:a"}', [], 0),
        ('{"promoted_from":7}', [], 0),
        ('{"promoted_from":true}', [], 0),
        ('{"promoted_from":{"project:a":1}}', [], 0),
        ('{"promoted_from":[]}', [], 0),
        ('{"promoted_from":["project:a"]}', ["project:a"], 1),
        ('{"promoted_from":["project:a"],"promotion_events":null}', ["project:a"], 1),
        ('{"promoted_from":["project:a"],"promotion_events":{}}', ["project:a"], 1),
        ('{"promoted_from":["project:a"],"promotion_events":"event"}', ["project:a"], 1),
        ('{"promoted_from":["project:a"],"promotion_events":false}', ["project:a"], 1),
        ('{"promoted_from":["project:a"],"promotion_events":12}', ["project:a"], 1),
        ('{"promoted_from":["project:a","project:a"],"promotion_events":[]}',
         ["project:a"], 0),
        ('{"promoted_from":["project:a","project:b"],"promotion_events":[{},null,7]}',
         ["project:a", "project:b"], 3),
        ('{"promoted_from":["project:a",7,true,null,{"k":1},["x"]]}',
         ["7", "None", "True", "['x']", "project:a", "{'k': 1}"], 1),
        ('{"promoted_from":[9223372036854775808123,NaN,Infinity,-Infinity]}',
         ["-inf", "9223372036854775808123", "inf", "nan"], 1),
        ('{"promoted_from":["first"],"promoted_from":["project:a"]}', ["project:a"], 1),
    ],
)
def test_status_and_stats_preserve_provenance_types(tmp_path, raw, sources, events):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = store.create_node(level="concept", content="promotion metadata fixture")
        with store.connection:
            store.connection.execute("UPDATE nodes SET provenance=? WHERE id=?", (raw, node.id))
        expected = {
            "promoted_concepts": int(bool(sources)),
            "promotion_events": events,
            "source_scopes": sources,
        }
        assert resources.memory_stats(store)["promotions"] == expected
        for scope in (None, "global", "project:a", "project:missing"):
            scoped = expected if scope in (None, "global") or scope in sources else EMPTY
            assert resources.memory_status(store, scope)["promotions"] == scoped


@pytest.mark.parametrize("raw,error", [
    ("{broken", json.JSONDecodeError), (" ", json.JSONDecodeError),
    ("null", TypeError), ("[]", TypeError), ("true", TypeError),
    ("7", TypeError), ('"text"', TypeError),
])
def test_invalid_provenance_remains_a_visible_error(tmp_path, raw, error):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = store.create_node(level="concept", content="invalid provenance fixture")
        with store.connection:
            store.connection.execute("UPDATE nodes SET provenance=? WHERE id=?", (raw, node.id))
        with pytest.raises(error):
            store.list_nodes(level="concept", scope="global")
        with pytest.raises(error):
            resources.memory_status(store)
        with pytest.raises(error):
            resources.memory_stats(store)


@pytest.mark.parametrize("limit,selected", [(0, []), (1, []), (2, ["b"]),
                                           (3, ["a", "b"]), (4, ["a", "b", "z"])])
def test_selection_filters_order_and_cap_precede_source_scope(tmp_path, monkeypatch, limit, selected):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for node_id, year in (("a", 2024), ("z", 2023), ("b", 2024), ("newest", 2025)):
            store.create_node(
                node_id=node_id, level="concept", content=f"ordered concept {node_id}",
                context={"scope": "global", "timestamp": f"{year}-01-01T00:00:00Z"},
                provenance={} if node_id == "newest" else {"promoted_from": [node_id]},
            )
        for node_id, level, scope, decayed in (
            ("decayed", "concept", "global", True),
            ("local", "concept", "project:a", False),
            ("trace", "trace", "global", False),
            ("schema", "schema", "global", False),
        ):
            store.create_node(
                node_id=node_id, level=level, content=f"excluded {node_id}",
                context={"scope": scope, "timestamp": "2026-01-01T00:00:00Z"},
                provenance={"promoted_from": [node_id]},
            )
            if decayed:
                store.soft_delete_node(node_id, reason="excluded fixture")
        expected_provenance = [
            (node.id, {key: value for key, value in node.provenance.items()
             if key not in ("source_traces", "corrections")})
            for node in store.list_nodes(level="concept", scope="global", limit=limit)
        ]
        with store.stream_global_concept_provenance(limit=limit) as provenances:
            assert list(provenances) == expected_provenance
        monkeypatch.setattr(resources, "DEFAULT_CONCEPT_LIMIT", limit)
        assert resources.memory_stats(store)["promotions"] == {
            "promoted_concepts": len(selected), "promotion_events": len(selected),
            "source_scopes": selected,
        }
        # Source filtering does not refill the selected prefix from older concepts.
        assert resources.promotion_summary(store, "z") == (
            {"promoted_concepts": 1, "promotion_events": 1, "source_scopes": ["z"]}
            if "z" in selected else EMPTY
        )


def test_native_status_and_stats_read_only_provenance_under_runtime_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    guard = RLock()
    monkeypatch.setattr(server, "RLock", lambda: guard)
    mcp = server.create_mcp_server(tmp_path / "native.sqlite3")
    store, conn = mcp.memory_store, mcp.memory_store.connection
    try:
        seed = store.create_node(
            level="concept", content="promoted concept", context={"scope": "global"},
            provenance={"promoted_from": ["project:a", "project:b"], "promotion_events": [1, 2]},
        )
        row = dict(conn.execute("SELECT * FROM nodes WHERE id=?", (seed.id,)).fetchone())
        columns = list(row)
        insert = f"INSERT INTO nodes ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})"
        unrelated = dict(
            row, content="unrelated knowledge " * 512,
            context=json.dumps({"scope": "global", "unrelated": "x" * 65536}),
            embedding=json.dumps([0.125] * 4096), source_traces=json.dumps(["source"] * 1024),
            corrections=json.dumps(["correction"] * 1024), provenance="{}",
        )
        with store.transaction():
            for index in range(1000):
                entry = dict(unrelated, id=f"unrelated-{index:04}")
                conn.execute(insert, [entry[key] for key in columns])
        scopes = (None, "global", "project:a", "project:missing")
        expected_status = {scope: resources.memory_status(store, scope) for scope in scopes}
        expected_stats = resources.memory_stats(store)
        assert expected_stats["promotions"] == {
            "promoted_concepts": 1, "promotion_events": 2,
            "source_scopes": ["project:a", "project:b"],
        }
        before, changes = list(conn.iterdump()), conn.total_changes
        conn.execute("PRAGMA query_only=ON")
        reads, unguarded = set(), []
        allowed = {"", "id", "level", "scope", "decayed", "confidence", "timestamp", "provenance"}

        def authorize(action, table, column, database, trigger):
            if action == sqlite3.SQLITE_READ and table == "nodes":
                reads.add(column)
                if not guard._is_owned():
                    unguarded.append(column)
                if column not in allowed:
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        def reject_hydration(*args, **kwargs):
            pytest.fail("promotion observation must not hydrate knowledge nodes")

        monkeypatch.setattr(storage, "_node_from_row", reject_hydration)
        monkeypatch.setattr(storage, "Node", reject_hydration)
        monkeypatch.setattr(store, "list_nodes", reject_hydration)
        conn.set_authorizer(authorize)
        try:
            with guard, pytest.raises(sqlite3.DatabaseError, match="prohibited"):
                conn.execute("SELECT context FROM nodes LIMIT 1")
            reads.clear()

            async def exercise():
                async with Client(mcp) as client:
                    for scope in scopes:
                        response = await client.call_tool("memory_status", {"scope": scope})
                        assert not response.is_error
                        assert response.data == expected_status[scope]
                    response = await client.read_resource("memory://stats")
                    assert json.loads(response[0].text) == expected_stats

            asyncio.run(exercise())
            assert "provenance" in reads
            assert reads <= allowed
            assert unguarded == []
        finally:
            conn.set_authorizer(None)
        assert conn.total_changes == changes
        assert list(conn.iterdump()) == before
    finally:
        mcp.auto_consolidation.close()
        store.close()
