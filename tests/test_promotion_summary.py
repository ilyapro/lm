"""Promotion observation reads bounded provenance without loading knowledge bodies."""

import asyncio
import json
import sqlite3
from threading import RLock

from fastmcp import Client
import pytest

from living_memory import resources, server, storage
from living_memory.storage import MemoryStore


EMPTY = {"promoted_concepts": 0, "promotion_events": 0, "source_scopes": []}


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
            {key: value for key, value in node.provenance.items()
             if key not in ("source_traces", "corrections")}
            for node in store.list_nodes(level="concept", scope="global", limit=limit)
        ]
        assert store.list_global_concept_provenance(limit=limit) == expected_provenance
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
