"""Scoped maintenance admits promotion owners before reading knowledge bodies."""

import asyncio
import json
import sqlite3
from threading import RLock

import pytest
from fastmcp import Client

from living_memory import consolidation as c, resources, storage
from living_memory.storage import MemoryStore
from test_promotion_summary import padded_provenances, promotion_runtime


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as opened:
        yield opened


def promotion(store, node_id, **kwargs):
    return store.create_node(
        node_id=node_id, level="concept", content=f"Promotion {node_id} awaits its sources.",
        provenance={"strategy": "cross-scope-promotion", "source_concepts": ["missing"]},
        **kwargs,
    )


@pytest.mark.parametrize("failure", [None, "parse", "object", "read", "close"])
def test_native_refresh_validates_entire_prefix_before_mutation(promotion_runtime, failure):
    mcp, guard, watch = promotion_runtime
    store = mcp.memory_store
    owner = promotion(store, "z-owner", context={"timestamp": "2026-01-01T00:00:00Z"})
    store.create_node(node_id="a-unrelated", level="concept", content="Unrelated metadata.",
                      context={"timestamp": "2026-01-01T00:00:00Z"})
    if failure in ("parse", "object"):
        with store.transaction():
            store.connection.execute("UPDATE nodes SET provenance=? WHERE id='a-unrelated'",
                                     ("{broken" if failure == "parse" else "[]",))
    watch.failure = failure
    before, changes = list(store.connection.iterdump()), store.connection.total_changes
    writes = []

    def authorize(action, table, column, database, trigger):
        if action in (sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_INSERT, sqlite3.SQLITE_DELETE):
            writes.append(table)
            assert watch.cursors and all(row.closed and row.exhausted for row in watch.cursors)
            assert guard._is_owned()
        return sqlite3.SQLITE_OK

    async def exercise():
        async with Client(mcp) as client:
            return await client.call_tool("memory_consolidate", {"scope": "project:empty"},
                                          raise_on_error=False)

    store.connection.set_authorizer(authorize)
    try:
        result = asyncio.run(exercise())
    finally:
        store.connection.set_authorizer(None)
    assert result.is_error == bool(failure)
    cursor, = watch.cursors
    assert cursor.reads == (1 if failure == "read" else 2) and cursor.closed
    assert not store.connection.in_transaction
    if failure:
        assert writes == [] and store.connection.total_changes == changes
        assert list(store.connection.iterdump()) == before
    else:
        assert writes and cursor.exhausted
        assert store.get_node(owner.id).decay_reason == "derivation: no current sources"


def test_native_refresh_retains_only_owner_ids(promotion_runtime):
    mcp, guard, watch = promotion_runtime
    store = mcp.memory_store
    padded_provenances(store)
    owner = promotion(store, "owner")

    async def exercise():
        async with Client(mcp) as client:
            result = await client.call_tool("memory_consolidate", {"scope": "project:empty"})
            assert not result.is_error

    asyncio.run(exercise())
    cursor, = watch.cursors
    assert cursor.reads == 97 and cursor.exhausted and cursor.closed
    assert watch.raw.peak <= 2 and watch.decoded.peak <= 2
    assert not watch.raw.live and not watch.decoded.live
    assert store.get_node(owner.id).decay_reason == "derivation: no current sources"


def test_refresh_cancellation_before_writes_closes_cursor(promotion_runtime, monkeypatch):
    mcp, guard, watch = promotion_runtime
    store = mcp.memory_store
    promotion(store, "z-owner", context={"timestamp": "2026-01-01T00:00:00Z"})
    store.create_node(node_id="a-unrelated", level="concept", content="Unrelated metadata.",
                      context={"timestamp": "2026-01-01T00:00:00Z"})
    original = c.Node.materializer.fget

    def cancel(node):
        if node.id == "a-unrelated":
            raise asyncio.CancelledError()
        return original(node)

    monkeypatch.setattr(c.Node, "materializer", property(cancel))
    before, changes = list(store.connection.iterdump()), store.connection.total_changes
    with pytest.raises(asyncio.CancelledError):
        c._refresh_promoted_global_concepts(store, step=lambda: guard)
    cursor, = watch.cursors
    assert cursor.reads == 2 and cursor.closed
    assert store.connection.total_changes == changes
    assert list(store.connection.iterdump()) == before


@pytest.mark.parametrize("column,raw", [
    ("context", "null"), ("embedding", "{broken"),
    ("source_traces", "{broken"), ("corrections", "{broken"),
])
def test_scoped_refresh_does_not_read_unrelated_bodies(store, monkeypatch, column, raw):
    unrelated = store.create_node(level="concept", content="Unrelated global knowledge.")
    needed = promotion(store, "needed")
    with store.transaction():
        store.connection.execute(f"UPDATE nodes SET {column}=? WHERE id=?", (raw, unrelated.id))
    hydrate = storage._node_from_row
    reads = []

    def observed(row):
        reads.append(row["id"])
        return hydrate(row)

    monkeypatch.setattr(storage, "_node_from_row", observed)
    result = c.memory_consolidate(store, scope="project:unrelated-pass")
    assert result.traces_considered == result.clusters_considered == 0
    assert unrelated.id not in reads
    assert store.get_node(needed.id).decay_reason == "derivation: no current sources"


def test_admission_uses_materializer_not_observation_metadata(store, monkeypatch):
    guard = RLock()
    for node_id, provenance in (
        ("owner", {"strategy": "cross-scope-promotion"}),
        ("summary-only", {"promoted_from": ["project:a"]}),
        ("procedural", {"strategy": "procedural"}),
        ("cluster", {"strategy": "token-jaccard"}),
        ("embedding", {"strategy": "embedding-cosine"}),
        ("legacy", {"cluster_key": None}),
        ("legacy-null", {"strategy": None, "cluster_key": None}),
        ("unknown", {"strategy": "other"}),
        ("list", {"strategy": ["cross-scope-promotion"]}),
    ):
        store.create_node(node_id=node_id, level="concept", content=node_id, provenance=provenance)
    for node_id, level, scope in (
        ("local", "concept", "project:a"), ("trace", "trace", "global"),
        ("schema", "schema", "global"),
    ):
        store.create_node(node_id=node_id, level=level, content=node_id,
                          context={"scope": scope}, provenance={"strategy": "cross-scope-promotion"})
    decayed = promotion(store, "decayed")
    store.soft_delete_node(decayed.id)
    corrected = promotion(store, "corrected")
    store.update_node(corrected.id, provenance={**corrected.provenance, "corrections": [{"new": "replacement"}]})
    superseded = promotion(store, "superseded")
    store.create_connection("trace", superseded.id, "supersedes")
    assert resources.promotion_summary(store)["promoted_concepts"] == 1
    refreshed, hydrated, unguarded = [], [], []
    hydrate = storage._node_from_row

    def observed(row):
        hydrated.append(row["id"])
        return hydrate(row)

    def refresh(store, node, **kwargs):
        assert store.connection.in_transaction and guard._is_owned()
        refreshed.append(node.id)
        return node

    monkeypatch.setattr(storage, "_node_from_row", observed)
    monkeypatch.setattr(c, "_refresh_derived_node", refresh)
    store.connection.set_trace_callback(lambda sql: unguarded.append(sql) if not guard._is_owned() else None)
    try:
        c._refresh_promoted_global_concepts(store, step=lambda: guard)
    finally:
        store.connection.set_trace_callback(None)
    assert refreshed == ["owner"]
    assert set(hydrated) == {"owner", "corrected", "superseded"}
    assert unguarded == []


@pytest.mark.parametrize("limit,expected", [(0, []), (1, []), (2, ["b"]), (3, ["b", "a"])])
def test_refresh_cap_precedes_owner_filter(store, monkeypatch, limit, expected):
    for node_id in ("a", "b"):
        promotion(store, node_id, context={"timestamp": "2024-01-01T00:00:00Z"})
    store.create_node(level="concept", content="Newest unrelated concept.",
                      context={"timestamp": "2025-01-01T00:00:00Z"})
    project = store.stream_global_concept_provenance

    def bounded(*, limit: int):
        assert limit == 100_000
        return project(limit=cap)

    cap = limit
    monkeypatch.setattr(store, "stream_global_concept_provenance", bounded)
    result = c._refresh_promoted_global_concepts(store)
    assert [node.id for node in result] == expected
    assert all(node.decayed for node in result)


@pytest.mark.parametrize("failure", ["owner_body", "source_body", "storage"])
def test_required_errors_remain_visible_and_leave_transaction_unchanged(store, monkeypatch, failure):
    source = store.create_node(level="concept", content="Current project evidence.",
                               context={"scope": "project:a"})
    owner = promotion(store, "owner")
    store.update_node(owner.id, provenance={**owner.provenance, "source_concepts": [source.id]})
    if failure != "storage":
        node_id = owner.id if failure == "owner_body" else source.id
        with store.transaction():
            store.connection.execute("UPDATE nodes SET embedding=? WHERE id=?", ("{broken", node_id))
        expected = json.JSONDecodeError
    else:
        def fail(*args, **kwargs):
            raise sqlite3.OperationalError("required storage unavailable")
        monkeypatch.setattr(store, "get_node", fail)
        expected = sqlite3.OperationalError
    before = list(store.connection.iterdump())
    with pytest.raises(expected):
        c.memory_consolidate(store, scope="project:unrelated-pass")
    assert not store.connection.in_transaction
    assert list(store.connection.iterdump()) == before
