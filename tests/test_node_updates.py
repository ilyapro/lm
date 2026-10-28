"""Node metadata preserves source storage and depends only on required indexes."""

import sqlite3

import pytest

from living_memory import storage
from living_memory.storage import MemoryStore


@pytest.fixture
def node_store(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = store.create_node(
            level="concept", content="cobalt routing ownership", embedding=[1.0, 0.0],
            context={"scope": "project:routing", "agent": "author", "task": "amber",
                     "procedure": "source context " * 1000},
            provenance={"source_traces": ["source"], "corrections": [{"old": "before"}],
                        "source_concepts": ["concept"], "strategy": "cross-scope-promotion"},
            stats={"unique_agents": 2, "confidence": 0.8},
        )
        yield store, node


def test_metadata_and_equal_inputs_preserve_source_bytes_without_encoding(node_store, monkeypatch):
    store, node = node_store
    before = dict(store.connection.execute("SELECT * FROM nodes WHERE id = ?", (node.id,)).fetchone())
    chunks = store.list_node_chunks(node.id)
    writes = []

    def observe(action, table, column, db, trigger):
        if action == sqlite3.SQLITE_UPDATE and table == "nodes":
            writes.append(column)
        return sqlite3.SQLITE_OK

    def unexpected(*args, **kwargs):
        pytest.fail("unchanged source must not be serialized or fingerprinted")

    monkeypatch.setattr(storage, "_json_dumps", unexpected)
    monkeypatch.setattr(storage, "_content_fingerprint", unexpected)
    monkeypatch.setattr(storage, "_utc_now", lambda: "2030-01-01T00:00:00Z")
    store.connection.set_authorizer(observe)
    try:
        updated = store.update_node(node.id, stats={"usefulness_score": 0.4})
        same = store.update_node(node.id, content=updated.content, context=updated.context,
                                 provenance=updated.provenance, stats=updated.stats, level=updated.level)
    finally:
        store.connection.set_authorizer(None)
    assert updated == same == store.get_node(node.id)
    assert updated.updated_at == "2030-01-01T00:00:00Z" != node.updated_at
    assert set(writes) == {"usefulness_score", "updated_at"}
    after = dict(store.connection.execute("SELECT * FROM nodes WHERE id = ?", (node.id,)).fetchone())
    assert {key: value for key, value in after.items() if value != before[key]} == {
        "usefulness_score": 0.4, "updated_at": "2030-01-01T00:00:00Z",
    }
    assert store.list_node_chunks(node.id) == chunks


def test_context_defaults_stats_normalization_and_provenance_replacement(node_store):
    store, node = node_store
    updated = store.update_node(node.id, context={"agent": 12}, stats={
        "unique_agents": "1", "confidence": 0.9, "access_count": "7",
        "last_accessed": 123, "temporal_hint": 456,
    }, provenance={"source_concepts": ["replacement"]})
    assert updated.context == {"agent": 12, "scope": node.scope, "timestamp": node.timestamp}
    assert updated.agent == "12" and updated.task == node.task
    assert updated.timestamp == node.timestamp and updated.created_at == node.created_at
    assert updated.stats == {
        "unique_agents": 1, "confidence": 0.5, "access_count": 7,
        "last_accessed": "123", "temporal_hint": "456", "usefulness_score": 0.0,
    }
    assert updated.provenance == {
        "source_concepts": ["replacement"],
        "source_traces": node.source_traces, "corrections": node.corrections,
    }
    cleared = store.update_node(node.id, provenance={"source_traces": (), "corrections": None})
    assert cleared.provenance == {"source_traces": [], "corrections": []}
    with store.transaction():
        store.connection.execute("UPDATE nodes SET context = '{}', confidence = 0.9 WHERE id = ?", (node.id,))
    normalized = store.update_node(node.id)
    assert normalized.context == {"scope": node.scope, "timestamp": node.timestamp}
    assert normalized.confidence == 0.5


@pytest.mark.parametrize("change", [
    {"content": "violet garden"}, {"context": {"task": "violet"}},
    {"context": {"scope": "project:garden"}}, {"level": "schema"},
])
def test_required_fts_failure_rolls_back_and_real_changes_remain_searchable(node_store, change):
    store, node = node_store
    chunks = store.list_node_chunks(node.id)
    denied = []

    def deny_fts(action, table, column, db, trigger):
        if trigger == "nodes_fts_update" and action in (
            sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
        ):
            denied.append(table)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    store.connection.set_authorizer(deny_fts)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            with store.transaction():
                store.update_node(node.id, stats={"usefulness_score": 0.7})
                store.update_node(node.id, **change)
    finally:
        store.connection.set_authorizer(None)
    assert denied
    assert store.get_node(node.id) == node
    assert store.list_node_chunks(node.id) == chunks
    updated = store.update_node(node.id, **change)
    fts = store.connection.execute(
        "SELECT content, task, level, scope FROM nodes_fts WHERE node_id = ?", (node.id,),
    ).fetchone()
    assert tuple(fts) == (updated.content, updated.task, updated.level, updated.scope)
    query = "violet" if "content" in change or "task" in change.get("context", {}) else "cobalt"
    matches = store.connection.execute(
        "SELECT node_id FROM nodes_fts WHERE nodes_fts MATCH ? AND level = ? AND scope = ?",
        (query, updated.level, updated.scope),
    ).fetchall()
    assert [row[0] for row in matches] == [node.id]
    if query == "violet":
        old_query = "cobalt" if "content" in change else "amber"
        assert not store.connection.execute(
            "SELECT node_id FROM nodes_fts WHERE nodes_fts MATCH ?", (old_query,),
        ).fetchall()


def test_metadata_update_does_not_hide_corrupt_source(node_store):
    store, node = node_store
    with store.transaction():
        store.connection.execute("UPDATE nodes SET provenance = 'not json' WHERE id = ?", (node.id,))
    before = tuple(store.connection.execute("SELECT * FROM nodes WHERE id = ?", (node.id,)).fetchone())
    with pytest.raises(ValueError):
        store.update_node(node.id, stats={"usefulness_score": 0.7})
    assert tuple(store.connection.execute("SELECT * FROM nodes WHERE id = ?", (node.id,)).fetchone()) == before


@pytest.mark.parametrize("field", ["context", "provenance"])
def test_json_updates_preserve_value_types(node_store, field):
    store, node = node_store
    store.update_node(node.id, **{field: {"value": 1}})
    updated = store.update_node(node.id, **{field: {"value": True}})
    assert getattr(updated, field)["value"] is True
