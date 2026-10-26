"""Pending discovery reads source bodies only after comparing identities."""

from contextlib import closing
import sqlite3

import pytest

from living_memory.chunking import TextChunk
from living_memory.storage import MemoryStore


SOURCE_INDEXES = {
    "idx_nodes_active_scope_source",
    "idx_nodes_active_level_source",
    "idx_chunk_source",
}
OLD_INDEXES = {"idx_nodes_active_scope_level", "idx_nodes_level_scope_active"}


def seed(store):
    wanted = []
    completed = []
    for scope in ("project:pending", "global"):
        for level in ("trace", "concept", "schema"):
            for kind in ("missing", "stale", "current", "mixed", "decayed", "blank"):
                node = store.create_node(
                    level=level,
                    content=" \t\n\v\f\r " if kind == "blank" else f"{scope} {level} {kind}",
                    context={"scope": scope},
                    timestamp="2026-01-01T00:00:00Z" if kind == "blank" else
                              "2026-01-03T00:00:00Z" if kind == "missing" else
                              "2026-01-02T00:00:00Z",
                )
                if kind in {"stale", "current", "mixed"}:
                    store.replace_node_chunks(node.id, [
                        (TextChunk(node.content, index, 0, 3, 0, len(node.content)), [1.0, 0.0])
                        for index in range(2)
                    ])
                    if kind in {"stale", "mixed"}:
                        with store.transaction():
                            store.connection.execute(
                                "UPDATE node_chunk_embeddings SET content_fingerprint = 'stale' "
                                "WHERE node_id = ? AND chunk_index >= ?",
                                (node.id, 0 if kind == "stale" else 1),
                            )
                if kind == "decayed":
                    store.soft_delete_node(node.id)
                elif kind in {"missing", "stale"}:
                    wanted.append(node)
                elif kind in {"current", "mixed"}:
                    completed.append(node)
    return sorted(wanted, key=lambda n: (n.timestamp, n.id)), completed


@pytest.mark.parametrize("scope", [None, "project:pending", "global", "project:absent"])
@pytest.mark.parametrize("level", [None, "trace", "concept", "schema"])
def test_pending_filters_order_limit_and_covered_reads(tmp_path, scope, level):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        wanted, completed = seed(store)
        wanted = [n for n in wanted if (scope is None or n.scope == scope)
                  and (level is None or n.level == level)]
        trimmed = []
        store.connection.create_function(
            "trim", 2, lambda content, chars: trimmed.append(content) or content.strip(chars),
        )
        statements = []
        store.connection.set_trace_callback(statements.append)
        for limit in (0, 1, 3, -1):
            result = store.list_unchunked_nodes(scope=scope, level=level, limit=limit)
            assert result == (wanted if limit < 0 else wanted[:limit])
        store.connection.set_trace_callback(None)
        assert not {n.content for n in completed}.intersection(trimmed)

        # Inspect the statement issued by MemoryStore, not a second query copy.
        sql = next(s for s in reversed(statements) if "WITH pending" in s)
        plan = [row[3] for row in store.connection.execute("EXPLAIN QUERY PLAN " + sql)]
        assert any("USING COVERING INDEX idx_nodes_active_" in step for step in plan)
        assert any("USING COVERING INDEX idx_chunk_source" in step for step in plan)
        assert not any("SCAN c" in step for step in plan)


def test_empty_discovery_and_fresh_source_changes(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        assert store.list_unchunked_nodes() == []
        node = store.create_node(level="concept", content="Current source", embedding=[1.0, 0.0])
        assert store.list_unchunked_nodes() == []
        store.update_node(node.id, stats={"confidence": 0.4})
        assert store.list_unchunked_nodes() == []
        updated = store.update_node(node.id, content="Changed source")
        assert store.list_unchunked_nodes() == [updated]
        store.update_node(node.id, embedding=[1.0, 0.0])
        assert store.list_unchunked_nodes() == []
        fresh = store.append_trace("Fresh source", {"scope": "project:new"})
        assert store.list_unchunked_nodes() == [fresh]
        moved = store.update_node(fresh.id, level="schema", context={"scope": "project:moved"})
        assert store.list_unchunked_nodes(scope="project:new") == []
        assert store.list_unchunked_nodes(level="trace") == []
        assert store.list_unchunked_nodes(scope="project:moved", level="schema") == [moved]
        store.soft_delete_node(fresh.id)
        assert store.list_unchunked_nodes() == []
        # Empty strings and NULL fingerprints can occur in supported old files.
        with store.transaction():
            store.connection.execute("UPDATE nodes SET content_fingerprint = NULL WHERE id = ?", (node.id,))
        assert [n.id for n in store.list_unchunked_nodes()] == [node.id]
        store.update_node(node.id, content="")
        assert store.list_unchunked_nodes() == []
        with pytest.raises(ValueError):
            store.list_unchunked_nodes(level="invalid")


def schema(conn):
    return dict(conn.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL"))


def rows(conn):
    return {table: [tuple(row) for row in conn.execute(f"SELECT rowid, * FROM {table} ORDER BY id")]
            for table in ("nodes", "node_chunk_embeddings")}


def test_existing_database_replaces_only_redundant_indexes(tmp_path):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store:
        wanted, _ = seed(store)
        original_rows = rows(store.connection)
    with closing(sqlite3.connect(path)) as conn, conn:
        for name in SOURCE_INDEXES:
            conn.execute(f"DROP INDEX {name}")
        conn.execute("CREATE INDEX idx_nodes_active_scope_level ON nodes(scope, level, id) WHERE decayed = 0")
        conn.execute("CREATE INDEX idx_nodes_level_scope_active ON nodes(level, scope) WHERE decayed = 0")
        before = schema(conn)
        assert SOURCE_INDEXES.isdisjoint(before)
        assert OLD_INDEXES <= before.keys()

    for _ in range(2):
        with MemoryStore(path) as store:
            after = schema(store.connection)
            assert after.keys() - before.keys() == SOURCE_INDEXES
            assert before.keys() - after.keys() == OLD_INDEXES
            assert all(after[name] == sql for name, sql in before.items() if name not in OLD_INDEXES)
            assert rows(store.connection) == original_rows
            assert store.list_unchunked_nodes() == wanted
            assert store.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
