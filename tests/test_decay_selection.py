"""Lifecycle selection preserves TTL semantics without reading live payloads."""

from datetime import UTC, datetime
import sqlite3

import pytest

from living_memory.decay import soft_delete_expired
from living_memory.storage import MemoryStore


NOW = datetime(2026, 10, 23, tzinfo=UTC)


@pytest.mark.parametrize(
    "timestamp,created_at,last_accessed,created_expires,read_expires",
    [
        ("2020-01-01", "2026-10-23", "2026-10-23", True, False),
        ("invalid", "2020-01-01", None, True, False),
        ("invalid", "invalid", "2020-01-01", False, True),
        ("2020-01-01", "2026-10-23", "invalid", True, True),
        ("2026-10-16T07:00:00+07:00", "2020-01-01", None, False, False),
        ("2026-10-15T23:59:59Z", "2026-10-23", None, True, True),
    ],
)
@pytest.mark.parametrize("anchor", ["created", "last_accessed"])
def test_timestamp_fallbacks_and_strict_cutoff(
    tmp_path, timestamp, created_at, last_accessed, created_expires, read_expires, anchor
):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = store.append_trace("complete fact", {"scope": "project:decay"})
        with store.transaction():
            store.connection.execute(
                "UPDATE nodes SET timestamp=?, created_at=?, last_accessed=? WHERE id=?",
                (timestamp, created_at, last_accessed, node.id),
            )
        expired = soft_delete_expired(store, now=NOW, ttl_days=7, ttl_anchor=anchor)
        should_expire = created_expires if anchor == "created" else read_expires
        assert [item.id for item in expired] == ([node.id] if should_expire else [])
        assert store.get_node(node.id).decayed is should_expire
        if expired:
            assert expired[0].content == "complete fact"
            assert expired[0].decay_reason == "ttl expired"


def test_scope_levels_and_tied_retirement_order(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        nodes = [
            store.create_node(
                level=level,
                content=f"fact {index}",
                context={"scope": scope, "timestamp": "2020-01-01"},
            )
            for index, (level, scope) in enumerate([
                ("trace", "project:decay"),
                ("trace", "project:decay"),
                ("trace", "project:other"),
                ("schema", "project:decay"),
            ])
        ]
        expected = sorted(node.id for node in nodes[:2])
        first = soft_delete_expired(
            store, scope="project:decay", now=NOW, ttl_days=7, max_expired=1
        )
        second = soft_delete_expired(
            store, scope="project:decay", now=NOW, ttl_days=7, max_expired=1
        )
        assert [node.id for node in first + second] == expected
        assert not store.get_node(nodes[2].id).decayed
        assert not store.get_node(nodes[3].id).decayed
        schema = soft_delete_expired(
            store, scope="project:decay", levels=("schema",), now=NOW, ttl_days=7
        )
        assert [node.id for node in schema] == [nodes[3].id]
        with pytest.raises(ValueError, match="invalid node level"):
            soft_delete_expired(store, levels=("invalid",), now=NOW)


def test_live_payloads_are_not_needed_for_ttl_eligibility(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.append_trace("live payload " * 1000, {"timestamp": "2026-10-23"})

        def lifecycle_reads(action, table, column, *rest):
            if action == sqlite3.SQLITE_READ and table == "nodes" and column == "content":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        store.connection.set_authorizer(lifecycle_reads)
        try:
            assert soft_delete_expired(store, now=NOW, ttl_days=7) == []
        finally:
            store.connection.set_authorizer(None)


@pytest.mark.parametrize("anchor", ["created", "last_accessed"])
def test_expired_trace_is_selected_among_100000_newer_live_traces(anchor):
    with MemoryStore(":memory:") as store:
        old = store.create_node(
            level="trace", content="expired oldest", node_id="expired-oldest",
            context={"timestamp": "2020-01-01"},
        )
        with store.transaction():
            store.connection.executemany(
                """
                INSERT INTO nodes (id, level, content, timestamp, created_at, updated_at)
                VALUES (?, 'trace', 'fresh', '2026-10-23', '2026-10-23', '2026-10-23')
                """,
                ((f"fresh-{index:06d}",) for index in range(100000)),
            )

        expired = soft_delete_expired(
            store, now=NOW, ttl_days=7, ttl_anchor=anchor,
            decay_concept_covered=False, max_expired=500,
        )
        assert [(node.id, node.decay_reason) for node in expired] == [
            (old.id, "ttl expired")
        ]
        assert store.get_node(old.id).decayed
        assert store.connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 100001
        assert soft_delete_expired(
            store, now=NOW, ttl_days=7, ttl_anchor=anchor,
            decay_concept_covered=False, max_expired=500,
        ) == []


@pytest.mark.parametrize("anchor", ["created", "last_accessed"])
@pytest.mark.parametrize("cap", [0, 2])
def test_retirement_order_uses_parsed_anchor_and_fallback(anchor, cap):
    with MemoryStore(":memory:") as store:
        dates = [
            ("tie-b", "2020-01-01T00:00:00Z", "2026-10-23", None),
            ("older", "2020-01-01T01:00:00+02:00", "2026-10-23", None),
            ("tie-a", "2020-01-01T07:00:00+07:00", "2026-10-23", "invalid"),
            ("fallback", "invalid", "2019-01-01", "2019-01-01"),
            ("read-recent", "2018-01-01", "2026-10-23", "2026-10-23"),
            ("unknown", "invalid", "invalid", None),
        ]
        for node_id, timestamp, created_at, last_accessed in dates:
            store.create_node(
                level="trace", node_id=node_id, content=node_id,
                context={"timestamp": timestamp},
                stats={"last_accessed": last_accessed},
            )
            with store.transaction():
                store.connection.execute(
                    "UPDATE nodes SET created_at=? WHERE id=?", (created_at, node_id)
                )
        expected = ["fallback", "older", "tie-a", "tie-b"]
        if anchor == "created":
            expected.insert(0, "read-recent")
        batches = [expected] if cap == 0 else [expected[i:i + cap] for i in range(0, len(expected), cap)]
        for batch in [*batches, []]:
            expired = soft_delete_expired(
                store, now=NOW, ttl_days=7, ttl_anchor=anchor,
                decay_concept_covered=False, max_expired=cap,
            )
            assert [(node.id, node.decay_reason) for node in expired] == [
                (node_id, "ttl expired") for node_id in batch
            ]
        assert not store.get_node("unknown").decayed


@pytest.mark.parametrize("level", ["concept", "schema"])
@pytest.mark.parametrize("coverage_rule", [False, True])
def test_live_cross_scope_coverage_guards_sources_unless_opted_in(level, coverage_rule):
    with MemoryStore(":memory:") as store:
        for node_id, scope in [("covered", "project:decay"), ("other", "project:other")]:
            store.create_node(
                level="trace", node_id=node_id, content=node_id,
                context={"scope": scope, "timestamp": "2020-01-01"},
            )
        cover = store.create_node(
            level=level, content="live digest in another scope",
            context={"scope": "project:digest", "timestamp": "2026-10-20"},
            provenance={"source_traces": ["covered"], "source_coverage": {"covered": "2026-10-20"}},
        )
        expired = soft_delete_expired(
            store, scope="project:decay", now=NOW, ttl_days=7,
            ttl_anchor="created", decay_concept_covered=coverage_rule, max_expired=1,
        )
        assert [(node.id, node.decay_reason) for node in expired] == (
            [("covered", "concept covered, unread since coverage")] if coverage_rule else []
        )
        assert not store.get_node("other").decayed
        assert not store.get_node(cover.id).decayed
        if not coverage_rule:
            store.soft_delete_node(cover.id, "manual forget")
            expired = soft_delete_expired(
                store, scope="project:decay", now=NOW, ttl_days=7,
                ttl_anchor="created", decay_concept_covered=False, max_expired=1,
            )
            assert [(node.id, node.decay_reason) for node in expired] == [
                ("covered", "ttl expired")
            ]
