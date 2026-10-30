"""Native health staleness contract and lazy SQLite cursor lifetime."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from datetime import UTC, datetime

import pytest
from fastmcp import Client

from living_memory import resources
from living_memory.config import MemoryConfig
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore


@pytest.fixture
def frozen_clock(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 29, 12, tzinfo=UTC)

    monkeypatch.setattr(resources, "datetime", Clock)
    return Clock


@pytest.mark.parametrize("timezone,naive_age", [("UTC", 2 * 86400), ("Etc/GMT-7", 2 * 86400 + 7 * 3600)])
@pytest.mark.parametrize("reverse_scan", [False, True])
def test_native_health_staleness_matches_frozen_oracle(
    tmp_path, monkeypatch, frozen_clock, timezone, naive_age, reverse_scan,
):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    with monkeypatch.context() as clock_env:
        clock_env.setenv("TZ", timezone)
        time.tzset()
        try:
            _check_native_health(tmp_path, monkeypatch, naive_age, reverse_scan)
        finally:
            clock_env.undo()
            time.tzset()


def _check_native_health(tmp_path, monkeypatch, naive_age, reverse_scan):
    server = create_mcp_server(
        config=MemoryConfig(db_path=tmp_path / "native.sqlite3"), expose_operator_tools=True,
    )
    store = server.memory_store
    old = "2026-09-29T12:00:00Z"
    # Ages are fixture facts, independent of the production parser and selection.
    fixtures = [
        ("z-tie", old, None, 30 * 86400, "project:alpha", "trace", False),
        ("a-tie", old, "", 30 * 86400, "project:alpha", "trace", False),
        ("never", "2026-10-19T12:00:00Z", None, 10 * 86400, "project:alpha", "trace", False),
        ("naive", old, "2026-10-27T12:00:00", naive_age, "project:alpha", "trace", False),
        ("offset", "invalid", "2026-10-28T17:00:00+05:00", 86400, "project:alpha", "trace", False),
        ("future", old, "2026-10-30T12:00:00Z", 0, "project:alpha", "trace", False),
        ("invalid-access", old, "invalid", None, "project:alpha", "trace", False),
        ("blank-access", old, " ", None, "project:alpha", "trace", False),
        ("invalid-created", "invalid", None, None, "project:alpha", "trace", False),
        ("missing", "", None, None, "project:alpha", "trace", False),
        ("foreign", old, None, 30 * 86400, "project:beta", "trace", False),
        ("global", old, None, 30 * 86400, "global", "trace", False),
        ("concept", old, None, 30 * 86400, "project:alpha", "concept", False),
        ("decayed", old, None, 30 * 86400, "project:alpha", "trace", True),
    ]
    try:
        for node_id, created, accessed, age, scope, level, decayed in fixtures:
            store.create_node(node_id=node_id, level=level, content=node_id * 130,
                              context={"scope": scope})
            with store.transaction():
                store.connection.execute(
                    "UPDATE nodes SET timestamp=?, last_accessed=?, access_count=3, decayed=? WHERE id=?",
                    (created, accessed, decayed, node_id),
                )
        conn = store.connection
        conn.execute(f"PRAGMA reverse_unordered_selects = {int(reverse_scan)}")
        before = list(conn.iterdump())
        changes = conn.total_changes
        conn.execute("PRAGMA query_only = ON")
        hydration = []

        class Connection:
            def __getattr__(self, name):
                return getattr(conn, name)

            def execute(self, sql, params=()):
                if "JOIN nodes ON nodes.id = selected.value" in sql:
                    hydration.append(json.loads(params[0]))
                return conn.execute(sql, params)

        monkeypatch.setattr(MemoryStore, "connection", property(lambda self: Connection()))

        async def check():
            async with Client(server) as client:
                for scope in ("project:alpha", None, "global", "project:empty"):
                    clause = " AND scope = ?" if scope else ""
                    # SQLite's actual input order is the tie contract, not ID order.
                    order = [row[0] for row in conn.execute(
                        "SELECT id, timestamp, last_accessed FROM nodes WHERE level = 'trace' AND decayed = 0" + clause,
                        [scope] if scope else [],
                    )]
                    facts = {row[0]: row for row in fixtures}
                    eligible = [node_id for node_id in order if facts[node_id][3] is not None]
                    ranked = sorted(eligible, key=lambda node_id: -facts[node_id][3])
                    ages = sorted(facts[node_id][3] for node_id in eligible)
                    expected_p50 = ages[round((len(ages) - 1) / 2)] if ages else None
                    expected_p90 = ages[round(9 * (len(ages) - 1) / 10)] if ages else None
                    for limit in (0, 1, 2, 5, 6, 100):
                        hydration.clear()
                        response = await client.call_tool("memory_health", {"scope": scope, "top_stale": limit})
                        assert not response.is_error
                        stale = response.data["staleness"]
                        selected = ranked[:limit]
                        assert stale["age_seconds_p50"] == expected_p50
                        assert stale["age_seconds_p90"] == expected_p90
                        assert stale["top_stale"] == [
                            {"id": node_id, "content_preview": (node_id * 130)[:120],
                             "timestamp": facts[node_id][1], "access_count": 3,
                             "last_accessed": facts[node_id][2] or None}
                            for node_id in selected
                        ]
                        assert hydration == ([selected] if selected else [])
                error = await client.call_tool("memory_health", {"top_stale": -1}, raise_on_error=False)
                assert error.is_error
                assert "top_stale must be >= 0" in error.content[0].text

        asyncio.run(check())
        assert conn.total_changes == changes
        assert list(conn.iterdump()) == before
    finally:
        store.close()


@pytest.mark.parametrize("limit", [0, 2])
@pytest.mark.parametrize("failure", [None, "iteration", "parse", "timestamp"])
def test_health_date_cursor_closes_on_success_and_failure(
    tmp_path, monkeypatch, frozen_clock, limit, failure,
):
    with MemoryStore(tmp_path / "cursor.sqlite3") as store:
        for i in range(3):
            store.create_node(node_id=str(i), level="trace", content=f"trace {i}")
        conn = store.connection
        cursors = []
        reads = []

        class Cursor:
            def __init__(self, cursor):
                self.cursor = cursor
                cursors.append(cursor)

            def __iter__(self):
                return self

            def __next__(self):
                if failure == "iteration" and reads:
                    raise RuntimeError("date iteration failed")
                row = next(self.cursor)
                reads.append(row["id"])
                return row

            def fetchall(self):
                pytest.fail("date rows must be consumed lazily")

            def close(self):
                self.cursor.close()

        class Connection:
            def __getattr__(self, name):
                return getattr(conn, name)

            def execute(self, sql, params=()):
                cursor = conn.execute(sql, params)
                return Cursor(cursor) if "SELECT id, timestamp, last_accessed" in sql else cursor

        monkeypatch.setattr(MemoryStore, "connection", property(lambda self: Connection()))
        if limit == 0:
            monkeypatch.setattr(resources.heapq, "nlargest", lambda *a, **kw: pytest.fail("zero must not rank IDs"))
        if failure in {"parse", "timestamp"}:
            class BrokenDate:
                def timestamp(self):
                    raise ValueError("timestamp failed")

            def parse(value):
                if failure == "parse":
                    raise OverflowError("parse failed")
                return BrokenDate()

            monkeypatch.setattr(frozen_clock, "fromisoformat", parse)
        before = list(conn.iterdump())
        conn.execute("PRAGMA query_only = ON")
        if failure:
            error = {"iteration": RuntimeError, "parse": OverflowError, "timestamp": ValueError}[failure]
            with pytest.raises(error, match="failed"):
                resources.memory_health(store, top_stale=limit)
        else:
            resources.memory_health(store, top_stale=limit)
            assert len(reads) == 3
        assert len(cursors) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed cursor"):
            cursors[0].fetchone()
        assert list(conn.iterdump()) == before
