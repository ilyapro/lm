"""The registered stats resource reports one read-only population observation."""

import asyncio
import json
from threading import RLock

from fastmcp import Client, FastMCP
import pytest

from living_memory.server import _register_resources
from living_memory.storage import MemoryStore


@pytest.mark.parametrize(
    ("case", "level_counts", "scope_counts"),
    [
        ("empty", [(0, 0), (0, 0), (0, 0)], []),
        ("active", [(3, 0), (2, 0), (1, 0)],
         [("global", 1, 1), ("project:alpha", 3, 3), ("project:Бета", 2, 2)]),
        ("mixed", [(1, 2), (2, 0), (1, 0)],
         [("global", 1, 1), ("project:alpha", 3, 2), ("project:Бета", 2, 1)]),
        ("decayed", [(0, 3), (0, 2), (0, 1)],
         [("global", 1, 0), ("project:alpha", 3, 0), ("project:Бета", 2, 0)]),
        ("legacy", [(0, 1), (2, 0), (1, 0), (1, 1)],
         [("global", 1, 1), ("project:alpha", 1, 1), ("project:Бета", 2, 1),
          ("raw scope", 2, 1)]),
    ],
)
def test_registered_stats_population(tmp_path, case, level_counts, scope_counts):
    with MemoryStore(tmp_path / "stats.sqlite3") as store:
        if case != "empty":
            for node_id, scope, level, confidence, decayed in [
                ("promotion", "global", "concept", 0.5, False),
                ("alpha-trace", "project:alpha", "trace", 0.25, False),
                ("alpha-trace-2", "project:alpha", "trace", 0.75, True),
                ("alpha-schema", "project:alpha", "schema", 0.75, False),
                ("beta-trace", "project:Бета", "trace", 0.25, True),
                ("beta-concept", "project:Бета", "concept", 0.5, False),
            ]:
                store.create_node(
                    node_id=node_id, level=level, content=node_id,
                    context={"scope": scope}, stats={"confidence": confidence},
                    provenance={
                        "promoted_from": ["project:Бета", "project:alpha"],
                        "promotion_events": [1, 2],
                    } if node_id == "promotion" else {},
                )
                if case == "decayed" or (case != "active" and decayed):
                    store.soft_delete_node(node_id)
        if case == "legacy":
            store.connection.execute("PRAGMA ignore_check_constraints=ON")
            with store.transaction():
                store.connection.execute(
                    "UPDATE nodes SET level='future-level', scope='raw scope' "
                    "WHERE id IN ('alpha-trace', 'alpha-trace-2')"
                )
        levels = dict(zip(("trace", "concept", "schema", "future-level"), level_counts))
        active = sum(a for a, _ in level_counts)
        decayed = sum(d for _, d in level_counts)
        expected = {
            "uri": "memory://stats",
            "counts": {
                "by_level": {
                    level: {"active": a, "decayed": d, "total": a + d}
                    for level, (a, d) in levels.items()
                },
                "total": active + decayed, "active": active, "decayed": decayed,
            },
            "scopes": [
                {"scope": scope, "count": count, "active": a}
                for scope, count, a in scope_counts
            ],
            "confidence": {
                "count": active, "average": 0.5 if active else 0.0,
                "minimum": 0.25 if active else 0.0, "maximum": 0.75 if active else 0.0,
            },
            "promotions": {
                "promoted_concepts": 1 if active else 0,
                "promotion_events": 2 if active else 0,
                "source_scopes": ["project:alpha", "project:Бета"] if active else [],
            },
            "retrieval_weights": [dict(row) for row in store.connection.execute(
                "SELECT scope, bm25, vector, graph, learning_rate, updated_at "
                "FROM retrieval_weights ORDER BY scope"
            )],
        }
        mcp = FastMCP("stats-test")
        _register_resources(mcp, store, RLock())
        before = list(store.connection.iterdump())
        changes = store.connection.total_changes
        store.connection.execute("PRAGMA query_only=ON")
        queries = []
        store.connection.set_trace_callback(queries.append)

        async def read():
            async with Client(mcp) as client:
                response, = await client.read_resource("memory://stats")
                return json.loads(response.text)

        try:
            assert asyncio.run(read()) == expected
        finally:
            store.connection.set_trace_callback(None)
        population_queries = [sql for sql in queries if "GROUP BY" in sql.upper()]
        assert len(population_queries) == 1
        assert store.connection.total_changes == changes
        assert list(store.connection.iterdump()) == before
