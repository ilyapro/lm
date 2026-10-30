"""Optional map failures preserve the required recall and its feedback."""

import asyncio
from contextlib import contextmanager
import json
import os
import sqlite3
import sys

import pytest

from living_memory import storage
from living_memory.recall_map import RecallMap, RecallMapBuilder
from living_memory.score_gate import QueryRelevance
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore
from test_transport_identity import FakeMCP


MAP_SCOPE = "project:mapdelivery"
MAP_QUERY = "deployment failure database migration rollback"
MAP_FAULTS = {
    "update": "BEFORE UPDATE OF recall_map ON recall_events",
    "attribution": """BEFORE INSERT ON recall_delivery_history
        WHEN (SELECT recall_map FROM recall_events
              WHERE id=NEW.delivery_event_id) IS NOT NULL""",
}


def _seed(store):
    for index in range(40):
        store.append_trace(
            f"quarterly ledger reconciliation entry {index}",
            {"scope": "project:mapdelivery-elsewhere"},
        )
    for procedure, count in (("canary-window", 6), ("checkpoint-drill", 4), ("throttle-runbook", 3)):
        for index in range(count):
            store.append_trace(
                f"deployment failure case {procedure} {index}: database "
                f"migration rollback step {index}",
                {"scope": MAP_SCOPE, "procedure_id": procedure},
            )


def reject_map(connection, stage):
    connection.execute(f"""CREATE TRIGGER reject_map {MAP_FAULTS[stage]}
        BEGIN SELECT RAISE(ABORT, 'optional map unavailable'); END""")


@pytest.fixture(autouse=True)
def knobs(monkeypatch):
    for name in (
        "LM_MAP_CURTAIL_DECAY", "LM_MAP_POOL_DEMOTION_GATE", "LM_MAP_POOL_DEMOTE_AFTER",
        "LM_MAP_POOL_USEFULNESS_GATE", "LM_MAP_POOL_MIN_USEFULNESS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_RECALL_MAP", "1")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
    monkeypatch.setenv("LM_DELIVERY_SESSION_DEDUP", "1")


@pytest.mark.parametrize("stage", ["qualification", "build", "serialization", *MAP_FAULTS])
@pytest.mark.parametrize("view", ["knowledge", "full"])
@pytest.mark.parametrize("policy", ["audit", "credit"])
def test_map_failure_preserves_selected_response_and_feedback(
    tmp_path, monkeypatch, caplog, stage, view, policy,
):
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", policy)
    now = storage._utc_now()
    monkeypatch.setattr(storage, "_utc_now", lambda: now)
    ambient = {"transport_session_id": "map-failure", "agent": "fixture"}
    seed = tmp_path / "seed.sqlite3"
    with MemoryStore(seed) as store:
        _seed(store)
        marked = store.append_trace("prior ledger reading", {"scope": "project:prior"})
        store.record_recall_event(
            query="ledger", scope=marked.scope, ambient_context=ambient,
            results=[{"node_id": marked.id}],
        )
        for name in ("baseline", "failure"):
            with sqlite3.connect(tmp_path / f"{name}.sqlite3") as twin:
                store.connection.backup(twin)

    def recall(mcp):
        return mcp.tools["memory_recall"](
            MAP_QUERY, scope=MAP_SCOPE, max_results=3, view=view,
            ambient_context=ambient, used=[marked.id],
        )

    baseline = create_mcp_server(tmp_path / "baseline.sqlite3", mcp_factory=FakeMCP)
    failed = create_mcp_server(tmp_path / "failure.sqlite3", mcp_factory=FakeMCP)
    try:
        expected = recall(baseline)
        assert expected["recall_map"]["clusters"]
        if stage in MAP_FAULTS:
            reject_map(failed.memory_store.connection, stage)
        else:
            def unavailable(*args, **kwargs):
                raise ValueError("optional map unavailable")

            target, method = {
                "qualification": (QueryRelevance, "eligible_ids"),
                "build": (RecallMapBuilder, "build"),
                "serialization": (RecallMap, "to_dict"),
            }[stage]
            monkeypatch.setattr(target, method, unavailable)

        actual = recall(failed)
        assert actual["count"] == 3
        assert all(
            item["delivery"] != "session_duplicate" and item.get("node", item)["content"]
            for item in actual["results"]
        )
        assert "recall_map" not in actual
        assert {k: v for k, v in actual.items() if k != "recall_event_id"} == {
            k: v for k, v in expected.items() if k not in ("recall_event_id", "recall_map")
        }
        assert actual["feedback_marks"] == {"accepted": 1, "dropped": 0, "by_reason": {}}
        assert any(
            record.name == "living_memory.server"
            and "optional recall map" in record.message and record.exc_info
            for record in caplog.records
        )
        store = failed.memory_store
        assert not store.connection.in_transaction
        # A separate reader sees the required event and marks, without map attribution.
        with sqlite3.connect(store.db_path) as reader:
            event = reader.execute(
                "SELECT results, recall_map FROM recall_events WHERE id=?",
                (actual["recall_event_id"],),
            ).fetchone()
            ids = [item.get("node", item)["id"] for item in actual["results"]]
            assert [item["node_id"] for item in json.loads(event[0])] == ids
            assert event[1] is None
            # The three head results are outside the organic tail ledger.
            assert reader.execute(
                "SELECT node_id FROM recall_delivery_history WHERE delivery_event_id=?",
                (actual["recall_event_id"],),
            ).fetchall() == []
            assert reader.execute(
                "SELECT node_id, accepted, source_id FROM recall_feedback_marks"
            ).fetchall() == [(marked.id, 1, actual["recall_event_id"])]
        assert store.recent_recall_map_history() == []
    finally:
        for mcp in (baseline, failed):
            mcp.auto_consolidation.close()
            mcp.memory_store.close()


@pytest.mark.parametrize("stage", ["validation", "retrieval", "access", "event"])
def test_required_failures_remain_errors(tmp_path, stage):
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    try:
        _seed(store)
        kwargs = {}
        if stage == "validation":
            kwargs["view"] = "invalid"
        elif stage == "retrieval":
            store.connection.set_authorizer(
                lambda action, table, *rest: sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_READ and table == "nodes" else sqlite3.SQLITE_OK
            )
        else:
            clause = ("BEFORE UPDATE OF access_count ON nodes" if stage == "access"
                      else "BEFORE INSERT ON recall_events")
            store.connection.execute(f"""CREATE TRIGGER reject_required {clause}
                BEGIN SELECT RAISE(ABORT, 'required recall unavailable'); END""")
        with pytest.raises((ValueError, sqlite3.DatabaseError)):
            mcp.tools["memory_recall"](MAP_QUERY, scope=MAP_SCOPE, max_results=3, **kwargs)
        store.connection.set_authorizer(None)
        assert store.list_recall_events() == []
        assert not store.connection.in_transaction
    finally:
        store.connection.set_authorizer(None)
        mcp.auto_consolidation.close()
        store.close()


@pytest.mark.parametrize("failure", [None, "read", "reduce", "close"])
@pytest.mark.parametrize("view", ["knowledge", "full"])
def test_native_embedding_read_finishes_before_map_writes(tmp_path, monkeypatch, failure, view):
    from fastmcp import Client

    monkeypatch.setenv("LM_DELIVERY_SESSION_DEDUP", "0")
    mcp = create_mcp_server(tmp_path / "embedding.sqlite3")
    store = mcp.memory_store
    for index in range(32):
        store.create_node(level="trace", content=f"quarterly ledger reconciliation {index}")
    for index in range(12):
        topic = "orbital telemetry calibration" if index % 2 else "orchard harvest rotation"
        store.create_node(
            node_id=f"vector-{index:02}", level="trace", content=f"{MAP_QUERY} {topic} {index}",
            context={"scope": MAP_SCOPE},
            embedding=[float(axis == index % 2) for axis in range(384)],
        )
    original_stream = store.iter_node_chunk_vectors
    original_reduce = RecallMapBuilder._node_vector
    original_build = RecallMapBuilder.build
    completed, reductions, hits = [], [], []
    active = False

    @contextmanager
    def stream(ids):
        with original_stream(ids) as groups:
            def rows():
                for index, group in enumerate(groups):
                    if active and failure == "read" and index == 1:
                        raise sqlite3.OperationalError("late embedding read")
                    yield group
            yield rows()
        completed.append(True)
        if active and failure == "close":
            raise sqlite3.OperationalError("late embedding close")

    def reduce(builder, node, chunks):
        reductions.append(node.id)
        if active and failure == "reduce" and len(reductions) == 2:
            raise ValueError("late embedding reducer")
        vector = original_reduce(builder, node, chunks)
        assert vector == [float(axis == int(node.id[-2:]) % 2) for axis in range(384)]
        return vector

    def build(builder, *args, **kwargs):
        result = original_build(builder, *args, **kwargs)
        assert completed  # No usable build precedes stream exhaustion and exit.
        hits.append(builder.last_cache_hit)
        return result

    monkeypatch.setattr(store, "iter_node_chunk_vectors", stream)
    monkeypatch.setattr(RecallMapBuilder, "_node_vector", reduce)
    monkeypatch.setattr(RecallMapBuilder, "build", build)

    async def scenario():
        nonlocal active
        arguments = {"query": MAP_QUERY, "scope": MAP_SCOPE, "max_results": 3,
                     "view": view, "ambient_context": {"task": "embedding survey"}}
        async with Client(mcp) as client:
            first = (await client.call_tool("memory_recall", arguments)).structured_content
            assert first["recall_map"]["clusters"] and hits == [False]
            assert reductions and len(completed) == 1
            reductions.clear()
            completed.clear()
            active = True
            second = (await client.call_tool("memory_recall", arguments)).structured_content
            assert second["count"] == first["count"] == 3
            assert reductions
            with sqlite3.connect(store.db_path) as reader:
                recorded, = reader.execute("SELECT recall_map FROM recall_events WHERE id=?",
                                           (second["recall_event_id"],)).fetchone()
                if failure:
                    assert "recall_map" not in second and recorded is None
                    assert hits == [False]
                else:
                    assert hits == [False, True] and len(completed) == 1
                    stored = json.loads(recorded)
                    stored.pop("sel")
                    assert stored == second["recall_map"] == first["recall_map"]
            assert not store.connection.in_transaction

    try:
        asyncio.run(scenario())
    finally:
        mcp.auto_consolidation.close()
        store.close()


def test_stdio_returns_knowledge_on_map_failure_and_preserves_required_errors(tmp_path):
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    path = tmp_path / "stdio.sqlite3"
    with MemoryStore(path) as store:
        _seed(store)
        reject_map(store.connection, "attribution")
    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "living_memory.server", "--db", str(path), "--transport", "stdio"],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )

    async def scenario():
        async with Client(transport) as client:
            arguments = {"query": MAP_QUERY, "scope": MAP_SCOPE, "max_results": 3}
            first = await client.call_tool("memory_recall", arguments)
            response = first.structured_content
            assert not first.is_error and response["count"] == 3
            assert "recall_map" not in response
            assert all(
                item["delivery"] != "session_duplicate" and item["content"]
                for item in response["results"]
            )
            with sqlite3.connect(path) as connection:
                event_id = response["recall_event_id"]
                results, stamp, recall_map = connection.execute(
                    "SELECT results, transport_session_id, recall_map FROM recall_events WHERE id=?",
                    (event_id,),
                ).fetchone()
                assert stamp and recall_map is None
                assert [item["node_id"] for item in json.loads(results)] == [
                    item["id"] for item in response["results"]
                ]
                connection.execute("DROP TRIGGER reject_map")
            following = await client.call_tool("memory_recall", {
                **arguments, "used": [response["results"][0]["id"]],
            })
            success = following.structured_content
            assert not following.is_error and success["recall_map"]["clusters"]
            assert success["feedback_marks"]["accepted"] == 1
            with sqlite3.connect(path) as connection:
                recorded, = connection.execute(
                    "SELECT recall_map FROM recall_events WHERE id=?", (success["recall_event_id"],),
                ).fetchone()
                stored_map = json.loads(recorded)
                assert stored_map.pop("sel")["e"] == success["recall_map"]["pool"]
                assert stored_map == success["recall_map"]
                connection.execute("""CREATE TRIGGER reject_event BEFORE INSERT ON recall_events
                    BEGIN SELECT RAISE(ABORT, 'required recall unavailable'); END""")
            failed = await client.call_tool("memory_recall", arguments, raise_on_error=False)
            assert failed.is_error
            invalid = await client.call_tool("memory_recall", {**arguments, "view": "invalid"}, raise_on_error=False)
            assert invalid.is_error
            with sqlite3.connect(path) as connection:
                assert connection.execute("SELECT COUNT(*) FROM recall_events").fetchone() == (2,)

    asyncio.run(scenario())
