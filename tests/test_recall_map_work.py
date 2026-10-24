"""Map continuity and consumption belong to the actual current-work consumer."""
from __future__ import annotations

import asyncio
import sqlite3

import pytest
from fastmcp import Client

from living_memory import storage as storage_module
from living_memory.recall_map import RecallMapBuilder, cache_key
from living_memory.server import create_mcp_server
from living_memory.storage import MaturedRecallHistory, MemoryStore, RecallWork

from test_recall_map_curtail import OPERATOR_VALVES, cluster, offered, residual
from test_transport_identity import _structured

SCOPE = "project:current-work"
NOTES = {
    "amber": (
        ("git publish", "Inspect the merge result in the owned checkout."),
        ("git publish", "Resolve overlapping source changes with their current owner."),
        ("release source", "Publish selected public files without project data."),
        ("release source", "Keep local history and credentials outside a public patch."),
    ),
    "cobalt": (
        ("sqlite backup", "The SQLite online backup API yields a consistent live snapshot."),
        ("sqlite backup", "Verify downloaded backup integrity before replacing the prior object."),
        ("dns authority", "Compare authoritative wildcard and apex answers before repeating a DNS write."),
        ("dns authority", "A public DNS read does not establish authoritative server agreement."),
    ),
}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for name in (*OPERATOR_VALVES, "LM_RECALL_MAP", "LM_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setenv("LM_DELIVERY_SESSION_DEDUP", "0")
    monkeypatch.setattr(MemoryStore, "matured_recall_history", lambda self, ids, at: {
        node_id: MaturedRecallHistory.known(100, 50, 50) for node_id in ids
    })
    clock = ["2026-01-01T00:00:00Z"]
    monkeypatch.setattr(storage_module, "_utc_now", lambda: clock[0])
    return clock


def seed(store):
    for index in range(40):
        store.append_trace(f"quarterly ledger reconciliation entry {index}",
                           {"scope": "project:other"})
    for query, notes in NOTES.items():
        for procedure, content in notes:
            store.append_trace(f"{query} {content}",
                               {"scope": SCOPE, "procedure_id": procedure})


async def recall(client, query="cobalt", **kwargs):
    return _structured(await client.call_tool(
        "memory_recall", {"query": query, "scope": SCOPE, "max_results": 1, **kwargs},
    ))


def labels(response):
    return {c["label"] for c in response["recall_map"]["clusters"]}


def test_two_real_clients_preserve_independent_residual_and_own_cadence(tmp_path, isolated):
    async def run():
        mcp = create_mcp_server(tmp_path / "work.sqlite3")
        seed(mcp.memory_store)
        baseline_path = tmp_path / "baseline.sqlite3"
        with sqlite3.connect(baseline_path) as destination:
            mcp.memory_store.connection.backup(destination)
        baseline = create_mcp_server(baseline_path)
        async with Client(baseline) as b:
            before = await recall(b)
        async with Client(mcp) as a:
            for minute in (1, 2, 3):
                isolated[0] = f"2026-01-01T00:0{minute}:00Z"
                a_response = await recall(a, "amber")
                assert labels(a_response) == {"git publish", "release source"}
            async with Client(mcp) as b:
                isolated[0] = "2026-01-01T00:04:00Z"
                after = await recall(b)
                event = mcp.memory_store.get_recall_event(after["recall_event_id"])
                assert event.transport_session_id
                assert "task" not in event.ambient_context
                assert "task_pattern" not in event.ambient_context
            isolated[0] = "2026-01-01T00:05:00Z"
            own = await recall(a, "amber")
        assert labels(before) == labels(after) == {"sqlite backup", "dns authority"}
        assert before["recall_map"] == after["recall_map"]
        assert before["results"] == after["results"]
        assert after["recall_map"]["covered"] == 3
        assert own["recall_map"]["curtailed"] and own["recall_map"]["streak"] == 3
        assert own["results"] == a_response["results"]
        mcp.memory_store.close()
        baseline.memory_store.close()
    asyncio.run(run())


@pytest.mark.parametrize("field", ["task", "task_pattern"])
def test_explicit_work_continues_on_new_transport_and_cold_server(tmp_path, isolated, field):
    async def run():
        path = tmp_path / "continued.sqlite3"
        mcp = create_mcp_server(path)
        seed(mcp.memory_store)
        for minute in (1, 2, 3):
            async with Client(mcp) as client:
                isolated[0] = f"2026-01-01T00:0{minute}:00Z"
                response = await recall(client, ambient_context={field: "backup"})
                assert labels(response)
        mcp.memory_store.close()
        restarted = create_mcp_server(path)
        async with Client(restarted) as client:
            isolated[0] = "2026-01-01T00:04:00Z"
            continued = await recall(client, ambient_context={field: "backup"})
            other = await recall(client, ambient_context={field: "Backup"})
        assert continued["recall_map"]["curtailed"]
        assert continued["recall_map"]["streak"] == 3
        assert labels(other) == {"sqlite backup", "dns authority"}
        restarted.memory_store.close()
    asyncio.run(run())


def test_no_context_is_not_a_shared_work_and_cache_tracks_the_same_identity(tmp_path):
    with MemoryStore(tmp_path / "cache.sqlite3") as store:
        seed(store)
        pool = residual(store)
        builder = RecallMapBuilder(store)
        for _ in range(5):
            built = builder.build(pool, scope=SCOPE)
            assert built and built.clusters and not built.curtailed
            store.record_recall_event(query="idle", scope=SCOPE, recall_map=built.to_dict())
            assert not builder.last_cache_hit
        assert builder.cache_size == 0 and not builder._curtail_memo
        for transport, hit in (("A", False), ("A", True), ("B", False)):
            assert builder.build(pool, scope=SCOPE, transport_session_id=transport)
            assert builder.last_cache_hit is hit
        assert cache_key(SCOPE, task="work-A") != cache_key(SCOPE, task="work A")
        assert cache_key(SCOPE, task="A") != cache_key(SCOPE, transport_session_id="A")
        assert cache_key(SCOPE, task="A") == cache_key(SCOPE, task_pattern="A", task="turn")


def offer_three(store, clock, **context):
    medoid = store.append_trace("useful backup fact", {"scope": SCOPE}).id
    payload = offered(cluster("sqlite backup", medoid))
    for minute in (1, 2, 3):
        clock[0] = f"2026-01-01T00:0{minute}:00Z"
        store.record_recall_event(query="idle", scope=SCOPE,
                                 ambient_context=context, recall_map=payload)
    clock[0] = "2026-01-01T00:04:00Z"
    return medoid


def verdict(store, **context):
    return RecallMapBuilder(store)._read_curtailment(RecallWork.resolve(SCOPE, **context))


@pytest.mark.parametrize("kind", ["access", "recall", "lookup", "query"])
def test_foreign_use_cannot_change_own_cadence(tmp_path, isolated, kind):
    with MemoryStore(tmp_path / "foreign.sqlite3") as store:
        medoid = offer_three(store, isolated, transport_session_id="A")
        assert verdict(store, transport_session_id="A").collapse
        store.record_recall_event(
            query="sqlite backup" if kind == "query" else "idle", scope=SCOPE,
            ambient_context={"transport_session_id": "B"},
            results=[{"node_id": medoid}] if kind == "recall" else [],
        )
        if kind == "access":
            store.record_access(medoid)
        elif kind == "lookup":
            store.record_lookup_event([medoid], transport_session_id="B")
        assert verdict(store, transport_session_id="A").collapse


@pytest.mark.parametrize("kind", ["recall", "lookup", "query"])
def test_mapless_use_in_continued_explicit_work_reopens_the_map(tmp_path, isolated, kind):
    with MemoryStore(tmp_path / "follow.sqlite3") as store:
        medoid = offer_three(store, isolated, task_pattern="backup", transport_session_id="A")
        # Continuation can name the same work with task alone.
        store.record_recall_event(
            query="sqlite backup" if kind == "query" else "idle", scope=SCOPE,
            ambient_context={"task": "backup", "transport_session_id": "B"},
            results=[{"node_id": medoid}] if kind == "recall" else [],
        )
        if kind == "lookup":
            isolated[0] = "2026-01-01T00:05:00Z"
            store.record_lookup_event([medoid], transport_session_id="B")
        result = verdict(store, task_pattern="backup")
        assert result.offers == 0 and not result.collapse


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("next_task", ["other", None])
def test_lookup_after_work_switch_is_not_attributed_to_earlier_work(tmp_path, isolated, tied, next_task):
    with MemoryStore(tmp_path / "switch.sqlite3") as store:
        medoid = offer_three(store, isolated, task="backup", transport_session_id="A")
        if tied:
            isolated[0] = "2026-01-01T00:03:00Z"
        store.record_recall_event(query="idle", scope=SCOPE,
                                 ambient_context={"task": next_task, "transport_session_id": "A"})
        isolated[0] = "2026-01-01T00:05:00Z"
        store.record_lookup_event([medoid], transport_session_id="A")
        assert verdict(store, task="backup").collapse


def test_pattern_only_history_excludes_unidentified_and_different_labels(tmp_path, isolated):
    with MemoryStore(tmp_path / "pattern.sqlite3") as store:
        offer_three(store, isolated)
        offer_three(store, isolated, task="other")
        offer_three(store, isolated, task="same-turn", task_pattern="another-pattern")
        assert not verdict(store, task_pattern="backup").collapse
        assert not verdict(store, task="same-turn").collapse
        assert verdict(store, task="other").collapse


def test_scopeless_service_keeps_cross_project_recall_and_feedback(tmp_path, isolated):
    async def run():
        mcp = create_mcp_server(tmp_path / "broad.sqlite3")
        seed(mcp.memory_store)
        async with Client(mcp) as client:
            response = _structured(await client.call_tool("memory_recall", {
                "query": "cobalt", "max_results": 1,
                "ambient_context": {"task": "backup"},
            }))
            selected = response["results"][0]
            assert selected["scope"] == SCOPE and selected["content_ref"]["node_id"] == selected["id"]
            event = mcp.memory_store.get_recall_event(response["recall_event_id"])
            assert event.scope == "global"
            assert mcp.memory_store.recent_recall_map_history(work=RecallWork.resolve("global", task="backup"))
            second = _structured(await client.call_tool("memory_recall", {
                "query": "cobalt", "max_results": 1, "used": [selected["id"]],
                "ambient_context": {"task": "backup"},
            }))
            assert second["feedback_marks"]["accepted"] == 1
            lookup = _structured(await client.call_tool("memory_lookup", {"node_id": selected["id"]}))
            assert lookup["results"][0]["id"] == selected["id"]
        async with Client(mcp) as other:
            result = await recall(other, irrelevant=[selected["id"]])
            assert result["feedback_marks"]["dropped"] == 1
        mcp.memory_store.close()
    asyncio.run(run())


def test_foreign_lookup_volume_cannot_disable_own_noise_limit(tmp_path, isolated):
    with MemoryStore(tmp_path / "lookup-bound.sqlite3") as store:
        medoid = offer_three(store, isolated, task="backup", transport_session_id="A")
        assert verdict(store, task="backup").collapse
        # Work can switch even on a connection that previously belonged to A.
        store.record_recall_event(query="idle", scope=SCOPE,
                                 ambient_context={"task": "other", "transport_session_id": "A"})
        isolated[0] = "2026-01-01T00:05:00Z"
        for _ in range(201):
            store.record_lookup_event([medoid], transport_session_id="A")
        assert verdict(store, task="backup").collapse
        assert verdict(store, task="backup").offers == 3
