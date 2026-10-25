"""Accepted trace writes set cadence; active traces supply consolidation material."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from living_memory.decay import apply_decay
from living_memory.server import _auto_consolidation_due, create_mcp_server
from living_memory.storage import MemoryStore
from test_lookup import FakeMCP


SCOPE = "project:cadence"
CONTEXT = {"scope": SCOPE, "agent": "fixture", "transport_session_id": "cadence-fixture"}
TOPIC = "Harbor crane maintenance requires one recorded calibration check item"


@pytest.fixture
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "fixed")
    monkeypatch.delenv("LM_CONSOLIDATE_MERGE_FLOOR", raising=False)
    instance = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    yield instance
    instance.auto_consolidation.close()
    instance.memory_store.close()


def seed(store, count):
    return [store.append_trace(f"{TOPIC} {index}", CONTEXT) for index in range(count)]


def remember(mcp, content):
    result = mcp.tools["memory_remember"](content, CONTEXT)
    assert mcp.auto_consolidation.wait_idle(timeout=30)
    return result


@pytest.mark.parametrize("policy,size,interval", [
    ("fixed", 100, 100), ("fixed", 101, 100),
    ("adaptive", 5, 5), ("adaptive", 6, 5),
])
def test_replacements_wait_for_next_write_interval(mcp, monkeypatch, policy, size, interval):
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", policy)
    store = mcp.memory_store
    original = seed(store, size)[0]
    for _ in range(3):
        assert remember(mcp, original.content)["auto_consolidation"] is None
    assert mcp.auto_consolidation.status()["passes_completed"] == 0
    assert store.trace_count(include_decayed=False) == size
    assert store.get_node(original.id).decayed

    boundary = (size // interval + 1) * interval
    latest = None
    for count in range(size + 4, boundary + interval + 1):
        # Both new content and supported same-content replacements advance it.
        content = original.content if count % 2 else f"{TOPIC} new {count}"
        latest = remember(mcp, content)
        assert bool(latest["auto_consolidation"]) == (count % interval == 0)
    assert mcp.auto_consolidation.status()["passes_completed"] == 2
    summary = mcp.auto_consolidation.last_result(SCOPE)["summary"]
    materialized = summary["concepts_created"] + summary["concepts_updated"]
    assert len(materialized) == 1
    concept = store.get_node(materialized[0])
    active = store.list_nodes(level="trace", scope=SCOPE, limit=1000)
    assert set(concept.source_traces) == {node.id for node in active}
    assert original.id not in concept.source_traces
    assert latest["node"]["id"] in concept.source_traces


@pytest.mark.parametrize("policy,interval", [("fixed", 100), ("adaptive", 5)])
@pytest.mark.parametrize("retire", ["forget", "decay"])
def test_retiring_traces_cannot_rearm_a_completed_interval(mcp, monkeypatch, policy, interval, retire):
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", policy)
    store = mcp.memory_store
    seeded = seed(store, interval - 1)
    assert remember(mcp, f"{TOPIC} first boundary")["auto_consolidation"]
    assert mcp.auto_consolidation.status()["passes_completed"] == 1
    if retire == "forget":
        mcp.tools["memory_forget"](seeded[0].id)
    else:
        result = apply_decay(
            store, scope=SCOPE, now=datetime.now(UTC) + timedelta(days=181),
            ttl_days=180, ttl_anchor="created", max_expired=1,
            include_superseded=False, decay_concept_covered=True,
        )
        assert len(result.expired) == 1
    assert store.trace_count() == interval
    assert store.trace_count(include_decayed=False) == interval - 1
    assert remember(mcp, f"{TOPIC} after retirement")["auto_consolidation"] is None
    assert store.trace_count(include_decayed=False) == interval
    assert mcp.auto_consolidation.status()["passes_completed"] == 1


@pytest.mark.parametrize("floor,concept_count", [("adaptive", 1), ("fixed", 0)])
def test_merge_floor_uses_active_material_at_a_write_boundary(mcp, monkeypatch, floor, concept_count):
    monkeypatch.setenv("LM_CONSOLIDATE_MERGE_FLOOR", floor)
    store = mcp.memory_store
    seeded = seed(store, 99)
    for node in seeded[12:]:
        store.soft_delete_node(node.id)
    response = remember(mcp, f"{TOPIC} first boundary")
    assert response["auto_consolidation"]["trace_count"] == 13
    summary = mcp.auto_consolidation.last_result(SCOPE)["summary"]
    assert summary["traces_considered"] == 13
    assert len(summary["concepts_created"]) == concept_count
    if concept_count:
        concept = store.get_node(summary["concepts_created"][0])
        assert set(concept.source_traces) == {node.id for node in seeded[:12]} | {response["node"]["id"]}


def test_adaptive_step_follows_writes_after_population_shrinks(mcp, monkeypatch):
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "adaptive")
    store = mcp.memory_store
    seeded = seed(store, 99)
    for node in seeded[5:]:
        store.soft_delete_node(node.id)
    scheduled = []
    for count in range(100, 126):
        result = remember(mcp, seeded[0].content)
        if result["auto_consolidation"]:
            scheduled.append(count)
    assert scheduled == [100, 125]
    assert store.trace_count(include_decayed=False) == 5
    assert mcp.auto_consolidation.status()["passes_completed"] == 2
    assert mcp.auto_consolidation.last_result(SCOPE)["summary"]["concepts_updated"]


@pytest.mark.parametrize("policy,seed_count", [("fixed", 99), ("adaptive", 4)])
def test_alternatives_crossing_an_interval_schedule_one_pass(mcp, monkeypatch, policy, seed_count):
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", policy)
    store = mcp.memory_store
    seed(store, seed_count)
    result = mcp.tools["memory_remember"](
        f"{TOPIC} boundary", CONTEXT,
        alternatives_considered=[
            {"approach": "Skip the check", "rejected_because": "Calibration must be recorded"},
            {"approach": "Use an old reading", "rejected_because": "The crane configuration changed"},
        ],
    )
    assert result["auto_consolidation"]["status"] == "scheduled"
    assert len(result["rejected_alternatives"]) == 2
    assert store.trace_count() == seed_count + 3
    assert mcp.auto_consolidation.wait_idle(timeout=30)
    assert mcp.auto_consolidation.status()["passes_completed"] == 1
    assert mcp.auto_consolidation.last_result(SCOPE)["summary"]["concepts_created"]


def test_reopen_indexes_retained_writes_without_new_progress_state(tmp_path, monkeypatch):
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "fixed")
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store:
        for _ in range(99):
            store.append_trace(TOPIC, CONTEXT)
        # Exercise opening a database with the existing active-only indexes.
        store.connection.execute("DROP INDEX idx_nodes_trace_scope")
    with MemoryStore(path) as store:
        store.append_trace(TOPIC, CONTEXT)
        assert _auto_consolidation_due(store, SCOPE) == 1
        assert _auto_consolidation_due(store, "project:other") is None
        plan = [row[3] for row in store.connection.execute(
            "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM nodes WHERE level='trace' AND scope=?", (SCOPE,)
        )]
        assert any("COVERING INDEX idx_nodes_trace_scope" in detail for detail in plan)
        assert not any("SCAN nodes" in detail for detail in plan)
        store.append_trace(TOPIC, CONTEXT)
        assert _auto_consolidation_due(store, SCOPE) is None


@pytest.mark.parametrize("policy,interval", [("fixed", 100), ("adaptive", 5)])
def test_teach_replacements_share_the_remember_interval(mcp, monkeypatch, policy, interval):
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", policy)
    store = mcp.memory_store
    seeded = seed(store, interval - 1)
    original = seeded[1]
    corrections = []
    for count in range(interval, 2 * interval + 1):
        taught = mcp.tools["memory_teach"](original.id, seeded[0].content, context=CONTEXT)
        corrections.append(taught["corrective_trace"]["id"])
        assert bool(taught["auto_consolidation"]) == (count % interval == 0)
        assert mcp.auto_consolidation.wait_idle(timeout=30)
        assert store.list_connections(
            source_id=corrections[-1], target_id=original.id, relation_type="supersedes",
        )
    assert mcp.auto_consolidation.status()["passes_completed"] == 2
    summary = mcp.auto_consolidation.last_result(SCOPE)["summary"]
    assert summary["concepts_updated"]
    concept = store.get_node(summary["concepts_updated"][0])
    assert corrections[-1] in concept.source_traces
    assert not set(corrections[:-1]) & set(concept.source_traces)
    assert original.id not in concept.source_traces
    assert store.get_node(original.id).corrections[-1]["new"] == seeded[0].content


def test_fastmcp_write_tools_keep_cadence_and_correction_links(tmp_path, monkeypatch):
    from fastmcp import Client

    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "fixed")
    monkeypatch.delenv("LM_CONSOLIDATE_MERGE_FLOOR", raising=False)
    mcp = create_mcp_server(tmp_path / "transport.sqlite3")
    store = mcp.memory_store
    seeded = seed(store, 99)

    async def scenario():
        async with Client(mcp) as client:
            response = await client.call_tool("memory_remember", {
                "content": seeded[0].content, "context": CONTEXT,
            })
            assert response.structured_content["auto_consolidation"]["trace_count"] == 99
            assert mcp.auto_consolidation.wait_idle(timeout=30)
            assert mcp.auto_consolidation.last_result(SCOPE)["summary"]["concepts_created"]
            for _ in range(3):
                response = await client.call_tool("memory_teach", {
                    "trace_id": seeded[1].id, "correction": seeded[0].content, "context": CONTEXT,
                })
                taught = response.structured_content
                assert taught["auto_consolidation"] is None
                trace_id = taught["corrective_trace"]["id"]
                lookup = await client.call_tool("memory_lookup", {"node_id": trace_id})
                node = lookup.structured_content["results"][0]
                assert node["content"] == seeded[0].content and not node["decayed"]
                assert store.list_connections(
                    source_id=trace_id, target_id=seeded[1].id, relation_type="supersedes",
                )
            assert store.trace_count() == 103
            assert mcp.auto_consolidation.status()["passes_completed"] == 1

    try:
        asyncio.run(scenario())
    finally:
        mcp.auto_consolidation.close()
        store.close()
