"""Exact-twin replacement uses the recall's current qualification evidence."""

import asyncio
from dataclasses import replace

from fastmcp import Client
import pytest

from living_memory.delivery import DeliveryHistory, shape_recall_results
from living_memory.models import Node
from living_memory.retrieval import MemoryRecallService, RecallResult
from living_memory.score_gate import QueryRelevance, apply_score_gate
from living_memory.server import _select_distinct_results, create_mcp_server


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
        "LM_RECALL_MIN_SCORE": "0.35",
        "LM_RECALL_SCHEMA_TRIGGER": "off",
        "LM_RECALL_MAP": "1",
        "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
        "LM_AUTO_CONSOLIDATE_POLICY": "off",
        "LM_DELIVERY_SESSION_DEDUP": "1",
        "LM_DELIVERY_SNIPPET_LADDER": "off",
    }.items():
        monkeypatch.setenv(key, value)


def row(node_id, content, strength=1.0):
    return RecallResult(Node(id=node_id, level="trace", content=content), 1.0,
                        lexical_score=strength)


@pytest.mark.parametrize("form", ["drop", "stub"])
@pytest.mark.parametrize("contents, strengths, limit, expected", [
    ("AABC", [1, 1, 0, 1], 2, [0, 3]),
    ("AABCD", [1, 1, 0, 0, 1], 2, [0, 1]),
    ("AAABBB", [1, 1, 1, 0, 1, 1], 3, [0, 1, 4]),
    ("AAA", [1, 1, 1], 2, [0, 1]),
    ("AAB", [1, 1, 0], 2, [0, 1]),
    ("AAB", [1, 1, 1], 1, [0]),
])
def test_replacement_qualifies_within_original_bound(
    monkeypatch, form, contents, strengths, limit, expected,
):
    monkeypatch.setenv("LM_RECALL_GATE_FORM", form)
    ranked = [row(str(i), text, strength)
              for i, (text, strength) in enumerate(zip(contents, strengths))]
    evidence = QueryRelevance(ranked)
    limited, residual = apply_score_gate(ranked, limit, relevance=evidence)
    selected, remaining = _select_distinct_results(
        limited, residual, limit, DeliveryHistory(), relevance=evidence,
    )
    assert selected == [ranked[i] for i in expected]
    assert remaining == [r for r in limited + residual if r not in selected]
    assert len(selected) == limit
    assert all(evidence.passes(r) for r in selected)


@pytest.mark.parametrize("limit, expected", [(2, [0, 1]), (3, [0, 1, 3])])
def test_stub_references_cannot_bear_or_release_a_content_slot(monkeypatch, limit, expected):
    monkeypatch.setenv("LM_RECALL_GATE_FORM", "stub")
    ranked = [row("ref", "A", 0), row("content", "A"), row("twin", "A"), row("other", "B")]
    evidence = QueryRelevance(ranked)
    limited, residual = apply_score_gate(ranked, limit, relevance=evidence)
    selected, remaining = _select_distinct_results(
        limited, residual, limit, DeliveryHistory(), relevance=evidence,
    )
    assert [r.node.id for r in selected] == [ranked[i].node.id for i in expected]
    assert selected[0].withheld == "below_threshold"
    assert {r.node.id for r in selected + remaining} == {r.node.id for r in ranked}
    shaped = shape_recall_results(
        selected, view="knowledge", delivery_history=DeliveryHistory(), session_dedup=False,
        snippet_max_chars=1000, context_value_max_chars=160,
    )
    assert shaped[0]["content"] == ""
    assert shaped[0]["content_ref"]["node_id"] == "ref"
    assert [entry["delivery"] for entry in shaped] == ["below_threshold"] + ["full"] * (limit - 1)


def test_replacement_preserves_query_demotions_and_inherited_correction_relevance():
    first, twin = row("first", "A"), row("twin", "A")
    demoted = row("demoted", "B")
    correction = row("correction", "New rule", 0)
    original = replace(row("original", "Old rule"), superseded=True)
    ranked = [first, twin, demoted, correction, original]
    evidence = QueryRelevance(
        ranked, threshold=0.05, demotions={"demoted": 0.01},
        corrections_by_superseded={"original": ["correction"]},
    )
    assert not evidence._passes(correction) and evidence.passes(correction)
    limited, residual = apply_score_gate(ranked, 2, relevance=evidence)
    selected, remaining = _select_distinct_results(
        limited, residual, 2, DeliveryHistory(), relevance=evidence,
    )
    assert selected == [first, correction]
    assert remaining == [twin, demoted, original]


def test_qualification_stops_at_the_selected_cut(monkeypatch):
    ranked = [row("first", "A"), row("twin", "A"), row("other", "B"), row("tail", "C")]
    evidence = QueryRelevance(ranked)
    passes = evidence.passes
    inspected = []

    def observe(result):
        inspected.append(result.node.id)
        assert result.node.id != "tail"
        return passes(result)

    monkeypatch.setattr(evidence, "passes", observe)
    limited, residual = apply_score_gate(ranked, 2, relevance=evidence)
    selected, _ = _select_distinct_results(limited, residual, 2, DeliveryHistory(), relevance=evidence)
    assert selected == [ranked[0], ranked[2]]
    assert inspected == ["first", "twin", "other"]
    inspected.clear()
    assert _select_distinct_results(selected, [ranked[3]], 2, DeliveryHistory(), relevance=evidence)[0] == selected
    assert inspected == []


@pytest.mark.parametrize("form", ["drop", "stub"])
def test_unqualified_exact_twins_remain_unqualified_references(monkeypatch, form):
    monkeypatch.setenv("LM_RECALL_GATE_FORM", form)
    ranked = [row(str(i), "A", 0) for i in range(4)]
    evidence = QueryRelevance(ranked)
    limited, residual = apply_score_gate(ranked, 2, relevance=evidence)
    assert _select_distinct_results(
        limited, residual, 2, DeliveryHistory(), relevance=evidence,
    ) == (limited, residual)
    assert all(r.withheld == "below_threshold" for r in limited)


@pytest.mark.parametrize("view", ["knowledge", "full"])
def test_native_correction_decay_and_scope_during_replacement(tmp_path, monkeypatch, view):
    monkeypatch.setenv("LM_RECALL_GATE_FORM", "drop")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    scope = "project:cobalt"
    twins = [store.create_node(level=level, content="Cobalt routing requires approval.",
                               context={"scope": scope}) for level in ("trace", "concept")]
    original = store.append_trace("Cobalt routing requires inventory.", {"scope": scope})
    foreign = store.append_trace("Cobalt routing requires approval.", {"scope": "project:foreign"})

    async def scenario():
        async with Client(mcp) as client:
            taught = (await client.call_tool("memory_teach", {
                "trace_id": original.id, "correction": "Cobalt routing requires receipts.",
                "context": {"scope": scope},
            })).structured_content
            correction_id = taught["corrective_trace"]["id"]
            args = {"query": "cobalt routing approval", "scope": scope, "depth": 0,
                    "max_results": 2, "view": view}
            active = (await client.call_tool("memory_recall", args)).structured_content
            assert [e.get("node", e)["id"] for e in active["results"]] == [twins[0].id, correction_id]
            assert [e["delivery"] for e in active["results"]] == ["full", "full"]
            store.soft_delete_node(correction_id, "expired synthetic correction")
            decayed = (await client.call_tool("memory_recall", args)).structured_content
            assert [e.get("node", e)["id"] for e in decayed["results"]] == [n.id for n in twins]
            assert store.get_node(foreign.id).access_count == 0
            assert store.get_node(correction_id).access_count == 1
            looked_up = (await client.call_tool("memory_lookup", {
                "node_id": original.id, "scope": scope,
            })).structured_content["results"][0]
            assert looked_up["superseded"] is True
            assert looked_up["superseded_by"] == [correction_id]
            assert looked_up["provenance"]["corrections"]

    try:
        asyncio.run(scenario())
    finally:
        mcp.auto_consolidation.close()
        store.close()


@pytest.mark.parametrize("view", ["knowledge", "full"])
@pytest.mark.parametrize("form", ["drop", "stub"])
@pytest.mark.parametrize("threshold", ["0.35", "0.7"])
def test_native_qualified_replacement_and_accounting(tmp_path, monkeypatch, view, form, threshold):
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", threshold)
    monkeypatch.setenv("LM_RECALL_GATE_FORM", form)
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    nodes = [store.create_node(level=level, content=content, context={"scope": "project:cobalt"})
             for level, content in [
                 ("trace", "Cobalt routing requires approval."),
                 ("concept", "Cobalt routing requires approval."),
                 ("trace", "Cobalt routing requires inventory."),
                 ("trace", "Cobalt porcelain kiln temperatures."),
             ]]
    observations = []
    rank = MemoryRecallService.rank_candidates

    def observe(service, *args, **kwargs):
        rows = rank(service, *args, **kwargs)
        observations.append((service, rows))
        return rows

    monkeypatch.setattr(MemoryRecallService, "rank_candidates", observe)
    selected = [nodes[0].id, nodes[2 if threshold == "0.35" else 1].id]

    async def scenario():
        async with Client(mcp) as client:
            args = {"query": "cobalt routing approval", "scope": "project:cobalt",
                    "depth": 0, "max_results": 2, "view": view}
            first = (await client.call_tool("memory_recall", args)).structured_content
            service, ranked = observations[-1]
            # Observe ordinary ranking; no ordering or channel score is replaced.
            assert [r.node.id for r in ranked] == [node.id for node in nodes]
            evidence = service.last_query_relevance
            assert evidence.passes(ranked[2]) == (threshold == "0.35")
            assert not evidence.passes(ranked[3])
            assert [entry.get("node", entry)["id"] for entry in first["results"]] == selected
            expected_deliveries = ["full", "full" if threshold == "0.35" else "twin_duplicate"]
            assert [entry["delivery"] for entry in first["results"]] == expected_deliveries
            residual = [node.id for node in nodes if node.id not in selected]
            assert [r.node.id for r in service.last_residual] == residual
            event = store.get_recall_event(first["recall_event_id"])
            assert event.result_ids == selected
            assert [store.get_node(node.id).access_count for node in nodes] == [
                int(node.id in selected) for node in nodes
            ]
            recorded_map = store.recent_recall_map_history()[0]["recall_map"]
            assert recorded_map["sel"]["n"] == len(residual)
            assert first["recall_map"] == {
                key: value for key, value in recorded_map.items()
                if view == "full" or key != "sel"
            }
            full_ids = [node_id for node_id, delivery in zip(selected, expected_deliveries)
                        if delivery == "full"]
            second = (await client.call_tool("memory_recall", {
                **args, "used": full_ids, "irrelevant": residual,
            })).structured_content
            assert second["feedback_marks"] == {
                "accepted": len(full_ids), "dropped": len(residual),
                "by_reason": {"not_delivered": len(residual)},
            }
            assert [entry.get("node", entry)["id"] for entry in second["results"]] == selected
            assert [entry["delivery"] for entry in second["results"]] == [
                "session_duplicate" if delivery == "full" else delivery
                for delivery in expected_deliveries
            ]
            assert not store.connection.in_transaction

    try:
        asyncio.run(scenario())
    finally:
        mcp.auto_consolidation.close()
        store.close()
