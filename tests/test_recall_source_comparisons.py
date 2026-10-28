"""Recall compares full sources only when exact scope/content can coincide."""

import asyncio
from contextlib import contextmanager
from dataclasses import replace
import json
import sys

from fastmcp import Client
import pytest

from living_memory.delivery import DeliveryHistory, recall_delivery_records, shape_recall_results
from living_memory.models import Node, content_identity
from living_memory.retrieval import MemoryRecallService, RecallResult
from living_memory.server import _select_distinct_results, create_mcp_server


def row(node_id, content="Shared fact.", **kwargs):
    return RecallResult(Node(id=node_id, level="trace", content=content, **kwargs), 1.0)


def shape(rows, view="knowledge", history=None):
    return shape_recall_results(
        rows, view=view, delivery_history=history or DeliveryHistory(), session_dedup=True,
        snippet_max_chars=1000, context_value_max_chars=160,
    )


@contextmanager
def identity_work():
    # Observe the native function and serializer regardless of imported aliases.
    work = {"ids": [], "bytes": 0}
    previous = sys.getprofile()

    def profile(frame, event, value):
        if event == "call" and frame.f_code is content_identity.__code__:
            work["ids"].append(frame.f_locals["node"].id)
        if (event == "return" and frame.f_code is json.dumps.__code__
                and frame.f_back.f_code is content_identity.__code__):
            work["bytes"] += len(value.encode())

    sys.setprofile(profile)
    try:
        yield work
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("view", ["knowledge", "full"])
def test_unique_scope_content_never_serializes_sources(view):
    rows = [row("a"), row("b", scope="project:other"), row("c", "Shared fact. "),
            row("d", "shared fact."), row("e", "Different fact.")]
    for result in rows:
        result.node.context = {"procedure": "x" * 8192}
        result.node.provenance = {"unrelated": "y" * 8192}
    with identity_work() as work:
        selected, residual = _select_distinct_results(rows[:4], rows[4:], 4, DeliveryHistory())
        shaped = shape(selected, view)
    assert selected == rows[:4] and residual == rows[4:]
    assert work == {"ids": [], "bytes": 0}
    assert [entry["delivery"] for entry in shaped] == ["full"] * 4


def test_selection_only_compares_consumed_collisions_in_its_bounded_pool():
    a, b, c, twin_c, twin_a = [row(i, text) for i, text in
                              [("a", "A"), ("b", "B"), ("c", "C"), ("cc", "C"), ("aa", "A")]]
    with identity_work() as work:
        selected, residual = _select_distinct_results([a, b], [c, twin_c, twin_a], 2, DeliveryHistory())
    assert selected == [a, b] and residual == [c, twin_c, twin_a]
    assert work == {"ids": [], "bytes": 0}


@pytest.mark.parametrize("contents, limit, selected_indices, residual_indices", [
    ("AAAB", 3, [0, 1, 3], [2]),
    ("AABBACC", 5, [0, 1, 2, 3, 5], [4, 6]),
    ("AAAAAAB", 3, [0, 1, 2], [3, 4, 5, 6]),
    ("AAABABC", 3, [0, 1, 3], [2, 4, 5, 6]),
])
def test_selection_keeps_available_replacements_and_ranked_fillers(
    contents, limit, selected_indices, residual_indices,
):
    rows = [row(str(index), content) for index, content in enumerate(contents)]
    selected, residual = _select_distinct_results(rows[:limit], rows[limit:], limit, DeliveryHistory())
    assert selected == [rows[index] for index in selected_indices]
    assert residual == [rows[index] for index in residual_indices]


@pytest.mark.parametrize("view", ["knowledge", "full"])
@pytest.mark.parametrize("distinction", [
    "context", "provenance", "sources", "corrections", "unknown_derivation",
    "derivation_sources", "derivation_group", "bookkeeping", "level", "known_derivation",
])
def test_collisions_keep_complete_source_equality(view, distinction):
    first, second, later = row("first"), row("second"), row("later", "Later fact.")
    if distinction == "context":
        second.node.context = {"procedure": "independent"}
    elif distinction == "provenance":
        second.node.provenance = {"unrelated": "independent"}
    elif distinction == "sources":
        second.node.source_traces = ["source-b"]
    elif distinction == "corrections":
        second.node.corrections = [{"correction": "replacement"}]
    elif "derivation" in distinction:
        for result in (first, second):
            result.node.level = "concept"
            result.node.provenance = {"strategy": "token-jaccard"}
            result.node.source_traces = [] if distinction == "unknown_derivation" else ["source-a"]
        if distinction == "derivation_sources":
            second.node.source_traces = ["source-b"]
        elif distinction == "derivation_group":
            second.node.context = {"task_pattern": "other group"}
    elif distinction == "bookkeeping":
        second.node.context = {"timestamp": "later", "scope": "ignored-context-scope"}
        second.node.provenance = {"prior_recalls": ["event"], "source_traces": ["ignored"],
                                  "corrections": ["ignored"]}
    else:
        second.node.level = "concept"
    twins = distinction in ("bookkeeping", "level", "known_derivation")
    with identity_work() as selection_work:
        selected, residual = _select_distinct_results([first, second], [later], 2, DeliveryHistory())
    assert selection_work["ids"] == ["first", "second"]
    assert selected == ([first, later] if twins else [first, second])
    assert residual == ([second] if twins else [later])
    with identity_work() as shaping_work:
        shaped = shape([first, second, later], view)
    assert shaping_work["ids"] == ["first", "second"]
    assert [entry["delivery"] for entry in shaped] == ["full", "twin_duplicate" if twins else "full", "full"]
    if twins:
        assert shaped[1]["content_ref"]["duplicate_of"] == first.node.id


@pytest.mark.parametrize("view", ["knowledge", "full"])
def test_withheld_and_empty_content_cannot_create_a_collision_or_bearer(view):
    withheld = replace(row("withheld"), withheld="below_threshold")
    visible, twin = row("visible"), row("twin")
    empty = [row("empty-a", ""), row("empty-b", "")]
    with identity_work() as work:
        shaped = shape([withheld, visible, *empty], view)
    assert work == {"ids": [], "bytes": 0}
    assert [entry["delivery"] for entry in shaped] == ["below_threshold", "full", "full", "full"]
    with identity_work() as work:
        shaped = shape([withheld, visible, twin], view)
    assert work["ids"] == ["visible", "twin"]
    assert shaped[-1]["content_ref"]["duplicate_of"] == "visible"


def test_session_repeat_keeps_its_slot_and_delivery_uses_the_first_bearer():
    first, twin, later = row("first"), row("twin"), row("later", "Later fact.")
    records = recall_delivery_records([twin], shape([twin]))
    history = DeliveryHistory([{"node_id": key, "delivery": value} for key, value in records.items()])
    selected, residual = _select_distinct_results([first, twin], [later], 2, history)
    assert selected == [first, twin] and residual == [later]
    shaped = shape(selected, history=history)
    assert [entry["delivery"] for entry in shaped] == ["full", "twin_duplicate"]
    assert shaped[1]["content_ref"]["duplicate_of"] == "first"


@pytest.mark.parametrize("view", ["knowledge", "full"])
@pytest.mark.parametrize("limit, twins, distinct, selected_indices", [
    (2, 2, 2, [0, 2]),
    (2, 3, 0, [0, 1]),
    (3, 3, 1, [0, 1, 3]),
    (3, 3, 2, [0, 3, 4]),
])
def test_native_recall_preserves_selection_history_and_feedback(
    tmp_path, monkeypatch, view, limit, twins, distinct, selected_indices,
):
    for key, value in {"LIVING_MEMORY_EMBEDDING_BACKEND": "hash", "LM_RECALL_MIN_SCORE": "off",
                       "LM_RECALL_MAP": "0", "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
                       "LM_AUTO_CONSOLIDATE_POLICY": "off", "LM_DELIVERY_SESSION_DEDUP": "1"}.items():
        monkeypatch.setenv(key, value)
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    facts = ["approval"] * twins + ["inventory", "receipts"][:distinct]
    nodes = [store.create_node(
        level=("trace", "concept", "schema")[index] if index < twins and index < 3 else "trace",
        content=f"Cobalt routing requires {fact}.", context={"scope": "project:source-test"},
    ) for index, fact in enumerate(facts)]
    selected_ids = [nodes[index].id for index in selected_indices]
    residual_ids = [node.id for index, node in enumerate(nodes) if index not in selected_indices]
    deliveries = ["twin_duplicate" if 0 < index < twins else "full" for index in selected_indices]
    full_ids = [node_id for node_id, delivery in zip(selected_ids, deliveries) if delivery == "full"]
    order = {node.id: i for i, node in enumerate(nodes)}
    services = []
    rank = MemoryRecallService.rank_candidates

    def ranked(service, *args, **kwargs):
        services.append(service)
        return sorted(rank(service, *args, **kwargs), key=lambda result: order[result.node.id])

    monkeypatch.setattr(MemoryRecallService, "rank_candidates", ranked)

    async def scenario():
        async with Client(mcp) as client:
            args = {"query": "cobalt routing", "scope": "project:source-test", "depth": 0,
                    "max_results": limit, "view": view}
            first = (await client.call_tool("memory_recall", args)).structured_content
            assert [entry.get("node", entry)["id"] for entry in first["results"]] == selected_ids
            assert [entry["delivery"] for entry in first["results"]] == deliveries
            assert [result.node.id for result in services[-1].last_residual] == residual_ids
            assert store.get_recall_event(first["recall_event_id"]).result_ids == selected_ids
            assert [store.get_node(node.id).access_count for node in nodes] == [
                int(index in selected_indices) for index in range(len(nodes))
            ]
            second = (await client.call_tool("memory_recall", {
                **args, "used": full_ids, "irrelevant": [residual_ids[0]],
            })).structured_content
            assert [entry.get("node", entry)["id"] for entry in second["results"]] == selected_ids
            assert [entry["delivery"] for entry in second["results"]] == [
                "session_duplicate" if delivery == "full" else delivery for delivery in deliveries
            ]
            assert second["feedback_marks"] == {
                "accepted": len(full_ids), "dropped": 1, "by_reason": {"not_delivered": 1},
            }
            assert not store.connection.in_transaction

    try:
        asyncio.run(scenario())
    finally:
        mcp.auto_consolidation.close()
        store.close()
