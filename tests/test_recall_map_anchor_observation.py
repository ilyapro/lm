"""Build-local anchor reads serve both reoffer digests and the cascade."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import os
import sqlite3
from unittest.mock import patch

import pytest

from living_memory.recall_map import (
    RecallMapBuilder, STAGE_ANCHOR, STAGE_PATH, STAGE_STRUCTURAL, _Curtailment,
)
from living_memory.retrieval import RecallResult
from living_memory.storage import MaturedRecallHistory, MemoryStore, recall_fingerprint

SCOPE = "project:lm"
DECISION_AT = "2026-08-23T12:00:00Z"




def seed_pool(store, *, early_stages=False):
    # Unrelated documents give the real label gate a corpus for rarity scores.
    for i in range(32):
        store.create_node(level="trace", content=f"quarterly ledger reconciliation entry {i}",
                          context={"scope": SCOPE})
    contexts = [{}, {}, {}]
    if early_stages:
        contexts += [{"procedure_id": "orchard-inspection"}, {"files": ["src/harvest/plan.py"]}]
    nodes = [
        store.create_node(
            node_id=f"member-{i}", level="trace", content=f"orchard harvest evidence {i}",
            context={"scope": SCOPE, **context}, timestamp=DECISION_AT,
        )
        for i, context in enumerate(contexts)
    ]
    for anchor_id, query, indices in (
        ("anchor-a", "inspect orchard branches", [0, 1, 3]),
        ("anchor-b", "schedule harvest rotation", [2, 4]),
    ):
        store.insert_query_anchor(
            anchor_id=anchor_id, scope=SCOPE, query=query,
            fingerprint=recall_fingerprint(query, SCOPE), embedding=[1.0, 0.0],
            now=DECISION_AT,
        )
        for index in indices:
            if index < len(nodes):
                store.upsert_query_anchor_edge(anchor_id, nodes[index].id, weight=0.5)
    return [RecallResult(node=node, score=1.0 - i * 0.01, vector_score=1.0) for i, node in enumerate(nodes)]


def due(digest="0" * 16):
    return _Curtailment(streak=7, offers=3, lead=4, pool_digest=digest, since_digest=3)


def build(builder, results):
    return builder.build(results, scope=SCOPE, task="orchard work")


@contextmanager
def observed(builder):
    """Count actual store reads and capture the digest returned by production."""
    digests = []
    original = builder._pool_digest

    def digest(*args, **kwargs):
        value = original(*args, **kwargs)
        digests.append(value)
        return value

    with (
        patch.object(builder.store, "list_query_anchor_edges", wraps=builder.store.list_query_anchor_edges) as edges,
        patch.object(builder.store, "get_query_anchor", wraps=builder.store.get_query_anchor) as anchors,
        patch.object(builder, "_pool_digest", side_effect=digest),
    ):
        yield edges, anchors, digests


def assert_reads(observation, node_ids, anchor_ids):
    edges, anchors, _ = observation
    assert Counter(call.kwargs["target_id"] for call in edges.call_args_list) == Counter(node_ids)
    assert Counter(call.args[0] for call in anchors.call_args_list) == Counter(anchor_ids)


@pytest.fixture(autouse=True)
def map_settings(monkeypatch):
    for name in list(os.environ):
        if name.startswith("LM_MAP_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("LM_MAP_CURTAIL_DECAY", "1")


@pytest.fixture
def store(tmp_path, monkeypatch):
    with MemoryStore(tmp_path / "anchors.sqlite3") as opened:
        yield opened


@pytest.mark.parametrize("early_stages", [False, True])
def test_digest_and_cascade_share_reads(store, monkeypatch, early_stages):
    results = seed_pool(store, early_stages=early_stages)
    builder = RecallMapBuilder(store)
    monkeypatch.setattr(builder, "_curtailment", lambda *a, **kw: due())
    with observed(builder) as reads:
        first = build(builder, results)
    assert first is not None and not first.curtailed
    assert len(reads[2]) == 1
    assert_reads(reads, [r.node.id for r in results], ["anchor-a", "anchor-b"])
    anchored = [c for c in first.clusters if c.stage == STAGE_ANCHOR]
    assert [c.member_ids for c in anchored] == [("member-0", "member-1"), ("member-2",)]
    if early_stages:
        assert {c.stage for c in builder._cache[first.key].templates} == {
            STAGE_STRUCTURAL, STAGE_PATH, STAGE_ANCHOR,
        }
    with observed(builder) as cached_reads:
        cached = build(builder, results)
    assert builder.last_cache_hit
    assert cached == first
    assert cached_reads[2] == reads[2]
    assert_reads(cached_reads, [r.node.id for r in results], ["anchor-a", "anchor-b"])


def test_offer_without_digest_observes_only_remaining_and_cache_hit_reads_nothing(store, monkeypatch):
    results = seed_pool(store, early_stages=True)
    builder = RecallMapBuilder(store)
    monkeypatch.setattr(builder, "_curtailment", lambda *a, **kw: _Curtailment(0, 0))
    with observed(builder) as reads:
        first = build(builder, results)
    assert first.pool_size == 5 and first.clusters
    assert_reads(reads, [r.node.id for r in results[:3]], ["anchor-a", "anchor-b"])
    assert reads[2] == []
    with observed(builder) as reads:
        assert build(builder, results) == first
    assert builder.last_cache_hit
    assert_reads(reads, [], [])
    assert reads[2] == []


@pytest.mark.parametrize("skip", ["empty", "within_retry", "decay_disabled"])
def test_skipped_build_does_not_observe_anchors(store, monkeypatch, skip):
    results = seed_pool(store)
    builder = RecallMapBuilder(store)
    verdict = _Curtailment(4, 3, lead=1, pool_digest="0" * 16)
    monkeypatch.setattr(builder, "_curtailment", lambda *a, **kw: verdict)
    if skip == "empty":
        results = []
    elif skip == "decay_disabled":
        monkeypatch.delenv("LM_MAP_CURTAIL_DECAY")
        verdict = due()
    with observed(builder) as reads:
        result = build(builder, results)
    assert result is None if skip == "empty" else result.curtailed
    assert_reads(reads, [], [])
    assert reads[2] == []


def test_unchanged_digest_stops_before_clustering(store, monkeypatch):
    results = seed_pool(store)
    builder = RecallMapBuilder(store)
    monkeypatch.setattr(builder, "_curtailment", lambda *a, **kw: due())
    with observed(builder) as initial:
        build(builder, results)
    monkeypatch.setattr(builder, "_curtailment", lambda *a, **kw: due(initial[2][0]))
    with observed(builder) as reads, patch.object(builder, "_cluster", side_effect=AssertionError("unexpected cascade")):
        result = build(builder, results)
    assert result.curtailed and result.pool_digest == initial[2][0]
    assert reads[2] == initial[2]
    assert_reads(reads, [r.node.id for r in results], ["anchor-a", "anchor-b"])


def test_next_build_sees_external_anchor_changes(store, monkeypatch):
    results = seed_pool(store)
    builder = RecallMapBuilder(store)
    monkeypatch.setattr(builder, "_curtailment", lambda *a, **kw: due())
    with observed(builder) as initial:
        first = build(builder, results)
    with sqlite3.connect(store.connection.execute("PRAGMA database_list").fetchone()[2]) as writer:
        writer.execute("UPDATE query_anchors SET decayed = 1 WHERE id = 'anchor-a'")
    with observed(builder) as reads:
        changed = build(builder, results)
    assert not builder.last_cache_hit
    assert changed != first
    assert [c.ask_hint for c in changed.clusters if c.stage == STAGE_ANCHOR] == [
        "schedule harvest rotation",
    ]
    assert reads[2] != initial[2]
    assert_reads(reads, [r.node.id for r in results], ["anchor-a", "anchor-b"])


def test_digest_observation_does_not_survive_a_rollback(store, monkeypatch):
    results = seed_pool(store)
    builder = RecallMapBuilder(store)
    monkeypatch.setattr(builder, "_curtailment", lambda *a, **kw: _Curtailment(3, 3))
    first = build(builder, results)
    store.connection.execute("UPDATE query_anchors SET decayed = 1 WHERE id = 'anchor-a'")
    transient = build(builder, results)
    store.connection.rollback()
    with observed(builder) as reads:
        restored = build(builder, results)
    assert transient.pool_digest != first.pool_digest
    assert restored == first
    assert_reads(reads, [r.node.id for r in results], ["anchor-a", "anchor-b"])


def test_ties_missing_decayed_and_blank_anchors(store, monkeypatch):
    results = seed_pool(store)
    for anchor_id, query in (("dead", "retired orchard question"), ("blank", " "), ("missing", "absent orchard question")):
        store.insert_query_anchor(anchor_id=anchor_id, scope=SCOPE, query=query,
                                  fingerprint=anchor_id, embedding=[1.0, 0.0])
        for result in results:
            store.upsert_query_anchor_edge(anchor_id, result.node.id, weight=1.0)
    store.decay_query_anchor("dead", "retired")
    store.connection.execute("PRAGMA foreign_keys = OFF")
    with store.connection:
        store.connection.execute("DELETE FROM query_anchors WHERE id = 'missing'")
    store.connection.execute("PRAGMA foreign_keys = ON")
    store.upsert_query_anchor_edge("anchor-b", "member-0", weight=0.5)
    store.upsert_query_anchor_edge("anchor-b", "member-1", weight=0.75)
    builder = RecallMapBuilder(store)
    monkeypatch.setattr(builder, "_curtailment", lambda *a, **kw: due())
    with observed(builder) as reads:
        result = build(builder, results)
    assert {c.ask_hint: c.member_ids for c in result.clusters} == {
        "inspect orchard branches": ("member-0",),
        "schedule harvest rotation": ("member-1", "member-2"),
    }
    assert_reads(reads, [r.node.id for r in results], ["anchor-a", "anchor-b", "dead", "blank", "missing"])


@pytest.mark.parametrize("method", ["list_query_anchor_edges", "get_query_anchor"])
def test_anchor_read_errors_propagate(store, monkeypatch, method):
    results = seed_pool(store)
    builder = RecallMapBuilder(store)
    monkeypatch.setattr(builder, "_curtailment", lambda *a, **kw: due())
    with patch.object(store, method, side_effect=sqlite3.OperationalError("anchor read failed")):
        with pytest.raises(sqlite3.OperationalError, match="anchor read failed"):
            build(builder, results)
    with observed(builder) as reads:
        assert len(build(builder, results).clusters) == 2
    assert_reads(reads, [r.node.id for r in results], ["anchor-a", "anchor-b"])
