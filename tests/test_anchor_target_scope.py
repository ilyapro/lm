"""Anchor roots use the strongest targets admitted by the recall scope plan."""

from dataclasses import replace

import pytest

from living_memory import query_anchors as qa
from living_memory.retrieval import GRAPH_SEED_LIMIT, MemoryRecallService
from living_memory.scope import resolve_scope
from living_memory.storage import MemoryStore
from test_recall_graph_preparation import Encoder, node


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    # Isolate candidate selection from the delivery quality threshold.
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "off")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "off")


def recall(service, **scope_args):
    return service.memory_recall(
        "needle", depth=1, max_results=100, log_access=False, log_event=False,
        repair_pending=False, **scope_args,
    )


def population(store, count, *, scope="project:foreign"):
    node(store, "wanted", scope="global")
    targets = [("wanted", 0.9)]
    for index in range(count):
        name = f"strong-{index:03d}"
        node(store, name, scope=scope)
        targets.append((name, 1.0))
    written = qa.upsert_anchor(store, "needle", "global", [1.0, 0.0], targets)
    assert len(written.edges) == count + 1
    return written.anchor


@pytest.mark.parametrize("count", [49, 50, 75])
@pytest.mark.parametrize("scope_args,default_scope,restricted", [
    ({"scope": "global"}, "project:alpha", True),
    ({"scope": "project:alpha"}, "global", True),
    ({"scope": "session:review", "ambient_context": {"project": "alpha"}}, "global", True),
    ({}, "global", False),
    ({}, "project:alpha", False),
    ({"ambient_context": {"project": "alpha"}}, "global", False),
], ids=["global", "project", "session", "all", "default-project", "ambient-project"])
def test_foreign_targets_do_not_take_restricted_seed_slots(count, scope_args, default_scope, restricted):
    with MemoryStore(":memory:") as store:
        store.config = replace(store.config, default_scope=default_scope)
        population(store, count)
        service = MemoryRecallService(store, embedder=Encoder())
        plan = resolve_scope(store=store, **scope_args)
        assert plan.restricted is restricted
        # Independent expectations from the fixture's weights and IDs.
        expected = {"wanted": 0.9} if restricted else {
            f"strong-{i:03d}": 1.0 for i in range(min(count, 50))
        }
        if not restricted and count == 49:
            expected["wanted"] = 0.9
        seeds = service._collect_anchor_seeds(plan, [1.0, 0.0])
        delivered = {row.node.id for row in recall(service, **scope_args)}
        assert {"seeds": seeds, "recall": delivered} == {
            "seeds": expected, "recall": set(expected),
        }


@pytest.mark.parametrize("count", [49, 50, 75])
def test_same_scope_targets_keep_the_fifty_root_cap(count):
    with MemoryStore(":memory:") as store:
        population(store, count, scope="global")
        service = MemoryRecallService(store, embedder=Encoder())
        expected = {f"strong-{i:03d}": 1.0 for i in range(min(count, 50))}
        if count == 49:
            expected["wanted"] = 0.9
        seeds = service._collect_anchor_seeds(resolve_scope(scope="global"), [1.0, 0.0])
        assert GRAPH_SEED_LIMIT == 50
        assert seeds == expected
        assert {row.node.id for row in recall(service, scope="global")} == set(expected)


@pytest.mark.parametrize("scope_args,expected", [
    ({"scope": "global"}, {"g"}),
    ({"scope": "project:alpha"}, {"g", "p"}),
    ({"scope": "session:review", "ambient_context": {"project": "alpha"}}, {"g", "p", "s"}),
    ({"ambient_context": {"scope": "project:alpha"}}, {"g", "p"}),
    ({}, {"g", "p", "s", "f"}),
])
def test_target_selection_obeys_all_scope_plan_members(scope_args, expected):
    with MemoryStore(":memory:") as store:
        for name, scope in (("g", "global"), ("p", "project:alpha"),
                            ("s", "session:review"), ("f", "project:foreign")):
            node(store, name, scope=scope)
        qa.upsert_anchor(store, "needle", "global", [1.0, 0.0], ["g", "p", "s", "f"])
        service = MemoryRecallService(store, embedder=Encoder())
        seeds = service._collect_anchor_seeds(resolve_scope(**scope_args), [1.0, 0.0])
        assert set(seeds) == expected
        assert {row.node.id for row in recall(service, **scope_args)} == expected


def test_shared_targets_keep_max_activation_and_id_ties_at_the_boundary():
    with MemoryStore(":memory:") as store:
        strong = [(f"strong-{i:03d}", 0.9) for i in range(48)]
        first = strong + [("shared", 0.95), ("a-boundary", 0.8), ("z-boundary", 0.8), ("sum", 0.45)]
        for name, _ in first:
            node(store, name, scope="global")
        qa.upsert_anchor(store, "needle", "global", [1.0, 0.0], first)
        qa.upsert_anchor(store, "nearby", "global", [4.0, 3.0], [("shared", 1.0), ("sum", 0.5)])
        service = MemoryRecallService(store, embedder=Encoder())
        expected = {"shared": 0.95, **dict(strong), "a-boundary": 0.8}
        seeds = service._collect_anchor_seeds(resolve_scope(scope="global"), [1.0, 0.0])
        assert list(seeds) == list(expected)
        assert seeds == expected
        assert {row.node.id for row in recall(service, scope="global")} == set(expected)
