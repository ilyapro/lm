"""Anchor roots use the strongest targets admitted by the recall scope plan."""

from collections import Counter
from dataclasses import replace
import sqlite3

import pytest

from living_memory import query_anchors as qa
from living_memory import storage
from living_memory.irrelevance import QUERY_IRRELEVANCE_SCHEMA_SQL
from living_memory.retrieval import GRAPH_SEED_LIMIT, MemoryRecallService, _Candidate
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


@pytest.mark.parametrize("scope_args,expected,delivered", [
    ({"scope": "global"}, {"g"}, {"g"}),
    ({"scope": "project:alpha"}, {"g", "p"}, {"p"}),
    ({"scope": "session:review", "ambient_context": {"project": "alpha"}}, {"g", "p", "s"}, {"p", "s"}),
    ({"ambient_context": {"scope": "project:alpha"}}, {"g", "p"}, {"p"}),
    ({}, {"g", "p", "s", "f"}, {"g", "p", "s", "f"}),
])
def test_target_selection_obeys_all_scope_plan_members(scope_args, expected, delivered):
    with MemoryStore(":memory:") as store:
        for name, scope in (("g", "global"), ("p", "project:alpha"),
                            ("s", "session:review"), ("f", "project:foreign")):
            node(store, name, scope=scope)
        qa.upsert_anchor(store, "needle", "global", [1.0, 0.0], ["g", "p", "s", "f"])
        service = MemoryRecallService(store, embedder=Encoder())
        seeds = service._collect_anchor_seeds(resolve_scope(**scope_args), [1.0, 0.0])
        assert set(seeds) == expected
        # The ranker's cross-scope gate still drops the weak global candidate
        # when a narrower candidate is present; scope admission is earlier.
        assert {row.node.id for row in recall(service, **scope_args)} == delivered


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


@pytest.mark.parametrize("state", ["missing", "decayed", "corrected"])
def test_current_target_validity_survives_scope_selection(state):
    with MemoryStore(":memory:") as store:
        population(store, 50, scope="global")
        for i in range(50):
            target = f"strong-{i:03d}"
            if state == "missing":
                store.delete_node(target)
            elif state == "decayed":
                store.soft_delete_node(target, "expired")
            else:
                if i == 0:
                    node(store, "replacement", scope="global")
                store.create_connection("replacement", target, "supersedes")
        service = MemoryRecallService(store, embedder=Encoder())
        expected = {"wanted": 0.9}
        if state == "corrected":
            expected["replacement"] = 1.0
        seeds = service._collect_anchor_seeds(resolve_scope(scope="global"), [1.0, 0.0])
        assert seeds == expected
        results = recall(service, scope="global")
        if state == "corrected":
            # Supersedes edges also let the walk reach live historical nodes.
            # They remain flagged and rank below the correction.
            stale = {f"strong-{i:03d}" for i in range(50)}
            assert {row.node.id for row in results} == set(expected) | stale
            assert {row.node.id for row in results if row.superseded} == stale
            ids = [row.node.id for row in results]
            assert all(ids.index("replacement") < ids.index(key) for key in stale)
        else:
            assert {row.node.id for row in results} == set(expected)


@pytest.mark.parametrize("change", ["scope", "decay", "delete"])
def test_admission_rechecks_selected_targets_without_refilling(tmp_path, monkeypatch, change):
    path = tmp_path / "concurrent.sqlite3"
    with MemoryStore(path) as store, MemoryStore(path) as writer:
        population(store, 50, scope="global")
        service = MemoryRecallService(store, embedder=Encoder())
        admit = service._admit_anchor_seeds
        selected = []

        def mutate_then_admit(plan, candidates, seeds):
            selected.append(dict(seeds))
            if change == "scope":
                writer.update_node("strong-000", context={"scope": "project:foreign"})
            elif change == "decay":
                writer.soft_delete_node("strong-000", "expired")
            else:
                writer.delete_node("strong-000")
            return admit(plan, candidates, seeds)

        monkeypatch.setattr(service, "_admit_anchor_seeds", mutate_then_admit)
        assert {row.node.id for row in recall(service, scope="global")} == {
            f"strong-{i:03d}" for i in range(1, 50)
        }
        assert selected == [{f"strong-{i:03d}": 1.0 for i in range(50)}]
        monkeypatch.setattr(service, "_admit_anchor_seeds", admit)
        # A later recall performs a fresh selection and can fill the vacancy.
        assert {row.node.id for row in recall(service, scope="global")} == {
            "wanted", *(f"strong-{i:03d}" for i in range(1, 50)),
        }


def test_target_scopes_are_current_after_matching_and_rollback(tmp_path):
    path = tmp_path / "scopes.sqlite3"
    with MemoryStore(path) as store, MemoryStore(path) as writer:
        population(store, 50)
        service = MemoryRecallService(store, embedder=Encoder())
        plan = resolve_scope(scope="global")
        matches = service._match_query_anchors(plan, [1.0, 0.0])
        assert service._collect_anchor_seeds(plan, [1.0, 0.0], matches=matches) == {"wanted": 0.9}
        writer.update_node("strong-000", context={"scope": "global"})
        assert service._collect_anchor_seeds(plan, [1.0, 0.0], matches=matches) == {
            "wanted": 0.9, "strong-000": 1.0,
        }
        with pytest.raises(RuntimeError, match="rollback"):
            with store.transaction():
                store.update_node("strong-000", context={"scope": "project:foreign"})
                assert service._collect_anchor_seeds(plan, [1.0, 0.0], matches=matches) == {"wanted": 0.9}
                raise RuntimeError("rollback")
        assert service._collect_anchor_seeds(plan, [1.0, 0.0], matches=matches) == {
            "wanted": 0.9, "strong-000": 1.0,
        }


def test_scope_selection_leaves_matches_and_independent_demotions_intact(monkeypatch):
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    monkeypatch.setenv("LM_QUERY_IRRELEVANCE_FACTOR", "0.5")
    with MemoryStore(":memory:") as store:
        anchor = population(store, 50)
        with store.transaction():
            store.connection.execute(QUERY_IRRELEVANCE_SCHEMA_SQL)
            store.connection.execute(
                "INSERT INTO query_irrelevance (anchor_id, node_id, weight, created_at, updated_at) "
                "VALUES (?, 'strong-000', 1, 't', 't')", (anchor.id,),
            )
        service = MemoryRecallService(store, embedder=Encoder())
        plan = resolve_scope(scope="global")
        matches = service._match_query_anchors(plan, [1.0, 0.0])
        targets = matches[0].targets
        assert len(targets) == 51
        assert service._collect_anchor_seeds(plan, [1.0, 0.0], matches=matches) == {"wanted": 0.9}
        assert matches[0].targets == targets
        assert service._collect_query_demotions(plan, [1.0, 0.0], matches=matches) == {
            "strong-000": 0.5,
        }
        assert service._collect_query_demotions(plan, [1.0, 0.0]) == {"strong-000": 0.5}


@pytest.mark.parametrize("failure", ["scope-read", "body-read", "selected-body"])
def test_required_reads_fail_before_access_or_event_writes(monkeypatch, failure):
    with MemoryStore(":memory:") as store:
        population(store, 50)
        service = MemoryRecallService(store, embedder=Encoder())
        if failure == "selected-body":
            with store.transaction():
                store.connection.execute("UPDATE nodes SET provenance = '[]' WHERE id = 'wanted'")
            error = TypeError
        else:
            def fail(*args, **kwargs):
                raise sqlite3.OperationalError("required read failed")

            monkeypatch.setattr(store, "get_node_scopes" if failure == "scope-read" else "get_recall_nodes", fail)
            error = sqlite3.OperationalError
        before = list(store.connection.iterdump())
        with pytest.raises(error):
            service.memory_recall("needle", scope="global", repair_pending=False)
        assert list(store.connection.iterdump()) == before


def test_bounded_anchor_bodies_preserve_independent_ordinary_and_source_slots(monkeypatch):
    with MemoryStore(":memory:") as store:
        population(store, 430)
        selected = {f"accepted-{i:03d}" for i in range(50)}
        accepted = [(f"accepted-{i:03d}", 0.95) for i in range(75)]
        for name, _ in accepted:
            node(store, name, scope="global")
        qa.upsert_anchor(store, "needle", "global", [1.0, 0.0], accepted)
        roots = [node(store, f"root-{i:03d}", scope="global") for i in range(51)]
        node(store, "leaf", scope="global")
        for root in roots:
            store.create_connection(root.id, "leaf", "requires", weight=0.5)
        service = MemoryRecallService(store, embedder=Encoder())
        plan = resolve_scope(scope="global")
        walk = service._walk_graph
        walks = []

        def observe(plan, candidates, roots, **kwargs):
            walks.append((kwargs.get("source_pass", False), kwargs["anchor_pass"], list(roots)))
            return walk(plan, candidates, roots, **kwargs)

        monkeypatch.setattr(service, "_walk_graph", observe)

        def run(seeds):
            candidates = {root.id: _Candidate(node=root, bm25_score=1 - i / 100,
                                             lexical_score=0.5 + i / 100)
                          for i, root in enumerate(roots)}
            service._collect_graph(plan, candidates, max_depth=1, causal_mode=False, anchor_seeds=seeds)
            return candidates

        control = run({})
        expected_walks = list(walks)
        assert expected_walks[0][2] == [(f"root-{i:03d}", 1 - i / 100) for i in range(50)]
        assert expected_walks[1][2] == [(f"root-{i:03d}", 0.5 + i / 100) for i in range(50, 0, -1)]
        walks.clear()
        decoded = Counter()
        decode = storage._node_from_row

        def selected_only(row):
            assert row["id"] in selected | {"leaf"}
            decoded[row["id"]] += 1
            return decode(row)

        monkeypatch.setattr(storage, "_node_from_row", selected_only)
        statements = []
        store.connection.set_trace_callback(statements.append)
        try:
            seeds = service._collect_anchor_seeds(plan, [1.0, 0.0])
            assert seeds == {name: 0.95 for name in selected}
            assert not decoded
            candidates = run(seeds)
        finally:
            store.connection.set_trace_callback(None)
        assert walks[:2] == expected_walks
        assert len(walks[2][2]) == 50
        assert {key for key in candidates if key not in control} == selected
        for key, original in control.items():
            assert (candidates[key].graph_score, candidates[key].source_graph_score) == (
                original.graph_score, original.source_graph_score,
            )
        assert {key: value for key, value in decoded.items() if key != "leaf"} == {
            name: 1 for name in selected
        }
        body_reads = [sql for sql in statements if "SELECT * FROM nodes" in sql]
        assert body_reads
        assert all("strong-" not in sql and "wanted" not in sql and "accepted-050" not in sql
                   for sql in body_reads)
        assert any("SELECT id, scope FROM nodes" in sql for sql in statements)
