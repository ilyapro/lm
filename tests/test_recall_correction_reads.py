"""Incident correction reads preserve membership without graph closure."""

from dataclasses import asdict

import pytest

from living_memory.replay import make_ranking_service
from living_memory.retrieval import MemoryRecallService, _Candidate
from living_memory.scope import ScopePlan
from living_memory.storage import MemoryStore


class Encoder:
    def embed(self, text):
        return [1.0, 0.0]


def node(store, name, scope="project:focus"):
    return store.create_node(
        node_id=name, level="trace", content=f"Harbor signal {name}",
        context={"scope": scope}, embedding=[1.0, 0.0],
    )


def test_incident_pairs_keep_both_directions_across_batches_and_node_states(tmp_path):
    with MemoryStore(tmp_path / "pairs.sqlite3") as store:
        ids = [f"n{i:03}" for i in range(405)]
        for name in ids + ["outside", "unrelated-a", "unrelated-b"]:
            node(store, name, scope="project:other" if name == "outside" else "project:focus")
        expected = [("n000", "n400"), ("n001", "n001"), ("n002", "outside"), ("outside", "n003")]
        for source, target in expected + [("unrelated-a", "unrelated-b")]:
            store.create_connection(source, target, "supersedes")
        store.create_connection("n000", "outside", "related")
        store.soft_delete_node("outside")
        import sqlite3
        store.connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        assert store.list_supersedes_pairs(iter(ids + ["n000", "missing"])) == expected
        assert store.list_supersedes_pairs(reversed(ids)) == expected
        statements = []
        store.connection.set_trace_callback(statements.append)
        assert store.list_supersedes_pairs([]) == []
        assert statements == []


def test_small_selection_does_not_walk_independent_correction_history(tmp_path):
    with MemoryStore(tmp_path / "pairs.sqlite3") as store:
        def add_groups(start, stop):
            for group in range(start, stop):
                ids = [f"g{group}v{version}" for version in range(4)]
                for name in ids:
                    node(store, name)
                for old, new in zip(ids, ids[1:]):
                    store.create_connection(new, old, "supersedes")

        def read_work():
            steps = 0

            def progress():
                nonlocal steps
                steps += 1
                return 0

            store.connection.set_progress_handler(progress, 1)
            try:
                pairs = store.list_supersedes_pairs(["g0v0", "g0v3"])
            finally:
                store.connection.set_progress_handler(None, 0)
            assert pairs == [("g0v1", "g0v0"), ("g0v3", "g0v2")]
            return steps

        add_groups(0, 12)
        small = read_work()
        add_groups(12, 120)
        large = read_work()
        # A type-wide index scan would grow with the tenfold independent history.
        assert large < small * 2


@pytest.mark.parametrize("selected", [[], ["a"], ["d"], ["a", "d"], ["a", "b", "c", "d"]])
def test_standalone_rank_and_replay_preserve_full_graph_result(tmp_path, selected):
    with MemoryStore(tmp_path / "pairs.sqlite3") as store:
        for name in ["a", "b", "c", "d", "outside", "x", "y"]:
            node(store, name, scope="project:other" if name == "outside" else "project:focus")
        store.soft_delete_node("c")
        pairs = [("b", "a"), ("c", "b"), ("d", "c"), ("a", "d"), ("outside", "d"), ("x", "y")]
        for source, target in pairs:
            store.create_connection(source, target, "supersedes")
        mapping = {}
        for source, target in pairs:
            mapping.setdefault(target, []).append(source)
        full = ({target: tuple(sorted(sources)) for target, sources in mapping.items()},
                {source for source, _ in pairs})
        candidates = {
            name: _Candidate(node=store.get_node(name), bm25_score=0.9, vector_score=0.7)
            for name in selected
        }
        plan = ScopePlan(requested_scope="project:focus", scopes=("project:focus", "global"))
        service = MemoryRecallService(store, embedder=Encoder())
        replay = make_ranking_service(store.get_retrieval_weights, pairs + pairs[:1])
        for modes in ({}, {"causal_mode": True}, {"decision_mode": True}, {"demotions": {"d": 0.1}}):
            expected = service.rank_candidates(candidates, plan, supersedes=full, **modes)
            assert [asdict(r) for r in service.rank_candidates(candidates, plan, **modes)] == [asdict(r) for r in expected]
            assert [asdict(r) for r in replay.rank_candidates(candidates, plan, **modes)] == [asdict(r) for r in expected]


def test_recall_reads_once_and_observes_external_changes_and_rollback(tmp_path, monkeypatch):
    path = tmp_path / "pairs.sqlite3"
    with MemoryStore(path) as store, MemoryStore(path) as writer:
        node(store, "stale")
        node(store, "correction", scope="project:other")
        service = MemoryRecallService(store, embedder=Encoder())
        original = store.list_supersedes_pairs
        calls = []

        def read(ids):
            ids = tuple(ids)
            calls.append(ids)
            return original(ids)

        monkeypatch.setattr(store, "list_supersedes_pairs", read)

        def recall():
            calls.clear()
            result = service.memory_recall(
                "Harbor", scope="project:focus", depth=0, max_results=1,
                log_access=False, log_event=False,
            )
            assert calls == [("stale",)]
            return result[0]

        before = recall()
        edge = writer.create_connection("correction", "stale", "supersedes")
        corrected = recall()
        assert corrected.superseded and corrected.score < before.score
        store.connection.execute("BEGIN")
        store.connection.execute("DELETE FROM connections WHERE id = ?", (edge.id,))
        assert not recall().superseded
        store.connection.rollback()
        assert recall() == corrected
        writer.delete_connection(edge.id)
        assert recall() == before
