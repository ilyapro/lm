"""File evidence has the same source identity on write, backfill and recall."""

import pytest

from living_memory.edge_backfill import backfill_edges
from living_memory.edge_derivation import derive_edges_for_new_trace
from living_memory.replay import snapshot_evidence
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore


@pytest.mark.parametrize("left,right,shared", [
    ({"scope": "project:a"}, {"scope": "project:b"}, False),
    ({"scope": "project:a"}, {"scope": "project:a"}, True),
    ({"repo": "/code/a"}, {"repo": "/other/a"}, False),
    ({"repo": "/code/a"}, {"scope": "project:a"}, False),
    ({"scope": "project:a", "repo": "/code/one"}, {"scope": "project:a", "repo": "/code/two"}, False),
    ({"scope": "project:a", "repo": "/code/x"}, {"scope": "project:b", "repo": "/code/x"}, True),
    ({"worktree": "/code/x/.worktrees/first"}, {"repo": "/code/x/.worktrees/second"}, True),
    ({"worktree": "/code/x/.worktrees/first", "repo": "/code/y"}, {"repo": "/code/y"}, False),
    ({"repo": "x", "worktree": "/src/y"}, {"repo": "/src/y"}, False),
    ({"project": "a"}, {"scope": "project:a"}, True),
    ({}, {}, False),
    ({"scope": "session:s"}, {"scope": "session:s"}, False),
    ({"repo": ["x"]}, {"repo": ["x"]}, False),
    ({"project": {"a": 1}}, {"project": {"a": 1}}, False),
    ({"files": ["/lib/parser.py"]}, {"files": ["/lib/parser.py"]}, True),
    ({"repo": "/lib", "files": ["./parser.py"]}, {"files": ["/lib/parser.py"]}, True),
])
def test_derivation_backfill_and_stored_evidence_agree(tmp_path, left, right, shared):
    with MemoryStore(tmp_path / "isolated.db") as store:
        a = store.append_trace("orchid pollen", {"files": ["README.md"], **left})
        b = store.append_trace("cobalt deployment", {"files": ["README.md"], **right})
        online = derive_edges_for_new_trace(store, a)
        assert len(online) == int(shared)
        if online:
            assert online[0].weight == .65
            store.delete_connection(online[0].id)
        assert backfill_edges(store, apply=True)["totals"]["new_rows"] == int(shared)
        for edge in store.list_connections():
            store.delete_connection(edge.id)
        # Older backfill rows persisted their raw spellings, before validation.
        edge = store.create_connection(a.id, b.id, "related", weight=.65, metadata={
            "rule_basis": "shared_files", "basis": "provenance_derivation",
            "files": a.context["files"],
        })
        assert bool(store.list_connections()) is shared
        assert store._has_graph_evidence(a.scope) is shared
        assert snapshot_evidence(store.connection)(a.scope)[1] is shared
        assert store.get_connection(edge.id) == edge


def test_fanout_counts_each_repository_independently(tmp_path):
    with MemoryStore(tmp_path / "isolated.db") as store:
        for i in range(21):
            store.append_trace(f"Foreign {i}", {"scope": "project:b", "files": ["README.md"]})
        a, b = [store.append_trace(text, {"scope": "project:a", "files": ["README.md"]})
                for text in ("orchid pollen", "moth wings")]
        edge, = derive_edges_for_new_trace(store, b)
        assert {edge.source_id, edge.target_id} == {a.id, b.id}
        store.delete_connection(edge.id)
        assert backfill_edges(store, apply=True)["totals"]["new_rows"] == 1
        assert store.list_connections()[0].weight == .65


@pytest.mark.parametrize("extra,weight,source", [
    ({}, None, None),
    ({"commit": "abc1234"}, .9, .9),
    ({"basis": "same_commit", "rule": "R4a"}, .9, .9),
    ({"kind": "content_reference"}, .9, .9),
    ({"kind": "derived_from"}, .6, .6),
    ({"basis": "cluster"}, .2, .2),
    ({"basis": "co_access"}, .25, None),
])
def test_historical_maximum_keeps_independent_contributions(tmp_path, extra, weight, source):
    files = ["one", "two", "three"]
    with MemoryStore(tmp_path / "isolated.db") as store:
        a, b = [store.append_trace(label, {"scope": "project:" + label, "files": files, "commit": "abc1234"})
                for label in ("orchid", "cobalt")]
        old = store.create_connection(a.id, b.id, "related", weight=.95,
                                      metadata={"rule": "R4b", "basis": "shared_files", "files": files, **extra})
        for source_only, expected in ((False, weight), (True, source)):
            edges = store.list_recall_connections([a.id], limit=1, source_only=source_only)[a.id]
            assert [edge.weight for edge in edges] == ([] if expected is None else [expected])
        assert store.get_connection(old.id) == old


def test_partial_shared_files_rank_before_neighbor_limit(tmp_path):
    files = ["README.md", "/shared/one", "/shared/two"]
    with MemoryStore(tmp_path / "isolated.db") as store:
        a, b, c = [store.append_trace(label, {"scope": "project:" + label, "files": files})
                   for label in ("orchid", "cobalt", "moth")]
        old = store.create_connection(a.id, b.id, "related", weight=.95,
                                      metadata={"basis": "shared_files", "files": files})
        genuine = store.create_connection(a.id, c.id, "related", weight=.85)
        for source_only in (False, True):
            edges = store.list_recall_connections([a.id], limit=2, source_only=source_only)[a.id]
            assert [(e.id, e.weight) for e in edges] == [(genuine.id, .85), (old.id, .8)]
            assert store.list_recall_connections([a.id], limit=1, source_only=source_only)[a.id] == [genuine]


class ZeroSemantic:
    def embed(self, text):
        return [0.0] * 384


def test_broad_recall_only_follows_shared_source_and_observes_changes(tmp_path):
    with MemoryStore(tmp_path / "isolated.db") as store:
        def note(text, scope, repo):
            return store.append_trace(text, {"scope": scope, "repo": repo, "files": ["README.md"]})

        seed = note("Orchid pollination moth wings", "project:a", "/src/orchid")
        other = note("Cobalt deployment", "project:b", "/src/deploy")
        control = note("Independent measurement", "project:c", "/src/orchid")
        for node in [other, control]:
            store.create_connection(seed.id, node.id, "related", weight=.65,
                                    metadata={"rule": "R4b", "basis": "shared_files", "files": ["README.md"]})
        service = MemoryRecallService(store, embedder=ZeroSemantic(), anchor_seeding=False)

        def scores():
            results = service.memory_recall("orchid pollination", log_event=False, log_access=False)
            return {r.node.id: r.graph_score for r in results}

        result = scores()
        assert other.id not in result
        assert result[control.id] == pytest.approx(.4225)
        store.connection.execute("UPDATE nodes SET context = json_set(context, '$.repo', '/src/orchid') WHERE id = ?", (other.id,))
        assert scores()[other.id] == pytest.approx(.4225)
