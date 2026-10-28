"""One query-anchor observation serves graph seeds and relative demotions."""

from collections import Counter

import pytest

from living_memory import query_anchors as qa
from living_memory import irrelevance
from living_memory.irrelevance import QUERY_IRRELEVANCE_SCHEMA_SQL
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore


class Encoder:
    def embed(self, text):
        return [0.0, 1.0] if text == "unrelated" else [1.0, 0.0]


class IndependentMatches(MemoryRecallService):
    """Control: each consumer matches independently against identical inputs."""

    def _collect_query_demotions(self, plan, query_embedding, **kwargs):
        return super()._collect_query_demotions(plan, query_embedding)


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    monkeypatch.setenv("LM_QUERY_IRRELEVANCE_FACTOR", "0.5")
    for key in ("LM_QUERY_IRRELEVANCE_FULL_COSINE", "LM_QUERY_IRRELEVANCE_MARK_WEIGHT",
                "LM_RECALL_MIN_SCORE", "LM_RECALL_SCHEMA_DEDUP", "LM_HUB_SUPPRESSION_FACTOR"):
        monkeypatch.delenv(key, raising=False)


def populate(store):
    for name, scope, vector in (
        ("semantic", "project:demo", [1.0, 0.0]),
        ("anchor-only", "project:demo", [0.0, 0.0]),
        ("stale", "project:demo", [0.8, 0.6]),
        ("correction", "project:demo", [0.7, 0.7]),
        ("outside", "project:other", [1.0, 0.0]),
        ("decayed", "project:demo", [1.0, 0.0]),
    ):
        store.create_node(level="concept", content=name, node_id=name,
                          context={"scope": scope}, embedding=vector)
    store.soft_delete_node("decayed", "inactive")
    for index in range(7):
        name = f"anchor-{index}"
        store.insert_query_anchor(scope="project:demo", query=name, fingerprint=name,
                                  anchor_id=name, embedding=[1.0, max(0, index - 1) * 0.1])
        for target in ("semantic", "anchor-only", "stale", "outside", "decayed"):
            store.upsert_query_anchor_edge(name, target, weight=0.5 + index * 0.05)
    for name, scope, vector in (
        ("foreign", "project:other", [1.0, 0.0]),
        ("wrong-width", "project:demo", [1.0, 0.0, 0.0]),
        ("zero", "project:demo", [0.0, 0.0]),
    ):
        store.insert_query_anchor(scope=scope, query=name, fingerprint=name,
                                  anchor_id=name, embedding=vector)
    store.create_connection("correction", "stale", "supersedes")
    with store.connection:
        store.connection.execute(QUERY_IRRELEVANCE_SCHEMA_SQL)
        store.connection.executemany(
            "INSERT INTO query_irrelevance "
            "(anchor_id, node_id, weight, marks, created_at, updated_at) "
            "VALUES (?, ?, ?, 1, 't', 't')",
            [(f"anchor-{index}", target, 0.25 + index * 0.1)
             for index in range(7) for target in ("semantic", "anchor-only")],
        )


def recall(service, *, query="situation", depth=1, scope="project:demo"):
    result = service.memory_recall(query, scope=scope, depth=depth, max_results=3,
                                   log_access=False, log_event=False)
    return ([row.to_dict() for row in result],
            [row.to_dict() for row in service.last_residual])


def count_work(monkeypatch, store):
    counts = Counter()
    for obj, name in ((qa, "_rank_anchors"), (store, "get_query_anchor"),
                      (irrelevance, "has_active_irrelevance"),
                      (store, "list_query_anchor_edges"), (store, "query_anchor_revision")):
        original = getattr(obj, name)

        def counted(*args, _original=original, _name=name, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(obj, name, counted)
    return counts


@pytest.mark.parametrize("numpy", [True, False], ids=["numpy", "python"])
@pytest.mark.parametrize("depth,seeding", [(0, True), (1, False), (1, True),
                                           (2, True), ("causal", True), ("decision", True)])
@pytest.mark.parametrize("scope", ["project:demo", None])
def test_shared_and_independent_observations_agree(tmp_path, monkeypatch, numpy, depth, seeding, scope):
    if not numpy:
        monkeypatch.setattr(qa, "_np", None)
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        populate(store)
        counts = count_work(monkeypatch, store)
        control = IndependentMatches(store, embedder=Encoder(), anchor_seeding=seeding)
        expected = recall(control, depth=depth, scope=scope)
        independent = counts.copy()
        counts.clear()
        shared = MemoryRecallService(store, embedder=Encoder(), anchor_seeding=seeding)
        assert recall(shared, depth=depth, scope=scope) == expected
        graph = depth != 0 and seeding
        assert counts["_rank_anchors"] == 1
        assert independent["_rank_anchors"] == (2 if graph else 1)
        assert counts["get_query_anchor"] * (2 if graph else 1) == independent["get_query_anchor"]
        # Cold preparation rechecks the revision inside its vector-read snapshot.
        assert counts["query_anchor_revision"] == 2
        assert independent["query_anchor_revision"] == (3 if graph else 2)
        assert counts["list_query_anchor_edges"] == independent["list_query_anchor_edges"]
        assert bool(counts["list_query_anchor_edges"]) == graph
        ids = [row["node"]["id"] for rows in expected for row in rows]
        assert "decayed" not in ids
        if scope:
            assert "outside" not in ids
        if graph:
            assert "anchor-only" in ids
            assert "correction" in ids


@pytest.mark.parametrize("case", ["off", "audit", "factor-one", "no-table", "cancelled",
                                  "no-anchors", "no-match", "empty-vector"])
@pytest.mark.parametrize("graph", [True, False])
def test_disabled_and_empty_paths_do_not_gain_work(tmp_path, monkeypatch, case, graph):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        populate(store)
        query = "situation"
        if case in ("off", "audit"):
            monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", case)
        elif case == "factor-one":
            monkeypatch.setenv("LM_QUERY_IRRELEVANCE_FACTOR", "1")
        elif case in ("no-table", "cancelled", "no-anchors"):
            with store.connection:
                store.connection.execute({"no-table": "DROP TABLE query_irrelevance",
                                          "cancelled": "UPDATE query_irrelevance SET weight = 0",
                                          "no-anchors": "UPDATE query_anchors SET decayed = 1"}[case])
        elif case == "no-match":
            query = "unrelated"
        elif case == "empty-vector":
            monkeypatch.setattr(Encoder, "embed", lambda self, text: [])
        counts = count_work(monkeypatch, store)
        control = IndependentMatches(store, embedder=Encoder())
        expected = recall(control, query=query, depth=int(graph))
        independent = counts.copy()
        counts.clear()
        assert recall(MemoryRecallService(store, embedder=Encoder()),
                      query=query, depth=int(graph)) == expected
        assert all(count <= independent[key] for key, count in counts.items())
        assert counts["_rank_anchors"] == int(case not in ("no-anchors", "empty-vector")
                                              and (graph or case == "no-match"))
        if not graph:
            assert counts["list_query_anchor_edges"] == 0


@pytest.mark.parametrize("second_connection", [False, True])
def test_next_recall_observes_membership_edges_and_feedback(tmp_path, second_connection):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as reader, MemoryStore(path) as other:
        writer = other if second_connection else reader
        populate(writer)
        shared = MemoryRecallService(reader, embedder=Encoder())
        before = recall(shared)
        with writer.connection:
            writer.connection.execute("UPDATE query_irrelevance SET weight = 0")
        after = recall(shared)
        assert after != before
        assert after == recall(IndependentMatches(reader, embedder=Encoder()))
        with writer.connection:
            writer.connection.execute("DELETE FROM query_anchor_edges WHERE target_id = 'anchor-only'")
        after_edges = recall(shared)
        assert after_edges != after
        assert after_edges == recall(IndependentMatches(reader, embedder=Encoder()))
        for index in range(7):
            writer.decay_query_anchor(f"anchor-{index}", "inactive")
        empty = recall(shared)
        assert empty == recall(IndependentMatches(reader, embedder=Encoder()))
        writer.reinforce_query_anchor("anchor-0")
        writer.upsert_query_anchor_edge("anchor-0", "anchor-only", weight=1.0)
        restored = recall(shared)
        assert restored != empty
        assert restored == recall(IndependentMatches(reader, embedder=Encoder()))
