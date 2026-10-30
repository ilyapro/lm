"""Source qualification traverses admitted identities without ranking new bodies."""

from dataclasses import asdict
from math import prod

import pytest

import living_memory.retrieval as retrieval
from living_memory.models import REJECTED_ALTERNATIVE_KIND
from living_memory.retrieval import MemoryRecallService, _Candidate
from living_memory.scope import resolve_scope
from living_memory.storage import MemoryStore
from test_recall_graph_preparation import Encoder, anchor, node


PLAN = resolve_scope(scope="project:graph")


def _source_fixture(store):
    for name in ("discovery", "history-a", "history-b", "source", "target", "strong", "weak", "join", "later"):
        node(store, name, content="needle" if name == "source" else name)
    node(store, "foreign", scope="project:other")
    node(store, "dead")
    store.soft_delete_node("dead", "expired")
    store.create_connection("discovery", "history-a", "related", metadata={"basis": "co_access"})
    for left, right, weight in (
        ("history-a", "history-b", 1), ("history-b", "target", 1),
        ("source", "strong", 0.8), ("source", "weak", 0.4),
        ("strong", "join", 0.9), ("weak", "join", 0.9),
        ("join", "target", 0.7), ("join", "source", 0.1),
        ("source", "later", 0.3), ("source", "foreign", 1),
        ("source", "dead", 1),
    ):
        store.create_connection(left, right, "requires", weight=weight)
    store.create_connection("source", "target", "contradicts", weight=0.6,
                            metadata={"kind": REJECTED_ALTERNATIVE_KIND})


def _candidates(store):
    return {
        name: _Candidate(node=store.get_node(name), bm25_score=score,
                         lexical_score=float(name == "source"))
        for name, score in (("discovery", 1), ("source", 0.2), ("target", 0.1))
    }


class SeededService(MemoryRecallService):
    """Fix channel admission while exercising the complete recall/ranking pipeline."""

    def _collect_bm25(self, query, plan, candidates, *, max_results):
        candidates.update(_candidates(self.store))

    def _collect_vector(self, query_embedding, plan, candidates, *, max_results):
        pass

    def rank_candidates(self, candidates, *args, **kwargs):
        self.candidates = {name: asdict(value) for name, value in candidates.items()}
        return super().rank_candidates(candidates, *args, **kwargs)


def _outcome(service, depth):
    delivered = service.memory_recall(
        "needle", scope="project:graph", depth=depth, max_results=2,
        log_access=False, log_event=False,
    )
    relevance = service.last_query_relevance
    return {
        "delivered": [asdict(value) for value in delivered],
        "residual": [asdict(value) for value in service.last_residual],
        "ranked": [asdict(value) for value in relevance.ranked],
        "qualified": [relevance.passes(value) for value in relevance.ranked],
        "candidates": service.candidates,
    }


@pytest.mark.parametrize("depth", [0, 1, 2, 3, "causal", "decision"])
def test_full_ordered_recall_matches_materialized_admission(tmp_path, monkeypatch, depth):
    monkeypatch.setattr(retrieval, "GRAPH_SEED_LIMIT", 1)
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "0.35")
    with MemoryStore(tmp_path / "recall.sqlite3") as store:
        _source_fixture(store)
        anchor(store, "later")
        identity_reader = store.get_recall_node_ids
        requests = []

        def identities(ids, plan):
            ids = tuple(ids)
            requests.append(ids)
            return identity_reader(ids, plan)

        monkeypatch.setattr(store, "get_recall_node_ids", identities)
        actual = _outcome(SeededService(store, embedder=Encoder()), depth)
        projected_requests = requests[:]
        requests.clear()

        def materialized(ids, plan):
            ids = tuple(ids)
            requests.append(ids)
            return set(store.get_recall_nodes(ids, plan))

        monkeypatch.setattr(store, "get_recall_node_ids", materialized)
        expected = _outcome(SeededService(store, embedder=Encoder()), depth)
        assert actual == expected
        assert requests == projected_requests
        if depth in (1, 2, 3, "causal"):
            assert any("strong" in ids for ids in requests)
            assert actual["candidates"]["later"]["source_graph_score"] == 0
        if depth == 3:
            target = actual["candidates"]["target"]
            assert target["source_graph_score"] == pytest.approx(0.8 * 0.9 * 0.7 * 0.85**3 * 0.72**3)
            assert target["graph_score"] == pytest.approx(0.65 * 0.85**2 * 0.72**3)
            assert target["path"] == ("discovery", "history-a", "history-b", "target")
        if depth == "decision":
            assert actual["candidates"]["target"]["source_graph_score"] == pytest.approx(0.6)


@pytest.mark.parametrize("depth", range(5))
def test_source_walk_matches_small_path_oracle_without_adding_candidates(tmp_path, depth):
    # Enumerating this small graph independently accounts for weaker routes,
    # equal routes, repeated frontiers and cycles without using the BFS state.
    adjacency = {
        "root": [("weak", 0.4), ("strong", 0.8), ("tie", 0.8),
                 ("foreign", 1), ("dead", 1), ("missing", 1)],
        "weak": [("join", 0.9)],
        "strong": [("join", 0.9)],
        "tie": [("join", 0.9)],
        "join": [("target", 0.7), ("root", 1)],
        "target": [("tail", 0.8), ("strong", 0.1)],
        "tail": [], "foreign": [("target", 1)], "dead": [("target", 1)],
        "missing": [("target", 1)],
    }
    admitted = set(adjacency) - {"foreign", "dead", "missing"}
    paths = [(('root',), ())]
    expected = {name: 0.0 for name in ("root", "target", "tail")}
    while paths:
        path, weights = paths.pop()
        if len(weights) == depth:
            continue
        for target, weight in adjacency[path[-1]]:
            if target in path or target not in admitted:
                continue
            route = (*weights, weight)
            hops = len(route)
            score = prod(route) * 0.72 ** (hops * (hops - 1) // 2)
            if target in expected:
                expected[target] = max(expected[target], min(1.5, score))
            paths.append(((*path, target), route))

    with MemoryStore(tmp_path / "oracle.sqlite3") as store:
        for name in set(adjacency) - {"missing"}:
            node(store, name, scope="project:other" if name == "foreign" else "project:graph")
        store.soft_delete_node("dead", "expired")
        candidates = {
            name: _Candidate(node=store.get_node(name), graph_score=0.2, path=("discovery", name))
            for name in expected
        }
        before = {name: asdict(value) for name, value in candidates.items()}
        MemoryRecallService(store, embedder=Encoder())._walk_graph(
            PLAN, candidates, [("root", 1)], max_depth=depth,
            neighbors_for_nodes=lambda ids: {name: adjacency[name] for name in ids},
            anchor_pass=False, source_pass=True,
        )
        assert set(candidates) == set(expected)
        for name, candidate in candidates.items():
            assert candidate.source_graph_score == pytest.approx(expected[name])
            result = asdict(candidate)
            result["source_graph_score"] = 0.0
            assert result == before[name]


@pytest.mark.parametrize("body", ["content", "context", "provenance"])
def test_source_only_corrupt_bodies_are_not_required(tmp_path, monkeypatch, body):
    monkeypatch.setattr(retrieval, "GRAPH_SEED_LIMIT", 1)
    with MemoryStore(tmp_path / "corrupt.sqlite3") as store:
        _source_fixture(store)
        service = MemoryRecallService(store, embedder=Encoder())

        def collect():
            candidates = _candidates(store)
            service._collect_graph(PLAN, candidates, max_depth=3, causal_mode=False)
            return {name: asdict(value) for name, value in candidates.items()}

        expected = collect()
        assert expected["target"]["source_graph_score"] > 0
        assert "strong" not in expected
        invalid_text = "x'22ff22'" if body == "context" else "x'ff'"
        with store.transaction():
            store.connection.execute(
                f"UPDATE nodes SET {body} = CAST({invalid_text} AS TEXT) WHERE id = 'strong'",
            )
        assert collect() == expected


@pytest.mark.parametrize("entry", ["ordinary", "anchor"])
def test_required_selected_bodies_still_fail(tmp_path, monkeypatch, entry):
    monkeypatch.setattr(retrieval, "GRAPH_SEED_LIMIT", 1)
    with MemoryStore(tmp_path / "required.sqlite3") as store:
        _source_fixture(store)
        if entry == "ordinary":
            store.create_connection("discovery", "strong", "requires")
        with store.transaction():
            store.connection.execute("UPDATE nodes SET provenance = '[]' WHERE id = 'strong'")
        with pytest.raises(TypeError):
            MemoryRecallService(store, embedder=Encoder())._collect_graph(
                PLAN, _candidates(store), max_depth=3, causal_mode=False,
                anchor_seeds={"strong": 1} if entry == "anchor" else {},
            )


@pytest.mark.parametrize("change", ["scope", "decay", "delete"])
def test_source_admission_observes_changes_after_adjacency(tmp_path, change):
    with MemoryStore(tmp_path / "fresh.sqlite3") as store:
        for name in ("root", "intermediate", "target"):
            node(store, name)
        candidates = {name: _Candidate(node=store.get_node(name)) for name in ("root", "target")}

        def neighbors(ids):
            if "root" in ids:
                if change == "scope":
                    store.update_node("intermediate", context={"scope": "project:other"})
                elif change == "decay":
                    store.soft_delete_node("intermediate", "expired")
                else:
                    store.delete_node("intermediate")
            graph = {"root": [("intermediate", 1)], "intermediate": [("target", 1)], "target": []}
            return {name: graph[name] for name in ids}

        MemoryRecallService(store, embedder=Encoder())._walk_graph(
            PLAN, candidates, [("root", 1)], max_depth=2, neighbors_for_nodes=neighbors,
            anchor_pass=False, source_pass=True,
        )
        assert candidates["target"].source_graph_score == 0
        assert set(candidates) == {"root", "target"}


def test_source_admission_misses_are_cached_only_for_one_walk(tmp_path, monkeypatch):
    with MemoryStore(tmp_path / "miss.sqlite3") as store:
        for name in ("root", "branch", "target"):
            node(store, name)
        graph = {
            "root": [("missing", 1), ("branch", 0.8)],
            "branch": [("missing", 1)], "missing": [("target", 1)], "target": [],
        }
        original = store.get_recall_node_ids
        reads = []

        def identities(ids, plan):
            ids = tuple(ids)
            reads.extend(ids)
            return original(ids, plan)

        def neighbors(ids):
            if "branch" in ids and store.get_node("missing") is None:
                node(store, "missing")
            return {name: graph[name] for name in ids}

        monkeypatch.setattr(store, "get_recall_node_ids", identities)
        service = MemoryRecallService(store, embedder=Encoder())

        def walk():
            candidates = {name: _Candidate(node=store.get_node(name)) for name in ("root", "target")}
            service._walk_graph(
                PLAN, candidates, [("root", 1)], max_depth=3, neighbors_for_nodes=neighbors,
                anchor_pass=False, source_pass=True,
            )
            return candidates["target"].source_graph_score

        assert walk() == 0
        assert reads.count("missing") == 1
        assert walk() == pytest.approx(0.72)
        assert reads.count("missing") == 2
