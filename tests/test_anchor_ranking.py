"""Anchor and seed membership follows similarity descending, then anchor ID."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from living_memory import query_anchors as qa
from living_memory import retrieval
from living_memory.scope import ScopePlan
from living_memory.storage import MemoryStore


@pytest.mark.parametrize("numpy", [True, False], ids=["numpy", "python"])
@pytest.mark.parametrize("prepared", [True, False], ids=["prepared", "uncached"])
@pytest.mark.parametrize("scopes", [
    ("*",), ("global", "project:test"), ("project:test", "global"),
], ids=["all", "global-first", "project-first"])
@pytest.mark.parametrize("stronger", [False, True], ids=["all-tied", "cutoff-tied"])
@pytest.mark.parametrize("limit", [1, 5, 100])
def test_anchor_and_seed_membership(monkeypatch, numpy, prepared, scopes, stronger, limit):
    if numpy:
        pytest.importorskip("numpy")
    else:
        monkeypatch.setattr(qa, "_np", None)
    monkeypatch.setattr(retrieval, "ANCHOR_MATCH_LIMIT", limit)
    # Integer vectors give exact norms and cosines in both engines. The
    # stronger anchors have larger IDs so similarity must take precedence.
    query = [1.0] + [0.0] * (7 if stronger else 1)
    tied = [3.0] + [1.0] * 7 if stronger else query
    tied_score = 0.75 if stronger else 1.0
    population = [
        (f"a{index:03d}", "project:test" if index < 15 else "global", tied, tied_score)
        for index in range(30)
    ]
    if stronger:
        population.extend([
            ("z0", "project:test", query, 1.0), ("z1", "global", query, 1.0),
        ])
    expected = sorted(population, key=lambda item: (-item[3], item[0]))[:limit]
    with MemoryStore(":memory:") as store:
        for name, scope, vector, _score in population:
            store.insert_query_anchor(
                anchor_id=name, scope=scope, query=name, fingerprint=name, embedding=vector,
            )
            store.create_node(node_id=f"target-{name}", level="trace", content=f"Target {name}")
            store.upsert_query_anchor_edge(name, f"target-{name}", weight=1.0)
        service = retrieval.MemoryRecallService(store)
        if not prepared:
            monkeypatch.setattr(service, "_anchor_vector_source", lambda: store)
        plan = ScopePlan(scopes[0], scopes)
        for _ in range(2):  # Exercise both cold and warm prepared selection.
            matches = service._match_query_anchors(plan, query)
            seeds = service._collect_anchor_seeds(plan, query, matches=matches)
            assert {
                "anchors": [(item.anchor.id, item.similarity) for item in matches],
                "seeds": seeds,
            } == {
                "anchors": [(name, score) for name, _scope, _vector, score in expected],
                "seeds": {f"target-{name}": score for name, _scope, _vector, score in expected},
            }


def test_distinct_scores_materialize_only_limited_ids():
    np = pytest.importorskip("numpy")
    limit = 5
    with MemoryStore(":memory:") as store:
        for index in range(128):
            name = f"a{index:03d}"
            store.insert_query_anchor(
                anchor_id=name, scope="global", query=name, fingerprint=name,
                embedding=[float(index), 1.0],
            )
        source = retrieval.MemoryRecallService(store)._anchor_vector_source()
        corpus = source.query_anchor_corpus(2, ["global"])
        assert len(np.unique(corpus.matrix[:, 0] / corpus.safe_norms)) == len(corpus.ids)

        class CountedIds(tuple):
            reads = 0

            def __getitem__(self, index):
                self.reads += 1
                return super().__getitem__(index)

        ids = CountedIds(corpus.ids)
        counted = replace(corpus, ids=ids)
        ranked = qa._rank_anchors(
            SimpleNamespace(query_anchor_corpus=lambda *_: counted), [1.0, 0.0],
            ["global"], min_similarity=0.0, limit=limit,
        )
        assert [name for name, _score in ranked] == [f"a{index:03d}" for index in range(127, 122, -1)]
        assert ids.reads == limit
