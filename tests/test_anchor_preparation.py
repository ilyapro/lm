"""Anchor matching reuses preparation without freezing live search decisions."""

import math
import weakref

import pytest

from living_memory import query_anchors as qa
from living_memory import retrieval
from living_memory.embeddings import cosine_similarity
from living_memory.storage import MemoryStore, pack_chunk_embedding


def anchor(store, name, vector, scope="global"):
    return store.insert_query_anchor(
        anchor_id=name, scope=scope, query=name, fingerprint=name, embedding=vector,
    )


def match(service, vector, scopes="global", **kwargs):
    source = service._anchor_vector_source()
    return [] if source is None else qa.match_anchors(source, vector, scopes, **kwargs)


@pytest.mark.parametrize("numpy", [True, False], ids=["numpy", "python"])
def test_queries_reuse_preparation_until_vector_revision_changes(tmp_path, monkeypatch, numpy):
    if not numpy:
        monkeypatch.setattr(qa, "_np", None)
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        anchor(store, "a", [1.0, 0.0])
        anchor(store, "b", [0.0, 1.0])
        service = retrieval.MemoryRecallService(store)
        preparations = []
        prepare = retrieval._prepare_anchor_vectors

        def counted(*args, **kwargs):
            corpus = prepare(*args, **kwargs)
            preparations.append(corpus.ids)
            return corpus

        monkeypatch.setattr(retrieval, "_prepare_anchor_vectors", counted)
        assert [m.anchor.id for m in match(service, [1.0, 0.0])] == ["a"]
        for query in ([0.0, 1.0], [1.0, 1.0], [1.0, 0.0]):
            assert match(service, query) == qa.match_anchors(store, query, "global")
        assert preparations == [("a", "b")]
        store.reinforce_query_anchor("a")
        node = store.create_node(level="concept", content="target")
        store.upsert_query_anchor_edge("a", node.id, weight=0.75)
        current = match(service, [1.0, 0.0])[0]
        assert current.anchor.reinforcement_count == 1
        assert current.targets == ((node.id, 0.75),)
        assert preparations == [("a", "b")]
        with store.connection:
            store.connection.execute(
                "UPDATE query_anchors SET embedding=? WHERE id='b'",
                (pack_chunk_embedding([1.0, 0.0]),),
            )
        assert [m.anchor.id for m in match(service, [1.0, 0.0])] == ["a", "b"]
        assert preparations == [("a", "b"), ("a", "b")]


def test_warm_queries_do_not_rebuild_matrix_or_anchor_norms(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for index in range(17):
            anchor(store, f"a{index:02d}", [float(index), 1.0, 0.0])
        service = retrieval.MemoryRecallService(store)
        match(service, [1.0, 0.0, 0.0])
        original_norm = np.linalg.norm
        query_norms = []

        def norm(array, *args, **kwargs):
            assert array.ndim == 1, "warm matching must not recalculate corpus norms"
            query_norms.append(True)
            return original_norm(array, *args, **kwargs)

        def no_preparation(*args, **kwargs):
            pytest.fail("warm matching must not decode or rescan corpus vectors")

        monkeypatch.setattr(np.linalg, "norm", norm)
        monkeypatch.setattr(np, "frombuffer", no_preparation)
        monkeypatch.setattr(store, "iter_query_anchor_vectors", no_preparation)
        for query in ([1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 1.0, 0.0]):
            match(service, query)
        assert len(query_norms) == 3


def test_scope_and_width_changes_replace_the_single_prepared_matrix(tmp_path):
    np = pytest.importorskip("numpy")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        anchor(store, "global", [1.0, 0.0])
        anchor(store, "project", [1.0, 0.0], "project:demo")
        anchor(store, "wide", [1.0, 0.0, 0.0], "project:demo")
        service = retrieval.MemoryRecallService(store)
        for scopes, query, expected in (
            (["project:demo", "global"], [1.0, 0.0], ("project", "global")),
            (["global", "project:demo"], [1.0, 0.0], ("global", "project")),
            (["project:demo", "project:demo"], [1.0, 0.0], ("project",)),
            (["project:demo"], [1.0, 0.0, 0.0], ("wide",)),
            (["project:absent"], [1.0, 0.0], ()),
        ):
            old = service._anchor_vectors
            previous_matrix = (
                weakref.ref(old.prepared.matrix) if old and old.prepared.ids else None
            )
            result = match(service, query, scopes)
            assert {m.anchor.id for m in result} == set(expected)
            prepared = service._anchor_vectors.prepared
            assert prepared.ids == expected
            if expected:
                assert prepared.matrix.dtype == np.dtype("<f4")
                assert prepared.matrix.nbytes == len(expected) * len(query) * 4
            if previous_matrix:
                assert previous_matrix() is None
            # The local test reference must not keep the previous matrix alive.
            del prepared


@pytest.mark.parametrize("numpy", [True, False], ids=["numpy", "python"])
def test_exact_thresholds_ties_and_zero_vectors(tmp_path, monkeypatch, numpy):
    np = pytest.importorskip("numpy")
    if not numpy:
        monkeypatch.setattr(qa, "_np", None)
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        vectors = [[0.6, 0.8], [0.95, math.sqrt(1.0 - 0.95**2)], [0.0, 0.0]]
        for index, vector in enumerate(vectors):
            anchor(store, f"a{index}", vector)
        for index in range(7):
            anchor(store, f"tie{index}", [1.0, 0.0])
        service = retrieval.MemoryRecallService(store)
        vector = [1.0, 0.0]
        expected = qa.match_anchors(store, vector, "global", limit=100, min_similarity=-1.0)
        for item in expected:
            thresholds = (
                float(np.nextafter(np.float32(item.similarity), np.float32(-np.inf))),
                item.similarity,
                float(np.nextafter(np.float32(item.similarity), np.float32(np.inf))),
            )
            for threshold in thresholds:
                for limit in (1, 5, 100):
                    assert match(service, vector, min_similarity=threshold, limit=limit) == (
                        qa.match_anchors(store, vector, "global", min_similarity=threshold, limit=limit)
                    )
        assert match(service, [0.0, 0.0], min_similarity=-1.0) == []
        if not numpy:
            for item in expected:
                assert item.similarity == cosine_similarity(vector, item.anchor.embedding)
