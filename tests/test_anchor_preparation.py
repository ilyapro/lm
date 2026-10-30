"""Anchor matching reuses preparation without freezing live search decisions."""

import math
import random
import sqlite3
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
        if not numpy:
            rows = service._anchor_vectors.by_width[2].matrix
            assert sum(len(row) * row.itemsize for row in rows) == 4 * 4
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


def test_scope_and_width_changes_reuse_prepared_rows_without_sql(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        anchor(store, "global", [1.0, 0.0])
        anchor(store, "project", [1.0, 0.0], "project:demo")
        anchor(store, "wide", [1.0, 0.0, 0.0], "project:demo")
        service = retrieval.MemoryRecallService(store)
        match(service, [1.0, 0.0])
        cache = service._anchor_vectors
        matrix = weakref.ref(cache.by_width[2].matrix)
        assert sum(c.matrix.nbytes for c in cache.by_width.values()) == 2 * 4
        assert cache.pending == {2: {"project:demo"}}
        match(service, [1.0, 0.0], ["project:demo"])
        assert matrix() is None
        match(service, [1.0, 0.0, 0.0], ["project:demo"])
        assert cache.pending == {2: set(), 3: set()}
        assert sum(c.matrix.nbytes for c in cache.by_width.values()) == 7 * 4
        matrix = weakref.ref(cache.by_width[2].matrix)

        def no_preparation(*args, **kwargs):
            pytest.fail("scope or width selection must reuse the prepared corpus")

        with monkeypatch.context() as guard:
            guard.setattr(store, "iter_query_anchor_vectors", no_preparation)
            guard.setattr(store, "query_anchor_vector_layout", no_preparation)
            guard.setattr(retrieval, "_prepare_anchor_vectors", no_preparation)
            for scopes, query, expected in (
                (["project:demo", "global"], [1.0, 0.0], {"project", "global"}),
                (["global", "project:demo"], [1.0, 0.0], {"global", "project"}),
                (["project:demo", "project:demo"], [1.0, 0.0], {"project"}),
                (["project:demo"], [1.0, 0.0, 0.0], {"wide"}),
                (["project:absent"], [1.0, 0.0], set()),
                (None, [1.0, 0.0], {"global", "project"}),
            ):
                assert {m.anchor.id for m in match(service, query, scopes)} == expected
                assert service._anchor_vectors is cache
        assert cache.by_width[2].matrix.dtype == np.dtype("<f4")
        del cache
        store.decay_query_anchor("global", "inactive")
        match(service, [1.0, 0.0])
        assert matrix() is None


@pytest.mark.parametrize("numpy", [True, False], ids=["numpy", "python"])
def test_native_recall_decodes_only_consumed_scopes_and_width(tmp_path, monkeypatch, numpy):
    if not numpy:
        monkeypatch.setattr(qa, "_np", None)

    class Encoder:
        def embed(self, text):
            return [1.0, 0.0, 0.0]

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.create_node(node_id="alpha", level="concept", content="orbital telemetry drift",
                          context={"scope": "project:alpha"}, embedding=[1.0, 0.0, 0.0])
        anchor(store, "healthy", [1.0, 0.0, 0.0], "project:alpha")
        store.upsert_query_anchor_edge("healthy", "alpha", weight=1.0)
        anchor(store, "unconsumed", [1.0, 0.0, 0.0], "project:foreign")
        service = retrieval.MemoryRecallService(store, embedder=Encoder())

        def recall(retriever):
            return retriever.memory_recall("orbital telemetry drift", scope="project:alpha",
                                           max_results=2, depth=1, repair_pending=False)

        expected = recall(service)
        assert [r.node.id for r in expected] == ["alpha"]
        original = store.iter_query_anchor_vectors
        reads = []

        def observed(*args, **kwargs):
            for row in original(*args, **kwargs):
                reads.append(row[:3])
                yield row

        monkeypatch.setattr(store, "iter_query_anchor_vectors", observed)
        for scope, width in (("project:foreign", 3), ("project:alpha", 2)):
            with store.transaction():
                store.connection.execute(
                    "UPDATE query_anchors SET embedding=X'00', scope=?, dimensions=? "
                    "WHERE id='unconsumed'", (scope, width),
                )
            for retriever in (service, retrieval.MemoryRecallService(store, embedder=Encoder())):
                reads.clear()
                result = recall(retriever)
                assert [r.node.id for r in result] == ["alpha"]
                assert reads == [("healthy", "project:alpha", 3)]
                assert match(retriever, [1.0, 0.0, 0.0], "project:alpha") == qa.match_anchors(
                    store, [1.0, 0.0, 0.0], "project:alpha",
                )
                with pytest.raises(ValueError):
                    match(retriever, [1.0] + [0.0] * (width - 1), None)
                assert [r.node.id for r in recall(retriever)] == ["alpha"]
        with store.transaction():
            store.connection.execute("UPDATE query_anchors SET dimensions=3 WHERE id='unconsumed'")
        events = store.connection.execute("SELECT COUNT(*) FROM recall_events").fetchone()[0]
        for retriever in (service, retrieval.MemoryRecallService(store, embedder=Encoder())):
            with pytest.raises(ValueError):
                recall(retriever)
        with pytest.raises(ValueError):
            qa.match_anchors(store, [1.0, 0.0, 0.0], "project:alpha")
        assert store.connection.execute("SELECT COUNT(*) FROM recall_events").fetchone()[0] == events


def test_consumed_vector_sql_failure_propagates(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        anchor(store, "a", [1.0, 0.0])
        service = retrieval.MemoryRecallService(store)

        def deny_vector(action, table, column, *_):
            return (sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ
                    and table == "query_anchors" and column == "embedding" else sqlite3.SQLITE_OK)

        store.connection.set_authorizer(deny_vector)
        try:
            with pytest.raises(sqlite3.DatabaseError, match="embedding"):
                match(service, [1.0, 0.0])
        finally:
            store.connection.set_authorizer(None)
        assert [m.anchor.id for m in match(service, [1.0, 0.0])] == ["a"]


def test_missing_scope_fill_uses_one_revision_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as reader, MemoryStore(path) as writer:
        anchor(writer, "a", [1.0, 0.0], "project:alpha")
        anchor(writer, "b", [1.0, 0.0], "project:beta")
        anchor(writer, "c", [1.0, 0.0], "global")
        service = retrieval.MemoryRecallService(reader)
        original = reader.query_anchor_vector_layout

        def layout(width):
            rows = original(width)
            with writer.transaction():
                writer.connection.execute("UPDATE query_anchors SET scope='project:beta' WHERE id='a'")
            return rows

        with monkeypatch.context() as guard:
            guard.setattr(reader, "query_anchor_vector_layout", layout)
            assert [m.anchor.id for m in match(service, [1.0, 0.0], "project:beta")] == ["b"]
        assert [m.anchor.id for m in match(service, [1.0, 0.0], "project:beta")] == ["a", "b"]
        source = service._anchor_vector_source()
        with writer.transaction():
            writer.connection.execute("UPDATE query_anchors SET scope='global' WHERE id='a'")
        assert [m.anchor.id for m in qa.match_anchors(source, [1.0, 0.0], "global")] == ["a", "c"]
        assert "global" in source.pending[2]


def test_scope_selection_preserves_scan_order_and_uses_contiguous_views(tmp_path):
    np = pytest.importorskip("numpy")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for name, scope in (("a", "global"), ("b", "global"),
                            ("c", "project:demo"), ("d", "project:other")):
            anchor(store, name, [1.0, 0.0], scope)
        service = retrieval.MemoryRecallService(store)
        source = service._anchor_vector_source()
        corpus = source.query_anchor_corpus(2)
        assert corpus.ids == ("a", "b", "c", "d")
        selected = corpus.select(["global", "project:demo"])
        assert np.shares_memory(corpus.matrix[selected], corpus.matrix)
        selected = corpus.select(["project:demo", "global"])
        assert [corpus.ids[index] for index in selected] == ["c", "a", "b"]
        assert not np.shares_memory(corpus.matrix[selected], corpus.matrix)
        assert source.query_anchor_corpus(2) is corpus


@pytest.mark.parametrize("numpy", [True, False], ids=["numpy", "python"])
def test_lazy_scope_fills_preserve_full_corpus_thresholds_and_order(tmp_path, monkeypatch, numpy):
    np = pytest.importorskip("numpy")
    if not numpy:
        monkeypatch.setattr(qa, "_np", None)
    rng = random.Random(7304)
    scopes = ("global", "project:alpha", "project:beta")
    widths = (3, 17, 384)
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for index in range(69):
            width = widths[index % 3]
            vector = [rng.uniform(-1.0, 1.0) for _ in range(width)]
            anchor(store, f"a{index:03d}", vector, rng.choice(scopes))
        rows = list(store.iter_query_anchor_vectors())
        corpora = {width: qa._prepare_anchor_vectors(rows, width, index_scopes=True)
                   for width in widths}

        class FullCorpus:
            def query_anchor_corpus(self, width, scopes):
                return corpora[width]

            def __getattr__(self, name):
                return getattr(store, name)

        full = FullCorpus()
        service = retrieval.MemoryRecallService(store)
        for selection in (["project:beta"], ["global"], ["project:alpha", "global"],
                          ["project:beta", "global", "project:alpha"], None,
                          ["project:beta", "project:beta"], ["project:absent"]):
            for width in widths:
                vector = [rng.uniform(-1.0, 1.0) for _ in range(width)]
                expected = qa.match_anchors(full, vector, selection, limit=100, min_similarity=-1.0)
                assert match(service, vector, selection, limit=100, min_similarity=-1.0) == expected
                for item in expected[:3]:
                    for threshold in (
                        float(np.nextafter(np.float32(item.similarity), np.float32(-np.inf))),
                        item.similarity,
                        float(np.nextafter(np.float32(item.similarity), np.float32(np.inf))),
                    ):
                        for limit in (1, 5, 100):
                            assert match(service, vector, selection, limit=limit,
                                         min_similarity=threshold) == qa.match_anchors(
                                full, vector, selection, limit=limit, min_similarity=threshold,
                            )


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
