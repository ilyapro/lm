"""The vector channel over chunk BLOBs: max-pool, cache validity, both math paths.

Covers the retrieval half of the chunking work. Storage's side (the codec, the
table, the write paths that keep chunks in step with content) is
``tests/test_chunk_store.py``; here the question is only what
``MemoryRecallService`` does with those rows.

Every node in these tests carries an explicit embedding, so the vectors are the
test's rather than the encoder's, and the assertions are on
``RecallResult.vector_score`` -- the one number this change produces -- rather
than on rank alone, which the rest of the fusion also moves.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import pytest

from living_memory import retrieval
from living_memory.chunking import TextChunk
from living_memory.embeddings import cosine_similarity
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore


SCOPE = "project:maxpool"
QUERY = "orbital telemetry drift"
#: The query direction. Every stored vector below is written relative to it, so
#: a cosine in an assertion is readable without running the arithmetic.
QUERY_VECTOR = [1.0, 0.0, 0.0]
ORTHOGONAL = [0.0, 1.0, 0.0]
#: cos(QUERY_VECTOR, .) == 0.8, comfortably above VECTOR_MATCH_THRESHOLD and
#: comfortably below STRONG_VECTOR_MATCH.
PARTIAL = [0.8, 0.6, 0.0]


class FixedEmbedder:
    """Maps the query to a known direction and everything else away from it.

    Node vectors are supplied explicitly at write time, so this only has to
    answer for query text and for whatever the lazy backfill decides to embed.
    """

    def embed(self, text: str) -> list[float]:
        return list(QUERY_VECTOR) if QUERY in text.lower() else list(ORTHOGONAL)


def make_chunk(text: str, index: int) -> TextChunk:
    """A chunk descriptor for a vector the test supplies directly.

    ``replace_node_chunks`` validates ordinals and stores the spans verbatim;
    it never re-reads them, so the spans only have to be self-consistent.
    """

    return TextChunk(
        text=text,
        chunk_index=index,
        token_start=index * 100,
        token_end=(index + 1) * 100,
        char_start=index * 100,
        char_end=(index + 1) * 100,
    )


def put_chunks(store: MemoryStore, node_id: str, vectors: Sequence[Sequence[float]]) -> None:
    """Replace one node's chunk set with exactly ``vectors``."""

    store.replace_node_chunks(
        node_id,
        [
            (make_chunk(f"{node_id} window {index}", index), list(vector))
            for index, vector in enumerate(vectors)
        ],
    )


def vector_scores(results: Sequence[Any]) -> dict[str, float]:
    return {result.node.id: result.vector_score for result in results}


def old_single_vector_scores(store: MemoryStore, scope: str, query: list[float]) -> dict[str, float]:
    """The pre-chunking vector channel, reimplemented from the code it replaced.

    ``_scan_scope_embeddings`` read ``nodes.embedding`` as JSON and scored one
    cosine per node, clipped at zero. The column is still there and still
    current for these fixtures, so it is a usable oracle for "a single-chunk
    node scores what it used to".
    """

    scores: dict[str, float] = {}
    for node_id, embedding_json in store.iter_embedding_rows(scope=scope):
        parsed = json.loads(embedding_json)
        if not parsed:
            continue
        scores[node_id] = max(0.0, cosine_similarity(query, parsed))
    return scores


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    with MemoryStore(tmp_path / "memory.sqlite3") as opened:
        yield opened


def add_node(
    store: MemoryStore,
    content: str,
    vector: Sequence[float],
    *,
    level: str = "trace",
) -> str:
    node = store.create_node(
        level=level,
        content=content,
        context={"scope": SCOPE, "agent": "tester"},
        embedding=list(vector),
    )
    return node.id


def recall(service: MemoryRecallService, **kwargs: Any) -> list[Any]:
    return service.memory_recall(QUERY, scope=SCOPE, depth=0, max_results=10, **kwargs)


def expected_score(best_cosine: float, chunk_count: int) -> float:
    """What the channel should report: the best window's cosine, less the
    length-bias correction for having ``chunk_count`` windows to try."""

    return max(
        0.0,
        best_cosine - retrieval.LENGTH_BIAS_LOG2_COEFFICIENT * math.log2(chunk_count),
    )


def test_node_score_is_the_max_chunk_cosine_not_the_mean(store: MemoryStore) -> None:
    """A node whose best chunk matches beats a node with the better average.

    This is the whole substitution: ``best_chunk`` has one window pointing
    straight at the query and one pointing away (max 1.0, mean 0.5), while
    ``best_average`` has two windows at 0.8 each (max 0.8, mean 0.8). Under
    mean-pool the ordering is the other way round, so the assertion fails on
    any aggregation that is not the maximum.
    """

    best_chunk = add_node(store, "alpha node", ORTHOGONAL)
    best_average = add_node(store, "beta node", ORTHOGONAL)
    put_chunks(store, best_chunk, [ORTHOGONAL, QUERY_VECTOR])
    put_chunks(store, best_average, [PARTIAL, PARTIAL])

    service = MemoryRecallService(store, embedder=FixedEmbedder())
    scores = vector_scores(recall(service))

    assert scores[best_chunk] == pytest.approx(expected_score(1.0, 2), abs=1e-6)
    assert scores[best_average] == pytest.approx(expected_score(0.8, 2), abs=1e-6)
    order = [result.node.id for result in recall(service)]
    assert order.index(best_chunk) < order.index(best_average)


def test_extra_windows_that_match_worse_cost_only_the_length_correction(
    store: MemoryStore,
) -> None:
    """Padding a node with irrelevant windows never helps it, and costs it.

    Guards the arithmetic against sum-like aggregations, which would reward the
    node with more windows even when every extra window is irrelevant, and
    pins the correction's shape: the two nodes share a best window, so the
    whole difference between their scores is ``beta * log2(k)``.
    """

    lean = add_node(store, "lean node", ORTHOGONAL)
    padded = add_node(store, "padded node", ORTHOGONAL)
    put_chunks(store, lean, [PARTIAL])
    put_chunks(store, padded, [ORTHOGONAL, PARTIAL, ORTHOGONAL, ORTHOGONAL, ORTHOGONAL])

    service = MemoryRecallService(store, embedder=FixedEmbedder())
    scores = vector_scores(recall(service))

    assert scores[lean] == pytest.approx(expected_score(0.8, 1), abs=1e-6)
    assert scores[padded] == pytest.approx(expected_score(0.8, 5), abs=1e-6)
    # Same best window, five windows instead of one: the raw max is identical
    # and the reported score is lower by exactly the correction.
    assert scores[lean] - scores[padded] == pytest.approx(
        retrieval.LENGTH_BIAS_LOG2_COEFFICIENT * math.log2(5), abs=1e-6
    )


@pytest.mark.parametrize("chunk_count", [1, 2, 4, 8, 16])
def test_length_correction_is_zero_at_one_window_and_log2_beyond(
    store: MemoryStore, chunk_count: int
) -> None:
    """The correction is exactly ``beta * log2(k)``, and exactly nothing at k=1.

    The zero at one window is what keeps the single-vector equivalence below
    honest: a short node must not be taxed for a length it does not have.
    """

    node_id = add_node(store, f"node with {chunk_count} windows", ORTHOGONAL)
    put_chunks(store, node_id, [QUERY_VECTOR] + [ORTHOGONAL] * (chunk_count - 1))

    service = MemoryRecallService(store, embedder=FixedEmbedder())
    score = vector_scores(recall(service))[node_id]

    penalty = 1.0 - score
    assert penalty == pytest.approx(
        retrieval.LENGTH_BIAS_LOG2_COEFFICIENT * math.log2(chunk_count), abs=1e-6
    )
    if chunk_count == 1:
        assert penalty == pytest.approx(0.0, abs=1e-9)


def test_single_chunk_nodes_score_exactly_what_the_old_path_scored(store: MemoryStore) -> None:
    """One window per node must reproduce the single-vector cosine exactly.

    A short node is one chunk whose vector *is* the node vector, so the change
    has to be a no-op for it -- otherwise every existing ranking test would be
    measuring a different channel.
    """

    ids = [
        add_node(store, "first single-window node", QUERY_VECTOR),
        add_node(store, "second single-window node", PARTIAL),
        add_node(store, "third single-window node", [0.28, 0.96, 0.0]),
        add_node(store, "fourth single-window node", ORTHOGONAL),
    ]
    for node_id in ids:
        assert store.count_node_chunks(node_id) == 1

    service = MemoryRecallService(store, embedder=FixedEmbedder())
    scores = vector_scores(recall(service))
    expected = old_single_vector_scores(store, SCOPE, QUERY_VECTOR)

    for node_id in ids:
        if expected[node_id] < retrieval.VECTOR_MATCH_THRESHOLD:
            # Below the threshold the channel contributes nothing, exactly as
            # before; such a node is only in `scores` if some other channel put
            # it there, and then with vector_score 0.
            assert scores.get(node_id, 0.0) == 0.0
            continue
        assert scores[node_id] == pytest.approx(expected[node_id], abs=1e-6)


def test_node_without_chunks_does_not_break_the_scan(store: MemoryStore) -> None:
    """A chunkless node is invisible to the vector channel, and only to it.

    Two ways to get one: content the chunker yields no window for, and a node
    whose chunks were dropped. Neither may take the scan down with it, and
    neither may keep scoring off a vector that no longer has chunks behind it.
    """

    scored = add_node(store, "node with chunks", QUERY_VECTOR)
    dropped = add_node(store, "node whose chunks were dropped", QUERY_VECTOR)
    assert store.delete_node_chunks(dropped) == 1
    blank = store.create_node(
        level="trace",
        content="   ",
        context={"scope": SCOPE, "agent": "tester"},
        embedding=list(QUERY_VECTOR),
    )
    assert store.count_node_chunks(blank.id) == 0

    service = MemoryRecallService(store, embedder=FixedEmbedder())
    results = recall(service)
    scores = vector_scores(results)

    assert scores[scored] == pytest.approx(expected_score(1.0, 1), abs=1e-6)
    assert scores.get(dropped, 0.0) == 0.0
    assert scores.get(blank.id, 0.0) == 0.0


def test_lazy_backfill_still_embeds_and_chunks_a_node_that_has_neither(
    store: MemoryStore,
) -> None:
    """The drain keeps working, and now leaves chunks behind it.

    A node written with no vector at all is invisible to the vector channel
    until the drain reaches it. Under schema v6 ``update_node`` chunks in the
    same transaction, so one recall is still enough to make it scannable.
    """

    node = store.create_node(
        level="trace",
        content=f"{QUERY} recovered by the lazy drain",
        context={"scope": SCOPE, "agent": "tester"},
    )
    assert node.embedding is None
    assert store.count_node_chunks(node.id) == 0

    service = MemoryRecallService(store, embedder=FixedEmbedder())
    scores = vector_scores(recall(service))

    assert store.count_node_chunks(node.id) == 1
    assert scores[node.id] == pytest.approx(expected_score(1.0, 1), abs=1e-6)


def test_matrix_is_reused_across_recalls_that_write(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The corpus is scanned once, not once per recall.

    A recall writes -- access counts and a recall event -- so the cheap
    "anything changed at all" probe moves every time. The chunk-table revision
    is what keeps those writes from throwing away the matrix.
    """

    put_chunks(store, add_node(store, "alpha", ORTHOGONAL), [PARTIAL, QUERY_VECTOR])
    put_chunks(store, add_node(store, "beta", ORTHOGONAL), [PARTIAL])

    scans: list[str | None] = []
    original = store.iter_chunk_embedding_rows

    def counting(**kwargs: Any) -> Iterator[tuple[str, int, memoryview]]:
        scans.append(kwargs.get("scope"))
        return original(**kwargs)

    monkeypatch.setattr(store, "iter_chunk_embedding_rows", counting)

    service = MemoryRecallService(store, embedder=FixedEmbedder())
    first = vector_scores(recall(service))
    after_first = len(scans)
    assert after_first >= 1

    for _ in range(3):
        assert vector_scores(recall(service)) == first
    assert len(scans) == after_first


def test_cache_does_not_serve_a_vector_whose_content_changed(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-embed after an edit must be visible to the very next recall.

    The old per-node dictionary was never invalidated, so it kept answering
    with the vector of text the node no longer contained. Both invalidation
    shapes are exercised: chunks replaced in place (same row count, so only the
    chunk-id half of the revision moves) and chunks dropped.

    Concept nodes, because trace content is append-only in storage -- which is
    also where the live staleness comes from: consolidation rewrites concepts.
    """

    edited = add_node(store, "node that will be edited", QUERY_VECTOR, level="concept")
    anchor = add_node(store, "node that will not", PARTIAL, level="concept")

    service = MemoryRecallService(store, embedder=FixedEmbedder())
    assert vector_scores(recall(service))[edited] == pytest.approx(
        expected_score(1.0, 1), abs=1e-6
    )

    # Re-embedded edit: same chunk count, different vector.
    store.update_node(edited, content="completely different subject matter", embedding=ORTHOGONAL)
    assert store.count_node_chunks(edited) == 1
    scores = vector_scores(recall(service))
    assert scores.get(edited, 0.0) == 0.0
    assert scores[anchor] == pytest.approx(expected_score(0.8, 1), abs=1e-6)

    # Edit with no new vector: storage drops the stale chunks, and the channel
    # must stop scoring the node rather than fall back to the JSON column,
    # which still holds the vector of the text that was replaced.
    store.update_node(anchor, content="another completely different subject")
    assert store.count_node_chunks(anchor) == 0
    assert vector_scores(recall(service)).get(anchor, 0.0) == 0.0


def test_cache_notices_chunks_written_by_another_connection(tmp_path: Path) -> None:
    """A second process writing chunks invalidates this service's matrix.

    ``total_changes`` only counts this connection's writes; the other half of
    the cheap probe, ``PRAGMA data_version``, is what covers the backfill
    script running against the same file as the server.
    """

    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as reader, MemoryStore(path) as writer:
        node_id = add_node(writer, "written elsewhere", ORTHOGONAL)
        service = MemoryRecallService(reader, embedder=FixedEmbedder())
        assert vector_scores(recall(service)).get(node_id, 0.0) == 0.0

        put_chunks(writer, node_id, [QUERY_VECTOR])
        assert vector_scores(recall(service))[node_id] == pytest.approx(
            expected_score(1.0, 1), abs=1e-6
        )


def test_numpy_and_pure_python_paths_agree(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same corpus scores the same with and without numpy.

    The two implementations differ in more than speed: the numpy path unit-
    normalizes the rows once at scan time and takes a dot product, the fallback
    calls ``cosine_similarity`` per chunk. Vectors that are not already unit
    length are in the fixture on purpose, because that is where an unnormalized
    dot product and a real cosine part ways.
    """

    corpus = {
        "many windows": [ORTHOGONAL, PARTIAL, [0.6, 0.8, 0.0]],
        "one window": [[0.28, 0.96, 0.0]],
        "not unit length": [[3.0, 4.0, 0.0], [0.5, 0.0, 0.0]],
        "negative cosine": [[-1.0, 0.0, 0.0], [0.5, 0.5, 0.7071]],
    }
    ids = {
        content: add_node(store, content, ORTHOGONAL) for content in corpus
    }
    for content, vectors in corpus.items():
        put_chunks(store, ids[content], vectors)

    assert retrieval._np is not None, "numpy is a runtime dependency; the comparison needs it"
    with_numpy = vector_scores(recall(MemoryRecallService(store, embedder=FixedEmbedder())))

    monkeypatch.setattr(retrieval, "_np", None)
    # A fresh service, so the blocks are *built* without numpy too, not just
    # scored without it.
    without_numpy = vector_scores(recall(MemoryRecallService(store, embedder=FixedEmbedder())))

    assert set(with_numpy) == set(without_numpy)
    for node_id, score in with_numpy.items():
        assert score == pytest.approx(without_numpy[node_id], abs=1e-6)

    # And the shared answer is the arithmetic both docstrings claim: the best
    # window's *cosine*. `[0.5, 0.0, 0.0]` scores 1.0, not the 0.5 an
    # unnormalized dot product would report; the negative-cosine node keeps its
    # 0.5-ish window rather than clipping to nothing.
    assert with_numpy[ids["not unit length"]] == pytest.approx(
        expected_score(1.0, 2), abs=1e-6
    )
    assert with_numpy[ids["many windows"]] == pytest.approx(expected_score(0.8, 3), abs=1e-6)
    assert with_numpy[ids["negative cosine"]] == pytest.approx(
        expected_score(0.5, 2), abs=1e-4
    )
