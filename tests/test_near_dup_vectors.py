"""The near-duplicate math: mean-pooled node vectors and the duplicate map.

``near_dup`` is the leaf both near-dup consumers stand on -- recall delivery
and the recall-path drain -- so what is pinned here is the arithmetic and the
policy, not any caller's behaviour. Two things get more attention than their
line count suggests, because both are places where a wrong answer looks like a
right one: a node that has no usable vector must be ABSENT rather than zero,
and a candidate longer than its bearer must survive rather than collapse.

Vectors are written explicitly in every test, so an assertion is readable
without running an encoder.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import pytest

from living_memory import near_dup
from living_memory.chunking import TextChunk
from living_memory.near_dup import (
    DuplicateCandidate,
    build_duplicate_map,
    cosine,
    mean_pooled_vectors,
)
from living_memory.storage import MemoryStore


SCOPE = "project:neardup"
#: Threshold used by the map tests. 0.9 is a readable 25.8 degrees, so the
#: fixtures below can be written as angles; the shipped default is higher.
THRESHOLD = 0.9


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    with MemoryStore(tmp_path / "memory.sqlite3") as opened:
        yield opened


def add_node(store: MemoryStore, content: str) -> str:
    node = store.create_node(
        level="trace",
        content=content,
        context={"scope": SCOPE, "agent": "tester"},
        embedding=[1.0, 0.0, 0.0],
    )
    return node.id


def make_chunk(index: int, node_id: str) -> TextChunk:
    """A chunk descriptor for a vector the test supplies directly.

    ``replace_node_chunks`` validates that ordinals are dense and ordered and
    stores the spans verbatim, so they only have to be self-consistent.
    """

    return TextChunk(
        text=f"{node_id} window {index}",
        chunk_index=index,
        token_start=index * 100,
        token_end=(index + 1) * 100,
        char_start=index * 100,
        char_end=(index + 1) * 100,
    )


def put_chunks(store: MemoryStore, node_id: str, vectors: Sequence[Sequence[float]]) -> None:
    """Make ``vectors`` the node's complete chunk set, widths and all.

    ``replace_node_chunks`` records ``len(vector)`` as each row's own
    ``dimensions``, so a mixed-width node is writable here exactly as a
    half-re-embedded database holds one.
    """

    store.replace_node_chunks(
        node_id,
        [(make_chunk(index, node_id), list(vector)) for index, vector in enumerate(vectors)],
    )


def unit(degrees: float) -> list[float]:
    """A unit vector at ``degrees`` from the x-axis, so cosines read as angles."""

    radians = math.radians(degrees)
    return [math.cos(radians), math.sin(radians), 0.0]


# --------------------------------------------------------------------------
# mean_pooled_vectors
# --------------------------------------------------------------------------


def test_mean_pool_averages_the_chunks_rather_than_taking_the_best(
    store: MemoryStore,
) -> None:
    """Two orthogonal windows pool to the bisector, not to either window.

    This is the whole difference from ``retrieval.py``: a max-pool would return
    one of the two inputs, which is how two long notes sharing one boilerplate
    window would score 1.0 against each other.
    """

    node_id = add_node(store, "two windows")
    put_chunks(store, node_id, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    vector = mean_pooled_vectors(store, [node_id])[node_id]

    half = math.sqrt(0.5)
    assert vector == pytest.approx([half, half, 0.0], abs=1e-12)
    assert cosine(vector, [1.0, 0.0, 0.0]) == pytest.approx(half, abs=1e-12)


def test_pooled_vector_is_unit_length(store: MemoryStore) -> None:
    """The contract says L2-normalized, and the collapse thresholds assume it."""

    node_id = add_node(store, "three windows")
    put_chunks(store, node_id, [unit(0.0), unit(30.0), unit(75.0)])

    vector = mean_pooled_vectors(store, [node_id])[node_id]

    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0, abs=1e-12)
    # The mean of the three directions, which for these angles is ~34.6
    # degrees -- not the middle one (30) and not the widest (75).
    assert cosine(vector, unit(34.6)) == pytest.approx(1.0, abs=1e-6)


def test_chunk_magnitudes_do_not_outvote_chunk_directions(store: MemoryStore) -> None:
    """A long unnormalized window contributes a direction, not a mass.

    The writer stores unit vectors; a fixture or a half-migrated row need not,
    and without per-chunk normalization the 10x vector below would pull the
    node's whole direction onto itself.
    """

    node_id = add_node(store, "lopsided magnitudes")
    put_chunks(store, node_id, [[10.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    vector = mean_pooled_vectors(store, [node_id])[node_id]

    half = math.sqrt(0.5)
    assert vector == pytest.approx([half, half, 0.0], abs=1e-12)


def test_node_without_chunks_is_absent_not_zero(store: MemoryStore) -> None:
    """No chunk rows means no key -- never a zero vector.

    Zero vectors are exactly equal to one another, so a zero-vector fallback
    would make every unvectorized node a duplicate of every other one.
    """

    empty_id = add_node(store, "not yet chunked")
    put_chunks(store, empty_id, [])
    chunked_id = add_node(store, "chunked")
    put_chunks(store, chunked_id, [[1.0, 0.0, 0.0]])

    vectors = mean_pooled_vectors(store, [empty_id, chunked_id])

    assert empty_id not in vectors
    assert list(vectors) == [chunked_id]


def test_node_with_mixed_width_chunks_is_absent(store: MemoryStore) -> None:
    """Rows disagreeing on ``dimensions`` are different spaces, so the node drops.

    A database caught mid-re-embed holds both widths. The neighbour node proves
    the skip is per-node: one bad node does not empty the batch.
    """

    mixed_id = add_node(store, "half re-embedded")
    put_chunks(store, mixed_id, [[1.0, 0.0, 0.0], [1.0, 0.0]])
    clean_id = add_node(store, "consistent widths")
    put_chunks(store, clean_id, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    vectors = mean_pooled_vectors(store, [mixed_id, clean_id])

    assert mixed_id not in vectors
    assert clean_id in vectors


def test_chunks_that_cancel_out_are_absent(store: MemoryStore) -> None:
    """A node whose windows sum to no direction has no comparable vector."""

    node_id = add_node(store, "opposing windows")
    put_chunks(store, node_id, [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    assert mean_pooled_vectors(store, [node_id]) == {}


def test_unknown_and_repeated_ids(store: MemoryStore) -> None:
    """An id with no node is absent; a repeated id is read once, not twice."""

    node_id = add_node(store, "present")
    put_chunks(store, node_id, [[1.0, 0.0, 0.0]])
    reads: list[str] = []

    class CountingStore:
        def list_node_chunks(self, requested: str) -> Sequence[Any]:
            reads.append(requested)
            return store.list_node_chunks(requested)

    vectors = mean_pooled_vectors(CountingStore(), [node_id, "01MISSINGMISSINGMISSINGMI", node_id])

    assert list(vectors) == [node_id]
    assert reads == [node_id, "01MISSINGMISSINGMISSINGMI"]


# --------------------------------------------------------------------------
# build_duplicate_map
# --------------------------------------------------------------------------


def test_threshold_zero_returns_an_empty_map() -> None:
    """The rollback path: identical vectors, no collapse, no exceptions."""

    items = [DuplicateCandidate("a", 100), DuplicateCandidate("b", 100)]
    vectors = {"a": unit(0.0), "b": unit(0.0)}

    assert build_duplicate_map(items, vectors, cosine_threshold=0.0, min_length_ratio=0.2) == {}
    assert build_duplicate_map(items, vectors, cosine_threshold=-1.0, min_length_ratio=0.2) == {}


def test_lower_ranked_paraphrase_collapses_into_the_higher_ranked_one() -> None:
    """The top-ranked result bears; the repeat below it is the duplicate."""

    items = [DuplicateCandidate("top", 100), DuplicateCandidate("repeat", 104)]
    vectors = {"top": unit(0.0), "repeat": unit(10.0)}

    collapsed = build_duplicate_map(
        items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2
    )

    assert collapsed == {"repeat": "top"}


def test_a_distinct_fact_below_the_threshold_is_left_alone() -> None:
    """0.85-0.95 is where different facts live; only above the bar collapses."""

    items = [DuplicateCandidate("first", 100), DuplicateCandidate("other", 100)]
    # cos(30 degrees) == 0.866: similar, not the same fact.
    vectors = {"first": unit(0.0), "other": unit(30.0)}

    assert (
        build_duplicate_map(items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2)
        == {}
    )


def test_length_guard_keeps_a_materially_longer_candidate() -> None:
    """Same fact plus a new detail: the detail has to reach the agent.

    The longer node is not merely spared -- it stays a first-class result, so a
    later short twin of it collapses into *it*.
    """

    items = [
        DuplicateCandidate("short", 100),
        DuplicateCandidate("detailed", 400),
        DuplicateCandidate("twin", 400),
    ]
    # `detailed` clears the bar against `short` (cos 20 == 0.94) and is spared
    # only by its length; `twin` misses `short` (cos 40 == 0.77) and clears the
    # bar against `detailed`, so it can only collapse if `detailed` still bears.
    vectors = {"short": unit(0.0), "detailed": unit(20.0), "twin": unit(40.0)}

    collapsed = build_duplicate_map(
        items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2
    )

    assert "detailed" not in collapsed
    assert collapsed == {"twin": "detailed"}


def test_length_guard_is_one_directional() -> None:
    """A candidate shorter than its bearer collapses; only longer is protected.

    A repeat that says less than the bearer carries no detail the bearer lacks,
    which is exactly the case the collapse exists for.
    """

    items = [DuplicateCandidate("long", 400), DuplicateCandidate("terse", 40)]
    vectors = {"long": unit(0.0), "terse": unit(1.0)}

    collapsed = build_duplicate_map(
        items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2
    )

    assert collapsed == {"terse": "long"}


def test_length_guard_boundary_is_the_ratio_it_is_given() -> None:
    """At exactly (1 + ratio) x bearer the candidate still collapses; past it, not.

    Pins the arithmetic the callers' env var will feed: ``min_length_ratio`` is
    the fraction by which a candidate may exceed its bearer and still be
    considered the same fact.
    """

    def collapse_with(candidate_length: int, ratio: float) -> dict[str, str]:
        return build_duplicate_map(
            [DuplicateCandidate("bearer", 100), DuplicateCandidate("candidate", candidate_length)],
            {"bearer": unit(0.0), "candidate": unit(0.0)},
            cosine_threshold=THRESHOLD,
            min_length_ratio=ratio,
        )

    assert collapse_with(120, 0.2) == {"candidate": "bearer"}
    assert collapse_with(121, 0.2) == {}
    # A wider ratio tolerates more; a zero ratio protects anything longer at all.
    assert collapse_with(121, 0.5) == {"candidate": "bearer"}
    assert collapse_with(101, 0.0) == {}
    assert collapse_with(100, 0.0) == {"candidate": "bearer"}


def test_bearer_chains_to_the_original_never_to_another_duplicate() -> None:
    """A cluster of mutual matches records every stub against the one root.

    B and C both repeat A -- cos(C,A) == 0.94 clears the bar directly -- so
    neither is ever handed a bearer whose own text has been replaced by a stub.
    Note *why* C names A: it matches A itself. Had it matched only B, the pair
    would be refused rather than chained; see the test below.
    """

    items = [
        DuplicateCandidate("a", 100),
        DuplicateCandidate("b", 100),
        DuplicateCandidate("c", 100),
    ]
    vectors = {"a": unit(0.0), "b": unit(10.0), "c": unit(20.0)}

    collapsed = build_duplicate_map(
        items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2
    )

    assert collapsed == {"b": "a", "c": "a"}
    assert not set(collapsed.values()) & set(collapsed)


def test_a_chained_bearer_below_the_bar_is_refused_not_recorded() -> None:
    """C matches only the stub B, so C keeps its content rather than name A.

    cos(C,B) == 0.94 clears the bar; cos(C,A) == 0.77 does not. Recording C
    against A anyway is what the field measurement caught: 8 delivered slots
    carrying a `content_ref` to a bearer they matched at as little as 0.904,
    inside the 0.85-0.95 band where different facts live. Cosine is not
    transitive, so the collapse is refused and C's text reaches the agent.
    """

    items = [
        DuplicateCandidate("a", 100),
        DuplicateCandidate("b", 100),
        DuplicateCandidate("c", 100),
    ]
    vectors = {"a": unit(0.0), "b": unit(20.0), "c": unit(40.0)}

    collapsed = build_duplicate_map(
        items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2
    )

    assert collapsed == {"b": "a"}
    assert "c" not in collapsed


def test_a_refused_chain_still_collapses_against_a_root_it_does_match() -> None:
    """Refusing one bearer is not refusing the candidate: the scan continues.

    ``b`` is a stub of ``a``. ``d`` matches ``b`` (cos 0.985) but not ``a``
    (cos 0.87), so that pair is refused -- and ``d`` then meets ``c``, a root of
    its own it clears the bar against (cos 0.94), and collapses there instead.
    """

    items = [
        DuplicateCandidate("a", 100),
        DuplicateCandidate("b", 100),
        DuplicateCandidate("c", 100),
        DuplicateCandidate("d", 100),
    ]
    vectors = {
        "a": unit(0.0),
        "b": unit(20.0),
        "c": unit(50.0),
        "d": unit(30.0),
    }

    collapsed = build_duplicate_map(
        items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2
    )

    assert collapsed == {"b": "a", "d": "c"}


def test_every_recorded_pair_clears_the_threshold_against_its_named_bearer() -> None:
    """The map's invariant, checked over a fan of angles rather than one case.

    Twelve nodes 8 degrees apart form a long chain of overlapping matches --
    exactly the shape that produced the dishonest content_refs in the field.
    Whatever the function records, the pair must measure above the bar.
    """

    items = [DuplicateCandidate(f"n{index}", 100) for index in range(12)]
    vectors = {f"n{index}": unit(index * 8.0) for index in range(12)}

    collapsed = build_duplicate_map(
        items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2
    )

    assert collapsed, "the fan must produce collapses, or this proves nothing"
    for duplicate_id, bearer_id in collapsed.items():
        assert (
            cosine(vectors[duplicate_id], vectors[bearer_id]) > THRESHOLD
        ), f"{duplicate_id} names {bearer_id} without clearing the bar against it"


def test_the_bearer_is_the_highest_ranked_match_not_the_closest() -> None:
    """Rank decides who bears, because rank is what the agent reads first.

    `candidate` is far nearer to `closer` (5 degrees, cos 0.996) than to
    `ranked-first` (25 degrees, cos 0.906), but only barely clears the bar
    against the latter -- and the latter is the one it is recorded against.
    `closer` stays a bearer of its own, so the answer cannot come from the
    chain-resolution step instead.
    """

    items = [
        DuplicateCandidate("ranked-first", 100),
        DuplicateCandidate("closer", 100),
        DuplicateCandidate("candidate", 100),
    ]
    vectors = {
        "ranked-first": unit(0.0),
        "closer": unit(30.0),
        "candidate": unit(25.0),
    }

    collapsed = build_duplicate_map(
        items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2
    )

    assert collapsed == {"candidate": "ranked-first"}


def test_length_guard_is_applied_against_the_resolved_bearer() -> None:
    """The guard compares the candidate with the text that actually ships.

    ``c`` matches ``b``, but ``b`` is a stub by then and ``a``'s text is what
    the agent receives -- so ``a``'s length is what decides whether ``c`` has a
    detail the answer would otherwise lose. ``c`` is within the ratio of ``b``
    and past it for ``a``, so the two readings disagree here on purpose.
    """

    items = [
        DuplicateCandidate("a", 100),
        DuplicateCandidate("b", 120),
        DuplicateCandidate("c", 140),
    ]
    vectors = {"a": unit(0.0), "b": unit(20.0), "c": unit(40.0)}

    collapsed = build_duplicate_map(
        items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2
    )

    assert collapsed == {"b": "a"}
    assert "c" not in collapsed


def test_a_node_without_a_vector_is_never_collapsed_and_never_a_bearer() -> None:
    """An unvectorized node takes no part: it is not a duplicate and bears none.

    The pair below is byte-identical in length and would look like an obvious
    twin -- but with nothing to compare, "unknown" is not "the same".
    """

    items = [
        DuplicateCandidate("unvectorized", 100),
        DuplicateCandidate("first-with-vector", 100),
        DuplicateCandidate("repeat", 100),
    ]
    vectors = {"first-with-vector": unit(0.0), "repeat": unit(0.0)}

    collapsed = build_duplicate_map(
        items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2
    )

    assert collapsed == {"repeat": "first-with-vector"}


def test_map_is_pure_and_accepts_plain_pairs() -> None:
    """Same inputs, same map; inputs unmutated; ``(id, length)`` pairs work.

    Both consumers hold results or nodes rather than dataclasses, so the pair
    form is part of the contract, and purity is what lets delivery stay a pure
    function with the map handed to it ready-made.
    """

    items: list[Any] = [("top", 100), ("repeat", 100)]
    vectors = {"top": unit(0.0), "repeat": unit(2.0)}
    items_before = list(items)
    vectors_before = {key: list(value) for key, value in vectors.items()}

    first = build_duplicate_map(items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2)
    second = build_duplicate_map(items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2)

    assert first == second == {"repeat": "top"}
    assert items == items_before
    assert vectors == vectors_before


def test_a_repeated_id_in_one_ranking_never_becomes_its_own_bearer() -> None:
    """Defensive: the same node twice cannot map to itself."""

    items = [DuplicateCandidate("a", 100), DuplicateCandidate("a", 100)]

    collapsed = build_duplicate_map(
        items, {"a": unit(0.0)}, cosine_threshold=THRESHOLD, min_length_ratio=0.2
    )

    assert collapsed == {}


# --------------------------------------------------------------------------
# cosine, both math paths
# --------------------------------------------------------------------------


def test_cosine_scores_zero_across_vector_widths() -> None:
    """Different widths are different spaces, not a comparable shared prefix.

    ``embeddings.cosine_similarity`` zips non-strictly and would report 1.0 for
    this pair; that is the silent cross-width comparison this module refuses.
    """

    assert cosine([1.0, 0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine([], []) == 0.0
    assert cosine([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_cosine_agrees_with_and_without_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both paths compute the same cosine, including for unnormalized input."""

    pairs = [
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([3.0, 4.0, 0.0], [0.5, 0.0, 0.0]),
        (unit(0.0), unit(18.0)),
        ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
        ([0.1] * 384, [0.1] * 383 + [0.2]),
    ]

    assert near_dup._np is not None, "numpy is a runtime dep; the comparison needs it"
    with_numpy = [cosine(left, right) for left, right in pairs]

    monkeypatch.setattr(near_dup, "_np", None)
    without_numpy = [cosine(left, right) for left, right in pairs]

    for numpy_value, pure_value in zip(with_numpy, without_numpy, strict=True):
        assert numpy_value == pytest.approx(pure_value, abs=1e-12)
    # And the shared answer is a real cosine: magnitude does not enter it.
    assert with_numpy[1] == pytest.approx(0.6, abs=1e-12)
    assert with_numpy[3] == pytest.approx(-1.0, abs=1e-12)


def test_pooling_and_collapse_agree_without_numpy(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole path -- pool then collapse -- returns the same thing either way.

    Unnormalized and zero chunk rows are in the fixture on purpose: those are
    where a per-row normalization and a raw sum would part ways.
    """

    corpus = {
        "bearer": [[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]],
        "repeat": [[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]],
        "zero row": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "distinct": [[0.0, 1.0, 0.0]],
    }
    ids = {content: add_node(store, content) for content in corpus}
    for content, vectors in corpus.items():
        put_chunks(store, ids[content], vectors)
    items = [DuplicateCandidate(ids[content], len(content)) for content in corpus]

    def pool_and_collapse() -> tuple[dict[str, list[float]], dict[str, str]]:
        pooled = mean_pooled_vectors(store, [item.node_id for item in items])
        return pooled, build_duplicate_map(
            items, pooled, cosine_threshold=THRESHOLD, min_length_ratio=0.2
        )

    assert near_dup._np is not None, "numpy is a runtime dep; the comparison needs it"
    numpy_vectors, numpy_map = pool_and_collapse()

    monkeypatch.setattr(near_dup, "_np", None)
    pure_vectors, pure_map = pool_and_collapse()

    assert set(numpy_vectors) == set(pure_vectors)
    for node_id, vector in numpy_vectors.items():
        assert vector == pytest.approx(pure_vectors[node_id], abs=1e-12)
    assert numpy_map == pure_map
    # "bearer" and "repeat" pool to the same direction despite the 5x magnitude
    # on the first row; "distinct" is orthogonal to both and survives.
    assert numpy_map == {ids["repeat"]: ids["bearer"]}
    assert ids["distinct"] not in numpy_map


# --------------------------------------------------------------------------
# module boundary
# --------------------------------------------------------------------------


def test_module_imports_nothing_from_its_consumers() -> None:
    """near_dup is the leaf both consumers stand on, so it depends on neither.

    An import in the other direction would couple recall delivery to the drain
    and make the two unmergeable independently. Checked on the source rather
    than on ``sys.modules``, which is polluted by whatever else the suite
    imported.
    """

    source = Path(near_dup.__file__).read_text(encoding="utf-8")
    forbidden = {"delivery", "retrieval", "server"}
    imported: set[str] = set()
    for statement in ast.walk(ast.parse(source)):
        if isinstance(statement, ast.Import):
            imported.update(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom) and statement.module:
            imported.add(statement.module)

    assert not {
        name for name in imported if name.rsplit(".", 1)[-1] in forbidden
    }, f"near_dup must not import its consumers; got {sorted(imported)}"
