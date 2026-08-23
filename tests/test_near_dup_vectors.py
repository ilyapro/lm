"""The near-duplicate math: mean-pooled node vectors and the duplicate map.

``near_dup`` is the leaf both near-dup consumers stand on -- recall delivery
and the recall-path drain -- so what is pinned here is the arithmetic and the
policy, not any caller's behaviour. Three things get more attention than their
line count suggests, because each is a place where a wrong answer looks like a
right one: a node that has no usable vector must be ABSENT rather than zero, a
candidate longer than its bearer must survive rather than collapse, and a
candidate naming an identifier its bearer does not name must survive at any
cosine at all.

Vectors are written explicitly in every test, so an assertion is readable
without running an encoder.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pytest

from living_memory import near_dup
from living_memory.chunking import TextChunk
from living_memory.near_dup import (
    IDENTIFIER_VETO_ENV,
    DuplicateCandidate,
    build_duplicate_map,
    cosine,
    extract_identifiers,
    identifier_veto_enabled,
    identifiers_absent_from,
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
# build_duplicate_map, with the identifier veto off
# --------------------------------------------------------------------------
#
# Everything in this section is the map as it was before the veto existed --
# rank order, chain resolution, the length guard, the absent-vector rule -- and
# every case reaches it through ``collapse``, which passes
# ``identifier_veto=False``. That makes the section do double duty: it pins the
# arithmetic, and it *is* the rollback contract, because
# ``LM_NEAR_DUP_IDENTIFIER_VETO=0`` promises exactly this map back. The veto's
# own behaviour is the section after it, where candidates carry text.


def collapse(
    items: Sequence[Any],
    vectors: Mapping[str, Sequence[float]],
    *,
    cosine_threshold: float,
    min_length_ratio: float,
) -> dict[str, str]:
    """``build_duplicate_map`` with the identifier veto explicitly off."""

    return build_duplicate_map(
        items,
        vectors,
        cosine_threshold=cosine_threshold,
        min_length_ratio=min_length_ratio,
        identifier_veto=False,
    )


def test_threshold_zero_returns_an_empty_map() -> None:
    """The rollback path: identical vectors, no collapse, no exceptions."""

    items = [DuplicateCandidate("a", 100), DuplicateCandidate("b", 100)]
    vectors = {"a": unit(0.0), "b": unit(0.0)}

    assert collapse(items, vectors, cosine_threshold=0.0, min_length_ratio=0.2) == {}
    assert collapse(items, vectors, cosine_threshold=-1.0, min_length_ratio=0.2) == {}


def test_lower_ranked_paraphrase_collapses_into_the_higher_ranked_one() -> None:
    """The top-ranked result bears; the repeat below it is the duplicate."""

    items = [DuplicateCandidate("top", 100), DuplicateCandidate("repeat", 104)]
    vectors = {"top": unit(0.0), "repeat": unit(10.0)}

    collapsed = collapse(
        items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2
    )

    assert collapsed == {"repeat": "top"}


def test_a_distinct_fact_below_the_threshold_is_left_alone() -> None:
    """0.85-0.95 is where different facts live; only above the bar collapses."""

    items = [DuplicateCandidate("first", 100), DuplicateCandidate("other", 100)]
    # cos(30 degrees) == 0.866: similar, not the same fact.
    vectors = {"first": unit(0.0), "other": unit(30.0)}

    assert (
        collapse(items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2)
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

    collapsed = collapse(
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

    collapsed = collapse(
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
        return collapse(
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

    collapsed = collapse(
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

    collapsed = collapse(
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

    collapsed = collapse(
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

    collapsed = collapse(
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

    collapsed = collapse(
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

    collapsed = collapse(
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

    collapsed = collapse(
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

    first = collapse(items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2)
    second = collapse(items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2)

    assert first == second == {"repeat": "top"}
    assert items == items_before
    assert vectors == vectors_before


def test_a_repeated_id_in_one_ranking_never_becomes_its_own_bearer() -> None:
    """Defensive: the same node twice cannot map to itself."""

    items = [DuplicateCandidate("a", 100), DuplicateCandidate("a", 100)]

    collapsed = collapse(
        items, {"a": unit(0.0)}, cosine_threshold=THRESHOLD, min_length_ratio=0.2
    )

    assert collapsed == {}


# --------------------------------------------------------------------------
# the identifier extractor
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        ("ulid", "recalled 01M0QJV1BBRXNHF7D5NS23FD35 twice", "01M0QJV1BBRXNHF7D5NS23FD35"),
        # 26 Crockford characters and not one digit: the ULID rule is what
        # catches this, and nothing else in the grammar would.
        ("crockford without digits", "id ABCDEFGHJKMNPQRSTVWXYZABCD here", "ABCDEFGHJKMNPQRSTVWXYZABCD"),
        ("repo path", "see src/living_memory/near_dup.py for it", "src/living_memory/near_dup.py"),
        ("home path", "backup at ~/.local/share/living-memory/global.sqlite3 now", "~/.local/share/living-memory/global.sqlite3"),
        ("absolute path", "wrote /home/sfx/p/lm today", "/home/sfx/p/lm"),
        ("goal-node name", "branch near-dup-identifier-veto merged", "near-dup-identifier-veto"),
        ("worktree slug", "узел layer-fauna СДЕЛАН", "layer-fauna"),
        ("underscore slug", "call build_duplicate_map now", "build_duplicate_map"),
        ("dotted module path", "living_memory.near_dup is the leaf", "living_memory.near_dup"),
        ("dotted symbol path", "near_dup.build_duplicate_map vetoes", "near_dup.build_duplicate_map"),
        ("bare filename", "edit near_dup.py first", "near_dup.py"),
        ("hex digest", "digest 19877a4383b2e2d2 frozen", "19877a4383b2e2d2"),
        ("short commit id", "at commit ce93bae exactly", "ce93bae"),
        ("ticket id", "ticket LM-123 was closed", "LM-123"),
        ("scoped name", "written with scope project:lm here", "project:lm"),
        ("date", "measured 2026-08-23 on alt", "2026-08-23"),
        ("version", "shipped v1.2 to the host", "v1.2"),
        ("measured number", "the pair scores cos 0.9547 on alt", "0.9547"),
        ("year", "written in 2026 by an agent", "2026"),
        ("camel case symbol", "MemoryStore satisfies it", "MemoryStore"),
        ("url", "fetched https://example.test/x from there", "https://example.test/x"),
    ],
)
def test_identifier_classes_are_extracted(label: str, text: str, expected: str) -> None:
    """Every class the veto is required to cover, one case each.

    Digit-optional is the point of the list: the measurement harness's regex
    requires a digit in the token and therefore cannot see ``layer-fauna``,
    which is exactly the class that caused the false collapses.
    """

    assert extract_identifiers(text) == (expected,), label


def test_the_motivating_class_carries_no_digit_at_all() -> None:
    """The digit-requiring metric's blind spot, stated as an assertion.

    ``layer-fauna`` / ``layer-actors`` scored 0.9547 on the alt corpus and were
    collapsed. Nothing in either token is a digit, so a digit-gated identifier
    rule reports zero identifiers lost and the pair looks like an honest
    repeat. This is why the veto's grammar is broader than the harness's.
    """

    for token in ("layer-fauna", "layer-actors"):
        assert not any(character.isdigit() for character in token)
        assert extract_identifiers(f"узел {token} СДЕЛАН") == (token,)


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("english prose", "the ballast survey runs at slack water"),
        ("russian prose", "узел проверен и записан в память"),
        ("long single word", "understanding notwithstanding"),
        ("emphasis in caps", "this is IMPORTANT and СДЕЛАН"),
        ("english abbreviation", "e.g. this one, i.e. that one"),
        ("russian abbreviation", "т.е. вот так"),
        ("small numbers", "alpha drift measured at 3 units, 12 in all"),
        ("short decimal", "a ratio of 1.5 between them"),
        ("sentence boundary", "the fact ends here. Another begins"),
    ],
)
def test_prose_yields_no_identifiers(label: str, text: str) -> None:
    """The other half of the grammar: what must NOT veto a collapse.

    Without this the veto degenerates into "never collapse anything", which
    would pass every veto test in this file and destroy the feature.
    """

    assert extract_identifiers(text) == (), label


def test_extraction_is_ordered_and_deduplicated() -> None:
    text = "near_dup.py and layer-fauna, then near_dup.py again"

    assert extract_identifiers(text) == ("near_dup.py", "layer-fauna")


def test_a_bearer_naming_the_fuller_path_no_longer_covers_the_bare_filename() -> None:
    """Token equality, and this is the assertion it cost. Inverted deliberately.

    A bearer spelling ``src/living_memory/near_dup.py`` does contain the string
    ``near_dup.py``, and this test used to read that as "the bearer carries
    that name, so nothing is hidden". The drain simulation showed why that
    reading is not safe: this corpus builds node names by SUFFIXING, so of two
    distinct nodes the shorter name is always a substring of the longer one.
    Under containment ``…/checkpoint-selected-profile-v2`` counted as present
    in ``…/checkpoint-selected-profile-v2-repaired``, and two tree nodes with
    two distinct recorded failures collapsed into one at cosine 0.99350
    (``01KTRR6WHFXFW16M691Q1N98E1`` -> ``01KTRJCB5FQ7ZT2WKD02TQ20B3``,
    project:x). No rule can pass this pair and block that one -- they are the
    same shape -- and the goal's accounting says which way to resolve it: a
    false veto costs one uncollapsed stub, a false pass costs a hidden fact.

    So the pair vetoes in both directions now. The price is real and is being
    paid on purpose: a candidate that honestly said less than its bearer no
    longer collapses into it.
    """

    assert identifiers_absent_from(
        "edit near_dup.py", "edit src/living_memory/near_dup.py"
    ) == ("near_dup.py",)
    assert identifiers_absent_from(
        "edit src/living_memory/near_dup.py", "edit near_dup.py"
    ) == ("src/living_memory/near_dup.py",)


def test_the_two_pairs_that_escaped_the_veto_at_099_are_blocked() -> None:
    """The measured survivors, verbatim from `artifacts/near-dup/drain-simulation.md`.

    Both are goal-tree nodes whose names differ only by a suffix, both were
    read as ``different_facts``, and both passed the containment veto because
    the candidate's path sat inside the bearer's. Under token equality the
    candidate's own path is the token that is missing.
    """

    checkpoint_bearer = (
        "OUTCOME fail: universal-frontier-advance-x/checkpoint-abi-v2-streaming/"
        "checkpoint-selected-profile-v2-repaired — Measure the repaired "
        "production-selected ABI v2 checkpoint/resume path and write tracked "
        "selected evidence."
    )
    checkpoint_candidate = checkpoint_bearer.replace(
        "checkpoint-selected-profile-v2-repaired", "checkpoint-selected-profile-v2"
    )
    assert identifiers_absent_from(checkpoint_candidate, checkpoint_bearer) == (
        "universal-frontier-advance-x/checkpoint-abi-v2-streaming/"
        "checkpoint-selected-profile-v2",
    )

    audit_bearer = (
        "OUTCOME pass: the-ceil/nw6-train-scoreblind-arms/"
        "nw6-stock-frontier-arm-reduced/stock-frontier-reduced-training-run/"
        "stock-factorized-bptt-throughput-repair/stock-factorized-source-repair-v2/"
        "stock-contract-preservation-audit-reintegrate — Reintegrate the tracked "
        "audit proving the source repair did not alter the frozen NW-6 stock "
        "comparison contract."
    )
    audit_candidate = audit_bearer.replace(
        "stock-contract-preservation-audit-reintegrate",
        "stock-contract-preservation-audit",
    )
    assert identifiers_absent_from(audit_candidate, audit_bearer) == (
        "the-ceil/nw6-train-scoreblind-arms/nw6-stock-frontier-arm-reduced/"
        "stock-frontier-reduced-training-run/"
        "stock-factorized-bptt-throughput-repair/"
        "stock-factorized-source-repair-v2/stock-contract-preservation-audit",
    )


def test_the_honest_repeats_in_the_same_band_still_collapse() -> None:
    """The other half of the >=0.99 band: the token rule must not cost these.

    Both were read as ``verbatim_repeat`` in the same simulation. The first
    carries no identifier at all, so there is nothing for either rule to
    compare; the second's only identifier is a date the bearer spells as its
    own token, which is what containment and equality agree about.
    """

    reworded_bearer = (
        "Rejected alternative: Perform sibling implementation tasks\n"
        "Rejected because: This node is explicitly scoped to critique only and "
        "not perform sibling tasks."
    )
    reworded_candidate = reworded_bearer.replace(
        "not perform sibling tasks.", "must not execute sibling tasks."
    )
    assert extract_identifiers(reworded_candidate) == ()
    assert identifiers_absent_from(reworded_candidate, reworded_bearer) == ()

    probe_bearer = (
        "Latency benchmark probe at 2026-05-24 — synthetic timing trace; safe to decay."
    )
    probe_candidate = probe_bearer.replace("probe at", "probe #2 at")
    assert extract_identifiers(probe_candidate) == ("2026-05-24",)
    assert identifiers_absent_from(probe_candidate, probe_bearer) == ()


def test_a_number_inside_a_finer_measurement_is_not_that_measurement() -> None:
    """The class nobody had noticed: ``0.99`` is inside ``0.99350``, not a token of it.

    Containment could not tell "the band is 0.99" from "the band is 0.99350",
    which is the same collapse as the node names one level down in the grammar.
    """

    assert identifiers_absent_from("band 0.99 here", "band 0.99350 here") == ("0.99",)
    assert identifiers_absent_from("band 0.99350 here", "band 0.99 here") == ("0.99350",)


def test_identifier_comparison_is_case_sensitive() -> None:
    """Case is meaning in paths, env vars and ids; the ambiguous way is to veto."""

    assert identifiers_absent_from("set LM_NEAR_DUP_IDENTIFIER_VETO", "set lm_near_dup_identifier_veto") == (
        "LM_NEAR_DUP_IDENTIFIER_VETO",
    )


# --------------------------------------------------------------------------
# the identifier veto in the map
# --------------------------------------------------------------------------


def at_cosine(value: float) -> list[float]:
    """A unit vector whose cosine against ``unit(0.0)`` is exactly ``value``."""

    return unit(math.degrees(math.acos(value)))


def vetoed(
    bearer: str,
    candidate: str,
    *,
    cosine_value: float = 0.99,
    identifier_veto: bool = True,
) -> dict[str, str]:
    """Two texts, bearer ranked first, at a chosen cosine: the map they produce."""

    return build_duplicate_map(
        [
            DuplicateCandidate("bearer", len(bearer), bearer),
            DuplicateCandidate("candidate", len(candidate), candidate),
        ],
        {"bearer": unit(0.0), "candidate": at_cosine(cosine_value)},
        cosine_threshold=THRESHOLD,
        min_length_ratio=0.2,
        identifier_veto=identifier_veto,
    )


def test_the_measured_false_collapse_is_refused_and_the_valve_restores_it() -> None:
    """The pair the goal is built on, at the cosine it was measured at.

    0.9547 is above the shipped 0.95 threshold, so before the veto this pair
    collapsed and one of two different tree nodes reached the agent as a stub.
    """

    bearer = "узел layer-fauna СДЕЛАН"
    candidate = "узел layer-actors СДЕЛАН"
    assert cosine(unit(0.0), at_cosine(0.9547)) == pytest.approx(0.9547, abs=1e-12)

    assert vetoed(bearer, candidate, cosine_value=0.9547) == {}
    assert vetoed(bearer, candidate, cosine_value=0.9547, identifier_veto=False) == {
        "candidate": "bearer"
    }


def test_no_cosine_is_high_enough_to_beat_the_veto() -> None:
    """A veto, not a re-ranking: identical vectors do not buy the collapse.

    Two texts differing in an identifier are different facts at any cosine, so
    the guard cannot be a threshold that a close enough pair slips past.
    """

    for value in (0.95, 0.99, 0.999, 1.0):
        assert vetoed("scan of layer-fauna", "scan of layer-actors", cosine_value=value) == {}


def test_an_honest_repeat_still_collapses_with_the_veto_on() -> None:
    """The veto is not an off switch: prose that repeats prose still collapses.

    Its own negative control -- ``test_prose_yields_no_identifiers`` proves the
    grammar is silent on this text, and this proves the map acts on that.
    """

    assert vetoed(
        "the ballast survey runs at slack water",
        "at slack water is when the survey runs",
    ) == {"candidate": "bearer"}


def test_the_veto_is_one_directional() -> None:
    """The bearer's own extra identifiers do not block the collapse.

    They are not lost by collapsing: the bearer keeps its full text, and that
    text is what the agent reads. Only the candidate's would vanish behind a
    stub, so only the candidate's are counted.
    """

    assert vetoed(
        "the survey ran, see src/living_memory/near_dup.py and ULID 01M0QJV1BBRXNHF7D5NS23FD35",
        "the survey ran at slack water",
    ) == {"candidate": "bearer"}


def test_an_identifier_the_bearer_also_names_does_not_veto() -> None:
    """Shared identifiers are shared facts; only the candidate's extras veto."""

    shared = "the drain collapsed layer-fauna at 0.9547"
    assert vetoed(shared, "layer-fauna collapsed at 0.9547 in the drain") == {
        "candidate": "bearer"
    }


def test_a_candidate_without_text_is_never_collapsed_while_the_veto_is_on() -> None:
    """An unenforceable veto fails closed, in both tuple and dataclass form.

    A caller that did not wire the text through must lose collapses, not the
    guard -- a silently unenforceable veto is the one failure mode that would
    let this ship while doing nothing.
    """

    vectors = {"bearer": unit(0.0), "candidate": unit(1.0)}
    for items in (
        [("bearer", 100), ("candidate", 100)],
        [DuplicateCandidate("bearer", 100), DuplicateCandidate("candidate", 100)],
        [("bearer", 100, "plain bearer text"), ("candidate", 100)],
    ):
        assert (
            build_duplicate_map(
                items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2
            )
            == {}
        ), items
    # ... and the three-element tuple form is how a caller supplies it.
    assert build_duplicate_map(
        [("bearer", 100, "plain bearer text"), ("candidate", 100, "text of the bearer, plain")],
        vectors,
        cosine_threshold=THRESHOLD,
        min_length_ratio=0.2,
    ) == {"candidate": "bearer"}


def test_a_bearer_without_text_blocks_the_collapse_too() -> None:
    """No bearer text means no way to ask whether it names the identifier."""

    assert (
        build_duplicate_map(
            [
                DuplicateCandidate("bearer", 100),
                DuplicateCandidate("candidate", 100, "a plain repeat"),
            ],
            {"bearer": unit(0.0), "candidate": unit(1.0)},
            cosine_threshold=THRESHOLD,
            min_length_ratio=0.2,
        )
        == {}
    )


def test_the_veto_is_measured_against_the_bearer_that_actually_ships() -> None:
    """``c`` matched the stub ``b``, but ``a``'s text is what the agent reads.

    ``c`` quotes an identifier neither ``b`` nor ``a`` names, and it is refused
    even though it cleared the bar against both. Note what this does NOT claim:
    a case where the stub names the identifier and the root does not is
    *unreachable*, and deliberately so. ``b`` only became a stub by passing this
    same veto, so every identifier token of ``b``'s text is already a token of
    ``a``'s; set membership is transitive, and the guarantee therefore composes
    down a chain instead of leaking at the second link. The invariant that
    follows is the checkable form of that.
    """

    items = [
        DuplicateCandidate("a", 100, "the survey report was filed"),
        DuplicateCandidate("b", 100, "filed, the survey report was"),
        DuplicateCandidate("c", 100, "filed the survey report layer-fauna"),
    ]
    vectors = {"a": unit(0.0), "b": unit(5.0), "c": unit(10.0)}

    collapsed = build_duplicate_map(
        items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2
    )

    assert collapsed == {"b": "a"}
    assert "c" not in collapsed


def test_no_collapsed_node_hides_an_identifier_from_its_recorded_bearer() -> None:
    """The map's veto-on invariant, over a fan of overlapping matches.

    Twelve nodes eight degrees apart, half of them carrying identifiers that
    the nodes above them do not: whatever the function chooses to record, the
    text the agent ends up reading must name everything the stub named. This is
    the property the whole node exists to establish, checked over the map
    rather than over one hand-built pair.
    """

    texts = {
        "n0": "the survey report was filed",
        "n1": "filed, the survey report was",
        "n2": "the survey report was filed.",
        "n3": "the survey report layer-fauna was filed",
        "n4": "the survey report was filed on 2026-08-23",
        "n5": "the survey report was filed, see near_dup.py",
        "n6": "the survey report was filed by MemoryStore",
        "n7": "the survey report was refiled",
        "n8": "the survey report 01M0QJV1BBRXNHF7D5NS23FD35 was filed",
        "n9": "the survey report has been filed",
        "n10": "filed the survey report layer-actors",
        "n11": "the survey report was duly filed",
    }
    items = [DuplicateCandidate(key, len(text), text) for key, text in texts.items()]
    vectors = {f"n{index}": unit(index * 8.0) for index in range(12)}

    collapsed = build_duplicate_map(
        items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2
    )

    assert collapsed, "the fan must produce collapses, or this proves nothing"
    for duplicate_id, bearer_id in collapsed.items():
        assert identifiers_absent_from(texts[duplicate_id], texts[bearer_id]) == (), (
            f"{duplicate_id} collapsed into {bearer_id} and took an identifier with it"
        )
    # The nodes that carry an identifier are exactly the ones left standing --
    # and note n7 and n9, which carry none: they still collapse INTO bearers
    # that do, because the bearer's text ships whole. The veto is one-directional.
    assert collapsed == {"n1": "n0", "n2": "n0", "n7": "n4", "n9": "n6", "n11": "n8"}
    # What the veto is buying, on the same fan: without it three of the six
    # identifier-bearing nodes disappear into a bearer naming a different one.
    assert build_duplicate_map(
        items,
        vectors,
        cosine_threshold=THRESHOLD,
        min_length_ratio=0.2,
        identifier_veto=False,
    ) == {"n1": "n0", "n2": "n0", "n4": "n3", "n5": "n3", "n6": "n3", "n9": "n7", "n11": "n8"}


def test_an_identifier_only_the_root_names_still_collapses() -> None:
    """The mirror of the test above: the root names it, so nothing is hidden.

    Same shape, same ranks -- only which node quotes the identifier moves --
    and the answer flips. That is what makes the previous test about the
    resolved root rather than about identifiers in general.
    """

    items = [
        DuplicateCandidate("a", 100, "the survey report near-dup-identifier-veto was filed"),
        DuplicateCandidate("b", 100, "the survey report was filed"),
        DuplicateCandidate("c", 100, "filed the survey report near-dup-identifier-veto"),
    ]
    vectors = {"a": unit(0.0), "b": unit(5.0), "c": unit(10.0)}

    assert build_duplicate_map(
        items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2
    ) == {"b": "a", "c": "a"}


def test_the_veto_off_map_ignores_the_text_entirely() -> None:
    """The rollback contract: with the valve off, content changes nothing.

    Same items twice -- once carrying identifier-heavy text, once stripped to
    the pre-veto ``(node_id, length)`` pairs -- must produce the same map. The
    last assertion is what keeps this honest: the corpus really does trip the
    veto, so the equality above is not the equality of two empty maps.
    """

    texts = {
        "a": "drain simulation on 2026-08-23 collapsed layer-fauna at 0.9547",
        "b": "drain simulation on 2026-08-23 collapsed layer-actors at 0.9547",
        "c": "drain simulation collapsed a node, see src/living_memory/near_dup.py",
        "d": "drain simulation collapsed a node",
    }
    items = [DuplicateCandidate(key, len(text), text) for key, text in texts.items()]
    pairs: list[Any] = [(key, len(text)) for key, text in texts.items()]
    vectors = {"a": unit(0.0), "b": unit(4.0), "c": unit(8.0), "d": unit(12.0)}

    with_text = build_duplicate_map(
        items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2, identifier_veto=False
    )
    without_text = build_duplicate_map(
        pairs, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2, identifier_veto=False
    )

    assert with_text == without_text == {"b": "a", "c": "a", "d": "a"}
    assert build_duplicate_map(
        items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2
    ) == {"d": "a"}


def test_the_veto_preserves_the_maps_other_invariants() -> None:
    """Rank-order bearers, roots-only values, and the threshold, all veto-on.

    Identifier-free text throughout, so what is under test is that turning the
    veto on did not disturb the rest of the map: the bearer is still the
    highest-ranked match rather than the closest, values are still roots, and
    every recorded pair still clears the bar against the bearer it names.
    """

    words = ("survey", "ballast", "quillon", "vantrex", "slack", "water")
    items = [
        DuplicateCandidate(f"n{index}", 100, f"the {words[index % len(words)]} was recorded")
        for index in range(12)
    ]
    vectors = {f"n{index}": unit(index * 8.0) for index in range(12)}

    collapsed = build_duplicate_map(
        items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2
    )

    assert collapsed, "the fan must produce collapses, or this proves nothing"
    assert not set(collapsed.values()) & set(collapsed)  # bearers are roots
    for duplicate_id, bearer_id in collapsed.items():
        assert cosine(vectors[duplicate_id], vectors[bearer_id]) > THRESHOLD
    # Rank, not proximity: n1 and n2 both match n0 and each other.
    assert collapsed["n1"] == "n0" and collapsed["n2"] == "n0"


def test_a_disabled_threshold_still_wins_over_the_veto() -> None:
    """``cosine_threshold <= 0`` is the outer rollback and returns before it."""

    items = [
        DuplicateCandidate("a", 100, "layer-fauna"),
        DuplicateCandidate("b", 100, "layer-actors"),
    ]

    assert (
        build_duplicate_map(
            items, {"a": unit(0.0), "b": unit(0.0)}, cosine_threshold=0.0, min_length_ratio=0.2
        )
        == {}
    )


def test_the_length_guard_and_the_veto_are_both_measured_against_the_root() -> None:
    """A candidate can be refused by either guard; neither shadows the other."""

    items = [
        DuplicateCandidate("a", 100, "the survey report was filed"),
        DuplicateCandidate("long", 400, "the survey report was filed" + " and detailed" * 20),
        DuplicateCandidate("named", 100, "the survey report layer-fauna was filed"),
    ]
    vectors = {"a": unit(0.0), "long": unit(5.0), "named": unit(5.0)}

    assert (
        build_duplicate_map(
            items, vectors, cosine_threshold=THRESHOLD, min_length_ratio=0.2
        )
        == {}
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("", True),
        ("1", True),
        ("on", True),
        ("true", True),
        ("yes", True),
        ("maybe", True),  # unrecognized must not silently disable the guard
        ("0", False),
        ("off", False),
        ("false", False),
        ("no", False),
        ("OFF", False),
        ("  0  ", False),
    ],
)
def test_valve_parsing(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool
) -> None:
    """One reader, so "the veto is on" cannot mean two things in two consumers."""

    if value is None:
        monkeypatch.delenv(IDENTIFIER_VETO_ENV, raising=False)
    else:
        monkeypatch.setenv(IDENTIFIER_VETO_ENV, value)

    assert identifier_veto_enabled() is expected


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
        return pooled, collapse(
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
