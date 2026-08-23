"""The recall-path drain collapsing a freshly vectorized verbatim repeat.

This is the one place in a read path that mutates the corpus, so most of what
is pinned here is what it REFUSES to do. The flag-off test is the important
one: with ``LM_DRAIN_NEAR_DUP_SUPERSEDES`` unset the drain must write chunks
and nothing else, which is what makes the whole behaviour revertible by an
operator without a revert of the commit.

Vectors are the test's, not the encoder's: every node's content carries a
keyword that ``FixedEmbedder`` maps to a written-down direction, so a cosine in
an assertion can be read off the fixture instead of run. Short content means
one chunk per node (``storage._chunk_embeddings_for`` reuses the node vector
for a single window), so a node's mean-pooled vector *is* the vector below.

The arithmetic itself -- mean pooling, the length guard, the duplicate map --
belongs to ``tests/test_near_dup_vectors.py``; what belongs here is which
nodes this pass offers to that arithmetic and what it does with the answer.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import pytest

from living_memory import retrieval
from living_memory.models import Node
from living_memory.near_dup import IDENTIFIER_VETO_ENV
from living_memory.retrieval import (
    DRAIN_NEAR_DUP_COSINE_ENV,
    DRAIN_NEAR_DUP_ENV,
    DRAIN_NEAR_DUP_KIND,
    MemoryRecallService,
)
from living_memory.storage import MemoryStore


SCOPE = "project:draindup"
QUERY = "alpha drift"

#: The direction every "alpha" node points in, and the query's own direction.
ALPHA = (1.0, 0.0, 0.0)
#: cos(ALPHA, .) == 0.96: a near-paraphrase, comfortably inside the band the
#: delivery path collapses at 0.95 and comfortably below this path's 0.99.
#: Different facts live here, and this pass must leave them alone.
NEAR_96 = (0.96, math.sqrt(1.0 - 0.96**2), 0.0)
#: Nothing to do with the query, for fixture nodes that only have to exist.
ORTHOGONAL = (0.0, 1.0, 0.0)

#: Content keyword -> direction. Longest key first so "alpha96" does not match
#: the "alpha" rule.
VECTORS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("alpha96", NEAR_96),
    ("alpha", ALPHA),
)


class FixedEmbedder:
    """Content keyword -> a direction this test wrote down."""

    def embed(self, text: str) -> list[float]:
        lowered = text.lower()
        for keyword, vector in VECTORS:
            if keyword in lowered:
                return list(vector)
        return list(ORTHOGONAL)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    with MemoryStore(tmp_path / "memory.sqlite3") as opened:
        yield opened


@pytest.fixture
def collapse_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn the gate on and leave the threshold at its shipped default."""

    monkeypatch.setenv(DRAIN_NEAR_DUP_ENV, "1")
    monkeypatch.delenv(DRAIN_NEAR_DUP_COSINE_ENV, raising=False)


@pytest.fixture(autouse=True)
def gate_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test inherits the gate from the environment it runs in.

    The identifier veto is deleted rather than set, so every scenario below
    runs against its shipped default -- on -- unless it says otherwise.
    """

    monkeypatch.delenv(DRAIN_NEAR_DUP_ENV, raising=False)
    monkeypatch.delenv(DRAIN_NEAR_DUP_COSINE_ENV, raising=False)
    monkeypatch.delenv(IDENTIFIER_VETO_ENV, raising=False)


@pytest.fixture
def identifier_veto_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn the identifier veto off for a test about some other variable.

    The ``alpha96`` fixture pair differs in a digit-bearing token, so the veto
    refuses it on its own merits. A test whose subject is the cosine threshold
    has to remove that second cause or it is no longer measuring one variable.
    """

    monkeypatch.setenv(IDENTIFIER_VETO_ENV, "0")


def established(store: MemoryStore, content: str, *, level: str = "trace") -> str:
    """A node that already has its vector, and so is never drained again.

    Written with an explicit embedding, which storage turns into the node's one
    chunk in the same transaction -- exactly the state a node reaches after an
    earlier recall drained it.
    """

    node = store.create_node(
        level=level,
        content=content,
        context={"scope": SCOPE, "agent": "tester"},
        embedding=list(FixedEmbedder().embed(content)),
    )
    return node.id


def unvectorized(store: MemoryStore, content: str, **kwargs: Any) -> str:
    """A node as ``memory_remember`` leaves it: no vector, and so no chunks.

    This is the input the drain exists for, and the only way into the collapse:
    a node that already has chunks is never handed to it.
    """

    node = store.create_node(
        level="trace",
        content=content,
        context={"scope": SCOPE, "agent": "tester"},
        **kwargs,
    )
    return node.id


def recall(store: MemoryStore, **kwargs: Any) -> list[Any]:
    service = MemoryRecallService(store, embedder=FixedEmbedder())
    return service.memory_recall(QUERY, scope=SCOPE, depth=0, max_results=10, **kwargs)


def supersedes_edges(store: MemoryStore) -> list[tuple[str, str, str]]:
    """``(source, target, kind)`` for every supersedes edge in the database."""

    rows = store.connection.execute(
        "SELECT source_id, target_id, metadata FROM connections WHERE type = 'supersedes'"
    ).fetchall()
    edges = []
    for row in rows:
        metadata = row["metadata"] or "{}"
        kind = ""
        if '"kind"' in metadata:
            import json

            kind = str(json.loads(metadata).get("kind", ""))
        edges.append((str(row["source_id"]), str(row["target_id"]), kind))
    return sorted(edges)


def drain_edges(store: MemoryStore) -> list[tuple[str, str]]:
    """Only the edges this pass writes, without the byte-exact path's."""

    return [
        (source, target)
        for source, target, kind in supersedes_edges(store)
        if kind == DRAIN_NEAR_DUP_KIND
    ]


# ----------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------


def test_flag_unset_leaves_the_drain_writing_chunks_and_nothing_else(
    store: MemoryStore,
) -> None:
    """The default: a verbatim repeat is chunked, and stays a second copy.

    The rollback contract in one assertion. An operator who unsets the variable
    gets this back with no revert and no migration, so the pre-change behaviour
    has to be reachable from the shipped build -- not merely from an older one.
    """

    original = established(store, "alpha drift measured at three units")
    repeat = unvectorized(store, "alpha drift measured at 3 units")

    results = recall(store)

    assert supersedes_edges(store) == []
    assert store.count_node_chunks(repeat) == 1
    assert {result.node.id for result in results} == {original, repeat}
    assert not any(result.superseded for result in results)
    assert store.get_node(repeat).decayed is False


def test_zero_threshold_is_the_second_rollback(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate on, threshold 0: the same pre-change behaviour, one var away.

    ``build_duplicate_map`` treats a non-positive threshold as "collapse
    nothing"; the drain has to reach that same floor, so an operator debugging
    a live server can neutralize the collapse without deciding whether the gate
    variable is also feeding something else.
    """

    monkeypatch.setenv(DRAIN_NEAR_DUP_ENV, "1")
    monkeypatch.setenv(DRAIN_NEAR_DUP_COSINE_ENV, "0")
    established(store, "alpha drift measured at three units")
    unvectorized(store, "alpha drift measured at 3 units")

    recall(store)

    assert supersedes_edges(store) == []


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no", "maybe"])
def test_only_an_explicit_yes_turns_the_collapse_on(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """A misread flag must fail closed: anything unrecognized means off."""

    monkeypatch.setenv(DRAIN_NEAR_DUP_ENV, value)
    established(store, "alpha drift measured at three units")
    unvectorized(store, "alpha drift measured at 3 units")

    recall(store)

    assert supersedes_edges(store) == []


# ----------------------------------------------------------------------
# The collapse
# ----------------------------------------------------------------------


def test_verbatim_repeat_is_superseded_by_the_node_it_repeats(
    store: MemoryStore, collapse_on: None
) -> None:
    """The point of the whole change, and the direction it goes in.

    The edge runs original -> repeat: the arriving copy is demoted and the node
    that was already there keeps its standing, its access history and its
    edges. Nothing is deleted and nothing is decayed -- the repeat keeps its
    row, its text and its chunks, and ``memory_lookup`` still returns it.
    """

    original = established(store, "alpha drift measured at three units")
    repeat = unvectorized(store, "alpha drift measured at 3 units")

    recall(store)

    assert drain_edges(store) == [(original, repeat)]
    surviving = store.get_node(original)
    demoted = store.get_node(repeat)
    assert surviving.decayed is False and demoted.decayed is False
    assert demoted.content == "alpha drift measured at 3 units"
    assert store.count_node_chunks(repeat) == 1


def test_collapse_records_the_cosine_that_caused_it(
    store: MemoryStore, collapse_on: None
) -> None:
    """The edge carries its own evidence, under a findable ``kind``.

    An operator reverting a bad calibration has to be able to select exactly
    the rows this pass wrote, and to see which of them were marginal; the
    byte-exact path's ``duplicate_content`` edges are a different claim and
    must not be swept up with them.
    """

    import json

    original = established(store, "alpha drift measured at three units")
    repeat = unvectorized(store, "alpha drift measured at 3 units")

    recall(store)

    row = store.connection.execute(
        "SELECT metadata FROM connections WHERE source_id = ? AND target_id = ?",
        (original, repeat),
    ).fetchone()
    metadata = json.loads(row["metadata"])
    assert metadata["kind"] == DRAIN_NEAR_DUP_KIND
    assert metadata["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert metadata["threshold"] == retrieval.DEFAULT_DRAIN_NEAR_DUP_COSINE
    assert metadata["scope"] == SCOPE


def test_a_pair_below_the_threshold_survives_as_two_nodes(
    store: MemoryStore, collapse_on: None, identifier_veto_off: None
) -> None:
    """0.96 is not a repeat, and this path must never reach into that band.

    The measured 0.85-0.95 band holds DIFFERENT facts, and 0.95-0.99 is where
    the delivery path hides a result an agent can still fetch. Only this path
    writes an edge that outlives the answer, so only this path needs the high
    threshold -- and needs it tested, because the two consumers share the
    comparison function and would otherwise share its threshold by accident.

    The veto is off here and in the test below so that the threshold is the one
    thing that moves between them; that it *also* refuses this pair is the
    subject of the identifier-veto section, not of this one.
    """

    established(store, "alpha drift measured at three units")
    repeat = unvectorized(store, "alpha96 drift measured at 3 units")

    recall(store)

    assert supersedes_edges(store) == []
    assert store.count_node_chunks(repeat) == 1


def test_a_lowered_threshold_reaches_the_pair_the_default_leaves_alone(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, identifier_veto_off: None
) -> None:
    """The threshold is the knob, and it is the *only* thing separating the two
    outcomes above -- same fixture, same gate, one variable."""

    monkeypatch.setenv(DRAIN_NEAR_DUP_ENV, "1")
    monkeypatch.setenv(DRAIN_NEAR_DUP_COSINE_ENV, "0.95")
    original = established(store, "alpha drift measured at three units")
    repeat = unvectorized(store, "alpha96 drift measured at 3 units")

    recall(store)

    assert drain_edges(store) == [(original, repeat)]


# ----------------------------------------------------------------------
# The identifier veto
# ----------------------------------------------------------------------


def test_a_template_pair_differing_in_an_identifier_is_never_superseded(
    store: MemoryStore, collapse_on: None
) -> None:
    """This pass's residual risk, closed: identical vectors, different facts.

    Both nodes embed to ALPHA, so their cosine is 1.0 -- above any threshold an
    operator could set, including the shipped 0.99. The template is the whole
    vector and the node name is a rounding error in it, which is exactly the
    class measured on the alt corpus at 0.9547. An edge here would outlive the
    answer that caused it, so it is the one this pass must never write.
    """

    established(store, "alpha drift recorded for узел layer-fauna СДЕЛАН")
    repeat = unvectorized(store, "alpha drift recorded for узел layer-actors СДЕЛАН")

    recall(store)

    assert drain_edges(store) == []
    assert supersedes_edges(store) == []
    assert store.get_node(repeat) is not None


def test_the_valve_off_restores_the_collapse_the_veto_refused(
    store: MemoryStore, collapse_on: None, identifier_veto_off: None
) -> None:
    """Same fixture, same gate, valve at 0: the drain honours the same env var.

    Without this the previous test proves only that *something* refused the
    pair. The pair collapsing the moment ``LM_NEAR_DUP_IDENTIFIER_VETO=0`` is
    what identifies the cause -- and shows this consumer reads the same valve
    as the delivery path rather than one of its own.
    """

    original = established(store, "alpha drift recorded for узел layer-fauna СДЕЛАН")
    repeat = unvectorized(store, "alpha drift recorded for узел layer-actors СДЕЛАН")

    recall(store)

    assert drain_edges(store) == [(original, repeat)]


def test_an_identifier_the_original_also_names_does_not_block_the_drain(
    store: MemoryStore, collapse_on: None
) -> None:
    """The veto is not a blanket refusal of identifier-bearing traces.

    Both nodes name ``layer-fauna``; the repeat says nothing the original does
    not, which is the case this pass exists for. Its negative control is the
    test two above -- same shape, one token changed.
    """

    original = established(store, "alpha drift recorded for узел layer-fauna СДЕЛАН")
    repeat = unvectorized(store, "alpha drift for узел layer-fauna, СДЕЛАН")

    recall(store)

    assert drain_edges(store) == [(original, repeat)]


# ----------------------------------------------------------------------
# The guards
# ----------------------------------------------------------------------


def test_a_trace_a_live_concept_names_is_never_superseded(
    store: MemoryStore, collapse_on: None
) -> None:
    """Provenance is not duplication.

    A concept is deliberately written byte-different from its sources and sits
    0.82-0.95 from them; 48% of the live corpus's active traces are named by
    some live concept. Superseding one would demote a node the consolidation
    layer still points at, and would do it inside a read path.
    """

    original = established(store, "alpha drift measured at three units")
    covered = unvectorized(store, "alpha drift measured at 3 units")
    store.create_node(
        level="concept",
        content="drift summary over the alpha telemetry cluster",
        context={"scope": SCOPE, "agent": "consolidator"},
        provenance={"source_traces": [covered]},
    )

    recall(store)

    assert supersedes_edges(store) == []
    assert store.get_node(covered).decayed is False
    assert original != covered


def test_a_decayed_concept_protects_nothing(
    store: MemoryStore, collapse_on: None
) -> None:
    """"Live" is load-bearing in the guard above.

    A decayed concept is no longer pointing at anything, so the traces under it
    go back to being ordinary corpus. Without this the guard would only ever
    accumulate: every concept ever written would keep protecting its sources
    forever, and the corpus would have no ceiling at all.
    """

    original = established(store, "alpha drift measured at three units")
    covered = unvectorized(store, "alpha drift measured at 3 units")
    concept = store.create_node(
        level="concept",
        content="drift summary over the alpha telemetry cluster",
        context={"scope": SCOPE, "agent": "consolidator"},
        provenance={"source_traces": [covered]},
    )
    store.soft_delete_node(concept.id, reason="superseded by a later era")

    recall(store)

    assert drain_edges(store) == [(original, covered)]


def test_a_materially_longer_repeat_is_never_superseded(
    store: MemoryStore, collapse_on: None
) -> None:
    """Same fact plus a new detail: the detail has to reach the agent.

    The guard is the delivery path's, at the delivery path's ratio, and it
    comes from the same function -- so a candidate more than 20% longer than
    its bearer survives here for exactly the reason it stays full text there.
    """

    original = established(store, "alpha drift measured at three units")
    longer = unvectorized(
        store,
        "alpha drift measured at 3 units, and the sensor was recalibrated first",
    )

    recall(store)

    assert len(store.get_node(longer).content) > len(store.get_node(original).content) * 1.2
    assert supersedes_edges(store) == []


def test_a_shorter_repeat_still_collapses(
    store: MemoryStore, collapse_on: None
) -> None:
    """The length guard is one-directional, and stays that way.

    A repeat that says *less* than its original is the ordinary case -- it is
    the original that holds the extra text -- so a guard that fired in both
    directions would collapse nothing at all.
    """

    original = established(
        store, "alpha drift measured at three units after the recalibration"
    )
    shorter = unvectorized(store, "alpha drift measured at 3 units")

    recall(store)

    assert drain_edges(store) == [(original, shorter)]


def test_a_correction_is_never_demoted_as_a_repeat(
    store: MemoryStore, collapse_on: None
) -> None:
    """``memory_teach``'s output looks exactly like this pass's target.

    A correction restates what it corrects and changes one thing, which is a
    near-verbatim repeat by construction. Demoting one would invert the only
    mechanism the system has for changing its mind, so a node with an outgoing
    supersedes edge is off limits however similar it is to anything.
    """

    # The corrected node is deliberately NOT a near-duplicate of anything here,
    # so the only thing standing between the correction and a demotion under
    # ``twin`` is the guard this test is about.
    stale = established(store, "beta reading of nine units, since corrected")
    correction = unvectorized(store, "alpha drift measured at 3 units")
    store.create_connection(correction, stale, "supersedes", weight=1.0)
    twin = established(store, "alpha drift measured at three units")

    recall(store)

    assert drain_edges(store) == []
    assert (correction, stale, "") in supersedes_edges(store)
    assert store.get_node(twin).decayed is False


def test_a_bearer_that_is_itself_superseded_bears_nothing(
    store: MemoryStore, collapse_on: None
) -> None:
    """Nothing is demoted *under* a stale node.

    The nearest neighbour may be a node some correction already replaced;
    hanging a fresh repeat off it would put the repeat below a node that is
    itself below something else, for no gain. The veto is a veto and not a
    search for the next best bearer: the runner-up is no evidence about the
    text the candidate actually repeats.
    """

    stale = established(store, "alpha drift measured at three units")
    correction = established(store, "beta reading unrelated to the query")
    store.create_connection(correction, stale, "supersedes", weight=1.0)
    repeat = unvectorized(store, "alpha drift measured at 3 units")

    recall(store)

    assert drain_edges(store) == []
    assert store.get_node(repeat).decayed is False


def test_a_node_with_no_chunk_vectors_is_never_collapsed(
    store: MemoryStore, collapse_on: None
) -> None:
    """No vector means no comparison, not a comparison against zero.

    ``mean_pooled_vectors`` leaves such a node out of its result entirely, and
    the pass has to read that absence as "unknown". The alternative -- a zero
    vector -- is worse than useless here: zero vectors are exactly equal to
    each other, so every unvectorized node would collapse into every other one.
    """

    original = established(store, "alpha drift measured at three units")
    repeat = established(store, "alpha drift measured at 3 units")
    # The state a content edit leaves behind: the stale chunks are deleted and
    # the node waits for the drain to give it new ones.
    store.replace_node_chunks(repeat, [])
    service = MemoryRecallService(store, embedder=FixedEmbedder())
    node = store.get_node(repeat)
    index = service._scope_chunk_index(SCOPE, service._chunk_corpus_revision())

    assert service._supersede_drained_near_dups(SCOPE, [node], index) == []
    assert supersedes_edges(store) == []

    # The control, on the same node through the same call: once it has chunks
    # to pool, it is exactly the repeat the pass exists to collapse. Without
    # this the assertion above would also hold for a pass that did nothing.
    service._rechunk_node(node)
    index = service._scope_chunk_index(SCOPE, service._chunk_corpus_revision())
    assert service._supersede_drained_near_dups(SCOPE, [node], index) == [
        (original, repeat)
    ]


def test_a_concept_is_neither_collapsed_nor_a_bearer(
    store: MemoryStore, collapse_on: None
) -> None:
    """The pass moves traces only, in both roles.

    Levels above ``trace`` are consolidation's to create and to retire; a
    digest that reads like its own sources is the expected shape of one, not a
    duplicate to be swept up by a read path.
    """

    concept = established(store, "alpha drift measured at three units", level="concept")
    repeat = unvectorized(store, "alpha drift measured at 3 units")

    recall(store)

    assert supersedes_edges(store) == []
    assert store.get_node(concept).decayed is False
    assert store.get_node(repeat).decayed is False


def test_two_fresh_twins_collapse_once_and_never_into_a_cycle(
    store: MemoryStore, collapse_on: None
) -> None:
    """With no older original between them, the smaller id survives.

    Two nodes that each nominate the other are the one shape that could close a
    supersedes cycle, which ``_enforce_correction_dominance`` would then have to
    break on every future recall. Exactly one edge, and which way it points is
    decided by id order -- creation order down to the millisecond, and stable
    below that, which is what the pass needs: the same pair must collapse the
    same way on a re-run.
    """

    first = unvectorized(store, "alpha drift measured at three units")
    second = unvectorized(store, "alpha drift measured at 3 units")
    bearer, superseded = sorted([first, second])

    recall(store)

    assert drain_edges(store) == [(bearer, superseded)]


def test_a_third_twin_lands_on_the_original_not_on_the_copy(
    store: MemoryStore, collapse_on: None
) -> None:
    """Bearers are roots, so no chain of stubs forms.

    ``build_duplicate_map`` promises a consumer never has to chase a chain to
    find the node holding the text; a drain that collapsed B into A and then C
    into B would break that promise from the other side.
    """

    original = established(store, "alpha drift measured at three units")
    second = unvectorized(store, "alpha drift measured at 3 units")
    third = unvectorized(store, "alpha drift measured at three (3) units")

    recall(store)

    assert drain_edges(store) == sorted([(original, second), (original, third)])


def test_the_pass_does_not_run_twice_over_the_same_node(
    store: MemoryStore, collapse_on: None
) -> None:
    """A second recall drains nothing, so it collapses nothing.

    The node is chunked now; ``list_unchunked_nodes`` no longer names it and
    neither does ``list_unembedded_nodes``. Guards that depend on state this
    pass itself wrote (the repeat is now superseded) keep it out of a rerun
    too, so the edge count is stable however many recalls follow.
    """

    original = established(store, "alpha drift measured at three units")
    repeat = unvectorized(store, "alpha drift measured at 3 units")

    recall(store)
    recall(store)
    recall(store)

    assert drain_edges(store) == [(original, repeat)]


# ----------------------------------------------------------------------
# The call the collapse happens inside
# ----------------------------------------------------------------------


def test_the_collapsing_call_returns_a_consistent_result_set(
    store: MemoryStore, collapse_on: None
) -> None:
    """A drain that supersedes must not lie to the recall it ran inside.

    The edge is written before ``rank_candidates`` reads the supersedes table,
    so the very answer that triggered the collapse already reflects it: the
    repeat comes back flagged, and the node it repeats comes back above it.
    Were the pass moved after ranking, this call would return two unflagged
    copies and only the *next* recall would be right.
    """

    original = established(store, "alpha drift measured at three units")
    repeat = unvectorized(store, "alpha drift measured at 3 units")

    results = recall(store)
    by_id = {result.node.id: result for result in results}

    assert set(by_id) == {original, repeat}
    assert by_id[repeat].superseded is True
    assert by_id[original].superseded is False
    order = [result.node.id for result in results]
    assert order.index(original) < order.index(repeat)


def test_the_repeat_is_still_reachable_after_the_collapse(
    store: MemoryStore, collapse_on: None
) -> None:
    """Demoted, not deleted: the corpus loses no text to this pass.

    ``supersedes`` is a ranking claim. The row, the content and the chunks stay
    exactly where they were, which is what makes the collapse recoverable by
    deleting one edge.
    """

    established(store, "alpha drift measured at three units")
    repeat = unvectorized(store, "alpha drift measured at 3 units")

    recall(store)
    fetched = store.get_node(repeat)

    assert fetched is not None
    assert fetched.decayed is False
    assert fetched.content == "alpha drift measured at 3 units"
    assert store.count_node_chunks(repeat) == 1


def test_the_cached_chunk_index_still_matches_the_database(
    store: MemoryStore, collapse_on: None
) -> None:
    """The drain's existing cache drop still covers everything this touches.

    The pass writes ``connections`` rows and nothing else -- no chunk row is
    inserted, deleted or edited, and no node is decayed -- so the matrix the
    drain's ``_chunk_index.pop`` rebuilt is still an exact description of the
    chunk corpus after the collapse. Checked against a matrix built from
    scratch afterwards rather than against itself, so a cached row that the
    mutation invalidated would show up as a difference.
    """

    original = established(store, "alpha drift measured at three units")
    repeat = unvectorized(store, "alpha drift measured at 3 units")

    service = MemoryRecallService(store, embedder=FixedEmbedder())
    service.memory_recall(QUERY, scope=SCOPE, depth=0, max_results=10)

    cached = service._chunk_index[SCOPE]
    fresh = service._build_chunk_index(SCOPE, cached.revision)
    assert [block.node_ids for block in cached.blocks] == [
        block.node_ids for block in fresh.blocks
    ]
    assert cached.row_count == fresh.row_count
    # The node the pass superseded is in there, with its chunk: a supersedes
    # edge is not a decay, so its vectors stay scannable.
    assert {original, repeat} <= {
        node_id for block in cached.blocks for node_id in block.node_ids
    }


def test_the_next_recall_reuses_the_cache_the_collapse_left_behind(
    store: MemoryStore, collapse_on: None
) -> None:
    """The write moves ``total_changes``, and that must cost only a probe.

    ``_chunk_corpus_revision``'s tier one (total_changes, data_version) is
    deliberately not trusted across writes, but tier two asks the chunk table
    itself -- which this pass never touched -- so the revision comes back
    unchanged and the 60k-row matrix is not thrown away for an edge insert.
    """

    established(store, "alpha drift measured at three units")
    unvectorized(store, "alpha drift measured at 3 units")

    service = MemoryRecallService(store, embedder=FixedEmbedder())
    service.memory_recall(QUERY, scope=SCOPE, depth=0, max_results=10)
    cached = service._chunk_index[SCOPE]

    service.memory_recall(QUERY, scope=SCOPE, depth=0, max_results=10)

    assert service._chunk_index[SCOPE] is cached


# ----------------------------------------------------------------------
# The pooling this pass does on the cached matrix
# ----------------------------------------------------------------------


def make_block(vectors: Sequence[Sequence[Sequence[float]]], ids: Sequence[str]) -> Any:
    """A ``_ChunkBlock`` holding one run of chunk rows per node."""

    accumulator = retrieval._BlockAccumulator()
    for node_id, rows in zip(ids, vectors, strict=True):
        for row in rows:
            accumulator.add(node_id, _as_blob(row))
    return accumulator.freeze(len(vectors[0][0]))


def _as_blob(vector: Sequence[float]) -> Any:
    from living_memory.storage import pack_chunk_embedding

    return memoryview(pack_chunk_embedding(list(vector)))


def test_block_pooling_is_the_mean_of_directions_not_the_max(
    store: MemoryStore,
) -> None:
    """The pooled matrix must answer the node-vs-node question, not recall's.

    ``retrieval`` max-pools a node's chunks against a *query* on purpose: one
    matching window is a real answer. Between two nodes that is wrong -- two
    long, mostly different notes sharing one boilerplate window would max-pool
    to ~1.0 -- so this pass mean-pools, and the assertion is written so any
    max-like aggregation fails it.
    """

    block = make_block([[ALPHA, ORTHOGONAL]], ["01AAAAAAAAAAAAAAAAAAAAAAAA"])

    pooled = retrieval._mean_pool_block(block)
    row = [float(value) for value in pooled[0]]

    # Mean of two orthogonal unit directions, renormalized: 45 degrees from
    # each. A max-pool would have returned ALPHA itself.
    assert row == pytest.approx([math.sqrt(0.5), math.sqrt(0.5), 0.0], abs=1e-6)


def put_chunks(store: MemoryStore, node_id: str, vectors: Sequence[Sequence[float]]) -> None:
    """Replace one node's chunk set with exactly ``vectors``."""

    from living_memory.chunking import TextChunk

    store.replace_node_chunks(
        node_id,
        [
            (
                TextChunk(
                    text=f"{node_id} window {index}",
                    chunk_index=index,
                    token_start=index * 100,
                    token_end=(index + 1) * 100,
                    char_start=index * 100,
                    char_end=(index + 1) * 100,
                ),
                list(vector),
            )
            for index, vector in enumerate(vectors)
        ],
    )


def test_the_collapse_is_the_same_with_and_without_numpy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collapse_on: None
) -> None:
    """One answer from two implementations, on a fixture that can tell them apart.

    The numpy path unit-normalizes chunk rows once at scan time and pools with
    ``add.reduceat``; the fallback normalizes each row as it adds it. The
    fixture's windows are deliberately not unit length and deliberately not
    parallel, which is where the two could part ways -- and where pooling
    *without* per-chunk normalization parts ways with both: the raw mean of
    these two windows sits at 0.970 to the repeat, below the threshold, so a
    magnitude-weighted pool would collapse nothing at all.
    """

    from living_memory import near_dup

    # Normalized, the two windows are +/-26.6 degrees off ALPHA, so their mean
    # direction is ALPHA exactly. Unnormalized, the longer one drags it off.
    windows = [[6.0, 3.0, 0.0], [2.0, -1.0, 0.0]]
    assert near_dup.cosine([4.0, 1.0, 0.0], list(ALPHA)) == pytest.approx(0.970, abs=1e-3)
    assert near_dup.cosine([4.0, 1.0, 0.0], list(ALPHA)) < (
        retrieval.DEFAULT_DRAIN_NEAR_DUP_COSINE
    )

    def run(path: Path) -> tuple[str, str, list[tuple[str, str]]]:
        with MemoryStore(path) as opened:
            original = established(opened, "alpha drift measured at three units")
            put_chunks(opened, original, windows)
            repeat = unvectorized(opened, "alpha drift measured at 3 units")
            recall(opened)
            return original, repeat, drain_edges(opened)

    assert retrieval._np is not None, "numpy is a runtime dependency; the comparison needs it"
    original, repeat, with_numpy = run(tmp_path / "with_numpy.sqlite3")
    assert with_numpy == [(original, repeat)]

    # Both modules, so this is the state a host without numpy is actually in.
    monkeypatch.setattr(retrieval, "_np", None)
    monkeypatch.setattr(near_dup, "_np", None)
    original, repeat, without_numpy = run(tmp_path / "without_numpy.sqlite3")

    assert without_numpy == [(original, repeat)]


def test_pooling_survives_a_node_whose_chunks_cancel(store: MemoryStore) -> None:
    """Opposed chunks leave a zero row, never a NaN.

    A node with no direction has to score 0 against everything -- which is
    ``near_dup``'s "absent, never zero" reached the other way round. A NaN row
    would silently poison every comparison in the block.
    """

    opposed = [ALPHA, (-1.0, 0.0, 0.0)]
    block = make_block([opposed], ["01AAAAAAAAAAAAAAAAAAAAAAAA"])

    pooled = retrieval._mean_pool_block(block)
    row = [float(value) for value in pooled[0]]

    assert row == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
