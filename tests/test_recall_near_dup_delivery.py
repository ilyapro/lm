"""Collapsing a semantic repeat of a recall answer into a stub.

Three layers are pinned here, because the feature is deliberately split
across them:

* the pure renderer (``delivery.shape_recall_results``), which is handed a
  finished ``{duplicate: bearer}`` map and must not grow a database read or an
  env read to use it — including how the new class orders against the two
  older stub classes, and that an absent map reproduces today's answer;
* the env valves beside it (``LM_RECALL_NEAR_DUP_COSINE`` /
  ``LM_RECALL_NEAR_DUP_LENGTH_RATIO``), including the ``0`` rollback;
* the single call site (``server._near_duplicate_map`` and ``memory_recall``),
  where the vectors are actually read — over a real ``MemoryStore``, with
  chunk vectors written explicitly so a cosine is readable as an angle rather
  than as whatever an encoder happened to produce.

The interesting cases are the ones where a wrong answer looks right: a node
with no vector must ship whole rather than collapse into a stranger, and a
candidate longer than its bearer must ship whole because "the same fact plus
a new detail" is not a repeat.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import pytest

from living_memory.chunking import TextChunk
from living_memory.delivery import (
    DEFAULT_NEAR_DUP_COSINE,
    DEFAULT_NEAR_DUP_LENGTH_RATIO,
    DELIVERY_FULL,
    DELIVERY_NEAR_DUPLICATE,
    DELIVERY_SESSION_DUPLICATE,
    DELIVERY_SNIPPET,
    DELIVERY_TWIN_DUPLICATE,
    LADDER_COMPLETE,
    NEAR_DUP_COSINE_ENV,
    NEAR_DUP_LENGTH_RATIO_ENV,
    PREVIEW_MAX_CHARS,
    near_dup_cosine_from_env,
    near_dup_length_ratio_from_env,
    shape_recall_results,
)
from living_memory.models import Node
from living_memory.retrieval import RecallResult
from living_memory.server import (
    _DROPPABLE_TRAILING_STUBS,
    _near_duplicate_map,
    create_mcp_server,
)
from living_memory.storage import MemoryStore

from test_mcp_server import FakeMCP

SCOPE = "project:neardupdelivery"
#: Every seeded node carries this stem so one query reaches all of them and
#: the answer's composition is decided by the near-dup map, not by retrieval.
QUERY_STEM = "quillon vantrex ballast survey"

#: Delivery knobs this file never wants inherited from the ambient shell.
_KNOBS = (
    "LM_DELIVERY_SNIPPET_CHARS",
    "LM_DELIVERY_SNIPPET_LADDER",
    "LM_DELIVERY_SESSION_DEDUP",
    "LM_DELIVERY_CONTEXT_VALUE_CHARS",
    "LM_DELIVERY_FULL_NODE_DIET",
    "LM_DELIVERY_PROVENANCE_VALUE_CHARS",
    "LM_DELIVERY_STATS_COMPACTION",
    "LM_DELIVERY_SPARSE",
    "LM_RECALL_NEAR_DUP_COSINE",
    "LM_RECALL_NEAR_DUP_LENGTH_RATIO",
    "LM_RECALL_REPEAT_GATING",
    "LM_RECALL_REPEAT_DROP_TRAILING_STUBS",
    "LM_AUTO_CONSOLIDATE_POLICY",
    "LM_DECAY_SWEEP_INTERVAL_SEC",
)


@pytest.fixture(autouse=True)
def _default_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every scenario from an unset environment (shipped defaults)."""

    for env in _KNOBS:
        monkeypatch.delenv(env, raising=False)


# --------------------------------------------------------------------------
# the pure renderer
# --------------------------------------------------------------------------


def make_node(node_id: str, content: str, **overrides: Any) -> Node:
    defaults: dict[str, Any] = {
        "id": node_id,
        "level": "trace",
        "content": content,
        "scope": SCOPE,
        "agent": "tester",
        "task": "near-dup-delivery",
        "timestamp": "2026-08-23T00:00:00Z",
        "context": {"scope": SCOPE},
        "created_at": "2026-08-23T00:00:00Z",
        "updated_at": "2026-08-23T00:00:00Z",
    }
    defaults.update(overrides)
    return Node(**defaults)


def make_result(node: Node, **overrides: Any) -> RecallResult:
    defaults: dict[str, Any] = {
        "node": node,
        "score": 1.0,
        "bm25_score": 0.25,
        "vector_score": 0.5,
        "graph_score": 0.0,
        "trigger_score": 0.0,
        "scope_rank": 0,
        "methods": ("bm25", "vector"),
        "path": (),
        "recall_event_id": "recall-evt-1",
    }
    defaults.update(overrides)
    return RecallResult(**defaults)


def shape(results: list[RecallResult], **overrides: Any) -> list[dict[str, Any]]:
    """The production defaults, so the wire shape under test is the shipped one."""

    kwargs: dict[str, Any] = {
        "already_delivered_ids": set(),
        "snippet_max_chars": 1200,
        "context_value_max_chars": 160,
        "session_dedup": True,
    }
    kwargs.update(overrides)
    return shape_recall_results(results, **kwargs)


def deliveries(shaped: list[dict[str, Any]]) -> list[str]:
    return [entry["delivery"] for entry in shaped]


def test_paraphrase_of_a_higher_ranked_result_ships_as_a_stub() -> None:
    """The whole point: the repeat arrives as a preview plus a way back.

    The bearer keeps its text, the repeat costs one line, and the collapsed
    node stays reachable — everything a byte twin already gets, granted to a
    node that is only a *semantic* twin.
    """

    bearer = make_node("bearer", "The ballast survey runs at slack water.\nDetail line.")
    repeat = make_node("repeat", "Slack water is when the ballast survey runs.\nOther.")
    other = make_node("other", "Unrelated fact about the quillon inventory.")
    results = [make_result(bearer), make_result(repeat), make_result(other)]

    shaped = shape(results, duplicate_of={"repeat": "bearer"})

    assert deliveries(shaped) == [DELIVERY_FULL, DELIVERY_NEAR_DUPLICATE, DELIVERY_FULL]
    assert shaped[0]["node"]["content"] == bearer.content  # the text still ships
    assert shaped[1]["node"]["content"] == "Slack water is when the ballast survey runs."
    assert shaped[1]["content_ref"] == {
        "node_id": "repeat",
        "full_content_chars": len(repeat.content),
        "duplicate_of": "bearer",
    }


def test_stub_preview_is_one_bounded_line_like_every_other_stub() -> None:
    long_first_line = "ballast survey " * 40
    repeat = make_node("repeat", f"{long_first_line}\nsecond line")

    shaped = shape(
        [make_result(make_node("bearer", "bearer text")), make_result(repeat)],
        duplicate_of={"repeat": "bearer"},
    )

    preview = shaped[1]["node"]["content"]
    assert "\n" not in preview
    assert len(preview) <= PREVIEW_MAX_CHARS
    assert preview.startswith("ballast survey")


def test_no_map_reproduces_todays_answer_byte_for_byte() -> None:
    """The rollback path through the pure function: absent map, absent effect.

    ``None`` and ``{}`` are both spellings of "the caller built nothing", and
    neither may differ from a call that predates the parameter.
    """

    results = [
        make_result(make_node("a", "first ballast survey fact")),
        make_result(make_node("b", "second ballast survey fact")),
    ]

    legacy = shape(results)

    assert shape(results, duplicate_of=None) == legacy
    assert shape(results, duplicate_of={}) == legacy
    assert DELIVERY_NEAR_DUPLICATE not in deliveries(legacy)


def test_map_naming_no_delivered_node_changes_nothing() -> None:
    """A bearer/duplicate pair from some other answer must not leak in."""

    results = [make_result(make_node("a", "a fact")), make_result(make_node("b", "b fact"))]

    assert shape(results, duplicate_of={"elsewhere": "a"}) == shape(results)


def test_near_duplicate_stub_does_not_consume_a_ladder_position() -> None:
    """A stub is not a bearer, so the ladder must not advance past it.

    Pinned by construction rather than by an arithmetic assertion: the node
    ranked below the stub gets exactly the content it would have got had the
    stub never been in the ranking at all.
    """

    ladder = (LADDER_COMPLETE, 120, 40)
    bearer = make_node("bearer", "b " * 300)
    repeat = make_node("repeat", "r " * 300)
    below = make_node("below", "l " * 300)

    with_stub = shape(
        [make_result(bearer), make_result(repeat), make_result(below)],
        snippet_ladder=ladder,
        duplicate_of={"repeat": "bearer"},
    )
    without_the_repeat = shape(
        [make_result(bearer), make_result(below)],
        snippet_ladder=ladder,
    )

    assert deliveries(with_stub) == [
        DELIVERY_FULL,
        DELIVERY_NEAR_DUPLICATE,
        DELIVERY_SNIPPET,
    ]
    assert with_stub[2] == without_the_repeat[1]


def test_byte_twin_keeps_its_own_class_when_it_is_also_a_near_duplicate() -> None:
    """Byte equality is the more precise statement, so it stays the answer."""

    text = "byte-identical ballast survey fact"
    results = [make_result(make_node("a", text)), make_result(make_node("b", text))]

    shaped = shape(results, duplicate_of={"b": "a"})

    assert deliveries(shaped) == [DELIVERY_FULL, DELIVERY_TWIN_DUPLICATE]
    assert shaped[1]["content_ref"]["duplicate_of"] == "a"


def test_session_duplicate_keeps_its_own_class_and_its_own_content_ref() -> None:
    """The older class is untouched, down to the absence of ``duplicate_of``.

    A node already delivered this session says something a near-dup stub does
    not — "you have read this, in another answer" — and that is the more
    useful thing to tell the agent, so the near-dup rule is checked last.
    """

    results = [
        make_result(make_node("a", "the ballast survey fact")),
        make_result(make_node("b", "the same ballast survey fact, reworded")),
    ]

    shaped = shape(results, already_delivered_ids={"b"}, duplicate_of={"b": "a"})

    assert deliveries(shaped) == [DELIVERY_FULL, DELIVERY_SESSION_DUPLICATE]
    assert "duplicate_of" not in shaped[1]["content_ref"]
    assert shaped[1]["content_ref"]["node_id"] == "b"


def test_shaping_mutates_neither_the_map_nor_the_results() -> None:
    """The purity claim in the docstring, checked rather than asserted."""

    nodes = [make_node("a", "a fact"), make_node("b", "a fact, said again")]
    results = [make_result(node) for node in nodes]
    duplicate_of = {"b": "a"}
    before_map = copy.deepcopy(duplicate_of)
    before_contents = [node.content for node in nodes]

    first = shape(results, duplicate_of=duplicate_of)
    second = shape(results, duplicate_of=duplicate_of)

    assert duplicate_of == before_map
    assert [node.content for node in nodes] == before_contents
    assert first == second  # identical inputs, identical output


# --------------------------------------------------------------------------
# the env valves
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, DEFAULT_NEAR_DUP_COSINE),
        ("", DEFAULT_NEAR_DUP_COSINE),
        ("0", 0.0),
        ("0.0", 0.0),
        (" 0.99 ", 0.99),
        ("-0.5", 0.0),  # negative is rollback, not an inverted threshold
        ("nonsense", DEFAULT_NEAR_DUP_COSINE),
        ("nan", DEFAULT_NEAR_DUP_COSINE),  # NaN loses every comparison it enters
    ],
)
def test_cosine_valve_parsing(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: float
) -> None:
    if raw is not None:
        monkeypatch.setenv(NEAR_DUP_COSINE_ENV, raw)

    assert near_dup_cosine_from_env() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, DEFAULT_NEAR_DUP_LENGTH_RATIO),
        ("0", 0.0),  # protect anything longer at all
        ("0.5", 0.5),
        ("-1", 0.0),
        ("nonsense", DEFAULT_NEAR_DUP_LENGTH_RATIO),
    ],
)
def test_length_ratio_valve_parsing(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: float
) -> None:
    if raw is not None:
        monkeypatch.setenv(NEAR_DUP_LENGTH_RATIO_ENV, raw)

    assert near_dup_length_ratio_from_env() == pytest.approx(expected)


def test_shipped_default_sits_above_the_distinct_fact_band() -> None:
    """0.85-0.95 is where the live corpus keeps *different* facts.

    A default inside that band would collapse distinct facts, which is the one
    failure this feature must not have.
    """

    assert DEFAULT_NEAR_DUP_COSINE == 0.95


# --------------------------------------------------------------------------
# the call site: where the vectors are actually read
# --------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    with MemoryStore(tmp_path / "memory.sqlite3") as opened:
        yield opened


def unit(degrees: float) -> list[float]:
    """A unit vector at ``degrees`` from the x-axis, so cosines read as angles."""

    radians = math.radians(degrees)
    return [math.cos(radians), math.sin(radians), 0.0]


def put_chunks(
    store: MemoryStore, node_id: str, vectors: Sequence[Sequence[float]]
) -> None:
    """Make ``vectors`` the node's complete chunk set, widths and all."""

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


def seed(store: MemoryStore, content: str, vector: Sequence[float] | None) -> Node:
    """One node, with the chunk vector the near-dup comparison will see.

    ``vector=None`` seeds a node that never reached the drain: a row in
    ``nodes``, nothing in ``node_chunk_embeddings``.
    """

    node = store.create_node(
        level="trace",
        content=content,
        context={"scope": SCOPE, "agent": "tester"},
        embedding=[1.0, 0.0, 0.0],
    )
    if vector is None:
        store.replace_node_chunks(node.id, [])
    else:
        put_chunks(store, node.id, [vector])
    return node


def results_for(nodes: Sequence[Node]) -> list[RecallResult]:
    """Wrap nodes as one answer, in the order given: rank is what bears."""

    return [make_result(node, score=1.0 - index / 10) for index, node in enumerate(nodes)]


def test_call_site_maps_a_paraphrase_and_leaves_a_different_fact_alone(
    store: MemoryStore,
) -> None:
    bearer = seed(store, "the ballast survey runs at slack water", unit(0))
    repeat = seed(store, "at slack water is when the survey runs", unit(5))
    distinct = seed(store, "the quillon inventory was recounted", unit(60))

    mapped = _near_duplicate_map(store, results_for([bearer, repeat, distinct]))

    # 5 degrees is cosine 0.996 — above the shipped 0.95; 60 degrees is 0.5.
    assert mapped == {repeat.id: bearer.id}


def test_call_site_never_collapses_a_materially_longer_candidate(
    store: MemoryStore,
) -> None:
    """Same fact plus a new detail: the detail has to reach the agent."""

    bearer = seed(store, "the ballast survey runs at slack water", unit(0))
    longer = seed(
        store,
        "the ballast survey runs at slack water, and the tide gauge reads 2.1m "
        "on the ebb, which is the number the mooring plan was signed against",
        unit(1),
    )

    assert _near_duplicate_map(store, results_for([bearer, longer])) is None


def test_length_ratio_valve_reaches_the_call_site(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard's width is the env value, not a constant frozen in the code."""

    bearer = seed(store, "the ballast survey runs at slack water", unit(0))
    longer = seed(store, "the ballast survey runs at slack water too", unit(1))
    ceiling = len(bearer.content) * (1 + DEFAULT_NEAR_DUP_LENGTH_RATIO)
    assert len(bearer.content) < len(longer.content) <= ceiling  # inside the default band

    answer = results_for([bearer, longer])
    assert _near_duplicate_map(store, answer) == {longer.id: bearer.id}

    monkeypatch.setenv(NEAR_DUP_LENGTH_RATIO_ENV, "0")
    assert _near_duplicate_map(store, answer) is None


def test_call_site_never_collapses_a_node_without_chunk_vectors(
    store: MemoryStore,
) -> None:
    """No vector means "unknown", never "identical to whatever is above"."""

    bearer = seed(store, "the ballast survey runs at slack water", unit(0))
    unvectorized = seed(store, "at slack water is when the survey runs", None)

    assert store.list_node_chunks(unvectorized.id) == []
    assert _near_duplicate_map(store, results_for([bearer, unvectorized])) is None


def test_cosine_zero_builds_no_map_and_reads_no_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The valve's promise is byte-for-byte rollback, including the cost.

    A store that refuses to be read proves the second half: at ``0`` the call
    site returns before the first ``list_node_chunks``, so rolling back costs
    one env read and not a wasted database pass.
    """

    class RefusingStore:
        def list_node_chunks(self, node_id: str) -> Sequence[Any]:
            raise AssertionError(f"read chunk vectors for {node_id} with the valve at 0")

    monkeypatch.setenv(NEAR_DUP_COSINE_ENV, "0")
    answer = results_for([make_node("a", "a fact"), make_node("b", "the same fact")])

    assert _near_duplicate_map(RefusingStore(), answer) is None  # type: ignore[arg-type]


def test_empty_answer_needs_no_map(store: MemoryStore) -> None:
    assert _near_duplicate_map(store, []) is None


# --------------------------------------------------------------------------
# end to end, through memory_recall
# --------------------------------------------------------------------------


def _server(tmp_path: Path) -> tuple[Any, MemoryStore]:
    """A real server whose tools are called directly.

    Direct calls carry no transport identity, so session dedup stays off and
    what the answer shows is the near-dup collapse and nothing else.
    """

    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    return mcp, mcp.memory_store


def _recall(mcp: Any, **arguments: Any) -> dict[str, Any]:
    return mcp.tools["memory_recall"](query=QUERY_STEM, scope=SCOPE, **arguments)


def _seed_answer(mcp: Any, store: MemoryStore) -> tuple[str, str, str]:
    """Seed one answer: two nodes saying one thing, one saying another.

    The first recall is what drains the scope into ``node_chunk_embeddings``;
    the fabricated vectors then replace the drained ones at the encoder's own
    width, so retrieval keeps working while the near-dup cosines are exact
    numbers this test chose rather than whatever the hash backend produced.
    Which of the pair retrieval ranks higher is left to retrieval, so their
    lengths are kept close enough that the length guard cannot fire either way.
    """

    contents = [
        f"{QUERY_STEM}: the ballast survey runs at slack water\n"
        "countersigned by the harbour office on a tuesday",
        f"{QUERY_STEM}: at slack water is when the ballast survey runs\n"
        "countersigned by the harbour office on a monday",
        f"{QUERY_STEM}: the quillon inventory was recounted and came up short\n"
        "the missing crates were logged against the ferry berth",
    ]
    lengths = [len(content) for content in contents[:2]]
    assert max(lengths) <= min(lengths) * (1 + DEFAULT_NEAR_DUP_LENGTH_RATIO)
    ids = [
        store.create_node(
            level="trace",
            content=content,
            context={"scope": SCOPE, "agent": "tester"},
        ).id
        for content in contents
    ]
    _recall(mcp, max_results=5)  # drains the scope into chunk vectors
    width = len(store.list_node_chunks(ids[0])[0].embedding)

    def wide(degrees: float) -> list[float]:
        vector = [0.0] * width
        vector[0] = math.cos(math.radians(degrees))
        vector[1] = math.sin(math.radians(degrees))
        return vector

    put_chunks(store, ids[0], [wide(0)])
    put_chunks(store, ids[1], [wide(4)])  # cosine 0.9976 — a repeat
    put_chunks(store, ids[2], [wide(70)])  # cosine 0.342 — a different fact
    return ids[0], ids[1], ids[2]


def test_recall_answer_collapses_a_repeat_and_lookup_still_returns_it(
    tmp_path: Path,
) -> None:
    """The postcondition, over the real handler: distinct facts in the slots.

    Which of the pair ranks higher is retrieval's business, so the assertion
    is on the shape of the answer — exactly one of them bears the content, the
    other is a stub pointing at it — and on the promise the stub makes:
    ``memory_lookup`` still returns the text in full.
    """

    mcp, store = _server(tmp_path)
    first, second, distinct = _seed_answer(mcp, store)

    response = _recall(mcp, max_results=5)
    by_id = {entry["node"]["id"]: entry for entry in response["results"]}

    assert set(by_id) == {first, second, distinct}
    pair = [by_id[first], by_id[second]]
    stubs = [entry for entry in pair if entry["delivery"] == DELIVERY_NEAR_DUPLICATE]
    bearers = [entry for entry in pair if entry["delivery"] != DELIVERY_NEAR_DUPLICATE]
    assert len(stubs) == 1 and len(bearers) == 1
    stub, bearer = stubs[0], bearers[0]
    assert stub["content_ref"]["duplicate_of"] == bearer["node"]["id"]
    assert by_id[distinct]["delivery"] != DELIVERY_NEAR_DUPLICATE

    stored = store.get_node(stub["node"]["id"])
    assert stub["content_ref"]["full_content_chars"] == len(stored.content)
    assert len(stub["node"]["content"]) < len(stored.content)  # a preview, not the text
    looked_up = mcp.tools["memory_lookup"](node_id=stub["node"]["id"])
    assert looked_up["results"][0]["content"] == stored.content


def test_valve_at_zero_restores_the_byte_only_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same corpus, same query, valve at 0: nobody collapses."""

    mcp, store = _server(tmp_path)
    _seed_answer(mcp, store)

    monkeypatch.setenv(NEAR_DUP_COSINE_ENV, "0")
    response = _recall(mcp, max_results=5)

    assert response["count"] == 3
    assert DELIVERY_NEAR_DUPLICATE not in deliveries(response["results"])
    assert all(entry["delivery"] == DELIVERY_FULL for entry in response["results"])


# --------------------------------------------------------------------------
# the gated trailing-stub drop
# --------------------------------------------------------------------------


def test_near_duplicate_joins_the_droppable_trailing_stub_classes() -> None:
    """The decision the goal asked to make explicit, taken and wired.

    A near-duplicate's text ships in full under a bearer in the very same
    response, so a gated delivery that drops its trailing stub run loses
    nothing the agent is not already reading — the same reasoning that put
    ``twin_duplicate`` in this list, only stronger. What the drop then does
    with the run is unchanged and stays covered by ``test_recall_gating``;
    what is new is membership, and this is the tuple the handler pops against.
    """

    assert DELIVERY_NEAR_DUPLICATE in _DROPPABLE_TRAILING_STUBS
    assert DELIVERY_SESSION_DUPLICATE in _DROPPABLE_TRAILING_STUBS
    assert DELIVERY_TWIN_DUPLICATE in _DROPPABLE_TRAILING_STUBS
    # Content-bearing classes must never be droppable: they are the only two
    # carrying text nothing else within the agent's reach holds.
    assert DELIVERY_FULL not in _DROPPABLE_TRAILING_STUBS
    assert DELIVERY_SNIPPET not in _DROPPABLE_TRAILING_STUBS
