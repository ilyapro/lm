"""The recall map's second delivery: server instructions, refreshed per session.

The response channel reaches an agent only after it has decided to recall.
This one reaches it before it acts at all — every MCP client hands the server
instructions to the agent at ``initialize``. Two things have to be true for
that to work, and this module pins both.

**The splice displaces nothing.** ``_server_instructions`` grew one additive
keyword and the section goes at the TAIL, after the default-scope line.
Clients clip this text at 2048 chars *from the front*, so a clip lands on
whatever sits last: the tail is the only place the lowest-priority, most
volatile content can go without putting the three laws at risk. With an empty
section the returned text is byte-identical to the static one — that is both
the zero-displacement guarantee and the degradation path for a store with no
history, a store that cannot answer, and the rolled-back valve.

**The refresh actually lands.** ``FastMCP`` takes ``instructions`` as a
construction kwarg and never re-reads the composed text, so the wiring cannot
be checked by reading it back off the object that composed it. What the
protocol serves is ``Server.create_initialization_options().instructions``,
read at call time, once per transport session by
``StreamableHTTPSessionManager``. Every test here that claims a refresh
"reached" a session asserts on *that* call and nothing weaker.

The refresh hangs off the recall/remember tool path, which buys it two
obligations it is measured against below: it never raises into the tool, and
it costs a bounded indexed read rather than anything an agent waits on.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastmcp")

from fastmcp import Client

from living_memory.embeddings import LocalEmbeddingModel
from living_memory.instructions_map import (
    HEADING,
    LINE_PREFIX,
    MAX_LABELS,
    MAX_SECTION_CHARS,
    compose_map_section,
)
from living_memory.server import (
    _INSTRUCTIONS_MAP_HISTORY_ROWS,
    _InstructionsRefresh,
    _instructions_with_map,
    _server_instructions,
    create_mcp_server,
)

from test_transport_identity import FakeMCP

# The client-side clip, restated here because this module's whole claim is
# about which end of the string it eats. Pinned as a budget in
# tests/test_instructions_imperative.py.
MAX_INSTRUCTION_CHARS = 2048

# Verbatim protocol that must outrank the map for position. Not a restatement
# of the presence contract (that lives in test_instructions_imperative.py) —
# these are here only so "the map sits last" can be checked against something.
PROTOCOL_PINS = (
    "ONE cognitive system",
    "You MUST recall BEFORE you act",
    "You MUST remember at the moment of insight",
    "You MUST teach the moment a belief changes",
    "load the Living Memory tools NOW",
)

SCOPES = (
    "global",
    "project:lm",
    "project:custom-scope",
    "project:a-fairly-long-scope-name-for-headroom",
)


def _map_payload(*labelled: tuple[str, int], pool: int = 60, more: int = 0) -> dict[str, Any]:
    """A persisted ``recall_map`` payload, shaped as ``RecallMap.to_dict``."""

    clusters = [
        {
            "label": label,
            "count": count,
            "medoid": {"node_id": f"node-{index}", "example": f"an example of {label}"},
            "ask_hint": f"what does memory hold about {label}",
            "plan_item": f"when touching {label} — ask memory about {label}",
        }
        for index, (label, count) in enumerate(labelled)
    ]
    payload: dict[str, Any] = {
        "clusters": clusters,
        "pool": pool,
        "covered": sum(count for _, count in labelled),
    }
    if more:
        payload["more"] = more
    return payload


def _full_section() -> str:
    """The largest section the composer will actually emit."""

    section = compose_map_section(
        [
            {
                "recall_map": _map_payload(
                    ("instructions channel", 9),
                    ("recall map clustering", 7),
                    ("delivery diet", 5),
                    ("transport identity", 4),
                    ("consolidation", 3),
                    ("latency budget", 2),
                    more=4,
                )
            }
        ]
    )
    assert section, "the fixture must produce a section for these tests to mean anything"
    return section


def _seed_delivered_map(store: Any, payload: dict[str, Any]) -> None:
    """Persist ``payload`` as the map the most recent recall delivered.

    Written straight onto the recall event rather than coaxed out of the
    builder: what these tests are about is the *history* the composer reads,
    and going through the builder would couple them to its clustering
    thresholds without exercising one line of the wiring.
    """

    store.attach_recall_map(store.list_recall_events()[0].id, payload)


# ── The splice: additive, tail-placed, byte-identical when empty ────────────


@pytest.mark.parametrize("scope", SCOPES)
def test_empty_section_reproduces_the_static_text_byte_for_byte(scope: str) -> None:
    """Zero displacement: the map is additive or it is nothing.

    A blank section is checked alongside the empty one because the composer
    returns ``""`` and the splice strips: neither may leave a dangling blank
    line, which would be a one-byte difference in a text pinned to the byte.
    """

    static = _server_instructions(scope)

    assert _server_instructions(scope, map_section="") == static
    assert _server_instructions(scope, map_section="   \n  \t ") == static
    assert static.endswith(f"Your default scope is {scope}.\n")
    assert HEADING not in static


@pytest.mark.parametrize("scope", SCOPES)
def test_section_is_spliced_at_the_tail_and_nowhere_else(scope: str) -> None:
    section = _full_section()
    static = _server_instructions(scope)
    text = _server_instructions(scope, map_section=section)

    # Everything the static text said, in the order it said it, untouched.
    assert text.startswith(static)
    # And the only thing added is the section, blank-line separated.
    assert text[len(static) :] == f"\n{section}\n"


def test_the_first_char_a_front_clip_drops_belongs_to_the_map() -> None:
    """Why the tail, stated as the property that motivates it.

    Clients clip from the front, so position is priority. The map occupies
    the final run of the string: a clip has to consume the whole section
    before it can touch the default-scope line, and the three laws and the
    bootstrap directive are further from the cut than anything else.
    """

    scope = "project:a-fairly-long-scope-name-for-headroom"
    text = _server_instructions(scope, map_section=_full_section())

    assert text.index(HEADING) > max(text.index(pin) for pin in PROTOCOL_PINS)
    # The clip that costs one character costs it from the map.
    assert len(text) - 1 >= text.index(HEADING)
    # And at the real budget nothing protocol-bearing is near the cut.
    for pin in PROTOCOL_PINS:
        assert pin in text[:MAX_INSTRUCTION_CHARS]


def test_section_budget_bounds_what_the_splice_can_add() -> None:
    """The splice adds the section plus exactly two separator newlines."""

    static = _server_instructions("global")
    text = _server_instructions("global", map_section=_full_section())

    assert len(text) - len(static) <= MAX_SECTION_CHARS + 2


# ── Composition from a live store ───────────────────────────────────────────


def test_empty_history_composes_the_static_text(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store

    assert _instructions_with_map(store, "global") == _server_instructions("global")
    # Which is also what the server was constructed with.
    assert mcp.instructions == _server_instructions(store.config.default_scope)


def test_delivered_map_reaches_the_composed_text(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    mcp.tools["memory_recall"]("what does memory hold about deploys")
    _seed_delivered_map(store, _map_payload(("deploy recipes", 3), ("bundle layout", 2)))

    composed = _instructions_with_map(store, store.config.default_scope)

    assert composed.endswith("\n")
    assert HEADING in composed
    assert f"{LINE_PREFIX}deploy recipes(3)" in composed
    assert composed.startswith(_server_instructions(store.config.default_scope))


def test_a_restarted_process_boots_with_the_map_its_predecessor_persisted(
    tmp_path: Path,
) -> None:
    """Composition at construction, not only on refresh — and it is load-bearing.

    Over stdio there is one transport session per process, so
    ``create_initialization_options()`` is called once, at start: everything
    a stdio session ever sees of the map is what construction composed. A
    server that only ever learned the map by refreshing would serve an empty
    section to every stdio client forever, and would lose the map across each
    restart of the HTTP one.
    """

    db_path = tmp_path / "memory.sqlite3"
    first = create_mcp_server(db_path, mcp_factory=FakeMCP)
    first.tools["memory_recall"]("what does memory hold about deploys")
    _seed_delivered_map(first.memory_store, _map_payload(("deploy recipes", 3)))

    restarted = create_mcp_server(db_path)

    # No tool has been called on this process, and no refresh has run.
    served = restarted._mcp_server.create_initialization_options().instructions
    assert HEADING in served
    assert f"{LINE_PREFIX}deploy recipes(3)" in served
    assert served.startswith(
        _server_instructions(restarted.memory_store.config.default_scope)
    )


def test_rolled_back_valve_removes_the_section_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``LM_RECALL_MAP=0`` is the rollback for the whole map, not one channel."""

    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    mcp.tools["memory_recall"]("what does memory hold about deploys")
    _seed_delivered_map(store, _map_payload(("deploy recipes", 3)))
    assert HEADING in _instructions_with_map(store, store.config.default_scope)

    monkeypatch.setenv("LM_RECALL_MAP", "0")

    assert _instructions_with_map(store, store.config.default_scope) == _server_instructions(
        store.config.default_scope
    )


# ── The refresh, asserted where the protocol reads it ───────────────────────


def test_refresh_reaches_the_next_sessions_initialize(tmp_path: Path) -> None:
    """The load-bearing test: mutate through the hook, read the served text.

    ``mcp._mcp_server.create_initialization_options()`` is what
    ``StreamableHTTPSessionManager`` calls inside each ``app.run(...)``, so
    what it reports here is what the next transport session's ``initialize``
    carries. The session driving the calls below already has its options and
    is deliberately not expected to see the change.
    """

    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    boot = mcp._mcp_server.create_initialization_options().instructions
    assert boot == _server_instructions(store.config.default_scope)

    async def scenario() -> None:
        async with Client(mcp) as client:
            await client.call_tool(
                "memory_recall", {"query": "what does memory hold about deploys"}
            )
            _seed_delivered_map(
                store, _map_payload(("deploy recipes", 3), ("bundle layout", 2))
            )
            await client.call_tool(
                "memory_remember",
                {"content": "The alt bundle recipe pins its own interpreter path."},
            )

    asyncio.run(scenario())

    served = mcp._mcp_server.create_initialization_options().instructions
    assert served != boot
    assert served.startswith(boot)
    assert HEADING in served
    assert f"{LINE_PREFIX}deploy recipes(3)" in served
    assert len(served) <= MAX_INSTRUCTION_CHARS


def test_recall_refreshes_too_and_a_second_map_supersedes_the_first(
    tmp_path: Path,
) -> None:
    """Both hooks are live, and the served text tracks the newest history."""

    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store

    async def scenario() -> list[str]:
        served: list[str] = []
        async with Client(mcp) as client:
            await client.call_tool("memory_recall", {"query": "first probe"})
            _seed_delivered_map(store, _map_payload(("deploy recipes", 3)))
            # A recall — not a remember — is what carries this one across.
            await client.call_tool("memory_recall", {"query": "second probe"})
            served.append(mcp._mcp_server.create_initialization_options().instructions)

            _seed_delivered_map(store, _map_payload(("latency budget", 8)))
            await client.call_tool("memory_recall", {"query": "third probe"})
            served.append(mcp._mcp_server.create_initialization_options().instructions)
        return served

    first, second = asyncio.run(scenario())

    assert f"{LINE_PREFIX}deploy recipes(3)" in first
    # Newest first: the map the latest recall delivered leads the line.
    assert f"{LINE_PREFIX}latency budget(8)" in second
    assert "deploy recipes(3)" in second
    assert first != second


def test_unchanged_history_does_not_rewrite_the_served_text(tmp_path: Path) -> None:
    """No churn: an unchanged composition is not assigned.

    Instructions are cached per session by clients; a text that is rewritten
    on every tool call for no reason is a text whose identity means nothing.
    """

    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    writes: list[str] = []

    class RecordingLowLevel:
        @property
        def instructions(self) -> str | None:
            return writes[-1] if writes else None

        @instructions.setter
        def instructions(self, value: str) -> None:
            writes.append(value)

    mcp._mcp_server = RecordingLowLevel()
    refresh = _InstructionsRefresh(mcp, store)

    refresh()
    refresh()
    assert writes == []

    mcp.tools["memory_recall"]("what does memory hold about deploys")
    _seed_delivered_map(store, _map_payload(("deploy recipes", 3)))
    refresh()
    refresh()
    refresh()

    assert len(writes) == 1
    assert HEADING in writes[0]


# ── Degradation: the hook is decoration, never a cost the tool pays ─────────


def test_factory_double_without_low_level_server_is_a_silent_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``create_mcp_server`` accepts doubles; a double owes us no ``_mcp_server``.

    Two things keep this from passing vacuously. The composition *would* have
    produced a section, so the no-op is the missing attribute and not an empty
    history. And the absent object is checked *before* any work: a double —
    every fake-factory test in this repo, and there are many — must not pay a
    query per tool call for a refresh that has nowhere to land.
    """

    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    assert not hasattr(mcp, "_mcp_server")
    before = mcp.instructions

    mcp.tools["memory_recall"]("what does memory hold about deploys")
    _seed_delivered_map(store, _map_payload(("deploy recipes", 3)))

    reads: list[int] = []
    original = store.recent_recall_map_history

    def counted(**kwargs: Any) -> Any:
        reads.append(1)
        return original(**kwargs)

    monkeypatch.setattr(store, "recent_recall_map_history", counted)

    # Both hooks run against an object with nothing to refresh.
    mcp.tools["memory_recall"]("what does memory hold about deploys")
    remembered = mcp.tools["memory_remember"]("A trace that must survive the no-op.")

    assert remembered["node"]["id"]
    assert mcp.instructions == before
    assert reads == []
    monkeypatch.undo()
    assert HEADING in _instructions_with_map(store, store.config.default_scope)


def test_refresh_swallows_a_store_that_cannot_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A map is decoration on instructions; losing it must not lose a write.

    The store is broken *after* a section is already being served, so the
    test pins the whole degradation and not just the absence of a traceback:
    the tools keep working, and what gets served falls back to the static
    text rather than to a half-composed one or a stale one.
    """

    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    static = _server_instructions(store.config.default_scope)

    class LowLevel:
        instructions: str | None = None

    low_level = LowLevel()
    mcp._mcp_server = low_level

    mcp.tools["memory_recall"]("what does memory hold about deploys")
    _seed_delivered_map(store, _map_payload(("deploy recipes", 3)))
    mcp.tools["memory_recall"]("a second query, which carries the map across")
    assert HEADING in (low_level.instructions or "")

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("the database is on fire")

    monkeypatch.setattr(store, "recent_recall_map_history", explode)

    remembered = mcp.tools["memory_remember"]("A trace written while history was broken.")
    recalled = mcp.tools["memory_recall"]("a query issued while history was broken")

    assert remembered["node"]["id"]
    assert recalled["query"] == "a query issued while history was broken"
    assert low_level.instructions == static
    assert _instructions_with_map(store, "global") == _server_instructions("global")


def test_refresh_makes_no_model_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No LLM, no embedding, on initialize or on refresh."""

    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    mcp.tools["memory_recall"]("what does memory hold about deploys")
    _seed_delivered_map(store, _map_payload(("deploy recipes", 3)))

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the refresh path must not embed anything")

    monkeypatch.setattr(LocalEmbeddingModel, "embed", forbidden)

    composed = _instructions_with_map(store, store.config.default_scope)

    assert HEADING in composed


def test_refresh_reads_a_window_bounded_independently_of_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook sits on the recall path, so its cost may not follow the store.

    Pinned as the window rather than as a duration, because the window is
    what the cost is linear in: composition screens every label of every
    cluster of every row against the register ban battery, so widening this
    is the regression that would actually be felt. Measured 2026-08-20 over a
    40-map history, three rows compose in 0.13ms against 0.68ms at twenty.
    """

    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    for index in range(40):
        mcp.tools["memory_recall"](f"probe {index} for the history window")
        _seed_delivered_map(
            store, _map_payload((f"cluster {index}", index + 1), more=index)
        )

    limits: list[int | None] = []
    original = store.recent_recall_map_history

    def spy(**kwargs: Any) -> Any:
        limits.append(kwargs.get("limit"))
        return original(**kwargs)

    monkeypatch.setattr(store, "recent_recall_map_history", spy)
    refresh = _InstructionsRefresh(mcp, store)
    mcp._mcp_server = type("LowLevel", (), {"instructions": None})()

    refresh()

    assert limits == [_INSTRUCTIONS_MAP_HISTORY_ROWS]
    assert _INSTRUCTIONS_MAP_HISTORY_ROWS <= MAX_LABELS
    # The window still fills the section: three maps carry more candidate
    # labels than the six slots the line has.
    assert HEADING in (mcp._mcp_server.instructions or "")


def test_refresh_stays_far_under_the_recall_it_hangs_off(tmp_path: Path) -> None:
    """A coarse tripwire, not a performance target.

    Deliberately loose: what it exists to catch is the refresh acquiring
    something categorically different — a full scan, a model call, a network
    hop — not a few microseconds of drift. Measured 2026-08-20: 0.13ms median
    against a ~9ms recall.
    """

    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    for index in range(40):
        mcp.tools["memory_recall"](f"probe {index} for the history window")
        _seed_delivered_map(
            store, _map_payload((f"cluster {index}", index + 1), more=index)
        )
    refresh = _InstructionsRefresh(mcp, store)

    samples: list[float] = []
    for _ in range(20):
        started = time.perf_counter()
        refresh()
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    median = samples[len(samples) // 2]

    assert median < 5, f"refresh median {median:.2f}ms — it is doing more than one read"
