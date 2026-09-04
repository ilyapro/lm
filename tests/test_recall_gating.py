"""End-to-end proof of repeat-gated automatic recall through the real handler.

Every scenario drives ``memory_recall`` exactly as a client does — over
``fastmcp.Client`` in-memory connections (each a distinct transport session)
or direct tool-function calls (no transport identity) — against a real
``MemoryStore``. Queries, scopes, and contents are synthetic and appear in no
replay corpus; the gate decision reads nothing but the per-fingerprint
accounting numbers:

* Fingerprint consistency — the (query, requested scope) fingerprint the
  handler checks equals the one stamped on the recorded recall_event, for
  scoped and scope-less recalls (a scope-less request plans ``global``).
* Warmup then compaction — identical (query, scope) repeated across distinct
  transport sessions serves full content while the fingerprint is below the
  gate thresholds, then explicit gating and trailing-drop opt-ins drop the
  trailing all-stub run (a fully-repeated response arrives empty). With the
  trailing-drop control unset, false, or malformed, the gated run stays
  delivered as re-fetchable ``session_duplicate`` stubs.
* Safe defaults — eligible repeats remain ungated and nonempty when the
  gating control is unset or malformed.
* Novelty — a node never delivered under the fingerprint ships
  content-bearing even while the request is gated.
* Feedback — one feedback link (the remember closure path) resets the
  fingerprint's unlinked streak and restores full delivery.
* Probe cadence — one full delivery goes through every ``probe_every``
  gate-eligible deliveries, so a fingerprint can re-earn links.
* Master valve — unique (organic-like) queries produce byte-identical
  responses with ``LM_RECALL_REPEAT_GATING=1`` versus ``0``.
* Protocol text — the served instructions and tool descriptions keep the
  mandatory recall-before-action / teach-on-correction wording.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastmcp")

from fastmcp import Client

from living_memory import storage as storage_module
from living_memory.scope import resolve_scope
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore, recall_fingerprint

from test_transport_identity import FakeMCP, _structured

GATE_SCOPE = "project:fpgate"
GATE_QUERY = "zaqorim vexilune cradenwisp drift"


@pytest.fixture(autouse=True)
def _default_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every scenario with delivery and repeat overrides absent."""

    for env in (
        "LM_DELIVERY_SNIPPET_CHARS",
        "LM_DELIVERY_SNIPPET_LADDER",
        "LM_DELIVERY_SESSION_DEDUP",
        "LM_DELIVERY_CONTEXT_VALUE_CHARS",
        "LM_DELIVERY_FULL_NODE_DIET",
        "LM_DELIVERY_PROVENANCE_VALUE_CHARS",
        "LM_DELIVERY_STATS_COMPACTION",
        "LM_DELIVERY_SPARSE",
        "LM_AUTO_CONSOLIDATE_POLICY",
        "LM_RECALL_REPEAT_GATING",
        "LM_RECALL_REPEAT_MIN_UNLINKED",
        "LM_RECALL_REPEAT_MAX_LINK_RATE",
        "LM_RECALL_REPEAT_MIN_SESSIONS",
        "LM_RECALL_REPEAT_PROBE_EVERY",
        "LM_RECALL_REPEAT_DROP_TRAILING_STUBS",
        "LM_DECAY_SWEEP_INTERVAL_SEC",
    ):
        monkeypatch.delenv(env, raising=False)


def _fast_gate(monkeypatch: pytest.MonkeyPatch, *, probe_every: int = 0) -> None:
    """Explicitly enable both controls and gate the third eligible repeat.

    With two unlinked deliveries the smoothed link rate is (0+1)/(2+2)=0.25,
    so a loose rate ceiling keeps Laplace smoothing out of the way; sessions
    accrue one per fresh client connection.
    """

    monkeypatch.setenv("LM_RECALL_REPEAT_GATING", "1")
    monkeypatch.setenv("LM_RECALL_REPEAT_DROP_TRAILING_STUBS", "1")
    monkeypatch.setenv("LM_RECALL_REPEAT_MIN_UNLINKED", "2")
    monkeypatch.setenv("LM_RECALL_REPEAT_MIN_SESSIONS", "2")
    monkeypatch.setenv("LM_RECALL_REPEAT_MAX_LINK_RATE", "0.9")
    monkeypatch.setenv("LM_RECALL_REPEAT_PROBE_EVERY", str(probe_every))


def _server(tmp_path: Path) -> tuple[Any, Any]:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    return mcp, mcp.memory_store


async def _recall(client: Client, query: str, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool("memory_recall", {"query": query, **arguments})
    return _structured(result)


async def _fresh_session_recall(mcp: Any, query: str, **arguments: Any) -> dict[str, Any]:
    """One recall on its own connection: a brand-new transport session."""

    async with Client(mcp) as client:
        return await _recall(client, query, **arguments)


def _by_id(response: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["node"]["id"]: entry for entry in response["results"]}


def _event_row(store: Any, event_id: str) -> Any:
    """Raw recall_events columns the RecallEvent model does not expose."""

    row = store.connection.execute(
        "SELECT fingerprint, gated, requested_scope, transport_session_id"
        " FROM recall_events WHERE id = ?",
        (event_id,),
    ).fetchone()
    assert row is not None
    return row


def _gate_content(tag: str, clauses: int = 3) -> str:
    """A distinctive multi-line body below every ladder budget in play."""

    lines = [f"{GATE_QUERY} {tag} ledger"]
    for index in range(clauses):
        lines.append(
            f"{tag} clause {index}: zaqorim cradenwisp paragraph keeping the "
            f"ledger heavy enough that stubbing is measurable on the wire"
        )
    return "\n".join(lines)


def _seed(store: Any, tags: tuple[str, ...]) -> dict[str, str]:
    contents: dict[str, str] = {}
    for tag in tags:
        node = store.append_trace(_gate_content(tag), {"scope": GATE_SCOPE})
        contents[node.id] = node.content
    return contents


def _assert_all_full(response: dict[str, Any], contents: dict[str, str]) -> None:
    by_id = _by_id(response)
    assert set(by_id) == set(contents)
    for node_id, entry in by_id.items():
        assert entry["delivery"] == "full"
        assert entry["node"]["content"] == contents[node_id]
        assert "content_ref" not in entry


def _assert_all_stubbed(response: dict[str, Any], contents: dict[str, str]) -> None:
    """The non-dropping gated shape: every repeat is a re-fetchable stub."""

    by_id = _by_id(response)
    assert set(by_id) == set(contents)
    for node_id, entry in by_id.items():
        assert entry["delivery"] == "session_duplicate"
        stub_content = entry["node"]["content"]
        assert stub_content != contents[node_id]
        assert contents[node_id].startswith(stub_content)  # one-line preview
        ref = entry["content_ref"]
        assert ref["node_id"] == node_id
        assert ref["full_content_chars"] == len(contents[node_id])


def _assert_gated_dropped(response: dict[str, Any]) -> None:
    """With both opt-ins set, the trailing all-stub run is dropped, so a
    fully-repeated delivery arrives empty (its recall_event still records
    every result id — asserted separately where it matters)."""

    assert response["results"] == []
    assert response["count"] == 0


# --- (a) Gate-check fingerprint == recorded recall_event fingerprint --------


def test_gate_fingerprint_matches_recorded_event_for_scoped_and_scopeless(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[Any, dict[str, Any], dict[str, Any]]:
        mcp, store = _server(tmp_path)
        _seed(store, ("anchor",))
        async with Client(mcp) as client:
            scoped = await _recall(client, GATE_QUERY, scope=GATE_SCOPE)
        async with Client(mcp) as client:
            scopeless = await _recall(client, GATE_QUERY)
        return store, scoped, scopeless

    store, scoped, scopeless = asyncio.run(scenario())

    scoped_row = _event_row(store, scoped["recall_event_id"])
    assert scoped_row["requested_scope"] == GATE_SCOPE
    assert scoped_row["fingerprint"] == recall_fingerprint(GATE_QUERY, GATE_SCOPE)

    # Scope-less: no explicit or ambient scope, and no query-token overlap
    # with any stored project name, plans requested_scope 'global'. The
    # reserved transport stamp must never perturb that plan.
    plan = resolve_scope(
        query=GATE_QUERY,
        scope=None,
        ambient_context={"transport_session_id": "any-connection"},
        store=store,
    )
    assert plan.requested_scope == "global"
    scopeless_row = _event_row(store, scopeless["recall_event_id"])
    assert scopeless_row["requested_scope"] == "global"
    assert scopeless_row["fingerprint"] == recall_fingerprint(GATE_QUERY, "global")

    # The requested scope binds the identity: same query, distinct fingerprints.
    assert scopeless_row["fingerprint"] != scoped_row["fingerprint"]


# --- (b) Warmup full across sessions, then compact gated deliveries ----------


def test_repeat_across_sessions_full_warmup_then_compact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fast_gate(monkeypatch)

    async def scenario() -> tuple[Any, dict[str, str], list[dict[str, Any]]]:
        mcp, store = _server(tmp_path)
        contents = _seed(store, ("alpha", "beta"))
        responses = []
        for _ in range(5):
            responses.append(
                await _fresh_session_recall(mcp, GATE_QUERY, scope=GATE_SCOPE)
            )
        return store, contents, responses

    store, contents, responses = asyncio.run(scenario())

    # Every call rode its own transport session, so per-session dedup can
    # never be the stubbing rule here — only the fingerprint history can.
    transports = [
        _event_row(store, response["recall_event_id"])["transport_session_id"]
        for response in responses
    ]
    assert all(transports)
    assert len(set(transports)) == len(transports)

    for response in responses[:2]:  # warmup: below min_unlinked
        _assert_all_full(response, contents)
        assert _event_row(store, response["recall_event_id"])["gated"] == 0
    for response in responses[2:]:
        _assert_gated_dropped(response)
        assert _event_row(store, response["recall_event_id"])["gated"] == 1

    # The recorded events stay unshaped: every delivery logs its result ids,
    # and the aggregate counted each connection as a distinct session.
    for response in responses:
        event = store.get_recall_event(response["recall_event_id"])
        assert set(event.result_ids) == set(contents)
    stats = store.get_recall_fingerprint_stats(
        recall_fingerprint(GATE_QUERY, GATE_SCOPE)
    )
    assert stats.transport_session_count == 5


# --- (b2) Missing/malformed gating never gates eligible repeats -------------


@pytest.mark.parametrize(
    "gating_flag",
    [None, "definitely-not-an-opt-in"],
    ids=("unset", "malformed"),
)
def test_eligible_repeats_stay_ungated_and_nonempty_without_gating_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gating_flag: str | None,
) -> None:
    _fast_gate(monkeypatch)
    if gating_flag is None:
        monkeypatch.delenv("LM_RECALL_REPEAT_GATING")
    else:
        monkeypatch.setenv("LM_RECALL_REPEAT_GATING", gating_flag)

    async def scenario() -> tuple[Any, dict[str, str], list[dict[str, Any]]]:
        mcp, store = _server(tmp_path)
        contents = _seed(store, ("alpha", "beta"))
        responses = [
            await _fresh_session_recall(mcp, GATE_QUERY, scope=GATE_SCOPE)
            for _ in range(4)
        ]
        return store, contents, responses

    store, contents, responses = asyncio.run(scenario())

    # Calls three and four are beyond every pinned threshold that makes the
    # explicit-on fixture gate, so a disabled decision cannot be mistaken for
    # an ineligible fingerprint. All four real-client calls remain useful.
    stats = store.get_recall_fingerprint_stats(
        recall_fingerprint(GATE_QUERY, GATE_SCOPE)
    )
    assert stats.deliveries_since_link >= 2
    assert stats.transport_session_count >= 2
    assert stats.smoothed_link_rate <= 0.9
    for response in responses:
        assert response["count"] == len(contents) > 0
        _assert_all_full(response, contents)
        assert _event_row(store, response["recall_event_id"])["gated"] == 0


# --- (c) A never-delivered node ships content-bearing while gated -----------


def test_novel_node_ships_content_bearing_while_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fast_gate(monkeypatch)

    async def scenario() -> tuple[Any, dict[str, str], Any, dict[str, Any], dict[str, Any]]:
        mcp, store = _server(tmp_path)
        contents = _seed(store, ("alpha", "beta"))
        for _ in range(3):
            gated_response = await _fresh_session_recall(
                mcp, GATE_QUERY, scope=GATE_SCOPE
            )
        novel = store.append_trace(
            _gate_content("novel-omicron"), {"scope": GATE_SCOPE}
        )
        after = await _fresh_session_recall(mcp, GATE_QUERY, scope=GATE_SCOPE)
        return store, contents, novel, gated_response, after

    store, contents, novel, gated_response, after = asyncio.run(scenario())

    _assert_gated_dropped(gated_response)  # the gate really closed
    assert _event_row(store, after["recall_event_id"])["gated"] == 1  # still gated

    # The novel node ships content-bearing; already-delivered nodes ranked
    # above it survive as stubs, while the trailing all-stub run is dropped —
    # so the last delivered entry always bears content.
    by_id = _by_id(after)
    entry = by_id[novel.id]
    assert entry["delivery"] == "full"
    assert entry["node"]["content"] == novel.content
    assert "content_ref" not in entry
    assert after["results"][-1]["node"]["id"] == novel.id
    for node_id, other in by_id.items():
        if node_id != novel.id:
            assert node_id in contents
            assert other["delivery"] == "session_duplicate"

    # Delivery compaction never rewrites history: the recorded event keeps
    # every ranked result id, dropped or not.
    event = store.get_recall_event(after["recall_event_id"])
    assert set(event.result_ids) == set(contents) | {novel.id}


# --- (d) One feedback link restores full delivery ---------------------------


def test_feedback_link_restores_full_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fast_gate(monkeypatch)

    async def scenario() -> tuple[Any, dict[str, str], dict[str, Any], dict[str, Any], dict[str, Any]]:
        mcp, store = _server(tmp_path)
        contents = _seed(store, ("alpha", "beta"))
        for _ in range(2):
            await _fresh_session_recall(mcp, GATE_QUERY, scope=GATE_SCOPE)
        async with Client(mcp) as client:
            gated_response = await _recall(client, GATE_QUERY, scope=GATE_SCOPE)
            # The closure content shares zero tokens with the query: transport
            # identity and scope are the matching rules, as in real usage.
            remembered = _structured(
                await client.call_tool(
                    "memory_remember",
                    {
                        "content": "tessellated brumal cogfern spindle closure note",
                        "context": {"scope": GATE_SCOPE},
                    },
                )
            )
        after = await _fresh_session_recall(mcp, GATE_QUERY, scope=GATE_SCOPE)
        return store, contents, gated_response, remembered, after

    store, contents, gated_response, remembered, after = asyncio.run(scenario())

    _assert_gated_dropped(gated_response)
    assert (
        gated_response["recall_event_id"]
        in remembered["implicit_feedback"]["recall_event_ids"]
    )
    # What restores delivery is the *link*, not the reinforcement: this
    # closure content shares zero tokens with the seeded results, so grounded
    # credit withholds reinforcement while the link still lands and lifts the
    # gate below.
    assert set(remembered["implicit_feedback"]["linked_node_ids"]) >= set(contents)
    assert remembered["implicit_feedback"]["feedback_applied"] is False

    stats = store.get_recall_fingerprint_stats(
        recall_fingerprint(GATE_QUERY, GATE_SCOPE)
    )
    assert stats.linked_count >= 1

    # Full delivery is restored: no entry is stubbed anymore. The closure
    # trace may join the results via its fresh related edges — as one more
    # content-bearing delivery.
    by_id = _by_id(after)
    assert set(contents) <= set(by_id)
    for node_id, entry in by_id.items():
        assert entry["delivery"] == "full"
        if node_id in contents:
            assert entry["node"]["content"] == contents[node_id]
    assert _event_row(store, after["recall_event_id"])["gated"] == 0


# --- (e) Probe cadence periodically serves full ------------------------------


def test_probe_cadence_periodically_serves_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fast_gate(monkeypatch, probe_every=3)

    async def scenario() -> tuple[Any, dict[str, str], list[dict[str, Any]]]:
        mcp, store = _server(tmp_path)
        contents = _seed(store, ("probe",))
        responses = []
        for _ in range(9):
            responses.append(
                await _fresh_session_recall(mcp, GATE_QUERY, scope=GATE_SCOPE)
            )
        return store, contents, responses

    store, contents, responses = asyncio.run(scenario())

    # Unlinked-streak walk with min_unlinked=2, probe_every=3: calls 1-2 are
    # warmup, then every third gate-eligible delivery (streak 2, 5, 8 before
    # the call) is a probe served full so the fingerprint can re-earn links.
    expected_gated = [False, False, False, True, True, False, True, True, False]
    for response, should_gate in zip(responses, expected_gated, strict=True):
        row = _event_row(store, response["recall_event_id"])
        assert bool(row["gated"]) is should_gate
        if should_gate:
            _assert_gated_dropped(response)
        else:
            _assert_all_full(response, contents)


# --- (e2) Trailing-drop is an independent strict opt-in ----------------------


@pytest.mark.parametrize(
    "drop_flag",
    [None, "0", "definitely-not-an-opt-in"],
    ids=("unset", "known-false", "malformed"),
)
def test_explicit_gating_without_drop_opt_in_keeps_refetchable_stubs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drop_flag: str | None,
) -> None:
    _fast_gate(monkeypatch)
    if drop_flag is None:
        monkeypatch.delenv("LM_RECALL_REPEAT_DROP_TRAILING_STUBS")
    else:
        monkeypatch.setenv("LM_RECALL_REPEAT_DROP_TRAILING_STUBS", drop_flag)

    async def scenario() -> tuple[
        Any,
        dict[str, str],
        dict[str, Any],
        dict[str, Any],
    ]:
        mcp, store = _server(tmp_path)
        contents = _seed(store, ("alpha", "beta"))
        for _ in range(2):
            await _fresh_session_recall(mcp, GATE_QUERY, scope=GATE_SCOPE)
        async with Client(mcp) as client:
            response = await _recall(client, GATE_QUERY, scope=GATE_SCOPE)
            fetched = _structured(
                await client.call_tool(
                    "memory_lookup",
                    {
                        "node_ids": [
                            entry["content_ref"]["node_id"]
                            for entry in response["results"]
                        ]
                    },
                )
            )
        return store, contents, response, fetched

    store, contents, response, fetched = asyncio.run(scenario())

    assert _event_row(store, response["recall_event_id"])["gated"] == 1
    assert response["count"] == len(contents) > 0
    _assert_all_stubbed(response, contents)

    # Follow the advertised content_ref path over the same real client: the
    # nonempty stubs retain direct, byte-complete access to every ranked node.
    assert fetched["missing"] == []
    assert fetched["count"] == len(contents)
    assert {node["id"]: node["content"] for node in fetched["results"]} == contents


# --- (f) Master valve: unique queries byte-identical on vs off ---------------


def test_unique_queries_byte_identical_with_gating_on_vs_valve_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Freeze every volatility source so twin stores replay identically:
    # timestamps, event ULIDs, and the opportunistic decay sweep.
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
    monkeypatch.setattr(storage_module, "_utc_now", lambda: "2026-01-01T00:00:00Z")
    state = {"n": 0}

    def fake_ulid() -> str:
        state["n"] += 1
        return f"01FAKEULID{state['n']:016d}"

    monkeypatch.setattr(storage_module, "new_ulid", fake_ulid)

    seed = tmp_path / "seed.sqlite3"
    with MemoryStore(seed) as store:
        _seed(store, ("alpha", "beta"))
        store.append_trace(
            "quivelt pandrossum ledger entry kept in the global scope",
            {"scope": "global"},
        )
    twins = (tmp_path / "a.sqlite3", tmp_path / "b.sqlite3")
    for twin in twins:
        for suffix in ("", "-wal", "-shm"):
            source = Path(str(seed) + suffix)
            if source.exists():
                shutil.copy(source, str(twin) + suffix)

    # Never-seen (query, scope) pairs — one scoped, one scope-less.
    unique_calls = (
        (f"{GATE_QUERY} unique aurelic", {"scope": GATE_SCOPE}),
        ("quivelt pandrossum ledger", {}),
    )

    def run(db_path: Path) -> list[dict[str, Any]]:
        mcp = create_mcp_server(db_path, mcp_factory=FakeMCP)
        state["n"] = 10_000  # identical event-id sequences across both runs
        return [
            mcp.tools["memory_recall"](query, **arguments)
            for query, arguments in unique_calls
        ]

    # Compare the actual master-valve states. Keep the independent drop
    # control opted in on both sides so only repeat gating changes.
    monkeypatch.setenv("LM_RECALL_REPEAT_GATING", "1")
    monkeypatch.setenv("LM_RECALL_REPEAT_DROP_TRAILING_STUBS", "1")
    responses_on = run(twins[0])
    monkeypatch.setenv("LM_RECALL_REPEAT_GATING", "0")
    responses_off = run(twins[1])

    for on, off in zip(responses_on, responses_off, strict=True):
        assert on["count"] >= 1
        assert off["count"] >= 1  # an empty-vs-empty tie would prove nothing
        assert json.dumps(on, sort_keys=True, ensure_ascii=False) == json.dumps(
            off, sort_keys=True, ensure_ascii=False
        )


# --- (g) Protocol wording survives on the served surfaces --------------------


def test_protocol_wording_served_intact(tmp_path: Path) -> None:
    async def scenario() -> tuple[str, dict[str, str]]:
        mcp, _ = _server(tmp_path)
        async with Client(mcp) as client:
            instructions = client.initialize_result.instructions
            tools = {
                tool.name: tool.description for tool in await client.list_tools()
            }
        return instructions, tools

    instructions, tools = asyncio.run(scenario())

    assert "You MUST recall BEFORE you act" in instructions
    assert "You MUST teach the moment a belief changes" in instructions
    assert "You MUST recall BEFORE acting" in tools["memory_recall"]
    # The capped protocol channel funds protocol only: no agent can act on a
    # gate an operator has to flip, so the served description must not spend
    # its budget advertising this one — and the chars it freed must show up as
    # the contentful opener (tests/test_instructions_imperative.py pins the
    # general ban and the opener).
    assert "Experimental repeat gating is default-off." not in tools["memory_recall"]
    assert "Recall before acting and at every new turn of thought." in tools["memory_recall"]
    assert len(tools["memory_recall"]) <= 1024
    assert "MUST teach the moment a belief changes" in tools["memory_teach"]
    assert "supersedes edge" in tools["memory_teach"]
