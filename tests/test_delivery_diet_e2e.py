"""End-to-end proof of the recall delivery diet over a real MCP client.

Every scenario drives the real FastMCP in-memory client (``fastmcp.Client``)
against ``create_mcp_server`` so transport-session identity, delivery shaping,
and the compact write confirmation are exercised exactly as a client sees
them on the wire:

* Session dedup — the first recall on a connection delivers full content, the
  repeat delivers ``session_duplicate`` stubs and a substantially smaller
  serialized response; a fresh connection gets full content again, and calls
  without any transport identity (direct tool-function calls, the degradation
  control) never stub.
* Twin dedup — a legacy byte-identical concept+trace pair, seeded through
  direct store writes exactly as pre-digest consolidation used to persist
  them, yields exactly one content-bearer per response; the twin stub points
  at it via ``content_ref.duplicate_of``.
* Snippets — content longer than the snippet limit arrives truncated with a
  ``content_ref``, and ``memory_lookup(node_id=...)`` returns the stored
  bytes unchanged (the re-fetch path).
* Remember diet — the ``memory_remember`` wire response stays bounded while
  the trace itself is durably stored and recallable.
* Feedback closure — recall -> stub recall -> remember on one connection
  still closes both events (``feedback_applied`` 0 -> 1) and lands them in
  the new trace's provenance: shaping the response never touches the
  recorded event. The remember content shares zero tokens with the recall
  query, so transport identity is the only available matching rule.
* Digest consolidation — ``memory_consolidate`` over a seeded near-identical
  cluster returns a concept byte-distinct from every source trace.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastmcp")

from fastmcp import Client

from living_memory.embeddings import tokenize
from living_memory.server import create_mcp_server

from test_transport_identity import FakeMCP, _structured

SNIPPET_LIMIT = 1200  # default LM_DELIVERY_SNIPPET_CHARS, pinned by the env fixture

RECALL_SCOPE = "project:dietprobe"
RECALL_STEM = "meridian dossier torque calibration"


@pytest.fixture(autouse=True)
def _default_delivery_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every scenario under default delivery/consolidation settings."""

    for env in (
        "LM_DELIVERY_SNIPPET_CHARS",
        "LM_DELIVERY_SESSION_DEDUP",
        "LM_AUTO_CONSOLIDATE_POLICY",
    ):
        monkeypatch.delenv(env, raising=False)


def _server(tmp_path: Path) -> tuple[Any, Any]:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    return mcp, mcp.memory_store


async def _recall(client: Client, query: str, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool("memory_recall", {"query": query, **arguments})
    return _structured(result)


def _wire_bytes(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, sort_keys=True))


def _by_id(response: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["node"]["id"]: entry for entry in response["results"]}


def _bulk_content(tag: str) -> str:
    """A body heavy enough to measure, yet below the snippet limit."""

    lines = [f"{RECALL_STEM} {tag} ledger"]
    for index in range(9):
        lines.append(
            f"{tag} clause {index}: torque calibration paragraph keeping the "
            f"ledger dense enough to make full delivery measurably heavy"
        )
    return "\n".join(lines)


# --- Session dedup: full once per transport session, stubs on repeat --------


def test_session_dedup_full_then_stub_then_fresh_session_full_again(
    tmp_path: Path,
) -> None:
    async def scenario() -> tuple[dict[str, str], dict[str, Any], dict[str, Any], dict[str, Any]]:
        mcp, store = _server(tmp_path)
        contents: dict[str, str] = {}
        for tag in ("alpha", "beta", "gamma"):
            node = store.append_trace(_bulk_content(tag), {"scope": RECALL_SCOPE})
            contents[node.id] = node.content

        async with Client(mcp) as client:
            first = await _recall(client, RECALL_STEM, scope=RECALL_SCOPE)
            second = await _recall(client, RECALL_STEM, scope=RECALL_SCOPE)
        async with Client(mcp) as client:
            fresh = await _recall(client, RECALL_STEM, scope=RECALL_SCOPE)
        return contents, first, second, fresh

    contents, first, second, fresh = asyncio.run(scenario())

    for content in contents.values():
        assert 800 < len(content) < SNIPPET_LIMIT  # full inline, no snippeting
    assert set(_by_id(first)) == set(_by_id(second)) == set(_by_id(fresh)) == set(contents)

    for node_id, entry in _by_id(first).items():
        assert entry["delivery"] == "full"
        assert "content_ref" not in entry
        assert entry["node"]["content"] == contents[node_id]
    full_node_keys = {node_id: set(entry["node"]) for node_id, entry in _by_id(first).items()}

    for node_id, entry in _by_id(second).items():
        assert entry["delivery"] == "session_duplicate"
        stub = entry["node"]
        assert stub["content"] != contents[node_id]
        assert len(stub["content"]) <= 200  # a one-line preview, not the body
        assert contents[node_id].startswith(stub["content"])
        assert set(stub) == full_node_keys[node_id]  # stubs keep the full node shape
        ref = entry["content_ref"]
        assert ref["node_id"] == node_id
        assert ref["full_content_chars"] == len(contents[node_id])
        assert node_id in ref["fetch"]

    # The repeat response is strictly and substantially smaller on the wire.
    first_bytes, second_bytes = _wire_bytes(first), _wire_bytes(second)
    assert second_bytes < first_bytes
    assert second_bytes <= 0.7 * first_bytes

    # A new connection is a new transport session: full content again.
    for node_id, entry in _by_id(fresh).items():
        assert entry["delivery"] == "full"
        assert entry["node"]["content"] == contents[node_id]


def test_direct_calls_without_transport_identity_never_stub(tmp_path: Path) -> None:
    """Degradation control: no request context means no transport session id,
    so repeated identical recalls keep legacy full delivery every time."""

    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    expected: dict[str, str] = {}
    for tag in ("alpha", "beta"):
        node = store.append_trace(_bulk_content(tag), {"scope": RECALL_SCOPE})
        expected[node.id] = node.content

    responses = [mcp.tools["memory_recall"](RECALL_STEM, scope=RECALL_SCOPE) for _ in range(2)]

    for response in responses:
        by_id = _by_id(response)
        assert set(by_id) == set(expected)
        for node_id, entry in by_id.items():
            assert entry["delivery"] == "full"
            assert "content_ref" not in entry
            assert entry["node"]["content"] == expected[node_id]

    events = store.list_recall_events()
    assert len(events) == 2
    assert all(event.transport_session_id is None for event in events)


# --- Twin dedup: a legacy byte-identical concept+trace pair -----------------


def test_twin_dedup_delivers_exactly_one_content_bearer(tmp_path: Path) -> None:
    scope = "project:dietwin"
    body = (
        "obsidian relay charter checklist\n"
        "grounding verification, antenna sweep, and a signed handover entry "
        "are required before the relay charter is considered active."
    )

    async def scenario() -> tuple[str, str, dict[str, Any]]:
        mcp, store = _server(tmp_path)
        trace = store.append_trace(body, {"scope": scope})
        # Pre-digest consolidation copied trace bytes verbatim into concepts;
        # seed that exact legacy shape through direct store writes.
        concept = store.create_node(
            level="concept",
            content=body,
            context={"scope": scope, "agent": "memory_consolidate"},
            provenance={"source_traces": [trace.id]},
        )
        assert store.get_node(trace.id).decayed is False  # the pair coexists
        async with Client(mcp) as client:
            response = await _recall(client, "obsidian relay charter", scope=scope)
        return trace.id, concept.id, response

    trace_id, concept_id, response = asyncio.run(scenario())

    by_id = _by_id(response)
    assert set(by_id) == {trace_id, concept_id}

    deliveries = {node_id: entry["delivery"] for node_id, entry in by_id.items()}
    bearers = [nid for nid, delivery in deliveries.items() if delivery == "full"]
    stubs = [nid for nid, delivery in deliveries.items() if delivery == "twin_duplicate"]
    assert len(bearers) == 1 and len(stubs) == 1  # never both full
    bearer_id, stub_id = bearers[0], stubs[0]

    assert by_id[bearer_id]["node"]["content"] == body
    assert "content_ref" not in by_id[bearer_id]

    stub = by_id[stub_id]
    assert stub["node"]["content"] != body
    assert body.startswith(stub["node"]["content"])  # one-line preview
    ref = stub["content_ref"]
    assert ref["duplicate_of"] == bearer_id
    assert ref["node_id"] == stub_id
    assert ref["full_content_chars"] == len(body)


# --- Snippets: long content truncated inline, re-fetched via lookup ---------


def test_long_content_snippet_and_lookup_refetch(tmp_path: Path) -> None:
    scope = "project:dietlong"
    paragraphs = ["glacier archive manifest overview"]
    for index in range(12):
        paragraphs.append(
            f"glacier archive manifest section {index}: "
            + "retention policy detail sentence for the archived manifest. " * 3
        )
    long_content = "\n\n".join(paragraphs)

    async def scenario() -> tuple[str, dict[str, Any], dict[str, Any]]:
        mcp, store = _server(tmp_path)
        node = store.append_trace(long_content, {"scope": scope})
        async with Client(mcp) as client:
            recall = await _recall(client, "glacier archive manifest", scope=scope)
            looked_up = _structured(
                await client.call_tool("memory_lookup", {"node_id": node.id})
            )
        return node.id, recall, looked_up

    node_id, recall, looked_up = asyncio.run(scenario())
    assert len(long_content) > SNIPPET_LIMIT

    entry = _by_id(recall)[node_id]
    assert entry["delivery"] == "snippet"
    snippet = entry["node"]["content"]
    assert snippet != long_content
    assert len(snippet) <= SNIPPET_LIMIT
    assert snippet.endswith("…")
    assert long_content.startswith(snippet[:-1])  # truncation, not paraphrase
    assert entry["content_ref"] == {
        "node_id": node_id,
        "fetch": f'memory_lookup(node_id="{node_id}")',
        "full_content_chars": len(long_content),
    }

    # The advertised re-fetch returns the stored bytes unchanged.
    assert looked_up["count"] == 1
    assert looked_up["missing"] == []
    assert looked_up["results"][0]["id"] == node_id
    assert looked_up["results"][0]["content"] == long_content


# --- Remember diet: bounded confirmation, durable and recallable write ------


def test_remember_response_bounded_while_write_durable_and_recallable(
    tmp_path: Path,
) -> None:
    scope = "project:dietwrite"
    body = "quenched manifold registry decision record\n" + "\n".join(
        f"registry stanza {index}: the manifold decision holds because of a "
        f"deliberately verbose justification paragraph repeated for bulk"
        for index in range(80)
    )

    async def scenario() -> tuple[Any, dict[str, Any], dict[str, Any]]:
        mcp, store = _server(tmp_path)
        async with Client(mcp) as client:
            remembered = _structured(
                await client.call_tool(
                    "memory_remember",
                    {"content": body, "context": {"scope": scope}},
                )
            )
            recalled = await _recall(client, "quenched manifold registry", scope=scope)
        return store, remembered, recalled

    store, remembered, recalled = asyncio.run(scenario())
    assert len(body) > 4096  # the stored payload alone dwarfs the response bound

    assert _wire_bytes(remembered) <= 4096
    assert set(remembered) == {"node", "implicit_feedback", "auto_consolidation", "auto_decay"}
    confirmation = remembered["node"]
    assert confirmation["level"] == "trace"
    assert confirmation["scope"] == scope
    for verbose_key in ("content", "context", "stats", "provenance"):
        assert verbose_key not in confirmation

    stored = store.get_node(confirmation["id"])
    assert stored is not None
    assert stored.content == body  # durably stored, byte-equal

    entry = _by_id(recalled)[confirmation["id"]]  # and recallable over the wire
    assert entry["delivery"] == "snippet"  # first delivery of a long body
    assert entry["content_ref"]["full_content_chars"] == len(body)


# --- Feedback closure: stub deliveries never break pending-recall matching --

CLOSURE_QUERY = "boreal auroral quasar lattice drift"
CLOSURE_SEED = "boreal auroral quasar lattice drift beacon calibration ledger"
CLOSURE_REMEMBER = "carbide flywheel gearset manifold spline"


def test_closure_links_recall_to_remember_despite_stub_delivery(
    tmp_path: Path,
) -> None:
    # Zero token overlap between remember content and recall query: the
    # text-similarity fallback cannot close anything, so transport identity
    # is the only matching rule in play.
    assert not set(tokenize(CLOSURE_REMEMBER)) & set(tokenize(CLOSURE_QUERY))
    scope = "project:dietclosure"

    async def scenario() -> tuple[Any, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
        mcp, store = _server(tmp_path)
        seeded = store.append_trace(CLOSURE_SEED, {"scope": scope})
        async with Client(mcp) as client:
            first = await _recall(client, CLOSURE_QUERY, scope=scope)
            second = await _recall(client, CLOSURE_QUERY, scope=scope)
            assert all(not event.feedback_applied for event in store.list_recall_events())
            remembered = _structured(
                await client.call_tool(
                    "memory_remember",
                    {"content": CLOSURE_REMEMBER, "context": {"scope": scope}},
                )
            )
        return store, seeded.id, first, second, remembered

    store, seeded_id, first, second, remembered = asyncio.run(scenario())

    # Precondition of the invariant: the second response really was a stub.
    assert _by_id(first)[seeded_id]["delivery"] == "full"
    assert _by_id(second)[seeded_id]["delivery"] == "session_duplicate"

    trace_id = remembered["node"]["id"]
    events = {event.id: event for event in store.list_recall_events()}
    assert set(events) == {first["recall_event_id"], second["recall_event_id"]}
    for event in events.values():
        assert event.feedback_applied  # 0 -> 1 on both events
        assert event.feedback_trace_id == trace_id
        assert event.result_ids == [seeded_id]  # the recorded event stays unshaped

    stored = store.get_node(trace_id)
    prior = stored.provenance["prior_recalls"]
    assert {entry["id"] for entry in prior} == set(events)
    assert remembered["node"]["prior_recall_count"] == 2
    assert set(remembered["implicit_feedback"]["recall_event_ids"]) == set(events)
    assert remembered["implicit_feedback"]["feedback_applied"] is True


# --- Digest consolidation: concepts stop being verbatim trace copies --------


def test_consolidate_over_wire_returns_digest_distinct_from_every_source(
    tmp_path: Path,
) -> None:
    scope = "project:dietdigest"

    async def scenario() -> tuple[list[Any], dict[str, Any]]:
        mcp, store = _server(tmp_path)
        sources = [
            store.append_trace(
                f"deploy rollback requires migration guard before release {index}",
                {
                    "scope": scope,
                    "agent": "agent-a" if index % 2 == 0 else "agent-b",
                },
                feedback={"confidence": 0.4, "usefulness_score": 0.2},
            )
            for index in range(100)
        ]
        async with Client(mcp) as client:
            response = _structured(
                await client.call_tool(
                    "memory_consolidate", {"scope": scope, "force": True}
                )
            )
        return sources, response

    sources, response = asyncio.run(scenario())
    source_contents = {source.content for source in sources}
    assert len(source_contents) == 100  # a genuinely multi-distinct cluster

    assert len(response["concepts_created"]) == 1
    concept = response["concepts_created"][0]
    assert concept["level"] == "concept"
    assert concept["scope"] == scope
    # memory_consolidate's direct report keeps full node dicts...
    assert set(concept["provenance"]["source_traces"]) == {s.id for s in sources}
    # ...and the concept content is a digest, byte-distinct from every source.
    assert concept["content"]
    assert concept["content"] not in source_contents
