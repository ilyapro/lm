"""End-to-end verification of transport-session feedback closure.

The chain under test spans real MCP transports: an identity-less client
(no ``agent``/``task``/``session_id`` anywhere) talks to a live server, the
server derives ``transport_session_id`` from the transport session and stamps
it into recall events and remember traces, and the pending-recall matcher
closes each recall event against the same connection's next remember — and
only that connection's.

Falsifiability controls:

* Every remember content shares **zero** tokens with every recall query
  (asserted with the production tokenizer), so the legacy text-similarity
  fallback cannot close anything — transport identity is the only available
  matching rule.
* Two clients are connected concurrently and the second client's recall event
  is *newer* than the first client's remember. Without the
  both-present-differing reject, the first remember would consume the newest
  pending event (they are scanned ``created_at DESC``); the mid-flight
  closed-count and the final ``feedback_trace_id`` linkage would both fail.
* Transport ids are asserted non-null and pairwise distinct straight from the
  ``recall_events.transport_session_id`` column after server shutdown.

The same scenario is exercised over TLS streamable-HTTP (subprocess server,
verified https), real stdio (subprocess server), and the in-memory transport,
and ``memory_health.feedback_closure`` is asserted over the wire so the
first-class metric reflects the closures it claims to measure.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastmcp")

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from living_memory.embeddings import tokenize
from living_memory.server import create_mcp_server
from living_memory.storage import _RECALL_TEXT_SIMILARITY_THRESHOLD

from test_tls_transport import (
    DEFAULT_SCOPE,
    TOKEN,
    _server_env,
    _serve,
    _ssl_context,
    _structured,
    _write_self_signed_cert,
)

# Pairwise-disjoint vocabularies: no remember content overlaps any recall
# query, so text similarity is exactly 0.0 for every (event, trace) pair and
# the weak/strong text fallbacks are unreachable.
QUERY_A = "boreal auroral quasar lattice drift"
QUERY_B = "zephyr mangrove citadel prism ember"
CONTENT_A = "carbide flywheel torque manifold spline"
CONTENT_B = "willow granite meadow lantern moss"
CONTENT_C = "quartz harbor sable ridge acorn"


def test_control_texts_share_no_tokens_with_queries() -> None:
    """Pin the dissimilar-text control the closure tests rely on."""

    assert _RECALL_TEXT_SIMILARITY_THRESHOLD > 0.0
    for content in (CONTENT_A, CONTENT_B, CONTENT_C):
        for query in (QUERY_A, QUERY_B):
            assert not set(tokenize(content)) & set(tokenize(query))


async def _closure_over_wire(client: Client) -> dict[str, Any]:
    health = _structured(await client.call_tool("memory_health", {}))
    block = health["feedback_closure"]
    assert isinstance(block, dict)
    return block


async def _two_client_scenario(client_a: Client, client_b: Client) -> dict[str, Any]:
    """Interleave two identity-less sessions: recall A, recall B, remember A, remember B."""

    async with client_a:
        async with client_b:
            await client_a.call_tool("memory_recall", {"query": QUERY_A})
            await client_b.call_tool("memory_recall", {"query": QUERY_B})
            before = await _closure_over_wire(client_a)

            remembered_a = _structured(
                await client_a.call_tool("memory_remember", {"content": CONTENT_A})
            )
            middle = await _closure_over_wire(client_b)

            remembered_b = _structured(
                await client_b.call_tool("memory_remember", {"content": CONTENT_B})
            )
            after = await _closure_over_wire(client_a)

    return {
        "trace_a": remembered_a["node"],
        "trace_b": remembered_b["node"],
        "before": before,
        "middle": middle,
        "after": after,
    }


def _assert_closure_progression(outcome: dict[str, Any]) -> None:
    """The wire-visible metric must count exactly our transport-only events."""

    before, middle, after = outcome["before"], outcome["middle"], outcome["after"]

    for block in (before, middle, after):
        coverage = block["identity_coverage"]
        assert coverage["explicit"]["events"] == 0
        assert coverage["none"]["events"] == 0
        assert coverage["transport_only"]["events"] == 2

    assert before["identity_coverage"]["transport_only"]["closed"] == 0
    # Client A's remember ran while client B's newer recall event was pending;
    # exactly one closure proves B's event was rejected by transport mismatch.
    assert middle["identity_coverage"]["transport_only"]["closed"] == 1
    assert after["identity_coverage"]["transport_only"]["closed"] == 2
    assert after["identity_coverage"]["transport_only"]["closure_ratio"] == 1.0
    assert after["closure_ratio"] == 1.0
    assert after["recall_events_in_window"] == 2
    assert after["feedback_applied_in_window"] == 2


def _fetch_events_by_query(db_path: Path) -> dict[str, sqlite3.Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT query, agent, task, session_id, transport_session_id,
                   feedback_applied, feedback_trace_id
            FROM recall_events
            """
        ).fetchall()
    finally:
        conn.close()
    return {str(row["query"]): row for row in rows}


def _fetch_node_context(db_path: Path, node_id: str) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT context FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    return json.loads(row[0])


def _assert_isolated_closure(db_path: Path, outcome: dict[str, Any]) -> None:
    """Ground-truth column assertions: distinct ids, own-trace-only closure."""

    events = _fetch_events_by_query(db_path)
    event_a, event_b = events[QUERY_A], events[QUERY_B]

    # Identity-less on the wire: the only identity is the transport stamp.
    for event in (event_a, event_b):
        assert event["agent"] is None
        assert event["task"] is None
        assert event["session_id"] is None
        assert isinstance(event["transport_session_id"], str)
        assert event["transport_session_id"]
    assert event_a["transport_session_id"] != event_b["transport_session_id"]

    # Each event was closed by its own connection's trace — never the peer's.
    assert event_a["feedback_applied"] == 1
    assert event_b["feedback_applied"] == 1
    assert event_a["feedback_trace_id"] == outcome["trace_a"]["id"]
    assert event_b["feedback_trace_id"] == outcome["trace_b"]["id"]

    # The stamp travelled to the remember trace over the wire and matches the
    # column value extracted on the recall side of the same connection.
    trace_stamp_a = _fetch_node_context(db_path, outcome["trace_a"]["id"])[
        "transport_session_id"
    ]
    trace_stamp_b = _fetch_node_context(db_path, outcome["trace_b"]["id"])[
        "transport_session_id"
    ]
    assert trace_stamp_a == event_a["transport_session_id"]
    assert trace_stamp_b == event_b["transport_session_id"]


# --- TLS streamable-HTTP: two identity-less clients, one live server --------


def test_tls_two_identityless_clients_close_only_their_own_events(
    tmp_path: Path,
) -> None:
    cert, key = _write_self_signed_cert(tmp_path)
    with _serve(tmp_path, tls=True, cert=cert, key=key) as srv:
        context = _ssl_context(srv["verify"])
        assert context is not None  # every request below completes a real TLS handshake

        def _client() -> Client:
            return Client(f"{srv['base_url']}/mcp/", auth=TOKEN, verify=context)

        outcome = asyncio.run(_two_client_scenario(_client(), _client()))
        db_path = srv["db_path"]

    _assert_closure_progression(outcome)
    _assert_isolated_closure(db_path, outcome)


# --- In-memory transport: the same interleaving behaves identically ---------


def test_in_memory_two_clients_close_only_their_own_events(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    db_path = Path(mcp.memory_store.db_path)

    outcome = asyncio.run(_two_client_scenario(Client(mcp), Client(mcp)))
    mcp.memory_store.close()

    _assert_closure_progression(outcome)
    _assert_isolated_closure(db_path, outcome)


# --- Real stdio transport: per-connection identity closes the loop too ------


def test_stdio_client_closes_own_event_via_transport_identity(tmp_path: Path) -> None:
    db_path = tmp_path / "stdio.sqlite3"
    env = _server_env()
    env.pop("LM_AUTH_TOKEN", None)  # bearer auth is an HTTP concern
    transport = StdioTransport(
        command=sys.executable,
        args=[
            "-m",
            "living_memory.server",
            "--db",
            str(db_path),
            "--transport",
            "stdio",
            "--default-scope",
            DEFAULT_SCOPE,
        ],
        env=env,
    )

    async def scenario() -> tuple[dict[str, Any], dict[str, Any]]:
        async with Client(transport) as client:
            await client.call_tool("memory_recall", {"query": QUERY_A})
            remembered = _structured(
                await client.call_tool("memory_remember", {"content": CONTENT_A})
            )
            closure = await _closure_over_wire(client)
        return remembered["node"], closure

    trace, closure = asyncio.run(scenario())

    assert closure["identity_coverage"]["transport_only"] == {
        "events": 1,
        "closed": 1,
        "closure_ratio": 1.0,
    }

    events = _fetch_events_by_query(db_path)
    event = events[QUERY_A]
    assert event["agent"] is None and event["session_id"] is None
    assert isinstance(event["transport_session_id"], str)
    assert event["transport_session_id"]
    assert event["feedback_applied"] == 1
    assert event["feedback_trace_id"] == trace["id"]
    trace_context = _fetch_node_context(db_path, trace["id"])
    assert trace_context["transport_session_id"] == event["transport_session_id"]


# --- Explicit identity still outranks an equal transport stamp --------------


def test_tls_explicit_session_precedence_survives_transport_match(
    tmp_path: Path,
) -> None:
    """Equal transport ids cannot resurrect an explicit session mismatch (e2e).

    Everything runs on ONE connection, so the derived transport id on the
    recall event and on both remember traces is identical — a transport-only
    matcher would close the event at the first remember. The explicit
    ``session_id`` precedence must decide instead: the mismatching remember
    leaves the event open, the matching one closes it. Scopes are pinned
    explicitly because a bare ``session_id`` would otherwise shift scope
    resolution into a ``session:`` scope and mask the precedence rules.
    """

    cert, key = _write_self_signed_cert(tmp_path)
    with _serve(tmp_path, tls=True, cert=cert, key=key) as srv:
        context = _ssl_context(srv["verify"])
        client = Client(f"{srv['base_url']}/mcp/", auth=TOKEN, verify=context)

        async def scenario() -> tuple[dict[str, Any], dict[str, Any]]:
            async with client:
                await client.call_tool(
                    "memory_recall",
                    {
                        "query": QUERY_B,
                        "scope": DEFAULT_SCOPE,
                        "ambient_context": {"session_id": "explicit-session-1"},
                    },
                )
                mismatched = _structured(
                    await client.call_tool(
                        "memory_remember",
                        {
                            "content": CONTENT_B,
                            "context": {
                                "scope": DEFAULT_SCOPE,
                                "session_id": "explicit-session-2",
                            },
                        },
                    )
                )
                matched = _structured(
                    await client.call_tool(
                        "memory_remember",
                        {
                            "content": CONTENT_C,
                            "context": {
                                "scope": DEFAULT_SCOPE,
                                "session_id": "explicit-session-1",
                            },
                        },
                    )
                )
            return mismatched["node"], matched["node"]

        mismatched_trace, matched_trace = asyncio.run(scenario())
        db_path = srv["db_path"]

    event = _fetch_events_by_query(db_path)[QUERY_B]
    assert event["session_id"] == "explicit-session-1"

    # The derived stamp coexists with explicit identity on both sides...
    assert isinstance(event["transport_session_id"], str)
    assert event["transport_session_id"]
    assert (
        _fetch_node_context(db_path, mismatched_trace["id"])["transport_session_id"]
        == event["transport_session_id"]
    )
    assert (
        _fetch_node_context(db_path, matched_trace["id"])["transport_session_id"]
        == event["transport_session_id"]
    )

    # ...but closure was decided by the explicit session, not the equal stamp.
    assert event["feedback_applied"] == 1
    assert event["feedback_trace_id"] == matched_trace["id"]
