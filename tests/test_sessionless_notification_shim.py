"""The session-less notification shim answers 202 instead of 400.

Regression for the Antigravity (``agy``) interop bug: agy emits
``notifications/roots/list_changed`` before it has echoed the ``Mcp-Session-Id``
that ``initialize`` returned. The streamable-http session manager rejects a
session-less message with 400, which agy treats as fatal and tears the whole
connection down (then flails into OAuth discovery probes). The shim swallows such
*notifications* with a 202 — lossless, since LM ignores client roots — while
leaving requests (which carry an ``id``) to fail normally so genuine
missing-session errors still surface. The SSE response stream is untouched.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

# Match scripts/test.sh so a bare `pytest .` merge-gate run stays fast.
os.environ.setdefault("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")

from starlette.middleware import Middleware
from starlette.testclient import TestClient

from living_memory.server import (
    _SessionlessNotificationShim,
    _is_jsonrpc_notification,
    create_mcp_server,
)

MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": "2025-06-18",
}


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    mcp = create_mcp_server(
        db_path=str(tmp_path / "shim.sqlite3"),
        default_scope="project:shim-test",
    )
    app = mcp.http_app(
        middleware=[Middleware(_SessionlessNotificationShim, mount_path="/mcp")]
    )
    with TestClient(app) as test_client:
        yield test_client


def test_sessionless_roots_notification_is_accepted(client: TestClient) -> None:
    """Exactly agy's failing message: a notification with no Mcp-Session-Id."""
    resp = client.post(
        "/mcp",
        headers=MCP_HEADERS,
        json={"jsonrpc": "2.0", "method": "notifications/roots/list_changed"},
    )
    assert resp.status_code == 202


def test_sessionless_request_is_not_swallowed(client: TestClient) -> None:
    """A request (carries id) must reach the manager and fail as before (400)."""
    resp = client.post(
        "/mcp",
        headers=MCP_HEADERS,
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    assert resp.status_code != 202
    assert resp.status_code == 400


def test_sessionless_initialize_round_trip_returns_result(client: TestClient) -> None:
    """A session-less ``initialize`` must stream its full result through the shim.

    ``initialize`` is itself a session-less POST (no Mcp-Session-Id yet) carrying an
    ``id``, so it travels the shim's body-replay path and produces a *streaming*
    (SSE) response. A regression where the replay returned a synthetic
    ``http.disconnect`` after the body aborted that stream mid-flight: the client
    still saw the session-id header but the JSON-RPC ``result`` never arrived
    ("initialize: request terminated without response"). Assert the result body —
    not just the header — comes back.
    """
    resp = client.post(
        "/mcp",
        headers=MCP_HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "shim-test", "version": "0"},
            },
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("mcp-session-id")
    assert '"result"' in resp.text
    assert '"protocolVersion"' in resp.text


def test_is_jsonrpc_notification_classification() -> None:
    assert _is_jsonrpc_notification(b'{"jsonrpc":"2.0","method":"notifications/x"}')
    assert not _is_jsonrpc_notification(b'{"jsonrpc":"2.0","id":1,"method":"tools/list"}')
    assert not _is_jsonrpc_notification(b"not json")
    assert not _is_jsonrpc_notification(b"{}")
    # Batch: all-notifications -> True; any request member -> False.
    assert _is_jsonrpc_notification(b'[{"method":"a"},{"method":"b"}]')
    assert not _is_jsonrpc_notification(b'[{"method":"a"},{"id":1,"method":"b"}]')
    assert not _is_jsonrpc_notification(b"[]")
