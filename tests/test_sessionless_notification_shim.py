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

import asyncio
import os
from pathlib import Path
from typing import Any, Iterator

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


@pytest.mark.parametrize("chunk_size", [1, 7])
@pytest.mark.parametrize(
    "body,disconnected,accepted",
    [
        pytest.param(
            b' \n{"jsonrpc":"2.0","id":1,"method":"tools/list",'
            b'"params":{"value":"\xe2\x82\xac"}}\n',
            False, False, id="request",
        ),
        pytest.param(b'{"jsonrpc":', False, False, id="malformed"),
        pytest.param(b'{"method":"notifications/x"}', False, True, id="notification"),
        pytest.param(b'[{"method":"a"},{"method":"b"}]', False, True, id="notifications"),
        pytest.param(b'[{"method":"a"},{"id":1,"method":"b"}]', False, False, id="mixed-batch"),
        pytest.param(b'{"method":"notifications/x"}', True, False, id="disconnect-notification"),
        pytest.param(b'{"jsonrpc":', True, False, id="disconnect-partial"),
        pytest.param(b"", False, False, id="empty"),
    ],
)
def test_fragmented_body_delivery(
    body: bytes, disconnected: bool, accepted: bool, chunk_size: int
) -> None:
    chunks = [body[i:i + chunk_size] for i in range(0, len(body), chunk_size)] or [b""]
    disconnect = {"type": "http.disconnect"}
    messages = iter([
        *(
            {"type": "http.request", "body": chunk,
             "more_body": disconnected or index < len(chunks) - 1}
            for index, chunk in enumerate(chunks)
        ),
        disconnect,
    ])
    receive_calls = 0
    sent = []

    async def receive() -> dict:
        nonlocal receive_calls
        receive_calls += 1
        return next(messages)

    async def send(message: dict) -> None:
        sent.append(message)

    async def app(scope: Any, downstream_receive: Any, downstream_send: Any) -> None:
        assert not accepted
        assert downstream_send is send
        request = await downstream_receive()
        assert request == {"type": "http.request", "body": body, "more_body": False}
        assert type(request["body"]) is bytes
        followup = await downstream_receive()
        assert followup == disconnect
        if not disconnected:
            assert followup is disconnect
        await downstream_send({"type": "http.response.start", "status": 400, "headers": []})
        await downstream_send({"type": "http.response.body", "body": b"downstream error"})

    scope = {"type": "http", "method": "POST", "path": "/mcp/", "headers": []}
    asyncio.run(_SessionlessNotificationShim(app)(scope, receive, send))
    assert sent[0]["status"] == (202 if accepted else 400)
    assert sent[1]["body"] == (b"" if accepted else b"downstream error")
    assert receive_calls == len(chunks) + (not accepted)


@pytest.mark.parametrize("overrides", [
    {"headers": [(b"Mcp-Session-Id", b"established-session")]},
    {"method": "GET"},
    {"path": "/other"},
    {"type": "lifespan"},
])
def test_bypass_preserves_channels(overrides: dict) -> None:
    scope = {"type": "http", "method": "POST", "path": "/mcp", "headers": [], **overrides}
    calls = []

    async def receive() -> dict:
        pytest.fail("the shim must not read a bypassed request")

    async def send(message: dict) -> None:
        pytest.fail("the shim must not answer a bypassed request")

    async def app(actual_scope: Any, actual_receive: Any, actual_send: Any) -> None:
        assert actual_scope is scope
        assert actual_receive is receive
        assert actual_send is send
        calls.append(actual_scope)

    asyncio.run(_SessionlessNotificationShim(app)(scope, receive, send))
    assert calls == [scope]


def test_is_jsonrpc_notification_classification() -> None:
    assert _is_jsonrpc_notification(b'{"jsonrpc":"2.0","method":"notifications/x"}')
    assert not _is_jsonrpc_notification(b'{"jsonrpc":"2.0","id":1,"method":"tools/list"}')
    assert not _is_jsonrpc_notification(b"not json")
    assert not _is_jsonrpc_notification(b"{}")
    # Batch: all-notifications -> True; any request member -> False.
    assert _is_jsonrpc_notification(b'[{"method":"a"},{"method":"b"}]')
    assert not _is_jsonrpc_notification(b'[{"method":"a"},{"id":1,"method":"b"}]')
    assert not _is_jsonrpc_notification(b"[]")
