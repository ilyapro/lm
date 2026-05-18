"""Integration tests for the POST /admin/restart endpoint."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DEPS_DIR = ROOT_DIR / ".cache" / "python-deps"

TOKEN = "test-restart-token-xyz"
DEFAULT_SCOPE = "project:restart-suite"
READY_TIMEOUT_SECONDS = 20.0
RESTART_TIMEOUT_SECONDS = 20.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_env() -> dict[str, str]:
    env = os.environ.copy()
    env["LM_AUTH_TOKEN"] = TOKEN
    parts = [str(SRC_DIR)]
    if DEPS_DIR.exists():
        parts.append(str(DEPS_DIR))
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _start_server(db_path: Path, port: int) -> subprocess.Popen[bytes]:
    cmd = [
        sys.executable,
        "-m",
        "living_memory.server",
        "--db",
        str(db_path),
        "--transport",
        "http",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--default-scope",
        DEFAULT_SCOPE,
    ]
    return subprocess.Popen(
        cmd,
        env=_server_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _wait_for_ready(port: int, deadline: float) -> None:
    last_err: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                # Probe the endpoint to ensure the ASGI app is wired (not just port open).
                response = httpx.post(
                    f"http://127.0.0.1:{port}/admin/restart", timeout=1.0
                )
                if response.status_code == 401:
                    return
        except (OSError, httpx.HTTPError) as err:
            last_err = err
        time.sleep(0.1)
    raise RuntimeError(f"server on port {port} did not become ready: {last_err!r}")


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


@pytest.fixture
def server(tmp_path: Path):
    pytest.importorskip("fastmcp")
    db_path = tmp_path / "restart.sqlite3"
    port = _free_port()
    proc = _start_server(db_path, port)
    try:
        _wait_for_ready(port, deadline=time.monotonic() + READY_TIMEOUT_SECONDS)
        yield {"proc": proc, "port": port, "db_path": db_path}
    finally:
        _terminate(proc)


def _post_restart(port: int, *, token: str | None) -> httpx.Response:
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    return httpx.post(
        f"http://127.0.0.1:{port}/admin/restart",
        headers=headers,
        timeout=5.0,
    )


def test_admin_restart_requires_bearer_token(server: dict[str, Any]) -> None:
    no_auth = _post_restart(server["port"], token=None)
    assert no_auth.status_code == 401
    body = no_auth.json()
    assert body["ok"] is False

    wrong_auth = _post_restart(server["port"], token="not-the-real-token")
    assert wrong_auth.status_code == 401


def test_admin_restart_returns_202_and_replaces_process(server: dict[str, Any]) -> None:
    proc = server["proc"]
    port = server["port"]
    pid_observed = proc.pid

    response = _post_restart(port, token=TOKEN)
    assert response.status_code == 202
    body = response.json()
    assert body["ok"] is True
    assert body["pid_before"] == pid_observed  # exec preserves PID
    assert "restart_at" in body and body["restart_at"]

    # Wait until the original socket is closed (old process tearing down)
    # then the new process re-binds and answers again.
    _wait_for_restart(port, deadline=time.monotonic() + RESTART_TIMEOUT_SECONDS)

    # PID stays the same across os.execv; the subprocess handle is still valid.
    assert proc.poll() is None, "subprocess should still be alive after exec"

    # Server is responsive with the original auth gating intact.
    challenge = _post_restart(port, token=None)
    assert challenge.status_code == 401


def test_admin_restart_preserves_args_and_db(server: dict[str, Any]) -> None:
    port = server["port"]
    db_path: Path = server["db_path"]

    # Seed the DB through the live server so we can confirm content survives.
    asyncio.run(_remember_via_mcp(port, "pre-restart-marker"))

    response = _post_restart(port, token=TOKEN)
    assert response.status_code == 202

    _wait_for_restart(port, deadline=time.monotonic() + RESTART_TIMEOUT_SECONDS)

    # --db argument survived: the same file is still on disk.
    assert db_path.exists()

    # --default-scope argument survived: a trace remembered without an explicit
    # scope inherits the configured default.
    after = asyncio.run(_remember_via_mcp(port, "post-restart-marker"))
    assert after["node"]["scope"] == DEFAULT_SCOPE

    # The trace we wrote pre-restart is still there.
    results = asyncio.run(_recall_via_mcp(port, "pre-restart-marker"))
    assert any(
        "pre-restart-marker" in (item["node"]["content"] or "")
        for item in results
    )


def _wait_for_restart(port: int, deadline: float) -> None:
    last_status: int | None = None
    last_err: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.post(
                f"http://127.0.0.1:{port}/admin/restart", timeout=1.0
            )
            last_status = response.status_code
            if response.status_code == 401:
                return
        except (OSError, httpx.HTTPError) as err:
            last_err = err
        time.sleep(0.1)
    raise RuntimeError(
        f"server did not return to ready after restart: status={last_status} err={last_err!r}"
    )


async def _mcp_call(port: int, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    from fastmcp import Client

    client = Client(f"http://127.0.0.1:{port}/mcp/", auth=TOKEN)
    async with client:
        result = await client.call_tool(tool, arguments)
    payload = getattr(result, "structured_content", None)
    if isinstance(payload, dict) and set(payload) == {"result"} and isinstance(payload["result"], dict):
        return payload["result"]
    assert isinstance(payload, dict)
    return payload


async def _remember_via_mcp(port: int, content: str) -> dict[str, Any]:
    return await _mcp_call(
        port,
        "memory_remember",
        {"content": content, "context": {"agent": "restart-test"}},
    )


async def _recall_via_mcp(port: int, query: str) -> list[dict[str, Any]]:
    payload = await _mcp_call(
        port,
        "memory_recall",
        {"query": query, "max_results": 5},
    )
    return list(payload.get("results", []))
