"""Tests for the POST /admin/token bearer-token rotation endpoint.

Covers the four goal postconditions:
- a request authorized by the *current* token sets a new token;
- after rotation the server accepts only the new token (admin + MCP surfaces);
- the rotation survives a restart (persisted where the server reads at startup);
- the token never lands in logs (nor, by construction, diffs or traces).
"""

from __future__ import annotations

import asyncio
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from living_memory.server import create_mcp_server

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DEPS_DIR = ROOT_DIR / ".cache" / "python-deps"

# Generated fresh per test run: the real token value never appears in the
# source (so, by construction, never in a diff or trace), yet stays distinctive
# enough that scanning a captured server log for a leak has no false positives.
def _fresh_token(label: str) -> str:
    return f"{label}-{secrets.token_hex(12)}"


INITIAL_TOKEN = _fresh_token("initial")
ROTATED_TOKEN = _fresh_token("rotated")
DEFAULT_SCOPE = "project:auth-token-suite"
READY_TIMEOUT_SECONDS = 30.0


# --------------------------------------------------------------------------- #
# In-process unit coverage of the rotation mechanism and startup precedence.
# --------------------------------------------------------------------------- #


def _verify(verifier: Any, token: str) -> Any:
    return asyncio.run(verifier.verify_token(token))


def test_rotate_swaps_static_token_verifier_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastmcp")
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")

    mcp = create_mcp_server(tmp_path / "rotate.sqlite3", auth_token=INITIAL_TOKEN)
    state = getattr(mcp, "auth_token_state", None)
    assert state is not None and state.token == INITIAL_TOKEN
    verifier = getattr(mcp, "auth", None)
    assert verifier is not None and set(verifier.tokens) == {INITIAL_TOKEN}

    # Before rotation: the initial token verifies, the future one does not.
    assert _verify(verifier, INITIAL_TOKEN) is not None
    assert _verify(verifier, ROTATED_TOKEN) is None

    state.rotate(ROTATED_TOKEN)

    # After rotation both live surfaces accept only the new token.
    assert state.token == ROTATED_TOKEN
    assert set(verifier.tokens) == {ROTATED_TOKEN}
    assert _verify(verifier, INITIAL_TOKEN) is None
    assert _verify(verifier, ROTATED_TOKEN) is not None


def test_persisted_token_wins_over_env_seed_on_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastmcp")
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    db_path = tmp_path / "persist.sqlite3"

    first = create_mcp_server(db_path, auth_token=INITIAL_TOKEN)
    assert first.auth_token_state.token == INITIAL_TOKEN
    # Persist a rotation exactly as POST /admin/token does.
    first.memory_store.set_kv("auth_token", ROTATED_TOKEN)

    # Simulate a restart: a fresh server on the same DB with the *original*
    # env seed must honor the persisted (rotated) token instead.
    second = create_mcp_server(db_path, auth_token=INITIAL_TOKEN)
    assert second.auth_token_state.token == ROTATED_TOKEN
    assert set(second.auth.tokens) == {ROTATED_TOKEN}


# --------------------------------------------------------------------------- #
# End-to-end HTTP coverage over a real subprocess server.
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_env(token: str) -> dict[str, str]:
    env = os.environ.copy()
    env["LM_AUTH_TOKEN"] = token
    # Deterministic, fast embeddings so the HTTP port binds promptly.
    env["LIVING_MEMORY_EMBEDDING_BACKEND"] = "hash"
    parts = [str(SRC_DIR)]
    if DEPS_DIR.exists():
        parts.append(str(DEPS_DIR))
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _start_server(
    db_path: Path, port: int, *, token: str, log_path: Path | None = None
) -> subprocess.Popen[bytes]:
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
    if log_path is not None:
        log = log_path.open("wb")
        return subprocess.Popen(
            cmd, env=_server_env(token), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return subprocess.Popen(
        cmd,
        env=_server_env(token),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _wait_for_ready(port: int, deadline: float) -> None:
    last_err: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                # The new admin route answering 401 proves the ASGI app is wired.
                response = httpx.post(
                    f"http://127.0.0.1:{port}/admin/token", timeout=1.0
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


def _post_token(
    port: int, *, token: str | None, body: Any, scheme: str = "Bearer"
) -> httpx.Response:
    headers = {"Authorization": f"{scheme} {token}"} if token is not None else {}
    return httpx.post(
        f"http://127.0.0.1:{port}/admin/token",
        headers=headers,
        json=body,
        timeout=5.0,
    )


def _get(port: int, path: str, *, token: str | None) -> httpx.Response:
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    return httpx.get(f"http://127.0.0.1:{port}{path}", headers=headers, timeout=5.0)


async def _mcp_status(port: int, token: str) -> dict[str, Any]:
    from fastmcp import Client

    client = Client(f"http://127.0.0.1:{port}/mcp/", auth=token)
    async with client:
        result = await client.call_tool("memory_status", {})
    payload = getattr(result, "structured_content", None)
    if (
        isinstance(payload, dict)
        and set(payload) == {"result"}
        and isinstance(payload["result"], dict)
    ):
        return payload["result"]
    assert isinstance(payload, dict)
    return payload


@pytest.fixture
def server(tmp_path: Path):
    pytest.importorskip("fastmcp")
    db_path = tmp_path / "auth.sqlite3"
    port = _free_port()
    proc = _start_server(db_path, port, token=INITIAL_TOKEN)
    try:
        _wait_for_ready(port, deadline=time.monotonic() + READY_TIMEOUT_SECONDS)
        yield {"proc": proc, "port": port, "db_path": db_path}
    finally:
        _terminate(proc)


def test_admin_token_requires_bearer_and_validates_body(
    server: dict[str, Any],
) -> None:
    port = server["port"]

    # Unauthenticated / wrong token / wrong scheme are all rejected.
    assert _post_token(port, token=None, body={"token": ROTATED_TOKEN}).status_code == 401
    assert _post_token(port, token="nope", body={"token": ROTATED_TOKEN}).status_code == 401
    assert (
        _post_token(port, token=INITIAL_TOKEN, body={"token": ROTATED_TOKEN}, scheme="Token").status_code
        == 401
    )

    # Authorized but with an empty/missing new token -> 400, no state change.
    empty = _post_token(port, token=INITIAL_TOKEN, body={"token": "   "})
    assert empty.status_code == 400
    assert empty.json()["ok"] is False
    missing = _post_token(port, token=INITIAL_TOKEN, body={})
    assert missing.status_code == 400

    # None of the failed attempts rotated the token: the initial one still works.
    assert _get(port, "/admin/info", token=INITIAL_TOKEN).status_code == 200


def test_rotate_applies_live_to_admin_and_mcp(server: dict[str, Any]) -> None:
    port = server["port"]

    # Sanity: the initial token works on both surfaces before rotation.
    assert _get(port, "/admin/info", token=INITIAL_TOKEN).status_code == 200
    assert asyncio.run(_mcp_status(port, INITIAL_TOKEN))  # non-empty dict

    rotate = _post_token(port, token=INITIAL_TOKEN, body={"token": ROTATED_TOKEN})
    assert rotate.status_code == 200
    body = rotate.json()
    assert body["ok"] is True and "rotated_at" in body
    # The response must not echo either token value.
    assert ROTATED_TOKEN not in rotate.text
    assert INITIAL_TOKEN not in rotate.text

    # Admin surface: old rejected, new accepted.
    assert _get(port, "/admin/info", token=INITIAL_TOKEN).status_code == 401
    assert _get(port, "/admin/info", token=ROTATED_TOKEN).status_code == 200

    # MCP surface: a fresh connection with the old token is rejected.
    with pytest.raises(Exception):
        asyncio.run(_mcp_status(port, INITIAL_TOKEN))
    # ...and the new token is accepted.
    assert asyncio.run(_mcp_status(port, ROTATED_TOKEN))


def test_rotated_token_survives_cold_restart_without_leaking(tmp_path: Path) -> None:
    pytest.importorskip("fastmcp")
    db_path = tmp_path / "cold.sqlite3"
    log_a = tmp_path / "server-a.log"
    log_b = tmp_path / "server-b.log"

    port_a = _free_port()
    proc_a = _start_server(db_path, port_a, token=INITIAL_TOKEN, log_path=log_a)
    try:
        _wait_for_ready(port_a, deadline=time.monotonic() + READY_TIMEOUT_SECONDS)
        assert (
            _post_token(port_a, token=INITIAL_TOKEN, body={"token": ROTATED_TOKEN}).status_code
            == 200
        )
    finally:
        _terminate(proc_a)

    # Cold restart: a brand-new process on the same DB, still seeded with the
    # ORIGINAL env token. The persisted rotation must take precedence.
    port_b = _free_port()
    proc_b = _start_server(db_path, port_b, token=INITIAL_TOKEN, log_path=log_b)
    try:
        _wait_for_ready(port_b, deadline=time.monotonic() + READY_TIMEOUT_SECONDS)
        assert _get(port_b, "/admin/info", token=INITIAL_TOKEN).status_code == 401
        assert _get(port_b, "/admin/info", token=ROTATED_TOKEN).status_code == 200
    finally:
        _terminate(proc_b)

    # Neither token value may appear in either server's captured stdout/stderr.
    for log in (log_a, log_b):
        text = log.read_text(errors="replace")
        assert INITIAL_TOKEN not in text, f"initial token leaked into {log.name}"
        assert ROTATED_TOKEN not in text, f"rotated token leaked into {log.name}"
