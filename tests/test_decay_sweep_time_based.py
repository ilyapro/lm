"""Tests for time-based, scope-agnostic decay sweep."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from living_memory.config import MemoryConfig
from living_memory.decay import apply_decay
from living_memory.server import (
    _DECAY_SWEEP_KV_KEY,
    _decay_sweep_if_due,
    _decay_sweep_interval_seconds,
    _maybe_decay_sweep,
    create_mcp_server,
)
from living_memory.storage import MemoryStore


class RecordingMCP:
    """Minimal FastMCP stand-in for tool/resource registration tests."""

    def __init__(self, name: str, instructions: str | None = None, **_: Any) -> None:
        self.name = name
        self.instructions = instructions
        self.tools: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def tool(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.tools[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        if func is None:
            return decorate
        return decorate(func)

    def resource(self, uri: str, **_kwargs: Any) -> Any:
        def decorate(func: Any) -> Any:
            self.resources[uri] = func
            return func

        return decorate

    def prompt(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.prompts[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        if func is None:
            return decorate
        return decorate(func)


# ---------- P3: kv table persistence ----------


def test_kv_get_returns_none_for_missing_key(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        assert store.get_kv("never-set") is None


def test_kv_set_then_get_roundtrips(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.set_kv("hello", "world")
        assert store.get_kv("hello") == "world"
        store.set_kv("hello", "again")
        assert store.get_kv("hello") == "again"


def test_kv_persists_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "memory.sqlite3"
    with MemoryStore(db) as store:
        store.set_kv("persistent", "value-1")
    with MemoryStore(db) as store:
        assert store.get_kv("persistent") == "value-1"


# ---------- P1: apply_decay(scope=None) sweeps all scopes ----------


def test_apply_decay_scope_none_sweeps_every_scope(tmp_path: Path) -> None:
    config = MemoryConfig(db_path=tmp_path / "memory.sqlite3", trace_ttl_days=7)
    with MemoryStore(config) as store:
        expired_ids = {}
        for scope in ("project:alpha", "project:beta", "global"):
            node = store.append_trace(
                f"ancient fact in {scope}",
                {
                    "scope": scope,
                    "agent": "agent-a",
                    "timestamp": "2020-01-01T00:00:00Z",
                },
            )
            expired_ids[scope] = node.id

        fresh = store.append_trace(
            "fresh fact",
            {"scope": "project:alpha", "agent": "agent-a"},
        )

        result = apply_decay(store, scope=None, now=datetime(2026, 5, 18, tzinfo=UTC))

        decayed_ids = {node.id for node in result.expired}
        assert set(expired_ids.values()) <= decayed_ids
        assert fresh.id not in decayed_ids
        for node_id in expired_ids.values():
            assert store.get_node(node_id).decayed is True
        assert store.get_node(fresh.id).decayed is False


# ---------- P2/P5: _decay_sweep_if_due time gate + env var ----------


def test_decay_sweep_interval_seconds_defaults_and_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LM_DECAY_SWEEP_INTERVAL_SEC", raising=False)
    assert _decay_sweep_interval_seconds() == 3600

    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "120")
    assert _decay_sweep_interval_seconds() == 120

    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
    assert _decay_sweep_interval_seconds() == 0

    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "garbage")
    assert _decay_sweep_interval_seconds() == 3600


def test_decay_sweep_if_due_first_call_sweeps_and_persists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "3600")
    config = MemoryConfig(db_path=tmp_path / "memory.sqlite3", trace_ttl_days=7)
    with MemoryStore(config) as store:
        old = store.append_trace(
            "ancient",
            {"scope": "global", "agent": "a", "timestamp": "2020-01-01T00:00:00Z"},
        )

        result = _decay_sweep_if_due(store)

        assert result is not None
        assert result["swept"] is True
        assert result["decayed_count"] >= 1
        assert result["expired_count"] >= 1
        assert "last_decay_sweep_at" in result
        assert store.get_kv(_DECAY_SWEEP_KV_KEY) is not None
        assert store.get_node(old.id).decayed is True


def test_decay_sweep_if_due_within_interval_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "3600")
    config = MemoryConfig(db_path=tmp_path / "memory.sqlite3", trace_ttl_days=7)
    with MemoryStore(config) as store:
        store.append_trace(
            "ancient",
            {"scope": "global", "agent": "a", "timestamp": "2020-01-01T00:00:00Z"},
        )

        first = _decay_sweep_if_due(store)
        assert first is not None
        second = _decay_sweep_if_due(store)
        assert second is None


def test_decay_sweep_if_due_runs_again_after_interval_elapses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "3600")
    config = MemoryConfig(db_path=tmp_path / "memory.sqlite3", trace_ttl_days=7)
    with MemoryStore(config) as store:
        store.append_trace(
            "ancient",
            {"scope": "global", "agent": "a", "timestamp": "2020-01-01T00:00:00Z"},
        )

        anchor = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
        first = _decay_sweep_if_due(store, now=anchor)
        assert first is not None

        # Within interval — skipped.
        skipped = _decay_sweep_if_due(store, now=anchor + timedelta(seconds=10))
        assert skipped is None

        # Past interval — runs again.
        second = _decay_sweep_if_due(store, now=anchor + timedelta(hours=2))
        assert second is not None


def test_decay_sweep_disabled_when_interval_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
    config = MemoryConfig(db_path=tmp_path / "memory.sqlite3", trace_ttl_days=7)
    with MemoryStore(config) as store:
        store.append_trace(
            "ancient",
            {"scope": "global", "agent": "a", "timestamp": "2020-01-01T00:00:00Z"},
        )

        assert _decay_sweep_if_due(store) is None
        # force=True bypasses the env gate so admin path stays usable.
        forced = _decay_sweep_if_due(store, force=True)
        assert forced is not None
        assert forced["swept"] is True


def test_maybe_decay_sweep_swallows_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "3600")
    config = MemoryConfig(db_path=tmp_path / "memory.sqlite3", trace_ttl_days=7)
    with MemoryStore(config) as store:
        # Make set_kv raise to simulate a storage failure mid-sweep.
        def boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("synthetic kv failure")

        monkeypatch.setattr(store, "set_kv", boom)
        # _maybe_decay_sweep must return None and never propagate.
        assert _maybe_decay_sweep(store) is None


# ---------- P4: hooks in memory_recall / memory_remember / memory_consolidate ----------


def _seed_expired_trace(db: Path) -> str:
    config = MemoryConfig(db_path=db, trace_ttl_days=7)
    with MemoryStore(config) as store:
        node = store.append_trace(
            "ancient hookable fact",
            {"scope": "global", "agent": "a", "timestamp": "2020-01-01T00:00:00Z"},
        )
        return node.id


def test_memory_recall_tool_triggers_decay_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "3600")
    db = tmp_path / "memory.sqlite3"
    expired_id = _seed_expired_trace(db)

    config = MemoryConfig(db_path=db, trace_ttl_days=7)
    mcp = create_mcp_server(config=config, mcp_factory=RecordingMCP)
    response = mcp.tools["memory_recall"]("anything", max_results=5)

    assert response.get("auto_decay") is not None
    assert response["auto_decay"]["swept"] is True
    assert mcp.memory_store.get_node(expired_id).decayed is True


def test_memory_remember_tool_triggers_decay_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "3600")
    db = tmp_path / "memory.sqlite3"
    expired_id = _seed_expired_trace(db)

    config = MemoryConfig(db_path=db, trace_ttl_days=7)
    mcp = create_mcp_server(config=config, mcp_factory=RecordingMCP)
    response = mcp.tools["memory_remember"](
        "new fact",
        {"scope": "global", "agent": "agent-a"},
    )

    assert response.get("auto_decay") is not None
    assert response["auto_decay"]["swept"] is True
    assert mcp.memory_store.get_node(expired_id).decayed is True


def test_memory_consolidate_tool_triggers_decay_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "3600")
    db = tmp_path / "memory.sqlite3"
    expired_id = _seed_expired_trace(db)

    config = MemoryConfig(db_path=db, trace_ttl_days=7)
    mcp = create_mcp_server(config=config, mcp_factory=RecordingMCP)
    response = mcp.tools["memory_consolidate"](scope="global")

    assert response.get("auto_decay") is not None
    assert response["auto_decay"]["swept"] is True
    assert mcp.memory_store.get_node(expired_id).decayed is True


def test_tool_hook_skips_after_first_sweep_within_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated tool calls within the interval must not re-sweep."""

    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "3600")
    db = tmp_path / "memory.sqlite3"
    config = MemoryConfig(db_path=db, trace_ttl_days=7)
    mcp = create_mcp_server(config=config, mcp_factory=RecordingMCP)

    first = mcp.tools["memory_recall"]("query one")
    second = mcp.tools["memory_recall"]("query two")

    assert first["auto_decay"] is not None
    assert second["auto_decay"] is None


# ---------- P6: POST /admin/decay-sweep ----------


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DEPS_DIR = ROOT_DIR / ".cache" / "python-deps"

ADMIN_TOKEN = "test-decay-sweep-token-abc"
READY_TIMEOUT_SECONDS = 20.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_env() -> dict[str, str]:
    env = os.environ.copy()
    env["LM_AUTH_TOKEN"] = ADMIN_TOKEN
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
        "project:decay-suite",
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
                response = httpx.post(
                    f"http://127.0.0.1:{port}/admin/decay-sweep", timeout=1.0
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
def decay_server(tmp_path: Path):
    pytest.importorskip("fastmcp")

    db_path = tmp_path / "decay.sqlite3"

    # Pre-seed an expired trace so the forced sweep has something to decay.
    config = MemoryConfig(db_path=db_path, trace_ttl_days=7)
    with MemoryStore(config) as store:
        store.append_trace(
            "ancient seed for admin sweep",
            {
                "scope": "global",
                "agent": "agent-a",
                "timestamp": "2020-01-01T00:00:00Z",
            },
        )

    port = _free_port()
    proc = _start_server(db_path, port)
    try:
        _wait_for_ready(port, deadline=time.monotonic() + READY_TIMEOUT_SECONDS)
        yield {"proc": proc, "port": port, "db_path": db_path}
    finally:
        _terminate(proc)


def _post_decay_sweep(
    port: int, *, token: str | None, scheme: str = "Bearer"
) -> httpx.Response:
    headers = {"Authorization": f"{scheme} {token}"} if token is not None else {}
    return httpx.post(
        f"http://127.0.0.1:{port}/admin/decay-sweep",
        headers=headers,
        timeout=10.0,
    )


def test_admin_decay_sweep_requires_bearer_token(
    decay_server: dict[str, Any]
) -> None:
    no_auth = _post_decay_sweep(decay_server["port"], token=None)
    assert no_auth.status_code == 401
    body = no_auth.json()
    assert body["ok"] is False

    wrong = _post_decay_sweep(decay_server["port"], token="nope")
    assert wrong.status_code == 401


def test_admin_decay_sweep_returns_200_with_count_and_duration(
    decay_server: dict[str, Any]
) -> None:
    response = _post_decay_sweep(decay_server["port"], token=ADMIN_TOKEN)
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["swept"] is True
    assert isinstance(body.get("decayed_count"), int)
    assert body["decayed_count"] >= 1
    assert isinstance(body.get("duration_ms"), int)
    assert body["duration_ms"] >= 0
    assert "last_decay_sweep_at" in body
