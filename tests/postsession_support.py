"""Synthetic session builders and a disposable MCP recipient for extraction tests."""

from __future__ import annotations

import importlib.util
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

from living_memory.postsession.session import (
    DeliveredNode, Evidence, FileMutation, ProposedOp, RecallInteraction,
    SessionRecord, SourceSpan,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DEPS_DIR = ROOT_DIR / ".cache" / "python-deps"
SCRIPT = ROOT_DIR / "scripts" / "post_session_extract.py"
TOKEN = "runner-live-test-token"
READY_TIMEOUT_SECONDS = 60.0


def _load_cli() -> Any:
    spec = importlib.util.spec_from_file_location("post_session_extract", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_record(**overrides: Any) -> SessionRecord:
    defaults: dict[str, Any] = dict(
        session_key="claude:runner-test-session",
        source="claude",
        cli="claude",
        path="/tmp/runner-test.jsonl",
        cli_session_id="runner-test-session",
        cwd="/home/user/p/demo",
        transport_session_ids=("source-transport-1",),
        transcript_sha256="ab" * 32,
    )
    defaults.update(overrides)
    return SessionRecord(**defaults)


def make_remember(record: SessionRecord, content: str, scope: str = "project:demo") -> ProposedOp:
    span = SourceSpan(record.source, record.path, 3)
    return ProposedOp.for_session(
        "remember",
        {"content": content, "context": {"scope": scope}},
        record,
        span,
        [Evidence(quote=content, locator=span.locator())],
    ).validate()


def make_teach(record: SessionRecord, trace_id: str, correction: str) -> ProposedOp:
    span = SourceSpan(record.source, record.path, 5)
    return ProposedOp.for_session(
        "teach",
        {
            "trace_id": trace_id,
            "correction": correction,
            "confidence": 0.85,
            "context": {"corrected_node_id": trace_id},
        },
        record,
        span,
        [Evidence(quote=correction, locator=span.locator())],
    ).validate()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_env() -> dict[str, str]:
    env = os.environ.copy()
    env["LM_AUTH_TOKEN"] = TOKEN
    # The runner is the consumer the guarded memory_attest registration
    # exists for: mirror the documented production setting for an
    # extraction-enabled deployment (server.py _register_tools).
    env["LM_EXPOSE_ATTEST"] = "1"
    env.setdefault("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    parts = [str(SRC_DIR)]
    if DEPS_DIR.exists():
        parts.append(str(DEPS_DIR))
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


@pytest.fixture(scope="module")
def live_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    pytest.importorskip("fastmcp")
    import httpx

    tmp_path = tmp_path_factory.mktemp("runner-live")
    port = _free_port()
    db_path = tmp_path / "scratch.sqlite3"  # a scratch DB, never the live file
    proc = subprocess.Popen(
        [
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
            "global",
        ],
        env=_server_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    try:
        while True:
            if proc.poll() is not None:
                tail = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
                raise RuntimeError(f"throwaway server died (code {proc.returncode}):\n{tail}")
            try:
                if httpx.get(f"{base}/health", timeout=2.0).status_code == 200:
                    break
            except Exception:  # noqa: BLE001 - not up yet
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("throwaway server never became healthy")
            time.sleep(0.2)
        yield {"url": f"{base}/mcp/", "db": db_path}
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)


ORIGINAL_FACT = (
    "scripts/deploy.sh reads config/ports.env and defaults PORT to 8080 when unset"
)


EXTRACTED_FACT = (
    "post_session_extract.py resolves its bearer token from --token, then "
    "LM_AUTH_TOKEN, then ~/.config/living-memory/env, and never opens global.sqlite3"
)


CORRECTION = (
    "scripts/deploy.sh now reads config/ports.env but defaults PORT to 8443, not "
    "8080: commit f00dfeed changed the fallback and the 8080 claim is stale"
)


def _live_record(recall_event_id: str, node_id: str) -> SessionRecord:
    record = make_record(
        session_key="claude:live-source-session",
        cli_session_id="live-source-session",
        transport_session_ids=("live-source-transport",),
        transcript_sha256="cd" * 32,
        path="/tmp/live-source.jsonl",
    )
    span = SourceSpan("claude", record.path, 7)
    record.recalls.append(
        RecallInteraction(
            ordinal=0,
            query="deploy port default",
            span=span,
            recall_event_id=recall_event_id,
            delivered=[DeliveredNode(node_id=node_id, rank=0, content=ORIGINAL_FACT)],
        )
    )
    record.file_mutations.append(
        FileMutation(
            ordinal=0,
            path="config/ports.env",
            kind="update",
            span=SourceSpan("claude", record.path, 9),
            unified_diff=(
                "--- a/config/ports.env\n"
                "+++ b/config/ports.env\n"
                "@@ -1,1 +1,2 @@\n"
                " PORT=8080\n"
                f"+# {ORIGINAL_FACT}\n"
            ),
        )
    )
    return record
