"""Tests for scripts/check_deployed_protocol.py — the deployed-protocol check.

Three layers:

* the expected texts are *imported* from ``living_memory.server``, never copied
  into the script (asserted by identity and by scanning the script source), so
  the check cannot pass by comparing a stale hardcoded copy of the protocol;
* the match / drift / missing-tool verdicts and their exit codes are exercised
  offline with a stubbed fetch;
* one live smoke starts a real server from *this* worktree on a spare port and
  runs the checker end to end as a subprocess, in both the matching and the
  differing direction.
"""

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

import httpx
import pytest

from living_memory import server as lm_server

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DEPS_DIR = ROOT_DIR / ".cache" / "python-deps"
SCRIPT = ROOT_DIR / "scripts" / "check_deployed_protocol.py"

TOKEN = "test-deployed-protocol-token"
LIVE_SCOPE = "project:deployed-protocol-check"
READY_TIMEOUT_SECONDS = 60.0


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("check_deployed_protocol", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_module()


@pytest.fixture
def no_env_file(tmp_path: Path) -> Path:
    """An --env-file that does not exist, so tests never read the real one."""

    return tmp_path / "absent-env"


# --- the expected texts come from the source, not from a copy ---------------


def test_expected_texts_are_imported_from_the_server_module() -> None:
    texts = checker.expected_texts("global")

    assert texts["server instructions"] == lm_server._server_instructions("global")
    # Identity, not equality: a copy of the string would compare equal today and
    # rot the moment the protocol text changes.
    assert texts["memory_recall"] is lm_server._RECALL_DESCRIPTION
    assert texts["memory_remember"] is lm_server._REMEMBER_DESCRIPTION
    assert texts["memory_teach"] is lm_server._TEACH_DESCRIPTION
    assert texts["memory_consolidate"] is lm_server._CONSOLIDATE_DESCRIPTION
    assert set(texts) == {"server instructions", *checker.PROTOCOL_TOOLS}


def test_script_source_carries_no_copy_of_any_protocol_text() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    protocol_texts = [
        lm_server._server_instructions("global"),
        lm_server._RECALL_DESCRIPTION,
        lm_server._REMEMBER_DESCRIPTION,
        lm_server._TEACH_DESCRIPTION,
        lm_server._CONSOLIDATE_DESCRIPTION,
    ]
    for text in protocol_texts:
        # Any fragment long enough to be a paste rather than a coincidence.
        assert text[:60] not in source
        assert text[-60:] not in source


def test_expected_texts_reject_a_renamed_protocol_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(lm_server, "_RECALL_DESCRIPTION")
    with pytest.raises(checker.CheckError) as excinfo:
        checker.expected_texts("global")
    assert "_RECALL_DESCRIPTION" in str(excinfo.value)


# --- verdicts (offline) -----------------------------------------------------


def _stub_fetch(monkeypatch: pytest.MonkeyPatch, served: dict[str, str]) -> None:
    monkeypatch.setattr(
        checker, "fetch_served_texts", lambda url, **kwargs: dict(served)
    )


def test_matching_host_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_env_file: Path,
) -> None:
    _stub_fetch(monkeypatch, checker.expected_texts("global"))

    code = checker.main(["--env-file", str(no_env_file)])

    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("OK: http://127.0.0.1:8765/mcp/ serves this checkout")
    assert "memory_recall" in out


def test_stale_tool_description_exits_non_zero_with_a_readable_diff(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_env_file: Path,
) -> None:
    served = checker.expected_texts("global")
    # A stale host carries a sentence the checkout no longer has. Built by
    # prefixing rather than by naming any current wording: these texts are
    # rewritten regularly, and the check must not depend on today's phrasing.
    served["memory_recall"] = (
        "Superseded protocol sentence from an older release. " + served["memory_recall"]
    )
    _stub_fetch(monkeypatch, served)

    code = checker.main(["--env-file", str(no_env_file)])

    out = capsys.readouterr().out
    assert code == 1
    assert out.startswith("DRIFT:")
    assert "1 of 5 protocol texts differ: memory_recall." in out
    # The diff names both sides and shows the differing sentence, not a re-flow
    # of the whole paragraph.
    assert "+++ served (host)" in out
    assert "+Superseded protocol sentence from an older release." in out
    assert "docs/deployment.md" in out


def test_missing_sentence_in_server_instructions_exits_non_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_env_file: Path,
) -> None:
    expected = checker.expected_texts("global")
    served = dict(expected)
    # A host that lost a sentence the checkout has: the removal side of drift.
    served["server instructions"] = expected["server instructions"].rsplit(". ", 1)[0]
    _stub_fetch(monkeypatch, served)

    code = checker.main(["--env-file", str(no_env_file)])

    out = capsys.readouterr().out
    stripped = [line.strip() for line in out.splitlines()]
    removals = [
        line
        for line in stripped
        if line.startswith("-") and not line.startswith("---")
    ]
    assert code == 1
    assert "server instructions" in out
    assert removals  # the diff shows what only the checkout has
    assert "running older code" in out


def test_missing_protocol_tool_counts_as_drift(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_env_file: Path,
) -> None:
    served = checker.expected_texts("global")
    del served["memory_teach"]
    _stub_fetch(monkeypatch, served)

    code = checker.main(["--env-file", str(no_env_file)])

    out = capsys.readouterr().out
    assert code == 1
    assert "memory_teach" in out
    assert "does not expose this tool" in out


def test_default_scope_difference_is_reported_as_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_env_file: Path,
) -> None:
    served = checker.expected_texts("global")
    served["server instructions"] = lm_server._server_instructions("project:other")
    _stub_fetch(monkeypatch, served)

    code = checker.main(["--env-file", str(no_env_file)])

    out = capsys.readouterr().out
    assert code == 1
    assert "--default-scope project:other" in out
    assert "configuration, not stale code" in out
    assert "running older code" not in out


def test_json_verdict_reports_per_text_equality(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_env_file: Path,
) -> None:
    import json

    served = checker.expected_texts("global")
    served["memory_remember"] = served["memory_remember"] + " Extra."
    _stub_fetch(monkeypatch, served)

    code = checker.main(["--env-file", str(no_env_file), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["ok"] is False
    assert payload["differences"] == ["memory_remember"]
    assert payload["texts"]["memory_recall"]["equal"] is True
    assert payload["texts"]["memory_remember"]["equal"] is False
    assert (
        payload["texts"]["memory_remember"]["served_chars"]
        == payload["texts"]["memory_remember"]["expected_chars"] + 7
    )


def test_unreachable_host_exits_two_without_claiming_drift(
    capsys: pytest.CaptureFixture[str], no_env_file: Path
) -> None:
    code = checker.main(
        [
            "--host",
            "127.0.0.1",
            "--port",
            str(_free_port()),  # nothing is listening there
            "--timeout",
            "5",
            "--env-file",
            str(no_env_file),
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.startswith("CANNOT CHECK:")


# --- token resolution -------------------------------------------------------


def test_token_precedence_explicit_then_environment_then_env_file(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / "env"
    env_file.write_text(
        "LM_DB_PATH=/tmp/x.sqlite3\nLM_AUTH_TOKEN=from-file\n", encoding="utf-8"
    )

    assert (
        checker.resolve_token(
            "explicit", env_file=env_file, environ={"LM_AUTH_TOKEN": "from-env"}
        )
        == "explicit"
    )
    assert (
        checker.resolve_token(
            None, env_file=env_file, environ={"LM_AUTH_TOKEN": "from-env"}
        )
        == "from-env"
    )
    assert checker.resolve_token(None, env_file=env_file, environ={}) == "from-file"
    assert (
        checker.resolve_token(None, env_file=tmp_path / "absent", environ={}) is None
    )


# --- live smoke: a server started from THIS worktree ------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_env() -> dict[str, str]:
    env = os.environ.copy()
    env["LM_AUTH_TOKEN"] = TOKEN
    # Keep the live server off the real embedding model: the check under test
    # never touches embeddings, and loading torch would dominate the runtime.
    env.setdefault("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    parts = [str(SRC_DIR)]
    if DEPS_DIR.exists():
        parts.append(str(DEPS_DIR))
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _terminate(proc: "subprocess.Popen[bytes]") -> None:
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
def live_server(tmp_path: Path) -> Iterator[int]:
    """A real server from this worktree, on a spare port, with its own database."""

    pytest.importorskip("fastmcp")
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "living_memory.server",
            "--db",
            str(tmp_path / "deployed-protocol-check.sqlite3"),
            "--transport",
            "http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--default-scope",
            LIVE_SCOPE,
        ],
        env=_server_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        cwd=str(ROOT_DIR),
    )
    try:
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr else b""
                raise RuntimeError(
                    f"server exited early ({proc.returncode}): "
                    f"{stderr.decode('utf-8', 'replace')}"
                )
            try:
                health = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
                if health.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError(f"server on port {port} never became ready")
        yield port
    finally:
        _terminate(proc)


def _run_checker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT_DIR),
    )


def test_live_server_from_this_worktree_passes_the_check(live_server: int) -> None:
    result = _run_checker(
        "--port",
        str(live_server),
        "--token",
        TOKEN,
        "--default-scope",
        LIVE_SCOPE,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("OK: ")
    # The expectation must come from this worktree, not from the installed
    # (editable) package that points at the shared checkout.
    assert f"checkout: {SRC_DIR / 'living_memory' / 'server.py'}" in result.stdout


def test_live_server_serving_other_texts_fails_the_check(live_server: int) -> None:
    # The live server runs --default-scope LIVE_SCOPE; comparing against a
    # different scope makes its served instructions genuinely differ from the
    # expectation, so this exercises the non-zero verdict over the wire.
    result = _run_checker(
        "--port",
        str(live_server),
        "--token",
        TOKEN,
        "--default-scope",
        "global",
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.startswith("DRIFT:")
    assert "server instructions" in result.stdout
    assert f"--default-scope {LIVE_SCOPE}" in result.stdout


def test_live_server_rejects_a_wrong_token_without_claiming_drift(
    live_server: int,
) -> None:
    result = _run_checker(
        "--port",
        str(live_server),
        "--token",
        "not-the-token",
        "--default-scope",
        LIVE_SCOPE,
        "--timeout",
        "10",
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert result.stdout == ""
    assert result.stderr.startswith("CANNOT CHECK:")
