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

import asyncio
import hashlib
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import httpx
import pytest

from living_memory import server as lm_server
from living_memory.instructions_map import compose_map_section
from living_memory.storage import MemoryStore

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
    """A synthetic target env file, with no token or private service settings."""
    path = tmp_path / "env"
    path.write_text("LM_EXPLICIT_FEEDBACK_PROMPT=optional\n", encoding="utf-8")
    return path


# --- the expected texts come from the source, not from a copy ---------------


def test_expected_texts_are_imported_from_the_server_module() -> None:
    texts = checker.expected_texts("global")

    assert checker.PROTOCOL_TOOLS == {
        "memory_recall", "memory_remember", "memory_teach", "memory_lookup"
    }
    assert texts == lm_server._effective_protocol_texts("global", "optional")
    assert texts["server instructions"] == lm_server._server_instructions("global")
    # Identity, not equality: a copy of the string would compare equal today and
    # rot the moment the protocol text changes.
    assert texts["memory_recall"] is lm_server._RECALL_DESCRIPTION
    assert texts["memory_remember"] is lm_server._REMEMBER_DESCRIPTION
    assert texts["memory_teach"] is lm_server._TEACH_DESCRIPTION
    assert texts["memory_lookup"] == lm_server._effective_protocol_texts(
        "global", "mandatory"
    )["memory_lookup"]
    assert set(texts) == {"server instructions", *checker.PROTOCOL_TOOLS}


def test_script_source_carries_no_copy_of_any_protocol_text() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    protocol_texts = [
        lm_server._server_instructions("global"),
        lm_server._RECALL_DESCRIPTION,
        lm_server._REMEMBER_DESCRIPTION,
        lm_server._TEACH_DESCRIPTION,
        checker.expected_texts("global")["memory_lookup"],
    ]
    for text in protocol_texts:
        # Any fragment long enough to be a paste rather than a coincidence.
        assert text[:60] not in source
        assert text[-60:] not in source


def test_expected_texts_reject_a_renamed_protocol_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(lm_server, "_RECALL_DESCRIPTION")
    with pytest.raises(checker.CheckError, match="_RECALL_DESCRIPTION"):
        checker.expected_texts("global")


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
    assert (
        "5 checked: memory_lookup, memory_recall, memory_remember, memory_teach, "
        "server instructions" in out
    )
    assert "memory_consolidate" not in out


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
    assert "Measured protocol text differs" in out


def test_missing_protocol_tool_counts_as_drift(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_env_file: Path,
) -> None:
    served = checker.expected_texts("global")
    del served["memory_lookup"]
    _stub_fetch(monkeypatch, served)

    code = checker.main(["--env-file", str(no_env_file)])

    out = capsys.readouterr().out
    assert code == 1
    assert "memory_lookup" in out
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
    assert "configuration drift" in out
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
    assert payload["checked"] == [
        "memory_lookup",
        "memory_recall",
        "memory_remember",
        "memory_teach",
        "server instructions",
    ]
    assert payload["texts"]["memory_recall"]["equal"] is True
    assert payload["texts"]["memory_remember"]["equal"] is False
    assert (
        payload["texts"]["memory_remember"]["served_chars"]
        == payload["texts"]["memory_remember"]["expected_chars"] + 7
    )


@pytest.mark.parametrize("arm", ["optional", "mandatory"])
@pytest.mark.parametrize("addition", ["canonical", "malformed", "repeated", "whitespace"])
def test_complete_instructions_determine_both_reports(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_env_file: Path,
    arm: str,
    addition: str,
) -> None:
    expected = checker.expected_texts(LIVE_SCOPE, arm)
    tail = "\n" + compose_map_section([{
        "recall_map": {"clusters": [{"label": "another consumer task", "count": 3}]},
    }]) + "\n"
    assert "## Memory nearby" in tail
    suffix = {
        "canonical": tail,
        "malformed": tail + "Unexpected private history.\n",
        "repeated": tail * 2,
        "whitespace": " \n",
    }[addition]
    served = dict(expected)
    served[checker.INSTRUCTIONS_KEY] += suffix
    _stub_fetch(monkeypatch, served)
    arguments = ["--env-file", str(no_env_file), "--default-scope", LIVE_SCOPE,
                 "--prompt-arm", arm]

    assert checker.main(arguments) == 1
    report = capsys.readouterr().out
    assert report.startswith("DRIFT:")
    assert "1 of 5 protocol texts differ: server instructions." in report
    assert "host data" not in report
    full_text = served[checker.INSTRUCTIONS_KEY]
    digest = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    assert f"served {len(full_text)} chars" in report
    assert f"served {len(full_text.encode('utf-8'))}" in report
    assert f"served sha256: {digest}" in report

    assert checker.main([*arguments, "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["differences"] == [checker.INSTRUCTIONS_KEY]
    assert "host_data" not in payload
    measurement = payload["texts"][checker.INSTRUCTIONS_KEY]
    assert measurement["equal"] is False
    assert measurement["served_chars"] == len(full_text)
    assert measurement["served_bytes"] == len(full_text.encode("utf-8"))
    assert measurement["served_sha256"] == digest
    assert measurement["expected_chars"] == len(expected[checker.INSTRUCTIONS_KEY])
    assert measurement["expected_bytes"] == len(expected[checker.INSTRUCTIONS_KEY].encode("utf-8"))
    assert measurement["expected_sha256"] == hashlib.sha256(
        expected[checker.INSTRUCTIONS_KEY].encode("utf-8")
    ).hexdigest()


class _ProtocolClient(SimpleNamespace):
    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def list_tools(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=name, description=text)
                for name, text in checker.expected_texts().items()
                if name in checker.PROTOCOL_TOOLS]


@pytest.mark.parametrize("interface", ["modern", "legacy"])
@pytest.mark.parametrize("value", ["protocol", "", None])
def test_client_reads_supported_instructions_without_masking_empty_values(
    monkeypatch: pytest.MonkeyPatch,
    interface: str,
    value: str | None,
) -> None:
    import fastmcp

    expected = checker.expected_texts()
    instructions = expected[checker.INSTRUCTIONS_KEY] if value == "protocol" else value
    if interface == "modern":
        client = _ProtocolClient(instructions=instructions, initialize_result=None)
    else:
        client = _ProtocolClient(initialize_result=SimpleNamespace(instructions=instructions))
    observed: dict[str, Any] = {}

    def make_client(url: str, **kwargs: Any) -> _ProtocolClient:
        observed.update(kwargs)
        return client

    monkeypatch.setattr(fastmcp, "Client", make_client)
    served = checker.fetch_served_texts("http://test.invalid/mcp/", token=TOKEN,
                                       verify=False, timeout=2.0)
    assert observed == {"auth": TOKEN, "verify": False, "timeout": 2.0, "init_timeout": 2.0}
    assert served[checker.INSTRUCTIONS_KEY] == (instructions if instructions is not None else "")
    assert set(served) == set(expected)
    assert [item.name for item in checker.compare(expected, served)] == (
        [] if value == "protocol" else [checker.INSTRUCTIONS_KEY]
    )


def test_modern_empty_instructions_do_not_fall_back_to_legacy_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fastmcp

    client = _ProtocolClient(instructions=None, initialize_result=SimpleNamespace(
        instructions=checker.expected_texts()[checker.INSTRUCTIONS_KEY],
    ))
    monkeypatch.setattr(fastmcp, "Client", lambda *args, **kwargs: client)
    served = checker.fetch_served_texts("http://test.invalid/mcp/")
    assert served[checker.INSTRUCTIONS_KEY] == ""


@pytest.mark.parametrize("attributes", [{}, {"initialize_result": None},
                                       {"initialize_result": SimpleNamespace()},
                                       {"instructions": 42}])
def test_unavailable_or_invalid_instructions_are_cannot_check(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_env_file: Path,
    attributes: dict[str, Any],
) -> None:
    import fastmcp

    monkeypatch.setattr(fastmcp, "Client", lambda *args, **kwargs: _ProtocolClient(**attributes))
    assert checker.main(["--env-file", str(no_env_file), "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("CANNOT CHECK:")


def test_instruction_read_failure_does_not_fall_back_or_disclose_exception_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_env_file: Path,
) -> None:
    import fastmcp

    class UnreadableClient(_ProtocolClient):
        @property
        def instructions(self) -> str:
            raise RuntimeError(TOKEN)

    client = UnreadableClient(initialize_result=SimpleNamespace(
        instructions=checker.expected_texts()[checker.INSTRUCTIONS_KEY],
    ))
    monkeypatch.setattr(fastmcp, "Client", lambda *args, **kwargs: client)
    assert checker.main(["--env-file", str(no_env_file), "--token", TOKEN]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "RuntimeError" in captured.err
    assert TOKEN not in captured.err


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
            "--prompt-arm",
            "optional",
            "--default-scope",
            "global",
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
    assert checker.resolve_token(
        "explicit", env_file=env_file, environ={"LM_AUTH_TOKEN": "from-env"}
    ) == "explicit"
    assert checker.resolve_token(
        None, env_file=env_file, environ={"LM_AUTH_TOKEN": "from-env"}
    ) == "from-env"
    assert checker.resolve_token(None, env_file=env_file, environ={}) == "from-file"
    assert checker.resolve_token(
        None, env_file=tmp_path / "absent", environ={}
    ) is None


def test_local_env_file_selects_mandatory_without_disclosing_secrets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    env_file = tmp_path / "env"
    env_file.write_text(
        "LM_AUTH_TOKEN=private-synthetic-token\n"
        "LM_EXPLICIT_FEEDBACK_PROMPT=mandatory\n",
        encoding="utf-8",
    )
    _stub_fetch(monkeypatch, checker.expected_texts("global", "mandatory"))
    assert checker.main(["--env-file", str(env_file)]) == 0
    report = capsys.readouterr().out
    assert "prompt arm=mandatory" in report
    assert "private-synthetic-token" not in report


def test_wrong_expected_prompt_arm_is_drift(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_env_file: Path,
) -> None:
    _stub_fetch(monkeypatch, checker.expected_texts("global", "mandatory"))
    assert checker.main(["--env-file", str(no_env_file)]) == 1
    out = capsys.readouterr().out
    assert "memory_recall" in out
    assert "served recall description matches prompt arm=mandatory" in out


def test_explicit_target_does_not_send_default_local_env_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_env = tmp_path / "local-env"
    local_env.write_text(
        "LM_AUTH_TOKEN=local-private-token\nLM_EXPLICIT_FEEDBACK_PROMPT=mandatory\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(checker, "DEFAULT_ENV_FILE", local_env)
    monkeypatch.delenv("LM_AUTH_TOKEN", raising=False)
    observed: list[str | None] = []

    def fetch(url: str, *, token: str | None, **kwargs: Any) -> dict[str, str]:
        observed.append(token)
        return checker.expected_texts("global", "optional")

    monkeypatch.setattr(checker, "fetch_served_texts", fetch)
    assert checker.main([
        "--host", "example.invalid", "--default-scope", "global",
        "--prompt-arm", "optional",
    ]) == 0
    assert observed == [None]


def test_unresolved_remote_expectations_are_an_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert checker.main(["--host", "example.invalid", "--port", "8765"]) == 2
    assert "default scope is unknown" in capsys.readouterr().err
    assert checker.main([
        "--host", "example.invalid", "--port", "8765",
        "--default-scope", "global",
    ]) == 2
    assert "PROMPT" in capsys.readouterr().err


def test_invalid_tls_certificate_is_cannot_check(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert checker.main([
        "--tls", "--port", "8766", "--prompt-arm", "optional",
        "--ca-cert", str(tmp_path / "absent-cert.pem"),
    ]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "CANNOT CHECK: cannot configure TLS verification" in captured.err


@pytest.mark.parametrize("arm", ["optional", "mandatory"])
def test_real_fastmcp_projection_keeps_history_private_and_detects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
) -> None:
    """Private synthetic history must not enter a new client's initialization."""
    path = tmp_path / "synthetic.sqlite3"
    with MemoryStore(path) as store:
        store.record_recall_event(
            query="find a deploy recipe",
            scope=LIVE_SCOPE,
            ambient_context={"task": "synthetic-protocol"},
            recall_map={
                "clusters": [{
                    "label": "deploy recipes", "count": 3,
                    "medoid": {"node_id": "01AAAAAAAAAAAAAAAAAAAAAAAA", "example": "synthetic"},
                    "ask_hint": "deploy recipes",
                    "plan_item": "on touching deploy recipes - recall 'deploy recipes' (3)",
                }],
                "pool": 3, "covered": 3,
            },
        )
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_PROMPT", arm)
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    mcp = lm_server.create_mcp_server(path, default_scope=LIVE_SCOPE)

    try:
        served = asyncio.run(checker._fetch_async(mcp, token=None, verify=None, timeout=10))
    finally:
        mcp.auto_consolidation.close()
        mcp.memory_store.close()
    assert "deploy recipes" not in served["server instructions"]
    expected = checker.expected_texts(LIVE_SCOPE, arm)
    assert checker.compare(expected, served) == []
    assert "## Memory nearby" not in served["server instructions"]

    wrong_arm = "mandatory" if arm == "optional" else "optional"
    assert [item.name for item in checker.compare(
        checker.expected_texts(LIVE_SCOPE, wrong_arm), served
    )] == ["memory_recall"]

    changed = dict(served)
    changed["server instructions"] = changed["server instructions"].replace(
        "You MUST recall BEFORE you act", "Recall later", 1
    )
    assert changed["server instructions"] != served["server instructions"]
    assert [item.name for item in checker.compare(expected, changed)] == ["server instructions"]
    changed = dict(served)
    changed["memory_lookup"] += " Unapproved change."
    assert [item.name for item in checker.compare(expected, changed)] == ["memory_lookup"]
    changed = dict(served)
    del changed["memory_lookup"]
    assert [item.name for item in checker.compare(expected, changed)] == ["memory_lookup"]
    changed = dict(served)
    changed["server instructions"] += "Unapproved trailing core text."
    assert changed["server instructions"] != served["server instructions"]
    assert [item.name for item in checker.compare(expected, changed)] == ["server instructions"]

# --- live smoke: a server started from THIS worktree ------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_env() -> dict[str, str]:
    env = os.environ.copy()
    env["LM_AUTH_TOKEN"] = TOKEN
    env["LM_EXPLICIT_FEEDBACK_PROMPT"] = "optional"
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
        "--prompt-arm",
        "optional",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("OK: ")
    assert "memory_lookup" in result.stdout
    assert "memory_consolidate" not in result.stdout
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
        "--prompt-arm",
        "optional",
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
        "--prompt-arm",
        "optional",
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert result.stdout == ""
    assert result.stderr.startswith("CANNOT CHECK:")
    assert "not-the-token" not in result.stderr
