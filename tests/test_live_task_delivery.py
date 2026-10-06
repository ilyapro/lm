"""Exercise the delivery observer against an isolated HTTP MCP service."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import pytest

from living_memory.server import create_mcp_server


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/observe_live_task_recall.py"
PRIVATE_QUERY = "alabaster-queue-73921"


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def isolated_reader(tmp_path: Path, request):
    pytest.importorskip("fastmcp")
    database = tmp_path / "isolated.sqlite3"
    mcp = create_mcp_server(database)
    node = mcp.memory_store.append_trace(
        "The integrated release comparison includes the accepted queue outcome.",
        {"scope": "project:test", "task": PRIVATE_QUERY, "agent": "fixture"},
    )
    mcp.memory_store.close()
    port = _port()
    env = os.environ.copy()
    source = ROOT / "src"
    if getattr(request, "param", None) == "changed-functions":
        source = tmp_path / "changed-src"
        shutil.copytree(ROOT / "src/living_memory", source / "living_memory")
        server = source / "living_memory/server.py"
        server.write_text(server.read_text().replace(
            "return bool(expected) and secrets.compare_digest(",
            "return (bool(expected) is True) and secrets.compare_digest(",
        ))
    env["PYTHONPATH"] = str(source)
    env["LIVING_MEMORY_EMBEDDING_BACKEND"] = "hash"
    env["LM_AUTH_TOKEN"] = "isolated-delivery-fixture-token"
    command = [sys.executable, "-m", "living_memory.server", "--db", str(database),
               "--transport", "http", "--host", "127.0.0.1", "--port", str(port)]
    with (tmp_path / "service.log").open("wb") as log:
        process = subprocess.Popen(
            command, cwd=tmp_path, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            health = None
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                        health = json.load(response)
                    break
                except OSError:
                    time.sleep(0.1)
            assert health is not None, "isolated service did not start"
            yield {"port": port, "health": health, "node_id": node.id, "env": env}
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


def _invoke(reader: dict, cases_file: Path, output: Path, *, stale: bool = False,
            native: bool = False, commit: str | None = None):
    health = reader["health"]
    command = [
        sys.executable, str(SCRIPT), "--port", str(reader["port"]),
        "--cases-file", str(cases_file), "--output", str(output),
        "--prior-boot-id", health["boot_id"] if stale else "previous-boot",
        "--prior-digest", health["code_identity"]["digest"] if stale else "0" * 64,
        "--env-file", str(cases_file.parent / "absent-env"),
    ]
    if native:
        revision = commit or subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
        ).strip()
        command.extend([
            "--native-completion", "--installed-root", str(ROOT),
            "--published-commit", revision, "--loaded-path", "src/living_memory/storage.py",
        ])
    return subprocess.run(command, env=reader["env"], text=True, capture_output=True, timeout=90)


def test_fresh_read_only_observation_has_no_private_values(
    isolated_reader: dict, tmp_path: Path
) -> None:
    cases_file = tmp_path / "private-cases.json"
    cases_file.write_text(json.dumps({"queries": [
        {"query": PRIVATE_QUERY, "expected_node_id": isolated_reader["node_id"]},
        {"query": PRIVATE_QUERY + " integrated release comparison",
         "expected_node_id": isolated_reader["node_id"]},
    ]}))
    output = tmp_path / "receipt.json"
    result = _invoke(isolated_reader, cases_file, output)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(output.read_text())
    assert receipt["ok"] is True
    assert receipt["boot_id"] == isolated_reader["health"]["boot_id"]
    assert receipt["code_identity"] == isolated_reader["health"]["code_identity"]
    assert receipt["unscoped_recall"] is True
    assert all(case["found"] and case["rank"] <= 6 for case in receipt["cases"])
    emitted = result.stdout + result.stderr + output.read_text()
    assert PRIVATE_QUERY not in emitted
    assert isolated_reader["node_id"] not in emitted


def test_stale_reader_and_failed_case_are_safe(
    isolated_reader: dict, tmp_path: Path
) -> None:
    cases_file = tmp_path / "private-cases.json"
    cases_file.write_text(json.dumps({"queries": [
        {"query": PRIVATE_QUERY, "expected_node_id": isolated_reader["node_id"]},
    ]}))
    output = tmp_path / "receipt.json"
    stale = _invoke(isolated_reader, cases_file, output, stale=True)
    assert stale.returncode == 2
    assert not output.exists()
    cases_file.write_text(json.dumps({"queries": [
        {"query": PRIVATE_QUERY, "expected_node_id": "private-absent-node"},
    ]}))
    missed = _invoke(isolated_reader, cases_file, output)
    assert missed.returncode == 1
    assert json.loads(output.read_text())["ok"] is False
    assert "private-absent-node" not in missed.stdout + missed.stderr + output.read_text()


def test_private_case_file_inside_git_is_rejected(tmp_path: Path) -> None:
    command = [
        sys.executable, str(SCRIPT), "--cases-file", str(__file__),
        "--output", str(tmp_path / "receipt.json"),
        "--prior-boot-id", "old", "--prior-digest", "0" * 64,
    ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=10)
    assert result.returncode == 2
    assert not (tmp_path / "receipt.json").exists()
    assert "CANNOT CHECK" in result.stderr


def test_native_projection_binds_process_published_code_and_fresh_public_response(
    isolated_reader: dict, tmp_path: Path,
) -> None:
    cases = tmp_path / "native-private.json"
    cases.write_text(json.dumps({"queries": [
        {"query": PRIVATE_QUERY, "expected_node_id": isolated_reader["node_id"]},
    ]}))
    output = tmp_path / "native-receipt.json"
    result = _invoke(isolated_reader, cases, output, native=True)
    assert result.returncode == 0, result.stderr
    observation = json.loads(result.stdout)
    assert observation == json.loads(output.read_text())
    assert observation["behavior_passed"] is True
    loaded = observation["loaded_bytes"]
    assert len(loaded) == 31
    assert {
        "pyproject.toml",
        "src/living_memory/__init__.py",
        "src/living_memory/server.py",
        "src/living_memory/config.py",
        "src/living_memory/storage.py",
        "src/living_memory/retrieval.py",
        "src/living_memory/embeddings.py",
        "src/living_memory/feedback.py",
    } <= set(loaded)
    assert loaded == {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in loaded
    }
    process = json.loads(observation["start_identity"])
    assert process["pid"] == observation["pid"]
    assert Path(f"/proc/{process['pid']}").exists()
    assert json.loads(observation["observer_identity"])["boot_id"] == process["boot_id"]
    assert len(observation["response_sha256"]) == 64
    assert observation["source_binding"]["candidate_digest"] == observation["code_identity"]["digest"]
    assert observation["source_binding"]["installed_import_matches"] is True
    emitted = result.stdout + result.stderr + output.read_text()
    assert PRIVATE_QUERY not in emitted
    assert isolated_reader["node_id"] not in emitted
    assert isolated_reader["env"]["LM_AUTH_TOKEN"] not in emitted
    wrong = _invoke(isolated_reader, cases, tmp_path / "wrong.json", native=True, commit="0" * 40)
    assert wrong.returncode == 2
    assert not (tmp_path / "wrong.json").exists()


@pytest.mark.parametrize("isolated_reader", ["changed-functions"], indirect=True)
def test_native_projection_rejects_foreign_loaded_code(
    isolated_reader: dict, tmp_path: Path,
) -> None:
    cases = tmp_path / "native-private.json"
    cases.write_text(json.dumps({"queries": [
        {"query": PRIVATE_QUERY, "expected_node_id": isolated_reader["node_id"]},
    ]}))
    output = tmp_path / "native-receipt.json"
    result = _invoke(isolated_reader, cases, output, native=True)
    assert result.returncode == 2
    assert not output.exists()
    assert "CANNOT CHECK" in result.stderr
