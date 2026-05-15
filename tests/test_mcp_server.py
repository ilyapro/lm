from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

from living_memory.server import create_mcp_server


class FakeMCP:
    def __init__(self, name: str, instructions: str | None = None) -> None:
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


def test_server_registers_exact_tools_resources_and_prompt(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)

    assert set(mcp.tools) == {
        "memory_remember",
        "memory_teach",
        "memory_connect",
        "memory_recall",
        "memory_consolidate",
        "memory_forget",
        "memory_status",
    }
    assert set(mcp.resources) == {
        "memory://global/concepts",
        "memory://project/{name}/concepts",
        "memory://stats",
        "memory://recent",
    }
    assert set(mcp.prompts) == {"memory://prompt/retrieval_context"}


def test_mcp_tools_delegate_to_memory_services(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)

    first = mcp.tools["memory_remember"](
        "deploy incident caused by missing migration",
        {"scope": "project:alpha", "agent": "agent-a"},
    )
    second = mcp.tools["memory_remember"](
        "missing migration requires rollback checklist",
        {"scope": "project:alpha", "agent": "agent-b"},
    )
    first_id = first["node"]["id"]
    second_id = second["node"]["id"]

    connected = mcp.tools["memory_connect"](first_id, second_id, "caused")
    assert connected["connection"]["type"] == "caused"

    recalled = mcp.tools["memory_recall"](
        "why did rollback checklist happen",
        scope="project:alpha",
        depth="causal",
        max_results=5,
    )
    assert recalled["count"] >= 1
    assert first_id in {result["node"]["id"] for result in recalled["results"]}

    taught = mcp.tools["memory_teach"](
        first_id,
        "deploy incident caused by migration 43 missing",
        context={"agent": "agent-c"},
    )
    assert taught["supersedes"]["type"] == "supersedes"
    assert taught["corrective_trace"]["content"] == "deploy incident caused by migration 43 missing"

    consolidated = mcp.tools["memory_consolidate"](scope="project:alpha", force=True)
    assert "concepts_created" in consolidated
    assert "decayed" in consolidated

    status = mcp.tools["memory_status"](scope="project:alpha")
    assert status["scope"] == "project:alpha"
    assert status["phase"]["trace_count"] >= 3

    forgotten = mcp.tools["memory_forget"](second_id, "test cleanup")
    assert forgotten["node"]["decayed"] is True
    assert forgotten["node"]["decay_reason"] == "test cleanup"


def test_server_default_scope_applies_to_unspecified_trace_scope(tmp_path: Path) -> None:
    mcp = create_mcp_server(
        tmp_path / "memory.sqlite3",
        default_scope="project:scope-test",
        mcp_factory=FakeMCP,
    )

    assert "Your default scope is project:scope-test" in str(mcp.instructions)

    remembered = mcp.tools["memory_remember"](
        "trace without explicit scope",
        {"agent": "agent-a"},
    )

    assert remembered["node"]["scope"] == "project:scope-test"
    assert remembered["node"]["context"]["scope"] == "project:scope-test"


def test_server_help_does_not_require_runtime_dependency() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "living_memory.server", "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0
    assert "SQLite" in result.stdout
    assert "--default-scope" in result.stdout
