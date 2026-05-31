from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore


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


def test_list_nodes_by_context_finds_exact_task_pattern_in_dense_scope(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        target = store.append_trace(
            "target trace for deterministic lookup",
            {"scope": "project:lookup", "task_pattern": "pattern-target"},
        )
        for index in range(45):
            store.append_trace(
                f"decoy trace {index}",
                {
                    "scope": "project:lookup",
                    "task_pattern": f"pattern-decoy-{index}",
                },
            )
        store.append_trace(
            "other scope matching task pattern",
            {"scope": "project:other", "task_pattern": "pattern-target"},
        )

        matches = store.list_nodes_by_context(
            scope="project:lookup",
            context_filters={"task_pattern": "pattern-target"},
            limit=1000,
        )

        assert [node.id for node in matches] == [target.id]


def test_list_nodes_by_context_and_filters_and_excludes_decayed_by_default(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        target = store.append_trace(
            "and-filter target trace",
            {
                "scope": "project:lookup",
                "task_pattern": "shared-pattern",
                "procedure_id": "proc-a",
                "lesson_kind": "fix",
            },
        )
        store.append_trace(
            "same task pattern different procedure",
            {
                "scope": "project:lookup",
                "task_pattern": "shared-pattern",
                "procedure_id": "proc-b",
                "lesson_kind": "fix",
            },
        )
        store.append_trace(
            "same procedure different lesson kind",
            {
                "scope": "project:lookup",
                "task_pattern": "shared-pattern",
                "procedure_id": "proc-a",
                "lesson_kind": "reference",
            },
        )

        matches = store.list_nodes_by_context(
            scope="project:lookup",
            context_filters={
                "task_pattern": "shared-pattern",
                "procedure_id": "proc-a",
                "lesson_kind": "fix",
            },
        )
        assert [node.id for node in matches] == [target.id]
        assert (
            store.list_nodes_by_context(
                scope="project:lookup",
                context_filters={"task_pattern": "does-not-exist"},
            )
            == []
        )

        decayed = store.append_trace(
            "decayed exact lookup trace",
            {"scope": "project:lookup", "task_pattern": "expired-pattern"},
        )
        store.soft_delete_node(decayed.id, "test decay")

        assert (
            store.list_nodes_by_context(
                scope="project:lookup",
                context_filters={"task_pattern": "expired-pattern"},
            )
            == []
        )
        with_decayed = store.list_nodes_by_context(
            scope="project:lookup",
            context_filters={"task_pattern": "expired-pattern"},
            include_decayed=True,
        )
        assert [node.id for node in with_decayed] == [decayed.id]

        with pytest.raises(ValueError, match="unsupported context lookup field"):
            store.list_nodes_by_context(
                scope="project:lookup",
                context_filters={"task_pattern[0]": "shared-pattern"},
            )


def test_list_nodes_by_context_returns_newest_first_and_honors_limit(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        older = store.append_trace(
            "older ordered lookup trace",
            {
                "scope": "project:lookup",
                "task_pattern": "ordered-pattern",
                "timestamp": "2026-01-01T00:00:00Z",
            },
        )
        newer = store.append_trace(
            "newer ordered lookup trace",
            {
                "scope": "project:lookup",
                "task_pattern": "ordered-pattern",
                "timestamp": "2026-01-02T00:00:00Z",
            },
        )

        matches = store.list_nodes_by_context(
            scope="project:lookup",
            context_filters={"task_pattern": "ordered-pattern"},
        )
        limited = store.list_nodes_by_context(
            scope="project:lookup",
            context_filters={"task_pattern": "ordered-pattern"},
            limit=1,
        )

        assert [node.id for node in matches] == [newer.id, older.id]
        assert [node.id for node in limited] == [newer.id]


def test_memory_lookup_tool_is_registered_and_deterministic(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    assert "memory_lookup" in mcp.tools

    target = mcp.tools["memory_remember"](
        "MCP lookup target trace",
        {"scope": "project:mcp-lookup", "task_pattern": "mcp-target-pattern"},
    )
    target_id = target["node"]["id"]
    for index in range(35):
        mcp.tools["memory_remember"](
            f"MCP lookup decoy trace {index}",
            {
                "scope": "project:mcp-lookup",
                "task_pattern": f"mcp-decoy-pattern-{index}",
            },
        )
    other_scope = mcp.tools["memory_remember"](
        "MCP lookup other scope trace",
        {"scope": "project:mcp-other", "task_pattern": "mcp-target-pattern"},
    )

    exact = mcp.tools["memory_lookup"](
        scope="project:mcp-lookup",
        task_pattern="mcp-target-pattern",
    )
    assert exact["scope"] == "project:mcp-lookup"
    assert exact["filters"] == {"task_pattern": "mcp-target-pattern"}
    assert exact["count"] == 1
    assert [result["id"] for result in exact["results"]] == [target_id]
    assert exact["results"][0]["context"]["task_pattern"] == "mcp-target-pattern"
    assert "score" not in exact["results"][0]
    assert other_scope["node"]["id"] not in {result["id"] for result in exact["results"]}

    and_target = mcp.tools["memory_remember"](
        "MCP lookup AND target trace",
        {
            "scope": "project:mcp-lookup",
            "task_pattern": "mcp-shared-pattern",
            "procedure_id": "proc-a",
            "lesson_kind": "fix",
        },
    )
    mcp.tools["memory_remember"](
        "MCP lookup AND nonmatch trace",
        {
            "scope": "project:mcp-lookup",
            "task_pattern": "mcp-shared-pattern",
            "procedure_id": "proc-b",
            "lesson_kind": "fix",
        },
    )
    and_lookup = mcp.tools["memory_lookup"](
        scope="project:mcp-lookup",
        task_pattern="mcp-shared-pattern",
        procedure_id="proc-a",
        lesson_kind="fix",
    )
    assert [result["id"] for result in and_lookup["results"]] == [and_target["node"]["id"]]

    missing = mcp.tools["memory_lookup"](
        scope="project:mcp-lookup",
        task_pattern="no-such-pattern",
    )
    assert missing["count"] == 0
    assert missing["results"] == []

    decayed = mcp.tools["memory_remember"](
        "MCP lookup decayed trace",
        {"scope": "project:mcp-lookup", "task_pattern": "expired-pattern"},
    )
    mcp.tools["memory_forget"](decayed["node"]["id"], "test decay")
    decayed_lookup = mcp.tools["memory_lookup"](
        scope="project:mcp-lookup",
        task_pattern="expired-pattern",
    )
    assert decayed_lookup["count"] == 0

    no_filters = mcp.tools["memory_lookup"](scope="project:mcp-lookup")
    assert no_filters["error"] == "at least one exact context filter is required"
    assert no_filters["results"] == []
