from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from living_memory.resources import node_to_dict
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


# --- Fetch-by-id: the re-fetch path for snippet/duplicate recall stubs ------


def test_memory_lookup_by_id_round_trips_full_content(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    long_content = (
        "лунный трактор пересекает ρ-многообразие — unicode survives\n"
        "line two: exact bytes preserved, including trailing spaces  \n"
        + "z" * 2000
    )
    remembered = mcp.tools["memory_remember"](long_content, {"scope": "project:idfetch"})
    target_id = remembered["node"]["id"]

    fetched = mcp.tools["memory_lookup"](node_id=target_id)
    assert "error" not in fetched
    assert fetched["scope"] is None
    assert fetched["filters"] == {}
    assert fetched["count"] == 1
    assert fetched["missing"] == []
    node = fetched["results"][0]
    assert node["id"] == target_id
    assert node["content"] == long_content  # byte-identical, never snippeted
    assert set(node) == set(node_to_dict(mcp.memory_store.get_node(target_id)))

    # `level` defaults to "trace" but is an exact-context concern; id fetch
    # ignores it, so a mismatching level cannot hide the node.
    by_level = mcp.tools["memory_lookup"](node_id=target_id, level="concept")
    assert [entry["id"] for entry in by_level["results"]] == [target_id]


def test_memory_lookup_by_ids_preserves_order_and_lists_missing(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    first = mcp.tools["memory_remember"]("id-fetch first", {"scope": "project:idfetch"})
    second = mcp.tools["memory_remember"]("id-fetch second", {"scope": "project:idfetch"})
    first_id = first["node"]["id"]
    second_id = second["node"]["id"]

    fetched = mcp.tools["memory_lookup"](
        node_ids=[second_id, "01NOPEnotarealnodeid000000", first_id]
    )
    assert fetched["count"] == 2
    assert [entry["id"] for entry in fetched["results"]] == [second_id, first_id]
    assert fetched["missing"] == ["01NOPEnotarealnodeid000000"]

    # node_id combines with node_ids, deduplicated in request order.
    combined = mcp.tools["memory_lookup"](node_id=first_id, node_ids=[second_id, first_id])
    assert [entry["id"] for entry in combined["results"]] == [first_id, second_id]
    assert combined["missing"] == []


def test_memory_lookup_id_with_scope_verifies_instead_of_filtering(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    remembered = mcp.tools["memory_remember"](
        "scoped id-fetch trace", {"scope": "project:alpha"}
    )
    target_id = remembered["node"]["id"]

    matching = mcp.tools["memory_lookup"](scope="project:alpha", node_id=target_id)
    assert [entry["id"] for entry in matching["results"]] == [target_id]
    assert matching["scope_mismatches"] == []

    crossed = mcp.tools["memory_lookup"](scope="project:beta", node_id=target_id)
    assert [entry["id"] for entry in crossed["results"]] == [target_id]  # not filtered
    assert crossed["scope_mismatches"] == [{"id": target_id, "scope": "project:alpha"}]


def test_memory_lookup_by_id_returns_decayed_nodes(tmp_path: Path) -> None:
    """Exact refs must survive decay: a content_ref handed out before a sweep
    still resolves, with the decayed flag visible to the caller."""

    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    remembered = mcp.tools["memory_remember"](
        "decay-surviving id-fetch trace", {"scope": "project:idfetch"}
    )
    target_id = remembered["node"]["id"]
    mcp.tools["memory_forget"](target_id, "test decay")

    fetched = mcp.tools["memory_lookup"](node_id=target_id)
    assert fetched["count"] == 1
    assert fetched["results"][0]["decayed"] is True
    assert fetched["results"][0]["content"] == "decay-surviving id-fetch trace"


def test_memory_lookup_ids_cannot_combine_with_context_filters(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    remembered = mcp.tools["memory_remember"](
        "combination probe trace",
        {"scope": "project:idfetch", "task_pattern": "combo-pattern"},
    )

    combined = mcp.tools["memory_lookup"](
        node_id=remembered["node"]["id"], task_pattern="combo-pattern"
    )
    assert combined["error"] == "node_id/node_ids cannot be combined with context filters"
    assert combined["count"] == 0
    assert combined["results"] == []


def test_memory_lookup_scope_optional_only_for_id_fetch(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)

    filters_without_scope = mcp.tools["memory_lookup"](task_pattern="some-pattern")
    assert filters_without_scope["error"] == "scope is required with context filters"
    assert filters_without_scope["results"] == []

    nothing = mcp.tools["memory_lookup"]()
    assert nothing["error"] == "at least one exact context filter is required"
    assert nothing["results"] == []
