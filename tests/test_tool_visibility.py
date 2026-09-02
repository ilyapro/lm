from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from living_memory.server import create_mcp_server


AGENT_TOOL_NAMES = {
    "memory_lookup",
    "memory_recall",
    "memory_remember",
    "memory_teach",
}
OPERATOR_TOOL_NAMES = {
    "memory_connect",
    "memory_consolidate",
    "memory_forget",
    "memory_health",
    "memory_status",
}


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.delenv("LM_EXPOSE_OPERATOR_TOOLS", raising=False)
    monkeypatch.delenv("LM_EXPOSE_ATTEST", raising=False)


def test_default_tools_list_advertises_only_agent_tools(tmp_path: Path) -> None:
    asyncio.run(
        _assert_advertised_tools(
            tmp_path / "default.sqlite3",
            AGENT_TOOL_NAMES,
        )
    )


def test_constructor_true_advertises_all_core_tools(tmp_path: Path) -> None:
    asyncio.run(
        _assert_advertised_tools(
            tmp_path / "constructor-true.sqlite3",
            AGENT_TOOL_NAMES | OPERATOR_TOOL_NAMES,
            expose_operator_tools=True,
        )
    )


def test_environment_opt_in_advertises_all_core_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LM_EXPOSE_OPERATOR_TOOLS", "1")
    asyncio.run(
        _assert_advertised_tools(
            tmp_path / "environment-true.sqlite3",
            AGENT_TOOL_NAMES | OPERATOR_TOOL_NAMES,
        )
    )


def test_explicit_false_overrides_environment_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LM_EXPOSE_OPERATOR_TOOLS", "1")
    asyncio.run(
        _assert_advertised_tools(
            tmp_path / "constructor-false.sqlite3",
            AGENT_TOOL_NAMES,
            expose_operator_tools=False,
        )
    )


def test_attestation_exposure_is_independent_of_operator_exposure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LM_EXPOSE_ATTEST", "1")
    asyncio.run(
        _assert_advertised_tools(
            tmp_path / "attest-only.sqlite3",
            AGENT_TOOL_NAMES | {"memory_attest"},
            expose_operator_tools=False,
        )
    )

    monkeypatch.setenv("LM_EXPOSE_OPERATOR_TOOLS", "1")
    asyncio.run(
        _assert_advertised_tools(
            tmp_path / "operator-only.sqlite3",
            AGENT_TOOL_NAMES | OPERATOR_TOOL_NAMES,
            expose_attest=False,
        )
    )


def test_hidden_operator_tools_remain_resolvable_and_callable(
    tmp_path: Path,
) -> None:
    asyncio.run(_exercise_hidden_operator_tools(tmp_path / "hidden.sqlite3"))


async def _assert_advertised_tools(
    db_path: Path,
    expected: set[str],
    **server_kwargs: Any,
) -> None:
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    mcp = create_mcp_server(db_path, **server_kwargs)
    async with Client(mcp) as client:
        assert {tool.name for tool in await client.list_tools()} == expected


async def _exercise_hidden_operator_tools(db_path: Path) -> None:
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    mcp = create_mcp_server(db_path)
    async with Client(mcp) as client:
        assert {tool.name for tool in await client.list_tools()} == AGENT_TOOL_NAMES

        for name in OPERATOR_TOOL_NAMES:
            assert await mcp.get_tool(name) is not None

        first = _structured(
            await client.call_tool(
                "memory_remember",
                {
                    "content": "operator visibility first trace",
                    "context": {"scope": "project:visibility", "agent": "test-a"},
                },
            )
        )
        second = _structured(
            await client.call_tool(
                "memory_remember",
                {
                    "content": "operator visibility second trace",
                    "context": {"scope": "project:visibility", "agent": "test-b"},
                },
            )
        )
        first_id = first["node"]["id"]
        second_id = second["node"]["id"]

        connected = _structured(
            await client.call_tool(
                "memory_connect",
                {
                    "id_a": first_id,
                    "id_b": second_id,
                    "relation_type": "related",
                },
            )
        )
        assert connected["connection"]["type"] == "related"

        consolidated = _structured(
            await client.call_tool(
                "memory_consolidate",
                {"scope": "project:visibility", "force": True},
            )
        )
        assert "concepts_created" in consolidated

        status = _structured(
            await client.call_tool("memory_status", {"scope": "project:visibility"})
        )
        assert status["scope"] == "project:visibility"

        health = _structured(
            await client.call_tool(
                "memory_health",
                {"scope": "project:visibility", "top_stale": 0},
            )
        )
        assert health["scope"] == "project:visibility"

        forgotten = _structured(
            await client.call_tool(
                "memory_forget",
                {"id": second_id, "reason": "visibility test"},
            )
        )
        assert forgotten["node"]["decayed"] is True


def _structured(result: Any) -> dict[str, Any]:
    payload = getattr(result, "structured_content", None)
    assert isinstance(payload, dict)
    if set(payload) == {"result"} and isinstance(payload["result"], dict):
        return payload["result"]
    return payload
