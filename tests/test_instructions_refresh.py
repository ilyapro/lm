"""Initialization carries the protocol without another consumer's map history."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastmcp")

from fastmcp import Client

from living_memory.instructions_map import HEADING
from living_memory.server import _server_instructions, create_mcp_server
from living_memory.storage import MemoryStore

from test_transport_identity import FakeMCP


def _foreign_map(store: MemoryStore) -> None:
    store.record_recall_event(
        query="release source publishing",
        scope="global",
        ambient_context={"task": "release-work", "transport_session_id": "foreign"},
        recall_map={
            "clusters": [{
                "label": "release source publishing",
                "count": 2,
                "medoid": {"node_id": "foreign-medoid"},
                "ask_hint": "release source publishing",
            }],
            "pool": 2,
            "covered": 2,
        },
    )


@pytest.mark.parametrize("map_enabled", ["0", "1"])
def test_restart_initializes_without_foreign_persisted_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, map_enabled: str
) -> None:
    monkeypatch.setenv("LM_RECALL_MAP", map_enabled)
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store:
        _foreign_map(store)

    mcp = create_mcp_server(path)
    try:
        served = mcp._mcp_server.create_initialization_options().instructions
        assert served == _server_instructions(mcp.memory_store.config.default_scope)
        assert HEADING not in served
        assert "release source publishing" not in served
    finally:
        mcp.memory_store.close()


def test_tool_calls_leave_the_next_clients_bootstrap_static(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    expected = _server_instructions(store.config.default_scope)

    async def scenario() -> None:
        async with Client(mcp) as first:
            assert (await first.initialize()).instructions == expected
            await first.call_tool("memory_recall", {"query": "first work"})
            _foreign_map(store)
            await first.call_tool(
                "memory_remember",
                {"content": "The release archive contains its own package manifest."},
            )
            await first.call_tool("memory_recall", {"query": "release archive manifest"})

        async with Client(mcp) as second:
            assert (await second.initialize()).instructions == expected
        assert mcp._mcp_server.create_initialization_options().instructions == expected

    try:
        asyncio.run(scenario())
    finally:
        store.close()


def test_bootstrap_and_tools_do_not_read_unscoped_map_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("bootstrap has no current work to query")

    monkeypatch.setattr(MemoryStore, "recent_recall_map_history", forbidden)
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    try:
        _foreign_map(mcp.memory_store)
        recalled = mcp.tools["memory_recall"]("a current query with no residual")
        remembered = mcp.tools["memory_remember"]("A trace written by the current work.")
        assert recalled["count"] == 0
        assert remembered["node"]["id"]
        assert mcp.instructions == _server_instructions(mcp.memory_store.config.default_scope)
    finally:
        mcp.memory_store.close()
