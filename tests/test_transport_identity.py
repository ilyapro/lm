"""Transport-session identity stamping at the MCP tool boundary.

These tests run the real FastMCP in-memory client (``fastmcp.Client``)
against ``create_mcp_server`` so the ambient request context — the source of
``Context.session_id`` — is genuine. They pin the contract of the reserved
``transport_session_id`` key: derived server-side per client connection and
stamped into recall ambient_context and remember/teach trace context, explicit
caller values win, scope resolution stays byte-identical, and everything
degrades to unstamped dicts when no request context exists (fake factories,
direct tool-function calls).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastmcp")

from fastmcp import Client

from living_memory import server as server_module
from living_memory.retrieval import MemoryRecallService
from living_memory.server import _with_transport_identity, create_mcp_server


def _structured(result: Any) -> dict[str, Any]:
    payload = getattr(result, "structured_content", None)
    assert isinstance(payload, dict)
    if set(payload) == {"result"} and isinstance(payload["result"], dict):
        return payload["result"]
    return payload


def _server(tmp_path: Path) -> tuple[Any, Any]:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    return mcp, mcp.memory_store


async def _recall_stamp(client: Client, store: Any, query: str) -> Any:
    await client.call_tool("memory_recall", {"query": query})
    return store.list_recall_events()[0].ambient_context.get("transport_session_id")


def test_recall_and_remember_stamped_with_one_session_id(tmp_path: Path) -> None:
    async def scenario() -> None:
        mcp, store = _server(tmp_path)
        async with Client(mcp) as client:
            await client.call_tool("memory_recall", {"query": "transport stamping smoke"})
            remembered = _structured(
                await client.call_tool(
                    "memory_remember", {"content": "transport stamping smoke trace"}
                )
            )
        events = store.list_recall_events()
        assert len(events) == 1
        stamp = events[0].ambient_context.get("transport_session_id")
        assert isinstance(stamp, str) and stamp
        trace = store.get_node(remembered["node"]["id"])
        assert trace.context["transport_session_id"] == stamp

    asyncio.run(scenario())


def test_transport_id_stable_within_connection_distinct_across(tmp_path: Path) -> None:
    async def scenario() -> None:
        mcp, store = _server(tmp_path)
        async with Client(mcp) as client:
            first = await _recall_stamp(client, store, "stability probe one")
            second = await _recall_stamp(client, store, "stability probe two")
        async with Client(mcp) as client:
            third = await _recall_stamp(client, store, "stability probe three")
        assert first and second and third
        assert first == second
        assert third != first

    asyncio.run(scenario())


def test_explicit_transport_session_id_survives_verbatim(tmp_path: Path) -> None:
    async def scenario() -> None:
        mcp, store = _server(tmp_path)
        async with Client(mcp) as client:
            await client.call_tool(
                "memory_recall",
                {
                    "query": "override probe",
                    "ambient_context": {"transport_session_id": "explicit-x"},
                },
            )
            remembered = _structured(
                await client.call_tool(
                    "memory_remember",
                    {
                        "content": "override probe trace",
                        "context": {"transport_session_id": "explicit-x"},
                    },
                )
            )
        event = store.list_recall_events()[0]
        assert event.ambient_context["transport_session_id"] == "explicit-x"
        trace = store.get_node(remembered["node"]["id"])
        assert trace.context["transport_session_id"] == "explicit-x"

    asyncio.run(scenario())


def test_scopeless_recall_scope_resolution_unchanged_by_stamping(tmp_path: Path) -> None:
    async def scenario() -> None:
        mcp, store = _server(tmp_path)

        query = "scope perturbation probe"
        async with Client(mcp) as client:
            await client.call_tool("memory_recall", {"query": query})
        stamped_event = store.list_recall_events()[0]
        assert stamped_event.ambient_context.get("transport_session_id")
        MemoryRecallService(store).memory_recall(query)
        direct_event = store.list_recall_events()[0]
        assert "transport_session_id" not in direct_event.ambient_context
        assert stamped_event.requested_scope == direct_event.requested_scope
        assert stamped_event.resolved_scopes == direct_event.resolved_scopes
        assert not stamped_event.requested_scope.startswith("session:")

        # A project named in the query narrows neither path: nothing is inferred.
        store.append_trace("seeded project fact", {"scope": "project:transportia"})
        project_query = "transportia scope probe"
        async with Client(mcp) as client:
            await client.call_tool("memory_recall", {"query": project_query})
        stamped_project = store.list_recall_events()[0]
        MemoryRecallService(store).memory_recall(project_query)
        direct_project = store.list_recall_events()[0]
        assert stamped_project.requested_scope == direct_project.requested_scope == "global"
        assert stamped_project.resolved_scopes == direct_project.resolved_scopes
        assert stamped_project.resolved_scopes[-1] == "*"

    asyncio.run(scenario())


def test_teach_without_context_stamps_corrective_trace(tmp_path: Path) -> None:
    async def scenario() -> None:
        mcp, store = _server(tmp_path)
        original = store.append_trace(
            "original fact to correct",
            {"scope": "project:teachprobe", "agent": "seed-agent"},
        )
        assert "transport_session_id" not in original.context
        async with Client(mcp) as client:
            session_stamp = await _recall_stamp(client, store, "teach stamp probe")
            taught = _structured(
                await client.call_tool(
                    "memory_teach",
                    {"trace_id": original.id, "correction": "corrected fact"},
                )
            )
        assert session_stamp
        corrective = store.get_node(taught["corrective_trace"]["id"])
        assert corrective.context["transport_session_id"] == session_stamp

    asyncio.run(scenario())


class FakeMCP:
    """Minimal factory matching tests/test_mcp_server.py — tools called directly."""

    def __init__(self, name: str, instructions: str | None = None) -> None:
        self.name = name
        self.instructions = instructions
        self.tools: dict[str, Any] = {}

    def tool(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.tools[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        if func is None:
            return decorate
        return decorate(func)

    def resource(self, uri: str, **_kwargs: Any) -> Any:
        def decorate(func: Any) -> Any:
            return func

        return decorate

    def prompt(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            return inner

        if func is None:
            return decorate
        return decorate(func)


def test_fake_factory_direct_calls_stay_unstamped(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store

    remembered = mcp.tools["memory_remember"](
        "degradation trace", {"scope": "project:degrade"}
    )
    assert "transport_session_id" not in store.get_node(remembered["node"]["id"]).context

    bare = mcp.tools["memory_remember"]("degradation trace without context")
    assert "transport_session_id" not in store.get_node(bare["node"]["id"]).context

    recalled = mcp.tools["memory_recall"]("degradation trace")
    assert recalled["count"] >= 0
    assert "transport_session_id" not in store.list_recall_events()[0].ambient_context

    taught = mcp.tools["memory_teach"](remembered["node"]["id"], "degradation corrected")
    corrective = store.get_node(taught["corrective_trace"]["id"])
    assert "transport_session_id" not in corrective.context


def test_with_transport_identity_outside_request_returns_input_unchanged() -> None:
    assert _with_transport_identity(None) is None
    payload = {"agent": "a"}
    assert _with_transport_identity(payload) is payload


def test_with_transport_identity_copies_and_respects_explicit_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server_module, "_transport_session_id", lambda: "derived-id")

    assert server_module._with_transport_identity(None) == {
        "transport_session_id": "derived-id"
    }

    original = {"agent": "a"}
    stamped = server_module._with_transport_identity(original)
    assert stamped == {"agent": "a", "transport_session_id": "derived-id"}
    assert original == {"agent": "a"}

    explicit = {"transport_session_id": "explicit-x"}
    assert server_module._with_transport_identity(explicit) == {
        "transport_session_id": "explicit-x"
    }
