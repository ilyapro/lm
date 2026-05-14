from __future__ import annotations

from pathlib import Path
from typing import Any

from living_memory.retrieval import MemoryRecallService
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


def test_recall_persists_event_with_result_provenance(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        trace = store.append_trace(
            "deploy incident was fixed by running migration 42",
            {"scope": "project:alpha", "agent": "agent-a"},
        )

        results = MemoryRecallService(store).memory_recall(
            "deploy migration incident",
            scope="project:alpha",
            max_results=3,
        )

        events = store.list_recall_events()
        assert len(events) == 1
        assert events[0].query == "deploy migration incident"
        assert events[0].scope == "project:alpha"
        assert trace.id in events[0].result_ids
        assert results[0].recall_event_id == events[0].id


def test_empty_recall_persists_event_without_creating_memory_trace(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        assert MemoryRecallService(store).memory_recall("nothing yet") == []

        events = store.list_recall_events()
        assert len(events) == 1
        assert events[0].result_ids == []
        assert store.trace_count() == 0


def test_mcp_remember_consumes_prior_recall_as_implicit_feedback(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store: MemoryStore = mcp.memory_store

    first = mcp.tools["memory_remember"](
        "checkout deploy failed because migration 42 was missing",
        {"scope": "project:alpha", "agent": "agent-a", "session_id": "s1"},
    )
    first_id = first["node"]["id"]

    recalled = mcp.tools["memory_recall"](
        "checkout deploy migration failure",
        scope="project:alpha",
        max_results=2,
        ambient_context={"agent": "agent-a", "session_id": "s1"},
    )
    event_id = recalled["recall_event_id"]
    before = store.get_node(first_id).usefulness_score

    remembered = mcp.tools["memory_remember"](
        "checkout deploy recovery requires restoring migration 42 before retry",
        {"scope": "project:alpha", "agent": "agent-a", "session_id": "s1"},
    )
    new_id = remembered["node"]["id"]
    new_trace = store.get_node(new_id)
    event = store.get_recall_event(event_id)
    connections = store.list_connections(source_id=new_id, relation_type="related")

    assert event.feedback_applied is True
    assert event.feedback_trace_id == new_id
    assert new_trace.provenance["prior_recalls"][0]["id"] == event_id
    assert first_id in new_trace.provenance["recalled_nodes"]
    assert first_id in new_trace.source_traces
    assert first_id in {connection.target_id for connection in connections}
    assert store.get_node(first_id).usefulness_score > before
    assert remembered["implicit_feedback"]["recall_event_ids"] == [event_id]


def test_mcp_teach_links_correction_to_prior_recall_without_reinforcing_original(
    tmp_path: Path,
) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store: MemoryStore = mcp.memory_store

    original = mcp.tools["memory_remember"](
        "payment retry limit is 3 attempts",
        {"scope": "project:billing", "agent": "agent-a"},
        {"confidence": 0.5, "usefulness_score": 0.0},
    )
    original_id = original["node"]["id"]
    recalled = mcp.tools["memory_recall"](
        "payment retry limit attempts",
        scope="project:billing",
        max_results=2,
    )
    event_id = recalled["recall_event_id"]

    taught = mcp.tools["memory_teach"](
        original_id,
        "payment retry limit is 5 attempts",
        context={"agent": "agent-b"},
    )
    corrective_id = taught["corrective_trace"]["id"]
    corrective = store.get_node(corrective_id)
    event = store.get_recall_event(event_id)
    original_after = store.get_node(original_id)

    assert event.feedback_trace_id == corrective_id
    assert corrective.provenance["prior_recalls"][0]["id"] == event_id
    assert original_id in corrective.provenance["recalled_nodes"]
    assert original_after.usefulness_score <= -0.25
    assert taught["implicit_feedback"]["recall_event_ids"] == [event_id]
