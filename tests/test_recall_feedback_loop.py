from __future__ import annotations

from pathlib import Path
from typing import Any

import living_memory.storage as storage_module
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


def test_mcp_remember_consumes_multiple_same_context_recalls(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store: MemoryStore = mcp.memory_store
    context = {
        "scope": "project:alpha",
        "agent": "agent-a",
        "task": "deploy-42",
        "session_id": "session-1",
    }

    mcp.tools["memory_remember"](
        "checkout deploy migration 42 missing caused payment checkout failure",
        context,
    )
    mcp.tools["memory_remember"](
        "checkout deploy migration 42 recovery requires restoring the schema",
        context,
    )
    first_recall = mcp.tools["memory_recall"](
        "checkout deploy migration failure",
        scope="project:alpha",
        max_results=3,
        ambient_context={
            "agent": "agent-a",
            "task": "deploy-42",
            "session_id": "session-1",
        },
    )
    second_recall = mcp.tools["memory_recall"](
        "checkout deploy migration recovery schema",
        scope="project:alpha",
        max_results=3,
        ambient_context={
            "agent": "agent-a",
            "task": "deploy-42",
            "session_id": "session-1",
        },
    )
    event_ids = [first_recall["recall_event_id"], second_recall["recall_event_id"]]

    remembered = mcp.tools["memory_remember"](
        "checkout deploy migration 42 failure recovery requires schema restore before retry",
        context,
    )
    new_id = remembered["node"]["id"]

    assert set(remembered["implicit_feedback"]["recall_event_ids"]) == set(event_ids)
    assert all(store.get_recall_event(event_id).feedback_applied for event_id in event_ids)
    assert {store.get_recall_event(event_id).feedback_trace_id for event_id in event_ids} == {new_id}


def test_mcp_remember_links_partial_metadata_with_same_agent_and_high_similarity(
    tmp_path: Path,
) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store: MemoryStore = mcp.memory_store

    mcp.tools["memory_remember"](
        "inventory reconciliation timeout retry uses batch checkpoint resume",
        {"scope": "project:inventory", "agent": "agent-a"},
    )
    first_recall = mcp.tools["memory_recall"](
        "inventory reconciliation timeout",
        scope="project:inventory",
        max_results=2,
        ambient_context={"agent": "agent-a"},
    )
    second_recall = mcp.tools["memory_recall"](
        "inventory reconciliation checkpoint retry",
        scope="project:inventory",
        max_results=2,
        ambient_context={"agent": "agent-a"},
    )
    event_ids = [first_recall["recall_event_id"], second_recall["recall_event_id"]]

    remembered = mcp.tools["memory_remember"](
        "inventory reconciliation timeout retry should resume from the checkpoint",
        {"scope": "project:inventory", "agent": "agent-a"},
    )

    assert set(remembered["implicit_feedback"]["recall_event_ids"]) == set(event_ids)
    assert all(store.get_recall_event(event_id).feedback_applied for event_id in event_ids)


def test_mcp_remember_caps_unqualified_exact_scope_fallback_at_one(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(storage_module, "_utc_now", lambda: "2026-05-22T09:00:00Z")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store: MemoryStore = mcp.memory_store

    mcp.tools["memory_remember"](
        "cache warming queue timeout retry policy for catalog refresh",
        {"scope": "project:catalog"},
    )
    first_recall = mcp.tools["memory_recall"](
        "cache warming queue timeout",
        scope="project:catalog",
        max_results=2,
    )
    second_recall = mcp.tools["memory_recall"](
        "cache warming retry policy",
        scope="project:catalog",
        max_results=2,
    )

    remembered = mcp.tools["memory_remember"](
        "cache warming queue timeout retry policy needs a catalog refresh backoff",
        {"scope": "project:catalog"},
    )
    applied = set(remembered["implicit_feedback"]["recall_event_ids"])

    assert applied == {second_recall["recall_event_id"]}
    assert store.get_recall_event(second_recall["recall_event_id"]).feedback_applied is True
    assert store.get_recall_event(first_recall["recall_event_id"]).feedback_applied is False


def test_mcp_teach_consumes_multiple_same_context_recalls_without_reinforcement(
    tmp_path: Path,
) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store: MemoryStore = mcp.memory_store
    context = {
        "scope": "project:billing",
        "agent": "agent-a",
        "task": "retry-policy",
        "session_id": "session-1",
    }

    original = mcp.tools["memory_remember"](
        "payment retry limit is 3 attempts for card declines",
        context,
        {"confidence": 0.5, "usefulness_score": 0.0},
    )
    original_id = original["node"]["id"]
    first_recall = mcp.tools["memory_recall"](
        "payment retry limit card declines",
        scope="project:billing",
        max_results=2,
        ambient_context={
            "agent": "agent-a",
            "task": "retry-policy",
            "session_id": "session-1",
        },
    )
    second_recall = mcp.tools["memory_recall"](
        "payment retry attempts card policy",
        scope="project:billing",
        max_results=2,
        ambient_context={
            "agent": "agent-a",
            "task": "retry-policy",
            "session_id": "session-1",
        },
    )
    event_ids = [first_recall["recall_event_id"], second_recall["recall_event_id"]]

    taught = mcp.tools["memory_teach"](
        original_id,
        "payment retry limit is 5 attempts for card declines",
        context={
            "agent": "agent-a",
            "task": "retry-policy",
            "session_id": "session-1",
        },
    )
    corrective_id = taught["corrective_trace"]["id"]
    original_after = store.get_node(original_id)

    assert set(taught["implicit_feedback"]["recall_event_ids"]) == set(event_ids)
    assert {store.get_recall_event(event_id).feedback_trace_id for event_id in event_ids} == {
        corrective_id
    }
    assert original_after.usefulness_score <= -0.25


def test_mcp_remember_does_not_link_different_task_or_session(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store: MemoryStore = mcp.memory_store

    mcp.tools["memory_remember"](
        "search index rebuild timeout fixed by replaying shard checkpoints",
        {"scope": "project:search", "agent": "agent-a", "task": "search-a"},
    )
    task_recall = mcp.tools["memory_recall"](
        "search index rebuild timeout",
        scope="project:search",
        max_results=2,
        ambient_context={"agent": "agent-a", "task": "search-a"},
    )
    task_event_id = task_recall["recall_event_id"]
    task_write = mcp.tools["memory_remember"](
        "search index rebuild timeout checkpoint replay procedure",
        {"scope": "project:search", "agent": "agent-a", "task": "search-b"},
    )

    mcp.tools["memory_remember"](
        "orders export cursor timeout fixed by checkpoint resume",
        {"scope": "project:orders", "agent": "agent-a", "task": "orders-a", "session_id": "s1"},
    )
    session_recall = mcp.tools["memory_recall"](
        "orders export cursor timeout",
        scope="project:orders",
        max_results=2,
        ambient_context={
            "agent": "agent-a",
            "task": "orders-a",
            "session_id": "s1",
        },
    )
    session_event_id = session_recall["recall_event_id"]
    session_write = mcp.tools["memory_remember"](
        "orders export cursor timeout checkpoint resume procedure",
        {"scope": "project:orders", "agent": "agent-a", "task": "orders-a", "session_id": "s2"},
    )

    assert task_write["implicit_feedback"]["recall_event_ids"] == []
    assert session_write["implicit_feedback"]["recall_event_ids"] == []
    assert store.get_recall_event(task_event_id).feedback_applied is False
    assert store.get_recall_event(session_event_id).feedback_applied is False


def test_mcp_remember_does_not_link_project_recall_to_global_write(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store: MemoryStore = mcp.memory_store

    mcp.tools["memory_remember"](
        "audit export checksum mismatch fixed by replaying manifest chunks",
        {
            "scope": "project:exports",
            "agent": "agent-a",
            "task": "export-audit",
            "session_id": "session-1",
        },
    )
    recalled = mcp.tools["memory_recall"](
        "audit export checksum mismatch manifest",
        scope="project:exports",
        max_results=2,
        ambient_context={
            "agent": "agent-a",
            "task": "export-audit",
            "session_id": "session-1",
        },
    )
    event_id = recalled["recall_event_id"]
    event = store.get_recall_event(event_id)
    assert "global" in event.resolved_scopes

    remembered = mcp.tools["memory_remember"](
        "audit export checksum mismatch manifest replay is generally useful",
        {
            "scope": "global",
            "agent": "agent-a",
            "task": "export-audit",
            "session_id": "session-1",
        },
    )

    assert remembered["implicit_feedback"]["recall_event_ids"] == []
    assert store.get_recall_event(event_id).feedback_applied is False
