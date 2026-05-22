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


def _apply_public_bm25_feedback_pressure(
    mcp: FakeMCP,
    *,
    scope: str,
    query: str,
    context: dict[str, str],
    ambient: dict[str, str],
    note_prefix: str,
) -> None:
    for cycle in range(4):
        for _index in range(5):
            recalled = mcp.tools["memory_recall"](
                query,
                scope=scope,
                max_results=1,
                ambient_context=ambient,
            )
            assert recalled["count"] == 1
            assert recalled["results"][0]["bm25_score"] == pytest.approx(1.0)
            assert "vector" in recalled["results"][0]["methods"]

        remembered = mcp.tools["memory_remember"](
            f"{note_prefix} {cycle} requires schema restore before retry",
            context,
        )

        # This is the 06f6422-style amplification path: one remember consumes
        # five compatible pending recall events and reinforces their top result.
        assert len(remembered["implicit_feedback"]["recall_event_ids"]) == 5


def test_implicit_feedback_amplification_cannot_disable_vector_recall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LM_RETRIEVAL_TUNING_POLICY", raising=False)
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store: MemoryStore = mcp.memory_store
    context = {
        "scope": "project:collapse",
        "agent": "agent-a",
        "task": "feedback-collapse",
        "session_id": "session-1",
    }
    ambient = {
        "agent": "agent-a",
        "task": "feedback-collapse",
        "session_id": "session-1",
    }

    mcp.tools["memory_remember"](
        "checkout deploy migration 42 missing caused payment checkout failure",
        context,
    )
    before = store.get_retrieval_weights("project:collapse").normalized()
    assert before.vector == pytest.approx(0.3)

    _apply_public_bm25_feedback_pressure(
        mcp,
        scope="project:collapse",
        query="checkout deploy migration 42 failure",
        context=context,
        ambient=ambient,
        note_prefix="checkout deploy migration 42 failure recovery note",
    )

    lexical = mcp.tools["memory_recall"](
        "checkout deploy migration 42 failure",
        scope="project:collapse",
        max_results=1,
        ambient_context=ambient,
    )
    assert lexical["count"] == 1
    assert lexical["results"][0]["bm25_score"] == pytest.approx(1.0)

    after = store.get_retrieval_weights("project:collapse").normalized()
    assert after.bm25 <= 0.85
    assert after.vector >= 0.15


def test_implicit_feedback_amplification_cannot_disable_eligible_graph_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LM_RETRIEVAL_TUNING_POLICY", raising=False)
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store: MemoryStore = mcp.memory_store
    context = {
        "scope": "global",
        "agent": "agent-a",
        "task": "feedback-collapse",
        "session_id": "session-1",
    }
    ambient = {
        "agent": "agent-a",
        "task": "feedback-collapse",
        "session_id": "session-1",
    }

    root = mcp.tools["memory_remember"](
        "incident rollback 314 database lock caused checkout outage",
        context,
    )
    dependent = mcp.tools["memory_remember"](
        "incident rollback 314 downstream cache drain required",
        context,
    )
    mcp.tools["memory_connect"](
        root["node"]["id"],
        dependent["node"]["id"],
        "caused",
    )

    before = store.get_retrieval_weights("global").normalized()
    assert before.vector == pytest.approx(0.4)
    assert before.graph == pytest.approx(0.2)

    _apply_public_bm25_feedback_pressure(
        mcp,
        scope="global",
        query="incident rollback 314 database lock checkout outage",
        context=context,
        ambient=ambient,
        note_prefix="incident rollback 314 database lock recovery note",
    )

    lexical = mcp.tools["memory_recall"](
        "incident rollback 314 database lock checkout outage",
        scope="global",
        max_results=1,
        ambient_context=ambient,
    )
    assert lexical["count"] == 1
    assert lexical["results"][0]["bm25_score"] == pytest.approx(1.0)

    after = store.get_retrieval_weights("global").normalized()
    assert after.bm25 <= 0.75
    assert after.vector >= 0.20
    assert after.graph >= 0.05
