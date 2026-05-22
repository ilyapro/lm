#!/usr/bin/env python3
"""Write-side micro-check for recall feedback linkage."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Path to write JSON evidence.")
    args = parser.parse_args()

    flows = {
        "explicit_same_context_bounded": _explicit_same_context_bounded(),
        "partial_same_agent_similarity": _partial_same_agent_similarity(),
        "unqualified_exact_scope_cap": _unqualified_exact_scope_cap(),
        "negative_isolation": _negative_isolation(),
    }
    applied_total = sum(int(flow["applied_count"]) for flow in flows.values())
    blocked_total = sum(int(flow["blocked_count"]) for flow in flows.values())
    max_write_ms = max(float(flow["write_ms"]) for flow in flows.values())
    evidence = {
        "generated_at": datetime.now(UTC).isoformat(),
        "db_policy": "temporary SQLite databases only; no live DB writes",
        "matcher_cap": 5,
        "flows": flows,
        "summary": {
            "applied_total": applied_total,
            "blocked_total": blocked_total,
            "max_write_ms": round(max_write_ms, 3),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence["summary"], sort_keys=True))
    return 0


def _explicit_same_context_bounded() -> dict[str, Any]:
    with _microcheck_server() as (mcp, store):
        context = {
            "scope": "project:micro-explicit",
            "agent": "codex",
            "task": "feedback-linkage",
            "session_id": "session-explicit",
        }
        mcp.tools["memory_remember"](
            "microcheck explicit feedback linkage migration recovery reference",
            context,
        )
        event_ids = [
            _recall_event(
                mcp,
                f"explicit feedback linkage migration recovery reference {index}",
                "project:micro-explicit",
                {
                    "agent": "codex",
                    "task": "feedback-linkage",
                    "session_id": "session-explicit",
                },
            )
            for index in range(6)
        ]

        started = time.perf_counter()
        response = mcp.tools["memory_remember"](
            "microcheck explicit feedback linkage migration recovery write",
            context,
        )
        write_ms = _elapsed_ms(started)
        applied_count = _applied_count(store, event_ids)
        _assert(applied_count == 5, "explicit same-context flow should apply the bounded cap")
        _assert(
            len(response["implicit_feedback"]["recall_event_ids"]) == 5,
            "explicit same-context response should report five applied recalls",
        )
        return {
            "seeded_recall_events": len(event_ids),
            "applied_count": applied_count,
            "blocked_count": len(event_ids) - applied_count,
            "write_ms": write_ms,
            "bounded_candidate_window": True,
        }


def _partial_same_agent_similarity() -> dict[str, Any]:
    with _microcheck_server() as (mcp, store):
        mcp.tools["memory_remember"](
            "microcheck inventory reconciliation checkpoint timeout retry reference",
            {"scope": "project:micro-partial", "agent": "codex"},
        )
        event_ids = [
            _recall_event(
                mcp,
                "inventory reconciliation checkpoint timeout",
                "project:micro-partial",
                {"agent": "codex"},
            ),
            _recall_event(
                mcp,
                "inventory reconciliation timeout retry",
                "project:micro-partial",
                {"agent": "codex"},
            ),
        ]

        started = time.perf_counter()
        response = mcp.tools["memory_remember"](
            "inventory reconciliation checkpoint timeout retry write",
            {"scope": "project:micro-partial", "agent": "codex"},
        )
        write_ms = _elapsed_ms(started)
        applied_count = _applied_count(store, event_ids)
        _assert(applied_count == 2, "partial same-agent similarity flow should apply both recalls")
        _assert(
            len(response["implicit_feedback"]["recall_event_ids"]) == 2,
            "partial same-agent response should report both recalls",
        )
        return {
            "seeded_recall_events": len(event_ids),
            "applied_count": applied_count,
            "blocked_count": len(event_ids) - applied_count,
            "write_ms": write_ms,
            "same_scope": True,
            "same_agent": True,
            "task_metadata_present": False,
        }


def _unqualified_exact_scope_cap() -> dict[str, Any]:
    with _microcheck_server() as (mcp, store):
        mcp.tools["memory_remember"](
            "microcheck cache warming retry timeout policy reference",
            {"scope": "project:micro-weak"},
        )
        event_ids = [
            _recall_event(mcp, "cache warming retry timeout", "project:micro-weak", None),
            _recall_event(mcp, "cache warming timeout policy", "project:micro-weak", None),
        ]

        started = time.perf_counter()
        response = mcp.tools["memory_remember"](
            "cache warming retry timeout policy write",
            {"scope": "project:micro-weak"},
        )
        write_ms = _elapsed_ms(started)
        applied_count = _applied_count(store, event_ids)
        _assert(applied_count == 1, "unqualified fallback should apply only one recall")
        _assert(
            len(response["implicit_feedback"]["recall_event_ids"]) == 1,
            "unqualified fallback response should report one recall",
        )
        return {
            "seeded_recall_events": len(event_ids),
            "applied_count": applied_count,
            "blocked_count": len(event_ids) - applied_count,
            "write_ms": write_ms,
            "weak_fallback_cap": 1,
        }


def _negative_isolation() -> dict[str, Any]:
    with _microcheck_server() as (mcp, store):
        mcp.tools["memory_remember"](
            "microcheck search index checkpoint rebuild timeout reference",
            {"scope": "project:micro-negative", "agent": "codex", "task": "task-a"},
        )
        task_event_id = _recall_event(
            mcp,
            "search index checkpoint rebuild timeout",
            "project:micro-negative",
            {"agent": "codex", "task": "task-a"},
        )
        mcp.tools["memory_remember"](
            "search index checkpoint rebuild timeout write",
            {"scope": "project:micro-negative", "agent": "codex", "task": "task-b"},
        )

        mcp.tools["memory_remember"](
            "microcheck orders cursor checkpoint timeout reference",
            {
                "scope": "project:micro-session",
                "agent": "codex",
                "task": "task-session",
                "session_id": "session-a",
            },
        )
        session_event_id = _recall_event(
            mcp,
            "orders cursor checkpoint timeout",
            "project:micro-session",
            {"agent": "codex", "task": "task-session", "session_id": "session-a"},
        )
        mcp.tools["memory_remember"](
            "orders cursor checkpoint timeout write",
            {
                "scope": "project:micro-session",
                "agent": "codex",
                "task": "task-session",
                "session_id": "session-b",
            },
        )

        mcp.tools["memory_remember"](
            "microcheck export checksum manifest replay reference",
            {
                "scope": "project:micro-project",
                "agent": "codex",
                "task": "project-global",
                "session_id": "session-global",
            },
        )
        project_event_id = _recall_event(
            mcp,
            "export checksum manifest replay",
            "project:micro-project",
            {"agent": "codex", "task": "project-global", "session_id": "session-global"},
        )
        started = time.perf_counter()
        global_response = mcp.tools["memory_remember"](
            "export checksum manifest replay global write",
            {
                "scope": "global",
                "agent": "codex",
                "task": "project-global",
                "session_id": "session-global",
            },
        )
        write_ms = _elapsed_ms(started)

        event_ids = [task_event_id, session_event_id, project_event_id]
        applied_count = _applied_count(store, event_ids)
        _assert(applied_count == 0, "negative isolation flow should apply no recalls")
        _assert(
            global_response["implicit_feedback"]["recall_event_ids"] == [],
            "project-to-global write should not report implicit feedback",
        )
        return {
            "seeded_recall_events": len(event_ids),
            "applied_count": applied_count,
            "blocked_count": len(event_ids),
            "write_ms": write_ms,
            "different_task_blocked": True,
            "different_session_blocked": True,
            "project_to_global_blocked": True,
        }


class _microcheck_server:
    def __enter__(self) -> tuple[Any, MemoryStore]:
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(self._tmpdir.name) / "memory.sqlite3"
        self.mcp = create_mcp_server(db_path, mcp_factory=FakeMCP)
        return self.mcp, self.mcp.memory_store

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.mcp.memory_store.close()
        self._tmpdir.cleanup()


def _recall_event(
    mcp: Any,
    query: str,
    scope: str,
    ambient_context: dict[str, Any] | None,
) -> str:
    response = mcp.tools["memory_recall"](
        query,
        scope=scope,
        max_results=2,
        ambient_context=ambient_context,
    )
    return str(response["recall_event_id"])


def _applied_count(store: MemoryStore, event_ids: list[str]) -> int:
    return sum(1 for event_id in event_ids if store.get_recall_event(event_id).feedback_applied)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    raise SystemExit(main())
