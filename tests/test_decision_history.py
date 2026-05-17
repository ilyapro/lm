from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from living_memory.prompts import retrieval_context_prompt
from living_memory.retrieval import _parse_depth, memory_connect, memory_recall
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


def test_memory_remember_returns_rejected_alternative_ids_only_when_supplied(tmp_path: Path) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)

    baseline = mcp.tools["memory_remember"](
        "Used /api/goals/create after /api/switch",
        {"scope": "project:ae", "task": "add goal"},
    )
    assert "rejected_alternatives" not in baseline

    remembered = mcp.tools["memory_remember"](
        "Used /api/goals/create after /api/switch",
        {"scope": "project:ae", "task": "add goal"},
        alternatives_considered=[
            {
                "approach": "hand-write files under projects/<name>/state/goals/",
                "rejected_because": "misses canonical marker; goals_list() ignores",
            },
            {
                "approach": "use goal_create_from_ticket without JIRA env",
                "rejected_because": "requires JIRA_BASE_URL+JIRA_API_TOKEN, not set",
            },
        ],
    )

    assert remembered["node"]["content"] == "Used /api/goals/create after /api/switch"
    assert len(remembered["rejected_alternatives"]) == 2
    assert all(remembered["rejected_alternatives"])


def test_rejected_alternatives_create_contradicts_edges_and_negative_trace_metadata(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        primary, rejected = store.append_trace_with_rejected_alternatives(
            "Used /api/goals/create after /api/switch",
            {"scope": "project:ae", "task": "add goal", "agent": "codex"},
            feedback={"confidence": 0.8, "unique_agents": 2, "usefulness_score": 0.4},
            alternatives_considered=[
                {
                    "approach": "hand-write files under projects/<name>/state/goals/",
                    "rejected_because": "misses canonical marker; goals_list() ignores",
                },
                {
                    "approach": "use goal_create_from_ticket without JIRA env",
                    "rejected_because": "requires JIRA_BASE_URL+JIRA_API_TOKEN, not set",
                },
            ],
        )

        assert len(rejected) == 2
        for alternative in rejected:
            assert alternative.scope == primary.scope
            assert alternative.task == primary.task
            assert alternative.context["is_rejected_alternative"] is True
            assert alternative.confidence <= primary.confidence * 0.5

        rows = store.connection.execute(
            """
            SELECT *
            FROM connections
            WHERE target_id = ? AND type = 'contradicts'
            """,
            (primary.id,),
        ).fetchall()

        assert len(rows) == 2
        assert {str(row["source_id"]) for row in rows} == {node.id for node in rejected}
        for row in rows:
            metadata = json.loads(row["metadata"])
            assert metadata["kind"] == "rejected_alternative"
            assert metadata["reason"].strip()


def test_rejected_alternative_ingest_rejects_invalid_batch_without_partial_trace(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        with pytest.raises(ValueError, match="rejected_because"):
            store.append_trace_with_rejected_alternatives(
                "Used /api/goals/create after /api/switch",
                {"scope": "project:ae", "task": "add goal"},
                alternatives_considered=[
                    {
                        "approach": "hand-write files under projects/<name>/state/goals/",
                        "rejected_because": "",
                    }
                ],
            )

        assert store.trace_count() == 0


def test_decision_depth_returns_primary_and_rejected_alternatives_only_for_decision_mode(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        primary, rejected = store.append_trace_with_rejected_alternatives(
            "Used /api/goals/create after /api/switch",
            {"scope": "project:ae", "task": "add goal"},
            alternatives_considered=[
                {
                    "approach": "hand-write files under projects/<name>/state/goals/",
                    "rejected_because": "misses canonical marker; goals_list() ignores",
                }
            ],
        )

        shallow_ids = {
            result.node.id
            for result in memory_recall(
                store,
                "Used /api/goals/create after /api/switch",
                scope="project:ae",
                depth="shallow",
                max_results=10,
            )
        }
        numeric_ids = {
            result.node.id
            for result in memory_recall(
                store,
                "Used /api/goals/create after /api/switch",
                scope="project:ae",
                depth=1,
                max_results=10,
            )
        }
        decision_ids = {
            result.node.id
            for result in memory_recall(
                store,
                "Used /api/goals/create after /api/switch",
                scope="project:ae",
                depth="decision",
                max_results=10,
            )
        }

        assert rejected[0].id not in shallow_ids
        assert rejected[0].id not in numeric_ids
        assert {primary.id, rejected[0].id}.issubset(decision_ids)
        assert _parse_depth("decision", "test query") == (1, False)


def test_causal_recall_does_not_traverse_rejected_alternative_contradictions(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        cause = store.append_trace("Missing migration caused deploy incident", {"scope": "project:ae"})
        effect, rejected = store.append_trace_with_rejected_alternatives(
            "Deploy incident happened during release",
            {"scope": "project:ae"},
            alternatives_considered=[
                {
                    "approach": "skip migration guard during release",
                    "rejected_because": "would miss the actual root cause",
                }
            ],
        )
        memory_connect(store, cause.id, effect.id, "caused")

        results = memory_recall(
            store,
            "why did deploy incident happen",
            scope="project:ae",
            depth="causal",
            max_results=10,
        )
        ids = {result.node.id for result in results}

        assert cause.id in ids
        assert effect.id in ids
        assert rejected[0].id not in ids


def test_retrieval_context_prompt_includes_decision_history_only_when_present(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.append_trace("Alpha deploy rollback requires migration dry run", {"scope": "project:alpha"})
        plain = retrieval_context_prompt(
            store,
            task="deploy rollback migration",
            scope="project:alpha",
            max_concepts=3,
        )
        assert "Alternatives rejected:" not in plain

        primary, _rejected = store.append_trace_with_rejected_alternatives(
            "Used /api/goals/create after /api/switch",
            {"scope": "project:ae", "task": "add goal"},
            alternatives_considered=[
                {
                    "approach": "hand-write files under projects/<name>/state/goals/",
                    "rejected_because": "misses canonical marker; goals_list() ignores",
                },
                {
                    "approach": "use goal_create_from_ticket without JIRA env",
                    "rejected_because": "requires JIRA_BASE_URL+JIRA_API_TOKEN, not set",
                },
            ],
        )

        block = retrieval_context_prompt(
            store,
            task="hand-write goals files",
            scope="project:ae",
            max_concepts=3,
        )

        assert primary.id in block
        assert "Alternatives rejected: 2" in block
        assert "misses canonical marker" in block
        assert "requires JIRA_BASE_URL+JIRA_API_TOKEN" in block
