from pathlib import Path

import pytest

from living_memory.retrieval import (
    MemoryRecallService,
    _parse_depth,
    memory_connect,
    memory_recall,
)
from living_memory.storage import MemoryStore


@pytest.mark.parametrize(
    "depth,expected_depth,expected_causal",
    [
        (None, 1, False),
        (0, 0, False),
        (1, 1, False),
        (2, 2, False),
        ("0", 0, False),
        ("1", 1, False),
        ("2", 2, False),
        ("none", 0, False),
        ("off", 0, False),
        ("shallow", 1, False),
        ("normal", 1, False),
        ("medium", 2, False),
        ("deep", 3, False),
        ("SHALLOW", 1, False),
        ("  Medium  ", 2, False),
        ("", 1, False),
        ("bogus", 1, False),
        ("causal", 2, True),
    ],
)
def test_parse_depth_accepts_string_aliases(
    depth: int | str | None, expected_depth: int, expected_causal: bool
) -> None:
    parsed_depth, causal_mode = _parse_depth(depth, "test query")
    assert parsed_depth == expected_depth
    assert causal_mode is expected_causal


def test_memory_recall_accepts_string_depth_aliases(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.append_trace("API deploy checklist", {"scope": "project:alpha"})
        for alias in ("shallow", "normal", "medium", "deep"):
            results = memory_recall(
                store,
                "API deploy checklist",
                scope="project:alpha",
                depth=alias,
                max_results=5,
            )
            assert results, f"recall returned no results for depth={alias!r}"


def test_causal_recall_returns_causes_for_why_queries(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        cause = store.append_trace(
            "Missing migration file caused checkout deploy incident",
            {"scope": "project:alpha", "agent": "agent-a"},
        )
        effect = store.append_trace(
            "Checkout deploy incident happened during release",
            {"scope": "project:alpha", "agent": "agent-b"},
        )
        memory_connect(store, cause.id, effect.id, "caused", weight=1.0)

        results = memory_recall(
            store,
            "why did checkout deploy incident happen",
            scope="project:alpha",
            depth="causal",
            max_results=5,
        )

        assert results[0].node.id == cause.id
        assert results[0].graph_score > 0
        assert cause.id in results[0].path


def test_graph_traverses_supported_edge_types(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        start = store.append_trace("API deploy checklist", {"scope": "project:alpha"})
        related = store.append_trace("API deploy runbook", {"scope": "project:alpha"})
        requirement = store.append_trace("API deploy requires feature flag", {"scope": "project:alpha"})
        contradiction = store.append_trace("API deploy should skip feature flag", {"scope": "project:alpha"})

        service = MemoryRecallService(store)
        service.memory_connect(start.id, related.id, "related")
        service.memory_connect(start.id, requirement.id, "requires")
        service.memory_connect(start.id, contradiction.id, "contradicts")

        results = service.memory_recall(
            "API deploy checklist",
            scope="project:alpha",
            depth=1,
            max_results=10,
        )
        ids = {result.node.id for result in results}

        assert {related.id, requirement.id, contradiction.id}.issubset(ids)


def test_superseding_correction_ranks_above_original(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        original = store.append_trace(
            "The deploy rollback command is lm rollback old",
            {"scope": "project:alpha", "agent": "agent-a"},
            feedback={"confidence": 0.4, "usefulness_score": -0.6},
        )
        corrected = store.append_trace(
            "The deploy rollback command is lm rollback current",
            {"scope": "project:alpha", "agent": "agent-b"},
            feedback={"confidence": 0.8, "unique_agents": 2, "usefulness_score": 0.8},
        )
        memory_connect(store, corrected.id, original.id, "supersedes", weight=1.0)

        results = memory_recall(
            store,
            "deploy rollback command old",
            scope="project:alpha",
            depth=1,
            max_results=3,
        )

        assert results[0].node.id == corrected.id
        assert original.id in {result.node.id for result in results}
