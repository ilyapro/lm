"""One fact, one node: a consolidation pass over unchanged input creates nothing.

The live corpus showed schemas multiplying on every pass when procedure groups
share labels: a group keyed by one task_pattern carries, as its procedure_id,
the key of another group. These tests pin both the consolidation lookup and
the write-path rule that keeps one active node per scope, level and content.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from living_memory.consolidation import memory_consolidate
from living_memory.storage import MemoryStore

SCOPE = "project:ae"

# (task_pattern, procedure_id): each group's procedure_id is another group's
# task_pattern, the shape of the "dashboard goal api supervision" copies.
_GROUPS = (
    ("dashboard_goal_api_supervision", None),
    ("active_goal_supervision", "dashboard_goal_api_supervision"),
    ("healthy_active_progress", "active_goal_supervision"),
)


def _append_label_sharing_groups(store: MemoryStore) -> None:
    agents = ("agent-a", "agent-b", "agent-c")
    for group_index, (task_pattern, procedure_id) in enumerate(_GROUPS):
        for step in range(3):
            context = {
                "scope": SCOPE,
                "agent": agents[step],
                "task_pattern": task_pattern,
                "step_order": step + 1,
                "timestamp": f"2026-09-{10 + group_index:02d}T0{step}:00:00Z",
            }
            if procedure_id is not None:
                context["procedure_id"] = procedure_id
            store.append_trace(
                f"{task_pattern} step {step + 1}: check the goal api for group {group_index}",
                context,
                feedback={"confidence": 0.6, "usefulness_score": 0.4},
            )


def _active_contents(store: MemoryStore, level: str) -> dict[str, str]:
    return {
        node.id: node.content
        for node in store.list_nodes(level=level, scope=SCOPE, limit=100_000)
    }


def test_second_pass_over_the_same_scope_creates_no_nodes(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _append_label_sharing_groups(store)

        first = memory_consolidate(store, scope=SCOPE)
        assert len(first.schemas_created) == len(_GROUPS)
        schemas_before = _active_contents(store, "schema")
        concepts_before = _active_contents(store, "concept")

        second = memory_consolidate(store, scope=SCOPE)

        assert second.schemas_created == []
        assert second.concepts_created == []
        assert _active_contents(store, "schema") == schemas_before
        assert _active_contents(store, "concept") == concepts_before


def test_each_group_keeps_its_own_schema_across_passes(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _append_label_sharing_groups(store)

        memory_consolidate(store, scope=SCOPE)
        memory_consolidate(store, scope=SCOPE)
        memory_consolidate(store, scope=SCOPE)

        schemas = store.list_nodes(level="schema", scope=SCOPE)
        keys = sorted(schema.context["procedure_key"] for schema in schemas)
        assert keys == sorted(
            task_pattern.replace("_", " ") for task_pattern, _ in _GROUPS
        )
        for schema in schemas:
            group = schema.context["procedure_key"].replace(" ", "_")
            assert all(group in line for line in schema.content.splitlines()[1:])


@pytest.mark.parametrize("level", ["schema", "concept"])
def test_identical_content_in_same_scope_and_level_leaves_one_active_node(
    tmp_path: Path, level: str
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        content = f"Procedure: deploy rollback\n1. same {level} body"
        first = store.create_node(level=level, content=content, context={"scope": SCOPE})
        second = store.create_node(level=level, content=content, context={"scope": SCOPE})

        active = store.list_nodes(level=level, scope=SCOPE)
        assert [node.id for node in active] == [second.id]
        older = store.get_node(first.id)
        assert older is not None
        assert older.decayed is True
        assert older.decay_reason == "duplicate_content"
        edges = store.list_connections(source_id=second.id, relation_type="supersedes")
        assert [edge.target_id for edge in edges] == [first.id]
        assert edges[0].metadata.get("kind") == "duplicate_content"


def test_identical_content_across_scopes_or_levels_stays_separate(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        content = "Procedure: deploy rollback\n1. shared body"
        schema_a = store.create_node(level="schema", content=content, context={"scope": SCOPE})
        schema_b = store.create_node(
            level="schema", content=content, context={"scope": "project:other"}
        )
        concept = store.create_node(level="concept", content=content, context={"scope": SCOPE})

        for node in (schema_a, schema_b, concept):
            refreshed = store.get_node(node.id)
            assert refreshed is not None and refreshed.decayed is False
