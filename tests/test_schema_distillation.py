"""Tests for schema-level procedural distillation."""

from __future__ import annotations

from pathlib import Path

from living_memory.consolidation import memory_consolidate
from living_memory.prompts import retrieval_context_prompt
from living_memory.retrieval import memory_recall
from living_memory.storage import MemoryStore


def _append_procedure_traces(
    store: MemoryStore,
    *,
    scope: str,
    procedure_id: str,
    count: int = 3,
) -> list:
    agents = ("agent-a", "agent-b", "agent-c")
    contents = [
        "check migration is reversible before release",
        "run migration dry run on staging",
        "execute rollback only after dry run passes",
        "audit rollback log entries for warnings",
    ]
    traces = []
    for index in range(count):
        body = contents[index % len(contents)]
        traces.append(
            store.append_trace(
                f"[{procedure_id}] {body}",
                {
                    "scope": scope,
                    "agent": agents[index % len(agents)],
                    "task": f"step-{index + 1}",
                    "procedure_id": procedure_id,
                    "step_order": index + 1,
                    "timestamp": f"2026-05-{10 + index:02d}T09:00:00Z",
                },
                feedback={"confidence": 0.6, "usefulness_score": 0.4},
            )
        )
    return traces


def test_consolidation_materializes_schema_for_three_procedural_traces(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        traces = _append_procedure_traces(
            store, scope="project:alpha", procedure_id="deploy_rollback", count=3
        )

        result = memory_consolidate(store, scope="project:alpha")

        assert len(result.schemas_created) == 1
        schema = result.schemas_created[0]
        assert schema.level == "schema"
        assert schema.scope == "project:alpha"
        assert schema.content.startswith("Procedure: deploy rollback")
        assert schema.context["trigger"] == "deploy rollback"
        assert len(schema.context["procedure"]) >= 3
        assert sorted(schema.source_traces) == sorted(trace.id for trace in traces)
        assert schema.provenance["strategy"] == "procedural"


def test_consolidation_skips_schema_below_three_procedural_traces(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _append_procedure_traces(
            store, scope="project:alpha", procedure_id="deploy_rollback", count=2
        )

        result = memory_consolidate(store, scope="project:alpha")

        assert result.schemas_created == []
        assert store.list_nodes(level="schema", scope="project:alpha") == []


def test_consolidation_groups_task_pattern_by_normalized_trigger(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        patterns = ("Deploy Rollback", "deploy_rollback", "deploy-rollback")
        for index, pattern in enumerate(patterns):
            store.append_trace(
                f"rollback step {index + 1}",
                {
                    "scope": "project:alpha",
                    "agent": f"agent-{index}",
                    "task_pattern": pattern,
                    "step_order": index + 1,
                },
                feedback={"confidence": 0.6, "usefulness_score": 0.4},
            )

        result = memory_consolidate(store, scope="project:alpha")

        assert len(result.schemas_created) == 1
        schema = result.schemas_created[0]
        assert schema.level == "schema"
        assert schema.context["trigger"] == "deploy rollback"
        assert schema.context["procedure_key"] == "deploy rollback"
        assert schema.context["task_pattern"] in patterns
        assert len(schema.context["procedure"]) == 3


def test_nodes_table_has_no_new_columns(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        rows = store.connection.execute("PRAGMA table_info(nodes)").fetchall()

        expected = {
            "id",
            "level",
            "content",
            "content_fingerprint",
            "embedding",
            "scope",
            "agent",
            "task",
            "context",
            "timestamp",
            "decayed",
            "decay_reason",
            "access_count",
            "last_accessed",
            "usefulness_score",
            "confidence",
            "unique_agents",
            "temporal_hint",
            "source_traces",
            "corrections",
            "provenance",
            "created_at",
            "updated_at",
        }
        actual = {str(row["name"]) for row in rows}
        assert actual == expected


def test_recall_returns_schema_in_top_three_when_trigger_matches(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _append_procedure_traces(
            store, scope="project:alpha", procedure_id="deploy_rollback", count=3
        )
        memory_consolidate(store, scope="project:alpha")

        for index in range(20):
            store.append_trace(
                f"unrelated chatter {index} about caching and reports",
                {"scope": "project:alpha", "agent": "noise"},
            )

        results = memory_recall(
            store,
            "how to do deploy rollback",
            scope="project:alpha",
            max_results=5,
        )

        schemas = store.list_nodes(level="schema", scope="project:alpha")
        assert schemas
        schema_id = schemas[0].id
        top_three = [result.node.id for result in results[:3]]
        assert schema_id in top_three
        schema_position = top_three.index(schema_id)
        for connected_trace_id in schemas[0].source_traces:
            for position, result_id in enumerate(top_three):
                if result_id == connected_trace_id:
                    assert position > schema_position


def test_retrieval_context_prompt_renders_skills_section_when_trigger_matches(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _append_procedure_traces(
            store, scope="project:alpha", procedure_id="deploy_rollback", count=3
        )
        memory_consolidate(store, scope="project:alpha")

        block = retrieval_context_prompt(
            store,
            task="how to do deploy rollback",
            scope="project:alpha",
        )

        schemas = store.list_nodes(level="schema", scope="project:alpha")
        assert schemas
        assert "skills:" in block
        assert schemas[0].id in block
        assert 'trigger="deploy rollback"' in block
        assert block.index("skills:") > block.index("concepts:")


def test_retrieval_context_prompt_omits_skills_section_without_match(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.create_node(
            level="schema",
            content="Procedure: deploy rollback\n1. run dry run",
            context={
                "scope": "project:alpha",
                "trigger": "deploy rollback",
                "procedure": ["run dry run"],
            },
            stats={"confidence": 0.8, "unique_agents": 2},
            provenance={"source_traces": ["trace-a"], "strategy": "procedural"},
        )

        block = retrieval_context_prompt(
            store,
            task="ad hoc query unrelated to procedures",
            scope="project:alpha",
        )

        assert "skills:" not in block


def test_consolidation_is_idempotent_for_procedural_schemas(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        traces = _append_procedure_traces(
            store, scope="project:alpha", procedure_id="deploy_rollback", count=3
        )

        first = memory_consolidate(store, scope="project:alpha")
        second = memory_consolidate(store, scope="project:alpha")

        assert len(first.schemas_created) == 1
        assert second.schemas_created == []
        schemas = store.list_nodes(level="schema", scope="project:alpha")
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema.context["trigger"] == "deploy rollback"
        assert sorted(schema.source_traces) == sorted(trace.id for trace in traces)


def test_consolidation_separates_distinct_procedure_triggers(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _append_procedure_traces(
            store, scope="project:alpha", procedure_id="deploy_rollback", count=3
        )
        _append_procedure_traces(
            store, scope="project:alpha", procedure_id="cache_warm", count=3
        )

        result = memory_consolidate(store, scope="project:alpha")

        triggers = sorted(schema.context["trigger"] for schema in result.schemas_created)
        assert triggers == ["cache warm", "deploy rollback"]
        for schema in result.schemas_created:
            assert schema.context["procedure"]
            assert schema.provenance["strategy"] == "procedural"


def test_consolidation_groups_by_task_pattern_when_procedure_ids_vary(
    tmp_path: Path,
) -> None:
    """Live cross-tree-consolidator shape: shared task_pattern, varying procedure_ids.

    Traces written by cross-tree-consolidator include procedure_id
    (tree_decomposition_<classification>, varies per tree) and task_pattern
    (sha256-derived, constant per pattern). Grouping must follow task_pattern.
    """

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        procedure_ids = (
            "tree_decomposition_build",
            "tree_decomposition_lemma",
            "tree_decomposition_doc",
        )
        agents = ("agent-a", "agent-b", "agent-c")
        traces = []
        for index, procedure_id in enumerate(procedure_ids):
            traces.append(
                store.append_trace(
                    f"tree:goal-{index + 1} decomposed in cycle {index + 1}",
                    {
                        "scope": "project:ae",
                        "agent": agents[index],
                        "task": f"tree:goal-{index + 1}",
                        "procedure_id": procedure_id,
                        "task_pattern": "208b33b133e3cc61",
                        "step_order": index + 1,
                    },
                    feedback={"confidence": 0.6, "usefulness_score": 0.4},
                )
            )

        result = memory_consolidate(store, scope="project:ae")

        assert len(result.schemas_created) == 1, (
            "expected one schema grouped by task_pattern across varying procedure_ids, "
            f"got {result.schemas_created}"
        )
        schema = result.schemas_created[0]
        assert schema.level == "schema"
        assert schema.context["task_pattern"] == "208b33b133e3cc61"
        assert schema.context["procedure_key"] == "208b33b133e3cc61"
        assert schema.context["trigger"].startswith("tree decomposition ")
        assert sorted(schema.source_traces) == sorted(trace.id for trace in traces)
        assert schema.provenance["strategy"] == "procedural"
        assert schema.provenance["group_field"] == "task_pattern"


def test_consolidation_clusters_traces_with_only_task_pattern(
    tmp_path: Path,
) -> None:
    """Three traces with task_pattern but no procedure_id must still materialize a schema."""

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        agents = ("agent-a", "agent-b", "agent-c")
        for index in range(3):
            store.append_trace(
                f"rollback step {index + 1}",
                {
                    "scope": "project:alpha",
                    "agent": agents[index],
                    "task_pattern": "208b33b133e3cc61",
                    "step_order": index + 1,
                },
                feedback={"confidence": 0.6, "usefulness_score": 0.4},
            )

        result = memory_consolidate(store, scope="project:alpha")

        assert len(result.schemas_created) == 1
        schema = result.schemas_created[0]
        assert schema.context["task_pattern"] == "208b33b133e3cc61"
        assert schema.context["procedure_key"] == "208b33b133e3cc61"
        assert schema.context["trigger"] == "208b33b133e3cc61"
        assert schema.provenance["group_field"] == "task_pattern"


def test_consolidation_prefers_procedure_id_for_trigger_when_grouping_by_task_pattern(
    tmp_path: Path,
) -> None:
    """When traces share task_pattern, the schema trigger should come from procedure_id (readable)."""

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        # Two traces with procedure_id="tree_decomposition_build" and one with a different
        # procedure_id; trigger should pick the most common procedure_id, not the hash.
        sequence = (
            "tree_decomposition_build",
            "tree_decomposition_build",
            "tree_decomposition_lemma",
        )
        agents = ("agent-a", "agent-b", "agent-c")
        for index, procedure_id in enumerate(sequence):
            store.append_trace(
                f"tree:goal-{index + 1} decomposed",
                {
                    "scope": "project:ae",
                    "agent": agents[index],
                    "task": f"tree:goal-{index + 1}",
                    "procedure_id": procedure_id,
                    "task_pattern": "abc123def456",
                    "step_order": index + 1,
                },
                feedback={"confidence": 0.6, "usefulness_score": 0.4},
            )

        result = memory_consolidate(store, scope="project:ae")

        assert len(result.schemas_created) == 1
        schema = result.schemas_created[0]
        assert schema.context["trigger"] == "tree decomposition build"
        assert schema.context["procedure_key"] == "abc123def456"
        assert schema.context["task_pattern"] == "abc123def456"
        assert schema.provenance["procedure_field"] == "procedure_id"
        assert schema.provenance["group_field"] == "task_pattern"


def test_consolidation_is_idempotent_when_grouping_by_task_pattern(
    tmp_path: Path,
) -> None:
    """Running consolidation twice on the same task_pattern group must not duplicate the schema."""

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        agents = ("agent-a", "agent-b", "agent-c")
        procedure_ids = (
            "tree_decomposition_build",
            "tree_decomposition_lemma",
            "tree_decomposition_doc",
        )
        for index, procedure_id in enumerate(procedure_ids):
            store.append_trace(
                f"tree step {index + 1}",
                {
                    "scope": "project:ae",
                    "agent": agents[index],
                    "procedure_id": procedure_id,
                    "task_pattern": "208b33b133e3cc61",
                    "step_order": index + 1,
                },
                feedback={"confidence": 0.6, "usefulness_score": 0.4},
            )

        first = memory_consolidate(store, scope="project:ae")
        second = memory_consolidate(store, scope="project:ae")

        assert len(first.schemas_created) == 1
        assert second.schemas_created == []
        schemas = store.list_nodes(level="schema", scope="project:ae")
        assert len(schemas) == 1
