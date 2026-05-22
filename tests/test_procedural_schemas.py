"""Tests for schema-level procedural distillation (P1..P5)."""

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
    """Append `count` procedural traces sharing the same procedure_id."""

    agents = ("agent-a", "agent-b", "agent-c")
    contents = [
        "check migration is reversible before release",
        "run migration dry run on staging",
        "execute rollback only after dry run passes",
        "audit rollback log entries for warnings",
    ]
    traces = []
    for index in range(count):
        # Tag the content with the procedure id so two procedure runs in the
        # same scope produce distinct content (the v3 write path dedupes
        # byte-identical content per (level, scope)).
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
    """P1: ≥3 procedural traces produce a schema node with trigger and procedure."""

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        traces = _append_procedure_traces(
            store, scope="project:alpha", procedure_id="deploy_rollback", count=3
        )

        result = memory_consolidate(store, scope="project:alpha")

        assert len(result.schemas_created) == 1
        schema = result.schemas_created[0]
        assert schema.level == "schema"
        assert schema.scope == "project:alpha"
        trigger = schema.context.get("trigger") or ""
        procedure = schema.context.get("procedure") or []
        assert trigger
        assert "deploy" in trigger
        assert "rollback" in trigger
        assert len(procedure) >= 3
        for step in procedure:
            assert isinstance(step, str)
            assert step.strip()
        assert sorted(schema.source_traces) == sorted(trace.id for trace in traces)
        assert schema.provenance.get("strategy") == "procedural"


def test_consolidation_skips_schema_below_three_procedural_traces(
    tmp_path: Path,
) -> None:
    """Only ≥3 traces share a procedure_id ⇒ no schema yet."""

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _append_procedure_traces(
            store, scope="project:alpha", procedure_id="deploy_rollback", count=2
        )

        result = memory_consolidate(store, scope="project:alpha")

        assert result.schemas_created == []
        assert store.list_nodes(level="schema", scope="project:alpha") == []


def test_nodes_table_has_no_new_columns(tmp_path: Path) -> None:
    """P2: schema persistence adds no new columns beyond the documented set.

    Schema v3 adds ``content_fingerprint`` to support write-time dedup of
    byte-identical traces per ``(level, scope)``. Other columns remain
    untouched.
    """

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
    """P3: recall matching a trigger surfaces schema in top-3 results."""

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

        assert results
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
    """P4: prompt includes a `skills:` section with matched schemas."""

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

        assert "skills:" in block
        schemas = store.list_nodes(level="schema", scope="project:alpha")
        assert schemas
        schema = schemas[0]
        assert schema.id in block
        assert 'trigger="deploy rollback"' in block
        skills_index = block.index("skills:")
        concepts_index = block.index("concepts:")
        assert skills_index > concepts_index


def test_retrieval_context_prompt_omits_skills_section_without_match(
    tmp_path: Path,
) -> None:
    """P5: no procedural matches ⇒ no `skills:` heading at all."""

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.append_trace(
            "random thought about caching reports",
            {"scope": "project:alpha", "agent": "agent-a"},
        )
        store.append_trace(
            "another unrelated note",
            {"scope": "project:alpha", "agent": "agent-b"},
        )

        block = retrieval_context_prompt(
            store,
            task="ad hoc query unrelated to procedures",
            scope="project:alpha",
        )

        assert "skills:" not in block
