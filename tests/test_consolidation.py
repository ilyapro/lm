from pathlib import Path

import pytest

from living_memory.consolidation import memory_consolidate
from living_memory.storage import MemoryStore


def test_consolidation_creates_concept_from_similar_traces_and_keeps_sources(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        traces = []
        for index in range(100):
            trace = store.append_trace(
                f"deploy rollback requires migration guard before release {index}",
                {
                    "scope": "project:alpha",
                    "agent": "agent-a" if index % 2 == 0 else "agent-b",
                    "timestamp": "2026-05-04T09:00:00Z",
                },
                feedback={"confidence": 0.4, "usefulness_score": 0.2},
            )
            traces.append(trace)

        for trace in traces[:5]:
            store.record_access(trace.id)

        result = memory_consolidate(store, scope="project:alpha")

        assert result.traces_considered == 100
        assert len(result.concepts_created) == 1
        concept = result.concepts_created[0]
        assert concept.level == "concept"
        assert concept.scope == "project:alpha"
        assert concept.source_traces == sorted(trace.id for trace in traces)
        assert concept.confidence > 0.5
        assert concept.unique_agents == 2
        assert concept.temporal_hint == "weekly:mon"

        for trace in traces:
            stored = store.get_node(trace.id)
            assert stored is not None
            assert stored.content == trace.content
            assert stored.decayed is False

        source_edges = store.list_connections(source_id=concept.id, relation_type="related")
        assert len(source_edges) == 100
        assert max(edge.weight for edge in source_edges) > min(edge.weight for edge in source_edges)

        second = memory_consolidate(store, scope="project:alpha")
        assert second.concepts_created == []
        assert len(second.concepts_updated) == 1
        assert store.list_nodes(level="concept", scope="project:alpha") == [second.concepts_updated[0]]


def test_consolidation_waits_for_one_hundred_similar_traces(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for index in range(99):
            store.append_trace(
                f"cache refresh requires same token family {index}",
                {"scope": "global", "agent": "agent-a"},
            )

        result = memory_consolidate(store, scope="global")

        assert result.concepts_created == []
        assert store.list_nodes(level="concept", scope="global") == []


def test_consolidation_keeps_scopes_isolated(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for index in range(100):
            store.append_trace(
                f"shared wording belongs to alpha scope {index}",
                {"scope": "project:alpha", "agent": "agent-a"},
            )
            store.append_trace(
                f"shared wording belongs to beta scope {index}",
                {"scope": "project:beta", "agent": "agent-a"},
            )

        alpha = memory_consolidate(store, scope="project:alpha")

        assert len(alpha.concepts_created) == 1
        assert store.list_nodes(level="concept", scope="project:beta") == []
        assert len(store.list_nodes(level="trace", scope="project:beta")) == 100


def test_consolidation_clusters_cross_language_traces_into_one_concept(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        traces = []
        for index in range(50):
            traces.append(
                store.append_trace(
                    "deployment failed because database migration was missing",
                    {"scope": "project:alpha", "agent": f"agent-en-{index % 2}"},
                )
            )
            traces.append(
                store.append_trace(
                    "развертывание упало потому что отсутствовала миграция базы данных",
                    {"scope": "project:alpha", "agent": f"agent-ru-{index % 2}"},
                )
            )

        result = memory_consolidate(store, scope="project:alpha")

        assert len(result.concepts_created) == 1
        concept = result.concepts_created[0]
        assert concept.source_traces == sorted(trace.id for trace in traces)
        assert store.list_nodes(level="concept", scope="project:alpha") == [concept]
