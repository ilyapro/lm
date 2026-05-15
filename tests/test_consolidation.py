from pathlib import Path

import pytest

from living_memory.config import MemoryConfig
from living_memory.consolidation import _cross_scope_promotion, memory_consolidate
from living_memory.resources import memory_status
from living_memory.storage import MemoryStore


def _phase4_config(path: Path) -> MemoryConfig:
    return MemoryConfig(
        db_path=path,
        phase_thresholds={0: 0, 1: 1, 2: 100, 3: 200, 4: 300, 5: 1_000_000},
    )


def _append_topic_traces(store: MemoryStore, scope: str, topic: str) -> None:
    for index in range(100):
        store.append_trace(
            f"{topic} stable project knowledge sample {index}",
            {
                "scope": scope,
                "agent": "agent-a" if index % 2 == 0 else "agent-b",
            },
            feedback={"confidence": 0.55, "usefulness_score": 0.2},
        )


def _consolidate_scopes(store: MemoryStore, scopes: list[str]) -> None:
    for scope in scopes:
        memory_consolidate(store, scope=scope)


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


def test_cross_scope_promotion_creates_one_global_concept_at_phase_four(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    scopes = ["project:alpha", "project:beta", "project:gamma"]
    with MemoryStore(_phase4_config(tmp_path / "memory.sqlite3")) as store:
        for scope in scopes:
            _append_topic_traces(store, scope, "database timeout connection pool retry budget")

        _consolidate_scopes(store, scopes)

        globals_ = store.list_nodes(level="concept", scope="global")
        assert len(globals_) == 1
        global_concept = globals_[0]
        assert sorted(global_concept.provenance["promoted_from"]) == scopes
        assert len(global_concept.source_traces) == 300
        assert global_concept.embedding is not None

        project_concepts = store.list_nodes(level="concept", include_decayed=False, limit=10)
        project_concept_ids = {
            concept.id for concept in project_concepts if concept.scope.startswith("project:")
        }
        promotion_edges = store.list_connections(
            target_id=global_concept.id,
            relation_type="related",
        )
        assert {edge.source_id for edge in promotion_edges} == project_concept_ids

        status = memory_status(store)
        assert status["promotions"]["promoted_concepts"] == 1
        assert status["promotions"]["promotion_events"] == 1


def test_cross_scope_promotion_reinforces_existing_global_without_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    initial_scopes = ["project:alpha", "project:beta", "project:gamma"]
    with MemoryStore(_phase4_config(tmp_path / "memory.sqlite3")) as store:
        for scope in initial_scopes:
            _append_topic_traces(store, scope, "database timeout connection pool retry budget")
        _consolidate_scopes(store, initial_scopes)

        original = store.list_nodes(level="concept", scope="global")[0]
        original_confidence = original.confidence

        _append_topic_traces(
            store,
            "project:delta",
            "database timeout connection pool retry budget",
        )
        result = memory_consolidate(store, scope="project:delta")

        globals_ = store.list_nodes(level="concept", scope="global")
        assert [concept.id for concept in globals_] == [original.id]
        reinforced = globals_[0]
        assert reinforced.confidence > original_confidence
        assert sorted(reinforced.provenance["promoted_from"]) == sorted(
            [*initial_scopes, "project:delta"]
        )
        assert len(reinforced.provenance["promotion_events"]) == 2
        assert result.concepts_promoted == [reinforced]


def test_cross_scope_promotion_is_phase_gated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    scopes = ["project:alpha", "project:beta", "project:gamma"]
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for scope in scopes:
            _append_topic_traces(store, scope, "database timeout connection pool retry budget")

        _consolidate_scopes(store, scopes)

        assert store.detect_phase().number < 4
        assert store.list_nodes(level="concept", scope="global") == []


def test_cross_scope_promotion_skips_concept_without_embedding(tmp_path: Path) -> None:
    with MemoryStore(_phase4_config(tmp_path / "memory.sqlite3")) as store:
        concept = store.create_node(
            level="concept",
            content="database timeout connection pool retry budget",
            context={"scope": "project:alpha"},
            stats={"confidence": 0.8, "unique_agents": 2},
        )

        promoted = _cross_scope_promotion(store, concept, phase_number=4)

        assert promoted is None
        assert store.list_nodes(level="concept", scope="global") == []


def test_cross_scope_promotion_ignores_dissimilar_project_concepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    topics = {
        "project:alpha": "database timeout connection pool retry budget",
        "project:beta": "frontend button layout mobile overflow color palette",
        "project:gamma": "invoice export pdf rounding mismatch totals",
    }
    with MemoryStore(_phase4_config(tmp_path / "memory.sqlite3")) as store:
        for scope, topic in topics.items():
            _append_topic_traces(store, scope, topic)

        _consolidate_scopes(store, list(topics))

        assert store.list_nodes(level="concept", scope="global") == []
