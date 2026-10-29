from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from living_memory.prompts import retrieval_context_prompt
from living_memory.resources import (
    global_concepts,
    memory_health,
    memory_stats,
    memory_status,
    project_concepts,
    recent_interactions,
)
from living_memory.storage import MemoryStore


@pytest.fixture
def health_now(monkeypatch):
    instant = datetime(2026, 10, 29, 12, tzinfo=UTC)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant

    monkeypatch.setattr("living_memory.resources.datetime", Clock)
    return instant


def test_concept_resources_are_browsable_by_global_and_project_scope(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        global_concept = store.create_node(
            level="concept",
            content="All projects require release notes before deploy",
            context={"scope": "global", "agent": "agent-a"},
            stats={"confidence": 0.9, "unique_agents": 2},
        )
        alpha_concept = store.create_node(
            level="concept",
            content="Alpha deploy requires migration dry run",
            context={"scope": "project:alpha", "agent": "agent-b"},
            stats={"confidence": 0.8, "unique_agents": 2},
        )
        beta_concept = store.create_node(
            level="concept",
            content="Beta deploy uses canary routing",
            context={"scope": "project:beta", "agent": "agent-c"},
            stats={"confidence": 0.8, "unique_agents": 2},
        )

        global_resource = global_concepts(store)
        alpha_resource = project_concepts(store, "alpha")

        assert global_resource["uri"] == "memory://global/concepts"
        assert [item["id"] for item in global_resource["concepts"]] == [global_concept.id]
        assert alpha_resource["uri"] == "memory://project/alpha/concepts"
        assert [item["id"] for item in alpha_resource["concepts"]] == [alpha_concept.id]
        assert beta_concept.id not in {item["id"] for item in alpha_resource["concepts"]}


def test_stats_status_and_recent_resources_report_store_state(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        first = store.append_trace(
            "first interaction",
            {"scope": "global", "agent": "agent-a", "timestamp": "2026-05-14T01:00:00Z"},
        )
        second = store.append_trace(
            "second interaction",
            {
                "scope": "project:alpha",
                "agent": "agent-b",
                "timestamp": "2026-05-14T02:00:00Z",
            },
        )
        store.create_node(
            level="concept",
            content="Alpha interactions share a deployment pattern",
            context={"scope": "project:alpha", "agent": "agent-c"},
            stats={"confidence": 0.7, "unique_agents": 2},
            provenance={"source_traces": [second.id]},
        )

        stats = memory_stats(store)
        status = memory_status(store, "project:alpha")
        recent = recent_interactions(store, limit=2)

        assert stats["uri"] == "memory://stats"
        assert stats["counts"]["by_level"]["trace"]["total"] == 2
        assert status["scope"] == "project:alpha"
        assert status["counts"]["by_level"]["concept"]["active"] == 1
        assert recent["uri"] == "memory://recent"
        assert [item["id"] for item in recent["traces"]] == [second.id, first.id]


def test_retrieval_context_prompt_formats_top_relevant_concepts(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        selected = store.create_node(
            level="concept",
            content="Alpha deploy rollback requires migration dry run before release",
            context={"scope": "project:alpha", "agent": "agent-a"},
            stats={"confidence": 0.92, "unique_agents": 2, "usefulness_score": 0.6},
        )
        low_confidence = store.create_node(
            level="concept",
            content="Alpha deploy rollback can skip migration checks",
            context={"scope": "project:alpha", "agent": "agent-b"},
            stats={"confidence": 0.2, "unique_agents": 1, "usefulness_score": -0.3},
        )
        other_project = store.create_node(
            level="concept",
            content="Beta deploy rollback requires migration dry run before release",
            context={"scope": "project:beta", "agent": "agent-c"},
            stats={"confidence": 0.95, "unique_agents": 2, "usefulness_score": 0.9},
        )

        block = retrieval_context_prompt(
            store,
            task="deploy rollback migration",
            scope="project:alpha",
            max_concepts=3,
            min_confidence=0.5,
            retrieval_policy="confidence",
        )

        assert block.startswith("BEGIN ACTIVE MEMORY CONTEXT")
        assert "scope_plan: project:alpha > global" in block
        assert selected.id in block
        assert selected.content in block
        assert low_confidence.id not in block
        assert other_project.id not in block
        assert "retrieval_policy: confidence" in block
        assert block.endswith("END ACTIVE MEMORY CONTEXT")


def test_memory_health_reports_activity_dedup_and_staleness(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        # Distinct contents have no textual duplicate excess.
        store.append_trace(
            "rollback migration release notes alpha",
            {"scope": "project:alpha", "agent": "agent-a"},
        )
        store.append_trace(
            "rollback migration release notes beta",
            {"scope": "project:alpha", "agent": "agent-b"},
        )
        store.append_trace(
            "unique fact about caching",
            {"scope": "project:alpha", "agent": "agent-c"},
        )
        store.record_recall_event(
            query="rollback",
            scope="project:alpha",
            requested_scope="project:alpha",
            resolved_scopes=["project:alpha"],
            results=[],
            ambient_context={},
        )

        report = memory_health(store, scope="project:alpha", window_hours=24, top_stale=2)

        assert report["scope"] == "project:alpha"
        assert report["counts"]["by_level"]["trace"]["active"] == 3
        assert report["activity"]["recall_in_window"] == 1
        assert report["activity"]["remember_in_window"] == 3
        assert report["activity"]["recall_to_remember_ratio"] > 0
        # Three distinct contents → duplicate_excess=0, density 0.0.
        assert report["dedup"]["total_traces"] == 3
        assert report["dedup"]["distinct_contents"] == 3
        assert report["dedup"]["duplicate_excess"] == 0
        assert report["dedup"]["duplicate_density"] == 0.0
        assert report["staleness"]["age_seconds_p50"] is not None
        assert len(report["staleness"]["top_stale"]) == 2
        assert "retrieval_policy" in report


@pytest.mark.parametrize(
    ("second_agent", "active", "decayed", "duplicate_excess", "duplicate_density"),
    [
        pytest.param("agent-a", 1, 1, 0, 0.0, id="same-source"),
        pytest.param("agent-b", 2, 0, 1, 0.5, id="distinct-sources"),
    ],
)
def test_memory_health_distinguishes_source_identity_from_text_repetition(
    tmp_path: Path,
    second_agent: str,
    active: int,
    decayed: int,
    duplicate_excess: int,
    duplicate_density: float,
) -> None:
    """Write dedup preserves source identity; health counts textual repetition."""

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.append_trace(
            "shared bootstrap content",
            {"scope": "project:alpha", "agent": "agent-a"},
        )
        store.append_trace(
            "shared bootstrap content",
            {"scope": "project:alpha", "agent": second_agent},
        )

        report = memory_health(store, scope="project:alpha", top_stale=0)

        assert report["counts"]["by_level"]["trace"]["active"] == active
        assert report["counts"]["by_level"]["trace"]["decayed"] == decayed
        assert report["dedup"]["total_traces"] == active
        assert report["dedup"]["distinct_contents"] == 1
        assert report["dedup"]["duplicate_excess"] == duplicate_excess
        assert report["dedup"]["duplicate_density"] == duplicate_density


def test_memory_health_handles_empty_scope(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        report = memory_health(store, scope="project:empty")

    assert report["counts"]["active"] == 0
    assert report["dedup"]["duplicate_density"] == 0.0
    assert report["staleness"]["age_seconds_p50"] is None
    assert report["staleness"]["top_stale"] == []
    assert report["activity"]["recall_to_remember_ratio"] is None


def test_memory_health_staleness_uses_time_since_access(tmp_path, health_now):
    def ago(days):
        return (health_now - timedelta(days=days)).isoformat()

    with MemoryStore(tmp_path / "staleness.sqlite3") as store:
        for name, created, accessed, scope, level, decayed in [
            ("recently-used", 90, 0, "project:alpha", "trace", False),
            ("stale", 60, 30, "project:alpha", "trace", False),
            ("never-used", 20, None, "project:alpha", "trace", False),
            ("foreign", 100, None, "project:beta", "trace", False),
            ("concept", 100, None, "project:alpha", "concept", False),
            ("decayed", 100, None, "project:alpha", "trace", True),
        ]:
            store.create_node(
                node_id=name, level=level, content=name, context={"scope": scope},
                timestamp=ago(created), stats={"decayed": decayed,
                    "last_accessed": ago(accessed) if accessed is not None else None,
                    "access_count": int(accessed is not None)},
            )
        before = list(store.connection.iterdump())
        report = memory_health(store, scope="project:alpha", top_stale=1)
        stale = report["staleness"]
        assert [row["id"] for row in stale["top_stale"]] == ["stale"]
        assert stale["age_seconds_p50"] == 20 * 86400
        assert stale["age_seconds_p90"] == 30 * 86400
        assert stale["decay_rate"] == 0.25
        assert stale["top_stale"][0] == {
            "id": "stale", "content_preview": "stale", "timestamp": ago(60),
            "access_count": 1, "last_accessed": ago(30),
        }
        all_rows = memory_health(store, scope="project:alpha", top_stale=20)["staleness"]
        assert [row["id"] for row in all_rows["top_stale"]] == [
            "stale", "never-used", "recently-used",
        ]
        assert memory_health(store, top_stale=1)["staleness"]["top_stale"][0]["id"] == "foreign"
        assert memory_health(store, top_stale=0)["staleness"]["top_stale"] == []
        with pytest.raises(ValueError, match="top_stale"):
            memory_health(store, top_stale=-1)
        assert list(store.connection.iterdump()) == before


@pytest.mark.parametrize("created,accessed,age", [
    ("2026-10-19T12:00:00Z", None, 10 * 86400),
    ("2026-10-19T12:00:00Z", "", 10 * 86400),
    ("invalid", "2026-10-28T17:00:00+05:00", 86400),
    ("2026-10-19T12:00:00Z", "2026-10-30T12:00:00Z", 0),
    ("2026-10-19T12:00:00Z", "invalid", None),
    ("2026-10-19T12:00:00Z", " ", None),
    ("invalid", None, None),
    ("", None, None),
])
def test_memory_health_staleness_date_eligibility(tmp_path, health_now, created, accessed, age):
    with MemoryStore(tmp_path / "dates.sqlite3") as store:
        target = store.create_node(level="trace", content="date eligibility")
        with store.transaction():
            store.connection.execute(
                "UPDATE nodes SET timestamp = ?, last_accessed = ? WHERE id = ?",
                (created, accessed, target.id),
            )
        stale = memory_health(store)["staleness"]
    assert stale["age_seconds_p50"] == age
    assert stale["age_seconds_p90"] == age
    assert [row["id"] for row in stale["top_stale"]] == ([] if age is None else [target.id])
