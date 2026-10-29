from datetime import UTC, datetime
from pathlib import Path

import pytest

from living_memory.config import MemoryConfig
from living_memory.consolidation import _cross_scope_promotion, memory_consolidate, memory_teach
from living_memory.decay import apply_decay
from living_memory.storage import MemoryStore
from living_memory.temporal import parse_timestamp


def test_consensus_confidence_caps_single_agent_and_rises_for_multiple_agents(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for index in range(100):
            store.append_trace(
                f"solo consensus fact repeats with stable wording {index}",
                {"scope": "project:solo", "agent": "agent-a"},
                feedback={"confidence": 0.9, "usefulness_score": 0.8},
            )
            store.append_trace(
                f"team consensus fact repeats with stable wording {index}",
                {"scope": "project:team", "agent": "agent-a" if index % 2 else "agent-b"},
                feedback={"confidence": 0.9, "usefulness_score": 0.8},
            )

        solo = memory_consolidate(store, scope="project:solo").concepts_created[0]
        team = memory_consolidate(store, scope="project:team").concepts_created[0]

        assert solo.unique_agents == 1
        assert solo.confidence == pytest.approx(0.5)
        assert team.unique_agents == 2
        assert team.confidence > 0.5


@pytest.mark.parametrize("raw,expected", [
    ("2026-10-29T12:00:00Z", datetime(2026, 10, 29, 12, tzinfo=UTC)),
    ("2026-10-29T15:00:00+03:00", datetime(2026, 10, 29, 12, tzinfo=UTC)),
    ("2026-10-29T12:00:00", datetime(2026, 10, 29, 12, tzinfo=UTC)),
    (" 2026-10-29 ", datetime(2026, 10, 29, tzinfo=UTC)),
    (None, None), ("", None), ("invalid", None),
])
def test_parse_timestamp_normalizes_utc_and_rejects_invalid_values(raw, expected) -> None:
    assert parse_timestamp(raw) == expected


@pytest.mark.parametrize("procedural", [False, True])
@pytest.mark.parametrize("hint", [None, "2026-Q3", "weekly:thu"])
def test_derived_writers_preserve_owner_metadata_without_inferring_recurrence(
    tmp_path: Path, procedural: bool, hint: str | None,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        scope = "project:incident"
        context = {"scope": scope, "timestamp": "2026-10-29T12:00:00Z"}
        if procedural:
            context["procedure_id"] = "verify_import"

        def append(index):
            trace = store.append_trace(
                f"Verify checksum before reopening writes after the import incident; observation {index}.",
                {**context, "agent": f"author-{index}", "step_order": index},
            )
            return store.update_node(trace.id, stats={"temporal_hint": "2026-Q3"})

        traces = [append(index) for index in range(3)]
        memory_consolidate(store, scope=scope, force=True, min_cluster_size=3)
        level = "schema" if procedural else "concept"
        first, = store.list_nodes(level=level, scope=scope)
        assert first.temporal_hint is None
        assert first.unique_agents == 3
        assert set(first.source_traces) == {trace.id for trace in traces}
        store.update_node(first.id, stats={"temporal_hint": hint})

        traces.append(append(3))
        memory_consolidate(store, scope=scope, force=True, min_cluster_size=3)
        updated, = store.list_nodes(level=level, scope=scope)
        assert updated.id == first.id
        assert updated.temporal_hint == hint
        assert updated.unique_agents == 4
        assert set(updated.source_traces) == {trace.id for trace in traces}
        assert all(updated.provenance["source_coverage"][source] == covered_at
                   for source, covered_at in first.provenance["source_coverage"].items())
        assert all(store.get_node(trace.id).content == trace.content for trace in traces)


@pytest.mark.parametrize("hint", [None, "2026-Q3", "weekly:thu"])
def test_promotion_preserves_owner_metadata_without_copying_source_hints(tmp_path: Path, hint) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        sources = [store.create_node(
            level="concept", content=f"Verify checksum before reopening import writes. Observation {index}.",
            context={"scope": f"project:incident-{index}", "agent": f"author-{index}"},
            embedding=[1.0, 0.0], stats={"temporal_hint": "weekly:thu"},
        ) for index in range(3)]
        first, = _cross_scope_promotion(store, sources[0])
        assert first.temporal_hint is None
        assert first.unique_agents == 3
        store.update_node(first.id, stats={"temporal_hint": hint})
        store.update_node(sources[0].id, stats={"usefulness_score": 0.6})

        memory_consolidate(store, scope="project:unrelated", force=True)
        updated, = store.list_nodes(level="concept", scope="global")
        assert updated.id == first.id
        assert updated.temporal_hint == hint
        assert updated.usefulness_score == pytest.approx(0.2)
        assert updated.content == first.content
        assert updated.unique_agents == first.unique_agents
        assert updated.confidence == first.confidence
        assert updated.source_traces == first.source_traces
        assert updated.provenance["source_concepts"] == first.provenance["source_concepts"]
        assert updated.provenance["source_coverage"] == first.provenance["source_coverage"]


def test_decay_soft_deletes_expired_and_superseded_nodes(tmp_path: Path) -> None:
    config = MemoryConfig(db_path=tmp_path / "memory.sqlite3", trace_ttl_days=7)
    with MemoryStore(config) as store:
        expired = store.append_trace(
            "old trace should decay",
            {
                "scope": "global",
                "agent": "agent-a",
                "timestamp": "2020-01-01T00:00:00Z",
            },
        )
        active_original = store.append_trace(
            "release channel is beta",
            {
                "scope": "global",
                "agent": "agent-a",
                "timestamp": "2026-05-14T00:00:00Z",
            },
        )
        taught = memory_teach(
            store,
            active_original.id,
            "release channel is stable",
            context={"agent": "agent-b"},
        )

        result = apply_decay(store, now=datetime(2026, 5, 14, tzinfo=UTC))

        assert {node.id for node in result.expired} == {expired.id}
        assert {node.id for node in result.superseded} == {active_original.id}
        assert store.get_node(expired.id).decayed is True
        assert store.get_node(active_original.id).decayed is True
        assert store.get_node(taught.corrective_trace.id).decayed is False
