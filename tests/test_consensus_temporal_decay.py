from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from living_memory.config import MemoryConfig
from living_memory.consolidation import memory_consolidate, memory_teach
from living_memory.decay import apply_decay
from living_memory.storage import MemoryStore
from living_memory.temporal import detect_weekly_hint, split_time_regimes


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


def test_weekly_temporal_hint_detects_dominant_monday() -> None:
    hint = detect_weekly_hint(
        [
            "2026-05-04T09:00:00Z",
            "2026-05-11T09:00:00Z",
            "2026-05-18T09:00:00Z",
            "2026-05-25T09:00:00Z",
        ]
    )

    assert hint == "weekly:mon"


def test_split_time_regimes_detects_disjoint_bursts() -> None:
    # Shape of the animal-planet mixed-era cluster: a tight pre-goal burst,
    # 15.76 days of silence, then two in-goal traces about a day apart.
    moments = [
        datetime(2026, 7, 26, 20, 36, 16, tzinfo=UTC),
        datetime(2026, 7, 26, 20, 36, 41, tzinfo=UTC),
        datetime(2026, 7, 26, 20, 37, 6, tzinfo=UTC),
        datetime(2026, 7, 26, 20, 37, 32, tzinfo=UTC),
        datetime(2026, 8, 11, 14, 46, 24, tzinfo=UTC),
        datetime(2026, 8, 12, 15, 7, 25, tzinfo=UTC),
    ]

    assert split_time_regimes(moments) == [[0, 1, 2, 3], [4, 5]]


def test_split_time_regimes_keeps_steady_cadence_whole() -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    moments = [base + timedelta(days=10 * index) for index in range(6)]

    # Every gap exceeds a week, but none dwarfs the cluster's own rhythm.
    assert split_time_regimes(moments) == [[0, 1, 2, 3, 4, 5]]


def test_split_time_regimes_finds_multiple_eras() -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    moments = []
    for clump in range(3):
        start = base + timedelta(days=30 * clump)
        moments.extend(
            [start, start + timedelta(minutes=5), start + timedelta(minutes=10)]
        )

    assert split_time_regimes(moments) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def test_split_time_regimes_splits_two_distant_points_and_skips_none() -> None:
    first = datetime(2026, 1, 1, tzinfo=UTC)
    second = first + timedelta(days=30)

    assert split_time_regimes([second, None, first]) == [[2], [0]]
    assert split_time_regimes([None, None]) == []


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
