from pathlib import Path

import pytest

from living_memory.phase import PhaseManager
from living_memory.storage import MemoryStore


@pytest.mark.parametrize(
    ("trace_count", "phase"),
    [
        (0, 0),
        (1, 1),
        (99, 1),
        (100, 2),
        (9_999, 2),
        (10_000, 3),
        (99_999, 3),
        (100_000, 4),
        (999_999, 4),
        (1_000_000, 5),
    ],
)
def test_phase_manager_detects_automatic_phase_boundaries(trace_count: int, phase: int) -> None:
    detected = PhaseManager().detect(trace_count)
    assert detected.number == phase
    assert detected.trace_count == trace_count
    assert detected.features
    if phase >= 1:
        assert "fts5 bm25 recall" in detected.features
    if phase >= 3:
        assert "concept graph" in detected.features


def test_phase_manager_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        PhaseManager().detect(-1)


def test_memory_store_detects_phase_from_total_trace_count(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        assert store.detect_phase().number == 0
        for index in range(100):
            store.append_trace(f"trace {index}", {"scope": "global", "agent": "agent-a"})

        assert store.trace_count() == 100
        assert store.detect_phase().number == 2

        first_trace = store.list_nodes(level="trace", include_decayed=True, limit=1)[0]
        store.soft_delete_node(first_trace.id, "test decay")
        assert store.trace_count(include_decayed=False) == 99
        assert store.detect_phase().number == 2
