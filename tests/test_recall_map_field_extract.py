"""Persisted pool measurements remain optional in offline field cases."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from living_memory.recall_map import _payload
from living_memory.storage import MemoryStore

path = Path(__file__).resolve().parents[1] / "scripts" / "recall_map_field_extract.py"
spec = importlib.util.spec_from_file_location("recall_map_field_extract", path)
assert spec is not None and spec.loader is not None
extract = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = extract
spec.loader.exec_module(extract)


@pytest.mark.parametrize("pool", [None, 0, 7])
def test_case_preserves_measured_and_unmeasured_pool(tmp_path: Path, pool: int | None) -> None:
    payload = _payload((), pool, 0, curtailed=pool != 0, streak=4)
    with MemoryStore(tmp_path / "field.sqlite3") as store:
        event = store.record_recall_event(query="marker", scope="project:test", recall_map=payload)
        row = store.connection.execute("SELECT * FROM recall_events WHERE id = ?", (event.id,)).fetchone()
        case = extract.build_case(row, payload, store, source="test")
    assert case["pool_size"] == pool
    assert case["persisted_payload"] == payload
    assert case["cluster_count"] == 0
    assert case["budget"] is None
