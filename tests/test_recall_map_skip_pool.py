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


@pytest.mark.parametrize("decay", [False, True])
def test_recall_residual_skips_pool_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decay: bool) -> None:
    import os
    from unittest.mock import patch

    from living_memory.config import MemoryConfig
    from living_memory.recall_map import CURTAIL_STREAK, RecallMapBuilder
    from living_memory.retrieval import MemoryRecallService

    for key in os.environ:
        if key.startswith("LM_MAP_"):
            monkeypatch.delenv(key, raising=False)
    if decay:
        monkeypatch.setenv("LM_MAP_CURTAIL_DECAY", "1")
    scope, task = "project:test", "skip-pool"
    config = MemoryConfig(db_path=tmp_path / "recall.sqlite3", embedding_model="hash")
    with MemoryStore(config) as store:
        for index in range(15):
            store.create_node(level="trace", content=f"database migration rollback {index}", context={"scope": scope})
        for index in range(CURTAIL_STREAK):
            store.record_recall_event(
                query="orchard inspection", scope=scope, ambient_context={"task": task},
                recall_map={"clusters": [{"label": "orchard inspection", "count": 1,
                                          "medoid": {"node_id": f"absent-{index}"}}],
                            "pool": 1, "covered": 1},
            )
        if decay:
            store.record_recall_event(
                query="orchard inspection", scope=scope, ambient_context={"task": task},
                recall_map={"clusters": [], "covered": 0, "curtailed": True,
                            "streak": CURTAIL_STREAK, "pd": "0123456789abcdef"},
            )
        service = MemoryRecallService(store)
        selected = service.memory_recall(
            "database migration rollback", scope=scope, max_results=3,
            ambient_context={"task": task},
        )
        assert len(selected) == 3 and service.last_residual
        with patch.object(store, "matured_recall_history", wraps=store.matured_recall_history) as history:
            built = RecallMapBuilder(store).build(service.last_residual, scope=scope, task=task)
        history.assert_not_called()
        assert built is not None and built.pool_size is None and built.selection is None
        payload = built.to_dict()
        assert payload == {"clusters": [], "covered": 0, "curtailed": True,
                           "streak": CURTAIL_STREAK + int(decay)}
        store.attach_recall_map(service.last_recall_event_id, payload)
    with MemoryStore(config) as reopened:
        assert reopened.recent_recall_map_history()[0]["recall_map"] == payload
