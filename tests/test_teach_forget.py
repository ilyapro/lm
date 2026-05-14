from pathlib import Path

import pytest

from living_memory.consolidation import memory_teach
from living_memory.decay import memory_forget
from living_memory.storage import MemoryStore


def test_memory_teach_creates_corrective_trace_and_supersedes_original(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        original = store.append_trace(
            "status endpoint is /healthz",
            {"scope": "project:api", "agent": "agent-a"},
            feedback={"confidence": 0.5, "usefulness_score": 0.4},
        )

        taught = memory_teach(
            store,
            original.id,
            "status endpoint is /readyz",
            confidence=0.95,
            context={"agent": "agent-b"},
        )

        assert taught.corrective_trace.level == "trace"
        assert taught.corrective_trace.content == "status endpoint is /readyz"
        assert taught.corrective_trace.scope == "project:api"
        assert taught.corrective_trace.confidence <= 0.5
        assert taught.corrective_trace.usefulness_score > taught.original.usefulness_score
        assert taught.supersedes.source_id == taught.corrective_trace.id
        assert taught.supersedes.target_id == original.id
        assert taught.supersedes.type == "supersedes"
        assert taught.original.corrections[-1]["new"] == "status endpoint is /readyz"

        ranked = store.search_content("status endpoint", scope="project:api")
        assert ranked[0][0].id == taught.corrective_trace.id


def test_memory_forget_soft_deletes_without_removing_node(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        trace = store.append_trace(
            "temporary observation can be forgotten",
            {"scope": "global", "agent": "agent-a"},
        )

        forgotten = memory_forget(store, trace.id, "operator requested")

        assert forgotten.decayed is True
        assert forgotten.decay_reason == "operator requested"
        assert store.get_node(trace.id) is not None
        assert store.list_nodes(level="trace", scope="global") == []
        assert store.list_nodes(level="trace", scope="global", include_decayed=True)[0].id == trace.id


def test_memory_teach_rejects_missing_or_empty_corrections(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        trace = store.append_trace("original fact", {"scope": "global", "agent": "agent-a"})

        with pytest.raises(ValueError):
            memory_teach(store, trace.id, " ")
        with pytest.raises(KeyError):
            memory_teach(store, "missing", "replacement fact")

