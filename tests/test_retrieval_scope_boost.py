"""Requested-scope results must outrank entrenched broader-scope results.

Regression for the 2026-06-09 incident where a `scope="project:x"` recall for
checkpoint/copy-sync knowledge returned month-old `global`-scope traces from a
different project above fresh in-scope traces: the old 0.04 per-rank scope
boost could not overcome the usefulness/access feedback boosts the entrenched
nodes had accumulated. The boost must win at comparable relevance while
remaining a soft re-rank — cross-scope precedents stay retrievable.
"""

from __future__ import annotations

from pathlib import Path

from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore


def test_requested_scope_outranks_entrenched_global_at_equal_relevance(
    tmp_path: Path,
) -> None:
    content = "scheduler frontier copy bubble attribution ownership map"
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        local = store.append_trace(
            content,
            {"scope": "project:alpha", "agent": "fresh-agent"},
        )
        entrenched = store.append_trace(
            content,
            {"scope": "global", "agent": "veteran-agent"},
            feedback={"usefulness_score": 0.25},
        )

        service = MemoryRecallService(store)
        results = service.memory_recall(
            content,
            scope="project:alpha",
            depth=0,
            max_results=5,
            log_access=False,
            log_event=False,
        )

        ids = [result.node.id for result in results]
        assert ids and ids[0] == local.id, ids
        # Soft re-rank, not a filter: the cross-scope precedent stays reachable.
        assert entrenched.id in ids
