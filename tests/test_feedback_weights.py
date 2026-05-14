from pathlib import Path

import pytest

from living_memory.feedback import apply_retrieval_feedback
from living_memory.retrieval import MemoryRecallService, RecallResult
from living_memory.storage import MemoryStore


def test_feedback_moves_retrieval_weights_toward_successful_method(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = store.append_trace(
            "Authentication latency fixed by token cache warmup",
            {"scope": "project:alpha", "agent": "agent-a"},
        )
        result = RecallResult(
            node=node,
            score=0.8,
            bm25_score=0.0,
            vector_score=0.9,
            graph_score=0.0,
            methods=("vector",),
        )

        before = store.get_retrieval_weights("project:alpha").normalized()
        for _index in range(100):
            update = apply_retrieval_feedback(store, result, useful=True, signal=1.0)

        after = update.weights.normalized()
        assert after.vector > before.vector
        assert after.bm25 < before.bm25
        assert after.vector != pytest.approx(before.vector)
        assert store.get_node(node.id).usefulness_score == pytest.approx(1.0)


def test_service_feedback_updates_rank_for_future_recalls(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        low = store.append_trace(
            "Cache warmup can reduce API latency",
            {"scope": "project:alpha", "agent": "agent-a"},
            feedback={"confidence": 0.5, "usefulness_score": 0.0},
        )
        high = store.append_trace(
            "Cache priming reduces API slowdowns",
            {"scope": "project:alpha", "agent": "agent-b"},
            feedback={"confidence": 0.5, "usefulness_score": 0.0},
        )

        service = MemoryRecallService(store)
        initial = service.memory_recall("cache latency", scope="project:alpha", max_results=2)
        chosen = next(result for result in initial if result.node.id == high.id)
        for _index in range(20):
            service.submit_feedback(chosen, useful=True, signal=1.0)

        later = service.memory_recall("cache latency", scope="project:alpha", max_results=2)
        assert later[0].node.id in {high.id, low.id}
        assert store.get_node(high.id).usefulness_score > store.get_node(low.id).usefulness_score
