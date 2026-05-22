from __future__ import annotations

from pathlib import Path

import pytest

from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore


BM25_PRESSURE_QUERY = "checkout deploy migration 42 failure"
SEMANTIC_EVAL_QUERY = "startup slowdown mitigation"
SEMANTIC_HOLDOUT_QUERY = "boot delay remedy"


class ControlledEmbeddingModel:
    def embed(self, text: str) -> list[float]:
        lowered = text.lower()
        if BM25_PRESSURE_QUERY in lowered:
            return [0.0, 0.0, 1.0]
        if lowered in {SEMANTIC_EVAL_QUERY, SEMANTIC_HOLDOUT_QUERY}:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]


@pytest.mark.parametrize(
    ("scope", "vector_floor", "bm25_ceiling"),
    [
        ("project:semantic", 0.15, 0.85),
        ("global", 0.20, 0.75),
        ("session:semantic", 0.10, 0.90),
    ],
)
def test_semantic_recall_survives_repeated_bm25_positive_feedback(
    tmp_path: Path,
    scope: str,
    vector_floor: float,
    bm25_ceiling: float,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        lexical = store.create_node(
            level="trace",
            content=f"{BM25_PRESSURE_QUERY} exact lexical anchor",
            context={"scope": scope, "agent": "agent-a"},
            embedding=[0.0, 0.0, 1.0],
        )
        semantic = store.create_node(
            level="trace",
            content="Cobalt primer prevents launch stalls",
            context={"scope": scope, "agent": "agent-b"},
            # 0.6 is below STRONG_VECTOR_MATCH, so the result depends on
            # the persisted vector weight remaining nonzero.
            embedding=[0.6, 0.8, 0.0],
        )
        service = MemoryRecallService(store, embedder=ControlledEmbeddingModel())

        for _index in range(80):
            exact = service.memory_recall(
                BM25_PRESSURE_QUERY,
                scope=scope,
                max_results=1,
                log_event=False,
            )
            assert exact
            assert exact[0].node.id == lexical.id
            assert exact[0].bm25_score == pytest.approx(1.0)
            service.submit_feedback(exact[0], useful=True, signal=1.0)

        weights = store.get_retrieval_weights(scope).normalized()
        assert weights.bm25 <= bm25_ceiling
        assert weights.vector >= vector_floor

        eval_results = service.memory_recall(
            SEMANTIC_EVAL_QUERY,
            scope=scope,
            max_results=3,
            log_event=False,
        )
        holdout_results = service.memory_recall(
            SEMANTIC_HOLDOUT_QUERY,
            scope=scope,
            max_results=3,
            log_event=False,
        )

        for results in (eval_results, holdout_results):
            semantic_result = next(
                (result for result in results if result.node.id == semantic.id),
                None,
            )
            assert semantic_result is not None
            assert semantic_result.bm25_score == pytest.approx(0.0)
            assert semantic_result.vector_score == pytest.approx(0.6)
            assert semantic_result.score > 0.0
            assert "vector" in semantic_result.methods
