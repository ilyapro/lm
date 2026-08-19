"""Residual ranked pool exposure: ``MemoryRecallService.last_residual``.

The recall map is built from the candidates ranked past the ``max_results``
cut. These tests pin the exposure contract: the residual is the tail of the
same ranking that produced the delivered slice, it is overwritten on every
call (to ``[]`` when nothing remains), and it stays invisible to delivery,
access logging, and the recorded recall event.
"""

from pathlib import Path

from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore

POOL_QUERY = "deployment failure database migration rollback"
POOL_SCOPE = "project:alpha"


def _seed_pool(store: MemoryStore, count: int) -> None:
    for index in range(count):
        store.append_trace(
            f"deployment failure case {index}: database migration rollback step {index}",
            {"scope": POOL_SCOPE, "agent": f"agent-{index}"},
        )


def test_last_residual_is_ranked_tail_disjoint_from_delivered(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _seed_pool(store, 12)
        service = MemoryRecallService(store)

        # Reference pass without logging: it warms the lazy embeddings and
        # captures the ranking this exact store state produces. It mutates
        # nothing rank-relevant, so the logged pass below must slice the same
        # ranking at the same cut.
        reference = service.memory_recall(
            POOL_QUERY, scope=POOL_SCOPE, max_results=4, log_access=False
        )
        reference_residual = list(service.last_residual)

        results = service.memory_recall(POOL_QUERY, scope=POOL_SCOPE, max_results=4)
        residual = service.last_residual

        assert len(results) == 4
        assert residual

        delivered_ids = [result.node.id for result in results]
        residual_ids = [result.node.id for result in residual]
        assert set(delivered_ids).isdisjoint(residual_ids)
        assert delivered_ids == [result.node.id for result in reference]
        assert residual_ids == [result.node.id for result in reference_residual]

        # Rank order holds across the cut: scores never increase from the
        # delivered slice into and through the residual.
        scores = [result.score for result in results] + [
            result.score for result in residual
        ]
        assert scores == sorted(scores, reverse=True)


def test_last_residual_empty_when_pool_fits_and_overwritten_per_call(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _seed_pool(store, 12)
        store.append_trace(
            "billing invoice rounding bug in currency conversion",
            {"scope": "project:beta", "agent": "agent-billing"},
        )
        service = MemoryRecallService(store)

        service.memory_recall(POOL_QUERY, scope=POOL_SCOPE, max_results=4)
        assert service.last_residual

        small = service.memory_recall(
            "billing invoice rounding currency",
            scope="project:beta",
            max_results=10,
        )
        assert small
        assert len(small) < 10
        assert service.last_residual == []

        service.memory_recall(POOL_QUERY, scope=POOL_SCOPE, max_results=4)
        assert service.last_residual
        assert service.memory_recall("   ", scope=POOL_SCOPE) == []
        assert service.last_residual == []


def test_delivery_event_and_access_logging_ignore_residual(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _seed_pool(store, 12)
        service = MemoryRecallService(store)

        results = service.memory_recall(POOL_QUERY, scope=POOL_SCOPE, max_results=4)
        residual = service.last_residual

        assert len(results) == 4
        assert residual

        event = store.get_recall_event(service.last_recall_event_id)
        assert event is not None
        assert len(event.results) <= 4
        assert event.result_ids == [result.node.id for result in results]
        assert set(event.result_ids).isdisjoint(
            {result.node.id for result in residual}
        )

        # The event id is stamped onto delivered results only; the residual is
        # the raw ranked tail, untouched by event recording.
        assert all(result.recall_event_id == event.id for result in results)
        assert all(result.recall_event_id is None for result in residual)

        for result in results:
            assert store.get_node(result.node.id).access_count == 1
        for result in residual:
            assert store.get_node(result.node.id).access_count == 0
