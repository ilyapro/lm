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


def test_weight_update_recovers_project_vector_floor_with_embedding_evidence(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        scope = "project:collapsed"
        store.create_node(
            level="trace",
            content="checkout deploy migration recovery",
            context={"scope": scope, "agent": "agent-a"},
            embedding=[1.0, 0.0, 0.0],
        )
        store.set_retrieval_weights(
            scope,
            bm25=1.0,
            vector=0.0,
            graph=0.0,
            learning_rate=0.05,
        )

        updated = store.update_retrieval_weights(
            scope,
            bm25_signal=1.0,
            vector_signal=-1.0,
        ).normalized()

        assert updated.bm25 == pytest.approx(0.85)
        assert updated.vector == pytest.approx(0.15)
        assert updated.graph == pytest.approx(0.0)


def test_weight_update_lifts_floor_from_bm25_before_other_channels(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        scope = "project:graph-heavy"
        store.create_node(
            level="trace",
            content="causal migration graph note",
            context={"scope": scope, "agent": "agent-a"},
            embedding=[0.0, 1.0, 0.0],
        )
        store.set_retrieval_weights(
            scope,
            bm25=0.10,
            vector=0.0,
            graph=0.90,
            learning_rate=0.05,
        )

        updated = store.update_retrieval_weights(scope).normalized()

        assert updated.bm25 == pytest.approx(0.0)
        assert updated.vector == pytest.approx(0.15)
        assert updated.graph == pytest.approx(0.85)


def test_weight_update_enforces_global_graph_floor_when_graph_evidence_exists(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        source = store.create_node(
            level="trace",
            content="incident root cause",
            context={"scope": "global", "agent": "agent-a"},
            embedding=[1.0, 0.0, 0.0],
        )
        target = store.create_node(
            level="trace",
            content="incident downstream effect",
            context={"scope": "global", "agent": "agent-a"},
            embedding=[0.0, 1.0, 0.0],
        )
        store.create_connection(source.id, target.id, "caused")
        store.set_retrieval_weights(
            "global",
            bm25=1.0,
            vector=0.0,
            graph=0.0,
            learning_rate=0.05,
        )

        updated = store.update_retrieval_weights("global", bm25_signal=1.0).normalized()

        assert updated.bm25 == pytest.approx(0.75)
        assert updated.vector == pytest.approx(0.20)
        assert updated.graph == pytest.approx(0.05)


def test_weight_update_does_not_apply_policy_floor_without_vector_or_graph_evidence(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        scope = "project:no-evidence"
        store.set_retrieval_weights(
            scope,
            bm25=1.0,
            vector=0.0,
            graph=0.0,
            learning_rate=0.05,
        )

        updated = store.update_retrieval_weights(scope, bm25_signal=1.0).normalized()

        assert updated.bm25 == pytest.approx(1.0)
        assert updated.vector == pytest.approx(0.0)
        assert updated.graph == pytest.approx(0.0)


def test_adaptive_tuning_raises_learning_rate_for_young_scope(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    monkeypatch.setenv("LM_RETRIEVAL_TUNING_POLICY", "adaptive")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = store.append_trace(
            "single fact",
            {"scope": "project:young", "agent": "agent-a"},
        )
        result = RecallResult(
            node=node,
            score=0.7,
            bm25_score=0.7,
            vector_score=0.0,
            graph_score=0.0,
            methods=("bm25",),
        )
        apply_retrieval_feedback(store, result, useful=True, signal=1.0)
        weights = store.get_retrieval_weights("project:young")
        # trace_count=1 → adaptive ladder selects 0.20
        assert weights.learning_rate == pytest.approx(0.20)


def test_adaptive_tuning_lowers_learning_rate_as_scope_matures(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    monkeypatch.setenv("LM_RETRIEVAL_TUNING_POLICY", "adaptive")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for index in range(60):
            store.append_trace(
                f"fact {index}",
                {"scope": "project:mature", "agent": f"agent-{index}"},
            )
        node = store.get_node(
            store.list_nodes(level="trace", scope="project:mature", limit=1)[0].id
        )
        result = RecallResult(
            node=node,
            score=0.7,
            bm25_score=0.7,
            vector_score=0.0,
            graph_score=0.0,
            methods=("bm25",),
        )
        apply_retrieval_feedback(store, result, useful=True, signal=1.0)
        weights = store.get_retrieval_weights("project:mature")
        # trace_count=60 falls in 50..500 band → 0.05
        assert weights.learning_rate == pytest.approx(0.05)


def test_fixed_tuning_policy_leaves_learning_rate_untouched(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    monkeypatch.delenv("LM_RETRIEVAL_TUNING_POLICY", raising=False)
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = store.append_trace(
            "single fact",
            {"scope": "project:young", "agent": "agent-a"},
        )
        result = RecallResult(
            node=node,
            score=0.7,
            bm25_score=0.7,
            vector_score=0.0,
            graph_score=0.0,
            methods=("bm25",),
        )
        apply_retrieval_feedback(store, result, useful=True, signal=1.0)
        weights = store.get_retrieval_weights("project:young")
        # default learning_rate stays at 0.05 — no adaptive retuning
        assert weights.learning_rate == pytest.approx(0.05)


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
