from __future__ import annotations

from pathlib import Path

import pytest

from living_memory.models import Node
from living_memory.retrieval import MemoryRecallService, RecallResult
from living_memory.storage import MemoryStore


_EXACT_IDENTIFIER_CASES = (
    pytest.param(
        "src/living_memory/retrieval.py",
        "Hybrid recall scoring code path: src/living_memory/retrieval.py owns "
        "rank_candidates and _collect_bm25.",
        (
            "Hybrid recall scoring notes mention src/living_memory/resources.py "
            "for memory_health rendering.",
            "Ranking weights combine BM25 vector and graph in the retrieval service.",
        ),
        id="path",
    ),
    pytest.param(
        "TraceInspectorPanel",
        "TraceInspectorPanel renders the selected trace metadata table.",
        (
            "TraceInspectorSidebar renders selected trace navigation filters.",
            "Trace inspection panel renders selected metadata without naming the component.",
        ),
        id="component",
    ),
    pytest.param(
        "EZ-13771",
        "Ticket EZ-13771 owns the tooltip marker experiment rollout and screenshot checks.",
        (
            "Ticket EZ-13772 owns the tooltip marker experiment rollout and screenshot checks.",
            "The tooltip marker experiment needs rollout screenshots and checks.",
        ),
        id="ticket-key",
    ),
    pytest.param(
        "06f6422",
        "Commit 06f6422 widened implicit recall feedback matching to five pending events.",
        (
            "Commit 06f6423 adjusted implicit recall feedback matching tests.",
            "Implicit recall feedback matching now consumes several pending events.",
        ),
        id="hash",
    ),
    pytest.param(
        "storybook dev --ci",
        "Run storybook dev --ci before Playwright screenshot capture for the tooltip stories.",
        (
            "Run storybook build --ci before publishing the tooltip stories.",
            "Run npm run check before Playwright screenshot capture for tooltip stories.",
        ),
        id="storybook-command",
    ),
)


_HOLDOUT_IDENTIFIER_CASES = (
    pytest.param(
        "packages/ui/src/components/PolicyFloorGauge.tsx",
        "packages/ui/src/components/PolicyFloorGauge.tsx renders retrieval floor badges.",
        (
            "packages/ui/src/components/PolicyWeightGauge.tsx renders retrieval weight badges.",
            "Retrieval floor badges render in the policy detail view.",
        ),
        id="holdout-path-component-file",
    ),
    pytest.param(
        "LM_RETRIEVAL_TUNING_POLICY",
        "Set LM_RETRIEVAL_TUNING_POLICY=adaptive only when testing adaptive feedback rates.",
        (
            "Set LM_RETRIEVAL_TRACE_LIMIT=adaptive only when testing trace list limits.",
            "Adaptive feedback rates are controlled by a retrieval tuning environment variable.",
        ),
        id="holdout-env-var",
    ),
    pytest.param(
        "01KS7V1J3T42BHW7H6401YMM23",
        "Trace 01KS7V1J3T42BHW7H6401YMM23 documents the storage floor enforcement gate repair.",
        (
            "Trace 01KS7V1J3T42BHW7H6401YMM24 documents a health skew visibility repair.",
            "The storage floor enforcement gate repair is documented in a trace.",
        ),
        id="holdout-ulid",
    ),
)


@pytest.mark.parametrize("query,target_content,decoy_contents", _EXACT_IDENTIFIER_CASES)
def test_exact_identifier_queries_rank_exact_trace_first_under_project_floor(
    tmp_path: Path,
    query: str,
    target_content: str,
    decoy_contents: tuple[str, ...],
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        target, service = _seed_identifier_case(
            store,
            target_content=target_content,
            decoy_contents=decoy_contents,
        )

        results = service.memory_recall(
            query,
            scope="project:lexical",
            depth=0,
            max_results=3,
            log_access=False,
            log_event=False,
        )

        assert results
        assert results[0].node.id == target.id, _ranked_contents(results)
        assert results[0].bm25_score == pytest.approx(1.0)
        assert "bm25" in results[0].methods


@pytest.mark.parametrize("query,target_content,decoy_contents", _HOLDOUT_IDENTIFIER_CASES)
def test_identifier_holdouts_remain_reachable_under_project_floor(
    tmp_path: Path,
    query: str,
    target_content: str,
    decoy_contents: tuple[str, ...],
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        target, service = _seed_identifier_case(
            store,
            target_content=target_content,
            decoy_contents=decoy_contents,
        )

        results = service.memory_recall(
            query,
            scope="project:lexical",
            depth=0,
            max_results=3,
            log_access=False,
            log_event=False,
        )

        target_results = [result for result in results if result.node.id == target.id]
        assert target_results, _ranked_contents(results)
        assert target_results[0].bm25_score > 0.0
        assert "bm25" in target_results[0].methods


def _seed_identifier_case(
    store: MemoryStore,
    *,
    target_content: str,
    decoy_contents: tuple[str, ...],
) -> tuple[Node, MemoryRecallService]:
    scope = "project:lexical"
    target = store.append_trace(
        target_content,
        {"scope": scope, "agent": "lexical-test"},
    )
    for index, decoy_content in enumerate(decoy_contents):
        store.append_trace(
            decoy_content,
            {"scope": scope, "agent": "lexical-test", "decoy_index": index},
        )

    service = MemoryRecallService(store)
    service.memory_recall(
        "embedding warmup",
        scope=scope,
        depth=0,
        max_results=1,
        log_access=False,
        log_event=False,
    )
    store.set_retrieval_weights(
        scope,
        bm25=1.0,
        vector=0.0,
        graph=0.0,
        learning_rate=0.05,
    )
    floored = store.update_retrieval_weights(
        scope,
        bm25_signal=1.0,
        vector_signal=-1.0,
    ).normalized()
    assert floored.bm25 == pytest.approx(0.85)
    assert floored.vector == pytest.approx(0.15)
    assert floored.graph == pytest.approx(0.0)

    return target, service


def _ranked_contents(results: list[RecallResult]) -> list[tuple[str, float, float, float]]:
    return [
        (
            result.node.content,
            result.score,
            result.bm25_score,
            result.vector_score,
        )
        for result in results
    ]
