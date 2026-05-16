from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest

from living_memory.config import MemoryConfig
from living_memory.consolidation import memory_consolidate, memory_teach
from living_memory.decay import apply_decay
from living_memory.feedback import apply_retrieval_feedback
from living_memory.prompts import retrieval_context_prompt
from living_memory.retrieval import RecallResult, memory_connect, memory_recall
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore


class RecordingMCP:
    def __init__(self, name: str, instructions: str | None = None) -> None:
        self.name = name
        self.instructions = instructions
        self.tools: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def tool(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.tools[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        if func is None:
            return decorate
        return decorate(func)

    def resource(self, uri: str, **_kwargs: Any) -> Any:
        def decorate(func: Any) -> Any:
            self.resources[uri] = func
            return func

        return decorate

    def prompt(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.prompts[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        if func is None:
            return decorate
        return decorate(func)


def test_cold_start_recall_is_empty_and_under_50ms(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        started = perf_counter()
        results = memory_recall(store, "anything remembered yet", max_results=5)
        elapsed = perf_counter() - started

        assert results == []
        assert elapsed < 0.050


def test_thousand_trace_semantic_recall_finds_relevant_record(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        relevant = store.append_trace(
            "Authentication slowdown in production login was fixed by token cache warmup",
            {"scope": "project:alpha", "agent": "agent-a"},
            feedback={"confidence": 0.5, "usefulness_score": 0.8},
        )
        for index in range(999):
            store.append_trace(
                f"billing export notification background record {index}",
                {"scope": "project:alpha", "agent": "agent-b"},
            )

        results = memory_recall(
            store,
            "auth timeout prod signin",
            scope="project:alpha",
            max_results=5,
        )

        assert store.trace_count() == 1000
        assert results
        assert results[0].node.id == relevant.id
        assert {"bm25", "vector"} & set(results[0].methods)


def test_mcp_remember_automatically_creates_concept_with_temporal_consensus_and_prompt(
    tmp_path: Path,
) -> None:
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=RecordingMCP)

    final_remember: dict[str, Any] | None = None
    for index in range(100):
        final_remember = mcp.tools["memory_remember"](
            f"weekly monday deploy rollback requires migration guard before release {index}",
            {
                "scope": "project:alpha",
                "agent": "agent-a" if index % 2 == 0 else "agent-b",
                "timestamp": "2026-05-04T09:00:00Z",
            },
            {"confidence": 0.5, "usefulness_score": 0.3},
        )

    assert final_remember is not None
    auto = final_remember["auto_consolidation"]
    assert auto is not None
    assert len(auto["concepts_created"]) == 1
    concept = auto["concepts_created"][0]
    assert concept["level"] == "concept"
    assert concept["scope"] == "project:alpha"
    assert len(concept["provenance"]["source_traces"]) == 100
    assert concept["stats"]["unique_agents"] == 2
    assert concept["stats"]["confidence"] > 0.5
    assert concept["stats"]["temporal_hint"] == "weekly:mon"

    block = mcp.prompts["memory://prompt/retrieval_context"](
        task="deploy rollback migration",
        scope="project:alpha",
        max_concepts=3,
        min_confidence=0.5,
        retrieval_policy="confidence",
    )
    assert block.startswith("BEGIN ACTIVE MEMORY CONTEXT")
    assert concept["id"] in block
    assert "scope_plan: project:alpha > global" in block
    assert "retrieval_policy: confidence" in block
    assert block.endswith("END ACTIVE MEMORY CONTEXT")


def test_adaptive_policy_consolidates_young_scope_at_low_thresholds(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "adaptive")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=RecordingMCP)

    triggers: list[int] = []
    last_with_concept: dict[str, Any] | None = None
    for index in range(15):
        result = mcp.tools["memory_remember"](
            f"weekly monday deploy rollback requires migration guard before release {index}",
            {
                "scope": "project:young",
                "agent": "agent-a" if index % 2 == 0 else "agent-b",
                "timestamp": "2026-05-04T09:00:00Z",
            },
            {"confidence": 0.5, "usefulness_score": 0.3},
        )
        if result["auto_consolidation"] is not None:
            triggers.append(index + 1)
            if result["auto_consolidation"]["concepts_created"]:
                last_with_concept = result

    # Adaptive ladder: trigger at count = 5, 10, 15 (step=5 while count<50).
    assert triggers == [5, 10, 15]
    assert last_with_concept is not None
    concept = last_with_concept["auto_consolidation"]["concepts_created"][0]
    # Merge floor for trace_count<25 is 3; cluster of similar traces meets it.
    assert concept["level"] == "concept"
    assert concept["scope"] == "project:young"
    assert len(concept["provenance"]["source_traces"]) >= 3


def test_fixed_policy_remains_default_and_skips_low_volume_consolidation(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    monkeypatch.delenv("LM_AUTO_CONSOLIDATE_POLICY", raising=False)
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=RecordingMCP)

    for index in range(20):
        result = mcp.tools["memory_remember"](
            f"weekly monday deploy rollback requires migration guard before release {index}",
            {"scope": "project:young", "agent": "agent-a"},
            {"confidence": 0.5},
        )
        assert result["auto_consolidation"] is None


def test_project_scope_isolation_keeps_other_projects_out_of_recall(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        alpha = store.append_trace(
            "Alpha checkout deploy uses migration 42",
            {"scope": "project:alpha", "agent": "agent-a"},
        )
        beta = store.append_trace(
            "Beta checkout deploy uses migration 99",
            {"scope": "project:beta", "agent": "agent-b"},
        )
        global_trace = store.append_trace(
            "Every project runs migration checks before checkout deploy",
            {"scope": "global", "agent": "agent-c"},
        )

        results = memory_recall(
            store,
            "checkout deploy migration",
            scope="project:alpha",
            max_results=10,
        )
        ids = {result.node.id for result in results}

        assert alpha.id in ids
        assert global_trace.id in ids
        assert beta.id not in ids


def test_corrected_fact_ranks_above_original_and_decay_is_soft(tmp_path: Path) -> None:
    config = MemoryConfig(db_path=tmp_path / "memory.sqlite3", trace_ttl_days=7)
    with MemoryStore(config) as store:
        original = store.append_trace(
            "payment retry limit is 3 attempts",
            {
                "scope": "project:billing",
                "agent": "agent-a",
                "timestamp": "2026-05-14T00:00:00Z",
            },
            feedback={"confidence": 0.5, "usefulness_score": 0.1},
        )
        taught = memory_teach(
            store,
            original.id,
            "payment retry limit is 5 attempts",
            context={"agent": "agent-b"},
        )

        ranked = memory_recall(
            store,
            "payment retry limit attempts",
            scope="project:billing",
            depth=1,
            max_results=5,
        )
        assert ranked[0].node.id == taught.corrective_trace.id
        assert original.id in {result.node.id for result in ranked}

        expired = store.append_trace(
            "temporary billing observation should age out",
            {
                "scope": "project:billing",
                "agent": "agent-a",
                "timestamp": "2020-01-01T00:00:00Z",
            },
        )
        decayed = apply_decay(store, now=datetime(2026, 5, 14, tzinfo=UTC))
        decayed_ids = {node.id for node in decayed.nodes}

        assert original.id in decayed_ids
        assert expired.id in decayed_ids
        assert store.get_node(original.id) is not None
        assert store.get_node(original.id).decayed is True
        assert store.get_node(expired.id) is not None
        assert store.get_node(expired.id).decayed is True


def test_causal_traversal_returns_causes_for_why_queries(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        cause = store.append_trace(
            "Missing migration file caused checkout deploy incident",
            {"scope": "project:alpha", "agent": "agent-a"},
        )
        effect = store.append_trace(
            "Checkout deploy incident happened during release",
            {"scope": "project:alpha", "agent": "agent-b"},
        )
        memory_connect(store, cause.id, effect.id, "caused", weight=1.0)

        results = memory_recall(
            store,
            "why did checkout deploy incident happen",
            scope="project:alpha",
            depth="causal",
            max_results=5,
        )

        assert results[0].node.id == cause.id
        assert results[0].graph_score > 0.0
        assert cause.id in results[0].path


def test_consensus_confidence_caps_single_agent_and_rises_with_multiple_agents(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for index in range(100):
            store.append_trace(
                f"solo release checklist consensus fact stable wording {index}",
                {"scope": "project:solo", "agent": "agent-a"},
                feedback={"confidence": 0.95, "usefulness_score": 0.7},
            )
            store.append_trace(
                f"team release checklist consensus fact stable wording {index}",
                {"scope": "project:team", "agent": "agent-a" if index % 2 else "agent-b"},
                feedback={"confidence": 0.95, "usefulness_score": 0.7},
            )

        solo = memory_consolidate(store, scope="project:solo").concepts_created[0]
        team = memory_consolidate(store, scope="project:team").concepts_created[0]

        assert solo.unique_agents == 1
        assert solo.confidence == pytest.approx(0.5)
        assert team.unique_agents == 2
        assert team.confidence > solo.confidence


def test_retrieval_weights_tune_after_100_feedback_cycles(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = store.append_trace(
            "Authentication latency fixed by token cache warmup",
            {"scope": "project:alpha", "agent": "agent-a"},
        )
        result = RecallResult(
            node=node,
            score=0.9,
            bm25_score=0.0,
            vector_score=0.95,
            graph_score=0.0,
            methods=("vector",),
        )
        before = store.get_retrieval_weights("project:alpha").normalized()

        update = None
        for _index in range(100):
            update = apply_retrieval_feedback(
                store,
                result,
                useful=True,
                signal=1.0,
                scope="project:alpha",
            )

        assert update is not None
        after = update.weights.normalized()
        assert (after.bm25, after.vector, after.graph) != pytest.approx(
            (before.bm25, before.vector, before.graph)
        )
        assert after.vector > before.vector


def test_prompt_context_generation_filters_and_formats_top_concepts(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        selected = store.create_node(
            level="concept",
            content="Alpha deploy rollback requires migration dry run before release",
            context={"scope": "project:alpha", "agent": "agent-a"},
            stats={"confidence": 0.92, "unique_agents": 2, "usefulness_score": 0.6},
        )
        low_confidence = store.create_node(
            level="concept",
            content="Alpha deploy rollback can skip migration checks",
            context={"scope": "project:alpha", "agent": "agent-b"},
            stats={"confidence": 0.2, "unique_agents": 1, "usefulness_score": -0.3},
        )
        other_project = store.create_node(
            level="concept",
            content="Beta deploy rollback requires migration dry run before release",
            context={"scope": "project:beta", "agent": "agent-c"},
            stats={"confidence": 0.95, "unique_agents": 2, "usefulness_score": 0.9},
        )

        block = retrieval_context_prompt(
            store,
            task="deploy rollback migration",
            scope="project:alpha",
            max_concepts=3,
            min_confidence=0.5,
            retrieval_policy="confidence",
        )

        assert selected.id in block
        assert selected.content in block
        assert low_confidence.id not in block
        assert other_project.id not in block
        assert "BEGIN ACTIVE MEMORY CONTEXT" in block
        assert "END ACTIVE MEMORY CONTEXT" in block
