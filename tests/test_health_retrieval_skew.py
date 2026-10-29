from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from living_memory.config import MemoryConfig, RetrievalSkewThresholds, load_config
from living_memory.models import RetrievalPolicyFloors, RetrievalWeights
from living_memory.resources import memory_health, retrieval_skew_metrics
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore


def _risk_by_scope(report: dict[str, object], scope: str) -> dict[str, object]:
    skew = report["retrieval_skew"]
    assert isinstance(skew, dict)
    risks = skew["scopes_at_risk"]
    assert isinstance(risks, list)
    matches = [risk for risk in risks if risk["scope"] == scope]
    assert len(matches) == 1
    return matches[0]


def test_memory_health_flags_deliberately_skewed_project_scope(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.set_retrieval_weights(
            "project:lopsided",
            bm25=0.98,
            vector=0.02,
            graph=0.0,
        )

        report = memory_health(store)

    risk = _risk_by_scope(report, "project:lopsided")
    assert risk["family"] == "project"
    assert risk["flags"] == ["bm25_monoculture", "near_zero_vector", "near_zero_graph"]
    assert risk["raw"] == {
        "bm25": pytest.approx(0.98),
        "vector": pytest.approx(0.02),
        "graph": pytest.approx(0.0),
    }

    assert risk["floor_violation"] == {}
    assert risk["effective_after_floor"] == risk["raw"]
    assert risk["applicable_floors"] == {
        "bm25_max": 1.0,
        "vector_min": 0.0,
        "graph_min": 0.0,
        "bm25_min": 0.10,
    }


def test_memory_health_does_not_flag_balanced_project_scope(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.set_retrieval_weights(
            "project:balanced",
            bm25=0.65,
            vector=0.30,
            graph=0.05,
        )

        report = memory_health(store, scope="project:balanced")

    skew = report["retrieval_skew"]
    assert skew["scopes_at_risk"] == []


def test_retrieval_skew_reports_thresholds_and_floors_in_effect(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        skew = memory_health(store)["retrieval_skew"]

    assert skew["thresholds"] == {
        "bm25_monoculture": pytest.approx(0.90),
        "near_zero_vector": pytest.approx(0.10),
        "near_zero_graph": pytest.approx(0.05),
    }
    assert skew["floors_in_effect"]["project"] == {
        "bm25_max": pytest.approx(0.85),
        "vector_min": pytest.approx(0.15),
        "graph_min": pytest.approx(0.05),
        "bm25_min": pytest.approx(0.10),
    }
    assert skew["floors_in_effect"]["global"] == {
        "bm25_max": pytest.approx(0.75),
        "vector_min": pytest.approx(0.20),
        "graph_min": pytest.approx(0.05),
        "bm25_min": pytest.approx(0.10),
    }


def test_retrieval_skew_thresholds_load_from_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "memory.toml"
    config_path.write_text(
        """
[retrieval_skew_thresholds]
bm25_monoculture = 0.70
near_zero_vector = 0.25
near_zero_graph = 0.08
""".strip(),
        encoding="utf-8",
    )
    config = load_config(config_path)

    assert config.retrieval_skew_thresholds == RetrievalSkewThresholds(
        bm25_monoculture=0.70,
        near_zero_vector=0.25,
        near_zero_graph=0.08,
    )

    with MemoryStore(
        MemoryConfig(
            db_path=tmp_path / "configured.sqlite3",
            retrieval_skew_thresholds=config.retrieval_skew_thresholds,
        )
    ) as store:
        store.set_retrieval_weights(
            "project:borderline",
            bm25=0.74,
            vector=0.20,
            graph=0.06,
        )
        risk = _risk_by_scope(memory_health(store), "project:borderline")

    assert risk["flags"] == ["bm25_monoculture", "near_zero_vector", "near_zero_graph"]


def test_recent_feedback_pressure_counts_dominant_methods(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        trace = store.append_trace(
            "feedback pressure fixture",
            {"scope": "project:pressure", "agent": "test-agent"},
        )
        event = store.record_recall_event(
            query="pressure",
            scope="project:pressure",
            results=[
                {
                    "node_id": trace.id,
                    "bm25_score": 1.0,
                    "vector_score": 0.2,
                    "graph_score": 0.0,
                    "methods": ["bm25", "vector"],
                },
                {
                    "node_id": trace.id,
                    "bm25_score": 0.1,
                    "vector_score": 0.9,
                    "graph_score": 0.0,
                    "methods": ["vector"],
                },
                {
                    "node_id": trace.id,
                    "bm25_score": 0.0,
                    "vector_score": 0.0,
                    "graph_score": 0.8,
                    "methods": ["graph"],
                },
            ],
        )
        store.mark_recall_event_feedback(event.id, trace.id)

        pressure = retrieval_skew_metrics(store, scope="project:pressure")[
            "recent_feedback_pressure"
        ]
        empty = retrieval_skew_metrics(store, scope="project:empty")[
            "recent_feedback_pressure"
        ]

    assert pressure["window_hours"] == 24
    assert pressure["consumed_recall_events"] == 1
    assert pressure["dominant_method_counts"] == {"bm25": 1, "vector": 1, "graph": 1}
    assert pressure["bm25_dominance_ratio"] == pytest.approx(1 / 3)

    assert empty["consumed_recall_events"] == 0
    assert empty["dominant_method_counts"] == {"bm25": 0, "vector": 0, "graph": 0}
    assert empty["bm25_dominance_ratio"] is None


# Expected distributions are independent of the production floor calculation.
@pytest.mark.parametrize(
    "policy,evidence,raw,expected,violations",
    [
        ("default", "none", (0.98, 0.02, 0), (0.98, 0.02, 0), set()),
        ("default", "vector", (0.98, 0.02, 0), (0.85, 0.15, 0), {"bm25", "vector"}),
        ("default", "graph", (0.98, 0.02, 0), (0.85, 0.02, 0.13), {"bm25", "graph"}),
        ("default", "both", (0.98, 0.02, 0), (0.80, 0.15, 0.05), {"bm25", "vector", "graph"}),
        ("custom", "none", (0.98, 0.02, 0), (0.98, 0.02, 0), set()),
        ("custom", "vector", (0.98, 0.02, 0), (0.60, 0.40, 0), {"bm25", "vector"}),
        ("custom", "graph", (0.98, 0.02, 0), (0.60, 0.02, 0.38), {"bm25", "graph"}),
        ("custom", "both", (0.98, 0.02, 0), (0.60, 0.30, 0.10), {"bm25", "vector", "graph"}),
        ("empty", "none", (0.98, 0.02, 0), (0.98, 0.02, 0), set()),
        ("empty", "vector", (0.98, 0.02, 0), (0.98, 0.02, 0), set()),
        ("empty", "graph", (0.98, 0.02, 0), (0.98, 0.02, 0), set()),
        ("empty", "both", (0.98, 0.02, 0), (0.98, 0.02, 0), set()),
        ("default", "none", (0, 0, 1), (0.10, 0, 0.90), {"bm25"}),
        ("default", "vector", (0, 0, 1), (0.10, 0.15, 0.75), {"bm25", "vector"}),
        ("default", "graph", (0, 0, 1), (0.10, 0, 0.90), {"bm25"}),
        ("default", "both", (0, 0, 1), (0.10, 0.15, 0.75), {"bm25", "vector"}),
        ("custom", "none", (0, 0, 1), (0.20, 0, 0.80), {"bm25"}),
        ("zero_minima", "both", (0.98, 0.02, 0), (0.98, 0.02, 0), set()),
        ("zero_minima", "none", (0.98, 0.02, 0), (0.98, 0.02, 0), set()),
        ("lexical_only", "both", (0, 0, 1), (0.20, 0, 0.80), {"bm25"}),
        ("lexical_only", "none", (0, 0, 1), (0.20, 0, 0.80), {"bm25"}),
        ("vector_only", "both", (0.98, 0.02, 0), (0.60, 0.40, 0), {"bm25", "vector"}),
        ("vector_only", "graph", (0.98, 0.02, 0), (0.98, 0.02, 0), set()),
        ("graph_only", "both", (0.98, 0.02, 0), (0.60, 0.02, 0.38), {"bm25", "graph"}),
        ("graph_only", "vector", (0.98, 0.02, 0), (0.98, 0.02, 0), set()),
        ("other_family", "both", (0.98, 0.02, 0), (0.98, 0.02, 0), set()),
        ("default", "both", (98, 2, 0), (0.80, 0.15, 0.05), {"bm25", "vector", "graph"}),
        ("default", "both", (0, 0, 0), (0.80, 0.15, 0.05), {"bm25", "vector", "graph"}),
    ],
)
def test_native_health_matches_floor_application_and_feedback_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
    evidence: str,
    raw: tuple[float, float, float],
    expected: tuple[float, float, float],
    violations: set[str],
) -> None:
    from fastmcp import Client

    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    config = MemoryConfig(db_path=tmp_path / "native.sqlite3")
    custom = RetrievalPolicyFloors(0.60, 0.25, 0.10, 0.20)
    if policy == "custom":
        config = replace(config, retrieval_policy_floors={"project": custom})
    elif policy == "empty":
        config = replace(config, retrieval_policy_floors={})
    elif policy == "zero_minima":
        config = replace(
            config, retrieval_policy_floors={"project": RetrievalPolicyFloors(0.60, 0, 0)},
        )
    elif policy == "other_family":
        config = replace(config, retrieval_policy_floors={"global": custom})
    elif policy in {"lexical_only", "vector_only", "graph_only"}:
        config = replace(config, retrieval_policy_floors={"project": RetrievalPolicyFloors(
            bm25_max=0.60,
            vector_min=0.25 if policy == "vector_only" else 0,
            graph_min=0.10 if policy == "graph_only" else 0,
            bm25_min=0.20,
        )})
    mcp = create_mcp_server(config=config, expose_operator_tools=True)
    store = mcp.memory_store
    scope = "project:health"
    try:
        # Other scopes and families cannot enable floors in this scope.
        _seed_floor_evidence(store, "project:other", "both")
        _seed_floor_evidence(store, "global", "both")
        _seed_floor_evidence(store, scope, evidence)
        assert store._has_vector_evidence(scope) == (evidence in {"vector", "both"})
        assert store._has_graph_evidence(scope) == (evidence in {"graph", "both"})
        evidence_reads = {}
        for channel in ("vector", "graph"):
            name = f"_has_{channel}_evidence"
            evidence_reads[channel] = Mock(wraps=getattr(store, name))
            monkeypatch.setattr(store, name, evidence_reads[channel])
        saved = store.set_retrieval_weights(
            scope, bm25=raw[0], vector=raw[1], graph=raw[2], learning_rate=0.07,
        )
        assert _triplet(saved) == raw
        store.set_retrieval_weights("project:other", bm25=1, vector=0, graph=0)
        before = list(store.connection.iterdump())
        changes = store.connection.total_changes
        store.connection.execute("PRAGMA query_only = ON")

        async def health() -> dict[str, object]:
            async with Client(mcp) as client:
                response = await client.call_tool("memory_health", {"scope": scope})
                assert not response.is_error
                return response.data

        report = asyncio.run(health())
        configured = config.retrieval_policy_floors.get("project")
        for channel, read in evidence_reads.items():
            enabled = configured is not None and getattr(configured, f"{channel}_min") > 0
            assert read.call_count == int(enabled)
            if enabled:
                read.assert_called_with(scope)
        assert store.connection.total_changes == changes
        assert list(store.connection.iterdump()) == before
        store.connection.execute("PRAGMA query_only = OFF")
        assert store.get_retrieval_weights(scope) == saved
        skew = report["retrieval_skew"]
        assert skew["floors_in_effect"] == {
            family: asdict(floor) for family, floor in config.retrieval_policy_floors.items()
        }
        assert [risk["scope"] for risk in skew["scopes_at_risk"]] == [scope]
        risk = _risk_by_scope(report, scope)
        assert tuple(risk["raw"].values()) == pytest.approx(_triplet(saved.normalized()))
        assert tuple(risk["effective_after_floor"].values()) == pytest.approx(expected)
        assert set(risk["floor_violation"]) == violations
        configured = config.retrieval_policy_floors.get("project")
        if configured is None:
            assert risk["applicable_floors"] == {}
        else:
            vector_min = configured.vector_min if evidence in {"vector", "both"} else 0
            graph_min = configured.graph_min if evidence in {"graph", "both"} else 0
            assert risk["applicable_floors"] == {
                "bm25_max": configured.bm25_max if vector_min or graph_min else 1.0,
                "vector_min": vector_min,
                "graph_min": graph_min,
                "bm25_min": configured.bm25_min,
            }
        applied = store.apply_retrieval_weight_floors(scope, saved)
        updated = store.update_retrieval_weights(scope)
        for channel, read in evidence_reads.items():
            enabled = configured is not None and getattr(configured, f"{channel}_min") > 0
            assert read.call_count == 3 * int(enabled)
        assert _triplet(applied) == pytest.approx(expected)
        assert _triplet(updated) == pytest.approx(expected)
        assert _triplet(store.get_retrieval_weights(scope)) == pytest.approx(expected)
        assert updated.learning_rate == saved.learning_rate
        assert sum(_triplet(updated)) == pytest.approx(1.0)
        assert _triplet(store.get_retrieval_weights("project:other")) == (1, 0, 0)
    finally:
        mcp.auto_consolidation.close()
        store.close()


def _triplet(weights: RetrievalWeights) -> tuple[float, float, float]:
    return weights.bm25, weights.vector, weights.graph


def _seed_floor_evidence(store: MemoryStore, scope: str, evidence: str) -> None:
    if evidence == "none":
        return
    source = store.create_node(
        level="trace", content="floor source", context={"scope": scope},
        embedding=[1.0, 0.0] if evidence in {"vector", "both"} else None,
    )
    if evidence in {"graph", "both"}:
        target = store.create_node(
            level="trace", content="floor target", context={"scope": scope},
        )
        store.create_connection(source.id, target.id, "caused")
