from __future__ import annotations

from pathlib import Path

import pytest

from living_memory.config import MemoryConfig, RetrievalSkewThresholds, load_config
from living_memory.resources import memory_health, retrieval_skew_metrics
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
    assert risk["flags"] == ["bm25_monoculture", "near_zero_vector"]
    assert risk["raw"] == {
        "bm25": pytest.approx(0.98),
        "vector": pytest.approx(0.02),
        "graph": pytest.approx(0.0),
    }

    violation = risk["floor_violation"]
    assert violation["bm25"]["ceiling"] == pytest.approx(0.85)
    assert violation["vector"]["floor"] == pytest.approx(0.15)
    assert risk["effective_after_floor"] == {
        "bm25": pytest.approx(0.85),
        "vector": pytest.approx(0.15),
        "graph": pytest.approx(0.0),
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
        "graph_min": pytest.approx(0.0),
    }
    assert skew["floors_in_effect"]["global"] == {
        "bm25_max": pytest.approx(0.75),
        "vector_min": pytest.approx(0.20),
        "graph_min": pytest.approx(0.05),
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

    assert risk["flags"] == ["bm25_monoculture", "near_zero_vector"]


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
