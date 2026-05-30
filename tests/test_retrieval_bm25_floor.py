"""Regression tests for the bm25 retrieval-weight floor.

Before the floor existed, winner-take-all feedback (feedback._method_signals)
drove the bm25 weight to exactly 0 for vector-dominant scopes, which silently
disabled exact-keyword / identifier recall. These tests pin the floor so bm25
can never be competed all the way out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from living_memory.models import RetrievalWeights
from living_memory.storage import MemoryStore


def test_floor_lifts_fully_collapsed_bm25(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        source = store.create_node(
            level="trace",
            content="alpha lexical token",
            context={"scope": "global"},
            embedding=[1.0, 0.0],
        )
        target = store.create_node(
            level="trace",
            content="beta downstream effect",
            context={"scope": "global"},
            embedding=[0.0, 1.0],
        )
        store.create_connection(source.id, target.id, "caused")

        floored = store.apply_retrieval_weight_floors(
            "global",
            RetrievalWeights(scope="global", bm25=0.0, vector=1.0, graph=0.0),
        )

    # global floors: bm25_min=0.10, vector_min=0.20, graph_min=0.05, bm25_max=0.75
    assert floored.bm25 == pytest.approx(0.10)
    assert floored.vector == pytest.approx(0.85)
    assert floored.graph == pytest.approx(0.05)
    assert floored.bm25 + floored.vector + floored.graph == pytest.approx(1.0)


def test_repeated_vector_dominant_feedback_never_zeroes_bm25(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        # Vector evidence only (no connection) — mirrors the real global corpus
        # where graph evidence may be absent but bm25 must still be protected.
        store.create_node(
            level="trace",
            content="semantic content for vector evidence",
            context={"scope": "global"},
            embedding=[1.0, 0.0],
        )
        store.set_retrieval_weights("global", bm25=0.4, vector=0.4, graph=0.2)

        # Drive 100 rounds of vector-dominant feedback: vector rewarded, the
        # other two channels penalised (the −0.25 winner-take-all signal).
        for _ in range(100):
            store.update_retrieval_weights(
                "global",
                bm25_signal=-0.25,
                vector_signal=1.0,
                graph_signal=-0.25,
            )

        weights = store.get_retrieval_weights("global").normalized()

    # Without the floor bm25 settles at exactly 0.0; with it bm25 stays pinned
    # at its floor and keeps contributing to ranking.
    assert weights.bm25 >= 0.10 - 1e-9
    assert weights.vector > weights.bm25  # vector still dominates, just not total
