"""Final recall selection with a present supersedes correction."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from living_memory.models import Node
from living_memory.retrieval import MemoryRecallService, RecallResult
from living_memory.score_gate import apply_score_gate
from living_memory.storage import MemoryStore


class _Encoder:
    def __init__(self, original_vector: float = 0.8):
        self.original_vector = original_vector

    def embed(self, text: str) -> list[float]:
        return {
            "bystander": [1.0, 0.0],
            "stale": [self.original_vector, math.sqrt(1 - self.original_vector**2)],
            "correction": [0.1, math.sqrt(0.99)],
        }.get(text, [1.0, 0.0])


def _recall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    threshold: str = "0.35",
    form: str = "drop",
    edge: bool = True,
    correction_scope: str = "project:app",
    decayed: bool = False,
    max_results: int = 3,
    original_vector: float = 0.8,
) -> tuple[list[str], list[str], list[str | None]]:
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", threshold)
    monkeypatch.setenv("LM_RECALL_GATE_FORM", form)
    monkeypatch.delenv("LM_RECALL_SCHEMA_TRIGGER", raising=False)
    monkeypatch.delenv("LM_RECALL_SCHEMA_DEDUP", raising=False)
    encoder = _Encoder(original_vector)
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        store.set_chunk_embedder(lambda texts: [encoder.embed(text) for text in texts])
        for name in ("bystander", "stale", "correction"):
            store.create_node(
                level="trace",
                content=name,
                context={"scope": correction_scope if name == "correction" else "project:app"},
                node_id=name,
                embedding=encoder.embed(name),
                stats={
                    # Deliberately entrenched original; this is not the
                    # typical state immediately after memory_teach.
                    "confidence": 1.0,
                    "usefulness_score": 1.0,
                    "access_count": 10000,
                    "unique_agents": 2,
                }
                if name == "stale"
                else None,
            )
        if edge:
            store.create_connection("correction", "stale", "supersedes")
        store.create_connection("stale", "bystander", "caused")
        if decayed:
            store.soft_delete_node("correction", "test")
        service = MemoryRecallService(store, embedder=encoder)
        delivered = service.memory_recall(
            "public-query",
            scope="project:app",
            depth="causal",
            max_results=max_results,
            log_access=False,
            log_event=False,
        )
        return (
            [result.node.id for result in delivered],
            [result.node.id for result in service.last_residual],
            [result.withheld for result in delivered],
        )


def test_public_recall_keeps_present_correction_when_original_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Strong current vector and causal source evidence pass after the
    # superseded penalty; accumulated feedback does not qualify the correction.
    assert _recall(tmp_path, monkeypatch, original_vector=0.99) == (
        ["bystander", "correction", "stale"], [], [None, None, None]
    )


def test_history_does_not_qualify_a_weak_corrected_source(tmp_path, monkeypatch):
    assert _recall(tmp_path, monkeypatch) == (
        ["bystander", "correction", "stale"], [],
        [None, "below_threshold", "below_threshold"],
    )


def test_correction_selection_respects_cap_and_stub_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _recall(tmp_path / "cap", monkeypatch, max_results=2, original_vector=0.99) == (
        ["bystander", "correction"], ["stale"], [None, None]
    )
    assert _recall(tmp_path / "stub", monkeypatch, form="stub", original_vector=0.99) == (
        ["bystander", "correction", "stale"], [], [None, None, None]
    )


@pytest.mark.parametrize("case", ["absent_edge", "outside_scope", "decayed"])
def test_unavailable_correction_does_not_change_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    delivered, residual, withheld = _recall(
        tmp_path,
        monkeypatch,
        edge=case != "absent_edge",
        correction_scope="project:elsewhere" if case == "outside_scope" else "project:app",
        decayed=case == "decayed",
        original_vector=0.99,
    )
    assert {node_id for node_id, reason in zip(delivered, withheld) if not reason} == {
        "bystander", "stale",
    }
    if case == "absent_edge":
        assert delivered[-1] == "correction"
        assert withheld[-1] == "below_threshold"
    else:
        assert delivered == ["bystander", "stale"]
        assert "correction" not in residual


def test_gate_withholds_weak_connected_pair_but_keeps_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _recall(tmp_path, monkeypatch, threshold="0.6") == (
        ["bystander", "correction", "stale"], [],
        [None, "below_threshold", "below_threshold"],
    )


def test_gate_off_preserves_ranked_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _recall(tmp_path, monkeypatch, threshold="") == (
        ["bystander", "correction", "stale"], [], [None, None, None]
    )


def test_present_correction_chain_propagates_relevance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "0.3")
    monkeypatch.setenv("LM_RECALL_GATE_FORM", "drop")
    ranked = [
        RecallResult(
            node=Node(id=name, level="trace", content=name, scope="project:app"),
            score=score,
            bm25_score=score,
            lexical_score=score,
        )
        for name, score in (
            ("bystander", 1.0),
            ("newest", 0.1),
            ("older", 0.1),
            ("stale", 1.0),
        )
    ]
    delivered, residual = apply_score_gate(
        ranked,
        3,
        corrections_by_superseded={"stale": ("older",), "older": ("newest",)},
    )
    assert [result.node.id for result in delivered] == ["bystander", "newest", "older"]
    assert all(result.withheld is None for result in delivered)
    assert [result.node.id for result in residual] == ["stale"]
