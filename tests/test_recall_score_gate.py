"""Recall quality gate (``score_gate``): weak results do not take a full slot.

Pins docs/recall-score-gate.md: off is the plain cut; ``drop`` fills from
below with passing results and keeps everything else in the residual;
``stub`` marks failing results inside the cut and ``delivery`` renders them
content-less; rank 1 always ships in full; the gate score ignores the store's
retrieval weights; trigger-found schemas are gated on their own scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from living_memory.delivery import (
    DELIVERY_BELOW_THRESHOLD,
    shape_recall_results,
)
from living_memory.models import Node
from living_memory.retrieval import MemoryRecallService, RecallResult
from living_memory.score_gate import (
    GATE_FORM_ENV,
    MIN_SCORE_ENV,
    REFERENCE_WEIGHTS,
    SCHEMA_TRIGGER_GATE_MIN,
    WITHHELD_BELOW_THRESHOLD,
    apply_score_gate,
    gate_score,
    min_score_from_env,
    passes_gate,
    trigger_gate_score,
)
from living_memory.storage import MemoryStore

SCOPE = "project:gate"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MIN_SCORE_ENV, raising=False)
    monkeypatch.delenv(GATE_FORM_ENV, raising=False)


def make_node(node_id: str, content: str = "", **overrides: Any) -> Node:
    defaults: dict[str, Any] = {
        "id": node_id,
        "level": "trace",
        "content": content or f"content of {node_id}",
        "scope": SCOPE,
        "timestamp": "2026-09-29T00:00:00Z",
        "context": {"scope": SCOPE},
        "created_at": "2026-09-29T00:00:00Z",
        "updated_at": "2026-09-29T00:00:00Z",
    }
    defaults.update(overrides)
    return Node(**defaults)


def make_result(node_id: str, strength: float, **overrides: Any) -> RecallResult:
    """A result whose bm25 channel is ``strength`` (gate score 0.4 * strength)."""

    defaults: dict[str, Any] = {
        "node": make_node(node_id),
        "score": strength,
        "bm25_score": strength,
        "vector_score": 0.0,
        "methods": ("bm25",),
    }
    defaults.update(overrides)
    return RecallResult(**defaults)


def ids(results: list[RecallResult]) -> list[str]:
    return [result.node.id for result in results]


# Gate scores: a=0.4, b=0.36, c=0.08, d=0.32, e=0.04, f=0.28 at threshold 0.3
# the passing set is a, b, d.
RANKED = [
    make_result("a", 1.0),
    make_result("b", 0.9),
    make_result("c", 0.2),
    make_result("d", 0.8),
    make_result("e", 0.1),
    make_result("f", 0.7),
]


# --- valve parsing --------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "  ", "abc", "0", "0.0", "-0.4", "nan", "inf"])
def test_invalid_or_zero_threshold_is_off(
    monkeypatch: pytest.MonkeyPatch, raw: str | None
) -> None:
    if raw is not None:
        monkeypatch.setenv(MIN_SCORE_ENV, raw)
    assert min_score_from_env() is None
    delivered, residual = apply_score_gate(RANKED, 3)
    assert delivered == RANKED[:3]
    assert residual == RANKED[3:]


def test_valid_threshold_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MIN_SCORE_ENV, " 0.35 ")
    assert min_score_from_env() == pytest.approx(0.35)


# --- drop -----------------------------------------------------------------


@pytest.mark.parametrize("form", [None, "drop", "DROP", "bogus"])
def test_drop_fills_from_below_and_residual_keeps_the_rest(
    monkeypatch: pytest.MonkeyPatch, form: str | None
) -> None:
    monkeypatch.setenv(MIN_SCORE_ENV, "0.3")
    if form is not None:
        monkeypatch.setenv(GATE_FORM_ENV, form)
    delivered, residual = apply_score_gate(RANKED, 3)
    assert ids(delivered) == ["a", "b", "d"]
    assert ids(residual) == ["c", "e", "f"]
    assert all(result.withheld is None for result in delivered)
    # Nothing ranked is lost: delivered + residual is a partition of ranked.
    assert sorted(ids(delivered) + ids(residual)) == sorted(ids(RANKED))


def test_drop_shortens_the_answer_when_too_few_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MIN_SCORE_ENV, "0.3")
    delivered, residual = apply_score_gate(RANKED, 10)
    assert ids(delivered) == ["a", "b", "d"]
    assert ids(residual) == ["c", "e", "f"]


def test_rank_one_always_delivered_in_full(monkeypatch: pytest.MonkeyPatch) -> None:
    weak = [make_result("w1", 0.05), make_result("w2", 0.9), make_result("w3", 0.01)]
    monkeypatch.setenv(MIN_SCORE_ENV, "0.99")
    for form in ("drop", "stub"):
        monkeypatch.setenv(GATE_FORM_ENV, form)
        delivered, _ = apply_score_gate(weak, 3)
        assert delivered[0] is weak[0]
        assert delivered[0].withheld is None


def test_empty_ranked_and_zero_max(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MIN_SCORE_ENV, "0.3")
    assert apply_score_gate([], 5) == ([], [])
    assert apply_score_gate(RANKED, 0) == ([], RANKED)


# --- stub -----------------------------------------------------------------


def test_stub_marks_failures_inside_the_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MIN_SCORE_ENV, "0.3")
    monkeypatch.setenv(GATE_FORM_ENV, "stub")
    delivered, residual = apply_score_gate(RANKED, 4)
    assert ids(delivered) == ["a", "b", "c", "d"]
    assert [result.withheld for result in delivered] == [
        None,
        None,
        WITHHELD_BELOW_THRESHOLD,
        None,
    ]
    assert ids(residual) == ["e", "f"]


def test_stub_shapes_as_content_less_stub_with_content_ref() -> None:
    long_text = "Detailed body sentence. " * 80
    results = [
        make_result("a", 1.0, node=make_node("a", long_text)),
        make_result("c", 0.2, node=make_node("c", long_text + " extra"), withheld="below_threshold"),
        make_result("d", 0.8, node=make_node("d", "short full content")),
    ]
    shaped = shape_recall_results(
        results,
        already_delivered_ids=set(),
        snippet_max_chars=1200,
        context_value_max_chars=160,
        session_dedup=False,
    )
    assert [entry["delivery"] for entry in shaped] == [
        "full",
        DELIVERY_BELOW_THRESHOLD,
        "full",  # the stub took no ladder slot: d is bearer #2, ladder 1000
    ]
    stub = shaped[1]
    assert stub["node"]["content"] == ""
    assert stub["content_ref"]["node_id"] == "c"
    assert stub["content_ref"]["full_content_chars"] == len(long_text + " extra")
    # Every score field survives, as on any other delivery class.
    for key in ("score", "bm25_score", "vector_score", "graph_score", "trigger_score"):
        assert key in stub


def test_withheld_result_is_not_a_twin_bearer() -> None:
    text = "identical text of two nodes"
    results = [
        make_result("top", 1.0),
        make_result("held", 0.1, node=make_node("held", text), withheld="below_threshold"),
        make_result("twin", 0.9, node=make_node("twin", text)),
    ]
    shaped = shape_recall_results(
        results,
        already_delivered_ids=set(),
        snippet_max_chars=1200,
        context_value_max_chars=160,
        session_dedup=False,
        duplicate_of={"twin": "held"},
    )
    assert [entry["delivery"] for entry in shaped] == ["full", DELIVERY_BELOW_THRESHOLD, "full"]
    assert shaped[2]["node"]["content"] == text


def test_shaping_without_withheld_is_unchanged() -> None:
    results = [make_result("a", 1.0), make_result("b", 0.5)]
    shaped = shape_recall_results(
        results,
        already_delivered_ids=set(),
        snippet_max_chars=1200,
        context_value_max_chars=160,
        session_dedup=False,
    )
    assert [entry["delivery"] for entry in shaped] == ["full", "full"]


# --- the gate score -------------------------------------------------------


def test_gate_score_uses_reference_weights_and_multipliers() -> None:
    base = make_result("x", 0.5, vector_score=0.5)
    expected = REFERENCE_WEIGHTS.bm25 * 0.5 + REFERENCE_WEIGHTS.vector * 0.5
    assert gate_score(base) == pytest.approx(expected)
    # Per-node demotion multiplies.
    assert gate_score(base, {"x": 0.5}) == pytest.approx(expected * 0.5)
    assert gate_score(base, {"other": 0.1}) == pytest.approx(expected)
    # Superseded penalty (0.2x) from the node feedback multiplier.
    assert gate_score(RecallResult(**{**_fields(base), "superseded": True})) == pytest.approx(
        expected * 0.2
    )
    # Node feedback multiplier: a useful node scores higher.
    useful = make_result("y", 0.5, vector_score=0.5, node=make_node("y", usefulness_score=0.8))
    assert gate_score(useful) > gate_score(base)
    # Causal boost applies only with graph evidence.
    graphed = make_result("g", 0.5, graph_score=0.5)
    assert gate_score(graphed, causal_mode=True) == pytest.approx(
        gate_score(graphed, causal_mode=False) * 1.5 * _causal_floor_ratio(graphed)
    )


def _fields(result: RecallResult) -> dict[str, Any]:
    return {name: getattr(result, name) for name in RecallResult.__slots__}


def _causal_floor_ratio(result: RecallResult) -> float:
    # Causal mode raises the graph-weight floor 0.25 -> 0.75; reproduce the
    # re-blend so the test isolates the 1.5x boost.
    def blend(floor: float) -> float:
        bm25, vector, graph = REFERENCE_WEIGHTS.bm25, REFERENCE_WEIGHTS.vector, REFERENCE_WEIGHTS.graph
        if graph < floor:
            deficit = floor - graph
            remaining = bm25 + vector
            bm25 -= deficit * bm25 / remaining
            vector -= deficit * vector / remaining
            graph = floor
        return bm25 * result.bm25_score + vector * result.vector_score + graph * result.graph_score

    return blend(0.75) / blend(0.25)


def test_gate_score_does_not_read_the_store_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MIN_SCORE_ENV, "0.2")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _seed_corpus(store)
        service = MemoryRecallService(store)

        def run() -> tuple[set[str], dict[str, float]]:
            delivered = service.memory_recall(
                "database migration rollback failure",
                scope=SCOPE,
                max_results=50,
                log_access=False,
                log_event=False,
            )
            everything = delivered + service.last_residual
            return set(ids(delivered)), {r.node.id: gate_score(r) for r in everything}

        store.set_retrieval_weights(SCOPE, bm25=0.9, vector=0.1, graph=0.0)
        passed_a, scores_a = run()
        store.set_retrieval_weights(SCOPE, bm25=0.05, vector=0.9, graph=0.05)
        passed_b, scores_b = run()

    assert scores_a == pytest.approx(scores_b)
    # Rank 1 may differ between the two weightings (it always ships); every
    # other slot is decided by the weight-free gate score alone.
    assert passed_a - _rank_one_only(passed_a, scores_a) == passed_b - _rank_one_only(passed_b, scores_b)
    assert any(score < 0.2 for score in scores_a.values())
    assert any(score >= 0.2 for score in scores_a.values())


def _rank_one_only(passed: set[str], scores: dict[str, float]) -> set[str]:
    return {node_id for node_id in passed if scores[node_id] < 0.2}


def _seed_corpus(store: MemoryStore) -> None:
    store.append_trace(
        "Database migration rollback after deploy failure: restore snapshot first",
        {"scope": SCOPE, "agent": "a"},
    )
    store.append_trace("Migration rollback checklist for the database", {"scope": SCOPE, "agent": "a"})
    store.append_trace("Deploy failure caused by a missing env var", {"scope": SCOPE, "agent": "b"})
    store.append_trace("Database vacuum schedule and disk space", {"scope": SCOPE, "agent": "b"})
    for index in range(30):
        store.append_trace(f"unrelated note {index} about lunch menus", {"scope": SCOPE, "agent": "c"})


# --- trigger-found schemas ------------------------------------------------


def test_trigger_found_schema_gated_on_its_own_scale() -> None:
    schema = make_node("s", level="schema")
    hit = make_result("s", 0.0, node=schema, trigger_score=0.975, score=0.975 * 1.8, methods=("trigger",))
    # Weak on the channel scale, clean on the trigger scale: passes.
    assert gate_score(hit) == 0.0
    assert trigger_gate_score(hit) == pytest.approx(0.975 / 0.95)
    assert passes_gate(hit, 0.4)
    # A demotion that halves it takes it out on both scales.
    assert not passes_gate(hit, 0.4, {"s": 0.4})
    assert trigger_gate_score(hit, {"s": 0.4}) < SCHEMA_TRIGGER_GATE_MIN
    # A trace with a trigger_score (not a schema) is not on the trigger scale.
    trace = make_result("t", 0.0, trigger_score=0.975)
    assert trigger_gate_score(trace) is None
    assert not passes_gate(trace, 0.4)


def test_trigger_found_schema_passes_through_the_live_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(MIN_SCORE_ENV, "0.9")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _seed_corpus(store)
        schema = store.create_node(
            level="schema",
            content="Procedure: quarterly key rotation for vault tokens",
            context={"scope": SCOPE, "trigger": "rotate vault keys"},
        )
        service = MemoryRecallService(store)
        delivered = service.memory_recall(
            "database migration rollback: rotate vault keys too",
            scope=SCOPE,
            max_results=10,
            log_access=False,
            log_event=False,
        )
        by_id = {r.node.id: r for r in delivered + service.last_residual}
    assert by_id[schema.id].trigger_score > 0.0
    assert gate_score(by_id[schema.id]) < 0.9
    assert schema.id in ids(delivered)


# --- off is the identity on the live path ---------------------------------


def _recall_snapshot(store: MemoryStore) -> tuple[list[dict[str, Any]], list[str]]:
    service = MemoryRecallService(store)
    delivered = service.memory_recall(
        "database migration rollback failure",
        scope=SCOPE,
        max_results=3,
        log_access=False,
        log_event=False,
    )
    return [r.to_dict() for r in delivered], ids(service.last_residual)


def test_off_is_identical_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _seed_corpus(store)
        _recall_snapshot(store)  # first recall backfills embeddings
        baseline = _recall_snapshot(store)
        for raw in ("", "0", "junk"):
            monkeypatch.setenv(MIN_SCORE_ENV, raw)
            monkeypatch.setenv(GATE_FORM_ENV, "stub")
            assert _recall_snapshot(store) == baseline
        assert len(baseline[0]) == 3
        assert all("withheld" not in entry for entry in baseline[0])


def test_live_path_drop_and_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        _seed_corpus(store)
        _recall_snapshot(store)  # first recall backfills embeddings
        off_delivered, off_residual = _recall_snapshot(store)
        monkeypatch.setenv(MIN_SCORE_ENV, "5")  # nothing passes
        drop_delivered, drop_residual = _recall_snapshot(store)
        monkeypatch.setenv(GATE_FORM_ENV, "stub")
        stub_delivered, stub_residual = _recall_snapshot(store)

    off_ids = [entry["node"]["id"] for entry in off_delivered]
    assert [entry["node"]["id"] for entry in drop_delivered] == off_ids[:1]
    assert drop_residual == off_ids[1:] + off_residual
    assert [entry["node"]["id"] for entry in stub_delivered] == off_ids
    assert [entry.get("withheld") for entry in stub_delivered] == [None] + [
        WITHHELD_BELOW_THRESHOLD
    ] * (len(off_ids) - 1)
    assert stub_residual == off_residual
