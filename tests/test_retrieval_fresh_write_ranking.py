"""Fresh-write ranking regression: a just-written trace must reach default top-10.

Reproduces the 2026-06-10 ``fix-recall-fresh-write-index`` escalation: indexing
was healthy (the fresh trace was the top FTS5 hit and vector-matched after the
lazy backfill), yet a probe trace with a unique marker word came back at rank 22.
The pre-fix ``feedback_weighted_score`` gave a fresh node (confidence 0.5,
usefulness 0, access_count 0) an active x0.775 penalty while entrenched
competitors compounded confidence x usefulness x access boosts up to ~x1.5, so
under learned vector-dominant weights (bm25=0.10, vector=0.85) an exact
lexical match lost to weak vector matches of boosted neighbours.

Guards the two fixes:
  (a) the confidence boost is neutral (x1.0) at the storage-default base
      confidence 0.5 — fresh, unvalidated nodes rank on relevance alone;
  (d) the compounded confidence x usefulness x access multiplier is capped at
      FEEDBACK_MULTIPLIER_CAP = 1.4 so entrenched nodes cannot bury a fresh
      exact match.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity
from living_memory.feedback import feedback_weighted_score
from living_memory.models import Node
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore


class FakeMCP:
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


def _node(**overrides: Any) -> Node:
    defaults: dict[str, Any] = {
        "id": "node-under-test",
        "level": "trace",
        "content": "content",
    }
    defaults.update(overrides)
    return Node(**defaults)


def test_fresh_node_feedback_multiplier_is_neutral() -> None:
    # A just-written trace: confidence 0.5 (storage default), usefulness 0,
    # access_count 0. Fix (a): the multiplier must be exactly 1.0 — the
    # pre-fix formula 0.45 + 0.65 * 0.5 = 0.775 actively penalized it.
    assert feedback_weighted_score(_node(), 1.0) == pytest.approx(1.0)


def test_confidence_boost_stays_monotonic_around_base() -> None:
    below = feedback_weighted_score(_node(confidence=0.2), 1.0)
    base = feedback_weighted_score(_node(confidence=0.5), 1.0)
    above = feedback_weighted_score(_node(confidence=0.9), 1.0)
    assert below < base < above
    assert below < 1.0 < above


def test_compounded_feedback_multiplier_is_capped() -> None:
    # Uncapped this would be 1.325 (confidence 1.0) x 1.55 (usefulness 1.0)
    # x 1.25 (saturated access) ~= 2.57. Fix (d) caps the compound at
    # FEEDBACK_MULTIPLIER_CAP (literal here so the test pins the calibrated
    # value instead of echoing the constant under test).
    entrenched = _node(confidence=1.0, usefulness_score=1.0, access_count=10_000)
    assert feedback_weighted_score(entrenched, 1.0) == pytest.approx(1.4)


def test_correction_signals_stay_outside_the_cap() -> None:
    # The cap covers confidence x usefulness x access only: the superseding
    # correction boost still multiplies on top, and the superseded penalty
    # still floors the score down.
    entrenched = _node(confidence=1.0, usefulness_score=1.0, access_count=10_000)
    superseding = feedback_weighted_score(entrenched, 1.0, superseding=True)
    assert superseding == pytest.approx(1.4 * 1.2)
    assert feedback_weighted_score(_node(), 1.0, superseded=True) == pytest.approx(0.2)


def test_negative_usefulness_penalty_passes_through_uncapped() -> None:
    # Only the upper bound is capped; explicit negative feedback must keep
    # its full effect.
    downvoted = _node(usefulness_score=-1.0)
    assert feedback_weighted_score(downvoted, 1.0) == pytest.approx(0.55)


# --- End-to-end regression: write -> ranked recall against boosted rivals ---

SCOPE = "project:freshwrite"
UNIQUE_MARKER = "zq7xfreshprobe"
QUERY = f"deployment pipeline recall tuning probe {UNIQUE_MARKER}"
# Tuned for a moderate vector match against QUERY under the hash backend
# (~0.34, mirroring the live incident's 0.449) — below STRONG_VECTOR_MATCH so
# the fresh trace stays on the blended bm25+vector scoring path.
FRESH_CONTENT = (
    f"Fresh probe {UNIQUE_MARKER} checking recall of writes recorded in the current session"
)
# Learned vector-dominant weights observed in the escalation snapshot. The
# competitor contents below must share no FTS term with QUERY so their bm25
# stays 0 and their relevance is purely the crafted vector match.
WEIGHTS = {"bm25": 0.10, "vector": 0.85, "graph": 0.05}
COMPETITOR_COUNT = 23
COMPETITOR_USEFULNESS = 1.0  # usefulness boost saturates at x1.55
COMPETITOR_ACCESS_COUNT = 600  # 0.04 * log1p(600) > 0.25 -> access boost caps at x1.25
# Pre-fix multipliers, kept as literals to document the counterfactual the
# scenario must invert: fresh 0.45 + 0.65 * 0.5, competitors x1.55 and x1.25
# usefulness/access boosts on top of the same 0.775 confidence factor.
OLD_FRESH_MULTIPLIER = 0.775
OLD_COMPETITOR_MULTIPLIER = 0.775 * 1.55 * 1.25
# Post-fix multipliers: neutral confidence (x1.0) for the fresh node, the
# compounded competitor boost capped at FEEDBACK_MULTIPLIER_CAP.
NEW_FRESH_MULTIPLIER = 1.0
NEW_COMPETITOR_MULTIPLIER = 1.4


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


def _competitor_embedding(query_embedding: list[float], target_cosine: float) -> list[float]:
    """Unit vector whose cosine against the query embedding is exactly ``target_cosine``."""

    unit_query = _unit(list(query_embedding))
    pivot = min(range(len(unit_query)), key=lambda index: abs(unit_query[index]))
    ortho = [-unit_query[pivot] * value for value in unit_query]
    ortho[pivot] += 1.0
    ortho = _unit(ortho)
    residual = math.sqrt(1.0 - target_cosine * target_cosine)
    return [
        target_cosine * query_value + residual * ortho_value
        for query_value, ortho_value in zip(unit_query, ortho, strict=True)
    ]


def test_fresh_write_reaches_default_top10_against_boosted_competitors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Deterministic embeddings; fixed tuning/consolidation policies so the
    # seeded scenario is not mutated by adaptive hooks.
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.delenv("LM_RETRIEVAL_TUNING_POLICY", raising=False)
    monkeypatch.delenv("LM_AUTO_CONSOLIDATE_POLICY", raising=False)

    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store: MemoryStore = mcp.memory_store
    store.set_retrieval_weights(SCOPE, **WEIGHTS)

    embedder = LocalEmbeddingModel(backend="hash")
    query_embedding = embedder.embed(QUERY)
    fresh_vector_score = cosine_similarity(query_embedding, embedder.embed(FRESH_CONTENT))
    # Incident profile: exact lexical hit plus a moderate vector match that
    # stays below the STRONG_VECTOR_MATCH floor (0.65).
    assert 0.08 < fresh_vector_score < 0.65
    # The fresh trace is the only FTS match for QUERY (unique marker), so its
    # rank-normalized bm25 score is 1.0.
    fresh_base = WEIGHTS["bm25"] * 1.0 + WEIGHTS["vector"] * fresh_vector_score

    # Mid-window weak vector match: strong enough that pre-fix boosts push
    # every competitor above the fresh trace, weak enough that the capped
    # neutral multipliers invert the ordering.
    competitor_cosine = 0.72 * fresh_base
    assert 0.08 < competitor_cosine < 0.65
    competitor_base = WEIGHTS["vector"] * competitor_cosine
    # Counterfactual this regression guards (fails the suite if content or
    # embedder drift moves the scenario out of the discriminating window):
    # pre-fix scoring buried the fresh trace below the boosted crowd...
    assert competitor_base * OLD_COMPETITOR_MULTIPLIER > fresh_base * OLD_FRESH_MULTIPLIER
    # ...post-fix scoring must rank the fresh trace above the same crowd.
    assert competitor_base * NEW_COMPETITOR_MULTIPLIER < fresh_base * NEW_FRESH_MULTIPLIER

    rival_embedding = _competitor_embedding(query_embedding, competitor_cosine)
    for index in range(COMPETITOR_COUNT):
        rival = store.append_trace(
            f"Established veteran runbook entry {index} covering legacy subsystem upkeep checklists",
            {"scope": SCOPE, "agent": "veteran-agent"},
            feedback={
                "usefulness_score": COMPETITOR_USEFULNESS,
                "access_count": COMPETITOR_ACCESS_COUNT,
            },
        )
        store.update_node(rival.id, embedding=rival_embedding)

    remembered = mcp.tools["memory_remember"](
        FRESH_CONTENT,
        {"scope": SCOPE, "agent": "fresh-agent", "task": "fresh-write-ranking"},
    )
    fresh_id = remembered["node"]["id"]

    # Ranked recall over a 10-result page (the tool default shrank to 5;
    # the ranking scenario needs the original page size), same process.
    recalled = mcp.tools["memory_recall"](QUERY, scope=SCOPE, max_results=10)

    assert recalled["count"] == 10, "boosted competitors must fill the result page"
    ranked_ids = [result["node"]["id"] for result in recalled["results"]]
    assert fresh_id in ranked_ids, (
        f"fresh trace {fresh_id} missing from default top-10: {ranked_ids}"
    )

    fresh_result = next(
        result for result in recalled["results"] if result["node"]["id"] == fresh_id
    )
    # Scenario integrity: the unique marker made the fresh trace the sole (and
    # therefore top) lexical hit, with the expected moderate vector match.
    assert fresh_result["bm25_score"] == pytest.approx(1.0)
    assert fresh_result["vector_score"] == pytest.approx(fresh_vector_score, abs=1e-5)
    for result in recalled["results"]:
        if result["node"]["id"] == fresh_id:
            continue
        assert result["bm25_score"] == 0.0
        assert result["vector_score"] == pytest.approx(competitor_cosine, abs=1e-5)
        assert result["node"]["stats"]["usefulness_score"] == COMPETITOR_USEFULNESS
        assert result["node"]["stats"]["access_count"] >= COMPETITOR_ACCESS_COUNT
