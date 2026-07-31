"""Feedback updates for retrieval ranking and learned weights."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from types import SimpleNamespace
from typing import Any

from living_memory.models import Node, RecallEvent, RetrievalWeights
from living_memory.storage import MemoryStore


_ADAPTIVE_LR_LADDER: tuple[tuple[int, float], ...] = (
    (10, 0.20),
    (50, 0.10),
    (500, 0.05),
)
_ADAPTIVE_LR_FLOOR = 0.02
_DEFAULT_PENDING_RECALL_LIMIT = 5

# Ranking feedback multipliers. Confidence is neutral at the storage-default
# base (0.5): a fresh, unvalidated node must rank on retrieval relevance
# alone rather than start with an implicit penalty, while confidence above or
# below the base still moves the boost monotonically around 1.0.
_BASE_CONFIDENCE = 0.5
_CONFIDENCE_SLOPE = 0.65
# Upper bound for the compounded confidence x usefulness x access multiplier.
# Uncapped, entrenched nodes (up to 1.325 x 1.55 x 1.25 ~= 2.57) bury fresh
# exact matches: under learned vector-dominant weights (bm25 0.10 / vector
# 0.85) a just-written trace with an exact lexical hit scores ~0.48 while a
# weak 0.38 vector match scores ~0.32 — any cap above ~1.45 lets the
# entrenched node win, so 1.4 keeps fresh writes recallable with margin while
# still rewarding validated, useful, frequently accessed knowledge.
FEEDBACK_MULTIPLIER_CAP = 1.4

# Diminishing returns for positive usefulness reinforcement.
#
# The cap above bounds how hard feedback can boost a score, but not how fast
# a node accumulates the usefulness feeding that boost. Every delivered
# recall result is implicitly reinforced by the next compatible remember
# (apply_pending_recall_feedback), so delivery itself breeds rank advantage:
# boosted nodes get delivered more, deliveries reinforce them further, and a
# handful of saturated nodes end up collecting most deliveries (measured
# 2026-07-31: ~76% of deliveries went to a few usefulness=1.0 nodes). To
# damp that loop at its source, a positive usefulness increment is scaled by
#
#     max(HEADROOM_FLOOR, 1 - usefulness) / (1 + ACCESS_DAMPING * ln(1 + accesses))
#
# so a node near saturation, or one that has already been delivered many
# times, needs disproportionately more fresh evidence per unit of further
# boost than an unproven node (an entrenched node at usefulness 1.0 with
# ~600 accesses gains ~30x slower than a fresh one). The headroom factor is
# floored, never zeroed, so usefulness 1.0 stays exactly reachable via the
# clamp; a node corrected into negative usefulness recovers with full
# headroom (factor 1.0), though the access divisor still applies — a
# heavily-delivered node re-earns its boost slowly no matter where it
# starts. Explicit negative feedback is exempt: corrections always apply at
# full strength. The cap's semantics are unchanged — it still bounds the
# compounded boost at 1.4 protecting fresh exact-match writes; these knobs
# only slow how fast entrenchment approaches that ceiling. Neutral values
# (floor 1.0, damping 0.0) reproduce the legacy flat 0.1 * signal increment
# exactly; tests/test_feedback_delivery_concentration.py runs a closed-loop
# simulation under both settings and pins the concentration reduction.
USEFULNESS_GAIN_HEADROOM_FLOOR = 0.25
USEFULNESS_GAIN_ACCESS_DAMPING = 1.0


def _retrieval_tuning_policy() -> str:
    return os.environ.get("LM_RETRIEVAL_TUNING_POLICY", "fixed").strip().lower()


def _adaptive_learning_rate(trace_count: int) -> float:
    """Effective learning rate for a scope of the given size.

    Young scopes converge fast on a few feedback signals; mature scopes
    move slowly to avoid oscillation around a settled policy.
    """

    for ceiling, rate in _ADAPTIVE_LR_LADDER:
        if trace_count < ceiling:
            return rate
    return _ADAPTIVE_LR_FLOOR


def _maybe_retune_learning_rate(store: MemoryStore, scope: str) -> None:
    if _retrieval_tuning_policy() != "adaptive":
        return
    row = store.connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM nodes
        WHERE level = 'trace' AND scope = ? AND decayed = 0
        """,
        (scope,),
    ).fetchone()
    target_lr = _adaptive_learning_rate(int(row["count"]) if row else 0)
    current = store.get_retrieval_weights(scope)
    if abs(current.learning_rate - target_lr) < 1e-6:
        return
    store.set_retrieval_weights(
        scope,
        bm25=current.bm25,
        vector=current.vector,
        graph=current.graph,
        learning_rate=target_lr,
    )


@dataclass(frozen=True, slots=True)
class FeedbackUpdate:
    """Result of applying feedback to a recall result."""

    node: Node | None
    weights: RetrievalWeights


@dataclass(frozen=True, slots=True)
class ImplicitRecallFeedback:
    """Result of connecting a new ingest trace to prior recall events."""

    trace: Node
    events: list[RecallEvent]
    linked_node_ids: list[str]
    feedback_applied: bool


class FeedbackService:
    """Apply explicit or implicit feedback to nodes and retrieval weights."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def apply(
        self,
        result: Any,
        *,
        useful: bool = True,
        signal: float = 1.0,
        scope: str | None = None,
    ) -> FeedbackUpdate:
        return apply_retrieval_feedback(
            self.store,
            result,
            useful=useful,
            signal=signal,
            scope=scope,
        )


def apply_retrieval_feedback(
    store: MemoryStore,
    result: Any,
    *,
    useful: bool = True,
    signal: float = 1.0,
    scope: str | None = None,
) -> FeedbackUpdate:
    """Update node usefulness and nudge weights by each method's evidence share."""

    magnitude = max(0.0, min(abs(float(signal)), 5.0))
    signed_signal = magnitude if useful else -magnitude

    node = _result_node(result, store)
    if node is not None:
        increment = 0.1 * signed_signal
        if signed_signal > 0.0:
            increment *= _positive_reinforcement_gain(node)
        updated_usefulness = max(-1.0, min(1.0, node.usefulness_score + increment))
        node = store.update_node(node.id, stats={"usefulness_score": updated_usefulness})

    target_scope = scope or (node.scope if node is not None else "global")
    _maybe_retune_learning_rate(store, target_scope)
    method_signals = _method_signals(result, signed_signal)
    weights = store.update_retrieval_weights(
        target_scope,
        bm25_signal=method_signals["bm25"],
        vector_signal=method_signals["vector"],
        graph_signal=method_signals["graph"],
    )
    return FeedbackUpdate(node=node, weights=weights)


def apply_pending_recall_feedback(
    store: MemoryStore,
    trace: Node,
    *,
    limit: int = _DEFAULT_PENDING_RECALL_LIMIT,
    reinforce_results: bool = True,
) -> ImplicitRecallFeedback:
    """Attach recent recall provenance to a new trace and optionally reinforce hits."""

    events = store.pending_recall_events(
        scope=trace.scope,
        context=trace.context,
        content=trace.content,
        limit=limit,
    )
    if not events:
        return ImplicitRecallFeedback(
            trace=trace,
            events=[],
            linked_node_ids=[],
            feedback_applied=False,
        )

    provenance = dict(trace.provenance)
    prior_recalls = list(provenance.get("prior_recalls", []))
    recalled_nodes = list(provenance.get("recalled_nodes", []))
    source_traces = list(trace.source_traces)
    linked_node_ids: list[str] = []
    feedback_applied = False

    for event in events:
        event_node_ids: list[str] = []
        for rank, result in enumerate(event.results):
            node_id = str(result.get("node_id") or "")
            if not node_id or node_id == trace.id:
                continue

            node = store.get_node(node_id)
            if node is None:
                continue

            event_node_ids.append(node_id)
            if node_id not in recalled_nodes:
                recalled_nodes.append(node_id)
            if node.level == "trace" and node_id not in source_traces:
                source_traces.append(node_id)

            weight = _implicit_connection_weight(rank)
            store.create_connection(
                trace.id,
                node_id,
                "related",
                weight=weight,
                metadata={
                    "basis": "implicit_recall_feedback",
                    "recall_event_id": event.id,
                    "recall_query": event.query,
                    "rank": rank + 1,
                },
            )
            if node_id not in linked_node_ids:
                linked_node_ids.append(node_id)

            if reinforce_results:
                synthetic_result = SimpleNamespace(
                    node_id=node_id,
                    bm25_score=float(result.get("bm25_score", 0.0) or 0.0),
                    vector_score=float(result.get("vector_score", 0.0) or 0.0),
                    graph_score=float(result.get("graph_score", 0.0) or 0.0),
                )
                signal = max(0.2, 1.0 / (rank + 1))
                apply_retrieval_feedback(
                    store,
                    synthetic_result,
                    useful=True,
                    signal=signal,
                    scope=event.scope,
                )
                feedback_applied = True

        prior_recalls.append(
            {
                "id": event.id,
                "query": event.query,
                "scope": event.scope,
                "timestamp": event.created_at,
                "result_ids": event_node_ids,
            }
        )
        store.mark_recall_event_feedback(event.id, trace.id)

    provenance["prior_recalls"] = prior_recalls
    provenance["recalled_nodes"] = recalled_nodes
    provenance["source_traces"] = source_traces
    updated = store.update_node(trace.id, provenance=provenance)
    return ImplicitRecallFeedback(
        trace=updated,
        events=events,
        linked_node_ids=linked_node_ids,
        feedback_applied=feedback_applied,
    )


def feedback_weighted_score(
    node: Node,
    base_score: float,
    *,
    superseded: bool = False,
    superseding: bool = False,
) -> float:
    """Apply confidence, usefulness, access, and correction feedback to a score."""

    confidence = max(0.0, min(node.confidence, 1.0))
    confidence_boost = 1.0 + _CONFIDENCE_SLOPE * (confidence - _BASE_CONFIDENCE)
    usefulness = max(-1.0, min(node.usefulness_score, 1.0))
    usefulness_boost = 1.0 + (0.55 * usefulness if usefulness >= 0.0 else 0.45 * usefulness)
    access_boost = 1.0 + min(0.25, 0.04 * _log1p(node.access_count))
    # Cap covers the compounded positive feedback only; correction signals
    # (superseding boost, superseded penalty) stay deliberate and uncapped.
    feedback_multiplier = min(
        confidence_boost * usefulness_boost * access_boost,
        FEEDBACK_MULTIPLIER_CAP,
    )
    correction_boost = 1.2 if superseding else 1.0
    superseded_penalty = 0.2 if superseded else 1.0
    return base_score * feedback_multiplier * correction_boost * superseded_penalty


def _positive_reinforcement_gain(node: Node) -> float:
    """Diminishing-returns factor for a positive usefulness increment.

    See the rationale at USEFULNESS_GAIN_HEADROOM_FLOOR: headroom shrinks
    gains as usefulness approaches saturation (floored so the 1.0 ceiling
    stays reachable; negative usefulness gets full headroom), and log-access
    crowding makes frequently delivered nodes need more evidence per unit of
    boost regardless of where their usefulness sits. Knobs are read at call
    time so the legacy rule is recoverable by setting them to their neutral
    values (floor 1.0, damping 0.0).
    """

    headroom = 1.0 - min(max(node.usefulness_score, 0.0), 1.0)
    saturation = max(USEFULNESS_GAIN_HEADROOM_FLOOR, headroom)
    crowding = 1.0 + USEFULNESS_GAIN_ACCESS_DAMPING * _log1p(node.access_count)
    return saturation / crowding


def _method_signals(result: Any, signed_signal: float) -> dict[str, float]:
    """Per-method credit proportional to each method's raw score share.

    Positive feedback rewards every method by its share of the recorded raw
    evidence for the reinforced result; negative feedback assigns blame by
    the same shares. Shares come from raw scores, not weight-multiplied
    contributions, so the update is independent of the current weights (a
    weight-share rule would compound its own bias). With no positive raw
    evidence there is nothing to attribute: every signal is zero and the
    weights stay put. The replay harness's ``proportional`` credit rule
    delegates here — this function is the single source of truth.
    """

    components = {
        "bm25": max(0.0, float(getattr(result, "bm25_score", 0.0) or 0.0)),
        "vector": max(0.0, float(getattr(result, "vector_score", 0.0) or 0.0)),
        "graph": max(0.0, float(getattr(result, "graph_score", 0.0) or 0.0)),
    }
    total = sum(components.values())
    if total <= 0.0:
        return {method: 0.0 for method in components}
    return {method: signed_signal * value / total for method, value in components.items()}


def _result_node(result: Any, store: MemoryStore) -> Node | None:
    node = getattr(result, "node", None)
    if isinstance(node, Node):
        return store.get_node(node.id) or node
    node_id = getattr(result, "node_id", None) or getattr(result, "id", None)
    if node_id:
        return store.get_node(str(node_id))
    return None


def _log1p(value: int) -> float:
    return math.log1p(max(0, value))


def _implicit_connection_weight(rank: int) -> float:
    return max(0.2, 0.75 / (rank + 1))
