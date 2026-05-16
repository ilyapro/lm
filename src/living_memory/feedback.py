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
    """Update node usefulness and nudge weights toward the winning method."""

    magnitude = max(0.0, min(abs(float(signal)), 5.0))
    signed_signal = magnitude if useful else -magnitude

    node = _result_node(result, store)
    if node is not None:
        updated_usefulness = max(-1.0, min(1.0, node.usefulness_score + 0.1 * signed_signal))
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
    limit: int = 1,
    reinforce_results: bool = True,
) -> ImplicitRecallFeedback:
    """Attach recent recall provenance to a new trace and optionally reinforce hits."""

    events = store.pending_recall_events(
        scope=trace.scope,
        context=trace.context,
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

    confidence_boost = 0.45 + 0.65 * max(0.0, min(node.confidence, 1.0))
    usefulness = max(-1.0, min(node.usefulness_score, 1.0))
    usefulness_boost = 1.0 + (0.55 * usefulness if usefulness >= 0.0 else 0.45 * usefulness)
    access_boost = 1.0 + min(0.25, 0.04 * _log1p(node.access_count))
    correction_boost = 1.2 if superseding else 1.0
    superseded_penalty = 0.2 if superseded else 1.0
    return base_score * confidence_boost * usefulness_boost * access_boost * correction_boost * superseded_penalty


def _method_signals(result: Any, signed_signal: float) -> dict[str, float]:
    components = {
        "bm25": max(0.0, float(getattr(result, "bm25_score", 0.0) or 0.0)),
        "vector": max(0.0, float(getattr(result, "vector_score", 0.0) or 0.0)),
        "graph": max(0.0, float(getattr(result, "graph_score", 0.0) or 0.0)),
    }
    dominant = max(components, key=components.get)
    if components[dominant] <= 0.0:
        dominant = "bm25"

    if signed_signal >= 0:
        signals = {method: -0.25 * signed_signal for method in components}
        signals[dominant] = signed_signal
    else:
        positive = abs(signed_signal)
        signals = {method: 0.15 * positive for method in components}
        signals[dominant] = -positive
    return signals


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
