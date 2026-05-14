"""Learning and maintenance loop for Living Memory."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import log1p
from typing import Any, Iterable, Mapping
import re

from living_memory.decay import DecayResult, apply_decay, memory_forget
from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity
from living_memory.feedback import ImplicitRecallFeedback, apply_pending_recall_feedback
from living_memory.models import Connection, Node
from living_memory.storage import MemoryStore
from living_memory.temporal import detect_temporal_hint

DEFAULT_MIN_CLUSTER_SIZE = 100
DEFAULT_RECENT_LIMIT = 10_000
JACCARD_SIMILARITY_THRESHOLD = 0.58
EMBEDDING_SIMILARITY_THRESHOLD = 0.65

_STOP_WORDS = {
    "a",
    "about",
    "after",
    "again",
    "and",
    "are",
    "but",
    "because",
    "before",
    "being",
    "between",
    "could",
    "during",
    "from",
    "have",
    "into",
    "only",
    "should",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "trace",
    "under",
    "when",
    "where",
    "which",
    "with",
    "would",
    "а",
    "без",
    "бы",
    "в",
    "во",
    "для",
    "до",
    "его",
    "ее",
    "если",
    "же",
    "за",
    "и",
    "из",
    "или",
    "им",
    "их",
    "к",
    "как",
    "ко",
    "ли",
    "на",
    "не",
    "но",
    "о",
    "об",
    "от",
    "по",
    "под",
    "при",
    "с",
    "со",
    "так",
    "то",
    "у",
    "уже",
    "что",
    "это",
    "этот",
    "эти",
    "эту",
}


@dataclass(slots=True)
class ConsolidationResult:
    """Summary of one consolidation pass."""

    concepts_created: list[Node] = field(default_factory=list)
    concepts_updated: list[Node] = field(default_factory=list)
    decayed: list[Node] = field(default_factory=list)
    clusters_considered: int = 0
    traces_considered: int = 0

    @property
    def concepts(self) -> list[Node]:
        return [*self.concepts_created, *self.concepts_updated]


@dataclass(slots=True)
class TeachResult:
    """Result of teaching a correction to the memory."""

    corrective_trace: Node
    supersedes: Connection
    original: Node
    implicit_feedback: ImplicitRecallFeedback | None = None


@dataclass(slots=True)
class _TraceCluster:
    scope: str
    traces: list[Node]
    token_counts: Counter[str]
    embedding_sum: list[float] | None = None
    embedding_count: int = 0
    strategy: str = "token-jaccard"

    @property
    def representative_tokens(self) -> set[str]:
        if not self.traces:
            return set()
        threshold = max(1, len(self.traces) // 2)
        common = {token for token, count in self.token_counts.items() if count >= threshold}
        if common:
            return common
        return set(self.token_counts)

    @property
    def representative_embedding(self) -> list[float] | None:
        if self.embedding_sum is None or self.embedding_count <= 0:
            return None
        return [value / self.embedding_count for value in self.embedding_sum]

    def add_trace(self, trace: Node, tokens: set[str], embedding: list[float] | None) -> None:
        self.traces.append(trace)
        self.token_counts.update(tokens)
        if embedding is None:
            return
        if self.embedding_sum is None:
            self.embedding_sum = [0.0] * len(embedding)
        for index, value in enumerate(embedding):
            if index < len(self.embedding_sum):
                self.embedding_sum[index] += value
        self.embedding_count += 1


class ConsolidationService:
    """Service facade for consolidation and correction ingestion."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
        recent_limit: int = DEFAULT_RECENT_LIMIT,
    ) -> None:
        if min_cluster_size < 1:
            raise ValueError("min_cluster_size must be positive")
        self.store = store
        self.min_cluster_size = min_cluster_size
        self.recent_limit = recent_limit

    def memory_consolidate(
        self, *, scope: str | None = None, force: bool = False
    ) -> ConsolidationResult:
        return memory_consolidate(
            self.store,
            scope=scope,
            force=force,
            min_cluster_size=self.min_cluster_size,
            recent_limit=self.recent_limit,
        )

    def memory_teach(
        self,
        trace_id: str,
        correction: str | Mapping[str, Any],
        *,
        confidence: float | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> TeachResult:
        return memory_teach(
            self.store,
            trace_id,
            correction,
            confidence=confidence,
            context=context,
        )

    def memory_forget(self, node_id: str, reason: str | None = None) -> Node:
        return memory_forget(self.store, node_id, reason)


def memory_consolidate(
    store: MemoryStore,
    *,
    scope: str | None = None,
    force: bool = False,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    recent_limit: int = DEFAULT_RECENT_LIMIT,
) -> ConsolidationResult:
    """Cluster similar active traces, promote stable clusters, and run decay."""

    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be positive")

    trace_limit = max(recent_limit, min_cluster_size) if force else recent_limit
    traces = store.list_nodes(
        level="trace",
        scope=scope,
        include_decayed=False,
        limit=trace_limit,
    )
    phase = store.detect_phase()
    embedder = (
        LocalEmbeddingModel(model_name=store.config.embedding_model)
        if force or phase.number >= 2
        else None
    )
    clusters = _cluster_traces(traces, embedder=embedder)
    result = ConsolidationResult(
        clusters_considered=len(clusters),
        traces_considered=len(traces),
    )

    for cluster in clusters:
        if len(cluster.traces) < min_cluster_size:
            continue
        concept, created = _merge_cluster_into_concept(store, cluster)
        _update_edge_weights_from_co_access(store, concept, cluster.traces)
        if created:
            result.concepts_created.append(concept)
        else:
            result.concepts_updated.append(concept)

    decay_result: DecayResult = apply_decay(store, scope=scope)
    result.decayed.extend(decay_result.nodes)
    return result


def memory_teach(
    store: MemoryStore,
    trace_id: str,
    correction: str | Mapping[str, Any],
    *,
    confidence: float | None = None,
    context: Mapping[str, Any] | None = None,
) -> TeachResult:
    """Create a corrective trace and connect it to the original as superseding."""

    original = store.get_node(trace_id)
    if original is None:
        raise KeyError(trace_id)

    correction_text = _correction_text(correction)
    if not correction_text:
        raise ValueError("correction must be non-empty")

    context_data = dict(original.context)
    context_data.update(dict(context or {}))
    context_data.setdefault("scope", original.scope)
    context_data["timestamp"] = _utc_now()
    teacher = str(context_data.get("agent") or "teacher")
    context_data["agent"] = teacher

    requested_confidence = 0.5 if confidence is None else float(confidence)
    feedback = {
        "confidence": requested_confidence,
        "unique_agents": 1,
        "usefulness_score": max(1.0, original.usefulness_score + 1.0),
    }
    corrective_trace = store.append_trace(correction_text, context_data, feedback=feedback)

    supersedes = _upsert_weighted_connection(
        store,
        corrective_trace.id,
        original.id,
        "supersedes",
        weight=1.0,
        metadata={"by": teacher, "correction": correction_text},
    )

    corrected_original = store.add_correction(
        original.id,
        old=original.content,
        new=correction_text,
        by=teacher,
    )
    corrected_original = store.update_node(
        corrected_original.id,
        stats={
            "confidence": min(corrected_original.confidence, 0.25),
            "unique_agents": corrected_original.unique_agents,
            "usefulness_score": min(corrected_original.usefulness_score, -0.25),
        },
    )
    implicit_feedback = apply_pending_recall_feedback(
        store,
        corrective_trace,
        reinforce_results=False,
    )
    corrective_trace = implicit_feedback.trace
    return TeachResult(
        corrective_trace=corrective_trace,
        supersedes=supersedes,
        original=corrected_original,
        implicit_feedback=implicit_feedback,
    )


def _cluster_traces(
    traces: Iterable[Node],
    *,
    embedder: LocalEmbeddingModel | None = None,
) -> list[_TraceCluster]:
    clusters: list[_TraceCluster] = []
    for trace in traces:
        tokens = _significant_tokens(trace.content)
        embedding = _trace_embedding(trace, embedder)
        if not tokens and embedding is None:
            continue

        best_cluster: _TraceCluster | None = None
        best_score = 0.0
        best_strategy = "token-jaccard"
        for cluster in clusters:
            if cluster.scope != trace.scope:
                continue
            score, strategy = _cluster_similarity(tokens, embedding, cluster)
            if score > best_score:
                best_score = score
                best_strategy = strategy
                best_cluster = cluster

        if best_cluster is not None and _accept_cluster_match(best_score, best_strategy):
            best_cluster.strategy = best_strategy
            best_cluster.add_trace(trace, tokens, embedding)
        else:
            clusters.append(
                _TraceCluster(
                    trace.scope,
                    [trace],
                    Counter(tokens),
                    embedding_sum=list(embedding) if embedding is not None else None,
                    embedding_count=1 if embedding is not None else 0,
                    strategy="embedding-cosine" if embedding is not None else "token-jaccard",
                )
            )
    return clusters


def _merge_cluster_into_concept(
    store: MemoryStore, cluster: _TraceCluster
) -> tuple[Node, bool]:
    source_traces = sorted({trace.id for trace in cluster.traces})
    cluster_key = _cluster_key(cluster)
    unique_agents = _unique_agent_count(cluster.traces)
    confidence = _consensus_confidence(cluster.traces, unique_agents)
    temporal_hint = detect_temporal_hint(cluster.traces)
    best_trace = max(cluster.traces, key=_trace_quality)
    usefulness = sum(max(0.0, trace.usefulness_score) for trace in cluster.traces) / len(
        cluster.traces
    )

    provenance = {
        "source_traces": source_traces,
        "cluster_key": cluster_key,
        "cluster_size": len(source_traces),
        "consolidated_at": _utc_now(),
        "strategy": cluster.strategy,
    }
    stats = {
        "confidence": confidence,
        "unique_agents": unique_agents,
        "temporal_hint": temporal_hint,
        "usefulness_score": usefulness,
    }

    existing = _find_existing_concept(store, cluster.scope, cluster_key)
    if existing is None:
        concept = store.create_node(
            level="concept",
            content=best_trace.content,
            context={
                "scope": cluster.scope,
                "agent": "memory_consolidate",
                "timestamp": _utc_now(),
            },
            stats=stats,
            provenance=provenance,
        )
        return concept, True

    merged_sources = sorted({*existing.source_traces, *source_traces})
    provenance["source_traces"] = merged_sources
    provenance["cluster_size"] = len(merged_sources)
    concept = store.update_node(
        existing.id,
        content=best_trace.content if _trace_quality(best_trace) >= _node_quality(existing) else None,
        stats=stats,
        provenance={**existing.provenance, **provenance},
    )
    return concept, False


def _find_existing_concept(store: MemoryStore, scope: str, cluster_key: str) -> Node | None:
    concepts = store.list_nodes(
        level="concept",
        scope=scope,
        include_decayed=False,
        limit=100_000,
    )
    for concept in concepts:
        if concept.provenance.get("cluster_key") == cluster_key:
            return concept
    return None


def _trace_embedding(trace: Node, embedder: LocalEmbeddingModel | None) -> list[float] | None:
    if embedder is None:
        return None
    if trace.embedding is not None:
        return trace.embedding
    return embedder.embed(trace.content)


def _cluster_similarity(
    tokens: set[str],
    embedding: list[float] | None,
    cluster: _TraceCluster,
) -> tuple[float, str]:
    embedding_score = 0.0
    representative_embedding = cluster.representative_embedding
    if embedding is not None and representative_embedding is not None:
        embedding_score = cosine_similarity(embedding, representative_embedding)

    jaccard_score = _jaccard(tokens, cluster.representative_tokens)
    if embedding_score >= jaccard_score:
        return embedding_score, "embedding-cosine"
    return jaccard_score, "token-jaccard"


def _accept_cluster_match(score: float, strategy: str) -> bool:
    if strategy == "embedding-cosine":
        return score >= EMBEDDING_SIMILARITY_THRESHOLD
    return score >= JACCARD_SIMILARITY_THRESHOLD


def _update_edge_weights_from_co_access(
    store: MemoryStore, concept: Node, traces: Iterable[Node]
) -> None:
    trace_list = list(traces)
    if not trace_list:
        return

    max_access = max((trace.access_count for trace in trace_list), default=0)
    for trace in trace_list:
        weight = _source_edge_weight(trace, max_access)
        _upsert_weighted_connection(
            store,
            concept.id,
            trace.id,
            "related",
            weight=weight,
            metadata={"basis": "cluster", "co_access": trace.access_count},
        )

    co_accessed = sorted(
        (trace for trace in trace_list if trace.access_count > 0),
        key=lambda trace: (-trace.access_count, trace.id),
    )[:20]
    for index, left in enumerate(co_accessed):
        for right in co_accessed[index + 1 :]:
            source_id, target_id = sorted((left.id, right.id))
            weight = _pair_edge_weight(left, right, max_access)
            _upsert_weighted_connection(
                store,
                source_id,
                target_id,
                "related",
                weight=weight,
                metadata={"basis": "co_access", "left": left.access_count, "right": right.access_count},
            )


def _upsert_weighted_connection(
    store: MemoryStore,
    source_id: str,
    target_id: str,
    relation_type: str,
    *,
    weight: float,
    metadata: Mapping[str, Any] | None = None,
) -> Connection:
    existing = store.list_connections(
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
    )
    if existing:
        weight = max(float(weight), existing[0].weight)
        merged_metadata = {**existing[0].metadata, **dict(metadata or {})}
    else:
        merged_metadata = dict(metadata or {})
    return store.create_connection(
        source_id,
        target_id,
        relation_type,
        weight=round(min(1.0, max(0.0, weight)), 6),
        metadata=merged_metadata,
    )


def _source_edge_weight(trace: Node, max_access: int) -> float:
    access = trace.access_count / max_access if max_access > 0 else 0.0
    usefulness = min(1.0, max(0.0, trace.usefulness_score))
    return 0.2 + 0.55 * access + 0.2 * usefulness + 0.05 * trace.confidence


def _pair_edge_weight(left: Node, right: Node, max_access: int) -> float:
    if max_access <= 0:
        return 0.0
    co_access = min(left.access_count, right.access_count) / max_access
    return 0.25 + 0.75 * co_access


def _consensus_confidence(traces: Iterable[Node], unique_agents: int) -> float:
    trace_list = list(traces)
    if not trace_list:
        return 0.0
    if unique_agents <= 1:
        return 0.5

    average_usefulness = sum(max(0.0, trace.usefulness_score) for trace in trace_list) / len(
        trace_list
    )
    average_confidence = sum(trace.confidence for trace in trace_list) / len(trace_list)
    agent_signal = min(0.2, 0.05 * (unique_agents - 1))
    support_signal = min(0.2, len(trace_list) / 500)
    confidence = 0.5 + agent_signal + support_signal + 0.15 * average_usefulness
    return round(min(0.95, max(confidence, average_confidence)), 6)


def _unique_agent_count(traces: Iterable[Node]) -> int:
    agents = {trace.agent for trace in traces if trace.agent}
    declared = max((trace.unique_agents for trace in traces), default=0)
    return max(len(agents), declared, 1)


def _trace_quality(trace: Node) -> float:
    return (max(0.0, trace.usefulness_score) + 0.1) * (
        log1p(trace.access_count) + 1.0
    ) * max(trace.confidence, 0.1)


def _node_quality(node: Node) -> float:
    return (max(0.0, node.usefulness_score) + 0.1) * (
        log1p(node.access_count) + 1.0
    ) * max(node.confidence, 0.1)


def _cluster_key(cluster: _TraceCluster) -> str:
    tokens = [
        token
        for token, _count in cluster.token_counts.most_common(12)
        if token in cluster.representative_tokens
    ]
    if not tokens:
        tokens = [token for token, _count in cluster.token_counts.most_common(8)]
    return " ".join(sorted(tokens))


def _significant_tokens(content: str) -> set[str]:
    tokens = set()
    for raw in re.findall(r"\w+", content.lower().replace("ё", "е")):
        token = raw.strip("_")
        if len(token) < 3 or token.isdigit() or token in _STOP_WORDS:
            continue
        tokens.add(_normalize_token(token))
    return {token for token in tokens if token}


def _normalize_token(token: str) -> str:
    if not _is_latin_token(token):
        return token
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _is_latin_token(token: str) -> bool:
    return all(("a" <= char <= "z") or char.isdigit() or char == "_" for char in token)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _correction_text(correction: str | Mapping[str, Any]) -> str:
    if isinstance(correction, str):
        return correction.strip()
    for key in ("content", "new", "correction"):
        value = correction.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
