"""Data models shared by storage, retrieval, and server layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
import re
import json

NodeLevel = Literal["trace", "concept", "schema"]
ConnectionType = Literal["related", "caused", "contradicts", "supersedes", "requires"]

NODE_LEVELS: tuple[str, ...] = ("trace", "concept", "schema")
CONNECTION_TYPES: tuple[str, ...] = (
    "related",
    "caused",
    "contradicts",
    "supersedes",
    "requires",
)
REJECTED_ALTERNATIVE_KIND = "rejected_alternative"


def content_identity(node: Node) -> tuple[Any, ...] | None:
    """Exact content within its scope and recorded source identity.

    Materialized knowledge needs known inputs and group. For ordinary nodes,
    source context and provenance stay part of identity; timestamps and recall
    bookkeeping do not make a repeated write a new source. Level is a separate
    storage lifecycle boundary; delivery may share identical ordinary content.
    """
    if node.materializer is not None:
        source = node.derivation_identity
        return (node.scope, node.content, source) if source is not None else None
    context = {key: value for key, value in node.context.items() if key != "timestamp"}
    # Scope is represented by node.scope, including on in-memory callers.
    context.pop("scope", None)
    provenance = {key: value for key, value in node.provenance.items()
                  if key not in ("prior_recalls", "source_traces", "corrections")}
    source = json.dumps(
        [context, provenance, sorted(node.source_traces), node.corrections],
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return node.scope, node.content, source


@dataclass(slots=True)
class Node:
    """Uniform memory node used for traces, concepts, and schemas."""

    id: str
    level: NodeLevel
    content: str
    context: dict[str, Any] = field(default_factory=dict)
    # Legacy whole-node similarity: materializers can retain a source representative.
    # Content search reads independently encoded node_chunk_embeddings.
    embedding: list[float] | None = None
    scope: str = "global"
    agent: str | None = None
    task: str | None = None
    timestamp: str = ""
    decayed: bool = False
    decay_reason: str | None = None
    access_count: int = 0
    last_accessed: str | None = None
    usefulness_score: float = 0.0
    confidence: float = 0.5
    unique_agents: int = 1
    temporal_hint: str | None = None
    source_traces: list[str] = field(default_factory=list)
    corrections: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    @property
    def materializer(self) -> str | None:
        """The recorded writer, distinct from incidental recall citations."""

        if self.level not in ("concept", "schema"):
            return None
        strategy = self.provenance.get("strategy")
        if strategy in ("procedural", "cross-scope-promotion"):
            return strategy
        if self.level == "concept" and (
            strategy in ("token-jaccard", "embedding-cosine")
            or (strategy is None and "cluster_key" in self.provenance)
        ):
            return "cluster"
        return None

    @property
    def materializer_sources(self) -> list[str]:
        if self.materializer == "cross-scope-promotion":
            return string_list(self.provenance.get("source_concepts"))
        return self.source_traces if self.materializer else []

    @property
    def derivation_identity(self) -> tuple[str, str, frozenset[str]] | None:
        """Recorded writer, group and inputs; content and labels are not keys."""

        materializer = self.materializer
        if materializer is None or not self.materializer_sources:
            return None
        raw = (self.context.get("procedure_key") or self.provenance.get("procedure_key")
               or self.context.get("task_pattern") or self.context.get("procedure_id") or "")
        group = re.sub(r"[\s_/-]+", " ", str(raw)).strip().lower()
        return materializer, group, frozenset(self.materializer_sources)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "usefulness_score": self.usefulness_score,
            "confidence": self.confidence,
            "unique_agents": self.unique_agents,
            "temporal_hint": self.temporal_hint,
        }

    def to_dict(self) -> dict[str, Any]:
        provenance = dict(self.provenance)
        provenance["source_traces"] = list(self.source_traces)
        provenance["corrections"] = list(self.corrections)
        return {
            "id": self.id,
            "level": self.level,
            "content": self.content,
            "embedding": self.embedding,
            "context": dict(self.context),
            "stats": self.stats,
            "provenance": provenance,
            "decayed": self.decayed,
            "decay_reason": self.decay_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class SourcePassage:
    """Character address in one exact content version, without a text copy."""

    content_fingerprint: str
    char_start: int
    char_end: int


@dataclass(slots=True)
class NodeChunkEmbedding:
    """One embedding window of a node, stored as a float32 little-endian BLOB.

    A node owns zero or more of these, ordered by ``chunk_index`` starting at 0
    with no gaps. ``dimensions`` is recorded per row and is the only authority
    on the vector's shape — never infer it from ``len(blob) // 4`` and never
    assume 384. ``content_fingerprint`` is the parent's
    ``nodes.content_fingerprint`` as it stood when the chunk was embedded, so a
    reader can tell a current chunk from one left behind by an edit.

    ``token_start``/``token_end`` are the half-open token span of the parent's
    content covered by this window, carried over from
    :class:`living_memory.chunking.TextChunk`; consecutive chunks overlap.
    """

    id: str
    node_id: str
    chunk_index: int
    dimensions: int
    embedding: list[float]
    token_start: int
    token_end: int
    content_fingerprint: str
    created_at: str = ""
    updated_at: str = ""
    char_start: int | None = None
    char_end: int | None = None

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


@dataclass(slots=True)
class Connection:
    """Adjacency edge between two memory nodes."""

    id: str
    source_id: str
    target_id: str
    type: ConnectionType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class RecallEvent:
    """Durable provenance record for one recall interaction."""

    id: str
    query: str
    scope: str
    requested_scope: str
    resolved_scopes: list[str] = field(default_factory=list)
    ambient_context: dict[str, Any] = field(default_factory=dict)
    depth: str | None = None
    max_results: int = 10
    results: list[dict[str, Any]] = field(default_factory=list)
    agent: str | None = None
    task: str | None = None
    session_id: str | None = None
    transport_session_id: str | None = None
    feedback_applied: bool = False
    feedback_trace_id: str | None = None
    feedback_applied_at: str | None = None
    created_at: str = ""

    @property
    def result_ids(self) -> list[str]:
        ids: list[str] = []
        for item in self.results:
            node_id = item.get("node_id")
            if node_id:
                ids.append(str(node_id))
        return ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "scope": self.scope,
            "requested_scope": self.requested_scope,
            "resolved_scopes": list(self.resolved_scopes),
            "ambient_context": dict(self.ambient_context),
            "depth": self.depth,
            "max_results": self.max_results,
            "result_ids": self.result_ids,
            "results": [dict(item) for item in self.results],
            "agent": self.agent,
            "task": self.task,
            "session_id": self.session_id,
            "transport_session_id": self.transport_session_id,
            "feedback_applied": self.feedback_applied,
            "feedback_trace_id": self.feedback_trace_id,
            "feedback_applied_at": self.feedback_applied_at,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class RetrievalWeights:
    """Self-tuning retrieval policy weights for a scope family."""

    scope: str
    bm25: float
    vector: float
    graph: float
    learning_rate: float = 0.05
    updated_at: str = ""

    def normalized(self) -> "RetrievalWeights":
        total = self.bm25 + self.vector + self.graph
        if total <= 0:
            return RetrievalWeights(self.scope, 1.0, 0.0, 0.0, self.learning_rate, self.updated_at)
        return RetrievalWeights(
            self.scope,
            self.bm25 / total,
            self.vector / total,
            self.graph / total,
            self.learning_rate,
            self.updated_at,
        )


@dataclass(frozen=True, slots=True)
class RetrievalPolicyFloors:
    """Minimum retrieval-channel policy for an active scope family."""

    bm25_max: float
    vector_min: float
    graph_min: float
    bm25_min: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("bm25_max", self.bm25_max),
            ("vector_min", self.vector_min),
            ("graph_min", self.graph_min),
            ("bm25_min", self.bm25_min),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if self.bm25_min > self.bm25_max:
            raise ValueError("bm25_min must not exceed bm25_max")
        if self.vector_min + self.graph_min + self.bm25_min > 1.0:
            raise ValueError("vector_min + graph_min + bm25_min must be at most 1.0")


@dataclass(frozen=True, slots=True)
class PhaseInfo:
    """Detected automatic capability phase for the current trace count."""

    number: int
    name: str
    trace_count: int
    min_traces: int
    features: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"Phase {self.number}: {self.name}"


def string_list(value: Any) -> list[str]:
    """Return a string list from JSON-like provenance values."""

    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
