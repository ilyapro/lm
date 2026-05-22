"""Data models shared by storage, retrieval, and server layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

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


@dataclass(slots=True)
class Node:
    """Uniform memory node used for traces, concepts, and schemas."""

    id: str
    level: NodeLevel
    content: str
    context: dict[str, Any] = field(default_factory=dict)
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

    def __post_init__(self) -> None:
        for name, value in (
            ("bm25_max", self.bm25_max),
            ("vector_min", self.vector_min),
            ("graph_min", self.graph_min),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if self.vector_min + self.graph_min > 1.0:
            raise ValueError("vector_min + graph_min must be at most 1.0")


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
