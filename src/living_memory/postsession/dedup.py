"""Exact proposal identity against bounded candidates from existing memory.

Retrieval may find related text; only equal content, scope and recorded
extraction source can suppress a proposal. Independent claims remain proposals
for the ordinary judge and write path.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .mining import hard_identifiers

SourceIdentity = tuple[str, str]


def extraction_identity(context: Mapping[str, Any]) -> SourceIdentity | None:
    """Recorded transcript and source span, when both are known."""

    transcript = context.get("source_transcript_sha256") or context.get("transcript_sha256")
    span = context.get("source_span")
    return (str(transcript), str(span)) if transcript and span else None


@dataclass(frozen=True, slots=True)
class NodeSnapshot:
    """Complete candidate content and its identity. Read-only."""

    node_id: str
    content: str
    scope: str | None = None
    created_at: str | None = None
    source_identity: SourceIdentity | None = None
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CandidateObservation:
    """One bounded read, retaining usable full candidates and read failures."""

    nodes: tuple[NodeSnapshot, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        return ("partial" if self.nodes else "unavailable") if self.errors else "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "read_status": self.status,
            "available": self.status == "success",
            "checked": len(self.nodes),
            "errors": list(self.errors),
        }


@runtime_checkable
class MemoryIndex(Protocol):
    """Read-only view of existing memory. No write method exists on purpose."""

    def candidates(
        self, content: str, *, scope: str | None = None, limit: int = 20
    ) -> CandidateObservation:
        """Plausible neighbours and the outcome of this bounded read."""


@dataclass(frozen=True, slots=True)
class DedupConfig:
    """Bounded candidate retrieval."""

    candidate_limit: int = 20


DEFAULT_DEDUP = DedupConfig()


@dataclass(frozen=True, slots=True)
class DedupVerdict:
    """Whether an exact repeat was found."""

    duplicate: bool
    channel: str = ""
    node_id: str = ""
    checked: int = 0
    read_status: str = "success"
    errors: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.read_status == "success"

    @property
    def code(self) -> str:
        """The gate rejection code this verdict maps to, or ``""``."""

        if not self.duplicate:
            return ""
        return "duplicate_within_run" if self.channel == "within_run" else (
            "duplicate_of_existing_node"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "duplicate": self.duplicate,
            "channel": self.channel,
            "node_id": self.node_id,
            "checked": self.checked,
            "available": self.available,
            "read_status": self.read_status,
            "errors": list(self.errors),
        }


# --------------------------------------------------------------------------
# Index implementations
# --------------------------------------------------------------------------


class NullMemoryIndex:
    """No memory configured for comparison."""

    def candidates(
        self, content: str, *, scope: str | None = None, limit: int = 20
    ) -> CandidateObservation:
        return CandidateObservation(errors=("memory index not configured",))


class StaticMemoryIndex:
    """An in-memory node set. Deterministic, no database, no model.

    Lexical candidates are ranked by shared hard identifiers, which is the
    cheap stand-in for FTS that keeps the offline tests honest: the same
    proposal that finds its twin through SQLite finds it here.
    """

    def __init__(self, nodes: Iterable[NodeSnapshot]) -> None:
        self.nodes = list(nodes)

    def candidates(
        self, content: str, *, scope: str | None = None, limit: int = 20
    ) -> CandidateObservation:
        wanted = set(hard_identifiers(content))
        scored: list[tuple[int, str, NodeSnapshot]] = []
        for node in self.nodes:
            if scope is not None and node.scope != scope:
                continue
            overlap = len(wanted & set(hard_identifiers(node.content))) if wanted else 0
            if overlap or not wanted:
                scored.append((-overlap, node.node_id, node))
        scored.sort()
        return CandidateObservation(tuple(node for _, _, node in scored[:limit]))


class StoreMemoryIndex:
    """Bounded FTS candidates from a caller-owned store; read errors propagate."""

    def __init__(self, store: Any, *, level: str = "trace") -> None:
        self.store = store
        self.level = level

    def candidates(
        self, content: str, *, scope: str | None = None, limit: int = 20
    ) -> CandidateObservation:
        query = " ".join(hard_identifiers(content)[:8]) or content[:200]
        rows = self.store.search_content(query, level=self.level, scope=scope, limit=limit)
        return CandidateObservation(tuple(
            NodeSnapshot(
                node_id=node.id,
                content=node.content,
                scope=getattr(node, "scope", None),
                created_at=getattr(node, "created_at", None),
                source_identity=extraction_identity(node.context),
            )
            for node, _score in rows
        ))


class Deduper:
    """Suppress exact repeats with the same scope and extraction provenance."""

    def __init__(
        self, index: MemoryIndex | None = None, *, config: DedupConfig = DEFAULT_DEDUP,
    ) -> None:
        self.index = index if index is not None else NullMemoryIndex()
        self.config = config
        self._proposed: dict[tuple[str, str, SourceIdentity], str] = {}
        self._lock = threading.RLock()

    def reset(self) -> None:
        with self._lock:
            self._proposed.clear()

    def check_within_run(
        self, fact: str, *, scope: str, source_identity: SourceIdentity | None,
    ) -> DedupVerdict:
        with self._lock:
            key = self._proposed.get((fact, scope, source_identity)) if source_identity else None
            return DedupVerdict(
                key is not None, "within_run" if key is not None else "",
                node_id=key or "", checked=len(self._proposed),
            )

    def remember_proposal(
        self, key: str, fact: str, *, scope: str, source_identity: SourceIdentity | None,
    ) -> None:
        if fact and source_identity:
            with self._lock:
                self._proposed[(fact, scope, source_identity)] = key

    def check(
        self, fact: str, *, scope: str, source_identity: SourceIdentity | None = None,
    ) -> DedupVerdict:
        with self._lock:
            observation = self.index.candidates(
                fact, scope=scope, limit=self.config.candidate_limit,
            )
        details = {
            "checked": len(observation.nodes), "read_status": observation.status,
            "errors": observation.errors,
        }
        for node in observation.nodes:
            if (fact and node.content == fact and node.scope == scope
                    and source_identity is not None and node.source_identity == source_identity):
                return DedupVerdict(True, "exact", node_id=node.node_id, **details)
        return DedupVerdict(False, **details)


def dedup_breakdown(verdicts: Iterable[DedupVerdict]) -> dict[str, Any]:
    """Match counts and incomplete reads; an exact match can also be partial.

    ``unique`` means no exact match in a successful bounded read, not absence.
    """

    counts = {"unique": 0, "exact": 0, "within_run": 0, "unavailable": 0, "partial": 0}
    for verdict in verdicts:
        if not verdict.available:
            counts[verdict.read_status] += 1
        if verdict.duplicate:
            counts[verdict.channel] += 1
        elif verdict.available:
            counts["unique"] += 1
    return counts
