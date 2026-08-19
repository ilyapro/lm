"""Near-duplicate detection against memory that already exists.

Why this is not the server's job today
--------------------------------------
The only write-time dedup Living Memory performs is byte identity within one
scope: ``MemoryStore._insert_node`` fingerprints ``content`` and links nodes
whose fingerprints match exactly (``storage.py``, the ``content_fingerprint``
lookup around lines 587-601). One added comma and the duplicate is a new node.
That is tolerable for a human-paced agent writing a handful of traces a
session; it is not tolerable for an extractor that will re-read the same
accumulated transcripts every time it runs. So the extractor dedups *before*
proposing, and does it on the two signals the retrieval stack itself uses:

**containment** -- :func:`living_memory.grounding.ground_results`, the same
IDF-weighted measure the live credit-assignment loop uses to decide whether a
trace actually used a recalled node. If a node already carries this fact's
tokens, the fact is already stored.

**cosine** -- :meth:`MemoryStore.find_similar_by_embedding` at the threshold
:data:`~living_memory.query_anchors.ANCHOR_DEDUP_COSINE_THRESHOLD` (0.95),
which is the precedent the codebase already set for "these two are the same
thing" and is imported rather than re-chosen here.

Two paths because they fail differently: containment misses a paraphrase that
shares no rare tokens, cosine misses a fact that differs only in one
identifier. A candidate has to survive both.

The store is reached through :class:`MemoryIndex`, a three-method protocol.
That is what lets the eval run work against a read-only snapshot, lets tests
run with no database at all, and keeps this module free of any write path --
proposing is this child's job, writing is the runner's.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from living_memory.grounding import ground_results

from .mining import hard_identifiers

__all__ = [
    "DEDUP_COSINE_THRESHOLD",
    "DedupConfig",
    "DEFAULT_DEDUP",
    "DedupVerdict",
    "NodeSnapshot",
    "MemoryIndex",
    "StaticMemoryIndex",
    "NullMemoryIndex",
    "StoreMemoryIndex",
    "Deduper",
]


def _anchor_cosine_threshold() -> float:
    """The 0.95 precedent, imported rather than re-invented.

    Imported lazily and defensively: ``query_anchors`` pulls in the storage
    layer, and this module must stay importable in an environment that has no
    database, which is how the offline tests run.
    """

    try:
        from living_memory.query_anchors import ANCHOR_DEDUP_COSINE_THRESHOLD
    except Exception:  # pragma: no cover - only when storage is unavailable
        return 0.95
    return float(ANCHOR_DEDUP_COSINE_THRESHOLD)


#: Cosine at or above which two nodes are the same node. See above.
DEDUP_COSINE_THRESHOLD = _anchor_cosine_threshold()


@dataclass(frozen=True, slots=True)
class NodeSnapshot:
    """The three fields dedup needs from an existing node. Read-only."""

    node_id: str
    content: str
    scope: str | None = None
    created_at: str | None = None


@runtime_checkable
class MemoryIndex(Protocol):
    """Read-only view of existing memory. No write method exists on purpose."""

    def candidates(
        self, content: str, *, scope: str | None = None, limit: int = 20
    ) -> Sequence[NodeSnapshot]:
        """Lexically plausible neighbours of ``content``."""

    def embedding(self, content: str) -> Sequence[float] | None:
        """Vector for ``content``, or ``None`` when this index has no encoder."""

    def similar_by_embedding(
        self,
        embedding: Sequence[float],
        *,
        scope: str | None = None,
        threshold: float = DEDUP_COSINE_THRESHOLD,
        limit: int = 10,
    ) -> Sequence[tuple[NodeSnapshot, float]]:
        """Nodes whose cosine to ``embedding`` reaches ``threshold``."""


@dataclass(frozen=True, slots=True)
class DedupConfig:
    """Thresholds. Tuned on ``train``, reported on ``eval``."""

    #: IDF containment of the proposal inside an existing node's content.
    min_containment: float = 0.75
    #: Cosine at which the vector channel calls it a duplicate.
    min_cosine: float = DEDUP_COSINE_THRESHOLD
    #: Lexical neighbours fetched per proposal.
    candidate_limit: int = 20
    #: Identifiers used to build the lexical query -- the fact's own anchors
    #: are a far better retrieval key than its prose.
    query_identifiers: int = 8
    #: Containment at which two proposals *within one run* collapse.
    within_run_containment: float = 0.70
    #: Look for the twin in every scope, not only the one being proposed into.
    #: This is the point of the module: ``_insert_node`` already dedups inside
    #: one scope (by byte identity), and ``memory_recall`` reads broad, so a
    #: near-twin sitting in ``global`` is a duplicate from the reader's side
    #: even though the server would happily store it again.
    cross_scope: bool = True


DEFAULT_DEDUP = DedupConfig()


@dataclass(frozen=True, slots=True)
class DedupVerdict:
    """Whether a proposal survives, and which channel killed it."""

    duplicate: bool
    channel: str = ""
    node_id: str = ""
    containment: float = 0.0
    cosine: float = 0.0
    checked: int = 0
    available: bool = True

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
            "containment": round(self.containment, 4),
            "cosine": round(self.cosine, 4),
            "checked": self.checked,
            "available": self.available,
        }


# --------------------------------------------------------------------------
# Index implementations
# --------------------------------------------------------------------------


class NullMemoryIndex:
    """No memory to compare against.

    Used when the eval run has no database snapshot. It returns nothing rather
    than pretending everything is new, and :class:`Deduper` marks the verdict
    ``available=False`` so the report can say "dedup did not run" instead of
    silently reporting zero duplicates.
    """

    available = False

    def candidates(
        self, content: str, *, scope: str | None = None, limit: int = 20
    ) -> Sequence[NodeSnapshot]:
        return ()

    def embedding(self, content: str) -> Sequence[float] | None:
        return None

    def similar_by_embedding(
        self,
        embedding: Sequence[float],
        *,
        scope: str | None = None,
        threshold: float = DEDUP_COSINE_THRESHOLD,
        limit: int = 10,
    ) -> Sequence[tuple[NodeSnapshot, float]]:
        return ()


class StaticMemoryIndex:
    """An in-memory node set. Deterministic, no database, no model.

    Lexical candidates are ranked by shared hard identifiers, which is the
    cheap stand-in for FTS that keeps the offline tests honest: the same
    proposal that finds its twin through SQLite finds it here.
    """

    available = True

    def __init__(
        self,
        nodes: Iterable[NodeSnapshot],
        *,
        embeddings: Mapping[str, Sequence[float]] | None = None,
        encoder: Any | None = None,
    ) -> None:
        self.nodes = list(nodes)
        self.embeddings = dict(embeddings or {})
        self.encoder = encoder

    def candidates(
        self, content: str, *, scope: str | None = None, limit: int = 20
    ) -> Sequence[NodeSnapshot]:
        wanted = set(hard_identifiers(content))
        scored: list[tuple[int, str, NodeSnapshot]] = []
        for node in self.nodes:
            if scope is not None and node.scope is not None and node.scope != scope:
                continue
            overlap = len(wanted & set(hard_identifiers(node.content))) if wanted else 0
            if overlap or not wanted:
                scored.append((-overlap, node.node_id, node))
        scored.sort()
        return [node for _, _, node in scored[:limit]]

    def embedding(self, content: str) -> Sequence[float] | None:
        if self.encoder is None:
            return None
        return self.encoder.embed(content)

    def similar_by_embedding(
        self,
        embedding: Sequence[float],
        *,
        scope: str | None = None,
        threshold: float = DEDUP_COSINE_THRESHOLD,
        limit: int = 10,
    ) -> Sequence[tuple[NodeSnapshot, float]]:
        from living_memory.embeddings import cosine_similarity

        matches: list[tuple[NodeSnapshot, float]] = []
        for node in self.nodes:
            vector = self.embeddings.get(node.node_id)
            if vector is None:
                continue
            if scope is not None and node.scope is not None and node.scope != scope:
                continue
            score = cosine_similarity(embedding, vector)
            if score >= threshold:
                matches.append((node, score))
        matches.sort(key=lambda item: (-item[1], item[0].node_id))
        return matches[:limit]


class StoreMemoryIndex:
    """The real thing: FTS neighbours plus the store's own cosine scan.

    Read-only by construction -- it calls ``search_content``,
    ``find_similar_by_embedding`` and nothing else. Open the store read-only
    if the caller can; this class does not, because deciding how to open a
    database belongs to whoever owns the connection.
    """

    available = True

    def __init__(self, store: Any, *, encoder: Any | None = None, level: str = "trace") -> None:
        self.store = store
        self.level = level
        self._encoder = encoder

    @property
    def encoder(self) -> Any | None:
        """The store's embedder when it has one, else a lazily built local model."""

        if self._encoder is None:
            self._encoder = getattr(self.store, "embedding_model", None) or getattr(
                self.store, "embedder", None
            )
        if self._encoder is None:
            from living_memory.embeddings import LocalEmbeddingModel

            self._encoder = LocalEmbeddingModel()
        return self._encoder

    def candidates(
        self, content: str, *, scope: str | None = None, limit: int = 20
    ) -> Sequence[NodeSnapshot]:
        query = " ".join(hard_identifiers(content)[:8]) or content[:200]
        try:
            rows = self.store.search_content(query, level=self.level, scope=scope, limit=limit)
        except Exception:  # pragma: no cover - a malformed FTS query must not abort a run
            return ()
        return [
            NodeSnapshot(
                node_id=node.id,
                content=node.content,
                scope=getattr(node, "scope", None),
                created_at=getattr(node, "created_at", None),
            )
            for node, _score in rows
        ]

    def embedding(self, content: str) -> Sequence[float] | None:
        encoder = self.encoder
        return None if encoder is None else encoder.embed(content)

    def similar_by_embedding(
        self,
        embedding: Sequence[float],
        *,
        scope: str | None = None,
        threshold: float = DEDUP_COSINE_THRESHOLD,
        limit: int = 10,
    ) -> Sequence[tuple[NodeSnapshot, float]]:
        rows = self.store.find_similar_by_embedding(
            list(embedding),
            level=self.level,
            scope=scope,
            threshold=threshold,
            limit=limit,
        )
        return [
            (
                NodeSnapshot(
                    node_id=node.id,
                    content=node.content,
                    scope=getattr(node, "scope", None),
                    created_at=getattr(node, "created_at", None),
                ),
                float(score),
            )
            for node, score in rows
        ]


# --------------------------------------------------------------------------
# The deduper
# --------------------------------------------------------------------------


def _containment(fact: str, node_content: str) -> float:
    """Share of ``fact``'s IDF mass that ``node_content`` already carries."""

    if not fact.strip() or not node_content.strip():
        return 0.0
    graded = ground_results(node_content, {"fact": fact}, min_containment=1.1)
    verdict = graded.get("fact")
    return float(verdict.containment) if verdict else 0.0


class Deduper:
    """Containment first, cosine second, plus a within-run memory.

    Containment runs first because it is free once the candidates are fetched
    and because its verdict is explainable ("node X already contains 0.83 of
    this fact's tokens"). The vector pass only runs for proposals containment
    let through, which is also what keeps the eval run affordable: encoding is
    the expensive half.
    """

    def __init__(
        self,
        index: MemoryIndex | None = None,
        *,
        config: DedupConfig = DEFAULT_DEDUP,
    ) -> None:
        self.index = index if index is not None else NullMemoryIndex()
        self.config = config
        self._proposed: list[tuple[str, str]] = []
        # One deduper is shared by every session of a run, and a run may judge
        # sessions concurrently. The within-run history and the store's single
        # sqlite connection are both serialized through this.
        self._lock = threading.RLock()

    @property
    def available(self) -> bool:
        return bool(getattr(self.index, "available", True))

    def reset(self) -> None:
        """Forget the within-run history. Call between independent runs."""

        with self._lock:
            self._proposed.clear()

    def check_within_run(self, fact: str) -> DedupVerdict:
        """Has this run already proposed the same thing?

        Cheap and stateful, and it runs before the store lookup: two candidates
        mined from the same failure routinely produce one fact twice, and
        paying for an FTS query and an encode to discover that is waste.
        """

        with self._lock:
            history = list(self._proposed)
        for key, previous in history:
            value = _containment(fact, previous)
            if value >= self.config.within_run_containment:
                return DedupVerdict(
                    True,
                    "within_run",
                    node_id=key,
                    containment=value,
                    checked=len(history),
                )
        return DedupVerdict(False, checked=len(history))

    def remember_proposal(self, key: str, fact: str) -> None:
        """Record an accepted proposal so later ones can collapse against it."""

        with self._lock:
            self._proposed.append((key, fact))

    def check(self, fact: str, *, scope: str | None = None) -> DedupVerdict:
        """Is ``fact`` already in memory? Never raises; degrades to "no".

        ``scope`` is the scope the caller intends to *write* into. Whether the
        lookup is restricted to it is :attr:`DedupConfig.cross_scope`, which
        defaults to searching everywhere -- see that field's comment.
        """

        if not self.available:
            return DedupVerdict(False, available=False)
        lookup_scope = None if self.config.cross_scope else scope
        with self._lock:
            neighbours = list(
                self.index.candidates(
                    fact, scope=lookup_scope, limit=self.config.candidate_limit
                )
            )
        best = 0.0
        best_id = ""
        for node in neighbours:
            value = _containment(fact, node.content)
            if value > best:
                best, best_id = value, node.node_id
        if best >= self.config.min_containment:
            return DedupVerdict(
                True,
                "containment",
                node_id=best_id,
                containment=best,
                checked=len(neighbours),
            )

        with self._lock:
            vector = self.index.embedding(fact)
            if vector is None:
                return DedupVerdict(False, containment=best, checked=len(neighbours))
            matches = self.index.similar_by_embedding(
                vector, scope=lookup_scope, threshold=self.config.min_cosine, limit=5
            )
        if matches:
            node, score = matches[0]
            return DedupVerdict(
                True,
                "embedding",
                node_id=node.node_id,
                containment=_containment(fact, node.content),
                cosine=score,
                checked=len(neighbours) + len(matches),
            )
        return DedupVerdict(False, containment=best, checked=len(neighbours))


def dedup_breakdown(verdicts: Iterable[DedupVerdict]) -> dict[str, Any]:
    """Per-channel counts for the run report."""

    counts: dict[str, int] = {"unique": 0, "containment": 0, "embedding": 0, "within_run": 0}
    unavailable = 0
    for verdict in verdicts:
        if not verdict.available:
            unavailable += 1
            continue
        if verdict.duplicate:
            counts[verdict.channel] = counts.get(verdict.channel, 0) + 1
        else:
            counts["unique"] += 1
    return {**counts, "unavailable": unavailable}
