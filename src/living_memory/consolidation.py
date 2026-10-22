"""Learning and maintenance loop for Living Memory.

Cluster promotion floor
-----------------------
A cluster of similar traces is promoted into a concept once it holds at
least ``min_cluster_size`` members. That floor is **adaptive by default**:
``adaptive_merge_floor`` scales it with the number of active traces the pass
actually loaded, so a young corpus grows concepts over mid-sized clusters
instead of waiting for the hundreds a mature one accumulates.

``LM_CONSOLIDATE_MERGE_FLOOR`` is the rollback valve, read on every pass —
by the auto-consolidation path in ``server.py`` and by a manual
``memory_consolidate`` call alike:

* unset, empty or ``adaptive`` — the ladder above (shipped default);
* ``fixed`` — the pre-adaptive behaviour, floor ``DEFAULT_MIN_CLUSTER_SIZE``
  (100) on both paths;
* a positive integer — that literal floor, for pinning one by hand.

Anything else falls back to ``adaptive`` rather than failing a maintenance
pass. An explicit ``min_cluster_size=`` argument always wins over the valve
and stays validated as positive.
"""

from __future__ import annotations

from collections import Counter
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from math import log1p
from typing import Any, Callable, Iterable, Mapping, Sequence
import logging
import os
import re

from living_memory.decay import DecayResult, apply_decay, memory_forget
from living_memory.edge_derivation import DERIVED_FROM_ANNOTATION, DERIVED_FROM_KIND
from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity
from living_memory.feedback import ImplicitRecallFeedback, apply_pending_recall_feedback
from living_memory.models import Connection, ConnectionType, Node, string_list
from living_memory.query_anchors import resolve_replacement
from living_memory.storage import _DUPLICATE_CONTENT_KIND, MemoryStore, _node_read_batches, new_ulid
from living_memory.temporal import detect_temporal_hint, parse_timestamp, split_time_regimes

_LOG = logging.getLogger(__name__)

try:
    import numpy as _np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - numpy is a normal runtime dep
    _np = None  # type: ignore[assignment]

DEFAULT_MIN_CLUSTER_SIZE = 100
_ENCODE_BATCH = 8
_CACHE_READ_BATCH = 500
MERGE_FLOOR_ENV = "LM_CONSOLIDATE_MERGE_FLOOR"
# (active traces below, floor to apply). Falls through to
# DEFAULT_MIN_CLUSTER_SIZE once the corpus is mature.
ADAPTIVE_MERGE_FLOOR_LADDER: tuple[tuple[int, int], ...] = (
    (25, 3),
    (100, 5),
    (1000, 25),
)
DEFAULT_RECENT_LIMIT = 10_000
JACCARD_SIMILARITY_THRESHOLD = 0.58
EMBEDDING_SIMILARITY_THRESHOLD = 0.65
CROSS_SCOPE_PROMOTION_PHASE = 4
CROSS_SCOPE_PROMOTION_THRESHOLD = 0.7
PROCEDURAL_MIN_CLUSTER_SIZE = 3
_STEP_TEXT_KEYS = ("step_description", "step_content")
DIGEST_MAX_CHARS = 1200
DIGEST_LEAD_MAX_CHARS = 600
DIGEST_FACT_MAX_CHARS = 280
DIGEST_FACT_PREFIX = "• "
DIGEST_NEAR_DUPLICATE_JACCARD = 0.75
DIGEST_FRAGMENT_MAX_TOKENS = 12

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
    concepts_promoted: list[Node] = field(default_factory=list)
    schemas_created: list[Node] = field(default_factory=list)
    schemas_updated: list[Node] = field(default_factory=list)
    decayed: list[Node] = field(default_factory=list)
    clusters_considered: int = 0
    traces_considered: int = 0

    @property
    def concepts(self) -> list[Node]:
        return [*self.concepts_created, *self.concepts_updated, *self.concepts_promoted]

    @property
    def schemas(self) -> list[Node]:
        return [*self.schemas_created, *self.schemas_updated]


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
    embeddings_by_trace_id: dict[str, list[float]] = field(default_factory=dict)
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
        self.embeddings_by_trace_id[trace.id] = list(embedding)
        if self.embedding_sum is None:
            self.embedding_sum = [0.0] * len(embedding)
        for index, value in enumerate(embedding):
            if index < len(self.embedding_sum):
                self.embedding_sum[index] += value
        self.embedding_count += 1

    def embedding_for(self, trace: Node) -> list[float] | None:
        if trace.embedding is not None:
            return trace.embedding
        return self.embeddings_by_trace_id.get(trace.id)


@dataclass(frozen=True, slots=True)
class _ProcedureKey:
    """Compound key for procedural distillation.

    `group_*` is the stable identifier used to cluster traces into one schema —
    `task_pattern` is preferred because real consumers (e.g. cross-tree
    consolidator) write a constant sha-derived hash there while the
    `procedure_id` label can vary across instances of the same pattern.

    `trigger`/`field`/`raw` is the human-readable label used for the schema's
    trigger string — `procedure_id` is preferred so retrieval can match natural
    language queries against the schema's trigger.
    """

    field: str
    raw: str
    trigger: str
    group_field: str
    group_raw: str
    group_id: str


@dataclass(slots=True)
class _EraAssessment:
    """Structural era analysis of one promotion candidate cluster.

    ``live`` holds the members a schema/concept may be derived from;
    ``excluded`` maps every dropped member to the structural reason
    (``corrected``, ``superseded-by:<id>``, ``displaced-via:<id>``,
    ``obsolete-era``). ``conflict`` is set only when contradiction evidence
    crosses a regime boundary — a temporal split alone never filters anything.
    """

    members: list[Node]
    live: list[Node]
    excluded: dict[str, str]
    conflict: bool
    regime_count: int

    @property
    def filtered(self) -> bool:
        return bool(self.excluded)

    def era_provenance(self) -> dict[str, Any]:
        live_moments = sorted(
            moment for member in self.live if (moment := _node_moment(member)) is not None
        )
        return {
            "status": "current",
            "policy": "current-era-only",
            "conflict": self.conflict,
            "regimes_detected": self.regime_count,
            "live_window": [
                _iso(live_moments[0]) if live_moments else None,
                _iso(live_moments[-1]) if live_moments else None,
            ],
            "excluded_sources": dict(sorted(self.excluded.items())),
            "assessed_at": _utc_now(),
        }


def _node_moment(node: Node) -> datetime | None:
    return parse_timestamp(node.timestamp) or parse_timestamp(node.created_at)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


def _assess_cluster_eras(store: MemoryStore, members: Sequence[Node]) -> _EraAssessment:
    """Detect whether a promotion candidate spans contradictory project eras.

    Detection is purely structural: teach corrections, supersedes/contradicts
    edges (including a member superseding a consolidated node whose sources
    are other members), and timestamp regimes from ``split_time_regimes``.
    Individually corrected members are always excluded from derivation; whole
    older regimes are excluded only on ``conflict`` — when some corrective
    evidence lands in a strictly later regime than the material it corrects.
    """

    member_list = list(store.get_nodes([member.id for member in members]).values())
    member_ids = {member.id for member in member_list}
    moments = {member.id: _node_moment(member) for member in member_list}

    excluded: dict[str, str] = {}
    # (obsolete-side moment, corrective-side moment) pairs. A pair whose
    # corrective side falls in a strictly later regime than its obsolete side
    # is the era-flip evidence.
    conflict_events: list[tuple[datetime | None, datetime | None]] = []

    for member in member_list:
        if member.corrections:
            excluded[member.id] = "corrected"
            last = member.corrections[-1]
            correction_moment = (
                parse_timestamp(last.get("timestamp")) if isinstance(last, Mapping) else None
            )
            conflict_events.append((moments[member.id], correction_moment))

    external_cache: dict[str, Node | None] = {}

    def _external(node_id: str) -> Node | None:
        if node_id not in external_cache:
            external_cache[node_id] = store.get_node(node_id)
        return external_cache[node_id]

    def _moment_of(node_id: str, fallback: str) -> datetime | None:
        if node_id in moments:
            return moments[node_id]
        node = _external(node_id)
        if node is not None:
            return _node_moment(node)
        return parse_timestamp(fallback)

    seen_edges: set[str] = set()
    for edges in store.list_connections_for_nodes(member_ids).values():
        for edge in edges:
            if edge.id in seen_edges or edge.type not in ("supersedes", "contradicts"):
                continue
            seen_edges.add(edge.id)
            if edge.metadata.get("kind") == _DUPLICATE_CONTENT_KIND:
                # Re-remembered identical content affirms the fact; it is not
                # a correction and must not read as era conflict.
                continue
            source_moment = _moment_of(edge.source_id, edge.created_at)
            target_moment = _moment_of(edge.target_id, edge.created_at)
            if edge.type == "supersedes":
                if edge.target_id in member_ids:
                    excluded.setdefault(edge.target_id, f"superseded-by:{edge.source_id}")
                    conflict_events.append((target_moment, source_moment))
                if edge.source_id in member_ids and edge.target_id not in member_ids:
                    target = _external(edge.target_id)
                    if target is not None:
                        displaced = (set(_derivation_sources(store, target)) & member_ids) - {edge.source_id}
                        for displaced_id in sorted(displaced):
                            excluded.setdefault(
                                displaced_id, f"displaced-via:{edge.target_id}"
                            )
                            conflict_events.append((moments[displaced_id], source_moment))
                    conflict_events.append((target_moment, source_moment))
            else:
                # contradicts carries no direction of truth; with timestamps
                # the newer side counts as the corrective one.
                if source_moment is None or target_moment is None:
                    continue
                older, newer = sorted((source_moment, target_moment))
                conflict_events.append((older, newer))

    regime_groups = split_time_regimes([moments[member.id] for member in member_list])
    dated_groups = [[member_list[index].id for index in group] for group in regime_groups]
    if not dated_groups:
        dated_groups = [[]]
    # Members without a parseable moment stay with the newest regime: a
    # missing timestamp must never mark a member obsolete.
    dated_groups[-1].extend(
        member.id for member in member_list if moments[member.id] is None
    )
    regime_count = len(dated_groups)

    conflict = False
    if regime_count >= 2:
        starts = [
            min(group_moments)
            for group in dated_groups
            if (group_moments := [moments[mid] for mid in group if moments[mid] is not None])
        ]

        def _regime_index(moment: datetime) -> int:
            return sum(1 for start in starts if start <= moment)

        for obsolete_moment, corrective_moment in conflict_events:
            if obsolete_moment is None or corrective_moment is None:
                continue
            obsolete_regime = _regime_index(obsolete_moment)
            if obsolete_regime >= 1 and _regime_index(corrective_moment) > obsolete_regime:
                conflict = True
                break
        if conflict:
            newest_ids = set(dated_groups[-1])
            for member in member_list:
                if member.id not in newest_ids:
                    excluded.setdefault(member.id, "obsolete-era")

    live = [member for member in member_list if member.id not in excluded]
    return _EraAssessment(
        members=member_list,
        live=live,
        excluded=excluded,
        conflict=conflict,
        regime_count=regime_count,
    )


def _is_superseded(store: MemoryStore, node_id: str) -> bool:
    return any(
        edge.metadata.get("kind") != _DUPLICATE_CONTENT_KIND
        for edge in store.list_connections(target_id=node_id, relation_type="supersedes")
    )


def adaptive_merge_floor(trace_count: int) -> int:
    """Cluster size required to promote a concept in a corpus this size.

    The single owner of the floor ladder: both the auto-consolidation path in
    ``server.py`` and a manual ``memory_consolidate`` call reach it through
    ``resolve_min_cluster_size``.
    """

    for ceiling, floor in ADAPTIVE_MERGE_FLOOR_LADDER:
        if trace_count < ceiling:
            return floor
    return DEFAULT_MIN_CLUSTER_SIZE


def resolve_min_cluster_size(min_cluster_size: int | None, *, trace_count: int) -> int:
    """Pick the promotion floor for one pass, honouring the rollback valve.

    An explicit caller value wins outright. Otherwise ``MERGE_FLOOR_ENV``
    decides, as documented in the module docstring: adaptive by default,
    ``fixed`` for the pre-adaptive floor of ``DEFAULT_MIN_CLUSTER_SIZE``, a
    positive integer to pin one. An unparseable value must not break a
    maintenance pass, so it reads as the default.
    """

    if min_cluster_size is not None:
        return min_cluster_size

    raw = os.environ.get(MERGE_FLOOR_ENV, "").strip().lower()
    if raw == "fixed":
        return DEFAULT_MIN_CLUSTER_SIZE
    if raw and raw != "adaptive":
        try:
            pinned = int(raw)
        except ValueError:
            pinned = 0
        if pinned >= 1:
            return pinned
    return adaptive_merge_floor(trace_count)


class ConsolidationService:
    """Service facade for consolidation and correction ingestion."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        min_cluster_size: int | None = None,
        recent_limit: int = DEFAULT_RECENT_LIMIT,
    ) -> None:
        if min_cluster_size is not None and min_cluster_size < 1:
            raise ValueError("min_cluster_size must be positive")
        self.store = store
        # None keeps the service on the adaptive default; a number pins the
        # floor for every pass this service runs.
        self.min_cluster_size = min_cluster_size
        self.recent_limit = recent_limit

    def memory_consolidate(
        self,
        *,
        scope: str | None = None,
        force: bool = False,
        min_cluster_size: int | None = None,
        guard: Callable[[], AbstractContextManager[Any]] | None = None,
    ) -> ConsolidationResult:
        return memory_consolidate(
            self.store,
            scope=scope,
            force=force,
            min_cluster_size=(
                self.min_cluster_size if min_cluster_size is None else min_cluster_size
            ),
            recent_limit=self.recent_limit,
            guard=guard,
        )

    def memory_teach(
        self,
        trace_id: str,
        correction: str | Mapping[str, Any],
        *,
        confidence: float | None = None,
        context: Mapping[str, Any] | None = None,
        preparation: _TeachPreparation | None = None,
    ) -> TeachResult:
        return memory_teach(
            self.store,
            trace_id,
            correction,
            confidence=confidence,
            context=context,
            preparation=preparation,
        )

    def memory_forget(self, node_id: str, reason: str | None = None) -> Node:
        return memory_forget(self.store, node_id, reason)


def memory_consolidate(
    store: MemoryStore,
    *,
    scope: str | None = None,
    force: bool = False,
    min_cluster_size: int | None = None,
    recent_limit: int = DEFAULT_RECENT_LIMIT,
    guard: Callable[[], AbstractContextManager[Any]] | None = None,
) -> ConsolidationResult:
    """Cluster similar active traces, promote stable clusters, and run decay.

    Leaving ``min_cluster_size`` unset means the adaptive promotion floor of
    the module docstring, sized from the active traces this pass loaded — the
    scope's own trace count when ``scope`` is given, the whole corpus when it
    is not.

    ``guard`` lets a caller that shares ``store`` across threads run the pass
    beside other work: every step that touches the store runs inside
    ``guard()``, one short step at a time (one cluster, one encode), and the
    clustering itself — pure computation on the snapshot the pass loaded —
    runs outside it. Without a guard the pass runs exactly as one block.
    """

    if min_cluster_size is not None and min_cluster_size < 1:
        raise ValueError("min_cluster_size must be positive")
    step = guard if guard is not None else nullcontext

    with step():
        _refresh_promoted_global_concepts(store)
        trace_limit = max(recent_limit, min_cluster_size or 0) if force else recent_limit
        traces = store.list_nodes(
            level="trace",
            scope=scope,
            include_decayed=False,
            limit=trace_limit,
        )
        merge_floor = resolve_min_cluster_size(min_cluster_size, trace_count=len(traces))
        phase = store.detect_phase()
        embedder = (
            LocalEmbeddingModel(model_name=store.config.embedding_model)
            if force or phase.number >= 2
            else None
        )
        if embedder is not None:
            # Hand the loaded encoder to the store. Every concept written below goes
            # through create_node/update_node with an embedding, and those chunk the
            # content — without this the store would load a second copy of the same
            # model to do it.
            model = embedder
            store.set_chunk_embedder(lambda texts: [model.embed(text) for text in texts])
    embeddings = _trace_embeddings(store, traces, embedder, step)
    clusters = _cluster_traces(traces, embedder=embedder, embeddings=embeddings)
    result = ConsolidationResult(
        clusters_considered=len(clusters),
        traces_considered=len(traces),
    )

    for schema, created in _materialize_procedural_schemas(
        store, traces, min_cluster_size=PROCEDURAL_MIN_CLUSTER_SIZE, step=step
    ):
        if created:
            result.schemas_created.append(schema)
        else:
            result.schemas_updated.append(schema)

    for cluster in clusters:
        if len(cluster.traces) < merge_floor:
            continue
        with step():
            materialized = _materialize_cluster(store, cluster, min_cluster_size=merge_floor)
        for concept, created in materialized:
            with step():
                result.concepts_promoted.extend(
                    _cross_scope_promotion(store, concept, phase_number=phase.number)
                )
            if created:
                result.concepts_created.append(concept)
            else:
                result.concepts_updated.append(concept)

    with step():
        decay_result: DecayResult = apply_decay(store, scope=scope)
    result.decayed.extend(decay_result.nodes)
    return result


consolidate = memory_consolidate


@dataclass(slots=True)
class _PreparedDigest:
    sources: tuple[tuple[str, str], ...]
    best_id: str
    content: str
    cluster_key: str

    def matches(self, members: Sequence[Node], best: Node) -> bool:
        return self.best_id == best.id and self.sources == tuple(
            (member.id, member.content) for member in members
        )


@dataclass(slots=True)
class _TeachPreparation:
    corrective_id: str
    source_inputs: dict[str, _DigestSource]
    digests: dict[str, _PreparedDigest]


def prepare_teach(
    store: MemoryStore,
    trace_id: str,
    correction: str | Mapping[str, Any],
    *,
    confidence: float | None = None,
    guard: Callable[[], AbstractContextManager[Any]] = nullcontext,
) -> _TeachPreparation:
    """Prepare expensive text work on call-local copies, without a transaction.

    Only reads use the shared-store guard. The atomic writer resolves the graph
    again and accepts each prepared digest only for its actual content inputs
    and best source. Changed descendants and concurrent writes use the normal
    writer; this snapshot never supplies stats, provenance or edge weights.
    """

    with guard():
        original = store.get_node(trace_id)
        if original is None:
            raise KeyError(trace_id)
        content = _correction_text(correction)
        if not content:
            raise ValueError("correction must be non-empty")
        corrective = Node(
            id=new_ulid(), level="trace", content=content,
            confidence=min(0.5, 0.5 if confidence is None else float(confidence)),
            usefulness_score=max(1.0, original.usefulness_score + 1.0),
        )
        owners = [
            node for node in _affected_derivatives(store, trace_id).values()
            if node.materializer in ("cluster", "cross-scope-promotion")
        ]
        observation = _SourceObservation(store, (
            source for node in owners for source in _derivation_sources(store, node)
        ))
        snapshots = []
        for node in owners:
            members = [
                corrective if member.id == trace_id else member
                for member in _current_derivation(store, node, observation=observation).live
            ]
            if members:
                snapshots.append((node, members))

    source_inputs = {
        source.node.id: source
        for source in _prepare_digest_sources(list({
            member.id: member for _, members in snapshots for member in members
        }.values()))
    }
    digests = {}
    for node, members in snapshots:
        prepared = [source_inputs[member.id] for member in members]
        best = (
            max(members, key=lambda member: (_trace_quality(member), member.id))
            if node.materializer == "cluster" else _highest_confidence_source(members)
        )
        digests[node.id] = _prepare_digest(node.scope, members, best, prepared)
    return _TeachPreparation(corrective.id, source_inputs, digests)


def memory_teach(
    store: MemoryStore,
    trace_id: str,
    correction: str | Mapping[str, Any],
    *,
    confidence: float | None = None,
    context: Mapping[str, Any] | None = None,
    preparation: _TeachPreparation | None = None,
) -> TeachResult:
    """Atomically correct a source and refresh its recorded derivatives."""

    with store.transaction():
        original = store.get_node(trace_id)
        if original is None:
            raise KeyError(trace_id)

        correction_text = _correction_text(correction)
        if not correction_text:
            raise ValueError("correction must be non-empty")

        # The correction replaces the step text, not its place.
        context_data = {key: value for key, value in original.context.items() if key not in _STEP_TEXT_KEYS}
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
        corrective_trace = store.append_trace(
            correction_text, context_data, feedback=feedback,
            node_id=preparation.corrective_id if preparation is not None else None,
        )

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
        if preparation is None:
            _refresh_derived_from(store, original.id)
        else:
            _refresh_derived_from(store, original.id, preparation=preparation)
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


def _derivation_sources(store: MemoryStore | _SourceObservation, node: Node) -> list[str]:
    """Read recorded materializer inputs, never incidental recall lineage."""

    if node.materializer:
        return node.materializer_sources
    return [
        edge.target_id
        for edge in store.list_connections(source_id=node.id, relation_type="related")
        if edge.metadata.get("kind") == DERIVED_FROM_KIND
    ]


class _SourceObservation:
    """Share current reads only within one guarded phase of derivation work.

    Sources being rewritten are read on demand until their source-first writer
    supplies the resulting node. Supersedes edges do not change during refresh;
    their endpoint validity is always checked against the current node values.
    Related inputs are shared until a writer updates their source node.
    """

    def __init__(
        self, store: MemoryStore, source_ids: Iterable[str], *, changing: Iterable[str] = (),
    ) -> None:
        self.store = store
        self.changing = set(changing)
        ids = set(source_ids) - self.changing
        self.nodes: dict[str, Node | None] = dict.fromkeys(ids)
        self.nodes.update(store.get_nodes(ids))
        self.edges: dict[tuple[str | None, str | None, ConnectionType | None], list[Connection]] = {}

    def get_node(self, node_id: str) -> Node | None:
        if node_id in self.changing:
            return self.store.get_node(node_id)
        if node_id not in self.nodes:
            self.nodes[node_id] = self.store.get_node(node_id)
        return self.nodes[node_id]

    def get_nodes(self, source_ids: Iterable[str]) -> dict[str, Node]:
        return {
            source: node
            for batch in _node_read_batches(source_ids) for source in batch
            if (node := self.get_node(source)) is not None
        }

    def list_connections(
        self, *, source_id: str | None = None, target_id: str | None = None,
        relation_type: ConnectionType | None = None,
    ) -> list[Connection]:
        key = (source_id, target_id, relation_type)
        if key not in self.edges:
            self.edges[key] = self.store.list_connections(
                source_id=source_id, target_id=target_id, relation_type=relation_type,
            )
        return self.edges[key]

    def updated(self, node: Node) -> None:
        self.changing.discard(node.id)
        self.nodes[node.id] = node
        # Writers connect the owner to its inputs, co-accessed input pairs,
        # and promotion inputs to their global concept.
        for source_id in (node.id, *node.materializer_sources):
            self.edges.pop((source_id, None, "related"), None)


def _current_derivation(
    store: MemoryStore, node: Node, *, additional: Iterable[Node] = (),
    observation: _SourceObservation | None = None,
) -> _EraAssessment:
    """Resolve each recorded input independently, without displacing other eras."""

    return _current_sources(
        store, [*_derivation_sources(observation or store, node), *(member.id for member in additional)],
        observation=observation,
    )


def _current_sources(
    store: MemoryStore, source_ids: Iterable[str], *, observation: _SourceObservation | None = None,
) -> _EraAssessment:
    """Resolve the same source identities for lookup, refresh and writing."""

    reader = observation or store
    members = list(reader.get_nodes(source_ids).values())
    live: dict[str, Node] = {}
    excluded: dict[str, str] = {}
    for member in members:
        replacement_id = resolve_replacement(reader, member.id)
        current = reader.get_node(replacement_id) if replacement_id else member
        if replacement_id:
            excluded[member.id] = f"superseded-by:{replacement_id}"
        if current is not None and not current.decayed and not current.corrections:
            live[current.id] = current
        else:
            excluded.setdefault(member.id, "no current source")
    # Superseding a recorded derivative explicitly rejects its cited inputs.
    # This does not reject independently current sources merely because they
    # have the same age or appear in the same clustering pass.
    displaced: dict[str, str] = {}
    for member in live.values():
        for edge in reader.list_connections(source_id=member.id, relation_type="supersedes"):
            if edge.metadata.get("kind") == _DUPLICATE_CONTENT_KIND:
                continue
            target = reader.get_node(edge.target_id)
            if target is not None:
                for source in set(_derivation_sources(reader, target)) & live.keys() - {member.id}:
                    displaced[source] = f"displaced-via:{target.id}"
    for source, reason in displaced.items():
        del live[source]
        excluded[source] = reason
    return _EraAssessment(members, list(live.values()), excluded, False, 1)


def _affected_derivatives(store: MemoryStore, source_id: str) -> dict[str, Node]:
    """Discover recorded consumers, including earlier source identities."""
    affected: dict[str, Node] = {}
    pending = [source_id]
    visited: set[str] = set()
    while pending:
        changed = pending.pop()
        if changed in visited:
            continue
        visited.add(changed)
        # A materializer may still cite an earlier version of this source.
        pending.extend(
            edge.target_id
            for edge in store.list_connections(source_id=changed, relation_type="supersedes")
        )
        for node in store.nodes_referencing_source(changed):
            if node.id == source_id or node.id in affected:
                continue
            if changed not in _derivation_sources(store, node):
                continue
            if node.corrections or _is_superseded(store, node.id):
                continue
            affected[node.id] = node
            pending.append(node.id)
    return affected


def _refresh_derived_from(
    store: MemoryStore, source_id: str, *, preparation: _TeachPreparation | None = None,
) -> None:
    """Refresh the current derivation subgraph, sources before their consumers.

    Discover it before changing membership so diamonds and cross-scope
    descendants see all of their updated inputs in one atomic teach.
    """

    affected = _affected_derivatives(store, source_id)

    observation = _SourceObservation(store, (
        source for node in affected.values() for source in _derivation_sources(store, node)
    ), changing=affected)
    source_inputs = dict(preparation.source_inputs) if preparation is not None else {}

    visiting: set[str] = set()
    refreshed: set[str] = set()

    def refresh(node_id: str) -> None:
        if node_id in refreshed:
            return
        if node_id in visiting:
            # A cycle has no source-first evaluation. Keep its evidence by id,
            # but it cannot remain current after one of its inputs changed.
            observation.updated(store.soft_delete_node(node_id, "derivation: cyclic sources"))
            return
        visiting.add(node_id)
        for dependency in _derivation_sources(observation, affected[node_id]):
            if dependency in affected:
                refresh(dependency)
        node = observation.get_node(node_id)
        if node is not None and not node.decayed:
            observation.updated(_refresh_derived_node(
                store, node, source_inputs=source_inputs,
                prepared_digest=preparation.digests.get(node.id) if preparation else None,
                observation=observation,
            ))
        visiting.remove(node_id)
        refreshed.add(node_id)

    for node_id in affected:
        refresh(node_id)


def _refresh_derived_node(
    store: MemoryStore, node: Node,
    *, source_inputs: dict[str, _DigestSource] | None = None,
    prepared_digest: _PreparedDigest | None = None,
    observation: _SourceObservation | None = None,
) -> Node:
    assessment = _current_derivation(store, node, observation=observation)
    members = assessment.live
    if not members:
        return store.soft_delete_node(node.id, "derivation: no current sources")

    materializer = node.materializer
    if materializer == "procedural":
        # Keep the recorded group and its retrieval label. A correction is an
        # explicit replacement even when its writer changes its context.
        key = _procedure_key(node)
        if key is not None:
            key = replace(key, trigger=str(node.context.get("trigger") or key.trigger))
            declared = [member for member in members if _declares_step(member)]
            binding = node.level == "schema" and len(declared) == len(members)
            current = replace(node, source_traces=[member.id for member in members])
            updated, _ = _create_or_update_schema(
                store, node.scope, key, members, current,
                era=assessment if binding else None,
                observation=observation,
            )
            _connect_schema_to_traces(store, updated, members)
            return updated
    elif materializer == "cluster":
        return _write_cluster_concept(
            store, node.scope, assessment, existing=node,
            strategy=str(node.provenance.get("strategy") or "token-jaccard"),
            source_inputs=source_inputs,
            prepared_digest=prepared_digest,
        )
    elif materializer == "cross-scope-promotion":
        return _write_global_promotion(
            store, members, existing=node, prepared_digest=prepared_digest,
            source_inputs=source_inputs,
        )

    # An explicit derived_from edge proves dependency, but cannot specify how
    # to rewrite an arbitrary expression. Its sources and old content remain
    # available through exact lookup with the normal decay validity.
    return store.soft_delete_node(node.id, "derivation: corrected source requires rederivation")


def _materialize_procedural_schemas(
    store: MemoryStore,
    traces: Iterable[Node],
    *,
    min_cluster_size: int,
    step: Callable[[], AbstractContextManager[Any]] = nullcontext,
) -> list[tuple[Node, bool]]:
    """Group procedural traces by their stable group key; one node per group.

    Each trace's stable group key is its `task_pattern` if present, otherwise
    its `procedure_id`. Traces sharing a `task_pattern` cluster together even
    when their `procedure_id` labels differ between instances.

    Only writer-declared steps make a procedure schema. A group without them gets
    an evidence carrier instead: a concept that the group's trigger finds and
    that holds its current records whole as evidence, never as steps. A schema
    the replaced mechanism built from such a group, visited or not, becomes
    that carrier in place.
    """

    # A node belongs to its scope and recorded group key, never to a label:
    # groups share labels, and matching on one let a group take over another's
    # node (recall's dedup reads the title instead). The oldest node of
    # a group is its node, a later one a duplicate. A taught or superseded node
    # belongs to a dead era; updating it would resurrect corrected content.
    nodes: dict[tuple[str, str], list[Node]] = {}
    triggered = store.get_nodes([node_id for node_id, _trigger in store.schema_triggers(None)])
    for node in sorted(triggered.values(), key=lambda node: (node.created_at, node.id)):
        dead = node.corrections or _is_superseded(store, node.id)
        if node.provenance.get("strategy") == "procedural" and not dead:
            nodes.setdefault((node.scope, _schema_group_id(node)), []).append(node)
    # A legacy node is repaired from its own records even when no visited trace
    # reaches its group; a record it cites from another group joins that group.
    legacy = {group: owned[0] for group, owned in nodes.items() if owned[0].level == "schema"}
    cited = store.get_nodes([source for node in legacy.values() for source in node.source_traces])
    groups: dict[tuple[str, str], dict[str, Node]] = {group: {} for group in legacy}
    for trace in [*traces, *cited.values()]:
        key = _procedure_key(trace)
        if key and not trace.decayed:
            groups.setdefault((trace.scope, key.group_id), {})[trace.id] = trace
    schemas: list[tuple[Node, bool]] = []
    for (group_scope, group_id), by_id in groups.items():
        existing, *duplicates = nodes.get((group_scope, group_id)) or [None]
        if len(by_id) < min_cluster_size and existing is None:
            continue
        with step(), store.transaction():
            # The pass can yield between loading its snapshot and writing this
            # group. A committed correction and the group's current sources
            # are authoritative at the write boundary.
            existing = store.get_node(existing.id) if existing else None
            if existing and (existing.decayed or existing.corrections or _is_superseded(store, existing.id)):
                continue
            if existing:
                assessment = _current_derivation(store, existing, additional=by_id.values())
                group_traces = assessment.live
            else:
                group_traces = list(store.get_nodes(by_id).values())
                assessment = _assess_cluster_eras(store, group_traces)
            declared = [trace for trace in group_traces if _declares_step(trace)]
            binding = len(declared) >= min_cluster_size or bool(
                existing and existing.level == "schema" and declared and len(declared) == len(group_traces)
            )
            if binding:
                assessment = (
                    replace(assessment, live=declared)
                    if existing else _assess_cluster_eras(store, declared)
                )
            if not assessment.live:
                # Every member is superseded or era-displaced: no current era to
                # distill, so nothing may be emitted for this group.
                if existing is not None and existing.level == "schema":
                    store.soft_delete_node(existing.id, "procedural: no current records")
                continue
            label = existing.context.get("trigger") if existing else None
            procedure_key = _select_group_procedure_key(assessment.live, label)
            # A schema speaks for the current era; evidence keeps every record
            # nothing superseded.
            members, era = (assessment.live, assessment) if binding else (group_traces, None)
            schema, created = _create_or_update_schema(
                store, group_scope, procedure_key, members, existing, era=era
            )
            _connect_schema_to_traces(store, schema, assessment.live)
            for duplicate in duplicates:
                store.soft_delete_node(duplicate.id, "procedural: duplicate group")
        schemas.append((schema, created))
    return schemas


def _select_group_procedure_key(traces: list[Node], label: str | None = None) -> _ProcedureKey:
    """Pick the best trigger label for a group of traces sharing one group_id."""

    keys = [key for trace in traces if (key := _procedure_key(trace))]
    procedure_id_keys = [key for key in keys if key.field == "procedure_id"]
    pool = procedure_id_keys or keys
    counts = Counter((key.field, key.raw, key.trigger) for key in pool)
    # A node keeps the trigger it is found by while a current record carries it.
    kept = [entry for entry in counts if entry[2] == label]
    trigger_field, trigger_raw, trigger = kept[0] if kept else counts.most_common(1)[0][0]
    representative = keys[0]
    return _ProcedureKey(
        field=trigger_field,
        raw=trigger_raw,
        trigger=trigger,
        group_field=representative.group_field,
        group_raw=representative.group_raw,
        group_id=representative.group_id,
    )


def _procedure_key(trace: Node) -> _ProcedureKey | None:
    """Extract group + trigger info from a trace's context.

    Grouping prefers `task_pattern` (stable across procedure_id variations);
    the trigger prefers `procedure_id` (human-readable for query matching).
    """

    context = trace.context or {}
    procedure_id_raw = str(context.get("procedure_id") or "").strip()
    task_pattern_raw = str(context.get("task_pattern") or "").strip()

    if not procedure_id_raw and not task_pattern_raw:
        return None

    if task_pattern_raw:
        group_field = "task_pattern"
        group_raw = task_pattern_raw
    else:
        group_field = "procedure_id"
        group_raw = procedure_id_raw
    group_id = _normalize_trigger(group_raw)
    if not group_id:
        return None

    if procedure_id_raw:
        trigger_field = "procedure_id"
        trigger_raw = procedure_id_raw
    else:
        trigger_field = group_field
        trigger_raw = group_raw
    trigger = _normalize_trigger(trigger_raw) or group_id
    if trigger == group_id:
        trigger_field = group_field
        trigger_raw = group_raw

    return _ProcedureKey(
        field=trigger_field,
        raw=trigger_raw,
        trigger=trigger,
        group_field=group_field,
        group_raw=group_raw,
        group_id=group_id,
    )


def _create_or_update_schema(
    store: MemoryStore,
    scope: str,
    procedure_key: _ProcedureKey,
    traces: list[Node],
    existing: Node | None,
    *,
    era: _EraAssessment | None = None,
    observation: _SourceObservation | None = None,
) -> tuple[Node, bool]:
    trigger = procedure_key.trigger
    group_id = procedure_key.group_id
    steps = _procedure_steps(traces)

    unique_agents = _unique_agent_count(traces)
    confidence = _consensus_confidence(traces, unique_agents)
    temporal_hint = detect_temporal_hint(traces)
    usefulness = sum(max(0.0, trace.usefulness_score) for trace in traces) / len(traces)

    era_block = era.era_provenance() if era is not None and era.filtered else None
    excluded_ids = set(era.excluded) if era is not None else set()

    base_provenance = {
        "procedure_key": group_id,
        "procedure_field": procedure_key.field,
        "procedure_pattern": procedure_key.raw,
        "group_field": procedure_key.group_field,
        "group_pattern": procedure_key.group_raw,
        "strategy": "procedural",
        "consolidated_at": _utc_now(),
    }
    if era_block is not None:
        base_provenance["era"] = era_block
    stats = {
        "confidence": confidence,
        "unique_agents": unique_agents,
        "temporal_hint": temporal_hint,
        "usefulness_score": usefulness,
    }
    context = {
        "scope": scope,
        "agent": "memory_consolidate",
        "timestamp": _utc_now(),
        "procedure_key": group_id,
        procedure_key.group_field: procedure_key.group_raw,
        procedure_key.field: procedure_key.raw,
        "trigger": trigger,
    }
    if era is not None:  # a recipe describes its era; an evidence carrier has none
        context["procedure"] = steps
    if era_block is not None:
        context["era_status"] = "current"

    cited = existing.source_traces if existing else []
    sources = sorted({*cited, *(trace.id for trace in traces)} - excluded_ids)
    content = (
        _format_schema_content(trigger, steps)
        if era is not None
        else _format_evidence_content(store, trigger, (observation or store).get_nodes(sources).values())
    )
    provenance = {
        **(existing.provenance if existing else {}),
        **base_provenance,
        "source_traces": sources,
        "cluster_size": len(sources),
    }
    level = "schema" if era is not None else "concept"
    if existing is None:
        schema = store.create_node(
            level=level,
            content=content,
            context=context,
            stats=stats,
            provenance=provenance,
        )
        return schema, True
    schema = store.update_node(
        existing.id,
        content=content,
        context=context,
        stats=stats,
        provenance=provenance,
        level=level,
    )
    return schema, False


def _schema_group_id(schema: Node) -> str:
    group = schema.context.get("procedure_key") or schema.provenance.get("procedure_key")
    if group is not None and str(group).strip():
        return _normalize_trigger(str(group))
    # A schema written before procedure_key was recorded carries its group
    # the way its source traces did.
    key = _procedure_key(schema)
    return key.group_id if key else ""


def _normalize_trigger(procedure_id: str) -> str:
    cleaned = procedure_id.replace("_", " ").replace("-", " ").replace("/", " ")
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _procedure_steps(traces: Iterable[Node]) -> list[str]:
    """One step per declared position, restated by its latest record, then
    the steps declared by text alone, in write order."""

    written = sorted(
        traces, key=lambda trace: (trace.timestamp or "", trace.created_at or "", trace.id)
    )
    slots: dict[tuple[int, int], str] = {}
    for index, trace in enumerate(written):
        step = _step_description(trace)
        if not step:
            continue
        position = _step_order(trace)
        slots[(0, position) if position is not None else (1, index)] = step
    return list(dict.fromkeys(slots[slot] for slot in sorted(slots)))


def _declares_step(trace: Node) -> bool:
    context = trace.context or {}
    return _step_order(trace) is not None or any(context.get(key) for key in _STEP_TEXT_KEYS)


def _step_order(trace: Node) -> int | None:
    context = trace.context or {}
    for key in ("step_order", "step"):
        value = context.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
    return None


def _step_description(trace: Node) -> str:
    context = trace.context or {}
    for key in _STEP_TEXT_KEYS:
        value = context.get(key)
        if value:
            return str(value).strip()
    return (trace.content or "").strip()


def _format_schema_content(trigger: str, steps: Iterable[str]) -> str:
    lines = [f"Procedure: {trigger}" if trigger else "Procedure"]
    for index, step in enumerate(steps, start=1):
        lines.append(f"{index}. {step}")
    return "\n".join(lines)


def _format_evidence_content(store: MemoryStore, trigger: str, records: Iterable[Node]) -> str:
    lines = [f"Evidence: {trigger} (case records, not steps)"]
    for record in sorted(records, key=lambda node: (node.timestamp or "", node.id), reverse=True):
        if not record.decayed and not _is_superseded(store, record.id):
            lines.append(f"- {record.id}: {record.content.strip()}")
    return "\n".join(lines)


def _connect_schema_to_traces(
    store: MemoryStore, schema: Node, traces: Iterable[Node]
) -> None:
    procedure_pattern = (
        schema.context.get("procedure_id")
        or schema.context.get("task_pattern")
        or schema.context.get("procedure_key")
    )
    for trace in traces:
        _upsert_weighted_connection(
            store,
            schema.id,
            trace.id,
            "related",
            weight=0.6,
            metadata={
                "basis": "procedural",
                "procedure_key": schema.context.get("procedure_key"),
                "procedure_id": procedure_pattern,
                # R5a: schema→source-trace edges are provenance ground truth
                # (typed-edge-rules.md §R5a). The metadata merge in
                # _upsert_weighted_connection annotates existing edges on the
                # next pass without adding rows; the kind drives asymmetric
                # traversal factors in retrieval._traversal. Concept edges
                # (cluster/co_access bases) stay unannotated by design.
                **DERIVED_FROM_ANNOTATION,
            },
        )


def _trace_embeddings(
    store: MemoryStore,
    traces: Sequence[Node],
    embedder: LocalEmbeddingModel | None,
    step: Callable[[], AbstractContextManager[Any]],
) -> dict[str, list[float]]:
    """Whole-content vector per trace, encoding only what the cache lacks.

    Traces are append-only, so a trace encoded by an earlier pass keeps its
    vector: a pass pays the model for the traces written since, not for the
    whole scope. Each encode batch runs inside ``step`` — the encoder is
    shared with the recall path — and the fresh vectors are cached for the
    next pass, unless the encoder changed under the pass (a model that fell
    back to the hashed encoder mid-pass must not poison the cache).
    """

    if embedder is None:
        return {}
    vectors = {trace.id: trace.embedding for trace in traces if trace.embedding is not None}
    missing = [trace for trace in traces if trace.embedding is None]
    if not missing:
        return vectors
    # Resolving the encoder may load the model: seconds, and no store access,
    # so not under the guard (a server has usually warmed it already).
    encoder = embedder.encoder_id()
    for start in range(0, len(missing), _CACHE_READ_BATCH):
        with step():
            vectors.update(
                store.get_consolidation_embeddings(
                    missing[start : start + _CACHE_READ_BATCH], encoder
                )
            )
    to_encode = [trace for trace in missing if trace.id not in vectors]
    fresh: list[tuple[Node, list[float]]] = []
    for start in range(0, len(to_encode), _ENCODE_BATCH):
        with step():
            for trace in to_encode[start : start + _ENCODE_BATCH]:
                vector = embedder.embed(trace.content)
                vectors[trace.id] = vector
                fresh.append((trace, vector))
    if fresh:
        with step():
            if embedder.encoder_id() == encoder:
                try:
                    store.put_consolidation_embeddings(fresh, encoder)
                except Exception:
                    # A cache write is an optimization; the pass has its vectors.
                    _LOG.warning("could not cache consolidation embeddings", exc_info=True)
    return vectors


def _cluster_traces(
    traces: Iterable[Node],
    *,
    embedder: LocalEmbeddingModel | None = None,
    embeddings: Mapping[str, list[float]] | None = None,
) -> list[_TraceCluster]:
    """Greedy single-pass clustering in trace order.

    Each trace joins the first same-scope cluster with the highest
    ``_cluster_similarity`` if that score clears its strategy's threshold,
    else starts a cluster. ``_cluster_traces_indexed`` computes exactly this
    with array arithmetic and a token index; the loop below is the reference
    it is checked against and the fallback without numpy or with ragged
    vectors.
    """

    prepared: list[tuple[Node, set[str], list[float] | None]] = []
    for trace in traces:
        tokens = _significant_tokens(trace.content)
        if embeddings is not None and trace.id in embeddings:
            embedding = embeddings[trace.id]
        else:
            embedding = _trace_embedding(trace, embedder)
        if not tokens and embedding is None:
            continue
        prepared.append((trace, tokens, embedding))
    if _np is not None:
        dimensions = {len(embedding) for _, _, embedding in prepared if embedding is not None}
        if len(dimensions) <= 1:
            return _cluster_traces_indexed(prepared, next(iter(dimensions), 0))
    return _cluster_prepared_reference(prepared)


def _cluster_prepared_reference(
    prepared: Iterable[tuple[Node, set[str], list[float] | None]],
) -> list[_TraceCluster]:
    clusters: list[_TraceCluster] = []
    for trace, tokens, embedding in prepared:

        best_cluster: _TraceCluster | None = None
        best_score = 0.0
        best_strategy = "token-jaccard"
        for cluster in clusters:
            if _cluster_group(cluster.traces[0]) != _cluster_group(trace):
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
                    embeddings_by_trace_id={trace.id: list(embedding)}
                    if embedding is not None
                    else {},
                    strategy="embedding-cosine" if embedding is not None else "token-jaccard",
                )
            )
    return clusters


class _ScopeIndex:
    """Per-scope clustering state for ``_cluster_traces_indexed``."""

    __slots__ = ("clusters", "sums", "reps", "rep_norms", "has_rep", "rep_tokens", "postings")

    def __init__(self, capacity: int, dimensions: int) -> None:
        self.clusters: list[_TraceCluster] = []
        self.sums = _np.zeros((capacity, dimensions), dtype=_np.float64)
        self.reps = _np.zeros((capacity, dimensions), dtype=_np.float64)
        self.rep_norms = _np.zeros(capacity, dtype=_np.float64)
        self.has_rep = _np.zeros(capacity, dtype=bool)
        self.rep_tokens: list[set[str]] = []
        self.postings: dict[str, set[int]] = {}

    def set_rep_tokens(self, index: int, tokens: set[str]) -> None:
        old = self.rep_tokens[index]
        for token in old - tokens:
            self.postings[token].discard(index)
        for token in tokens - old:
            self.postings.setdefault(token, set()).add(index)
        self.rep_tokens[index] = tokens

    def set_rep(self, index: int, count: int) -> None:
        rep = self.sums[index] / count
        self.reps[index] = rep
        self.rep_norms[index] = float(rep @ rep)
        self.has_rep[index] = True


def _cluster_traces_indexed(
    prepared: Sequence[tuple[Node, set[str], list[float] | None]],
    dimensions: int,
) -> list[_TraceCluster]:
    """``_cluster_prepared_reference`` without the per-cluster Python loop.

    The reference scores every trace against every same-scope cluster, one
    384-wide pure-Python cosine and one representative-token rebuild per pair:
    O(traces x clusters) interpreted work, minutes on a 6600-trace scope. Here
    the cosines of one trace against all clusters are one matrix-vector
    product, Jaccard is computed only for clusters sharing a token (every
    other cluster scores 0 on it, as ``_jaccard`` would), and each cluster's
    representative vector and token set are rebuilt only when it grows.
    Selection keeps the reference's rule: the first cluster holding the
    maximum score, only if that score is positive.
    """

    clusters: list[_TraceCluster] = []
    group_sizes = Counter(_cluster_group(trace) for trace, _, _ in prepared)
    width = max(1, dimensions)
    scopes: dict[tuple[str, str | None], _ScopeIndex] = {}
    for trace, tokens, embedding in prepared:
        group = _cluster_group(trace)
        state = scopes.get(group)
        if state is None:
            state = scopes[group] = _ScopeIndex(group_sizes[group], width)
        size = len(state.clusters)
        best = -1
        if size:
            cosine = _np.zeros(size, dtype=_np.float64)
            if embedding is not None:
                vector = _np.asarray(embedding, dtype=_np.float64)
                norm = float(vector @ vector)
                if norm > 0.0:
                    dots = state.reps[:size] @ vector
                    products = state.rep_norms[:size] * norm
                    valid = state.has_rep[:size] & (products > 0.0)
                    _np.divide(dots, _np.sqrt(products), out=cosine, where=valid)
            jaccard = _np.zeros(size, dtype=_np.float64)
            if tokens:
                shared: Counter[int] = Counter()
                for token in tokens:
                    posting = state.postings.get(token)
                    if posting:
                        shared.update(posting)
                for index, inter in shared.items():
                    union = len(tokens) + len(state.rep_tokens[index]) - inter
                    jaccard[index] = inter / union
            scores = _np.maximum(cosine, jaccard)
            candidate = int(_np.argmax(scores))
            if scores[candidate] > 0.0:
                best = candidate
        if best >= 0:
            if cosine[best] >= jaccard[best]:
                strategy, score = "embedding-cosine", float(cosine[best])
            else:
                strategy, score = "token-jaccard", float(jaccard[best])
            if _accept_cluster_match(score, strategy):
                cluster = state.clusters[best]
                cluster.strategy = strategy
                cluster.traces.append(trace)
                cluster.token_counts.update(tokens)
                if embedding is not None:
                    cluster.embeddings_by_trace_id[trace.id] = list(embedding)
                    state.sums[best] += vector
                    cluster.embedding_count += 1
                    state.set_rep(best, cluster.embedding_count)
                state.set_rep_tokens(best, cluster.representative_tokens)
                continue
        cluster = _TraceCluster(
            trace.scope,
            [trace],
            Counter(tokens),
            embedding_count=1 if embedding is not None else 0,
            embeddings_by_trace_id={trace.id: list(embedding)} if embedding is not None else {},
            strategy="embedding-cosine" if embedding is not None else "token-jaccard",
        )
        index = len(state.clusters)
        state.clusters.append(cluster)
        state.rep_tokens.append(set())
        if embedding is not None:
            state.sums[index] = _np.asarray(embedding, dtype=_np.float64)
            state.set_rep(index, 1)
        state.set_rep_tokens(index, cluster.representative_tokens)
        clusters.append(cluster)
    for state in scopes.values():
        for index, cluster in enumerate(state.clusters):
            if cluster.embedding_count:
                cluster.embedding_sum = state.sums[index].tolist()
    return clusters


def _cluster_group(node: Node) -> tuple[str, str | None]:
    key = _procedure_key(node)
    return node.scope, key.group_id if key else None


def _source_ancestors(store: MemoryStore, members: Sequence[Node]) -> dict[str, set[str]]:
    """Map recorded input versions to the current members replacing them."""

    ancestors: dict[str, set[str]] = {}
    for member in members:
        pending = [member.id]
        seen: set[str] = set()
        while pending:
            source_id = pending.pop()
            if source_id in seen:
                continue
            seen.add(source_id)
            ancestors.setdefault(source_id, set()).add(member.id)
            pending.extend(
                edge.target_id for edge in store.list_connections(
                    source_id=source_id, relation_type="supersedes",
                )
            )
    return ancestors


def _partition_sources(
    owners: Sequence[Node], members: Sequence[Node], ancestors: Mapping[str, set[str]],
) -> tuple[dict[str, set[str]], set[str]]:
    """Partition by recorded inputs, keeping ambiguous owners independent."""

    assigned: dict[str, set[str]] = {}
    unowned = {member.id for member in members}
    for owner in owners:
        owned = set().union(*(ancestors.get(source, set()) for source in owner.materializer_sources))
        if owned:
            assigned[owner.id] = owned
            unowned.difference_update(owned)
    return assigned, unowned


def _materialize_cluster(
    store: MemoryStore, cluster: _TraceCluster, *, min_cluster_size: int,
) -> list[tuple[Node, bool]]:
    """Continue recorded owners and materialize only the unowned remainder.

    Clustering admits related new evidence. Recorded inputs decide which node
    owns it: overlapping owners retain their separate derivations, and a new
    member is never broadcast across them. Neither a lexical label nor a lossy
    digest identifies the sources of another derivation.
    """

    with store.transaction():
        current = _current_sources(store, (member.id for member in cluster.traces))
        members = {member.id: member for member in current.live}
        if not members:
            return []
        owners = [
            node for node in store.list_nodes(level="concept", scope=cluster.scope, limit=100_000)
            if node.materializer == "cluster" and node.source_traces
            and not node.corrections and not _is_superseded(store, node.id)
        ]
        ancestors = _source_ancestors(store, current.live)
        assigned, unowned = _partition_sources(owners, current.live, ancestors)

        fresh = _assess_cluster_eras(store, cluster.traces)
        admitted = {member.id for member in fresh.live}
        admitted.update(
            member.id for member in current.live
            if any(
                source_id in ancestors and member.id in ancestors[source_id]
                and (reason == "corrected" or reason.startswith("superseded-by:"))
                for source_id, reason in fresh.excluded.items()
            )
        )
        unowned.intersection_update(admitted)
        if len(assigned) == 1:
            owner = next(owner for owner in owners if owner.id in assigned)
            compatible = {source for source in unowned if _cluster_group(members[source]) == _cluster_group(owner)}
            assigned[owner.id].update(compatible)
            unowned.difference_update(compatible)

        result: list[tuple[Node, bool]] = []
        for owner in owners:
            if owner.id not in assigned:
                continue
            assessment = _current_derivation(
                store, owner, additional=(members[source] for source in sorted(assigned[owner.id])),
            )
            result.append((_write_cluster_concept(
                store, cluster.scope, assessment, existing=owner, strategy=cluster.strategy,
                embeddings=cluster.embeddings_by_trace_id,
            ), False))
        groups: dict[tuple[str, str | None], list[Node]] = {}
        for source in sorted(unowned):
            member = members[source]
            groups.setdefault(_cluster_group(member), []).append(member)
        for (scope, _group), new_members in groups.items():
            # The admission floor applies before era filtering. A correction
            # may leave a smaller current derivation; splitting into different
            # current groups or away from other owners needs its own support.
            if len(new_members) < min_cluster_size and (
                assigned or len(groups) > 1 or len(cluster.traces) < min_cluster_size
            ):
                continue
            assessment = replace(fresh, live=new_members)
            result.append((_write_cluster_concept(
                store, scope, assessment, strategy=cluster.strategy,
                embeddings=cluster.embeddings_by_trace_id,
            ), True))
        return result


def _write_cluster_concept(
    store: MemoryStore, scope: str, assessment: _EraAssessment,
    *, existing: Node | None = None, strategy: str,
    embeddings: Mapping[str, list[float]] | None = None,
    source_inputs: dict[str, _DigestSource] | None = None,
    prepared_digest: _PreparedDigest | None = None,
) -> Node:
    """Write one owner's digest, retrieval label, inputs and edges atomically."""

    members = assessment.live
    best_trace = max(members, key=lambda trace: (_trace_quality(trace), trace.id))
    if prepared_digest is None or not prepared_digest.matches(members, best_trace):
        prepared = _digest_sources(members, source_inputs)
        prepared_digest = _prepare_digest(scope, members, best_trace, prepared)
    digest = prepared_digest.content
    cluster_key = prepared_digest.cluster_key
    source_traces = sorted({trace.id for trace in members})
    unique_agents = _unique_agent_count(members)
    concept_embedding = best_trace.embedding or (embeddings or {}).get(best_trace.id)
    provenance = {
        **(existing.provenance if existing else {}),
        "source_traces": source_traces,
        "cluster_key": cluster_key,
        "cluster_size": len(source_traces),
        "consolidated_at": _utc_now(),
        "strategy": strategy,
    }
    if assessment.filtered:
        provenance["era"] = assessment.era_provenance()
    stats = {
        "confidence": _consensus_confidence(members, unique_agents),
        "unique_agents": unique_agents,
        "temporal_hint": detect_temporal_hint(members),
        "usefulness_score": sum(max(0.0, trace.usefulness_score) for trace in members) / len(members),
    }
    with store.transaction():
        if existing is None:
            context = {"scope": scope, "agent": "memory_consolidate", "timestamp": _utc_now()}
            key = _procedure_key(members[0])
            if key is not None:
                context[key.group_field] = key.group_raw
            concept = store.create_node(
                level="concept", content=digest, context=context,
                embedding=concept_embedding, stats=stats, provenance=provenance,
            )
        else:
            concept = store.update_node(
                existing.id, content=digest, embedding=concept_embedding,
                stats=stats, provenance=provenance,
            )
        _update_edge_weights_from_co_access(store, concept, members)
    return concept


@dataclass(slots=True)
class _DigestSource:
    node: Node
    tokens: set[str]
    sentences: list[tuple[str, set[str]]]


def _digest_sources(
    members: Sequence[Node], source_inputs: dict[str, _DigestSource] | None,
) -> list[_DigestSource]:
    inputs = source_inputs if source_inputs is not None else {}
    missing = [
        member for member in members
        if member.id not in inputs or inputs[member.id].node.content != member.content
    ]
    if missing:
        inputs.update((source.node.id, source) for source in _prepare_digest_sources(missing))
    return [inputs[member.id] for member in members]


def _prepare_digest(
    scope: str, members: Sequence[Node], best: Node, sources: Sequence[_DigestSource],
) -> _PreparedDigest:
    tokens: Counter[str] = Counter()
    for source in sources:
        tokens.update(source.tokens)
    return _PreparedDigest(
        sources=tuple((member.id, member.content) for member in members),
        best_id=best.id,
        content=_synthesize_digest(members, best=best, prepared_sources=sources),
        cluster_key=_cluster_key(_TraceCluster(scope, list(members), tokens)),
    )


def _prepare_digest_sources(sources: Sequence[Node]) -> list[_DigestSource]:
    """Extract whole-source and sentence signals together for one materialization."""

    prepared = []
    for source in sources:
        tokens: set[str] = set()
        sentences = []
        for sentence in _split_sentences(source.content):
            sentence_tokens: set[str] = set()
            for raw in _raw_tokens(sentence):
                token = _significant_token(raw)
                if token is not None:
                    tokens.add(token)
                    sentence_tokens.add(token)
            sentences.append((sentence, {token for token in sentence_tokens if token}))
        # Keep the whole-source insertion order used by _significant_tokens:
        # Counter's equal-count ordering also determines the retrieval label.
        prepared.append(_DigestSource(source, {token for token in tokens if token}, sentences))
    return prepared


def _synthesize_digest(
    sources: Sequence[Node],
    *,
    best: Node,
    max_chars: int = DIGEST_MAX_CHARS,
    prepared_sources: Sequence[_DigestSource] | None = None,
) -> str:
    """Build a deterministic extractive digest of the sources' contents.

    Leads with the best source's opening sentences, then folds in the most
    distinctive sentence of each remaining source in cluster-centrality order,
    skipping near-duplicate sentences. Single-source and all-identical-content
    inputs keep the sole content — that already is the digest. For inputs with
    two or more distinct contents the result is byte-distinct from every
    source content and strictly shorter than their concatenation, except for
    degenerate corners (formatting-only variants with no novel token, or
    sources too tiny to fit any distinctive line) where the strongest single
    content is kept.
    """

    distinct_contents = {source.content for source in sources}
    if len(distinct_contents) <= 1:
        return best.content

    budget = min(max_chars, sum(len(content) for content in distinct_contents) - 1)
    lead = _digest_lead(best.content)
    if len(lead) + len(DIGEST_FACT_PREFIX) + 2 > budget:
        return best.content

    prepared = prepared_sources if prepared_sources is not None else _prepare_digest_sources(sources)
    token_counts: Counter[str] = Counter()
    for source in prepared:
        token_counts.update(source.tokens)

    covered_significant = _significant_tokens(lead)
    covered_raw = set(_raw_tokens(lead))
    included_sentences = [frozenset(_raw_tokens(part)) for part in _split_sentences(lead)]
    ordered_others = sorted(
        (source for source in prepared if source.node.id != best.id),
        key=lambda source: (-_token_centrality(source.tokens, token_counts), source.node.id),
    )

    lines = [lead]
    length = len(lead)
    for source in ordered_others:
        sentence = _most_distinctive_sentence(
            source.sentences, covered_significant, covered_raw, included_sentences
        )
        if sentence is None:
            continue
        line = DIGEST_FACT_PREFIX + sentence
        if length + 1 + len(line) > budget:
            continue
        lines.append(line)
        length += 1 + len(line)
        covered_significant |= _significant_tokens(sentence)
        raw = frozenset(_raw_tokens(sentence))
        covered_raw |= raw
        included_sentences.append(raw)

    if len(lines) == 1:
        # Every non-lead sentence was redundant (near-identical cluster). A
        # novel-token fragment keeps the digest byte-distinct from each source
        # and covering more than one of them without echoing whole variants.
        fragment = _novel_fragment_line(
            [source.node for source in ordered_others], covered_raw, budget - length - 1,
        )
        if fragment is None:
            return best.content
        lines.append(fragment)

    digest = "\n".join(lines)
    if digest in distinct_contents:
        # A source already contains these exact bytes (e.g. a re-remembered
        # digest); byte-distinctness is unattainable without fabricating text.
        return best.content
    return digest


def _digest_lead(content: str) -> str:
    text = content.strip()
    if len(text) <= DIGEST_LEAD_MAX_CHARS:
        return text
    sentences = _split_sentences(text)
    lead = sentences[0]
    if len(lead) > DIGEST_LEAD_MAX_CHARS:
        return lead[: DIGEST_LEAD_MAX_CHARS - 1].rstrip() + "…"
    for sentence in sentences[1:]:
        if len(lead) + 1 + len(sentence) > DIGEST_LEAD_MAX_CHARS:
            break
        lead = f"{lead} {sentence}"
    return lead


def _most_distinctive_sentence(
    sentences: Sequence[tuple[str, set[str]]],
    covered_significant: set[str],
    covered_raw: set[str],
    included_sentences: list[frozenset[str]],
) -> str | None:
    best_sentence: str | None = None
    best_rank: tuple[int, int, int] | None = None
    for index, (sentence, tokens) in enumerate(sentences):
        significant_novelty = len(tokens - covered_significant)
        if significant_novelty < 1:
            continue
        raw = frozenset(_raw_tokens(sentence))
        if any(
            _jaccard(raw, included) >= DIGEST_NEAR_DUPLICATE_JACCARD
            for included in included_sentences
        ):
            continue
        rank = (-significant_novelty, -len(raw - covered_raw), index)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_sentence = sentence
    if best_sentence is not None and len(best_sentence) > DIGEST_FACT_MAX_CHARS:
        best_sentence = best_sentence[: DIGEST_FACT_MAX_CHARS - 1].rstrip() + "…"
    return best_sentence


def _novel_fragment_line(
    ordered_others: Sequence[Node],
    covered_raw: set[str],
    budget_left: int,
) -> str | None:
    for source in ordered_others:
        novel = _novel_tokens_in_order(source.content, covered_raw)
        if not novel:
            continue
        line = DIGEST_FACT_PREFIX + " ".join(novel)
        while novel and len(line) > budget_left:
            novel.pop()
            line = DIGEST_FACT_PREFIX + " ".join(novel)
        if novel:
            return line
    return None


def _novel_tokens_in_order(content: str, covered_raw: set[str]) -> list[str]:
    preferred: list[str] = []
    fallback: list[str] = []
    for token in _raw_tokens(content):
        if token in covered_raw:
            continue
        bucket = (
            preferred
            if len(token) >= 3 and not token.isdigit() and token not in _STOP_WORDS
            else fallback
        )
        if len(bucket) < DIGEST_FRAGMENT_MAX_TOKENS and token not in bucket:
            bucket.append(token)
        if len(preferred) >= DIGEST_FRAGMENT_MAX_TOKENS:
            break
    return preferred or fallback


def _token_centrality(tokens: set[str], token_counts: Counter[str]) -> float:
    if not tokens:
        return 0.0
    return sum(token_counts[token] for token in tokens) / len(tokens)


def _split_sentences(content: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;])\s+|\n+", content.strip())
    return [part.strip() for part in parts if part and part.strip()]


def _raw_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower().replace("ё", "е"))


def _cross_scope_promotion(
    store: MemoryStore, concept: Node, *, phase_number: int
) -> list[Node]:
    if phase_number < CROSS_SCOPE_PROMOTION_PHASE:
        return []
    concept = store.get_node(concept.id)
    if concept is None or concept.decayed:
        return []
    if concept.level != "concept" or not concept.scope.startswith("project:"):
        return []
    if concept.embedding is None or concept.corrections or _is_superseded(store, concept.id):
        return []

    matches = store.find_similar_by_embedding(
        concept.embedding, level="concept", exclude_scope=concept.scope,
        scope_prefix="project:", threshold=CROSS_SCOPE_PROMOTION_THRESHOLD, limit=100_000,
    )
    group = _cluster_group(concept)[1]
    sources_by_scope: dict[str, tuple[Node, float]] = {concept.scope: (concept, 1.0)}
    for candidate, similarity in matches:
        if candidate.id == concept.id or candidate.scope == concept.scope:
            continue
        if candidate.corrections or _is_superseded(store, candidate.id) or _cluster_group(candidate)[1] != group:
            continue
        current = sources_by_scope.get(candidate.scope)
        if current is None or (similarity, candidate.confidence) > (current[1], current[0].confidence):
            sources_by_scope[candidate.scope] = (candidate, similarity)
    if len(sources_by_scope) < 3:
        return []

    with store.transaction():
        current = _current_sources(store, (source.id for source, _ in sources_by_scope.values())).live
        members = {source.id: source for source in current
                   if source.level == "concept" and source.scope.startswith("project:")
                   and _cluster_group(source)[1] == group}
        if len(_source_scopes(members.values())) < 3:
            return []
        similarities = {source.id: similarity for source, similarity in sources_by_scope.values()}
        owners = [
            node for node in store.list_nodes(level="concept", scope="global", limit=100_000)
            if node.materializer == "cross-scope-promotion"
            and not node.corrections and not _is_superseded(store, node.id)
        ]
        sources = list(members.values())
        assigned, unowned = _partition_sources(owners, sources, _source_ancestors(store, sources))
        if len(assigned) == 1:
            owner = next(owner for owner in owners if owner.id in assigned)
            if _cluster_group(owner)[1] == group:
                assigned[owner.id].update(unowned)
                unowned.clear()
        result = []
        for owner in owners:
            if owner.id not in assigned:
                continue
            current = _current_derivation(
                store, owner, additional=(members[source] for source in sorted(assigned[owner.id])),
            ).live
            updated = _write_global_promotion(store, current, existing=owner, similarities=similarities)
            if updated != owner:
                result.append(updated)
        remaining = [members[source] for source in sorted(unowned)]
        if len(_source_scopes(remaining)) >= 3:
            result.append(_write_global_promotion(store, remaining, similarities=similarities))
        return result


def _write_global_promotion(
    store: MemoryStore,
    project_sources: list[Node],
    *,
    existing: Node | None = None,
    similarities: Mapping[str, float] | None = None,
    prepared_digest: _PreparedDigest | None = None,
    source_inputs: dict[str, _DigestSource] | None = None,
) -> Node:
    """One writer for a promotion's content, current inputs and validity stats."""

    best_source = _highest_confidence_source(project_sources)
    digest = (
        prepared_digest.content
        if prepared_digest is not None and prepared_digest.matches(project_sources, best_source)
        else _synthesize_digest(
            project_sources, best=best_source,
            prepared_sources=_digest_sources(project_sources, source_inputs),
        )
    )
    source_scopes = _source_scopes(project_sources)
    source_concepts = _source_concept_ids(project_sources)
    promoted_at = _utc_now()
    provenance = dict(existing.provenance) if existing else {}
    provenance.update({
        "source_traces": _source_trace_ids(project_sources),
        "promoted_from": source_scopes,
        "source_concepts": source_concepts,
        "active_promoted_from": source_scopes,
        "active_source_concepts": source_concepts,
        "strategy": "cross-scope-promotion",
    })
    provenance.setdefault("promoted_at", promoted_at)
    stats = {
        "confidence": _active_promotion_confidence(project_sources),
        "unique_agents": _promoted_unique_agents(project_sources),
        "temporal_hint": _promoted_temporal_hint(project_sources),
        "usefulness_score": _promoted_usefulness(project_sources),
    }
    if existing is not None and existing.content == digest and all(
        existing.provenance.get(key) == provenance[key]
        for key in ("source_concepts", "active_source_concepts", "active_promoted_from")
    ) and existing.source_traces == provenance["source_traces"] and all(
        getattr(existing, key) == value for key, value in stats.items()
    ):
        return existing
    if similarities is not None and (
        existing is None or set(existing.materializer_sources) != set(source_concepts)
    ):
        events = list(provenance.get("promotion_events", []))
        events.append(_promotion_event(project_sources, similarities=similarities, promoted_at=promoted_at))
        provenance["promotion_events"] = events
        provenance["last_promoted_at"] = promoted_at
    with store.transaction():
        if existing is None:
            context = {"scope": "global", "agent": "memory_consolidate", "timestamp": promoted_at}
            key = _procedure_key(best_source)
            if key is not None:
                context[key.group_field] = key.group_raw
            global_concept = store.create_node(
                level="concept", content=digest,
                context=context,
                embedding=best_source.embedding, stats=stats, provenance=provenance,
            )
        else:
            global_concept = store.update_node(
                existing.id, content=digest, stats=stats, provenance=provenance,
            )
        _connect_project_concepts_to_global(
            store, project_sources, global_concept, similarities=similarities or {},
        )
    return global_concept


def _connect_project_concepts_to_global(
    store: MemoryStore,
    project_sources: Iterable[Node],
    global_concept: Node,
    *,
    similarities: Mapping[str, float],
) -> None:
    for source in project_sources:
        _upsert_weighted_connection(
            store,
            source.id,
            global_concept.id,
            "related",
            weight=max(0.1, similarities.get(source.id, CROSS_SCOPE_PROMOTION_THRESHOLD)),
            metadata={
                "basis": "cross_scope_promotion",
                "source_scope": source.scope,
                "target_scope": "global",
                "similarity": round(similarities.get(source.id, 0.0), 6),
            },
        )


def _highest_confidence_source(project_sources: Iterable[Node]) -> Node:
    return max(
        project_sources,
        key=lambda source: (
            source.confidence,
            source.unique_agents,
            source.usefulness_score,
            _node_quality(source),
            source.id,
        ),
    )


def _source_scopes(project_sources: Iterable[Node]) -> list[str]:
    return sorted({source.scope for source in project_sources})


def _source_concept_ids(project_sources: Iterable[Node]) -> list[str]:
    return sorted({source.id for source in project_sources})


def _source_trace_ids(project_sources: Iterable[Node]) -> list[str]:
    return sorted({trace_id for source in project_sources for trace_id in source.source_traces})


def _promotion_event(
    project_sources: Iterable[Node],
    *,
    similarities: Mapping[str, float],
    promoted_at: str,
) -> dict[str, Any]:
    sources = list(project_sources)
    return {
        "promoted_at": promoted_at,
        "source_scopes": _source_scopes(sources),
        "source_concepts": _source_concept_ids(sources),
        "similarities": {
            source.id: round(similarities.get(source.id, 0.0), 6) for source in sources
        },
    }


def _promoted_confidence(project_sources: Iterable[Node]) -> float:
    sources = list(project_sources)
    if not sources:
        return 0.0
    base = max(source.confidence for source in sources)
    scope_bonus = min(0.12, 0.02 * max(0, len(_source_scopes(sources)) - 1))
    return round(min(0.99, base + scope_bonus), 6)



def _promoted_unique_agents(project_sources: Iterable[Node]) -> int:
    return sum(max(1, source.unique_agents) for source in project_sources)


def _promoted_usefulness(project_sources: Iterable[Node]) -> float:
    sources = list(project_sources)
    if not sources:
        return 0.0
    return round(sum(source.usefulness_score for source in sources) / len(sources), 6)


def _promoted_temporal_hint(project_sources: Iterable[Node]) -> str | None:
    hints = [source.temporal_hint for source in project_sources if source.temporal_hint]
    if not hints:
        return None
    return Counter(hints).most_common(1)[0][0]


def _refresh_promoted_global_concepts(store: MemoryStore) -> list[Node]:
    return [
        _refresh_derived_node(store, node)
        for node in store.list_nodes(level="concept", scope="global", include_decayed=False, limit=100_000)
        if node.provenance.get("strategy") == "cross-scope-promotion"
        and not node.corrections and not _is_superseded(store, node.id)
    ]


def _active_promotion_confidence(project_sources: Iterable[Node]) -> float:
    sources = list(project_sources)
    if not sources:
        return 0.5
    if len(_source_scopes(sources)) < 3:
        return round(max(source.confidence for source in sources), 6)
    return _promoted_confidence(sources)


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
    for raw in _raw_tokens(content):
        token = _significant_token(raw)
        if token is not None:
            tokens.add(token)
    return {token for token in tokens if token}


def _significant_token(raw: str) -> str | None:
    token = raw.strip("_")
    if len(token) < 3 or token.isdigit() or token in _STOP_WORDS:
        return None
    return _normalize_token(token)


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
