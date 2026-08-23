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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import log1p
from typing import Any, Collection, Iterable, Mapping, Sequence
import os
import re

from living_memory.decay import DecayResult, apply_decay, memory_forget
from living_memory.edge_derivation import DERIVED_FROM_ANNOTATION
from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity
from living_memory.feedback import ImplicitRecallFeedback, apply_pending_recall_feedback
from living_memory.models import Connection, Node, string_list
from living_memory.storage import _DUPLICATE_CONTENT_KIND, MemoryStore
from living_memory.temporal import detect_temporal_hint, parse_timestamp, split_time_regimes

DEFAULT_MIN_CLUSTER_SIZE = 100
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
GLOBAL_PROMOTION_DEDUP_THRESHOLD = 0.8
PROCEDURAL_MIN_CLUSTER_SIZE = 3
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

    member_list = list(members)
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
                        displaced = (set(target.source_traces) & member_ids) - {edge.source_id}
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
    ) -> ConsolidationResult:
        return memory_consolidate(
            self.store,
            scope=scope,
            force=force,
            min_cluster_size=(
                self.min_cluster_size if min_cluster_size is None else min_cluster_size
            ),
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
    min_cluster_size: int | None = None,
    recent_limit: int = DEFAULT_RECENT_LIMIT,
) -> ConsolidationResult:
    """Cluster similar active traces, promote stable clusters, and run decay.

    Leaving ``min_cluster_size`` unset means the adaptive promotion floor of
    the module docstring, sized from the active traces this pass loaded — the
    scope's own trace count when ``scope`` is given, the whole corpus when it
    is not.
    """

    if min_cluster_size is not None and min_cluster_size < 1:
        raise ValueError("min_cluster_size must be positive")

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
    clusters = _cluster_traces(traces, embedder=embedder)
    result = ConsolidationResult(
        clusters_considered=len(clusters),
        traces_considered=len(traces),
    )

    for schema, created in _materialize_procedural_schemas(
        store, traces, min_cluster_size=PROCEDURAL_MIN_CLUSTER_SIZE
    ):
        if created:
            result.schemas_created.append(schema)
        else:
            result.schemas_updated.append(schema)

    for cluster in clusters:
        if len(cluster.traces) < merge_floor:
            continue
        assessment = _assess_cluster_eras(store, cluster.traces)
        if not assessment.live:
            # Every member is superseded or era-displaced: there is no current
            # era to speak for, so nothing may be promoted from this cluster.
            continue
        concept, created = _merge_cluster_into_concept(store, cluster, assessment)
        _update_edge_weights_from_co_access(store, concept, assessment.live)
        promoted = _cross_scope_promotion(store, concept, phase_number=phase.number)
        if promoted is not None:
            result.concepts_promoted.append(promoted)
        if created:
            result.concepts_created.append(concept)
        else:
            result.concepts_updated.append(concept)

    decay_result: DecayResult = apply_decay(store, scope=scope)
    result.decayed.extend(decay_result.nodes)
    return result


consolidate = memory_consolidate


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


def _materialize_procedural_schemas(
    store: MemoryStore,
    traces: Iterable[Node],
    *,
    min_cluster_size: int,
) -> list[tuple[Node, bool]]:
    """Group procedural traces by their stable group key and emit schema nodes.

    Each trace's stable group key is its `task_pattern` if present, otherwise
    its `procedure_id`. Traces sharing a `task_pattern` cluster together even
    when their `procedure_id` labels differ between instances.
    """

    groups: dict[tuple[str, str], list[Node]] = {}
    keys_by_group: dict[tuple[str, str], list[_ProcedureKey]] = {}
    for trace in traces:
        key = _procedure_key(trace)
        if not key:
            continue
        group_tuple = (trace.scope, key.group_id)
        groups.setdefault(group_tuple, []).append(trace)
        keys_by_group.setdefault(group_tuple, []).append(key)

    schemas: list[tuple[Node, bool]] = []
    for (scope, _group_id), group_traces in groups.items():
        if len(group_traces) < min_cluster_size:
            continue
        assessment = _assess_cluster_eras(store, group_traces)
        if not assessment.live:
            # Every member is superseded or era-displaced: no current era to
            # distill, so no schema may be emitted for this group.
            continue
        live_ids = {member.id for member in assessment.live}
        live_keys = [
            key
            for trace, key in zip(group_traces, keys_by_group[(scope, _group_id)])
            if trace.id in live_ids
        ]
        procedure_key = _select_group_procedure_key(
            live_keys or keys_by_group[(scope, _group_id)]
        )
        schema, created = _create_or_update_schema(
            store, scope, procedure_key, assessment.live, era=assessment
        )
        _connect_schema_to_traces(store, schema, assessment.live)
        schemas.append((schema, created))
    return schemas


def _select_group_procedure_key(keys: list[_ProcedureKey]) -> _ProcedureKey:
    """Pick the best trigger label for a group of traces sharing one group_id."""

    procedure_id_keys = [key for key in keys if key.field == "procedure_id"]
    pool = procedure_id_keys or keys
    counts = Counter((key.field, key.raw, key.trigger) for key in pool)
    (trigger_field, trigger_raw, trigger), _frequency = counts.most_common(1)[0]
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
    *,
    era: _EraAssessment | None = None,
) -> tuple[Node, bool]:
    trigger = procedure_key.trigger
    group_id = procedure_key.group_id
    steps = _procedure_steps(traces)
    content = _format_schema_content(trigger, steps)

    sorted_trace_ids = sorted({trace.id for trace in traces})
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
        "procedure": steps,
    }
    if era_block is not None:
        context["era_status"] = "current"

    existing = _find_existing_schema(store, scope, procedure_key)
    if existing is None:
        provenance = {
            **base_provenance,
            "source_traces": sorted_trace_ids,
            "cluster_size": len(sorted_trace_ids),
        }
        schema = store.create_node(
            level="schema",
            content=content,
            context=context,
            stats=stats,
            provenance=provenance,
        )
        return schema, True

    merged_sources = sorted({*existing.source_traces, *sorted_trace_ids} - excluded_ids)
    provenance = {
        **existing.provenance,
        **base_provenance,
        "source_traces": merged_sources,
        "cluster_size": len(merged_sources),
    }
    schema = store.update_node(
        existing.id,
        content=content,
        context=context,
        stats=stats,
        provenance=provenance,
    )
    return schema, False


def _find_existing_schema(
    store: MemoryStore, scope: str, procedure_key: _ProcedureKey
) -> Node | None:
    """Locate a schema for the same group key, tolerating legacy provenance shape."""

    schemas = store.list_nodes(
        level="schema",
        scope=scope,
        include_decayed=False,
        limit=100_000,
    )
    for schema in schemas:
        candidates = (
            schema.context.get("procedure_key"),
            schema.context.get("task_pattern"),
            schema.context.get("procedure_id"),
            schema.context.get("trigger"),
            schema.provenance.get("procedure_key"),
            schema.provenance.get("group_pattern"),
            schema.provenance.get("procedure_pattern"),
            schema.provenance.get("procedure_id"),
        )
        normalized = {
            _normalize_trigger(str(value))
            for value in candidates
            if value is not None and str(value).strip()
        }
        if procedure_key.group_id not in normalized:
            continue
        if schema.corrections or _is_superseded(store, schema.id):
            # A taught/superseded schema belongs to a dead era; updating it
            # would resurrect corrected content. Emit a fresh one instead.
            continue
        return schema
    return None


def _normalize_trigger(procedure_id: str) -> str:
    cleaned = procedure_id.replace("_", " ").replace("-", " ").replace("/", " ")
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _procedure_steps(traces: Iterable[Node]) -> list[str]:
    ordered = sorted(
        traces,
        key=lambda trace: (
            _step_order(trace),
            trace.timestamp or "",
            trace.created_at or "",
            trace.id,
        ),
    )
    steps: list[str] = []
    seen: set[str] = set()
    for trace in ordered:
        step = _step_description(trace)
        if not step or step in seen:
            continue
        seen.add(step)
        steps.append(step)
    return steps


def _step_order(trace: Node) -> int:
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
    return 10_000_000


def _step_description(trace: Node) -> str:
    context = trace.context or {}
    for key in ("step_description", "step_content"):
        value = context.get(key)
        if value:
            return str(value).strip()
    task = trace.task or context.get("task")
    content = (trace.content or "").strip()
    if task and content and str(task).strip() not in content:
        return f"{str(task).strip()}: {content}"
    return content


def _format_schema_content(trigger: str, steps: Iterable[str]) -> str:
    lines = [f"Procedure: {trigger}" if trigger else "Procedure"]
    for index, step in enumerate(steps, start=1):
        lines.append(f"{index}. {step}")
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
                    embeddings_by_trace_id={trace.id: list(embedding)}
                    if embedding is not None
                    else {},
                    strategy="embedding-cosine" if embedding is not None else "token-jaccard",
                )
            )
    return clusters


def _merge_cluster_into_concept(
    store: MemoryStore, cluster: _TraceCluster, assessment: _EraAssessment
) -> tuple[Node, bool]:
    members = assessment.live
    source_traces = sorted({trace.id for trace in members})
    cluster_key = _cluster_key(cluster)
    unique_agents = _unique_agent_count(members)
    confidence = _consensus_confidence(members, unique_agents)
    temporal_hint = detect_temporal_hint(members)
    best_trace = max(members, key=lambda trace: (_trace_quality(trace), trace.id))
    concept_embedding = cluster.embedding_for(best_trace) or cluster.representative_embedding
    digest = _synthesize_digest(members, best=best_trace)
    usefulness = sum(max(0.0, trace.usefulness_score) for trace in members) / len(members)

    provenance = {
        "source_traces": source_traces,
        "cluster_key": cluster_key,
        "cluster_size": len(source_traces),
        "consolidated_at": _utc_now(),
        "strategy": cluster.strategy,
    }
    if assessment.filtered:
        provenance["era"] = assessment.era_provenance()
    stats = {
        "confidence": confidence,
        "unique_agents": unique_agents,
        "temporal_hint": temporal_hint,
        "usefulness_score": usefulness,
    }

    existing = _find_existing_concept(
        store,
        cluster.scope,
        cluster_key,
        candidate_contents={digest, *(trace.content for trace in members)},
    )
    if existing is None:
        concept = store.create_node(
            level="concept",
            content=digest,
            context={
                "scope": cluster.scope,
                "agent": "memory_consolidate",
                "timestamp": _utc_now(),
            },
            embedding=concept_embedding,
            stats=stats,
            provenance=provenance,
        )
        return concept, True

    merged_sources = sorted(
        {*existing.source_traces, *source_traces} - set(assessment.excluded)
    )
    provenance["source_traces"] = merged_sources
    provenance["cluster_size"] = len(merged_sources)
    content_update = (
        digest
        if digest != existing.content
        and _trace_quality(best_trace) >= _node_quality(existing)
        else None
    )
    concept = store.update_node(
        existing.id,
        content=content_update,
        embedding=concept_embedding,
        stats=stats,
        provenance={**existing.provenance, **provenance},
    )
    return concept, False


def _synthesize_digest(
    sources: Sequence[Node],
    *,
    best: Node,
    max_chars: int = DIGEST_MAX_CHARS,
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

    token_counts: Counter[str] = Counter()
    for source in sources:
        token_counts.update(_significant_tokens(source.content))

    covered_significant = _significant_tokens(lead)
    covered_raw = set(_raw_tokens(lead))
    included_sentences = [frozenset(_raw_tokens(part)) for part in _split_sentences(lead)]
    ordered_others = sorted(
        (source for source in sources if source.id != best.id),
        key=lambda source: (-_token_centrality(source, token_counts), source.id),
    )

    lines = [lead]
    length = len(lead)
    for source in ordered_others:
        sentence = _most_distinctive_sentence(
            source.content, covered_significant, covered_raw, included_sentences
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
        fragment = _novel_fragment_line(ordered_others, covered_raw, budget - length - 1)
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
    content: str,
    covered_significant: set[str],
    covered_raw: set[str],
    included_sentences: list[frozenset[str]],
) -> str | None:
    best_sentence: str | None = None
    best_rank: tuple[int, int, int] | None = None
    for index, sentence in enumerate(_split_sentences(content)):
        significant_novelty = len(_significant_tokens(sentence) - covered_significant)
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


def _token_centrality(node: Node, token_counts: Counter[str]) -> float:
    tokens = _significant_tokens(node.content)
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
) -> Node | None:
    if phase_number < CROSS_SCOPE_PROMOTION_PHASE:
        return None
    if concept.level != "concept" or not concept.scope.startswith("project:"):
        return None
    if concept.embedding is None or concept.corrections:
        return None

    matches = store.find_similar_by_embedding(
        concept.embedding,
        level="concept",
        exclude_scope=concept.scope,
        scope_prefix="project:",
        threshold=CROSS_SCOPE_PROMOTION_THRESHOLD,
        limit=100_000,
    )
    sources_by_scope: dict[str, tuple[Node, float]] = {concept.scope: (concept, 1.0)}
    for candidate, similarity in matches:
        if candidate.id == concept.id or candidate.scope == concept.scope:
            continue
        if candidate.corrections:
            # Corrected project concepts must not feed the global digest.
            continue
        current = sources_by_scope.get(candidate.scope)
        if current is None or (similarity, candidate.confidence) > (
            current[1],
            current[0].confidence,
        ):
            sources_by_scope[candidate.scope] = (candidate, similarity)

    if len(sources_by_scope) < 3:
        return None

    project_sources = [source for source, _similarity in sources_by_scope.values()]
    similarities = {source.id: similarity for source, similarity in sources_by_scope.values()}
    global_matches = store.find_similar_by_embedding(
        concept.embedding,
        level="concept",
        scope="global",
        threshold=GLOBAL_PROMOTION_DEDUP_THRESHOLD,
        limit=10,
    )
    if global_matches:
        return _reinforce_global_concept(
            store,
            global_matches[0][0],
            project_sources,
            similarities=similarities,
        )
    return _create_global_promoted_concept(
        store,
        project_sources,
        similarities=similarities,
    )


def _create_global_promoted_concept(
    store: MemoryStore,
    project_sources: list[Node],
    *,
    similarities: Mapping[str, float],
) -> Node:
    best_source = _highest_confidence_source(project_sources)
    digest = _synthesize_digest(project_sources, best=best_source)
    source_scopes = _source_scopes(project_sources)
    source_concepts = _source_concept_ids(project_sources)
    source_traces = _source_trace_ids(project_sources)
    promoted_at = _utc_now()
    provenance = {
        "source_traces": source_traces,
        "promoted_from": source_scopes,
        "source_concepts": source_concepts,
        "promoted_at": promoted_at,
        "strategy": "cross-scope-promotion",
        "promotion_events": [
            _promotion_event(project_sources, similarities=similarities, promoted_at=promoted_at)
        ],
    }
    global_concept = store.create_node(
        level="concept",
        content=digest,
        context={
            "scope": "global",
            "agent": "memory_consolidate",
            "timestamp": promoted_at,
        },
        embedding=best_source.embedding,
        stats={
            "confidence": _promoted_confidence(project_sources),
            "unique_agents": _promoted_unique_agents(project_sources),
            "temporal_hint": _promoted_temporal_hint(project_sources),
            "usefulness_score": _promoted_usefulness(project_sources),
        },
        provenance=provenance,
    )
    _connect_project_concepts_to_global(
        store,
        project_sources,
        global_concept,
        similarities=similarities,
    )
    return global_concept


def _reinforce_global_concept(
    store: MemoryStore,
    global_concept: Node,
    project_sources: list[Node],
    *,
    similarities: Mapping[str, float],
) -> Node | None:
    source_scopes = set(_source_scopes(project_sources))
    source_concepts = set(_source_concept_ids(project_sources))
    existing_scopes = set(string_list(global_concept.provenance.get("promoted_from")))
    existing_concepts = set(string_list(global_concept.provenance.get("source_concepts")))
    source_traces = set(_source_trace_ids(project_sources))
    existing_traces = set(global_concept.source_traces)
    new_scopes = source_scopes - existing_scopes
    new_concepts = source_concepts - existing_concepts
    new_traces = source_traces - existing_traces

    _connect_project_concepts_to_global(
        store,
        project_sources,
        global_concept,
        similarities=similarities,
    )
    if not new_scopes and not new_concepts and not new_traces:
        return None

    promoted_at = _utc_now()
    provenance = dict(global_concept.provenance)
    provenance["source_traces"] = sorted(existing_traces | source_traces)
    provenance["promoted_from"] = sorted(existing_scopes | source_scopes)
    provenance["source_concepts"] = sorted(existing_concepts | source_concepts)
    provenance.setdefault("promoted_at", promoted_at)
    provenance["last_promoted_at"] = promoted_at
    provenance["strategy"] = "cross-scope-promotion"
    events = list(provenance.get("promotion_events", []))
    events.append(_promotion_event(project_sources, similarities=similarities, promoted_at=promoted_at))
    provenance["promotion_events"] = events

    stats = {
        "confidence": _reinforced_confidence(global_concept, project_sources, len(new_scopes)),
        "unique_agents": max(global_concept.unique_agents, _promoted_unique_agents(project_sources)),
        "temporal_hint": global_concept.temporal_hint or _promoted_temporal_hint(project_sources),
        "usefulness_score": max(global_concept.usefulness_score, _promoted_usefulness(project_sources)),
    }
    return store.update_node(global_concept.id, stats=stats, provenance=provenance)


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


def _reinforced_confidence(
    global_concept: Node, project_sources: Iterable[Node], new_scope_count: int
) -> float:
    if new_scope_count <= 0:
        return global_concept.confidence
    source_confidence = _promoted_confidence(project_sources)
    boost = min(0.1, 0.02 * new_scope_count)
    return round(min(0.99, max(global_concept.confidence, source_confidence) + boost), 6)


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
    refreshed: list[Node] = []
    global_concepts = store.list_nodes(
        level="concept",
        scope="global",
        include_decayed=False,
        limit=100_000,
    )
    for global_concept in global_concepts:
        source_concept_ids = string_list(global_concept.provenance.get("source_concepts"))
        if not source_concept_ids:
            continue

        active_sources = [
            source
            for source_id in source_concept_ids
            if (source := store.get_node(source_id)) is not None
            and source.level == "concept"
            and source.scope.startswith("project:")
            and not source.decayed
        ]
        active_scopes = _source_scopes(active_sources)
        active_concepts = _source_concept_ids(active_sources)
        if (
            string_list(global_concept.provenance.get("active_promoted_from")) == active_scopes
            and string_list(global_concept.provenance.get("active_source_concepts"))
            == active_concepts
        ):
            continue

        provenance = dict(global_concept.provenance)
        provenance["active_promoted_from"] = active_scopes
        provenance["active_source_concepts"] = active_concepts
        provenance["last_promotion_refresh_at"] = _utc_now()
        stats = {
            "confidence": min(
                global_concept.confidence,
                _active_promotion_confidence(active_sources),
            ),
            "unique_agents": _promoted_unique_agents(active_sources),
            "temporal_hint": _promoted_temporal_hint(active_sources),
            "usefulness_score": _promoted_usefulness(active_sources),
        }
        refreshed.append(store.update_node(global_concept.id, stats=stats, provenance=provenance))
    return refreshed


def _active_promotion_confidence(project_sources: Iterable[Node]) -> float:
    sources = list(project_sources)
    if not sources:
        return 0.5
    if len(_source_scopes(sources)) < 3:
        return round(max(source.confidence for source in sources), 6)
    return _promoted_confidence(sources)


def _find_existing_concept(
    store: MemoryStore,
    scope: str,
    cluster_key: str,
    *,
    candidate_contents: Collection[str] = (),
) -> Node | None:
    concepts = store.list_nodes(
        level="concept",
        scope=scope,
        include_decayed=False,
        limit=100_000,
    )
    contents = frozenset(candidate_contents)
    content_twin: Node | None = None
    for concept in concepts:
        if concept.corrections:
            # A taught concept holds a corrected belief; merging fresh cluster
            # content into it would resurrect it. Let a new concept form.
            continue
        if concept.provenance.get("cluster_key") == cluster_key:
            return concept
        if content_twin is None and concept.content in contents:
            content_twin = concept
    # cluster_key is the primary identity; the content twin covers key drift
    # between runs for both digest concepts (an unchanged cluster re-digests to
    # the same bytes) and legacy concepts that copied a source trace verbatim
    # before digests existed. Merge into the twin instead of duplicating it.
    return content_twin


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
