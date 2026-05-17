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
from living_memory.models import Connection, Node, string_list
from living_memory.storage import MemoryStore
from living_memory.temporal import detect_temporal_hint

DEFAULT_MIN_CLUSTER_SIZE = 100
DEFAULT_RECENT_LIMIT = 10_000
JACCARD_SIMILARITY_THRESHOLD = 0.58
EMBEDDING_SIMILARITY_THRESHOLD = 0.65
CROSS_SCOPE_PROMOTION_PHASE = 4
CROSS_SCOPE_PROMOTION_THRESHOLD = 0.7
GLOBAL_PROMOTION_DEDUP_THRESHOLD = 0.8
PROCEDURAL_MIN_CLUSTER_SIZE = 3

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
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    recent_limit: int = DEFAULT_RECENT_LIMIT,
) -> ConsolidationResult:
    """Cluster similar active traces, promote stable clusters, and run decay."""

    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be positive")

    _refresh_promoted_global_concepts(store)
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

    for schema, created in _materialize_procedural_schemas(
        store, traces, min_cluster_size=PROCEDURAL_MIN_CLUSTER_SIZE
    ):
        if created:
            result.schemas_created.append(schema)
        else:
            result.schemas_updated.append(schema)

    for cluster in clusters:
        if len(cluster.traces) < min_cluster_size:
            continue
        concept, created = _merge_cluster_into_concept(store, cluster)
        _update_edge_weights_from_co_access(store, concept, cluster.traces)
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
        procedure_key = _select_group_procedure_key(keys_by_group[(scope, _group_id)])
        schema, created = _create_or_update_schema(
            store, scope, procedure_key, group_traces
        )
        _connect_schema_to_traces(store, schema, group_traces)
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

    base_provenance = {
        "procedure_key": group_id,
        "procedure_field": procedure_key.field,
        "procedure_pattern": procedure_key.raw,
        "group_field": procedure_key.group_field,
        "group_pattern": procedure_key.group_raw,
        "strategy": "procedural",
        "consolidated_at": _utc_now(),
    }
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

    merged_sources = sorted({*existing.source_traces, *sorted_trace_ids})
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
        if procedure_key.group_id in normalized:
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
    store: MemoryStore, cluster: _TraceCluster
) -> tuple[Node, bool]:
    source_traces = sorted({trace.id for trace in cluster.traces})
    cluster_key = _cluster_key(cluster)
    unique_agents = _unique_agent_count(cluster.traces)
    confidence = _consensus_confidence(cluster.traces, unique_agents)
    temporal_hint = detect_temporal_hint(cluster.traces)
    best_trace = max(cluster.traces, key=_trace_quality)
    concept_embedding = cluster.embedding_for(best_trace) or cluster.representative_embedding
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
            embedding=concept_embedding,
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
        embedding=concept_embedding,
        stats=stats,
        provenance={**existing.provenance, **provenance},
    )
    return concept, False


def _cross_scope_promotion(
    store: MemoryStore, concept: Node, *, phase_number: int
) -> Node | None:
    if phase_number < CROSS_SCOPE_PROMOTION_PHASE:
        return None
    if concept.level != "concept" or not concept.scope.startswith("project:"):
        return None
    if concept.embedding is None:
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
        content=best_source.content,
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
