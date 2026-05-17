"""Prompt builders for active memory retrieval context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from living_memory.models import Node
from living_memory.retrieval import MemoryRecallService, REJECTED_ALTERNATIVE_KIND
from living_memory.scope import ScopePlan, resolve_scope
from living_memory.storage import MemoryStore

DEFAULT_CONTEXT_CONCEPTS = 5
DEFAULT_CONTEXT_SCHEMAS = 3


@dataclass(frozen=True, slots=True)
class ContextConcept:
    node: Node
    score: float
    methods: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ContextSchema:
    node: Node
    score: float
    methods: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DecisionHistory:
    primary: Node
    reasons: tuple[str, ...]
    score: float


def retrieval_context_prompt(
    store: MemoryStore,
    *,
    task: str = "",
    scope: str | None = None,
    max_concepts: int = DEFAULT_CONTEXT_CONCEPTS,
    max_schemas: int = DEFAULT_CONTEXT_SCHEMAS,
    min_confidence: float = 0.0,
    retrieval_policy: str = "balanced",
    agent: str | None = None,
    ambient_context: dict[str, Any] | None = None,
) -> str:
    """Build a formatted active context block from top-ranked concepts and schemas."""

    query = str(task or "").strip()
    ambient = dict(ambient_context or {})
    if agent:
        ambient["agent"] = agent
    plan = resolve_scope(query=query, scope=scope, ambient_context=ambient, store=store)
    concepts = select_context_concepts(
        store,
        task=query,
        plan=plan,
        max_concepts=max_concepts,
        min_confidence=min_confidence,
        retrieval_policy=retrieval_policy,
        ambient_context=ambient,
    )
    schemas = select_context_schemas(
        store,
        task=query,
        plan=plan,
        max_schemas=max_schemas,
        min_confidence=min_confidence,
        ambient_context=ambient,
    )
    decisions = select_decision_history(
        store,
        task=query,
        plan=plan,
        max_items=max_concepts,
        ambient_context=ambient,
    )
    return format_retrieval_context(
        concepts,
        schemas=schemas,
        decision_history=decisions,
        task=query,
        plan=plan,
        store=store,
        min_confidence=min_confidence,
        retrieval_policy=retrieval_policy,
    )


def select_context_concepts(
    store: MemoryStore,
    *,
    task: str,
    plan: ScopePlan,
    max_concepts: int = DEFAULT_CONTEXT_CONCEPTS,
    min_confidence: float = 0.0,
    retrieval_policy: str = "balanced",
    ambient_context: dict[str, Any] | None = None,
) -> list[ContextConcept]:
    """Select concept nodes by scope, task relevance, confidence, and policy weights."""

    limit = max(0, int(max_concepts))
    if limit == 0:
        return []

    min_confidence = max(0.0, min(1.0, float(min_confidence)))
    selected: dict[str, ContextConcept] = {}
    if task.strip():
        service = MemoryRecallService(store)
        recall_results = service.memory_recall(
            task,
            scope=plan.requested_scope,
            ambient_context=ambient_context,
            max_results=max(limit * 6, 20),
            log_access=False,
        )
        for result in recall_results:
            node = result.node
            if node.level != "concept" or node.confidence < min_confidence:
                continue
            if not plan.allows(node.scope):
                continue
            selected[node.id] = ContextConcept(
                node=node,
                score=result.score,
                methods=tuple(result.methods),
            )

    task_tokens = _tokens(task)
    for item_scope in plan.scopes:
        weights = store.get_retrieval_weights(item_scope).normalized()
        for node in store.list_nodes(
            level="concept",
            scope=item_scope,
            include_decayed=False,
            limit=1_000,
        ):
            if node.confidence < min_confidence or node.id in selected:
                continue
            score = _policy_score(
                node,
                task_tokens=task_tokens,
                plan=plan,
                policy=retrieval_policy,
                bm25_weight=weights.bm25,
                vector_weight=weights.vector,
                graph_weight=weights.graph,
            )
            selected[node.id] = ContextConcept(node=node, score=score, methods=("policy",))

    concepts = sorted(
        selected.values(),
        key=lambda concept: (
            concept.score,
            concept.node.confidence,
            concept.node.usefulness_score,
            concept.node.access_count,
            concept.node.timestamp,
        ),
        reverse=True,
    )
    return concepts[:limit]


def select_context_schemas(
    store: MemoryStore,
    *,
    task: str,
    plan: ScopePlan,
    max_schemas: int = DEFAULT_CONTEXT_SCHEMAS,
    min_confidence: float = 0.0,
    ambient_context: dict[str, Any] | None = None,
) -> list[ContextSchema]:
    """Select schema nodes whose trigger matches the task."""

    limit = max(0, int(max_schemas))
    if limit == 0:
        return []
    if not task.strip():
        return []

    min_confidence = max(0.0, min(1.0, float(min_confidence)))
    service = MemoryRecallService(store)
    recall_results = service.memory_recall(
        task,
        scope=plan.requested_scope,
        ambient_context=ambient_context,
        max_results=max(limit * 4, 12),
        log_access=False,
        log_event=False,
    )
    selected: list[ContextSchema] = []
    for result in recall_results:
        node = result.node
        if node.level != "schema":
            continue
        if node.confidence < min_confidence:
            continue
        if not plan.allows(node.scope):
            continue
        if not node.context.get("trigger"):
            continue
        if "trigger" not in result.methods:
            continue
        selected.append(
            ContextSchema(node=node, score=result.score, methods=tuple(result.methods))
        )
        if len(selected) >= limit:
            break
    return selected


def select_decision_history(
    store: MemoryStore,
    *,
    task: str,
    plan: ScopePlan,
    max_items: int = DEFAULT_CONTEXT_CONCEPTS,
    ambient_context: dict[str, Any] | None = None,
) -> list[DecisionHistory]:
    """Return primary traces that have rejected alternatives relevant to the task."""

    limit = max(0, int(max_items))
    if limit == 0 or not task.strip():
        return []

    service = MemoryRecallService(store)
    recall_results = service.memory_recall(
        task,
        scope=plan.requested_scope,
        ambient_context=ambient_context,
        depth="decision",
        max_results=max(limit * 6, 20),
        log_access=False,
    )

    selected: dict[str, DecisionHistory] = {}
    for result in recall_results:
        node = result.node
        if node.context.get("is_rejected_alternative") or not plan.allows(node.scope):
            continue
        reasons = _rejected_alternative_reasons(store, node, plan)
        if not reasons:
            continue
        selected[node.id] = DecisionHistory(primary=node, reasons=tuple(reasons), score=result.score)
        if len(selected) >= limit:
            break
    return list(selected.values())


def format_retrieval_context(
    concepts: list[ContextConcept],
    *,
    schemas: list[ContextSchema] | None = None,
    decision_history: list[DecisionHistory] | None = None,
    task: str,
    plan: ScopePlan,
    store: MemoryStore,
    min_confidence: float,
    retrieval_policy: str,
) -> str:
    policy_weights = store.get_retrieval_weights(plan.requested_scope).normalized()
    lines = [
        "BEGIN ACTIVE MEMORY CONTEXT",
        f"scope_plan: {' > '.join(plan.scopes)}",
        f"task: {task or '(none)'}",
        (
            "retrieval_policy: "
            f"{retrieval_policy} "
            f"bm25={policy_weights.bm25:.2f} "
            f"vector={policy_weights.vector:.2f} "
            f"graph={policy_weights.graph:.2f}"
        ),
        f"min_confidence: {max(0.0, min(1.0, float(min_confidence))):.2f}",
        "concepts:",
    ]
    if not concepts:
        lines.append("- none")
    for index, concept in enumerate(concepts, start=1):
        node = concept.node
        method_text = ",".join(concept.methods) if concept.methods else "policy"
        lines.append(
            (
                f"{index}. id={node.id} scope={node.scope} "
                f"confidence={node.confidence:.2f} usefulness={node.usefulness_score:.2f} "
                f"score={concept.score:.4f} methods={method_text}"
            )
        )
        lines.append(f"   {_single_line(node.content)}")
    if schemas:
        lines.append("skills:")
        for index, schema in enumerate(schemas, start=1):
            node = schema.node
            trigger = str(node.context.get("trigger") or "")
            method_text = ",".join(schema.methods) if schema.methods else "trigger"
            lines.append(
                (
                    f"{index}. id={node.id} scope={node.scope} "
                    f"trigger=\"{trigger}\" confidence={node.confidence:.2f} "
                    f"score={schema.score:.4f} methods={method_text}"
                )
            )
            lines.append(f"   {_single_line(node.content)}")
    if decision_history:
        lines.append("decision_history:")
        for index, decision in enumerate(decision_history, start=1):
            node = decision.primary
            reasons = "; ".join(_single_line(reason) for reason in decision.reasons)
            lines.append(
                (
                    f"{index}. id={node.id} scope={node.scope} "
                    f"confidence={node.confidence:.2f} score={decision.score:.4f}"
                )
            )
            lines.append(f"   {_single_line(node.content)}")
            lines.append(f"   Alternatives rejected: {len(decision.reasons)} ({reasons})")
    lines.append("END ACTIVE MEMORY CONTEXT")
    return "\n".join(lines)


def _policy_score(
    node: Node,
    *,
    task_tokens: set[str],
    plan: ScopePlan,
    policy: str,
    bm25_weight: float,
    vector_weight: float,
    graph_weight: float,
) -> float:
    node_tokens = _tokens(node.content)
    if task_tokens and node_tokens:
        overlap = len(task_tokens & node_tokens) / len(task_tokens | node_tokens)
    else:
        overlap = 0.5

    lexical_score = 0.2 + overlap
    semantic_score = 0.25 + 0.75 * overlap
    graph_score = min(1.0, 0.1 * node.access_count)
    weighted = bm25_weight * lexical_score + vector_weight * semantic_score + graph_weight * graph_score

    scope_boost = 1.0 + max(0, len(plan.scopes) - plan.rank(node.scope) - 1) * 0.05
    confidence_boost = 0.5 + node.confidence
    usefulness_boost = 1.0 + max(-0.5, min(0.5, node.usefulness_score))
    policy_name = str(policy or "balanced").lower()
    if policy_name in {"confidence", "trusted"}:
        confidence_boost += 0.35 * node.confidence
    elif policy_name in {"project", "local"} and node.scope == plan.requested_scope:
        scope_boost += 0.2
    elif policy_name in {"recent", "fresh"}:
        weighted += 0.02 * min(10, node.access_count)
    return weighted * scope_boost * confidence_boost * usefulness_boost


def _tokens(value: str) -> set[str]:
    return {token.strip("_") for token in re.findall(r"\w+", value.lower()) if len(token.strip("_")) > 2}


def _single_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _rejected_alternative_reasons(
    store: MemoryStore,
    primary: Node,
    plan: ScopePlan,
) -> list[str]:
    reasons: list[str] = []
    for connection in store.list_connections(target_id=primary.id, relation_type="contradicts"):
        if connection.metadata.get("kind") != REJECTED_ALTERNATIVE_KIND:
            continue
        reason = str(connection.metadata.get("reason") or "").strip()
        if not reason:
            continue
        rejected = store.get_node(connection.source_id)
        if rejected is None or rejected.decayed or not plan.allows(rejected.scope):
            continue
        if not rejected.context.get("is_rejected_alternative"):
            continue
        reasons.append(reason)
    return reasons
