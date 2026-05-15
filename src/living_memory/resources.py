"""Browsable resources and status views for Living Memory."""

from __future__ import annotations

from typing import Any

from living_memory.models import Connection, Node, RetrievalWeights, string_list
from living_memory.scope import normalize_scope
from living_memory.storage import MemoryStore

DEFAULT_CONCEPT_LIMIT = 100_000
DEFAULT_RECENT_LIMIT = 20


def global_concepts(store: MemoryStore, *, limit: int = DEFAULT_CONCEPT_LIMIT) -> dict[str, Any]:
    """Return active global concepts for the static global concepts resource."""

    return concepts_for_scope(store, "global", uri="memory://global/concepts", limit=limit)


def project_concepts(
    store: MemoryStore, name: str, *, limit: int = DEFAULT_CONCEPT_LIMIT
) -> dict[str, Any]:
    """Return active concepts for a project resource template."""

    scope = normalize_scope(name)
    if not scope.startswith("project:"):
        raise ValueError("project concept resources require a project scope")
    project_name = scope.split(":", 1)[1]
    return concepts_for_scope(
        store,
        scope,
        uri=f"memory://project/{project_name}/concepts",
        limit=limit,
    )


def concepts_for_scope(
    store: MemoryStore,
    scope: str,
    *,
    uri: str | None = None,
    limit: int = DEFAULT_CONCEPT_LIMIT,
) -> dict[str, Any]:
    concepts = store.list_nodes(
        level="concept",
        scope=scope,
        include_decayed=False,
        limit=max(0, int(limit)),
    )
    return {
        "uri": uri or f"memory://{scope}/concepts",
        "scope": scope,
        "count": len(concepts),
        "concepts": [node_to_dict(node) for node in concepts],
    }


def memory_stats(store: MemoryStore) -> dict[str, Any]:
    """Return system health, phase, and aggregate metrics."""

    return {
        "uri": "memory://stats",
        "phase": phase_to_dict(store),
        "counts": count_nodes(store),
        "confidence": confidence_summary(store),
        "promotions": promotion_summary(store),
        "scopes": scope_summary(store),
        "retrieval_weights": retrieval_weights_summary(store),
    }


def recent_interactions(
    store: MemoryStore, *, limit: int = DEFAULT_RECENT_LIMIT
) -> dict[str, Any]:
    """Return the latest active traces across scopes."""

    traces = store.list_nodes(
        level="trace",
        include_decayed=False,
        limit=max(0, int(limit)),
    )
    return {
        "uri": "memory://recent",
        "limit": max(0, int(limit)),
        "count": len(traces),
        "traces": [node_to_dict(trace) for trace in traces],
    }


def memory_status(store: MemoryStore, scope: str | None = None) -> dict[str, Any]:
    """Return the reflective status used by the MCP status tool."""

    normalized_scope = normalize_scope(scope) if scope else None
    return {
        "scope": normalized_scope or "all",
        "phase": phase_to_dict(store),
        "counts": count_nodes(store, scope=normalized_scope),
        "confidence": confidence_summary(store, scope=normalized_scope),
        "coverage": coverage_summary(store, scope=normalized_scope),
        "promotions": promotion_summary(store, scope=normalized_scope),
        "retrieval_policy": retrieval_policy_for_scope(store, normalized_scope or "global"),
    }


def node_to_dict(node: Node) -> dict[str, Any]:
    """Serialize a node without expanding stored embeddings."""

    return {
        "id": node.id,
        "level": node.level,
        "content": node.content,
        "scope": node.scope,
        "agent": node.agent,
        "task": node.task,
        "timestamp": node.timestamp,
        "context": dict(node.context),
        "stats": node.stats,
        "provenance": {
            **dict(node.provenance),
            "source_traces": list(node.source_traces),
            "corrections": list(node.corrections),
        },
        "decayed": node.decayed,
        "decay_reason": node.decay_reason,
        "created_at": node.created_at,
        "updated_at": node.updated_at,
    }


def connection_to_dict(connection: Connection) -> dict[str, Any]:
    return {
        "id": connection.id,
        "source_id": connection.source_id,
        "target_id": connection.target_id,
        "type": connection.type,
        "weight": connection.weight,
        "metadata": dict(connection.metadata),
        "created_at": connection.created_at,
        "updated_at": connection.updated_at,
    }


def phase_to_dict(store: MemoryStore) -> dict[str, Any]:
    phase = store.detect_phase()
    return {
        "number": phase.number,
        "name": phase.name,
        "label": phase.label,
        "trace_count": phase.trace_count,
        "min_traces": phase.min_traces,
        "features": list(phase.features),
    }


def count_nodes(store: MemoryStore, scope: str | None = None) -> dict[str, Any]:
    clauses: list[str] = []
    params: list[Any] = []
    if scope is not None:
        clauses.append("scope = ?")
        params.append(scope)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = store.connection.execute(
        f"""
        SELECT level, decayed, COUNT(*) AS count
        FROM nodes
        {where}
        GROUP BY level, decayed
        """,
        params,
    ).fetchall()

    by_level = {
        "trace": {"active": 0, "decayed": 0, "total": 0},
        "concept": {"active": 0, "decayed": 0, "total": 0},
        "schema": {"active": 0, "decayed": 0, "total": 0},
    }
    for row in rows:
        level = str(row["level"])
        state = "decayed" if int(row["decayed"]) else "active"
        count = int(row["count"])
        by_level.setdefault(level, {"active": 0, "decayed": 0, "total": 0})
        by_level[level][state] = count
        by_level[level]["total"] += count

    return {
        "by_level": by_level,
        "total": sum(level_counts["total"] for level_counts in by_level.values()),
        "active": sum(level_counts["active"] for level_counts in by_level.values()),
        "decayed": sum(level_counts["decayed"] for level_counts in by_level.values()),
    }


def confidence_summary(store: MemoryStore, scope: str | None = None) -> dict[str, Any]:
    clauses = ["decayed = 0"]
    params: list[Any] = []
    if scope is not None:
        clauses.append("scope = ?")
        params.append(scope)
    row = store.connection.execute(
        f"""
        SELECT
            COUNT(*) AS count,
            COALESCE(AVG(confidence), 0.0) AS avg_confidence,
            COALESCE(MIN(confidence), 0.0) AS min_confidence,
            COALESCE(MAX(confidence), 0.0) AS max_confidence
        FROM nodes
        WHERE {' AND '.join(clauses)}
        """,
        params,
    ).fetchone()
    return {
        "count": int(row["count"]),
        "average": float(row["avg_confidence"]),
        "minimum": float(row["min_confidence"]),
        "maximum": float(row["max_confidence"]),
    }


def coverage_summary(store: MemoryStore, scope: str | None = None) -> dict[str, Any]:
    clauses = ["decayed = 0"]
    params: list[Any] = []
    if scope is not None:
        clauses.append("scope = ?")
        params.append(scope)
    where = f"WHERE {' AND '.join(clauses)}"
    rows = store.connection.execute(
        f"""
        SELECT scope, level, COUNT(*) AS count
        FROM nodes
        {where}
        GROUP BY scope, level
        ORDER BY scope, level
        """,
        params,
    ).fetchall()

    scopes: dict[str, dict[str, int]] = {}
    for row in rows:
        item_scope = str(row["scope"])
        level = str(row["level"])
        scopes.setdefault(item_scope, {"trace": 0, "concept": 0, "schema": 0})
        scopes[item_scope][level] = int(row["count"])
    return {"scopes": scopes, "scope_count": len(scopes)}


def promotion_summary(store: MemoryStore, scope: str | None = None) -> dict[str, Any]:
    """Summarize active global concepts created or reinforced by promotion."""

    concepts = store.list_nodes(
        level="concept",
        scope="global",
        include_decayed=False,
        limit=DEFAULT_CONCEPT_LIMIT,
    )
    promoted_concepts = 0
    promotion_events = 0
    source_scopes: set[str] = set()
    for concept in concepts:
        promoted_from = string_list(concept.provenance.get("promoted_from"))
        if not promoted_from:
            continue
        if scope not in (None, "global") and scope not in promoted_from:
            continue

        promoted_concepts += 1
        source_scopes.update(promoted_from)
        events = concept.provenance.get("promotion_events")
        promotion_events += len(events) if isinstance(events, list) else 1

    return {
        "promoted_concepts": promoted_concepts,
        "promotion_events": promotion_events,
        "source_scopes": sorted(source_scopes),
    }


def scope_summary(store: MemoryStore) -> list[dict[str, Any]]:
    rows = store.connection.execute(
        """
        SELECT scope, COUNT(*) AS count, SUM(CASE WHEN decayed = 0 THEN 1 ELSE 0 END) AS active
        FROM nodes
        GROUP BY scope
        ORDER BY scope
        """
    ).fetchall()
    return [
        {
            "scope": str(row["scope"]),
            "count": int(row["count"]),
            "active": int(row["active"] or 0),
        }
        for row in rows
    ]


def retrieval_weights_summary(store: MemoryStore) -> list[dict[str, Any]]:
    rows = store.connection.execute(
        """
        SELECT scope, bm25, vector, graph, learning_rate, updated_at
        FROM retrieval_weights
        ORDER BY scope
        """
    ).fetchall()
    return [
        {
            "scope": str(row["scope"]),
            "bm25": float(row["bm25"]),
            "vector": float(row["vector"]),
            "graph": float(row["graph"]),
            "learning_rate": float(row["learning_rate"]),
            "updated_at": str(row["updated_at"]),
        }
        for row in rows
    ]


def retrieval_policy_for_scope(store: MemoryStore, scope: str) -> dict[str, Any]:
    return weights_to_dict(store.get_retrieval_weights(scope).normalized())


def weights_to_dict(weights: RetrievalWeights) -> dict[str, Any]:
    return {
        "scope": weights.scope,
        "bm25": weights.bm25,
        "vector": weights.vector,
        "graph": weights.graph,
        "learning_rate": weights.learning_rate,
        "updated_at": weights.updated_at,
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
