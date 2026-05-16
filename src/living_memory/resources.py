"""Browsable resources and status views for Living Memory."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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


def memory_health(
    store: MemoryStore,
    *,
    scope: str | None = None,
    window_hours: int = 168,
    top_stale: int = 5,
) -> dict[str, Any]:
    """Health report: activity ratios, dedup density, staleness, retrieval policy."""

    if window_hours < 1:
        raise ValueError("window_hours must be >= 1")
    if top_stale < 0:
        raise ValueError("top_stale must be >= 0")

    normalized_scope = normalize_scope(scope) if scope else None
    now = datetime.now(UTC)
    window_cutoff = (now - timedelta(hours=window_hours)).isoformat().replace("+00:00", "Z")

    scope_clauses: list[str] = []
    scope_params: list[Any] = []
    if normalized_scope:
        scope_clauses.append("scope = ?")
        scope_params.append(normalized_scope)
    scope_where = (" WHERE " + " AND ".join(scope_clauses)) if scope_clauses else ""

    recall_total = int(
        store.connection.execute(
            f"SELECT COUNT(*) AS c FROM recall_events{scope_where}",
            scope_params,
        ).fetchone()["c"]
    )
    window_clauses = list(scope_clauses) + ["created_at >= ?"]
    recall_window = int(
        store.connection.execute(
            f"SELECT COUNT(*) AS c FROM recall_events WHERE {' AND '.join(window_clauses)}",
            list(scope_params) + [window_cutoff],
        ).fetchone()["c"]
    )

    trace_clauses = ["level = 'trace'", "decayed = 0"]
    trace_params: list[Any] = []
    if normalized_scope:
        trace_clauses.append("scope = ?")
        trace_params.append(normalized_scope)
    trace_where = " AND ".join(trace_clauses)

    remember_window = int(
        store.connection.execute(
            f"SELECT COUNT(*) AS c FROM nodes WHERE {trace_where} AND timestamp >= ?",
            list(trace_params) + [window_cutoff],
        ).fetchone()["c"]
    )
    ratio = (recall_window / remember_window) if remember_window > 0 else None

    dup_row = store.connection.execute(
        f"""
        SELECT COUNT(*) AS total, COUNT(DISTINCT content) AS distinct_content
        FROM nodes WHERE {trace_where}
        """,
        trace_params,
    ).fetchone()
    total_traces = int(dup_row["total"])
    distinct_contents = int(dup_row["distinct_content"])
    duplicate_excess = max(0, total_traces - distinct_contents)
    duplicate_density = (duplicate_excess / total_traces) if total_traces > 0 else 0.0

    age_rows = store.connection.execute(
        f"SELECT timestamp, last_accessed FROM nodes WHERE {trace_where}",
        trace_params,
    ).fetchall()
    ages: list[float] = []
    now_epoch = now.timestamp()
    for row in age_rows:
        ref = row["last_accessed"] or row["timestamp"]
        if not ref:
            continue
        try:
            parsed = datetime.fromisoformat(str(ref).replace("Z", "+00:00"))
        except ValueError:
            continue
        ages.append(max(0.0, now_epoch - parsed.timestamp()))

    stale_rows: list[Any] = []
    if top_stale > 0:
        stale_rows = store.connection.execute(
            f"""
            SELECT id, content, timestamp, access_count, last_accessed
            FROM nodes WHERE {trace_where}
            ORDER BY (CASE WHEN last_accessed IS NULL THEN 0 ELSE 1 END),
                     timestamp ASC
            LIMIT ?
            """,
            list(trace_params) + [int(top_stale)],
        ).fetchall()

    counts = count_nodes(store, scope=normalized_scope)
    trace_counts = counts["by_level"]["trace"]
    decay_pool = trace_counts["active"] + trace_counts["decayed"]
    decay_rate = (trace_counts["decayed"] / decay_pool) if decay_pool > 0 else 0.0

    policy_scope = normalized_scope or "global"
    return {
        "scope": normalized_scope or "all",
        "window_hours": window_hours,
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "counts": counts,
        "activity": {
            "recall_total": recall_total,
            "recall_in_window": recall_window,
            "remember_in_window": remember_window,
            "recall_to_remember_ratio": ratio,
        },
        "dedup": {
            "total_traces": total_traces,
            "distinct_contents": distinct_contents,
            "duplicate_excess": duplicate_excess,
            "duplicate_density": duplicate_density,
        },
        "staleness": {
            "decay_rate": decay_rate,
            "age_seconds_p50": _percentile(ages, 0.5),
            "age_seconds_p90": _percentile(ages, 0.9),
            "top_stale": [
                {
                    "id": str(row["id"]),
                    "content_preview": str(row["content"])[:120],
                    "timestamp": str(row["timestamp"]),
                    "access_count": int(row["access_count"]),
                    "last_accessed": (
                        str(row["last_accessed"]) if row["last_accessed"] else None
                    ),
                }
                for row in stale_rows
            ],
        },
        "retrieval_policy": retrieval_policy_for_scope(store, policy_scope),
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(quantile * (len(ordered) - 1)))))
    return ordered[idx]


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
