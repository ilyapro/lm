"""Browsable resources and status views for Living Memory."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import closing
from datetime import UTC, datetime, timedelta
from typing import Any
import heapq
import json

from living_memory.config import RetrievalSkewThresholds
from living_memory.health_audit import (
    DEFAULT_LATENCY_QUERY,
    access_metrics,
    feedback_metrics,
    hot_recall_latency,
    instructions_metrics,
    storage_metrics,
)
from living_memory.models import (
    Connection,
    Node,
    RecallEvent,
    RetrievalPolicyFloors,
    RetrievalWeights,
    string_list,
)
from living_memory.scope import ScopePlan, normalize_scope, resolve_scope, scope_family
from living_memory.storage import MemoryStore
from living_memory.validity import read_validity

DEFAULT_CONCEPT_LIMIT = 100_000
DEFAULT_RECENT_LIMIT = 20
DEFAULT_RETRIEVAL_SKEW_PRESSURE_WINDOW_HOURS = 24


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
        "concepts": nodes_to_dict(store, concepts, ScopePlan(scope, (scope,))),
    }


def memory_stats(store: MemoryStore) -> dict[str, Any]:
    """Return current counts, confidence, promotions and retrieval weights."""

    return {
        "uri": "memory://stats",
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
        "traces": nodes_to_dict(store, traces, resolve_scope()),
    }


def memory_status(store: MemoryStore, scope: str | None = None) -> dict[str, Any]:
    """Return the reflective status used by the MCP status tool."""

    normalized_scope = normalize_scope(scope) if scope else None
    return {
        "scope": normalized_scope or "all",
        "counts": count_nodes(store, scope=normalized_scope),
        "confidence": confidence_summary(store, scope=normalized_scope),
        "coverage": coverage_summary(store, scope=normalized_scope),
        "promotions": promotion_summary(store, scope=normalized_scope),
        "retrieval_policy": retrieval_policy_for_scope(store, normalized_scope or "global"),
    }


def nodes_to_dict(store: MemoryStore, nodes: Sequence[Node], plan: ScopePlan) -> list[dict[str, Any]]:
    """Full stored nodes plus relation validity within this read surface."""
    validity = read_validity(store, (node.id for node in nodes), plan)
    return [node_to_dict(node) | validity[node.id] for node in nodes]


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
    """Summarize promotions within the newest active global concept limit."""

    promoted_concepts = 0
    promotion_events = 0
    source_scopes: set[str] = set()
    with store.stream_global_concept_provenance(limit=DEFAULT_CONCEPT_LIMIT) as provenances:
        for _node_id, provenance in provenances:
            promoted_from = string_list(provenance.get("promoted_from"))
            if not promoted_from:
                continue
            if scope not in (None, "global") and scope not in promoted_from:
                continue

            promoted_concepts += 1
            source_scopes.update(promoted_from)
            events = provenance.get("promotion_events")
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
    latency_samples: int = 0,
    latency_query: str = DEFAULT_LATENCY_QUERY,
    instructions_text: str | None = None,
) -> dict[str, Any]:
    """Health report with additive audit metrics for the existing surface."""

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

    ages: list[float] = []
    now_epoch = now.timestamp()

    def dated_traces() -> Iterator[tuple[str, float]]:
        for row in age_rows:
            ref = row["last_accessed"] or row["timestamp"]
            if not ref:
                continue
            try:
                parsed = datetime.fromisoformat(str(ref).replace("Z", "+00:00"))
            except ValueError:
                continue
            age = max(0.0, now_epoch - parsed.timestamp())
            ages.append(age)
            yield str(row["id"]), age

    with closing(store.connection.execute(
        f"SELECT id, timestamp, last_accessed FROM nodes WHERE {trace_where}",
        trace_params,
    )) as age_rows:
        if top_stale >= 1:
            stale_ids = [
                node_id for node_id, _ in heapq.nlargest(
                    int(top_stale), dated_traces(), key=lambda item: item[1],
                )
            ]
        else:
            for _ in dated_traces():
                pass
            stale_ids = []
    ages.sort()

    stale_rows: list[Any] = []
    if stale_ids:
        stale_rows = store.connection.execute(
            f"""
            SELECT nodes.id, content, timestamp, access_count, last_accessed
            FROM json_each(?) AS selected JOIN nodes ON nodes.id = selected.value
            WHERE {trace_where}
            ORDER BY selected.key
            """,
            [json.dumps(stale_ids), *trace_params],
        ).fetchall()

    counts = count_nodes(store, scope=normalized_scope)
    trace_counts = counts["by_level"]["trace"]
    decay_pool = trace_counts["active"] + trace_counts["decayed"]
    decay_rate = (trace_counts["decayed"] / decay_pool) if decay_pool > 0 else 0.0

    policy_scope = normalized_scope or "global"
    report: dict[str, Any] = {
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
            "age_seconds_p50": ages[round(0.5 * (len(ages) - 1))] if ages else None,
            "age_seconds_p90": ages[round(0.9 * (len(ages) - 1))] if ages else None,
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
        "retrieval_skew": retrieval_skew_metrics(store, scope=normalized_scope),
        "feedback_closure": feedback_closure_metrics(
            store,
            scope=normalized_scope,
            window_hours=window_hours,
        ),
    }
    report.update(
        {
            "feedback": feedback_metrics(store, scope=normalized_scope),
            "access": access_metrics(store, scope=normalized_scope),
            "storage": storage_metrics(store),
            "latency": hot_recall_latency(
                store,
                scope=normalized_scope,
                samples=latency_samples,
                query=latency_query,
            ),
            "instructions": instructions_metrics(
                instructions_text,
                default_scope=store.config.default_scope,
            ),
        }
    )
    return report


def retrieval_skew_metrics(
    store: MemoryStore,
    *,
    scope: str | None = None,
    window_hours: int = DEFAULT_RETRIEVAL_SKEW_PRESSURE_WINDOW_HOURS,
) -> dict[str, Any]:
    """Return retrieval-weight skew flags and recent feedback pressure."""

    if window_hours < 1:
        raise ValueError("window_hours must be >= 1")

    normalized_scope = normalize_scope(scope) if scope else None
    thresholds = store.config.retrieval_skew_thresholds
    floors = {
        family: _policy_floor_to_dict(floor)
        for family, floor in store.config.retrieval_policy_floors.items()
    }
    return {
        "thresholds": _skew_thresholds_to_dict(thresholds),
        "floors_in_effect": floors,
        "scopes_at_risk": _retrieval_scopes_at_risk(
            store,
            thresholds=thresholds,
            scope=normalized_scope,
        ),
        "recent_feedback_pressure": _recent_feedback_pressure(
            store,
            scope=normalized_scope,
            window_hours=window_hours,
        ),
    }


_FEEDBACK_CLOSURE_IDENTITY_CLASSES = ("explicit", "transport_only", "none")


def feedback_closure_metrics(
    store: MemoryStore,
    *,
    scope: str | None = None,
    window_hours: int,
) -> dict[str, Any]:
    """Windowed feedback-closure ratio partitioned by recall identity class.

    Each recall event lands in exactly one class, by precedence: ``explicit``
    carries a nonblank persisted session_id (agent and task are optional),
    ``transport_only`` has no such session but a nonempty transport-derived id,
    ``none`` carries neither. An event counts as
    closed when ``feedback_applied = 1`` and its ``created_at`` is inside the
    window (feedback timing is not windowed separately).
    """

    if window_hours < 1:
        raise ValueError("window_hours must be >= 1")

    normalized_scope = normalize_scope(scope) if scope else None
    cutoff = (
        (datetime.now(UTC) - timedelta(hours=window_hours))
        .isoformat()
        .replace("+00:00", "Z")
    )

    clauses = ["created_at >= ?"]
    params: list[Any] = [cutoff]
    if normalized_scope:
        clauses.append("scope = ?")
        params.append(normalized_scope)

    rows = store.connection.execute(
        f"""
        SELECT
            session_id,
            COALESCE(transport_session_id, '') != '' AS has_transport,
            COUNT(*) AS events,
            SUM(CASE WHEN feedback_applied = 1 THEN 1 ELSE 0 END) AS closed
        FROM recall_events
        WHERE {' AND '.join(clauses)}
        GROUP BY session_id, has_transport
        """,
        params,
    ).fetchall()

    identity_coverage: dict[str, dict[str, Any]] = {
        name: {"events": 0, "closed": 0, "closure_ratio": None}
        for name in _FEEDBACK_CLOSURE_IDENTITY_CLASSES
    }
    for row in rows:
        # The writer has already resolved the session alias and stringified it.
        # Use the explicit reconnect consumer's nonblank check on those bytes.
        session_id = row["session_id"]
        identity_class = ("explicit" if session_id and session_id.strip() else
                          "transport_only" if row["has_transport"] else "none")
        coverage = identity_coverage[identity_class]
        coverage["events"] += int(row["events"])
        coverage["closed"] += int(row["closed"] or 0)
    for coverage in identity_coverage.values():
        if coverage["events"]:
            coverage["closure_ratio"] = coverage["closed"] / coverage["events"]

    total_events = sum(item["events"] for item in identity_coverage.values())
    total_closed = sum(item["closed"] for item in identity_coverage.values())
    return {
        "window_hours": window_hours,
        "recall_events_in_window": total_events,
        "feedback_applied_in_window": total_closed,
        "closure_ratio": (total_closed / total_events) if total_events > 0 else None,
        "identity_coverage": identity_coverage,
    }


def recall_events_summary(
    store: MemoryStore,
    *,
    scope: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Summary of recall events for the dashboard resource."""

    normalized_scope = normalize_scope(scope) if scope else None

    scope_clauses: list[str] = []
    scope_params: list[Any] = []
    if normalized_scope:
        scope_clauses.append("(scope = ? OR requested_scope = ?)")
        scope_params.extend([normalized_scope, normalized_scope])
    scope_where = (" WHERE " + " AND ".join(scope_clauses)) if scope_clauses else ""

    total = int(
        store.connection.execute(
            f"SELECT COUNT(*) AS c FROM recall_events{scope_where}",
            scope_params,
        ).fetchone()["c"]
    )
    feedback_applied = int(
        store.connection.execute(
            f"SELECT COUNT(*) AS c FROM recall_events{' WHERE ' + ' AND '.join(scope_clauses + ['feedback_applied = 1']) if scope_clauses else ' WHERE feedback_applied = 1'}",
            scope_params,
        ).fetchone()["c"]
    )

    events = store.list_recall_events(
        scope=normalized_scope,
        limit=max(0, int(limit)),
    )
    recent = [recall_event_to_dict(e) for e in events]

    return {
        "uri": "memory://recall_events",
        "scope": normalized_scope or "all",
        "total": total,
        "feedback_applied": feedback_applied,
        "recent": recent,
    }


def connections_summary(
    store: MemoryStore,
    *,
    scope: str | None = None,
) -> dict[str, Any]:
    """Summary of graph connections for the dashboard resource."""

    rows = store.connection.execute(
        "SELECT type, COUNT(*) AS count FROM connections GROUP BY type ORDER BY count DESC"
    ).fetchall()
    by_type = [{"type": str(row["type"]), "count": int(row["count"])} for row in rows]
    total = sum(item["count"] for item in by_type)

    return {
        "uri": "memory://connections",
        "total": total,
        "by_type": by_type,
    }


def recall_event_to_dict(event: RecallEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "query": event.query,
        "scope": event.scope,
        "requested_scope": event.requested_scope,
        "transport_session_id": event.transport_session_id,
        "result_count": len(event.results),
        "feedback_applied": event.feedback_applied,
        "created_at": event.created_at,
    }


def weights_to_dict(weights: RetrievalWeights) -> dict[str, Any]:
    return {
        "scope": weights.scope,
        "bm25": weights.bm25,
        "vector": weights.vector,
        "graph": weights.graph,
        "learning_rate": weights.learning_rate,
        "updated_at": weights.updated_at,
    }


def _skew_thresholds_to_dict(thresholds: RetrievalSkewThresholds) -> dict[str, float]:
    return {
        "bm25_monoculture": thresholds.bm25_monoculture,
        "near_zero_vector": thresholds.near_zero_vector,
        "near_zero_graph": thresholds.near_zero_graph,
    }


def _policy_floor_to_dict(floor: RetrievalPolicyFloors) -> dict[str, float]:
    return {
        "bm25_max": floor.bm25_max,
        "vector_min": floor.vector_min,
        "graph_min": floor.graph_min,
        "bm25_min": floor.bm25_min,
    }


def _retrieval_scopes_at_risk(
    store: MemoryStore,
    *,
    thresholds: RetrievalSkewThresholds,
    scope: str | None,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    where = ""
    if scope is not None:
        where = "WHERE scope = ?"
        params.append(scope)
    rows = store.connection.execute(
        f"""
        SELECT scope, bm25, vector, graph, learning_rate, updated_at
        FROM retrieval_weights
        {where}
        ORDER BY scope
        """,
        params,
    ).fetchall()

    risks: list[dict[str, Any]] = []
    for row in rows:
        row_scope = str(row["scope"])
        if row_scope == "default":
            continue
        family = scope_family(row_scope)
        raw_weights = RetrievalWeights(
            scope=row_scope,
            bm25=float(row["bm25"]),
            vector=float(row["vector"]),
            graph=float(row["graph"]),
            learning_rate=float(row["learning_rate"]),
            updated_at=str(row["updated_at"]),
        ).normalized()
        raw = _weights_triplet(raw_weights)
        applicable, effective = store.evaluate_retrieval_weight_floors(row_scope, raw_weights)
        floor = _policy_floor_to_dict(applicable) if applicable is not None else {}
        flags = _retrieval_skew_flags(raw, thresholds=thresholds)
        floor_violation = _retrieval_floor_violation(raw, floor) if floor else {}
        if not flags and not floor_violation:
            continue

        risks.append(
            {
                "scope": row_scope,
                "family": family,
                "raw": raw,
                "effective_after_floor": _weights_triplet(effective),
                "applicable_floors": floor,
                "flags": flags,
                "floor_violation": floor_violation,
            }
        )
    return risks


def _weights_triplet(weights: RetrievalWeights) -> dict[str, float]:
    return {
        "bm25": weights.bm25,
        "vector": weights.vector,
        "graph": weights.graph,
    }


def _retrieval_skew_flags(
    raw: dict[str, float],
    *,
    thresholds: RetrievalSkewThresholds,
) -> list[str]:
    flags: list[str] = []
    if raw["bm25"] > thresholds.bm25_monoculture:
        flags.append("bm25_monoculture")
    if raw["vector"] < thresholds.near_zero_vector:
        flags.append("near_zero_vector")
    if raw["graph"] < thresholds.near_zero_graph:
        flags.append("near_zero_graph")
    return flags


def _retrieval_floor_violation(
    raw: dict[str, float],
    floor: dict[str, float],
) -> dict[str, dict[str, float]]:
    violation: dict[str, dict[str, float]] = {}
    if raw["bm25"] > floor["bm25_max"]:
        violation["bm25"] = {
            "current": raw["bm25"],
            "ceiling": floor["bm25_max"],
        }
    elif raw["bm25"] < floor.get("bm25_min", 0.0):
        violation["bm25"] = {
            "current": raw["bm25"],
            "floor": floor["bm25_min"],
        }
    if raw["vector"] < floor["vector_min"]:
        violation["vector"] = {
            "current": raw["vector"],
            "floor": floor["vector_min"],
        }
    if raw["graph"] < floor["graph_min"]:
        violation["graph"] = {
            "current": raw["graph"],
            "floor": floor["graph_min"],
        }
    return violation


def _recent_feedback_pressure(
    store: MemoryStore,
    *,
    scope: str | None,
    window_hours: int,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    cutoff = (now - timedelta(hours=window_hours)).isoformat().replace("+00:00", "Z")
    clauses = ["feedback_applied = 1", "feedback_applied_at >= ?"]
    params: list[Any] = [cutoff]
    if scope is not None:
        clauses.append("scope = ?")
        params.append(scope)

    rows = store.connection.execute(
        f"""
        SELECT results
        FROM recall_events
        WHERE {' AND '.join(clauses)}
        """,
        params,
    )

    counts = {"bm25": 0, "vector": 0, "graph": 0}
    consumed = 0
    try:
        for row in rows:
            consumed += 1
            for result in _json_result_list(row["results"]):
                dominant = _dominant_result_method(result)
                if dominant is not None:
                    counts[dominant] += 1
    finally:
        rows.close()

    total = sum(counts.values())
    return {
        "window_hours": window_hours,
        "consumed_recall_events": consumed,
        "dominant_method_counts": counts,
        "bm25_dominance_ratio": (counts["bm25"] / total) if total > 0 else None,
    }


def _json_result_list(value: Any) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [dict(item) for item in parsed if isinstance(item, dict)]


def _dominant_result_method(result: dict[str, Any]) -> str | None:
    scores = {
        "bm25": _float_value(result.get("bm25_score")),
        "vector": _float_value(result.get("vector_score")),
        "graph": _float_value(result.get("graph_score")),
    }
    method, score = max(scores.items(), key=lambda item: item[1])
    if score > 0.0:
        return method

    methods = result.get("methods")
    if isinstance(methods, list):
        for item in methods:
            method_name = str(item)
            if method_name in scores:
                return method_name
    return None


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
