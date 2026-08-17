"""Reusable audit metrics for the Living Memory health surface."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import Any
from urllib.parse import quote
import sqlite3
import time

from living_memory.config import MemoryConfig
from living_memory.models import Node, RecallEvent, RetrievalWeights
from living_memory.retrieval import memory_recall
from living_memory.scope import normalize_scope, scope_family
from living_memory.storage import MemoryStore

DEFAULT_LATENCY_QUERY = (
    "Living Memory hot recall latency baseline duplicate density feedback applied"
)
DEFAULT_TOP_CANDIDATE_SCOPES = 20
INSTRUCTIONS_CONTRACT_COMMAND = (
    "PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 pytest -q "
    "tests/test_instructions_imperative.py -p no:cacheprovider"
)

_LEAKAGE_SCOPE_PATTERNS = (
    ("=", "rise"),
    ("LIKE", "rise/%"),
    ("LIKE", "scope:rise%"),
    ("=", "breakthrough"),
    ("LIKE", "breakthrough/%"),
    ("=", "ocpa-generative-action-substrate-v1"),
    ("LIKE", "ocpa-generative-action-substrate-v1/%"),
    ("=", "project:rise"),
    ("LIKE", "project:rise/%"),
    ("=", "project:breakthrough"),
    ("LIKE", "project:breakthrough/%"),
    ("=", "project:ocpa-generative-action-substrate-v1"),
    ("LIKE", "project:ocpa-generative-action-substrate-v1/%"),
)


StoreSource = MemoryStore | sqlite3.Connection | str | Path


def health_audit_metrics(
    source: StoreSource,
    *,
    scope: str | None = None,
    top_candidate_scopes: int = DEFAULT_TOP_CANDIDATE_SCOPES,
    latency_samples: int = 0,
    latency_query: str = DEFAULT_LATENCY_QUERY,
    instructions_text: str | None = None,
) -> dict[str, Any]:
    """Return deterministic, JSON-serializable health audit metric sections.

    ``source`` may be an existing ``MemoryStore``, a fixture
    ``sqlite3.Connection``, or a database path. Path inputs are opened with
    SQLite ``mode=ro`` and never run ``MemoryStore`` schema initialization.
    """

    store, should_close = _coerce_store(source)
    try:
        normalized_scope = normalize_scope(scope) if scope else None
        return {
            "dedup": duplicate_density(store, scope=normalized_scope),
            "feedback": feedback_metrics(store, scope=normalized_scope),
            "access": access_metrics(store, scope=normalized_scope),
            "scope_hygiene": scope_hygiene_metrics(
                store,
                top_candidate_scopes=top_candidate_scopes,
            ),
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
    finally:
        if should_close:
            store.close()


compute_health_audit = health_audit_metrics


def duplicate_density(store: MemoryStore, *, scope: str | None = None) -> dict[str, Any]:
    """Return active trace exact-content collision metrics."""

    clauses = ["level = 'trace'", "decayed = 0"]
    params: list[Any] = []
    if scope is not None:
        clauses.append("scope = ?")
        params.append(scope)

    row = store.connection.execute(
        f"""
        SELECT COUNT(*) AS total_traces,
               COUNT(DISTINCT content) AS distinct_contents
        FROM nodes
        WHERE {' AND '.join(clauses)}
        """,
        params,
    ).fetchone()
    total_traces = int(row["total_traces"])
    distinct_contents = int(row["distinct_contents"])
    duplicate_excess = max(0, total_traces - distinct_contents)
    return {
        "total_traces": total_traces,
        "distinct_contents": distinct_contents,
        "duplicate_excess": duplicate_excess,
        "duplicate_density": (
            duplicate_excess / total_traces if total_traces > 0 else 0.0
        ),
    }


def feedback_metrics(store: MemoryStore, *, scope: str | None = None) -> dict[str, Any]:
    """Return recall-event feedback application ratio metrics."""

    where = "WHERE scope = ?" if scope is not None else ""
    params: list[Any] = [scope] if scope is not None else []
    row = store.connection.execute(
        f"""
        SELECT COUNT(*) AS recall_events,
               SUM(CASE WHEN feedback_applied = 1 THEN 1 ELSE 0 END)
                   AS feedback_applied
        FROM recall_events
        {where}
        """,
        params,
    ).fetchone()
    recall_events = int(row["recall_events"])
    feedback_applied = int(row["feedback_applied"] or 0)
    feedback_missing = recall_events - feedback_applied
    return {
        "recall_events": recall_events,
        "feedback_applied": feedback_applied,
        "feedback_missing": feedback_missing,
        "feedback_applied_ratio": _ratio(feedback_applied, recall_events),
    }


def access_metrics(store: MemoryStore, *, scope: str | None = None) -> dict[str, Any]:
    """Return never-accessed ratios for active nodes and active traces."""

    return {
        "active_nodes": _never_accessed_metric(store, [], scope=scope),
        "active_traces": _never_accessed_metric(
            store,
            ["level = 'trace'"],
            scope=scope,
        ),
    }


def scope_hygiene_metrics(
    store: MemoryStore,
    *,
    top_candidate_scopes: int = DEFAULT_TOP_CANDIDATE_SCOPES,
) -> dict[str, Any]:
    """Return baseline-compatible scope leakage candidate metrics."""

    candidate_cte = _candidate_scope_cte()
    candidate_count = int(
        store.connection.execute(
            f"{candidate_cte} SELECT COUNT(*) AS c FROM candidate_scopes"
        ).fetchone()["c"]
    )

    node_predicate = _scope_leakage_predicate("scope")
    row = store.connection.execute(
        f"""
        SELECT COUNT(*) AS active_candidate_nodes,
               SUM(CASE WHEN level = 'trace' THEN 1 ELSE 0 END)
                   AS active_candidate_traces,
               SUM(
                   CASE
                   WHEN level = 'trace' AND access_count = 0 THEN 1
                   ELSE 0
                   END
               ) AS active_candidate_traces_never_accessed
        FROM nodes
        WHERE decayed = 0 AND ({node_predicate})
        """
    ).fetchone()
    active_candidate_nodes = int(row["active_candidate_nodes"])
    active_candidate_traces = int(row["active_candidate_traces"] or 0)
    active_candidate_traces_never_accessed = int(
        row["active_candidate_traces_never_accessed"] or 0
    )

    zero_recall = int(
        store.connection.execute(
            f"""
            {candidate_cte}
            SELECT COUNT(*) AS c
            FROM candidate_scopes AS cs
            WHERE NOT EXISTS (
                SELECT 1
                FROM recall_events AS r
                WHERE r.scope = cs.scope
                   OR r.requested_scope = cs.scope
            )
            """
        ).fetchone()["c"]
    )

    top_rows = store.connection.execute(
        f"""
        {candidate_cte},
        active_trace_counts AS (
            SELECT scope,
                   COUNT(*) AS active_traces,
                   SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END)
                       AS active_candidate_traces_never_accessed
            FROM nodes
            WHERE decayed = 0
              AND level = 'trace'
              AND ({node_predicate})
            GROUP BY scope
        ),
        recall_scope_counts AS (
            SELECT scope, COUNT(*) AS recall_events_as_scope
            FROM recall_events
            WHERE {_scope_leakage_predicate('scope')}
            GROUP BY scope
        ),
        recall_requested_counts AS (
            SELECT requested_scope AS scope,
                   COUNT(*) AS recall_events_as_requested_scope
            FROM recall_events
            WHERE {_scope_leakage_predicate('requested_scope')}
            GROUP BY requested_scope
        )
        SELECT cs.scope AS scope,
               COALESCE(atc.active_traces, 0) AS active_traces,
               COALESCE(atc.active_candidate_traces_never_accessed, 0)
                   AS active_candidate_traces_never_accessed,
               COALESCE(rsc.recall_events_as_scope, 0)
                   AS recall_events_as_scope,
               COALESCE(rrc.recall_events_as_requested_scope, 0)
                   AS recall_events_as_requested_scope
        FROM candidate_scopes AS cs
        LEFT JOIN active_trace_counts AS atc ON atc.scope = cs.scope
        LEFT JOIN recall_scope_counts AS rsc ON rsc.scope = cs.scope
        LEFT JOIN recall_requested_counts AS rrc ON rrc.scope = cs.scope
        ORDER BY active_traces DESC, recall_events_as_scope ASC, cs.scope ASC
        LIMIT ?
        """,
        (max(0, int(top_candidate_scopes)),),
    ).fetchall()

    return {
        "candidate_scopes": candidate_count,
        "active_candidate_nodes": active_candidate_nodes,
        "active_candidate_traces": active_candidate_traces,
        "active_candidate_traces_never_accessed": (
            active_candidate_traces_never_accessed
        ),
        "active_candidate_trace_never_accessed_ratio": _ratio(
            active_candidate_traces_never_accessed,
            active_candidate_traces,
        ),
        "candidate_scopes_with_zero_recall_events": zero_recall,
        "top_candidate_scopes": [
            {
                "scope": str(row["scope"]),
                "active_traces": int(row["active_traces"]),
                "active_candidate_traces_never_accessed": int(
                    row["active_candidate_traces_never_accessed"]
                ),
                "recall_events_as_scope": int(row["recall_events_as_scope"]),
                "recall_events_as_requested_scope": int(
                    row["recall_events_as_requested_scope"]
                ),
            }
            for row in top_rows
        ],
    }


def storage_metrics(store: MemoryStore) -> dict[str, Any]:
    """Return filesystem and SQLite page-size storage metrics."""

    db_path = store.db_path
    db_path_text = str(db_path)
    db_size_bytes: int | None
    if db_path_text == ":memory:":
        db_size_bytes = None
    else:
        try:
            db_size_bytes = int(Path(db_path).stat().st_size)
        except OSError:
            db_size_bytes = None

    page_count = int(store.connection.execute("PRAGMA page_count").fetchone()[0])
    page_size = int(store.connection.execute("PRAGMA page_size").fetchone()[0])
    return {
        "db_path": db_path_text,
        "db_size_bytes": db_size_bytes,
        "page_count": page_count,
        "page_size": page_size,
        "page_bytes": page_count * page_size,
    }


def hot_recall_latency(
    store: MemoryStore,
    *,
    scope: str | None = None,
    samples: int = 0,
    query: str = DEFAULT_LATENCY_QUERY,
) -> dict[str, Any] | None:
    """Measure opt-in hot recall latency without access or event writes."""

    sample_count = int(samples)
    if sample_count <= 0:
        return None

    latency_store = _read_only_latency_store(store)

    def run_once() -> float:
        started = time.perf_counter()
        memory_recall(
            latency_store,
            query,
            scope=scope,
            ambient_context={"benchmark": "memory_health.latency_probe"},
            max_results=1,
            depth=1,
            log_access=False,
            log_event=False,
        )
        return round((time.perf_counter() - started) * 1000.0, 3)

    warmup_ms = run_once()
    samples_ms = [run_once() for _ in range(sample_count)]
    return {
        "query": query,
        "scope": scope or "all",
        "max_results": 1,
        "depth": 1,
        "log_access": False,
        "log_event": False,
        "samples": sample_count,
        "warmup_ms": warmup_ms,
        "samples_ms": samples_ms,
        "summary": {
            "min_ms": min(samples_ms),
            "median_ms": round(float(median(samples_ms)), 3),
            "mean_ms": round(float(mean(samples_ms)), 3),
            "max_ms": max(samples_ms),
        },
    }


def instructions_metrics(
    instructions_text: str | None,
    *,
    default_scope: str,
) -> dict[str, Any] | None:
    """Return supplemental instruction text metrics when text is supplied."""

    if instructions_text is None:
        return None
    return {
        "default_scope": default_scope,
        "length_chars": len(instructions_text),
        "length_bytes": len(instructions_text.encode("utf-8")),
        "must_count": instructions_text.count("MUST"),
        "must_not_count": instructions_text.count("MUST NOT"),
        "before_count": instructions_text.count("BEFORE"),
        "after_count": instructions_text.count("AFTER"),
        "contract_command": INSTRUCTIONS_CONTRACT_COMMAND,
    }


class ReadOnlyAuditStore(MemoryStore):
    """MemoryStore-compatible wrapper that never initializes or writes schema."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        db_path: str | Path = ":memory:",
        config: MemoryConfig | None = None,
        owns_connection: bool = False,
    ) -> None:
        base_config = config or MemoryConfig()
        self.config = replace(base_config, db_path=Path(db_path))
        self.db_path = Path(db_path)
        self._conn = connection
        self._conn.row_factory = sqlite3.Row
        self._owns_connection = owns_connection

    @classmethod
    def open_path(
        cls,
        db_path: str | Path,
        *,
        config: MemoryConfig | None = None,
    ) -> "ReadOnlyAuditStore":
        path = Path(db_path).expanduser()
        uri = f"file:{quote(str(path.resolve()), safe='/')}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return cls(connection, db_path=path, config=config, owns_connection=True)

    @classmethod
    def from_store(cls, store: MemoryStore) -> "ReadOnlyAuditStore":
        return cls(store.connection, db_path=store.db_path, config=store.config)

    def close(self) -> None:
        if self._owns_connection:
            self._conn.close()

    def list_unembedded_nodes(self, *args: Any, **kwargs: Any) -> list[Node]:
        return []

    def list_unchunked_nodes(self, *args: Any, **kwargs: Any) -> list[Node]:
        """Empty for the same reason as ``list_unembedded_nodes``.

        Reporting work here would invite the caller to embed it, and this store
        cannot write. The inherited implementation is also shape-tolerant, but
        an audit should not depend on a pre-v6 snapshot to stay read-only.
        """

        return []

    def update_node(self, *args: Any, **kwargs: Any) -> Node:
        raise RuntimeError("health audit latency probe is read-only")

    def replace_node_chunks(self, *args: Any, **kwargs: Any) -> int:
        raise RuntimeError("health audit store is read-only")

    def delete_node_chunks(self, *args: Any, **kwargs: Any) -> int:
        raise RuntimeError("health audit store is read-only")

    def drop_node_embedding_column(self, *args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("health audit store is read-only")

    def record_access(self, *args: Any, **kwargs: Any) -> Node:
        raise RuntimeError("health audit latency probe is read-only")

    def record_recall_event(self, *args: Any, **kwargs: Any) -> RecallEvent:
        raise RuntimeError("health audit latency probe is read-only")

    def mark_recall_event_feedback(self, *args: Any, **kwargs: Any) -> RecallEvent:
        raise RuntimeError("health audit latency probe is read-only")

    def set_kv(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("health audit store is read-only")

    def create_node(self, *args: Any, **kwargs: Any) -> Node:
        raise RuntimeError("health audit store is read-only")

    def append_trace(self, *args: Any, **kwargs: Any) -> Node:
        raise RuntimeError("health audit store is read-only")

    def append_trace_with_rejected_alternatives(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Node, list[Node]]:
        raise RuntimeError("health audit store is read-only")

    def create_trace(self, *args: Any, **kwargs: Any) -> Node:
        raise RuntimeError("health audit store is read-only")

    def add_correction(self, *args: Any, **kwargs: Any) -> Node:
        raise RuntimeError("health audit store is read-only")

    def soft_delete_node(self, *args: Any, **kwargs: Any) -> Node:
        raise RuntimeError("health audit store is read-only")

    def delete_node(self, *args: Any, **kwargs: Any) -> Node:
        raise RuntimeError("health audit store is read-only")

    def create_connection(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("health audit store is read-only")

    def connect_nodes(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("health audit store is read-only")

    def update_connection(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("health audit store is read-only")

    def delete_connection(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("health audit store is read-only")

    def set_retrieval_weights(self, *args: Any, **kwargs: Any) -> RetrievalWeights:
        raise RuntimeError("health audit store is read-only")

    def update_retrieval_weights(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> RetrievalWeights:
        raise RuntimeError("health audit store is read-only")

    def get_retrieval_weights(self, scope: str) -> RetrievalWeights:
        try:
            return super().get_retrieval_weights(scope)
        except (LookupError, sqlite3.OperationalError):
            family = scope_family(scope)
            config_weights = (
                self.config.retrieval_weights.get(scope)
                or self.config.retrieval_weights.get(family)
                or self.config.retrieval_weights["default"]
            )
            return RetrievalWeights(
                scope=scope,
                bm25=config_weights.bm25,
                vector=config_weights.vector,
                graph=config_weights.graph,
                learning_rate=config_weights.learning_rate,
            )


def open_read_only_store(db_path: str | Path) -> ReadOnlyAuditStore:
    """Open a database file for audit reads using SQLite ``mode=ro``."""

    return ReadOnlyAuditStore.open_path(db_path)


def _coerce_store(source: StoreSource) -> tuple[MemoryStore, bool]:
    if isinstance(source, MemoryStore):
        return source, False
    if isinstance(source, sqlite3.Connection):
        source.row_factory = sqlite3.Row
        return ReadOnlyAuditStore(source), False
    return ReadOnlyAuditStore.open_path(source), True


def _read_only_latency_store(store: MemoryStore) -> MemoryStore:
    if isinstance(store, ReadOnlyAuditStore):
        return store
    return ReadOnlyAuditStore.from_store(store)


def _never_accessed_metric(
    store: MemoryStore,
    extra_clauses: list[str],
    *,
    scope: str | None,
) -> dict[str, Any]:
    clauses = ["decayed = 0", *extra_clauses]
    params: list[Any] = []
    if scope is not None:
        clauses.append("scope = ?")
        params.append(scope)
    row = store.connection.execute(
        f"""
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END)
                   AS never_accessed
        FROM nodes
        WHERE {' AND '.join(clauses)}
        """,
        params,
    ).fetchone()
    total = int(row["total"])
    never_accessed = int(row["never_accessed"] or 0)
    return {
        "total": total,
        "never_accessed": never_accessed,
        "never_accessed_ratio": _ratio(never_accessed, total),
    }


def _candidate_scope_cte() -> str:
    return f"""
    WITH candidate_scopes(scope) AS (
        SELECT DISTINCT scope
        FROM nodes
        WHERE {_scope_leakage_predicate('scope')}
        UNION
        SELECT DISTINCT scope
        FROM recall_events
        WHERE {_scope_leakage_predicate('scope')}
        UNION
        SELECT DISTINCT requested_scope AS scope
        FROM recall_events
        WHERE {_scope_leakage_predicate('requested_scope')}
    )
    """


def _scope_leakage_predicate(column: str) -> str:
    if column not in {"scope", "requested_scope"}:
        raise ValueError(f"unsupported scope column: {column}")
    parts: list[str] = []
    for operator, value in _LEAKAGE_SCOPE_PATTERNS:
        parts.append(f"{column} {operator} {_sql_literal(value)}")
    return " OR ".join(parts)


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def iter_metric_paths(data: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    """Yield leaf metric paths for lightweight smoke checks."""

    yield from _iter_metric_paths("", data)


def _iter_metric_paths(prefix: str, value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_metric_paths(child_prefix, value[key])
    elif isinstance(value, list):
        yield prefix, value
    else:
        yield prefix, value
