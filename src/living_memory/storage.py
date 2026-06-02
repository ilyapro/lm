"""SQLite persistence for the Living Memory uniform node store."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
import hashlib
import json
import re
import secrets
import sqlite3
import time
from datetime import UTC, datetime

from living_memory.config import MemoryConfig, RetrievalWeightConfig
from living_memory.embeddings import cosine_similarity, tokenize
from living_memory.models import (
    CONNECTION_TYPES,
    NODE_LEVELS,
    REJECTED_ALTERNATIVE_KIND,
    Connection,
    ConnectionType,
    Node,
    NodeLevel,
    RecallEvent,
    RetrievalWeights,
)
from living_memory.phase import PhaseManager
from living_memory.scope import normalize_scope

SCHEMA_VERSION = 3
_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_RECALL_TEXT_SIMILARITY_THRESHOLD = 0.55
_DUPLICATE_CONTENT_DECAY_REASON = "duplicate_content"
_DUPLICATE_CONTENT_KIND = "duplicate_content"
_WEIGHT_EPSILON = 1e-12
_CONTEXT_LOOKUP_FIELDS = frozenset(
    {
        "task_pattern",
        "procedure_id",
        "lesson_kind",
        "scope",
        "task",
    }
)


def _content_fingerprint(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class MemoryStore:
    """Stable CRUD API over one SQLite file.

    Raw traces are append-only: callers can add stats, provenance, or decay markers,
    but trace content is never updated in place.
    """

    def __init__(
        self,
        config: MemoryConfig | str | Path | None = None,
        base_config: MemoryConfig | None = None,
    ) -> None:
        if base_config is not None and not isinstance(base_config, MemoryConfig):
            raise TypeError("base_config must be a MemoryConfig")
        if base_config is not None:
            if config is None:
                self.config = base_config
            elif isinstance(config, MemoryConfig):
                self.config = config
            else:
                self.config = replace(base_config, db_path=Path(config))
        elif config is None:
            self.config = MemoryConfig()
        elif isinstance(config, MemoryConfig):
            self.config = config
        else:
            self.config = MemoryConfig(db_path=Path(config))

        self.db_path = Path(self.config.db_path)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        # Wait for the write lock instead of erroring when another process holds
        # it — the server can run as two processes (HTTP on loopback + HTTPS on
        # all interfaces) against this one WAL database; without this a
        # concurrent writer would get SQLITE_BUSY.
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._initialize_schema()
        self._seed_retrieval_weights()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def get_kv(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM kv WHERE key = ?",
            (str(key),),
        ).fetchone()
        return str(row["value"]) if row else None

    def set_kv(self, key: str, value: str) -> None:
        now = _utc_now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO kv (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (str(key), str(value), now),
            )

    def get_last_decay_sweep_at(self) -> str | None:
        return self.get_kv("last_decay_sweep_at")

    def set_last_decay_sweep_at(self, timestamp: str) -> None:
        self.set_kv("last_decay_sweep_at", timestamp)

    def create_node(
        self,
        *,
        level: NodeLevel,
        content: str,
        context: Mapping[str, Any] | None = None,
        embedding: Iterable[float] | None = None,
        stats: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        node_id: str | None = None,
        timestamp: str | None = None,
    ) -> Node:
        with self._conn:
            new_id = self._insert_node(
                level=level,
                content=content,
                context=context,
                embedding=embedding,
                stats=stats,
                provenance=provenance,
                node_id=node_id,
                timestamp=timestamp,
            )
        return self.get_node(new_id)  # type: ignore[return-value]

    def _insert_node(
        self,
        *,
        level: NodeLevel,
        content: str,
        context: Mapping[str, Any] | None = None,
        embedding: Iterable[float] | None = None,
        stats: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        node_id: str | None = None,
        timestamp: str | None = None,
    ) -> str:
        self._validate_level(level)
        if not content:
            raise ValueError("content must be non-empty")

        context_data = dict(context or {})
        stats_data = dict(stats or {})
        provenance_data = dict(provenance or {})
        now = _utc_now()
        node_timestamp = str(timestamp or context_data.get("timestamp") or now)
        scope = normalize_scope(str(context_data.get("scope") or self.config.default_scope))
        agent = _optional_str(context_data.get("agent"))
        task = _optional_str(context_data.get("task"))
        context_data["scope"] = scope
        context_data["timestamp"] = node_timestamp
        if agent is not None:
            context_data.setdefault("agent", agent)
        if task is not None:
            context_data.setdefault("task", task)

        unique_agents = int(stats_data.get("unique_agents", 1 if agent else 0))
        confidence = float(stats_data.get("confidence", 0.5 if unique_agents <= 1 else 0.75))
        if unique_agents <= 1:
            confidence = min(confidence, 0.5)

        source_traces = _as_list(provenance_data.pop("source_traces", []))
        corrections = _as_list(provenance_data.pop("corrections", []))
        new_id = node_id or new_ulid()
        fingerprint = _content_fingerprint(content)

        duplicate_ids: list[str] = []
        if level == "trace":
            duplicate_ids = [
                str(row["id"])
                for row in self._conn.execute(
                    """
                    SELECT id FROM nodes
                    WHERE level = 'trace'
                      AND scope = ?
                      AND content_fingerprint = ?
                      AND decayed = 0
                    """,
                    (scope, fingerprint),
                ).fetchall()
            ]

        self._conn.execute(
            """
            INSERT INTO nodes (
                id, level, content, content_fingerprint, embedding, scope, agent, task, context,
                timestamp, decayed, decay_reason, access_count, last_accessed,
                usefulness_score, confidence, unique_agents, temporal_hint,
                source_traces, corrections, provenance, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                level,
                content,
                fingerprint,
                _json_dumps(list(embedding)) if embedding is not None else None,
                scope,
                agent,
                task,
                _json_dumps(context_data),
                node_timestamp,
                int(stats_data.get("access_count", 0)),
                _optional_str(stats_data.get("last_accessed")),
                float(stats_data.get("usefulness_score", 0.0)),
                confidence,
                unique_agents,
                _optional_str(stats_data.get("temporal_hint")),
                _json_dumps(source_traces),
                _json_dumps(corrections),
                _json_dumps(provenance_data),
                now,
                now,
            ),
        )

        for old_id in duplicate_ids:
            self._insert_connection(
                new_id,
                old_id,
                "supersedes",
                weight=1.0,
                metadata={
                    "kind": _DUPLICATE_CONTENT_KIND,
                    "fingerprint": fingerprint,
                },
            )
            self._conn.execute(
                """
                UPDATE nodes
                SET decayed = 1, decay_reason = ?, updated_at = ?
                WHERE id = ?
                """,
                (_DUPLICATE_CONTENT_DECAY_REASON, now, old_id),
            )

        return new_id

    def append_trace(
        self,
        content: str,
        context: Mapping[str, Any] | None = None,
        *,
        feedback: Mapping[str, Any] | None = None,
    ) -> Node:
        stats = dict(feedback or {})
        return self.create_node(level="trace", content=content, context=context, stats=stats)

    def append_trace_with_rejected_alternatives(
        self,
        content: str,
        context: Mapping[str, Any] | None = None,
        *,
        feedback: Mapping[str, Any] | None = None,
        alternatives_considered: Iterable[Mapping[str, Any]] | None = None,
    ) -> tuple[Node, list[Node]]:
        """Append one primary trace and rejected-alternative traces atomically."""

        alternatives = _normalize_rejected_alternatives(alternatives_considered)
        if not alternatives:
            return self.append_trace(content, context, feedback=feedback), []

        rejected_ids: list[str] = []
        with self._conn:
            primary_id = self._insert_node(
                level="trace",
                content=content,
                context=context,
                stats=dict(feedback or {}),
            )
            primary = self.get_node(primary_id)
            if primary is None:
                raise RuntimeError("primary trace insert failed")

            base_context = dict(primary.context)
            for alternative in alternatives:
                approach = alternative["approach"]
                reason = alternative["rejected_because"]
                rejected_context = dict(base_context)
                rejected_context["is_rejected_alternative"] = True
                rejected_context["rejected_for"] = primary.id
                rejected_context["approach"] = approach
                rejected_stats = {
                    "confidence": max(0.0, min(1.0, primary.confidence * 0.5)),
                    "unique_agents": primary.unique_agents,
                    "usefulness_score": min(0.0, primary.usefulness_score),
                }
                rejected_id = self._insert_node(
                    level="trace",
                    content=_rejected_alternative_content(approach, reason),
                    context=rejected_context,
                    stats=rejected_stats,
                )
                self._insert_connection(
                    rejected_id,
                    primary.id,
                    "contradicts",
                    weight=1.0,
                    metadata={
                        "kind": REJECTED_ALTERNATIVE_KIND,
                        "reason": reason,
                        "approach": approach,
                    },
                )
                rejected_ids.append(rejected_id)

        primary = self.get_node(primary_id)
        if primary is None:
            raise RuntimeError("primary trace insert failed")
        rejected_nodes = [node for node_id in rejected_ids if (node := self.get_node(node_id)) is not None]
        return primary, rejected_nodes

    def create_trace(
        self,
        content: str,
        context: Mapping[str, Any] | None = None,
        *,
        feedback: Mapping[str, Any] | None = None,
    ) -> Node:
        return self.append_trace(content, context, feedback=feedback)

    def get_node(self, node_id: str) -> Node | None:
        row = self._conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return _node_from_row(row) if row else None

    def get_nodes(self, node_ids: Iterable[str]) -> dict[str, Node]:
        ids = list(set(node_ids))
        if not ids:
            return {}
        nodes: dict[str, Node] = {}
        for i in range(0, len(ids), 999):
            chunk = ids[i : i + 999]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT * FROM nodes WHERE id IN ({placeholders})", chunk
            ).fetchall()
            for row in rows:
                node = _node_from_row(row)
                nodes[node.id] = node
        return nodes

    def list_nodes(
        self,
        *,
        level: NodeLevel | None = None,
        scope: str | None = None,
        include_decayed: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Node]:
        clauses: list[str] = []
        params: list[Any] = []
        if level is not None:
            self._validate_level(level)
            clauses.append("level = ?")
            params.append(level)
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        if not include_decayed:
            clauses.append("decayed = 0")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([int(limit), int(offset)])
        rows = self._conn.execute(
            f"SELECT * FROM nodes {where} ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [_node_from_row(row) for row in rows]

    def list_nodes_by_context(
        self,
        *,
        scope: str,
        context_filters: dict[str, str],
        level: NodeLevel | None = "trace",
        include_decayed: bool = False,
        limit: int = 1000,
    ) -> list[Node]:
        clauses: list[str] = ["scope = ?"]
        params: list[Any] = [normalize_scope(scope)]
        if level is not None:
            self._validate_level(level)
            clauses.append("level = ?")
            params.append(level)
        if not include_decayed:
            clauses.append("decayed = 0")

        for field, value in context_filters.items():
            if field not in _CONTEXT_LOOKUP_FIELDS:
                raise ValueError(f"unsupported context lookup field: {field}")
            clauses.append(f"JSON_EXTRACT(context, '$.{field}') = ?")
            params.append(str(value))

        params.append(int(limit))
        rows = self._conn.execute(
            f"""
            SELECT *
            FROM nodes
            WHERE {' AND '.join(clauses)}
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [_node_from_row(row) for row in rows]

    def update_node(
        self,
        node_id: str,
        *,
        content: str | None = None,
        context: Mapping[str, Any] | None = None,
        embedding: Iterable[float] | None = None,
        stats: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> Node:
        existing = self.get_node(node_id)
        if existing is None:
            raise KeyError(node_id)
        if existing.level == "trace" and content is not None and content != existing.content:
            raise ValueError("trace content is append-only and cannot be updated")

        context_data = existing.context if context is None else dict(context)
        context_data.setdefault("scope", existing.scope)
        context_data.setdefault("timestamp", existing.timestamp)
        stats_data = existing.stats
        if stats:
            stats_data.update(stats)
        provenance_data = existing.provenance if provenance is None else dict(provenance)
        source_traces = _as_list(provenance_data.pop("source_traces", existing.source_traces))
        corrections = _as_list(provenance_data.pop("corrections", existing.corrections))
        new_content = existing.content if content is None else content
        now = _utc_now()
        unique_agents = int(stats_data.get("unique_agents", existing.unique_agents))
        confidence = float(stats_data.get("confidence", existing.confidence))
        if unique_agents <= 1:
            confidence = min(confidence, 0.5)

        with self._conn:
            self._conn.execute(
                """
                UPDATE nodes
                SET content = ?, embedding = ?, scope = ?, agent = ?, task = ?, context = ?,
                    access_count = ?, last_accessed = ?, usefulness_score = ?,
                    confidence = ?, unique_agents = ?, temporal_hint = ?,
                    source_traces = ?, corrections = ?, provenance = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    new_content,
                    _json_dumps(list(embedding)) if embedding is not None else _json_dumps(existing.embedding)
                    if existing.embedding is not None
                    else None,
                    str(context_data.get("scope") or existing.scope),
                    _optional_str(context_data.get("agent", existing.agent)),
                    _optional_str(context_data.get("task", existing.task)),
                    _json_dumps(context_data),
                    int(stats_data.get("access_count", existing.access_count)),
                    _optional_str(stats_data.get("last_accessed")),
                    float(stats_data.get("usefulness_score", existing.usefulness_score)),
                    confidence,
                    unique_agents,
                    _optional_str(stats_data.get("temporal_hint")),
                    _json_dumps(source_traces),
                    _json_dumps(corrections),
                    _json_dumps(provenance_data),
                    now,
                    node_id,
                ),
            )
        return self.get_node(node_id)  # type: ignore[return-value]

    def add_correction(self, node_id: str, *, old: str, new: str, by: str) -> Node:
        existing = self.get_node(node_id)
        if existing is None:
            raise KeyError(node_id)
        corrections = list(existing.corrections)
        corrections.append({"timestamp": _utc_now(), "old": old, "new": new, "by": by})
        provenance = dict(existing.provenance)
        provenance["corrections"] = corrections
        provenance["source_traces"] = existing.source_traces
        return self.update_node(node_id, provenance=provenance)

    def soft_delete_node(self, node_id: str, reason: str | None = None) -> Node:
        existing = self.get_node(node_id)
        if existing is None:
            raise KeyError(node_id)
        now = _utc_now()
        with self._conn:
            self._conn.execute(
                """
                UPDATE nodes
                SET decayed = 1, decay_reason = ?, updated_at = ?
                WHERE id = ?
                """,
                (reason, now, node_id),
            )
        return self.get_node(node_id)  # type: ignore[return-value]

    def delete_node(self, node_id: str, reason: str | None = None) -> Node:
        return self.soft_delete_node(node_id, reason)

    def create_connection(
        self,
        source_id: str,
        target_id: str,
        relation_type: ConnectionType,
        *,
        weight: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> Connection:
        self._validate_connection_type(relation_type)
        if self.get_node(source_id) is None:
            raise KeyError(source_id)
        if self.get_node(target_id) is None:
            raise KeyError(target_id)

        with self._conn:
            self._insert_connection(
                source_id,
                target_id,
                relation_type,
                weight=weight,
                metadata=metadata,
            )
        row = self._conn.execute(
            """
            SELECT * FROM connections
            WHERE source_id = ? AND target_id = ? AND type = ?
            """,
            (source_id, target_id, relation_type),
        ).fetchone()
        return _connection_from_row(row)

    def _insert_connection(
        self,
        source_id: str,
        target_id: str,
        relation_type: ConnectionType,
        *,
        weight: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        self._validate_connection_type(relation_type)
        now = _utc_now()
        connection_id = new_ulid()
        self._conn.execute(
            """
            INSERT INTO connections (
                id, source_id, target_id, type, weight, metadata, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id, type) DO UPDATE SET
                weight = excluded.weight,
                metadata = excluded.metadata,
                updated_at = excluded.updated_at
            """,
            (
                connection_id,
                source_id,
                target_id,
                relation_type,
                float(weight),
                _json_dumps(dict(metadata or {})),
                now,
                now,
            ),
        )
        return connection_id

    def connect_nodes(
        self,
        source_id: str,
        target_id: str,
        relation_type: ConnectionType,
        *,
        weight: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> Connection:
        return self.create_connection(
            source_id,
            target_id,
            relation_type,
            weight=weight,
            metadata=metadata,
        )

    def get_connection(self, connection_id: str) -> Connection | None:
        row = self._conn.execute(
            "SELECT * FROM connections WHERE id = ?",
            (connection_id,),
        ).fetchone()
        return _connection_from_row(row) if row else None

    def update_connection(
        self,
        connection_id: str,
        *,
        weight: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Connection:
        existing = self.get_connection(connection_id)
        if existing is None:
            raise KeyError(connection_id)

        now = _utc_now()
        with self._conn:
            self._conn.execute(
                """
                UPDATE connections
                SET weight = ?, metadata = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    existing.weight if weight is None else float(weight),
                    _json_dumps(existing.metadata if metadata is None else dict(metadata)),
                    now,
                    connection_id,
                ),
            )
        return self.get_connection(connection_id)  # type: ignore[return-value]

    def delete_connection(self, connection_id: str) -> None:
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM connections WHERE id = ?",
                (connection_id,),
            )
        if cursor.rowcount == 0:
            raise KeyError(connection_id)

    def list_connections(
        self,
        *,
        node_id: str | None = None,
        source_id: str | None = None,
        target_id: str | None = None,
        relation_type: ConnectionType | None = None,
    ) -> list[Connection]:
        clauses: list[str] = []
        params: list[Any] = []
        if node_id is not None:
            clauses.append("(source_id = ? OR target_id = ?)")
            params.extend([node_id, node_id])
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        if target_id is not None:
            clauses.append("target_id = ?")
            params.append(target_id)
        if relation_type is not None:
            self._validate_connection_type(relation_type)
            clauses.append("type = ?")
            params.append(relation_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM connections {where} ORDER BY weight DESC, created_at DESC",
            params,
        ).fetchall()
        return [_connection_from_row(row) for row in rows]

    def list_connections_for_nodes(self, node_ids: Iterable[str]) -> dict[str, list[Connection]]:
        ids = list(set(node_ids))
        if not ids:
            return {}
        
        connections = []
        for i in range(0, len(ids), 400):
            chunk = ids[i : i + 400]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT * FROM connections WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders}) ORDER BY weight DESC, created_at DESC",
                chunk + chunk
            ).fetchall()
            connections.extend([_connection_from_row(row) for row in rows])
            
        conns_by_node: dict[str, list[Connection]] = {node_id: [] for node_id in ids}
        seen = set()
        for conn in connections:
            if conn.id in seen:
                continue
            seen.add(conn.id)
            if conn.source_id in conns_by_node:
                conns_by_node[conn.source_id].append(conn)
            if conn.target_id in conns_by_node and conn.target_id != conn.source_id:
                conns_by_node[conn.target_id].append(conn)
        return conns_by_node

    def record_access(self, node_id: str) -> Node:
        if self.get_node(node_id) is None:
            raise KeyError(node_id)
        now = _utc_now()
        with self._conn:
            self._conn.execute(
                """
                UPDATE nodes
                SET access_count = access_count + 1, last_accessed = ?, updated_at = ?
                WHERE id = ?
                """,
                (now, now, node_id),
            )
        return self.get_node(node_id)  # type: ignore[return-value]

    def record_recall_event(
        self,
        *,
        query: str,
        scope: str,
        requested_scope: str | None = None,
        resolved_scopes: Iterable[str] | None = None,
        ambient_context: Mapping[str, Any] | None = None,
        depth: str | int | None = None,
        max_results: int = 10,
        results: Iterable[Mapping[str, Any]] | None = None,
    ) -> RecallEvent:
        """Persist one recall interaction for later feedback/provenance."""

        normalized_results = [dict(item) for item in results or []]
        event_id = new_ulid()
        now = _utc_now()
        ambient = dict(ambient_context or {})
        agent = _optional_str(ambient.get("agent"))
        task = _optional_str(ambient.get("task"))
        session_id = _optional_str(ambient.get("session_id") or ambient.get("session"))
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO recall_events (
                    id, query, scope, requested_scope, resolved_scopes, ambient_context,
                    depth, max_results, results, agent, task, session_id,
                    feedback_applied, feedback_trace_id, feedback_applied_at, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, ?)
                """,
                (
                    event_id,
                    query,
                    scope,
                    requested_scope or scope,
                    _json_dumps(list(resolved_scopes or (scope,))),
                    _json_dumps(ambient),
                    None if depth is None else str(depth),
                    int(max_results),
                    _json_dumps(normalized_results),
                    agent,
                    task,
                    session_id,
                    now,
                ),
            )
        return self.get_recall_event(event_id)  # type: ignore[return-value]

    def get_recall_event(self, event_id: str) -> RecallEvent | None:
        row = self._conn.execute(
            "SELECT * FROM recall_events WHERE id = ?",
            (event_id,),
        ).fetchone()
        return _recall_event_from_row(row) if row else None

    def list_recall_events(
        self,
        *,
        scope: str | None = None,
        pending_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RecallEvent]:
        clauses: list[str] = []
        params: list[Any] = []
        if scope is not None:
            clauses.append("(scope = ? OR requested_scope = ?)")
            params.extend([scope, scope])
        if pending_only:
            clauses.append("feedback_applied = 0")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([int(limit), int(offset)])
        rows = self._conn.execute(
            f"SELECT * FROM recall_events {where} ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [_recall_event_from_row(row) for row in rows]

    def pending_recall_events(
        self,
        *,
        scope: str,
        context: Mapping[str, Any] | None = None,
        content: str | None = None,
        limit: int = 1,
    ) -> list[RecallEvent]:
        """Return recent unconsumed recalls compatible with a new ingest trace."""

        context_data = dict(context or {})
        rows = self._conn.execute(
            """
            SELECT *
            FROM recall_events
            WHERE feedback_applied = 0
              AND (scope = ? OR requested_scope = ? OR resolved_scopes LIKE ?)
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (scope, scope, f"%{_json_dumps(scope)}%", max(1, int(limit) * 20)),
        ).fetchall()
        events: list[RecallEvent] = []
        weak_fallbacks = 0
        for row in rows:
            event = _recall_event_from_row(row)
            match_strength = _recall_event_match_strength(event, scope, context_data, content)
            if match_strength is None:
                continue
            if match_strength == "weak":
                if weak_fallbacks >= 1:
                    continue
                weak_fallbacks += 1
            events.append(event)
            if len(events) >= limit:
                break
        return events

    def mark_recall_event_feedback(self, event_id: str, trace_id: str) -> RecallEvent:
        if self.get_node(trace_id) is None:
            raise KeyError(trace_id)
        if self.get_recall_event(event_id) is None:
            raise KeyError(event_id)
        now = _utc_now()
        with self._conn:
            self._conn.execute(
                """
                UPDATE recall_events
                SET feedback_applied = 1,
                    feedback_trace_id = ?,
                    feedback_applied_at = ?
                WHERE id = ?
                """,
                (trace_id, now, event_id),
            )
        return self.get_recall_event(event_id)  # type: ignore[return-value]

    def search_content(
        self,
        query: str,
        *,
        level: NodeLevel | None = None,
        scope: str | None = None,
        limit: int = 10,
    ) -> list[tuple[Node, float]]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []
        if level is not None:
            self._validate_level(level)

        clauses = ["nodes_fts MATCH ?", "n.decayed = 0"]
        params: list[Any] = [fts_query]
        if level is not None:
            clauses.append("n.level = ?")
            params.append(level)
        if scope is not None:
            clauses.append("n.scope = ?")
            params.append(scope)
        params.append(int(limit))
        rows = self._conn.execute(
            f"""
            SELECT n.*, bm25(nodes_fts) AS score
            FROM nodes_fts
            JOIN nodes n ON n.rowid = nodes_fts.rowid
            WHERE {' AND '.join(clauses)}
            ORDER BY score ASC, n.confidence DESC, n.usefulness_score DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [(_node_from_row(row), float(row["score"])) for row in rows]

    def iter_embedding_rows(
        self,
        *,
        scope: str | None = None,
        level: NodeLevel | None = None,
        include_decayed: bool = False,
    ) -> Iterator[tuple[str, str]]:
        """Yield ``(node_id, embedding_json)`` for embedded nodes only.

        Reads just the two columns required for vector similarity scans, which
        skips parsing of the much larger ``context``/``provenance`` JSON columns
        and avoids constructing full Node dataclass instances. Use this for
        bulk cosine scans and fall back to ``get_node`` for the top matches.
        """

        clauses: list[str] = ["embedding IS NOT NULL"]
        params: list[Any] = []
        if level is not None:
            self._validate_level(level)
            clauses.append("level = ?")
            params.append(level)
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        if not include_decayed:
            clauses.append("decayed = 0")
        sql = f"SELECT id, embedding FROM nodes WHERE {' AND '.join(clauses)}"
        cur = self._conn.execute(sql, params)
        for row in cur:
            yield str(row["id"]), str(row["embedding"])

    def list_unembedded_nodes(
        self,
        *,
        scope: str | None = None,
        level: NodeLevel | None = None,
        limit: int = 200,
    ) -> list[Node]:
        """Return active nodes with no stored embedding (for lazy backfill)."""

        clauses: list[str] = ["embedding IS NULL", "decayed = 0"]
        params: list[Any] = []
        if level is not None:
            self._validate_level(level)
            clauses.append("level = ?")
            params.append(level)
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        params.append(int(limit))
        rows = self._conn.execute(
            f"SELECT * FROM nodes WHERE {' AND '.join(clauses)}"
            " ORDER BY timestamp DESC, id DESC LIMIT ?",
            params,
        ).fetchall()
        return [_node_from_row(row) for row in rows]

    def find_similar_by_embedding(
        self,
        embedding: list[float],
        *,
        level: NodeLevel = "concept",
        exclude_scope: str | None = None,
        scope_prefix: str | None = None,
        scope: str | None = None,
        threshold: float = 0.7,
        limit: int = 50,
    ) -> list[tuple[Node, float]]:
        """Scan active embedded nodes and return cosine matches above a threshold."""

        self._validate_level(level)
        if not embedding or limit <= 0:
            return []

        clauses = ["level = ?", "decayed = 0", "embedding IS NOT NULL"]
        params: list[Any] = [level]
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        if exclude_scope is not None:
            clauses.append("scope != ?")
            params.append(exclude_scope)
        if scope_prefix is not None:
            upper_bound = _prefix_upper_bound(scope_prefix)
            clauses.append("scope >= ?")
            params.append(scope_prefix)
            if upper_bound is not None:
                clauses.append("scope < ?")
                params.append(upper_bound)

        rows = self._conn.execute(
            f"""
            SELECT *
            FROM nodes
            WHERE {' AND '.join(clauses)}
            ORDER BY scope ASC, confidence DESC, usefulness_score DESC, id ASC
            """,
            params,
        )

        matches: list[tuple[Node, float]] = []
        for row in rows:
            node = _node_from_row(row)
            score = cosine_similarity(embedding, node.embedding)
            if score >= threshold:
                matches.append((node, score))

        matches.sort(
            key=lambda item: (
                -item[1],
                -item[0].confidence,
                -item[0].usefulness_score,
                item[0].scope,
                item[0].id,
            )
        )
        return matches[: int(limit)]

    def trace_count(self, *, include_decayed: bool = True) -> int:
        if include_decayed:
            row = self._conn.execute("SELECT COUNT(*) AS count FROM nodes WHERE level = 'trace'").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS count FROM nodes WHERE level = 'trace' AND decayed = 0"
            ).fetchone()
        return int(row["count"])

    def count_traces(self, *, include_decayed: bool = True) -> int:
        return self.trace_count(include_decayed=include_decayed)

    def detect_phase(self) -> Any:
        return PhaseManager(self.config.phase_thresholds).detect(self.trace_count(include_decayed=True))

    def get_retrieval_weights(self, scope: str) -> RetrievalWeights:
        candidates = [scope, _scope_family(scope), "default"]
        for candidate in candidates:
            row = self._conn.execute(
                "SELECT * FROM retrieval_weights WHERE scope = ?",
                (candidate,),
            ).fetchone()
            if row is not None:
                return _weights_from_row(row)
        raise LookupError("default retrieval weights are missing")

    def set_retrieval_weights(
        self,
        scope: str,
        *,
        bm25: float,
        vector: float,
        graph: float,
        learning_rate: float = 0.05,
    ) -> RetrievalWeights:
        now = _utc_now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO retrieval_weights (scope, bm25, vector, graph, learning_rate, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope) DO UPDATE SET
                    bm25 = excluded.bm25,
                    vector = excluded.vector,
                    graph = excluded.graph,
                    learning_rate = excluded.learning_rate,
                    updated_at = excluded.updated_at
                """,
                (scope, float(bm25), float(vector), float(graph), float(learning_rate), now),
            )
        return self.get_retrieval_weights(scope)

    def update_retrieval_weights(
        self,
        scope: str,
        *,
        bm25_signal: float = 0.0,
        vector_signal: float = 0.0,
        graph_signal: float = 0.0,
    ) -> RetrievalWeights:
        current = self.get_retrieval_weights(scope)
        updated = RetrievalWeights(
            scope=scope,
            bm25=max(0.0, current.bm25 + current.learning_rate * bm25_signal),
            vector=max(0.0, current.vector + current.learning_rate * vector_signal),
            graph=max(0.0, current.graph + current.learning_rate * graph_signal),
            learning_rate=current.learning_rate,
            updated_at=current.updated_at,
        ).normalized()
        updated = self.apply_retrieval_weight_floors(scope, updated)
        return self.set_retrieval_weights(
            scope,
            bm25=updated.bm25,
            vector=updated.vector,
            graph=updated.graph,
            learning_rate=updated.learning_rate,
        )

    def apply_retrieval_weight_floors(
        self,
        scope: str,
        weights: RetrievalWeights,
    ) -> RetrievalWeights:
        """Return normalized weights with active scope-family floors applied."""

        normalized = weights.normalized()
        floors = self.config.retrieval_policy_floors.get(_scope_family(scope))
        if floors is None:
            return normalized

        vector_min = floors.vector_min if self._has_vector_evidence(scope) else 0.0
        graph_min = floors.graph_min if self._has_graph_evidence(scope) else 0.0
        # bm25 is a lexical channel that is always available, so its floor is
        # enforced unconditionally (no evidence gate). Without it, winner-take-all
        # feedback (feedback._method_signals) drives bm25 to 0 and exact-keyword /
        # identifier recall stops contributing to ranking.
        bm25_min = floors.bm25_min
        if vector_min <= 0.0 and graph_min <= 0.0 and bm25_min <= 0.0:
            return normalized

        values = {
            "bm25": normalized.bm25,
            "vector": normalized.vector,
            "graph": normalized.graph,
        }
        minimum = {"bm25": bm25_min, "vector": vector_min, "graph": graph_min}

        for target in ("vector", "graph", "bm25"):
            self._lift_retrieval_weight(values, minimum, target)

        bm25_surplus = max(0.0, values["bm25"] - floors.bm25_max)
        if bm25_surplus > _WEIGHT_EPSILON:
            for target in ("vector", "graph"):
                if minimum[target] <= 0.0:
                    continue
                values["bm25"] -= bm25_surplus
                values[target] += bm25_surplus
                break

        floored = RetrievalWeights(
            scope=scope,
            bm25=max(0.0, values["bm25"]),
            vector=max(0.0, values["vector"]),
            graph=max(0.0, values["graph"]),
            learning_rate=normalized.learning_rate,
            updated_at=normalized.updated_at,
        ).normalized()
        return floored

    @staticmethod
    def _lift_retrieval_weight(
        values: dict[str, float],
        minimum: dict[str, float],
        target: str,
    ) -> None:
        deficit = max(0.0, minimum[target] - values[target])
        if deficit <= _WEIGHT_EPSILON:
            return
        for donor in ("bm25", "graph", "vector"):
            if donor == target or deficit <= _WEIGHT_EPSILON:
                continue
            available = max(0.0, values[donor] - minimum[donor])
            taken = min(deficit, available)
            values[donor] -= taken
            values[target] += taken
            deficit -= taken

    def _has_vector_evidence(self, scope: str) -> bool:
        row = self._conn.execute(
            """
            SELECT 1
            FROM nodes
            WHERE scope = ? AND decayed = 0 AND embedding IS NOT NULL
            LIMIT 1
            """,
            (scope,),
        ).fetchone()
        return row is not None

    def _has_graph_evidence(self, scope: str) -> bool:
        row = self._conn.execute(
            """
            SELECT 1
            FROM connections c
            JOIN nodes source ON source.id = c.source_id
            JOIN nodes target ON target.id = c.target_id
            WHERE source.decayed = 0
              AND target.decayed = 0
              AND (source.scope = ? OR target.scope = ?)
            LIMIT 1
            """,
            (scope, scope),
        ).fetchone()
        return row is not None

    def _initialize_schema(self) -> None:
        with self._conn:
            self._migrate_pre_v3_schema()
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    level TEXT NOT NULL CHECK (level IN ('trace', 'concept', 'schema')),
                    content TEXT NOT NULL,
                    content_fingerprint TEXT,
                    embedding TEXT,
                    scope TEXT NOT NULL DEFAULT 'global',
                    agent TEXT,
                    task TEXT,
                    context TEXT NOT NULL DEFAULT '{}',
                    timestamp TEXT NOT NULL,
                    decayed INTEGER NOT NULL DEFAULT 0 CHECK (decayed IN (0, 1)),
                    decay_reason TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed TEXT,
                    usefulness_score REAL NOT NULL DEFAULT 0.0,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    unique_agents INTEGER NOT NULL DEFAULT 0,
                    temporal_hint TEXT,
                    source_traces TEXT NOT NULL DEFAULT '[]',
                    corrections TEXT NOT NULL DEFAULT '[]',
                    provenance TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                    node_id UNINDEXED,
                    content,
                    level UNINDEXED,
                    scope UNINDEXED,
                    tokenize = 'unicode61'
                );

                CREATE TRIGGER IF NOT EXISTS nodes_fts_insert
                AFTER INSERT ON nodes
                BEGIN
                    INSERT INTO nodes_fts(rowid, node_id, content, level, scope)
                    VALUES (new.rowid, new.id, new.content, new.level, new.scope);
                END;

                CREATE TRIGGER IF NOT EXISTS nodes_fts_delete
                AFTER DELETE ON nodes
                BEGIN
                    DELETE FROM nodes_fts WHERE rowid = old.rowid;
                END;

                CREATE TRIGGER IF NOT EXISTS nodes_fts_update
                AFTER UPDATE OF content, level, scope ON nodes
                BEGIN
                    DELETE FROM nodes_fts WHERE rowid = old.rowid;
                    INSERT INTO nodes_fts(rowid, node_id, content, level, scope)
                    VALUES (new.rowid, new.id, new.content, new.level, new.scope);
                END;

                CREATE TABLE IF NOT EXISTS connections (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES nodes(id),
                    target_id TEXT NOT NULL REFERENCES nodes(id),
                    type TEXT NOT NULL CHECK (
                        type IN ('related', 'caused', 'contradicts', 'supersedes', 'requires')
                    ),
                    weight REAL NOT NULL DEFAULT 1.0,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(source_id, target_id, type)
                );

                CREATE TABLE IF NOT EXISTS recall_events (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    scope TEXT NOT NULL DEFAULT 'global',
                    requested_scope TEXT NOT NULL DEFAULT 'global',
                    resolved_scopes TEXT NOT NULL DEFAULT '[]',
                    ambient_context TEXT NOT NULL DEFAULT '{}',
                    depth TEXT,
                    max_results INTEGER NOT NULL DEFAULT 10,
                    results TEXT NOT NULL DEFAULT '[]',
                    agent TEXT,
                    task TEXT,
                    session_id TEXT,
                    feedback_applied INTEGER NOT NULL DEFAULT 0 CHECK (feedback_applied IN (0, 1)),
                    feedback_trace_id TEXT REFERENCES nodes(id),
                    feedback_applied_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS retrieval_weights (
                    scope TEXT PRIMARY KEY,
                    bm25 REAL NOT NULL,
                    vector REAL NOT NULL,
                    graph REAL NOT NULL,
                    learning_rate REAL NOT NULL DEFAULT 0.05,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_nodes_trace_time
                    ON nodes(timestamp DESC)
                    WHERE level = 'trace' AND decayed = 0;

                CREATE INDEX IF NOT EXISTS idx_nodes_trace_scope_time
                    ON nodes(scope, timestamp DESC)
                    WHERE level = 'trace' AND decayed = 0;

                CREATE INDEX IF NOT EXISTS idx_nodes_concepts_content
                    ON nodes(content)
                    WHERE level IN ('concept', 'schema') AND decayed = 0;

                CREATE INDEX IF NOT EXISTS idx_nodes_concepts_scope_content
                    ON nodes(scope, content)
                    WHERE level IN ('concept', 'schema') AND decayed = 0;

                CREATE INDEX IF NOT EXISTS idx_nodes_embedded_active_scope
                    ON nodes(level, scope)
                    WHERE decayed = 0 AND embedding IS NOT NULL;

                CREATE INDEX IF NOT EXISTS idx_nodes_level_scope_active
                    ON nodes(level, scope)
                    WHERE decayed = 0;

                CREATE INDEX IF NOT EXISTS idx_nodes_dedup
                    ON nodes(level, scope, content_fingerprint)
                    WHERE decayed = 0 AND content_fingerprint IS NOT NULL;

                CREATE INDEX IF NOT EXISTS idx_connections_source_type
                    ON connections(source_id, type);

                CREATE INDEX IF NOT EXISTS idx_connections_target_type
                    ON connections(target_id, type);

                CREATE INDEX IF NOT EXISTS idx_recall_events_scope_pending_created
                    ON recall_events(scope, feedback_applied, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_recall_events_session_pending_created
                    ON recall_events(session_id, feedback_applied, created_at DESC);
                """
            )
            self._backfill_missing_content_fingerprints()
            self._conn.execute(
                """
                INSERT INTO metadata (key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )

    def _migrate_pre_v3_schema(self) -> None:
        """Add content_fingerprint to nodes for DBs created at schema_version <= 2."""

        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='nodes'"
        ).fetchone()
        if row is None:
            return
        columns = {
            r["name"] for r in self._conn.execute("PRAGMA table_info(nodes)").fetchall()
        }
        if "content_fingerprint" not in columns:
            self._conn.execute(
                "ALTER TABLE nodes ADD COLUMN content_fingerprint TEXT"
            )

    def _backfill_missing_content_fingerprints(self) -> None:
        """Compute SHA-256 fingerprints for any rows still missing one.

        Idempotent: only touches rows where content_fingerprint IS NULL. After
        the schema-v3 migration runs once, fresh inserts populate the column
        directly and this loop becomes a no-op.
        """

        rows = self._conn.execute(
            """
            SELECT id, content
            FROM nodes
            WHERE content_fingerprint IS NULL AND content IS NOT NULL
            """
        ).fetchall()
        if not rows:
            return
        updates = [
            (_content_fingerprint(str(r["content"])), str(r["id"])) for r in rows
        ]
        self._conn.executemany(
            "UPDATE nodes SET content_fingerprint = ? WHERE id = ?",
            updates,
        )

    def get_kv(self, key: str) -> str | None:
        """Read a server-wide kv entry, or None if unset."""

        row = self._conn.execute(
            "SELECT value FROM kv WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row["value"])

    def set_kv(self, key: str, value: str) -> None:
        """Upsert a server-wide kv entry with an UTC updated_at stamp."""

        now = _utc_now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO kv (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (str(key), str(value), now),
            )

    def _seed_retrieval_weights(self) -> None:
        now = _utc_now()
        with self._conn:
            for scope, weights in self.config.retrieval_weights.items():
                self._conn.execute(
                    """
                    INSERT INTO retrieval_weights (scope, bm25, vector, graph, learning_rate, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(scope) DO NOTHING
                    """,
                    (
                        scope,
                        weights.bm25,
                        weights.vector,
                        weights.graph,
                        weights.learning_rate,
                        now,
                    ),
                )

    @staticmethod
    def _validate_level(level: str) -> None:
        if level not in NODE_LEVELS:
            raise ValueError(f"invalid node level: {level}")

    @staticmethod
    def _validate_connection_type(relation_type: str) -> None:
        if relation_type not in CONNECTION_TYPES:
            raise ValueError(f"invalid connection type: {relation_type}")


def new_ulid() -> str:
    timestamp_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    value = (timestamp_ms << 80) | secrets.randbits(80)
    chars = []
    for index in range(26):
        shift = 125 - index * 5
        chars.append(_ULID_ALPHABET[(value >> shift) & 0x1F])
    return "".join(chars)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _scope_family(scope: str) -> str:
    if ":" in scope:
        return scope.split(":", 1)[0]
    return scope


def _prefix_upper_bound(prefix: str) -> str | None:
    if not prefix:
        return None
    last = ord(prefix[-1])
    if last >= 0x10FFFF:
        return None
    return prefix[:-1] + chr(last + 1)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _json_loads(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


def _fts_query(query: str) -> str:
    tokens = re.findall(r"\w+", query)
    return " OR ".join(f'"{token}"' for token in tokens)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    raise ValueError("value must be a list")


def _normalize_rejected_alternatives(
    alternatives: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in alternatives or ():
        if not isinstance(item, Mapping):
            raise ValueError("alternatives_considered items must be objects")
        approach = str(item.get("approach") or "").strip()
        reason = str(item.get("rejected_because") or item.get("reason") or "").strip()
        if not approach:
            raise ValueError("alternative approach must be non-empty")
        if not reason:
            raise ValueError("alternative rejected_because must be non-empty")
        normalized.append({"approach": approach, "rejected_because": reason})
    return normalized


def _rejected_alternative_content(approach: str, reason: str) -> str:
    return f"Rejected alternative: {approach}\nRejected because: {reason}"


def _node_from_row(row: sqlite3.Row) -> Node:
    source_traces = _json_loads(row["source_traces"], [])
    corrections = _json_loads(row["corrections"], [])
    provenance = _json_loads(row["provenance"], {})
    provenance["source_traces"] = source_traces
    provenance["corrections"] = corrections

    return Node(
        id=str(row["id"]),
        level=row["level"],
        content=str(row["content"]),
        embedding=_json_loads(row["embedding"], None),
        context=_json_loads(row["context"], {}),
        scope=str(row["scope"]),
        agent=row["agent"],
        task=row["task"],
        timestamp=str(row["timestamp"]),
        decayed=bool(row["decayed"]),
        decay_reason=row["decay_reason"],
        access_count=int(row["access_count"]),
        last_accessed=row["last_accessed"],
        usefulness_score=float(row["usefulness_score"]),
        confidence=float(row["confidence"]),
        unique_agents=int(row["unique_agents"]),
        temporal_hint=row["temporal_hint"],
        source_traces=source_traces,
        corrections=corrections,
        provenance=provenance,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _connection_from_row(row: sqlite3.Row) -> Connection:
    return Connection(
        id=str(row["id"]),
        source_id=str(row["source_id"]),
        target_id=str(row["target_id"]),
        type=row["type"],
        weight=float(row["weight"]),
        metadata=_json_loads(row["metadata"], {}),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _recall_event_from_row(row: sqlite3.Row) -> RecallEvent:
    return RecallEvent(
        id=str(row["id"]),
        query=str(row["query"]),
        scope=str(row["scope"]),
        requested_scope=str(row["requested_scope"]),
        resolved_scopes=[str(item) for item in _json_loads(row["resolved_scopes"], [])],
        ambient_context=_json_loads(row["ambient_context"], {}),
        depth=row["depth"],
        max_results=int(row["max_results"]),
        results=[dict(item) for item in _json_loads(row["results"], [])],
        agent=row["agent"],
        task=row["task"],
        session_id=row["session_id"],
        feedback_applied=bool(row["feedback_applied"]),
        feedback_trace_id=row["feedback_trace_id"],
        feedback_applied_at=row["feedback_applied_at"],
        created_at=str(row["created_at"]),
    )


def _recall_event_matches(
    event: RecallEvent,
    scope: str,
    context: Mapping[str, Any],
    content: str | None = None,
) -> bool:
    return _recall_event_match_strength(event, scope, context, content) is not None


def _recall_event_match_strength(
    event: RecallEvent,
    scope: str,
    context: Mapping[str, Any],
    content: str | None,
) -> str | None:
    exact_scope_match = scope == event.scope or scope == event.requested_scope
    if not exact_scope_match:
        return None

    context_session = _optional_str(context.get("session_id") or context.get("session"))
    context_task = _optional_str(context.get("task"))
    context_agent = _optional_str(context.get("agent"))

    if event.session_id and context_session and event.session_id != context_session:
        return None
    if event.task and context_task and event.task != context_task:
        return None

    same_session = bool(event.session_id and context_session and event.session_id == context_session)
    same_task = bool(event.task and context_task and event.task == context_task)
    if same_session or same_task:
        return "strong"

    same_agent = bool(event.agent and context_agent and event.agent == context_agent)
    if event.agent and context_agent and event.agent != context_agent:
        return None

    text_match = content is None or _recall_event_text_similarity(event, content) >= _RECALL_TEXT_SIMILARITY_THRESHOLD
    if same_agent and text_match:
        return "strong"
    if text_match:
        return "weak"

    return None


def _recall_event_text_similarity(event: RecallEvent, content: str) -> float:
    query_tokens = set(tokenize(event.query))
    content_tokens = set(tokenize(content))
    if not query_tokens or not content_tokens:
        return 0.0
    overlap = len(query_tokens & content_tokens)
    return overlap / max(1, min(len(query_tokens), len(content_tokens)))


def _weights_from_row(row: sqlite3.Row) -> RetrievalWeights:
    return RetrievalWeights(
        scope=str(row["scope"]),
        bm25=float(row["bm25"]),
        vector=float(row["vector"]),
        graph=float(row["graph"]),
        learning_rate=float(row["learning_rate"]),
        updated_at=str(row["updated_at"]),
    )
