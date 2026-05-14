"""Soft-deletion and decay services for Living Memory nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Iterable

from living_memory.models import Node
from living_memory.storage import MemoryStore
from living_memory.temporal import parse_timestamp


@dataclass(slots=True)
class DecayResult:
    """Nodes soft-deleted during a maintenance pass."""

    expired: list[Node] = field(default_factory=list)
    superseded: list[Node] = field(default_factory=list)

    @property
    def nodes(self) -> list[Node]:
        return [*self.expired, *self.superseded]


def memory_forget(store: MemoryStore, node_id: str, reason: str | None = None) -> Node:
    """Manually decay a node without removing its stored record."""

    return store.soft_delete_node(node_id, reason or "manual forget")


def apply_decay(
    store: MemoryStore,
    *,
    scope: str | None = None,
    now: datetime | None = None,
    ttl_days: int | None = None,
    include_superseded: bool = True,
) -> DecayResult:
    """Soft-delete expired nodes and, optionally, superseded originals."""

    result = DecayResult()
    result.expired.extend(
        soft_delete_expired(store, scope=scope, now=now, ttl_days=ttl_days)
    )
    if include_superseded:
        result.superseded.extend(soft_delete_superseded(store, scope=scope))
    return result


def soft_delete_expired(
    store: MemoryStore,
    *,
    scope: str | None = None,
    now: datetime | None = None,
    ttl_days: int | None = None,
    levels: Iterable[str] = ("trace",),
) -> list[Node]:
    """Decay nodes whose last activity is older than the configured TTL."""

    ttl = store.config.trace_ttl_days if ttl_days is None else int(ttl_days)
    if ttl < 0:
        raise ValueError("ttl_days must be non-negative")

    cutoff = (now or datetime.now(UTC)) - timedelta(days=ttl)
    expired: list[Node] = []
    for level in levels:
        nodes = store.list_nodes(
            level=level, scope=scope, include_decayed=False, limit=100_000
        )
        for node in nodes:
            reference = parse_timestamp(node.last_accessed) or parse_timestamp(node.timestamp)
            if reference is not None and reference < cutoff:
                expired.append(store.soft_delete_node(node.id, "ttl expired"))
    return expired


def soft_delete_superseded(store: MemoryStore, *, scope: str | None = None) -> list[Node]:
    """Decay active nodes that have been replaced by a superseding node."""

    clauses = [
        "c.type = 'supersedes'",
        "target.decayed = 0",
        "source.decayed = 0",
    ]
    params: list[str] = []
    if scope is not None:
        clauses.append("target.scope = ?")
        params.append(scope)

    rows = store.connection.execute(
        f"""
        SELECT DISTINCT target.id
        FROM connections c
        JOIN nodes source ON source.id = c.source_id
        JOIN nodes target ON target.id = c.target_id
        WHERE {' AND '.join(clauses)}
        """,
        params,
    ).fetchall()

    decayed: list[Node] = []
    for row in rows:
        decayed.append(store.soft_delete_node(str(row["id"]), "superseded"))
    return decayed
