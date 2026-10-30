"""Soft-deletion and decay services for Living Memory nodes.

Nodes are never removed physically here: every path in this module ends in
:meth:`MemoryStore.soft_delete_node`, which sets ``decayed = 1`` and records a
``decay_reason``. A sweep is therefore reversible by an operator with one
``UPDATE nodes SET decayed = 0 WHERE decay_reason = ...``.

The TTL anchor is what gives the live working set a ceiling. Anchoring on
``last_accessed`` — the behaviour that shipped first — makes a frequently read
node immortal, because every recall bumps ``last_accessed``: the set of live
traces then grows monotonically and only ever shrinks by hand. Anchoring on
creation makes the ceiling arrive on its own, one trace at a time, while
:data:`TTL_ANCHOR_ENV` keeps the old behaviour one env var away.

Three valves, all read at call time so a sweep already running in a server
picks them up on the next pass (the server never has to know about them):

``LM_DECAY_TTL_ANCHOR``
    ``created`` (default) or ``last_accessed`` (the legacy anchor, restored
    byte-for-byte). Anything else falls back to the default.
``LM_DECAY_CONCEPT_COVERED``
    Off by default. On, a trace covered by a live concept and unread since
    that coverage decays regardless of its age — see
    :func:`concept_coverage` for why this cannot be the default.
``LM_DECAY_MAX_PER_SWEEP``
    Ceiling on how many nodes one TTL pass may retire (default
    :data:`DEFAULT_MAX_EXPIRED_PER_SWEEP`; ``0`` means uncapped).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from heapq import nsmallest
from typing import Any, Iterable

from living_memory.relations import correction_sql
from living_memory.models import NODE_LEVELS, Node
from living_memory.storage import MemoryStore
from living_memory.temporal import parse_timestamp

#: Selects the point in time a trace's TTL counts from.
TTL_ANCHOR_ENV = "LM_DECAY_TTL_ANCHOR"
#: Opt-in: decay traces a live concept already covers and nobody read since.
CONCEPT_COVERED_ENV = "LM_DECAY_CONCEPT_COVERED"
#: Ceiling on nodes retired by one TTL pass; ``0`` disables the cap.
MAX_PER_SWEEP_ENV = "LM_DECAY_MAX_PER_SWEEP"

TTL_ANCHOR_CREATED = "created"
TTL_ANCHOR_LAST_ACCESSED = "last_accessed"
TTL_ANCHORS: tuple[str, ...] = (TTL_ANCHOR_CREATED, TTL_ANCHOR_LAST_ACCESSED)

#: Creation, not last read.
#:
#: Chosen with the live corpus measured (read-only, 2026-08-23): 12,887 active
#: traces, the oldest created 2026-05-14 — 100.1 days old against
#: ``trace_ttl_days = 180``. Not one active trace was older than the TTL, so
#: flipping the anchor from ``last_accessed`` to creation decayed NOTHING on
#: the day it shipped; the first trace can only reach the cutoff around
#: 2026-11-10, and then one day's worth at a time. This default is a ceiling
#: that arrives by itself, deliberately chosen with ~80 days of head-room in
#: hand — it is not a cull of the live corpus, and a reader who finds it
#: firing is seeing the ceiling work, not a regression.
DEFAULT_TTL_ANCHOR = TTL_ANCHOR_CREATED

#: Nodes one TTL pass may retire. The steady-state inflow of the live corpus is
#: ~127 traces/day against an hourly sweep, so this never binds in normal
#: operation; it exists so that a lowered ``trace_ttl_days``, a backdated
#: import, or :data:`CONCEPT_COVERED_ENV` cannot retire the working set in a
#: single pass. What the cap defers is not lost — the next sweep takes the next
#: batch, oldest first.
DEFAULT_MAX_EXPIRED_PER_SWEEP = 500

TTL_EXPIRED_REASON = "ttl expired"
CONCEPT_COVERED_REASON = "concept covered, unread since coverage"

#: Levels whose ``source_traces`` count as coverage of a trace.
_COVERING_LEVELS: tuple[str, ...] = ("concept", "schema")
_ENABLED_ENV_FLAGS = frozenset({"1", "true", "yes", "on"})


@dataclass(slots=True)
class DecayResult:
    """Nodes soft-deleted during a maintenance pass."""

    expired: list[Node] = field(default_factory=list)
    superseded: list[Node] = field(default_factory=list)

    @property
    def nodes(self) -> list[Node]:
        return [*self.expired, *self.superseded]


def memory_forget(store: MemoryStore, node_id: str, reason: str | None = None) -> Node:
    """Forget a source and refresh its derivatives, preserving stored history."""

    from living_memory.consolidation import memory_forget as forget

    return forget(store, node_id, reason)


def apply_decay(
    store: MemoryStore,
    *,
    scope: str | None = None,
    now: datetime | None = None,
    ttl_days: int | None = None,
    include_superseded: bool = True,
    ttl_anchor: str | None = None,
    decay_concept_covered: bool | None = None,
    max_expired: int | None = None,
) -> DecayResult:
    """Soft-delete expired nodes and, optionally, superseded originals.

    The three keyword valves mirror the environment variables and exist for
    callers (and tests) that want to be explicit; ``None`` means "read the
    environment", which is what the server's sweep does.
    """

    result = DecayResult()
    result.expired.extend(
        soft_delete_expired(
            store,
            scope=scope,
            now=now,
            ttl_days=ttl_days,
            ttl_anchor=ttl_anchor,
            decay_concept_covered=decay_concept_covered,
            max_expired=max_expired,
        )
    )
    if include_superseded:
        result.superseded.extend(soft_delete_superseded(store, scope=scope))
    return result


def resolve_ttl_anchor() -> str:
    """Resolve :data:`TTL_ANCHOR_ENV`; unset or unrecognised means the default."""

    raw = (os.environ.get(TTL_ANCHOR_ENV) or "").strip().lower()
    if raw in {TTL_ANCHOR_LAST_ACCESSED, "last-accessed", "legacy"}:
        return TTL_ANCHOR_LAST_ACCESSED
    return DEFAULT_TTL_ANCHOR


def concept_coverage_rule_enabled() -> bool:
    """True when the operator opted into decaying concept-covered traces."""

    return (os.environ.get(CONCEPT_COVERED_ENV) or "").strip().lower() in _ENABLED_ENV_FLAGS


def max_expired_per_sweep() -> int:
    """Resolve :data:`MAX_PER_SWEEP_ENV`; ``0`` means uncapped."""

    raw = (os.environ.get(MAX_PER_SWEEP_ENV) or "").strip()
    if not raw:
        return DEFAULT_MAX_EXPIRED_PER_SWEEP
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_EXPIRED_PER_SWEEP
    return max(0, value)


def concept_coverage(
    store: MemoryStore, *, include_dates: bool = True,
) -> dict[str, datetime | None]:
    """Map sources of live concepts and schemas to their coverage start.

    Coverage is recorded as the concept's ``provenance.source_traces``, which
    makes this map do double duty. Under shipped defaults it is a *guard*: a
    trace listed there is provenance for something live, and a TTL pass must
    walk past it however old it is. Under :data:`CONCEPT_COVERED_ENV` it is
    the *rule*: the digest already carries the fact, so a source nobody has
    read since it was consolidated may retire.

    Both policies use the same coverage fact. The default retains source
    evidence; retiring unused covered sources is an explicit operator choice,
    bounded by :data:`MAX_PER_SWEEP_ENV` just like TTL retirement.

    With ``include_dates=False``, read only source membership and return
    ``None`` for every value. The guard needs no provenance or coverage dates.
    Both projections include live owners across all scopes.

    Each owner's ``provenance.source_coverage`` maps current source IDs to
    their continuous coverage start. Maintenance preserves retained entries;
    new or replacement sources start at their inclusion. Missing or malformed
    entries have unknown age: owner creation, consolidation and access dates
    cannot establish when an individual source joined. Across live owners the
    earliest start wins, with unknown age taking precedence. A recorded read
    then prevents coverage retirement when its relation to coverage is unknown;
    the opt-in TTL fallback still applies.
    """

    placeholders = ", ".join("?" for _ in _COVERING_LEVELS)
    columns = "source_traces, provenance" if include_dates else "source_traces"
    rows = store.connection.execute(
        f"""
        SELECT {columns}
        FROM nodes
        WHERE decayed = 0 AND level IN ({placeholders})
        """,
        list(_COVERING_LEVELS),
    )

    coverage: dict[str, datetime | None] = {}
    for row in rows:
        source_ids = _json_list(row["source_traces"])
        if not include_dates:
            coverage.update(dict.fromkeys(source_ids))
            continue
        try:
            provenance = json.loads(row["provenance"])
        except (TypeError, ValueError):
            provenance = None
        for trace_id in source_ids:
            covered_at = _coverage_moment(provenance, trace_id)
            if trace_id not in coverage:
                coverage[trace_id] = covered_at
                continue
            known = coverage[trace_id]
            if known is None or covered_at is None:
                # Coverage of unknown age wins over a dated one, whichever row
                # came first: it is the reading that keeps the most traces.
                coverage[trace_id] = None
            elif covered_at < known:
                coverage[trace_id] = covered_at
    return coverage


def soft_delete_expired(
    store: MemoryStore,
    *,
    scope: str | None = None,
    now: datetime | None = None,
    ttl_days: int | None = None,
    levels: Iterable[str] = ("trace",),
    ttl_anchor: str | None = None,
    decay_concept_covered: bool | None = None,
    max_expired: int | None = None,
) -> list[Node]:
    """Decay nodes whose TTL has run out, oldest first, up to the sweep cap.

    Concept coverage is checked before age: a covered trace is skipped whatever
    its anchor says, unless the operator enabled
    :data:`CONCEPT_COVERED_ENV`, in which case a covered trace unread since
    coverage decays and a covered trace still being read falls through to the
    normal TTL check.
    """

    ttl = store.config.trace_ttl_days if ttl_days is None else int(ttl_days)
    if ttl < 0:
        raise ValueError("ttl_days must be non-negative")

    anchor = ttl_anchor if ttl_anchor is not None else resolve_ttl_anchor()
    if anchor not in TTL_ANCHORS:
        raise ValueError(f"ttl_anchor must be one of {TTL_ANCHORS}")
    coverage_rule = (
        concept_coverage_rule_enabled()
        if decay_concept_covered is None
        else bool(decay_concept_covered)
    )
    cap = max_expired_per_sweep() if max_expired is None else max(0, int(max_expired))

    moment = now or datetime.now(UTC)
    cutoff = moment - timedelta(days=ttl)
    coverage = concept_coverage(store, include_dates=coverage_rule)

    def eligible_candidates() -> Iterable[tuple[datetime, str, str]]:
        for level in levels:
            clauses = ["decayed = 0"]
            params: list[str] = []
            if level is not None:
                if level not in NODE_LEVELS:
                    raise ValueError(f"invalid node level: {level}")
                clauses.append("level = ?")
                params.append(level)
            if scope is not None:
                clauses.append("scope = ?")
                params.append(scope)
            # Eligibility needs dates and identity, not content or provenance.
            # Full nodes are read by soft_delete_node only for selected retirements.
            rows = store.connection.execute(
                f"""
                SELECT id, timestamp, created_at, last_accessed FROM nodes
                WHERE {' AND '.join(clauses)}
                """,
                params,
            )
            for row in rows:
                node_id = str(row["id"])
                reference = _ttl_reference(row, anchor)
                if node_id in coverage:
                    if not coverage_rule:
                        # The invariant: no trace listed in a live concept's
                        # source_traces is decayed. Its digest points back here.
                        continue
                    if not _read_since(row, coverage[node_id]):
                        yield (reference or moment, node_id, CONCEPT_COVERED_REASON)
                        continue
                if reference is not None and reference < cutoff:
                    yield (reference, node_id, TTL_EXPIRED_REASON)

    # Oldest reference first, id as tie-breaker: whatever the cap defers is the
    # youngest of the batch, and the next sweep resumes exactly where this one
    # stopped instead of re-picking an arbitrary subset.
    # Both selectors exhaust eligibility before any mutation. A finite cap
    # retains only O(cap) candidates; the key keeps exact ties stable without
    # comparing reasons, including repeated visits through duplicated levels.
    if cap:
        candidates = nsmallest(cap, eligible_candidates(), key=lambda candidate: (candidate[0], candidate[1]))
    else:
        candidates = sorted(eligible_candidates(), key=lambda candidate: (candidate[0], candidate[1]))

    return [store.soft_delete_node(node_id, reason) for _, node_id, reason in candidates]


def soft_delete_superseded(store: MemoryStore, *, scope: str | None = None) -> list[Node]:
    """Decay active nodes that have been replaced by a superseding node.

    Only genuine correction relations enter this path. Concept coverage must
    not guard it: a corrected trace that stays live
    because a concept cites it keeps outranking the correction in recall.
    """

    clauses = [
        correction_sql("c"),
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


def _ttl_reference(row: Any, anchor: str) -> datetime | None:
    """The moment a node's TTL counts from, under the selected anchor."""

    if anchor == TTL_ANCHOR_LAST_ACCESSED:
        return parse_timestamp(row["last_accessed"]) or parse_timestamp(row["timestamp"])
    # ``timestamp`` is the node's own creation moment (storage fills it from
    # the writer's context or from now); ``created_at`` is the row insert and
    # can differ for a backdated write.
    return parse_timestamp(row["timestamp"]) or parse_timestamp(row["created_at"])


def _read_since(row: Any, covered_at: datetime | None) -> bool:
    """True when the trace was read after a concept started covering it."""

    read_at = parse_timestamp(row["last_accessed"])
    if read_at is None:
        return False
    if covered_at is None:
        # Coverage of unknown age: treat the trace as still in use rather than
        # guess it redundant.
        return True
    return read_at > covered_at


def _coverage_moment(provenance: Any, trace_id: str) -> datetime | None:
    """Only an explicit per-source start establishes coverage age."""

    coverage = provenance.get("source_coverage") if isinstance(provenance, dict) else None
    moment = coverage.get(trace_id) if isinstance(coverage, dict) else None
    return parse_timestamp(moment) if isinstance(moment, str) else None


def _json_list(raw: Any) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if item]
