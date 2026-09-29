"""Global hub suppression: nodes marked ``irrelevant`` across many questions.

Goal recall-precision. A node that agents marked ``irrelevant`` for several
DIFFERENT questions and never confirmed as useful is demoted for every query;
the first positive evidence lifts the demotion. ``MemoryRecallService`` merges
the returned multipliers with the query-relative ones
(``living_memory.irrelevance``) by taking the stronger (smaller) multiplier per
node, inside ``rank_candidates``.

The rule
--------
* **Valve.** ``LM_HUB_SUPPRESSION_FACTOR`` is the multiplier, a float in
  ``[0, 1)``. Unset, unparsable, non-finite or out of range means off:
  :func:`hub_demotions` returns ``{}`` before touching the store, so ranking
  is byte-identical to a build without this module. ``LM_HUB_MIN_QUERIES``
  (default 3, integers >= 1) is the number of distinct questions needed.
* **Distinct questions.** Only accepted ``irrelevant`` rows of
  ``recall_feedback_marks`` count. Each is keyed by the query anchor of the
  recall event it was passed on, resolved the way
  ``query_anchors.resolve_query_anchor`` resolves identity: exact
  ``(normalize_scope(scope), recall_fingerprint(query, scope))`` in
  ``query_anchors``; failing that, the ``anchor_id`` of the
  ``query_irrelevance`` row whose ``last_event_id`` is that event (the
  cosine-near-duplicate anchor the write path chose); failing that, the
  whitespace-collapsed query in its scope. Repeats of one question therefore
  count once however often they are marked.
* **Positive evidence lifts.** An accepted ``used`` mark at any time, or any
  ``recall_credit_ledger`` row (``grounded``/``lookup``) or
  ``recall_explicit_credit`` row for the node credited at or after the node's
  first accepted ``irrelevant`` mark. Credit that predates every complaint
  does not protect a node that later became noise; one credit after the first
  complaint does.

Every read tolerates absent tables (pre-feedback, pre-v7 or read-only
snapshots): a missing marks table means no hubs, a missing credit or anchor
table means that evidence or that identity stage is simply absent.

Cost
----
The inputs are append-only (nothing in LM deletes from them), so the hub set
is cached on the store keyed by ``MAX(rowid)`` of ``recall_feedback_marks``,
``recall_credit_ledger``, ``recall_explicit_credit`` and ``query_anchors``
(four index-only lookups). Marks and credit grow on almost every recall, and a
full recount costs tens of milliseconds on the sfx store, so a changed key
does not mean a recount: positive evidence in the appended rows lifts its
hubs immediately, and a recount (the only way a node becomes a hub) runs at
most once per ``HUB_REFRESH_SECONDS`` and only when new irrelevant marks
arrived. Numbers and the sfx census are in docs/hub-suppression.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import sqlite3
import time
from typing import Any

from living_memory.scope import normalize_scope
from living_memory.storage import recall_fingerprint

#: ``LM_HUB_MIN_QUERIES`` default: distinct questions needed to call a node a hub.
DEFAULT_HUB_MIN_QUERIES = 3

#: Minimum age of the cached hub set before new irrelevant marks trigger a
#: full recount. Lifts are never delayed; only *becoming* a hub is.
HUB_REFRESH_SECONDS = 60.0

_MARKS = "recall_feedback_marks"
_LEDGER = "recall_credit_ledger"
_EXPLICIT = "recall_explicit_credit"
_ANCHORS = "query_anchors"
_IRRELEVANCE = "query_irrelevance"
_EVENTS = "recall_events"
_REVISION_TABLES = (_MARKS, _LEDGER, _EXPLICIT, _ANCHORS)


def hub_suppression_factor() -> float | None:
    """``LM_HUB_SUPPRESSION_FACTOR`` in ``[0, 1)``, or ``None`` (off)."""

    raw = os.environ.get("LM_HUB_SUPPRESSION_FACTOR", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0.0 or value >= 1.0:
        return None
    return value


def hub_min_queries() -> int:
    """``LM_HUB_MIN_QUERIES`` (>= 1); invalid values fall back to the default."""

    raw = os.environ.get("LM_HUB_MIN_QUERIES", "").strip()
    if not raw:
        return DEFAULT_HUB_MIN_QUERIES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_HUB_MIN_QUERIES
    return value if value >= 1 else DEFAULT_HUB_MIN_QUERIES


@dataclass(frozen=True)
class HubCount:
    """One node's evidence. ``hub`` is the verdict at the requested threshold."""

    node_id: str
    distinct_anchors: int
    distinct_queries: int
    irrelevant_marks: int
    first_irrelevant_at: str
    used_marks: int
    credits_after_first: int
    hub: bool


def _present(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?, ?, ?, ?, ?)",
        (_MARKS, _LEDGER, _EXPLICIT, _ANCHORS, _IRRELEVANCE, _EVENTS),
    ).fetchall()
    return {str(row[0]) for row in rows}


def _before_clause(column: str, before: str | None) -> tuple[str, tuple[Any, ...]]:
    return ("", ()) if before is None else (f" AND {column} < ?", (before,))


def hub_counts(
    conn: sqlite3.Connection,
    *,
    min_queries: int = DEFAULT_HUB_MIN_QUERIES,
    before: str | None = None,
) -> list[HubCount]:
    """Per-node irrelevance evidence for every node with an accepted irrelevant mark.

    Pure over ``conn`` (read-only use is fine), so offline harnesses can call it
    on a snapshot. ``before`` restricts every input (marks by ``marked_at``,
    credit by ``credited_at``) to rows strictly earlier than that timestamp,
    e.g. a train/eval split point. Sorted by ``distinct_anchors`` descending.
    """

    present = _present(conn)
    if _MARKS not in present:
        return []
    clause, params = _before_clause("m.marked_at", before)
    if _EVENTS in present:
        rows = conn.execute(
            f"""
            SELECT m.node_id, m.recall_event_id, MIN(m.marked_at), COUNT(*),
                   e.query, e.scope
            FROM {_MARKS} m LEFT JOIN {_EVENTS} e ON e.id = m.recall_event_id
            WHERE m.accepted = 1 AND m.mark = 'irrelevant'{clause}
            GROUP BY m.node_id, m.recall_event_id
            """,
            params,
        ).fetchall()
    else:
        rows = [
            (node_id, event_id, marked_at, count, None, None)
            for node_id, event_id, marked_at, count in conn.execute(
                f"""
                SELECT m.node_id, m.recall_event_id, MIN(m.marked_at), COUNT(*)
                FROM {_MARKS} m
                WHERE m.accepted = 1 AND m.mark = 'irrelevant'{clause}
                GROUP BY m.node_id, m.recall_event_id
                """,
                params,
            ).fetchall()
        ]
    if not rows:
        return []

    near_anchor: dict[tuple[str, str], str] = {}
    if _IRRELEVANCE in present:
        for anchor_id, node_id, event_id in conn.execute(
            f"SELECT anchor_id, node_id, last_event_id FROM {_IRRELEVANCE}"
            " WHERE last_event_id IS NOT NULL"
        ):
            near_anchor[(str(event_id), str(node_id))] = str(anchor_id)
    has_anchors = _ANCHORS in present
    anchor_by_event: dict[str, tuple[str, str | None]] = {}

    anchors: dict[str, set[str]] = {}
    queries: dict[str, set[str]] = {}
    marks: dict[str, int] = {}
    first: dict[str, str] = {}
    for node_id, event_id, marked_at, count, query, scope in rows:
        node_id = str(node_id)
        event_id = str(event_id)
        if query is None:
            raw_key, exact = f"event:{event_id}", None
        elif event_id in anchor_by_event:
            raw_key, exact = anchor_by_event[event_id]
        else:
            anchor_scope = normalize_scope(scope or "global")
            raw_key = f"query:{anchor_scope}\n{' '.join(str(query).split())}"
            exact = None
            if has_anchors:
                hit = conn.execute(
                    f"SELECT id FROM {_ANCHORS} WHERE scope = ? AND fingerprint = ?",
                    (anchor_scope, recall_fingerprint(query, anchor_scope)),
                ).fetchone()
                exact = None if hit is None else f"anchor:{hit[0]}"
            anchor_by_event[event_id] = (raw_key, exact)
        anchor_key = exact
        if anchor_key is None:
            near = near_anchor.get((event_id, node_id))
            anchor_key = f"anchor:{near}" if near else raw_key
        anchors.setdefault(node_id, set()).add(anchor_key)
        queries.setdefault(node_id, set()).add(raw_key)
        marks[node_id] = marks.get(node_id, 0) + int(count)
        stamp = str(marked_at)
        if node_id not in first or stamp < first[node_id]:
            first[node_id] = stamp

    used: dict[str, int] = {}
    clause, params = _before_clause("marked_at", before)
    for node_id, count in conn.execute(
        f"""
        SELECT node_id, COUNT(*) FROM {_MARKS}
        WHERE accepted = 1 AND mark = 'used'{clause} GROUP BY node_id
        """,
        params,
    ):
        used[str(node_id)] = int(count)

    credits: dict[str, int] = {}
    clause, params = _before_clause("c.credited_at", before)
    first_clause, first_params = _before_clause("marked_at", before)
    for table in (_LEDGER, _EXPLICIT):
        if table not in present:
            continue
        for node_id, count in conn.execute(
            f"""
            WITH first AS (
                SELECT node_id, MIN(marked_at) AS first_at FROM {_MARKS}
                WHERE accepted = 1 AND mark = 'irrelevant'{first_clause}
                GROUP BY node_id
            )
            SELECT c.node_id, COUNT(*) FROM {table} c
            JOIN first f ON f.node_id = c.node_id
            WHERE c.credited_at >= f.first_at{clause}
            GROUP BY c.node_id
            """,
            first_params + params,
        ):
            credits[str(node_id)] = credits.get(str(node_id), 0) + int(count)

    threshold = max(1, int(min_queries))
    result = [
        HubCount(
            node_id=node_id,
            distinct_anchors=len(anchors[node_id]),
            distinct_queries=len(queries[node_id]),
            irrelevant_marks=marks[node_id],
            first_irrelevant_at=first[node_id],
            used_marks=used.get(node_id, 0),
            credits_after_first=credits.get(node_id, 0),
            hub=(
                len(anchors[node_id]) >= threshold
                and not used.get(node_id)
                and not credits.get(node_id)
            ),
        )
        for node_id in anchors
    ]
    result.sort(key=lambda item: (-item.distinct_anchors, -item.irrelevant_marks, item.node_id))
    return result


def hub_ids(
    conn: sqlite3.Connection,
    *,
    min_queries: int = DEFAULT_HUB_MIN_QUERIES,
    before: str | None = None,
) -> dict[str, int]:
    """``{node_id: distinct anchors}`` for the hubs only (the suppressed set)."""

    return {
        item.node_id: item.distinct_anchors
        for item in hub_counts(conn, min_queries=min_queries, before=before)
        if item.hub
    }


def store_revision(conn: sqlite3.Connection) -> tuple[Any, ...]:
    """Cache key: ``MAX(rowid)`` of every append-only input table (``None`` if absent).

    Order is ``(marks, ledger, explicit credit, anchors)``.
    """

    present = _present(conn)
    return tuple(
        conn.execute(f"SELECT MAX(rowid) FROM {table}").fetchone()[0]
        if table in present
        else None
        for table in _REVISION_TABLES
    )


@dataclass
class _HubCache:
    min_queries: int
    revision: tuple[Any, ...]
    #: hub id -> first counted irrelevant mark, the lower bound for lifting credit
    hubs: dict[str, str]
    computed_at: float
    #: ``MAX(rowid)`` of the marks table at the last full count; irrelevant
    #: marks past it are pending until the next recount
    counted_marks: Any


def _connection(store: Any) -> sqlite3.Connection | None:
    conn = getattr(store, "connection", None)
    if conn is None:
        conn = getattr(store, "_conn", None)
    return conn if isinstance(conn, sqlite3.Connection) else None


def _full(conn: sqlite3.Connection, min_queries: int, revision: tuple[Any, ...]) -> _HubCache:
    return _HubCache(
        min_queries=min_queries,
        revision=revision,
        hubs={
            item.node_id: item.first_irrelevant_at
            for item in hub_counts(conn, min_queries=min_queries)
            if item.hub
        },
        computed_at=time.monotonic(),
        counted_marks=revision[0],
    )


def _new_irrelevant(conn: sqlite3.Connection, since: Any) -> bool:
    row = conn.execute(
        f"""
        SELECT 1 FROM {_MARKS}
        WHERE rowid > ? AND accepted = 1 AND mark = 'irrelevant' LIMIT 1
        """,
        (since or 0,),
    ).fetchone()
    return row is not None


def _lift(conn: sqlite3.Connection, cache: _HubCache, revision: tuple[Any, ...]) -> None:
    """Drop hubs that gained positive evidence in rows appended since ``cache.revision``."""

    if not cache.hubs:
        return
    old_marks, old_ledger, old_explicit, _ = cache.revision
    new_marks, new_ledger, new_explicit, _ = revision
    lifted: set[str] = set()
    if new_marks is not None and new_marks != old_marks:
        for (node_id,) in conn.execute(
            f"""
            SELECT DISTINCT node_id FROM {_MARKS}
            WHERE rowid > ? AND accepted = 1 AND mark = 'used'
            """,
            (old_marks or 0,),
        ):
            lifted.add(str(node_id))
    for table, old, new in ((_LEDGER, old_ledger, new_ledger), (_EXPLICIT, old_explicit, new_explicit)):
        if new is None or new == old:
            continue
        for node_id, credited_at in conn.execute(
            f"SELECT node_id, credited_at FROM {table} WHERE rowid > ?", (old or 0,)
        ):
            first = cache.hubs.get(str(node_id))
            if first is not None and str(credited_at) >= first:
                lifted.add(str(node_id))
    for node_id in lifted:
        cache.hubs.pop(node_id, None)


def hub_demotions(store: Any) -> dict[str, float]:
    """``{node_id: multiplier in [0, 1)}`` for globally demoted hubs; ``{}`` when off.

    Unchanged inputs are served from the cache on ``store``. When rows were
    appended, positive evidence among them lifts its hubs at once (an
    index-range read of the new rows only); a full recount, the only way a
    node *becomes* a hub, runs when new irrelevant marks arrived and the last
    recount is older than :data:`HUB_REFRESH_SECONDS`.
    """

    factor = hub_suppression_factor()
    if factor is None:
        return {}
    conn = _connection(store)
    if conn is None:
        return {}
    min_queries = hub_min_queries()
    try:
        revision = store_revision(conn)
        cache: _HubCache | None = getattr(store, "_hub_suppression_cache", None)
        if cache is None or cache.min_queries != min_queries:
            cache = _full(conn, min_queries, revision)
        else:
            if cache.revision != revision:
                _lift(conn, cache, revision)
                cache.revision = revision
            if (
                cache.counted_marks != revision[0]
                and time.monotonic() - cache.computed_at >= HUB_REFRESH_SECONDS
            ):
                if _new_irrelevant(conn, cache.counted_marks):
                    cache = _full(conn, min_queries, revision)
                else:
                    cache.counted_marks = revision[0]
        try:
            store._hub_suppression_cache = cache
        except AttributeError:  # pragma: no cover - slotted stand-ins
            pass
    except sqlite3.Error:
        return {}
    return {node_id: factor for node_id in cache.hubs}
