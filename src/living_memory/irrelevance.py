"""Query-relative irrelevance: "this node is not for THIS question".

An explicit ``irrelevant`` mark (goal explicit-recall-feedback) says the
delivered node did not belong to the query it was delivered for. It does not
say the node is wrong — ``memory_teach`` covers that — nor that it is useless:
the same node can be exactly right for another question. So the mark must not
touch anything global. ``usefulness_score``, ``confidence`` and the per-scope
retrieval weights never move here. What moves is one row keyed by
(query anchor, node).

Where the association lives
---------------------------
The query side is the anchor machinery (``living_memory.query_anchors``): the
event's query resolves to its anchor through the same identity a grounded
consumption uses (fingerprint over (query, scope), then in-scope cosine at
``ANCHOR_DEDUP_COSINE_THRESHOLD``), without reinforcing it. At retrieval time
the incoming query is matched to anchors by ``match_anchors`` with the anchor
channel's own floor and limit, so "the same question" means here exactly what
it means to the anchor channel.

The (anchor, node) row cannot go into an existing table. ``query_anchor_edges``
is read as positive evidence: its weight saturates upward and every edge is a
graph seed, so a negative row there would *pull the node in*. ``connections``
carries a CHECK on ``type`` and links node to node. Hence one additive table,
``query_irrelevance``, created by ``CREATE TABLE IF NOT EXISTS`` on the first
write (the first accepted irrelevant mark under
``LM_EXPLICIT_FEEDBACK_POLICY=credit``). Every read tolerates its absence, and
the absence of the anchor tables too, so a pre-v7 or read-only snapshot simply
has no demotions.

Strength, accumulation, cancellation
------------------------------------
* Each accepted mark adds ``IRRELEVANCE_MARK_WEIGHT`` (0.5) to the row's
  weight, saturating at 1.0: one mark is half strength, a second mark on the
  same anchor/node (a later event that resolves to the same anchor) is full.
  ``marks`` keeps counting past saturation.
* At retrieval the demotion multiplier for a node is
  ``1 - (1 - F) * weight * closeness``, with ``F`` =
  ``LM_QUERY_IRRELEVANCE_FACTOR`` (default 0.5; 1.0 disables) and
  ``closeness = (cos - floor) / (1 - floor)`` of the matched anchor, where
  ``floor`` is the anchor channel's match floor. The exact query scores
  closeness 1.0; a paraphrase just over the floor barely moves. The multiplier
  is bounded below by ``F``: a saturated mark on the exact query can at most
  scale the score by ``F``, never remove the node. Several matched anchors
  marking one node keep the strongest claim, not the product.
* Positive evidence for the same (anchor, node) **cancels** the row: grounded,
  lookup or explicit ``used`` credit all reinforce the anchor through
  ``feedback._reinforce_query_anchors``, which calls
  :func:`cancel_query_irrelevance` for the credited targets. The weight goes
  to 0 (the row stays, with ``cancels`` counted, for audit). Observed use of a
  node for this question beats an earlier claim that it did not belong; a
  fresh ``irrelevant`` mark starts accumulating again from zero.

Rows do not follow ``supersedes``: a correction is new content and gets a
fresh hearing. A decayed anchor stops matching, so its demotions lapse with it.
Docs and the reasoning behind the default: docs/query-irrelevance.md.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
import logging
import math
import os
import sqlite3
from typing import Any

from living_memory.query_anchors import (
    ANCHOR_MATCH_COSINE_THRESHOLD,
    ANCHOR_MATCH_LIMIT,
    match_anchors,
    resolve_query_anchor,
)
from living_memory.storage import MemoryStore

_LOG = logging.getLogger(__name__)

QUERY_IRRELEVANCE_TABLE = "query_irrelevance"

#: Additive DDL; see the module note for why no existing table fits.
QUERY_IRRELEVANCE_SCHEMA_SQL = f"""
    CREATE TABLE IF NOT EXISTS {QUERY_IRRELEVANCE_TABLE} (
        anchor_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        weight REAL NOT NULL DEFAULT 0.0,
        marks INTEGER NOT NULL DEFAULT 0,
        cancels INTEGER NOT NULL DEFAULT 0,
        last_event_id TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (anchor_id, node_id)
    )
"""

#: Weight one accepted irrelevant mark adds to an (anchor, node) row.
IRRELEVANCE_MARK_WEIGHT = 0.5

#: ``LM_QUERY_IRRELEVANCE_FACTOR`` default: the multiplier a saturated mark on
#: the exact query applies. Reasoning in docs/query-irrelevance.md.
DEFAULT_QUERY_IRRELEVANCE_FACTOR = 0.5


def query_irrelevance_factor() -> float:
    """``LM_QUERY_IRRELEVANCE_FACTOR`` in [0, 1]; 1.0 disables demotion.

    Unparsable, non-finite or out-of-range values fall back to the default,
    like the other LM valves.
    """

    raw = os.environ.get("LM_QUERY_IRRELEVANCE_FACTOR", "").strip()
    if not raw:
        return DEFAULT_QUERY_IRRELEVANCE_FACTOR
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_QUERY_IRRELEVANCE_FACTOR
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        return DEFAULT_QUERY_IRRELEVANCE_FACTOR
    return value


def _table_present(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (QUERY_IRRELEVANCE_TABLE,),
    ).fetchone()
    return row is not None


def _anchor_tables_present(store: MemoryStore) -> bool:
    probe = getattr(store, "_anchor_tables_present", None)
    return bool(probe()) if probe is not None else False


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------
# Write path
# ----------------------------------------------------------------------


def record_query_irrelevance(
    store: MemoryStore,
    event: Any,
    node_ids: Sequence[str],
    embedder: Any,
    *,
    now: str | None = None,
) -> str | None:
    """Accumulate one irrelevance mark per node on ``event``'s query anchor.

    ``embedder`` is the store's chunk encoder (``feedback._anchor_embedder``):
    a list of texts in, a list of vectors out. Returns the anchor id, or
    ``None`` when nothing was written — blank query, no encoder, no anchor
    tables, no nodes. Never lowers anything global.
    """

    query = str(getattr(event, "query", "") or "").strip()
    targets = list(dict.fromkeys(str(node_id) for node_id in node_ids if node_id))
    if not query or not targets or embedder is None or not _anchor_tables_present(store):
        return None
    vectors = embedder([query])
    if not vectors or not vectors[0]:
        return None
    stamp = now or _now()
    anchor = resolve_query_anchor(store, query, event.scope, vectors[0], stamp)
    conn = store.connection
    with conn:
        conn.execute(QUERY_IRRELEVANCE_SCHEMA_SQL)
        for node_id in targets:
            conn.execute(
                f"""
                INSERT INTO {QUERY_IRRELEVANCE_TABLE} (
                    anchor_id, node_id, weight, marks, cancels, last_event_id,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, 1, 0, ?, ?, ?)
                ON CONFLICT(anchor_id, node_id) DO UPDATE SET
                    weight = MIN(1.0, {QUERY_IRRELEVANCE_TABLE}.weight + excluded.weight),
                    marks = {QUERY_IRRELEVANCE_TABLE}.marks + 1,
                    last_event_id = excluded.last_event_id,
                    updated_at = excluded.updated_at
                """,
                (anchor.id, node_id, IRRELEVANCE_MARK_WEIGHT, event.id, stamp, stamp),
            )
    return anchor.id


def cancel_query_irrelevance(
    store: MemoryStore,
    anchor_id: str,
    node_ids: Iterable[str],
    *,
    now: str | None = None,
) -> int:
    """Zero the irrelevance of ``node_ids`` on ``anchor_id``; count rows cancelled.

    Called on positive credit (grounded, lookup, explicit ``used``) for the
    same anchor. A no-op without the table, which is every store that never
    saw a credited irrelevant mark.
    """

    targets = list(dict.fromkeys(str(node_id) for node_id in node_ids if node_id))
    conn = store.connection
    if not targets or not _table_present(conn):
        return 0
    stamp = now or _now()
    cancelled = 0
    with conn:
        for node_id in targets:
            cursor = conn.execute(
                f"""
                UPDATE {QUERY_IRRELEVANCE_TABLE}
                SET weight = 0.0, cancels = cancels + 1, updated_at = ?
                WHERE anchor_id = ? AND node_id = ? AND weight > 0.0
                """,
                (stamp, str(anchor_id), node_id),
            )
            cancelled += int(cursor.rowcount or 0)
    return cancelled


# ----------------------------------------------------------------------
# Read path
# ----------------------------------------------------------------------


def has_active_irrelevance(store: Any) -> bool:
    """Cheap probe: is there any live demotion row at all?"""

    try:
        conn = store.connection
        if not _table_present(conn):
            return False
        row = conn.execute(
            f"SELECT 1 FROM {QUERY_IRRELEVANCE_TABLE} WHERE weight > 0.0 LIMIT 1"
        ).fetchone()
    except sqlite3.Error:
        return False
    return row is not None


def query_demotions(
    store: Any,
    query_embedding: Sequence[float],
    scope_plan: Any,
    *,
    factor: float | None = None,
    min_similarity: float = ANCHOR_MATCH_COSINE_THRESHOLD,
    limit: int = ANCHOR_MATCH_LIMIT,
) -> dict[str, float]:
    """``{node_id: multiplier}`` for the incoming query; ``{}`` when none apply.

    ``store`` may be the retrieval layer's cached-vector face; only
    ``match_anchors`` and ``connection`` are used. Each multiplier is in
    ``[factor, 1)``.
    """

    strength_cap = 1.0 - (query_irrelevance_factor() if factor is None else factor)
    if strength_cap <= 0.0 or not query_embedding or not has_active_irrelevance(store):
        return {}
    matches = match_anchors(
        store,
        query_embedding,
        scope_plan,
        limit=limit,
        min_similarity=min_similarity,
        include_targets=False,
    )
    if not matches:
        return {}
    span = max(1e-9, 1.0 - min_similarity)
    closeness = {
        match.anchor.id: min(1.0, max(0.0, (match.similarity - min_similarity) / span))
        for match in matches
    }
    placeholders = ", ".join("?" for _ in closeness)
    rows = store.connection.execute(
        f"""
        SELECT anchor_id, node_id, weight FROM {QUERY_IRRELEVANCE_TABLE}
        WHERE weight > 0.0 AND anchor_id IN ({placeholders})
        """,
        tuple(closeness),
    ).fetchall()
    strongest: dict[str, float] = {}
    for anchor_id, node_id, weight in rows:
        strength = min(1.0, max(0.0, float(weight))) * closeness[str(anchor_id)]
        if strength > strongest.get(str(node_id), 0.0):
            strongest[str(node_id)] = strength
    return {
        node_id: 1.0 - strength_cap * strength
        for node_id, strength in strongest.items()
        if strength > 0.0
    }


def list_query_irrelevance(store: MemoryStore) -> list[Mapping[str, Any]]:
    """All rows, for tests and audits; ``[]`` without the table."""

    conn = store.connection
    if not _table_present(conn):
        return []
    cursor = conn.execute(
        f"SELECT * FROM {QUERY_IRRELEVANCE_TABLE} ORDER BY anchor_id, node_id"
    )
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]
