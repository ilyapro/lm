"""Counterfactual consumption gate: would these traces be surfaced again?

The retrospective metric next door
(:mod:`living_memory.postsession.usage_metric`) can only judge memory that has
already lived through a stretch of recall traffic. A writer's *candidate*
output has no history at all, so the only way to judge it before it is
committed is to put it into a disposable copy of the world and re-run the
traffic that actually happened.

That is this module. The protocol, end to end:

1. Freeze the source database into an immutable snapshot
   (:func:`living_memory.retrieval_harness.create_snapshot`, the SQLite backup
   API) and take a scratch :func:`~living_memory.retrieval_harness.working_copy`
   of it. Every mutation below happens on that disposable copy.
2. **Remove** the candidate cohort from the working copy, together with every
   row that references it, and assert the copy is still internally coherent.
3. **Re-insert** the cohort through the store's own write path and assert the
   round trip was lossless — the working copy must be row-equivalent to what it
   was before step 2.
4. **Replay** the recorded post-cutoff recall requests end to end through the
   *current* retrieval code, via
   :func:`living_memory.retrieval_harness.run_item`, which calls
   ``memory_recall(..., log_access=False, log_event=False)`` so the replay
   cannot pollute the event log it is reading from.
5. Measure how often the cohort surfaces in the **delivered** results, plus the
   grounded variant, under the identical definitions the retrospective metric
   uses.

Why remove and then re-insert at all
------------------------------------

Because that is the operation the gate performs on a real candidate: the
extractor's traces are not in the database, and the gate has to put them there
before it can ask whether retrieval finds them. Validating the gate on organic
memory therefore means pushing organic nodes through *the same* insertion path
— otherwise the organic control and the judged candidate would be treated
differently and the comparison would prove nothing. The removal also makes the
surgery falsifiable: :func:`assert_cohort_restored` compares a per-node digest
taken before removal against one taken after re-insertion, so a lossy excision
fails loudly instead of quietly deflating the score.

``reinsert_mode`` picks what "the same insertion path" preserves:

``faithful``
    Restore the node exactly — stats, decay state, connections, anchor edges
    and the ``recall_events.feedback_trace_id`` back-references. The round trip
    is then an identity, which is what makes the organic self-check a test of
    *replay* rather than a test of how much a node's history is worth.
``fresh``
    Re-insert with no accumulated history: no ``access_count``, no
    ``usefulness_score``, no graph edges — the state a newly written trace
    actually has. This is the arm a candidate cohort is judged in; run against
    the organic cohort it prices what history is worth.

Content-derived signals are restored in both modes (chunk embeddings are a pure
function of the content, and a candidate trace gets embedded too); only
history-derived signals differ.

One protocol, not two
---------------------

Cohort selection, the consumption definition and the grounded variant are all
imported from :mod:`~living_memory.postsession.usage_metric` and never
re-implemented here, so "retrospective rate" and "counterfactual rate" are the
same arithmetic over two different sets of delivered results. That is the whole
point: the two numbers are only comparable if nothing but the deliveries
differs.

Which traffic you replay is part of the answer
----------------------------------------------

Measured on the 2026-08-19 field database against the held-out organic cohort
``nodes.created_at in [2026-08-12, 2026-08-15)`` (949 nodes):

============================  ==============  ==============  ======
replayed traffic              retrospective   counterfactual  gap
============================  ==============  ==============  ======
post-cutoff (from 08-15)      13.2%           11.6%           1.6 pt
contemporaneous (from 08-12)  43.7%           30.8%           13.0 pt
============================  ==============  ==============  ======

Same cohort, same code, same definitions — only the replayed event set differs.
The wider window replays requests recorded *while* the cohort was being written,
against today's larger corpus and today's ranker, and reproduces 57% of their
recorded deliveries against 61% for the post-cutoff window. That sensitivity is
a property of the method, so ``diagnostic_replay_since`` runs a second window
and publishes it under ``diagnostics.alternate_replay`` rather than leaving it
to whoever happened to run the tool twice.

SAFETY. ``MemoryStore.__init__`` migrates and writes whatever file it opens, so
it is *only* ever pointed at the scratch working copy. The source database is
read through ``mode=ro`` URIs and the SQLite backup API. A plain ``cp`` of the
live file would silently drop its WAL — half the database — which is why
:func:`~living_memory.retrieval_harness.backup_database` is used instead.

Nothing here imports the extractor, the judge or any transcript code.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from living_memory.chunking import TextChunk
from living_memory.config import MemoryConfig
from living_memory.grounding import DEFAULT_MIN_CONTAINMENT
from living_memory.postsession.usage_metric import (
    DEFAULT_WITHIN_DAYS,
    CohortNode,
    CohortSelector,
    ConsumptionIndex,
    PrivacyGuardError,
    check_privacy,
    load_cohort_nodes,
    load_consumption_index,
    measure_cohort,
    normalize_instant,
    parse_timestamp,
)

# The grounded block and the two rounding helpers are deliberately imported
# rather than rewritten. The contract for this module is that the retrospective
# and counterfactual rates are computed under ONE definition; a second copy of
# the denominator rule ("divide by nodes_with_closed_consumer, not by the
# cohort") is exactly the fork that would make the two numbers incomparable.
from living_memory.postsession.usage_metric import (  # noqa: PLC2701
    _grounded_block,
    _pct,
    _ratio,
)
from living_memory.replay import (
    ReplayEvent,
    ReplayResult,
    apply_grounding,
    build_label_data,
    open_readonly,
)
from living_memory.retrieval import MemoryRecallService
from living_memory.retrieval_harness import (
    GoldsetItem,
    HarnessRun,
    frozen_snapshot,
    run_item,
    working_copy,
)
from living_memory.storage import (
    CHUNK_EMBEDDING_TABLE,
    QUERY_ANCHOR_EDGE_TABLE,
    MemoryStore,
    unpack_chunk_embedding,
)

__all__ = [
    "DEFAULT_MIN_CONTAINMENT",
    "REINSERT_MODES",
    "SELFCHECK_MAX_GAP",
    "CoherenceError",
    "CohortSurgery",
    "CounterfactualConfig",
    "NodeSnapshot",
    "PrivacyGuardError",
    "ReplayRequest",
    "assert_cohort_absent",
    "assert_cohort_restored",
    "assert_database_coherent",
    "build_replay_requests",
    "capture_cohort",
    "check_privacy",
    "consumption_block",
    "counterfactual_index",
    "foreign_key_violations",
    "load_request_specs",
    "query_digest",
    "reinsert_cohort",
    "remove_cohort",
    "replay_requests",
    "run_counterfactual",
    "subset_index",
]

#: The two ways a removed cohort can come back. See the module docstring.
REINSERT_MODES = ("faithful", "fresh")

#: The self-check bar: a counterfactual harness that cannot land within this
#: many rate points of a known organic number is not trusted to judge anything.
SELFCHECK_MAX_GAP = 0.10

#: Supplied to ``create_node`` itself, so already correct after re-insertion.
#: Every *other* ``nodes`` column is stamped fresh by the write path and has to
#: be put back — and the set is read from the schema rather than listed here,
#: because the field database and a fresh one disagree about it: the legacy
#: ``nodes.embedding`` column still exists until an operator drops it, and a
#: hard-coded list would silently fail to restore whatever it does not name.
_INSERT_SUPPLIED_COLUMNS = frozenset({"id", "level", "content"})

#: The columns that carry *accumulated history* rather than content or identity:
#: exactly the inputs to ``feedback.feedback_weighted_score``. A ``fresh``
#: re-insertion leaves these at what the write path computed for a brand new
#: node, which is the state a candidate trace really arrives in.
#:
#: ``created_at`` is deliberately NOT here. Consumption is defined as "delivered
#: by a strictly later event", so rewriting it to today would make every
#: replayed query *earlier* than the node and silently zero the measurement.
_HISTORY_COLUMNS = frozenset(
    {"access_count", "last_accessed", "usefulness_score", "confidence", "unique_agents"}
)

_SQL_PARAM_CHUNK = 900

#: Replayed requests are recorded recall events, which is what the harness's
#: ``content_grounded`` stratum already means; reusing the name keeps the
#: goldset vocabulary single-valued.
_STRATUM = "content_grounded"


class CoherenceError(RuntimeError):
    """The working copy is not internally consistent — the run must not report."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _chunked(values: Sequence[str], size: int = _SQL_PARAM_CHUNK) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def query_digest(text: str) -> str:
    """Stable short hash of a query.

    Published artifacts carry aggregates only; a query never appears verbatim,
    so identity across runs is carried by this digest instead.
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cohort capture / removal / re-insertion
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NodeSnapshot:
    """Everything about one node that removal has to be able to put back.

    The digest deliberately excludes surrogate keys the store re-mints on
    write — a chunk row's ``id`` and its ``created_at``/``updated_at`` — because
    :meth:`MemoryStore.replace_node_chunks` allocates them and no retrieval
    decision reads them. Every column that does affect retrieval is compared.
    """

    node_id: str
    row: dict[str, Any]
    chunks: tuple[tuple[Any, ...], ...]
    connections: tuple[tuple[Any, ...], ...]
    anchor_edges: tuple[tuple[Any, ...], ...]
    closed_events: tuple[str, ...]

    def digest(self) -> tuple[Any, ...]:
        return (
            tuple(sorted(self.row.items())),
            self.chunks,
            self.connections,
            self.anchor_edges,
            self.closed_events,
        )


@dataclass(slots=True)
class CohortSurgery:
    """What the remove/re-insert round trip touched, for the audit trail."""

    nodes: int = 0
    chunk_rows: int = 0
    connection_rows: int = 0
    anchor_edge_rows: int = 0
    closed_event_refs: int = 0
    dedup_connections_repaired: int = 0
    collateral_decay_repaired: int = 0
    reinsert_mode: str = "faithful"

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "chunk_rows": self.chunk_rows,
            "connection_rows": self.connection_rows,
            "anchor_edge_rows": self.anchor_edge_rows,
            "closed_event_refs": self.closed_event_refs,
            "dedup_connections_repaired": self.dedup_connections_repaired,
            "collateral_decay_repaired": self.collateral_decay_repaired,
            "reinsert_mode": self.reinsert_mode,
        }


def _chunk_rows(connection: sqlite3.Connection, node_id: str) -> tuple[tuple[Any, ...], ...]:
    rows = connection.execute(
        f"""
        SELECT chunk_index, dimensions, embedding, token_start, token_end, content_fingerprint
        FROM {CHUNK_EMBEDDING_TABLE} WHERE node_id = ? ORDER BY chunk_index
        """,
        (node_id,),
    ).fetchall()
    return tuple(
        (
            int(row["chunk_index"]),
            int(row["dimensions"]),
            bytes(row["embedding"]),
            int(row["token_start"]),
            int(row["token_end"]),
            str(row["content_fingerprint"] or ""),
        )
        for row in rows
    )


def _connection_rows(connection: sqlite3.Connection, node_id: str) -> tuple[tuple[Any, ...], ...]:
    rows = connection.execute(
        """
        SELECT id, source_id, target_id, type, weight, metadata, created_at, updated_at
        FROM connections WHERE source_id = ? OR target_id = ? ORDER BY id
        """,
        (node_id, node_id),
    ).fetchall()
    return tuple(tuple(row) for row in rows)


def _anchor_edge_rows(connection: sqlite3.Connection, node_id: str) -> tuple[tuple[Any, ...], ...]:
    rows = connection.execute(
        f"""
        SELECT anchor_id, target_id, weight, hits, created_at, updated_at
        FROM {QUERY_ANCHOR_EDGE_TABLE} WHERE target_id = ? ORDER BY anchor_id
        """,
        (node_id,),
    ).fetchall()
    return tuple(tuple(row) for row in rows)


def capture_cohort(
    connection: sqlite3.Connection, node_ids: Sequence[str]
) -> dict[str, NodeSnapshot]:
    """Read back everything the cohort owns, before anything is touched."""

    captured: dict[str, NodeSnapshot] = {}
    for node_id in node_ids:
        row = connection.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            raise CoherenceError(f"cohort node missing: {node_id}")
        data = {key: row[key] for key in row.keys()}
        if not str(data.get("content") or ""):
            raise CoherenceError(f"cohort node has empty content, cannot round-trip: {node_id}")
        closed = [
            str(event["id"])
            for event in connection.execute(
                "SELECT id FROM recall_events WHERE feedback_trace_id = ? ORDER BY id",
                (node_id,),
            )
        ]
        captured[node_id] = NodeSnapshot(
            node_id=node_id,
            row=data,
            chunks=_chunk_rows(connection, node_id),
            connections=_connection_rows(connection, node_id),
            anchor_edges=_anchor_edge_rows(connection, node_id),
            closed_events=tuple(closed),
        )
    return captured


def _fingerprint_neighbourhood(
    connection: sqlite3.Connection, snapshots: Mapping[str, NodeSnapshot]
) -> dict[str, tuple[Any, ...]]:
    """Decay state of every node sharing a cohort content fingerprint.

    ``MemoryStore._insert_node`` supersedes-and-decays any live trace with the
    same ``(scope, content_fingerprint)`` as the row being written. Re-inserting
    the cohort can therefore fire that rule against a bystander, or against an
    earlier cohort member, which would perturb the very corpus the replay is
    measured on. This is the bounded set where that can happen, so it is the set
    the repair pass watches.
    """

    fingerprints = sorted(
        {str(snap.row.get("content_fingerprint") or "") for snap in snapshots.values()} - {""}
    )
    state: dict[str, tuple[Any, ...]] = {}
    for chunk in _chunked(fingerprints):
        placeholders = ",".join("?" * len(chunk))
        rows = connection.execute(
            "SELECT id, decayed, decay_reason, updated_at FROM nodes "
            f"WHERE content_fingerprint IN ({placeholders})",
            list(chunk),
        )
        for row in rows:
            state[str(row["id"])] = (
                int(row["decayed"]),
                row["decay_reason"],
                str(row["updated_at"]),
            )
    return state


def _connection_ids_touching(connection: sqlite3.Connection, node_ids: Sequence[str]) -> set[str]:
    found: set[str] = set()
    for chunk in _chunked(list(node_ids)):
        placeholders = ",".join("?" * len(chunk))
        rows = connection.execute(
            f"SELECT id FROM connections "
            f"WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
            [*chunk, *chunk],
        )
        found.update(str(row["id"]) for row in rows)
    return found


def remove_cohort(store: MemoryStore, snapshots: Mapping[str, NodeSnapshot]) -> CohortSurgery:
    """Excise the cohort and every row that references it, coherently.

    Order is forced by ``PRAGMA foreign_keys = ON``: the referencing rows go
    first, the ``nodes`` row last. ``nodes_fts`` and its shadow tables
    (``nodes_fts_config/content/data/docsize/idx``) need no explicit handling —
    the schema's ``AFTER DELETE ON nodes`` trigger keeps them in step, which is
    exactly why the delete goes through the store's own connection rather than
    around it. Chunks go through :meth:`MemoryStore.delete_node_chunks` for the
    same reason.
    """

    surgery = CohortSurgery(nodes=len(snapshots))
    connection = store.connection
    for node_id in snapshots:
        with connection:
            # A cohort node may itself be the trace that closed a recall event.
            # The reference has to be released before the row can go, and it is
            # restored on the way back in — the grounded denominator depends on
            # it.
            cursor = connection.execute(
                "UPDATE recall_events SET feedback_trace_id = NULL WHERE feedback_trace_id = ?",
                (node_id,),
            )
            surgery.closed_event_refs += int(cursor.rowcount or 0)
            cursor = connection.execute(
                f"DELETE FROM {QUERY_ANCHOR_EDGE_TABLE} WHERE target_id = ?", (node_id,)
            )
            surgery.anchor_edge_rows += int(cursor.rowcount or 0)
            cursor = connection.execute(
                "DELETE FROM connections WHERE source_id = ? OR target_id = ?",
                (node_id, node_id),
            )
            surgery.connection_rows += int(cursor.rowcount or 0)
        surgery.chunk_rows += store.delete_node_chunks(node_id)
        with connection:
            connection.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
    return surgery


def _restore_column_names(connection: sqlite3.Connection, mode: str) -> tuple[str, ...]:
    """Which ``nodes`` columns re-insertion has to put back, read from the schema."""

    skip = set(_INSERT_SUPPLIED_COLUMNS)
    if mode == "fresh":
        skip |= _HISTORY_COLUMNS
    return tuple(
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(nodes)")
        if str(row["name"]) not in skip
    )


def _restore_columns(
    connection: sqlite3.Connection, snapshot: NodeSnapshot, columns: Sequence[str]
) -> None:
    assignments = ", ".join(f"{column} = ?" for column in columns)
    connection.execute(
        f"UPDATE nodes SET {assignments} WHERE id = ?",
        [*(snapshot.row[column] for column in columns), snapshot.node_id],
    )


def _restore_chunks(store: MemoryStore, snapshot: NodeSnapshot) -> int:
    """Put the node's chunk vectors back through the store's chunk API.

    The vectors are a deterministic function of the content, so restoring them
    is the same corpus a re-embed would produce, without paying the model for
    every cohort node. A candidate trace, which has none, is embedded by the
    ordinary lazy drain on the read path.
    """

    if not snapshot.chunks:
        return 0
    chunks: list[tuple[TextChunk, list[float]]] = []
    fingerprint = str(snapshot.row.get("content_fingerprint") or "")
    for index, dimensions, blob, token_start, token_end, chunk_fingerprint in snapshot.chunks:
        chunks.append(
            (
                TextChunk(
                    text="",
                    chunk_index=index,
                    token_start=token_start,
                    token_end=token_end,
                    char_start=0,
                    char_end=0,
                ),
                unpack_chunk_embedding(blob, dimensions),
            )
        )
        fingerprint = chunk_fingerprint or fingerprint
    return store.replace_node_chunks(snapshot.node_id, chunks, content_fingerprint=fingerprint)


def reinsert_cohort(
    store: MemoryStore,
    snapshots: Mapping[str, NodeSnapshot],
    *,
    mode: str = "faithful",
    baseline_state: Mapping[str, tuple[Any, ...]] | None = None,
    baseline_connection_ids: set[str] | None = None,
    surgery: CohortSurgery | None = None,
) -> CohortSurgery:
    """Put the cohort back through ``MemoryStore.create_node``.

    ``create_node`` is the store's real write path — the same one a candidate
    trace would take — so the FTS row is rebuilt by the schema trigger and the
    content fingerprint is recomputed rather than trusted. Two of its documented
    side effects have to be undone afterwards, and both are counted rather than
    hidden: it stamps ``created_at``/``updated_at`` with *now*, and it
    supersedes-and-decays any live trace sharing the fingerprint it is writing.
    """

    if mode not in REINSERT_MODES:
        raise ValueError(f"unknown reinsert mode {mode!r}; expected one of {REINSERT_MODES}")
    surgery = surgery or CohortSurgery(nodes=len(snapshots))
    surgery.reinsert_mode = mode
    connection = store.connection
    columns = _restore_column_names(connection, mode)

    ordered = sorted(
        snapshots.values(), key=lambda snap: (str(snap.row["created_at"]), snap.node_id)
    )
    for snapshot in ordered:
        raw_context = snapshot.row.get("context")
        try:
            context = json.loads(raw_context) if isinstance(raw_context, str) else {}
        except ValueError:
            context = {}
        store.create_node(
            level=str(snapshot.row["level"]),
            content=str(snapshot.row["content"]),
            context=context if isinstance(context, dict) else {},
            node_id=snapshot.node_id,
            timestamp=str(snapshot.row["timestamp"]),
        )
        with connection:
            _restore_columns(connection, snapshot, columns)
        _restore_chunks(store, snapshot)

    with connection:
        # The back-references are a property of the *event*, not of the node's
        # ranking history, so both modes restore them: without them the grounded
        # variant would be measured against a different denominator than the
        # retrospective arm and the two would not be comparable.
        for snapshot in ordered:
            for event_id in snapshot.closed_events:
                connection.execute(
                    "UPDATE recall_events SET feedback_trace_id = ? WHERE id = ?",
                    (snapshot.node_id, event_id),
                )
        if mode == "faithful":
            for snapshot in ordered:
                for row in snapshot.connections:
                    connection.execute(
                        "INSERT OR IGNORE INTO connections (id, source_id, target_id, type, "
                        "weight, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        list(row),
                    )
                for row in snapshot.anchor_edges:
                    connection.execute(
                        f"INSERT OR IGNORE INTO {QUERY_ANCHOR_EDGE_TABLE} "
                        "(anchor_id, target_id, weight, hits, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        list(row),
                    )

    _repair_write_path_side_effects(
        store, snapshots, baseline_state, baseline_connection_ids, surgery, mode=mode
    )
    return surgery


def _repair_write_path_side_effects(
    store: MemoryStore,
    snapshots: Mapping[str, NodeSnapshot],
    baseline_state: Mapping[str, tuple[Any, ...]] | None,
    baseline_connection_ids: set[str] | None,
    surgery: CohortSurgery,
    *,
    mode: str,
) -> None:
    """Undo the supersedes-and-decay the write path fires on duplicate content.

    A duplicate-content supersedes edge always has the row being written as its
    source, so every edge this rule can mint touches a cohort id — which bounds
    the repair to the cohort's own connection neighbourhood.
    """

    connection = store.connection
    watched = sorted(snapshots)
    if baseline_connection_ids is not None:
        current = _connection_ids_touching(connection, watched)
        # ``fresh`` keeps no edges of its own, so its baseline is the empty set:
        # anything present now is collateral from the duplicate rule.
        expected = baseline_connection_ids if mode == "faithful" else set()
        surplus = sorted(current - expected)
        if surplus:
            with connection:
                for chunk in _chunked(surplus):
                    placeholders = ",".join("?" * len(chunk))
                    connection.execute(
                        f"DELETE FROM connections WHERE id IN ({placeholders})", list(chunk)
                    )
            surgery.dedup_connections_repaired += len(surplus)

    if not baseline_state:
        return
    repaired = 0
    with connection:
        for node_id, (decayed, reason, updated_at) in sorted(baseline_state.items()):
            row = connection.execute(
                "SELECT decayed, decay_reason, updated_at FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()
            if row is None:
                continue
            if (int(row["decayed"]), row["decay_reason"], str(row["updated_at"])) == (
                decayed,
                reason,
                updated_at,
            ):
                continue
            connection.execute(
                "UPDATE nodes SET decayed = ?, decay_reason = ?, updated_at = ? WHERE id = ?",
                (decayed, reason, updated_at, node_id),
            )
            repaired += 1
    surgery.collateral_decay_repaired += repaired


# ---------------------------------------------------------------------------
# Coherence assertions
# ---------------------------------------------------------------------------


def foreign_key_violations(connection: sqlite3.Connection) -> set[tuple[Any, ...]]:
    """Every ``PRAGMA foreign_key_check`` row, as a comparable key.

    The field database carries a standing set of these — 35 on the 2026-08-19
    snapshot, all ``recall_events.feedback_trace_id`` pointing at traces that
    were hard-deleted before the constraint existed. ``PRAGMA foreign_keys=ON``
    only polices new writes, so those rows are simply history. Demanding zero
    would make the harness unusable on the very database it exists to measure;
    what it must actually guarantee is that *it* adds none, which is why the
    baseline is captured and subtracted instead of assumed empty.
    """

    return {
        (str(row[0]), row[1], str(row[2]), row[3])
        for row in connection.execute("PRAGMA foreign_key_check")
    }


def assert_database_coherent(
    connection: sqlite3.Connection,
    *,
    stage: str,
    fts_check_connection: sqlite3.Connection | None = None,
    baseline_violations: set[tuple[Any, ...]] | None = None,
) -> dict[str, Any]:
    """Foreign keys, the FTS index and its shadow tables all agree.

    ``nodes_fts`` is an ordinary (not external-content) FTS5 table kept in step
    by triggers, so the two failure modes are a stale row whose ``rowid`` no
    longer names a node, and a corrupt shadow table. The first is a join, the
    second is FTS5's own ``integrity-check``, which walks
    ``nodes_fts_config/content/data/docsize/idx`` and is issued as a write
    statement — hence the separate writable connection.
    """

    violations = foreign_key_violations(connection)
    baseline = baseline_violations if baseline_violations is not None else set()
    introduced = violations - baseline
    if introduced:
        raise CoherenceError(
            f"{stage}: the harness introduced {len(introduced)} foreign-key violation(s)"
        )
    orphans = int(
        connection.execute(
            "SELECT count(*) FROM nodes_fts f LEFT JOIN nodes n ON n.rowid = f.rowid "
            "WHERE n.id IS NULL"
        ).fetchone()[0]
    )
    if orphans:
        raise CoherenceError(f"{stage}: {orphans} nodes_fts row(s) name no node")
    node_count = int(connection.execute("SELECT count(*) FROM nodes").fetchone()[0])
    fts_count = int(connection.execute("SELECT count(*) FROM nodes_fts").fetchone()[0])
    if node_count != fts_count:
        raise CoherenceError(f"{stage}: nodes_fts holds {fts_count} rows for {node_count} nodes")
    stale_chunks = int(
        connection.execute(
            f"SELECT count(*) FROM {CHUNK_EMBEDDING_TABLE} e "
            "LEFT JOIN nodes n ON n.id = e.node_id WHERE n.id IS NULL"
        ).fetchone()[0]
    )
    if stale_chunks:
        raise CoherenceError(f"{stage}: {stale_chunks} orphan chunk embedding row(s)")
    fts_checked = False
    if fts_check_connection is not None:
        try:
            fts_check_connection.execute(
                "INSERT INTO nodes_fts(nodes_fts) VALUES('integrity-check')"
            )
        except sqlite3.DatabaseError as exc:  # pragma: no cover - corruption guard
            raise CoherenceError(f"{stage}: nodes_fts integrity-check failed ({exc})") from exc
        fts_checked = True
    return {
        "stage": stage,
        "nodes": node_count,
        "nodes_fts": fts_count,
        "preexisting_foreign_key_violations": len(violations & baseline)
        if baseline_violations is not None
        else len(violations),
        "foreign_key_violations_introduced": 0,
        "orphan_fts_rows": 0,
        "orphan_chunk_rows": 0,
        "fts_integrity_checked": fts_checked,
    }


def assert_cohort_absent(connection: sqlite3.Connection, node_ids: Sequence[str]) -> dict[str, Any]:
    """Nothing anywhere still points at the removed cohort."""

    checks = {
        "nodes": "SELECT count(*) FROM nodes WHERE id IN ({p})",
        "nodes_fts": "SELECT count(*) FROM nodes_fts WHERE node_id IN ({p})",
        "node_chunk_embeddings": (
            f"SELECT count(*) FROM {CHUNK_EMBEDDING_TABLE} WHERE node_id IN ({{p}})"
        ),
        "connections": (
            "SELECT count(*) FROM connections WHERE source_id IN ({p}) OR target_id IN ({p})"
        ),
        "query_anchor_edges": (
            f"SELECT count(*) FROM {QUERY_ANCHOR_EDGE_TABLE} WHERE target_id IN ({{p}})"
        ),
        "recall_events_feedback_trace_id": (
            "SELECT count(*) FROM recall_events WHERE feedback_trace_id IN ({p})"
        ),
    }
    report: dict[str, Any] = {}
    ids = list(node_ids)
    for name, template in checks.items():
        total = 0
        for chunk in _chunked(ids):
            placeholders = ",".join("?" * len(chunk))
            params = list(chunk) * template.count("{p}")
            total += int(connection.execute(template.format(p=placeholders), params).fetchone()[0])
        if total:
            raise CoherenceError(f"after remove: {total} surviving reference(s) in {name}")
        report[name] = 0
    return report


def assert_cohort_restored(
    connection: sqlite3.Connection, snapshots: Mapping[str, NodeSnapshot]
) -> dict[str, Any]:
    """The round trip was lossless: re-captured digests equal the originals.

    This is the falsifiable part of the surgery. If removal dropped a column, a
    chunk vector or an edge, the counterfactual would score a *different* corpus
    than the retrospective arm and the comparison between them would be
    meaningless — so the run stops here instead.
    """

    recaptured = capture_cohort(connection, sorted(snapshots))
    mismatched = [
        node_id
        for node_id, snapshot in sorted(snapshots.items())
        if recaptured[node_id].digest() != snapshot.digest()
    ]
    if mismatched:
        raise CoherenceError(
            f"after re-insert: {len(mismatched)} node(s) did not round-trip "
            f"(first: {mismatched[0]})"
        )
    return {"nodes_compared": len(snapshots), "mismatched": 0}


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplayRequest:
    """One recorded retrieval request, ready to be re-run.

    Requests are keyed on everything ``memory_recall`` actually reads — query,
    requested scope, ambient context, depth and ``max_results`` — so two events
    that would take the same code path on the same corpus are executed once and
    share the result. Anything that could change the ranking is part of the key.
    """

    key: str
    item: GoldsetItem
    event_ordinals: tuple[int, ...]


def load_request_specs(
    connection: sqlite3.Connection, event_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """The retrieval arguments each recorded event was served with."""

    specs: dict[str, dict[str, Any]] = {}
    for chunk in _chunked(list(event_ids)):
        placeholders = ",".join("?" * len(chunk))
        rows = connection.execute(
            "SELECT id, query, scope, requested_scope, depth, max_results, ambient_context "
            f"FROM recall_events WHERE id IN ({placeholders})",
            list(chunk),
        )
        for row in rows:
            try:
                ambient = json.loads(row["ambient_context"] or "{}")
            except ValueError:
                ambient = {}
            specs[str(row["id"])] = {
                "query": str(row["query"] or ""),
                "scope": str(row["requested_scope"] or row["scope"] or "") or None,
                "depth": row["depth"],
                "max_results": int(row["max_results"] or 0),
                "ambient_context": ambient if isinstance(ambient, dict) else {},
            }
    return specs


def build_replay_requests(
    index: ConsumptionIndex,
    ordinals: Sequence[int],
    specs: Mapping[str, Mapping[str, Any]],
) -> tuple[list[ReplayRequest], dict[str, int]]:
    """Turn recorded events into deduplicated, replayable goldset items."""

    grouped: dict[str, list[int]] = {}
    payloads: dict[str, Mapping[str, Any]] = {}
    stats = {"events": len(ordinals), "missing_spec": 0, "unusable_request": 0, "requests": 0}
    for ordinal in ordinals:
        event = index.events[ordinal]
        spec = specs.get(event.id)
        if spec is None:
            stats["missing_spec"] += 1
            continue
        if not str(spec["query"]).strip() or int(spec["max_results"]) <= 0:
            stats["unusable_request"] += 1
            continue
        key = hashlib.sha256(
            json.dumps(
                [
                    spec["query"],
                    spec["scope"],
                    spec["depth"],
                    spec["max_results"],
                    spec["ambient_context"],
                ],
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:24]
        grouped.setdefault(key, []).append(ordinal)
        payloads.setdefault(key, spec)

    requests: list[ReplayRequest] = []
    for key in sorted(grouped):
        spec = payloads[key]
        first = index.events[grouped[key][0]]
        requests.append(
            ReplayRequest(
                key=key,
                item=GoldsetItem(
                    query_id=key,
                    query=str(spec["query"]),
                    scope=spec["scope"],
                    ambient_context=dict(spec["ambient_context"]) or None,
                    depth=spec["depth"],
                    max_results=int(spec["max_results"]),
                    relevant_node_ids=tuple(sorted({r.node_id for r in first.results}))
                    or (first.id,),
                    stratum=_STRATUM,
                    tail=False,
                    source_event_id=first.id,
                    provenance={"query_sha256_16": query_digest(str(spec["query"]))},
                ),
                event_ordinals=tuple(grouped[key]),
            )
        )
    stats["requests"] = len(requests)
    return requests, stats


def replay_requests(
    service: MemoryRecallService,
    requests: Sequence[ReplayRequest],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, HarnessRun]:
    """Re-run every request through the harness's read-only entry point.

    :func:`living_memory.retrieval_harness.run_item` is used rather than a bare
    ``memory_recall`` call precisely because it pins
    ``log_access=False, log_event=False``: a replay that logged its own recalls
    would write new rows into the very ``recall_events`` table the measurement
    reads, and each pass would change the next one's answer.
    """

    runs: dict[str, HarnessRun] = {}
    total = len(requests)
    for position, request in enumerate(requests, start=1):
        runs[request.key] = run_item(service, request.item)
        if progress is not None:
            progress(position, total)
    return runs


def _as_replay_results(run: HarnessRun) -> list[ReplayResult]:
    return [
        ReplayResult(
            node_id=result.node_id,
            rank=result.rank,
            level=result.level,
            scope=result.scope,
            score=result.score,
            bm25_score=result.bm25_score,
            vector_score=result.vector_score,
            graph_score=result.graph_score,
            trigger_score=result.trigger_score,
        )
        for result in run.results
    ]


def _rebuild_index(events: Sequence[ReplayEvent], stats: Mapping[str, int]) -> ConsumptionIndex:
    event_times = [parse_timestamp(event.created_at) for event in events]
    by_node: dict[str, list[tuple[str, int]]] = {}
    for ordinal, event in enumerate(events):
        for result in event.results:
            by_node.setdefault(result.node_id, []).append((event.created_at, ordinal))
    for entries in by_node.values():
        entries.sort()
    merged = dict(stats)
    merged["distinct_delivered_nodes"] = len(by_node)
    return ConsumptionIndex(
        events=list(events), event_times=event_times, by_node=by_node, stats=merged
    )


def subset_index(index: ConsumptionIndex, ordinals: Sequence[int]) -> ConsumptionIndex:
    """The same index restricted to a subset of its events, ordinals rebuilt."""

    events = [index.events[ordinal] for ordinal in ordinals]
    return _rebuild_index(
        events,
        {
            "events_scanned": len(events),
            "events_with_deliveries": len(events),
            "events_closed_by_trace": sum(1 for event in events if event.feedback_trace_id),
        },
    )


def counterfactual_index(
    index: ConsumptionIndex,
    requests: Sequence[ReplayRequest],
    runs: Mapping[str, HarnessRun],
) -> ConsumptionIndex:
    """The replayed deliveries, wearing each recorded event's identity.

    Each event keeps its own ``created_at`` and ``feedback_trace_id`` — the
    consumption definition is "delivered by a *strictly later* event", and the
    grounded variant needs the trace that closed it — while its ``results`` are
    what the current retrieval code returns today.
    """

    events: list[ReplayEvent] = []
    for request in requests:
        run = runs.get(request.key)
        if run is None:
            continue
        for ordinal in request.event_ordinals:
            recorded = index.events[ordinal]
            events.append(
                ReplayEvent(
                    id=recorded.id,
                    created_at=recorded.created_at,
                    scope=request.item.scope or "",
                    requested_scope=request.item.scope or "",
                    resolved_scopes=(),
                    depth=None,
                    max_results=request.item.max_results,
                    query="",
                    feedback_trace_id=recorded.feedback_trace_id,
                    feedback_applied_at=None,
                    results=_as_replay_results(run),
                )
            )
    events.sort(key=lambda event: (event.created_at, event.id))
    return _rebuild_index(
        events,
        {
            "events_scanned": len(events),
            "events_with_deliveries": sum(1 for event in events if event.results),
            "events_closed_by_trace": sum(1 for event in events if event.feedback_trace_id),
        },
    )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def consumption_block(
    connection: sqlite3.Connection,
    index: ConsumptionIndex,
    nodes: Sequence[CohortNode],
    selector: CohortSelector,
    *,
    min_containment: float = DEFAULT_MIN_CONTAINMENT,
    within_days: int = DEFAULT_WITHIN_DAYS,
) -> dict[str, Any]:
    """One arm's consumption figures, under the retrospective metric's rules.

    ``measure_cohort`` and ``_grounded_block`` come from
    :mod:`~living_memory.postsession.usage_metric` unchanged, so the only thing
    that distinguishes the retrospective arm from the counterfactual arm is
    which deliveries the index was built from. ``_consumed`` carries the node
    ids out for the per-node agreement diagnostic and is stripped before the
    report is published.
    """

    measurement = measure_cohort(index, nodes, selector, within_days=within_days)
    relevant = sorted(measurement.closed_consuming_events)
    grounding: dict[str, Any] | None = None
    if relevant:
        subset = [index.events[ordinal] for ordinal in relevant]
        label_data = build_label_data(connection, subset)
        grounding = apply_grounding(subset, label_data, min_containment)
        grounding["corpus_events"] = len(subset)
    grounded = _grounded_block(
        measurement, index, min_containment=min_containment, within_days=within_days
    )
    return {
        "nodes": measurement.nodes,
        "consumed_later": measurement.consumed_later,
        "consumed_within_7d": measurement.consumed_within_7d,
        "rate": _ratio(measurement.consumed_later, measurement.nodes),
        "rate_pct": _pct(measurement.consumed_later, measurement.nodes),
        "within_7d_rate": _ratio(measurement.consumed_within_7d, measurement.nodes),
        "within_days": within_days,
        "grounded": grounded,
        "grounding_diagnostics": grounding,
        "_consumed": {node.id for node in nodes if index.later_deliveries(node)},
    }


def _agreement(retro: set[str], counter: set[str], cohort: int) -> dict[str, Any]:
    """How much the two arms agree *per node*, not just in aggregate.

    Two rates can coincide while naming disjoint nodes, which would mean the
    replay reproduced the number by accident. This is the guard against that.
    """

    both = len(retro & counter)
    union = len(retro | counter)
    return {
        "retrospective_consumed": len(retro),
        "counterfactual_consumed": len(counter),
        "both": both,
        "retrospective_only": len(retro - counter),
        "counterfactual_only": len(counter - retro),
        "jaccard": _ratio(both, union) if union else None,
        "recall_of_retrospective": _ratio(both, len(retro)) if retro else None,
        "precision_vs_retrospective": _ratio(both, len(counter)) if counter else None,
        "cohort_nodes": cohort,
    }


def _delivery_fidelity(index: ConsumptionIndex, counter: ConsumptionIndex) -> dict[str, Any]:
    """Corpus-wide replay fidelity: does the replay reproduce recorded results?

    Independent of the cohort, and the reason a degenerate replay — one that
    returns nothing, or everything — cannot pass the self-check by coincidence:
    the recorded and replayed delivery sets are compared event by event over the
    whole corpus, not just the cohort.
    """

    recorded_by_id = {
        event.id: {result.node_id for result in event.results} for event in index.events
    }
    matched = 0
    recorded_total = 0
    replayed_total = 0
    events = 0
    for event in counter.events:
        recorded = recorded_by_id.get(event.id)
        if recorded is None:
            continue
        replayed = {result.node_id for result in event.results}
        events += 1
        matched += len(recorded & replayed)
        recorded_total += len(recorded)
        replayed_total += len(replayed)
    return {
        "events": events,
        "recorded_delivered": recorded_total,
        "replayed_delivered": replayed_total,
        "reproduced": matched,
        "reproduced_share_of_recorded": _ratio(matched, recorded_total),
        "reproduced_share_of_replayed": _ratio(matched, replayed_total),
    }


def _gap(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(abs(left - right), 4)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CounterfactualConfig:
    """Everything a run needs, so the caller's intent is one auditable object."""

    window: tuple[str, str]
    replay_since: str
    as_of: str
    cohort_rule: str
    cohort_kind: str = "organic_holdout"
    agent: str | None = None
    agent_is_null: bool = False
    context_key: tuple[str, str | None] | None = None
    reinsert_mode: str = "faithful"
    also_fresh_arm: bool = False
    allow_partial_replay: bool = False
    #: A second replay window, run through the same protocol and published under
    #: ``diagnostics.alternate_replay``. The gate's answer depends on which
    #: recorded traffic it is handed, and that dependence belongs in the artifact
    #: rather than in whoever ran it once.
    diagnostic_replay_since: str | None = None
    min_containment: float = DEFAULT_MIN_CONTAINMENT
    within_days: int = DEFAULT_WITHIN_DAYS
    max_requests: int | None = None
    notes: list[str] = field(default_factory=list)

    def selector(self) -> CohortSelector:
        return CohortSelector(
            label="holdout",
            start=normalize_instant(self.window[0]),
            end=normalize_instant(self.window[1]),
            agent=self.agent,
            agent_is_null=self.agent_is_null,
            context_key=self.context_key[0] if self.context_key else None,
            context_value=self.context_key[1] if self.context_key else None,
        )


def _arm(
    working_db: Path,
    *,
    snapshots: Mapping[str, NodeSnapshot],
    baseline_state: Mapping[str, tuple[Any, ...]],
    baseline_connection_ids: set[str],
    cohort: Sequence[CohortNode],
    selector: CohortSelector,
    recorded_index: ConsumptionIndex,
    replay_index: ConsumptionIndex,
    requests: Sequence[ReplayRequest],
    config: CounterfactualConfig,
    mode: str,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Remove, re-insert in ``mode``, replay, measure. One disposable copy."""

    store = MemoryStore(MemoryConfig(db_path=working_db))
    try:
        service = MemoryRecallService(store)
        readonly = open_readonly(working_db)
        try:
            # The field database carries pre-existing dangling references; the
            # invariant this harness owns is that it introduces none of its own.
            baseline_violations = foreign_key_violations(readonly)
            coherence = {
                "before": assert_database_coherent(
                    readonly,
                    stage="before",
                    fts_check_connection=store.connection,
                    baseline_violations=baseline_violations,
                )
            }

            log(f"[{mode}] removing {len(snapshots)} cohort node(s)…")
            surgery = remove_cohort(store, snapshots)
            absent = assert_cohort_absent(readonly, sorted(snapshots))
            coherence["after_remove"] = assert_database_coherent(
                readonly,
                stage="after_remove",
                fts_check_connection=store.connection,
                baseline_violations=baseline_violations,
            )

            log(f"[{mode}] re-inserting {len(snapshots)} cohort node(s)…")
            reinsert_cohort(
                store,
                snapshots,
                mode=mode,
                baseline_state=baseline_state,
                baseline_connection_ids=baseline_connection_ids,
                surgery=surgery,
            )
            coherence["after_reinsert"] = assert_database_coherent(
                readonly,
                stage="after_reinsert",
                fts_check_connection=store.connection,
                baseline_violations=baseline_violations,
            )
            round_trip = (
                assert_cohort_restored(readonly, snapshots)
                if mode == "faithful"
                else {
                    "nodes_compared": len(snapshots),
                    "mismatched": None,
                    "skipped": "fresh mode deliberately drops history",
                }
            )

            log(f"[{mode}] replaying {len(requests)} recorded request(s)…")
            started = time.monotonic()

            def progress(done: int, total: int) -> None:
                if done % 100 == 0 or done == total:
                    log(f"[{mode}]   {done}/{total} replayed")

            runs = replay_requests(service, requests, progress=progress)
            elapsed = time.monotonic() - started

            counter = counterfactual_index(recorded_index, requests, runs)
            block = consumption_block(
                readonly,
                counter,
                cohort,
                selector,
                min_containment=config.min_containment,
                within_days=config.within_days,
            )
            fidelity = _delivery_fidelity(replay_index, counter)
        finally:
            readonly.close()
    finally:
        store.close()

    block["surgery"] = surgery.to_dict()
    block["coherence"] = coherence
    block["cohort_absent_after_remove"] = absent
    block["round_trip"] = round_trip
    block["replay"] = {
        "requests": len(requests),
        "events": len(counter.events),
        "seconds": round(elapsed, 1),
        "seconds_per_request": round(elapsed / max(1, len(requests)), 3),
        "fidelity": fidelity,
    }
    return block


@dataclass(slots=True)
class _ReplayPlan:
    """One replay window, resolved: what to replay and what it recorded."""

    replay_since: str
    complete: bool
    index: ConsumptionIndex
    requests: list[ReplayRequest]
    stats: dict[str, Any]
    retrospective: dict[str, Any]


def _replay_plan(
    connection: sqlite3.Connection,
    index: ConsumptionIndex,
    cohort: Sequence[CohortNode],
    selector: CohortSelector,
    *,
    replay_since: str,
    as_of: str,
    config: CounterfactualConfig,
) -> _ReplayPlan:
    """Resolve one replay window into requests plus its recorded-side figures.

    Both arms are pinned to the events this returns: the retrospective block is
    computed over exactly the events the replay will be given, so the only thing
    that can differ between the arms is the deliveries themselves.
    """

    ordinals = [
        ordinal
        for ordinal, event in enumerate(index.events)
        if replay_since <= event.created_at <= as_of
    ]
    if not ordinals:
        raise ValueError("no recorded recall event falls in the replay window")
    specs = load_request_specs(connection, [index.events[ordinal].id for ordinal in ordinals])
    requests, stats = build_replay_requests(index, ordinals, specs)
    dropped = 0
    if config.max_requests is not None and config.max_requests < len(requests):
        dropped = len(requests) - config.max_requests
        requests = requests[: config.max_requests]
    covered = sorted({ordinal for request in requests for ordinal in request.event_ordinals})
    replay_index = subset_index(index, covered)
    stats["events_replayed"] = len(covered)
    stats["requests_dropped"] = dropped
    return _ReplayPlan(
        replay_since=replay_since,
        complete=replay_since <= selector.start,
        index=replay_index,
        requests=requests,
        stats=stats,
        retrospective=consumption_block(
            connection,
            replay_index,
            cohort,
            selector,
            min_containment=config.min_containment,
            within_days=config.within_days,
        ),
    )


def _alternate_block(plan: _ReplayPlan, arm: dict[str, Any]) -> dict[str, Any]:
    """Headline comparison for a second replay window, published, not buried.

    The gate's answer depends on which recorded traffic it is given, and that
    dependence is a property a reader has to be able to see. Replaying traffic
    recorded *while* the cohort was being written means replaying it against
    today's larger corpus and today's ranker, which is a harder reproduction
    than post-cutoff traffic — so this number is expected to be worse, and
    hiding it would make the headline look better than the method is.
    """

    retro = plan.retrospective
    gap = _gap(retro["rate"], arm["rate"])
    return {
        "replay_since": plan.replay_since,
        "replay_is_complete": plan.complete,
        "events_replayed": plan.stats["events_replayed"],
        "requests": len(plan.requests),
        "retrospective_rate": retro["rate"],
        "counterfactual_rate": arm["rate"],
        "retrospective_consumed_later": retro["consumed_later"],
        "counterfactual_consumed_later": arm["consumed_later"],
        "gap": gap,
        "within_max_gap": bool(gap is not None and gap <= SELFCHECK_MAX_GAP),
        "reproduced_share_of_recorded": arm["replay"]["fidelity"][
            "reproduced_share_of_recorded"
        ],
        "agreement": _agreement(retro["_consumed"], arm["_consumed"], retro["nodes"]),
        "grounded": {
            "retrospective_rate": retro["grounded"]["rate"],
            "counterfactual_rate": arm["grounded"]["rate"],
            "retrospective_denominator": retro["grounded"]["nodes_with_closed_consumer"],
            "counterfactual_denominator": arm["grounded"]["nodes_with_closed_consumer"],
            "gap": _gap(retro["grounded"]["rate"], arm["grounded"]["rate"]),
        },
    }


def _strip_internal(payload: Any) -> None:
    """Drop node-id sets from the published report; aggregates only."""

    if isinstance(payload, dict):
        payload.pop("_consumed", None)
        for value in payload.values():
            _strip_internal(value)
    elif isinstance(payload, list):
        for value in payload:
            _strip_internal(value)


def run_counterfactual(
    source_db: str | Path | None,
    config: CounterfactualConfig,
    *,
    snapshot: str | Path | None = None,
    log: Callable[[str], None] = lambda _message: None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Run the whole protocol and return the publishable report."""

    if config.reinsert_mode not in REINSERT_MODES:
        raise ValueError(f"unknown reinsert mode {config.reinsert_mode!r}")
    selector = config.selector()
    as_of = normalize_instant(config.as_of)
    replay_since = normalize_instant(config.replay_since)
    # An event recorded before the earliest cohort node was written can never
    # consume one — consumption is strictly later — so replaying from the
    # cohort window's start covers *every* possible consumer. Only then does the
    # retrospective rate measured over the replayed events equal the cohort's
    # unrestricted retrospective rate, which is what the self-check compares
    # against. Starting later leaves recorded consumptions the replay was never
    # given a chance to reproduce, and that has to be an explicit choice.
    complete_replay = replay_since <= selector.start
    if not complete_replay and not config.allow_partial_replay:
        raise ValueError(
            f"replay_since {replay_since} is later than the cohort window start "
            f"{selector.start}: the replay would skip events that recorded "
            "consumptions of this cohort. Pass allow_partial_replay to accept that."
        )

    with frozen_snapshot(snapshot, source_db) as (snapshot_path, manifest):
        log(f"snapshot: {snapshot_path}")
        readonly = open_readonly(snapshot_path)
        try:
            index = load_consumption_index(readonly, as_of=as_of)
            cohort = load_cohort_nodes(readonly, selector, as_of=as_of)
            if not cohort:
                raise ValueError("the cohort selector matched no nodes")
            # The retrospective arm is pinned to the events that will *actually*
            # be replayed, cap included. Comparing a full recorded history
            # against a truncated replay would make the counterfactual look
            # empty for a reason that has nothing to do with retrieval.
            plan = _replay_plan(
                readonly,
                index,
                cohort,
                selector,
                replay_since=replay_since,
                as_of=as_of,
                config=config,
            )
            requests, request_stats, replay_index = plan.requests, plan.stats, plan.index
            retrospective = plan.retrospective
            if request_stats["requests_dropped"]:
                config.notes.append(
                    f"--max-requests capped the replay at {len(requests)}, dropping "
                    f"{request_stats['requests_dropped']}: a smoke run, not a baseline"
                )
            log(
                f"cohort {len(cohort)} node(s); replay {len(replay_index.events)} recorded "
                f"event(s) -> {len(requests)} distinct request(s)"
            )
            alternate_plan = (
                _replay_plan(
                    readonly,
                    index,
                    cohort,
                    selector,
                    replay_since=normalize_instant(config.diagnostic_replay_since),
                    as_of=as_of,
                    config=config,
                )
                if config.diagnostic_replay_since
                else None
            )
            retrospective_all = consumption_block(
                readonly,
                index,
                cohort,
                selector,
                min_containment=config.min_containment,
                within_days=config.within_days,
            )
            snapshots = capture_cohort(readonly, [node.id for node in cohort])
            baseline_state = _fingerprint_neighbourhood(readonly, snapshots)
            baseline_connection_ids = _connection_ids_touching(readonly, sorted(snapshots))
        finally:
            readonly.close()

        arms: dict[str, dict[str, Any]] = {}
        modes = [config.reinsert_mode]
        if config.also_fresh_arm and "fresh" not in modes:
            modes.append("fresh")
        for mode in modes:
            with working_copy(snapshot_path) as working_db:
                arms[mode] = _arm(
                    working_db,
                    snapshots=snapshots,
                    baseline_state=baseline_state,
                    baseline_connection_ids=baseline_connection_ids,
                    cohort=cohort,
                    selector=selector,
                    recorded_index=index,
                    replay_index=replay_index,
                    requests=requests,
                    config=config,
                    mode=mode,
                    log=log,
                )

        alternate: dict[str, Any] | None = None
        if alternate_plan is not None:
            log(f"alternate replay window from {alternate_plan.replay_since}…")
            with working_copy(snapshot_path) as working_db:
                alternate_arm = _arm(
                    working_db,
                    snapshots=snapshots,
                    baseline_state=baseline_state,
                    baseline_connection_ids=baseline_connection_ids,
                    cohort=cohort,
                    selector=selector,
                    recorded_index=index,
                    replay_index=alternate_plan.index,
                    requests=alternate_plan.requests,
                    config=config,
                    mode=config.reinsert_mode,
                    log=log,
                )
            alternate = _alternate_block(alternate_plan, alternate_arm)

    primary = arms[config.reinsert_mode]
    agreement = _agreement(retrospective["_consumed"], primary["_consumed"], len(cohort))
    grounded_retro = retrospective["grounded"]
    grounded_counter = primary["grounded"]
    gap = _gap(retrospective["rate"], primary["rate"])

    selfcheck = {
        "cohort_kind": config.cohort_kind,
        "cohort_rule": config.cohort_rule,
        "retrospective_rate": retrospective["rate"],
        "counterfactual_rate": primary["rate"],
        "gap": gap,
        "max_gap": SELFCHECK_MAX_GAP,
        "passed": bool(gap is not None and gap <= SELFCHECK_MAX_GAP),
        "cohort_nodes": len(cohort),
        "reinsert_mode": config.reinsert_mode,
        "retrospective_consumed_later": retrospective["consumed_later"],
        "counterfactual_consumed_later": primary["consumed_later"],
        "retrospective_rate_all_events": retrospective_all["rate"],
        "replay_is_complete": complete_replay,
        "grounded": {
            "denominator": grounded_counter["nodes_with_closed_consumer"],
            "denominator_basis": "nodes_with_closed_consumer",
            "retrospective_denominator": grounded_retro["nodes_with_closed_consumer"],
            "counterfactual_denominator": grounded_counter["nodes_with_closed_consumer"],
            "retrospective_rate": grounded_retro["rate"],
            "counterfactual_rate": grounded_counter["rate"],
            "gap": _gap(grounded_retro["rate"], grounded_counter["rate"]),
            "retrospective_consumed_later": grounded_retro["consumed_later"],
            "counterfactual_consumed_later": grounded_counter["consumed_later"],
            "min_containment": config.min_containment,
        },
        "agreement": agreement,
    }

    notes = list(config.notes)
    notes.append(
        "both rates use one definition, imported from postsession.usage_metric: "
        "only the delivered results differ between the arms"
    )
    if complete_replay:
        notes.append(
            "complete replay: every event that could consume this cohort was replayed, "
            "so the compared and all-events retrospective rates coincide"
        )
    else:
        notes.append(
            "cutoff design: the cohort closes at replay_since, so both arms score only "
            "post-cutoff traffic and consumptions recorded before it are out of scope"
        )
        notes.append(
            "a candidate cohort is written at one instant, which makes its replay "
            "complete by construction; the alternate window is the closer analogue"
        )
    notes.append(
        "grounded.rate divides by nodes_with_closed_consumer per arm, never by the "
        "cohort: most recall events carry no feedback_trace_id"
    )
    notes.append(
        "the replay runs on a disposable working copy; the source database is only "
        "ever opened mode=ro or through the SQLite backup API"
    )
    notes.append(
        "recorded results came from older retrieval code on a smaller corpus, so a "
        "residual gap is expected; the bar is what bounds it"
    )
    if alternate is not None:
        notes.append(
            f"diagnostics.alternate_replay re-runs the protocol from "
            f"{alternate['replay_since']}: gap {alternate['gap']}, "
            f"within_max_gap {alternate['within_max_gap']}"
        )
        notes.append(
            f"that window replays traffic contemporaneous with the writes against "
            f"today's corpus and ranker, reproducing "
            f"{alternate['reproduced_share_of_recorded']} of recorded deliveries"
        )

    report: dict[str, Any] = {
        "meta": {
            "artifact": "counterfactual_consumption",
            "question": (
                "if this cohort were written into the world again, would the recorded "
                "later queries surface it"
            ),
            "as_of": as_of,
            "generated_at_utc": generated_at
            or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_db": str(source_db) if source_db is not None else None,
            "snapshot_sha256": manifest.get("snapshot_sha256"),
            "snapshot_row_counts": manifest.get("row_counts"),
            "open_mode": "ro + sqlite backup api; MemoryStore only on the working copy",
        },
        "protocol": {
            "steps": [
                "freeze the source database with the SQLite backup API",
                "take a disposable working copy of the frozen snapshot",
                "remove the cohort and every row referencing it; assert coherence",
                "re-insert through MemoryStore.create_node; assert the round trip",
                "replay recorded requests via retrieval_harness.run_item (no logging)",
                "measure consumption under postsession.usage_metric's definitions",
            ],
            "cohort_window": [selector.start, selector.end],
            "replay_since": replay_since,
            "replay_is_complete": complete_replay,
            "reinsert_mode": config.reinsert_mode,
            "reinsert_modes_available": list(REINSERT_MODES),
            "min_containment": config.min_containment,
            "within_days": config.within_days,
            "request_key": "query + requested scope + ambient context + depth + max_results",
            "request_stats": request_stats,
            "selector": selector.to_dict(),
        },
        "arms": {
            "retrospective": retrospective,
            "retrospective_all_events": retrospective_all,
            **{f"counterfactual_{mode}": block for mode, block in arms.items()},
        },
        "selfcheck": selfcheck,
        "diagnostics": {
            "alternate_replay": alternate,
            "history_reset_arm": {
                "counterfactual_rate": arms["fresh"]["rate"],
                "delta_vs_faithful": _gap(arms["fresh"]["rate"], primary["rate"]),
                "grounded_rate": arms["fresh"]["grounded"]["rate"],
                "grounded_denominator": arms["fresh"]["grounded"]["nodes_with_closed_consumer"],
                "reproduced_share_of_recorded": arms["fresh"]["replay"]["fidelity"][
                    "reproduced_share_of_recorded"
                ],
            }
            if "fresh" in arms and config.reinsert_mode != "fresh"
            else None,
        },
        "notes": notes,
    }
    _strip_internal(report)
    check_privacy(report)
    return report
