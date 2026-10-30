"""SQLite persistence for the Living Memory uniform node store."""

from __future__ import annotations

from array import array
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, closing, contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from itertools import groupby
from threading import RLock
from typing import Any, Literal
import hashlib
import json
import re
import secrets
import sqlite3
import sys
import time
from datetime import UTC, datetime, timedelta

from living_memory.chunking import TextChunk, chunk_text
from living_memory.config import MemoryConfig, RetrievalWeightConfig
from living_memory.embeddings import (
    LocalEmbeddingModel,
    cosine_similarity,
    fts_prefix_term_allowed,
)
from living_memory.models import (
    content_identity,
    CONNECTION_TYPES,
    NODE_LEVELS,
    REJECTED_ALTERNATIVE_KIND,
    Connection,
    ConnectionType,
    Node,
    NodeChunkEmbedding,
    NodeLevel,
    RecallEvent,
    RecallEventHeader,
    RetrievalPolicyFloors,
    RetrievalWeights,
    SourcePassage,
)
from living_memory.relations import (
    source_relation_weight_sql,
    correction_sql, is_correction, is_semantic_relation, semantic_relation_sql,
)
from living_memory.scope import ALL_SCOPES, ScopePlan, normalize_scope, resolve_scope
from living_memory.temporal import parse_timestamp

SCHEMA_VERSION = 8

CHUNK_EMBEDDING_TABLE = "node_chunk_embeddings"
QUERY_ANCHOR_TABLE = "query_anchors"
QUERY_ANCHOR_EDGE_TABLE = "query_anchor_edges"
RECALL_ATTESTATION_TABLE = "recall_attestations"
RECALL_HISTORY_EVENT_TABLE = "recall_history_events"
RECALL_HISTORY_RESULT_TABLE = "recall_history_result_nodes"
RECALL_DELIVERY_HISTORY_TABLE = "recall_delivery_history"
RECALL_DELIVERY_HISTORY_STATE_TABLE = "recall_delivery_history_state"
RECALL_LOOKUP_EVENT_TABLE = "recall_lookup_events"
TRANSCRIPT_GROUNDING_TABLE = "transcript_grounding_verdicts"
RECALL_CREDIT_LEDGER_TABLE = "recall_credit_ledger"
RECALL_EXPLICIT_CREDIT_TABLE = "recall_explicit_credit"
RECALL_FEEDBACK_MARKS_TABLE = "recall_feedback_marks"
CONSOLIDATION_EMBEDDING_TABLE = "consolidation_embeddings"
#: Every basis a (recall event, node) credit can be claimed under. The first
#: two live in ``recall_credit_ledger``, whose CHECK cannot be widened without
#: a table rebuild; ``explicit`` lives in ``recall_explicit_credit`` and the
#: once-per-pair rule is enforced across both tables. See the note on
#: ``_RECALL_EXPLICIT_CREDIT_SCHEMA_SQL``.
RECALL_CREDIT_BASES: tuple[str, ...] = ("grounded", "lookup", "explicit")

# Frozen by artifacts/recall-map/prereg.json and
# artifacts/recall-map/relevance/policy.json.  Keep these literals local to
# storage: the history ledger is the production reconstruction of those
# delivery/outcome definitions, not a generic record of every result.
RECALL_DELIVERY_HISTORY_HORIZON_HOURS = 24
RECALL_DELIVERY_HISTORY_HEAD_CUT = 3
RECALL_DELIVERY_HISTORY_ITEMS_PER_EVENT = 6
MAX_RECALL_HISTORY_CANDIDATES = 200
MAX_RECALL_HISTORY_DELIVERIES_PER_NODE = 1024
#: Bumped to 2 when ``lookup_consumed`` joined the ledger: an existing file's
#: rows carry no lookup outcome at all, and silently reading a missing column as
#: "absent" is the one answer the tri-state exists to forbid. The bump makes
#: every pre-existing file re-derive through
#: :meth:`MemoryStore._backfill_recall_delivery_history`, which marks every
#: window that closed before the first recorded lookup NULL.
_RECALL_DELIVERY_HISTORY_FORMAT = 2
#: numpy dtype string for a stored chunk BLOB. Part of the read contract that
#: the vector channel consumes: ``np.frombuffer(blob, dtype=CHUNK_EMBEDDING_DTYPE)``.
CHUNK_EMBEDDING_DTYPE = "<f4"
CHUNK_EMBEDDING_ITEMSIZE = 4

#: SQL literal for the ASCII whitespace set, for TRIM(x, chars): SQLite's
#: one-argument TRIM strips spaces and nothing else.
_SQL_WHITESPACE = "' ' || CHAR(9) || CHAR(10) || CHAR(11) || CHAR(12) || CHAR(13)"

# Revisions share metadata's transaction with the chunk rows. Empty scopes keep
# their revision so deleting and repopulating a scope cannot reuse an identity.
_CHUNK_SCOPE_REVISION_PREFIX = "chunk_scope_revision:"
_CHUNK_SCOPE_REVISION_FORMAT = "chunk_scope_revision_format"


def _chunk_scope_revision_write(scopes: str) -> str:
    return f"""INSERT INTO metadata (key, value)
        SELECT '{_CHUNK_SCOPE_REVISION_PREFIX}' || scope, 1 FROM ({scopes})
        WHERE scope IS NOT NULL GROUP BY scope
        ON CONFLICT(key) DO UPDATE SET value = value + 1;"""


# BEFORE INSERT also observes a row displaced by REPLACE: SQLite need not run
# its DELETE triggers when recursive_triggers is off. Both scopes must advance.
_CHUNK_CORPUS_TRIGGERS = (
    f"""CREATE TRIGGER chunk_corpus_insert
    BEFORE INSERT ON node_chunk_embeddings BEGIN
        {_chunk_scope_revision_write("SELECT scope FROM nodes WHERE id = NEW.node_id OR id IN (SELECT node_id FROM node_chunk_embeddings WHERE id = NEW.id)")}
    END""",
    f"""CREATE TRIGGER chunk_corpus_delete
    AFTER DELETE ON node_chunk_embeddings BEGIN
        {_chunk_scope_revision_write("SELECT scope FROM nodes WHERE id = OLD.node_id")}
    END""",
    f"""CREATE TRIGGER chunk_corpus_update
    BEFORE UPDATE OF id, node_id, chunk_index, embedding ON node_chunk_embeddings
    WHEN OLD.id IS NOT NEW.id OR OLD.node_id IS NOT NEW.node_id
        OR OLD.chunk_index IS NOT NEW.chunk_index
        OR OLD.embedding IS NOT NEW.embedding
    BEGIN
        {_chunk_scope_revision_write("SELECT scope FROM nodes WHERE id IN (OLD.node_id, NEW.node_id) OR id IN (SELECT node_id FROM node_chunk_embeddings WHERE id = NEW.id)")}
    END""",
)

_CHUNK_MEMBERSHIP_TRIGGERS = (
    f"""CREATE TRIGGER chunk_corpus_membership
    BEFORE UPDATE OF id, scope, decayed ON nodes
    WHEN (OLD.id IS NOT NEW.id OR OLD.scope IS NOT NEW.scope
          OR OLD.decayed IS NOT NEW.decayed)
        AND EXISTS (SELECT 1 FROM node_chunk_embeddings
                    WHERE node_id IN (OLD.id, NEW.id))
    BEGIN
        {_chunk_scope_revision_write("SELECT OLD.scope AS scope UNION SELECT NEW.scope UNION SELECT scope FROM nodes WHERE id = NEW.id")}
    END""",
    f"""CREATE TRIGGER chunk_corpus_node_delete
    BEFORE DELETE ON nodes
    WHEN EXISTS (SELECT 1 FROM node_chunk_embeddings WHERE node_id = OLD.id)
    BEGIN
        {_chunk_scope_revision_write("SELECT OLD.scope AS scope")}
    END""",
    f"""CREATE TRIGGER chunk_corpus_node_insert
    BEFORE INSERT ON nodes
    WHEN EXISTS (SELECT 1 FROM node_chunk_embeddings WHERE node_id = NEW.id)
    BEGIN
        {_chunk_scope_revision_write("SELECT scope FROM nodes WHERE id = NEW.id UNION SELECT NEW.scope")}
    END""",
)

# Independent of chunks: anchor membership and vectors have their own revision.
# INSERT/DELETE also cover SQL replacement, whose implicit deletion need not
# fire a DELETE trigger when SQLite recursive_triggers is disabled.
_QUERY_ANCHOR_CORPUS_TRIGGERS = (
    """CREATE TRIGGER query_anchor_corpus_insert
    AFTER INSERT ON query_anchors BEGIN
        UPDATE metadata SET value = value + 1 WHERE key = 'query_anchor_revision';
    END
    """.strip(),
    """CREATE TRIGGER query_anchor_corpus_delete
    AFTER DELETE ON query_anchors BEGIN
        UPDATE metadata SET value = value + 1 WHERE key = 'query_anchor_revision';
    END
    """.strip(),
    """CREATE TRIGGER query_anchor_corpus_update
    AFTER UPDATE OF id, scope, dimensions, embedding, decayed ON query_anchors
    WHEN (OLD.decayed = 0 OR NEW.decayed = 0) AND (
        OLD.id IS NOT NEW.id OR OLD.scope IS NOT NEW.scope
        OR OLD.dimensions IS NOT NEW.dimensions
        OR OLD.embedding IS NOT NEW.embedding
        OR OLD.decayed IS NOT NEW.decayed
    )
    BEGIN
        UPDATE metadata SET value = value + 1 WHERE key = 'query_anchor_revision';
    END
    """.strip(),
)

_QUERY_ANCHOR_REVISION_SQL = """
    INSERT OR IGNORE INTO metadata (key, value) VALUES ('query_anchor_revision', '0');
""" + "\n".join(
    sql.replace("CREATE TRIGGER ", "CREATE TRIGGER IF NOT EXISTS ", 1) + ";"
    for sql in _QUERY_ANCHOR_CORPUS_TRIGGERS
)


def install_derived_only_authorizer(
    conn: sqlite3.Connection, *, corpus: Literal["chunks", "anchors"]
) -> None:
    """Restrict maintenance to one derived corpus and its schema-owned revision.

    Install after MemoryStore migrations. Reads and transactions remain open;
    direct knowledge/metadata writes, foreign trigger writes and DDL are denied.
    The same trigger SQL creates the schema and establishes its permitted effects.
    Check it when SQLite prepares a write, including after a schema change forces
    re-preparation. A trigger name alone cannot authorize a metadata update.
    """

    if corpus == "chunks":
        tables = {CHUNK_EMBEDDING_TABLE}
        triggers = _CHUNK_CORPUS_TRIGGERS
        revision_key = _CHUNK_SCOPE_REVISION_FORMAT
    elif corpus == "anchors":
        tables = {QUERY_ANCHOR_TABLE, QUERY_ANCHOR_EDGE_TABLE}
        triggers = _QUERY_ANCHOR_CORPUS_TRIGGERS
        revision_key = "query_anchor_revision"
    else:
        raise ValueError(f"unknown derived corpus: {corpus!r}")

    expected = {sql.split()[2]: sql for sql in triggers}
    placeholders = ",".join("?" for _ in expected)
    writes = {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}
    read_and_transaction_actions = {
        sqlite3.SQLITE_READ, sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_TRANSACTION, sqlite3.SQLITE_SAVEPOINT, sqlite3.SQLITE_RECURSIVE,
    }

    def revision_schema_matches() -> bool:
        actual = dict(conn.execute(
            f"SELECT name, sql FROM main.sqlite_master "
            f"WHERE type = 'trigger' AND name IN ({placeholders})",
            tuple(expected),
        ))
        # SQLite's source argument does not distinguish main from TEMP triggers.
        shadow = conn.execute(
            f"SELECT 1 FROM temp.sqlite_master "
            f"WHERE type = 'trigger' AND name IN ({placeholders}) LIMIT 1",
            tuple(expected),
        ).fetchone()
        revision = conn.execute(
            "SELECT value FROM main.metadata WHERE key = ?", (revision_key,)
        ).fetchone()
        return (
            actual == expected and shadow is None and revision is not None
            and (corpus != "chunks" or revision[0] == "1")
        )

    def authorizer(
        action: int,
        table: str | None,
        column: str | None,
        database: str | None,
        source: str | None,
    ) -> int:
        if action in writes:
            if database == "main":
                if table in tables and source is None and revision_schema_matches():
                    return sqlite3.SQLITE_OK
                if (
                    table == "metadata" and source in expected
                    and (
                        (action == sqlite3.SQLITE_UPDATE and column == "value")
                        or (corpus == "chunks" and action == sqlite3.SQLITE_INSERT)
                    )
                ):
                    return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        if action in read_and_transaction_actions:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_PRAGMA:
            # FTS5 checks data_version while reading its index.
            if table == "table_info" or (table == "data_version" and column is None):
                return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    conn.set_authorizer(authorizer)


#: Query-anchor DDL (schema v7). Held as one constant because it has two
#: callers that must never drift apart: ``_initialize_schema`` runs it so a
#: fresh database gets the tables, and ``_migrate_pre_v7_schema`` runs it so a
#: pre-v7 file gets exactly the same objects. ``_migrate_pre_v6_schema``'s
#: docstring records why a second hand-written copy of a table's DDL inside the
#: migration is the wrong shape: two copies to keep in step. Purely additive —
#: it touches no existing table, so the ``nodes`` and ``connections`` DDL that a
#: 500 MB live database already carries (with its baked-in ``level`` and
#: ``type`` CHECK constraints) stays byte-identical.
_QUERY_ANCHOR_SCHEMA_SQL = """
    -- Query anchors (schema v7): the graph's entry from query space. One row
    -- is one remembered *question* — the operator's own words, embedded by the
    -- same model and packed the same way as node chunks — so a repeat of a
    -- situation is found by "query <-> past query" instead of
    -- "query <-> node content", which measurement puts at 0.10-0.27 across
    -- languages while the query-to-query match sits at ~0.92.
    --
    -- `scope` is the consuming recall event's scope and an anchor never
    -- crosses it. `fingerprint` is storage.recall_fingerprint(query, scope),
    -- the exact-match dedup key that recall_events already stamps; UNIQUE with
    -- the scope so the same question in two scopes stays two anchors.
    -- `dimensions` is recorded rather than inferred from the BLOB's byte
    -- length, for the reason unpack_chunk_embedding states: a length-derived
    -- width cannot tell a truncated write from a short vector.
    -- Decay is soft, like a node's, so a quiet anchor stops matching without
    -- taking its learned edges with it.
    CREATE TABLE IF NOT EXISTS query_anchors (
        id TEXT PRIMARY KEY,
        scope TEXT NOT NULL,
        query TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        dimensions INTEGER NOT NULL CHECK (dimensions > 0),
        embedding BLOB NOT NULL,
        reinforcement_count INTEGER NOT NULL DEFAULT 0,
        usefulness_score REAL NOT NULL DEFAULT 0.0,
        decayed INTEGER NOT NULL DEFAULT 0 CHECK (decayed IN (0, 1)),
        decay_reason TEXT,
        first_seen TEXT NOT NULL,
        last_matched_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(scope, fingerprint)
    );

    -- Weighted anchor -> node edges. Bipartite by construction: `anchor_id`
    -- only ever names a query_anchors row and `target_id` only ever a nodes
    -- row, so no anchor edge can be a self-loop or close a cycle. `hits`
    -- counts reinforcements separately from `weight` so accumulation stays
    -- visible after the weight saturates at 1.0.
    CREATE TABLE IF NOT EXISTS query_anchor_edges (
        anchor_id TEXT NOT NULL REFERENCES query_anchors(id),
        target_id TEXT NOT NULL REFERENCES nodes(id),
        weight REAL NOT NULL DEFAULT 0.0,
        hits INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (anchor_id, target_id)
    );

    -- Anchor-scan driver: match_anchors walks one scope's live anchors.
    -- Partial on decayed = 0 because a decayed anchor must not match, and the
    -- decayed share grows without bound as unreinforced anchors age out.
    CREATE INDEX IF NOT EXISTS idx_query_anchors_scope_active
        ON query_anchors(scope)
        WHERE decayed = 0;

    -- Edge migration drives off the target: "every anchor edge pointing at the
    -- node being superseded". PRIMARY KEY(anchor_id, target_id) leads with
    -- anchor_id and cannot serve that, so without this index every supersedes
    -- write would scan the whole edge table.
    CREATE INDEX IF NOT EXISTS idx_query_anchor_edges_target
        ON query_anchor_edges(target_id);
"""

#: Columns of ``query_anchors`` in DDL order, for the pre-v7 ALTER TABLE guard.
_QUERY_ANCHOR_COLUMNS: tuple[tuple[str, str], ...] = (
    ("reinforcement_count", "INTEGER NOT NULL DEFAULT 0"),
    ("usefulness_score", "REAL NOT NULL DEFAULT 0.0"),
    ("decayed", "INTEGER NOT NULL DEFAULT 0"),
    ("decay_reason", "TEXT"),
    ("last_matched_at", "TEXT"),
)

#: Columns of ``query_anchor_edges`` that a partial earlier rollout could lack.
_QUERY_ANCHOR_EDGE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("weight", "REAL NOT NULL DEFAULT 0.0"),
    ("hits", "INTEGER NOT NULL DEFAULT 0"),
)

#: Grounded-usage attestation ledger DDL (schema v8). Held as one constant for
#: the same reason ``_QUERY_ANCHOR_SCHEMA_SQL`` is: whatever creates this table
#: must create exactly one shape. Right now that is a single caller,
#: ``_initialize_schema``, and there is deliberately **no**
#: ``_migrate_pre_v8_schema``:
#:
#: ``_initialize_schema`` runs its ``CREATE TABLE IF NOT EXISTS`` script on
#: *every* open, so a v7-era file gains ``recall_attestations`` the moment it is
#: reopened by this build — a migration method that only ran ``CREATE TABLE IF
#: NOT EXISTS`` would be dead code. The v6 and v7 migrations exist for the two
#: things that script genuinely cannot do: repair an index whose partial
#: predicate names a dropped column (v6), and ``ALTER TABLE ... ADD COLUMN`` a
#: table that a partial earlier rollout already created short (v7). Neither
#: applies here: ``recall_attestations`` is a brand-new table that has never
#: shipped in any other shape, so no deployed database can carry a short
#: version of it. If a future release adds a column, *that* release adds the
#: ``ALTER TABLE`` guard — the ``CREATE`` half stays here, never copied.
#: ``tests/test_usage_attestation.py`` pins the reopen behaviour by dropping the
#: table from a live file and reopening it.
#:
#: Strictly additive, like v7: one new table touching no existing one, so the
#: ``nodes``/``connections`` DDL a large live database already carries stays
#: byte-identical.
_RECALL_ATTESTATION_SCHEMA_SQL = """
    -- Grounded-usage attestations (schema v8): the idempotency ledger for the
    -- retroactive credit path (living_memory.attestation). One row is one
    -- graded submission of session-artifact evidence against one recall event.
    --
    -- UNIQUE(recall_event_id, evidence_sha256) is the whole point: the same
    -- (event, evidence) pair is graded and credited exactly once, however many
    -- times an offline extractor is re-run over the same transcript. The row is
    -- claimed *before* any credit is applied, so a crash mid-apply leaves the
    -- key taken and the retry replays instead of double-crediting.
    --
    -- `containments` is the full per-result verdict the server computed
    -- (node_id, rank, containment, grounded, evidence_index) — recorded rather
    -- than recomputed so a replay returns the same numbers the credited run
    -- did, even after the graded nodes decay or their content changes.
    -- `agent`/`task`/`session_id`/`transport_session_id`/`source_session_key`
    -- are audit only: they say who attested and which finished session the
    -- evidence came from. They are never a gate. The offline extractor connects
    -- with a new transport session id by construction, so a same-session check
    -- would break the feature outright; honesty comes from the server
    -- recomputing containment over its own copy of the event's results.
    CREATE TABLE IF NOT EXISTS recall_attestations (
        id TEXT PRIMARY KEY,
        recall_event_id TEXT NOT NULL REFERENCES recall_events(id),
        evidence_sha256 TEXT NOT NULL,
        evidence_items INTEGER NOT NULL,
        evidence_chars INTEGER NOT NULL,
        min_containment REAL NOT NULL,
        containments TEXT NOT NULL DEFAULT '[]',
        grounded_node_ids TEXT NOT NULL DEFAULT '[]',
        anchor_ids TEXT NOT NULL DEFAULT '[]',
        credited INTEGER NOT NULL DEFAULT 0 CHECK (credited IN (0, 1)),
        closed INTEGER NOT NULL DEFAULT 0 CHECK (closed IN (0, 1)),
        closed_by_attestation INTEGER NOT NULL DEFAULT 0
            CHECK (closed_by_attestation IN (0, 1)),
        feedback_trace_id TEXT REFERENCES nodes(id),
        agent TEXT,
        task TEXT,
        session_id TEXT,
        transport_session_id TEXT,
        source_session_key TEXT,
        created_at TEXT NOT NULL,
        UNIQUE(recall_event_id, evidence_sha256)
    );

    -- Audit driver: "everything attested out of one finished session". The
    -- UNIQUE index above already serves the per-event probe the write path
    -- makes, and cannot serve this one.
    CREATE INDEX IF NOT EXISTS idx_recall_attestations_source_session
        ON recall_attestations(source_session_key, created_at DESC)
        WHERE source_session_key IS NOT NULL;
"""

# Frozen recall-map relevance history (additive to schema v8).  The three data tables are
# deliberately normalized instead of reparsing recall_events JSON in the map
# builder:
#
# * recall_history_events preserves every possible consumer, including an
#   event with no result ids (its mere presence suppresses scope/task fallback
#   when its transport matches);
# * recall_history_result_nodes preserves the node-id limb of consumption;
# * recall_delivery_history contains the union, per event, of the organic tail
#   and delivered map medoids.  The PRIMARY KEY implements the evaluator's
#   per-event de-duplication when a node appears in both sources.
#
# ``transport_matched`` must stay separate from the two consumed bits.  The
# frozen rule is transport first and scope/task only when no later transport
# event exists, so a later transport near-miss can invalidate an earlier
# fallback hit.  Storing only one eager ``consumed`` bit would make that case
# irrecoverable.
#
# ``recall_lookup_events`` and the ``lookup_consumed`` column are the fourth
# table's exogenous counterpart.  The three columns above all resolve to "a
# later *recall* returned this node again", which is the ranker's own echo: a
# schema that the trigger boost mixes into nearly every recall satisfies them
# forever without anybody reading it.  An id-fetch of a specific ULID through
# ``memory_lookup`` is the one act only a reader performs, so it is recorded
# separately and correlated separately.
_RECALL_DELIVERY_HISTORY_SCHEMA_SQL = f"""
    CREATE TABLE IF NOT EXISTS {RECALL_HISTORY_EVENT_TABLE} (
        recall_event_id TEXT PRIMARY KEY REFERENCES recall_events(id),
        occurred_at TEXT NOT NULL,
        transport_session_id TEXT,
        scope TEXT NOT NULL,
        task TEXT
    );

    CREATE TABLE IF NOT EXISTS {RECALL_HISTORY_RESULT_TABLE} (
        recall_event_id TEXT NOT NULL
            REFERENCES {RECALL_HISTORY_EVENT_TABLE}(recall_event_id),
        node_id TEXT NOT NULL,
        PRIMARY KEY (recall_event_id, node_id)
    ) WITHOUT ROWID;

    CREATE TABLE IF NOT EXISTS {RECALL_DELIVERY_HISTORY_TABLE} (
        delivery_event_id TEXT NOT NULL
            REFERENCES {RECALL_HISTORY_EVENT_TABLE}(recall_event_id),
        node_id TEXT NOT NULL,
        delivered_at TEXT NOT NULL,
        outcome_end TEXT NOT NULL,
        transport_session_id TEXT,
        scope TEXT NOT NULL,
        task TEXT,
        transport_matched INTEGER NOT NULL DEFAULT 0
            CHECK (transport_matched IN (0, 1)),
        transport_consumed INTEGER NOT NULL DEFAULT 0
            CHECK (transport_consumed IN (0, 1)),
        fallback_consumed INTEGER NOT NULL DEFAULT 0
            CHECK (fallback_consumed IN (0, 1)),
        -- Tri-state on purpose: 1 followed, 0 observed-and-not-followed,
        -- NULL unobservable.  NULL is not a third flavour of "no": it is the
        -- honest answer for a window that closed before this server recorded
        -- lookups at all, and a reader that cannot tell the two apart would
        -- read a cold start as the whole corpus being ignored.
        lookup_consumed INTEGER
            CHECK (lookup_consumed IS NULL OR lookup_consumed IN (0, 1)),
        PRIMARY KEY (delivery_event_id, node_id)
    ) WITHOUT ROWID;

    -- One row per (id-fetch call, requested node id).  Missing ids are kept:
    -- the event records what was *asked for*, and asking for a node that is
    -- gone is still a follow of the card that offered it.  No foreign key to
    -- nodes for exactly that reason.
    CREATE TABLE IF NOT EXISTS {RECALL_LOOKUP_EVENT_TABLE} (
        lookup_event_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        occurred_at TEXT NOT NULL,
        transport_session_id TEXT,
        PRIMARY KEY (lookup_event_id, node_id)
    ) WITHOUT ROWID;

    CREATE TABLE IF NOT EXISTS {RECALL_DELIVERY_HISTORY_STATE_TABLE} (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        format_version INTEGER NOT NULL,
        complete INTEGER NOT NULL CHECK (complete IN (0, 1)),
        unavailable_reason TEXT,
        updated_at TEXT NOT NULL,
        CHECK (
            (complete = 1 AND unavailable_reason IS NULL)
            OR (complete = 0 AND unavailable_reason IS NOT NULL)
        )
    );

    -- Selection performs one bounded range scan per candidate.  The resolved
    -- consumed expression is part of the order because the frozen evaluator
    -- sorts equal delivery instants by the boolean outcome before walking the
    -- tail streak in reverse.
    CREATE INDEX IF NOT EXISTS idx_recall_delivery_history_node_matured
        ON {RECALL_DELIVERY_HISTORY_TABLE}(
            node_id,
            outcome_end DESC,
            delivered_at DESC,
            (CASE WHEN transport_matched = 1
                THEN transport_consumed ELSE fallback_consumed END) DESC,
            delivery_event_id DESC
        );

    -- Online outcome maintenance touches only the preceding 24-hour ranges
    -- compatible with the new event.
    CREATE INDEX IF NOT EXISTS idx_recall_delivery_history_transport_time
        ON {RECALL_DELIVERY_HISTORY_TABLE}(
            transport_session_id, delivered_at, outcome_end
        )
        WHERE transport_session_id IS NOT NULL;

    CREATE INDEX IF NOT EXISTS idx_recall_delivery_history_scope_task_time
        ON {RECALL_DELIVERY_HISTORY_TABLE}(scope, task, delivered_at, outcome_end)
        WHERE task IS NOT NULL;

    -- Late attach_recall_map calls resolve already-recorded consumers from
    -- these normalized rows without touching legacy JSON.
    CREATE INDEX IF NOT EXISTS idx_recall_history_events_transport_time
        ON {RECALL_HISTORY_EVENT_TABLE}(transport_session_id, occurred_at)
        WHERE transport_session_id IS NOT NULL;

    CREATE INDEX IF NOT EXISTS idx_recall_history_events_scope_task_time
        ON {RECALL_HISTORY_EVENT_TABLE}(scope, task, occurred_at)
        WHERE task IS NOT NULL;

    -- Both directions of the lookup correlation are range probes on this
    -- index: a new delivery asks "was this node fetched in (delivered_at,
    -- outcome_end]" and a new lookup asks the ledger the mirror question.
    CREATE INDEX IF NOT EXISTS idx_recall_lookup_events_node_time
        ON {RECALL_LOOKUP_EVENT_TABLE}(node_id, occurred_at);

    -- Serves the MIN(occurred_at) observability epoch alone.  The composite
    -- index above cannot: its leading column is node_id, so the earliest
    -- lookup overall is not at either of its ends.
    CREATE INDEX IF NOT EXISTS idx_recall_lookup_events_time
        ON {RECALL_LOOKUP_EVENT_TABLE}(occurred_at);
"""

#: Transcript-grounding verdict ledger (additive to schema v8; goal
#: transcript-grounding, phase 3). Public, unlike its v7/v8 siblings, because
#: it genuinely has a second caller outside this module:
#: ``living_memory.transcript_ledger.ensure_ledger_table`` runs the same
#: script so the offline importer can create the table on a database file the
#: redeployed server has not opened yet. One constant, two callers, exactly
#: one shape — the ``_QUERY_ANCHOR_SCHEMA_SQL`` rule.
#:
#: There is deliberately no ``_migrate_pre_v9_schema`` and no
#: ``SCHEMA_VERSION`` bump, for the reason recorded on
#: ``_RECALL_ATTESTATION_SCHEMA_SQL``: ``_initialize_schema`` runs ``CREATE
#: TABLE IF NOT EXISTS`` on every open, so that *is* the whole migration for
#: a brand-new table that has never shipped in any other shape, and there is
#: no version-keyed branch a bump could guard. Strictly additive: one table
#: and one index touching nothing that exists, so the ``nodes`` /
#: ``connections`` / ``recall_events`` DDL a live database already carries
#: stays byte-identical (``tests/test_transcript_ledger.py`` pins this
#: against master's fresh-database schema).
TRANSCRIPT_GROUNDING_SCHEMA_SQL = f"""
    -- Offline transcript-grounding verdicts (goal transcript-grounding,
    -- phase 3): one row is one graded delivery of one node by one recall
    -- event — the IDF-containment of the delivered content against the
    -- session transcript remainder AFTER the delivery instant, at the
    -- calibrated threshold behind `method_version`. Rows arrive only through
    -- scripts/transcript_ledger_import.py (living_memory.transcript_ledger);
    -- nothing on the live recall/remember path reads or writes this table.
    -- The phase-5 valve is the intended consumer, env-gated and shipped
    -- separately.
    --
    -- UNIQUE(recall_event_id, node_id, method_version) is the idempotency
    -- key: re-running the importer over the same verdict file replays
    -- instead of duplicating, and a re-grade under a *new* method_version
    -- lands beside the old rows instead of silently rewriting history. The
    -- importer never updates an existing row — a same-key row with different
    -- numbers is a method_version discipline violation, reported and not
    -- applied.
    --
    -- No foreign keys, deliberately (the recall_lookup_events rationale):
    -- verdicts are graded offline against a read-only snapshot, so by import
    -- time the event may be pruned or the node hard-deleted, and a
    -- `field = 'alt'` row names ids from the *other* host's store, which
    -- this file never carried. A row referencing nothing is still a valid
    -- measurement; this ledger is a record, never a gate.
    CREATE TABLE IF NOT EXISTS {TRANSCRIPT_GROUNDING_TABLE} (
        id TEXT PRIMARY KEY,
        recall_event_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        containment REAL NOT NULL CHECK (containment >= 0.0),
        grounded INTEGER NOT NULL CHECK (grounded IN (0, 1)),
        method_version TEXT NOT NULL,
        field TEXT NOT NULL CHECK (field IN ('local', 'alt')),
        delivered_at TEXT NOT NULL,
        graded_at TEXT NOT NULL,
        transcript_session_key TEXT NOT NULL,
        imported_at TEXT NOT NULL,
        UNIQUE(recall_event_id, node_id, method_version)
    );

    -- The phase-5 valve's read: "the graded verdicts for one node, newest
    -- first". The UNIQUE index leads with recall_event_id and cannot serve
    -- a per-node probe.
    CREATE INDEX IF NOT EXISTS idx_transcript_grounding_node_graded
        ON {TRANSCRIPT_GROUNDING_TABLE}(node_id, graded_at DESC);
"""

#: Recall credit ledger (additive to schema v8; goal feedback-usage-signal,
#: lookup-credit). One row per (recall event, node) pair that has been
#: *credited* — its usefulness and its scope's retrieval weights updated —
#: by whichever usage signal arrived first: the
#: consuming trace grounding the node's content (``basis = 'grounded'``,
#: ``source_id`` is the trace id) or a same-session ``memory_lookup`` of the
#: delivered id (``basis = 'lookup'``, ``source_id`` is the lookup event id).
#: ``UNIQUE(recall_event_id, node_id)`` is the whole dedup rule: the second
#: signal for a pair finds the row taken and applies nothing, so credit is at
#: most once per pair no matter which signal comes first, and the row says
#: which one did. Consumers compose the claim and numeric updates in one
#: store transaction, so a failed effect leaves the pair available to a later
#: eligible call. Optional query-anchor work is independent of numeric credit.
#:
#: Private like ``_RECALL_ATTESTATION_SCHEMA_SQL``: only ``_initialize_schema``
#: runs it. No ``_migrate_pre_v9_schema`` and no ``SCHEMA_VERSION`` bump, for
#: the reason recorded there and on ``TRANSCRIPT_GROUNDING_SCHEMA_SQL``:
#: ``CREATE TABLE IF NOT EXISTS`` on every open *is* the migration for a
#: brand-new table, and there is no version-keyed branch a bump could guard.
#: Strictly additive — one table, no index beyond the UNIQUE constraint's own
#: (which leads with ``recall_event_id`` and so serves the per-event read),
#: touching nothing that exists — so every DDL string a live database already
#: carries stays byte-identical.
_RECALL_CREDIT_LEDGER_SCHEMA_SQL = f"""
    -- No foreign keys, deliberately (the recall_lookup_events rationale): a
    -- lookup names ids the caller was handed, and by the time a closure or a
    -- late lookup claims a pair the event may be pruned or the node hard
    -- deleted. A row referencing nothing still records that the pair's
    -- credit was spent; this ledger is a guard against double credit, never
    -- a gate on the write that earned it.
    CREATE TABLE IF NOT EXISTS {RECALL_CREDIT_LEDGER_TABLE} (
        recall_event_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        basis TEXT NOT NULL CHECK (basis IN ('grounded', 'lookup')),
        source_id TEXT NOT NULL,
        credited_at TEXT NOT NULL,
        UNIQUE(recall_event_id, node_id)
    );
"""

#: Explicit credit: a ``used`` mark an agent passed on memory_recall,
#: memory_remember or memory_teach (goal explicit-recall-feedback). A third
#: basis for the same once-per-(event, node) credit the ledger above guards.
#:
#: A side table rather than a rebuild of ``recall_credit_ledger``: that
#: table's ``CHECK (basis IN ('grounded', 'lookup'))`` cannot be altered in
#: place, and a copy/drop/rename rebuild would rewrite the live ledger on the
#: first open after deploy — not additive, and not reversible by rolling the
#: code back. The side table keeps every existing DDL string byte-identical
#: and is created like the ledger, ``CREATE TABLE IF NOT EXISTS`` on every
#: open. The price is that the dedup spans two tables:
#: :meth:`MemoryStore.claim_recall_credit` inserts into either one with a
#: single ``INSERT ... SELECT ... WHERE NOT EXISTS`` against the other, so
#: the check and the claim are one statement, one transaction, and at most
#: one of the two tables ever holds a given pair whichever basis arrives
#: first. Readers that only know the ledger (grounded|lookup) see explicit
#: credit as absent, which is what they measured before it existed.
_RECALL_EXPLICIT_CREDIT_SCHEMA_SQL = f"""
    -- No foreign keys, for the recall_credit_ledger reason: the row records
    -- that a pair's credit was spent, never gates the write that earned it.
    CREATE TABLE IF NOT EXISTS {RECALL_EXPLICIT_CREDIT_TABLE} (
        recall_event_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        source_id TEXT NOT NULL,
        credited_at TEXT NOT NULL,
        UNIQUE(recall_event_id, node_id)
    );
"""

#: Audit of every explicit ``used``/``irrelevant`` mark an agent passed,
#: accepted or not. ``accepted = 1`` only for an id a recall event on the
#: same transport session delivered (the newest such event within the
#: lookup-credit window); a dropped mark keeps its ``reject_reason``
#: (``not_delivered``, ``no_transport``, ``empty``, ``duplicate``,
#: ``conflict``) and ``recall_event_id = ''``. ``source_id`` is the trace id
#: for memory_remember/memory_teach and the new recall event id for
#: memory_recall; ``rank`` is the 0-based rank in the delivering event.
#: Recording happens under ``LM_EXPLICIT_FEEDBACK_POLICY`` ``audit`` and
#: ``credit``; ``off`` writes nothing. Additive, created on every open, no
#: foreign keys, no SCHEMA_VERSION bump — the ledger's rules. The one index
#: supports inspection of accepted and dropped marks by recall event.
#: Whole-content vectors the consolidation pass clusters traces by. Nodes no
#: longer carry a vector column (``node_chunk_embeddings`` holds per-window
#: vectors, and a multi-window node's whole-content vector is not any of its
#: chunks), so without this table every pass re-encoded every active trace of
#: the scope through the model — the pass's dominant cost, paid for the whole
#: corpus each time. Rows are keyed by the content fingerprint and the encoder
#: that produced them, so an edited node or a changed model is a miss, never a
#: stale hit. Values are float64 (``array('d')``) so a cached vector is
#: bit-identical to a fresh encode and clustering does not drift. Additive,
#: created on every open, no foreign keys, no SCHEMA_VERSION bump — the
#: ledger's rules.
_CONSOLIDATION_EMBEDDING_SCHEMA_SQL = f"""
    CREATE TABLE IF NOT EXISTS {CONSOLIDATION_EMBEDDING_TABLE} (
        node_id TEXT PRIMARY KEY,
        content_fingerprint TEXT NOT NULL,
        encoder TEXT NOT NULL,
        dimensions INTEGER NOT NULL CHECK (dimensions > 0),
        embedding BLOB NOT NULL,
        created_at TEXT NOT NULL
    );
"""

_RECALL_FEEDBACK_MARKS_SCHEMA_SQL = f"""
    CREATE TABLE IF NOT EXISTS {RECALL_FEEDBACK_MARKS_TABLE} (
        id INTEGER PRIMARY KEY,
        recall_event_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        mark TEXT NOT NULL CHECK (mark IN ('used', 'irrelevant')),
        accepted INTEGER NOT NULL,
        reject_reason TEXT,
        via_tool TEXT NOT NULL,
        source_id TEXT NOT NULL,
        transport_session_id TEXT,
        agent TEXT,
        rank INTEGER,
        marked_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_recall_feedback_marks_event
        ON {RECALL_FEEDBACK_MARKS_TABLE}(recall_event_id, mark, accepted);
"""

#: Scratch tokenizer behind :meth:`MemoryStore.term_document_frequencies`. It
#: folds caller-supplied terms with the *same* tokenizer that indexes
#: ``nodes_fts``, so a DF lookup matches the index by construction instead of
#: through a hand-written Python approximation of unicode61 — whose folding is
#: easy to get subtly wrong (case folding everywhere, but diacritic removal on
#: Latin characters only: "Café" folds to "cafe" while Cyrillic "й" survives).
#: The ``tokenize`` option must stay identical to the ``nodes_fts`` DDL in
#: ``_initialize_schema``. Both objects live in ``temp``: per-connection,
#: never written to the database file, gone with the connection — so no
#: migration concern attaches to them. The ``instance`` vocab type yields one
#: row per (token, source rowid), which both attributes tokens back to the
#: input term that produced them and exposes inputs that fold to more or
#: fewer than one token.
_FTS_TERM_TOKENIZER_SQL = (
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS temp.fts_term_tokenizer
        USING fts5(term, tokenize = 'unicode61')
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS temp.fts_term_tokenizer_instances
        USING fts5vocab(temp, 'fts_term_tokenizer', 'instance')
    """,
)

#: Batch encoder for chunk texts: takes the chunk texts of one node in order and
#: returns one vector per text. Injected via :meth:`MemoryStore.set_chunk_embedder`
#: so a caller that already holds a loaded model does not pay for a second one.
ChunkEmbedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]
_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
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

#: ``ambient_context`` as JSON a ``json_*`` function may safely be pointed at.
#: The column is declared TEXT with no constraint, so a legacy row can hold
#: anything; ``json_type``/``json_extract`` raise on a malformed document, and
#: an exception inside ``_initialize_schema`` is a database that cannot be
#: opened at all. An empty object locates no path, which is the same answer as
#: "the key is not there".
_AMBIENT_JSON_OR_EMPTY = (
    "CASE WHEN json_valid(ambient_context) THEN ambient_context ELSE '{}' END"
)

#: Rows :meth:`MemoryStore._backfill_recall_events_task_pattern` still owes a
#: column to. Written once and used for both the probe and the update so the
#: pass is provably terminal: whatever the probe finds, the update fills.
_TASK_PATTERN_BACKFILL_PREDICATE = f"""
    task_pattern IS NULL
    AND ambient_context LIKE '%"task_pattern"%'
    AND json_type({_AMBIENT_JSON_OR_EMPTY}, '$.task_pattern') = 'text'
"""


def _node_read_batches(
    node_ids: Iterable[str], *, batch_size: int = 999,
) -> Iterator[list[str]]:
    """Keep each caller's bounded ID batches in SQLite primary-key order."""
    ids = list(set(node_ids))
    for i in range(0, len(ids), batch_size):
        yield sorted(ids[i : i + batch_size])


def _content_fingerprint(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def pack_chunk_embedding(values: Iterable[float]) -> bytes:
    """Encode a vector as a little-endian float32 BLOB.

    ``array('f')`` is C ``float`` — 4 bytes per item on every platform CPython
    supports — written in native byte order, so it is byte-swapped on a
    big-endian host to keep the on-disk layout little-endian everywhere.

    The float64 -> float32 narrowing is the only lossy step: values that are
    exactly representable in float32 (which every value read back out of a BLOB
    is) round-trip bit-for-bit, and the encoding is idempotent for the rest.
    """

    buffer = array("f", values)
    if sys.byteorder != "little":  # pragma: no cover - little-endian CI
        buffer.byteswap()
    return buffer.tobytes()


def unpack_chunk_embedding(blob: bytes | memoryview, dimensions: int) -> list[float]:
    """Decode a float32 little-endian BLOB of exactly ``dimensions`` values.

    ``dimensions`` comes from the row that stored the BLOB, never from the byte
    length: a length-derived dimension cannot tell a truncated write from a
    short vector, so a mismatch has to be an error rather than a guess.
    """

    if dimensions < 0:
        raise ValueError("dimensions must not be negative")
    raw = bytes(blob)
    expected = dimensions * CHUNK_EMBEDDING_ITEMSIZE
    if len(raw) != expected:
        raise ValueError(
            f"chunk embedding blob is {len(raw)} bytes, expected {expected} "
            f"for {dimensions} float32 values"
        )
    buffer = array("f")
    buffer.frombytes(raw)
    if sys.byteorder != "little":  # pragma: no cover - little-endian CI
        buffer.byteswap()
    return list(buffer)


def recall_fingerprint(query: str, requested_scope: str | None) -> str:
    """Stable identity for one exact recall request: (query, requested scope).

    SHA-256 over the whitespace-collapsed query plus the requested scope.
    Query anchors and hubs share this identity. Exact-match only: no
    semantic clustering and no content inspection beyond whitespace
    normalization, so stored fingerprints stay stable across releases.
    """

    collapsed_query = " ".join(str(query).split())
    scope_part = "" if requested_scope is None else str(requested_scope)
    return hashlib.sha256(f"{collapsed_query}\n{scope_part}".encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class NodeEndpoint:
    """Stored identity and decay flag for reference-edge candidates."""

    id: str
    decayed: bool


@dataclass(frozen=True, slots=True)
class NodeContent:
    """Source identity, content and level for grounding and trace provenance."""

    id: str
    content: str
    level: NodeLevel


@dataclass(frozen=True, slots=True)
class ConceptBrowseFields:
    """Scalar source fields used to rank an empty-task concept browse."""

    id: str
    scope: str
    confidence: float
    usefulness_score: float
    access_count: int
    timestamp: str


@dataclass(frozen=True)
class QueryAnchor:
    """One remembered query: its text, its vector, and its reinforcement state.

    ``embedding`` is decoded eagerly via :func:`unpack_chunk_embedding` using
    the row's own ``dimensions``. The scan path
    (:meth:`MemoryStore.iter_query_anchor_vectors`) deliberately does not build
    these — it yields raw BLOB views so a matcher pays for decoding only the
    vectors it keeps.
    """

    id: str
    scope: str
    query: str
    fingerprint: str
    dimensions: int
    embedding: tuple[float, ...]
    reinforcement_count: int
    usefulness_score: float
    decayed: bool
    decay_reason: str | None
    first_seen: str
    last_matched_at: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class QueryAnchorEdge:
    """One weighted anchor -> node edge."""

    anchor_id: str
    target_id: str
    weight: float
    hits: int
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class QueryAnchorLink:
    """An edge and its anchor's label state; a missing anchor has no query."""

    target_id: str
    anchor_id: str
    weight: float
    query: str | None
    decayed: bool


@dataclass(frozen=True)
class RecallAttestation:
    """One graded submission of session evidence against one recall event.

    The durable record of a server-side verdict: what was submitted
    (``evidence_sha256`` over the canonical items, plus their count and size),
    what the server decided about each of the event's own results
    (``containments``), and what that decision was allowed to change
    (``grounded_node_ids``, ``anchor_ids``, ``credited``, ``closed``). A repeat
    of the same ``(recall_event_id, evidence_sha256)`` replays this record and
    applies nothing.
    """

    id: str
    recall_event_id: str
    evidence_sha256: str
    evidence_items: int
    evidence_chars: int
    min_containment: float
    containments: list[dict[str, Any]]
    grounded_node_ids: list[str]
    anchor_ids: list[str]
    credited: bool
    closed: bool
    closed_by_attestation: bool
    feedback_trace_id: str | None
    agent: str | None
    task: str | None
    session_id: str | None
    transport_session_id: str | None
    source_session_key: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class MaturedRecallHistory:
    """Frozen 24-hour delivery outcomes for one candidate node.

    ``available`` is intentionally not inferred from the counts.  Known
    absence is ``available=True`` with three zeroes; truncation, a malformed
    legacy ledger, and SQLite read failure carry ``None`` counts so a selector
    cannot accidentally turn an unknown history into the cold-start feature.

    The ``lookup_*`` counts are additive and describe a *different* signal on
    the same windows: not "was it delivered again" but "did somebody fetch it
    by id".  They are carried beside M/C/K rather than folded into them
    because the frozen evaluator reads the triple and must keep reading
    exactly what it read before.  ``lookup_known`` is the denominator a
    caller needs and cannot reconstruct: ``matured`` counts every window,
    ``lookup_known`` only the ones whose lookup outcome was observable.
    """

    available: bool
    matured: int | None
    consumed: int | None
    trailing_nonconsumed: int | None
    unavailable_reason: str | None = None
    #: Matured windows with a correlated id-fetch (``lookup_consumed = 1``).
    lookup_consumed: int | None = None
    #: Matured windows whose lookup outcome is recorded at all (0 or 1).
    lookup_known: int | None = None
    #: Newest-first run of *known* windows without a lookup, stopping at the
    #: first followed one.  Unobservable windows are skipped rather than
    #: breaking the run: a gap in recording is not evidence either way.
    lookup_trailing_absent: int | None = None

    @classmethod
    def known(
        cls,
        matured: int,
        consumed: int,
        trailing_nonconsumed: int,
        *,
        lookup_consumed: int = 0,
        lookup_known: int = 0,
        lookup_trailing_absent: int = 0,
    ) -> "MaturedRecallHistory":
        return cls(
            available=True,
            matured=int(matured),
            consumed=int(consumed),
            trailing_nonconsumed=int(trailing_nonconsumed),
            lookup_consumed=int(lookup_consumed),
            lookup_known=int(lookup_known),
            lookup_trailing_absent=int(lookup_trailing_absent),
        )

    @classmethod
    def unavailable(cls, reason: str) -> "MaturedRecallHistory":
        return cls(
            available=False,
            matured=None,
            consumed=None,
            trailing_nonconsumed=None,
            unavailable_reason=str(reason),
        )

    # Short aliases mirror the M/C/K notation in the frozen policy without
    # forcing callers to use unexplained one-letter constructor fields.
    @property
    def m(self) -> int | None:
        return self.matured

    @property
    def c(self) -> int | None:
        return self.consumed

    @property
    def k(self) -> int | None:
        return self.trailing_nonconsumed


@dataclass(frozen=True)
class RecallWork:
    """Current map consumer, derived from existing recall-event fields.

    An explicit pattern or task names work across connections. Otherwise only
    the transport identifies a consumer. Labels are opaque, case-sensitive
    strings; no knowledge-search normalization merges independent work.
    """

    scope: str
    kind: str
    value: str

    @classmethod
    def resolve(
        cls, scope: str, *, task: Any = None, task_pattern: Any = None,
        transport_session_id: Any = None,
    ) -> RecallWork | None:
        label = _optional_str(task_pattern) or _optional_str(task)
        if label:
            return cls(normalize_scope(scope), "work", label)
        transport = _optional_str(transport_session_id)
        return cls(normalize_scope(scope), "transport", transport) if transport else None

    @property
    def key(self) -> str:
        return json.dumps((self.scope, self.kind, self.value), separators=(",", ":"))

    def predicate(self, alias: str = "") -> tuple[str, tuple[str, ...]]:
        """SQL over recall_events, with the same precedence as resolution."""

        prefix = f"{alias}." if alias else ""
        pattern = f"{prefix}task_pattern"
        task = f"{prefix}task"
        no_pattern = f"({pattern} IS NULL OR {pattern} = '')"
        if self.kind == "work":
            identity = f"({pattern} = ? OR ({no_pattern} AND {task} = ?))"
            values = (self.value, self.value)
        else:
            identity = (f"{no_pattern} AND ({task} IS NULL OR {task} = '') "
                        f"AND {prefix}transport_session_id = ?")
            values = (self.value,)
        return f"{prefix}scope = ? AND ({identity})", (self.scope, *values)


class MemoryStore:
    """Stable CRUD API over one SQLite file.

    Raw traces are append-only: callers can add stats, provenance, or decay markers,
    but trace content is never updated in place.
    """

    def __init__(
        self,
        config: MemoryConfig | str | Path | None = None,
        base_config: MemoryConfig | None = None,
        *,
        chunk_embedder: ChunkEmbedder | None = None,
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
        self._chunk_embedder_lock = RLock()
        self.set_chunk_embedder(chunk_embedder)
        self._node_embedding_column_cache: bool | None = None
        self._chunk_table_cache: bool | None = None
        self._anchor_tables_cache: bool | None = None
        # TEMP objects are per-connection, so this cache cannot go stale the
        # way an on-disk schema probe can.
        self._fts_term_tokenizer_ready = False
        self._savepoint_serial = 0
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

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[None]:
        """Compose reads/writes; optionally reserve an outer writer."""

        if immediate and not self._conn.in_transaction:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return

        self._savepoint_serial = getattr(self, "_savepoint_serial", 0) + 1
        name = f"memory_write_{self._savepoint_serial}"
        self._conn.execute(f"SAVEPOINT {name}")
        try:
            yield
        except BaseException:
            self._conn.execute(f"ROLLBACK TO {name}")
            raise
        finally:
            self._conn.execute(f"RELEASE {name}")

    def nodes_referencing_source(self, source_id: str) -> list[Node]:
        return self.nodes_referencing_sources([source_id])[source_id]

    def nodes_referencing_sources(self, source_ids: Iterable[str]) -> dict[str, list[Node]]:
        """Load active recorded dependents after admitting their source identity.

        Project only the fields used by Node's materializer policy. Ordinary
        recall citations must not load a body, context or full provenance.
        Known writers own their inputs; explicit derived_from edges are the
        fallback for other nodes. Dependencies can cross scope boundaries.
        Each bounded group shares one corpus scan and selected body read.
        Preserve each source's discovery order independently of body order.
        """

        result: dict[str, list[Node]] = {}
        for batch in _node_read_batches(source_ids, batch_size=64):
            result.update(self._nodes_referencing_source_batch(batch))
        return result

    def _nodes_referencing_source_batch(self, source_ids: list[str]) -> dict[str, list[Node]]:
        placeholders = ",".join("?" for _ in source_ids)
        rows = self._conn.execute(
            f"""
            WITH explicit AS MATERIALIZED (
                SELECT source_id, target_id FROM connections
                WHERE target_id IN ({placeholders}) AND type = 'related'
                  AND json_extract(metadata, '$.kind') = 'derived_from'
            )
            SELECT n.id, n.level, n.source_traces,
                   n.provenance -> '$.strategy' AS strategy,
                   json_type(n.provenance, '$.cluster_key') IS NOT NULL AS has_cluster_key,
                   n.provenance -> '$.source_concepts' AS source_concepts,
                   (SELECT json_group_array(target_id) FROM explicit
                    WHERE source_id = n.id) AS explicit_dependencies
            FROM nodes n
            WHERE n.decayed = 0 AND (
                EXISTS (SELECT 1 FROM json_each(n.source_traces)
                        WHERE value IN ({placeholders}))
                OR EXISTS (
                    SELECT 1 FROM json_each(n.provenance, '$.source_concepts')
                    WHERE value IN ({placeholders})
                )
                OR n.id IN (SELECT source_id FROM explicit)
            )
            ORDER BY n.level, n.scope, n.id
            """,
            source_ids * 3,
        )
        admitted: dict[str, list[str]] = {source: [] for source in source_ids}
        for row in rows:
            provenance = {
                "strategy": _json_loads(row["strategy"], None),
                "source_concepts": _json_loads(row["source_concepts"], []),
            }
            if row["has_cluster_key"]:
                provenance["cluster_key"] = None
            reference = Node(
                id=str(row["id"]), level=row["level"], content="",
                source_traces=_json_loads(row["source_traces"], []), provenance=provenance,
            )
            sources = (
                reference.materializer_sources if reference.materializer
                else _json_loads(row["explicit_dependencies"], [])
            )
            for source in source_ids:
                if source in sources:
                    admitted[source].append(reference.id)
        nodes = self.get_nodes(node_id for ids in admitted.values() for node_id in ids)
        return {source: [nodes[node_id] for node_id in ids if node_id in nodes]
                for source, ids in admitted.items()}

    # ------------------------------------------------------------------
    # Schema-shape probes
    #
    # Three shapes are live at once during the v6 rollout: pre-v6 (no chunk
    # table), post-v6 (chunks plus nodes.embedding, what the production server
    # reads while the backfill runs), and post-drop (chunks only, after an
    # operator calls drop_node_embedding_column). Read-only stores that never
    # run migrations — health_audit.ReadOnlyAuditStore, and anything opening a
    # snapshot — meet whichever shape the file happens to have, so every read
    # path that names `nodes.embedding` or the chunk table asks first instead of
    # assuming. Cached because they answer from DDL that only an explicit
    # migration step changes, and both such steps invalidate the cache.
    # ------------------------------------------------------------------

    def _node_embedding_column_present(self, *, refresh: bool = False) -> bool:
        """Whether the legacy ``nodes.embedding`` JSON column still exists."""

        cached = getattr(self, "_node_embedding_column_cache", None)
        if cached is not None and not refresh:
            return bool(cached)
        columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(nodes)").fetchall()
        }
        present = "embedding" in columns
        self._node_embedding_column_cache = present
        return present

    def _chunk_table_present(self, *, refresh: bool = False) -> bool:
        """Whether ``node_chunk_embeddings`` exists (false on a pre-v6 file)."""

        cached = getattr(self, "_chunk_table_cache", None)
        if cached is not None and not refresh:
            return bool(cached)
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (CHUNK_EMBEDDING_TABLE,),
        ).fetchone()
        present = row is not None
        self._chunk_table_cache = present
        return present

    def _anchor_tables_present(self, *, refresh: bool = False) -> bool:
        """Whether the v7 query-anchor tables exist (false on a pre-v7 file).

        Every anchor read degrades to "no anchors" on a database that lacks
        them, so a snapshot opened read-only — by ``health_audit`` or by an
        offline script that never runs migrations — reports an empty anchor
        graph instead of raising ``no such table``.
        """

        cached = getattr(self, "_anchor_tables_cache", None)
        if cached is not None and not refresh:
            return bool(cached)
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?)",
            (QUERY_ANCHOR_TABLE, QUERY_ANCHOR_EDGE_TABLE),
        ).fetchall()
        present = len(rows) == 2
        self._anchor_tables_cache = present
        return present

    def _invalidate_schema_shape_cache(self) -> None:
        self._node_embedding_column_cache = None
        self._chunk_table_cache = None
        self._anchor_tables_cache = None

    def get_kv(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM kv WHERE key = ?",
            (str(key),),
        ).fetchone()
        return str(row["value"]) if row else None

    def set_kv(self, key: str, value: str) -> None:
        now = _utc_now()
        with self.transaction():
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
        representative_embedding: Iterable[float] | None = None,
        prepared_chunks: Sequence[tuple[TextChunk, Sequence[float]]] | None = None,
        stats: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        node_id: str | None = None,
        timestamp: str | None = None,
    ) -> Node:
        """Store content and its search chunks.

        ``embedding`` describes this content and can supply its single chunk.
        ``representative_embedding`` is for consolidation similarity only;
        it is stored in the legacy node column and never reused as a chunk.
        ``prepared_chunks`` applies content vectors prepared by the caller
        without entering the encoder from the write transaction.
        """

        with self.transaction():
            new_id = self._insert_node(
                level=level,
                content=content,
                context=context,
                embedding=embedding,
                representative_embedding=representative_embedding,
                prepared_chunks=prepared_chunks,
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
        representative_embedding: Iterable[float] | None = None,
        prepared_chunks: Sequence[tuple[TextChunk, Sequence[float]]] | None = None,
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

        candidate = Node(
            id=new_id, level=level, content=content, context=context_data,
            provenance=provenance_data, source_traces=source_traces,
            corrections=corrections, scope=scope,
        )
        duplicate_ids = []
        identity_ready = False
        identity = None
        for row in self._conn.execute(
            """
            SELECT * FROM nodes
            WHERE level = ?
              AND scope = ?
              AND content_fingerprint = ?
              AND decayed = 0
            """,
            (level, scope, fingerprint),
        ).fetchall():
            peer = _node_from_row(row)
            if not identity_ready:
                identity = content_identity(candidate)
                identity_ready = True
            if identity is not None and identity == content_identity(peer):
                duplicate_ids.append(str(row["id"]))

        content_vector = None if embedding is None else list(embedding)
        vector = (
            list(representative_embedding)
            if representative_embedding is not None else content_vector
        )
        legacy_embedding = self._node_embedding_column_present()
        embedding_column = "embedding, " if legacy_embedding else ""
        embedding_placeholder = "?, " if legacy_embedding else ""
        embedding_value = (
            (_json_dumps(vector) if vector is not None else None,) if legacy_embedding else ()
        )
        self._conn.execute(
            f"""
            INSERT INTO nodes (
                id, level, content, content_fingerprint, {embedding_column}scope, agent, task, context,
                timestamp, decayed, decay_reason, access_count, last_accessed,
                usefulness_score, confidence, unique_agents, temporal_hint,
                source_traces, corrections, provenance, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, {embedding_placeholder}?, ?, ?, ?, ?, 0, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                level,
                content,
                fingerprint,
                *embedding_value,
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

        # A supplied vector makes content searchable now. Only content_vector
        # can supply a single chunk; representatives require content encoding.
        # Callers without a vector own eager indexing or leave bounded repair.
        if prepared_chunks is not None:
            self._replace_node_chunks(new_id, prepared_chunks, fingerprint)
        elif vector:
            self._write_node_chunks(new_id, content, fingerprint, content_vector)

        for old_id in duplicate_ids:
            self._insert_connection(
                new_id,
                old_id,
                "related",
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
        node_id: str | None = None,
    ) -> Node:
        stats = dict(feedback or {})
        return self.create_node(
            level="trace", content=content, context=context, stats=stats, node_id=node_id,
        )

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
        with self.transaction():
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
        nodes: dict[str, Node] = {}
        for chunk in _node_read_batches(node_ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT * FROM nodes WHERE id IN ({placeholders}) ORDER BY id", chunk
            ).fetchall()
            for row in rows:
                node = _node_from_row(row)
                nodes[node.id] = node
        return nodes

    def get_recall_nodes(self, node_ids: Iterable[str], plan: ScopePlan) -> dict[str, Node]:
        """Materialize live candidates admitted by the current recall plan.

        Admission and body selection share one SQL statement per batch, so an
        excluded node's body is neither returned nor decoded. Exact readers
        retain their independent full-node and decayed-history contracts.
        """
        nodes: dict[str, Node] = {}
        for row in self._recall_node_rows(node_ids, plan, columns="*"):
            node = _node_from_row(row)
            nodes[node.id] = node
        return nodes

    def get_recall_node_ids(self, node_ids: Iterable[str], plan: ScopePlan) -> set[str]:
        """Read admitted identities for source paths without selecting bodies."""
        return {str(row["id"]) for row in self._recall_node_rows(node_ids, plan, columns="id")}

    def _recall_node_rows(
        self, node_ids: Iterable[str], plan: ScopePlan, *, columns: Literal["*", "id"],
    ) -> Iterator[sqlite3.Row]:
        scopes = plan.named_scopes if plan.restricted else ()
        if plan.restricted and not scopes:
            return
        scope_filter = f" AND scope IN ({','.join('?' for _ in scopes)})" if scopes else ""
        for chunk in _node_read_batches(node_ids):
            placeholders = ",".join("?" for _ in chunk)
            yield from self._conn.execute(
                f"SELECT {columns} FROM nodes WHERE id IN ({placeholders}) AND decayed = 0"
                f"{scope_filter} ORDER BY id", (*chunk, *scopes),
            ).fetchall()

    def get_node_scopes(self, node_ids: Iterable[str]) -> dict[str, str]:
        """Read existing identities and authoritative scopes, including decayed nodes."""
        scopes: dict[str, str] = {}
        for chunk in _node_read_batches(node_ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT id, scope FROM nodes WHERE id IN ({placeholders}) ORDER BY id", chunk
            ).fetchall()
            scopes.update((str(row["id"]), str(row["scope"])) for row in rows)
        return scopes

    def get_node_endpoints(self, node_ids: Iterable[str]) -> dict[str, NodeEndpoint]:
        """Read current reference endpoints without materializing node bodies."""
        endpoints: dict[str, NodeEndpoint] = {}
        for chunk in _node_read_batches(node_ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT id, decayed FROM nodes WHERE id IN ({placeholders}) ORDER BY id", chunk
            ).fetchall()
            for row in rows:
                endpoint = NodeEndpoint(id=str(row["id"]), decayed=bool(row["decayed"]))
                endpoints[endpoint.id] = endpoint
        return endpoints

    def get_node_contents(self, node_ids: Iterable[str]) -> dict[str, NodeContent]:
        """Read source facts without context or statistics, including decayed nodes."""
        contents: dict[str, NodeContent] = {}
        for chunk in _node_read_batches(node_ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT id, content, level FROM nodes WHERE id IN ({placeholders}) ORDER BY id", chunk
            ).fetchall()
            for row in rows:
                source = NodeContent(id=str(row["id"]), content=str(row["content"]), level=row["level"])
                contents[source.id] = source
        return contents

    def get_node_knowledge(
        self, node_ids: Iterable[str], *, context_key: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Read exact content and stored validity without materializing Nodes.

        JSON projection happens in SQLite; only selected values cross into
        Python. Raw JSON preserves null, booleans and large integers. Relation
        validity and correction-text shaping remain the readers' shared work.
        """
        context_column = ""
        params: tuple[str, ...] = ()
        if context_key is not None:
            # JSON path quotes delimit a literal key. Encode embedded quotes as
            # Unicode escapes so dots/brackets in the same key remain literal.
            key = _json_dumps(context_key)[1:-1].replace(r'\"', r'\u0022')
            params = (f'$."{key}"',)
            context_column = ", context -> ? AS context_value"
        nodes: dict[str, dict[str, Any]] = {}
        for chunk in _node_read_batches(node_ids):
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                "SELECT id, level, content, scope, decayed, decay_reason, corrections,"
                " provenance -> '$.supersedes' AS supersedes"
                f"{context_column} FROM nodes WHERE id IN ({placeholders}) ORDER BY id",
                (*params, *chunk),
            ).fetchall()
            for row in rows:
                provenance = {"corrections": _json_loads(row["corrections"], [])}
                if row["supersedes"] is not None:
                    provenance["supersedes"] = _json_loads(row["supersedes"], None)
                node = {
                    "id": str(row["id"]), "level": row["level"],
                    "content": str(row["content"]), "scope": str(row["scope"]),
                    "decayed": bool(row["decayed"]), "decay_reason": row["decay_reason"],
                    "provenance": provenance,
                }
                if context_key is not None:
                    node["context"] = (
                        {context_key: _json_loads(row["context_value"], None)}
                        if row["context_value"] is not None else {}
                    )
                nodes[node["id"]] = node
        return nodes

    def list_nodes(
        self,
        *,
        level: NodeLevel | None = None,
        scope: str | None = None,
        source_ids: Iterable[str] | None = None,
        include_decayed: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Node]:
        """List nodes, optionally citing any of the given source identities.

        Source traces and source concepts are citations; callers still verify
        materializer ownership. Related edges alone are not recorded inputs.
        """
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
        if source_ids is not None:
            sources = _json_dumps(list(source_ids))
            clauses.append("""(
                EXISTS (
                    SELECT 1 FROM json_each(source_traces)
                    WHERE value IN (SELECT value FROM json_each(?))
                ) OR EXISTS (
                    SELECT 1 FROM json_each(provenance, '$.source_concepts')
                    WHERE value IN (SELECT value FROM json_each(?))
                )
            )""")
            params.extend([sources, sources])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([int(limit), int(offset)])
        rows = self._conn.execute(
            f"SELECT * FROM nodes {where} ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [_node_from_row(row) for row in rows]

    def list_concept_browse_fields(self, *, scope: str, limit: int) -> list[ConceptBrowseFields]:
        """Read the newest active concept prefix before consumer filtering."""
        rows = self._conn.execute(
            """
            SELECT id, scope, confidence, usefulness_score, access_count, timestamp
            FROM nodes WHERE level = 'concept' AND scope = ? AND decayed = 0
            ORDER BY timestamp DESC, id DESC LIMIT ?
            """,
            (scope, int(limit)),
        ).fetchall()
        return [ConceptBrowseFields(
            id=str(row["id"]), scope=str(row["scope"]),
            confidence=float(row["confidence"]),
            usefulness_score=float(row["usefulness_score"]),
            access_count=int(row["access_count"]), timestamp=str(row["timestamp"]),
        ) for row in rows]

    def list_global_concept_provenance(self, *, limit: int) -> list[tuple[str, dict[str, Any]]]:
        """Read identity/provenance for the newest active global concept prefix.

        Apply the limit before consumer-specific admission. Observation counts
        promotion metadata; refresh admits recorded materializer owners and
        rereads their full current state in its writer transaction.
        """
        rows = self._conn.execute(
            """
            SELECT id, provenance FROM nodes
            WHERE level = 'concept' AND scope = 'global' AND decayed = 0
            ORDER BY timestamp DESC, id DESC LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        result = []
        for row in rows:
            provenance = _json_loads(row["provenance"], {})
            if not isinstance(provenance, dict):
                raise TypeError("node provenance must be a JSON object")
            result.append((str(row["id"]), provenance))
        return result

    def schema_triggers(
        self, scope: str | None, *, strategy: str | None = None,
    ) -> list[tuple[str, str]]:
        """``(id, trigger)`` of live triggered schemas and group carriers in ``scope`` (``None``: all).

        Reads two columns instead of whole nodes: a recall compares every
        trigger in its scopes and fetches only the schemas it matched.
        A strategy selector also reads provenance with the node decoder, so a
        materializer can discover its owners before hydrating their bodies.
        """

        projection = ", provenance" if strategy is not None else ""
        rows = self._conn.execute(
            f"""
            SELECT id, JSON_EXTRACT(context, '$.trigger') {projection}
            FROM nodes
            WHERE level IN ('schema', 'concept') AND (? IS NULL OR scope = ?) AND decayed = 0
              AND JSON_EXTRACT(context, '$.trigger') IS NOT NULL
            """,
            (scope, scope),
        ).fetchall()
        result = []
        for row in rows:
            if not row[1]:
                continue
            if strategy is not None:
                provenance = _json_loads(row["provenance"], {})
                if not isinstance(provenance, dict):
                    raise TypeError("node provenance must be a JSON object")
                if provenance.get("strategy") != strategy:
                    continue
            result.append((str(row[0]), str(row[1])))
        return result

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
        representative_embedding: Iterable[float] | None = None,
        prepared_chunks: Sequence[tuple[TextChunk, Sequence[float]]] | None = None,
        stats: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        level: NodeLevel | None = None,  # moves the node, same id and history, to a level
    ) -> Node:
        """Update changed facts atomically, using the vector roles of create_node."""

        row = self._conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(node_id)
        existing = _node_from_row(row)
        if existing.level == "trace" and content is not None and content != existing.content:
            raise ValueError("trace content is append-only and cannot be updated")

        context_data = dict(existing.context if context is None else context)
        context_data.setdefault("scope", existing.scope)
        context_data.setdefault("timestamp", existing.timestamp)
        stats_data = existing.stats
        if stats:
            stats_data.update(stats)
        provenance_data = dict(existing.provenance if provenance is None else provenance)
        source_traces = _as_list(provenance_data.pop("source_traces", existing.source_traces))
        corrections = _as_list(provenance_data.pop("corrections", existing.corrections))
        new_content = existing.content if content is None else content
        content_changed = new_content != existing.content
        content_vector = None if embedding is None else list(embedding)
        vector = (
            list(representative_embedding)
            if representative_embedding is not None else content_vector
        )
        unique_agents = int(stats_data.get("unique_agents", existing.unique_agents))
        confidence = float(stats_data.get("confidence", existing.confidence))
        if unique_agents <= 1:
            confidence = min(confidence, 0.5)

        values = {
            "level": level or existing.level,
            "scope": str(context_data.get("scope") or existing.scope),
            "agent": _optional_str(context_data.get("agent", existing.agent)),
            "task": _optional_str(context_data.get("task", existing.task)),
            "access_count": int(stats_data.get("access_count", existing.access_count)),
            "last_accessed": _optional_str(stats_data.get("last_accessed")),
            "usefulness_score": float(stats_data.get("usefulness_score", existing.usefulness_score)),
            "confidence": confidence,
            "unique_agents": unique_agents,
            "temporal_hint": _optional_str(stats_data.get("temporal_hint")),
        }
        values = {key: value for key, value in values.items() if value != getattr(existing, key)}
        if content_changed:
            values["content"] = new_content
            values["content_fingerprint"] = _content_fingerprint(new_content)
        # Omitted JSON families retain their stored bytes. Compare supplied
        # JSON as JSON: Python equality conflates values such as 1 and True.
        json_values = {}
        if context is not None or context_data != existing.context:
            json_values["context"] = context_data
        if provenance is not None:
            json_values["provenance"] = provenance_data
            if "source_traces" in provenance:
                json_values["source_traces"] = source_traces
            if "corrections" in provenance:
                json_values["corrections"] = corrections
        if vector is not None and self._node_embedding_column_present():
            json_values["embedding"] = vector
        for key, value in json_values.items():
            encoded = _json_dumps(value)
            if encoded != row[key]:
                values[key] = encoded
        values["updated_at"] = _utc_now()

        with self.transaction():
            self._conn.execute(
                f"UPDATE nodes SET {', '.join(f'{key} = ?' for key in values)} WHERE id = ?",
                (*values.values(), node_id),
            )
            # Representatives affect clustering, not content search. An unchanged
            # digest with current chunks needs no encoding or chunk replacement.
            if prepared_chunks is not None:
                self._replace_node_chunks(node_id, prepared_chunks, _content_fingerprint(new_content))
            elif content_vector or (vector and representative_embedding is not None and (
                content_changed or not self.node_chunks_are_current(node_id)
            )):
                fingerprint = values.get("content_fingerprint") or _content_fingerprint(new_content)
                self._write_node_chunks(node_id, new_content, fingerprint, content_vector)
            elif content_changed and self._chunk_table_present():
                self._conn.execute(
                    f"DELETE FROM {CHUNK_EMBEDDING_TABLE} WHERE node_id = ?",
                    (node_id,),
                )
        return self.get_node(node_id)  # type: ignore[return-value]

    def add_correction(
        self,
        node_id: str,
        *,
        old: str,
        new: str,
        by: str,
        stats: Mapping[str, Any] | None = None,
    ) -> Node:
        """Append correction metadata and apply accompanying stats in one update."""

        existing = self.get_node(node_id)
        if existing is None:
            raise KeyError(node_id)
        corrections = list(existing.corrections)
        corrections.append({"timestamp": _utc_now(), "old": old, "new": new, "by": by})
        provenance = dict(existing.provenance)
        provenance["corrections"] = corrections
        provenance["source_traces"] = existing.source_traces
        return self.update_node(node_id, provenance=provenance, stats=stats)

    def soft_delete_node(self, node_id: str, reason: str | None = None) -> Node:
        existing = self.get_node(node_id)
        if existing is None:
            raise KeyError(node_id)
        now = _utc_now()
        with self.transaction():
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
        merge: bool = False,
    ) -> Connection:
        """Replace an edge, or merge a contribution with ``merge=True``.

        Contributions take the maximum weight, clamped to [0, 1] and rounded
        to six decimals, and shallow-merge metadata with incoming keys winning.
        An unchanged contribution retains the row and its updated_at; every
        correction or verified exact-copy call still moves attached anchors.
        A semantic contribution replaces prior technical provenance on the
        same pair instead of inheriting its bookkeeping marker or weight.
        """

        with self.transaction(immediate=True):
            return self._insert_connection(
                source_id,
                target_id,
                relation_type,
                weight=weight,
                metadata=metadata,
                merge=merge,
                validate_endpoints=True,
            )

    def _insert_connection(
        self,
        source_id: str,
        target_id: str,
        relation_type: ConnectionType,
        *,
        weight: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
        merge: bool = False,
        validate_endpoints: bool = False,
    ) -> Connection:
        """Apply one edge write inside the caller's transaction."""

        self._validate_connection_type(relation_type)
        existing = None
        metadata_data = dict(metadata or {})
        if merge:
            row = self._conn.execute(
                """
                SELECT * FROM connections
                WHERE source_id = ? AND target_id = ? AND type = ?
                """,
                (source_id, target_id, relation_type),
            ).fetchone()
            if row is not None:
                existing = _connection_from_row(row)
                if (is_semantic_relation(relation_type, metadata_data)
                        and not is_semantic_relation(existing.type, existing.metadata)):
                    # Fresh semantic evidence replaces technical history; the
                    # old marker and weight do not describe this contribution.
                    existing = None
                else:
                    weight = max(float(weight), existing.weight)
                    metadata_data = {**existing.metadata, **metadata_data}
            weight = round(min(1.0, max(0.0, float(weight))), 6)

        # An existing edge's foreign keys already establish both endpoints.
        # New edges and replacement writes validate identity without node payloads.
        if validate_endpoints and existing is None:
            for node_id in (source_id, target_id):
                if self._conn.execute("SELECT 1 FROM nodes WHERE id = ?", (node_id,)).fetchone() is None:
                    raise KeyError(node_id)

        now = _utc_now()
        metadata_json = _json_dumps(metadata_data)
        if (existing is not None and weight == existing.weight
                and metadata_json == _json_dumps(existing.metadata)):
            connection = existing
        else:
            row = self._conn.execute(
                """
                INSERT INTO connections (
                    id, source_id, target_id, type, weight, metadata, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, target_id, type) DO UPDATE SET
                    weight = excluded.weight,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                RETURNING *
                """,
                (
                    new_ulid(),
                    source_id,
                    target_id,
                    relation_type,
                    float(weight),
                    metadata_json,
                    now,
                    now,
                ),
            ).fetchone()
            connection = _connection_from_row(row)
        duplicate = (
            relation_type in ("related", "supersedes")
            and metadata_data.get("kind") == _DUPLICATE_CONTENT_KIND
            and any(node.id == source_id for node in self.list_duplicate_replacements(target_id))
        )
        if is_correction(relation_type, metadata_data) or duplicate:
            self._follow_replacement_with_anchor_edges(source_id, target_id, now=now)
        return connection

    def _follow_replacement_with_anchor_edges(
        self, source_id: str, target_id: str, *, now: str
    ) -> int:
        """Move anchors to an explicitly corrected or verified exact-copy target.

        The shared writer calls this only after checking correction semantics
        or exact duplicate identity. It runs inside that writer's transaction;
        a repeated contribution can therefore move newly attached anchors too.
        Each write moves one edge step, so cycles cannot recurse here. Era
        displacement has no relation and migrates anchors explicitly.
        """

        if source_id == target_id:
            return 0
        if not self._anchor_tables_present():
            return 0
        return self.repoint_query_anchor_edges(target_id, source_id, now=now)

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
        with self.transaction():
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
        with self.transaction():
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
        clauses: list[str] = [semantic_relation_sql()]
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

    def list_supersedes_pairs(self, node_ids: Iterable[str]) -> list[tuple[str, str]]:
        """Return unique (correction, original) pairs touching any requested id.

        Both directions matter even when the counterpart is decayed or outside
        the caller's scope. Endpoint indexes keep a small selection from walking
        the type-wide covering index. Known duplicate bookkeeping is excluded.
        """

        ids = sorted(set(node_ids))
        pairs: set[tuple[str, str]] = set()
        for offset in range(0, len(ids), 400):
            batch = ids[offset : offset + 400]
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"""
                SELECT source_id, target_id
                FROM connections INDEXED BY idx_connections_source_type
                WHERE source_id IN ({placeholders}) AND {correction_sql()}
                UNION
                SELECT source_id, target_id
                FROM connections INDEXED BY idx_connections_target_type
                WHERE target_id IN ({placeholders}) AND {correction_sql()}
                """,
                batch + batch,
            )
            pairs.update((str(row["source_id"]), str(row["target_id"])) for row in rows)
        return sorted(pairs)

    def list_duplicate_replacements(self, node_id: str) -> list[Node]:
        """Live exact copies, separately from semantic corrections."""
        return duplicate_replacements(self._conn, node_id)

    def list_duplicate_predecessors(self, node_id: str) -> list[Node]:
        """Exact prior copies for source lineage, including retained history."""
        return _duplicate_nodes(self._conn, node_id, successors=False)

    def list_connections_for_nodes(self, node_ids: Iterable[str]) -> dict[str, list[Connection]]:
        ids = list(set(node_ids))
        if not ids:
            return {}
        
        connections = []
        for i in range(0, len(ids), 400):
            chunk = ids[i : i + 400]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT * FROM connections WHERE (source_id IN ({placeholders}) OR target_id IN ({placeholders})) AND {semantic_relation_sql()} ORDER BY weight DESC, created_at DESC",
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

    def list_recall_connections(
        self, node_ids: Iterable[str], *, limit: int, source_only: bool = False,
    ) -> dict[str, list[Connection]]:
        """Bound each adjacency before decoding connections and their metadata.

        Keep the bulk reader's query, batch order and tie order. A connection
        shared by two requested nodes occupies a slot in each adjacency but
        needs only one object. Close the cursor as soon as every list is full.
        Source qualification projects and orders source contributions before
        this bound, so historical weights cannot spend its adjacency slots.
        """

        ids = list(set(node_ids))
        result: dict[str, list[Connection]] = {node_id: [] for node_id in ids}
        if limit <= 0:
            return result
        remaining = len(ids)
        seen: set[str] = set()
        weight = source_relation_weight_sql() if source_only else "weight"
        positive = f" AND ({weight}) > 0.0" if source_only else ""
        for i in range(0, len(ids), 400):
            chunk = ids[i : i + 400]
            placeholders = ",".join("?" for _ in chunk)
            cursor = self._conn.execute(
                f"SELECT *, {weight} AS recall_weight FROM connections WHERE (source_id IN ({placeholders}) OR target_id IN ({placeholders})) AND {semantic_relation_sql()}{positive} ORDER BY recall_weight DESC, created_at DESC",
                chunk + chunk,
            )
            try:
                for row in cursor:
                    if row["id"] in seen:
                        continue
                    targets = [
                        node_id for node_id in dict.fromkeys((row["source_id"], row["target_id"]))
                        if node_id in result and len(result[node_id]) < limit
                    ]
                    if not targets:
                        continue
                    connection = _connection_from_row(row)
                    if source_only:
                        connection = replace(connection, weight=float(row["recall_weight"]))
                    seen.add(connection.id)
                    for node_id in targets:
                        result[node_id].append(connection)
                        if len(result[node_id]) == limit:
                            remaining -= 1
                    if remaining == 0:
                        return result
            finally:
                cursor.close()
        return result

    def record_access(self, node_id: str) -> Node:
        if self.get_node(node_id) is None:
            raise KeyError(node_id)
        now = _utc_now()
        with self.transaction():
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
        recall_map: Mapping[str, Any] | None = None,
    ) -> RecallEvent:
        """Persist a recall event without recording node access."""

        _, event = self.record_recall(
            query=query,
            scope=scope,
            requested_scope=requested_scope,
            resolved_scopes=resolved_scopes,
            ambient_context=ambient_context,
            depth=depth,
            max_results=max_results,
            results=results,
            recall_map=recall_map,
            log_access=False,
        )
        assert event is not None
        return event

    def record_recall(
        self,
        *,
        query: str,
        nodes: Iterable[Node] = (),
        log_access: bool = True,
        log_event: bool = True,
        scope: str,
        requested_scope: str | None = None,
        resolved_scopes: Iterable[str] | None = None,
        ambient_context: Mapping[str, Any] | None = None,
        depth: str | int | None = None,
        max_results: int = 10,
        results: Iterable[Mapping[str, Any]] | None = None,
        recall_map: Mapping[str, Any] | None = None,
    ) -> tuple[list[Node], RecallEvent | None]:
        """Persist selected-node access and its recall event in one savepoint.

        Return the selected nodes in order with access metadata from SQLite's
        increments, preserving their selected content and validity. Missing
        nodes or event/history failures roll back the entire write.
        Access and event logging can be disabled independently. Recall composes
        this savepoint with its required response formation transaction; a
        failed projection rolls back consumption too. Network delivery remains
        outside that transaction.

        ``recall_map`` is the map served with this delivery, stored verbatim as
        compact JSON. The live recall path leaves it unset and calls
        :meth:`attach_recall_map` instead: the map is built from the residual
        pool *this* call produced, so it does not exist until the event has been
        recorded and the ranked results have been shaped. Callers that already
        hold a map when they record the event pass it here and pay one write
        instead of two.
        """

        selected = list(nodes)
        if not log_access and not log_event:
            return selected, None
        if not selected and not log_event:
            return selected, None
        normalized_results = [dict(item) for item in results or []]
        event_id = new_ulid()
        now = _utc_now()
        ambient = dict(ambient_context or {})
        agent = _optional_str(ambient.get("agent"))
        task = _optional_str(ambient.get("task"))
        # Lifted out of the ambient JSON exactly as `task` is, and for the same
        # consumer: it is the label the recall map keys on, so the delivery
        # history has to be filterable by it.
        task_pattern = _optional_str(ambient.get("task_pattern"))
        session_id = _optional_str(ambient.get("session_id") or ambient.get("session"))
        transport_session_id = _optional_str(ambient.get("transport_session_id"))
        encoded_recall_map = _encode_recall_map(recall_map)
        with self.transaction():
            if log_access and selected:
                # JSON keeps this one bounded-parameter batch even for large
                # result sets. Counts preserve repeated ids in caller order.
                rows = self._conn.execute(
                    """
                    WITH accesses AS (
                        SELECT value AS id, COUNT(*) AS increments
                        FROM json_each(?) GROUP BY value
                    )
                    UPDATE nodes
                    SET access_count = access_count + (
                            SELECT increments FROM accesses WHERE accesses.id = nodes.id
                        ),
                        last_accessed = ?, updated_at = ?
                    WHERE id IN (SELECT id FROM accesses)
                    RETURNING id, access_count, last_accessed, updated_at
                    """,
                    (_json_dumps([node.id for node in selected]), now, now),
                ).fetchall()
                access = {row["id"]: dict(row) for row in rows}
                for node in selected:
                    if node.id not in access:
                        raise KeyError(node.id)
                # Reconstruct each occurrence's metadata without mutating the
                # selected snapshots, including when a caller repeats an id.
                updated = []
                for node in reversed(selected):
                    metadata = access[node.id]
                    updated.append(replace(node, **metadata))
                    metadata["access_count"] -= 1
                selected = list(reversed(updated))
            if not log_event:
                return selected, None
            self._conn.execute(
                """
                INSERT INTO recall_events (
                    id, query, scope, requested_scope, resolved_scopes, ambient_context,
                    depth, max_results, results, agent, task, session_id,
                    transport_session_id, feedback_applied,
                    feedback_trace_id, feedback_applied_at, created_at, recall_map,
                    task_pattern
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, ?, ?, ?)
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
                    transport_session_id,
                    now,
                    encoded_recall_map,
                    task_pattern,
                ),
            )
            self._record_recall_history_event(
                event_id=event_id,
                occurred_at=now,
                transport_session_id=transport_session_id,
                scope=scope,
                task=task,
                results=normalized_results,
                recall_map=recall_map,
            )
            event = self.get_recall_event(event_id)
        return selected, event

    def attach_recall_map(
        self, event_id: str, recall_map: Mapping[str, Any] | None
    ) -> None:
        """Record the map served with an already-recorded recall event.

        One statement, no read-back: this runs on the live recall path, after
        the response has been shaped, and its result is never consumed. An
        unknown ``event_id`` raises ``KeyError`` rather than writing nothing
        silently — an unattributable delivery is the failure this column exists
        to prevent.
        """

        encoded = _encode_recall_map(recall_map)
        with self.transaction():
            updated = self._conn.execute(
                "UPDATE recall_events SET recall_map = ? WHERE id = ?",
                (encoded, event_id),
            ).rowcount
            if updated:
                self._replace_recall_history_deliveries(event_id, recall_map)
        if not updated:
            raise KeyError(event_id)

    def matured_recall_history(
        self,
        candidate_ids: Iterable[str],
        decision_at: str | datetime,
    ) -> dict[str, MaturedRecallHistory]:
        """Read exact matured M/C/K for a bounded candidate-id batch.

        A delivery is visible only when it predates ``decision_at`` and its
        frozen 24-hour ``outcome_end`` is no later than that instant.  Rows are
        newest first under the evaluator's ordering, so the leading false run
        is K.  Each candidate range stops at
        ``MAX_RECALL_HISTORY_DELIVERIES_PER_NODE + 1``: the extra row detects
        truncation, which is reported as unavailable rather than silently
        producing an inexact feature.

        This method issues SELECT statements only.  In particular it never
        repairs or backfills from selection; schema initialization and the two
        recall-event write paths own all ledger maintenance.
        """

        ids = list(dict.fromkeys(str(node_id) for node_id in candidate_ids))
        if not ids:
            return {}

        def unavailable(reason: str) -> dict[str, MaturedRecallHistory]:
            return {
                node_id: MaturedRecallHistory.unavailable(reason)
                for node_id in ids
            }

        if any(not node_id for node_id in ids):
            return unavailable("invalid_candidate_id")
        if len(ids) > MAX_RECALL_HISTORY_CANDIDATES:
            return unavailable("candidate_batch_truncated")
        instant = _normalize_recall_history_instant(decision_at)
        if instant is None:
            return unavailable("invalid_decision_instant")

        try:
            state = self._conn.execute(
                f"""
                SELECT format_version, complete, unavailable_reason
                FROM {RECALL_DELIVERY_HISTORY_STATE_TABLE}
                WHERE singleton = 1
                """
            ).fetchone()
            if state is None:
                return unavailable("history_ledger_unavailable")
            if int(state["format_version"]) != _RECALL_DELIVERY_HISTORY_FORMAT:
                return unavailable("history_format_unavailable")
            if not int(state["complete"]):
                return unavailable(
                    str(state["unavailable_reason"] or "history_ledger_unavailable")
                )

            histories: dict[str, MaturedRecallHistory] = {}
            for node_id in ids:
                rows = self._conn.execute(
                    f"""
                    SELECT transport_matched, transport_consumed,
                           fallback_consumed, lookup_consumed
                    FROM {RECALL_DELIVERY_HISTORY_TABLE}
                    WHERE node_id = ?
                      AND delivered_at < ?
                      AND outcome_end <= ?
                    ORDER BY
                        outcome_end DESC,
                        delivered_at DESC,
                        (CASE WHEN transport_matched = 1
                            THEN transport_consumed ELSE fallback_consumed END) DESC,
                        delivery_event_id DESC
                    LIMIT ?
                    """,
                    (
                        node_id,
                        instant,
                        instant,
                        MAX_RECALL_HISTORY_DELIVERIES_PER_NODE + 1,
                    ),
                ).fetchall()
                if len(rows) > MAX_RECALL_HISTORY_DELIVERIES_PER_NODE:
                    histories[node_id] = MaturedRecallHistory.unavailable(
                        "history_truncated"
                    )
                    continue
                outcomes = [
                    bool(
                        row["transport_consumed"]
                        if row["transport_matched"]
                        else row["fallback_consumed"]
                    )
                    for row in rows
                ]
                trailing = 0
                for consumed in outcomes:
                    if consumed:
                        break
                    trailing += 1
                # Read as a tri-state, in the same newest-first order.  A NULL
                # window contributes to neither count and does not end the
                # trailing run.
                lookups = [
                    None if row["lookup_consumed"] is None
                    else bool(row["lookup_consumed"])
                    for row in rows
                ]
                lookup_trailing = 0
                for followed in lookups:
                    if followed is None:
                        continue
                    if followed:
                        break
                    lookup_trailing += 1
                histories[node_id] = MaturedRecallHistory.known(
                    matured=len(outcomes),
                    consumed=sum(outcomes),
                    trailing_nonconsumed=trailing,
                    lookup_consumed=sum(1 for followed in lookups if followed),
                    lookup_known=sum(
                        1 for followed in lookups if followed is not None
                    ),
                    lookup_trailing_absent=lookup_trailing,
                )
            return histories
        except sqlite3.Error:
            return unavailable("history_read_failed")
        except (TypeError, ValueError):
            return unavailable("malformed_history_data")

    def record_lookup_event(
        self,
        node_ids: Iterable[str],
        *,
        transport_session_id: str | None = None,
        occurred_at: str | datetime | None = None,
    ) -> str | None:
        """Persist one ``memory_lookup`` id-fetch and correlate it forward.

        This is the only exogenous follow signal the server can observe: the
        three consumed bits on ``recall_delivery_history`` all mean "a later
        recall returned this node again", which the ranker produces on its own
        for any node it likes.  Fetching a specific ULID is something only a
        reader who was handed that ULID does.

        The write is about the *request*: every id the caller asked for is
        recorded, including ids that resolved to nothing, and nothing else is.
        Returns the new lookup event id, or ``None`` when no usable id was
        asked for.

        Deliberately not a node write.  ``access_count``, ``last_accessed``,
        ``recall_events`` and pending feedback are all untouched — a signal
        that fed the counters it is meant to audit would be the same echo it
        replaces.
        """

        requested: list[str] = []
        for candidate in node_ids:
            text = _optional_str(candidate)
            # Blank-after-strip is dropped as garbage; anything else is stored
            # exactly as asked for, so the event stays a record of the request.
            if text is None or not text.strip() or text in requested:
                continue
            requested.append(text)
        if not requested:
            return None
        instant = _normalize_recall_history_instant(
            _utc_now() if occurred_at is None else occurred_at
        )
        if instant is None:
            raise ValueError(f"unparsable lookup instant: {occurred_at!r}")
        transport = _optional_str(transport_session_id)
        event_id = new_ulid()
        with self.transaction():
            self._conn.executemany(
                f"""
                INSERT INTO {RECALL_LOOKUP_EVENT_TABLE} (
                    lookup_event_id, node_id, occurred_at, transport_session_id
                ) VALUES (?, ?, ?, ?)
                """,
                ((event_id, node_id, instant, transport) for node_id in requested),
            )
            self._apply_recall_lookup_consumer(
                occurred_at=instant, node_ids=requested
            )
        return event_id

    def _apply_recall_lookup_consumer(
        self, *, occurred_at: str, node_ids: Sequence[str]
    ) -> None:
        """Mark every open delivery window this lookup lands inside as followed.

        The mirror of :meth:`_apply_recall_history_consumer`, walking back over
        the same ``(delivered_at, outcome_end]`` half-open ranges, with one
        deliberate difference: no transport or scope/task predicate narrows the
        match.  The correlation is global because the two ends of it sit on
        different MCP connections by construction — the dashboard probe that
        offers a card and the agent that fetches it are separate transports —
        so a transport-scoped join would match nothing and report a permanent
        absence.  What makes the looser join honest is the rarity of the act:
        an id-fetch names one exact ULID, which a caller only has because
        something handed it over.

        A window still recorded as unobservable is promoted rather than
        skipped: evidence of a fetch outranks not knowing.
        """

        self._conn.executemany(
            f"""
            UPDATE {RECALL_DELIVERY_HISTORY_TABLE}
            SET lookup_consumed = 1
            WHERE node_id = ?
              AND delivered_at < ? AND outcome_end >= ?
            """,
            ((node_id, occurred_at, occurred_at) for node_id in node_ids),
        )

    def _first_recorded_lookup_at(self) -> str | None:
        """The instant this database first observed a lookup, or None.

        The observability epoch for the tri-state.  A delivery window that
        closed before it could not have produced a lookup event no matter what
        the reader did, so its outcome is unknown rather than absent.  Derived
        from the data instead of stored as a marker so a rebuild reaches the
        same verdict as the online path from the same rows.
        """

        row = self._conn.execute(
            f"SELECT MIN(occurred_at) AS first_at FROM {RECALL_LOOKUP_EVENT_TABLE}"
        ).fetchone()
        return None if row is None else _optional_str(row["first_at"])

    # ------------------------------------------------------------------
    # Recall credit ledger
    #
    # One claim per (event, node) pair, whichever usage signal comes first —
    # the consuming trace grounding the node or a same-session lookup of it.
    # See the note on _RECALL_CREDIT_LEDGER_SCHEMA_SQL.
    # ------------------------------------------------------------------

    def claim_recall_credit(
        self,
        recall_event_id: str,
        node_id: str,
        *,
        basis: str,
        source_id: str,
        credited_at: str | datetime | None = None,
    ) -> bool:
        """Take the (event, node) credit for ``basis``; True only when new.

        One ``INSERT OR IGNORE`` in a composable transaction. Consumers must
        include the claim and numeric updates in their enclosing transaction.
        A pair already credited — by an
        earlier grounding, lookup or explicit mark — leaves the rows as they
        were and returns False, and the caller must then apply nothing.
        ``basis`` is ``'grounded'`` (``source_id`` is the consuming trace id),
        ``'lookup'`` (the lookup event id) or ``'explicit'`` (the id of the
        audit row as ``mark:<id>``). The first two land in
        ``recall_credit_ledger``, ``explicit`` in ``recall_explicit_credit``;
        each insert carries a ``NOT EXISTS`` probe of the other table in the
        same statement, so the pair is claimed at most once across all three
        bases in either arrival order.
        """

        if basis not in RECALL_CREDIT_BASES:
            raise ValueError(f"unknown credit basis: {basis!r}")
        if credited_at is None:
            instant = _utc_now()
        else:
            instant = _normalize_recall_history_instant(credited_at)
            if instant is None:
                raise ValueError(f"unparsable credit instant: {credited_at!r}")
        event_id, target = str(recall_event_id), str(node_id)
        with self.transaction():
            if basis == "explicit":
                cursor = self._conn.execute(
                    f"""
                    INSERT OR IGNORE INTO {RECALL_EXPLICIT_CREDIT_TABLE} (
                        recall_event_id, node_id, source_id, credited_at
                    )
                    SELECT ?, ?, ?, ?
                    WHERE NOT EXISTS (
                        SELECT 1 FROM {RECALL_CREDIT_LEDGER_TABLE}
                        WHERE recall_event_id = ? AND node_id = ?
                    )
                    """,
                    (event_id, target, str(source_id), instant, event_id, target),
                )
            else:
                cursor = self._conn.execute(
                    f"""
                    INSERT OR IGNORE INTO {RECALL_CREDIT_LEDGER_TABLE} (
                        recall_event_id, node_id, basis, source_id, credited_at
                    )
                    SELECT ?, ?, ?, ?, ?
                    WHERE NOT EXISTS (
                        SELECT 1 FROM {RECALL_EXPLICIT_CREDIT_TABLE}
                        WHERE recall_event_id = ? AND node_id = ?
                    )
                    """,
                    (event_id, target, basis, str(source_id), instant, event_id, target),
                )
        return cursor.rowcount == 1

    def credited_node_ids(self, recall_event_id: str) -> set[str]:
        """Node ids whose credit for one recall event has already been spent,
        under any basis (ledger and explicit side table)."""

        rows = self._conn.execute(
            f"""
            SELECT node_id FROM {RECALL_CREDIT_LEDGER_TABLE}
            WHERE recall_event_id = ?
            UNION
            SELECT node_id FROM {RECALL_EXPLICIT_CREDIT_TABLE}
            WHERE recall_event_id = ?
            """,
            (str(recall_event_id), str(recall_event_id)),
        ).fetchall()
        return {str(row["node_id"]) for row in rows}

    # ------------------------------------------------------------------
    # Explicit feedback marks (audit). See _RECALL_FEEDBACK_MARKS_SCHEMA_SQL.
    # ------------------------------------------------------------------

    def record_feedback_marks(self, marks: Sequence[Mapping[str, Any]]) -> list[int]:
        """Append audit rows, one per mark, in one transaction; returns row ids.

        Each mapping carries the table's columns except ``id``; ``marked_at``
        defaults to the storage clock and ``recall_event_id``/``source_id`` to
        ``''``.
        """

        ids: list[int] = []
        if not marks:
            return ids
        now = _utc_now()
        with self.transaction():
            for mark in marks:
                rank = mark.get("rank")
                cursor = self._conn.execute(
                    f"""
                    INSERT INTO {RECALL_FEEDBACK_MARKS_TABLE} (
                        recall_event_id, node_id, mark, accepted, reject_reason,
                        via_tool, source_id, transport_session_id, agent, rank,
                        marked_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(mark.get("recall_event_id") or ""),
                        str(mark.get("node_id") or ""),
                        str(mark["mark"]),
                        1 if mark.get("accepted") else 0,
                        _optional_str(mark.get("reject_reason")),
                        str(mark.get("via_tool") or ""),
                        str(mark.get("source_id") or ""),
                        _optional_str(mark.get("transport_session_id")),
                        _optional_str(mark.get("agent")),
                        None if rank is None else int(rank),
                        str(mark.get("marked_at") or now),
                    ),
                )
                ids.append(int(cursor.lastrowid))
        return ids

    def set_feedback_marks_source(self, mark_ids: Iterable[int], source_id: str) -> None:
        """Fill ``source_id`` on marks recorded before their carrier existed.

        memory_recall and memory_teach record marks before the recall event /
        corrective trace that carried them is written (a mark must be
        resolved against *earlier* deliveries and must precede implicit
        feedback), then stamp the id here.
        """

        ids = [int(mark_id) for mark_id in mark_ids]
        if not ids or not source_id:
            return
        with self.transaction():
            self._conn.executemany(
                f"UPDATE {RECALL_FEEDBACK_MARKS_TABLE} SET source_id = ? WHERE id = ?",
                [(str(source_id), mark_id) for mark_id in ids],
            )

    def irrelevant_marked_node_ids(self, recall_event_id: str) -> set[str]:
        """Nodes an accepted ``irrelevant`` mark names for one recall event."""

        rows = self._conn.execute(
            f"""
            SELECT DISTINCT node_id FROM {RECALL_FEEDBACK_MARKS_TABLE}
            WHERE recall_event_id = ? AND mark = 'irrelevant' AND accepted = 1
            """,
            (str(recall_event_id),),
        ).fetchall()
        return {str(row["node_id"]) for row in rows}

    def recall_events_for_transport(
        self,
        transport_session_id: str | None,
        *,
        limit: int = 200,
    ) -> list[RecallEvent]:
        """The most recent recall events stamped with one transport session.

        Newest first, served by ``idx_recall_events_transport_created``, and
        bounded like :meth:`recall_delivery_history`: on a long-lived session only
        the ``limit`` most recent events are candidates for the lookup-credit
        join, and a ``None`` or empty transport id identifies no session and
        returns nothing without touching the database. Closed events are
        included on purpose — a lookup after the closure of an event whose
        trace grounded nothing is still evidence that a delivered node was
        used; the credit ledger, not ``feedback_applied``, decides whether
        that evidence has already been spent.
        """

        if not transport_session_id or limit <= 0:
            return []
        rows = self._conn.execute(
            """
            SELECT *
            FROM recall_events
            WHERE transport_session_id = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (str(transport_session_id), int(limit)),
        ).fetchall()
        return [_recall_event_from_row(row) for row in rows]

    def recall_deliveries(
        self,
        transport_session_id: str | None,
        node_ids: Iterable[str],
        *,
        earliest: datetime,
        latest: datetime,
        limit: int = 200,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Newest delivery facts for requested ids in a bounded session window.

        The event cap precedes all id/time filtering, including closed events.
        Candidates are selected before their membership is read, in stored
        created_at/rowid order. Time eligibility uses the same parsed UTC
        seconds as feedback. SQL projects matching objects from each eligible
        result array, in result position order, until every requested id has
        a delivery. Unneeded bodies are never parsed; malformed necessary JSON
        and storage errors propagate. Non-array bodies and non-object results
        cannot prove delivery. No unrelated bodies or ambient context are
        transferred. The first eligible occurrence of each id wins.
        No delivery-history projection or its availability state is consulted.
        Transport always identifies its own deliveries. Explicit callers may
        also supply a durable session: reconnects require equal session, no
        both-present task/agent conflicts, and either the same recall scope
        or a saved broad read. A broad read's preferred scope does not restrict
        the destination of new knowledge. Lookup omits context and retains
        its transport-only selection.
        """
        requested = set(node_ids)
        if not transport_session_id or not requested or limit <= 0:
            return {}
        params = {
            **_recall_identity(context),
            "transport_session_id": str(transport_session_id),
            "limit": int(limit),
        }
        predicates = ["transport_session_id = :transport_session_id"]
        if params["session_id"] and params["session_id"].strip():
            params["scope"] = resolve_scope(ambient_context=context, store=self).requested_scope
            params["all_scopes"] = ALL_SCOPES
            # Breadth relaxes only scope equality, never caller identity or
            # delivered-id membership. Named resolved scopes are preferences
            # or read fallbacks, not permission to join a different scope.
            scope_match = f"""({_recall_scope_match_sql(params, allow_transport=False)}
                OR EXISTS (SELECT 1 FROM json_each(resolved_scopes)
                           WHERE value = :all_scopes))"""
            predicates.append(" AND ".join([
                "session_id = :session_id",
                "COALESCE(transport_session_id, '') != :transport_session_id",
                scope_match,
                *(_compatible_recall_identity_sql(field) for field in ("task", "agent")
                  if params[field]),
            ]))
        # Separate indexed probes avoid an OR scan. They are disjoint, capped
        # independently, then share one newest-event cap before result/time tests.
        probes = " UNION ALL ".join(
            f"""SELECT * FROM (
                SELECT rowid AS event_rowid, id, created_at FROM recall_events
                WHERE {predicate}
                ORDER BY created_at DESC, rowid DESC LIMIT :limit
            )"""
            for predicate in predicates
        )
        # Candidate identity and membership must come from the same read snapshot.
        with self.transaction():
            events = self._conn.execute(
                f"""
                WITH recent AS MATERIALIZED (
                    SELECT * FROM ({probes})
                    ORDER BY created_at DESC, event_rowid DESC LIMIT :limit
                )
                SELECT * FROM recent ORDER BY created_at DESC, event_rowid DESC
                """,
                params,
            ).fetchall()
            hits: dict[str, dict[str, Any]] = {}
            for event in events:
                instant = parse_timestamp(event["created_at"])
                if instant is None or not earliest <= instant.replace(microsecond=0) <= latest:
                    continue
                # Keep the ordered event choice outside the JSON query: sorting a
                # session-wide join can evaluate older bodies before its first hit.
                rows = self._conn.execute(
                    f"""
                    SELECT event.id AS event_id, event.query, event.scope, event.created_at,
                           json_extract(result.value, '$.node_id') AS node_id,
                           result.key AS rank, result.value AS result,
                           EXISTS(SELECT 1 FROM {RECALL_CREDIT_LEDGER_TABLE} AS credit
                                  WHERE credit.recall_event_id = event.id
                                    AND credit.node_id = json_extract(result.value, '$.node_id'))
                           OR EXISTS(SELECT 1 FROM {RECALL_EXPLICIT_CREDIT_TABLE} AS credit
                                     WHERE credit.recall_event_id = event.id
                                       AND credit.node_id = json_extract(result.value, '$.node_id'))
                           AS credited
                    FROM recall_events AS event
                    JOIN json_each(CASE json_type(event.results)
                        WHEN 'array' THEN event.results ELSE '[]' END) AS result
                    WHERE event.rowid = :event_rowid
                      AND CASE WHEN result.type = 'object'
                          THEN json_type(result.value, '$.node_id') END = 'text'
                      AND CASE WHEN result.type = 'object'
                          THEN json_extract(result.value, '$.node_id') END
                          IN (SELECT value FROM json_each(:requested))
                    ORDER BY result.key
                    """,
                    {"event_rowid": event["event_rowid"],
                     "requested": _json_dumps(sorted(requested - hits.keys()))},
                )
                try:
                    for row in rows:
                        result = _json_loads(row["result"], {})
                        node_id = str(result.get("node_id") or "")
                        if node_id in requested and node_id not in hits:
                            hits[node_id] = {**dict(row), "result": result}
                finally:
                    rows.close()
                if len(hits) == len(requested):
                    break
        return hits

    def _record_recall_history_event(
        self,
        *,
        event_id: str,
        occurred_at: str,
        transport_session_id: str | None,
        scope: str,
        task: str | None,
        results: Sequence[Mapping[str, Any]],
        recall_map: Mapping[str, Any] | None,
    ) -> None:
        """Normalize one new event, its result ids, and its deliveries."""

        normalized_at = _normalize_recall_history_instant(occurred_at)
        if normalized_at is None:  # pragma: no cover - _utc_now is canonical
            self._mark_recall_history_unavailable("malformed_event_timestamp")
            return
        result_ids = _recall_result_node_ids(results)
        self._conn.execute(
            f"""
            INSERT INTO {RECALL_HISTORY_EVENT_TABLE} (
                recall_event_id, occurred_at, transport_session_id, scope, task
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(recall_event_id) DO UPDATE SET
                occurred_at = excluded.occurred_at,
                transport_session_id = excluded.transport_session_id,
                scope = excluded.scope,
                task = excluded.task
            """,
            (event_id, normalized_at, transport_session_id, scope, task),
        )
        self._conn.execute(
            f"DELETE FROM {RECALL_HISTORY_RESULT_TABLE} WHERE recall_event_id = ?",
            (event_id,),
        )
        self._conn.executemany(
            f"""
            INSERT INTO {RECALL_HISTORY_RESULT_TABLE} (recall_event_id, node_id)
            VALUES (?, ?)
            """,
            ((event_id, node_id) for node_id in dict.fromkeys(result_ids)),
        )
        self._replace_recall_history_deliveries(
            event_id,
            recall_map,
            result_ids=result_ids,
        )
        self._apply_recall_history_consumer(
            occurred_at=normalized_at,
            transport_session_id=transport_session_id,
            scope=scope,
            task=task,
            result_ids=result_ids,
        )

    def _replace_recall_history_deliveries(
        self,
        event_id: str,
        recall_map: Mapping[str, Any] | None,
        *,
        result_ids: Sequence[str] | None = None,
    ) -> None:
        """Replace one event's organic/map union and resolve later outcomes."""

        event = self._conn.execute(
            f"""
            SELECT occurred_at, transport_session_id, scope, task
            FROM {RECALL_HISTORY_EVENT_TABLE}
            WHERE recall_event_id = ?
            """,
            (event_id,),
        ).fetchone()
        if event is None:
            self._mark_recall_history_unavailable("history_event_missing")
            return
        if result_ids is None:
            raw = self._conn.execute(
                "SELECT results FROM recall_events WHERE id = ?", (event_id,)
            ).fetchone()
            parsed = _decode_recall_results(None if raw is None else raw["results"])
            if parsed is None:
                self._mark_recall_history_unavailable("malformed_legacy_results")
                return
            result_ids = _recall_result_node_ids(parsed)

        delivered_ids = _organic_tail_node_ids(result_ids)
        delivered_ids.extend(_recall_map_medoid_ids(recall_map))
        delivered_ids = list(dict.fromkeys(delivered_ids))
        delivered_at = str(event["occurred_at"])
        outcome_end = _shift_recall_history_hours(
            delivered_at, RECALL_DELIVERY_HISTORY_HORIZON_HOURS
        )
        if outcome_end is None:  # pragma: no cover - normalized table invariant
            self._mark_recall_history_unavailable("malformed_event_timestamp")
            return
        self._conn.execute(
            f"DELETE FROM {RECALL_DELIVERY_HISTORY_TABLE} WHERE delivery_event_id = ?",
            (event_id,),
        )
        self._conn.executemany(
            f"""
            INSERT INTO {RECALL_DELIVERY_HISTORY_TABLE} (
                delivery_event_id, node_id, delivered_at, outcome_end,
                transport_session_id, scope, task
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    event_id,
                    node_id,
                    delivered_at,
                    outcome_end,
                    event["transport_session_id"],
                    event["scope"],
                    event["task"],
                )
                for node_id in delivered_ids
            ),
        )
        if delivered_ids:
            self._refresh_recall_history_delivery_outcomes(
                event_id, delivered_ids, delivered_at, outcome_end
            )

    def _refresh_recall_history_delivery_outcomes(
        self,
        event_id: str,
        node_ids: Sequence[str],
        delivered_at: str,
        outcome_end: str,
    ) -> None:
        """Resolve consumers already present when a map is attached late."""

        event = self._conn.execute(
            f"""
            SELECT transport_session_id, scope, task
            FROM {RECALL_HISTORY_EVENT_TABLE}
            WHERE recall_event_id = ?
            """,
            (event_id,),
        ).fetchone()
        if event is None:  # pragma: no cover - caller just fetched this row
            return
        marks = ",".join("?" for _ in node_ids)
        transport_matched = False
        transport_consumed: set[str] = set()
        transport = _optional_str(event["transport_session_id"])
        if transport:
            transport_matched = self._conn.execute(
                f"""
                SELECT 1 FROM {RECALL_HISTORY_EVENT_TABLE}
                WHERE transport_session_id = ?
                  AND occurred_at > ? AND occurred_at <= ?
                LIMIT 1
                """,
                (transport, delivered_at, outcome_end),
            ).fetchone() is not None
            if transport_matched:
                transport_consumed = {
                    str(row["node_id"])
                    for row in self._conn.execute(
                        f"""
                        SELECT DISTINCT result.node_id
                        FROM {RECALL_HISTORY_EVENT_TABLE} AS consumer
                        JOIN {RECALL_HISTORY_RESULT_TABLE} AS result
                          ON result.recall_event_id = consumer.recall_event_id
                        WHERE consumer.transport_session_id = ?
                          AND consumer.occurred_at > ?
                          AND consumer.occurred_at <= ?
                          AND result.node_id IN ({marks})
                        """,
                        (transport, delivered_at, outcome_end, *node_ids),
                    )
                }

        fallback_consumed: set[str] = set()
        task = _optional_str(event["task"])
        if event["scope"] and task:
            fallback_consumed = {
                str(row["node_id"])
                for row in self._conn.execute(
                    f"""
                    SELECT DISTINCT result.node_id
                    FROM {RECALL_HISTORY_EVENT_TABLE} AS consumer
                    JOIN {RECALL_HISTORY_RESULT_TABLE} AS result
                      ON result.recall_event_id = consumer.recall_event_id
                    WHERE consumer.scope = ? AND consumer.task = ?
                      AND consumer.occurred_at > ?
                      AND consumer.occurred_at <= ?
                      AND result.node_id IN ({marks})
                    """,
                    (event["scope"], task, delivered_at, outcome_end, *node_ids),
                )
            }
        # ``None`` means this window is outside what the lookup recorder ever
        # saw, and stays NULL.  On a rebuild that is every window older than
        # the first recorded lookup; on a fresh install with no lookups yet it
        # is all of them, which is the point — day one must not read as a
        # corpus nobody follows.  Windows still open when the first lookup
        # lands are promoted by _apply_recall_lookup_consumer instead.
        #
        # One asymmetry is deliberate and worth naming: a window opened before
        # any lookup existed and never followed stays NULL online, while a
        # later rebuild resolves it to 0 once the epoch predates its
        # outcome_end.  Both readings are safe (the rebuild is the better
        # informed of the two) and both err away from inventing an absence.
        lookup_consumed: set[str] | None = None
        first_lookup_at = self._first_recorded_lookup_at()
        if first_lookup_at is not None and outcome_end >= first_lookup_at:
            lookup_consumed = {
                str(row["node_id"])
                for row in self._conn.execute(
                    f"""
                    SELECT DISTINCT node_id
                    FROM {RECALL_LOOKUP_EVENT_TABLE}
                    WHERE node_id IN ({marks})
                      AND occurred_at > ? AND occurred_at <= ?
                    """,
                    (*node_ids, delivered_at, outcome_end),
                )
            }
        self._conn.executemany(
            f"""
            UPDATE {RECALL_DELIVERY_HISTORY_TABLE}
            SET transport_matched = ?, transport_consumed = ?,
                fallback_consumed = ?, lookup_consumed = ?
            WHERE delivery_event_id = ? AND node_id = ?
            """,
            (
                (
                    int(transport_matched),
                    int(node_id in transport_consumed),
                    int(node_id in fallback_consumed),
                    None if lookup_consumed is None
                    else int(node_id in lookup_consumed),
                    event_id,
                    node_id,
                )
                for node_id in node_ids
            ),
        )

    def _apply_recall_history_consumer(
        self,
        *,
        occurred_at: str,
        transport_session_id: str | None,
        scope: str,
        task: str | None,
        result_ids: Sequence[str],
    ) -> None:
        """Apply one new event to compatible deliveries in its prior 24h."""

        unique_results = tuple(dict.fromkeys(result_ids))
        if transport_session_id:
            self._conn.execute(
                f"""
                UPDATE {RECALL_DELIVERY_HISTORY_TABLE}
                SET transport_matched = 1
                WHERE transport_session_id = ?
                  AND delivered_at < ? AND outcome_end >= ?
                """,
                (transport_session_id, occurred_at, occurred_at),
            )
            self._conn.executemany(
                f"""
                UPDATE {RECALL_DELIVERY_HISTORY_TABLE}
                SET transport_consumed = 1
                WHERE transport_session_id = ?
                  AND delivered_at < ? AND outcome_end >= ?
                  AND node_id = ?
                """,
                (
                    (transport_session_id, occurred_at, occurred_at, node_id)
                    for node_id in unique_results
                ),
            )
        if scope and task:
            self._conn.executemany(
                f"""
                UPDATE {RECALL_DELIVERY_HISTORY_TABLE}
                SET fallback_consumed = 1
                WHERE scope = ? AND task = ?
                  AND delivered_at < ? AND outcome_end >= ?
                  AND node_id = ?
                """,
                (
                    (scope, task, occurred_at, occurred_at, node_id)
                    for node_id in unique_results
                ),
            )

    def _mark_recall_history_unavailable(self, reason: str) -> None:
        self._conn.execute(
            f"""
            INSERT INTO {RECALL_DELIVERY_HISTORY_STATE_TABLE} (
                singleton, format_version, complete, unavailable_reason, updated_at
            ) VALUES (1, ?, 0, ?, ?)
            ON CONFLICT(singleton) DO UPDATE SET
                format_version = excluded.format_version,
                complete = 0,
                unavailable_reason = excluded.unavailable_reason,
                updated_at = excluded.updated_at
            """,
            (_RECALL_DELIVERY_HISTORY_FORMAT, reason, _utc_now()),
        )

    def recent_recall_map_history(
        self,
        scope: str | None = None,
        task: str | None = None,
        transport_session_id: str | None = None,
        limit: int = 20,
        task_pattern: str | None = None,
        *,
        work: RecallWork | None = None,
    ) -> list[dict[str, Any]]:
        """Map-bearing recall events, newest first, in a bounded result window.

        ``work`` selects the exact consumer used by cache and curtailment:
        scope plus effective task_pattern/task, or transport when both are
        absent. The optional individual filters remain available to diagnostic
        readers; ``task_pattern`` there includes patternless rows matching the
        supplied task. Passing ``work`` replaces that diagnostic disjunction.

        Scope matches the persisted resolved scope. Transport and pattern have
        existing indexes; task is a residual filter. Rows carry id, timestamp,
        scope, task, query, transport and the decoded delivered map.
        """

        if limit <= 0:
            return []
        clauses = ["recall_map IS NOT NULL"]
        params: list[Any] = []
        if transport_session_id is not None:
            clauses.append("transport_session_id = ?")
            params.append(transport_session_id)
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        if work is not None:
            predicate, values = work.predicate()
            clauses.append(predicate)
            params.extend(values)
        elif task_pattern is not None:
            clauses.append("(task_pattern = ? OR (task_pattern IS NULL AND task IS ?))")
            params.extend([task_pattern, task])
        elif task is not None:
            clauses.append("task = ?")
            params.append(task)
        params.append(int(limit))
        rows = self._conn.execute(
            f"""
            SELECT id, created_at, scope, task, query, transport_session_id, recall_map
            FROM recall_events
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "scope": row["scope"],
                "task": row["task"],
                "query": row["query"],
                "transport_session_id": row["transport_session_id"],
                "recall_map": _json_loads(row["recall_map"], None),
            }
            for row in rows
        ]

    def recall_work_followups(
        self, work: RecallWork, *, since: str, medoid_ids: Sequence[str],
        limit: int = 200,
    ) -> list[dict[str, Any]] | None:
        """Bounded recall and lookup evidence for one current-work consumer.

        Recall events include mapless results. A lookup has only a transport;
        its latest preceding recall identifies the work, including continuation
        on another connection. Conflicting work at the same timestamp cannot
        attribute a lookup. Corpus-wide delivery outcomes remain independent.
        ``None`` means the observation exceeded its bound, not non-use.
        """

        predicate, values = work.predicate()
        recalls = self._conn.execute(
            f"""SELECT id, created_at, query, results FROM recall_events
                WHERE {predicate} AND created_at >= ?
                ORDER BY created_at DESC, rowid DESC LIMIT ?""",
            (*values, since, limit + 1),
        ).fetchall()
        if len(recalls) > limit:
            return None
        evidence = [
            {"at": _normalize_recall_history_instant(row["created_at"]),
             "query": row["query"],
             "node_ids": _recall_result_node_ids(_json_loads(row["results"], [])),
             "event_id": row["id"]}
            for row in recalls
        ]
        if not medoid_ids:
            return evidence
        marks = ",".join("?" for _ in medoid_ids)
        consumer_predicate, consumer_values = work.predicate("consumer")
        tied_predicate, tied_values = work.predicate("tied")
        lookups = self._conn.execute(
            f"""SELECT lookup.node_id, lookup.occurred_at
                FROM {RECALL_LOOKUP_EVENT_TABLE} AS lookup
                JOIN recall_events AS consumer ON consumer.rowid = (
                    SELECT e.rowid FROM recall_events AS e
                    JOIN {RECALL_HISTORY_EVENT_TABLE} AS h
                      ON h.recall_event_id = e.id
                    WHERE e.transport_session_id = lookup.transport_session_id
                      AND h.occurred_at <= lookup.occurred_at
                    ORDER BY e.created_at DESC, e.rowid DESC LIMIT 1
                )
                WHERE lookup.node_id IN ({marks}) AND lookup.occurred_at > ?
                  AND ({consumer_predicate})
                  AND NOT EXISTS (
                      SELECT 1 FROM recall_events AS tied
                      WHERE tied.transport_session_id = consumer.transport_session_id
                        AND tied.created_at = consumer.created_at
                        AND NOT COALESCE(({tied_predicate}), 0)
                  )
                ORDER BY lookup.occurred_at DESC LIMIT ?""",
            (*medoid_ids, _normalize_recall_history_instant(since),
             *consumer_values, *tied_values, limit + 1),
        ).fetchall()
        if len(lookups) > limit:
            return None
        evidence.extend(
            {"at": row["occurred_at"], "query": "", "node_ids": [row["node_id"]],
             "event_id": None}
            for row in lookups
        )
        return evidence

    def get_recall_event_header(self, event_id: str) -> RecallEventHeader | None:
        """Read current identity/query/scope without materializing provenance."""
        row = self._conn.execute(
            "SELECT id, query, scope FROM recall_events WHERE id = ?",
            (event_id,),
        ).fetchone()
        return RecallEventHeader(**row) if row else None

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

    def recall_delivery_history(
        self,
        transport_session_id: str | None = None,
        *,
        max_events: int = 200,
    ) -> list[dict[str, Any]]:
        """Read shaped-content evidence from a bounded session event window.

        Legacy results remain readable but carry no evidence of delivered
        content. The caller compares recorded bodies with the current candidate,
        not node ids.
        """
        with self.stream_recall_delivery_history(
            transport_session_id, max_events=max_events,
        ) as results:
            return list(results)

    @contextmanager
    def stream_recall_delivery_history(
        self,
        transport_session_id: str | None = None,
        *,
        max_events: int = 200,
    ) -> Iterator[Iterator[dict[str, Any]]]:
        """Yield delivery rows one event at a time within the cursor's lifetime.

        Consume inside the context so exhaustion, failure and early termination
        all close the cursor. Selected malformed events propagate to the caller.
        """
        if not transport_session_id or max_events <= 0:
            yield iter(())
            return
        cursor = self._conn.execute(
            """
            SELECT results FROM recall_events WHERE transport_session_id = ?
            ORDER BY created_at DESC, rowid DESC LIMIT ?
            """,
            (transport_session_id, int(max_events)),
        )
        try:
            yield (
                item for row in cursor for item in _json_loads(row["results"], [])
                if isinstance(item, dict) and isinstance(item.get("delivery"), dict)
            )
        finally:
            cursor.close()

    def record_recall_delivery(
        self, event_id: str, deliveries: Mapping[str, Mapping[str, str]],
    ) -> None:
        """Attach the shaper's final bodies without replacing selection history.

        This atomic optional write occurs after response construction. A missing
        receipt permits re-delivery; it must not make memory unavailable. It
        records a formed response, not acknowledgment by the network recipient.
        """
        with self.transaction():
            row = self._conn.execute(
                "SELECT results FROM recall_events WHERE id = ?", (event_id,)
            ).fetchone()
            if row is None:
                raise KeyError(event_id)
            results = _json_loads(row["results"], [])
            for result in results:
                result["delivery"] = dict(deliveries.get(
                    str(result.get("node_id", "")), {"kind": "omitted"},
                ))
            self._conn.execute(
                "UPDATE recall_events SET results = ? WHERE id = ?",
                (_json_dumps(results), event_id),
            )

    def pending_recall_events(
        self,
        *,
        scope: str,
        context: Mapping[str, Any] | None = None,
        limit: int = 1,
    ) -> list[RecallEvent]:
        """Return bounded pending recalls with compatible work or transport identity."""

        params = {
            **_recall_identity(context),
            "scope": scope,
            "limit": max(1, int(limit)),
        }
        matches = _pending_recall_match_sql(params)
        if not matches:
            return []
        # Each positive identity has its own indexed, capped probe. Combining
        # them as an OR lets SQLite choose missing-identity ranges instead.
        probes = " UNION ALL ".join(
            f"""SELECT * FROM (
                SELECT rowid AS event_rowid, created_at FROM recall_events INDEXED BY {index}
                WHERE feedback_applied = 0 AND {predicate}
                ORDER BY created_at DESC, rowid DESC LIMIT :limit
            )"""
            for index, predicate in matches
        )
        rows = self._conn.execute(
            f"""
            WITH selected AS (
                SELECT * FROM ({probes})
                ORDER BY created_at DESC, event_rowid DESC LIMIT :limit
            )
            SELECT event.* FROM selected
            JOIN recall_events AS event ON event.rowid = selected.event_rowid
            ORDER BY selected.created_at DESC, selected.event_rowid DESC
            """,
            params,
        ).fetchall()
        return [_recall_event_from_row(row) for row in rows]

    def mark_recall_event_feedback(self, event_id: str, trace_id: str) -> RecallEvent:
        if self.get_node(trace_id) is None:
            raise KeyError(trace_id)
        with self.transaction():
            updated = self._conn.execute(
                """
                UPDATE recall_events
                SET feedback_applied = 1,
                    feedback_trace_id = ?,
                    feedback_applied_at = ?
                WHERE id = ?
                """,
                (trace_id, _utc_now(), event_id),
            ).rowcount
            if not updated:
                raise KeyError(event_id)
        return self.get_recall_event(event_id)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Grounded-usage attestation ledger (schema v8)
    #
    # Claim and completion compose with the attestation's required transaction.
    # The unique (event, evidence) key guards credit, while completion records
    # its durable verdict. Existing rows always replay conservatively: an
    # incomplete claim does not prove the absence of earlier effects.
    # ------------------------------------------------------------------

    def find_recall_attestation(
        self, recall_event_id: str, evidence_sha256: str
    ) -> RecallAttestation | None:
        """The recorded verdict for one (event, evidence) pair, if any."""

        row = self._conn.execute(
            """
            SELECT * FROM recall_attestations
            WHERE recall_event_id = ? AND evidence_sha256 = ?
            """,
            (recall_event_id, evidence_sha256),
        ).fetchone()
        return _recall_attestation_from_row(row) if row else None

    def get_recall_attestation(self, attestation_id: str) -> RecallAttestation | None:
        row = self._conn.execute(
            "SELECT * FROM recall_attestations WHERE id = ?",
            (attestation_id,),
        ).fetchone()
        return _recall_attestation_from_row(row) if row else None

    def list_recall_attestations(
        self,
        *,
        recall_event_id: str | None = None,
        source_session_key: str | None = None,
        limit: int = 100,
    ) -> list[RecallAttestation]:
        """Recorded attestations, newest first. Audit read, never a gate."""

        clauses: list[str] = []
        params: list[Any] = []
        if recall_event_id is not None:
            clauses.append("recall_event_id = ?")
            params.append(recall_event_id)
        if source_session_key is not None:
            clauses.append("source_session_key = ?")
            params.append(source_session_key)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        rows = self._conn.execute(
            f"""
            SELECT * FROM recall_attestations {where}
            ORDER BY created_at DESC, rowid DESC LIMIT ?
            """,
            params,
        ).fetchall()
        return [_recall_attestation_from_row(row) for row in rows]

    def claim_recall_attestation(
        self,
        *,
        recall_event_id: str,
        evidence_sha256: str,
        evidence_items: int,
        evidence_chars: int,
        min_containment: float,
        containments: Iterable[Mapping[str, Any]] | None = None,
        grounded_node_ids: Iterable[str] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> RecallAttestation | None:
        """Take the (event, evidence) key, recording the computed verdict.

        Returns ``None`` when the key is already held — the caller must then
        replay the recorded verdict rather than apply anything. Nothing about
        the attesting client is a gate: the context fields are stamped for
        audit exactly as ``record_recall_event`` stamps a recall's ambient
        context.
        """

        ambient = dict(context or {})
        attestation_id = new_ulid()
        now = _utc_now()
        try:
            with self.transaction():
                self._conn.execute(
                    """
                    INSERT INTO recall_attestations (
                        id, recall_event_id, evidence_sha256, evidence_items,
                        evidence_chars, min_containment, containments,
                        grounded_node_ids, anchor_ids, credited, closed,
                        closed_by_attestation, feedback_trace_id, agent, task,
                        session_id, transport_session_id, source_session_key,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, '[]', 0, 0, 0, NULL, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attestation_id,
                        recall_event_id,
                        evidence_sha256,
                        int(evidence_items),
                        int(evidence_chars),
                        float(min_containment),
                        _json_dumps([dict(item) for item in containments or []]),
                        _json_dumps([str(node_id) for node_id in grounded_node_ids or []]),
                        _optional_str(ambient.get("agent")),
                        _optional_str(ambient.get("task")),
                        _optional_str(ambient.get("session_id") or ambient.get("session")),
                        _optional_str(ambient.get("transport_session_id")),
                        _optional_str(ambient.get("source_session_key")),
                        now,
                    ),
                )
        except sqlite3.IntegrityError:
            return None
        return self.get_recall_attestation(attestation_id)

    def complete_recall_attestation(
        self,
        attestation_id: str,
        *,
        credited: bool,
        closed: bool,
        closed_by_attestation: bool,
        anchor_ids: Iterable[str] | None = None,
        feedback_trace_id: str | None = None,
    ) -> RecallAttestation:
        """Record what a claimed attestation actually changed."""

        if self.get_recall_attestation(attestation_id) is None:
            raise KeyError(attestation_id)
        with self.transaction():
            self._conn.execute(
                """
                UPDATE recall_attestations
                SET credited = ?, closed = ?, closed_by_attestation = ?,
                    anchor_ids = ?, feedback_trace_id = ?
                WHERE id = ?
                """,
                (
                    1 if credited else 0,
                    1 if closed else 0,
                    1 if closed_by_attestation else 0,
                    _json_dumps([str(anchor_id) for anchor_id in anchor_ids or []]),
                    feedback_trace_id,
                    attestation_id,
                ),
            )
        return self.get_recall_attestation(attestation_id)  # type: ignore[return-value]

    def search_content(
        self,
        query: str,
        *,
        level: NodeLevel | None = None,
        scope: str | None = None,
        limit: int = 10,
    ) -> list[tuple[Node, float]]:
        """Search content and recorded task through the same lexical index."""

        rows = self._search_content_rows(query, projection="n.*", level=level, scope=scope, limit=limit)
        return [(_node_from_row(row), float(row["score"])) for row in rows]

    def search_content_ids(
        self,
        query: str,
        *,
        level: NodeLevel | None = None,
        scope: str | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Discover lexical identities and ranks without materializing bodies."""

        rows = self._search_content_rows(query, projection="n.id", level=level, scope=scope, limit=limit)
        return [(str(row["id"]), float(row["score"])) for row in rows]

    def _search_content_rows(
        self,
        query: str,
        *,
        projection: Literal["n.*", "n.id"],
        level: NodeLevel | None,
        scope: str | None,
        limit: int,
    ) -> list[sqlite3.Row]:
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
        return self._conn.execute(
            f"""
            SELECT {projection}, bm25(nodes_fts) AS score
            FROM nodes_fts
            JOIN nodes n ON n.rowid = nodes_fts.rowid
            WHERE {' AND '.join(clauses)}
            ORDER BY score ASC, n.confidence DESC, n.usefulness_score DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

    def term_document_frequencies(self, terms: Sequence[str]) -> dict[str, int]:
        """Document frequency per term, read from the ``nodes_fts_vocab`` index.

        Returns ``{term: number of indexed documents whose content contains
        the term}``, keyed by the terms exactly as given (duplicate inputs
        collapse into one key). A term the index has never seen maps to 0.

        Normalization contract: every input term is folded by the *same*
        ``unicode61`` tokenizer that indexes ``nodes_fts`` — Unicode case
        folding plus diacritic removal on Latin script characters — via the
        scratch TEMP tables in ``_FTS_TERM_TOKENIZER_SQL``, so any casing or
        accenting of a word matches the index by construction ("Café", "CAFE"
        and "cafe" all count the same documents) and callers extracting
        candidate terms from raw text get identical folding without
        reimplementing it. An input that folds to anything other than exactly
        one token — an empty or punctuation-only string, a multi-word phrase,
        a separator-joined compound like ``recall_map`` — maps to 0; split
        such inputs into single words before calling.

        Counts use only the content column, so recorded task terms cannot
        change content-derived labels. They include soft-deleted
        (``decayed = 1``) nodes: soft deletion touches no FTS-synced column,
        so the row stays indexed. Cost per call: one TEMP-table round trip for
        the fold plus one term-seek SELECT per distinct token (the fts5vocab
        equality plan); no corpus scan.
        """

        result: dict[str, int] = {str(term): 0 for term in terms}
        if not result:
            return result
        folded = self._fold_fts_terms(list(result))
        token_frequency: dict[str, int] = {}
        for term, token in folded.items():
            if token is None:
                continue
            frequency = token_frequency.get(token)
            if frequency is None:
                row = self._conn.execute(
                    "SELECT doc FROM nodes_fts_vocab WHERE term = ? AND col = 'content'", (token,)
                ).fetchone()
                frequency = int(row["doc"]) if row is not None else 0
                token_frequency[token] = frequency
            result[term] = frequency
        return result

    def fts_document_count(self) -> int:
        """Total number of documents in the ``nodes_fts`` index (c-TF-IDF ``N``).

        Counted from the ``nodes_fts_docsize`` shadow table — one small row
        per indexed document, maintained by FTS5 because ``nodes_fts`` keeps
        the default ``columnsize=1`` — so this reads a handful of pages where
        ``COUNT(*)`` over ``nodes_fts`` itself would walk every content-bearing
        row. Includes soft-deleted (decayed) nodes, matching the corpus
        :meth:`term_document_frequencies` counts over.
        """

        row = self._conn.execute("SELECT COUNT(*) FROM nodes_fts_docsize").fetchone()
        return int(row[0])

    def _fold_fts_terms(self, terms: Sequence[str]) -> dict[str, str | None]:
        """Fold each unique term to its single ``unicode61`` token, else None.

        None marks a term the DF contract sends to 0: it folded to zero
        tokens or to several. ``terms`` must be unique (the callers pass dict
        keys); rowids attribute each token instance back to its input.
        """

        if not self._fts_term_tokenizer_ready:
            for statement in _FTS_TERM_TOKENIZER_SQL:
                self._conn.execute(statement)
            self._fts_term_tokenizer_ready = True
        with self.transaction():
            self._conn.execute("DELETE FROM temp.fts_term_tokenizer")
            self._conn.executemany(
                "INSERT INTO temp.fts_term_tokenizer(rowid, term) VALUES (?, ?)",
                list(enumerate(terms)),
            )
            instances: dict[int, list[str]] = {}
            for row in self._conn.execute(
                "SELECT doc, term FROM temp.fts_term_tokenizer_instances"
            ):
                instances.setdefault(int(row["doc"]), []).append(str(row["term"]))
        folded: dict[str, str | None] = {}
        for index, term in enumerate(terms):
            tokens = instances.get(index, [])
            folded[term] = tokens[0] if len(tokens) == 1 else None
        return folded

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

        Superseded by :meth:`iter_chunk_embedding_rows`. Kept working for as
        long as ``nodes.embedding`` exists; once an operator drops the column
        this yields nothing rather than raising, so a reader still on the legacy
        path degrades to an empty vector channel instead of a crash.
        """

        if not self._node_embedding_column_present():
            return

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

    # ------------------------------------------------------------------
    # Chunk embeddings (schema v6)
    # ------------------------------------------------------------------

    def set_chunk_embedder(self, embedder: ChunkEmbedder | None) -> None:
        """Install the batch encoder used to embed chunk texts on the write path.

        A caller that already holds a loaded model should hand it over rather
        than let the store build a second one: the model is the expensive part,
        and two copies double both load time and resident memory. Passing
        ``None`` restores the lazy default.
        """

        with self._chunk_embedder_lock:
            self._chunk_embedder = (
                embedder if embedder is not None else LocalEmbeddingModel(
                    model_name=self.config.embedding_model,
                ).embed_batch
            )

    def _resolve_chunk_embedder(self) -> ChunkEmbedder:
        with self._chunk_embedder_lock:
            return self._chunk_embedder

    def iter_chunk_embedding_rows(
        self,
        *,
        scope: str | None = None,
        level: NodeLevel | None = None,
        include_decayed: bool = False,
        include_identity: bool = False,
    ) -> Iterator[tuple[str, int, memoryview] | tuple[str, int, memoryview, str]]:
        """Yield ``(node_id, chunk_index, embedding)`` for live chunks.

        The v6 analogue of :meth:`iter_embedding_rows` and the read contract the
        vector channel consumes. Each third element is a ``memoryview`` over a
        float32 little-endian BLOB, ready for
        ``np.frombuffer(view, dtype=CHUNK_EMBEDDING_DTYPE)``; its length in
        values is the row's recorded ``dimensions``, which
        :meth:`chunk_embedding_dimensions` reports for the same filters.

        Rows arrive grouped by node and ordered by ``chunk_index`` within a
        node, so a consumer can max-pool a node's chunks in one pass without
        sorting. That is a promise of the ``ORDER BY``, not of a query plan:
        the scoped form pays a temp b-tree for it (~3 ms of the scan below),
        which is worth not having the grouping silently break the day the
        planner picks a different join order. Liveness comes from the join to
        ``nodes``: soft deletion only sets ``nodes.decayed``, and nothing
        hard-deletes chunk rows, so absence of a row never means "node is gone".

        Measured on an on-disk 12840-node / 44940-chunk database (65.8 MB of
        vectors, against 134.6 MB of JSON for the same corpus today): ~44 ms to
        scan every chunk, ~75 ms to go from cold to one ``(44940, 384)`` float32
        matrix via ``np.frombuffer(b"".join(views), dtype=CHUNK_EMBEDDING_DTYPE)``,
        and ~1.2 ms for the cosine matmul after that. Most of the scan is the
        irreducible cost of pulling 45k rows through the Python sqlite3 driver —
        an unjoined ``SELECT`` over the same rows is ~24 ms — so a consumer that
        wants this cheaper should cache the matrix, not micro-tune the query.

        ``include_identity`` appends the immutable chunk row id. A max-pool
        winner can then resolve its source address even if the node's chunks
        are replaced concurrently; an ordinal alone cannot identify a generation.

        Yields nothing on a pre-v6 database rather than raising, so a snapshot
        opened without migrations degrades instead of crashing.
        """

        if not self._chunk_table_present():
            return

        clauses = ["n.decayed = 0"] if not include_decayed else []
        params: list[Any] = []
        if level is not None:
            self._validate_level(level)
            clauses.append("n.level = ?")
            params.append(level)
        if scope is not None:
            clauses.append("n.scope = ?")
            params.append(scope)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        # A bare tuple cursor: at 45k rows the per-row sqlite3.Row wrapper is a
        # measurable share of the scan, and nothing here needs lookup by name.
        cursor = self._conn.cursor()
        cursor.row_factory = None
        cursor.execute(
            f"""
            SELECT c.node_id, c.chunk_index, c.embedding{', c.id' if include_identity else ''}
            FROM {CHUNK_EMBEDDING_TABLE} c
            JOIN nodes n ON n.id = c.node_id
            {where}
            ORDER BY c.node_id ASC, c.chunk_index ASC
            """,
            params,
        )
        while True:
            batch = cursor.fetchmany(1024)
            if not batch:
                return
            for row in batch:
                item = (str(row[0]), int(row[1]), memoryview(row[2]))
                yield (*item, str(row[3])) if include_identity else item

    def get_chunk_passages(self, chunk_ids: Sequence[str]) -> dict[str, SourcePassage]:
        """Resolve selected row identities without reading vectors or node text.

        Legacy chunks have no character coordinates. Never guess them from
        token offsets: the tokenizer used to write those rows is unknown.
        """
        if not chunk_ids or not self._chunk_table_present():
            return {}
        columns = {row[1] for row in self._conn.execute(
            f"PRAGMA table_info({CHUNK_EMBEDDING_TABLE})"
        )}
        if not {"char_start", "char_end"} <= columns:
            return {}
        passages = {}
        for batch in _node_read_batches(chunk_ids):
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"""SELECT c.id, c.content_fingerprint, c.char_start, c.char_end
                FROM {CHUNK_EMBEDDING_TABLE} c JOIN nodes n ON n.id = c.node_id
                WHERE c.id IN ({placeholders}) AND n.decayed = 0
                  AND c.content_fingerprint = n.content_fingerprint
                  AND c.char_start >= 0 AND c.char_end > c.char_start""",
                batch,
            )
            for row in rows:
                passages[str(row[0])] = SourcePassage(str(row[1]), int(row[2]), int(row[3]))
        return passages

    def chunk_corpus_revisions(
        self, scopes: Sequence[str] | None = None,
    ) -> dict[str, int] | None:
        """Transactional revisions for the requested scopes, or every scope.

        Membership and vector changes advance only the affected scopes. A
        missing scope is empty at revision zero; retained empty-scope revisions
        prevent delete/reinsert aliasing. Read-only snapshots without this
        format return None, requiring uncached reads. This is one SQL snapshot,
        not a transaction around the caller's subsequent reads.
        """

        if scopes is None:
            rows = self._conn.execute(
                "SELECT key, value FROM metadata WHERE key = ? "
                "OR (key >= ? AND key < ?)",
                (_CHUNK_SCOPE_REVISION_FORMAT, _CHUNK_SCOPE_REVISION_PREFIX,
                 "chunk_scope_revision;"),
            ).fetchall()
        else:
            keys = [_CHUNK_SCOPE_REVISION_FORMAT,
                    *(_CHUNK_SCOPE_REVISION_PREFIX + scope for scope in scopes)]
            rows = self._conn.execute(
                f"SELECT key, value FROM metadata WHERE key IN ({','.join('?' for _ in keys)})",
                keys,
            ).fetchall()
        values = dict(rows)
        if values.pop(_CHUNK_SCOPE_REVISION_FORMAT, None) != "1":
            return None
        revisions = {key[len(_CHUNK_SCOPE_REVISION_PREFIX):]: int(value)
                     for key, value in values.items()}
        if scopes is not None:
            for scope in scopes:
                revisions.setdefault(scope, 0)
        return revisions

    def chunk_embedding_dimensions(
        self,
        *,
        scope: str | None = None,
        level: NodeLevel | None = None,
        include_decayed: bool = False,
    ) -> tuple[int, ...]:
        """Distinct recorded dimensions over the same rows the scan would yield.

        The authority on vector width: a BLOB's byte length cannot distinguish a
        short vector from a truncated write, so consumers reshape by this and
        treat a result other than a single value as a corpus that must be
        grouped (or re-embedded) before it can become one matrix.
        """

        if not self._chunk_table_present():
            return ()

        clauses = ["n.decayed = 0"] if not include_decayed else []
        params: list[Any] = []
        if level is not None:
            self._validate_level(level)
            clauses.append("n.level = ?")
            params.append(level)
        if scope is not None:
            clauses.append("n.scope = ?")
            params.append(scope)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"""
            SELECT DISTINCT c.dimensions AS dimensions
            FROM {CHUNK_EMBEDDING_TABLE} c
            JOIN nodes n ON n.id = c.node_id
            {where}
            ORDER BY dimensions ASC
            """,
            params,
        ).fetchall()
        return tuple(int(row["dimensions"]) for row in rows)

    def list_node_chunks(self, node_id: str) -> list[NodeChunkEmbedding]:
        """Return one node's chunks decoded into vectors, in ordinal order."""

        if not self._chunk_table_present():
            return []
        rows = self._conn.execute(
            f"""
            SELECT * FROM {CHUNK_EMBEDDING_TABLE}
            WHERE node_id = ?
            ORDER BY chunk_index ASC
            """,
            (str(node_id),),
        ).fetchall()
        return [_chunk_embedding_from_row(row) for row in rows]

    def get_node_chunk_vectors(
        self, node_ids: Sequence[str],
    ) -> dict[str, list[list[float]]]:
        """Read only vectors for selected nodes, in each node's chunk order."""

        with self.iter_node_chunk_vectors(node_ids) as groups:
            return dict(groups)

    @contextmanager
    def iter_node_chunk_vectors(
        self, node_ids: Sequence[str],
    ) -> Iterator[Iterator[tuple[str, list[list[float]]]]]:
        """Yield selected nodes' decoded chunks one node at a time.

        Each node appears once, with all its chunks in ordinal order. Release
        each group before advancing to bound decoded storage by the largest
        selected node. Exiting the context closes any active batch cursor.
        """

        def groups() -> Iterator[tuple[str, list[list[float]]]]:
            if not node_ids or not self._chunk_table_present():
                return
            batch_size = min(999, self._conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER))
            for batch in _node_read_batches(map(str, node_ids), batch_size=batch_size):
                placeholders = ",".join("?" for _ in batch)
                rows = self._conn.execute(
                    f"""SELECT node_id, chunk_index, dimensions, embedding
                    FROM {CHUNK_EMBEDDING_TABLE}
                    WHERE node_id IN ({placeholders})
                    ORDER BY node_id, chunk_index""",
                    batch,
                )
                try:
                    for node_id, node_rows in groupby(rows, key=lambda row: str(row["node_id"])):
                        chunks = []
                        for row in node_rows:
                            # Validate every selected row, including unused widths.
                            int(row["chunk_index"])
                            chunks.append(unpack_chunk_embedding(row["embedding"], int(row["dimensions"])))
                        yield node_id, chunks
                        del chunks
                finally:
                    rows.close()

        with closing(groups()) as iterator:
            yield iterator

    def count_node_chunks(self, node_id: str | None = None) -> int:
        """Count chunk rows, for one node or for the whole database."""

        if not self._chunk_table_present():
            return 0
        if node_id is None:
            row = self._conn.execute(
                f"SELECT COUNT(*) AS count FROM {CHUNK_EMBEDDING_TABLE}"
            ).fetchone()
        else:
            row = self._conn.execute(
                f"SELECT COUNT(*) AS count FROM {CHUNK_EMBEDDING_TABLE} WHERE node_id = ?",
                (str(node_id),),
            ).fetchone()
        return int(row["count"])

    def node_chunks_are_current(self, node_id: str) -> bool:
        """Whether ``node_id`` has chunks and all of them match its content.

        False both for a node that was never chunked and for one whose content
        changed after chunking — the two cases a re-embed has to cover, which is
        why :meth:`list_unchunked_nodes` returns them together.
        """

        if not self._chunk_table_present():
            return False
        row = self._conn.execute(
            f"""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN c.content_fingerprint = n.content_fingerprint
                            THEN 1 ELSE 0 END) AS fresh
            FROM {CHUNK_EMBEDDING_TABLE} c
            JOIN nodes n ON n.id = c.node_id
            WHERE c.node_id = ?
            """,
            (str(node_id),),
        ).fetchone()
        total = int(row["total"] or 0)
        return total > 0 and int(row["fresh"] or 0) == total

    def list_unchunked_nodes(
        self,
        *,
        scope: str | None = None,
        level: NodeLevel | None = None,
        limit: int = 200,
        after: tuple[str, str] | None = None,
    ) -> list[Node]:
        """Active nodes whose chunks are missing or stale, oldest work first.

        Full-node decoding is strict. Optional repair can select positions with
        :meth:`list_unchunked_node_positions` before hydrating individual sources.
        """

        return [_node_from_row(row) for row in self._unchunked_node_rows(
            scope=scope, level=level, limit=limit, after=after, positions_only=False,
        )]

    def list_unchunked_node_positions(
        self,
        *,
        scope: str | None = None,
        level: NodeLevel | None = None,
        limit: int = 200,
        after: tuple[str, str] | None = None,
    ) -> list[tuple[str, str]]:
        """Pending timestamp/id positions, without decoding source bodies."""

        return [(str(row["timestamp"]), str(row["id"]))
                for row in self._unchunked_node_rows(
                    scope=scope, level=level, limit=limit, after=after, positions_only=True,
                )]

    def _unchunked_node_rows(
        self, *, scope: str | None, level: NodeLevel | None, limit: int,
        after: tuple[str, str] | None, positions_only: bool,
    ) -> list[sqlite3.Row]:
        """Select current pending sources with either full rows or positions.

        The v6 analogue of :meth:`list_unembedded_nodes`, and deliberately not
        the same question: a node whose content was edited still has a
        ``nodes.embedding`` and so is invisible to the legacy query, but its
        chunks were dropped as stale and it does need re-embedding.

        Whitespace-only content is excluded. The chunker yields no windows for
        text with no tokens, so such a node can never acquire a chunk row — and
        a backfill driven by "keep going until this list is empty" would spin on
        it forever. SQLite's one-argument ``TRIM`` strips spaces only, hence the
        explicit whitespace set.

        Materialize pending identities before reading content: completed nodes
        and their chunk vectors stay behind covering indexes. Whitespace and
        the limit are applied together only after that single-statement probe.
        With ``after``, start after that timestamp/id and wrap to the beginning;
        the position need not still identify a pending or existing node.
        """

        if not self._chunk_table_present():
            return []

        clauses = ["n.decayed = 0"]
        params: list[Any] = []
        if level is not None:
            self._validate_level(level)
            clauses.append("n.level = ?")
            params.append(level)
        if scope is not None:
            clauses.append("n.scope = ?")
            params.append(scope)
        order = "n.timestamp ASC, n.id ASC"
        if after is not None:
            order = "(n.timestamp, n.id) <= (?, ?) ASC, " + order
            params.extend(after)
        params.append(int(limit))
        return self._conn.execute(
            f"""
            WITH pending AS MATERIALIZED (
                SELECT n.id FROM nodes n
                WHERE {' AND '.join(clauses)}
                  AND NOT EXISTS (
                    SELECT 1 FROM {CHUNK_EMBEDDING_TABLE} c
                    WHERE c.node_id = n.id
                      AND c.content_fingerprint = n.content_fingerprint
                  )
            )
            SELECT {'n.timestamp, n.id' if positions_only else 'n.*'}
            FROM pending p JOIN nodes n ON n.id = p.id
            WHERE TRIM(n.content, {_SQL_WHITESPACE}) != ''
            ORDER BY {order}
            LIMIT ?
            """,
            params,
        ).fetchall()

    def delete_node_chunks(self, node_id: str) -> int:
        """Remove every chunk of one node. Returns the number of rows deleted."""

        if not self._chunk_table_present():
            return 0
        with self.transaction():
            cur = self._conn.execute(
                f"DELETE FROM {CHUNK_EMBEDDING_TABLE} WHERE node_id = ?",
                (str(node_id),),
            )
        return int(cur.rowcount or 0)

    def replace_node_chunks(
        self,
        node_id: str,
        chunks: Sequence[tuple[TextChunk, Sequence[float]]],
        *,
        content_fingerprint: str | None = None,
    ) -> int:
        """Make ``chunks`` the node's complete chunk set, atomically.

        Delete-then-insert inside one transaction rather than upsert-by-ordinal:
        a re-embed can produce fewer chunks than last time (an edit shortened the
        content, or a different tokenizer packs windows differently), and an
        upsert would leave the surplus tail behind. A node is never observable
        holding chunks from two different embeddings of its content.

        ``content_fingerprint`` defaults to the node's current
        ``nodes.content_fingerprint``, which is what makes a later edit
        detectable.
        """

        node_id = str(node_id)
        if not self._chunk_table_present():
            raise RuntimeError(
                f"{CHUNK_EMBEDDING_TABLE} is missing; open the database through "
                "MemoryStore so migrations run before writing chunks"
            )
        row = self._conn.execute(
            "SELECT content_fingerprint FROM nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        if row is None:
            raise KeyError(node_id)
        fingerprint = str(content_fingerprint or row["content_fingerprint"] or "")
        with self.transaction():
            written = self._replace_node_chunks(node_id, chunks, fingerprint)
        return written

    def refresh_pending_chunks(
        self, selected: Node, embed: Callable[[str], Sequence[float]], *,
        guard: Callable[[], AbstractContextManager[Any]] = nullcontext,
    ) -> bool:
        """Prepare a complete pending index and apply it only to the same source.

        Model work uses immutable text outside the shared-store guard and any
        transaction. Current source, representative and index identities decide
        whether the prepared replacement still belongs to this node.
        """
        def identity(node: Node) -> tuple[Any, ...]:
            return (node.content, node.scope, node.level, node.decayed,
                    node.materializer, node.derivation_identity,
                    tuple(node.embedding or ()))

        def chunk_ids() -> tuple[str, ...]:
            return tuple(row[0] for row in self._conn.execute(
                f"SELECT id FROM {CHUNK_EMBEDDING_TABLE} WHERE node_id = ? ORDER BY id",
                (selected.id,),
            ))

        with guard():
            current = self.get_node(selected.id)
            if (current is None or current.decayed or identity(current) != identity(selected)
                    or self.node_chunks_are_current(selected.id)):
                return False
            before = chunk_ids()
            source = identity(current)
            content = current.content
            representative = bool(
                current.materializer in ("cluster", "cross-scope-promotion") and current.embedding
            )

        embedding = None if representative else tuple(embed(content))
        pairs = tuple((chunk, tuple(vector)) for chunk, vector in
                      self._chunk_embeddings_for(content, embedding, require_complete=True))
        if not pairs:
            raise ValueError("pending index requires a complete nonempty chunk set")
        with guard(), self.transaction(immediate=True):
            current = self.get_node(selected.id)
            if (current is None or identity(current) != source or chunk_ids() != before
                    or self.node_chunks_are_current(selected.id)):
                return False
            self._replace_node_chunks(selected.id, pairs, _content_fingerprint(content))
            if embedding is not None and self._node_embedding_column_present():
                self._conn.execute("UPDATE nodes SET embedding = ?, updated_at = ? WHERE id = ?",
                                   (_json_dumps(embedding), _utc_now(), selected.id))
        return True

    def refresh_legacy_chunks(
        self, node_id: str, content: str, scored_chunk_id: str, *,
        max_chars: int, max_chunks: int,
        guard: Callable[[], AbstractContextManager[Any]] = nullcontext,
    ) -> bool:
        """Conditionally replace one selected legacy index, leaving its node alone.

        Bound both source tokenization and encoding before doing model work.
        Read and apply under the caller's store guard; tokenize and encode only
        immutable text outside it and the write transaction. Recheck the source,
        scope and immutable scored row under the writer lock, so a concurrent
        edit or replacement wins. Direct callers retain their own synchronization.
        """
        if len(content) > max_chars or max_chunks <= 0:
            return False
        fingerprint = _content_fingerprint(content)

        def legacy_scope() -> str | None:
            row = self._conn.execute(
                f"""SELECT n.scope FROM {CHUNK_EMBEDDING_TABLE} c
                JOIN nodes n ON n.id = c.node_id
                WHERE c.id = ? AND c.node_id = ? AND n.decayed = 0
                  AND n.content_fingerprint = ? AND c.content_fingerprint = ?
                  AND n.content = ? AND (c.char_start IS NULL OR c.char_end IS NULL)""",
                (scored_chunk_id, node_id, fingerprint, fingerprint, content),
            ).fetchone()
            return row["scope"] if row is not None else None

        with guard():
            if (not self._chunk_table_present()
                    or self._conn.execute("PRAGMA query_only").fetchone()[0]):
                return False
            scope = legacy_scope()
            if scope is None:
                return False
            # Inspect only enough row metadata to reject an oversized old index.
            rows = self._conn.execute(
                f"SELECT 1 FROM {CHUNK_EMBEDDING_TABLE} WHERE node_id = ? LIMIT ?",
                (node_id, max_chunks + 1),
            ).fetchall()
            if len(rows) > max_chunks:
                return False
        pairs = self._chunk_embeddings_for(content, None, max_chunks=max_chunks)
        if not pairs:
            return False
        with guard(), self.transaction(immediate=True):
            if legacy_scope() != scope:
                return False
            self._replace_node_chunks(node_id, pairs, fingerprint)
        return True

    def _replace_node_chunks(
        self,
        node_id: str,
        chunks: Sequence[tuple[TextChunk, Sequence[float]]],
        fingerprint: str,
    ) -> int:
        """Transaction body of :meth:`replace_node_chunks` (caller holds the txn)."""

        self._conn.execute(
            f"DELETE FROM {CHUNK_EMBEDDING_TABLE} WHERE node_id = ?",
            (node_id,),
        )
        if not chunks:
            return 0
        source = self._conn.execute(
            "SELECT content, content_fingerprint FROM nodes WHERE id = ?", (node_id,),
        ).fetchone()
        now = _utc_now()
        payload = []
        for expected_index, (chunk, vector) in enumerate(chunks):
            values = list(vector)
            if not values:
                raise ValueError(f"chunk {expected_index} of {node_id} has an empty vector")
            if chunk.chunk_index != expected_index:
                raise ValueError(
                    f"chunk ordinals must be dense and ordered from 0; got "
                    f"{chunk.chunk_index} at position {expected_index} of {node_id}"
                )
            addressed = (
                source is not None and source["content_fingerprint"] == fingerprint
                and 0 <= chunk.char_start < chunk.char_end <= len(source["content"])
                and source["content"][chunk.char_start:chunk.char_end] == chunk.text
            )
            payload.append(
                (
                    new_ulid(),
                    node_id,
                    expected_index,
                    len(values),
                    pack_chunk_embedding(values),
                    int(chunk.token_start),
                    int(chunk.token_end),
                    fingerprint,
                    now,
                    now,
                    chunk.char_start if addressed else None,
                    chunk.char_end if addressed else None,
                )
            )
        self._conn.executemany(
            f"""
            INSERT INTO {CHUNK_EMBEDDING_TABLE} (
                id, node_id, chunk_index, dimensions, embedding,
                token_start, token_end, content_fingerprint, created_at, updated_at,
                char_start, char_end
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        return len(payload)

    def _chunk_embeddings_for(
        self,
        content: str,
        node_embedding: Sequence[float] | None,
        *,
        max_chunks: int | None = None,
        require_complete: bool = False,
    ) -> list[tuple[TextChunk, Sequence[float]]]:
        """Split ``content`` into windows and pair each with its vector.

        The single-window case reuses ``node_embedding`` instead of encoding
        again: when the whole content fits one window the chunk text *is* the
        content, so the vector the caller already computed is the chunk's
        vector, and re-encoding would only spend the model's time to reproduce
        it. That case covers the short nodes, which are the majority of writes.

        Multi-window content is where the whole-node vector is wrong — it only
        ever saw the first window — so every chunk is encoded on its own.
        """

        chunks = chunk_text(content)
        if not chunks or (max_chunks is not None and len(chunks) > max_chunks):
            return []
        if len(chunks) == 1 and node_embedding:
            return [(chunks[0], list(node_embedding))]
        # Preparation uses only immutable text and encoder-owned state. Keep
        # injected encoders and their replacement serialized too; never consult
        # SQLite, session state or the current knowledge projection here.
        with self._chunk_embedder_lock:
            vectors = [
                list(vector) for vector in
                self._resolve_chunk_embedder()([chunk.text for chunk in chunks])
            ]
        if len(vectors) != len(chunks):
            raise ValueError(
                f"chunk embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )
        if (max_chunks is not None or require_complete) and any(not vector for vector in vectors):
            raise ValueError("complete chunk indexing requires every chunk vector")
        return [
            (chunk, list(vector))
            for chunk, vector in zip(chunks, vectors, strict=True)
            if list(vector)
        ]

    def _write_node_chunks(
        self,
        node_id: str,
        content: str,
        fingerprint: str,
        node_embedding: Sequence[float] | None,
    ) -> int:
        """Re-chunk one node inside the caller's transaction."""

        pairs = self._chunk_embeddings_for(content, node_embedding)
        return self._replace_node_chunks(node_id, pairs, fingerprint)

    def drop_node_embedding_column(self) -> bool:
        """Drop the legacy ``nodes.embedding`` JSON column. Operator command only.

        Deliberately not part of schema init and not part of any backfill: the
        production MCP server runs code that still reads ``nodes.embedding``, and
        it must keep serving reads for the whole time chunks are being written.
        Additive first, drop later, and only when an operator says so — a drop
        wired into ``_initialize_schema`` would fire the moment a new build
        opened the database, under a server that cannot survive it.

        Returns True if the column was dropped, False if it was already gone
        (idempotent). Reclaiming the freed pages needs a separate ``VACUUM``,
        which is left to the operator because it rewrites the whole file.
        """

        if not self._node_embedding_column_present(refresh=True):
            return False
        with self.transaction():
            self._conn.execute("DROP INDEX IF EXISTS idx_nodes_embedded_active_scope")
            self._conn.execute("ALTER TABLE nodes DROP COLUMN embedding")
        self._invalidate_schema_shape_cache()
        return True

    def list_unembedded_nodes(
        self,
        *,
        scope: str | None = None,
        level: NodeLevel | None = None,
        limit: int = 200,
    ) -> list[Node]:
        """Return active nodes with no stored embedding (for lazy backfill).

        Empty once ``nodes.embedding`` is dropped: with no column there is no
        such thing as an unembedded node, and the lazy-backfill loop that drains
        this must terminate rather than raise. The chunk-era question is
        :meth:`list_unchunked_nodes`.
        """

        if not self._node_embedding_column_present():
            return []

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

    @staticmethod
    def _scope_filter(
        alias: str = "",
        *,
        scope: str | None = None,
        exclude_scope: str | None = None,
        scope_prefix: str | None = None,
    ) -> tuple[list[str], list[Any]]:
        """Shared scope predicates for the two halves of a similarity search.

        Both the node query and the chunk query behind it have to narrow to the
        same set, or the chunk side does far more work than the node side will
        use. ``alias`` is the table qualifier ("" for a bare nodes query, "n."
        when joined).
        """

        clauses: list[str] = []
        params: list[Any] = []
        if scope is not None:
            clauses.append(f"{alias}scope = ?")
            params.append(scope)
        if exclude_scope is not None:
            clauses.append(f"{alias}scope != ?")
            params.append(exclude_scope)
        if scope_prefix is not None:
            clauses.append(f"{alias}scope >= ?")
            params.append(scope_prefix)
            upper_bound = _prefix_upper_bound(scope_prefix)
            if upper_bound is not None:
                clauses.append(f"{alias}scope < ?")
                params.append(upper_bound)
        return clauses, params

    def _chunk_max_pool_scores(
        self,
        embedding: Sequence[float],
        *,
        level: NodeLevel | None = None,
        scope: str | None = None,
        exclude_scope: str | None = None,
        scope_prefix: str | None = None,
    ) -> dict[str, float]:
        """Cosine of ``embedding`` against each live node's best chunk.

        One pass rather than a query per node, so the caller pays the same
        single scan the JSON path paid. Reads ``dimensions`` alongside the BLOB
        instead of going through :meth:`iter_chunk_embedding_rows`, because
        decoding needs the recorded width and the byte length is not allowed to
        stand in for it.
        """

        if not self._chunk_table_present():
            return {}

        query = list(embedding)
        clauses = ["n.decayed = 0"]
        params: list[Any] = []
        if level is not None:
            self._validate_level(level)
            clauses.append("n.level = ?")
            params.append(level)
        scope_clauses, scope_params = self._scope_filter(
            "n.", scope=scope, exclude_scope=exclude_scope, scope_prefix=scope_prefix
        )
        clauses.extend(scope_clauses)
        params.extend(scope_params)
        rows = self._conn.execute(
            f"""
            SELECT c.node_id AS node_id, c.dimensions AS dimensions,
                   c.embedding AS embedding
            FROM {CHUNK_EMBEDDING_TABLE} c
            JOIN nodes n ON n.id = c.node_id
            WHERE {' AND '.join(clauses)}
            """,
            params,
        )
        best: dict[str, float] = {}
        for row in rows:
            node_id = str(row["node_id"])
            vector = unpack_chunk_embedding(row["embedding"], int(row["dimensions"]))
            score = cosine_similarity(query, vector)
            if score > best.get(node_id, float("-inf")):
                best[node_id] = score
        return best

    def promotion_candidate_page(
        self, *, limit: int, after: tuple[str, str] = ("", ""),
        through_rowid: int | None = None,
    ) -> tuple[list[Node], int]:
        """Page existing project representatives in a finite operation window.

        The first read pins the insertion boundary; subsequent pages advance by
        scope/id through the active level/source index. New rows belong to a
        later pass. No embeddings are prepared or repaired by this read.
        """
        if through_rowid is None:
            through_rowid = int(self._conn.execute(
                "SELECT COALESCE(MAX(rowid), 0) FROM nodes"
            ).fetchone()[0])
        if limit <= 0 or not self._node_embedding_column_present():
            return [], through_rowid
        rows = self._conn.execute(
            """
            SELECT * FROM nodes INDEXED BY idx_nodes_active_level_source
            WHERE level = 'concept' AND decayed = 0 AND embedding IS NOT NULL
              AND scope >= 'project:' AND scope < 'project;'
              AND (scope, id) > (?, ?) AND rowid <= ?
              AND NOT EXISTS (
                  SELECT 1 FROM connections WHERE target_id = nodes.id AND type = 'supersedes'
              )
            ORDER BY scope, id LIMIT ?
            """,
            (*after, through_rowid, int(limit)),
        ).fetchall()
        return [_node_from_row(row) for row in rows], through_rowid

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
        """Scan active embedded nodes and return cosine matches above a threshold.

        Compatibility path for the v6 rollout, so consolidation's callers keep
        working without every one of them learning about chunks. While
        ``nodes.embedding`` is present it stays the scoring vector and results
        are bit-identical to v5 — that matters because the callers' thresholds
        (cross-scope promotion and semantic retrieval) were calibrated
        against single-vector scores, and max-pooling chunks would silently
        raise every score and merge more aggressively than anyone asked for.
        Once the column is gone every node scores from its chunks instead, as
        the best of them — the same max-pool quantity the vector channel uses.
        Nothing is lost in the handover: while the column exists a node only
        ever has chunks if it also has a legacy vector, so the set of nodes this
        can match is the same on both sides of the drop.
        """

        self._validate_level(level)
        if not embedding or limit <= 0:
            return []

        legacy_embedding = self._node_embedding_column_present()
        chunk_scores: dict[str, float] = {}
        if not legacy_embedding:
            chunk_scores = self._chunk_max_pool_scores(
                embedding,
                level=level,
                scope=scope,
                exclude_scope=exclude_scope,
                scope_prefix=scope_prefix,
            )

        clauses = ["level = ?", "decayed = 0"]
        params: list[Any] = [level]
        if legacy_embedding:
            clauses.append("embedding IS NOT NULL")
        scope_clauses, scope_params = self._scope_filter(
            scope=scope, exclude_scope=exclude_scope, scope_prefix=scope_prefix
        )
        clauses.extend(scope_clauses)
        params.extend(scope_params)

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
            if legacy_embedding:
                score = cosine_similarity(embedding, node.embedding)
            else:
                score = chunk_scores.get(node.id, 0.0)
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
        with self.transaction():
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

        _, effective = self.evaluate_retrieval_weight_floors(scope, weights)
        return effective

    def evaluate_retrieval_weight_floors(
        self,
        scope: str,
        weights: RetrievalWeights,
    ) -> tuple[RetrievalPolicyFloors | None, RetrievalWeights]:
        """Read applicable floors and effective weights from the same evidence."""

        normalized = weights.normalized()
        floors = self.config.retrieval_policy_floors.get(_scope_family(scope))
        if floors is None:
            return None, normalized

        vector_min = (
            floors.vector_min
            if floors.vector_min > 0.0 and self._has_vector_evidence(scope) else 0.0
        )
        graph_min = (
            floors.graph_min
            if floors.graph_min > 0.0 and self._has_graph_evidence(scope) else 0.0
        )
        # bm25 is a lexical channel that is always available, so its floor is
        # enforced unconditionally (no evidence gate). Without it, winner-take-all
        # feedback (feedback._method_signals) drives bm25 to 0 and exact-keyword /
        # identifier recall stops contributing to ranking.
        bm25_min = floors.bm25_min
        applicable = RetrievalPolicyFloors(
            # A BM25 surplus can move only to a channel with an active minimum.
            bm25_max=floors.bm25_max if vector_min > 0.0 or graph_min > 0.0 else 1.0,
            vector_min=vector_min,
            graph_min=graph_min,
            bm25_min=bm25_min,
        )
        if vector_min <= 0.0 and graph_min <= 0.0 and bm25_min <= 0.0:
            return applicable, normalized

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
        return applicable, floored

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
        """Whether ``scope`` has anything for the vector channel to match on.

        True for either storage shape, so the weight floors keep answering the
        same question across the rollout: a scope whose nodes are chunked but
        whose legacy column was dropped still has vector evidence, and a scope
        on a pre-v6 file still has it through the JSON column.
        """

        if self._node_embedding_column_present():
            row = self._conn.execute(
                """
                SELECT 1
                FROM nodes
                WHERE scope = ? AND decayed = 0 AND embedding IS NOT NULL
                LIMIT 1
                """,
                (scope,),
            ).fetchone()
            if row is not None:
                return True
        if not self._chunk_table_present():
            return False
        row = self._conn.execute(
            f"""
            SELECT 1
            FROM {CHUNK_EMBEDDING_TABLE} c
            JOIN nodes n ON n.id = c.node_id
            WHERE n.scope = ? AND n.decayed = 0
            LIMIT 1
            """,
            (scope,),
        ).fetchone()
        return row is not None

    def _has_graph_evidence(self, scope: str) -> bool:
        """Whether an active scoped node has a semantic edge to an active node."""
        relation = semantic_relation_sql("c")
        row = self._conn.execute(
            f"""
            SELECT 1
            FROM nodes selected
            WHERE selected.scope = ? AND selected.decayed = 0
              AND (
                EXISTS (
                    SELECT 1 FROM connections c
                    JOIN nodes target ON target.id = c.target_id
                    WHERE c.source_id = selected.id AND target.decayed = 0
                      AND {relation}
                )
                OR EXISTS (
                    SELECT 1 FROM connections c
                    JOIN nodes source ON source.id = c.source_id
                    WHERE c.target_id = selected.id AND source.decayed = 0
                      AND {relation}
                )
              )
            LIMIT 1
            """,
            (scope,),
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Query anchors (schema v7)
    #
    # Thin table access only: rows in, dataclasses out, no policy. Which
    # anchor an incoming query reinforces, how much weight one grounded
    # consumption is worth, and when an anchor has gone stale all live in
    # living_memory.query_anchors. The one exception is deliberate and is
    # documented on _insert_connection: corrections and verified exact copies
    # re-point anchor edges in the same transaction.
    # ------------------------------------------------------------------

    def get_query_anchor(self, anchor_id: str) -> QueryAnchor | None:
        if not self._anchor_tables_present():
            return None
        row = self._conn.execute(
            f"SELECT * FROM {QUERY_ANCHOR_TABLE} WHERE id = ?",
            (str(anchor_id),),
        ).fetchone()
        return None if row is None else _query_anchor_from_row(row)

    def has_query_anchor(self, scope: str, fingerprint: str) -> bool:
        """Test exact identity, including decayed anchors, without reading the payload."""

        if not self._anchor_tables_present():
            return False
        return self._conn.execute(
            f"SELECT 1 FROM {QUERY_ANCHOR_TABLE} WHERE scope = ? AND fingerprint = ?",
            (normalize_scope(scope), str(fingerprint)),
        ).fetchone() is not None

    def find_query_anchor(self, scope: str, fingerprint: str) -> QueryAnchor | None:
        """Exact-identity lookup on the ``UNIQUE(scope, fingerprint)`` key."""

        if not self._anchor_tables_present():
            return None
        row = self._conn.execute(
            f"SELECT * FROM {QUERY_ANCHOR_TABLE} WHERE scope = ? AND fingerprint = ?",
            (normalize_scope(scope), str(fingerprint)),
        ).fetchone()
        return None if row is None else _query_anchor_from_row(row)

    def insert_query_anchor(
        self,
        *,
        scope: str,
        query: str,
        fingerprint: str,
        embedding: Sequence[float],
        now: str | None = None,
        usefulness_score: float = 0.0,
        anchor_id: str | None = None,
    ) -> QueryAnchor:
        """Insert one anchor. Raises ``sqlite3.IntegrityError`` on a repeat key.

        ``scope`` is normalized on the way in, exactly as :meth:`_insert_node`
        normalizes a node's. An anchor stored under a raw scope string that the
        scoped reads normalize differently would be invisible to every one of
        them — written, counted, and never matched.
        """

        vector = [float(value) for value in embedding]
        if not vector:
            raise ValueError("anchor embedding must not be empty")
        stamp = now or _utc_now()
        new_id = str(anchor_id) if anchor_id else new_ulid()
        with self.transaction():
            self._conn.execute(
                f"""
                INSERT INTO {QUERY_ANCHOR_TABLE} (
                    id, scope, query, fingerprint, dimensions, embedding,
                    reinforcement_count, usefulness_score, decayed, decay_reason,
                    first_seen, last_matched_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, 0, NULL, ?, ?, ?, ?)
                """,
                (
                    new_id,
                    normalize_scope(scope),
                    str(query),
                    str(fingerprint),
                    len(vector),
                    pack_chunk_embedding(vector),
                    float(usefulness_score),
                    stamp,
                    stamp,
                    stamp,
                    stamp,
                ),
            )
        anchor = self.get_query_anchor(new_id)
        if anchor is None:  # pragma: no cover - insert just succeeded
            raise RuntimeError("query anchor insert failed")
        return anchor

    def reinforce_query_anchor(
        self,
        anchor_id: str,
        *,
        now: str | None = None,
        usefulness_delta: float = 0.0,
    ) -> QueryAnchor | None:
        """Bump reinforcement, refresh freshness, and revive a decayed anchor.

        Revival is the point: an anchor that aged out and is then asked again
        is exactly the repeated situation anchors exist for, and leaving it
        decayed would throw away every edge it had learned.
        """

        if not self._anchor_tables_present():
            return None
        stamp = now or _utc_now()
        with self.transaction():
            self._conn.execute(
                f"""
                UPDATE {QUERY_ANCHOR_TABLE}
                SET reinforcement_count = reinforcement_count + 1,
                    usefulness_score = usefulness_score + ?,
                    decayed = 0,
                    decay_reason = NULL,
                    last_matched_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (float(usefulness_delta), stamp, stamp, str(anchor_id)),
            )
        return self.get_query_anchor(anchor_id)

    def decay_query_anchor(
        self, anchor_id: str, reason: str, *, now: str | None = None
    ) -> QueryAnchor | None:
        """Soft-delete one anchor, keeping its row and its edges."""

        if not self._anchor_tables_present():
            return None
        stamp = now or _utc_now()
        with self.transaction():
            self._conn.execute(
                f"""
                UPDATE {QUERY_ANCHOR_TABLE}
                SET decayed = 1, decay_reason = ?, updated_at = ?
                WHERE id = ?
                """,
                (str(reason), stamp, str(anchor_id)),
            )
        return self.get_query_anchor(anchor_id)

    def list_query_anchors(
        self,
        *,
        scope: str | None = None,
        include_decayed: bool = False,
        limit: int | None = None,
    ) -> list[QueryAnchor]:
        if not self._anchor_tables_present():
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if scope is not None:
            clauses.append("scope = ?")
            params.append(normalize_scope(scope))
        if not include_decayed:
            clauses.append("decayed = 0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        suffix = ""
        if limit is not None:
            suffix = " LIMIT ?"
            params.append(int(limit))
        rows = self._conn.execute(
            f"""
            SELECT * FROM {QUERY_ANCHOR_TABLE}
            {where}
            ORDER BY last_matched_at DESC, created_at DESC
            {suffix}
            """,
            params,
        ).fetchall()
        return [_query_anchor_from_row(row) for row in rows]

    def query_anchor_vector_layout(self, dimensions: int) -> list[tuple[str, str]]:
        """Live ids and scopes in vector scan order, without reading BLOBs."""

        if not self._anchor_tables_present():
            return []
        return [
            (str(row[0]), str(row[1]))
            for row in self._conn.execute(
                f"SELECT id, scope FROM {QUERY_ANCHOR_TABLE} "
                "WHERE decayed = 0 AND dimensions = ?",
                (int(dimensions),),
            )
        ]

    def iter_query_anchor_vectors(
        self,
        scopes: Sequence[str] | None = None,
        *,
        dimensions: int | None = None,
        include_decayed: bool = False,
        limit: int | None = None,
    ) -> Iterator[tuple[str, str, int, memoryview]]:
        """Yield ``(anchor_id, scope, dimensions, embedding)`` for live anchors.

        The anchor analogue of :meth:`iter_chunk_embedding_rows`: a bare tuple
        cursor over BLOB views, so a matcher decodes only what it keeps. The
        corpus is small by design — one row per distinct remembered question,
        estimated at a few thousand — so this is a scan, not an ANN index.
        ``dimensions`` is the row's own recorded width, never the BLOB length.
        A requested width filters rows in SQL before their BLOBs are read.

        Yields nothing on a pre-v7 database rather than raising.
        """

        if not self._anchor_tables_present():
            return
        clauses: list[str] = []
        params: list[Any] = []
        if not include_decayed:
            clauses.append("decayed = 0")
        if dimensions is not None:
            clauses.append("dimensions = ?")
            params.append(int(dimensions))
        scope_list = (
            [normalize_scope(scope) for scope in scopes] if scopes is not None else None
        )
        if scope_list is not None:
            if not scope_list:
                return
            placeholders = ",".join("?" for _ in scope_list)
            clauses.append(f"scope IN ({placeholders})")
            params.extend(scope_list)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        suffix = ""
        if limit is not None:
            suffix = " LIMIT ?"
            params.append(int(limit))
        cursor = self._conn.cursor()
        cursor.row_factory = None
        cursor.execute(
            f"""
            SELECT id, scope, dimensions, embedding
            FROM {QUERY_ANCHOR_TABLE}
            {where}
            {suffix}
            """,
            params,
        )
        while True:
            batch = cursor.fetchmany(512)
            if not batch:
                return
            for anchor_id, scope, dimensions, blob in batch:
                yield str(anchor_id), str(scope), int(dimensions), memoryview(blob)

    def upsert_query_anchor_edge(
        self,
        anchor_id: str,
        target_id: str,
        *,
        weight: float,
        now: str | None = None,
    ) -> QueryAnchorEdge:
        """Accumulate weight onto an anchor -> node edge; never clobber it.

        Anchor weight adds, saturating at 1.0, while structural contributions
        use a maximum. Grounding the same question on the same node again is
        new evidence. ``hits`` keeps counting after weight saturates.
        """

        if not self._anchor_tables_present():
            raise RuntimeError("query anchor tables are absent (pre-v7 database)")
        stamp = now or _utc_now()
        increment = min(1.0, max(0.0, float(weight)))
        with self.transaction():
            self._conn.execute(
                f"""
                INSERT INTO {QUERY_ANCHOR_EDGE_TABLE} (
                    anchor_id, target_id, weight, hits, created_at, updated_at
                )
                VALUES (?, ?, ?, 1, ?, ?)
                ON CONFLICT(anchor_id, target_id) DO UPDATE SET
                    weight = MIN(1.0, {QUERY_ANCHOR_EDGE_TABLE}.weight + excluded.weight),
                    hits = {QUERY_ANCHOR_EDGE_TABLE}.hits + 1,
                    updated_at = excluded.updated_at
                """,
                (str(anchor_id), str(target_id), increment, stamp, stamp),
            )
        row = self._conn.execute(
            f"""
            SELECT * FROM {QUERY_ANCHOR_EDGE_TABLE}
            WHERE anchor_id = ? AND target_id = ?
            """,
            (str(anchor_id), str(target_id)),
        ).fetchone()
        return _query_anchor_edge_from_row(row)

    def list_query_anchor_edges(
        self,
        *,
        anchor_id: str | None = None,
        target_id: str | None = None,
        active_targets_only: bool = False,
    ) -> list[QueryAnchorEdge]:
        """List anchor edges, heaviest first.

        ``active_targets_only`` joins ``nodes`` and drops soft-deleted targets:
        node deletion is soft everywhere in this store, so an edge row
        outliving its target's usefulness is normal and liveness has to be
        asked for rather than assumed.
        """

        if not self._anchor_tables_present():
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if anchor_id is not None:
            clauses.append("e.anchor_id = ?")
            params.append(str(anchor_id))
        if target_id is not None:
            clauses.append("e.target_id = ?")
            params.append(str(target_id))
        join = ""
        if active_targets_only:
            join = "JOIN nodes n ON n.id = e.target_id"
            clauses.append("n.decayed = 0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"""
            SELECT e.* FROM {QUERY_ANCHOR_EDGE_TABLE} e
            {join}
            {where}
            ORDER BY e.weight DESC, e.target_id ASC
            """,
            params,
        ).fetchall()
        return [_query_anchor_edge_from_row(row) for row in rows]

    def get_query_anchor_links(self, target_ids: Sequence[str]) -> list[QueryAnchorLink]:
        """Read edge weights and anchor label state for selected targets.

        Keep missing and decayed anchors visible to the caller's policy. The
        left join also retains dangling edges so their weights are validated.
        """

        links: list[QueryAnchorLink] = []
        if not target_ids or not self._anchor_tables_present():
            return links
        batch_size = min(999, self._conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER))
        for batch in _node_read_batches(map(str, target_ids), batch_size=batch_size):
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"""SELECT e.target_id, e.anchor_id, e.weight, a.id, a.query, a.decayed
                FROM {QUERY_ANCHOR_EDGE_TABLE} e
                LEFT JOIN {QUERY_ANCHOR_TABLE} a ON a.id = e.anchor_id
                WHERE e.target_id IN ({placeholders})
                ORDER BY e.weight DESC, e.target_id ASC""",
                batch,
            )
            links.extend(
                QueryAnchorLink(
                    target_id=str(row["target_id"]), anchor_id=str(row["anchor_id"]),
                    weight=float(row["weight"]),
                    query=None if row["id"] is None else str(row["query"]),
                    decayed=bool(row["decayed"]),
                )
                for row in rows
            )
        return links

    def count_query_anchors(self, *, include_decayed: bool = True) -> int:
        if not self._anchor_tables_present():
            return 0
        where = "" if include_decayed else "WHERE decayed = 0"
        row = self._conn.execute(
            f"SELECT COUNT(*) AS count FROM {QUERY_ANCHOR_TABLE} {where}"
        ).fetchone()
        return int(row["count"])

    def count_query_anchor_edges(self) -> int:
        if not self._anchor_tables_present():
            return 0
        row = self._conn.execute(
            f"SELECT COUNT(*) AS count FROM {QUERY_ANCHOR_EDGE_TABLE}"
        ).fetchone()
        return int(row["count"])

    def repoint_query_anchor_edges(
        self, from_target: str, to_target: str, *, now: str | None = None
    ) -> int:
        """Move every anchor edge on ``from_target`` onto ``to_target``.

        Merging, not overwriting: an anchor that already points at
        ``to_target`` keeps the heavier claim plus the arriving one, capped at
        1.0, and its ``hits`` add — the same accumulate-never-clobber contract
        as :meth:`upsert_query_anchor_edge`, which is why a plain
        ``UPDATE ... SET target_id = ?`` is wrong here: it raises on the
        PRIMARY KEY collision, and ``UPDATE OR REPLACE`` would silently discard
        one of the two edges' learned weight.

        Runs no transaction of its own so it can join the one that is writing
        the ``supersedes`` edge; callers outside such a transaction should wrap
        it in ``with store.connection:``. Returns the number of edge rows
        moved. No identity, existence, or cycle checking — that is policy and
        lives in :mod:`living_memory.query_anchors`.
        """

        if not self._anchor_tables_present():
            return 0
        source = str(from_target)
        destination = str(to_target)
        if source == destination:
            return 0
        rows = self._conn.execute(
            f"""
            SELECT anchor_id, weight, hits, created_at
            FROM {QUERY_ANCHOR_EDGE_TABLE}
            WHERE target_id = ?
            """,
            (source,),
        ).fetchall()
        if not rows:
            return 0
        stamp = now or _utc_now()
        for row in rows:
            self._conn.execute(
                f"""
                INSERT INTO {QUERY_ANCHOR_EDGE_TABLE} (
                    anchor_id, target_id, weight, hits, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(anchor_id, target_id) DO UPDATE SET
                    weight = MIN(1.0, {QUERY_ANCHOR_EDGE_TABLE}.weight + excluded.weight),
                    hits = {QUERY_ANCHOR_EDGE_TABLE}.hits + excluded.hits,
                    updated_at = excluded.updated_at
                """,
                (
                    str(row["anchor_id"]),
                    destination,
                    float(row["weight"]),
                    int(row["hits"]),
                    str(row["created_at"]),
                    stamp,
                ),
            )
        self._conn.execute(
            f"DELETE FROM {QUERY_ANCHOR_EDGE_TABLE} WHERE target_id = ?",
            (source,),
        )
        return len(rows)

    def query_anchor_revision(self) -> int | None:
        """Transactional revision of anchor ids, scopes, widths and vectors.

        SQLite triggers maintain it across connections and maintenance SQL,
        including decay and revival between reads. Reinforcement and edge or
        feedback bookkeeping leave it unchanged unless they revive an anchor.
        The indexed metadata lookup does not scan the anchor corpus.

        Unmigrated read-only databases return ``None``: their vectors may be
        read, but cannot be cached under a stable revision.
        """

        if not self._anchor_tables_present():
            return None
        row = self._conn.execute(
            "SELECT value FROM metadata WHERE key = 'query_anchor_revision'"
        ).fetchone()
        return int(row[0]) if row is not None else None

    def list_anchor_edge_targets(self) -> list[str]:
        """Every distinct node id that some anchor edge points at."""

        if not self._anchor_tables_present():
            return []
        rows = self._conn.execute(
            f"SELECT DISTINCT target_id FROM {QUERY_ANCHOR_EDGE_TABLE} ORDER BY target_id"
        ).fetchall()
        return [str(row["target_id"]) for row in rows]

    def _initialize_chunk_revisions(self) -> None:
        """Install scope revisions atomically, including existing membership."""

        with self.transaction(immediate=True):
            row = self._conn.execute(
                "SELECT value FROM metadata WHERE key = ?",
                (_CHUNK_SCOPE_REVISION_FORMAT,),
            ).fetchone()
            if row is not None and row[0] == "1":
                return
            for sql in (*_CHUNK_CORPUS_TRIGGERS, *_CHUNK_MEMBERSHIP_TRIGGERS):
                name = sql.split()[2]
                self._conn.execute(f"DROP TRIGGER IF EXISTS {name}")
                self._conn.execute(sql)
            self._conn.execute(
                "INSERT OR IGNORE INTO metadata (key, value) "
                "SELECT DISTINCT ? || scope, '0' FROM nodes "
                "WHERE EXISTS (SELECT 1 FROM node_chunk_embeddings WHERE node_id = nodes.id)",
                (_CHUNK_SCOPE_REVISION_PREFIX,),
            )
            self._conn.execute("DELETE FROM metadata WHERE key = 'chunk_corpus_revision'")
            self._conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, '1')",
                (_CHUNK_SCOPE_REVISION_FORMAT,),
            )

    def _initialize_schema(self) -> None:
        with self._conn:
            self._migrate_pre_v3_schema()
            self._migrate_pre_v4_schema()
            self._migrate_pre_v6_schema()
            self._migrate_pre_v7_schema()
            self._migrate_recall_map_column()
            self._migrate_recall_events_task_pattern_column()
            self._migrate_task_fts()
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
                    task,
                    level UNINDEXED,
                    scope UNINDEXED,
                    tokenize = 'unicode61'
                );

                CREATE TRIGGER IF NOT EXISTS nodes_fts_insert
                AFTER INSERT ON nodes
                BEGIN
                    INSERT INTO nodes_fts(rowid, node_id, content, task, level, scope)
                    VALUES (new.rowid, new.id, new.content, new.task, new.level, new.scope);
                END;

                CREATE TRIGGER IF NOT EXISTS nodes_fts_delete
                AFTER DELETE ON nodes
                BEGIN
                    DELETE FROM nodes_fts WHERE rowid = old.rowid;
                END;

                CREATE TRIGGER IF NOT EXISTS nodes_fts_update
                AFTER UPDATE OF content, task, level, scope ON nodes
                BEGIN
                    DELETE FROM nodes_fts WHERE rowid = old.rowid;
                    INSERT INTO nodes_fts(rowid, node_id, content, task, level, scope)
                    VALUES (new.rowid, new.id, new.content, new.task, new.level, new.scope);
                END;

                -- Document-frequency reader over the index the triggers above
                -- maintain, for c-TF-IDF labeling (term_document_frequencies
                -- and fts_document_count). 'row' type: one row per term with
                -- doc = how many indexed documents contain a term per column.
                -- The labeler reads content only; task is retrieval evidence.
                -- fts5vocab
                -- stores nothing — it is a stateless view of the FTS index —
                -- so this CREATE on every open *is* the whole migration for
                -- pre-existing databases (the _RECALL_ATTESTATION_SCHEMA_SQL
                -- rationale), there is no data shape a SCHEMA_VERSION bump
                -- could protect, and the statement is additive: no existing
                -- table's DDL changes by a byte.
                CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts_vocab
                USING fts5vocab('nodes_fts', 'col');

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

                -- Per-chunk embeddings (schema v6): many float32 little-endian
                -- BLOB vectors per node, replacing the single JSON-text vector
                -- in nodes.embedding. `dimensions` is recorded rather than
                -- inferred from the blob length, and `content_fingerprint` is
                -- the parent's fingerprint at embed time so an edit to the
                -- parent's content is detectable without re-embedding.
                -- Node deletion is soft (soft_delete_node only sets decayed=1),
                -- so a chunk row outliving its node's usefulness is normal:
                -- readers must join nodes and filter decayed = 0 rather than
                -- treat row presence as liveness.
                CREATE TABLE IF NOT EXISTS node_chunk_embeddings (
                    id TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL REFERENCES nodes(id),
                    chunk_index INTEGER NOT NULL,
                    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
                    embedding BLOB NOT NULL,
                    token_start INTEGER NOT NULL,
                    token_end INTEGER NOT NULL,
                    content_fingerprint TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    char_start INTEGER,
                    char_end INTEGER,
                    UNIQUE(node_id, chunk_index)
                );

                -- `recall_map` is the recall map served with this delivery,
                -- verbatim (compact JSON: cluster labels, counts, medoid node
                -- ids, ask hints), or NULL when no map was built: what the
                -- agent was actually shown, recorded at delivery time so a
                -- later pass can ask whether the next recall followed one of
                -- its clusters.
                --
                -- `task_pattern` is the ambient recurring-work label, lifted
                -- out of `ambient_context` into a column of its own for the
                -- same reason `task` is: it is the label the recall map keys
                -- its cache on (`recall_map.cache_key`, where a task_pattern
                -- wins over a task), and a key nothing can filter on is a key
                -- whose delivery history is always empty. That was the
                -- documented debt behind a curtail rule that could never fire
                -- for a per-turn `task` — see docs/recall-map.md section 9.
                --
                -- Both are last in DDL order, in this order, because
                -- `_migrate_recall_map_column` and
                -- `_migrate_recall_events_task_pattern_column` append them
                -- with ALTER TABLE in that sequence, and a migrated file and a
                -- fresh one must end up with the same column order. Their
                -- comments live out here rather than beside the columns:
                -- SQLite reconstructs this DDL text on ALTER TABLE ... DROP
                -- COLUMN, and a comment attached to a dropped column leaves
                -- the stored statement unparseable.
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
                    transport_session_id TEXT,
                    feedback_applied INTEGER NOT NULL DEFAULT 0 CHECK (feedback_applied IN (0, 1)),
                    feedback_trace_id TEXT REFERENCES nodes(id),
                    feedback_applied_at TEXT,
                    created_at TEXT NOT NULL,
                    recall_map TEXT,
                    task_pattern TEXT
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

                -- Write cadence includes replaced and decayed traces. Keep
                -- its per-scope count covering, independent of active indexes.
                CREATE INDEX IF NOT EXISTS idx_nodes_trace_scope
                    ON nodes(level, scope)
                    WHERE level = 'trace';

                CREATE INDEX IF NOT EXISTS idx_nodes_concepts_content
                    ON nodes(content)
                    WHERE level IN ('concept', 'schema') AND decayed = 0;

                CREATE INDEX IF NOT EXISTS idx_nodes_concepts_scope_content
                    ON nodes(scope, content)
                    WHERE level IN ('concept', 'schema') AND decayed = 0;

                -- Source identities cover pending-chunk discovery for broad,
                -- scope-only and level-only reads. The scope-leading index
                -- also drives active chunk scans without reading node bodies.
                CREATE INDEX IF NOT EXISTS idx_nodes_active_scope_source
                    ON nodes(scope, level, id, content_fingerprint)
                    WHERE decayed = 0;

                CREATE INDEX IF NOT EXISTS idx_nodes_active_level_source
                    ON nodes(level, scope, id, content_fingerprint)
                    WHERE decayed = 0;

                DROP INDEX IF EXISTS idx_nodes_active_scope_level;
                DROP INDEX IF EXISTS idx_nodes_level_scope_active;

                -- Freshness probes must not visit rows containing vector BLOBs.
                -- UNIQUE(node_id, chunk_index) separately owns ordinal identity.
                CREATE INDEX IF NOT EXISTS idx_chunk_source
                    ON node_chunk_embeddings(node_id, content_fingerprint);

                CREATE INDEX IF NOT EXISTS idx_nodes_dedup
                    ON nodes(level, scope, content_fingerprint)
                    WHERE decayed = 0 AND content_fingerprint IS NOT NULL;

                CREATE INDEX IF NOT EXISTS idx_connections_source_type
                    ON connections(source_id, type);

                CREATE INDEX IF NOT EXISTS idx_connections_target_type
                    ON connections(target_id, type);

                -- Covering index for type-wide graph readers.
                CREATE INDEX IF NOT EXISTS idx_connections_type
                    ON connections(type, source_id, target_id);

                -- Write-path typed-edge derivation (edge_derivation.py):
                -- bounded candidate lookups on memory_remember. The commit
                -- expression must match _commit_group_candidates verbatim for
                -- the probe to be index-served; the files index bounds the
                -- R4b json_each scan to active file-citing rows.
                CREATE INDEX IF NOT EXISTS idx_nodes_commit_prefix
                    ON nodes(substr(lower(trim(json_extract(context, '$.commit'))), 1, 7))
                    WHERE json_extract(context, '$.commit') IS NOT NULL AND decayed = 0;

                CREATE INDEX IF NOT EXISTS idx_nodes_files_present
                    ON nodes(id)
                    WHERE json_extract(context, '$.files') IS NOT NULL AND decayed = 0;

                CREATE INDEX IF NOT EXISTS idx_recall_events_scope_pending_created
                    ON recall_events(scope, feedback_applied, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_recall_events_session_pending_created
                    ON recall_events(session_id, feedback_applied, created_at DESC);

                -- Pending feedback requires a shared work or transport identity.
                -- COALESCE also serves compatibility checks on the other fields.
                CREATE INDEX IF NOT EXISTS idx_recall_pending_session
                    ON recall_events(COALESCE(session_id, ''), created_at DESC)
                    WHERE feedback_applied = 0;

                CREATE INDEX IF NOT EXISTS idx_recall_pending_task_session
                    ON recall_events(COALESCE(task, ''), COALESCE(session_id, ''), created_at DESC)
                    WHERE feedback_applied = 0;

                CREATE INDEX IF NOT EXISTS idx_recall_pending_transport
                    ON recall_events(COALESCE(transport_session_id, ''), created_at DESC)
                    WHERE feedback_applied = 0;

                CREATE INDEX IF NOT EXISTS idx_recall_pending_transport_task_session
                    ON recall_events(COALESCE(transport_session_id, ''),
                        COALESCE(task, ''), COALESCE(session_id, ''), created_at DESC)
                    WHERE feedback_applied = 0;

                CREATE INDEX IF NOT EXISTS idx_recall_pending_transport_session
                    ON recall_events(COALESCE(transport_session_id, ''),
                        COALESCE(session_id, ''), created_at DESC)
                    WHERE feedback_applied = 0;

                DROP INDEX IF EXISTS idx_recall_pending_agent;

                CREATE INDEX IF NOT EXISTS idx_recall_events_requested_scope_pending_created
                    ON recall_events(requested_scope, feedback_applied, created_at DESC);

                -- Session content delivery history: equality probe
                -- on transport_session_id plus the most-recent-events window,
                -- instead of scanning every session's events. Runs after
                -- _migrate_pre_v4_schema, so the column exists on legacy DBs.
                CREATE INDEX IF NOT EXISTS idx_recall_events_transport_created
                    ON recall_events(transport_session_id, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_recall_events_session_created
                    ON recall_events(session_id, created_at DESC);

                -- The curtail rule's delivery window, keyed the way the map
                -- itself is keyed: equality on (scope, task_pattern), so
                -- recent_recall_map_history sorts two bounded ranges of this
                -- index instead of the scope's whole event partition. Runs
                -- after the task_pattern column migration, so the column
                -- exists on legacy DBs.
                --
                -- Deliberately NOT partial on `recall_map IS NOT NULL`, though
                -- every consumer of this read carries that predicate: a
                -- partial index naming a column makes `ALTER TABLE ... DROP
                -- COLUMN` on it fail outright, and the recall_map column's own
                -- DDL comment above is written the way it is precisely to keep
                -- that operation available. `recall_map IS NOT NULL` stays a
                -- residual filter over the range this index already narrowed.
                CREATE INDEX IF NOT EXISTS idx_recall_events_scope_pattern_created
                    ON recall_events(scope, task_pattern, created_at DESC);
                """
            )
            self._initialize_chunk_revisions()
            # Query anchors (v7). Same script the pre-v7 migration runs, so a
            # fresh database and a migrated one end up with identical objects.
            self._conn.executescript(_QUERY_ANCHOR_SCHEMA_SQL)
            self._conn.executescript(_QUERY_ANCHOR_REVISION_SQL)
            self._anchor_tables_cache = None
            # Attestation ledger (v8). Runs after the main script because its
            # foreign keys name recall_events and nodes, and unconditionally on
            # every open because that *is* the pre-v8 migration — see the note
            # on _RECALL_ATTESTATION_SCHEMA_SQL for why no _migrate_pre_v8_schema
            # exists.
            self._conn.executescript(_RECALL_ATTESTATION_SCHEMA_SQL)
            # Transcript-grounding verdict ledger. Additive like the
            # attestation ledger above and created the same way — on every
            # open, because that *is* the whole migration; see the note on
            # TRANSCRIPT_GROUNDING_SCHEMA_SQL. Offline import only: no live
            # read or write path touches it.
            self._conn.executescript(TRANSCRIPT_GROUNDING_SCHEMA_SQL)
            # Recall credit ledger. Additive and created the same way, on
            # every open; see the note on _RECALL_CREDIT_LEDGER_SCHEMA_SQL.
            self._conn.executescript(_RECALL_CREDIT_LEDGER_SCHEMA_SQL)
            # Explicit credit and the explicit-mark audit: additive and
            # created the same way; see the notes on the two DDL strings.
            self._conn.executescript(_RECALL_EXPLICIT_CREDIT_SCHEMA_SQL)
            self._conn.executescript(_RECALL_FEEDBACK_MARKS_SCHEMA_SQL)
            self._conn.executescript(_CONSOLIDATION_EMBEDDING_SCHEMA_SQL)
            # Frozen relevance history. The schema is additive; the
            # idempotent reconstruction below is what populates it for a live
            # database whose recall_events predate this release.
            self._conn.executescript(_RECALL_DELIVERY_HISTORY_SCHEMA_SQL)
            # Must precede the reconstruction: the script above only creates
            # tables that are absent, so a database that already carries the
            # ledger reaches the rebuild without the tri-state column the
            # rebuild writes.
            self._migrate_recall_delivery_history_lookup_column()
            self._backfill_recall_delivery_history()
            self._create_legacy_embedding_index()
            self._backfill_missing_content_fingerprints()
            self._migrate_chunk_passage_columns()
            self._invalidate_derived_proxy_chunks()
            self._backfill_recall_events_task_pattern()
            self._conn.execute(
                """
                INSERT INTO metadata (key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )

    def _migrate_chunk_passage_columns(self) -> None:
        """Add source coordinates without rewriting or re-encoding old chunks."""
        columns = {row[1] for row in self._conn.execute(
            f"PRAGMA table_info({CHUNK_EMBEDDING_TABLE})"
        )}
        for column in ("char_start", "char_end"):
            if column not in columns:
                self._conn.execute(f"ALTER TABLE {CHUNK_EMBEDDING_TABLE} ADD COLUMN {column} INTEGER")

    def _invalidate_derived_proxy_chunks(self) -> None:
        """Queue single-window materializer chunks for ordinary content encoding.

        Multi-window writes already encoded their own text. Only the two
        representative-bearing materializers need this one-time invalidation;
        node content, representatives, sources and validity remain untouched.
        """

        key = "derived_content_chunks"
        if self._conn.execute("SELECT 1 FROM metadata WHERE key = ?", (key,)).fetchone():
            return
        self._conn.execute(
            f"""
            DELETE FROM {CHUNK_EMBEDDING_TABLE} WHERE node_id IN (
                SELECT n.id FROM nodes n
                WHERE n.level = 'concept' AND (
                    json_extract(n.provenance, '$.strategy') IN (
                        'token-jaccard', 'embedding-cosine', 'cross-scope-promotion'
                    ) OR (json_extract(n.provenance, '$.strategy') IS NULL
                          AND json_type(n.provenance, '$.cluster_key') IS NOT NULL)
                ) AND (SELECT COUNT(*) FROM {CHUNK_EMBEDDING_TABLE} c
                       WHERE c.node_id = n.id) = 1
            )
            """
        )
        self._conn.execute("INSERT INTO metadata (key, value) VALUES (?, '1')", (key,))

    def _migrate_task_fts(self) -> None:
        """Rebuild the derived FTS index once when opening a pre-task database.

        FTS5 cannot add a column in place. The nodes table remains the source
        of truth, so this transaction changes no stored memory or node ids.
        """

        old_fts = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'nodes_fts'"
        ).fetchone()
        if old_fts is None:
            return
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(nodes_fts)")
        }
        if "task" in columns:
            return
        self._conn.executescript(
            """
            BEGIN IMMEDIATE;
            DROP TRIGGER IF EXISTS nodes_fts_insert;
            DROP TRIGGER IF EXISTS nodes_fts_delete;
            DROP TRIGGER IF EXISTS nodes_fts_update;
            DROP TABLE IF EXISTS nodes_fts_vocab;
            DROP TABLE nodes_fts;
            CREATE VIRTUAL TABLE nodes_fts USING fts5(
                node_id UNINDEXED, content, task, level UNINDEXED,
                scope UNINDEXED, tokenize = 'unicode61'
            );
            INSERT INTO nodes_fts(rowid, node_id, content, task, level, scope)
            SELECT rowid, id, content, task, level, scope FROM nodes;
            CREATE TRIGGER nodes_fts_insert AFTER INSERT ON nodes BEGIN
                INSERT INTO nodes_fts(rowid, node_id, content, task, level, scope)
                VALUES (new.rowid, new.id, new.content, new.task, new.level, new.scope);
            END;
            CREATE TRIGGER nodes_fts_delete AFTER DELETE ON nodes BEGIN
                DELETE FROM nodes_fts WHERE rowid = old.rowid;
            END;
            CREATE TRIGGER nodes_fts_update
            AFTER UPDATE OF content, task, level, scope ON nodes BEGIN
                DELETE FROM nodes_fts WHERE rowid = old.rowid;
                INSERT INTO nodes_fts(rowid, node_id, content, task, level, scope)
                VALUES (new.rowid, new.id, new.content, new.task, new.level, new.scope);
            END;
            CREATE VIRTUAL TABLE nodes_fts_vocab USING fts5vocab('nodes_fts', 'col');
            COMMIT;
            """
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

    def _migrate_pre_v4_schema(self) -> None:
        """Add transport_session_id to recall_events for DBs created at schema_version <= 3."""

        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='recall_events'"
        ).fetchone()
        if row is None:
            return
        columns = {
            r["name"]
            for r in self._conn.execute("PRAGMA table_info(recall_events)").fetchall()
        }
        if "transport_session_id" not in columns:
            self._conn.execute(
                "ALTER TABLE recall_events ADD COLUMN transport_session_id TEXT"
            )

    def _migrate_pre_v6_schema(self) -> None:
        """Reconcile the legacy embedding index for DBs created at schema_version <= 5.

        The ``node_chunk_embeddings`` table itself is created by the
        ``CREATE TABLE IF NOT EXISTS`` script below — a pre-v6 database and a
        fresh one need the same DDL, so duplicating it here would be two copies
        to keep in step.

        What the script cannot do is survive the *other* half of v6:
        ``idx_nodes_embedded_active_scope`` is partial on
        ``embedding IS NOT NULL``, so once an operator runs
        :meth:`drop_node_embedding_column` a plain ``CREATE INDEX IF NOT
        EXISTS`` for it raises ``no such column: embedding`` and the database
        can never be reopened. Creating that index moved to
        ``_create_legacy_embedding_index``, which runs only while the column is
        there; this migration drops the stale index object if a database still
        carries one without the column. Self-guarding and idempotent, like its
        siblings.
        """

        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='nodes'"
        ).fetchone()
        if row is None:
            return
        if self._node_embedding_column_present():
            return
        self._conn.execute("DROP INDEX IF EXISTS idx_nodes_embedded_active_scope")

    def _migrate_pre_v7_schema(self) -> None:
        """Add the query-anchor tables for DBs created at schema_version <= 6.

        Self-guarding and idempotent, in the shape of
        :meth:`_migrate_pre_v5_schema`: probe ``sqlite_master``, read
        ``PRAGMA table_info``, and only then run ``CREATE TABLE IF NOT
        EXISTS`` / ``ALTER TABLE ... ADD COLUMN``.

        Strictly additive by construction. ``nodes.level`` and
        ``connections.type`` carry baked-in CHECK constraints on a 500 MB live
        database, so an anchor could not be a new node level nor an anchor edge
        a new connection type without a full table rebuild; two new tables cost
        one ``CREATE`` each and leave both existing DDL strings byte-identical.

        The ``CREATE`` half is shared with :meth:`_initialize_schema` through
        ``_QUERY_ANCHOR_SCHEMA_SQL`` rather than copied here — see the note on
        that constant, and :meth:`_migrate_pre_v6_schema` for why a second copy
        of a table's DDL inside a migration is the wrong shape. What is local
        to this method is the ``ALTER TABLE`` half: a database that already
        carries ``query_anchors`` from a partial earlier rollout would be left
        untouched by ``CREATE TABLE IF NOT EXISTS``, so any column the current
        DDL adds is added explicitly. Only NULLable or defaulted columns can be
        added this way, which is why every optional anchor column has a
        default.
        """

        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (QUERY_ANCHOR_TABLE,),
        ).fetchone()
        edge_row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (QUERY_ANCHOR_EDGE_TABLE,),
        ).fetchone()
        if row is None or edge_row is None:
            self._conn.executescript(_QUERY_ANCHOR_SCHEMA_SQL)
            self._anchor_tables_cache = None
            return

        columns = {
            str(r["name"])
            for r in self._conn.execute(
                f"PRAGMA table_info({QUERY_ANCHOR_TABLE})"
            ).fetchall()
        }
        for name, declaration in _QUERY_ANCHOR_COLUMNS:
            if name not in columns:
                self._conn.execute(
                    f"ALTER TABLE {QUERY_ANCHOR_TABLE} ADD COLUMN {name} {declaration}"
                )

        edge_columns = {
            str(r["name"])
            for r in self._conn.execute(
                f"PRAGMA table_info({QUERY_ANCHOR_EDGE_TABLE})"
            ).fetchall()
        }
        for name, declaration in _QUERY_ANCHOR_EDGE_COLUMNS:
            if name not in edge_columns:
                self._conn.execute(
                    f"ALTER TABLE {QUERY_ANCHOR_EDGE_TABLE} "
                    f"ADD COLUMN {name} {declaration}"
                )

    def _migrate_recall_map_column(self) -> None:
        """Add ``recall_events.recall_map`` to databases created without it.

        Self-guarding and idempotent, in the shape of
        :meth:`_migrate_pre_v4_schema` and :meth:`_migrate_pre_v5_schema`:
        probe ``sqlite_master``, read ``PRAGMA table_info``, ``ALTER TABLE ...
        ADD COLUMN`` only what is missing. Reopening a migrated file is a no-op
        and the column keeps whatever it holds.

        No ``SCHEMA_VERSION`` bump goes with it, and not for want of one: every
        ``_migrate_*`` method here runs unconditionally on *every* open and
        decides for itself, so the stamped version gates nothing and a bump
        would protect no data shape. The change is additive — one nullable
        column on one table, no existing column's DDL touched — so a database
        opened by an older build simply carries a column that build never
        reads. The naming break with the ``_migrate_pre_vN_schema`` siblings is
        deliberate: this migration belongs to no version boundary.
        """

        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='recall_events'"
        ).fetchone()
        if row is None:
            return
        columns = {
            r["name"]
            for r in self._conn.execute("PRAGMA table_info(recall_events)").fetchall()
        }
        if "recall_map" not in columns:
            self._conn.execute("ALTER TABLE recall_events ADD COLUMN recall_map TEXT")

    def _migrate_recall_events_task_pattern_column(self) -> None:
        """Add ``recall_events.task_pattern`` to databases created without it.

        Additive and idempotent, in the shape of
        :meth:`_migrate_recall_map_column`, and called immediately after it so
        that a migrated file and a fresh one agree on column order: both end
        ``… created_at, recall_map, task_pattern``.

        The column exists because ``task_pattern`` is the label the recall map
        keys its cache on and, until now, the one key component that was never
        a column — so ``recent_recall_map_history`` could only filter by
        ``task``, and a client sending a per-turn ``task`` (an AE chat's
        ``chat:<id>/turn-<N>``) had an empty delivery history under every key
        it ever used. Existing rows arrive NULL and are filled by
        :meth:`_backfill_recall_events_task_pattern` from the ambient JSON that
        already carried the value; rows that never carried one stay NULL, which
        is what makes the ``task`` fallback in
        :meth:`recent_recall_map_history` mean something.
        """

        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='recall_events'"
        ).fetchone()
        if row is None:
            return
        columns = {
            r["name"]
            for r in self._conn.execute("PRAGMA table_info(recall_events)").fetchall()
        }
        if "task_pattern" not in columns:
            self._conn.execute("ALTER TABLE recall_events ADD COLUMN task_pattern TEXT")

    def _backfill_recall_events_task_pattern(self) -> None:
        """Lift ``ambient_context.task_pattern`` into the column for old rows.

        The value was never lost — every recall event stores its whole ambient
        context as JSON — so the column starts out as complete as the history
        it describes, rather than beginning at this release and leaving the
        curtail rule blind to everything before it.

        Guarded by a probe carrying the *same* predicate as the update, so the
        pass is genuinely terminal: after it runs, nothing matches and a reopen
        stops at the ``LIKE`` on a single column instead of parsing JSON for
        every event in the table. Only string values are lifted, which is the
        shape the write path's ``_optional_str`` produces; a caller that put an
        object or a number under that key keeps it in the JSON and gets no
        column. The ``json_valid`` CASE is not decoration — ``ambient_context``
        is declared TEXT with no JSON constraint, and ``json_type`` raises on a
        malformed document rather than returning NULL, which would make one bad
        legacy row fatal to opening the database.
        """

        pending = self._conn.execute(
            f"SELECT 1 FROM recall_events WHERE {_TASK_PATTERN_BACKFILL_PREDICATE} LIMIT 1"
        ).fetchone()
        if pending is None:
            return
        self._conn.execute(
            f"""
            UPDATE recall_events
            SET task_pattern = json_extract({_AMBIENT_JSON_OR_EMPTY}, '$.task_pattern')
            WHERE {_TASK_PATTERN_BACKFILL_PREDICATE}
            """
        )

    def _migrate_recall_delivery_history_lookup_column(self) -> None:
        """Add ``lookup_consumed`` to a ledger table that predates it.

        Additive and idempotent, in the shape ``_migrate_recall_map_column``
        uses.  The column arrives NULL on every existing row, which is the
        correct reading and not merely a convenient one — those windows closed
        before any lookup could be recorded.  The format bump then re-derives
        them through the same rule rather than leaving the value to a default.
        """

        table = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (RECALL_DELIVERY_HISTORY_TABLE,),
        ).fetchone()
        if table is None:  # pragma: no cover - the schema script just ran
            return
        columns = {
            row["name"]
            for row in self._conn.execute(
                f"PRAGMA table_info({RECALL_DELIVERY_HISTORY_TABLE})"
            ).fetchall()
        }
        if "lookup_consumed" in columns:
            return
        self._conn.execute(
            f"""
            ALTER TABLE {RECALL_DELIVERY_HISTORY_TABLE}
            ADD COLUMN lookup_consumed INTEGER
                CHECK (lookup_consumed IS NULL OR lookup_consumed IN (0, 1))
            """
        )

    def _backfill_recall_delivery_history(self) -> None:
        """Idempotently reconstruct the frozen normalized history ledger.

        One state row is the commit marker.  A completed reconstruction and a
        recorded malformed-legacy verdict are both terminal and therefore
        cheap on every later open.  If the process dies before the marker is
        written, SQLite rolls the surrounding initialization transaction back;
        the next open starts the reconstruction again.

        ``recall_lookup_events`` is deliberately absent from the tables wiped
        below.  It is a primary record, not a projection of ``recall_events``,
        so a rebuild reads it and never rewrites it; wiping it would erase the
        observability epoch and turn every re-derived window unknown.
        """

        state = self._conn.execute(
            f"""
            SELECT format_version, complete, unavailable_reason
            FROM {RECALL_DELIVERY_HISTORY_STATE_TABLE}
            WHERE singleton = 1
            """
        ).fetchone()
        if (
            state is not None
            and int(state["format_version"]) == _RECALL_DELIVERY_HISTORY_FORMAT
            and (int(state["complete"]) or state["unavailable_reason"] is not None)
        ):
            return

        for table in (
            RECALL_DELIVERY_HISTORY_TABLE,
            RECALL_HISTORY_RESULT_TABLE,
            RECALL_HISTORY_EVENT_TABLE,
        ):
            self._conn.execute(f"DELETE FROM {table}")
        self._conn.execute(f"DELETE FROM {RECALL_DELIVERY_HISTORY_STATE_TABLE}")

        rows = self._conn.execute(
            """
            SELECT id, created_at, transport_session_id, scope, task,
                   results, recall_map
            FROM recall_events
            ORDER BY created_at, rowid
            """
        ).fetchall()
        for row in rows:
            occurred_at = _normalize_recall_history_instant(row["created_at"])
            results = _decode_recall_results(row["results"])
            map_ok, recall_map = _decode_recall_map(row["recall_map"])
            if occurred_at is None or results is None or not map_ok:
                for table in (
                    RECALL_DELIVERY_HISTORY_TABLE,
                    RECALL_HISTORY_RESULT_TABLE,
                    RECALL_HISTORY_EVENT_TABLE,
                ):
                    self._conn.execute(f"DELETE FROM {table}")
                reason = (
                    "malformed_legacy_timestamp"
                    if occurred_at is None
                    else "malformed_legacy_results"
                    if results is None
                    else "malformed_legacy_recall_map"
                )
                self._mark_recall_history_unavailable(reason)
                return
            self._record_recall_history_event(
                event_id=str(row["id"]),
                occurred_at=occurred_at,
                transport_session_id=_optional_str(row["transport_session_id"] or None),
                scope=str(row["scope"]),
                task=_optional_str(row["task"] or None),
                results=results,
                recall_map=recall_map,
            )

        self._conn.execute(
            f"""
            INSERT INTO {RECALL_DELIVERY_HISTORY_STATE_TABLE} (
                singleton, format_version, complete, unavailable_reason, updated_at
            ) VALUES (1, ?, 1, NULL, ?)
            ON CONFLICT(singleton) DO UPDATE SET
                format_version = excluded.format_version,
                complete = 1,
                unavailable_reason = NULL,
                updated_at = excluded.updated_at
            """,
            (_RECALL_DELIVERY_HISTORY_FORMAT, _utc_now()),
        )

    def _create_legacy_embedding_index(self) -> None:
        """Create the pre-chunk vector index, but only while its column exists."""

        if not self._node_embedding_column_present():
            return
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_nodes_embedded_active_scope
                ON nodes(level, scope)
                WHERE decayed = 0 AND embedding IS NOT NULL
            """
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

    def get_consolidation_embeddings(
        self, nodes: Sequence[Node], encoder: str
    ) -> dict[str, list[float]]:
        """Cached whole-content vectors for ``nodes`` under ``encoder``.

        A row counts only when its fingerprint matches the node's current
        content; anything else is a miss for the caller to encode.
        """

        wanted = {node.id: _content_fingerprint(node.content) for node in nodes}
        found: dict[str, list[float]] = {}
        ids = list(wanted)
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            rows = self._conn.execute(
                f"""
                SELECT node_id, content_fingerprint, dimensions, embedding
                FROM {CONSOLIDATION_EMBEDDING_TABLE}
                WHERE encoder = ? AND node_id IN ({",".join("?" * len(batch))})
                """,
                (encoder, *batch),
            ).fetchall()
            for row in rows:
                node_id = str(row["node_id"])
                if row["content_fingerprint"] != wanted.get(node_id):
                    continue
                values = array("d")
                values.frombytes(bytes(row["embedding"]))
                if sys.byteorder != "little":  # pragma: no cover - little-endian CI
                    values.byteswap()
                if len(values) == int(row["dimensions"]):
                    found[node_id] = values.tolist()
        return found

    def put_consolidation_embeddings(
        self, entries: Sequence[tuple[Node, Sequence[float]]], encoder: str
    ) -> None:
        """Upsert whole-content vectors computed by a consolidation pass."""

        rows = []
        now = _utc_now()
        for node, vector in entries:
            if not vector:
                continue
            values = array("d", (float(value) for value in vector))
            if sys.byteorder != "little":  # pragma: no cover - little-endian CI
                values.byteswap()
            rows.append(
                (
                    node.id,
                    _content_fingerprint(node.content),
                    encoder,
                    len(values),
                    values.tobytes(),
                    now,
                )
            )
        if not rows:
            return
        with self.transaction():
            self._conn.executemany(
                f"""
                INSERT INTO {CONSOLIDATION_EMBEDDING_TABLE}
                    (node_id, content_fingerprint, encoder, dimensions, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    content_fingerprint = excluded.content_fingerprint,
                    encoder = excluded.encoder,
                    dimensions = excluded.dimensions,
                    embedding = excluded.embedding,
                    created_at = excluded.created_at
                """,
                rows,
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
        with self.transaction():
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
        with self.transaction():
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


def _normalize_recall_history_instant(value: str | datetime | Any) -> str | None:
    """Canonical fixed-width UTC text for exact indexed time comparisons."""

    if isinstance(value, datetime):
        parsed = value
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        else:
            parsed = parsed.astimezone(UTC)
    else:
        parsed = parse_timestamp(value)
    if parsed is None:
        return None
    return parsed.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _shift_recall_history_hours(value: str, hours: int) -> str | None:
    parsed = parse_timestamp(value)
    if parsed is None:
        return None
    return (parsed + timedelta(hours=int(hours))).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )


def _decode_recall_results(value: Any) -> list[dict[str, Any]] | None:
    """Decode legacy results; ``None`` means the top-level data is malformed."""

    if isinstance(value, list):
        parsed = value
    else:
        try:
            parsed = json.loads(str(value or "[]"))
        except (TypeError, ValueError):
            return None
    if not isinstance(parsed, list):
        return None
    # The frozen evaluator ignores non-object entries rather than inventing an
    # identity for them; preserve that narrow fail-open behaviour.
    return [dict(item) for item in parsed if isinstance(item, Mapping)]


def _decode_recall_map(value: Any) -> tuple[bool, dict[str, Any] | None]:
    """Decode a stored map, distinguishing SQL NULL from malformed JSON."""

    if value is None or value == "":
        return True, None
    if isinstance(value, Mapping):
        return True, dict(value)
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return False, None
    if not isinstance(parsed, Mapping):
        return False, None
    return True, dict(parsed)


def _recall_result_node_ids(results: Sequence[Mapping[str, Any]]) -> list[str]:
    """The evaluator's result-id sequence, before tail slicing."""

    return [
        str(item["node_id"])
        for item in results
        if isinstance(item, Mapping) and item.get("node_id")
    ]


def _organic_tail_node_ids(result_ids: Sequence[str]) -> list[str]:
    """Frozen organic delivery: result-id ranks 3..8, de-duplicated there."""

    start = RECALL_DELIVERY_HISTORY_HEAD_CUT
    stop = start + RECALL_DELIVERY_HISTORY_ITEMS_PER_EVENT
    return list(dict.fromkeys(str(node_id) for node_id in result_ids[start:stop]))


def _recall_map_medoid_ids(recall_map: Mapping[str, Any] | None) -> list[str]:
    """Frozen map delivery: first six clusters' unique medoid node ids."""

    if not isinstance(recall_map, Mapping):
        return []
    clusters = recall_map.get("clusters")
    if not isinstance(clusters, list):
        return []
    node_ids: list[str] = []
    for cluster in clusters[:RECALL_DELIVERY_HISTORY_ITEMS_PER_EVENT]:
        if not isinstance(cluster, Mapping):
            continue
        medoid = cluster.get("medoid")
        node_id = (
            str(medoid.get("node_id") or "")
            if isinstance(medoid, Mapping)
            else ""
        )
        if node_id:
            node_ids.append(node_id)
    return list(dict.fromkeys(node_ids))


def _encode_recall_map(payload: Mapping[str, Any] | None) -> str | None:
    """Compact JSON for ``recall_events.recall_map``; nothing served stays NULL.

    An empty map and no map are the same fact — no clusters were shown — and
    both must read as NULL, so ``recall_map IS NOT NULL`` means "a map was
    delivered" for every consumer of :meth:`MemoryStore.recent_recall_map_history`.
    """

    if not payload:
        return None
    return _json_dumps(dict(payload))


def _json_loads(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


_FTS_QUERY_TERM_RE = re.compile(r"\w+\*?")
_FTS_QUERY_IDENTIFIER_RE = re.compile(r"\w+(?:[-/.]\w+)+")


def _fts_query(query: str) -> str:
    """Turn free text into an FTS5 MATCH expression of terms and identifiers.

    Every ``\\w+`` run becomes an exact quoted term, as it always has. A run
    immediately followed by ``*`` becomes a prefix term (``"миграц"*``) when
    :func:`living_memory.embeddings.fts_prefix_term_allowed` accepts it --
    an all-Cyrillic term while Cyrillic stemming is on. That is how the stems
    ``retrieval._expanded_query`` appends reach the unstemmed ``unicode61``
    index. For every other run the star is dropped, so Latin terms keep their
    exact form and ``LM_TOKENIZE_CYRILLIC_STEM=off`` yields the previous
    term expression byte for byte. Compound identifiers also supply a phrase:
    ``harbor-914-slate`` is more specific evidence than those three words in
    arbitrary positions. Keep the individual terms so partial content remains
    reachable; FTS5 scores phrase specificity in both content and task, with
    no task-equality override or extra search.
    """

    terms: list[str] = []
    for match in _FTS_QUERY_TERM_RE.findall(query):
        if match.endswith("*"):
            term = match[:-1]
            if fts_prefix_term_allowed(term):
                terms.append(f'"{term}"*')
                continue
            terms.append(f'"{term}"')
        else:
            terms.append(f'"{match}"')
    terms.extend(
        f'"{identifier}"'
        for identifier in dict.fromkeys(_FTS_QUERY_IDENTIFIER_RE.findall(query))
    )
    return " OR ".join(terms)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    raise ValueError("value must be a list")


# Models without the schema in view guess these names for the two fields;
# accepting them keeps the whole write from failing on a key spelling.
_ALTERNATIVE_APPROACH_KEYS = ("approach", "alternative", "option", "name")
_ALTERNATIVE_REASON_KEYS = (
    "rejected_because",
    "reason",
    "rejected",
    "why",
    "why_rejected",
    "rationale",
)


def _first_text(item: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_rejected_alternatives(
    alternatives: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in alternatives or ():
        if not isinstance(item, Mapping):
            raise ValueError("alternatives_considered items must be objects")
        approach = _first_text(item, _ALTERNATIVE_APPROACH_KEYS)
        reason = _first_text(item, _ALTERNATIVE_REASON_KEYS)
        if not approach:
            raise ValueError(
                "alternative approach must be non-empty: each alternatives_considered "
                'item is {"approach": ..., "rejected_because": ...}, got keys '
                f"{sorted(map(str, item))}"
            )
        if not reason:
            raise ValueError(
                "alternative rejected_because must be non-empty: each "
                'alternatives_considered item is {"approach": ..., "rejected_because": ...}, '
                f"got keys {sorted(map(str, item))}"
            )
        normalized.append({"approach": approach, "rejected_because": reason})
    return normalized


def _rejected_alternative_content(approach: str, reason: str) -> str:
    return f"Rejected alternative: {approach}\nRejected because: {reason}"


def _query_anchor_from_row(row: sqlite3.Row) -> QueryAnchor:
    dimensions = int(row["dimensions"])
    return QueryAnchor(
        id=str(row["id"]),
        scope=str(row["scope"]),
        query=str(row["query"]),
        fingerprint=str(row["fingerprint"]),
        dimensions=dimensions,
        embedding=tuple(unpack_chunk_embedding(row["embedding"], dimensions)),
        reinforcement_count=int(row["reinforcement_count"]),
        usefulness_score=float(row["usefulness_score"]),
        decayed=bool(row["decayed"]),
        decay_reason=_optional_str(row["decay_reason"]),
        first_seen=str(row["first_seen"]),
        last_matched_at=_optional_str(row["last_matched_at"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _query_anchor_edge_from_row(row: sqlite3.Row) -> QueryAnchorEdge:
    return QueryAnchorEdge(
        anchor_id=str(row["anchor_id"]),
        target_id=str(row["target_id"]),
        weight=float(row["weight"]),
        hits=int(row["hits"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _chunk_embedding_from_row(row: sqlite3.Row) -> NodeChunkEmbedding:
    dimensions = int(row["dimensions"])
    return NodeChunkEmbedding(
        id=str(row["id"]),
        node_id=str(row["node_id"]),
        chunk_index=int(row["chunk_index"]),
        dimensions=dimensions,
        embedding=unpack_chunk_embedding(row["embedding"], dimensions),
        token_start=int(row["token_start"]),
        token_end=int(row["token_end"]),
        content_fingerprint=str(row["content_fingerprint"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        char_start=_row_value(row, "char_start"),
        char_end=_row_value(row, "char_end"),
    )


def _row_value(row: sqlite3.Row, column: str, default: Any = None) -> Any:
    """Read a column that a given schema shape may not have.

    ``sqlite3.Row`` raises ``IndexError`` for an absent column, and after the
    operator drops ``nodes.embedding`` every ``SELECT *`` over nodes comes back
    without it.
    """

    try:
        return row[column]
    except IndexError:
        return default


def duplicate_replacements(connection: sqlite3.Connection, node_id: str) -> list[Node]:
    """Read exact duplicate successors without opening or migrating a store.

    Historical duplicate_content rows can have a procedure-key-only basis.
    Current bytes, level, scope and source identity must establish a true copy
    before bookkeeping can redirect an anchor. It never establishes correction.
    """
    return _duplicate_nodes(connection, node_id, successors=True)


def _duplicate_nodes(
    connection: sqlite3.Connection, node_id: str, *, successors: bool,
) -> list[Node]:
    endpoint, peer = ("target_id", "source_id") if successors else ("source_id", "target_id")
    live = "AND n.decayed = 0" if successors else ""
    rows = connection.execute(
        f"""
        SELECT n.* FROM connections c JOIN nodes n ON n.id = c.{peer}
        WHERE c.{endpoint} = ? AND c.{peer} <> ? {live}
          AND c.type IN ('related', 'supersedes')
          AND json_extract(c.metadata, '$.kind') = 'duplicate_content'
        ORDER BY c.weight DESC, c.created_at DESC
        """,
        (node_id, node_id),
    ).fetchall()
    if not rows:
        return []
    row = connection.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
    if row is None:
        return []
    original = _node_from_row(row)
    # Correction history changes validity, not the source of an exact copy.
    # Keep its lineage so a later teach can refresh owners of the prior copy.
    identity = content_identity(replace(original, corrections=[]))
    return [node for row in rows if (node := _node_from_row(row)).level == original.level
            and identity is not None
            and content_identity(replace(node, corrections=[])) == identity]


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
        embedding=_json_loads(_row_value(row, "embedding"), None),
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
        transport_session_id=row["transport_session_id"],
        feedback_applied=bool(row["feedback_applied"]),
        feedback_trace_id=row["feedback_trace_id"],
        feedback_applied_at=row["feedback_applied_at"],
        created_at=str(row["created_at"]),
    )


def _recall_attestation_from_row(row: sqlite3.Row) -> RecallAttestation:
    return RecallAttestation(
        id=str(row["id"]),
        recall_event_id=str(row["recall_event_id"]),
        evidence_sha256=str(row["evidence_sha256"]),
        evidence_items=int(row["evidence_items"]),
        evidence_chars=int(row["evidence_chars"]),
        min_containment=float(row["min_containment"]),
        containments=[dict(item) for item in _json_loads(row["containments"], [])],
        grounded_node_ids=[str(item) for item in _json_loads(row["grounded_node_ids"], [])],
        anchor_ids=[str(item) for item in _json_loads(row["anchor_ids"], [])],
        credited=bool(row["credited"]),
        closed=bool(row["closed"]),
        closed_by_attestation=bool(row["closed_by_attestation"]),
        feedback_trace_id=row["feedback_trace_id"],
        agent=row["agent"],
        task=row["task"],
        session_id=row["session_id"],
        transport_session_id=row["transport_session_id"],
        source_session_key=row["source_session_key"],
        created_at=str(row["created_at"]),
    )


def _recall_identity(context: Mapping[str, Any] | None) -> dict[str, str | None]:
    """Read the persisted caller fields, including the existing session alias."""
    context = context or {}
    return {
        "session_id": _optional_str(context.get("session_id") or context.get("session")),
        **{field: _optional_str(context.get(field))
           for field in ("task", "agent", "transport_session_id")},
    }


def _same_recall_identity_sql(field: str) -> str:
    return f"COALESCE({field}, '') = :{field}"


def _compatible_recall_identity_sql(field: str) -> str:
    return f"COALESCE({field}, '') IN ('', :{field})"


def _recall_scope_match_sql(context: Mapping[str, Any], *, allow_transport: bool) -> str:
    scopes = ["scope = :scope", "requested_scope = :scope"]
    if allow_transport and context["transport_session_id"]:
        scopes.append(_same_recall_identity_sql("transport_session_id"))
    return f"({' OR '.join(scopes)})"


def _pending_recall_match_sql(context: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Require a shared work/session identity or a compatible equal transport.

    Explicit session/task conflicts reject even on the same transport. An
    explicit match permits reconnects and agent handoffs within the recall's
    scope. Equal transport also permits scope-less recall/ingest divergence.
    Missing identity and matching agent names are never positive evidence.
    """

    explicit = [field for field in ("task", "session_id") if context[field]]
    scope_match = _recall_scope_match_sql(context, allow_transport=True)
    matches = []
    missing = []
    for position, field in enumerate(explicit):
        # Disjoint work matches: an equal task, then an equal session when
        # the supplied task was absent on the event. Both-present conflicts
        # remain vetoes; neither agent nor transport overrides them.
        index = "idx_recall_pending_task_session" if "task" in explicit else "idx_recall_pending_session"
        matches.append((index, " AND ".join([
            *missing, _same_recall_identity_sql(field),
            *(_compatible_recall_identity_sql(other) for other in explicit[position + 1:]),
            scope_match,
        ])))
        missing.append(f"COALESCE({field}, '') = ''")
    if context["transport_session_id"]:
        clauses = [_same_recall_identity_sql("transport_session_id"), *missing]
        if context["agent"]:
            clauses.append(_compatible_recall_identity_sql("agent"))
        # On multiplexed connections, missing explicit fields are exact index
        # keys too: another task/session on this transport must not widen work.
        suffix = "_task_session" if "task" in explicit else "_session" if explicit else ""
        matches.append((f"idx_recall_pending_transport{suffix}", " AND ".join(clauses)))
    return matches


def _weights_from_row(row: sqlite3.Row) -> RetrievalWeights:
    return RetrievalWeights(
        scope=str(row["scope"]),
        bm25=float(row["bm25"]),
        vector=float(row["vector"]),
        graph=float(row["graph"]),
        learning_rate=float(row["learning_rate"]),
        updated_at=str(row["updated_at"]),
    )
