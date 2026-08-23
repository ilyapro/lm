"""SQLite persistence for the Living Memory uniform node store."""

from __future__ import annotations

from array import array
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import re
import secrets
import sqlite3
import sys
import time
from datetime import UTC, datetime, timedelta

from living_memory.chunking import TextChunk, chunk_text
from living_memory.config import MemoryConfig, RetrievalWeightConfig
from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity, tokenize
from living_memory.models import (
    CONNECTION_TYPES,
    NODE_LEVELS,
    REJECTED_ALTERNATIVE_KIND,
    Connection,
    ConnectionType,
    Node,
    NodeChunkEmbedding,
    NodeLevel,
    RecallEvent,
    RetrievalWeights,
)
from living_memory.phase import PhaseManager
from living_memory.scope import normalize_scope
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

# Frozen by artifacts/recall-map/prereg.json and
# artifacts/recall-map/relevance/policy.json.  Keep these literals local to
# storage: the history ledger is the production reconstruction of those
# delivery/outcome definitions, not a generic record of every result.
RECALL_DELIVERY_HISTORY_HORIZON_HOURS = 24
RECALL_DELIVERY_HISTORY_HEAD_CUT = 3
RECALL_DELIVERY_HISTORY_ITEMS_PER_EVENT = 6
MAX_RECALL_HISTORY_CANDIDATES = 200
MAX_RECALL_HISTORY_DELIVERIES_PER_NODE = 1024
_RECALL_DELIVERY_HISTORY_FORMAT = 1
#: numpy dtype string for a stored chunk BLOB. Part of the read contract that
#: the vector channel consumes: ``np.frombuffer(blob, dtype=CHUNK_EMBEDDING_DTYPE)``.
CHUNK_EMBEDDING_DTYPE = "<f4"
CHUNK_EMBEDDING_ITEMSIZE = 4

#: SQL literal for the ASCII whitespace set, for TRIM(x, chars): SQLite's
#: one-argument TRIM strips spaces and nothing else.
_SQL_WHITESPACE = "' ' || CHAR(9) || CHAR(10) || CHAR(11) || CHAR(12) || CHAR(13)"

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
        PRIMARY KEY (delivery_event_id, node_id)
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
_RECALL_REPEAT_GATING_ENV = "LM_RECALL_REPEAT_GATING"
_RECALL_REPEAT_MIN_UNLINKED_ENV = "LM_RECALL_REPEAT_MIN_UNLINKED"
_RECALL_REPEAT_MAX_LINK_RATE_ENV = "LM_RECALL_REPEAT_MAX_LINK_RATE"
_RECALL_REPEAT_MIN_SESSIONS_ENV = "LM_RECALL_REPEAT_MIN_SESSIONS"
_RECALL_REPEAT_PROBE_EVERY_ENV = "LM_RECALL_REPEAT_PROBE_EVERY"
_RECALL_REPEAT_DROP_TRAILING_STUBS_ENV = "LM_RECALL_REPEAT_DROP_TRAILING_STUBS"
_ENABLED_ENV_FLAGS = frozenset({"1", "true", "yes", "on"})
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

    SHA-256 over the whitespace-collapsed query plus the requested scope as
    stamped on ``recall_events.requested_scope``. Exact-match only: no
    semantic clustering and no content inspection beyond whitespace
    normalization, so stored fingerprints stay stable across releases.
    """

    collapsed_query = " ".join(str(query).split())
    scope_part = "" if requested_scope is None else str(requested_scope)
    return hashlib.sha256(f"{collapsed_query}\n{scope_part}".encode("utf-8")).hexdigest()


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
    """

    available: bool
    matured: int | None
    consumed: int | None
    trailing_nonconsumed: int | None
    unavailable_reason: str | None = None

    @classmethod
    def known(
        cls, matured: int, consumed: int, trailing_nonconsumed: int
    ) -> "MaturedRecallHistory":
        return cls(
            available=True,
            matured=int(matured),
            consumed=int(consumed),
            trailing_nonconsumed=int(trailing_nonconsumed),
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
class RecallFingerprintStats:
    """Delivered-vs-linked signal aggregated per recall fingerprint."""

    fingerprint: str
    first_seen: str
    last_seen: str
    delivery_count: int
    linked_count: int
    deliveries_since_link: int
    last_linked_at: str | None
    transport_session_count: int
    last_transport_session_id: str | None
    updated_at: str

    @property
    def smoothed_link_rate(self) -> float:
        """Laplace-smoothed linked/delivered rate: (linked+1) / (delivered+2)."""

        return (self.linked_count + 1) / (self.delivery_count + 2)


@dataclass(frozen=True)
class FingerprintGatePolicy:
    """Pure thresholds for gating repeated low-signal recall fingerprints.

    Threshold defaults were derived from the frozen animal-planet dev split only
    (repeated fingerprints there show 7/352 feedback linkage); never tune
    them against the eval or holdout splits. Both legacy repeat controls are
    strict opt-ins and therefore default off.

    When explicitly enabled, ``drop_trailing_stubs`` controls how compact a
    gated delivery gets: the trailing run of stub entries
    (``session_duplicate``/``twin_duplicate``/``near_duplicate``) is dropped
    from a gated response entirely, because no node in it is telling the agent
    anything new — it was already delivered under this exact fingerprint's
    recent history, or its content ships under a bearer in this very response
    — and even stub entries cost ~1.1k chars each on the wire (dev-split
    measurement: fully-stubbed gated deliveries otherwise keep ~65% of their
    ungated size). Content-bearing entries and anything ranked above them
    survive.
    """

    enabled: bool = False
    min_unlinked: int = 5
    max_link_rate: float = 0.2
    min_sessions: int = 2
    probe_every: int = 25
    drop_trailing_stubs: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "FingerprintGatePolicy":
        """Read ``LM_RECALL_REPEAT_*`` knobs with strict opt-in repeat flags."""

        source: Mapping[str, str] = os.environ if env is None else env
        return cls(
            enabled=_env_flag_enabled(source, _RECALL_REPEAT_GATING_ENV),
            min_unlinked=_env_int(source, _RECALL_REPEAT_MIN_UNLINKED_ENV, cls.min_unlinked),
            max_link_rate=_env_rate(source, _RECALL_REPEAT_MAX_LINK_RATE_ENV, cls.max_link_rate),
            min_sessions=_env_int(source, _RECALL_REPEAT_MIN_SESSIONS_ENV, cls.min_sessions),
            probe_every=_env_int(source, _RECALL_REPEAT_PROBE_EVERY_ENV, cls.probe_every),
            drop_trailing_stubs=_env_flag_enabled(
                source, _RECALL_REPEAT_DROP_TRAILING_STUBS_ENV
            ),
        )


def should_gate_fingerprint(
    stats: RecallFingerprintStats | None,
    policy: FingerprintGatePolicy,
) -> bool:
    """Decide whether the next delivery for this fingerprint gets gated.

    Evaluated against the aggregate as it stands before the delivery is
    recorded. Gate iff the fingerprint accumulated ``min_unlinked``
    deliveries since its last feedback link, its Laplace-smoothed link rate
    is at most ``max_link_rate``, and it repeated across ``min_sessions``
    transport sessions. A deterministic probe forces one full delivery
    within every ``probe_every`` consecutive gate-eligible deliveries so a
    fingerprint that turns useful again can re-earn links (0 disables
    probing). Unknown fingerprints (``stats is None``) never gate.
    """

    if not policy.enabled or stats is None:
        return False
    if stats.deliveries_since_link < policy.min_unlinked:
        return False
    if stats.smoothed_link_rate > policy.max_link_rate:
        return False
    if stats.transport_session_count < policy.min_sessions:
        return False
    if policy.probe_every > 0 and (
        (stats.deliveries_since_link - policy.min_unlinked) % policy.probe_every == 0
    ):
        return False
    return True


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
        self._chunk_embedder = chunk_embedder
        self._node_embedding_column_cache: bool | None = None
        self._chunk_table_cache: bool | None = None
        self._anchor_tables_cache: bool | None = None
        # TEMP objects are per-connection, so this cache cannot go stale the
        # way an on-disk schema probe can.
        self._fts_term_tokenizer_ready = False
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

        vector = None if embedding is None else list(embedding)
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

        # Chunks follow the embedding, not the insert: memory_remember and
        # memory_teach pass no vector (traces are embedded later, lazily, by the
        # recall path), and encoding them here would put the model on the write
        # path of every remember. A node with no vector simply has no chunks
        # yet, exactly as it has no nodes.embedding yet.
        if vector:
            self._write_node_chunks(new_id, content, fingerprint, vector)

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
        content_changed = new_content != existing.content
        new_fingerprint = _content_fingerprint(new_content)
        vector = None if embedding is None else list(embedding)
        now = _utc_now()
        unique_agents = int(stats_data.get("unique_agents", existing.unique_agents))
        confidence = float(stats_data.get("confidence", existing.confidence))
        if unique_agents <= 1:
            confidence = min(confidence, 0.5)

        legacy_embedding = self._node_embedding_column_present()
        embedding_assignment = "embedding = ?, " if legacy_embedding else ""
        embedding_value: tuple[Any, ...] = ()
        if legacy_embedding:
            if vector is not None:
                stored_embedding = _json_dumps(vector)
            elif existing.embedding is not None:
                stored_embedding = _json_dumps(existing.embedding)
            else:
                stored_embedding = None
            embedding_value = (stored_embedding,)

        with self._conn:
            self._conn.execute(
                f"""
                UPDATE nodes
                SET content = ?, content_fingerprint = ?, {embedding_assignment}scope = ?,
                    agent = ?, task = ?, context = ?,
                    access_count = ?, last_accessed = ?, usefulness_score = ?,
                    confidence = ?, unique_agents = ?, temporal_hint = ?,
                    source_traces = ?, corrections = ?, provenance = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    new_content,
                    new_fingerprint,
                    *embedding_value,
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
            # Chunk maintenance, in the same transaction as the row it describes.
            #
            # A new vector re-chunks: this is the path the recall channel's lazy
            # backfill takes (_ensure_embedding -> update_node(embedding=...)),
            # and the path consolidation takes for a concept.
            #
            # Content that changed without a new vector only invalidates: the
            # old chunks describe text that no longer exists, and keeping them
            # would answer queries with content the node lost. Re-embedding is
            # left to whoever holds the model — list_unchunked_nodes reports the
            # node until then. Everything else (an access-count bump, a
            # correction, a provenance merge) leaves chunks alone, so the model
            # is not invoked by writes that cannot have changed the text.
            if vector:
                self._write_node_chunks(node_id, new_content, new_fingerprint, vector)
            elif content_changed and self._chunk_table_present():
                self._conn.execute(
                    f"DELETE FROM {CHUNK_EMBEDDING_TABLE} WHERE node_id = ?",
                    (node_id,),
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
        if relation_type == "supersedes":
            self._follow_supersedes_with_anchor_edges(source_id, target_id, now=now)
        return connection_id

    def _follow_supersedes_with_anchor_edges(
        self, source_id: str, target_id: str, *, now: str
    ) -> int:
        """Re-point anchor edges off a superseded node onto its replacement.

        Hooked at the lowest write primitive on purpose: every path that
        supersedes a node funnels through :meth:`_insert_connection` — teach
        (``consolidation.memory_teach``), duplicate-content dedup
        (:meth:`_insert_node`), and derived corrections
        (``edge_derivation.rule_r1c_content_correction``, which persists via
        :meth:`create_connection`) — so one hook covers all three and cannot be
        forgotten by a fourth. It also runs inside the caller's transaction, so
        an anchor never observes a node as superseded while its edges still
        point at the old target.

        Chains resolve incrementally rather than by walking: when B supersedes
        A the edges land on B, so a later "C supersedes B" moves the same edges
        to C. That is also the cycle guard — each step only ever follows the
        edge being written, so a ``supersedes`` cycle cannot make this recurse.
        A self-superseding edge is dropped outright.

        Era displacement has no edge to hook and is handled explicitly by
        :func:`living_memory.query_anchors.migrate_anchor_edges_for_exclusions`.
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
        recall_map: Mapping[str, Any] | None = None,
    ) -> RecallEvent:
        """Persist one recall interaction for later feedback/provenance.

        ``recall_map`` is the map served with this delivery, stored verbatim as
        compact JSON. The live recall path leaves it unset and calls
        :meth:`attach_recall_map` instead: the map is built from the residual
        pool *this* call produced, so it does not exist until the event has been
        recorded and the ranked results have been shaped. Callers that already
        hold a map when they record the event pass it here and pay one write
        instead of two.
        """

        normalized_results = [dict(item) for item in results or []]
        event_id = new_ulid()
        now = _utc_now()
        ambient = dict(ambient_context or {})
        agent = _optional_str(ambient.get("agent"))
        task = _optional_str(ambient.get("task"))
        session_id = _optional_str(ambient.get("session_id") or ambient.get("session"))
        transport_session_id = _optional_str(ambient.get("transport_session_id"))
        encoded_recall_map = _encode_recall_map(recall_map)
        # Stamped from the same value the requested_scope column stores, so
        # callers can reproduce the fingerprint from (query, requested scope)
        # without any storage round-trip.
        fingerprint = recall_fingerprint(query, requested_scope or scope)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO recall_events (
                    id, query, scope, requested_scope, resolved_scopes, ambient_context,
                    depth, max_results, results, agent, task, session_id,
                    transport_session_id, fingerprint, gated, feedback_applied,
                    feedback_trace_id, feedback_applied_at, created_at, recall_map
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, NULL, NULL, ?, ?)
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
                    fingerprint,
                    now,
                    encoded_recall_map,
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
            self._apply_recall_fingerprint_delivery(fingerprint, transport_session_id, now)
        return self.get_recall_event(event_id)  # type: ignore[return-value]

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
        with self._conn:
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
                    SELECT transport_matched, transport_consumed, fallback_consumed
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
                histories[node_id] = MaturedRecallHistory.known(
                    matured=len(outcomes),
                    consumed=sum(outcomes),
                    trailing_nonconsumed=trailing,
                )
            return histories
        except sqlite3.Error:
            return unavailable("history_read_failed")
        except (TypeError, ValueError):
            return unavailable("malformed_history_data")

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
        self._conn.executemany(
            f"""
            UPDATE {RECALL_DELIVERY_HISTORY_TABLE}
            SET transport_matched = ?, transport_consumed = ?,
                fallback_consumed = ?
            WHERE delivery_event_id = ? AND node_id = ?
            """,
            (
                (
                    int(transport_matched),
                    int(node_id in transport_consumed),
                    int(node_id in fallback_consumed),
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
    ) -> list[dict[str, Any]]:
        """Recent deliveries that carried a recall map, newest first.

        The read side of ``recall_events.recall_map``, for the downstream
        consumers of what was already shown: the server-instructions channel
        (what did this session's earlier recalls map?), consumption curtailment
        (which clusters keep being delivered and never followed?), and the
        effect gate (what was on offer when the next recall came in?). Rows
        without a map are not history and never appear.

        Every filter is an equality probe on an indexed column, so the scan is
        bounded by the window rather than by the table:
        ``transport_session_id`` rides ``idx_recall_events_transport_created``
        and ``scope`` the leading column of
        ``idx_recall_events_scope_pending_created``. ``scope`` matches the
        resolved ``scope`` column only — not the ``scope OR requested_scope``
        disjunction :meth:`list_recall_events` uses, which no index can serve —
        and ``task`` is a residual filter over whatever the indexed columns
        already narrowed.

        Each row is ``id``, ``created_at``, ``scope``, ``task``, ``query``,
        ``transport_session_id`` and ``recall_map`` decoded back into the
        payload that was delivered.
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
        if task is not None:
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

    def _apply_recall_fingerprint_delivery(
        self,
        fingerprint: str,
        transport_session_id: str | None,
        now: str,
    ) -> None:
        """O(1) per-fingerprint aggregate upsert for one recorded delivery.

        A transport session counts as new unless it matches the previous
        delivery's stored id exactly; a missing transport id always counts
        as a distinct session (NULL never equals the previous stamp).
        """

        self._conn.execute(
            """
            INSERT INTO recall_fingerprints (
                fingerprint, first_seen, last_seen, delivery_count, linked_count,
                deliveries_since_link, last_linked_at, transport_session_count,
                last_transport_session_id, updated_at
            )
            VALUES (?, ?, ?, 1, 0, 1, NULL, 1, ?, ?)
            ON CONFLICT(fingerprint) DO UPDATE SET
                last_seen = excluded.last_seen,
                delivery_count = delivery_count + 1,
                deliveries_since_link = deliveries_since_link + 1,
                transport_session_count = transport_session_count + CASE
                    WHEN excluded.last_transport_session_id IS NOT NULL
                     AND recall_fingerprints.last_transport_session_id IS NOT NULL
                     AND excluded.last_transport_session_id
                         = recall_fingerprints.last_transport_session_id
                    THEN 0 ELSE 1 END,
                last_transport_session_id = excluded.last_transport_session_id,
                updated_at = excluded.updated_at
            """,
            (fingerprint, now, now, transport_session_id, now),
        )

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

    def delivered_node_ids(
        self,
        transport_session_id: str | None,
        *,
        max_events: int = 200,
    ) -> set[str]:
        """Node ids already delivered to one transport session.

        Unions the ``node_id`` entries (same key semantics as
        ``RecallEvent.result_ids``: falsy ids skipped) from the ``results``
        JSON of the most recent ``max_events`` recall_events stamped with
        ``transport_session_id``. The window bounds per-call work on
        long-lived sessions: events older than the ``max_events`` most
        recent ones fall outside the dedup horizon, so their nodes count as
        undelivered again. A ``None`` or empty transport id identifies no
        session and returns an empty set without touching the database.
        """

        if not transport_session_id or max_events <= 0:
            return set()
        rows = self._conn.execute(
            """
            SELECT results
            FROM recall_events
            WHERE transport_session_id = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (transport_session_id, int(max_events)),
        ).fetchall()
        delivered: set[str] = set()
        for row in rows:
            for item in _json_loads(row["results"], []):
                if not isinstance(item, Mapping):
                    continue
                node_id = item.get("node_id")
                if node_id:
                    delivered.add(str(node_id))
        return delivered

    def fingerprint_delivered_node_ids(
        self,
        fingerprint: str | None,
        *,
        max_events: int = 200,
    ) -> set[str]:
        """Node ids already delivered for one recall fingerprint.

        Same semantics and horizon as ``delivered_node_ids``, keyed by the
        exact-request fingerprint instead of the transport session: unions
        the ``node_id`` entries from the ``results`` JSON of the most recent
        ``max_events`` stamped events (served by
        ``idx_recall_events_fingerprint_created``), so older deliveries fall
        outside the dedup horizon. A ``None`` or empty fingerprint returns
        an empty set without touching the database.
        """

        if not fingerprint or max_events <= 0:
            return set()
        rows = self._conn.execute(
            """
            SELECT results
            FROM recall_events
            WHERE fingerprint = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (fingerprint, int(max_events)),
        ).fetchall()
        delivered: set[str] = set()
        for row in rows:
            for item in _json_loads(row["results"], []):
                if not isinstance(item, Mapping):
                    continue
                node_id = item.get("node_id")
                if node_id:
                    delivered.add(str(node_id))
        return delivered

    def get_recall_fingerprint_stats(
        self, fingerprint: str | None
    ) -> RecallFingerprintStats | None:
        """Aggregate recall signal for one fingerprint, None when never seen."""

        if not fingerprint:
            return None
        row = self._conn.execute(
            "SELECT * FROM recall_fingerprints WHERE fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        return None if row is None else _recall_fingerprint_stats_from_row(row)

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
        # Same-transport events are candidates regardless of scope so equal
        # stamps can close across the recall/ingest default-scope divergence;
        # `transport_session_id = NULL` never matches, so unstamped traces
        # keep the exact-scope candidate set.
        context_transport = _optional_str(context_data.get("transport_session_id"))
        rows = self._conn.execute(
            """
            SELECT *
            FROM recall_events
            WHERE feedback_applied = 0
              AND (
                scope = ? OR requested_scope = ? OR resolved_scopes LIKE ?
                OR transport_session_id = ?
              )
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (
                scope,
                scope,
                f"%{_json_dumps(scope)}%",
                context_transport,
                max(1, int(limit) * 20),
            ),
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
        event_row = self._conn.execute(
            "SELECT fingerprint FROM recall_events WHERE id = ?",
            (event_id,),
        ).fetchone()
        if event_row is None:
            raise KeyError(event_id)
        fingerprint = _optional_str(event_row["fingerprint"])
        now = _utc_now()
        with self._conn:
            flipped = self._conn.execute(
                """
                UPDATE recall_events
                SET feedback_applied = 1,
                    feedback_trace_id = ?,
                    feedback_applied_at = ?
                WHERE id = ? AND feedback_applied = 0
                """,
                (trace_id, now, event_id),
            ).rowcount
            if flipped:
                # Credit the fingerprint aggregate only on the 0 -> 1 flip so
                # re-marking one event can never inflate linked_count.
                if fingerprint:
                    self._conn.execute(
                        """
                        UPDATE recall_fingerprints
                        SET linked_count = linked_count + 1,
                            deliveries_since_link = 0,
                            last_linked_at = ?,
                            updated_at = ?
                        WHERE fingerprint = ?
                        """,
                        (now, now, fingerprint),
                    )
            else:
                # Already applied: keep the legacy overwrite of the trace
                # pointer without re-crediting the aggregate.
                self._conn.execute(
                    """
                    UPDATE recall_events
                    SET feedback_trace_id = ?, feedback_applied_at = ?
                    WHERE id = ?
                    """,
                    (trace_id, now, event_id),
                )
        return self.get_recall_event(event_id)  # type: ignore[return-value]

    def mark_recall_event_gated(self, event_id: str) -> RecallEvent:
        """Flag one recall event as served with a gated (compact) delivery."""

        if self.get_recall_event(event_id) is None:
            raise KeyError(event_id)
        with self._conn:
            self._conn.execute(
                "UPDATE recall_events SET gated = 1 WHERE id = ?",
                (event_id,),
            )
        return self.get_recall_event(event_id)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Grounded-usage attestation ledger (schema v8)
    #
    # Two-step by design: ``claim_recall_attestation`` takes the
    # (event, evidence) key and records the verdict the server computed,
    # ``complete_recall_attestation`` records what that verdict was allowed to
    # change once the writes have actually happened. Claiming first is what
    # makes the ledger a real guard: the key is unavailable for the whole
    # window in which credit is being applied, so a concurrent or retried
    # submission of the same evidence replays instead of crediting twice. The
    # failure mode this leaves is a claimed-but-uncredited row after a crash
    # mid-apply, i.e. under-crediting — the conservative direction for a signal
    # whose whole purpose is to not be inflatable.
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
            with self._conn:
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
        with self._conn:
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

        Counts cover the whole FTS index, which includes soft-deleted
        (``decayed = 1``) nodes: soft deletion touches no FTS-synced column,
        so the row stays indexed — the same corpus ``bm25()`` ranks over in
        :meth:`search_content`. Cost per call: one TEMP-table round trip for
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
                    "SELECT doc FROM nodes_fts_vocab WHERE term = ?", (token,)
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
        with self._conn:
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

        self._chunk_embedder = embedder

    def _resolve_chunk_embedder(self) -> ChunkEmbedder:
        if self._chunk_embedder is None:
            model = LocalEmbeddingModel(model_name=self.config.embedding_model)
            self._chunk_embedder = lambda texts: [model.embed(text) for text in texts]
        return self._chunk_embedder

    def iter_chunk_embedding_rows(
        self,
        *,
        scope: str | None = None,
        level: NodeLevel | None = None,
        include_decayed: bool = False,
    ) -> Iterator[tuple[str, int, memoryview]]:
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
            SELECT c.node_id, c.chunk_index, c.embedding
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
            for node_id, chunk_index, blob in batch:
                yield str(node_id), int(chunk_index), memoryview(blob)

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
    ) -> list[Node]:
        """Active nodes whose chunks are missing or stale, oldest work first.

        The v6 analogue of :meth:`list_unembedded_nodes`, and deliberately not
        the same question: a node whose content was edited still has a
        ``nodes.embedding`` and so is invisible to the legacy query, but its
        chunks were dropped as stale and it does need re-embedding.

        Whitespace-only content is excluded. The chunker yields no windows for
        text with no tokens, so such a node can never acquire a chunk row — and
        a backfill driven by "keep going until this list is empty" would spin on
        it forever. SQLite's one-argument ``TRIM`` strips spaces only, hence the
        explicit whitespace set.
        """

        if not self._chunk_table_present():
            return []

        clauses = [
            "n.decayed = 0",
            f"TRIM(n.content, {_SQL_WHITESPACE}) != ''",
        ]
        params: list[Any] = []
        if level is not None:
            self._validate_level(level)
            clauses.append("n.level = ?")
            params.append(level)
        if scope is not None:
            clauses.append("n.scope = ?")
            params.append(scope)
        params.append(int(limit))
        rows = self._conn.execute(
            f"""
            SELECT n.* FROM nodes n
            WHERE {' AND '.join(clauses)}
              AND NOT EXISTS (
                    SELECT 1 FROM {CHUNK_EMBEDDING_TABLE} c
                    WHERE c.node_id = n.id
                      AND c.content_fingerprint = n.content_fingerprint
              )
            ORDER BY n.timestamp DESC, n.id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [_node_from_row(row) for row in rows]

    def delete_node_chunks(self, node_id: str) -> int:
        """Remove every chunk of one node. Returns the number of rows deleted."""

        if not self._chunk_table_present():
            return 0
        with self._conn:
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
        with self._conn:
            written = self._replace_node_chunks(node_id, chunks, fingerprint)
        return written

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
                )
            )
        self._conn.executemany(
            f"""
            INSERT INTO {CHUNK_EMBEDDING_TABLE} (
                id, node_id, chunk_index, dimensions, embedding,
                token_start, token_end, content_fingerprint, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        return len(payload)

    def _chunk_embeddings_for(
        self,
        content: str,
        node_embedding: Sequence[float] | None,
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
        if not chunks:
            return []
        if len(chunks) == 1:
            if node_embedding:
                return [(chunks[0], list(node_embedding))]
            vectors = self._resolve_chunk_embedder()([chunks[0].text])
        else:
            vectors = self._resolve_chunk_embedder()([chunk.text for chunk in chunks])
        vectors = list(vectors)
        if len(vectors) != len(chunks):
            raise ValueError(
                f"chunk embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )
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
        with self._conn:
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
        (``_find_existing_concept``, cross-scope promotion) were calibrated
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

    # ------------------------------------------------------------------
    # Query anchors (schema v7)
    #
    # Thin table access only: rows in, dataclasses out, no policy. Which
    # anchor an incoming query reinforces, how much weight one grounded
    # consumption is worth, and when an anchor has gone stale all live in
    # living_memory.query_anchors. The one exception is deliberate and is
    # documented on _insert_connection: writing a `supersedes` edge re-points
    # anchor edges in the same transaction, because a caller that forgets to
    # would silently leave anchors pointing into the past.
    # ------------------------------------------------------------------

    def get_query_anchor(self, anchor_id: str) -> QueryAnchor | None:
        if not self._anchor_tables_present():
            return None
        row = self._conn.execute(
            f"SELECT * FROM {QUERY_ANCHOR_TABLE} WHERE id = ?",
            (str(anchor_id),),
        ).fetchone()
        return None if row is None else _query_anchor_from_row(row)

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
        with self._conn:
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
        with self._conn:
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
        with self._conn:
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

    def iter_query_anchor_vectors(
        self,
        scopes: Sequence[str] | None = None,
        *,
        include_decayed: bool = False,
        limit: int | None = None,
    ) -> Iterator[tuple[str, str, int, memoryview]]:
        """Yield ``(anchor_id, scope, dimensions, embedding)`` for live anchors.

        The anchor analogue of :meth:`iter_chunk_embedding_rows`: a bare tuple
        cursor over BLOB views, so a matcher decodes only what it keeps. The
        corpus is small by design — one row per distinct remembered question,
        estimated at a few thousand — so this is a scan, not an ANN index.
        ``dimensions`` is the row's own recorded width, never the BLOB length.

        Yields nothing on a pre-v7 database rather than raising.
        """

        if not self._anchor_tables_present():
            return
        clauses: list[str] = []
        params: list[Any] = []
        if not include_decayed:
            clauses.append("decayed = 0")
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

        Read-modify-write in one statement, the contract
        ``consolidation._upsert_weighted_connection`` exists to provide and
        that :meth:`create_connection` does *not*: its
        ``ON CONFLICT ... DO UPDATE SET weight = excluded.weight`` overwrites,
        so a re-derivation at a lower weight silently downgrades an edge that
        repeated use had strengthened.

        Anchor edges go one step further than that mirror's ``max()``: weight
        *adds*, saturating at 1.0. An anchor edge is a repetition signal — the
        same question grounding on the same node again is new evidence, not a
        restatement of the old one-shot structural score that ``max()`` is
        right for. ``hits`` keeps counting after the weight saturates, so the
        evidence stays legible.
        """

        if not self._anchor_tables_present():
            raise RuntimeError("query anchor tables are absent (pre-v7 database)")
        stamp = now or _utc_now()
        increment = min(1.0, max(0.0, float(weight)))
        with self._conn:
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

    def query_anchor_revision(self) -> tuple[Any, ...]:
        """``(live anchor count, greatest anchor id)``, or ``()`` before v7.

        The cache key for the live-anchor matrix, the anchor analogue of
        ``retrieval._chunk_table_revision``. Measured cold, the scan behind
        :meth:`iter_query_anchor_vectors` is ~5 ms at 4,000 anchors x 384
        dims — small against the chunk corpus, but not free enough to repeat on
        every recall when the recall's whole latency budget is single-digit
        milliseconds. A caller that caches the matrix re-asks this instead.

        Two components, and the pair is complete for what the matrix holds —
        the id and vector of every live anchor. Only three things can change
        that set. An insert mints a fresh ULID whose leading 48 bits are the
        current millisecond, so it raises ``MAX(id)``. A decay and a revival
        both move the live count. What is deliberately *not* covered is a
        reinforcement, because it changes neither: reinforcing updates counters
        and freshness and never rewrites the stored vector, exactly so that the
        cached matrix survives the most frequent write on the anchor path.

        ``MAX(updated_at)`` is absent for the same reason it is absent from the
        chunk revision: ``updated_at`` is in no index, so that aggregate would
        scan the table, embedding BLOBs included. Two statements rather than
        one for the same reason as well — folding them gives up both index fast
        paths.
        """

        if not self._anchor_tables_present():
            return ()
        counted = self._conn.execute(
            f"SELECT COUNT(*) FROM {QUERY_ANCHOR_TABLE} WHERE decayed = 0"
        ).fetchone()
        greatest = self._conn.execute(
            f"SELECT MAX(id) FROM {QUERY_ANCHOR_TABLE}"
        ).fetchone()
        return (int(counted[0]), greatest[0])

    def list_anchor_edge_targets(self) -> list[str]:
        """Every distinct node id that some anchor edge points at."""

        if not self._anchor_tables_present():
            return []
        rows = self._conn.execute(
            f"SELECT DISTINCT target_id FROM {QUERY_ANCHOR_EDGE_TABLE} ORDER BY target_id"
        ).fetchall()
        return [str(row["target_id"]) for row in rows]

    def _initialize_schema(self) -> None:
        with self._conn:
            self._migrate_pre_v3_schema()
            self._migrate_pre_v4_schema()
            self._migrate_pre_v5_schema()
            self._migrate_pre_v6_schema()
            self._migrate_pre_v7_schema()
            self._migrate_recall_map_column()
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

                -- Document-frequency reader over the index the triggers above
                -- maintain, for c-TF-IDF labeling (term_document_frequencies
                -- and fts_document_count). 'row' type: one row per term with
                -- doc = how many indexed documents contain it; only `content`
                -- contributes tokens, the UNINDEXED columns none. fts5vocab
                -- stores nothing — it is a stateless view of the FTS index —
                -- so this CREATE on every open *is* the whole migration for
                -- pre-existing databases (the _RECALL_ATTESTATION_SCHEMA_SQL
                -- rationale), there is no data shape a SCHEMA_VERSION bump
                -- could protect, and the statement is additive: no existing
                -- table's DDL changes by a byte.
                CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts_vocab
                USING fts5vocab('nodes_fts', 'row');

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
                    UNIQUE(node_id, chunk_index)
                );

                -- `recall_map` is the recall map served with this delivery,
                -- verbatim (compact JSON: cluster labels, counts, medoid node
                -- ids, ask hints), or NULL when no map was built: what the
                -- agent was actually shown, recorded at delivery time so a
                -- later pass can ask whether the next recall followed one of
                -- its clusters. It is last in DDL order because
                -- `_migrate_recall_map_column` appends it with ALTER TABLE,
                -- and a migrated file and a fresh one must end up with the
                -- same column order. Its comment lives out here rather than
                -- beside the column: SQLite reconstructs this DDL text on
                -- ALTER TABLE ... DROP COLUMN, and a comment attached to a
                -- dropped column leaves the stored statement unparseable.
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
                    fingerprint TEXT,
                    gated INTEGER NOT NULL DEFAULT 0,
                    feedback_applied INTEGER NOT NULL DEFAULT 0 CHECK (feedback_applied IN (0, 1)),
                    feedback_trace_id TEXT REFERENCES nodes(id),
                    feedback_applied_at TEXT,
                    created_at TEXT NOT NULL,
                    recall_map TEXT
                );

                -- Per-fingerprint recall-signal accounting (schema v5):
                -- delivered-vs-linked aggregates for exact repeated
                -- (query, requested_scope) recall requests. Maintained online
                -- by record_recall_event / mark_recall_event_feedback and
                -- rebuilt from stamped recall_events on migration.
                CREATE TABLE IF NOT EXISTS recall_fingerprints (
                    fingerprint TEXT PRIMARY KEY,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    delivery_count INTEGER NOT NULL DEFAULT 0,
                    linked_count INTEGER NOT NULL DEFAULT 0,
                    deliveries_since_link INTEGER NOT NULL DEFAULT 0,
                    last_linked_at TEXT,
                    transport_session_count INTEGER NOT NULL DEFAULT 0,
                    last_transport_session_id TEXT,
                    updated_at TEXT NOT NULL
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

                -- Chunk-scan driver (iter_chunk_embedding_rows), the v6 mirror
                -- of idx_nodes_embedded_active_scope. That index exists so the
                -- vector channel can enumerate one scope's embedded active
                -- nodes without touching the table, and it cannot serve the
                -- chunk scan: it leads with `level` while the chunk scan always
                -- filters `scope` and only sometimes `level`, and its
                -- `embedding IS NOT NULL` predicate names the very column v6
                -- replaces (and a later operator step drops). Carrying `id`
                -- makes the range covering, so the scan walks a scope's active
                -- nodes index-only and probes UNIQUE(node_id, chunk_index) once
                -- per node — which also yields chunks in ordinal order with no
                -- sort. Partial on decayed = 0 because chunk rows outlive their
                -- node's soft delete and liveness only lives here.
                CREATE INDEX IF NOT EXISTS idx_nodes_active_scope_level
                    ON nodes(scope, level, id)
                    WHERE decayed = 0;

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

                -- Type-leading index for the per-recall supersedes scan
                -- (retrieval._supersedes_sets: SELECT source_id, target_id WHERE
                -- type = 'supersedes'). The source/target-leading indexes above
                -- cannot serve a type-only filter, so without this the query is a
                -- full table scan. Carrying source_id/target_id makes it a
                -- covering, index-only scan with no per-row table lookups.
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

                -- Session-delivery dedup (delivered_node_ids): equality probe
                -- on transport_session_id plus the most-recent-events window,
                -- instead of scanning every session's events. Runs after
                -- _migrate_pre_v4_schema, so the column exists on legacy DBs.
                CREATE INDEX IF NOT EXISTS idx_recall_events_transport_created
                    ON recall_events(transport_session_id, created_at DESC);

                -- Fingerprint delivery history (fingerprint_delivered_node_ids
                -- and gating reads): equality probe on the request fingerprint
                -- plus the most-recent-events window. Runs after
                -- _migrate_pre_v5_schema, so the column exists on legacy DBs.
                CREATE INDEX IF NOT EXISTS idx_recall_events_fingerprint_created
                    ON recall_events(fingerprint, created_at DESC);
                """
            )
            # Query anchors (v7). Same script the pre-v7 migration runs, so a
            # fresh database and a migrated one end up with identical objects.
            self._conn.executescript(_QUERY_ANCHOR_SCHEMA_SQL)
            self._anchor_tables_cache = None
            # Attestation ledger (v8). Runs after the main script because its
            # foreign keys name recall_events and nodes, and unconditionally on
            # every open because that *is* the pre-v8 migration — see the note
            # on _RECALL_ATTESTATION_SCHEMA_SQL for why no _migrate_pre_v8_schema
            # exists.
            self._conn.executescript(_RECALL_ATTESTATION_SCHEMA_SQL)
            # Frozen relevance history. The schema is additive; the
            # idempotent reconstruction below is what populates it for a live
            # database whose recall_events predate this release.
            self._conn.executescript(_RECALL_DELIVERY_HISTORY_SCHEMA_SQL)
            self._backfill_recall_delivery_history()
            self._create_legacy_embedding_index()
            self._backfill_missing_content_fingerprints()
            self._backfill_recall_fingerprint_signal()
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

    def _migrate_pre_v5_schema(self) -> None:
        """Add fingerprint + gated to recall_events for DBs created at schema_version <= 4."""

        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='recall_events'"
        ).fetchone()
        if row is None:
            return
        columns = {
            r["name"]
            for r in self._conn.execute("PRAGMA table_info(recall_events)").fetchall()
        }
        if "fingerprint" not in columns:
            self._conn.execute("ALTER TABLE recall_events ADD COLUMN fingerprint TEXT")
        if "gated" not in columns:
            self._conn.execute(
                "ALTER TABLE recall_events ADD COLUMN gated INTEGER NOT NULL DEFAULT 0"
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

    def _backfill_recall_delivery_history(self) -> None:
        """Idempotently reconstruct the frozen normalized history ledger.

        One state row is the commit marker.  A completed reconstruction and a
        recorded malformed-legacy verdict are both terminal and therefore
        cheap on every later open.  If the process dies before the marker is
        written, SQLite rolls the surrounding initialization transaction back;
        the next open starts the reconstruction again.
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

    def _backfill_recall_fingerprint_signal(self) -> None:
        """Stamp fingerprints on legacy recall_events and rebuild aggregates.

        Idempotent: fires only while rows still have ``fingerprint IS NULL``
        (fresh inserts stamp the column directly, so after the schema-v5
        migration runs once this is a no-op and never clobbers the online
        aggregates). Rebuilding ``recall_fingerprints`` from the stamped
        events lets gating activate from historical evidence on live DBs
        instead of starting cold.
        """

        rows = self._conn.execute(
            """
            SELECT id, query, requested_scope
            FROM recall_events
            WHERE fingerprint IS NULL
            """
        ).fetchall()
        if not rows:
            return
        self._conn.executemany(
            "UPDATE recall_events SET fingerprint = ? WHERE id = ?",
            [
                (
                    recall_fingerprint(
                        str(r["query"]), _optional_str(r["requested_scope"])
                    ),
                    str(r["id"]),
                )
                for r in rows
            ],
        )
        self._rebuild_recall_fingerprint_aggregates()

    def _rebuild_recall_fingerprint_aggregates(self) -> None:
        """Recompute ``recall_fingerprints`` from stamped recall_events.

        Replays deliveries in recorded order (created_at, rowid — the house
        recall_events ordering) interleaved with feedback links at their
        applied-at time, applying the same per-delivery / per-link rules as
        ``record_recall_event`` and ``mark_recall_event_feedback``, so a
        migrated DB gates exactly like one that accumulated the signal
        online. Within one timestamp deliveries sort before links.
        """

        self._conn.execute("DELETE FROM recall_fingerprints")
        actions: list[tuple[str, int, int, str, str | None]] = []
        for row in self._conn.execute(
            """
            SELECT rowid, fingerprint, transport_session_id, feedback_applied,
                   feedback_applied_at, created_at
            FROM recall_events
            WHERE fingerprint IS NOT NULL
            """
        ):
            fingerprint = str(row["fingerprint"])
            created_at = str(row["created_at"])
            order = int(row["rowid"])
            actions.append(
                (
                    created_at,
                    0,
                    order,
                    fingerprint,
                    _optional_str(row["transport_session_id"]),
                )
            )
            if row["feedback_applied"]:
                linked_at = _optional_str(row["feedback_applied_at"]) or created_at
                # A link can never precede its own delivery: clamp malformed
                # applied-at stamps to the event's created_at.
                actions.append((max(linked_at, created_at), 1, order, fingerprint, None))
        if not actions:
            return
        actions.sort(key=lambda item: (item[0], item[1], item[2]))
        aggregates: dict[str, dict[str, Any]] = {}
        for at, kind, _order, fingerprint, transport_id in actions:
            state = aggregates.get(fingerprint)
            if kind == 0:
                if state is None:
                    aggregates[fingerprint] = {
                        "first_seen": at,
                        "last_seen": at,
                        "delivery_count": 1,
                        "linked_count": 0,
                        "deliveries_since_link": 1,
                        "last_linked_at": None,
                        "transport_session_count": 1,
                        "last_transport_session_id": transport_id,
                    }
                    continue
                state["delivery_count"] += 1
                state["deliveries_since_link"] += 1
                state["last_seen"] = at
                previous = state["last_transport_session_id"]
                if transport_id is None or previous is None or transport_id != previous:
                    state["transport_session_count"] += 1
                state["last_transport_session_id"] = transport_id
            elif state is not None:
                state["linked_count"] += 1
                state["deliveries_since_link"] = 0
                state["last_linked_at"] = at
        now = _utc_now()
        self._conn.executemany(
            """
            INSERT INTO recall_fingerprints (
                fingerprint, first_seen, last_seen, delivery_count, linked_count,
                deliveries_since_link, last_linked_at, transport_session_count,
                last_transport_session_id, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    fingerprint,
                    state["first_seen"],
                    state["last_seen"],
                    state["delivery_count"],
                    state["linked_count"],
                    state["deliveries_since_link"],
                    state["last_linked_at"],
                    state["transport_session_count"],
                    state["last_transport_session_id"],
                    now,
                )
                for fingerprint, state in aggregates.items()
            ],
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


def _env_flag_enabled(env: Mapping[str, str], name: str) -> bool:
    raw = str(env.get(name) or "").strip().lower()
    return raw in _ENABLED_ENV_FLAGS


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = str(env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(0, value)


def _env_rate(env: Mapping[str, str], name: str, default: float) -> float:
    raw = str(env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return min(1.0, max(0.0, value))


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


def _recall_fingerprint_stats_from_row(row: sqlite3.Row) -> RecallFingerprintStats:
    return RecallFingerprintStats(
        fingerprint=str(row["fingerprint"]),
        first_seen=str(row["first_seen"]),
        last_seen=str(row["last_seen"]),
        delivery_count=int(row["delivery_count"]),
        linked_count=int(row["linked_count"]),
        deliveries_since_link=int(row["deliveries_since_link"]),
        last_linked_at=row["last_linked_at"],
        transport_session_count=int(row["transport_session_count"]),
        last_transport_session_id=row["last_transport_session_id"],
        updated_at=str(row["updated_at"]),
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
    context_session = _optional_str(context.get("session_id") or context.get("session"))
    context_task = _optional_str(context.get("task"))
    context_agent = _optional_str(context.get("agent"))
    context_transport = _optional_str(context.get("transport_session_id"))

    # Equal transport ids are same-connection evidence strong enough to span
    # the recall/ingest default-scope divergence (a scope-less recall plans
    # requested_scope='global' while a scope-less trace lands on the
    # configured default scope). The text fallbacks below never gain that
    # bypass: without both transport ids present, candidacy still requires
    # the exact scope match.
    transport_equal = bool(
        event.transport_session_id
        and context_transport
        and event.transport_session_id == context_transport
    )
    exact_scope_match = scope == event.scope or scope == event.requested_scope
    if not exact_scope_match and not transport_equal:
        return None

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

    # Transport identity ranks below explicit context (the rejects/strongs
    # above) but above the text fallbacks; one-sided stamping falls through
    # to the legacy rules unchanged.
    if event.transport_session_id and context_transport:
        if transport_equal:
            return "strong"
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
