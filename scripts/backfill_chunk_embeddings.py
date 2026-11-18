#!/usr/bin/env python3
"""Live-safe, resumable backfill of chunk embeddings for every active node.

Backfill encodes current text windows for active sources in the selected
encoder space. Verify and dry-run inspect coverage on a WAL-aware read-only
connection. The separate drop-embedding-column command removes legacy node
vectors only after explicit confirmation and backup.

Safe under a live server
------------------------
Opening MemoryStore runs its normal migrations. After open, the backfill
restricts writes to chunk replacement, source fingerprints and schema-owned revisions.
Per-source transactions retry on SQLITE_BUSY.

Resumability
------------
A source is ready when its dense chunk set covers the current fingerprint in
the selected encoder space and each BLOB has the expected width. Backfill
encodes outside the transaction and checks the source and observed chunk IDs
under the write lock before replacing a set. Re-run to continue from the
remaining sources. Use --limit to budget source attempts per invocation.

The bounded batch is ``--batch-size`` nodes: one page of pending nodes, one
encode call, then one write transaction per node.

Every chunk is encoded from its own text. ``nodes.embedding`` is never recycled
as chunk 0's vector even for a single-window node (which the write path does do,
legitimately, because there it was just computed by the same model): some legacy
vectors in the live database were written while the model was unavailable and are
hash-fallback vectors, and mixing those into the chunk corpus would put two
incompatible vector spaces in one table.

Backup is mandatory
-------------------
``backfill`` and ``drop-embedding-column`` refuse to start without one, and
``--backup`` takes it through ``sqlite3.Connection.backup()``. That API is
WAL-correct: it copies the committed database as one consistent snapshot while a
server keeps writing. ``cp global.sqlite3 backup.sqlite3`` is **not** a valid
backup here: the WAL can hold committed transactions absent from the main file.

``--dry-run`` needs no backup: it opens the database read-only
(``mode=ro`` + ``PRAGMA query_only``), never constructs a ``MemoryStore`` (whose
constructor migrates, i.e. writes), and reports what a real run would write.

Usage
-----
    # rehearse, read-only, no backup needed
    python3 scripts/backfill_chunk_embeddings.py backfill --db DB --dry-run

    # real run, taking its own WAL-correct backup first
    python3 scripts/backfill_chunk_embeddings.py backfill --db DB \\
        --backup /var/backups/lm/global-pre-chunk.sqlite3 --json summary.json

    python3 scripts/backfill_chunk_embeddings.py verify --db DB
    python3 scripts/backfill_chunk_embeddings.py drop-embedding-column --db DB \\
        --existing-backup /var/backups/lm/global-pre-chunk.sqlite3 --yes
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Iterator, Sequence
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_SRC))

from living_memory.chunking import (  # noqa: E402  (path bootstrap above)
    TextChunk,
    chunk_text,
    get_tokenizer,
)
from living_memory.embeddings import (  # noqa: E402
    EncodedVector,
    DEFAULT_EMBEDDING_MODEL,
    LocalEmbeddingModel,
)
from living_memory.storage import (  # noqa: E402
    CHUNK_EMBEDDING_ITEMSIZE,
    CHUNK_EMBEDDING_TABLE,
    ChunkEmbedder,
    MemoryStore,
    install_derived_only_authorizer,
    # Private on purpose: SQLite's one-argument TRIM strips spaces only, so the
    # whitespace set the pending predicate must trim is a detail of the schema.
    # Importing the one definition beats keeping a second copy in step with it.
    _SQL_WHITESPACE,
    _content_fingerprint,
)

LOG = logging.getLogger("backfill_chunk_embeddings")

#: Pending nodes per page: one source selection, one encode call,
#: then one transaction per node. Larger pages amortize the encoder's warm-up
#: over more texts; smaller ones shorten the window a kill can waste.
DEFAULT_BATCH_NODES = 64

#: Texts per forward pass inside the encoder. 32 is what the phase-0 throughput
#: measurement (~90 texts/s on this CPU) was taken at.
DEFAULT_ENCODE_BATCH = 32

#: Measured 2026-08-17 on this CPU at ``--encode-batch 32``. Only used to project
#: a duration in ``--dry-run``; the real run measures its own rate.
DEFAULT_ASSUMED_RATE = 90.0

#: The live server writes to the same database. Wait for the lock rather than
#: failing the run; MemoryStore's own default (5 s) is tuned for short requests.
DEFAULT_BUSY_TIMEOUT_MS = 30_000
BUSY_RETRIES = 6
BUSY_RETRY_BASE_DELAY = 0.25

#: A node whose content keeps changing under the backfill is re-chunked, but not
#: forever: after this many attempts in one invocation it is left for the next
#: run so the loop cannot livelock on a hot node.
MAX_NODE_ATTEMPTS = 3

DEFAULT_PROGRESS_INTERVAL = 5.0

#: A backup with fewer nodes than this fraction of the source is rejected: that
#: is what a ``cp`` without the ``-wal`` looks like from the outside.
DEFAULT_MIN_BACKUP_FRACTION = 0.99

_SQLITE_BUSY_CODES = frozenset({5, 6})  # SQLITE_BUSY, SQLITE_LOCKED


class BackfillError(RuntimeError):
    """An operator-facing refusal: printed as one line, exit code 2."""


# ---------------------------------------------------------------------------
# Pending-node predicate
# ---------------------------------------------------------------------------

_ACTIVE_CONTENT_CLAUSES = (
    "n.decayed = 0",
    f"TRIM(n.content, {_SQL_WHITESPACE}) != ''",
)


@dataclass(frozen=True)
class ChunkSelection:
    active: str
    filters: tuple[str, ...]
    pending: str
    params: tuple[Any, ...]

    @classmethod
    def for_connection(
        cls, conn: sqlite3.Connection, encoder_id: str, scope: str | None, level: str | None,
        node_id: str | None = None,
    ) -> ChunkSelection:
        clauses = list(_ACTIVE_CONTENT_CLAUSES)
        filters = []
        for column, value in (("scope", scope), ("level", level), ("id", node_id)):
            if value is not None:
                clauses.append(f"n.{column} = ?")
                filters.append(value)
        active = " AND ".join(clauses)
        # A pre-v6 file or a chunk table predating producer identity has no
        # provably compatible rows. Read-only commands must not migrate either.
        width = encoder_width(encoder_id)
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({CHUNK_EMBEDDING_TABLE})")}
        if "encoder_id" not in columns:
            return cls(active, tuple(filters), active, tuple(filters))

        conn.create_function("content_digest", 1, _content_fingerprint, deterministic=True)
        ready = f"""SELECT 1 FROM {CHUNK_EMBEDDING_TABLE} c
            WHERE c.node_id = n.id GROUP BY c.node_id
            HAVING MIN(c.chunk_index) = 0 AND MAX(c.chunk_index) = COUNT(*) - 1
               AND MIN(c.content_fingerprint = n.content_fingerprint
                       AND c.encoder_id IS ? AND c.dimensions = ?
                       AND typeof(c.embedding) = 'blob' AND length(c.embedding) = ?) = 1"""
        pending = f"{active} AND (n.content_fingerprint IS NOT content_digest(n.content) OR NOT EXISTS ({ready}))"
        return cls(active, tuple(filters), pending,
                   (*filters, encoder_id, width, width * CHUNK_EMBEDDING_ITEMSIZE))

    def count(self, conn: sqlite3.Connection) -> int:
        return conn.execute(f"SELECT COUNT(*) FROM nodes n WHERE {self.pending}", self.params).fetchone()[0]


def encoder_width(encoder_id: str) -> int:
    prefix, _, suffix = encoder_id.rpartition(":")
    if prefix and suffix.isascii() and suffix.isdigit():
        try:
            width = int(suffix)
        except ValueError:
            pass
        else:
            if 0 < width <= (2**63 - 1) // CHUNK_EMBEDDING_ITEMSIZE:
                return width
    raise BackfillError(f"expected an encoder ID ending in a positive width: {encoder_id!r}")


# ---------------------------------------------------------------------------
# Connections, schema shape, backups
# ---------------------------------------------------------------------------


def open_readonly(path: Path) -> sqlite3.Connection:
    """Open a database that must not be written to, and make that structural.

    ``mode=ro`` refuses writes at the VFS layer and ``query_only`` refuses them
    at the statement layer. Nothing here constructs a ``MemoryStore``: its
    constructor runs migrations, which is a write.

    SQLite still materializes empty ``-wal``/``-shm`` sidecars for a read-only
    WAL database (it needs the shared-memory index to read a WAL). No frame is
    ever committed, so the database itself is byte-identical afterwards.
    ``immutable=1`` would avoid even that, and is deliberately not used: it makes
    SQLite ignore the ``-wal`` entirely, which on the live database means reading
    a stale snapshot — the very failure mode ``cp``-without-``-wal`` produces.
    """

    uri = f"file:{quote(str(path.resolve()), safe='/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = 1")
    return conn


def has_chunk_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (CHUNK_EMBEDDING_TABLE,),
    ).fetchone()
    return row is not None


def has_node_column(conn: sqlite3.Connection, column: str) -> bool:
    return column in {
        str(row[1]) for row in conn.execute("PRAGMA table_info(nodes)").fetchall()
    }


def count_nodes(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])


@dataclass
class BackupCheck:
    """Outcome of establishing that a usable backup exists."""

    path: str
    created: bool
    bytes: int
    source_nodes: int
    backup_nodes: int
    quick_check: str
    seconds: float


def take_backup(source: Path, dest: Path, *, overwrite: bool) -> float:
    """Copy ``source`` to ``dest`` with the sqlite3 backup API. Returns seconds.

    WAL-correct by construction: the API copies the committed database as one
    consistent snapshot, including everything sitting in the ``-wal``, while
    another process keeps writing. The source is opened read-only.
    """

    if dest.exists():
        if not overwrite:
            raise BackfillError(
                f"backup target already exists: {dest} "
                "(pass --overwrite-backup to replace it, or use --existing-backup "
                "to reuse it as-is)"
            )
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    src = open_readonly(source)
    try:
        target = sqlite3.connect(str(dest))
        try:
            src.backup(target)
        finally:
            target.close()
    finally:
        src.close()
    return time.perf_counter() - started


def ensure_backup(
    db: Path,
    *,
    backup: Path | None,
    existing_backup: Path | None,
    overwrite: bool,
    min_fraction: float,
) -> BackupCheck:
    """Refuse to proceed unless a usable backup exists; take one if asked to."""

    if backup is None and existing_backup is None:
        raise BackfillError(
            "refusing to run without a backup. Either --backup PATH (this script "
            "takes a WAL-correct one via the sqlite3 backup API) or "
            "--existing-backup PATH (validated before the run). A plain `cp` of "
            "the .sqlite3 without its -wal is NOT a valid backup of this database."
        )
    if backup is not None and existing_backup is not None:
        raise BackfillError("pass either --backup or --existing-backup, not both")

    target = backup if backup is not None else existing_backup
    assert target is not None
    if target.resolve() == db.resolve():
        raise BackfillError("the backup must not be the database being modified")

    created = False
    seconds = 0.0
    if backup is not None:
        LOG.info("taking backup: %s -> %s", db, backup)
        seconds = take_backup(db, backup, overwrite=overwrite)
        created = True
    elif not target.is_file():
        raise BackfillError(f"--existing-backup does not exist: {target}")

    source_conn = open_readonly(db)
    try:
        source_nodes = count_nodes(source_conn)
    finally:
        source_conn.close()

    backup_conn = open_readonly(target)
    try:
        quick = str(backup_conn.execute("PRAGMA quick_check").fetchone()[0])
        try:
            backup_nodes = count_nodes(backup_conn)
        except sqlite3.DatabaseError as exc:
            raise BackfillError(f"backup {target} has no readable nodes table: {exc}")
    finally:
        backup_conn.close()

    if quick != "ok":
        raise BackfillError(f"backup {target} failed PRAGMA quick_check: {quick}")
    minimum = math.ceil(source_nodes * min_fraction)
    if backup_nodes < minimum:
        raise BackfillError(
            f"backup {target} holds {backup_nodes} nodes but the database has "
            f"{source_nodes} ({minimum} required at --min-backup-fraction "
            f"{min_fraction}). A copy made with `cp` while the server was running "
            "looks exactly like this: the newest transactions were still in the "
            "-wal. Retake it with --backup."
        )

    return BackupCheck(
        path=str(target),
        created=created,
        bytes=target.stat().st_size,
        source_nodes=source_nodes,
        backup_nodes=backup_nodes,
        quick_check=quick,
        seconds=round(seconds, 3),
    )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


@dataclass
class EmbedderInfo:
    backend: str
    model: str
    dimensions: int
    encoder_id: str


def build_chunk_embedder(
    *, model_name: str, encode_batch: int, allow_fallback: bool
) -> tuple[ChunkEmbedder, EmbedderInfo]:
    """Return a batch encoder for chunk texts, plus what it actually resolved to.

    ``LocalEmbeddingModel.embed`` encodes one text per call; a backfill of ~45k
    chunks wants the underlying sentence-transformers batch path instead, which
    is where the measured ~90 texts/s comes from. So the model is loaded through
    the same resolver every other caller uses (identical model, identical
    normalization) and its loaded encoder is used directly when there is one.

    Refuses to write hash-fallback vectors into a real database unless
    ``allow_fallback`` says so: silently filling the chunk table with vectors
    from a different embedding space is the worst outcome available here, and it
    is invisible afterwards. The test suite runs with
    ``LIVING_MEMORY_EMBEDDING_BACKEND=hash`` and opts in explicitly.

    This function is the seam tests replace to inject a deterministic encoder.
    """

    model = LocalEmbeddingModel(model_name=model_name)
    model.warmup()
    # warmup() populates the private attribute when a real encoder loaded; None
    # means every embed() call would go to the hash fallback.
    backing = getattr(model, "_model", None)
    if backing is None:
        if not allow_fallback:
            raise BackfillError(
                f"the embedding model {model_name!r} did not load, so every "
                "vector would come from the hash fallback — a different vector "
                "space from the one recall uses. Fix the model cache, or pass "
                "--allow-fallback-embeddings if that is genuinely what you want "
                "(a test fixture, never production)."
            )
        LOG.warning("encoding with the hash fallback: %s did not load", model_name)

        def embed(texts: Sequence[str]) -> list[list[float]]:
            return [model.embed(text) for text in texts]

        backend = "hash-fallback"
    else:

        def embed(texts: Sequence[str]) -> list[list[float]]:
            items = list(texts)
            if not items:
                return []
            vectors = backing.encode(
                items,
                batch_size=encode_batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return [EncodedVector(row, f"st:{model_name}:{len(row)}") for row in vectors.tolist()]

        backend = "sentence-transformers"

    return embed, EmbedderInfo(
        backend=backend, model=model_name, dimensions=int(model.dimensions), encoder_id=model.encoder_id()
    )


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    if seconds != seconds or seconds in (float("inf"), float("-inf")):  # NaN/inf
        return "unknown"
    seconds = max(0.0, seconds)
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class ProgressReporter:
    """processed/total with a rate and an ETA, at most one line per interval."""

    def __init__(
        self,
        total: int,
        *,
        interval: float = DEFAULT_PROGRESS_INTERVAL,
        stream: Any = None,
        enabled: bool = True,
    ) -> None:
        self.total = total
        self.interval = interval
        self.stream = stream if stream is not None else sys.stdout
        self.enabled = enabled
        self.started = time.perf_counter()
        self._last = self.started

    def update(self, nodes: int, chunks: int, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if not force and now - self._last < self.interval:
            return
        self._last = now
        elapsed = max(now - self.started, 1e-9)
        node_rate = nodes / elapsed
        remaining = max(self.total - nodes, 0)
        eta = remaining / node_rate if node_rate > 0 else float("inf")
        percent = (100.0 * nodes / self.total) if self.total else 100.0
        print(
            f"  {nodes}/{self.total} nodes ({percent:5.1f}%)  "
            f"{chunks} chunks  {node_rate:6.1f} nodes/s  "
            f"{chunks / elapsed:6.1f} chunks/s  "
            f"elapsed {format_duration(elapsed)}  eta {format_duration(eta)}",
            file=self.stream,
            flush=True,
        )


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


@dataclass
class BackfillSummary:
    """Everything a runbook step needs to report, in one JSON-able object."""

    mode: str
    db: str
    started_at: str
    elapsed_seconds: float = 0.0
    pending_before: int = 0
    pending_after: int = 0
    nodes_processed: int = 0
    nodes_written: int = 0
    chunks_written: int = 0
    chunk_bytes_written: int = 0
    chunk_writes: int = 0
    nodes_rechunked_after_content_change: int = 0
    nodes_skipped_no_chunks: int = 0
    nodes_skipped_no_fingerprint: int = 0
    nodes_skipped_attempt_limit: int = 0
    nodes_vanished: int = 0
    zero_vector_chunks: int = 0
    busy_retries: int = 0
    nodes_per_second: float = 0.0
    chunks_per_second: float = 0.0
    max_chunks_in_one_node: int = 0
    encoder_id: str = ""
    embedding_backend: str = ""
    embedding_model: str = ""
    embedding_dimensions: int = 0
    tokenizer: str = ""
    chunk_rows_total: int = 0
    chunk_bytes_total: int = 0
    node_embedding_column_present: bool = True
    complete: bool = False
    backup: dict[str, Any] | None = None
    projected_seconds: float | None = None
    assumed_rate: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def dry_run(
    db: Path,
    *,
    scope: str | None,
    level: str | None,
    limit: int,
    dimensions: int | None,
    assumed_rate: float,
    progress_interval: float,
    quiet: bool,
    node_id: str | None = None,
    encoder_id: str | None = None,
) -> BackfillSummary:
    """Report what a real run would write, touching nothing.

    Chunks every pending node for real (the chunker is cheap; the encoder is not)
    so the chunk and byte totals are counted rather than extrapolated. The vector
    width defaults to that of the selected encoder space.
    """

    summary = BackfillSummary(mode="dry-run", db=str(db), started_at=_utc_now_iso())
    summary.encoder_id = encoder_id or LocalEmbeddingModel().encoder_id()
    width = encoder_width(summary.encoder_id)
    if dimensions is not None:
        if dimensions < 1:
            raise BackfillError("--dimensions must be positive")
        width = dimensions
    started = time.perf_counter()
    conn = open_readonly(db)
    try:
        conn.execute("BEGIN")
        selection = ChunkSelection.for_connection(conn, summary.encoder_id, scope, level, node_id)
        summary.node_embedding_column_present = has_node_column(conn, "embedding")
        summary.tokenizer = get_tokenizer().name
        if has_chunk_table(conn):
            summary.chunk_rows_total = int(
                conn.execute(f"SELECT COUNT(*) FROM {CHUNK_EMBEDDING_TABLE}").fetchone()[0]
            )
            summary.chunk_bytes_total = int(
                conn.execute(
                    f"SELECT COALESCE(SUM(LENGTH(embedding)), 0) FROM {CHUNK_EMBEDDING_TABLE}"
                ).fetchone()[0]
            )
        summary.embedding_dimensions = width
        summary.pending_before = selection.count(conn)
        if not quiet:
            print(
                f"dry run: {summary.pending_before} pending nodes, "
                f"tokenizer {summary.tokenizer}, vector width {width} "
                f"({summary.encoder_id})",
                flush=True,
            )
        reporter = ProgressReporter(
            summary.pending_before, interval=progress_interval, enabled=not quiet
        )
        # Store migration supplies a missing column or NULL fingerprint. An
        # explicitly empty fingerprint survives migration and prevents a write.
        missing = "n.content_fingerprint = ''" if has_node_column(conn, "content_fingerprint") else "0"
        for content, missing_fingerprint in conn.execute(
            f"SELECT n.content, {missing} FROM nodes n WHERE {selection.pending} ORDER BY n.id LIMIT ?",
            (*selection.params, limit or -1),
        ):
            summary.nodes_processed += 1
            if missing_fingerprint:
                summary.nodes_skipped_no_fingerprint += 1
                continue
            chunks = chunk_text(content)
            if not chunks:
                summary.nodes_skipped_no_chunks += 1
                continue
            summary.nodes_written += 1
            summary.chunks_written += len(chunks)
            summary.chunk_bytes_written += len(chunks) * width * CHUNK_EMBEDDING_ITEMSIZE
            summary.max_chunks_in_one_node = max(
                summary.max_chunks_in_one_node, len(chunks)
            )
            reporter.update(summary.nodes_processed, summary.chunks_written)
        summary.pending_after = summary.pending_before
    finally:
        conn.close()

    summary.elapsed_seconds = round(time.perf_counter() - started, 3)
    summary.nodes_per_second = _rate(summary.nodes_processed, summary.elapsed_seconds)
    summary.chunks_per_second = _rate(summary.chunks_written, summary.elapsed_seconds)
    summary.assumed_rate = assumed_rate
    summary.projected_seconds = (
        round(summary.chunks_written / assumed_rate, 1) if assumed_rate > 0 else None
    )
    summary.complete = True
    return summary


def _rate(count: int, elapsed: float) -> float:
    return round(count / elapsed, 2) if elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# Real run
# ---------------------------------------------------------------------------


def _is_busy(exc: sqlite3.OperationalError) -> bool:
    code = getattr(exc, "sqlite_errorcode", None)
    if code is not None:
        return code in _SQLITE_BUSY_CODES
    message = str(exc).lower()
    return "locked" in message or "busy" in message


def _chunk_ids(conn: sqlite3.Connection, node_id: str) -> tuple[str, ...]:
    return tuple(row[0] for row in conn.execute(
        f"SELECT id FROM {CHUNK_EMBEDDING_TABLE} WHERE node_id = ? ORDER BY chunk_index",
        (node_id,),
    ))


def _write_chunks_with_retry(
    store: MemoryStore,
    node_id: str,
    pairs: Sequence[tuple[TextChunk, Sequence[float]]],
    source: sqlite3.Row,
    chunk_ids: tuple[str, ...],
    selection: ChunkSelection,
    summary: BackfillSummary,
) -> int:
    """One node's chunk set, one transaction, retried while the server holds the lock."""

    for attempt in range(BUSY_RETRIES + 1):
        try:
            with store.transaction(immediate=True):
                current = store.connection.execute(
                    f"SELECT n.id, n.content, n.content_fingerprint, n.scope, n.level, "
                    f"({selection.pending}) AS pending FROM nodes n WHERE n.id = ?",
                    (*selection.params, node_id),
                ).fetchone()
                if current is None:
                    raise KeyError(node_id)
                if (not current["pending"] or any(current[key] != source[key] for key in source.keys())
                        or _chunk_ids(store.connection, node_id) != chunk_ids):
                    return 0
                fingerprint = _content_fingerprint(source["content"])
                if fingerprint != source["content_fingerprint"]:
                    store.connection.execute(
                        "UPDATE nodes SET content_fingerprint = ? WHERE id = ?", (fingerprint, node_id)
                    )
                return store.replace_node_chunks(
                    node_id, pairs, content_fingerprint=fingerprint
                )
        except sqlite3.OperationalError as exc:
            if not _is_busy(exc) or attempt == BUSY_RETRIES:
                raise
            summary.busy_retries += 1
            delay = BUSY_RETRY_BASE_DELAY * (2**attempt)
            LOG.warning(
                "database busy writing chunks for %s (attempt %d), retrying in %.2fs",
                node_id,
                attempt + 1,
                delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def run_backfill(
    store: MemoryStore,
    *,
    embedder: ChunkEmbedder,
    embedder_info: EmbedderInfo,
    summary: BackfillSummary,
    batch_nodes: int = DEFAULT_BATCH_NODES,
    limit: int = 0,
    scope: str | None = None,
    level: str | None = None,
    node_id: str | None = None,
    progress_interval: float = DEFAULT_PROGRESS_INTERVAL,
    quiet: bool = False,
) -> BackfillSummary:
    """Chunk and embed every pending node, committing one node at a time.

    Raises whatever the encoder or the database raises: an interrupted run is a
    supported state, not an error to swallow, and the caller reports what was
    committed before the failure.
    """

    if batch_nodes <= 0 or limit < 0:
        raise BackfillError("batch size must be positive and limit nonnegative")
    conn = store.connection
    summary.encoder_id = embedder_info.encoder_id
    selection = ChunkSelection.for_connection(conn, summary.encoder_id, scope, level, node_id)
    summary.embedding_backend = embedder_info.backend
    summary.embedding_model = embedder_info.model
    summary.embedding_dimensions = embedder_info.dimensions
    summary.tokenizer = get_tokenizer().name
    summary.pending_before = selection.count(conn)
    summary.node_embedding_column_present = has_node_column(conn, "embedding")

    if not quiet:
        print(
            f"backfill: {summary.pending_before} pending nodes, "
            f"encoder {summary.embedding_backend} ({summary.embedding_model}, "
            f"dim {summary.embedding_dimensions}), tokenizer {summary.tokenizer}, "
            f"batch {batch_nodes} nodes",
            flush=True,
        )
    reporter = ProgressReporter(
        summary.pending_before, interval=progress_interval, enabled=not quiet
    )
    attempts: dict[str, int] = {}
    after_id: str | None = None
    retry_pass = False
    skipped: set[str] = set()

    try:
        while True:
            if limit and summary.nodes_processed >= limit:
                break
            size = min(batch_nodes, limit - summary.nodes_processed) if limit else batch_nodes
            cursor_clause = "n.id > ? AND " if after_id is not None else ""
            params = (after_id, *selection.params) if after_id is not None else selection.params
            with store.transaction():
                page = conn.execute(
                    f"SELECT n.id, n.content, n.content_fingerprint, n.scope, n.level "
                    f"FROM nodes n WHERE {cursor_clause}{selection.pending} ORDER BY n.id LIMIT ?",
                    (*params, size),
                ).fetchall()
                generations = {row["id"]: _chunk_ids(conn, row["id"]) for row in page}
            if not page:
                if retry_pass:
                    after_id, retry_pass = None, False
                    continue
                break
            after_id = page[-1]["id"]
            planned: list[tuple[sqlite3.Row, list[TextChunk]]] = []
            texts: list[str] = []
            for source in page:
                node_id, content, fingerprint = source[:3]
                if node_id in skipped:
                    continue
                if not fingerprint:
                    summary.nodes_processed += 1
                    summary.nodes_skipped_no_fingerprint += 1
                    skipped.add(node_id)
                    LOG.warning(
                        "node %s has no content_fingerprint; skipping (its chunks "
                        "could never be recognized as current)",
                        node_id,
                    )
                    continue
                chunks = chunk_text(content)
                if not chunks:
                    summary.nodes_processed += 1
                    summary.nodes_skipped_no_chunks += 1
                    skipped.add(node_id)
                    continue
                planned.append((source, chunks))
                texts.extend(chunk.text for chunk in chunks)

            if not planned:
                continue

            vectors = list(embedder(texts))
            if len(vectors) != len(texts):
                raise BackfillError(
                    f"encoder returned {len(vectors)} vectors for {len(texts)} chunk texts"
                )

            if any(not isinstance(v, EncodedVector) or v.encoder_id != summary.encoder_id
                   or len(v) != embedder_info.dimensions for v in vectors):
                raise BackfillError(f"encoder returned vectors outside {summary.encoder_id}")

            cursor = 0
            for source, chunks in planned:
                node_id = source["id"]
                pairs = [
                    (chunk, vectors[cursor + offset])
                    for offset, chunk in enumerate(chunks)
                ]
                cursor += len(chunks)
                summary.zero_vector_chunks += sum(
                    1 for _, vector in pairs if not any(vector)
                )
                try:
                    written = _write_chunks_with_retry(
                        store, node_id, pairs, source, generations[node_id], selection, summary
                    )
                except KeyError:
                    # The node row is gone between the read and the write. Living
                    # Memory normally soft-deletes via nodes.decayed.
                    summary.nodes_processed += 1
                    summary.nodes_vanished += 1
                    skipped.add(node_id)
                    LOG.warning("node %s disappeared mid-run; skipping", node_id)
                    continue
                summary.nodes_processed += 1
                if not written:
                    seen = attempts.get(node_id, 0) + 1
                    attempts[node_id] = seen
                    if seen >= MAX_NODE_ATTEMPTS:
                        skipped.add(node_id)
                        summary.nodes_skipped_attempt_limit += 1
                    else:
                        retry_pass = True
                    continue
                if node_id in attempts:
                    summary.nodes_rechunked_after_content_change += 1
                summary.nodes_written += 1
                summary.chunk_writes += 1
                summary.chunks_written += written
                summary.chunk_bytes_written += sum(
                    len(vector) * CHUNK_EMBEDDING_ITEMSIZE for _, vector in pairs
                )
                summary.max_chunks_in_one_node = max(
                    summary.max_chunks_in_one_node, written
                )
                reporter.update(summary.nodes_processed, summary.chunks_written)
    finally:
        summary.elapsed_seconds = round(time.perf_counter() - reporter.started, 3)
        summary.nodes_per_second = _rate(
            summary.nodes_processed, summary.elapsed_seconds
        )
        summary.chunks_per_second = _rate(
            summary.chunks_written, summary.elapsed_seconds
        )
        summary.pending_after = selection.count(conn)
        summary.chunk_rows_total = store.count_node_chunks()
        summary.chunk_bytes_total = int(
            conn.execute(
                f"SELECT COALESCE(SUM(LENGTH(embedding)), 0) FROM {CHUNK_EMBEDDING_TABLE}"
            ).fetchone()[0]
        )
        summary.node_embedding_column_present = has_node_column(conn, "embedding")
        if not quiet:
            reporter.update(
                summary.nodes_processed, summary.chunks_written, force=True
            )

    summary.complete = summary.pending_after == 0
    return summary


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def verify_report(
    db: Path, *, scope: str | None, level: str | None, encoder_id: str | None = None,
    before_drop: bool = False, node_id: str | None = None,
) -> dict[str, Any]:
    """Read-only coverage, consistency and size report. Safe on the live DB."""

    encoder_id = encoder_id or LocalEmbeddingModel().encoder_id()
    conn = open_readonly(db)
    try:
        conn.execute("BEGIN")
        chunk_table = has_chunk_table(conn)
        embedding_column = has_node_column(conn, "embedding")
        selection = ChunkSelection.for_connection(conn, encoder_id, scope, level, node_id)
        active = conn.execute(
            f"SELECT COUNT(*) FROM nodes n WHERE {selection.active}", selection.filters
        ).fetchone()[0]
        pending = selection.count(conn)
        report: dict[str, Any] = {
            "db": str(db),
            "encoder_id": encoder_id,
            "checked_at": _utc_now_iso(),
            "schema_version": _schema_version(conn),
            "chunk_table_present": chunk_table,
            "node_embedding_column_present": embedding_column,
            "active_chunkable_nodes": active,
            "pending_nodes": pending,
            "covered_nodes": active - pending,
            "coverage": round((active - pending) / active, 6) if active else 1.0,
        }
        if embedding_column:
            row = conn.execute(
                "SELECT COUNT(*) AS rows, COALESCE(SUM(LENGTH(embedding)), 0) AS bytes "
                "FROM nodes WHERE embedding IS NOT NULL"
            ).fetchone()
            report["legacy_embedding_rows"] = int(row["rows"])
            report["legacy_embedding_bytes"] = int(row["bytes"])
        if not chunk_table:
            report["chunk_rows"] = 0
            report["chunk_bytes"] = 0
            report["inconsistent_nodes"] = 0
            report["ok"] = False
            return report

        row = conn.execute(
            f"SELECT COUNT(*) AS rows, COALESCE(SUM(LENGTH(embedding)), 0) AS bytes "
            f"FROM {CHUNK_EMBEDDING_TABLE}"
        ).fetchone()
        report["chunk_rows"] = int(row["rows"])
        report["chunk_bytes"] = int(row["bytes"])
        live = conn.execute(
            f"SELECT COUNT(*) AS rows, COALESCE(SUM(LENGTH(c.embedding)), 0) AS bytes "
            f"FROM {CHUNK_EMBEDDING_TABLE} c JOIN nodes n ON n.id = c.node_id "
            "WHERE n.decayed = 0"
        ).fetchone()
        report["live_chunk_rows"] = int(live["rows"])
        report["live_chunk_bytes"] = int(live["bytes"])
        report["dimensions"] = [
            int(r[0])
            for r in conn.execute(
                f"SELECT DISTINCT dimensions FROM {CHUNK_EMBEDDING_TABLE} "
                "ORDER BY dimensions"
            ).fetchall()
        ]
        report["nodes_with_chunks"] = int(
            conn.execute(
                f"SELECT COUNT(DISTINCT node_id) FROM {CHUNK_EMBEDDING_TABLE}"
            ).fetchone()[0]
        )
        # The invariant a killed run must never break: one node, one generation
        # of chunks, dense ordinals from 0, one vector width.
        # Dropping legacy vectors retains the original corpus-wide integrity gate.
        population = "" if before_drop else f"JOIN nodes n ON n.id=c.node_id WHERE {selection.active}"
        inconsistent = [
            str(r[0])
            for r in conn.execute(
                f"""
                SELECT c.node_id FROM {CHUNK_EMBEDDING_TABLE} c {population}
                GROUP BY c.node_id
                HAVING COUNT(DISTINCT c.content_fingerprint) > 1
                    OR COUNT(DISTINCT dimensions) > 1
                    OR MIN(chunk_index) != 0
                    OR MAX(chunk_index) != COUNT(*) - 1
                ORDER BY node_id
                LIMIT 50
                """, () if before_drop else selection.filters
            ).fetchall()
        ]
        report["inconsistent_nodes"] = len(inconsistent)
        report["inconsistent_node_ids"] = inconsistent
        report["zero_vector_chunks"] = int(
            conn.execute(
                f"SELECT COUNT(*) FROM {CHUNK_EMBEDDING_TABLE} "
                "WHERE embedding = ZEROBLOB(LENGTH(embedding))"
            ).fetchone()[0]
        )
        report["ok"] = (
            pending == 0
            and not inconsistent
            and (not before_drop or len(report["dimensions"]) <= 1)
        )
        return report
    finally:
        conn.close()


def _schema_version(conn: sqlite3.Connection) -> str | None:
    try:
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.DatabaseError:
        return None
    return str(row["value"]) if row else None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _emit(summary_dict: dict[str, Any], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary_dict, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"wrote {path}", flush=True)


def _print_summary(summary: BackfillSummary) -> None:
    mib = summary.chunk_bytes_written / (1024 * 1024)
    total_mib = summary.chunk_bytes_total / (1024 * 1024)
    print("")
    print(f"{summary.mode} summary")
    print(f"  elapsed            {format_duration(summary.elapsed_seconds)} "
          f"({summary.elapsed_seconds:.3f}s)")
    print(f"  encoder ID         {summary.encoder_id}")
    print(f"  pending before     {summary.pending_before}")
    print(f"  pending after      {summary.pending_after}")
    print(f"  nodes processed    {summary.nodes_processed}")
    print(f"  nodes written      {summary.nodes_written}")
    print(f"  chunks written     {summary.chunks_written} "
          f"(max {summary.max_chunks_in_one_node} in one node)")
    print(f"  chunk bytes        {summary.chunk_bytes_written} ({mib:.1f} MiB)")
    print(f"  rate               {summary.nodes_per_second} nodes/s, "
          f"{summary.chunks_per_second} chunks/s")
    if summary.projected_seconds is not None:
        print(f"  projected encode   {format_duration(summary.projected_seconds)} "
              f"at {summary.assumed_rate} texts/s")
    for label, value in (
        ("skipped: no chunks", summary.nodes_skipped_no_chunks),
        ("skipped: no fingerprint", summary.nodes_skipped_no_fingerprint),
        ("skipped: attempt limit", summary.nodes_skipped_attempt_limit),
        ("re-chunked (content changed)", summary.nodes_rechunked_after_content_change),
        ("vanished mid-run", summary.nodes_vanished),
        ("zero-norm vectors", summary.zero_vector_chunks),
        ("busy retries", summary.busy_retries),
    ):
        if value:
            print(f"  {label:<30} {value}")
    if summary.mode != "dry-run":
        print(f"  chunk table now    {summary.chunk_rows_total} rows, "
              f"{summary.chunk_bytes_total} bytes ({total_mib:.1f} MiB)")
        print(f"  nodes.embedding    "
              f"{'present (untouched)' if summary.node_embedding_column_present else 'absent'}")


def cmd_backfill(args: argparse.Namespace) -> int:
    db = args.db.expanduser()
    if not db.is_file():
        raise BackfillError(f"--db does not exist: {db}")

    if args.limit < 0 or min(args.batch_size, args.encode_batch) <= 0:
        raise BackfillError("batch sizes must be positive and limit nonnegative")
    if args.dry_run:
        summary = dry_run(
            db,
            scope=args.scope,
            level=args.level,
            node_id=args.node_id,
            limit=args.limit,
            dimensions=args.dimensions,
            encoder_id=args.encoder_id or LocalEmbeddingModel(model_name=args.model).encoder_id(),
            assumed_rate=args.assumed_rate,
            progress_interval=args.progress_interval,
            quiet=args.quiet,
        )
        _print_summary(summary)
        print("\ndry run: nothing was written", flush=True)
        _emit(summary.as_dict(), args.json)
        return 0

    backup = ensure_backup(
        db,
        backup=args.backup,
        existing_backup=args.existing_backup,
        overwrite=args.overwrite_backup,
        min_fraction=args.min_backup_fraction,
    )
    print(
        f"backup ok: {backup.path} ({backup.bytes} bytes, {backup.backup_nodes} nodes"
        + (f", taken in {format_duration(backup.seconds)}" if backup.created else "")
        + ")",
        flush=True,
    )

    embedder, info = build_chunk_embedder(
        model_name=args.model,
        encode_batch=args.encode_batch,
        allow_fallback=args.allow_fallback_embeddings,
    )
    if args.encoder_id and args.encoder_id != info.encoder_id:
        raise BackfillError(f"requested {args.encoder_id}, encoder resolved to {info.encoder_id}")
    summary = BackfillSummary(
        mode="backfill", db=str(db), started_at=_utc_now_iso(), backup=asdict(backup)
    )
    store = MemoryStore(db)
    try:
        store.connection.execute(f"PRAGMA busy_timeout = {int(args.busy_timeout_ms)}")
        # After the migration, before any chunk write: from here the connection
        # physically cannot touch nodes.embedding.
        install_derived_only_authorizer(store.connection, corpus="chunks")
        try:
            run_backfill(
                store,
                embedder=embedder,
                embedder_info=info,
                summary=summary,
                batch_nodes=args.batch_size,
                limit=args.limit,
                scope=args.scope,
                level=args.level,
                node_id=args.node_id,
                progress_interval=args.progress_interval,
                quiet=args.quiet,
            )
        except BaseException:
            # A failed run is a resumable state, so report what got committed
            # before letting the failure surface.
            _print_summary(summary)
            _emit(summary.as_dict(), args.json)
            raise
        finally:
            store.connection.set_authorizer(None)
    finally:
        store.close()

    _print_summary(summary)
    _emit(summary.as_dict(), args.json)
    if summary.pending_after and not args.limit:
        print(
            f"\n{summary.pending_after} nodes are still pending — re-run to finish "
            "(nodes written by the live server during the run land here).",
            flush=True,
        )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    db = args.db.expanduser()
    if not db.is_file():
        raise BackfillError(f"--db does not exist: {db}")
    report = verify_report(
        db, scope=args.scope, level=args.level, node_id=args.node_id,
        encoder_id=args.encoder_id or LocalEmbeddingModel(model_name=args.model).encoder_id(),
    )
    width = max(len(key) for key in report)
    for key, value in report.items():
        if key == "inconsistent_node_ids" and not value:
            continue
        print(f"{key:<{width}}  {value}")
    _emit(report, args.json)
    return 0 if report.get("ok") else 1


def cmd_drop_column(args: argparse.Namespace) -> int:
    """Drop ``nodes.embedding``. Never reached by the backfill — by design."""

    db = args.db.expanduser()
    if not db.is_file():
        raise BackfillError(f"--db does not exist: {db}")
    if not args.yes:
        raise BackfillError(
            "refusing to drop nodes.embedding without --yes. Run this only after "
            "the chunk-reading code is deployed AND the server has been "
            "restarted: a running old server still reads that column."
        )
    backup = ensure_backup(
        db,
        backup=args.backup,
        existing_backup=args.existing_backup,
        overwrite=args.overwrite_backup,
        min_fraction=args.min_backup_fraction,
    )
    print(f"backup ok: {backup.path} ({backup.bytes} bytes)", flush=True)

    report = verify_report(
        db, scope=None, level=None, before_drop=True,
        encoder_id=args.encoder_id or LocalEmbeddingModel(model_name=args.model).encoder_id(),
    )
    print(
        f"coverage: {report['covered_nodes']}/{report['active_chunkable_nodes']} nodes "
        f"chunked ({report['pending_nodes']} pending, "
        f"{report['inconsistent_nodes']} inconsistent)",
        flush=True,
    )
    if not report["ok"] and not args.force:
        raise BackfillError(
            "refusing to drop nodes.embedding while the chunk table is not "
            f"complete ({report['pending_nodes']} pending, "
            f"{report['inconsistent_nodes']} inconsistent, dimensions "
            f"{report.get('dimensions')}). Finish the backfill first, or pass "
            "--force if you accept losing those nodes from the vector channel."
        )

    store = MemoryStore(db)
    try:
        store.connection.execute(f"PRAGMA busy_timeout = {int(args.busy_timeout_ms)}")
        before = db.stat().st_size
        dropped = store.drop_node_embedding_column()
        if args.vacuum:
            print("VACUUM (rewrites the whole file; needs a maintenance window)", flush=True)
            store.connection.execute("VACUUM")
    finally:
        store.close()
    after = db.stat().st_size
    print(
        f"nodes.embedding {'dropped' if dropped else 'was already absent'}; "
        f"file {before} -> {after} bytes"
        + ("" if args.vacuum else " (run VACUUM to reclaim the freed pages)"),
        flush=True,
    )
    _emit(
        {
            "db": str(db),
            "dropped": dropped,
            "vacuumed": bool(args.vacuum),
            "bytes_before": before,
            "bytes_after": after,
            "backup": asdict(backup),
            "coverage_before_drop": report,
            "finished_at": _utc_now_iso(),
        },
        args.json,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_db_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="SQLite database to operate on. Required and never defaulted: the "
        "live database must be named explicitly, not reached by accident.",
    )


def _add_backup_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("backup (one of these is mandatory)")
    group.add_argument(
        "--backup",
        type=Path,
        default=None,
        help="Take a WAL-correct backup here first, via the sqlite3 backup API.",
    )
    group.add_argument(
        "--existing-backup",
        type=Path,
        default=None,
        help="Use an existing backup, validated (quick_check + node count) first.",
    )
    group.add_argument(
        "--overwrite-backup",
        action="store_true",
        help="Allow --backup to replace an existing file.",
    )
    group.add_argument(
        "--min-backup-fraction",
        type=float,
        default=DEFAULT_MIN_BACKUP_FRACTION,
        help="Reject a backup holding less than this fraction of the source's "
        f"nodes (default {DEFAULT_MIN_BACKUP_FRACTION}).",
    )


def _add_encoder_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model to resolve.")
    parser.add_argument("--encoder-id", help="Target producer ID, e.g. st:MODEL:384. "
                        "Defaults to the resolved encoder. Writes require an exact match.")


def _add_filter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--node-id", help="Select one source by its exact ID.")
    parser.add_argument("--scope", default=None, help="Only nodes in this scope.")
    parser.add_argument(
        "--level",
        default=None,
        choices=("trace", "concept", "schema"),
        help="Only nodes at this level.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backfill_chunk_embeddings.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Log at DEBUG instead of INFO."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "backfill",
        help="Refresh missing, stale or incompatible chunks (resumable, live-safe).",
        description="Encode pending source windows and replace their chunks atomically.",
    )
    _add_db_argument(run)
    _add_backup_arguments(run)
    _add_encoder_arguments(run)
    _add_filter_arguments(run)
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be written and write nothing. Opens the database "
        "read-only, so no backup is required and no migration runs.",
    )
    run.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_NODES,
        help=f"Pending nodes per page/encode call (default {DEFAULT_BATCH_NODES}). "
        "Each node still commits in its own transaction.",
    )
    run.add_argument(
        "--encode-batch",
        type=int,
        default=DEFAULT_ENCODE_BATCH,
        help=f"Texts per encoder forward pass (default {DEFAULT_ENCODE_BATCH}).",
    )
    run.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many nodes (0 = until nothing is pending).",
    )
    run.add_argument(
        "--allow-fallback-embeddings",
        action="store_true",
        help="Proceed even if the model did not load and vectors would come from "
        "the hash fallback. For fixtures only — never for a real database.",
    )
    run.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Override vector width for --dry-run byte estimates only.",
    )
    run.add_argument(
        "--assumed-rate",
        type=float,
        default=DEFAULT_ASSUMED_RATE,
        help="Texts/s used to project a --dry-run duration (default "
        f"{DEFAULT_ASSUMED_RATE}, measured 2026-08-17 on CPU).",
    )
    run.add_argument(
        "--busy-timeout-ms",
        type=int,
        default=DEFAULT_BUSY_TIMEOUT_MS,
        help=f"SQLite busy timeout (default {DEFAULT_BUSY_TIMEOUT_MS}).",
    )
    run.add_argument(
        "--progress-interval",
        type=float,
        default=DEFAULT_PROGRESS_INTERVAL,
        help=f"Seconds between progress lines (default {DEFAULT_PROGRESS_INTERVAL}).",
    )
    run.add_argument("--quiet", action="store_true", help="No progress lines.")
    run.add_argument("--json", type=Path, default=None, help="Write the summary here.")
    run.set_defaults(func=cmd_backfill)

    verify = subparsers.add_parser(
        "verify",
        help="Read-only coverage / consistency / size report.",
        description="Read-only report for active sources in the selected encoder space.",
    )
    _add_db_argument(verify)
    _add_encoder_arguments(verify)
    _add_filter_arguments(verify)
    verify.add_argument("--json", type=Path, default=None, help="Write the report here.")
    verify.set_defaults(func=cmd_verify)

    drop = subparsers.add_parser(
        "drop-embedding-column",
        help="Drop the legacy nodes.embedding column (separate, explicit step).",
        description="Drop nodes.embedding. RUN ONLY AFTER the chunk-reading code "
        "is deployed and the server has been restarted — a running old server "
        "still reads this column. Never invoked by the backfill.",
    )
    _add_db_argument(drop)
    _add_encoder_arguments(drop)
    _add_backup_arguments(drop)
    drop.add_argument(
        "--yes", action="store_true", help="Required acknowledgement of the above."
    )
    drop.add_argument(
        "--force",
        action="store_true",
        help="Drop even if some nodes still have no current chunks.",
    )
    drop.add_argument(
        "--vacuum",
        action="store_true",
        help="VACUUM afterwards to reclaim the freed pages (needs an exclusive "
        "lock and a full file rewrite: maintenance window only).",
    )
    drop.add_argument(
        "--busy-timeout-ms", type=int, default=DEFAULT_BUSY_TIMEOUT_MS,
        help=f"SQLite busy timeout (default {DEFAULT_BUSY_TIMEOUT_MS}).",
    )
    drop.add_argument("--json", type=Path, default=None, help="Write the result here.")
    drop.set_defaults(func=cmd_drop_column)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        return int(args.func(args))
    except BackfillError as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print(
            "\ninterrupted: every committed node has a complete chunk set; "
            "re-run to finish the rest",
            file=sys.stderr,
            flush=True,
        )
        return 130


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
