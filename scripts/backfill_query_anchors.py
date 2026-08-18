#!/usr/bin/env python3
"""Offline, resumable backfill of query anchors from consumed recall history.

What this is for
----------------
Schema v7 gives the graph an entry from query space: ``query_anchors`` is one
row per remembered *question*, ``query_anchor_edges`` its weighted edges to the
nodes a grounded consumption of that question actually used. The live loop
writes those on every content-grounded consumption from the moment it is
deployed. The history that came before it — 9,105 consumed recall events on the
2026-08-18 live database — carries exactly the same signal and no anchors. This
script reads that history, re-labels it with the *same* grounding module the
live loop uses, and writes the anchors it implies. It writes nothing else.

Two subcommands, deliberately separate:

``backfill``
    Create/reinforce anchors and their edges. Additive only — see "Safe under a
    live server" — and it ends with the anchor-edge migration sweep.
``verify``
    Read-only integrity report. Safe on the live database. Exit 0 only when no
    anchor edge is dead, no anchor sits in a scope its source event was not
    asked in, every anchor has a usable vector and at least one edge, and no
    ``(scope, fingerprint)`` is duplicated.

``--until CUTOFF`` is what makes a leak-free evaluation build possible: it
restricts anchor construction to events with ``created_at < CUTOFF``, so a
harness run scored on post-cutoff queries cannot be reading anchors built from
the very events it is being scored on. It is not a convenience flag; a scored
run without it is not evidence.

Safe under a live server
------------------------
The backfill only ever writes rows of ``query_anchors`` and
``query_anchor_edges``. ``nodes``, ``connections`` and ``recall_events`` are
read, never written: after the store is opened (which runs the additive v6 ->
v7 migration) a SQLite *authorizer* is installed that denies every write
outside those two tables, so "it must not touch the node graph" is enforced by
the database rather than by review. ``PRAGMA busy_timeout`` is raised and every
anchor write retries on ``SQLITE_BUSY``, because the MCP server is writing to
the same WAL database throughout.

Resumability
------------
There is no run state on disk: the database itself records progress, and the
marker is the anchor's own clock. Anchors are stamped with their *source
event's* ``created_at`` rather than with the wall clock — which is what makes
the reconstruction honest, an anchor last used in May looking like an anchor
last used in May — so replaying history oldest-first makes an anchor's
``last_matched_at`` the timestamp of the last event it absorbed.

An event is therefore *pending* unless the anchor its (query, scope) resolves
to — by fingerprint first, then by the same cosine dedup the write path uses —
already carries an edge to every live node the event grounded on **and** is
already fresh as of that event. Both halves are needed: the edge test alone
would re-reinforce every repeat on every re-run, and the clock test alone would
miss edges an earlier, differently-grounded consumption never wrote. Together
they make a resumed run land on exactly the anchors, weights and hit counts an
uninterrupted run produces, and make a re-run after a complete run write
nothing at all — including when the live path has meanwhile anchored the same
questions itself.

One event is applied as one anchor upsert followed by one statement per edge,
each committed by ``storage.py``. A kill can therefore land *inside* an event's
edge set, never inside an edge: the worst state it can leave is an anchor whose
edge set is short and whose clock has moved, which the predicate above still
reports as pending (the edge half fails), so the next run completes it. What a
kill cannot do is duplicate an anchor or invent an edge to something the event
never grounded on.

Backup is mandatory
-------------------
``backfill`` refuses to start without one, and ``--backup`` takes it through
``sqlite3.Connection.backup()``. That API is WAL-correct: it copies the
committed database as one consistent snapshot while a server keeps writing.
``cp global.sqlite3 backup.sqlite3`` is **not** a valid backup here — the live
database carries a ~500 MB WAL holding the newest transactions, and a copy
without ``-wal`` silently loses them.

``--dry-run`` needs no backup: it opens the database read-only (``mode=ro`` +
``PRAGMA query_only``), never constructs a ``MemoryStore`` (whose constructor
migrates, i.e. writes), and reports what a real run would write.

Usage
-----
    # rehearse, read-only, no backup needed
    python3 scripts/backfill_query_anchors.py backfill --db DB --dry-run

    # real run, taking its own WAL-correct backup first
    python3 scripts/backfill_query_anchors.py backfill --db DB \\
        --backup /var/backups/lm/global-pre-anchors.sqlite3 --json summary.json

    # leak-free evaluation build, on a snapshot
    python3 scripts/backfill_query_anchors.py backfill --db SNAPSHOT \\
        --until 2026-08-01T00:00:00Z --backup /tmp/snapshot-pre-anchors.sqlite3

    python3 scripts/backfill_query_anchors.py verify --db DB --json verify.json
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import partial
import json
import logging
import math
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Callable
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_SRC))

from living_memory import query_anchors as qa  # noqa: E402  (path bootstrap above)
from living_memory.embeddings import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    LocalEmbeddingModel,
)
from living_memory.grounding import ground_results  # noqa: E402
from living_memory.query_anchors import (  # noqa: E402
    ANCHOR_DEDUP_COSINE_THRESHOLD,
    ANCHOR_EDGE_WEIGHT,
    MAX_SUPERSEDES_CHAIN_DEPTH,
    # Private on purpose: "where does an edge to this node actually go" is
    # anchor policy — it follows `supersedes` and refuses dead targets — and
    # the resume predicate has to ask exactly the question the write path
    # answers. A second copy here would drift from it silently.
    _live_target,
)
from living_memory.scope import normalize_scope  # noqa: E402
from living_memory.storage import (  # noqa: E402
    QUERY_ANCHOR_EDGE_TABLE,
    QUERY_ANCHOR_TABLE,
    MemoryStore,
    recall_fingerprint,
)
from living_memory.temporal import parse_timestamp  # noqa: E402

# The labeling threshold is imported from the live credit-assignment loop, not
# restated: the whole point of the retro pass is that history is labeled the
# way the live path labels the present.
from living_memory.feedback import RECALL_CREDIT_MIN_CONTAINMENT  # noqa: E402

LOG = logging.getLogger("backfill_query_anchors")

#: One encoder call per batch of queries, so the encoder's batch dimension is
#: not spent one short text at a time. Each event still commits on its own.
DEFAULT_BATCH_EVENTS = 200

#: Texts per forward pass inside the encoder. Queries are short, so this is
#: larger than the chunk backfill's 32.
DEFAULT_ENCODE_BATCH = 64

#: Only used to project a duration in ``--dry-run``; the real run measures its
#: own rate.
DEFAULT_ASSUMED_RATE = 200.0

#: The live server writes to the same database. Wait for the lock rather than
#: failing the run; MemoryStore's own default (5 s) is tuned for short requests.
DEFAULT_BUSY_TIMEOUT_MS = 30_000
BUSY_RETRIES = 6
BUSY_RETRY_BASE_DELAY = 0.25

DEFAULT_PROGRESS_INTERVAL = 5.0

#: A backup with fewer nodes than this fraction of the source is rejected: that
#: is what a ``cp`` without the ``-wal`` looks like from the outside.
DEFAULT_MIN_BACKUP_FRACTION = 0.99

#: How many offending ids a report lists before it stops.
MAX_REPORTED_EXAMPLES = 20

_SQLITE_BUSY_CODES = frozenset({5, 6})  # SQLITE_BUSY, SQLITE_LOCKED

#: The only two tables this process may write. Everything else — the node
#: graph, the connections, the recall history it reads — is read-only for the
#: rest of the run, enforced by the authorizer below.
WRITABLE_TABLES = frozenset({QUERY_ANCHOR_TABLE, QUERY_ANCHOR_EDGE_TABLE})

_WRITE_ACTIONS_TABLE_ARG = frozenset(
    {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}
)
_ALWAYS_DENIED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_DROP_VIEW,
    }
)

#: ``Sequence[str] -> one vector per text``. Same shape as
#: ``storage.ChunkEmbedder``; queries are single texts, so there is one vector
#: per anchor rather than one per window.
QueryEmbedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


class BackfillError(RuntimeError):
    """An operator-facing refusal: printed as one line, exit code 2."""


# ---------------------------------------------------------------------------
# Source events
#
# The unit of work is one *consumed* recall event: a delivered recall that a
# later trace linked to itself (`feedback_applied = 1` with a
# `feedback_trace_id`). That trace is the evidence of what the query was for,
# and grounding it against the event's results is what turns a delivery into a
# label. An unconsumed event has no such evidence and is not a training signal.
# ---------------------------------------------------------------------------

_CONSUMED_CLAUSES = (
    "e.feedback_applied = 1",
    "e.feedback_trace_id IS NOT NULL",
    "TRIM(e.query) != ''",
)


@dataclass(frozen=True)
class SourceEvent:
    """One consumed recall event, with its consuming trace's content."""

    id: str
    query: str
    scope: str
    created_at: str
    trace_id: str
    trace_content: str
    result_ids: tuple[str, ...]


def consumed_where(*, until: str | None, scope: str | None) -> tuple[str, list[Any]]:
    """``WHERE`` body selecting the consumed events in range."""

    clauses = list(_CONSUMED_CLAUSES)
    params: list[Any] = []
    if until is not None:
        # Strictly before: the cutoff instant belongs to the evaluated side.
        clauses.append("e.created_at < ?")
        params.append(until)
    if scope is not None:
        clauses.append("e.scope = ?")
        params.append(scope)
    return " AND ".join(clauses), params


def count_consumed(
    conn: sqlite3.Connection, *, until: str | None, scope: str | None
) -> int:
    where, params = consumed_where(until=until, scope=scope)
    row = conn.execute(
        f"SELECT COUNT(*) FROM recall_events e WHERE {where}", params
    ).fetchone()
    return int(row[0])


def iter_source_events(
    conn: sqlite3.Connection,
    *,
    until: str | None,
    scope: str | None,
    page_size: int = DEFAULT_BATCH_EVENTS,
) -> Iterator[list[SourceEvent]]:
    """Yield pages of consumed events, oldest first, with their trace content.

    Keyset pagination on ``(created_at, id)`` rather than one long-lived
    cursor: the caller commits anchor writes between pages on this same
    connection, and a streaming read held open across those commits is a
    needless thing to reason about. The key is the same tuple the loop is
    ordered by, so a page boundary cannot skip or repeat an event.

    Events whose consuming trace no longer exists are dropped here: without the
    trace there is nothing to ground against.
    """

    where, base_params = consumed_where(until=until, scope=scope)
    last_created = ""
    last_id = ""
    while True:
        rows = conn.execute(
            f"""
            SELECT e.id, e.query, e.scope, e.created_at, e.feedback_trace_id,
                   e.results, t.content AS trace_content
            FROM recall_events e
            LEFT JOIN nodes t ON t.id = e.feedback_trace_id
            WHERE {where}
              AND (e.created_at > ? OR (e.created_at = ? AND e.id > ?))
            ORDER BY e.created_at ASC, e.id ASC
            LIMIT ?
            """,
            [*base_params, last_created, last_created, last_id, int(page_size)],
        ).fetchall()
        if not rows:
            return
        last_created = str(rows[-1]["created_at"])
        last_id = str(rows[-1]["id"])
        page: list[SourceEvent] = []
        for row in rows:
            trace_id = str(row["feedback_trace_id"] or "")
            content = row["trace_content"]
            if not trace_id or content is None:
                LOG.debug("event %s has no consuming trace row; skipping", row["id"])
                continue
            page.append(
                SourceEvent(
                    id=str(row["id"]),
                    query=str(row["query"] or ""),
                    scope=str(row["scope"] or "global"),
                    created_at=str(row["created_at"] or ""),
                    trace_id=trace_id,
                    trace_content=str(content),
                    result_ids=_parse_result_ids(row["results"], trace_id),
                )
            )
        yield page


def _parse_result_ids(raw: Any, trace_id: str) -> tuple[str, ...]:
    """Node ids of one event's delivered results, in rank order.

    The consuming trace itself is dropped when it appears among its own
    results, exactly as ``feedback.apply_pending_recall_feedback`` drops it: a
    trace grounding on itself is not evidence of anything.
    """

    try:
        items = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return ()
    if not isinstance(items, list):
        return ()
    ids: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            continue
        node_id = str(item.get("node_id") or "")
        if not node_id or node_id == trace_id or node_id in seen:
            continue
        seen.add(node_id)
        ids.append(node_id)
    return tuple(ids)


def fetch_node_rows(
    conn: sqlite3.Connection, node_ids: Sequence[str]
) -> dict[str, tuple[str, bool]]:
    """``node_id -> (content, decayed)`` for the ids that still exist."""

    found: dict[str, tuple[str, bool]] = {}
    ids = list(dict.fromkeys(node_ids))
    for start in range(0, len(ids), 500):
        window = ids[start : start + 500]
        placeholders = ",".join("?" for _ in window)
        rows = conn.execute(
            f"SELECT id, content, decayed FROM nodes WHERE id IN ({placeholders})",
            window,
        ).fetchall()
        for row in rows:
            found[str(row["id"])] = (str(row["content"] or ""), bool(row["decayed"]))
    return found


def grounded_target_ids(
    event: SourceEvent,
    contents: Mapping[str, str],
    *,
    min_containment: float,
) -> list[str]:
    """The event's results the consuming trace demonstrably used, in rank order.

    ``grounding.ground_results`` is called per event, with the trace and that
    event's results as the IDF corpus — the same call, on the same documents,
    that ``feedback.apply_pending_recall_feedback`` makes when it decides who
    earns credit. Reproducing the live verdict is the point, so the corpus is
    not widened to the whole history even though this pass could afford it.

    One honest caveat: a node's content today is not necessarily the content
    that was delivered back then. History cannot record what was edited since,
    so a retro label is the label the *current* content earns.
    """

    result_contents = {
        node_id: contents[node_id]
        for node_id in event.result_ids
        if node_id in contents
    }
    if not result_contents:
        return []
    verdicts = ground_results(
        event.trace_content, result_contents, min_containment=min_containment
    )
    return [
        node_id
        for node_id in event.result_ids
        if (verdict := verdicts.get(node_id)) is not None and verdict.grounded
    ]


def parse_cutoff(value: str | None) -> str | None:
    """Normalize ``--until`` to the timestamp shape ``recall_events`` stores.

    Every ``created_at`` in that table is ``YYYY-MM-DDTHH:MM:SSZ`` (UTC, second
    precision, fixed width), so once the cutoff is in the same shape the
    comparison can be an indexed string comparison. A bare date is accepted and
    means midnight UTC.
    """

    if value is None:
        return None
    moment = parse_timestamp(value)
    if moment is None:
        raise BackfillError(
            f"--until is not a timestamp I can parse: {value!r}. Use an ISO-8601 "
            "instant (2026-08-01T00:00:00Z) or a bare date (2026-08-01)."
        )
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Connections, schema shape, backups
# ---------------------------------------------------------------------------


def open_readonly(path: Path) -> sqlite3.Connection:
    """Open a database that must not be written to, and make that structural.

    ``mode=ro`` refuses writes at the VFS layer and ``query_only`` refuses them
    at the statement layer. Nothing here constructs a ``MemoryStore``: its
    constructor runs migrations, which is a write.

    ``immutable=1`` is deliberately not used: it makes SQLite ignore the
    ``-wal`` entirely, which on the live database means reading a stale
    snapshot — the very failure mode ``cp``-without-``-wal`` produces.
    """

    uri = f"file:{quote(str(path.resolve()), safe='/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = 1")
    return conn


def has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def has_anchor_tables(conn: sqlite3.Connection) -> bool:
    return all(has_table(conn, name) for name in sorted(WRITABLE_TABLES))


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


def install_anchor_only_authorizer(conn: sqlite3.Connection) -> None:
    """Let the database enforce "anchor tables only" for the rest of this process.

    This script reads the whole graph — nodes, connections, recall events — to
    decide what an anchor should point at, and a bug that turned one of those
    reads into a write would be a mutation of the live memory itself, not of a
    derived table. Rather than trusting that no code path here writes outside
    the anchor tables, deny it at the connection: reads stay open, writes are
    allowed on ``query_anchors``/``query_anchor_edges`` and nowhere else, and
    DDL is refused outright. Install *after* ``MemoryStore.__init__`` has run
    its migrations — the v7 ``CREATE TABLE`` is a legitimate write and must not
    be blocked.
    """

    def authorizer(action: int, arg1: str | None, arg2: str | None, *_rest: Any) -> int:
        if action in _ALWAYS_DENIED_ACTIONS:
            return sqlite3.SQLITE_DENY
        if action in _WRITE_ACTIONS_TABLE_ARG and arg1 not in WRITABLE_TABLES:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    conn.set_authorizer(authorizer)


# ---------------------------------------------------------------------------
# Read-only replacement resolution
#
# `query_anchors.resolve_replacement` needs a MemoryStore, whose constructor
# migrates — i.e. writes. `--dry-run` and `verify` must be safe against the
# live database, so they walk the same `supersedes` chain over a read-only
# connection instead. The contract is that walk's, restated in SQL: climb from
# target to source, only through *live* superseders, stop on a cycle or at
# MAX_SUPERSEDES_CHAIN_DEPTH, and return nothing when nothing supersedes.
# ---------------------------------------------------------------------------


def read_only_replacement(
    conn: sqlite3.Connection, node_id: str, *, cache: dict[str, str | None]
) -> str | None:
    """The live node that currently replaces ``node_id``, or ``None``."""

    if node_id in cache:
        return cache[node_id]
    current = str(node_id)
    seen = {current}
    for _ in range(MAX_SUPERSEDES_CHAIN_DEPTH):
        row = conn.execute(
            """
            SELECT c.source_id FROM connections c
            JOIN nodes n ON n.id = c.source_id
            WHERE c.target_id = ? AND c.type = 'supersedes'
              AND c.source_id != ? AND n.decayed = 0
            ORDER BY c.weight DESC, c.created_at DESC
            LIMIT 1
            """,
            (current, current),
        ).fetchone()
        if row is None:
            break
        successor = str(row[0])
        if successor in seen:
            break
        seen.add(successor)
        current = successor
    result = None if current == str(node_id) else current
    cache[str(node_id)] = result
    return result


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


@dataclass
class EmbedderInfo:
    backend: str
    model: str
    dimensions: int


def build_query_embedder(
    *, model_name: str, encode_batch: int, allow_fallback: bool
) -> tuple[QueryEmbedder, EmbedderInfo]:
    """Return a batch encoder for query texts, plus what it actually resolved to.

    An anchor vector has to live in the same space as the node chunks it will
    be compared against, so the model is loaded through the same resolver every
    other caller uses (identical model, identical normalization) and its loaded
    encoder is used directly when there is one — ``LocalEmbeddingModel.embed``
    encodes one text per call, which wastes the batch dimension on a few
    thousand queries.

    Refuses to write hash-fallback vectors into a real database unless
    ``allow_fallback`` says so: filling ``query_anchors`` with vectors from a
    different embedding space would make every anchor unmatchable, and it is
    invisible afterwards. The test suite runs with
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
                "anchor vector would come from the hash fallback — a different "
                "vector space from the one recall uses, in which no anchor would "
                "ever match. Fix the model cache, or pass "
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
            return vectors.tolist()

        backend = "sentence-transformers"

    return embed, EmbedderInfo(
        backend=backend, model=model_name, dimensions=int(model.dimensions)
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

    def update(self, events: int, anchors: int, edges: int, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if not force and now - self._last < self.interval:
            return
        self._last = now
        elapsed = max(now - self.started, 1e-9)
        rate = events / elapsed
        remaining = max(self.total - events, 0)
        eta = remaining / rate if rate > 0 else float("inf")
        percent = (100.0 * events / self.total) if self.total else 100.0
        print(
            f"  {events}/{self.total} events ({percent:5.1f}%)  "
            f"{anchors} anchors  {edges} edges  {rate:6.1f} events/s  "
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
    until: str | None = None
    scope: str | None = None
    min_containment: float = RECALL_CREDIT_MIN_CONTAINMENT
    elapsed_seconds: float = 0.0
    events_total: int = 0
    events_processed: int = 0
    events_grounded: int = 0
    events_applied: int = 0
    events_ungrounded: int = 0
    events_already_anchored: int = 0
    events_without_live_target: int = 0
    events_skipped_bad_scope: int = 0
    events_skipped_zero_vector: int = 0
    results_without_node: int = 0
    anchors_created: int = 0
    anchors_reinforced: int = 0
    anchors_merged_by_cosine: int = 0
    edge_writes: int = 0
    grounded_targets: int = 0
    targets_skipped_dead: int = 0
    busy_retries: int = 0
    events_per_second: float = 0.0
    embedding_backend: str = ""
    embedding_model: str = ""
    embedding_dimensions: int = 0
    anchors_total: int = 0
    anchor_edges_total: int = 0
    projected_anchors: int = 0
    projected_edges: int = 0
    targets_missing: int = 0
    targets_decayed_with_replacement: int = 0
    targets_decayed_without_replacement: int = 0
    anchor_tables_present: bool = False
    sweep: dict[str, Any] = field(default_factory=dict)
    complete: bool = False
    backup: dict[str, Any] | None = None
    projected_seconds: float | None = None
    assumed_rate: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rate(count: int, elapsed: float) -> float:
    return round(count / elapsed, 2) if elapsed > 0 else 0.0


def _normalized_scope(event: SourceEvent) -> str | None:
    """The event's scope as an anchor would store it, or ``None`` if unusable.

    ``normalize_scope`` raises on a prefix it does not know. A single such row
    in five months of history must not end a twelve-minute run, so it is
    counted and skipped instead.
    """

    try:
        return normalize_scope(event.scope)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def dry_run(
    db: Path,
    *,
    until: str | None,
    scope: str | None,
    limit: int,
    min_containment: float,
    assumed_rate: float,
    progress_interval: float,
    page_size: int,
    quiet: bool,
) -> BackfillSummary:
    """Report what a real run would write, touching nothing.

    Grounds every event for real — the labeling is the cheap half and the whole
    projection depends on it — and projects the anchor count as the number of
    distinct ``(scope, fingerprint)`` identities among grounded events. That is
    an *upper* bound: cosine dedup can only merge further, and it cannot be
    evaluated without the encoder (which would need a model load and, on a
    pre-v7 file, an anchor table to scan).
    """

    summary = BackfillSummary(
        mode="dry-run",
        db=str(db),
        started_at=_utc_now_iso(),
        until=until,
        scope=scope,
        min_containment=min_containment,
    )
    started = time.perf_counter()
    conn = open_readonly(db)
    try:
        summary.anchor_tables_present = has_anchor_tables(conn)
        if summary.anchor_tables_present:
            summary.anchors_total = int(
                conn.execute(f"SELECT COUNT(*) FROM {QUERY_ANCHOR_TABLE}").fetchone()[0]
            )
            summary.anchor_edges_total = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM {QUERY_ANCHOR_EDGE_TABLE}"
                ).fetchone()[0]
            )
        summary.embedding_dimensions = EMBEDDING_DIMENSIONS
        summary.events_total = count_consumed(conn, until=until, scope=scope)
        if not quiet:
            print(
                f"dry run: {summary.events_total} consumed recall events"
                + (f" before {until}" if until else "")
                + (f" in scope {scope}" if scope else "")
                + f", labeling at containment >= {min_containment}",
                flush=True,
            )
        reporter = ProgressReporter(
            summary.events_total, interval=progress_interval, enabled=not quiet
        )
        identities: set[tuple[str, str]] = set()
        edges: set[tuple[str, str, str]] = set()
        replacement_cache: dict[str, str | None] = {}

        for page in iter_source_events(
            conn, until=until, scope=scope, page_size=page_size
        ):
            if limit and summary.events_processed >= limit:
                break
            if limit:
                page = page[: max(limit - summary.events_processed, 0)]
            rows = fetch_node_rows(
                conn, [node_id for event in page for node_id in event.result_ids]
            )
            contents = {node_id: content for node_id, (content, _d) in rows.items()}
            for event in page:
                summary.events_processed += 1
                summary.results_without_node += sum(
                    1 for node_id in event.result_ids if node_id not in rows
                )
                anchor_scope = _normalized_scope(event)
                if anchor_scope is None:
                    summary.events_skipped_bad_scope += 1
                    continue
                grounded = grounded_target_ids(
                    event, contents, min_containment=min_containment
                )
                if not grounded:
                    summary.events_ungrounded += 1
                    continue
                summary.events_grounded += 1
                summary.grounded_targets += len(grounded)
                fingerprint = recall_fingerprint(event.query, anchor_scope)
                live_here = 0
                for node_id in grounded:
                    # Grounded ids always have a row: a result whose node is
                    # gone was never graded. Only liveness can disqualify one
                    # here, and only when nothing has replaced it.
                    _content, decayed = rows[node_id]
                    target = node_id
                    if decayed:
                        replacement = read_only_replacement(
                            conn, node_id, cache=replacement_cache
                        )
                        if replacement is None:
                            summary.targets_decayed_without_replacement += 1
                            continue
                        summary.targets_decayed_with_replacement += 1
                        target = replacement
                    live_here += 1
                    edges.add((anchor_scope, fingerprint, target))
                if live_here:
                    identities.add((anchor_scope, fingerprint))
                else:
                    summary.events_without_live_target += 1
            reporter.update(
                summary.events_processed, len(identities), len(edges)
            )
        summary.projected_anchors = len(identities)
        summary.projected_edges = len(edges)
        if not quiet:
            reporter.update(
                summary.events_processed, len(identities), len(edges), force=True
            )
    finally:
        conn.close()

    summary.elapsed_seconds = round(time.perf_counter() - started, 3)
    summary.events_per_second = _rate(summary.events_processed, summary.elapsed_seconds)
    summary.assumed_rate = assumed_rate
    summary.projected_seconds = (
        round(summary.events_grounded / assumed_rate, 1) if assumed_rate > 0 else None
    )
    summary.complete = True
    return summary


# ---------------------------------------------------------------------------
# Real run
# ---------------------------------------------------------------------------


def _is_busy(exc: sqlite3.OperationalError) -> bool:
    code = getattr(exc, "sqlite_errorcode", None)
    if code is not None:
        return code in _SQLITE_BUSY_CODES
    message = str(exc).lower()
    return "locked" in message or "busy" in message


def _with_busy_retry(
    what: str, summary: BackfillSummary, call: Callable[[], Any]
) -> Any:
    """Run ``call``, retrying with backoff while the live server holds the lock."""

    for attempt in range(BUSY_RETRIES + 1):
        try:
            return call()
        except sqlite3.OperationalError as exc:
            if not _is_busy(exc) or attempt == BUSY_RETRIES:
                raise
            summary.busy_retries += 1
            delay = BUSY_RETRY_BASE_DELAY * (2**attempt)
            LOG.warning(
                "database busy %s (attempt %d), retrying in %.2fs",
                what,
                attempt + 1,
                delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def _resolve_anchor(
    store: MemoryStore,
    *,
    anchor_scope: str,
    fingerprint: str,
    vector: Sequence[float],
    dedup_threshold: float,
) -> Any:
    """The anchor ``upsert_anchor`` would land on, without writing anything.

    The two stages of ``query_anchors.upsert_anchor``'s dedup, asked as a
    question instead of as a write: the indexed ``UNIQUE(scope, fingerprint)``
    probe first, then the scoped cosine scan at the same threshold. The resume
    predicate has to be the write path's own idea of identity — a predicate
    that only knew about fingerprints would re-apply every cosine-merged event
    on every re-run.
    """

    anchor = store.find_query_anchor(anchor_scope, fingerprint)
    if anchor is not None:
        return anchor
    matches = qa.match_anchors(
        store,
        vector,
        anchor_scope,
        limit=1,
        min_similarity=dedup_threshold,
        include_targets=False,
    )
    return matches[0].anchor if matches else None


def run_backfill(
    store: MemoryStore,
    *,
    embedder: QueryEmbedder,
    embedder_info: EmbedderInfo,
    summary: BackfillSummary,
    until: str | None = None,
    scope: str | None = None,
    limit: int = 0,
    page_size: int = DEFAULT_BATCH_EVENTS,
    min_containment: float = RECALL_CREDIT_MIN_CONTAINMENT,
    edge_weight: float = ANCHOR_EDGE_WEIGHT,
    dedup_threshold: float = ANCHOR_DEDUP_COSINE_THRESHOLD,
    progress_interval: float = DEFAULT_PROGRESS_INTERVAL,
    quiet: bool = False,
) -> BackfillSummary:
    """Build anchors from every grounded consumption in range, oldest first.

    Oldest first is not cosmetic: an anchor's identity is the first phrasing
    that created it and every later near-miss reinforces that one, so replaying
    history in its own order reproduces the anchor set the live path would have
    grown, rather than an order-dependent variant of it.

    Raises whatever the encoder or the database raises: an interrupted run is a
    supported state, not an error to swallow, and the caller reports what was
    committed before the failure.
    """

    conn = store.connection
    summary.embedding_backend = embedder_info.backend
    summary.embedding_model = embedder_info.model
    summary.embedding_dimensions = embedder_info.dimensions
    summary.until = until
    summary.scope = scope
    summary.min_containment = min_containment
    summary.anchor_tables_present = True
    summary.events_total = count_consumed(conn, until=until, scope=scope)

    if not quiet:
        print(
            f"backfill: {summary.events_total} consumed recall events"
            + (f" before {until}" if until else "")
            + (f" in scope {scope}" if scope else "")
            + f", encoder {summary.embedding_backend} ({summary.embedding_model}, "
            f"dim {summary.embedding_dimensions}), batch {page_size} events",
            flush=True,
        )
    reporter = ProgressReporter(
        summary.events_total, interval=progress_interval, enabled=not quiet
    )

    try:
        for page in iter_source_events(
            conn, until=until, scope=scope, page_size=page_size
        ):
            if limit and summary.events_processed >= limit:
                break
            if limit:
                page = page[: max(limit - summary.events_processed, 0)]

            rows = fetch_node_rows(
                conn, [node_id for event in page for node_id in event.result_ids]
            )
            contents = {node_id: content for node_id, (content, _d) in rows.items()}

            # Label the whole page first, then encode the queries of the events
            # that survived labeling in one call: the encoder amortizes across
            # texts, and an ungrounded event must cost no encoding at all.
            planned: list[tuple[SourceEvent, str, list[str]]] = []
            for event in page:
                summary.events_processed += 1
                anchor_scope = _normalized_scope(event)
                if anchor_scope is None:
                    summary.events_skipped_bad_scope += 1
                    LOG.warning(
                        "recall event %s has an unusable scope %r; skipping",
                        event.id,
                        event.scope,
                    )
                    continue
                grounded = grounded_target_ids(
                    event, contents, min_containment=min_containment
                )
                if not grounded:
                    summary.events_ungrounded += 1
                    continue
                summary.events_grounded += 1
                summary.grounded_targets += len(grounded)
                planned.append((event, anchor_scope, grounded))

            if not planned:
                reporter.update(
                    summary.events_processed,
                    summary.anchors_created,
                    summary.edge_writes,
                )
                continue

            vectors = list(embedder([event.query for event, _s, _g in planned]))
            if len(vectors) != len(planned):
                raise BackfillError(
                    f"encoder returned {len(vectors)} vectors for {len(planned)} queries"
                )

            for (event, anchor_scope, grounded), vector in zip(planned, vectors):
                if not any(vector):
                    # A zero vector has no direction, so the anchor could never
                    # be matched and could never be deduplicated against.
                    summary.events_skipped_zero_vector += 1
                    LOG.warning(
                        "query of event %s encoded to a zero vector; skipping",
                        event.id,
                    )
                    continue
                live_targets = {
                    target
                    for target in (
                        _live_target(store, node_id) for node_id in grounded
                    )
                    if target
                }
                if not live_targets:
                    # Every node this event used is gone with nothing replacing
                    # it. Writing the anchor anyway would create one with no
                    # edges — an anchor that answers nothing, and a `verify`
                    # failure.
                    summary.events_without_live_target += 1
                    summary.targets_skipped_dead += len(grounded)
                    continue

                anchor = _resolve_anchor(
                    store,
                    anchor_scope=anchor_scope,
                    fingerprint=recall_fingerprint(event.query, anchor_scope),
                    vector=vector,
                    dedup_threshold=dedup_threshold,
                )
                stamp = event.created_at
                if anchor is not None:
                    # The anchor's own clock is the progress marker: because a
                    # reinforcement stamps it with the applied event's time and
                    # this loop runs oldest-first, an anchor whose freshness is
                    # already at or past this event has already absorbed it.
                    # That is what makes a resumed run land on exactly the same
                    # anchors, weights and hit counts as an uninterrupted one,
                    # with no state outside the database.
                    reference = anchor.last_matched_at or anchor.first_seen or ""
                    covered = live_targets <= {
                        edge.target_id
                        for edge in store.list_query_anchor_edges(anchor_id=anchor.id)
                    }
                    if covered and reference >= event.created_at:
                        summary.events_already_anchored += 1
                        continue
                    # Never move freshness backwards. History being replayed
                    # into an anchor the *live* path already reinforced must
                    # add the missing edges without making a live anchor look
                    # months stale. On a purely historical range this is always
                    # the event's own time.
                    stamp = max(stamp, reference)

                result = _with_busy_retry(
                    f"writing the anchor for event {event.id}",
                    summary,
                    partial(
                        qa.upsert_anchor,
                        store,
                        event.query,
                        anchor_scope,
                        vector,
                        grounded,
                        stamp,
                        edge_weight=edge_weight,
                        dedup_threshold=dedup_threshold,
                    ),
                )
                summary.events_applied += 1
                if result.created:
                    summary.anchors_created += 1
                else:
                    summary.anchors_reinforced += 1
                    if result.matched_by == "cosine":
                        summary.anchors_merged_by_cosine += 1
                summary.edge_writes += len(result.edges)
                summary.targets_skipped_dead += len(result.skipped_targets)
                reporter.update(
                    summary.events_processed,
                    summary.anchors_created,
                    summary.edge_writes,
                )
    finally:
        summary.elapsed_seconds = round(time.perf_counter() - reporter.started, 3)
        summary.events_per_second = _rate(
            summary.events_processed, summary.elapsed_seconds
        )
        summary.anchors_total = store.count_query_anchors()
        summary.anchor_edges_total = store.count_query_anchor_edges()
        if not quiet:
            reporter.update(
                summary.events_processed,
                summary.anchors_created,
                summary.edge_writes,
                force=True,
            )

    # Complete means "the whole range was walked", which is what makes a re-run
    # a no-op. A --limit that actually bit leaves work behind.
    summary.complete = not limit or summary.events_processed >= summary.events_total
    return summary


def run_edge_migration_sweep(
    store: MemoryStore, summary: BackfillSummary, *, quiet: bool = False
) -> None:
    """Re-point every anchor edge whose target has since been replaced.

    Historical consumptions used nodes that have been superseded or
    era-displaced in the months since — measured at ~4.6% of the projected
    edges. The write path resolves a target before writing the edge, so the
    edges this run just wrote are already current; the sweep exists for the
    ones an *earlier* run wrote before their target was replaced, and for a
    database written before the ``supersedes`` hook existed. Idempotent: a
    second pass moves nothing.
    """

    sweep = _with_busy_retry(
        "migrating anchor edges", summary, lambda: qa.sweep_anchor_edge_migration(store)
    )
    summary.sweep = {
        "targets_examined": sweep.targets_examined,
        "targets_migrated": sweep.targets_migrated,
        "edges_moved": sweep.edges_moved,
        "migrations": [list(pair) for pair in sweep.migrations[:MAX_REPORTED_EXAMPLES]],
    }
    summary.anchor_edges_total = store.count_query_anchor_edges()
    if not quiet:
        print(
            f"edge migration sweep: {sweep.targets_examined} targets examined, "
            f"{sweep.targets_migrated} replaced, {sweep.edges_moved} edges moved",
            flush=True,
        )


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def verify_report(db: Path, *, max_examples: int = MAX_REPORTED_EXAMPLES) -> dict[str, Any]:
    """Read-only integrity report. Safe on the live database.

    Exit 0 requires all four of the properties an anchor corpus is worth
    nothing without:

    * **no dead edges** — no edge into a node that does not exist, and none
      into a decayed node that *has* a live replacement reachable through
      ``supersedes``. An edge into a decayed node with nothing replacing it is
      not a defect: the node simply aged out, the edge is inert, and a later
      match just finds one fewer target. An edge into a superseded node is a
      defect, because it keeps steering a repeated question into the past.
    * **no cross-scope anchor** — every anchor's ``(scope, fingerprint)`` is
      the identity of some consumed recall event *in that scope*. Since
      ``fingerprint = recall_fingerprint(query, scope)``, that equality is
      exactly "this anchor's scope is its source event's scope".
    * **every anchor usable** — a vector of the recorded width, not all zeros,
      and at least one edge. An anchor with no edges matches queries and offers
      nothing.
    * **no duplicate identity** — ``UNIQUE(scope, fingerprint)`` guarantees it
      on a v7 file; the check is here because a partially-migrated database is
      the case where the guarantee is missing.
    """

    conn = open_readonly(db)
    try:
        report: dict[str, Any] = {
            "db": str(db),
            "checked_at": _utc_now_iso(),
            "schema_version": _schema_version(conn),
            "anchor_tables_present": has_anchor_tables(conn),
        }
        if not report["anchor_tables_present"]:
            report["ok"] = False
            report["reason"] = (
                "query_anchors / query_anchor_edges are absent: this database is "
                "pre-v7, so the backfill has not run against it"
            )
            return report

        report["anchors"] = int(
            conn.execute(f"SELECT COUNT(*) FROM {QUERY_ANCHOR_TABLE}").fetchone()[0]
        )
        report["anchors_live"] = int(
            conn.execute(
                f"SELECT COUNT(*) FROM {QUERY_ANCHOR_TABLE} WHERE decayed = 0"
            ).fetchone()[0]
        )
        report["anchor_edges"] = int(
            conn.execute(f"SELECT COUNT(*) FROM {QUERY_ANCHOR_EDGE_TABLE}").fetchone()[0]
        )
        report["edges_per_anchor"] = (
            round(report["anchor_edges"] / report["anchors"], 3)
            if report["anchors"]
            else 0.0
        )
        report["scopes"] = {
            str(row[0]): int(row[1])
            for row in conn.execute(
                f"SELECT scope, COUNT(*) FROM {QUERY_ANCHOR_TABLE} "
                "GROUP BY scope ORDER BY COUNT(*) DESC"
            ).fetchall()
        }

        duplicates = [
            (str(row[0]), str(row[1]), int(row[2]))
            for row in conn.execute(
                f"""
                SELECT scope, fingerprint, COUNT(*) AS n FROM {QUERY_ANCHOR_TABLE}
                GROUP BY scope, fingerprint HAVING n > 1
                ORDER BY n DESC LIMIT ?
                """,
                (max_examples,),
            ).fetchall()
        ]
        report["duplicate_identities"] = len(duplicates)
        report["duplicate_identity_examples"] = [
            {"scope": scope, "fingerprint": fingerprint, "rows": count}
            for scope, fingerprint, count in duplicates
        ]

        unusable = [
            str(row[0])
            for row in conn.execute(
                f"""
                SELECT id FROM {QUERY_ANCHOR_TABLE}
                WHERE dimensions <= 0
                   OR LENGTH(embedding) != dimensions * 4
                   OR LENGTH(embedding) = 0
                   OR embedding = ZEROBLOB(LENGTH(embedding))
                ORDER BY id LIMIT ?
                """,
                (max_examples,),
            ).fetchall()
        ]
        report["anchors_with_unusable_embedding"] = int(
            conn.execute(
                f"""
                SELECT COUNT(*) FROM {QUERY_ANCHOR_TABLE}
                WHERE dimensions <= 0
                   OR LENGTH(embedding) != dimensions * 4
                   OR LENGTH(embedding) = 0
                   OR embedding = ZEROBLOB(LENGTH(embedding))
                """
            ).fetchone()[0]
        )
        report["unusable_embedding_examples"] = unusable
        report["embedding_dimensions"] = [
            int(row[0])
            for row in conn.execute(
                f"SELECT DISTINCT dimensions FROM {QUERY_ANCHOR_TABLE} "
                "ORDER BY dimensions"
            ).fetchall()
        ]

        edgeless = [
            str(row[0])
            for row in conn.execute(
                f"""
                SELECT a.id FROM {QUERY_ANCHOR_TABLE} a
                WHERE NOT EXISTS (
                    SELECT 1 FROM {QUERY_ANCHOR_EDGE_TABLE} e WHERE e.anchor_id = a.id
                )
                ORDER BY a.id LIMIT ?
                """,
                (max_examples,),
            ).fetchall()
        ]
        report["anchors_without_edges"] = int(
            conn.execute(
                f"""
                SELECT COUNT(*) FROM {QUERY_ANCHOR_TABLE} a
                WHERE NOT EXISTS (
                    SELECT 1 FROM {QUERY_ANCHOR_EDGE_TABLE} e WHERE e.anchor_id = a.id
                )
                """
            ).fetchone()[0]
        )
        report["anchors_without_edges_examples"] = edgeless

        missing = [
            str(row[0])
            for row in conn.execute(
                f"""
                SELECT DISTINCT e.target_id FROM {QUERY_ANCHOR_EDGE_TABLE} e
                LEFT JOIN nodes n ON n.id = e.target_id
                WHERE n.id IS NULL
                ORDER BY e.target_id LIMIT ?
                """,
                (max_examples,),
            ).fetchall()
        ]
        report["edges_to_missing_nodes"] = int(
            conn.execute(
                f"""
                SELECT COUNT(*) FROM {QUERY_ANCHOR_EDGE_TABLE} e
                LEFT JOIN nodes n ON n.id = e.target_id
                WHERE n.id IS NULL
                """
            ).fetchone()[0]
        )
        report["missing_target_examples"] = missing

        decayed_targets = conn.execute(
            f"""
            SELECT e.target_id, COUNT(*) AS edges FROM {QUERY_ANCHOR_EDGE_TABLE} e
            JOIN nodes n ON n.id = e.target_id
            WHERE n.decayed = 1
            GROUP BY e.target_id
            """
        ).fetchall()
        replacement_cache: dict[str, str | None] = {}
        stale_edges = 0
        stale_examples: list[dict[str, str]] = []
        inert_edges = 0
        for row in decayed_targets:
            target = str(row[0])
            replacement = read_only_replacement(conn, target, cache=replacement_cache)
            if replacement is None:
                inert_edges += int(row[1])
                continue
            stale_edges += int(row[1])
            if len(stale_examples) < max_examples:
                stale_examples.append({"target": target, "replacement": replacement})
        report["edges_to_superseded_nodes"] = stale_edges
        report["superseded_target_examples"] = stale_examples
        report["edges_to_decayed_nodes_without_replacement"] = inert_edges

        report.update(_scope_consistency(conn, max_examples=max_examples))

        report["ok"] = (
            report["duplicate_identities"] == 0
            and report["anchors_with_unusable_embedding"] == 0
            and report["anchors_without_edges"] == 0
            and report["edges_to_missing_nodes"] == 0
            and report["edges_to_superseded_nodes"] == 0
            and report["anchors_outside_source_scope"] == 0
        )
        return report
    finally:
        conn.close()


def _scope_consistency(
    conn: sqlite3.Connection, *, max_examples: int
) -> dict[str, Any]:
    """Check every anchor's identity against the consumed recall history.

    Builds the set of ``(normalized scope, recall_fingerprint(query, that
    scope))`` over every consumed recall event and asks whether each anchor's
    own ``(scope, fingerprint)`` is in it. An anchor built in a scope other
    than its source event's would have been fingerprinted under that other
    scope, so it cannot be in the set; an anchor whose query was never asked
    in its scope cannot either.

    Cosine-merged anchors are safe here: a reinforcing upsert keeps the stored
    query, fingerprint and scope of the anchor it matched, and that anchor was
    born from an event of its own.
    """

    identities: set[tuple[str, str]] = set()
    cursor = conn.execute(
        f"SELECT e.query, e.scope FROM recall_events e "
        f"WHERE {' AND '.join(_CONSUMED_CLAUSES)}"
    )
    while True:
        rows = cursor.fetchmany(1000)
        if not rows:
            break
        for row in rows:
            try:
                scope = normalize_scope(str(row[1] or "global"))
            except ValueError:
                continue
            identities.add((scope, recall_fingerprint(str(row[0] or ""), scope)))

    offenders: list[dict[str, str]] = []
    count = 0
    for row in conn.execute(
        f"SELECT id, scope, fingerprint, query FROM {QUERY_ANCHOR_TABLE} ORDER BY id"
    ):
        if (str(row[1]), str(row[2])) in identities:
            continue
        count += 1
        if len(offenders) < max_examples:
            offenders.append(
                {
                    "anchor_id": str(row[0]),
                    "scope": str(row[1]),
                    "query": str(row[3])[:120],
                }
            )
    return {
        "consumed_event_identities": len(identities),
        "anchors_outside_source_scope": count,
        "outside_source_scope_examples": offenders,
    }


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


def _emit(payload: dict[str, Any], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"wrote {path}", flush=True)


def _print_summary(summary: BackfillSummary) -> None:
    print("")
    print(f"{summary.mode} summary")
    print(
        f"  elapsed              {format_duration(summary.elapsed_seconds)} "
        f"({summary.elapsed_seconds:.3f}s)"
    )
    print(f"  events in range      {summary.events_total}")
    print(f"  events processed     {summary.events_processed}")
    print(f"  events grounded      {summary.events_grounded}")
    if summary.mode == "dry-run":
        print(f"  projected anchors    {summary.projected_anchors} (upper bound)")
        print(f"  projected edges      {summary.projected_edges}")
        if summary.projected_seconds is not None:
            print(
                f"  projected encode     {format_duration(summary.projected_seconds)} "
                f"at {summary.assumed_rate} queries/s"
            )
    else:
        print(f"  events applied       {summary.events_applied}")
        print(f"  anchors created      {summary.anchors_created}")
        print(
            f"  anchors reinforced   {summary.anchors_reinforced} "
            f"({summary.anchors_merged_by_cosine} merged by cosine)"
        )
        print(f"  edge writes          {summary.edge_writes}")
    for label, value in (
        ("ungrounded (no credit)", summary.events_ungrounded),
        ("already anchored", summary.events_already_anchored),
        ("no live target left", summary.events_without_live_target),
        ("unusable scope", summary.events_skipped_bad_scope),
        ("zero-vector query", summary.events_skipped_zero_vector),
        ("results with no node row", summary.results_without_node),
        ("targets missing", summary.targets_missing),
        ("targets decayed -> replaced", summary.targets_decayed_with_replacement),
        ("targets decayed, no heir", summary.targets_decayed_without_replacement),
        ("targets skipped (dead)", summary.targets_skipped_dead),
        ("busy retries", summary.busy_retries),
    ):
        if value:
            print(f"  {label:<28} {value}")
    if summary.sweep:
        print(
            f"  edge sweep           {summary.sweep['edges_moved']} edges moved "
            f"across {summary.sweep['targets_migrated']} replaced targets"
        )
    if summary.mode != "dry-run":
        print(
            f"  anchor tables now    {summary.anchors_total} anchors, "
            f"{summary.anchor_edges_total} edges"
        )
    print(f"  rate                 {summary.events_per_second} events/s")


def cmd_backfill(args: argparse.Namespace) -> int:
    db = args.db.expanduser()
    if not db.is_file():
        raise BackfillError(f"--db does not exist: {db}")
    until = parse_cutoff(args.until)
    try:
        scope = normalize_scope(args.scope) if args.scope else None
    except ValueError as exc:
        raise BackfillError(f"--scope is not a scope: {exc}")

    if args.dry_run:
        summary = dry_run(
            db,
            until=until,
            scope=scope,
            limit=args.limit,
            min_containment=args.min_containment,
            assumed_rate=args.assumed_rate,
            progress_interval=args.progress_interval,
            page_size=args.batch_size,
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

    embedder, info = build_query_embedder(
        model_name=args.model,
        encode_batch=args.encode_batch,
        allow_fallback=args.allow_fallback_embeddings,
    )
    summary = BackfillSummary(
        mode="backfill",
        db=str(db),
        started_at=_utc_now_iso(),
        backup=asdict(backup),
    )
    store = MemoryStore(db)
    try:
        store.connection.execute(f"PRAGMA busy_timeout = {int(args.busy_timeout_ms)}")
        # After the migration, before any anchor write: from here the connection
        # physically cannot touch nodes, connections or recall_events.
        install_anchor_only_authorizer(store.connection)
        try:
            run_backfill(
                store,
                embedder=embedder,
                embedder_info=info,
                summary=summary,
                until=until,
                scope=scope,
                limit=args.limit,
                page_size=args.batch_size,
                min_containment=args.min_containment,
                progress_interval=args.progress_interval,
                quiet=args.quiet,
            )
            run_edge_migration_sweep(store, summary, quiet=args.quiet)
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
    if args.limit and summary.events_processed >= args.limit:
        print(
            "\nstopped at --limit; re-run without it to finish the range",
            flush=True,
        )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    db = args.db.expanduser()
    if not db.is_file():
        raise BackfillError(f"--db does not exist: {db}")
    report = verify_report(db)
    width = max(len(key) for key in report)
    for key, value in report.items():
        if key.endswith("_examples") and not value:
            continue
        print(f"{key:<{width}}  {value}")
    _emit(report, args.json)
    return 0 if report.get("ok") else 1


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backfill_query_anchors.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Log at DEBUG instead of INFO."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "backfill",
        help="Build query anchors from consumed recall history (resumable).",
        description="Build query anchors from consumed recall history. Writes "
        "only query_anchors and query_anchor_edges — enforced by a SQLite "
        "authorizer, not by review.",
    )
    _add_db_argument(run)
    _add_backup_arguments(run)
    run.add_argument(
        "--until",
        default=None,
        help="Only use events with created_at < CUTOFF (ISO-8601 instant or bare "
        "date, UTC). This is what makes a leak-free evaluation build possible: "
        "anchors from before the cutoff, queries scored after it.",
    )
    run.add_argument(
        "--scope",
        default=None,
        help="Only use events recorded in this scope.",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be written and write nothing. Opens the database "
        "read-only, so no backup is required and no migration runs.",
    )
    run.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_EVENTS,
        help=f"Events per page/encode call (default {DEFAULT_BATCH_EVENTS}). Each "
        "event still commits on its own.",
    )
    run.add_argument(
        "--encode-batch",
        type=int,
        default=DEFAULT_ENCODE_BATCH,
        help=f"Queries per encoder forward pass (default {DEFAULT_ENCODE_BATCH}).",
    )
    run.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after examining this many events (0 = the whole range).",
    )
    run.add_argument(
        "--min-containment",
        type=float,
        default=RECALL_CREDIT_MIN_CONTAINMENT,
        help="Grounding threshold for the retro labels (default "
        f"{RECALL_CREDIT_MIN_CONTAINMENT}, the live credit loop's own value). "
        "Changing it makes the retro anchors disagree with the live ones.",
    )
    run.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model (default {DEFAULT_EMBEDDING_MODEL}).",
    )
    run.add_argument(
        "--allow-fallback-embeddings",
        action="store_true",
        help="Proceed even if the model did not load and vectors would come from "
        "the hash fallback. For fixtures only — never for a real database.",
    )
    run.add_argument(
        "--assumed-rate",
        type=float,
        default=DEFAULT_ASSUMED_RATE,
        help="Queries/s used to project a --dry-run encode duration (default "
        f"{DEFAULT_ASSUMED_RATE}).",
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
        help="Read-only anchor integrity report.",
        description="Read-only report. Exit 0 only when no anchor edge is dead, "
        "no anchor sits outside its source event's scope, every anchor has a "
        "usable vector and at least one edge, and no identity is duplicated.",
    )
    _add_db_argument(verify)
    verify.add_argument("--json", type=Path, default=None, help="Write the report here.")
    verify.set_defaults(func=cmd_verify)

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
            "\ninterrupted: every committed anchor is written; re-run to finish "
            "the rest",
            file=sys.stderr,
            flush=True,
        )
        return 130


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
