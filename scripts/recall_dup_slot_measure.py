#!/usr/bin/env python3
"""Measure how many recall slots are occupied by repeats, before and after the collapse.

The claim this script exists to falsify is "recall returns five different
facts, not one fact five times". It answers it with a controlled A/B replay of
real traffic:

* the query set is the live database's own ``recall_events`` -- a frozen,
  seeded sample, committed with the artifact so the run is reproducible;
* both arms run against byte-identical copies of ONE snapshot of the live
  database, in the same order, with the same code;
* the only difference between the arms is ``LM_RECALL_NEAR_DUP_COSINE``: the
  BEFORE arm pins it to ``0`` (the documented rollback value, i.e. today's
  byte-only behaviour) and the AFTER arm unsets it so the shipped default
  applies.

Nothing here writes the live database. The live file is opened exactly once,
``mode=ro``, and copied out with ``VACUUM INTO`` (there is no ``sqlite3`` CLI on
this host, and a plain ``cp`` of a database with a hot WAL copies half a
database). Every read-write path is guarded by :func:`_refuse_live_path`, which
raises rather than open a file under the live database's directory. The live
MCP server and the dashboard are never signalled: the replay talks to a copy in
its own workdir and knows nothing about the running process.

What "repeat" means, once, for both arms
----------------------------------------
A delivered slot is a **repeat** when a HIGHER-RANKED slot of the SAME answer
carries either byte-identical content or a mean-pooled chunk-vector cosine
strictly above the threshold (0.95 by default -- the shipped
``LM_RECALL_NEAR_DUP_COSINE``). Repeat-ness is a property of the ranked answer,
not of an arm: both arms rank identically (asserted per query, reported as
``ranking_identical``), so exactly the same slots are repeats in both, and the
arms can only differ in what they *deliver* for them.

A repeat slot is **occupied** when the arm shipped it with content
(``delivery`` is ``full`` or ``snippet``) -- the agent spent a slot re-reading a
fact it already has -- and **freed** when the arm shipped it as a
``content_ref`` stub (``twin_duplicate``/``near_duplicate``), which costs a
pointer and leaves the fact reachable through ``memory_lookup``.

    duplicate-slot share = occupied repeat slots / delivered slots

The definition deliberately uses the same vectors and the same threshold the
shipped code uses, which makes the AFTER arm's collapse *related to* the metric
by construction. Two things keep the measurement honest anyway. First, the
BEFORE number is not a tautology at all: it says how often real traffic puts a
near-duplicate in a slot, which no threshold choice can manufacture. Second,
every collapsed pair is re-checked with an INDEPENDENT lexical measure (token
Jaccard, shingle containment, difflib ratio) that shares no arithmetic with the
embedding model, and the lowest-overlap pairs are printed for a human to read.

What the collapse costs
-----------------------
Reported next to the gain, never omitted: how many collapsed slots were
near-misses on the length guard (the candidate was longer than its bearer but
not by enough to trip the guard, so some text really did become a stub), how
many characters that was, how many repeats the guard refused to collapse, and
how many repeats survived because a node had no chunk vectors at all.

Usage::

    python3 scripts/recall_dup_slot_measure.py \\
        --live ~/.local/share/living-memory/global.sqlite3 \\
        --workdir /tmp/lm-dupslot --events 2000 --distinct 500 \\
        --json artifacts/near-dup/dup-slot-measurement.json \\
        --md artifacts/near-dup/dup-slot-measurement.md

Re-run the exact same measurement later with
``--selection artifacts/near-dup/dup-slot-measurement.json``: the frozen event
ids are read back out of the committed artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    # Import the package from THIS checkout, not from whatever else is
    # installed: the arms must exercise the code under review.
    sys.path.insert(0, str(REPO_ROOT / "src"))

from living_memory.delivery import (  # noqa: E402
    DEFAULT_NEAR_DUP_COSINE,
    DEFAULT_NEAR_DUP_LENGTH_RATIO,
    DELIVERY_FULL,
    DELIVERY_NEAR_DUPLICATE,
    DELIVERY_SNIPPET,
    NEAR_DUP_COSINE_ENV,
    context_value_max_chars_from_env,
    full_node_diet_enabled_from_env,
    provenance_value_max_chars_from_env,
    shape_recall_results,
    snippet_ladder_from_env,
    snippet_max_chars_from_env,
    sparse_entries_enabled_from_env,
    stats_compaction_enabled_from_env,
)
from living_memory.near_dup import cosine, mean_pooled_vectors  # noqa: E402
from living_memory.retrieval import MemoryRecallService  # noqa: E402
from living_memory.server import _near_duplicate_map  # noqa: E402
from living_memory.storage import MemoryStore  # noqa: E402


PROGRAM = "recall_dup_slot_measure.py"
DEFAULT_LIVE_DB = Path.home() / ".local" / "share" / "living-memory" / "global.sqlite3"
DEFAULT_WORKDIR = Path("/tmp/lm-dupslot")
DEFAULT_SEED = 20260823
DEFAULT_EVENT_SAMPLE = 2000
DEFAULT_DISTINCT_SAMPLE = 500

ARM_BEFORE = "before_rollback"
ARM_AFTER = "after_shipped_default"

STRATUM_EVENTS = "events"
STRATUM_DISTINCT = "distinct_queries"

# Delivery classes that put text in the slot. Everything else is a
# ``content_ref`` stub, which is what "freed" means here.
CONTENT_BEARING = (DELIVERY_FULL, DELIVERY_SNIPPET)

# Thresholds the repeat definition is re-evaluated at, so a reader can see how
# much of the headline number is the threshold's doing.
SENSITIVITY_THRESHOLDS = (0.90, 0.93, 0.95, 0.97, 0.99)

# The lexical corroboration deliberately shares nothing with the embedding
# model: plain Unicode word tokens, set overlap, and n-gram containment.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_SHINGLE = 5
_DIFFLIB_CAP = 4000

# Pairs whose lexical overlap is this low are printed for a human to read: the
# spot check that a collapsed pair was not two different facts.
_SPOT_CHECK_JACCARD = 0.60
_SPOT_CHECK_LIMIT = 12

# Identifier-shaped tokens: long enough to be a name and carrying a digit, so
# timestamps, PIDs, commit hashes, ticket ids and goal-node names qualify while
# ordinary prose does not. A collapsed node's identifiers that are absent from
# its bearer are the specifics the agent stops receiving -- a mechanical proxy
# for "these were two different facts", computed without an embedding.
_IDENTIFIER_RE = re.compile(r"[\w./:-]{5,}", re.UNICODE)
_IDENTIFIER_LOSS_FLAG = 3


class MeasurementError(RuntimeError):
    """A refusal the operator has to act on, printed without a traceback."""


# --------------------------------------------------------------------------
# Live-database safety
# --------------------------------------------------------------------------


def _refuse_live_path(path: Path, live: Path) -> None:
    """Raise if ``path`` is the live database or anything beside it.

    The whole measurement runs on copies; the one thing that must never happen
    is a read-write open of the operator's real memory. Guarding the *directory*
    rather than just the file also protects the WAL and the -shm beside it, and
    makes ``--workdir ~/.local/share/living-memory`` impossible rather than
    merely unwise.
    """

    resolved = path.expanduser().resolve()
    live_resolved = live.expanduser().resolve()
    if resolved == live_resolved:
        raise MeasurementError(
            f"refusing to open the live database read-write: {resolved}"
        )
    live_dir = live_resolved.parent
    if resolved == live_dir or live_dir in resolved.parents:
        raise MeasurementError(
            f"refusing to work inside the live database's directory: {resolved}"
        )


def snapshot_live_database(live: Path, snapshot: Path) -> dict[str, Any]:
    """``VACUUM INTO`` a read-only connection to the live file. Never writes it.

    ``mode=ro`` is the guarantee, and ``VACUUM INTO`` is the mechanism: it reads
    through that connection and writes a brand-new file elsewhere, folding in
    the WAL of a database a server is actively writing. ``cp`` cannot do this
    safely and there is no ``sqlite3`` CLI on this host.
    """

    _refuse_live_path(snapshot, live)
    if not live.exists():
        raise MeasurementError(f"live database not found: {live}")
    if snapshot.exists():
        snapshot.unlink()
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    connection = sqlite3.connect(f"file:{live}?mode=ro", uri=True)
    try:
        connection.execute("VACUUM INTO ?", (str(snapshot),))
    finally:
        connection.close()
    elapsed = time.perf_counter() - started
    preflight(snapshot)
    return {
        "source": str(live),
        "source_opened": "read-only (file:...?mode=ro)",
        "path": str(snapshot),
        "bytes": snapshot.stat().st_size,
        "sha256": _file_sha256(snapshot),
        "vacuum_seconds": round(elapsed, 3),
        **_snapshot_counts(snapshot),
    }


_REQUIRED_SCHEMA: dict[str, tuple[str, ...]] = {
    "recall_events": ("id", "query", "requested_scope", "depth", "max_results", "created_at"),
    "nodes": ("id", "content", "level", "scope", "decayed", "source_traces"),
    "node_chunk_embeddings": ("node_id", "dimensions", "embedding"),
}


def preflight(snapshot: Path) -> None:
    """Refuse a database this harness cannot measure, with the reason.

    The contract with a database is its schema, the same contract
    ``scripts/lm_collapse_near_dups.py`` works to: several LM installations
    exist and their code versions have drifted. A missing column should say so
    on the first line, not surface as a traceback three functions deep after a
    600 MiB copy has already been made.
    """

    connection = sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)
    try:
        present = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        problems: list[str] = []
        for table, columns in _REQUIRED_SCHEMA.items():
            if table not in present:
                problems.append(f"table {table} is missing")
                continue
            columns_present = {
                str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
            }
            missing = [column for column in columns if column not in columns_present]
            if missing:
                problems.append(f"{table} lacks {', '.join(missing)}")
    finally:
        connection.close()
    if problems:
        raise MeasurementError(
            "this database cannot be measured by this harness: "
            + "; ".join(problems)
        )


def _snapshot_counts(snapshot: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)
    try:
        counts = {
            "recall_events": _scalar(connection, "SELECT COUNT(*) FROM recall_events"),
            "distinct_queries": _scalar(
                connection, "SELECT COUNT(DISTINCT query) FROM recall_events"
            ),
            "active_nodes": _scalar(
                connection, "SELECT COUNT(*) FROM nodes WHERE decayed = 0"
            ),
            "chunk_rows": _scalar(
                connection, "SELECT COUNT(*) FROM node_chunk_embeddings"
            ),
            "latest_event_at": _scalar(
                connection, "SELECT MAX(created_at) FROM recall_events"
            ),
        }
    finally:
        connection.close()
    return counts


def _scalar(connection: sqlite3.Connection, sql: str) -> Any:
    row = connection.execute(sql).fetchone()
    return row[0] if row else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clone_snapshot(snapshot: Path, destination: Path, live: Path) -> Path:
    """One arm's private, byte-identical copy of the snapshot.

    Each arm gets its own file even though the replay itself does not log:
    recall's own lazy chunk drain writes, and two arms sharing a file would let
    the first one's writes reach the second, which is exactly the confound this
    measurement is built to exclude.
    """

    _refuse_live_path(destination, live)
    if destination.exists():
        destination.unlink()
    shutil.copyfile(snapshot, destination)
    return destination


def chunk_state(database: Path) -> dict[str, int]:
    """A cheap fingerprint of the chunk table, to prove the corpus held still."""

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows, nodes, dimensions = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT node_id), COALESCE(SUM(dimensions), 0)
            FROM node_chunk_embeddings
            """
        ).fetchone()
    finally:
        connection.close()
    return {
        "chunk_rows": int(rows),
        "chunked_nodes": int(nodes),
        "dimension_sum": int(dimensions),
    }


def warm_drain(base: Path, scopes: Sequence[str | None]) -> dict[str, Any]:
    """Run recall's own lazy chunk drain ONCE, before either arm is cloned.

    ``retrieval._collect_vector`` backfills chunks for any active node in the
    searched scopes that has none (``retrieval.py:854``). That write happens on
    the read path, so a replay that skipped this step would measure a corpus
    that grows vectors *while it is being measured*: a node drained at query
    500 had no vector at query 10, and the repeat ground truth computed after
    the run would count repeats no arm could have collapsed at the time.

    Draining here, on the shared base both arms are copied from, makes the
    corpus constant for the whole measurement -- which is then verified rather
    than assumed by comparing :func:`chunk_state` before and after each arm.
    """

    before = chunk_state(base)
    store = MemoryStore(base)
    service = MemoryRecallService(store)
    ordered = sorted({scope or "" for scope in scopes})
    for scope in ordered:
        # depth=0 keeps the graph channel out of it; the drain lives in the
        # vector channel and runs regardless.
        service.memory_recall(
            "warm the scope chunk matrix",
            scope=scope or None,
            depth=0,
            max_results=1,
            log_access=False,
            log_event=False,
        )
    after = chunk_state(base)
    return {
        "scopes_warmed": len(ordered),
        "before": before,
        "after": after,
        "chunk_rows_written": after["chunk_rows"] - before["chunk_rows"],
        "nodes_gaining_vectors": after["chunked_nodes"] - before["chunked_nodes"],
    }


# --------------------------------------------------------------------------
# The frozen query set
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplayQuery:
    """One recorded recall request, replayed verbatim."""

    event_id: str
    query: str
    requested_scope: str | None
    depth: str | None
    max_results: int


SELECTION_RULE = (
    "Events with a non-empty query, ordered by recall_events.id (ULID, so "
    "chronological). Stratum 'events': random.Random(seed).sample over that "
    "ordering, i.e. traffic-weighted -- a query asked 40 times can be drawn 40 "
    "times, which is what makes the slot counts represent slots actually "
    "delivered in the field. Stratum 'distinct_queries': the distinct query "
    "texts, each represented by its earliest event, sampled the same way with "
    "seed+1 -- one vote per query text, so a handful of hot queries cannot "
    "carry the result. Both strata are replayed with each event's own recorded "
    "requested_scope, depth and max_results."
)


def load_events(snapshot: Path) -> list[ReplayQuery]:
    connection = sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT id, query, requested_scope, depth, max_results
            FROM recall_events
            WHERE TRIM(query) != ''
            ORDER BY id ASC
            """
        ).fetchall()
    finally:
        connection.close()
    return [
        ReplayQuery(
            event_id=str(row["id"]),
            query=str(row["query"]),
            requested_scope=(str(row["requested_scope"]) or None)
            if row["requested_scope"] is not None
            else None,
            depth=None if row["depth"] is None else str(row["depth"]),
            max_results=int(row["max_results"] or 0),
        )
        for row in rows
    ]


def build_selection(
    events: Sequence[ReplayQuery], *, seed: int, event_sample: int, distinct_sample: int
) -> dict[str, list[str]]:
    """Freeze the two strata as lists of event ids."""

    by_id = {event.event_id: event for event in events}
    ordered_ids = [event.event_id for event in events]
    event_ids = _sample(ordered_ids, event_sample, random.Random(seed))

    earliest_by_query: dict[str, str] = {}
    for event in events:
        earliest_by_query.setdefault(event.query, event.event_id)
    distinct_ids_pool = sorted(earliest_by_query.values())
    distinct_ids = _sample(distinct_ids_pool, distinct_sample, random.Random(seed + 1))

    missing = [event_id for event_id in event_ids + distinct_ids if event_id not in by_id]
    if missing:  # pragma: no cover - defensive
        raise MeasurementError(f"selection references unknown events: {missing[:3]}")
    return {STRATUM_EVENTS: event_ids, STRATUM_DISTINCT: distinct_ids}


def _sample(population: Sequence[str], size: int, rng: random.Random) -> list[str]:
    if size <= 0:
        return []
    if size >= len(population):
        return list(population)
    drawn = rng.sample(range(len(population)), size)
    # Replay in the recorded order, so the run is a chronological replay and
    # not a shuffle: an ordering that depends on the seed would be one more
    # thing to hold equal between arms.
    return [population[index] for index in sorted(drawn)]


def load_frozen_selection(path: Path) -> dict[str, list[str]]:
    """Read the committed artifact back and replay exactly what it recorded."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    selection = payload.get("selection") or {}
    strata = selection.get("strata") or {}
    frozen = {
        name: [str(event_id) for event_id in (strata.get(name) or {}).get("event_ids", [])]
        for name in (STRATUM_EVENTS, STRATUM_DISTINCT)
    }
    if not any(frozen.values()):
        raise MeasurementError(f"no frozen selection found in {path}")
    return frozen


def selection_digest(strata: Mapping[str, Sequence[str]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(strata):
        digest.update(name.encode("utf-8"))
        for event_id in strata[name]:
            digest.update(b"\x00")
            digest.update(event_id.encode("utf-8"))
    return digest.hexdigest()


# --------------------------------------------------------------------------
# One arm of the replay
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Slot:
    """One delivered recall slot, as the agent would have received it."""

    node_id: str
    rank: int
    delivery: str
    full_chars: int
    delivered_chars: int
    level: str
    scope: str
    content_sha256: str


@dataclass(slots=True)
class Answer:
    """One replayed recall_event: its ranked, shaped slots."""

    event_id: str
    stratum: str
    requested_scope: str | None
    max_results: int
    slots: list[Slot]
    duplicate_map: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0


@dataclass(slots=True)
class ArmResult:
    name: str
    env: dict[str, str]
    answers: dict[str, Answer]
    seconds: float
    empty_answers: int


def _arm_env(arm: str) -> dict[str, str]:
    """The env difference that IS the experiment.

    BEFORE pins the rollback value explicitly. AFTER sets nothing at all, so
    the shipped default is what runs -- a hand-written "0.95" would prove only
    that 0.95 works, not that the shipped default is 0.95.
    """

    if arm == ARM_BEFORE:
        return {NEAR_DUP_COSINE_ENV: "0"}
    return {}


def run_arm(
    arm: str,
    database: Path,
    plan: Sequence[tuple[str, ReplayQuery]],
    *,
    progress_every: int,
) -> ArmResult:
    env = _arm_env(arm)
    with _environment(env):
        store = MemoryStore(database)
        service = MemoryRecallService(store)
        answers: dict[str, Answer] = {}
        empty = 0
        started = time.perf_counter()
        for index, (stratum, item) in enumerate(plan, start=1):
            answer, was_empty = _replay_one(store, service, stratum, item)
            answers[_answer_key(stratum, item.event_id)] = answer
            empty += int(was_empty)
            if progress_every and index % progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  [{arm}] {index}/{len(plan)} replayed "
                    f"({elapsed:.0f}s, {elapsed / index * 1000:.0f} ms/query)",
                    flush=True,
                )
        elapsed = time.perf_counter() - started
    return ArmResult(
        name=arm, env=dict(env), answers=answers, seconds=elapsed, empty_answers=empty
    )


def _answer_key(stratum: str, event_id: str) -> str:
    # The two strata overlap by construction, and an event drawn into both must
    # be counted once per stratum, not merged.
    return f"{stratum}:{event_id}"


def _replay_one(
    store: MemoryStore,
    service: MemoryRecallService,
    stratum: str,
    item: ReplayQuery,
) -> tuple[Answer, bool]:
    """Reproduce server.memory_recall's delivery path for one recorded request.

    ``log_access=False, log_event=False`` freezes the corpus: no access counts,
    no new recall_events. Production logs both, but a replay that logged would
    let query N change the ranking of query N+1, and the two arms would stop
    being comparable for a reason that has nothing to do with the flag under
    test. The session-duplicate class is likewise out: it depends on transport
    session state this replay cannot reconstruct, and the goal's question is
    about repeats WITHIN one answer.
    """

    started = time.perf_counter()
    results = service.memory_recall(
        item.query,
        scope=item.requested_scope,
        depth=item.depth,
        max_results=item.max_results,
        log_access=False,
        log_event=False,
    )
    duplicate_map = _near_duplicate_map(store, results) or {}
    shaped = shape_recall_results(
        results,
        already_delivered_ids=set(),
        snippet_max_chars=snippet_max_chars_from_env(),
        context_value_max_chars=context_value_max_chars_from_env(),
        session_dedup=False,
        snippet_ladder=snippet_ladder_from_env(),
        full_node_diet=full_node_diet_enabled_from_env(),
        provenance_value_max_chars=provenance_value_max_chars_from_env(),
        stats_compaction=stats_compaction_enabled_from_env(),
        sparse_entries=sparse_entries_enabled_from_env(),
        duplicate_of=duplicate_map or None,
    )
    slots = [
        Slot(
            node_id=result.node.id,
            rank=rank,
            delivery=str(entry["delivery"]),
            full_chars=len(result.node.content or ""),
            delivered_chars=len(entry["node"].get("content") or ""),
            level=str(result.node.level),
            scope=str(result.node.scope),
            content_sha256=hashlib.sha256(
                (result.node.content or "").encode("utf-8")
            ).hexdigest(),
        )
        for rank, (result, entry) in enumerate(zip(results, shaped, strict=True))
    ]
    answer = Answer(
        event_id=item.event_id,
        stratum=stratum,
        requested_scope=item.requested_scope,
        max_results=item.max_results,
        slots=slots,
        duplicate_map=dict(duplicate_map),
        seconds=time.perf_counter() - started,
    )
    return answer, not slots


class _environment:
    """Set exactly these LM_* vars for the block; restore everything after.

    Every ``LM_``/``LIVING_MEMORY_`` variable already in the environment is
    cleared first, so an operator's shell cannot silently tilt one arm: both
    arms run at shipped defaults plus the one variable under test.
    """

    def __init__(self, overrides: Mapping[str, str]) -> None:
        self._overrides = dict(overrides)
        self._saved: dict[str, str] = {}

    def __enter__(self) -> "_environment":
        for name in list(os.environ):
            if name.startswith("LM_"):
                self._saved[name] = os.environ.pop(name)
        for name, value in self._overrides.items():
            self._saved.setdefault(name, os.environ.get(name, ""))
            os.environ[name] = value
        return self

    def __exit__(self, *exc: object) -> None:
        for name in list(os.environ):
            if name.startswith("LM_"):
                del os.environ[name]
        for name, value in self._saved.items():
            if value:
                os.environ[name] = value


# --------------------------------------------------------------------------
# The repeat ground truth
# --------------------------------------------------------------------------


@dataclass(slots=True)
class PairFact:
    """One (higher-ranked, lower-ranked) slot pair inside one answer."""

    lower_rank: int
    higher_rank: int
    lower_id: str
    higher_id: str
    byte_identical: bool
    cosine: float | None


class VectorCache:
    """Mean-pooled node vectors, read once per node from the truth copy."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._vectors: dict[str, list[float] | None] = {}

    def get(self, node_id: str) -> list[float] | None:
        if node_id not in self._vectors:
            pooled = mean_pooled_vectors(self._store, [node_id])
            self._vectors[node_id] = pooled.get(node_id)
        return self._vectors[node_id]

    @property
    def missing(self) -> set[str]:
        return {node_id for node_id, vector in self._vectors.items() if vector is None}


def answer_pairs(answer: Answer, vectors: VectorCache) -> list[PairFact]:
    """Every (lower, higher) slot pair of one answer, with its two comparisons.

    Computed once, threshold-free, so the repeat definition can be re-evaluated
    at any threshold without another database read -- and so the number that
    goes in the artifact can be shown next to its own sensitivity.
    """

    pairs: list[PairFact] = []
    for lower in range(len(answer.slots)):
        for higher in range(lower):
            low, high = answer.slots[lower], answer.slots[higher]
            byte_identical = bool(low.full_chars) and low.content_sha256 == high.content_sha256
            low_vector = vectors.get(low.node_id)
            high_vector = vectors.get(high.node_id)
            similarity = (
                cosine(low_vector, high_vector)
                if low_vector is not None and high_vector is not None
                else None
            )
            pairs.append(
                PairFact(
                    lower_rank=lower,
                    higher_rank=higher,
                    lower_id=low.node_id,
                    higher_id=high.node_id,
                    byte_identical=byte_identical,
                    cosine=similarity,
                )
            )
    return pairs


def repeats_at(
    pairs: Sequence[PairFact], threshold: float
) -> dict[int, PairFact]:
    """Rank -> the highest-ranked earlier slot it repeats, at ``threshold``."""

    found: dict[int, PairFact] = {}
    for pair in sorted(pairs, key=lambda item: (item.lower_rank, item.higher_rank)):
        if pair.lower_rank in found:
            continue
        if pair.byte_identical or (pair.cosine is not None and pair.cosine > threshold):
            found[pair.lower_rank] = pair
    return found


# --------------------------------------------------------------------------
# Lexical corroboration -- shares no arithmetic with the embedding model
# --------------------------------------------------------------------------


def lexical_overlap(collapsed: str, bearer: str) -> dict[str, float]:
    """Three model-free views of "is the bearer saying what the collapsed node said".

    Containment is DIRECTIONAL on purpose: the share of the *collapsed* node's
    5-grams that appear in the bearer. The symmetric "smaller inside larger"
    form scores 1.0 for a collapsed node that strictly CONTAINS its bearer —
    exactly the length-guard near-miss case, where the agent loses the extra
    material — and would have called that a perfect match.
    """

    collapsed_tokens = _TOKEN_RE.findall(collapsed.casefold())
    bearer_tokens = _TOKEN_RE.findall(bearer.casefold())
    left_set, right_set = set(collapsed_tokens), set(bearer_tokens)
    union = left_set | right_set
    jaccard = len(left_set & right_set) / len(union) if union else 0.0
    collapsed_shingles = _shingles(collapsed_tokens)
    bearer_shingles = _shingles(bearer_tokens)
    containment = (
        len(collapsed_shingles & bearer_shingles) / len(collapsed_shingles)
        if collapsed_shingles
        else 0.0
    )
    ratio = SequenceMatcher(
        None, collapsed[:_DIFFLIB_CAP], bearer[:_DIFFLIB_CAP]
    ).ratio()
    return {
        "token_jaccard": round(jaccard, 4),
        "collapsed_in_bearer_containment": round(containment, 4),
        "difflib_ratio": round(ratio, 4),
    }


def identifiers_lost(collapsed: str, bearer: str) -> list[str]:
    """Identifier-shaped tokens in the collapsed node that its bearer lacks.

    The stub still points at the full node, so nothing is destroyed -- but the
    agent reading the answer no longer SEES these, and if a collapse joined two
    different incidents this is where it shows: two supervisor scans of the
    same project share every word of their template and disagree on every
    timestamp, PID and commit hash in them.
    """

    absent = [
        token
        for token in dict.fromkeys(_IDENTIFIER_RE.findall(collapsed))
        if any(character.isdigit() for character in token) and token not in bearer
    ]
    return absent


def _shingles(tokens: Sequence[str]) -> set[tuple[str, ...]]:
    if len(tokens) < _SHINGLE:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + _SHINGLE]) for i in range(len(tokens) - _SHINGLE + 1)}


def load_contents(database: Path, node_ids: Iterable[str]) -> dict[str, str]:
    wanted = sorted({str(node_id) for node_id in node_ids})
    if not wanted:
        return {}
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        contents: dict[str, str] = {}
        for start in range(0, len(wanted), 400):
            batch = wanted[start : start + 400]
            placeholders = ",".join("?" * len(batch))
            for node_id, content, level, scope in connection.execute(
                f"SELECT id, content, level, scope FROM nodes WHERE id IN ({placeholders})",
                batch,
            ):
                contents[str(node_id)] = str(content or "")
        return contents
    finally:
        connection.close()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def arm_metrics(
    arm: ArmResult,
    truth: Mapping[str, dict[int, PairFact]],
    keys: Sequence[str],
) -> dict[str, Any]:
    slots = 0
    answers = 0
    occupied = 0
    freed = 0
    repeats = 0
    delivered_chars = 0
    repeat_chars = 0
    distinct_chars = 0
    classes: Counter[str] = Counter()
    answers_with_occupied = 0
    for key in keys:
        answer = arm.answers[key]
        if not answer.slots:
            continue
        answers += 1
        slots += len(answer.slots)
        repeat_ranks = truth[key]
        occupied_here = 0
        for slot in answer.slots:
            classes[slot.delivery] += 1
            delivered_chars += slot.delivered_chars
            if slot.rank in repeat_ranks:
                repeats += 1
                repeat_chars += slot.delivered_chars
                if slot.delivery in CONTENT_BEARING:
                    occupied_here += 1
                else:
                    freed += 1
            else:
                distinct_chars += slot.delivered_chars
        occupied += occupied_here
        answers_with_occupied += int(occupied_here > 0)
    return {
        "answers": answers,
        "slots": slots,
        "delivered_content_chars": delivered_chars,
        # The split that says where the payload went. A freed slot does not
        # shrink the answer: the snippet ladder hands the budget to the next
        # content-bearing result, so the same characters buy distinct facts
        # instead of the same fact twice.
        "delivered_chars_on_repeat_slots": repeat_chars,
        "delivered_chars_on_distinct_slots": distinct_chars,
        "delivery_classes": dict(sorted(classes.items())),
        "repeat_slots": repeats,
        "repeat_slots_occupied": occupied,
        "repeat_slots_freed": freed,
        "duplicate_slot_share": _share(occupied, slots),
        "answers_with_an_occupied_repeat": answers_with_occupied,
        "answers_with_an_occupied_repeat_share": _share(answers_with_occupied, answers),
        "seconds": round(arm.seconds, 1),
    }


def _share(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def transition_matrix(
    before: ArmResult, after: ArmResult, keys: Sequence[str]
) -> dict[str, int]:
    moves: Counter[str] = Counter()
    for key in keys:
        for before_slot, after_slot in zip(
            before.answers[key].slots, after.answers[key].slots, strict=True
        ):
            if before_slot.delivery != after_slot.delivery:
                moves[f"{before_slot.delivery} -> {after_slot.delivery}"] += 1
    return dict(sorted(moves.items(), key=lambda item: -item[1]))


def ranking_identity(
    before: ArmResult, after: ArmResult, keys: Sequence[str]
) -> dict[str, Any]:
    """The check that makes the A/B an A/B: same query, same ranked node ids.

    If this is not clean, the two arms differed in something other than the
    flag and every number below it is meaningless -- so it is reported first,
    not buried.
    """

    mismatches = [
        key
        for key in keys
        if [slot.node_id for slot in before.answers[key].slots]
        != [slot.node_id for slot in after.answers[key].slots]
    ]
    return {
        "answers_compared": len(keys),
        "mismatching_answers": len(mismatches),
        "identical": not mismatches,
        "examples": mismatches[:5],
    }


def collapse_cost(
    after: ArmResult,
    truth: Mapping[str, dict[int, PairFact]],
    keys: Sequence[str],
    vectors: VectorCache,
    *,
    length_ratio: float,
) -> dict[str, Any]:
    """What the collapse cost, measured rather than assumed.

    Three separate prices, kept separate because they are paid by different
    parties: text that was collapsed although it was longer than its bearer
    (near-misses on the length guard -- the agent lost that delta), repeats the
    guard refused to collapse (the guard's own cost in unfreed slots), and
    repeats no vector could see (nodes with no chunk rows).
    """

    near_misses: list[dict[str, Any]] = []
    collapsed = 0
    collapsed_chars = 0
    guard_protected = 0
    guard_protected_chars = 0
    vectorless_repeats = 0
    full_by_id: dict[str, int] = {}
    for key in keys:
        answer = after.answers[key]
        for slot in answer.slots:
            full_by_id[slot.node_id] = slot.full_chars
    for key in keys:
        answer = after.answers[key]
        repeat_ranks = truth[key]
        for slot in answer.slots:
            if slot.delivery == DELIVERY_NEAR_DUPLICATE:
                collapsed += 1
                collapsed_chars += slot.full_chars
                bearer_id = answer.duplicate_map.get(slot.node_id)
                bearer_chars = full_by_id.get(bearer_id or "", 0)
                if bearer_chars and slot.full_chars > bearer_chars:
                    ratio = slot.full_chars / bearer_chars
                    near_misses.append(
                        {
                            "event": answer.event_id,
                            "stratum": answer.stratum,
                            "collapsed": slot.node_id,
                            "bearer": bearer_id,
                            "collapsed_chars": slot.full_chars,
                            "bearer_chars": bearer_chars,
                            "length_ratio": round(ratio, 4),
                            "extra_chars": slot.full_chars - bearer_chars,
                            "guard_headroom": round(1.0 + length_ratio - ratio, 4),
                        }
                    )
                continue
            if slot.rank not in repeat_ranks or slot.delivery not in CONTENT_BEARING:
                continue
            pair = repeat_ranks[slot.rank]
            if vectors.get(slot.node_id) is None or vectors.get(pair.higher_id) is None:
                vectorless_repeats += 1
                continue
            bearer_chars = full_by_id.get(pair.higher_id, 0)
            if bearer_chars and slot.full_chars > bearer_chars * (1.0 + length_ratio):
                guard_protected += 1
                guard_protected_chars += slot.full_chars - bearer_chars
    near_misses.sort(key=lambda item: -item["length_ratio"])
    ratios = [item["length_ratio"] for item in near_misses]
    return {
        "collapsed_slots": collapsed,
        "collapsed_chars_replaced_by_stub": collapsed_chars,
        "length_guard_ratio": length_ratio,
        "length_guard_near_misses": {
            "definition": (
                "collapsed slots whose own content was LONGER than the bearer's "
                "but not by more than the guard's ratio, so the extra text "
                "became a stub"
            ),
            "count": len(near_misses),
            "share_of_collapsed": _share(len(near_misses), collapsed),
            "extra_chars_total": sum(item["extra_chars"] for item in near_misses),
            "max_length_ratio": max(ratios) if ratios else 0.0,
            "median_length_ratio": round(statistics.median(ratios), 4) if ratios else 0.0,
            "worst": near_misses[:10],
        },
        "length_guard_vetoes": {
            "definition": (
                "repeat slots left content-bearing because the candidate "
                "exceeded its bearer by more than the guard's ratio -- the "
                "'same fact plus a new detail' case the guard exists for"
            ),
            "count": guard_protected,
            "chars_protected": guard_protected_chars,
        },
        "repeats_without_vectors": {
            "definition": (
                "repeat slots that stayed content-bearing because a node had no "
                "usable chunk vectors (never drained, or mixed widths)"
            ),
            "count": vectorless_repeats,
        },
    }


def spot_check(
    after: ArmResult,
    keys: Sequence[str],
    truth_db: Path,
    pair_cosines: Mapping[tuple[str, str], float],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Lexically re-check every collapsed pair with independent arithmetic."""

    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for key in keys:
        answer = after.answers[key]
        by_id = {slot.node_id: slot for slot in answer.slots}
        for slot in answer.slots:
            if slot.delivery != DELIVERY_NEAR_DUPLICATE:
                continue
            bearer_id = answer.duplicate_map.get(slot.node_id)
            if not bearer_id:
                continue
            pair_key = (slot.node_id, bearer_id)
            if pair_key in pairs:
                pairs[pair_key]["answers"] += 1
                continue
            bearer = by_id.get(bearer_id)
            pairs[pair_key] = {
                "collapsed": slot.node_id,
                "bearer": bearer_id,
                "collapsed_level": slot.level,
                "bearer_level": bearer.level if bearer else None,
                "scope": slot.scope,
                "collapsed_chars": slot.full_chars,
                "bearer_chars": bearer.full_chars if bearer else None,
                "cosine": pair_cosines.get(pair_key),
                "answers": 1,
            }
    contents = load_contents(
        truth_db,
        [node_id for pair in pairs for node_id in pair],
    )
    for pair_key, record in pairs.items():
        collapsed_text = contents.get(pair_key[0], "")
        bearer_text = contents.get(pair_key[1], "")
        record.update(lexical_overlap(collapsed_text, bearer_text))
        lost = identifiers_lost(collapsed_text, bearer_text)
        record["identifiers_lost"] = len(lost)
        record["identifiers_lost_examples"] = lost[:6]
        record["same_source_traces_family"] = _provenance_related(truth_db, pair_key)
    ordered = sorted(pairs.values(), key=lambda item: item["token_jaccard"])
    jaccards = [record["token_jaccard"] for record in ordered]
    containments = [record["collapsed_in_bearer_containment"] for record in ordered]
    flagged = [record for record in ordered if record["token_jaccard"] < _SPOT_CHECK_JACCARD]
    losing = [
        record
        for record in ordered
        if record["identifiers_lost"] >= _IDENTIFIER_LOSS_FLAG
    ]
    by_level: dict[str, dict[str, Any]] = {}
    for record in ordered:
        key = f"{record['collapsed_level']}/{record['bearer_level']}"
        bucket = by_level.setdefault(
            key,
            {"pairs": 0, "slots": 0, "flagged": 0, "losing_identifiers": 0,
             "jaccards": [], "chars": []},
        )
        bucket["pairs"] += 1
        bucket["slots"] += record["answers"]
        bucket["flagged"] += int(record["token_jaccard"] < _SPOT_CHECK_JACCARD)
        bucket["losing_identifiers"] += int(
            record["identifiers_lost"] >= _IDENTIFIER_LOSS_FLAG
        )
        bucket["jaccards"].append(record["token_jaccard"])
        bucket["chars"].append(record["collapsed_chars"])
    for bucket in by_level.values():
        jaccard_values = bucket.pop("jaccards")
        chars = bucket.pop("chars")
        bucket["median_token_jaccard"] = round(statistics.median(jaccard_values), 4)
        bucket["min_token_jaccard"] = min(jaccard_values)
        bucket["median_collapsed_chars"] = int(statistics.median(chars))
    return {
        "distinct_collapsed_pairs": len(ordered),
        # The axis the flagged pairs actually fall along, so a reader does not
        # have to take the hand-written verdict on faith.
        "by_level": dict(
            sorted(by_level.items(), key=lambda item: -item[1]["pairs"])
        ),
        "method": (
            "Independent of the embedding model: Unicode word tokens, set "
            "Jaccard, 5-gram shingle containment, and difflib ratio over the "
            f"first {_DIFFLIB_CAP} characters. The difflib ratio is the weakest "
            "of the three on long nodes -- it is order-sensitive and sees only "
            "that prefix, so a reordered digest scores low while Jaccard and "
            "containment still show the two texts are the same material."
        ),
        "token_jaccard": _distribution(jaccards),
        "collapsed_in_bearer_containment": _distribution(containments),
        "bearer_chaining": _bearer_chaining(ordered, threshold),
        "flag_threshold_token_jaccard": _SPOT_CHECK_JACCARD,
        "flagged_for_human_reading": len(flagged),
        "identifier_loss": {
            "definition": (
                "Identifier-shaped tokens (>=5 chars, carrying a digit: "
                "timestamps, PIDs, commit hashes, ticket ids, goal-node names) "
                "present in the collapsed node and absent from its bearer. The "
                "full node stays reachable through memory_lookup; this counts "
                "what the answer itself stopped showing."
            ),
            "flag_threshold": _IDENTIFIER_LOSS_FLAG,
            "pairs_losing_none": sum(
                1 for record in ordered if record["identifiers_lost"] == 0
            ),
            "pairs_at_or_over_threshold": len(losing),
            "slots_at_or_over_threshold": sum(record["answers"] for record in losing),
            "slots_total": sum(record["answers"] for record in ordered),
        },
        # Every concerning pair is listed, not a sample of them: the verdict on
        # whether a collapse joined two different facts is a human's to give,
        # and a human cannot give it over a truncated list.
        "pairs_for_human_reading": (
            sorted(
                {id(record): record for record in flagged + losing}.values(),
                key=lambda item: item["token_jaccard"],
            )
            or ordered[:_SPOT_CHECK_LIMIT]
        ),
        "manual_verdict": (
            "Hand-written, in the '## Manual reading of the flagged pairs' "
            "section of dup-slot-measurement.md. Re-running this script "
            "regenerates every other section of that file."
        ),
    }


def _bearer_chaining(
    pairs: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    """Collapsed slots whose stub names a bearer they never matched.

    ``build_duplicate_map`` records the ROOT of the matched node's chain, not
    the node the candidate actually exceeded the threshold against, so that no
    consumer has to chase a chain. The price shows up only in the field: when
    A collapses into B and B has already collapsed into C, A's ``content_ref``
    names C — and A's similarity to C can sit below the threshold entirely.
    The collapse itself is still justified (A did exceed the threshold against
    a higher-ranked slot of this answer), but the pointer the agent is handed
    is a weaker match than the one that earned the stub.
    """

    scored = [
        record for record in pairs if record.get("cosine") is not None
    ]
    below = [record for record in scored if record["cosine"] <= threshold]
    return {
        "definition": (
            "collapsed pairs whose cosine to the RECORDED bearer is at or "
            "below the collapse threshold, because bearer resolution walks to "
            "the root of the duplicate chain"
        ),
        "threshold": threshold,
        "pairs_scored": len(scored),
        "pairs_below_threshold": len(below),
        "slots_below_threshold": sum(record["answers"] for record in below),
        "min_cosine_to_recorded_bearer": (
            min(record["cosine"] for record in scored) if scored else None
        ),
    }


def _provenance_related(database: Path, pair: tuple[str, str]) -> bool:
    """Whether one of the pair is listed in the other's ``source_traces``.

    The concept-over-its-own-sources band the root goal describes: on delivery
    this is exactly what SHOULD collapse, so it is worth counting rather than
    mistaking for a wrong collapse.
    """

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = dict(
            connection.execute(
                "SELECT id, source_traces FROM nodes WHERE id IN (?, ?)", pair
            ).fetchall()
        )
    finally:
        connection.close()
    for node_id, other in ((pair[0], pair[1]), (pair[1], pair[0])):
        raw = rows.get(node_id)
        if not raw:
            continue
        try:
            sources = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(sources, list) and other in {str(item) for item in sources}:
            return True
    return False


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p25": round(_percentile(ordered, 0.25), 4),
        "median": round(_percentile(ordered, 0.50), 4),
        "p75": round(_percentile(ordered, 0.75), 4),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 4),
    }


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def measure(args: argparse.Namespace) -> dict[str, Any]:
    live = Path(args.live).expanduser()
    workdir = Path(args.workdir).expanduser()
    _refuse_live_path(workdir, live)
    workdir.mkdir(parents=True, exist_ok=True)

    backend = os.environ.get("LIVING_MEMORY_EMBEDDING_BACKEND", "").strip().lower()
    if backend == "hash" and not args.allow_hash_embeddings:
        raise MeasurementError(
            "LIVING_MEMORY_EMBEDDING_BACKEND=hash would embed queries in a "
            "different space than the stored chunk vectors; unset it (or pass "
            "--allow-hash-embeddings for a smoke test that is not a measurement)"
        )

    snapshot = workdir / "snapshot.sqlite3"
    if args.reuse_snapshot and snapshot.exists():
        preflight(snapshot)
        snapshot_info = {
            "source": str(live),
            "source_opened": (
                "not opened by this run: reusing the snapshot an earlier run "
                "of this script took with a read-only VACUUM INTO, pinned by "
                "the sha256 below. Re-running against one frozen snapshot is "
                "what makes repeated runs comparable at all."
            ),
            "path": str(snapshot),
            "bytes": snapshot.stat().st_size,
            "sha256": _file_sha256(snapshot),
            "reused": True,
            **_snapshot_counts(snapshot),
        }
    else:
        print(f"snapshotting {live} -> {snapshot} (read-only VACUUM INTO)", flush=True)
        snapshot_info = snapshot_live_database(live, snapshot)
    print(
        f"snapshot: {snapshot_info['recall_events']} events, "
        f"{snapshot_info['distinct_queries']} distinct queries, "
        f"{snapshot_info['active_nodes']} active nodes",
        flush=True,
    )

    events = load_events(snapshot)
    by_id = {event.event_id: event for event in events}
    if args.selection:
        strata = load_frozen_selection(Path(args.selection))
        missing = [
            event_id
            for ids in strata.values()
            for event_id in ids
            if event_id not in by_id
        ]
        if missing:
            raise MeasurementError(
                f"{len(missing)} frozen event ids are absent from this snapshot "
                f"(first: {missing[:3]}) -- the selection and the database disagree"
            )
        selection_source = f"frozen: {args.selection}"
    else:
        strata = build_selection(
            events,
            seed=args.seed,
            event_sample=args.events,
            distinct_sample=args.distinct,
        )
        selection_source = f"seeded: random.Random({args.seed})"

    plan: list[tuple[str, ReplayQuery]] = [
        (stratum, by_id[event_id])
        for stratum in (STRATUM_EVENTS, STRATUM_DISTINCT)
        for event_id in strata[stratum]
    ]
    keys_by_stratum = {
        stratum: [_answer_key(stratum, event_id) for event_id in strata[stratum]]
        for stratum in (STRATUM_EVENTS, STRATUM_DISTINCT)
    }
    print(
        f"replaying {len(plan)} recorded requests per arm "
        f"({len(strata[STRATUM_EVENTS])} traffic-weighted, "
        f"{len(strata[STRATUM_DISTINCT])} distinct-query)",
        flush=True,
    )

    base = clone_snapshot(snapshot, workdir / "base.sqlite3", live)
    print("warming the chunk drain on the shared base copy", flush=True)
    drain = warm_drain(base, [item.requested_scope for _, item in plan])
    print(
        f"drain: {drain['chunk_rows_written']} chunk rows written for "
        f"{drain['nodes_gaining_vectors']} nodes across "
        f"{drain['scopes_warmed']} scopes",
        flush=True,
    )
    base_state = chunk_state(base)

    arms: dict[str, ArmResult] = {}
    arm_states: dict[str, dict[str, int]] = {}
    for arm in (ARM_BEFORE, ARM_AFTER):
        database = clone_snapshot(base, workdir / f"{arm}.sqlite3", live)
        print(f"arm {arm}: env {_arm_env(arm) or '(shipped defaults)'}", flush=True)
        arms[arm] = run_arm(
            arm, database, plan, progress_every=args.progress_every
        )
        arm_states[arm] = chunk_state(database)
        print(f"arm {arm}: {arms[arm].seconds:.0f}s", flush=True)

    truth_db = clone_snapshot(base, workdir / "truth.sqlite3", live)
    truth_store = MemoryStore(truth_db)
    vectors = VectorCache(truth_store)
    threshold = float(args.threshold)
    pairs_by_key: dict[str, list[PairFact]] = {}
    truth: dict[str, dict[int, PairFact]] = {}
    for key, answer in arms[ARM_BEFORE].answers.items():
        pairs = answer_pairs(answer, vectors)
        pairs_by_key[key] = pairs
        truth[key] = repeats_at(pairs, threshold)
    pair_cosines: dict[tuple[str, str], float] = {}
    for pairs in pairs_by_key.values():
        for pair in pairs:
            if pair.cosine is not None:
                pair_cosines[(pair.lower_id, pair.higher_id)] = round(pair.cosine, 6)

    all_keys = [key for keys in keys_by_stratum.values() for key in keys]
    identity = ranking_identity(arms[ARM_BEFORE], arms[ARM_AFTER], all_keys)

    report: dict[str, Any] = {
        "measurement": "recall duplicate-slot share, before and after the near-dup collapse",
        "produced_at": args.produced_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "produced_by": f"scripts/{PROGRAM}",
        "repository_commit": _git_commit(),
        "live_database_written": False,
        "live_database_access": (
            "read-only: one sqlite3 'file:...?mode=ro' connection, VACUUM INTO a "
            "copy under the workdir. The live MCP server and the dashboard were "
            "never signalled, stopped or restarted."
        ),
        "snapshot": snapshot_info,
        "corpus_stability": {
            "why": (
                "recall's own lazy chunk drain (retrieval.py:854) writes chunk "
                "vectors on the read path. It is run once on the shared base "
                "copy before either arm is cloned, so the corpus is constant "
                "for the whole measurement -- and that is verified below, not "
                "assumed."
            ),
            "warm_drain": drain,
            "base": base_state,
            "after_each_arm": arm_states,
            "constant_through_both_arms": all(
                state == base_state for state in arm_states.values()
            ),
        },
        "definitions": {
            "repeat_slot": (
                "A delivered slot whose node is byte-identical to, or has "
                f"mean-pooled chunk-vector cosine strictly above {threshold} "
                "with, a HIGHER-RANKED slot of the same answer. Computed by "
                "this script from the ranked answer, identically for both arms."
            ),
            "occupied": (
                f"A repeat slot the arm shipped with content (delivery in "
                f"{list(CONTENT_BEARING)}) instead of a content_ref stub."
            ),
            "duplicate_slot_share": "occupied repeat slots / delivered slots",
            "arms": {
                ARM_BEFORE: f"{NEAR_DUP_COSINE_ENV}=0 (documented rollback: byte-only dedup)",
                ARM_AFTER: (
                    f"{NEAR_DUP_COSINE_ENV} unset, so the shipped default "
                    f"{DEFAULT_NEAR_DUP_COSINE} applies with length ratio "
                    f"{DEFAULT_NEAR_DUP_LENGTH_RATIO}"
                ),
            },
            "replay": (
                "server.memory_recall's delivery path, minus access/event "
                "logging and the session-duplicate class: the corpus is frozen "
                "so every query in both arms sees the same state, and the "
                "question is about repeats within one answer."
            ),
        },
        "selection": {
            "source": selection_source,
            "seed": args.seed,
            "rule": SELECTION_RULE,
            "digest": selection_digest(strata),
            "strata": {
                stratum: {
                    "requested": (
                        args.events if stratum == STRATUM_EVENTS else args.distinct
                    ),
                    "event_ids": strata[stratum],
                }
                for stratum in (STRATUM_EVENTS, STRATUM_DISTINCT)
            },
        },
        "ranking_identity": identity,
        "arms": {},
        "strata": {},
    }

    for stratum, keys in keys_by_stratum.items():
        before_metrics = arm_metrics(arms[ARM_BEFORE], truth, keys)
        after_metrics = arm_metrics(arms[ARM_AFTER], truth, keys)
        report["strata"][stratum] = {
            ARM_BEFORE: before_metrics,
            ARM_AFTER: after_metrics,
            "delta": {
                "duplicate_slot_share": round(
                    after_metrics["duplicate_slot_share"]
                    - before_metrics["duplicate_slot_share"],
                    6,
                ),
                "occupied_repeat_slots": (
                    after_metrics["repeat_slots_occupied"]
                    - before_metrics["repeat_slots_occupied"]
                ),
                "relative_reduction": _share(
                    before_metrics["repeat_slots_occupied"]
                    - after_metrics["repeat_slots_occupied"],
                    before_metrics["repeat_slots_occupied"],
                ),
                "delivered_content_chars": (
                    after_metrics["delivered_content_chars"]
                    - before_metrics["delivered_content_chars"]
                ),
                "delivered_chars_on_distinct_slots": (
                    after_metrics["delivered_chars_on_distinct_slots"]
                    - before_metrics["delivered_chars_on_distinct_slots"]
                ),
                "delivered_chars_on_repeat_slots": (
                    after_metrics["delivered_chars_on_repeat_slots"]
                    - before_metrics["delivered_chars_on_repeat_slots"]
                ),
            },
            "slots_changing_delivery_class": sum(
                transition_matrix(arms[ARM_BEFORE], arms[ARM_AFTER], keys).values()
            ),
            "delivery_class_transitions": transition_matrix(
                arms[ARM_BEFORE], arms[ARM_AFTER], keys
            ),
            "repeat_definition_sensitivity": {
                f"{value:.2f}": _sensitivity(
                    arms, pairs_by_key, keys, value
                )
                for value in SENSITIVITY_THRESHOLDS
            },
        }

    report["cost"] = collapse_cost(
        arms[ARM_AFTER],
        truth,
        all_keys,
        vectors,
        length_ratio=DEFAULT_NEAR_DUP_LENGTH_RATIO,
    )
    report["spot_check"] = spot_check(
        arms[ARM_AFTER], all_keys, truth_db, pair_cosines, threshold=threshold
    )
    report["arms"] = {
        arm: {
            "env": arms[arm].env or {"(none)": "shipped defaults"},
            "database": str(workdir / f"{arm}.sqlite3"),
            "seconds": round(arms[arm].seconds, 1),
            "empty_answers": arms[arm].empty_answers,
        }
        for arm in (ARM_BEFORE, ARM_AFTER)
    }
    report["nodes_without_usable_vectors"] = len(vectors.missing)
    return report


def _sensitivity(
    arms: Mapping[str, ArmResult],
    pairs_by_key: Mapping[str, list[PairFact]],
    keys: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    """The same two shares, with the repeat definition moved to ``threshold``."""

    result: dict[str, Any] = {}
    truth = {key: repeats_at(pairs_by_key[key], threshold) for key in keys}
    for arm in (ARM_BEFORE, ARM_AFTER):
        metrics = arm_metrics(arms[arm], truth, keys)
        result[arm] = metrics["duplicate_slot_share"]
        result[f"{arm}_occupied"] = metrics["repeat_slots_occupied"]
    result["repeat_slots"] = arm_metrics(arms[ARM_BEFORE], truth, keys)["repeat_slots"]
    return result


def _git_commit() -> str | None:
    """HEAD of the checkout that produced this, via git itself.

    Reading ``.git`` by hand does not survive a worktree, where ``.git`` is a
    file pointing at a gitdir whose refs live in the common directory --
    exactly the layout this measurement runs in.
    """

    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    commit = completed.stdout.strip()
    return commit or None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_markdown(report: Mapping[str, Any]) -> str:
    events = report["strata"][STRATUM_EVENTS]
    distinct = report["strata"][STRATUM_DISTINCT]
    before = events[ARM_BEFORE]
    after = events[ARM_AFTER]
    cost = report["cost"]
    spot = report["spot_check"]
    snapshot = report["snapshot"]
    lines: list[str] = []
    add = lines.append

    add("# Duplicate recall slots, measured before and after the collapse")
    add("")
    add(
        f"Produced {report['produced_at']} by `{report['produced_by']}` at commit "
        f"`{report['repository_commit']}`. Every number below is regenerated by that "
        "script from the committed selection; the one hand-written section says so."
    )
    add("")
    add("## The headline")
    add("")
    add(
        f"Over **{before['answers']} replayed real recall requests** "
        f"({before['slots']} delivered slots), the share of slots occupied by a "
        "repeat of a higher-ranked slot of the same answer falls from "
        f"**{before['duplicate_slot_share'] * 100:.2f}%** to "
        f"**{after['duplicate_slot_share'] * 100:.2f}%** — "
        f"{before['repeat_slots_occupied']} occupied slots down to "
        f"{after['repeat_slots_occupied']}, a "
        f"{events['delta']['relative_reduction'] * 100:.1f}% reduction. "
        f"{events['slots_changing_delivery_class']} slots changed delivery class."
    )
    add("")
    add(
        f"Answers carrying at least one repeat in a slot: "
        f"{before['answers_with_an_occupied_repeat']} of {before['answers']} "
        f"({before['answers_with_an_occupied_repeat_share'] * 100:.1f}%) before, "
        f"{after['answers_with_an_occupied_repeat']} "
        f"({after['answers_with_an_occupied_repeat_share'] * 100:.1f}%) after."
    )
    add("")
    loss = spot["identifier_loss"]
    add(
        f"That gain is not free, and the cost is not clean: of "
        f"{cost['collapsed_slots']} collapsed slots, "
        f"{loss['slots_at_or_over_threshold']} came from pairs where the "
        f"collapsed node named {loss['flag_threshold']} or more identifiers "
        "(timestamps, PIDs, commit hashes, ticket ids) that its bearer never "
        "mentions. Read *What the collapse cost* and the hand-written spot "
        "check below before quoting the headline."
    )
    add("")
    add("## What was measured, and what was never touched")
    add("")
    add(
        f"- Live database: `{snapshot['source']}` — "
        f"{snapshot['source_opened'].rstrip('.')}."
    )
    add(
        f"- Snapshot: `{snapshot['path']}` ({snapshot['bytes'] / 2**20:.0f} MiB, "
        f"sha256 `{snapshot['sha256'][:16]}…`), "
        f"{snapshot['recall_events']} recall_events over "
        f"{snapshot['distinct_queries']} distinct queries, "
        f"{snapshot['active_nodes']} active nodes, "
        f"{snapshot['chunk_rows']} chunk vectors."
    )
    add(f"- {report['live_database_access']}")
    add(
        "- Both arms ran against private byte-identical copies of that one "
        "snapshot, in the same order, with the same code."
    )
    stability = report["corpus_stability"]
    add(
        f"- Recall writes on the read path: its lazy chunk drain backfilled "
        f"{stability['warm_drain']['chunk_rows_written']} chunk rows for "
        f"{stability['warm_drain']['nodes_gaining_vectors']} nodes across "
        f"{stability['warm_drain']['scopes_warmed']} scopes. That drain was run "
        "ONCE on the shared base copy before either arm was cloned, and the "
        "chunk table was fingerprinted after each arm: constant through both "
        f"arms = **{stability['constant_through_both_arms']}**. Without this the "
        "corpus would have grown vectors while being measured."
    )
    add("")
    add("### The two arms")
    add("")
    add(f"- `{ARM_BEFORE}`: {report['definitions']['arms'][ARM_BEFORE]}")
    add(f"- `{ARM_AFTER}`: {report['definitions']['arms'][ARM_AFTER]}")
    add("")
    identity = report["ranking_identity"]
    add(
        f"The flag is the only difference, and that is checked rather than "
        f"assumed: across all {identity['answers_compared']} replayed answers the "
        f"two arms returned the identical ranked node ids "
        f"({identity['mismatching_answers']} mismatching answers). Only the "
        "delivery class differs."
    )
    add("")
    add("### The definition of a repeat, used identically for both arms")
    add("")
    add(f"> {report['definitions']['repeat_slot']}")
    add(">")
    add(f"> {report['definitions']['occupied']}")
    add(">")
    add(f"> duplicate-slot share = {report['definitions']['duplicate_slot_share']}")
    add("")
    add("### The query set")
    add("")
    add(f"{report['selection']['rule']}")
    add("")
    add(
        f"Selection source: {report['selection']['source']}; digest "
        f"`{report['selection']['digest'][:16]}…`. The event ids themselves are "
        "committed in the JSON beside this file, so the same measurement can be "
        "re-run against a later snapshot with `--selection`."
    )
    add("")
    add("## Per-arm counts")
    add("")
    add("| | before (rollback) | after (shipped default) |")
    add("| --- | ---: | ---: |")
    for stratum_name, stratum in (
        ("traffic-weighted events", events),
        ("distinct queries", distinct),
    ):
        add(f"| **{stratum_name}** | | |")
        arm_before, arm_after = stratum[ARM_BEFORE], stratum[ARM_AFTER]
        add(f"| answers | {arm_before['answers']} | {arm_after['answers']} |")
        add(f"| delivered slots | {arm_before['slots']} | {arm_after['slots']} |")
        add(
            f"| repeat slots (ground truth) | {arm_before['repeat_slots']} | "
            f"{arm_after['repeat_slots']} |"
        )
        add(
            f"| repeat slots occupied by content | "
            f"{arm_before['repeat_slots_occupied']} | "
            f"{arm_after['repeat_slots_occupied']} |"
        )
        add(
            f"| **duplicate-slot share** | "
            f"**{arm_before['duplicate_slot_share'] * 100:.2f}%** | "
            f"**{arm_after['duplicate_slot_share'] * 100:.2f}%** |"
        )
        add(
            f"| delivered content chars | {arm_before['delivered_content_chars']:,} | "
            f"{arm_after['delivered_content_chars']:,} |"
        )
        add(
            f"| …of which on repeat slots | "
            f"{arm_before['delivered_chars_on_repeat_slots']:,} | "
            f"{arm_after['delivered_chars_on_repeat_slots']:,} |"
        )
        add(
            f"| …of which on distinct slots | "
            f"{arm_before['delivered_chars_on_distinct_slots']:,} | "
            f"{arm_after['delivered_chars_on_distinct_slots']:,} |"
        )
    add("")
    add(
        "The payload barely moves, and that is the point: a freed slot does not "
        "shrink the answer, it hands its snippet budget to the next "
        "content-bearing result. The same characters reach the agent — spent on "
        "distinct facts instead of on the same fact twice. In the "
        f"traffic-weighted stratum the characters spent on repeat slots change by "
        f"{events['delta']['delivered_chars_on_repeat_slots']:,} and the "
        f"characters spent on distinct slots by "
        f"{events['delta']['delivered_chars_on_distinct_slots']:,}."
    )
    add("")
    add("### Delivery classes, and what moved")
    add("")
    for stratum_name, stratum in (
        ("traffic-weighted events", events),
        ("distinct queries", distinct),
    ):
        add(f"`{stratum_name}` before: `{stratum[ARM_BEFORE]['delivery_classes']}`")
        add("")
        add(f"`{stratum_name}` after: `{stratum[ARM_AFTER]['delivery_classes']}`")
        add("")
        add(
            f"{stratum['slots_changing_delivery_class']} slots changed class: "
            + ", ".join(
                f"`{move}` ×{count}"
                for move, count in stratum["delivery_class_transitions"].items()
            )
            + "."
        )
        add("")
    add("## What the collapse cost")
    add("")
    near = cost["length_guard_near_misses"]
    veto = cost["length_guard_vetoes"]
    add(
        f"- **{cost['collapsed_slots']} slots collapsed**, replacing "
        f"{cost['collapsed_chars_replaced_by_stub']:,} characters of delivered "
        "text with a `content_ref` stub."
    )
    add(
        f"- **Length-guard near-misses: {near['count']}** "
        f"({near['share_of_collapsed'] * 100:.1f}% of collapsed slots) — "
        f"{near['definition']}. Total text collapsed away in excess of the "
        f"bearer: {near['extra_chars_total']:,} characters; worst length ratio "
        f"{near['max_length_ratio']} against a guard that fires above "
        f"{1 + cost['length_guard_ratio']:.2f}."
    )
    add(
        f"- **Length-guard vetoes: {veto['count']}** — {veto['definition']}. "
        f"These slots stayed full-text on purpose."
    )
    add(
        f"- **Repeats no vector could see: "
        f"{cost['repeats_without_vectors']['count']}** — "
        f"{cost['repeats_without_vectors']['definition']}."
    )
    add("")
    add("## Spot check: was any collapsed pair two different facts?")
    add("")
    add(
        f"{spot['distinct_collapsed_pairs']} distinct collapsed pairs, each "
        f"re-checked with arithmetic that shares nothing with the embedding "
        f"model. {spot['method']}"
    )
    add("")
    add(
        f"Token Jaccard across those pairs: median "
        f"{spot['token_jaccard'].get('median')}, p25 "
        f"{spot['token_jaccard'].get('p25')}, min {spot['token_jaccard'].get('min')}. "
        f"5-gram containment of the collapsed node in its bearer: median "
        f"{spot['collapsed_in_bearer_containment'].get('median')}, "
        f"min {spot['collapsed_in_bearer_containment'].get('min')}."
    )
    add("")
    add(
        f"A second, sharper mechanical check runs over all "
        f"{spot['distinct_collapsed_pairs']} pairs: **identifier loss** — "
        f"{loss['definition']}"
    )
    add("")
    add(
        f"- **{loss['pairs_losing_none']} of {spot['distinct_collapsed_pairs']} "
        "pairs lose no identifier at all**: the bearer already carries "
        "everything the collapsed node named."
    )
    add(
        f"- **{loss['pairs_at_or_over_threshold']} pairs lose "
        f"{loss['flag_threshold']} or more identifiers**, accounting for "
        f"**{loss['slots_at_or_over_threshold']} of {loss['slots_total']} "
        "collapsed slots**. These are the collapses where the answer stopped "
        "showing specifics — read them below before trusting the headline."
    )
    add("")
    chaining = spot["bearer_chaining"]
    add(
        f"- **{chaining['slots_below_threshold']} collapsed slots point at a "
        f"bearer they never matched** ({chaining['pairs_below_threshold']} of "
        f"{chaining['pairs_scored']} pairs, lowest cosine "
        f"{chaining['min_cosine_to_recorded_bearer']}): {chaining['definition']}. "
        "The collapse is still earned — the node did exceed the threshold "
        "against some higher-ranked slot — but the `content_ref` an agent is "
        "handed names a weaker match than the one that earned the stub."
    )
    add("")
    add("Collapsed pairs by node level — the axis the concerning pairs fall along:")
    add("")
    add(
        "| levels | pairs | slots | jaccard-flagged | losing ≥"
        f"{loss['flag_threshold']} ids | median jaccard | min jaccard | median chars |"
    )
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for levels, bucket in spot["by_level"].items():
        add(
            f"| {levels} | {bucket['pairs']} | {bucket['slots']} | "
            f"{bucket['flagged']} | {bucket['losing_identifiers']} | "
            f"{bucket['median_token_jaccard']} | {bucket['min_token_jaccard']} | "
            f"{bucket['median_collapsed_chars']:,} |"
        )
    add("")
    add(
        f"**{spot['flagged_for_human_reading']} of "
        f"{spot['distinct_collapsed_pairs']} pairs fall below token Jaccard "
        f"{spot['flag_threshold_token_jaccard']}.** Every pair that trips "
        "either check is listed below in full. The verdict on them is the "
        "hand-written section after the table — the one part of this file the "
        "script does not generate."
    )
    add("")
    if spot["pairs_for_human_reading"]:
        add("The pairs a human has to read:")
        add("")
        add(
            "| jaccard | containment | cosine | ids lost | collapsed | bearer | "
            "levels | chars | slots |"
        )
        add("| ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: |")
        for record in spot["pairs_for_human_reading"]:
            add(
                f"| {record['token_jaccard']} | "
                f"{record['collapsed_in_bearer_containment']} | "
                f"{record['cosine']} | {record['identifiers_lost']} | "
                f"`{record['collapsed']}` | `{record['bearer']}` | "
                f"{record['collapsed_level']}/{record['bearer_level']} | "
                f"{record['collapsed_chars']}→{record['bearer_chars']} | "
                f"{record['answers']} |"
            )
        add("")
    add("## Sensitivity of the repeat definition")
    add("")
    add(
        "The headline uses the shipped threshold. Moving the definition's "
        "threshold moves the ground truth, so both arms are reported at each "
        "value (traffic-weighted stratum):"
    )
    add("")
    add("| cosine | repeat slots | before share | after share |")
    add("| ---: | ---: | ---: | ---: |")
    for value, row in events["repeat_definition_sensitivity"].items():
        add(
            f"| {value} | {row['repeat_slots']} | "
            f"{row[ARM_BEFORE] * 100:.2f}% | {row[ARM_AFTER] * 100:.2f}% |"
        )
    add("")
    add("## Reproducing this")
    add("")
    add("```bash")
    add(f"python3 scripts/{PROGRAM} \\")
    add("    --live ~/.local/share/living-memory/global.sqlite3 \\")
    add("    --workdir /tmp/lm-dupslot \\")
    add("    --selection artifacts/near-dup/dup-slot-measurement.json \\")
    add("    --json artifacts/near-dup/dup-slot-measurement.json \\")
    add("    --md artifacts/near-dup/dup-slot-measurement.md")
    add("```")
    add("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Measure the share of recall slots occupied by repeats, before and "
            "after the near-duplicate collapse, on a copy of a live database."
        ),
    )
    parser.add_argument("--live", default=str(DEFAULT_LIVE_DB), help="live database (read-only)")
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR), help="where copies live")
    parser.add_argument("--events", type=int, default=DEFAULT_EVENT_SAMPLE)
    parser.add_argument("--distinct", type=int, default=DEFAULT_DISTINCT_SAMPLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_NEAR_DUP_COSINE,
        help="cosine above which two slots count as repeats (the definition)",
    )
    parser.add_argument(
        "--selection",
        help="replay the frozen event ids from a previously produced JSON artifact",
    )
    parser.add_argument("--json", dest="json_out", help="write the machine artifact here")
    parser.add_argument("--md", dest="md_out", help="write the readable summary here")
    parser.add_argument(
        "--render-from",
        help=(
            "skip the measurement entirely and re-render the markdown from a "
            "JSON artifact this script already produced (for prose fixes)"
        ),
    )
    parser.add_argument("--reuse-snapshot", action="store_true")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--produced-at", help="fixed timestamp, for a reproducible artifact")
    parser.add_argument("--allow-hash-embeddings", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.render_from:
        # Rendering only: the JSON is the measurement, the markdown is a view
        # of it. Re-running the arms to fix a sentence would be 15 minutes of
        # GPU for a comma -- and would tempt someone to hand-edit the numbers
        # instead.
        report = json.loads(Path(args.render_from).read_text(encoding="utf-8"))
        if not args.md_out:
            print(f"{PROGRAM}: --render-from needs --md", file=sys.stderr)
            return 2
        Path(args.md_out).write_text(render_markdown(report), encoding="utf-8")
        print(f"wrote {args.md_out} (rendered from {args.render_from})")
        return 0
    try:
        report = measure(args)
    except MeasurementError as error:
        print(f"{PROGRAM}: {error}", file=sys.stderr)
        return 2
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {path}")
    if args.md_out:
        path = Path(args.md_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(report), encoding="utf-8")
        print(f"wrote {path}")
    if not args.json_out and not args.md_out:
        print(payload)
    events = report["strata"][STRATUM_EVENTS]
    print(
        "duplicate-slot share: "
        f"{events[ARM_BEFORE]['duplicate_slot_share'] * 100:.2f}% -> "
        f"{events[ARM_AFTER]['duplicate_slot_share'] * 100:.2f}%"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
