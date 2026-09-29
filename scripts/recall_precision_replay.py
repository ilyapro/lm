#!/usr/bin/env python3
"""Live-path replay harness for the recall precision valves (goal recall-precision).

Why the live path: re-ranking recorded scores only sees the results that were
delivered, so it cannot show what takes the place of a cut result. Here every
eval/holdout recall event is re-issued through the real
``MemoryRecallService.memory_recall`` on a *counterfactual* copy of a frozen
snapshot, once per arm, with each arm's env valves applied around the call.

Snapshots (``snapshot``): made with the SQLite backup API. The sfx live DB is
opened only as ``file:...?mode=ro`` (``retrieval_harness.create_snapshot``).
alt's disk is nearly full, so nothing is written there: a remote python backs
the ``mode=ro`` store up into ``:memory:`` and streams ``Connection.serialize()``
to stdout over ssh; the bytes land on sfx. A ``manifest.json`` (captured_at,
sha256, row counts) sits next to the snapshots.

Split (``split``): per host, the replayable recall events at or after
:data:`MARKS_ERA_START` (non-empty results, not A/B by the conservative
session-level filter of ``scripts/explicit_feedback_agreement.py``) in
``(created_at, id)`` order: the first 50% train, the next 25% eval, the last
25% holdout. Boundaries are committed in ``artifacts/recall-precision/split.json``.

Counterfactual store for a cutoff (eval start or holdout start): a backup copy
of the snapshot with every row created at or after the cutoff removed from
``recall_feedback_marks`` (marked_at), ``query_irrelevance``,
``recall_credit_ledger`` / ``recall_explicit_credit`` (credited_at),
``query_anchors``, ``query_anchor_edges``, ``connections`` and
``recall_lookup_events`` (occurred_at). Rows created before the cutoff but
updated after it are counted as residual leakage, not rolled back. Traps from
LM memory 01M3H92E4X2SF3MMVY97P2NTSD: ``transport_session_id`` is stripped from
the replayed ambient context and the recall runs with ``log_access=False,
log_event=False``. Candidates created after the replayed event are dropped
before ranking (``HorizonRecallService``; counted as ``future_candidates_hidden``)
so they neither take nor leave a slot; any that still reach the output are
filtered and counted as ``future_filtered``. Marked nodes created at or after
the cutoff are reported and left out of the labelled metrics.

Arms (``--arm NAME:K=V,K=V``): an env dict applied around every recall of that
arm (previous values restored after). ``baseline`` is always present with
every valve in :data:`KNOWN_VALVES` unset. Each arm gets its own
``MemoryStore``/``MemoryRecallService`` so no per-instance cache crosses arms;
arm order alternates per event.

Metrics are per host and arm, never merged across hosts. See ``summarize``.

Usage::

    python3 scripts/recall_precision_replay.py snapshot --host sfx \\
        --source-db ~/.local/share/living-memory/global.sqlite3
    python3 scripts/recall_precision_replay.py snapshot --host alt --ssh alt \\
        --remote-db /home/user/.local/share/living-memory/global.sqlite3
    python3 scripts/recall_precision_replay.py split
    python3 scripts/recall_precision_replay.py run --host sfx --segment eval \\
        --arm gate04:LM_RECALL_MIN_SCORE=0.4 --sample 50 \\
        --out artifacts/recall-precision/smoke/sfx.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import explicit_feedback_agreement as efa  # noqa: E402
from living_memory.grounding import ground_token_sets, token_set  # noqa: E402
from living_memory.retrieval_harness import (  # noqa: E402
    backup_database,
    create_snapshot,
    describe_snapshot,
    manifest_path_for,
    sha256_file,
)

#: Start of the explicit-marks era (``recall_feedback_marks`` in production).
MARKS_ERA_START = "2026-09-27T14:45:00Z"
SNAPSHOT_DIR = Path("/home/sfx/p/ae/artifacts/recall-precision/snapshots")
SPLIT_PATH = REPO_ROOT / "artifacts" / "recall-precision" / "split.json"
SEGMENTS = ("train", "eval", "holdout")
HOSTS = ("sfx", "alt")
#: Every valve an arm may set; ``baseline`` has all of them unset.
KNOWN_VALVES = (
    "LM_RECALL_MIN_SCORE",
    "LM_RECALL_GATE_FORM",
    "LM_HUB_SUPPRESSION_FACTOR",
    "LM_HUB_MIN_QUERIES",
    "LM_RECALL_SCHEMA_DEDUP",
    "LM_QUERY_IRRELEVANCE_FACTOR",
)
#: ``(table, timestamp column)`` stripped at/after the cutoff when present.
STRIP_TABLES = (
    ("recall_feedback_marks", "marked_at"),
    ("query_irrelevance", "created_at"),
    ("recall_credit_ledger", "credited_at"),
    ("recall_explicit_credit", "credited_at"),
    ("query_anchor_edges", "created_at"),
    ("query_anchors", "created_at"),
    ("connections", "created_at"),
    ("recall_lookup_events", "occurred_at"),
)
#: Tables whose pre-cutoff rows may still carry post-cutoff updates.
UPDATED_COLUMNS = (
    ("query_irrelevance", "updated_at"),
    ("query_anchor_edges", "updated_at"),
    ("query_anchors", "updated_at"),
    ("connections", "updated_at"),
)
HUB_MIN_QUERIES = 3
RANK_BUCKETS = ((1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"), (6, 10, "6-10"), (11, 10**6, "11+"))
REMOTE_BACKUP = (
    "import sqlite3,sys\n"
    "src=sqlite3.connect('file:'+sys.argv[1]+'?mode=ro',uri=True)\n"
    "mem=sqlite3.connect(':memory:')\n"
    "src.backup(mem)\n"
    "src.close()\n"
    "sys.stdout.buffer.write(mem.serialize())\n"
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def stamp_prefix(value: str) -> str:
    """``YYYY-MM-DDTHH:MM:SS`` of a stamp: compares ``>=`` for every suffix form."""

    return efa.normalize_ts(value)[:19]


def open_readonly(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = 1")
    return connection


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


# --------------------------------------------------------------------- snapshots


def snapshot_local(source: str | Path, out: str | Path) -> dict[str, Any]:
    return create_snapshot(source, out)


def snapshot_remote(ssh_host: str, remote_db: str, out: str | Path, *, timeout: int = 1800) -> dict[str, Any]:
    """Stream a backup of ``remote_db`` from ``ssh_host`` into ``out`` (nothing written remotely)."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_name(out_path.name + ".partial")
    captured_at = utc_now()
    with partial.open("wb") as handle:
        subprocess.run(
            ["ssh", ssh_host, "python3", "-", remote_db],
            input=REMOTE_BACKUP.encode(),
            stdout=handle,
            check=True,
            timeout=timeout,
        )
    check = sqlite3.connect(str(partial))
    try:
        verdict = check.execute("PRAGMA quick_check").fetchone()[0]
    finally:
        check.close()
    if verdict != "ok":
        raise RuntimeError(f"streamed snapshot failed quick_check: {verdict}")
    partial.replace(out_path)
    manifest = describe_snapshot(out_path, captured_at=captured_at)
    manifest["source_path"] = f"{ssh_host}:{remote_db}"
    manifest["transport"] = "ssh backup->:memory: serialize() stream"
    manifest_path_for(out_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def write_snapshot_index(directory: Path) -> Path:
    index: dict[str, Any] = {}
    for host in HOSTS:
        sidecar = manifest_path_for(directory / f"{host}.sqlite3")
        if sidecar.exists():
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            index[host] = {
                key: data.get(key)
                for key in ("captured_at", "snapshot_sha256", "snapshot_bytes", "source_path", "row_counts", "recall_events_created_at")
            }
            connection = open_readonly(directory / f"{host}.sqlite3")
            try:
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                index[host]["row_counts"] = {
                    **(index[host]["row_counts"] or {}),
                    **{
                        table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                        for table in sorted({t for t, _ in STRIP_TABLES} | {"node_chunk_embeddings"})
                        if table in tables
                    },
                }
            finally:
                connection.close()
    path = directory / "manifest.json"
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------------- split


def window_events(store: efa.Store) -> list[efa.Event]:
    """Replayable events in the marks era, ``(created_at, id)`` order."""

    events = [store.events[event_id] for event_id in store.window_ids if store.events[event_id].query.strip()]
    events.sort(key=lambda event: (event.created_at, event.id))
    return events


def compute_split(events: Sequence[efa.Event]) -> dict[str, Any]:
    n = len(events)
    if n < 4:
        raise ValueError(f"too few events to split: {n}")
    eval_index, holdout_index = n // 2, (3 * n) // 4

    def point(index: int) -> dict[str, Any]:
        return {"index": index, "event_id": events[index].id, "created_at": events[index].created_at}

    return {
        "events": n,
        "first": point(0),
        "last": point(n - 1),
        "eval_start": point(eval_index),
        "holdout_start": point(holdout_index),
        "counts": {"train": eval_index, "eval": holdout_index - eval_index, "holdout": n - holdout_index},
    }


def segment_of(event: efa.Event, host_split: Mapping[str, Any]) -> str:
    key = (event.created_at, event.id)
    evl, hold = host_split["eval_start"], host_split["holdout_start"]
    if key < (evl["created_at"], evl["event_id"]):
        return "train"
    if key < (hold["created_at"], hold["event_id"]):
        return "eval"
    return "holdout"


def cutoff_for(host_split: Mapping[str, Any], segment: str) -> str:
    if segment == "eval":
        return host_split["eval_start"]["created_at"]
    if segment == "holdout":
        return host_split["holdout_start"]["created_at"]
    raise ValueError(f"no counterfactual cutoff for segment {segment!r}")


def load_host_store(snapshot: str | Path, since: str = MARKS_ERA_START) -> efa.Store:
    connection = open_readonly(snapshot)
    try:
        return efa.load_store(connection, since)
    finally:
        connection.close()


def build_split(snapshots: Mapping[str, str | Path]) -> dict[str, Any]:
    split: dict[str, Any] = {
        "rule": (
            "per host: recall events with created_at >= marks_era_start, non-empty recorded "
            "results, non-blank query, not A/B (explicit_feedback_agreement.load_store window), "
            "ordered by (created_at, id); first 50% train, next 25% eval, last 25% holdout. "
            "Membership is by (created_at, id) against the boundary events; events captured "
            "after the snapshot belong to holdout."
        ),
        "marks_era_start": MARKS_ERA_START,
        "hosts": {},
    }
    for host, snapshot in snapshots.items():
        store = load_host_store(snapshot)
        host_split = compute_split(window_events(store))
        host_split["snapshot_sha256"] = sha256_file(snapshot)
        host_split["excluded_ab"] = store.excluded_ab
        split["hosts"][host] = host_split
    return split


# ---------------------------------------------------------------- counterfactual


def build_counterfactual(snapshot: str | Path, target: str | Path, cutoff: str) -> dict[str, Any]:
    """Backup-copy ``snapshot`` to ``target`` and strip rows learned at/after ``cutoff``."""

    stamp = stamp_prefix(cutoff)
    backup_database(snapshot, target)
    connection = sqlite3.connect(str(target))
    removed: dict[str, int] = {}
    residual: dict[str, int] = {}
    try:
        with connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "query_anchors" in tables:
                late = "(SELECT id FROM query_anchors WHERE created_at >= ?)"
                for dependent in ("query_anchor_edges", "query_irrelevance"):
                    if dependent in tables:
                        removed[f"{dependent}_of_late_anchors"] = connection.execute(
                            f"DELETE FROM {dependent} WHERE anchor_id IN {late}", (stamp,)
                        ).rowcount
            for table, column in STRIP_TABLES:
                if table in tables and column in _table_columns(connection, table):
                    removed[table] = connection.execute(
                        f"DELETE FROM {table} WHERE {column} >= ?", (stamp,)
                    ).rowcount
            for table, column in UPDATED_COLUMNS:
                if table in tables and column in _table_columns(connection, table):
                    residual[table] = int(
                        connection.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} >= ?", (stamp,)).fetchone()[0]
                    )
    finally:
        connection.close()
    return {"cutoff": cutoff, "path": str(target), "removed": removed, "updated_after_cutoff_kept": residual}


# ------------------------------------------------------------------------- labels


@dataclass(frozen=True, slots=True)
class Label:
    node_id: str
    mark: str
    rank: int  # 1-based recorded rank


def event_labels(store: efa.Store) -> dict[str, dict[str, Label]]:
    """Latest accepted mark per ``(event, node)`` with its 1-based recorded rank."""

    latest: dict[tuple[str, str], efa.Mark] = {}
    for mark in store.marks or ():
        if not mark.accepted:
            continue
        key = (mark.event_id, mark.node_id)
        if key not in latest or mark.marked_at >= latest[key].marked_at:
            latest[key] = mark
    labels: dict[str, dict[str, Label]] = defaultdict(dict)
    for (event_id, node_id), mark in latest.items():
        event = store.events.get(event_id)
        if mark.rank is not None:
            rank = int(mark.rank) + 1
        elif event is not None and node_id in event.results:
            rank = event.results.index(node_id) + 1
        else:
            rank = 0
        labels[event_id][node_id] = Label(node_id=node_id, mark=mark.mark, rank=rank)
    return labels


def train_hubs(store: efa.Store, host_split: Mapping[str, Any], min_queries: int = HUB_MIN_QUERIES) -> set[str]:
    """Nodes with irrelevant marks from >= ``min_queries`` distinct queries and no used mark, on train marks."""

    queries: dict[str, set[str]] = defaultdict(set)
    used: Counter[str] = Counter()
    for mark in store.marks or ():
        event = store.events.get(mark.event_id)
        if not mark.accepted or event is None or segment_of(event, host_split) != "train":
            continue
        if mark.mark == "used":
            used[mark.node_id] += 1
        elif mark.mark == "irrelevant":
            queries[mark.node_id].add(" ".join(event.query.lower().split()))
    return {node for node, seen in queries.items() if len(seen) >= min_queries and used[node] == 0}


def rank_bucket(rank: int) -> str:
    for low, high, name in RANK_BUCKETS:
        if low <= rank <= high:
            return name
    return "unknown"


# ------------------------------------------------------------------------- replay


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    event_id: str
    created_at: str
    query: str
    scope: str | None
    ambient_context: dict[str, Any]
    depth: Any
    max_results: int
    recorded: tuple[str, ...]
    trace_id: str | None


def parse_depth(raw: Any) -> Any:
    if raw is None:
        return 1
    text = str(raw).strip()
    return int(text) if re.fullmatch(r"-?\d+", text) else text


def load_replay_events(snapshot: str | Path, event_ids: Sequence[str]) -> list[ReplayEvent]:
    connection = open_readonly(snapshot)
    connection.row_factory = sqlite3.Row
    try:
        out: list[ReplayEvent] = []
        for event_id in event_ids:
            row = connection.execute(
                "SELECT id, created_at, query, scope, requested_scope, ambient_context, depth, "
                "max_results, results, feedback_trace_id FROM recall_events WHERE id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                continue
            ambient = efa_json(row["ambient_context"], {})
            if not isinstance(ambient, dict):
                ambient = {}
            ambient.pop("transport_session_id", None)
            decoded = efa_json(row["results"], [])
            ranked = sorted(
                (item for item in decoded if isinstance(item, dict) and item.get("node_id")),
                key=lambda item: item.get("rank", 0),
            )
            out.append(
                ReplayEvent(
                    event_id=str(row["id"]),
                    created_at=efa.normalize_ts(row["created_at"]),
                    query=str(row["query"]),
                    scope=row["requested_scope"] or row["scope"],
                    ambient_context=ambient,
                    depth=parse_depth(row["depth"]),
                    max_results=int(row["max_results"] or 10),
                    recorded=tuple(str(item["node_id"]) for item in ranked),
                    trace_id=row["feedback_trace_id"],
                )
            )
        return out
    finally:
        connection.close()


def efa_json(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def parse_arm(spec: str) -> tuple[str, dict[str, str]]:
    """``NAME:K=V,K=V`` -> ``(NAME, {K: V})``; ``NAME`` alone is an empty env."""

    name, _, body = spec.partition(":")
    name = name.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"bad arm name in {spec!r}")
    env: dict[str, str] = {}
    for item in filter(None, (part.strip() for part in body.split(","))):
        key, sep, value = item.partition("=")
        if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key.strip()):
            raise ValueError(f"bad K=V {item!r} in arm {spec!r}")
        env[key.strip()] = value.strip()
    return name, env


@contextlib.contextmanager
def applied_env(env: Mapping[str, str]) -> Iterator[None]:
    """Unset every known valve, then apply ``env``; restore everything after."""

    keys = set(KNOWN_VALVES) | set(env)
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key in KNOWN_VALVES:
            os.environ.pop(key, None)
        os.environ.update(env)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass(slots=True)
class Delivered:
    node_id: str
    full: bool
    level: str
    title: str
    created_at: str


def schema_title(content: str) -> str:
    first = (content or "").strip().splitlines()[0] if (content or "").strip() else ""
    return " ".join(first.lower().split())


def replay(
    db_path: str | Path, arms: Mapping[str, Mapping[str, str]], events: Sequence[ReplayEvent]
) -> tuple[dict[str, list[list[Delivered]]], dict[str, list[float]], dict[str, int]]:
    from living_memory.config import MemoryConfig
    from living_memory.retrieval import MemoryRecallService
    from living_memory.storage import MemoryStore

    class HorizonRecallService(MemoryRecallService):
        """Ranks only candidates that existed at the replayed event's time.

        Nodes created after the event have no edges in the counterfactual copy
        (their edges are younger than the cutoff), so they can only enter as
        direct bm25/vector/trigger candidates; dropping them before ranking
        lets the slot go to what was really there instead of a hole.
        """

        horizon: str | None = None
        hidden = 0

        def rank_candidates(self, candidates, plan, **kwargs):  # type: ignore[no-untyped-def]
            if self.horizon is not None:
                kept = {
                    key: candidate
                    for key, candidate in candidates.items()
                    if efa.normalize_ts(str(candidate.node.created_at)) <= self.horizon
                }
                self.hidden += len(candidates) - len(kept)
                candidates = kept
            return super().rank_candidates(candidates, plan, **kwargs)

    names = list(arms)
    stores = {name: MemoryStore(MemoryConfig(db_path=Path(db_path))) for name in names}
    lists: dict[str, list[list[Delivered]]] = {name: [] for name in names}
    latency: dict[str, list[float]] = {name: [] for name in names}
    try:
        services = {name: HorizonRecallService(stores[name]) for name in names}
        for index, event in enumerate(events):
            for name in names if index % 2 == 0 else list(reversed(names)):
                services[name].horizon = event.created_at
                with applied_env(arms[name]):
                    started = time.perf_counter()
                    results = services[name].memory_recall(
                        event.query,
                        scope=event.scope,
                        ambient_context=dict(event.ambient_context),
                        depth=event.depth,
                        max_results=event.max_results,
                        log_access=False,
                        log_event=False,
                    )
                    latency[name].append(time.perf_counter() - started)
                lists[name].append(
                    [
                        Delivered(
                            node_id=result.node_id,
                            full=not getattr(result, "withheld", None),
                            level=str(result.node.level),
                            title=schema_title(result.node.content) if str(result.node.level) == "schema" else "",
                            created_at=efa.normalize_ts(str(result.node.created_at)),
                        )
                        for result in results
                    ]
                )
        hidden = {name: services[name].hidden for name in names}
    finally:
        for store in stores.values():
            store.close()
    return lists, latency, hidden


# -------------------------------------------------------------------- grounding


class EntrantGrader:
    """Grounded(n) - grounded(twin(n)) against the event's closing trace (efa method)."""

    def __init__(self, snapshot: str | Path, node_created: Mapping[str, str]) -> None:
        self.connection = open_readonly(snapshot)
        self.index = efa.VectorIndex(self.connection, node_created)
        self.contents = efa.Contents(self.connection)

    def close(self) -> None:
        self.connection.close()

    def grade(
        self, event: ReplayEvent, delivered: Sequence[str], entrants: Sequence[str]
    ) -> list[dict[str, Any]]:
        if not event.trace_id or not entrants:
            return []
        trace_tokens = self.contents.tokens(event.trace_id)
        trace_vector = self.index.vector(event.trace_id)
        if trace_tokens is None or trace_vector is None:
            return []
        cos_all = self.index.matrix @ trace_vector
        eligible = self.index.created < event.created_at
        for node_id in (*event.recorded, *delivered, event.trace_id):
            position = self.index.position.get(node_id)
            if position is not None:
                eligible[position] = False
        docs = {node_id: self.contents.tokens(node_id) for node_id in delivered}
        docs = {node_id: tokens for node_id, tokens in docs.items() if tokens is not None}
        out: list[dict[str, Any]] = []
        for node_id in entrants:
            vector = self.index.vector(node_id)
            if vector is None or node_id not in docs:
                out.append({"node_id": node_id, "graded": False})
                continue
            cos = float(vector @ trace_vector)
            twin = efa.pick_twin(cos, cos_all, eligible, self.index.ids, f"{event.event_id}:{node_id}")
            graded = dict(docs)
            twin_tokens = self.contents.tokens(twin) if twin else None
            if twin and twin_tokens is not None:
                graded[f"twin::{twin}"] = twin_tokens
            verdict = ground_token_sets(trace_tokens, graded, min_containment=efa.MIN_CONTAINMENT)
            out.append(
                {
                    "node_id": node_id,
                    "graded": True,
                    "grounded": bool(verdict[node_id].grounded),
                    "has_twin": twin_tokens is not None,
                    "twin_grounded": bool(verdict[f"twin::{twin}"].grounded) if twin_tokens is not None else False,
                }
            )
        return out


# ------------------------------------------------------------------------ metrics


def score_event(
    event: ReplayEvent,
    delivered: Sequence[Delivered],
    labels: Mapping[str, Label],
    hubs: set[str],
    cutoff: str,
    node_created: Mapping[str, str],
) -> dict[str, Any]:
    """Per-event row of one arm (entrant grading is attached by the caller)."""

    stamp = stamp_prefix(cutoff)
    future = [item for item in delivered if item.created_at and item.created_at > event.created_at]
    kept = [item for item in delivered if not (item.created_at and item.created_at > event.created_at)]
    full = [item for item in kept if item.full]
    full_ids = [item.node_id for item in full]
    top3 = set(full_ids[:3])
    counted = {
        node_id: label
        for node_id, label in labels.items()
        if stamp_prefix(node_created.get(node_id, "") or "0000") < stamp
    }
    by_rank: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    for node_id, label in counted.items():
        bucket = by_rank[rank_bucket(label.rank)]
        delivered_full = node_id in full_ids
        if label.mark == "irrelevant":
            totals["irr_marked"] += 1
            totals["irr_full"] += delivered_full
            totals["irr_top3"] += node_id in top3
            bucket["irr_marked"] += 1
            bucket["irr_full"] += delivered_full
        elif label.mark == "used":
            totals["used_marked"] += 1
            totals["used_full"] += delivered_full
            totals["used_lost"] += not delivered_full
            bucket["used_marked"] += 1
            bucket["used_full"] += delivered_full
    titles = Counter(item.title for item in full if item.level == "schema" and item.title)
    recorded = set(event.recorded)
    return {
        "event_id": event.event_id,
        "full": len(full),
        "stub": sum(1 for item in kept if not item.full),
        "future_filtered": len(future),
        "rank1": full_ids[0] if full_ids else None,
        "recorded_rank1": event.recorded[0] if event.recorded else None,
        "full_ids": full_ids,
        "labels_excluded_created_after_cutoff": len(labels) - len(counted),
        **{key: int(totals[key]) for key in ("irr_marked", "irr_full", "irr_top3", "used_marked", "used_full", "used_lost")},
        "by_rank": {name: dict(counter) for name, counter in by_rank.items()},
        "top3_full": len(full_ids[:3]),
        "hub_top3": sum(1 for node_id in full_ids[:3] if node_id in hubs),
        "dup_schema_slots": sum(count - 1 for count in titles.values()),
        "entrants": [node_id for node_id in full_ids if node_id not in recorded and node_id not in labels],
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize(rows: Sequence[Mapping[str, Any]], base_rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Aggregate one arm; with ``base_rows`` (baseline arm, same events) add deltas."""

    n = len(rows)
    total = Counter()
    for row in rows:
        for key in ("full", "stub", "future_filtered", "irr_marked", "irr_full", "irr_top3", "used_marked",
                    "used_full", "used_lost", "top3_full", "hub_top3", "dup_schema_slots",
                    "labels_excluded_created_after_cutoff"):
            total[key] += row[key]
    by_rank: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for bucket, counter in row["by_rank"].items():
            by_rank[bucket].update(counter)
    entrants = [grade for row in rows for grade in row.get("entrant_grades", ())]
    graded = [grade for grade in entrants if grade.get("graded")]
    with_twin = [grade for grade in graded if grade["has_twin"]]
    summary: dict[str, Any] = {
        "events": n,
        "full_slots_per_event": _ratio(total["full"], n),
        "stub_slots_per_event": _ratio(total["stub"], n),
        "future_filtered": int(total["future_filtered"]),
        "labelled_created_after_cutoff_excluded": int(total["labels_excluded_created_after_cutoff"]),
        "irrelevant": {
            "marked": int(total["irr_marked"]),
            "delivered_full": int(total["irr_full"]),
            "delivered_top3": int(total["irr_top3"]),
        },
        "used": {
            "marked": int(total["used_marked"]),
            "delivered_full": int(total["used_full"]),
            "lost": int(total["used_lost"]),
            "lost_share": _ratio(total["used_lost"], total["used_marked"]),
        },
        "rank1_kept_vs_recorded": _ratio(sum(1 for r in rows if r["rank1"] and r["rank1"] == r["recorded_rank1"]), n),
        "hub_top3_share": _ratio(total["hub_top3"], total["top3_full"]),
        "hub_top3_slots": int(total["hub_top3"]),
        "dup_schema_slots": int(total["dup_schema_slots"]),
        "entrants": {
            "per_event": _ratio(sum(len(row["entrants"]) for row in rows), n),
            "total": sum(len(row["entrants"]) for row in rows),
            "graded": len(graded),
            "grounded_rate": _ratio(sum(g["grounded"] for g in graded), len(graded)),
            "twin_grounded_rate": _ratio(sum(g["twin_grounded"] for g in with_twin), len(with_twin)),
            "excess": _ratio(
                sum(g["grounded"] - (g["twin_grounded"] if g["has_twin"] else 0) for g in graded), len(graded)
            ),
        },
        "by_recorded_rank": {
            name: {
                "irr_marked": by_rank[name]["irr_marked"],
                "irr_full": by_rank[name]["irr_full"],
                "irr_removed_share": _ratio(by_rank[name]["irr_marked"] - by_rank[name]["irr_full"], by_rank[name]["irr_marked"]),
                "used_marked": by_rank[name]["used_marked"],
                "used_full": by_rank[name]["used_full"],
                "used_lost_share": _ratio(by_rank[name]["used_marked"] - by_rank[name]["used_full"], by_rank[name]["used_marked"]),
            }
            for _, _, name in RANK_BUCKETS
            if name in by_rank
        },
    }
    if base_rows is not None:
        base = summarize(base_rows)
        kept = sum(1 for row, ref in zip(rows, base_rows, strict=True) if row["rank1"] and row["rank1"] == ref["rank1"])
        new_vs_base = [
            len(set(row["full_ids"]) - set(ref["full_ids"])) for row, ref in zip(rows, base_rows, strict=True)
        ]
        summary["vs_baseline"] = {
            "rank1_kept": _ratio(kept, n),
            "irrelevant_full_cut": _ratio(base["irrelevant"]["delivered_full"] - total["irr_full"], base["irrelevant"]["delivered_full"]),
            "irrelevant_top3_cut": _ratio(base["irrelevant"]["delivered_top3"] - total["irr_top3"], base["irrelevant"]["delivered_top3"]),
            "used_full_lost": _ratio(base["used"]["delivered_full"] - total["used_full"], base["used"]["delivered_full"]),
            "full_slots_cut": _ratio(total["full"] * -1 + base["full_slots_per_event"] * n, base["full_slots_per_event"] * n)
            if base["full_slots_per_event"]
            else None,
            "hub_top3_cut": _ratio(base["hub_top3_slots"] - total["hub_top3"], base["hub_top3_slots"]),
            "dup_schema_slots_cut": _ratio(base["dup_schema_slots"] - total["dup_schema_slots"], base["dup_schema_slots"]),
            "new_vs_baseline_per_event": _ratio(sum(new_vs_base), n),
        }
    return summary


# -------------------------------------------------------------------------- run


def run_host(
    *,
    host: str,
    snapshot: str | Path,
    host_split: Mapping[str, Any],
    segment: str,
    arms: Mapping[str, Mapping[str, str]],
    workdir: str | Path,
    sample: int | None = None,
    seed: int = 7,
    grade_entrants: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    snapshot_sha = sha256_file(snapshot)
    if host_split.get("snapshot_sha256") and host_split["snapshot_sha256"] != snapshot_sha:
        raise ValueError(f"{host}: snapshot sha256 differs from split.json")
    store = load_host_store(snapshot)
    events_all = [event for event in window_events(store) if segment_of(event, host_split) == segment]
    chosen = [event.id for event in events_all]
    if sample is not None and sample < len(chosen):
        chosen = sorted(random.Random(seed).sample(chosen, sample), key=lambda eid: (store.events[eid].created_at, eid))
    cutoff = cutoff_for(host_split, segment)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    counterfactual = build_counterfactual(snapshot, workdir / f"{host}-{segment}.sqlite3", cutoff)
    labels = event_labels(store)
    hubs = train_hubs(store, host_split)
    events = load_replay_events(snapshot, chosen)
    prepared = time.perf_counter()
    lists, latency, hidden = replay(counterfactual["path"], arms, events)
    replayed = time.perf_counter()
    grader = EntrantGrader(snapshot, store.node_created) if grade_entrants else None
    rows: dict[str, list[dict[str, Any]]] = {}
    try:
        for name in arms:
            rows[name] = []
            for event, delivered in zip(events, lists[name], strict=True):
                row = score_event(event, delivered, labels.get(event.event_id, {}), hubs, cutoff, store.node_created)
                if grader is not None:
                    row["entrant_grades"] = grader.grade(event, row["full_ids"], row["entrants"])
                rows[name].append(row)
    finally:
        if grader is not None:
            grader.close()
    finished = time.perf_counter()
    summaries = {
        name: summarize(rows[name], None if name == "baseline" else rows["baseline"]) for name in arms
    }
    for name in arms:
        values = sorted(latency[name])
        summaries[name]["latency_s"] = {
            "p50": round(values[len(values) // 2], 4) if values else None,
            "p90": round(values[int(0.9 * (len(values) - 1))], 4) if values else None,
            "total": round(sum(values), 2),
        }
        summaries[name]["future_candidates_hidden"] = hidden[name]
    return {
        "host": host,
        "segment": segment,
        "snapshot": str(snapshot),
        "snapshot_sha256": snapshot_sha,
        "cutoff": cutoff,
        "segment_events": len(events_all),
        "replayed_events": len(events),
        "sample": sample,
        "seed": seed,
        "arms": {name: dict(env) for name, env in arms.items()},
        "hubs_train_defined": len(hubs),
        "counterfactual": counterfactual,
        "summary": summaries,
        "events": {name: [{k: v for k, v in row.items() if k != "full_ids"} for row in rows[name]] for name in arms},
        "runtime_s": {
            "prepare": round(prepared - started, 2),
            "replay": round(replayed - prepared, 2),
            "score": round(finished - replayed, 2),
            "total": round(finished - started, 2),
            "replay_per_event_arm": round((replayed - prepared) / max(1, len(events) * len(arms)), 4),
        },
        "embedding_backend": os.environ.get("LIVING_MEMORY_EMBEDDING_BACKEND") or "auto",
        "generated_at": utc_now(),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(results: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Recall precision live-path replay", ""]
    lines.append("Per host, never merged. Labels: accepted explicit marks on the replayed event; marks are position-biased, see the rank breakdown.")
    lines.append("")
    for result in results:
        lines.append(f"## {result['host']} — {result['segment']}")
        lines.append("")
        lines.append(
            f"- snapshot `{result['snapshot_sha256'][:12]}`, cutoff {result['cutoff']}, "
            f"events {result['replayed_events']}/{result['segment_events']} (sample={result['sample']}, seed={result['seed']}), "
            f"train-defined hubs {result['hubs_train_defined']}"
        )
        cf = result["counterfactual"]
        lines.append(f"- stripped at/after cutoff: {json.dumps(cf['removed'], sort_keys=True)}")
        lines.append(f"- pre-cutoff rows updated after cutoff (kept): {json.dumps(cf['updated_after_cutoff_kept'], sort_keys=True)}")
        runtime = result["runtime_s"]
        lines.append(
            f"- runtime: total {runtime['total']} s (prepare {runtime['prepare']}, replay {runtime['replay']}, "
            f"score {runtime['score']}); {runtime['replay_per_event_arm']} s per event×arm"
        )
        lines.append("")
        lines.append("| arm | full/ev | stub/ev | irr marked | irr full | irr top3 | used marked | used lost | rank1=recorded | hub top3 share | dup schema | entrants/ev | entrant excess | future filtered | labels excl. |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, s in result["summary"].items():
            lines.append(
                "| " + " | ".join(
                    _fmt(v)
                    for v in (
                        name, s["full_slots_per_event"], s["stub_slots_per_event"], s["irrelevant"]["marked"],
                        s["irrelevant"]["delivered_full"], s["irrelevant"]["delivered_top3"], s["used"]["marked"],
                        s["used"]["lost"], s["rank1_kept_vs_recorded"], s["hub_top3_share"], s["dup_schema_slots"],
                        s["entrants"]["per_event"], s["entrants"]["excess"], s["future_filtered"],
                        s["labelled_created_after_cutoff_excluded"],
                    )
                ) + " |"
            )
        deltas = {name: s["vs_baseline"] for name, s in result["summary"].items() if "vs_baseline" in s}
        if deltas:
            lines.append("")
            lines.append("| arm vs baseline | rank1 kept | irr full cut | irr top3 cut | used lost | full slots cut | hub top3 cut | dup schema cut | new/ev |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for name, d in deltas.items():
                lines.append(
                    "| " + " | ".join(
                        _fmt(v) for v in (name, d["rank1_kept"], d["irrelevant_full_cut"], d["irrelevant_top3_cut"],
                                          d["used_full_lost"], d["full_slots_cut"], d["hub_top3_cut"],
                                          d["dup_schema_slots_cut"], d["new_vs_baseline_per_event"])
                    ) + " |"
                )
        lines.append("")
        lines.append("By recorded rank (irr removed share / used lost share):")
        lines.append("")
        lines.append("| arm | " + " | ".join(name for _, _, name in RANK_BUCKETS) + " |")
        lines.append("|---|" + "---|" * len(RANK_BUCKETS))
        for name, s in result["summary"].items():
            cells = []
            for _, _, bucket in RANK_BUCKETS:
                b = s["by_recorded_rank"].get(bucket)
                cells.append(
                    "—" if b is None else f"{_fmt(b['irr_removed_share'])} (n={b['irr_marked']}) / {_fmt(b['used_lost_share'])} (n={b['used_marked']})"
                )
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="freeze a host's store via the SQLite backup API")
    snap.add_argument("--host", choices=HOSTS, required=True)
    snap.add_argument("--source-db", help="local live DB (opened mode=ro)")
    snap.add_argument("--ssh", help="ssh host to stream the backup from")
    snap.add_argument("--remote-db", help="DB path on the ssh host")
    snap.add_argument("--dir", type=Path, default=SNAPSHOT_DIR)
    split = sub.add_parser("split", help="write split.json from the snapshots")
    split.add_argument("--dir", type=Path, default=SNAPSHOT_DIR)
    split.add_argument("--out", type=Path, default=SPLIT_PATH)
    run = sub.add_parser("run", help="replay arms on one host's counterfactual snapshot")
    run.add_argument("--host", choices=HOSTS, required=True)
    run.add_argument("--snapshot", type=Path, help="default: <dir>/<host>.sqlite3")
    run.add_argument("--dir", type=Path, default=SNAPSHOT_DIR)
    run.add_argument("--split", type=Path, default=SPLIT_PATH)
    run.add_argument("--segment", choices=("eval", "holdout"), default="eval")
    run.add_argument("--arm", action="append", default=[], help="NAME:K=V,K=V (repeatable)")
    run.add_argument("--sample", type=int)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--no-grade", action="store_true", help="skip entrant grounding")
    run.add_argument("--workdir", type=Path)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--markdown", type=Path)
    render = sub.add_parser("render", help="markdown from run JSON files")
    render.add_argument("--result", type=Path, action="append", required=True)
    render.add_argument("--markdown", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "snapshot":
        out = args.dir / f"{args.host}.sqlite3"
        if args.ssh:
            if not args.remote_db:
                raise SystemExit("--remote-db is required with --ssh")
            manifest = snapshot_remote(args.ssh, args.remote_db, out)
        elif args.source_db:
            manifest = snapshot_local(Path(args.source_db).expanduser(), out)
        else:
            raise SystemExit("one of --source-db / --ssh is required")
        write_snapshot_index(args.dir)
        print(json.dumps({k: manifest[k] for k in ("captured_at", "snapshot_sha256", "row_counts")}, indent=2))
        return 0
    if args.command == "split":
        snapshots = {host: args.dir / f"{host}.sqlite3" for host in HOSTS if (args.dir / f"{host}.sqlite3").exists()}
        split = build_split(snapshots)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({host: data["counts"] for host, data in split["hosts"].items()}))
        return 0
    if args.command == "run":
        arms: dict[str, dict[str, str]] = {"baseline": {}}
        for spec in args.arm:
            name, env = parse_arm(spec)
            if name == "baseline" and env:
                raise SystemExit("the baseline arm has every valve unset")
            arms[name] = env
        split = json.loads(args.split.read_text(encoding="utf-8"))
        snapshot = args.snapshot or args.dir / f"{args.host}.sqlite3"
        workdir = args.workdir or Path(tempfile.mkdtemp(prefix="lm-precision-replay-"))
        try:
            result = run_host(
                host=args.host,
                snapshot=snapshot,
                host_split=split["hosts"][args.host],
                segment=args.segment,
                arms=arms,
                workdir=workdir,
                sample=args.sample,
                seed=args.seed,
                grade_entrants=not args.no_grade,
            )
        finally:
            if args.workdir is None:
                shutil.rmtree(workdir, ignore_errors=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.markdown:
            args.markdown.write_text(render_markdown([result]), encoding="utf-8")
        print(json.dumps({"host": args.host, "runtime_s": result["runtime_s"], "baseline": result["summary"]["baseline"]["irrelevant"]}))
        return 0
    if args.command == "render":
        results = [json.loads(path.read_text(encoding="utf-8")) for path in args.result]
        args.markdown.write_text(render_markdown(results), encoding="utf-8")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
