#!/usr/bin/env python3
"""Snapshot replay that decides ``LM_IMPLICIT_LINK_POLICY`` (``all`` vs ``credited``).

``apply_pending_recall_feedback`` links the closing trace with a ``related``
edge (``metadata.basis = implicit_recall_feedback``) to every resolved
delivered node. The ``credited`` valve would link only the nodes that hold a
``recall_credit_ledger`` row for the event. This script builds both graphs
from one frozen snapshot and replays later combat recall traffic through the
real retrieval path (``MemoryRecallService.memory_recall``) on each one.

The method, cutoff, A/B filter, labels and decision rule are preregistered in
``artifacts/explicit-feedback/link-policy/preregistration.md``. This module
implements that text; ``report.md`` quotes it verbatim.

Snapshot discipline: the source DB is opened only as ``file:...?mode=ro``
(through ``retrieval_harness.create_snapshot`` / ``backup_database``). Arms are
separate backup copies. Only the copies are mutated, and ``MemoryStore`` only
ever opens a copy.

Usage (per host, never merged)::

    python3 scripts/implicit_link_policy_replay.py snapshot \\
        --source-db ~/.local/share/living-memory/global.sqlite3 --out /tmp/lp/sfx.sqlite3
    python3 scripts/implicit_link_policy_replay.py run --host sfx \\
        --snapshot /tmp/lp/sfx.sqlite3 --out artifacts/explicit-feedback/link-policy/sfx.json
    python3 scripts/implicit_link_policy_replay.py render \\
        --result artifacts/explicit-feedback/link-policy/sfx.json \\
        --result artifacts/explicit-feedback/link-policy/alt.json \\
        --json artifacts/explicit-feedback/link-policy/report.json \\
        --markdown artifacts/explicit-feedback/link-policy/report.md
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from living_memory.grounding import ground_results  # noqa: E402
from living_memory.retrieval_harness import (  # noqa: E402
    backup_database,
    create_snapshot,
    load_manifest,
)

IMPLICIT_BASIS = "implicit_recall_feedback"
DEFAULT_CUTOFF = "2026-09-20T00:00:00Z"
LINK_TIME_SLACK = timedelta(seconds=60)
PREREG_PATH = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
    / "explicit-feedback"
    / "link-policy"
    / "preregistration.md"
)

# Preregistered A/B filter (conservative; see preregistration.md).
AB_SCOPES = frozenset({"project:target", "project:repo", "project:x", "project:benchmark"})
AB_QUERY_RE = re.compile(
    r"ledger|billing|tree-context|fixture|kit acceptance|acceptance kit|evaluation-live|development-live",
    re.IGNORECASE,
)
AB_NODE_CONTEXT_RE = re.compile(r"fixture|tree-context-ab", re.IGNORECASE)
AB_RUN_RE = re.compile(r"-live-|--repaired-|baseline-r\d|ab-", re.IGNORECASE)

# Preregistered decision rule.
MIN_LABELLED_EVENTS = 100
EPS_MRR = 0.01
EPS_HIT3 = 0.01
MIN_UNCREDITED_GRAPH_DROP = 0.05
MAX_LATENCY_RATIO = 1.10
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 7


def _instant(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def open_readonly(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


# ---------------------------------------------------------------------------
# A/B filter
# ---------------------------------------------------------------------------


def event_trips_ab(scope: str | None, requested_scope: str | None, query: str | None) -> bool:
    if (scope or "") in AB_SCOPES or (requested_scope or "") in AB_SCOPES:
        return True
    return bool(AB_QUERY_RE.search(query or ""))


def node_trips_ab(scope: str | None, context: Mapping[str, Any], raw_context: str) -> bool:
    if (scope or "") in AB_SCOPES:
        return True
    if AB_NODE_CONTEXT_RE.search(raw_context or ""):
        return True
    run = context.get("run")
    return isinstance(run, str) and bool(AB_RUN_RE.search(run))


def ab_transport_sessions(connection: sqlite3.Connection) -> set[str]:
    """Transport sessions with at least one event or written node tripping a rule."""

    flagged: set[str] = set()
    for row in connection.execute(
        "SELECT transport_session_id, scope, requested_scope, query FROM recall_events "
        "WHERE transport_session_id IS NOT NULL AND transport_session_id != ''"
    ):
        if event_trips_ab(row["scope"], row["requested_scope"], row["query"]):
            flagged.add(str(row["transport_session_id"]))
    for row in connection.execute(
        "SELECT scope, context FROM nodes WHERE context LIKE '%transport_session_id%'"
    ):
        context = _json(row["context"], {})
        if not isinstance(context, dict):
            continue
        session = context.get("transport_session_id")
        if not session:
            continue
        if node_trips_ab(row["scope"], context, row["context"] or ""):
            flagged.add(str(session))
    return flagged


# ---------------------------------------------------------------------------
# Counterfactual edge selection
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EdgePlan:
    """Which pre-cutoff implicit edges the ``credited`` arm drops, and why."""

    cutoff: str
    ledger_start: str | None
    implicit_pre_cutoff: int = 0
    kept_ledger: int = 0
    dropped_ledger: int = 0
    kept_grounded_pre_ledger: int = 0
    dropped_pre_ledger: int = 0
    eventual_credit_only: int = 0
    drop_ids: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "cutoff": self.cutoff,
            "ledger_start": self.ledger_start,
            "implicit_pre_cutoff": self.implicit_pre_cutoff,
            "kept_ledger_era": self.kept_ledger,
            "dropped_ledger_era": self.dropped_ledger,
            "kept_pre_ledger_grounded": self.kept_grounded_pre_ledger,
            "dropped_pre_ledger": self.dropped_pre_ledger,
            "ledger_credit_only_after_link": self.eventual_credit_only,
            "dropped_total": len(self.drop_ids),
        }


def plan_credited_drops(connection: sqlite3.Connection, cutoff: str) -> EdgePlan:
    """Select pre-cutoff implicit edges that had no credit when they were linked.

    Reads only. See preregistration.md ("Arm credited") for the rule.
    """

    cutoff_at = _instant(cutoff)
    start_row = connection.execute(
        "SELECT MIN(credited_at) FROM recall_credit_ledger"
    ).fetchone()
    ledger_start = start_row[0] if start_row and start_row[0] else None
    ledger_start_at = _instant(ledger_start) if ledger_start else None
    plan = EdgePlan(cutoff=_iso(cutoff_at), ledger_start=ledger_start)

    edges = connection.execute(
        "SELECT id, source_id, target_id, metadata, created_at FROM connections "
        "WHERE type = 'related' AND json_extract(metadata, '$.basis') = ? AND created_at < ?",
        (IMPLICIT_BASIS, _iso(cutoff_at)),
    ).fetchall()
    plan.implicit_pre_cutoff = len(edges)

    closed_by: dict[str, set[str]] = defaultdict(set)
    for row in connection.execute(
        "SELECT id, feedback_trace_id FROM recall_events WHERE feedback_trace_id IS NOT NULL"
    ):
        closed_by[str(row["feedback_trace_id"])].add(str(row["id"]))

    ledger: dict[tuple[str, str], str] = {}
    for row in connection.execute(
        "SELECT recall_event_id, node_id, credited_at FROM recall_credit_ledger"
    ):
        ledger[(str(row["recall_event_id"]), str(row["node_id"]))] = str(row["credited_at"])

    pre_ledger: list[tuple[sqlite3.Row, set[str]]] = []
    for edge in edges:
        metadata = _json(edge["metadata"], {})
        events = set(closed_by.get(str(edge["source_id"]), set()))
        meta_event = metadata.get("recall_event_id") if isinstance(metadata, dict) else None
        if meta_event:
            events.add(str(meta_event))
        created_at = _instant(edge["created_at"])
        if ledger_start_at is None or created_at < ledger_start_at:
            pre_ledger.append((edge, events))
            continue
        target = str(edge["target_id"])
        link_deadline = created_at + LINK_TIME_SLACK
        credited_at_link = False
        credited_later = False
        for event_id in events:
            stamp = ledger.get((event_id, target))
            if stamp is None:
                continue
            if _instant(stamp) <= link_deadline:
                credited_at_link = True
                break
            credited_later = True
        if credited_at_link:
            plan.kept_ledger += 1
        else:
            plan.dropped_ledger += 1
            plan.drop_ids.append(str(edge["id"]))
            if credited_later:
                plan.eventual_credit_only += 1

    if pre_ledger:
        grounded = _pre_ledger_grounded_pairs(connection, pre_ledger)
        for edge, _events in pre_ledger:
            if (str(edge["source_id"]), str(edge["target_id"])) in grounded:
                plan.kept_grounded_pre_ledger += 1
            else:
                plan.dropped_pre_ledger += 1
                plan.drop_ids.append(str(edge["id"]))
    return plan


def _pre_ledger_grounded_pairs(
    connection: sqlite3.Connection, pre_ledger: Sequence[tuple[sqlite3.Row, set[str]]]
) -> set[tuple[str, str]]:
    """(trace, node) pairs the live grader marks grounded for some closed event."""

    from living_memory.feedback import RECALL_CREDIT_MIN_CONTAINMENT

    events_by_trace: dict[str, set[str]] = defaultdict(set)
    for edge, events in pre_ledger:
        events_by_trace[str(edge["source_id"])].update(events)

    grounded: set[tuple[str, str]] = set()
    content_cache: dict[str, str | None] = {}

    def content(node_id: str) -> str | None:
        if node_id not in content_cache:
            row = connection.execute(
                "SELECT content FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()
            content_cache[node_id] = None if row is None else str(row["content"])
        return content_cache[node_id]

    for trace_id, event_ids in events_by_trace.items():
        trace_content = content(trace_id)
        if trace_content is None:
            continue
        for event_id in sorted(event_ids):
            row = connection.execute(
                "SELECT results FROM recall_events WHERE id = ?", (event_id,)
            ).fetchone()
            if row is None:
                continue
            results: dict[str, str] = {}
            for result in _json(row["results"], []):
                node_id = str((result or {}).get("node_id") or "")
                if not node_id or node_id == trace_id or node_id in results:
                    continue
                node_content = content(node_id)
                if node_content is not None:
                    results[node_id] = node_content
            if not results:
                continue
            verdicts = ground_results(
                trace_content, results, min_containment=RECALL_CREDIT_MIN_CONTAINMENT
            )
            for node_id, verdict in verdicts.items():
                if verdict.grounded:
                    grounded.add((trace_id, node_id))
        # Content of one trace's results is rarely shared; keep memory flat.
        if len(content_cache) > 20000:
            content_cache.clear()
    return grounded


def truncate_at_cutoff(connection: sqlite3.Connection, cutoff: str) -> dict[str, int]:
    """Both arms: forget every edge and anchor learned at or after the cutoff."""

    stamp = _iso(_instant(cutoff))
    removed = {}
    with connection:
        removed["connections"] = connection.execute(
            "DELETE FROM connections WHERE created_at >= ?", (stamp,)
        ).rowcount
        removed["query_anchor_edges"] = connection.execute(
            "DELETE FROM query_anchor_edges WHERE created_at >= ? OR anchor_id IN "
            "(SELECT id FROM query_anchors WHERE created_at >= ?)",
            (stamp, stamp),
        ).rowcount
        removed["query_anchors"] = connection.execute(
            "DELETE FROM query_anchors WHERE created_at >= ?", (stamp,)
        ).rowcount
    return removed


def drop_edges(connection: sqlite3.Connection, edge_ids: Iterable[str]) -> int:
    ids = list(edge_ids)
    removed = 0
    with connection:
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            removed += connection.execute(
                f"DELETE FROM connections WHERE id IN ({','.join('?' * len(chunk))})", chunk
            ).rowcount
    return removed


def edge_counts(connection: sqlite3.Connection) -> dict[str, int]:
    total = connection.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
    implicit = connection.execute(
        "SELECT COUNT(*) FROM connections WHERE json_extract(metadata, '$.basis') = ?",
        (IMPLICIT_BASIS,),
    ).fetchone()[0]
    related = connection.execute(
        "SELECT COUNT(*) FROM connections WHERE type = 'related'"
    ).fetchone()[0]
    return {"connections": int(total), "related": int(related), "implicit": int(implicit)}


def build_arms(snapshot: str | Path, workdir: str | Path, cutoff: str) -> dict[str, Any]:
    """Make ``all.sqlite3`` and ``credited.sqlite3`` in ``workdir`` from ``snapshot``.

    The snapshot is only read (``mode=ro``); both arms are backup copies.
    """

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    reader = open_readonly(snapshot)
    try:
        plan = plan_credited_drops(reader, cutoff)
    finally:
        reader.close()

    all_path = workdir / "all.sqlite3"
    credited_path = workdir / "credited.sqlite3"
    backup_database(snapshot, all_path)
    writer = sqlite3.connect(str(all_path))
    try:
        truncated = truncate_at_cutoff(writer, cutoff)
        all_counts = edge_counts(writer)
    finally:
        writer.close()
    backup_database(all_path, credited_path)
    writer = sqlite3.connect(str(credited_path))
    try:
        dropped = drop_edges(writer, plan.drop_ids)
        credited_counts = edge_counts(writer)
    finally:
        writer.close()
    return {
        "paths": {"all": str(all_path), "credited": str(credited_path)},
        "plan": plan.summary(),
        "truncated_at_cutoff": truncated,
        "credited_edges_dropped": dropped,
        "edge_counts": {"all": all_counts, "credited": credited_counts},
    }


# ---------------------------------------------------------------------------
# Eval traffic and labels
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvalEvent:
    event_id: str
    created_at: str
    query: str
    scope: str | None
    ambient_context: dict[str, Any]
    depth: Any
    max_results: int
    credited: frozenset[str]


def load_eval_events(
    connection: sqlite3.Connection, cutoff: str, *, limit: int | None = None
) -> tuple[list[EvalEvent], dict[str, int]]:
    stamp = _iso(_instant(cutoff))
    flagged = ab_transport_sessions(connection)
    credits: dict[str, set[str]] = defaultdict(set)
    for row in connection.execute(
        "SELECT l.recall_event_id, l.node_id FROM recall_credit_ledger l "
        "JOIN nodes n ON n.id = l.node_id WHERE l.basis IN ('grounded', 'lookup')"
    ):
        credits[str(row[0])].add(str(row[1]))
    stats = {"post_cutoff": 0, "excluded_ab": 0, "empty_results": 0, "kept": 0, "labelled": 0}
    events: list[EvalEvent] = []
    for row in connection.execute(
        "SELECT id, created_at, query, scope, requested_scope, ambient_context, depth, "
        "max_results, results, transport_session_id FROM recall_events "
        "WHERE created_at >= ? ORDER BY created_at, id",
        (stamp,),
    ):
        stats["post_cutoff"] += 1
        session = row["transport_session_id"]
        if (session and str(session) in flagged) or event_trips_ab(
            row["scope"], row["requested_scope"], row["query"]
        ):
            stats["excluded_ab"] += 1
            continue
        if not _json(row["results"], []):
            stats["empty_results"] += 1
            continue
        ambient = _json(row["ambient_context"], {})
        if not isinstance(ambient, dict):
            ambient = {}
        ambient.pop("transport_session_id", None)
        credited = frozenset(credits.get(str(row["id"]), set()))
        events.append(
            EvalEvent(
                event_id=str(row["id"]),
                created_at=str(row["created_at"]),
                query=str(row["query"]),
                scope=row["requested_scope"] or row["scope"],
                ambient_context=ambient,
                depth=row["depth"] if row["depth"] is not None else 1,
                max_results=int(row["max_results"] or 10),
                credited=credited,
            )
        )
        stats["kept"] += 1
        if credited:
            stats["labelled"] += 1
        if limit is not None and len(events) >= limit:
            break
    return events, stats


# ---------------------------------------------------------------------------
# Replay and metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventOutcome:
    event_id: str
    labelled: bool
    rr: float
    hit1: int
    hit3: int
    credited_returned: int
    credited_total: int
    graph_hits_credited: int
    graph_hits_uncredited: int
    graph_only_uncredited: int
    results: int
    latency_s: float


def score_event(
    event: EvalEvent,
    ranked: Sequence[tuple[str, Sequence[str]]],
    latency_s: float,
) -> EventOutcome:
    """Score one replayed list of ``(node_id, methods)`` against the event's labels."""

    rr = 0.0
    first: int | None = None
    credited_returned = 0
    graph_c = graph_u = graph_only_u = 0
    for index, (node_id, methods) in enumerate(ranked):
        is_credited = node_id in event.credited
        has_graph = "graph" in methods
        if is_credited:
            credited_returned += 1
            if first is None:
                first = index + 1
        if has_graph:
            if is_credited:
                graph_c += 1
            else:
                graph_u += 1
                if set(methods) <= {"graph"}:
                    graph_only_u += 1
    if first is not None:
        rr = 1.0 / first
    return EventOutcome(
        event_id=event.event_id,
        labelled=bool(event.credited),
        rr=rr,
        hit1=int(first == 1),
        hit3=int(first is not None and first <= 3),
        credited_returned=credited_returned,
        credited_total=len(event.credited),
        graph_hits_credited=graph_c,
        graph_hits_uncredited=graph_u,
        graph_only_uncredited=graph_only_u,
        results=len(ranked),
        latency_s=latency_s,
    )


def replay_arms(
    db_paths: Mapping[str, str | Path], events: Sequence[EvalEvent]
) -> dict[str, list[EventOutcome]]:
    """Replay every event through the real recall path on each arm copy.

    Both stores stay open and the arm order alternates per event, so machine
    load drifts hit both arms alike and the latency comparison stays paired.
    """

    from living_memory.config import MemoryConfig
    from living_memory.retrieval import MemoryRecallService
    from living_memory.storage import MemoryStore

    arms = list(db_paths)
    stores = {arm: MemoryStore(MemoryConfig(db_path=Path(db_paths[arm]))) for arm in arms}
    try:
        services = {arm: MemoryRecallService(stores[arm]) for arm in arms}
        created: dict[str, str] = {}
        outcomes: dict[str, list[EventOutcome]] = {arm: [] for arm in arms}
        for index, event in enumerate(events):
            order = arms if index % 2 == 0 else list(reversed(arms))
            for arm in order:
                started = time.perf_counter()
                results = services[arm].memory_recall(
                    event.query,
                    scope=event.scope,
                    ambient_context=dict(event.ambient_context),
                    depth=event.depth,
                    max_results=event.max_results,
                    log_access=False,
                    log_event=False,
                )
                latency = time.perf_counter() - started
                ranked: list[tuple[str, tuple[str, ...]]] = []
                for result in results:
                    node_id = result.node_id
                    if node_id not in created:
                        created[node_id] = str(result.node.created_at)
                    if created[node_id] > event.created_at:
                        continue
                    ranked.append((node_id, tuple(str(m) for m in result.methods)))
                outcomes[arm].append(score_event(event, ranked, latency))
        return outcomes
    finally:
        for store in stores.values():
            store.close()


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def summarize(outcomes: Sequence[EventOutcome]) -> dict[str, Any]:
    labelled = [o for o in outcomes if o.labelled]
    n = len(outcomes)
    nl = len(labelled)
    gc = sum(o.graph_hits_credited for o in outcomes)
    gu = sum(o.graph_hits_uncredited for o in outcomes)
    credited_returned = sum(o.credited_returned for o in outcomes)
    uncredited_returned = sum(o.results for o in outcomes) - credited_returned
    latencies = [o.latency_s for o in outcomes]

    def mean(values: Iterable[float], count: int) -> float | None:
        return sum(values) / count if count else None

    return {
        "events": n,
        "labelled_events": nl,
        "mrr": mean((o.rr for o in labelled), nl),
        "hit@1": mean((o.hit1 for o in labelled), nl),
        "hit@3": mean((o.hit3 for o in labelled), nl),
        "credited_node_recall": (
            sum(o.credited_returned for o in labelled) / sum(o.credited_total for o in labelled)
            if labelled
            else None
        ),
        "graph_hits_credited": gc,
        "graph_hits_uncredited": gu,
        "graph_only_uncredited": sum(o.graph_only_uncredited for o in outcomes),
        "uncredited_graph_hits_per_event": mean((o.graph_hits_uncredited for o in outcomes), n),
        "graph_share_of_credited_hits": gc / credited_returned if credited_returned else None,
        "graph_share_of_uncredited_hits": gu / uncredited_returned if uncredited_returned else None,
        "latency_p50_ms": None if not latencies else 1000 * statistics.median(latencies),
        "latency_p95_ms": None if not latencies else 1000 * (_percentile(latencies, 0.95) or 0.0),
    }


def paired_bootstrap(
    base: Sequence[EventOutcome],
    other: Sequence[EventOutcome],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """95% CIs for credited − all deltas, resampling events jointly."""

    pairs_l = [(a, b) for a, b in zip(base, other, strict=True) if a.labelled]
    pairs_all = list(zip(base, other, strict=True))
    rng = random.Random(seed)

    def ci(pairs: Sequence[tuple[EventOutcome, EventOutcome]], attr: str) -> dict[str, Any]:
        if not pairs:
            return {"delta": None, "lo": None, "hi": None}
        diffs = [float(getattr(b, attr)) - float(getattr(a, attr)) for a, b in pairs]
        point = sum(diffs) / len(diffs)
        draws = []
        for _ in range(resamples):
            sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
            draws.append(sum(sample) / len(sample))
        draws.sort()
        return {
            "delta": point,
            "lo": draws[int(0.025 * resamples)],
            "hi": draws[int(0.975 * resamples) - 1],
        }

    return {
        "mrr": ci(pairs_l, "rr"),
        "hit@3": ci(pairs_l, "hit3"),
        "uncredited_graph_hits_per_event": ci(pairs_all, "graph_hits_uncredited"),
    }


def decide(all_summary: Mapping[str, Any], credited_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the preregistered rule to one host."""

    checks: dict[str, Any] = {}
    labelled = int(all_summary.get("labelled_events") or 0)
    checks["R0_labelled>=100"] = labelled >= MIN_LABELLED_EVENTS
    d_mrr = (credited_summary["mrr"] or 0.0) - (all_summary["mrr"] or 0.0)
    d_hit3 = (credited_summary["hit@3"] or 0.0) - (all_summary["hit@3"] or 0.0)
    checks["R1_quality"] = d_mrr >= -EPS_MRR and d_hit3 >= -EPS_HIT3
    base_u = all_summary["uncredited_graph_hits_per_event"] or 0.0
    cred_u = credited_summary["uncredited_graph_hits_per_event"] or 0.0
    rel_drop = (base_u - cred_u) / base_u if base_u > 0 else 0.0
    checks["R2_uncredited_graph_drop>=5%"] = rel_drop >= MIN_UNCREDITED_GRAPH_DROP
    p50_a = all_summary.get("latency_p50_ms") or 0.0
    p50_c = credited_summary.get("latency_p50_ms") or 0.0
    checks["R3_latency"] = p50_a <= 0 or p50_c <= MAX_LATENCY_RATIO * p50_a
    if not checks["R0_labelled>=100"]:
        verdict = "insufficient"
        recommendation = "all"
    elif all(checks.values()):
        verdict = "pass"
        recommendation = "credited"
    else:
        verdict = "fail"
        recommendation = "all"
    return {
        "checks": checks,
        "delta_mrr": d_mrr,
        "delta_hit@3": d_hit3,
        "uncredited_graph_relative_drop": rel_drop,
        "latency_ratio": (p50_c / p50_a) if p50_a else None,
        "verdict": verdict,
        "recommendation": recommendation,
    }


def overall_recommendation(per_host: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values = {host: data["decision"]["recommendation"] for host, data in per_host.items()}
    distinct = set(values.values())
    if distinct == {"credited"}:
        overall = "credited"
    elif distinct == {"all"} or not distinct:
        overall = "all"
    else:
        overall = "split"
    return {
        "per_host": values,
        "overall": overall,
        "code_default": "all" if overall != "credited" else "credited",
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def run_host(
    host: str,
    snapshot: str | Path,
    cutoff: str,
    *,
    workdir: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    own_dir = workdir is None
    directory = Path(tempfile.mkdtemp(prefix=f"lm-link-policy-{host}-")) if own_dir else Path(workdir)
    try:
        arms = build_arms(snapshot, directory, cutoff)
        reader = open_readonly(snapshot)
        try:
            events, traffic = load_eval_events(reader, cutoff, limit=limit)
        finally:
            reader.close()
        outcomes = replay_arms(arms["paths"], events)
        summaries = {arm: summarize(outcomes[arm]) for arm in outcomes}
        manifest = load_manifest(snapshot)
        return {
            "host": host,
            "cutoff": _iso(_instant(cutoff)),
            "snapshot": {
                "path": str(snapshot),
                "sha256": manifest.get("snapshot_sha256"),
                "captured_at": manifest.get("captured_at"),
                "row_counts": manifest.get("row_counts"),
                "recall_events_created_at": manifest.get("recall_events_created_at"),
            },
            "edge_plan": arms["plan"],
            "truncated_at_cutoff": arms["truncated_at_cutoff"],
            "edge_counts": arms["edge_counts"],
            "traffic": traffic,
            "arms": summaries,
            "bootstrap": paired_bootstrap(outcomes["all"], outcomes["credited"]),
            "decision": decide(summaries["all"], summaries["credited"]),
        }
    finally:
        if own_dir:
            shutil.rmtree(directory, ignore_errors=True)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(results: Sequence[Mapping[str, Any]], overall: Mapping[str, Any]) -> str:
    prereg = PREREG_PATH.read_text(encoding="utf-8") if PREREG_PATH.exists() else ""
    lines = [
        "# LM_IMPLICIT_LINK_POLICY: snapshot replay report",
        "",
        "## Preregistration (verbatim, committed before any result)",
        "",
    ]
    lines += ["> " + line if line else ">" for line in prereg.rstrip().splitlines()]
    lines += ["", "## Results per host (measured separately, never merged)", ""]
    header = "| metric | " + " | ".join(
        f"{r['host']} all | {r['host']} credited" for r in results
    ) + " |"
    lines += [header, "|---" * (1 + 2 * len(results)) + "|"]
    metric_keys = [
        "events",
        "labelled_events",
        "mrr",
        "hit@1",
        "hit@3",
        "credited_node_recall",
        "graph_hits_credited",
        "graph_hits_uncredited",
        "graph_only_uncredited",
        "uncredited_graph_hits_per_event",
        "graph_share_of_credited_hits",
        "graph_share_of_uncredited_hits",
        "latency_p50_ms",
        "latency_p95_ms",
    ]
    for key in metric_keys:
        cells = []
        for r in results:
            cells += [_fmt(r["arms"]["all"].get(key)), _fmt(r["arms"]["credited"].get(key))]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    for key in ("implicit", "related", "connections"):
        cells = []
        for r in results:
            cells += [_fmt(r["edge_counts"]["all"][key]), _fmt(r["edge_counts"]["credited"][key])]
        lines.append(f"| edges: {key} | " + " | ".join(cells) + " |")
    lines += ["", "### Counterfactual edge plan", ""]
    plan_keys = list(results[0]["edge_plan"].keys()) if results else []
    lines += ["| field | " + " | ".join(r["host"] for r in results) + " |"]
    lines += ["|---" * (1 + len(results)) + "|"]
    for key in plan_keys:
        lines.append(
            f"| {key} | " + " | ".join(_fmt(r["edge_plan"].get(key)) for r in results) + " |"
        )
    for key in ("post_cutoff", "excluded_ab", "empty_results", "kept", "labelled"):
        lines.append(
            f"| traffic: {key} | " + " | ".join(_fmt(r["traffic"].get(key)) for r in results) + " |"
        )
    lines += ["", "### Paired bootstrap (credited − all, 95% CI; reported, not in the rule)", ""]
    lines += ["| delta | " + " | ".join(r["host"] for r in results) + " |"]
    lines += ["|---" * (1 + len(results)) + "|"]
    for key in ("mrr", "hit@3", "uncredited_graph_hits_per_event"):
        cells = []
        for r in results:
            b = r["bootstrap"][key]
            cells.append(f"{_fmt(b['delta'])} [{_fmt(b['lo'])}, {_fmt(b['hi'])}]")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += ["", "### Decision rule applied", ""]
    lines += ["| check | " + " | ".join(r["host"] for r in results) + " |"]
    lines += ["|---" * (1 + len(results)) + "|"]
    check_keys = list(results[0]["decision"]["checks"].keys()) if results else []
    for key in check_keys:
        lines.append(
            f"| {key} | " + " | ".join(_fmt(r["decision"]["checks"][key]) for r in results) + " |"
        )
    for key in ("delta_mrr", "delta_hit@3", "uncredited_graph_relative_drop", "latency_ratio"):
        lines.append(
            f"| {key} | " + " | ".join(_fmt(r["decision"].get(key)) for r in results) + " |"
        )
    lines.append(
        "| verdict → recommendation | "
        + " | ".join(
            f"{r['decision']['verdict']} → `{r['decision']['recommendation']}`" for r in results
        )
        + " |"
    )
    lines += [
        "",
        "## Recommendation",
        "",
        f"- Per host: "
        + ", ".join(f"{h}: `LM_IMPLICIT_LINK_POLICY={v}`" for h, v in overall["per_host"].items()),
        f"- Overall: **{overall['overall']}**; suggested valve default for the owner of "
        f"`LM_IMPLICIT_LINK_POLICY` (explicit-marks-core / operator): `{overall['code_default']}`. "
        "No code default is changed by this replay.",
        "",
    ]
    notes = PREREG_PATH.with_name("notes.md")
    if notes.exists():
        lines += [notes.read_text(encoding="utf-8").rstrip(), ""]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="freeze a live DB (read-only source)")
    snap.add_argument("--source-db", required=True)
    snap.add_argument("--out", required=True)
    run = sub.add_parser("run", help="build both arms and replay one host")
    run.add_argument("--host", required=True)
    run.add_argument("--snapshot", required=True)
    run.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    run.add_argument("--workdir", default=None)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--out", required=True)
    render = sub.add_parser("render", help="combine per-host results into the report")
    render.add_argument("--result", action="append", required=True)
    render.add_argument("--json", required=True)
    render.add_argument("--markdown", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "snapshot":
        manifest = create_snapshot(args.source_db, args.out)
        print(json.dumps(manifest["row_counts"]), file=sys.stderr)
        return 0
    if args.command == "run":
        result = run_host(
            args.host, args.snapshot, args.cutoff, workdir=args.workdir, limit=args.limit
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        d = result["decision"]
        print(
            f"{args.host}: {result['traffic']['kept']} events "
            f"({result['traffic']['labelled']} labelled) dMRR {d['delta_mrr']:+.4f} "
            f"dhit@3 {d['delta_hit@3']:+.4f} uncredited-graph drop "
            f"{d['uncredited_graph_relative_drop']:.3f} -> {d['recommendation']}",
            file=sys.stderr,
        )
        return 0
    results = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.result]
    per_host = {r["host"]: r for r in results}
    overall = overall_recommendation(per_host)
    Path(args.json).write_text(
        json.dumps({"hosts": per_host, "recommendation": overall}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(args.markdown).write_text(render_markdown(results, overall), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
