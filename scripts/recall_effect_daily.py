#!/usr/bin/env python3
"""Daily recall-effect metric on live traffic, with A/B-session exclusion.

For every UTC day in ``[--since, --until]`` this reports, over recall events of
live (combat) traffic:

* ``rank1_used`` — share of recall events whose rank-1 result was used;
* ``top3_used`` — share whose top-3 contained a used result;
* ``useful_node_share`` — used delivered nodes / all delivered nodes.

"Used" means a ``recall_credit_ledger`` row for the (event, node) pair with a
basis in ``--used-bases`` (default ``grounded,lookup``). The same numbers are
also given per basis, for the ``explicit`` basis when it exists (a ledger row
with basis ``explicit`` or a side table ``*explicit*credit*`` keyed by
``recall_event_id, node_id``), and for accepted ``used`` marks in
``recall_feedback_marks`` when that table exists — so the same script produces
the "after" report once explicit marks are deployed. Every block is split into
closed events (``feedback_applied = 1``) and never-closed ones.

A/B exclusion works per *transport session*: if any evidence ties a session to
a harness/fixture run, every recall event of that session is excluded. The
evidence, in order of strength:

``receipt``
    The session id appears as ``mcp-session-id`` in a capture header written
    by the AE tree-context harness (``**/capture-memory-*/*.headers.json``
    under ``--receipts-root``) or in a ``--receipt-ids`` file. This is the
    harness's own record of the sessions it launched against the live server.
``fixture_provenance``
    A node written in the same session carries fixture provenance
    (``context.fixture``, ``context.project == "target"``, ``context.run``
    naming a harness run such as ``*--live--r3*`` / ``live-a`` /
    ``tree-context-ab-*``, or content starting with / containing
    ``AE fixture``), or an event's ambient context does.
``fixture_task``
    An event's task (column or ambient context) or a node written in the
    session has a task / node_path whose first segment is a fixture goal path
    listed in the harness ``corpus_manifest.json`` files under
    ``--receipts-root`` (non-``audit`` splits; ``--fixture-family`` adds more).
    This catches harness runs older than the capture proxy.
``synthetic_scope``
    An event of the session used a synthetic project scope ``project:_...``
    (AE end-to-end tests such as ``project:_chat_inject_*``).
``fixture_query_family``
    Weakest layer, for never-closed harness sessions that left no node and
    predate the capture proxy: EVERY query of the session names a manifest
    fixture goal family as a whole token (e.g. ``ev-calibration``).
``fixture_scope``
    An event of the session was recalled in ``project:target`` (the AE fixture
    project) within 30 minutes of traffic already proven above. Live
    supervisors use ``project:target`` too, so the scope alone is not enough.

The operator's legacy filter (scope ``project:target|repo|x`` or a query
matching ledger/billing/tree-context/fixture/kit acceptance) is evaluated only
for comparison: the report lists what it lets through (``legacy_leak``) and
what it drops needlessly (``legacy_overexclude``). It leaked on 2026-09-23,
when tree-context-ab runs recalled in scope ``global`` with fixture-worded
queries such as "development-packaging routing accepted kits route only".

The database is opened only as ``file:<path>?mode=ro``. ``--snapshot-to`` first
copies it with ``living_memory.retrieval_harness.backup_database`` and reads the
copy (also read-only).
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

DEFAULT_RECEIPTS_ROOT = Path("/home/sfx/p/ae/artifacts")
DEFAULT_USED_BASES = ("grounded", "lookup")
TOP_K = 3

LEGACY_SCOPES = frozenset({"project:target", "project:repo", "project:x"})
LEGACY_QUERY_RE = re.compile(r"ledger|billing|tree-context|fixture|kit acceptance", re.I)

FIXTURE_SCOPES = frozenset({"project:target"})
FIXTURE_RUN_RE = re.compile(
    r"--live--r\d|^live-[a-z](?:\b|-|$)|tree-context-(?:ab|baseline|redesign)|^r\d+--live-[a-z]",
    re.I,
)
FIXTURE_CONTENT_RE = re.compile(r"\bAE fixture\b")
FIXTURE_PROJECTS = frozenset({"target"})
CAPTURE_DIR_PREFIX = "capture-memory-"
FIXTURE_SCOPE_WINDOW_S = 30 * 60
CORPUS_MANIFEST = "corpus_manifest.json"
NON_FIXTURE_SPLITS = frozenset({"audit"})
SYNTHETIC_SCOPE_RE = re.compile(r"^project:_")


# --------------------------------------------------------------------------- IO


def open_ro(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def snapshot(db: Path, target: Path) -> Path:
    src_dir = Path(__file__).resolve().parent.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from living_memory.retrieval_harness import backup_database

    backup_database(db, target)
    return target


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _json(text: Any, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------- receipts


def scan_receipts(roots: Iterable[Path]) -> tuple[dict[str, str], set[str]]:
    """Harness receipts under ``roots``.

    Returns (mcp-session-id -> capture dir that recorded it, fixture task
    families). Families are the first path segment of ``goal_path`` of the
    fixture cases (splits other than ``audit``, which replays real trees) in
    the harness ``corpus_manifest.json`` files.
    """

    found: dict[str, str] = {}
    families: set[str] = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if CORPUS_MANIFEST in filenames:
                families |= _manifest_families(Path(dirpath) / CORPUS_MANIFEST)
            if not os.path.basename(dirpath).startswith(CAPTURE_DIR_PREFIX):
                continue
            dirnames[:] = []
            for name in filenames:
                if not name.endswith(".headers.json"):
                    continue
                try:
                    with open(os.path.join(dirpath, name), encoding="utf-8") as fh:
                        headers = json.load(fh)
                except (OSError, ValueError):
                    continue
                for side in ("response", "request"):
                    part = headers.get(side) if isinstance(headers, dict) else None
                    if not isinstance(part, dict):
                        continue
                    for key, value in part.items():
                        if key.lower() == "mcp-session-id" and value:
                            found.setdefault(str(value), os.path.relpath(dirpath, root))
    return found, families


def _manifest_families(path: Path) -> set[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    out = set()
    for case in data.get("cases", []) if isinstance(data, dict) else []:
        if not isinstance(case, dict) or case.get("split") in NON_FIXTURE_SPLITS:
            continue
        fam = _task_family({"task": case.get("goal_path")})
        if fam:
            out.add(fam)
    return out


def load_receipt_ids(paths: Iterable[Path]) -> dict[str, str]:
    found: dict[str, str] = {}
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        data = _json(text, None)
        ids = data if isinstance(data, list) else text.split()
        for sid in ids:
            if sid:
                found.setdefault(str(sid).strip(), f"receipt-ids:{Path(path).name}")
    return found


# ------------------------------------------------------------------------ load


def _day(ts: str) -> str:
    return (ts or "")[:10]


def _shift(day: str, days: int) -> str:
    return (dt.date.fromisoformat(day) + dt.timedelta(days=days)).isoformat()


def load_events(conn: sqlite3.Connection, since: str, until: str) -> list[dict[str, Any]]:
    cols = _columns(conn, "recall_events")
    pick = [
        c
        for c in (
            "id", "query", "scope", "requested_scope", "agent", "task", "ambient_context",
            "transport_session_id", "feedback_applied", "feedback_trace_id", "results",
            "created_at",
        )
        if c in cols
    ]
    rows = conn.execute(
        f"SELECT {', '.join(pick)} FROM recall_events WHERE created_at >= ? AND created_at < ?"
        " ORDER BY created_at",
        (since, _shift(until, 1)),
    ).fetchall()
    events = []
    for row in rows:
        ev = {c: row[c] for c in pick}
        results = _json(ev.get("results"), [])
        ordered = sorted(
            (r for r in results if isinstance(r, dict) and r.get("node_id")),
            key=lambda r: (r.get("rank") if isinstance(r.get("rank"), int) else 10**6),
        )
        seen: list[str] = []
        for r in ordered:
            if r["node_id"] not in seen:
                seen.append(r["node_id"])
        ev["delivered"] = seen
        ev["ambient"] = _json(ev.get("ambient_context"), {})
        ev["session"] = ev.get("transport_session_id") or f"event:{ev['id']}"
        ev["day"] = _day(ev["created_at"])
        ev["closed"] = bool(ev.get("feedback_applied"))
        events.append(ev)
    return events


def load_credit(conn: sqlite3.Connection, event_ids: set[str]) -> dict[str, dict[tuple[str, str], str]]:
    """basis -> {(event, node): source}. Includes explicit side tables."""

    credit: dict[str, dict[tuple[str, str], str]] = defaultdict(dict)
    if _table_exists(conn, "recall_credit_ledger"):
        for row in conn.execute("SELECT recall_event_id, node_id, basis FROM recall_credit_ledger"):
            if row[0] in event_ids:
                credit[str(row[2])][(row[0], row[1])] = "recall_credit_ledger"
    side_tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%explicit%credit%'"
        )
    ]
    for table in side_tables:
        cols = _columns(conn, table)
        if not {"recall_event_id", "node_id"} <= cols:
            continue
        for row in conn.execute(f"SELECT recall_event_id, node_id FROM {table}"):
            if row[0] in event_ids:
                credit["explicit"].setdefault((row[0], row[1]), table)
    return credit


def load_marks(conn: sqlite3.Connection, event_ids: set[str]) -> list[dict[str, Any]] | None:
    if not _table_exists(conn, "recall_feedback_marks"):
        return None
    cols = _columns(conn, "recall_feedback_marks")
    pick = [
        c
        for c in (
            "recall_event_id", "node_id", "mark", "accepted", "rank", "via_tool",
            "transport_session_id", "agent", "marked_at",
        )
        if c in cols
    ]
    marks = []
    for row in conn.execute(f"SELECT {', '.join(pick)} FROM recall_feedback_marks"):
        m = {c: row[c] for c in pick}
        if m.get("recall_event_id") in event_ids:
            marks.append(m)
    return marks


def load_session_nodes(
    conn: sqlite3.Connection, sessions: set[str], since: str, until: str
) -> dict[str, list[dict[str, Any]]]:
    """Nodes written in the given transport sessions (a day of slack each side)."""

    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not _table_exists(conn, "nodes"):
        return out
    rows = conn.execute(
        "SELECT id, substr(content, 1, 600) AS head, context, created_at FROM nodes"
        " WHERE created_at >= ? AND created_at < ?",
        (_shift(since, -1), _shift(until, 2)),
    )
    for row in rows:
        ctx = _json(row["context"], {})
        if not isinstance(ctx, dict):
            continue
        sid = ctx.get("transport_session_id")
        if sid and sid in sessions:
            out[sid].append({"id": row["id"], "head": row["head"] or "", "context": ctx})
    return out


# ------------------------------------------------------------------ exclusion


def legacy_match(ev: dict[str, Any]) -> bool:
    scopes = {ev.get("requested_scope"), ev.get("scope")}
    return bool(scopes & LEGACY_SCOPES) or bool(LEGACY_QUERY_RE.search(ev.get("query") or ""))


def _node_provenance(node: dict[str, Any]) -> str | None:
    ctx = node["context"]
    if ctx.get("fixture"):
        return "context.fixture"
    if str(ctx.get("project", "")).lower() in FIXTURE_PROJECTS:
        return "context.project=target"
    run = str(ctx.get("run", "") or ctx.get("run_id", ""))
    if run and FIXTURE_RUN_RE.search(run):
        return f"context.run={run[:60]}"
    if FIXTURE_CONTENT_RE.search(node["head"]):
        return "content:AE fixture"
    return None


def _task_family(ctx: dict[str, Any]) -> str | None:
    for key in ("node_path", "task"):
        value = ctx.get(key)
        if isinstance(value, str) and value.strip():
            head = value.strip().split("/")[0].strip()
            if head:
                return head
    return None


def _epoch(ts: str) -> float:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def _session_families(evs: list[dict[str, Any]], nodes: list[dict[str, Any]]) -> set[str]:
    fams = set()
    for ev in evs:
        amb = ev["ambient"] if isinstance(ev["ambient"], dict) else {}
        for ctx in ({"task": ev.get("task")}, amb):
            fam = _task_family(ctx)
            if fam:
                fams.add(fam)
    for node in nodes:
        fam = _task_family(node["context"])
        if fam:
            fams.add(fam)
    return fams


def classify_sessions(
    events: list[dict[str, Any]],
    session_nodes: dict[str, list[dict[str, Any]]],
    receipts: dict[str, str],
    families: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    """session -> {'reasons': {reason: detail}}; empty reasons = kept."""

    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        by_session[ev["session"]].append(ev)

    verdict: dict[str, dict[str, Any]] = {}
    for sid, evs in by_session.items():
        reasons: dict[str, str] = {}
        if sid in receipts:
            reasons["receipt"] = receipts[sid]
        for node in session_nodes.get(sid, []):
            why = _node_provenance(node)
            if why:
                reasons.setdefault("fixture_provenance", f"{node['id']} {why}")
                break
        if "fixture_provenance" not in reasons:
            for ev in evs:
                amb = ev["ambient"] if isinstance(ev["ambient"], dict) else {}
                if amb.get("fixture") or str(amb.get("project", "")).lower() in FIXTURE_PROJECTS:
                    reasons["fixture_provenance"] = f"{ev['id']} ambient_context"
                    break
        verdict[sid] = {"reasons": reasons}

    families = set(families or ())
    for sid, v in verdict.items():
        if v["reasons"]:
            continue
        hits = sorted(_session_families(by_session[sid], session_nodes.get(sid, [])) & families)
        if hits:
            v["reasons"]["fixture_task"] = f"family={hits[0]}"

    # Synthetic project scopes (``project:_chat_inject_...``) are created by
    # AE end-to-end tests, never by live work.
    for sid, v in verdict.items():
        if v["reasons"]:
            continue
        for ev in by_session[sid]:
            scope = str(ev.get("requested_scope") or "")
            if SYNTHETIC_SCOPE_RE.match(scope):
                v["reasons"]["synthetic_scope"] = f"{ev['id']} {scope}"
                break

    # Never-closed harness sessions leave no node and may predate the capture
    # proxy: every query of the session names a manifest fixture goal family.
    if families:
        fam_re = re.compile(
            r"(?<![\w-])(?:" + "|".join(re.escape(f) for f in sorted(families, key=len, reverse=True)) + r")(?![\w-])",
            re.I,
        )
        for sid, v in verdict.items():
            if v["reasons"]:
                continue
            evs = by_session[sid]
            if evs and all(fam_re.search(ev.get("query") or "") for ev in evs):
                v["reasons"]["fixture_query_family"] = (
                    f"{len(evs)} queries name {fam_re.search(evs[0].get('query') or '').group(0)}"
                )

    # project:target is the AE fixture project, but live supervisors use it
    # too, so it only counts next to proven harness activity.
    proven = sorted(
        _epoch(ev["created_at"])
        for sid, v in verdict.items()
        if v["reasons"]
        for ev in by_session[sid]
    )
    for sid, v in verdict.items():
        if v["reasons"]:
            continue
        for ev in by_session[sid]:
            if not ({ev.get("requested_scope"), ev.get("scope")} & FIXTURE_SCOPES):
                continue
            t = _epoch(ev["created_at"])
            i = bisect.bisect_left(proven, t - FIXTURE_SCOPE_WINDOW_S)
            if i < len(proven) and proven[i] <= t + FIXTURE_SCOPE_WINDOW_S:
                v["reasons"]["fixture_scope"] = (
                    f"{ev['id']} {ev.get('requested_scope')} within "
                    f"{FIXTURE_SCOPE_WINDOW_S // 60} min of proven harness traffic"
                )
                break
    return verdict


# --------------------------------------------------------------------- metrics


def _share(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _block(events: list[dict[str, Any]], used: set[tuple[str, str]]) -> dict[str, Any]:
    n = r1 = t3 = delivered = used_nodes = 0
    for ev in events:
        nodes = ev["delivered"]
        if not nodes:
            continue
        n += 1
        flags = [(ev["id"], node) in used for node in nodes]
        r1 += int(flags[0])
        t3 += int(any(flags[:TOP_K]))
        delivered += len(nodes)
        used_nodes += sum(flags)
    return {
        "events_with_results": n,
        "rank1_used_events": r1,
        "top3_used_events": t3,
        "delivered_nodes": delivered,
        "used_delivered_nodes": used_nodes,
        "rank1_used": _share(r1, n),
        "top3_used": _share(t3, n),
        "useful_node_share": _share(used_nodes, delivered),
    }


def metrics(
    events: list[dict[str, Any]],
    credit: dict[str, dict[tuple[str, str], str]],
    used_bases: tuple[str, ...],
    marked_used: set[tuple[str, str]] | None,
) -> dict[str, Any]:
    used = set().union(*(credit.get(b, {}).keys() for b in used_bases)) if used_bases else set()
    bases = sorted(set(credit) | set(used_bases) | {"explicit"})
    out: dict[str, Any] = {
        "events": len(events),
        "closed_events": sum(ev["closed"] for ev in events),
        "never_closed_events": sum(not ev["closed"] for ev in events),
        "used": _block(events, used),
        "per_basis": {b: _block(events, set(credit.get(b, {}).keys())) for b in bases},
    }
    explicit = set(credit.get("explicit", {}).keys())
    out["used_with_explicit"] = _block(events, used | explicit)
    if marked_used is not None:
        out["marked_used"] = _block(events, marked_used)
        out["used_or_marked"] = _block(events, used | explicit | marked_used)
    return out


def _mark_stats(marks: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    ids = {ev["id"] for ev in events}
    mine = [m for m in marks if m.get("recall_event_id") in ids]
    counts = Counter(
        f"{m.get('mark')}:{'accepted' if m.get('accepted') else 'rejected'}" for m in mine
    )
    marked_events = {m["recall_event_id"] for m in mine if m.get("accepted")}
    return {
        "marks": dict(sorted(counts.items())),
        "events_with_accepted_mark": len(marked_events),
        "events_with_accepted_mark_share": _share(len(marked_events), len(ids)),
        "via_tool": dict(Counter(str(m.get("via_tool")) for m in mine if m.get("accepted"))),
    }


# ---------------------------------------------------------------------- report


def build_report(
    db: str,
    host_label: str,
    since: str,
    until: str,
    receipts: dict[str, str],
    families: Iterable[str] = (),
    used_bases: tuple[str, ...] = DEFAULT_USED_BASES,
    leak_day: str = "2026-09-23",
    opened_as: str | None = None,
) -> dict[str, Any]:
    conn = open_ro(db)
    try:
        events = load_events(conn, since, until)
        event_ids = {ev["id"] for ev in events}
        sessions = {ev["session"] for ev in events}
        session_nodes = load_session_nodes(conn, sessions, since, until)
        credit = load_credit(conn, event_ids)
        marks = load_marks(conn, event_ids)
        ledger_bases = (
            sorted(r[0] for r in conn.execute("SELECT DISTINCT basis FROM recall_credit_ledger"))
            if _table_exists(conn, "recall_credit_ledger")
            else []
        )
    finally:
        conn.close()

    families = sorted(families or ())
    verdict = classify_sessions(events, session_nodes, receipts, families)
    for ev in events:
        ev["excluded_reasons"] = sorted(verdict[ev["session"]]["reasons"])
        ev["legacy"] = legacy_match(ev)

    marked_used = (
        {(m["recall_event_id"], m["node_id"]) for m in marks if m.get("mark") == "used" and m.get("accepted")}
        if marks is not None
        else None
    )

    days = []
    exclusion_days = []
    day = since
    while day <= until:
        day_events = [ev for ev in events if ev["day"] == day]
        kept = [ev for ev in day_events if not ev["excluded_reasons"]]
        legacy_kept = [ev for ev in day_events if not ev["legacy"]]
        closed = [ev for ev in kept if ev["closed"]]
        never = [ev for ev in kept if not ev["closed"]]
        entry = {
            "day": day,
            "all_events": len(day_events),
            "kept_events": len(kept),
            "excluded_events": len(day_events) - len(kept),
            "metrics": metrics(kept, credit, used_bases, marked_used),
            "closed": metrics(closed, credit, used_bases, marked_used),
            "never_closed": metrics(never, credit, used_bases, marked_used),
            "legacy_filter": {
                "kept_events": len(legacy_kept),
                "used": _block(legacy_kept, set().union(*(credit.get(b, {}).keys() for b in used_bases))),
            },
        }
        if marks is not None:
            entry["marks"] = _mark_stats(marks, kept)
        days.append(entry)

        excl = [ev for ev in day_events if ev["excluded_reasons"]]
        by_reason = Counter(r for ev in excl for r in ev["excluded_reasons"])
        primary = Counter(ev["excluded_reasons"][0] if ev["excluded_reasons"] else "" for ev in excl)
        leak = [ev for ev in excl if not ev["legacy"]]
        over = [ev for ev in day_events if ev["legacy"] and not ev["excluded_reasons"]]
        sess_rows = []
        for sid in sorted({ev["session"] for ev in excl}):
            sevs = [ev for ev in excl if ev["session"] == sid]
            sess_rows.append(
                {
                    "session": sid,
                    "events": len(sevs),
                    "reasons": verdict[sid]["reasons"],
                    "legacy_caught_events": sum(ev["legacy"] for ev in sevs),
                    "sample_query": (sevs[0].get("query") or "")[:120],
                    "scopes": sorted({str(ev.get("requested_scope")) for ev in sevs}),
                }
            )
        exclusion_days.append(
            {
                "day": day,
                "excluded_events": len(excl),
                "excluded_sessions": len(sess_rows),
                "events_by_reason": dict(sorted(by_reason.items())),
                "events_by_strongest_reason": dict(sorted(primary.items())),
                "legacy_would_exclude": sum(ev["legacy"] for ev in day_events),
                "legacy_leak_events": len(leak),
                "legacy_overexclude_events": len(over),
                "legacy_overexclude_sample": [
                    {"session": ev["session"], "scope": ev.get("requested_scope"), "query": (ev.get("query") or "")[:100]}
                    for ev in over[:5]
                ],
                "sessions": sess_rows,
            }
        )
        day = _shift(day, 1)

    kept_all = [ev for ev in events if not ev["excluded_reasons"]]
    leak_events = [ev for ev in events if ev["day"] == leak_day and ev["excluded_reasons"] and not ev["legacy"]]
    leak_sessions = sorted({ev["session"] for ev in leak_events})
    return {
        "host_label": host_label,
        "db": str(db),
        "opened_as": opened_as or f"file:{db}?mode=ro",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window": {"since": since, "until": until, "days_utc": "created_at[:10], inclusive"},
        "definitions": {
            "used": f"recall_credit_ledger row for (event, node) with basis in {list(used_bases)}",
            "rank1_used": "events whose rank-1 delivered node is used / events with >=1 delivered node",
            "top3_used": "events with a used node among ranks 1..3 / events with >=1 delivered node",
            "useful_node_share": "used delivered (event,node) pairs / delivered pairs",
            "closed": "recall_events.feedback_applied = 1 (closed by a later remember/teach)",
            "explicit": "ledger basis 'explicit' or a *explicit*credit* side table; marked_used = accepted "
            "'used' rows of recall_feedback_marks",
            "exclusion": "per transport session: receipt > fixture_provenance > fixture_task > synthetic_scope > fixture_query_family > fixture_scope (near proven traffic)",
        },
        "sources": {
            "receipt_ids": len(receipts),
            "receipt_ids_matched_sessions": len({ev["session"] for ev in events if ev["session"] in receipts}),
            "ledger_bases": ledger_bases,
            "explicit_present": bool(credit.get("explicit")),
            "recall_feedback_marks_present": marks is not None,
            "fixture_task_families": families,
        },
        "totals": {
            "all_events": len(events),
            "kept_events": len(kept_all),
            "excluded_events": len(events) - len(kept_all),
            "metrics": metrics(kept_all, credit, used_bases, marked_used),
        },
        "leak_check": {
            "day": leak_day,
            "events_excluded_new_but_not_legacy": len(leak_events),
            "sessions": len(leak_sessions),
            "by_reason": dict(Counter(r for ev in leak_events for r in ev["excluded_reasons"])),
            "sample": [
                {"session": ev["session"], "scope": ev.get("requested_scope"), "query": (ev.get("query") or "")[:100]}
                for ev in leak_events[:8]
            ],
        },
        "days": days,
        "exclusion": exclusion_days,
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def render_md(report: dict[str, Any], session_limit: int = 15) -> str:
    lines: list[str] = []
    host = report["host_label"]
    w = report["window"]
    src = report["sources"]
    lines += [
        f"# Recall effect by day — {host} ({w['since']}..{w['until']})",
        "",
        f"- DB: `{report['db']}` opened as `{report['opened_as']}`; generated {report['generated_at']}.",
        f"- Used = {report['definitions']['used']}. Shares are over events with ≥1 delivered node.",
        f"- Ledger bases present: {', '.join(src['ledger_bases']) or 'none'}; explicit credit present: "
        f"{src['explicit_present']}; recall_feedback_marks present: {src['recall_feedback_marks_present']}.",
        f"- A/B exclusion per transport session. Receipt ids scanned: {src['receipt_ids']} "
        f"(matched sessions in window: {src['receipt_ids_matched_sessions']}). Fixture task families "
        f"from harness corpus manifests: {', '.join(src['fixture_task_families']) or 'none'}.",
        "",
    ]
    t = report["totals"]
    m = t["metrics"]["used"]
    lines += [
        "## Window total (kept traffic)",
        "",
        f"Events {t['all_events']}, kept {t['kept_events']}, excluded {t['excluded_events']}. "
        f"rank-1 used {_pct(m['rank1_used'])}, top-3 used {_pct(m['top3_used'])}, "
        f"useful nodes {_pct(m['useful_node_share'])} ({m['used_delivered_nodes']}/{m['delivered_nodes']}).",
        "",
        "## Daily metric (kept traffic)",
        "",
        "| day | events | excl | kept | closed | never | r1 used | top3 used | useful nodes | grounded r1/top3/nodes | lookup r1/top3/nodes | explicit r1/top3/nodes | legacy-filter r1 / useful |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for d in report["days"]:
        mm = d["metrics"]
        u = mm["used"]
        pb = mm["per_basis"]

        def trio(b: str) -> str:
            blk = pb.get(b)
            if not blk:
                return "—"
            return f"{_pct(blk['rank1_used'])} / {_pct(blk['top3_used'])} / {_pct(blk['useful_node_share'])}"

        lines.append(
            f"| {d['day']} | {d['all_events']} | {d['excluded_events']} | {d['kept_events']} | "
            f"{mm['closed_events']} | {mm['never_closed_events']} | {_pct(u['rank1_used'])} | "
            f"{_pct(u['top3_used'])} | {_pct(u['useful_node_share'])} | {trio('grounded')} | "
            f"{trio('lookup')} | {trio('explicit')} | {_pct(d['legacy_filter']['used']['rank1_used'])} / "
            f"{_pct(d['legacy_filter']['used']['useful_node_share'])} |"
        )
    lines += [
        "",
        "## Closed vs never-closed (kept traffic)",
        "",
        "| day | closed n | closed r1 | closed top3 | closed useful | never n | never r1 | never top3 | never useful |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in report["days"]:
        c = d["closed"]["used"]
        n = d["never_closed"]["used"]
        lines.append(
            f"| {d['day']} | {c['events_with_results']} | {_pct(c['rank1_used'])} | {_pct(c['top3_used'])} | "
            f"{_pct(c['useful_node_share'])} | {n['events_with_results']} | {_pct(n['rank1_used'])} | "
            f"{_pct(n['top3_used'])} | {_pct(n['useful_node_share'])} |"
        )
    if report["sources"]["recall_feedback_marks_present"]:
        lines += [
            "",
            "## Explicit marks (kept traffic)",
            "",
            "| day | events with accepted mark | marked-used r1 / top3 / nodes | used∪explicit∪marked useful | marks |",
            "|---|---:|---|---:|---|",
        ]
        for d in report["days"]:
            mk = d.get("marks", {})
            mu = d["metrics"].get("marked_used", {})
            uo = d["metrics"].get("used_or_marked", {})
            lines.append(
                f"| {d['day']} | {mk.get('events_with_accepted_mark', 0)} ({_pct(mk.get('events_with_accepted_mark_share'))}) | "
                f"{_pct(mu.get('rank1_used'))} / {_pct(mu.get('top3_used'))} / {_pct(mu.get('useful_node_share'))} | "
                f"{_pct(uo.get('useful_node_share'))} | {json.dumps(mk.get('marks', {}), ensure_ascii=False)} |"
            )
    lk = report["leak_check"]
    lines += [
        "",
        f"## Leak check {lk['day']}",
        "",
        f"Events on {lk['day']} excluded by the new filter that the legacy filter "
        f"(scope project:target|repo|x + ledger/billing/tree-context/fixture/kit acceptance) let through: "
        f"**{lk['events_excluded_new_but_not_legacy']}** in {lk['sessions']} sessions; by reason "
        f"{json.dumps(lk['by_reason'])}.",
        "",
    ]
    leak_day = next((d for d in report["days"] if d["day"] == lk["day"]), None)
    if leak_day:
        new_u = leak_day["metrics"]["used"]
        old_u = leak_day["legacy_filter"]["used"]
        lines += [
            "| filter | kept events | r1 used | top3 used | useful nodes |",
            "|---|---:|---:|---:|---:|",
            f"| legacy (scope/keywords) | {leak_day['legacy_filter']['kept_events']} | {_pct(old_u['rank1_used'])} | "
            f"{_pct(old_u['top3_used'])} | {_pct(old_u['useful_node_share'])} |",
            f"| this script (session evidence) | {leak_day['kept_events']} | {_pct(new_u['rank1_used'])} | "
            f"{_pct(new_u['top3_used'])} | {_pct(new_u['useful_node_share'])} |",
            "",
        ]
    for s in lk["sample"]:
        lines.append(f"- `{s['session'][:12]}` {s['scope']}: {s['query']}")
    lines += [
        "",
        "## Exclusions per day",
        "",
        "| day | excluded events | sessions | by reason (events; a session may carry several) | legacy would drop | legacy leak | legacy over-exclude |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for x in report["exclusion"]:
        lines.append(
            f"| {x['day']} | {x['excluded_events']} | {x['excluded_sessions']} | "
            f"{json.dumps(x['events_by_reason'])} | {x['legacy_would_exclude']} | "
            f"{x['legacy_leak_events']} | {x['legacy_overexclude_events']} |"
        )
    lines += [
        "",
        f"### Excluded sessions (up to {session_limit} per day, leaked-by-legacy first; full list in the JSON)",
        "",
    ]
    for x in report["exclusion"]:
        if not x["sessions"]:
            continue
        ranked = sorted(x["sessions"], key=lambda s: (s["legacy_caught_events"] >= s["events"], -s["events"]))
        lines.append(f"**{x['day']}** — {x['excluded_sessions']} sessions")
        lines.append("")
        for s in ranked[:session_limit]:
            why = "; ".join(f"{k}: {v}" for k, v in sorted(s["reasons"].items()))
            lines.append(
                f"- `{s['session'][:12]}` {s['events']} ev (legacy caught {s['legacy_caught_events']}), "
                f"{','.join(s['scopes'])} — {why} — “{s['sample_query'][:80]}”"
            )
        if len(ranked) > session_limit:
            lines.append(f"- … {len(ranked) - session_limit} more")
        lines.append("")
    lines += [
        "## Legacy over-exclusion sample (kept by the new filter)",
        "",
    ]
    for x in report["exclusion"]:
        for s in x["legacy_overexclude_sample"][:2]:
            lines.append(f"- {x['day']} `{s['session'][:12]}` {s['scope']}: {s['query']}")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--host-label", required=True)
    ap.add_argument("--since", required=True, help="first UTC day, YYYY-MM-DD")
    ap.add_argument("--until", required=True, help="last UTC day (inclusive), YYYY-MM-DD")
    ap.add_argument("--json", type=Path, dest="json_out")
    ap.add_argument("--md", type=Path, dest="md_out")
    ap.add_argument(
        "--receipts-root", type=Path, action="append", default=None,
        help=f"scan capture-memory-*/ headers for mcp-session-id (default {DEFAULT_RECEIPTS_ROOT} if present)",
    )
    ap.add_argument("--no-default-receipts", action="store_true")
    ap.add_argument("--receipt-ids", type=Path, action="append", default=[],
                    help="file with session ids (JSON list or whitespace separated)")
    ap.add_argument("--fixture-family", action="append", default=[],
                    help="extra fixture task family (first task path segment)")
    ap.add_argument("--used-bases", default=",".join(DEFAULT_USED_BASES))
    ap.add_argument("--leak-day", default="2026-09-23")
    ap.add_argument("--snapshot-to", type=Path,
                    help="copy --db with retrieval_harness.backup_database first and read the copy")
    args = ap.parse_args(argv)

    for day in (args.since, args.until):
        dt.date.fromisoformat(day)
    roots = list(args.receipts_root or [])
    if not roots and not args.no_default_receipts and DEFAULT_RECEIPTS_ROOT.is_dir():
        roots = [DEFAULT_RECEIPTS_ROOT]
    receipts, families = scan_receipts(roots)
    families |= {f.strip() for f in args.fixture_family if f.strip()}
    receipts.update({k: v for k, v in load_receipt_ids(args.receipt_ids).items() if k not in receipts})

    db = args.db
    opened_as = None
    if args.snapshot_to:
        snapshot(db, args.snapshot_to)
        opened_as = f"snapshot {args.snapshot_to} (backup_database of {db}), read as file:...?mode=ro"
        db = args.snapshot_to

    report = build_report(
        str(db), args.host_label, args.since, args.until, receipts, families,
        used_bases=tuple(b for b in args.used_bases.split(",") if b),
        leak_day=args.leak_day, opened_as=opened_as,
    )
    report["sources"]["receipt_roots"] = [str(r) for r in roots]
    report["sources"]["receipt_id_files"] = [str(p) for p in args.receipt_ids]
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_md(report), encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
