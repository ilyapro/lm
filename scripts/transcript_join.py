#!/usr/bin/env python3
"""Join sealed postsession-corpus transcripts to recall_events (phase 1).

Host-portable by construction: one file, standard library only, no repository
imports, runs on bare ``python3`` (3.8+). Copy this single script to the alt
host next to its own store snapshot, manifest and index, and it produces the
same artifact shapes as the local field run.

    scripts/transcript_join.py \
        --store /path/to/global.sqlite3 \
        --manifest artifacts/transcript-grounding/join/corpus-fresh.json \
        --index /path/to/corpus-index.jsonl \
        --sealed-manifest artifacts/post-session/corpus.json \
        --drift-report artifacts/transcript-grounding/join/verify-vs-sealed.json \
        --out-dir artifacts/transcript-grounding/join \
        --field-label local

    scripts/transcript_join.py --self-test   # offline fixture suite, no store

The store is opened ONLY read-only, via the sqlite3 ``file:`` URI with
``mode=ro``; nothing in this script can write to it.

Which manifest seals the index
------------------------------
The index must hash to ``index_sha256`` of **whichever manifest ``--manifest``
was handed**, or the run refuses to start: rows that do not match that
manifest are not the rows it describes, and its counts, split digests and
window would then describe something else. A freshly built, self-sealed
manifest+index pair therefore verifies exactly as a sealed one does.

That is the whole gate. It deliberately does *not* require the pair to be the
one sealed under ``artifacts/post-session/``: on a live host most sealed
sessions eventually roll off disk, so the sealed index would name transcripts
that can no longer be read, and a fresh build is the more honest corpus.
``--sealed-manifest`` and ``--drift-report`` are how that choice is put on the
record rather than hidden: the coverage JSON reports the index sha256, the
manifest sha256, whether that index sha256 equals the sealed corpus's
``index_sha256`` (false whenever a fresh corpus is used) and the measured
drift of the fresh corpus against the seal. Neither flag relaxes the gate.

Matching tiers, highest wins (reported under ``coverage.by_method``)
--------------------------------------------------------------------
``event_id_echo``
    One pass over the corpus transcripts extracting verbatim recall-event
    ULIDs into ``ulid -> (session, record_index)``. The server envelope's
    ``recall_event_id`` is echoed verbatim into the session transcript at the
    delivery moment, so a hit yields the transcript AND the delivery record
    position in one step. A hit only counts when the event's ``created_at``
    falls inside the transcript's own time window (±300 s slack): transcripts
    also quote *old* event ids in recalled provenance, and those stale quotes
    must not steal the match. ``delivery_record_index`` is the FIRST sealed
    record containing the id -- an id cannot appear before its event exists,
    so this is at or after the true delivery, and a provenance re-quote can
    only shrink phase 2's remainder, never inflate it. A same-window quote in
    a *different* conversation makes tier 1 ambiguous, and the transport tier
    then resolves the event to its own session.
``transport_session``
    Events sharing ``transport_session_id`` with an echo-matched event
    inherit that session's transcript (delivery position unknown -> null).
``cli_session``
    ``recall_events.session_id`` equals an index row's ``cli_session_id`` or
    is listed in its ``linked_session_ids``.
``time_cwd_window``
    Bounded window: ``created_at`` within ``[started_at-300s, ended_at+300s]``
    AND ``ambient_context.cwd`` equals the index row's ``cwd``. A unique
    candidate is required; ambiguity means unmatched.

At every tier, candidate transcripts are collapsed by ``split_key`` first:
the corpus records the same conversation more than once (an AE node result
and the codex rollout it was printed from), and those copies are one
candidate, not two. Distinct identity groups surviving the collapse mean
ambiguity, and the event falls through to the next tier (or to unmatched).

Sealed prefix
-------------
At most ``index.records`` records are read per transcript, so results stay
stable as live transcripts keep growing past the seal.

Denominator (pinned)
--------------------
All recall_events whose ``created_at`` lies inside the corpus window
``[min started_at, max ended_at]`` over the index rows actually used, bounds
inclusive, compared in UTC. The definition is embedded verbatim in the
coverage JSON so the falsifier cannot be gamed by moving the denominator --
in particular it is never narrowed to the events that happen to carry a
``transport_session_id``, which would hand the tiers their own denominator.

Outputs (under ``--out-dir``)
-----------------------------
``coverage-<field-label>.json``
    field label, store path+sha256+origin, manifest sha256, index sha256,
    ``index.equals_sealed_corpus_index_sha256``, ``sealed_corpus`` and
    ``drift_vs_sealed_manifest``, the denominator definition,
    ``coverage.{total_events,matched_events,share,by_method}``,
    ``fallback_precision`` (the blinded audit below), ``selection_bias``,
    and ``falsifier_60pct`` with threshold 0.6 and a pass/fail verdict.
``match-table-<field-label>.jsonl.gz``
    one row per denominator event -- matched and unmatched alike:
    ``event_id, created_at, matched, method, corpus_session_id,
    transcript_path, transcript_sha256, delivery_record_index`` (0-based
    record position of the echo, null below tier 1). Never query text, never
    node content: phase 2 needs ``delivery_record_index`` to slice the
    post-delivery remainder and drop the LM tool-result echo, and needs
    nothing else from this table.

Blinded precision audit
-----------------------
Tiers 3-4 are re-run on the tier-1/2-matched gold subset through the same
matcher functions the main pass uses (they never see echo state), and
per-tier agreement with the gold identity group is reported. If the gold
subset is under ``GOLD_MIN_EVENTS`` the audit is flagged low-confidence.

Selection-bias measurement (``selection_bias``)
-----------------------------------------------
The coverage share alone cannot tell a neutral join from a join that finds
exactly the events the goal wanted to exclude. On the local field the share
hid precisely that: ``feedback_applied=1`` ("consumed", i.e. a later
``memory_remember`` credited the event) ran at 0.1717 store-wide but 0.6472
among matched events -- 3.8x -- because the dominant reason an event id is
written into a transcript at all is the provenance of the remember that
consumes it. So the run reports the consumed share store-wide, across the
denominator, among matched and unmatched events, per tier, and within each
calendar month separately (the month control rules out a recency confound:
if the enrichment survives inside every month it is not an artefact of
matched events simply being newer). A high coverage produced by that
mechanism does not support the downstream claim, so this measurement is not
optional. Stores without the ``feedback_applied`` column report
``measured: false`` rather than a silent zero.

Exit codes: 0 on success (including a "fail" falsifier verdict -- that is a
measurement, not an error), 1 on self-test failure, 2 on bad inputs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import sqlite3
import sys
import tempfile
import urllib.parse
from datetime import datetime, timezone

GENERATOR = "scripts/transcript_join.py"

#: Slack, in seconds, applied around a transcript's [started_at, ended_at]
#: window -- both for validating tier-1 echoes and for the tier-4 fallback.
WINDOW_SLACK_SECONDS = 300

#: The phase-1 falsifier: coverage share below this stops the goal.
FALSIFIER_THRESHOLD = 0.60

#: Gold subsets smaller than this flag the fallback-precision audit as
#: low-confidence.
GOLD_MIN_EVENTS = 300

#: A calendar month enters the selection-bias within-month control only when
#: both its cells (matched, unmatched) hold at least this many events; thin
#: months would otherwise turn noise into a verdict.
MONTH_MIN_CELL_EVENTS = 30

#: The column that marks an event as consumed: a later ``memory_remember``
#: credited it, which is exactly the reinforcement phase 2 exists to replace.
CONSUMED_COLUMN = "feedback_applied"

METHODS = ("event_id_echo", "transport_session", "cli_session", "time_cwd_window")

#: Crockford base32, the ULID alphabet (no I, L, O, U). Lookarounds keep a
#: 26-char run embedded in a longer base32 blob from matching.
_ULID_RE = re.compile(
    rb"(?<![0-9A-HJKMNP-TV-Z])[0-9A-HJKMNP-TV-Z]{26}(?![0-9A-HJKMNP-TV-Z])"
)

_TS_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d+))?\s*(Z|z|[+-]\d{2}:?\d{2})?$"
)

CONSUMED_DEFINITION = (
    "recall_events.{} = 1 -- a later memory_remember credited this recall "
    "event, i.e. it already carries the reinforcement signal that phase 2 "
    "exists to recover for the silent remainder".format(CONSUMED_COLUMN)
)

SELECTION_BIAS_RATIONALE = (
    "the coverage share alone cannot distinguish a neutral join from one that "
    "preferentially finds already-consumed events; on the local field the "
    "consumed share was 0.1717 store-wide but 0.6472 among matched events "
    "(3.8x), holding inside every calendar month separately, which showed the "
    "event_id_echo tier to be closer to a consumption detector than a neutral "
    "join -- a high coverage produced by that same mechanism would not support "
    "the goal's downstream claim, so the enrichment decides as much as the "
    "share does"
)

DENOMINATOR_DEFINITION = (
    "all recall_events whose created_at lies inside the corpus window "
    "[min(started_at), max(ended_at)] taken over the corpus-index rows "
    "actually used, bounds inclusive, timestamps compared as UTC epoch "
    "seconds; events without a parseable created_at are outside the "
    "denominator; the denominator is never narrowed to events carrying a "
    "transport_session_id or any other join-friendly field"
)

AMBIGUITY_RULE = (
    "candidate transcripts are collapsed by split_key identity group; a "
    "single surviving group is a match (deterministic representative: "
    "smallest session_key, tier 1 breaking ties by earliest echo record), "
    "multiple surviving groups are ambiguity and the tier yields nothing"
)


class JoinError(RuntimeError):
    """A run cannot proceed (bad store, unsealed index, empty window)."""


# --------------------------------------------------------------------------
# small pure helpers
# --------------------------------------------------------------------------


def parse_ts(value):
    """ISO-8601-ish text -> UTC epoch seconds, or None.

    Hand-rolled instead of ``datetime.fromisoformat`` so the one file runs
    identically on 3.8 through current: the corpus carries ``Z`` suffixes and
    3-digit fractions, the store carries plain ``Z`` seconds, memories carry
    ``+07:00`` offsets. Naive timestamps are taken as UTC.
    """

    if not value:
        return None
    match = _TS_RE.match(str(value).strip())
    if not match:
        return None
    try:
        moment = datetime(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
            int(match.group(5)),
            int(match.group(6)),
            int((match.group(7) or "").ljust(6, "0")[:6] or 0),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None
    epoch = moment.timestamp()
    offset = match.group(8)
    if offset and offset not in ("Z", "z"):
        sign = 1 if offset[0] == "+" else -1
        epoch -= sign * (int(offset[1:3]) * 3600 + int(offset[-2:]) * 60)
    return epoch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def falsifier_verdict(share):
    return "pass" if share >= FALSIFIER_THRESHOLD else "fail"


def _ratio(part, whole):
    return round(part / whole, 6) if whole else None


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def open_store_ro(store_path):
    """Open the sqlite store strictly read-only (file: URI, mode=ro)."""

    resolved = os.path.abspath(store_path)
    if not os.path.isfile(resolved):
        raise JoinError("store not found: {}".format(resolved))
    uri = "file:{}?mode=ro".format(urllib.parse.quote(resolved))
    return sqlite3.connect(uri, uri=True)


def fetch_events(conn):
    """Every recall event, with only the join-relevant fields.

    ``query`` and ``results`` are deliberately never selected: nothing this
    script writes may carry query text or node content.

    Returns ``(events, has_consumed_column)``. The consumed flag feeds the
    selection-bias measurement only -- it is aggregated into the coverage
    JSON and never written into the match table, whose columns are fixed.
    A store predating the column reports the measurement as unmeasured
    instead of silently counting every event as unconsumed.
    """

    columns = {row[1] for row in conn.execute("PRAGMA table_info(recall_events)")}
    has_consumed = CONSUMED_COLUMN in columns
    events = []
    cursor = conn.execute(
        "SELECT id, session_id, transport_session_id, created_at, ambient_context, {}"
        " FROM recall_events".format(CONSUMED_COLUMN if has_consumed else "NULL")
    )
    for event_id, session_id, transport_id, created_at, ambient_raw, consumed in cursor:
        cwd = None
        if ambient_raw:
            try:
                ambient = json.loads(ambient_raw)
            except ValueError:
                ambient = None
            if isinstance(ambient, dict):
                value = ambient.get("cwd")
                if isinstance(value, str) and value:
                    cwd = value
        events.append(
            {
                "id": event_id,
                "session_id": session_id,
                "transport_session_id": transport_id,
                "created_at": created_at,
                "ts": parse_ts(created_at),
                "cwd": cwd,
                "consumed": bool(consumed) if has_consumed else None,
            }
        )
    return events, has_consumed


def load_index(index_path, manifest, manifest_path=None):
    """Read index rows after proving they are the rows *this* manifest seals.

    The comparison is against ``index_sha256`` of the manifest handed to
    ``--manifest`` and nothing else, so a freshly built self-sealed pair
    verifies exactly as the sealed corpus pair does. What a run must never do
    is mix them: reading rows one manifest does not describe would silently
    reinterpret its counts, split digests and corpus window.
    """

    recorded = manifest.get("index_sha256")
    if not recorded:
        raise JoinError(
            "manifest carries no index_sha256, so it seals no index: {}".format(
                manifest_path or "<manifest>"
            )
        )
    actual = sha256_file(index_path)
    if recorded != actual:
        raise JoinError(
            "index does not match the manifest it was handed with ({}): "
            "manifest index_sha256 {} != actual index sha256 {}".format(
                manifest_path or "<manifest>", recorded, actual
            )
        )
    rows = []
    with open(index_path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            row["_start"] = parse_ts(row.get("started_at"))
            row["_end"] = parse_ts(row.get("ended_at"))
            row["_group"] = row.get("split_key") or row["session_key"]
            rows.append(row)
    if not rows:
        raise JoinError("index has no rows: {}".format(index_path))
    return rows, actual


def corpus_window(rows):
    starts = [row["_start"] for row in rows if row["_start"] is not None]
    ends = [row["_end"] for row in rows if row["_end"] is not None]
    if not starts or not ends:
        raise JoinError("no index row carries a parseable started_at/ended_at")
    return min(starts), max(ends)


def sealed_reference(sealed_path, manifest_sha, index_sha):
    """Record how the corpus actually used relates to the sealed corpus.

    This answers, on the record and without changing what the run does: is
    the manifest we joined against the sealed one, and does the index we read
    hash to the sealed corpus's ``index_sha256``? Both are expected to be
    false whenever a fresh corpus is used, and a reader must be able to see
    that from the artifact instead of inferring it from a path.
    """

    if not sealed_path:
        return {
            "path": None,
            "compared": False,
            "note": (
                "no --sealed-manifest handed, so the run makes no claim about "
                "the sealed corpus either way"
            ),
        }
    with open(sealed_path, "r", encoding="utf-8") as handle:
        sealed = json.load(handle)
    sealed_sha = sha256_file(sealed_path)
    sealed_index_sha = sealed.get("index_sha256")
    return {
        "path": os.path.abspath(sealed_path),
        "compared": True,
        "sha256": sealed_sha,
        "index_sha256": sealed_index_sha,
        "generated_at": sealed.get("generated_at"),
        "sessions": (sealed.get("counts") or {}).get("sessions"),
        "manifest_used_is_the_sealed_manifest": manifest_sha == sealed_sha,
        "index_used_matches_sealed_index_sha256": index_sha == sealed_index_sha,
    }


def drift_summary(report_path):
    """Condense a ``postsession_corpus.py --verify --report`` JSON.

    Both verify modes are summarised, because which one ran depends on
    whether the sealed index still exists: ``rederive`` rescans the sealed
    roots and can only count the drift, ``index`` checks the handed rows one
    by one and can also name appends, rewrites and disappearances.
    """

    if not report_path:
        return {"report_path": None, "measured": False}
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    summary = {
        "report_path": os.path.abspath(report_path),
        "report_sha256": sha256_file(report_path),
        "measured": True,
        "mode": report.get("mode"),
        "verify_ok": report.get("ok"),
    }
    if report.get("mode") == "rederive":
        rederived = report.get("rederived") or {}
        drift = rederived.get("counts_drift") or {}
        sessions = drift.get("sessions") or {}
        summary["method"] = (
            "the sealed index is gone, so the sealed roots were rescanned and "
            "the corpus re-derived; drift is counted, not matched per session_key"
        )
        summary["sessions"] = {
            "sealed": sessions.get("recorded"),
            "on_disk_now": sessions.get("rederived"),
        }
        summary["counts_drift"] = drift
        summary["split_seal_match"] = {
            name: bool(value.get("match"))
            for name, value in (rederived.get("seal") or {}).items()
        }
        summary["rederived_index_sha256"] = rederived.get("index_sha256")
        summary["rederived_index_sha256_equals_sealed"] = rederived.get(
            "index_sha256_match"
        )
    else:
        changed = report.get("changed") or []
        appended = sum(1 for item in changed if item.get("kind") == "append")
        summary["method"] = (
            "the index handed to --index was checked row by row against the "
            "sealed manifest"
        )
        summary["index"] = report.get("index")
        summary["rows_checked"] = report.get("checked")
        summary["rows_byte_identical"] = report.get("verified")
        summary["appended"] = appended
        summary["rewritten"] = len(changed) - appended
        summary["missing"] = len(report.get("missing") or [])
        summary["split_mismatches"] = len(report.get("split_mismatches") or [])
        summary["holdout"] = {
            "appended": len(report.get("holdout_appended") or []),
            "rewritten": len(report.get("holdout_rewritten") or []),
            "missing": len(report.get("holdout_missing") or []),
        }
    return summary


# --------------------------------------------------------------------------
# tier 1: the echo scan
# --------------------------------------------------------------------------


def scan_transcripts(rows, known_ids, stats):
    """One pass over every transcript's sealed prefix.

    Returns ``{event_id: {session_key: hit}}`` where a hit remembers the row
    and the smallest (earliest) 0-based record index the id appeared at --
    the delivery moment; later re-quotes of the same id do not move it.
    """

    known = {eid.encode("ascii") for eid in known_ids}
    hits = {}
    for row in rows:
        limit = int(row.get("records") or 0)
        try:
            handle = open(row["path"], "rb")
        except OSError:
            stats["missing_transcripts"] += 1
            continue
        with handle:
            for record_index, line in enumerate(handle):
                if record_index >= limit:
                    break
                for match in _ULID_RE.finditer(line):
                    token = match.group()
                    if token not in known:
                        continue
                    event_id = token.decode("ascii")
                    per_session = hits.setdefault(event_id, {})
                    if row["session_key"] not in per_session:
                        per_session[row["session_key"]] = {
                            "row": row,
                            "record_index": record_index,
                        }
        stats["transcripts_scanned"] += 1
    return hits


def _within_window(ts, row):
    """Does the event time fit the row's window (±slack)? Unknown side passes."""

    if ts is None:
        return True
    if row["_start"] is not None and ts < row["_start"] - WINDOW_SLACK_SECONDS:
        return False
    if row["_end"] is not None and ts > row["_end"] + WINDOW_SLACK_SECONDS:
        return False
    return True


def choose_tier1(event, hits, stats):
    per_session = hits.get(event["id"])
    if not per_session:
        return None
    valid = []
    for hit in per_session.values():
        if _within_window(event["ts"], hit["row"]):
            valid.append(hit)
        else:
            stats["tier1_hits_rejected_by_time"] += 1
    if not valid:
        return None
    groups = {hit["row"]["_group"] for hit in valid}
    if len(groups) != 1:
        stats["tier1_ambiguous"] += 1
        return None
    best = min(valid, key=lambda h: (h["row"]["session_key"], h["record_index"]))
    return _match(best["row"], "event_id_echo", best["record_index"])


def _match(row, method, record_index):
    return {
        "method": method,
        "session_key": row["session_key"],
        "path": row["path"],
        "sha256": row.get("sha256"),
        "group": row["_group"],
        "record_index": record_index,
    }


# --------------------------------------------------------------------------
# tiers 2-4
# --------------------------------------------------------------------------


def build_transport_map(events, tier1_choices):
    """transport_session_id -> {identity group: inherited match}."""

    transport_map = {}
    for event in events:
        choice = tier1_choices.get(event["id"])
        transport_id = event["transport_session_id"]
        if choice is None or not transport_id:
            continue
        groups = transport_map.setdefault(transport_id, {})
        kept = groups.get(choice["group"])
        if kept is None or choice["session_key"] < kept["session_key"]:
            inherited = dict(choice)
            inherited["method"] = "transport_session"
            inherited["record_index"] = None
            groups[choice["group"]] = inherited
    return transport_map


def make_tier2(transport_map, stats):
    def tier2(event):
        transport_id = event["transport_session_id"]
        if not transport_id:
            return None
        groups = transport_map.get(transport_id)
        if not groups:
            return None
        if len(groups) != 1:
            stats["tier2_ambiguous"] += 1
            return None
        return dict(next(iter(groups.values())))

    return tier2


def make_tier3(rows, stats):
    cli_map = {}
    linked_map = {}
    for row in rows:
        cli_id = row.get("cli_session_id")
        if cli_id:
            cli_map.setdefault(cli_id, []).append(row)
        for linked in row.get("linked_session_ids") or ():
            linked_map.setdefault(linked, []).append(row)

    def tier3(event):
        session_id = event["session_id"]
        if not session_id:
            return None
        direct = cli_map.get(session_id, [])
        linked = linked_map.get(session_id, [])
        seen = set()
        candidates = []  # (0 direct | 1 linked, row) -- direct rows first
        for rank, pool in ((0, direct), (1, linked)):
            for row in pool:
                if row["session_key"] in seen:
                    continue
                seen.add(row["session_key"])
                candidates.append((rank, row))
        if not candidates:
            return None
        groups = {row["_group"] for _rank, row in candidates}
        if len(groups) != 1:
            stats["tier3_ambiguous"] += 1
            return None
        _rank, best = min(candidates, key=lambda c: (c[0], c[1]["session_key"]))
        return _match(best, "cli_session", None)

    return tier3


def make_tier4(rows, stats):
    cwd_map = {}
    for row in rows:
        if row.get("cwd") and row["_start"] is not None and row["_end"] is not None:
            cwd_map.setdefault(row["cwd"], []).append(row)

    def tier4(event):
        if not event["cwd"] or event["ts"] is None:
            return None
        candidates = [
            row
            for row in cwd_map.get(event["cwd"], [])
            if row["_start"] - WINDOW_SLACK_SECONDS
            <= event["ts"]
            <= row["_end"] + WINDOW_SLACK_SECONDS
        ]
        if not candidates:
            return None
        groups = {row["_group"] for row in candidates}
        if len(groups) != 1:
            stats["tier4_ambiguous"] += 1
            return None
        best = min(candidates, key=lambda row: row["session_key"])
        return _match(best, "time_cwd_window", None)

    return tier4


# --------------------------------------------------------------------------
# the join
# --------------------------------------------------------------------------


def run_join(
    store_path,
    manifest_path,
    index_path,
    out_dir,
    field_label,
    sealed_manifest_path=None,
    drift_report_path=None,
    store_origin=None,
    self_verify_report_path=None,
):
    """The whole pipeline; returns the coverage document it wrote."""

    required = [
        ("store", store_path),
        ("manifest", manifest_path),
        ("index", index_path),
    ]
    if sealed_manifest_path:
        required.append(("sealed-manifest", sealed_manifest_path))
    if drift_report_path:
        required.append(("drift-report", drift_report_path))
    if self_verify_report_path:
        required.append(("self-verify-report", self_verify_report_path))
    for label, path in required:
        if not os.path.isfile(path):
            raise JoinError("{} not found: {}".format(label, path))
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest_sha = sha256_file(manifest_path)
    rows, index_sha = load_index(index_path, manifest, manifest_path)
    sealed = sealed_reference(sealed_manifest_path, manifest_sha, index_sha)
    drift = drift_summary(drift_report_path)
    own_drift = drift_summary(self_verify_report_path)
    window_lo, window_hi = corpus_window(rows)

    store_sha = sha256_file(store_path)
    conn = open_store_ro(store_path)
    try:
        events, has_consumed = fetch_events(conn)
    finally:
        conn.close()

    stats = {
        "transcripts_scanned": 0,
        "missing_transcripts": 0,
        "tier1_hits_rejected_by_time": 0,
        "tier1_ambiguous": 0,
        "tier2_ambiguous": 0,
        "tier3_ambiguous": 0,
        "tier4_ambiguous": 0,
    }

    denominator = [
        event
        for event in events
        if event["ts"] is not None and window_lo <= event["ts"] <= window_hi
    ]
    denominator.sort(key=lambda event: (event["created_at"], event["id"]))

    # Tier 1 scans for EVERY event id in the store, not only the denominator:
    # membership is what tells a real echo from a random base32 token, and
    # out-of-window ids showing up as stale quotes must be recognized to be
    # rejected by the time check rather than silently skipped.
    hits = scan_transcripts(rows, {event["id"] for event in events}, stats)
    # The structural ceiling for tier 1, before any time or ambiguity check:
    # an event whose id appears in no surviving transcript can never be echo
    # matched, so this separates "the join is weak" from "the transcripts are
    # gone". Counted over the whole store and over the denominator alone.
    stats["events_with_any_echo"] = len(hits)
    stats["denominator_events_with_any_echo"] = sum(
        1 for event in denominator if event["id"] in hits
    )
    tier1_choices = {}
    for event in events:
        choice = choose_tier1(event, hits, stats)
        if choice is not None:
            tier1_choices[event["id"]] = choice

    tier2 = make_tier2(build_transport_map(events, tier1_choices), stats)
    tier3 = make_tier3(rows, stats)
    tier4 = make_tier4(rows, stats)

    by_method = {name: 0 for name in METHODS}
    table_rows = []
    matches = {}
    for event in denominator:
        match = tier1_choices.get(event["id"]) or tier2(event) or tier3(event) or tier4(event)
        if match is not None:
            matches[event["id"]] = match
            by_method[match["method"]] += 1
        table_rows.append(
            {
                "event_id": event["id"],
                "created_at": event["created_at"],
                "matched": match is not None,
                "method": match["method"] if match else None,
                "corpus_session_id": match["session_key"] if match else None,
                "transcript_path": match["path"] if match else None,
                "transcript_sha256": match["sha256"] if match else None,
                "delivery_record_index": match["record_index"] if match else None,
            }
        )

    total = len(denominator)
    matched = len(matches)
    share = round(matched / total, 6) if total else 0.0

    audit = audit_fallback_precision(denominator, matches, tier3, tier4)
    bias = selection_bias_report(events, denominator, matches, has_consumed)

    os.makedirs(out_dir, exist_ok=True)
    table_path = os.path.join(out_dir, "match-table-{}.jsonl.gz".format(field_label))
    table_sha = write_match_table(table_rows, table_path)

    coverage = {
        "generator": GENERATOR,
        "generated_at": utc_now_iso(),
        "python": sys.version.split()[0],
        "field_label": field_label,
        "store": {
            "path": os.path.abspath(store_path),
            "sha256": store_sha,
            "origin": store_origin,
            "access": "sqlite3 file: URI, mode=ro (read-only)",
        },
        "manifest": {
            "path": os.path.abspath(manifest_path),
            "sha256": manifest_sha,
            "index_sha256": manifest.get("index_sha256"),
            "generated_at": manifest.get("generated_at"),
            "sessions": (manifest.get("counts") or {}).get("sessions"),
            "is_sealed_corpus": sealed.get("manifest_used_is_the_sealed_manifest"),
        },
        "index": {
            "path": os.path.abspath(index_path),
            "sha256": index_sha,
            "rows": len(rows),
            "verified_against_manifest": True,
            "gate": (
                "sha256(index) == index_sha256 of the manifest handed to "
                "--manifest; the gate never compares against a manifest the "
                "run was not given"
            ),
            "equals_sealed_corpus_index_sha256": sealed.get(
                "index_used_matches_sealed_index_sha256"
            ),
        },
        "sealed_corpus": sealed,
        "drift_vs_sealed_manifest": drift,
        "drift_vs_own_manifest": own_drift,
        "denominator": {
            "definition": DENOMINATOR_DEFINITION,
            "window_utc": {
                "min_started_at_epoch": window_lo,
                "max_ended_at_epoch": window_hi,
            },
            "store_total_events": len(events),
        },
        "coverage": {
            "total_events": total,
            "matched_events": matched,
            "share": share,
            "by_method": dict(by_method, unmatched=total - matched),
        },
        "fallback_precision": audit,
        "selection_bias": bias,
        "falsifier_60pct": {
            "threshold": FALSIFIER_THRESHOLD,
            "share": share,
            "verdict": falsifier_verdict(share),
        },
        "match_table": {
            "path": os.path.abspath(table_path),
            "rows": len(table_rows),
            "sha256": table_sha,
        },
        "rules": {
            "window_slack_seconds": WINDOW_SLACK_SECONDS,
            "sealed_prefix": "at most index.records records read per transcript",
            "record_index_base": 0,
            "ambiguity": AMBIGUITY_RULE,
        },
        "diagnostics": stats,
    }
    coverage_path = os.path.join(out_dir, "coverage-{}.json".format(field_label))
    with open(coverage_path, "w", encoding="utf-8") as handle:
        json.dump(coverage, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    coverage["_coverage_path"] = coverage_path
    return coverage


def audit_fallback_precision(denominator, matches, tier3, tier4):
    """Blinded: tiers 3-4 re-predict the tier-1/2 gold subset.

    The tier functions are the exact closures the main pass used; they carry
    no echo state, so they cannot peek at the gold answer. Agreement is at
    identity-group level: predicting the other recording of the same
    conversation is agreement, predicting a different conversation is not.
    """

    gold = [
        (event, matches[event["id"]])
        for event in denominator
        if event["id"] in matches
        and matches[event["id"]]["method"] in ("event_id_echo", "transport_session")
    ]
    by_tier = {}
    for name, tier in (("cli_session", tier3), ("time_cwd_window", tier4)):
        predicted = 0
        agreed = 0
        for event, gold_match in gold:
            prediction = tier(event)
            if prediction is None:
                continue
            predicted += 1
            if prediction["group"] == gold_match["group"]:
                agreed += 1
        by_tier[name] = {
            "predicted": predicted,
            "agreed": agreed,
            "precision": _ratio(agreed, predicted),
        }
    return {
        "gold_subset_events": len(gold),
        "gold_min_events": GOLD_MIN_EVENTS,
        "low_confidence": len(gold) < GOLD_MIN_EVENTS,
        "agreement_level": "split_key identity group",
        "by_tier": by_tier,
    }


def _consumed_cell(events):
    """Consumed count and share over a set of events."""

    total = len(events)
    consumed = sum(1 for event in events if event["consumed"])
    return {"events": total, "consumed": consumed, "share": _ratio(consumed, total)}


def _month_key(ts):
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")


def selection_bias_report(
    events, denominator, matches, has_consumed, min_cell=MONTH_MIN_CELL_EVENTS
):
    """Is the join neutral, or does it preferentially find consumed events?

    Reports the consumed share store-wide, over the denominator, among
    matched and unmatched events and per tier, plus the within-month control
    that separates a real enrichment from matched events merely being newer.
    Enrichment ratios are computed from the unrounded shares.
    """

    if not has_consumed:
        return {
            "measured": False,
            "reason": (
                "the store has no recall_events.{} column, so consumption "
                "cannot be observed on this field".format(CONSUMED_COLUMN)
            ),
        }

    matched_events = [event for event in denominator if event["id"] in matches]
    unmatched_events = [event for event in denominator if event["id"] not in matches]
    store_wide = _consumed_cell(events)
    matched_cell = _consumed_cell(matched_events)
    unmatched_cell = _consumed_cell(unmatched_events)

    by_method = {}
    for name in METHODS:
        by_method[name] = _consumed_cell(
            [
                event
                for event in matched_events
                if matches[event["id"]]["method"] == name
            ]
        )
    by_method["unmatched"] = unmatched_cell

    months = {}
    for event in denominator:
        cell = months.setdefault(_month_key(event["ts"]), {"matched": [], "unmatched": []})
        cell["matched" if event["id"] in matches else "unmatched"].append(event)

    month_report = {}
    comparable = 0
    holds = True
    for key in sorted(months, key=lambda k: (k is None, k)):
        matched_month = _consumed_cell(months[key]["matched"])
        unmatched_month = _consumed_cell(months[key]["unmatched"])
        is_comparable = (
            matched_month["events"] >= min_cell and unmatched_month["events"] >= min_cell
        )
        exceeds = None
        if is_comparable:
            comparable += 1
            exceeds = matched_month["share"] > unmatched_month["share"]
            if not exceeds:
                holds = False
        month_report[key] = {
            "matched": matched_month,
            "unmatched": unmatched_month,
            "comparable": is_comparable,
            "matched_exceeds_unmatched": exceeds,
        }

    return {
        "measured": True,
        "consumed_definition": CONSUMED_DEFINITION,
        "why_this_is_measured": SELECTION_BIAS_RATIONALE,
        "store_wide": store_wide,
        "denominator": _consumed_cell(denominator),
        "matched": matched_cell,
        "unmatched": unmatched_cell,
        "enrichment_matched_over_store_wide": _enrichment(matched_cell, store_wide),
        "enrichment_matched_over_unmatched": _enrichment(matched_cell, unmatched_cell),
        "by_method": by_method,
        "within_month_control": {
            "min_events_per_cell": min_cell,
            "control": (
                "a recency confound is ruled out only if the matched/unmatched "
                "consumed-share gap also holds inside each calendar month taken "
                "separately"
            ),
            "comparable_months": comparable,
            "matched_exceeds_unmatched_in_every_comparable_month": (
                holds if comparable else None
            ),
            "months": month_report,
        },
    }


def _enrichment(cell, baseline):
    """Ratio of two consumed shares, from the unrounded counts."""

    if not cell["events"] or not baseline["events"] or not baseline["consumed"]:
        return None
    return round(
        (cell["consumed"] / cell["events"]) / (baseline["consumed"] / baseline["events"]),
        6,
    )


def write_match_table(table_rows, path):
    """Deterministic gzip (mtime=0): same rows -> same bytes -> same sha."""

    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as gz:
        for row in table_rows:
            gz.write(
                (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                )
            )
    payload = buffer.getvalue()
    with open(path, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# self-test: synthetic fixtures, every tier plus the traps
# --------------------------------------------------------------------------


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _fid(tag):
    """A synthetic 26-char ULID-alphabet id (no I/L/O/U in tags)."""

    body = "01" + tag.upper()
    _check(len(body) <= 26, "fixture tag too long: " + tag)
    _check(
        all(ch in "0123456789ABCDEFGHJKMNPQRSTVWXYZ" for ch in body),
        "fixture tag outside the ULID alphabet: " + tag,
    )
    return body + "0" * (26 - len(body))


def _iso(hour, minute, day=1):
    return "2026-08-{:02d}T{:02d}:{:02d}:00Z".format(day, hour, minute)


class _Fixture:
    """Builds a tiny store + transcripts + sealed index/manifest in tmp."""

    def __init__(self, root):
        self.root = root
        self.transcripts = os.path.join(root, "transcripts")
        os.makedirs(self.transcripts)
        self.rows = []
        self.events = []

    def add_transcript(self, name, lines, sealed_records=None):
        """Write JSONL lines; seal (hash+count) only the first sealed_records."""

        path = os.path.join(self.transcripts, name + ".jsonl")
        blobs = [
            (json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8")
            for line in lines
        ]
        keep = len(blobs) if sealed_records is None else sealed_records
        with open(path, "wb") as handle:
            for blob in blobs:
                handle.write(blob)
        sealed = b"".join(blobs[:keep])
        return path, hashlib.sha256(sealed).hexdigest(), keep

    def add_session(self, key, lines, *, sealed_records=None, split_key=None, **extra):
        path, sha, records = self.add_transcript(key, lines, sealed_records)
        row = {
            "session_key": key,
            "path": path,
            "sha256": sha,
            "records": records,
            "split_key": split_key or key,
            "split": "train",
        }
        row.update(extra)
        self.rows.append(row)
        return row

    def add_event(
        self, event_id, created_at, session_id=None, transport=None, cwd=None,
        consumed=False,
    ):
        self.events.append((event_id, created_at, session_id, transport, cwd, consumed))

    def write_store(self, name="store.sqlite3", with_consumed=True):
        """The store; ``with_consumed=False`` models a pre-column store."""

        path = os.path.join(self.root, name)
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE recall_events ("
            " id TEXT PRIMARY KEY, query TEXT NOT NULL DEFAULT '',"
            " ambient_context TEXT NOT NULL DEFAULT '{}',"
            " session_id TEXT, transport_session_id TEXT, created_at TEXT NOT NULL"
            + (
                ", {} INTEGER NOT NULL DEFAULT 0)".format(CONSUMED_COLUMN)
                if with_consumed
                else ")"
            )
        )
        columns = 6 + (1 if with_consumed else 0)
        for event_id, created_at, session_id, transport, cwd, consumed in self.events:
            ambient = json.dumps({"cwd": cwd}) if cwd else "{}"
            values = [
                event_id,
                "SECRET-QUERY-TEXT do not leak",
                ambient,
                session_id,
                transport,
                created_at,
            ]
            if with_consumed:
                values.append(1 if consumed else 0)
            conn.execute(
                "INSERT INTO recall_events VALUES ({})".format(
                    ", ".join("?" * columns)
                ),
                values,
            )
        conn.commit()
        conn.close()
        return path

    def write_index_and_manifest(self):
        index_path = os.path.join(self.root, "corpus-index.jsonl")
        with open(index_path, "w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        manifest_path = os.path.join(self.root, "corpus.json")
        manifest = {
            "manifest_version": 2,
            "index_sha256": sha256_file(index_path),
            "counts": {"sessions": len(self.rows)},
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return index_path, manifest_path


def _filler(n):
    return [{"type": "message", "i": i} for i in range(n)]


def _echo(event_id):
    return {"type": "tool_result", "recall_event_id": event_id}


def build_fixture(root):
    fx = _Fixture(root)
    ev = {
        name: _fid(name)
        for name in (
            "EVA", "EVB", "EVC", "EVCB", "EVD", "EVE", "EVF", "EVG",
            "EVH", "EVJ", "EVK", "EVX1", "EVX2", "EVM",
        )
    }

    # tier 1: sesA echoes EVA at record 2; also the tier-2 donor for EVB.
    fx.add_session(
        "s-a",
        _filler(2) + [_echo(ev["EVA"])] + _filler(1),
        started_at=_iso(1, 0), ended_at=_iso(2, 0),
        cli_session_id="cli-a", cwd="/w/a",
    )
    # tier 3 target: direct cli id and a linked id.
    fx.add_session(
        "s-b", _filler(2),
        started_at=_iso(3, 0), ended_at=_iso(4, 0),
        cli_session_id="cli-b", linked_session_ids=["link-b1"], cwd="/w/b",
    )
    # tier 4 target: unique cwd+window.
    fx.add_session(
        "s-c", _filler(2), started_at=_iso(5, 0), ended_at=_iso(6, 0), cwd="/w/c",
    )
    # tier 4 ambiguity: two DIFFERENT identity groups share cwd and window.
    fx.add_session(
        "s-d", _filler(2), started_at=_iso(7, 0), ended_at=_iso(8, 0), cwd="/w/dup",
    )
    fx.add_session(
        "s-e", _filler(2), started_at=_iso(7, 0), ended_at=_iso(8, 0), cwd="/w/dup",
    )
    # sealed prefix: the echo of EVF lands AFTER the sealed 3 records (live
    # growth), so tier 1 must not see it.
    fx.add_session(
        "s-f",
        _filler(3) + _filler(2) + [_echo(ev["EVF"])],
        sealed_records=3,
        started_at=_iso(9, 0), ended_at=_iso(10, 0),
    )
    # stale-quote trap: s-g quotes EVG verbatim, but EVG happened hours after
    # s-g ended -- the echo must be rejected and tier 3 must pick s-h instead.
    fx.add_session(
        "s-g",
        [{"type": "assistant", "text": "old event {} came up".format(ev["EVG"])}],
        started_at=_iso(1, 0), ended_at=_iso(1, 30),
    )
    fx.add_session(
        "s-h", _filler(2),
        started_at=_iso(10, 30), ended_at=_iso(11, 30), cli_session_id="cli-h",
    )
    # duplicate recording: one conversation, two transcripts, same split_key;
    # the echo appears in both and must collapse to one candidate (min key).
    fx.add_session(
        "s-i1", _filler(1) + [_echo(ev["EVH"])] + _filler(1),
        split_key="g-i", started_at=_iso(12, 0), ended_at=_iso(13, 0),
    )
    fx.add_session(
        "s-i2", _filler(3) + [_echo(ev["EVH"])] + _filler(1),
        split_key="g-i", started_at=_iso(12, 0), ended_at=_iso(13, 0),
    )
    # tier-2 ambiguity: two echoes in DIFFERENT groups share one transport id.
    fx.add_session(
        "s-j1", [_echo(ev["EVX1"])], started_at=_iso(14, 0), ended_at=_iso(15, 0),
    )
    fx.add_session(
        "s-j2", [_echo(ev["EVX2"])] + _filler(1),
        started_at=_iso(14, 0), ended_at=_iso(15, 0),
    )
    # a row whose transcript no longer exists on disk.
    fx.rows.append(
        {
            "session_key": "s-m",
            "path": os.path.join(fx.transcripts, "gone.jsonl"),
            "sha256": "0" * 64,
            "records": 2,
            "split_key": "s-m",
            "split": "train",
            "started_at": _iso(2, 0),
            "ended_at": _iso(2, 30),
        }
    )

    # events; corpus window is [01:00, 15:00] on 2026-08-01. The ``consumed``
    # flags are laid out so the selection-bias block has something to measure:
    # matched 4/9, unmatched 1/4, store-wide 6/14 (EVM sits outside the
    # denominator but inside the store-wide baseline).
    fx.add_event(ev["EVA"], _iso(1, 10), session_id="cli-a", transport="tp-1", consumed=True)
    fx.add_event(ev["EVB"], _iso(1, 20), transport="tp-1", cwd="/w/a")
    fx.add_event(ev["EVC"], _iso(3, 10), session_id="cli-b", consumed=True)
    fx.add_event(ev["EVCB"], _iso(3, 20), session_id="link-b1")
    fx.add_event(ev["EVD"], _iso(5, 30), cwd="/w/c")
    fx.add_event(ev["EVE"], _iso(7, 30), cwd="/w/dup")     # ambiguous -> unmatched
    fx.add_event(ev["EVK"], _iso(6, 30))                   # nothing -> unmatched
    fx.add_event(ev["EVF"], _iso(9, 30), consumed=True)    # echo beyond seal -> unmatched
    fx.add_event(ev["EVG"], _iso(11, 0), session_id="cli-h")
    fx.add_event(ev["EVH"], _iso(12, 30), session_id="cli-b", consumed=True)  # wrong on purpose (audit)
    fx.add_event(ev["EVX1"], _iso(14, 10), transport="tp-x", consumed=True)
    fx.add_event(ev["EVX2"], _iso(14, 20), transport="tp-x")
    fx.add_event(ev["EVJ"], _iso(14, 30), transport="tp-x")    # tier-2 ambiguous
    fx.add_event(ev["EVM"], _iso(0, 0, day=5), consumed=True)  # outside denominator
    return fx, ev


def _selection_bias_checks():
    """The bias block on synthetic events spanning two calendar months.

    Built so the enrichment is real and month-invariant: matched events run
    0.5 consumed against 0.1 unmatched in *each* month, with the echo tier
    (0.9) carrying it and the transport tier (0.1) sitting at the unmatched
    base rate -- the shape the local field actually measured.
    """

    failures = []
    events = []
    matches = {}
    for month in (5, 6):
        for i in range(40):
            eid = "m{}-hit-{}".format(month, i)
            echo = i < 20
            events.append(
                {
                    "id": eid,
                    "ts": parse_ts("2026-{:02d}-10T00:00:00Z".format(month)),
                    "consumed": (i < 18) if echo else (i < 22),
                }
            )
            matches[eid] = {"method": "event_id_echo" if echo else "transport_session"}
        for i in range(40):
            eid = "m{}-miss-{}".format(month, i)
            events.append(
                {
                    "id": eid,
                    "ts": parse_ts("2026-{:02d}-11T00:00:00Z".format(month)),
                    "consumed": i < 4,
                }
            )

    report = selection_bias_report(events, events, matches, True)
    expected = {
        "measured": True,
        "store_wide": {"events": 160, "consumed": 48, "share": 0.3},
        "matched": {"events": 80, "consumed": 40, "share": 0.5},
        "unmatched": {"events": 80, "consumed": 8, "share": 0.1},
        "enrichment_matched_over_store_wide": round(0.5 / 0.3, 6),
        "enrichment_matched_over_unmatched": 5.0,
    }
    for key, value in expected.items():
        if report[key] != value:
            failures.append(
                "selection_bias[{}] = {!r}, expected {!r}".format(key, report[key], value)
            )
    if report["by_method"]["event_id_echo"] != {
        "events": 40, "consumed": 36, "share": 0.9
    }:
        failures.append("selection_bias by_method echo: {}".format(report["by_method"]))
    if report["by_method"]["transport_session"] != {
        "events": 40, "consumed": 4, "share": 0.1
    }:
        failures.append(
            "selection_bias by_method transport: {}".format(report["by_method"])
        )
    if report["by_method"]["cli_session"] != {
        "events": 0, "consumed": 0, "share": None
    }:
        failures.append("an empty tier must report share None, not 0")

    control = report["within_month_control"]
    if control["comparable_months"] != 2 or not control[
        "matched_exceeds_unmatched_in_every_comparable_month"
    ]:
        failures.append("within-month control: {}".format(control))
    if sorted(control["months"]) != ["2026-05", "2026-06"]:
        failures.append("month keys: {}".format(sorted(control["months"])))
    if control["months"]["2026-05"]["matched"]["share"] != 0.5:
        failures.append("month cell: {}".format(control["months"]["2026-05"]))

    # a month where the gap does NOT hold must flip the control off
    broken = [
        dict(event, consumed=False)
        if event["id"].startswith("m6-hit")
        else event
        for event in events
    ]
    broken_report = selection_bias_report(broken, broken, matches, True)
    broken_control = broken_report["within_month_control"]
    if broken_control["matched_exceeds_unmatched_in_every_comparable_month"] is not False:
        failures.append(
            "a month without the gap must falsify the control: {}".format(broken_control)
        )
    if broken_control["months"]["2026-05"]["matched_exceeds_unmatched"] is not True:
        failures.append("the surviving month must still report its own verdict")

    # thin cells are excluded rather than turned into a verdict
    thin = selection_bias_report(events, events, matches, True, min_cell=1000)
    thin_control = thin["within_month_control"]
    if (
        thin_control["comparable_months"] != 0
        or thin_control["matched_exceeds_unmatched_in_every_comparable_month"] is not None
        or thin_control["months"]["2026-05"]["matched_exceeds_unmatched"] is not None
    ):
        failures.append("thin months must not produce a verdict: {}".format(thin_control))

    unmeasured = selection_bias_report(events, events, matches, False)
    if unmeasured.get("measured") is not False or "reason" not in unmeasured:
        failures.append("a store without the column must report measured false")
    return failures


def self_test():
    failures = []

    # the pure pieces first
    reference = datetime(2026, 8, 28, 22, 20, 18, tzinfo=timezone.utc).timestamp()
    for value, expected in (
        ("2026-08-28T22:20:18Z", reference),
        ("2026-08-28T22:20:18.500Z", reference + 0.5),
        ("2026-08-28T22:20:18", reference),
        ("2026-08-23T00:00:00+07:00", parse_ts("2026-08-22T17:00:00Z")),
        ("not-a-timestamp", None),
        (None, None),
    ):
        got = parse_ts(value)
        if got != expected:
            failures.append("parse_ts({!r}) = {!r}, expected {!r}".format(value, got, expected))
    if falsifier_verdict(0.6) != "pass" or falsifier_verdict(0.599999) != "fail":
        failures.append("falsifier_verdict threshold is wrong")
    failures.extend(_selection_bias_checks())
    if failures:
        for line in failures:
            print("SELF-TEST FAIL:", line, file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="transcript-join-selftest-") as root:
        fx, ev = build_fixture(root)
        store = fx.write_store()
        index_path, manifest_path = fx.write_index_and_manifest()
        out_dir = os.path.join(root, "out")
        coverage = run_join(store, manifest_path, index_path, out_dir, "selftest")

        try:
            cov = coverage["coverage"]
            _check(cov["total_events"] == 13, "denominator: {}".format(cov["total_events"]))
            _check(cov["matched_events"] == 9, "matched: {}".format(cov["matched_events"]))
            _check(
                cov["by_method"]
                == {
                    "event_id_echo": 4,
                    "transport_session": 1,
                    "cli_session": 3,
                    "time_cwd_window": 1,
                    "unmatched": 4,
                },
                "by_method: {}".format(cov["by_method"]),
            )
            _check(cov["share"] == round(9 / 13, 6), "share: {}".format(cov["share"]))
            _check(
                coverage["falsifier_60pct"]["verdict"] == "pass",
                "falsifier verdict on 9/13",
            )
            _check(
                coverage["denominator"]["store_total_events"] == 14,
                "store_total_events",
            )

            audit = coverage["fallback_precision"]
            _check(audit["gold_subset_events"] == 5, "gold subset: {}".format(audit))
            _check(audit["low_confidence"] is True, "low_confidence flag")
            _check(
                audit["by_tier"]["cli_session"] == {"predicted": 2, "agreed": 1, "precision": 0.5},
                "cli_session audit: {}".format(audit["by_tier"]),
            )
            _check(
                audit["by_tier"]["time_cwd_window"]
                == {"predicted": 1, "agreed": 1, "precision": 1.0},
                "time_cwd audit: {}".format(audit["by_tier"]),
            )

            bias = coverage["selection_bias"]
            _check(bias["measured"] is True, "selection bias not measured")
            for key, expected in (
                ("store_wide", {"events": 14, "consumed": 6, "share": round(6 / 14, 6)}),
                ("denominator", {"events": 13, "consumed": 5, "share": round(5 / 13, 6)}),
                ("matched", {"events": 9, "consumed": 4, "share": round(4 / 9, 6)}),
                ("unmatched", {"events": 4, "consumed": 1, "share": 0.25}),
            ):
                _check(bias[key] == expected, "selection_bias[{}]: {}".format(key, bias[key]))
            _check(
                bias["by_method"]["event_id_echo"]
                == {"events": 4, "consumed": 3, "share": 0.75},
                "selection_bias by_method: {}".format(bias["by_method"]),
            )
            _check(
                bias["by_method"]["unmatched"] == bias["unmatched"],
                "the unmatched cell must agree between by_method and the top level",
            )

            diag = coverage["diagnostics"]
            _check(diag["missing_transcripts"] == 1, "missing transcripts: {}".format(diag))
            # EVA, EVG (stale quote), EVH (both copies), EVX1, EVX2 echo
            # somewhere; EVF's echo lies beyond the sealed prefix, so the
            # ceiling must not count it.
            _check(
                diag["events_with_any_echo"] == 5
                and diag["denominator_events_with_any_echo"] == 5,
                "echo ceiling: {}".format(diag),
            )
            _check(diag["tier1_hits_rejected_by_time"] == 1, "stale-quote rejection: {}".format(diag))
            _check(diag["tier2_ambiguous"] >= 1, "tier2 ambiguity: {}".format(diag))
            _check(diag["tier4_ambiguous"] >= 1, "tier4 ambiguity: {}".format(diag))

            # the match table, row by row
            table_path = coverage["match_table"]["path"]
            with gzip.open(table_path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            _check(len(rows) == 13, "match table rows: {}".format(len(rows)))
            by_id = {row["event_id"]: row for row in rows}
            allowed = {
                "event_id", "created_at", "matched", "method", "corpus_session_id",
                "transcript_path", "transcript_sha256", "delivery_record_index",
            }
            for row in rows:
                _check(set(row) == allowed, "row keys: {}".format(sorted(row)))
            _check(
                rows == sorted(rows, key=lambda r: (r["created_at"], r["event_id"])),
                "match table is not sorted",
            )

            row_a = by_id[ev["EVA"]]
            _check(
                row_a["method"] == "event_id_echo"
                and row_a["corpus_session_id"] == "s-a"
                and row_a["delivery_record_index"] == 2,
                "EVA row: {}".format(row_a),
            )
            row_b = by_id[ev["EVB"]]
            _check(
                row_b["method"] == "transport_session"
                and row_b["corpus_session_id"] == "s-a"
                and row_b["delivery_record_index"] is None,
                "EVB row: {}".format(row_b),
            )
            _check(by_id[ev["EVC"]]["method"] == "cli_session", "EVC method")
            _check(by_id[ev["EVCB"]]["method"] == "cli_session", "EVCB (linked) method")
            row_d = by_id[ev["EVD"]]
            _check(
                row_d["method"] == "time_cwd_window" and row_d["corpus_session_id"] == "s-c",
                "EVD row: {}".format(row_d),
            )
            for name in ("EVE", "EVK", "EVF", "EVJ"):
                row = by_id[ev[name]]
                _check(
                    row["matched"] is False and row["method"] is None
                    and row["corpus_session_id"] is None
                    and row["delivery_record_index"] is None,
                    "{} should be unmatched: {}".format(name, row),
                )
            row_g = by_id[ev["EVG"]]
            _check(
                row_g["method"] == "cli_session" and row_g["corpus_session_id"] == "s-h",
                "EVG (stale quote) row: {}".format(row_g),
            )
            row_h = by_id[ev["EVH"]]
            _check(
                row_h["method"] == "event_id_echo"
                and row_h["corpus_session_id"] == "s-i1"
                and row_h["delivery_record_index"] == 1,
                "EVH (duplicate collapse) row: {}".format(row_h),
            )
            _check(ev["EVM"] not in by_id, "EVM leaked into the denominator")

            # nothing anywhere may carry query text
            with open(coverage["_coverage_path"], "r", encoding="utf-8") as handle:
                coverage_text = handle.read()
            with gzip.open(table_path, "rt", encoding="utf-8") as handle:
                table_text = handle.read()
            _check(
                "SECRET-QUERY-TEXT" not in coverage_text
                and "SECRET-QUERY-TEXT" not in table_text,
                "query text leaked into an output artifact",
            )

            # determinism: a second run over the same inputs must byte-match
            second = run_join(store, manifest_path, index_path, os.path.join(root, "out2"), "selftest")
            _check(
                second["match_table"]["sha256"] == coverage["match_table"]["sha256"],
                "match table is not deterministic",
            )

            # the gate verifies against WHICHEVER manifest was handed, and an
            # unqualified run claims nothing about the sealed corpus
            _check(
                coverage["index"]["verified_against_manifest"] is True
                and coverage["index"]["equals_sealed_corpus_index_sha256"] is None
                and coverage["sealed_corpus"]["compared"] is False
                and coverage["drift_vs_sealed_manifest"]["measured"] is False
                and coverage["drift_vs_own_manifest"]["measured"] is False,
                "unqualified run made a sealed claim: {}".format(
                    coverage["sealed_corpus"]
                ),
            )

            alien_path = os.path.join(root, "alien-sealed.json")
            with open(alien_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "manifest_version": 2,
                        "index_sha256": "00" * 32,
                        "counts": {"sessions": 3953},
                    },
                    handle,
                )
            drift_path = os.path.join(root, "verify-vs-sealed.json")
            with open(drift_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "mode": "rederive",
                        "ok": True,
                        "rederived": {
                            "counts_drift": {
                                "sessions": {"recorded": 3953, "rederived": 4668}
                            },
                            "seal": {"train": {"match": False}},
                            "index_sha256": "11" * 32,
                            "index_sha256_match": False,
                        },
                    },
                    handle,
                )
            qualified = run_join(
                store,
                manifest_path,
                index_path,
                os.path.join(root, "out4"),
                "selftest",
                sealed_manifest_path=alien_path,
                drift_report_path=drift_path,
                store_origin="/live/global.sqlite3",
            )
            _check(
                qualified["index"]["equals_sealed_corpus_index_sha256"] is False
                and qualified["manifest"]["is_sealed_corpus"] is False
                and qualified["sealed_corpus"]["index_sha256"] == "00" * 32,
                "a fresh corpus must report itself as not the seal: {}".format(
                    qualified["sealed_corpus"]
                ),
            )
            _check(
                qualified["drift_vs_sealed_manifest"]["sessions"]
                == {"sealed": 3953, "on_disk_now": 4668}
                and qualified["drift_vs_sealed_manifest"]["split_seal_match"]
                == {"train": False},
                "drift summary: {}".format(qualified["drift_vs_sealed_manifest"]),
            )
            _check(
                qualified["store"]["origin"] == "/live/global.sqlite3",
                "store origin not recorded: {}".format(qualified["store"]),
            )
            _check(
                qualified["coverage"] == coverage["coverage"],
                "recording the sealed comparison changed the measurement",
            )

            # a --self-verify-report is summarised under its own key, and the
            # index-mode branch of the summariser is exercised here
            own_path = os.path.join(root, "verify-own.json")
            with open(own_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "mode": "index",
                        "ok": True,
                        "index": "corpus-index-alt.jsonl",
                        "checked": 12,
                        "verified": 11,
                        "changed": [{"kind": "append"}, {"kind": "rewrite"}],
                        "missing": [{"session_key": "s-m"}],
                        "split_mismatches": [],
                    },
                    handle,
                )
            with_own = run_join(
                store,
                manifest_path,
                index_path,
                os.path.join(root, "out7"),
                "selftest",
                self_verify_report_path=own_path,
            )
            own = with_own["drift_vs_own_manifest"]
            _check(
                own["measured"] is True
                and own["mode"] == "index"
                and own["rows_checked"] == 12
                and own["rows_byte_identical"] == 11
                and own["appended"] == 1
                and own["rewritten"] == 1
                and own["missing"] == 1,
                "own-manifest drift summary: {}".format(own),
            )
            _check(
                with_own["coverage"] == coverage["coverage"]
                and with_own["drift_vs_sealed_manifest"]["measured"] is False,
                "recording own-manifest drift changed the measurement",
            )

            # a store predating feedback_applied reports the bias unmeasured
            # instead of counting every event as unconsumed
            old_store = fx.write_store(name="store-old.sqlite3", with_consumed=False)
            old = run_join(
                old_store, manifest_path, index_path, os.path.join(root, "out8"), "selftest"
            )
            _check(
                old["selection_bias"]["measured"] is False
                and "reason" in old["selection_bias"]
                and old["coverage"] == coverage["coverage"],
                "pre-column store: {}".format(old["selection_bias"]),
            )

            # a self-sealed manifest verifies as its own seal
            self_sealed = run_join(
                store,
                manifest_path,
                index_path,
                os.path.join(root, "out5"),
                "selftest",
                sealed_manifest_path=manifest_path,
            )
            _check(
                self_sealed["index"]["equals_sealed_corpus_index_sha256"] is True
                and self_sealed["manifest"]["is_sealed_corpus"] is True,
                "self-sealed manifest failed its own seal comparison",
            )

            # a manifest that seals no index at all must refuse
            bare_path = os.path.join(root, "bare.json")
            with open(bare_path, "w", encoding="utf-8") as handle:
                json.dump({"manifest_version": 2}, handle)
            try:
                run_join(
                    store, bare_path, index_path, os.path.join(root, "out6"), "selftest"
                )
            except JoinError:
                pass
            else:
                _check(False, "manifest without index_sha256 was accepted")

            # an index that does not hash to the manifest it came with must refuse
            with open(index_path, "a", encoding="utf-8") as handle:
                handle.write("\n")
            try:
                run_join(store, manifest_path, index_path, os.path.join(root, "out3"), "selftest")
            except JoinError:
                pass
            else:
                _check(False, "tampered index was accepted")
        except AssertionError as error:
            print("SELF-TEST FAIL:", error, file=sys.stderr)
            return 1

    print("self-test OK: 13 denominator events, 9 matched, every tier, both traps")
    return 0


# --------------------------------------------------------------------------
# entry
# --------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="transcript_join.py",
        description=(
            "Join postsession-corpus transcripts to recall_events and measure "
            "per-tier coverage of the silent share. Store is opened read-only "
            "(sqlite file: URI, mode=ro); the index must hash to the "
            "index_sha256 of whichever manifest --manifest was handed, so a "
            "freshly built self-sealed pair verifies like the sealed one. "
            "Writes coverage-<field-label>.json and "
            "match-table-<field-label>.jsonl.gz into --out-dir."
        ),
    )
    parser.add_argument("--store", help="sqlite store (a pinned snapshot on live hosts)")
    parser.add_argument(
        "--manifest",
        help="corpus manifest sealing --index (a fresh build, or corpus.json)",
    )
    parser.add_argument("--index", help="corpus-index.jsonl rebuilt for this host")
    parser.add_argument("--out-dir", help="directory for the two output artifacts")
    parser.add_argument("--field-label", help="artifact suffix, e.g. 'local' or 'alt'")
    parser.add_argument(
        "--sealed-manifest",
        help=(
            "sealed artifacts/post-session/corpus.json, compared against and "
            "recorded only (never read for rows, never relaxes the gate)"
        ),
    )
    parser.add_argument(
        "--drift-report",
        help=(
            "postsession_corpus.py --verify --report JSON, summarised into "
            "coverage.drift_vs_sealed_manifest"
        ),
    )
    parser.add_argument(
        "--self-verify-report",
        help=(
            "postsession_corpus.py --verify --report JSON for the manifest "
            "handed to --manifest itself, summarised into "
            "coverage.drift_vs_own_manifest (how far the freshly built corpus "
            "moved between build and join)"
        ),
    )
    parser.add_argument(
        "--store-origin",
        help="live store path a --store snapshot was copied from, for the record",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the offline synthetic fixture suite and exit (no store needed)",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    required = ("store", "manifest", "index", "out_dir", "field_label")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error(
            "missing required arguments: "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )

    try:
        coverage = run_join(
            args.store,
            args.manifest,
            args.index,
            args.out_dir,
            args.field_label,
            sealed_manifest_path=args.sealed_manifest,
            drift_report_path=args.drift_report,
            store_origin=args.store_origin,
            self_verify_report_path=args.self_verify_report,
        )
    except (JoinError, OSError, ValueError) as error:
        print("ERROR:", error, file=sys.stderr)
        return 2

    cov = coverage["coverage"]
    print(
        "coverage {} / {} = {} ({}); by_method {}; wrote {}".format(
            cov["matched_events"],
            cov["total_events"],
            cov["share"],
            coverage["falsifier_60pct"]["verdict"],
            json.dumps(cov["by_method"], sort_keys=True),
            coverage["_coverage_path"],
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
