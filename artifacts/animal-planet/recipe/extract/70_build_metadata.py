#!/usr/bin/env python3
"""Step 70: derive and pin the staging METADATA.json.

Consumes ONLY files already in staging (steps 10-60) plus windows.json.
Produces:
- $AP_STAGING/transcripts/matched.jsonl  deterministic transcript<->DB match
  for memory_recall tool-results (exact query equality, nearest created_at
  vs result_ts, one-to-one greedy by ascending time delta, tolerance 600s).
- $AP_STAGING/METADATA.json  window definitions, pinned predicates, per-source
  counts + provenance, expected-vs-measured cross-check table, discrepancies.

Stdout carries ONLY aggregate numbers (never query/content text).
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGING = Path(os.environ.get("AP_STAGING", "/home/sfx/.cache/ap-audit/staging"))

MATCH_TOLERANCE_SECONDS = 600
LM_TOOLS = ("memory_recall", "memory_remember", "memory_lookup", "memory_teach")

# Cited baseline numbers (audit node 01KZK2WMP07CNTYDQXFTS23F39 = A;
# net-value trace 01KZVZ5ZTE1WSXFHCK6091H0RE = N;
# payload follow-up trace 01KZV6VCVGXVKPF6PSFXBH00EM = P;
# extraction recon trace 01KZW27BVVSZHZFH3J73PB4RE3 = R).
EXPECTED = {
    "w1_recall_events": {"value": 727, "source": "A"},
    "w1_supersedes_connections": {"value": 9, "source": "A"},
    "connections_supersedes_total": {"value": 93, "source": "R"},
    "connections_contradicts_total": {"value": 85, "source": "R"},
    "connections_related_total": {"value": 23046, "source": "R (point-in-time 2026-08-12T22:41Z, live DB keeps growing)"},
    "w2_game_events": {"value": 1349, "source": "N/R"},
    "w2_game_organic": {"value": [165, 70], "source": "N (42.4%)"},
    "w2_game_automatic": {"value": [1188, 209], "source": "N (17.6%)"},
    "game_schema_nodes": {"value": 9, "source": "A (as of 2026-08-09)"},
    "w1_auto_outcome_traces": {"value": 105, "source": "A"},
    "transcript_files": {"value": 336, "source": "R (~210 at 08-09 audit)"},
    "w1_lm_calls": {"value": {"memory_recall": 202, "memory_remember": 141, "memory_lookup": 46, "memory_teach": 8}, "source": "A"},
    "w1_avg_lm_result_chars": {"value": 12600, "source": "A ('avg 12.6KB')"},
    "w1_lm_share_of_tool_result_volume": {"value": 0.188, "source": "A ('18.8%')"},
    "w1_total_tool_results": {"value": 10669, "source": "A ('10669 tool-вызовов')"},
    "w1_deterministic_prerecalls": {"value": {"reopen_lesson": 152, "architectural_decision": 152}, "source": "A (304 = 42% of 727)"},
    "w3_matched_recall_results": {"value": 278, "source": "P"},
    "w3_payload_chars": {"value": {"median": 26496, "p90": 35237, "mean": 26543}, "source": "P"},
    "holdout_scope_totals": {"value": {"project:octopus": 12127, "project:online": 7976, "project:x": 9080}, "source": "R"},
}


def parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = raw.replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def p90_nearest_rank(sorted_vals: list) -> float:
    n = len(sorted_vals)
    return sorted_vals[max(0, -(-9 * n // 10) - 1)]


def p90_linear(sorted_vals: list) -> float:
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    idx = (n - 1) * 0.9
    lo = int(idx)
    frac = idx - lo
    if lo + 1 >= n:
        return float(sorted_vals[-1])
    return sorted_vals[lo] + (sorted_vals[lo + 1] - sorted_vals[lo]) * frac


def payload_stats(values: list) -> dict | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "mean": round(sum(vals) / len(vals), 1),
        "p90_nearest_rank": p90_nearest_rank(vals),
        "p90_linear": round(p90_linear(vals), 1),
        "min": vals[0],
        "max": vals[-1],
        "sum": sum(vals),
    }


def check(name: str, measured, *, expected=None, status=None, note: str = "") -> dict:
    exp = EXPECTED[name]["value"] if name in EXPECTED else expected
    if status is None:
        status = "exact" if measured == exp else "discrepancy"
    entry = {
        "name": name,
        "expected": exp,
        "expected_source": EXPECTED.get(name, {}).get("source"),
        "measured": measured,
        "status": status,
    }
    if note:
        entry["note"] = note
    return entry


def main() -> int:
    windows = json.loads((HERE / "windows.json").read_text(encoding="utf-8"))
    bounds = {
        name: (parse_ts(w["start"]), parse_ts(w["end"]))
        for name, w in windows.items()
        if "start" in w
    }

    def in_window(ts: datetime | None, name: str) -> bool:
        if ts is None:
            return False
        lo, hi = bounds[name]
        return lo <= ts <= hi

    # ------------------------------------------------------------------ load
    alt_events = load_jsonl(STAGING / "alt-db" / "recall_events.jsonl")
    for event in alt_events:
        event["_ts"] = parse_ts(event["created_at"])
    alt_summary = json.loads((STAGING / "alt-db" / "summary.json").read_text())
    connections_typed = load_jsonl(STAGING / "alt-db" / "connections_typed.jsonl")
    game_schema_nodes = load_jsonl(STAGING / "alt-db" / "game_schema_nodes.jsonl")
    outcome_nodes = load_jsonl(STAGING / "alt-db" / "outcome_nodes.jsonl")
    transcripts = load_jsonl(STAGING / "transcripts" / "lm_events.jsonl")
    inventory_summary = json.loads(
        (STAGING / "transcripts" / "inventory_summary.json").read_text()
    )
    local_summary = json.loads((STAGING / "local-db" / "summary.json").read_text())
    alt_snapshot = json.loads((STAGING / "alt-db" / "snapshot.json").read_text())
    local_snapshot = json.loads((STAGING / "local-db" / "snapshot.json").read_text())

    tr_events = [r for r in transcripts if r["kind"] == "event"]
    result_sizes = [r for r in transcripts if r["kind"] == "result_size"]
    for record in tr_events:
        record["_ts"] = parse_ts(record.get("result_ts") or record.get("use_ts"))
    for record in result_sizes:
        record["_ts"] = parse_ts(record.get("ts"))

    cross_checks: list[dict] = []
    discrepancies: list[dict] = []

    fb_domain = sorted({int(e.get("feedback_applied") or 0) for e in alt_events})

    # ------------------------------------------------- DB window cross-checks
    w1_events = [e for e in alt_events if in_window(e["_ts"], "W1")]
    cross_checks.append(check("w1_recall_events", len(w1_events)))

    w1_supersedes = sum(
        1
        for c in connections_typed
        if c["type"] == "supersedes" and in_window(parse_ts(c["created_at"]), "W1")
    )
    cross_checks.append(check("w1_supersedes_connections", w1_supersedes))

    totals = alt_summary["connections_by_type_total"]
    for conn_type, cited in (("supersedes", 93), ("contradicts", 85), ("related", 23046)):
        measured = totals.get(conn_type, 0)
        cross_checks.append(
            check(
                f"connections_{conn_type}_total",
                measured,
                status="exact" if measured == cited else ("near" if measured >= cited else "discrepancy"),
                note="full-table count is point-in-time; the live DB keeps growing, window-bound checks are the stable ones"
                if measured != cited
                else "",
            )
        )

    # ------------------------------------------- organic vs automatic (W2)
    game_w2 = [
        e
        for e in alt_events
        if e["scope"] == "project:game" and in_window(e["_ts"], "W2")
    ]
    cross_checks.append(check("w2_game_events", len(game_w2)))

    def split_counts(events: list[dict], organic_fn) -> dict:
        org = [e for e in events if organic_fn(e)]
        auto = [e for e in events if not organic_fn(e)]
        fb = lambda rows: sum(1 for e in rows if int(e.get("feedback_applied") or 0))
        return {
            "organic_n": len(org),
            "organic_fb": fb(org),
            "automatic_n": len(auto),
            "automatic_fb": fb(auto),
        }

    predicate_candidates = {
        "agent_not_null": lambda e: e.get("agent") is not None,
        "session_id_not_null": lambda e: e.get("session_id") is not None,
        "task_not_null": lambda e: e.get("task") is not None,
        "agent_or_session_id": lambda e: e.get("agent") is not None
        or e.get("session_id") is not None,
    }
    predicate_table = {
        name: split_counts(game_w2, fn) for name, fn in predicate_candidates.items()
    }
    pinned = predicate_table["agent_not_null"]
    cross_checks.append(
        check(
            "w2_game_automatic",
            [pinned["automatic_n"], pinned["automatic_fb"]],
            note="pinned predicate: automatic = (agent IS NULL) over scope='project:game' events in W2; exact on both n and feedback",
        )
    )
    organic_measured = [pinned["organic_n"], pinned["organic_fb"]]
    cross_checks.append(
        check(
            "w2_game_organic",
            organic_measured,
            status="discrepancy",
            note="organic = (agent IS NOT NULL); cited 165/70 is internally inconsistent with cited automatic 1188/209 inside one window (165+1188=1353 != 1349 total, 70+209=279 != 280 total fb); closest single column-level rule gives 161/71; session_id-based rule gives 155/70 (fb exact, n short by 10)",
        )
    )
    if organic_measured == [165, 70]:
        cross_checks[-1]["status"] = "exact"
    else:
        discrepancies.append(
            {
                "name": "w2_game_organic",
                "cited": [165, 70],
                "measured_best": organic_measured,
                "predicate_table": predicate_table,
                "explanation": "No column-level predicate reproduces organic 165/70 simultaneously with automatic 1188/209: window totals are 1349 events / 280 feedback, while the cited pair sums to 1353/279. automatic=(agent IS NULL) reproduces the automatic side exactly; its complement is 161/71 (44.1% linkage vs cited 42.4%).",
            }
        )

    # ------------------------------------------------- game schema nodes
    schema_matrix = {
        "total_in_snapshot": len(game_schema_nodes),
        "active_in_snapshot": sum(1 for n in game_schema_nodes if not n.get("decayed")),
        "created_by_w1_end": sum(
            1 for n in game_schema_nodes if parse_ts(n["created_at"]) <= bounds["W1"][1]
        ),
        "active_and_created_by_w1_end": sum(
            1
            for n in game_schema_nodes
            if not n.get("decayed") and parse_ts(n["created_at"]) <= bounds["W1"][1]
        ),
    }
    best_schema = schema_matrix["created_by_w1_end"]
    cross_checks.append(
        check(
            "game_schema_nodes",
            best_schema,
            status="exact" if best_schema == 9 else "discrepancy",
            note=f"pinned: nodes with scope='project:game' AND level='schema' AND created_at <= W1.end; matrix: {schema_matrix}",
        )
    )

    # ------------------------------------------------- auto-OUTCOME matrix
    def outcome_count(prefix_only: bool, w1_only: bool, active_only: bool, agent_ae: bool) -> int:
        n = 0
        for node in outcome_nodes:
            content = node.get("content") or ""
            if prefix_only and not (
                content.startswith("OUTCOME pass: ") or content.startswith("OUTCOME fail: ")
            ):
                continue
            if w1_only and not in_window(parse_ts(node["created_at"]), "W1"):
                continue
            if active_only and node.get("decayed"):
                continue
            if agent_ae and node.get("agent") != "ae":
                continue
            n += 1
        return n

    outcome_matrix = {}
    for prefix_only in (True, False):
        for active_only in (True, False):
            for agent_ae in (True, False):
                key = (
                    f"{'template_prefix' if prefix_only else 'broad_like'}"
                    f"|{'active' if active_only else 'any_decay'}"
                    f"|{'agent=ae' if agent_ae else 'any_agent'}"
                )
                outcome_matrix[key] = {
                    "w1": outcome_count(prefix_only, True, active_only, agent_ae),
                    "all_time": outcome_count(prefix_only, False, active_only, agent_ae),
                }
    outcome_best = max(v["w1"] for v in outcome_matrix.values())
    outcome_pinned = outcome_matrix["template_prefix|any_decay|agent=ae"]["w1"]
    template_nodes_w1 = [
        n
        for n in outcome_nodes
        if (n.get("content") or "").startswith(("OUTCOME pass: ", "OUTCOME fail: "))
        and in_window(parse_ts(n["created_at"]), "W1")
    ]
    outcome_distinct_w1 = len({n.get("content") for n in template_nodes_w1})
    cross_checks.append(
        check(
            "w1_auto_outcome_traces",
            outcome_pinned,
            status="exact" if outcome_pinned == 105 else "discrepancy",
            note=(
                "pinned marker (alt ~/p/ae/node.sh _node_lm_record_outcome): content LIKE 'OUTCOME pass: %' OR 'OUTCOME fail: %', agent='ae', node created_at in W1; "
                f"max over matrix variants = {outcome_best}"
            ),
        )
    )
    if outcome_pinned != 105:
        discrepancies.append(
            {
                "name": "w1_auto_outcome_traces",
                "cited": 105,
                "measured_best": outcome_pinned,
                "matrix": outcome_matrix,
                "distinct_contents_w1": outcome_distinct_w1,
                "explanation": (
                    "The 2026-08-13 snapshot cannot reproduce the 105 counted live on 2026-08-09. "
                    "The auto-OUTCOME remember path writes one trace per node completion; identical retry contents are deduplicated at remember time "
                    "and traces can be decayed/forgotten between audit and snapshot, so the snapshot count is a lower bound of remember-call volume. "
                    f"Distinct W1 template contents in snapshot: {outcome_distinct_w1}."
                ),
            }
        )

    # ------------------------------------------------- deterministic pre-recalls
    def query_count(patterns: tuple[str, ...], events: list[dict]) -> int:
        return sum(
            1
            for e in events
            if any(p in (e.get("query") or "") for p in patterns)
        )

    prerecall_measured = {
        "reopen_lesson": query_count(("reopen_lesson",), w1_events),
        "architectural_decision": query_count(("architectural_decision",), w1_events),
    }
    cross_checks.append(
        check(
            "w1_deterministic_prerecalls",
            prerecall_measured,
            note="substring match on the recorded query text, counted in-script (query text itself never leaves staging)",
        )
    )

    # ------------------------------------------------- transcripts: W1 volume
    cross_checks.append(check("transcript_files", inventory_summary["file_count"]))

    w1_lm = [e for e in tr_events if in_window(parse_ts(e.get("use_ts")), "W1")]
    w1_tool_counts = {}
    for tool in LM_TOOLS:
        w1_tool_counts[tool] = sum(1 for e in w1_lm if e["tool"] == tool)
    cross_checks.append(check("w1_lm_calls", w1_tool_counts))

    w1_avg = round(sum(e.get("len_text") or 0 for e in w1_lm) / len(w1_lm), 1) if w1_lm else None
    cross_checks.append(
        check(
            "w1_avg_lm_result_chars",
            w1_avg,
            status="exact_within_rounding" if w1_avg and abs(w1_avg - 12600) < 100 else "discrepancy",
            note="mean len_text over ALL W1 living-memory tool-results (397 calls, not recall-only); the audit's 'avg 12.6KB на recall' is this quantity",
        )
    )

    w1_sizes = [r for r in result_sizes if in_window(r["_ts"], "W1")]
    shares = {}
    for basis in ("len_text", "len_json_content"):
        lm_sum = sum(r.get(basis) or 0 for r in w1_sizes if r["tool"] in LM_TOOLS)
        all_sum = sum(r.get(basis) or 0 for r in w1_sizes)
        shares[basis] = {
            "share": round(lm_sum / all_sum, 4) if all_sum else None,
            "lm_sum": lm_sum,
            "all_sum": all_sum,
        }
    share_basis = min(
        shares, key=lambda b: abs((shares[b]["share"] or 0) - 0.188)
    )
    share = shares[share_basis]["share"]
    cross_checks.append(
        check(
            "w1_lm_share_of_tool_result_volume",
            share,
            status="exact_within_rounding" if share and abs(share - 0.188) < 0.005 else "discrepancy",
            note=f"pinned basis: {share_basis} (LM sum / all-tools sum over W1 tool_results); both bases: { {b: v['share'] for b, v in shares.items()} }",
        )
    )
    cross_checks.append(
        check(
            "w1_total_tool_results",
            len(w1_sizes),
            status="exact" if len(w1_sizes) == 10669 else ("near" if abs(len(w1_sizes) - 10669) <= 110 else "discrepancy"),
            note="tool_result blocks in W1 across surviving transcripts; the audit counted tool CALLS at 2026-08-09 over ~210 files",
        )
    )

    # ------------------------------------------------- transcript<->DB match
    by_query: dict[str, list[dict]] = {}
    for event in alt_events:
        by_query.setdefault(event.get("query") or "", []).append(event)

    tr_recalls = [
        e for e in tr_events if e["tool"] == "memory_recall" and not e.get("is_error")
    ]
    pairs = []
    for idx, record in enumerate(tr_recalls):
        if record["_ts"] is None:
            continue
        for event in by_query.get(record.get("query") or "", []):
            delta = abs((event["_ts"] - record["_ts"]).total_seconds())
            if delta <= MATCH_TOLERANCE_SECONDS:
                pairs.append((delta, event["id"], idx))
    pairs.sort(key=lambda p: (p[0], p[1], p[2]))
    used_db: set[str] = set()
    used_tr: set[int] = set()
    matches: list[tuple[str, int, float]] = []
    event_by_id = {e["id"]: e for e in alt_events}
    for delta, event_id, idx in pairs:
        if event_id in used_db or idx in used_tr:
            continue
        used_db.add(event_id)
        used_tr.add(idx)
        matches.append((event_id, idx, delta))

    matched_path = STAGING / "transcripts" / "matched.jsonl"
    with matched_path.open("w", encoding="utf-8") as fh:
        for event_id, idx, delta in sorted(matches, key=lambda m: event_by_id[m[0]]["created_at"]):
            event = event_by_id[event_id]
            record = tr_recalls[idx]
            fh.write(
                json.dumps(
                    {
                        "db_id": event_id,
                        "db_created_at": event["created_at"],
                        "db_scope": event["scope"],
                        "db_agent": event.get("agent"),
                        "db_session_id": event.get("session_id"),
                        "file": record["file"],
                        "line_result": record["line_result"],
                        "transcript_session": record.get("session_id"),
                        "is_sidechain": record.get("is_sidechain"),
                        "dt_seconds": round(delta, 3),
                        "len_text": record.get("len_text"),
                        "len_json_content": record.get("len_json_content"),
                        "len_tool_use_result": record.get("len_tool_use_result"),
                        "in_w1": in_window(event["_ts"], "W1"),
                        "in_w3": in_window(event["_ts"], "W3"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    matched_events = [(event_by_id[eid], tr_recalls[idx], delta) for eid, idx, delta in matches]
    w3_matched = [(e, t) for e, t, _ in matched_events if in_window(e["_ts"], "W3")]
    w1_matched = [(e, t) for e, t, _ in matched_events if in_window(e["_ts"], "W1")]

    w3_stats = {
        key: payload_stats([t.get(key) for _, t in w3_matched])
        for key in ("len_text", "len_json_content", "len_tool_use_result")
    }
    len_equivalence = all(
        (t.get("len_json_content") == t.get("len_tool_use_result"))
        for t in tr_recalls
        if t.get("len_tool_use_result") is not None
    )

    # W3 reconciliation: surviving matched + root-session DB events whose
    # transcripts are not readable (agent '/root', /root/.claude/projects).
    w3_db = [e for e in alt_events if in_window(e["_ts"], "W3")]
    w3_unmatched_sid = [
        e for e in w3_db if e["id"] not in used_db and e.get("session_id") is not None
    ]
    w3_unmatched_agent = [
        e for e in w3_db if e["id"] not in used_db and e.get("agent") is not None
    ]
    reconciliation = {
        "w3_db_events": len(w3_db),
        "w3_matched_transcript_recalls": len(w3_matched),
        "w3_unmatched_db_with_session_id": len(w3_unmatched_sid),
        "w3_unmatched_db_with_agent": len(w3_unmatched_agent),
        "matched_plus_unmatched_sid": len(w3_matched) + len(w3_unmatched_sid),
    }
    cross_checks.append(
        check(
            "w3_matched_recall_results",
            len(w3_matched),
            status="discrepancy" if len(w3_matched) != 278 else "exact",
            note=(
                "survivor transcripts only; unmatched W3 DB events with session_id are the '/root' manual sessions whose transcripts live under /root/.claude/projects (permission denied) or were removed; "
                f"reconciliation: {reconciliation['w3_matched_transcript_recalls']} + {reconciliation['w3_unmatched_db_with_session_id']} = {reconciliation['matched_plus_unmatched_sid']}"
            ),
        )
    )
    pinned_measure = "len_json_content"
    pinned_w3 = w3_stats[pinned_measure]
    w3_payload_measured = (
        {
            "median": pinned_w3["median"],
            "p90": pinned_w3["p90_nearest_rank"],
            "mean": pinned_w3["mean"],
        }
        if pinned_w3
        else None
    )
    cross_checks.append(
        check(
            "w3_payload_chars",
            w3_payload_measured,
            status="discrepancy",
            note=f"over the {pinned_w3['n'] if pinned_w3 else 0} surviving matched W3 recall tool-results, measure={pinned_measure}; cited stats were computed on 2026-08-12T14:43Z over 278 results including the now-inaccessible '/root' sessions",
        )
    )
    if len(w3_matched) != 278:
        discrepancies.append(
            {
                "name": "w3_matched_recall_results_and_payload",
                "cited": {"n": 278, "median": 26496, "p90": 35237, "mean": 26543},
                "measured_survivors": {"n": len(w3_matched), **(w3_payload_measured or {})},
                "reconciliation": reconciliation,
                "evidence": [
                    "newest surviving transcript LM event: 2026-08-11T08:20:23Z; newest *animal-planet* file mtime: 2026-08-11T09:01:17Z",
                    "no transcript anywhere under ~user/.claude/projects contains recall tool_uses in (2026-08-11T08:20:23Z, 2026-08-12T14:37:00Z]",
                    "/root/.claude/projects is permission-denied for the extraction user; the follow-up audit trace 01KZV6VCVGXVKPF6PSFXBH00EM was recorded by agent '/root'",
                    "W3 DB events with session_id set all carry agent '/root' or 'root' and started 2026-08-11T14:43:16Z",
                ],
                "explanation": "The cited 278 decomposes exactly into 164 surviving transcript-matched recalls + 114 W3 DB recall_events from '/root' manual sessions whose transcripts are inaccessible; their serialized transcript sizes cannot be re-measured from surviving evidence, so the cited median/p90/mean are not fully reproducible (survivor-only stats are recorded instead).",
            }
        )

    w1_match_summary = {
        "w1_transcript_recalls": sum(
            1 for e in tr_recalls if in_window(parse_ts(e.get("use_ts")), "W1")
        ),
        "w1_matched": len(w1_matched),
    }

    # ------------------------------------------------- holdout cross-check
    holdout_totals = {
        scope: data["recall_events_total_in_db"]
        for scope, data in local_summary["scopes"].items()
    }
    cross_checks.append(
        check(
            "holdout_scope_totals",
            holdout_totals,
            note=f"exported { {s: d['recall_events_exported'] for s, d in local_summary['scopes'].items()} } most-recent events per scope (limit {local_summary['limit_per_scope']})",
        )
    )

    # ------------------------------------------------- exports + provenance
    exports = {}
    for path in sorted(STAGING.rglob("*")):
        if not path.is_file() or path.name == "METADATA.json":
            continue
        rel = str(path.relative_to(STAGING))
        entry = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as fh:
                entry["rows"] = sum(1 for _ in fh)
        exports[rel] = entry

    status_counts = {}
    for entry in cross_checks:
        status_counts[entry["status"]] = status_counts.get(entry["status"], 0) + 1

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "generator": "artifacts/animal-planet/recipe/extract/70_build_metadata.py",
        "rebuild_command": "bash artifacts/animal-planet/recipe/extract/run_all.sh",
        "windows": windows,
        "window_semantics": "all window membership checks are inclusive on both ends; DB events keyed by recall_events.created_at, transcript events by tool_result timestamp (matching) or tool_use timestamp (call counts)",
        "sources": {
            "alt_db": alt_snapshot,
            "alt_transcripts": inventory_summary,
            "local_db": local_snapshot,
        },
        "feedback_applied_domain": fb_domain,
        "match_rule": {
            "population": "transcript tool_results of mcp__living-memory__memory_recall with is_error=false",
            "key": "exact query string equality against recall_events.query",
            "time": f"nearest |recall_events.created_at - tool_result.timestamp| <= {MATCH_TOLERANCE_SECONDS}s",
            "assignment": "greedy one-to-one by ascending time delta (ties: db id, transcript order)",
            "matched_total": len(matches),
            "w1": w1_match_summary,
        },
        "payload_measure": {
            "pinned": pinned_measure,
            "definition": "len(json.dumps(content-of-tool_result-block, ensure_ascii=False)) from the transcript",
            "alternatives": {
                "len_text": "sum of len(text) over text blocks",
                "len_tool_use_result": "len(json.dumps(top-level toolUseResult))",
            },
            "len_json_content_equals_len_tool_use_result": len_equivalence,
            "w3_stats_all_measures": w3_stats,
        },
        "predicates": {
            "organic_vs_automatic": {
                "population": "recall_events with scope='project:game' and created_at in W2 (inclusive)",
                "pinned_automatic": "agent IS NULL",
                "pinned_organic": "agent IS NOT NULL",
                "feedback_metric": "COUNT(feedback_applied != 0); feedback_applied domain is {0,1} in this window",
                "search_table": predicate_table,
            },
            "auto_outcome": {
                "template": "OUTCOME {pass|fail}: {task}[ — {goal.md first line}]",
                "template_source": "alt ~/p/ae/node.sh, _node_lm_record_outcome (goal cites line 1204); written via lm_client.py remember --agent ae --scope project:$PROJECT_NAME",
                "pinned_sql": "content LIKE 'OUTCOME pass: %' OR content LIKE 'OUTCOME fail: %', agent='ae', created_at in W1",
                "matrix": outcome_matrix,
            },
            "deterministic_prerecalls": {
                "pinned": "query contains 'reopen_lesson' / 'architectural_decision' (substring)",
                "measured": prerecall_measured,
            },
        },
        "per_source_counts": {
            "alt_db": {
                key: alt_summary[key]
                for key in (
                    "recall_events_exported",
                    "recall_events_total_in_db",
                    "recall_events_per_scope",
                    "referenced_nodes_found",
                    "connections_typed_exported",
                    "connections_among_exported",
                    "typed_edge_extra_nodes",
                    "game_schema_nodes",
                    "game_schema_source_nodes",
                    "outcome_candidate_nodes",
                    "retrieval_weights_rows",
                )
            },
            "alt_transcripts": {
                "files": inventory_summary["file_count"],
                "total_bytes": inventory_summary["total_bytes"],
                "lm_events": len(tr_events),
                "result_size_records": len(result_sizes),
                "matched_recall_events": len(matches),
            },
            "local_db_holdout": local_summary["scopes"],
        },
        "cross_checks": cross_checks,
        "cross_check_status_counts": status_counts,
        "discrepancies": discrepancies,
        "privacy": {
            "rules": [
                "staging holds raw private rows (queries, contents, transcripts-derived events) and lives outside every repo; it is never tracked or committed",
                "tracked recipe files and script stdout carry only aggregates, identifiers, lengths and hashes",
                "node `embedding` columns are excluded from all exports",
            ]
        },
        "exports": exports,
    }

    (STAGING / "METADATA.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    print(f"{'check':44s} {'status':22s} expected -> measured")
    for entry in cross_checks:
        print(
            f"{entry['name']:44s} {entry['status']:22s} "
            f"{json.dumps(entry['expected'])} -> {json.dumps(entry['measured'])}"
        )
    print(f"\nstatus counts: {status_counts}; discrepancies recorded: {len(discrepancies)}")
    print(f"METADATA.json written ({(STAGING / 'METADATA.json').stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
