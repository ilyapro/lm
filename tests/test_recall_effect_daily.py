"""Tests for ``scripts/recall_effect_daily.py`` on synthetic fixture databases.

The live store is never opened here. Each test builds a small database with the
columns the script reads (recall_events, recall_credit_ledger, nodes and,
where needed, recall_feedback_marks or an explicit-credit side table) plus a
fake AE artifacts tree holding capture headers and a corpus manifest.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("recall_effect_daily", ROOT / "scripts" / "recall_effect_daily.py")
red = importlib.util.module_from_spec(SPEC)
sys.modules["recall_effect_daily"] = red
SPEC.loader.exec_module(red)

LEDGER_CHECKED = """
CREATE TABLE recall_credit_ledger (
    recall_event_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    basis TEXT NOT NULL CHECK (basis IN ('grounded', 'lookup')),
    source_id TEXT NOT NULL,
    credited_at TEXT NOT NULL,
    UNIQUE(recall_event_id, node_id)
)"""
LEDGER_OPEN = LEDGER_CHECKED.replace(" CHECK (basis IN ('grounded', 'lookup'))", "")


def make_db(path: Path, ledger_sql: str = LEDGER_CHECKED) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(
        f"""
        CREATE TABLE nodes (id TEXT PRIMARY KEY, content TEXT, context TEXT, created_at TEXT);
        CREATE TABLE recall_events (
            id TEXT PRIMARY KEY, query TEXT NOT NULL, scope TEXT NOT NULL DEFAULT 'global',
            requested_scope TEXT NOT NULL DEFAULT 'global', resolved_scopes TEXT DEFAULT '[]',
            ambient_context TEXT NOT NULL DEFAULT '{{}}', results TEXT NOT NULL DEFAULT '[]',
            agent TEXT, task TEXT, session_id TEXT,
            feedback_applied INTEGER NOT NULL DEFAULT 0, feedback_trace_id TEXT,
            feedback_applied_at TEXT, created_at TEXT NOT NULL, transport_session_id TEXT
        );
        {ledger_sql};
        """
    )
    return conn


def add_event(conn, eid, session, created_at, nodes, *, query="ordinary live query", scope="global",
              closed=False, task=None, ambient=None):
    results = [{"node_id": n, "rank": i + 1} for i, n in enumerate(nodes)]
    amb = dict(ambient or {})
    if session:
        amb.setdefault("transport_session_id", session)
    conn.execute(
        "INSERT INTO recall_events (id, query, scope, requested_scope, ambient_context, results, task,"
        " feedback_applied, created_at, transport_session_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (eid, query, scope, scope, json.dumps(amb), json.dumps(results), task, int(closed), created_at, session),
    )


def credit(conn, eid, node, basis):
    conn.execute(
        "INSERT INTO recall_credit_ledger VALUES (?,?,?,?,?)", (eid, node, basis, "src", "2026-09-23T00:00:00Z")
    )


def add_node(conn, nid, session, created_at, content="closing note", **ctx):
    ctx["transport_session_id"] = session
    conn.execute("INSERT INTO nodes VALUES (?,?,?,?)", (nid, content, json.dumps(ctx), created_at))


def make_receipts(root: Path, session_ids, families=("development-packaging", "ev-recovery")) -> Path:
    run = root / "tree-context-ab" / "live-20260923T122922Z" / "baseline"
    run.mkdir(parents=True)
    cases = [{"case_id": f"x/{f}", "goal_path": f"{f}/child", "split": "development"} for f in families]
    cases.append({"case_id": "audit/real", "goal_path": "real-live-tree", "split": "audit"})
    (run / "corpus_manifest.json").write_text(json.dumps({"cases": cases}))
    for i, sid in enumerate(session_ids):
        cap = run / "runs" / f"case--r{i}" / "capture-memory-abc"
        cap.mkdir(parents=True)
        (cap / "000001.headers.json").write_text(
            json.dumps({"request": {"Host": "127.0.0.1"}, "response": {"mcp-session-id": sid}})
        )
    return root


def report(db: Path, receipts_root: Path | None, since="2026-09-22", until="2026-09-24", **kw):
    ids, fams = red.scan_receipts([receipts_root] if receipts_root else [])
    return red.build_report(str(db), "test", since, until, ids, fams, **kw)


def reasons_by_event(rep):
    out = {}
    for x in rep["exclusion"]:
        for s in x["sessions"]:
            out[s["session"]] = set(s["reasons"])
    return out


# ------------------------------------------------------------------ exclusion


def test_receipt_catches_23_09_style_leak_the_legacy_filter_misses(tmp_path):
    db = tmp_path / "lm.sqlite3"
    conn = make_db(db)
    # A/B session: scope global, fixture-worded query without legacy keywords.
    add_event(conn, "ab1", "S_AB", "2026-09-23T13:57:47Z", ["n1", "n2"],
              query="development-packaging routing accepted kits route only")
    add_event(conn, "ab2", "S_AB", "2026-09-23T13:57:48Z", ["n1", "n3"],
              query="kit inputs constraints tubes coolant seals labels")
    credit(conn, "ab1", "n1", "grounded")
    credit(conn, "ab2", "n1", "grounded")
    add_event(conn, "live1", "S_LIVE", "2026-09-23T15:00:00Z", ["n4", "n5"], query="lm storage migration")
    conn.commit()
    conn.close()
    rep = report(db, make_receipts(tmp_path / "ae", ["S_AB"]))

    day = next(d for d in rep["days"] if d["day"] == "2026-09-23")
    assert day["excluded_events"] == 2 and day["kept_events"] == 1
    assert day["metrics"]["used"]["rank1_used"] == 0.0
    # The legacy filter keeps the A/B events, so its numbers are inflated.
    assert day["legacy_filter"]["kept_events"] == 3
    assert day["legacy_filter"]["used"]["useful_node_share"] == pytest.approx(2 / 6, abs=1e-4)
    assert rep["leak_check"]["events_excluded_new_but_not_legacy"] == 2
    assert rep["leak_check"]["by_reason"] == {"receipt": 2}
    assert "receipt" in reasons_by_event(rep)["S_AB"]


def test_exclusion_is_session_wide_across_days(tmp_path):
    db = tmp_path / "lm.sqlite3"
    conn = make_db(db)
    add_event(conn, "a", "S1", "2026-09-22T23:59:00Z", ["n1"])
    add_event(conn, "b", "S1", "2026-09-23T00:01:00Z", ["n1"])
    add_node(conn, "c1", "S1", "2026-09-23T00:02:00Z", fixture="tree-context baseline kit-acceptance")
    conn.commit()
    conn.close()
    rep = report(db, None)
    assert [d["excluded_events"] for d in rep["days"]] == [1, 1, 0]
    assert rep["exclusion"][0]["sessions"][0]["reasons"]["fixture_provenance"].startswith("c1 context.fixture")


@pytest.mark.parametrize(
    "content,ctx",
    [
        ("done", {"run": "development_kit-acceptance--claude_stream-json--live--r3--r3"}),
        ("done", {"run": "live-c-r3"}),
        ("done", {"project": "target"}),
        ("routing (AE fixture kit-acceptance, project target) closed", {}),
    ],
)
def test_fixture_provenance_markers(tmp_path, content, ctx):
    db = tmp_path / "lm.sqlite3"
    conn = make_db(db)
    add_event(conn, "e", "S1", "2026-09-23T10:00:00Z", ["n1"])
    add_node(conn, "c", "S1", "2026-09-23T10:05:00Z", content=content, **ctx)
    conn.commit()
    conn.close()
    assert reasons_by_event(report(db, None))["S1"] == {"fixture_provenance"}


def test_fixture_task_family_comes_from_manifest_not_audit_split(tmp_path):
    db = tmp_path / "lm.sqlite3"
    conn = make_db(db)
    add_event(conn, "e1", "S_FIX", "2026-09-23T10:00:00Z", ["n1"], task="ev-recovery/repost")
    add_event(conn, "e2", "S_AUDIT", "2026-09-23T10:00:00Z", ["n1"], task="real-live-tree/child")
    add_event(conn, "e3", "S_NODE", "2026-09-23T10:00:00Z", ["n1"])
    add_node(conn, "c3", "S_NODE", "2026-09-23T10:01:00Z", node_path="development-packaging/routing")
    conn.commit()
    conn.close()
    rep = report(db, make_receipts(tmp_path / "ae", []))
    reasons = reasons_by_event(rep)
    assert reasons["S_FIX"] == {"fixture_task"}
    assert reasons["S_NODE"] == {"fixture_task"}
    assert "S_AUDIT" not in reasons
    assert "real-live-tree" not in rep["sources"]["fixture_task_families"]


def test_project_target_counts_only_near_proven_harness_traffic(tmp_path):
    db = tmp_path / "lm.sqlite3"
    conn = make_db(db)
    add_event(conn, "p", "S_PROVEN", "2026-09-23T12:00:00Z", ["n1"])
    add_event(conn, "near", "S_NEAR", "2026-09-23T12:20:00Z", ["n1"], scope="project:target")
    add_event(conn, "far", "S_FAR", "2026-09-23T18:00:00Z", ["n1"], scope="project:target",
              query="ae supervisor active goals")
    conn.commit()
    conn.close()
    rep = report(db, make_receipts(tmp_path / "ae", ["S_PROVEN"]))
    reasons = reasons_by_event(rep)
    assert reasons["S_NEAR"] == {"fixture_scope"}
    assert "S_FAR" not in reasons


def test_synthetic_scope_and_never_closed_fixture_queries(tmp_path):
    db = tmp_path / "lm.sqlite3"
    conn = make_db(db)
    add_event(conn, "c", "S_CHAT", "2026-09-23T10:00:00Z", ["n1"], scope="project:_chat_inject_live_e2e_1")
    add_event(conn, "q1", "S_Q", "2026-09-23T10:00:00Z", ["n1"], query="ev-recovery repost shipment journal")
    add_event(conn, "m1", "S_MIX", "2026-09-23T10:00:00Z", ["n1"], query="ev-recovery bench defect")
    add_event(conn, "m2", "S_MIX", "2026-09-23T10:01:00Z", ["n1"], query="codex websocket recorder proxy")
    add_event(conn, "w", "S_WORD", "2026-09-23T10:00:00Z", ["n1"], query="xev-recovery-ish token")
    conn.commit()
    conn.close()
    reasons = reasons_by_event(report(db, make_receipts(tmp_path / "ae", [])))
    assert reasons["S_CHAT"] == {"synthetic_scope"}
    assert reasons["S_Q"] == {"fixture_query_family"}
    assert "S_MIX" not in reasons  # a session doing other work too is live
    assert "S_WORD" not in reasons  # whole-token match only


def test_live_project_x_is_kept_and_reported_as_legacy_overexclusion(tmp_path):
    db = tmp_path / "lm.sqlite3"
    conn = make_db(db)
    add_event(conn, "x1", "S_X", "2026-09-23T10:00:00Z", ["n1"], scope="project:x", query="gpu kernel profile")
    add_event(conn, "l1", "S_L", "2026-09-23T10:00:00Z", ["n1"], query="credit ledger basis pitfall")
    conn.commit()
    conn.close()
    rep = report(db, None)
    day = next(x for x in rep["exclusion"] if x["day"] == "2026-09-23")
    assert day["excluded_events"] == 0
    assert day["legacy_would_exclude"] == 2
    assert day["legacy_overexclude_events"] == 2


# --------------------------------------------------------------- per-day math


def test_per_day_math_closed_split_and_per_basis(tmp_path):
    db = tmp_path / "lm.sqlite3"
    conn = make_db(db)
    d = "2026-09-23T10:00:0{}Z"
    add_event(conn, "e1", "S1", d.format(1), ["a", "b", "c", "d"], closed=True)  # rank1 grounded
    add_event(conn, "e2", "S2", d.format(2), ["a", "b", "c", "d"], closed=True)  # rank3 lookup
    add_event(conn, "e3", "S3", d.format(3), ["a", "b", "c", "d"])  # rank4 grounded: not top3
    add_event(conn, "e4", "S4", d.format(4), ["a", "b"])  # nothing used
    add_event(conn, "e5", "S5", d.format(5), [])  # no results: not in the denominator
    credit(conn, "e1", "a", "grounded")
    credit(conn, "e1", "b", "lookup")
    credit(conn, "e2", "c", "lookup")
    credit(conn, "e3", "d", "grounded")
    credit(conn, "e9", "a", "grounded")  # event outside the window
    conn.commit()
    conn.close()
    rep = report(db, None, since="2026-09-23", until="2026-09-23")
    m = rep["days"][0]["metrics"]
    assert m["events"] == 5 and m["closed_events"] == 2 and m["never_closed_events"] == 3
    u = m["used"]
    assert u["events_with_results"] == 4
    assert u["rank1_used"] == 0.25
    assert u["top3_used"] == 0.5
    assert u["delivered_nodes"] == 14 and u["used_delivered_nodes"] == 4
    assert u["useful_node_share"] == round(4 / 14, 4)
    g = m["per_basis"]["grounded"]
    assert (g["rank1_used_events"], g["top3_used_events"], g["used_delivered_nodes"]) == (1, 1, 2)
    lk = m["per_basis"]["lookup"]
    assert (lk["rank1_used_events"], lk["top3_used_events"], lk["used_delivered_nodes"]) == (0, 2, 2)
    closed = rep["days"][0]["closed"]["used"]
    never = rep["days"][0]["never_closed"]["used"]
    assert (closed["events_with_results"], closed["rank1_used"], closed["top3_used"]) == (2, 0.5, 1.0)
    assert (never["events_with_results"], never["rank1_used"], never["top3_used"]) == (2, 0.0, 0.0)
    assert m["per_basis"]["explicit"]["used_delivered_nodes"] == 0


def test_duplicate_delivered_node_counts_once_and_rank_order_is_respected(tmp_path):
    db = tmp_path / "lm.sqlite3"
    conn = make_db(db)
    results = [{"node_id": "b", "rank": 2}, {"node_id": "a", "rank": 1}, {"node_id": "b", "rank": 3}]
    conn.execute(
        "INSERT INTO recall_events (id, query, results, created_at, transport_session_id) VALUES (?,?,?,?,?)",
        ("e", "q", json.dumps(results), "2026-09-23T10:00:00Z", "S"),
    )
    credit(conn, "e", "a", "grounded")
    conn.commit()
    conn.close()
    u = report(db, None, since="2026-09-23", until="2026-09-23")["days"][0]["metrics"]["used"]
    assert (u["rank1_used"], u["delivered_nodes"], u["used_delivered_nodes"]) == (1.0, 2, 1)


# ----------------------------------------------------------- explicit handling


def test_explicit_ledger_basis_is_reported_but_not_in_used_by_default(tmp_path):
    db = tmp_path / "lm.sqlite3"
    conn = make_db(db, LEDGER_OPEN)
    add_event(conn, "e1", "S1", "2026-09-23T10:00:00Z", ["a", "b"])
    add_event(conn, "e2", "S2", "2026-09-23T10:00:01Z", ["a", "b"])
    credit(conn, "e1", "a", "explicit")
    credit(conn, "e2", "b", "grounded")
    conn.commit()
    conn.close()
    rep = report(db, None, since="2026-09-23", until="2026-09-23")
    m = rep["days"][0]["metrics"]
    assert rep["sources"]["explicit_present"] is True
    assert "explicit" in rep["sources"]["ledger_bases"]
    assert m["used"]["rank1_used"] == 0.0
    assert m["per_basis"]["explicit"]["rank1_used"] == 0.5
    assert m["used_with_explicit"]["rank1_used"] == 0.5
    assert m["used_with_explicit"]["used_delivered_nodes"] == 2
    rep2 = report(db, None, since="2026-09-23", until="2026-09-23", used_bases=("grounded", "lookup", "explicit"))
    assert rep2["days"][0]["metrics"]["used"]["used_delivered_nodes"] == 2


def test_explicit_side_table_and_feedback_marks(tmp_path):
    db = tmp_path / "lm.sqlite3"
    conn = make_db(db)
    conn.executescript(
        """
        CREATE TABLE recall_explicit_credit (recall_event_id TEXT, node_id TEXT, credited_at TEXT);
        CREATE TABLE recall_feedback_marks (
            id INTEGER PRIMARY KEY, recall_event_id TEXT, node_id TEXT, mark TEXT, accepted INTEGER,
            reject_reason TEXT, via_tool TEXT, source_id TEXT, transport_session_id TEXT, agent TEXT,
            rank INTEGER, marked_at TEXT
        );
        """
    )
    add_event(conn, "e1", "S1", "2026-09-23T10:00:00Z", ["a", "b", "c"])
    add_event(conn, "e2", "S2", "2026-09-23T10:00:01Z", ["a", "b", "c"])
    conn.execute("INSERT INTO recall_explicit_credit VALUES ('e1', 'b', 't')")
    marks = [
        ("e1", "b", "used", 1, "memory_remember", 1),
        ("e1", "c", "irrelevant", 1, "memory_remember", 2),
        ("e2", "a", "used", 0, "memory_recall", 0),  # rejected: not delivered to that session
    ]
    for eid, node, mark, acc, tool, rank in marks:
        conn.execute(
            "INSERT INTO recall_feedback_marks (recall_event_id, node_id, mark, accepted, via_tool,"
            " transport_session_id, agent, rank, marked_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (eid, node, mark, acc, tool, "S1", "claude", rank, "2026-09-23T10:05:00Z"),
        )
    conn.commit()
    conn.close()
    rep = report(db, None, since="2026-09-23", until="2026-09-23")
    day = rep["days"][0]
    assert rep["sources"]["recall_feedback_marks_present"] is True
    assert day["metrics"]["per_basis"]["explicit"]["top3_used_events"] == 1
    mu = day["metrics"]["marked_used"]
    assert (mu["rank1_used_events"], mu["top3_used_events"], mu["used_delivered_nodes"]) == (0, 1, 1)
    assert day["marks"]["marks"] == {"irrelevant:accepted": 1, "used:accepted": 1, "used:rejected": 1}
    assert day["marks"]["events_with_accepted_mark"] == 1
    assert day["marks"]["events_with_accepted_mark_share"] == 0.5
    assert "## Explicit marks" in red.render_md(rep)


def test_no_marks_table_means_no_marks_section(tmp_path):
    db = tmp_path / "lm.sqlite3"
    conn = make_db(db)
    add_event(conn, "e1", "S1", "2026-09-23T10:00:00Z", ["a"])
    conn.commit()
    conn.close()
    rep = report(db, None)
    assert rep["sources"]["recall_feedback_marks_present"] is False
    assert "marked_used" not in rep["days"][1]["metrics"]
    assert "## Explicit marks" not in red.render_md(rep)


# ------------------------------------------------------------- IO contract


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_reads_read_only_and_writes_json_and_md(tmp_path):
    db = tmp_path / "lm.sqlite3"
    conn = make_db(db)
    add_event(conn, "ab", "S_AB", "2026-09-23T13:57:47Z", ["a"], query="development-packaging routing")
    add_event(conn, "lv", "S_LV", "2026-09-23T14:00:00Z", ["a"])
    credit(conn, "lv", "a", "grounded")
    conn.commit()
    conn.close()
    before = _sha(db)
    ae = make_receipts(tmp_path / "ae", ["S_AB"])
    out_json, out_md = tmp_path / "out" / "r.json", tmp_path / "out" / "r.md"
    rc = red.main([
        "--db", str(db), "--host-label", "fx", "--since", "2026-09-22", "--until", "2026-09-24",
        "--receipts-root", str(ae), "--json", str(out_json), "--md", str(out_md),
    ])
    assert rc == 0
    assert _sha(db) == before
    assert not Path(str(db) + "-wal").exists()
    data = json.loads(out_json.read_text())
    assert data["opened_as"] == f"file:{db}?mode=ro"
    assert data["totals"]["excluded_events"] == 1
    assert data["sources"]["receipt_roots"] == [str(ae)]
    md = out_md.read_text()
    assert "2026-09-23" in md and "Leak check 2026-09-23" in md and "`S_AB`" in md


def test_open_ro_refuses_writes(tmp_path):
    db = tmp_path / "lm.sqlite3"
    make_db(db).close()
    conn = red.open_ro(db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("CREATE TABLE t (x)")


def test_snapshot_to_reads_a_backup_copy(tmp_path):
    db = tmp_path / "lm.sqlite3"
    conn = make_db(db)
    add_event(conn, "e", "S", "2026-09-23T10:00:00Z", ["a"])
    conn.commit()
    conn.close()
    snap = tmp_path / "snap" / "copy.sqlite3"
    out = tmp_path / "r.json"
    red.main([
        "--db", str(db), "--host-label", "fx", "--since", "2026-09-23", "--until", "2026-09-23",
        "--no-default-receipts", "--snapshot-to", str(snap), "--json", str(out),
    ])
    data = json.loads(out.read_text())
    assert snap.exists() and data["db"] == str(snap)
    assert data["opened_as"].startswith("snapshot")
    assert data["totals"]["all_events"] == 1


def test_receipt_ids_file(tmp_path):
    db = tmp_path / "lm.sqlite3"
    conn = make_db(db)
    add_event(conn, "e", "S_FILE", "2026-09-23T10:00:00Z", ["a"])
    conn.commit()
    conn.close()
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps(["S_FILE"]))
    out = tmp_path / "r.json"
    red.main([
        "--db", str(db), "--host-label", "fx", "--since", "2026-09-23", "--until", "2026-09-23",
        "--no-default-receipts", "--receipt-ids", str(ids), "--json", str(out),
    ])
    data = json.loads(out.read_text())
    assert data["exclusion"][0]["sessions"][0]["reasons"] == {"receipt": "receipt-ids:ids.json"}
