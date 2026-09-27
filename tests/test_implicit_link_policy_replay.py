"""Tests for scripts/implicit_link_policy_replay.py (fixture DBs only)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from living_memory.storage import MemoryStore

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "implicit_link_policy_replay.py"
CUTOFF = "2026-09-20T00:00:00Z"
SCOPE = "project:fixture-lp"


def _load_module():
    spec = importlib.util.spec_from_file_location("implicit_link_policy_replay", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lp = _load_module()

GROUNDED_WORDS = "kestrel manifold quasar ferrite obsidian lattice turbine zephyr"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _node(store: MemoryStore, content: str, created_at: str, **context) -> str:
    node = store.create_node(level="trace", content=content, context={"scope": SCOPE, **context})
    store._conn.execute(
        "UPDATE nodes SET created_at = ?, updated_at = ? WHERE id = ?",
        (created_at, created_at, node.id),
    )
    store._conn.commit()
    return node.id


def _event(
    store: MemoryStore,
    query: str,
    created_at: str,
    node_ids: list[str],
    *,
    closed_by: str | None = None,
    session: str | None = None,
) -> str:
    results = [
        {"node_id": nid, "rank": i + 1, "bm25_score": 1.0, "methods": ["bm25"]}
        for i, nid in enumerate(node_ids)
    ]
    event = store.record_recall_event(query=query, scope=SCOPE, results=results)
    store._conn.execute(
        "UPDATE recall_events SET created_at = ?, feedback_trace_id = ?, transport_session_id = ?, "
        "requested_scope = ? WHERE id = ?",
        (created_at, closed_by, session, SCOPE, event.id),
    )
    store._conn.commit()
    return event.id


def _edge(conn: sqlite3.Connection, edge_id: str, src: str, dst: str, created_at: str, **meta) -> None:
    conn.execute(
        "INSERT INTO connections (id, source_id, target_id, type, weight, metadata, created_at, updated_at) "
        "VALUES (?, ?, ?, 'related', 0.5, ?, ?, ?)",
        (edge_id, src, dst, json.dumps(meta), created_at, created_at),
    )


def _ledger(conn: sqlite3.Connection, event_id: str, node_id: str, basis: str, at: str, source: str) -> None:
    conn.execute(
        "INSERT INTO recall_credit_ledger (recall_event_id, node_id, basis, source_id, credited_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (event_id, node_id, basis, source, at),
    )


@pytest.fixture()
def fixture_db(tmp_path: Path) -> dict:
    db = tmp_path / "source.sqlite3"
    ids: dict[str, str] = {}
    with MemoryStore(db) as store:
        # Delivered nodes.
        ids["a"] = _node(store, f"note on {GROUNDED_WORDS} tuning", "2026-08-01T00:00:00Z")
        ids["b"] = _node(store, "unrelated cooking recipe with basil and tomatoes", "2026-08-01T00:00:00Z")
        ids["c"] = _node(store, "gardening schedule for spring tulips", "2026-08-01T00:00:00Z")
        # Pre-ledger closing trace: grounded on a, not on b.
        ids["s1"] = _node(store, f"closing work: {GROUNDED_WORDS} verified", "2026-08-10T00:00:00Z")
        ids["e1"] = _event(store, "turbine lattice", "2026-08-09T23:59:00Z", [ids["a"], ids["b"]], closed_by=ids["s1"])
        # Ledger-era closing trace.
        ids["s2"] = _node(store, "closing work two", "2026-09-10T00:00:00Z")
        ids["e2"] = _event(
            store, "second query", "2026-09-09T23:59:00Z", [ids["a"], ids["b"], ids["c"]], closed_by=ids["s2"]
        )
        # Post-cutoff closing trace and eval event.
        ids["s3"] = _node(store, "closing work three", "2026-09-22T00:00:00Z")
        ids["e3"] = _event(
            store, "kestrel manifold", "2026-09-21T23:59:00Z", [ids["a"], ids["b"]], closed_by=ids["s3"], session="combat1"
        )
        # A/B session: flagged through a written node's run context.
        ids["ab_node"] = _node(
            store, "fixture closure", "2026-09-22T01:00:00Z", run="tree-context-ab-live-x", transport_session_id="abs1"
        )
        ids["e_ab"] = _event(store, "kestrel quasar", "2026-09-22T00:30:00Z", [ids["a"]], session="abs1")

        conn = store._conn
        conn.execute("DELETE FROM connections")
        # Pre-ledger implicit edges.
        _edge(conn, "E1a", ids["s1"], ids["a"], "2026-08-10T00:00:00Z", basis=lp.IMPLICIT_BASIS, recall_event_id=ids["e1"], rank=1)
        _edge(conn, "E1b", ids["s1"], ids["b"], "2026-08-10T00:00:00Z", basis=lp.IMPLICIT_BASIS, recall_event_id=ids["e1"], rank=2)
        # Ledger-era implicit edges: a credited at link time, b never, c only 2h later.
        _edge(conn, "E2a", ids["s2"], ids["a"], "2026-09-10T00:00:00Z", basis=lp.IMPLICIT_BASIS, recall_event_id=ids["e2"], rank=1)
        _edge(conn, "E2b", ids["s2"], ids["b"], "2026-09-10T00:00:00Z", basis=lp.IMPLICIT_BASIS, recall_event_id=ids["e2"], rank=2)
        _edge(conn, "E2c", ids["s2"], ids["c"], "2026-09-10T00:00:00Z", basis=lp.IMPLICIT_BASIS, recall_event_id=ids["e2"], rank=3)
        # Non-implicit edge before the cutoff: must survive in both arms.
        _edge(conn, "Ecl", ids["a"], ids["c"], "2026-08-15T00:00:00Z", basis="cluster")
        # Post-cutoff edges: removed from both arms.
        _edge(conn, "E3a", ids["s3"], ids["a"], "2026-09-22T00:00:00Z", basis=lp.IMPLICIT_BASIS, recall_event_id=ids["e3"], rank=1)
        _edge(conn, "Eco", ids["b"], ids["c"], "2026-09-23T00:00:00Z", basis="co_access")

        _ledger(conn, "early", ids["c"], "lookup", "2026-09-07T07:00:00Z", "lk0")  # ledger start
        _ledger(conn, ids["e2"], ids["a"], "grounded", "2026-09-10T00:00:00Z", ids["s2"])
        _ledger(conn, ids["e2"], ids["c"], "lookup", "2026-09-10T02:00:00Z", "lk1")
        _ledger(conn, ids["e3"], ids["a"], "grounded", "2026-09-22T00:00:00Z", ids["s3"])
        conn.commit()
    return {"db": db, "ids": ids}


def test_plan_drops_exactly_uncredited_pre_cutoff_edges(fixture_db):
    conn = lp.open_readonly(fixture_db["db"])
    try:
        plan = lp.plan_credited_drops(conn, CUTOFF)
    finally:
        conn.close()
    # E1b: pre-ledger, not grounded. E2b: never credited. E2c: lookup credit
    # only after the link (not available at link time). E1a grounded, E2a
    # credited at link time. Post-cutoff and non-implicit edges are never planned.
    assert sorted(plan.drop_ids) == ["E1b", "E2b", "E2c"]
    summary = plan.summary()
    assert summary["implicit_pre_cutoff"] == 5
    assert summary["kept_ledger_era"] == 1
    assert summary["dropped_ledger_era"] == 2
    assert summary["kept_pre_ledger_grounded"] == 1
    assert summary["dropped_pre_ledger"] == 1
    assert summary["ledger_credit_only_after_link"] == 1


def test_build_arms_is_read_only_on_source_and_removes_only_planned_edges(fixture_db, tmp_path):
    source = fixture_db["db"]
    before_sha = _sha(source)
    before_mtime = source.stat().st_mtime_ns
    arms = lp.build_arms(source, tmp_path / "arms", CUTOFF)
    assert _sha(source) == before_sha
    assert source.stat().st_mtime_ns == before_mtime
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as conn:
        source_edges = {r[0] for r in conn.execute("SELECT id FROM connections")}
    assert source_edges == {"E1a", "E1b", "E2a", "E2b", "E2c", "Ecl", "E3a", "Eco"}

    def ids(path: str) -> set[str]:
        with sqlite3.connect(path) as conn:
            return {r[0] for r in conn.execute("SELECT id FROM connections")}

    all_edges = ids(arms["paths"]["all"])
    credited_edges = ids(arms["paths"]["credited"])
    assert all_edges == {"E1a", "E1b", "E2a", "E2b", "E2c", "Ecl"}
    assert credited_edges == {"E1a", "E2a", "Ecl"}
    assert arms["credited_edges_dropped"] == 3
    assert arms["edge_counts"]["all"]["implicit"] == 5
    assert arms["edge_counts"]["credited"]["implicit"] == 2
    assert arms["truncated_at_cutoff"]["connections"] == 2


def test_ab_filter_excludes_flagged_sessions(fixture_db):
    conn = lp.open_readonly(fixture_db["db"])
    try:
        assert "abs1" in lp.ab_transport_sessions(conn)
        events, stats = lp.load_eval_events(conn, CUTOFF)
    finally:
        conn.close()
    ids = fixture_db["ids"]
    assert [e.event_id for e in events] == [ids["e3"]]
    assert events[0].credited == frozenset({ids["a"]})
    assert stats["excluded_ab"] == 1
    assert "transport_session_id" not in events[0].ambient_context


def test_event_rules():
    assert lp.event_trips_ab("project:target", None, "anything")
    assert lp.event_trips_ab("global", "project:x", "q")
    assert lp.event_trips_ab("global", "global", "Tree-Context A/B run")
    assert not lp.event_trips_ab("project:lm", "project:lm", "implicit link policy")


def test_score_event_and_decision_rule():
    event = lp.EvalEvent("e", "2026-09-21", "q", SCOPE, {}, 1, 5, frozenset({"n2"}))
    out = lp.score_event(event, [("n1", ("graph",)), ("n2", ("bm25", "graph")), ("n3", ("bm25",))], 0.01)
    assert out.rr == 0.5 and out.hit1 == 0 and out.hit3 == 1
    assert out.graph_hits_credited == 1
    assert out.graph_hits_uncredited == 1
    assert out.graph_only_uncredited == 1

    base = {"labelled_events": 200, "mrr": 0.30, "hit@3": 0.40, "uncredited_graph_hits_per_event": 1.0, "latency_p50_ms": 100.0}
    ok = dict(base, mrr=0.295, **{"hit@3": 0.395}, uncredited_graph_hits_per_event=0.9, latency_p50_ms=105.0)
    assert lp.decide(base, ok)["recommendation"] == "credited"
    worse = dict(ok, mrr=0.28)
    assert lp.decide(base, worse)["recommendation"] == "all"
    no_gain = dict(ok, uncredited_graph_hits_per_event=0.97)
    assert lp.decide(base, no_gain)["recommendation"] == "all"
    few = dict(base, labelled_events=50)
    assert lp.decide(few, ok)["verdict"] == "insufficient"
    assert lp.overall_recommendation(
        {"sfx": {"decision": {"recommendation": "credited"}}, "alt": {"decision": {"recommendation": "all"}}}
    ) == {"per_host": {"sfx": "credited", "alt": "all"}, "overall": "split", "code_default": "all"}


def test_run_host_end_to_end_on_fixture(fixture_db, tmp_path):
    source = fixture_db["db"]
    before = _sha(source)
    result = lp.run_host("fixture", source, CUTOFF, workdir=tmp_path / "work")
    assert _sha(source) == before
    assert result["traffic"]["kept"] == 1
    assert result["arms"]["all"]["events"] == 1
    assert result["edge_counts"]["credited"]["implicit"] == 2
    assert result["decision"]["verdict"] == "insufficient"
