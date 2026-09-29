"""Tests for ``scripts/recall_precision_replay.py`` on a tiny synthetic store.

The store is built with the real schema (``MemoryStore``, hash embeddings) and
backdated rows. Arms are exercised through the real
``MemoryRecallService.memory_recall``; the valve under test is a stand-in gate
patched into ``living_memory.retrieval`` that reads an env var, so the test
proves env is applied per arm and restored, not any production valve.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from living_memory.storage import MemoryStore, pack_chunk_embedding

REPO = Path(__file__).resolve().parents[1]
SCOPE = "project:fx-precision"


def _load_module() -> Any:
    name = "recall_precision_replay"
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rp = _load_module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hour(h: float) -> str:
    base = 15 * 60 + h * 60  # minutes from 2026-09-27T00:00Z
    day, minutes = divmod(int(base), 24 * 60)
    return f"2026-09-{27 + day:02d}T{minutes // 60:02d}:{minutes % 60:02d}:00Z"


QUERIES = [
    "kestrel manifold lattice turbine",
    "kestrel manifold quasar setup",
    "kestrel ferrite manifold obsidian",
    "kestrel manifold zephyr check",
    "kestrel manifold lattice review",
    "kestrel manifold obsidian audit",
    "kestrel manifold turbine plan",
    "kestrel manifold quasar wrapup",
]


def _backdate(conn: sqlite3.Connection, node_id: str, stamp: str) -> None:
    conn.execute("UPDATE nodes SET created_at = ?, updated_at = ?, timestamp = ? WHERE id = ?", (stamp, stamp, stamp, node_id))


def build_store(path: Path) -> dict[str, Any]:
    """8 events hourly from 2026-09-27T15:00Z; node ``hub`` marked irrelevant on 3 train queries."""

    store = MemoryStore(path)
    ids: dict[str, str] = {}
    words = ["lattice", "quasar", "ferrite", "obsidian", "turbine", "zephyr"]
    for index in range(8):
        extra = words[index % len(words)]
        node = store.create_node(
            level="trace",
            content=f"kestrel manifold {extra} note {index} about the manifold",
            context={"scope": SCOPE},
        )
        ids[f"n{index}"] = node.id
    hub = store.create_node(level="trace", content="kestrel manifold hub generic text", context={"scope": SCOPE})
    ids["hub"] = hub.id
    for title_twin in range(2):
        schema = store.create_node(
            level="schema",
            content=f"Procedure: kestrel manifold rollout\n1. step kestrel {title_twin}",
            context={"scope": SCOPE, "procedure": ["step"]},
        )
        ids[f"schema{title_twin}"] = schema.id
    future = store.create_node(
        level="trace", content="kestrel manifold lattice turbine quasar future", context={"scope": SCOPE}
    )
    ids["future"] = future.id
    conn = store._conn
    for key, node_id in ids.items():
        _backdate(conn, node_id, "2026-09-29T06:00:00Z" if key == "future" else "2026-09-26T10:00:00Z")
    conn.commit()
    from living_memory.retrieval import MemoryRecallService

    service = MemoryRecallService(store)
    events: list[str] = []
    for index, query in enumerate(QUERIES):
        results = service.memory_recall(query, scope=SCOPE, max_results=5, log_access=False, log_event=False)
        recorded = [r.node_id for r in results if r.node_id != ids["future"]]
        if ids["hub"] not in recorded:
            recorded = [ids["hub"], *recorded][:5]
        event = store.record_recall_event(
            query=query,
            scope=SCOPE,
            requested_scope=SCOPE,
            ambient_context={"transport_session_id": f"t{index}", "agent": "claude"},
            depth=1,
            max_results=5,
            results=[{"node_id": nid, "rank": rank + 1} for rank, nid in enumerate(recorded)],
        )
        conn.execute("UPDATE recall_events SET created_at = ? WHERE id = ?", (_hour(index), event.id))
        events.append(event.id)
        hub_rank = recorded.index(ids["hub"])
        marks = [(ids["hub"], "irrelevant", hub_rank)]
        other = next(n for n in recorded if n != ids["hub"])
        marks.append((other, "used", recorded.index(other)))
        for node_id, mark, rank in marks:
            conn.execute(
                "INSERT INTO recall_feedback_marks (recall_event_id, node_id, mark, accepted, via_tool, "
                "source_id, transport_session_id, agent, rank, marked_at) VALUES (?, ?, ?, 1, 'memory_recall', "
                "'src', ?, 'claude', ?, ?)",
                (event.id, node_id, mark, f"t{index}", rank, _hour(index + 0.5)),
            )
    # Post-cutoff learning that the counterfactual must strip.
    conn.execute(
        "INSERT INTO connections (id, source_id, target_id, type, weight, metadata, created_at, updated_at) "
        "VALUES ('late-edge', ?, ?, 'related', 0.9, '{}', ?, ?)",
        (ids["n0"], ids["n1"], _hour(7), _hour(7)),
    )
    conn.execute(
        "INSERT INTO recall_credit_ledger (recall_event_id, node_id, basis, source_id, credited_at) "
        "VALUES (?, ?, 'lookup', 'src', ?)",
        (events[6], ids["n2"], _hour(6.2)),
    )
    conn.commit()
    store.close()
    return {"ids": ids, "events": events}


@pytest.fixture()
def synthetic(tmp_path: Path) -> dict[str, Any]:
    path = tmp_path / "snap.sqlite3"
    info = build_store(path)
    info["path"] = path
    return info


def test_parse_arm_and_env_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    assert rp.parse_arm("gate04:LM_RECALL_MIN_SCORE=0.4,LM_RECALL_GATE_FORM=stub") == (
        "gate04",
        {"LM_RECALL_MIN_SCORE": "0.4", "LM_RECALL_GATE_FORM": "stub"},
    )
    assert rp.parse_arm("plain") == ("plain", {})
    with pytest.raises(ValueError):
        rp.parse_arm("bad name:X=1")
    with pytest.raises(ValueError):
        rp.parse_arm("a:lower=1")
    monkeypatch.setenv("LM_HUB_SUPPRESSION_FACTOR", "0.1")
    monkeypatch.delenv("LM_RECALL_MIN_SCORE", raising=False)
    with rp.applied_env({"LM_RECALL_MIN_SCORE": "0.4"}):
        assert os.environ["LM_RECALL_MIN_SCORE"] == "0.4"
        assert "LM_HUB_SUPPRESSION_FACTOR" not in os.environ  # baseline valves unset
    assert os.environ["LM_HUB_SUPPRESSION_FACTOR"] == "0.1"
    assert "LM_RECALL_MIN_SCORE" not in os.environ


def test_split_boundaries_and_membership(synthetic: dict[str, Any]) -> None:
    split = rp.build_split({"sfx": synthetic["path"]})
    host = split["hosts"]["sfx"]
    assert host["events"] == 8
    assert host["counts"] == {"train": 4, "eval": 2, "holdout": 2}
    assert host["eval_start"]["event_id"] == synthetic["events"][4]
    assert host["holdout_start"]["event_id"] == synthetic["events"][6]
    store = rp.load_host_store(synthetic["path"])
    segments = [rp.segment_of(store.events[eid], host) for eid in synthetic["events"]]
    assert segments == ["train"] * 4 + ["eval"] * 2 + ["holdout"] * 2
    assert rp.cutoff_for(host, "eval").startswith("2026-09-27T19:00:00")


def test_counterfactual_strips_post_cutoff_rows(synthetic: dict[str, Any], tmp_path: Path) -> None:
    before = _sha(synthetic["path"])
    target = tmp_path / "cf.sqlite3"
    report = rp.build_counterfactual(synthetic["path"], target, _hour(4))
    assert _sha(synthetic["path"]) == before
    assert report["removed"]["recall_feedback_marks"] == 8  # events 4..7, two marks each (marked at +0.5h)
    assert report["removed"]["connections"] >= 1
    assert report["removed"]["recall_credit_ledger"] == 1
    conn = sqlite3.connect(target)
    try:
        assert conn.execute("SELECT COUNT(*) FROM connections WHERE id = 'late-edge'").fetchone()[0] == 0
        assert conn.execute("SELECT MAX(marked_at) FROM recall_feedback_marks").fetchone()[0] < _hour(4)
    finally:
        conn.close()


def test_train_hubs_rule(synthetic: dict[str, Any]) -> None:
    store = rp.load_host_store(synthetic["path"])
    host = rp.compute_split(rp.window_events(store))
    hubs = rp.train_hubs(store, host)
    assert hubs == {synthetic["ids"]["hub"]}
    assert rp.train_hubs(store, host, min_queries=5) == set()


def test_score_event_metrics() -> None:
    event = rp.ReplayEvent(
        event_id="e1", created_at="2026-09-28T10:00:00.000000Z", query="q", scope=None,
        ambient_context={}, depth=1, max_results=5, recorded=("a", "b", "c", "d"), trace_id=None,
    )
    d = rp.Delivered
    delivered = [
        d("a", True, "trace", "", "2026-09-01T00:00:00.000000Z"),
        d("s1", True, "schema", "procedure: x", "2026-09-01T00:00:00.000000Z"),
        d("s2", True, "schema", "procedure: x", "2026-09-01T00:00:00.000000Z"),
        d("c", False, "trace", "", "2026-09-01T00:00:00.000000Z"),
        d("late", True, "trace", "", "2026-09-29T00:00:00.000000Z"),
    ]
    labels = {
        "a": rp.Label("a", "irrelevant", 1),
        "b": rp.Label("b", "used", 2),
        "c": rp.Label("c", "irrelevant", 3),
        "new": rp.Label("new", "used", 4),
    }
    created = {"a": "2026-09-01T00:00:00.000000Z", "b": "2026-09-01T00:00:00.000000Z",
               "c": "2026-09-01T00:00:00.000000Z", "new": "2026-09-28T09:00:00.000000Z"}
    row = rp.score_event(event, delivered, labels, {"s1"}, "2026-09-28T08:00:00Z", created)
    assert row["full"] == 3 and row["stub"] == 1 and row["future_filtered"] == 1
    assert row["labels_excluded_created_after_cutoff"] == 1
    assert (row["irr_marked"], row["irr_full"], row["irr_top3"]) == (2, 1, 1)
    assert (row["used_marked"], row["used_full"], row["used_lost"]) == (1, 0, 1)
    assert row["hub_top3"] == 1 and row["top3_full"] == 3
    assert row["dup_schema_slots"] == 1
    assert row["entrants"] == ["s1", "s2"]
    assert row["rank1"] == "a"
    assert row["by_rank"]["1"] == {"irr_marked": 1, "irr_full": 1}
    assert row["by_rank"]["3"] == {"irr_marked": 1, "irr_full": 0}


def test_run_host_live_path_arms(synthetic: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import living_memory.retrieval as retrieval

    original = retrieval.apply_score_gate
    seen: list[str | None] = []

    def fake_gate(ranked, max_results, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(os.environ.get("LM_FAKE_CUT"))
        if os.environ.get("LM_FAKE_CUT"):
            return list(ranked[:1]), list(ranked[1:])
        return original(ranked, max_results, **kwargs)

    monkeypatch.setattr(retrieval, "apply_score_gate", fake_gate)
    monkeypatch.delenv("LM_FAKE_CUT", raising=False)
    before = _sha(synthetic["path"])
    split = rp.build_split({"sfx": synthetic["path"]})
    result = rp.run_host(
        host="sfx",
        snapshot=synthetic["path"],
        host_split=split["hosts"]["sfx"],
        segment="eval",
        arms={"baseline": {}, "cut": {"LM_FAKE_CUT": "1"}},
        workdir=tmp_path / "wd",
    )
    assert _sha(synthetic["path"]) == before
    assert "LM_FAKE_CUT" not in os.environ
    assert sorted(seen, key=str) == ["1", "1", None, None]
    assert result["replayed_events"] == 2 and result["hubs_train_defined"] == 1
    base, cut = result["summary"]["baseline"], result["summary"]["cut"]
    assert base["full_slots_per_event"] > 1
    assert cut["full_slots_per_event"] == 1
    assert cut["vs_baseline"]["rank1_kept"] == 1.0
    assert cut["vs_baseline"]["full_slots_cut"] > 0
    # The node created after every event is hidden before ranking, never delivered.
    assert base["future_filtered"] == 0 and base["future_candidates_hidden"] >= 1
    future = synthetic["ids"]["future"]
    for rows in result["events"].values():
        assert all(future not in row["entrants"] for row in rows)
    assert base["irrelevant"]["marked"] == 2 and base["used"]["marked"] == 2
    markdown = rp.render_markdown([result])
    assert "## sfx — eval" in markdown and "| cut |" in markdown


def test_entrant_grader_placebo(tmp_path: Path) -> None:
    path = tmp_path / "g.sqlite3"
    with MemoryStore(path):
        pass
    conn = sqlite3.connect(path)
    vector = pack_chunk_embedding([1.0, 0.0, 0.0, 0.0])
    rows = {
        "trace": ("alpha bravo charlie delta echo foxtrot", "2026-09-28T12:00:00Z"),
        "ent": ("alpha bravo charlie delta", "2026-09-20T00:00:00Z"),
        "twin": ("zulu yankee xray whiskey", "2026-09-20T00:00:00Z"),
    }
    for node_id, (content, stamp) in rows.items():
        conn.execute(
            "INSERT INTO nodes (id, level, content, scope, context, timestamp, created_at, updated_at) "
            "VALUES (?, 'trace', ?, 'project:lm', '{}', ?, ?, ?)",
            (node_id, content, stamp, stamp, stamp),
        )
        conn.execute(
            "INSERT INTO node_chunk_embeddings (id, node_id, chunk_index, dimensions, embedding, "
            "token_start, token_end, content_fingerprint, created_at, updated_at) "
            "VALUES (?, ?, 0, 4, ?, 0, 1, 'fp', ?, ?)",
            (f"c-{node_id}", node_id, vector, stamp, stamp),
        )
    conn.commit()
    conn.close()
    created = {k: rp.efa.normalize_ts(v[1]) for k, v in rows.items()}
    grader = rp.EntrantGrader(path, created)
    try:
        event = rp.ReplayEvent("e", "2026-09-28T10:00:00.000000Z", "q", None, {}, 1, 5, (), "trace")
        grades = grader.grade(event, ["ent"], ["ent"])
    finally:
        grader.close()
    assert grades == [{"node_id": "ent", "graded": True, "grounded": True, "has_twin": True, "twin_grounded": False}]


def test_snapshot_remote_streams_without_remote_writes(synthetic: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_run = subprocess.run

    def fake_ssh(cmd, **kwargs):  # type: ignore[no-untyped-def]
        assert cmd[:4] == ["ssh", "alt", "python3", "-"]
        return real_run([sys.executable, "-", *cmd[4:]], **kwargs)

    monkeypatch.setattr(rp.subprocess, "run", fake_ssh)
    before = _sha(synthetic["path"])
    out = tmp_path / "snaps" / "alt.sqlite3"
    manifest = rp.snapshot_remote("alt", str(synthetic["path"]), out)
    assert _sha(synthetic["path"]) == before
    assert manifest["snapshot_sha256"] == _sha(out)
    assert manifest["source_path"] == f"alt:{synthetic['path']}"
    assert manifest["row_counts"]["recall_events"] == 8
    index = json.loads(rp.write_snapshot_index(out.parent).read_text())
    assert index["alt"]["row_counts"]["recall_feedback_marks"] == 16
