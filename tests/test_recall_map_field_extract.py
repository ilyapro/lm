"""Offline replay uses the builder's selected members and real cache paths."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from living_memory.config import MemoryConfig
from living_memory.recall_map import RecallMapBuilder, cache_key
from living_memory.retrieval import MemoryRecallService, RecallResult
from living_memory.storage import MemoryStore


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recall_map_field_extract.py"
spec = importlib.util.spec_from_file_location("field_extract", SCRIPT)
assert spec is not None and spec.loader is not None
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)

SCOPE = "project:game"
TASK = "quality clips coverage registry refresh"
QUERY = "database migration rollback"
CONTENT = "database migration rollback verify release note"
NO_DRIFT = {
    "pairs_probed": 0,
    "pairs_with_drift": 0,
    "members_renamed_after_a_directory": 0,
    "examples": [],
}


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    for name in os.environ:
        if name.startswith(("LM_MAP_", "LM_RECALL_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def seed(store):
    for index in range(40):
        store.create_node(
            level="trace", content=f"quarterly ledger reconciliation entry {index}",
            context={"scope": "project:background"},
        )
    return [
        store.create_node(
            node_id=f"node-{index:02}", level="trace", content=f"{CONTENT} {index}",
            context={"scope": SCOPE, "procedure_id": (
                "schema migration" if index < 6 else "release checking"
            )},
            timestamp="2026-11-16T00:00:00Z",
        )
        for index in range(12)
    ]


def cases(head=()):
    return [
        {
            "created_at": f"2026-11-16T00:0{index}:00Z",
            "query": QUERY, "requested_scope": SCOPE, "depth": "deep",
            "max_results": 2, "ambient_context": {"task": TASK},
            "scope": SCOPE, "task": TASK, "task_pattern": None,
            "cache_key": cache_key(SCOPE, task=TASK),
            "head_node_ids": list(head), "clusters": [],
        }
        for index in range(2)
    ]


@pytest.mark.parametrize("populated", [False, True])
def test_real_replay_and_recount(tmp_path, monkeypatch, populated):
    snapshot = tmp_path / "snapshot.sqlite3"
    with MemoryStore(MemoryConfig(db_path=snapshot, embedding_model="hash")) as store:
        if populated:
            seed(store)
    original = snapshot.read_bytes()
    selected = []
    recounted = []
    real_pool = RecallMapBuilder._pool
    real_recount = RecallMapBuilder._recount

    def pool(self, results, **kwargs):
        members, accounting = real_pool(self, results, **kwargs)
        selected.append(members)
        return members, accounting

    def recount(self, templates, members):
        assert templates and any(members is items for items in selected)
        result = real_recount(self, templates, members)
        recounted.append(result)
        return result

    monkeypatch.setattr(RecallMapBuilder, "_pool", pool)
    monkeypatch.setattr(RecallMapBuilder, "_recount", recount)
    inputs = cases(["node-00", "node-01"] if populated else [])
    # Input order differs from the chronological order of cache seeding.
    summary = extract.attach_replay(list(reversed(inputs)), snapshot)
    expected_map = {
        "pool": 10, "covered": 10, "more": 0, "curtailed": False,
        "clusters": [
            {"label": label, "count": count, "stage": "structural",
             "medoid_node_id": f"node-{index:02}", "example_chars": len(f"{CONTENT} {index}")}
            for label, count, index in (("schema migration", 4, 2), ("release checking", 6, 6))
        ],
    } if populated else None
    for index, case in enumerate(inputs):
        assert case["replay"] == {
            "head_node_ids": ["node-00", "node-01"] if populated else [],
            "head_reproduced": True,
            "pool_node_ids": [f"node-{i:02}" for i in range(2, 12)] if populated else [],
            "pool_size": 10 if populated else 0,
            "residual_size": 10 if populated else 0,
            "pool_stage_composition": {"structural": 10} if populated else {},
            "cold": expected_map, "warm": expected_map,
            "warm_cache_hit": bool(populated and index),
        }
    assert summary == {"cases": 2, "head_reproduced": 2, **({"pool_structural": 20} if populated else {})}
    assert extract.recount_drift_probe(inputs, snapshot) == {
        **NO_DRIFT, "pairs_probed": int(populated),
    }
    assert len(recounted) == (2 if populated else 0)
    if populated:
        assert all(groups and sum(len(group.members) for group in groups) == 10 for groups in recounted)
    assert snapshot.read_bytes() == original


def test_synthetic_selection_preserves_topics_order_and_capacity(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot.sqlite3"
    with MemoryStore(MemoryConfig(db_path=snapshot, embedding_model="hash")) as store:
        nodes = seed(store)
        path = store.create_node(
            node_id="path", level="trace", content="transaction rollback boundary",
            context={"scope": SCOPE, "files": ["src/storage/transactions/replay.py"]},
        )
        journal = store.create_node(
            node_id="journal", level="trace", content="completed synthetic run",
            context={"scope": SCOPE, "kind": "execution_journal"},
        )
    ranked = [RecallResult(node=node, score=1.0, vector_score=1.0)
              for node in (nodes[0], nodes[0], nodes[1], nodes[2], nodes[3], path, journal)]
    ranked.append(RecallResult(node=nodes[6], score=0.0, vector_score=0.0))

    def recall(self, **kwargs):
        self.last_residual = ranked
        return []

    observed = []

    class BoundedBuilder(RecallMapBuilder):
        def __init__(self, store):
            super().__init__(store, max_pool_nodes=3)

        def _pool(self, results, **kwargs):
            members, selection = super()._pool(results, **kwargs)
            self.selected = members
            assert [(member.node.id, member.rank, member.topic) for member in members] == [
                ("node-00", 0, ("structural", "procedure_id", "schema migration")),
                ("node-01", 2, ("structural", "procedure_id", "schema migration")),
                ("path", 5, ("path", "transactions")),
            ]
            return members, selection

        def _recount(self, templates, members):
            assert members is self.selected
            result = super()._recount(templates, members)
            observed.append(result)
            return result

    monkeypatch.setattr(MemoryRecallService, "memory_recall", recall)
    monkeypatch.setattr(extract, "RecallMapBuilder", BoundedBuilder)
    inputs = cases()
    assert extract.attach_replay(inputs, snapshot) == {
        "cases": 2, "head_reproduced": 2, "pool_structural": 4, "pool_path": 2,
    }
    for case in inputs:
        replay = case["replay"]
        assert replay["pool_node_ids"] == ["node-00", "node-01", "path"]
        assert replay["pool_size"] == 3 and replay["residual_size"] == 8
        assert replay["pool_stage_composition"] == {"structural": 2, "path": 1}
        assert replay["cold"] == replay["warm"]
    assert inputs[1]["replay"]["warm_cache_hit"]
    assert extract.recount_drift_probe(inputs, snapshot) == {**NO_DRIFT, "pairs_probed": 1}
    assert len(observed) == 2 and all(groups for groups in observed)

    # A new structural topic cannot inherit the cached path membership.
    changed = deepcopy(path)
    changed.context["procedure_id"] = "new structural topic"
    def changed_recall(self, **kwargs):
        self.last_residual = [RecallResult(node=path if not getattr(self, "seen", False) else changed,
                                          score=1.0, vector_score=1.0)]
        self.seen = True
        return []

    monkeypatch.setattr(MemoryRecallService, "memory_recall", changed_recall)
    monkeypatch.setattr(extract, "RecallMapBuilder", RecallMapBuilder)
    real_recount = RecallMapBuilder._recount
    refused = []
    def recount(self, templates, members):
        result = real_recount(self, templates, members)
        refused.append(result)
        return result
    monkeypatch.setattr(RecallMapBuilder, "_recount", recount)
    assert extract.recount_drift_probe(cases(), snapshot) == NO_DRIFT
    assert refused == [None]


@pytest.mark.parametrize("populated", [False, True])
def test_cli_with_disposable_snapshots_and_journals(tmp_path, populated):
    snapshot = tmp_path / "field.sqlite3"
    streams = tmp_path / "streams"
    streams.mkdir()
    events = []
    with MemoryStore(MemoryConfig(db_path=snapshot, embedding_model="hash")) as store:
        if populated:
            seed(store)
        for index in range(2):
            event = store.record_recall_event(
                query=QUERY, scope=SCOPE, requested_scope=SCOPE,
                ambient_context={"task": TASK}, depth="deep", max_results=2,
                recall_map={"clusters": [], "pool": 0, "covered": 0},
            )
            events.append({"kind": "probe", "query": QUERY, "ts": event.created_at,
                           "session_id": "synthetic", "outcome": "ok"})
    (streams / "journal.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    manifest = tmp_path / "manifest.json"
    analysis = tmp_path / "analysis.json"
    original = snapshot.read_bytes()
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--streams", str(streams),
         "--field-snapshot", str(snapshot), "--e2e-snapshot", str(snapshot),
         "--e2e-journals", str(tmp_path / "absent-*.jsonl"), "--since", "2000-01-01",
         "--out-manifest", str(manifest), "--out-analysis", str(analysis)],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(analysis.read_text())
    assert result["replay"]["field_alt"]["cases"] == 2
    assert result["recount_drift_probe"] == {**NO_DRIFT, "pairs_probed": int(populated)}
    saved = json.loads(manifest.read_text())
    assert saved["splits"]["counts"] == {"train": 2, "eval": 0, "holdout": 0}
    assert [case["replay"]["warm_cache_hit"] for case in saved["cases"]] == [False, populated]
    assert snapshot.read_bytes() == original
