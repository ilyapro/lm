"""A cold server recovers the interval crossed by a committed knowledge write."""

import json
import multiprocessing
import os
import sqlite3

import pytest

from living_memory import server
from living_memory.auto_consolidation import PENDING_KV_KEY
from test_lookup import FakeMCP


SCOPE = "project:harbor"
CONTEXT = {"scope": SCOPE}
TOPIC = "Harbor crane calibration measurement"


def interrupted_writer(path, tool, prior, cut):
    mcp = server.create_mcp_server(path, mcp_factory=FakeMCP)
    remember = mcp.tools["memory_remember"]
    for i in range(prior):
        trace = remember(f"{TOPIC} {i}", CONTEXT)["node"]
    scheduler = mcp.auto_consolidation
    store = mcp.memory_store

    def stop_after_commit(*args):
        assert not store.connection.in_transaction
        os._exit(73)

    if cut == "before_commit":
        record = scheduler.record_request

        def stop_before_commit(scope, mark):
            record(scope, mark)
            assert store.connection.in_transaction
            assert json.loads(store.get_kv(PENDING_KV_KEY))[scope] == mark
            os._exit(73)

        scheduler.record_request = stop_before_commit
    elif tool == "teach":
        server._teach_result_to_dict = stop_after_commit
    else:
        server.derive_edges_for_new_trace = stop_after_commit

    if tool == "teach":
        mcp.tools["memory_teach"](trace["id"], f"{TOPIC} corrected", context=CONTEXT)
    elif tool == "alternatives":
        remember(f"{TOPIC} revised", CONTEXT, alternatives_considered=[
            {"approach": "Skip calibration", "rejected_because": "Required by inspection"},
            {"approach": "Reuse old reading", "rejected_because": "Equipment changed"},
        ])
    else:
        remember(f"{TOPIC} revised", CONTEXT)


@pytest.mark.parametrize("tool,prior,cut,committed,due", [
    ("remember", 4, "before_commit", 4, False),
    ("remember", 3, "after_commit", 4, False),
    ("remember", 4, "after_commit", 5, True),
    ("teach", 4, "before_commit", 4, False),
    ("teach", 4, "after_commit", 5, True),
    ("alternatives", 3, "after_commit", 6, True),
])
def test_cold_server_observes_knowledge_boundary(tmp_path, monkeypatch, tool, prior, cut, committed, due):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "adaptive")
    path = tmp_path / "boundary.sqlite3"
    writer = multiprocessing.get_context("spawn").Process(
        target=interrupted_writer, args=(path, tool, prior, cut),
    )
    writer.start()
    try:
        writer.join(30)
        assert writer.exitcode == 73
    finally:
        if writer.is_alive():
            writer.kill()
            writer.join()
        writer.close()

    with sqlite3.connect(path) as reader:
        assert reader.execute("SELECT COUNT(*) FROM nodes WHERE level='trace'").fetchone() == (committed,)
        active, = reader.execute(
            "SELECT COUNT(*) FROM nodes WHERE level='trace' AND scope=? AND decayed=0", (SCOPE,),
        ).fetchone()
        row = reader.execute("SELECT value FROM kv WHERE key=?", (PENDING_KV_KEY,)).fetchone()
        assert set(json.loads(row[0]) if row else {}) == ({SCOPE} if due else set())

    cold = server.create_mcp_server(path, mcp_factory=FakeMCP)
    scheduler = cold.auto_consolidation
    try:
        assert scheduler.wait_idle(30, include_retries=True)
        assert scheduler.status()["passes_completed"] == int(due)
        if due:
            assert scheduler.last_result(SCOPE)["summary"]["traces_considered"] == active
            assert cold.memory_store.get_kv(PENDING_KV_KEY) == "{}"
        next_write = cold.tools["memory_remember"](f"{TOPIC} after restart", CONTEXT)
        assert bool(next_write["auto_consolidation"]) == (committed == 4)
        assert scheduler.wait_idle(30, include_retries=True)
        assert scheduler.status()["passes_completed"] == 1
        assert cold.memory_store.get_kv(PENDING_KV_KEY) == "{}"
    finally:
        scheduler.close()
        cold.memory_store.close()
