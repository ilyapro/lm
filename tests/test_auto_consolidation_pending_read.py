"""Unread durable pending work survives ordinary committed writes."""

import json
import sqlite3
import threading
from unittest.mock import Mock

import pytest

import living_memory.server as server
from living_memory.auto_consolidation import PENDING_KV_KEY
from living_memory.storage import MemoryStore
from test_lookup import FakeMCP


OLD = "project:old"
NEW = "project:new"
PENDING = '{ "project:old": "2026-10-01T00:00:00Z" }'


def pending_bytes(path):
    with sqlite3.connect(path) as reader:
        return reader.execute("SELECT value FROM kv WHERE key=?", (PENDING_KV_KEY,)).fetchone()[0]


@pytest.fixture
def unread(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "fixed")
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store:
        original = store.append_trace("Harbor datum calibration", {"scope": OLD}).id
        for index in range(99):
            store.append_trace(f"Tide measurement {index}", {"scope": NEW})
        store.set_kv(PENDING_KV_KEY, PENDING)

    state = {"unavailable": True, "loads": 0}
    get_kv = MemoryStore.get_kv

    def load(store, key):
        if key == PENDING_KV_KEY:
            state["loads"] += 1
            if state["unavailable"]:
                raise sqlite3.OperationalError("pending read unavailable")
        return get_kv(store, key)

    monkeypatch.setattr(MemoryStore, "get_kv", load)
    mcp = server.create_mcp_server(path, mcp_factory=FakeMCP)
    runs = []
    monkeypatch.setattr(mcp.auto_consolidation, "_run_pass", lambda scope: runs.append(scope) or {})
    try:
        yield path, mcp, original, state, runs
    finally:
        mcp.auto_consolidation.close()
        mcp.memory_store.close()


def write(path, mcp, original, tool, scope):
    content = "Harbor datum uses the corrected tide measurement"
    if tool == "memory_teach":
        result = mcp.tools[tool](original, content, context={"scope": scope})
        node = result["corrective_trace"]["id"]
    else:
        result = mcp.tools[tool](content, context={"scope": scope})
        node = result["node"]["id"]
    assert not mcp.memory_store.connection.in_transaction
    with sqlite3.connect(path) as reader:
        assert reader.execute("SELECT content FROM nodes WHERE id=?", (node,)).fetchone() == (content,)
        if tool == "memory_teach":
            assert reader.execute(
                "SELECT target_id FROM connections WHERE source_id=? AND type='supersedes'", (node,),
            ).fetchall() == [(original,)]
    return result


@pytest.mark.parametrize("tool", ["memory_remember", "memory_teach"])
@pytest.mark.parametrize("due", [False, True])
def test_next_write_recovers_unread_pending(unread, monkeypatch, caplog, tool, due):
    path, mcp, original, state, runs = unread
    scheduler = mcp.auto_consolidation
    assert state["loads"] == 1
    assert "pending read unavailable" in caplog.text
    assert pending_bytes(path) == PENDING
    state["unavailable"] = False
    expected = {OLD, NEW} if due else {OLD}
    starts = []
    real_start = threading.Thread.start

    def start(thread):
        # The merged obligation is durable before admission, after the write.
        saved = json.loads(pending_bytes(path))
        assert set(saved) == expected
        assert saved[OLD] == json.loads(PENDING)[OLD]
        assert not mcp.memory_store.connection.in_transaction
        starts.append(thread)
        return real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    result = write(path, mcp, original, tool, NEW if due else OLD)
    if due:
        assert result["auto_consolidation"]["status"] == "scheduled"
        assert result["auto_consolidation"]["trace_count"] == 100
    else:
        assert result["auto_consolidation"] is None
    assert scheduler.wait_idle(timeout=5, include_retries=True)
    assert len(starts) == 1
    assert set(runs) == expected and len(runs) == len(expected)
    assert scheduler.status()["passes_completed"] == len(expected)
    assert pending_bytes(path) == "{}"
    assert state["loads"] == 2

    # Successful initialization keeps the idle path entirely in memory.
    save = Mock(wraps=scheduler._save_pending)
    monkeypatch.setattr(scheduler, "_save_pending", save)
    assert write(path, mcp, original, tool, OLD)["auto_consolidation"] is None
    assert state["loads"] == 2
    save.assert_not_called()
    assert len(starts) == 1 and len(runs) == len(expected)


@pytest.mark.parametrize("payload", [None, "", '{"project:old":', "[]", '{"project:old": null}'])
def test_repeated_read_error_retains_bytes_and_new_requests(unread, monkeypatch, caplog, payload):
    path, mcp, original, state, runs = unread
    scheduler = mcp.auto_consolidation
    if payload is not None:
        state["unavailable"] = False
        mcp.memory_store.set_kv(PENDING_KV_KEY, payload)
    raw = PENDING if payload is None else payload
    save = Mock(wraps=scheduler._save_pending)
    start = Mock(wraps=scheduler._ensure_worker)
    monkeypatch.setattr(scheduler, "_save_pending", save)
    monkeypatch.setattr(scheduler, "_ensure_worker", start)

    # A due request is retained in memory while the unread bytes are untouched.
    assert write(path, mcp, original, "memory_remember", NEW)["auto_consolidation"] is None
    assert scheduler.status()["queued"] == [NEW]
    assert pending_bytes(path) == raw
    assert write(path, mcp, original, "memory_teach", OLD)["auto_consolidation"] is None
    assert state["loads"] == 3
    assert pending_bytes(path) == raw
    assert "could not schedule post-commit auto-consolidation" in caplog.text
    assert runs == []
    save.assert_not_called()
    start.assert_not_called()

    # Repair the resource; a non-boundary write resumes both obligations.
    state["unavailable"] = False
    mcp.memory_store.set_kv(PENDING_KV_KEY, PENDING)
    assert write(path, mcp, original, "memory_remember", OLD)["auto_consolidation"] is None
    assert scheduler.wait_idle(timeout=5, include_retries=True)
    assert state["loads"] == 4
    assert set(runs) == {OLD, NEW} and len(runs) == 2
    assert set(json.loads(save.call_args_list[0].args[0])) == {OLD, NEW}
    assert scheduler.status()["passes_completed"] == 2
    assert pending_bytes(path) == "{}"
