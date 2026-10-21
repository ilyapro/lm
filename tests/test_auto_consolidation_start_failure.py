"""Optional worker startup cannot change a committed knowledge outcome."""

import json
import sqlite3
import threading

import pytest

import living_memory.server as server
from living_memory.auto_consolidation import AutoConsolidationScheduler, PENDING_KV_KEY
from living_memory.storage import MemoryStore
from test_lookup import FakeMCP


SCOPE = "project:worker-start"
CONTENT = "Calibration uses the harbor datum"


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "fixed")
    path = tmp_path / "memory.sqlite3"
    mcp = server.create_mcp_server(path, mcp_factory=FakeMCP)
    original = None
    for index in range(99):
        node = mcp.memory_store.append_trace(
            f"Calibration fixture number {index} is a recorded measurement",
            {"scope": SCOPE, "agent": "fixture"},
        )
        original = original or node
    yield path, mcp, original.id
    mcp.auto_consolidation.close()
    mcp.memory_store.close()


def write(mcp, tool, original):
    if tool == "memory_teach":
        result = mcp.tools[tool](original, CONTENT, context={"scope": SCOPE})
        return result, result["corrective_trace"]["id"]
    result = mcp.tools[tool](CONTENT, context={"scope": SCOPE})
    return result, result["node"]["id"]


def durable(path):
    with sqlite3.connect(path) as connection:
        traces = dict(connection.execute("SELECT id, content FROM nodes WHERE level='trace'"))
        edges = list(connection.execute(
            "SELECT source_id, target_id FROM connections WHERE type='supersedes'"
        ))
        pending = connection.execute("SELECT value FROM kv WHERE key=?", (PENDING_KV_KEY,)).fetchone()
    return traces, edges, json.loads(pending[0]) if pending else None


@pytest.mark.parametrize("tool", ["memory_remember", "memory_teach"])
def test_start_failure_preserves_write_pending_and_cold_resume(seeded, monkeypatch, tool):
    path, mcp, original = seeded
    starts = []

    def fail_start(thread):
        # A separate connection must see the obligation before any launch.
        assert SCOPE in durable(path)[2]
        assert not mcp.memory_store.connection.in_transaction
        starts.append(thread)
        raise RuntimeError("can't start new thread")

    with monkeypatch.context() as fault:
        fault.setattr(threading.Thread, "start", fail_start)
        result, trace_id = write(mcp, tool, original)
        assert result["auto_consolidation"] == {
            "status": "scheduled", "scope": SCOPE, "coalesced": False,
            "trace_count": 100, "worker_start_pending": True,
        }
        traces, edges, pending = durable(path)
        assert len(traces) == 100 and traces[trace_id] == CONTENT
        assert edges == ([(trace_id, original)] if tool == "memory_teach" else [])
        assert list(pending) == [SCOPE]
        for _ in range(3):
            report = mcp.auto_consolidation.request(SCOPE)
            assert report["coalesced"] and report["worker_start_pending"]
            assert durable(path)[2] == pending
        assert len(starts) == 4
        assert mcp.auto_consolidation.status()["queued"] == [SCOPE]
        assert mcp.auto_consolidation.status()["passes_completed"] == 0
        mcp.auto_consolidation.close()

        # Failed startup during server construction must also remain usable.
        resumed = server.create_mcp_server(path, mcp_factory=FakeMCP)
        try:
            assert resumed.auto_consolidation.status()["queued"] == [SCOPE]
            assert durable(path) == (traces, edges, pending)
        finally:
            resumed.auto_consolidation.close()
            resumed.memory_store.close()
        assert len(starts) == 5  # one attempt per call, no background retry loop

    recovered = server.create_mcp_server(path, mcp_factory=FakeMCP)
    try:
        assert recovered.auto_consolidation.wait_idle(timeout=30, include_retries=True)
        assert recovered.auto_consolidation.status()["passes_completed"] == 1
        assert recovered.auto_consolidation.last_result(SCOPE)["status"] == "completed"
        assert durable(path) == (traces, edges, {})
    finally:
        recovered.auto_consolidation.close()
        recovered.memory_store.close()


@pytest.mark.parametrize("retry", ["request", "resume"])
def test_resource_recovery_retries_without_rewriting_knowledge(tmp_path, monkeypatch, retry):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        guard = threading.RLock()
        runs = []
        scheduler = AutoConsolidationScheduler(
            lambda scope: runs.append(scope) or {},
            load_pending=lambda: store.get_kv(PENDING_KV_KEY),
            save_pending=lambda raw: store.set_kv(PENDING_KV_KEY, raw),
            persist_guard=guard,
        )

        def fail_start(thread):
            raise RuntimeError("can't start new thread")

        try:
            with monkeypatch.context() as fault:
                fault.setattr(threading.Thread, "start", fail_start)
                assert scheduler.request(SCOPE)["worker_start_pending"]
            if retry == "request":
                assert scheduler.request(SCOPE)["coalesced"]
            else:
                assert scheduler.resume() == [SCOPE]
            assert scheduler.wait_idle(timeout=5)
            assert runs == [SCOPE]
            assert store.trace_count() == 0
            assert store.get_kv(PENDING_KV_KEY) == "{}"
        finally:
            scheduler.close()
        with pytest.raises(RuntimeError, match="closed"):
            scheduler.request(SCOPE)
        assert scheduler.resume() == []
        assert store.get_kv(PENDING_KV_KEY) == "{}"


@pytest.mark.parametrize("tool", ["memory_remember", "memory_teach"])
def test_pending_save_failure_is_nonfatal_but_never_reported_scheduled(seeded, caplog, tool):
    path, mcp, original = seeded
    store = mcp.memory_store
    store.connection.execute(f"""CREATE TRIGGER reject_pending BEFORE INSERT ON kv
        WHEN NEW.key = '{PENDING_KV_KEY}'
        BEGIN SELECT RAISE(ABORT, 'pending save fault'); END""")
    result, trace_id = write(mcp, tool, original)
    traces, edges, pending = durable(path)
    assert len(traces) == 100 and traces[trace_id] == CONTENT
    assert edges == ([(trace_id, original)] if tool == "memory_teach" else [])
    assert pending is None
    assert result["auto_consolidation"] is None
    assert "could not schedule post-commit auto-consolidation" in caplog.text
    assert "pending save fault" in caplog.text
    assert mcp.auto_consolidation.status()["passes_completed"] == 0

    store.connection.execute("DROP TRIGGER reject_pending")
    assert mcp.auto_consolidation.request(SCOPE)["coalesced"]
    assert mcp.auto_consolidation.wait_idle(timeout=30)
    assert mcp.auto_consolidation.status()["passes_completed"] == 1
    assert durable(path) == (traces, edges, {})


def test_request_and_completion_order_saves_guard_before_condition(tmp_path):
    # Pause completion just before it can acquire the store guard. A new
    # request must be saved first, and the older completion must snapshot the
    # current pending set rather than overwrite it with its earlier state.
    finishing = threading.Event()
    release_second = threading.Event()
    entered_second = threading.Event()
    lock = threading.Lock()
    saves = []

    class Guard:
        def __enter__(self):
            if threading.current_thread().name == "living-memory-auto-consolidation":
                finishing.set()
            assert lock.acquire(timeout=5)

        def __exit__(self, *args):
            lock.release()

    guard = Guard()
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        def save(raw):
            saves.append(json.loads(raw))
            store.set_kv(PENDING_KV_KEY, raw)

        def run_pass(scope):
            if scope == "project:second":
                entered_second.set()
                assert release_second.wait(timeout=5)
            return {}

        scheduler = AutoConsolidationScheduler(run_pass, save_pending=save, persist_guard=guard)
        try:
            # Reenter from the first request's save callback so the worker's
            # completion is waiting for a guard held by this test thread.
            with scheduler._cond:
                scheduler.request("project:first")
                assert lock.acquire(timeout=5)
            try:
                assert finishing.wait(timeout=5)
                # Completion must release the condition before taking guard.
                assert scheduler._cond.acquire(timeout=1)
                scheduler._cond.release()
            finally:
                lock.release()
            scheduler.request("project:second")
            assert entered_second.wait(timeout=5)
            assert list(json.loads(store.get_kv(PENDING_KV_KEY))) == ["project:second"]
            release_second.set()
            assert scheduler.wait_idle(timeout=5)
            assert saves[-1] == {} and store.get_kv(PENDING_KV_KEY) == "{}"
        finally:
            release_second.set()
            scheduler.close()
