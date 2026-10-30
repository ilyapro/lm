"""Optional worker startup cannot change a committed knowledge outcome."""

import json
import sqlite3
import threading
from unittest.mock import Mock

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


def write(mcp, tool, original, *, content=CONTENT, scope=SCOPE):
    if tool == "memory_teach":
        result = mcp.tools[tool](original, content, context={"scope": scope})
        return result, result["corrective_trace"]["id"]
    result = mcp.tools[tool](content, context={"scope": scope})
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
@pytest.mark.parametrize("next_scope", [SCOPE, "project:other"])
def test_next_ordinary_write_resumes_failed_start(seeded, monkeypatch, tool, next_scope):
    path, mcp, original = seeded
    scheduler = mcp.auto_consolidation
    starts = []
    real_start = threading.Thread.start

    def start(thread):
        traces, edges, pending = durable(path)
        assert SCOPE in pending
        assert not mcp.memory_store.connection.in_transaction
        assert len(traces) == 100 + len(starts)
        assert len(edges) == (len(starts) + 1 if tool == "memory_teach" else 0)
        starts.append(thread)
        if len(starts) == 1:
            raise RuntimeError("can't start new thread")
        return real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    failed, first_id = write(mcp, tool, original)
    assert failed["auto_consolidation"]["worker_start_pending"]
    pending = durable(path)[2]
    assert scheduler.status()["passes_completed"] == 0

    healthy, second_id = write(
        mcp, tool, original, content=CONTENT + " with tide correction", scope=next_scope,
    )
    assert healthy["auto_consolidation"] is None  # no newly due pass
    assert scheduler.wait_idle(timeout=30, include_retries=True)
    assert len(starts) == 2
    assert scheduler.status()["passes_completed"] == 1
    assert scheduler.status()["queued"] == []
    assert scheduler.last_result(SCOPE)["status"] == "completed"
    traces, edges, settled = durable(path)
    assert len(traces) == 101 and traces[first_id] == CONTENT
    assert traces[second_id] == CONTENT + " with tide correction"
    assert set(edges) == ({(first_id, original), (second_id, original)} if tool == "memory_teach" else set())
    assert list(pending) == [SCOPE] and settled == {}


@pytest.mark.parametrize("tool", ["memory_remember", "memory_teach"])
@pytest.mark.parametrize("repeat_fault", [False, True])
def test_pending_save_failure_recovers_on_ordinary_write(seeded, monkeypatch, caplog, tool, repeat_fault):
    path, mcp, original = seeded
    store = mcp.memory_store
    store.connection.execute(f"""CREATE TRIGGER reject_pending BEFORE INSERT ON kv
        WHEN NEW.key = '{PENDING_KV_KEY}'
        BEGIN SELECT RAISE(ABORT, 'pending save fault'); END""")
    starts = []
    real_start = threading.Thread.start

    def start(thread):
        # In-memory pending membership is not a successful save. Launch only
        # once a separate reader sees both the pending set and healthy write.
        traces, edges, pending = durable(path)
        assert SCOPE in pending
        assert len(traces) == 101 + int(repeat_fault)
        assert not store.connection.in_transaction
        starts.append(thread)
        return real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    result, trace_id = write(mcp, tool, original)
    traces, edges, pending = durable(path)
    assert len(traces) == 100 and traces[trace_id] == CONTENT
    assert edges == ([(trace_id, original)] if tool == "memory_teach" else [])
    assert pending is None
    assert result["auto_consolidation"] is None
    assert "could not schedule post-commit auto-consolidation" in caplog.text
    assert "pending save fault" in caplog.text
    assert mcp.auto_consolidation.status()["passes_completed"] == 0
    assert starts == []

    if repeat_fault:
        failed, failed_id = write(mcp, tool, original, content=CONTENT + " during save fault")
        assert failed["auto_consolidation"] is None
        assert durable(path)[0][failed_id] == CONTENT + " during save fault"
        assert durable(path)[2] is None and starts == []

    store.connection.execute("DROP TRIGGER reject_pending")
    recovered, recovered_id = write(mcp, tool, original, content=CONTENT + " after save recovery")
    assert recovered["auto_consolidation"] is None
    assert mcp.auto_consolidation.wait_idle(timeout=30)
    assert mcp.auto_consolidation.status()["passes_completed"] == 1
    assert len(starts) == 1
    traces, edges, pending = durable(path)
    assert len(traces) == 101 + int(repeat_fault)
    assert traces[trace_id] == CONTENT
    assert traces[recovered_id] == CONTENT + " after save recovery"
    expected = {(trace_id, original), (recovered_id, original)}
    if repeat_fault:
        expected.add((failed_id, original))
    assert set(edges) == (expected if tool == "memory_teach" else set())
    assert pending == {}


@pytest.mark.parametrize("tool", ["memory_remember", "memory_teach"])
def test_ordinary_write_without_pending_only_checks_in_memory(seeded, monkeypatch, tool):
    path, mcp, original = seeded
    scheduler = mcp.auto_consolidation
    calls = []
    for name in ("request", "_save_pending", "_load_pending", "_ensure_worker"):
        spy = Mock(wraps=getattr(scheduler, name))
        monkeypatch.setattr(scheduler, name, spy)
        calls.append(spy)
    result, trace_id = write(mcp, tool, original, scope="project:other")
    assert result["auto_consolidation"] is None
    assert durable(path)[0][trace_id] == CONTENT
    assert durable(path)[2] is None
    assert scheduler.status()["passes_completed"] == 0
    for spy in calls:
        spy.assert_not_called()


@pytest.mark.parametrize("tool", ["memory_remember", "memory_teach"])
@pytest.mark.parametrize("coalesced", [False, True])
def test_ordinary_write_adds_no_pass_to_running_work(seeded, monkeypatch, tool, coalesced):
    path, mcp, original = seeded
    scheduler = mcp.auto_consolidation
    entered, release = threading.Event(), threading.Event()
    runs = []

    def run_pass(scope):
        runs.append(scope)
        entered.set()
        assert release.wait(timeout=10)
        return {}

    monkeypatch.setattr(scheduler, "_run_pass", run_pass)
    try:
        first, _ = write(mcp, tool, original)
        assert first["auto_consolidation"]["status"] == "scheduled"
        assert entered.wait(timeout=5)
        thread = scheduler._thread
        if coalesced:
            for index in range(99):
                mcp.memory_store.append_trace(f"Next calibration fixture {index}", {"scope": SCOPE})
            due, _ = write(mcp, tool, original, content=CONTENT + " at next boundary")
            assert due["auto_consolidation"]["coalesced"]
            assert scheduler.status()["queued"] == [SCOPE]
        with monkeypatch.context() as ordinary:
            calls = []
            for name in ("request", "_save_pending", "_load_pending", "_ensure_worker"):
                spy = Mock(wraps=getattr(scheduler, name))
                ordinary.setattr(scheduler, name, spy)
                calls.append(spy)
            for index in range(3):
                result, _ = write(mcp, tool, original, content=f"{CONTENT} ordinary {index}")
                assert result["auto_consolidation"] is None
            for spy in calls:
                spy.assert_not_called()
        assert scheduler._thread is thread
    finally:
        release.set()
    assert scheduler.wait_idle(timeout=5)
    assert runs == [SCOPE] * (2 if coalesced else 1)
    assert durable(path)[2] == {}


@pytest.mark.parametrize("tool", ["memory_remember", "memory_teach"])
@pytest.mark.parametrize("delays", [(), (3600,)])
def test_ordinary_write_preserves_pass_retry_policy(seeded, monkeypatch, tool, delays):
    _, mcp, original = seeded
    scheduler = mcp.auto_consolidation

    def fail_pass(scope):
        raise RuntimeError("pass unavailable")

    monkeypatch.setattr(scheduler, "_run_pass", fail_pass)
    monkeypatch.setattr(scheduler, "_retry_delays", delays)
    write(mcp, tool, original)
    assert scheduler.wait_idle(timeout=5)
    queue, attempts, last = dict(scheduler._queue), dict(scheduler._attempts), scheduler.last_result(SCOPE)
    assert last["status"] == "failed" and last["attempt"] == 1
    result, _ = write(mcp, tool, original, content=CONTENT + " during pass failure")
    assert result["auto_consolidation"] is None
    assert scheduler._queue == queue
    assert scheduler._attempts == attempts
    assert scheduler.last_result(SCOPE) == last


@pytest.mark.parametrize("next_scope", ["project:first", "project:second"])
def test_request_and_completion_order_saves_guard_before_condition(tmp_path, next_scope):
    # Pause completion just before it can acquire the store guard. A new
    # request must be saved first, and the older completion must snapshot the
    # current pending set rather than overwrite it with its earlier state.
    finishing = threading.Event()
    release_second = threading.Event()
    entered_second = threading.Event()
    lock = threading.RLock()
    saves = []
    runs = []

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
            runs.append(scope)
            if len(runs) == 2:
                entered_second.set()
                assert release_second.wait(timeout=5)
            return {}

        scheduler = AutoConsolidationScheduler(run_pass, save_pending=save, persist_guard=guard)
        try:
            with guard:
                scheduler.request("project:first")
                assert finishing.wait(timeout=5)
                # Completion must release the condition before taking guard.
                assert scheduler._cond.acquire(timeout=1)
                scheduler._cond.release()
                scheduler.request(next_scope)
            assert entered_second.wait(timeout=5)
            assert [set(saved) for saved in saves] == [
                {"project:first"}, {next_scope}, {next_scope},
            ]
            assert list(json.loads(store.get_kv(PENDING_KV_KEY))) == [next_scope]
            release_second.set()
            assert scheduler.wait_idle(timeout=5)
            assert saves[-1] == {} and store.get_kv(PENDING_KV_KEY) == "{}"
            assert runs == ["project:first", next_scope]
        finally:
            release_second.set()
            scheduler.close()
