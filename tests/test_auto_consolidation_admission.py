"""A pending save admits work to both existing and newly started workers."""

import json
import sqlite3
import threading

import pytest

import living_memory.server as server
from living_memory.auto_consolidation import AutoConsolidationScheduler, PENDING_KV_KEY
from living_memory.storage import MemoryStore
from test_lookup import FakeMCP


FIRST = "project:first"
SECOND = "project:second"


def pending(path):
    with sqlite3.connect(path) as reader:
        row = reader.execute("SELECT value FROM kv WHERE key=?", (PENDING_KV_KEY,)).fetchone()
    return json.loads(row[0]) if row else {}


def reject_saves(store):
    store.connection.execute(f"""CREATE TRIGGER reject_pending BEFORE INSERT ON kv
        WHEN NEW.key = '{PENDING_KV_KEY}'
        BEGIN SELECT RAISE(ABORT, 'pending save fault'); END""")


class HeldPasses:
    def __init__(self, monkeypatch, scheduler, path, *, first_fails):
        self.entered = threading.Event()
        self.release_first = threading.Event()
        self.progressed = threading.Event()
        self.entered_next = threading.Event()
        self.release_next = threading.Event()
        self.runs = []
        self.snapshots = []
        real_wait = scheduler._cond.wait

        def wait(timeout=None):
            # The worker has scanned the queue. This is positive evidence of
            # progress after the first pass, without a timed negative assertion.
            if self.release_first.is_set() and threading.current_thread() is scheduler._thread:
                self.progressed.set()
            return real_wait(timeout)

        def run(scope):
            self.runs.append(scope)
            self.snapshots.append(pending(path))
            if len(self.runs) == 1:
                self.entered.set()
                assert self.release_first.wait(timeout=10)
                if first_fails:
                    raise RuntimeError("first pass fault")
            else:
                self.entered_next.set()
                self.progressed.set()
                assert self.release_next.wait(timeout=10)
            return {}

        monkeypatch.setattr(scheduler, "_run_pass", run)
        monkeypatch.setattr(scheduler._cond, "wait", wait)

    def finish_first(self):
        self.release_first.set()
        assert self.progressed.wait(timeout=5)

    def release(self):
        self.release_first.set()
        self.release_next.set()


@pytest.mark.parametrize("same_scope", [False, True])
@pytest.mark.parametrize("first_fails", [False, True])
def test_live_worker_admits_failed_request_only_after_save(tmp_path, monkeypatch, same_scope, first_fails):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store:
        scheduler = AutoConsolidationScheduler(
            lambda scope: {},
            save_pending=lambda raw: store.set_kv(PENDING_KV_KEY, raw),
            retry_delays=(3600,),
        )
        held = HeldPasses(monkeypatch, scheduler, path, first_fails=first_fails)
        scope = FIRST if same_scope else SECOND
        try:
            scheduler.request(FIRST)
            assert held.entered.wait(timeout=5)
            thread = scheduler._thread
            reject_saves(store)
            with pytest.raises(sqlite3.IntegrityError, match="pending save fault"):
                scheduler.request(scope)
            assert set(pending(path)) == {FIRST}
            if first_fails:
                # Exactly one save fails; the accepted pass then enters backoff.
                store.connection.execute("DROP TRIGGER reject_pending")
            held.finish_first()
            assert held.runs == [FIRST]
            if not first_fails:
                # A failed completion save must not admit the queued request.
                with pytest.raises(sqlite3.IntegrityError, match="pending save fault"):
                    scheduler.retry_pending()
                store.connection.execute("DROP TRIGGER reject_pending")
            queue = dict(scheduler._queue)
            attempts = dict(scheduler._attempts)
            scheduler.retry_pending()
            assert held.entered_next.wait(timeout=5)
            assert held.runs == [FIRST, scope]
            assert scope in held.snapshots[-1]
            assert scheduler._thread is thread
            if first_fails and not same_scope:
                # Admitting another scope neither expedites nor resets A's retry.
                assert scheduler._queue[FIRST] == queue[FIRST]
                assert scheduler._attempts == attempts
            held.release_next.set()
            assert scheduler.wait_idle(timeout=5)
            assert set(pending(path)) == ({FIRST} if first_fails and not same_scope else set())
        finally:
            held.release()
            scheduler.close()


def test_completion_save_admits_outstanding_request(tmp_path, monkeypatch):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store:
        scheduler = AutoConsolidationScheduler(
            lambda scope: {}, save_pending=lambda raw: store.set_kv(PENDING_KV_KEY, raw),
        )
        held = HeldPasses(monkeypatch, scheduler, path, first_fails=False)
        try:
            scheduler.request(FIRST)
            assert held.entered.wait(timeout=5)
            reject_saves(store)
            with pytest.raises(sqlite3.IntegrityError, match="pending save fault"):
                scheduler.request(SECOND)
            store.connection.execute("DROP TRIGGER reject_pending")
            held.finish_first()
            assert held.entered_next.wait(timeout=5)
            assert held.runs == [FIRST, SECOND]
            assert set(held.snapshots[-1]) == {SECOND}
            held.release_next.set()
            assert scheduler.wait_idle(timeout=5)
            assert pending(path) == {}
        finally:
            held.release()
            scheduler.close()


@pytest.mark.parametrize("unsaved_request", [False, True])
def test_settlement_recovery_preserves_backoff_and_unsaved_scopes(tmp_path, monkeypatch, unsaved_request):
    path = tmp_path / "memory.sqlite3"
    third = "project:third"
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    runs, snapshots = [], []

    def run(scope):
        runs.append(scope)
        snapshots.append(pending(path))
        if scope == FIRST:
            raise RuntimeError("pass fault")
        if scope == SECOND:
            entered.set()
            assert release.wait(timeout=10)
        return {}

    with MemoryStore(path) as store:
        scheduler = AutoConsolidationScheduler(
            run, save_pending=lambda raw: store.set_kv(PENDING_KV_KEY, raw),
            retry_delays=(3600,),
        )
        finish = scheduler._finish_ok

        def finish_ok(scope, *args):
            finish(scope, *args)
            if scope == SECOND:
                finished.set()

        monkeypatch.setattr(scheduler, "_finish_ok", finish_ok)
        try:
            scheduler.request(FIRST)
            assert scheduler.wait_idle(timeout=5)
            retry, attempts, last = scheduler._queue[FIRST], dict(scheduler._attempts), scheduler.last_result(FIRST)
            scheduler.request(SECOND)
            assert entered.wait(timeout=5)
            reject_saves(store)
            if unsaved_request:
                with pytest.raises(sqlite3.IntegrityError, match="pending save fault"):
                    scheduler.request(third)
            release.set()
            assert finished.wait(timeout=5)
            assert scheduler.last_result(SECOND)["status"] == "completed"
            assert scheduler.status()["pending_save"] is True
            assert set(pending(path)) == {FIRST, SECOND}
            with pytest.raises(sqlite3.IntegrityError, match="pending save fault"):
                scheduler.retry_pending()
            assert runs == [FIRST, SECOND]
            store.connection.execute("DROP TRIGGER reject_pending")
            scheduler.retry_pending()
            assert scheduler.wait_idle(timeout=5)
            assert runs == [FIRST, SECOND] + ([third] if unsaved_request else [])
            if unsaved_request:
                assert set(snapshots[-1]) == {FIRST, third}
            assert set(pending(path)) == {FIRST}
            assert scheduler._queue[FIRST] == retry
            assert scheduler._attempts == attempts
            assert scheduler.last_result(FIRST) == last
            assert scheduler.status()["pending_save"] is False
            assert not scheduler.wait_idle(timeout=0, include_retries=True)
        finally:
            release.set()
            scheduler.close()

        resumed = []
        cold = AutoConsolidationScheduler(
            lambda scope: resumed.append(scope) or {},
            load_pending=lambda: store.get_kv(PENDING_KV_KEY),
            save_pending=lambda raw: store.set_kv(PENDING_KV_KEY, raw),
        )
        try:
            assert cold.resume() == [FIRST]
            assert cold.wait_idle(timeout=5, include_retries=True)
            assert resumed == [FIRST] and pending(path) == {}
        finally:
            cold.close()


@pytest.mark.parametrize("accepted", ["queued", "retry"])
def test_failed_request_coalesces_into_an_accepted_pass(tmp_path, monkeypatch, accepted):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store:
        scheduler = AutoConsolidationScheduler(
            lambda scope: {}, save_pending=lambda raw: store.set_kv(PENDING_KV_KEY, raw),
            retry_delays=(3600,) if accepted == "queued" else (0,),
        )
        held = HeldPasses(monkeypatch, scheduler, path, first_fails=True)
        scope = SECOND if accepted == "queued" else FIRST
        try:
            scheduler.request(FIRST)
            assert held.entered.wait(timeout=5)
            if accepted == "queued":
                scheduler.request(scope)
            reject_saves(store)
            with pytest.raises(sqlite3.IntegrityError, match="pending save fault"):
                scheduler.request(scope)
            held.finish_first()
            # Accepted work can run despite the unsaved coalesced request.
            assert held.entered_next.wait(timeout=5)
            assert held.runs == [FIRST, scope]
            assert scope in held.snapshots[-1]
            assert scope not in scheduler.status()["queued"]
            store.connection.execute("DROP TRIGGER reject_pending")
            held.release_next.set()
            assert scheduler.wait_idle(timeout=5)
            scheduler.retry_pending()
            assert scheduler.wait_idle(timeout=5)
            assert held.runs == [FIRST, scope]
            assert set(pending(path)) == ({FIRST} if accepted == "queued" else set())
        finally:
            held.release()
            scheduler.close()


@pytest.mark.parametrize("tool", ["memory_remember", "memory_teach"])
@pytest.mark.parametrize("next_scope", [SECOND, "project:other"])
@pytest.mark.parametrize("first_fails", [False, True])
def test_public_write_commits_feedback_and_recovers_live_admission(
    tmp_path, monkeypatch, caplog, tool, next_scope, first_fails,
):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "fixed")
    monkeypatch.setattr(server, "_transport_session_id", lambda: "admission-transport")
    path = tmp_path / "memory.sqlite3"
    mcp = server.create_mcp_server(path, mcp_factory=FakeMCP)
    store = mcp.memory_store
    scheduler = mcp.auto_consolidation
    monkeypatch.setattr(scheduler, "_retry_delays", (3600,))
    held = HeldPasses(monkeypatch, scheduler, path, first_fails=first_fails)
    context = {"scope": SECOND, "agent": "fixture", "session_id": "admission-session"}
    try:
        for index in range(99):
            store.append_trace(f"Harbor datum calibration measurement {index}", context)
        recalled = mcp.tools["memory_recall"](
            "Harbor datum calibration", scope=SECOND, ambient_context=context,
            max_results=1, depth=0,
        )
        original = recalled["results"][0]["id"]
        event_id = recalled["recall_event_id"]
        content = "Harbor datum calibration requires a fresh measurement"

        def write(scope, *, used=None):
            kwargs = {"context": {**context, "scope": scope}, "used": used}
            if tool == "memory_teach":
                result = mcp.tools[tool](original, content, **kwargs)
                return result, result["corrective_trace"]["id"]
            result = mcp.tools[tool](content, **kwargs)
            return result, result["node"]["id"]

        scheduler.request(FIRST)
        assert held.entered.wait(timeout=5)
        thread = scheduler._thread
        reject_saves(store)
        result, trace_id = write(SECOND, used=[original])
        assert result["auto_consolidation"] is None
        assert result["feedback_marks"]["accepted"] == 1
        assert "could not schedule post-commit auto-consolidation" in caplog.text
        assert "pending save fault" in caplog.text
        assert not store.connection.in_transaction
        with sqlite3.connect(path) as reader:
            assert reader.execute("SELECT content FROM nodes WHERE id=?", (trace_id,)).fetchone() == (content,)
            assert reader.execute(
                "SELECT feedback_trace_id FROM recall_events WHERE id=? AND feedback_applied_at IS NOT NULL",
                (event_id,),
            ).fetchone() == (trace_id,)
            assert reader.execute(
                "SELECT node_id, source_id, accepted FROM recall_feedback_marks",
            ).fetchall() == [(original, trace_id, 1)]
            edges = reader.execute(
                "SELECT source_id, target_id FROM connections WHERE type='supersedes'",
            ).fetchall()
            assert edges == ([(trace_id, original)] if tool == "memory_teach" else [])
        assert set(pending(path)) == {FIRST}
        held.finish_first()
        assert held.runs == [FIRST]

        store.connection.execute("DROP TRIGGER reject_pending")
        healthy, healthy_id = write(next_scope)
        assert healthy["auto_consolidation"] is None  # #101 here, or #1 elsewhere
        assert held.entered_next.wait(timeout=5)
        assert held.runs == [FIRST, SECOND]
        assert SECOND in held.snapshots[-1]
        assert scheduler._thread is thread
        with sqlite3.connect(path) as reader:
            assert reader.execute("SELECT content FROM nodes WHERE id=?", (healthy_id,)).fetchone() == (content,)
        held.release_next.set()
        assert scheduler.wait_idle(timeout=5)
        assert set(pending(path)) == ({FIRST} if first_fails else set())
    finally:
        held.release()
        scheduler.close()
        store.close()
