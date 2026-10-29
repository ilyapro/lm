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
