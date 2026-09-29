"""A due consolidation pass runs beside the write path, not inside it.

The red case on the synchronous design: the write that makes a pass due
answers only after the pass, and a recall issued meanwhile waits for it too.
The pass here has an artificial duration so the gap is unambiguous.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import json
import threading
import time
from typing import Any

import pytest

from living_memory import consolidation
from living_memory.auto_consolidation import PENDING_KV_KEY, AutoConsolidationScheduler
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore

SCOPE = "project:bg"
PASS_STEPS = 100
PASS_STEP_SECONDS = 0.1  # a 10 s pass, taken in short guarded steps
WRITE_BUDGET_SECONDS = 5.0  # half the pass: headroom for a loaded host
RECALL_BUDGET_SECONDS = 5.0


class FakeMCP:
    def __init__(self, name: str, instructions: str | None = None) -> None:
        self.name = name
        self.instructions = instructions
        self.tools: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def tool(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.tools[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        return decorate if func is None else decorate(func)

    def resource(self, uri: str, **_kwargs: Any) -> Any:
        def decorate(func: Any) -> Any:
            self.resources[uri] = func
            return func

        return decorate

    def prompt(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.prompts[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        return decorate if func is None else decorate(func)


def _slow_pass(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """Stretch every pass by PASS_STEPS guarded steps; return a started flag."""

    started = threading.Event()
    real = consolidation.memory_consolidate

    def slow(store: MemoryStore, **kwargs: Any) -> consolidation.ConsolidationResult:
        step = kwargs.get("guard") or nullcontext
        started.set()
        for _ in range(PASS_STEPS):
            with step():
                time.sleep(PASS_STEP_SECONDS)
        return real(store, **kwargs)

    monkeypatch.setattr(consolidation, "memory_consolidate", slow)
    return started


def test_due_write_returns_before_the_pass_and_recall_is_served_during_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LM_AUTO_CONSOLIDATE_POLICY", raising=False)
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    started = _slow_pass(monkeypatch)
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    for index in range(98):
        store.append_trace(
            f"harbor crane maintenance window checklist item {index}",
            {"scope": SCOPE, "agent": "agent-a"},
        )
    # Warm both paths first, so the timings below measure the pass's effect
    # and not one-time loading; the 99th trace is not due.
    assert mcp.tools["memory_remember"](
        "harbor crane maintenance window checklist warm-up item",
        {"scope": SCOPE, "agent": "agent-a"},
    )["auto_consolidation"] is None
    mcp.tools["memory_recall"]("harbor crane checklist", scope=SCOPE)

    began = time.perf_counter()
    response = mcp.tools["memory_remember"](
        "harbor crane maintenance window checklist final item",
        {"scope": SCOPE, "agent": "agent-a"},
    )
    write_seconds = time.perf_counter() - began

    # The hundredth trace made the pass due; the write did not wait for it.
    assert write_seconds < WRITE_BUDGET_SECONDS, write_seconds
    assert response["auto_consolidation"] == {
        "status": "scheduled",
        "scope": SCOPE,
        "coalesced": False,
        "trace_count": 100,
    }
    assert store.get_node(response["node"]["id"]) is not None

    assert started.wait(timeout=5)
    began = time.perf_counter()
    recalled = mcp.tools["memory_recall"]("harbor crane checklist", scope=SCOPE)
    recall_seconds = time.perf_counter() - began
    assert mcp.auto_consolidation.status()["running"] == SCOPE  # still mid-pass
    assert recall_seconds < RECALL_BUDGET_SECONDS, recall_seconds
    assert recalled["results"]

    assert mcp.auto_consolidation.wait_idle(timeout=60)
    finished = mcp.auto_consolidation.last_result(SCOPE)
    assert finished is not None and finished["status"] == "completed"
    assert finished["summary"]["traces_considered"] >= 100
    assert finished["summary"]["concepts_created"]  # the pass really ran
    assert store.get_kv(PENDING_KV_KEY) == "{}"


def test_teach_schedules_the_pass_its_trace_makes_due(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LM_AUTO_CONSOLIDATE_POLICY", raising=False)
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    _slow_pass(monkeypatch)
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    first = None
    for index in range(98):
        node = store.append_trace(
            f"tide gauge calibration note {index}", {"scope": SCOPE, "agent": "agent-a"}
        )
        first = first or node
    mcp.tools["memory_remember"]("tide gauge warm-up note", {"scope": SCOPE, "agent": "a"})

    began = time.perf_counter()
    taught = mcp.tools["memory_teach"](first.id, "tide gauge calibration uses the harbor datum")
    assert time.perf_counter() - began < WRITE_BUDGET_SECONDS
    assert taught["auto_consolidation"]["status"] == "scheduled"
    assert mcp.auto_consolidation.wait_idle(timeout=60)
    assert mcp.auto_consolidation.last_result(SCOPE)["status"] == "completed"


def _gate() -> tuple[threading.Event, threading.Event]:
    return threading.Event(), threading.Event()


def test_requests_during_a_pass_coalesce_into_one_follow_up() -> None:
    entered, release = _gate()
    runs: list[str] = []

    def run_pass(scope: str) -> dict[str, Any]:
        runs.append(scope)
        if len(runs) == 1:
            entered.set()
            assert release.wait(timeout=5)
        return {"scope": scope}

    scheduler = AutoConsolidationScheduler(run_pass)
    first = scheduler.request("project:a")
    assert first["coalesced"] is False
    assert entered.wait(timeout=5)
    later = [scheduler.request("project:a") for _ in range(5)]
    assert all(report["coalesced"] for report in later)
    release.set()
    assert scheduler.wait_idle(timeout=5)
    assert runs == ["project:a", "project:a"]  # the running pass plus one
    assert scheduler.status()["passes_completed"] == 2
    scheduler.close()


def test_one_pass_at_a_time_across_scopes() -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    def run_pass(scope: str) -> dict[str, Any]:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {}

    scheduler = AutoConsolidationScheduler(run_pass)
    for scope in ("project:a", "project:b", "project:c", "project:a"):
        scheduler.request(scope)
    assert scheduler.wait_idle(timeout=5)
    assert peak == 1
    scheduler.close()


def test_failed_pass_is_retried_and_stays_pending_across_restart(tmp_path: Path) -> None:
    kv: dict[str, str] = {}
    attempts: list[str] = []

    def failing(scope: str) -> dict[str, Any]:
        attempts.append(scope)
        raise RuntimeError("disk on fire")

    scheduler = AutoConsolidationScheduler(
        failing,
        load_pending=lambda: kv.get(PENDING_KV_KEY),
        save_pending=lambda raw: kv.__setitem__(PENDING_KV_KEY, raw),
        retry_delays=(0.05,),
    )
    scheduler.request("project:a")
    assert scheduler.wait_idle(timeout=5, include_retries=True)
    assert attempts == ["project:a", "project:a"]  # first run + one retry
    last = scheduler.last_result("project:a")
    assert last["status"] == "failed" and "disk on fire" in last["error"]
    # Retries exhausted: the scope is still marked pending for the next start.
    assert "project:a" in json.loads(kv[PENDING_KV_KEY])
    scheduler.close()

    resumed: list[str] = []
    restarted = AutoConsolidationScheduler(
        lambda scope: resumed.append(scope) or {"ok": True},
        load_pending=lambda: kv.get(PENDING_KV_KEY),
        save_pending=lambda raw: kv.__setitem__(PENDING_KV_KEY, raw),
    )
    assert restarted.resume() == ["project:a"]
    assert restarted.wait_idle(timeout=5)
    assert resumed == ["project:a"]
    assert json.loads(kv[PENDING_KV_KEY]) == {}
    restarted.close()


def test_server_start_resumes_a_scope_left_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    db = tmp_path / "memory.sqlite3"
    with MemoryStore(db) as store:
        for index in range(12):
            store.append_trace(
                f"ferry timetable change notice {index}", {"scope": SCOPE, "agent": "a"}
            )
        store.set_kv(PENDING_KV_KEY, json.dumps({SCOPE: "2026-09-29T00:00:00Z"}))

    mcp = create_mcp_server(db, mcp_factory=FakeMCP)
    assert mcp.auto_consolidation.wait_idle(timeout=60)
    assert mcp.auto_consolidation.last_result(SCOPE)["status"] == "completed"
    assert json.loads(mcp.memory_store.get_kv(PENDING_KV_KEY)) == {}


def test_explicit_consolidate_stays_synchronous_and_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    for index in range(12):
        mcp.memory_store.append_trace(
            f"lighthouse lamp replacement schedule {index}", {"scope": SCOPE, "agent": "a"}
        )
    payload = mcp.tools["memory_consolidate"](scope=SCOPE, force=True)
    assert payload["traces_considered"] == 12
    assert payload["concepts_created"]
    assert isinstance(payload["concepts_created"][0], dict)  # full nodes, not ids
    assert mcp.auto_consolidation.status()["passes_completed"] == 0
