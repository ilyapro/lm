"""Optional decay outcomes over real FastMCP and disposable SQLite stores."""

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from threading import RLock
from types import SimpleNamespace

from fastmcp import Client
import pytest
from starlette.testclient import TestClient

from living_memory import consolidation, server
from living_memory.config import MemoryConfig


SCOPE = "project:optional-decay"
CONTENT = "Cobalt routing requires validated migration before release"
OLD_MARKER = "2020-01-01T00:00:00Z"
ERRORS = {"before_effect": "decay unavailable", "marker_failure": "decay marker fault"}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "3600")
    monkeypatch.setenv("LM_DECAY_TTL_ANCHOR", "created")
    monkeypatch.setenv("LM_DECAY_CONCEPT_COVERED", "0")
    monkeypatch.setenv("LM_RECALL_MAP", "0")
    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", "grounded")
    guard = RLock()
    monkeypatch.setattr(server, "RLock", lambda: guard)
    path = tmp_path / "memory.sqlite3"
    mcp = server.create_mcp_server(config=MemoryConfig(db_path=path, trace_ttl_days=7))
    store = mcp.memory_store
    old = store.append_trace("Expired maintenance fixture", {
        "scope": "project:expired-decay", "timestamp": OLD_MARKER,
    })
    current = store.append_trace(CONTENT, {"scope": SCOPE})
    store.set_kv(server._DECAY_SWEEP_KV_KEY, OLD_MARKER)
    r = SimpleNamespace(mcp=mcp, store=store, guard=guard, path=path,
                        old=old, current=current)
    try:
        yield r
    finally:
        mcp.auto_consolidation.close()
        store.close()


def durable(r):
    with sqlite3.connect(r.path) as connection:
        return {
            "expired_decayed": bool(connection.execute(
                "SELECT decayed FROM nodes WHERE id=?", (r.old.id,),
            ).fetchone()[0]),
            "marker": connection.execute(
                "SELECT value FROM kv WHERE key=?", (server._DECAY_SWEEP_KV_KEY,),
            ).fetchone()[0],
        }


def reject_marker(r):
    r.store.connection.execute(f"""CREATE TRIGGER reject_decay_marker BEFORE INSERT ON kv
        WHEN NEW.key = '{server._DECAY_SWEEP_KV_KEY}'
        BEGIN SELECT RAISE(ABORT, 'decay marker fault'); END""")


async def call(client, tool, **arguments):
    result = await client.call_tool(tool, arguments, raise_on_error=False)
    assert not result.is_error, result
    return result.data


@pytest.mark.parametrize("tool", ["memory_recall", "memory_remember", "memory_consolidate"])
@pytest.mark.parametrize("outcome", ["disabled", "recent", "success", *ERRORS])
def test_optional_outcome_and_ordinary_recovery(
    runtime, monkeypatch, caplog, record_property, tool, outcome,
):
    r = runtime
    gate_calls, apply_calls, observations = [], [], []
    sweep, apply = server._decay_sweep_if_due, server.apply_decay
    failure_active = True

    def checked_sweep(store, **kwargs):
        assert r.guard._is_owned()
        assert not store.connection.in_transaction
        gate_calls.append(True)
        return sweep(store, **kwargs)

    def checked_apply(store, **kwargs):
        apply_calls.append(True)
        if failure_active and outcome == "before_effect":
            raise RuntimeError(ERRORS[outcome])
        return apply(store, **kwargs)

    def capture(stage, response):
        observations.append({"stage": stage, "response": response, **durable(r),
                             "gate_calls": len(gate_calls), "apply_calls": len(apply_calls),
                             "failure_logs": [record.getMessage() for record in caplog.records
                                              if record.name == "living_memory.server" and
                                              record.getMessage().startswith("optional decay sweep failed")]})

    async def scenario():
        nonlocal failure_active
        async with Client(r.mcp) as client:
            # Remember must still close real feedback in the same transport session.
            with monkeypatch.context() as setup:
                setup.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
                recalled = await call(client, "memory_recall", query=CONTENT, scope=SCOPE)
            assert r.current.id in {entry["id"] for entry in recalled["results"]}
            if outcome == "disabled":
                monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
            if outcome == "recent":
                r.store.set_kv(server._DECAY_SWEEP_KV_KEY, datetime.now(UTC).isoformat())
            if outcome == "marker_failure":
                reject_marker(r)
            before = durable(r)
            monkeypatch.setattr(server, "_decay_sweep_if_due", checked_sweep)
            monkeypatch.setattr(server, "apply_decay", checked_apply)
            arguments = {
                "memory_recall": {"query": CONTENT, "scope": SCOPE},
                "memory_remember": {"content": CONTENT, "context": {"scope": SCOPE}},
                "memory_consolidate": {"scope": SCOPE},
            }[tool]
            response = await call(client, tool, **arguments)
            capture("primary", response)
            assert len(gate_calls) == 1
            assert len(apply_calls) == int(outcome not in {"disabled", "recent"})
            assert not r.store.connection.in_transaction

            if tool == "memory_recall":
                assert r.current.id in {entry["id"] for entry in response["results"]}
                assert r.store.get_recall_event(response["recall_event_id"]) is not None
            elif tool == "memory_remember":
                trace_id = response["node"]["id"]
                assert response["implicit_feedback"] == {
                    "recall_event_ids": [recalled["recall_event_id"]], "feedback_applied": True,
                }
                with sqlite3.connect(r.path) as connection:
                    assert connection.execute("SELECT content FROM nodes WHERE id=?", (trace_id,)).fetchone() == (CONTENT,)
                    assert connection.execute(
                        "SELECT feedback_trace_id FROM recall_events WHERE id=?",
                        (recalled["recall_event_id"],),
                    ).fetchone() == (trace_id,)
            else:
                assert response["traces_considered"] == 1

            if outcome in ERRORS:
                assert response["auto_decay"] == {"swept": None, "error": ERRORS[outcome]}
                assert durable(r) == {"marker": OLD_MARKER,
                                      "expired_decayed": outcome == "marker_failure"}
                repeated = await call(client, "memory_recall", query=CONTENT, scope=SCOPE)
                capture("still_failing", repeated)
                assert repeated["auto_decay"] == response["auto_decay"]
                assert len(gate_calls) == len(apply_calls) == 2
                assert durable(r)["marker"] == OLD_MARKER
                records = [record for record in caplog.records if
                           record.name == "living_memory.server" and
                           record.message == "optional decay sweep failed; completion unknown"]
                assert len(records) == 2
                assert all(record.levelname == "WARNING" and record.exc_info for record in records)
                assert ERRORS[outcome] in caplog.text
                if outcome == "marker_failure":
                    r.store.connection.execute("DROP TRIGGER reject_decay_marker")
                failure_active = False
                recovered = await call(client, "memory_recall", query=CONTENT, scope=SCOPE)
                capture("recovered", recovered)
                assert recovered["auto_decay"]["swept"] is True
                assert recovered["auto_decay"]["expired_count"] == int(outcome == "before_effect")
                assert durable(r)["expired_decayed"]
                assert durable(r)["marker"] == recovered["auto_decay"]["last_decay_sweep_at"]
                completed = durable(r)
                skipped = await call(client, "memory_recall", query=CONTENT, scope=SCOPE)
                capture("recent_after_recovery", skipped)
                assert skipped["auto_decay"] is None
                assert durable(r) == completed
                assert len(gate_calls) == 4
                assert len(apply_calls) == 3
            elif outcome == "success":
                report = response["auto_decay"]
                assert set(report) == {"swept", "decayed_count", "expired_count",
                                       "superseded_count", "duration_ms", "last_decay_sweep_at"}
                assert report["swept"] is True
                assert report["decayed_count"] == report["expired_count"] == 1
                assert report["superseded_count"] == 0
                assert report["duration_ms"] >= 0
                assert durable(r) == {"marker": report["last_decay_sweep_at"], "expired_decayed": True}
            else:
                assert response["auto_decay"] is None
                assert durable(r) == before
            if outcome not in ERRORS:
                assert "optional decay sweep failed" not in caplog.text

    asyncio.run(scenario())
    record_property("native_observations", json.dumps(observations))


@pytest.mark.parametrize("route", ["forced", "admin"])
@pytest.mark.parametrize("fault", ERRORS)
def test_explicit_sweep_failure_remains_an_error(runtime, monkeypatch, route, fault):
    r = runtime
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
    if fault == "marker_failure":
        reject_marker(r)
    else:
        def fail(*args, **kwargs):
            raise RuntimeError(ERRORS[fault])
        monkeypatch.setattr(server, "apply_decay", fail)
    if route == "forced":
        with pytest.raises((RuntimeError, sqlite3.IntegrityError), match=ERRORS[fault]):
            server._decay_sweep_if_due(r.store, force=True)
    else:
        r.mcp.auth_token_state.rotate("fixture-decay-token")
        with TestClient(r.mcp.http_app()) as client:
            response = client.post("/admin/decay-sweep", headers={
                "Authorization": "Bearer fixture-decay-token",
            })
        assert response.status_code == 500
        assert response.json()["ok"] is False
        assert response.json()["error"] == ERRORS[fault]
        assert "swept" not in response.json()
    assert durable(r) == {"marker": OLD_MARKER, "expired_decayed": fault == "marker_failure"}


def test_requested_consolidation_error_remains_a_tool_error(runtime, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("requested consolidation failed")

    monkeypatch.setattr(consolidation.ConsolidationService, "memory_consolidate", fail)

    async def scenario():
        async with Client(runtime.mcp) as client:
            result = await client.call_tool("memory_consolidate", {"scope": SCOPE, "force": True},
                                            raise_on_error=False)
            assert result.is_error
            assert "requested consolidation failed" in result.content[0].text

    asyncio.run(scenario())
