"""Independent servers persist and settle only the passes their writes cover."""

import json
import threading
from contextlib import ExitStack

import pytest

from living_memory.auto_consolidation import PENDING_KV_KEY
from living_memory.server import create_mcp_server
from test_lookup import FakeMCP


def open_server(stack, path):
    server = create_mcp_server(path, mcp_factory=FakeMCP)
    stack.callback(server.memory_store.close)
    stack.callback(server.auto_consolidation.close)
    return server


def write_interval(server, scope, first):
    for index in range(first, first + 5):
        result = server.tools["memory_remember"](
            f"Harbor crane calibration observation {index}", {"scope": scope}
        )
        assert bool(result["auto_consolidation"]) == (index % 5 == 0)
    return result


def check_recovery(path, scope, expected_count, latest):
    with ExitStack() as stack:
        recovered = open_server(stack, path)
        scheduler = recovered.auto_consolidation
        assert scheduler.wait_idle(timeout=30, include_retries=True)
        assert scheduler.status()["passes_completed"] == 1
        summary = scheduler.last_result(scope)["summary"]
        assert summary["traces_considered"] == expected_count
        concepts = [recovered.memory_store.get_node(node) for node in summary["concepts_created"]]
        assert any(latest in concept.source_traces for concept in concepts)
        assert recovered.memory_store.get_kv(PENDING_KV_KEY) == "{}"
        idle_write = recovered.tools["memory_remember"]("Fresh harbor observation", {"scope": scope})
        assert idle_write["auto_consolidation"] is None
        assert scheduler.wait_idle(timeout=5, include_retries=True)
        assert scheduler.status()["passes_completed"] == 1


@pytest.mark.parametrize("same_scope", [False, True])
@pytest.mark.parametrize("finish_first", [0, 1])
def test_other_writer_survives_completion_and_restart(tmp_path, monkeypatch, same_scope, finish_first):
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "adaptive")
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    path = tmp_path / "memory.sqlite3"
    scopes = ["project:harbor", "project:harbor" if same_scope else "project:canal"]
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    unfinished = 1 - finish_first

    def pass_for(writer):
        def run(scope):
            assert scope == scopes[writer]
            entered[writer].set()
            assert release[writer].wait(15)
            if writer == unfinished:
                raise RuntimeError("synthetic interruption")
            return {}
        return run

    with ExitStack() as stack:
        servers = [open_server(stack, path) for _ in scopes]
        stack.callback(lambda: [gate.set() for gate in release])
        for i, server in enumerate(servers):
            monkeypatch.setattr(server.auto_consolidation, "_run_pass", pass_for(i))
            result = write_interval(server, scopes[i], 6 if i and same_scope else 1)
            if i == unfinished:
                latest = result["node"]["id"]
            assert entered[i].wait(5)
        store = servers[0].memory_store
        assert set(json.loads(store.get_kv(PENDING_KV_KEY))) == set(scopes)
        release[finish_first].set()
        assert servers[finish_first].auto_consolidation.wait_idle(timeout=5, include_retries=True)
        assert servers[unfinished].auto_consolidation.status()["running"] == scopes[unfinished]
        assert set(json.loads(store.get_kv(PENDING_KV_KEY))) == {scopes[unfinished]}
        release[unfinished].set()
        scheduler = servers[unfinished].auto_consolidation
        assert scheduler.wait_idle(timeout=5)
        assert scheduler.last_result(scopes[unfinished])["status"] == "failed"

    check_recovery(path, scopes[unfinished], 10 if same_scope else 5, latest)


def test_peer_resuming_an_older_request_cannot_settle_a_followup(tmp_path, monkeypatch):
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "adaptive")
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    path = tmp_path / "memory.sqlite3"
    scope = "project:harbor"
    entered, release = threading.Event(), threading.Event()

    def interrupted(scope):
        raise RuntimeError("synthetic interruption")

    with ExitStack() as stack:
        writer = open_server(stack, path)
        monkeypatch.setattr(writer.auto_consolidation, "_run_pass", interrupted)
        write_interval(writer, scope, 1)
        assert writer.auto_consolidation.wait_idle(timeout=5)
        from living_memory import consolidation
        real_pass = consolidation.memory_consolidate

        def wait_then_run(*args, **kwargs):
            entered.set()
            assert release.wait(15)
            return real_pass(*args, **kwargs)

        monkeypatch.setattr(consolidation, "memory_consolidate", wait_then_run)
        peer = open_server(stack, path)
        stack.callback(release.set)
        assert entered.wait(5)
        result = write_interval(writer, scope, 6)
        assert writer.auto_consolidation.wait_idle(timeout=5)
        release.set()
        assert peer.auto_consolidation.wait_idle(timeout=30, include_retries=True)
        assert peer.auto_consolidation.last_result(scope)["status"] == "completed"
        assert scope in json.loads(writer.memory_store.get_kv(PENDING_KV_KEY))
        monkeypatch.setattr(consolidation, "memory_consolidate", real_pass)

    with ExitStack() as stack:
        recovered = open_server(stack, path)
        assert recovered.auto_consolidation.wait_idle(timeout=30, include_retries=True)
        assert recovered.auto_consolidation.last_result(scope)["status"] == "completed"
        assert recovered.auto_consolidation.status()["passes_completed"] == 1
        assert recovered.memory_store.get_node(result["node"]["id"])
        assert recovered.memory_store.get_kv(PENDING_KV_KEY) == "{}"
