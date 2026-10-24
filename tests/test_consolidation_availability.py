"""Independent consolidation owners remain atomic while ordinary tools proceed."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event, RLock
import json
import sqlite3

import pytest

from living_memory import consolidation as c, server
from living_memory.auto_consolidation import PENDING_KV_KEY

SCOPE = "project:harbor"


class FakeMCP:
    def __init__(self, name, instructions=None):
        self.instructions = instructions
        self.tools = {}

    def tool(self, func=None, **kwargs):
        def register(handler):
            self.tools[handler.__name__] = handler
            return handler
        return register if func is None else register(func)

    def resource(self, uri):
        return lambda handler: handler

    def prompt(self, func=None, **kwargs):
        return (lambda handler: handler) if func is None else func


@pytest.fixture
def memory(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    locks = []
    waiting = Event()

    class Lock:
        def __init__(self):
            self.lock = RLock()
            locks.append(self)

        def __enter__(self):
            if not self.lock.acquire(blocking=False):
                waiting.set()
                self.lock.acquire()
            return self

        def __exit__(self, *args):
            self.lock.release()

    monkeypatch.setattr(server, "RLock", Lock)
    mcp = server.create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    traces = [store.append_trace(
        f"Harbor inspection checks marker {i} before departure.",
        {"scope": SCOPE, "agent": f"inspector-{i}"},
    ) for i in range(6)]
    owners = [store.create_node(
        level="concept", content=f"Harbor inspection owner {i} awaiting consolidation.",
        context={"scope": SCOPE},
        provenance={"strategy": "token-jaccard", "source_traces": [t.id for t in members]},
    ) for i, members in enumerate((traces[:4], traces[2:]))]
    # Retrieval clustering is independent of the guard and tested separately.
    cluster = c._TraceCluster(SCOPE, traces, Counter({"harbor": 6}))
    monkeypatch.setattr(c, "_cluster_traces", lambda *args, **kwargs: [cluster])
    try:
        yield mcp, locks[0], waiting, traces, owners, cluster
    finally:
        mcp.auto_consolidation.close()
        store.close()


def test_reader_waits_for_one_complete_owner_then_proceeds_before_the_next(memory, monkeypatch):
    mcp, lock, waiting, traces, owners, cluster = memory
    writing, release_write, between, release_pass = (Event() for _ in range(4))
    original = c._update_edge_weights_from_co_access
    written = []

    def edges(store, owner, members):
        if not written:
            writing.set()
            assert release_write.wait(15)
        original(store, owner, members)
        written.append(owner.id)

    monkeypatch.setattr(c, "_update_edge_weights_from_co_access", edges)

    @contextmanager
    def step():
        with lock:
            yield
        if len(written) == 1:
            assert not mcp.memory_store.connection.in_transaction
            between.set()
            assert release_pass.wait(15)

    with ThreadPoolExecutor(max_workers=2) as workers:
        work = workers.submit(c.memory_consolidate, mcp.memory_store, scope=SCOPE, guard=step)
        try:
            assert writing.wait(10)
            read = workers.submit(mcp.tools["memory_lookup"], node_ids=[n.id for n in owners])
            assert waiting.wait(10)
            assert not read.done()
            with sqlite3.connect(mcp.memory_store.db_path) as external:
                assert external.execute("SELECT content FROM nodes WHERE id=?", (owners[0].id,)).fetchone() == (owners[0].content,)
            release_write.set()
            assert between.wait(10)
            response = read.result(timeout=10)
            assert not work.done()
            assert [r["id"] for r in response["results"]] == [n.id for n in owners]
            current = mcp.memory_store.get_node(written[0])
            assert current.content != next(n.content for n in owners if n.id == current.id)
            assert all(mcp.memory_store.list_connections(source_id=current.id, target_id=source) for source in current.source_traces)
        finally:
            release_write.set()
            release_pass.set()
        result = work.result(timeout=15)
    assert {n.id for n in result.concepts_updated} == {n.id for n in owners}


@pytest.mark.parametrize("change", ["source", "owner", "decay", "union", "content", "metadata"])
def test_between_owner_write_uses_current_sources_and_preserves_foreground_feedback(memory, monkeypatch, change):
    mcp, lock, _, traces, owners, cluster = memory
    store = mcp.memory_store
    written = []
    original = c._write_cluster_concept

    def write(*args, **kwargs):
        result = original(*args, **kwargs)
        written.append(result.id)
        return result

    monkeypatch.setattr(c, "_write_cluster_concept", write)
    changed = False
    facts = {}

    @contextmanager
    def step():
        nonlocal changed
        with lock:
            yield
        if written and not changed:
            changed = True
            assert not store.connection.in_transaction
            remaining = next(n for n in owners if n.id != written[0])
            recalled = mcp.tools["memory_recall"](
                "harbor inspection", scope=SCOPE,
                ambient_context={"transport_session_id": "foreground"},
            )
            remembered = mcp.tools["memory_remember"](
                "Harbor inspection records the final departure time.",
                {"scope": SCOPE, "transport_session_id": "foreground"},
                used=[recalled["results"][0]["id"]],
            )
            assert remembered["feedback_marks"]["accepted"] == 1
            assert store.get_node(remembered["node"]["id"]) is not None
            facts.update(remaining=remaining.id, event=recalled["recall_event_id"], remembered=remembered["node"]["id"])
            if change == "source":
                taught = mcp.tools["memory_teach"](traces[2].id, "Harbor inspection now uses the signed departure manifest.")
                facts["replacement"] = taught["corrective_trace"]["id"]
            elif change == "owner":
                mcp.tools["memory_teach"](remaining.id, "Harbor inspection uses an independent current procedure.")
            elif change == "decay":
                mcp.tools["memory_forget"](remaining.id, "retired by operator")
            elif change == "union":
                current = store.get_node(remaining.id)
                store.update_node(remaining.id, provenance={**current.provenance, "source_traces": [*current.source_traces, remembered["node"]["id"]]})
                store.update_node(traces[2].id, stats={"access_count": 77, "usefulness_score": 4})
            else:
                updates = {"content": "Harbor inspection verifies the calibrated compass."} if change == "content" else {}
                store.update_node(
                    traces[2].id, **updates, embedding=[0.0, 1.0],
                    context={"scope": SCOPE, "agent": "independent-inspector"},
                    stats={"access_count": 99, "usefulness_score": 1000, "confidence": 0.9, "unique_agents": 12},
                )

    c._materialize_cluster(store, cluster, min_cluster_size=3, step=step)
    current = store.get_node(facts["remaining"])
    assert store.get_node(facts["remembered"]) is not None
    assert store.connection.execute("SELECT feedback_applied FROM recall_events WHERE id=?", (facts["event"],)).fetchone()[0] == 1
    if change == "source":
        assert facts["replacement"] in current.source_traces
        assert traces[2].id not in current.source_traces
        assert "signed departure manifest" in current.content
    elif change == "owner":
        assert current.corrections and current.content == next(n.content for n in owners if n.id == current.id)
        assert current.id not in written
    elif change == "decay":
        assert current.decayed and current.decay_reason == "retired by operator"
        assert current.id not in written
    elif change == "union":
        assert facts["remembered"] in current.source_traces
        edge, = store.list_connections(source_id=current.id, target_id=traces[2].id, relation_type="related")
        assert edge.metadata["co_access"] == 77
    else:
        members = c._current_derivation(store, current).live
        best = max(members, key=lambda member: (c._trace_quality(member), member.id))
        expected = c._prepare_digest(SCOPE, members, best, c._prepare_digest_sources(members))
        assert best.id == traces[2].id
        assert current.content == expected.content
        assert current.provenance["cluster_key"] == expected.cluster_key
        assert current.embedding == best.embedding == [0.0, 1.0]
        assert current.unique_agents == 12
        assert current.confidence == c._consensus_confidence(members, 12)
        assert current.usefulness_score == sum(member.usefulness_score for member in members) / len(members)
        edge, = store.list_connections(source_id=current.id, target_id=best.id, relation_type="related")
        assert edge.metadata["co_access"] == 99
        if change == "content":
            assert "calibrated compass" in current.content
            assert traces[2].content not in current.content


def test_interrupted_pass_rolls_back_current_owner_and_resumes_durable_pending(memory, monkeypatch):
    mcp, _, _, traces, owners, cluster = memory
    store = mcp.memory_store
    original = c._update_edge_weights_from_co_access
    written = []

    def fail(store, owner, members):
        original(store, owner, members)
        written.append(owner.id)
        if len(written) == 2:
            raise RuntimeError("interrupted owner write")

    monkeypatch.setattr(c, "_update_edge_weights_from_co_access", fail)
    mcp.auto_consolidation._retry_delays = ()
    mcp.auto_consolidation.request(SCOPE)
    assert mcp.auto_consolidation.wait_idle(15, include_retries=True)
    assert mcp.auto_consolidation.last_result(SCOPE)["status"] == "failed"
    assert SCOPE in json.loads(store.get_kv(PENDING_KV_KEY))
    assert store.get_node(written[0]).content != next(n.content for n in owners if n.id == written[0])
    interrupted = store.get_node(written[1])
    assert interrupted.content == next(n.content for n in owners if n.id == interrupted.id)
    assert not store.list_connections(source_id=interrupted.id, relation_type="related")
    assert not store.connection.in_transaction
    mcp.auto_consolidation.close()
    monkeypatch.setattr(c, "_update_edge_weights_from_co_access", original)
    # Reopening the actual server consumes the same persisted pending scope.
    resumed = server.create_mcp_server(store.db_path, mcp_factory=FakeMCP)
    try:
        assert resumed.auto_consolidation.wait_idle(15)
        assert resumed.auto_consolidation.last_result(SCOPE)["status"] == "completed"
        assert json.loads(resumed.memory_store.get_kv(PENDING_KV_KEY)) == {}
        assert {n.id for n in resumed.memory_store.list_nodes(level="concept", scope=SCOPE)} == {n.id for n in owners}
        for owner in owners:
            current = resumed.memory_store.get_node(owner.id)
            assert current.content != owner.content
            assert set(current.source_traces) == set(owner.source_traces)
            assert all(resumed.memory_store.list_connections(source_id=owner.id, target_id=source) for source in current.source_traces)
    finally:
        resumed.auto_consolidation.close()
        resumed.memory_store.close()
