"""Ordinary tools see whole knowledge while teach prepares detached content."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event, RLock
import sqlite3

import pytest

from living_memory import consolidation as c, server
from living_memory.storage import MemoryStore
from test_lookup import FakeMCP


CONTEXT = {"scope": "project:harbor", "agent": "engineer", "session_id": "preparation"}
CORRECTION = "Harbor crane requires the signed electronic manifest."
SECOND_CORRECTION = "Harbor crane inspection requires an independent brake inspector."
TEXTS = [
    "Harbor crane requires the paper permit.",
    "Harbor crane inspection includes the brake assembly.",
    "Harbor crane maintenance records the cable tension.",
]


@pytest.fixture(params=["audit", "credit"])
def mcp(tmp_path, monkeypatch, request):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", request.param)
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "fixed")
    monkeypatch.setenv("LM_RECALL_SESSION_DEDUP", "0")
    monkeypatch.setattr(server, "_transport_session_id", lambda: "preparation-transport")
    instance = server.create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    instance.auto_consolidation.close()
    yield instance
    instance.memory_store.close()


def derived(store, name, members):
    return store.create_node(
        level="concept", content=c._synthesize_digest(members, best=members[0]),
        context=CONTEXT,
        provenance={"strategy": "token-jaccard", "cluster_key": name,
                    "source_traces": [member.id for member in members]},
    )


def seed(mcp, *, diamond=False):
    store = mcp.memory_store
    traces = [store.append_trace(content, CONTEXT) for content in TEXTS]
    nodes = [derived(store, "left", traces[:2])]
    if diamond:
        nodes.append(derived(store, "right", [traces[0], traces[2]]))
        nodes.append(derived(store, "join", nodes[:]))
    return traces, nodes


def recall(mcp, **kwargs):
    return mcp.tools["memory_recall"](
        "Harbor crane permit inspection", scope=CONTEXT["scope"],
        ambient_context=CONTEXT, max_results=10, depth=0, **kwargs,
    )


def lookup(mcp, nodes):
    return mcp.tools["memory_lookup"](node_ids=[node.id for node in nodes])


@contextmanager
def preparing(mcp, monkeypatch, trace, **kwargs):
    entered, release = Event(), Event()
    prepare = c._prepare_digest

    def pause(*args, **kw):
        result = prepare(*args, **kw)
        if not entered.is_set():
            entered.set()
            assert release.wait(15), "test did not release preparation"
        return result

    monkeypatch.setattr(c, "_prepare_digest", pause)
    with ThreadPoolExecutor(max_workers=3) as workers:
        taught = workers.submit(
            mcp.tools["memory_teach"], trace.id, CORRECTION, context=CONTEXT, **kwargs,
        )
        try:
            assert entered.wait(10), "teach did not reach pure preparation"
            yield workers, taught
        finally:
            release.set()


def dump(store):
    with sqlite3.connect(store.db_path) as reader:
        return list(reader.iterdump())


def test_recall_and_lookup_advance_before_whole_diamond_commit(mcp, monkeypatch):
    traces, nodes = seed(mcp, diamond=True)
    store = mcp.memory_store
    with preparing(mcp, monkeypatch, traces[0]) as (workers, taught):
        before = workers.submit(lookup, mcp, nodes).result(timeout=10)
        recalled = workers.submit(recall, mcp).result(timeout=10)
        assert not taught.done()
        assert not store.connection.in_transaction
        assert [item["content"] for item in before["results"]] == [node.content for node in nodes]
        assert recalled["results"] and any(TEXTS[0] in item["content"] for item in recalled["results"])
        assert store.get_recall_event(recalled["recall_event_id"]) is not None
        assert not store.get_node(traces[0].id).corrections
    result = taught.result(timeout=10)
    after = lookup(mcp, nodes)["results"]
    assert all(CORRECTION in item["content"] and TEXTS[0] not in item["content"] for item in after)
    assert all(not item["decayed"] for item in after)
    new_id = result["corrective_trace"]["id"]
    assert set(store.get_node(nodes[0].id).source_traces) == {new_id, traces[1].id}
    assert set(store.get_node(nodes[1].id).source_traces) == {new_id, traces[2].id}
    assert set(store.get_node(nodes[2].id).source_traces) == {node.id for node in nodes[:2]}
    assert store.get_recall_event(recalled["recall_event_id"]).feedback_trace_id == new_id
    old = lookup(mcp, traces[:1])["results"][0]
    assert old["provenance"]["corrections"][-1]["new"] == CORRECTION
    for item in recall(mcp)["results"]:
        if item["id"] == traces[0].id:
            assert item["provenance"]["corrections"][-1]["new"] == CORRECTION
        else:
            assert TEXTS[0] not in item["content"]


def test_concurrent_teach_and_remember_survive_and_refresh_current_union(mcp, monkeypatch):
    traces, nodes = seed(mcp)
    store = mcp.memory_store
    with preparing(mcp, monkeypatch, traces[0]) as (workers, taught):
        second = workers.submit(
            mcp.tools["memory_teach"], traces[1].id, SECOND_CORRECTION,
            context={**CONTEXT, "agent": "inspector"},
        ).result(timeout=10)
        independent = workers.submit(
            mcp.tools["memory_remember"], "Independent harbor gate closes at dusk.",
            context={**CONTEXT, "agent": "gatekeeper"},
        ).result(timeout=10)
        # A new owner and added membership also belong to the final graph.
        with MemoryStore(store.db_path) as writer:
            new_owner = derived(writer, "new owner", [traces[0], traces[2]])
            current = writer.get_node(nodes[0].id)
            writer.update_node(current.id, provenance={
                **current.provenance, "source_traces": [*current.source_traces, traces[2].id],
            })
        assert not taught.done()
    first = taught.result(timeout=10)
    current = store.get_node(nodes[0].id)
    assert set(current.source_traces) == {
        first["corrective_trace"]["id"], second["corrective_trace"]["id"], traces[2].id,
    }
    members = c._current_derivation(store, current).live
    best = max(members, key=lambda member: (c._trace_quality(member), member.id))
    assert current.content == c._synthesize_digest(members, best=best)
    assert store.get_node(second["corrective_trace"]["id"]).content == SECOND_CORRECTION
    assert TEXTS[0] not in current.content and TEXTS[1] not in current.content
    assert CORRECTION in store.get_node(new_owner.id).content
    assert store.get_node(independent["node"]["id"]).content == "Independent harbor gate closes at dusk."
    assert store.get_node(traces[1].id).corrections[-1]["new"] == SECOND_CORRECTION


def test_required_failure_keeps_work_committed_during_preparation(mcp, monkeypatch):
    traces, _ = seed(mcp)
    store = mcp.memory_store
    delivered = recall(mcp)["results"]
    marked = [item["id"] for item in delivered]
    with preparing(mcp, monkeypatch, traces[0], used=marked[:1], irrelevant=marked[1:]) as (workers, taught):
        workers.submit(
            mcp.tools["memory_remember"], "Independent harbor entry needs a visitor badge.",
            context={**CONTEXT, "agent": "gatekeeper"},
        ).result(timeout=10)
        workers.submit(recall, mcp).result(timeout=10)
        store.connection.execute("""CREATE TRIGGER reject_closure
            BEFORE UPDATE OF feedback_applied_at ON recall_events
            WHEN NEW.feedback_applied_at IS NOT NULL
            BEGIN SELECT RAISE(ABORT, 'required closure failure'); END""")
        before = dump(store)
    with pytest.raises(sqlite3.IntegrityError, match="required closure failure"):
        taught.result(timeout=10)
    assert dump(store) == before
    assert not store.connection.in_transaction


def test_current_quality_selects_digest_lead_and_keeps_access_updates(mcp, monkeypatch):
    traces, nodes = seed(mcp)
    store = mcp.memory_store
    with preparing(mcp, monkeypatch, traces[0]) as (workers, taught):
        workers.submit(recall, mcp).result(timeout=10)
        with MemoryStore(store.db_path) as writer:
            writer.update_node(traces[1].id, stats={
                "usefulness_score": 1000.0, "access_count": 99, "confidence": 0.5,
            })
    taught.result(timeout=10)
    current = store.get_node(nodes[0].id)
    assert current.content.startswith(TEXTS[1])
    assert "signed electronic manifest" in current.content
    assert store.get_node(traces[1].id).access_count == 99
    assert store.get_node(traces[1].id).usefulness_score == 1000.0
    edge, = store.list_connections(source_id=current.id, target_id=traces[1].id, relation_type="related")
    assert edge.metadata["co_access"] == 99


def test_unchanged_cluster_renders_once_in_ordinary_handler(mcp, monkeypatch):
    traces, nodes = seed(mcp)
    rendered = []
    synthesize = c._synthesize_digest

    def observe(*args, **kwargs):
        rendered.append(kwargs["best"].id)
        assert not mcp.memory_store.connection.in_transaction
        return synthesize(*args, **kwargs)

    monkeypatch.setattr(c, "_synthesize_digest", observe)
    result = mcp.tools["memory_teach"](traces[0].id, CORRECTION, context=CONTEXT)
    assert rendered == [result["corrective_trace"]["id"]]
    assert CORRECTION in lookup(mcp, nodes)["results"][0]["content"]


def test_tool_reader_waits_for_complete_required_write(tmp_path, monkeypatch):
    attempted, writing, release = Event(), Event(), Event()

    class ObservedLock:
        def __init__(self):
            self.lock = RLock()

        def __enter__(self):
            if not self.lock.acquire(blocking=False):
                attempted.set()
                self.lock.acquire()
            return self

        def __exit__(self, *args):
            self.lock.release()

    monkeypatch.setattr(server, "RLock", ObservedLock)
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    mcp = server.create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    mcp.auto_consolidation.close()
    traces, nodes = seed(mcp, diamond=True)
    refresh = c._refresh_derived_from

    def pause_write(*args, **kwargs):
        refresh(*args, **kwargs)
        writing.set()
        assert release.wait(15), "test did not release required write"

    monkeypatch.setattr(c, "_refresh_derived_from", pause_write)
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            taught = workers.submit(mcp.tools["memory_teach"], traces[0].id, CORRECTION, context=CONTEXT)
            try:
                assert writing.wait(10)
                reader = workers.submit(lookup, mcp, nodes)
                assert attempted.wait(10)
                assert not reader.done()
                with sqlite3.connect(mcp.memory_store.db_path) as connection:
                    for node in nodes:
                        assert connection.execute("SELECT content FROM nodes WHERE id=?", (node.id,)).fetchone() == (node.content,)
                    assert connection.execute("SELECT count(*) FROM connections WHERE type='supersedes'").fetchone() == (0,)
            finally:
                release.set()
            taught.result(timeout=10)
            assert all(CORRECTION in item["content"] for item in reader.result(timeout=10)["results"])
    finally:
        mcp.memory_store.close()


def test_cycle_validity_is_decided_by_current_graph(mcp, monkeypatch):
    traces, nodes = seed(mcp)
    store = mcp.memory_store
    with preparing(mcp, monkeypatch, traces[0]) as (_, taught):
        with MemoryStore(store.db_path) as writer:
            cyclic = derived(writer, "cycle", [nodes[0]])
            writer.update_node(nodes[0].id, provenance={
                **nodes[0].provenance, "source_traces": [traces[0].id, cyclic.id],
            })
    taught.result(timeout=10)
    historical = lookup(mcp, [nodes[0], cyclic])["results"]
    assert all(item["decayed"] for item in historical)
    assert "cyclic sources" in historical[0]["decay_reason"]
