"""Forget prepares outside the runtime lock and commits the current graph."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import sqlite3
from threading import Event, RLock

import pytest

from living_memory import consolidation as c, server
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.storage import MemoryStore
from test_forget_derivation import SCOPE, TEXTS, derived, sources, deterministic_runtime
from test_lookup import FakeMCP


@pytest.fixture
def mcp(tmp_path, monkeypatch):
    guard = RLock()
    monkeypatch.setattr(server, "RLock", lambda: guard)
    instance = server.create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    instance.auto_consolidation.close()
    instance.test_guard = guard
    yield instance
    instance.memory_store.close()


@contextmanager
def preparing(mcp, monkeypatch, trace, *, release=None):
    entered = Event()
    release = release or Event()
    prepare = c._prepare_digest

    def pause(*args, **kwargs):
        result = prepare(*args, **kwargs)
        if not entered.is_set():
            assert not mcp.test_guard._is_owned()
            assert not mcp.memory_store.connection.in_transaction
            entered.set()
            assert release.wait(15), "test did not release preparation"
        return result

    monkeypatch.setattr(c, "_prepare_digest", pause)
    with ThreadPoolExecutor(max_workers=3) as workers:
        forgotten = workers.submit(mcp.tools["memory_forget"], trace.id)
        try:
            assert entered.wait(10), "forget did not reach digest preparation"
            yield workers, forgotten
        finally:
            release.set()


def test_tools_advance_during_preparation_and_final_writer_uses_current_union(mcp, monkeypatch):
    store = mcp.memory_store
    traces = sources(store)
    owner = derived(store, "release", traces)
    correction = "Workshop release validates the signed package manifest."
    with preparing(mcp, monkeypatch, traces[0]) as (workers, forgotten):
        old = workers.submit(mcp.tools["memory_lookup"], node_id=owner.id).result(timeout=10)
        assert old["results"][0]["content"] == owner.content
        recalled = workers.submit(mcp.tools["memory_recall"], "workshop release", scope=SCOPE).result(timeout=10)
        assert recalled["results"]
        second = workers.submit(mcp.tools["memory_teach"], traces[1].id, correction).result(timeout=10)
        fresh = workers.submit(mcp.tools["memory_remember"], "Workshop tools stay dry.",
                               context={"scope": SCOPE}).result(timeout=10)
        with MemoryStore(store.db_path) as writer:
            fresh_node = writer.get_node(fresh["node"]["id"])
            current = writer.get_node(owner.id)
            writer.update_node(owner.id, provenance={
                **current.provenance, "source_traces": [*current.source_traces, fresh_node.id],
            })
            new_owner = derived(writer, "new consumer", [traces[0], fresh_node])
        assert not forgotten.done()
    assert forgotten.result(timeout=10)["node"]["decayed"]
    current = store.get_node(owner.id)
    assert set(current.source_traces) == {second["corrective_trace"]["id"], traces[2].id, fresh_node.id}
    assert correction in current.content and TEXTS[2] in current.content
    assert all(text not in current.content for text in TEXTS[:2])
    assert store.get_node(new_owner.id).source_traces == [fresh_node.id]
    assert store.get_node(new_owner.id).content == fresh_node.content


@pytest.mark.parametrize("change", ["content", "quality", "validity", "cycle", "replacement"])
def test_current_inputs_revalidate_prepared_digest(mcp, monkeypatch, change):
    store = mcp.memory_store
    traces = sources(store)
    owner = derived(store, "release", traces)
    with preparing(mcp, monkeypatch, traces[0]) as (_, forgotten):
        with MemoryStore(store.db_path) as writer:
            if change == "content":
                writer.update_node(traces[1].id, content="Workshop release checks a signed checksum.")
            elif change == "quality":
                writer.update_node(traces[1].id, stats={"usefulness_score": 1000.0, "access_count": 99})
            elif change == "validity":
                c.memory_forget(writer, traces[1].id)
            elif change == "cycle":
                writer.update_node(owner.id, provenance={
                    **owner.provenance, "source_traces": [*owner.source_traces, owner.id],
                })
            else:
                replacement = derived(writer, "replacement", [traces[0], traces[2]])
                writer.create_connection(replacement.id, traces[1].id, "supersedes")
    forgotten.result(timeout=10)
    current = store.get_node(owner.id)
    assert store.get_node(traces[0].id).decayed
    if change == "cycle":
        assert current.decayed and "cyclic" in current.decay_reason
        return
    assert not current.decayed and TEXTS[0] not in current.content
    if change == "content":
        assert "Workshop release checks a signed checksum." in current.content
        assert TEXTS[1] not in current.content
    elif change == "quality":
        assert current.content.startswith(TEXTS[1])
        assert store.get_node(traces[1].id).access_count == 99
        edge, = store.list_connections(source_id=owner.id, target_id=traces[1].id, relation_type="related")
        assert edge.metadata["co_access"] == 99
    elif change == "validity":
        assert current.source_traces == [traces[2].id]
        assert current.content == TEXTS[2]
    else:
        assert set(current.source_traces) == {replacement.id, traces[2].id}
        assert TEXTS[1] not in current.content
        assert store.get_node(replacement.id).source_traces == [traces[2].id]


def test_unchanged_digest_renders_once_outside_lock_without_encoding(mcp, monkeypatch):
    store = mcp.memory_store
    traces = sources(store)
    owner = derived(store, "release", traces)
    rendered = []
    render = c._synthesize_digest

    def observe(*args, **kwargs):
        assert not mcp.test_guard._is_owned()
        assert not store.connection.in_transaction
        rendered.append(True)
        return render(*args, **kwargs)

    def no_encoding(*args, **kwargs):
        raise AssertionError("forget must not encode under the runtime lock")

    monkeypatch.setattr(c, "_synthesize_digest", observe)
    monkeypatch.setattr(LocalEmbeddingModel, "embed", no_encoding)
    store.set_chunk_embedder(no_encoding)
    mcp.tools["memory_forget"](traces[0].id)
    assert rendered == [True]
    assert TEXTS[0] not in store.get_node(owner.id).content


@pytest.mark.parametrize("fail", [False, True])
def test_readers_see_atomic_outcome_with_concurrent_work(mcp, monkeypatch, fail):
    store = mcp.memory_store
    traces = sources(store)
    owner = derived(store, "release", traces)
    writing, release, attempted = Event(), Event(), Event()
    prepared_release = Event()
    refresh = c._refresh_derived_from

    def pause_write(*args, **kwargs):
        refresh(*args, **kwargs)
        assert mcp.test_guard._is_owned()
        writing.set()
        assert release.wait(15), "test did not release mutation"
        if fail:
            raise RuntimeError("required forget failure")

    def read():
        assert not mcp.test_guard.acquire(blocking=False)
        attempted.set()
        return mcp.tools["memory_lookup"](node_id=owner.id)

    with preparing(mcp, monkeypatch, traces[0], release=prepared_release) as (workers, forgotten):
        independent = workers.submit(mcp.tools["memory_remember"], "Workshop lights turn off at night.",
                                     context={"scope": SCOPE}).result(timeout=10)
        monkeypatch.setattr(c, "_refresh_derived_from", pause_write)
        with sqlite3.connect(store.db_path) as reader:
            before = list(reader.iterdump())
        prepared_release.set()
        try:
            assert writing.wait(10), "forget did not reach required mutation"
            lookup = workers.submit(read)
            assert attempted.wait(10)
            assert not lookup.done()
            with sqlite3.connect(store.db_path) as reader:
                assert list(reader.iterdump()) == before
        finally:
            release.set()
        if fail:
            with pytest.raises(RuntimeError, match="required forget failure"):
                forgotten.result(timeout=10)
        else:
            assert forgotten.result(timeout=10)["node"]["decayed"]
        current = lookup.result(timeout=10)["results"][0]
        assert (TEXTS[0] in current["content"]) == fail
        assert store.get_node(independent["node"]["id"]).content == "Workshop lights turn off at night."
        if fail:
            with sqlite3.connect(store.db_path) as reader:
                assert list(reader.iterdump()) == before
