"""Fresh trace preparation leaves knowledge available until atomic apply."""

import asyncio
import sqlite3
from threading import Event, RLock

from fastmcp import Client
import pytest

from living_memory import server
from living_memory.storage import MemoryStore
from test_fresh_write_indexing import assert_current_chunks, mcp


SOURCE = "Cobalt registry certificates need validation before release."
CONTENT = "Preparation owns immutable text before knowledge publication. " * 80
ALTERNATIVES = [{"option": "  Skip certificate validation  ", "why": "  Invalid signatures remain unsafe.  "}]
REJECTED = "Rejected alternative: Skip certificate validation\nRejected because: Invalid signatures remain unsafe."
CONTEXT = {"scope": "project:preparation"}


async def call(client, tool, **arguments):
    return (await client.call_tool(tool, arguments)).data


async def write(client, operation, original, **kwargs):
    if operation == "teach":
        result = await call(client, "memory_teach", trace_id=original,
                            correction={"new": "\n " + CONTENT + " \n"}, **kwargs)
        return result, [result["corrective_trace"]["id"]], [CONTENT.strip()]
    arguments = {"alternatives_considered": ALTERNATIVES} if operation == "alternatives" else {}
    result = await call(client, "memory_remember", content=CONTENT, **arguments, **kwargs)
    return result, [result["node"]["id"], *result.get("rejected_alternatives", [])], [
        CONTENT, *([REJECTED] if arguments else []),
    ]


@pytest.mark.parametrize("operation", ["remember", "alternatives", "teach"])
def test_native_preparation_keeps_readers_and_independent_writes_available(mcp, operation):
    store = mcp.memory_store
    entered, release = Event(), Event()
    encoded = []

    def held_encoder(texts):
        encoded.extend(texts)
        if not entered.is_set():
            assert not store.connection.in_transaction
            entered.set()
            assert release.wait(10), "preparation was not released"
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def scenario():
        async with Client(mcp) as owner, Client(mcp) as peer:
            original = (await call(owner, "memory_remember", content=SOURCE, context=CONTEXT))["node"]["id"]
            store.set_chunk_embedder(held_encoder)
            pending = asyncio.create_task(write(owner, operation, original, context=CONTEXT))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                old = await asyncio.wait_for(call(peer, "memory_lookup", node_id=original), 3)
                assert old["results"][0]["content"] == SOURCE
                assert not pending.done()
                # This connection also commits a write, not just a lock probe.
                with MemoryStore(store.db_path) as writer:
                    writer.connection.execute("BEGIN IMMEDIATE")
                    independent = writer.append_trace("Independent gate requires a badge.", CONTEXT)
                    writer.connection.commit()
                    assert writer.trace_count() == 2
                    assert not writer.get_node(original).corrections
                # A native write that does not need the busy encoder also runs.
                await asyncio.wait_for(call(peer, "memory_connect", id_a=original,
                                           id_b=independent.id, relation_type="related"), 3)
                assert not pending.done()
            finally:
                release.set()
            result, ids, contents = await asyncio.wait_for(pending, 5)
            assert "indexing" not in result
            from living_memory.chunking import chunk_text
            assert encoded == [chunk.text for content in contents for chunk in chunk_text(content)]
            assert len(chunk_text(CONTENT)) == 14
            for node_id, content in zip(ids, contents, strict=True):
                actual = await call(peer, "memory_lookup", node_id=node_id)
                assert actual["results"][0]["content"] == content
                assert_current_chunks(store, node_id, content)
                assert all(row.embedding == [1.0, 0.0, 0.0] for row in store.list_node_chunks(node_id))
            assert store.get_node(independent.id).content == independent.content
            assert store.list_connections(source_id=original, target_id=independent.id)

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["remember", "alternatives", "teach"])
@pytest.mark.parametrize("fail_required", [False, True])
def test_native_apply_is_visible_only_after_required_commit(tmp_path, monkeypatch, operation, fail_required):
    attempted, applying, release = Event(), Event(), Event()

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

    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
    monkeypatch.setenv("LM_RECALL_MAP", "0")
    monkeypatch.setattr(server, "RLock", ObservedLock)
    mcp = server.create_mcp_server(tmp_path / "memory.db")
    store = mcp.memory_store
    insert = store._insert_node
    written = []

    def hold_apply(**kwargs):
        result = insert(**kwargs)
        written.append(result)
        if not applying.is_set():
            assert store.connection.in_transaction
            applying.set()
            assert release.wait(10), "apply was not released"
        return result

    async def scenario():
        async with Client(mcp) as client:
            original = (await call(client, "memory_remember", content=SOURCE, context=CONTEXT))["node"]["id"]
            prior = await call(client, "memory_recall", query="Cobalt registry", ambient_context=CONTEXT)
            if fail_required:
                store.connection.execute("""CREATE TRIGGER reject_closure
                    BEFORE UPDATE OF feedback_applied_at ON recall_events
                    WHEN NEW.feedback_applied_at IS NOT NULL
                    BEGIN SELECT RAISE(ABORT, 'required closure failure'); END""")
            monkeypatch.setattr(store, "_insert_node", hold_apply)
            pending = asyncio.create_task(write(client, operation, original, context=CONTEXT, used=[original]))
            try:
                assert await asyncio.to_thread(applying.wait, 5)
                reader = asyncio.create_task(call(client, "memory_lookup", node_id=written[0]))
                assert await asyncio.to_thread(attempted.wait, 5)
                assert not reader.done()
                with sqlite3.connect(store.db_path) as observer:
                    assert observer.execute("SELECT count(*) FROM nodes").fetchone() == (1,)
                    assert observer.execute("SELECT count(*) FROM connections WHERE type='supersedes'").fetchone() == (0,)
                    assert observer.execute("SELECT feedback_trace_id FROM recall_events WHERE id=?",
                                            (prior["recall_event_id"],)).fetchone() == (None,)
            finally:
                release.set()
            if fail_required:
                with pytest.raises(Exception, match="required closure failure"):
                    await pending
                assert (await reader)["results"] == []
                assert store.trace_count() == 1
                assert store.connection.execute("SELECT count(*) FROM recall_feedback_marks").fetchone()[0] == 0
                assert store.count_node_chunks(written[0]) == 0
            else:
                result, ids, contents = await pending
                assert (await reader)["results"][0]["content"] == contents[0]
                assert store.get_recall_event(prior["recall_event_id"]).feedback_trace_id == ids[0]
                assert store.trace_count() == 1 + len(ids)
            assert not store.connection.in_transaction

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        mcp.auto_consolidation.close()
        store.close()


@pytest.mark.parametrize("operation", ["remember", "teach"])
def test_apply_uses_current_feedback_and_scope_after_preparation(mcp, monkeypatch, operation):
    store = mcp.memory_store
    entered, release = Event(), Event()
    prepare = server._prepare_trace_indices

    def hold_prepared(*args):
        result = prepare(*args)
        entered.set()
        assert release.wait(10)
        return result

    async def scenario():
        async with Client(mcp) as owner, Client(mcp) as peer:
            original = (await call(owner, "memory_remember", content=SOURCE, context=CONTEXT))["node"]["id"]
            first = await call(owner, "memory_recall", query="Cobalt registry certificates", scope=CONTEXT["scope"])
            monkeypatch.setattr(server, "_prepare_trace_indices", hold_prepared)
            pending = asyncio.create_task(write(owner, operation, original, used=[original]))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                # Query encoding retains its own guard. Hold the completed
                # preparation here so native feedback can advance independently.
                own = await asyncio.wait_for(call(owner, "memory_recall", query="Cobalt registry certificates"), 3)
                foreign = await asyncio.wait_for(call(peer, "memory_recall", query="Cobalt registry certificates"), 3)
                with MemoryStore(store.db_path) as writer:
                    current = writer.get_node(original)
                    writer.update_node(original, context={**current.context, "scope": "project:current"},
                                       stats={"usefulness_score": 9.0})
                    concurrent = writer.append_trace(CONTENT, {
                        "scope": "global",
                        "transport_session_id": current.context["transport_session_id"],
                    })
                assert not pending.done()
            finally:
                release.set()
            result, ids, contents = await pending
            current = store.get_node(ids[0])
            assert store.get_recall_event(first["recall_event_id"]).feedback_trace_id == current.id
            assert store.get_recall_event(own["recall_event_id"]).feedback_trace_id == current.id
            assert store.get_recall_event(foreign["recall_event_id"]).feedback_trace_id is None
            assert result["feedback_marks"]["accepted"] == 1
            if operation == "teach":
                assert current.scope == "project:current"
                # Accepted used credit clamps the current original score to 1
                # before teach derives the corrective score.
                assert current.usefulness_score == 2.0
                old = await call(peer, "memory_lookup", node_id=original)
                assert old["results"][0]["superseded"]
            else:
                assert store.get_node(concurrent.id).decayed
                assert current.scope == "global"
            assert_current_chunks(store, current.id, contents[0])

    asyncio.run(scenario())


def test_preparation_failure_is_local_to_one_normalized_alternative(mcp):
    store = mcp.memory_store
    encoded = []

    def encode(texts):
        encoded.extend(texts)
        if any("Skip certificate" in text for text in texts):
            raise RuntimeError("private encoder failure")
        return [[1.0, 0.0] for _ in texts]

    store.set_chunk_embedder(encode)

    async def scenario():
        async with Client(mcp) as client:
            result = await call(client, "memory_remember", content=SOURCE, context=CONTEXT,
                                alternatives_considered=[*ALTERNATIVES, {
                                    "alternative": "  Keep signed records  ", "reason": "  Separate audit cost  ",
                                }])
            primary = result["node"]["id"]
            failed, successful = result["rejected_alternatives"]
            assert result["indexing"] == {"status": "deferred", "node_ids": [failed]}
            assert store.get_node(failed).content == REJECTED
            assert store.count_node_chunks(failed) == 0
            for node_id in (primary, successful):
                assert_current_chunks(store, node_id, store.get_node(node_id).content)
            assert REJECTED in encoded

    asyncio.run(scenario())


def test_changed_source_cannot_receive_prepared_chunks(mcp, monkeypatch):
    store = mcp.memory_store
    append = store.append_trace

    def transform(content, *args, **kwargs):
        return append(content + " Additional stored condition.", *args, **kwargs)

    monkeypatch.setattr(store, "append_trace", transform)

    async def scenario():
        async with Client(mcp) as client:
            result = await call(client, "memory_remember", content=SOURCE)
            target = result["node"]["id"]
            assert result["indexing"] == {"status": "deferred", "node_ids": [target]}
            assert store.count_node_chunks(target) == 0
            assert store.get_node(target).content == SOURCE + " Additional stored condition."

    asyncio.run(scenario())
