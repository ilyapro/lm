"""Native pending-index and feedback preparation keep independent reads usable."""

import asyncio
import json
from threading import RLock

from fastmcp import Client
import pytest

from living_memory import feedback, server
from living_memory.chunking import chunk_text
from living_memory.irrelevance import list_query_irrelevance
from living_memory.query_anchors import resolve_query_anchor
from living_memory.storage import MemoryStore, recall_fingerprint
from test_recall_query_preparation import (
    CONTENT, QUERY, SCOPE, VECTOR, call, entry, recall, runtime as query_runtime,
)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    guard = RLock()
    monkeypatch.setattr(server, "RLock", lambda: guard)
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    monkeypatch.setenv("LM_LOOKUP_CREDIT_POLICY", "delivered")
    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", "grounded")
    fixture = query_runtime.__wrapped__(tmp_path, monkeypatch)
    r = next(fixture)
    r.guard, r.unlocked_sql, r.batches = guard, [], []
    r.feedback_hold = False
    original = feedback._anchor_embedder

    def anchor_encoder(store):
        encode = original(store)

        def run(queries):
            assert not guard._is_owned()
            assert not store.connection.in_transaction
            r.batches.append(tuple(queries))
            if r.feedback_hold:
                r.entered.set()
                assert r.release.wait(10), "feedback encoder was not released"
            return encode(queries)

        return run

    monkeypatch.setattr(feedback, "_anchor_embedder", anchor_encoder)
    r.store.connection.set_trace_callback(
        lambda statement: r.unlocked_sql.append(statement) if not guard._is_owned() else None,
    )
    try:
        yield r
        assert not r.unlocked_sql
    finally:
        r.release.set()
        r.store.connection.set_trace_callback(None)
        with pytest.raises(StopIteration):
            next(fixture)


def anchor(store, query=QUERY, scope=SCOPE):
    return store.find_query_anchor(scope, recall_fingerprint(query, scope))


def pending_source(r, clustered):
    with r.guard:
        node = r.store.create_node(
            level="concept", content=CONTENT * 80, context={"scope": SCOPE},
            provenance={"strategy": "embedding-cosine", "source_traces": [r.node.id]}
            if clustered else None,
        )
        if clustered:
            r.store.connection.execute("UPDATE nodes SET embedding = ? WHERE id = ?",
                                       (json.dumps([0.0, 1.0, 0.0]), node.id))
        node = r.store.get_node(node.id)
    encode = r.store._resolve_chunk_embedder()

    def held(texts):
        assert not r.guard._is_owned()
        assert not r.store.connection.in_transaction
        r.batches.append(tuple(texts))
        r.entered.set()
        assert r.release.wait(10), "pending encoder was not released"
        return encode(texts)

    r.store.set_chunk_embedder(held)
    return node


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("view", ["knowledge", "full"])
def test_pending_index_keeps_exact_read_available_and_preserves_representative(runtime, clustered, view):
    r = runtime
    node = pending_source(r, clustered)

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as reader:
            operation = asyncio.create_task(recall(owner, view=view))
            try:
                assert await asyncio.to_thread(r.entered.wait, 4)
                lookup = await asyncio.wait_for(call(reader, "memory_lookup", node_id=r.node.id), 3)
                assert entry(lookup)["content"] == CONTENT
                await asyncio.wait_for(call(reader, "memory_status", scope=SCOPE), 3)
                assert not operation.done()
            finally:
                r.release.set()
            result = await asyncio.wait_for(operation, 5)
            with r.guard:
                chunks = r.store.list_node_chunks(node.id)
                assert len(chunks) == len(chunk_text(node.content)) == len(r.batches[0])
                assert r.store.node_chunks_are_current(node.id)
                assert all(node.content[c.char_start:c.char_end] for c in chunks)
                current = r.store.get_node(node.id)
                assert current.embedding == (node.embedding if clustered else VECTOR)
                assert current.provenance == node.provenance
                event = r.store.get_recall_event(result["recall_event_id"])
                assert len(r.store.recall_delivery_history(event.transport_session_id)) == 1
                assert event.result_ids == [entry(result)["id"]]
            assert (node.content in r.encoded) == (not clustered)

    asyncio.run(scenario())


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("change", ["edit", "delete", "decay", "scope", "index", "representative"])
def test_pending_revalidation_preserves_current_source_and_index(runtime, clustered, change):
    r = runtime
    node = pending_source(r, clustered)

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as reader:
            operation = asyncio.create_task(recall(owner))
            try:
                assert await asyncio.to_thread(r.entered.wait, 4)
                with MemoryStore(r.store.db_path, chunk_embedder=lambda texts: [VECTOR for _ in texts]) as writer:
                    if change == "edit":
                        writer.update_node(node.id, content="Current amber condition.", embedding=[0.0, 1.0, 0.0])
                    elif change == "delete":
                        with writer.transaction():
                            writer.delete_node_chunks(node.id)
                            writer.connection.execute("DELETE FROM nodes WHERE id = ?", (node.id,))
                    elif change == "decay":
                        await asyncio.wait_for(call(reader, "memory_forget", id=node.id), 3)
                    elif change == "scope":
                        writer.update_node(node.id, context={"scope": "project:moved"})
                    elif change == "index":
                        writer.replace_node_chunks(node.id, [(c, [0.0, 1.0, 0.0]) for c in chunk_text(node.content)])
                    else:
                        writer.connection.execute("UPDATE nodes SET embedding = ? WHERE id = ?",
                                                  (json.dumps([0.0, 0.0, 1.0]), node.id))
                    expected = writer.list_node_chunks(node.id)
                    current = writer.get_node(node.id)
                lookup = await asyncio.wait_for(call(reader, "memory_lookup", node_id=node.id), 3)
                assert bool(lookup["results"]) == (change != "delete")
                assert not operation.done()
            finally:
                r.release.set()
            await asyncio.wait_for(operation, 5)
            with r.guard:
                assert r.store.list_node_chunks(node.id) == expected
                final = r.store.get_node(node.id)
                if current is None:
                    assert final is None
                else:
                    assert (final.content, final.scope, final.decayed, final.embedding, final.provenance) == (
                        current.content, current.scope, current.decayed, current.embedding, current.provenance,
                    )

    asyncio.run(scenario())


async def consume(client, route, node_id):
    if route == "lookup":
        return await call(client, "memory_lookup", node_id=node_id)
    if route in ("recall", "negative"):
        return await recall(client, query="next certificate request", **{
            "used" if route == "recall" else "irrelevant": [node_id],
        })
    if route == "teach":
        return await call(client, "memory_teach", trace_id=node_id,
                          correction=CONTENT + " Confirmed for release.", used=[node_id], context={"scope": SCOPE})
    return await call(client, "memory_remember", content=CONTENT + " Confirmed for release.",
                      context={"scope": SCOPE}, **({"used": [node_id]} if route == "explicit" else {}))


@pytest.mark.parametrize("route", ["lookup", "recall", "negative", "explicit", "implicit", "teach"])
def test_missing_anchor_preparation_keeps_reads_available_for_every_consumer(runtime, route):
    r = runtime

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as reader:
            prior = await recall(owner)
            r.feedback_hold = True
            operation = asyncio.create_task(consume(owner, route, r.node.id))
            try:
                assert await asyncio.to_thread(r.entered.wait, 4)
                lookup = await asyncio.wait_for(call(reader, "memory_lookup", node_id=r.node.id), 3)
                assert entry(lookup)["content"] == CONTENT
                assert not operation.done()
                with r.guard:
                    assert r.store.credited_node_ids(prior["recall_event_id"]) == set()
                    assert r.store.count_query_anchors() == 0
            finally:
                r.release.set()
            await asyncio.wait_for(operation, 5)
            assert r.batches == [(QUERY,)]
            with r.guard:
                assert r.store.count_query_anchors() == 1
                assert r.store.credited_node_ids(prior["recall_event_id"]) == (
                    set() if route == "negative" else {r.node.id}
                )
                if route == "negative":
                    assert list_query_irrelevance(r.store)[0]["weight"] == 0.5
                else:
                    edge, = r.store.list_query_anchor_edges(anchor_id=anchor(r.store).id)
                    assert edge.hits == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["lookup", "explicit", "implicit"])
def test_simultaneous_feedback_consumers_claim_current_credit_once(runtime, route):
    r = runtime

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as reader:
            prior = await recall(owner)
            r.feedback_hold = True
            first = asyncio.create_task(consume(owner, route, r.node.id))
            second = None
            try:
                assert await asyncio.to_thread(r.entered.wait, 4)
                second = asyncio.create_task(consume(owner, route, r.node.id))
                await asyncio.wait_for(call(reader, "memory_lookup", node_id=r.node.id), 3)
                assert not first.done()
            finally:
                r.release.set()
            await asyncio.wait_for(asyncio.gather(first, second), 5)
            with r.guard:
                assert r.store.credited_node_ids(prior["recall_event_id"]) == {r.node.id}
                assert r.store.count_query_anchors() == 1
                current = anchor(r.store)
                assert current.reinforcement_count == 0
                edge, = r.store.list_query_anchor_edges(anchor_id=current.id)
                assert edge.hits == 1
                assert r.store.connection.execute("SELECT count(*) FROM recall_credit_ledger").fetchone()[0] == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["explicit", "teach"])
def test_missing_anchor_required_failure_rolls_back_knowledge_and_all_feedback(runtime, operation):
    r = runtime

    async def scenario():
        async with Client(r.mcp) as owner:
            prior = await recall(owner)
            with r.guard:
                before = r.store.get_node(r.node.id)
                chunks = r.store.list_node_chunks(r.node.id)
                r.store.connection.execute("""CREATE TRIGGER reject_closure
                    BEFORE UPDATE OF feedback_applied_at ON recall_events
                    WHEN NEW.feedback_applied_at IS NOT NULL
                    BEGIN SELECT RAISE(ABORT, 'required closure failure'); END""")
            with pytest.raises(Exception, match="required closure failure"):
                await consume(owner, operation, r.node.id)
            with r.guard:
                assert r.store.get_node(r.node.id) == before
                assert r.store.list_node_chunks(r.node.id) == chunks
                assert r.store.get_recall_event(prior["recall_event_id"]).feedback_trace_id is None
                for table, expected in [("nodes", 1), ("connections", 0), ("recall_feedback_marks", 0),
                                        ("recall_credit_ledger", 0), ("query_anchors", 0),
                                        ("query_anchor_edges", 0), ("query_irrelevance", 0)]:
                    assert r.store.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == expected
                assert not r.store.connection.in_transaction
            assert r.batches == [(QUERY,)]

    asyncio.run(scenario())
