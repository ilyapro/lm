"""A demanded legacy repair leaves the shared store available to other clients."""

import asyncio
import json
from threading import Event, RLock
from types import SimpleNamespace

from fastmcp import Client
import pytest

from living_memory import server
from living_memory.chunking import chunk_text
from living_memory.recall_map import RecallMapBuilder
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore
from test_legacy_passage_refresh import CONTENT, FACT, QUERY, SCOPE, Encoder, legacy, rows


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
        "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic",
        "LM_AUTO_CONSOLIDATE_POLICY": "off",
        "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
        "LM_RECALL_MAP": "0",
        "LM_RECALL_MIN_SCORE": "off",
        "LM_DELIVERY_SESSION_DEDUP": "0",
        "LM_DELIVERY_SNIPPET_LADDER": "2400,1000,700,500,300,200",
    }.items():
        monkeypatch.setenv(key, value)
    state = SimpleNamespace(entered=Event(), release=Event(), guard=RLock(),
                            hold=False, fail=False, batches=[], unlocked_sql=[])
    encoder = Encoder()

    def encode(texts):
        state.batches.append(len(texts))
        if state.hold and len(texts) > 1:
            assert not state.guard._is_owned()
            assert not state.store.connection.in_transaction
            state.entered.set()
            assert state.release.wait(10), "legacy encoder was not released"
        if state.fail:
            raise RuntimeError("legacy encoder failed")
        return encoder.embed_batch(texts)

    def build(store):
        state.service = MemoryRecallService(store, embedder=encoder)
        return state.service

    monkeypatch.setattr(server, "RLock", lambda: state.guard)
    monkeypatch.setattr(server, "MemoryRecallService", build)
    state.mcp = server.create_mcp_server(tmp_path / "memory.sqlite3")
    state.store = state.mcp.memory_store
    state.node = legacy(state.store)
    state.store.set_chunk_embedder(encode)
    state.batches.clear()

    def trace(statement):
        if not state.guard._is_owned():
            state.unlocked_sql.append(statement)

    state.store.connection.set_trace_callback(trace)
    try:
        yield state
        assert not state.unlocked_sql
    finally:
        state.release.set()
        state.store.connection.set_trace_callback(None)
        state.mcp.auto_consolidation.close()
        state.store.close()


async def call(client, tool, **arguments):
    return (await client.call_tool(tool, arguments)).data


async def recall(client, **arguments):
    return await call(client, "memory_recall", **{
        "query": QUERY, "scope": SCOPE, "depth": 0, "max_results": 1, **arguments,
    })


def entry(response):
    result, = response["results"]
    return result.get("node", result)


@pytest.mark.parametrize("view", ["knowledge", "full"])
def test_native_lookup_and_second_recall_finish_before_legacy_encoding(runtime, view):
    r = runtime
    r.hold = True

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as peer:
            pending = asyncio.create_task(recall(owner, view=view))
            try:
                assert await asyncio.to_thread(r.entered.wait, 5)
                with r.guard:
                    own_event = r.store.get_recall_event(r.service.last_recall_event_id)
                    assert own_event.results[0]["delivery"]["kind"] == "snippet"
                lookup = await asyncio.wait_for(call(peer, "memory_lookup", node_id=r.node.id), 3)
                assert entry(lookup)["content"] == CONTENT
                other = await asyncio.wait_for(recall(peer, query=QUERY + "?"), 3)
                assert other["recall_event_id"] != own_event.id
                assert FACT not in entry(other)["content"]
                assert r.service.last_recall_event_id == other["recall_event_id"]
                assert not pending.done()
                assert len(r.batches) == 1
            finally:
                r.release.set()
            response = await asyncio.wait_for(pending, 5)
            assert response["recall_event_id"] == own_event.id
            assert response["query"] == QUERY
            assert FACT not in entry(response)["content"]
            with r.guard:
                event = r.store.get_recall_event(own_event.id)
                assert event.to_dict() == own_event.to_dict()
                if view == "full":
                    assert response["results"][0]["vector_score"] == round(event.results[0]["vector_score"], 6)
                assert len(r.store.recall_delivery_history(event.transport_session_id)) == 1
                assert r.store.get_node(r.node.id).access_count == 2
                chunks = r.store.list_node_chunks(r.node.id)
                assert all(chunk.char_start is not None for chunk in chunks)
            following = await recall(owner)
            assert FACT in entry(following)["content"]
            assert len(r.batches) == 1
            assert 1 < r.batches[0] <= 32

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["edit", "delete", "decay", "scope", "generation"])
def test_concurrent_writer_wins_while_native_legacy_encoding_waits(runtime, change):
    r = runtime
    r.hold = True

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as peer:
            pending = asyncio.create_task(recall(owner))
            try:
                assert await asyncio.to_thread(r.entered.wait, 5)
                with MemoryStore(r.store.db_path, chunk_embedder=Encoder().embed_batch) as writer:
                    if change == "edit":
                        writer.update_node(r.node.id, content=FACT + " Revised condition.",
                                           embedding=[0.0, 1.0, 0.0])
                    elif change == "delete":
                        with writer.transaction():
                            writer.delete_node_chunks(r.node.id)
                            writer.connection.execute("DELETE FROM nodes WHERE id=?", (r.node.id,))
                    elif change == "decay":
                        await asyncio.wait_for(call(peer, "memory_forget", id=r.node.id), 3)
                    elif change == "scope":
                        writer.update_node(r.node.id, context={"scope": "project:moved"})
                    else:
                        writer.replace_node_chunks(r.node.id, [
                            (chunk, [0.0, 1.0, 0.0]) for chunk in chunk_text(CONTENT)
                        ])
                    expected_rows = rows(writer, r.node.id)
                    current = writer.get_node(r.node.id)
                    expected_node = current.to_dict() if current else None
                lookup = await asyncio.wait_for(call(peer, "memory_lookup", node_id=r.node.id), 3)
                assert bool(lookup["results"]) == (change != "delete")
                assert not pending.done()
            finally:
                r.release.set()
            response = await asyncio.wait_for(pending, 5)
            assert entry(response)["id"] == r.node.id
            assert FACT not in entry(response)["content"]
            assert entry(response)["scope"] == SCOPE
            with r.guard:
                assert rows(r.store, r.node.id) == expected_rows
                node = r.store.get_node(r.node.id)
                assert (node.to_dict() if node else None) == expected_node

    asyncio.run(scenario())


def test_response_map_and_explicit_credit_keep_their_own_event(runtime, monkeypatch):
    r = runtime
    monkeypatch.setenv("LM_RECALL_MAP", "1")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    with r.guard:
        prior = r.store.create_node(level="concept", content="Prior certificate rule.",
                                    context={"scope": "project:prior"}, embedding=[1.0, 0.0, 0.0])
        for index in range(4):
            r.store.create_node(level="concept", content=f"Amber logistics note {index}.",
                                context={"scope": SCOPE, "topic": "amber logistics"},
                                embedding=[0.6, 0.8, 0.0])
    builds = []
    build = RecallMapBuilder.build

    def observe_map(builder, residual, **kwargs):
        builds.append((tuple(result.node_id for result in residual), kwargs["relevance"]))
        return build(builder, residual, **kwargs)

    monkeypatch.setattr(RecallMapBuilder, "build", observe_map)

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as peer:
            earlier = await recall(owner, scope="project:prior")
            r.hold = True
            pending = asyncio.create_task(recall(owner, used=[prior.id], view="full"))
            try:
                assert await asyncio.to_thread(r.entered.wait, 5)
                with r.guard:
                    own_id = r.service.last_recall_event_id
                    own_residual = tuple(result.node_id for result in r.service.last_residual)
                    assert own_residual and builds == [(own_residual, r.service.last_query_relevance)]
                    stored_map = json.loads(r.store.connection.execute(
                        "SELECT recall_map FROM recall_events WHERE id=?", (own_id,),
                    ).fetchone()[0])
                    assert stored_map
                foreign = await asyncio.wait_for(recall(peer, scope="project:prior"), 3)
                own_later = await asyncio.wait_for(recall(owner, scope="project:prior", used=[prior.id]), 3)
                assert r.service.last_recall_event_id == own_later["recall_event_id"]
                assert not r.service.last_residual
                assert not pending.done()
            finally:
                r.release.set()
            response = await asyncio.wait_for(pending, 5)
            assert response["recall_event_id"] == own_id
            assert response["recall_map"] == stored_map
            assert response["feedback_marks"] == {"accepted": 1, "dropped": 0, "by_reason": {}}
            with r.guard:
                marks = r.store.connection.execute(
                    "SELECT recall_event_id, source_id FROM recall_feedback_marks WHERE node_id=?",
                    (prior.id,),
                ).fetchall()
                assert {tuple(mark) for mark in marks} == {
                    (earlier["recall_event_id"], own_id),
                    (earlier["recall_event_id"], own_later["recall_event_id"]),
                }
                credits = r.store.connection.execute(
                    "SELECT recall_event_id, source_id FROM recall_explicit_credit WHERE node_id=?",
                    (prior.id,),
                ).fetchall()
                assert [tuple(credit) for credit in credits] == [(earlier["recall_event_id"], own_id)]
                assert r.store.get_recall_event(own_id).result_ids == [r.node.id]
                assert r.store.get_recall_event(foreign["recall_event_id"]).result_ids == [prior.id]
                assert len(r.store.list_recall_events()) == 4

    asyncio.run(scenario())


def test_concurrent_failed_row_attempts_once_without_suppressing_another_source(runtime, caplog):
    r = runtime
    with r.guard:
        other = legacy(r.store, CONTENT + " Independent source.")
        r.store.update_node(other.id, context={"scope": "project:independent"})
    r.batches.clear()
    r.hold = r.fail = True

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as peer:
            pending = asyncio.create_task(recall(owner))
            try:
                assert await asyncio.to_thread(r.entered.wait, 5)
                same = await asyncio.wait_for(recall(peer), 3)
                assert FACT not in entry(same)["content"]
                assert len(r.batches) == 1
                assert not pending.done()
            finally:
                r.release.set()
            await asyncio.wait_for(pending, 5)
            await recall(peer)
            assert len(r.batches) == 1
            r.hold = r.fail = False
            repaired = await recall(peer, scope="project:independent")
            assert entry(repaired)["id"] == other.id
            assert FACT not in entry(repaired)["content"]
            following = await recall(peer, scope="project:independent")
            assert FACT in entry(following)["content"]
            assert len(r.batches) == 2
            warnings = [record for record in caplog.records
                        if "could not refresh selected legacy" in record.message]
            assert len(warnings) == 1

    asyncio.run(scenario())
