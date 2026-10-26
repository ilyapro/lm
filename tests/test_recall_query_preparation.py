"""Query encoding leaves stored knowledge available to independent clients."""

import asyncio
from threading import Event
from types import SimpleNamespace

from fastmcp import Client
from fastmcp.exceptions import ToolError
import pytest

from living_memory import embeddings, server
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore


QUERY = "cobalt certificate"
CONTENT = "Cobalt certificate must be renewed before release."
SCOPE = "project:query-preparation"
VECTOR = [1.0, 0.0, 0.0]


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
        "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic",
        "LM_AUTO_CONSOLIDATE_POLICY": "off",
        "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
        "LM_RECALL_MAP": "0",
        "LM_RECALL_MIN_SCORE": "off",
    }.items():
        monkeypatch.setenv(key, value)
    state = SimpleNamespace(entered=Event(), release=Event(), hold=False, fail=False, encoded=[])

    def encode(text, dimensions):
        state.encoded.append(text)
        if text == QUERY:
            if state.hold:
                state.entered.set()
                assert state.release.wait(10), "query encoder was not released"
            if state.fail:
                raise RuntimeError("query encoding failed")
        return list(VECTOR)

    def service(store):
        state.service = MemoryRecallService(store, embedder=LocalEmbeddingModel(dimensions=3, backend="hash"))
        return state.service

    monkeypatch.setattr(embeddings, "_embed_hash", encode)
    monkeypatch.setattr(server, "MemoryRecallService", service)
    state.mcp = server.create_mcp_server(tmp_path / "memory.sqlite3")
    state.store = state.mcp.memory_store
    state.node = state.store.create_node(level="concept", content=CONTENT,
                                        context={"scope": SCOPE}, embedding=VECTOR)
    state.encoded.clear()
    try:
        yield state
    finally:
        state.release.set()
        state.mcp.auto_consolidation.close()
        state.store.close()


async def call(client, name, **arguments):
    return (await client.call_tool(name, arguments)).data


async def recall(client, **arguments):
    return await call(client, "memory_recall", **{
        "query": QUERY, "scope": SCOPE, "max_results": 1, **arguments,
    })


def entry(response):
    result, = response["results"]
    return result.get("node", result)


@pytest.mark.parametrize("view", ["knowledge", "full"])
def test_native_exact_lookup_finishes_while_query_encoder_is_held(runtime, view):
    r = runtime
    r.hold = True

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as peer:
            pending = asyncio.create_task(recall(owner, query=" \n" + QUERY + " \t", view=view))
            try:
                assert await asyncio.to_thread(r.entered.wait, 5)
                assert not r.store.connection.in_transaction
                lookup = await asyncio.wait_for(call(peer, "memory_lookup", node_id=r.node.id), 3)
                assert entry(lookup)["content"] == CONTENT
                assert not pending.done() and not r.release.is_set()
                status = await asyncio.wait_for(call(peer, "memory_status", scope=SCOPE), 3)
                assert status
            finally:
                r.release.set()
            response = await asyncio.wait_for(pending, 5)
            assert response["query"] == " \n" + QUERY + " \t"
            assert entry(response)["id"] == r.node.id
            assert entry(response)["content"] == CONTENT
            event = r.store.get_recall_event(response["recall_event_id"])
            assert event.query == QUERY
            assert event.transport_session_id
            assert event.result_ids == [r.node.id]
            assert event.results[0]["vector_score"] == 1.0
            assert len(r.store.recall_delivery_history(event.transport_session_id)) == 1
            assert r.store.get_node(r.node.id).access_count == 1
            assert r.encoded == [QUERY]

    asyncio.run(scenario())


def test_recall_reads_current_source_scope_and_validity_after_encoding(runtime):
    r = runtime
    obsolete = r.store.create_node(level="concept", content=CONTENT + " Old rule.",
                                   context={"scope": SCOPE}, embedding=VECTOR)
    r.hold = True
    updated = "Cobalt certificate now requires a countersignature."
    new_scope = "project:current-query-source"

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as peer:
            pending = asyncio.create_task(recall(owner, scope=None, max_results=5, view="full"))
            try:
                assert await asyncio.to_thread(r.entered.wait, 5)
                forgotten = await asyncio.wait_for(call(peer, "memory_forget", id=obsolete.id), 3)
                assert forgotten["node"]["decayed"]
                # Only this disposable database is mutated. The supported store
                # writer changes both source and scope while encoding is held.
                with MemoryStore(r.store.db_path) as writer:
                    writer.update_node(r.node.id, content=updated, context={"scope": new_scope},
                                       embedding=VECTOR)
                lookup = await asyncio.wait_for(call(peer, "memory_lookup", node_id=r.node.id), 3)
                assert entry(lookup)["content"] == updated
                assert not pending.done()
            finally:
                r.release.set()
            response = await asyncio.wait_for(pending, 5)
            assert entry(response)["id"] == r.node.id
            assert entry(response)["content"] == updated
            assert entry(response)["scope"] == new_scope
            assert not entry(response).get("decayed", False)
            assert not r.store.get_node(r.node.id).decayed
            event = r.store.get_recall_event(response["recall_event_id"])
            assert new_scope in event.resolved_scopes
            assert event.result_ids == [r.node.id]
            assert event.results[0]["vector_score"] == 1.0
            assert r.encoded == [QUERY]

    asyncio.run(scenario())


@pytest.mark.parametrize("query,max_results", [("", 1), (" \n\t", 1), (QUERY, 0), (QUERY, -1)])
def test_empty_recall_does_not_encode_or_record_event(runtime, query, max_results):
    r = runtime
    r.service.last_recall_event_id = "previous"
    r.service.last_residual = [object()]
    r.service.last_query_relevance = object()
    assert r.service.memory_recall(query, max_results=max_results) == []
    assert r.service.last_recall_event_id is None
    assert r.service.last_residual == []
    assert r.service.last_query_relevance is None

    async def scenario():
        async with Client(r.mcp) as client:
            response = await recall(client, query=query, max_results=max_results)
            assert response["count"] == 0
            assert response["recall_event_id"] is None

    asyncio.run(scenario())
    assert r.encoded == []
    assert r.store.list_recall_events() == []


def test_direct_service_normalizes_once_and_propagates_encoding_failure(runtime):
    r = runtime
    result, = r.service.memory_recall(" \n" + QUERY + " \t", scope=SCOPE)
    assert result.node_id == r.node.id
    assert result.vector_score == 1.0
    assert r.encoded == [QUERY]
    r.fail = True
    with pytest.raises(RuntimeError, match="query encoding failed"):
        r.service.memory_recall(QUERY, scope=SCOPE)
    assert r.service.last_recall_event_id is None
    assert r.service.last_residual == []
    assert r.service.last_query_relevance is None
    assert len(r.store.list_recall_events()) == 1
    assert r.store.get_node(r.node.id).access_count == 1


def test_native_encoding_failure_does_not_consume_marks_or_record_delivery(runtime, monkeypatch):
    r = runtime
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")

    async def scenario():
        async with Client(r.mcp) as client:
            first = await recall(client)
            event = r.store.get_recall_event(first["recall_event_id"])
            history = r.store.recall_delivery_history(event.transport_session_id)
            r.fail = True
            with pytest.raises(ToolError, match="query encoding failed"):
                await recall(client, used=[r.node.id])
            assert len(r.store.list_recall_events()) == 1
            assert r.store.get_node(r.node.id).access_count == 1
            assert r.store.recall_delivery_history(event.transport_session_id) == history
            assert r.store.connection.execute("SELECT count(*) FROM recall_feedback_marks").fetchone()[0] == 0
            assert r.store.connection.execute("SELECT count(*) FROM recall_explicit_credit").fetchone()[0] == 0
            r.fail = False
            resumed = await recall(client, used=[r.node.id])
            assert resumed["feedback_marks"]["accepted"] == 1
            mark, = r.store.connection.execute("SELECT * FROM recall_feedback_marks").fetchall()
            assert mark["recall_event_id"] == first["recall_event_id"]
            assert mark["source_id"] == resumed["recall_event_id"]

    asyncio.run(scenario())


def test_invalid_view_rejects_before_encoding(runtime):
    r = runtime

    async def scenario():
        async with Client(r.mcp) as client:
            with pytest.raises(ToolError):
                await recall(client, view="unknown")

    asyncio.run(scenario())
    assert r.encoded == []
    assert r.store.list_recall_events() == []


def test_prepared_query_uses_current_delivery_and_feedback_attribution(runtime, monkeypatch):
    r = runtime
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "audit")
    prepare = r.service.prepare_query

    def hold_prepared(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        if not r.entered.is_set():
            r.entered.set()
            assert r.release.wait(10), "prepared query was not released"
        return prepared

    monkeypatch.setattr(r.service, "prepare_query", hold_prepared)

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as peer:
            pending = asyncio.create_task(recall(owner, used=[r.node.id]))
            try:
                assert await asyncio.to_thread(r.entered.wait, 5)
                assert r.service.last_recall_event_id is None
                foreign = await asyncio.wait_for(recall(peer), 3)
                own = await asyncio.wait_for(recall(owner), 3)
                assert foreign["results"][0]["delivery"] == "full"
                assert own["results"][0]["delivery"] == "full"
                assert r.service.last_recall_event_id == own["recall_event_id"]
                assert not pending.done()
            finally:
                r.release.set()
            response = await asyncio.wait_for(pending, 5)
            assert response["results"][0]["delivery"] == "session_duplicate"
            assert response["feedback_marks"] == {"accepted": 1, "dropped": 0, "by_reason": {}}
            event = r.store.get_recall_event(response["recall_event_id"])
            own_event = r.store.get_recall_event(own["recall_event_id"])
            foreign_event = r.store.get_recall_event(foreign["recall_event_id"])
            assert event.id != own_event.id != foreign_event.id
            assert event.transport_session_id == own_event.transport_session_id
            assert event.transport_session_id != foreign_event.transport_session_id
            mark, = r.store.connection.execute("SELECT * FROM recall_feedback_marks").fetchall()
            assert mark["recall_event_id"] == own_event.id
            assert mark["source_id"] == event.id
            assert mark["transport_session_id"] == event.transport_session_id
            assert len(r.store.list_recall_events()) == 3
            assert r.encoded == [QUERY, QUERY, QUERY]

    asyncio.run(scenario())
