"""Passage reads through ordinary FastMCP delivery on isolated source generations."""

import asyncio
from collections import Counter
import sqlite3

from fastmcp import Client
import pytest

from fixture_vectors import fixture_vector
from living_memory import server
from living_memory.chunking import TextChunk
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore
from test_semantic_passage_delivery import BACKGROUND, FACT, QUERY, SCOPE, QueryEmbedder, put_match


@pytest.fixture
def app(tmp_path, monkeypatch):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash", "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic",
        "LM_DELIVERY_SESSION_DEDUP": "1", "LM_DELIVERY_SNIPPET_LADDER": "300",
        "LM_AUTO_CONSOLIDATE_POLICY": "off", "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
        "LM_RECALL_MAP": "0", "LM_RECALL_MIN_SCORE": "0.15",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(server, "MemoryRecallService", lambda store: MemoryRecallService(
        store, embedder=QueryEmbedder(),
    ))
    mcp = server.create_mcp_server(tmp_path / "memory.sqlite3")
    try:
        yield mcp
    finally:
        mcp.auto_consolidation.close()
        mcp.memory_store.close()


async def recall(client, query=QUERY, limit=10, view="knowledge"):
    return (await client.call_tool("memory_recall", {
        "query": query, "scope": SCOPE, "depth": 0, "max_results": limit, "view": view,
    })).data


def test_short_full_results_survive_unavailable_coordinates(app, monkeypatch):
    store = app.memory_store
    nodes = [store.append_trace(f"Crimson walrus recall number {i}.", {"scope": SCOPE})
             for i in range(3)]
    calls = []

    def unavailable(ids):
        calls.append(ids)
        raise sqlite3.OperationalError("coordinates unavailable")

    monkeypatch.setattr(store, "get_chunk_passages", unavailable)

    async def scenario():
        async with Client(app) as client:
            return await recall(client, "Crimson walrus recall", 3)

    rows = asyncio.run(scenario())["results"]
    assert {row["id"]: (row["delivery"], row["content"]) for row in rows} == {
        node.id: ("full", node.content) for node in nodes
    }
    assert calls == []


@pytest.mark.parametrize("view", ["knowledge", "full"])
def test_mixed_consumers_and_concurrent_generation(app, monkeypatch, view):
    store = app.memory_store
    read = store.get_chunk_passages
    long_body = BACKGROUND + "\n\n" + FACT

    def add(body, level="trace"):
        start = body.index("open the amber") if FACT in body else 0
        return put_match(store, body, start, start + 10, level=level)

    seen = add("Already delivered. " + long_body)
    monkeypatch.setenv("LM_DELIVERY_SNIPPET_LADDER", "full")

    async def scenario():
        async with Client(app) as client:
            assert (await recall(client))["results"][0]["delivery"] == "full"
            monkeypatch.setenv("LM_DELIVERY_SNIPPET_LADDER", "300")
            twins = [add(long_body) for _ in range(2)]
            short = add("Brief note.")
            other = add("Independent source. " + long_body, "concept")
            recipe = add("Recipe. " + long_body, "schema")
            weak = add("Unrelated archived note.")
            store.replace_node_chunks(weak.id, [(
                TextChunk(weak.content, 0, 0, 5, 0, len(weak.content)), fixture_vector([0.15, 0.99, 0]),
            )])
            chunks = {chunk.id: node.id for node in [*twins, other, short, recipe, weak, seen]
                      for chunk in store.list_node_chunks(node.id)}
            calls, sql = [], []

            def concurrent_read(ids):
                calls.append([chunks[key] for key in ids])
                # Replace a required source after scoring, before resolving its old row.
                with MemoryStore(store.db_path) as writer:
                    writer.update_node(other.id, content="Changed after selection.")
                return read(ids)

            monkeypatch.setattr(store, "get_chunk_passages", concurrent_read)
            store.connection.set_trace_callback(sql.append)
            try:
                answer = await recall(client, view=view)
            finally:
                store.connection.set_trace_callback(None)
            rows = [(row.get("node", row), row["delivery"]) for row in answer["results"]]
            kinds = {body["id"]: kind for body, kind in rows}
            assert Counter(kinds.values()) == {
                "full": 2, "snippet": 2, "session_duplicate": 1, "twin_duplicate": 1,
                "below_threshold": 1,
            }
            assert (kinds[recipe.id], kinds[short.id], kinds[weak.id], kinds[seen.id]) == (
                "full", "full", "below_threshold", "session_duplicate",
            )
            snippets = [body for body, kind in rows if kind == "snippet"]
            assert other.id in {body["id"] for body in snippets}
            assert all(FACT in body["content"] for body in snippets)
            assert calls == [[body["id"] for body in snippets]]
            assert sum("SELECT c.id, c.content_fingerprint, c.char_start" in stmt for stmt in sql) == 1
            event = store.get_recall_event(answer["recall_event_id"])
            assert [row["node_id"] for row in event.results] == [body["id"] for body, _ in rows]
            assert [row["delivery"]["kind"] for row in event.results] == [kind for _, kind in rows]
            assert store.get_node(other.id).content == "Changed after selection."

    asyncio.run(scenario())
