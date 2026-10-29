"""Native recalls advance bounded repairs across clients and scope changes."""

import asyncio

from fastmcp import Client
import pytest

from living_memory import retrieval, server, storage
from living_memory.retrieval import LAZY_CHUNK_NODE_LIMIT
from test_recall_query_preparation import SCOPE, VECTOR, entry, recall, runtime


@pytest.mark.parametrize("reset", ["none", "restart", "eviction"])
@pytest.mark.parametrize("consumer", ["recall", "prompt"])
@pytest.mark.parametrize("fault", ["encoding", "hydration"])
def test_native_pending_progress_scope_isolation_and_retry(runtime, monkeypatch, reset, consumer, fault):
    r = runtime
    if reset == "eviction":
        monkeypatch.setattr(retrieval, "LAZY_CHUNK_PLAN_LIMIT", 1)
    limit = LAZY_CHUNK_NODE_LIMIT

    def source(label, scope, timestamp):
        return r.store.create_node(
            level="concept", content=f"{label}. " * 150,
            context={"scope": scope}, timestamp=timestamp,
        )

    failures = [source(f"Failing source {index}", SCOPE, "2026-01-02T00:00:00Z")
                for index in range(limit)]
    tail = source("Healthy tail", SCOPE, "2026-01-03T00:00:00Z")
    other_scope = "project:independent"
    earlier = source("Earlier independent source", other_scope, "2026-01-01T00:00:00Z")
    later = [source(f"Later independent source {index}", other_scope, "2026-01-04T00:00:00Z")
             for index in range(limit)]
    attempts = []
    fail = True
    original = r.store.list_node_chunks(r.node.id)
    timestamps = {n.id: (n.timestamp, n.updated_at) for n in failures}
    if fault == "hydration":
        with r.store.transaction():
            r.store.connection.executemany("UPDATE nodes SET provenance = '[]' WHERE id = ?",
                                          [(n.id,) for n in failures])
        hydrate = storage._node_from_row

        def decode(row):
            if row["id"] in timestamps and row["provenance"] == "[]":
                attempts.append(row["content"].split(".")[0])
            return hydrate(row)

        monkeypatch.setattr(storage, "_node_from_row", decode)

    def encode(texts):
        assert not r.store.connection.in_transaction
        attempts.append(texts[0].split(".")[0])
        if fault == "encoding" and fail and texts[0].startswith("Failing source"):
            raise RuntimeError("injected source-local encoding failure")
        return [VECTOR for _ in texts]

    r.store.set_chunk_embedder(encode)

    async def read(scope=SCOPE):
        attempts.clear()
        async with Client(r.mcp) as client:
            if consumer == "recall":
                response = await recall(client, scope=scope)
            else:
                response = await client.get_prompt("memory://prompt/retrieval_context", {
                    "task": "cobalt certificate", "scope": scope,
                })
        assert len(attempts) <= limit
        assert not r.store.connection.in_transaction
        assert r.store.list_node_chunks(r.node.id) == original
        if scope == SCOPE:
            if consumer == "recall":
                assert entry(response)["id"] == r.node.id
            else:
                assert r.node.id in response.messages[0].content.text
        return list(attempts)

    async def scenario():
        nonlocal fail
        first = await read()
        assert len(first) == limit and all(text.startswith("Failing source") for text in first)
        assert not r.store.node_chunks_are_current(tail.id)

        await read(other_scope)
        assert r.store.node_chunks_are_current(earlier.id)
        assert sum(r.store.node_chunks_are_current(node.id) for node in later) == limit - 1

        if reset == "restart":
            path = r.store.db_path
            r.mcp.auto_consolidation.close()
            r.store.close()
            r.mcp = server.create_mcp_server(path)
            r.store = r.mcp.memory_store
            r.store.set_chunk_embedder(encode)
        if reset != "none":
            assert await read() == first
            assert not r.store.node_chunks_are_current(tail.id)

        assert (await read())[0] == "Healthy tail"
        assert r.store.node_chunks_are_current(tail.id)
        assert all(not r.store.node_chunks_are_current(n.id) for n in failures)
        assert {row["id"]: (row["timestamp"], row["updated_at"])
                for row in r.store.connection.execute("SELECT id, timestamp, updated_at FROM nodes")
                if row["id"] in timestamps} == timestamps

        fail = False
        if fault == "hydration":
            with r.store.transaction():
                r.store.connection.executemany("UPDATE nodes SET provenance = '{}' WHERE id = ?",
                                              [(n.id,) for n in failures])
        await read()
        assert all(r.store.node_chunks_are_current(n.id) for n in failures)
        await read(other_scope)
        assert all(r.store.node_chunks_are_current(n.id) for n in later)
        assert r.store.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    asyncio.run(scenario())
