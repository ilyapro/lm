"""Visible graph neighbors survive stronger edges across scope boundaries."""

import asyncio
import json

import pytest
from fastmcp import Client

from living_memory.server import create_mcp_server


async def graph_response(path, *, count, reverse, broad):
    server = create_mcp_server(path)
    store = server.memory_store
    try:
        with store.transaction():
            for key, content, scope in (
                ("root", "zygomorph instigator", "project:p"),
                ("target", "vermilion silks under moonlight", "global"),
            ):
                store.create_node(level="trace", node_id=key, content=content, context={"scope": scope})
            def connect(neighbor, weight):
                endpoints = (neighbor, "root") if reverse else ("root", neighbor)
                store.create_connection(*endpoints, "caused", weight=weight)

            connect("target", 0.9)
            for i in range(200):
                key = f"foreign-{i:03}"
                store.create_node(level="trace", node_id=key, content=f"saturn lamp {i}",
                                  context={"scope": "project:other"})
                if i < count:
                    connect(key, 1.0)

        async with Client(server) as client:
            args = {} if broad else {"scope": "project:p"}
            response = await client.call_tool("memory_recall", {
                "query": "zygomorph", "depth": "causal", "max_results": 5, **args,
            })
            payload = dict(response.data)
            # Each independent delivery records a fresh event.
            assert payload.pop("recall_event_id")
            return payload
    finally:
        server.auto_consolidation.close()
        store.close()


@pytest.mark.parametrize("reverse", [False, True])
def test_scoped_causal_response_ignores_foreign_edges(tmp_path, monkeypatch, reverse):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    responses = [asyncio.run(graph_response(tmp_path / f"{count}.db", count=count,
                                            reverse=reverse, broad=False))
                 for count in (0, 199, 200)]
    assert [row["id"] for row in responses[0]["results"]] == ["root", "target"]
    assert responses[0] == responses[1] == responses[2]
    payload = json.dumps(responses[2])
    assert all(secret not in payload for secret in ("foreign-", "saturn lamp", "project:other"))
