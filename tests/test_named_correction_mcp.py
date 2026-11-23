"""Named preference and correction precedence at the public recall cut."""

import asyncio

from fastmcp import Client
import pytest

from living_memory.server import create_mcp_server

SCOPE = "project:names"
QUERY = "rotate vault keys"


@pytest.fixture
def mcp(tmp_path, monkeypatch):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
        "LM_AUTO_CONSOLIDATE_POLICY": "off",
        "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
        "LM_DELIVERY_SESSION_DEDUP": "0",
        "LM_RECALL_MAP": "0",
        "LM_RECALL_MIN_SCORE": "off",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("LM_RECALL_SCHEMA_TRIGGER", raising=False)
    server = create_mcp_server(tmp_path / "memory.sqlite3")
    try:
        yield server
    finally:
        server.auto_consolidation.close()
        server.memory_store.close()


@pytest.mark.parametrize("named", [False, True], ids=["default", "name"])
@pytest.mark.parametrize("case", [
    "connected", "qualified", "independent", "partial", "outside", "decayed", "threshold",
])
def test_public_recall_order(mcp, monkeypatch, named, case):
    if named:
        monkeypatch.setenv("LM_RECALL_SCHEMA_TRIGGER", "name")
    if case == "threshold":
        monkeypatch.setenv("LM_RECALL_MIN_SCORE", "1")
    elif case == "qualified":
        monkeypatch.setenv("LM_RECALL_MIN_SCORE", "0.05")
    store = mcp.memory_store
    schema = store.create_node(
        level="schema", content="Rotate a spare key during inventory.",
        context={"scope": SCOPE, "trigger": QUERY},
    )
    other = store.append_trace(
        "Rotate vault keys only after sealing the vault.",
        {"scope": "project:outside" if case == "outside" else SCOPE},
    )
    linked = case not in ("independent", "partial")
    if linked:
        store.create_connection(other.id, schema.id, "supersedes")
    if case == "decayed":
        store.soft_delete_node(other.id)

    if case in ("outside", "decayed"):
        expected = [schema]
    elif named and case == "independent":
        expected = [schema, other]
    else:
        expected = [other, schema]

    query = QUERY + " after sealing" if case == "partial" else QUERY

    async def read():
        async with Client(mcp) as client:
            for limit in (1, 2):
                response = (await client.call_tool("memory_recall", {
                    "query": query, "scope": SCOPE, "max_results": limit,
                })).data
                selected = expected[:limit]
                assert response["count"] == len(selected)
                assert [r["id"] for r in response["results"]] == [n.id for n in selected]
                for item, node in zip(response["results"], selected, strict=True):
                    stale = node == schema and linked
                    assert item["superseded"] is stale
                    assert item["superseded_by"] == ([other.id] if stale and case != "outside" else [])
                    assert item["scope"] == SCOPE and not item["decayed"]
                    assert item["content_ref"]["node_id"] == node.id
                    if case == "threshold":
                        assert item["delivery"] == "below_threshold"
                        assert item["content"] == ""
                    else:
                        assert item["delivery"] == "full"
                        assert item["content"] == node.content

    asyncio.run(read())
