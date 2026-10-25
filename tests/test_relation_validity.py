"""Relation-only corrections retain validity at the real MCP boundary."""

import asyncio
import json

import pytest

pytest.importorskip("fastmcp")
from fastmcp import Client

from living_memory.server import create_mcp_server
from test_transport_identity import _structured


SCOPE = "project:relation-validity"
QUERY = "amber hatch pressure rule"
ORIGINAL = "Amber hatch pressure rule: open first.\n\n  Исходник — сохранить.\t\n"


async def remember(client, content, scope=SCOPE):
    response = _structured(await client.call_tool("memory_remember", {
        "content": content, "context": {"scope": scope, "detail": "original context"},
    }))
    return response["node"]["id"]


async def read(client, path, node_id, *, scope=SCOPE):
    tool, view = path.split(":")
    arguments = {"scope": scope, "view": view}
    if tool == "recall":
        arguments.update(query=QUERY, depth=0, max_results=20)
    else:
        arguments["node_id"] = node_id
    response = _structured(await client.call_tool(f"memory_{tool}", arguments))
    return next(
        entry for entry in response["results"]
        if entry.get("node", entry)["id"] == node_id
    ), response


@pytest.mark.parametrize("path", [
    "recall:knowledge", "recall:full", "lookup:knowledge", "lookup:full",
])
@pytest.mark.parametrize("correction_state", ["active", "decayed", "foreign", "missing"])
def test_relation_only_correction_validity(tmp_path, monkeypatch, path, correction_state):
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store

    async def scenario():
        async with Client(mcp) as client:
            original = await remember(client, ORIGINAL)
            correction = await remember(
                client,
                f"Correction to {original}: amber hatch pressure rule requires release first. "
                "CORRECTION_BODY_SENTINEL",
                "project:foreign-validity" if correction_state == "foreign" else SCOPE,
            )
            edges = store.list_connections(
                source_id=correction, target_id=original, relation_type="supersedes",
            )
            assert len(edges) == 1 and edges[0].metadata["rule"] == "R1c"
            assert store.get_node(original).corrections == []
            if correction_state == "decayed":
                store.soft_delete_node(correction, "expired correction")
            elif correction_state == "missing":
                # Model an unreadable counterpart while retaining its known relation.
                get_nodes = store.get_nodes
                monkeypatch.setattr(store, "get_nodes", lambda ids: {
                    key: node for key, node in get_nodes(ids).items() if key != correction
                })

            entry, response = await read(client, path, original)
            assert entry["superseded"] is True
            expected = [correction] if correction_state in ("active", "decayed") else []
            assert entry["superseded_by"] == expected
            body = entry.get("node", entry)
            assert body["provenance"]["corrections"] == []
            assert body["content"] == ORIGINAL
            if correction_state == "foreign":
                assert correction not in json.dumps(response)
                assert "CORRECTION_BODY_SENTINEL" not in json.dumps(response)
            if correction_state == "decayed" and path.startswith("recall"):
                assert all(e.get("node", e)["id"] != correction for e in response["results"])
            if expected:
                fetched = _structured(await client.call_tool("memory_lookup", {
                    "node_id": expected[0], "scope": SCOPE,
                }))["results"][0]
                assert fetched["id"] == correction
                assert fetched["decayed"] is (correction_state == "decayed")
                assert fetched["superseded"] is False
            assert store.get_node(original).content == ORIGINAL
            assert store.get_node(original).corrections == []

    try:
        asyncio.run(scenario())
    finally:
        store.close()


@pytest.mark.parametrize("path", [
    "recall:knowledge", "recall:full", "lookup:knowledge", "lookup:full",
])
@pytest.mark.parametrize("cycle", [False, True])
def test_direct_correction_links_preserve_chains_cycles_and_unaffected_nodes(tmp_path, path, cycle):
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store

    async def scenario():
        async with Client(mcp) as client:
            original = await remember(client, ORIGINAL)
            middle = await remember(client, f"Correction to {original}: amber hatch pressure rule B.")
            latest = await remember(client, f"Correction to {middle}: amber hatch pressure rule C.")
            untouched = await remember(client, "Amber hatch pressure rule: inspect the seal.")
            if cycle:
                store.add_connection(original, latest, "supersedes")
            expected = {original: [middle], middle: [latest], latest: [original] if cycle else [], untouched: []}
            for node_id, correction_ids in expected.items():
                entry, _ = await read(client, path, node_id)
                assert entry["superseded"] is bool(correction_ids)
                assert entry["superseded_by"] == correction_ids
                assert entry.get("node", entry)["provenance"]["corrections"] == []
            fetched = _structured(await client.call_tool("memory_lookup", {
                "node_id": original, "context_key": "detail", "scope": "project:other",
            }))
            assert fetched["results"][0]["content"] == ORIGINAL
            assert fetched["results"][0]["context"] == {"detail": "original context"}
            assert fetched["results"][0]["superseded"] is True
            assert fetched["results"][0]["superseded_by"] == []
            assert fetched["scope_mismatches"] == [{"id": original, "scope": SCOPE}]
            unscoped = _structured(await client.call_tool("memory_lookup", {"node_id": original}))
            assert unscoped["results"][0]["superseded_by"] == [middle]

    try:
        asyncio.run(scenario())
    finally:
        store.close()
