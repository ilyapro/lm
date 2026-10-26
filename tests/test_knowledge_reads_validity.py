"""Knowledge reads preserve relation validity and their own scope/selection."""

import asyncio
import json

import pytest

pytest.importorskip("fastmcp")
from fastmcp import Client

from living_memory.server import create_mcp_server
from test_transport_identity import _structured


@pytest.mark.parametrize("surface", ["filter", "recent", "concepts", "prompt"])
@pytest.mark.parametrize("scope", ["project:read-validity", "global"])
@pytest.mark.parametrize("correction_state", ["active", "decayed", "missing", "foreign", "global"])
def test_native_knowledge_reads_share_relation_validity(
    tmp_path, monkeypatch, surface, scope, correction_state,
):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    level = "concept" if surface in ("concepts", "prompt") else "trace"
    original_content = "Amber hatch pressure rule: open first.\n\n  Исходник.\t\n"
    context = {
        "scope": scope, "task_pattern": "amber-hatch", "procedure_id": "amber-operation",
        "detail": {"nested": ["full context", "x" * 800]},
    }

    async def scenario():
        async with Client(mcp) as client:
            async def remember(content, context):
                result = _structured(await client.call_tool("memory_remember", {
                    "content": content, "context": context,
                }))
                return result["node"]["id"]

            original = await remember(original_content, context)
            untouched = await remember("Amber hatch seal: inspect before opening.", context)
            foreign = await remember("Foreign selection sentinel.", {
                **context, "scope": "project:foreign-validity",
            })
            for node_id in (original, untouched, foreign):
                store.update_node(node_id, level=level)

            async def read_surface():
                before = list(store.connection.iterdump())
                if surface == "filter":
                    response = _structured(await client.call_tool("memory_lookup", {
                        "scope": scope, "task_pattern": "amber-hatch", "level": level,
                    }))
                    entries = response["results"]
                    expected = store.list_nodes_by_context(
                        scope=scope, context_filters={"task_pattern": "amber-hatch"}, level=level,
                    )
                elif surface == "prompt":
                    response = await client.get_prompt("memory://prompt/retrieval_context", {
                        "scope": scope, "task": "",
                    })
                    body = response.messages[0].content.text
                    assert foreign not in body
                    assert original in body and untouched in body
                    assert list(store.connection.iterdump()) == before
                    return body
                else:
                    uri = "memory://recent" if surface == "recent" else (
                        "memory://global/concepts" if scope == "global"
                        else "memory://project/read-validity/concepts"
                    )
                    response = json.loads((await client.read_resource(uri))[0].text)
                    entries = response["traces" if surface == "recent" else "concepts"]
                    expected = store.list_nodes(
                        level=level, scope=None if surface == "recent" else scope,
                        include_decayed=False, limit=20 if surface == "recent" else 100_000,
                    )
                assert [entry["id"] for entry in entries] == [node.id for node in expected]
                assert response["count"] == len(entries)
                assert list(store.connection.iterdump()) == before
                return {entry["id"]: entry for entry in entries}

            before = await read_surface()
            exact_before = _structured(await client.call_tool("memory_lookup", {
                "node_id": original, "scope": scope,
            }))["results"][0]
            assert exact_before["superseded"] is False
            assert exact_before["superseded_by"] == []
            if surface != "prompt":
                assert before[original].get("superseded", False) is False
                assert before[original]["content"] == original_content

            correction_scope = {
                "foreign": "project:foreign-validity", "global": "global",
            }.get(correction_state, scope)
            correction = await remember(
                f"Correction to {original}: release pressure first. CORRECTION_BODY_SENTINEL",
                {"scope": correction_scope},
            )
            edges = store.list_connections(
                source_id=correction, target_id=original, relation_type="supersedes",
            )
            assert len(edges) == 1 and edges[0].metadata["rule"] == "R1c"
            assert store.get_node(original).corrections == []
            if correction_state == "decayed":
                store.soft_delete_node(correction, "expired correction")
            elif correction_state == "missing":
                # Retain the known edge while its counterpart cannot be read.
                get_nodes = store.get_nodes
                monkeypatch.setattr(store, "get_nodes", lambda ids: {
                    key: node for key, node in get_nodes(ids).items() if key != correction
                })

            after = await read_surface()
            readable = correction_state != "missing" and (
                surface == "recent" or correction_scope == scope
                or (surface == "prompt" and correction_scope == "global")
            )
            references = [correction] if readable else []
            if surface == "prompt":
                original_line = next(line for line in after.splitlines() if f"id={original} " in line)
                control_line = next(line for line in after.splitlines() if f"id={untouched} " in line)
                assert "superseded=true" in original_line
                assert f"superseded_by={json.dumps(references)}" in original_line
                assert "superseded=false superseded_by=[]" in control_line
                assert "CORRECTION_BODY_SENTINEL" not in after
            else:
                entry = after[original]
                assert entry["superseded"] is True
                assert entry["superseded_by"] == references
                assert entry["decayed"] is False
                assert entry["provenance"]["corrections"] == []
                assert entry["content"] == original_content
                assert entry["context"]["detail"] == context["detail"]
                assert {k: v for k, v in entry.items() if k not in ("superseded", "superseded_by")} == {
                    k: v for k, v in before[original].items() if k not in ("superseded", "superseded_by")
                }
                assert after[untouched]["superseded"] is False
                assert after[untouched]["superseded_by"] == []
            if not readable and surface != "recent":
                assert correction not in json.dumps(after)
                assert "CORRECTION_BODY_SENTINEL" not in json.dumps(after)

            exact_after = _structured(await client.call_tool("memory_lookup", {
                "node_id": original, "scope": scope,
            }))["results"][0]
            assert exact_after["superseded"] is True
            assert exact_after["superseded_by"] == (
                [correction] if correction_state != "missing" and correction_scope in (scope, "global") else []
            )

    try:
        asyncio.run(scenario())
    finally:
        store.close()
