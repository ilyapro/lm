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
            await client.call_tool("memory_connect", {
                "id_a": correction, "id_b": original, "relation_type": "supersedes",
            })
            edges = store.list_connections(
                source_id=correction, target_id=original, relation_type="supersedes",
            )
            assert len(edges) == 1 and edges[0].metadata == {}
            assert store.get_node(original).corrections == []
            if correction_state == "decayed":
                store.soft_delete_node(correction, "expired correction")
            elif correction_state == "missing":
                # Retain the known edge while its counterpart is absent.
                get_node_scopes = store.get_node_scopes
                monkeypatch.setattr(store, "get_node_scopes", lambda ids: {
                    key: scope for key, scope in get_node_scopes(ids).items() if key != correction
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


@pytest.mark.parametrize("writer", ["teach", "historical_duplicate"])
def test_native_reads_preserve_teach_and_historical_relation_semantics(tmp_path, monkeypatch, writer):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    scope = "project:read-validity"

    async def scenario():
        async with Client(mcp) as client:
            original = _structured(await client.call_tool("memory_remember", {
                "content": "Amber hatch opening is allowed before approval.",
                "context": {"scope": scope, "task_pattern": "amber-hatch"},
            }))["node"]["id"]
            replacement = "Amber hatch opening is forbidden before approval."
            if writer == "teach":
                correction = _structured(await client.call_tool("memory_teach", {
                    "trace_id": original, "correction": replacement,
                }))["corrective_trace"]["id"]
            else:
                correction = _structured(await client.call_tool("memory_remember", {
                    "content": replacement, "context": {"scope": scope},
                }))["node"]["id"]
                store.create_connection(correction, original, "supersedes", metadata={
                    "kind": "duplicate_content",
                })

            async def check(level):
                before = list(store.connection.iterdump())
                filtered = _structured(await client.call_tool("memory_lookup", {
                    "scope": scope, "task_pattern": "amber-hatch", "level": level,
                }))["results"]
                uri = "memory://recent" if level == "trace" else "memory://project/read-validity/concepts"
                resource = json.loads((await client.read_resource(uri))[0].text)
                entries = resource["traces" if level == "trace" else "concepts"]
                for selection in (filtered, entries):
                    entry = next(item for item in selection if item["id"] == original)
                    assert entry["superseded"] is (writer == "teach")
                    assert entry["superseded_by"] == ([correction] if writer == "teach" else [])
                    assert bool(entry["provenance"]["corrections"]) is (writer == "teach")
                    assert entry["decayed"] is False
                assert list(store.connection.iterdump()) == before

            await check("trace")
            store.update_node(original, level="concept")
            await check("concept")

    try:
        asyncio.run(scenario())
    finally:
        store.close()


def test_prompt_projects_validity_for_schemas_and_decision_history(tmp_path, monkeypatch):
    from living_memory.prompts import ContextSchema, DecisionHistory, format_retrieval_context
    from living_memory.scope import resolve_scope
    from living_memory.storage import MemoryStore

    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        schema = store.create_node(level="schema", content="Hatch recipe", context={"trigger": "hatch"})
        decision = store.append_trace("Hatch decision")
        correction = store.append_trace("Corrected hatch recipe")
        store.create_connection(correction.id, schema.id, "supersedes")
        store.soft_delete_node(correction.id, "historical correction")
        before = list(store.connection.iterdump())
        block = format_retrieval_context(
            [], schemas=[ContextSchema(schema, 0.5)],
            decision_history=[DecisionHistory(decision, ("Rejected approach",), 0.5)],
            task="hatch", plan=resolve_scope(), store=store,
            min_confidence=0.0, retrieval_policy="balanced",
        )
        schema_line = next(line for line in block.splitlines() if f"id={schema.id} " in line)
        decision_line = next(line for line in block.splitlines() if f"id={decision.id} " in line)
        assert f'superseded=true superseded_by=["{correction.id}"]' in schema_line
        assert "superseded=false superseded_by=[]" in decision_line
        assert list(store.connection.iterdump()) == before
