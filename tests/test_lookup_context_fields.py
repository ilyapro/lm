import asyncio
import json

from fastmcp import Client
import pytest

from living_memory.server import create_mcp_server


@pytest.mark.parametrize(("field", "text", "other"), [
    ("task_pattern", "alpha", {"task_pattern\0suffix": "alpha"}),
    ("procedure_id", '["alpha"]', {"procedure_id": ["alpha"]}),
    ("lesson_kind", '{"x":1}', {"lesson_kind": {"x": 1}}),
])
def test_exact_context_page(tmp_path, monkeypatch, field, text, other):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    scope = "project:lookup"
    timestamp = "2026-01-01T00:00:00Z"
    try:
        with store.transaction():
            ids = [
                store.append_trace(str(i), {
                    "scope": scope, "timestamp": timestamp, field: text,
                }).id for i in range(1001)
            ]
            # A historical object can have an extended key before the exact one.
            historical = store.append_trace("historical", {"scope": scope})
            context = {field + "\0suffix": "different", field: text, "scope": scope}
            store.connection.execute(
                "UPDATE nodes SET context = ? WHERE id = ?", (json.dumps(context), historical.id),
            )
            store.append_trace("newer false match", {"scope": scope, **other})

        async def lookup():
            async with Client(mcp) as client:
                return (await client.call_tool("memory_lookup", {"scope": scope, field: text})).data

        page = asyncio.run(lookup())
        assert page["count"] == 1000
        assert [node["id"] for node in page["results"]] == [
            historical.id, *sorted(ids, reverse=True)[:999],
        ]
    finally:
        mcp.auto_consolidation.close()
        store.close()
