"""Exact knowledge reads project only selected sources under the read guard."""

import asyncio
import json
import threading

from fastmcp import Client
from fastmcp.exceptions import ToolError
import pytest

from living_memory import server, storage
from living_memory.delivery import knowledge_node
from living_memory.resources import node_to_dict
from test_lookup import _lookup_events


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    guard = threading.RLock()
    monkeypatch.setattr(server, "RLock", lambda: guard)
    mcp = server.create_mcp_server(tmp_path / "memory.sqlite3")
    try:
        yield mcp, guard
    finally:
        mcp.auto_consolidation.close()
        mcp.memory_store.close()


def test_native_paired_projection_work_and_full_view(runtime, monkeypatch):
    mcp, guard = runtime
    store = mcp.memory_store
    content = "Exact current knowledge remains available.\n  Исходник\t"
    small = store.append_trace(content, {"scope": "project:probe", "task": "selected-task"})
    large = store.append_trace(content, {
        **small.context, "procedure": "p" * (2 * 1024 * 1024),
    })
    store.update_node(large.id, provenance={"analysis_history": "h" * (2 * 1024 * 1024)})
    original_loads = storage._json_loads
    original_node = storage._node_from_row
    observations = []

    async def scenario():
        async with Client(mcp) as client:
            for node_id, selectors in [
                (small.id, {}), (large.id, {}),
                (large.id, {"context_key": "task"}), (large.id, {"view": "full"}),
            ]:
                reads, nodes = [], []

                def counted(value, default):
                    assert guard._is_owned()
                    reads.append(len((value or "").encode()))
                    return original_loads(value, default)

                def materialized(row):
                    nodes.append(row["id"])
                    return original_node(row)

                with monkeypatch.context() as capture:
                    capture.setattr(storage, "_json_loads", counted)
                    capture.setattr(storage, "_node_from_row", materialized)
                    response = (await client.call_tool("memory_lookup", {
                        "node_id": node_id, **selectors,
                    })).data
                assert response["results"][0]["content"] == content
                observations.append((response["results"][0], sum(reads), nodes))

    asyncio.run(scenario())
    small_result, small_bytes, small_nodes = observations[0]
    large_result, large_bytes, large_nodes = observations[1]
    selected, selected_bytes, selected_nodes = observations[2]
    full, full_bytes, full_nodes = observations[3]
    assert large_result == {**small_result, "id": large.id}
    assert selected == {**large_result, "context": {"task": "selected-task"}}
    assert full == node_to_dict(store.get_node(large.id)) | {
        "superseded": False, "superseded_by": [],
    }
    assert small_bytes == large_bytes == len("[]")
    assert selected_bytes == small_bytes + len(json.dumps("selected-task"))
    assert full_bytes > 4 * 1024 * 1024
    assert small_nodes == large_nodes == selected_nodes == []
    assert full_nodes == [large.id]
    assert len(_lookup_events(store)) == 4


@pytest.mark.parametrize("key", [
    "task", "a.b[0]", 'a".b', "a\\", 'a\\".b', "a\\u0022.b", "ключ😀", "line\n\x00",
])
def test_context_key_preserves_literal_names_and_json_values(runtime, key):
    mcp, _ = runtime
    store = mcp.memory_store
    values = [None, True, False, 0, 10**40, 1.125, "", "значение", [True, None, 10**40], {"x": False}]
    node = store.append_trace("A literal context field.", {"scope": "project:probe"})

    async def scenario():
        async with Client(mcp) as client:
            for value in values:
                store.update_node(node.id, context={"scope": node.scope, key: value})
                response = (await client.call_tool("memory_lookup", {
                    "node_id": node.id, "context_key": key,
                })).data
                selected = response["results"][0]["context"]
                assert selected == {key: value}
                assert type(selected[key]) is type(value)
            missing = (await client.call_tool("memory_lookup", {
                "node_id": node.id, "context_key": "missing",
            })).data
            assert missing["results"][0]["context"] == {}

    asyncio.run(scenario())


def test_projection_uses_stored_corrections_and_shared_shaping(runtime):
    mcp, _ = runtime
    store = mcp.memory_store
    node = store.append_trace("Original precise content.", {"scope": "project:probe"})
    corrections = [{"id": "correction", "reason": "r" * 2000, "detail": {"accepted": True}}]
    store.update_node(node.id, corrections=corrections, provenance={
        "corrections": [{"wrong": "not authoritative"}],
        "supersedes": ["earlier"], "history": "unselected",
    })
    expected = knowledge_node(node_to_dict(store.get_node(node.id))) | {
        "superseded": False, "superseded_by": [],
    }

    async def scenario():
        async with Client(mcp) as client:
            response = (await client.call_tool("memory_lookup", {"node_id": node.id})).data
            assert response["results"] == [expected]

    asyncio.run(scenario())


@pytest.mark.parametrize("column,selectors,selected", [
    ("context", {}, False), ("context", {"context_key": "task"}, True),
    ("embedding", {}, False), ("source_traces", {}, False),
    ("corrections", {}, True), ("provenance", {}, True),
])
def test_only_selected_source_errors_fail_the_read(runtime, column, selectors, selected):
    mcp, _ = runtime
    store = mcp.memory_store
    node = store.append_trace("Selected source errors stay honest.", {"scope": "project:probe"})
    # Disposable rows only. All selected source failures precede usage recording.
    store.connection.execute(f"UPDATE nodes SET {column} = ? WHERE id = ?", ("{invalid", node.id))
    store.connection.commit()

    async def scenario():
        async with Client(mcp) as client:
            if selected:
                with pytest.raises(ToolError, match="malformed JSON|Expecting"):
                    await client.call_tool("memory_lookup", {"node_id": node.id, **selectors})
                assert _lookup_events(store) == []
            else:
                response = (await client.call_tool("memory_lookup", {"node_id": node.id})).data
                assert response["results"][0]["content"] == node.content
                assert len(_lookup_events(store)) == 1
            with pytest.raises(ToolError, match="Expecting"):
                await client.call_tool("memory_lookup", {"node_id": node.id, "view": "full"})

    asyncio.run(scenario())


def test_storage_projection_batches_without_full_columns(tmp_path):
    with storage.MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = store.append_trace("Batch source.", {"scope": "project:probe"})
        ids = [f"absent-{index}" for index in range(1200)] + [node.id, node.id]
        columns = []

        def authorize(action, table, column, *_args):
            if action == storage.sqlite3.SQLITE_READ and table == "nodes":
                columns.append(column)
            return storage.sqlite3.SQLITE_OK

        store.connection.set_authorizer(authorize)
        try:
            result = store.get_node_knowledge(ids)
        finally:
            store.connection.set_authorizer(None)
        assert list(result) == [node.id]
        assert set(columns) == {
            "id", "level", "content", "scope", "decayed", "decay_reason", "corrections", "provenance",
        }
