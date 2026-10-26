"""Exact knowledge reads survive optional observation failures over ordinary MCP."""

import asyncio
import logging
import sqlite3

from fastmcp import Client
from fastmcp.exceptions import ToolError
import pytest

from living_memory.resources import node_to_dict
from living_memory.server import create_mcp_server
from living_memory.storage import RECALL_CREDIT_LEDGER_TABLE
from test_lookup import _lookup_events


@pytest.fixture
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.delenv("LM_LOOKUP_CREDIT_POLICY", raising=False)
    server = create_mcp_server(tmp_path / "memory.sqlite3")
    try:
        yield server
    finally:
        server.auto_consolidation.close()
        server.memory_store.close()


def _credits(store):
    return [tuple(row) for row in store.connection.execute(
        f"SELECT recall_event_id, node_id, basis, source_id FROM {RECALL_CREDIT_LEDGER_TABLE}"
    )]


def test_query_only_preserves_views_validity_and_writable_credit(mcp, caplog):
    store = mcp.memory_store
    content = "Only publish the cobalt package when the signature matches."
    target = store.append_trace(content, {
        "scope": "project:probe", "task_pattern": "cobalt-signature",
    })
    old = store.append_trace("Publish the cobalt package.", {"scope": target.scope})
    store.connect_nodes(target.id, old.id, "supersedes")
    store.soft_delete_node(old.id, "replaced")
    other = store.append_trace("Separate scope.", {"scope": "project:other"})
    requested = [target.id, old.id, other.id, "missing-id"]
    arguments = {"node_id": target.id, "node_ids": requested, "scope": target.scope}

    async def scenario():
        async with Client(mcp) as client:
            baseline = (await client.call_tool("memory_lookup", arguments)).data
            assert baseline["results"][0]["content"] == content
            before_events = _lookup_events(store)
            assert len(before_events) == len(requested)
            recall = (await client.call_tool("memory_recall", {
                "query": "cobalt signature", "scope": target.scope,
                "depth": 0, "max_results": 1,
            })).data
            assert [entry["id"] for entry in recall["results"]] == [target.id]
            event_id = recall["recall_event_id"]
            assert store.get_recall_event(event_id).transport_session_id
            usefulness = store.get_node(target.id).usefulness_score
            assert _credits(store) == []

            store.connection.execute("PRAGMA query_only=ON")
            try:
                for selectors in ({}, {"view": "full"}, {"context_key": "task_pattern"}):
                    response = (await client.call_tool(
                        "memory_lookup", arguments | selectors,
                    )).data
                    assert response["count"] == 3
                    assert response["missing"] == ["missing-id"]
                    assert response["scope_mismatches"] == [
                        {"id": other.id, "scope": other.scope}
                    ]
                    entries = response["results"]
                    assert [entry["id"] for entry in entries] == requested[:-1]
                    assert entries[0]["content"] == content
                    assert entries[1]["decayed"] is True
                    assert entries[1]["decay_reason"] == "replaced"
                    assert entries[1]["superseded"] is True
                    assert entries[1]["superseded_by"] == [target.id]
                    if selectors.get("view") == "full":
                        assert entries[0] == node_to_dict(store.get_node(target.id)) | {
                            "superseded": False, "superseded_by": [],
                        }
                    elif "context_key" in selectors:
                        assert entries[0]["context"] == {"task_pattern": "cobalt-signature"}
                        assert entries[1]["context"] == {}
                    else:
                        assert response == baseline
                context = (await client.call_tool("memory_lookup", {
                    "scope": target.scope, "task_pattern": "cobalt-signature",
                })).data
                assert context["results"][0]["content"] == content
                assert _lookup_events(store) == before_events
                assert _credits(store) == []
                assert store.get_node(target.id).usefulness_score == usefulness
            finally:
                store.connection.execute("PRAGMA query_only=OFF")

            restored = (await client.call_tool("memory_lookup", arguments)).data
            assert restored == baseline
            events = _lookup_events(store)
            new_event_ids = {row["lookup_event_id"] for row in events} - {
                row["lookup_event_id"] for row in before_events
            }
            assert len(new_event_ids) == 1
            lookup_id = new_event_ids.pop()
            assert {row["node_id"] for row in events if row["lookup_event_id"] == lookup_id} == set(requested)
            credits = _credits(store)
            assert credits == [(event_id, target.id, "lookup", lookup_id)]
            credited_usefulness = store.get_node(target.id).usefulness_score
            assert credited_usefulness > usefulness
            await client.call_tool("memory_lookup", arguments)
            assert _credits(store) == credits
            assert store.get_node(target.id).usefulness_score == credited_usefulness
            assert len(_lookup_events(store)) == 3 * len(requested)

    with caplog.at_level(logging.WARNING, logger="living_memory.server"):
        asyncio.run(scenario())
    warnings = [r for r in caplog.records if "lookup usage recording failed" in r.message]
    assert len(warnings) == 3
    assert all("OperationalError; 4 requested id(s)" in r.message for r in warnings)
    assert all(r.exc_info is None and len(r.message) < 200 for r in warnings)


@pytest.mark.parametrize("table", ["nodes", "connections"])
def test_source_and_validity_read_errors_remain_tool_errors(mcp, table):
    store = mcp.memory_store
    node = store.append_trace("Read errors must remain errors.", {"scope": "project:probe"})
    denied = []

    def authorize(action, name, *_args):
        if action == sqlite3.SQLITE_READ and name == table:
            denied.append(name)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    async def scenario():
        async with Client(mcp) as client:
            store.connection.set_authorizer(authorize)
            try:
                with pytest.raises(ToolError, match="prohibited|authorized"):
                    await client.call_tool("memory_lookup", {"node_id": node.id})
            finally:
                store.connection.set_authorizer(None)
            assert denied == [table]
            assert _lookup_events(store) == []
            assert _credits(store) == []

    asyncio.run(scenario())


def test_unknown_recording_outcome_is_not_retried_or_credited(mcp, monkeypatch, caplog):
    store = mcp.memory_store
    node = store.append_trace("A recording outcome can be unknown.", {"scope": "project:probe"})
    record = store.record_lookup_event
    attempts = []

    def record_then_fail(*args, **kwargs):
        attempts.append(record(*args, **kwargs))
        raise RuntimeError("sensitive diagnostic " * 1000)

    monkeypatch.setattr(store, "record_lookup_event", record_then_fail)

    async def scenario():
        async with Client(mcp) as client:
            await client.call_tool("memory_recall", {
                "query": "recording outcome unknown", "scope": node.scope,
                "depth": 0, "max_results": 1,
            })
            response = (await client.call_tool("memory_lookup", {"node_id": node.id})).data
            assert response["results"][0]["content"] == node.content

    with caplog.at_level(logging.WARNING, logger="living_memory.server"):
        asyncio.run(scenario())
    assert len(attempts) == 1
    assert [row["lookup_event_id"] for row in _lookup_events(store)] == attempts
    assert _credits(store) == []
    warnings = [r for r in caplog.records if "lookup usage recording failed" in r.message]
    assert len(warnings) == 1
    assert "RuntimeError; 1 requested id(s)" in warnings[0].message
    assert "sensitive diagnostic" not in caplog.text
    assert warnings[0].exc_info is None
