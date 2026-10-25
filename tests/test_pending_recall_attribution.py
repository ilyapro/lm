"""Current-call provenance through ordinary MCP tools on disposable stores."""

import asyncio

import pytest
from fastmcp import Client

from living_memory.server import create_mcp_server


SCOPE = "project:attribution"
TEXT = "cobalt routing uses stable endpoint ownership"


@pytest.fixture
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", "grounded")
    instance = create_mcp_server(tmp_path / "memory.sqlite3")
    yield instance
    instance.auto_consolidation.close()
    instance.memory_store.close()


async def call(client, tool, **arguments):
    result = await client.call_tool(tool, arguments)
    assert not result.is_error
    return result.data


@pytest.mark.parametrize("tool", ["memory_remember", "memory_teach"])
@pytest.mark.parametrize("current_recall", [False, True])
def test_only_current_delivery_contributes_provenance_credit_and_anchors(mcp, tool, current_recall):
    store = mcp.memory_store
    original_context = {
        "scope": SCOPE, "agent": "shared", "task": "source-task",
        "session_id": "source-session", "session": "source-session",
        "transport_session_id": "source-connection", "step_order": 2,
        "task_pattern": "endpoint-ownership", "step_description": "outdated step",
    }
    original = store.append_trace(TEXT, original_context)

    async def scenario():
        async with Client(mcp) as client:
            own = None
            if current_recall:
                recalled = await call(client, "memory_recall", query=TEXT, scope=SCOPE,
                                      ambient_context={"agent": "shared"}, max_results=1)
                assert [result["id"] for result in recalled["results"]] == [original.id]
                own = recalled["recall_event_id"]

            # Same words and agent, but no evidence of delivery to this call.
            strangers = [store.record_recall_event(
                query=TEXT, scope=SCOPE, ambient_context=context,
                results=[{"node_id": original.id, "bm25_score": 1.0}],
            ) for context in (
                {}, {"agent": "shared"}, original_context,
                {"agent": "shared", "transport_session_id": "foreign-connection"},
            )]
            store.connection.execute(
                "UPDATE recall_events SET created_at = '2000-01-01T00:00:00Z' WHERE id = ?",
                (strangers[0].id,),
            )
            before = [store.get_recall_event(event.id) for event in strangers]
            arguments = {"context": {"scope": SCOPE, "agent": "shared"}}
            if tool == "memory_remember":
                arguments["content"] = TEXT + " after verification"
            else:
                arguments.update(trace_id=original.id, correction=TEXT + " after verification")
            result = await call(client, tool, **arguments)
            return own, before, result

    own, strangers, result = asyncio.run(scenario())
    trace_id = result["node" if tool == "memory_remember" else "corrective_trace"]["id"]
    trace = store.get_node(trace_id)
    expected = [own] if own else []
    assert result["implicit_feedback"]["recall_event_ids"] == expected
    assert [item["id"] for item in trace.provenance.get("prior_recalls", [])] == expected
    assert trace.provenance.get("recalled_nodes", []) == ([original.id] if own else [])
    assert trace.source_traces == ([original.id] if own else [])
    assert all(store.get_recall_event(event.id) == event for event in strangers)
    assert all(not store.credited_node_ids(event.id) for event in strangers)
    if own:
        assert store.get_recall_event(own).feedback_trace_id == trace_id
    credited = bool(own and tool == "memory_remember")
    assert store.connection.execute("SELECT COUNT(*) FROM recall_credit_ledger").fetchone()[0] == int(credited)
    anchors = store.list_query_anchors(scope=SCOPE)
    assert len(anchors) == int(credited)
    if credited:
        assert store.credited_node_ids(own) == {original.id}
        assert anchors[0].reinforcement_count == 1
        assert [edge.target_id for edge in store.list_query_anchor_edges(anchor_id=anchors[0].id)] == [original.id]
    if tool == "memory_teach":
        assert not any(trace.context.get(key) for key in ("session", "session_id", "task"))
        assert trace.context["transport_session_id"] != "source-connection"
        assert trace.context["task_pattern"] == "endpoint-ownership"
        assert trace.context["step_order"] == 2
        assert "step_description" not in trace.context
        assert store.get_node(original.id).usefulness_score <= -0.25
        assert result["supersedes"]["type"] == "supersedes"
    elif not own:
        assert store.get_node(original.id).usefulness_score == original.usefulness_score


@pytest.mark.parametrize("identity", [{"task": "current-work"}, {"session_id": "current-session"}])
def test_explicit_work_continues_after_real_client_reconnect(mcp, identity):
    store = mcp.memory_store
    original = store.append_trace(TEXT, {"scope": SCOPE})

    async def scenario():
        async with Client(mcp) as client:
            recalled = await call(client, "memory_recall", query=TEXT, scope=SCOPE,
                                  ambient_context={**identity, "agent": "first"})
        async with Client(mcp) as client:
            result = await call(client, "memory_remember", content=TEXT,
                                context={"scope": SCOPE, **identity, "agent": "second"})
        return recalled, result

    recalled, result = asyncio.run(scenario())
    event = store.get_recall_event(recalled["recall_event_id"])
    trace = store.get_node(result["node"]["id"])
    assert event.transport_session_id != trace.context["transport_session_id"]
    assert event.feedback_trace_id == trace.id
    assert trace.provenance["recalled_nodes"] == [original.id]
    assert store.credited_node_ids(event.id) == {original.id}


@pytest.mark.parametrize("tool", ["memory_remember", "memory_teach"])
def test_explicit_irrelevance_survives_without_implicit_work_match(mcp, tool):
    store = mcp.memory_store
    original = store.append_trace(TEXT, {"scope": SCOPE})

    async def scenario():
        async with Client(mcp) as client:
            recalled = await call(client, "memory_recall", query=TEXT, scope=SCOPE,
                                  ambient_context={"task": "recall-task"})
            arguments = {"context": {"scope": SCOPE, "task": "different-task"},
                         "irrelevant": [original.id]}
            if tool == "memory_remember":
                arguments["content"] = "verified an independent endpoint convention"
            else:
                arguments.update(trace_id=original.id, correction="endpoint ownership is assigned per cluster")
            result = await call(client, tool, **arguments)
        return recalled, result

    recalled, result = asyncio.run(scenario())
    event = store.get_recall_event(recalled["recall_event_id"])
    assert not event.feedback_applied
    assert result["implicit_feedback"]["recall_event_ids"] == []
    assert result["feedback_marks"] == {"accepted": 1, "dropped": 0, "by_reason": {}}
    assert store.irrelevant_marked_node_ids(event.id) == {original.id}
    assert not store.credited_node_ids(event.id)
    assert store.get_node(original.id).usefulness_score < original.usefulness_score
