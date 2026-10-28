"""Explicit feedback follows a declared caller session across real MCP connections."""

import asyncio

from fastmcp import Client
import pytest

from test_current_feedback_delivery import credits, node
from test_derived_read_preparation import runtime
from test_recall_query_preparation import CONTENT, SCOPE, call, recall


async def mark(client, route, context, used, irrelevant, trace_id):
    fields = {"used": used, "irrelevant": irrelevant}
    if route == "recall":
        return await recall(client, query="following request", ambient_context=context, **fields)
    if route == "remember":
        return await call(client, "memory_remember", content="New independent observation.",
                          context=context, **fields)
    return await call(client, "memory_teach", trace_id=trace_id,
                      correction=CONTENT + " Check the issuer as well.", context=context, **fields)


@pytest.mark.parametrize("route", ["recall", "remember", "teach"])
def test_native_marks_keep_original_delivery_after_connection_closes(runtime, route):
    r = runtime
    other = node(r, "other")
    # Session is the only declared caller identity; task/agent are optional.
    context = {"scope": SCOPE, "session_id": "caller-a"}

    async def scenario():
        async with Client(r.mcp) as first:
            prior = await recall(first, ambient_context=context, max_results=8)
            assert {item["id"] for item in prior["results"]} == {r.node.id, other}
        # Closing the first context ends its MCP session. A fresh client must
        # derive a different transport, without an override or a server restart.
        async with Client(r.mcp) as second:
            lookup = await call(second, "memory_lookup", node_id=r.node.id)
            assert lookup["results"][0]["content"] == CONTENT
            assert credits(r) == []  # lookup remains transport-only
            response = await mark(second, route, context, [r.node.id], [other], r.node.id)
            with r.guard:
                event = r.store.get_recall_event(prior["recall_event_id"])
                rows = r.store.connection.execute(
                    "SELECT * FROM recall_feedback_marks ORDER BY id"
                ).fetchall()
                assert len(rows) == 2
                assert all(row["transport_session_id"] != event.transport_session_id for row in rows)
            assert response["feedback_marks"] == {"accepted": 2, "dropped": 0, "by_reason": {}}
            assert {row["recall_event_id"] for row in rows} == {event.id}
            assert {row["mark"] for row in rows} == {"used", "irrelevant"}
            source = (response["recall_event_id"] if route == "recall" else
                      response["node"]["id"] if route == "remember" else
                      response["corrective_trace"]["id"])
            assert {row["source_id"] for row in rows} == {source}
            assert credits(r) == [(event.id, r.node.id)]
            with r.guard:
                negative = r.store.connection.execute(
                    "SELECT node_id FROM query_irrelevance"
                ).fetchall()
                assert [row[0] for row in negative] == [other]

    asyncio.run(scenario())
