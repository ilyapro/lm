"""Broad deliveries remain attributable when reconnecting callers scope new knowledge."""

import asyncio

from fastmcp import Client
import pytest

from living_memory.storage import recall_fingerprint
from test_current_feedback_delivery import credits
from test_derived_read_preparation import runtime
from test_explicit_feedback_reconnect import mark
from test_recall_query_preparation import CONTENT, QUERY, SCOPE, VECTOR, call, recall


CALLER = {"session_id": "caller", "task": "task", "agent": "agent", "project": "preferred"}
DESTINATION = "project:destination"


@pytest.mark.parametrize("route", ["remember", "teach"])
@pytest.mark.parametrize("identity", [{"session_id": "caller"}, CALLER])
def test_native_broad_read_accepts_scoped_write_after_reconnect(runtime, route, identity):
    r = runtime
    with r.guard:
        other = r.store.create_node(
            level="concept", content=CONTENT + " Check the issuer.",
            context={"scope": "project:elsewhere"}, embedding=VECTOR,
        ).id

    async def scenario():
        async with Client(r.mcp) as first:
            prior = await recall(first, scope=None, ambient_context=identity, max_results=8)
            assert {item["id"] for item in prior["results"]} == {r.node.id, other}
        async with Client(r.mcp) as second:
            # Exact lookup on the new connection still has no delivery credit.
            await call(second, "memory_lookup", node_id=r.node.id)
            assert credits(r) == []
            response = await mark(second, route, {**identity, "scope": DESTINATION},
                                  [r.node.id, "never-delivered"], [other], r.node.id)
            assert response["feedback_marks"] == {
                "accepted": 2, "dropped": 1, "by_reason": {"not_delivered": 1},
            }
            source = response["node" if route == "remember" else "corrective_trace"]
            assert source["scope"] == DESTINATION
            with r.guard:
                event = r.store.get_recall_event(prior["recall_event_id"])
                assert event.scope == event.requested_scope == "session:caller"
                assert event.resolved_scopes[-1] == "*"
                rows = r.store.connection.execute(
                    "SELECT * FROM recall_feedback_marks WHERE accepted = 1 ORDER BY id"
                ).fetchall()
                assert {row["recall_event_id"] for row in rows} == {event.id}
                assert {row["source_id"] for row in rows} == {source["id"]}
                assert all(row["transport_session_id"] != event.transport_session_id for row in rows)
                anchor = r.store.find_query_anchor(event.scope, recall_fingerprint(QUERY, event.scope))
                assert anchor is not None
                negative = r.store.connection.execute(
                    "SELECT node_id FROM query_irrelevance WHERE anchor_id = ?", (anchor.id,),
                ).fetchall()
                assert [row[0] for row in negative] == [other]
                before = r.store.get_node(r.node.id).usefulness_score
            assert credits(r) == [(event.id, r.node.id)]
        async with Client(r.mcp) as third:
            repeated = await mark(third, "remember", {**identity, "scope": SCOPE},
                                  [r.node.id], [], r.node.id)
            assert repeated["feedback_marks"]["accepted"] == 1
        assert credits(r) == [(event.id, r.node.id)]
        with r.guard:
            assert r.store.get_node(r.node.id).usefulness_score == before

    asyncio.run(scenario())
