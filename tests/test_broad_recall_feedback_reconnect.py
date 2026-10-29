"""Broad deliveries remain attributable when reconnecting callers scope new knowledge."""

import asyncio
import json

from fastmcp import Client
from fastmcp.exceptions import ToolError
import pytest

from living_memory import consolidation, server
from living_memory.storage import recall_fingerprint
from test_current_feedback_delivery import credits
from test_derived_read_preparation import runtime
from test_explicit_feedback_reconnect import mark
from test_recall_query_preparation import CONTENT, QUERY, SCOPE, VECTOR, call, recall


CALLER = {"session_id": "caller", "task": "task", "agent": "agent", "project": "preferred"}
DESTINATION = "project:destination"


@pytest.mark.parametrize("route", ["remember", "teach"])
@pytest.mark.parametrize("identity", [{"session_id": "caller"}, {"session": "caller"}, CALLER])
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


@pytest.mark.parametrize("route", ["remember", "teach"])
@pytest.mark.parametrize("old,current", [
    ({}, CALLER),
    (CALLER, {}),
    ({"session_id": ""}, {"session_id": ""}),
    (CALLER, {"session_id": " "}),
    ({"task": "shared", "agent": "worker"}, {"task": "shared", "agent": "worker"}),
    (CALLER, {**CALLER, "session_id": "peer"}),
    (CALLER, {**CALLER, "task": "other"}),
    (CALLER, {**CALLER, "agent": "other"}),
])
def test_native_broad_read_does_not_replace_caller_identity(runtime, route, old, current):
    r = runtime

    async def scenario():
        async with Client(r.mcp) as first:
            prior = await recall(first, scope=None, ambient_context=old)
            assert prior["results"][0]["id"] == r.node.id
        async with Client(r.mcp) as second:
            response = await mark(second, route, {**current, "scope": DESTINATION},
                                  [r.node.id], [], r.node.id)
            assert response["feedback_marks"] == {
                "accepted": 0, "dropped": 1, "by_reason": {"not_delivered": 1},
            }
            assert credits(r) == []

    asyncio.run(scenario())


@pytest.mark.parametrize("channel", ["scope", "ambient_context.scope"])
def test_native_restricted_read_fallback_is_not_reconnect_authority(runtime, channel):
    r = runtime
    # The explicit session read includes the ambient project as a fallback.
    # That does not authorize a reconnect whose current scope is that project.
    identity = {**CALLER, "project": SCOPE}
    arguments = ({"scope": "session:caller", "ambient_context": identity}
                 if channel == "scope" else
                 {"scope": None, "ambient_context": {**identity, "scope": "session:caller"}})

    async def scenario():
        async with Client(r.mcp) as first:
            prior = await recall(first, **arguments)
            assert prior["results"][0]["id"] == r.node.id
        with r.guard:
            event = r.store.get_recall_event(prior["recall_event_id"])
            assert SCOPE in event.resolved_scopes and "*" not in event.resolved_scopes
        async with Client(r.mcp) as second:
            response = await mark(second, "remember", {**identity, "scope": SCOPE},
                                  [r.node.id], [], r.node.id)
            assert response["feedback_marks"]["by_reason"] == {"not_delivered": 1}
            assert credits(r) == []

    asyncio.run(scenario())


@pytest.mark.parametrize("channel", ["scope", "ambient_context.scope"])
def test_native_broad_feedback_does_not_widen_next_read_filter(runtime, channel):
    r = runtime
    arguments = ({"scope": DESTINATION, "ambient_context": CALLER}
                 if channel == "scope" else
                 {"scope": None, "ambient_context": {**CALLER, "scope": DESTINATION}})

    async def scenario():
        async with Client(r.mcp) as first:
            prior = await recall(first, scope=None, ambient_context=CALLER)
        async with Client(r.mcp) as second:
            response = await recall(second, **arguments, used=[r.node.id])
            assert response["feedback_marks"]["accepted"] == 1
            assert response["results"] == []
            assert credits(r) == [(prior["recall_event_id"], r.node.id)]

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["remember", "teach"])
@pytest.mark.parametrize("change", ["session_id", "task", "agent", "breadth", "results", "created_at"])
def test_native_broad_delivery_is_revalidated_after_preparation(runtime, route, change):
    r = runtime

    async def scenario():
        async with Client(r.mcp) as first:
            prior = await recall(first, scope=None, ambient_context=CALLER)
        r.feedback_hold = True
        async with Client(r.mcp) as second:
            pending = asyncio.create_task(mark(second, route, {**CALLER, "scope": DESTINATION},
                                               [r.node.id], [], r.node.id))
            try:
                assert await asyncio.to_thread(r.entered.wait, 4)
                with r.guard, r.store.transaction():
                    column, value = {
                        "breadth": ("resolved_scopes", json.dumps(["session:caller", "global"])),
                        "results": ("results", "[]"), "created_at": ("created_at", "2000-01-01"),
                    }.get(change, (change, "other"))
                    r.store.connection.execute(f"UPDATE recall_events SET {column} = ? WHERE id = ?",
                                               (value, prior["recall_event_id"]))
            finally:
                r.release.set()
            response = await asyncio.wait_for(pending, 5)
            assert response["feedback_marks"]["by_reason"] == {"not_delivered": 1}
            assert credits(r) == []
            with r.guard:
                assert r.store.count_query_anchors() == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["remember", "teach"])
def test_native_scoped_write_rolls_back_broad_feedback_on_required_failure(runtime, monkeypatch, route):
    r = runtime
    module = server if route == "remember" else consolidation
    original = module.apply_pending_recall_feedback

    def fail(store, *args, **kwargs):
        assert r.guard._is_owned() and store.connection.in_transaction
        assert store.connection.execute(
            "SELECT count(*) FROM recall_feedback_marks WHERE accepted = 1"
        ).fetchone()[0] == 1
        assert len(credits(r)) == 1
        raise RuntimeError("required feedback failure")

    async def scenario():
        async with Client(r.mcp) as first:
            prior = await recall(first, scope=None, ambient_context=CALLER)
        with r.guard:
            before = r.store.get_node(r.node.id)
            chunks = r.store.list_node_chunks(r.node.id)
        monkeypatch.setattr(module, "apply_pending_recall_feedback", fail)
        async with Client(r.mcp) as second:
            with pytest.raises(ToolError, match="required feedback failure"):
                await mark(second, route, {**CALLER, "scope": DESTINATION}, [r.node.id], [], r.node.id)
        with r.guard:
            assert r.store.get_node(r.node.id) == before
            assert r.store.list_node_chunks(r.node.id) == chunks
            assert r.store.get_recall_event(prior["recall_event_id"]).feedback_trace_id is None
            for table, count in [("nodes", 1), ("connections", 0), ("recall_feedback_marks", 0),
                                 ("recall_credit_ledger", 0), ("recall_explicit_credit", 0),
                                 ("query_anchors", 0), ("query_anchor_edges", 0)]:
                assert r.store.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == count
            assert not r.store.connection.in_transaction
        monkeypatch.setattr(module, "apply_pending_recall_feedback", original)
        async with Client(r.mcp) as third:
            response = await mark(third, route, {**CALLER, "scope": DESTINATION},
                                  [r.node.id], [], r.node.id)
            assert response["feedback_marks"]["accepted"] == 1
        assert credits(r) == [(prior["recall_event_id"], r.node.id)]

    asyncio.run(scenario())
