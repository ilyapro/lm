"""Optional native feedback needs current headers, not full provenance bodies."""

import asyncio

from fastmcp import Client
import pytest

from living_memory import storage
from living_memory.irrelevance import list_query_irrelevance
from living_memory.models import RecallEvent, RecallEventHeader
from test_current_feedback_delivery import credits, node
from test_derived_read_preparation import runtime
from test_recall_query_preparation import CONTENT, SCOPE, call, recall


async def consume(client, route, used, irrelevant=(), *, trace_id=None):
    if route == "lookup":
        return await call(client, "memory_lookup", node_ids=used)
    fields = {"used": used, "irrelevant": list(irrelevant)}
    if route == "recall":
        return await recall(client, query="", **fields)
    if route == "remember":
        return await call(client, "memory_remember", content="Independent amber observation.",
                          context={"scope": SCOPE}, **fields)
    return await call(client, "memory_teach", trace_id=trace_id,
                      correction="Independent amber correction.", context={"scope": SCOPE},
                      **fields)


async def seed(r, client, ids):
    prior = await recall(client)
    with r.guard:
        r.correction_target = r.store.append_trace("Separate correction target.", {"scope": SCOPE}).id
        transport = r.store.get_recall_event(prior["recall_event_id"]).transport_session_id
        event = r.store.record_recall_event(
            query="specific delivery query", scope=SCOPE,
            ambient_context={"transport_session_id": transport,
                             "unneeded_body": "ambient-body-" + "x" * 262144},
            results=[{"node_id": target, "vector_score": 0.75} for target in ids] + [
                {"node_id": f"unrelated-{index}", "extra": "result-body-" + "y" * 4096}
                for index in range(32 - len(ids))
            ],
        )
        # Closed events remain eligible for optional feedback. No required
        # implicit consumer needs these bodies during the following calls.
        with r.store.transaction():
            r.store.connection.execute("UPDATE recall_events SET feedback_applied = 1")
        header = r.store.get_recall_event_header(event.id)
        assert type(header) is RecallEventHeader
        assert (header.id, header.query, header.scope) == (event.id, event.query, event.scope)
        full = r.store.get_recall_event(event.id)
        assert isinstance(full, RecallEvent)
        assert full.results == event.results and full.ambient_context == event.ambient_context
        return event


def remove_event(store, event_id):
    for table, key in (
        ("recall_delivery_history", "delivery_event_id"),
        ("recall_history_result_nodes", "recall_event_id"),
        ("recall_history_events", "recall_event_id"),
        ("recall_events", "id"),
    ):
        store.connection.execute(f"DELETE FROM {table} WHERE {key} = ?", (event_id,))


@pytest.mark.parametrize("route", ["lookup", "recall", "remember", "teach"])
@pytest.mark.parametrize("policy", ["credit", "audit", "off"])
def test_native_optional_feedback_reads_no_unneeded_body(runtime, monkeypatch, route, policy):
    r = runtime
    negative = node(r, "negative")

    async def scenario():
        async with Client(r.mcp) as client:
            event = await seed(r, client, [r.node.id, negative])
            monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", policy)
            monkeypatch.setenv("LM_LOOKUP_CREDIT_POLICY", "off" if policy == "off" else "delivered")
            full_reads, header_reads, projected, ambient_reads = [], [], [], []
            original_event, original_json = storage._recall_event_from_row, storage._json_loads
            original_header = r.store.get_recall_event_header

            def materialize(row):
                full_reads.append(row["id"])
                return original_event(row)

            def decode(value, default):
                if "ambient-body-" in str(value):
                    ambient_reads.append(value)
                parsed = original_json(value, default)
                if isinstance(parsed, dict) and "node_id" in parsed:
                    projected.append(parsed["node_id"])
                return parsed

            def header(event_id):
                header_reads.append(event_id)
                return original_header(event_id)

            monkeypatch.setattr(storage, "_recall_event_from_row", materialize)
            monkeypatch.setattr(storage, "_json_loads", decode)
            monkeypatch.setattr(r.store, "get_recall_event_header", header)
            numeric = policy == "credit" or (route == "lookup" and policy != "off")
            scores = []
            for attempt in range(2):
                response = await consume(client, route, [r.node.id], [negative],
                                         trace_id=r.correction_target)
                if route == "lookup":
                    assert response["results"][0]["content"] == CONTENT
                else:
                    assert response["feedback_marks"] == (
                        {"accepted": 0, "dropped": 0, "by_reason": {}, "ignored": True}
                        if policy == "off" else {"accepted": 2, "dropped": 0, "by_reason": {}}
                    )
                assert event.id not in full_reads
                assert not ambient_reads
                assert credits(r) == ([(event.id, r.node.id)] if numeric else [])
                with r.guard:
                    scores.append(r.store.get_node(r.node.id).usefulness_score)
                    assert r.store.count_query_anchors() == int(numeric)
                    marks = r.store.connection.execute(
                        "SELECT recall_event_id, rank, accepted FROM recall_feedback_marks ORDER BY id"
                    ).fetchall()
                    assert [tuple(row) for row in marks] == (
                        [(event.id, 0, 1), (event.id, 1, 1)] * (attempt + 1)
                        if route != "lookup" and policy != "off" else []
                    )
                    assert [row["node_id"] for row in list_query_irrelevance(r.store)] == (
                        [negative] if route != "lookup" and policy == "credit" else []
                    )
            assert scores[0] == scores[1]
            assert (scores[0] > 0) == numeric
            assert header_reads == ([event.id] * (1 if route == "lookup" else 2)
                                    if policy != "off" else [])
            expected = {r.node.id} if route == "lookup" else {r.node.id, negative}
            assert set(projected) == (expected if policy != "off" else set())

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["lookup", "recall", "remember", "teach"])
def test_native_feedback_rejects_event_removed_after_delivery_selection(runtime, monkeypatch, route):
    r = runtime

    async def scenario():
        async with Client(r.mcp) as client:
            event = await seed(r, client, [r.node.id])
            original = r.store.get_recall_event_header
            reads = []

            def remove_and_read(event_id):
                reads.append(event_id)
                remove_event(r.store, event_id)
                return original(event_id)

            monkeypatch.setattr(r.store, "get_recall_event_header", remove_and_read)
            response = await consume(client, route, [r.node.id], trace_id=r.correction_target)
            assert reads == [event.id]
            assert credits(r) == []
            with r.guard:
                assert r.store.get_node(r.node.id).usefulness_score == 0
                assert r.store.count_query_anchors() == 0
            if route != "lookup":
                assert response["feedback_marks"] == {
                    "accepted": 0, "dropped": 1, "by_reason": {"not_delivered": 1},
                }

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["lookup", "recall", "remember", "teach"])
@pytest.mark.parametrize("change", ["delete", "query_scope"])
def test_native_feedback_revalidates_event_after_unlocked_encoding(runtime, route, change):
    r = runtime

    async def scenario():
        async with Client(r.mcp) as client:
            event = await seed(r, client, [r.node.id])
            r.feedback_hold = True
            operation = asyncio.create_task(consume(client, route, [r.node.id],
                                                    trace_id=r.correction_target))
            try:
                assert await asyncio.to_thread(r.entered.wait, 4)
                with r.guard, r.store.transaction():
                    if change == "delete":
                        remove_event(r.store, event.id)
                    else:
                        r.store.connection.execute(
                            "UPDATE recall_events SET query = ?, scope = ? WHERE id = ?",
                            ("current query", "project:current", event.id),
                        )
            finally:
                r.release.set()
            response = await asyncio.wait_for(operation, 5)
            assert credits(r) == ([(event.id, r.node.id)] if change == "query_scope" else [])
            with r.guard:
                anchors = r.store.connection.execute("SELECT query, scope FROM query_anchors").fetchall()
                assert [tuple(row) for row in anchors] == (
                    [("current query", "project:current")] if change == "query_scope" else []
                )
            if route != "lookup":
                assert response["feedback_marks"]["accepted"] == int(change == "query_scope")

    asyncio.run(scenario())
