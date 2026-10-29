"""Native feedback selects current bounded delivery facts without session bodies."""

import asyncio
import json

from fastmcp import Client
import pytest

from living_memory import storage
from living_memory.feedback import apply_lookup_credit, resolve_explicit_marks
from test_derived_read_preparation import runtime
from test_recall_query_preparation import CONTENT, SCOPE, VECTOR, call, recall


NOW = "2026-10-27T12:00:00Z"


def delivered(r, transport, ids, *, at=NOW, query="delivery query", scope=SCOPE):
    with r.guard:
        event = r.store.record_recall_event(
            query=query, scope=scope,
            ambient_context={"transport_session_id": transport, "unrelated": "x" * 512},
            results=[{"node_id": node_id, "rank": 99, "vector_score": 0.75}
                     for node_id in ids],
        )
        with r.store.transaction():
            r.store.connection.execute(
                "UPDATE recall_events SET created_at = ?, feedback_applied = 1 WHERE id = ?",
                (at, event.id),
            )
        return event.id


def node(r, name):
    with r.guard:
        return r.store.create_node(level="concept", content=f"{CONTENT} {name}",
                                   context={"scope": SCOPE}, embedding=VECTOR).id


def credits(r):
    with r.guard:
        return [tuple(row) for row in r.store.connection.execute(
            "SELECT recall_event_id, node_id FROM recall_credit_ledger ORDER BY rowid"
        )] + [tuple(row) for row in r.store.connection.execute(
            "SELECT recall_event_id, node_id FROM recall_explicit_credit ORDER BY rowid"
        )]


async def mark_or_lookup(client, route, ids):
    if route == "lookup":
        return await call(client, "memory_lookup", node_ids=ids)
    return await recall(client, query="following request", used=ids)


@pytest.mark.parametrize("route", ["lookup", "explicit"])
@pytest.mark.parametrize("filler_time", [NOW, "9999-01-01", "invalid"])
def test_native_cap_precedes_id_and_time_filters(runtime, monkeypatch, route, filler_time):
    r = runtime
    monkeypatch.setattr(storage, "_utc_now", lambda: NOW)
    boundary, newest = node(r, "boundary"), node(r, "newest")

    async def scenario():
        async with Client(r.mcp) as client:
            prior = await recall(client)
            with r.guard:
                transport = r.store.get_recall_event(prior["recall_event_id"]).transport_session_id
            at_boundary = delivered(r, transport, [boundary])
            for index in range(198):
                # Ineligible matching events still spend the cap; unrelated
                # eligible events do too. The initial delivery is now #201.
                delivered(r, transport, [r.node.id if filler_time != NOW else f"other-{index}"],
                          at=filler_time)
            latest = delivered(r, transport, ["other", newest, newest])
            for _ in range(3):
                delivered(r, "foreign", [r.node.id, boundary, newest], at="9999-01-01")
            response = await mark_or_lookup(client, route, [boundary, newest, r.node.id, boundary])
            assert set(credits(r)) == {(at_boundary, boundary), (latest, newest)}
            if route == "lookup":
                assert [item["id"] for item in response["results"]] == [boundary, newest, r.node.id]
                assert credits(r) == [(latest, newest), (at_boundary, boundary)]
            else:
                assert response["feedback_marks"] == {
                    "accepted": 2, "dropped": 2,
                    "by_reason": {"not_delivered": 1, "duplicate": 1},
                }
                with r.guard:
                    marks = r.store.connection.execute(
                        "SELECT node_id, rank FROM recall_feedback_marks WHERE accepted = 1 ORDER BY id"
                    ).fetchall()
                    assert [tuple(row) for row in marks] == [(boundary, 0), (newest, 1)]

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["lookup", "explicit"])
def test_native_time_precision_ordering_and_duplicate_rank(runtime, monkeypatch, route):
    r = runtime
    monkeypatch.setattr(storage, "_utc_now", lambda: NOW)
    times = [
        ("2026-10-26T12:00:00.999999Z", True),
        ("2026-10-26T11:59:59.999999Z", False),
        ("2026-10-27T12:00:00.999999Z", True),
        ("2026-10-27T12:00:01Z", False),
        ("2026-10-27T19:00:00+07:00", True),
        ("2026-10-27T11:00:00", True),
        ("2026-10-27", True),
        ("invalid", False),
    ]
    ids = [node(r, str(index)) for index in range(len(times))]
    tied, offset = node(r, "tied"), node(r, "offset")

    async def scenario():
        async with Client(r.mcp) as client:
            prior = await recall(client)
            with r.guard:
                transport = r.store.get_recall_event(prior["recall_event_id"]).transport_session_id
            expected = set()
            for target, (instant, eligible) in zip(ids, times, strict=True):
                event_id = delivered(r, transport, [target], at=instant)
                if eligible:
                    expected.add((event_id, target))
            delivered(r, transport, [tied])
            latest = delivered(r, transport, ["other", tied, tied], scope="project:delivered-scope")
            expected.add((latest, tied))
            # Newest means stored text order, not reordering parsed instants.
            delivered(r, transport, [offset], at="2026-10-27T11:30:00Z")
            lexical = delivered(r, transport, [offset], at="2026-10-27T13:00:00+02:00")
            expected.add((lexical, offset))
            await mark_or_lookup(client, route, [*reversed(ids), tied, offset])
            assert set(credits(r)) == expected
            with r.guard:
                anchor = r.store.find_query_anchor(
                    "project:delivered-scope",
                    storage.recall_fingerprint("delivery query", "project:delivered-scope"),
                )
                assert anchor is not None
                if route == "explicit":
                    rank = r.store.connection.execute(
                        "SELECT rank FROM recall_feedback_marks WHERE node_id = ?", (tied,),
                    ).fetchone()[0]
                    assert rank == 1

    asyncio.run(scenario())


def test_invalid_lookup_instant_has_no_delivery_reads(runtime):
    r = runtime
    with r.guard:
        statements = []
        r.store.connection.set_trace_callback(statements.append)
        try:
            assert not apply_lookup_credit(r.store, "lookup", [r.node.id], "session", "invalid").credited
            marks = resolve_explicit_marks(r.store, [r.node.id], None, "session", "invalid")
            assert marks[0].reject_reason == "not_delivered"
            assert not statements
        finally:
            r.store.connection.set_trace_callback(
                lambda sql: r.unlocked_sql.append(sql) if not r.guard._is_owned() else None,
            )


def test_native_lookup_uses_only_delivery_facts_for_new_and_repeated_credit(runtime, monkeypatch):
    r = runtime
    monkeypatch.setattr(storage, "_utc_now", lambda: NOW)

    async def scenario():
        async with Client(r.mcp) as client:
            prior = await recall(client)
            with r.guard:
                transport = r.store.get_recall_event(prior["recall_event_id"]).transport_session_id
            for index in range(199):
                delivered(r, transport, [f"other-{index}-{j}" for j in range(5)])
            target_event = delivered(r, transport, [r.node.id, "unrelated"])
            materialized, projected = [], []
            original_event, original_json = storage._recall_event_from_row, storage._json_loads

            def event(row):
                materialized.append(row["id"])
                return original_event(row)

            def decode(value, default):
                parsed = original_json(value, default)
                if isinstance(parsed, dict) and "node_id" in parsed:
                    projected.append(parsed["node_id"])
                return parsed

            monkeypatch.setattr(storage, "_recall_event_from_row", event)
            monkeypatch.setattr(storage, "_json_loads", decode)
            first = await call(client, "memory_lookup", node_id=r.node.id)
            assert first["results"][0]["content"] == CONTENT
            assert materialized == []
            assert projected and set(projected) == {r.node.id}
            materialized.clear()
            await call(client, "memory_lookup", node_id=r.node.id)
            unknown = await call(client, "memory_lookup", node_id="unknown")
            assert unknown["missing"] == ["unknown"] and not unknown["results"]
            assert materialized == []
            assert credits(r) == [(target_event, r.node.id)]
            with r.guard:
                assert r.store.count_query_anchors() == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("replacement", [False, True])
def test_native_lookup_rechecks_current_result_body_after_encoding(runtime, replacement):
    r = runtime

    async def scenario():
        async with Client(r.mcp) as client:
            prior = await recall(client)
            r.feedback_hold = True
            operation = asyncio.create_task(call(client, "memory_lookup", node_id=r.node.id))
            try:
                assert await asyncio.to_thread(r.entered.wait, 4)
                with r.guard:
                    old = r.store.get_recall_event(prior["recall_event_id"])
                    # The normalized history still names this target; current
                    # authoritative event results no longer do.
                    with r.store.transaction():
                        r.store.connection.execute("UPDATE recall_events SET results = ? WHERE id = ?",
                                                   (json.dumps([]), old.id))
                    if replacement:
                        current = delivered(r, old.transport_session_id, ["other", r.node.id],
                                            at=old.created_at, query="current query")
            finally:
                r.release.set()
            response = await asyncio.wait_for(operation, 5)
            assert response["results"][0]["content"] == CONTENT
            assert credits(r) == ([(current, r.node.id)] if replacement else [])
            with r.guard:
                assert r.store.count_query_anchors() == int(replacement)

    asyncio.run(scenario())
