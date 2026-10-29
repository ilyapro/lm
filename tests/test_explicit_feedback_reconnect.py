"""Explicit feedback follows a declared caller session across real MCP connections."""

import asyncio
import sqlite3

from fastmcp import Client
from fastmcp.exceptions import ToolError
import pytest

from living_memory import storage
from living_memory.feedback import resolve_explicit_marks
from test_current_feedback_delivery import credits, node
from test_derived_read_preparation import runtime
from test_recall_query_preparation import CONTENT, QUERY, SCOPE, call, recall
from test_tls_transport import TOKEN, _serve


async def mark(client, route, context, used, irrelevant, trace_id):
    fields = {"used": used, "irrelevant": irrelevant}
    if route == "recall":
        return await recall(client, query="following request", scope=context.get("scope"),
                            ambient_context=context, **fields)
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
            health = await call(second, "memory_health", scope=SCOPE)
            assert health["feedback_closure"]["identity_coverage"] == {
                "explicit": {"events": 1, "closed": 0, "closure_ratio": 0.0},
                "transport_only": {"events": 0, "closed": 0, "closure_ratio": None},
                "none": {"events": 0, "closed": 0, "closure_ratio": None},
            }
            with r.guard:
                pending = r.store.pending_recall_events(
                    scope=SCOPE, context={**context, "transport_session_id": "new-transport"},
                )
                assert [event.id for event in pending] == [prior["recall_event_id"]]
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


@pytest.mark.parametrize("route", ["recall", "remember", "teach"])
@pytest.mark.parametrize("old,current,accepted", [
    ({}, {}, False),
    ({"agent": "worker"}, {"agent": "worker"}, False),
    ({"task": "shared"}, {"task": "shared"}, False),
    ({"task": "shared", "agent": "worker"}, {"task": "shared", "agent": "worker"}, False),
    ({}, {"session_id": "caller"}, False),
    ({"session_id": "caller"}, {}, False),
    ({"session_id": "caller"}, {"session_id": "other"}, False),
    ({"session_id": " "}, {"session_id": " "}, False),
    ({"session_id": "caller", "task": "a"}, {"session_id": "caller", "task": "b"}, False),
    ({"session_id": "caller", "agent": "a"}, {"session_id": "caller", "agent": "b"}, False),
    ({"session_id": "caller"}, {"session_id": "caller", "scope": "project:foreign"}, False),
    ({"session": "caller"}, {"session_id": "caller"}, True),
    ({"session_id": "caller"}, {"session": "caller", "task": "a", "agent": "a"}, True),
    ({"session_id": "caller", "task": "a", "agent": "a"}, {"session_id": "caller"}, True),
])
def test_native_reconnect_requires_session_and_rejects_context_conflicts(runtime, route, old, current, accepted):
    r = runtime

    async def scenario():
        async with Client(r.mcp) as first:
            prior = await recall(first, ambient_context={"scope": SCOPE, **old})
        async with Client(r.mcp) as second:
            response = await mark(second, route, {"scope": SCOPE, **current},
                                  [r.node.id, "never-delivered"], [], r.node.id)
            assert response["feedback_marks"] == {
                "accepted": int(accepted), "dropped": 2 - int(accepted),
                "by_reason": {"not_delivered": 2 - int(accepted)},
            }
            with r.guard:
                rows = r.store.connection.execute(
                    "SELECT recall_event_id, accepted FROM recall_feedback_marks ORDER BY id"
                ).fetchall()
                assert tuple(rows[0]) == (prior["recall_event_id"] if accepted else "", int(accepted))
                assert tuple(rows[1]) == ("", 0)

    asyncio.run(scenario())


def test_native_equal_queries_and_tasks_keep_independent_sessions_isolated(runtime):
    r = runtime
    contexts = [{"scope": SCOPE, "task": "shared", "agent": "worker", "session_id": name}
                for name in ("caller-a", "caller-b")]

    async def scenario():
        async with Client(r.mcp) as a, Client(r.mcp) as b:
            prior = [await recall(client, ambient_context=context)
                     for client, context in zip((a, b), contexts, strict=True)]
        async with Client(r.mcp) as a, Client(r.mcp) as b:
            # B's event is newer; it must not consume A's marks or credit.
            for client, context, event in zip((a, b), contexts, prior, strict=True):
                response = await mark(client, "remember", context, [r.node.id], [], r.node.id)
                assert response["feedback_marks"]["accepted"] == 1
                with r.guard:
                    row = r.store.connection.execute(
                        "SELECT recall_event_id, source_id FROM recall_feedback_marks ORDER BY id DESC LIMIT 1"
                    ).fetchone()
                    assert tuple(row) == (event["recall_event_id"], response["node"]["id"])
            assert credits(r) == [(event["recall_event_id"], r.node.id) for event in prior]

    asyncio.run(scenario())


def test_native_repeated_reconnect_marks_keep_once_only_credit_and_conflict_precedence(runtime):
    r = runtime
    other = node(r, "conflict")
    context = {"scope": SCOPE, "session_id": "caller"}

    async def scenario():
        async with Client(r.mcp) as first:
            prior = await recall(first, ambient_context=context, max_results=8)
            await call(first, "memory_lookup", node_id=r.node.id)
        with r.guard:
            before = r.store.get_node(r.node.id).usefulness_score
        for _ in range(2):
            async with Client(r.mcp) as client:
                response = await mark(client, "remember", context,
                                      [r.node.id, r.node.id, other, ""], [other], r.node.id)
                assert response["feedback_marks"] == {
                    "accepted": 1, "dropped": 4,
                    "by_reason": {"duplicate": 1, "conflict": 2, "empty": 1},
                }
            assert credits(r) == [(prior["recall_event_id"], r.node.id)]
            with r.guard:
                assert r.store.get_node(r.node.id).usefulness_score == before
                assert r.store.connection.execute("SELECT count(*) FROM recall_explicit_credit").fetchone()[0] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["recall", "remember", "teach"])
def test_session_alone_uses_same_inferred_scope_on_both_sides(runtime, route):
    r = runtime
    context = {"session_id": "caller"}

    async def scenario():
        async with Client(r.mcp) as first:
            prior = await recall(first, scope=None, ambient_context=context)
        async with Client(r.mcp) as second:
            if route == "recall":
                response = await recall(second, scope=None, ambient_context=context, used=[r.node.id])
            else:
                response = await mark(second, route, context, [r.node.id], [], r.node.id)
            assert response["feedback_marks"]["accepted"] == 1
            assert credits(r) == [(prior["recall_event_id"], r.node.id)]

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["remember", "teach"])
def test_reconnect_marks_roll_back_with_failed_required_source_write(runtime, route):
    r = runtime
    context = {"scope": SCOPE, "session_id": "caller"}

    async def scenario():
        async with Client(r.mcp) as first:
            prior = await recall(first, ambient_context=context)
        with r.guard:
            before = r.store.get_node(r.node.id)
            r.store.connection.execute("""CREATE TRIGGER reject_closure
                BEFORE UPDATE OF feedback_applied_at ON recall_events
                WHEN NEW.feedback_applied_at IS NOT NULL
                BEGIN SELECT RAISE(ABORT, 'required closure failure'); END""")
        async with Client(r.mcp) as second:
            with pytest.raises(ToolError, match="required closure failure"):
                await mark(second, route, context, [r.node.id], [], r.node.id)
        with r.guard:
            assert r.store.get_node(r.node.id) == before
            assert r.store.get_recall_event(prior["recall_event_id"]).feedback_trace_id is None
            for table, count in [("nodes", 1), ("connections", 0), ("recall_feedback_marks", 0),
                                 ("recall_credit_ledger", 0), ("recall_explicit_credit", 0),
                                 ("query_anchors", 0), ("query_anchor_edges", 0)]:
                assert r.store.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == count
            assert not r.store.connection.in_transaction

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["session", "task", "agent", "scope", "results", "time"])
def test_reconnect_revalidates_delivery_after_unlocked_preparation(runtime, change):
    r = runtime
    context = {"scope": SCOPE, "session_id": "caller", "task": "task", "agent": "agent"}

    async def scenario():
        async with Client(r.mcp) as first:
            prior = await recall(first, ambient_context=context)
        r.feedback_hold = True
        async with Client(r.mcp) as second:
            pending = asyncio.create_task(recall(second, query="next", ambient_context=context, used=[r.node.id]))
            try:
                assert await asyncio.to_thread(r.entered.wait, 4)
                with r.guard, r.store.transaction():
                    column, value = {"session": ("session_id", "other"), "task": ("task", "other"),
                                     "agent": ("agent", "other"), "scope": ("scope", "project:other"),
                                     "results": ("results", "[]"), "time": ("created_at", "2000-01-01")}[change]
                    r.store.connection.execute(f"UPDATE recall_events SET {column} = ? WHERE id = ?",
                                               (value, prior["recall_event_id"]))
                    if change == "scope":
                        r.store.connection.execute("UPDATE recall_events SET requested_scope = ? WHERE id = ?",
                                                   (value, prior["recall_event_id"]))
            finally:
                r.release.set()
            response = await asyncio.wait_for(pending, 5)
            assert response["feedback_marks"]["by_reason"] == {"not_delivered": 1}
            assert credits(r) == []
            with r.guard:
                assert r.store.count_query_anchors() == 0

    asyncio.run(scenario())


def test_streamable_http_reconnect_uses_saved_session_without_server_restart(tmp_path, monkeypatch):
    for key, value in {"LM_EXPLICIT_FEEDBACK_POLICY": "credit", "LM_AUTO_CONSOLIDATE_POLICY": "off",
                       "LM_DECAY_SWEEP_INTERVAL_SEC": "0", "LM_RECALL_MAP": "0",
                       "LM_RECALL_MIN_SCORE": "off", "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic"}.items():
        monkeypatch.setenv(key, value)
    context = {"scope": SCOPE, "session_id": "caller"}
    with _serve(tmp_path, tls=False) as srv:
        async def scenario():
            async with Client(f"{srv['base_url']}/mcp/", auth=TOKEN) as first:
                written = await call(first, "memory_remember", content=CONTENT, context={"scope": SCOPE})
                target = written["node"]["id"]
                prior = await recall(first, query=CONTENT, ambient_context=context)
                assert [item["id"] for item in prior["results"]] == [target]
            async with Client(f"{srv['base_url']}/mcp/", auth=TOKEN) as second:
                response = await recall(second, query="next", ambient_context=context, used=[target])
                assert response["feedback_marks"]["accepted"] == 1
            return prior["recall_event_id"], response["recall_event_id"], target

        old, new, target = asyncio.run(scenario())
        assert srv["proc"].poll() is None
        with sqlite3.connect(srv["db_path"]) as db:
            rows = db.execute("SELECT transport_session_id, session_id FROM recall_events WHERE id IN (?, ?)",
                              (old, new)).fetchall()
            assert len(rows) == 2 and rows[0][0] != rows[1][0]
            assert {row[1] for row in rows} == {"caller"}
            assert db.execute("SELECT recall_event_id, source_id, node_id, accepted FROM recall_feedback_marks").fetchall() == [
                (old, new, target, 1),
            ]


NOW = "2026-10-27T12:00:00Z"
CONTEXT = {"scope": SCOPE, "session_id": "caller", "task": "task", "agent": "agent"}


def delivery(r, transport, ids, *, context=CONTEXT, at=NOW, scope=SCOPE):
    event = r.store.record_recall_event(
        query=QUERY, scope=scope, ambient_context={**context, "transport_session_id": transport},
        results=[{"node_id": target, "vector_score": 0.8} for target in ids],
    )
    with r.store.transaction():
        r.store.connection.execute("UPDATE recall_events SET created_at = ?, feedback_applied = 1 WHERE id = ?",
                                   (at, event.id))
    return event.id


@pytest.mark.parametrize("filler_time", [NOW, "invalid", "9999-01-01"])
def test_reconnect_uses_one_combined_cap_before_id_and_time_filters(runtime, monkeypatch, filler_time):
    r = runtime
    boundary = node(r, "boundary")
    with r.guard:
        delivery(r, "old", [r.node.id])
        last_eligible = delivery(r, "old", [boundary])
        for index in range(199):
            delivery(r, "current" if index % 2 else "old", [f"unrelated-{index}"], at=filler_time)
        for _ in range(205):
            delivery(r, "peer", [r.node.id, boundary], context={**CONTEXT, "session_id": "peer"})
        materialized, projected, statements = [], [], []
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
        r.store.connection.set_trace_callback(statements.append)
        try:
            marks = resolve_explicit_marks(r.store, [r.node.id, boundary], None, "current", NOW, context=CONTEXT)
            assert [mark.accepted for mark in marks] == [False, True]
            assert marks[1].event.id == last_eligible
            assert materialized == [last_eligible]
            assert projected == [boundary]
            query = next(sql for sql in statements if "WITH recent AS MATERIALIZED" in sql)
            plans = [row[3] for row in r.store.connection.execute("EXPLAIN QUERY PLAN " + query)]
            assert any("idx_recall_events_session_created" in plan for plan in plans)
            assert any("idx_recall_events_transport_created" in plan for plan in plans)
            assert not any(plan.startswith("SCAN recall_events") for plan in plans)
        finally:
            r.store.connection.set_trace_callback(
                lambda sql: r.unlocked_sql.append(sql) if not r.guard._is_owned() else None,
            )


def test_reconnect_preserves_time_boundaries_text_order_and_first_result_rank(runtime):
    r = runtime
    cases = [
        ("2026-10-26T12:00:00.999999Z", True),
        ("2026-10-26T11:59:59.999999Z", False),
        ("2026-10-27T12:00:00.999999Z", True),
        ("2026-10-27T12:00:01Z", False),
        ("2026-10-27T19:00:00+07:00", True),
        ("invalid", False),
    ]
    with r.guard:
        expected = {}
        for index, (at, eligible) in enumerate(cases):
            target = str(index)
            event_id = delivery(r, "old", [target], at=at)
            if eligible:
                expected[target] = (event_id, 0)
        delivery(r, "current", ["tie"])
        latest = delivery(r, "old", ["other", "tie", "tie"])
        expected["tie"] = (latest, 1)
        delivery(r, "current", ["offset"], at="2026-10-27T11:30:00Z")
        lexical = delivery(r, "old", ["offset"], at="2026-10-27T13:00:00+02:00")
        expected["offset"] = (lexical, 0)
        marks = resolve_explicit_marks(r.store, [*(str(i) for i in range(len(cases))), "tie", "offset"],
                                       None, "current", NOW, context=CONTEXT)
        assert {mark.node_id: (mark.event.id, mark.rank) for mark in marks if mark.accepted} == expected


def test_same_transport_contract_and_no_transport_rejection_are_unchanged(runtime):
    r = runtime
    with r.guard:
        event_id = delivery(r, "current", [r.node.id], scope="project:foreign",
                            context={"session_id": "other", "task": "other", "agent": "other"})
        marks = resolve_explicit_marks(r.store, [r.node.id], None, "current", NOW, context=CONTEXT)
        assert marks[0].accepted and marks[0].event.id == event_id
        statements = []
        r.store.connection.set_trace_callback(statements.append)
        try:
            for transport, at, reason in [(None, NOW, "no_transport"), ("current", "invalid", "not_delivered")]:
                marks = resolve_explicit_marks(r.store, [r.node.id], None, transport, at, context=CONTEXT)
                assert marks[0].reject_reason == reason
            assert not statements
        finally:
            r.store.connection.set_trace_callback(
                lambda sql: r.unlocked_sql.append(sql) if not r.guard._is_owned() else None,
            )
