"""Explicit feedback on a map member follows the returned version and query."""

import asyncio
import sqlite3

from fastmcp import Client
from fastmcp.exceptions import ToolError
import pytest

from living_memory import storage as storage_module
from living_memory.irrelevance import list_query_irrelevance
from living_memory.server import create_mcp_server
from test_recall_map_work import SCOPE, recall, seed


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
        "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic",
        "LM_AUTO_CONSOLIDATE_POLICY": "off",
        "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
        "LM_DELIVERY_SESSION_DEDUP": "0",
        "LM_EXPLICIT_FEEDBACK_POLICY": "credit",
    }.items():
        monkeypatch.setenv(key, value)
    clock = ["2026-01-01T12:00:00Z"]
    monkeypatch.setattr(storage_module, "_utc_now", lambda: clock[0])
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    try:
        seed(mcp.memory_store)
        yield mcp, mcp.memory_store, clock
    finally:
        mcp.auto_consolidation.close()
        mcp.memory_store.close()


async def call(client, tool, **arguments):
    return (await client.call_tool(tool, arguments)).data


def member(response):
    node_id = response["recall_map"]["clusters"][0]["medoid"]["node_id"]
    assert node_id not in {entry["id"] for entry in response["results"]}
    return node_id


def test_returned_reference_selects_query_despite_competing_map(runtime):
    mcp, store, _ = runtime

    async def scenario():
        async with Client(mcp) as client:
            first = await recall(client)
            second = await recall(client, query="cobalt backup")
            node_id = member(first)
            assert node_id in {c["medoid"]["node_id"] for c in second["recall_map"]["clusters"]}
            assert node_id not in {r["id"] for r in second["results"]}
            event = first["recall_event_id"]
            unqualified = await call(client, "memory_lookup", node_id=node_id)
            assert "feedback_ref" not in unqualified["results"][0]
            wrong = await call(client, "memory_lookup", node_id=node_id, map_event_id="unknown")
            assert wrong == unqualified
            ordinary = await call(client, "memory_remember", content="Unrelated observation.",
                                  context={"scope": SCOPE}, used=[node_id])
            assert ordinary["feedback_marks"]["by_reason"] == {"not_delivered": 1}
            response = await call(client, "memory_lookup", node_ids=[node_id, "missing"],
                                  map_event_id=event, view="full")
            entry, = response["results"]
            assert entry["content"] == store.get_node(node_id).content
            assert response["missing"] == ["missing"]
            ref = entry["feedback_ref"]
            assert store.connection.execute("SELECT count(*) FROM recall_explicit_credit").fetchone()[0] == 0
            before = store.get_node(node_id).usefulness_score
            target = await call(client, "memory_remember", content="Different synthetic observation.",
                                context={"scope": SCOPE}, irrelevant=[ref])
            assert target["feedback_marks"]["accepted"] == 1
            assert [r["node_id"] for r in list_query_irrelevance(store)] == [node_id]
            teach = await call(client, "memory_teach", trace_id=target["node"]["id"],
                               correction="Corrected observation.", context={"scope": SCOPE}, used=[ref])
            assert teach["feedback_marks"]["accepted"] == 1
            negative, = list_query_irrelevance(store)
            assert negative["marks"] == negative["cancels"] == 1
            marked = await recall(client, "amber", used=[ref])
            assert marked["feedback_marks"]["accepted"] == 1
            rows = store.connection.execute(
                "SELECT recall_event_id, node_id, rank FROM recall_feedback_marks WHERE accepted=1"
            ).fetchall()
            assert [tuple(row) for row in rows] == [(event, node_id, None)] * 3
            claims = store.connection.execute("SELECT recall_event_id, node_id FROM recall_explicit_credit")
            assert [tuple(row) for row in claims] == [(event, node_id)]
            anchor, = [a for a in store.list_query_anchors(scope=SCOPE) if a.query == first["query"]]
            edge, = store.list_query_anchor_edges(anchor_id=anchor.id)
            assert edge.target_id == node_id
            assert store.get_node(node_id).usefulness_score == before
            conflict = await recall(client, "amber", irrelevant=[node_id], used=[ref])
            assert conflict["feedback_marks"]["by_reason"] == {"conflict": 2}
    asyncio.run(scenario())


@pytest.mark.parametrize("reason", ["reconnect", "caller", "scope", "version", "expired", "cap", "forged"])
def test_reference_preserves_attribution_and_version(runtime, reason):
    mcp, store, clock = runtime
    context = {"agent": "reader", "session_id": "synthetic-session", "scope": SCOPE}

    async def scenario():
        async with Client(mcp) as first:
            response = await recall(first, ambient_context=context)
            node_id = member(response)
            args = {"node_id": node_id, "map_event_id": response["recall_event_id"]}
            async with Client(mcp) as stranger:
                foreign = await call(stranger, "memory_lookup", **args)
                assert "feedback_ref" not in foreign["results"][0]
            read = await call(first, "memory_lookup", **args)
            ref = read["results"][0]["feedback_ref"]
            if reason == "version":
                store.update_node(node_id, level="concept")
                store.update_node(node_id, content="Changed exact content.")
            elif reason == "expired":
                clock[0] = "2026-01-02T12:00:01Z"
            elif reason == "cap":
                transport = store.get_recall_event(args["map_event_id"]).transport_session_id
                for _ in range(200):
                    store.record_recall_event(query="other", scope=SCOPE,
                                              ambient_context=context | {"transport_session_id": transport})
            elif reason == "forged":
                ref = f"map-read:unobserved:{node_id}"
        current = dict(context)
        if reason == "caller":
            current["agent"] = "different"
        elif reason == "scope":
            current["scope"] = "project:other"
        async with Client(mcp) as next_client:
            result = await call(next_client, "memory_remember", context=current, content="Fresh observation.",
                                irrelevant=[ref])
            expected = {} if reason == "reconnect" else {"not_delivered": 1}
            assert result["feedback_marks"]["by_reason"] == expected
            assert result["feedback_marks"]["accepted"] == int(reason == "reconnect")
    asyncio.run(scenario())


def test_failed_lookup_or_receipt_cannot_authorize_a_mark(runtime):
    mcp, store, _ = runtime

    async def scenario():
        async with Client(mcp) as client:
            response = await recall(client)
            args = {"node_id": member(response), "map_event_id": response["recall_event_id"]}
            invalid = await call(client, "memory_lookup", **args, view="bad")
            assert invalid["results"] == []
            store.connection.set_authorizer(lambda act, table, col, *_:
                sqlite3.SQLITE_DENY if (act, table, col) == (sqlite3.SQLITE_READ, "nodes", "content")
                else sqlite3.SQLITE_OK)
            try:
                with pytest.raises(ToolError):
                    await call(client, "memory_lookup", **args)
            finally:
                store.connection.set_authorizer(None)
            store.connection.execute("""CREATE TEMP TRIGGER reject_map_read BEFORE INSERT ON recall_map_reads
                                        BEGIN SELECT RAISE(ABORT, 'unavailable'); END""")
            read = await call(client, "memory_lookup", **args)
            assert read["results"][0]["id"] == args["node_id"]
            assert "feedback_ref" not in read["results"][0]
            assert store.connection.execute("SELECT count(*) FROM recall_map_reads").fetchone()[0] == 0
            dropped = await recall(client, "amber", used=[args["node_id"]])
            assert dropped["feedback_marks"]["by_reason"] == {"not_delivered": 1}
    asyncio.run(scenario())


@pytest.mark.parametrize("policy", ["audit", "off"])
def test_observation_obeys_explicit_policy_without_map_click_credit(runtime, monkeypatch, policy):
    mcp, store, _ = runtime
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", policy)

    async def scenario():
        async with Client(mcp) as client:
            response = await recall(client)
            node_id = member(response)
            score = store.get_node(node_id).usefulness_score
            read = await call(client, "memory_lookup", map_event_id=response["recall_event_id"],
                              node_id=node_id)
            entry, = read["results"]
            if policy == "audit":
                result = await recall(client, "amber", used=[entry["feedback_ref"]])
                assert result["feedback_marks"]["accepted"] == 1
            else:
                assert "feedback_ref" not in entry
                assert not store.connection.execute("SELECT * FROM recall_map_reads").fetchall()
            assert not store.connection.execute("SELECT * FROM recall_explicit_credit").fetchall()
            assert not store.connection.execute("SELECT * FROM recall_credit_ledger").fetchall()
            assert store.get_node(node_id).usefulness_score == score
    asyncio.run(scenario())


def test_recall_rechecks_version_after_resolving_earlier_event(runtime, monkeypatch):
    mcp, store, _ = runtime

    async def scenario():
        async with Client(mcp) as client:
            shown = await recall(client)
            node_id = member(shown)
            read = await call(client, "memory_lookup", node_id=node_id, map_event_id=shown["recall_event_id"])
            record = store.record_recall

            def change_then_record(**kwargs):
                store.update_node(node_id, level="concept")
                store.update_node(node_id, content="Changed while forming the carrier.")
                return record(**kwargs)

            monkeypatch.setattr(store, "record_recall", change_then_record)
            marked = await recall(client, "amber", used=[read["results"][0]["feedback_ref"]])
            assert marked["feedback_marks"]["by_reason"] == {"not_delivered": 1}
            assert not store.connection.execute("SELECT * FROM recall_explicit_credit").fetchall()
    asyncio.run(scenario())


def test_unconfirmed_audit_write_does_not_report_or_apply_map_credit(runtime):
    mcp, store, _ = runtime

    async def scenario():
        async with Client(mcp) as client:
            shown = await recall(client)
            read = await call(client, "memory_lookup", node_id=member(shown),
                              map_event_id=shown["recall_event_id"])
            store.connection.execute("""CREATE TEMP TRIGGER refuse_mark BEFORE INSERT ON recall_feedback_marks
                                        BEGIN SELECT RAISE(ABORT, 'unavailable'); END""")
            marked = await recall(client, "amber", used=[read["results"][0]["feedback_ref"]])
            assert marked["feedback_marks"] == {
                "accepted": 0, "dropped": 1, "by_reason": {"unconfirmed": 1},
            }
            assert not store.connection.execute("SELECT * FROM recall_explicit_credit").fetchall()
    asyncio.run(scenario())
