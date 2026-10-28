"""Exact reads tolerate optional preparation failure without stale credit."""

import asyncio
import logging
import sqlite3

from fastmcp import Client
from fastmcp.exceptions import ToolError
import pytest

from living_memory import feedback
from living_memory.storage import MemoryStore
from test_derived_read_preparation import anchor, consume, runtime
from test_lookup import _lookup_events
from test_recall_query_preparation import CONTENT, QUERY, SCOPE, VECTOR, call, entry, recall


def assert_uncredited(r):
    assert r.store.connection.execute("SELECT count(*) FROM recall_credit_ledger").fetchone()[0] == 0
    assert r.store.count_query_anchors() == 0
    assert all(not event.feedback_applied for event in r.store.list_recall_events())


@pytest.mark.parametrize("selectors,change", [
    ({}, None), ({"view": "full"}, None), ({"context_key": "scope"}, None),
    ({}, "edit"), ({}, "delete"), ({}, "decay"), ({}, "scope"), ({}, "correction"),
])
def test_churning_lookup_reads_current_source_without_credit(runtime, monkeypatch, caplog, selectors, change):
    r = runtime
    batches, replacement = [], []

    async def scenario():
        async with Client(r.mcp) as owner:
            prior = await recall(owner)
            with r.guard:
                transport = r.store.get_recall_event(prior["recall_event_id"]).transport_session_id

            def changing_encoder(store):
                def run(texts):
                    assert not r.guard._is_owned() and not store.connection.in_transaction
                    batches.append(tuple(texts))
                    with MemoryStore(store.db_path) as writer:
                        writer.record_recall_event(
                            query=f"concurrent query {len(batches)}", scope=SCOPE,
                            ambient_context={"transport_session_id": transport},
                            results=[{"node_id": r.node.id, "vector_score": 1.0}],
                        )
                        if len(batches) == 2:
                            if change == "edit":
                                writer.update_node(r.node.id, content="Current certificate instructions.", embedding=VECTOR)
                            elif change == "delete":
                                with writer.transaction():
                                    writer.delete_node_chunks(r.node.id)
                                    writer.connection.execute("DELETE FROM nodes WHERE id = ?", (r.node.id,))
                            elif change == "decay":
                                writer.soft_delete_node(r.node.id, "expired")
                            elif change == "scope":
                                writer.update_node(r.node.id, context={"scope": "project:moved"})
                            elif change == "correction":
                                correction = writer.create_node(
                                    level="concept", content="Use the renewed cobalt certificate.",
                                    context={"scope": SCOPE}, embedding=VECTOR,
                                )
                                writer.create_connection(correction.id, r.node.id, "supersedes")
                                replacement.append(correction.id)
                    return [VECTOR for _ in texts]
                return run

            monkeypatch.setattr(feedback, "_anchor_embedder", changing_encoder)
            response = await call(owner, "memory_lookup", node_id=r.node.id,
                                  node_ids=[r.node.id, "missing-id"], scope=SCOPE, **selectors)
            assert batches == [(QUERY,), ("concurrent query 1",)]
            assert response["missing"] == ([r.node.id, "missing-id"] if change == "delete" else ["missing-id"])
            assert response["scope_mismatches"] == (
                [{"id": r.node.id, "scope": "project:moved"}] if change == "scope" else []
            )
            if change == "delete":
                assert response["count"] == 0 and response["results"] == []
            else:
                result = entry(response)
                assert result["content"] == ("Current certificate instructions." if change == "edit" else CONTENT)
                assert result["decayed"] == (change == "decay")
                if change == "decay":
                    assert result["decay_reason"] == "expired"
                assert result["superseded"] == (change == "correction")
                assert result["superseded_by"] == replacement
                if "context_key" in selectors:
                    assert result["context"] == {"scope": SCOPE}
                elif selectors.get("view") == "full":
                    assert "provenance" in result and "context" in result
            with r.guard:
                assert_uncredited(r)
                events = _lookup_events(r.store)
                assert len(events) == 2
                assert {row["node_id"] for row in events} == {r.node.id, "missing-id"}
                assert len({row["lookup_event_id"] for row in events}) == 1

    with caplog.at_level(logging.WARNING, logger="living_memory.feedback"):
        asyncio.run(scenario())
    warning, = [record for record in caplog.records if "lookup credit preparation failed" in record.message]
    assert "RuntimeError; 2 requested id(s)" in warning.message
    assert len(warning.message) < 200 and warning.exc_info is None
    assert QUERY not in caplog.text and CONTENT not in caplog.text


@pytest.mark.parametrize("fault", ["delivery", "anchor", "encoding"])
def test_optional_preparation_failure_skips_credit_and_next_call_can_credit_once(runtime, monkeypatch, caplog, fault):
    r = runtime
    target, name = {
        "delivery": (r.store, "recall_deliveries_for_transport"),
        "anchor": (r.store, "has_query_anchor"),
        "encoding": (feedback, "_encode_missing_feedback_queries"),
    }[fault]
    original = getattr(target, name)
    attempts = []

    def fail(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("sensitive query/content " * 1000)

    async def scenario():
        async with Client(r.mcp) as owner:
            prior = await recall(owner)
            monkeypatch.setattr(target, name, fail)
            response = await call(owner, "memory_lookup", node_id=r.node.id)
            assert entry(response)["content"] == CONTENT
            assert attempts == [1]
            with r.guard:
                assert_uncredited(r)
                assert len(_lookup_events(r.store)) == 1
            monkeypatch.setattr(target, name, original)
            for _ in range(2):
                assert await call(owner, "memory_lookup", node_id=r.node.id) == response
            with r.guard:
                assert len(_lookup_events(r.store)) == 3
                credit, = r.store.connection.execute("SELECT * FROM recall_credit_ledger").fetchall()
                assert credit["recall_event_id"] == prior["recall_event_id"]
                assert credit["node_id"] == r.node.id and credit["basis"] == "lookup"
                assert r.store.count_query_anchors() == 1
                current = anchor(r.store)
                assert current.reinforcement_count == 0
                edge, = r.store.list_query_anchor_edges(anchor_id=current.id)
                assert edge.hits == 1

    with caplog.at_level(logging.WARNING, logger="living_memory.feedback"):
        asyncio.run(scenario())
    warning, = [record for record in caplog.records if "lookup credit preparation failed" in record.message]
    assert len(warning.message) < 200 and warning.exc_info is None
    assert "sensitive query/content" not in caplog.text


@pytest.mark.parametrize("table", ["nodes", "connections"])
def test_source_failure_propagates_after_optional_preparation_failure(runtime, monkeypatch, table):
    r = runtime

    def fail(*args, **kwargs):
        raise RuntimeError("optional input unavailable")

    def authorize(action, name, *_args):
        return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ and name == table else sqlite3.SQLITE_OK

    async def scenario():
        async with Client(r.mcp) as owner:
            await recall(owner)
            monkeypatch.setattr(r.store, "recall_deliveries_for_transport", fail)
            with r.guard:
                r.store.connection.set_authorizer(authorize)
            try:
                with pytest.raises(ToolError, match="prohibited|authorized"):
                    await call(owner, "memory_lookup", node_id=r.node.id)
            finally:
                with r.guard:
                    r.store.connection.set_authorizer(None)
            with r.guard:
                assert_uncredited(r)
                assert _lookup_events(r.store) == []

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["recall", "explicit", "teach"])
@pytest.mark.parametrize("read", ["recall_deliveries_for_transport", "has_query_anchor"])
def test_required_preparation_read_failure_precedes_mutation(runtime, monkeypatch, route, read):
    r = runtime

    def fail(*args, **kwargs):
        raise RuntimeError("required preparation read failed")

    async def scenario():
        async with Client(r.mcp) as owner:
            await recall(owner)
            monkeypatch.setattr(r.store, read, fail)
            with pytest.raises(ToolError, match="required preparation read failed"):
                await consume(owner, route, r.node.id)
            with r.guard:
                assert_uncredited(r)
                assert len(r.store.list_recall_events()) == 1
                for table, count in [("nodes", 1), ("connections", 0), ("recall_feedback_marks", 0),
                                     ("recall_explicit_credit", 0)]:
                    assert r.store.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == count
                assert not r.store.connection.in_transaction

    asyncio.run(scenario())
