"""Required recall preparation and consumption have one durable outcome."""

import asyncio
from contextlib import contextmanager
import sqlite3

from fastmcp import Client
from fastmcp.exceptions import ToolError
import pytest

from living_memory import server
from living_memory.retrieval import MemoryRecallService
from test_transport_identity import FakeMCP


SCOPE = "project:formed-recall"
QUERY = "cobalt release signature verification"
CONTENT = "Only publish the cobalt release when signature verification succeeds."
CONTEXT = {"scope": SCOPE, "agent": "fixture", "session_id": "fixture",
           "transport_session_id": "fixture-transport"}


@pytest.fixture(params=["direct", "native"])
def runtime(tmp_path, monkeypatch, request):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
        "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic",
        "LM_AUTO_CONSOLIDATE_POLICY": "off",
        "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
        "LM_RECALL_MAP": "0",
        "LM_RECALL_MIN_SCORE": "off",
        "LM_EXPLICIT_FEEDBACK_POLICY": "credit",
    }.items():
        monkeypatch.setenv(key, value)
    mcp = server.create_mcp_server(
        tmp_path / "memory.sqlite3",
        **({"mcp_factory": FakeMCP} if request.param == "direct" else {}),
    )
    store = mcp.memory_store
    node = store.append_trace(CONTENT, CONTEXT)
    correction = store.append_trace("A later countersignature rule.", {"scope": SCOPE})
    store.create_connection(correction.id, node.id, "supersedes")
    store.soft_delete_node(correction.id)
    # Prepare the healthy index without creating consumption facts.
    selected = MemoryRecallService(store).memory_recall(
        QUERY, scope=SCOPE, depth=0, max_results=1, log_access=False, log_event=False,
    )
    assert [result.node_id for result in selected] == [node.id]
    assert selected[0].semantic_passage is not None
    assert store.list_recall_events() == []

    def call(name="memory_recall", **arguments):
        if name == "memory_recall":
            arguments = {"query": QUERY, "scope": SCOPE, "depth": 0,
                         "max_results": 1, "ambient_context": CONTEXT, **arguments}
        if request.param == "direct":
            return mcp.tools[name](**arguments)

        async def native():
            async with Client(mcp) as client:
                return (await client.call_tool(name, arguments)).data

        return asyncio.run(native())

    try:
        yield store, node, correction, call
    finally:
        store.connection.set_authorizer(None)
        mcp.auto_consolidation.close()
        store.close()


def snapshot(store):
    # A separate connection observes durable facts, including both credit ledgers.
    with sqlite3.connect(store.db_path) as reader:
        return list(reader.iterdump())


@contextmanager
def fault(store, monkeypatch, stage):
    seen = []

    def deny(table_name, column_name):
        def authorizer(action, table, column, *rest):
            if action == sqlite3.SQLITE_READ and (table, column) == (table_name, column_name):
                seen.append((table, column))
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        return authorizer

    with monkeypatch.context() as patch:
        if stage == "source":
            store.connection.set_authorizer(deny("nodes", "content"))
        elif stage == "passage":
            store.connection.set_authorizer(deny("node_chunk_embeddings", "char_start"))
        elif stage == "correction":
            original = server.correction_references

            def correction_read(*args, **kwargs):
                store.connection.set_authorizer(deny("nodes", "scope"))
                try:
                    return original(*args, **kwargs)
                finally:
                    store.connection.set_authorizer(None)

            patch.setattr(server, "correction_references", correction_read)
        elif stage == "shape":
            original = server.shape_recall_results

            def shape(*args, **kwargs):
                original(*args, **kwargs)
                seen.append("shape")
                raise ValueError("required shape unavailable")

            patch.setattr(server, "shape_recall_results", shape)
        else:
            clause = ("BEFORE UPDATE OF access_count ON nodes" if stage == "access"
                      else "BEFORE INSERT ON recall_events")
            store.connection.execute(f"""CREATE TEMP TRIGGER reject_required {clause}
                BEGIN SELECT RAISE(ABORT, 'required consumption unavailable'); END""")
        try:
            yield seen
        finally:
            store.connection.set_authorizer(None)
            if stage in ("access", "event"):
                store.connection.execute("DROP TRIGGER reject_required")


@pytest.mark.parametrize("stage", ["source", "passage", "correction", "shape", "access", "event"])
@pytest.mark.parametrize("view", ["knowledge", "full"])
@pytest.mark.parametrize("mark", ["used", "irrelevant"])
def test_required_failure_has_no_consumption_or_later_explicit_attribution(
    runtime, monkeypatch, stage, view, mark,
):
    store, node, correction, call = runtime
    before = snapshot(store)
    with fault(store, monkeypatch, stage) as seen:
        with pytest.raises((sqlite3.DatabaseError, ValueError, ToolError)):
            call(view=view)
    assert seen or stage in ("access", "event")
    assert not store.connection.in_transaction
    assert snapshot(store) == before

    recovered = call(view=view, **{mark: [node.id]})
    assert recovered["feedback_marks"] == {
        "accepted": 0, "dropped": 1, "by_reason": {"not_delivered": 1},
    }
    entry, = recovered["results"]
    body = entry.get("node", entry)
    assert (body["id"], body["content"], body["scope"]) == (node.id, CONTENT, SCOPE)
    assert entry["superseded"] and entry["superseded_by"] == [correction.id]
    assert entry["delivery"] == "full"
    event, = store.list_recall_events()
    assert event.id == recovered["recall_event_id"] and event.result_ids == [node.id]
    assert (event.agent, event.session_id, event.transport_session_id) == (
        "fixture", "fixture", "fixture-transport",
    )
    assert store.get_node(node.id).access_count == 1
    assert len(store.recall_delivery_history("fixture-transport")) == 1
    assert store.connection.execute("SELECT count(*) FROM recall_explicit_credit").fetchone()[0] == 0
    mark, = store.connection.execute("SELECT * FROM recall_feedback_marks").fetchall()
    assert not mark["accepted"] and mark["recall_event_id"] == ""
    assert mark["source_id"] == event.id


@pytest.mark.parametrize("prior", [False, True])
def test_failed_response_is_not_implicit_feedback_but_real_prior_event_remains_eligible(
    runtime, monkeypatch, prior,
):
    store, node, correction, call = runtime
    prior_id = call()["recall_event_id"] if prior else None
    before = snapshot(store)
    with fault(store, monkeypatch, "shape"):
        with pytest.raises((ValueError, ToolError)):
            call(used=[node.id])
    assert snapshot(store) == before
    remembered = call("memory_remember", content=CONTENT, context=CONTEXT)
    assert remembered["implicit_feedback"]["recall_event_ids"] == ([prior_id] if prior else [])
    assert {row[0] for row in store.connection.execute(
        "SELECT recall_event_id FROM recall_credit_ledger"
    )} == ({prior_id} if prior else set())
    again = call("memory_remember", content=CONTENT, context=CONTEXT)
    assert again["implicit_feedback"]["recall_event_ids"] == []


@pytest.mark.parametrize("stage", ["passage", "correction", "shape"])
def test_failed_call_preserves_prior_explicit_eligibility_and_mark_source(
    runtime, monkeypatch, stage,
):
    store, node, correction, call = runtime
    prior = call()["recall_event_id"]
    before = snapshot(store)
    with fault(store, monkeypatch, stage):
        with pytest.raises((sqlite3.DatabaseError, ValueError, ToolError)):
            call(used=[node.id])
    assert snapshot(store) == before
    recovered = call(used=[node.id])
    assert recovered["feedback_marks"]["accepted"] == 1
    mark, = store.connection.execute("SELECT * FROM recall_feedback_marks").fetchall()
    assert mark["recall_event_id"] == prior and mark["source_id"] == recovered["recall_event_id"]
    claim, = store.connection.execute("SELECT * FROM recall_explicit_credit").fetchall()
    assert claim["recall_event_id"] == prior and claim["node_id"] == node.id
    assert claim["source_id"] == f"mark:{mark['id']}"


def test_optional_receipt_failure_keeps_formed_answer_eligible(runtime):
    store, node, correction, call = runtime
    store.connection.execute("""CREATE TEMP TRIGGER reject_receipt
        BEFORE UPDATE OF results ON recall_events
        BEGIN SELECT RAISE(ABORT, 'optional receipt unavailable'); END""")
    first = call()
    assert first["results"][0]["content"] == CONTENT
    assert store.recall_delivery_history("fixture-transport") == []
    store.connection.execute("DROP TRIGGER reject_receipt")
    second = call(used=[node.id])
    assert second["feedback_marks"]["accepted"] == 1
    assert second["results"][0]["delivery"] == "full"
    claim, = store.connection.execute("SELECT * FROM recall_explicit_credit").fetchall()
    assert claim["recall_event_id"] == first["recall_event_id"]


def test_optional_maintenance_failure_preserves_success(runtime, monkeypatch):
    store, node, correction, call = runtime

    def fail(*args, **kwargs):
        assert not store.connection.in_transaction
        raise sqlite3.OperationalError("optional maintenance unavailable")

    monkeypatch.setattr(MemoryRecallService, "refresh_legacy_passages", fail)
    response = call()
    assert response["results"][0]["content"] == CONTENT
    event, = store.list_recall_events()
    assert event.id == response["recall_event_id"] and event.result_ids == [node.id]
    assert store.get_node(node.id).access_count == 1
    assert call(used=[node.id])["feedback_marks"]["accepted"] == 1


def test_direct_response_boundary_starts_after_selection_and_rolls_back_projection(runtime):
    store, node, correction, call = runtime
    service = MemoryRecallService(store)
    before = snapshot(store)

    def select(limited, residual, maximum, relevance):
        assert not store.connection.in_transaction
        return limited, residual

    with pytest.raises(ValueError, match="projection unavailable"):
        with service.recall_response(
            QUERY, scope=SCOPE, depth=0, max_results=1, select_results=select,
        ) as results:
            assert store.connection.in_transaction
            assert results[0].node.access_count == 1
            assert results[0].recall_event_id == service.last_recall_event_id
            assert snapshot(store) == before
            raise ValueError("projection unavailable")
    assert not store.connection.in_transaction
    assert service.last_recall_event_id is None and snapshot(store) == before


@pytest.mark.parametrize("log_access,log_event", [(False, False), (True, False), (False, True), (True, True)])
def test_direct_service_passage_failure_preserves_logging_contract(
    runtime, monkeypatch, log_access, log_event,
):
    store, node, correction, call = runtime
    service = MemoryRecallService(store)
    arguments = dict(scope=SCOPE, depth=0, max_results=1,
                     log_access=log_access, log_event=log_event)
    before = snapshot(store)
    with fault(store, monkeypatch, "passage"):
        with pytest.raises(sqlite3.DatabaseError):
            service.memory_recall(QUERY, **arguments)
    assert service.last_recall_event_id is None
    assert snapshot(store) == before
    result, = service.memory_recall(QUERY, **arguments)
    assert result.semantic_passage is not None and result.scored_chunk_id is not None
    assert result.node.access_count == int(log_access)
    assert len(store.list_recall_events()) == int(log_event)
    assert bool(result.recall_event_id) == log_event
