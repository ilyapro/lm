"""Required remember writes share one outcome on disposable SQLite stores."""

import asyncio
import sqlite3

import pytest

import living_memory.server as server
from test_lookup import FakeMCP


CONTEXT = {
    "scope": "project:remember-atomicity",
    "agent": "fixture",
    "task": "cobalt migration",
    "session_id": "fixture-session",
}
SOURCE = "Cobalt routing requires validated migration before release"
CONTENT = f"Verified finding: {SOURCE}"
ALTERNATIVES = [{
    "approach": "Skip migration validation",
    "rejected_because": "Unvalidated routing can break release",
}]
FAULTS = {
    "credit": """BEFORE INSERT ON recall_credit_ledger
        BEGIN SELECT RAISE(ABORT, 'required feedback fault'); END""",
    "closure": """BEFORE UPDATE OF feedback_applied_at ON recall_events
        WHEN NEW.feedback_applied_at IS NOT NULL
        BEGIN SELECT RAISE(ABORT, 'required feedback fault'); END""",
    "provenance": """AFTER UPDATE OF provenance ON nodes
        WHEN json_array_length(NEW.provenance, '$.prior_recalls') > 0
        BEGIN SELECT RAISE(ABORT, 'required feedback fault'); END""",
}


@pytest.fixture
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", "grounded")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    monkeypatch.setenv("LM_IMPLICIT_LINK_POLICY", "all")
    monkeypatch.setattr(server, "_transport_session_id", lambda: "fixture-transport")
    # Time-based maintenance has its own outcome, outside the requested write.
    monkeypatch.setattr(server, "_decay_sweep_if_due", lambda store: None)
    instance = server.create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    yield instance
    instance.auto_consolidation.close()
    instance.memory_store.close()


def seed_recall(mcp, *, extra=False):
    source = mcp.tools["memory_remember"](SOURCE, CONTEXT)["node"]["id"]
    ids = [source]
    if extra:
        ids.append(mcp.tools["memory_remember"](
            "Cobalt routing uses amber labels for the workshop display", CONTEXT
        )["node"]["id"])
    recall = mcp.tools["memory_recall"](
        "Cobalt routing", scope=CONTEXT["scope"], ambient_context=CONTEXT,
        depth=0, max_results=len(ids),
    )
    assert {result["id"] for result in recall["results"]} == set(ids)
    return ids, recall["recall_event_id"]


def snapshot(path):
    # A second connection observes committed state, including chunks, weights,
    # anchors, marks and all provenance, rather than just counting live nodes.
    with sqlite3.connect(path) as connection:
        return list(connection.iterdump())


@pytest.mark.parametrize("stage", FAULTS)
@pytest.mark.parametrize("alternatives", [None, ALTERNATIVES])
def test_required_feedback_failure_rolls_back_and_retry_is_coherent(mcp, tmp_path, stage, alternatives):
    ids, event_id = seed_recall(mcp)
    store = mcp.memory_store
    store.connection.execute(f"CREATE TRIGGER reject_feedback {FAULTS[stage]}")
    before = snapshot(tmp_path / "memory.sqlite3")
    with pytest.raises(sqlite3.IntegrityError, match="required feedback fault"):
        mcp.tools["memory_remember"](CONTENT, CONTEXT, alternatives_considered=alternatives)
    assert not store.connection.in_transaction
    assert snapshot(tmp_path / "memory.sqlite3") == before

    store.connection.execute("DROP TRIGGER reject_feedback")
    result = mcp.tools["memory_remember"](CONTENT, CONTEXT, alternatives_considered=alternatives)
    trace = store.get_node(result["node"]["id"])
    assert store.trace_count() == 2 + len(alternatives or [])
    assert not store.get_node(ids[0]).decayed
    assert trace.source_traces == ids
    assert trace.provenance["recalled_nodes"] == ids
    assert [event["id"] for event in trace.provenance["prior_recalls"]] == [event_id]
    assert store.get_recall_event(event_id).feedback_trace_id == trace.id
    assert store.get_recall_event(event_id).feedback_applied_at
    assert result["implicit_feedback"] == {
        "recall_event_ids": [event_id], "linked_node_ids": ids, "feedback_applied": True,
    }
    assert [tuple(row) for row in store.connection.execute(
        "SELECT node_id, basis, source_id FROM recall_credit_ledger"
    )] == [(ids[0], "grounded", trace.id)]
    assert {edge.target_id for edge in store.list_connections(
        source_id=trace.id, relation_type="related"
    )} == set(ids)
    if alternatives is not None:
        rejected = store.get_node(result["rejected_alternatives"][0])
        assert rejected.context["rejected_for"] == trace.id
        assert store.list_connections(source_id=rejected.id, relation_type="contradicts")[0].target_id == trace.id


def test_required_failure_also_rolls_back_ordered_optional_marks(mcp, tmp_path):
    ids, _event_id = seed_recall(mcp, extra=True)
    store = mcp.memory_store
    store.connection.execute(f"CREATE TRIGGER reject_feedback {FAULTS['provenance']}")
    before = snapshot(tmp_path / "memory.sqlite3")
    with pytest.raises(sqlite3.IntegrityError, match="required feedback fault"):
        mcp.tools["memory_remember"](
            CONTENT, CONTEXT, alternatives_considered=ALTERNATIVES,
            used=ids[:1], irrelevant=ids[1:],
        )
    assert snapshot(tmp_path / "memory.sqlite3") == before
    assert not store.connection.in_transaction


def test_success_preserves_alternatives_marks_and_confirmation(mcp):
    ids, event_id = seed_recall(mcp, extra=True)
    store = mcp.memory_store
    result = mcp.tools["memory_remember"](
        CONTENT, CONTEXT, alternatives_considered=ALTERNATIVES,
        used=ids[:1], irrelevant=ids[1:],
    )
    trace = store.get_node(result["node"]["id"])
    assert result["node"] == {
        "id": trace.id, "level": "trace", "scope": CONTEXT["scope"],
        "created_at": trace.timestamp, "prior_recall_count": 1,
        "linked_node_count": 2, "source_trace_count": 2,
    }
    assert set(trace.provenance["recalled_nodes"]) == set(ids)
    assert result["implicit_feedback"]["linked_node_ids"] == ids[:1]
    assert store.get_recall_event(event_id).feedback_trace_id == trace.id
    assert result["feedback_marks"] == {"accepted": 2, "dropped": 0, "by_reason": {}}
    assert {tuple(row) for row in store.connection.execute(
        "SELECT node_id, mark, source_id, accepted FROM recall_feedback_marks"
    )} == {(ids[0], "used", trace.id, 1), (ids[1], "irrelevant", trace.id, 1)}
    assert store.credited_node_ids(event_id) == {ids[0]}
    assert store.connection.execute("SELECT COUNT(*) FROM recall_explicit_credit").fetchone()[0] == 1
    assert store.connection.execute("SELECT COUNT(*) FROM recall_credit_ledger").fetchone()[0] == 0
    rejected = store.get_node(result["rejected_alternatives"][0])
    assert ALTERNATIVES[0]["rejected_because"] in rejected.content
    assert rejected.context["rejected_for"] == trace.id
    assert store.list_connections(source_id=rejected.id, relation_type="contradicts")[0].target_id == trace.id


def test_optional_mark_failure_does_not_fail_required_write(mcp):
    ids, event_id = seed_recall(mcp)
    store = mcp.memory_store
    store.connection.execute("""CREATE TRIGGER reject_marks BEFORE INSERT ON recall_feedback_marks
        BEGIN SELECT RAISE(ABORT, 'optional mark fault'); END""")
    result = mcp.tools["memory_remember"](CONTENT, CONTEXT, used=ids)
    assert store.get_node(result["node"]["id"]).source_traces == ids
    assert store.get_recall_event(event_id).feedback_trace_id == result["node"]["id"]
    assert store.credited_node_ids(event_id) == set(ids)


@pytest.mark.parametrize("stage", ["edges", "instructions", "scheduling"])
def test_optional_post_write_failure_keeps_confirmation(mcp, monkeypatch, stage):
    ids, event_id = seed_recall(mcp)
    store = mcp.memory_store
    transaction_states = []

    def fail(*args, **kwargs):
        transaction_states.append(store.connection.in_transaction)
        raise RuntimeError("optional post-write fault")

    if stage == "edges":
        monkeypatch.setattr(server, "derive_edges_for_new_trace", fail)
    elif stage == "instructions":
        mcp._mcp_server = mcp
        monkeypatch.setattr(server, "_instructions_with_map", fail)
    else:
        monkeypatch.setattr(server, "_auto_consolidation_due", lambda *args: 100)
        monkeypatch.setattr(mcp.auto_consolidation, "request", fail)
    result = mcp.tools["memory_remember"](CONTENT, CONTEXT)
    assert store.get_node(result["node"]["id"]).source_traces == ids
    assert store.get_recall_event(event_id).feedback_trace_id == result["node"]["id"]
    assert result["auto_consolidation"] is None
    assert transaction_states == [False]


def test_required_write_has_one_outer_commit_before_derivation(mcp, monkeypatch):
    seed_recall(mcp)
    store = mcp.memory_store
    statements = []
    transaction_states = []
    store.connection.set_trace_callback(statements.append)

    def after_commit(*args):
        transaction_states.append(store.connection.in_transaction)
        store.connection.set_trace_callback(None)

    monkeypatch.setattr(server, "derive_edges_for_new_trace", after_commit)
    mcp.tools["memory_remember"](CONTENT, CONTEXT, alternatives_considered=ALTERNATIVES)
    depth = commits = 0
    for statement in statements:
        if statement.startswith("SAVEPOINT "):
            depth += 1
        elif statement.startswith("RELEASE "):
            depth -= 1
            commits += depth == 0
        assert not statement.startswith(("COMMIT", "END TRANSACTION"))
    assert depth == 0
    assert transaction_states == [False]
    assert commits == 1, f"required write has {commits} separate durable boundaries"


def test_teach_closure_failure_remains_atomic(mcp, tmp_path):
    ids, event_id = seed_recall(mcp)
    store = mcp.memory_store
    store.connection.execute(f"CREATE TRIGGER reject_feedback {FAULTS['closure']}")
    before = snapshot(tmp_path / "memory.sqlite3")
    with pytest.raises(sqlite3.IntegrityError, match="required feedback fault"):
        mcp.tools["memory_teach"](ids[0], CONTENT, context=CONTEXT)
    assert snapshot(tmp_path / "memory.sqlite3") == before
    store.connection.execute("DROP TRIGGER reject_feedback")
    result = mcp.tools["memory_teach"](ids[0], CONTENT, context=CONTEXT)
    assert store.trace_count() == 2
    assert store.get_recall_event(event_id).feedback_trace_id == result["corrective_trace"]["id"]


def test_fastmcp_error_has_no_durable_write_and_retry_succeeds(tmp_path, monkeypatch):
    from fastmcp import Client

    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", "grounded")
    path = tmp_path / "mcp.sqlite3"
    mcp = server.create_mcp_server(path)
    store = mcp.memory_store

    async def scenario():
        async with Client(mcp) as client:
            await client.call_tool("memory_remember", {"content": SOURCE, "context": CONTEXT})
            recall = await client.call_tool("memory_recall", {
                "query": "Cobalt routing", "scope": CONTEXT["scope"],
                "ambient_context": CONTEXT, "depth": 0, "max_results": 1,
            })
            event_id = recall.structured_content["recall_event_id"]
            store.connection.execute(f"CREATE TRIGGER reject_feedback {FAULTS['closure']}")
            before = snapshot(path)
            arguments = {
                "content": CONTENT, "context": CONTEXT,
                "alternatives_considered": ALTERNATIVES,
            }
            failed = await client.call_tool("memory_remember", arguments, raise_on_error=False)
            assert failed.is_error
            assert snapshot(path) == before
            store.connection.execute("DROP TRIGGER reject_feedback")
            result = await client.call_tool("memory_remember", arguments)
            assert not result.is_error
            assert store.trace_count() == 3
            assert store.get_recall_event(event_id).feedback_trace_id == result.structured_content["node"]["id"]
            assert len(result.structured_content["rejected_alternatives"]) == 1

    try:
        asyncio.run(scenario())
    finally:
        mcp.auto_consolidation.close()
        store.close()
