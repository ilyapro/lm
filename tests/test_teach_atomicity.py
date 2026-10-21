"""Teach, its provenance and accompanying marks have one durable outcome."""

import asyncio
import json
import sqlite3

import pytest

import living_memory.server as server
from test_lookup import FakeMCP


CONTEXT = {"scope": "project:teach-atomicity", "agent": "fixture", "session_id": "fixture"}
CONTENTS = [
    "Harbor calibration needs scheduled crane maintenance procedure",
    "Harbor calibration crane inspection includes dock sensor procedure",
]
CORRECTION = "Harbor calibration requires a fresh approved crane maintenance reading"
FAULTS = {
    "correction": f"""BEFORE INSERT ON nodes WHEN NEW.content='{CORRECTION}'
        BEGIN SELECT RAISE(ABORT, 'required teach fault'); END""",
    "provenance": """AFTER UPDATE OF provenance ON nodes
        WHEN json_array_length(NEW.provenance, '$.prior_recalls') > 0
        BEGIN SELECT RAISE(ABORT, 'required teach fault'); END""",
    "closure": """BEFORE UPDATE OF feedback_applied_at ON recall_events
        WHEN NEW.feedback_applied_at IS NOT NULL
        BEGIN SELECT RAISE(ABORT, 'required teach fault'); END""",
}


def snapshot(path):
    with sqlite3.connect(path) as connection:
        return list(connection.iterdump())


@pytest.fixture(params=["audit", "credit"])
def mcp(tmp_path, monkeypatch, request):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", request.param)
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "fixed")
    monkeypatch.setenv("LM_IMPLICIT_LINK_POLICY", "all")
    monkeypatch.setattr(server, "_transport_session_id", lambda: "fixture-transport")
    instance = server.create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    yield instance
    instance.auto_consolidation.close()
    instance.memory_store.close()


def seed(mcp):
    ids = [mcp.memory_store.append_trace(content, CONTEXT).id for content in CONTENTS]
    recall = mcp.tools["memory_recall"](
        "Harbor calibration crane procedure", scope=CONTEXT["scope"],
        ambient_context=CONTEXT, max_results=2, depth=0,
    )
    assert {item["id"] for item in recall["results"]} == set(ids)
    return ids, recall["recall_event_id"]


def teach(mcp, ids):
    return mcp.tools["memory_teach"](
        ids[0], CORRECTION, context=CONTEXT, used=ids[:1], irrelevant=ids[1:],
    )


def assert_outcome(mcp, ids, event_id, result):
    store = mcp.memory_store
    trace = store.get_node(result["corrective_trace"]["id"])
    assert store.trace_count() == 3
    assert trace.content == CORRECTION
    assert trace.context["agent"] == CONTEXT["agent"]
    assert trace.context["session_id"] == CONTEXT["session_id"]
    assert trace.context["transport_session_id"] == "fixture-transport"
    assert set(trace.provenance["recalled_nodes"]) == set(ids)
    assert [event["id"] for event in trace.provenance["prior_recalls"]] == [event_id]
    assert store.get_recall_event(event_id).feedback_trace_id == trace.id
    assert store.get_node(ids[0]).corrections[-1]["new"] == CORRECTION
    assert [(edge.source_id, edge.target_id) for edge in store.list_connections(
        relation_type="supersedes"
    )] == [(trace.id, ids[0])]
    assert {edge.target_id for edge in store.list_connections(
        source_id=trace.id, relation_type="related"
    )} == {ids[0]}
    assert result["feedback_marks"] == {"accepted": 2, "dropped": 0, "by_reason": {}}
    assert {tuple(row) for row in store.connection.execute(
        "SELECT node_id, mark, accepted, source_id, transport_session_id, agent "
        "FROM recall_feedback_marks"
    )} == {
        (ids[0], "used", 1, trace.id, "fixture-transport", CONTEXT["agent"]),
        (ids[1], "irrelevant", 1, trace.id, "fixture-transport", CONTEXT["agent"]),
    }


def test_one_commit_is_visible_before_scheduling(mcp, monkeypatch):
    ids, event_id = seed(mcp)
    store = mcp.memory_store
    statements, scheduled = [], []

    def schedule(*args, **kwargs):
        store.connection.set_trace_callback(None)
        assert not store.connection.in_transaction
        with sqlite3.connect(store.db_path) as reader:
            marks = reader.execute("SELECT source_id FROM recall_feedback_marks").fetchall()
            assert len(marks) == 2 and marks[0] == marks[1] and marks[0][0]
            assert reader.execute(
                "SELECT content FROM nodes WHERE id=?", marks[0]
            ).fetchone() == (CORRECTION,)
        scheduled.append(True)

    monkeypatch.setattr(server, "_schedule_auto_consolidation_if_due", schedule)
    store.connection.set_trace_callback(statements.append)
    result = teach(mcp, ids)
    assert_outcome(mcp, ids, event_id, result)
    assert scheduled == [True]
    depth = commits = 0
    for statement in statements:
        if statement.startswith("SAVEPOINT "):
            depth += 1
        elif statement.startswith("RELEASE "):
            depth -= 1
            commits += depth == 0
        assert not statement.startswith(("COMMIT", "END TRANSACTION"))
    assert depth == 0
    assert commits == 1


@pytest.mark.parametrize("stage", FAULTS)
def test_required_failure_rolls_back_marks_and_retry_succeeds(mcp, monkeypatch, stage):
    ids, event_id = seed(mcp)
    store = mcp.memory_store
    scheduled = []
    monkeypatch.setattr(server, "_schedule_auto_consolidation_if_due", lambda *a, **kw: scheduled.append(True))
    store.connection.execute(f"CREATE TRIGGER reject_teach {FAULTS[stage]}")
    before = snapshot(store.db_path)
    with pytest.raises(sqlite3.IntegrityError, match="required teach fault"):
        teach(mcp, ids)
    assert not store.connection.in_transaction
    assert scheduled == []
    assert snapshot(store.db_path) == before
    store.connection.execute("DROP TRIGGER reject_teach")
    assert_outcome(mcp, ids, event_id, teach(mcp, ids))
    assert scheduled == [True]


@pytest.mark.parametrize("stage", ["marks", "credit", "source"])
def test_optional_feedback_failure_keeps_correction(mcp, stage):
    ids, event_id = seed(mcp)
    store = mcp.memory_store
    faults = {
        "marks": "BEFORE INSERT ON recall_feedback_marks",
        "credit": "BEFORE INSERT ON recall_explicit_credit",
        "source": "BEFORE UPDATE OF source_id ON recall_feedback_marks WHEN NEW.mark='irrelevant'",
    }
    store.connection.execute(f"""CREATE TRIGGER reject_optional {faults[stage]}
        BEGIN SELECT RAISE(ABORT, 'optional teach fault'); END""")
    result = teach(mcp, ids)
    trace_id = result["corrective_trace"]["id"]
    assert not store.connection.in_transaction
    with sqlite3.connect(store.db_path) as reader:
        assert reader.execute("SELECT content FROM nodes WHERE id=?", (trace_id,)).fetchone() == (CORRECTION,)
        assert reader.execute("SELECT feedback_trace_id FROM recall_events WHERE id=?", (event_id,)).fetchone() == (trace_id,)
        assert reader.execute("SELECT source_id, target_id FROM connections WHERE type='supersedes'").fetchall() == [(trace_id, ids[0])]
        marks = reader.execute("SELECT source_id FROM recall_feedback_marks").fetchall()
    assert marks == ([] if stage == "marks" else [("",), ("",)] if stage == "source" else [(trace_id,), (trace_id,)])


@pytest.mark.parametrize("invalid", ["missing", "empty", "confidence"])
def test_invalid_arguments_leave_no_marks(mcp, invalid):
    ids, _ = seed(mcp)
    store = mcp.memory_store
    before = snapshot(store.db_path)
    with pytest.raises((KeyError, ValueError)):
        mcp.tools["memory_teach"](
            "missing" if invalid == "missing" else ids[0],
            " " if invalid == "empty" else CORRECTION,
            confidence="invalid" if invalid == "confidence" else None,
            context=CONTEXT, used=ids[:1], irrelevant=ids[1:],
        )
    assert snapshot(store.db_path) == before


def test_http_failure_retry_and_transport_attribution(tmp_path, monkeypatch):
    from fastmcp import Client
    from test_tls_transport import TOKEN, _serve, _structured

    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "fixed")
    with _serve(tmp_path, tls=False) as srv:
        path = srv["db_path"]

        async def scenario():
            async with Client(f"{srv['base_url']}/mcp/", auth=TOKEN) as client:
                ids = []
                for content in CONTENTS:
                    result = _structured(await client.call_tool("memory_remember", {"content": content, "context": CONTEXT}))
                    ids.append(result["node"]["id"])
                recall = _structured(await client.call_tool("memory_recall", {
                    "query": "Harbor calibration crane procedure", "scope": CONTEXT["scope"],
                    "ambient_context": CONTEXT, "max_results": 2, "depth": 0,
                }))
                assert {item["id"] for item in recall["results"]} == set(ids)
                with sqlite3.connect(path) as connection:
                    connection.execute(f"CREATE TRIGGER reject_teach {FAULTS['closure']}")
                before = snapshot(path)
                arguments = {"trace_id": ids[0], "correction": CORRECTION, "context": CONTEXT,
                             "used": ids[:1], "irrelevant": ids[1:]}
                failed = await client.call_tool("memory_teach", arguments, raise_on_error=False)
                assert failed.is_error
                assert snapshot(path) == before
                with sqlite3.connect(path) as connection:
                    connection.execute("DROP TRIGGER reject_teach")
                result = _structured(await client.call_tool("memory_teach", arguments))
                assert result["feedback_marks"]["accepted"] == 2
                return ids, recall["recall_event_id"], result["corrective_trace"]["id"]

        ids, event_id, trace_id = asyncio.run(scenario())
        with sqlite3.connect(path) as connection:
            stamp, closed_by = connection.execute(
                "SELECT transport_session_id, feedback_trace_id FROM recall_events WHERE id=?", (event_id,)
            ).fetchone()
            assert stamp and closed_by == trace_id
            assert connection.execute(
                "SELECT source_id, transport_session_id FROM recall_feedback_marks"
            ).fetchall() == [(trace_id, stamp), (trace_id, stamp)]
            context = json.loads(connection.execute("SELECT context FROM nodes WHERE id=?", (trace_id,)).fetchone()[0])
            assert context["transport_session_id"] == stamp
            assert context["agent"] == CONTEXT["agent"]
            assert connection.execute("SELECT source_id, target_id FROM connections WHERE type='supersedes'").fetchall() == [(trace_id, ids[0])]
