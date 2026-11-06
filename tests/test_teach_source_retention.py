"""Native correction closes recall provenance without grading its sources."""

import asyncio
import sqlite3
import weakref

from fastmcp import Client
from fastmcp.exceptions import ToolError
import pytest

from living_memory.server import create_mcp_server
from living_memory.storage import NodeProvenance, _node_read_batches


CONTEXT = {"scope": "project:teach-sources", "session_id": "caller", "task": "routing"}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "fixed")
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
    monkeypatch.setenv("LM_RECALL_MAP", "0")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    # Exercise the required native write without launching optional maintenance.
    monkeypatch.setattr(mcp.auto_consolidation, "request", lambda *args, **kwargs: None)
    try:
        yield mcp, mcp.memory_store
    finally:
        mcp.auto_consolidation.close()
        mcp.memory_store.close()


def event(store, ids, *, context=CONTEXT):
    return store.record_recall_event(
        query="routing evidence", scope=context["scope"], ambient_context=context,
        results=[{"node_id": node_id, "rank": 50 - i, "bm25_score": 1.0}
                 for i, node_id in enumerate(ids)],
    )


def teach(mcp, original):
    async def call():
        async with Client(mcp) as client:
            return (await client.call_tool("memory_teach", {
                "trace_id": original.id, "correction": "Use verified routing ownership.",
                "context": CONTEXT,
            })).data
    return asyncio.run(call())


@pytest.mark.parametrize("count", [3, 1001])
def test_native_teach_preserves_order_levels_and_closure_without_credit(runtime, count):
    mcp, store = runtime
    original = store.append_trace("Original routing claim.", CONTEXT)
    with store.transaction():
        nodes = [store.create_node(level="concept" if i == 1 else "trace",
                                   content=f"Evidence {i}.", context=CONTEXT)
                 for i in range(count)]
    a, concept, decayed = nodes[:3]
    store.soft_delete_node(decayed.id)
    ids = [node.id for node in nodes]
    older = event(store, [concept.id, "missing", a.id, a.id])
    newer = event(store, [decayed.id, *ids, "missing", decayed.id])
    incompatible = event(store, [a.id], context={**CONTEXT, "session_id": "other"})
    # Deterministic selection order independent of clock precision.
    with store.transaction():
        store.connection.execute("UPDATE recall_events SET created_at = '2026-01-01' WHERE id = ?",
                                 (older.id,))
    stats = [(node.id, store.get_node(node.id).usefulness_score) for node in nodes]
    weights = store.get_retrieval_weights(CONTEXT["scope"])
    response = teach(mcp, original)
    trace = store.get_node(response["corrective_trace"]["id"])
    assert trace.content == "Use verified routing ownership."
    expected_ids = [decayed.id, a.id, concept.id, *ids[3:]]
    assert trace.provenance["recalled_nodes"] == expected_ids
    assert trace.source_traces == [node_id for node_id in expected_ids if node_id != concept.id]
    assert [row["result_ids"] for row in trace.provenance["prior_recalls"]] == [
        [decayed.id, *ids, decayed.id], [concept.id, a.id, a.id],
    ]
    assert [row["id"] for row in trace.provenance["prior_recalls"]] == [newer.id, older.id]
    for prior in (older, newer):
        closed = store.get_recall_event(prior.id)
        assert closed.feedback_applied and closed.feedback_trace_id == trace.id
        assert closed.results == prior.results
        assert store.credited_node_ids(prior.id) == set()
    assert not store.get_recall_event(incompatible.id).feedback_applied
    assert response["implicit_feedback"] == {
        "feedback_applied": False, "recall_event_ids": [newer.id, older.id],
    }
    assert store.count_query_anchors() == 0
    assert store.list_query_anchor_edges() == []
    assert [(node.id, store.get_node(node.id).usefulness_score) for node in nodes] == stats
    assert store.get_retrieval_weights(CONTEXT["scope"]) == weights
    assert store.get_node(original.id).corrections[-1]["new"] == trace.content
    supersedes = store.list_connections(source_id=trace.id, relation_type="supersedes")
    assert len(supersedes) == 1 and supersedes[0].target_id == original.id
    assert not store.connection.in_transaction


@pytest.mark.parametrize("failure", ["content", "level", "type", "query"])
def test_native_teach_required_source_errors_roll_back_every_write(runtime, failure):
    mcp, store = runtime
    original = store.append_trace("Original routing claim.", CONTEXT)
    with store.transaction():
        nodes = [store.append_trace(f"Source {i}.", CONTEXT) for i in range(1001)]
    ids = [node.id for node in nodes]
    requested = [*ids, "missing", ids[0]]
    # Decode/conversion failures occur in the real second batch.
    broken = list(_node_read_batches(requested))[-1][-1]
    if broken == "missing":
        broken = list(_node_read_batches(requested))[-1][-2]
    store.soft_delete_node(broken)
    event(store, requested)
    if failure in {"content", "level"}:
        with store.transaction():
            store.connection.execute("PRAGMA ignore_check_constraints = ON")
            store.connection.execute(
                f"UPDATE nodes SET {failure} = CAST(x'ff' AS TEXT) WHERE id = ?", (broken,),
            )
    before = store.connection.serialize()
    row_factory = store.connection.row_factory

    class BadContent:
        def __str__(self):
            raise TypeError("source content conversion failed")

    def rows(cursor, row):
        value = sqlite3.Row(cursor, row)
        if [column[0] for column in cursor.description] == ["id", "content", "level"]:
            if value["id"] == broken:
                value = dict(value)
                value["content"] = BadContent()
        return value

    def authorize(action, table, column, database, trigger):
        if action == sqlite3.SQLITE_READ and table == "nodes" and column == "content":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    if failure == "type":
        store.connection.row_factory = rows
    if failure == "query":
        # Install after correction preparation, when the required source read begins.
        pending = store.pending_recall_events

        def deny_source_query(**kwargs):
            events = pending(**kwargs)
            store.connection.set_authorizer(authorize)
            return events

        store.pending_recall_events = deny_source_query
    try:
        with pytest.raises(ToolError, match={
            "content": "Could not decode to UTF-8", "level": "Could not decode to UTF-8",
            "type": "source content conversion failed", "query": "prohibited",
        }[failure]):
            teach(mcp, original)
    finally:
        store.connection.row_factory = row_factory
        store.connection.set_authorizer(None)
    assert store.connection.serialize() == before
    assert not store.connection.in_transaction


def test_native_teach_releases_each_source_body_before_reading_the_next(runtime, monkeypatch):
    mcp, store = runtime
    original = store.append_trace("Original routing claim.", CONTEXT)
    with store.transaction():
        ids = [store.append_trace(f"Source {i}: " + "evidence " * 500, CONTEXT).id
               for i in range(1001)]
    event(store, [*ids, ids[0], "missing"])
    live = weakref.WeakValueDictionary()
    decoded = []

    class SourceRow:
        def __init__(self, row):
            self.row = row

        def __getitem__(self, key):
            return self.row[key]

    def rows(cursor, row):
        value = sqlite3.Row(cursor, row)
        if [column[0] for column in cursor.description] == ["id", "content", "level"]:
            assert store.connection.in_transaction
            assert not live  # No fetchall or previous body retained between rows/batches.
            value = SourceRow(value)
            live[value["id"]] = value
            decoded.append(value["id"])
        return value

    reader = store.get_node_provenance
    observed = []

    def read(ids):
        facts = reader(ids)
        assert not live
        assert all(isinstance(fact, NodeProvenance) for fact in facts.values())
        observed.append(set(facts))
        return facts

    monkeypatch.setattr(store, "get_node_provenance", read)
    store.connection.row_factory = rows
    try:
        teach(mcp, original)
    finally:
        store.connection.row_factory = sqlite3.Row
    assert observed == [set(ids)]
    assert len(decoded) == len(ids) and set(decoded) == set(ids)


def test_source_facts_keep_content_conversion_and_fresh_reads(runtime):
    _, store = runtime
    source = store.append_trace("Source evidence.", CONTEXT)
    with store.transaction():
        store.connection.execute("UPDATE nodes SET content = ?, provenance = CAST(x'ff' AS TEXT) WHERE id = ?",
                                 (b"blob content", source.id))
    assert store.get_node_contents([source.id])[source.id].content == "b'blob content'"
    assert store.get_node_provenance(["missing", source.id, source.id]) == {
        source.id: NodeProvenance(source.id, "trace"),
    }
    assert store.get_node_provenance([]) == {}
    with store.transaction():
        store.connection.execute("UPDATE nodes SET level = 'concept' WHERE id = ?", (source.id,))
    assert store.get_node_provenance([source.id])[source.id].level == "concept"
