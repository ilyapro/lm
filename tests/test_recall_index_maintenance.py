"""Ordinary recall keeps current evidence when bounded chunk indexing fails."""

import asyncio
from contextlib import contextmanager
import logging
import sqlite3

from fastmcp import Client
from fastmcp.exceptions import ToolError
import pytest

from living_memory import retrieval
from living_memory.retrieval import LAZY_CHUNK_NODE_LIMIT, MemoryRecallService
from living_memory.server import create_mcp_server


SCOPE = "project:maintenance"
QUERY = "cobalt release signature verification"
CONTENT = "Only publish the cobalt release when signature verification succeeds."
ARGUMENTS = {"query": QUERY, "scope": SCOPE, "depth": 0, "max_results": 1}


@pytest.fixture
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
    monkeypatch.setenv("LM_RECALL_MAP", "0")
    server = create_mcp_server(tmp_path / "memory.sqlite3")
    try:
        yield server
    finally:
        server.auto_consolidation.close()
        server.memory_store.close()


async def recall(client, **arguments):
    return (await client.call_tool("memory_recall", ARGUMENTS | arguments)).data


def body(response):
    entry, = response["results"]
    return entry.get("node", entry)


def deny_index(store, node, *, ordinal=0):
    """Fail a real chunk insert, observing attempts outside SQLite rollback."""
    attempts = []
    store.connection.create_function(
        "observe_chunk_attempt", 1, lambda index: attempts.append(index) or 0,
    )
    with store.transaction():
        store.connection.execute("CREATE TABLE denied_index (id TEXT PRIMARY KEY)")
        store.connection.execute("INSERT INTO denied_index VALUES (?)", (node.id,))
        store.connection.execute(f"""
            CREATE TRIGGER refuse_index BEFORE INSERT ON node_chunk_embeddings
            WHEN NEW.node_id IN (SELECT id FROM denied_index)
            BEGIN
                SELECT observe_chunk_attempt(NEW.chunk_index);
                SELECT CASE WHEN NEW.chunk_index = {ordinal}
                    THEN RAISE(ABORT, 'private diagnostic must not appear') END;
            END
        """)
    return attempts


def event_for(store, response):
    event = store.get_recall_event(response["recall_event_id"])
    assert event.transport_session_id
    assert event.requested_scope == SCOPE
    return event


@pytest.mark.parametrize("view", ["knowledge", "full"])
@pytest.mark.parametrize("feedback_policy", ["audit", "credit"])
@pytest.mark.parametrize("fault", ["index", "hydration"])
def test_unrelated_index_failure_preserves_delivery_and_feedback(
    mcp, monkeypatch, caplog, view, feedback_policy, fault,
):
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", feedback_policy)
    store = mcp.memory_store
    target = store.append_trace(CONTENT, {"scope": SCOPE})
    old = store.append_trace("Publish the cobalt release unconditionally.", {"scope": SCOPE})
    store.connect_nodes(target.id, old.id, "supersedes")
    store.soft_delete_node(old.id, "replaced")
    foreign = store.append_trace(CONTENT, {"scope": "project:excluded"})
    blocker = store.create_node(
        level="concept", content="Violet botanical inventory.",
        context={"scope": SCOPE, "timestamp": "2026-02-02T00:00:00Z"},
    )

    async def scenario():
        async with Client(mcp) as client:
            baseline = await recall(client, view=view)
            assert body(baseline)["id"] == target.id
            assert body(baseline)["content"] == CONTENT
            assert store.node_chunks_are_current(target.id)
            assert store.node_chunks_are_current(blocker.id)
            initial_score = event_for(store, baseline).results[0]["vector_score"]
            assert initial_score > 0
            store.update_node(blocker.id, content="Violet botanical inventory. " * 160)
            provenance = store.connection.execute(
                "SELECT provenance FROM nodes WHERE id = ?", (blocker.id,),
            ).fetchone()[0]
            if fault == "hydration":
                with store.transaction():
                    store.connection.execute("UPDATE nodes SET provenance = '[]' WHERE id = ?", (blocker.id,))
            before_blocker = tuple(store.connection.execute(
                "SELECT * FROM nodes WHERE id = ?", (blocker.id,),
            ).fetchone())
            earlier = store.append_trace("Amber mineral inventory.", {
                "scope": "global", "timestamp": "2026-02-01T00:00:00Z",
            })
            later = store.append_trace("Indigo floral inventory.", {
                "scope": SCOPE, "timestamp": "2026-02-03T00:00:00Z",
            })
            attempts = deny_index(store, blocker, ordinal=1) if fault == "index" else []
            access = store.get_node(target.id).access_count
            events = len(store.list_recall_events())
            usefulness = store.get_node(target.id).usefulness_score
            response = await recall(client, view=view, used=[target.id])
            assert response["count"] == 1
            assert body(response)["id"] == target.id
            assert body(response)["content"] == CONTENT
            assert body(response)["scope"] == SCOPE
            assert body(response).get("decayed", False) is False
            assert response["results"][0]["superseded"] is False
            assert response["results"][0]["superseded_by"] == []
            assert response["results"][0]["delivery"] == "session_duplicate"
            assert response["results"][0]["content_ref"] == {
                "node_id": target.id, "full_content_chars": len(CONTENT),
            }
            event = event_for(store, response)
            assert event.result_ids == [target.id]
            assert event.results[0]["vector_score"] == initial_score
            assert store.get_node(target.id).access_count == access + 1
            assert len(store.list_recall_events()) == events + 1
            assert response["feedback_marks"] == {"accepted": 1, "dropped": 0, "by_reason": {}}
            mark, = store.connection.execute("SELECT * FROM recall_feedback_marks").fetchall()
            assert mark["node_id"] == target.id and mark["mark"] == "used"
            assert mark["recall_event_id"] == baseline["recall_event_id"]
            assert mark["source_id"] == response["recall_event_id"]
            assert mark["transport_session_id"] == event.transport_session_id
            credits = store.connection.execute("SELECT * FROM recall_explicit_credit").fetchall()
            assert len(credits) == (1 if feedback_policy == "credit" else 0)
            assert (store.get_node(target.id).usefulness_score > usefulness) == (feedback_policy == "credit")
            # Both neighbors commit while the failing source stays unchanged.
            assert attempts == ([0, 1] if fault == "index" else [])
            assert store.node_chunks_are_current(earlier.id)
            assert store.count_node_chunks(blocker.id) == 0
            assert tuple(store.connection.execute(
                "SELECT * FROM nodes WHERE id = ?", (blocker.id,),
            ).fetchone()) == before_blocker
            assert store.node_chunks_are_current(later.id)
            assert not store.node_chunks_are_current(foreign.id)
            assert not store.connection.in_transaction
            if fault == "hydration":
                with pytest.raises(TypeError):
                    store.list_unchunked_nodes(scope=SCOPE)
                with pytest.raises(TypeError):
                    store.get_node(blocker.id)
                with store.transaction():
                    store.connection.execute("UPDATE nodes SET provenance = ? WHERE id = ?", (provenance, blocker.id))
            else:
                assert [n.id for n in store.list_unchunked_nodes(scope=SCOPE)] == [blocker.id]
                store.connection.execute("DROP TRIGGER refuse_index")
            recovered = await recall(client, view=view)
            assert body(recovered)["id"] == target.id
            assert body(recovered)["content"] == CONTENT
            assert store.node_chunks_are_current(blocker.id)
            assert store.node_chunks_are_current(later.id)
            assert not store.node_chunks_are_current(foreign.id)
            assert not store.connection.in_transaction
            assert store.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    with caplog.at_level(logging.WARNING, logger="living_memory.retrieval"):
        asyncio.run(scenario())
    warnings = [r for r in caplog.records if "lazy chunk indexing deferred" in r.message]
    assert len(warnings) == 1
    assert ("IntegrityError" if fault == "index" else "TypeError") in warnings[0].message
    assert len(warnings[0].message) < 150 and warnings[0].exc_info is None
    assert "private diagnostic" not in caplog.text
    assert CONTENT not in caplog.text


@pytest.mark.parametrize("column, malformed, error", [
    ("provenance", "[]", TypeError), ("embedding", "not-json", ValueError),
])
def test_direct_recall_preserves_healthy_result_and_retries_malformed_source(
    mcp, caplog, column, malformed, error,
):
    store = mcp.memory_store
    target = store.append_trace(CONTENT, {"scope": SCOPE})
    service = MemoryRecallService(store)
    service.refresh_pending_chunks(scope=SCOPE)
    blocker = store.create_node(level="concept", content="Violet botanical inventory.", context={"scope": SCOPE})
    original = store.connection.execute(f"SELECT {column} FROM nodes WHERE id = ?", (blocker.id,)).fetchone()[0]
    with store.transaction():
        store.connection.execute(f"UPDATE nodes SET {column} = ? WHERE id = ?", (malformed, blocker.id))
    arguments = ARGUMENTS | {"log_access": False, "log_event": False}
    baseline = service.memory_recall(**arguments, repair_pending=False)
    assert [result.node_id for result in baseline] == [target.id]
    with caplog.at_level(logging.WARNING, logger="living_memory.retrieval"):
        for _ in range(2):
            assert service.memory_recall(**arguments) == baseline
            assert store.count_node_chunks(blocker.id) == 0
            with pytest.raises(error):
                store.list_unchunked_nodes(scope=SCOPE)
    warnings = [record for record in caplog.records if "lazy chunk indexing deferred" in record.message]
    assert len(warnings) == 2
    assert all("1/1 nodes" in record.message for record in warnings)
    with store.transaction():
        store.connection.execute(f"UPDATE nodes SET {column} = ? WHERE id = ?", (original, blocker.id))
    assert service.memory_recall(**arguments)[0].node_id == target.id
    assert store.node_chunks_are_current(blocker.id)


@pytest.mark.parametrize("route", ["direct", "native"])
def test_required_malformed_knowledge_still_raises(mcp, route):
    store = mcp.memory_store
    target = store.append_trace(CONTENT, {"scope": SCOPE})
    with store.transaction():
        store.connection.execute("UPDATE nodes SET provenance = '[]' WHERE id = ?", (target.id,))
    if route == "direct":
        with pytest.raises(TypeError):
            MemoryRecallService(store).memory_recall(**ARGUMENTS)
    else:
        async def scenario():
            async with Client(mcp) as client:
                with pytest.raises(ToolError, match="list indices"):
                    await recall(client)
        asyncio.run(scenario())
    assert store.list_recall_events() == []
    assert store.connection.execute("SELECT access_count FROM nodes WHERE id = ?", (target.id,)).fetchone()[0] == 0


@pytest.mark.parametrize("column", ["content_fingerprint", "provenance"])
def test_pending_discovery_and_hydration_sql_read_errors_still_raise(mcp, column):
    store = mcp.memory_store
    store.append_trace("Violet botanical inventory.", {"scope": SCOPE})
    denied = []

    def authorize(action, table, field, *_):
        if action == sqlite3.SQLITE_READ and table == "nodes" and field == column:
            denied.append(field)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    store.connection.set_authorizer(authorize)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="prohibited|authorized"):
            MemoryRecallService(store).refresh_pending_chunks(scope=SCOPE)
    finally:
        store.connection.set_authorizer(None)
    assert denied
    assert store.list_recall_events() == []
    assert not store.connection.in_transaction


@pytest.mark.parametrize("change", ["content", "scope", "decay", "delete", "index"])
def test_pending_hydration_uses_current_source_after_discovery(mcp, change):
    store = mcp.memory_store
    node = store.create_node(level="concept", content="Violet botanical inventory.", context={"scope": SCOPE})
    service = MemoryRecallService(store)
    changed = False
    expected_chunks = []

    @contextmanager
    def guard():
        nonlocal changed, expected_chunks
        yield
        if changed:
            return
        changed = True
        if change == "content":
            store.update_node(node.id, content="Amber mineral inventory.")
        elif change == "scope":
            store.update_node(node.id, context={"scope": "project:moved"})
        elif change == "decay":
            store.soft_delete_node(node.id)
        elif change == "delete":
            with store.transaction():
                store.connection.execute("DELETE FROM nodes WHERE id = ?", (node.id,))
        else:
            store.update_node(node.id, embedding=service.embedder.embed(node.content))
            expected_chunks = store.list_node_chunks(node.id)

    service.refresh_pending_chunks(scope=SCOPE, guard=guard)
    if change == "content":
        assert store.node_chunks_are_current(node.id)
        current = store.get_node(node.id)
        chunk, = store.list_node_chunks(node.id)
        assert current.content[chunk.char_start:chunk.char_end] == "Amber mineral inventory."
        assert chunk.embedding == pytest.approx(service.embedder.embed(current.content))
    else:
        assert store.list_node_chunks(node.id) == expected_chunks
    assert not store.connection.in_transaction


def test_empty_pending_selection_resets_position_without_encoder_setup(mcp, monkeypatch):
    store = mcp.memory_store
    service = MemoryRecallService(store)
    monkeypatch.setattr(retrieval, "LAZY_CHUNK_NODE_LIMIT", 1)
    store.append_trace("Violet botanical inventory.", {
        "scope": SCOPE, "timestamp": "2026-01-02T00:00:00Z",
    })
    service.refresh_pending_chunks(scope=SCOPE)
    with monkeypatch.context() as patch:
        def unexpected_share():
            pytest.fail("empty selection must not initialize the encoder")
        patch.setattr(service, "_share_embedder_with_store", unexpected_share)
        service.refresh_pending_chunks(scope=SCOPE)
    earlier = store.append_trace("Amber mineral inventory.", {
        "scope": SCOPE, "timestamp": "2026-01-01T00:00:00Z",
    })
    later = store.append_trace("Indigo floral inventory.", {
        "scope": SCOPE, "timestamp": "2026-01-03T00:00:00Z",
    })
    service.refresh_pending_chunks(scope=SCOPE)
    assert store.node_chunks_are_current(earlier.id)
    assert not store.node_chunks_are_current(later.id)


@pytest.mark.parametrize("fault", ["global", "selected_nodes"])
def test_failed_repair_batch_has_bounded_attempts_and_diagnostics(mcp, caplog, fault):
    store = mcp.memory_store
    attempts = []
    store.connection.create_function("chunk_attempt", 1, lambda node: attempts.append(node) or 0)

    async def scenario():
        async with Client(mcp) as client:
            target = (await client.call_tool("memory_remember", {
                "content": CONTENT, "context": {"scope": SCOPE},
            })).data["node"]["id"]
            baseline = await recall(client)
            score = event_for(store, baseline).results[0]["vector_score"]
            assert score > 0 and store.node_chunks_are_current(target)
            pending = [store.append_trace(f"Violet botanical inventory {index}.", {
                "scope": SCOPE if index % 2 else "global",
                "timestamp": f"2026-02-01T00:00:{index:02d}Z",
            }).id for index in range(LAZY_CHUNK_NODE_LIMIT + 1)]
            store.connection.execute("CREATE TABLE denied_index (id TEXT PRIMARY KEY)")
            with store.transaction():
                store.connection.executemany("INSERT INTO denied_index VALUES (?)", [(node,) for node in pending[:-1]])
            condition = "1" if fault == "global" else "NEW.node_id IN (SELECT id FROM denied_index)"
            store.connection.execute(f"""CREATE TRIGGER reject_index BEFORE INSERT ON node_chunk_embeddings
                BEGIN
                    SELECT chunk_attempt(NEW.node_id);
                    SELECT CASE WHEN {condition} THEN RAISE(ABORT, 'private diagnostic') END;
                END""")
            for round_index in range(2):
                attempts.clear()
                response = await recall(client)
                assert body(response)["id"] == target
                assert body(response)["content"] == CONTENT
                assert event_for(store, response).results[0]["vector_score"] == score
                expected = pending[:-1] if round_index == 0 else [pending[-1], *pending[:-2]]
                assert attempts == expected
                assert len(attempts) == LAZY_CHUNK_NODE_LIMIT
                assert all(store.count_node_chunks(node) == 0 for node in pending[:-1])
                assert store.node_chunks_are_current(pending[-1]) == (
                    round_index == 1 and fault == "selected_nodes"
                )
                assert not store.connection.in_transaction
            store.connection.execute("DROP TRIGGER reject_index")
            for _ in range(2):
                await recall(client)
            assert all(store.node_chunks_are_current(node) for node in pending)
            assert store.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    with caplog.at_level(logging.WARNING, logger="living_memory.retrieval"):
        asyncio.run(scenario())
    warnings = [record for record in caplog.records if "lazy chunk indexing deferred" in record.message]
    assert len(warnings) == 2
    for index, record in enumerate(warnings):
        failed = LAZY_CHUNK_NODE_LIMIT - (index == 1 and fault == "selected_nodes")
        assert f"{failed}/{LAZY_CHUNK_NODE_LIMIT}" in record.message
        assert "IntegrityError" in record.message
        assert len(record.message) < 150 and record.exc_info is None
    assert "private diagnostic" not in caplog.text and CONTENT not in caplog.text


@pytest.mark.parametrize("view", ["knowledge", "full"])
def test_edited_source_remains_lexical_without_stale_vector_evidence(mcp, view):
    store = mcp.memory_store
    target = store.create_node(level="concept", content=CONTENT, context={"scope": SCOPE})
    updated = "Only publish the cobalt release when the harbor authority signs the manifest."

    async def scenario():
        async with Client(mcp) as client:
            warm = await recall(client, view=view)
            assert event_for(store, warm).results[0]["vector_score"] > 0
            store.update_node(target.id, content=updated)
            assert not store.node_chunks_are_current(target.id)
            attempts = deny_index(store, target)
            for attempt in (1, 2):
                response = await recall(client, query="cobalt harbor authority manifest", view=view)
                assert body(response)["content"] == updated
                summary, = event_for(store, response).results
                assert summary["vector_score"] == 0
                assert summary["bm25_score"] > 0
                assert "vector" not in summary["methods"]
                assert attempts == [0] * attempt
                assert store.count_node_chunks(target.id) == 0
                assert not store.connection.in_transaction
            store.connection.execute("DROP TRIGGER refuse_index")
            recovered = await recall(client, query="cobalt harbor authority manifest", view=view)
            assert body(recovered)["content"] == updated
            assert event_for(store, recovered).results[0]["vector_score"] > 0
            assert store.node_chunks_are_current(target.id)

    asyncio.run(scenario())


@pytest.mark.parametrize("table", ["nodes", "connections"])
def test_required_source_and_validity_read_failures_still_raise(mcp, table):
    store = mcp.memory_store
    target = store.append_trace(CONTENT, {"scope": SCOPE})
    denied = []

    def authorize(action, name, *_):
        if action == sqlite3.SQLITE_READ and name == table:
            denied.append(name)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    async def scenario():
        async with Client(mcp) as client:
            await recall(client)
            blocker = store.append_trace("Violet botanical inventory.", {"scope": SCOPE})
            deny_index(store, blocker)
            access = store.get_node(target.id).access_count
            events = len(store.list_recall_events())
            store.connection.set_authorizer(authorize)
            try:
                with pytest.raises(ToolError, match="prohibited|authorized"):
                    await recall(client)
            finally:
                store.connection.set_authorizer(None)
            assert denied
            assert store.get_node(target.id).access_count == access
            assert len(store.list_recall_events()) == events
            assert not store.connection.in_transaction

    asyncio.run(scenario())


@pytest.mark.parametrize("required_write", ["access", "event"])
def test_required_recall_writes_still_raise_when_indexing_also_fails(mcp, required_write):
    store = mcp.memory_store
    target = store.append_trace(CONTENT, {"scope": SCOPE})
    statements = {
        "access": "BEFORE UPDATE OF access_count ON nodes WHEN NEW.access_count > OLD.access_count",
        "event": "BEFORE INSERT ON recall_events",
    }

    async def scenario():
        async with Client(mcp) as client:
            await recall(client)
            blocker = store.append_trace("Violet botanical inventory.", {"scope": SCOPE})
            attempts = deny_index(store, blocker)
            access = store.get_node(target.id).access_count
            events = len(store.list_recall_events())
            store.connection.execute(f"""CREATE TRIGGER refuse_required {statements[required_write]}
                BEGIN SELECT RAISE(ABORT, 'required write unavailable'); END""")
            with pytest.raises(ToolError, match="required write unavailable"):
                await recall(client, used=[target.id])
            assert attempts == [0]
            assert store.get_node(target.id).access_count == access
            assert len(store.list_recall_events()) == events
            assert store.connection.execute("SELECT COUNT(*) FROM recall_feedback_marks").fetchone()[0] == 0
            assert not store.connection.in_transaction

    asyncio.run(scenario())
