"""New MCP knowledge owns its current index and survives derived-only faults."""

import asyncio
import logging
import sqlite3

from fastmcp import Client
import pytest

from living_memory.chunking import chunk_text
from living_memory.retrieval import LAZY_CHUNK_NODE_LIMIT
from living_memory.server import create_mcp_server


SCOPE = "project:fresh-write"
CONTEXT = {"scope": SCOPE, "agent": "fixture", "task": "cobalt certificate"}
CONTENT = "The cobalt registry rejects image pulls until its expired certificate is renewed."
QUERY = "cobalt registry expired certificate"
LONG = "Violet plants grow beside the amber pond. " * 160 + CONTENT


@pytest.fixture
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
    monkeypatch.setenv("LM_RECALL_MAP", "0")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    server = create_mcp_server(tmp_path / "memory.sqlite3")
    try:
        yield server
    finally:
        server.auto_consolidation.close()
        server.memory_store.close()


async def remember(client, content, **kwargs):
    return (await client.call_tool("memory_remember", {
        "content": content, "context": CONTEXT, **kwargs,
    })).data


async def recall(client, **kwargs):
    return (await client.call_tool("memory_recall", {
        "query": QUERY, "scope": SCOPE, "depth": 0, "max_results": 3,
        "ambient_context": CONTEXT, **kwargs,
    })).data


async def write(client, operation, original, content, **kwargs):
    if operation == "remember":
        response = await remember(client, content, **kwargs)
        return response, response["node"]["id"]
    response = (await client.call_tool("memory_teach", {
        "trace_id": original, "correction": content, "context": CONTEXT, **kwargs,
    })).data
    return response, response["corrective_trace"]["id"]


def assert_current_chunks(store, node_id, content):
    windows = chunk_text(content)
    chunks = store.list_node_chunks(node_id)
    assert len(chunks) == len(windows)
    assert store.node_chunks_are_current(node_id)
    for chunk, window in zip(chunks, windows, strict=True):
        assert chunk.chunk_index == window.chunk_index
        assert (chunk.char_start, chunk.char_end) == (window.char_start, window.char_end)
        assert content[chunk.char_start:chunk.char_end] == window.text


@pytest.mark.parametrize("operation", ["remember", "teach"])
@pytest.mark.parametrize("legacy_backlog", [False, True])
def test_new_index_does_not_wait_for_other_nodes(mcp, operation, legacy_backlog):
    store = mcp.memory_store

    async def scenario():
        async with Client(mcp) as client:
            original = (await remember(client, "The cobalt registry certificate never expires."))["node"]["id"]
            pending = []
            for number in range(LAZY_CHUNK_NODE_LIMIT + 1):
                content = f"Botanical observation {number}: violet plants grow beside the amber pond."
                if legacy_backlog:
                    pending.append(store.append_trace(content, CONTEXT).id)
                else:
                    await remember(client, content)
            foreign = (await remember(client, CONTENT, context={"scope": "project:excluded"}))["node"]["id"]
            response, target = await write(client, operation, original, CONTENT)
            assert "indexing" not in response
            assert_current_chunks(store, target, CONTENT)
            assert all(store.count_node_chunks(node_id) == 0 for node_id in pending)
            # A pre-existing maintenance failure must not hide this new index.
            if pending:
                pending_ids = ",".join(f"'{node_id}'" for node_id in pending)
                store.connection.execute(f"""CREATE TRIGGER reject_old_index
                    BEFORE INSERT ON node_chunk_embeddings WHEN NEW.node_id IN ({pending_ids})
                    BEGIN SELECT RAISE(ABORT, 'old index unavailable'); END""")
            result = await recall(client)
            ids = {entry["id"] for entry in result["results"]}
            assert target in ids and foreign not in ids
            scored = next(row for row in store.get_recall_event(result["recall_event_id"]).results
                          if row["node_id"] == target)
            assert scored["vector_score"] > 0
            assert all(store.count_node_chunks(node_id) == 0 for node_id in pending)
            looked = (await client.call_tool("memory_lookup", {"node_id": target})).data
            assert looked["results"][0]["content"] == CONTENT
            if operation == "teach":
                old = (await client.call_tool("memory_lookup", {"node_id": original})).data["results"][0]
                assert old["superseded"] and target in old["superseded_by"]
                for entry in result["results"]:
                    if entry["id"] == original:
                        assert entry["superseded"] and target in entry["superseded_by"]

    asyncio.run(scenario())


def test_multichunk_correction_keeps_both_content_identities(mcp):
    store = mcp.memory_store

    async def scenario():
        async with Client(mcp) as client:
            original = (await remember(client, LONG))["node"]["id"]
            assert len(chunk_text(LONG)) > 1
            assert_current_chunks(store, original, LONG)
            first = await recall(client)
            assert original in {entry["id"] for entry in first["results"]}
            response, corrected = await write(client, "teach", original, CONTENT)
            assert "indexing" not in response
            assert_current_chunks(store, corrected, CONTENT)
            assert len(store.list_node_chunks(corrected)) == 1
            result = await recall(client)
            assert corrected in {entry["id"] for entry in result["results"]}
            for entry in result["results"]:
                if entry["id"] == original:
                    assert entry["superseded"] and corrected in entry["superseded_by"]
            for node_id, content in ((original, LONG), (corrected, CONTENT)):
                looked = (await client.call_tool("memory_lookup", {"node_id": node_id})).data
                assert looked["results"][0]["content"] == content
            with pytest.raises(ValueError, match="append-only"):
                store.update_node(corrected, content="Changed trace content")
            assert_current_chunks(store, corrected, CONTENT)

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["remember", "teach"])
def test_partial_index_failure_keeps_atomic_knowledge_and_feedback(mcp, operation, caplog):
    store = mcp.memory_store
    attempted = []
    store.connection.create_function("chunk_attempt", 1, lambda index: attempted.append(index) or 0)

    async def scenario():
        async with Client(mcp) as client:
            original = (await remember(client, CONTENT))["node"]["id"]
            prior = await recall(client)
            store.connection.execute("""CREATE TRIGGER reject_index BEFORE INSERT ON node_chunk_embeddings
                BEGIN
                    SELECT chunk_attempt(NEW.chunk_index);
                    SELECT CASE WHEN NEW.chunk_index=1 THEN RAISE(ABORT, 'private diagnostic') END;
                END""")
            response, target = await write(client, operation, original, LONG, used=[original])
            assert response["indexing"] == {"status": "deferred", "node_ids": [target]}
            assert attempted == [0, 1]
            assert store.count_node_chunks(target) == 0
            assert not store.connection.in_transaction
            # A fresh connection sees the whole outcome, including teach/marks.
            with sqlite3.connect(store.db_path) as reader:
                assert reader.execute("SELECT content FROM nodes WHERE id=?", (target,)).fetchone() == (LONG,)
                assert reader.execute("SELECT feedback_trace_id FROM recall_events WHERE id=?",
                                      (prior["recall_event_id"],)).fetchone() == (target,)
                assert reader.execute("SELECT node_id, source_id FROM recall_feedback_marks").fetchall() == [(original, target)]
                if operation == "teach":
                    assert reader.execute("SELECT source_id, target_id FROM connections WHERE type='supersedes'").fetchall() == [(target, original)]
            looked = (await client.call_tool("memory_lookup", {"node_id": target})).data
            assert looked["results"][0]["content"] == LONG
            store.connection.execute("DROP TRIGGER reject_index")
            await recall(client)
            assert_current_chunks(store, target, LONG)
            assert store.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    with caplog.at_level(logging.WARNING, logger="living_memory.server"):
        asyncio.run(scenario())
    warnings = [record for record in caplog.records if "new trace indexing deferred" in record.message]
    assert len(warnings) == 1 and "IntegrityError" in warnings[0].message
    assert warnings[0].exc_info is None and "private diagnostic" not in caplog.text


def test_deferred_write_repairs_while_an_earlier_node_still_fails(mcp):
    store = mcp.memory_store
    attempts = []
    store.connection.create_function("chunk_attempt", 1, lambda node: attempts.append(node) or 0)
    store.connection.execute("""CREATE TRIGGER reject_index BEFORE INSERT ON node_chunk_embeddings
        BEGIN SELECT RAISE(ABORT, 'temporary index outage'); END""")

    async def scenario():
        async with Client(mcp) as client:
            blocker = await remember(client, "Violet botanical inventory.", context={
                **CONTEXT, "timestamp": "2026-02-01T00:00:00Z",
            })
            target = await remember(client, CONTENT, context={
                **CONTEXT, "timestamp": "2026-02-02T00:00:00Z",
            })
            blocker_id, target_id = blocker["node"]["id"], target["node"]["id"]
            for response, node_id in ((blocker, blocker_id), (target, target_id)):
                assert response["indexing"] == {"status": "deferred", "node_ids": [node_id]}
                assert store.count_node_chunks(node_id) == 0
            store.connection.execute("DROP TRIGGER reject_index")
            store.connection.execute("CREATE TABLE denied_index (id TEXT PRIMARY KEY)")
            with store.transaction():
                store.connection.execute("INSERT INTO denied_index VALUES (?)", (blocker_id,))
            store.connection.execute("""CREATE TRIGGER reject_index BEFORE INSERT ON node_chunk_embeddings
                BEGIN
                    SELECT chunk_attempt(NEW.node_id);
                    SELECT CASE WHEN NEW.node_id IN (SELECT id FROM denied_index)
                        THEN RAISE(ABORT, 'node index unavailable') END;
                END""")
            scores = []
            for attempt in (1, 2, 3):
                result = await recall(client)
                assert next(row for row in result["results"] if row["id"] == target_id)["content"] == CONTENT
                scored = next(row for row in store.get_recall_event(result["recall_event_id"]).results
                              if row["node_id"] == target_id)
                assert scored["vector_score"] > 0
                scores.append(scored["vector_score"])
                assert_current_chunks(store, target_id, CONTENT)
                assert store.count_node_chunks(blocker_id) == 0
                assert attempts.count(blocker_id) == attempt
                assert attempts.count(target_id) == 1
                assert not store.connection.in_transaction
            for node_id, content in ((blocker_id, blocker["node"]["content"]), (target_id, CONTENT)):
                looked = (await client.call_tool("memory_lookup", {"node_id": node_id})).data
                assert looked["results"][0]["content"] == content
            store.connection.execute("DROP TRIGGER reject_index")
            recovered = await recall(client)
            scored = next(row for row in store.get_recall_event(recovered["recall_event_id"]).results
                          if row["node_id"] == target_id)
            assert scores == [scored["vector_score"]] * 3
            assert_current_chunks(store, blocker_id, blocker["node"]["content"])
            assert store.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["raise", "missing", "empty_tail"])
@pytest.mark.parametrize("operation", ["remember", "teach"])
def test_unavailable_encoding_never_reports_a_partial_index(mcp, operation, failure):
    store = mcp.memory_store

    def unavailable(texts):
        if failure == "raise":
            raise RuntimeError("encoder unavailable")
        vectors = [[1.0, 0.0] for _ in texts]
        if failure == "missing":
            return vectors[:-1]
        vectors[-1] = []
        return vectors

    async def scenario():
        async with Client(mcp) as client:
            original = (await remember(client, CONTENT))["node"]["id"]
            store.set_chunk_embedder(unavailable)
            response, target = await write(client, operation, original, LONG)
            assert response["indexing"] == {"status": "deferred", "node_ids": [target]}
            assert store.get_node(target).content == LONG
            assert store.count_node_chunks(target) == 0
            assert target in {node.id for node in store.list_unchunked_nodes()}

    asyncio.run(scenario())


def test_rejected_alternatives_own_their_indices(mcp):
    store = mcp.memory_store

    async def scenario():
        async with Client(mcp) as client:
            result = await remember(client, CONTENT, alternatives_considered=[{
                "approach": "Disable certificate validation",
                "rejected_because": "Invalid signatures must remain rejected",
            }])
            for node_id in [result["node"]["id"], *result["rejected_alternatives"]]:
                assert_current_chunks(store, node_id, store.get_node(node_id).content)

    asyncio.run(scenario())
