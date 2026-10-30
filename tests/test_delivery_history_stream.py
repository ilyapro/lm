"""Streaming delivery evidence over disposable events and native MCP recalls."""

import asyncio
from contextlib import contextmanager
from dataclasses import replace
import gc
import hashlib
import json
import sqlite3
import tracemalloc

import pytest
from fastmcp import Client

from living_memory import server
from living_memory.delivery import DeliveryHistory
from living_memory.models import Node
from living_memory.retrieval import MemoryRecallService, RecallResult, _recall_result_summary
from living_memory.storage import MemoryStore


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def evidence(node, *, excerpt=None):
    return {"node_id": node.id, "score": 0.75, "delivery": {
        "kind": "full" if excerpt is None else "snippet",
        "content_sha256": digest(node.content),
        "delivered_sha256": digest(node.content if excerpt is None else excerpt),
    }}


class HistoryCursor:
    def __init__(self, cursor, failure):
        self.cursor = cursor
        self.failure = failure
        self.reads = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.failure == "read" and self.reads == 1:
            raise sqlite3.OperationalError("synthetic late history read")
        row = next(self.cursor)
        self.reads += 1
        return row

    def fetchall(self):
        pytest.fail("delivery evidence must not collect encoded rows")

    def close(self):
        self.cursor.close()
        self.closed = True
        if self.failure == "close":
            raise sqlite3.OperationalError("synthetic late history close")


class HistoryConnection:
    def __init__(self, connection):
        self.connection = connection
        self.cursors = []
        self.failure = None

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def execute(self, sql, *args):
        cursor = self.connection.execute(sql, *args)
        if " ".join(sql.split()).startswith(
            "SELECT results FROM recall_events WHERE transport_session_id = ?"
        ):
            cursor = HistoryCursor(cursor, self.failure)
            self.cursors.append(cursor)
        return cursor


@pytest.fixture
def event_store():
    # Only the acquisition table is needed; the methods under test are real.
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TABLE recall_events (results TEXT, transport_session_id TEXT, created_at TEXT)")
    connection.execute("CREATE INDEX transport_events ON recall_events (transport_session_id, created_at DESC)")
    store = MemoryStore.__new__(MemoryStore)
    store._conn = HistoryConnection(connection)
    try:
        yield store
    finally:
        connection.close()


def insert(store, raw, transport="reader", at="2026-01-01"):
    store.connection.execute("INSERT INTO recall_events VALUES (?, ?, ?)", (raw, transport, at))


def stream_list(store, *args, **kwargs):
    with store.stream_recall_delivery_history(*args, **kwargs) as results:
        return list(results)


@pytest.mark.parametrize("reader", [stream_list, MemoryStore.recall_delivery_history])
def test_window_precedes_filtering_and_preserves_full_rows(event_store, reader):
    store = event_store
    full = {"node_id": "full", "score": 0.5, "methods": ["bm25"], "delivery": {}}
    snippet = {"node_id": "snippet", "score": 0.9, "delivery": {"kind": "snippet"}}
    insert(store, "malformed", at="2025-01-01")
    insert(store, json.dumps([full, snippet]))
    insert(store, '[null, false, 2, {}, {"delivery": []}]')
    insert(store, 'null', transport="other", at="2028-01-01")
    insert(store, '12', transport="reader-extra", at="2028-01-01")
    # A later rowid with an older time must not displace the current window.
    insert(store, 'null', at="2024-01-01")
    assert reader(store, "reader", max_events=1) == []
    assert reader(store, "reader", max_events=2) == [full, snippet]
    with pytest.raises(json.JSONDecodeError):
        reader(store, "reader", max_events=3)
    assert all(cursor.closed for cursor in store.connection.cursors)
    queries = len(store.connection.cursors)
    for transport, limit in [(None, 200), ("", 200), ("reader", 0), ("reader", -1)]:
        assert reader(store, transport, max_events=limit) == []
    assert len(store.connection.cursors) == queries
    assert reader(store, "missing") == []


@pytest.mark.parametrize("raw,error", [
    ("not json", json.JSONDecodeError), ("null", TypeError), ("12", TypeError),
    ("true", TypeError), ("{}", None), ('"text"', None), (None, None), ("", None),
])
@pytest.mark.parametrize("reader", [stream_list, MemoryStore.recall_delivery_history])
def test_selected_json_contract_and_cursor_close(event_store, raw, error, reader):
    insert(event_store, raw)
    if error:
        with pytest.raises(error):
            reader(event_store, "reader")
    else:
        assert reader(event_store, "reader") == []
    assert event_store.connection.cursors[-1].closed


def test_early_termination_and_consumer_error_close_cursor(event_store):
    insert(event_store, '[{"delivery": {}}, {"delivery": {}}]')
    for fail in (False, True):
        try:
            with event_store.stream_recall_delivery_history("reader") as results:
                assert next(results) == {"delivery": {}}
                if fail:
                    raise RuntimeError("synthetic consumer failure")
        except RuntimeError:
            assert fail
        cursor = event_store.connection.cursors[-1]
        assert cursor.closed
        with pytest.raises(sqlite3.ProgrammingError):
            next(cursor.cursor)
    insert(event_store, '[{"node_id": "new", "delivery": {}}]')
    assert stream_list(event_store, "reader")[0]["node_id"] == "new"


def test_proof_keeps_all_versions_and_exact_excerpts_in_the_window(event_store):
    node = Node(id="source", level="trace", content="Current fact.")
    rows = [evidence(replace(node, content=f"Version {i}"), excerpt=f"Excerpt {j}")
            for i in range(240) for j in range(2)]
    rows.extend([
        evidence(node, excerpt="Current excerpt."),
        {"node_id": "legacy"},
        *({"node_id": kind, "delivery": {**evidence(node)["delivery"], "kind": kind}}
          for kind in ("twin_duplicate", "session_duplicate", "below_threshold", "omitted")),
    ])
    insert(event_store, json.dumps(rows))
    with event_store.stream_recall_delivery_history("reader") as results:
        history = DeliveryHistory(results)
    expected = {(node.id, digest(f"Version {i}")): {digest(f"Excerpt {j}") for j in range(2)}
                for i in range(240)}
    expected[(node.id, digest(node.content))] = {digest("Current excerpt.")}
    assert history._versions == expected
    assert history.has_content(node)
    assert history.covers(node, "Current excerpt.")
    assert not history.covers(node, node.content)
    assert not history.covers(node, "Different excerpt.")
    assert not history.has_content(replace(node, id="independent-source"))
    assert not history.has_content(replace(node, content="Updated fact."))


def test_repeated_evidence_retention_is_per_event(event_store):
    nodes = [Node(id=f"synthetic-{i}", level="trace", content=f"Known fact {i}") for i in range(20)]
    summaries = []
    for i, node in enumerate(nodes):
        summary = _recall_result_summary(i, RecallResult(
            node=node, score=0.9, bm25_score=0.6, vector_score=0.7, graph_score=0.5,
            methods=("bm25", "vector", "graph"), path=("anchor", "neighbor", node.id),
        ))
        summary["delivery"] = evidence(node)["delivery"]
        summaries.append(summary)
    encoded = json.dumps(summaries)
    for i in range(205):
        insert(event_store, encoded, at=f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}Z")
    del summaries, summary, encoded

    def measure(stream):
        gc.collect()
        tracemalloc.start()
        try:
            if stream:
                with event_store.stream_recall_delivery_history("reader") as results:
                    history = DeliveryHistory(results)
            else:
                history = DeliveryHistory(event_store.recall_delivery_history("reader"))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return history, peak

    streamed, stream_peak = measure(True)
    listed, list_peak = measure(False)
    expected = {(node.id, digest(node.content)): {digest(node.content)} for node in nodes}
    assert streamed._versions == listed._versions == expected
    assert all(streamed.covers(node, "Any excerpt") for node in nodes)
    assert [cursor.reads for cursor in event_store.connection.cursors] == [200, 200]
    assert all(cursor.closed for cursor in event_store.connection.cursors)
    # A relative regression bound for this repeated-proof fixture, not a runtime cap.
    assert stream_peak < list_peak / 10
    print(f"delivery history Python peak: stream={stream_peak}, list={list_peak}")


@pytest.mark.parametrize("failure", [None, "read", "parse", "reduce", "close"])
@pytest.mark.parametrize("view", ["knowledge", "full"])
def test_native_recall_finishes_history_before_response_and_discards_partial_proof(
    tmp_path, monkeypatch, caplog, failure, view,
):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
        "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic",
        "LM_AUTO_CONSOLIDATE_POLICY": "off", "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
        "LM_RECALL_MAP": "0", "LM_RECALL_MIN_SCORE": "off",
    }.items():
        monkeypatch.setenv(key, value)
    mcp = server.create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    node = store.create_node(level="concept", content="Cobalt release requires the verified signature.",
                             context={"scope": "project:synthetic"})
    args = {"query": "cobalt release signature", "scope": node.scope,
            "depth": 0, "max_results": 1, "view": view}

    async def scenario():
        async with Client(mcp) as client:
            async def recall():
                response = (await client.call_tool("memory_recall", args)).data
                entry = response["results"][0]
                assert entry.get("node", entry)["id"] == node.id
                return response, entry, entry.get("node", entry)

            first, _, _ = await recall()
            event = store.get_recall_event(first["recall_event_id"])
            store.record_recall_event(
                query=event.query, scope=event.scope,
                ambient_context={"transport_session_id": event.transport_session_id},
                results=event.results,
            )
            if failure == "parse":
                with store.transaction():
                    store.connection.execute("UPDATE recall_events SET results = 'null' WHERE id = ?", (event.id,))
            watch = HistoryConnection(store.connection)
            watch.failure = failure
            monkeypatch.setattr(store, "_conn", watch)
            original_response = MemoryRecallService.recall_response

            @contextmanager
            def checked_response(self, *args, **kwargs):
                assert watch.cursors and watch.cursors[-1].closed
                with original_response(self, *args, **kwargs) as results:
                    yield results

            monkeypatch.setattr(MemoryRecallService, "recall_response", checked_response)

            def forbid_list(*args, **kwargs):
                pytest.fail("native recall must consume the stream")

            monkeypatch.setattr(store, "recall_delivery_history", forbid_list)
            if failure == "reduce":
                class FailedHistory(DeliveryHistory):
                    def __init__(self, results=()):
                        def interrupted():
                            for result in results:
                                yield result
                                raise ValueError("synthetic late history reducer")
                        super().__init__(interrupted())
                monkeypatch.setattr(server, "DeliveryHistory", FailedHistory)

            _, entry, body = await recall()
            assert watch.cursors[-1].reads >= 1
            assert entry["delivery"] == ("full" if failure else "session_duplicate")
            if failure:
                assert body["content"] == node.content
                assert "could not read recall delivery history" in caplog.text
            else:
                assert watch.cursors[-1].reads == 2
            watch.failure = None
            monkeypatch.setattr(server, "DeliveryHistory", DeliveryHistory)
            if failure == "parse":
                with store.transaction():
                    store.connection.execute("UPDATE recall_events SET results = ? WHERE id = ?",
                                             (json.dumps(event.results), event.id))
            updated = node.content + " Check the countersignature too."
            store.update_node(node.id, content=updated)
            _, entry, body = await recall()
            assert entry["delivery"] == "full" and body["content"] == updated
            _, entry, _ = await recall()
            assert entry["delivery"] == "session_duplicate"
            assert len(watch.cursors) == 3 and all(cursor.closed for cursor in watch.cursors)

    try:
        asyncio.run(scenario())
    finally:
        mcp.auto_consolidation.close()
        store.close()
