"""Atomic consumption of selected knowledge on disposable SQLite stores."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
import sqlite3

import pytest

from living_memory import storage as storage_module
from living_memory.delivery import DeliveryHistory, shape_recall_results
from living_memory.feedback import resolve_explicit_marks
from living_memory.models import RecallEvent, SourcePassage
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore

SCOPE = "project:consumption"
QUERY = "cobalt routing"
TABLES = (
    "nodes", "recall_events", "recall_history_events", "recall_history_result_nodes",
    "recall_delivery_history", "recall_delivery_history_state",
)


class FixedEmbedder:
    def embed(self, text):
        return [1.0, 0.0, 0.0]


@pytest.fixture
def store():
    with MemoryStore(":memory:") as store:
        yield store


def seed(store, count=6):
    return [
        store.create_node(
            level="concept", content=f"cobalt routing fact {index}",
            context={"scope": SCOPE}, embedding=[1.0, 0.0, 0.0],
            provenance={"corrections": [{"reason": "synthetic correction"}]},
        )
        for index in range(count)
    ]


def service(store):
    return MemoryRecallService(store, embedder=FixedEmbedder())


def recall(retrieval, **kwargs):
    return retrieval.memory_recall(QUERY, scope=SCOPE, depth=0, max_results=5, **kwargs)


def snapshot(store):
    return {
        table: [tuple(row) for row in store.connection.execute(f"SELECT * FROM {table}")]
        for table in TABLES
    }


def consume(store, nodes, **kwargs):
    return store.record_recall(
        query=QUERY, scope=SCOPE, nodes=nodes,
        results=[{"node_id": node.id, "rank": rank + 1} for rank, node in enumerate(nodes)],
        **kwargs,
    )


@pytest.mark.parametrize("limit", [1, 5])
def test_selected_order_access_metadata_and_one_tail_transaction(store, monkeypatch, limit):
    nodes = seed(store)
    retrieval = service(store)
    selected = []
    sql = []
    decoded = []
    decode = storage_module._recall_event_from_row

    def track_decode(row):
        decoded.append(row["id"])
        return decode(row)

    monkeypatch.setattr(storage_module, "_recall_event_from_row", track_decode)

    def select(limited, residual, maximum, relevance):
        chosen = [residual[0], *reversed(limited[1:])]
        selected.extend(chosen)
        sql.clear()
        return chosen, [limited[0], *residual[1:]]

    store.connection.set_trace_callback(sql.append)
    results = retrieval.memory_recall(
        QUERY, scope=SCOPE, depth=0, max_results=limit, select_results=select,
        ambient_context={"transport_session_id": "consumer", "agent": "agent",
                         "task": "task", "session": "session", "task_pattern": "pattern",
                         "context_data": "z" * 131072},
    )
    store.connection.set_trace_callback(None)
    statements = [" ".join(statement.split()) for statement in sql]
    depth = commits = 0
    for statement in statements:
        if statement.startswith("SAVEPOINT "):
            depth += 1
        elif statement.startswith("RELEASE "):
            depth -= 1
            commits += depth == 0
        assert not statement.startswith(("COMMIT", "END TRANSACTION"))
    assert depth == 0 and commits == 1
    assert not any(s.startswith("SELECT") and "FROM nodes" in s for s in statements)
    assert not any(s.startswith("SELECT") and "FROM recall_events" in s for s in statements)
    assert decoded == []
    assert [result.node.id for result in results] == [result.node.id for result in selected]
    for result, original in zip(results, selected, strict=True):
        assert result.node == store.get_node(result.node.id)
        assert result.node.access_count == 1
        assert original.node.access_count == 0
        assert replace(result.node, access_count=0, last_accessed=original.node.last_accessed,
                       updated_at=original.node.updated_at) == original.node
        chunk, = store.list_node_chunks(result.node.id)
        assert result.scored_chunk_id == chunk.id
        assert result.semantic_passage == SourcePassage(
            content_fingerprint=chunk.content_fingerprint,
            char_start=chunk.char_start, char_end=chunk.char_end,
        )
        assert original.semantic_passage is None and original.scored_chunk_id is None
        assert replace(result, node=original.node, recall_event_id=None,
                       semantic_passage=original.semantic_passage,
                       scored_chunk_id=original.scored_chunk_id) == original
    delivered = {result.node.id for result in results}
    assert all(store.get_node(node.id).access_count == int(node.id in delivered) for node in nodes)
    event = store.get_recall_event(retrieval.last_recall_event_id)
    assert decoded == [event.id]
    assert event.result_ids == [result.node.id for result in results]
    assert (event.agent, event.task, event.session_id, event.transport_session_id) == (
        "agent", "task", "session", "consumer",
    )
    assert (event.scope, event.requested_scope, event.depth, event.max_results) == (SCOPE, SCOPE, "0", limit)
    assert event.ambient_context["task_pattern"] == "pattern"
    assert event.ambient_context["context_data"] == "z" * 131072
    marks = resolve_explicit_marks(
        store, [results[0].node.id], [retrieval.last_residual[0].node.id], "consumer",
    )
    assert marks[0].accepted and marks[0].event.id == event.id and marks[0].rank == 0
    assert not marks[1].accepted and marks[1].reject_reason == "not_delivered"
    shaped = shape_recall_results(
        results, view="knowledge", query=QUERY, delivery_history=DeliveryHistory(), snippet_max_chars=1200,
        context_value_max_chars=160, session_dedup=True,
    )
    assert all(item["content_ref"]["node_id"] == item["id"] for item in shaped)
    assert all(item["provenance"]["corrections"] for item in shaped)


@pytest.mark.parametrize("log_access", [False, True])
@pytest.mark.parametrize("log_event", [None, False, True])
@pytest.mark.parametrize("node_count", [0, 6])
def test_independent_flags_and_empty_results(store, log_access, log_event, node_count):
    nodes = seed(store, node_count)
    retrieval = service(store)
    results = recall(retrieval, log_access=log_access, log_event=log_event)
    has_event = log_access if log_event is None else log_event
    assert (retrieval.last_recall_event_id is not None) == has_event
    assert len(store.list_recall_events()) == int(has_event)
    delivered = {result.node.id for result in results}
    for node in nodes:
        assert store.get_node(node.id).access_count == int(log_access and node.id in delivered)
    assert all(result.node.access_count == int(log_access) for result in results)
    if has_event:
        assert store.get_recall_event(retrieval.last_recall_event_id).result_ids == [r.node.id for r in results]
    assert not store.connection.in_transaction


@pytest.mark.parametrize("query,limit", [(" ", 5), (QUERY, 0), (QUERY, -1)])
def test_empty_request_has_no_consumption(store, query, limit):
    seed(store)
    retrieval = service(store)
    recall(retrieval)
    before = snapshot(store)
    assert retrieval.memory_recall(query, scope=SCOPE, max_results=limit, log_event=True) == []
    assert retrieval.last_recall_event_id is None
    assert snapshot(store) == before


@pytest.mark.parametrize("stage", ["event", "history"])
def test_failed_tail_rolls_back_access_event_and_history(store, monkeypatch, stage):
    seed(store)
    retrieval = service(store)
    recall(retrieval)
    before = snapshot(store)
    if stage == "event":
        store.connection.execute("""
            CREATE TEMP TRIGGER reject_event BEFORE INSERT ON recall_events
            BEGIN SELECT RAISE(ABORT, 'injected event failure'); END
        """)
        expected = sqlite3.IntegrityError
    else:
        method = "_record_recall_history_event"
        original = getattr(store, method)

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected tail failure")

        monkeypatch.setattr(store, method, fail)
        expected = RuntimeError
    with pytest.raises(expected, match="injected"):
        recall(retrieval)
    assert retrieval.last_recall_event_id is None
    assert snapshot(store) == before
    assert not store.connection.in_transaction


def test_full_event_readback_failure_rolls_back_consumption(store, monkeypatch):
    nodes = seed(store)
    before = snapshot(store)
    original = store.get_recall_event

    def fail(event_id):
        assert original(event_id) is not None
        raise RuntimeError("injected readback failure")

    monkeypatch.setattr(store, "get_recall_event", fail)
    with pytest.raises(RuntimeError, match="injected readback"):
        consume(store, nodes)
    assert snapshot(store) == before
    assert not store.connection.in_transaction


@pytest.mark.parametrize("event_id_only", [False, True])
@pytest.mark.parametrize("invalid", [
    {"ambient_context": {"bad": object()}},
    {"resolved_scopes": [object()]},
    {"max_results": "invalid"},
    {"recall_map": {"bad": object()}},
    {"results": [{"node_id": "first", "bad": object()}]},
    {"query": None},
])
def test_record_validation_failure_rolls_back_consumption(store, event_id_only, invalid):
    nodes = seed(store)
    before = snapshot(store)
    with pytest.raises((TypeError, ValueError, sqlite3.IntegrityError)):
        store.record_recall(
            nodes=nodes, scope=SCOPE, event_id_only=event_id_only,
            **{"query": QUERY, **invalid},
        )
    assert snapshot(store) == before
    assert not store.connection.in_transaction


@pytest.mark.parametrize("record_event_only", [False, True])
def test_full_event_returns_sqlite_and_json_normalized_values(store, record_event_only):
    arguments = dict(
        query=17, scope=SCOPE, requested_scope="", resolved_scopes=[17, SCOPE],
        ambient_context={"agent": 7, "task": " task ", "session": "session",
                         "transport_session_id": "transport", "nested": {1: (2, 3)}},
        depth=2, max_results="3", results=[{"node_id": "first", "nested": {1: (2, 3)}},
                                         {"node_id": "first"}],
    )
    if record_event_only:
        event = store.record_recall_event(**arguments)
    else:
        _, event = store.record_recall(**arguments)
    assert isinstance(event, RecallEvent)
    assert event == store.get_recall_event(event.id)
    assert event.query == "17" and event.requested_scope == SCOPE
    assert event.resolved_scopes == ["17", SCOPE]
    assert event.ambient_context["nested"] == {"1": [2, 3]}
    assert event.results[0]["nested"] == {"1": [2, 3]}
    assert event.result_ids == ["first", "first"]
    assert (event.depth, event.max_results, event.agent, event.task) == ("2", 3, "7", " task ")
    assert event.created_at and not event.feedback_applied


@pytest.mark.parametrize("log_event", [False, True])
@pytest.mark.parametrize("event_id_only", [False, True])
def test_missing_node_rolls_back_batch_and_preserves_outer_write(store, log_event, event_id_only):
    nodes = seed(store, 2)
    missing = replace(nodes[1], id="absent")
    with store.transaction():
        store.set_kv("outer-write", "kept")
        before = snapshot(store)
        with pytest.raises(KeyError, match="absent"):
            consume(store, [nodes[0], missing], log_event=log_event, event_id_only=event_id_only)
        assert snapshot(store) == before
        assert store.connection.in_transaction
    assert store.get_kv("outer-write") == "kept"
    assert not store.connection.in_transaction


def test_outer_rollback_undoes_successful_consumption(store):
    nodes = seed(store, 2)
    before = snapshot(store)
    with pytest.raises(RuntimeError, match="outer"):
        with store.transaction():
            updated, event = consume(store, nodes)
            assert all(node.access_count == 1 for node in updated)
            assert store.get_recall_event(event.id) is not None
            assert store.connection.in_transaction
            raise RuntimeError("outer")
    assert snapshot(store) == before
    assert not store.connection.in_transaction


@pytest.mark.parametrize("event_id_only", [False, True])
def test_access_preserves_selected_validity_and_repeated_node_order(store, event_id_only):
    node = seed(store, 1)[0]
    node = store.delete_node(node.id, reason="synthetic decay")
    updated, recorded = consume(store, [node, node], event_id_only=event_id_only)
    event = store.get_recall_event(recorded) if event_id_only else recorded
    assert [item.access_count for item in updated] == [1, 2]
    assert event.result_ids == [node.id, node.id]
    assert node.access_count == 0
    assert all(item.decayed and item.decay_reason == node.decay_reason for item in updated)
    assert all(item.corrections == node.corrections and item.content == node.content for item in updated)
    assert store.get_node(node.id).access_count == 2


def test_concurrent_stale_snapshots_keep_both_increments_and_events(tmp_path):
    database = tmp_path / "concurrent.sqlite3"
    with MemoryStore(database) as first, MemoryStore(database) as second:
        nodes = seed(first, 2)
        snapshots = [nodes, [second.get_node(node.id) for node in nodes]]
        ready = Barrier(2, timeout=10)

        def write(store, selected):
            ready.wait()
            return consume(store, selected)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(write, store, selected) for store, selected in zip(
                (first, second), snapshots, strict=True,
            )]
            written = [future.result(timeout=15) for future in futures]
        assert sorted(updated[0].access_count for updated, _ in written) == [1, 2]
        assert all(first.get_node(node.id).access_count == 2 for node in nodes)
        assert len(first.list_recall_events()) == 2
        assert all(event.result_ids == [node.id for node in nodes] for _, event in written)
        assert not first.connection.in_transaction and not second.connection.in_transaction
