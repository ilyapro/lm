"""Offline receipts describe committed effects, including on retry and races."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace
import sqlite3

from fastmcp import Client
from fastmcp.exceptions import ToolError
import pytest

from living_memory import feedback
from living_memory.attestation import (
    ATTESTATION_MIN_CONTAINMENT,
    AttestationError,
    attest_recall_usage,
    canonical_evidence,
)
from living_memory.query_anchors import resolve_query_anchor
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore


CONTENT = "Cobalt certificate renewal requires preserving the trusted issuer chain."
QUERY = "cobalt certificate renewal"
SCOPE = "project:attestation-atomicity"
VECTOR = [1.0, 0.0, 0.0]


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
        "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic",
        "LM_AUTO_CONSOLIDATE_POLICY": "off",
        "LM_RETRIEVAL_TUNING_POLICY": "fixed",
        "LM_LOOKUP_CREDIT_POLICY": "off",
    }.items():
        monkeypatch.setenv(key, value)
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", expose_attest=True)
    store = mcp.memory_store
    node = store.create_node(level="concept", content=CONTENT,
                             context={"scope": SCOPE}, embedding=VECTOR)
    trace = store.create_node(level="trace", content="The certificate rollout completed.",
                              context={"scope": SCOPE}, embedding=VECTOR)
    event = store.record_recall_event(query=QUERY, scope=SCOPE, results=[{
        "node_id": node.id, "bm25_score": 1.0, "vector_score": 0.8, "graph_score": 0.0,
    }])
    r = SimpleNamespace(mcp=mcp, store=store, node=node, trace=trace, event=event)
    try:
        yield r
    finally:
        mcp.auto_consolidation.close()
        store.close()


async def native(r, **kwargs):
    async with Client(r.mcp) as client:
        return (await client.call_tool("memory_attest", {
            "recall_event_id": r.event.id, "evidence": [CONTENT], **kwargs,
        })).data


def attest(r, route="direct", **kwargs):
    if route == "native":
        return asyncio.run(native(r, **kwargs))
    return attest_recall_usage(r.store, r.event.id, [CONTENT], **kwargs)


def durable_state(store):
    # A separate connection sees committed data, including auxiliary tables.
    with sqlite3.connect(store.db_path) as reader:
        return tuple(reader.iterdump())


@pytest.mark.parametrize("route", ["direct", "native"])
@pytest.mark.parametrize("stage", [
    "claim", "numeric", "closure", "completion", "after_completion", "commit",
    "endpoint_id", "endpoint_decayed",
])
def test_required_failure_rolls_back_then_retry_and_replay_are_honest(
    runtime, monkeypatch, route, stage,
):
    r = runtime
    before = durable_state(r.store)
    method = {
        "claim": "claim_recall_attestation", "numeric": "update_retrieval_weights",
        "closure": "mark_recall_event_feedback", "completion": "complete_recall_attestation",
        "after_completion": "complete_recall_attestation",
    }.get(stage)
    with monkeypatch.context() as fault:
        if stage.startswith("endpoint_"):
            original = r.store.mark_recall_event_feedback
            column = stage.removeprefix("endpoint_")

            def fail(*args, **kwargs):
                def authorize(action, table, name, *_):
                    return (sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ
                            and table == "nodes" and name == column else sqlite3.SQLITE_OK)

                r.store.connection.set_authorizer(authorize)
                try:
                    return original(*args, **kwargs)
                finally:
                    r.store.connection.set_authorizer(None)

            fault.setattr(r.store, "mark_recall_event_feedback", fail)
        elif method:
            original = getattr(r.store, method)

            def fail(*args, **kwargs):
                if stage != "completion":
                    original(*args, **kwargs)
                raise RuntimeError("injected attestation failure")

            fault.setattr(r.store, method, fail)
        else:
            def deny_commit(action, operation, *_):
                return (sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_TRANSACTION
                        and operation == "COMMIT" else sqlite3.SQLITE_OK)

            r.store.connection.set_authorizer(deny_commit)
        try:
            with pytest.raises(ToolError if route == "native" else (RuntimeError, sqlite3.DatabaseError)):
                attest(r, route, trace_id=r.trace.id)
        finally:
            r.store.connection.set_authorizer(None)

    assert not r.store.connection.in_transaction
    assert durable_state(r.store) == before
    assert r.store.list_recall_attestations() == []
    result = attest(r, route, trace_id=r.trace.id)
    assert result["replay"] is False and result["credited"] is True
    assert result["closed_by_attestation"] is True
    assert result["feedback_trace_id"] == r.trace.id
    assert r.store.get_node(r.node.id).usefulness_score == pytest.approx(0.1)
    anchor, = r.store.list_query_anchors()
    edge, = r.store.list_query_anchor_edges(anchor_id=anchor.id)
    assert result["anchor_ids"] == [anchor.id] and edge.hits == 1
    committed = durable_state(r.store)
    assert attest(r, route, trace_id=r.trace.id) == dict(result, replay=True)
    assert durable_state(r.store) == committed


def test_native_attestation_accepts_decayed_closing_trace(runtime):
    r = runtime
    r.store.soft_delete_node(r.trace.id)
    result = attest(r, "native", trace_id=r.trace.id)
    assert result["credited"] and result["closed_by_attestation"]
    assert result["feedback_trace_id"] == r.trace.id
    assert r.store.get_node(r.trace.id).decayed
    assert r.store.get_recall_event(r.event.id).feedback_trace_id == r.trace.id


def test_direct_api_never_returns_a_provisional_nested_receipt(runtime):
    r = runtime
    before = durable_state(r.store)
    with r.store.transaction():
        with pytest.raises(RuntimeError, match="attestation must own its transaction"):
            attest(r, trace_id=r.trace.id)
    assert durable_state(r.store) == before
    result = attest(r)
    committed = durable_state(r.store)
    with r.store.transaction():
        with pytest.raises(RuntimeError, match="attestation must own its transaction"):
            attest(r)
    assert durable_state(r.store) == committed
    assert result["credited"]


@pytest.mark.parametrize("past_effect", [False, True])
def test_incomplete_claim_replays_without_guessing_past_effects(runtime, past_effect):
    r = runtime
    items, digest = canonical_evidence([CONTENT])
    claim = r.store.claim_recall_attestation(
        recall_event_id=r.event.id, evidence_sha256=digest,
        evidence_items=len(items), evidence_chars=sum(map(len, items)),
        min_containment=ATTESTATION_MIN_CONTAINMENT, grounded_node_ids=[r.node.id],
    )
    if past_effect:
        feedback.apply_retrieval_feedback(r.store, SimpleNamespace(
            node_id=r.node.id, bm25_score=1.0, vector_score=0.8, graph_score=0.0,
        ), scope=SCOPE)
    before = durable_state(r.store)
    result = attest(r, trace_id="a trace that no longer exists")
    assert result["attestation_id"] == claim.id
    assert result["replay"] is True and result["credited"] is False
    assert result["anchor_ids"] == [] and result["closed"] is False
    assert durable_state(r.store) == before


def test_independent_connections_concurrently_attest_once(runtime, monkeypatch):
    r = runtime
    ready = Barrier(2)

    def encode(queries):
        ready.wait(timeout=5)
        return [VECTOR for _ in queries]

    monkeypatch.setattr(feedback, "_anchor_embedder", lambda store: encode)
    with MemoryStore(r.store.db_path) as other, ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(attest_recall_usage, store, r.event.id, [CONTENT],
                               trace_id=r.trace.id) for store in (r.store, other)]
        results = [future.result(timeout=10) for future in futures]
    first, replay = sorted(results, key=lambda result: result["replay"])
    assert first["replay"] is False and replay == dict(first, replay=True)
    assert r.store.get_node(r.node.id).usefulness_score == pytest.approx(0.1)
    assert len(r.store.list_recall_attestations()) == 1
    anchor, = r.store.list_query_anchors()
    edge, = r.store.list_query_anchor_edges(anchor_id=anchor.id)
    assert edge.hits == 1 and anchor.reinforcement_count == 0
    assert r.store.get_recall_event(r.event.id).feedback_trace_id == r.trace.id


@pytest.mark.parametrize("failure", ["encode", "presence", "anchor_write", "edge_write"])
def test_optional_anchor_failure_keeps_numeric_credit_and_durable_receipt(
    runtime, monkeypatch, failure,
):
    r = runtime

    def fail(*args, **kwargs):
        raise RuntimeError("optional anchor failure")

    with monkeypatch.context() as fault:
        if failure == "encode":
            fault.setattr(feedback, "_anchor_embedder", lambda store: fail)
        else:
            fault.setattr(r.store, {
                "presence": "has_query_anchor", "anchor_write": "insert_query_anchor",
                "edge_write": "upsert_query_anchor_edge",
            }[failure], fail)
        result = attest(r, trace_id=r.trace.id)
    assert result["credited"] is True and result["closed"] is True
    assert result["anchor_ids"] == []
    assert r.store.get_node(r.node.id).usefulness_score == pytest.approx(0.1)
    before = durable_state(r.store)
    assert attest(r, trace_id=r.trace.id) == dict(result, replay=True)
    assert durable_state(r.store) == before


def test_existing_anchor_needs_no_encoding(runtime, monkeypatch):
    r = runtime
    existing = resolve_query_anchor(r.store, QUERY, SCOPE, VECTOR)

    def no_encoder(store):
        pytest.fail("an existing exact anchor must not invoke the encoder")

    monkeypatch.setattr(feedback, "_anchor_embedder", no_encoder)
    result = attest(r)
    assert result["anchor_ids"] == [existing.id]
    assert r.store.get_query_anchor(existing.id).reinforcement_count == 1


def test_native_preparation_releases_runtime_lock_and_sqlite_writer(runtime, monkeypatch):
    r = runtime
    entered, release = Event(), Event()

    def encode(queries):
        entered.set()
        assert release.wait(10), "anchor encoder was not released"
        return [VECTOR for _ in queries]

    monkeypatch.setattr(feedback, "_anchor_embedder", lambda store: encode)

    async def scenario():
        pending = asyncio.create_task(native(r, trace_id=r.trace.id))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert not r.store.connection.in_transaction
            async with Client(r.mcp) as peer:
                lookup = await asyncio.wait_for(peer.call_tool("memory_lookup", {"node_id": r.node.id}), 3)
                assert lookup.data["results"][0]["content"] == CONTENT
            with MemoryStore(r.store.db_path) as writer:
                writer.mark_recall_event_feedback(r.event.id, r.node.id)
            assert not pending.done()
        finally:
            release.set()
        result = await asyncio.wait_for(pending, 5)
        assert result["credited"] and result["closed"]
        assert result["closed_by_attestation"] is False
        assert result["feedback_trace_id"] == r.node.id
        assert r.store.get_recall_event(r.event.id).feedback_trace_id == r.node.id

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["content", "decay", "delete", "event", "trace", "scope", "ledger"])
def test_application_revalidates_server_owned_inputs_after_preparation(runtime, monkeypatch, change):
    r = runtime
    mutated = False
    new_scope = "project:current-attestation"

    def encode(queries):
        nonlocal mutated
        assert not r.store.connection.in_transaction
        if not mutated:
            mutated = True
            with MemoryStore(r.store.db_path) as writer:
                if change == "content":
                    writer.update_node(r.node.id, content="Violet gardens need regular watering.", embedding=VECTOR)
                elif change == "decay":
                    writer.soft_delete_node(r.node.id)
                elif change in {"delete", "event", "trace"}:
                    table, identity = ("recall_events", r.event.id) if change == "event" else (
                        "nodes", r.trace.id if change == "trace" else r.node.id,
                    )
                    with writer.transaction():
                        if table == "nodes":
                            writer.delete_node_chunks(identity)
                        else:
                            writer.connection.execute("DELETE FROM recall_delivery_history WHERE delivery_event_id = ?", (identity,))
                            writer.connection.execute("DELETE FROM recall_history_result_nodes WHERE recall_event_id = ?", (identity,))
                            writer.connection.execute("DELETE FROM recall_history_events WHERE recall_event_id = ?", (identity,))
                        writer.connection.execute(f"DELETE FROM {table} WHERE id = ?", (identity,))
                elif change == "scope":
                    writer.update_node(r.node.id, context={"scope": new_scope}, embedding=VECTOR)
                    with writer.transaction():
                        writer.connection.execute("UPDATE recall_events SET scope = ? WHERE id = ?",
                                                  (new_scope, r.event.id))
                else:
                    attest_recall_usage(writer, r.event.id, [CONTENT], trace_id=r.trace.id)
        return [VECTOR for _ in queries]

    monkeypatch.setattr(feedback, "_anchor_embedder", lambda store: encode)
    if change in {"event", "trace"}:
        with pytest.raises(AttestationError, match="unknown recall_event_id|resolves to no node"):
            attest(r, trace_id=r.trace.id)
        assert r.store.list_recall_attestations() == []
        assert r.store.get_node(r.node.id).usefulness_score == 0.0
        return
    result = attest(r, trace_id=r.trace.id)
    assert result["replay"] == (change == "ledger")
    assert result["credited"] == (change in {"scope", "ledger"})
    if change in {"decay", "delete"}:
        assert result["results"] == []
    if not result["credited"]:
        assert result["anchor_ids"] == []
        assert result["grounded_node_ids"] == []
    if change == "scope":
        assert result["scope"] == new_scope
        anchor, = r.store.list_query_anchors()
        assert anchor.scope == new_scope
        assert r.store.get_retrieval_weights(new_scope).updated_at


def test_authoritative_reads_follow_sqlite_writer_reservation(runtime, monkeypatch):
    r = runtime
    get_event = r.store.get_recall_event
    changed = False

    def read(event_id):
        nonlocal changed
        if r.store.connection.in_transaction and not changed:
            changed = True
            # This hook executes after preparation, inside the operation's
            # transaction. Credit must use this final source content.
            r.store.update_node(r.node.id, content="Violet gardens need regular watering.", embedding=VECTOR)
        return get_event(event_id)

    monkeypatch.setattr(r.store, "get_recall_event", read)
    result = attest(r)
    assert changed and result["credited"] is False
    assert result["grounded_node_ids"] == []
    assert r.store.get_node(r.node.id).usefulness_score == 0.0
