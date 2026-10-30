"""Graph candidates need current scope and liveness before their bodies are read."""

from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import count
import sqlite3
from threading import RLock

import pytest

from living_memory import query_anchors, server, storage
from living_memory.retrieval import MemoryRecallService
from living_memory.scope import ScopePlan, resolve_scope
from living_memory.storage import MemoryStore
from test_lookup import FakeMCP
from test_recall_graph_preparation import Encoder, anchor, node, recall


QUERY = "Calibration pear service deploy checklist"
SCOPE = "project:alpha"


@pytest.fixture
def native(tmp_path, monkeypatch):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
        "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic",
        "LM_AUTO_CONSOLIDATE_POLICY": "off",
        "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
        "LM_EXPLICIT_FEEDBACK_POLICY": "credit",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(storage, "_utc_now", lambda: "2026-10-28T12:00:00Z")
    moment = query_anchors._moment
    monkeypatch.setattr(query_anchors, "_moment", lambda value: (
        datetime(2026, 10, 28, 12, tzinfo=UTC) if value is None else moment(value)
    ))
    monkeypatch.setattr(server, "_transport_session_id", lambda: "graph-admission")
    guard = RLock()
    monkeypatch.setattr(server, "RLock", lambda: guard)
    services = []

    def service(store):
        result = MemoryRecallService(store)
        services.append(result)
        return result

    monkeypatch.setattr(server, "MemoryRecallService", service)

    @contextmanager
    def opened(name):
        tool_service_index = len(services)
        mcp = server.create_mcp_server(tmp_path / name, mcp_factory=FakeMCP)
        try:
            # Tools register their service before the independent prompt service.
            yield mcp, mcp.memory_store, services[tool_service_index], guard
        finally:
            mcp.auto_consolidation.close()
            mcp.memory_store.close()

    return opened


@pytest.mark.parametrize("excluded", ["scope", "decay"])
def test_native_excluded_neighbor_preserves_full_delivery_and_credit(native, tmp_path, monkeypatch, excluded):
    with native("seed.sqlite3") as (mcp, store, _, guard):
        alpha = mcp.tools["memory_remember"](QUERY, context={"scope": SCOPE})["node"]["id"]
        beta = mcp.tools["memory_remember"](
            "Frost granite unrelated source", context={"scope": "project:beta"},
        )["node"]["id"]
        with guard:
            store.create_connection(alpha, beta, "related")
            if excluded == "decay":
                store.update_node(beta, context={"scope": SCOPE})
                store.soft_delete_node(beta, "expired")
        prior = mcp.tools["memory_recall"](QUERY, scope=SCOPE, depth=0, max_results=3)
        assert [item["id"] for item in prior["results"]] == [alpha]
        with guard:
            for name in ("healthy.sqlite3", "corrupt.sqlite3"):
                with sqlite3.connect(tmp_path / name) as copy:
                    store.connection.backup(copy)

    outcomes = []
    for corrupt in (False, True):
        with native("corrupt.sqlite3" if corrupt else "healthy.sqlite3") as (mcp, store, service, guard):
            ids = count()
            monkeypatch.setattr(storage, "new_ulid", lambda: f"generated-{next(ids):06}")
            with guard, store.transaction():
                if corrupt:
                    store.connection.execute("UPDATE nodes SET provenance = '[]' WHERE id = ?", (beta,))
            unlocked = []
            store.connection.set_trace_callback(
                lambda sql: unlocked.append(sql) if not guard._is_owned() else None,
            )
            try:
                response = mcp.tools["memory_recall"](
                    QUERY, scope=SCOPE, depth=1, max_results=3, view="full", used=[alpha],
                )
            finally:
                store.connection.set_trace_callback(None)
            assert not unlocked
            with guard:
                event = store.get_recall_event(response["recall_event_id"])
                assert event.result_ids == [alpha]
                assert store.credited_node_ids(prior["recall_event_id"]) == {alpha}
                assert response["feedback_marks"] == {"accepted": 1, "dropped": 0, "by_reason": {}}
                assert store.get_node(alpha).access_count == 2
                ranked = service.last_query_relevance.ranked
                outcomes.append({
                    "response": response,
                    "event": asdict(event),
                    "ranked": [asdict(result) for result in ranked],
                    "qualification": [service.last_query_relevance.passes(result) for result in ranked],
                    "alpha": asdict(store.get_node(alpha)),
                    "marks": [tuple(row) for row in store.connection.execute("SELECT * FROM recall_feedback_marks")],
                    "credit": [tuple(row) for row in store.connection.execute("SELECT * FROM recall_explicit_credit")],
                    "weights": asdict(store.get_retrieval_weights(SCOPE)),
                    "anchors": [asdict(value) for value in store.list_query_anchors()],
                    "edges": [asdict(value) for value in store.list_query_anchor_edges()],
                })
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("selected", ["root", "neighbor"])
def test_native_selected_corruption_fails_without_event_access_or_credit(native, selected):
    with native("selected.sqlite3") as (mcp, store, _, guard):
        alpha = mcp.tools["memory_remember"](QUERY, context={"scope": SCOPE})["node"]["id"]
        beta = mcp.tools["memory_remember"](
            "Frost granite unrelated source", context={"scope": SCOPE},
        )["node"]["id"]
        with guard:
            store.create_connection(alpha, beta, "related")
        prior = mcp.tools["memory_recall"](QUERY, scope=SCOPE, depth=0, max_results=3)
        with guard:
            before = list(store.connection.iterdump())
            target = alpha if selected == "root" else beta
            provenance = store.connection.execute("SELECT provenance FROM nodes WHERE id = ?", (target,)).fetchone()[0]
            with store.transaction():
                store.connection.execute("UPDATE nodes SET provenance = '[]' WHERE id = ?", (target,))
        with pytest.raises(TypeError):
            mcp.tools["memory_recall"](
                QUERY, scope=SCOPE, depth=0 if selected == "root" else 1, max_results=3, used=[alpha],
            )
        with guard:
            assert store.credited_node_ids(prior["recall_event_id"]) == set()
            with store.transaction():
                store.connection.execute("UPDATE nodes SET provenance = ? WHERE id = ?", (provenance, target))
            assert list(store.connection.iterdump()) == before


@pytest.mark.parametrize("decayed", [False, True])
def test_native_exact_lookup_keeps_out_of_scope_and_decayed_bodies(native, decayed):
    with native("lookup.sqlite3") as (mcp, store, _, guard):
        beta = mcp.tools["memory_remember"](
            "Frost granite unrelated source", context={"scope": "project:beta"},
        )["node"]["id"]
        with guard:
            if decayed:
                store.soft_delete_node(beta, "expired")
        response = mcp.tools["memory_lookup"](node_id=beta, scope=SCOPE)
        assert response["results"][0]["id"] == beta
        assert response["results"][0]["decayed"] is decayed
        assert response["scope_mismatches"]
        with guard, store.transaction():
            store.connection.execute("UPDATE nodes SET provenance = '[]' WHERE id = ?", (beta,))
        assert mcp.tools["memory_lookup"](node_id=beta, scope=SCOPE) == response
        with pytest.raises(TypeError):
            mcp.tools["memory_lookup"](node_id=beta, scope=SCOPE, view="full")


@pytest.mark.parametrize("entry", ["graph", "anchor"])
@pytest.mark.parametrize("excluded", ["scope", "decay"])
def test_graph_and_anchor_admission_do_not_decode_excluded_bodies(tmp_path, monkeypatch, entry, excluded):
    with MemoryStore(tmp_path / "graph.sqlite3") as store:
        node(store, "root", content="needle")
        node(store, "leaf", scope="project:other" if excluded == "scope" else "project:graph")
        if entry == "graph":
            store.create_connection("root", "leaf", "related", metadata={"basis": "cluster"})
        else:
            anchor(store, "leaf")
        if excluded == "decay":
            store.soft_delete_node("leaf", "expired")
        service = MemoryRecallService(store, embedder=Encoder())
        expected = recall(service)
        decode = storage._node_from_row

        def admitted_only(row):
            assert row["id"] != "leaf"
            return decode(row)

        monkeypatch.setattr(storage, "_node_from_row", admitted_only)
        with store.transaction():
            store.connection.execute("UPDATE nodes SET provenance = '[]' WHERE id = 'leaf'")
        assert recall(service) == expected


@pytest.mark.parametrize("body", ["content", "context", "provenance", "access_count"])
def test_candidate_reader_filters_before_body_decoding_and_keeps_exact_reads(tmp_path, body):
    with MemoryStore(tmp_path / "reader.sqlite3") as store:
        alpha = node(store, "alpha", scope=SCOPE)
        node(store, "beta", scope="project:beta")
        dead = node(store, "dead", scope=SCOPE)
        dead = store.soft_delete_node(dead.id, "expired")
        plan = resolve_scope(scope=SCOPE)
        ids = [f"missing-{i}" for i in range(1100)] + ["alpha", "beta", "dead", "alpha"]
        assert store.get_recall_nodes(iter(ids), plan) == {"alpha": alpha}
        assert store.get_recall_nodes([], plan) == {}
        assert store.get_nodes(["dead"]) == {"dead": dead}
        invalid_text = "x'22ff22'" if body == "context" else "x'ff'"
        with store.transaction():
            for target in ("beta", "dead"):
                store.connection.execute(
                    f"UPDATE nodes SET {body} = CAST({invalid_text} AS TEXT) WHERE id = ?", (target,),
                )
        assert store.get_recall_nodes(ids, plan) == {"alpha": alpha}
        for target in ("beta", "dead"):
            with pytest.raises(sqlite3.OperationalError, match="Could not decode to UTF-8"):
                store.get_node(target)
            with pytest.raises(sqlite3.OperationalError, match="Could not decode to UTF-8"):
                store.get_nodes([target])
        with pytest.raises(sqlite3.OperationalError, match="Could not decode to UTF-8"):
            store.get_recall_nodes(ids, resolve_scope())


@pytest.mark.parametrize("plan", [
    resolve_scope(scope=SCOPE),
    resolve_scope(scope="session:review", ambient_context={"project": "alpha"}),
    resolve_scope(ambient_context={"project": "beta"}),
    ScopePlan("global", ()),
])
def test_candidate_reader_matches_live_scope_plan_and_refreshes_current_facts(tmp_path, plan):
    path = tmp_path / "scopes.sqlite3"
    with MemoryStore(path) as store, MemoryStore(path) as writer:
        scopes = ["global", SCOPE, "project:beta", "session:review"]
        for i, scope in enumerate(scopes):
            node(store, str(i), scope=scope)
        ids = [str(i) for i in range(len(scopes))]

        def expected():
            return {key: value for key, value in store.get_nodes(ids).items()
                    if not value.decayed and plan.allows(value.scope)}

        assert store.get_recall_nodes(ids, plan) == expected()
        writer.update_node("1", context={"scope": "project:beta"})
        writer.soft_delete_node("0", "expired")
        assert store.get_recall_nodes(ids, plan) == expected()
        with pytest.raises(RuntimeError, match="rollback"):
            with store.transaction():
                store.update_node("1", context={"scope": SCOPE})
                assert store.get_recall_nodes(ids, plan) == expected()
                raise RuntimeError("rollback")
        assert store.get_recall_nodes(ids, plan) == expected()
