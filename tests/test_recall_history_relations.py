"""Recall history is observable without becoming evidence between topics."""

import asyncio

import pytest

pytest.importorskip("fastmcp")
from fastmcp import Client

import living_memory.server as server
from living_memory.edge_derivation import derive_edges_for_new_trace
from living_memory.relations import is_semantic_relation, semantic_relation_sql, source_relation_weight
from living_memory.replay import snapshot_evidence
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore
from test_transport_identity import _structured


SCOPE = "project:topic-history"
CONTEXT = {"scope": SCOPE, "session_id": "one-work", "task": "two-topics"}
ORCHID = "Orchid pollination requires moth wing contact before sunrise."
DEPLOYMENT = "Cobalt deployment accepts unverified package transfer."
CORRECTION = "Cobalt deployment requires verified package transfer before publication."
QUERY = "Orchid pollination moth wing"
IMPLICIT = {"basis": "implicit_recall_feedback", "recall_event_id": "old-event", "rank": 1}


class ZeroSemantic:
    def embed(self, text):
        return [0.0] * 384


@pytest.mark.parametrize("tool", ["memory_remember", "memory_teach"])
@pytest.mark.parametrize("retired_policy", ["all", "credited"])
def test_two_topics_keep_credit_and_history_without_graph_leakage(
    tmp_path, monkeypatch, tool, retired_policy,
):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", "grounded")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    # An installed, obsolete setting must not restore the removed writer.
    monkeypatch.setenv("LM_IMPLICIT_LINK_POLICY", retired_policy)
    services = []

    def service(store):
        value = MemoryRecallService(store)
        services.append(value)
        return value

    monkeypatch.setattr(server, "MemoryRecallService", service)
    mcp = server.create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    orchid = store.append_trace(ORCHID, {"scope": SCOPE})
    deployment = store.append_trace(DEPLOYMENT, {"scope": SCOPE})

    async def scenario():
        async with Client(mcp) as client:
            async def call(name, **args):
                response = await client.call_tool(name, args)
                assert not response.is_error
                return _structured(response)

            first = await call("memory_recall", query=QUERY, scope=SCOPE,
                               ambient_context=CONTEXT, max_results=1, depth=0)
            assert [r["id"] for r in first["results"]] == [orchid.id]
            second = await call("memory_recall", query="Cobalt deployment package transfer",
                                scope=SCOPE, ambient_context=CONTEXT, max_results=1,
                                depth=0, used=[orchid.id])
            assert [r["id"] for r in second["results"]] == [deployment.id]
            arguments = {"context": CONTEXT, "used": [deployment.id]}
            if tool == "memory_teach":
                arguments.update(trace_id=deployment.id, correction=CORRECTION)
            else:
                arguments["content"] = CORRECTION
            written = await call(tool, **arguments)
            new_id = written["corrective_trace" if tool == "memory_teach" else "node"]["id"]
            assert written["feedback_marks"]["accepted"] == 1
            assert set(written["implicit_feedback"]["recall_event_ids"]) == {
                first["recall_event_id"], second["recall_event_id"],
            }
            assert store.connection.execute(
                "SELECT count(*) FROM connections WHERE source_id = ? AND type = 'related'",
                (new_id,),
            ).fetchone()[0] == 0
            fetched = await call("memory_lookup", node_id=new_id, view="full")
            provenance = fetched["results"][0]["provenance"]
            assert set(provenance["recalled_nodes"]) == {orchid.id, deployment.id}
            assert {p["query"] for p in provenance["prior_recalls"]} == {
                QUERY, "Cobalt deployment package transfer",
            }
            for event_id, node_id in [(first["recall_event_id"], orchid.id),
                                      (second["recall_event_id"], deployment.id)]:
                assert store.get_recall_event(event_id).feedback_trace_id == new_id
                assert store.credited_node_ids(event_id) == {node_id}
                before = store.get_node(node_id).usefulness_score
                await call("memory_lookup", node_id=node_id)
                assert store.get_node(node_id).usefulness_score == before
            assert store.connection.execute("SELECT count(*) FROM recall_explicit_credit").fetchone()[0] == 2
            assert store.connection.execute("SELECT count(*) FROM recall_credit_ledger").fetchone()[0] == 0

            # Use the actual MCP recall handler with semantic/anchor evidence
            # disabled, so any cross-topic result can only come from graph.
            services[0].embedder = ZeroSemantic()
            services[0].anchor_seeding = False

            async def recall(query=QUERY, depth=1):
                response = await call("memory_recall", query=query, scope=SCOPE,
                                      depth=depth, max_results=10, view="full")
                return {r["node"]["id"]: r for r in response["results"]}

            assert set(await recall()) == {orchid.id}
            historical = store.create_connection(new_id, orchid.id, "related", weight=.75, metadata=IMPLICIT)
            assert set(await recall()) == {orchid.id}
            assert set(await recall(depth="causal")) == {orchid.id}
            assert store.get_connection(historical.id).metadata == IMPLICIT
            if tool == "memory_teach":
                correction_results = await recall("Cobalt deployment package transfer")
                assert new_id in correction_results
                original = await call("memory_lookup", node_id=deployment.id)
                assert original["results"][0]["superseded"]
                assert original["results"][0]["superseded_by"] == [new_id]

            # The same pair becomes traversable when explicitly connected.
            await call("memory_connect", id_a=new_id, id_b=orchid.id,
                       relation_type="related", weight=.75, metadata={"basis": "explicit"})
            linked = (await recall())[new_id]
            assert linked["bm25_score"] == linked["vector_score"] == 0
            assert linked["graph_score"] == pytest.approx(.4875)

    try:
        asyncio.run(scenario())
    finally:
        mcp.auto_consolidation.close()
        store.close()


@pytest.mark.parametrize("relation_type,metadata,semantic", [
    ("related", IMPLICIT, False),
    ("related", {**IMPLICIT, "kind": "unknown"}, False),
    ("related", {**IMPLICIT, "kind": "content_reference", "rule": "R6"}, True),
    ("related", {**IMPLICIT, "kind": "derived_from", "rule": "R5a"}, True),
    ("related", {**IMPLICIT, "basis": "shared_files"}, True),
    ("related", {}, True),
    ("requires", IMPLICIT, True),
    ("caused", IMPLICIT, True),
    ("supersedes", IMPLICIT, True),
    ("supersedes", {"kind": "duplicate_content"}, False),
])
def test_semantics_agree_across_readers_and_keep_raw_history(tmp_path, relation_type, metadata, semantic):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        a = store.append_trace(ORCHID, {"scope": SCOPE})
        b = store.append_trace(DEPLOYMENT, {"scope": SCOPE})
        edge = store.create_connection(a.id, b.id, relation_type, metadata=metadata)
        assert is_semantic_relation(relation_type, metadata) is semantic
        assert bool(store.connection.execute(
            f"SELECT 1 FROM connections c WHERE {semantic_relation_sql('c')}"
        ).fetchone()) is semantic
        assert store.get_connection(edge.id).metadata == metadata
        assert bool(store.list_connections(node_id=a.id)) is semantic
        assert bool(store.list_connections_for_nodes([a.id])[a.id]) is semantic
        assert bool(store.list_recall_connections([a.id], limit=1)[a.id]) is semantic
        assert store._has_graph_evidence(SCOPE) is semantic
        assert snapshot_evidence(store.connection)(SCOPE)[1] is semantic


def test_historical_edges_do_not_spend_adjacency_limit(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        a, b, c = [store.append_trace(text, {"scope": SCOPE}) for text in (ORCHID, DEPLOYMENT, CORRECTION)]
        history = store.create_connection(a.id, b.id, "related", weight=1, metadata=IMPLICIT)
        genuine = store.create_connection(a.id, c.id, "requires", weight=.5)
        assert store.list_recall_connections([a.id], limit=1)[a.id] == [genuine]
        assert store.get_connection(history.id) == history


@pytest.mark.parametrize("relation_type,metadata,expected", [
    ("related", {}, 1),
    ("related", {"basis": "unknown", "kind": "unknown"}, 1),
    ("related", {"basis": "co_access"}, 0),
    ("related", {"basis": "cluster", "co_access": 1000}, 0.2),
    ("related", {**IMPLICIT, "kind": "content_reference"}, 0.9),
    ("related", {**IMPLICIT, "kind": "derived_from"}, 0.6),
    ("related", {"rule": "R4a", "basis": "same_commit", "left": 1000, "right": 1000}, 0.9),
    ("related", {"rule": "R4b", "basis": "shared_files", "files": ["a.py"]}, 0.65),
    ("requires", {"basis": "co_access"}, 1),
    ("supersedes", {"basis": "co_access"}, 1),
    ("supersedes", {"kind": "duplicate_content"}, 0),
])
def test_source_contribution_keeps_independent_and_unknown_relations(relation_type, metadata, expected):
    assert source_relation_weight(relation_type, 1, metadata) == pytest.approx(expected)
    assert source_relation_weight(relation_type, 0.1, metadata) == pytest.approx(min(0.1, expected))


def test_content_reference_replaces_technical_history_in_shared_writer(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        a = store.append_trace(ORCHID, {"scope": SCOPE})
        b = store.append_trace(f"Pollination measurement uses the method in {a.id}.", {"scope": SCOPE})
        store.create_connection(b.id, a.id, "related", weight=.99, metadata=IMPLICIT)
        derive_edges_for_new_trace(store, b)
        edge, = store.list_connections(source_id=b.id, target_id=a.id)
        assert edge.weight == .9
        assert edge.metadata == {"rule": "R6", "kind": "content_reference"}


@pytest.mark.parametrize("evidence", ["reference", "shared_files"])
def test_backfill_adds_semantic_evidence_to_technical_pair_only(tmp_path, evidence):
    from living_memory.edge_backfill import backfill_edges

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        context = {"scope": SCOPE, "files": ["pollination.py"]} if evidence == "shared_files" else {"scope": SCOPE}
        a = store.append_trace(ORCHID, context)
        text = f"Pollination uses {a.id}." if evidence == "reference" else "Moth wing measurement."
        b = store.append_trace(text, context)
        source, target = (b.id, a.id) if evidence == "reference" else tuple(sorted((a.id, b.id)))
        historical = store.create_connection(source, target, "related", weight=.99, metadata=IMPLICIT)
        plan = backfill_edges(store)
        assert plan["totals"]["semantic_updates"] == 1
        assert store.get_connection(historical.id) == historical
        applied = backfill_edges(store, apply=True)
        edge, = store.list_connections(source_id=source, target_id=target)
        assert edge.weight == (.9 if evidence == "reference" else .65)
        assert edge.metadata.get("basis") == (None if evidence == "reference" else "shared_files")
        assert "recall_event_id" not in edge.metadata
        assert backfill_edges(store, apply=True)["totals"]["semantic_updates"] == 0
        assert store.get_connection(historical.id) == edge
        store.connection.execute(applied["rollback"])
        assert store.get_connection(historical.id) == edge
