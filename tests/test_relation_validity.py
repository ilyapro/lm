"""Relation-only corrections retain validity at the real MCP boundary."""

import asyncio
import json

import pytest

pytest.importorskip("fastmcp")
from fastmcp import Client

from living_memory.server import create_mcp_server
from test_transport_identity import _structured


SCOPE = "project:relation-validity"
QUERY = "amber hatch pressure rule"
ORIGINAL = "Amber hatch pressure rule: open first.\n\n  Исходник — сохранить.\t\n"


@pytest.fixture(autouse=True)
def deterministic_retrieval(monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "off")
    monkeypatch.setenv("LM_RECALL_NEAR_DUP_COSINE", "0")


async def remember(client, content, scope=SCOPE):
    response = _structured(await client.call_tool("memory_remember", {
        "content": content, "context": {"scope": scope, "detail": "original context"},
    }))
    return response["node"]["id"]


async def read(client, path, node_id, *, scope=SCOPE):
    tool, view = path.split(":")
    arguments = {"scope": scope, "view": view}
    if tool == "recall":
        arguments.update(query=QUERY, depth=0, max_results=20)
    else:
        arguments["node_id"] = node_id
    response = _structured(await client.call_tool(f"memory_{tool}", arguments))
    return next(
        entry for entry in response["results"]
        if entry.get("node", entry)["id"] == node_id
    ), response


@pytest.mark.parametrize("path", [
    "recall:knowledge", "recall:full", "lookup:knowledge", "lookup:full",
])
@pytest.mark.parametrize("correction_state", ["active", "decayed", "foreign", "missing", "ineligible"])
def test_relation_only_correction_validity(tmp_path, monkeypatch, path, correction_state):
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store

    async def scenario():
        async with Client(mcp) as client:
            original = await remember(client, ORIGINAL)
            correction = await remember(
                client,
                f"Correction to {original}: amber hatch pressure rule requires release first. "
                "CORRECTION_BODY_SENTINEL",
                "project:foreign-validity" if correction_state == "foreign" else SCOPE,
            )
            edges = store.list_connections(
                source_id=correction, target_id=original, relation_type="supersedes",
            )
            assert len(edges) == 1 and edges[0].metadata["rule"] == "R1c"
            assert store.get_node(original).corrections == []
            if correction_state == "decayed":
                store.soft_delete_node(correction, "expired correction")
            elif correction_state == "ineligible":
                store.update_node(correction, context={"is_rejected_alternative": True})
            elif correction_state == "missing":
                # Model an absent counterpart while retaining its known relation.
                get_node_scopes = store.get_node_scopes
                monkeypatch.setattr(store, "get_node_scopes", lambda ids: {
                    key: scope for key, scope in get_node_scopes(ids).items() if key != correction
                })

            entry, response = await read(client, path, original)
            assert entry["superseded"] is True
            expected = [correction] if correction_state in ("active", "decayed", "ineligible") else []
            assert entry["superseded_by"] == expected
            body = entry.get("node", entry)
            assert body["provenance"]["corrections"] == []
            assert body["content"] == ORIGINAL
            if correction_state == "foreign":
                assert correction not in json.dumps(response)
                assert "CORRECTION_BODY_SENTINEL" not in json.dumps(response)
            if correction_state in ("decayed", "ineligible") and path.startswith("recall"):
                assert all(e.get("node", e)["id"] != correction for e in response["results"])
                event = store.get_recall_event(response["recall_event_id"])
                assert correction not in event.result_ids
                assert store.connection.execute("SELECT count(*) FROM recall_lookup_events").fetchone()[0] == 0
            if expected:
                fetched = _structured(await client.call_tool("memory_lookup", {
                    "node_id": expected[0], "scope": SCOPE,
                }))["results"][0]
                assert fetched["id"] == correction
                assert fetched["decayed"] is (correction_state == "decayed")
                assert fetched["superseded"] is False
            if correction_state == "ineligible" and path.startswith("recall"):
                marked = _structured(await client.call_tool("memory_recall", {
                    "query": QUERY, "scope": SCOPE, "depth": 0, "used": [correction],
                }))
                assert marked["feedback_marks"]["accepted"] == 0
                assert marked["feedback_marks"]["dropped"] == 1
                assert store.connection.execute(
                    "SELECT count(*) FROM recall_credit_ledger WHERE node_id = ?", (correction,),
                ).fetchone()[0] == 0
            assert store.get_node(original).content == ORIGINAL
            assert store.get_node(original).corrections == []

    try:
        asyncio.run(scenario())
    finally:
        store.close()


@pytest.mark.parametrize("path", [
    "recall:knowledge", "recall:full", "lookup:knowledge", "lookup:full",
])
@pytest.mark.parametrize("cycle", [False, True])
def test_direct_correction_links_preserve_chains_cycles_and_unaffected_nodes(tmp_path, path, cycle):
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store

    async def scenario():
        async with Client(mcp) as client:
            original = await remember(client, ORIGINAL)
            middle = await remember(client, f"Correction to {original}: amber hatch pressure rule B.")
            latest = await remember(client, f"Correction to {middle}: amber hatch pressure rule C.")
            untouched = await remember(client, "Amber hatch pressure rule: inspect the seal.")
            if cycle:
                store.create_connection(original, latest, "supersedes")
            expected = {original: [middle], middle: [latest], latest: [original] if cycle else [], untouched: []}
            for node_id, correction_ids in expected.items():
                entry, _ = await read(client, path, node_id)
                assert entry["superseded"] is bool(correction_ids)
                assert entry["superseded_by"] == correction_ids
                assert entry.get("node", entry)["provenance"]["corrections"] == []
            fetched = _structured(await client.call_tool("memory_lookup", {
                "node_id": original, "context_key": "detail", "scope": "project:other",
            }))
            assert fetched["results"][0]["content"] == ORIGINAL
            assert fetched["results"][0]["context"] == {"detail": "original context"}
            assert fetched["results"][0]["superseded"] is True
            assert fetched["results"][0]["superseded_by"] == []
            assert fetched["scope_mismatches"] == [{"id": original, "scope": SCOPE}]
            unscoped = _structured(await client.call_tool("memory_lookup", {"node_id": original}))
            assert unscoped["results"][0]["superseded_by"] == [middle]

    try:
        asyncio.run(scenario())
    finally:
        store.close()


@pytest.mark.parametrize('metadata', [
    {'kind': 'drain_near_duplicate', 'cosine': .9993, 'threshold': .99},
    {'by': 'lm_collapse_near_dups.py', 'reason': 'near-duplicate collapse', 'cosine': 1.0},
    {'kind': 'duplicate_content'},
    {'kind': 'duplicate_content', 'collapsed_by': 'lm_collapse_copies.py', 'basis': 'procedure_key'},
])
@pytest.mark.parametrize('path', ['recall:knowledge', 'recall:full', 'lookup:knowledge', 'lookup:full'])
def test_historical_bookkeeping_never_corrects_distinct_knowledge(tmp_path, metadata, path):
    from living_memory.decay import soft_delete_superseded
    from living_memory.replay import load_supersedes_pairs, snapshot_evidence
    from living_memory.query_anchors import resolve_replacement

    mcp = create_mcp_server(tmp_path / 'memory.sqlite3')
    store = mcp.memory_store

    async def scenario():
        async with Client(mcp) as client:
            permissive = await remember(client, 'Amber hatch pressure rule: opening is allowed before approval.')
            restrictive = await remember(client, 'Amber hatch pressure rule: opening is forbidden before approval.')
            edge = store.create_connection(permissive, restrictive, 'supersedes', metadata=metadata)
            # Retain the historical row without granting its semantic assertion.
            assert store.get_connection(edge.id).metadata == metadata
            assert store.list_supersedes_pairs([restrictive]) == []
            assert store.list_connections(node_id=restrictive) == []
            assert store.list_connections_for_nodes([restrictive])[restrictive] == []
            assert store.list_recall_connections([restrictive], limit=1)[restrictive] == []
            assert load_supersedes_pairs(store.connection) == []
            assert snapshot_evidence(store.connection)(SCOPE)[1] is False
            assert store._has_graph_evidence(SCOPE) is False
            assert resolve_replacement(store, restrictive) is None
            assert soft_delete_superseded(store, scope=SCOPE) == []
            entry, response = await read(client, path, restrictive)
            assert entry['superseded'] is False
            assert entry['superseded_by'] == []
            assert 'forbidden before approval' in entry.get('node', entry)['content']
            assert entry['delivery'] == 'full' if path.startswith('recall') else True
            if path.startswith('recall'):
                assert {e.get('node', e)['id'] for e in response['results']} == {permissive, restrictive}
            assert not store.get_node(restrictive).decayed

    try:
        asyncio.run(scenario())
    finally:
        store.close()


@pytest.mark.parametrize('metadata', [
    {}, {'kind': 'manual_correction'}, {'kind': 'unknown'},
    {'by': 'lm_collapse_near_dups.py'},
    {'by': 'another_producer', 'reason': 'near-duplicate collapse'},
])
def test_unrecognized_metadata_does_not_cancel_supersedes(tmp_path, metadata):
    from living_memory.storage import MemoryStore
    from living_memory.validity import supersedes_sets
    from living_memory.replay import load_supersedes_pairs
    from living_memory.decay import soft_delete_superseded

    with MemoryStore(tmp_path / 'memory.sqlite3') as store:
        old = store.append_trace('amber hatch allowed', {'scope': SCOPE})
        new = store.append_trace('amber hatch forbidden', {'scope': SCOPE})
        store.create_connection(new.id, old.id, 'supersedes', metadata=metadata)
        assert supersedes_sets(store, [old.id])[0] == {old.id: (new.id,)}
        assert load_supersedes_pairs(store.connection) == [(new.id, old.id)]
        assert [n.id for n in soft_delete_superseded(store, scope=SCOPE)] == [old.id]


@pytest.mark.parametrize('historical', [
    {'kind': 'drain_near_duplicate'},
    {'by': 'lm_collapse_near_dups.py', 'reason': 'near-duplicate collapse'},
])
@pytest.mark.parametrize('correction', [
    {'by': 'teacher', 'correction': 'Requires approval.'},
    {'rule': 'R1c', 'kind': 'content_correction'},
])
def test_real_correction_contribution_replaces_historical_heuristic_provenance(tmp_path, historical, correction):
    from living_memory.storage import MemoryStore
    with MemoryStore(tmp_path / 'memory.sqlite3') as store:
        old = store.append_trace('amber hatch allowed', {'scope': SCOPE})
        new = store.append_trace('amber hatch forbidden', {'scope': SCOPE})
        edge = store.create_connection(new.id, old.id, 'supersedes', metadata=historical)
        assert store.list_supersedes_pairs([old.id]) == []
        updated = store.create_connection(new.id, old.id, 'supersedes',
                                          metadata=correction, weight=.9, merge=True)
        assert updated.id == edge.id
        assert updated.metadata == correction
        assert updated.weight == .9
        assert store.list_supersedes_pairs([old.id]) == [(new.id, old.id)]


def test_metadata_less_snapshot_preserves_unknown_supersedes():
    import sqlite3
    from living_memory.replay import load_supersedes_pairs, snapshot_evidence

    with sqlite3.connect(':memory:') as connection:
        connection.row_factory = sqlite3.Row
        connection.executescript("""
            CREATE TABLE nodes (id TEXT, scope TEXT, decayed INTEGER);
            CREATE TABLE connections (source_id TEXT, target_id TEXT, type TEXT);
            INSERT INTO nodes VALUES ('new', 'global', 0), ('old', 'global', 0);
            INSERT INTO connections VALUES ('new', 'old', 'supersedes');
        """)
        assert load_supersedes_pairs(connection) == [('new', 'old')]
        assert snapshot_evidence(connection)('global') == (False, True)
