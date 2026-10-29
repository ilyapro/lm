"""Prose is reference data; explicit correction relations own validity."""

import asyncio
import json
import sqlite3

import pytest

pytest.importorskip("fastmcp")
from fastmcp import Client

from living_memory import consolidation
from living_memory.decay import soft_delete_superseded
from living_memory.query_anchors import resolve_replacement, upsert_anchor
from living_memory.relations import (
    correction_sql, is_correction, is_semantic_relation, semantic_relation_sql,
    source_relation_weight, source_relation_weight_sql,
)
from living_memory.replay import load_supersedes_pairs
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore
from test_relation_validity import ORIGINAL, QUERY, SCOPE, read, remember
from test_transport_identity import _structured


INFERRED = {"rule": "R1c", "kind": "content_correction", "matched": "correction to"}


@pytest.fixture(autouse=True)
def deterministic_retrieval(monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "off")
    monkeypatch.setenv("LM_RECALL_NEAR_DUP_COSINE", "0")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")


@pytest.mark.parametrize("text,labels", [
    ("This is not a correction to {id}; the source remains valid.", {}),
    ('The example string "supersedes {id}" is quoted data.', {}),
    ("Correction: retry after 15 seconds. The separate source {id} remains valid.", {}),
    ("Source {id} remains valid.", {"type": "correction"}),
    ("Source {id} remains valid.", {"type": "memory_correction"}),
    ("Source {id} remains valid.", {"type": "decomposition_correction"}),
    ("Source {id} remains valid.", {"lesson_kind": "correction"}),
    ("Source {id} remains valid.", {}),
])
def test_native_remember_preserves_referenced_knowledge(tmp_path, text, labels):
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store

    async def scenario():
        async with Client(mcp) as client:
            original = await remember(client, ORIGINAL)
            before, _ = await read(client, "lookup:knowledge", original)
            source = store.get_node(original)
            note_text = text.format(id=original)
            result = _structured(await client.call_tool("memory_remember", {
                "content": note_text, "context": {"scope": SCOPE, **labels},
            }))
            note = result["node"]["id"]
            assert store.get_node(original) == source
            assert store.get_node(note).content == note_text
            assert store.connection.execute(
                "SELECT count(*) FROM connections WHERE type = 'supersedes'"
            ).fetchone()[0] == 0
            reference, = store.list_connections(source_id=note, target_id=original)
            assert reference.type == "related"
            assert reference.metadata == {"rule": "R6", "kind": "content_reference"}
            for path in ("lookup:knowledge", "lookup:full", "recall:knowledge", "recall:full"):
                entry, _ = await read(client, path, original)
                assert entry["superseded"] is before["superseded"] is False
                assert entry["superseded_by"] == []
                body = entry.get("node", entry)
                if path.startswith("lookup"):
                    assert body["content"] == ORIGINAL
                else:
                    assert entry["content_ref"]["node_id"] == original
                assert body["provenance"]["corrections"] == []

    try:
        asyncio.run(scenario())
    finally:
        mcp.auto_consolidation.close()
        store.close()


@pytest.mark.parametrize("writer", ["teach", "connect"])
def test_native_explicit_correction_survives_inferred_metadata(tmp_path, writer):
    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store

    async def scenario():
        async with Client(mcp) as client:
            original = await remember(client, ORIGINAL)
            independent = await remember(client, "Amber hatch seal: inspect before opening.")
            correction_text = f"Release pressure first. Source {independent} remains valid."
            if writer == "teach":
                result = _structured(await client.call_tool("memory_teach", {
                    "trace_id": original, "correction": correction_text, "context": {"scope": SCOPE},
                }))
                correction = result["corrective_trace"]["id"]
                edge, = store.list_connections(source_id=correction, relation_type="supersedes")
                # Reproduce the shallow merge made by the historical producer.
                mixed = store.create_connection(
                    correction, original, "supersedes", metadata=INFERRED, weight=.9, merge=True,
                )
                assert mixed.id == edge.id
                assert mixed.metadata == {**edge.metadata, **INFERRED}
                assert mixed.metadata["correction"] == correction_text
                assert store.get_node(original).corrections[-1]["new"] == correction_text
            else:
                correction = await remember(client, correction_text)
                historical = store.create_connection(
                    correction, original, "supersedes", metadata=INFERRED,
                )
                entry, _ = await read(client, "lookup:knowledge", original)
                assert entry["superseded"] is False
                await client.call_tool("memory_connect", {
                    "id_a": correction, "id_b": original, "relation_type": "supersedes",
                })
                assert store.get_connection(historical.id).metadata == {}
                assert store.get_node(original).corrections == []

            assert store.get_node(original).content == ORIGINAL
            for path in ("lookup:knowledge", "lookup:full", "recall:knowledge", "recall:full"):
                entry, _ = await read(client, path, original)
                assert entry["superseded"] is True
                assert entry["superseded_by"] == [correction]
                untouched, _ = await read(client, path, independent)
                assert untouched["superseded"] is False
            assert load_supersedes_pairs(store.connection) == [(correction, original)]
            assert resolve_replacement(store, original) == correction
            assert [node.id for node in soft_delete_superseded(store, scope=SCOPE)] == [original]
            assert not store.get_node(independent).decayed

    try:
        asyncio.run(scenario())
    finally:
        mcp.auto_consolidation.close()
        store.close()


@pytest.mark.parametrize("metadata,expected", [
    ({}, True),
    ({"kind": "unknown"}, True),
    (INFERRED, False),
    ({"rule": "R1c"}, False),
    ({"kind": "content_correction"}, False),
    ({**INFERRED, "correction": None}, False),
    ({**INFERRED, "correction": ""}, False),
    ({**INFERRED, "correction": True}, False),
    ({**INFERRED, "correction": 1}, False),
    ({**INFERRED, "correction": ["corrected"]}, False),
    ({**INFERRED, "correction": {"text": "corrected"}}, False),
    ({**INFERRED, "correction": "Release pressure first."}, True),
    ({"kind": "duplicate_content"}, False),
    ({"kind": "drain_near_duplicate"}, False),
    ({"by": "lm_collapse_near_dups.py", "reason": "near-duplicate collapse"}, False),
    ({"kind": "duplicate_content", "correction": "Release pressure first."}, True),
])
@pytest.mark.parametrize("relation", ["supersedes", "related"])
def test_python_sql_share_correction_and_graph_semantics(metadata, expected, relation):
    expected_correction = relation == "supersedes" and expected
    expected_semantic = relation != "supersedes" or expected
    assert is_correction(relation, metadata) is expected_correction
    assert is_semantic_relation(relation, metadata) is expected_semantic
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE connections (type TEXT, metadata TEXT, weight REAL)")
        db.execute("INSERT INTO connections VALUES (?, ?, .9)", (relation, json.dumps(metadata)))
        for alias in ("", "c"):
            result = db.execute(
                f"SELECT {correction_sql(alias)}, {semantic_relation_sql(alias)}, "
                f"{source_relation_weight_sql(alias)} FROM connections AS c"
            ).fetchone()
            assert result == (
                int(expected_correction), int(expected_semantic),
                source_relation_weight(relation, .9, metadata),
            )


@pytest.mark.parametrize("explicit", [False, True])
def test_consolidation_uses_explicit_replacement_only(tmp_path, explicit):
    from test_derivative_preparation import cluster, sources

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        members = sources(store)
        owner = cluster(store, members)
        old = members[0]
        new = store.append_trace("Harbor crane requires the electronic permit.", {"scope": old.scope})
        metadata = {**INFERRED, **({"correction": new.content} if explicit else {})}
        edge = store.create_connection(new.id, old.id, "supersedes", metadata=metadata)
        expected_id = new.id if explicit else old.id
        for observation in (None, consolidation._SourceObservation(store, owner.source_traces)):
            current = consolidation._current_derivation(store, owner, observation=observation)
            assert {node.id for node in current.live} == {expected_id, members[1].id, members[2].id}
        selection = consolidation._current_sources(store, [*(node.id for node in members), new.id])
        assert (old.id in selection.excluded) is explicit
        assert consolidation._is_superseded(store, old.id) is explicit
        assert store.get_connection(edge.id).metadata == metadata


def test_history_filter_does_not_restore_persisted_lifecycle_effects(tmp_path):
    db = tmp_path / "memory.sqlite3"
    with MemoryStore(db) as store:
        old = store.append_trace("Amber hatch requires the original gauge.")
        note = store.append_trace("Quoted gauge example.")
        anchor = upsert_anchor(store, QUERY, "global", [1., 0.], [old.id]).anchor
        edge = store.create_connection(note.id, old.id, "supersedes", metadata=INFERRED)
        # Model already-persisted effects; the relation filter cannot infer their cause.
        with store.transaction():
            store.repoint_query_anchor_edges(old.id, note.id)
            store.soft_delete_node(old.id, "superseded")
        before = store.get_node(old.id)
    with MemoryStore(db) as store:
        assert store.get_node(old.id) == before
        assert store.get_connection(edge.id).metadata == INFERRED
        assert store.list_supersedes_pairs([old.id]) == []
        assert [edge.target_id for edge in store.list_query_anchor_edges(anchor_id=anchor.id)] == [note.id]
