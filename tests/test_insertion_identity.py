"""Insertion reuses source identity while preserving every peer validation."""

import asyncio
from collections import Counter
import sqlite3
from unittest.mock import patch

from fastmcp import Client
import pytest

from living_memory import server, storage


CONTENT = "Same fact with independent recorded source."
CONTEXT = {"scope": "project:insertion-test", "transport_session_id": "fixture-transport"}
FINGERPRINT = "f2d38a30070508373d458123440dae4fff53f531b8db517b36f77a35674b8593"


@pytest.fixture(params=["direct", "native"])
def writer(request, tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setattr(server, "_decay_sweep_if_due", lambda store: None)
    mcp = server.create_mcp_server(tmp_path / "memory.sqlite3") if request.param == "native" else None
    store = mcp.memory_store if mcp else storage.MemoryStore(tmp_path / "memory.sqlite3")

    def write(context):
        if mcp is None:
            return store.append_trace(CONTENT, context).id

        async def invoke():
            async with Client(mcp) as client:
                result = await client.call_tool("memory_remember", {
                    "content": CONTENT, "context": context,
                }, raise_on_error=False)
                if result.is_error:
                    raise RuntimeError(result.content[0].text)
                return result.structured_content["node"]["id"]

        return asyncio.run(invoke())

    try:
        yield store, write
    finally:
        if mcp:
            mcp.auto_consolidation.close()
        store.close()


def assert_outcome(store, target, peers, twins=()):
    rows = store.connection.execute("SELECT id, decayed, decay_reason FROM nodes").fetchall()
    assert {row["id"] for row in rows} == {target, *peers}
    assert {row["id"] for row in rows if row["decayed"]} == set(twins)
    assert {row["decay_reason"] for row in rows if row["decayed"]} == (
        {"duplicate_content"} if twins else set()
    )
    assert [(edge.source_id, edge.target_id, edge.type, edge.weight, edge.metadata)
            for edge in store.list_connections()] == [
        (target, twin, "related", 1.0, {"kind": "duplicate_content", "fingerprint": FINGERPRINT})
        for twin in twins
    ]
    assert not store.connection.in_transaction


@pytest.mark.parametrize("peer_count, twin_slot", [(0, None), (1, None), (16, None), (16, 7)])
def test_candidate_identity_is_computed_once_per_nonempty_peer_set(writer, peer_count, twin_slot):
    store, write = writer
    context = {**CONTEXT, "procedure": "p" * 131072}
    peers = [store.append_trace(CONTENT, {**context, "source_slot": i}).id
             for i in range(peer_count)]
    calls = Counter()
    identity = storage.content_identity

    def measured(node):
        calls[node.id] += 1
        return identity(node)

    with patch.object(storage, "content_identity", measured):
        target = write({**context, "source_slot": twin_slot if twin_slot is not None else 999})
    twins = [peers[twin_slot]] if twin_slot is not None else []
    # Each duplicate edge independently validates both stored endpoints.
    expected = Counter({peer: 1 for peer in peers})
    if peer_count:
        expected[target] = 1
    if twins:
        expected.update([target, *twins])
    assert calls == expected
    assert_outcome(store, target, peers, twins)


@pytest.mark.parametrize("fault", ["malformed_peer", "duplicate_edge"])
def test_selected_peer_and_duplicate_edge_failures_roll_back(writer, fault):
    store, write = writer
    peer = store.append_trace(CONTENT, CONTEXT).id
    if fault == "malformed_peer":
        with store.transaction():
            store.connection.execute("UPDATE nodes SET provenance = '[]' WHERE id = ?", (peer,))
        error = "list indices"
    else:
        store.connection.execute("""CREATE TRIGGER reject_duplicate AFTER INSERT ON connections
            WHEN json_extract(NEW.metadata, '$.kind') = 'duplicate_content'
            BEGIN SELECT RAISE(ABORT, 'duplicate edge fault'); END""")
        error = "duplicate edge fault"
    before = list(store.connection.iterdump())
    with pytest.raises((TypeError, sqlite3.IntegrityError, RuntimeError), match=error):
        write(CONTEXT)
    assert list(store.connection.iterdump()) == before
    assert not store.connection.in_transaction
    if fault == "malformed_peer":
        with store.transaction():
            store.connection.execute("UPDATE nodes SET provenance = '{}' WHERE id = ?", (peer,))
    else:
        store.connection.execute("DROP TRIGGER reject_duplicate")
    target = write(CONTEXT)
    assert_outcome(store, target, [peer], [peer])


@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("broken_index", [0, 1, None])
def test_peer_decode_precedes_lazy_identity_even_for_unknown_derivations(tmp_path, unknown, broken_index):
    with storage.MemoryStore(tmp_path / "memory.sqlite3") as store:
        provenance = {"strategy": "procedural"} if unknown else {}
        peers = [store.create_node(level="schema", content=CONTENT,
                                   context={**CONTEXT, "source_slot": i}, provenance=provenance).id
                 for i in range(3)]
        if broken_index is not None:
            with store.transaction():
                store.connection.execute("UPDATE nodes SET provenance = '[]' WHERE id = ?", (peers[broken_index],))
        before = list(store.connection.iterdump())
        events = []
        decode, identity = storage._node_from_row, storage.content_identity

        def decoded(row):
            events.append(("decode", row["id"]))
            return decode(row)

        def identified(node):
            events.append(("identity", node.id))
            return identity(node)

        def insert():
            return store.create_node(level="schema", content=CONTENT, node_id="candidate",
                                     context={**CONTEXT, "source_slot": 999}, provenance=provenance)

        with patch.object(storage, "_node_from_row", decoded), patch.object(storage, "content_identity", identified):
            if broken_index is None:
                insert()
            else:
                with pytest.raises(TypeError, match="list indices"):
                    insert()
        expected = []
        for i, peer in enumerate(peers):
            expected.append(("decode", peer))
            if i == broken_index:
                break
            if i == 0:
                expected.append(("identity", "candidate"))
            if not unknown:
                expected.append(("identity", peer))
        if broken_index is None:
            expected.append(("decode", "candidate"))  # create_node returns the stored node.
            assert_outcome(store, "candidate", peers)
        else:
            assert list(store.connection.iterdump()) == before
        assert events == expected


def test_peer_identity_failure_is_not_skipped_after_first_peer(tmp_path):
    with storage.MemoryStore(tmp_path / "memory.sqlite3") as store:
        peers = [store.append_trace(CONTENT, {**CONTEXT, "source_slot": i}).id for i in range(2)]
        # Valid JSON decodes, but is not a source context mapping.
        with store.transaction():
            store.connection.execute("UPDATE nodes SET context = '[]' WHERE id = ?", (peers[1],))
        before = list(store.connection.iterdump())
        with pytest.raises(AttributeError, match="items"):
            store.append_trace(CONTENT, {**CONTEXT, "source_slot": 999})
        assert list(store.connection.iterdump()) == before


@pytest.mark.parametrize("distinction", [
    "context", "provenance", "sources", "corrections", "scope", "level", "bookkeeping",
    "known_derivation", "unknown_derivation", "derivation_sources", "derivation_group", "materializer",
])
def test_insertion_keeps_full_source_identity(tmp_path, distinction):
    first = {"level": "concept", "content": CONTENT, "context": dict(CONTEXT), "provenance": {}}
    second = {**first, "context": dict(CONTEXT), "provenance": {}}
    if distinction in ("known_derivation", "unknown_derivation", "derivation_sources", "derivation_group", "materializer"):
        for args in (first, second):
            args["provenance"] = {"strategy": "procedural", "source_traces": ["source-a"]}
        if distinction == "unknown_derivation":
            first["provenance"].pop("source_traces")
            second["provenance"].pop("source_traces")
        elif distinction == "derivation_sources":
            second["provenance"]["source_traces"] = ["source-b"]
        elif distinction == "derivation_group":
            second["context"]["procedure_key"] = "other group"
        elif distinction == "materializer":
            second["provenance"]["strategy"] = "token-jaccard"
    elif distinction == "context":
        second["context"]["agent"] = "independent"
    elif distinction == "provenance":
        second["provenance"]["origin"] = "independent"
    elif distinction == "sources":
        second["provenance"]["source_traces"] = ["source-b"]
    elif distinction == "corrections":
        second["provenance"]["corrections"] = [{"correction": "replacement"}]
    elif distinction == "scope":
        second["context"]["scope"] = "project:other"
    elif distinction == "level":
        second["level"] = "schema"
    else:
        second["context"]["timestamp"] = "2026-01-01T00:00:00Z"
        second["provenance"]["prior_recalls"] = [{"id": "recall-event"}]
    with storage.MemoryStore(tmp_path / "memory.sqlite3") as store:
        peer = store.create_node(**first).id
        target = store.create_node(**second).id
        twins = [peer] if distinction in ("bookkeeping", "known_derivation") else []
        assert_outcome(store, target, [peer], twins)
