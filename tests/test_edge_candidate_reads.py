"""Bounded provenance readers retain rule semantics without full node bodies."""

import asyncio
import json
import sqlite3
import threading

from fastmcp import Client
import pytest

from living_memory import edge_derivation as ed, server, storage


def tuples(edges):
    return sorted((e.source_id, e.target_id, e.type, e.weight,
                   json.dumps(e.metadata, sort_keys=True)) for e in edges)


def expected(a, b, weight, metadata):
    return (*sorted((a.id, b.id)), "related", weight, json.dumps(metadata, sort_keys=True))


@pytest.mark.parametrize("kind,peers,duplicates,edge_count", [
    ("commit", 9, 1, 9), ("commit", 10, 1, 0),
    ("file", 19, 1, 19), ("file", 20, 1, 0),
    ("file", 18, 2, 18), ("file", 19, 2, 0),
])
def test_caps_include_source_and_duplicate_occurrences(
    tmp_path, kind, peers, duplicates, edge_count,
):
    with storage.MemoryStore(tmp_path / "memory.sqlite3") as store:
        context = {"commit": "abc1234ff"} if kind == "commit" else {"files": ["shared"]}
        targets = [store.create_node(level=("trace", "concept", "schema")[i % 3],
                                     content=f"Peer {i}", context=context) for i in range(peers)]
        stale = store.append_trace("Retired peer", context)
        store.soft_delete_node(stale.id)
        note = store.append_trace("Source", context if kind == "commit" else {"files": ["shared"] * duplicates})
        metadata = ({"rule": "R4a", "basis": "same_commit", "commit": "abc1234"}
                    if kind == "commit" else {"rule": "R4b", "basis": "shared_files", "files": ["shared"]})
        oracle = [expected(note, peer, 0.9 if kind == "commit" else 0.65, metadata)
                  for peer in targets[:edge_count]]
        with pytest.raises(RuntimeError, match="rollback"):
            with store.transaction():
                assert tuples(ed.derive_edges_for_new_trace(store, note)) == sorted(oracle)
                raise RuntimeError("rollback")
        assert store.list_connections() == []


@pytest.mark.parametrize("kind", ["commit", "file"])
def test_json_truthiness_and_multiple_shared_files(tmp_path, kind):
    with storage.MemoryStore(tmp_path / "memory.sqlite3") as store:
        context = {"commit": " ABC1234abcdef "} if kind == "commit" else {"files": ["one", "one", "two"]}
        flags = [True, False, [], {}, "false", 0, None, 1]
        peers = [store.append_trace(f"Peer {i}", {**context, "is_rejected_alternative": flag})
                 for i, flag in enumerate(flags)]
        note = store.append_trace("Source", {**context, "is_rejected_alternative": True})
        metadata = ({"rule": "R4a", "basis": "same_commit", "commit": "abc1234"}
                    if kind == "commit" else {"rule": "R4b", "basis": "shared_files", "files": ["one", "two"]})
        oracle = [expected(note, peers[i], 0.9 if kind == "commit" else 0.8, metadata)
                  for i in [1, 2, 3, 5, 6]]
        assert tuples(ed.derive_edges_for_new_trace(store, note)) == sorted(oracle)


@pytest.mark.parametrize("value,prefix,selected,linked", [
    (" ABC1234ff ", "abc1234", True, True),
    ("\tabc1234ff\t", "abc1234", False, False),
    (123456789, "1234567", True, True),
    (1234567.5, "1234567", True, True),
    (123456789012345678901234567890, "1.23456", True, False),
    (["abcdefg"], '["abcde', True, False),
    ({"abcdefg": 1}, '{"abcde', True, False),
    ([], "abcdefg", False, False), ({}, "abcdefg", False, False),
    (False, "abcdefg", False, False), (None, "abcdefg", False, False),
])
def test_commit_sql_membership_and_python_normalization(tmp_path, value, prefix, selected, linked):
    with storage.MemoryStore(tmp_path / "memory.sqlite3") as store:
        peer = store.append_trace("Peer", {"commit": value})
        note = store.append_trace("Source", {"commit": prefix})
        candidates = ed._commit_group_candidates(store, note, prefix)
        assert [c.id for c in candidates] == ([peer.id] if selected else [])
        if selected:
            assert candidates[0].context["commit"] == value
            assert type(candidates[0].context["commit"]) is type(value)
        oracle = [expected(note, peer, 0.9, {"rule": "R4a", "basis": "same_commit", "commit": prefix})] if linked else []
        assert tuples(ed.derive_edges_for_new_trace(store, note)) == oracle


@pytest.mark.parametrize("kind", ["commit", "file"])
@pytest.mark.parametrize("denied", ["content", "provenance", "id", "decayed", "context", "SQL"])
def test_required_reads_fail_and_unused_bodies_can_be_denied(tmp_path, monkeypatch, kind, denied):
    with storage.MemoryStore(tmp_path / "memory.sqlite3") as store:
        context = {"commit": "abc1234ff"} if kind == "commit" else {"files": ["shared"]}
        peer = store.create_node(level="trace", content="b" * 131072,
                                 context={**context, "procedure": "p" * 131072},
                                 provenance={"unrelated": "h" * 65536})
        note = store.append_trace("Source", context)
        # Unrelated malformed provenance must never reach a full node decoder.
        store.connection.execute("UPDATE nodes SET provenance = '{' WHERE id = ?", (peer.id,))
        reads = set()

        def authorize(action, table, column, *_args):
            if action == sqlite3.SQLITE_SELECT and denied == "SQL":
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_READ and table == "nodes":
                reads.add(column)
                if column == denied:
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        def unexpected(_row):
            pytest.fail("candidate materialized a full Node")

        monkeypatch.setattr(storage, "_node_from_row", unexpected)
        store.connection.set_authorizer(authorize)
        try:
            if denied in {"content", "provenance"}:
                assert len(ed.derive_edges_for_new_trace(store, note)) == 1
                assert reads == {"id", "decayed", "context"}
            else:
                with pytest.raises(sqlite3.DatabaseError):
                    ed.derive_edges_for_new_trace(store, note)
        finally:
            store.connection.set_authorizer(None)
        if denied not in {"content", "provenance"}:
            assert store.list_connections() == []


@pytest.mark.parametrize("kind", ["commit", "file"])
def test_malformed_required_context_is_not_a_missing_candidate(tmp_path, kind):
    with storage.MemoryStore(tmp_path / "memory.sqlite3") as store:
        context = {"commit": "abc1234ff"} if kind == "commit" else {"files": ["shared"]}
        peer = store.append_trace("Peer", context)
        note = store.append_trace("Source", context)
        # Corruption fixture: expression indexes otherwise reject this UPDATE.
        store.connection.execute("DROP INDEX idx_nodes_commit_prefix")
        store.connection.execute("DROP INDEX idx_nodes_files_present")
        store.connection.execute("UPDATE nodes SET context = '{' WHERE id = ?", (peer.id,))
        with pytest.raises(sqlite3.OperationalError, match="malformed JSON"):
            ed.derive_edges_for_new_trace(store, note)
        assert store.list_connections() == []


@pytest.mark.parametrize("field", ["commit", "is_rejected_alternative"])
def test_malformed_selected_json_is_not_swallowed(tmp_path, field):
    with storage.MemoryStore(tmp_path / "memory.sqlite3") as store:
        row = store.connection.execute(f'SELECT \'peer\' AS id, 0 AS decayed, \'{{\' AS "{field}"').fetchone()
        with pytest.raises(json.JSONDecodeError):
            ed._candidate_from_row(row)


def test_rules_merge_weights_metadata_and_remain_idempotent(tmp_path):
    with storage.MemoryStore(tmp_path / "memory.sqlite3") as store:
        context = {"commit": "abc1234ff", "files": ["one", "two"]}
        peer = store.append_trace("Peer", context)
        note = store.append_trace("Source", context)
        store.create_connection(*sorted((peer.id, note.id)), "related", weight=0.94, metadata={"kept": True})
        ed.derive_edges_for_new_trace(store, note)
        first = store.list_connections()
        assert tuples(first) == [expected(peer, note, 0.94, {
            "kept": True, "rule": "R4b", "basis": "shared_files", "commit": "abc1234", "files": ["one", "two"],
        })]
        ed.derive_edges_for_new_trace(store, note)
        assert tuples(store.list_connections()) == tuples(first)
        # The final contribution by itself is an unchanged upsert.
        edge = ed.rule_r4b_shared_files(note, {f: [peer, note] for f in ["one", "two"]})[0]
        before = store.list_connections()[0]
        assert ed.upsert_derived_edge(store, edge) == before


@pytest.mark.parametrize("kind", ["commit", "file"])
@pytest.mark.parametrize("fault", ["removed", "writer"])
def test_writer_revalidates_endpoints_and_enclosing_transaction_rolls_back(tmp_path, monkeypatch, kind, fault):
    with storage.MemoryStore(tmp_path / "memory.sqlite3") as store:
        context = {"commit": "abc1234ff"} if kind == "commit" else {"files": ["shared"]}
        peers = [store.append_trace(f"Peer {i}", context) for i in range(2)]
        note = store.append_trace("Source", context)
        name = "_commit_group_candidates" if kind == "commit" else "_file_citation_candidates"
        read = getattr(ed, name)

        def removed(*args):
            rows = read(*args)
            store.connection.execute("DELETE FROM nodes WHERE id = ?", (peers[1].id,))
            return rows

        if fault == "removed":
            monkeypatch.setattr(ed, name, removed)
        else:
            store.connection.execute(f"""CREATE TRIGGER reject_edge BEFORE INSERT ON connections
                WHEN NEW.source_id = '{peers[1].id}' OR NEW.target_id = '{peers[1].id}'
                BEGIN SELECT RAISE(ABORT, 'edge fault'); END""")
        with pytest.raises(KeyError if fault == "removed" else sqlite3.IntegrityError):
            with store.transaction():
                ed.derive_edges_for_new_trace(store, note)
        assert store.get_node(peers[1].id) == peers[1]
        assert store.list_connections() == []


@pytest.mark.parametrize("kind", ["commit", "file"])
@pytest.mark.parametrize("fault", [None, "context", "SQL", "writer"])
def test_fastmcp_remember_keeps_required_commit_and_guard(tmp_path, monkeypatch, kind, fault):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setattr(server, "_decay_sweep_if_due", lambda store: None)
    guard = threading.RLock()
    monkeypatch.setattr(server, "RLock", lambda: guard)
    mcp = server.create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    try:
        context = {"commit": "abc1234ff"} if kind == "commit" else {"files": ["shared"]}
        peer = store.append_trace("Peer", context)
        name = "_commit_group_candidates" if kind == "commit" else "_file_citation_candidates"
        read = getattr(ed, name)
        observations = []

        def authorize(action, table, column, *_args):
            if fault == "SQL" and action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_READ and table == "nodes" and column in {"content", "provenance", fault}:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        def observed(*args):
            observations.append((guard._is_owned(), store.connection.in_transaction))
            store.connection.set_authorizer(authorize)
            try:
                return read(*args)
            finally:
                store.connection.set_authorizer(None)

        monkeypatch.setattr(ed, name, observed)
        if fault == "writer":
            store.connection.execute("""CREATE TRIGGER reject_edge BEFORE INSERT ON connections
                BEGIN SELECT RAISE(ABORT, 'edge fault'); END""")

        async def scenario():
            async with Client(mcp) as client:
                result = await client.call_tool("memory_remember", {"content": "Source", "context": context})
                assert not result.is_error
                return result.data["node"]["id"]

        node_id = asyncio.run(scenario())
        assert observations == [(True, False)]
        node = store.get_node(node_id)
        assert node is not None
        metadata = ({"rule": "R4a", "basis": "same_commit", "commit": "abc1234"}
                    if kind == "commit" else {"rule": "R4b", "basis": "shared_files", "files": ["shared"]})
        assert tuples(store.list_connections()) == ([expected(node, peer, 0.9 if kind == "commit" else 0.65, metadata)] if fault is None else [])
        assert not store.connection.in_transaction
    finally:
        mcp.auto_consolidation.close()
        store.close()
