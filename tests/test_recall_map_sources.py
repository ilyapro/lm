"""Recall-map source projections preserve storage and cascade boundaries."""

from dataclasses import replace
import sqlite3
from unittest.mock import patch

import pytest

from living_memory.chunking import TextChunk
from living_memory.health_audit import ReadOnlyAuditStore
from living_memory.recall_map import RecallMapBuilder, _Curtailment, _Member
from living_memory.retrieval import RecallResult
from living_memory.storage import MemoryStore
from test_recall_map_anchor_observation import SCOPE, build, due, seed_pool


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LM_MAP_CURTAIL_DECAY", "1")
    with MemoryStore(tmp_path / "sources.sqlite3") as opened:
        yield opened


def embedded(store, node_id="embedded"):
    node = store.create_node(
        node_id=node_id, level="trace", content="orbital telemetry calibration",
        context={"scope": SCOPE}, embedding=[1.0, 0.0],
    )
    return RecallResult(node=node, score=0.8, vector_score=1.0)


@pytest.mark.parametrize("verdict", [_Curtailment(0, 0), due()])
def test_build_reads_only_consumed_source_fields_and_stage_survivors(store, verdict):
    results = seed_pool(store, early_stages=True) + [embedded(store)]
    builder = RecallMapBuilder(store)
    with patch.object(builder, "_curtailment", return_value=verdict):
        expected = build(builder, results)
    allowed = {
        "query_anchors": {"id", "query", "decayed"},
        "query_anchor_edges": {"target_id", "anchor_id", "weight"},
        "node_chunk_embeddings": {"node_id", "dimensions", "embedding", "chunk_index"},
    }

    def authorize(action, table, column, database, trigger):
        if action == sqlite3.SQLITE_READ and table in allowed and column not in allowed[table]:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    # A malformed vector on an already claimed node must never be decoded.
    claimed = results[0].node
    store.update_node(claimed.id, embedding=[1.0, 0.0])
    with store.connection:
        store.connection.execute(
            "UPDATE node_chunk_embeddings SET embedding = X'ff' WHERE node_id = ?", (claimed.id,),
        )
    builder = RecallMapBuilder(store)
    store.connection.set_authorizer(authorize)
    try:
        with (
            patch.object(builder, "_curtailment", return_value=verdict),
            patch.object(store, "get_node_chunk_vectors", wraps=store.get_node_chunk_vectors) as vectors,
        ):
            actual = build(builder, results)
        assert actual == expected
        assert vectors.call_args.args == ([results[-1].node.id],)
    finally:
        store.connection.set_authorizer(None)


@pytest.mark.parametrize("limit", [7, 999])
def test_source_batches_fit_sqlite_variable_limit_and_deduplicate_ids(store, limit):
    results = seed_pool(store) + [embedded(store)]
    ids = [result.node.id for result in results]
    expected_links = store.get_query_anchor_links(ids)
    expected_vectors = store.get_node_chunk_vectors(ids)
    requested = ids + [f"absent-{i}" for i in range(limit)] + ids
    store.connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, limit)
    assert sorted(store.get_query_anchor_links(requested), key=lambda row: row.target_id) == expected_links
    assert store.get_node_chunk_vectors(requested) == expected_vectors


def test_chunk_ordinal_width_and_node_fallback_define_vectors(store):
    results = [embedded(store, node_id) for node_id in ("mixed", "empty", "fallback", "absent")]
    chunk = TextChunk("sample", 0, 0, 1, 0, 6)
    store.replace_node_chunks(results[0].node.id, [
        (replace(chunk, chunk_index=0), [0.0, 1.0]),
        (replace(chunk, chunk_index=1), [1.0, 0.0, 0.0]),
        (replace(chunk, chunk_index=2), [0.0, 0.0, 1.0]),
        (replace(chunk, chunk_index=3), [0.0, 1.0, 0.0]),
    ])
    with store.connection:
        store.connection.execute(
            "UPDATE node_chunk_embeddings SET chunk_index = chunk_index + 10 WHERE node_id = 'mixed'"
        )
        store.connection.execute("""UPDATE node_chunk_embeddings SET chunk_index =
            CASE chunk_index WHEN 10 THEN 8 WHEN 11 THEN 3 WHEN 12 THEN 7 WHEN 13 THEN 4 END
            WHERE node_id = 'mixed'""")
    # Older unconstrained tables can contain an empty vector.
    store.connection.execute("PRAGMA ignore_check_constraints = ON")
    with store.connection:
        store.connection.execute(
            "UPDATE node_chunk_embeddings SET dimensions = 0, embedding = X'' WHERE node_id = 'empty'"
        )
    for result in results[2:]:
        store.delete_node_chunks(result.node.id)
    results[1].node.embedding = [0.0, 1.0]
    results[2].node.embedding = [0.0, 0.0, 2.0]
    results[3].node.embedding = []
    members = [_Member(rank, result.node) for rank, result in enumerate(results)]
    _, vectors, rest = RecallMapBuilder(store)._embedding_membership(members)
    assert vectors["mixed"] == pytest.approx([3 ** -0.5] * 3)
    assert vectors["empty"] == [0.0, 1.0]
    assert vectors["fallback"] == [0.0, 0.0, 1.0]
    assert [member.node.id for member in rest] == ["absent"]


@pytest.mark.parametrize("missing", [(), ("query_anchors",), ("query_anchor_edges",),
                                   ("node_chunk_embeddings",),
                                   ("query_anchors", "query_anchor_edges", "node_chunk_embeddings")])
def test_read_only_sources_tolerate_missing_tables_and_chunk_coordinates(store, missing):
    results = seed_pool(store) + [embedded(store)]
    store.connection.execute("PRAGMA foreign_keys = OFF")
    with store.connection:
        for table in missing:
            store.connection.execute(f"DROP TABLE {table}")
        if "node_chunk_embeddings" not in missing:
            store.connection.execute("ALTER TABLE node_chunk_embeddings DROP COLUMN char_start")
            store.connection.execute("ALTER TABLE node_chunk_embeddings DROP COLUMN char_end")
    with ReadOnlyAuditStore.open_path(store.db_path) as reader:
        builder = RecallMapBuilder(reader)
        members = [_Member(rank, result.node) for rank, result in enumerate(results)]
        assignment = builder._anchor_assignment(members)
        assert assignment.by_node == (
            {} if {"query_anchors", "query_anchor_edges"}.intersection(missing) else
            {"member-0": "anchor-a", "member-1": "anchor-a", "member-2": "anchor-b"}
        )
        _, vectors, _ = builder._embedding_membership(members)
        assert vectors == {"embedded": [1.0, 0.0]}
        with patch.object(builder, "_curtailment", return_value=due()):
            result = build(builder, results)
        assert result is not None and result.clusters
        assert reader.connection.total_changes == 0
        assert not reader.connection.in_transaction


@pytest.mark.parametrize("column, value, error", [
    ("dimensions", "invalid", "invalid literal"),
    ("embedding", b"bad", "chunk embedding blob"),
])
def test_bad_consumed_chunk_data_propagates(store, column, value, error):
    result = embedded(store)
    with store.connection:
        store.connection.execute(f"UPDATE node_chunk_embeddings SET {column} = ?", (value,))
    with pytest.raises(ValueError, match=error):
        build(RecallMapBuilder(store), [result])


@pytest.mark.parametrize("anchor_state", ["live", "decayed", "missing"])
def test_bad_edge_weight_propagates_even_without_live_anchor(store, anchor_state):
    results = seed_pool(store)
    store.connection.execute("PRAGMA foreign_keys = OFF")
    with store.connection:
        store.connection.execute("UPDATE query_anchor_edges SET weight = 'bad' WHERE anchor_id = 'anchor-a'")
        if anchor_state == "decayed":
            store.connection.execute("UPDATE query_anchors SET decayed = 1 WHERE id = 'anchor-a'")
        elif anchor_state == "missing":
            store.connection.execute("DELETE FROM query_anchors WHERE id = 'anchor-a'")
    with pytest.raises(ValueError, match="could not convert string to float"):
        build(RecallMapBuilder(store), results)


def test_chunk_sql_failure_propagates(store):
    result = embedded(store)

    def authorize(action, table, column, database, trigger):
        return (sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ
                and table == "node_chunk_embeddings" else sqlite3.SQLITE_OK)

    store.connection.set_authorizer(authorize)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="prohibited"):
            build(RecallMapBuilder(store), [result])
    finally:
        store.connection.set_authorizer(None)


def test_empty_source_sets_do_not_probe_storage(store):
    statements = []
    store.connection.set_trace_callback(statements.append)
    try:
        assert store.get_query_anchor_links([]) == []
        assert store.get_node_chunk_vectors([]) == {}
    finally:
        store.connection.set_trace_callback(None)
    assert statements == []
