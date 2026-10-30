"""Selected chunk reduction preserves rank order and owns its read lifetime."""

from dataclasses import replace
from math import inf, nextafter, sqrt
import sqlite3
from unittest.mock import patch

import pytest

from living_memory import storage
from living_memory.chunking import TextChunk
from living_memory.recall_map import RecallMapBuilder, STAGE_EMBEDDING, _Member
from living_memory.retrieval import RecallResult
from living_memory.storage import MemoryStore
from test_recall_map import seed_corpus


SCOPE = "project:chunk-stream"
CHUNK = TextChunk("sample", 0, 0, 1, 0, 6)


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "chunks.sqlite3") as opened:
        yield opened


def pool(store, count=24, width=8, chunks=1):
    topics = ("orbital telemetry calibration", "orchard harvest rotation",
              "network packet routing", "glacier temperature survey")
    results = []
    for index in range(count):
        vector = [float(axis == index % 4) for axis in range(width)]
        node = store.create_node(
            node_id=f"node-{count - index:03}", level="trace",
            content=f"{topics[index % 4]} sample {index}",
            context={"scope": SCOPE}, embedding=vector,
        )
        store.replace_node_chunks(node.id, [
            (replace(CHUNK, chunk_index=ordinal), [2.0 * v for v in vector])
            for ordinal in range(chunks)
        ])
        results.append(RecallResult(node=node, score=1.0 - index / (count + 1), vector_score=1.0))
    return results


def membership_ids(groups):
    return [[member.node.id for member in group] for group in groups]


@pytest.mark.parametrize("chunks", [1, 17, 65])
def test_reduction_retains_only_one_decoded_node_and_build_recounts(store, monkeypatch, chunks):
    seed_corpus(store)
    results = pool(store, chunks=chunks)
    members = [_Member(i, result.node) for i, result in enumerate(results)]
    expected_groups = [[result.node.id for result in results[axis::4]] for axis in range(4)]
    expected_vectors = {
        result.node.id: [float(axis == index % 4) for axis in range(8)]
        for index, result in enumerate(results)
    }
    live = peak = decoded = 0
    unpack = storage.unpack_chunk_embedding

    class TrackedVector(list):
        def __init__(self, values):
            nonlocal live, peak, decoded
            super().__init__(values)
            live += 1
            decoded += 1
            peak = max(peak, live)

        def __del__(self):
            nonlocal live
            live -= 1

    monkeypatch.setattr(storage, "unpack_chunk_embedding", lambda *a: TrackedVector(unpack(*a)))
    builder = RecallMapBuilder(store)
    groups, vectors, rest = builder._embedding_membership(members)
    assert membership_ids(groups) == expected_groups
    assert vectors == expected_vectors and list(vectors) == [r.node.id for r in results]
    assert rest == [] and live == 0 and peak == chunks
    assert decoded == len(results) * chunks

    for hit in (False, True):
        built = builder.build(results, scope=SCOPE, task="survey")
        assert builder.last_cache_hit is hit
        # The ordinary wire budget carries the first two of the four groups.
        assert [list(cluster.member_ids) for cluster in built.clusters] == expected_groups[:2]
        assert [cluster.medoid.node_id for cluster in built.clusters] == [g[0] for g in expected_groups[:2]]
        assert {cluster.stage for cluster in built.clusters} == {STAGE_EMBEDDING}
        assert built.pool_size == len(results) and built.covered == len(results) // 2
        assert built.withheld == 0 and built.dropped == 2
        assert live == 0 and peak == chunks
        if not hit:
            expected = built
        else:
            assert built == expected
    assert decoded == 3 * len(results) * chunks


@pytest.mark.parametrize("limit", [2, 999])
def test_grouped_reader_batches_duplicates_and_missing_ids_without_fanout(store, limit):
    results = pool(store, count=8, chunks=3)
    ids = [r.node.id for r in results]
    requested = ids + [f"missing-{i}" for i in range(limit)] + ids
    store.connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, limit)
    statements = []
    store.connection.set_trace_callback(statements.append)
    try:
        with store.iter_node_chunk_vectors(requested) as groups:
            actual = list(groups)
    finally:
        store.connection.set_trace_callback(None)
    assert len(actual) == len(ids) and set(dict(actual)) == set(ids)
    assert dict(actual) == {
        result.node.id: [[2.0 * float(axis == index % 4) for axis in range(8)]] * 3
        for index, result in enumerate(results)
    }
    selects = [sql for sql in statements if "SELECT node_id, chunk_index" in sql]
    assert len(selects) == (len(set(requested)) + limit - 1) // limit
    assert store.get_node_chunk_vectors(requested) == dict(actual)


def test_member_duplicates_keep_individual_fallbacks_and_original_order(store):
    results = pool(store, count=4)
    for result in results:
        store.delete_node_chunks(result.node.id)
    first = results[0].node
    members = [_Member(0, first), _Member(1, replace(first, embedding=[])),
               _Member(2, results[1].node), _Member(3, replace(first, embedding=[0.0, 2.0]))]
    groups, vectors, rest = RecallMapBuilder(store)._embedding_membership(members)
    assert vectors == {first.id: [0.0, 1.0], results[1].node.id: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    assert groups == [[members[0], members[3]], [members[2]]]
    assert rest == [members[1]]


@pytest.mark.parametrize("threshold, joined", [(nextafter(0.6, -inf), True), (0.6, True),
                                               (nextafter(0.6, inf), False)])
def test_sum_cancellation_and_boundary_cosine_keep_exact_vectors(store, threshold, joined):
    results = pool(store, count=2)
    store.replace_node_chunks(results[0].node.id, [
        (replace(CHUNK, chunk_index=i), [value, 0.0])
        for i, value in enumerate([2.0 ** 54, 1.0, -(2.0 ** 54)])
    ])
    store.replace_node_chunks(results[1].node.id, [(CHUNK, [3.0, 4.0])])
    members = [_Member(i, result.node) for i, result in enumerate(results)]
    groups, vectors, rest = RecallMapBuilder(store, cluster_cosine=threshold)._embedding_membership(members)
    assert vectors == {members[0].node.id: [1.0, 0.0], members[1].node.id: [0.6, 0.8]}
    assert groups == ([members] if joined else [[member] for member in members])
    assert rest == []


def test_equal_similarity_chooses_first_ranked_leader(store):
    results = pool(store, count=3)
    for result, vector in zip(results, ([1.0, 0.0], [0.0, 1.0], [1.0, 1.0])):
        store.replace_node_chunks(result.node.id, [(CHUNK, vector)])
    members = [_Member(i, r.node) for i, r in enumerate(results)]
    groups, vectors, rest = RecallMapBuilder(store, cluster_cosine=0.7)._embedding_membership(members)
    assert groups == [[members[0], members[2]], [members[1]]]
    assert vectors[members[2].node.id] == [1.0 / sqrt(2.0)] * 2
    assert rest == []


def test_first_nonempty_ordinal_sets_width(store):
    result = pool(store, count=1)[0]
    store.replace_node_chunks(result.node.id, [
        (replace(CHUNK, chunk_index=i), vector)
        for i, vector in enumerate(([1.0], [3.0, 4.0], [1.0, 0.0, 0.0]))
    ])
    store.connection.execute("PRAGMA ignore_check_constraints = ON")
    with store.connection:
        store.connection.execute(
            "UPDATE node_chunk_embeddings SET dimensions=0, embedding=X'' WHERE chunk_index=0"
        )
    member = _Member(0, result.node)
    groups, vectors, rest = RecallMapBuilder(store)._embedding_membership([member])
    assert groups == [[member]] and rest == []
    assert vectors == {result.node.id: [0.6, 0.8]}


@pytest.mark.parametrize("column,value", [("chunk_index", "bad"), ("dimensions", "bad"),
                                         ("embedding", b"bad")])
def test_unused_widths_are_validated_after_an_earlier_node_is_reduced(store, column, value):
    results = pool(store, count=2)
    bad = results[0].node.id  # Descending rank IDs put this node last in SQL.
    store.replace_node_chunks(bad, [(CHUNK, [1.0, 0.0]),
                                   (replace(CHUNK, chunk_index=1), [1.0, 0.0, 0.0])])
    with store.connection:
        store.connection.execute(
            f"UPDATE node_chunk_embeddings SET {column} = ? WHERE node_id = ? AND chunk_index = 1",
            (value, bad),
        )
    builder = RecallMapBuilder(store)
    with patch.object(builder, "_node_vector", wraps=builder._node_vector) as reduce:
        with pytest.raises(ValueError):
            builder.build(results, scope=SCOPE, task="survey")
    assert [call.args[0].id for call in reduce.call_args_list] == [results[1].node.id]
    assert builder.cache_size == 0


@pytest.mark.parametrize("failure", [None, "early", "read", "reduce", "close"])
def test_stream_closes_actual_cursors_before_result_or_failure(store, monkeypatch, failure):
    results = pool(store, count=4)
    connection = store.connection
    cursors = []
    reductions = []

    class Cursor:
        def __init__(self, raw):
            self.raw, self.closed, self.read = raw, False, 0

        def __iter__(self):
            return self

        def __next__(self):
            self.read += 1
            if failure == "read" and self.read == 3:
                raise sqlite3.OperationalError("late chunk read")
            return next(self.raw)

        def close(self):
            self.raw.close()
            self.closed = True
            if failure == "close":
                raise sqlite3.OperationalError("late chunk close")

    class Connection:
        def __getattr__(self, name):
            return getattr(connection, name)

        def execute(self, sql, *args):
            cursor = connection.execute(sql, *args)
            if "SELECT node_id, chunk_index" in sql:
                cursor = Cursor(cursor)
                cursors.append(cursor)
            return cursor

    monkeypatch.setattr(store, "_conn", Connection())
    builder = RecallMapBuilder(store)
    reduce = builder._node_vector

    def node_vector(node, chunks):
        reductions.append(node.id)
        if failure == "reduce":
            raise ValueError("chunk reducer")
        return reduce(node, chunks)

    monkeypatch.setattr(builder, "_node_vector", node_vector)
    if failure == "early":
        with store.iter_node_chunk_vectors([r.node.id for r in results]) as groups:
            next(groups)
        assert list(groups) == []
    elif failure is None:
        assert builder.build(results, scope=SCOPE, task="survey") is not None
    else:
        with pytest.raises((ValueError, sqlite3.OperationalError)):
            builder.build(results, scope=SCOPE, task="survey")
        assert builder.cache_size == 0 and reductions
    assert cursors and all(cursor.closed for cursor in cursors)
    for cursor in cursors:
        with pytest.raises(sqlite3.ProgrammingError, match="closed cursor"):
            cursor.raw.fetchone()
    assert not connection.in_transaction
    # The live connection cannot keep a read descriptor/lock via the stream.
    with sqlite3.connect(store.db_path, timeout=0) as writer:
        assert writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone() == (0, 0, 0)


def test_empty_stream_does_not_probe_storage(store):
    statements = []
    store.connection.set_trace_callback(statements.append)
    with store.iter_node_chunk_vectors([]) as groups:
        assert list(groups) == []
    store.connection.set_trace_callback(None)
    assert statements == []
