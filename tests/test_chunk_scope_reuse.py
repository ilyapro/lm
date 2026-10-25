"""Public SQLite fixtures for shared scoped/broad chunk partitions."""

from collections import Counter
from contextlib import ExitStack
import sqlite3

import pytest

from living_memory import retrieval
from living_memory.health_audit import ReadOnlyAuditStore
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore, pack_chunk_embedding
from test_retrieval_chunk_maxpool import put_chunks


class FixedEmbedder:
    def embed(self, text):
        return [1.0, 0.0, 0.0]


@pytest.fixture(autouse=True)
def deterministic_policy(monkeypatch):
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "off")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")


@pytest.fixture(params=[False, True], ids=["same-connection", "second-connection"])
def stores(tmp_path, request):
    with ExitStack() as stack:
        path = tmp_path / "memory.sqlite3"
        reader = stack.enter_context(MemoryStore(path))
        writer = stack.enter_context(MemoryStore(path)) if request.param else reader
        yield reader, writer


def add(store, node_id, scope="project:alpha", vector=(1.0, 0.0, 0.0)):
    return store.create_node(
        node_id=node_id, level="trace", content=f"Public fact {node_id}",
        context={"scope": scope}, embedding=vector,
    )


def service(store):
    return MemoryRecallService(store, embedder=FixedEmbedder())


def recall(warm, scope=None):
    return warm.memory_recall(
        "orbital telemetry drift", scope=scope, depth=0, max_results=100,
        log_access=False, log_event=False,
    )


def reads_from(store, monkeypatch):
    reads = Counter()
    original = store.iter_chunk_embedding_rows

    def counted(**kwargs):
        for row in original(**kwargs):
            reads[row[0]] += 1
            yield row

    monkeypatch.setattr(store, "iter_chunk_embedding_rows", counted)
    return reads


@pytest.mark.parametrize("scope", [None, "project:alpha"])
@pytest.mark.parametrize("numpy", [True, False], ids=["numpy", "python"])
def test_independent_move_reuses_rows_in_ordinary_recall(stores, monkeypatch, scope, numpy):
    if not numpy:
        monkeypatch.setattr(retrieval, "_np", None)
    reader, writer = stores
    for number in range(80):
        add(writer, f"alpha-{number:03}")
    add(writer, "beta", "project:beta", (0.0, 1.0, 0.0))
    add(writer, "shared", "global")
    reads = reads_from(reader, monkeypatch)
    warm = service(reader)
    before = recall(warm, scope)
    alpha = warm._chunk_index["project:alpha"]
    shared = warm._chunk_index["global"]
    reads.clear()

    writer.update_node("beta", context={"scope": "project:gamma"})

    assert recall(warm, scope) == before
    assert warm._chunk_index["project:alpha"] is alpha
    assert warm._chunk_index["global"] is shared
    assert reads == (Counter({"beta": 1}) if scope is None else Counter())
    assert None not in warm._chunk_index

    # Scoped and broad calls use the same retained arrays in either order.
    recall(warm, "project:alpha" if scope is None else None)
    assert warm._chunk_index["project:alpha"] is alpha
    assert not any(reads[node_id] for node_id in reads if node_id.startswith("alpha-"))


def test_revision_map_tracks_both_move_endpoints_and_empty_scope_repopulation(stores):
    reader, writer = stores
    add(writer, "a")
    add(writer, "b", "project:beta")
    warm = service(reader)
    recall(warm)
    before = reader.chunk_corpus_revisions()
    writer.update_node("b", context={"scope": "project:gamma"})
    moved = reader.chunk_corpus_revisions()
    assert moved["project:alpha"] == before["project:alpha"]
    assert moved["project:beta"] > before["project:beta"]
    assert moved["project:gamma"] > 0
    assert recall(warm, "project:beta") == []
    assert [r.node.id for r in recall(warm, "project:gamma")] == ["b"]
    add(writer, "c", "project:beta")
    assert [r.node.id for r in recall(warm, "project:beta")] == ["c"]
    assert recall(warm) == recall(service(reader))


@pytest.mark.parametrize("mutation", [
    "vector", "chunks", "chunk-delete", "node-delete", "decay-revival",
    "chunk-move", "chunk-replace", "chunk-update-replace", "node-replace", "node-update-replace",
])
def test_changed_scope_is_fresh_without_reloading_other_scopes(stores, monkeypatch, mutation):
    reader, writer = stores
    add(writer, "a")
    add(writer, "b", "project:beta")
    add(writer, "c", "project:gamma", (0.0, 1.0, 0.0))
    if mutation == "decay-revival":
        writer.soft_delete_node("c")
    warm = service(reader)
    recall(warm)
    alpha = warm._chunk_index["project:alpha"]
    reads = reads_from(reader, monkeypatch)
    with writer.connection:
        if mutation == "vector":
            writer.connection.execute(
                "UPDATE node_chunk_embeddings SET embedding=? WHERE node_id='b'",
                (pack_chunk_embedding([0.0, 1.0, 0.0]),),
            )
        elif mutation == "chunks":
            put_chunks(writer, "b", [[0.0, 1.0, 0.0], [0.8, 0.6, 0.0]])
        elif mutation == "chunk-delete":
            writer.delete_node_chunks("b")
            # The ordinary bounded lazy repair supplies the next chunk set.
        elif mutation == "node-delete":
            writer.connection.execute("PRAGMA foreign_keys=OFF")
            writer.connection.execute("DELETE FROM nodes WHERE id='b'")
        elif mutation == "decay-revival":
            writer.connection.execute("UPDATE nodes SET decayed=1 WHERE id='b'")
            writer.connection.execute("UPDATE nodes SET decayed=0 WHERE id='c'")
        elif mutation == "chunk-move":
            writer.delete_node_chunks("c")
            writer.connection.execute("UPDATE node_chunk_embeddings SET node_id='c' WHERE node_id='b'")
        elif mutation in {"chunk-replace", "chunk-update-replace"}:
            writer.connection.execute("PRAGMA recursive_triggers=OFF")
            if mutation == "chunk-replace":
                writer.connection.execute(
                    "INSERT OR REPLACE INTO node_chunk_embeddings "
                    "SELECT (SELECT id FROM node_chunk_embeddings WHERE node_id='b'), "
                    "node_id, chunk_index, dimensions, embedding, token_start, token_end, "
                    "content_fingerprint, created_at, updated_at "
                    "FROM node_chunk_embeddings WHERE node_id='c'"
                )
            else:
                writer.connection.execute(
                    "UPDATE OR REPLACE node_chunk_embeddings SET "
                    "id=(SELECT id FROM node_chunk_embeddings WHERE node_id='b') WHERE node_id='c'"
                )
        elif mutation == "node-replace":
            writer.connection.execute("PRAGMA recursive_triggers=OFF")
            columns = [row[1] for row in writer.connection.execute("PRAGMA table_info(nodes)")]
            selection = ["'project:gamma'" if column == "scope" else column for column in columns]
            writer.connection.execute(
                f"INSERT OR REPLACE INTO nodes ({','.join(columns)}) "
                f"SELECT {','.join(selection)} FROM nodes WHERE id='b'"
            )
        else:
            writer.connection.execute("PRAGMA foreign_keys=OFF")
            writer.connection.execute("PRAGMA recursive_triggers=OFF")
            writer.connection.execute("UPDATE OR REPLACE nodes SET id='b' WHERE id='c'")

    actual = recall(warm)
    assert reads["a"] == 0
    assert warm._chunk_index["project:alpha"] is alpha
    assert actual == recall(service(reader))
    for scope in ("project:alpha", "project:beta", "project:gamma"):
        assert recall(warm, scope) == recall(service(reader), scope)


@pytest.mark.parametrize("scope", [None, "project:alpha"])
def test_rollback_does_not_cache_a_revision_that_another_write_can_reuse(tmp_path, scope):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        add(store, "a")
        add(store, "b", vector=(0.0, 1.0, 0.0))
        warm = service(store)
        before = recall(warm, scope)
        committed = dict(warm._chunk_index)
        revision = store.chunk_corpus_revisions()
        store.connection.execute(
            "UPDATE node_chunk_embeddings SET embedding=? WHERE node_id='a'",
            (pack_chunk_embedding([0.0, 1.0, 0.0]),),
        )
        assert recall(warm, scope) == []
        assert warm._chunk_index == committed
        store.connection.rollback()
        assert store.chunk_corpus_revisions() == revision
        assert recall(warm, scope) == before
        with store.connection:
            store.connection.execute(
                "UPDATE node_chunk_embeddings SET embedding=? WHERE node_id='b'",
                (pack_chunk_embedding([1.0, 0.0, 0.0]),),
            )
        assert [r.node.id for r in recall(warm, scope)] == ["a", "b"]
        assert recall(warm, scope) == recall(service(store), scope)


def test_scope_move_during_partition_read_keeps_one_version_per_node(tmp_path, monkeypatch):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as reader, MemoryStore(path) as writer:
        add(writer, "a")
        add(writer, "b", "project:beta")
        warm = service(reader)
        recall(warm)
        put_chunks(writer, "b", [[0.8, 0.6, 0.0]])
        original = reader.iter_chunk_embedding_rows
        moved = False

        def move_during_read(**kwargs):
            nonlocal moved
            if kwargs.get("scope") == "project:beta" and not moved:
                moved = True
                writer.update_node("a", context={"scope": "project:beta"})
            yield from original(**kwargs)

        monkeypatch.setattr(reader, "iter_chunk_embedding_rows", move_during_read)
        actual = recall(warm)
        assert moved
        assert [r.node.id for r in actual].count("a") == 1
        assert not reader.connection.in_transaction
        assert recall(warm) == recall(service(reader))
        assert recall(warm, "project:alpha") == []


def test_read_only_legacy_snapshot_does_not_cache_vectors(tmp_path, monkeypatch):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store:
        add(store, "a")
        with store.connection:
            for suffix in ("insert", "update", "delete", "membership", "node_insert", "node_delete"):
                store.connection.execute(f"DROP TRIGGER chunk_corpus_{suffix}")
            store.connection.execute("DELETE FROM metadata WHERE key LIKE 'chunk_scope_revision%'")
            store.connection.execute("INSERT INTO metadata VALUES ('chunk_corpus_revision', '3')")
    with ReadOnlyAuditStore.open_path(path) as reader, sqlite3.connect(path) as writer:
        warm = service(reader)
        assert reader.chunk_corpus_revisions() is None
        assert [r.node.id for r in recall(warm)] == ["a"]
        with writer:
            writer.execute("UPDATE node_chunk_embeddings SET embedding=?", (pack_chunk_embedding([0.0, 1.0, 0.0]),))
        assert recall(warm) == recall(warm, "project:alpha") == []
        assert warm._chunk_index == {}
        original = reader.get_node("a")
    with MemoryStore(path) as migrated:
        assert migrated.get_node("a") == original
        assert migrated.chunk_corpus_revisions() == {"project:alpha": 0}
        assert migrated.connection.execute("SELECT 1 FROM metadata WHERE key='chunk_corpus_revision'").fetchone() is None


@pytest.mark.parametrize("scope", [None, "project:alpha"])
def test_fresh_correction_and_eligibility_do_not_depend_on_vector_revision(stores, monkeypatch, scope):
    reader, writer = stores
    add(writer, "stale")
    add(writer, "correction")
    warm = service(reader)
    assert not any(r.superseded for r in recall(warm, scope))
    indexes = dict(warm._chunk_index)
    revisions = reader.chunk_corpus_revisions()
    reads = reads_from(reader, monkeypatch)
    writer.create_connection("correction", "stale", "supersedes")
    actual = recall(warm, scope)
    assert actual == recall(service(reader), scope)
    assert [r.node.id for r in actual] == ["correction", "stale"]
    assert actual[1].superseded and actual[1].correction_ids == ("correction",)
    reads.clear()
    writer.update_node("correction", context={"is_rejected_alternative": True})
    actual = recall(warm, scope)
    assert reads == {}
    assert [r.node.id for r in actual] == ["stale"]
    assert actual[0].superseded
    assert reader.chunk_corpus_revisions() == revisions
    assert all(warm._chunk_index[key] is value for key, value in indexes.items())


@pytest.mark.parametrize("numpy", [True, False], ids=["numpy", "python"])
def test_partition_gather_preserves_order_width_and_normalization(tmp_path, monkeypatch, numpy):
    if not numpy:
        monkeypatch.setattr(retrieval, "_np", None)
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        for number in range(27):
            scope = ("global", "project:alpha", "project:beta")[number % 3]
            node = add(store, f"node-{number:03}", scope)
            # Interleaved scopes, multiple chunks, two widths, near-unit rows
            # sharing a broad width block with non-unit and zero rows.
            vectors = [[1.00005, 0.0, 0.0], [0.8, 0.6, 0.0]]
            if number % 3 == 0:
                vectors += [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]
            if number % 4 == 0:
                vectors.append([0.2, 0.9])
            put_chunks(store, node.id, vectors)
        warm = service(store)
        plan = warm.scope_resolver.resolve(store=store)
        for query in ([1.0, 0.0, 0.0], [0.3, 0.9, 0.2], [0.2, 0.9]):
            gathered = list(warm._chunk_blocks(plan, len(query)))
            scanned = [block for block in warm._build_chunk_index(None, None).blocks
                       if block.dimension == len(query)]
            assert len(gathered) == len(scanned) == 1
            query_array = retrieval._as_query_array(query)
            assert retrieval._pooled_chunk_similarities(gathered[0], query, query_array) == (
                retrieval._pooled_chunk_similarities(scanned[0], query, query_array)
            )
        assert recall(warm) == recall(service(store))


@pytest.mark.parametrize("numpy", [True, False], ids=["numpy", "python"])
def test_broad_layout_survives_scope_switches_and_releases_replaced_blocks(stores, monkeypatch, numpy):
    if not numpy:
        monkeypatch.setattr(retrieval, "_np", None)
    reader, writer = stores
    for number in range(9):
        add(writer, f"node-{number}", ("global", "project:alpha", "project:beta")[number % 3])
    prepared = []
    prepare = retrieval._prepare_chunk_gather

    def counted(blocks):
        result = prepare(blocks)
        prepared.append(result)
        return result

    monkeypatch.setattr(retrieval, "_prepare_chunk_gather", counted)
    warm = service(reader)
    before = recall(warm)
    layout = warm._chunk_gathers[3]
    if numpy:
        assert layout.template.rows is None
    else:
        assert all(any(row is original for block in layout.blocks for original in block.rows)
                   for row in layout.template.rows)
    reads = reads_from(reader, monkeypatch)
    for scope in ("project:alpha", None, "global", "project:beta", None):
        result = recall(warm, scope)
        if scope is None:
            assert result == before
        assert warm._chunk_gathers[3] is layout
    assert len(prepared) == 1
    assert reads == {}

    # A scoped refresh must release a layout retaining obsolete broad arrays.
    writer.update_node("node-2", context={"scope": "project:gamma"})
    recall(warm, "project:beta")
    assert 3 not in warm._chunk_gathers
    assert len(prepared) == 1
    after = recall(warm)
    assert [(r.node.id, r.score) for r in after] == [(r.node.id, r.score) for r in before]
    assert next(r.node.scope for r in after if r.node.id == "node-2") == "project:gamma"
    assert len(prepared) == 2
    assert warm._chunk_gathers[3] is not layout
    assert all(block is not layout.blocks[-1] for block in warm._chunk_gathers[3].blocks)


def test_nonfinite_norm_preserves_the_complete_blocks_normalization_rule(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        add(store, "a", vector=(3.0, 4.0, 0.0))
        add(store, "b", "project:beta")
        with store.connection:
            store.connection.execute(
                "UPDATE node_chunk_embeddings SET embedding=? WHERE node_id='b'",
                (pack_chunk_embedding([float("nan"), 0.0, 0.0]),),
            )
        warm = service(store)
        plan = warm.scope_resolver.resolve(store=store)
        gathered, = warm._chunk_blocks(plan, 3)
        scanned, = warm._build_chunk_index(None, None).blocks
        assert not gathered.needs_normalization and not scanned.needs_normalization
        query = [1.0, 0.0, 0.0]
        q_arr = retrieval._as_query_array(query)
        actual = dict((node_id, score) for score, node_id in retrieval._pooled_chunk_similarities(gathered, query, q_arr))
        expected = dict((node_id, score) for score, node_id in retrieval._pooled_chunk_similarities(scanned, query, q_arr))
        assert actual["a"] == expected["a"] == 3.0
