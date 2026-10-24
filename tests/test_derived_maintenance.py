"""Derived maintenance preserves source knowledge and transactional cache freshness."""

from contextlib import ExitStack
import sqlite3

import pytest

from living_memory.chunking import chunk_text
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore, install_derived_only_authorizer


class FixedEmbedder:
    def embed(self, text):
        return [1.0, 0.0]


@pytest.fixture(params=["chunks", "anchors"])
def maintenance(tmp_path, request):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = store.create_node(
            level="concept", content="cobalt prism", embedding=[0.0, 1.0],
            context={"scope": "project:synthetic"},
        )
        yield store, node, request.param


def write(store, node, corpus):
    if corpus == "chunks":
        store.replace_node_chunks(
            node.id, [(chunk, [1.0, 0.0]) for chunk in chunk_text(node.content)]
        )
    else:
        with store.transaction():
            anchor = store.insert_query_anchor(
                scope=node.scope, query="orbital telemetry drift",
                fingerprint="synthetic-query", embedding=[1.0, 0.0],
            )
            store.upsert_query_anchor_edge(anchor.id, node.id, weight=1.0)


def revision(store, corpus):
    return store.chunk_corpus_revision() if corpus == "chunks" else store.query_anchor_revision()


def snapshot(store):
    return {
        table: [tuple(row) for row in store.connection.execute(f"SELECT * FROM main.{table}")]
        for table in (
            "nodes", "connections", "recall_events", "metadata",
            "node_chunk_embeddings", "query_anchors", "query_anchor_edges",
        )
    }


def recall(service):
    return service.memory_recall(
        "orbital telemetry drift", scope="project:synthetic", depth=1,
        log_access=False, log_event=False,
    )


@pytest.mark.parametrize("second_connection", [False, True])
def test_writer_refreshes_warm_recall_and_preserves_knowledge(maintenance, second_connection):
    writer, node, corpus = maintenance
    with ExitStack() as stack:
        reader = stack.enter_context(MemoryStore(writer.db_path)) if second_connection else writer
        warm = MemoryRecallService(reader, embedder=FixedEmbedder())
        assert recall(warm) == []
        before = snapshot(writer)
        previous_revision = revision(reader, corpus)
        install_derived_only_authorizer(writer.connection, corpus=corpus)

        write(writer, node, corpus)

        assert revision(reader, corpus) > previous_revision
        actual = recall(warm)
        assert actual == recall(MemoryRecallService(reader, embedder=FixedEmbedder()))
        assert [result.node.id for result in actual] == [node.id]
        assert actual[0].node.content == node.content
        assert actual[0].bm25_score == 0
        assert (actual[0].vector_score if corpus == "chunks" else actual[0].graph_score) > 0
        after = snapshot(writer)
        for table in ("nodes", "connections", "recall_events"):
            assert after[table] == before[table]
        other_key = "query_anchor_revision" if corpus == "chunks" else "chunk_corpus_revision"
        assert dict(after["metadata"])[other_key] == dict(before["metadata"])[other_key]
        assert not writer.connection.in_transaction


def test_forbidden_write_rolls_back_derived_rows_and_revision(maintenance):
    store, node, corpus = maintenance
    before = snapshot(store)
    warm = MemoryRecallService(store, embedder=FixedEmbedder())
    assert recall(warm) == []
    install_derived_only_authorizer(store.connection, corpus=corpus)
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        with store.transaction():
            write(store, node, corpus)
            assert revision(store, corpus) > revision_value(before, corpus)
            assert [result.node.id for result in recall(warm)] == [node.id]
            store.connection.execute("UPDATE nodes SET content='forbidden'")
    assert snapshot(store) == before
    assert not store.connection.in_transaction
    assert recall(warm) == recall(MemoryRecallService(store, embedder=FixedEmbedder())) == []
    # A committed retry must remain discoverable even if rollback reused a revision.
    write(store, node, corpus)
    assert [result.node.id for result in recall(warm)] == [node.id]


def revision_value(state, corpus):
    key = "chunk_corpus_revision" if corpus == "chunks" else "query_anchor_revision"
    return int(dict(state["metadata"])[key])


def test_direct_writes_stay_within_one_main_corpus(maintenance):
    store, node, corpus = maintenance
    other_tables = ("query_anchors", "query_anchor_edges") if corpus == "chunks" else ("node_chunk_embeddings",)
    table = "node_chunk_embeddings" if corpus == "chunks" else "query_anchors"
    store.connection.execute(f"CREATE TEMP TABLE {table} (id TEXT)")
    store.connection.execute("ATTACH ':memory:' AS other")
    store.connection.execute(f"CREATE TABLE other.{table} (id TEXT)")
    before = snapshot(store)
    install_derived_only_authorizer(store.connection, corpus=corpus)
    for sql in (
        "UPDATE nodes SET content='forbidden'", "DELETE FROM nodes",
        "UPDATE connections SET weight=1", "DELETE FROM recall_events",
        "UPDATE metadata SET value='0'", "UPDATE metadata SET key='forbidden'",
        "INSERT INTO metadata VALUES ('forbidden', '0')", "DELETE FROM metadata",
        *(f"DELETE FROM main.{other}" for other in other_tables),
        f"INSERT INTO temp.{table} VALUES ('forbidden')",
        f"INSERT INTO other.{table} VALUES ('forbidden')",
        "CREATE TABLE forbidden (id TEXT)", "CREATE TEMP TABLE forbidden (id TEXT)",
        "CREATE VIRTUAL TABLE forbidden USING fts5(content)",
        "CREATE INDEX forbidden ON nodes(content)",
        "CREATE VIEW forbidden AS SELECT * FROM nodes",
        "CREATE TRIGGER forbidden AFTER UPDATE ON nodes BEGIN SELECT 1; END",
        "DROP TABLE nodes", "ALTER TABLE nodes ADD COLUMN forbidden TEXT",
        "ATTACH ':memory:' AS forbidden", "DETACH other",
        "PRAGMA writable_schema=ON", "PRAGMA foreign_keys=OFF",
    ):
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            store.connection.execute(sql)
    assert snapshot(store) == before


@pytest.mark.parametrize("effect", ["metadata", "knowledge", "derived"])
def test_foreign_trigger_effects_are_denied_atomically(maintenance, effect):
    store, node, corpus = maintenance
    table = "node_chunk_embeddings" if corpus == "chunks" else "query_anchors"
    prefix = "chunk_corpus" if corpus == "chunks" else "query_anchor_corpus"
    sql = {
        "metadata": "UPDATE metadata SET value='forbidden'",
        "knowledge": "UPDATE nodes SET content='forbidden'",
        "derived": f"DELETE FROM {table} WHERE 0",
    }[effect]
    store.connection.execute(
        f"CREATE TRIGGER {prefix}_foreign AFTER INSERT ON {table} BEGIN {sql}; END"
    )
    before = snapshot(store)
    install_derived_only_authorizer(store.connection, corpus=corpus)
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        write(store, node, corpus)
    assert snapshot(store) == before
    assert not store.connection.in_transaction


@pytest.mark.parametrize("mutation", ["replace", "temp-shadow", "missing-trigger", "missing-revision"])
def test_trigger_identity_and_revision_are_required(maintenance, mutation):
    store, node, corpus = maintenance
    table = "node_chunk_embeddings" if corpus == "chunks" else "query_anchors"
    prefix = "chunk_corpus" if corpus == "chunks" else "query_anchor_corpus"
    if mutation in {"replace", "missing-trigger"}:
        store.connection.execute(f"DROP TRIGGER {prefix}_insert")
    if mutation in {"replace", "temp-shadow"}:
        temporary = "TEMP" if mutation == "temp-shadow" else ""
        store.connection.execute(
            f"CREATE {temporary} TRIGGER {prefix}_insert AFTER INSERT ON main.{table} "
            "BEGIN UPDATE metadata SET value='forbidden'; END"
        )
    if mutation == "missing-revision":
        with store.connection:
            store.connection.execute(
                "DELETE FROM metadata WHERE key=?",
                ("chunk_corpus_revision" if corpus == "chunks" else "query_anchor_revision",),
            )
    before = snapshot(store)
    install_derived_only_authorizer(store.connection, corpus=corpus)
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        write(store, node, corpus)
    assert snapshot(store) == before


def test_schema_change_rechecks_a_prepared_writer(maintenance):
    store, node, corpus = maintenance
    install_derived_only_authorizer(store.connection, corpus=corpus)
    write(store, node, corpus)
    table = "node_chunk_embeddings" if corpus == "chunks" else "query_anchors"
    prefix = "chunk_corpus" if corpus == "chunks" else "query_anchor_corpus"
    with sqlite3.connect(store.db_path) as other:
        other.executescript(
            f"DROP TRIGGER {prefix}_insert; "
            f"CREATE TRIGGER {prefix}_insert AFTER INSERT ON {table} "
            "BEGIN UPDATE metadata SET value='forbidden'; END;"
        )
    before = snapshot(store)
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        write(store, node, corpus)
    assert snapshot(store) == before
