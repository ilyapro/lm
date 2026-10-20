"""Warm anchor discovery follows live membership, vectors and scoped reads."""

from contextlib import ExitStack
from pathlib import Path
import runpy
import sqlite3

import pytest

from living_memory import query_anchors as qa
from living_memory.health_audit import ReadOnlyAuditStore
from living_memory.irrelevance import QUERY_IRRELEVANCE_SCHEMA_SQL
from living_memory.recall_map import RecallMapBuilder
from living_memory.retrieval import MemoryRecallService
from living_memory.scope import ScopePlan
from living_memory.storage import MemoryStore, pack_chunk_embedding


class FixedEmbedder:
    def embed(self, text):
        return [0.0, 1.0] if text == "restored situation" else [1.0, 0.0]


@pytest.fixture(params=["memory", "same-connection", "second-connection"])
def stores(tmp_path, request):
    with ExitStack() as stack:
        path = ":memory:" if request.param == "memory" else tmp_path / "memory.sqlite3"
        reader = stack.enter_context(MemoryStore(path))
        writer = stack.enter_context(MemoryStore(path)) if request.param == "second-connection" else reader
        yield reader, writer


def service(store):
    return MemoryRecallService(store, embedder=FixedEmbedder())


def anchor(store, name, vector, scope="global"):
    return store.insert_query_anchor(
        scope=scope, query=name, fingerprint=name, embedding=vector, anchor_id=name
    )


def matches(retriever, vector, scope="global"):
    source = retriever._anchor_vector_source()
    return [] if source is None else qa.match_anchors(source, vector, scope)


def assert_parity(warm, vector, expected, scope="global"):
    actual = matches(warm, vector, scope)
    fresh = matches(service(warm.store), vector, scope)
    assert actual == fresh
    assert [match.anchor.id for match in actual] == expected
    assert all(not match.anchor.decayed for match in actual)


@pytest.mark.parametrize("numpy", [True, False], ids=["numpy", "python"])
def test_balanced_decay_revival_updates_matches_seeds_and_demotions(stores, numpy, monkeypatch):
    if not numpy:
        monkeypatch.setattr(qa, "_np", None)
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    monkeypatch.setenv("LM_QUERY_IRRELEVANCE_FACTOR", "0.5")
    reader, writer = stores
    a = anchor(writer, "a", [1.0, 0.0])
    b = anchor(writer, "b", [0.0, 1.0])
    targets = []
    for item in (a, b):
        node = writer.create_node(level="concept", content=item.id, context={"scope": "global"})
        writer.upsert_query_anchor_edge(item.id, node.id, weight=1.0)
        targets.append(node.id)
        with writer.connection:
            writer.connection.execute(QUERY_IRRELEVANCE_SCHEMA_SQL)
            writer.connection.execute(
                "INSERT INTO query_irrelevance (anchor_id, node_id, weight, marks, created_at, updated_at) "
                "VALUES (?, ?, 1.0, 1, 't', 't')", (item.id, node.id)
            )
    writer.decay_query_anchor(b.id, "inactive")
    warm = service(reader)
    assert_parity(warm, [1.0, 0.0], [a.id])
    before = reader.query_anchor_revision()
    writer.decay_query_anchor(a.id, "inactive")
    writer.reinforce_query_anchor(b.id)

    assert_parity(warm, [0.0, 1.0], [b.id])
    assert_parity(warm, [1.0, 0.0], [])
    assert reader.query_anchor_revision() != before
    plan = ScopePlan("global", ("global",))
    for retriever in (warm, service(reader)):
        assert retriever._collect_anchor_seeds(plan, [1.0, 0.0]) == {}
        assert retriever._collect_query_demotions(plan, [1.0, 0.0]) == {}
        assert retriever._collect_anchor_seeds(plan, [0.0, 1.0]) == {targets[1]: 1.0}
        assert retriever._collect_query_demotions(plan, [0.0, 1.0]) == {targets[1]: 0.5}


@pytest.mark.parametrize("mutation", ["embedding", "dimensions", "scope", "id", "replace", "delete-insert"])
def test_sql_mutations_refresh_cached_vectors(stores, mutation):
    reader, writer = stores
    anchor(writer, "a", [1.0, 0.0])
    anchor(writer, "z", [0.0, 1.0], "project:elsewhere")
    warm = service(reader)
    assert_parity(warm, [1.0, 0.0], ["a"])
    assert_parity(warm, [1.0, 0.0], [], "project:destination")
    expected, vector, scope = ["a"], [1.0, 0.0], "global"
    with writer.connection:
        if mutation == "embedding":
            vector = [0.0, 1.0]
            writer.connection.execute("UPDATE query_anchors SET embedding = ? WHERE id = 'a'", (pack_chunk_embedding(vector),))
        elif mutation == "dimensions":
            vector = [1.0, 0.0, 0.0]
            writer.connection.execute("UPDATE query_anchors SET dimensions = 3, embedding = ? WHERE id = 'a'", (pack_chunk_embedding(vector),))
        elif mutation == "scope":
            scope = "project:destination"
            writer.connection.execute("UPDATE query_anchors SET scope = ? WHERE id = 'a'", (scope,))
        elif mutation == "id":
            expected = ["b"]
            writer.connection.execute("UPDATE query_anchors SET id = 'b' WHERE id = 'a'")
        elif mutation == "replace":
            # SQLite REPLACE need not fire DELETE triggers with recursive_triggers off.
            writer.connection.execute("PRAGMA recursive_triggers = OFF")
            writer.connection.execute(
                "INSERT OR REPLACE INTO query_anchors "
                "SELECT id, scope, query, fingerprint, dimensions, embedding, "
                "reinforcement_count, usefulness_score, 1, 'inactive', first_seen, "
                "last_matched_at, created_at, updated_at FROM query_anchors WHERE id = 'a'"
            )
            expected = []
        else:
            writer.connection.execute("DELETE FROM query_anchors WHERE id = 'a'")
            anchor(writer, "b", vector)
            expected = ["b"]
    assert_parity(warm, vector, expected, scope)
    if mutation == "scope":
        assert_parity(warm, vector, [], "global")
    if mutation in {"embedding", "dimensions"}:
        assert_parity(warm, [1.0, 0.0], [])


def test_bookkeeping_and_chunk_writes_preserve_anchor_cache(stores, monkeypatch):
    reader, writer = stores
    a = anchor(writer, "a", [1.0, 0.0])
    warm = service(reader)
    assert_parity(warm, [1.0, 0.0], [a.id])
    cached = warm._anchor_vectors
    original = reader.iter_query_anchor_vectors
    scans = []

    def scan(*args, **kwargs):
        scans.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(reader, "iter_query_anchor_vectors", scan)
    writer.reinforce_query_anchor(a.id, usefulness_delta=0.5)
    node = writer.create_node(level="concept", content="target", context={"scope": "global"}, embedding=[1.0, 0.0])
    writer.upsert_query_anchor_edge(a.id, node.id, weight=0.5)
    with writer.connection:
        writer.connection.execute(
            "UPDATE query_anchors SET id=id, scope=scope, dimensions=dimensions, "
            "embedding=embedding, decayed=decayed, query='current wording', updated_at='t'"
        )
    for _ in range(3):
        result = matches(warm, [1.0, 0.0])
        assert result[0].anchor.query == "current wording"
        assert result[0].anchor.reinforcement_count == 1
        assert result[0].targets == ((node.id, 0.5),)
        assert warm._anchor_vectors is cached
    assert scans == []
    chunk_revision = reader.chunk_corpus_revision()
    writer.decay_query_anchor(a.id, "inactive")
    assert matches(warm, [1.0, 0.0]) == []
    assert reader.chunk_corpus_revision() == chunk_revision
    writer.reinforce_query_anchor(a.id)
    assert_parity(warm, [1.0, 0.0], [a.id])


def test_revision_and_membership_roll_back_together(stores):
    reader, writer = stores
    a = anchor(writer, "a", [1.0, 0.0])
    warm = service(reader)
    assert_parity(warm, [1.0, 0.0], [a.id])
    before = reader.query_anchor_revision()
    writer.connection.execute("UPDATE query_anchors SET decayed=1 WHERE id=?", (a.id,))
    assert writer.query_anchor_revision() != before
    if writer is not reader:
        assert reader.query_anchor_revision() == before
        assert_parity(warm, [1.0, 0.0], [a.id])
    writer.connection.rollback()
    assert reader.query_anchor_revision() == before
    assert_parity(warm, [1.0, 0.0], [a.id])


def test_uncommitted_vectors_cannot_poison_a_reused_revision(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        anchor(store, "a", [1.0, 0.0])
        anchor(store, "b", [0.0, 1.0])
        warm = service(store)
        assert_parity(warm, [1.0, 0.0], ["a"])
        store.connection.execute(
            "UPDATE query_anchors SET embedding=? WHERE id='a'",
            (pack_chunk_embedding([0.0, 1.0]),),
        )
        assert_parity(warm, [1.0, 0.0], [])
        store.connection.rollback()
        with store.connection:
            store.connection.execute(
                "UPDATE query_anchors SET embedding=? WHERE id='b'",
                (pack_chunk_embedding([1.0, 0.0]),),
            )
        assert_parity(warm, [1.0, 0.0], ["a", "b"])


def test_recall_map_tracks_the_anchor_revision(stores):
    reader, writer = stores
    a = anchor(writer, "a", [1.0, 0.0])
    b = anchor(writer, "b", [0.0, 1.0])
    writer.decay_query_anchor(b.id, "inactive")
    builder = RecallMapBuilder(reader)
    before = builder._corpus_revision()
    writer.reinforce_query_anchor(a.id)
    assert builder._corpus_revision() == before
    writer.decay_query_anchor(a.id, "inactive")
    writer.reinforce_query_anchor(b.id)
    assert builder._corpus_revision() != before


def test_warm_recall_finds_the_revived_anchors_target(stores):
    reader, writer = stores
    a = anchor(writer, "a", [1.0, 0.0])
    b = anchor(writer, "b", [0.0, 1.0])
    targets = []
    for item, content in ((a, "cobalt prism"), (b, "amber cylinder")):
        node = writer.create_node(
            level="concept", content=content, context={"scope": "global"},
            embedding=[0.0, 0.0],
        )
        writer.upsert_query_anchor_edge(item.id, node.id, weight=1.0)
        targets.append(node.id)
    writer.decay_query_anchor(b.id, "inactive")
    warm = service(reader)

    def recall(retriever, query):
        return retriever.memory_recall(
            query, scope="global", depth=1, log_access=False, log_event=False,
        )

    assert [r.node.id for r in recall(warm, "past situation")] == [targets[0]]
    assert recall(warm, "restored situation") == []
    writer.decay_query_anchor(a.id, "inactive")
    writer.reinforce_query_anchor(b.id)
    assert recall(warm, "past situation") == recall(service(reader), "past situation") == []
    result = recall(warm, "restored situation")
    assert result == recall(service(reader), "restored situation")
    assert [r.node.id for r in result] == [targets[1]]
    assert result[0].graph_score > 0 and result[0].vector_score == result[0].bm25_score == 0


def test_existing_database_and_read_only_reader(tmp_path):
    path = tmp_path / "memory.sqlite3"
    with MemoryStore(path) as store:
        a = anchor(store, "a", [1.0, 0.0])
        with store.connection:
            for suffix in ("insert", "delete", "update"):
                store.connection.execute(f"DROP TRIGGER IF EXISTS query_anchor_corpus_{suffix}")
            store.connection.execute("DELETE FROM metadata WHERE key='query_anchor_revision'")
    with ReadOnlyAuditStore.open_path(path) as reader, sqlite3.connect(path) as writer:
        assert reader.query_anchor_revision() is None
        warm = service(reader)
        builder = RecallMapBuilder(reader)
        before = builder._corpus_revision()
        assert_parity(warm, [1.0, 0.0], [a.id])
        with writer:
            writer.execute("UPDATE query_anchors SET embedding=?", (pack_chunk_embedding([0.0, 1.0]),))
        assert_parity(warm, [0.0, 1.0], [a.id])
        assert builder._corpus_revision() != before
        original = reader.get_query_anchor(a.id)
    with MemoryStore(path) as store:
        assert store.get_query_anchor(a.id) == original
        assert store.query_anchor_revision() == 0
        store.decay_query_anchor(a.id, "inactive")
        after = store.query_anchor_revision()
        assert after > 0
    with MemoryStore(path) as store, ReadOnlyAuditStore.open_path(path) as reader:
        assert store.query_anchor_revision() == reader.query_anchor_revision() == after
        warm = service(reader)
        assert_parity(warm, [0.0, 1.0], [])
        store.reinforce_query_anchor(a.id)
        assert_parity(warm, [0.0, 1.0], [a.id])


def test_anchor_maintenance_allows_only_trigger_revision_writes(stores):
    reader, writer = stores
    install = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/backfill_query_anchors.py"))["install_anchor_only_authorizer"]
    warm = service(reader)
    assert matches(warm, [1.0, 0.0]) == []
    install(writer.connection)
    try:
        a = anchor(writer, "a", [1.0, 0.0])
        assert_parity(warm, [1.0, 0.0], [a.id])
        writer.decay_query_anchor(a.id, "inactive")
        assert_parity(warm, [1.0, 0.0], [])
        writer.reinforce_query_anchor(a.id)
        assert_parity(warm, [1.0, 0.0], [a.id])
        for statement in (
            "UPDATE metadata SET value='0' WHERE key='query_anchor_revision'",
            "DELETE FROM metadata",
            "UPDATE nodes SET decayed=1",
            "DROP TRIGGER query_anchor_corpus_update",
        ):
            with pytest.raises(sqlite3.DatabaseError):
                writer.connection.execute(statement)
    finally:
        writer.connection.set_authorizer(None)


@pytest.mark.parametrize("mutation", ["decay", "scope"])
def test_match_readback_excludes_anchor_removed_after_scan(stores, mutation):
    reader, writer = stores
    a = anchor(writer, "a", [1.0, 0.0])
    source = service(reader)._anchor_vector_source()
    if mutation == "decay":
        writer.decay_query_anchor(a.id, "inactive")
    else:
        with writer.connection:
            writer.connection.execute("UPDATE query_anchors SET scope='project:elsewhere'")
    assert qa.match_anchors(source, [1.0, 0.0], "global") == []
