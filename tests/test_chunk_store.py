"""Chunk-embedding storage: BLOB codec, table shape, write paths, migration.

Covers the schema-v6 half of the chunking work: the float32 little-endian codec,
the ``node_chunk_embeddings`` table and its constraints, the write paths that
must keep chunks in step with node content, the v5 -> v6 migration, the
explicitly-invoked ``nodes.embedding`` drop, and the read-only stores that never
run migrations.
"""

from __future__ import annotations

import sqlite3
import struct
import zlib
from pathlib import Path
from typing import Sequence

import pytest

from living_memory import replay
from living_memory.chunking import TextChunk, chunk_text
from living_memory.health_audit import ReadOnlyAuditStore
from living_memory.storage import (
    CHUNK_EMBEDDING_DTYPE,
    CHUNK_EMBEDDING_TABLE,
    SCHEMA_VERSION,
    MemoryStore,
    pack_chunk_embedding,
    unpack_chunk_embedding,
)

# Long enough to need several ~126-token windows, short enough to stay readable.
LONG_CONTENT = " ".join(
    f"paragraph {index} describes a distinct step of the deployment runbook "
    f"with enough words to fill out the token window properly"
    for index in range(60)
)


def stub_embedder(dimensions: int) -> tuple[object, list[list[str]]]:
    """A deterministic per-chunk encoder plus the log of what it was asked to embed.

    Vectors differ per chunk text, so a test can tell "the chunk was embedded"
    from "the node's single vector was copied into chunk 0".
    """

    calls: list[list[str]] = []

    def embed(texts: Sequence[str]) -> list[list[float]]:
        calls.append(list(texts))
        # crc32, not hash(): str hashing is salted per process, and the
        # idempotency test would then be comparing against a moving target.
        return [
            [float(zlib.crc32(text.encode("utf-8")) % 1000 + index) / 1000.0] * dimensions
            for index, text in enumerate(texts)
        ]

    return embed, calls


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    with MemoryStore(tmp_path / "memory.sqlite3") as opened:
        yield opened


# ---------------------------------------------------------------------------
# BLOB codec
# ---------------------------------------------------------------------------


def test_blob_roundtrip_is_exact_for_float32_representable_values() -> None:
    values = [0.0, -0.0, 1.0, -1.0, 0.5, -0.25, 3.25, 2.0**-14, -(2.0**16)]
    assert unpack_chunk_embedding(pack_chunk_embedding(values), len(values)) == values


def test_blob_roundtrip_is_exact_at_a_non_384_dimension() -> None:
    """Dimension is data, not a constant: 7 and 1536 must behave like 384."""

    for dimensions in (1, 7, 128, 1536):
        values = [float(index) / 64.0 for index in range(dimensions)]
        blob = pack_chunk_embedding(values)
        assert len(blob) == dimensions * 4
        assert unpack_chunk_embedding(blob, dimensions) == values


def test_blob_roundtrip_is_idempotent_for_values_needing_narrowing() -> None:
    """float64 -> float32 rounds once; encoding the result again changes nothing."""

    values = [0.1, 0.2, 1.0 / 3.0, 1e-8, 1.2345678901234]
    once = unpack_chunk_embedding(pack_chunk_embedding(values), len(values))
    assert once != values  # the narrowing really is lossy
    assert unpack_chunk_embedding(pack_chunk_embedding(once), len(once)) == once


def test_blob_byte_layout_is_little_endian_float32() -> None:
    """Assert the bytes themselves, not just that we can read our own writes."""

    blob = pack_chunk_embedding([1.0, -2.0, 0.5])
    assert blob == struct.pack("<3f", 1.0, -2.0, 0.5)
    assert blob[:4] == b"\x00\x00\x80\x3f"  # 1.0f, low byte first
    assert blob != struct.pack(">3f", 1.0, -2.0, 0.5)
    assert CHUNK_EMBEDDING_DTYPE == "<f4"


def test_unpack_rejects_a_length_that_contradicts_the_recorded_dimension() -> None:
    """Length must never be allowed to stand in for the recorded dimension."""

    blob = pack_chunk_embedding([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="expected"):
        unpack_chunk_embedding(blob, 4)
    with pytest.raises(ValueError, match="expected"):
        unpack_chunk_embedding(blob[:-1], 3)


def test_numpy_reads_the_blob_through_the_published_dtype() -> None:
    numpy = pytest.importorskip("numpy")
    values = [1.5, -0.25, 7.0]
    array = numpy.frombuffer(pack_chunk_embedding(values), dtype=CHUNK_EMBEDDING_DTYPE)
    assert array.tolist() == values


# ---------------------------------------------------------------------------
# Table shape
# ---------------------------------------------------------------------------


def test_schema_v6_creates_the_chunk_table_with_its_index(store: MemoryStore) -> None:
    # v6 objects must survive every later additive migration, so this pins the
    # floor rather than the current version (which test_storage.py owns).
    assert SCHEMA_VERSION >= 6
    columns = {
        row["name"]: row
        for row in store.connection.execute(
            f"PRAGMA table_info({CHUNK_EMBEDDING_TABLE})"
        ).fetchall()
    }
    assert columns["id"]["pk"] == 1
    for required in (
        "node_id",
        "chunk_index",
        "dimensions",
        "embedding",
        "token_start",
        "token_end",
        "content_fingerprint",
        "created_at",
        "updated_at",
    ):
        assert columns[required]["notnull"] == 1, required
    assert columns["embedding"]["type"] == "BLOB"

    foreign_keys = store.connection.execute(
        f"PRAGMA foreign_key_list({CHUNK_EMBEDDING_TABLE})"
    ).fetchall()
    assert [(row["table"], row["from"], row["to"]) for row in foreign_keys] == [
        ("nodes", "node_id", "id")
    ]

    indexes = {
        row["name"]
        for row in store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    }
    assert "idx_nodes_active_scope_level" in indexes


def test_chunk_scan_is_index_driven_not_a_table_scan(store: MemoryStore) -> None:
    plan = " ".join(
        str(row[3])
        for row in store.connection.execute(
            f"""
            EXPLAIN QUERY PLAN
            SELECT c.node_id, c.chunk_index, c.embedding
            FROM {CHUNK_EMBEDDING_TABLE} c JOIN nodes n ON n.id = c.node_id
            WHERE n.decayed = 0 AND n.scope = ?
            ORDER BY c.node_id, c.chunk_index
            """,
            ("global",),
        ).fetchall()
    )
    assert "idx_nodes_active_scope_level" in plan
    assert "SCAN n" not in plan


def test_ids_are_ulids_and_timestamps_are_utc(store: MemoryStore) -> None:
    node = store.create_node(level="concept", content="short note", embedding=[1.0, 0.0])
    row = store.connection.execute(
        f"SELECT * FROM {CHUNK_EMBEDDING_TABLE} WHERE node_id = ?", (node.id,)
    ).fetchone()
    assert len(str(row["id"])) == 26
    assert str(row["created_at"]).endswith("Z")
    assert str(row["updated_at"]).endswith("Z")


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def test_unique_constraint_rejects_a_duplicate_ordinal(store: MemoryStore) -> None:
    node = store.create_node(level="concept", content="short note", embedding=[1.0, 0.0])
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        store.connection.execute(
            f"""
            INSERT INTO {CHUNK_EMBEDDING_TABLE}
                (id, node_id, chunk_index, dimensions, embedding,
                 token_start, token_end, content_fingerprint, created_at, updated_at)
            VALUES ('DUPLICATE', ?, 0, 2, ?, 0, 2, 'fp', 'now', 'now')
            """,
            (node.id, pack_chunk_embedding([1.0, 0.0])),
        )


def test_foreign_key_rejects_a_chunk_with_no_parent_node(store: MemoryStore) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        store.connection.execute(
            f"""
            INSERT INTO {CHUNK_EMBEDDING_TABLE}
                (id, node_id, chunk_index, dimensions, embedding,
                 token_start, token_end, content_fingerprint, created_at, updated_at)
            VALUES ('ORPHAN', 'NO-SUCH-NODE', 0, 2, ?, 0, 2, 'fp', 'now', 'now')
            """,
            (pack_chunk_embedding([1.0, 0.0]),),
        )


def test_replace_rejects_sparse_or_out_of_order_ordinals(store: MemoryStore) -> None:
    node = store.create_node(level="concept", content="short note", embedding=[1.0, 0.0])
    gapped = [
        (TextChunk(text="a", chunk_index=0, token_start=0, token_end=1, char_start=0, char_end=1), [1.0]),
        (TextChunk(text="b", chunk_index=2, token_start=1, token_end=2, char_start=1, char_end=2), [0.5]),
    ]
    with pytest.raises(ValueError, match="dense and ordered"):
        store.replace_node_chunks(node.id, gapped)


# ---------------------------------------------------------------------------
# Write paths
# ---------------------------------------------------------------------------


def test_create_without_an_embedding_writes_no_chunks(store: MemoryStore) -> None:
    """remember/teach must not put the encoder on the write path."""

    embed, calls = stub_embedder(4)
    store.set_chunk_embedder(embed)
    trace = store.append_trace(LONG_CONTENT, {"scope": "global"})
    assert store.count_node_chunks(trace.id) == 0
    assert calls == []
    assert [node.id for node in store.list_unchunked_nodes()] == [trace.id]


def test_lazy_backfill_through_update_node_chunks_a_long_node(store: MemoryStore) -> None:
    """The recall path's _ensure_embedding -> update_node(embedding=) route."""

    embed, calls = stub_embedder(4)
    store.set_chunk_embedder(embed)
    trace = store.append_trace(LONG_CONTENT, {"scope": "global"})

    store.update_node(trace.id, embedding=[0.1] * 4)

    expected = chunk_text(LONG_CONTENT)
    assert len(expected) > 1, "fixture must exceed one window or it proves nothing"
    chunks = store.list_node_chunks(trace.id)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(expected)))
    assert [(chunk.token_start, chunk.token_end) for chunk in chunks] == [
        (window.token_start, window.token_end) for window in expected
    ]
    assert calls == [[window.text for window in expected]]
    assert {chunk.dimensions for chunk in chunks} == {4}
    assert store.node_chunks_are_current(trace.id)
    assert store.list_unchunked_nodes() == []


def test_a_single_window_node_reuses_the_vector_it_was_given(store: MemoryStore) -> None:
    """Short content is the common case; re-encoding it would be pure waste."""

    embed, calls = stub_embedder(3)
    store.set_chunk_embedder(embed)
    node = store.create_node(
        level="concept", content="one short window", embedding=[0.25, 0.5, 0.75]
    )
    chunks = store.list_node_chunks(node.id)
    assert len(chunks) == 1
    assert chunks[0].embedding == [0.25, 0.5, 0.75]
    assert calls == []


def test_chunk_dimension_is_recorded_not_assumed(store: MemoryStore) -> None:
    embed, _calls = stub_embedder(7)
    store.set_chunk_embedder(embed)
    store.update_node(
        store.append_trace(LONG_CONTENT, {"scope": "global"}).id, embedding=[0.0] * 7
    )
    assert store.chunk_embedding_dimensions() == (7,)
    for chunk in store.list_node_chunks(store.list_nodes(level="trace")[0].id):
        assert chunk.dimensions == 7
        assert len(chunk.embedding) == 7


def test_content_change_invalidates_stale_chunks(store: MemoryStore) -> None:
    embed, _calls = stub_embedder(4)
    store.set_chunk_embedder(embed)
    node = store.create_node(level="concept", content=LONG_CONTENT, embedding=[0.1] * 4)
    before = store.count_node_chunks(node.id)
    assert before > 1

    store.update_node(node.id, content="a completely different and much shorter concept")

    assert store.count_node_chunks(node.id) == 0, "stale chunks must not survive an edit"
    assert not store.node_chunks_are_current(node.id)
    assert [pending.id for pending in store.list_unchunked_nodes()] == [node.id]


def test_re_embedding_never_leaves_a_mix_of_old_and_new_chunks(store: MemoryStore) -> None:
    """Fewer chunks than last time must not leave the old tail behind."""

    embed, _calls = stub_embedder(4)
    store.set_chunk_embedder(embed)
    node = store.create_node(level="concept", content=LONG_CONTENT, embedding=[0.1] * 4)
    long_count = store.count_node_chunks(node.id)
    assert long_count > 1

    short = "now the concept is a single short sentence"
    updated = store.update_node(node.id, content=short, embedding=[0.2] * 4)

    chunks = store.list_node_chunks(node.id)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunk_text(short))))
    assert len(chunks) < long_count
    assert {chunk.content_fingerprint for chunk in chunks} == {
        store.connection.execute(
            "SELECT content_fingerprint FROM nodes WHERE id = ?", (updated.id,)
        ).fetchone()["content_fingerprint"]
    }
    assert store.node_chunks_are_current(node.id)


def test_re_embedding_unchanged_content_is_idempotent(store: MemoryStore) -> None:
    embed, _calls = stub_embedder(4)
    store.set_chunk_embedder(embed)
    node = store.create_node(level="concept", content=LONG_CONTENT, embedding=[0.1] * 4)
    first = [
        (chunk.chunk_index, chunk.dimensions, chunk.embedding, chunk.token_start, chunk.token_end)
        for chunk in store.list_node_chunks(node.id)
    ]

    for _ in range(2):
        store.update_node(node.id, embedding=[0.1] * 4)

    second = [
        (chunk.chunk_index, chunk.dimensions, chunk.embedding, chunk.token_start, chunk.token_end)
        for chunk in store.list_node_chunks(node.id)
    ]
    assert second == first
    assert store.count_node_chunks() == len(first)


def test_updates_that_cannot_change_the_text_leave_chunks_alone(store: MemoryStore) -> None:
    embed, calls = stub_embedder(4)
    store.set_chunk_embedder(embed)
    node = store.create_node(level="concept", content=LONG_CONTENT, embedding=[0.1] * 4)
    calls.clear()
    before = store.list_node_chunks(node.id)

    store.record_access(node.id)
    store.add_correction(node.id, old="x", new="y", by="tester")
    store.update_node(node.id, stats={"usefulness_score": 0.9})

    assert store.list_node_chunks(node.id) == before
    assert calls == []


def test_replace_node_chunks_is_the_offline_backfill_entry_point(store: MemoryStore) -> None:
    """The public write API a backfill drives with its own model and chunker."""

    trace = store.append_trace(LONG_CONTENT, {"scope": "global"})
    windows = chunk_text(LONG_CONTENT)
    vectors = [[float(index), 1.0, 0.0] for index in range(len(windows))]

    written = store.replace_node_chunks(trace.id, list(zip(windows, vectors)))

    assert written == len(windows)
    assert [chunk.embedding for chunk in store.list_node_chunks(trace.id)] == vectors
    assert store.node_chunks_are_current(trace.id)
    assert store.list_unchunked_nodes() == []

    # Re-running the same backfill converges rather than accumulating.
    assert store.replace_node_chunks(trace.id, list(zip(windows, vectors))) == len(windows)
    assert store.count_node_chunks(trace.id) == len(windows)

    assert store.delete_node_chunks(trace.id) == len(windows)
    assert store.count_node_chunks(trace.id) == 0
    with pytest.raises(KeyError):
        store.replace_node_chunks("NO-SUCH-NODE", list(zip(windows, vectors)))


def test_unchunkable_content_is_not_reported_as_pending_forever(store: MemoryStore) -> None:
    """A node the chunker can never window must not livelock a backfill loop."""

    embed, _calls = stub_embedder(4)
    store.set_chunk_embedder(embed)
    blank = store.append_trace("   \n\t ", {"scope": "global"})
    assert chunk_text(blank.content) == []

    store.update_node(blank.id, embedding=[0.1] * 4)

    assert store.count_node_chunks(blank.id) == 0
    assert blank.id not in {node.id for node in store.list_unchunked_nodes()}


def test_soft_deleted_nodes_keep_their_rows_but_leave_the_scan(store: MemoryStore) -> None:
    """Nothing hard-deletes chunks, so liveness has to come from the parent."""

    embed, _calls = stub_embedder(4)
    store.set_chunk_embedder(embed)
    node = store.create_node(level="concept", content=LONG_CONTENT, embedding=[0.1] * 4)
    written = store.count_node_chunks(node.id)

    store.soft_delete_node(node.id, reason="test")

    assert store.count_node_chunks(node.id) == written
    assert list(store.iter_chunk_embedding_rows()) == []
    assert store.chunk_embedding_dimensions() == ()
    assert len(list(store.iter_chunk_embedding_rows(include_decayed=True))) == written


# ---------------------------------------------------------------------------
# Bulk read contract
# ---------------------------------------------------------------------------


def test_scan_yields_node_ordinal_memoryview_grouped_and_ordered(store: MemoryStore) -> None:
    embed, _calls = stub_embedder(4)
    store.set_chunk_embedder(embed)
    first = store.create_node(
        level="concept", content=LONG_CONTENT, embedding=[0.1] * 4, context={"scope": "global"}
    )
    second = store.create_node(
        level="trace",
        content=LONG_CONTENT + " second",
        embedding=[0.2] * 4,
        context={"scope": "project:other"},
    )

    rows = list(store.iter_chunk_embedding_rows())
    assert {node_id for node_id, _index, _blob in rows} == {first.id, second.id}
    for node_id, index, blob in rows:
        assert isinstance(blob, memoryview)
        assert len(blob) == 4 * 4

    seen: dict[str, list[int]] = {}
    order: list[str] = []
    for node_id, index, _blob in rows:
        if node_id not in seen:
            seen[node_id] = []
            order.append(node_id)
        seen[node_id].append(index)
    assert len(order) == len(set(order)), "each node's chunks must arrive contiguously"
    for node_id, ordinals in seen.items():
        assert ordinals == list(range(len(ordinals)))

    scoped = list(store.iter_chunk_embedding_rows(scope="project:other"))
    assert {node_id for node_id, _index, _blob in scoped} == {second.id}
    assert store.chunk_embedding_dimensions(scope="project:other") == (4,)
    assert list(store.iter_chunk_embedding_rows(level="trace")) == scoped


def test_scan_blobs_decode_back_to_the_stored_vectors(store: MemoryStore) -> None:
    embed, _calls = stub_embedder(5)
    store.set_chunk_embedder(embed)
    node = store.create_node(level="concept", content=LONG_CONTENT, embedding=[0.5] * 5)

    by_ordinal = {chunk.chunk_index: chunk.embedding for chunk in store.list_node_chunks(node.id)}
    for _node_id, index, blob in store.iter_chunk_embedding_rows():
        assert unpack_chunk_embedding(blob, 5) == by_ordinal[index]


# ---------------------------------------------------------------------------
# Compatibility for the legacy vector readers
# ---------------------------------------------------------------------------


def test_legacy_similarity_and_evidence_still_work_with_chunks_present(
    store: MemoryStore,
) -> None:
    embed, _calls = stub_embedder(3)
    store.set_chunk_embedder(embed)
    target = store.create_node(
        level="concept", content="deployment runbook", embedding=[1.0, 0.0, 0.0]
    )
    store.create_node(level="concept", content="unrelated topic", embedding=[0.0, 1.0, 0.0])

    matches = store.find_similar_by_embedding([1.0, 0.0, 0.0], threshold=0.9)
    assert [node.id for node, _score in matches] == [target.id]
    assert store._has_vector_evidence("global") is True
    assert store._has_vector_evidence("project:empty") is False


def test_dropping_the_column_is_explicit_and_leaves_reads_served_by_chunks(
    tmp_path: Path,
) -> None:
    db = tmp_path / "memory.sqlite3"
    with MemoryStore(db) as store:
        embed, _calls = stub_embedder(3)
        store.set_chunk_embedder(embed)
        target = store.create_node(
            level="concept", content="deployment runbook", embedding=[1.0, 0.0, 0.0]
        )
        store.create_node(level="concept", content="unrelated topic", embedding=[0.0, 1.0, 0.0])
        assert "embedding" in _node_columns(store.connection)

        assert store.drop_node_embedding_column() is True
        assert store.drop_node_embedding_column() is False, "must be idempotent"
        assert "embedding" not in _node_columns(store.connection)

        # Served from chunks now, with the same answer as before the drop.
        matches = store.find_similar_by_embedding([1.0, 0.0, 0.0], threshold=0.9)
        assert [node.id for node, _score in matches] == [target.id]
        # Scope narrowing must still narrow once the scoring moved to chunks.
        assert store.find_similar_by_embedding(
            [1.0, 0.0, 0.0], threshold=0.9, exclude_scope="global"
        ) == []
        assert store.find_similar_by_embedding(
            [1.0, 0.0, 0.0], threshold=0.9, scope_prefix="project:"
        ) == []
        assert [
            node.id
            for node, _score in store.find_similar_by_embedding(
                [1.0, 0.0, 0.0], threshold=0.9, scope="global"
            )
        ] == [target.id]
        assert store._has_vector_evidence("global") is True
        # The legacy readers degrade to empty rather than raising.
        assert list(store.iter_embedding_rows()) == []
        assert store.list_unembedded_nodes() == []
        assert store.get_node(target.id).embedding is None

    # Reopening must not try to rebuild the index that named the dropped column.
    with MemoryStore(db) as reopened:
        assert "embedding" not in _node_columns(reopened.connection)
        assert reopened.count_node_chunks() > 0
        fresh = reopened.create_node(
            level="concept", content="written after the drop", embedding=[0.0, 0.0, 1.0]
        )
        assert reopened.count_node_chunks(fresh.id) == 1
        reopened.update_node(fresh.id, stats={"usefulness_score": 0.5})


def _node_columns(connection: sqlite3.Connection) -> set[str]:
    return {str(row["name"]) for row in connection.execute("PRAGMA table_info(nodes)").fetchall()}


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def _seed_v5_database(path: Path, *, rows: int = 40) -> list[tuple[str, str]]:
    """A populated schema-v5 file: nodes with JSON embeddings, no chunk table."""

    connection = sqlite3.connect(str(path))
    try:
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO metadata (key, value) VALUES ('schema_version', '5');
            CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                level TEXT NOT NULL,
                content TEXT NOT NULL,
                content_fingerprint TEXT,
                embedding TEXT,
                scope TEXT NOT NULL DEFAULT 'global',
                agent TEXT,
                task TEXT,
                context TEXT NOT NULL DEFAULT '{}',
                timestamp TEXT NOT NULL,
                decayed INTEGER NOT NULL DEFAULT 0,
                decay_reason TEXT,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed TEXT,
                usefulness_score REAL NOT NULL DEFAULT 0.0,
                confidence REAL NOT NULL DEFAULT 0.5,
                unique_agents INTEGER NOT NULL DEFAULT 0,
                temporal_hint TEXT,
                source_traces TEXT NOT NULL DEFAULT '[]',
                corrections TEXT NOT NULL DEFAULT '[]',
                provenance TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX idx_nodes_embedded_active_scope
                ON nodes(level, scope)
                WHERE decayed = 0 AND embedding IS NOT NULL;
            """
        )
        seeded: list[tuple[str, str]] = []
        for index in range(rows):
            node_id = f"LEGACY{index:020d}"
            content = f"legacy node {index}: {LONG_CONTENT[:200]}"
            connection.execute(
                """
                INSERT INTO nodes (id, level, content, embedding, scope, timestamp,
                                   created_at, updated_at)
                VALUES (?, 'trace', ?, ?, 'project:legacy',
                        '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
                """,
                (node_id, content, "[0.5, 0.25, 0.125]"),
            )
            seeded.append((node_id, content))
        connection.commit()
        return seeded
    finally:
        connection.close()


def test_populated_v5_database_migrates_to_v6(tmp_path: Path) -> None:
    db = tmp_path / "legacy.sqlite3"
    seeded = _seed_v5_database(db)

    with MemoryStore(db) as store:
        version = store.connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        # The stamp lands on the current SCHEMA_VERSION: every migration in the
        # chain runs on open, so a v5 file passes through v6 to whatever is
        # newest. What this test asserts about v6 is the objects below.
        assert version["value"] == str(SCHEMA_VERSION)

        tables = {
            row["name"]
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert CHUNK_EMBEDDING_TABLE in tables

        # Additive: every legacy row and its JSON vector survive untouched, so
        # the running server can keep reading while chunks get backfilled.
        assert store.trace_count() == len(seeded)
        for node_id, content in seeded:
            node = store.get_node(node_id)
            assert node.content == content
            assert node.embedding == [0.5, 0.25, 0.125]
        assert len(list(store.iter_embedding_rows(scope="project:legacy"))) == len(seeded)

        # And no chunks appear on their own: backfill is a separate, explicit step.
        assert store.count_node_chunks() == 0
        assert len(store.list_unchunked_nodes(limit=1000)) == len(seeded)

        embed, _calls = stub_embedder(3)
        store.set_chunk_embedder(embed)
        first_id = seeded[0][0]
        store.update_node(first_id, embedding=[0.5, 0.25, 0.125])
        assert store.count_node_chunks(first_id) > 0
        assert store.node_chunks_are_current(first_id)

    with MemoryStore(db) as reopened:
        assert reopened.count_node_chunks() > 0


def test_migration_is_idempotent_across_reopens(tmp_path: Path) -> None:
    db = tmp_path / "legacy.sqlite3"
    _seed_v5_database(db, rows=5)
    shapes = []
    for _ in range(3):
        with MemoryStore(db) as store:
            shapes.append(
                sorted(
                    (str(row["type"]), str(row["name"]), str(row["sql"] or ""))
                    for row in store.connection.execute(
                        "SELECT type, name, sql FROM sqlite_master"
                    ).fetchall()
                )
            )
    assert shapes[0] == shapes[1] == shapes[2]


# ---------------------------------------------------------------------------
# Read-only stores that never run migrations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["pre_v6", "post_v6"])
def test_read_only_audit_store_tolerates_both_shapes(tmp_path: Path, shape: str) -> None:
    db = tmp_path / f"{shape}.sqlite3"
    if shape == "pre_v6":
        _seed_v5_database(db, rows=5)
    else:
        with MemoryStore(db) as store:
            embed, _calls = stub_embedder(3)
            store.set_chunk_embedder(embed)
            store.create_node(
                level="concept",
                content=LONG_CONTENT,
                embedding=[1.0, 0.0, 0.0],
                context={"scope": "project:legacy"},
            )

    audit = ReadOnlyAuditStore.open_path(db)
    try:
        # This store never runs a migration, so on a pre-v6 file the chunk table
        # simply is not there. Every chunk-aware read must answer anyway.
        assert audit.list_unembedded_nodes() == []
        assert audit.list_unchunked_nodes() == []
        similar = audit.find_similar_by_embedding([1.0, 0.0, 0.0], threshold=0.9)
        # The v5 fixture holds only traces, so a concept-level query finds
        # nothing there; the v6 one holds the matching concept.
        assert [node.level for node, _score in similar] == (
            [] if shape == "pre_v6" else ["concept"]
        )
        # Both shapes keep nodes.embedding, so the evidence gate answers the
        # same on either side of the migration.
        assert audit._has_vector_evidence("project:legacy") is True
        assert audit._has_vector_evidence("project:absent") is False

        chunk_rows = list(audit.iter_chunk_embedding_rows())
        if shape == "pre_v6":
            assert chunk_rows == []
            assert audit.chunk_embedding_dimensions() == ()
            assert audit.count_node_chunks() == 0
        else:
            assert len(chunk_rows) > 1
            assert audit.chunk_embedding_dimensions() == (3,)
            assert audit.count_node_chunks() == len(chunk_rows)

        with pytest.raises(RuntimeError):
            audit.replace_node_chunks("whatever", [])
        with pytest.raises(RuntimeError):
            audit.drop_node_embedding_column()
    finally:
        audit.close()


@pytest.mark.parametrize("shape", ["pre_v6", "post_v6"])
def test_replay_open_readonly_tolerates_both_shapes(tmp_path: Path, shape: str) -> None:
    db = tmp_path / f"replay_{shape}.sqlite3"
    if shape == "pre_v6":
        _seed_v5_database(db, rows=5)
        connection = sqlite3.connect(str(db))
        connection.executescript(
            """
            CREATE TABLE recall_events (
                id TEXT PRIMARY KEY, query TEXT NOT NULL, scope TEXT NOT NULL,
                requested_scope TEXT NOT NULL, resolved_scopes TEXT NOT NULL DEFAULT '[]',
                depth TEXT, max_results INTEGER NOT NULL DEFAULT 10,
                results TEXT NOT NULL DEFAULT '[]', feedback_trace_id TEXT,
                feedback_applied_at TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE connections (
                id TEXT PRIMARY KEY, source_id TEXT NOT NULL, target_id TEXT NOT NULL,
                type TEXT NOT NULL
            );
            """
        )
        connection.commit()
        connection.close()
    else:
        with MemoryStore(db) as store:
            embed, _calls = stub_embedder(3)
            store.set_chunk_embedder(embed)
            store.create_node(
                level="concept",
                content=LONG_CONTENT,
                embedding=[1.0, 0.0, 0.0],
                context={"scope": "project:legacy"},
            )

    connection = replay.open_readonly(db)
    try:
        events, stats = replay.load_replay_events(connection)
        assert events == []
        assert stats["total_events"] == 0
        # replay reads nodes.embedding directly and never migrates. v6 is
        # additive, so that column is still there and the gate is unchanged.
        evidence = replay.snapshot_evidence(connection)
        assert evidence("project:legacy") == (True, False)
        assert evidence("project:absent") == (False, False)
    finally:
        connection.close()
