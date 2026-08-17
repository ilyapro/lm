"""Tests for ``scripts/backfill_chunk_embeddings.py``.

Every test runs against a fixture database built in ``tmp_path``. The live
database at ``~/.local/share/living-memory/global.sqlite3`` is never opened here,
not even read-only.

The properties under test are the ones that make an 8-minute write against a
database a live server is serving:

* resumable — an interrupted run plus a re-run equals an uninterrupted run, and
  the interruption is a real ``SIGKILL`` of a real subprocess, not a stand-in;
* atomic per node — no node is ever left holding a partial or mixed-generation
  chunk set, including when its content changes mid-run;
* additive — ``nodes.embedding`` comes out byte-identical, because the running
  server still reads it, and the column drop is a different subcommand;
* refuses to start without a WAL-correct backup;
* ``--dry-run`` writes nothing and predicts what the real run writes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import textwrap
from typing import Any, Iterator, Sequence

import pytest

from living_memory.chunking import chunk_text, reset_tokenizer_cache
from living_memory.storage import (
    CHUNK_EMBEDDING_TABLE,
    MemoryStore,
    pack_chunk_embedding,
    unpack_chunk_embedding,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "backfill_chunk_embeddings.py"

# Long enough to need many ~126-token windows; each node's text is distinct so
# chunk vectors differ between nodes under the hash encoder too.
_PARAGRAPHS = " ".join(
    f"paragraph {index} describes a distinct step of the deployment runbook with "
    f"enough words to fill out the whole token window properly"
    for index in range(40)
)
FIXTURE_NODES = 12


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "backfill_chunk_embeddings_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules[cls.__module__], so a
    # module executed without being registered there dies on its first dataclass.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bf = _load_script()


@pytest.fixture(autouse=True)
def _deterministic_encoder_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the hash encoder and the heuristic tokenizer for every test here.

    Both are deterministic and load no torch, which is what lets an interrupted
    run and an uninterrupted run be compared byte-for-byte. Pinned explicitly
    rather than inherited from the ambient environment so the comparison does not
    depend on how the suite was invoked.
    """

    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LIVING_MEMORY_CHUNK_TOKENIZER", "heuristic")
    reset_tokenizer_cache()
    yield
    reset_tokenizer_cache()


def child_env() -> dict[str, str]:
    """Environment for a subprocess run of the script under the same pinning."""

    env = dict(os.environ)
    env["LIVING_MEMORY_EMBEDDING_BACKEND"] = "hash"
    env["LIVING_MEMORY_CHUNK_TOKENIZER"] = "heuristic"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    return env


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _node_content(index: int) -> str:
    if index % 3 == 0:
        return f"node {index}: one short line about the deployment runbook"
    return f"node {index} :: {_PARAGRAPHS}"


@pytest.fixture()
def source_db(tmp_path: Path) -> Path:
    """A closed database of nodes that have a legacy vector but no chunks.

    That is the state the live database is in before the backfill: rows written
    by code that had no chunk table. Copies of this one file (never two separate
    builds) are what makes node ids identical across runs being compared.
    """

    path = tmp_path / "source.sqlite3"
    with MemoryStore(path) as store:
        for index in range(FIXTURE_NODES):
            store.create_node(
                level="trace",
                content=_node_content(index),
                context={"scope": "project:lm"},
                embedding=[0.1 + index / 100.0] * 8,
            )
        store.connection.execute(f"DELETE FROM {CHUNK_EMBEDDING_TABLE}")
        store.connection.commit()
        assert store.count_node_chunks() == 0
        assert len(store.list_unchunked_nodes(limit=100)) == FIXTURE_NODES
    return path


def fixture_copy(source: Path, name: str) -> Path:
    target = source.parent / name
    shutil.copyfile(source, target)
    return target


def backup_arg(db: Path, name: str = "backup.sqlite3") -> list[str]:
    return ["--backup", str(db.parent / name)]


def cli(*args: str) -> int:
    return bf.main(list(args))


def chunk_state(db: Path) -> list[tuple[Any, ...]]:
    """The chunk content that two runs must agree on, ids and clocks excluded."""

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            f"""
            SELECT node_id, chunk_index, dimensions, embedding, token_start,
                   token_end, content_fingerprint
            FROM {CHUNK_EMBEDDING_TABLE}
            ORDER BY node_id, chunk_index
            """
        ).fetchall()
    finally:
        conn.close()
    return [(*row[:3], bytes(row[3]), *row[4:]) for row in rows]


def chunk_identities(db: Path) -> set[tuple[str, str, str]]:
    """``(row id, created_at, updated_at)`` — changes iff a row was rewritten."""

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            f"SELECT id, created_at, updated_at FROM {CHUNK_EMBEDDING_TABLE}"
        ).fetchall()
    finally:
        conn.close()
    return {(str(row[0]), str(row[1]), str(row[2])) for row in rows}


def legacy_embeddings(db: Path) -> dict[str, str | None]:
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("SELECT id, embedding FROM nodes").fetchall()
    finally:
        conn.close()
    return {str(row[0]): row[1] for row in rows}


def fingerprints_per_node(db: Path) -> dict[str, set[str]]:
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            f"SELECT node_id, content_fingerprint FROM {CHUNK_EMBEDDING_TABLE}"
        ).fetchall()
    finally:
        conn.close()
    grouped: dict[str, set[str]] = {}
    for node_id, fingerprint in rows:
        grouped.setdefault(str(node_id), set()).add(str(fingerprint))
    return grouped


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_chunks_for(content: str) -> list[tuple[int, int, list[float]]]:
    """What a correct run must have stored for ``content``: spans and vectors.

    Derived independently of the script's loop — the chunker for the windows, the
    same deterministic encoder for the vectors, and the storage codec's float32
    narrowing — so it fails if chunks were cut from different text, stored out of
    order, or embedded from something other than their own window.
    """

    embedder, _info = bf.build_chunk_embedder(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        encode_batch=8,
        allow_fallback=True,
    )
    chunks = chunk_text(content)
    vectors = embedder([chunk.text for chunk in chunks])
    return [
        (
            chunk.token_start,
            chunk.token_end,
            unpack_chunk_embedding(pack_chunk_embedding(vector), len(vector)),
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def stored_chunks_for(db: Path, node_id: str) -> list[tuple[int, int, list[float]]]:
    with MemoryStore(db) as store:
        return [
            (chunk.token_start, chunk.token_end, chunk.embedding)
            for chunk in store.list_node_chunks(node_id)
        ]


def run_in_process(
    db: Path,
    *,
    wrap: Any = None,
    batch_nodes: int = 1,
    limit: int = 0,
) -> Any:
    """Drive ``run_backfill`` directly so a test can wrap the encoder.

    The encoder call is the seam an interruption or a concurrent content change
    is injected through: it happens after the page's content has been read and
    before any of that page's chunks are committed.
    """

    embedder, info = bf.build_chunk_embedder(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        encode_batch=8,
        allow_fallback=True,
    )
    summary = bf.BackfillSummary(mode="backfill", db=str(db), started_at="fixture")
    store = bf.MemoryStore(db)
    try:
        bf.install_chunk_only_authorizer(store.connection)
        try:
            return bf.run_backfill(
                store,
                embedder=wrap(embedder) if wrap else embedder,
                embedder_info=info,
                summary=summary,
                batch_nodes=batch_nodes,
                limit=limit,
                quiet=True,
            )
        finally:
            store.connection.set_authorizer(None)
    finally:
        store.close()


def complete_run(db: Path, name: str = "backup-complete.sqlite3") -> int:
    return cli(
        "backfill",
        "--db",
        str(db),
        *backup_arg(db, name),
        "--allow-fallback-embeddings",
        "--quiet",
    )


# ---------------------------------------------------------------------------
# Baseline: a full run
# ---------------------------------------------------------------------------


def test_full_run_chunks_every_node_and_leaves_nodes_untouched(source_db: Path) -> None:
    db = fixture_copy(source_db, "run.sqlite3")
    before = legacy_embeddings(db)

    assert complete_run(db) == 0

    report = bf.verify_report(db, scope=None, level=None)
    assert report["pending_nodes"] == 0
    assert report["covered_nodes"] == FIXTURE_NODES
    assert report["nodes_with_chunks"] == FIXTURE_NODES
    assert report["inconsistent_nodes"] == 0
    assert report["ok"] is True
    # Additive: the column the running server reads is still there, unmodified.
    assert report["node_embedding_column_present"] is True
    assert legacy_embeddings(db) == before

    # Chunk boundaries are the chunker's, not this script's idea of them, and
    # each window's vector is that window's own encoding.
    expected = sum(len(chunk_text(_node_content(i))) for i in range(FIXTURE_NODES))
    assert report["chunk_rows"] == expected
    with MemoryStore(db) as store:
        nodes = {node.content: node.id for node in store.list_nodes(limit=100)}
    for index in (0, 1, FIXTURE_NODES - 1):
        content = _node_content(index)
        assert stored_chunks_for(db, nodes[content]) == expected_chunks_for(content)


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------


KILL_DRIVER = textwrap.dedent(
    """
    import importlib.util, os, signal, sys

    spec = importlib.util.spec_from_file_location("bf_child", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    sys.modules["bf_child"] = module
    spec.loader.exec_module(module)

    kill_after = int(os.environ["KILL_AFTER_NODES"])
    build = module.build_chunk_embedder
    seen = {"calls": 0}

    def patched(**kwargs):
        embedder, info = build(**kwargs)

        def wrapped(texts):
            seen["calls"] += 1
            if seen["calls"] > kill_after:
                # A real, uncatchable kill at a deterministic point: after
                # `kill_after` nodes have committed, while the next one is
                # between "content read" and "chunks written".
                sys.stdout.flush()
                os.kill(os.getpid(), signal.SIGKILL)
            return embedder(texts)

        return wrapped, info

    module.build_chunk_embedder = patched
    raise SystemExit(module.main(sys.argv[2:]))
    """
)


def test_sigkill_mid_run_then_resume_matches_an_uninterrupted_run(
    source_db: Path, tmp_path: Path
) -> None:
    reference = fixture_copy(source_db, "reference.sqlite3")
    assert complete_run(reference, "backup-reference.sqlite3") == 0
    expected_state = chunk_state(reference)
    assert expected_state

    interrupted = fixture_copy(source_db, "interrupted.sqlite3")
    driver = tmp_path / "kill_driver.py"
    driver.write_text(KILL_DRIVER, encoding="utf-8")
    env = child_env()
    env["KILL_AFTER_NODES"] = "3"
    proc = subprocess.run(
        [
            sys.executable,
            str(driver),
            str(SCRIPT_PATH),
            "backfill",
            "--db",
            str(interrupted),
            "--backup",
            str(tmp_path / "backup-interrupted.sqlite3"),
            "--allow-fallback-embeddings",
            "--batch-size",
            "1",
            "--quiet",
        ],
        env=env,
        capture_output=True,
        timeout=300,
    )

    # The child really was killed, not merely stopped early.
    assert proc.returncode == -signal.SIGKILL, proc.stderr.decode()

    killed_state = chunk_state(interrupted)
    assert 0 < len(killed_state) < len(expected_state)
    mid_report = bf.verify_report(interrupted, scope=None, level=None)
    # The property a kill must not be able to break, checked while the database
    # is still in the killed state.
    assert mid_report["inconsistent_nodes"] == 0
    assert mid_report["pending_nodes"] == FIXTURE_NODES - 3
    assert all(len(v) == 1 for v in fingerprints_per_node(interrupted).values())

    assert complete_run(interrupted, "backup-resume.sqlite3") == 0
    assert chunk_state(interrupted) == expected_state


def test_limited_pass_then_resume_matches_an_uninterrupted_run(
    source_db: Path,
) -> None:
    reference = fixture_copy(source_db, "reference-limit.sqlite3")
    assert complete_run(reference, "backup-ref-limit.sqlite3") == 0
    expected_state = chunk_state(reference)

    partial = fixture_copy(source_db, "partial.sqlite3")
    first = run_in_process(partial, batch_nodes=4, limit=5)
    assert first.nodes_written == 5
    assert first.pending_after == FIXTURE_NODES - 5
    assert first.complete is False

    second = run_in_process(partial, batch_nodes=4)
    assert second.nodes_written == FIXTURE_NODES - 5
    assert second.pending_after == 0
    assert second.complete is True
    assert chunk_state(partial) == expected_state


def test_rerun_is_idempotent_with_no_duplicates_and_no_rewrites(
    source_db: Path,
) -> None:
    db = fixture_copy(source_db, "idempotent.sqlite3")
    assert complete_run(db) == 0
    state = chunk_state(db)
    identities = chunk_identities(db)
    legacy = legacy_embeddings(db)

    second = run_in_process(db, batch_nodes=4)

    assert second.nodes_processed == 0
    assert second.chunks_written == 0
    assert second.chunk_writes == 0
    assert chunk_state(db) == state
    # Same row ids and same updated_at: untouched, not rewritten to equal values.
    assert chunk_identities(db) == identities
    assert legacy_embeddings(db) == legacy


# ---------------------------------------------------------------------------
# Content changing under the run
# ---------------------------------------------------------------------------


def test_content_changed_mid_run_ends_with_one_consistent_generation(
    source_db: Path,
) -> None:
    db = fixture_copy(source_db, "mutating.sqlite3")
    with MemoryStore(db) as store:
        target = store.list_unchunked_nodes(limit=1)[0]
    new_content = "rewritten while the backfill was encoding it :: " + _PARAGRAPHS
    new_fingerprint = hashlib.sha256(new_content.encode("utf-8")).hexdigest()

    def wrap(embedder: Any) -> Any:
        state = {"done": False}

        def wrapped(texts: Sequence[str]) -> Any:
            if not state["done"]:
                state["done"] = True
                # A concurrent writer, on its own connection, exactly in the
                # window between this node's content being read and its chunks
                # being committed.
                side = sqlite3.connect(db)
                try:
                    side.execute(
                        "UPDATE nodes SET content = ?, content_fingerprint = ? "
                        "WHERE id = ?",
                        (new_content, new_fingerprint, target.id),
                    )
                    side.commit()
                finally:
                    side.close()
            return embedder(texts)

        return wrapped

    summary = run_in_process(db, wrap=wrap, batch_nodes=1)

    # Exactly one generation per node, and for the mutated node the stored
    # windows and vectors are the NEW content's — not old chunks relabelled with
    # the new fingerprint, which is what makes a chunk set quietly wrong.
    per_node = fingerprints_per_node(db)
    assert all(len(values) == 1 for values in per_node.values())
    assert per_node[target.id] == {new_fingerprint}
    assert stored_chunks_for(db, target.id) == expected_chunks_for(new_content)
    assert stored_chunks_for(db, target.id) != expected_chunks_for(target.content)

    assert summary.nodes_rechunked_after_content_change == 1
    assert summary.pending_after == 0
    with MemoryStore(db) as store:
        assert store.node_chunks_are_current(target.id)
    assert bf.verify_report(db, scope=None, level=None)["inconsistent_nodes"] == 0


def test_a_node_deleted_mid_run_does_not_abandon_the_rest(source_db: Path) -> None:
    db = fixture_copy(source_db, "vanishing.sqlite3")
    with MemoryStore(db) as store:
        doomed = store.list_unchunked_nodes(limit=1)[0]

    def wrap(embedder: Any) -> Any:
        state = {"done": False}

        def wrapped(texts: Sequence[str]) -> Any:
            vectors = embedder(texts)
            if not state["done"]:
                state["done"] = True
                # Hard-deleted after its content was read and encoded, so the
                # write is the first thing to notice.
                side = sqlite3.connect(db)
                try:
                    side.execute("DELETE FROM nodes WHERE id = ?", (doomed.id,))
                    side.commit()
                finally:
                    side.close()
            return vectors

        return wrapped

    summary = run_in_process(db, wrap=wrap, batch_nodes=1)

    assert summary.nodes_vanished == 1
    assert summary.nodes_written == FIXTURE_NODES - 1
    assert summary.pending_after == 0


def test_a_node_that_yields_no_chunks_is_skipped_instead_of_looping(
    source_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pending list keeps returning it, so the loop must remember it."""

    db = fixture_copy(source_db, "no-chunks.sqlite3")
    with MemoryStore(db) as store:
        stubborn = store.list_unchunked_nodes(limit=1)[0]

    real_chunk_text = bf.chunk_text

    def fake_chunk_text(text: str, **kwargs: Any) -> Any:
        if text == stubborn.content:
            return []
        return real_chunk_text(text, **kwargs)

    monkeypatch.setattr(bf, "chunk_text", fake_chunk_text)

    summary = run_in_process(db, batch_nodes=3)

    assert summary.nodes_skipped_no_chunks == 1
    assert summary.nodes_written == FIXTURE_NODES - 1
    # Honestly reported as still pending rather than silently called done.
    assert summary.pending_after == 1
    assert summary.complete is False


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing_and_predicts_the_real_run(source_db: Path) -> None:
    db = fixture_copy(source_db, "dry.sqlite3")
    digest = file_digest(db)
    summary_path = db.parent / "dry.json"

    assert cli("backfill", "--db", str(db), "--dry-run", "--json", str(summary_path)) == 0

    assert file_digest(db) == digest
    # SQLite materializes empty -wal/-shm sidecars even for a read-only WAL
    # database. Empty is the point: no frame was ever committed.
    wal = db.parent / "dry.sqlite3-wal"
    assert not wal.exists() or wal.stat().st_size == 0
    with MemoryStore(db) as store:
        # Reopening migrates, so ask before that: the count must be zero either
        # way, and the pending set must be untouched.
        assert store.count_node_chunks() == 0
        assert len(store.list_unchunked_nodes(limit=100)) == FIXTURE_NODES

    dry = json.loads(summary_path.read_text(encoding="utf-8"))
    real = fixture_copy(source_db, "dry-real.sqlite3")
    assert complete_run(real, "backup-dry-real.sqlite3") == 0
    report = bf.verify_report(real, scope=None, level=None)

    assert dry["pending_before"] == FIXTURE_NODES
    assert dry["chunks_written"] == report["chunk_rows"]
    assert dry["chunk_bytes_written"] == report["chunk_bytes"]


def test_dry_run_needs_no_backup_but_a_real_run_does(source_db: Path) -> None:
    db = fixture_copy(source_db, "dry-no-backup.sqlite3")
    assert cli("backfill", "--db", str(db), "--dry-run", "--quiet") == 0
    assert cli("backfill", "--db", str(db), "--allow-fallback-embeddings") == 2


def test_dry_run_reports_work_on_a_database_with_no_chunk_table(
    source_db: Path,
) -> None:
    """The live database is pre-v6 until the first real run migrates it."""

    db = fixture_copy(source_db, "pre-v6.sqlite3")
    conn = sqlite3.connect(db)
    try:
        conn.execute(f"DROP TABLE {CHUNK_EMBEDDING_TABLE}")
        conn.commit()
    finally:
        conn.close()

    summary = bf.dry_run(
        db,
        scope=None,
        level=None,
        limit=0,
        dimensions=None,
        assumed_rate=90.0,
        progress_interval=1000.0,
        quiet=True,
    )
    assert summary.pending_before == FIXTURE_NODES
    assert summary.chunks_written == sum(
        len(chunk_text(_node_content(i))) for i in range(FIXTURE_NODES)
    )
    conn = sqlite3.connect(db)
    try:
        present = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (CHUNK_EMBEDDING_TABLE,),
        ).fetchone()
    finally:
        conn.close()
    assert present is None, "dry run must not migrate the database it inspects"


# ---------------------------------------------------------------------------
# Backup gate
# ---------------------------------------------------------------------------


def test_refuses_to_run_without_a_backup_and_writes_nothing(source_db: Path) -> None:
    db = fixture_copy(source_db, "no-backup.sqlite3")
    digest = file_digest(db)

    assert cli("backfill", "--db", str(db), "--allow-fallback-embeddings") == 2

    assert file_digest(db) == digest
    with MemoryStore(db) as store:
        assert store.count_node_chunks() == 0


def test_refuses_a_missing_or_incomplete_existing_backup(source_db: Path) -> None:
    db = fixture_copy(source_db, "bad-backup.sqlite3")
    missing = db.parent / "nope.sqlite3"
    assert (
        cli(
            "backfill",
            "--db",
            str(db),
            "--existing-backup",
            str(missing),
            "--allow-fallback-embeddings",
        )
        == 2
    )

    # What a `cp` taken without the -wal looks like: a valid database missing the
    # newest rows.
    stale = db.parent / "stale-backup.sqlite3"
    shutil.copyfile(db, stale)
    conn = sqlite3.connect(stale)
    try:
        conn.execute("DELETE FROM nodes WHERE id IN (SELECT id FROM nodes LIMIT 2)")
        conn.commit()
    finally:
        conn.close()

    assert (
        cli(
            "backfill",
            "--db",
            str(db),
            "--existing-backup",
            str(stale),
            "--allow-fallback-embeddings",
        )
        == 2
    )
    with MemoryStore(db) as store:
        assert store.count_node_chunks() == 0


def test_refuses_to_overwrite_an_existing_backup_unless_told_to(
    source_db: Path,
) -> None:
    db = fixture_copy(source_db, "overwrite.sqlite3")
    target = db.parent / "occupied.sqlite3"
    target.write_bytes(b"not a database")

    assert (
        cli(
            "backfill",
            "--db",
            str(db),
            "--backup",
            str(target),
            "--allow-fallback-embeddings",
        )
        == 2
    )
    assert target.read_bytes() == b"not a database"

    assert (
        cli(
            "backfill",
            "--db",
            str(db),
            "--backup",
            str(target),
            "--overwrite-backup",
            "--allow-fallback-embeddings",
            "--quiet",
        )
        == 0
    )


def test_the_backup_it_takes_includes_transactions_still_in_the_wal(
    source_db: Path,
) -> None:
    """Why ``cp`` is not a valid backup here, as an assertion rather than a claim."""

    db = fixture_copy(source_db, "wal.sqlite3")
    store = MemoryStore(db)
    try:
        # Hold the connection open with checkpointing disabled, so the new row
        # exists only in the -wal — the live database's steady state, at 159 MB
        # of WAL as measured on 2026-08-17.
        store.connection.execute("PRAGMA wal_autocheckpoint = 0")
        store.create_node(
            level="trace", content="written into the wal only", embedding=[0.5] * 8
        )
        naive_copy = db.parent / "naive-cp.sqlite3"
        shutil.copyfile(db, naive_copy)  # no -wal: exactly the wrong way

        api_copy = db.parent / "api-backup.sqlite3"
        bf.take_backup(db, api_copy, overwrite=False)
    finally:
        store.close()

    def node_count(path: Path) -> int:
        conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        try:
            return int(conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])
        finally:
            conn.close()

    assert node_count(naive_copy) == FIXTURE_NODES
    assert node_count(api_copy) == FIXTURE_NODES + 1


def test_refuses_a_backup_that_is_the_database_itself(source_db: Path) -> None:
    db = fixture_copy(source_db, "self-backup.sqlite3")
    assert (
        cli(
            "backfill",
            "--db",
            str(db),
            "--existing-backup",
            str(db),
            "--allow-fallback-embeddings",
        )
        == 2
    )


def test_refuses_hash_fallback_vectors_unless_explicitly_allowed(
    source_db: Path,
) -> None:
    """Filling the chunk table from a different vector space is invisible later."""

    db = fixture_copy(source_db, "fallback.sqlite3")
    assert cli("backfill", "--db", str(db), *backup_arg(db, "b-fallback.sqlite3")) == 2
    with MemoryStore(db) as store:
        assert store.count_node_chunks() == 0


# ---------------------------------------------------------------------------
# Live-server safety
# ---------------------------------------------------------------------------


def test_the_authorizer_denies_every_write_outside_the_chunk_table(
    source_db: Path,
) -> None:
    db = fixture_copy(source_db, "authorized.sqlite3")
    store = MemoryStore(db)
    try:
        bf.install_chunk_only_authorizer(store.connection)
        for sql in (
            "UPDATE nodes SET embedding = NULL",
            "UPDATE nodes SET content = 'x'",
            "DELETE FROM nodes",
            "ALTER TABLE nodes DROP COLUMN embedding",
            f"DROP TABLE {CHUNK_EMBEDDING_TABLE}",
            "INSERT INTO metadata (key, value) VALUES ('k', 'v')",
        ):
            with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
                store.connection.execute(sql)

        # Reads and chunk writes still work.
        assert store.connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        assert store.delete_node_chunks("no-such-node") == 0
        store.connection.set_authorizer(None)
    finally:
        store.close()


def test_backfill_never_drops_the_legacy_column(source_db: Path) -> None:
    db = fixture_copy(source_db, "keeps-column.sqlite3")
    before = legacy_embeddings(db)

    assert complete_run(db) == 0

    conn = sqlite3.connect(db)
    try:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(nodes)")}
    finally:
        conn.close()
    assert "embedding" in columns
    assert legacy_embeddings(db) == before


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def test_verify_detects_a_mixed_generation_node(source_db: Path) -> None:
    """Otherwise every ``inconsistent_nodes == 0`` assertion above is vacuous."""

    db = fixture_copy(source_db, "mixed.sqlite3")
    assert complete_run(db) == 0
    assert bf.verify_report(db, scope=None, level=None)["inconsistent_nodes"] == 0

    conn = sqlite3.connect(db)
    try:
        conn.execute(
            f"UPDATE {CHUNK_EMBEDDING_TABLE} SET content_fingerprint = 'other' "
            "WHERE chunk_index = 1 AND node_id = "
            f"(SELECT node_id FROM {CHUNK_EMBEDDING_TABLE} WHERE chunk_index = 1 LIMIT 1)"
        )
        conn.commit()
    finally:
        conn.close()

    report = bf.verify_report(db, scope=None, level=None)
    assert report["inconsistent_nodes"] == 1
    assert report["ok"] is False


def test_verify_is_read_only_and_reports_an_unfinished_database(
    source_db: Path,
) -> None:
    db = fixture_copy(source_db, "verify-ro.sqlite3")
    digest = file_digest(db)

    assert cli("verify", "--db", str(db)) == 1  # nothing chunked yet

    assert file_digest(db) == digest
    report = bf.verify_report(db, scope=None, level=None)
    assert report["pending_nodes"] == FIXTURE_NODES
    assert report["covered_nodes"] == 0
    assert report["ok"] is False


# ---------------------------------------------------------------------------
# The column drop: a separate, explicit step
# ---------------------------------------------------------------------------


def test_column_drop_needs_yes_a_backup_and_full_coverage(source_db: Path) -> None:
    db = fixture_copy(source_db, "drop.sqlite3")

    # No --yes.
    assert cli("drop-embedding-column", "--db", str(db), *backup_arg(db, "d1.sqlite3")) == 2
    # No backup.
    assert cli("drop-embedding-column", "--db", str(db), "--yes") == 2
    # Coverage is empty: dropping now would empty the vector channel.
    assert (
        cli(
            "drop-embedding-column",
            "--db",
            str(db),
            "--yes",
            *backup_arg(db, "d2.sqlite3"),
        )
        == 2
    )
    conn = sqlite3.connect(db)
    try:
        assert "embedding" in {
            str(row[1]) for row in conn.execute("PRAGMA table_info(nodes)")
        }
    finally:
        conn.close()

    assert complete_run(db) == 0
    result = db.parent / "drop.json"
    assert (
        cli(
            "drop-embedding-column",
            "--db",
            str(db),
            "--yes",
            *backup_arg(db, "d3.sqlite3"),
            "--json",
            str(result),
        )
        == 0
    )

    conn = sqlite3.connect(db)
    try:
        assert "embedding" not in {
            str(row[1]) for row in conn.execute("PRAGMA table_info(nodes)")
        }
    finally:
        conn.close()
    assert json.loads(result.read_text(encoding="utf-8"))["dropped"] is True
    # Idempotent, and the chunks survive the drop.
    assert (
        cli(
            "drop-embedding-column",
            "--db",
            str(db),
            "--yes",
            *backup_arg(db, "d4.sqlite3"),
        )
        == 0
    )
    assert bf.verify_report(db, scope=None, level=None)["ok"] is True


def test_dropped_database_still_reopens_and_backfills(source_db: Path) -> None:
    """The drop must not leave a file the next MemoryStore cannot open."""

    db = fixture_copy(source_db, "reopen.sqlite3")
    assert complete_run(db) == 0
    assert (
        cli(
            "drop-embedding-column",
            "--db",
            str(db),
            "--yes",
            *backup_arg(db, "r1.sqlite3"),
        )
        == 0
    )

    with MemoryStore(db) as store:
        node = store.create_node(
            level="trace", content="written after the drop " + _PARAGRAPHS
        )
        assert store.list_unchunked_nodes(limit=10)[0].id == node.id

    summary = run_in_process(db, batch_nodes=4)
    assert summary.nodes_written == 1
    assert summary.pending_after == 0
