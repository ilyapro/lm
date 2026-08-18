"""Tests for ``scripts/backfill_query_anchors.py``.

Every test runs against a fixture database built in ``tmp_path``. The live
database at ``~/.local/share/living-memory/global.sqlite3`` is never opened
here, not even read-only.

The properties under test are the ones that make a long write against a
database a live server is serving:

* it builds anchors **only** from grounded consumptions, in the consuming
  event's scope, with edges only to the nodes that event actually used;
* resumable — an interrupted run plus a re-run equals an uninterrupted run, and
  the interruption is a real ``SIGKILL`` of a real subprocess, not a stand-in;
* it writes **nothing** outside ``query_anchors``/``query_anchor_edges``, and
  that is enforced by a SQLite authorizer rather than by review;
* ``--until`` really is a cutoff, which is the whole basis of a leak-free
  evaluation build;
* it refuses to start without a WAL-correct backup;
* ``--dry-run`` writes nothing and bounds what the real run writes;
* ``verify`` fails on each defect it claims to catch, and passes after the
  sweep repairs the one that is repairable.
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

from living_memory.query_anchors import ANCHOR_EDGE_WEIGHT
from living_memory.storage import (
    QUERY_ANCHOR_EDGE_TABLE,
    QUERY_ANCHOR_TABLE,
    MemoryStore,
    recall_fingerprint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "backfill_query_anchors.py"

SCOPE = "project:demo"

#: Six distinct operator questions. Distinct in *tokens*, so the deterministic
#: hash encoder the suite runs under keeps them far apart (measured: every
#: off-diagonal cosine below 0.5, against a 0.95 dedup threshold) and dedup
#: behaviour in these tests is the script's, not the fixture encoder's.
QUERIES = (
    "как перезапустить сервер living memory",
    "поревьювь мердж реквест 9806",
    "где лежит runbook по чанкам",
    "почему падает тест retrieval harness",
    "какие веса каналов сейчас",
    "что делать при busy timeout sqlite",
)
GROUNDED_EVENTS = len(QUERIES)


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "backfill_query_anchors_under_test", SCRIPT_PATH
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
    """Pin the hash encoder for every test here.

    It is deterministic and loads no torch, which is what lets an interrupted
    run and an uninterrupted run be compared byte-for-byte. Pinned explicitly
    rather than inherited from the ambient environment so the comparison does
    not depend on how the suite was invoked.
    """

    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    yield


def child_env() -> dict[str, str]:
    """Environment for a subprocess run of the script under the same pinning."""

    env = dict(os.environ)
    env["LIVING_MEMORY_EMBEDDING_BACKEND"] = "hash"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    return env


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _target_content(index: int) -> str:
    return (
        f"узел {index}: чтобы поднять сервис, выполните systemctl restart "
        f"living-memory и проверьте boot_id маркер-каппа-{index}"
    )


DISTRACTOR = (
    "совершенно посторонний узел про миграцию таблиц postgres и партиционирование"
)

GLOBAL_CONTENT = "глобальный узел: как поднять сервис на другой машине маркер-омега"


def _record(
    store: MemoryStore,
    *,
    query: str,
    scope: str,
    result_ids: Sequence[str],
    trace_content: str,
    created_at: str,
) -> str:
    """One consumed recall event with a fixed ``created_at``.

    ``record_recall_event`` stamps the wall clock, and every timestamp in these
    tests is load-bearing (``--until``, and the fact that anchors carry their
    source event's clock), so it is overwritten here deliberately.
    """

    event = store.record_recall_event(
        query=query,
        scope=scope,
        requested_scope=scope,
        results=[{"node_id": node_id, "score": 0.9} for node_id in result_ids],
    )
    trace = store.create_node(
        level="trace", content=trace_content, context={"scope": scope}
    )
    store.mark_recall_event_feedback(event.id, trace.id)
    store.connection.execute(
        "UPDATE recall_events SET created_at = ? WHERE id = ?", (created_at, event.id)
    )
    store.connection.commit()
    return event.id


@pytest.fixture()
def source_db(tmp_path: Path) -> Path:
    """Consumed recall history with no anchors: the pre-backfill live state.

    Copies of this one file (never two separate builds) are what makes node and
    event ids identical across runs being compared.

    Contents, all of which some test depends on:

    * six grounded consumptions, one per distinct query, January .. June;
    * one repeat of the first query in July, grounding on the same node;
    * one consumption whose trace used none of what it was shown;
    * one consumption of the first query in ``global`` rather than ``demo``;
    * one event carrying a second, ungrounded result alongside its grounded one.
    """

    path = tmp_path / "source.sqlite3"
    with MemoryStore(path) as store:
        targets = [
            store.create_node(
                level="trace", content=_target_content(i), context={"scope": SCOPE}
            )
            for i in range(GROUNDED_EVENTS)
        ]
        distractor = store.create_node(
            level="trace", content=DISTRACTOR, context={"scope": SCOPE}
        )
        for index, query in enumerate(QUERIES):
            results = [targets[index].id]
            if index == 1:
                # Delivered but never used: an edge to it would be a lie.
                results.append(distractor.id)
            _record(
                store,
                query=query,
                scope=SCOPE,
                result_ids=results,
                trace_content=f"сделал так: {targets[index].content} — готово",
                created_at=f"2026-0{index + 1}-01T10:00:00Z",
            )
        _record(
            store,
            query=QUERIES[0],
            scope=SCOPE,
            result_ids=[targets[0].id],
            trace_content=f"снова то же самое: {targets[0].content}",
            created_at="2026-07-01T10:00:00Z",
        )
        _record(
            store,
            query="что там с погодой на выходных",
            scope=SCOPE,
            result_ids=[targets[2].id],
            trace_content="написал заметку про дождь и ветер, ничего из показанного не пригодилось",
            created_at="2026-07-02T10:00:00Z",
        )
        shared = store.create_node(
            level="trace",
            content=GLOBAL_CONTENT,
            context={"scope": "global"},
        )
        _record(
            store,
            query=QUERIES[0],
            scope="global",
            result_ids=[shared.id],
            trace_content=f"в другом скоупе: {shared.content}",
            created_at="2026-07-03T10:00:00Z",
        )
        # Leave the file at v6, without the anchor tables: that is the state
        # the live database is in before this migration, and it makes the
        # first real run exercise the v6 -> v7 migration rather than skip it.
        store.connection.execute(f"DROP TABLE {QUERY_ANCHOR_EDGE_TABLE}")
        store.connection.execute(f"DROP TABLE {QUERY_ANCHOR_TABLE}")
        store.connection.execute(
            "UPDATE metadata SET value = '6' WHERE key = 'schema_version'"
        )
        store.connection.commit()
    return path


def fixture_copy(source: Path, name: str) -> Path:
    target = source.parent / name
    shutil.copyfile(source, target)
    return target


def cli(*args: str) -> int:
    return bf.main(list(args))


def complete_run(db: Path, name: str = "backup.sqlite3", *extra: str) -> int:
    return cli(
        "backfill",
        "--db",
        str(db),
        "--backup",
        str(db.parent / name),
        "--allow-fallback-embeddings",
        "--quiet",
        *extra,
    )


def _rows(db: Path, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(db)
    try:
        return [tuple(row) for row in conn.execute(sql, list(params)).fetchall()]
    finally:
        conn.close()


def anchor_state(db: Path) -> list[tuple[Any, ...]]:
    """Anchor content two runs must agree on. The ULID id is excluded.

    Everything else *is* included, timestamps and all: anchors are stamped with
    their source event's clock rather than the wall clock, so two runs over the
    same history are comparable down to ``updated_at``.
    """

    return [
        (*row[:4], bytes(row[4]), *row[5:])
        for row in _rows(
            db,
            f"""
            SELECT scope, query, fingerprint, dimensions, embedding,
                   reinforcement_count, decayed, decay_reason, first_seen,
                   last_matched_at, created_at, updated_at
            FROM {QUERY_ANCHOR_TABLE} ORDER BY scope, fingerprint
            """,
        )
    ]


def edge_state(db: Path) -> list[tuple[Any, ...]]:
    """Anchor edges keyed by the anchor's query rather than by its ULID."""

    return _rows(
        db,
        f"""
        SELECT a.scope, a.fingerprint, e.target_id, e.weight, e.hits,
               e.created_at, e.updated_at
        FROM {QUERY_ANCHOR_EDGE_TABLE} e
        JOIN {QUERY_ANCHOR_TABLE} a ON a.id = e.anchor_id
        ORDER BY a.scope, a.fingerprint, e.target_id
        """,
    )


def graph_digest(db: Path) -> str:
    """One hash over everything the backfill is forbidden to touch."""

    digest = hashlib.sha256()
    conn = sqlite3.connect(db)
    try:
        for sql in (
            "SELECT id, content, scope, decayed, usefulness_score, access_count, "
            "updated_at FROM nodes ORDER BY id",
            "SELECT source_id, target_id, type, weight, updated_at FROM connections "
            "ORDER BY source_id, target_id, type",
            "SELECT id, query, scope, results, feedback_applied, feedback_trace_id, "
            "created_at FROM recall_events ORDER BY id",
            "SELECT scope, bm25, vector, graph FROM retrieval_weights ORDER BY scope",
        ):
            for row in conn.execute(sql):
                digest.update(repr(tuple(row)).encode("utf-8"))
    finally:
        conn.close()
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchor_for(db: Path, query: str, scope: str = SCOPE) -> tuple[Any, ...] | None:
    rows = _rows(
        db,
        f"SELECT id, scope, query, reinforcement_count, first_seen, last_matched_at "
        f"FROM {QUERY_ANCHOR_TABLE} WHERE scope = ? AND fingerprint = ?",
        (scope, recall_fingerprint(query, scope)),
    )
    return rows[0] if rows else None


def targets_of(db: Path, query: str, scope: str = SCOPE) -> set[str]:
    return {
        str(row[0])
        for row in _rows(
            db,
            f"""
            SELECT e.target_id FROM {QUERY_ANCHOR_EDGE_TABLE} e
            JOIN {QUERY_ANCHOR_TABLE} a ON a.id = e.anchor_id
            WHERE a.scope = ? AND a.fingerprint = ?
            """,
            (scope, recall_fingerprint(query, scope)),
        )
    }


def node_id_of(db: Path, content: str) -> str:
    """The one node with exactly this content.

    Exact rather than a substring match: every consuming trace in the fixture
    quotes the node it grounded on, so a LIKE would match the trace too.
    """

    rows = _rows(db, "SELECT id FROM nodes WHERE content = ?", (content,))
    assert len(rows) == 1, f"{content!r} matched {len(rows)} nodes"
    return str(rows[0][0])


# ---------------------------------------------------------------------------
# Baseline: a full run
# ---------------------------------------------------------------------------


def test_full_run_anchors_grounded_history_and_nothing_else(source_db: Path) -> None:
    db = fixture_copy(source_db, "run.sqlite3")
    before = graph_digest(db)

    assert complete_run(db) == 0

    # One anchor per distinct (query, scope) that some consumption grounded on.
    # The repeat of QUERIES[0] reinforces rather than duplicating; the same
    # query asked in `global` is a separate anchor; the ungrounded consumption
    # produces nothing at all.
    assert len(anchor_state(db)) == GROUNDED_EVENTS + 1
    assert anchor_for(db, "что там с погодой на выходных") is None

    for index, query in enumerate(QUERIES):
        assert targets_of(db, query) == {node_id_of(db, _target_content(index))}
    assert targets_of(db, QUERIES[0], "global") == {node_id_of(db, GLOBAL_CONTENT)}

    # The delivered-but-unused result never becomes an edge.
    distractor = _rows(db, "SELECT id FROM nodes WHERE content = ?", (DISTRACTOR,))[0][0]
    assert str(distractor) not in targets_of(db, QUERIES[1])

    # Nothing outside the two anchor tables moved.
    assert graph_digest(db) == before

    report = bf.verify_report(db)
    assert report["ok"] is True
    assert report["anchors"] == GROUNDED_EVENTS + 1
    assert report["anchors_without_edges"] == 0
    assert report["edges_to_missing_nodes"] == 0
    assert report["edges_to_superseded_nodes"] == 0
    assert report["anchors_outside_source_scope"] == 0


def test_anchors_carry_their_source_events_clock(source_db: Path) -> None:
    db = fixture_copy(source_db, "clock.sqlite3")
    assert complete_run(db) == 0

    first = anchor_for(db, QUERIES[0])
    assert first is not None
    # Created by the January event, last reinforced by the July repeat: the
    # anchor's history is the history it was built from, not the clock of the
    # machine that rebuilt it.
    assert first[4] == "2026-01-01T10:00:00Z"
    assert first[5] == "2026-07-01T10:00:00Z"
    assert first[3] == 1  # one reinforcement on top of the insert

    third = anchor_for(db, QUERIES[2])
    assert third is not None and third[4] == "2026-03-01T10:00:00Z"


def test_repeated_query_reinforces_one_anchor_instead_of_duplicating(
    source_db: Path,
) -> None:
    db = fixture_copy(source_db, "dedup.sqlite3")
    assert complete_run(db) == 0

    identities = _rows(
        db,
        f"SELECT scope, fingerprint, COUNT(*) FROM {QUERY_ANCHOR_TABLE} "
        "GROUP BY scope, fingerprint HAVING COUNT(*) > 1",
    )
    assert identities == []

    edges = _rows(
        db,
        f"""
        SELECT e.weight, e.hits FROM {QUERY_ANCHOR_EDGE_TABLE} e
        JOIN {QUERY_ANCHOR_TABLE} a ON a.id = e.anchor_id
        WHERE a.scope = ? AND a.fingerprint = ?
        """,
        (SCOPE, recall_fingerprint(QUERIES[0], SCOPE)),
    )
    # Two consumptions of the same (query, node) pair: weight accumulated, and
    # `hits` counts the evidence separately from the saturating weight.
    assert edges == [(pytest.approx(2 * ANCHOR_EDGE_WEIGHT), 2)]
    assert _rows(
        db,
        f"SELECT weight, hits FROM {QUERY_ANCHOR_EDGE_TABLE} e "
        f"JOIN {QUERY_ANCHOR_TABLE} a ON a.id = e.anchor_id WHERE a.scope = ? "
        "AND a.fingerprint = ?",
        (SCOPE, recall_fingerprint(QUERIES[1], SCOPE)),
    ) == [(pytest.approx(ANCHOR_EDGE_WEIGHT), 1)]


def test_same_query_in_two_scopes_stays_two_anchors(source_db: Path) -> None:
    db = fixture_copy(source_db, "scopes.sqlite3")
    assert complete_run(db) == 0

    here = anchor_for(db, QUERIES[0], SCOPE)
    there = anchor_for(db, QUERIES[0], "global")
    assert here is not None and there is not None
    assert here[0] != there[0]
    assert targets_of(db, QUERIES[0], SCOPE) != targets_of(db, QUERIES[0], "global")


# ---------------------------------------------------------------------------
# The authorizer
# ---------------------------------------------------------------------------


def test_authorizer_denies_every_write_outside_the_anchor_tables(
    source_db: Path,
) -> None:
    db = fixture_copy(source_db, "authorizer.sqlite3")
    store = MemoryStore(db)
    try:
        bf.install_anchor_only_authorizer(store.connection)

        with pytest.raises(sqlite3.DatabaseError):
            store.connection.execute(
                "UPDATE nodes SET usefulness_score = 1.0 WHERE 1"
            )
        with pytest.raises(sqlite3.DatabaseError):
            store.connection.execute(
                "INSERT INTO connections (id, source_id, target_id, type, weight, "
                "metadata, created_at, updated_at) VALUES ('x','a','b','related',1,"
                "'{}','t','t')"
            )
        with pytest.raises(sqlite3.DatabaseError):
            store.connection.execute("UPDATE recall_events SET gated = 1")
        with pytest.raises(sqlite3.DatabaseError):
            store.connection.execute("DROP TABLE recall_events")
        with pytest.raises(sqlite3.DatabaseError):
            store.connection.execute("ALTER TABLE nodes ADD COLUMN sneaky TEXT")

        # Reads stay open, and the two anchor tables stay writable.
        assert store.connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] > 0
        store.connection.execute(
            f"DELETE FROM {QUERY_ANCHOR_EDGE_TABLE} WHERE anchor_id = 'nothing'"
        )
    finally:
        store.connection.set_authorizer(None)
        store.close()


# ---------------------------------------------------------------------------
# --until
# ---------------------------------------------------------------------------


def test_until_excludes_events_at_or_after_the_cutoff(source_db: Path) -> None:
    db = fixture_copy(source_db, "until.sqlite3")
    assert complete_run(db, "backup-until.sqlite3", "--until", "2026-04-01") == 0

    # January, February, March are in; April onwards is not. The boundary is
    # strict: the April event is stamped exactly 2026-04-01T10:00:00Z and the
    # cutoff instant belongs to the evaluated side.
    assert {row[1] for row in anchor_state(db)} == {
        QUERIES[0],
        QUERIES[1],
        QUERIES[2],
    }
    assert all(row[10] < "2026-04-01T00:00:00Z" for row in anchor_state(db))
    assert anchor_for(db, QUERIES[3]) is None

    # And the cutoff is the only thing holding the rest back.
    assert complete_run(db, "backup-until-2.sqlite3") == 0
    assert len(anchor_state(db)) == GROUNDED_EVENTS + 1


def test_until_rejects_an_unparseable_cutoff(source_db: Path) -> None:
    db = fixture_copy(source_db, "until-bad.sqlite3")
    before = file_digest(db)
    assert (
        cli(
            "backfill",
            "--db",
            str(db),
            "--backup",
            str(db.parent / "backup-bad.sqlite3"),
            "--until",
            "last tuesday",
            "--allow-fallback-embeddings",
            "--quiet",
        )
        == 2
    )
    assert file_digest(db) == before


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------


def test_rerun_after_a_complete_run_writes_nothing(source_db: Path) -> None:
    db = fixture_copy(source_db, "rerun.sqlite3")
    assert complete_run(db) == 0
    anchors, edges = anchor_state(db), edge_state(db)
    ids = _rows(db, f"SELECT id FROM {QUERY_ANCHOR_TABLE} ORDER BY id")

    assert (
        cli(
            "backfill",
            "--db",
            str(db),
            "--existing-backup",
            str(db.parent / "backup.sqlite3"),
            "--allow-fallback-embeddings",
            "--quiet",
            "--json",
            str(db.parent / "rerun.json"),
        )
        == 0
    )

    assert anchor_state(db) == anchors
    assert edge_state(db) == edges
    # Not merely equal in content: the same rows, never rewritten.
    assert _rows(db, f"SELECT id FROM {QUERY_ANCHOR_TABLE} ORDER BY id") == ids
    summary = json.loads((db.parent / "rerun.json").read_text("utf-8"))
    assert summary["events_applied"] == 0
    assert summary["events_already_anchored"] == GROUNDED_EVENTS + 2
    assert summary["complete"] is True


def test_history_replayed_into_a_live_written_anchor_adds_edges_without_aging_it(
    source_db: Path,
) -> None:
    """The overlap case on the live database: the write path got there first.

    From the moment the live loop is deployed it anchors consumptions itself,
    with the wall clock. The retro pass then meets an anchor that is *newer*
    than the history it is replaying and whose edges are those of one recent
    consumption. It must contribute the edges that history knows about without
    stamping a live anchor as months old — and it must still be idempotent.
    """

    db = fixture_copy(source_db, "overlap.sqlite3")
    other = node_id_of(db, DISTRACTOR)
    with MemoryStore(db) as store:
        live = store.insert_query_anchor(
            scope=SCOPE,
            query=QUERIES[2],
            fingerprint=recall_fingerprint(QUERIES[2], SCOPE),
            embedding=[0.25] * 384,
            now="2026-09-01T12:00:00Z",
        )
        store.upsert_query_anchor_edge(
            live.id, other, weight=0.25, now="2026-09-01T12:00:00Z"
        )

    assert complete_run(db) == 0

    anchor = anchor_for(db, QUERIES[2])
    assert anchor is not None
    # The March consumption's node joined the live one; freshness stayed live.
    assert targets_of(db, QUERIES[2]) == {other, node_id_of(db, _target_content(2))}
    assert anchor[5] == "2026-09-01T12:00:00Z"

    state, edges = anchor_state(db), edge_state(db)
    assert complete_run(db, "backup-overlap-2.sqlite3") == 0
    assert (anchor_state(db), edge_state(db)) == (state, edges)


KILL_DRIVER = textwrap.dedent(
    """
    import importlib.util, os, signal, sys

    spec = importlib.util.spec_from_file_location("bf_child", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    sys.modules["bf_child"] = module
    spec.loader.exec_module(module)

    kill_after = int(os.environ["KILL_AFTER_CALLS"])
    build = module.build_query_embedder
    seen = {"calls": 0}

    def patched(**kwargs):
        embedder, info = build(**kwargs)

        def wrapped(texts):
            seen["calls"] += 1
            if seen["calls"] > kill_after:
                # A real, uncatchable kill at a deterministic point: with
                # --batch-size 1 the encoder is called once per event, so this
                # lands after `kill_after` events have been written and before
                # the next one is.
                sys.stdout.flush()
                os.kill(os.getpid(), signal.SIGKILL)
            return embedder(texts)

        return wrapped, info

    module.build_query_embedder = patched
    raise SystemExit(module.main(sys.argv[2:]))
    """
)


def test_sigkill_mid_run_then_resume_matches_an_uninterrupted_run(
    source_db: Path, tmp_path: Path
) -> None:
    reference = fixture_copy(source_db, "reference.sqlite3")
    assert complete_run(reference, "backup-reference.sqlite3") == 0
    expected_anchors = anchor_state(reference)
    expected_edges = edge_state(reference)
    assert expected_anchors

    interrupted = fixture_copy(source_db, "interrupted.sqlite3")
    driver = tmp_path / "kill_driver.py"
    driver.write_text(KILL_DRIVER, encoding="utf-8")
    env = child_env()
    env["KILL_AFTER_CALLS"] = "3"
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

    killed_anchors = anchor_state(interrupted)
    assert 0 < len(killed_anchors) < len(expected_anchors)
    # Whatever the kill landed on, it did not leave an anchor that answers
    # nothing behind: that is the state a resume could not distinguish from a
    # finished one.
    assert bf.verify_report(interrupted)["anchors_without_edges"] == 0

    assert complete_run(interrupted, "backup-resume.sqlite3") == 0
    assert anchor_state(interrupted) == expected_anchors
    assert edge_state(interrupted) == expected_edges


def test_limited_pass_then_resume_matches_an_uninterrupted_run(
    source_db: Path,
) -> None:
    reference = fixture_copy(source_db, "reference-limit.sqlite3")
    assert complete_run(reference, "backup-ref-limit.sqlite3") == 0
    expected_anchors = anchor_state(reference)
    expected_edges = edge_state(reference)

    partial = fixture_copy(source_db, "partial.sqlite3")
    assert complete_run(partial, "backup-partial.sqlite3", "--limit", "4") == 0
    assert 0 < len(anchor_state(partial)) < len(expected_anchors)

    assert complete_run(partial, "backup-partial-2.sqlite3") == 0
    assert anchor_state(partial) == expected_anchors
    assert edge_state(partial) == expected_edges


# ---------------------------------------------------------------------------
# Backup gate
# ---------------------------------------------------------------------------


def test_refuses_to_run_without_a_backup(source_db: Path) -> None:
    db = fixture_copy(source_db, "nobackup.sqlite3")
    before = file_digest(db)
    assert cli("backfill", "--db", str(db), "--allow-fallback-embeddings") == 2
    assert file_digest(db) == before


def test_refuses_both_backup_flags_at_once(source_db: Path) -> None:
    db = fixture_copy(source_db, "bothbackups.sqlite3")
    assert (
        cli(
            "backfill",
            "--db",
            str(db),
            "--backup",
            str(db.parent / "a.sqlite3"),
            "--existing-backup",
            str(db.parent / "b.sqlite3"),
            "--allow-fallback-embeddings",
        )
        == 2
    )


def test_refuses_a_backup_that_is_the_database_itself(source_db: Path) -> None:
    db = fixture_copy(source_db, "selfbackup.sqlite3")
    assert (
        cli(
            "backfill",
            "--db",
            str(db),
            "--backup",
            str(db),
            "--allow-fallback-embeddings",
        )
        == 2
    )


def test_refuses_a_backup_missing_the_newest_rows(source_db: Path) -> None:
    """What a ``cp`` without the ``-wal`` looks like from the outside."""

    db = fixture_copy(source_db, "stale.sqlite3")
    stale = db.parent / "stale-backup.sqlite3"
    bf.take_backup(db, stale, overwrite=True)
    conn = sqlite3.connect(stale)
    try:
        conn.execute("DELETE FROM nodes WHERE id IN (SELECT id FROM nodes LIMIT 3)")
        conn.commit()
    finally:
        conn.close()

    before = file_digest(db)
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
    assert file_digest(db) == before


def test_refuses_the_hash_fallback_without_the_opt_in(source_db: Path) -> None:
    db = fixture_copy(source_db, "fallback.sqlite3")
    before = file_digest(db)
    assert (
        cli(
            "backfill",
            "--db",
            str(db),
            "--backup",
            str(db.parent / "backup-fallback.sqlite3"),
        )
        == 2
    )
    # The refusal happens after the backup is taken and before any write.
    assert file_digest(db) == before


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing_and_bounds_the_real_run(source_db: Path) -> None:
    db = fixture_copy(source_db, "dry.sqlite3")
    before = file_digest(db)
    report = db.parent / "dry.json"

    assert (
        cli(
            "backfill",
            "--db",
            str(db),
            "--dry-run",
            "--quiet",
            "--json",
            str(report),
        )
        == 0
    )
    assert file_digest(db) == before

    projection = json.loads(report.read_text("utf-8"))
    assert projection["events_total"] == GROUNDED_EVENTS + 3
    assert projection["events_grounded"] == GROUNDED_EVENTS + 2
    assert projection["anchor_tables_present"] is False

    assert complete_run(db) == 0
    # The projection counts distinct (scope, fingerprint) identities, so it is
    # an upper bound the real run meets exactly when cosine dedup merges
    # nothing further — which is the case for these deliberately distinct
    # queries.
    assert projection["projected_anchors"] == len(anchor_state(db))
    assert projection["projected_edges"] == len(edge_state(db))


def test_dry_run_needs_no_backup_and_no_migration(source_db: Path) -> None:
    db = fixture_copy(source_db, "dry-nomigrate.sqlite3")
    version = _rows(db, "SELECT value FROM metadata WHERE key = 'schema_version'")
    assert cli("backfill", "--db", str(db), "--dry-run", "--quiet") == 0
    assert _rows(db, "SELECT value FROM metadata WHERE key = 'schema_version'") == version
    assert _rows(
        db, "SELECT name FROM sqlite_master WHERE name = ?", (QUERY_ANCHOR_TABLE,)
    ) == []


# ---------------------------------------------------------------------------
# Edge migration and verify
# ---------------------------------------------------------------------------


def _supersede_without_the_hook(
    db: Path, *, old: str, new: str, when: str = "2026-08-01T10:00:00Z"
) -> None:
    """Write a ``supersedes`` edge the way a pre-hook release would have.

    ``MemoryStore._insert_connection`` re-points anchor edges in the same
    transaction that writes a ``supersedes``, so going through the store would
    repair the very state under test. Raw SQL reproduces a database written
    before that hook existed — which is exactly the case the sweep is for.
    """

    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT INTO connections (id, source_id, target_id, type, weight, "
            "metadata, created_at, updated_at) VALUES (?, ?, ?, 'supersedes', 1.0, "
            "'{}', ?, ?)",
            (f"conn-{old}-{new}", new, old, when, when),
        )
        conn.execute(
            "UPDATE nodes SET decayed = 1, decay_reason = 'superseded' WHERE id = ?",
            (old,),
        )
        conn.commit()
    finally:
        conn.close()


def test_verify_fails_on_an_edge_into_a_superseded_node_and_the_sweep_fixes_it(
    source_db: Path,
) -> None:
    db = fixture_copy(source_db, "supersede.sqlite3")
    assert complete_run(db) == 0
    assert bf.verify_report(db)["ok"] is True

    old = node_id_of(db, _target_content(2))
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT INTO nodes (id, level, content, scope, context, timestamp, "
            "created_at, updated_at) VALUES ('REPLACEMENT01', 'trace', "
            "'узел-заместитель маркер-каппа-2 актуальная версия', ?, '{}', ?, ?, ?)",
            (SCOPE, "2026-08-01T10:00:00Z", "2026-08-01T10:00:00Z", "2026-08-01T10:00:00Z"),
        )
        conn.commit()
    finally:
        conn.close()
    _supersede_without_the_hook(db, old=old, new="REPLACEMENT01")

    broken = bf.verify_report(db)
    assert broken["ok"] is False
    assert broken["edges_to_superseded_nodes"] == 1
    assert broken["superseded_target_examples"] == [
        {"target": old, "replacement": "REPLACEMENT01"}
    ]
    assert cli("verify", "--db", str(db)) == 1

    # The sweep at the end of a backfill pass repairs it: the anchor now
    # answers with the replacement instead of pointing into the past.
    assert complete_run(db, "backup-sweep.sqlite3") == 0
    assert targets_of(db, QUERIES[2]) == {"REPLACEMENT01"}
    assert bf.verify_report(db)["ok"] is True
    assert cli("verify", "--db", str(db)) == 0


def test_verify_tolerates_an_edge_into_a_decayed_node_with_no_replacement(
    source_db: Path,
) -> None:
    db = fixture_copy(source_db, "decayed.sqlite3")
    assert complete_run(db) == 0
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "UPDATE nodes SET decayed = 1, decay_reason = 'ttl' WHERE id = ?",
            (node_id_of(db, _target_content(4)),),
        )
        conn.commit()
    finally:
        conn.close()

    report = bf.verify_report(db)
    # Inert, not broken: nothing replaced the node, so the edge simply offers
    # one target fewer. Failing here would make ordinary decay look like a bug.
    assert report["edges_to_decayed_nodes_without_replacement"] == 1
    assert report["edges_to_superseded_nodes"] == 0
    assert report["ok"] is True


def test_verify_fails_on_an_edge_into_a_node_that_no_longer_exists(
    source_db: Path,
) -> None:
    db = fixture_copy(source_db, "missing.sqlite3")
    assert complete_run(db) == 0
    conn = sqlite3.connect(db)
    try:
        conn.execute("DELETE FROM nodes WHERE id = ?", (node_id_of(db, _target_content(5)),))
        conn.commit()
    finally:
        conn.close()

    report = bf.verify_report(db)
    assert report["edges_to_missing_nodes"] == 1
    assert report["ok"] is False
    assert cli("verify", "--db", str(db)) == 1


def test_verify_fails_on_an_anchor_outside_its_source_events_scope(
    source_db: Path,
) -> None:
    db = fixture_copy(source_db, "crossscope.sqlite3")
    assert complete_run(db) == 0

    # QUERIES[1] was only ever asked in project:demo. An anchor for it in
    # `global` cannot have come from a consumption in `global`.
    with MemoryStore(db) as store:
        anchor = store.insert_query_anchor(
            scope="global",
            query=QUERIES[1],
            fingerprint=recall_fingerprint(QUERIES[1], "global"),
            embedding=[0.5] * 8,
        )
        store.upsert_query_anchor_edge(
            anchor.id, node_id_of(db, _target_content(1)), weight=0.25
        )

    report = bf.verify_report(db)
    assert report["anchors_outside_source_scope"] == 1
    assert report["outside_source_scope_examples"][0]["scope"] == "global"
    assert report["ok"] is False
    assert cli("verify", "--db", str(db)) == 1


def test_verify_fails_on_an_anchor_with_no_edges_or_no_vector(
    source_db: Path,
) -> None:
    db = fixture_copy(source_db, "edgeless.sqlite3")
    assert complete_run(db) == 0
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            f"DELETE FROM {QUERY_ANCHOR_EDGE_TABLE} WHERE anchor_id = "
            f"(SELECT id FROM {QUERY_ANCHOR_TABLE} WHERE fingerprint = ?)",
            (recall_fingerprint(QUERIES[3], SCOPE),),
        )
        conn.commit()
    finally:
        conn.close()
    report = bf.verify_report(db)
    assert report["anchors_without_edges"] == 1
    assert report["ok"] is False

    conn = sqlite3.connect(db)
    try:
        conn.execute(
            f"UPDATE {QUERY_ANCHOR_TABLE} SET embedding = ZEROBLOB(dimensions * 4) "
            "WHERE fingerprint = ?",
            (recall_fingerprint(QUERIES[4], SCOPE),),
        )
        conn.commit()
    finally:
        conn.close()
    assert bf.verify_report(db)["anchors_with_unusable_embedding"] == 1


def test_verify_reports_a_pre_v7_database_as_not_ok(source_db: Path) -> None:
    db = fixture_copy(source_db, "prev7.sqlite3")
    report = bf.verify_report(db)
    assert report["anchor_tables_present"] is False
    assert report["ok"] is False
    assert cli("verify", "--db", str(db)) == 1


def test_event_whose_targets_all_decayed_creates_no_anchor(source_db: Path) -> None:
    db = fixture_copy(source_db, "dead.sqlite3")
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "UPDATE nodes SET decayed = 1, decay_reason = 'ttl' WHERE id = ?",
            (node_id_of(db, _target_content(3)),),
        )
        conn.commit()
    finally:
        conn.close()

    assert complete_run(db) == 0
    # No anchor rather than an anchor with no edges: an anchor that answers
    # nothing is worse than no anchor, and `verify` treats it as a defect.
    assert anchor_for(db, QUERIES[3]) is None
    assert bf.verify_report(db)["ok"] is True


def test_missing_db_is_a_refusal_not_a_traceback(tmp_path: Path) -> None:
    assert cli("verify", "--db", str(tmp_path / "nope.sqlite3")) == 2
    assert cli("backfill", "--db", str(tmp_path / "nope.sqlite3"), "--dry-run") == 2
