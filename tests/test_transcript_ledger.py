"""The transcript-grounding verdict ledger is additive, idempotent, and dead to the live path.

Phase 3 of the transcript-grounding goal ships exactly one schema change — a
CREATE-only table plus one index — and an offline importer. What a live
500 MB database must be able to rely on is pinned here:

* **Byte-identity of everything that already exists.** A fresh database
  created by this build carries, for every object master's build creates,
  the byte-identical ``sqlite_master`` SQL. The master build is loaded from
  ``git show master:src/living_memory/storage.py`` and executed as a module,
  so the comparison is against the code actually deployed, not a copy that
  could drift.
* **Reopening is the whole migration.** A database file created by master's
  build gains exactly the two new objects when the new build opens it —
  nothing else appears, nothing changes, ``schema_version`` stays at 8, and
  the data survives.
* **Import replays instead of duplicating**, never updates an existing row,
  and reports same-key-different-numbers as the method_version discipline
  violation it is.
* **The live recall/remember path never touches the table** — proven by
  tracing every SQL statement the live tools execute, not by reading the
  source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import sqlite3
import subprocess
import sys
import types

import pytest

from living_memory.server import create_mcp_server
from living_memory.storage import (
    CONSOLIDATION_EMBEDDING_TABLE,
    MemoryStore,
    RECALL_CREDIT_LEDGER_TABLE,
    TRANSCRIPT_GROUNDING_TABLE,
)
from living_memory.transcript_ledger import (
    TranscriptLedgerError,
    ensure_ledger_table,
    import_verdicts,
    parse_verdict,
    read_verdicts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_INDEX = "idx_transcript_grounding_node_graded"
#: The only objects this release may add to a database: the transcript
#: grounding ledger and its index, and the recall credit ledger
#: (feedback-usage-signal), which is created the same way — on every open,
#: additive, no SCHEMA_VERSION bump — and whose UNIQUE constraint leaves no
#: named index in sqlite_master (autoindexes carry NULL sql).
#: The explicit-feedback tables (goal explicit-recall-feedback) join them on
#: the same terms: explicit credit side table, mark audit and its index.
#: The consolidation vector cache (goal remember-returns-before-consolidation)
#: is created on every open too, additive, no SCHEMA_VERSION bump.
NEW_OBJECTS = {
    TRANSCRIPT_GROUNDING_TABLE,
    LEDGER_INDEX,
    RECALL_CREDIT_LEDGER_TABLE,
    "recall_explicit_credit",
    "recall_feedback_marks",
    "idx_recall_feedback_marks_event",
    CONSOLIDATION_EMBEDDING_TABLE,
}

SCOPE = "project:ledger"
AMBIENT = {"agent": "seeder", "task": "ledger-fixture", "session_id": "seed-session"}


@pytest.fixture(autouse=True)
def _hash_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")


class FakeMCP:
    def __init__(self, name: str, instructions: str | None = None) -> None:
        self.name = name
        self.instructions = instructions
        self.tools: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def tool(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.tools[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        return decorate if func is None else decorate(func)

    def resource(self, uri: str, **_kwargs: Any) -> Any:
        def decorate(func: Any) -> Any:
            self.resources[uri] = func
            return func

        return decorate

    def prompt(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.prompts[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        return decorate if func is None else decorate(func)


# ---------------------------------------------------------------------------
# Helpers: master's storage build, schema snapshots, verdict payloads
# ---------------------------------------------------------------------------

_MASTER_STORAGE_CACHE: list[types.ModuleType] = []


def _master_storage() -> types.ModuleType:
    """master's ``storage.py``, executed as a module against current siblings.

    Only ``storage.py`` changes in this release's schema surface, so master's
    storage against the current sibling modules *is* the deployed DDL. Loaded
    once per test process.
    """

    if _MASTER_STORAGE_CACHE:
        return _MASTER_STORAGE_CACHE[0]
    try:
        source = subprocess.run(
            ["git", "show", "master:src/living_memory/storage.py"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover
        pytest.skip(f"master's storage.py unavailable via git: {error}")
    module = types.ModuleType("living_memory._storage_at_master")
    module.__file__ = str(REPO_ROOT / "src" / "living_memory" / "storage.py")
    sys.modules[module.__name__] = module
    exec(compile(source, "master:src/living_memory/storage.py", "exec"), module.__dict__)
    _MASTER_STORAGE_CACHE.append(module)
    return module


def _schema_objects(db_path: Path) -> dict[str, str]:
    """``sqlite_master`` name -> verbatim SQL for every named object."""

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    return {str(name): str(sql) for name, sql in rows}


def _schema_version(db_path: Path) -> str:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
    finally:
        conn.close()
    return str(row[0])


def _assert_only_ledger_added(before: dict[str, str], after: dict[str, str]) -> None:
    """Everything shared is byte-identical; additions are at most the ledger."""

    for name, sql in before.items():
        assert name in after, f"{name} disappeared"
        assert after[name] == sql, f"{name} DDL changed:\n{sql!r}\n->\n{after[name]!r}"
    added = set(after) - set(before)
    # Subset, not equality: once this release reaches master, master's build
    # creates the ledger too and the difference legitimately collapses to
    # nothing.
    assert added <= NEW_OBJECTS, f"unexpected schema objects: {sorted(added - NEW_OBJECTS)}"
    assert NEW_OBJECTS <= set(after)


def _verdict(index: int = 0, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "recall_event_id": f"01EVENT{index:020d}",
        "node_id": f"01NODE{index:021d}",
        "containment": round(0.30 + index * 0.01, 6),
        "grounded": 1,
        "method_version": "idf-containment-v1",
        "field": "local",
        "delivered_at": "2026-08-20T10:00:00Z",
        "graded_at": "2026-08-28T09:00:00Z",
        "transcript_session_key": f"claude/2026-08-20T10-00-00Z/key{index}",
    }
    payload.update(overrides)
    return payload


def _write_jsonl(path: Path, payloads: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(payload) + "\n" for payload in payloads), encoding="utf-8"
    )
    return path


def _ledger_rows(db_path: Path) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            f"""
            SELECT id, recall_event_id, node_id, containment, grounded,
                   method_version, field, delivered_at, graded_at,
                   transcript_session_key, imported_at
            FROM {TRANSCRIPT_GROUNDING_TABLE}
            ORDER BY recall_event_id, node_id, method_version
            """
        ).fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# (a) Fresh-database DDL: everything pre-existing is byte-identical to master
# ---------------------------------------------------------------------------


def test_fresh_db_preexisting_ddl_byte_identical_to_master(tmp_path: Path) -> None:
    master = _master_storage()
    with master.MemoryStore(tmp_path / "master.sqlite3"):
        pass
    with MemoryStore(tmp_path / "current.sqlite3"):
        pass

    _assert_only_ledger_added(
        _schema_objects(tmp_path / "master.sqlite3"),
        _schema_objects(tmp_path / "current.sqlite3"),
    )
    assert _schema_version(tmp_path / "current.sqlite3") == "8"


# ---------------------------------------------------------------------------
# (b) Reopening an existing database is the whole migration
# ---------------------------------------------------------------------------


def test_reopening_master_created_db_adds_only_the_ledger(tmp_path: Path) -> None:
    master = _master_storage()
    db = tmp_path / "existing.sqlite3"
    with master.MemoryStore(db) as store:
        node = store.append_trace("ledger fixture trace body", {"scope": SCOPE, **AMBIENT})
        event = store.record_recall_event(
            query="ledger fixture",
            scope=SCOPE,
            ambient_context=dict(AMBIENT),
            results=[
                {
                    "node_id": node.id,
                    "rank": 0,
                    "bm25_score": 1.0,
                    "vector_score": 0.5,
                    "graph_score": 0.0,
                }
            ],
        )
    before = _schema_objects(db)
    assert _schema_version(db) == "8"

    with MemoryStore(db) as reopened:
        assert reopened.get_node(node.id) is not None
        assert reopened.get_recall_event(event.id) is not None

    _assert_only_ledger_added(before, _schema_objects(db))
    assert _schema_version(db) == "8"


def test_dropped_ledger_reappears_on_reopen(tmp_path: Path) -> None:
    """CREATE IF NOT EXISTS on every open is the migration — same as v8."""

    db = tmp_path / "dropped.sqlite3"
    with MemoryStore(db) as store:
        store.connection.execute(f"DROP TABLE {TRANSCRIPT_GROUNDING_TABLE}")
        store.connection.commit()
    assert TRANSCRIPT_GROUNDING_TABLE not in _schema_objects(db)

    with MemoryStore(db):
        pass
    after = _schema_objects(db)
    assert NEW_OBJECTS <= set(after)


# ---------------------------------------------------------------------------
# (c) Import: replay instead of duplicate, never update, report mismatches
# ---------------------------------------------------------------------------


def test_import_is_idempotent_and_never_updates(tmp_path: Path) -> None:
    db = tmp_path / "import.sqlite3"
    with MemoryStore(db):
        pass
    payloads = [_verdict(0), _verdict(1, field="alt", grounded=0), _verdict(2)]
    jsonl = _write_jsonl(tmp_path / "verdicts.jsonl", payloads)
    verdicts = read_verdicts(jsonl)

    conn = sqlite3.connect(db)
    try:
        first = import_verdicts(conn, verdicts, imported_at="2026-08-29T00:00:00Z")
        assert (first.total, first.inserted, first.replayed, first.mismatched) == (3, 3, 0, 0)
    finally:
        conn.close()
    rows = _ledger_rows(db)
    assert len(rows) == 3
    assert all(row[10] == "2026-08-29T00:00:00Z" for row in rows)

    # Second run with a *different* stamp: pure replay, no write of any kind —
    # ids and imported_at are untouched, row count stays 3.
    conn = sqlite3.connect(db)
    try:
        second = import_verdicts(conn, verdicts, imported_at="2026-08-30T00:00:00Z")
        assert (second.total, second.inserted, second.replayed, second.mismatched) == (
            3,
            0,
            3,
            0,
        )
    finally:
        conn.close()
    assert _ledger_rows(db) == rows

    # A same-key verdict with different numbers is reported, never applied.
    mutated = [
        _verdict(0, containment=0.99, grounded=0),
        _verdict(1, field="alt", grounded=0),
        _verdict(3),
    ]
    conn = sqlite3.connect(db)
    try:
        third = import_verdicts(conn, read_verdicts(_write_jsonl(tmp_path / "v2.jsonl", mutated)))
        assert (third.total, third.inserted, third.replayed, third.mismatched) == (3, 1, 1, 1)
        assert third.mismatches == (
            (payloads[0]["recall_event_id"], payloads[0]["node_id"], "idf-containment-v1"),
        )
    finally:
        conn.close()
    final = _ledger_rows(db)
    assert len(final) == 4
    assert rows[0] in final, "the mismatched key kept its first-written row"


def test_dry_run_plans_without_writing(tmp_path: Path) -> None:
    """apply=False reports the same numbers a real run would, on a virgin file too."""

    db = tmp_path / "dry.sqlite3"
    sqlite3.connect(db).close()  # not even a MemoryStore file: no table at all
    jsonl = _write_jsonl(tmp_path / "verdicts.jsonl", [_verdict(0), _verdict(0), _verdict(1)])
    verdicts = read_verdicts(jsonl)

    conn = sqlite3.connect(db)
    try:
        stats = import_verdicts(conn, verdicts, apply=False)
        assert (stats.total, stats.inserted, stats.replayed, stats.mismatched) == (3, 2, 1, 0)
        assert not stats.applied
        # A real import on the bare file must be refused until the table exists.
        with pytest.raises(TranscriptLedgerError):
            import_verdicts(conn, verdicts)
        ensure_ledger_table(conn)
        applied = import_verdicts(conn, verdicts)
        assert (applied.inserted, applied.replayed) == (2, 1)
    finally:
        conn.close()
    assert len(_ledger_rows(db)) == 2


def test_parse_verdict_rejects_out_of_shape_payloads() -> None:
    for broken in (
        _verdict(0, field="remote"),
        _verdict(0, containment=1.5),
        _verdict(0, containment="high"),
        _verdict(0, grounded=2),
        _verdict(0, delivered_at="yesterday"),
        {key: value for key, value in _verdict(0).items() if key != "transcript_session_key"},
    ):
        with pytest.raises(TranscriptLedgerError):
            parse_verdict(broken)


# ---------------------------------------------------------------------------
# (d) The CLI end to end
# ---------------------------------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    env["LIVING_MEMORY_EMBEDDING_BACKEND"] = "hash"
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "transcript_ledger_import.py"), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_cli_import_replay_and_mismatch_exit_codes(tmp_path: Path) -> None:
    db = tmp_path / "cli.sqlite3"
    with MemoryStore(db):
        pass
    jsonl = _write_jsonl(tmp_path / "verdicts.jsonl", [_verdict(0), _verdict(1)])

    dry = _run_cli("--db", str(db), "--verdicts", str(jsonl), "--dry-run")
    assert dry.returncode == 0, dry.stderr
    assert json.loads(dry.stdout)["inserted"] == 2
    assert not _ledger_rows(db)

    first = _run_cli("--db", str(db), "--verdicts", str(jsonl))
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["inserted"] == 2

    second = _run_cli("--db", str(db), "--verdicts", str(jsonl))
    assert second.returncode == 0, second.stderr
    report = json.loads(second.stdout)
    assert (report["inserted"], report["replayed"], report["mismatched"]) == (0, 2, 0)
    assert len(_ledger_rows(db)) == 2

    mismatch = _write_jsonl(tmp_path / "regraded.jsonl", [_verdict(0, containment=0.01)])
    third = _run_cli("--db", str(db), "--verdicts", str(mismatch))
    assert third.returncode == 1
    assert json.loads(third.stdout)["mismatched"] == 1

    malformed = tmp_path / "broken.jsonl"
    malformed.write_text('{"recall_event_id": "x"\n', encoding="utf-8")
    fourth = _run_cli("--db", str(db), "--verdicts", str(malformed))
    assert fourth.returncode == 2
    assert "not valid JSON" in fourth.stderr


# ---------------------------------------------------------------------------
# (e) The live recall/remember path never touches the ledger
# ---------------------------------------------------------------------------


def test_live_recall_remember_paths_never_touch_the_ledger(tmp_path: Path) -> None:
    """Trace every statement the live tools run; the table must never appear.

    Positive controls first: the trace must show the remember write and the
    recall-event record, or an empty trace would vacuously pass.
    """

    mcp = create_mcp_server(tmp_path / "live.sqlite3", mcp_factory=FakeMCP)
    store: MemoryStore = mcp.memory_store
    statements: list[str] = []
    store.connection.set_trace_callback(statements.append)
    try:
        remembered = mcp.tools["memory_remember"](
            "transcript grounding fixture body", {"scope": SCOPE, **AMBIENT}
        )
        recalled = mcp.tools["memory_recall"]("grounding fixture", scope=SCOPE)
        assert recalled["count"] >= 1
        mcp.tools["memory_lookup"](node_id=remembered["node"]["id"])
    finally:
        store.connection.set_trace_callback(None)

    assert statements, "the trace captured nothing — the guard proved nothing"
    assert any("INSERT INTO nodes" in statement for statement in statements)
    assert any("recall_events" in statement for statement in statements)
    offenders = [s for s in statements if TRANSCRIPT_GROUNDING_TABLE in s]
    assert not offenders, f"live path touched the ledger: {offenders[:3]}"
