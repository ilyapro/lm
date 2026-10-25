"""The transcript-grounding verdict ledger is additive, idempotent, and dead to the live path.

The offline importer and additive storage objects preserve these invariants:

* **Fresh databases contain the ledgers.** Current DDL includes every ledger
  and its indexes without requiring historical experimental tables.
* **Reopening preserves existing data and DDL** apart from the explicit
  task-index migration, additive ledgers and pending-work indexes;
  ``schema_version`` stays at 8.
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

import pytest

from storage_schema_compat import (
    ADDITIVE_OBJECTS,
    assert_preserved_schema,
    assert_schema_upgrade,
    memory_rows,
    prepare_pre_ledger_schema,
    schema_objects,
    schema_version,
)

from living_memory.server import create_mcp_server
from living_memory.storage import (
    MemoryStore,
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
# Verdict payloads
# ---------------------------------------------------------------------------


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
# (a) Fresh-database ledgers
# ---------------------------------------------------------------------------


def test_fresh_db_creates_required_ledgers(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "current.sqlite3"):
        pass
    assert ADDITIVE_OBJECTS <= set(schema_objects(tmp_path / "current.sqlite3"))
    assert schema_version(tmp_path / "current.sqlite3") == "8"


# ---------------------------------------------------------------------------
# (b) Reopening an existing database is the whole migration
# ---------------------------------------------------------------------------


def test_reopening_pre_ledger_schema_preserves_data_and_builds_task_index(tmp_path: Path) -> None:
    db = tmp_path / "existing.sqlite3"
    with MemoryStore(db) as store:
        # The task phrase must be absent from content to prove task indexing.
        node = store.append_trace("preserved transcript body", {"scope": SCOPE, **AMBIENT})
        store.record_recall_event(
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
        rows = memory_rows(store.connection)
    before = prepare_pre_ledger_schema(db)

    with MemoryStore(db) as reopened:
        assert memory_rows(reopened.connection) == rows
        assert [n.id for n, _ in reopened.search_content('"ledger-fixture"')] == [node.id]
        assert [n.id for n, _ in reopened.search_content("body")] == [node.id]

    after = schema_objects(db)
    assert_schema_upgrade(before, after)
    assert schema_version(db) == "8"

    with MemoryStore(db) as again:
        assert memory_rows(again.connection) == rows
        assert [n.id for n, _ in again.search_content('"ledger-fixture"')] == [node.id]
        assert [n.id for n, _ in again.search_content("body")] == [node.id]
    assert schema_objects(db) == after
    assert schema_version(db) == "8"


@pytest.mark.parametrize("name, replacement", [
    ("nodes", "CREATE TABLE nodes (id TEXT)"),
    ("nodes_fts", "CREATE VIRTUAL TABLE nodes_fts USING fts5(content)"),
    ("nodes_fts_update", "CREATE TRIGGER nodes_fts_update AFTER UPDATE ON nodes BEGIN SELECT 1; END"),
    ("nodes_fts_vocab", "CREATE VIRTUAL TABLE nodes_fts_vocab USING fts5vocab('nodes_fts', 'row')"),
])
def test_task_index_allowance_rejects_unrelated_or_incomplete_ddl(
    tmp_path: Path, name: str, replacement: str,
) -> None:
    db = tmp_path / "schema.sqlite3"
    with MemoryStore(db):
        pass
    before = schema_objects(db)
    after = {**before, name: replacement}
    with pytest.raises(AssertionError):
        assert_preserved_schema(before, after)


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
