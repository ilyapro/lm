"""Disposable read-only evaluation checks; no corpus or external judge."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from living_memory import health_audit
from living_memory.postsession.dedup import Deduper, NodeSnapshot, StaticMemoryIndex
from living_memory.postsession.judge import FakeJudge
from living_memory.postsession.mining import mine
from living_memory.postsession.session import FileMutation, SessionRecord, SourceSpan, ToolCall
from living_memory.storage import MemoryStore

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "postsession_insights.py"
FACT = (
    '`pytest -q` on top of `addopts = "-q"` in pyproject.toml collapses the report '
    "to one line, which is why the ValueError in src/app.py looked like a silent pass."
)
SCOPE = "project:demo"
SOURCE = ("b" * 64, SourceSpan("claude", "/tmp/synthetic.jsonl", 10, "result").locator())


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("postsession_insights", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def memory(tmp_path):
    path = tmp_path / "memory #?.sqlite3"
    with MemoryStore(path) as store, store.connection:
        node = store.create_node(
            level="trace", content=FACT,
            context={
                "scope": SCOPE,
                "source_transcript_sha256": SOURCE[0],
                "source_span": SOURCE[1],
            },
        )
        store.connection.execute("UPDATE nodes SET content_fingerprint = NULL")
        store.connection.execute("DELETE FROM retrieval_weights")
    return path, NodeSnapshot(node.id, FACT, SCOPE, source_identity=SOURCE)


def _snapshot(path):
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        return path.read_bytes(), tuple(conn.iterdump())


def _capture_readers(monkeypatch):
    connections = []
    original = health_audit.open_read_only_store

    def opened(*args, **kwargs):
        store = original(*args, **kwargs)
        connections.append(store.connection)
        return store

    monkeypatch.setattr(health_audit, "open_read_only_store", opened)
    return connections


def _assert_closed(connections):
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


def _args(tmp_path, *flags):
    index = tmp_path / "index.jsonl"
    if not index.exists():
        index.write_text("")
    return [
        "--index", str(index), "--manifest", str(tmp_path / "absent-manifest.json"),
        "--out", str(tmp_path / "report.json"), "--judge", "fake", *flags,
    ]


def test_helper_reads_without_initializing_and_matches_synthetic_index(cli, memory, monkeypatch):
    path, node = memory
    before = _snapshot(path)
    connections = _capture_readers(monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("read-only evaluation called MemoryStore.__init__")

    monkeypatch.setattr(MemoryStore, "__init__", forbidden)
    cases = [
        (FACT, SCOPE, SOURCE),
        (FACT + " Approval is mandatory.", SCOPE, SOURCE),
        (FACT.replace("collapses", "preserves"), SCOPE, SOURCE),
        (FACT, "global", SOURCE),
        (FACT, SCOPE, ("c" * 64, SOURCE[1])),
        (FACT, SCOPE, (SOURCE[0], "other span")),
        (FACT, SCOPE, None),
    ]
    reference = Deduper(StaticMemoryIndex([node]))
    with cli.build_deduper(path) as deduper:
        def check(case):
            fact, scope, source = case
            return deduper.check(fact, scope=scope, source_identity=source)

        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(check, cases * 3))
        expected = [
            reference.check(fact, scope=scope, source_identity=source)
            for fact, scope, source in cases * 3
        ]
        assert actual == expected
        assert [v.duplicate for v in actual[:len(cases)]] == [True] + [False] * 6
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connections[0].execute("DELETE FROM nodes")
    _assert_closed(connections)
    assert _snapshot(path) == before


def test_helper_closes_on_operation_error(cli, memory, monkeypatch):
    connections = _capture_readers(monkeypatch)
    with pytest.raises(RuntimeError, match="extraction failed"):
        with cli.build_deduper(memory[0]):
            raise RuntimeError("extraction failed")
    _assert_closed(connections)


def test_helper_without_database_is_explicitly_unavailable(cli, monkeypatch):
    connections = _capture_readers(monkeypatch)
    with cli.build_deduper(None) as deduper:
        verdict = deduper.check(FACT, scope=SCOPE, source_identity=SOURCE)
        assert not verdict.available
        assert not verdict.duplicate
    assert connections == []


@pytest.mark.parametrize("missing_parent", [False, True])
def test_helper_and_cli_do_not_create_missing_paths(cli, tmp_path, missing_parent):
    parent = tmp_path / "absent" if missing_parent else tmp_path
    path = parent / "memory.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        with cli.build_deduper(path):
            pytest.fail("missing database accepted")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *_args(tmp_path, "--run", "--db", str(path))],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "postsession dedup:" in result.stderr
    assert "wrote " not in result.stdout
    assert not (tmp_path / "report.json").exists()
    assert not path.exists()
    assert parent.exists() is not missing_parent


@pytest.mark.parametrize("kind", ["empty", "invalid", "missing_fts", "missing_column"])
def test_helper_and_cli_reject_unsupported_database_without_writes(
    cli, tmp_path, memory, monkeypatch, kind,
):
    path = memory[0]
    if kind in {"empty", "invalid"}:
        path = tmp_path / "unsupported.sqlite3"
        path.write_bytes(b"not SQLite" if kind == "invalid" else b"")
    else:
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.execute(
                "DROP TABLE nodes_fts" if kind == "missing_fts"
                else "ALTER TABLE nodes DROP COLUMN confidence"
            )
    before = path.read_bytes()
    connections = _capture_readers(monkeypatch)
    with pytest.raises(sqlite3.DatabaseError):
        with cli.build_deduper(path):
            pytest.fail("unsupported database accepted")
    _assert_closed(connections)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *_args(tmp_path, "--run", "--db", str(path))],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "postsession dedup:" in result.stderr
    assert not (tmp_path / "report.json").exists()
    assert path.read_bytes() == before


def _record():
    def span(line):
        return SourceSpan("claude", "/tmp/synthetic.jsonl", line)

    return SessionRecord(
        session_key="claude:synthetic", source="claude", cli="claude",
        path="/tmp/synthetic.jsonl", transcript_sha256=SOURCE[0],
        repo="/tmp/demo", transport_session_ids=("synthetic-transport",),
        tool_calls=[
            ToolCall(
                0, "Bash", None, {"command": "python -m pytest -q"}, span(10),
                ok=False, result_text=(
                    "Traceback (most recent call last):\nValueError: bad token\n"
                    'src/app.py failed; pyproject.toml has addopts = "-q"\n'
                ),
            ),
            ToolCall(
                1, "Bash", None, {"command": "python -m pytest -q"}, span(30),
                ok=True, result_text="12 passed in 3.20s",
            ),
        ],
        file_mutations=[FileMutation(0, "src/app.py", "update", span(20), unified_diff="@@\n+ok\n")],
    )


@pytest.mark.parametrize("mode", ["--run", "--report"])
@pytest.mark.parametrize("with_db", [True, False])
def test_cli_fake_judge_preserves_dedup_and_report(cli, tmp_path, memory, monkeypatch, mode, with_db):
    record = _record()
    assert mine(record)[0].span.locator() == SOURCE[1]
    # Only synthetic entries are exposed to calibration/extraction. The reserved
    # entry must never reach load_session, even when --report reads both splits.
    entries = [
        {"session_key": record.session_key, "source": "claude", "path": record.path, "split": split}
        for split in ("eval", "train", cli.RESERVED_SPLIT)
    ]
    (tmp_path / "index.jsonl").write_text("\n".join(json.dumps(e) for e in entries))

    def load(entry):
        assert entry.split in cli.ALLOWED_SPLITS
        return record

    monkeypatch.setattr(cli, "load_session", load)
    monkeypatch.setattr(cli, "build_judge", lambda *a, **kw: FakeJudge([{
        "verdict": "insight", "fact": FACT,
        "why_durable": "rediscovering this costs a debugging cycle",
        "identifiers": ["pyproject.toml", "src/app.py"],
        "evidence_locator": SOURCE[1],
    }]))
    connections = _capture_readers(monkeypatch)
    before = _snapshot(memory[0])
    flags = [mode, "--workers", "3", "--judge-budget", "1"]
    if with_db:
        flags += ["--db", str(memory[0])]
    assert cli.main(_args(tmp_path, *flags)) == 0
    report_text = (tmp_path / "report.json").read_text()
    report = json.loads(report_text)
    assert report["dedup"]["index_available"] is with_db
    assert report["dedup"]["exact" if with_db else "unavailable"] == 1
    assert report["extraction"]["proposed"] == (0 if with_db else 1)
    assert report["judge"]["calls"] == 1
    assert report["splits_read"] == (["eval", "train"] if mode == "--report" else ["eval"])
    assert cli.RESERVED_SPLIT not in report_text
    if with_db:
        _assert_closed(connections)
    else:
        assert connections == []
    assert _snapshot(memory[0]) == before


@pytest.mark.parametrize("stage", ["search", "worker", "report"])
def test_cli_closes_reader_on_failure(cli, tmp_path, memory, monkeypatch, stage):
    connections = _capture_readers(monkeypatch)
    before = _snapshot(memory[0])

    def fail(*args, **kwargs):
        if stage == "search":
            def broken_search(*a, **kw):
                raise sqlite3.OperationalError("search failed")
            monkeypatch.setattr(kwargs["deduper"].index.store, "search_content", broken_search)
            kwargs["deduper"].check(FACT, scope=SCOPE, source_identity=SOURCE)
        raise RuntimeError(f"{stage} failed")

    if stage == "report":
        monkeypatch.setattr(cli, "write_report", fail)
    else:
        def run(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=2) as pool:
                return pool.submit(fail, *args, **kwargs).result()
        monkeypatch.setattr(cli, "run_extraction", run)
    args = _args(tmp_path, "--run", "--db", str(memory[0]))
    if stage == "search":
        assert cli.main(args) == 2
    else:
        with pytest.raises(RuntimeError, match=f"{stage} failed"):
            cli.main(args)
    _assert_closed(connections)
    assert not (tmp_path / "report.json").exists()
    assert _snapshot(memory[0]) == before


def test_calibration_does_not_open_memory(cli, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("calibration opened a deduper")

    monkeypatch.setattr(cli, "build_deduper", forbidden)
    path = tmp_path / "absent" / "memory.sqlite3"
    assert cli.main(_args(tmp_path, "--calibrate", "--db", str(path))) == 0
    assert not path.parent.exists()
    report = json.loads((tmp_path / "report.json").read_text())
    assert "gate_calibration" in report
    assert "dedup" not in report


def test_reader_sees_committed_wal_content(cli, tmp_path):
    path = tmp_path / "wal.sqlite3"
    with MemoryStore(path) as writer:
        with cli.build_deduper(path) as deduper:
            assert not deduper.check(FACT, scope=SCOPE, source_identity=SOURCE).duplicate
            writer.create_node(
                level="trace", content=FACT,
                context={"scope": SCOPE, "source_transcript_sha256": SOURCE[0], "source_span": SOURCE[1]},
            )
            assert deduper.check(FACT, scope=SCOPE, source_identity=SOURCE).duplicate
