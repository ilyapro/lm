from __future__ import annotations

from contextlib import closing
from pathlib import Path
import json
import os
import sqlite3
import subprocess
import sys

import pytest

from living_memory.maintenance import reseat_retrieval_floors
from living_memory.storage import MemoryStore


@pytest.mark.parametrize("mode", [(), ("--dry-run",)], ids=["default", "dry-run"])
def test_reseat_floors_dry_run_reports_audit_without_writing(
    tmp_path: Path, mode: tuple[str, ...],
) -> None:
    db_path = tmp_path / "memory #?.sqlite3"
    _seed_collapsed_project(db_path, "project:collapsed")
    with closing(sqlite3.connect(db_path)) as connection, connection:
        connection.execute("UPDATE nodes SET content_fingerprint = NULL")
    before = _database_snapshot(db_path)

    result = _run_maintenance(db_path, *mode)

    assert result["mode"] == "dry-run"
    assert result["applied"] is False
    assert result["affected_count"] == 1
    change = result["changes"][0]
    assert change["scope"] == "project:collapsed"
    assert change["family"] == "project"
    assert change["floor"] == {
        "bm25_max": 0.85,
        "graph_min": 0.05,
        "vector_min": 0.15,
        "bm25_min": 0.10,
    }
    assert change["before"]["bm25"] == pytest.approx(1.0)
    assert change["before"]["vector"] == pytest.approx(0.0)
    assert change["after"]["bm25"] == pytest.approx(0.85)
    assert change["after"]["vector"] == pytest.approx(0.15)

    assert _database_snapshot(db_path) == before
    with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as connection:
        assert connection.execute(
            "SELECT content_fingerprint FROM nodes"
        ).fetchone() == (None,)


@pytest.mark.parametrize("mode", [(), ("--dry-run",)], ids=["default", "dry-run"])
@pytest.mark.parametrize("missing_parent", [False, True])
def test_preview_missing_database_fails_without_creating_paths(
    tmp_path: Path, mode: tuple[str, ...], missing_parent: bool,
) -> None:
    parent = tmp_path / "absent" if missing_parent else tmp_path
    db_path = parent / "absent.sqlite3"

    result = _invoke_maintenance(db_path, *mode)

    assert result.returncode == 2
    assert result.stdout == ""
    assert "reseat-floors:" in result.stderr
    assert not db_path.exists()
    assert parent.exists() is not missing_parent


@pytest.mark.parametrize("mode", [(), ("--dry-run",)], ids=["default", "dry-run"])
@pytest.mark.parametrize("schema", ["empty", "weights-only", "invalid"])
def test_preview_incompatible_database_fails_without_initialization(
    tmp_path: Path, mode: tuple[str, ...], schema: str,
) -> None:
    db_path = tmp_path / "incompatible.sqlite3"
    if schema == "invalid":
        db_path.write_text("not a SQLite database", encoding="utf-8")
    else:
        with closing(sqlite3.connect(db_path)) as connection, connection:
            if schema == "weights-only":
                connection.execute(
                    "CREATE TABLE retrieval_weights (scope TEXT, bm25 REAL, vector REAL, "
                    "graph REAL, learning_rate REAL, updated_at TEXT)"
                )
    before = db_path.read_bytes()

    result = _invoke_maintenance(db_path, *mode)

    assert result.returncode == 2
    assert result.stdout == ""
    assert "reseat-floors:" in result.stderr
    assert db_path.read_bytes() == before


def test_preview_reads_committed_wal_without_changing_data(tmp_path: Path) -> None:
    db_path = tmp_path / "wal.sqlite3"
    _seed_collapsed_project(db_path, "project:collapsed")
    with MemoryStore(db_path) as writer:
        writer.connection.execute("PRAGMA wal_autocheckpoint = 0")
        writer.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        with writer.connection:
            writer.connection.execute("UPDATE nodes SET content_fingerprint = NULL")
            writer.connection.execute(
                "UPDATE retrieval_weights SET bm25 = 0.99, vector = 0.01 "
                "WHERE scope = 'project:collapsed'"
            )
        wal_path = Path(str(db_path) + "-wal")
        wal_before = wal_path.read_bytes()
        assert wal_before
        before = _database_snapshot(db_path)

        result = _run_maintenance(db_path)

        assert result["changes"][0]["before"]["bm25"] == pytest.approx(0.99)
        assert _database_snapshot(db_path) == before
        assert wal_path.read_bytes() == wal_before


def test_preview_uses_configured_family_floors_and_matches_apply(tmp_path: Path) -> None:
    db_path = tmp_path / "configured.sqlite3"
    _seed_collapsed_project(db_path, "project:collapsed")
    with MemoryStore(db_path) as store:
        source = store.create_node(
            level="trace", content="cause", context={"scope": "global"},
            embedding=[1.0, 0.0],
        )
        target = store.create_node(
            level="trace", content="effect", context={"scope": "global"},
        )
        store.create_connection(source.id, target.id, "caused")
        for scope in ("global", "project:no-evidence"):
            store.set_retrieval_weights(scope, bm25=1.0, vector=0.0, graph=0.0)
        store.set_retrieval_weights("session:lexical", bm25=0.0, vector=1.0, graph=0.0)
        with store.connection:
            store.connection.execute("UPDATE nodes SET content_fingerprint = NULL")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""[storage]
db_path = {json.dumps(str(db_path))}
[retrieval_policy_floors.project]
bm25_max = 0.60
vector_min = 0.35
[retrieval_policy_floors.global]
bm25_max = 0.55
vector_min = 0.30
graph_min = 0.15
[retrieval_policy_floors.session]
bm25_min = 0.25
""", encoding="utf-8",
    )
    before = _database_snapshot(db_path)

    preview = _run_maintenance(None, "--config", str(config_path))
    explicit_preview = _run_maintenance(None, "--config", str(config_path), "--dry-run")
    assert preview == explicit_preview
    assert _database_snapshot(db_path) == before
    applied = _run_maintenance(None, "--config", str(config_path), "--apply")

    assert applied["changes"] == preview["changes"]
    expected = {
        "global": (0.55, 0.30, 0.15),
        "project:collapsed": (0.60, 0.40, 0.0),
        "session:lexical": (0.25, 0.75, 0.0),
    }
    assert applied["affected_count"] == len(expected)
    for change in preview["changes"]:
        weights = change["after"]
        assert tuple(weights[key] for key in ("bm25", "vector", "graph")) == pytest.approx(
            expected[change["scope"]]
        )
    with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as connection:
        for scope, weights in {**expected, "project:no-evidence": (1.0, 0.0, 0.0)}.items():
            assert connection.execute(
                "SELECT bm25, vector, graph FROM retrieval_weights WHERE scope = ?", (scope,),
            ).fetchone() == pytest.approx(weights)
        assert connection.execute(
            "SELECT COUNT(*) FROM nodes WHERE content_fingerprint IS NULL"
        ).fetchone() == (0,)


def test_reseat_floors_apply_reports_and_persists_audited_change(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    _seed_collapsed_project(db_path, "project:collapsed")

    result = _run_maintenance(db_path, "--apply")

    assert result["mode"] == "apply"
    assert result["applied"] is True
    assert result["affected_count"] == 1
    change = result["changes"][0]
    assert change["before"]["bm25"] == pytest.approx(1.0)
    assert change["before"]["vector"] == pytest.approx(0.0)
    assert change["after"]["bm25"] == pytest.approx(0.85)
    assert change["after"]["vector"] == pytest.approx(0.15)

    with MemoryStore(db_path) as store:
        persisted = store.get_retrieval_weights("project:collapsed").normalized()
        assert persisted.bm25 == pytest.approx(0.85)
        assert persisted.vector == pytest.approx(0.15)
        assert persisted.graph == pytest.approx(0.0)


def test_reseat_floors_uses_storage_evidence_gating_not_separate_clamp(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "memory.sqlite3"
    with MemoryStore(db_path) as store:
        store.set_retrieval_weights(
            "project:no-evidence",
            bm25=1.0,
            vector=0.0,
            graph=0.0,
            learning_rate=0.05,
        )

        changes = reseat_retrieval_floors(store, apply=True)
        persisted = store.get_retrieval_weights("project:no-evidence").normalized()

    assert changes == []
    assert persisted.bm25 == pytest.approx(1.0)
    assert persisted.vector == pytest.approx(0.0)
    assert persisted.graph == pytest.approx(0.0)


def test_reseat_floors_reuses_global_vector_and_graph_floor_logic(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "memory.sqlite3"
    with MemoryStore(db_path) as store:
        source = store.create_node(
            level="trace",
            content="global root cause",
            context={"scope": "global"},
            embedding=[1.0, 0.0],
        )
        target = store.create_node(
            level="trace",
            content="global downstream effect",
            context={"scope": "global"},
            embedding=[0.0, 1.0],
        )
        store.create_connection(source.id, target.id, "caused")
        store.set_retrieval_weights(
            "global",
            bm25=1.0,
            vector=0.0,
            graph=0.0,
            learning_rate=0.05,
        )

        changes = reseat_retrieval_floors(store, apply=True)
        persisted = store.get_retrieval_weights("global").normalized()

    assert [change.scope for change in changes] == ["global"]
    assert changes[0].after.bm25 == pytest.approx(0.75)
    assert changes[0].after.vector == pytest.approx(0.20)
    assert changes[0].after.graph == pytest.approx(0.05)
    assert persisted.bm25 == pytest.approx(0.75)
    assert persisted.vector == pytest.approx(0.20)
    assert persisted.graph == pytest.approx(0.05)


def _seed_collapsed_project(db_path: Path, scope: str) -> None:
    with MemoryStore(db_path) as store:
        store.create_node(
            level="trace",
            content="checkout deploy migration recovery",
            context={"scope": scope},
            embedding=[1.0, 0.0],
        )
        store.set_retrieval_weights(
            scope,
            bm25=1.0,
            vector=0.0,
            graph=0.0,
            learning_rate=0.05,
        )


def _database_snapshot(db_path: Path) -> tuple[str, ...]:
    with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as connection:
        return tuple(connection.iterdump())


def _run_maintenance(db_path: Path | None, *args: str) -> dict[str, object]:
    completed = _invoke_maintenance(db_path, *args)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _invoke_maintenance(db_path: Path | None, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "living_memory.maintenance",
            "reseat-floors",
            *(["--db", str(db_path)] if db_path is not None else []),
            *args,
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
