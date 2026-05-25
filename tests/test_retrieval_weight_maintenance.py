from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from living_memory.maintenance import reseat_retrieval_floors
from living_memory.storage import MemoryStore


def test_reseat_floors_dry_run_reports_audit_without_writing(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    _seed_collapsed_project(db_path, "project:collapsed")

    result = _run_maintenance(db_path, "--dry-run")

    assert result["mode"] == "dry-run"
    assert result["applied"] is False
    assert result["affected_count"] == 1
    change = result["changes"][0]
    assert change["scope"] == "project:collapsed"
    assert change["family"] == "project"
    assert change["floor"] == {"bm25_max": 0.85, "graph_min": 0.05, "vector_min": 0.15}
    assert change["before"]["bm25"] == pytest.approx(1.0)
    assert change["before"]["vector"] == pytest.approx(0.0)
    assert change["after"]["bm25"] == pytest.approx(0.85)
    assert change["after"]["vector"] == pytest.approx(0.15)

    with MemoryStore(db_path) as store:
        persisted = store.get_retrieval_weights("project:collapsed")
        assert persisted.bm25 == pytest.approx(1.0)
        assert persisted.vector == pytest.approx(0.0)


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


def _run_maintenance(db_path: Path, mode: str) -> dict[str, object]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "living_memory.maintenance",
            "reseat-floors",
            "--db",
            str(db_path),
            mode,
        ],
        check=True,
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)
