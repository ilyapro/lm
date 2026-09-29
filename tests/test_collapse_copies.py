"""scripts/lm_collapse_copies.py: one pass collapses accumulated copies, once."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from living_memory.consolidation import _normalize_trigger, _schema_group_id
from living_memory.storage import MemoryStore

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "lm_collapse_copies.py"
SCOPE = "project:ae"


def _load_script():
    spec = importlib.util.spec_from_file_location("lm_collapse_copies", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


collapse = _load_script()


def _schema(store: MemoryStore, content: str, key: str, *, scope: str = SCOPE, access: int = 0):
    return store.create_node(
        level="schema",
        content=content,
        context={"scope": scope, "procedure_key": key, "task_pattern": key.replace(" ", "_")},
        stats={"access_count": access, "usefulness_score": 0.1 * access},
        provenance={"procedure_key": key},
    )


def _make_identical(db: Path, content: str, *node_ids: str) -> None:
    """Rewrite nodes to the same bytes, the way the live copies look."""

    fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
    with sqlite3.connect(db) as connection:
        connection.executemany(
            "UPDATE nodes SET content = ?, content_fingerprint = ? WHERE id = ?",
            [(content, fingerprint, node_id) for node_id in node_ids],
        )


def _dump(db: Path) -> list[str]:
    with sqlite3.connect(db) as connection:
        return list(connection.iterdump())


@pytest.fixture()
def copies_db(tmp_path: Path):
    db = tmp_path / "memory.sqlite3"
    body = "Procedure: dashboard goal api supervision\n1. use the loopback api"
    with MemoryStore(db) as store:
        a = _schema(store, body + " a", "dashboard goal api supervision", access=3)
        b = _schema(store, body + " b", "dashboard goal api supervision", access=7)
        c = _schema(store, body + " c", "dashboard goal api supervision", access=1)
        # Same procedure key, different content: still a copy of the group.
        d = _schema(store, body + "\n2. newer step", "dashboard_goal_api_supervision", access=2)
        # A corrected schema is never kept, however used it is.
        corrected = _schema(store, body + " e", "dashboard goal api supervision", access=50)
        other_group = _schema(store, body + " f", "active goal supervision", access=4)
        other_scope = _schema(store, body, "dashboard goal api supervision", scope="project:x")
        concept_1 = store.create_node(level="concept", content="concept body 1", context={"scope": SCOPE})
        concept_2 = store.create_node(level="concept", content="concept body 2", context={"scope": SCOPE})
        lone = store.create_node(level="concept", content="lone concept", context={"scope": SCOPE})
    _make_identical(db, body, a.id, b.id, c.id, corrected.id)
    _make_identical(db, "concept body", concept_1.id, concept_2.id)
    with sqlite3.connect(db) as connection:
        connection.execute(
            "UPDATE nodes SET corrections = ? WHERE id = ?",
            (json.dumps(["01CORRECTION"]), corrected.id),
        )
        connection.execute(
            "UPDATE nodes SET access_count = 5, last_accessed = '2026-09-28T00:00:00Z' WHERE id = ?",
            (concept_1.id,),
        )
    ids = {
        "a": a.id,
        "b": b.id,
        "c": c.id,
        "d": d.id,
        "corrected": corrected.id,
        "other_group": other_group.id,
        "other_scope": other_scope.id,
        "concept_1": concept_1.id,
        "concept_2": concept_2.id,
        "lone": lone.id,
    }
    return db, ids


def test_dry_run_changes_nothing(copies_db, capsys) -> None:
    db, _ids = copies_db
    before = _dump(db)

    assert collapse.main([str(db)]) == 0

    assert _dump(db) == before
    assert "2 groups, 5 copies to decay" in capsys.readouterr().out


def test_apply_keeps_one_node_per_group_and_rerun_is_a_no_op(copies_db) -> None:
    db, ids = copies_db

    assert collapse.main([str(db), "--apply"]) == 0

    with MemoryStore(db) as store:
        schemas = store.list_nodes(level="schema", scope=SCOPE)
        by_key: dict[str, list] = {}
        for schema in schemas:
            by_key.setdefault(_schema_group_id(schema), []).append(schema)
        assert {key: len(nodes) for key, nodes in by_key.items()} == {
            "dashboard goal api supervision": 1,
            "active goal supervision": 1,
        }
        keeper = by_key["dashboard goal api supervision"][0]
        # Most accessed among the updatable ones; the corrected node stays down.
        assert keeper.id == ids["b"]
        assert keeper.access_count == 3 + 7 + 1 + 2 + 50
        for name in ("a", "c", "d", "corrected"):
            node = store.get_node(ids[name])
            assert node is not None and node.decayed is True
            assert node.decay_reason == "duplicate_content"
            edges = store.list_connections(source_id=keeper.id, relation_type="supersedes")
            assert ids[name] in {edge.target_id for edge in edges}
        edge_meta = {
            edge.target_id: edge.metadata
            for edge in store.list_connections(source_id=keeper.id, relation_type="supersedes")
        }
        assert edge_meta[ids["a"]]["kind"] == "duplicate_content"
        assert edge_meta[ids["a"]]["basis"] == "content"
        assert edge_meta[ids["d"]]["basis"] == "procedure_key"

        for name in ("other_group", "other_scope", "lone"):
            node = store.get_node(ids[name])
            assert node is not None and node.decayed is False

        concepts = store.list_nodes(level="concept", scope=SCOPE)
        assert sorted(node.id for node in concepts) == sorted([ids["concept_1"], ids["lone"]])
        kept_concept = store.get_node(ids["concept_1"])
        assert kept_concept is not None
        assert kept_concept.access_count == 5
        assert kept_concept.last_accessed == "2026-09-28T00:00:00Z"

    after_first = _dump(db)
    assert collapse.main([str(db), "--apply"]) == 0
    assert _dump(db) == after_first


def test_group_key_mirrors_consolidation() -> None:
    for raw in ("dashboard_goal_api_supervision", " Tree-Decomposition/Build ", "208b33b1"):
        assert collapse.normalize_trigger(raw) == _normalize_trigger(raw)

    class _Schema:
        def __init__(self, context, provenance):
            self.context = context
            self.provenance = provenance

    shapes = (
        ({"procedure_key": "a b", "procedure_id": "c"}, {}),
        ({"task_pattern": "task_x", "procedure_id": "proc_y"}, {}),
        ({"procedure_id": "proc_y"}, {}),
        ({}, {"procedure_key": "from provenance"}),
    )
    for context, provenance in shapes:
        assert collapse.schema_group_id(context, provenance) == _schema_group_id(
            _Schema(context, provenance)
        )
