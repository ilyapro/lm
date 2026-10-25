"""scripts/lm_collapse_copies.py: one pass collapses accumulated copies, once."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

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
    """Construct historical copies without exercising write-time deduplication."""

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
        # A shared procedure key does not make different content a copy.
        d = _schema(store, body + "\n2. newer step", "dashboard_goal_api_supervision", access=2)
        # Correction history is part of the unowned node's identity.
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
    assert "2 groups, 3 copies to decay" in capsys.readouterr().out


def test_apply_keeps_one_node_per_group_and_rerun_is_a_no_op(copies_db) -> None:
    db, ids = copies_db

    assert collapse.main([str(db), "--apply"]) == 0

    with MemoryStore(db) as store:
        schemas = store.list_nodes(level="schema", scope=SCOPE)
        assert {schema.id for schema in schemas} == {
            ids["b"], ids["d"], ids["corrected"], ids["other_group"]
        }
        keeper = store.get_node(ids["b"])
        assert keeper is not None
        # Most accessed among exact copies, excluding distinct correction history.
        assert keeper.id == ids["b"]
        assert keeper.access_count == 3 + 7 + 1
        assert keeper.usefulness_score == pytest.approx(0.7)
        for name in ("a", "c"):
            node = store.get_node(ids[name])
            assert node is not None and node.decayed is True
            assert node.decay_reason == "duplicate_content"
            edges = store.list_connections(source_id=keeper.id, relation_type="related")
            assert ids[name] in {edge.target_id for edge in edges}
        edge_meta = {
            edge.target_id: edge.metadata
            for edge in store.list_connections(source_id=keeper.id, relation_type="related")
        }
        assert edge_meta[ids["a"]]["kind"] == "duplicate_content"
        assert edge_meta[ids["a"]]["basis"] == "content"
        assert not store.list_connections(relation_type="supersedes")

        for name in ("d", "corrected", "other_group", "other_scope", "lone"):
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


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Dashboard restart is permitted.", "Dashboard restart is not permitted."),
        ("Dashboard restart is allowed.", "Dashboard restart is forbidden."),
        ("Restart after approval.", "Restart after approval and after all work has stopped."),
    ],
)
def test_same_procedure_key_preserves_distinct_statements(tmp_path, left, right) -> None:
    db = tmp_path / "different.sqlite3"
    with MemoryStore(db) as store:
        first = _schema(store, left, "dashboard restart")
        second = _schema(store, right, "dashboard restart")

    assert collapse.main([str(db), "--apply"]) == 0

    with MemoryStore(db) as store:
        assert {node.id for node in store.list_nodes()} == {first.id, second.id}
        assert store.list_connections() == []


@pytest.mark.parametrize(
    ("left_context", "right_context", "left_provenance", "right_provenance", "collapse_expected"),
    [
        ({}, {}, {"strategy": "procedural", "source_traces": ["source-a"]},
         {"strategy": "procedural", "source_traces": ["source-b"]}, False),
        ({"procedure_key": "one"}, {"procedure_key": "two"},
         {"strategy": "procedural", "source_traces": ["source-a"]},
         {"strategy": "procedural", "source_traces": ["source-a"]}, False),
        ({}, {}, {"strategy": "procedural", "source_traces": ["source-a"]}, {}, False),
        ({}, {}, {"strategy": "procedural"}, {"strategy": "procedural"}, False),
        ({"procedure_key": "one_group"}, {"procedure_key": "one group"},
         {"strategy": "procedural", "source_traces": ["source-a", "source-b"]},
         {"strategy": "procedural", "source_traces": ["source-b", "source-a"]}, True),
    ],
)
def test_exact_content_requires_matching_derivation(
    tmp_path, left_context, right_context, left_provenance, right_provenance, collapse_expected
) -> None:
    db = tmp_path / "derivations.sqlite3"
    with MemoryStore(db) as store:
        first = store.create_node(
            level="schema", content="first", context={"scope": SCOPE, **left_context},
            provenance=left_provenance,
        )
        second = store.create_node(
            level="schema", content="second", context={"scope": SCOPE, **right_context},
            provenance=right_provenance,
        )
    _make_identical(db, "Restart only after approval.", first.id, second.id)

    assert collapse.main([str(db), "--apply"]) == 0

    with MemoryStore(db) as store:
        assert len(store.list_nodes()) == (1 if collapse_expected else 2)
        assert len(store.list_connections(relation_type="related")) == int(collapse_expected)
        assert not store.list_connections(relation_type="supersedes")


@pytest.mark.parametrize("difference", ["scope", "level"])
def test_exact_content_preserves_scope_and_level(tmp_path, difference) -> None:
    db = tmp_path / "boundaries.sqlite3"
    with MemoryStore(db) as store:
        first = store.create_node(level="trace", content="first", context={"scope": SCOPE})
        second = store.create_node(
            level="concept" if difference == "level" else "trace", content="second",
            context={"scope": "project:other" if difference == "scope" else SCOPE},
        )
    _make_identical(db, "Shared bytes", first.id, second.id)

    assert collapse.main([str(db), "--apply"]) == 0

    with MemoryStore(db) as store:
        assert len(store.list_nodes()) == 2
        assert store.list_connections() == []


@pytest.mark.parametrize(
    ("metadata", "is_correction"),
    [
        ({"kind": "drain_near_duplicate"}, False),
        ({"kind": "duplicate_content", "basis": "procedure_key"}, False),
        ({"by": "lm_collapse_near_dups.py", "reason": "near-duplicate collapse"}, False),
        ({"kind": "teach"}, True),
        ({"kind": "unknown"}, True),
        ({}, True),
    ],
)
def test_keeper_ranking_uses_genuine_corrections(tmp_path, metadata, is_correction) -> None:
    db = tmp_path / "correction.sqlite3"
    with MemoryStore(db) as store:
        popular = _schema(store, "popular", "restart", access=10)
        alternative = _schema(store, "alternative", "restart", access=1)
        replacement = _schema(store, "Correction: stop before restart.", "restart")
        store.create_connection(replacement.id, popular.id, "supersedes", metadata=metadata)
    _make_identical(db, "Restart only after approval.", popular.id, alternative.id)

    assert collapse.main([str(db), "--apply"]) == 0

    with MemoryStore(db) as store:
        kept = alternative.id if is_correction else popular.id
        assert {node.id for node in store.list_nodes()} == {kept, replacement.id}


def test_anchor_stats_and_existing_relation_metadata_are_merged(copies_db) -> None:
    db, ids = copies_db
    with MemoryStore(db) as store:
        store.create_connection(ids["b"], ids["a"], "related", metadata={"note": "retained"})
    with sqlite3.connect(db) as connection:
        connection.execute(
            """INSERT INTO query_anchors
               (id, scope, query, fingerprint, dimensions, embedding, first_seen, created_at, updated_at)
               VALUES ('anchor', ?, 'restart approval', 'query-fingerprint', 1, ?, 'now', 'now', 'now')""",
            (SCOPE, bytes(4)),
        )
        connection.executemany(
            """INSERT INTO query_anchor_edges
               (anchor_id, target_id, weight, hits, created_at, updated_at)
               VALUES ('anchor', ?, ?, ?, 'now', 'now')""",
            [(ids["a"], 0.8, 3), (ids["b"], 0.2, 7)],
        )

    assert collapse.main([str(db), "--apply"]) == 0

    with MemoryStore(db) as store:
        anchors = store.list_query_anchor_edges(anchor_id="anchor")
        assert len(anchors) == 1
        assert anchors[0].target_id == ids["b"]
        assert anchors[0].hits == 10
        assert anchors[0].weight == 0.8
        edge = next(edge for edge in store.list_connections(source_id=ids["b"], relation_type="related")
                    if edge.target_id == ids["a"])
        assert edge.metadata["kind"] == "duplicate_content"
        assert edge.metadata["note"] == "retained"
