"""Procedural discovery selects writer identities before reading node bodies."""

import json
import sqlite3
from collections import Counter

import pytest

from living_memory import consolidation as c, storage
from living_memory.retrieval import memory_recall
from living_memory.storage import MemoryStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    with MemoryStore(tmp_path / "memory.sqlite3") as opened:
        yield opened


def selection_after_hydration(store, scope, strategy):
    triggers = store.schema_triggers(scope)
    nodes = store.get_nodes(node_id for node_id, _ in triggers)
    return [(node_id, trigger) for node_id, trigger in triggers
            if nodes[node_id].provenance.get("strategy") == strategy]


def outcome(read):
    try:
        return read()
    except (ValueError, TypeError, sqlite3.OperationalError) as error:
        return type(error)


@pytest.mark.parametrize("context", [
    None, "", "{bad", "null", "[]", '"trigger"', "1", "false", "{}",
    *[json.dumps({"trigger": trigger}) for trigger in
      [None, "", False, 0, 0.0, True, 1, -2.5, "0", " ", "harbor", [], {}, [1]]],
    '{"trigger":"first","trigger":"last"}',
])
def test_trigger_representations_match_discovery_before_hydration(store, context):
    store.create_node(level="schema", node_id="owner", content="Harbor departure procedure.")
    # Nullable legacy fields use the same row layout without modern NOT NULL constraints.
    store.connection.execute("CREATE TEMP TABLE nodes AS SELECT * FROM main.nodes")
    store.connection.execute("UPDATE nodes SET context=?, provenance=?",
                             (context, '{"strategy":"procedural"}'))
    expected = outcome(lambda: selection_after_hydration(store, None, "procedural"))
    assert outcome(lambda: store.schema_triggers(None, strategy="procedural")) == expected


@pytest.mark.parametrize("provenance", [
    None, "", "{bad", "null", "[]", '"procedural"', "1", "false", "{}",
    *[json.dumps({"strategy": strategy}) for strategy in
      [None, "", False, 0, True, 1, "external", "procedural", [], {}, ["procedural"]]],
    '{"strategy":"external","strategy":"procedural"}',
    '{"strategy":"procedural","strategy":"external"}',
    '{"strategy":"procedural","strategy":[]}',
    '{"strategy":"procedural","unused":NaN}',
    '{"strategy":"procedural","unused":Infinity}',
    '{"strategy":"procedural","unused":123456789012345678901234567890}',
])
def test_provenance_representations_match_hydration_before_selection(store, provenance):
    store.create_node(level="concept", node_id="owner", content="Harbor departure evidence.",
                      context={"scope": "project:other", "trigger": "harbor"})
    store.connection.execute("CREATE TEMP TABLE nodes AS SELECT * FROM main.nodes")
    store.connection.execute("UPDATE nodes SET provenance=?", (provenance,))
    expected = outcome(lambda: selection_after_hydration(store, None, "procedural"))
    assert outcome(lambda: store.schema_triggers(None, strategy="procedural")) == expected
    if isinstance(expected, type):
        with pytest.raises(expected):
            c.memory_consolidate(store, scope="project:unrelated")


@pytest.mark.parametrize("context,level,decayed,scope", [
    ('{"trigger":""}', "schema", 0, None),
    ('{"trigger":false}', "concept", 0, None),
    ('{"trigger":0}', "schema", 0, None),
    ('{}', "concept", 0, None),
    ('{bad', "trace", 0, None),
    ('{bad', "schema", 1, None),
    ('{bad', "concept", 0, "project:elsewhere"),
])
def test_untriggered_inactive_and_out_of_scope_rows_do_not_validate_provenance(
    store, context, level, decayed, scope,
):
    store.create_node(level=level, node_id="owner", content="Unselected harbor record.")
    store.connection.execute("CREATE TEMP TABLE nodes AS SELECT * FROM main.nodes")
    store.connection.execute(
        "UPDATE nodes SET context=?, provenance='{bad', decayed=?", (context, decayed),
    )
    assert selection_after_hydration(store, scope, "procedural") == []
    assert store.schema_triggers(scope, strategy="procedural") == []


@pytest.mark.parametrize("strategy", ["", "external", "[]", "{}", "1", "true"])
def test_strategy_selector_matches_strings_without_coercing_other_types(store, strategy):
    for index, value in enumerate([strategy, [], {}, 1, True]):
        store.create_node(
            level="schema", node_id=f"owner-{index}", content=f"Harbor procedure {index}.",
            context={"trigger": "harbor"}, provenance={"strategy": value},
        )
    assert store.schema_triggers(None, strategy=strategy) == [("owner-0", "harbor")]
    assert len(store.schema_triggers(None)) == 5


def test_reader_projects_only_trigger_and_writer_for_large_unrelated_owners(store, monkeypatch):
    marker = "unused payload " * 16000
    store.create_node(
        level="schema", node_id="foreign", content=marker,
        context={"trigger": "harbor", "unused": marker},
        provenance={"strategy": "external", "unused": marker},
    )
    store.create_node(level="concept", node_id="owner", content="Harbor case records.",
                      context={"trigger": "harbor"}, provenance={"strategy": "procedural"})
    rows, decoded = [], []
    loads = json.loads

    def row_factory(cursor, values):
        rows.append(values)
        return sqlite3.Row(cursor, values)

    def decode(raw, *args, **kwargs):
        decoded.append(raw)
        return loads(raw, *args, **kwargs)

    monkeypatch.setattr(json, "loads", decode)
    store.connection.row_factory = row_factory
    try:
        assert store.schema_triggers(None, strategy="procedural") == [("owner", "harbor")]
    finally:
        store.connection.row_factory = sqlite3.Row
    assert len(rows) == 2
    assert all(len(row) == 3 for row in rows)
    assert all(marker not in str(value) for row in rows for value in row)
    assert decoded and all(marker not in raw for raw in decoded)


def test_ordinary_consolidate_preserves_state_without_hydrating_foreign_bodies(store, monkeypatch):
    noise = set()
    for index in range(200):
        node = store.create_node(
            level="schema" if index % 2 else "concept", content=f"Independent evidence {index}.",
            context={"scope": "project:noise", "trigger": f"noise {index}", "unused": "x" * 16384},
            provenance={"strategy": "external", "unused": "y" * 4096},
        )
        noise.add(node.id)
    sources = [store.append_trace(
        f"Harbor inspection case {index}: retain the complete account.",
        {"scope": "project:orphan", "task_pattern": "harbor inspection"},
    ) for index in range(2)]
    owner = store.create_node(
        level="schema", node_id="oldest", content="Harbor inspection numbered case history.",
        context={"scope": "project:orphan", "trigger": "harbor inspection",
                 "task_pattern": "harbor inspection"},
        provenance={"strategy": "procedural", "source_traces": [n.id for n in sources],
                    "source_coverage": {sources[0].id: "2026-01-01T00:00:00Z", sources[1].id: None}},
    )
    store.record_access(owner.id)
    duplicate = store.create_node(
        level="schema", node_id="younger", content=owner.content + " Duplicate account.",
        context=owner.context, provenance=owner.provenance,
    )
    incidental = store.append_trace("Unrelated docking record.", {"scope": "project:incidental"})
    store.create_connection(owner.id, incidental.id, "related", metadata={"kind": "derived_from"})
    reader, hydrate = store.schema_triggers, storage._node_from_row
    snapshots, measurements = [], []

    class Rollback(Exception):
        pass

    for narrow in (False, True):
        counts = Counter()
        ids = iter(f"generated-{i:04}" for i in range(1000))

        def measured(row):
            counts["full_nodes"] += 1
            counts["metadata_chars"] += len(row["context"]) + len(row["provenance"])
            counts["foreign_nodes"] += row["id"] in noise
            return hydrate(row)

        with monkeypatch.context() as patch:
            patch.setattr(storage, "_node_from_row", measured)
            patch.setattr(storage, "new_ulid", lambda: next(ids))
            patch.setattr(c, "new_ulid", lambda: next(ids))
            patch.setattr(storage, "_utc_now", lambda: "2026-10-30T00:00:00Z")
            patch.setattr(c, "_utc_now", lambda: "2026-10-30T00:00:00Z")
            if not narrow:
                # The consumer's existing Python strategy check follows full hydration.
                patch.setattr(store, "schema_triggers", lambda scope, **kwargs: reader(scope))
            with pytest.raises(Rollback), store.transaction():
                result = c.memory_consolidate(store, scope="project:unrelated")
                measurements.append(dict(counts))
                assert [node.id for node in result.schemas_updated] == [owner.id]
                current = store.get_node(owner.id)
                assert current.level == "concept" and not current.decayed
                assert set(current.source_traces) == {n.id for n in sources}
                assert current.provenance["source_coverage"] == owner.provenance["source_coverage"]
                assert current.access_count == 1
                assert all(n.content in current.content for n in sources)
                assert incidental.content not in current.content
                assert store.get_node(duplicate.id).decayed
                # Include nodes, edges, feedback, indexes and every other durable row.
                snapshots.append(list(store.connection.iterdump()))
                raise Rollback
    assert snapshots[0] == snapshots[1]
    before, after = measurements
    assert before["foreign_nodes"] == 200 and after["foreign_nodes"] == 0
    assert before["full_nodes"] - after["full_nodes"] == 200
    assert before["metadata_chars"] - after["metadata_chars"] > 4_000_000


@pytest.mark.parametrize("level,strategy", [("schema", None), ("concept", "external"),
                                            ("schema", "procedural"), ("concept", "procedural")])
def test_ordinary_recall_keeps_schemas_and_carriers_of_every_writer(store, level, strategy):
    owner = store.create_node(
        level=level, content="Harbor inspection: verify the manifest before departure.",
        context={"scope": "project:harbor", "trigger": "harbor inspection"},
        provenance={"strategy": strategy},
    )
    results = memory_recall(store, "harbor inspection", scope="project:harbor", max_results=5)
    assert owner.id in [result.node.id for result in results]
