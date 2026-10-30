"""Procedural discovery selects writer identities before reading node bodies."""

import json
import sqlite3
from collections import Counter
from dataclasses import replace

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
    if isinstance(expected, type):
        with pytest.raises(expected):
            c.memory_consolidate(store, scope="project:unrelated")


@pytest.mark.parametrize("provenance", [
    None, "", "{bad", "null", "[]", '"procedural"', "1", "false", "{}",
    *[json.dumps({"strategy": strategy}) for strategy in
      [None, "", False, 0, True, 1, "external", "procedural", [], {}, ["procedural"],
       "procedural\0suffix", "\ud800"]],
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


def test_reader_loads_only_selection_metadata_for_large_unrelated_owners(store, monkeypatch):
    content, context, provenance = (label * 16000 for label in
                                    ("unused content ", "unused context ", "writer metadata "))
    store.create_node(
        level="schema", node_id="foreign", content=content,
        context={"trigger": "harbor", "unused": context},
        provenance={"strategy": "external", "unused": provenance},
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
    assert all(content not in str(value) and context not in str(value)
               for row in rows for value in row)
    assert any(provenance in raw for raw in decoded)
    assert all(content not in raw and context not in raw for raw in decoded)


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


def procedural_owner(store, node_id, group, sources=(), *, level="schema"):
    return store.create_node(
        node_id=node_id, level=level, content=f"Harbor records for {group}.",
        context={"scope": "project:harbor", "trigger": "shared label", "task_pattern": group},
        provenance={"strategy": "procedural", "source_traces": list(sources)},
    )


def cited_trace(store, node_id, group="inspection", *, large=False):
    return store.create_node(
        node_id=node_id, level="trace", content=f"Independent harbor record {node_id}.",
        context={"scope": "project:harbor", "task_pattern": group,
                 **({"unused": "x" * 16384} if large else {})},
        provenance={"unused": "y" * 4096} if large else {},
    )


def test_unvisited_concept_group_repeatedly_skips_duplicate_source_bodies(store, monkeypatch):
    owner = procedural_owner(store, "a-current", "inspection", level="concept")
    sources = {cited_trace(store, f"source-{i:04}", large=True).id for i in range(200)}
    duplicate = procedural_owner(store, "z-duplicate", "inspection", sources)
    hydrate = storage._node_from_row
    before = list(store.connection.iterdump())
    for _ in range(2):
        hydrated = Counter()

        def read(row):
            hydrated[row["id"]] += 1
            return hydrate(row)

        with monkeypatch.context() as patch:
            patch.setattr(storage, "_node_from_row", read)
            assert c.memory_consolidate(store, scope="project:unrelated") == c.ConsolidationResult()
        assert hydrated == Counter({owner.id: 1, duplicate.id: 1})
        assert list(store.connection.iterdump()) == before


@pytest.mark.parametrize("invalidity", ["correction", "decay", "supersedes"])
def test_all_scope_repair_ignores_invalid_owners_and_keeps_shared_sources(store, monkeypatch, invalidity):
    unused = cited_trace(store, "unused")
    historical = procedural_owner(store, "a-historical", "inspection", [unused.id])
    source = cited_trace(store, "shared-source")
    owner = procedural_owner(store, "b-current", "inspection", [source.id, source.id, "missing"])
    duplicate = procedural_owner(store, "c-duplicate", "inspection", [unused.id, source.id])
    # The same label does not merge this separate group's owner or its citations.
    other = procedural_owner(store, "d-other-group", "navigation", [source.id])
    if invalidity == "correction":
        store.update_node(historical.id, provenance={**historical.provenance, "corrections": [source.id]})
    elif invalidity == "decay":
        store.soft_delete_node(historical.id, "retired")
    else:
        store.create_connection(source.id, historical.id, "supersedes")
    historical = store.get_node(historical.id)
    incidental = cited_trace(store, "incidental", group="navigation")
    store.create_connection(owner.id, incidental.id, "related", metadata={"kind": "derived_from"})
    hydrated, hydrate = Counter(), storage._node_from_row

    def read(row):
        hydrated[row["id"]] += 1
        return hydrate(row)

    monkeypatch.setattr(storage, "_node_from_row", read)
    result = c.memory_consolidate(store, scope="project:unrelated")
    assert hydrated[unused.id] == hydrated[incidental.id] == 0
    assert [node.id for node in result.schemas_updated] == [owner.id, other.id]
    for updated in result.schemas_updated:
        assert updated.level == "concept" and updated.source_traces == [source.id]
        assert source.content in updated.content and incidental.content not in updated.content
    assert store.get_node(historical.id) == historical
    assert store.get_node(duplicate.id).decay_reason == "procedural: duplicate group"


def test_cited_body_takes_precedence_over_visited_snapshot_for_grouping(store):
    sources = [cited_trace(store, f"source-{i}") for i in range(3)]
    owner = procedural_owner(store, "owner", "orphan", [node.id for node in sources])
    stale = [replace(source, decayed=True) for source in sources]
    result = c._materialize_procedural_schemas(store, stale, min_cluster_size=3)
    assert [node.context["task_pattern"] for node, _ in result] == ["orphan", "inspection"]
    assert result[0][0].id == owner.id
    for node, _ in result:
        assert node.source_traces == [source.id for source in sources]
        assert all(source.content in node.content for source in sources)


def test_cited_source_groups_keep_full_reader_order_across_batches(store, monkeypatch):
    with store.transaction():
        sources = [cited_trace(store, f"source-{i:04}") for i in range(1200)]
    all_ids = [node.id for node in sources]
    # Each read sorts inside its own set-derived batches. Select two groups
    # whose first appearances reverse if their subset is sorted as one batch.
    ordered = list(store.get_nodes(all_ids))
    early, late = ordered[996:999], ordered[999:1002]
    assert max(late) < min(early)
    for group, ids in (("early", early), ("late", late)):
        for source in ids:
            store.update_node(source, context={"scope": "project:harbor", "task_pattern": group})
    owner = procedural_owner(store, "z-orphan", "orphan", [*early, *late])
    store.connection.execute("UPDATE nodes SET created_at='2026-01-01T00:00:00Z' WHERE id=?", (owner.id,))
    procedural_owner(store, "a-duplicate", "orphan", all_ids)
    # Reconstruct the exact source iterable used by discovery, including repeats.
    owners = store.get_nodes(node_id for node_id, _ in store.schema_triggers(None, strategy="procedural"))
    full_order = list(store.get_nodes(source for node in owners.values() for source in node.source_traces))
    selected = set(early + late)
    expected_groups = list(dict.fromkeys(
        "early" if source in early else "late" for source in full_order if source in selected
    ))
    assert expected_groups != ["late", "early"]
    hydrate, hydrated = storage._node_from_row, Counter()

    def read(row):
        hydrated[row["id"]] += 1
        return hydrate(row)

    monkeypatch.setattr(storage, "_node_from_row", read)
    result = c.memory_consolidate(store, scope="project:unrelated")
    assert [node.context["task_pattern"] for node in result.schemas_created] == expected_groups
    assert [node.id for node in result.schemas_updated] == [owner.id]
    assert all(hydrated[source] == 0 for source in set(all_ids) - selected)


@pytest.mark.parametrize("needed", [False, True])
def test_only_required_cited_bodies_must_decode(store, needed):
    source = cited_trace(store, "source")
    procedural_owner(store, "a-owner", "inspection", [source.id], level="schema" if needed else "concept")
    procedural_owner(store, "z-duplicate", "inspection", [source.id])
    # Valid selection metadata, unreadable full body. Exact readers remain strict.
    store.connection.execute("UPDATE nodes SET embedding='[bad' WHERE id=?", (source.id,))
    before = list(store.connection.iterdump())
    if needed:
        with pytest.raises(ValueError):
            c.memory_consolidate(store, scope="project:unrelated")
    else:
        assert c.memory_consolidate(store, scope="project:unrelated") == c.ConsolidationResult()
    assert list(store.connection.iterdump()) == before
    with pytest.raises(ValueError):
        store.get_node(source.id)


@pytest.mark.parametrize("boundary", ["discovery", "validity", "identity", "unused_identity", "body", "write"])
def test_required_procedural_sql_failures_propagate(store, monkeypatch, boundary):
    source = cited_trace(store, "source")
    owner = procedural_owner(store, "owner", "inspection", [source.id],
                             level="concept" if boundary == "unused_identity" else "schema")
    if boundary == "unused_identity":
        procedural_owner(store, "z-duplicate", "inspection", [source.id])
    before = list(store.connection.iterdump())

    def deny(table, field):
        store.connection.set_authorizer(
            lambda action, name, column, *_: sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_READ and name == table and column == field
            else sqlite3.SQLITE_OK
        )

    if boundary == "discovery":
        deny("nodes", "context")
    else:
        method = {"validity": "list_connections", "identity": "get_node_scopes", "unused_identity": "get_node_scopes",
                  "body": "get_nodes", "write": "get_node"}[boundary]
        original = getattr(store, method)

        def read(*args, **kwargs):
            if boundary == "body":
                args = (list(args[0]),)
                if source.id in args[0]:
                    deny("nodes", "content")
            elif boundary == "validity":
                deny("connections", "target_id")
            elif boundary in ("identity", "unused_identity"):
                deny("nodes", "scope")
            elif args[0] == owner.id:
                deny("nodes", "content")
            return original(*args, **kwargs)

        monkeypatch.setattr(store, method, read)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            c.memory_consolidate(store, scope="project:unrelated")
    finally:
        store.connection.set_authorizer(None)
    assert not store.connection.in_transaction
    assert list(store.connection.iterdump()) == before
