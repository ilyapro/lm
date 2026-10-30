"""Recorded dependencies admit bodies without admitting incidental citations."""

import asyncio
from collections import Counter

from fastmcp import Client
import pytest

from living_memory import consolidation as c, server, storage
from living_memory.storage import MemoryStore


SCOPE = "project:harbor"
CORRECTION = "Harbor crane requires the electronic permit."


@pytest.fixture(autouse=True)
def hash_backend(monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as opened:
        yield opened


def sources(store):
    return [store.append_trace(text, {"scope": SCOPE}) for text in (
        "Harbor crane requires the paper permit.",
        "Harbor crane inspection checks the cable tension.",
    )]


def derived(store, name, members, *, strategy="token-jaccard", scope=SCOPE):
    provenance = {"cluster_key": name, "source_traces": [node.id for node in members]}
    if strategy is not None:
        provenance["strategy"] = strategy
    if strategy == "cross-scope-promotion":
        provenance["source_concepts"] = provenance["source_traces"]
    return store.create_node(
        level="concept", content=f"Harbor {name} awaits its recorded evidence.",
        context={"scope": scope}, provenance=provenance,
    )


def incidental(store, source, index=0):
    return store.create_node(
        level="trace", content=f"Independent scheduling lesson {index}.",
        context={"scope": "project:elsewhere", "procedure": ["Unrelated step."]},
        provenance={"source_traces": [source.id], "unrelated": "evidence " * 100},
    )


def prepare(store, source, operation):
    if operation == "teach":
        return c.prepare_teach(store, source.id, CORRECTION)
    return c.prepare_forget(store, source.id)


def change(store, source, operation, preparation=None):
    if operation == "teach":
        return c.memory_teach(store, source.id, CORRECTION, preparation=preparation)
    return c.memory_forget(store, source.id, preparation=preparation)


def test_admission_matches_materializer_policy_and_explicit_fallback(store, monkeypatch):
    source, other = sources(store)
    candidates = []
    for level in ("trace", "concept", "schema"):
        for strategy in (None, "token-jaccard", "embedding-cosine", "procedural",
                         "cross-scope-promotion", "unknown"):
            for key in ("source_traces", "source_concepts"):
                for explicit in (False, True):
                    node = store.create_node(
                        level=level, content=f"Independent candidate {len(candidates)}.",
                        context={"scope": "global"},
                        provenance={"strategy": strategy, "cluster_key": None,
                                    "source_traces": [other.id], key: [source.id]},
                    )
                    store.create_connection(node.id, source.id, "related", metadata=(
                        {"kind": "derived_from"} if explicit else {"kind": "ordinary"}
                    ))
                    candidates.append(node)
    for value in (source.id, {"citation": source.id}):
        candidates.append(store.create_node(
            level="concept", content=f"Non-list promotion inputs: {value}",
            provenance={"strategy": "cross-scope-promotion", "source_concepts": value},
        ))
    candidates.append(store.create_node(
        level="concept", content="An ordinary cited concept without a writer.",
        provenance={"source_traces": [source.id]},
    ))
    legacy = derived(store, "legacy", [source], strategy=None)
    candidates.append(legacy)
    decayed = derived(store, "decayed", [source])
    store.soft_delete_node(decayed.id, "withdrawn")
    expected = {node.id for node in candidates if source.id in c._derivation_sources(store, node)}
    assert legacy.id in expected
    decoded = []
    decode = storage._node_from_row

    def counted(row):
        decoded.append(row["id"])
        return decode(row)

    monkeypatch.setattr(storage, "_node_from_row", counted)
    assert {node.id for node in store.nodes_referencing_source(source.id)} == expected
    assert set(decoded) == expected and len(decoded) == len(expected)

    # Body parsing must remain downstream of admission, including for known
    # writers whose unrelated explicit edge is not one of their actual inputs.
    with store.transaction():
        store.connection.executemany("UPDATE nodes SET embedding = 'broken' WHERE id = ?", (
            (node.id,) for node in candidates if node.id not in expected
        ))
    decoded.clear()
    assert {node.id for node in store.nodes_referencing_source(source.id)} == expected
    assert set(decoded) == expected and len(decoded) == len(expected)


def test_grouped_sources_keep_writer_inputs_and_explicit_edges_separate(store, monkeypatch):
    left, right = sources(store)
    cluster = derived(store, "cluster", [left])
    store.create_connection(cluster.id, right.id, "related", metadata={"kind": "derived_from"})
    promotion = store.create_node(
        level="concept", content="Global harbor promotion.", context={"scope": "global"},
        provenance={"strategy": "cross-scope-promotion", "source_traces": [left.id],
                    "source_concepts": [right.id]},
    )
    opaque = store.create_node(level="schema", content="Opaque shared dependency.")
    for source in (left, right):
        store.create_connection(opaque.id, source.id, "related", metadata={"kind": "derived_from"})
    noise = incidental(store, left)
    with store.transaction():
        store.connection.execute("UPDATE nodes SET embedding = 'broken' WHERE id = ?", (noise.id,))
    decoded = []
    decode = storage._node_from_row

    def counted(row):
        decoded.append(row["id"])
        return decode(row)

    monkeypatch.setattr(storage, "_node_from_row", counted)
    result = store.nodes_referencing_sources([right.id, left.id, right.id, "missing"])
    assert {source: [node.id for node in nodes] for source, nodes in result.items()} == {
        left.id: [cluster.id, opaque.id], right.id: [promotion.id, opaque.id], "missing": [],
    }
    assert Counter(decoded) == {cluster.id: 1, promotion.id: 1, opaque.id: 1}


@pytest.mark.parametrize("width,depth,scans", [(0, 0, 1), (130, 1, 4), (1, 12, 13), (12, 4, 38)])
def test_discovery_preserves_stack_order_with_bounded_pending_groups(
    store, monkeypatch, width, depth, scans,
):
    root = store.append_trace("Harbor root evidence.")
    paths = []
    for branch in reversed(range(width)):
        parent = root
        path = []
        for layer in range(depth):
            parent = store.create_node(
                node_id=f"owner-{branch:03}-{layer:03}", level="concept",
                content=f"Harbor branch {branch} layer {layer}.",
                provenance={"strategy": "token-jaccard", "source_traces": [parent.id]},
            )
            path.append(parent.id)
        paths.insert(0, path)
    for index in range(20):
        incidental(store, root, index)
    groups, decoded = [], []
    read, decode = store.nodes_referencing_sources, storage._node_from_row

    def grouped(ids):
        groups.append(list(ids))
        return read(ids)

    def counted(row):
        decoded.append(row["id"])
        return decode(row)

    monkeypatch.setattr(store, "nodes_referencing_sources", grouped)
    monkeypatch.setattr(storage, "_node_from_row", counted)
    before = store.connection.total_changes
    result = c._affected_derivatives(store, root.id)
    expected = [path[0] for path in paths] + [node for path in reversed(paths) for node in path[1:]]
    assert list(result) == expected
    assert Counter(decoded) == {node: 1 for node in expected}
    assert store.connection.total_changes == before
    assert len(groups) == scans
    assert max(map(len, groups)) <= 64
    if width > 1 and depth > 1:
        # Waiting sibling results stay in one group while newly found depth
        # uses individual reads, rather than retaining nested lookahead groups.
        assert [len(group) for group in groups] == [1, width] + [1] * (width * (depth - 1))


def citation_candidates(store, source):
    """Independent full-candidate reference for complete writer outcome parity."""
    rows = store.connection.execute("""
        SELECT n.* FROM nodes n WHERE n.decayed = 0 AND (
            EXISTS (SELECT 1 FROM json_each(n.source_traces) WHERE value = ?)
            OR EXISTS (SELECT 1 FROM json_each(n.provenance, '$.source_concepts') WHERE value = ?)
            OR n.id IN (SELECT source_id FROM connections
                        WHERE target_id = ? AND type = 'related'
                          AND json_extract(metadata, '$.kind') = 'derived_from')
        )
    """, (source, source, source)).fetchall()
    return [storage._node_from_row(row) for row in rows]


@pytest.mark.parametrize("operation", ["teach", "forget"])
@pytest.mark.parametrize("prepared", [False, True])
def test_complete_correction_refresh_and_reads_match_reference(store, monkeypatch, operation, prepared):
    members = sources(store)
    left = derived(store, "left", members)
    legacy = derived(store, "legacy", members, strategy=None)
    derived(store, "procedural", members, strategy="procedural")
    derived(store, "promotion", [left, legacy], strategy="cross-scope-promotion", scope="global")
    explicit = store.create_node(level="schema", content="Opaque derived restriction.")
    store.create_connection(explicit.id, members[0].id, "related", metadata={"kind": "derived_from"})
    noise = {incidental(store, members[0], i).id for i in range(100)}
    reference = lambda sources: {source: citation_candidates(store, source) for source in sources}
    admitted = store.nodes_referencing_sources
    outcomes, reads = [], []

    class Rollback(Exception):
        pass

    for reader in (reference, admitted):
        decoded, statements = Counter(), []
        decode = storage._node_from_row

        def counted(row):
            decoded["all"] += 1
            decoded["incidental"] += row["id"] in noise
            return decode(row)

        identifiers = iter(f"fixture-id-{i:06}" for i in range(1000))
        with monkeypatch.context() as patch:
            patch.setattr(store, "nodes_referencing_sources", reader)
            patch.setattr(storage, "_node_from_row", counted)
            for module in (c, storage):
                patch.setattr(module, "new_ulid", lambda: next(identifiers))
                patch.setattr(module, "_utc_now", lambda: "2026-01-02T00:00:00Z")
            with pytest.raises(Rollback), store.transaction():
                store.connection.set_trace_callback(statements.append)
                try:
                    preparation = prepare(store, members[0], operation) if prepared else None
                    result = change(store, members[0], operation, preparation)
                finally:
                    store.connection.set_trace_callback(None)
                outcomes.append((result, list(store.connection.iterdump())))
                reads.append((decoded, sum(sql.lstrip().upper().startswith("SELECT") for sql in statements)))
                raise Rollback
    assert outcomes[0] == outcomes[1]
    assert reads[0][0]["incidental"] == 100 * (2 if prepared else 1)
    assert reads[1][0]["incidental"] == 0
    assert reads[0][0]["all"] - reads[1][0]["all"] >= reads[0][0]["incidental"]
    assert reads[1][1] < reads[0][1]


@pytest.mark.parametrize("operation", ["teach", "forget"])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("column,value", [("provenance", "[]"), ("embedding", "broken")])
def test_nondependent_corruption_does_not_veto_source_change(
    store, monkeypatch, operation, prepared, column, value,
):
    members = sources(store)
    owner = derived(store, "release", members)
    noise = incidental(store, members[0])
    with store.transaction():
        store.connection.execute(f"UPDATE nodes SET {column} = ? WHERE id = ?", (value, noise.id))
    before = tuple(store.connection.execute("SELECT * FROM nodes WHERE id = ?", (noise.id,)).fetchone())
    decode = storage._node_from_row

    def only_dependents(row):
        assert row["id"] != noise.id
        return decode(row)

    monkeypatch.setattr(storage, "_node_from_row", only_dependents)
    preparation = prepare(store, members[0], operation) if prepared else None
    change(store, members[0], operation, preparation)
    current = store.get_node(owner.id)
    assert not current.decayed and members[0].content not in current.content
    assert members[1].content in current.content
    assert CORRECTION in current.content if operation == "teach" else CORRECTION not in current.content
    assert tuple(store.connection.execute("SELECT * FROM nodes WHERE id = ?", (noise.id,)).fetchone()) == before


@pytest.mark.parametrize("operation", ["teach", "forget"])
@pytest.mark.parametrize("required", ["owner", "input", "explicit"])
@pytest.mark.parametrize("column,value,error", [
    ("embedding", "broken", ValueError), ("provenance", "[]", TypeError),
])
def test_required_corruption_still_fails_preparation_and_atomic_write(
    store, operation, required, column, value, error,
):
    members = sources(store)
    owner = derived(store, "release", members)
    store.create_connection(owner.id, members[0].id, "related", metadata={"kind": "derived_from"})
    explicit = store.create_node(level="schema", content="Required opaque dependency.")
    store.create_connection(explicit.id, members[0].id, "related", metadata={"kind": "derived_from"})
    preparation = prepare(store, members[0], operation)
    damaged = {"owner": owner, "input": members[1], "explicit": explicit}[required]
    with store.transaction():
        store.connection.execute(f"UPDATE nodes SET {column} = ? WHERE id = ?", (value, damaged.id))
    before = list(store.connection.iterdump())
    with pytest.raises(error):
        prepare(store, members[0], operation)
    assert list(store.connection.iterdump()) == before
    for captured in (None, preparation):
        with pytest.raises(error):
            change(store, members[0], operation, captured)
        assert list(store.connection.iterdump()) == before
        assert not store.connection.in_transaction


@pytest.mark.parametrize("operation", ["teach", "forget"])
def test_writer_rediscovers_new_dependencies_after_preparation(store, operation):
    members = sources(store)
    preparation = prepare(store, members[0], operation)
    with MemoryStore(store.db_path) as writer:
        owner = derived(writer, "new consumer", members, scope="project:elsewhere")
        explicit = writer.create_node(level="schema", content="New opaque dependency.")
        writer.create_connection(explicit.id, members[0].id, "related", metadata={"kind": "derived_from"})
        noise = incidental(writer, members[0])
        with writer.transaction():
            writer.connection.execute("UPDATE nodes SET provenance = '[]' WHERE id = ?", (noise.id,))
    change(store, members[0], operation, preparation)
    current = store.get_node(owner.id)
    assert not current.decayed and members[0].content not in current.content
    assert members[1].content in current.content
    assert store.get_node(explicit.id).decayed


@pytest.mark.parametrize("operation", ["teach", "forget"])
def test_native_source_change_admits_before_both_preparation_and_write(tmp_path, monkeypatch, operation):
    mcp = server.create_mcp_server(tmp_path / "native.sqlite3")
    mcp.auto_consolidation.close()
    store = mcp.memory_store
    try:
        members = sources(store)
        group = c._TraceCluster(SCOPE, members, Counter({"harbor": 2}))
        (owner, _), = c._materialize_cluster(store, group, min_cluster_size=1)
        noise = incidental(store, members[0])
        with store.transaction():
            store.connection.execute("UPDATE nodes SET provenance = '[]' WHERE id = ?", (noise.id,))
        phases = []
        read = store.nodes_referencing_sources
        decode = storage._node_from_row

        def observed(source):
            phases.append(store.connection.in_transaction)
            return read(source)

        def only_dependents(row):
            assert row["id"] != noise.id
            return decode(row)

        monkeypatch.setattr(store, "nodes_referencing_sources", observed)
        monkeypatch.setattr(storage, "_node_from_row", only_dependents)

        async def scenario():
            async with Client(mcp) as client:
                arguments = ({"trace_id": members[0].id, "correction": CORRECTION}
                             if operation == "teach" else {"id": members[0].id})
                result = await client.call_tool(f"memory_{operation}", arguments)
                assert not result.is_error

        asyncio.run(scenario())
        assert set(phases) == {False, True}
        current = store.get_node(owner.id)
        assert not current.decayed and members[0].content not in current.content
        assert members[1].content in current.content
        if operation == "teach":
            assert CORRECTION in current.content
    finally:
        store.close()


def test_native_teach_required_group_read_rolls_back_and_retry_succeeds(tmp_path, monkeypatch):
    mcp = server.create_mcp_server(tmp_path / "native.sqlite3")
    mcp.auto_consolidation.close()
    store = mcp.memory_store
    try:
        members = sources(store)
        left = derived(store, "left", members)
        right = derived(store, "right", members)
        joined = derived(store, "joined", [left, right], scope="global")
        read = store.nodes_referencing_sources
        failed_groups = []

        def fail(ids):
            if len(ids) > 1 and store.connection.in_transaction:
                failed_groups.append(list(ids))
                store.connection.execute("SELECT required_missing_column FROM nodes")
            return read(ids)

        monkeypatch.setattr(store, "nodes_referencing_sources", fail)
        before = list(store.connection.iterdump())

        async def scenario():
            async with Client(mcp) as client:
                arguments = {"trace_id": members[0].id, "correction": CORRECTION}
                failed = await client.call_tool("memory_teach", arguments, raise_on_error=False)
                assert failed.is_error
                assert failed_groups
                assert list(store.connection.iterdump()) == before
                assert not store.connection.in_transaction
                monkeypatch.setattr(store, "nodes_referencing_sources", read)
                return (await client.call_tool("memory_teach", arguments)).data

        result = asyncio.run(scenario())
        replacement = result["corrective_trace"]["id"]
        for owner, inputs in ((left, {replacement, members[1].id}),
                              (right, {replacement, members[1].id}),
                              (joined, {left.id, right.id})):
            current = store.get_node(owner.id)
            assert not current.decayed and set(current.source_traces) == inputs
            assert CORRECTION in current.content and members[0].content not in current.content
            assert {edge.target_id for edge in store.list_connections(
                source_id=owner.id, relation_type="related",
            ) if edge.metadata.get("kind") == "derived_from"} == inputs
    finally:
        store.close()
