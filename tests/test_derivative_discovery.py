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
