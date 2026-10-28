"""Explicit forgetting updates current derivations while retaining history."""

import asyncio
from dataclasses import asdict
import sqlite3

from fastmcp import Client
import pytest

from living_memory import consolidation as c, decay, server
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.storage import MemoryStore


SCOPE = "project:workshop"
TEXTS = [
    "Workshop release requires the retired paper permit.",
    "Workshop release checks the package checksum.",
    "Workshop release records the new package version.",
]


@pytest.fixture(autouse=True)
def deterministic_runtime(monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_DECAY_SWEEP_INTERVAL_SEC", "0")
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "0.35")
    monkeypatch.setenv("LM_RECALL_GATE_FORM", "drop")
    monkeypatch.setenv("LM_RECALL_MAP", "0")
    monkeypatch.setenv("LM_DELIVERY_SESSION_DEDUP", "0")
    monkeypatch.setattr(LocalEmbeddingModel, "embed", lambda self, text: [1.0, 0.0])


@pytest.fixture
def store(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite3") as instance:
        yield instance


def sources(store, *, scope=SCOPE):
    return [store.append_trace(text, {"scope": scope}) for text in TEXTS]


def derived(store, name, members, *, scope=SCOPE, strategy="token-jaccard"):
    provenance = {"strategy": strategy, "cluster_key": name,
                  "source_traces": [member.id for member in members]}
    if strategy == "cross-scope-promotion":
        provenance["source_concepts"] = provenance["source_traces"]
    return store.create_node(
        level="concept", content=c._synthesize_digest(members, best=members[0]),
        context={"scope": scope}, provenance=provenance,
    )


def procedure(store, *, declared=True, scope=SCOPE, group="release", texts=TEXTS):
    traces = [store.append_trace(text, {
        "scope": scope, "procedure_id": "workshop_release", "task_pattern": group,
        **({"step_order": index + 1} if declared else {}),
    }) for index, text in enumerate(texts)]
    (node, _), = c._materialize_procedural_schemas(store, traces, min_cluster_size=3)
    return traces, node


@pytest.fixture(params=["decay", "consolidation", "service"])
def forget(request, store):
    return {
        "decay": lambda node_id: decay.memory_forget(store, node_id, "withdrawn"),
        "consolidation": lambda node_id: c.memory_forget(store, node_id, "withdrawn"),
        "service": lambda node_id: c.ConsolidationService(store).memory_forget(node_id, "withdrawn"),
    }[request.param]


@pytest.mark.parametrize("kind", ["schema", "evidence", "cluster"])
def test_supported_entries_refresh_partial_and_last_support(store, forget, kind):
    if kind == "cluster":
        traces = sources(store)
        owner = derived(store, "release", traces)
    else:
        traces, owner = procedure(store, declared=kind == "schema")
    _, other = procedure(store, scope="project:other")
    _, independent = procedure(store, group="independent", texts=[text + " Independently." for text in TEXTS])
    controls = {node.id: asdict(node) for node in (other, independent)}
    for index, trace in enumerate(traces):
        previous = store.get_node(owner.id)
        forgotten = forget(trace.id)
        assert forgotten.content == trace.content
        assert forgotten.decayed and forgotten.decay_reason == "withdrawn"
        current = store.get_node(owner.id)
        if index < len(traces) - 1:
            assert not current.decayed
            assert set(current.source_traces) == {t.id for t in traces[index + 1:]}
            assert all(text not in current.content for text in TEXTS[:index + 1])
            assert all(text in current.content for text in TEXTS[index + 1:])
        else:
            assert current.decayed and current.decay_reason == "derivation: no current sources"
            assert current.content == previous.content
            assert current.source_traces == previous.source_traces
    assert {node_id: asdict(store.get_node(node_id)) for node_id in controls} == controls
    with MemoryStore(store.db_path) as reopened:
        assert reopened.get_node(owner.id) == current
        assert all(reopened.get_node(t.id).decayed for t in traces)


def test_native_forget_recall_lookup_and_reopen(tmp_path):
    path = tmp_path / "native.sqlite3"
    mcp = server.create_mcp_server(path)
    store = mcp.memory_store
    traces, schema = procedure(store)
    descendant = derived(store, "summary", [schema])

    async def scenario():
        async with Client(mcp) as client:
            async def call(tool, **arguments):
                result = await client.call_tool(tool, arguments)
                return result.data

            for index, trace in enumerate(traces):
                forgotten = await call("memory_forget", id=trace.id, reason="withdrawn")
                assert forgotten["node"]["decayed"]
                recalled = await call("memory_recall", query="workshop release", scope=SCOPE,
                                      max_results=10)
                assert trace.id not in {r["id"] for r in recalled["results"]}
                if index < 2:
                    current = next(r for r in recalled["results"] if r["id"] == schema.id)
                    assert current["delivery"] == "full" and not current["decayed"]
                    assert all(text not in current["content"] for text in TEXTS[:index + 1])
                    assert all(text in current["content"] for text in TEXTS[index + 1:])
                else:
                    assert not {schema.id, descendant.id} & {r["id"] for r in recalled["results"]}
                history = await call("memory_lookup", node_ids=[trace.id, schema.id, descendant.id])
                rows = {r["id"]: r for r in history["results"]}
                assert rows[trace.id]["content"] == trace.content and rows[trace.id]["decayed"]
                for node in (schema, descendant):
                    assert rows[node.id]["decayed"] == (index == 2)
                    if index < 2:
                        assert all(text not in rows[node.id]["content"] for text in TEXTS[:index + 1])

    try:
        asyncio.run(scenario())
    finally:
        mcp.auto_consolidation.close()
        store.close()
    with MemoryStore(path) as reopened:
        assert all(reopened.get_node(node.id).decayed for node in (*traces, schema, descendant))
        assert reopened.get_node(traces[0].id).content == TEXTS[0]
        assert TEXTS[0] not in reopened.get_node(schema.id).content


@pytest.mark.parametrize("prepared", [False, True])
def test_diamond_and_cross_scope_promotion_use_current_inputs(store, prepared):
    traces = sources(store)
    left = derived(store, "left", traces[:2])
    right = derived(store, "right", [traces[0], traces[2]])
    joined = derived(store, "joined", [left, right])
    independent = derived(store, "other", traces[1:], scope="project:other")
    promoted = derived(store, "promotion", [joined, independent],
                       scope="global", strategy="cross-scope-promotion")
    preparation = c.prepare_forget(store, traces[0].id) if prepared else None
    c.memory_forget(store, traces[0].id, preparation=preparation)
    for node in (left, right, joined, promoted):
        current = store.get_node(node.id)
        assert not current.decayed and TEXTS[0] not in current.content
    assert TEXTS[1] in store.get_node(left.id).content
    assert TEXTS[2] in store.get_node(right.id).content
    assert all(text in store.get_node(joined.id).content for text in TEXTS[1:])
    assert store.get_node(independent.id) == independent
    assert store.get_node(promoted.id).scope == "global"


def test_unknown_writer_and_cycle_become_historical_but_citations_stay_independent(store):
    traces = sources(store)
    opaque = store.create_node(level="schema", content="Opaque release requirement.",
                               context={"scope": SCOPE})
    for trace in traces[:2]:
        store.create_connection(opaque.id, trace.id, "related", metadata={"kind": "derived_from"})
    descendant = derived(store, "opaque descendant", [opaque])
    cyclic = derived(store, "cycle", [traces[0]])
    store.update_node(cyclic.id, provenance={
        **cyclic.provenance, "source_traces": [traces[0].id, cyclic.id],
    })
    cited = store.create_node(level="concept", content="Independent release observation.",
                              provenance={"source_traces": [traces[0].id]})
    c.memory_forget(store, traces[0].id)
    for node in (opaque, descendant, cyclic):
        current = store.get_node(node.id)
        assert current.decayed and current.content == node.content
    assert store.get_node(opaque.id).decay_reason == "derivation: changed source requires rederivation"
    assert store.get_node(cyclic.id).decay_reason == "derivation: cyclic sources"
    assert store.get_node(cited.id) == cited


@pytest.mark.parametrize("relation", ["correction", "exact-copy"])
@pytest.mark.parametrize("target", ["original", "replacement"])
def test_replacement_authority_and_predecessor_consumers(store, relation, target):
    traces = sources(store)
    if relation == "correction":
        replacement = c.memory_teach(store, traces[0].id, "Workshop release uses a manifest.").corrective_trace
    else:
        replacement = store.append_trace(traces[0].content, {"scope": SCOPE})
    owner = derived(store, "earlier identity", traces[:2])
    node_id = traces[0].id if target == "original" else replacement.id
    c.memory_forget(store, node_id, preparation=c.prepare_forget(store, node_id))
    current = store.get_node(owner.id)
    expected = {traces[1].id} | ({replacement.id} if target == "original" else set())
    assert set(current.source_traces) == expected
    assert not current.decayed and TEXTS[1] in current.content
    assert (replacement.content in current.content) == (target == "original")
    assert store.get_node(traces[0].id).content == traces[0].content


@pytest.mark.parametrize("fault", ["source", "downstream", "edge"])
def test_required_failures_roll_back_entire_forget(store, forget, fault):
    traces = sources(store)
    owner = derived(store, "release", traces)
    descendant = derived(store, "descendant", [owner])
    trigger = {
        "source": f"BEFORE UPDATE OF decayed ON nodes WHEN NEW.id='{traces[0].id}'",
        "downstream": f"BEFORE UPDATE OF content ON nodes WHEN NEW.id='{descendant.id}'",
        "edge": f"BEFORE INSERT ON connections WHEN NEW.source_id='{owner.id}'",
    }[fault]
    store.connection.execute(f"""CREATE TRIGGER reject_forget {trigger}
        BEGIN SELECT RAISE(ABORT, 'required forget failure'); END""")
    before = list(store.connection.iterdump())
    with pytest.raises(sqlite3.IntegrityError, match="required forget failure"):
        forget(traces[0].id)
    assert not store.connection.in_transaction
    assert list(store.connection.iterdump()) == before
    with MemoryStore(store.db_path) as reopened:
        assert reopened.get_node(traces[0].id) == traces[0]
        assert reopened.get_node(owner.id) == owner
        assert reopened.get_node(descendant.id) == descendant
    store.connection.execute("DROP TRIGGER reject_forget")
    forget(traces[0].id)
    assert TEXTS[0] not in store.get_node(descendant.id).content
