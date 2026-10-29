"""Explicit connections commit current derivatives with their relation."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from living_memory import consolidation as c, memory_connect
from living_memory.retrieval import MemoryRecallService
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore
from test_teach_derivation import CORRECTION, SCOPE, STEPS, procedure
from test_transport_identity import FakeMCP, _structured


@pytest.fixture(autouse=True)
def deterministic(monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "off")
    monkeypatch.setenv("LM_RECALL_NEAR_DUP_COSINE", "0")
    monkeypatch.setenv("LM_AUTO_CONSOLIDATE_POLICY", "off")
    monkeypatch.setenv("LM_DELIVERY_SESSION_DEDUP", "0")


@pytest.mark.parametrize("declared", [True, False])
def test_native_connect_lookup_recall_preserve_identity_and_independent_edges(tmp_path, declared):
    from fastmcp import Client

    mcp = create_mcp_server(tmp_path / "memory.sqlite3")
    store = mcp.memory_store
    traces, schema = procedure(store)
    _, independent = procedure(store, group="independent", texts=[text + " Independently." for text in STEPS])
    _, elsewhere = procedure(store, scope="project:other")
    opaque = store.create_node(level="schema", content="Opaque paper permit requirement.", context={"scope": SCOPE})
    store.create_connection(opaque.id, traces[0].id, "related", metadata={"kind": "derived_from"})
    controls = [
        store.create_connection(schema.id, traces[0].id, "requires", metadata={"basis": "explicit"}),
        store.create_connection(traces[0].id, independent.id, "related", weight=.4),
        store.create_connection(elsewhere.id, independent.id, "contradicts", weight=.3),
    ]

    async def scenario():
        async with Client(mcp) as client:
            async def call(name, **kwargs):
                return _structured(await client.call_tool(name, kwargs))

            first = await call("memory_recall", query="workshop release", scope=SCOPE, depth=0, max_results=10)
            assert any(item["id"] == schema.id and STEPS[0] in item["content"] for item in first["results"])
            replacement = await call("memory_remember", content=CORRECTION, context={
                "scope": "project:replacement", "agent": "replacement-writer",
                **({"step_order": 1} if declared else {}),
            })
            new_id = replacement["node"]["id"]
            old = store.get_node(traces[0].id)
            new = store.get_node(new_id)
            node_ids = {row[0] for row in store.connection.execute("SELECT id FROM nodes")}
            untouched = [store.get_node(node.id) for node in (independent, elsewhere)]
            result = await call("memory_connect", id_a=new_id, id_b=old.id, relation_type="supersedes", weight=.7)
            assert set(result) == {"connection"}
            edge = result["connection"]
            assert (edge["source_id"], edge["target_id"], edge["type"], edge["weight"], edge["metadata"]) == (
                new_id, old.id, "supersedes", .7, {},
            )
            assert {row[0] for row in store.connection.execute("SELECT id FROM nodes")} == node_ids
            assert store.get_node(old.id) == old and store.get_node(new_id) == new
            assert [store.get_node(node.id) for node in untouched] == untouched
            assert [store.get_connection(edge.id) for edge in controls] == controls
            current = store.get_node(schema.id)
            assert current.scope == schema.scope and current.created_at == schema.created_at
            assert current.level == ("schema" if declared else "concept")
            assert not current.decayed and not current.corrections
            assert set(current.source_traces) == {new_id, traces[1].id, traces[2].id}
            assert all(text in current.content for text in (CORRECTION, *STEPS[1:]))
            assert STEPS[0] not in current.content
            for view in ("knowledge", "full"):
                looked = await call("memory_lookup", node_ids=[old.id, schema.id, opaque.id], view=view)
                by_id = {item.get("node", item)["id"]: item for item in looked["results"]}
                assert by_id[old.id]["superseded"] and by_id[old.id]["superseded_by"] == [new_id]
                assert by_id[old.id].get("node", by_id[old.id])["content"] == STEPS[0]
                assert by_id[schema.id].get("node", by_id[schema.id])["content"] == current.content
                assert by_id[opaque.id].get("node", by_id[opaque.id])["decayed"]
            recalled = await call("memory_recall", query="workshop release", scope=SCOPE, depth=0, max_results=10)
            delivered = next(item for item in recalled["results"] if item["id"] == schema.id)
            assert delivered["delivery"] == "full" and delivered["content"] == current.content
            assert all(item["id"] != opaque.id for item in recalled["results"])

    try:
        asyncio.run(scenario())
    finally:
        mcp.auto_consolidation.close()
        store.close()


@pytest.mark.parametrize("entry", ["public", "service"])
def test_python_connect_refreshes_existing_replacement_row(tmp_path, entry):
    from test_derivative_preparation import cluster, sources

    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        traces = sources(store)
        owner = cluster(store, traces)
        replacement = store.append_trace("Harbor crane requires the electronic manifest.", {"scope": traces[0].scope})
        inferred = store.create_connection(replacement.id, traces[0].id, "supersedes", weight=.9,
                                           metadata={"rule": "R1c", "kind": "content_correction"})
        connect = (lambda *args, **kwargs: memory_connect(store, *args, **kwargs)) if entry == "public" else MemoryRecallService(store).memory_connect
        edge = connect(replacement.id, traces[0].id, "supersedes", weight=.6)
        assert edge.id == inferred.id and edge.created_at == inferred.created_at
        assert edge.weight == .6 and edge.metadata == {}
        current = store.get_node(owner.id)
        assert replacement.content in current.content and traces[0].content not in current.content
        assert replacement.id in current.source_traces


@pytest.mark.parametrize("case", ["related", "caused", "requires", "contradicts", "inferred", "duplicate", "self", "decayed"])
def test_inapplicable_connections_do_not_refresh(tmp_path, monkeypatch, case):
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        traces, schema = procedure(store)
        new = store.append_trace(CORRECTION, {"scope": SCOPE})
        metadata = {"inferred": {"rule": "R1c"}, "duplicate": {"kind": "duplicate_content"}}.get(case)
        if case == "self":
            new = traces[0]
        if case == "decayed":
            store.soft_delete_node(new.id)
        before = store.get_node(schema.id)

        def unexpected(*args, **kwargs):
            raise AssertionError("inapplicable relation must not refresh derivatives")

        monkeypatch.setattr(c, "_derivation_snapshots", unexpected)
        monkeypatch.setattr(c, "_refresh_derived_from", unexpected)
        relation = case if case in ("related", "caused", "requires", "contradicts") else "supersedes"
        edge = memory_connect(store, new.id, traces[0].id, relation, metadata=metadata)
        assert edge.metadata == (metadata or {})
        assert store.get_node(schema.id) == before


@pytest.fixture
def native(tmp_path):
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    yield mcp
    mcp.auto_consolidation.close()
    mcp.memory_store.close()


@pytest.mark.parametrize("failure", ["snapshot", "refresh", "chunks"])
def test_failed_connect_rolls_back_relation_anchors_and_derivatives(native, monkeypatch, failure):
    from test_teach_preparation_concurrency import seed
    from living_memory.query_anchors import upsert_anchor

    store = native.memory_store
    traces, _ = seed(native, diamond=True)
    for trace in traces:
        store.update_node(trace.id, embedding=[1., 0.])
    new = store.append_trace("Harbor crane requires the electronic manifest.", {"scope": traces[0].scope})
    upsert_anchor(store, "Harbor crane", traces[0].scope, [1., 0.], [traces[0].id])
    if failure == "snapshot":
        target = "_prepare_derivation_digests"
    elif failure == "refresh":
        target = "_refresh_derived_node"
    else:
        target = None

    def fail(*args, **kwargs):
        if failure == "refresh":
            original(*args, **kwargs)
        else:
            assert not store.connection.in_transaction
        raise RuntimeError("required connect failure")

    if target:
        original = getattr(c, target)
        monkeypatch.setattr(c, target, fail)
    else:
        store.set_chunk_embedder(fail)
    before = list(store.connection.iterdump())
    with pytest.raises(RuntimeError, match="required connect failure"):
        native.tools["memory_connect"](new.id, traces[0].id, "supersedes")
    assert list(store.connection.iterdump()) == before
    assert not store.connection.in_transaction


def test_connect_preparation_allows_reads_and_revalidates_changed_replacement(native, monkeypatch):
    from test_teach_preparation_concurrency import seed, lookup, recall

    store = native.memory_store
    traces, owners = seed(native, diamond=True)
    new = store.create_node(level="concept", content="Harbor crane requires the electronic manifest.", context={"scope": traces[0].scope})
    entered, release = Event(), Event()
    prepare = c._prepare_digest

    def pause(*args, **kwargs):
        assert not store.connection.in_transaction
        result = prepare(*args, **kwargs)
        if not entered.is_set():
            entered.set()
            assert release.wait(15)
        return result

    monkeypatch.setattr(c, "_prepare_digest", pause)
    with ThreadPoolExecutor(max_workers=2) as workers:
        pending = workers.submit(native.tools["memory_connect"], new.id, traces[0].id, "supersedes")
        try:
            assert entered.wait(10)
            before = workers.submit(lookup, native, owners).result(timeout=10)
            assert [item["content"] for item in before["results"]] == [node.content for node in owners]
            assert workers.submit(recall, native).result(timeout=10)["results"]
            assert not pending.done()
            with MemoryStore(store.db_path) as writer:
                writer.update_node(new.id, content="Harbor crane requires the certified digital manifest.")
        finally:
            release.set()
        pending.result(timeout=10)
    for owner in owners:
        current = store.get_node(owner.id)
        assert "certified digital manifest" in current.content
        assert new.content not in current.content and traces[0].content not in current.content
