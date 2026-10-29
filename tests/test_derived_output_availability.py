"""Derived content preparation leaves native reads available until atomic apply."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
from math import sqrt
from threading import Event
import asyncio

from fastmcp import Client
import pytest

from living_memory import consolidation as c, embeddings
from living_memory.chunking import chunk_text
from living_memory.embeddings import LocalEmbeddingModel
from test_consolidation_availability import runtime


SCOPE = "project:harbor"
REPRESENTATIVE = [1.0, 0.0, 0.0]
TEXTS = [
    "Harbor release requires the retired paper permit.",
    "Harbor release checks the signed package checksum.",
    "Harbor release records the new package version.",
]
CORRECTION = "Harbor release requires the electronic departure manifest."


def content_vector(text):
    digest = sha256(text.encode()).digest()
    values = [0.0, float(digest[0] + 1), float(digest[1] + 1)]
    norm = sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def native(mcp, name, **arguments):
    async def request():
        async with Client(mcp) as client:
            result = await client.call_tool(name, arguments)
            assert not result.is_error
            return result.data
    return asyncio.run(request())


def synthetic_model(vector, *, entered=None, release=None):
    class Backend:
        def encode(self, texts, **kwargs):
            if entered is not None:
                entered.set()
                assert release.wait(30), "test did not release encoder"
            return [vector for _ in texts] if isinstance(texts, list) else vector

    model = LocalEmbeddingModel(dimensions=3, backend="auto", model_name="harbor-output-fixture")
    model._model = Backend()
    model._load_attempted = True
    return model


class OutputGate:
    def __init__(self, store, lock, *, exclude=()):
        self.store, self.lock = store, lock
        self.entered, self.release = Event(), Event()
        self.calls = []
        self.exclude = list(exclude)

    def __call__(self, texts):
        assert not self.lock.lock._is_owned()
        assert not self.store.connection.in_transaction
        self.calls.append(list(texts))
        if list(texts) != self.exclude and not self.entered.is_set():
            self.entered.set()
            assert self.release.wait(30), "test did not release output encoder"
        return [content_vector(text) for text in texts]


@contextmanager
def guarded_sql(store, lock):
    unguarded = []
    store.connection.set_trace_callback(
        lambda sql: unguarded.append(sql) if not lock.lock._is_owned() else None
    )
    try:
        yield
    finally:
        store.connection.set_trace_callback(None)
    assert unguarded == []


def finish_while_lookup_available(mcp, lock, gate, operation, nodes):
    store = mcp.memory_store
    with guarded_sql(store, lock), ThreadPoolExecutor(max_workers=2) as workers:
        work = workers.submit(operation)
        try:
            assert gate.entered.wait(15), "derived output did not reach encoder"
            read = workers.submit(native, mcp, "memory_lookup", node_ids=[node.id for node in nodes])
            response = read.result(timeout=10)
            assert {row["id"]: row["content"] for row in response["results"]} == {
                node.id: node.content for node in nodes
            }
            assert not gate.release.is_set()
            assert not work.done()
        finally:
            gate.release.set()
        return work.result(timeout=20)


def assert_content_chunks(store, node):
    windows = chunk_text(node.content)
    chunks = store.list_node_chunks(node.id)
    assert len(chunks) == len(windows)
    assert store.node_chunks_are_current(node.id)
    assert node.embedding == REPRESENTATIVE
    for chunk, window in zip(chunks, windows, strict=True):
        assert chunk.embedding == pytest.approx(content_vector(window.text))
        assert chunk.embedding != node.embedding
        assert node.content[chunk.char_start:chunk.char_end] == window.text


@pytest.mark.parametrize("route", ["manual", "scheduled"])
@pytest.mark.parametrize("phase", ["create", "update"])
@pytest.mark.parametrize("runtime", [None], indirect=True)
def test_native_cluster_output_keeps_lookup_available(runtime, monkeypatch, route, phase):
    mcp, lock, _ = runtime
    store = mcp.memory_store
    encoder = synthetic_model(REPRESENTATIVE)
    monkeypatch.setattr(c, "LocalEmbeddingModel", lambda **kwargs: encoder)
    store.set_chunk_embedder(lambda texts: [content_vector(text) for text in texts])
    traces = [store.append_trace(text, {"scope": SCOPE, "agent": f"inspector-{i}"})
              for i, text in enumerate(TEXTS)]
    if phase == "update":
        native(mcp, "memory_consolidate", scope=SCOPE)
        owner, = store.list_nodes(level="concept", scope=SCOPE)
        owner = store.update_node(owner.id, content="Harbor release digest awaiting current evidence.")
    else:
        owner = None
    mcp.auto_consolidation._retry_delays = ()
    gate = OutputGate(store, lock)
    store.set_chunk_embedder(gate)

    def consolidate():
        if route == "manual":
            return native(mcp, "memory_consolidate", scope=SCOPE)
        mcp.auto_consolidation.request(SCOPE)
        assert mcp.auto_consolidation.wait_idle(40, include_retries=True)
        assert mcp.auto_consolidation.last_result(SCOPE)["status"] == "completed"

    finish_while_lookup_available(mcp, lock, gate, consolidate, [owner or traces[0]])
    current, = store.list_nodes(level="concept", scope=SCOPE)
    assert set(current.source_traces) == {trace.id for trace in traces}
    if owner is not None:
        assert current.id == owner.id
    assert gate.calls == [[window.text for window in chunk_text(current.content)]]
    assert_content_chunks(store, current)
    monkeypatch.setattr(LocalEmbeddingModel, "embed", lambda self, text: content_vector(text))
    recalled = native(mcp, "memory_recall", query=current.content, scope=SCOPE, max_results=10)
    assert current.id in {row["id"] for row in recalled["results"]}


@pytest.mark.parametrize("phase", ["create", "update", "final_refresh"])
@pytest.mark.parametrize("runtime", [None], indirect=True)
def test_promotion_output_keeps_native_lookup_available(runtime, phase):
    mcp, lock, _ = runtime
    store = mcp.memory_store
    mcp.auto_consolidation.close()
    store.set_chunk_embedder(lambda texts: [content_vector(text) for text in texts])
    sources = [store.create_node(
        level="concept", content=text, context={"scope": f"project:harbor-{i}"},
        embedding=REPRESENTATIVE,
    ) for i, text in enumerate(TEXTS)]
    owner = None
    if phase != "create":
        owner, = c._cross_scope_promotion(store, sources[0], step=lambda: lock)
        sources[1] = store.update_node(
            sources[1].id, content="Harbor release checks the independently signed package checksum.",
            embedding=REPRESENTATIVE,
        )
    gate = OutputGate(store, lock)
    store.set_chunk_embedder(gate)

    def promote():
        if phase == "final_refresh":
            return c._refresh_promoted_global_concepts(store, step=lambda: lock)
        return c._cross_scope_promotion(store, sources[0], step=lambda: lock)

    finish_while_lookup_available(mcp, lock, gate, promote, [owner or sources[0]])
    current, = store.list_nodes(level="concept", scope="global")
    if owner is not None:
        assert current.id == owner.id and current.content != owner.content
    assert set(current.materializer_sources) == {source.id for source in sources}
    assert gate.calls == [[window.text for window in chunk_text(current.content)]]
    assert_content_chunks(store, current)


def source_first_diamond(store):
    store.set_chunk_embedder(lambda texts: [content_vector(text) for text in texts])
    traces = [store.create_node(
        level="trace", content=text, context={"scope": SCOPE, "agent": f"inspector-{i}"},
        embedding=REPRESENTATIVE,
        stats={"usefulness_score": 0.0 if i == 0 else 100.0, "confidence": 0.5},
    ) for i, text in enumerate(TEXTS)]

    def derived(name, members):
        return store.create_node(
            level="concept", content=c._synthesize_digest(members, best=members[0]),
            context={"scope": SCOPE}, representative_embedding=REPRESENTATIVE,
            provenance={"strategy": "token-jaccard", "cluster_key": name,
                        "source_traces": [member.id for member in members]},
        )

    left = derived("left", traces[:2])
    right = derived("right", [traces[0], traces[2]])
    joined = derived("joined", [left, right])
    return traces, [left, right, joined]


@pytest.mark.parametrize("operation", ["memory_teach", "memory_forget"])
@pytest.mark.parametrize("runtime", [None], indirect=True)
def test_native_source_change_prepares_source_first_outputs_outside_sql(runtime, operation):
    mcp, lock, _ = runtime
    store = mcp.memory_store
    mcp.auto_consolidation.close()
    traces, owners = source_first_diamond(store)
    gate = OutputGate(store, lock, exclude=[CORRECTION])
    store.set_chunk_embedder(gate)
    arguments = ({"trace_id": traces[0].id, "correction": CORRECTION}
                 if operation == "memory_teach" else {"id": traces[0].id})
    result = finish_while_lookup_available(
        mcp, lock, gate, lambda: native(mcp, operation, **arguments), [traces[0], *owners],
    )
    old = store.get_node(traces[0].id)
    assert old.decayed if operation == "memory_forget" else old.corrections[-1]["new"] == CORRECTION
    outputs = gate.calls[1:] if operation == "memory_teach" else gate.calls
    for owner in owners:
        current = store.get_node(owner.id)
        assert not current.decayed and TEXTS[0] not in current.content
        if operation == "memory_teach":
            assert CORRECTION in current.content
        assert [window.text for window in chunk_text(current.content)] in outputs
        assert_content_chunks(store, current)
    joined = store.get_node(owners[-1].id)
    assert set(joined.source_traces) == {node.id for node in owners[:2]}
    if operation == "memory_teach":
        assert result["corrective_trace"]["id"] in store.get_node(owners[0].id).source_traces


@pytest.mark.parametrize("runtime", [None], indirect=True)
def test_output_wait_for_shared_encoder_leaves_native_lookup_available(runtime, monkeypatch):
    mcp, lock, _ = runtime
    store = mcp.memory_store
    mcp.auto_consolidation.close()
    traces = [store.create_node(
        level="trace", content=text, context={"scope": SCOPE}, embedding=REPRESENTATIVE,
    ) for text in TEXTS]
    entered, waiting, release = Event(), Event(), Event()
    shared_lock = embeddings._ENCODER_LOCK

    class ObservedEncoderLock:
        def __enter__(self):
            if not shared_lock.acquire(blocking=False):
                assert not lock.lock._is_owned()
                assert not store.connection.in_transaction
                waiting.set()
                shared_lock.acquire()

        def __exit__(self, *args):
            shared_lock.release()

    output = synthetic_model([0.0, 1.0, 0.0])
    other = synthetic_model([0.0, 0.0, 1.0], entered=entered, release=release)
    store.set_chunk_embedder(output.embed_batch)
    monkeypatch.setattr(embeddings, "_ENCODER_LOCK", ObservedEncoderLock())
    with guarded_sql(store, lock), ThreadPoolExecutor(max_workers=3) as workers:
        contender = workers.submit(other.embed, "Independent encoder work")
        work = None
        try:
            assert entered.wait(15)
            work = workers.submit(native, mcp, "memory_consolidate", scope=SCOPE)
            assert waiting.wait(15), "derived output did not wait for shared encoder"
            read = workers.submit(native, mcp, "memory_lookup", node_id=traces[0].id)
            assert read.result(timeout=10)["results"][0]["content"] == traces[0].content
            assert not work.done() and not contender.done() and not release.is_set()
        finally:
            release.set()
        contender.result(timeout=15)
        assert work is not None
        work.result(timeout=20)
    current, = store.list_nodes(level="concept", scope=SCOPE)
    assert current.embedding == REPRESENTATIVE
    assert store.node_chunks_are_current(current.id)
    assert all(chunk.embedding == [0.0, 1.0, 0.0] for chunk in store.list_node_chunks(current.id))
