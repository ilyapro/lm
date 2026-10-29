"""Prepared derived chunks belong to the current atomic source result."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event, RLock

import pytest

from living_memory import consolidation as c
from living_memory.chunking import chunk_text
from living_memory.embeddings import LocalEmbeddingModel
from living_memory.storage import MemoryStore

SCOPE = "project:harbor"
CORRECTION = "Harbor departure requires an electronic manifest."


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    encoder = LocalEmbeddingModel(backend="hash", dimensions=3)
    with MemoryStore(tmp_path / "memory.sqlite3", chunk_embedder=encoder.embed_batch) as opened:
        yield opened


def sources(store, *, level="trace"):
    return [store.create_node(
        level=level, content=text, context={"scope": SCOPE, "agent": f"inspector-{i}"},
        embedding=[1., 0., 0.], stats={"usefulness_score": i * 100.},
    ) for i, text in enumerate((
        "Harbor departure requires the paper permit.",
        "Harbor departure checks the brake assembly.",
        "Harbor departure measures cable tension.",
    ))]


def derived(store, name, members):
    return store.create_node(
        level="concept", content=f"Harbor {name} awaits its current evidence.",
        context={"scope": SCOPE}, representative_embedding=[1., 0., 0.],
        provenance={"strategy": "embedding-cosine", "cluster_key": name,
                    "source_traces": [member.id for member in members]},
    )


def materialize(store, members, guard):
    group = c._TraceCluster(SCOPE, members, Counter({"harbor": len(members)}))
    return c._materialize_cluster(store, group, min_cluster_size=1, step=lambda: guard)


def assert_chunks(store, node):
    encoder = LocalEmbeddingModel(backend="hash", dimensions=3)
    chunks = store.list_node_chunks(node.id)
    windows = chunk_text(node.content)
    assert len(chunks) == len(windows)
    assert store.node_chunks_are_current(node.id)
    for chunk, window in zip(chunks, windows, strict=True):
        assert chunk.embedding == pytest.approx(encoder.embed(window.text), abs=1e-7)
        assert chunk.embedding != node.embedding


@contextmanager
def preparing(store, guard):
    entered, release = Event(), Event()
    encoder = LocalEmbeddingModel(backend="hash", dimensions=3)
    encoded = []

    def encode(texts):
        assert not guard._is_owned()
        assert not store.connection.in_transaction
        encoded.append(list(texts))
        if not entered.is_set():
            entered.set()
            assert release.wait(15), "test did not release output preparation"
        return encoder.embed_batch(texts)

    store.set_chunk_embedder(encode)
    with ThreadPoolExecutor(max_workers=2) as workers:
        try:
            yield workers, entered, release, encoded
        finally:
            release.set()


@pytest.mark.parametrize("change", ["source_content", "source_decay", "source_correction", "union"])
def test_changed_sources_and_owner_are_prepared_again(store, change):
    members = sources(store, level="concept" if change == "source_content" else "trace")
    owner = derived(store, "inspection", members)
    guard = RLock()
    with preparing(store, guard) as (workers, entered, release, encoded):
        work = workers.submit(materialize, store, members, guard)
        assert entered.wait(10)
        with MemoryStore(store.db_path, chunk_embedder=LocalEmbeddingModel(backend="hash", dimensions=3).embed_batch) as writer:
            if change == "source_content":
                writer.update_node(members[0].id, content=CORRECTION)
            elif change == "source_decay":
                writer.soft_delete_node(members[0].id, "retired source")
            elif change == "source_correction":
                c.memory_teach(writer, members[0].id, CORRECTION)
            elif change == "union":
                added = writer.append_trace("Harbor departure verifies the signed cargo ledger.", {"scope": SCOPE})
                writer.update_node(owner.id, provenance={
                    **owner.provenance, "source_traces": [*owner.source_traces, added.id],
                })
        release.set()
        (current, created), = work.result(timeout=15)
    assert not created and current.id == owner.id
    # A concurrent teach has already committed the new digest and its chunks;
    # the resumed materializer must keep that current index without encoding.
    assert len(encoded) == 1 if change == "source_correction" else len(encoded) >= 2
    assert_chunks(store, current)
    if change in ("source_content", "source_correction"):
        assert CORRECTION in current.content and members[0].content not in current.content
    elif change == "source_decay":
        assert members[0].id not in current.source_traces and members[0].content not in current.content
    elif change == "union":
        assert added.id in current.source_traces and added.content in current.content


@pytest.mark.parametrize("change", ["decay", "correction", "materializer", "scope"])
def test_invalidated_owner_is_not_overwritten_after_preparation(store, change):
    members = sources(store)
    owner = derived(store, "inspection", members)
    guard = RLock()
    with preparing(store, guard) as (workers, entered, release, _):
        work = workers.submit(materialize, store, members, guard)
        assert entered.wait(10)
        with MemoryStore(store.db_path) as writer:
            if change == "decay":
                writer.soft_delete_node(owner.id, "retired owner")
            elif change == "correction":
                writer.add_correction(owner.id, old=owner.content, new=CORRECTION, by="inspector")
            elif change == "materializer":
                writer.update_node(owner.id, provenance={**owner.provenance, "strategy": "procedural"})
            else:
                writer.update_node(owner.id, context={**owner.context, "scope": "project:new-harbor"})
            preserved = writer.get_node(owner.id)
        release.set()
        assert work.result(timeout=15) == []
    assert store.get_node(owner.id) == preserved


@pytest.mark.parametrize("change", ["group", "competing_owner"])
def test_new_source_admission_is_rechecked_after_preparation(store, change):
    members = sources(store)
    owner = derived(store, "inspection", members[:2])
    guard = RLock()
    group = c._TraceCluster(SCOPE, members, Counter())
    with preparing(store, guard) as (workers, entered, release, _):
        work = workers.submit(c._materialize_cluster, store, group, min_cluster_size=3, step=lambda: guard)
        assert entered.wait(10)
        with MemoryStore(store.db_path) as writer:
            if change == "group":
                writer.update_node(owner.id, context={**owner.context, "task_pattern": "independent"})
            else:
                competing = derived(writer, "other inspection", members[2:])
        release.set()
        (current, created), = work.result(timeout=15)
    assert not created and current.id == owner.id
    assert set(current.source_traces) == {member.id for member in members[:2]}
    assert members[2].content not in current.content
    assert_chunks(store, current)
    if change == "competing_owner":
        assert store.get_node(competing.id).source_traces == [members[2].id]


def test_new_owner_during_create_preparation_receives_the_current_union(store):
    members = sources(store)
    guard = RLock()
    with preparing(store, guard) as (workers, entered, release, _):
        work = workers.submit(materialize, store, members, guard)
        assert entered.wait(10)
        with MemoryStore(store.db_path) as writer:
            owner = derived(writer, "concurrent inspection", members[:2])
        release.set()
        (current, created), = work.result(timeout=15)
    assert not created and current.id == owner.id
    assert set(current.source_traces) == {member.id for member in members}
    assert [node.id for node in store.list_nodes(level="concept")] == [owner.id]
    assert_chunks(store, current)


@pytest.mark.parametrize("operation", ["teach", "forget"])
@pytest.mark.parametrize("failure", ["encode", "chunks", "edges"])
def test_source_first_failure_rolls_back_source_and_all_derivatives(store, monkeypatch, operation, failure):
    members = sources(store)
    left = derived(store, "left", members[:2])
    right = derived(store, "right", [members[0], members[2]])
    derived(store, "joined", [left, right])
    before = list(store.connection.iterdump())
    if failure == "encode":
        def fail(texts):
            assert not store.connection.in_transaction
            raise RuntimeError("required output failure")
        store.set_chunk_embedder(fail)
    elif failure == "chunks":
        original = store._replace_node_chunks
        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("required output failure")
        monkeypatch.setattr(store, "_replace_node_chunks", fail)
    else:
        original = c._update_edge_weights_from_co_access
        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("required output failure")
        monkeypatch.setattr(c, "_update_edge_weights_from_co_access", fail)
    with pytest.raises(RuntimeError, match="required output failure"):
        if operation == "teach":
            c.memory_teach(store, members[0].id, CORRECTION)
        else:
            c.memory_forget(store, members[0].id)
    assert list(store.connection.iterdump()) == before
    assert not store.connection.in_transaction


@pytest.mark.parametrize("operation", ["teach", "forget"])
def test_caller_owned_transaction_rejects_missing_preparation_without_encoding(store, operation):
    members = sources(store)
    derived(store, "inspection", members)
    before = list(store.connection.iterdump())
    def no_encoding(texts):
        pytest.fail("caller-owned SQL must not enter output encoder")
    store.set_chunk_embedder(no_encoding)
    with store.transaction():
        with pytest.raises(c._DerivedOutputRequired):
            if operation == "teach":
                c.memory_teach(store, members[0].id, CORRECTION)
            else:
                c.memory_forget(store, members[0].id)
        assert list(store.connection.iterdump()) == before


def test_promotion_noop_does_not_prepare_or_replace_missing_chunks(store):
    members = [store.create_node(
        level="concept", content=f"Harbor scope {i} reports a distinct inspection fact.",
        context={"scope": f"project:harbor-{i}"}, embedding=[1., 0., 0.],
    ) for i in range(3)]
    promoted, = c._cross_scope_promotion(store, members[0])
    store.replace_node_chunks(promoted.id, [])
    def no_encoding(texts):
        pytest.fail("unchanged promotion must retain its no-op behavior")
    store.set_chunk_embedder(no_encoding)
    assert c._cross_scope_promotion(store, members[0]) == []
    assert store.list_node_chunks(promoted.id) == []


@pytest.mark.parametrize("policy", ["audit", "credit"])
@pytest.mark.parametrize("failure", ["encode", "chunks", "closure"])
def test_native_teach_retries_keep_marks_and_derivatives_in_one_transaction(tmp_path, monkeypatch, policy, failure):
    from living_memory import server
    from test_lookup import FakeMCP

    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", policy)
    monkeypatch.setattr(server, "_transport_session_id", lambda: "output-atomicity")
    mcp = server.create_mcp_server(tmp_path / "feedback.sqlite3", mcp_factory=FakeMCP)
    mcp.auto_consolidation.close()
    store = mcp.memory_store
    try:
        members = sources(store)
        left = derived(store, "left", members[:2])
        right = derived(store, "right", [members[0], members[2]])
        derived(store, "joined", [left, right])
        context = {"scope": SCOPE, "agent": "inspector", "session_id": "output-atomicity"}
        recalled = mcp.tools["memory_recall"](
            "Harbor departure permit inspection", scope=SCOPE, ambient_context=context,
        )
        ids = [row["id"] for row in recalled["results"]]
        assert ids
        if failure == "closure":
            store.connection.execute("""CREATE TRIGGER reject_closure
                BEFORE UPDATE OF feedback_applied_at ON recall_events
                WHEN NEW.feedback_applied_at IS NOT NULL
                BEGIN SELECT RAISE(ABORT, 'required output failure'); END""")
        before = list(store.connection.iterdump())
        if failure == "encode":
            original = store._chunk_embeddings_for
            def fail(content, *args, **kwargs):
                assert not store.connection.in_transaction
                if content != CORRECTION:
                    raise RuntimeError("required output failure")
                return original(content, *args, **kwargs)
            monkeypatch.setattr(store, "_chunk_embeddings_for", fail)
        elif failure == "chunks":
            original = store._replace_node_chunks
            def fail(node_id, *args, **kwargs):
                result = original(node_id, *args, **kwargs)
                if node_id == right.id:
                    raise RuntimeError("required output failure")
                return result
            monkeypatch.setattr(store, "_replace_node_chunks", fail)
        with pytest.raises(Exception, match="required output failure"):
            mcp.tools["memory_teach"](
                members[0].id, CORRECTION, context=context, used=ids[:1], irrelevant=ids[1:],
            )
        assert list(store.connection.iterdump()) == before
        assert not store.connection.in_transaction
    finally:
        store.close()
