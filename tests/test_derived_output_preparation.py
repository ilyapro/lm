"""Prepared derived chunks belong to the current atomic source result."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
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


@pytest.mark.parametrize("phase", ["create", "refresh"])
def test_promotion_rechecks_original_authors_after_detached_preparation(store, phase):
    members = sources(store)
    projects = [derived(store, f"project {i}", [member]) for i, member in enumerate(members)]
    projects = [store.update_node(
        project.id, context={"scope": f"project:harbor-{i}", "agent": "memory_consolidate"},
    ) for i, project in enumerate(projects)]
    owner = None
    if phase == "refresh":
        owner, = c._cross_scope_promotion(store, projects[0])
        assert owner.unique_agents == 3
        store.update_node(projects[0].id, content=CORRECTION)
    guard = RLock()
    with preparing(store, guard) as (workers, entered, release, _):
        work = workers.submit(c._cross_scope_promotion, store, projects[0], step=lambda: guard)
        assert entered.wait(10)
        with MemoryStore(store.db_path) as writer:
            for member in members:
                writer.update_node(member.id, context={**member.context, "agent": "same-inspector"})
        release.set()
        current, = work.result(timeout=15)
    assert current.unique_agents == 1 and current.confidence == .5
    if owner is not None:
        assert current.id == owner.id


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


@pytest.mark.parametrize("change", [
    "scope", "level", "source_content", "deleted", "decayed", "reowned", "corrected", "superseded",
])
def test_final_promotion_refresh_rechecks_current_selection_and_sources(store, change):
    members = [store.create_node(
        level="concept", content=f"Harbor scope {i} reports a distinct inspection fact.",
        context={"scope": f"project:harbor-{i}"}, embedding=[1., 0., 0.],
    ) for i in range(3)]
    promoted, = c._cross_scope_promotion(store, members[0])
    store.update_node(members[0].id, content="Harbor departure checks an electronic manifest.")
    guard = RLock()
    with preparing(store, guard) as (workers, entered, release, _):
        work = workers.submit(c._refresh_promoted_global_concepts, store, step=lambda: guard)
        assert entered.wait(10)
        with MemoryStore(store.db_path) as writer:
            if change == "scope":
                writer.update_node(promoted.id, context={**promoted.context, "scope": SCOPE})
            elif change == "level":
                writer.update_node(promoted.id, level="schema")
            elif change == "deleted":
                with writer.transaction():
                    writer.connection.execute(
                        "DELETE FROM connections WHERE source_id=? OR target_id=?", (promoted.id, promoted.id),
                    )
                    writer.delete_node_chunks(promoted.id)
                    writer.connection.execute("DELETE FROM nodes WHERE id=?", (promoted.id,))
            elif change == "decayed":
                writer.soft_delete_node(promoted.id, "retired promotion")
            elif change == "reowned":
                writer.update_node(promoted.id, provenance={**promoted.provenance, "strategy": "procedural"})
            elif change == "corrected":
                writer.update_node(promoted.id, provenance={
                    **promoted.provenance, "corrections": [{"new": "replacement"}],
                })
            elif change == "superseded":
                writer.create_connection(members[0].id, promoted.id, "supersedes")
            else:
                writer.update_node(members[1].id, content="Harbor departure verifies the cargo signature.")
            preserved = writer.get_node(promoted.id)
        release.set()
        result = work.result(timeout=15)
    current = store.get_node(promoted.id)
    if change == "source_content":
        assert [node.id for node in result] == [promoted.id]
        assert "cargo signature" in current.content and "electronic manifest" in current.content
        assert members[1].content not in current.content
        assert_chunks(store, current)
    else:
        assert result == []
        assert current == preserved


def test_owner_losing_all_sources_during_preparation_is_invalidated(store):
    members = sources(store)
    owner = derived(store, "inspection", members)
    guard = RLock()
    with preparing(store, guard) as (workers, entered, release, _):
        work = workers.submit(materialize, store, members, guard)
        assert entered.wait(10)
        with MemoryStore(store.db_path) as writer:
            for member in members:
                writer.soft_delete_node(member.id, "retired source")
        release.set()
        assert work.result(timeout=15) == []
    current = store.get_node(owner.id)
    assert current.decayed and current.decay_reason == "derivation: no current sources"
    assert c._current_derivation(store, current).live == []


def independent_owners(store, *, level="trace"):
    source, *others = sources(store, level=level)
    others.extend(store.create_node(
        level=level, content=f"Harbor inspection {i} checks separate equipment.",
        context={"scope": SCOPE}, embedding=[1., 0., 0.], stats={"usefulness_score": 100.},
    ) for i in range(2))
    return source, [derived(store, f"owner {i}", [source, other]) for i, other in enumerate(others)]


def output_work(store, monkeypatch, guard):
    retries, encoded = [], []
    prepare = c._DerivedOutputRequired.prepare
    encoder = LocalEmbeddingModel(backend="hash", dimensions=3)

    def counted(needed, current_store, preparation):
        assert not guard._is_owned() and not store.connection.in_transaction
        retries.append([item.key for item in (needed, *needed.additional)])
        return prepare(needed, current_store, preparation)

    def encode(texts):
        assert not guard._is_owned() and not store.connection.in_transaction
        encoded.append(list(texts))
        return encoder.embed_batch(texts)

    monkeypatch.setattr(c._DerivedOutputRequired, "prepare", counted)
    store.set_chunk_embedder(encode)
    return retries, encoded


@pytest.mark.parametrize("operation", ["teach", "forget", "connect"])
def test_independent_outputs_prepare_before_the_first_writer(store, monkeypatch, operation):
    source, owners = independent_owners(store)
    replacement = store.append_trace(CORRECTION, {"scope": SCOPE}) if operation == "connect" else None
    guard = RLock()
    retries, encoded = output_work(store, monkeypatch, guard)
    if operation == "teach":
        preparation = c.prepare_teach(store, source.id, CORRECTION, guard=lambda: guard)
        assert len(encoded) == len(owners)
        with guard:
            replacement = c.memory_teach(store, source.id, CORRECTION, preparation=preparation).corrective_trace
    elif operation == "forget":
        preparation = c.prepare_forget(store, source.id, guard=lambda: guard)
        assert len(encoded) == len(owners)
        with guard:
            c.memory_forget(store, source.id, preparation=preparation)
    else:
        c.memory_connect(store, replacement.id, source.id, "supersedes", guard=lambda: guard)
    assert retries == []
    assert len(encoded) == len(owners)
    for owner in owners:
        current = store.get_node(owner.id)
        expected_ids = set(owner.source_traces) - {source.id}
        if replacement:
            expected_ids.add(replacement.id)
        assert set(current.source_traces) == expected_ids
        members = list(store.get_nodes(expected_ids).values())
        best = max(members, key=lambda member: (c._trace_quality(member), member.id))
        assert current.content == c._synthesize_digest(members, best=best)
        assert_chunks(store, current)


@pytest.mark.parametrize("change", ["missing", "stale_owner", "stale_source", "best"])
def test_missing_or_stale_independent_outputs_share_one_rollback(store, monkeypatch, change):
    source, owners = independent_owners(store, level="concept" if change == "stale_source" else "trace")
    preparation = c.prepare_forget(store, source.id)
    if change == "missing":
        preparation.digests = {key: replace(value, chunks=None) for key, value in preparation.digests.items()}
    else:
        for index, owner in enumerate(owners):
            if change == "stale_owner":
                store.update_node(owner.id, content=owner.content + " New owner observation.")
            else:
                other = next(item for item in owner.source_traces if item != source.id)
                if change == "stale_source":
                    store.update_node(other, content=f"Harbor inspection {index} uses a calibrated scanner.")
                else:
                    # A newly admitted stronger source invalidates both best and membership.
                    stronger = store.create_node(
                        level="trace", content=f"Harbor inspection {index} checks the strongest current evidence.",
                        context={"scope": SCOPE}, embedding=[0., 1., 0.], stats={"usefulness_score": 1000.},
                    )
                    store.update_node(owner.id, provenance={
                        **owner.provenance, "source_traces": [*owner.source_traces, stronger.id],
                    })
    before = list(store.connection.iterdump())
    guard = RLock()
    retries, encoded = output_work(store, monkeypatch, guard)
    prepare = c._DerivedOutputRequired.prepare

    def rolled_back(*args):
        assert list(store.connection.iterdump()) == before
        return prepare(*args)

    monkeypatch.setattr(c._DerivedOutputRequired, "prepare", rolled_back)
    c.memory_forget(store, source.id, preparation=preparation)
    assert len(retries) == 1 and set(retries[0]) == {owner.id for owner in owners}
    assert len(encoded) == len(owners)
    for owner in owners:
        current = store.get_node(owner.id)
        assert source.id not in current.source_traces
        if change == "stale_source":
            assert "calibrated scanner" in current.content
        if change == "best":
            assert current.embedding == [0., 1., 0.]
        assert_chunks(store, current)


@pytest.mark.parametrize("mode", ["unchanged", "missing_chunks", "no_representative", "procedural"])
def test_preparation_encodes_only_required_outputs(store, monkeypatch, mode):
    members = sources(store)
    owner = derived(store, "inspection", members)
    remaining = members[1:]
    best = max(remaining, key=lambda member: (c._trace_quality(member), member.id))
    expected = c._synthesize_digest(remaining, best=best)
    store.update_node(owner.id, content=expected, representative_embedding=[1., 0., 0.])
    if mode == "missing_chunks":
        store.replace_node_chunks(owner.id, [])
    elif mode == "no_representative":
        store.update_node(owner.id, content="Harbor evidence still needs refreshing.")
        for member in remaining:
            store.update_node(member.id, representative_embedding=[])
    elif mode == "procedural":
        store.update_node(owner.id, context={**owner.context, "procedure_id": "harbor-check"},
                          provenance={**owner.provenance, "strategy": "procedural"})
    retries, encoded = output_work(store, monkeypatch, RLock())
    preparation = c.prepare_forget(store, members[0].id)
    c.memory_forget(store, members[0].id, preparation=preparation)
    assert retries == []
    assert len(encoded) == (1 if mode == "missing_chunks" else 0)
    if mode in ("unchanged", "missing_chunks"):
        assert_chunks(store, store.get_node(owner.id))


@pytest.mark.parametrize("scope", ["project:other-harbor", "", None])
def test_prepared_teach_uses_corrective_scope_in_output_identity(store, monkeypatch, scope):
    source, owners = independent_owners(store)
    context = {"scope": scope}
    retries, encoded = output_work(store, monkeypatch, RLock())
    preparation = c.prepare_teach(store, source.id, CORRECTION, context=context)
    taught = c.memory_teach(store, source.id, CORRECTION, context=context, preparation=preparation)
    assert taught.corrective_trace.scope == (scope or store.config.default_scope)
    assert retries == [] and len(encoded) == len(owners)


def test_diamond_batches_current_siblings_without_encoding_unwritten_sources(store, monkeypatch):
    source, owners = independent_owners(store)
    joined = [derived(store, f"joined {i}", owners[i:i + 2]) for i in (0, 2)]
    tip = derived(store, "tip", joined)
    retries, encoded = output_work(store, monkeypatch, RLock())
    preparation = c.prepare_teach(store, source.id, CORRECTION)
    assert len(encoded) == len(owners)
    assert all(preparation.digests[node.id].chunks is None for node in [*joined, tip])
    c.memory_teach(store, source.id, CORRECTION, preparation=preparation)
    assert [set(batch) for batch in retries] == [{node.id for node in joined}, {tip.id}]
    assert len(encoded) == len(owners) + len(joined) + 1
    assert all("awaits its current evidence" not in text for batch in encoded for text in batch)
    for node in [*owners, *joined, tip]:
        current = store.get_node(node.id)
        assert CORRECTION in current.content and source.content not in current.content
        assert_chunks(store, current)


@pytest.mark.parametrize("materializer", ["cluster", "promotion"])
def test_multi_output_materialization_batches_required_chunks(store, monkeypatch, materializer):
    members = [store.create_node(
        level="trace" if materializer == "cluster" else "concept",
        content=f"Harbor inspection {i} records distinct equipment {i}.",
        context={"scope": SCOPE if materializer == "cluster" else f"project:harbor-{i}",
                 "procedure_id": f"group-{i}" if materializer == "cluster" else "shared"},
        embedding=[1., 0., 0.],
    ) for i in range(4)]
    owners = []
    if materializer == "promotion":
        owners = [store.create_node(
            level="concept", content=f"Harbor promotion {i} awaits evidence.",
            context={"scope": "global", "procedure_id": "shared"},
            representative_embedding=[1., 0., 0.],
            provenance={"strategy": "cross-scope-promotion", "source_concepts": [n.id for n in members]},
        ) for i in range(4)]
        expected_order = [node.id for node in store.list_nodes(level="concept", scope="global")]
    guard = RLock()
    retries, encoded = output_work(store, monkeypatch, guard)
    if materializer == "cluster":
        result = [node for node, _ in materialize(store, members, guard)]
    else:
        result = c._cross_scope_promotion(store, members[0], step=lambda: guard)
        assert [node.id for node in result] == expected_order
        assert {node.id for node in result} == {node.id for node in owners}
    assert len(result) == 4 and len(encoded) == 4
    assert len(retries) == 1 and len(retries[0]) == 4
    for node in result:
        assert_chunks(store, node)


@pytest.mark.parametrize("policy", ["audit", "credit"])
@pytest.mark.parametrize("missing", [False, True])
def test_native_independent_outputs_retry_outside_feedback_and_commit_once(tmp_path, monkeypatch, policy, missing):
    from living_memory import server
    from test_lookup import FakeMCP

    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", policy)
    monkeypatch.setattr(server, "_transport_session_id", lambda: "batched-outputs")
    guard = RLock()
    monkeypatch.setattr(server, "RLock", lambda: guard)
    mcp = server.create_mcp_server(tmp_path / "native.sqlite3", mcp_factory=FakeMCP)
    mcp.auto_consolidation.close()
    store = mcp.memory_store
    try:
        source, owners = independent_owners(store)
        context = {"scope": SCOPE, "agent": "inspector", "session_id": "batched-outputs"}
        recalled = mcp.tools["memory_recall"]("Harbor departure permit inspection", scope=SCOPE,
                                               ambient_context=context)
        marked = [row["id"] for row in recalled["results"]]
        assert marked
        before = list(store.connection.iterdump())
        if missing:
            prepare = server.prepare_teach

            def without_chunks(*args, **kwargs):
                result = prepare(*args, **kwargs)
                result.digests = {key: replace(value, chunks=None) for key, value in result.digests.items()}
                return result

            monkeypatch.setattr(server, "prepare_teach", without_chunks)
        retries, _ = output_work(store, monkeypatch, guard)
        prepare_required = c._DerivedOutputRequired.prepare

        def rolled_back(*args):
            assert list(store.connection.iterdump()) == before
            return prepare_required(*args)

        monkeypatch.setattr(c._DerivedOutputRequired, "prepare", rolled_back)
        result = mcp.tools["memory_teach"](source.id, CORRECTION, context=context,
                                            used=marked[:1], irrelevant=marked[1:])
        assert len(retries) == int(missing)
        if missing:
            assert set(retries[0]) == {node.id for node in owners}
        corrective_id = result["corrective_trace"]["id"]
        assert result["feedback_marks"]["accepted"] == len(marked)
        rows = store.connection.execute("SELECT source_id FROM recall_feedback_marks").fetchall()
        assert len(rows) == len(marked) and {row[0] for row in rows} == {corrective_id}
        assert store.get_recall_event(recalled["recall_event_id"]).feedback_trace_id == corrective_id
        assert store.node_chunks_are_current(corrective_id)
        for owner in owners:
            current = store.get_node(owner.id)
            assert corrective_id in current.source_traces and CORRECTION in current.content
            assert_chunks(store, current)
    finally:
        store.close()


@pytest.mark.parametrize("operation", ["teach", "forget", "connect"])
def test_native_eager_outputs_reject_concurrent_source_and_edge_changes(tmp_path, monkeypatch, operation):
    from living_memory import server
    from test_lookup import FakeMCP

    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    guard = RLock()
    monkeypatch.setattr(server, "RLock", lambda: guard)
    mcp = server.create_mcp_server(tmp_path / "native-race.sqlite3", mcp_factory=FakeMCP)
    mcp.auto_consolidation.close()
    store = mcp.memory_store
    entered, release = Event(), Event()
    try:
        source, owners = independent_owners(store, level="concept")
        replacement = store.append_trace(CORRECTION, {"scope": SCOPE})
        prepare = c._prepare_derivation_digests

        def pause(*args):
            result = prepare(*args)
            assert not guard._is_owned() and not store.connection.in_transaction
            assert len(result.digests) == 4 and all(value.chunks for value in result.digests.values())
            entered.set()
            assert release.wait(15)
            return result

        monkeypatch.setattr(c, "_prepare_derivation_digests", pause)
        retries, _ = output_work(store, monkeypatch, guard)
        with ThreadPoolExecutor(max_workers=2) as workers:
            if operation == "teach":
                work = workers.submit(mcp.tools["memory_teach"], source.id, CORRECTION)
            elif operation == "forget":
                work = workers.submit(mcp.tools["memory_forget"], source.id)
            else:
                work = workers.submit(mcp.tools["memory_connect"], replacement.id, source.id, "supersedes")
            try:
                assert entered.wait(10)
                lookup = workers.submit(mcp.tools["memory_lookup"], node_ids=[node.id for node in owners])
                assert len(lookup.result(timeout=10)["results"]) == len(owners)
                with MemoryStore(store.db_path) as writer:
                    for index, owner in enumerate(owners):
                        other = next(item for item in owner.source_traces if item != source.id)
                        if index % 2:
                            new = writer.create_node(
                                level="concept", content=f"Harbor inspection {index} uses a calibrated scanner.",
                                context={"scope": SCOPE}, embedding=[1., 0., 0.],
                                stats={"usefulness_score": 100.},
                            )
                            writer.create_connection(new.id, other, "supersedes")
                        else:
                            writer.update_node(other, content=f"Harbor inspection {index} uses a calibrated scanner.")
            finally:
                release.set()
            work.result(timeout=15)
        assert len(retries) == 1 and set(retries[0]) == {node.id for node in owners}
        for owner in owners:
            current = store.get_node(owner.id)
            assert "calibrated scanner" in current.content and source.content not in current.content
            if current.embedding:
                assert_chunks(store, current)
    finally:
        release.set()
        store.close()
