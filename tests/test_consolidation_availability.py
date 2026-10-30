"""Independent consolidation owners remain atomic while ordinary tools proceed."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event, RLock
import asyncio
import json
import sqlite3

import pytest

from living_memory import consolidation as c, embeddings, server
from living_memory.auto_consolidation import PENDING_KV_KEY
from living_memory.embeddings import LocalEmbeddingModel

SCOPE = "project:harbor"


class FakeMCP:
    def __init__(self, name, instructions=None):
        self.instructions = instructions
        self.tools = {}

    def tool(self, func=None, **kwargs):
        def register(handler):
            self.tools[handler.__name__] = handler
            return handler
        return register if func is None else register(func)

    def resource(self, uri):
        return lambda handler: handler

    def prompt(self, func=None, **kwargs):
        return (lambda handler: handler) if func is None else func


@pytest.fixture
def runtime(tmp_path, monkeypatch, request):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    locks = []
    waiting = Event()

    class Lock:
        def __init__(self):
            self.lock = RLock()
            locks.append(self)

        def __enter__(self):
            if not self.lock.acquire(blocking=False):
                waiting.set()
                self.lock.acquire()
            return self

        def __exit__(self, *args):
            self.lock.release()

    monkeypatch.setattr(server, "RLock", Lock)
    mcp = server.create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=getattr(request, "param", FakeMCP))
    try:
        yield mcp, locks[0], waiting
    finally:
        mcp.auto_consolidation.close()
        mcp.memory_store.close()


@pytest.fixture
def memory(runtime, monkeypatch):
    mcp, lock, waiting = runtime
    store = mcp.memory_store
    traces = [store.append_trace(
        f"Harbor inspection checks marker {i} before departure.",
        {"scope": SCOPE, "agent": f"inspector-{i}"},
    ) for i in range(6)]
    owners = [store.create_node(
        level="concept", content=f"Harbor inspection owner {i} awaiting consolidation.",
        context={"scope": SCOPE},
        provenance={"strategy": "token-jaccard", "source_traces": [t.id for t in members]},
    ) for i, members in enumerate((traces[:4], traces[2:]))]
    # Retrieval clustering is independent of the guard and tested separately.
    cluster = c._TraceCluster(SCOPE, traces, Counter({"harbor": 6}))
    monkeypatch.setattr(c, "_cluster_traces", lambda *args, **kwargs: [cluster])
    return mcp, lock, waiting, traces, owners, cluster


def procedure(store, group):
    traces = [store.append_trace(
        f"Harbor {group} checks marker {i} before departure.",
        {"scope": SCOPE, "agent": f"inspector-{i}", "task_pattern": group,
         "procedure_id": "harbor_departure", "step_order": i + 1},
    ) for i in range(3)]
    results = c._materialize_procedural_schemas(store, traces, min_cluster_size=3)
    owner = next(node for node, _ in results if node.context["task_pattern"] == group)
    return traces, owner


@pytest.mark.parametrize("route", ["manual", "scheduled"])
@pytest.mark.parametrize("stage", ["inference", "batch_wait", "final_identity"])
@pytest.mark.parametrize("runtime", [None], indirect=True)
def test_native_encoder_work_leaves_exact_lookup_available(runtime, monkeypatch, route, stage):
    from fastmcp import Client

    mcp, lock, _ = runtime
    store = mcp.memory_store
    traces = [store.append_trace(
        f"Harbor inspection checks marker {i} before departure.",
        {"scope": SCOPE, "agent": f"inspector-{i}"},
    ) for i in range(3)]
    ready, entered, release, encoder_waiting = (Event() for _ in range(4))
    encoder_lock = embeddings._ENCODER_LOCK

    class ObservedEncoderLock:
        def __enter__(self):
            if not encoder_lock.acquire(blocking=False):
                encoder_waiting.set()
                encoder_lock.acquire()

        def __exit__(self, *args):
            encoder_lock.release()

    class Backend:
        def __init__(self, block):
            self.block = block

        def encode(self, texts, **kwargs):
            if self.block:
                entered.set()
                ready.set()
                assert release.wait(30)
            return [[1.0, 0.0, 0.0] for _ in texts] if isinstance(texts, list) else [1.0, 0.0, 0.0]

    def model(block):
        result = LocalEmbeddingModel(dimensions=3, backend="auto", model_name="harbor-fixture")
        result._model = Backend(block)
        result._load_attempted = True
        return result

    encoder, other = model(stage == "inference"), model(True)
    batch, identity = encoder.embed_batch, encoder.encoder_id
    identities = 0

    def wait_for_other():
        ready.set()
        assert entered.wait(15)

    def encode(texts):
        if stage == "batch_wait":
            wait_for_other()
        return batch(texts)

    def encoder_id():
        nonlocal identities
        identities += 1
        if stage == "final_identity" and identities == 2:
            wait_for_other()
        return identity()

    monkeypatch.setattr(embeddings, "_ENCODER_LOCK", ObservedEncoderLock())
    monkeypatch.setattr(encoder, "embed_batch", encode)
    monkeypatch.setattr(encoder, "encoder_id", encoder_id)
    monkeypatch.setattr(c, "LocalEmbeddingModel", lambda **kwargs: encoder)
    mcp.auto_consolidation._retry_delays = ()
    unguarded = []
    store.connection.set_trace_callback(
        lambda sql: unguarded.append(sql) if not lock.lock._is_owned() else None
    )

    def call(name, arguments):
        async def request():
            async with Client(mcp) as client:
                response = await client.call_tool(name, arguments)
                assert not response.is_error
                return response.data
        return asyncio.run(request())

    def consolidate():
        if route == "manual":
            return call("memory_consolidate", {"scope": SCOPE})
        mcp.auto_consolidation.request(SCOPE)
        assert mcp.auto_consolidation.wait_idle(40, include_retries=True)
        assert mcp.auto_consolidation.last_result(SCOPE)["status"] == "completed"

    try:
        with ThreadPoolExecutor(max_workers=3) as workers:
            work = workers.submit(consolidate)
            contender = None
            try:
                assert ready.wait(15)
                if stage != "inference":
                    contender = workers.submit(other.embed, "Independent encoder inference")
                    assert encoder_waiting.wait(15)
                read = workers.submit(call, "memory_lookup", {"node_id": traces[0].id})
                response = read.result(timeout=10)
                assert response["results"][0]["content"] == traces[0].content
                assert not release.is_set()
                assert not work.done()
                if contender is not None:
                    assert not contender.done()
            finally:
                release.set()
            work.result(timeout=15)
            if contender is not None:
                contender.result(timeout=15)
    finally:
        store.connection.set_trace_callback(None)
    assert unguarded == []
    concepts = store.list_nodes(level="concept", scope=SCOPE)
    assert len(concepts) == 1
    assert set(concepts[0].source_traces) == {trace.id for trace in traces}
    assert store.get_consolidation_embeddings(traces, identity()) == {
        trace.id: [1.0, 0.0, 0.0] for trace in traces
    }


@pytest.mark.parametrize("route", ["manual", "scheduled"])
@pytest.mark.parametrize("runtime", [None], indirect=True)
def test_native_procedural_pass_guards_every_sql_statement(runtime, monkeypatch, route):
    from fastmcp import Client

    mcp, lock, _ = runtime
    store = mcp.memory_store
    traces, owner = procedure(store, "inspection")
    statements, unguarded = [], []
    cluster = c._cluster_traces

    def group(*args, **kwargs):
        assert not lock.lock._is_owned()
        return cluster(*args, **kwargs)

    monkeypatch.setattr(c, "_cluster_traces", group)
    mcp.auto_consolidation._retry_delays = ()

    def sql(statement):
        statements.append(statement)
        if not lock.lock._is_owned():
            unguarded.append(statement)

    store.connection.set_trace_callback(sql)
    try:
        if route == "manual":
            async def consolidate():
                async with Client(mcp) as client:
                    response = await client.call_tool("memory_consolidate", {"scope": SCOPE})
                    assert not response.is_error
                    return response.data

            result = asyncio.run(consolidate())
            assert [node["id"] for node in result["schemas_updated"]] == [owner.id]
        else:
            mcp.auto_consolidation.request(SCOPE)
            assert mcp.auto_consolidation.wait_idle(15, include_retries=True)
            assert mcp.auto_consolidation.last_result(SCOPE)["status"] == "completed"
    finally:
        store.connection.set_trace_callback(None)
    assert statements
    assert unguarded == []
    assert store.get_node(owner.id).context["procedure"] == [trace.content for trace in traces]


@pytest.mark.parametrize("commit", [True, False])
def test_procedural_discovery_waits_for_transaction_and_reads_one_snapshot(runtime, monkeypatch, commit):
    mcp, lock, waiting = runtime
    store = mcp.memory_store
    traces, owner = procedure(store, "inspection")
    entered = Event()
    snapshots = []
    triggers, get_nodes = store.schema_triggers, store.get_nodes

    def discover(scope, **kwargs):
        entered.set()
        return triggers(scope, **kwargs)

    def read(ids):
        nodes = get_nodes(ids)
        snapshots.append(nodes)
        return nodes

    monkeypatch.setattr(store, "schema_triggers", discover)
    monkeypatch.setattr(store, "get_nodes", read)

    class Rollback(Exception):
        pass

    with ThreadPoolExecutor(max_workers=1) as workers:
        try:
            with lock, store.transaction():
                store.update_node(owner.id, context={**owner.context, "transaction_fact": "new"})
                store.update_node(traces[0].id, context={**traces[0].context, "transaction_fact": "new"})
                work = workers.submit(c._materialize_procedural_schemas, store, [], min_cluster_size=3, step=lambda: lock)
                assert waiting.wait(10)
                assert not entered.is_set()
                assert not work.done()
                if not commit:
                    raise Rollback
        except Rollback:
            pass
        result = work.result(timeout=15)
    assert entered.is_set()
    expected = "new" if commit else None
    assert snapshots[0][owner.id].context.get("transaction_fact") == expected
    assert snapshots[1][traces[0].id].context.get("transaction_fact") == expected
    assert [node.id for node, _ in result] == [owner.id]
    assert store.get_node(owner.id).context["procedure"] == [trace.content for trace in traces]


@pytest.mark.parametrize("change", [
    "source", "owner", "owner_decay", "owner_group", "owner_edge", "source_decay", "source_union",
    "duplicate", "duplicate_decay", "duplicate_group", "duplicate_content", "duplicate_edge", "unchanged",
])
def test_procedural_write_rechecks_snapshot_and_preserves_independent_owner(runtime, change):
    mcp, lock, _ = runtime
    store = mcp.memory_store
    traces, owner = procedure(store, "inspection")
    others, independent = procedure(store, "navigation")
    duplicate = store.create_node(
        level="schema", content=owner.content + "\nAdditional inspection record.",
        context=owner.context, provenance=owner.provenance,
    )
    discovered, resume = Event(), Event()
    steps = 0

    @contextmanager
    def step():
        nonlocal steps
        with lock:
            yield
        steps += 1
        assert not store.connection.in_transaction
        if steps == 1:
            discovered.set()
            assert resume.wait(15)

    with ThreadPoolExecutor(max_workers=1) as workers:
        work = workers.submit(c._materialize_procedural_schemas, store, traces + others, min_cluster_size=3, step=step)
        try:
            assert discovered.wait(10)
            # Foreground tools acquire the same runtime lock between discovery and writes.
            if change in ("source", "owner", "duplicate"):
                target = {"source": traces[0], "owner": owner, "duplicate": duplicate}[change]
                taught = mcp.tools["memory_teach"](target.id, "Harbor departure uses the signed manifest.")
            elif change in ("owner_decay", "source_decay", "duplicate_decay"):
                target = {"owner_decay": owner, "source_decay": traces[0], "duplicate_decay": duplicate}[change]
                mcp.tools["memory_forget"](target.id, "retired by operator")
            else:
                with lock, store.transaction():
                    if change in ("owner_group", "duplicate_group"):
                        target = owner if change == "owner_group" else duplicate
                        store.update_node(target.id, context={**target.context, "procedure_key": "independent revision", "task_pattern": "independent revision"})
                    elif change == "duplicate_content":
                        store.update_node(duplicate.id, content="Harbor departure checks the revised inspection record.")
                    elif change in ("owner_edge", "duplicate_edge"):
                        replacement = store.append_trace("Harbor departure replaces the duplicate procedure.", {"scope": SCOPE})
                        target = owner if change == "owner_edge" else duplicate
                        store.create_connection(replacement.id, target.id, "supersedes")
                    elif change == "source_union":
                        added = store.append_trace("Harbor departure records the final manifest.", {**traces[0].context, "step_order": 4})
                        store.update_node(owner.id, provenance={**owner.provenance, "source_traces": [*owner.source_traces, added.id]})
            with lock:
                current_owner = store.get_node(owner.id)
                current_duplicate = store.get_node(duplicate.id)
        finally:
            resume.set()
        result = work.result(timeout=15)
    updated = {node.id for node, _ in result}
    assert independent.id in updated
    assert store.get_node(independent.id).context["procedure"] == [trace.content for trace in others]
    current = store.get_node(owner.id)
    if change in ("owner", "owner_decay", "owner_group", "owner_edge"):
        assert current == current_owner
        assert owner.id not in updated
    else:
        assert owner.id in updated
        if change == "source":
            assert taught["corrective_trace"]["id"] in current.source_traces
            assert traces[0].id not in current.source_traces
            assert current.context["procedure"] == ["Harbor departure uses the signed manifest.", *[t.content for t in traces[1:]]]
        elif change == "source_decay":
            assert traces[0].id not in current.source_traces
            assert current.context["procedure"] == [trace.content for trace in traces[1:]]
        elif change == "source_union":
            assert added.id in current.source_traces
            assert added.content in current.content
    if change.startswith("duplicate") or change in ("owner", "owner_decay", "owner_group", "owner_edge", "source", "source_decay"):
        assert store.get_node(duplicate.id) == current_duplicate
    else:
        assert store.get_node(duplicate.id).decay_reason == "procedural: duplicate group"


def test_procedural_grouping_and_between_owner_reads_release_the_guard(runtime):
    mcp, lock, _ = runtime
    store = mcp.memory_store
    traces, owner = procedure(store, "inspection")
    others, independent = procedure(store, "navigation")
    between, resume = Event(), Event()
    steps = 0

    def snapshot():
        assert not lock.lock._is_owned()
        yield from traces + others

    @contextmanager
    def step():
        nonlocal steps
        with lock:
            yield
        steps += 1
        assert not store.connection.in_transaction
        if steps == 2:
            between.set()
            assert resume.wait(15)

    with ThreadPoolExecutor(max_workers=2) as workers:
        work = workers.submit(c._materialize_procedural_schemas, store, snapshot(), min_cluster_size=3, step=step)
        try:
            assert between.wait(10)
            read = workers.submit(mcp.tools["memory_lookup"], node_ids=[owner.id, independent.id])
            assert len(read.result(timeout=10)["results"]) == 2
            assert not work.done()
        finally:
            resume.set()
        result = work.result(timeout=15)
    assert steps == 3  # one discovery snapshot, then each owner separately
    assert {node.id for node, _ in result} == {owner.id, independent.id}


def test_reader_waits_for_one_complete_owner_then_proceeds_before_the_next(memory, monkeypatch):
    mcp, lock, waiting, traces, owners, cluster = memory
    writing, release_write, between, release_pass = (Event() for _ in range(4))
    original = c._update_edge_weights_from_co_access
    written = []

    def edges(store, owner, members):
        if not written:
            writing.set()
            assert release_write.wait(15)
        original(store, owner, members)
        written.append(owner.id)

    monkeypatch.setattr(c, "_update_edge_weights_from_co_access", edges)

    @contextmanager
    def step():
        with lock:
            yield
        if len(written) == 1:
            assert not mcp.memory_store.connection.in_transaction
            between.set()
            assert release_pass.wait(15)

    with ThreadPoolExecutor(max_workers=2) as workers:
        work = workers.submit(c.memory_consolidate, mcp.memory_store, scope=SCOPE, guard=step)
        try:
            assert writing.wait(10)
            read = workers.submit(mcp.tools["memory_lookup"], node_ids=[n.id for n in owners])
            assert waiting.wait(10)
            assert not read.done()
            with sqlite3.connect(mcp.memory_store.db_path) as external:
                assert external.execute("SELECT content FROM nodes WHERE id=?", (owners[0].id,)).fetchone() == (owners[0].content,)
            release_write.set()
            assert between.wait(10)
            response = read.result(timeout=10)
            assert not work.done()
            assert [r["id"] for r in response["results"]] == [n.id for n in owners]
            current = mcp.memory_store.get_node(written[0])
            assert current.content != next(n.content for n in owners if n.id == current.id)
            assert all(mcp.memory_store.list_connections(source_id=current.id, target_id=source) for source in current.source_traces)
        finally:
            release_write.set()
            release_pass.set()
        result = work.result(timeout=15)
    assert {n.id for n in result.concepts_updated} == {n.id for n in owners}


@pytest.mark.parametrize("change", ["source", "owner", "decay", "union", "content", "metadata"])
def test_between_owner_write_uses_current_sources_and_preserves_foreground_feedback(memory, monkeypatch, change):
    mcp, lock, _, traces, owners, cluster = memory
    store = mcp.memory_store
    if change == "content":
        # Derived sources can change content while retaining their identity.
        traces[2] = store.update_node(traces[2].id, level="concept")
    written = []
    original = c._write_cluster_concept

    def write(*args, **kwargs):
        result = original(*args, **kwargs)
        written.append(result.id)
        return result

    monkeypatch.setattr(c, "_write_cluster_concept", write)
    changed = False
    facts = {}

    @contextmanager
    def step():
        nonlocal changed
        with lock:
            yield
        if written and not changed:
            changed = True
            assert not store.connection.in_transaction
            remaining = next(n for n in owners if n.id != written[0])
            recalled = mcp.tools["memory_recall"](
                "harbor inspection", scope=SCOPE,
                ambient_context={"transport_session_id": "foreground"},
            )
            remembered = mcp.tools["memory_remember"](
                "Harbor inspection records the final departure time.",
                {"scope": SCOPE, "transport_session_id": "foreground"},
                used=[recalled["results"][0]["id"]],
            )
            assert remembered["feedback_marks"]["accepted"] == 1
            assert store.get_node(remembered["node"]["id"]) is not None
            facts.update(remaining=remaining.id, event=recalled["recall_event_id"], remembered=remembered["node"]["id"])
            if change == "source":
                taught = mcp.tools["memory_teach"](traces[2].id, "Harbor inspection now uses the signed departure manifest.")
                facts["replacement"] = taught["corrective_trace"]["id"]
            elif change == "owner":
                mcp.tools["memory_teach"](remaining.id, "Harbor inspection uses an independent current procedure.")
            elif change == "decay":
                mcp.tools["memory_forget"](remaining.id, "retired by operator")
            elif change == "union":
                current = store.get_node(remaining.id)
                store.update_node(remaining.id, provenance={**current.provenance, "source_traces": [*current.source_traces, remembered["node"]["id"]]})
                store.update_node(traces[2].id, stats={"access_count": 77, "usefulness_score": 4})
            else:
                updates = {"content": "Harbor inspection verifies the calibrated compass."} if change == "content" else {}
                store.update_node(
                    traces[2].id, **updates, embedding=[0.0, 1.0],
                    context={"scope": SCOPE, "agent": "independent-inspector"},
                    stats={"access_count": 99, "usefulness_score": 1000, "confidence": 0.9, "unique_agents": 12},
                )

    c._materialize_cluster(store, cluster, min_cluster_size=3, step=step)
    current = store.get_node(facts["remaining"])
    assert store.get_node(facts["remembered"]) is not None
    assert store.connection.execute("SELECT feedback_applied FROM recall_events WHERE id=?", (facts["event"],)).fetchone()[0] == 1
    if change == "source":
        assert facts["replacement"] in current.source_traces
        assert traces[2].id not in current.source_traces
        assert "signed departure manifest" in current.content
    elif change == "owner":
        assert current.corrections and current.content == next(n.content for n in owners if n.id == current.id)
        assert current.id not in written
    elif change == "decay":
        assert current.decayed and current.decay_reason == "retired by operator"
        assert current.id not in written
    elif change == "union":
        assert facts["remembered"] in current.source_traces
        edge, = store.list_connections(source_id=current.id, target_id=traces[2].id, relation_type="related")
        assert edge.metadata["co_access"] == 77
    else:
        members = c._current_derivation(store, current).live
        best = max(members, key=lambda member: (c._trace_quality(member), member.id))
        expected = c._prepare_digest(SCOPE, members, best, c._prepare_digest_sources(members))
        assert best.id == traces[2].id
        assert current.content == expected.content
        assert current.provenance["cluster_key"] == expected.cluster_key
        assert current.embedding == best.embedding == [0.0, 1.0]
        assert current.unique_agents == 12
        assert current.confidence == c._consensus_confidence(members, 12)
        assert current.usefulness_score == sum(member.usefulness_score for member in members) / len(members)
        edge, = store.list_connections(source_id=current.id, target_id=best.id, relation_type="related")
        assert edge.metadata["co_access"] == 99
        if change == "content":
            assert "calibrated compass" in current.content
            assert traces[2].content not in current.content


def test_interrupted_pass_rolls_back_current_owner_and_resumes_durable_pending(memory, monkeypatch):
    mcp, _, _, traces, owners, cluster = memory
    store = mcp.memory_store
    original = c._update_edge_weights_from_co_access
    written = []

    def fail(store, owner, members):
        original(store, owner, members)
        written.append(owner.id)
        if len(written) == 2:
            raise RuntimeError("interrupted owner write")

    monkeypatch.setattr(c, "_update_edge_weights_from_co_access", fail)
    mcp.auto_consolidation._retry_delays = ()
    mcp.auto_consolidation.request(SCOPE)
    assert mcp.auto_consolidation.wait_idle(15, include_retries=True)
    assert mcp.auto_consolidation.last_result(SCOPE)["status"] == "failed"
    assert SCOPE in json.loads(store.get_kv(PENDING_KV_KEY))
    assert store.get_node(written[0]).content != next(n.content for n in owners if n.id == written[0])
    interrupted = store.get_node(written[1])
    assert interrupted.content == next(n.content for n in owners if n.id == interrupted.id)
    assert not store.list_connections(source_id=interrupted.id, relation_type="related")
    assert not store.connection.in_transaction
    mcp.auto_consolidation.close()
    monkeypatch.setattr(c, "_update_edge_weights_from_co_access", original)
    # Reopening the actual server consumes the same persisted pending scope.
    resumed = server.create_mcp_server(store.db_path, mcp_factory=FakeMCP)
    try:
        assert resumed.auto_consolidation.wait_idle(15)
        assert resumed.auto_consolidation.last_result(SCOPE)["status"] == "completed"
        assert json.loads(resumed.memory_store.get_kv(PENDING_KV_KEY)) == {}
        assert {n.id for n in resumed.memory_store.list_nodes(level="concept", scope=SCOPE)} == {n.id for n in owners}
        for owner in owners:
            current = resumed.memory_store.get_node(owner.id)
            assert current.content != owner.content
            assert set(current.source_traces) == set(owner.source_traces)
            assert all(resumed.memory_store.list_connections(source_id=owner.id, target_id=source) for source in current.source_traces)
    finally:
        resumed.auto_consolidation.close()
        resumed.memory_store.close()
