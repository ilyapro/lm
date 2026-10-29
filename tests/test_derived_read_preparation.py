"""Native pending-index and feedback preparation keep independent reads usable."""

import asyncio
import json
from threading import RLock

from fastmcp import Client
import pytest

from living_memory import feedback, retrieval, server
from living_memory.chunking import chunk_text
from living_memory.irrelevance import list_query_irrelevance
from living_memory.query_anchors import resolve_query_anchor
from living_memory.storage import MemoryStore, recall_fingerprint
from test_recall_query_preparation import (
    CONTENT, QUERY, SCOPE, VECTOR, call, entry, recall, runtime as query_runtime,
)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    guard = RLock()
    monkeypatch.setattr(server, "RLock", lambda: guard)
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    monkeypatch.setenv("LM_LOOKUP_CREDIT_POLICY", "delivered")
    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", "grounded")
    fixture = query_runtime.__wrapped__(tmp_path, monkeypatch)
    r = next(fixture)
    r.guard, r.unlocked_sql, r.batches = guard, [], []
    r.feedback_hold = False
    original = feedback._anchor_embedder

    def anchor_encoder(store):
        encode = original(store)

        def run(queries):
            assert not guard._is_owned()
            assert not store.connection.in_transaction
            r.batches.append(tuple(queries))
            if r.feedback_hold:
                r.entered.set()
                assert r.release.wait(10), "feedback encoder was not released"
            return encode(queries)

        return run

    monkeypatch.setattr(feedback, "_anchor_embedder", anchor_encoder)
    r.store.connection.set_trace_callback(
        lambda statement: r.unlocked_sql.append(statement) if not guard._is_owned() else None,
    )
    try:
        yield r
        assert not r.unlocked_sql
    finally:
        r.release.set()
        r.store.connection.set_trace_callback(None)
        with pytest.raises(StopIteration):
            next(fixture)


def anchor(store, query=QUERY, scope=SCOPE):
    return store.find_query_anchor(scope, recall_fingerprint(query, scope))


def pending_source(r, clustered):
    with r.guard:
        node = r.store.create_node(
            level="concept", content=CONTENT * 80, context={"scope": SCOPE},
            provenance={"strategy": "embedding-cosine", "source_traces": [r.node.id]}
            if clustered else None,
        )
        if clustered:
            with r.store.transaction():
                r.store.connection.execute("UPDATE nodes SET embedding = ? WHERE id = ?",
                                           (json.dumps([0.0, 1.0, 0.0]), node.id))
        node = r.store.get_node(node.id)
    encode = r.store._resolve_chunk_embedder()

    def held(texts):
        assert not r.guard._is_owned()
        assert not r.store.connection.in_transaction
        r.batches.append(tuple(texts))
        r.entered.set()
        assert r.release.wait(10), "pending encoder was not released"
        return encode(texts)

    r.store.set_chunk_embedder(held)
    return node


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("view", ["knowledge", "full", "prompt"])
def test_pending_index_keeps_exact_read_available_and_preserves_representative(runtime, clustered, view):
    r = runtime
    node = pending_source(r, clustered)

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as reader:
            operation = asyncio.create_task(
                owner.get_prompt("memory://prompt/retrieval_context", {"task": QUERY, "scope": SCOPE})
                if view == "prompt" else recall(owner, view=view)
            )
            try:
                assert await asyncio.to_thread(r.entered.wait, 4)
                lookup = await asyncio.wait_for(call(reader, "memory_lookup", node_id=r.node.id), 3)
                assert entry(lookup)["content"] == CONTENT
                await asyncio.wait_for(call(reader, "memory_status", scope=SCOPE), 3)
                assert not operation.done()
            finally:
                r.release.set()
            result = await asyncio.wait_for(operation, 5)
            with r.guard:
                chunks = r.store.list_node_chunks(node.id)
                assert len(chunks) == len(chunk_text(node.content)) == len(r.batches[0])
                assert r.store.node_chunks_are_current(node.id)
                assert all(node.content[c.char_start:c.char_end] for c in chunks)
                current = r.store.get_node(node.id)
                assert current.embedding == (node.embedding if clustered else VECTOR)
                assert current.provenance == node.provenance
                if view == "prompt":
                    assert node.id in result.messages[0].content.text
                    assert r.store.list_recall_events() == []
                else:
                    event = r.store.get_recall_event(result["recall_event_id"])
                    assert len(r.store.recall_delivery_history(event.transport_session_id)) == 1
                    assert event.result_ids == [entry(result)["id"]]
            assert (node.content in r.encoded) == (not clustered)

    asyncio.run(scenario())


@pytest.mark.parametrize("clustered", [False, True])
@pytest.mark.parametrize("change", ["edit", "delete", "decay", "scope", "index", "representative"])
def test_pending_revalidation_preserves_current_source_and_index(runtime, clustered, change):
    r = runtime
    node = pending_source(r, clustered)

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as reader:
            operation = asyncio.create_task(recall(owner))
            try:
                assert await asyncio.to_thread(r.entered.wait, 4)
                with MemoryStore(r.store.db_path, chunk_embedder=lambda texts: [VECTOR for _ in texts]) as writer:
                    if change == "edit":
                        writer.update_node(node.id, content="Current amber condition.", embedding=[0.0, 1.0, 0.0])
                    elif change == "delete":
                        with writer.transaction():
                            writer.delete_node_chunks(node.id)
                            writer.connection.execute("DELETE FROM nodes WHERE id = ?", (node.id,))
                    elif change == "decay":
                        await asyncio.wait_for(call(reader, "memory_forget", id=node.id), 3)
                    elif change == "scope":
                        writer.update_node(node.id, context={"scope": "project:moved"})
                    elif change == "index":
                        writer.replace_node_chunks(node.id, [(c, [0.0, 1.0, 0.0]) for c in chunk_text(node.content)])
                    else:
                        with writer.transaction():
                            writer.connection.execute("UPDATE nodes SET embedding = ? WHERE id = ?",
                                                      (json.dumps([0.0, 0.0, 1.0]), node.id))
                    expected = writer.list_node_chunks(node.id)
                    current = writer.get_node(node.id)
                lookup = await asyncio.wait_for(call(reader, "memory_lookup", node_id=node.id), 3)
                assert bool(lookup["results"]) == (change != "delete")
                assert not operation.done()
            finally:
                r.release.set()
            await asyncio.wait_for(operation, 5)
            with r.guard:
                assert r.store.list_node_chunks(node.id) == expected
                final = r.store.get_node(node.id)
                if current is None:
                    assert final is None
                else:
                    assert (final.content, final.scope, final.decayed, final.embedding, final.provenance) == (
                        current.content, current.scope, current.decayed, current.embedding, current.provenance,
                    )

    asyncio.run(scenario())


async def consume(client, route, node_id):
    if route == "lookup":
        return await call(client, "memory_lookup", node_id=node_id)
    if route in ("recall", "negative"):
        return await recall(client, query="next certificate request", **{
            "used" if route == "recall" else "irrelevant": [node_id],
        })
    if route == "teach":
        return await call(client, "memory_teach", trace_id=node_id,
                          correction=CONTENT + " Confirmed for release.", used=[node_id], context={"scope": SCOPE})
    return await call(client, "memory_remember", content=CONTENT + " Confirmed for release.",
                      context={"scope": SCOPE}, **({"used": [node_id]} if route == "explicit" else {}))


@pytest.mark.parametrize("route", ["lookup", "recall", "negative", "explicit", "implicit", "teach"])
def test_missing_anchor_preparation_keeps_reads_available_for_every_consumer(runtime, route):
    r = runtime

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as reader:
            prior = await recall(owner)
            r.feedback_hold = True
            operation = asyncio.create_task(consume(owner, route, r.node.id))
            try:
                assert await asyncio.to_thread(r.entered.wait, 4)
                lookup = await asyncio.wait_for(call(reader, "memory_lookup", node_id=r.node.id), 3)
                assert entry(lookup)["content"] == CONTENT
                assert not operation.done()
                with r.guard:
                    assert r.store.credited_node_ids(prior["recall_event_id"]) == set()
                    assert r.store.count_query_anchors() == 0
            finally:
                r.release.set()
            await asyncio.wait_for(operation, 5)
            assert r.batches == [(QUERY,)]
            with r.guard:
                assert r.store.count_query_anchors() == 1
                assert r.store.credited_node_ids(prior["recall_event_id"]) == (
                    set() if route == "negative" else {r.node.id}
                )
                if route == "negative":
                    assert list_query_irrelevance(r.store)[0]["weight"] == 0.5
                else:
                    edge, = r.store.list_query_anchor_edges(anchor_id=anchor(r.store).id)
                    assert edge.hits == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["lookup", "explicit", "implicit"])
def test_simultaneous_feedback_consumers_claim_current_credit_once(runtime, route):
    r = runtime

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as reader:
            prior = await recall(owner)
            r.feedback_hold = True
            first = asyncio.create_task(consume(owner, route, r.node.id))
            second = None
            try:
                assert await asyncio.to_thread(r.entered.wait, 4)
                second = asyncio.create_task(consume(owner, route, r.node.id))
                await asyncio.wait_for(call(reader, "memory_lookup", node_id=r.node.id), 3)
                assert not first.done()
            finally:
                r.release.set()
            await asyncio.wait_for(asyncio.gather(first, second), 5)
            with r.guard:
                assert r.store.credited_node_ids(prior["recall_event_id"]) == {r.node.id}
                assert r.store.count_query_anchors() == 1
                current = anchor(r.store)
                assert current.reinforcement_count == 0
                edge, = r.store.list_query_anchor_edges(anchor_id=current.id)
                assert edge.hits == 1
                assert sum(r.store.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                           for table in ("recall_credit_ledger", "recall_explicit_credit")) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["explicit", "teach"])
def test_missing_anchor_required_failure_rolls_back_knowledge_and_all_feedback(runtime, operation):
    r = runtime

    async def scenario():
        async with Client(r.mcp) as owner:
            prior = await recall(owner)
            with r.guard:
                before = r.store.get_node(r.node.id)
                chunks = r.store.list_node_chunks(r.node.id)
                r.store.connection.execute("""CREATE TRIGGER reject_closure
                    BEFORE UPDATE OF feedback_applied_at ON recall_events
                    WHEN NEW.feedback_applied_at IS NOT NULL
                    BEGIN SELECT RAISE(ABORT, 'required closure failure'); END""")
            with pytest.raises(Exception, match="required closure failure"):
                await consume(owner, operation, r.node.id)
            with r.guard:
                assert r.store.get_node(r.node.id) == before
                assert r.store.list_node_chunks(r.node.id) == chunks
                assert r.store.get_recall_event(prior["recall_event_id"]).feedback_trace_id is None
                for table, expected in [("nodes", 1), ("connections", 0), ("recall_feedback_marks", 0),
                                        ("recall_credit_ledger", 0), ("query_anchors", 0),
                                        ("query_anchor_edges", 0), ("recall_explicit_credit", 0)]:
                    assert r.store.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == expected
                assert list_query_irrelevance(r.store) == []
                assert not r.store.connection.in_transaction
            assert r.batches == [(QUERY,)]

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["create", "replace", "delete"])
@pytest.mark.parametrize("route", ["lookup", "implicit"])
def test_feedback_uses_current_anchor_identity_after_encoding(runtime, change, route):
    r = runtime
    other_query = "other pending feedback query"
    with r.guard:
        other = r.store.create_node(level="concept", content="Other pending rule.",
                                    context={"scope": SCOPE}, embedding=VECTOR)
        existing = None if change == "create" else resolve_query_anchor(r.store, QUERY, SCOPE, [0.0, 1.0])

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as reader:
            first = await recall(owner)
            # A second delivered id forces missing preparation even when the
            # first query already has an authoritative anchor.
            with r.guard:
                event = r.store.get_recall_event(first["recall_event_id"])
                second = r.store.record_recall_event(
                    query=other_query, scope=SCOPE, ambient_context={"transport_session_id": event.transport_session_id},
                    results=[{"node_id": other.id, "vector_score": 1.0}],
                )
            r.feedback_hold = True
            operation = asyncio.create_task(
                call(owner, "memory_lookup", node_ids=[r.node.id, other.id]) if route == "lookup" else
                call(owner, "memory_remember", content=CONTENT + " " + other.content, context={"scope": SCOPE})
            )
            created = None
            try:
                assert await asyncio.to_thread(r.entered.wait, 4)
                with MemoryStore(r.store.db_path) as writer:
                    if existing is not None:
                        with writer.transaction():
                            writer.connection.execute("DELETE FROM query_anchors WHERE id = ?", (existing.id,))
                    if change != "delete":
                        created = resolve_query_anchor(writer, QUERY, SCOPE, [0.0, 0.0, 1.0])
                await asyncio.wait_for(call(reader, "memory_lookup", node_id=r.node.id), 3)
            finally:
                r.release.set()
            await asyncio.wait_for(operation, 5)
            with r.guard:
                final = anchor(r.store)
                if created is not None:
                    assert final.id == created.id and final.embedding == created.embedding
                else:
                    assert final.id != existing.id and final.embedding == tuple(VECTOR)
                assert r.store.credited_node_ids(first["recall_event_id"]) == {r.node.id}
                assert r.store.credited_node_ids(second.id) == {other.id}
                assert r.store.list_query_anchor_edges(anchor_id=final.id)[0].hits == 1
            assert r.batches == (
                [(QUERY, other_query) if route == "lookup" else (other_query, QUERY)] if change == "create" else
                [(other_query,), (QUERY,)] if change == "delete" else [(other_query,)]
            )

    asyncio.run(scenario())


def test_new_delivery_during_feedback_preparation_owns_credit_event_map_and_delivery(runtime, monkeypatch):
    r = runtime
    monkeypatch.setenv("LM_RECALL_MAP", "1")
    with r.guard:
        for i in range(4):
            r.store.create_node(level="concept", content=f"Cobalt certificate logistics note {i}.",
                                context={"scope": SCOPE, "topic": "certificate logistics"},
                                embedding=[0.6, 0.8, 0.0])

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as peer:
            prior = await recall(owner, view="full")
            r.feedback_hold = True
            operation = asyncio.create_task(recall(owner, query="next certificate request", used=[r.node.id], view="full"))
            try:
                assert await asyncio.to_thread(r.entered.wait, 4)
                newest = await asyncio.wait_for(recall(owner, query="new delivery query", view="full"), 3)
                foreign = await asyncio.wait_for(recall(peer, query="foreign delivery query", view="full"), 3)
                assert r.service.last_recall_event_id == foreign["recall_event_id"]
                assert not operation.done()
            finally:
                r.release.set()
            response = await asyncio.wait_for(operation, 5)
            assert response["recall_event_id"] not in (prior["recall_event_id"], newest["recall_event_id"], foreign["recall_event_id"])
            assert response["results"][0]["delivery"] == "session_duplicate"
            assert response["feedback_marks"] == {"accepted": 1, "dropped": 0, "by_reason": {}}
            assert r.batches == [(QUERY,), ("new delivery query",)]
            with r.guard:
                event = r.store.get_recall_event(response["recall_event_id"])
                own = r.store.get_recall_event(newest["recall_event_id"])
                peer_event = r.store.get_recall_event(foreign["recall_event_id"])
                assert event.transport_session_id == own.transport_session_id != peer_event.transport_session_id
                assert event.query == response["query"]
                assert event.result_ids == [entry(response)["id"]]
                assert event.results[0]["delivery"]["kind"] == "session_duplicate"
                assert r.store.credited_node_ids(prior["recall_event_id"]) == set()
                assert r.store.credited_node_ids(own.id) == {r.node.id}
                assert r.store.credited_node_ids(peer_event.id) == set()
                mark, = r.store.connection.execute("SELECT * FROM recall_feedback_marks").fetchall()
                assert mark["source_id"] == event.id and mark["recall_event_id"] == own.id
                assert mark["transport_session_id"] == event.transport_session_id
                stored_map = json.loads(r.store.connection.execute(
                    "SELECT recall_map FROM recall_events WHERE id = ?", (event.id,),
                ).fetchone()[0])
                assert response["recall_map"] == stored_map
                assert anchor(r.store) is None
                assert anchor(r.store, "new delivery query") is not None

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["edit", "delete", "decay", "scope", "correction"])
def test_feedback_rechecks_current_target_and_grounding(runtime, change):
    r = runtime

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as reader:
            prior = await recall(owner)
            r.feedback_hold = True
            operation = asyncio.create_task(consume(owner, "implicit", r.node.id))
            replacement = None
            try:
                assert await asyncio.to_thread(r.entered.wait, 4)
                with MemoryStore(r.store.db_path) as writer:
                    if change == "edit":
                        writer.update_node(r.node.id, content="The amber pond requires drainage.", embedding=[0.0, 1.0, 0.0])
                    elif change == "delete":
                        with writer.transaction():
                            writer.delete_node_chunks(r.node.id)
                            writer.connection.execute("DELETE FROM nodes WHERE id = ?", (r.node.id,))
                    elif change == "decay":
                        await asyncio.wait_for(call(reader, "memory_forget", id=r.node.id), 3)
                    elif change == "scope":
                        writer.update_node(r.node.id, context={"scope": "project:moved"})
                    else:
                        replacement = writer.create_node(level="concept", content="Use the renewed cobalt certificate.",
                                                         context={"scope": SCOPE}, embedding=VECTOR)
                        writer.create_connection(replacement.id, r.node.id, "supersedes")
                await asyncio.wait_for(call(reader, "memory_lookup", node_id=r.node.id), 3)
            finally:
                r.release.set()
            response = await asyncio.wait_for(operation, 5)
            with r.guard:
                event = r.store.get_recall_event(prior["recall_event_id"])
                assert event.feedback_trace_id == response["node"]["id"]
                current = anchor(r.store)
                if change in ("edit", "delete"):
                    assert current is None and r.store.credited_node_ids(event.id) == set()
                else:
                    assert current.scope == SCOPE
                    edges = r.store.list_query_anchor_edges(anchor_id=current.id)
                    expected = [] if change == "decay" else [replacement.id if replacement else r.node.id]
                    assert [edge.target_id for edge in edges] == expected
                    assert all(edge.hits == 1 for edge in edges)
                assert r.batches == [(QUERY,)]

    asyncio.run(scenario())


def test_concurrent_negative_mark_survives_lookup_credit(runtime):
    r = runtime

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as reader:
            prior = await recall(owner)
            r.feedback_hold = True
            operation = asyncio.create_task(consume(owner, "lookup", r.node.id))
            try:
                assert await asyncio.to_thread(r.entered.wait, 4)
                with r.guard:
                    resolve_query_anchor(r.store, QUERY, SCOPE, VECTOR)
                marked = await asyncio.wait_for(consume(owner, "negative", r.node.id), 3)
                assert marked["feedback_marks"]["accepted"] == 1
                await asyncio.wait_for(call(reader, "memory_lookup", node_id=r.node.id), 3)
            finally:
                r.release.set()
            await asyncio.wait_for(operation, 5)
            with r.guard:
                # The lookup re-resolves the newest same-second delivery and
                # must not erase the independently accepted negative signal.
                rows = list_query_irrelevance(r.store)
                assert len(rows) == 1 and rows[0]["weight"] == 0.5
                assert rows[0]["last_event_id"] == prior["recall_event_id"]
                assert sum(len(r.store.credited_node_ids(event.id)) for event in r.store.list_recall_events()) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["raise", "incomplete"])
def test_pending_encoder_failure_keeps_other_selected_and_current_indexes(runtime, fault):
    r = runtime
    with r.guard:
        failing = r.store.create_node(level="concept", content="Failed pending source. " * 150,
                                      context={"scope": SCOPE})
        later = r.store.append_trace("Independent later source.", {"scope": SCOPE})
        before = r.store.list_node_chunks(r.node.id)

    def encode(texts):
        assert not r.guard._is_owned() and not r.store.connection.in_transaction
        if fault == "raise":
            raise RuntimeError("pending model failure")
        return [VECTOR, *([] for _ in texts[1:])]

    r.store.set_chunk_embedder(encode)

    async def scenario():
        async with Client(r.mcp) as client:
            response = await recall(client)
            assert entry(response)["id"] == r.node.id
            with r.guard:
                assert r.store.list_node_chunks(r.node.id) == before
                assert r.store.list_node_chunks(failing.id) == []
                assert r.store.node_chunks_are_current(later.id)
                assert not r.store.connection.in_transaction

    asyncio.run(scenario())


def test_due_decay_precedes_bounded_pending_selection_and_is_reported_once(runtime, monkeypatch):
    r = runtime
    monkeypatch.setattr(retrieval, "LAZY_CHUNK_NODE_LIMIT", 1)
    with r.guard:
        old = r.store.append_trace("Expired pending rule.", {"scope": SCOPE, "timestamp": "2020-01-01T00:00:00Z"})
        current = r.store.append_trace("Current pending rule.", {"scope": SCOPE})
    sweeps = []

    def sweep(store):
        assert r.guard._is_owned()
        store.soft_delete_node(old.id, "expired")
        sweeps.append(old.id)
        return {"decayed": [old.id]}

    monkeypatch.setattr(server, "_maybe_decay_sweep", sweep)

    async def scenario():
        async with Client(r.mcp) as client:
            response = await recall(client)
            assert response["auto_decay"] == {"decayed": [old.id]}
            assert sweeps == [old.id]
            with r.guard:
                assert r.store.list_node_chunks(old.id) == []
                assert r.store.node_chunks_are_current(current.id)

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["raise", "invalid_vector", "empty_batch"])
def test_failed_missing_anchor_batch_keeps_current_existing_effects(runtime, monkeypatch, fault):
    r = runtime
    with r.guard:
        existing = resolve_query_anchor(r.store, QUERY, SCOPE, VECTOR)
        other = r.store.create_node(level="concept", content="Other certificate rule.",
                                    context={"scope": SCOPE}, embedding=VECTOR)

    def encoder(store):
        def fail(texts):
            assert not r.guard._is_owned() and not store.connection.in_transaction
            r.batches.append(tuple(texts))
            if fault == "raise":
                raise RuntimeError("missing anchor model failure")
            return [None] if fault == "invalid_vector" else []
        return fail

    async def scenario():
        async with Client(r.mcp) as client:
            prior = await recall(client)
            with r.guard:
                event = r.store.get_recall_event(prior["recall_event_id"])
                other_event = r.store.record_recall_event(
                    query="missing query", scope=SCOPE,
                    ambient_context={"transport_session_id": event.transport_session_id},
                    results=[{"node_id": other.id, "vector_score": 1.0}],
                )
            monkeypatch.setattr(feedback, "_anchor_embedder", encoder)
            await call(client, "memory_remember", content=CONTENT + " " + other.content,
                       context={"scope": SCOPE}, used=[r.node.id, other.id])
            assert r.batches == [("missing query",)]
            with r.guard:
                assert anchor(r.store).id == existing.id
                assert anchor(r.store).reinforcement_count == 1
                assert anchor(r.store, "missing query") is None
                assert r.store.list_query_anchor_edges(anchor_id=existing.id)[0].hits == 1
                assert r.store.credited_node_ids(prior["recall_event_id"]) == {r.node.id}
                assert r.store.credited_node_ids(other_event.id) == {other.id}

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["recall", "negative", "explicit", "implicit", "teach"])
def test_continually_changing_feedback_fails_before_mutation_without_unbounded_retry(runtime, monkeypatch, route):
    r = runtime
    queries = []
    transport = None

    def changing_encoder(store):
        def run(texts):
            assert not r.guard._is_owned() and not store.connection.in_transaction
            queries.append(tuple(texts))
            with MemoryStore(store.db_path) as writer:
                writer.record_recall_event(query=f"concurrent query {len(queries)}", scope=SCOPE,
                                           ambient_context={"transport_session_id": transport},
                                           results=[{"node_id": r.node.id, "vector_score": 1.0}])
            return [VECTOR for _ in texts]
        return run

    async def scenario():
        nonlocal transport
        async with Client(r.mcp) as owner:
            prior = await recall(owner)
            with r.guard:
                transport = r.store.get_recall_event(prior["recall_event_id"]).transport_session_id
            monkeypatch.setattr(feedback, "_anchor_embedder", changing_encoder)
            with pytest.raises(Exception, match="feedback inputs changed during preparation; retry"):
                await consume(owner, route, r.node.id)
            assert queries == [(QUERY,), ("concurrent query 1",)]
            with r.guard:
                assert r.store.connection.execute("SELECT count(*) FROM nodes").fetchone()[0] == 1
                assert r.store.connection.execute("SELECT count(*) FROM recall_feedback_marks").fetchone()[0] == 0
                assert all(not r.store.credited_node_ids(event.id) and not event.feedback_applied
                           for event in r.store.list_recall_events())
                assert r.store.count_query_anchors() == 0

    asyncio.run(scenario())
