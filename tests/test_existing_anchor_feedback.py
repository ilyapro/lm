"""Feedback reuses exact authoritative anchors without query inference."""

import asyncio

from fastmcp import Client
import pytest

from living_memory import feedback
from living_memory.irrelevance import list_query_irrelevance, record_query_irrelevance
from living_memory.query_anchors import resolve_query_anchor
from living_memory.storage import recall_fingerprint
from test_recall_query_preparation import CONTENT, QUERY, SCOPE, VECTOR, call, entry, recall, runtime


@pytest.fixture(autouse=True)
def policies(monkeypatch):
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    monkeypatch.setenv("LM_LOOKUP_CREDIT_POLICY", "on")
    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", "grounded")


def anchor_for(store, query=QUERY, scope=SCOPE):
    return store.find_query_anchor(scope, recall_fingerprint(query, scope))


async def consume(client, route, node_id):
    if route == "lookup":
        return await call(client, "memory_lookup", node_id=node_id)
    return await call(client, "memory_remember", content=CONTENT + " Confirmed for release.",
                      context={"scope": SCOPE}, **({"used": [node_id]} if route == "explicit" else {}))


@pytest.mark.parametrize("route", ["lookup", "explicit", "implicit"])
@pytest.mark.parametrize("existing", [False, True])
def test_native_feedback_encodes_only_missing_identity(runtime, route, existing):
    r = runtime

    async def scenario():
        async with Client(r.mcp) as client:
            prior = await recall(client)
            if existing:
                stored = resolve_query_anchor(r.store, "  cobalt\ncertificate  ", SCOPE, [0.0, 1.0])
                record_query_irrelevance(r.store, stored.id, "earlier", [r.node.id])
            r.encoded.clear()
            await consume(client, route, r.node.id)
            assert r.encoded.count(QUERY) == (0 if existing else 1)
            anchor = anchor_for(r.store)
            assert anchor.reinforcement_count == int(existing)
            assert len(r.store.list_query_anchors(include_decayed=True)) == 1
            if existing:
                assert anchor.id == stored.id and anchor.query == stored.query
                assert anchor.embedding == [0.0, 1.0]
                negative, = list_query_irrelevance(r.store)
                assert negative["weight"] == (0.5 if route == "lookup" else 0.0)
            edge, = r.store.list_query_anchor_edges(anchor_id=anchor.id)
            assert edge.target_id == r.node.id and edge.hits == 1
            assert r.store.credited_node_ids(prior["recall_event_id"]) == {r.node.id}

    asyncio.run(scenario())


def test_native_existing_feedback_and_exact_read_finish_while_peer_encodes(runtime):
    r = runtime

    async def scenario():
        async with Client(r.mcp) as owner, Client(r.mcp) as encoder, Client(r.mcp) as reader:
            await recall(owner)
            await call(owner, "memory_lookup", node_id=r.node.id)
            anchor = anchor_for(r.store)
            prior = await recall(owner)
            record_query_irrelevance(r.store, anchor.id, "earlier", [r.node.id])
            r.encoded.clear()
            r.hold = True
            pending = asyncio.create_task(recall(encoder))
            try:
                assert await asyncio.to_thread(r.entered.wait, 5)
                assert not r.store.connection.in_transaction
                followed = await asyncio.wait_for(call(owner, "memory_lookup", node_id=r.node.id), 3)
                independent = await asyncio.wait_for(call(reader, "memory_lookup", node_id=r.node.id), 3)
                assert entry(followed)["content"] == entry(independent)["content"] == CONTENT
                assert not pending.done() and not r.release.is_set()
                assert r.store.credited_node_ids(prior["recall_event_id"]) == {r.node.id}
                assert anchor_for(r.store).reinforcement_count == anchor.reinforcement_count + 1
                assert list_query_irrelevance(r.store)[0]["weight"] == 0.5
                assert r.encoded == [QUERY]
            finally:
                r.release.set()
            assert entry(await asyncio.wait_for(pending, 5))["content"] == CONTENT

    asyncio.run(scenario())


@pytest.mark.parametrize("route", ["explicit", "implicit"])
@pytest.mark.parametrize("encoder_state", ["unavailable", "failing"])
def test_later_confirmation_without_encoder_cancels_without_reinforcement(runtime, monkeypatch, route, encoder_state):
    r = runtime

    async def scenario():
        async with Client(r.mcp) as client:
            prior = await recall(client)
            await call(client, "memory_lookup", node_id=r.node.id)
            anchor = anchor_for(r.store)
            record_query_irrelevance(r.store, anchor.id, "earlier", [r.node.id])
            before = r.store.get_node(r.node.id).usefulness_score
            if encoder_state == "unavailable":
                monkeypatch.setattr(feedback, "_anchor_embedder", lambda store: None)
            else:
                r.fail = True
            r.encoded.clear()
            await consume(client, route, r.node.id)
            assert QUERY not in r.encoded
            assert anchor_for(r.store) == anchor
            assert r.store.get_node(r.node.id).usefulness_score == before
            assert r.store.list_query_anchor_edges(anchor_id=anchor.id)[0].hits == 1
            negative, = list_query_irrelevance(r.store)
            assert negative["weight"] == 0 and negative["cancels"] == 1
            assert r.store.credited_node_ids(prior["recall_event_id"]) == {r.node.id}

    asyncio.run(scenario())


@pytest.mark.parametrize("preparation", ["ok", "unavailable", "raises", "empty_batch", "empty_vector"])
def test_mixed_events_preserve_existing_effects_when_missing_preparation_fails(runtime, monkeypatch, preparation):
    r = runtime
    stored = resolve_query_anchor(r.store, QUERY, SCOPE, VECTOR)
    record_query_irrelevance(r.store, stored.id, "earlier", [r.node.id])
    events = [r.store.record_recall_event(query=query, scope=SCOPE, results=[{"node_id": r.node.id}])
              for query in ("missing query", QUERY)]
    batches = []

    def encode(queries):
        batches.append(list(queries))
        if preparation == "raises":
            raise RuntimeError("query encoder unavailable")
        return [] if preparation == "empty_batch" else [([] if preparation == "empty_vector" else VECTOR) for _ in queries]

    monkeypatch.setattr(feedback, "_anchor_embedder", lambda store: None if preparation == "unavailable" else encode)
    outcome = feedback._apply_query_feedback(
        r.store, [(event, (r.node.id,), (r.node.id,)) for event in events],
        irrelevant_work=[(events[1], ["separate-negative"])],
    )
    assert batches == ([] if preparation == "unavailable" else [["missing query"]])
    assert stored.id in outcome
    assert anchor_for(r.store).reinforcement_count == 1
    assert len(r.store.list_query_anchors()) == (2 if preparation == "ok" else 1)
    rows = {row["node_id"]: row for row in list_query_irrelevance(r.store)}
    assert rows[r.node.id]["weight"] == 0
    assert rows["separate-negative"]["weight"] == 0.5


def test_exact_identity_deduplicates_missing_vectors_without_semantic_reuse(runtime, monkeypatch):
    r = runtime
    original = resolve_query_anchor(r.store, QUERY, SCOPE, VECTOR)
    requests = [("  cobalt\ncertificate  ", SCOPE), ("Cobalt certificate", SCOPE),
                ("certificate cobalt", SCOPE), ("cobalt certificate if approved", SCOPE),
                ("cobalt certificate not approved", SCOPE), (QUERY, "project:other"),
                ("missing query", SCOPE), (" missing\nquery ", SCOPE)]
    events = [r.store.record_recall_event(query=query, scope=scope, results=[])
              for query, scope in requests]
    batches = []

    def encode(queries):
        batches.append(list(queries))
        return [VECTOR for _ in queries]

    monkeypatch.setattr(feedback, "_anchor_embedder", lambda store: encode)
    feedback._apply_query_feedback(r.store, [], irrelevant_work=[(e, [r.node.id]) for e in events])
    assert batches == [[query for query, _scope in requests[1:-1]]]
    assert len(r.store.list_query_anchors(include_decayed=True)) == 7
    assert anchor_for(r.store).id == original.id
    missing = anchor_for(r.store, "missing query")
    rows = {row["anchor_id"]: row for row in list_query_irrelevance(r.store)}
    assert rows[missing.id]["marks"] == 2


@pytest.mark.parametrize("effect", ["positive", "negative"])
@pytest.mark.parametrize("state", ["decayed", "deleted"])
def test_anchor_lifecycle_uses_existing_revival_and_creation_rules(runtime, effect, state):
    r = runtime
    stored = resolve_query_anchor(r.store, QUERY, SCOPE, VECTOR)
    if state == "decayed":
        r.store.decay_query_anchor(stored.id, "expired")
    else:
        with r.store.transaction():
            r.store.connection.execute("DELETE FROM query_anchors WHERE id = ?", (stored.id,))
    event = r.store.record_recall_event(query=QUERY, scope=SCOPE, results=[])
    r.encoded.clear()
    feedback._apply_query_feedback(
        r.store, [(event, (r.node.id,), ())] if effect == "positive" else [],
        irrelevant_work=[(event, [r.node.id])] if effect == "negative" else [],
    )
    anchor = anchor_for(r.store)
    assert r.encoded == ([QUERY] if state == "deleted" else [])
    assert (anchor.id == stored.id) == (state == "decayed")
    assert anchor.decayed == (state == "decayed" and effect == "negative")
    assert anchor.reinforcement_count == int(state == "decayed" and effect == "positive")
