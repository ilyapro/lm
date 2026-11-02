"""Implicit feedback preserves each event's corpus, ranks and source lineage."""

import math
import sqlite3
import weakref
from collections import Counter
from dataclasses import replace

import pytest

from living_memory import feedback, grounding
from living_memory.server import create_mcp_server
from living_memory.storage import recall_fingerprint
from test_anchor_live_path import FakeMCP


CONTEXT = {"scope": "project:source-facts", "task": "routing"}
CONTENT = "cobalt routing uses stable endpoint ownership after verification"
BODIES = [
    "cobalt routing uses stable endpoint ownership",
    "cobalt amber birch delta",
    "amber birch delta",
    "stable endpoint ownership",
]


@pytest.fixture
def sources(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", "grounded")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    nodes = [store.create_node(level="concept" if i == 3 else "trace", content=body,
                               context=CONTEXT) for i, body in enumerate(BODIES)]
    try:
        yield mcp, nodes
    finally:
        mcp.auto_consolidation.close()
        store.close()


def overlapping_events(store, nodes, *, existing):
    a, b, c, d = [node.id for node in nodes]
    deliveries = [[a], [b], ["missing", b, c, b], [d, a, "missing", a], [b, c, a]]
    events = []
    for i, ids in enumerate(deliveries):
        query = f"routing observation {i}"
        events.append(store.record_recall_event(
            query=query, scope=CONTEXT["scope"], ambient_context=CONTEXT,
            results=[{"node_id": node_id, "bm25_score": 1.0} for node_id in ids],
        ))
        if existing:
            store.insert_query_anchor(scope=CONTEXT["scope"], query=query,
                                      fingerprint=recall_fingerprint(query, CONTEXT["scope"]),
                                      embedding=[1.0, 0.0])
    return events


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("policy", ["grounded", "grounded_negative", "all"])
def test_overlapping_events_keep_individual_grounding_ranks_and_credit(sources, monkeypatch, existing, policy):
    mcp, nodes = sources
    store = mcp.memory_store
    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", policy)
    events = overlapping_events(store, nodes, existing=existing)
    a, b, c, d = [node.id for node in nodes]
    signals = []
    grades = []
    apply = feedback.apply_retrieval_feedback
    ground = grounding.ground_token_sets

    def record_grounding(*args, **kwargs):
        graded = ground(*args, **kwargs)
        grades.append(graded)
        return graded

    def record(store, result, **kwargs):
        signals.append((result.node_id, kwargs["useful"], kwargs["signal"]))
        return apply(store, result, **kwargs)

    monkeypatch.setattr(feedback, "apply_retrieval_feedback", record)
    monkeypatch.setattr(grounding, "ground_token_sets", record_grounding)
    response = mcp.tools["memory_remember"](CONTENT, CONTEXT)
    trace = store.get_node(response["node"]["id"])
    assert trace.content == CONTENT and trace.scope == CONTEXT["scope"]
    assert [row["result_ids"] for row in trace.provenance["prior_recalls"]] == [
        [b, c, a], [d, a, a], [b, c, b], [b], [a],
    ]
    assert trace.provenance["recalled_nodes"] == [b, c, a, d]
    assert trace.source_traces == [b, c, a]
    assert all(store.get_recall_event(event.id).feedback_trace_id == trace.id for event in events)
    assert not store.pending_recall_events(scope=trace.scope, context=CONTEXT)
    expected = [{a}, {b}, {b, c}, {d, a}, {b, c, a}] if policy == "all" else [
        {a}, set(), {b}, {d, a}, {a},
    ]
    assert [store.credited_node_ids(event.id) for event in events] == expected
    if policy == "all":
        assert not grades
    else:
        # Same source b, three different event corpora. These hand-computed
        # weights straddle 0.22; caching verdicts or using union IDF is wrong.
        applied = grades[-len(events):]
        assert applied[3][b].containment == pytest.approx(
            math.log(2) / (math.log(2) + 3 * math.log(3))
        )
        assert applied[2][b].containment == pytest.approx(0.25)
        assert applied[0][b].containment == pytest.approx(
            math.log(7 / 3) / (math.log(7 / 3) + 3 * math.log(3))
        )
    assert [signal for signal in signals if signal[0] == a] == [
        (a, True, 1 / 3), (a, True, 0.5), (a, True, 1.0),
    ]
    # Overlapping events accumulate on current statistics; repeated results
    # in one event spend just the first occurrence's credit.
    assert store.get_node(a).usefulness_score > 0.15
    assert len([signal for signal in signals if signal[0] == b and signal[1]]) == (
        3 if policy == "all" else 1
    )
    if policy != "all":
        for event, targets in zip(events, expected):
            anchor = store.find_query_anchor(event.scope, recall_fingerprint(event.query, event.scope))
            if anchor is None:
                assert not targets
            else:
                assert {edge.target_id for edge in store.list_query_anchor_edges(anchor_id=anchor.id)} == targets
    before = [store.get_node(node.id).usefulness_score for node in nodes]
    mcp.tools["memory_remember"](CONTENT + " confirmed", CONTEXT)
    assert [store.get_node(node.id).usefulness_score for node in nodes] == before


@pytest.mark.parametrize("deny_fts", [False, True])
def test_native_repeated_source_feedback_preserves_knowledge_without_fts_writes(sources, monkeypatch, deny_fts):
    mcp, nodes = sources
    store = mcp.memory_store
    source = store.update_node(nodes[0].id, stats={"usefulness_score": 0.55, "access_count": 600})
    events, anchors = [], []
    for i in range(5):
        query = f"routing source {i}"
        events.append(store.record_recall_event(
            query=query, scope=source.scope, ambient_context=CONTEXT,
            results=[{"node_id": source.id, "bm25_score": 1.0}],
        ))
        anchors.append(store.insert_query_anchor(
            scope=source.scope, query=query, fingerprint=recall_fingerprint(query, source.scope),
            embedding=[1.0, 0.0],
        ))
    old_weights = store.get_retrieval_weights(source.scope)
    progress, fts_writes, source_writes = [], [], []
    update = store.update_node

    def observe_update(node_id, **kwargs):
        result = update(node_id, **kwargs)
        if node_id == source.id:
            progress.append(result.usefulness_score)
        return result

    def observe_sql(action, table, column, db, trigger):
        if action == sqlite3.SQLITE_UPDATE and table == "nodes" and column in (
            "content", "content_fingerprint", "context", "task", "level", "scope",
        ):
            source_writes.append(column)
        if trigger == "nodes_fts_update" and action in (
            sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
        ):
            fts_writes.append(table)
            if deny_fts:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    monkeypatch.setattr(store, "update_node", observe_update)
    calls = Counter()
    tokenize = grounding.token_set

    def counted(content):
        calls[content] += 1
        return tokenize(content)

    monkeypatch.setattr(grounding, "token_set", counted)
    store.connection.set_authorizer(observe_sql)
    try:
        response = mcp.tools["memory_remember"](CONTENT, CONTEXT)
    finally:
        store.connection.set_authorizer(None)
    assert fts_writes == source_writes == []
    assert calls == {CONTENT: 1, source.content: 1}
    assert len(progress) == len(events)
    gains = [current - previous for previous, current in zip([source.usefulness_score, *progress], progress)]
    assert all(gain > 0 for gain in gains)
    assert all(later < earlier for earlier, later in zip(gains, gains[1:]))
    current = store.get_node(source.id)
    assert current == replace(source, usefulness_score=progress[-1], updated_at=current.updated_at)
    trace = store.get_node(response["node"]["id"])
    assert trace.content == CONTENT and trace.source_traces == [source.id]
    assert trace.provenance["recalled_nodes"] == [source.id]
    assert {prior["id"] for prior in trace.provenance["prior_recalls"]} == {event.id for event in events}
    assert all(prior["result_ids"] == [source.id] for prior in trace.provenance["prior_recalls"])
    for event, anchor in zip(events, anchors):
        assert store.get_recall_event(event.id).feedback_trace_id == trace.id
        assert store.credited_node_ids(event.id) == {source.id}
        current_anchor = store.find_query_anchor(event.scope, anchor.fingerprint)
        assert current_anchor.id == anchor.id and current_anchor.embedding == anchor.embedding
        assert current_anchor.reinforcement_count == anchor.reinforcement_count + 1
        edge, = store.list_query_anchor_edges(anchor_id=anchor.id)
        assert edge.target_id == source.id and edge.hits == 1
    assert store.get_retrieval_weights(source.scope).bm25 > old_weights.bm25
    assert not store.pending_recall_events(scope=source.scope, context=CONTEXT)


@pytest.mark.parametrize("existing", [False, True])
def test_required_source_failure_leaves_knowledge_and_feedback_unchanged(sources, existing):
    mcp, nodes = sources
    store = mcp.memory_store
    events = overlapping_events(store, nodes, existing=existing)
    with store.transaction():
        store.connection.execute("UPDATE nodes SET content = CAST(x'ff' AS TEXT) WHERE id = ?",
                                 (nodes[0].id,))
    before = store.count_query_anchors()
    with pytest.raises(sqlite3.OperationalError, match="Could not decode to UTF-8"):
        mcp.tools["memory_remember"](CONTENT, CONTEXT)
    assert store.connection.execute("SELECT count(*) FROM nodes").fetchone()[0] == len(nodes)
    assert all(store.get_recall_event(event.id).feedback_trace_id is None for event in events)
    assert all(not store.credited_node_ids(event.id) for event in events)
    assert store.count_query_anchors() == before
    assert not store.list_query_anchor_edges()
    assert not store.connection.in_transaction


def test_self_source_is_not_grounded_or_recorded_in_provenance(sources):
    mcp, nodes = sources
    store = mcp.memory_store
    trace, concept = nodes[0], nodes[3]
    event = store.record_recall_event(
        query="source identity", scope=trace.scope, ambient_context=CONTEXT,
        results=[{"node_id": node_id} for node_id in [trace.id, "missing", concept.id, concept.id]],
    )
    with store.transaction():
        outcome = feedback.apply_pending_recall_feedback(store, trace)
    assert outcome.trace.source_traces == []
    assert outcome.trace.provenance["recalled_nodes"] == [concept.id]
    assert outcome.trace.provenance["prior_recalls"][0]["result_ids"] == [concept.id, concept.id]
    assert store.credited_node_ids(event.id) == {concept.id}
    assert store.get_node(trace.id).usefulness_score == 0.0


@pytest.mark.parametrize("case", ["no_events", "missing", "all", "no_reinforcement"])
def test_ungraded_feedback_never_tokenizes(sources, monkeypatch, case):
    mcp, nodes = sources
    store = mcp.memory_store
    if case != "no_events":
        store.record_recall_event(
            query="routing", scope=CONTEXT["scope"], ambient_context=CONTEXT,
            results=[{"node_id": "missing" if case == "missing" else nodes[0].id}],
        )
    if case == "all":
        monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", "all")

    def forbidden(*args, **kwargs):
        pytest.fail("ungraded feedback must not tokenize")

    monkeypatch.setattr(grounding, "token_set", forbidden)
    if case == "no_reinforcement":
        mcp.tools["memory_teach"](nodes[0].id, "Corrected routing ownership", context=CONTEXT)
    else:
        mcp.tools["memory_remember"](CONTENT, CONTEXT)
    assert not store.pending_recall_events(scope=CONTEXT["scope"], context=CONTEXT)


def test_source_tokens_are_fresh_after_unlocked_preparation_and_next_call(sources, monkeypatch):
    mcp, nodes = sources
    store = mcp.memory_store
    source = store.create_node(level="concept", content=nodes[0].content, context=CONTEXT)
    changed = "unrelated violet garden requires daily watering"
    events = []
    for i in range(3):
        events.append(store.record_recall_event(
            query=f"routing observation {i}", scope=source.scope, ambient_context=CONTEXT,
            results=[{"node_id": source.id}],
        ))
    calls = Counter()
    tokenize = grounding.token_set
    encode = feedback._encode_missing_feedback_queries

    def counted(content):
        calls[content] += 1
        return tokenize(content)

    def change_during_encoding(*args, **kwargs):
        assert not store.connection.in_transaction
        store.update_node(source.id, content=changed)
        return encode(*args, **kwargs)

    monkeypatch.setattr(grounding, "token_set", counted)
    monkeypatch.setattr(feedback, "_encode_missing_feedback_queries", change_during_encoding)
    response = mcp.tools["memory_remember"](CONTENT, CONTEXT)
    assert calls == {CONTENT: 3, source.content: 1, changed: 2}
    assert all(store.get_recall_event(event.id).feedback_trace_id == response["node"]["id"]
               for event in events)
    assert all(not store.credited_node_ids(event.id) for event in events)
    assert store.count_query_anchors() == 0

    # Independent new call, same source identity, newly matching content.
    monkeypatch.setattr(feedback, "_encode_missing_feedback_queries", encode)
    store.update_node(source.id, content=source.content)
    event = store.record_recall_event(
        query="routing next observation", scope=source.scope, ambient_context=CONTEXT,
        results=[{"node_id": source.id}],
    )
    calls.clear()
    mcp.tools["memory_remember"](CONTENT, CONTEXT)
    assert calls == {CONTENT: 3, source.content: 3}
    assert store.credited_node_ids(event.id) == {source.id}


@pytest.mark.parametrize("repeat", [False, True])
def test_tokens_live_only_until_their_last_event(sources, monkeypatch, repeat):
    mcp, nodes = sources
    store = mcp.memory_store
    a, b, c, d = nodes
    groups = [[a, b], [c], [a], [d]] if repeat else [[b], [c], [d]]
    live_sources = [[a, b], [a, c], [a], [d]] if repeat else groups
    for index, group in reversed(list(enumerate(groups))):
        query = f"routing retention {index}"
        store.record_recall_event(
            query=query, scope=CONTEXT["scope"], ambient_context=CONTEXT,
            results=[{"node_id": node.id} for node in group],
        )
        store.insert_query_anchor(scope=CONTEXT["scope"], query=query,
                                  fingerprint=recall_fingerprint(query, CONTEXT["scope"]),
                                  embedding=[1.0, 0.0])
    refs = {}
    observed = []
    tokenize = grounding.token_set
    ground = grounding.ground_token_sets

    def tracked(content):
        assert content not in refs  # One allocation per consumed body here.
        tokens = tokenize(content)
        refs[content] = weakref.ref(tokens)
        return tokens

    def inspect_live(*args, **kwargs):
        observed.append({body for body, ref in refs.items() if ref() is not None})
        return ground(*args, **kwargs)

    monkeypatch.setattr(grounding, "token_set", tracked)
    monkeypatch.setattr(grounding, "ground_token_sets", inspect_live)
    mcp.tools["memory_remember"](CONTENT, CONTEXT)
    assert observed == [{CONTENT, *(node.content for node in group)} for group in live_sources]
    assert all(ref() is None for ref in refs.values())
