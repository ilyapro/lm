"""Implicit feedback preserves each event's corpus, ranks and source lineage."""

import sqlite3

import pytest

from living_memory import feedback
from living_memory.server import create_mcp_server
from living_memory.storage import recall_fingerprint
from test_mcp_server import FakeMCP


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
    apply = feedback.apply_retrieval_feedback

    def record(store, result, **kwargs):
        signals.append((result.node_id, kwargs["useful"], kwargs["signal"]))
        return apply(store, result, **kwargs)

    monkeypatch.setattr(feedback, "apply_retrieval_feedback", record)
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
