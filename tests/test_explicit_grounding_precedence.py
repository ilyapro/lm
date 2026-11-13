"""Accepted query feedback outranks lexical overlap for its own delivery."""

import sqlite3

import pytest

from living_memory import feedback
from living_memory.irrelevance import list_query_irrelevance
from test_explicit_feedback import (
    AMBIENT, BODIES, CONSUMING_TRACE, OTHER_INDEX, QUERY, SCOPE, UNRELATED_TRACE,
    USED_INDEX, _clean_env, _ledger, _marks, _recall, _remember, _seed, _stamp,
)


REJECTED_QUOTE = (
    f'Query note: The recalled claim "{BODIES[USED_INDEX]}" was unrelated '
    'to the question. The correct fix is quartz sundial pigment.'
)


@pytest.fixture
def recalled(tmp_path, monkeypatch):
    _clean_env.__wrapped__(monkeypatch)
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    monkeypatch.setenv("LM_EXPOSE_ATTEST", "1")
    mcp, store, ids, event = _seed(tmp_path / "memory.sqlite3", monkeypatch)
    try:
        yield mcp, store, ids, event
    finally:
        mcp.auto_consolidation.close()
        store.close()


def assert_closed(store, response, event, ids):
    trace = store.get_node(response["node"]["id"])
    assert store.get_recall_event(event).feedback_trace_id == trace.id
    assert set(ids) <= set(trace.provenance["recalled_nodes"])
    assert set(ids) <= set(trace.source_traces)
    assert event in {item["id"] for item in trace.provenance["prior_recalls"]}


def negative(store, target):
    return next(row for row in list_query_irrelevance(store) if row["node_id"] == target)


@pytest.mark.parametrize("policy", ["credit", "audit", "off"])
@pytest.mark.parametrize("content,mark", [
    (REJECTED_QUOTE, True), (UNRELATED_TRACE, True), (CONSUMING_TRACE, False),
])
def test_registered_remember_respects_explicit_feedback(recalled, monkeypatch, policy, content, mark):
    mcp, store, ids, event = recalled
    target = ids[USED_INDEX]
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", policy)
    response = _remember(mcp, content, irrelevant=[target] if mark else [])
    blocked = mark and policy == "credit"
    credited = content != UNRELATED_TRACE and not blocked
    assert response["feedback_marks"]["accepted"] == int(mark and policy != "off")
    assert (store.get_node(target).usefulness_score > 0) is credited
    assert _ledger(store) == ([(event, target, "grounded")] if credited else [])
    edges = [edge.target_id for anchor in store.list_query_anchors()
             for edge in store.list_query_anchor_edges(anchor_id=anchor.id)]
    assert edges == ([target] if credited else [])
    if blocked:
        row = negative(store, target)
        assert (row["weight"], row["marks"], row["cancels"]) == (0.5, 1, 0)
    else:
        assert list_query_irrelevance(store) == []
    assert_closed(store, response, event, ids)


@pytest.mark.parametrize("credit_policy", ["grounded", "grounded_negative", "all"])
def test_prior_recall_mark_survives_closure_and_reconnect(recalled, monkeypatch, credit_policy):
    mcp, store, ids, event = recalled
    target = ids[USED_INDEX]
    response = mcp.tools["memory_recall"](
        "", scope=SCOPE, ambient_context=AMBIENT, irrelevant=[target],
    )
    assert response["feedback_marks"]["accepted"] == 1
    assert _marks(store)[-1]["recall_event_id"] == event
    _stamp(monkeypatch, "reconnected-transport")
    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", credit_policy)
    response = _remember(mcp, REJECTED_QUOTE)
    assert store.get_node(target).usefulness_score == 0
    assert target not in store.credited_node_ids(event)
    assert negative(store, target)["cancels"] == 0
    assert_closed(store, response, event, ids)


@pytest.mark.parametrize("confirmation", ["new_event", "used"])
def test_independent_confirmation_can_cancel_negative(recalled, confirmation):
    mcp, store, ids, event = recalled
    target = ids[USED_INDEX]
    _remember(mcp, REJECTED_QUOTE, irrelevant=[target])
    assert negative(store, target)["weight"] == 0.5
    if confirmation == "new_event":
        event = _recall(mcp)["recall_event_id"]
        response = _remember(mcp, CONSUMING_TRACE)
    else:
        response = _remember(mcp, CONSUMING_TRACE, used=[target])
        assert response["feedback_marks"]["accepted"] == 1
    assert store.get_node(target).usefulness_score > 0
    assert target in store.credited_node_ids(event)
    assert (negative(store, target)["weight"], negative(store, target)["cancels"]) == (0, 1)
    before = store.get_node(target).usefulness_score
    _remember(mcp, CONSUMING_TRACE, used=[target])
    assert store.get_node(target).usefulness_score == before
    assert negative(store, target)["cancels"] == 1


def test_event_query_and_node_isolation(recalled):
    mcp, store, ids, old_event = recalled
    target, other = ids[USED_INDEX], ids[OTHER_INDEX]
    new_event = mcp.tools["memory_recall"](
        "redis eviction", scope=SCOPE, ambient_context=AMBIENT, max_results=8,
    )["recall_event_id"]
    response = _remember(mcp, REJECTED_QUOTE + " " + BODIES[OTHER_INDEX], irrelevant=[target])
    assert _marks(store)[-1]["recall_event_id"] == new_event
    assert (old_event, target, "grounded") in _ledger(store)
    assert (new_event, target, "grounded") not in _ledger(store)
    assert (old_event, other, "grounded") in _ledger(store)
    row = negative(store, target)
    assert row["last_event_id"] == new_event and row["weight"] == 0.5 and row["cancels"] == 0
    anchors = {anchor.query: anchor for anchor in store.list_query_anchors()}
    assert target in {edge.target_id for edge in store.list_query_anchor_edges(anchor_id=anchors[QUERY].id)}
    assert target not in {edge.target_id for edge in store.list_query_anchor_edges(anchor_id=anchors["redis eviction"].id)}
    assert_closed(store, response, old_event, ids)


@pytest.mark.parametrize("policy", ["credit", "audit", "off"])
def test_registered_attestation_obeys_same_precedence_and_keeps_receipt(recalled, monkeypatch, policy):
    mcp, store, ids, event = recalled
    target = ids[USED_INDEX]
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", policy)
    closure = _remember(mcp, UNRELATED_TRACE, irrelevant=[target])
    result = mcp.tools["memory_attest"](event, [REJECTED_QUOTE])
    assert result["grounded_node_ids"] == ([] if policy == "credit" else [target])
    assert result["credited"] is (policy != "credit")
    assert result["feedback_trace_id"] == closure["node"]["id"]
    entry = next(item for item in result["results"] if item["node_id"] == target)
    assert entry["containment"] > result["min_containment"]
    assert entry["grounded"] is (policy != "credit")
    before = store.get_node(target).usefulness_score
    assert mcp.tools["memory_attest"](event, [REJECTED_QUOTE]) == dict(result, replay=True)
    assert store.get_node(target).usefulness_score == before
    if policy == "credit":
        assert negative(store, target)["weight"] == 0.5


@pytest.mark.parametrize("failure", ["negative_write", "audit_write"])
def test_optional_feedback_failure_preserves_knowledge(recalled, monkeypatch, failure):
    mcp, store, ids, event = recalled
    target = ids[USED_INDEX]

    def fail(*args, **kwargs):
        raise RuntimeError("optional feedback failure")

    if failure == "negative_write":
        monkeypatch.setattr(feedback, "record_query_irrelevance", fail)
    else:
        monkeypatch.setattr(store, "record_feedback_marks", fail)
    response = _remember(mcp, REJECTED_QUOTE, irrelevant=[target])
    assert_closed(store, response, event, ids)
    assert list_query_irrelevance(store) == []
    assert (store.get_node(target).usefulness_score > 0) is (failure == "audit_write")


@pytest.mark.parametrize("failure", ["precedence_read", "closure"])
def test_required_failure_rolls_back_knowledge_marks_and_credit(recalled, monkeypatch, failure):
    mcp, store, ids, event = recalled
    with sqlite3.connect(store.db_path) as reader:
        before = list(reader.iterdump())

    def fail(*args, **kwargs):
        raise RuntimeError("required feedback failure")

    with monkeypatch.context() as fault:
        fault.setattr(store, "irrelevant_marked_node_ids" if failure == "precedence_read"
                      else "mark_recall_event_feedback", fail)
        with pytest.raises(RuntimeError, match="required feedback failure"):
            _remember(mcp, REJECTED_QUOTE + " " + BODIES[OTHER_INDEX], irrelevant=[ids[USED_INDEX]])
    with sqlite3.connect(store.db_path) as reader:
        assert list(reader.iterdump()) == before
    assert not store.connection.in_transaction
    response = _remember(mcp, REJECTED_QUOTE, irrelevant=[ids[USED_INDEX]])
    assert_closed(store, response, event, ids)
    assert negative(store, ids[USED_INDEX])["weight"] == 0.5
