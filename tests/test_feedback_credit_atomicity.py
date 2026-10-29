"""Optional pair credit commits atomically at ordinary handler boundaries."""

import sqlite3

import pytest

import living_memory.feedback as feedback
import living_memory.server as server
from living_memory.irrelevance import list_query_irrelevance
from test_explicit_feedback import AMBIENT, SCOPE, UNRELATED_TRACE, _seed


@pytest.fixture
def recalled(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    monkeypatch.setenv("LM_EXPLICIT_CREDIT_WEIGHT", "1")
    monkeypatch.setenv("LM_LOOKUP_CREDIT_POLICY", "delivered")
    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", "grounded")
    monkeypatch.setenv("LM_RETRIEVAL_TUNING_POLICY", "fixed")
    monkeypatch.setattr(server, "_decay_sweep_if_due", lambda store: None)
    mcp, store, ids, event_id = _seed(tmp_path / "feedback.sqlite3", monkeypatch)
    correction_target = store.append_trace(UNRELATED_TRACE, {"scope": SCOPE}).id
    outcomes = []
    for name in ("apply_explicit_marks", "apply_lookup_credit"):
        original = getattr(server, name)

        def capture(*args, _original=original, **kwargs):
            outcome = _original(*args, **kwargs)
            outcomes.append(outcome)
            return outcome

        monkeypatch.setattr(server, name, capture)

    def call(handler, used, irrelevant=None):
        context = {"scope": SCOPE, **AMBIENT}
        if handler == "lookup":
            response = mcp.tools["memory_lookup"](node_ids=used)
            assert {entry["id"] for entry in response["results"]} == set(used)
        elif handler == "recall":
            # An empty query carries marks without redelivering their targets.
            response = mcp.tools["memory_recall"](
                "", scope=SCOPE, ambient_context=AMBIENT, used=used, irrelevant=irrelevant,
            )
            assert response["results"] == []
        elif handler == "teach":
            response = mcp.tools["memory_teach"](
                correction_target, UNRELATED_TRACE + " revised", context=context,
                used=used, irrelevant=irrelevant,
            )
            assert store.get_node(response["corrective_trace"]["id"]) is not None
        else:
            response = mcp.tools["memory_remember"](
                UNRELATED_TRACE, context=context, used=used, irrelevant=irrelevant,
            )
            assert store.get_node(response["node"]["id"]) is not None
        if handler != "lookup":
            assert response["feedback_marks"] == {
                "accepted": len(used) + len(irrelevant or []), "dropped": 0, "by_reason": {},
            }
        assert not store.connection.in_transaction
        return outcomes[-1]

    yield mcp, store, ids, event_id, call
    mcp.auto_consolidation.close()
    store.close()


def numeric_state(store, ids):
    # Independent reader observes durable numeric state, including every weight.
    with sqlite3.connect(store.db_path) as reader:
        scores = [reader.execute(
            "SELECT usefulness_score FROM nodes WHERE id = ?", (node_id,),
        ).fetchone()[0] for node_id in ids]
        weights = reader.execute(
            "SELECT scope, bm25, vector, graph, learning_rate FROM retrieval_weights ORDER BY scope"
        ).fetchall()
    return scores, weights


def inject_numeric_failure(monkeypatch, store, stage):
    def fail_before(*args, **kwargs):
        raise RuntimeError("numeric feedback fault")

    if stage == "before":
        monkeypatch.setattr(feedback, "apply_retrieval_feedback", fail_before)
    else:
        update = store.update_retrieval_weights

        def fail_weights(*args, **kwargs):
            if stage == "after_weights":
                update(*args, **kwargs)
            raise RuntimeError("numeric feedback fault")

        monkeypatch.setattr(store, "update_retrieval_weights", fail_weights)


@pytest.mark.parametrize("handler", ["remember", "recall", "teach", "lookup"])
@pytest.mark.parametrize("stage", ["before", "after_node", "after_weights"])
def test_failed_numeric_effect_can_be_applied_once_on_an_eligible_repeat(
    recalled, monkeypatch, handler, stage,
):
    _mcp, store, ids, event_id, call = recalled
    target = ids[3]
    before = numeric_state(store, ids)
    with monkeypatch.context() as fault:
        inject_numeric_failure(fault, store, stage)
        outcome = call(handler, [target])

    assert outcome.credited == [] and outcome.anchor_ids == []
    assert store.credited_node_ids(event_id) == set()
    assert numeric_state(store, ids) == before
    if handler != "lookup":
        assert [tuple(row) for row in store.connection.execute(
            "SELECT recall_event_id, node_id, accepted FROM recall_feedback_marks"
        )] == [(event_id, target, 1)]
    else:
        assert store.connection.execute("SELECT COUNT(*) FROM recall_lookup_events").fetchone()[0] == 1

    outcome = call(handler, [target])
    assert outcome.credited == [(event_id, target)]
    assert store.credited_node_ids(event_id) == {target}
    after = numeric_state(store, ids)
    assert after[0][3] > before[0][3] and after[1] != before[1]
    assert call(handler, [target]).credited == []
    assert numeric_state(store, ids) == after


@pytest.mark.parametrize("handler", ["remember", "recall", "teach", "lookup"])
def test_successful_numeric_effect_is_applied_once(recalled, handler):
    _mcp, store, ids, event_id, call = recalled
    before = numeric_state(store, ids)
    assert call(handler, ids[:1]).credited == [(event_id, ids[0])]
    after = numeric_state(store, ids)
    assert after[0][0] > before[0][0] and after[1] != before[1]
    assert call(handler, ids[:1]).credited == []
    assert numeric_state(store, ids) == after


@pytest.mark.parametrize("first,second", [("remember", "lookup"), ("lookup", "remember")])
def test_failed_pair_remains_available_to_another_credit_basis(recalled, monkeypatch, first, second):
    _mcp, store, ids, event_id, call = recalled
    with monkeypatch.context() as fault:
        inject_numeric_failure(fault, store, "after_weights")
        assert call(first, ids[:1]).credited == []
    assert call(second, ids[:1]).credited == [(event_id, ids[0])]
    after = numeric_state(store, ids)
    assert call(first, ids[:1]).credited == []
    assert numeric_state(store, ids) == after


def test_lookup_cannot_spend_credit_without_its_event(recalled, monkeypatch):
    _mcp, store, ids, event_id, call = recalled
    before = numeric_state(store, ids)
    with monkeypatch.context() as fault:
        fault.setattr(store, "get_recall_event_header", lambda event_id: None)
        assert call("lookup", ids[:1]).credited == []
    assert store.credited_node_ids(event_id) == set()
    assert numeric_state(store, ids) == before
    assert call("lookup", ids[:1]).credited == [(event_id, ids[0])]


@pytest.mark.parametrize("handler", ["remember", "lookup"])
def test_failed_pair_preserves_other_pairs_and_their_outcomes(recalled, monkeypatch, handler):
    _mcp, store, ids, event_id, call = recalled
    before = numeric_state(store, ids)
    apply = feedback.apply_retrieval_feedback

    def fail_middle(store, result, **kwargs):
        update = apply(store, result, **kwargs)
        if result.node_id == ids[1]:
            raise RuntimeError("middle pair fault after numeric writes")
        return update

    with monkeypatch.context() as fault:
        fault.setattr(feedback, "apply_retrieval_feedback", fail_middle)
        outcome = call(handler, ids[:3])
    assert outcome.credited == [(event_id, ids[0]), (event_id, ids[2])]
    assert store.credited_node_ids(event_id) == {ids[0], ids[2]}
    scores = numeric_state(store, ids)[0]
    assert scores[0] > before[0][0] and scores[2] > before[0][2]
    assert scores[1] == before[0][1]
    anchor, = store.list_query_anchors(scope=SCOPE)
    assert {edge.target_id for edge in store.list_query_anchor_edges(anchor_id=anchor.id)} == {ids[0], ids[2]}

    assert call(handler, ids[:3]).credited == [(event_id, ids[1])]
    after = numeric_state(store, ids)
    assert after[0][0] == scores[0] and after[0][2] == scores[2]
    assert after[0][1] > scores[1]
    assert call(handler, ids[:3]).credited == []
    assert numeric_state(store, ids) == after


@pytest.mark.parametrize("handler", ["remember", "lookup"])
def test_optional_anchor_failure_keeps_numeric_credit(recalled, monkeypatch, handler):
    _mcp, store, ids, event_id, call = recalled
    before = numeric_state(store, ids)

    def fail(*args, **kwargs):
        raise RuntimeError("anchor fault")

    with monkeypatch.context() as fault:
        fault.setattr(store, "upsert_query_anchor_edge", fail)
        outcome = call(handler, ids[:1])
    assert outcome.credited == [(event_id, ids[0])] and outcome.anchor_ids == []
    assert store.credited_node_ids(event_id) == {ids[0]}
    after = numeric_state(store, ids)
    assert after[0][0] > before[0][0] and after[1] != before[1]
    assert call(handler, ids[:1]).credited == []
    assert numeric_state(store, ids) == after


def test_numeric_failure_preserves_cancellation_and_independent_negative_mark(recalled, monkeypatch):
    _mcp, store, ids, event_id, call = recalled
    call("remember", [], irrelevant=ids[:1])
    before = numeric_state(store, ids)
    with monkeypatch.context() as fault:
        inject_numeric_failure(fault, store, "after_weights")
        outcome = call("remember", ids[:1], irrelevant=ids[1:2])
    assert outcome.credited == []
    assert store.credited_node_ids(event_id) == set()
    assert numeric_state(store, ids) == before
    rows = {row["node_id"]: row for row in list_query_irrelevance(store)}
    assert rows[ids[0]]["weight"] == 0 and rows[ids[0]]["cancels"] == 1
    assert rows[ids[1]]["weight"] == 0.5
    assert call("remember", ids[:1]).credited == [(event_id, ids[0])]


def test_optional_cancellation_failure_keeps_numeric_credit(recalled, monkeypatch):
    _mcp, store, ids, event_id, call = recalled
    call("remember", [], irrelevant=ids[:1])

    def fail(*args, **kwargs):
        raise RuntimeError("cancellation fault")

    with monkeypatch.context() as fault:
        fault.setattr(feedback, "cancel_query_irrelevance", fail)
        outcome = call("remember", ids[:1], irrelevant=ids[1:2])
    assert outcome.credited == [(event_id, ids[0])]
    after = numeric_state(store, ids)
    rows = {row["node_id"]: row for row in list_query_irrelevance(store)}
    assert rows[ids[0]]["weight"] == rows[ids[1]]["weight"] == 0.5
    assert call("remember", ids[:1]).credited == []
    assert numeric_state(store, ids) == after
    rows = {row["node_id"]: row for row in list_query_irrelevance(store)}
    assert rows[ids[0]]["weight"] == 0 and rows[ids[0]]["cancels"] == 1


@pytest.mark.parametrize("policy", ["all", "grounded_negative"])
@pytest.mark.parametrize("stage", ["after_node", "after_weights"])
def test_required_implicit_numeric_failure_rolls_back_the_knowledge_write(
    recalled, monkeypatch, policy, stage,
):
    mcp, store, ids, event_id, _call = recalled
    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", policy)
    with sqlite3.connect(store.db_path) as reader:
        before = list(reader.iterdump())
    with monkeypatch.context() as fault:
        inject_numeric_failure(fault, store, stage)
        with pytest.raises(RuntimeError, match="numeric feedback fault"):
            mcp.tools["memory_remember"](UNRELATED_TRACE, {"scope": SCOPE, **AMBIENT})
    with sqlite3.connect(store.db_path) as reader:
        assert list(reader.iterdump()) == before
    assert not store.connection.in_transaction
    assert store.credited_node_ids(event_id) == set()
    response = mcp.tools["memory_remember"](UNRELATED_TRACE, {"scope": SCOPE, **AMBIENT})
    assert response["implicit_feedback"]["feedback_applied"] is True
    scores, _ = numeric_state(store, ids)
    assert all(score > 0 if policy == "all" else score < 0 for score in scores)
    assert store.credited_node_ids(event_id) == (set(ids) if policy == "all" else set())
