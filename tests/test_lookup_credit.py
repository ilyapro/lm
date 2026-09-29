"""A same-session ``memory_lookup`` of a delivered node is a usage signal.

Grounding credits a delivered result only when the closing trace quotes it,
and on the 2026-09-07 snapshots that is 19.0% of closures. An id-fetch of a
node that a recall on the *same transport* delivered is the other honest "I
used this" the server can see, and it names a delivered node in 26.2% of
closures with almost no overlap. This module pins how that fetch becomes
credit and, above all, how it is kept from double-counting:

* **Same assignment as a grounded result.** The looked-up node's usefulness
  moves by exactly what a grounded closure of the same delivery would move
  it, the event scope's retrieval weights move by the same per-channel
  shares, and one anchor edge points at the looked-up node only.
* **Nothing else moves.** A node the event did not deliver, a node delivered
  on another transport, a lookup with no transport identity, and a delivery
  older than the window all credit nothing — while the lookup event itself is
  still recorded, because the record and the credit are separate acts.
* **Once per (event, node), whichever signal is first.** The recall credit
  ledger holds one row per credited pair; a grounded closure after a lookup
  finds the row taken and reinforces nothing (linkage stays exhaustive), and a
  lookup after a grounded closure finds the same.
* **Reversible and derived.** ``LM_LOOKUP_CREDIT_POLICY=off`` keeps the record
  and skips the credit; an unknown value is the default. A failed credit is
  logged and the lookup returns as if nothing had been attempted.
* **Additive schema.** The ledger appears on a database master's build
  created, on reopen, with ``schema_version`` still 8 and every pre-existing
  DDL string byte-identical.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
import logging
import sqlite3
import subprocess
import sys
import types

import pytest

import living_memory.feedback as feedback_module
import living_memory.server as server_module
from living_memory.feedback import (
    DEFAULT_LOOKUP_CREDIT_POLICY,
    DEFAULT_LOOKUP_CREDIT_WINDOW_SECONDS,
    LOOKUP_CREDIT_POLICIES,
    _lookup_credit_policy,
    _lookup_credit_window,
    apply_lookup_credit,
)
from living_memory.query_anchors import ANCHOR_EDGE_WEIGHT
from living_memory.server import create_mcp_server
from living_memory.storage import (
    CONSOLIDATION_EMBEDDING_TABLE,
    RECALL_CREDIT_LEDGER_TABLE,
    RECALL_LOOKUP_EVENT_TABLE,
    MemoryStore,
    QueryAnchor,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCOPE = "project:lookup-credit"
QUERY = "kappa ledger"
AMBIENT = {"agent": "agent-a", "task": "credit-task", "session_id": "credit-session"}
TRANSPORT = "agent-connection"
OTHER_TRANSPORT = "other-connection"

#: Eight delivered results sharing only the two-token query, so a trace that
#: quotes one node's distinctive vocabulary grounds that node and nothing
#: else. Lifted from tests/test_grounded_credit_assignment.py on purpose: the
#: lookup credit is required to equal the grounded credit on this fixture.
BODIES = [
    "alembic migration checksum drift blocked the staging rollout entirely",
    "toolbar palette swatches moved into the ColorDock component last week",
    "cert-manager wildcard certificate renewal needs a dns01 solver token",
    "redis eviction storm traced to a runaway zset with unbounded members",
    "grafana dashboard panel queries broke after the datasource uid rename",
    "kafka consumer lag alert fires when the rebalance protocol thrashes",
    "terraform state lock stuck behind an abandoned dynamodb lease record",
    "webpack chunk splitting regressed the vendor bundle size by a third",
]
USED_INDEX = 3
USED_BODY = BODIES[USED_INDEX]
CONSUMING_TRACE = f"{QUERY} follow-up: confirmed that {USED_BODY} and closed it out"


class FakeMCP:
    def __init__(self, name: str, instructions: str | None = None) -> None:
        self.name = name
        self.instructions = instructions
        self.tools: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def tool(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.tools[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        return decorate if func is None else decorate(func)

    def resource(self, uri: str, **_kwargs: Any) -> Any:
        def decorate(func: Any) -> Any:
            self.resources[uri] = func
            return func

        return decorate

    def prompt(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.prompts[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        return decorate if func is None else decorate(func)


@pytest.fixture(autouse=True)
def _hash_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    for name in (
        "LM_RETRIEVAL_TUNING_POLICY",
        "LM_RECALL_CREDIT_POLICY",
        "LM_LOOKUP_CREDIT_POLICY",
        "LM_LOOKUP_CREDIT_WINDOW_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _stamp(monkeypatch: pytest.MonkeyPatch, transport: str | None) -> None:
    """Make every tool call look like it arrived on ``transport``."""

    monkeypatch.setattr(server_module, "_transport_session_id", lambda: transport)


def _seed(
    db_path: Path, monkeypatch: pytest.MonkeyPatch, transport: str | None = TRANSPORT
) -> tuple[Any, MemoryStore, list[str], str]:
    """Eight seeded nodes, one recall on ``transport`` delivering all eight."""

    _stamp(monkeypatch, transport)
    mcp = create_mcp_server(db_path, mcp_factory=FakeMCP)
    store: MemoryStore = mcp.memory_store
    node_ids = [
        mcp.tools["memory_remember"](
            f"{QUERY} {body}",
            {"scope": SCOPE, "agent": "seeder", "session_id": "seed-session"},
        )["node"]["id"]
        for body in BODIES
    ]
    recalled = mcp.tools["memory_recall"](
        QUERY,
        scope=SCOPE,
        max_results=len(BODIES),
        depth=0,
        ambient_context=dict(AMBIENT),
    )
    delivered = [result["node"]["id"] for result in recalled["results"]]
    assert sorted(delivered) == sorted(node_ids), "the event must deliver all eight"
    event = store.get_recall_event(recalled["recall_event_id"])
    assert event is not None
    assert event.transport_session_id == transport
    return mcp, store, node_ids, event.id


def _consume(mcp: Any, content: str = CONSUMING_TRACE) -> dict[str, Any]:
    return mcp.tools["memory_remember"](content, {"scope": SCOPE, **AMBIENT})


def _usefulness(store: MemoryStore, node_ids: list[str]) -> list[float]:
    return [store.get_node(node_id).usefulness_score for node_id in node_ids]


def _anchors(store: MemoryStore, scope: str | None = None) -> list[QueryAnchor]:
    return store.list_query_anchors(scope=scope, include_decayed=True)


def _edges(store: MemoryStore, anchor_id: str) -> dict[str, tuple[float, int]]:
    return {
        edge.target_id: (edge.weight, edge.hits)
        for edge in store.list_query_anchor_edges(anchor_id=anchor_id)
    }


def _ledger(store: MemoryStore) -> list[tuple[str, str, str, str]]:
    return [
        (str(row[0]), str(row[1]), str(row[2]), str(row[3]))
        for row in store.connection.execute(
            f"""
            SELECT recall_event_id, node_id, basis, source_id
            FROM {RECALL_CREDIT_LEDGER_TABLE}
            ORDER BY recall_event_id, node_id
            """
        )
    ]


def _lookup_events(store: MemoryStore) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in store.connection.execute(
            f"""
            SELECT lookup_event_id, node_id, occurred_at, transport_session_id
            FROM {RECALL_LOOKUP_EVENT_TABLE}
            ORDER BY occurred_at, node_id
            """
        )
    ]


def _weights(store: MemoryStore, scope: str = SCOPE) -> tuple[float, float, float]:
    weights = store.get_retrieval_weights(scope)
    return (weights.bm25, weights.vector, weights.graph)


def _snapshot(store: MemoryStore, node_ids: list[str]) -> dict[str, Any]:
    """Everything a credit may move: usefulness, weights, anchors, ledger."""

    return {
        "usefulness": _usefulness(store, node_ids),
        "weights": _weights(store),
        "anchors": [
            (anchor.id, anchor.reinforcement_count, _edges(store, anchor.id))
            for anchor in _anchors(store)
        ],
        "ledger": _ledger(store),
    }


def _rewind_event(store: MemoryStore, event_id: str, *, hours: float) -> None:
    """Set ``recall_events.created_at`` back by ``hours``, in the column's own
    second-precision ``...Z`` format."""

    instant = datetime.now(UTC) - timedelta(hours=hours)
    store.connection.execute(
        "UPDATE recall_events SET created_at = ? WHERE id = ?",
        (instant.strftime("%Y-%m-%dT%H:%M:%SZ"), event_id),
    )
    store.connection.commit()


# ---------------------------------------------------------------------------
# The headline: one lookup, one credit, to the looked-up node only
# ---------------------------------------------------------------------------


def test_lookup_of_a_delivered_node_credits_it_like_a_grounded_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "headline.sqlite3", monkeypatch)
    used = node_ids[USED_INDEX]
    before = _snapshot(store, node_ids)
    assert before["anchors"] == [] and before["ledger"] == []

    fetched = mcp.tools["memory_lookup"](node_id=used)

    # The response contract is untouched.
    assert "error" not in fetched
    assert [entry["id"] for entry in fetched["results"]] == [used]
    assert fetched["missing"] == []

    after = _snapshot(store, node_ids)
    moved = [
        index
        for index, (new, old) in enumerate(
            zip(after["usefulness"], before["usefulness"], strict=True)
        )
        if new != old
    ]
    assert moved == [USED_INDEX], "only the looked-up node may move"
    assert after["usefulness"][USED_INDEX] > before["usefulness"][USED_INDEX]
    assert after["weights"] != before["weights"], "the event scope's weights learned"

    # One anchor for the event's query, in the event's scope, with one edge.
    anchors = _anchors(store)
    assert len(anchors) == 1
    assert anchors[0].query == QUERY
    assert anchors[0].scope == SCOPE
    assert _edges(store, anchors[0].id) == {used: (pytest.approx(ANCHOR_EDGE_WEIGHT), 1)}
    assert store.count_query_anchor_edges() == 1

    # The ledger: one row, basis lookup, sourced from the recorded lookup event.
    events = _lookup_events(store)
    assert [event["node_id"] for event in events] == [used]
    assert events[0]["transport_session_id"] == TRANSPORT
    assert _ledger(store) == [(event_id, used, "lookup", events[0]["lookup_event_id"])]


def test_lookup_credit_equals_grounded_credit_for_the_same_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same fixture, two signals, identical credit.

    Arm A credits the used node through a lookup; arm B through the grounded
    closure the credit-assignment work ships. Both start from the same
    seeded state, so the feedback call — the event's recorded per-channel
    scores for that result, ``useful=True``, ``signal`` at the delivered
    rank, the event's scope — and the usefulness increment must come out
    equal. (The resulting weights are compared as signals, not as stored
    values: the closure's linkage edges give the scope graph evidence before
    the credit lands, which lifts the graph-weight floor — an artifact of
    ordering, not of the credit.)
    """

    calls: list[tuple[int, str, tuple[float, float, float], bool, float, str]] = []
    original = feedback_module.apply_retrieval_feedback

    def recording(store: MemoryStore, result: Any, **kwargs: Any) -> Any:
        calls.append(
            (
                id(store),
                str(result.node_id),
                (result.bm25_score, result.vector_score, result.graph_score),
                bool(kwargs["useful"]),
                float(kwargs["signal"]),
                str(kwargs["scope"]),
            )
        )
        return original(store, result, **kwargs)

    monkeypatch.setattr(feedback_module, "apply_retrieval_feedback", recording)
    mcp_a, store_a, ids_a, _event_a = _seed(tmp_path / "lookup.sqlite3", monkeypatch)
    mcp_b, store_b, ids_b, _event_b = _seed(tmp_path / "grounded.sqlite3", monkeypatch)
    base_a = _usefulness(store_a, ids_a)
    base_b = _usefulness(store_b, ids_b)
    assert base_a == base_b and _weights(store_a) == _weights(store_b)
    assert calls == [], "seeding credits nothing"

    mcp_a.tools["memory_lookup"](node_id=ids_a[USED_INDEX])
    consumed = _consume(mcp_b)
    assert consumed["implicit_feedback"]["feedback_applied"] is True

    def credited(store: MemoryStore, ids: list[str]) -> list[tuple[int, Any, bool, float, str]]:
        return [
            (ids.index(node_id), scores, useful, signal, scope)
            for owner, node_id, scores, useful, signal, scope in calls
            if owner == id(store)
        ]

    lookup_calls = credited(store_a, ids_a)
    grounded_calls = credited(store_b, ids_b)
    rank = store_a.get_recall_event(_event_a).result_ids.index(ids_a[USED_INDEX])
    assert lookup_calls == [
        (USED_INDEX, lookup_calls[0][1], True, pytest.approx(max(0.2, 1.0 / (rank + 1))), SCOPE)
    ]
    assert lookup_calls == grounded_calls

    delta_a = [new - old for new, old in zip(_usefulness(store_a, ids_a), base_a, strict=True)]
    delta_b = [new - old for new, old in zip(_usefulness(store_b, ids_b), base_b, strict=True)]
    assert delta_a[USED_INDEX] > 0
    assert delta_a == pytest.approx(delta_b)
    edges_a = {ids_a.index(k): v for k, v in _edges(store_a, _anchors(store_a)[0].id).items()}
    edges_b = {ids_b.index(k): v for k, v in _edges(store_b, _anchors(store_b)[0].id).items()}
    assert edges_a == edges_b == {USED_INDEX: (pytest.approx(ANCHOR_EDGE_WEIGHT), 1)}
    assert [row[1:3] for row in _ledger(store_a)] == [(ids_a[USED_INDEX], "lookup")]
    assert [row[1:3] for row in _ledger(store_b)] == [(ids_b[USED_INDEX], "grounded")]


def test_one_lookup_of_several_delivered_ids_credits_each_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "several.sqlite3", monkeypatch)
    first, second = node_ids[0], node_ids[5]
    before = _usefulness(store, node_ids)

    fetched = mcp.tools["memory_lookup"](
        node_ids=[first, "01NOPEnotarealnodeid000000", second, first]
    )
    assert [entry["id"] for entry in fetched["results"]] == [first, second]
    assert fetched["missing"] == ["01NOPEnotarealnodeid000000"]

    after = _usefulness(store, node_ids)
    moved = {index for index, (a, b) in enumerate(zip(after, before, strict=True)) if a != b}
    assert moved == {0, 5}
    lookup_id = _lookup_events(store)[0]["lookup_event_id"]
    assert _ledger(store) == sorted(
        [(event_id, first, "lookup", lookup_id), (event_id, second, "lookup", lookup_id)]
    )
    anchors = _anchors(store)
    assert len(anchors) == 1, "one event, one anchor, however many of its results were fetched"
    assert set(_edges(store, anchors[0].id)) == {first, second}

    # The same ids again: the ledger holds, nothing moves a second time.
    mcp.tools["memory_lookup"](node_ids=[first, second])
    assert _usefulness(store, node_ids) == after
    assert len(_ledger(store)) == 2
    assert _edges(store, anchors[0].id) == {
        first: (pytest.approx(ANCHOR_EDGE_WEIGHT), 1),
        second: (pytest.approx(ANCHOR_EDGE_WEIGHT), 1),
    }


def test_lookup_credits_the_newest_delivery_of_the_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, first_event = _seed(tmp_path / "newest.sqlite3", monkeypatch)
    used = node_ids[USED_INDEX]
    second_event = mcp.tools["memory_recall"](
        QUERY, scope=SCOPE, max_results=len(BODIES), depth=0, ambient_context=dict(AMBIENT)
    )["recall_event_id"]
    assert second_event != first_event

    mcp.tools["memory_lookup"](node_id=used)

    assert [row[0] for row in _ledger(store)] == [second_event]
    assert store.credited_node_ids(second_event) == {used}
    assert store.credited_node_ids(first_event) == set()


# ---------------------------------------------------------------------------
# What must never earn credit
# ---------------------------------------------------------------------------


def _assert_recorded_but_uncredited(
    store: MemoryStore,
    node_ids: list[str],
    before: dict[str, Any],
    *,
    lookups: int,
    transport: str | None,
) -> None:
    assert _snapshot(store, node_ids) == before
    events = _lookup_events(store)
    assert len({event["lookup_event_id"] for event in events}) == lookups
    assert {event["transport_session_id"] for event in events} == {transport}


def test_lookup_of_an_undelivered_node_credits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, _event_id = _seed(tmp_path / "undelivered.sqlite3", monkeypatch)
    # Written straight to the store: no recall delivered it, and nothing on
    # the server path ran that could close the pending event.
    stranger = store.append_trace(f"{QUERY} {USED_BODY} restated elsewhere", {"scope": SCOPE})
    watched = [*node_ids, stranger.id]
    before = _snapshot(store, watched)

    fetched = mcp.tools["memory_lookup"](node_id=stranger.id)

    assert [entry["id"] for entry in fetched["results"]] == [stranger.id]
    _assert_recorded_but_uncredited(store, watched, before, lookups=1, transport=TRANSPORT)


def test_lookup_of_a_node_delivered_on_another_transport_credits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, _event_id = _seed(tmp_path / "other-transport.sqlite3", monkeypatch)
    before = _snapshot(store, node_ids)

    _stamp(monkeypatch, OTHER_TRANSPORT)
    mcp.tools["memory_lookup"](node_id=node_ids[USED_INDEX])

    _assert_recorded_but_uncredited(store, node_ids, before, lookups=1, transport=OTHER_TRANSPORT)


def test_lookup_without_transport_identity_credits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, _event_id = _seed(tmp_path / "unstamped.sqlite3", monkeypatch)
    before = _snapshot(store, node_ids)

    _stamp(monkeypatch, None)
    mcp.tools["memory_lookup"](node_id=node_ids[USED_INDEX])

    _assert_recorded_but_uncredited(store, node_ids, before, lookups=1, transport=None)


def test_delivery_on_an_unstamped_recall_credits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No transport on the *recall* side means no join either."""

    mcp, store, node_ids, _event_id = _seed(tmp_path / "unstamped-recall.sqlite3", monkeypatch, None)
    before = _snapshot(store, node_ids)

    _stamp(monkeypatch, TRANSPORT)
    mcp.tools["memory_lookup"](node_id=node_ids[USED_INDEX])

    _assert_recorded_but_uncredited(store, node_ids, before, lookups=1, transport=TRANSPORT)


def test_lookup_after_the_window_credits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "window.sqlite3", monkeypatch)
    used = node_ids[USED_INDEX]
    before = _snapshot(store, node_ids)

    _rewind_event(store, event_id, hours=25)
    mcp.tools["memory_lookup"](node_id=used)
    _assert_recorded_but_uncredited(store, node_ids, before, lookups=1, transport=TRANSPORT)

    # Positive control on the same rows: inside the window, the same lookup
    # credits, so the miss above was the window and not something else.
    _rewind_event(store, event_id, hours=23)
    mcp.tools["memory_lookup"](node_id=used)
    assert _usefulness(store, node_ids)[USED_INDEX] > before["usefulness"][USED_INDEX]
    assert [row[:3] for row in _ledger(store)] == [(event_id, used, "lookup")]


def test_window_is_env_tunable_and_falls_back_on_garbage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _lookup_credit_window() == timedelta(seconds=DEFAULT_LOOKUP_CREDIT_WINDOW_SECONDS)
    monkeypatch.setenv("LM_LOOKUP_CREDIT_WINDOW_SECONDS", "3600")
    assert _lookup_credit_window() == timedelta(hours=1)
    for garbage in ("soon", "-5", "0", "1.5", ""):
        monkeypatch.setenv("LM_LOOKUP_CREDIT_WINDOW_SECONDS", garbage)
        assert _lookup_credit_window() == timedelta(seconds=DEFAULT_LOOKUP_CREDIT_WINDOW_SECONDS)

    mcp, store, node_ids, event_id = _seed(tmp_path / "short-window.sqlite3", monkeypatch)
    used = node_ids[USED_INDEX]
    before = _snapshot(store, node_ids)
    _rewind_event(store, event_id, hours=2)

    monkeypatch.setenv("LM_LOOKUP_CREDIT_WINDOW_SECONDS", "3600")
    mcp.tools["memory_lookup"](node_id=used)
    _assert_recorded_but_uncredited(store, node_ids, before, lookups=1, transport=TRANSPORT)

    monkeypatch.setenv("LM_LOOKUP_CREDIT_WINDOW_SECONDS", "soon")
    mcp.tools["memory_lookup"](node_id=used)
    assert _usefulness(store, node_ids)[USED_INDEX] > before["usefulness"][USED_INDEX]


def test_a_delivery_after_the_lookup_is_not_a_delivery_it_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Order matters: the event must precede the fetch, at second precision."""

    mcp, store, node_ids, event_id = _seed(tmp_path / "future.sqlite3", monkeypatch)
    used = node_ids[USED_INDEX]
    before = _snapshot(store, node_ids)

    _rewind_event(store, event_id, hours=-1)
    mcp.tools["memory_lookup"](node_id=used)
    _assert_recorded_but_uncredited(store, node_ids, before, lookups=1, transport=TRANSPORT)


# ---------------------------------------------------------------------------
# Once per (event, node), whichever signal comes first
# ---------------------------------------------------------------------------


def test_grounded_closure_after_a_lookup_does_not_credit_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "lookup-then-close.sqlite3", monkeypatch)
    used = node_ids[USED_INDEX]

    mcp.tools["memory_lookup"](node_id=used)
    after_lookup = _snapshot(store, node_ids)
    lookup_id = _lookup_events(store)[0]["lookup_event_id"]
    assert after_lookup["ledger"] == [(event_id, used, "lookup", lookup_id)]
    assert len(after_lookup["anchors"]) == 1

    consumed = _consume(mcp)
    trace_id = consumed["node"]["id"]

    # The closure happened and its linkage is exhaustive, exactly as before.
    assert consumed["implicit_feedback"]["recall_event_ids"] == [event_id]
    assert set(consumed["implicit_feedback"]["linked_node_ids"]) == set(node_ids)
    trace = store.get_node(trace_id)
    assert set(trace.provenance["recalled_nodes"]) == set(node_ids)
    assert set(trace.source_traces) == set(node_ids)
    assert [entry["id"] for entry in trace.provenance["prior_recalls"]] == [event_id]
    related = {
        connection.target_id
        for connection in store.list_connections(source_id=trace_id, relation_type="related")
    }
    assert set(node_ids) <= related
    assert store.get_recall_event(event_id).feedback_trace_id == trace_id

    # But the one node it grounded was already credited by the lookup: no
    # second usefulness move, no second weight update, no second anchor edge
    # and no second ledger row. The closure reinforced nothing at all.
    assert consumed["implicit_feedback"]["feedback_applied"] is False
    assert _snapshot(store, node_ids) == after_lookup
    assert _edges(store, _anchors(store)[0].id) == {used: (pytest.approx(ANCHOR_EDGE_WEIGHT), 1)}


def test_lookup_after_a_grounded_closure_credits_only_what_the_closure_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "close-then-lookup.sqlite3", monkeypatch)
    used, unused = node_ids[USED_INDEX], node_ids[0]

    consumed = _consume(mcp)
    trace_id = consumed["node"]["id"]
    assert consumed["implicit_feedback"]["feedback_applied"] is True
    after_closure = _snapshot(store, node_ids)
    assert after_closure["ledger"] == [(event_id, used, "grounded", trace_id)]
    anchor = _anchors(store)[0]

    # The grounded node again: its credit is spent, the lookup is recorded.
    mcp.tools["memory_lookup"](node_id=used)
    assert _snapshot(store, node_ids) == after_closure
    assert len(_lookup_events(store)) == 1

    # A delivered node the trace did not quote: the lookup is the first usage
    # signal for it, and it earns the credit the closure could not give it —
    # the union the whole feature exists for.
    mcp.tools["memory_lookup"](node_id=unused)
    assert _usefulness(store, node_ids)[0] > after_closure["usefulness"][0]
    assert _usefulness(store, node_ids)[USED_INDEX] == after_closure["usefulness"][USED_INDEX]
    lookup_id = [event for event in _lookup_events(store) if event["node_id"] == unused][0][
        "lookup_event_id"
    ]
    assert _ledger(store) == sorted(
        [(event_id, used, "grounded", trace_id), (event_id, unused, "lookup", lookup_id)]
    )
    anchors = _anchors(store)
    assert [a.id for a in anchors] == [anchor.id], "same question, same anchor"
    assert anchors[0].reinforcement_count == anchor.reinforcement_count + 1
    assert _edges(store, anchor.id) == {
        used: (pytest.approx(ANCHOR_EDGE_WEIGHT), 1),
        unused: (pytest.approx(ANCHOR_EDGE_WEIGHT), 1),
    }


def test_grounded_negative_does_not_blame_a_looked_up_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ledger says the node was used; the closure must not call it unused."""

    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", "grounded_negative")
    mcp, store, node_ids, _event_id = _seed(tmp_path / "negative.sqlite3", monkeypatch)
    looked_up = 0
    mcp.tools["memory_lookup"](node_id=node_ids[looked_up])
    after_lookup = _usefulness(store, node_ids)

    _consume(mcp)

    after = _usefulness(store, node_ids)
    assert after[looked_up] == after_lookup[looked_up], "looked-up: neither blamed nor re-credited"
    assert after[USED_INDEX] > after_lookup[USED_INDEX]
    for index, (new, old) in enumerate(zip(after, after_lookup, strict=True)):
        if index not in (looked_up, USED_INDEX):
            assert new < old, f"ungrounded result {index} should be penalized"


# ---------------------------------------------------------------------------
# The env switch
# ---------------------------------------------------------------------------


def test_policy_off_records_the_lookup_and_credits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_LOOKUP_CREDIT_POLICY", "off")
    mcp, store, node_ids, event_id = _seed(tmp_path / "off.sqlite3", monkeypatch)
    used = node_ids[USED_INDEX]
    before = _snapshot(store, node_ids)

    fetched = mcp.tools["memory_lookup"](node_id=used)
    assert [entry["id"] for entry in fetched["results"]] == [used]
    _assert_recorded_but_uncredited(store, node_ids, before, lookups=1, transport=TRANSPORT)

    # An unknown value is the default, never a silent off.
    monkeypatch.setenv("LM_LOOKUP_CREDIT_POLICY", "nonsense")
    mcp.tools["memory_lookup"](node_id=used)
    assert _usefulness(store, node_ids)[USED_INDEX] > before["usefulness"][USED_INDEX]
    assert [row[:3] for row in _ledger(store)] == [(event_id, used, "lookup")]
    assert len({event["lookup_event_id"] for event in _lookup_events(store)}) == 2


def test_lookup_policy_surface_is_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    assert LOOKUP_CREDIT_POLICIES == ("delivered", "off")
    assert DEFAULT_LOOKUP_CREDIT_POLICY == "delivered"
    assert _lookup_credit_policy() == "delivered"
    for policy in LOOKUP_CREDIT_POLICIES:
        monkeypatch.setenv("LM_LOOKUP_CREDIT_POLICY", policy.upper())
        assert _lookup_credit_policy() == policy
    monkeypatch.setenv("LM_LOOKUP_CREDIT_POLICY", "nonsense")
    assert _lookup_credit_policy() == DEFAULT_LOOKUP_CREDIT_POLICY


def test_apply_lookup_credit_reads_nothing_when_off_or_unstamped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with MemoryStore(tmp_path / "noread.sqlite3") as store:
        statements: list[str] = []
        store.connection.set_trace_callback(statements.append)
        try:
            monkeypatch.setenv("LM_LOOKUP_CREDIT_POLICY", "off")
            outcome = apply_lookup_credit(store, "LOOKUP", ["X"], TRANSPORT, datetime.now(UTC))
            assert (outcome.lookup_event_id, outcome.credited, outcome.anchor_ids) == ("LOOKUP", [], [])
            monkeypatch.delenv("LM_LOOKUP_CREDIT_POLICY")
            for transport in (None, ""):
                outcome = apply_lookup_credit(store, "LOOKUP", ["X"], transport, datetime.now(UTC))
                assert outcome.credited == []
            outcome = apply_lookup_credit(store, "LOOKUP", ["", "  "], TRANSPORT, datetime.now(UTC))
            assert outcome.credited == []
        finally:
            store.connection.set_trace_callback(None)
        assert statements == []


# ---------------------------------------------------------------------------
# Derived: a failed credit never fails the lookup
# ---------------------------------------------------------------------------


def test_failed_credit_leaves_the_lookup_and_its_record_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "failure.sqlite3", monkeypatch)
    used = node_ids[USED_INDEX]
    before = _usefulness(store, node_ids)

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("retrieval weights are on fire")

    monkeypatch.setattr(feedback_module, "apply_retrieval_feedback", boom)
    with caplog.at_level(logging.WARNING, logger="living_memory.feedback"):
        fetched = mcp.tools["memory_lookup"](node_id=used)

    assert "error" not in fetched
    assert [entry["id"] for entry in fetched["results"]] == [used]
    events = _lookup_events(store)
    assert [event["node_id"] for event in events] == [used]
    assert _usefulness(store, node_ids) == before
    assert _anchors(store) == []
    assert any("lookup credit failed" in record.message for record in caplog.records)
    # Claimed before applied, as the attestation ledger does: the pair reads
    # as spent, which under-credits rather than ever crediting twice.
    assert _ledger(store) == [(event_id, used, "lookup", events[0]["lookup_event_id"])]


# ---------------------------------------------------------------------------
# The ledger itself
# ---------------------------------------------------------------------------


def test_claim_is_first_writer_wins_and_reads_back(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "ledger.sqlite3") as store:
        assert store.credited_node_ids("E1") == set()
        assert store.claim_recall_credit("E1", "N1", basis="lookup", source_id="L1") is True
        assert store.claim_recall_credit("E1", "N1", basis="grounded", source_id="T1") is False
        assert store.claim_recall_credit("E1", "N2", basis="grounded", source_id="T1") is True
        assert store.claim_recall_credit("E2", "N1", basis="lookup", source_id="L2") is True
        assert store.credited_node_ids("E1") == {"N1", "N2"}
        assert store.credited_node_ids("E2") == {"N1"}
        assert _ledger(store) == [
            ("E1", "N1", "lookup", "L1"),
            ("E1", "N2", "grounded", "T1"),
            ("E2", "N1", "lookup", "L2"),
        ]
        with pytest.raises(ValueError, match="basis"):
            store.claim_recall_credit("E1", "N3", basis="attested", source_id="A1")
        with pytest.raises(ValueError, match="instant"):
            store.claim_recall_credit(
                "E1", "N3", basis="lookup", source_id="L1", credited_at="not a time"
            )
        assert store.credited_node_ids("E1") == {"N1", "N2"}

        # Committed, not merely staged: a second connection sees the rows.
        other = sqlite3.connect(tmp_path / "ledger.sqlite3")
        try:
            assert other.execute(f"SELECT COUNT(*) FROM {RECALL_CREDIT_LEDGER_TABLE}").fetchone()[0] == 3
        finally:
            other.close()


def test_recall_events_for_transport_is_newest_first_bounded_and_scoped_to_the_stamp(
    tmp_path: Path,
) -> None:
    with MemoryStore(tmp_path / "transport.sqlite3") as store:
        assert store.recall_events_for_transport(None) == []
        assert store.recall_events_for_transport("") == []
        ids = [
            store.record_recall_event(
                query=f"q{index}",
                scope=SCOPE,
                ambient_context={"transport_session_id": TRANSPORT},
            ).id
            for index in range(3)
        ]
        stranger = store.record_recall_event(
            query="elsewhere", scope=SCOPE, ambient_context={"transport_session_id": OTHER_TRANSPORT}
        ).id
        unstamped = store.record_recall_event(query="nowhere", scope=SCOPE).id

        listed = [event.id for event in store.recall_events_for_transport(TRANSPORT)]
        assert listed == list(reversed(ids))
        assert stranger not in listed and unstamped not in listed
        assert [event.id for event in store.recall_events_for_transport(TRANSPORT, limit=1)] == [ids[-1]]
        assert store.recall_events_for_transport(TRANSPORT, limit=0) == []
        plan = " ".join(
            str(row[3])
            for row in store.connection.execute(
                """
                EXPLAIN QUERY PLAN SELECT * FROM recall_events
                WHERE transport_session_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?
                """,
                (TRANSPORT, 200),
            )
        )
        assert "idx_recall_events_transport_created" in plan


# ---------------------------------------------------------------------------
# Additive schema: the ledger appears on reopen, nothing else changes
# ---------------------------------------------------------------------------

_MASTER_STORAGE_CACHE: list[types.ModuleType] = []


def _master_storage() -> types.ModuleType:
    """master's ``storage.py``, executed as a module against current siblings.

    Same helper as tests/test_transcript_ledger.py: the comparison is against
    the code actually deployed, not a copy that could drift.
    """

    if _MASTER_STORAGE_CACHE:
        return _MASTER_STORAGE_CACHE[0]
    try:
        source = subprocess.run(
            ["git", "show", "master:src/living_memory/storage.py"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover
        pytest.skip(f"master's storage.py unavailable via git: {error}")
    module = types.ModuleType("living_memory._storage_at_master_for_lookup_credit")
    module.__file__ = str(REPO_ROOT / "src" / "living_memory" / "storage.py")
    sys.modules[module.__name__] = module
    exec(compile(source, "master:src/living_memory/storage.py", "exec"), module.__dict__)
    _MASTER_STORAGE_CACHE.append(module)
    return module


def _schema_objects(db_path: Path) -> dict[str, str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL").fetchall()
    finally:
        conn.close()
    return {str(name): str(sql) for name, sql in rows}


def _schema_version(db_path: Path) -> str:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT value FROM metadata WHERE key = 'schema_version'").fetchone()
    finally:
        conn.close()
    return str(row[0])


def test_ledger_is_created_on_a_master_built_database_on_reopen(tmp_path: Path) -> None:
    master = _master_storage()
    db = tmp_path / "existing.sqlite3"
    with master.MemoryStore(db) as store:
        node = store.append_trace("ledger fixture trace body", {"scope": SCOPE, **AMBIENT})
        event = store.record_recall_event(
            query="ledger fixture",
            scope=SCOPE,
            ambient_context=dict(AMBIENT),
            results=[{"node_id": node.id, "rank": 0, "bm25_score": 1.0, "vector_score": 0.5}],
        )
    before = _schema_objects(db)
    assert _schema_version(db) == "8"

    with MemoryStore(db) as reopened:
        assert reopened.get_node(node.id) is not None
        assert reopened.get_recall_event(event.id) is not None
        # Usable at once: the whole migration was the open.
        assert reopened.claim_recall_credit(event.id, node.id, basis="grounded", source_id="T") is True
        assert reopened.claim_recall_credit(event.id, node.id, basis="lookup", source_id="L") is False
        assert reopened.credited_node_ids(event.id) == {node.id}

    after = _schema_objects(db)
    for name, sql in before.items():
        assert name in after, f"{name} disappeared"
        assert after[name] == sql, f"{name} DDL changed:\n{sql!r}\n->\n{after[name]!r}"
    # Subset, not equality: once this lands on master, master's build creates
    # the ledger too and the difference legitimately collapses to nothing.
    # The explicit-feedback tables (recall_explicit_credit, the mark audit and
    # its index) and the consolidation vector cache are created the same
    # additive way on the same open.
    assert set(after) - set(before) <= {
        RECALL_CREDIT_LEDGER_TABLE,
        "recall_explicit_credit",
        "recall_feedback_marks",
        "idx_recall_feedback_marks_event",
        CONSOLIDATION_EMBEDDING_TABLE,
    }
    assert RECALL_CREDIT_LEDGER_TABLE in after
    assert _schema_version(db) == "8"

    # Idempotent: reopening again neither duplicates nor drops the table.
    with MemoryStore(db) as again:
        assert again.credited_node_ids(event.id) == {node.id}
    assert _schema_objects(db) == after


def test_dropped_ledger_reappears_on_reopen(tmp_path: Path) -> None:
    db = tmp_path / "dropped.sqlite3"
    with MemoryStore(db) as store:
        store.connection.execute(f"DROP TABLE {RECALL_CREDIT_LEDGER_TABLE}")
        store.connection.commit()
    assert RECALL_CREDIT_LEDGER_TABLE not in _schema_objects(db)
    with MemoryStore(db) as store:
        assert store.claim_recall_credit("E", "N", basis="lookup", source_id="L") is True
    assert RECALL_CREDIT_LEDGER_TABLE in _schema_objects(db)
