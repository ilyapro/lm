"""The live loop turns a grounded consumption into a query anchor.

``apply_pending_recall_feedback`` is the one moment where both halves of "this
question was answered by these nodes" exist at once: the consumed event carries
the query, and the grounding verdict that gates credit already says which
results the consuming trace actually used. This module pins what the write does
with that moment, and every assertion here is a rule the graph depends on:

* **Edges only to the grounded subset.** The fixture is the one the credit work
  left behind — eight delivered results, a trace grounded in exactly one — and
  the anchor is required to carry exactly one edge. Anchoring the delivered set
  instead would rebuild, inside the graph, the defect grounding removed from
  credit assignment (87.9% of reinforcements went to unused results), except
  that this time the noise would be *retrieved*.
* **Scope is the consumed event's.** Feedback closes across scopes whenever a
  transport session or a requested-scope match links a recall in one scope to
  an ingest in another; an anchor written under the trace's scope would answer
  a question nobody asked there.
* **No grounded result, no anchor** — including under
  ``LM_RECALL_CREDIT_POLICY=all``, which computes no verdicts at all and
  therefore has no honest target set, and under ``reinforce_results=False``.
* **A repeat reinforces rather than duplicates**, and reinforcement refreshes
  freshness. That last one is load-bearing because of an asymmetry in decay:
  ``decay.soft_delete_expired`` ages nodes off ``last_accessed``, which
  ``MemoryStore.record_access`` refreshes for *delivered* results — and anchors
  are never delivered. If reinforcement did not refresh them, nothing else ever
  would and every anchor would retire on schedule no matter how much it earned.

The anchor write is also required to stay cheap (one batched query vector per
consuming remember, no second grounding pass, no re-embedding of results) and
to be strictly derived: if it fails, the trace, its provenance, and its credit
survive intact.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import living_memory.feedback as feedback_module
from living_memory.feedback import apply_pending_recall_feedback
from living_memory.query_anchors import (
    ANCHOR_EDGE_WEIGHT,
    ANCHOR_TTL_DAYS,
    decay_stale_anchors,
    match_anchors,
)
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore, QueryAnchor, recall_fingerprint

SCOPE = "project:anchors"
QUERY = "kappa ledger"
AMBIENT = {"agent": "agent-a", "task": "anchor-task", "session_id": "anchor-session"}

#: Eight delivered results sharing only the two-token query, so a trace that
#: quotes one node's distinctive vocabulary grounds that node and nothing else.
#: Lifted from tests/test_grounded_credit_assignment.py on purpose: the anchor
#: write reads the same verdict that fixture was built to exhibit.
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

GROUNDED_TRACE = f"{QUERY} follow-up: confirmed that {USED_BODY} and closed it out"
#: Consumes the same event (context identity matches) while grounding nothing:
#: it shares no distinctive vocabulary with any delivered node.
UNGROUNDED_TRACE = "wrote up the offsite agenda and booked the room for thursday"


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
    monkeypatch.delenv("LM_RETRIEVAL_TUNING_POLICY", raising=False)
    monkeypatch.delenv("LM_RECALL_CREDIT_POLICY", raising=False)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _seed(db_path: Path) -> tuple[Any, MemoryStore, list[str], str]:
    """Eight seeded nodes, one recall delivering all eight, nothing consumed."""

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
    return mcp, store, node_ids, recalled["recall_event_id"]


def _consume(mcp: Any, content: str = GROUNDED_TRACE) -> dict[str, Any]:
    return mcp.tools["memory_remember"](content, {"scope": SCOPE, **AMBIENT})


def _anchors(store: MemoryStore, scope: str | None = None) -> list[QueryAnchor]:
    return store.list_query_anchors(scope=scope, include_decayed=True)


def _edge_targets(store: MemoryStore, anchor_id: str) -> dict[str, float]:
    return {
        edge.target_id: edge.weight
        for edge in store.list_query_anchor_edges(anchor_id=anchor_id)
    }


# ---------------------------------------------------------------------------
# The headline: one anchor, edges only to what was used
# ---------------------------------------------------------------------------


def test_grounded_consumption_anchors_the_query_to_the_used_node_only(
    tmp_path: Path,
) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "headline.sqlite3")
    assert _anchors(store) == [], "nothing is anchored before a consumption"

    consumed = _consume(mcp)

    anchors = _anchors(store)
    assert len(anchors) == 1
    anchor = anchors[0]
    assert anchor.query == QUERY
    assert anchor.scope == SCOPE
    assert anchor.fingerprint == recall_fingerprint(QUERY, SCOPE)
    assert anchor.embedding, "an anchor without a vector can never be matched"
    assert not anchor.decayed

    # The whole point: one edge, to the one node the trace actually used.
    targets = _edge_targets(store, anchor.id)
    assert targets == {node_ids[USED_INDEX]: pytest.approx(ANCHOR_EDGE_WEIGHT)}
    unused = set(node_ids) - {node_ids[USED_INDEX]}
    assert unused.isdisjoint(targets), "delivered-but-unused results must not be anchored"
    assert store.count_query_anchor_edges() == 1

    # And the outcome object reports it, so a caller need not re-query.
    feedback = consumed["implicit_feedback"]
    assert feedback["feedback_applied"] is True
    assert set(feedback["linked_node_ids"]) == set(node_ids)
    assert store.get_recall_event(event_id).feedback_trace_id == consumed["node"]["id"]

    trace = store.get_node(consumed["node"]["id"])
    outcome = apply_pending_recall_feedback(store, trace)
    assert outcome.events == [], "the consuming remember already closed the event"


def test_anchor_edge_records_a_hit_and_survives_as_a_match(tmp_path: Path) -> None:
    """The vector lands in the same space the matcher searches."""

    mcp, store, node_ids, _event_id = _seed(tmp_path / "matchable.sqlite3")
    _consume(mcp)
    anchor = _anchors(store)[0]

    embedder = store._resolve_chunk_embedder()
    matches = match_anchors(store, embedder([QUERY])[0], SCOPE)

    assert [match.anchor.id for match in matches] == [anchor.id]
    assert matches[0].similarity == pytest.approx(1.0, abs=1e-6)
    assert matches[0].targets == ((node_ids[USED_INDEX], pytest.approx(ANCHOR_EDGE_WEIGHT)),)

    edges = store.list_query_anchor_edges(anchor_id=anchor.id)
    assert [edge.hits for edge in edges] == [1]


# ---------------------------------------------------------------------------
# No grounding, no anchor
# ---------------------------------------------------------------------------


def test_ungrounded_consumption_writes_no_anchor(tmp_path: Path) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "ungrounded.sqlite3")

    consumed = _consume(mcp, UNGROUNDED_TRACE)

    # The event really was consumed — linkage is exhaustive as ever — and yet
    # nothing was used, so there is nothing for an anchor to point at.
    assert set(consumed["implicit_feedback"]["linked_node_ids"]) == set(node_ids)
    assert store.get_recall_event(event_id).feedback_trace_id == consumed["node"]["id"]
    assert _anchors(store) == []
    assert store.count_query_anchor_edges() == 0

    # Same verdict through the function's own return contract.
    direct = MemoryStore(tmp_path / "ungrounded-direct.sqlite3")
    node = direct.append_trace(f"{QUERY} {USED_BODY}", {"scope": SCOPE})
    direct.record_recall_event(
        query=QUERY,
        scope=SCOPE,
        ambient_context={"session_id": "direct"},
        results=[{"node_id": node.id, "vector_score": 0.9}],
    )
    trace = direct.append_trace(UNGROUNDED_TRACE, {"scope": SCOPE, "session_id": "direct"})
    outcome = apply_pending_recall_feedback(direct, trace)

    assert [event.id for event in outcome.events], "the event must have been consumed"
    assert outcome.grounded_node_ids == []
    assert outcome.anchor_ids == []
    assert _anchors(direct) == []
    direct.close()


def test_policy_all_writes_no_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-grounding fallback must not anchor the full delivered set.

    Under ``all`` every delivered result is reinforced and no grounding verdict
    is computed, so there is no honest edge set to write. Anchoring all eight
    would put the 87.9%-noise rule back into the graph.
    """

    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", "all")
    mcp, store, node_ids, _event_id = _seed(tmp_path / "policy-all.sqlite3")

    before = [store.get_node(node_id).usefulness_score for node_id in node_ids]
    _consume(mcp)
    after = [store.get_node(node_id).usefulness_score for node_id in node_ids]

    # The policy is genuinely active: all eight were reinforced ...
    assert all(new > old for new, old in zip(after, before, strict=True))
    # ... and not one of them was anchored.
    assert _anchors(store) == []
    assert store.count_query_anchor_edges() == 0


def test_teach_path_writes_no_anchor(tmp_path: Path) -> None:
    """``reinforce_results=False`` grades nothing, so it anchors nothing."""

    mcp, store, node_ids, _event_id = _seed(tmp_path / "teach.sqlite3")

    taught = mcp.tools["memory_teach"](
        node_ids[USED_INDEX],
        GROUNDED_TRACE,
        context=dict(AMBIENT, scope=SCOPE),
    )
    corrective = store.get_node(taught["corrective_trace"]["id"])

    assert set(corrective.provenance["recalled_nodes"]) == set(node_ids)
    assert _anchors(store) == []


# ---------------------------------------------------------------------------
# Dedup and freshness
# ---------------------------------------------------------------------------


def test_repeat_of_the_same_query_reinforces_one_anchor(tmp_path: Path) -> None:
    """A second grounded consumption of the same question adds weight, not rows."""

    mcp, store, node_ids, _event_id = _seed(tmp_path / "repeat.sqlite3")
    _consume(mcp)
    first = _anchors(store)[0]

    # Same query, same scope, a differently-worded trace that still uses the
    # same node — a repeated situation, not a duplicate write.
    mcp.tools["memory_recall"](
        QUERY,
        scope=SCOPE,
        max_results=len(BODIES),
        depth=0,
        ambient_context=dict(AMBIENT),
    )
    _consume(mcp, f"second pass on the {QUERY}: {USED_BODY}, verified again today")

    anchors = _anchors(store)
    assert len(anchors) == 1, "a repeat must reinforce, never duplicate"
    again = anchors[0]
    assert again.id == first.id
    assert again.reinforcement_count == first.reinforcement_count + 1
    assert again.last_matched_at >= first.last_matched_at
    assert again.first_seen == first.first_seen

    # Weight accumulates on the repeated edge; hits keep counting the evidence.
    edges = {
        edge.target_id: edge
        for edge in store.list_query_anchor_edges(anchor_id=again.id)
    }
    repeated = edges[node_ids[USED_INDEX]]
    assert repeated.weight == pytest.approx(2 * ANCHOR_EDGE_WEIGHT)
    assert repeated.hits == 2
    # The second consumption may honestly ground the first consuming trace —
    # it quotes the same body and the second recall delivers it — so the
    # anchor may gain an edge. What it may never gain is an edge to one of the
    # seven seeded results no trace has ever used.
    assert (set(node_ids) - {node_ids[USED_INDEX]}).isdisjoint(edges)


def test_reinforcement_refreshes_freshness_against_decay(tmp_path: Path) -> None:
    """An anchor that keeps earning stays alive; an idle one retires.

    Nothing but this write path ever refreshes an anchor: ``record_access``
    fires for delivered results and anchors are never delivered.
    """

    mcp, store, _node_ids, _event_id = _seed(tmp_path / "freshness.sqlite3")
    _consume(mcp)
    anchor = _anchors(store)[0]

    # Age it past the TTL by rewinding its last match, then confirm the sweep
    # really does retire an anchor nothing has reinforced.
    stale = datetime.now(UTC) - timedelta(days=ANCHOR_TTL_DAYS + 1)
    store.reinforce_query_anchor(anchor.id, now=stale.isoformat())
    assert [a.id for a in decay_stale_anchors(store)] == [anchor.id]
    assert store.get_query_anchor(anchor.id).decayed is True

    # Now the same question earns again. The live path revives it and moves its
    # freshness to now, so the next sweep leaves it alone.
    mcp.tools["memory_recall"](
        QUERY,
        scope=SCOPE,
        max_results=len(BODIES),
        depth=0,
        ambient_context=dict(AMBIENT),
    )
    _consume(mcp, f"revisited the {QUERY} today: {USED_BODY}, still the right note")

    revived = store.get_query_anchor(anchor.id)
    assert revived.decayed is False
    assert revived.last_matched_at > stale.isoformat()
    assert decay_stale_anchors(store) == []
    assert store.get_query_anchor(anchor.id).decayed is False


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def test_anchor_lands_in_the_consumed_events_scope(tmp_path: Path) -> None:
    """Feedback crosses scopes; an anchor does not follow it across.

    A recall recorded in ``project:alpha`` for a request that named
    ``project:beta`` is consumable by a ``project:beta`` trace. The question was
    asked of alpha's memory and its answer belongs to alpha.
    """

    event_scope = "project:alpha"
    trace_scope = "project:beta"
    store = MemoryStore(tmp_path / "scope.sqlite3")

    used = store.append_trace(f"{QUERY} {USED_BODY}", {"scope": event_scope})
    other = store.append_trace(f"{QUERY} {BODIES[0]}", {"scope": event_scope})
    store.record_recall_event(
        query=QUERY,
        scope=event_scope,
        requested_scope=trace_scope,
        resolved_scopes=[event_scope, trace_scope],
        ambient_context={"session_id": "cross-scope"},
        results=[
            {"node_id": used.id, "vector_score": 0.9},
            {"node_id": other.id, "vector_score": 0.4},
        ],
    )
    trace = store.append_trace(
        GROUNDED_TRACE, {"scope": trace_scope, "session_id": "cross-scope"}
    )

    outcome = apply_pending_recall_feedback(store, trace)

    assert outcome.grounded_node_ids == [used.id]
    anchors = _anchors(store)
    assert len(anchors) == 1
    assert anchors[0].scope == event_scope
    assert outcome.anchor_ids == [anchors[0].id]
    assert _anchors(store, scope=trace_scope) == []
    assert _edge_targets(store, anchors[0].id) == {used.id: pytest.approx(ANCHOR_EDGE_WEIGHT)}
    store.close()


# ---------------------------------------------------------------------------
# Cost and isolation
# ---------------------------------------------------------------------------


def test_one_batched_query_vector_per_consumption(tmp_path: Path) -> None:
    """The anchor costs one short vector, and the results are not re-embedded."""

    mcp, store, node_ids, _event_id = _seed(tmp_path / "cost.sqlite3")

    batches: list[list[str]] = []
    inner = store._resolve_chunk_embedder()

    def counting(texts: Any) -> Any:
        batches.append([str(text) for text in texts])
        return inner(texts)

    store.set_chunk_embedder(counting)
    _consume(mcp)

    query_batches = [batch for batch in batches if QUERY in batch]
    assert query_batches == [[QUERY]], "exactly one call, carrying only the query"

    node_contents = {store.get_node(node_id).content for node_id in node_ids}
    assert not any(
        text in node_contents for batch in batches for text in batch
    ), "no delivered result may be embedded a second time"


def test_grounding_runs_once_on_the_live_path() -> None:
    """The anchor write reuses the credit verdict; it does not re-ground."""

    import inspect

    source = inspect.getsource(feedback_module)
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert code.count("ground_results(") == 1
    assert "ground_token_sets" not in code
    assert "containment(" not in code


def test_anchor_failure_leaves_the_trace_and_its_credit_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An anchor is derived data; the user's write must outlive its failure."""

    mcp, store, node_ids, event_id = _seed(tmp_path / "failure.sqlite3")
    before = store.get_node(node_ids[USED_INDEX]).usefulness_score

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("anchor tables are on fire")

    monkeypatch.setattr(feedback_module, "upsert_anchor", boom)

    with caplog.at_level(logging.WARNING, logger="living_memory.feedback"):
        consumed = _consume(mcp)

    trace = store.get_node(consumed["node"]["id"])
    assert trace is not None
    assert set(trace.provenance["recalled_nodes"]) == set(node_ids)
    assert store.get_recall_event(event_id).feedback_trace_id == trace.id
    assert store.get_node(node_ids[USED_INDEX]).usefulness_score > before
    assert consumed["implicit_feedback"]["feedback_applied"] is True
    assert _anchors(store) == []
    assert any("query anchor write failed" in record.message for record in caplog.records)
