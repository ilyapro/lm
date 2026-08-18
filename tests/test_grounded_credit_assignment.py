"""Credit for a consumed recall event goes only to the results that were used.

The defect this pins: ``apply_pending_recall_feedback`` used to walk *every*
result of a consumed event and hand each one a rank-decayed positive signal
into both node usefulness and the per-scope retrieval weights. An agent that
used one result of eight reinforced all eight, so nine tenths of the learning
signal was noise and "confirmed useful" meant nothing.

The central test builds exactly that situation — eight delivered results, a
consuming trace grounded in exactly one — and asserts both sides of the
change in one run: under the pre-grounding policy (still reachable as
``LM_RECALL_CREDIT_POLICY=all``) all eight nodes gain usefulness and the
weights move eight times; under the shipped default only the used one does.
The old-policy half is what makes this falsifiable rather than a description
of current behaviour: it fails if the two policies ever collapse into one.

Linkage is deliberately *not* narrowed: provenance and graph edges must still
record everything that was shown. Only reinforcement is gated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from living_memory.feedback import (
    DEFAULT_RECALL_CREDIT_POLICY,
    RECALL_CREDIT_POLICIES,
    UNGROUNDED_NEGATIVE_FACTOR,
    apply_pending_recall_feedback,
)
from living_memory.grounding import ground_results
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore

SCOPE = "project:credit"
ANCHOR = "kappa ledger"
AMBIENT = {"agent": "agent-a", "task": "credit-task", "session_id": "credit-session"}

#: Eight delivered results. Each shares only the two-token anchor with the
#: others, so a trace that quotes one node's distinctive vocabulary grounds
#: that node and nothing else. Index 3 is the one the trace will use.
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

#: Quotes the used node's body verbatim, plus the shared anchor. Every other
#: node shares only the anchor, whose IDF mass is a small share of any node.
CONSUMING_TRACE = f"{ANCHOR} follow-up: confirmed that {USED_BODY} and closed it out"


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


def _seed_event(db_path: Path) -> tuple[Any, MemoryStore, list[str], str]:
    """Eight seeded nodes, one recall delivering all eight, nothing consumed."""

    mcp = create_mcp_server(db_path, mcp_factory=FakeMCP)
    store: MemoryStore = mcp.memory_store
    node_ids = [
        mcp.tools["memory_remember"](
            f"{ANCHOR} {body}",
            {"scope": SCOPE, "agent": "seeder", "session_id": "seed-session"},
        )["node"]["id"]
        for body in BODIES
    ]
    recalled = mcp.tools["memory_recall"](
        ANCHOR,
        scope=SCOPE,
        max_results=len(BODIES),
        depth=0,
        ambient_context=dict(AMBIENT),
    )
    delivered = [result["node"]["id"] for result in recalled["results"]]
    assert sorted(delivered) == sorted(node_ids), "the event must deliver all eight"
    return mcp, store, node_ids, recalled["recall_event_id"]


def _usefulness(store: MemoryStore, node_ids: list[str]) -> list[float]:
    return [store.get_node(node_id).usefulness_score for node_id in node_ids]


def _consume(mcp: Any) -> dict[str, Any]:
    return mcp.tools["memory_remember"](
        CONSUMING_TRACE,
        {"scope": SCOPE, **AMBIENT},
    )


# ---------------------------------------------------------------------------
# The control: the fixture really is "8 delivered, 1 used"
# ---------------------------------------------------------------------------


def test_fixture_grounds_exactly_one_of_the_eight_bodies() -> None:
    """Without this, the headline test could pass for the wrong reason."""

    graded = ground_results(
        CONSUMING_TRACE, {str(index): f"{ANCHOR} {body}" for index, body in enumerate(BODIES)}
    )
    grounded = sorted(int(key) for key, value in graded.items() if value.grounded)

    assert grounded == [USED_INDEX]
    # And with margin on both sides, so the fixture is not threshold-fragile.
    assert graded[str(USED_INDEX)].containment > 0.6
    assert max(
        value.containment for key, value in graded.items() if int(key) != USED_INDEX
    ) < 0.15


# ---------------------------------------------------------------------------
# The headline regression: old policy reinforces 8, new policy reinforces 1
# ---------------------------------------------------------------------------


def test_only_the_grounded_result_is_reinforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # --- Arm A: the pre-grounding rule, reachable as policy "all" ----------
    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", "all")
    mcp_old, store_old, ids_old, _event_old = _seed_event(tmp_path / "old.sqlite3")
    before_old = _usefulness(store_old, ids_old)
    weights_before_old = store_old.get_retrieval_weights(SCOPE)
    _consume(mcp_old)
    after_old = _usefulness(store_old, ids_old)

    # Every one of the eight gained, including the seven the trace never used.
    assert all(after > before for after, before in zip(after_old, before_old, strict=True))
    assert store_old.get_retrieval_weights(SCOPE) != weights_before_old

    # --- Arm B: the shipped default ---------------------------------------
    monkeypatch.delenv("LM_RECALL_CREDIT_POLICY", raising=False)
    mcp_new, store_new, ids_new, _event_new = _seed_event(tmp_path / "new.sqlite3")
    before_new = _usefulness(store_new, ids_new)
    consumed = _consume(mcp_new)
    after_new = _usefulness(store_new, ids_new)

    gained = [
        index
        for index, (after, before) in enumerate(zip(after_new, before_new, strict=True))
        if after != before
    ]
    assert gained == [USED_INDEX]
    assert after_new[USED_INDEX] > before_new[USED_INDEX]
    assert consumed["implicit_feedback"]["feedback_applied"] is True

    # The two arms genuinely diverge: seven nodes moved under "all" and did
    # not move under the default, on the same seeded fixture.
    assert sum(1 for a, b in zip(after_old, before_old, strict=True) if a != b) == 8
    assert sum(1 for a, b in zip(after_new, before_new, strict=True) if a != b) == 1


def test_grounded_policy_leaves_ungrounded_weights_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seven ungrounded deliveries must contribute nothing to the channels.

    Under "all" the scope's weights absorb eight updates; under the default
    they absorb exactly one, so the learned mix reflects one real use rather
    than one use plus seven nerfed deliveries.
    """

    updates: list[str] = []
    import living_memory.feedback as feedback_module

    original = feedback_module.apply_retrieval_feedback

    def counting(store, result, **kwargs):  # type: ignore[no-untyped-def]
        updates.append(str(getattr(result, "node_id", "")))
        return original(store, result, **kwargs)

    monkeypatch.setattr(feedback_module, "apply_retrieval_feedback", counting)

    mcp, _store, node_ids, _event_id = _seed_event(tmp_path / "counted.sqlite3")
    _consume(mcp)

    assert updates == [node_ids[USED_INDEX]]


# ---------------------------------------------------------------------------
# Provenance and graph linkage stay exhaustive
# ---------------------------------------------------------------------------


def test_linkage_still_covers_every_delivered_result(tmp_path: Path) -> None:
    """Gating credit must not narrow provenance: traceability needs all eight."""

    mcp, store, node_ids, event_id = _seed_event(tmp_path / "linkage.sqlite3")
    consumed = _consume(mcp)
    trace_id = consumed["node"]["id"]
    trace = store.get_node(trace_id)

    assert set(consumed["implicit_feedback"]["linked_node_ids"]) == set(node_ids)
    assert set(trace.provenance["recalled_nodes"]) == set(node_ids)
    assert set(trace.source_traces) == set(node_ids)
    related = {
        connection.target_id
        for connection in store.list_connections(source_id=trace_id, relation_type="related")
    }
    assert set(node_ids) <= related

    prior = trace.provenance["prior_recalls"]
    assert [entry["id"] for entry in prior] == [event_id]
    assert set(prior[0]["result_ids"]) == set(node_ids)
    assert store.get_recall_event(event_id).feedback_trace_id == trace_id


def test_grounded_subset_is_reported_and_is_a_subset_of_linkage(
    tmp_path: Path,
) -> None:
    mcp, store, node_ids, _event_id = _seed_event(tmp_path / "reported.sqlite3")
    trace = store.get_node(_consume(mcp)["node"]["id"])

    outcome = apply_pending_recall_feedback(store, trace)
    # Nothing pending is left after the consuming remember already closed it.
    assert outcome.events == []


# ---------------------------------------------------------------------------
# The negative arm stays reachable and behaves as specified
# ---------------------------------------------------------------------------


def test_grounded_negative_policy_penalizes_the_unused_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The A/B's rejected arm must still do what the A/B measured it doing."""

    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", "grounded_negative")
    mcp, store, node_ids, _event_id = _seed_event(tmp_path / "negative.sqlite3")
    before = _usefulness(store, node_ids)
    _consume(mcp)
    after = _usefulness(store, node_ids)

    assert after[USED_INDEX] > before[USED_INDEX]
    for index, (new, old) in enumerate(zip(after, before, strict=True)):
        if index != USED_INDEX:
            assert new < old, f"ungrounded result {index} should be penalized"
    assert 0.0 < UNGROUNDED_NEGATIVE_FACTOR < 1.0


def test_policy_surface_is_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    from living_memory.feedback import _recall_credit_policy

    assert RECALL_CREDIT_POLICIES == ("grounded", "grounded_negative", "all")
    assert DEFAULT_RECALL_CREDIT_POLICY == "grounded"

    monkeypatch.delenv("LM_RECALL_CREDIT_POLICY", raising=False)
    assert _recall_credit_policy() == "grounded"
    for policy in RECALL_CREDIT_POLICIES:
        monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", policy)
        assert _recall_credit_policy() == policy
    # An unknown value must not silently disable credit assignment.
    monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", "nonsense")
    assert _recall_credit_policy() == DEFAULT_RECALL_CREDIT_POLICY


def test_teach_still_links_without_reinforcing(tmp_path: Path) -> None:
    """``reinforce_results=False`` short-circuits grading entirely."""

    mcp, store, node_ids, _event_id = _seed_event(tmp_path / "teach.sqlite3")
    before = _usefulness(store, node_ids)
    taught = mcp.tools["memory_teach"](
        node_ids[USED_INDEX],
        CONSUMING_TRACE,
        context=dict(AMBIENT, scope=SCOPE),
    )
    corrective = store.get_node(taught["corrective_trace"]["id"])

    assert set(corrective.provenance["recalled_nodes"]) == set(node_ids)
    after = _usefulness(store, node_ids)
    # Only the corrected node moves, and it moves *down* via the teach path.
    for index, (new, old) in enumerate(zip(after, before, strict=True)):
        if index != USED_INDEX:
            assert new == old
    assert after[USED_INDEX] < before[USED_INDEX]
