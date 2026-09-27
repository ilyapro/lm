"""Explicit ``used`` / ``irrelevant`` recall feedback (goal explicit-recall-feedback).

What this module pins, in the order docs/explicit-feedback.md states it:

* **Optional fields on three tools.** memory_recall, memory_remember and
  memory_teach accept ``used``/``irrelevant``; a call without them answers
  exactly as before (no ``feedback_marks`` key, no audit row).
* **Only this session's deliveries.** A marked id is accepted only when a
  recall event on the same transport delivered it (newest such event within
  the lookup-credit window). Everything else is dropped with a reason and
  still audited with ``accepted = 0``.
* **Valve.** ``audit`` (default) records and claims nothing; ``credit``
  claims basis ``explicit`` once per (event, node) across grounded, lookup
  and explicit, in either order, and hands irrelevant marks to the
  query-irrelevance hook; ``off`` records nothing.
* **Link hygiene.** No ``related`` edge from the closing trace to a node
  marked irrelevant for that event — same call or later call — under every
  valve value but ``off``; ``LM_IMPLICIT_LINK_POLICY=credited`` links only
  credited results.
* **Prompt arms.** Both description arms fit the 1024-char budget and keep
  the recall law.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import sqlite3

import pytest

import living_memory.feedback as feedback_module
import living_memory.server as server_module
from living_memory.feedback import (
    DEFAULT_EXPLICIT_FEEDBACK_POLICY,
    DEFAULT_IMPLICIT_LINK_POLICY,
    _explicit_credit_weight,
    _explicit_feedback_policy,
    _implicit_link_policy,
)
from living_memory.server import (
    _IRRELEVANT_FIELD_DESCRIPTION,
    _IRRELEVANT_FIELD_DESCRIPTION_MANDATORY,
    _RECALL_DESCRIPTION,
    _RECALL_DESCRIPTION_MANDATORY,
    _USED_FIELD_DESCRIPTION,
    _USED_FIELD_DESCRIPTION_MANDATORY,
    _explicit_feedback_prompt_arm,
    _explicit_feedback_texts,
    create_mcp_server,
)
from living_memory.storage import (
    RECALL_CREDIT_LEDGER_TABLE,
    RECALL_EXPLICIT_CREDIT_TABLE,
    RECALL_FEEDBACK_MARKS_TABLE,
    MemoryStore,
)

SCOPE = "project:explicit-feedback"
QUERY = "kappa ledger"
AMBIENT = {"agent": "agent-a", "task": "marks-task", "session_id": "marks-session"}
TRANSPORT = "agent-connection"
OTHER_TRANSPORT = "other-connection"

#: The grounded-credit fixture of tests/test_lookup_credit.py: eight results
#: sharing only the query, so a trace quoting one body grounds that one only.
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
OTHER_INDEX = 5
CONSUMING_TRACE = f"{QUERY} follow-up: confirmed that {BODIES[USED_INDEX]} and closed it out"
#: Shares no content token with any body: grounds nothing.
UNRELATED_TRACE = "quartz sundial pigment observation unrelated to anything seeded"


class FakeMCP:
    def __init__(self, name: str, instructions: str | None = None) -> None:
        self.name = name
        self.instructions = instructions
        self.tools: dict[str, Any] = {}
        self.descriptions: dict[str, str | None] = {}
        self.resources: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def tool(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            name = str(kwargs.get("name") or inner.__name__)
            self.tools[name] = inner
            self.descriptions[name] = kwargs.get("description")
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
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    for name in (
        "LM_RETRIEVAL_TUNING_POLICY",
        "LM_RECALL_CREDIT_POLICY",
        "LM_LOOKUP_CREDIT_POLICY",
        "LM_LOOKUP_CREDIT_WINDOW_SECONDS",
        "LM_EXPLICIT_FEEDBACK_POLICY",
        "LM_EXPLICIT_CREDIT_WEIGHT",
        "LM_IMPLICIT_LINK_POLICY",
        "LM_EXPLICIT_FEEDBACK_PROMPT",
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stamp(monkeypatch: pytest.MonkeyPatch, transport: str | None) -> None:
    monkeypatch.setattr(server_module, "_transport_session_id", lambda: transport)


def _seed(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, MemoryStore, list[str], str]:
    """Eight nodes, one recall on TRANSPORT delivering all eight."""

    _stamp(monkeypatch, TRANSPORT)
    mcp = create_mcp_server(db_path, mcp_factory=FakeMCP)
    store: MemoryStore = mcp.memory_store
    node_ids = [
        mcp.tools["memory_remember"](
            f"{QUERY} {body}",
            {"scope": SCOPE, "agent": "seeder", "session_id": "seed-session"},
        )["node"]["id"]
        for body in BODIES
    ]
    recalled = _recall(mcp)
    assert sorted(result["node"]["id"] for result in recalled["results"]) == sorted(node_ids)
    return mcp, store, node_ids, recalled["recall_event_id"]


def _recall(mcp: Any, **kwargs: Any) -> dict[str, Any]:
    return mcp.tools["memory_recall"](
        QUERY,
        scope=SCOPE,
        max_results=len(BODIES),
        depth=0,
        ambient_context=dict(AMBIENT),
        **kwargs,
    )


def _remember(mcp: Any, content: str = CONSUMING_TRACE, **kwargs: Any) -> dict[str, Any]:
    return mcp.tools["memory_remember"](content, {"scope": SCOPE, **AMBIENT}, **kwargs)


def _marks(store: MemoryStore) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in store.connection.execute(
            f"SELECT * FROM {RECALL_FEEDBACK_MARKS_TABLE} ORDER BY id"
        )
    ]


def _ledger(store: MemoryStore) -> list[tuple[str, str, str]]:
    return [
        (str(row[0]), str(row[1]), str(row[2]))
        for row in store.connection.execute(
            f"SELECT recall_event_id, node_id, basis FROM {RECALL_CREDIT_LEDGER_TABLE} "
            "ORDER BY recall_event_id, node_id"
        )
    ]


def _explicit(store: MemoryStore) -> list[tuple[str, str, str]]:
    return [
        (str(row[0]), str(row[1]), str(row[2]))
        for row in store.connection.execute(
            f"SELECT recall_event_id, node_id, source_id FROM {RECALL_EXPLICIT_CREDIT_TABLE} "
            "ORDER BY recall_event_id, node_id"
        )
    ]


def _usefulness(store: MemoryStore, node_ids: list[str]) -> list[float]:
    return [store.get_node(node_id).usefulness_score for node_id in node_ids]


def _weights(store: MemoryStore) -> tuple[float, float, float]:
    weights = store.get_retrieval_weights(SCOPE)
    return (weights.bm25, weights.vector, weights.graph)


def _implicit_targets(store: MemoryStore, trace_id: str, event_id: str) -> set[str]:
    return {
        connection.target_id
        for connection in store.list_connections(source_id=trace_id, relation_type="related")
        if connection.metadata.get("basis") == "implicit_recall_feedback"
        and connection.metadata.get("recall_event_id") == event_id
    }


def _rank(store: MemoryStore, event_id: str, node_id: str) -> int:
    event = store.get_recall_event(event_id)
    assert event is not None
    return event.result_ids.index(node_id)


# ---------------------------------------------------------------------------
# Valves: defaults and fallbacks
# ---------------------------------------------------------------------------


def test_valve_defaults_and_unknown_values_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    assert DEFAULT_EXPLICIT_FEEDBACK_POLICY == "audit"
    assert DEFAULT_IMPLICIT_LINK_POLICY == "all"
    assert _explicit_feedback_policy() == "audit"
    assert _implicit_link_policy() == "all"
    assert _explicit_credit_weight() == 1.0
    assert _explicit_feedback_prompt_arm() == "optional"
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "bogus")
    monkeypatch.setenv("LM_IMPLICIT_LINK_POLICY", "bogus")
    monkeypatch.setenv("LM_EXPLICIT_CREDIT_WEIGHT", "-3")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_PROMPT", "bogus")
    assert _explicit_feedback_policy() == "audit"
    assert _implicit_link_policy() == "all"
    assert _explicit_credit_weight() == 1.0
    assert _explicit_feedback_prompt_arm() == "optional"
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", " CREDIT ")
    monkeypatch.setenv("LM_IMPLICIT_LINK_POLICY", "credited")
    monkeypatch.setenv("LM_EXPLICIT_CREDIT_WEIGHT", "0.5")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_PROMPT", "mandatory")
    assert _explicit_feedback_policy() == "credit"
    assert _implicit_link_policy() == "credited"
    assert _explicit_credit_weight() == 0.5
    assert _explicit_feedback_prompt_arm() == "mandatory"
    monkeypatch.setenv("LM_EXPLICIT_CREDIT_WEIGHT", "nan")
    assert _explicit_credit_weight() == 1.0


# ---------------------------------------------------------------------------
# Field semantics on each tool
# ---------------------------------------------------------------------------


def test_calls_without_fields_are_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mcp, store, node_ids, _event_id = _seed(tmp_path / "compat.sqlite3", monkeypatch)
    recalled = _recall(mcp)
    remembered = _remember(mcp)
    taught = mcp.tools["memory_teach"](node_ids[0], "corrected fact about kappa ledger")
    for response in (recalled, remembered, taught):
        assert "feedback_marks" not in response
    assert _marks(store) == []


def test_recall_carried_marks_are_audited_against_earlier_deliveries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "recall.sqlite3", monkeypatch)
    used, off_topic = node_ids[USED_INDEX], node_ids[OTHER_INDEX]
    before = (_usefulness(store, node_ids), _weights(store))

    response = _recall(mcp, used=[used], irrelevant=[off_topic])

    assert response["feedback_marks"] == {"accepted": 2, "dropped": 0, "by_reason": {}}
    rows = _marks(store)
    assert [(row["node_id"], row["mark"]) for row in rows] == [
        (used, "used"),
        (off_topic, "irrelevant"),
    ]
    for row in rows:
        assert row["accepted"] == 1
        assert row["reject_reason"] is None
        assert row["recall_event_id"] == event_id
        assert row["via_tool"] == "memory_recall"
        # The carrier is the new recall event, not the delivering one.
        assert row["source_id"] == response["recall_event_id"] != event_id
        assert row["transport_session_id"] == TRANSPORT
        assert row["agent"] == AMBIENT["agent"]
        assert row["rank"] == _rank(store, event_id, row["node_id"])
        assert row["marked_at"]
    # Default valve is audit: nothing reinforced, nothing claimed.
    assert _ledger(store) == [] and _explicit(store) == []
    assert (_usefulness(store, node_ids), _weights(store)) == before


def test_recall_marks_cannot_name_the_recall_carrying_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, _event_id = _seed(tmp_path / "self.sqlite3", monkeypatch)
    # A fresh transport has no earlier delivery; this very recall delivers
    # the node, but a mark only ever refers to an earlier one.
    _stamp(monkeypatch, "fresh-connection")
    response = _recall(mcp, used=[node_ids[0]])
    assert response["feedback_marks"] == {
        "accepted": 0,
        "dropped": 1,
        "by_reason": {"not_delivered": 1},
    }


def test_remember_marks_use_the_trace_as_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "remember.sqlite3", monkeypatch)
    response = _remember(mcp, UNRELATED_TRACE, used=[node_ids[0]], irrelevant=[node_ids[1]])
    assert response["feedback_marks"]["accepted"] == 2
    rows = _marks(store)
    assert {row["source_id"] for row in rows} == {response["node"]["id"]}
    assert {row["via_tool"] for row in rows} == {"memory_remember"}
    assert {row["recall_event_id"] for row in rows} == {event_id}


def test_teach_marks_use_the_corrective_trace_as_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "teach.sqlite3", monkeypatch)
    taught = mcp.tools["memory_teach"](
        node_ids[2],
        "the certificate renewal needs an http01 solver, not dns01",
        context={"scope": SCOPE, **AMBIENT},
        used=[node_ids[2]],
        irrelevant=[node_ids[4]],
    )
    assert taught["feedback_marks"] == {"accepted": 2, "dropped": 0, "by_reason": {}}
    rows = _marks(store)
    assert {row["source_id"] for row in rows} == {taught["corrective_trace"]["id"]}
    assert {row["via_tool"] for row in rows} == {"memory_teach"}
    assert {row["recall_event_id"] for row in rows} == {event_id}


def test_teach_that_fails_on_its_arguments_records_no_marks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, _event_id = _seed(tmp_path / "teach-fail.sqlite3", monkeypatch)
    with pytest.raises(KeyError):
        mcp.tools["memory_teach"]("01NOSUCHNODE", "fix", used=[node_ids[0]])
    with pytest.raises(ValueError):
        mcp.tools["memory_teach"](node_ids[0], "   ", used=[node_ids[0]])
    assert _marks(store) == []


# ---------------------------------------------------------------------------
# Dropping: foreign session, undelivered, no transport, malformed
# ---------------------------------------------------------------------------


def test_foreign_session_and_unknown_ids_are_dropped_but_audited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, _event_id = _seed(tmp_path / "foreign.sqlite3", monkeypatch)
    _stamp(monkeypatch, OTHER_TRANSPORT)
    response = _remember(mcp, UNRELATED_TRACE, used=[node_ids[0], "01UNKNOWNNODE"])
    assert response["feedback_marks"] == {
        "accepted": 0,
        "dropped": 2,
        "by_reason": {"not_delivered": 2},
    }
    rows = _marks(store)
    assert [(row["node_id"], row["accepted"], row["reject_reason"]) for row in rows] == [
        (node_ids[0], 0, "not_delivered"),
        ("01UNKNOWNNODE", 0, "not_delivered"),
    ]
    assert {row["recall_event_id"] for row in rows} == {""}
    assert {row["transport_session_id"] for row in rows} == {OTHER_TRANSPORT}
    assert all(row["rank"] is None for row in rows)


def test_no_transport_empty_duplicate_and_conflict_reasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "reasons.sqlite3", monkeypatch)
    response = _remember(
        mcp,
        UNRELATED_TRACE,
        used=[node_ids[0], node_ids[0], "  ", node_ids[1]],
        irrelevant=[node_ids[1], node_ids[2]],
    )
    assert response["feedback_marks"] == {
        "accepted": 2,
        "dropped": 4,
        "by_reason": {"duplicate": 1, "empty": 1, "conflict": 2},
    }
    accepted = {(row["node_id"], row["mark"]) for row in _marks(store) if row["accepted"]}
    assert accepted == {(node_ids[0], "used"), (node_ids[2], "irrelevant")}

    _stamp(monkeypatch, None)
    response = _remember(mcp, UNRELATED_TRACE + " again", used=[node_ids[3]])
    assert response["feedback_marks"] == {
        "accepted": 0,
        "dropped": 1,
        "by_reason": {"no_transport": 1},
    }
    last = _marks(store)[-1]
    assert (last["recall_event_id"], last["transport_session_id"]) == ("", None)


def test_delivery_outside_the_window_is_not_delivered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "window.sqlite3", monkeypatch)
    monkeypatch.setenv("LM_LOOKUP_CREDIT_WINDOW_SECONDS", "60")
    store.connection.execute(
        "UPDATE recall_events SET created_at = '2020-01-01T00:00:00Z' WHERE id = ?", (event_id,)
    )
    store.connection.commit()
    response = _remember(mcp, UNRELATED_TRACE, used=[node_ids[0]])
    assert response["feedback_marks"]["by_reason"] == {"not_delivered": 1}


def test_mark_resolves_to_the_newest_delivering_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, first_event = _seed(tmp_path / "newest.sqlite3", monkeypatch)
    second_event = _recall(mcp)["recall_event_id"]
    _remember(mcp, UNRELATED_TRACE, used=[node_ids[0]])
    assert _marks(store)[0]["recall_event_id"] == second_event != first_event


# ---------------------------------------------------------------------------
# Credit: valve and once-per-pair dedup across grounded | lookup | explicit
# ---------------------------------------------------------------------------


def test_store_level_claim_is_once_per_pair_across_both_tables(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "claims.sqlite3") as store:
        assert store.claim_recall_credit("E1", "N1", basis="explicit", source_id="mark:1")
        assert not store.claim_recall_credit("E1", "N1", basis="grounded", source_id="T")
        assert not store.claim_recall_credit("E1", "N1", basis="lookup", source_id="L")
        assert not store.claim_recall_credit("E1", "N1", basis="explicit", source_id="mark:2")
        assert store.claim_recall_credit("E1", "N2", basis="lookup", source_id="L")
        assert not store.claim_recall_credit("E1", "N2", basis="explicit", source_id="mark:3")
        assert store.credited_node_ids("E1") == {"N1", "N2"}
        assert _ledger(store) == [("E1", "N2", "lookup")]
        assert _explicit(store) == [("E1", "N1", "mark:1")]


def _grounded_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[float]:
    mcp, store, node_ids, _event_id = _seed(tmp_path / "baseline.sqlite3", monkeypatch)
    _remember(mcp)
    return _usefulness(store, node_ids)


def test_explicit_before_grounded_credits_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = _grounded_baseline(tmp_path, monkeypatch)
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    mcp, store, node_ids, event_id = _seed(tmp_path / "explicit-first.sqlite3", monkeypatch)
    used = node_ids[USED_INDEX]
    # Same closing call: marks run before implicit feedback, so explicit wins
    # the pair and grounding finds it spent.
    response = _remember(mcp, used=[used])
    assert response["feedback_marks"]["accepted"] == 1
    mark_id = _marks(store)[0]["id"]
    assert _explicit(store) == [(event_id, used, f"mark:{mark_id}")]
    assert _ledger(store) == []
    # Exactly one credit's worth, the same the grounded path alone gives.
    assert _usefulness(store, node_ids) == pytest.approx(baseline)


def test_grounded_before_explicit_credits_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    mcp, store, node_ids, event_id = _seed(tmp_path / "grounded-first.sqlite3", monkeypatch)
    used = node_ids[USED_INDEX]
    _remember(mcp)
    assert _ledger(store) == [(event_id, used, "grounded")]
    after_grounded = (_usefulness(store, node_ids), _weights(store))
    response = _remember(mcp, UNRELATED_TRACE, used=[used])
    assert response["feedback_marks"]["accepted"] == 1
    assert _explicit(store) == []
    assert (_usefulness(store, node_ids), _weights(store)) == after_grounded


def test_lookup_before_explicit_credits_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    mcp, store, node_ids, event_id = _seed(tmp_path / "lookup-first.sqlite3", monkeypatch)
    target = node_ids[1]
    mcp.tools["memory_lookup"](node_id=target)
    assert _ledger(store) == [(event_id, target, "lookup")]
    after_lookup = _usefulness(store, node_ids)
    _remember(mcp, UNRELATED_TRACE, used=[target])
    assert _explicit(store) == []
    assert _usefulness(store, node_ids) == after_lookup


def test_explicit_before_lookup_credits_once_and_equals_lookup_credit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reference: a lookup's credit alone.
    mcp, store, node_ids, _event_id = _seed(tmp_path / "lookup-ref.sqlite3", monkeypatch)
    target_index = 1
    mcp.tools["memory_lookup"](node_id=node_ids[target_index])
    lookup_only = (_usefulness(store, node_ids), _weights(store))

    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    mcp, store, node_ids, event_id = _seed(tmp_path / "explicit-then-lookup.sqlite3", monkeypatch)
    target = node_ids[target_index]
    _remember(mcp, UNRELATED_TRACE, used=[target])
    assert [row[:2] for row in _explicit(store)] == [(event_id, target)]
    # Weight 1.0: the grounded assignment, i.e. what a lookup would have given.
    assert _usefulness(store, node_ids) == pytest.approx(lookup_only[0])
    assert _weights(store) == pytest.approx(lookup_only[1])
    snapshot = (_usefulness(store, node_ids), _weights(store))
    mcp.tools["memory_lookup"](node_id=target)
    assert _ledger(store) == []
    assert (_usefulness(store, node_ids), _weights(store)) == snapshot


def test_explicit_credit_weight_scales_the_signal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    monkeypatch.setenv("LM_EXPLICIT_CREDIT_WEIGHT", "0")
    mcp, store, node_ids, event_id = _seed(tmp_path / "weight0.sqlite3", monkeypatch)
    before = _usefulness(store, node_ids)
    _remember(mcp, UNRELATED_TRACE, used=[node_ids[1]])
    # Claimed (the pair is spent) but nothing moved at weight zero.
    assert [row[:2] for row in _explicit(store)] == [(event_id, node_ids[1])]
    assert _usefulness(store, node_ids) == before


def test_audit_mode_claims_no_ledger_rows_and_leaves_grounding_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _grounded_baseline(tmp_path, monkeypatch)
    mcp, store, node_ids, event_id = _seed(tmp_path / "audit.sqlite3", monkeypatch)
    used = node_ids[USED_INDEX]
    response = _remember(mcp, used=[used, node_ids[0]])
    assert response["feedback_marks"]["accepted"] == 2
    assert _explicit(store) == []
    # Grounding still claims its own pair exactly as without marks.
    assert _ledger(store) == [(event_id, used, "grounded")]
    assert _usefulness(store, node_ids) == pytest.approx(baseline)
    assert store.list_query_anchors(scope=SCOPE, include_decayed=True) != []


def test_off_mode_records_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "off")
    mcp, store, node_ids, event_id = _seed(tmp_path / "off.sqlite3", monkeypatch)
    off_topic = node_ids[OTHER_INDEX]
    response = _remember(mcp, used=[node_ids[USED_INDEX]], irrelevant=[off_topic])
    assert response["feedback_marks"] == {
        "accepted": 0,
        "dropped": 0,
        "by_reason": {},
        "ignored": True,
    }
    assert _marks(store) == [] and _explicit(store) == []
    # No hygiene either: off is the pre-feature behaviour, all eight linked.
    assert _implicit_targets(store, response["node"]["id"], event_id) == set(node_ids)
    recalled = _recall(mcp, irrelevant=[off_topic])
    assert recalled["feedback_marks"]["ignored"] is True
    assert _marks(store) == []


# ---------------------------------------------------------------------------
# Irrelevance: hook, global usefulness, link hygiene
# ---------------------------------------------------------------------------


def test_irrelevance_hook_runs_only_under_credit_and_never_lowers_usefulness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        feedback_module,
        "_apply_query_irrelevance",
        lambda store, event, node_ids: calls.append((event.id, list(node_ids))),
    )
    mcp, store, node_ids, event_id = _seed(tmp_path / "hook-audit.sqlite3", monkeypatch)
    _remember(mcp, UNRELATED_TRACE, irrelevant=[node_ids[2]])
    assert calls == []

    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    mcp, store, node_ids, event_id = _seed(tmp_path / "hook-credit.sqlite3", monkeypatch)
    _remember(mcp, UNRELATED_TRACE, irrelevant=[node_ids[2], node_ids[6], node_ids[2]])
    assert calls == [(event_id, [node_ids[2], node_ids[6]])]


def test_default_hook_is_a_no_op_for_global_usefulness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    mcp, store, node_ids, _event_id = _seed(tmp_path / "noop.sqlite3", monkeypatch)
    before = (_usefulness(store, node_ids), _weights(store))
    _remember(mcp, UNRELATED_TRACE, irrelevant=node_ids[:4])
    assert (_usefulness(store, node_ids), _weights(store)) == before


@pytest.mark.parametrize("policy", ["audit", "credit"])
def test_no_related_edge_for_irrelevant_in_the_same_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", policy)
    mcp, store, node_ids, event_id = _seed(tmp_path / f"same-{policy}.sqlite3", monkeypatch)
    off_topic = node_ids[OTHER_INDEX]
    response = _remember(mcp, irrelevant=[off_topic])
    trace_id = response["node"]["id"]
    assert _implicit_targets(store, trace_id, event_id) == set(node_ids) - {off_topic}
    assert off_topic not in response["implicit_feedback"]["linked_node_ids"]
    # Provenance still records what was shown.
    trace = store.get_node(trace_id)
    assert off_topic in trace.provenance["recalled_nodes"]


def test_no_related_edge_for_irrelevant_in_the_same_teach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "same-teach.sqlite3", monkeypatch)
    off_topic = node_ids[OTHER_INDEX]
    taught = mcp.tools["memory_teach"](
        node_ids[0],
        f"{QUERY} correction: the alembic drift was a checksum of the wrong revision",
        context={"scope": SCOPE, **AMBIENT},
        irrelevant=[off_topic],
    )
    trace_id = taught["corrective_trace"]["id"]
    assert event_id in taught["implicit_feedback"]["recall_event_ids"]
    targets = _implicit_targets(store, trace_id, event_id)
    assert off_topic not in targets
    assert targets  # the rest of the delivery is still linked


def test_later_irrelevant_mark_removes_that_events_edge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "later.sqlite3", monkeypatch)
    off_topic = node_ids[OTHER_INDEX]
    trace_id = _remember(mcp)["node"]["id"]
    assert _implicit_targets(store, trace_id, event_id) == set(node_ids)

    response = _recall(mcp, irrelevant=[off_topic])
    assert response["feedback_marks"]["accepted"] == 1
    assert _implicit_targets(store, trace_id, event_id) == set(node_ids) - {off_topic}
    assert store.list_connections(source_id=trace_id, target_id=off_topic, relation_type="related") == []


def test_later_mark_leaves_an_edge_that_belongs_to_other_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "foreign-edge.sqlite3", monkeypatch)
    off_topic = node_ids[OTHER_INDEX]
    trace_id = _remember(mcp)["node"]["id"]
    # Somebody else's evidence rewrote the (trace, node, related) edge.
    store.create_connection(trace_id, off_topic, "related", metadata={"basis": "manual"})
    _recall(mcp, irrelevant=[off_topic])
    assert len(store.list_connections(source_id=trace_id, target_id=off_topic, relation_type="related")) == 1


def test_link_policy_credited_links_only_credited_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_IMPLICIT_LINK_POLICY", "credited")
    mcp, store, node_ids, event_id = _seed(tmp_path / "credited.sqlite3", monkeypatch)
    looked_up = node_ids[1]
    mcp.tools["memory_lookup"](node_id=looked_up)
    response = _remember(mcp)
    trace_id = response["node"]["id"]
    assert _implicit_targets(store, trace_id, event_id) == {node_ids[USED_INDEX], looked_up}
    # Provenance stays exhaustive.
    assert set(store.get_node(trace_id).provenance["recalled_nodes"]) >= set(node_ids)


def test_link_policy_credited_counts_explicit_credit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_IMPLICIT_LINK_POLICY", "credited")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    mcp, store, node_ids, event_id = _seed(tmp_path / "credited-explicit.sqlite3", monkeypatch)
    response = _remember(mcp, UNRELATED_TRACE, used=[node_ids[0]])
    assert _implicit_targets(store, response["node"]["id"], event_id) == {node_ids[0]}


def test_link_policy_all_is_todays_behaviour(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "all.sqlite3", monkeypatch)
    response = _remember(mcp)
    assert _implicit_targets(store, response["node"]["id"], event_id) == set(node_ids)
    assert set(response["implicit_feedback"]["linked_node_ids"]) == set(node_ids)


# ---------------------------------------------------------------------------
# Schema: additive, audit CHECK
# ---------------------------------------------------------------------------


def test_marks_table_rejects_unknown_mark_kinds(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "check.sqlite3") as store:
        with pytest.raises(sqlite3.IntegrityError):
            store.record_feedback_marks([{"node_id": "N", "mark": "maybe", "via_tool": "t"}])
        ids = store.record_feedback_marks(
            [{"node_id": "N", "mark": "used", "via_tool": "memory_recall", "accepted": True}]
        )
        store.set_feedback_marks_source(ids, "SRC")
        row = _marks(store)[0]
        assert (row["source_id"], row["recall_event_id"], row["accepted"]) == ("SRC", "", 1)


def test_ledger_ddl_keeps_its_two_basis_check(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "ddl.sqlite3") as store:
        sql = store.connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?", (RECALL_CREDIT_LEDGER_TABLE,)
        ).fetchone()[0]
    assert "CHECK (basis IN ('grounded', 'lookup'))" in sql


# ---------------------------------------------------------------------------
# Prompt arms
# ---------------------------------------------------------------------------

MAX_DESCRIPTION_CHARS = 1024


@pytest.mark.parametrize("arm", ["optional", "mandatory"])
def test_both_prompt_arms_fit_the_budget_and_keep_the_law(arm: str) -> None:
    texts = _explicit_feedback_texts(arm)
    for name in ("recall", "remember", "teach"):
        assert len(texts[name]) <= MAX_DESCRIPTION_CHARS, (name, len(texts[name]))
    recall = texts["recall"]
    assert len(recall) < MAX_DESCRIPTION_CHARS
    assert recall.split(".", 1)[0] == "Recall before acting and at every new turn of thought"
    for phrase in (
        "You MUST recall BEFORE acting",
        "MUST recall MID-WORK",
        "nothing related is missed",
        "stuck or surprised means overdue",
        "'if only I knew' means recall NOW",
        "memory first",
        "only after recall returns nothing",
        "world-before-memory",
        "when uncertain, recall",
        "level:schema",
        "follow it literally",
        "content_ref",
        "memory_lookup",
        "conventions",
        "before inventing one",
        "action plus target",
        "recall cross-project",
        "across scopes",
    ):
        assert phrase in recall, (arm, phrase)
    # The register bans of tests/test_instructions_imperative.py.
    assert "%" not in recall and " + " not in recall
    assert re.search(r"\bLM_[A-Z0-9_]+|experiment|opt[-\s]?in", recall, re.IGNORECASE) is None


def test_optional_arm_is_byte_identical_and_mandatory_is_binding() -> None:
    optional = _explicit_feedback_texts("optional")
    mandatory = _explicit_feedback_texts("mandatory")
    assert optional["recall"] is _RECALL_DESCRIPTION
    assert (optional["used"], optional["irrelevant"]) == (
        _USED_FIELD_DESCRIPTION,
        _IRRELEVANT_FIELD_DESCRIPTION,
    )
    assert mandatory["recall"] is _RECALL_DESCRIPTION_MANDATORY
    assert "You MUST mark each recall's results" in mandatory["recall"]
    assert "used" in mandatory["recall"] and "irrelevant" in mandatory["recall"]
    assert (mandatory["used"], mandatory["irrelevant"]) == (
        _USED_FIELD_DESCRIPTION_MANDATORY,
        _IRRELEVANT_FIELD_DESCRIPTION_MANDATORY,
    )
    assert mandatory["remember"] == optional["remember"]
    assert mandatory["teach"] == optional["teach"]


@pytest.mark.parametrize("arm", ["optional", "mandatory"])
def test_server_registers_the_active_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_PROMPT", arm)
    mcp = create_mcp_server(tmp_path / f"{arm}.sqlite3", mcp_factory=FakeMCP)
    texts = _explicit_feedback_texts(arm)
    assert mcp.descriptions["memory_recall"] == texts["recall"]
    assert mcp.descriptions["memory_remember"] == texts["remember"]
    assert mcp.descriptions["memory_teach"] == texts["teach"]
    annotation = mcp.tools["memory_recall"].__annotations__["used"]
    assert texts["used"] in repr(annotation)


def test_real_fastmcp_schema_carries_optional_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    mcp = create_mcp_server(tmp_path / "real.sqlite3")
    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    for name in ("memory_recall", "memory_remember", "memory_teach"):
        properties = tools[name].parameters["properties"]
        required = set(tools[name].parameters.get("required", []))
        for field in ("used", "irrelevant"):
            assert field in properties, (name, field)
            assert field not in required
    assert "used" not in tools["memory_lookup"].parameters["properties"]
