"""Explicit ``used`` / ``irrelevant`` recall feedback (goal explicit-recall-feedback).

What this module pins, in the order docs/explicit-feedback.md states it:

* **Optional fields on three tools.** memory_recall, memory_remember and
  memory_teach accept ``used``/``irrelevant``; a call without them answers
  exactly as before (no ``feedback_marks`` key, no audit row).
* **Transport deliveries.** Without durable caller context, a marked id
  requires an earlier recall on the same transport (newest such event within
  the lookup-credit window). Dropped marks are audited with ``accepted = 0``;
  tests/test_explicit_feedback_reconnect.py covers durable session context.
* **Valve.** ``audit`` (default) records and claims nothing; ``credit``
  claims basis ``explicit`` once per (event, node) across grounded, lookup
  and explicit, in either order, and hands irrelevant marks to the
  query-irrelevance hook; ``off`` records nothing.
* **History.** Every shown result remains in provenance; explicit marks do
  not assert or erase a knowledge relation to the closing trace.
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
    _explicit_credit_weight,
    _explicit_feedback_policy,
)
from living_memory.irrelevance import list_query_irrelevance
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
    assert sorted(result["id"] for result in recalled["results"]) == sorted(node_ids)
    return mcp, store, node_ids, recalled["recall_event_id"]


def _recall(mcp: Any, **kwargs: Any) -> dict[str, Any]:
    return mcp.tools["memory_recall"](
        QUERY,
        scope=SCOPE,
        max_results=len(BODIES),
        depth=0,
        ambient_context=kwargs.pop("ambient_context", dict(AMBIENT)),
        **kwargs,
    )


def _remember(mcp: Any, content: str = CONSUMING_TRACE, **kwargs: Any) -> dict[str, Any]:
    return mcp.tools["memory_remember"](
        content, kwargs.pop("context", {"scope": SCOPE, **AMBIENT}), **kwargs,
    )


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
    return {row[0] for row in store.connection.execute(
        "SELECT target_id FROM connections WHERE source_id = ? AND type = 'related' "
        "AND json_extract(metadata, '$.basis') = 'implicit_recall_feedback' "
        "AND json_extract(metadata, '$.recall_event_id') = ?", (trace_id, event_id),
    )}


def _rank(store: MemoryStore, event_id: str, node_id: str) -> int:
    event = store.get_recall_event(event_id)
    assert event is not None
    return event.result_ids.index(node_id)


# ---------------------------------------------------------------------------
# Valves: defaults and fallbacks
# ---------------------------------------------------------------------------


def test_valve_defaults_and_unknown_values_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    assert DEFAULT_EXPLICIT_FEEDBACK_POLICY == "audit"
    assert _explicit_feedback_policy() == "audit"
    assert _explicit_credit_weight() == 1.0
    assert _explicit_feedback_prompt_arm() == "optional"
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "bogus")
    monkeypatch.setenv("LM_EXPLICIT_CREDIT_WEIGHT", "-3")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_PROMPT", "bogus")
    assert _explicit_feedback_policy() == "audit"
    assert _explicit_credit_weight() == 1.0
    assert _explicit_feedback_prompt_arm() == "optional"
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", " CREDIT ")
    monkeypatch.setenv("LM_EXPLICIT_CREDIT_WEIGHT", "0.5")
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_PROMPT", "mandatory")
    assert _explicit_feedback_policy() == "credit"
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
    # An independent caller has no earlier delivery; this very recall delivers
    # the node, but a mark only ever refers to an earlier one.
    _stamp(monkeypatch, "fresh-connection")
    response = _recall(mcp, ambient_context={**AMBIENT, "session_id": "new-caller"}, used=[node_ids[0]])
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
    response = _remember(mcp, UNRELATED_TRACE, used=[node_ids[0], "01UNKNOWNNODE"],
                         context={"scope": SCOPE, **AMBIENT, "session_id": "other-caller"})
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
    # Delivery history does not establish relations under any mark policy.
    assert _implicit_targets(store, response["node"]["id"], event_id) == set()
    recalled = _recall(mcp, irrelevant=[off_topic])
    assert recalled["feedback_marks"]["ignored"] is True
    assert _marks(store) == []


# ---------------------------------------------------------------------------
# Irrelevance: query effects, global usefulness and retained history
# ---------------------------------------------------------------------------


def _apply_marks(store: MemoryStore, used: list[str], irrelevant: list[str]) -> Any:
    return feedback_module.apply_explicit_marks(
        store, used, irrelevant, via_tool="memory_remember",
        source_id="feedback-test", transport_session_id=TRANSPORT,
    )


def _count_query_vectors(store: MemoryStore) -> list[list[str]]:
    calls: list[list[str]] = []
    embedder = store._resolve_chunk_embedder()

    def counting(texts: Any) -> Any:
        calls.append(list(texts))
        return embedder(texts)

    store.set_chunk_embedder(counting)
    return calls


@pytest.mark.parametrize("lookup_first", [False, True])
def test_mixed_feedback_reuses_anchor_and_preserves_credit_and_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lookup_first: bool
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    mcp, store, nodes, event_id = _seed(tmp_path / "mixed.sqlite3", monkeypatch)
    used, irrelevant = nodes[:2]
    _apply_marks(store, [], [used])
    if lookup_first:
        mcp.tools["memory_lookup"](node_id=used)
        assert list_query_irrelevance(store)[0]["weight"] == 0.5
    before = _usefulness(store, nodes)
    anchor_before = store.list_query_anchors(scope=SCOPE)[0]
    calls = _count_query_vectors(store)

    result = _apply_marks(store, [used], [irrelevant])

    assert result.summary() == {"accepted": 2, "dropped": 0, "by_reason": {}}
    assert calls == []
    assert result.credited == ([] if lookup_first else [(event_id, used)])
    anchor, = store.list_query_anchors(scope=SCOPE)
    assert anchor.id == anchor_before.id
    assert anchor.reinforcement_count == anchor_before.reinforcement_count + (not lookup_first)
    assert result.anchor_ids == ([] if lookup_first else [anchor.id])
    edge, = store.list_query_anchor_edges(anchor_id=anchor.id)
    assert edge.target_id == used and edge.hits == 1
    rows = {row["node_id"]: row for row in list_query_irrelevance(store)}
    assert rows[used]["weight"] == 0 and rows[used]["cancels"] == 1
    assert rows[irrelevant]["weight"] == 0.5 and rows[irrelevant]["marks"] == 1
    assert {row["anchor_id"] for row in rows.values()} == {anchor.id}
    after = _usefulness(store, nodes)
    assert after[1:] == before[1:]
    if lookup_first:
        assert after[0] == before[0]
        assert _ledger(store) == [(event_id, used, "lookup")]
        assert _explicit(store) == []
    else:
        assert after[0] > before[0]
        assert [row[:2] for row in _explicit(store)] == [(event_id, used)]
    assert store.credited_node_ids(event_id) == {used}

    _apply_marks(store, [used], [])
    assert calls == []
    assert _usefulness(store, nodes) == after
    assert store.list_query_anchors(scope=SCOPE)[0].reinforcement_count == anchor.reinforcement_count
    assert store.list_query_anchor_edges(anchor_id=anchor.id)[0].hits == 1
    assert {row["node_id"]: row for row in list_query_irrelevance(store)} == rows


def test_explicit_feedback_batches_mixed_positive_negative_and_blank_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    _mcp, store, nodes, _event_id = _seed(tmp_path / "batch.sqlite3", monkeypatch)
    queries = ["  apple orchard  ", "bronze sculpture", "  coral reef  ", "  "]
    events = [
        store.record_recall_event(
            query=query, scope=SCOPE,
            ambient_context={"transport_session_id": TRANSPORT},
            results=[{"node_id": node_id} for node_id in nodes[index * 2:index * 2 + 2]],
        )
        for index, query in enumerate(queries)
    ]
    calls = _count_query_vectors(store)

    result = _apply_marks(store, [nodes[0], nodes[2], nodes[6]], [nodes[1], nodes[4], nodes[7]])

    assert result.summary()["accepted"] == 6
    assert calls == [[queries[0], queries[1], queries[2].strip()]]
    anchors = {anchor.query: anchor for anchor in store.list_query_anchors(scope=SCOPE)}
    assert set(anchors) == {queries[0], queries[1], queries[2].strip()}
    assert set(result.anchor_ids) == {anchors[queries[0]].id, anchors[queries[1]].id}
    assert store.list_query_anchor_edges(anchor_id=anchors[queries[2].strip()].id) == []
    assert {(row["last_event_id"], row["node_id"]) for row in list_query_irrelevance(store)} == {
        (events[0].id, nodes[1]), (events[2].id, nodes[4]),
    }
    assert set(result.credited) == {(events[i].id, nodes[2 * i]) for i in (0, 1, 3)}


@pytest.mark.parametrize("failure", ["positive_edge", "cancellation", "negative"])
def test_optional_query_write_failure_preserves_independent_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    _mcp, store, nodes, event_id = _seed(tmp_path / "write-failure.sqlite3", monkeypatch)
    used, irrelevant = nodes[:2]
    _apply_marks(store, [], [used])
    anchor_before = store.list_query_anchors(scope=SCOPE)[0]
    calls = _count_query_vectors(store)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("optional write unavailable")

    if failure == "positive_edge":
        monkeypatch.setattr(store, "upsert_query_anchor_edge", fail)
    else:
        function = "cancel_query_irrelevance" if failure == "cancellation" else "record_query_irrelevance"
        monkeypatch.setattr(feedback_module, function, fail)
    result = _apply_marks(store, [used], [irrelevant])

    assert result.summary()["accepted"] == 2
    assert result.credited == [(event_id, used)]
    assert calls == []
    anchor, = store.list_query_anchors(scope=SCOPE)
    assert anchor.id == anchor_before.id
    assert anchor.reinforcement_count == anchor_before.reinforcement_count + 1
    edges = store.list_query_anchor_edges(anchor_id=anchor.id)
    assert [edge.target_id for edge in edges] == ([] if failure == "positive_edge" else [used])
    rows = {row["node_id"]: row for row in list_query_irrelevance(store)}
    assert rows[used]["weight"] == (0.5 if failure == "cancellation" else 0)
    if failure == "negative":
        assert irrelevant not in rows
    else:
        assert rows[irrelevant]["weight"] == 0.5


@pytest.mark.parametrize("preparation", ["blank", "no_encoder", "raises", "empty_batch", "empty_vector"])
def test_unavailable_query_preparation_preserves_explicit_credit_and_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preparation: str
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    _mcp, store, nodes, event_id = _seed(tmp_path / "preparation.sqlite3", monkeypatch)
    if preparation == "blank":
        with store.transaction():
            store.connection.execute("UPDATE recall_events SET query = '  ' WHERE id = ?", (event_id,))
    calls: list[list[str]] = []

    def encode(texts: Any) -> Any:
        calls.append(list(texts))
        if preparation == "raises":
            raise RuntimeError("encoder unavailable")
        return [] if preparation == "empty_batch" else [[] for _ in texts]

    monkeypatch.setattr(feedback_module, "_anchor_embedder", lambda store: None if preparation == "no_encoder" else encode)
    result = _apply_marks(store, [nodes[0]], [nodes[1]])

    assert result.summary()["accepted"] == 2
    assert result.credited == [(event_id, nodes[0])]
    assert [row[:2] for row in _explicit(store)] == [(event_id, nodes[0])]
    assert calls == ([] if preparation in {"blank", "no_encoder"} else [[QUERY]])
    assert result.anchor_ids == [] and store.list_query_anchors(scope=SCOPE) == []
    assert list_query_irrelevance(store) == []


@pytest.mark.parametrize("failure", ["empty_vector", "anchor_write"])
def test_failed_event_does_not_suppress_other_events_in_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    _mcp, store, nodes, event_id = _seed(tmp_path / "batch-failure.sqlite3", monkeypatch)
    other = store.record_recall_event(
        query="coral reef", scope=SCOPE,
        ambient_context={"transport_session_id": TRANSPORT},
        results=[{"node_id": node_id} for node_id in nodes[2:4]],
    )
    calls: list[list[str]] = []
    embedder = store._resolve_chunk_embedder()
    insert = store.insert_query_anchor

    def encode(texts: Any) -> Any:
        calls.append(list(texts))
        vectors = embedder(texts)
        if failure == "empty_vector":
            vectors[0] = []
        return vectors

    def insert_anchor(**kwargs: Any) -> Any:
        if kwargs["query"] == QUERY:
            raise RuntimeError("one anchor unavailable")
        return insert(**kwargs)

    store.set_chunk_embedder(encode)
    if failure == "anchor_write":
        monkeypatch.setattr(store, "insert_query_anchor", insert_anchor)
    result = _apply_marks(store, [nodes[0], nodes[2]], [nodes[1], nodes[3]])

    assert calls == [[QUERY, other.query]]
    assert result.summary()["accepted"] == 4
    assert result.credited == [(event_id, nodes[0]), (other.id, nodes[2])]
    anchor, = store.list_query_anchors(scope=SCOPE)
    assert anchor.query == other.query
    assert store.list_query_anchor_edges(anchor_id=anchor.id)[0].target_id == nodes[2]
    row, = list_query_irrelevance(store)
    assert row["last_event_id"] == other.id and row["node_id"] == nodes[3]


def test_zero_weight_credit_still_cancels_and_records_negatives_once_prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    monkeypatch.setenv("LM_EXPLICIT_CREDIT_WEIGHT", "0")
    _mcp, store, nodes, event_id = _seed(tmp_path / "zero-weight.sqlite3", monkeypatch)
    _apply_marks(store, [], [nodes[0]])
    before = (_usefulness(store, nodes), _weights(store), store.list_query_anchors(scope=SCOPE))
    calls = _count_query_vectors(store)

    result = _apply_marks(store, [nodes[0]], [nodes[1]])

    assert calls == []
    assert result.credited == [(event_id, nodes[0])] and result.anchor_ids == []
    assert (_usefulness(store, nodes), _weights(store), store.list_query_anchors(scope=SCOPE)) == before
    rows = {row["node_id"]: row for row in list_query_irrelevance(store)}
    assert rows[nodes[0]]["cancels"] == 1 and rows[nodes[0]]["weight"] == 0
    assert rows[nodes[1]]["weight"] == 0.5


def test_irrelevance_hook_runs_only_under_credit_and_never_lowers_usefulness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        feedback_module,
        "record_query_irrelevance",
        lambda store, anchor_id, event_id, node_ids: calls.append((event_id, list(node_ids))),
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
    assert _implicit_targets(store, trace_id, event_id) == set()
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
    assert targets == set()


def test_later_irrelevant_mark_preserves_delivery_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "later.sqlite3", monkeypatch)
    off_topic = node_ids[OTHER_INDEX]
    trace_id = _remember(mcp)["node"]["id"]
    assert _implicit_targets(store, trace_id, event_id) == set()
    historical = store.create_connection(
        trace_id, off_topic, "related", weight=.75,
        metadata={"basis": "implicit_recall_feedback", "recall_event_id": event_id},
    )
    provenance = store.get_node(trace_id).provenance

    response = _recall(mcp, irrelevant=[off_topic])
    assert response["feedback_marks"]["accepted"] == 1
    assert store.get_connection(historical.id) == historical
    assert store.get_node(trace_id).provenance == provenance
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


def test_lookup_credit_does_not_assert_relation_to_closing_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "credited.sqlite3", monkeypatch)
    looked_up = node_ids[1]
    mcp.tools["memory_lookup"](node_id=looked_up)
    response = _remember(mcp)
    trace_id = response["node"]["id"]
    assert _implicit_targets(store, trace_id, event_id) == set()
    # Provenance stays exhaustive.
    assert set(store.get_node(trace_id).provenance["recalled_nodes"]) >= set(node_ids)


def test_explicit_credit_does_not_assert_relation_to_closing_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LM_EXPLICIT_FEEDBACK_POLICY", "credit")
    mcp, store, node_ids, event_id = _seed(tmp_path / "credited-explicit.sqlite3", monkeypatch)
    response = _remember(mcp, UNRELATED_TRACE, used=[node_ids[0]])
    assert _implicit_targets(store, response["node"]["id"], event_id) == set()


def test_closing_trace_does_not_link_delivered_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "all.sqlite3", monkeypatch)
    response = _remember(mcp)
    assert _implicit_targets(store, response["node"]["id"], event_id) == set()


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
        "MID-WORK for nearby knowledge",
        "stuck or surprised means overdue",
        "'if only I knew' means recall NOW",
        "Nonempty recall neither forbids nor replaces needed current-source checks",
        "When uncertain, recall",
        "level:schema",
        "grants no authority",
        "applicable complete recipes",
        "current system/operator instructions and task conditions",
        "Teach refuted claims at once",
        "content_ref",
        "memory_lookup",
        "conventions",
        "concept before inventing",
        "action plus target",
        "omit both scope and ambient_context.scope for cross-project knowledge",
    ):
        assert phrase in recall, (arm, phrase)
    # The register bans of tests/test_instructions_imperative.py.
    assert "%" not in recall and " + " not in recall
    assert re.search(r"\bLM_[A-Z0-9_]+|experiment|opt[-\s]?in", recall, re.IGNORECASE) is None


def test_feedback_arms_share_guidance_and_keep_distinct_mark_requirements() -> None:
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
