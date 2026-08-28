"""Executable guards for the cold-start candidate dataset.

The properties worth defending here are the ones a plausible-looking rewrite
would silently break: that "cold" means exactly what the live reader means by
it, that a component never straddles two splits, that an evaluation set is not
one correlated cluster wearing a thousand hats, and that the verifier actually
refuses a tampered block instead of nodding at it.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "recall_map_coldstart_dataset.py"
MANIFEST = ROOT / "artifacts" / "recall-map" / "relevance" / "dataset-manifest.json"
ANALYSIS = ROOT / "artifacts" / "recall-map" / "relevance" / "feature-analysis.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("recall_map_coldstart_dataset_tested", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


C = _load_module()

AS_OF = C.COLDSTART_AS_OF
HORIZON = 24
_WINDOW_END = C.shift_hours(AS_OF, -HORIZON)


def _shift(instant: str, hours: int) -> str:
    return C.shift_hours(instant, hours)


def _build_db(path: Path, *, events, ledger, nodes) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE recall_events (
            id TEXT PRIMARY KEY, query TEXT, scope TEXT, task TEXT,
            session_id TEXT, ambient_context TEXT, results TEXT,
            created_at TEXT, transport_session_id TEXT, recall_map TEXT
        );
        CREATE TABLE nodes (id TEXT PRIMARY KEY, level TEXT, created_at TEXT);
        CREATE TABLE recall_delivery_history (
            delivery_event_id TEXT, node_id TEXT, delivered_at TEXT,
            outcome_end TEXT, transport_matched INTEGER, transport_consumed INTEGER,
            fallback_consumed INTEGER, lookup_consumed INTEGER,
            PRIMARY KEY (delivery_event_id, node_id)
        );
        CREATE TABLE recall_delivery_history_state (
            singleton INTEGER PRIMARY KEY, format_version INTEGER,
            complete INTEGER, unavailable_reason TEXT, updated_at TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO recall_delivery_history_state VALUES (1, 2, 1, NULL, ?)", (AS_OF,)
    )
    for event in events:
        connection.execute(
            "INSERT INTO recall_events (id, query, scope, task, session_id, "
            "ambient_context, results, created_at, transport_session_id, recall_map) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                event["id"],
                event.get("query", "a query"),
                event.get("scope", "global"),
                event.get("task"),
                event.get("session_id"),
                json.dumps(event.get("ambient", {})),
                json.dumps(
                    [
                        {"node_id": node_id, "level": "trace", "score": 0.5,
                         "bm25_score": 0.1, "vector_score": 0.2,
                         "graph_score": 0.0, "trigger_score": 0.0}
                        for node_id in event.get("results", [])
                    ]
                ),
                event["created_at"],
                event.get("transport"),
                None,
            ),
        )
    for node_id, created_at in nodes.items():
        connection.execute(
            "INSERT INTO nodes VALUES (?,?,?)", (node_id, "trace", created_at)
        )
    for row in ledger:
        connection.execute(
            "INSERT INTO recall_delivery_history VALUES (?,?,?,?,?,?,?,?)",
            (
                row["event"],
                row["node"],
                row["at"],
                _shift(row["at"], HORIZON),
                int(row.get("transport_matched", 0)),
                int(row.get("transport_consumed", 0)),
                int(row.get("fallback_consumed", 0)),
                None,
            ),
        )
    connection.commit()
    connection.close()
    return path


# The decision instant of the probe event.  Its maturation cut is exactly
# 24 hours earlier, which is the boundary the cold rule turns on.
DECIDED_AT = "2026-08-05T00:00:00Z"
CUT = "2026-08-04T00:00:00Z"
RESIDUAL = ["n3", "n4", "n5", "n6", "n7", "n8"]


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    path = tmp_path_factory.mktemp("coldstart") / "local.sqlite3"
    events = [
        # Probe: nine results, so ranks 3..8 are the residual the pool sees.
        {
            "id": "E1",
            "created_at": DECIDED_AT,
            "transport": "T1",
            "task": "alpha",
            "session_id": "S1",
            "results": [f"n{index}" for index in range(9)],
        },
        # Transport-limb consumer one hour later: this is what gives E1 an
        # opportunity to be consumed at all.
        {"id": "E2", "created_at": "2026-08-05T01:00:00Z", "transport": "T1", "task": "alpha"},
        # No later event shares T2 or (global, beta), so E3 has no opportunity.
        {
            "id": "E3",
            "created_at": "2026-08-10T00:00:00Z",
            "transport": "T2",
            "task": "beta",
            "results": ["m0", "m1", "m2", "m3"],
        },
        # Shares E1's session identity, so it must land in E1's component even
        # though it shares neither transport nor cache key.
        {
            "id": "E4",
            "created_at": "2026-08-11T00:00:00Z",
            "transport": "T3",
            "task": "gamma",
            "session_id": "S1",
            "results": [f"p{index}" for index in range(5)],
        },
        {"id": "E5", "created_at": "2026-08-11T00:30:00Z", "transport": "T3", "task": "gamma"},
        # Newest event: proves the snapshot reaches as_of + horizon_hours.
        {"id": "E9", "created_at": _shift(AS_OF, HORIZON + 6), "transport": "T9"},
    ]
    ledger = [
        # Each residual candidate's own delivery row supplies its label.
        {"event": "E1", "node": "n3", "at": DECIDED_AT, "transport_matched": 1,
         "transport_consumed": 1},
        {"event": "E1", "node": "n4", "at": DECIDED_AT},
        {"event": "E1", "node": "n5", "at": DECIDED_AT},
        {"event": "E1", "node": "n6", "at": DECIDED_AT, "fallback_consumed": 1},
        {"event": "E1", "node": "n7", "at": DECIDED_AT},
        {"event": "E1", "node": "n8", "at": DECIDED_AT},
        # Priors.  n3 matured a day before the cut; n5 matures exactly ON it;
        # n4's window closes an hour too late to count.
        {"event": "E0a", "node": "n3", "at": "2026-08-03T00:00:00Z"},
        {"event": "E0b", "node": "n5", "at": CUT},
        {"event": "E0c", "node": "n4", "at": "2026-08-04T01:00:00Z"},
        {"event": "E3", "node": "m3", "at": "2026-08-10T00:00:00Z"},
        {"event": "E4", "node": "p3", "at": "2026-08-11T00:00:00Z"},
        {"event": "E4", "node": "p4", "at": "2026-08-11T00:00:00Z"},
    ]
    nodes = {
        name: "2026-07-01T00:00:00Z"
        for name in [f"n{i}" for i in range(9)] + ["m3", "p3", "p4"]
    }
    _build_db(path, events=events, ledger=ledger, nodes=nodes)
    result = C.build_population(path)
    yield result
    result["connection"].close()


def _by_node(extracted) -> dict[str, object]:
    return {row.node_id: row for row in extracted["candidates"]}


def test_cold_means_zero_matured_windows_at_the_decision_instant(extracted) -> None:
    rows = _by_node(extracted)
    assert set(rows) == set(RESIDUAL) | {"m3", "p3", "p4"}

    # n3's prior window closed a full day before the cut.
    assert rows["n3"].matured == 1 and not rows["n3"].cold
    # n5's prior window closes exactly ON the decision instant.  The live
    # reader's predicate is ``outcome_end <= decision``, so it counts.
    assert rows["n5"].matured == 1 and not rows["n5"].cold
    # n4's prior window closes one hour after the decision.  It does not.
    assert rows["n4"].matured == 0 and rows["n4"].cold
    for name in ("n6", "n7", "n8"):
        assert rows[name].matured == 0 and rows[name].cold

    cold = [row for row in extracted["candidates"] if row.event_id == "E1" and row.cold]
    assert sorted(row.node_id for row in cold) == ["n4", "n6", "n7", "n8"]


def test_a_candidates_own_delivery_never_counts_as_its_own_history(extracted) -> None:
    """The row being labelled must not also be the history that scores it."""

    rows = _by_node(extracted)
    assert rows["m3"].matured == 0 and rows["m3"].cold
    assert rows["p3"].matured == 0 and rows["p4"].matured == 0


def test_the_label_follows_the_ledgers_transport_first_rule(extracted) -> None:
    rows = _by_node(extracted)
    assert rows["n3"].consumed is True   # transport matched and consumed
    assert rows["n6"].consumed is True   # transport did not match; fallback did
    for name in ("n4", "n5", "n7", "n8"):
        assert rows[name].consumed is False


def test_only_the_residual_tail_enters_the_population(extracted) -> None:
    """Ranks below the head cut are delivered, not residual, and must be absent."""

    seen = {row.node_id for row in extracted["candidates"]}
    assert seen.isdisjoint({"n0", "n1", "n2", "m0", "m1", "m2", "p0", "p1", "p2"})
    assert extracted["protocol"]["organic_head_cut"] == 3
    assert min(row.rank for row in extracted["candidates"]) == 3


def test_items_without_a_consumption_opportunity_are_marked_not_dropped(extracted) -> None:
    rows = _by_node(extracted)
    assert rows["n3"].opportunity is True
    assert rows["m3"].opportunity is False


def test_a_shared_session_identity_bridges_two_events_into_one_component(extracted) -> None:
    candidates = extracted["candidates"]
    identities = extracted["identities"]
    C.build_components(
        candidates, identities, component_event_ids={"E1", "E3", "E4"}
    )
    components = {row.event_id: row.component_id for row in candidates}
    assert components["E1"] == components["E4"], "session identity S1 must bridge E1 and E4"
    assert components["E3"] != components["E1"]


def test_the_candidate_window_closes_a_horizon_before_the_as_of(extracted) -> None:
    start, end = extracted["protocol"]["candidate_window"]
    assert end == _shift(AS_OF, -HORIZON)
    assert start < end
    # Every candidate's own 24h window therefore closes at or before the as-of.
    for row in extracted["candidates"]:
        assert _shift(row.decided_at, HORIZON) <= AS_OF


def test_an_unmatured_snapshot_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "young.sqlite3"
    _build_db(
        path,
        events=[
            {"id": "E1", "created_at": "2026-08-05T00:00:00Z", "transport": "T1",
             "task": "alpha", "results": [f"n{i}" for i in range(5)]},
            {"id": "E2", "created_at": "2026-08-05T01:00:00Z", "transport": "T1",
             "task": "alpha"},
        ],
        ledger=[{"event": "E1", "node": "n3", "at": "2026-08-05T00:00:00Z"}],
        nodes={"n3": "2026-07-01T00:00:00Z"},
    )
    with pytest.raises(SystemExit, match="have not matured"):
        C.build_population(path)


def test_an_incomplete_ledger_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.sqlite3"
    _build_db(
        path,
        events=[{"id": "E9", "created_at": _shift(AS_OF, HORIZON + 6), "transport": "T9"}],
        ledger=[],
        nodes={},
    )
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE recall_delivery_history_state SET complete = 0, unavailable_reason = 'backfill'"
    )
    connection.commit()
    connection.close()
    with pytest.raises(SystemExit, match="ledger is incomplete"):
        C.build_population(path)


# ---------------------------------------------------------------------------
# Assignment: pure functions, no database
# ---------------------------------------------------------------------------


SKEWED = {f"c{index:03d}": size for index, size in enumerate([1500] + [10] * 200)}


def _loads(placement, sizes):
    loads = {name: 0 for name in C.SPLIT_NAMES}
    for component, split in placement.items():
        loads[split] += sizes[component]
    return loads


def test_balanced_assignment_keeps_every_component_whole_and_in_one_split() -> None:
    placement = C._balanced_assignment(SKEWED, seed=C.SPLIT_SEED)
    assert set(placement) == set(SKEWED)
    assert set(placement.values()) == set(C.SPLIT_NAMES)
    # Conservation is the leakage guard: a component that appeared in two
    # splits would put one cache or session identity on both sides.
    assert sum(_loads(placement, SKEWED).values()) == sum(SKEWED.values())


def test_the_outsized_component_is_absorbed_by_train_not_by_an_evaluation_set() -> None:
    """Train carries no generalization claim, so the lump belongs there."""

    placement = C._balanced_assignment(SKEWED, seed=C.SPLIT_SEED)
    biggest = max(SKEWED, key=lambda key: SKEWED[key])
    assert placement[biggest] == "train"


def test_balanced_assignment_beats_the_uniform_draw_on_target_deviation() -> None:
    total = sum(SKEWED.values())
    targets = {"train": 0.60 * total, "eval": 0.20 * total, "holdout": 0.20 * total}

    def deviation(placement) -> float:
        loads = _loads(placement, SKEWED)
        return sum(abs(loads[name] - targets[name]) for name in C.SPLIT_NAMES)

    balanced = deviation(C._balanced_assignment(SKEWED, seed=C.SPLIT_SEED))
    uniform = deviation(C._uniform_draw_assignment(sorted(SKEWED), seed=C.SPLIT_SEED))
    assert balanced < uniform


def test_balanced_assignment_is_deterministic() -> None:
    first = C._balanced_assignment(SKEWED, seed=C.SPLIT_SEED)
    second = C._balanced_assignment(dict(reversed(list(SKEWED.items()))), seed=C.SPLIT_SEED)
    assert first == second


def test_the_uniform_draw_it_replaced_really_does_concentrate_a_split() -> None:
    """The rejected rule is rejected for a measured reason, not a stylistic one."""

    placement = C._uniform_draw_assignment(sorted(SKEWED), seed=C.SPLIT_SEED)
    loads = {name: 0 for name in C.SPLIT_NAMES}
    for component, split in placement.items():
        loads[split] += SKEWED[component]
    dominated = placement[max(SKEWED, key=lambda key: SKEWED[key])]
    assert SKEWED["c000"] / loads[dominated] > 0.5, (
        "the 1500-item component should swallow whichever split it lands in"
    )


# ---------------------------------------------------------------------------
# The published artifact
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def published() -> dict:
    return json.loads(MANIFEST.read_text())["coldstart"]


def test_the_committed_block_verifies(published) -> None:
    assert C.verify_coldstart_block(published) == []


def test_the_analysis_binds_the_extended_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text())
    analysis = json.loads(ANALYSIS.read_text())
    assert analysis["dataset_manifest_sha256"] == C.sha256_text(C.canonical_json(manifest))


def test_every_split_carries_cold_candidates_and_they_are_disjoint(published) -> None:
    sets = {}
    for name in C.SPLIT_NAMES:
        summary = published["splits"][name]
        assert summary["cold_candidates"] > 0, f"{name} has no cold candidates"
        assert 0.0 < summary["cold_candidate_share"] < 1.0
        assert summary["rate"] == round(summary["consumed"] / summary["items"], 6)
        sets[name] = set(summary["components"])
    assert sets["train"] & sets["eval"] == set()
    assert sets["train"] & sets["holdout"] == set()
    assert sets["eval"] & sets["holdout"] == set()
    assert published["population"]["opportunity_bearing"]["items"] == sum(
        published["splits"][name]["items"] for name in C.SPLIT_NAMES
    )


def test_neither_evaluation_set_is_one_correlated_cluster(published) -> None:
    # The holdout carries the generalization claim and holds no forced
    # material, so it answers for all of its items.
    holdout = published["splits"]["holdout"]["concentration"]
    assert holdout["forced_components"] == 0
    assert holdout["largest_component_share"] <= C.MAX_EVAL_COMPONENT_SHARE
    assert holdout["effective_components"] > 10
    # Eval answers for what it drew freely. A component forced there by the
    # pre-registered observed-map rule had no discretion in the matter, and its
    # weight is published rather than hidden behind the bar.
    ev = published["splits"]["eval"]["concentration"]
    assert ev["largest_freely_assigned_component_share"] <= C.MAX_EVAL_COMPONENT_SHARE
    assert ev["effective_components_excluding_forced"] > 10
    assert ev["forced_component_items"] > 0, "the forced arm must be visible, not absent"
    assert ev["largest_component_share"] > ev["largest_freely_assigned_component_share"], (
        "the unconditional figure must still be published beside the free one"
    )


def test_the_published_item_fractions_land_on_the_declared_targets(published) -> None:
    """The measured outcome the balancing rule exists to produce.

    Eval is allowed to overshoot, but only by material it did not choose: the
    forced observed-map arm is placed before any balancing happens. Anything
    beyond that is a balancing failure, so the slack is bounded by the forced
    arm's own size rather than by a comfortable constant.
    """

    total = published["population"]["opportunity_bearing"]["items"]
    targets = published["split"]["target_item_fractions"]
    forced_share = (
        published["splits"]["eval"]["concentration"]["forced_component_items"] / total
    )
    assert forced_share > 0
    for name in C.SPLIT_NAMES:
        share = published["splits"][name]["items"] / total
        slack = 0.01 + (forced_share if name == "eval" else forced_share / 2)
        assert share == pytest.approx(targets[name], abs=slack), name
    # And the balancer still reports where it actually landed.
    achieved = published["split"]["carry_over"]["achieved_item_fractions"]
    for name in C.SPLIT_NAMES:
        assert achieved[name] == pytest.approx(
            published["splits"][name]["items"] / total, abs=1e-5
        )


def test_the_rejected_uniform_draw_is_published_with_its_measured_failure(published) -> None:
    diagnostic = published["split"]["uniform_draw_diagnostic"]
    assert diagnostic["rejected"] is True
    worst = max(
        diagnostic[name]["largest_component_share"] for name in C.SPLIT_NAMES
    )
    assert worst > C.MAX_EVAL_COMPONENT_SHARE, (
        "the diagnostic must record the concentration that motivated the change"
    )


def test_the_artifact_publishes_no_identity_and_only_opaque_digests(published) -> None:
    module = C.sealed()
    leaked = set(module._walk_keys(published)) & module.SENSITIVE_ARTIFACT_KEYS
    assert leaked == set()
    for name in C.SPLIT_NAMES:
        for digest in published["splits"][name]["components"]:
            assert module.OPAQUE_DIGEST_RE.fullmatch(digest)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda b: b["splits"]["holdout"].update(cold_candidates=0), "no cold candidates"),
        (
            lambda b: b["splits"]["eval"]["components"].append(
                b["splits"]["train"]["components"][0]
            ),
            "components overlap",
        ),
        (lambda b: b["split"].update(pairwise_component_disjoint=False), "disjointness"),
        (lambda b: b["splits"]["train"].update(consumed=1), "rate is not consumed over items"),
        (lambda b: b["protocol"].update(organic_head_cut=0), "organic_head_cut"),
        (
            lambda b: b["protocol"].update(mutable_node_stats_read=["access_count"]),
            "mutable node stats",
        ),
        (
            lambda b: b["maturation"].update(snapshot_newest_event="2026-08-22T00:00:00Z"),
            "does not cover as_of",
        ),
        (
            lambda b: b["splits"]["holdout"]["concentration"].update(
                largest_component_share=0.9
            ),
            "dominated by one component",
        ),
        (
            lambda b: b["privacy"].update(candidate_only_holdout_accessed=True),
            "non-access assertion",
        ),
        (lambda b: b["splits"]["eval"]["components"].append("nothex"), "non-opaque"),
        (
            lambda b: b["split"]["holdout_contamination"].update(intersection=3),
            "the sealed partition is contaminated",
        ),
        (
            lambda b: b["split"]["holdout_contamination"].update(holds=False),
            "non-contamination assertion",
        ),
        (
            lambda b: b["split"]["carry_over"]["v1_reproduction"].update(
                reproduces_pinned_v1=False
            ),
            "reproduction receipt",
        ),
        (
            lambda b: b["split"]["carry_over"]["v1_reproduction"]["splits"]["holdout"].update(
                items=1
            ),
            "does not match the pinned v1",
        ),
        (
            lambda b: b["splits"]["holdout"].update(
                cold_candidates=C.MINIMUM_COLD_CANDIDATES_PER_SPLIT - 1
            ),
            "below the pre-registered minimum",
        ),
        (
            lambda b: b["splits"]["train"]["stratified"].pop("by_era"),
            "is not stratified by_era",
        ),
        (
            lambda b: b["splits"]["eval"]["stratified"]["by_era"][C.ERA_ENLARGEMENT].update(
                items=0
            ),
            "received nothing from the enlargement",
        ),
        (
            lambda b: b["splits"]["holdout"]["concentration"].update(forced_components=2),
            "forced observed-map component",
        ),
        (
            lambda b: b["splits"]["eval"]["concentration"].update(
                largest_freely_assigned_component_share=0.9
            ),
            "dominated by one component",
        ),
        (
            lambda b: b["protocol"]["organic_cap_is_not_a_free_lever"][
                "measured_on_the_snapshot"
            ].update(last_ledgered_rank=12),
            "the cap claim is stale",
        ),
        (
            lambda b: b["protocol"].update(candidate_window=["2026-08-01T00:00:00Z", _WINDOW_END]),
            "does not open at the enlarged window start",
        ),
        (lambda b: b["splits"]["train"].pop("family_unscoreable"), "family_unscoreable"),
        (
            lambda b: b["enlargement"]["window"].update(
                snapshot_first_event="2026-05-13T00:00:00Z"
            ),
            "excludes an era",
        ),
        (
            lambda b: b["split"].pop("assignment_inputs"),
            "publishes no assignment_inputs",
        ),
        (
            lambda b: b["split"]["assignment_inputs"]["components"].pop(),
            "does not reproduce from assignment_inputs alone",
        ),
        (
            lambda b: b["split"]["assignment_inputs"]["components"][0].update(n=99999),
            "does not reproduce from assignment_inputs alone",
        ),
        (
            lambda b: b["split"]["assignment_inputs"]["constants"].update(
                seed="recall-map-coldstart-components-v2"
            ),
            "is not the published split seed",
        ),
        (
            lambda b: b["split"]["assignment_inputs"]["constants"].update(
                role_precedence=["holdout", "eval", "train"]
            ),
            "is not the carry-over order",
        ),
        (
            lambda b: b["split"]["assignment_inputs"]["declared"].pop(),
            "does not enumerate the assignment's inputs in order",
        ),
        (
            lambda b: b["split"].pop("assignment_reproduction"),
            "publishes no assignment_reproduction receipt",
        ),
        (
            lambda b: b["split"]["assignment_reproduction"].update(
                reproduces_published_v2=False
            ),
            "does not claim reproduction",
        ),
        (
            lambda b: b["split"]["assignment_reproduction"]["items"].update(holdout=1),
            "misreports the holdout item count",
        ),
    ],
)
def test_the_verifier_refuses_a_tampered_block(published, mutate, expected) -> None:
    tampered = deepcopy(published)
    mutate(tampered)
    problems = C.verify_coldstart_block(tampered)
    assert any(expected in problem for problem in problems), problems


def test_the_extractor_never_reads_a_mutable_node_statistic() -> None:
    """The columns the sealed feature allowlist rejects must not be selected."""

    source = SCRIPT.read_text()
    for column in ("access_count", "usefulness_score", "last_accessed"):
        assert f'"{column}"' not in source
        assert f"SELECT {column}" not in source
        assert f", {column}" not in source


def test_the_field_subtree_is_fail_closed(tmp_path: Path) -> None:
    prohibited = tmp_path / "artifacts" / "recall-map" / "relevance" / "field" / "store.sqlite3"
    with pytest.raises(SystemExit, match="prohibited"):
        C.assert_not_holdout(prohibited)


# ---------------------------------------------------------------------------
# The carry-over rule
#
# The enlargement's whole risk is here.  Neither half of the split survives a
# change of population -- ``component_id`` digests the component's entire
# member set, and ``_balanced_assignment`` is greedy over the whole population
# -- so a naive re-run scatters already-read components into the sealed holdout
# while every published count still looks healthy.  These are the guards that
# would catch that happening again.
# ---------------------------------------------------------------------------


_CARRY_SIZES = {"cA": 10, "cB": 10, "cC": 10, "cD": 10}
_CARRY_EVENTS = {
    "e1": "cA", "e2": "cA",   # train + holdout
    "e3": "cB", "e4": "cB",   # eval + holdout
    "e5": "cC",               # holdout only
    "e6": "cD",               # never seen in v1
}
_CARRY_V1 = {
    "e1": "train", "e2": "holdout",
    "e3": "eval", "e4": "holdout",
    "e5": "holdout",
}


def test_a_component_inherits_the_most_read_role_among_its_events() -> None:
    out = C.carry_over_assignment(
        _CARRY_SIZES, _CARRY_EVENTS, v1_role_by_event=_CARRY_V1
    )
    # train beats holdout, eval beats holdout, and a pure-holdout component
    # keeps the only role it ever had.
    assert out["placement"]["cA"] == "train"
    assert out["placement"]["cB"] == "eval"
    assert out["placement"]["cC"] == "holdout"
    assert out["reason"]["cA"] == "inherited_most_read_v1_role"
    assert out["reason"]["cD"] == "freshly_assigned"
    assert out["fresh_components"] == ["cD"]


def test_a_map_delivery_forces_its_whole_component_out_of_train_and_holdout() -> None:
    out = C.carry_over_assignment(
        _CARRY_SIZES,
        _CARRY_EVENTS,
        v1_role_by_event=_CARRY_V1,
        forced_eval_components=["cA", "cC"],
    )
    assert out["placement"]["cA"] == "eval", "forcing must beat an inherited train role"
    assert out["placement"]["cC"] == "eval", "forcing must beat an inherited holdout role"
    assert out["reason"]["cA"] == "touches_observed_map_delivery"
    assert set(out["forced_eval_components"]) == {"cA", "cC"}
    # And a forced component is never left for the balancer to place again.
    assert "cA" not in out["fresh_components"]


def test_nothing_already_read_can_reach_the_holdout_however_hungry_it_is() -> None:
    """The holdout is starved of fresh material and still refuses read work."""

    sizes = {f"c{index}": 5 for index in range(40)}
    events = {f"e{index}": f"c{index}" for index in range(40)}
    # 30 of 40 components were already read; only 10 are genuinely new, so the
    # balancer cannot fill a 20% holdout without reaching for read material.
    v1 = {f"e{index}": ("train" if index % 2 else "eval") for index in range(30)}
    out = C.carry_over_assignment(sizes, events, v1_role_by_event=v1)
    holdout = {
        component for component, split in out["placement"].items() if split == "holdout"
    }
    read = {events[event] for event, role in v1.items() if role in ("train", "eval")}
    assert holdout, "the fresh material must still be able to fill a holdout"
    assert holdout & read == set(), "already-read work reached the holdout"
    assert holdout <= set(out["fresh_components"])


def test_the_split_refuses_to_publish_a_contaminated_holdout(extracted, monkeypatch) -> None:
    """Fail closed: the assertion is enforced, not merely reported."""

    candidates = [row for row in extracted["candidates"] if row.opportunity]
    identities = extracted["identities"]
    universe = {"E1", "E3", "E4"}
    by_event = C.build_components(candidates, identities, component_event_ids=universe)
    monkeypatch.setattr(
        C,
        "carry_over_assignment",
        lambda *args, **kwargs: {
            "placement": {component: "holdout" for component in set(by_event.values())},
            "reason": {},
            "fresh_components": [],
            "forced_eval_components": [],
            "inherited_components": [],
            "achieved_item_fractions": {},
        },
    )
    with pytest.raises(SystemExit, match="sealed partition would be destroyed"):
        C.assign_three_way_splits(
            candidates,
            identities,
            component_event_ids=universe,
            v1_role_by_event={"E1": "train"},
        )


def test_a_drifted_v1_rebuild_is_refused_before_any_role_is_carried_over(extracted) -> None:
    """A v1 role that is not the v1 role actually read protects nothing."""

    with pytest.raises(SystemExit, match="did not reproduce"):
        C.reconstruct_v1_partition(
            extracted["v1_candidates"],
            extracted["identities"],
            component_event_ids={"E1", "E3", "E4"},
        )


# ---------------------------------------------------------------------------
# The rank cap is not a second lever
# ---------------------------------------------------------------------------


class _StubEvent:
    def __init__(self, event_id: str, result_ids: list[str]) -> None:
        self.id = event_id
        self.result_ids = result_ids


def test_a_rank_with_results_but_no_outcomes_is_reported_as_unledgered() -> None:
    events = [_StubEvent("E", [f"n{index}" for index in range(6)])]
    outcome = {("E", "n3"): True, ("E", "n4"): False}
    census = C.rank_coverage_census(events, outcome)
    assert census["first_ledgered_rank"] == 3
    assert census["last_ledgered_rank"] == 4
    assert census["by_rank"]["5"] == {"results_recorded": 1, "outcome_ledgered": 0}
    assert census["by_rank"]["0"] == {"results_recorded": 1, "outcome_ledgered": 0}


def test_lifting_the_cap_would_add_only_rows_that_carry_no_label(tmp_path) -> None:
    """Why the enlargement is the window and only the window.

    An event delivering twelve results, with the ledger covering ranks 3..8 as
    the live one does.  Lifting the per-event cap reaches ranks 9, 10 and 11 and
    finds no outcome for any of them, so it buys three skipped rows and not one
    extra observation.
    """

    decided = "2026-08-06T00:00:00Z"
    events = [
        {
            "id": "D1",
            "created_at": decided,
            "transport": "TD",
            "task": "delta",
            "results": [f"d{index}" for index in range(12)],
        },
        {"id": "D2", "created_at": "2026-08-06T01:00:00Z", "transport": "TD", "task": "delta"},
        {"id": "D9", "created_at": _shift(AS_OF, HORIZON + 6), "transport": "TX"},
    ]
    ledger = [{"event": "D1", "node": f"d{rank}", "at": decided} for rank in range(3, 9)]
    nodes = {f"d{index}": "2026-07-01T00:00:00Z" for index in range(12)}
    path = _build_db(tmp_path / "local.sqlite3", events=events, ledger=ledger, nodes=nodes)

    bindings = C.sealed().load_frozen_bindings()
    connection = bindings.effect.open_readonly(path)
    try:
        earliest = connection.execute("SELECT MIN(created_at) FROM recall_events").fetchone()[0]
        loaded = bindings.effect.load_events(
            connection,
            start=str(earliest),
            end=AS_OF,
            with_map=bindings.effect._has_recall_map_column(connection),
        )
        by_node, outcome, _ = C._load_delivery_ledger(connection)
        shared = {
            "events": loaded,
            "connection": connection,
            "effect": bindings.effect,
            "source_name": "local",
            "window_start": "2026-08-01T00:00:00Z",
            "window_end": _WINDOW_END,
            "head_cut": 3,
            "horizon": HORIZON,
            "consumer_index": bindings.effect.ConsumerIndex(loaded, horizon_hours=HORIZON),
            "by_node": by_node,
            "outcome": outcome,
        }
        capped, _, capped_skipped, _ = C._extract_window(organic_cap=6, **shared)
        uncapped, _, uncapped_skipped, _ = C._extract_window(organic_cap=None, **shared)
    finally:
        connection.close()

    assert [row.rank for row in capped] == [3, 4, 5, 6, 7, 8]
    assert capped_skipped == 0
    assert [row.rank for row in uncapped] == [3, 4, 5, 6, 7, 8], (
        "an unledgered rank must never become a candidate"
    )
    assert uncapped_skipped == 3
    assert C.rank_coverage_census(loaded, outcome)["last_ledgered_rank"] == 8


def test_family_unscoreable_counts_a_row_whose_envelope_recorded_no_scores() -> None:
    def _row(scores):
        return C.Candidate(
            source="s",
            event_id="e",
            node_id="n",
            decided_at=DECIDED_AT,
            rank=3,
            consumed=False,
            matured=0,
            matured_consumed=0,
            trailing_nonconsumed=0,
            opportunity=True,
            scores=scores,
        )

    full = {name: 1.0 for name in C.SCORE_FIELDS}
    partial = dict(full, graph_score=None)
    assert C._family_unscoreable([_row(full), _row(full)]) == 0
    assert C._family_unscoreable([_row(full), _row(partial)]) == 1
    assert C._family_unscoreable([_row({})]) == 1


# ---------------------------------------------------------------------------
# The published enlargement
# ---------------------------------------------------------------------------


def test_every_split_clears_the_pre_registered_cold_minimum(published) -> None:
    before = published["enlargement"]["population_before"]
    for name in C.SPLIT_NAMES:
        summary = published["splits"][name]
        assert summary["cold_candidates"] >= C.MINIMUM_COLD_CANDIDATES_PER_SPLIT, name
        assert summary["cold_candidates"] > before[name]["cold_candidates"], name
        assert summary["cold_base_rate"] == summary["cold_rate"]
    assert published["split"]["every_split_meets_the_minimum"] is True
    assert (
        published["split"]["minimum_cold_candidates_per_split"]
        == C.MINIMUM_COLD_CANDIDATES_PER_SPLIT
    )


def test_the_published_holdout_holds_no_already_read_v1_work(published) -> None:
    contamination = published["split"]["holdout_contamination"]
    assert contamination["holds"] is True
    assert contamination["intersection"] == 0
    assert contamination["v1_read_events_in_scope"] > 0, (
        "an assertion checked against no already-read event proves nothing"
    )
    reproduction = published["split"]["carry_over"]["v1_reproduction"]
    assert reproduction["reproduces_pinned_v1"] is True
    for name in C.SPLIT_NAMES:
        assert reproduction["splits"][name] == C.V1_SPLITS[name], name


def test_the_per_component_draw_would_have_scattered_read_work_into_the_holdout(
    published,
) -> None:
    """The trap is measured on this population, not asserted in prose."""

    leak = published["split"]["uniform_draw_diagnostic"][
        "already_read_components_it_would_put_in_holdout"
    ]
    assert leak["components"] > 0 and leak["items"] > 0, (
        "if the obvious rule were harmless the carry-over would need no defence"
    )


def test_each_split_is_cut_by_era_and_rank_band_and_the_cuts_close(published) -> None:
    for name in C.SPLIT_NAMES:
        summary = published["splits"][name]
        strata = summary["stratified"]
        for axis in ("by_era", "by_rank_band"):
            cut = strata[axis]
            assert sum(row["items"] for row in cut.values()) == summary["items"], (name, axis)
            assert sum(row["cold_candidates"] for row in cut.values()) == summary[
                "cold_candidates"
            ], (name, axis)
            for label, row in cut.items():
                assert row["family_unscoreable"] is not None, (name, axis, label)
                assert row["concentration"]["effective_components"] is not None
        # Both sides of the enlargement must actually be present in every split.
        assert strata["by_era"][C.ERA_ENLARGEMENT]["items"] > 0, name
        assert strata["by_era"][C.ERA_PREEXISTING]["items"] > 0, name


def test_the_enlargement_publishes_what_it_cost_as_well_as_what_it_bought(published) -> None:
    cost = published["enlargement"]["concentration_cost"]
    before = cost["effective_components_before"]
    after = cost["effective_components_after"]
    for name in C.SPLIT_NAMES:
        assert before[name] == C.V1_SPLITS[name]["effective_components"], name
        assert after[name] == published["splits"][name]["concentration"][
            "effective_components"
        ], name
    # The cost is real on this population; the artifact must not imply otherwise.
    assert any(after[name] < before[name] for name in C.SPLIT_NAMES), (
        "a concentration_cost block that records no cost is decoration"
    )
    assert cost["identity_rule_unchanged"] is True


def test_the_window_opens_before_the_first_event_that_was_ever_recorded(published) -> None:
    window = published["enlargement"]["window"]
    assert window["v2"][0] == C.CANDIDATE_WINDOW_START
    assert window["v1"][0] == C.V1_CANDIDATE_WINDOW_START
    assert window["v2"][0] < window["v1"][0]
    assert window["opens_before_the_first_persisted_event"] is True
    # Checkable off the artifact alone, not only enforced inside the run.
    assert window["v2"][0] <= window["snapshot_first_event"] < window["v1"][0]
    assert published["enlargement"]["rank_cap"]["changed"] is False


def test_the_outcome_ledger_stops_at_rank_eight_on_the_published_snapshot(published) -> None:
    census = published["protocol"]["organic_cap_is_not_a_free_lever"][
        "measured_on_the_snapshot"
    ]
    assert census["first_ledgered_rank"] == published["protocol"]["organic_head_cut"]
    assert census["last_ledgered_rank"] == 8
    for rank, row in census["by_rank"].items():
        if 3 <= int(rank) <= 8:
            assert row["outcome_ledgered"] == row["results_recorded"], rank
        else:
            assert row["outcome_ledgered"] == 0, rank
            assert row["results_recorded"] > 0, rank


# ---------------------------------------------------------------------------
# The published assignment, and re-executing it from the artifact alone
#
# ``coldstart-prereg-v2.json#split_protocol.assignment_rule
# .blindness_invariant_binding`` does not accept "the assignment is blind" as a
# claim: the manifest must publish the function's name and its inputs, and a
# re-executable check that recomputes the assignment from those inputs ALONE
# and reproduces the published membership digests.  These guards defend that
# property from the two ways it decays -- an input table that quietly stops
# covering the population, and a replay that reads more than it declares.
# ---------------------------------------------------------------------------


def test_the_assignment_inputs_declare_the_whole_component_universe(published) -> None:
    inputs = published["split"]["assignment_inputs"]
    assert inputs["function"]["name"] == published["split"]["assignment"]
    assert inputs["function"]["logical_location"] == (
        "scripts/recall_map_coldstart_dataset.py:carry_over_assignment"
    )
    assert inputs["constants"]["seed"] == published["split"]["seed"]
    assert inputs["constants"]["split_names_order"] == list(C.SPLIT_NAMES)
    assert inputs["constants"]["role_precedence"] == list(C.ROLE_PRECEDENCE)
    assert [entry["name"] for entry in inputs["declared"]] == [
        entry["name"] for entry in C.DECLARED_ASSIGNMENT_INPUTS
    ]
    for entry in inputs["declared"]:
        assert entry["what"].strip()
        assert entry["why_it_carries_no_consumption_label"].strip()

    components = inputs["components"]
    digests = [entry["c"] for entry in components]
    assert len(components) == published["split"]["components"]
    assert digests == sorted(digests) and len(set(digests)) == len(digests)
    # The compact encoding is only readable if omission means what it says.
    assert all("r" not in entry or entry["r"] for entry in components)
    assert all("f" not in entry or entry["f"] is True for entry in components)
    assert inputs["components_encoding"].strip()

    sizes = {entry["c"]: entry["n"] for entry in components}
    assert sum(sizes.values()) == published["population"]["opportunity_bearing"]["items"]
    # Every component that actually carries rows is accounted for, and no
    # published member is a zero-item entry.
    for name in C.SPLIT_NAMES:
        members = published["splits"][name]["components"]
        assert all(member in sizes for member in members), name
        assert all(sizes[member] > 0 for member in members), name
    assert sum(1 for size in sizes.values() if size > 0) == sum(
        len(published["splits"][name]["components"]) for name in C.SPLIT_NAMES
    )
    reading = inputs["blindness_invariant_reading"]
    assert reading["verdict"] == "does not violate the invariant"
    assert reading["this_assignment_also_reads"] and reading["why"]


def test_the_assignment_reproduces_from_its_published_inputs_alone(published) -> None:
    """The check the plan demands, run against the shipped artifact."""

    inputs = published["split"]["assignment_inputs"]
    members = C.replay_published_assignment(inputs)
    sizes = {entry["c"]: entry["n"] for entry in inputs["components"]}
    for name in C.SPLIT_NAMES:
        summary = published["splits"][name]
        assert C._membership_digest(members[name]) == summary["membership_digest"], name
        assert members[name] == summary["components"], name
        assert sum(sizes[member] for member in members[name]) == summary["items"], name
    # And the receipt in the artifact says the same thing.
    receipt = published["split"]["assignment_reproduction"]
    assert receipt["reproduces_published_v2"] is True
    assert receipt["inputs_used"] == [
        entry["name"] for entry in C.DECLARED_ASSIGNMENT_INPUTS
    ]
    assert len(receipt["procedure"]) == 7
    for name in C.SPLIT_NAMES:
        assert receipt["membership_digests"][name] == published["splits"][name][
            "membership_digest"
        ]
        assert receipt["components"][name] == len(published["splits"][name]["components"])
        assert receipt["items"][name] == published["splits"][name]["items"]


def test_the_reproduction_reads_the_published_block_and_nothing_else(
    published, monkeypatch
) -> None:
    """A replay that could reach the database would prove nothing.

    The receipt's whole value is that its input is the printed table, so the
    test denies it every other source: sqlite is made to explode on contact,
    the extraction entry points are removed, and the block is handed over as a
    detached JSON round-trip with the rest of the manifest thrown away.  Then
    the table is perturbed, to show the replay is really a function of it
    rather than an echo of the digests it is checked against.
    """

    def _no_database(*args, **kwargs):
        raise AssertionError("the replay opened a database")

    monkeypatch.setattr(C.sqlite3, "connect", _no_database)
    monkeypatch.setattr(C, "build_population", _no_database)
    monkeypatch.setattr(C, "carry_over_assignment", _no_database)
    detached = json.loads(json.dumps(published["split"]["assignment_inputs"]))

    members = C.replay_published_assignment(detached)
    for name in C.SPLIT_NAMES:
        assert C._membership_digest(members[name]) == published["splits"][name][
            "membership_digest"
        ], name

    # It is a function of the table, not an echo of the digests it is checked
    # against: perturb a declared input and the answer moves with it.  Both
    # perturbations are chosen to be decisive rather than marginal -- the
    # observed-map flag relocates a component outright, and a zero item count
    # removes it from every published membership by rule 7.
    held = published["splits"]["holdout"]["components"][0]
    forced = json.loads(json.dumps(detached))
    next(entry for entry in forced["components"] if entry["c"] == held)["f"] = True
    moved = C.replay_published_assignment(forced)
    assert held not in moved["holdout"] and held in moved["eval"]
    assert C._membership_digest(moved["holdout"]) != published["splits"]["holdout"][
        "membership_digest"
    ], "a replay whose answer survives a changed input is not reading the input"

    emptied = json.loads(json.dumps(detached))
    next(entry for entry in emptied["components"] if entry["c"] == held)["n"] = 0
    without = C.replay_published_assignment(emptied)
    assert held not in without["holdout"]

    # And it fails closed rather than reporting a partition it did not check.
    with pytest.raises(SystemExit, match="did not reproduce"):
        C.assignment_reproduction_receipt(
            forced,
            {
                name: published["splits"][name]["membership_digest"]
                for name in C.SPLIT_NAMES
            },
        )


def test_the_verify_assignment_command_runs_off_the_artifact_alone() -> None:
    """Runnable by a third party holding the manifest, with no --source."""

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "verify-assignment", "--manifest", str(MANIFEST)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "all three published membership digests" in result.stdout
    for name in C.SPLIT_NAMES:
        assert name in result.stdout


def test_the_verify_assignment_command_refuses_a_doctored_manifest(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text())
    manifest["coldstart"]["split"]["assignment_inputs"]["components"][0]["n"] += 1000
    doctored = tmp_path / "dataset-manifest.json"
    doctored.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "verify-assignment", "--manifest", str(doctored)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode != 0
    assert "did not reproduce" in result.stderr


# ---------------------------------------------------------------------------
# Independence accounting
#
# The enlargement's cost was published; what was missing was the join between
# that cost and the sign failure the fit then recorded.  These guards defend
# the two halves of the join separately, because they have different licences:
# the structural tables may look at all three splits, and the direction
# accounting may look at coldstart_train and nothing else.
# ---------------------------------------------------------------------------


def _candidate(component_id: str, index: int, *, cold: bool = False, consumed: bool = False):
    return C.Candidate(
        source="local",
        event_id=f"{component_id}-e{index}",
        node_id=f"{component_id}-n{index}",
        decided_at="2026-08-10T00:00:00Z",
        rank=3 + index % 6,
        consumed=consumed,
        matured=0 if cold else 5,
        matured_consumed=0,
        trailing_nonconsumed=0,
        opportunity=True,
        component_id=component_id,
    )


def _synthetic(sizes: dict[str, int]) -> list:
    return [
        _candidate(component_id, index)
        for component_id, size in sizes.items()
        for index in range(size)
    ]


def test_a_cap_keeps_every_component_inside_the_split_it_was_assigned_to() -> None:
    """The correction the manifest now carries, as an executable claim.

    ``an_intra_split_cap_does_not_weaken_disjointness`` says a cap trims usage,
    not assignment.  If that is true then capping cannot introduce a component
    that the uncapped split did not already own, and cannot move one anywhere.
    """

    rows = _synthetic({"a": 900, "b": 40, "c": 7})
    before = {row.component_id for row in rows}
    for cap in C.CAP_GRID:
        kept = C.cap_survivors(rows, cap)
        assert {row.component_id for row in kept} <= before
        for component_id in before:
            held = sum(1 for row in rows if row.component_id == component_id)
            assert sum(1 for row in kept if row.component_id == component_id) == min(held, cap)


def test_the_caps_nest_so_a_smaller_cap_never_admits_a_row_a_larger_one_dropped() -> None:
    """``cap_feasibility.rule.caps_nest``, which the published table relies on.

    The survivor order must not depend on the cap.  A rule that re-drew per cap
    would make two adjacent rows of the table incomparable, and the whole point
    of the grid is to read down it.
    """

    rows = _synthetic({"big": 700, "mid": 63, "small": 9})
    previous: set[str] | None = None
    for cap in sorted(C.CAP_GRID):
        kept = {row.event_id for row in C.cap_survivors(rows, cap)}
        if previous is not None:
            assert previous <= kept
        previous = kept


def test_the_cap_draw_reads_no_label_and_no_family_member_value() -> None:
    """A cap that selected on the outcome would manufacture the direction.

    Flipping every consumption bit and every score must not move one row in or
    out of the surviving set.
    """

    rows = _synthetic({"a": 300, "b": 25})
    baseline = [row.event_id for row in C.cap_survivors(rows, 50)]
    for row in rows:
        row.consumed = not row.consumed
        row.matured = 0 if row.matured else 5
        row.scores = {"score": 1.0, "bm25_score": -3.0}
    assert [row.event_id for row in C.cap_survivors(rows, 50)] == baseline


def test_component_equalized_weighting_drops_no_row_and_equalises_the_components() -> None:
    rows = _synthetic({"giant": 1000, "tiny": 4})
    weights = C._equalized_weights(rows)
    assert len(weights) == len(rows)
    assert all(weight > 0 for weight in weights)
    per_component: dict[str, float] = {}
    for row, weight in zip(rows, weights, strict=True):
        per_component[row.component_id] = per_component.get(row.component_id, 0.0) + weight
    assert pytest.approx(per_component["giant"], rel=1e-9) == per_component["tiny"]
    assert pytest.approx(sum(weights), rel=1e-9) == 1.0


def test_the_effective_sample_size_is_the_row_count_when_nothing_is_reweighted() -> None:
    """Kish's figure has to be on the raw item count's scale to be readable.

    Otherwise the capped, equalized and uncapped arms could not be compared
    without a conversion the artifact does not publish.
    """

    assert C._effective_sample_size([1.0] * 137) == 137.0
    concentrated = C._effective_sample_size(C._equalized_weights(_synthetic({"a": 1000, "b": 4})))
    assert concentrated is not None and concentrated < 1004


def test_the_published_component_sizes_account_for_every_item_in_the_split(published) -> None:
    for name in C.SPLIT_NAMES:
        summary = published["splits"][name]
        distribution = summary["component_sizes"]
        pairs = distribution["size_and_cold_by_size_desc"]
        assert sum(size for size, _cold in pairs) == summary["items"]
        assert sum(cold for _size, cold in pairs) == summary["cold_candidates"]
        assert len(pairs) == len(summary["components"])
        assert all(cold <= size for size, cold in pairs)
        assert distribution["effective_components"] == summary["concentration"][
            "effective_components"
        ]


def test_the_cap_table_retains_exactly_what_the_component_sizes_imply(published) -> None:
    """The feasibility table has to be a fact about the published population.

    Recomputing each cell from ``component_sizes`` alone means a reader holding
    only the artifact can check it, which is the difference between a
    measurement and an assertion.
    """

    table = published["enlargement"]["cap_feasibility"]["table"]
    for cap in C.CAP_GRID:
        for name in C.SPLIT_NAMES:
            sizes = [
                size
                for size, _cold in published["splits"][name]["component_sizes"][
                    "size_and_cold_by_size_desc"
                ]
            ]
            cell = table[str(cap)][name]
            assert cell["items"] == sum(min(size, cap) for size in sizes)
            assert cell["items_dropped"] == published["splits"][name]["items"] - cell["items"]
            assert cell["meets_the_cold_minimum"] is (
                cell["cold_candidates"] >= C.MINIMUM_COLD_CANDIDATES_PER_SPLIT
            )


def test_a_cap_buys_independence_and_the_table_says_how_much(published) -> None:
    """The question item 2 exists to answer, asked of the shipped artifact.

    A table that reported caps but left every split as concentrated as the
    uncapped population would be a table with nothing in it.
    """

    feasibility = published["enlargement"]["cap_feasibility"]
    uncapped = feasibility["uncapped_effective_components"]
    for cap in C.CAP_GRID:
        for name in C.SPLIT_NAMES:
            capped = feasibility["table"][str(cap)][name]["effective_components"]
            assert capped >= uncapped[name]
    assert feasibility["table"]["25"]["train"]["effective_components"] > uncapped["train"] * 10


def test_the_direction_accounting_measures_train_and_no_other_split(published) -> None:
    """The seal boundary, as a property of the artifact rather than a promise.

    ``coldstart_eval`` carries the falsifiable direction test a successor plan
    still has to register, and ``coldstart_holdout`` is sealed outright.  A
    family-member statistic keyed by either would spend a test before it was
    written.
    """

    accounting = published["train_direction_accounting"]
    assert accounting["split"] == "train"
    assert accounting["splits_measured"] == ["train"]
    keys = set(C._walk_mapping_keys(accounting))
    assert "eval" not in keys and "holdout" not in keys
    # Every arm's row count is bounded by train, so no arm can be quietly
    # reading a wider population than the one it names.
    arms = [accounting["unweighted"], accounting["component_equalized"]]
    arms += list(accounting["by_cap"].values()) + list(accounting["by_era"].values())
    assert all(arm["rows"] <= published["splits"]["train"]["items"] for arm in arms)
    assert sum(arm["rows"] for arm in accounting["by_era"].values()) == (
        published["splits"]["train"]["items"]
    )


def test_no_family_member_statistic_is_published_on_a_sealed_split(published) -> None:
    """The other half of the same boundary: the structural tables stay structural.

    ``cap_feasibility`` and ``component_equalized`` do cover eval and holdout,
    which is allowed precisely because they read component membership and the
    cold flag and nothing else.  If a member name ever appears under them, that
    licence has been exceeded.
    """

    members = {member for _source, member, _sign in C.FAMILY_DIRECTIONS}
    for block in (
        published["enlargement"]["cap_feasibility"],
        published["enlargement"]["component_equalized"],
    ):
        assert not (members & set(C._walk_mapping_keys(block)))
        for key in C._walk_mapping_keys(block):
            assert key not in {"consumed", "rate", "cold_rate", "cold_base_rate"}


def test_the_unweighted_arm_reproduces_the_step_three_numbers_it_explains(published) -> None:
    """An account of a HALT that does not reproduce the HALT is an account of something else."""

    accounting = published["train_direction_accounting"]
    reproduction = accounting["reproduces_published_step3"]
    assert reproduction["agrees"] is True
    for _source, member, _sign in C.FAMILY_DIRECTIONS:
        compared = reproduction["compared"][member]
        pinned = C.PUBLISHED_STEP3_RANK_UNIFORM[member]
        assert compared["published_whole_tail"] == pinned["whole_tail"]
        assert compared["published_cold"] == pinned["cold"]
        assert compared["whole_tail"] == pinned["whole_tail"]
        assert compared["cold"] == pinned["cold"]
        arm = accounting["unweighted"]["members"][member]
        assert arm["whole_tail"] == pinned["whole_tail"]
        assert arm["cold"] == pinned["cold"]


def test_every_arm_publishes_the_cohort_counts_the_cold_clause_needs(published) -> None:
    """``cold_subpopulation_clause`` makes "too small to measure" blocking.

    An arm that reported a cold number without its two cohort counts would let
    a successor read a direction off a cohort the v2 plan already declared
    INDETERMINATE.
    """

    accounting = published["train_direction_accounting"]
    arms = [accounting["unweighted"], accounting["component_equalized"]]
    arms += list(accounting["by_cap"].values()) + list(accounting["by_era"].values())
    for arm in arms:
        assert isinstance(arm["cold_rows"], int)
        assert isinstance(arm["cold_consumed"], int)
        assert arm["cold_check_measurable"] is (
            arm["cold_rows"] >= C.MINIMUM_COLD_ROWS_FOR_A_DIRECTION_CHECK
            and arm["cold_consumed"] >= C.MINIMUM_CONSUMED_COLD_ROWS_FOR_A_DIRECTION_CHECK
        )


def test_the_stability_fold_restates_the_arms_and_invents_nothing(published) -> None:
    """``direction_stability`` is the shape a successor plan reads.

    It must therefore be derivable from the arms above it, key by key, or it is
    a second opinion dressed as a summary.
    """

    accounting = published["train_direction_accounting"]
    by_member = accounting["direction_stability"]["by_member"]
    lookup = {"unweighted": accounting["unweighted"]}
    lookup |= {f"cap_{cap}": accounting["by_cap"][str(cap)] for cap in C.CAP_GRID}
    lookup["component_equalized"] = accounting["component_equalized"]
    lookup |= {f"era_{name}": accounting["by_era"][name] for name in C.ERA_NAMES}
    for _source, member, sign in C.FAMILY_DIRECTIONS:
        entry = by_member[member]
        assert entry["registered_direction"] == ("positive" if sign > 0 else "negative")
        assert set(entry["matches_registered_by_arm"]) == set(lookup)
        for label, verdicts in entry["matches_registered_by_arm"].items():
            source = lookup[label]["members"][member]
            assert verdicts["whole_tail"] == source["whole_tail_matches_registered"]
            assert verdicts["cold"] == source["cold_matches_registered"]
            for key in ("whole_tail", "cold"):
                if source[key] is not None:
                    assert source[f"{key}_matches_registered"] is (source[key] * sign > 0)


def test_the_accounting_registers_no_direction_and_selects_no_estimator(published) -> None:
    """This node measures; ``coldstart-prereg-v3`` decides.

    A ``selected``/``chosen``/``registered_cap`` key here would be this node
    pre-empting the plan it exists to inform, which is the failure mode the
    separation was drawn to prevent.
    """

    accounting = published["train_direction_accounting"]
    keys = set(C._walk_mapping_keys(accounting)) | set(
        C._walk_mapping_keys(published["enlargement"]["cap_feasibility"])
    )
    for forbidden in ("selected_cap", "chosen_cap", "registered_cap", "verdict", "threshold"):
        assert forbidden not in keys
    revision = json.loads((ROOT / "artifacts/recall-map/relevance/policy.json").read_text())[
        "coldstart_revision"
    ]
    assert revision["holdout_evaluations"] == 0
    assert revision["sealed_partition_reads_observed"] == 0
    assert revision["field_subtree_accessed"] is False
    assert revision["verdict"] == "negative_result"
    # Stated as the INVARIANT rather than as a literal step number. The plan
    # that halts is coldstart-prereg-v3 now and its step numbering is its own;
    # what this test is actually for is that measuring here did not spend the
    # sealed partition, and a hard-coded step pins a superseded plan instead.
    assert revision["halted_at"]["step"] > 0
    assert revision["holdout_first_read_after_parameters_sealed"] is False


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (
            lambda b: b["splits"]["train"].pop("component_sizes"),
            "publishes no component_sizes distribution",
        ),
        (
            lambda b: b["splits"]["eval"]["component_sizes"][
                "size_and_cold_by_size_desc"
            ].append([1, 0]),
            "component sizes do not sum to its item count",
        ),
        (
            lambda b: b["splits"]["holdout"]["component_sizes"][
                "size_and_cold_by_size_desc"
            ].reverse(),
            "component sizes are not sorted descending",
        ),
        (
            lambda b: b["enlargement"].pop("cap_feasibility"),
            "publishes no cap_feasibility table",
        ),
        (
            lambda b: b["enlargement"]["cap_feasibility"]["table"]["50"]["train"].update(
                items=99999
            ),
            "retains more items than the split has",
        ),
        (
            lambda b: b["enlargement"]["cap_feasibility"]["table"]["100"]["holdout"].update(
                cold_candidates=1
            ),
            "misreports the cold minimum",
        ),
        (
            lambda b: b["enlargement"]["cap_feasibility"].update(
                caps_clearing_the_cold_minimum_on_every_split=[]
            ),
            "misreported in the feasible cap set",
        ),
        (
            lambda b: b["enlargement"].pop("component_equalized"),
            "publishes no component_equalized table",
        ),
        (
            lambda b: b["enlargement"]["component_equalized"]["per_split"]["eval"].update(
                effective_sample_size=999999
            ),
            "exceeds its row count",
        ),
        (
            lambda b: b.pop("train_direction_accounting"),
            "publishes no train_direction_accounting",
        ),
        (
            lambda b: b["train_direction_accounting"].update(splits_measured=["train", "eval"]),
            "names a split other than train",
        ),
        (
            lambda b: b["train_direction_accounting"]["by_cap"].__setitem__(
                "eval", b["train_direction_accounting"]["by_cap"]["25"]
            ),
            "carries a cohort keyed 'eval'",
        ),
        (
            lambda b: b["train_direction_accounting"]["by_cap"].pop("400"),
            "does not cover the cap grid",
        ),
        (
            lambda b: b["train_direction_accounting"]["by_era"].clear(),
            "does not cover both eras",
        ),
        (
            lambda b: b["train_direction_accounting"]["unweighted"].update(cold_consumed=1),
            "misreports cold_check_measurable",
        ),
        # Moving the arm and moving the receipt trip different guards on
        # purpose: one says the measurement drifted off the HALT, the other
        # says the receipt stopped quoting the measurement it cites.
        (
            lambda b: b["train_direction_accounting"]["reproduces_published_step3"][
                "compared"
            ]["bm25_score"].update(whole_tail=0.5),
            "does not reproduce the published step-3 reading",
        ),
        (
            lambda b: b["train_direction_accounting"]["unweighted"]["members"][
                "bm25_score"
            ].update(whole_tail=0.5),
            "does not quote the unweighted arm",
        ),
        (
            lambda b: b["train_direction_accounting"]["component_equalized"]["members"][
                "vector_score"
            ].update(cold_matches_registered=False),
            "misreports whether cold matches its registered direction",
        ),
        (
            lambda b: b["train_direction_accounting"]["direction_stability"]["by_member"][
                "bm25_score"
            ].update(stable_across_every_arm=True),
            "misreports overall stability",
        ),
        (
            lambda b: b["train_direction_accounting"]["direction_stability"]["by_member"][
                "trigger_score"
            ]["matches_registered_by_arm"]["cap_50"].update(cold=False),
            "restates arm cap_50 cold differently",
        ),
        (
            lambda b: b["train_direction_accounting"]["reproduces_published_step3"].update(
                agrees=False
            ),
            "does not claim to reproduce the published step-3 numbers",
        ),
    ],
)
def test_the_verifier_refuses_a_tampered_accounting(published, mutate, expected) -> None:
    tampered = deepcopy(published)
    mutate(tampered)
    problems = C.verify_independence_accounting(tampered)
    assert any(expected in problem for problem in problems), problems
