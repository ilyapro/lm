"""Contracts for the post-chunking retrieval-weight recalibration.

Two independent jobs:

1. **Pin what this goal promised not to touch.** The policy floors, the floor
   application, the weight update arithmetic and the proportional credit rule
   are the learning mechanics; a recalibration is allowed to move weight
   *values* and nothing else. These tests fail if any of that behaviour drifts,
   including the parts that are only reachable through the evidence gates.
2. **Hold the published report to its own claims.** ``post-chunking-ab.json``
   asserts a three-way time split, a selection rule fixed before the holdout,
   and a decision. The report is only evidence if those claims are checkable,
   so they are checked here against the artifact rather than trusted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from living_memory.config import (
    DEFAULT_RETRIEVAL_POLICY_FLOORS,
    DEFAULT_RETRIEVAL_WEIGHTS,
    MemoryConfig,
)
from living_memory.feedback import _method_signals
from living_memory.models import RetrievalPolicyFloors, RetrievalWeights
from living_memory.replay import _FloorShim, assumed_evidence
from living_memory.storage import MemoryStore, _scope_family

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "artifacts/replay/post-chunking-ab.json"
MARKDOWN_PATH = REPO_ROOT / "artifacts/replay/post-chunking-ab.md"

#: The generalization contract this goal was given: the holdout slice must
#: carry at least this many labeled events for its verdict to mean anything.
MIN_HOLDOUT_EVENTS = 1000

SLICE_NAMES = ("train", "eval", "holdout")


# ---------------------------------------------------------------------------
# 1. Pinned mechanics — floors, floor application, weight updates, credit
# ---------------------------------------------------------------------------


def test_policy_floors_are_exactly_the_configured_values() -> None:
    """`DEFAULT_RETRIEVAL_POLICY_FLOORS` is out of scope for this goal."""

    assert DEFAULT_RETRIEVAL_POLICY_FLOORS == {
        "project": RetrievalPolicyFloors(
            bm25_max=0.85, vector_min=0.15, graph_min=0.05, bm25_min=0.10
        ),
        "global": RetrievalPolicyFloors(
            bm25_max=0.75, vector_min=0.20, graph_min=0.05, bm25_min=0.10
        ),
        "session": RetrievalPolicyFloors(
            bm25_max=0.90, vector_min=0.10, graph_min=0.0, bm25_min=0.10
        ),
    }


#: (scope, input triple, expected floored triple) with both evidence gates open.
FLOOR_CASES_WITH_EVIDENCE = (
    ("project:lm", (1.0, 0.0, 0.0), (0.80, 0.15, 0.05)),
    ("project:lm", (0.0, 1.0, 0.0), (0.10, 0.85, 0.05)),
    ("project:lm", (0.0, 0.0, 1.0), (0.10, 0.15, 0.75)),
    ("project:lm", (0.7, 0.3, 0.0), (0.65, 0.30, 0.05)),
    # Already inside every floor: the pass must be the identity, which is what
    # makes a published weight triple mean what it says.
    ("project:lm", (0.1506, 0.7994, 0.05), (0.1506, 0.7994, 0.05)),
    ("global", (1.0, 0.0, 0.0), (0.75, 0.20, 0.05)),
    ("global", (0.0, 1.0, 0.0), (0.10, 0.85, 0.05)),
    ("global", (0.4, 0.4, 0.2), (0.40, 0.40, 0.20)),
    ("global", (0.9, 0.05, 0.05), (0.75, 0.20, 0.05)),
    ("session:x", (1.0, 0.0, 0.0), (0.90, 0.10, 0.0)),
    ("session:x", (0.8, 0.2, 0.0), (0.80, 0.20, 0.0)),
    # No floors configured for these families: normalization only.
    ("default", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ("weird", (0.5, 0.5, 0.0), (0.5, 0.5, 0.0)),
    # All-zero input normalizes to bm25-only first, then gets floored.
    ("project:lm", (0.0, 0.0, 0.0), (0.80, 0.15, 0.05)),
)

#: The same call with both evidence gates shut. bm25_min still applies (it has
#: no gate), but with `vector_min` and `graph_min` disabled there is no target
#: to move a bm25 surplus into, so `bm25_max` cannot bite either.
FLOOR_CASES_WITHOUT_EVIDENCE = (
    ("project:lm", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ("project:lm", (0.0, 1.0, 0.0), (0.10, 0.90, 0.0)),
    ("project:lm", (0.0, 0.0, 1.0), (0.10, 0.0, 0.90)),
    ("project:lm", (0.7, 0.3, 0.0), (0.70, 0.30, 0.0)),
    ("global", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
)


@pytest.mark.parametrize(("scope", "given", "expected"), FLOOR_CASES_WITH_EVIDENCE)
def test_apply_floors_is_pinned_with_evidence(
    scope: str, given: tuple[float, float, float], expected: tuple[float, float, float]
) -> None:
    shim = _FloorShim(MemoryConfig(), assumed_evidence)
    floored = MemoryStore.apply_retrieval_weight_floors(
        shim, scope, RetrievalWeights(scope, *given)
    )
    assert (floored.bm25, floored.vector, floored.graph) == pytest.approx(expected)


@pytest.mark.parametrize(("scope", "given", "expected"), FLOOR_CASES_WITHOUT_EVIDENCE)
def test_apply_floors_is_pinned_without_evidence(
    scope: str, given: tuple[float, float, float], expected: tuple[float, float, float]
) -> None:
    shim = _FloorShim(MemoryConfig(), lambda _scope: (False, False))
    floored = MemoryStore.apply_retrieval_weight_floors(
        shim, scope, RetrievalWeights(scope, *given)
    )
    assert (floored.bm25, floored.vector, floored.graph) == pytest.approx(expected)


def test_apply_floors_always_normalizes() -> None:
    """Whatever the floors do, the result is a distribution."""

    shim = _FloorShim(MemoryConfig(), assumed_evidence)
    for scope, given, _expected in FLOOR_CASES_WITH_EVIDENCE + FLOOR_CASES_WITHOUT_EVIDENCE:
        floored = MemoryStore.apply_retrieval_weight_floors(
            shim, scope, RetrievalWeights(scope, *given)
        )
        assert floored.bm25 + floored.vector + floored.graph == pytest.approx(1.0)


@pytest.fixture()
def bare_store(tmp_path: Path) -> Any:
    store = MemoryStore(MemoryConfig(db_path=tmp_path / "bare.sqlite3"))
    yield store
    store.close()


@pytest.fixture()
def evidenced_store(tmp_path: Path) -> Any:
    """A store where both evidence gates are open, so the floors engage."""

    store = MemoryStore(MemoryConfig(db_path=tmp_path / "evidenced.sqlite3"))
    first = store.create_node(
        level="trace", content="alpha", context={"scope": "global"}, embedding=[0.1] * 8
    )
    second = store.create_node(
        level="trace", content="beta", context={"scope": "global"}, embedding=[0.2] * 8
    )
    store.connect_nodes(first.id, second.id, "related")
    assert store._has_vector_evidence("global")
    assert store._has_graph_evidence("global")
    yield store
    store.close()


#: (scope, seed triple, learning rate, signals, expected) on a store with no
#: nodes and no edges: the evidence gates are shut, so this pins the additive
#: `weight + learning_rate * signal` arithmetic and the normalization alone.
UPDATE_CASES_NO_EVIDENCE = (
    ("global", (0.4, 0.4, 0.2), 0.1, {"bm25_signal": 1.0}, (0.454545, 0.363636, 0.181818)),
    ("global", (0.4, 0.4, 0.2), 0.1, {"vector_signal": 1.0}, (0.363636, 0.454545, 0.181818)),
    (
        "global",
        (0.4, 0.4, 0.2),
        0.1,
        {"bm25_signal": -1.0, "vector_signal": 0.5, "graph_signal": 0.5},
        (0.30, 0.45, 0.25),
    ),
    ("global", (0.9, 0.05, 0.05), 0.2, {"bm25_signal": 1.0}, (0.916667, 0.041667, 0.041667)),
    (
        "project:t",
        (0.7, 0.3, 0.0),
        0.05,
        {"vector_signal": 1.0, "bm25_signal": -1.0},
        (0.65, 0.35, 0.0),
    ),
    ("project:t", (0.1, 0.85, 0.05), 0.2, {"vector_signal": 1.0}, (0.10, 0.875, 0.025)),
    ("session:t", (0.8, 0.2, 0.0), 0.05, {"graph_signal": 1.0}, (0.761905, 0.190476, 0.047619)),
)

#: Same call on a store that *has* vector and graph evidence, so the update and
#: the floor pass compose. The second row is the load-bearing one: without the
#: gates the same input lands at 0.9167 bm25, with them it is clamped to 0.75.
UPDATE_CASES_WITH_EVIDENCE = (
    ("global", (0.4, 0.4, 0.2), 0.1, {"bm25_signal": 1.0}, (0.454545, 0.363636, 0.181818)),
    ("global", (0.9, 0.05, 0.05), 0.2, {"bm25_signal": 1.0}, (0.75, 0.20, 0.05)),
    ("global", (0.0, 1.0, 0.0), 0.2, {"vector_signal": 1.0}, (0.10, 0.85, 0.05)),
)


@pytest.mark.parametrize(
    ("scope", "seed", "learning_rate", "signals", "expected"), UPDATE_CASES_NO_EVIDENCE
)
def test_update_retrieval_weights_is_pinned_without_evidence(
    bare_store: Any,
    scope: str,
    seed: tuple[float, float, float],
    learning_rate: float,
    signals: dict[str, float],
    expected: tuple[float, float, float],
) -> None:
    bare_store.set_retrieval_weights(
        scope, bm25=seed[0], vector=seed[1], graph=seed[2], learning_rate=learning_rate
    )
    updated = bare_store.update_retrieval_weights(scope, **signals)
    assert (updated.bm25, updated.vector, updated.graph) == pytest.approx(expected, abs=1e-6)
    assert updated.learning_rate == pytest.approx(learning_rate)


@pytest.mark.parametrize(
    ("scope", "seed", "learning_rate", "signals", "expected"), UPDATE_CASES_WITH_EVIDENCE
)
def test_update_retrieval_weights_is_pinned_with_evidence(
    evidenced_store: Any,
    scope: str,
    seed: tuple[float, float, float],
    learning_rate: float,
    signals: dict[str, float],
    expected: tuple[float, float, float],
) -> None:
    evidenced_store.set_retrieval_weights(
        scope, bm25=seed[0], vector=seed[1], graph=seed[2], learning_rate=learning_rate
    )
    updated = evidenced_store.update_retrieval_weights(scope, **signals)
    assert (updated.bm25, updated.vector, updated.graph) == pytest.approx(expected, abs=1e-6)


def test_update_retrieval_weights_persists_what_it_returns(bare_store: Any) -> None:
    bare_store.set_retrieval_weights("global", bm25=0.4, vector=0.4, graph=0.2)
    updated = bare_store.update_retrieval_weights("global", vector_signal=1.0)
    reread = bare_store.get_retrieval_weights("global")
    assert (reread.bm25, reread.vector, reread.graph) == pytest.approx(
        (updated.bm25, updated.vector, updated.graph)
    )


def test_method_signals_stays_the_proportional_rule() -> None:
    """`feedback._method_signals` is explicitly out of scope for this goal."""

    result = SimpleNamespace(bm25_score=0.2, vector_score=0.6, graph_score=0.2)
    assert _method_signals(result, 1.0) == pytest.approx(
        {"bm25": 0.2, "vector": 0.6, "graph": 0.2}
    )
    assert _method_signals(result, -0.5) == pytest.approx(
        {"bm25": -0.1, "vector": -0.3, "graph": -0.1}
    )
    empty = SimpleNamespace(bm25_score=0.0, vector_score=0.0, graph_score=0.0)
    assert _method_signals(empty, 1.0) == {"bm25": 0.0, "vector": 0.0, "graph": 0.0}
    negative = SimpleNamespace(bm25_score=-0.4, vector_score=1.0, graph_score=0.0)
    assert _method_signals(negative, 1.0) == pytest.approx(
        {"bm25": 0.0, "vector": 1.0, "graph": 0.0}
    )


# ---------------------------------------------------------------------------
# 2. Config defaults
# ---------------------------------------------------------------------------


def test_default_retrieval_weights_are_distributions() -> None:
    for family, weights in DEFAULT_RETRIEVAL_WEIGHTS.items():
        total = weights.bm25 + weights.vector + weights.graph
        assert total == pytest.approx(1.0), family
        assert min(weights.bm25, weights.vector, weights.graph) >= 0.0, family


# ``DEFAULT_RETRIEVAL_WEIGHTS`` is the only value this goal may change, and it
# may only change it if the A/B said so. Both branches are asserted below, so
# the pair is meaningful whichever way the experiment came out: a silent edit
# fails one, a silently discarded win fails the other.


def test_config_defaults_when_the_report_kept_the_incumbent(report: Any) -> None:
    change = report["config_change"]
    if change["changed"]:
        pytest.skip("the report adopted a new triple; covered by the adoption test")
    # Nothing was adopted, so the pre-goal defaults must still be here verbatim.
    assert [
        (family, weights.bm25, weights.vector, weights.graph)
        for family, weights in sorted(DEFAULT_RETRIEVAL_WEIGHTS.items())
    ] == [
        ("default", 1.0, 0.0, 0.0),
        ("global", 0.4, 0.4, 0.2),
        ("project", 0.7, 0.3, 0.0),
        ("session", 0.8, 0.2, 0.0),
    ]


def test_config_defaults_when_the_report_adopted_a_triple(report: Any) -> None:
    change = report["config_change"]
    if not change["changed"]:
        pytest.skip("the report kept the incumbent; covered by the incumbent test")
    triple = change["triple"]
    shim = _FloorShim(MemoryConfig(), assumed_evidence)
    for family in change["families"]:
        weights = DEFAULT_RETRIEVAL_WEIGHTS[family]
        assert (weights.bm25, weights.vector, weights.graph) == pytest.approx(
            (triple["bm25"], triple["vector"], triple["graph"])
        ), family
        # A published default must survive the floor pass unchanged, otherwise
        # the number in the report is not the number a new scope is seeded with.
        floored = MemoryStore.apply_retrieval_weight_floors(
            shim,
            family,
            RetrievalWeights(family, weights.bm25, weights.vector, weights.graph),
        )
        assert (floored.bm25, floored.vector, floored.graph) == pytest.approx(
            (weights.bm25, weights.vector, weights.graph)
        ), family
        assert _scope_family(family) == family


# ---------------------------------------------------------------------------
# 3. The published report keeps its own promises
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    assert REPORT_PATH.exists(), f"{REPORT_PATH} has not been generated"
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def markdown() -> str:
    assert MARKDOWN_PATH.exists(), f"{MARKDOWN_PATH} has not been generated"
    return MARKDOWN_PATH.read_text(encoding="utf-8")


def test_report_publishes_three_named_time_slices(report: dict[str, Any]) -> None:
    split = report["split"]
    assert split["c1"] < split["c2"], "C1 must be strictly before C2"
    assert sorted(split["slices"]) == sorted(SLICE_NAMES)


def test_time_slices_are_non_empty_and_holdout_is_large_enough(
    report: dict[str, Any],
) -> None:
    slices = report["split"]["slices"]
    for name in SLICE_NAMES:
        assert slices[name]["count"] > 0, f"{name} slice is empty"
        assert len(slices[name]["query_ids"]) == slices[name]["count"], name
    assert slices["holdout"]["count"] >= MIN_HOLDOUT_EVENTS


def test_time_slices_are_genuinely_disjoint(report: dict[str, Any]) -> None:
    """Set disjointness, not just a description of the cutting rule."""

    slices = report["split"]["slices"]
    ids = {name: set(slices[name]["query_ids"]) for name in SLICE_NAMES}
    for name in SLICE_NAMES:
        assert len(ids[name]) == slices[name]["count"], f"{name} repeats a query id"
    assert ids["train"] & ids["eval"] == set()
    assert ids["train"] & ids["holdout"] == set()
    assert ids["eval"] & ids["holdout"] == set()


def test_time_slices_respect_the_published_cutoffs(report: dict[str, Any]) -> None:
    """The slices really are cut on time, in the stated order."""

    split = report["split"]
    slices = split["slices"]
    assert slices["train"]["max_created_at"] <= split["c1"]
    assert slices["eval"]["min_created_at"] > split["c1"]
    assert slices["eval"]["max_created_at"] <= split["c2"]
    assert slices["holdout"]["min_created_at"] > split["c2"]
    assert slices["train"]["max_created_at"] < slices["eval"]["min_created_at"]
    assert slices["eval"]["max_created_at"] < slices["holdout"]["min_created_at"]


def test_slice_digests_match_their_published_ids(report: dict[str, Any]) -> None:
    slices = report["split"]["slices"]
    for name in SLICE_NAMES:
        recomputed = hashlib.sha256(
            "\n".join(sorted(slices[name]["query_ids"])).encode("utf-8")
        ).hexdigest()
        assert recomputed == slices[name]["query_ids_sha256"], name


def test_every_slice_was_actually_scored(report: dict[str, Any]) -> None:
    """Each stage reports the same event count as its slice."""

    slices = report["split"]["slices"]
    stages = report["stages"]
    for name in SLICE_NAMES:
        assert stages[name]["events"] == slices[name]["count"], name
        leaderboard = stages[name]["leaderboard"]
        assert leaderboard, f"{name} has no scored schemes"
        for entry in leaderboard:
            assert entry["metrics"]["events"] == slices[name]["count"], (name, entry["scheme"])


def test_holdout_compares_the_winner_against_the_incumbent(report: dict[str, Any]) -> None:
    decision = report["decision"]
    holdout = {entry["scheme"]: entry["metrics"] for entry in report["stages"]["holdout"]["leaderboard"]}
    incumbent_name = report["incumbent"]["scheme"]
    assert incumbent_name in holdout
    assert decision["winner"] in holdout
    bar = decision["generalization_bar"]
    assert bar["hit@5"]["incumbent"] == pytest.approx(holdout[incumbent_name]["hit@5"])
    assert bar["mrr"]["incumbent"] == pytest.approx(holdout[incumbent_name]["mrr"])
    assert bar["hit@5"]["winner"] == pytest.approx(holdout[decision["winner"]]["hit@5"])
    assert bar["mrr"]["winner"] == pytest.approx(holdout[decision["winner"]]["mrr"])


def test_the_generalization_bar_is_applied_as_stated(report: dict[str, Any]) -> None:
    """Adoption implies both holdout conditions; rejection implies no config change."""

    decision = report["decision"]
    bar = decision["generalization_bar"]
    for key in ("hit@5", "mrr"):
        assert bar[key]["passes"] == (bar[key]["winner"] >= bar[key]["incumbent"]), key
    clears = bar["hit@5"]["passes"] and bar["mrr"]["passes"]
    assert decision["clears_bar"] == clears
    if decision["adopted"]:
        assert clears, "a scheme was adopted without clearing the bar"
        assert decision["winner"] != report["incumbent"]["scheme"]
    else:
        assert not report["config_change"]["changed"] or decision["adopted"]


def test_winner_is_the_eval_leader(report: dict[str, Any]) -> None:
    """Selection happened on eval, not on holdout."""

    eval_leaderboard = report["stages"]["eval"]["leaderboard"]
    assert eval_leaderboard[0]["scheme"] == report["decision"]["winner"]
    holdout_leader = report["stages"]["holdout"]["leaderboard"][0]["scheme"]
    if holdout_leader != report["decision"]["winner"]:
        # Explicitly allowed and the whole point of a holdout: the eval winner
        # need not lead on holdout. What must not happen is the report quietly
        # renaming the holdout leader as the winner.
        assert report["decision"]["winner"] == eval_leaderboard[0]["scheme"]


def test_report_records_the_required_comparison_schemes(report: dict[str, Any]) -> None:
    """Incumbent, floor defaults, uniform and a weight grid were all compared."""

    scored = {entry["scheme"] for entry in report["stages"]["train"]["leaderboard"]}
    assert report["incumbent"]["scheme"] in scored
    assert "config_defaults_current" in scored
    assert "uniform" in scored
    assert sum(1 for name in scored if name.startswith("explicit_b")) >= 20
    assert sum(1 for name in scored if name.startswith("config_b")) >= 20
    assert sum(1 for name in scored if name.startswith("trajectory_")) >= 1


def test_report_states_the_live_database_was_not_written_by_the_experiment(
    report: dict[str, Any],
) -> None:
    live = report["provenance"]["live_database"]
    assert live["mutated_by_this_experiment"] is False
    assert live["retrieval_weights_written_by_this_experiment"] is False


def test_report_does_not_claim_the_live_weights_are_frozen(report: dict[str, Any]) -> None:
    """A "nothing was mutated" claim must not be read as "nothing changed".

    The MCP server's own learning loop keeps moving the live rows while this
    work runs. When the report observed the live database, it has to publish
    that drift rather than let the read-only claim imply the rows stood still.
    """

    observation = report["provenance"]["live_database"]["observation"]
    if not observation.get("observed"):
        pytest.skip("this run was not given --observe-live-db")
    assert "mode=ro" in observation["opened"]
    assert observation["scopes"] > 0
    assert observation["scopes_drifted_since_snapshot"] == len(observation["drift"])
    for scope, entry in observation["drift"].items():
        if entry.get("snapshot") is None:
            continue
        assert entry["l1_distance"] > 0.0, scope


def test_report_floors_block_matches_the_configured_floors(report: dict[str, Any]) -> None:
    for family, published in report["floors"].items():
        floors = DEFAULT_RETRIEVAL_POLICY_FLOORS[family]
        assert published == {
            "bm25_max": floors.bm25_max,
            "vector_min": floors.vector_min,
            "graph_min": floors.graph_min,
            "bm25_min": floors.bm25_min,
        }


def test_markdown_states_the_rule_before_the_holdout_numbers(markdown: str) -> None:
    """The selection rule has to be readable before the holdout table."""

    rule_at = markdown.index("Selection rule, fixed before the holdout slice was read")
    holdout_at = markdown.index("Stage 3 — HOLDOUT")
    assert rule_at < holdout_at
    assert markdown.index("Stage 2 — EVAL") < holdout_at
    assert markdown.index("Stage 1 — TRAIN") < markdown.index("Stage 2 — EVAL")


def test_markdown_carries_the_operator_runbook(markdown: str) -> None:
    assert "Operator runbook" in markdown
    assert "NOT executed" in markdown
    assert "set_retrieval_weights" in markdown


def test_markdown_reports_every_slice_and_the_per_scope_breakdown(
    markdown: str, report: dict[str, Any]
) -> None:
    for name in SLICE_NAMES:
        assert str(report["split"]["slices"][name]["count"]) in markdown, name
    assert "Per-scope breakdown on holdout" in markdown
    assert report["stages"]["holdout"]["per_scope"], "no per-scope breakdown was published"
