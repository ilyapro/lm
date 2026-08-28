"""Guards for the cold-start fit v3 and for the artifact it publishes.

These tests never open the store.  The arithmetic tests run on synthetic rows,
and the artifact tests read the committed JSON, so the suite stays fast and
stays honest about what it is checking: that the fit obeys the sealed plan and
that the published record cannot quietly drift away from it.

The fit under test recorded a NEGATIVE RESULT at decision step 6: the revised
ordering does not clear ``guard_c_warm_non_degradation`` on coldstart_eval.
That is a PRE-HOLDOUT gate, so the sealed partition was never opened and
``holdout_evaluations`` is 0.  A large part of this file pins the shape of that
record.  A negative result that could be quietly upgraded to a positive one by
an edit is not a record of anything -- and the reverse matters too: the record
must keep publishing the parameters step 4 really did seal, the frontier, and
every three-arm reading, or the failure stops being falsifiable.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pytest

from living_memory import recall_map

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = REPO_ROOT / "artifacts/recall-map/relevance/policy.json"
PREREG_PATH = REPO_ROOT / "artifacts/recall-map/relevance/coldstart-prereg-v3.json"
MANIFEST_PATH = REPO_ROOT / "artifacts/recall-map/relevance/dataset-manifest.json"
ANALYSIS_PATH = REPO_ROOT / "artifacts/recall-map/relevance/feature-analysis.json"


def _load_module():
    path = REPO_ROOT / "scripts/recall_map_coldstart_fit.py"
    spec = importlib.util.spec_from_file_location("recall_map_coldstart_fit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fit = _load_module()


@pytest.fixture(scope="module")
def policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prereg() -> dict:
    return json.loads(PREREG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def revision(policy: dict) -> dict:
    return policy["coldstart_revision"]


_ROW_SEQUENCE = itertools.count()


def _row(**kwargs):
    index = next(_ROW_SEQUENCE)
    base = {
        "split": "train",
        "cold": False,
        "consumed": False,
        "level": "trace",
        "matured": 5,
        "matured_consumed": 2,
        "trailing_nonconsumed": 1,
        "component_id": "c",
        # The label-free identity fields the component cap orders on, and the
        # instant the era cut reads.  Distinct per row so a synthetic cohort
        # behaves like a real one under ``cap_survivors``.
        "source": "local",
        "event_id": f"e{index}",
        "rank": index,
        "decided_at": "2026-08-10T00:00:00Z",
        "era": fit.ERA_PREEXISTING,
        "scores": {
            "score": 0.5,
            "trigger_score": 0.0,
            "bm25_score": 0.0,
            "vector_score": 0.0,
            "graph_score": 0.0,
        },
    }
    scores = kwargs.pop("scores", None)
    base.update(kwargs)
    if scores:
        base["scores"] = {**base["scores"], **scores}
    return base


def _unseal():
    return fit.SealBreak(reason="test", parameter_sha256="0" * 64)


# ---------------------------------------------------------------------------
# The seal
# ---------------------------------------------------------------------------


def test_prereg_seal_verifies_under_the_rendering_it_names(prereg):
    """The seal is taken over judge.canonical_json, i.e. indent=2, not compact."""

    assert fit.verify_prereg_seal(prereg) == fit.EXPECTED_PLAN_SHA256


def test_prereg_seal_uses_indented_not_compact_rendering(prereg):
    subset = {key: prereg[key] for key in prereg["plan_sha256_over"]}
    compact = fit.sha256_text(fit.canonical_json(subset))
    indented = fit.sha256_text(fit.canonical_json_indented(subset))
    assert indented == prereg["plan_sha256"]
    assert compact != prereg["plan_sha256"], "the two renderings must not be confused"


def test_a_tampered_prereg_is_refused(prereg):
    tampered = json.loads(json.dumps(prereg))
    tampered["verdict_rule"]["generalization_threshold"] = 1.0
    with pytest.raises(SystemExit):
        fit.verify_prereg_seal(tampered)


def test_the_fit_is_bound_to_v3_and_names_the_plan_it_supersedes(prereg):
    assert fit.EXPECTED_PLAN_SHA256 == prereg["plan_sha256"]
    assert fit.DEFAULT_PREREG.name == "coldstart-prereg-v3.json"
    assert prereg["supersedes"]["plan_sha256"] == fit.SUPERSEDED_PLAN_SHA256
    assert fit.EXPECTED_PLAN_SHA256 != fit.SUPERSEDED_PLAN_SHA256


# ---------------------------------------------------------------------------
# The sealed-partition guard
# ---------------------------------------------------------------------------


def test_sealed_partition_cannot_be_opened_without_the_capability():
    splits = fit.SealedSplits(_rows={"train": [_row()], "eval": [], "holdout": [_row()]})
    assert splits.open("train")
    with pytest.raises(SystemExit):
        splits.open(fit.SEALED_SPLIT)
    with pytest.raises(SystemExit):
        splits.open(fit.SEALED_SPLIT, unseal="unseal")  # a string is not authority
    assert splits.sealed_reads == 0


def test_the_capability_is_what_opens_the_sealed_partition():
    splits = fit.SealedSplits(_rows={"train": [], "eval": [], "holdout": [_row()]})
    assert splits.open(fit.SEALED_SPLIT, unseal=_unseal())
    assert splits.sealed_reads == 1


def test_a_second_holdout_read_is_refused():
    """``single_holdout_pass.forbidden``: a second read for any reason."""

    splits = fit.SealedSplits(_rows={"train": [], "eval": [], "holdout": [_row()]})
    splits.open(fit.SEALED_SPLIT, unseal=_unseal())
    with pytest.raises(SystemExit):
        splits.open(fit.SEALED_SPLIT, unseal=_unseal())
    assert splits.sealed_reads == 1


def test_counting_sealed_rows_is_not_a_read():
    """Step 2 must audit split sizes; that is not an evaluation."""

    splits = fit.SealedSplits(
        _rows={"train": [], "eval": [], "holdout": [_row(cold=True), _row()]}
    )
    assert splits.counts(fit.SEALED_SPLIT) == {
        "items": 2,
        "consumed": 0,
        "cold_candidates": 1,
        "cold_consumed": 0,
    }
    assert splits.sealed_reads == 0


# ---------------------------------------------------------------------------
# Rank calibration
# ---------------------------------------------------------------------------


def test_rank_uniform_is_midrank_and_centred():
    table = fit.StepFunction.fit("t", [0.0, 1.0, 2.0, 3.0])
    # Four distinct values, each 25% of the rows, so each is an atom.
    assert [round(u, 6) for _value, u in table.atoms] == [-0.375, -0.125, 0.125, 0.375]


def test_a_tie_block_shares_one_midrank():
    """The available-but-empty history composite is exactly such a block."""

    table = fit.StepFunction.fit("t", [0.0] * 50 + [1.0] * 50)
    atoms = dict(table.atoms)
    assert atoms[0.0] == pytest.approx(-0.25)
    assert atoms[1.0] == pytest.approx(0.25)


def test_the_calibration_is_monotone():
    values = [float(index) for index in range(1000)]
    table = fit.StepFunction.fit("t", values)
    us = [table.u(value) for value in values]
    assert us == sorted(us)


def test_rank_normal_has_its_guaranteed_percentiles_on_the_fitting_split():
    """p50 -> 0.0 and p84.13 -> ~1.0 are properties of the transform, not the data."""

    values = [float(index) for index in range(10_000)]
    table = fit.StepFunction.fit("t", values)
    assert fit.rank_normal(table, values[5000]) == pytest.approx(0.0, abs=0.02)
    assert fit.rank_normal(table, values[8413]) == pytest.approx(1.0, abs=0.02)


def test_rank_normal_is_clipped_to_the_registered_range():
    table = fit.StepFunction.fit("t", [float(i) for i in range(100)])
    assert fit.rank_normal(table, -1e9) >= -fit.RANK_NORMAL_CLIP
    assert fit.rank_normal(table, 1e9) <= fit.RANK_NORMAL_CLIP


# ---------------------------------------------------------------------------
# The two arms
# ---------------------------------------------------------------------------


def _scorer(train_rows):
    fit_block = json.loads(POLICY_PATH.read_text(encoding="utf-8"))["selected_policy"]["fit"]
    return fit.fit_tables(
        train_rows, fit_block["standardization_means"], fit_block["scales"]
    )


def test_the_node_intrinsic_arm_is_bounded_however_many_arms_there_are():
    """``N = (L + H) / 2`` cannot exceed [-0.5, +0.5]; that IS the v2 fix."""

    rows = [
        _row(level=level, matured=matured, matured_consumed=0, trailing_nonconsumed=0)
        for level in ("trace", "concept", "schema")
        for matured in (0, 1, 5, 50)
    ]
    scorer = _scorer(rows)
    for row in rows:
        assert -0.5 <= scorer.node_intrinsic(row) <= 0.5


def test_a_cold_row_still_takes_the_available_but_empty_history_arm():
    """M=C=K=0 is ``known(matured=0)``, not ``unavailable``; the raw arm is unchanged."""

    fit_block = json.loads(POLICY_PATH.read_text(encoding="utf-8"))["selected_policy"]["fit"]
    cold = _row(cold=True, matured=0, matured_consumed=0, trailing_nonconsumed=0)
    raw = fit.history_raw(
        cold, fit_block["standardization_means"], fit_block["scales"]
    )
    assert round(raw, 4) == -3.6134


def test_the_history_ordering_is_bit_for_bit_the_frozen_one():
    """``rank_uniform_centered`` is strictly monotone, so no pair is re-ranked."""

    fit_block = json.loads(POLICY_PATH.read_text(encoding="utf-8"))["selected_policy"]["fit"]
    means, scales = fit_block["standardization_means"], fit_block["scales"]
    rows = [
        _row(matured=m, matured_consumed=c, trailing_nonconsumed=0)
        for m in (0, 1, 3, 9, 40)
        for c in (0, 1)
        if c <= m
    ]
    scorer = _scorer(rows)
    raws = sorted(rows, key=lambda row: fit.history_raw(row, means, scales))
    calibrated = sorted(rows, key=scorer.history_u)
    assert [fit.history_raw(r, means, scales) for r in raws] == [
        fit.history_raw(r, means, scales) for r in calibrated
    ]


def test_there_is_no_branch_on_coldness_anywhere_in_the_scorer():
    """Two rows identical but for the cold FLAG must score identically."""

    rows = [_row(matured=m) for m in (0, 1, 2, 3, 4, 5)]
    scorer = _scorer(rows)
    warm = _row(cold=False, matured=0, matured_consumed=0, trailing_nonconsumed=0)
    cold = _row(cold=True, matured=0, matured_consumed=0, trailing_nonconsumed=0)
    assert scorer.score(warm) == scorer.score(cold)


def test_an_unscoreable_family_is_inadmissible_not_neutral():
    """``missing_value_rule``: fail closed, and both shapes share the denominator."""

    rows = [_row(scores={"score": value}) for value in (0.0, 1.0, 2.0)]
    scorer = _scorer(rows)
    broken = _row(scores={"bm25_score": None})
    assert fit.family_unscoreable(broken) is True
    assert scorer.score(broken) is None
    assert scorer.ablation_score(broken) is None


def test_the_ablation_is_the_same_shape_with_the_family_neutralised():
    rows = [_row(scores={"score": value}) for value in (0.0, 1.0, 2.0)]
    scorer = _scorer(rows)
    row = _row()
    assert scorer.ablation_score(row) == pytest.approx(scorer.node_intrinsic(row))


# ---------------------------------------------------------------------------
# Component-aware estimation
# ---------------------------------------------------------------------------


def test_the_estimation_cap_is_derived_from_the_published_grid_not_chosen(manifest):
    """``estimation_cap.selection_rule``: floor first, then the largest cap.

    The rule reads the manifest's own cap table and no row, and it consults no
    direction, no sign and no label -- which is what stops a cap chosen after
    seeing which signs failed from being the thing that unfails them.
    """

    selection = fit.select_estimation_cap(manifest["coldstart"])
    assert selection["estimation_cap"] == fit.ESTIMATION_CAP == 50
    assert selection["agrees_with_the_registered_value"] is True
    assert selection["read_no_row"] is True
    assert selection["reads_no_direction_sign_or_label"] is True
    per_cap = selection["effective_components_per_split_per_cap"]
    assert per_cap["50"]["meets_the_whole_split_floor"] is True
    assert per_cap["100"]["meets_the_whole_split_floor"] is False, "50 is the LARGEST feasible"


def test_substituting_any_floor_in_the_stated_range_returns_the_same_cap(manifest):
    """The plan claims the choice is insensitive; check it rather than believe it."""

    table = manifest["coldstart"]["enlargement"]["cap_feasibility"]["table"]

    def largest_feasible(floor: float) -> int | None:
        feasible = [
            cap
            for cap in fit.CAP_GRID
            if all(
                table[str(cap)][name]["effective_components"] >= floor
                for name in ("train", "eval", "holdout")
            )
        ]
        return max(feasible) if feasible else None

    for floor in range(77, 114):
        assert largest_feasible(floor) == 50, floor


def test_the_cap_draw_cannot_see_a_label_a_score_or_the_cold_flag():
    """``which_rows_survive``: label-free by construction, not by intention."""

    rows = [
        _row(component_id="big", cold=index % 2 == 0, consumed=index % 3 == 0)
        for index in range(120)
    ]
    kept = fit.cap_survivors(rows, 50)
    flipped = [
        {**row, "consumed": not row["consumed"], "cold": not row["cold"]} for row in rows
    ]
    kept_after = fit.cap_survivors(flipped, 50)
    assert [row["event_id"] for row in kept] == [row["event_id"] for row in kept_after]
    assert len(kept) == 50


def test_the_cap_order_key_is_the_extractors_key_byte_for_byte():
    """A second copy of the rule would be a copy that can drift from the manifest."""

    row = _row(component_id="c9", event_id="e9", rank=3)
    expected = fit.sha256_text(
        fit.canonical_json([fit.CAP_SUBSAMPLE_NAMESPACE, "c9", "local", "e9", 3])
    )
    assert fit.cap_order_key(row) == expected


def test_the_cap_keeps_every_component_so_disjointness_is_untouched():
    rows = [
        _row(component_id=f"c{index % 7}") for index in range(400)
    ]
    kept = fit.cap_survivors(rows, 5)
    assert {row["component_id"] for row in kept} == {row["component_id"] for row in rows}
    assert len(kept) == 7 * 5


def test_caps_nest_so_the_arms_are_a_sequence_of_nested_subsamples():
    rows = [_row(component_id="one") for _ in range(300)]
    smaller = {row["event_id"] for row in fit.cap_survivors(rows, 25)}
    larger = {row["event_id"] for row in fit.cap_survivors(rows, 50)}
    assert smaller < larger


def test_effective_components_is_the_inverse_simpson_index():
    rows = [_row(component_id="a") for _ in range(75)] + [
        _row(component_id="b") for _ in range(25)
    ]
    # 1 / (0.75^2 + 0.25^2) = 1 / 0.625 = 1.6
    assert fit.effective_components(rows) == pytest.approx(1.6)


def test_the_effective_component_floors_are_the_registered_ones(prereg):
    floor = prereg["component_aware_estimation"]["effective_component_floor"]
    assert fit.EFFECTIVE_COMPONENT_FLOOR_WHOLE_SPLIT == floor["whole_split"] == 100
    assert fit.EFFECTIVE_COMPONENT_FLOOR_COLD == floor["cold_subpopulation"] == 16


def test_three_arms_are_built_and_only_the_capped_one_carries_a_verdict():
    rows = [_row(component_id=f"c{index % 3}") for index in range(90)]
    arms = fit.build_arms(rows, 10)
    assert set(arms) == set(fit.ARM_NAMES)
    assert arms["capped"].carries_the_verdict is True
    assert arms["uncapped"].carries_the_verdict is False
    assert arms["component_equalized"].carries_the_verdict is False
    assert len(arms["capped"].rows) == 30
    assert len(arms["uncapped"].rows) == 90
    assert sum(arms["component_equalized"].weights) == pytest.approx(1.0)


def test_a_weighted_arm_reports_no_exact_binomial():
    """``why_a_cap_and_not_component_equalized_weights`` (1), enforced not asserted."""

    rows = [
        _row(component_id=f"c{index % 4}", cold=True, consumed=index % 2 == 0, scores={"score": float(index)})
        for index in range(40)
    ]
    arms = fit.build_arms(rows, 5)
    table = fit.three_arm_admission(arms, lambda row: row["scores"]["score"], 20.0)
    assert table["capped"]["binomial_p"] is not None
    assert table["component_equalized"]["binomial_p"] is None
    assert "no integer counts" in table["component_equalized"][
        "binomial_not_applicable_under_weights"
    ]


# ---------------------------------------------------------------------------
# Threshold selection -- label-free, swept uncapped, tested capped
# ---------------------------------------------------------------------------


def test_the_threshold_rule_takes_the_highest_feasible_score():
    """``why_highest``: the smallest admission consistent with the floors."""

    rows = [
        _row(cold=index % 2 == 0, component_id=f"c{index}", scores={"score": float(index)})
        for index in range(100)
    ]

    def score_of(row):
        return row["scores"]["score"]

    # Every component holds one row, so the cap keeps all of them and the
    # capped test and the uncapped sweep agree -- which is what makes the
    # remaining assertions about the RULE and not about the draw.
    arms = fit.build_arms(rows, 50)
    assert len(arms["capped"].rows) == len(rows)
    result = fit.select_threshold(rows, arms, score_of)
    assert result["reads_no_label"] is True
    assert result["feasible_threshold_count"] > 0
    admitted = sum(1 for row in rows if score_of(row) >= result["threshold"])
    assert fit.OVERALL_BAND[0] <= admitted / len(rows) <= fit.OVERALL_BAND[1]
    higher = [row for row in rows if score_of(row) > result["threshold"]]
    assert len(higher) / len(rows) < fit.OVERALL_BAND[0]


def test_the_threshold_rule_reads_no_consumption_label():
    """Flipping every label must not move the selected threshold by one bit."""

    rows = [
        _row(
            cold=index % 2 == 0,
            consumed=index % 3 == 0,
            component_id=f"c{index}",
            scores={"score": float(index)},
        )
        for index in range(100)
    ]
    flipped = [{**row, "consumed": not row["consumed"]} for row in rows]

    def score_of(row):
        return row["scores"]["score"]

    original = fit.select_threshold(rows, fit.build_arms(rows, 50), score_of)
    after = fit.select_threshold(flipped, fit.build_arms(flipped, 50), score_of)
    assert original["threshold"] == after["threshold"]
    assert original["feasible_threshold_count"] == after["feasible_threshold_count"]


def test_the_qualifying_test_is_capped_even_when_the_sweep_is_not():
    """One component may not decide an admitted fraction that carries a verdict.

    ``big`` holds 200 rows and every one of them scores at the top, so on the
    RAW split the band is satisfied by that single cache key.  Under the cap
    it contributes at most ``cap`` rows and the same threshold no longer
    qualifies.  That difference IS the defect the plan exists to close.
    """

    big = [
        _row(component_id="big", cold=True, scores={"score": 100.0 + index})
        for index in range(200)
    ]
    rest = [
        _row(component_id=f"c{index}", cold=True, scores={"score": index * 0.1})
        for index in range(200)
    ]
    rows = big + rest

    def score_of(row):
        return row["scores"]["score"]

    arms = fit.build_arms(rows, 10)
    capped = fit.AdmissionCurve(arms["capped"], score_of)
    uncapped = fit.AdmissionCurve(arms["uncapped"], score_of)
    assert uncapped.at(100.0)[0] == pytest.approx(0.5)
    assert capped.at(100.0)[0] < 0.1
    report = fit.frontier_report(rows, arms, score_of, "synthetic")
    assert set(report["arms"]) == set(fit.ARM_NAMES)
    assert report["feasible_threshold_count"] == report["arms"]["capped"][
        "feasible_threshold_count"
    ]


def test_no_feasible_threshold_is_reported_as_zero_not_as_a_fallback():
    """``if_no_such_threshold_exists``: HALT, never a nearest-miss threshold."""

    rows = [_row(cold=True) for _ in range(10)]
    result = fit.select_threshold(rows, fit.build_arms(rows, 50), lambda row: 1.0)
    assert result["feasible_threshold_count"] == 0
    assert result["threshold"] is None


def test_the_ablation_threshold_is_volume_matched_and_ties_break_higher():
    rows = [
        _row(cold=True, component_id=f"c{i}", scores={"score": float(i)}) for i in range(10)
    ]
    scorer = _scorer(rows)
    ablation = fit.select_ablation_threshold(
        rows, fit.build_arms(rows, 50), scorer, registered_cold_admitted=3
    )
    assert ablation["cold_admitted_on_train"] >= 3
    assert "higher threshold" in ablation["rule"]
    assert ablation["counted_on"].startswith("the component-capped")


def test_binomial_tail_matches_a_known_value():
    # P(X >= 8 | n=10, p=0.5) = 56/1024
    assert fit.binomial_tail(8, 10, 0.5) == pytest.approx(56 / 1024)
    assert fit.binomial_tail(0, 10, 0.3) == pytest.approx(1.0)


def test_cold_auc_counts_ties_as_a_half():
    scored = [(1.0, _row(consumed=True)), (1.0, _row(consumed=False))]
    assert fit.cold_auc(scored) == pytest.approx(0.5)


def test_weighted_cold_auc_reduces_to_the_unweighted_one_at_uniform_weights():
    scored = [
        (float(index), _row(consumed=index % 3 == 0)) for index in range(30)
    ]
    assert fit.cold_auc(scored, [1.0] * 30) == pytest.approx(fit.cold_auc(scored))


# ---------------------------------------------------------------------------
# Direction checks
# ---------------------------------------------------------------------------


def test_the_registered_statistic_is_an_affine_image_of_the_auc():
    """``direction_constraint.statistic``: mean difference of rank-uniform == AUC - 0.5.

    This identity is what lets the published sign be cross-checked by anyone
    with a rank routine, and it is the reason the estimator cannot be moved by a
    handful of extreme rows.
    """

    rows = [
        _row(consumed=index % 3 == 0, scores={"score": float(index)})
        for index in range(60)
    ]
    table = fit.StepFunction.fit("score", [float(index) for index in range(60)])
    difference = fit.weighted_association(rows, None, "score", table)
    scored = [(float(index), row) for index, row in enumerate(rows)]
    assert difference == pytest.approx(fit.cold_auc(scored) - 0.5, abs=0.01)


def test_direction_check_is_indeterminate_when_cold_rows_are_too_few(prereg):
    """Too small to measure must never read as 'passed'."""

    rows = [_row(cold=True, consumed=index % 2 == 0) for index in range(10)]
    arm = fit.build_arms(rows, 50)["capped"]
    report = fit.arm_direction_reading(arm, _scorer(rows), prereg)
    assert report["cold_check_measurable"] is False
    assert report["members"]["result_score"]["verdict"] == "indeterminate"


def test_an_indeterminate_cold_check_blocks_and_never_passes(prereg):
    rows = [_row(cold=True, consumed=index % 2 == 0) for index in range(10)]
    arms = fit.build_arms(rows, 50)
    three = fit.three_arm_directions(arms, _scorer(rows), prereg)
    verdict = fit.binding_direction_verdict(three)
    assert verdict["holds"] is False
    assert verdict["indeterminate_is_not_a_pass"] is True
    assert set(verdict["failing_members"]) == set(verdict["member_verdicts"])


def test_direction_check_fails_on_a_flipped_sign(prereg):
    clause = prereg["direction_constraint"]["cold_subpopulation_clause"]
    n = int(clause["minimum_cold_rows_for_a_direction_check"]) * 2
    # consumed rows carry LOWER score, so a required-positive member must fail
    rows = [
        _row(cold=True, consumed=True, scores={"score": 0.0}) for _ in range(n // 2)
    ] + [_row(cold=True, consumed=False, scores={"score": 1.0}) for _ in range(n // 2)]
    arm = fit.build_arms(rows, 500)["capped"]
    report = fit.arm_direction_reading(arm, _scorer(rows), prereg)
    assert report["cold_check_measurable"] is True
    assert report["members"]["result_score"]["verdict"] == "fail"


def test_only_the_capped_arm_decides_a_direction_verdict(prereg):
    """A reported arm may neither rescue a failing verdict nor overturn a passing one."""

    clause = prereg["direction_constraint"]["cold_subpopulation_clause"]
    n = int(clause["minimum_cold_rows_for_a_direction_check"]) * 2
    rows = [
        _row(cold=True, consumed=True, scores={"score": 0.0}) for _ in range(n // 2)
    ] + [_row(cold=True, consumed=False, scores={"score": 1.0}) for _ in range(n // 2)]
    three = fit.three_arm_directions(fit.build_arms(rows, 500), _scorer(rows), prereg)
    verdict = fit.binding_direction_verdict(three)
    assert verdict["decided_on"] == fit.VERDICT_ARM == "capped"
    assert verdict["holds"] is False
    assert "never a gate" in verdict["the_reported_arms_carry_no_verdict"]


# ---------------------------------------------------------------------------
# The published artifact: bindings the wire sibling must not break
# ---------------------------------------------------------------------------


def test_revision_id_was_bumped_off_the_frozen_policy(revision):
    """Stated as the INVARIANT, not as the literal ``directional-zsum-r1``.

    coldstart-wire-v3 promotes this revision into ``selected_policy`` and moves
    ``recall_map.RELEVANCE_POLICY_ID`` in the same commit, so a hard-coded id
    here would go red the moment that lands while saying nothing about what
    actually matters: that the revision is not already the in-force policy.
    """

    assert revision["id"] != recall_map.RELEVANCE_POLICY_ID
    assert revision["id"] == fit.REVISION_ID


def test_revision_digest_matches_its_binding(policy, revision):
    digest = fit.sha256_text(fit.canonical_json(revision))
    assert policy["bindings"]["canonical_subobjects"]["coldstart_revision_sha256"] == digest


def test_selected_policy_and_its_digest_still_agree(policy):
    """The in-force policy is untouched, so its binding must still hold.

    Stated as the INVARIANT rather than as the literal ``directional-zsum-r1``:
    the sibling coldstart-wire-v2 promotes a revision into ``selected_policy``
    and moves ``recall_map.RELEVANCE_POLICY_ID`` in the same commit, so a
    hard-coded id here would go red the moment that lands while saying nothing
    about what actually matters -- that the artifact and the code agree.
    """

    digest = fit.sha256_text(fit.canonical_json(policy["selected_policy"]))
    assert policy["bindings"]["canonical_subobjects"]["selected_policy_sha256"] == digest
    assert policy["selected_policy"]["id"] == recall_map.RELEVANCE_POLICY_ID
    assert digest == recall_map.RELEVANCE_POLICY_DIGEST


def test_this_run_left_the_in_force_policy_byte_identical(policy):
    assert (
        policy["bindings"]["canonical_subobjects"]["selected_policy_sha256_unchanged"] is True
    )
    assert revision_is_not_in_force(policy)


def revision_is_not_in_force(policy: dict) -> bool:
    return policy["selected_policy"]["id"] != policy["coldstart_revision"]["id"]


def test_dataset_manifest_binding_is_rebound_to_the_republished_manifest(policy):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert policy["bindings"]["dataset_manifest"]["canonical_json_sha256"] == fit.sha256_text(
        fit.canonical_json(manifest)
    )
    assert policy["bindings"]["dataset_manifest"]["stored_bytes_sha256"] == fit.sha256_bytes(
        MANIFEST_PATH
    )
    assert policy["bindings"]["feature_analysis"]["stored_bytes_sha256"] == fit.sha256_bytes(
        ANALYSIS_PATH
    )


def test_holdout_boundary_is_left_fail_closed(policy):
    """The post-deployment population is a different thing; do not disturb it."""

    digest = fit.sha256_text(fit.canonical_json(policy["holdout_boundary"]))
    assert policy["bindings"]["canonical_subobjects"]["holdout_boundary_sha256"] == digest


def test_revision_never_names_the_reserved_tokens(revision):
    """``scripts/recall_map_relevance_eval.py`` rejects these before any path."""

    text = json.dumps(revision)
    for marker in ("candidate_holdout", "candidate-holdout", "relevance/field"):
        assert marker not in text
    assert revision["field_subtree_accessed"] is False
    assert revision["post_deployment_holdout_accessed"] is False


# ---------------------------------------------------------------------------
# The published artifact: the dataset the parent goal requires
# ---------------------------------------------------------------------------


def test_every_split_is_cited_with_a_non_zero_cold_share(revision, manifest):
    block = manifest["coldstart"]
    for key, name in (
        ("train_set", "train"),
        ("eval_set", "eval"),
        ("holdout_set", "holdout"),
    ):
        cited = revision["dataset"][key]
        published = block["splits"][name]
        assert cited["ref"].endswith(f"#coldstart.splits.{name}")
        assert cited["cold_candidate_share"] > 0.0
        assert cited["cold_candidates"] == published["cold_candidates"]
        assert cited["items"] == published["items"]
        assert cited["membership_digest"] == published["membership_digest"]


def test_every_split_clears_the_registered_cold_minimum(manifest, prereg):
    minimum = int(prereg["split_protocol"]["minimum_cold_candidates_per_split"])
    assert minimum == 400, "the minimum did not move"
    for name, split in manifest["coldstart"]["splits"].items():
        assert split["cold_candidates"] >= minimum, name


def test_the_splits_are_pairwise_component_disjoint(manifest):
    splits = manifest["coldstart"]["splits"]
    for a, b in (("train", "eval"), ("train", "holdout"), ("eval", "holdout")):
        assert not set(splits[a]["components"]) & set(splits[b]["components"]), f"{a}/{b}"


def test_no_already_read_v1_event_reaches_the_holdout(manifest):
    assert manifest["coldstart"]["split"]["holdout_contamination"]["intersection"] == 0


def test_every_split_reports_effective_components(revision, manifest):
    """A generalization claim rests on independent units, not on row counts."""

    for key, name in (
        ("train_set", "train"),
        ("eval_set", "eval"),
        ("holdout_set", "holdout"),
    ):
        cited = revision["dataset"][key]["concentration"]["effective_components"]
        assert cited is not None
        assert cited == manifest["coldstart"]["splits"][name]["concentration"][
            "effective_components"
        ]
        # and it must be far below the row count, which is the whole point
        assert cited < revision["dataset"][key]["items"]


def test_every_split_reports_the_capped_estimation_population(revision, manifest):
    """The population estimated from must be the population the manifest describes."""

    published = manifest["coldstart"]["estimation_population"]
    assert published["estimation_cap"] == fit.ESTIMATION_CAP
    assert published["namespace"] == fit.CAP_SUBSAMPLE_NAMESPACE
    assert published["problems"] == []
    assert published["every_split_clears_the_cold_minimum_under_the_cap"] is True
    assert published["every_split_keeps_a_non_zero_cold_share_under_the_cap"] is True
    assert published["pairwise_component_disjoint"] is True
    for name in ("train", "eval", "holdout"):
        estimation = manifest["coldstart"]["splits"][name]["estimation"]
        assert estimation["estimation_cap"] == fit.ESTIMATION_CAP
        assert estimation["capped_cold_candidates"] >= 400, name
        assert estimation["capped_cold_candidate_share"] > 0.0, name
        assert (
            estimation["effective_components_capped"]
            >= fit.EFFECTIVE_COMPONENT_FLOOR_WHOLE_SPLIT
        ), name
        assert published["per_split"][name]["capped_cold_candidates"] == estimation[
            "capped_cold_candidates"
        ]


def test_the_republished_capped_population_is_the_one_the_revision_reports(
    revision, manifest
):
    for name in ("train", "eval", "holdout"):
        cited = revision["cold_candidate_share_by_split"][name]
        estimation = manifest["coldstart"]["splits"][name]["estimation"]
        assert cited["capped_cold_candidates"] == estimation["capped_cold_candidates"]
        assert cited["capped_cold_candidate_share"] == estimation[
            "capped_cold_candidate_share"
        ]
        assert cited["cold_candidate_share"] > 0.0, name


def test_the_cap_never_moves_a_component_between_splits(manifest):
    """Disjointness under the cap is the same disjointness, and is re-checked."""

    published = manifest["coldstart"]["estimation_population"]
    assert set(published["pairwise_component_intersections"]) == {
        "train_eval",
        "train_holdout",
        "eval_holdout",
    }
    assert all(value == 0 for value in published["pairwise_component_intersections"].values())


def test_the_verdict_arm_is_the_capped_one_and_the_others_carry_none(revision):
    block = revision["component_aware_estimation"]
    assert block["estimation_cap"] == fit.ESTIMATION_CAP
    assert block["no_verdict_rests_on_a_reported_arm"] is True
    assert block["cap_selection"]["agrees_with_the_registered_value"] is True
    assert block["cap_selection"]["reads_no_direction_sign_or_label"] is True
    assert block["effective_component_floors"] == {
        "whole_split": fit.EFFECTIVE_COMPONENT_FLOOR_WHOLE_SPLIT,
        "cold_subpopulation": fit.EFFECTIVE_COMPONENT_FLOOR_COLD,
    }


def test_the_assignment_is_re_executable_from_its_declared_inputs_alone(revision, manifest):
    """``blindness_invariant_binding``, settled by replay rather than by prose."""

    binding = revision["dataset"]["assignment_blindness_invariant"]
    assert binding["membership_digests_reproduced"] is True
    published = {
        name: manifest["coldstart"]["splits"][name]["membership_digest"]
        for name in ("train", "eval", "holdout")
    }
    assert binding["reproduced_membership_digests"] == published
    assert binding["components_replayed"] == len(
        manifest["coldstart"]["split"]["assignment_inputs"]["components"]
    )


def test_no_cold_statistic_is_among_the_published_assignment_inputs(revision):
    """The HALT trigger is narrow and specific; this is the check that binds it."""

    binding = revision["dataset"]["assignment_blindness_invariant"]
    assert set(binding["per_component_fields_published"]) <= {"c", "n", "r", "f"}
    for field in binding["per_component_fields_published"]:
        for token in ("cold", "consum", "matur", "label", "score"):
            assert token not in field


def test_the_v1_uniform_draw_is_recomputed_beside_the_shipped_assignment(revision):
    """``residual_risk_stated`` is binding ON THE FIT, not on the manifest."""

    risk = revision["dataset"]["residual_risk_of_the_size_aware_assignment"]
    assert risk["recomputed_by_the_fit"] is True
    assert risk["reads_no_row_and_breaks_no_seal"] is True
    for name in ("train", "eval", "holdout"):
        assert risk["seed_uniform_draw"][name]["cold_candidates"] >= 0
        assert risk["shipped_assignment"][name]["cold_candidates"] > 0
    # the disclosure is only worth something if the two rules actually differ
    assert any(risk["cold_difference_shipped_minus_uniform"].values())


# ---------------------------------------------------------------------------
# The published artifact: the negative result
# ---------------------------------------------------------------------------


def test_the_sealed_partition_was_never_evaluated(revision):
    assert revision["holdout_evaluations"] == 0
    assert revision["sealed_partition_reads_observed"] == 0
    assert revision["refit_on_holdout"] is False
    assert revision["no_holdout_refit"] is True
    assert revision["generalization_threshold"]["evaluated_on_holdout"] is False
    assert revision["generalization_threshold"]["holdout_result"] is None


def test_generalization_threshold_is_taken_from_the_prereg_and_did_not_move(revision, prereg):
    cited = revision["generalization_threshold"]
    assert cited["value"] == prereg["verdict_rule"]["generalization_threshold"] == 1.5
    assert cited["comparison"] == prereg["verdict_rule"]["generalization_threshold_comparison"]
    assert cited["quantity"] == prereg["verdict_rule"]["generalization_threshold_quantity"]
    assert cited["stated_before_fitting"] is True


def test_a_halted_revision_is_not_offered_to_the_wire(revision):
    """The wire sibling consumes constants; a halted fit must not look ready."""

    assert revision["verdict"] == "negative_result"
    assert revision["wireable"] is False
    assert revision["status"] == "negative_result_not_wireable"
    assert revision["failed_steps"], "a negative result must name what failed"
    assert revision["fitted_candidate"]["the_holdout_was_opened"] is False


def test_the_sealed_parameters_are_published_but_carry_no_claim(revision):
    """Step 4 really did seal them, so hiding them would weaken the record.

    A halt AFTER the seal is not the same object as a halt before it: the
    guards that failed were run against these exact parameters, and a reader
    who cannot see them cannot re-run the failure.  What must not happen is
    that publishing them reads as readiness -- so ``wireable`` and ``verdict``
    are the fields that decide, and the block says so on its face.
    """

    candidate = revision["fitted_candidate"]
    assert candidate is not None
    assert candidate["parameter_sha256"] == revision["reproduction"][
        "published_parameter_sha256"
    ]
    assert candidate["these_parameters_were_sealed_to_disk_before_any_gate_after_step_4"] is True
    assert "no generalization claim" in candidate["not_wireable_unless_the_verdict_is_pass"]
    assert revision["wireable"] is False


def test_the_parameters_reproduce_from_the_train_split_alone(revision):
    """``refit_checkability``: 'not refit on' is reproduced, not asserted."""

    reproduction = revision["reproduction"]
    assert reproduction["train_only_reproduces_published"] is True
    assert reproduction["withheld_holdout_reproduces_published"] is True
    assert reproduction["byte_identical"] is True


def test_the_halt_is_at_step_6_on_the_warm_non_degradation_guard(revision):
    halted = revision["halted_at"]
    assert halted["step"] == 6
    step = next(entry for entry in revision["decision_procedure"] if entry["step"] == 6)
    assert step["verdict"] == "fail"
    guard_c = step["guard_c"]
    assert guard_c["holds"] is False
    assert guard_c["matched_volume_comparison"] == "computed", "K was large enough to bind"
    assert guard_c["K_admitted_by_the_frozen_scorer"] >= fit.GUARD_C_MINIMUM_K
    assert (
        guard_c["revised_warm_precision_at_matched_volume"]
        < guard_c["frozen_warm_precision"]
    ), "this is the shortfall that carries the negative result"
    assert guard_c["binding_condition"].startswith(
        "revised_warm_precision_at_matched_volume >="
    )


def test_the_guard_that_failed_is_the_one_that_protects_the_root_goal(revision):
    """Guard (c) is what stops the map from turning into noise; it is not optional."""

    finding = revision["findings"]["negative_result"]
    assert finding["halted_at_step"] == 6
    assert finding["the_shortfall"] > 0.0
    text = " ".join(finding["reading"])
    assert "must not start returning noise" in text
    assert "ORDERINGS of the same rows" in text


def test_the_failure_is_recorded_in_both_arms_not_only_the_binding_one(revision):
    """A capped failure that the uncapped arm contradicts is a fact a reader is owed."""

    step = next(entry for entry in revision["decision_procedure"] if entry["step"] == 6)
    capped = step["guard_c"]
    uncapped = step["guard_c_uncapped_reported_only"]
    assert uncapped["holds"] is False, "the reported arm agrees; the halt is not a cap artifact"
    assert uncapped["frozen_warm_precision"] != capped["frozen_warm_precision"]


def test_the_gates_that_did_hold_are_recorded_as_holding(revision):
    """A negative result is only worth something if it says what it is NOT about."""

    step5 = next(entry for entry in revision["decision_procedure"] if entry["step"] == 5)
    step6 = next(entry for entry in revision["decision_procedure"] if entry["step"] == 6)
    assert step5["verdict"] == "pass"
    assert step5["binding_verdict"]["holds"] is True
    assert set(step5["binding_verdict"]["member_verdicts"].values()) == {"pass"}
    assert step5["secondary_anchor"]["whole_tail_verdict"] == "pass"
    assert step6["guard_d"]["holds"] is True, "the bands held on train and eval"


def test_the_sealed_train_measurement_reproduced_before_anything_was_selected(revision):
    """``train_check_is_non_evidentiary``: a reproduction check, and it passed."""

    step = next(entry for entry in revision["decision_procedure"] if entry["step"] == 3)
    check = step["reproduction_of_the_sealed_measurement"]
    assert check["reproduces_the_sealed_measurement"] is True
    assert check["mismatches"] == []
    assert check["comparisons"] > 200
    assert check["and_none_of_them_is_a_gate"] is True
    # the six known non-matching readings are published, not failed
    published = check["registered_signs_not_matching_their_own_train_reading"]
    assert len(published) == 6
    assert "result_score_unweighted_cold" in published


def test_the_train_readings_are_published_in_every_registered_arm(revision, prereg):
    step = next(entry for entry in revision["decision_procedure"] if entry["step"] == 3)
    measured = step["published_train_readings_nine_arms"]
    registered = prereg["direction_constraint"]["registered_directions"]["bm25_score"][
        "measurement"
    ]
    assert set(measured) == set(registered)
    for arm, sealed in registered.items():
        assert measured[arm]["bm25_score"]["whole_tail"] == pytest.approx(
            sealed["whole_tail"], abs=1e-6
        )


def test_the_era_component_identity_question_is_settled_as_a_named_field(revision):
    """``era_exchangeability`` binds the fit to answer this, not to leave it open."""

    step = next(entry for entry in revision["decision_procedure"] if entry["step"] == 5)
    identity = step["era_component_identity_on_eval"]
    assert isinstance(identity["the_same_component"], bool)
    assert identity["component_identities_are_not_published"].startswith(
        "dataset-manifest.json#privacy.item_level_rows stays false"
    )


def test_the_simpson_guard_is_reported_vacuous_and_not_passed(revision):
    """The plan registered this expectation in advance; a vacuous guard is not a pass."""

    step = next(entry for entry in revision["decision_procedure"] if entry["step"] == 5)
    guard = step["simpson_guard"]
    assert guard["vacuous"] is True
    assert guard["reported_as_vacuous_not_as_passed"] is True
    assert len(guard["measurable_eras"]) < 2


def test_the_frontier_is_published_even_though_the_run_halted(revision):
    """``threshold_rule.published`` does not depend on the verdict."""

    step4 = next(entry for entry in revision["decision_procedure"] if entry["step"] == 4)
    step6 = next(entry for entry in revision["decision_procedure"] if entry["step"] == 6)
    for arms in (
        step4["feasible_threshold_count_all_arms"],
        step6["feasible_threshold_count_on_eval_all_arms"],
    ):
        assert set(arms) == set(fit.ARM_NAMES)
        assert all(isinstance(value, int) for value in arms.values())
    assert step4["feasible_threshold_count"] == step4["feasible_threshold_count_all_arms"][
        "capped"
    ]


def test_the_record_names_what_this_node_may_not_do_about_it(revision):
    forbidden = revision["findings"]["what_this_node_may_not_do_about_it"][
        "forbidden_by_step_9"
    ]
    joined = " ".join(forbidden)
    for lever in ("family", "branch", "re-draw", "band", "holdout", "estimation_cap", "arm"):
        assert lever in joined


def test_no_holdout_number_exists_anywhere_in_the_record(revision):
    """The holdout was never opened; nothing may quietly carry a number from it."""

    assert revision["holdout_cold_admitted"] is None
    assert revision["holdout_cold_admitted_exceeds_zero"] is False
    assert revision["holdout_cold_lift"] is None
    assert revision["holdout_first_read_after_parameters_sealed"] is False
    assert revision["dataset"]["holdout_set"]["cited_from_the_manifest_only"]
    assert revision["findings"]["the_holdout_is_still_virgin"]


def test_the_family_member_set_is_the_closed_registered_one(revision, prereg):
    assert prereg["feature_family"]["closed"]
    registered = [member["name"] for member in prereg["feature_family"]["members"]]
    step = next(entry for entry in revision["decision_procedure"] if entry["step"] == 5)
    measured = list(step["three_arm_direction_readings"]["capped"]["members"])
    assert sorted(measured) == sorted(registered)
    assert "graph_score" not in measured, "permanently excluded by the plan"
