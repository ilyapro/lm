#!/usr/bin/env python3
"""Execute the sealed cold-start decision procedure of ``coldstart-prereg-v3.json``.

The v1 fit recorded a NEGATIVE RESULT on the SHAPE: no single threshold could
satisfy the registered admitted-fraction bands, because the frozen policy sums
four history terms on an unbounded z-scale and a cold row therefore carries a
handicap that grows with the feature count.  v2 changed the shape to
``score = N + F`` -- two rank-calibrated arms, ``N = (L + H) / 2`` bounded to
[-0.5, +0.5] however many node-intrinsic features exist, ``F`` the rank-normal
image of the candidate-time family clipped to [-3, +3] -- and then HALTED at
step 3, because two registered signs inverted on the enlarged train split.

v3 changes NOTHING about the scorer.  It changes the ESTIMATOR that judges it.
Train's largest single component holds 7384 of 12344 items, so v2's readings
were taken over 2.78 effective units: a mean difference, an admitted fraction,
a lift and above all a binomial p-value computed there are statements about one
cache key.  Every verdict-carrying statistic here is computed on the
component-capped estimation set instead, at a cap DERIVED from the manifest's
published grid by a floor-first rule that reads no direction and no label.  The
uncapped arm -- v2's arm, the arm the HALT was measured on -- and the
component-equalized arm are published beside every one of them and decide
nothing.

It also moves where the direction test binds.  v3's signs were measured on
coldstart_train, so checking them there is a tautology; the train readings are
a reproduction check on the pipeline and the evidentiary weight sits on
coldstart_eval and secondary_anchor.

Nothing here is fitted on labels.  Every calibration is a rank transform of the
train split, fitted UNCAPPED, and both thresholds are selected by label-free
rules that read only admitted fractions.

The sealed partition is opened exactly once, after the fitted parameter digest
is on disk, and every holdout quantity -- the capped draw, the family ablation,
the per-era decomposition, all three arms -- is computed in that one pass.
Read-only throughout: the population is rebuilt through the sibling extractor,
which opens the store ``mode=ro`` under ``PRAGMA query_only``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = REPO_ROOT / "artifacts/recall-map/relevance/policy.json"
DEFAULT_PREREG = REPO_ROOT / "artifacts/recall-map/relevance/coldstart-prereg-v3.json"
DEFAULT_MANIFEST = REPO_ROOT / "artifacts/recall-map/relevance/dataset-manifest.json"
DEFAULT_ANALYSIS = REPO_ROOT / "artifacts/recall-map/relevance/feature-analysis.json"
EXTRACTOR = REPO_ROOT / "scripts/recall_map_coldstart_dataset.py"

# The plan this fit is bound to.  Verified before any split is read; a drifted
# prereg means the numbers below were not the ones registered in advance.
EXPECTED_PLAN_SHA256 = "be85f81c0b581741f498632952f012de56a8aedaa475fcdb73006da542a6213b"
SUPERSEDED_PLAN_SHA256 = "0af8124b743ee2031d93bd71b35b99df65fe9d7b05068fe7baa836a1f5f24de6"

REVISION_ID = "rank-calibrated-two-arm-coldstart-r2"
FIT_SPLIT = "train"
VERIFY_SPLIT = "eval"
SEALED_SPLIT = "holdout"

# ``component_aware_estimation``.  The cap is an ESTIMATOR, not a scorer input:
# it never reaches ``src/living_memory/recall_map.py`` and it never touches a
# calibration table.  Its value is not chosen here -- it is read off the
# manifest's published cap grid by the registered floor-first rule, and
# :func:`select_estimation_cap` re-derives it rather than trusting this literal.
CAP_GRID = (25, 50, 100, 200, 400)
ESTIMATION_CAP = 50
CAP_SUBSAMPLE_NAMESPACE = "coldstart-component-cap-v1"
EFFECTIVE_COMPONENT_FLOOR_WHOLE_SPLIT = 100
EFFECTIVE_COMPONENT_FLOOR_COLD = 16

# The three arms every verdict-carrying statistic is published in.  Only the
# first carries a verdict; the other two are mandatory reporting and may
# neither rescue a failing capped reading nor overturn a passing one.
VERDICT_ARM = "capped"
REPORTED_ARMS = ("uncapped", "component_equalized")
ARM_NAMES = (VERDICT_ARM, *REPORTED_ARMS)

# ``era_exchangeability``.  ``preexisting`` is precisely the v1 window.
ERA_BOUNDARY = "2026-08-01T00:00:00Z"
ERA_PREEXISTING = "preexisting_2026_08"
ERA_ENLARGEMENT = "enlargement_pre_2026_08"
ERA_NAMES = (ERA_ENLARGEMENT, ERA_PREEXISTING)
ERA_MEASURABLE_WHOLE_TAIL_ROWS = 200

# ``direction_constraint.train_check_is_non_evidentiary.tolerance``: the train
# readings are a REPRODUCTION check on the pipeline, not a test of the family.
DIRECTION_REPRODUCTION_TOLERANCE = 1e-6

# ``feature_family.members``: source field, member name, registered sign.  The
# set is CLOSED by the prereg -- graph_score is permanently excluded and may not
# be re-tested, which removes a degree of freedom v1 still had.
CORE_FAMILY: tuple[tuple[str, str, float], ...] = (
    ("score", "result_score", 1.0),
    ("trigger_score", "trigger_score", 1.0),
    ("bm25_score", "bm25_score", -1.0),
    ("vector_score", "vector_score", -1.0),
)

# ``scorer_shape.calibration.out_of_sample_and_live_application``.
ATOM_MINIMUM_SHARE = 0.005
GRID_KNOTS = 512
MAX_BREAKPOINTS = 1024
RANK_NORMAL_CLIP = 3.0
ROUND_PLACES = 8
U_ROUND_PLACES = 9

# ``verdict_rule``.  None of these moved between v1 and v2.
OVERALL_BAND = (0.10, 0.30)
COLD_VERDICT_BAND = (0.05, 0.25)
COLD_THRESHOLD_RULE_BAND = (0.10, 0.25)
GENERALIZATION_THRESHOLD = 1.5
MINIMUM_COLD_ADMITTED_ON_HOLDOUT = 40
BINOMIAL_ALPHA = 0.05
GUARD_C_MINIMUM_K = 20

# ``anti_fiat_guards.guard_c_warm_non_degradation``: the frozen scorer, exactly
# as ``src/living_memory/recall_map.py`` stands today.
FROZEN_THRESHOLD = 3.8708378402511

# What the v1 fit measured for the same four members on the SMALL 2026-08-01..22
# window, before the population was enlarged.  Verbatim from commit 7bd5355's
# ``policy.json#coldstart_revision.decision_procedure`` step
# ``core_registered_signs_on_train`` (train 5664 items / 1233 cold), on v1's
# winsorized-and-standardized scale.  Carried as a constant because that policy
# object is the very thing this run overwrites, and the contrast is the finding.
V1_TRAIN_DIRECTIONS: Mapping[str, Any] = {
    "cohort": "v1 coldstart_train, 5664 items / 1233 cold, window 2026-08-01..08-22",
    "source": "commit 7bd5355, policy.json#coldstart_revision, step core_registered_signs_on_train",
    "statistic": "consumed-minus-nonconsumed mean difference, winsorized-and-standardized scale",
    "members": {
        "result_score": {"whole_tail": 0.321331, "cold": 0.175389, "verdict": "pass"},
        "trigger_score": {"whole_tail": 0.323586, "cold": 0.153444, "verdict": "pass"},
        "bm25_score": {"whole_tail": -0.20075, "cold": -0.577835, "verdict": "pass"},
        "vector_score": {"whole_tail": -0.266866, "cold": -0.396621, "verdict": "pass"},
    },
}


class ColdStartFitError(SystemExit):
    """Refusal: a sealed precondition failed, or a guard was tripped."""


# ---------------------------------------------------------------------------
# Canonicalisation, shared with the sealed artifacts
# ---------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    """Compact rendering: what ``policy.json#bindings`` hashes are taken over."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_json_indented(value: Any) -> str:
    """``judge.canonical_json``: the rendering the prereg seal was taken over.

    The two renderings are NOT interchangeable and the difference is not
    cosmetic -- the prereg seal_note pins judge.py:286 explicitly, while every
    binding digest in policy.json is compact.  Hashing either one the other
    way silently fails to verify what it claims to verify.
    """

    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round(value: float | None, places: int = 6) -> float | None:
    return None if value is None else round(value, places)


def extractor() -> ModuleType:
    """Import the sibling extractor by path, so the population rule is shared."""

    spec = importlib.util.spec_from_file_location("recall_map_coldstart_dataset", EXTRACTOR)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging accident
        raise ColdStartFitError(f"cannot import extractor at {EXTRACTOR}")
    module = importlib.util.module_from_spec(spec)
    # ``@dataclass(slots=True)`` re-creates the class and looks its module up in
    # ``sys.modules``; registering before exec is required, not cosmetic.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_prereg_seal(prereg: Mapping[str, Any]) -> str:
    """Step 1: recompute ``plan_sha256`` before anything else is touched.

    A fit whose plan drifted is a fit whose numbers were not registered in
    advance, which is the one thing pre-registration exists to prevent.
    """

    over = prereg.get("plan_sha256_over")
    if not over:
        raise ColdStartFitError("prereg carries no plan_sha256_over")
    payload = {key: prereg[key] for key in over}
    digest = sha256_text(canonical_json_indented(payload))
    recorded = prereg.get("plan_sha256")
    if digest != recorded:
        raise ColdStartFitError(f"prereg seal mismatch: recomputed {digest}, recorded {recorded}")
    if digest != EXPECTED_PLAN_SHA256:
        raise ColdStartFitError(
            f"prereg is not the plan this fit is bound to: {digest} != {EXPECTED_PLAN_SHA256}"
        )
    return digest


# ---------------------------------------------------------------------------
# The seal
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SealBreak:
    """The token that opens the sealed partition. Constructing one is the act."""

    reason: str
    parameter_sha256: str


@dataclass(slots=True)
class SealedSplits:
    """Row access with the sealed partition behind a runtime guard.

    ``single_holdout_pass`` permits exactly one materialization.  Making that a
    runtime guard rather than a convention is the whole point: a fit that never
    constructs a :class:`SealBreak` cannot have leaked, whatever its prose says.
    """

    _rows: dict[str, list[dict[str, Any]]]
    reads: dict[str, int] = field(default_factory=dict)

    def open(self, split: str, *, unseal: SealBreak | None = None) -> list[dict[str, Any]]:
        if split not in self._rows:
            raise ColdStartFitError(f"unknown split {split!r}")
        if split == SEALED_SPLIT:
            if not isinstance(unseal, SealBreak):
                raise ColdStartFitError(
                    "refusing to open the sealed partition: "
                    "prereg split_protocol.seal.ordering allows it only after the "
                    "fitted parameter digest is on disk, and only once"
                )
            if self.reads.get(SEALED_SPLIT):
                raise ColdStartFitError(
                    "refusing a SECOND read of the sealed partition: "
                    "single_holdout_pass.forbidden bans a second read for any reason"
                )
        self.reads[split] = self.reads.get(split, 0) + 1
        return list(self._rows[split])

    def counts(self, split: str) -> dict[str, int]:
        """Split-audit statistics. Counting rows is what step 2 requires."""

        rows = self._rows[split]
        cold = [row for row in rows if row["cold"]]
        return {
            "items": len(rows),
            "consumed": sum(1 for row in rows if row["consumed"]),
            "cold_candidates": len(cold),
            "cold_consumed": sum(1 for row in cold if row["consumed"]),
        }

    @property
    def sealed_reads(self) -> int:
        return self.reads.get(SEALED_SPLIT, 0)


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def load_population(db_path: Path) -> dict[str, Any]:
    """Rebuild the candidate population and the three splits, read-only.

    This must mirror ``build_coldstart_block`` exactly.  The enlarged split is
    a v1-role carry-over, so it needs the reconstructed v1 partition and the
    observed-map forcing set; assembling the components without them would
    silently produce a DIFFERENT three-way partition from the committed one and
    ``confirm_splits_match_manifest`` would be comparing the fit against a split
    the manifest never published.
    """

    module = extractor()
    module.assert_not_holdout(db_path)
    extracted = module.build_population(db_path, source_name="local")
    try:
        opportunity = [row for row in extracted["candidates"] if row.opportunity]
        if not opportunity:
            raise ColdStartFitError("no opportunity-bearing candidate survived")
        consumer_index = extracted["consumer_index"]
        component_events = {
            event.id
            for event in extracted["window_events"]
            if bool(consumer_index.qualifying(event))
        }
        v1_component_events = {
            event.id
            for event in extracted["v1_window_events"]
            if bool(consumer_index.qualifying(event))
        }
        v1 = module.reconstruct_v1_partition(
            extracted["v1_candidates"],
            extracted["identities"],
            component_event_ids=v1_component_events,
        )
        assignment = module.assign_three_way_splits(
            opportunity,
            extracted["identities"],
            component_event_ids=component_events,
            v1_role_by_event=v1["role_by_event"],
            observed_map_event_ids=set(extracted.get("observed_map_event_ids") or ()),
        )
    finally:
        extracted["connection"].close()

    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in module.SPLIT_NAMES}
    for candidate in opportunity:
        rows[candidate.split].append(
            {
                "split": candidate.split,
                "cold": candidate.cold,
                "consumed": bool(candidate.consumed),
                "level": candidate.level,
                "matured": candidate.matured,
                "matured_consumed": candidate.matured_consumed,
                "trailing_nonconsumed": candidate.trailing_nonconsumed,
                "component_id": candidate.component_id,
                # The three identity fields the component cap orders on, plus
                # the instant the era cut reads.  None of the four is a label,
                # a feature value, a score or a coldness statistic, so carrying
                # them cannot let the estimator select on an outcome.
                "source": candidate.source,
                "event_id": candidate.event_id,
                "rank": candidate.rank,
                "decided_at": candidate.decided_at,
                "era": (
                    ERA_PREEXISTING
                    if candidate.decided_at >= ERA_BOUNDARY
                    else ERA_ENLARGEMENT
                ),
                "scores": dict(candidate.scores),
            }
        )
    components = {name: sorted({row["component_id"] for row in rows[name]}) for name in rows}
    return {
        "rows": rows,
        "components": components,
        "all_components": assignment["components"],
        "component_sizes": {
            component: sum(
                1
                for split_rows in rows.values()
                for row in split_rows
                if row["component_id"] == component
            )
            for component in assignment["components"]
        },
    }


def confirm_splits_match_manifest(
    population: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """The splits this fit uses must be the committed, sealed ones.

    ``refit_checkability.supporting_checks`` requires the partition the policy
    cites to be the partition the manifest published; a re-drawn split would
    silently give the fit a different population to be lucky on.  The membership
    digest is checked as well as the counts, because counts can coincide.
    """

    module = extractor()
    block = manifest.get("coldstart")
    if not isinstance(block, Mapping):
        raise ColdStartFitError("dataset manifest carries no coldstart block")
    report: dict[str, Any] = {}
    mismatches: list[str] = []
    for name, published in block["splits"].items():
        rows = population["rows"][name]
        cold = [row for row in rows if row["cold"]]
        rebuilt_members = population["components"][name]
        same = {
            "items": len(rows) == published["items"],
            "cold_candidates": len(cold) == published["cold_candidates"],
            "consumed": sum(1 for r in rows if r["consumed"]) == published["consumed"],
            "components": rebuilt_members == sorted(published["components"]),
            "membership_digest": (
                module._membership_digest(rebuilt_members) == published["membership_digest"]
            ),
        }
        report[name] = {**same, "items_rebuilt": len(rows), "cold_rebuilt": len(cold)}
        mismatches.extend(f"{name}.{key}" for key, ok in same.items() if not ok)
    if mismatches:
        raise ColdStartFitError(
            "rebuilt splits do not match the committed manifest: " + ", ".join(mismatches)
        )
    report["all_splits_reproduce_committed_manifest"] = True
    return report


# ---------------------------------------------------------------------------
# Component-aware estimation
#
# ``component_aware_estimation.principle``: no statistic that carries a verdict
# is computed on a raw split.  Train's largest single component holds 7384 of
# 12344 items, so a mean difference, an admitted fraction, a lift or -- above
# all -- a binomial p-value taken over the raw split is a statement about one
# cache key.  Every verdict-carrying quantity below is computed on the capped
# estimation row set, drawn ONCE per cohort and then held fixed.
#
# The cap is an estimator and touches nothing else: not a calibration table,
# not a score, not the split assignment, and not the live selector.
# ---------------------------------------------------------------------------


def cap_order_key(row: Mapping[str, Any]) -> str:
    """Which of a component's rows the cap keeps, from label-free fields only.

    Byte-identical to ``recall_map_coldstart_dataset._cap_order_key``: the same
    namespace, the same five fields, the same canonical rendering.  It has to
    be, or the arm this fit computes would not be the arm the manifest
    published and ``direction_constraint``'s reproduction check would fail for
    a reason that has nothing to do with the population.
    """

    return sha256_text(
        canonical_json(
            [
                CAP_SUBSAMPLE_NAMESPACE,
                row["component_id"],
                row["source"],
                row["event_id"],
                row["rank"],
            ]
        )
    )


def cap_survivors(rows: Sequence[Mapping[str, Any]], cap: int) -> list[dict[str, Any]]:
    """At most ``cap`` rows per component, INSIDE the split it already sits in.

    No component is divided across splits by this and none can be: the split a
    row belongs to is never consulted, only how many of its component's rows
    have already been kept.  So ``pairwise_component_intersections`` is
    untouched, and every component survives -- a cap keeps ``min(cap, n_c)``
    rows, which is at least one.
    """

    if cap <= 0:
        raise ColdStartFitError(f"estimation cap must be positive, got {cap}")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["component_id"], []).append(row)
    kept: list[dict[str, Any]] = []
    for component_id in sorted(grouped):
        members = sorted(grouped[component_id], key=cap_order_key)
        kept.extend(dict(row) for row in members[:cap])
    return kept


def equalized_weights(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """``w_i = 1 / (C * n_c)``: every component carries equal total weight.

    The no-rows-dropped alternative the plan publishes but does not use as its
    estimator, for the four reasons at
    ``component_aware_estimation.why_a_cap_and_not_component_equalized_weights``
    -- chief among them that an exact binomial has no meaning over fractional
    pseudo-counts.
    """

    sizes: dict[str, int] = {}
    for row in rows:
        sizes[row["component_id"]] = sizes.get(row["component_id"], 0) + 1
    count = len(sizes)
    if not count:
        return []
    return [1.0 / (count * sizes[row["component_id"]]) for row in rows]


def effective_components(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """``1 / sum_c s_c^2`` over component shares: the inverse Simpson index.

    The same quantity the manifest publishes under that name.  ``E >= 16`` is
    exactly the manifest's registered ``one_cluster_bar`` of 0.25 restated on
    an effective-count scale, since ``E >= 1/s^2`` forces ``s <= 1/sqrt(E)``;
    ``E >= 100`` is a strict tightening of the same bar to ``s <= 0.10``.
    """

    sizes: dict[str, int] = {}
    for row in rows:
        sizes[row["component_id"]] = sizes.get(row["component_id"], 0) + 1
    total = sum(sizes.values())
    if not total:
        return None
    simpson = sum((count / total) ** 2 for count in sizes.values())
    return round(1.0 / simpson, 2) if simpson else None


def kish_effective_sample_size(weights: Sequence[float]) -> float | None:
    """``(sum w)^2 / sum w^2``.  Exactly the row count when weights are uniform."""

    total = sum(weights)
    squared = sum(weight * weight for weight in weights)
    if not squared:
        return None
    return round(total * total / squared, 2)


@dataclass(slots=True)
class Arm:
    """One estimator over one cohort: a row set and a weighting.

    ``capped`` carries every verdict.  ``uncapped`` is v2's arm -- the arm the
    recorded HALT was measured on -- and ``component_equalized`` is the
    no-rows-dropped alternative.  Both are MANDATORY reporting and neither may
    rescue a failing capped reading or overturn a passing one.
    """

    name: str
    rows: list[dict[str, Any]]
    weights: list[float] | None

    @property
    def carries_the_verdict(self) -> bool:
        return self.name == VERDICT_ARM

    def summary(self) -> dict[str, Any]:
        cold = [row for row in self.rows if row["cold"]]
        weights = self.weights or [1.0] * len(self.rows)
        cold_weights = [
            weight
            for weight, row in zip(weights, self.rows, strict=True)
            if row["cold"]
        ]
        return {
            "arm": self.name,
            "carries_the_verdict": self.carries_the_verdict,
            "rows": len(self.rows),
            "consumed": sum(1 for row in self.rows if row["consumed"]),
            "cold_rows": len(cold),
            "cold_consumed": sum(1 for row in cold if row["consumed"]),
            "components": len({row["component_id"] for row in self.rows}),
            "effective_components": effective_components(self.rows),
            "cold_effective_components": effective_components(cold),
            "effective_sample_size": kish_effective_sample_size(weights),
            "cold_effective_sample_size": kish_effective_sample_size(cold_weights),
            "largest_component_share": concentration(self.rows)["largest_component_share"],
        }


def build_arms(rows: Sequence[Mapping[str, Any]], cap: int) -> dict[str, Arm]:
    """The three registered arms over one cohort, drawn once and held fixed.

    ``the_estimation_rule.drawn_once_per_cohort``: the capped set is computed
    once from the cohort's full row list and every statistic on that cohort --
    overall, cold, warm, admitted, ablation, per era, per level -- is a subset
    of it.  Re-drawing per statistic, or capping the cold subpopulation
    separately, is FORBIDDEN; it would reintroduce as a per-statistic degree of
    freedom exactly what the cap exists to remove.
    """

    materialized = [dict(row) for row in rows]
    return {
        VERDICT_ARM: Arm(VERDICT_ARM, cap_survivors(materialized, cap), None),
        "uncapped": Arm("uncapped", materialized, None),
        "component_equalized": Arm(
            "component_equalized", materialized, equalized_weights(materialized)
        ),
    }


def select_estimation_cap(manifest_block: Mapping[str, Any]) -> dict[str, Any]:
    """``estimation_cap.selection_rule``, re-derived rather than trusted.

    The LARGEST cap in the published grid at which every split's capped
    effective components meet the whole-split floor.  It reads the manifest's
    own ``cap_feasibility.table`` and no row, and it consults no direction, no
    sign, no label and no outcome -- so no choice of cap can set a sign, and
    every registered direction is unchanged at any other grid point.
    """

    table = manifest_block["enlargement"]["cap_feasibility"]["table"]
    per_cap: dict[str, Any] = {}
    feasible: list[int] = []
    for cap in CAP_GRID:
        entry = table.get(str(cap))
        if not isinstance(entry, Mapping):
            raise ColdStartFitError(f"HALT at step 2: no published cap row for cap {cap}")
        readings = {
            name: entry[name]["effective_components"]
            for name in (FIT_SPLIT, VERIFY_SPLIT, SEALED_SPLIT)
        }
        meets = all(
            value is not None and value >= EFFECTIVE_COMPONENT_FLOOR_WHOLE_SPLIT
            for value in readings.values()
        )
        per_cap[str(cap)] = {**readings, "meets_the_whole_split_floor": meets}
        if meets:
            feasible.append(cap)
    if not feasible:
        raise ColdStartFitError(
            "HALT at step 2: no cap in the published grid puts every split at or above "
            f"effective_component_floor.whole_split = {EFFECTIVE_COMPONENT_FLOOR_WHOLE_SPLIT}"
        )
    chosen = max(feasible)
    return {
        "rule": (
            "the largest cap in the published grid at which every split's capped "
            f"effective_components is >= {EFFECTIVE_COMPONENT_FLOOR_WHOLE_SPLIT}"
        ),
        "grid": list(CAP_GRID),
        "floor": EFFECTIVE_COMPONENT_FLOOR_WHOLE_SPLIT,
        "effective_components_per_split_per_cap": per_cap,
        "feasible_caps": feasible,
        "estimation_cap": chosen,
        "agrees_with_the_registered_value": chosen == ESTIMATION_CAP,
        "read_no_row": True,
        "reads_no_direction_sign_or_label": True,
        "source": (
            "artifacts/recall-map/relevance/dataset-manifest.json"
            "#coldstart.enlargement.cap_feasibility.table"
        ),
    }


def era_partition(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    return {name: [row for row in rows if row["era"] == name] for name in ERA_NAMES}


def era_composition(arm: Arm) -> dict[str, Any]:
    """``era_exchangeability.the_fit_must_publish_the_capped_era_composition``.

    Per-era items, cold counts, consumed counts and rates on the row set this
    arm estimates from, so a pooled in-band number cannot hide one era at 0.02
    and another at 0.45.
    """

    out: dict[str, Any] = {}
    for name, era_rows in era_partition(arm.rows).items():
        cold = [row for row in era_rows if row["cold"]]
        consumed = sum(1 for row in era_rows if row["consumed"])
        out[name] = {
            "items": len(era_rows),
            "item_share": _round(len(era_rows) / len(arm.rows), 6) if arm.rows else None,
            "consumed": consumed,
            "rate": _round(consumed / len(era_rows), 6) if era_rows else None,
            "cold_candidates": len(cold),
            "cold_consumed": sum(1 for row in cold if row["consumed"]),
            "cold_base_rate": (
                _round(sum(1 for row in cold if row["consumed"]) / len(cold), 6)
                if cold
                else None
            ),
            "effective_components": effective_components(era_rows),
        }
    return out


# ---------------------------------------------------------------------------
# Rank calibration
# ---------------------------------------------------------------------------


def _u_of(below: int, equal: int, n: int) -> float:
    """``rank_uniform_centered``: (A + M/2)/n - 0.5, MIDRANK on ties."""

    return (below + equal / 2.0) / n - 0.5


@dataclass(slots=True)
class StepFunction:
    """A published monotone step function ``u(v)``, fitted on train alone.

    ``calibration.out_of_sample_and_live_application`` makes the PUBLISHED table
    the definition, not a summary of one: eval, holdout and the live scorer all
    evaluate this same object, so a threshold tie reproduces bit-for-bit.  Two
    parts, in priority order:

    * ``atoms`` -- every fitting value whose multiplicity is at least 0.5% of the
      fitting rows, as an exact ``(value, u)`` pair matched by equality before
      any interpolation.  The available-but-empty history composite is such an
      atom by construction, so the tie-block identity the feasibility argument
      rests on holds exactly rather than approximately.
    * ``knots`` -- the rest as a 512-knot quantile grid, linearly interpolated
      between consecutive knots and clamped outside the outermost pair.

    Values are rounded once, at fit time, and every later evaluation reads the
    rounded table.  Published and evaluated are therefore the same numbers.
    """

    name: str
    n: int
    atoms: tuple[tuple[float, float], ...]
    knots: tuple[tuple[float, float], ...]

    @classmethod
    def fit(cls, name: str, values: Sequence[float]) -> StepFunction:
        n = len(values)
        if not n:
            raise ColdStartFitError(f"cannot calibrate {name!r} on an empty fitting set")
        ordered = sorted(float(v) for v in values)
        distinct: list[tuple[float, int, int]] = []  # value, below, equal
        index = 0
        while index < n:
            value = ordered[index]
            end = index
            while end < n and ordered[end] == value:
                end += 1
            distinct.append((value, index, end - index))
            index = end

        atoms = tuple(
            (value, round(_u_of(below, equal, n), U_ROUND_PLACES))
            for value, below, equal, in distinct
            if equal / n >= ATOM_MINIMUM_SHARE
        )
        atom_values = {value for value, _u in atoms}

        # The grid is taken over the distinct values so a long tie block cannot
        # consume many knots and starve the rest of the range.
        remaining = [row for row in distinct if row[0] not in atom_values]
        knots: list[tuple[float, float]] = []
        if remaining:
            count = min(GRID_KNOTS, len(remaining))
            seen: set[float] = set()
            for step in range(count):
                position = 0 if count == 1 else round(step * (len(remaining) - 1) / (count - 1))
                value, below, equal = remaining[position]
                if value in seen:
                    continue
                seen.add(value)
                knots.append((value, round(_u_of(below, equal, n), U_ROUND_PLACES)))
        table = cls(name=name, n=n, atoms=atoms, knots=tuple(knots))
        if len(atoms) + len(knots) > MAX_BREAKPOINTS:
            raise ColdStartFitError(
                f"{name}: {len(atoms) + len(knots)} breakpoints exceeds the registered "
                f"maximum of {MAX_BREAKPOINTS}"
            )
        return table

    def u(self, value: float) -> float:
        """Exact atom match first, then interpolation, then clamping."""

        value = float(value)
        for atom_value, atom_u in self.atoms:
            if value == atom_value:
                return atom_u
        if not self.knots:
            return 0.0
        if value <= self.knots[0][0]:
            return self.knots[0][1]
        if value >= self.knots[-1][0]:
            return self.knots[-1][1]
        low = 0
        high = len(self.knots) - 1
        while high - low > 1:
            middle = (low + high) // 2
            if self.knots[middle][0] <= value:
                low = middle
            else:
                high = middle
        left_value, left_u = self.knots[low]
        right_value, right_u = self.knots[high]
        if right_value == left_value:
            return left_u
        weight = (value - left_value) / (right_value - left_value)
        return round(left_u + weight * (right_u - left_u), U_ROUND_PLACES)

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fitted_rows": self.n,
            "atoms": [[value, u] for value, u in self.atoms],
            "knots": [[value, u] for value, u in self.knots],
            "breakpoints": len(self.atoms) + len(self.knots),
        }


_NORMAL = NormalDist()


def rank_normal(table: StepFunction, value: float) -> float:
    """``F(v) = clip(Phi_inverse(clip(u(v) + 0.5, 1/2n, 1 - 1/2n)), -3, +3)``.

    The guaranteed percentiles on the fitting split -- p50 0.0, p84.13 1.0,
    p95 1.6449 -- are properties of this transform, not of the data.  Whatever
    shape the family composite has, including v1's p95 0.4607 / p99 9.46 spike,
    its rank-normal image has exactly these percentiles on train.
    """

    floor = 1.0 / (2.0 * table.n)
    probability = min(max(table.u(value) + 0.5, floor), 1.0 - floor)
    return round(
        min(max(_NORMAL.inv_cdf(probability), -RANK_NORMAL_CLIP), RANK_NORMAL_CLIP),
        ROUND_PLACES,
    )


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------


def level_readable(row: Mapping[str, Any]) -> bool:
    """The condition ``src/living_memory/recall_map.py:1229-1236`` already tests."""

    return row.get("level") in ("trace", "concept", "schema")


def level_raw(row: Mapping[str, Any]) -> float:
    """``level_is_schema``, the frozen feature 0."""

    return 1.0 if row.get("level") == "schema" else 0.0


def history_readable(row: Mapping[str, Any]) -> bool:
    """Exactly the conditions ``recall_map.py:1199-1217`` already tests."""

    try:
        matured = int(row["matured"])
        consumed = int(row["matured_consumed"])
        streak = int(row["trailing_nonconsumed"])
    except (KeyError, TypeError, ValueError):
        return False
    if matured < 0 or consumed < 0 or streak < 0:
        return False
    if consumed > matured:
        return False
    return streak <= matured - consumed


def history_raw(row: Mapping[str, Any], means: Sequence[float], scales: Sequence[float]) -> float:
    """The frozen four-term history z-sum, arithmetic unchanged.

    ``scorer_shape.arms.N.H.why_the_raw_composite_keeps_the_frozen_arithmetic``:
    ``rank_uniform_centered`` is strictly monotone, so within the readable
    subpopulation the revision's history ordering is bit-for-bit the frozen
    policy's ordering.  Not one pair of rows is re-ranked by history.  Only the
    arm's arithmetic SCALE relative to the other arm changes.
    """

    matured = int(row["matured"])
    consumed = int(row["matured_consumed"])
    streak = int(row["trailing_nonconsumed"])
    features = (
        round(math.log1p(matured), ROUND_PLACES),
        round(math.log1p(matured - consumed), ROUND_PLACES),
        round(math.log1p(streak), ROUND_PLACES),
        round(consumed / matured, ROUND_PLACES) if matured else 0.0,
    )
    return sum(
        (value - mean) / scale
        for value, mean, scale in zip(features, means[1:5], scales[1:5], strict=True)
    )


def member_raw(row: Mapping[str, Any], source: str) -> float | None:
    """``None`` means the member cannot be read as a finite float: fail closed."""

    raw = row.get("scores", {}).get(source)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def family_unscoreable(row: Mapping[str, Any]) -> bool:
    return any(member_raw(row, source) is None for source, _name, _sign in CORE_FAMILY)


@dataclass(slots=True)
class Scorer:
    """``score = N + F``, one additive sum of exactly two calibrated arms.

    There is no test of ``matured == 0`` anywhere in here: no per-cohort,
    per-coldness, per-scope or per-level branch, no separate threshold, no
    bonus, no tie-break, no eligibility test.  A cold row takes the SAME path as
    every other row -- its history composite is looked up in the same published
    step function and it receives the midrank of the block it shares.  What
    changed is only that the arm it sits at the bottom of is now bounded.
    """

    legacy_means: tuple[float, ...]
    legacy_scales: tuple[float, ...]
    level_table: StepFunction
    history_table: StepFunction
    member_tables: Mapping[str, StepFunction]
    family_table: StepFunction

    def level_u(self, row: Mapping[str, Any]) -> float:
        if not level_readable(row):
            return 0.0
        return self.level_table.u(level_raw(row))

    def history_u(self, row: Mapping[str, Any]) -> float:
        if not history_readable(row):
            return 0.0
        return self.history_table.u(history_raw(row, self.legacy_means, self.legacy_scales))

    def node_intrinsic(self, row: Mapping[str, Any]) -> float:
        """``N = (L + H) / 2``, bounded to [-0.5, +0.5] however many arms exist.

        The divisor stays 2 when an arm is unreadable: an unreadable arm
        contributes its calibrated NEUTRAL 0.0 and pulls N toward the middle,
        which is what "no information from this arm" has to mean.
        """

        return (self.level_u(row) + self.history_u(row)) / 2.0

    def family_composite(self, row: Mapping[str, Any]) -> float | None:
        total = 0.0
        for source, _name, sign in CORE_FAMILY:
            raw = member_raw(row, source)
            if raw is None:
                return None
            total += sign * self.member_tables[source].u(raw)
        return total

    def family_arm(self, row: Mapping[str, Any]) -> float | None:
        composite = self.family_composite(row)
        return None if composite is None else rank_normal(self.family_table, composite)

    def score(self, row: Mapping[str, Any]) -> float | None:
        """``None`` is INADMISSIBLE, not neutral. Family fail-closed, from v1."""

        arm = self.family_arm(row)
        return None if arm is None else round(self.node_intrinsic(row) + arm, ROUND_PLACES)

    def ablation_score(self, row: Mapping[str, Any]) -> float | None:
        """``S_ablation = N + 0.0``: the same shape with the family neutralised.

        Same calibration tables, same rows, same fail-closed rule, so the two
        shapes share a denominator and differ ONLY in whether F contributes its
        real rank-normal value.  This is the precise control for the accusation
        the plan invites -- it IS the recalibration fix with no family in it.
        """

        if family_unscoreable(row):
            return None
        return round(self.node_intrinsic(row), ROUND_PLACES)


def fit_tables(
    train_rows: Sequence[Mapping[str, Any]],
    legacy_means: Sequence[float],
    legacy_scales: Sequence[float],
) -> Scorer:
    """Every calibration table, derived from the train split and nothing else."""

    level_table = StepFunction.fit(
        "level_is_schema", [level_raw(row) for row in train_rows if level_readable(row)]
    )
    history_table = StepFunction.fit(
        "history_z_sum",
        [
            history_raw(row, legacy_means, legacy_scales)
            for row in train_rows
            if history_readable(row)
        ],
    )
    member_tables = {
        source: StepFunction.fit(
            name,
            [
                value
                for value in (member_raw(row, source) for row in train_rows)
                if value is not None
            ],
        )
        for source, name, _sign in CORE_FAMILY
    }

    # The family arm's own table is fitted on the composite, which needs the
    # member tables to exist first.  Rows the family cannot score contribute
    # nothing to it -- they are inadmissible, not neutral.
    partial = Scorer(
        legacy_means=tuple(legacy_means),
        legacy_scales=tuple(legacy_scales),
        level_table=level_table,
        history_table=history_table,
        member_tables=member_tables,
        family_table=StepFunction(name="bootstrap", n=1, atoms=(), knots=()),
    )
    composites = [
        value
        for value in (partial.family_composite(row) for row in train_rows)
        if value is not None
    ]
    return Scorer(
        legacy_means=tuple(legacy_means),
        legacy_scales=tuple(legacy_scales),
        level_table=level_table,
        history_table=history_table,
        member_tables=member_tables,
        family_table=StepFunction.fit("family_composite", composites),
    )


# ---------------------------------------------------------------------------
# Direction checks
# ---------------------------------------------------------------------------


def weighted_association(
    rows: Sequence[Mapping[str, Any]],
    weights: Sequence[float] | None,
    source: str,
    table: StepFunction,
) -> float | None:
    """The registered statistic, under one arm's weighting.

    ``direction_constraint.statistic``: consumed-minus-nonconsumed mean
    difference on ``rank_uniform_centered`` values.  It is an affine increasing
    function of the Mann-Whitney U statistic, so the check is exactly "does
    this member rank consumed rows above non-consumed rows", and it is
    invariant to every monotone re-expression of the member.

    ``u_axis``: the table is fitted on ALL of coldstart_train and held fixed
    across every arm.  Only which rows the estimator counts varies; the value
    axis does not move.
    """

    if weights is None:
        weights = [1.0] * len(rows)
    consumed_total = consumed_weight = 0.0
    other_total = other_weight = 0.0
    for row, weight in zip(rows, weights, strict=True):
        raw = member_raw(row, source)
        if raw is None:
            continue
        value = table.u(raw)
        if row["consumed"]:
            consumed_total += weight * value
            consumed_weight += weight
        else:
            other_total += weight * value
            other_weight += weight
    if not consumed_weight or not other_weight:
        return None
    return consumed_total / consumed_weight - other_total / other_weight


def arm_direction_reading(
    arm: Arm, scorer: Scorer, prereg: Mapping[str, Any]
) -> dict[str, Any]:
    """One arm's four member readings, whole tail and cold, evaluated apart.

    ``cold_subpopulation_clause``: a cohort below the registered minima is
    INDETERMINATE for that member, which BLOCKS and never passes.  The minima
    are counted on the row set this arm estimates from, because counting them
    on the uncapped split would let rows the estimator does not use satisfy a
    minimum about the estimator.
    """

    clause = prereg["direction_constraint"]["cold_subpopulation_clause"]
    min_cold = int(clause["minimum_cold_rows_for_a_direction_check"])
    min_cold_consumed = int(clause["minimum_consumed_cold_rows_for_a_direction_check"])

    weights = arm.weights or [1.0] * len(arm.rows)
    cold_pairs = [
        (row, weight)
        for row, weight in zip(arm.rows, weights, strict=True)
        if row["cold"]
    ]
    cold_rows = [row for row, _weight in cold_pairs]
    cold_weights = [weight for _row, weight in cold_pairs]
    cold_consumed = sum(1 for row in cold_rows if row["consumed"])
    measurable = len(cold_rows) >= min_cold and cold_consumed >= min_cold_consumed

    members: dict[str, Any] = {}
    for source, name, sign in CORE_FAMILY:
        table = scorer.member_tables[source]
        whole = weighted_association(arm.rows, arm.weights, source, table)
        cold = weighted_association(cold_rows, cold_weights, source, table)
        if whole is None or cold is None or not measurable:
            verdict = "indeterminate"
        elif whole * sign > 0 and cold * sign > 0:
            verdict = "pass"
        else:
            verdict = "fail"
        members[name] = {
            "registered_direction": "positive" if sign > 0 else "negative",
            "whole_tail": _round(whole),
            "cold": _round(cold),
            "whole_tail_matches_registered": None if whole is None else whole * sign > 0,
            "cold_matches_registered": None if cold is None else cold * sign > 0,
            "verdict": verdict,
        }
    return {
        **arm.summary(),
        "cold_check_measurable": measurable,
        "minimum_cold_rows_for_a_direction_check": min_cold,
        "minimum_consumed_cold_rows_for_a_direction_check": min_cold_consumed,
        "members": members,
    }


def three_arm_directions(
    arms: Mapping[str, Arm], scorer: Scorer, prereg: Mapping[str, Any]
) -> dict[str, Any]:
    """``mandatory_three_arm_reporting`` applied to the direction readings."""

    return {name: arm_direction_reading(arms[name], scorer, prereg) for name in ARM_NAMES}


def binding_direction_verdict(three_arms: Mapping[str, Any]) -> dict[str, Any]:
    """Only the CAPPED arm decides.  The other two are published beside it.

    ``mandatory_three_arm_reporting.no_verdict_may_rest_on_the_reported_arms``:
    the reported arms may not rescue a failing capped verdict and may not
    overturn a passing one.  A disagreement between them is a fact the reader
    is entitled to, so it is named rather than dropped.
    """

    capped = three_arms[VERDICT_ARM]
    verdicts = {name: entry["verdict"] for name, entry in capped["members"].items()}
    disagreements = {
        name: {
            arm: {
                "whole_tail": three_arms[arm]["members"][name]["whole_tail_matches_registered"],
                "cold": three_arms[arm]["members"][name]["cold_matches_registered"],
            }
            for arm in REPORTED_ARMS
        }
        for name in verdicts
        if any(
            three_arms[arm]["members"][name]["whole_tail_matches_registered"] is not True
            or three_arms[arm]["members"][name]["cold_matches_registered"] is not True
            for arm in REPORTED_ARMS
        )
    }
    return {
        "decided_on": VERDICT_ARM,
        "member_verdicts": verdicts,
        "holds": set(verdicts.values()) == {"pass"},
        "failing_members": {
            name: capped["members"][name]
            for name, verdict in verdicts.items()
            if verdict != "pass"
        },
        "indeterminate_is_not_a_pass": True,
        "reported_arms_that_disagree_with_the_registered_sign": disagreements,
        "the_reported_arms_carry_no_verdict": (
            "mandatory_three_arm_reporting: the uncapped and component-equalized arms are "
            "published and are never a gate, in either direction"
        ),
    }


def train_direction_arms(
    train_rows: Sequence[Mapping[str, Any]], scorer: Scorer
) -> dict[str, Any]:
    """The nine published arms on coldstart_train, exactly as the manifest cut them.

    ``direction_constraint.train_check_is_non_evidentiary``: THE DIRECTIONS
    REGISTERED IN THE PLAN WERE TAKEN FROM coldstart_train, so a check of those
    directions on coldstart_train is a tautology, not a test, and this is NOT
    run as a gate.  What it is instead is a REPRODUCTION check on the pipeline:
    if any of the nine arms fails to reproduce the value sealed in the plan,
    the population, the extractor or the u-axis moved between the sealing of
    the manifest and this run, and every number in the plan would then be
    describing a different dataset.  That is a HALT.
    """

    rows = [dict(row) for row in train_rows]
    arms: dict[str, tuple[list[dict[str, Any]], list[float] | None]] = {
        "unweighted": (rows, None),
        "component_equalized": (rows, equalized_weights(rows)),
    }
    for cap in CAP_GRID:
        arms[f"cap_{cap}"] = (cap_survivors(rows, cap), None)
    for name, era_rows in era_partition(rows).items():
        arms[f"era_{name}"] = ([dict(row) for row in era_rows], None)

    out: dict[str, Any] = {}
    for label, (arm_rows, weights) in arms.items():
        effective = weights or [1.0] * len(arm_rows)
        cold_pairs = [
            (row, weight)
            for row, weight in zip(arm_rows, effective, strict=True)
            if row["cold"]
        ]
        cold_rows = [row for row, _w in cold_pairs]
        cold_weights = [weight for _r, weight in cold_pairs]
        entry: dict[str, Any] = {
            "rows": len(arm_rows),
            "cold_rows": len(cold_rows),
            "cold_consumed": sum(1 for row in cold_rows if row["consumed"]),
            "effective_sample_size": kish_effective_sample_size(effective),
            "cold_effective_sample_size": kish_effective_sample_size(cold_weights),
        }
        for source, name, _sign in CORE_FAMILY:
            table = scorer.member_tables[source]
            entry[name] = {
                "whole_tail": _round(
                    weighted_association(arm_rows, weights, source, table)
                ),
                "cold": _round(
                    weighted_association(cold_rows, cold_weights, source, table)
                ),
            }
        out[label] = entry
    return out


def reproduce_registered_train_readings(
    measured: Mapping[str, Any], prereg: Mapping[str, Any]
) -> dict[str, Any]:
    """Every sealed train reading, re-derived and compared within tolerance.

    A mismatch is a HALT -- not because a sign moved, but because it would mean
    the dataset moved under the plan.  A registered sign that does not match
    its OWN train reading is neither a pass nor a failure: it is published, and
    the six such readings already known are listed in advance at
    ``direction_constraint.expected_train_readings``.
    """

    registered = prereg["direction_constraint"]["registered_directions"]
    tolerance = float(
        prereg["direction_constraint"]["train_check_is_non_evidentiary"]["tolerance"]
    )
    mismatches: list[dict[str, Any]] = []
    compared = 0
    for member, block in registered.items():
        for arm, sealed in block["measurement"].items():
            arm_measured = measured.get(arm)
            if arm_measured is None:
                mismatches.append({"member": member, "arm": arm, "why": "arm not computed"})
                continue
            for key in (
                "rows",
                "cold_rows",
                "cold_consumed",
                "effective_sample_size",
                "cold_effective_sample_size",
            ):
                compared += 1
                if abs(float(arm_measured[key]) - float(sealed[key])) > tolerance:
                    mismatches.append(
                        {
                            "member": member,
                            "arm": arm,
                            "field": key,
                            "sealed": sealed[key],
                            "measured": arm_measured[key],
                        }
                    )
            for key in ("whole_tail", "cold"):
                compared += 1
                sealed_value = sealed[key]
                value = arm_measured[member][key]
                if value is None or abs(float(value) - float(sealed_value)) > tolerance:
                    mismatches.append(
                        {
                            "member": member,
                            "arm": arm,
                            "field": key,
                            "sealed": sealed_value,
                            "measured": value,
                        }
                    )
    known = prereg["direction_constraint"]["expected_train_readings"]
    published_non_matches = {
        f"{member}_{arm}_{reading}": measured[arm][member][reading]
        for member, block in registered.items()
        for arm, _sealed in block["measurement"].items()
        for reading in ("whole_tail", "cold")
        if measured[arm][member][reading] is not None
        and measured[arm][member][reading]
        * (1.0 if block["registered_direction"] == "positive" else -1.0)
        <= 0
    }
    return {
        "tolerance": tolerance,
        "comparisons": compared,
        "mismatches": mismatches,
        "reproduces_the_sealed_measurement": not mismatches,
        "this_is_a_reproduction_check_not_a_test": (
            "direction_constraint.train_check_is_non_evidentiary: the registered signs were "
            "TAKEN from coldstart_train, so checking them there is a tautology. What is "
            "checked here is that the pipeline reproduces the sealed numbers; a mismatch "
            "means the dataset moved, and that is the HALT."
        ),
        "registered_signs_not_matching_their_own_train_reading": published_non_matches,
        "these_were_registered_in_advance": known,
        "and_none_of_them_is_a_gate": True,
    }


def secondary_anchor_report(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """``direction_constraint.text`` (b) on the third cohort.

    ``secondary_anchor`` is not a split this fit holds rows for; it is an
    aggregate cohort published at ``feature-analysis.json#associations``.  The
    whole-tail sign is therefore checkable and is checked.  The COLD half of
    clause (c) is NOT checkable on it and is reported INDETERMINATE, never as
    passed: ``dataset-manifest.json#coldstart.privacy.item_level_rows`` is false,
    so no cold subset of that cohort exists to measure.  This is v1's reading
    carried forward unchanged, and it is a stated limitation of both plans
    rather than a result.
    """

    associations = analysis.get("associations", {})
    members: dict[str, Any] = {}
    verdicts: list[str] = []
    for source, name, sign in CORE_FAMILY:
        cohort = associations.get(source, {}).get("secondary_anchor", {})
        difference = cohort.get("difference")
        if difference is None:
            verdict = "indeterminate"
        elif difference * sign > 0:
            verdict = "pass"
        else:
            verdict = "fail"
        verdicts.append(verdict)
        members[name] = {
            "required_direction": "positive" if sign > 0 else "negative",
            "whole_tail_difference": difference,
            "available": cohort.get("available"),
            "missing": cohort.get("missing"),
            "verdict": verdict,
        }
    return {
        "cohort": "feature-analysis.json#associations.*.secondary_anchor",
        "whole_tail_checked": True,
        "whole_tail_verdict": "fail" if "fail" in verdicts else "pass",
        "cold_subpopulation_checked": False,
        "cold_subpopulation_verdict": "indeterminate",
        "why_the_cold_half_is_indeterminate": [
            "secondary_anchor is an aggregate cohort in feature-analysis.json, not a split this",
            "fit holds rows for, and dataset-manifest.json#coldstart.privacy.item_level_rows is",
            "false. No cold subset of it exists to measure, so clause (c) cannot be evaluated on",
            "this cohort by any run of this script. Reported as indeterminate, never as passed.",
            "Carried unchanged from the v1 fit, which reached the same wall.",
        ],
        "members": members,
    }


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def binomial_tail(successes: int, trials: int, probability: float) -> float | None:
    """One-sided exact P(X >= successes), the registered small-sample guard."""

    if trials <= 0:
        return None
    if probability <= 0.0:
        return 0.0 if successes > 0 else 1.0
    if probability >= 1.0:
        return 1.0
    total = 0.0
    for value in range(successes, trials + 1):
        total += math.exp(
            log_comb(trials, value)
            + value * math.log(probability)
            + (trials - value) * math.log1p(-probability)
        )
    return min(1.0, total)


def cold_auc(
    scored: Sequence[tuple[float, Mapping[str, Any]]],
    weights: Sequence[float] | None = None,
) -> float | None:
    """P(a consumed cold row outranks a non-consumed one), ties counted a half.

    Threshold-free and volume-free, so it cannot be flattered or punished by an
    ablation landing at a different admitted volume -- which is exactly why
    ``guard_b`` requires it IN ADDITION to lift, not instead of it.

    Under a weighting the same quantity is the weighted Mann-Whitney statistic:
    every ordered pair contributes the product of its two weights.  That arm is
    reported and never decides.
    """

    if weights is None:
        positives = sorted(score for score, row in scored if row["consumed"])
        negatives = sorted(score for score, row in scored if not row["consumed"])
        if not positives or not negatives:
            return None
        wins = 0.0
        for value in positives:
            below = _count_strictly_below(negatives, value)
            equal = _count_equal(negatives, value)
            wins += below + equal / 2.0
        return wins / (len(positives) * len(negatives))

    positive_pairs = [
        (score, weight)
        for (score, row), weight in zip(scored, weights, strict=True)
        if row["consumed"]
    ]
    negative_pairs = [
        (score, weight)
        for (score, row), weight in zip(scored, weights, strict=True)
        if not row["consumed"]
    ]
    if not positive_pairs or not negative_pairs:
        return None
    wins = 0.0
    for value, weight in positive_pairs:
        for other, other_weight in negative_pairs:
            if value > other:
                wins += weight * other_weight
            elif value == other:
                wins += weight * other_weight / 2.0
    denominator = sum(weight for _s, weight in positive_pairs) * sum(
        weight for _s, weight in negative_pairs
    )
    return wins / denominator if denominator else None


def _count_strictly_below(ordered: Sequence[float], value: float) -> int:
    low, high = 0, len(ordered)
    while low < high:
        middle = (low + high) // 2
        if ordered[middle] < value:
            low = middle + 1
        else:
            high = middle
    return low


def _count_equal(ordered: Sequence[float], value: float) -> int:
    low, high = 0, len(ordered)
    while low < high:
        middle = (low + high) // 2
        if ordered[middle] <= value:
            low = middle + 1
        else:
            high = middle
    return high - _count_strictly_below(ordered, value)


def admission_report(
    rows: Sequence[Mapping[str, Any]],
    score_of,
    threshold: float,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Everything ``per_split_reporting`` asks for at one threshold, on one arm.

    An unscoreable row is counted in the denominator and never in the admitted
    set: fail closed, and the two shapes therefore share a denominator.

    ``weights`` is the component-equalized reporting arm.  Under it there are
    no integer counts and therefore no exact binomial -- inventing an effective
    n and rounding would smuggle a new fitted quantity into a guard -- so the
    p-value is reported as unavailable rather than approximated.  That is the
    first of the four reasons the plan registers a cap as the estimator.
    """

    weighted = weights is not None
    row_weights = list(weights) if weighted else [1.0] * len(rows)
    scored = [
        (score_of(row), row, weight)
        for row, weight in zip(rows, row_weights, strict=True)
    ]
    scoreable = [(score, row, weight) for score, row, weight in scored if score is not None]
    admitted = [(row, weight) for score, row, weight in scoreable if score >= threshold]
    cold = [
        (row, weight) for row, weight in zip(rows, row_weights, strict=True) if row["cold"]
    ]
    cold_scored = [
        (score, row, weight) for score, row, weight in scoreable if row["cold"]
    ]
    cold_admitted = [
        (row, weight) for score, row, weight in cold_scored if score >= threshold
    ]
    warm = [
        (row, weight)
        for row, weight in zip(rows, row_weights, strict=True)
        if not row["cold"]
    ]
    warm_admitted = [(row, weight) for row, weight in admitted if not row["cold"]]

    def mass(pairs: Sequence[tuple[Mapping[str, Any], float]]) -> float:
        return sum(weight for _row, weight in pairs)

    def consumed_mass(pairs: Sequence[tuple[Mapping[str, Any], float]]) -> float:
        return sum(weight for row, weight in pairs if row["consumed"])

    total_mass = sum(row_weights)
    cold_mass = mass(cold)
    cold_admitted_mass = mass(cold_admitted)
    cold_base = consumed_mass(cold) / cold_mass if cold_mass else None
    cold_precision = (
        consumed_mass(cold_admitted) / cold_admitted_mass if cold_admitted_mass else None
    )
    lift = (
        cold_precision / cold_base
        if cold_precision is not None and cold_base not in (None, 0.0)
        else None
    )
    cold_consumed_admitted = sum(1 for row, _w in cold_admitted if row["consumed"])
    by_level: dict[str, int] = {}
    for row, _weight in cold_admitted:
        key = str(row.get("level"))
        by_level[key] = by_level.get(key, 0) + 1

    report: dict[str, Any] = {
        "arm": "component_equalized" if weighted else "unweighted_rows",
        "items": len(rows),
        "scoreable": len(scoreable),
        "family_unscoreable": len(rows) - len(scoreable),
        "history_arm_unreadable": sum(1 for row in rows if not history_readable(row)),
        "level_arm_unreadable": sum(1 for row in rows if not level_readable(row)),
        "threshold": threshold,
        "admitted": len(admitted),
        "admitted_fraction": _round(mass(admitted) / total_mass, 6) if total_mass else None,
        "cold_candidates": len(cold),
        "cold_candidate_share": _round(cold_mass / total_mass, 6) if total_mass else None,
        "cold_admitted": len(cold_admitted),
        "cold_admitted_fraction": (
            _round(cold_admitted_mass / cold_mass, 6) if cold_mass else None
        ),
        "cold_admitted_by_level": dict(sorted(by_level.items())),
        "cold_base_rate": _round(cold_base, 6),
        "cold_selected_precision": _round(cold_precision, 6),
        "cold_lift": _round(lift, 4),
        "cold_admitted_consumed": cold_consumed_admitted,
        "cold_auc": _round(
            cold_auc(
                [(score, row) for score, row, _w in cold_scored],
                [weight for _s, _r, weight in cold_scored] if weighted else None,
            ),
            6,
        ),
        "warm_candidates": len(warm),
        "warm_admitted": len(warm_admitted),
        "warm_selected_precision": (
            _round(consumed_mass(warm_admitted) / mass(warm_admitted), 6)
            if mass(warm_admitted)
            else None
        ),
        "effective_components": effective_components(rows),
        "cold_effective_components": effective_components([row for row, _w in cold]),
    }
    if weighted:
        report["binomial_p"] = None
        report["binomial_not_applicable_under_weights"] = (
            "component_aware_estimation.why_a_cap_and_not_component_equalized_weights (1): "
            "under fractional weights there are no integer counts and no exact test. A "
            "p-value over a fractional pseudo-count is not a p-value, so none is reported."
        )
        report["effective_sample_size"] = kish_effective_sample_size(row_weights)
        report["cold_effective_sample_size"] = kish_effective_sample_size(
            [weight for _row, weight in cold]
        )
    else:
        report["binomial_p"] = (
            None
            if cold_base is None or not cold_admitted
            else float(
                f"{binomial_tail(cold_consumed_admitted, len(cold_admitted), cold_base):.6g}"
            )
        )
    return report


def three_arm_admission(
    arms: Mapping[str, Arm], score_of, threshold: float
) -> dict[str, Any]:
    """``mandatory_three_arm_reporting`` applied to an admitted-fraction table."""

    return {
        name: {
            **admission_report(arms[name].rows, score_of, threshold, arms[name].weights),
            "arm": name,
            "carries_the_verdict": arms[name].carries_the_verdict,
        }
        for name in ARM_NAMES
    }


# ---------------------------------------------------------------------------
# Threshold selection -- label-free
# ---------------------------------------------------------------------------


class AdmissionCurve:
    """Admitted fractions at any threshold, on one arm, precomputed once.

    The arm supplies the row set and the weighting; the sweep supplies the
    candidate values.  Keeping the two apart is what
    ``threshold_rule.why_the_sweep_set_is_uncapped_while_the_test_is_capped``
    asks for: restricting the candidate VALUES to capped rows would make the
    threshold's own admissible values depend on the cap draw for no gain, while
    the qualifying TEST is a verdict-carrying admitted fraction and is capped.
    """

    def __init__(self, arm: Arm, score_of) -> None:
        weights = arm.weights or [1.0] * len(arm.rows)
        scored = [
            (score_of(row), row, weight)
            for row, weight in zip(arm.rows, weights, strict=True)
        ]
        scoreable = sorted(
            ((score, row, weight) for score, row, weight in scored if score is not None),
            key=lambda triple: triple[0],
        )
        self.arm = arm.name
        self.total = sum(weights)
        self.cold_total = sum(
            weight for row, weight in zip(arm.rows, weights, strict=True) if row["cold"]
        )
        self._scores = [score for score, _row, _weight in scoreable]
        self._cold_scores = [score for score, row, _w in scoreable if row["cold"]]
        self._suffix = self._suffix_mass([weight for _s, _r, weight in scoreable])
        self._cold_suffix = self._suffix_mass(
            [weight for _s, row, weight in scoreable if row["cold"]]
        )

    @staticmethod
    def _suffix_mass(weights: Sequence[float]) -> list[float]:
        suffix = [0.0] * (len(weights) + 1)
        for index in range(len(weights) - 1, -1, -1):
            suffix[index] = suffix[index + 1] + weights[index]
        return suffix

    @staticmethod
    def _lower_bound(ordered: Sequence[float], value: float) -> int:
        low, high = 0, len(ordered)
        while low < high:
            middle = (low + high) // 2
            if ordered[middle] < value:
                low = middle + 1
            else:
                high = middle
        return low

    def at(self, threshold: float) -> tuple[float, float]:
        """(overall admitted fraction, cold admitted fraction) at ``threshold``.

        Ties are resolved by admitting all rows AT or above the threshold,
        exactly as ``threshold_rule`` registers.
        """

        overall = self._suffix[self._lower_bound(self._scores, threshold)]
        cold = self._cold_suffix[self._lower_bound(self._cold_scores, threshold)]
        return (
            overall / self.total if self.total else 0.0,
            cold / self.cold_total if self.cold_total else 0.0,
        )

    def cold_admitted_rows(self, threshold: float) -> int:
        """A COUNT of rows, not a mass: what the volume match is stated in."""

        return len(self._cold_scores) - self._lower_bound(self._cold_scores, threshold)


def candidate_thresholds(rows: Sequence[Mapping[str, Any]], score_of) -> list[float]:
    """The distinct scores of ALL rows of the sweep cohort, DESCENDING."""

    return sorted(
        {score for score in (score_of(row) for row in rows) if score is not None},
        reverse=True,
    )


def select_threshold(
    sweep_rows: Sequence[Mapping[str, Any]], arms: Mapping[str, Arm], score_of
) -> dict[str, Any]:
    """``verdict_rule.threshold_rule``, exactly as registered.

    Sweep the distinct train scores DESCENDING and take the FIRST (highest)
    threshold at which the CAPPED overall admitted fraction lies in [0.10, 0.30]
    AND the CAPPED cold admitted fraction lies in [0.10, 0.25].  The highest
    qualifying threshold is the smallest admission consistent with the
    registered floors, which is the conservative end of the band and the one
    that least risks turning the map into noise.

    Reads no label: both quantities are ratios of masses of rows above a score,
    and coldness is a property of the ledger, not of the outcome.
    """

    scores = candidate_thresholds(sweep_rows, score_of)
    curve = AdmissionCurve(arms[VERDICT_ARM], score_of)
    feasible = [
        score
        for score in scores
        if _in_selection_bands(*curve.at(score))
    ]
    return {
        "rule": (
            "highest score in the uncapped train sweep set at which the CAPPED overall "
            f"admitted fraction is in {list(OVERALL_BAND)} and the CAPPED cold admitted "
            f"fraction is in {list(COLD_THRESHOLD_RULE_BAND)}"
        ),
        "reads_no_label": True,
        "sweep_set": "all coldstart_train distinct scores, uncapped",
        "qualifying_test_computed_on": "the component-capped estimation set at the registered cap",
        "sweep_candidates": len(scores),
        "feasible_threshold_count": len(feasible),
        "threshold": feasible[0] if feasible else None,
        "bands": {
            "overall": list(OVERALL_BAND),
            "cold_for_the_selection_rule": list(COLD_THRESHOLD_RULE_BAND),
            "cold_for_the_verdict": list(COLD_VERDICT_BAND),
        },
    }


def _in_selection_bands(overall_fraction: float, cold_fraction: float) -> bool:
    return (
        OVERALL_BAND[0] <= overall_fraction <= OVERALL_BAND[1]
        and COLD_THRESHOLD_RULE_BAND[0] <= cold_fraction <= COLD_THRESHOLD_RULE_BAND[1]
    )


def frontier_report(
    sweep_rows: Sequence[Mapping[str, Any]],
    arms: Mapping[str, Arm],
    score_of,
    label: str,
) -> dict[str, Any]:
    """``feasible_threshold_count`` in all three arms, published whatever the verdict.

    ``threshold_rule.published`` is an affirmative obligation that does not
    depend on the outcome: the frontier on coldstart_train and on
    coldstart_eval must both be auditable the way v1's and v2's were.
    """

    scores = candidate_thresholds(sweep_rows, score_of)
    per_arm: dict[str, Any] = {}
    for name in ARM_NAMES:
        curve = AdmissionCurve(arms[name], score_of)
        feasible = 0
        at_cold_floor = None
        at_overall_ceiling = None
        for score in scores:
            overall_fraction, cold_fraction = curve.at(score)
            if _in_selection_bands(overall_fraction, cold_fraction):
                feasible += 1
            if at_cold_floor is None and cold_fraction >= COLD_THRESHOLD_RULE_BAND[0]:
                at_cold_floor = {
                    "threshold": score,
                    "cold_admitted_fraction": _round(cold_fraction, 6),
                    "overall_admitted_fraction": _round(overall_fraction, 6),
                }
            if at_overall_ceiling is None and overall_fraction >= OVERALL_BAND[1]:
                at_overall_ceiling = {
                    "threshold": score,
                    "cold_admitted_fraction": _round(cold_fraction, 6),
                    "overall_admitted_fraction": _round(overall_fraction, 6),
                }
        per_arm[name] = {
            "arm": name,
            "carries_the_verdict": name == VERDICT_ARM,
            "feasible_threshold_count": feasible,
            "any_single_threshold_satisfies_both_bands": bool(feasible),
            "first_threshold_reaching_the_cold_floor": at_cold_floor,
            "threshold_at_the_overall_ceiling": at_overall_ceiling,
        }
    return {
        "cohort": label,
        "sweep_candidates": len(scores),
        "arms": per_arm,
        "feasible_threshold_count": per_arm[VERDICT_ARM]["feasible_threshold_count"],
    }


def select_ablation_threshold(
    sweep_rows: Sequence[Mapping[str, Any]],
    arms: Mapping[str, Arm],
    scorer: Scorer,
    registered_cold_admitted: int,
) -> dict[str, Any]:
    """``guard_b.ablation_threshold_rule``: volume-matched on capped train cold rows.

    The HIGHEST threshold at which the ablation admits at least as many cold
    train rows as the registered shape does.  Where the achievable counts are
    coarse the tie breaks toward the higher threshold -- fewer admitted rows,
    which raises the ablation's precision and therefore makes it HARDER for the
    registered shape to beat.  The conservative direction is chosen on purpose.

    The count it matches on is a count of CAPPED rows, because a cold admitted
    count is a verdict-carrying statistic.
    """

    scores = candidate_thresholds(sweep_rows, scorer.ablation_score)
    curve = AdmissionCurve(arms[VERDICT_ARM], scorer.ablation_score)
    for score in scores:
        admitted = curve.cold_admitted_rows(score)
        if admitted >= registered_cold_admitted:
            return {
                "t_ablation": score,
                "cold_admitted_on_train": admitted,
                "matched_against_registered_cold_admitted": registered_cold_admitted,
                "counted_on": "the component-capped coldstart_train estimation set",
                "rule": (
                    "highest ablation threshold admitting at least as many capped cold train "
                    "rows as the registered shape; ties break toward the higher threshold"
                ),
                "volume_match_reached": True,
            }
    return {
        "t_ablation": scores[-1] if scores else None,
        "cold_admitted_on_train": (
            curve.cold_admitted_rows(scores[-1]) if scores else 0
        ),
        "matched_against_registered_cold_admitted": registered_cold_admitted,
        "counted_on": "the component-capped coldstart_train estimation set",
        "rule": "the ablation cannot reach the registered volume; its floor threshold is used",
        "volume_match_reached": False,
    }


# ---------------------------------------------------------------------------
# Guard (c): the frozen scorer, unchanged
# ---------------------------------------------------------------------------


def frozen_score(
    row: Mapping[str, Any], means: Sequence[float], scales: Sequence[float]
) -> float:
    """``recall_map.relevance_score`` exactly as it stands today.

    Five z-terms: feature 0 is ``level_is_schema`` and features 1-4 are the
    matured-history terms.  This is the thing guard (c) compares against, so it
    is recomputed here rather than approximated.
    """

    matured = int(row["matured"])
    consumed = int(row["matured_consumed"])
    streak = int(row["trailing_nonconsumed"])
    features = (
        float(row["level"] == "schema"),
        round(math.log1p(matured), ROUND_PLACES),
        round(math.log1p(matured - consumed), ROUND_PLACES),
        round(math.log1p(streak), ROUND_PLACES),
        round(consumed / matured, ROUND_PLACES) if matured else 0.0,
    )
    return sum(
        (value - mean) / scale
        for value, mean, scale in zip(features, means, scales, strict=True)
    )


def guard_c_warm_non_degradation(
    eval_rows: Sequence[Mapping[str, Any]], scorer: Scorer, threshold: float
) -> dict[str, Any]:
    """Volume-matched precision on warm eval rows, on ONE arm's row set.

    ``anti_fiat_guards.all_four_are_computed_on_the_capped_estimation_set``:
    guard (c)'s K is a count of CAPPED warm rows the frozen scorer admits.  Its
    ``K < 20`` fallback is unchanged and is now more likely to fire, because the
    capped warm cohort is smaller -- that is a tightening, and the condition it
    falls back to, ``revised_warm_lift >= 1.0``, is unchanged.

    Precision falls with admitted volume, so comparing the two policies at their
    own thresholds would measure the THRESHOLDS rather than the scorers.
    Matching on K compares the two ORDERINGS of the same rows, which is the
    question guard (c) is asking.  The root goal requires that the map does not
    start returning noise; a revision that admits cold nodes by degrading the
    ordering of warm ones has moved the problem, not solved it.
    """

    warm = [row for row in eval_rows if not row["cold"]]
    frozen = [
        (frozen_score(row, scorer.legacy_means, scorer.legacy_scales), row) for row in warm
    ]
    frozen_admitted = [row for score, row in frozen if score >= FROZEN_THRESHOLD]
    k = len(frozen_admitted)
    frozen_precision = (
        sum(1 for row in frozen_admitted if row["consumed"]) / k if k else None
    )

    revised = sorted(
        ((score, row) for score, row in ((scorer.score(r), r) for r in warm) if score is not None),
        key=lambda pair: -pair[0],
    )
    top_k = [row for _score, row in revised[:k]]
    revised_precision = (
        sum(1 for row in top_k if row["consumed"]) / len(top_k) if top_k else None
    )

    revised_admitted = [row for score, row in revised if score >= threshold]
    warm_base = sum(1 for row in warm if row["consumed"]) / len(warm) if warm else None
    revised_warm_lift = (
        (sum(1 for row in revised_admitted if row["consumed"]) / len(revised_admitted))
        / warm_base
        if revised_admitted and warm_base
        else None
    )

    indeterminate = k < GUARD_C_MINIMUM_K
    if indeterminate:
        holds = revised_warm_lift is not None and revised_warm_lift >= 1.0
        binding_on = "revised_warm_lift >= 1.0 (registered fallback, K < 20)"
    else:
        holds = (
            revised_precision is not None
            and frozen_precision is not None
            and revised_precision >= frozen_precision
        )
        binding_on = "revised_warm_precision_at_matched_volume >= frozen_warm_precision"
    return {
        "cohort": "coldstart_eval, warm rows only",
        "warm_rows": len(warm),
        "warm_effective_components": effective_components(warm),
        "K_admitted_by_the_frozen_scorer": k,
        "matched_volume_comparison": "indeterminate" if indeterminate else "computed",
        "frozen_warm_precision": _round(frozen_precision, 6),
        "revised_warm_precision_at_matched_volume": _round(revised_precision, 6),
        "revised_warm_admitted": len(revised_admitted),
        "warm_base_rate": _round(warm_base, 6),
        "revised_warm_lift": _round(revised_warm_lift, 4),
        "binding_condition": binding_on,
        "holds": bool(holds),
        "computed_before_the_holdout_is_opened": True,
    }


# ---------------------------------------------------------------------------
# Parameter digest
# ---------------------------------------------------------------------------


def parameter_payload(
    scorer: Scorer, threshold: float, t_ablation: float | None
) -> dict[str, Any]:
    """Exactly the quantities a holdout row could have influenced.

    Every calibration table is in here, not a summary of one: the published step
    function IS the definition, so a table that changed by one knot changes this
    digest and the withheld-holdout reproduction fails.
    """

    return {
        "shape": "score = N + F, N = (L + H) / 2",
        "members": [
            {"source_field": source, "name": name, "signed_weight": sign}
            for source, name, sign in CORE_FAMILY
        ],
        "threshold": threshold,
        "t_ablation": t_ablation,
        "tables": {
            "level": scorer.level_table.payload(),
            "history": scorer.history_table.payload(),
            "family_composite": scorer.family_table.payload(),
            "members": {
                source: scorer.member_tables[source].payload()
                for source, _name, _sign in CORE_FAMILY
            },
        },
    }


def parameter_digest(payload: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(payload))


def fit_on_train(
    train_rows: Sequence[Mapping[str, Any]],
    legacy_means: Sequence[float],
    legacy_scales: Sequence[float],
    cap: int = ESTIMATION_CAP,
) -> tuple[Scorer, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Every fitted quantity, derived from the train split and nothing else.

    The calibration tables are fitted on ALL of coldstart_train, uncapped, per
    ``component_aware_estimation.what_the_cap_does_not_touch``; the cap enters
    only where a verdict-carrying admitted fraction is tested.
    """

    scorer = fit_tables(train_rows, legacy_means, legacy_scales)
    arms = build_arms(train_rows, cap)
    selection = select_threshold(train_rows, arms, scorer.score)
    threshold = selection["threshold"]
    if threshold is None:
        return scorer, selection, {}, {}
    registered = admission_report(arms[VERDICT_ARM].rows, scorer.score, threshold)
    ablation = select_ablation_threshold(
        train_rows, arms, scorer, registered["cold_admitted"]
    )
    payload = parameter_payload(scorer, threshold, ablation["t_ablation"])
    return scorer, selection, ablation, payload


# ---------------------------------------------------------------------------
# Per-split reporting
# ---------------------------------------------------------------------------


def concentration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How many INDEPENDENT fitting units a split really holds.

    The inverse Simpson index over component item shares.  The enlargement that
    made the 400-cold minimum reachable bought those cold rows and paid in
    independence: months of pre-August work share cache keys, so train collapsed
    from 12.05 effective components to 2.78 and eval from 115.96 to 4.18.  A
    generalization claim resting on 2.78 independent units has to be stated
    rather than hidden behind a 12344-row count, so this sits beside every row
    count in the published revision.
    """

    sizes: dict[str, int] = {}
    for row in rows:
        sizes[row["component_id"]] = sizes.get(row["component_id"], 0) + 1
    total = sum(sizes.values())
    if not total:
        return {
            "components": 0,
            "effective_components": None,
            "largest_component_items": 0,
            "largest_component_share": None,
        }
    simpson = sum((count / total) ** 2 for count in sizes.values())
    largest = max(sizes.values())
    return {
        "components": len(sizes),
        "effective_components": round(1.0 / simpson, 2) if simpson else None,
        "largest_component_items": largest,
        "largest_component_share": _round(largest / total, 6),
    }


def estimation_reporting(arms: Mapping[str, Arm], cap: int) -> dict[str, Any]:
    """What the split looks like as an ESTIMATION population, at the registered cap.

    ``split_protocol.per_split_reporting`` requires capped items, capped cold
    candidates, the retained share, effective components whole and cold, the
    largest component share on the capped set, and the per-era composition
    capped and uncapped.  This is the block the republished manifest carries,
    so the population the manifest describes is the population estimated from.
    """

    capped = arms[VERDICT_ARM]
    uncapped = arms["uncapped"]
    capped_cold = [row for row in capped.rows if row["cold"]]
    return {
        "estimation_cap": cap,
        "namespace": CAP_SUBSAMPLE_NAMESPACE,
        "capped_items": len(capped.rows),
        "capped_cold_candidates": len(capped_cold),
        "capped_cold_candidate_share": (
            _round(len(capped_cold) / len(capped.rows), 6) if capped.rows else None
        ),
        "capped_cold_consumed": sum(1 for row in capped_cold if row["consumed"]),
        "capped_cold_base_rate": (
            _round(
                sum(1 for row in capped_cold if row["consumed"]) / len(capped_cold), 6
            )
            if capped_cold
            else None
        ),
        "items_retained_share": (
            _round(len(capped.rows) / len(uncapped.rows), 6) if uncapped.rows else None
        ),
        "components_under_the_cap": len({row["component_id"] for row in capped.rows}),
        "components_uncapped": len({row["component_id"] for row in uncapped.rows}),
        "no_component_is_lost_to_the_cap": (
            len({row["component_id"] for row in capped.rows})
            == len({row["component_id"] for row in uncapped.rows})
        ),
        "effective_components_capped": effective_components(capped.rows),
        "effective_components_uncapped": effective_components(uncapped.rows),
        "cold_effective_components_capped": effective_components(capped_cold),
        "largest_component_share_capped": concentration(capped.rows)[
            "largest_component_share"
        ],
        "largest_component_share_uncapped": concentration(uncapped.rows)[
            "largest_component_share"
        ],
        "clears_the_whole_split_effective_component_floor": (
            (effective_components(capped.rows) or 0.0)
            >= EFFECTIVE_COMPONENT_FLOOR_WHOLE_SPLIT
        ),
        "clears_the_cold_effective_component_floor": (
            (effective_components(capped_cold) or 0.0) >= EFFECTIVE_COMPONENT_FLOOR_COLD
        ),
        "effective_component_floors": {
            "whole_split": EFFECTIVE_COMPONENT_FLOOR_WHOLE_SPLIT,
            "cold_subpopulation": EFFECTIVE_COMPONENT_FLOOR_COLD,
        },
        "era_composition_capped": era_composition(capped),
        "era_composition_uncapped": era_composition(uncapped),
        # A list, not a string: this block is republished into the manifest,
        # where check_privacy caps every published string at 200 characters.
        "the_cap_is_an_estimator_not_a_scorer_input": [
            "score = N + F is computed identically for capped and uncapped rows;",
            "the cap is not visible to src/living_memory/recall_map.py at all, and",
            "no calibration table is fitted on a capped row set.",
        ],
    }


def split_reporting(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    published: Mapping[str, Any],
    arms: Mapping[str, Arm] | None = None,
    cap: int = ESTIMATION_CAP,
) -> dict[str, Any]:
    """Every label-bearing item of ``split_protocol.per_split_reporting``.

    Computed from the rows this fit actually holds, then cross-checked against
    the manifest, so a split that drifted between the two artifacts is caught
    here rather than believed.  For the sealed partition this is computed inside
    the single pass, from the same materialization as the verdict.
    """

    cold = [row for row in rows if row["cold"]]
    total = len(rows)
    report = {
        "ref": (
            "artifacts/recall-map/relevance/dataset-manifest.json"
            f"#coldstart.splits.{name}"
        ),
        "items": total,
        "consumed": sum(1 for row in rows if row["consumed"]),
        "rate": _round(sum(1 for row in rows if row["consumed"]) / total, 6) if total else None,
        "cold_candidates": len(cold),
        "cold_candidate_share": _round(len(cold) / total, 6) if total else None,
        "cold_consumed": sum(1 for row in cold if row["consumed"]),
        "cold_base_rate": (
            _round(sum(1 for row in cold if row["consumed"]) / len(cold), 6) if cold else None
        ),
        "history_arm_unreadable": sum(1 for row in rows if not history_readable(row)),
        "level_arm_unreadable": sum(1 for row in rows if not level_readable(row)),
        "family_unscoreable": sum(1 for row in rows if family_unscoreable(row)),
        "membership_digest": published["membership_digest"],
        "component_digest_list": (
            "published in full at the manifest ref above; membership_digest is sha256 over "
            "the canonical JSON of exactly that sorted list, so citing the digest binds the "
            "list. dataset-manifest.json#privacy.item_level_rows stays false."
        ),
        "concentration": concentration(rows),
        "estimation": estimation_reporting(arms or build_arms(rows, cap), cap),
    }
    report["agrees_with_manifest"] = {
        key: report[key] == published[key]
        for key in ("items", "consumed", "cold_candidates", "cold_consumed")
    }
    report["agrees_with_manifest"]["cold_base_rate"] = (
        report["cold_base_rate"] == published["cold_rate"]
    )
    return report


def uniform_draw_residual_risk(
    population: Mapping[str, Any], manifest_block: Mapping[str, Any]
) -> dict[str, Any]:
    """``assignment_rule.residual_risk_stated``, which is binding ON THIS FIT.

    The shipped assignment balances on component ITEM COUNT, which the blindness
    invariant permits explicitly -- and component size correlates with coldness,
    so balancing on size moves per-split cold counts indirectly.  That is a real
    leakage surface.  The plan's mitigation is not to deny it but to make its
    SIZE auditable: recompute the v1 seed-uniform per-component draw over the
    published component list and publish its per-split cold counts beside the
    shipped assignment's, so anyone can read off how much the size-balancing
    moved.

    This is a component-level population statistic -- it never goes through
    :meth:`SealedSplits.open` and never scores a row -- so it is available
    before the seal is broken, exactly as the plan says it is.
    """

    module = extractor()
    placement = module._uniform_draw_assignment(
        population["all_components"], seed=module.SPLIT_SEED
    )
    every = [row for rows in population["rows"].values() for row in rows]
    per_split: dict[str, Any] = {}
    for name in module.SPLIT_NAMES:
        rows = [row for row in every if placement[row["component_id"]] == name]
        cold = [row for row in rows if row["cold"]]
        per_split[name] = {
            "items": len(rows),
            "cold_candidates": len(cold),
            **concentration(rows),
        }
    shipped = {
        name: {
            "items": manifest_block["splits"][name]["items"],
            "cold_candidates": manifest_block["splits"][name]["cold_candidates"],
            "effective_components": manifest_block["splits"][name]["concentration"][
                "effective_components"
            ],
        }
        for name in module.SPLIT_NAMES
    }
    return {
        "why_this_is_published": (
            "coldstart-prereg-v2.json#split_protocol.assignment_rule.residual_risk_stated "
            "binds coldstart-fit-v2 to recompute the v1 seed-uniform draw over the published "
            "component list and publish its per-split cold counts alongside the shipped "
            "assignment's, so the size of the size-balancing leakage is auditable"
        ),
        "seed": module.SPLIT_SEED,
        "rule": "uniform per-component sha256 draw, fractions 0.60 / 0.20 / 0.20",
        "recomputed_by_the_fit": True,
        "reads_no_row_and_breaks_no_seal": True,
        "seed_uniform_draw": per_split,
        "shipped_assignment": shipped,
        "cold_difference_shipped_minus_uniform": {
            name: shipped[name]["cold_candidates"] - per_split[name]["cold_candidates"]
            for name in module.SPLIT_NAMES
        },
        "reading": [
            "The two rules are not close, and the difference is disclosed rather than denied.",
            "The seed-uniform draw is unbiased in COMPONENTS and these components differ in",
            "size by three orders of magnitude, so it partitions ITEMS arbitrarily -- which is",
            "why the manifest rejected it and why its holdout concentrates.",
            "Read the cold_difference row as the magnitude of the leakage surface the",
            "invariant's explicit permission to balance on item count opens up. It is the only",
            "such surface: the v1 role the carry-over reads is itself a size-balanced placement",
            "over permitted inputs, so it inherits this surface rather than adding a second.",
        ],
    }


# ---------------------------------------------------------------------------
# The single holdout pass
# ---------------------------------------------------------------------------


def holdout_single_pass(
    splits: SealedSplits,
    scorer: Scorer,
    threshold: float,
    t_ablation: float | None,
    published: Mapping[str, Any],
    parameter_sha256: str,
    cap: int = ESTIMATION_CAP,
) -> dict[str, Any]:
    """``single_holdout_pass``: ONE materialization, every quantity from it.

    The capped estimation set is drawn INSIDE this pass, before any statistic,
    and held fixed for all of them.  ``the_cap_is_not_a_second_read``: it is a
    subset of rows already materialized here, selected by a sha256 order over
    identity fields, so drawing it costs no additional read of the partition.

    The ablation shares the pass on purpose.  It exists to stop a bias
    correction passing as learning, so running it as a second evaluation would
    have the guard designed to protect the verdict destroy the verdict instead.
    Both shapes read the same sealed tables and differ by one term.

    Anything not computed here is UNAVAILABLE.  There is no second read for a
    forgotten statistic, another threshold, or a hypothesis this pass suggests.
    """

    rows = splits.open(
        SEALED_SPLIT,
        unseal=SealBreak(
            reason=(
                "prereg split_protocol.seal.ordering step 5: the fitted parameter digest is "
                "on disk and every pre-holdout gate has been cleared"
            ),
            parameter_sha256=parameter_sha256,
        ),
    )
    arms = build_arms(rows, cap)
    capped = arms[VERDICT_ARM]
    registered_arms = three_arm_admission(arms, scorer.score, threshold)
    ablation_arms = (
        three_arm_admission(arms, scorer.ablation_score, t_ablation)
        if t_ablation is not None
        else None
    )
    capped_cold = [row for row in capped.rows if row["cold"]]
    floors = {
        "effective_components_whole_split": effective_components(capped.rows),
        "whole_split_floor": EFFECTIVE_COMPONENT_FLOOR_WHOLE_SPLIT,
        "clears_whole_split_floor": (
            (effective_components(capped.rows) or 0.0)
            >= EFFECTIVE_COMPONENT_FLOOR_WHOLE_SPLIT
        ),
        "effective_components_cold": effective_components(capped_cold),
        "cold_floor": EFFECTIVE_COMPONENT_FLOOR_COLD,
        "clears_cold_floor": (
            (effective_components(capped_cold) or 0.0) >= EFFECTIVE_COMPONENT_FLOOR_COLD
        ),
        "checked_inside_the_single_pass_because_a_pre_check_would_be_a_second_read": True,
    }
    per_era = {
        name: {
            "capped": admission_report(
                [row for row in capped.rows if row["era"] == name], scorer.score, threshold
            ),
            "uncapped": admission_report(
                [row for row in rows if row["era"] == name], scorer.score, threshold
            ),
        }
        for name in ERA_NAMES
    }
    return {
        "rule": "coldstart_holdout materialized exactly once; every quantity below is from it",
        "materializations": splits.sealed_reads,
        "holdout_first_read_after_parameters_sealed": True,
        "parameter_sha256_sealed_before_this_pass": parameter_sha256,
        "estimation_cap": cap,
        "capped_estimation_set_drawn_once_inside_this_pass": True,
        "split": split_reporting(SEALED_SPLIT, rows, published, arms, cap),
        "effective_component_floors": floors,
        "registered_shape": registered_arms[VERDICT_ARM],
        "registered_shape_arms": registered_arms,
        "ablation_shape": (
            None if ablation_arms is None else ablation_arms[VERDICT_ARM]
        ),
        "ablation_shape_arms": ablation_arms,
        "per_era": per_era,
        "computed_in_this_one_pass": [
            "the registered shape's score for every holdout row, from the sealed tables",
            "the ablation shape's score for every holdout row, from the same sealed tables",
            "the component-capped estimation set, drawn once before any statistic",
            "admitted sets under the registered threshold and under t_ablation",
            "overall and cold admitted fractions, under both, in all three arms",
            "cold admitted counts and cold consumed-among-admitted counts, under both",
            "holdout_cold_lift and holdout_cold_auc, under both, in all three arms",
            "the exact binomial p-value for the registered shape on the capped set",
            "cold_admitted_by_level, history_arm_unreadable, family_unscoreable",
            "the holdout's capped effective_components, whole split and cold",
            "the per-era decomposition, capped and uncapped",
            "warm counts and precisions, for the record only -- guard (c) binds on eval",
        ],
    }


# ---------------------------------------------------------------------------
# The decision procedure
# ---------------------------------------------------------------------------


def check_bands(report: Mapping[str, Any], label: str) -> list[str]:
    """Guard (d), on one split.  The bands are v1's, carried unchanged."""

    failures: list[str] = []
    overall = report["admitted_fraction"]
    cold = report["cold_admitted_fraction"]
    if overall is None or not OVERALL_BAND[0] <= overall <= OVERALL_BAND[1]:
        failures.append(f"{label}.overall_outside_{list(OVERALL_BAND)}")
    if cold is None or not COLD_VERDICT_BAND[0] <= cold <= COLD_VERDICT_BAND[1]:
        failures.append(f"{label}.cold_outside_{list(COLD_VERDICT_BAND)}")
    return failures


def verify_blindness_invariant(manifest_block: Mapping[str, Any]) -> dict[str, Any]:
    """Step 2's third clause, re-executed rather than read.

    ``blindness_invariant_binding`` requires the manifest to publish the
    assignment function's name and its inputs plus "a re-executable check that
    recomputes the assignment from those inputs alone and reproduces the
    published membership digests", and it HALTS the fit if the assignment reads
    any cold statistic.

    Both halves are settled by one act: replay the assignment giving it the
    published input table as its ONLY data argument.  If the three membership
    digests come back, the assignment provably is a function of exactly those
    published fields -- and the published fields are the opaque digest, the item
    count, the v1 partition roles and the observed-map flag.  No cold statistic
    is among them, and none could be hiding, because nothing else was passed.
    """

    module = extractor()
    split = manifest_block.get("split", {})
    inputs = split.get("assignment_inputs")
    if not isinstance(inputs, Mapping):
        raise ColdStartFitError(
            "HALT at step 2: the manifest publishes no coldstart.split.assignment_inputs, so "
            "the assignment cannot be recomputed from its declared inputs alone"
        )
    published_digests = {
        name: manifest_block["splits"][name]["membership_digest"] for name in module.SPLIT_NAMES
    }
    receipt = module.assignment_reproduction_receipt(inputs, published_digests)

    # What the replay was actually handed, field by field.  The invariant's
    # exclusion list is a consumption label, a feature value, a score and a
    # coldness statistic; a field that is not published cannot be read.
    published_fields = sorted({key for entry in inputs["components"] for key in entry})
    banned = sorted(
        field
        for field in published_fields
        if any(token in field for token in ("cold", "consum", "matur", "label", "score"))
    )
    if banned:
        raise ColdStartFitError(
            "HALT at step 2: the published assignment inputs carry a forbidden statistic: "
            + ", ".join(banned)
        )
    return {
        "function": inputs["function"]["name"],
        "verifier": "scripts/recall_map_coldstart_dataset.py:replay_published_assignment",
        "re_executable_by_a_third_party": (
            "python3 scripts/recall_map_coldstart_dataset.py verify-assignment"
        ),
        "components_replayed": len(inputs["components"]),
        "per_component_fields_published": published_fields,
        "inputs_used": receipt["inputs_used"],
        "membership_digests_reproduced": receipt["membership_digests"] == published_digests,
        "reproduced_membership_digests": receipt["membership_digests"],
        "no_cold_statistic_among_the_published_inputs": True,
        "why_that_is_settled_rather_than_asserted": [
            "The replay is handed the published component table and nothing else -- no rows,",
            "no database, no placement dict. It reproduces all three membership digests from",
            "it. So the shipped assignment IS a function of exactly those published fields,",
            "and a statistic that is not published cannot have been read.",
        ],
        "the_fourth_and_fifth_inputs": manifest_block["split"]["assignment_inputs"][
            "blindness_invariant_reading"
        ],
        "this_fit_concurs": (
            "The invariant names seed, opaque digest and item count. This assignment also "
            "reads each component's set of v1 partition ROLES and whether it holds an "
            "observed-map delivery event. Neither is a consumption label, a feature value, a "
            "score or a coldness statistic, which is the invariant's own exclusion list and "
            "the narrow HALT trigger. The decisive argument is composition: a v1 role is "
            "_balanced_assignment(sizes, seed) evaluated over the v1 subpopulation, i.e. a "
            "deterministic function of the three permitted inputs, so reading it reads no "
            "kind of information the invariant does not already permit. The observed-map flag "
            "is pre-registered by the plan itself at split_protocol.forced_assignment, so the "
            "plan cannot forbid its own registered rule. The fit therefore proceeds, with the "
            "reading recorded here rather than waved through."
        ),
    }


def simpson_guard(
    arms: Mapping[str, Arm], scorer: Scorer, prereg: Mapping[str, Any]
) -> dict[str, Any]:
    """``era_exchangeability.simpson_guard``: pooled agreement no stratum supports.

    If a core member's POOLED capped reading matches its registered sign while
    EVERY measurable era contradicts it on the same reading, that member FAILS.
    An era below the measurability bar is INDETERMINATE for the guard and
    cannot contradict anything, and where fewer than two eras are measurable
    the guard is reported VACUOUS -- never as passed.  The plan registers in
    advance that it is EXPECTED to be vacuous on coldstart_eval, because eval's
    pre-August slice is a single component and can contribute at most
    ``estimation_cap`` capped rows.
    """

    clause = prereg["direction_constraint"]["cold_subpopulation_clause"]
    min_cold = int(clause["minimum_cold_rows_for_a_direction_check"])
    min_cold_consumed = int(clause["minimum_consumed_cold_rows_for_a_direction_check"])
    capped = arms[VERDICT_ARM]
    pooled = arm_direction_reading(capped, scorer, prereg)

    eras: dict[str, Any] = {}
    for name, era_rows in era_partition(capped.rows).items():
        cold = [row for row in era_rows if row["cold"]]
        cold_consumed = sum(1 for row in cold if row["consumed"])
        measurable_cold = len(cold) >= min_cold and cold_consumed >= min_cold_consumed
        measurable_whole = len(era_rows) >= ERA_MEASURABLE_WHOLE_TAIL_ROWS
        entry: dict[str, Any] = {
            "capped_rows": len(era_rows),
            "capped_cold_rows": len(cold),
            "capped_cold_consumed": cold_consumed,
            "measurable_for_the_whole_tail_reading": measurable_whole,
            "measurable_for_the_cold_reading": measurable_cold,
            "members": {},
        }
        for source, member, sign in CORE_FAMILY:
            table = scorer.member_tables[source]
            whole = weighted_association(era_rows, None, source, table)
            cold_value = weighted_association(cold, None, source, table)
            entry["members"][member] = {
                "whole_tail": _round(whole),
                "cold": _round(cold_value),
                "whole_tail_matches_registered": (
                    None if whole is None or not measurable_whole else whole * sign > 0
                ),
                "cold_matches_registered": (
                    None
                    if cold_value is None or not measurable_cold
                    else cold_value * sign > 0
                ),
            }
        eras[name] = entry

    failures: dict[str, Any] = {}
    for _source, member, _sign in CORE_FAMILY:
        for reading, key in (
            ("whole_tail", "measurable_for_the_whole_tail_reading"),
            ("cold", "measurable_for_the_cold_reading"),
        ):
            measurable = [name for name, entry in eras.items() if entry[key]]
            if len(measurable) < 2:
                continue
            pooled_matches = pooled["members"][member][f"{reading}_matches_registered"]
            if pooled_matches is not True:
                continue
            if all(
                eras[name]["members"][member][f"{reading}_matches_registered"] is False
                for name in measurable
            ):
                failures[f"{member}.{reading}"] = {
                    "pooled": pooled["members"][member][reading],
                    "eras": {
                        name: eras[name]["members"][member][reading] for name in measurable
                    },
                }
    measurable_eras = sorted(
        {
            name
            for name, entry in eras.items()
            if entry["measurable_for_the_whole_tail_reading"]
            or entry["measurable_for_the_cold_reading"]
        }
    )
    vacuous = len(measurable_eras) < 2
    return {
        "rule": prereg["era_exchangeability"]["simpson_guard"]["rule"],
        "measurable_eras": measurable_eras,
        "vacuous": vacuous,
        "reported_as_vacuous_not_as_passed": vacuous,
        "registered_expectation_that_it_will_be_vacuous_on_eval": (
            prereg["era_exchangeability"]["simpson_guard"][
                "registered_expectation_that_it_will_be_vacuous_on_eval"
            ]
        ),
        "per_era": eras,
        "failures": failures,
        "holds": not failures,
    }


def era_component_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The one-line check ``era_exchangeability`` binds this fit to publish.

    eval's pre-August slice is one component of 1002 items and its August-era
    largest component holds 1571; 1002 + 1571 = 2573, which is eval's largest
    component overall.  That arithmetic is consistent with a SINGLE component
    straddling the era boundary, in which case eval's entire era difference is
    a within-component difference and the era cut on eval is not a cut at all.
    The plan does not assert it -- component identities are opaque digests --
    and requires the fit to settle it as a named field.
    """

    def largest(era_rows: Sequence[Mapping[str, Any]]) -> tuple[str | None, int]:
        sizes: dict[str, int] = {}
        for row in era_rows:
            sizes[row["component_id"]] = sizes.get(row["component_id"], 0) + 1
        if not sizes:
            return None, 0
        component = max(sorted(sizes), key=lambda key: sizes[key])
        return component, sizes[component]

    partition = era_partition(rows)
    pre_component, pre_items = largest(partition[ERA_ENLARGEMENT])
    august_component, august_items = largest(partition[ERA_PREEXISTING])
    overall_component, overall_items = largest(rows)
    return {
        "question": (
            "is coldstart_eval's pre-August component the same component as its largest "
            "August-era component?"
        ),
        "pre_august_components": len({row["component_id"] for row in partition[ERA_ENLARGEMENT]}),
        "pre_august_largest_component_items": pre_items,
        "august_era_largest_component_items": august_items,
        "largest_component_overall_items": overall_items,
        "the_same_component": bool(
            pre_component is not None and pre_component == august_component
        ),
        "the_largest_overall_is_that_component": bool(
            overall_component is not None and overall_component == pre_component
        ),
        "arithmetic_the_plan_flagged": (
            f"{pre_items} + {august_items} = {pre_items + august_items} against a largest "
            f"component overall of {overall_items}"
        ),
        "reading": (
            "one component straddles the era boundary, so eval's era difference is a "
            "within-component difference and the era cut on eval is not a cut"
            if pre_component is not None and pre_component == august_component
            else "the pre-August slice and the largest August-era component are DIFFERENT "
            "components, so the era cut on eval is a real cut between units"
        ),
        "component_identities_are_not_published": (
            "dataset-manifest.json#privacy.item_level_rows stays false; only the answer to "
            "the identity question and the item counts are published"
        ),
    }


def verify_split_structure_from_the_manifest(
    manifest_block: Mapping[str, Any], prereg: Mapping[str, Any], cap: int
) -> dict[str, Any]:
    """Decision step 2, WITHOUT READING A ROW.

    Four clauses, all answerable from ``dataset-manifest.json#coldstart``
    alone: each split's capped cold-candidate count at the registered cap
    clears ``minimum_cold_candidates_per_split``; the three component lists are
    pairwise disjoint; the published assignment satisfies the blindness
    invariant and reproduces its membership digests; and each split's capped
    effective components clears the whole-split floor.  Any failure is a HALT,
    and the plan forbids proceeding on a smaller split, a different cap or a
    lower floor.
    """

    minimum = int(prereg["split_protocol"]["minimum_cold_candidates_per_split"])
    table = manifest_block["enlargement"]["cap_feasibility"]["table"][str(cap)]
    splits = manifest_block["splits"]
    per_split: dict[str, Any] = {}
    failures: list[str] = []
    for name in (FIT_SPLIT, VERIFY_SPLIT, SEALED_SPLIT):
        entry = table[name]
        capped_cold = int(entry["cold_candidates"])
        capped_effective = float(entry["effective_components"])
        share = entry["cold_candidate_share"]
        clears_cold = capped_cold >= minimum
        clears_floor = capped_effective >= EFFECTIVE_COMPONENT_FLOOR_WHOLE_SPLIT
        non_zero_share = bool(share) and float(share) > 0.0
        per_split[name] = {
            "capped_items": entry["items"],
            "capped_cold_candidates": capped_cold,
            "capped_cold_candidate_share": share,
            "capped_effective_components": capped_effective,
            "items_retained_share": entry["items_retained_share"],
            "clears_the_cold_minimum": clears_cold,
            "clears_the_whole_split_effective_component_floor": clears_floor,
            "cold_share_is_non_zero": non_zero_share,
        }
        if not clears_cold:
            failures.append(f"{name}.capped_cold_candidates_below_{minimum}")
        if not clears_floor:
            failures.append(
                f"{name}.capped_effective_components_below_"
                f"{EFFECTIVE_COMPONENT_FLOOR_WHOLE_SPLIT}"
            )
        if not non_zero_share:
            failures.append(f"{name}.cold_share_is_zero")

    lists = {name: sorted(splits[name]["components"]) for name in splits}
    intersections = {
        f"{a}_{b}": len(set(lists[a]) & set(lists[b]))
        for a, b in (
            (FIT_SPLIT, VERIFY_SPLIT),
            (FIT_SPLIT, SEALED_SPLIT),
            (VERIFY_SPLIT, SEALED_SPLIT),
        )
    }
    if any(intersections.values()):
        failures.append("component_lists_are_not_pairwise_disjoint")
    return {
        "read_no_row": True,
        "minimum_cold_candidates_per_split": minimum,
        "estimation_cap": cap,
        "effective_component_floor_whole_split": EFFECTIVE_COMPONENT_FLOOR_WHOLE_SPLIT,
        "per_split": per_split,
        "pairwise_component_intersections": intersections,
        "pairwise_component_disjoint": not any(intersections.values()),
        "the_cap_cannot_lose_a_component": (
            "a cap keeps min(cap, n_c) rows of every component, which is at least one, so "
            "the three component lists under the cap are the three published lists and "
            "disjointness is untouched"
        ),
        "failures": failures,
        "holds": not failures,
    }


def halt_diagnostic(
    train_rows: Sequence[Mapping[str, Any]],
    train_arms: Mapping[str, Arm],
    eval_rows: Sequence[Mapping[str, Any]],
    eval_arms: Mapping[str, Arm],
    scorer: Scorer,
) -> dict[str, Any]:
    """The frontier and the measured fractions, published even after a HALT.

    ``verdict_rule.threshold_rule.published`` is an affirmative obligation that
    does not depend on the verdict: ``feasible_threshold_count`` on
    coldstart_train AND on coldstart_eval "must both be published, whatever the
    verdict, so the frontier is auditable the way v1's and v2's were".

    This REPAIRS NOTHING and step 9 is not in tension with it.  It selects no
    threshold, seals no parameter, moves no band, re-draws no split and never
    touches the sealed partition.
    """

    would_be = select_threshold(train_rows, train_arms, scorer.score)
    threshold = would_be["threshold"]
    return {
        "status": "diagnostic_only_no_parameter_was_sealed_and_no_verdict_rests_on_it",
        "why_published_after_a_halt": (
            "verdict_rule.threshold_rule.published requires feasible_threshold_count on "
            "coldstart_train and coldstart_eval in all three arms whatever the verdict"
        ),
        "why_this_is_not_a_repair": [
            "No threshold was selected, no parameter digest was sealed and no band, minimum,",
            "split, cap, floor or family member moved. The verdict remains the recorded HALT.",
            "The sealed partition was not opened: holdout_evaluations stays 0 and the",
            "SealedSplits capability guard was never constructed.",
        ],
        "frontier_on_train": frontier_report(
            train_rows, train_arms, scorer.score, "coldstart_train"
        ),
        "frontier_on_eval": frontier_report(
            eval_rows, eval_arms, scorer.score, "coldstart_eval"
        ),
        "threshold_the_rule_would_have_returned": threshold,
        "admission_at_that_threshold": (
            {
                "train": three_arm_admission(train_arms, scorer.score, threshold),
                "eval": three_arm_admission(eval_arms, scorer.score, threshold),
            }
            if threshold is not None
            else None
        ),
    }


def run_decision_procedure(
    splits: SealedSplits,
    population: Mapping[str, Any],
    prereg: Mapping[str, Any],
    policy: Mapping[str, Any],
    manifest_block: Mapping[str, Any],
    plan_sha256: str,
    seal_to_disk,
) -> dict[str, Any]:
    """``decision_procedure``, the nine steps, literally and in order.

    Every HALT is a stop, not a branch to a repair.  Step 9 forbids widening
    the family, re-signing a member, adding a branch, re-drawing a split,
    moving a band or a minimum, CHANGING estimation_cap, CHANGING either
    effective-component floor, switching the verdict onto a reported arm, or
    re-reading the holdout -- so where the plan says HALT this function records
    the diagnostic in full and returns; it never reaches for a lever.
    """

    steps: list[dict[str, Any]] = []
    fit_block = policy["selected_policy"]["fit"]
    means = fit_block["standardization_means"]
    scales = fit_block["scales"]
    result: dict[str, Any] = {"steps": steps, "halted_at": None}

    # What the run has actually computed so far.  A HALT after step 4 still
    # sealed a parameter digest to disk, so the artifact owes the reader those
    # parameters, the reproduction check over them and every guard number that
    # was reached -- hiding them would make the negative result less falsifiable
    # than the run that produced it.  ``holdout`` is absent unless step 7 ran,
    # which is what keeps the verdict path and the artifact honest.
    carried: dict[str, Any] = {}

    def halt(step: int, name: str, why: str) -> dict[str, Any]:
        result["halted_at"] = {"step": step, "name": name, "why": why}
        result["verdict"] = "negative_result"
        result["failed_steps"] = [
            entry["name"] for entry in steps if entry.get("verdict") in ("fail", "halt")
        ]
        if carried:
            result["pending"] = dict(carried)
        return result

    # -- step 1 -------------------------------------------------------------
    steps.append(
        {
            "step": 1,
            "name": "plan_seal_verified",
            "plan_sha256": plan_sha256,
            "expected": EXPECTED_PLAN_SHA256,
            "supersedes_plan_sha256": prereg["supersedes"].get("plan_sha256")
            if isinstance(prereg.get("supersedes"), Mapping)
            else None,
            "verified_before_anything_was_read": True,
            "verdict": "pass",
        }
    )

    # -- step 2: from the manifest alone, no row is read ---------------------
    cap_selection = select_estimation_cap(manifest_block)
    cap = int(cap_selection["estimation_cap"])
    structure = verify_split_structure_from_the_manifest(manifest_block, prereg, cap)
    blindness = verify_blindness_invariant(manifest_block)
    step2_ok = structure["holds"] and blindness["membership_digests_reproduced"]
    steps.append(
        {
            "step": 2,
            "name": "split_structure_cap_and_floors_from_the_manifest",
            "estimation_cap_selection": cap_selection,
            "split_structure": structure,
            "blindness_invariant_binding": blindness,
            "counting_published_rows_is_not_a_read": splits.sealed_reads == 0,
            "verdict": "pass" if step2_ok else "fail",
        }
    )
    if not step2_ok:
        return halt(2, "split_structure_cap_and_floors_from_the_manifest", "a step 2 clause failed")

    # -- step 3: coldstart_train ONLY ---------------------------------------
    train_rows = splits.open(FIT_SPLIT)
    scorer = fit_tables(train_rows, means, scales)
    train_arms = build_arms(train_rows, cap)
    carried.update({"estimation_cap": cap, "cap_selection": cap_selection, "train_arms": train_arms})
    measured_arms = train_direction_arms(train_rows, scorer)
    reproduction_check = reproduce_registered_train_readings(measured_arms, prereg)
    train_directions = three_arm_directions(train_arms, scorer, prereg)
    train_estimation = estimation_reporting(train_arms, cap)
    cold_floor_train = train_estimation["clears_the_cold_effective_component_floor"]
    step3_ok = reproduction_check["reproduces_the_sealed_measurement"] and cold_floor_train
    steps.append(
        {
            "step": 3,
            "name": "calibrate_on_train_and_reproduce_the_sealed_readings",
            "fitted_from_split": FIT_SPLIT,
            "rows_used_for_every_calibration_table": len(train_rows),
            "tables_are_fitted_uncapped": (
                "component_aware_estimation.what_the_cap_does_not_touch: every calibration is "
                "fitted on ALL of coldstart_train. The estimation cap is an estimator and is "
                "not an input to any table."
            ),
            "tables": {
                key: {"fitted_rows": table.n, "breakpoints": len(table.atoms) + len(table.knots)}
                for key, table in (
                    ("level", scorer.level_table),
                    ("history", scorer.history_table),
                    ("family_composite", scorer.family_table),
                    *((source, scorer.member_tables[source]) for source, _n, _s in CORE_FAMILY),
                )
            },
            "estimation_set": train_estimation,
            "published_train_readings_nine_arms": measured_arms,
            "reproduction_of_the_sealed_measurement": reproduction_check,
            "three_arm_direction_readings": train_directions,
            "the_train_direction_check_is_not_a_gate": (
                "direction_constraint.train_check_is_non_evidentiary: the registered signs "
                "were taken from coldstart_train, so a check of them here is a tautology. A "
                "registered sign not matching its own train reading is PUBLISHED, not failed."
            ),
            "cold_effective_component_floor": {
                "floor": EFFECTIVE_COMPONENT_FLOOR_COLD,
                "measured": train_estimation["cold_effective_components_capped"],
                "holds": cold_floor_train,
            },
            "verdict": "pass" if step3_ok else "fail",
        }
    )
    if not step3_ok:
        eval_rows_for_diagnostic = splits.open(VERIFY_SPLIT)
        steps[-1]["diagnostic"] = halt_diagnostic(
            train_rows,
            train_arms,
            eval_rows_for_diagnostic,
            build_arms(eval_rows_for_diagnostic, cap),
            scorer,
        )
        return halt(
            3,
            "calibrate_on_train_and_reproduce_the_sealed_readings",
            "the sealed train measurement did not reproduce, or the cold "
            "effective-component floor failed on train",
        )

    # -- step 4: label-free selection on CAPPED fractions, then SEAL --------
    selection = select_threshold(train_rows, train_arms, scorer.score)
    train_frontier = frontier_report(train_rows, train_arms, scorer.score, "coldstart_train")
    threshold = selection["threshold"]
    if threshold is None:
        steps.append(
            {
                "step": 4,
                "name": "select_threshold_and_seal_parameters",
                "threshold_selection": selection,
                "frontier_on_train": train_frontier,
                "feasible_threshold_count": selection["feasible_threshold_count"],
                "parameters_sealed": False,
                "verdict": "halt",
                "why": (
                    "the CAPPED feasible_threshold_count is 0: no single threshold puts the "
                    "overall and the cold admitted fraction inside their registered bands on "
                    "the capped estimation set of the fit split. "
                    "verdict_rule.threshold_rule.if_no_such_threshold_exists says HALT and "
                    "record a NEGATIVE RESULT; the holdout is not opened."
                ),
            }
        )
        return halt(4, "select_threshold_and_seal_parameters", "feasible_threshold_count is 0")

    train_admission_arms = three_arm_admission(train_arms, scorer.score, threshold)
    train_admission = train_admission_arms[VERDICT_ARM]
    ablation = select_ablation_threshold(
        train_rows, train_arms, scorer, train_admission["cold_admitted"]
    )
    payload = parameter_payload(scorer, threshold, ablation["t_ablation"])
    digest = parameter_digest(payload)
    seal_receipt = seal_to_disk(payload, digest)
    carried.update(
        {
            "scorer": scorer,
            "threshold": threshold,
            "ablation": ablation,
            "payload": payload,
            "digest": digest,
            "train_admission": train_admission,
            "train_admission_arms": train_admission_arms,
            "train_rows": train_rows,
            "train_estimation": train_estimation,
            "train_frontier": train_frontier,
            "train_directions": train_directions,
            "train_direction_reproduction": reproduction_check,
            "published_train_readings_nine_arms": measured_arms,
        }
    )
    steps.append(
        {
            "step": 4,
            "name": "select_threshold_and_seal_parameters",
            "threshold_selection": selection,
            "frontier_on_train": train_frontier,
            "feasible_threshold_count": selection["feasible_threshold_count"],
            "feasible_threshold_count_all_arms": {
                name: entry["feasible_threshold_count"]
                for name, entry in train_frontier["arms"].items()
            },
            "threshold": threshold,
            "admission_at_the_threshold": train_admission_arms,
            "ablation_threshold": ablation,
            "derived_fit_floor": prereg["verdict_rule"]["derived_fit_floor"]["value"],
            "parameter_sha256": digest,
            "parameters_sealed": True,
            "seal_receipt": seal_receipt,
            "nothing_here_read_a_label": True,
            "verdict": "pass",
        }
    )

    # -- step 5: coldstart_eval and secondary_anchor ------------------------
    eval_rows = splits.open(VERIFY_SPLIT)
    eval_arms = build_arms(eval_rows, cap)
    eval_estimation = estimation_reporting(eval_arms, cap)
    eval_directions = three_arm_directions(eval_arms, scorer, prereg)
    eval_verdict = binding_direction_verdict(eval_directions)
    analysis_path = REPO_ROOT / "artifacts/recall-map/relevance/feature-analysis.json"
    anchor = secondary_anchor_report(json.loads(analysis_path.read_text(encoding="utf-8")))
    guard_simpson = simpson_guard(eval_arms, scorer, prereg)
    era_identity = era_component_identity(eval_rows)
    cold_floor_eval = eval_estimation["clears_the_cold_effective_component_floor"]
    carried.update(
        {
            "eval_rows": eval_rows,
            "eval_arms": eval_arms,
            "eval_estimation": eval_estimation,
            "eval_directions": eval_directions,
            "eval_direction_verdict": eval_verdict,
            "era_component_identity_on_eval": era_identity,
            "secondary_anchor": anchor,
            "guards": {"simpson_guard_on_eval": guard_simpson},
        }
    )
    step5_ok = (
        eval_verdict["holds"]
        and anchor["whole_tail_verdict"] == "pass"
        and guard_simpson["holds"]
        and cold_floor_eval
    )
    steps.append(
        {
            "step": 5,
            "name": "registered_signs_on_eval_and_secondary_anchor",
            "estimation_set": eval_estimation,
            "three_arm_direction_readings": eval_directions,
            "binding_verdict": eval_verdict,
            "secondary_anchor": anchor,
            "simpson_guard": guard_simpson,
            "era_component_identity_on_eval": era_identity,
            "cold_effective_component_floor": {
                "floor": EFFECTIVE_COMPONENT_FLOOR_COLD,
                "measured": eval_estimation["cold_effective_components_capped"],
                "holds": cold_floor_eval,
            },
            "an_indeterminate_cold_check_blocks_and_never_passes": True,
            "verdict": "pass" if step5_ok else "fail",
        }
    )
    if not step5_ok:
        return halt(
            5,
            "registered_signs_on_eval_and_secondary_anchor",
            "a registered sign failed, was indeterminate, the Simpson guard fired, or the "
            "cold effective-component floor failed on eval",
        )

    # -- step 6: the pre-holdout gates --------------------------------------
    eval_admission_arms = three_arm_admission(eval_arms, scorer.score, threshold)
    eval_admission = eval_admission_arms[VERDICT_ARM]
    eval_frontier = frontier_report(eval_rows, eval_arms, scorer.score, "coldstart_eval")
    band_failures = check_bands(train_admission, "train") + check_bands(eval_admission, "eval")
    guard_c = guard_c_warm_non_degradation(eval_arms[VERDICT_ARM].rows, scorer, threshold)
    guard_c_uncapped = guard_c_warm_non_degradation(eval_rows, scorer, threshold)
    step6_ok = not band_failures and guard_c["holds"]
    carried.update(
        {
            "eval_admission": eval_admission,
            "eval_admission_arms": eval_admission_arms,
            "eval_frontier": eval_frontier,
        }
    )
    carried["guards"] = {
        **carried.get("guards", {}),
        "guard_c_warm_non_degradation": guard_c,
        "guard_c_warm_non_degradation_uncapped_reported_only": guard_c_uncapped,
        "guard_d_admitted_fraction_bands": {
            "id": "admitted_fraction_bands",
            "bands": {"overall": list(OVERALL_BAND), "cold": list(COLD_VERDICT_BAND)},
            "computed_on": "the component-capped estimation set of each split",
            "train": {
                "admitted_fraction": train_admission["admitted_fraction"],
                "cold_admitted_fraction": train_admission["cold_admitted_fraction"],
            },
            "eval": {
                "admitted_fraction": eval_admission["admitted_fraction"],
                "cold_admitted_fraction": eval_admission["cold_admitted_fraction"],
            },
            "holdout": "NOT MEASURED: the holdout was never opened",
            "failures": band_failures,
            "holds_on_train_and_eval": not band_failures,
        },
    }
    steps.append(
        {
            "step": 6,
            "name": "guard_d_bands_on_train_and_eval_and_guard_c_on_eval",
            "guard_d": {
                "bands": {"overall": list(OVERALL_BAND), "cold": list(COLD_VERDICT_BAND)},
                "computed_on": "the component-capped estimation set of each split",
                "train": train_admission,
                "eval": eval_admission,
                "train_all_arms": train_admission_arms,
                "eval_all_arms": eval_admission_arms,
                "failures": band_failures,
                "holds": not band_failures,
            },
            "guard_c": guard_c,
            "guard_c_uncapped_reported_only": guard_c_uncapped,
            "feasible_threshold_count_on_eval": eval_frontier["feasible_threshold_count"],
            "feasible_threshold_count_on_eval_all_arms": {
                name: entry["feasible_threshold_count"]
                for name, entry in eval_frontier["arms"].items()
            },
            "frontier_on_eval": eval_frontier,
            "pre_holdout_gate": (
                "if either fails here the holdout is NEVER OPENED and the outcome is a "
                "recorded NEGATIVE RESULT"
            ),
            "verdict": "pass" if step6_ok else "fail",
        }
    )
    if not step6_ok:
        return halt(6, "guard_d_bands_on_train_and_eval_and_guard_c_on_eval", "a gate failed")

    # -- step 7: the sealed partition, exactly once -------------------------
    holdout = holdout_single_pass(
        splits,
        scorer,
        threshold,
        ablation["t_ablation"],
        manifest_block["splits"][SEALED_SPLIT],
        digest,
        cap,
    )
    steps.append({"step": 7, "name": "single_holdout_pass", **holdout, "verdict": "executed"})

    # -- step 8: the verdict ------------------------------------------------
    registered = holdout["registered_shape"]
    ablated = holdout["ablation_shape"]
    lift = registered["cold_lift"]
    minimum_admitted = int(
        prereg["verdict_rule"]["companion_conditions"]["minimum_cold_admitted_on_holdout"]
    )
    guard_a = {
        "id": "holdout_cold_lift",
        "condition": f"holdout_cold_lift >= {GENERALIZATION_THRESHOLD}",
        "computed_on": "the component-capped holdout estimation set, inside the single pass",
        "holdout_cold_lift": lift,
        "holdout_cold_admitted": registered["cold_admitted"],
        "holdout_cold_base_rate": registered["cold_base_rate"],
        "holdout_cold_selected_precision": registered["cold_selected_precision"],
        "reported_arms": {
            name: {
                "cold_lift": holdout["registered_shape_arms"][name]["cold_lift"],
                "cold_admitted": holdout["registered_shape_arms"][name]["cold_admitted"],
            }
            for name in REPORTED_ARMS
        },
        "holds": lift is not None and lift >= GENERALIZATION_THRESHOLD,
    }
    lift_margin = (
        None
        if lift is None or ablated is None or ablated["cold_lift"] is None
        else round(lift - ablated["cold_lift"], 4)
    )
    auc_margin = (
        None
        if registered["cold_auc"] is None or ablated is None or ablated["cold_auc"] is None
        else round(registered["cold_auc"] - ablated["cold_auc"], 6)
    )
    guard_b = {
        "id": "family_ablation",
        "what_the_ablation_is": "S_ablation = N + 0.0, the same shape with the family neutralised",
        "computed_on": "the same capped holdout estimation set as the registered shape",
        "t_ablation": ablation["t_ablation"],
        "t_ablation_sealed_with_the_parameters": True,
        "registered": {"cold_lift": lift, "cold_auc": registered["cold_auc"]},
        "ablation": (
            None
            if ablated is None
            else {
                "cold_lift": ablated["cold_lift"],
                "cold_auc": ablated["cold_auc"],
                "cold_admitted": ablated["cold_admitted"],
                "cold_admitted_fraction": ablated["cold_admitted_fraction"],
                "admitted_fraction": ablated["admitted_fraction"],
                "admitted": ablated["admitted"],
            }
        ),
        "family_ablation_lift_margin": lift_margin,
        "family_ablation_auc_margin": auc_margin,
        "ablation_alone_clears_the_generalization_threshold": (
            ablated is not None
            and ablated["cold_lift"] is not None
            and ablated["cold_lift"] >= GENERALIZATION_THRESHOLD
        ),
        "ablation_train_admission": {
            "cold_admitted_on_train": ablation["cold_admitted_on_train"],
            "matched_against_registered_cold_admitted": ablation[
                "matched_against_registered_cold_admitted"
            ],
            "volume_match_reached": ablation.get("volume_match_reached"),
        },
        "ablation_on_eval": three_arm_admission(
            eval_arms, scorer.ablation_score, ablation["t_ablation"]
        )[VERDICT_ARM]
        if ablation["t_ablation"] is not None
        else None,
        "holds": bool(
            lift_margin is not None and lift_margin > 0 and auc_margin is not None and auc_margin > 0
        ),
        "strict": "a tie is a FAIL, on both comparisons",
    }
    holdout_bands = check_bands(registered, "holdout")
    guard_d = {
        "id": "admitted_fraction_bands",
        "bands": {"overall": list(OVERALL_BAND), "cold": list(COLD_VERDICT_BAND)},
        "computed_on": "the component-capped estimation set of each split",
        "train": {
            "admitted_fraction": train_admission["admitted_fraction"],
            "cold_admitted_fraction": train_admission["cold_admitted_fraction"],
        },
        "eval": {
            "admitted_fraction": eval_admission["admitted_fraction"],
            "cold_admitted_fraction": eval_admission["cold_admitted_fraction"],
        },
        "holdout": {
            "admitted_fraction": registered["admitted_fraction"],
            "cold_admitted_fraction": registered["cold_admitted_fraction"],
        },
        "failures": band_failures + holdout_bands,
        "holds": not (band_failures + holdout_bands),
    }
    binomial = {
        "test": "one-sided exact binomial P(X >= consumed | n = admitted cold, p = cold base rate)",
        "alpha": BINOMIAL_ALPHA,
        "computed_on": (
            "the component-capped holdout estimation set: integer counts of real rows, so "
            "the exact one-sided test is exact"
        ),
        "n": registered["cold_admitted"],
        "consumed": registered["cold_admitted_consumed"],
        "p": registered["cold_base_rate"],
        "p_value": registered["binomial_p"],
        "holds": registered["binomial_p"] is not None
        and registered["binomial_p"] <= BINOMIAL_ALPHA,
    }
    admitted_floor = {
        "minimum_cold_admitted_on_holdout": minimum_admitted,
        "counted_on": "the component-capped holdout estimation set",
        "observed": registered["cold_admitted"],
        "observed_uncapped": holdout["registered_shape_arms"]["uncapped"]["cold_admitted"],
        "strictly_greater_than_zero": registered["cold_admitted"] > 0,
        "holds": registered["cold_admitted"] >= minimum_admitted
        and registered["cold_admitted"] > 0,
    }
    floors = holdout["effective_component_floors"]
    effective_floor_guard = {
        "id": "holdout_effective_component_floors",
        "whole_split": {
            "floor": EFFECTIVE_COMPONENT_FLOOR_WHOLE_SPLIT,
            "measured": floors["effective_components_whole_split"],
            "holds": floors["clears_whole_split_floor"],
        },
        "cold_subpopulation": {
            "floor": EFFECTIVE_COMPONENT_FLOOR_COLD,
            "measured": floors["effective_components_cold"],
            "holds": floors["clears_cold_floor"],
        },
        "holds": floors["clears_whole_split_floor"] and floors["clears_cold_floor"],
    }
    result["pending"] = {
        **carried,
        "scorer": scorer,
        "estimation_cap": cap,
        "cap_selection": cap_selection,
        "threshold": threshold,
        "ablation": ablation,
        "payload": payload,
        "digest": digest,
        "train_admission": train_admission,
        "train_admission_arms": train_admission_arms,
        "eval_admission": eval_admission,
        "eval_admission_arms": eval_admission_arms,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "train_arms": train_arms,
        "eval_arms": eval_arms,
        "train_estimation": train_estimation,
        "eval_estimation": eval_estimation,
        "guards": {
            "guard_a_generalization": guard_a,
            "guard_b_family_ablation": guard_b,
            "guard_c_warm_non_degradation": guard_c,
            "guard_d_admitted_fraction_bands": guard_d,
            "binomial_guard": binomial,
            "minimum_cold_admitted": admitted_floor,
            "effective_component_floors_on_the_holdout": effective_floor_guard,
            "simpson_guard_on_eval": guard_simpson,
        },
        "holdout": holdout,
        "train_frontier": train_frontier,
        "eval_frontier": eval_frontier,
        "train_directions": train_directions,
        "train_direction_reproduction": reproduction_check,
        "published_train_readings_nine_arms": measured_arms,
        "eval_directions": eval_directions,
        "eval_direction_verdict": eval_verdict,
        "era_component_identity_on_eval": era_identity,
        "secondary_anchor": anchor,
    }
    return result


def finalize_verdict(result: dict[str, Any], reproduction: Mapping[str, Any]) -> dict[str, Any]:
    """Steps 8 and 9: the conjunction, with nothing left implicit.

    ``decision_procedure`` step 8 lists seven conditions and PASS requires ALL
    of them.  They are evaluated here as one conjunction over named booleans so
    that a reader can see which term carried the verdict, and step 9 is recorded
    beside it so that a FAIL cannot later be read as an invitation to repair.
    """

    pending = result["pending"]
    guards = pending["guards"]
    conditions = {
        "guard_a_holdout_cold_lift": guards["guard_a_generalization"]["holds"],
        "guard_b_family_ablation_both_strict": guards["guard_b_family_ablation"]["holds"],
        "guard_c_warm_non_degradation_cleared_at_step_6": guards[
            "guard_c_warm_non_degradation"
        ]["holds"],
        "guard_d_bands_on_every_split": guards["guard_d_admitted_fraction_bands"]["holds"],
        "cold_admitted_on_holdout_at_least_40_and_above_zero": guards["minimum_cold_admitted"][
            "holds"
        ],
        "binomial_p_at_or_below_alpha": guards["binomial_guard"]["holds"],
        "holdout_capped_effective_components_clear_both_floors": guards[
            "effective_component_floors_on_the_holdout"
        ]["holds"],
        "every_core_member_held_its_registered_sign_at_step_5": bool(
            pending["eval_direction_verdict"]["holds"]
        ),
        "withheld_holdout_reproduction_byte_identical": bool(reproduction["byte_identical"]),
    }
    passed = all(conditions.values())
    result["steps"].append(
        {
            "step": 8,
            "name": "verdict",
            "pass_requires_all_of": conditions,
            "guards": guards,
            "withheld_holdout_reproduction": {
                key: reproduction[key]
                for key in (
                    "train_only_reproduces_published",
                    "withheld_holdout_reproduces_published",
                    "byte_identical",
                    "published_parameter_sha256",
                )
            },
            "verdict": "pass" if passed else "fail",
        }
    )
    result["steps"].append(
        {
            "step": 9,
            "name": "no_repair_of_a_failed_condition",
            "applies": not passed,
            "text": (
                "Any other outcome is a recorded FAIL or NEGATIVE RESULT. Neither may be "
                "repaired by widening the family, re-signing a member, adding a branch, "
                "re-drawing a split, moving a band or a minimum, CHANGING estimation_cap, "
                "CHANGING either effective-component floor, switching the verdict onto a "
                "reported arm, or re-reading the holdout."
            ),
            "levers_not_reached_for": [
                "the family is closed by the prereg and was not widened or re-signed",
                "no branch on coldness, cohort, level or scope exists in the scorer",
                "the splits are the committed ones and reproduce their membership digests",
                "every band, minimum and threshold is v1's, carried unchanged",
                "estimation_cap was re-derived from the published grid by the registered "
                "floor-first rule and not chosen",
                "neither effective-component floor moved",
                "the verdict rests on the capped arm alone; the reported arms decided nothing",
                "the sealed partition was materialized exactly once",
            ],
            "verdict": "pass",
        }
    )
    result["verdict"] = "pass" if passed else "negative_result"
    result["failed_steps"] = [
        entry["name"] for entry in result["steps"] if entry.get("verdict") in ("fail", "halt")
    ]
    return result


def compute_reproduction(
    population: Mapping[str, Any],
    policy: Mapping[str, Any],
    published_digest: str,
    splits_match: Mapping[str, Any],
) -> dict[str, Any]:
    """``refit_checkability.withheld_holdout_reproduction``, the strong form.

    Re-fit from the train split alone, then again with the sealed partition
    physically absent from the input, and require both parameter digests --
    every calibration table, both thresholds and the member set -- to equal the
    published one byte for byte.  If any holdout row had touched any parameter
    the digests would differ, which converts an unfalsifiable claim about what a
    script did not read into a deterministic equality anyone can re-run.
    """

    fit_block = policy["selected_policy"]["fit"]
    means, scales = fit_block["standardization_means"], fit_block["scales"]

    _scorer, _selection, _ablation, train_only = fit_on_train(
        population["rows"][FIT_SPLIT], means, scales
    )
    train_only_digest = parameter_digest(train_only)

    withheld = {name: rows for name, rows in population["rows"].items() if name != SEALED_SPLIT}
    _scorer2, _selection2, _ablation2, withheld_payload = fit_on_train(
        withheld[FIT_SPLIT], means, scales
    )
    withheld_digest = parameter_digest(withheld_payload)

    return {
        "splits_match_manifest": dict(splits_match),
        "train_only_refit_sha256": train_only_digest,
        "train_only_reproduces_published": train_only_digest == published_digest,
        "withheld_holdout_refit_sha256": withheld_digest,
        "withheld_holdout_reproduces_published": withheld_digest == published_digest,
        "published_parameter_sha256": published_digest,
        "byte_identical": train_only_digest == withheld_digest == published_digest,
        "procedure": (
            "re-derive every calibration table, the registered threshold and t_ablation from "
            "the train split alone, and again with the sealed partition absent from the "
            "input; both parameter digests must equal the published one"
        ),
        "why_this_is_the_strong_form": (
            "the published step functions ARE the definition, not a summary of one, so a "
            "table that moved by a single knot changes the digest and this check fails"
        ),
    }


# ---------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------


def build_findings(
    outcome: Mapping[str, Any], manifest_block: Mapping[str, Any]
) -> dict[str, Any]:
    """What this run actually established, stated so it cannot be read past.

    A recorded negative result is a legitimate outcome and the honesty bound is
    explicit that it is preferable to a repaired one, so on a HALT the job here
    is to say precisely which condition failed and by how much.  On a PASS the
    job is the opposite and equally strict: to name what the result does NOT
    establish, so a single holdout reading is not read as more than it is.
    """

    halted = outcome.get("halted_at")
    steps = {entry["step"]: entry for entry in outcome["steps"]}
    common = {
        "what_changed_from_v2": [
            "Nothing about the scorer. score = N + F is carried bit for bit: the same two",
            "arms, the same rank_uniform_centered and rank_normal constructions, the same",
            "clip, the same averaging, the same no_cold_branch ban.",
            "What changed is the ESTIMATOR that judges it. v2 applied its component",
            "discipline to the SPLIT ASSIGNMENT and not to the estimator, and computed a",
            "direction reading over 12344 rows that are 2.78 effective units. Every",
            "verdict-carrying statistic in this run is computed on the component-capped",
            "estimation set instead, and the uncapped arm -- which IS v2's arm and IS the",
            "arm the recorded HALT was measured on -- is published beside it and decides",
            "nothing.",
            "And where the direction test BINDS moved: v3's signs were measured on",
            "coldstart_train, so checking them there would be a tautology. The train check",
            "is a reproduction check on the pipeline and the whole evidentiary weight sits",
            "on coldstart_eval and secondary_anchor, cohorts this plan had not read.",
        ],
        "the_estimator_is_not_the_result": [
            "A cap is a choice, and a choice made after seeing which signs failed can always",
            "be accused of being the choice that unfails them. Two structural defences are on",
            "the record and both are checkable: the cap is DERIVED from the published grid by",
            "a floor-first rule that reads no direction, no sign and no label -- substituting",
            "any floor in {77..113} returns the same cap -- and the registered signs are read",
            "off the WHOLE six-arm component-aware set by unanimity, so changing the cap to",
            "any other grid point changes no registered sign at all.",
        ],
    }

    if halted:
        step = halted["step"]
        entry = steps.get(step, {})
        detail: dict[str, Any] = {
            "halted_at_step": step,
            "step_name": halted["name"],
            "why": halted["why"],
        }
        if step == 2:
            detail["split_structure"] = entry.get("split_structure")
            detail["estimation_cap_selection"] = entry.get("estimation_cap_selection")
            detail["reading"] = [
                "The population could not be judged at all under the registered floors. No",
                "row was read, no table was fitted and no threshold was selected, so this",
                "says nothing about the scorer and everything about what the split holds.",
            ]
        elif step == 3:
            detail["reproduction_of_the_sealed_measurement"] = entry.get(
                "reproduction_of_the_sealed_measurement"
            )
            detail["cold_effective_component_floor"] = entry.get(
                "cold_effective_component_floor"
            )
            detail["reading"] = [
                "Either the pipeline no longer reproduces the numbers sealed into the plan --",
                "which means the population, the extractor or the u-axis moved and every",
                "number in the plan describes a different dataset -- or the capped cold",
                "subpopulation of coldstart_train does not carry the registered minimum of",
                "independent units. Neither is repairable inside this plan.",
            ]
        elif step == 4:
            detail["frontier_on_train"] = entry.get("frontier_on_train")
            detail["reading"] = [
                "No single threshold puts the overall and the cold admitted fraction inside",
                "their registered bands on the capped estimation set. This is the shape",
                "failure v1 recorded, now re-recorded under a component-aware estimator, and",
                "verdict_rule.threshold_rule.if_no_such_threshold_exists forbids widening a",
                "band to convert it into a victory. The holdout was not opened.",
            ]
        elif step == 5:
            detail["binding_verdict"] = entry.get("binding_verdict")
            detail["secondary_anchor"] = entry.get("secondary_anchor")
            detail["simpson_guard"] = entry.get("simpson_guard")
            detail["cold_effective_component_floor"] = entry.get(
                "cold_effective_component_floor"
            )
            detail["reading"] = [
                "A registered sign did not survive transport to a cohort this plan had not",
                "read. That is the test the plan moved onto coldstart_eval precisely so that",
                "it could fail honestly, and it is the outcome",
                "direction_constraint.tie_break_rule named in advance as the most likely one.",
                "The holdout was not opened. Step 9 forbids re-signing the member, dropping",
                "it, or re-drawing the split to find a cohort where it holds.",
            ]
        elif step == 6:
            guard_d = entry.get("guard_d") or {}
            guard_c = entry.get("guard_c") or {}
            detail["guard_d_failures"] = guard_d.get("failures")
            detail["guard_d_holds"] = guard_d.get("holds")
            detail["guard_d_admitted_fractions"] = {
                "train": {
                    "admitted_fraction": (guard_d.get("train") or {}).get("admitted_fraction"),
                    "cold_admitted_fraction": (guard_d.get("train") or {}).get(
                        "cold_admitted_fraction"
                    ),
                },
                "eval": {
                    "admitted_fraction": (guard_d.get("eval") or {}).get("admitted_fraction"),
                    "cold_admitted_fraction": (guard_d.get("eval") or {}).get(
                        "cold_admitted_fraction"
                    ),
                },
                "bands": {"overall": list(OVERALL_BAND), "cold": list(COLD_VERDICT_BAND)},
            }
            detail["guard_c"] = guard_c
            detail["guard_c_uncapped_reported_only"] = entry.get(
                "guard_c_uncapped_reported_only"
            )
            detail["the_shortfall"] = (
                None
                if guard_c.get("frozen_warm_precision") is None
                or guard_c.get("revised_warm_precision_at_matched_volume") is None
                else round(
                    guard_c["frozen_warm_precision"]
                    - guard_c["revised_warm_precision_at_matched_volume"],
                    6,
                )
            )
            detail["reading"] = [
                "The train-fitted threshold did not transport, or the revised ordering",
                "degraded warm precision at matched volume. Both are pre-holdout gates and",
                "failing them here costs no holdout: the sealed partition was never opened.",
                "Guard (c) is the guard that protects the root goal's one hard constraint --",
                "that the map must not start returning noise. A revision that admits cold",
                "nodes by degrading the ordering of warm ones has moved the problem, not",
                "solved it, and that is exactly what a matched-volume precision shortfall",
                "measures. Because it is matched on K, the comparison is between the two",
                "ORDERINGS of the same rows and not between the two thresholds.",
            ]
        return {
            "negative_result": detail,
            "the_holdout_is_still_virgin": (
                "holdout_evaluations is 0 for this run and no cold-start holdout number has "
                "ever existed under v1, v2 or v3. The partition remains available in full."
            ),
            "what_this_node_may_not_do_about_it": {
                "forbidden_by_step_9": [
                    "widen or re-sign the family",
                    "add a branch",
                    "re-draw a split",
                    "move a band, a minimum or the generalization threshold",
                    "change estimation_cap or either effective-component floor",
                    "switch the verdict onto a reported arm",
                    "re-read the holdout",
                ],
                "what_is_needed": (
                    "a decision above this node. This node has now seen the failing numbers "
                    "and may not author the plan that responds to them."
                ),
            },
            **common,
        }

    holdout_step = steps.get(7, {})
    verdict_step = steps.get(8, {})
    registered = holdout_step.get("registered_shape") or {}
    return {
        "positive_result": {
            "what_was_established": (
                "a node with no delivery history is admitted on candidate-time features it "
                "owns, and the admission generalizes: on a component-disjoint holdout read "
                "exactly once, against a threshold and an ablation threshold fixed and "
                "sealed to disk before the partition was opened"
            ),
            "holdout_cold_lift": registered.get("cold_lift"),
            "holdout_cold_admitted": registered.get("cold_admitted"),
            "holdout_cold_base_rate": registered.get("cold_base_rate"),
            "holdout_cold_selected_precision": registered.get("cold_selected_precision"),
            "pass_requires_all_of": verdict_step.get("pass_requires_all_of"),
        },
        "what_this_result_does_not_establish": [
            "It is ONE evaluation of ONE holdout partition of ONE store's history. It is not",
            "a live measurement and it makes no claim about queries_with_a_map, about the",
            "share of maps without clusters, or about the repeat rate of one label set --",
            "those are the root goal's live acceptance measures and they are measured after",
            "the wire, not here.",
            "The holdout is a historical partition of already-persisted state, not a",
            "post-deployment field population. policy.json#holdout_boundary names a different",
            "thing entirely and this run neither created nor read it.",
            "The generalization claim rests on the CAPPED effective component count of the",
            "holdout, not on its row count, and both are published above.",
            "split_protocol.known_residual_contamination stands: the family's aggregate",
            "directions were published before this plan was written, so the correct reading",
            "is 'directions and shape pre-registered, magnitudes and thresholds held out' and",
            "not 'a fully blind holdout'.",
        ],
        "the_ablation_is_the_control_that_carries_this": [
            "S_ablation = N + 0.0 IS the imputation-and-recalibration fix with no family in",
            "it -- the same tables, the same rows, the same fail-closed rule, differing in",
            "one term. If recalibration alone carried the result the ablation would match the",
            "registered shape and guard (b) would have failed.",
            "Both margins are published, and whether the ablation alone clears 1.5 is",
            "recorded as an explicit finding either way.",
        ],
        "what_the_wire_sibling_still_owes": (
            "this block is not in force. coldstart-wire-v3 promotes it into selected_policy "
            "and implements it in src/living_memory/recall_map.py in ONE commit, bumping "
            "RELEVANCE_POLICY_ID and RELEVANCE_POLICY_DIGEST together."
        ),
        "measured_on": {
            "split": "coldstart_holdout",
            "items": manifest_block["splits"][SEALED_SPLIT]["items"],
            "cold_candidates": manifest_block["splits"][SEALED_SPLIT]["cold_candidates"],
            "capped_items": (holdout_step.get("split") or {})
            .get("estimation", {})
            .get("capped_items"),
            "capped_cold_candidates": (holdout_step.get("split") or {})
            .get("estimation", {})
            .get("capped_cold_candidates"),
            "capped_effective_components": (holdout_step.get("split") or {})
            .get("estimation", {})
            .get("effective_components_capped"),
        },
        **common,
    }


def build_revision_block(
    outcome: Mapping[str, Any],
    population: Mapping[str, Any],
    reproduction: Mapping[str, Any],
    prereg: Mapping[str, Any],
    manifest_block: Mapping[str, Any],
    plan_sha256: str,
    residual_risk: Mapping[str, Any],
    splits: SealedSplits,
    split_reports: Mapping[str, Any],
    in_force_policy_id: str,
) -> dict[str, Any]:
    """``coldstart_revision``: the whole fitted result, and nothing in force.

    ``selected_policy`` is left byte-identical on purpose.  Bumping it here
    would require moving ``src/living_memory/recall_map.py`` in the same commit,
    because ``tests/test_recall_map_pool.py`` asserts the artifact's id and
    canonical digest against ``RELEVANCE_POLICY_ID`` and
    ``RELEVANCE_POLICY_DIGEST``, and ``scripts/prepare_field_store.py`` bakes
    ``selected_policy_sha256`` into a field-store receipt.  That source file
    belongs to the wire sibling, which promotes this block.
    """

    pending = outcome.get("pending") or {}
    guards = pending.get("guards", {})
    passed = outcome["verdict"] == "pass"
    holdout = pending.get("holdout")
    registered = (holdout or {}).get("registered_shape") or {}

    return {
        "id": REVISION_ID,
        "supersedes": "rank-calibrated-two-arm-coldstart-r1",
        "status": (
            "validated_on_the_sealed_holdout_awaiting_wire"
            if passed
            else "negative_result_not_wireable"
        ),
        "wireable": passed,
        "in_force_policy_unchanged": in_force_policy_id,
        "why_selected_policy_was_not_bumped": [
            "selected_policy is welded to src/living_memory/recall_map.py: "
            "tests/test_recall_map_pool.py asserts policy.json#selected_policy.id equals "
            "recall_map.RELEVANCE_POLICY_ID and that the canonical digest equals "
            "RELEVANCE_POLICY_DIGEST, and scripts/prepare_field_store.py bakes "
            "selected_policy_sha256 into a field-store receipt.",
            "Bumping it here would require moving that source file in this commit, which this "
            "node does not own. The sibling coldstart-wire-v3 promotes this block into "
            "selected_policy and implements it in the same commit, which is the only shape "
            "that keeps the two welded artifacts in step -- and splitting it the other way is "
            "what killed an earlier attempt.",
            "bindings.canonical_subobjects.selected_policy_sha256_unchanged records that this "
            "run left the in-force policy byte-identical.",
        ],
        "prereg": {
            "path": "artifacts/recall-map/relevance/coldstart-prereg-v3.json",
            "plan_sha256": plan_sha256,
            "expected_plan_sha256": EXPECTED_PLAN_SHA256,
            "seal_verified_before_any_split_was_read": True,
            "supersedes_plan": prereg["supersedes"],
            "superseded_plan_sha256": SUPERSEDED_PLAN_SHA256,
        },
        "component_aware_estimation": {
            "principle": prereg["component_aware_estimation"]["principle"],
            "estimation_cap": pending.get("estimation_cap", ESTIMATION_CAP),
            "cap_selection": pending.get("cap_selection"),
            "namespace": CAP_SUBSAMPLE_NAMESPACE,
            "effective_component_floors": {
                "whole_split": EFFECTIVE_COMPONENT_FLOOR_WHOLE_SPLIT,
                "cold_subpopulation": EFFECTIVE_COMPONENT_FLOOR_COLD,
            },
            "arms": {
                "capped": "carries every verdict",
                "uncapped": "reported only; this is v2's arm and the arm the HALT was measured on",
                "component_equalized": "reported only; no exact binomial exists under weights",
            },
            "no_verdict_rests_on_a_reported_arm": True,
            "the_cap_is_not_visible_to_the_scorer": (
                "score = N + F is computed identically for capped and uncapped rows and the "
                "cap never reaches src/living_memory/recall_map.py; no calibration table is "
                "fitted on a capped row set"
            ),
        },
        "scorer_shape": {
            "shape": "score = N + F",
            "N": "(L + H) / 2, bounded to [-0.5, +0.5]",
            "F": "rank_normal of the signed family composite, clipped to [-3, +3]",
            "no_branch_on_coldness": (
                "there is no test of matured == 0 anywhere in the scorer: no per-cohort, "
                "per-coldness, per-scope or per-level branch, no separate threshold, no bonus, "
                "no tie-break and no eligibility test. A cold row takes the same path as every "
                "other row."
            ),
            "nothing_is_fitted_on_labels": (
                "every calibration is a rank transform of the train split and both thresholds "
                "are selected by label-free rules that read only admitted fractions. That is "
                "strictly stronger than v1, which let labels touch the scalar threshold."
            ),
        },
        "dataset": {
            "manifest": "artifacts/recall-map/relevance/dataset-manifest.json#coldstart",
            "as_of": manifest_block["as_of"],
            "population": "opportunity_bearing",
            "enlarged_to": manifest_block["enlargement"]["window"]
            if isinstance(manifest_block.get("enlargement"), Mapping)
            and "window" in manifest_block["enlargement"]
            else None,
            "train_set": split_reports[FIT_SPLIT],
            "eval_set": split_reports[VERIFY_SPLIT],
            "holdout_set": split_reports[SEALED_SPLIT],
            "splits_reproduce_committed_manifest": reproduction["splits_match_manifest"],
            "pairwise_component_disjoint": True,
            "assignment_blindness_invariant": next(
                entry["blindness_invariant_binding"]
                for entry in outcome["steps"]
                if entry["name"] == "split_structure_cap_and_floors_from_the_manifest"
            ),
            "split_structure_verified_from_the_manifest_alone": next(
                entry["split_structure"]
                for entry in outcome["steps"]
                if entry["name"] == "split_structure_cap_and_floors_from_the_manifest"
            ),
            "effective_components_is_the_number_to_read": [
                "The three splits hold 12344 / 5285 / 4115 items but only "
                f"{split_reports[FIT_SPLIT]['concentration']['effective_components']} / "
                f"{split_reports[VERIFY_SPLIT]['concentration']['effective_components']} / "
                f"{split_reports[SEALED_SPLIT]['concentration']['effective_components']} "
                "effective components, because months of pre-August work share cache keys and "
                "the sealed identity rule keeps a component atomic.",
                "The enlargement that made the 400-cold minimum reachable is therefore not "
                "free: it bought cold candidates and paid in independent fitting units, "
                "collapsing train from 12.05 effective components and eval from 115.96.",
                "Any generalization claim made from this fit rests on that number, not on the "
                "row count, and it is published here so it cannot be read past.",
            ],
            "residual_risk_of_the_size_aware_assignment": residual_risk,
        },
        "generalization_threshold": {
            "quantity": prereg["verdict_rule"]["generalization_threshold_quantity"],
            "value": prereg["verdict_rule"]["generalization_threshold"],
            "comparison": prereg["verdict_rule"]["generalization_threshold_comparison"],
            "taken_from": "coldstart-prereg-v3.json#verdict_rule.generalization_threshold",
            "not_moved": prereg["verdict_rule"]["generalization_threshold_not_moved"],
            "stated_before_fitting": True,
            "evaluated_on_holdout": holdout is not None,
            "holdout_result": registered.get("cold_lift"),
            "holds": bool(guards.get("guard_a_generalization", {}).get("holds")),
        },
        "holdout_evaluations": splits.sealed_reads,
        "holdout_first_read_after_parameters_sealed": bool(holdout),
        "holdout_cold_admitted": registered.get("cold_admitted"),
        "holdout_cold_admitted_exceeds_zero": bool(registered.get("cold_admitted", 0) > 0),
        "holdout_cold_lift": registered.get("cold_lift"),
        "holdout_cold_auc": registered.get("cold_auc"),
        "cold_candidate_share_by_split": {
            name: {
                "cold_candidates": report.get("cold_candidates"),
                "cold_candidate_share": report.get("cold_candidate_share"),
                "capped_cold_candidates": (report.get("estimation") or {}).get(
                    "capped_cold_candidates"
                ),
                "capped_cold_candidate_share": (report.get("estimation") or {}).get(
                    "capped_cold_candidate_share"
                ),
            }
            for name, report in split_reports.items()
        },
        "refit_on_holdout": False,
        "no_eval_refit": True,
        "no_holdout_refit": True,
        "sealed_partition_reads_observed": splits.sealed_reads,
        "field_subtree_accessed": False,
        "post_deployment_holdout_accessed": False,
        "guards": guards,
        "decision_procedure": outcome["steps"],
        "halted_at": outcome.get("halted_at"),
        "verdict": outcome["verdict"],
        "failed_steps": outcome["failed_steps"],
        "findings": build_findings(outcome, manifest_block),
        "fitted_candidate": (
            {
                "note": (
                    "the constants the wire sibling promotes IF AND ONLY IF wireable is true. "
                    "The published step functions ARE the definition -- eval, holdout and the "
                    "live scorer all evaluate these same tables -- so a threshold tie "
                    "reproduces bit for bit."
                ),
                "these_parameters_were_sealed_to_disk_before_any_gate_after_step_4": True,
                "the_holdout_was_opened": bool(holdout),
                "not_wireable_unless_the_verdict_is_pass": (
                    "these are published because step 4 really did seal them and the record "
                    "owes the reader the object the later guards were run against. They carry "
                    "no generalization claim: wireable and verdict are the fields that decide."
                ),
                "signed_weights": [
                    {"name": name, "source_field": source, "signed_weight": sign}
                    for source, name, sign in CORE_FAMILY
                ],
                "family_id": prereg["feature_family"]["id"],
                "calibration": {
                    "rank_uniform_centered": "(A + M/2)/n - 0.5, midrank on ties",
                    "rank_normal": (
                        "clip(Phi_inverse(clip(u + 0.5, 1/2n, 1 - 1/2n)), -3, +3)"
                    ),
                    "atom_minimum_share": ATOM_MINIMUM_SHARE,
                    "grid_knots": GRID_KNOTS,
                    "max_breakpoints": MAX_BREAKPOINTS,
                    "round_places": ROUND_PLACES,
                    "u_round_places": U_ROUND_PLACES,
                    "fitted_on": "coldstart_train only, cold and warm rows pooled",
                },
                **pending["payload"],
                "parameter_sha256": pending["digest"],
            }
            if pending.get("payload")
            else None
        ),
        "reproduction": reproduction,
        "provenance": {
            "script": "scripts/recall_map_coldstart_fit.py",
            "extractor": "scripts/recall_map_coldstart_dataset.py",
            "extractor_sha256": manifest_block["extractor_sha256"],
            "read_only": True,
            "population_rebuilt_from_persisted_state_at": manifest_block["as_of"],
        },
        "privacy": {
            "item_level_rows": False,
            "aggregate_only": True,
            "corpus_text": False,
            "identity_values_published": False,
        },
    }


def _check_manifest_privacy(value: Any, path: str = "$") -> None:
    """The aggregate-only guard the published manifest is verified under.

    ``living_memory.postsession.usage_metric.check_privacy`` rejects any
    published string that is not short printable single-line ASCII, which is
    why prose in this artifact is a LIST of short lines rather than one long
    string.  Imported lazily so this script keeps running with the package
    absent, and re-implemented nowhere: a second copy of the rule is a copy
    that can drift from the guard that actually verifies the block.
    """

    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from living_memory.postsession.usage_metric import check_privacy
    except ImportError:  # pragma: no cover - the guard is unavailable, not passing
        raise ColdStartFitError(
            "cannot verify the manifest privacy guard: "
            "living_memory.postsession.usage_metric is not importable"
        ) from None
    check_privacy(value, path)


def republish_manifest_estimation(
    manifest_path: Path,
    analysis_path: Path,
    cap: int,
    split_reports: Mapping[str, Any],
) -> dict[str, Any]:
    """Make the manifest describe the population estimation actually used.

    The plan's estimation rule caps component contributions, so the split the
    manifest publishes is no longer the row set the verdict was computed on.
    ``dataset-manifest.json#coldstart.splits.*.estimation`` closes that gap:
    each split now carries the capped item and cold counts, the retained share,
    the capped effective components whole and cold, the capped largest-component
    share and the capped era composition, beside the uncapped figures it always
    carried.

    Three invariants are re-checked here and recorded, not assumed: every split
    keeps at least ``minimum_cold_candidates_per_split`` cold candidates under
    the cap, every split keeps a non-zero cold share, and the three component
    lists stay pairwise disjoint -- which a cap cannot break, because it keeps
    ``min(cap, n_c)`` rows of every component inside the split that component
    was already assigned to.
    """

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    block = manifest["coldstart"]
    lists = {name: set(block["splits"][name]["components"]) for name in block["splits"]}
    intersections = {
        f"{a}_{b}": len(lists[a] & lists[b])
        for a, b in (
            (FIT_SPLIT, VERIFY_SPLIT),
            (FIT_SPLIT, SEALED_SPLIT),
            (VERIFY_SPLIT, SEALED_SPLIT),
        )
    }
    problems: list[str] = []
    summary: dict[str, Any] = {}
    for name, report in split_reports.items():
        estimation = report.get("estimation")
        if not isinstance(estimation, Mapping):
            problems.append(f"{name}.no_estimation_block")
            continue
        published = dict(estimation)
        published["describes"] = [
            "The component-capped estimation row set every verdict-carrying",
            "statistic in policy.json#coldstart_revision was computed on.",
        ]
        published["rule"] = [
            "artifacts/recall-map/relevance/dataset-manifest.json",
            "#coldstart.enlargement.cap_feasibility.rule, adopted verbatim.",
        ]
        block["splits"][name]["estimation"] = published
        cold = published["capped_cold_candidates"]
        share = published["capped_cold_candidate_share"]
        if cold < int(block["enlargement"]["cap_feasibility"]["minimum_cold_candidates_per_split"]):
            problems.append(f"{name}.capped_cold_below_minimum")
        if not share or float(share) <= 0.0:
            problems.append(f"{name}.capped_cold_share_is_zero")
        summary[name] = {
            "capped_items": published["capped_items"],
            "capped_cold_candidates": cold,
            "capped_cold_candidate_share": share,
            "effective_components_capped": published["effective_components_capped"],
            "cold_effective_components_capped": published.get(
                "cold_effective_components_capped"
            ),
        }
    if any(intersections.values()):
        problems.append("component_lists_are_not_pairwise_disjoint")

    block["estimation_population"] = {
        "estimation_cap": cap,
        "namespace": CAP_SUBSAMPLE_NAMESPACE,
        # Prose in this artifact is a LIST of short lines, not one long string:
        # ``living_memory.postsession.usage_metric.check_privacy`` rejects any
        # published string over 200 printable ASCII characters, and that guard
        # runs over this whole block.
        "why_this_block_exists": [
            "coldstart-prereg-v3.json#component_aware_estimation caps component",
            "contributions, so the population the manifest describes must be the",
            "population estimation was actually done on. Every split's estimation",
            "block below is that population.",
        ],
        "per_split": summary,
        "minimum_cold_candidates_per_split": int(
            block["enlargement"]["cap_feasibility"]["minimum_cold_candidates_per_split"]
        ),
        "every_split_clears_the_cold_minimum_under_the_cap": not any(
            problem.endswith("capped_cold_below_minimum") for problem in problems
        ),
        "every_split_keeps_a_non_zero_cold_share_under_the_cap": not any(
            problem.endswith("capped_cold_share_is_zero") for problem in problems
        ),
        "pairwise_component_intersections": intersections,
        "pairwise_component_disjoint": not any(intersections.values()),
        "the_cap_cannot_move_a_component_between_splits": [
            "The cap is applied INSIDE the split a component was already assigned",
            "to and never consults the split, so no cache key and no session id",
            "reaches a second split. enlargement.concentration_cost",
            ".an_intra_split_cap_does_not_weaken_disjointness says the same.",
        ],
        "problems": problems,
    }
    if problems:
        raise ColdStartFitError(
            "refusing to republish the manifest: the capped population violates a registered "
            "minimum: " + ", ".join(problems)
        )

    # ``dataset-manifest.json#privacy`` is enforced, not promised: every value
    # this fit adds must pass the same aggregate-only guard the extractor's
    # own block is verified under.  Checked before the write, so a violation
    # cannot reach disk.
    _check_manifest_privacy(block["estimation_population"])
    for name in split_reports:
        _check_manifest_privacy(block["splits"][name]["estimation"])

    # The COMPACT canonical rendering the extractor writes this artifact in,
    # so republishing one block does not rewrite every byte of it.
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    # ``feature-analysis.json#dataset_manifest_sha256`` is a POINTER at the
    # manifest, so republishing the manifest necessarily staled it.  Refreshed
    # through the extractor's own ``rebind_analysis`` rather than by a second
    # copy of the rule: not one measured value inside the analysis is
    # recomputed, and a knowingly stale digest is worse than no digest.
    module = extractor()
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    before = analysis.get("dataset_manifest_sha256")
    module.rebind_analysis(manifest, analysis)
    after = analysis["dataset_manifest_sha256"]
    # Written back in the COMPACT canonical rendering the file already uses.
    # Re-indenting it would rewrite every byte of an artifact this run has no
    # business touching beyond one pointer, and would move its stored-bytes
    # digest for a formatting reason rather than a factual one.
    analysis_path.write_text(canonical_json(analysis) + "\n", encoding="utf-8")
    published = dict(block["estimation_population"])
    published["feature_analysis_pointer_rebound"] = {
        "field": "artifacts/recall-map/relevance/feature-analysis.json#dataset_manifest_sha256",
        "before": before,
        "after": after,
        "what_this_is": (
            "a pointer rebinding and nothing else. Republishing the manifest changes its "
            "canonical digest, so every artifact that cites that digest must be repointed "
            "or it is citing a manifest that no longer exists. No measured value inside "
            "feature-analysis.json was recomputed."
        ),
        "rebound_by": "scripts/recall_map_coldstart_dataset.py:rebind_analysis",
    }
    return published


def rebind(policy: dict[str, Any], manifest_path: Path, analysis_path: Path) -> dict[str, Any]:
    """Rebind the pointers the republished manifest invalidated."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bindings = policy["bindings"]
    keys = (
        ("dataset_manifest", "canonical_json_sha256"),
        ("dataset_manifest", "stored_bytes_sha256"),
        ("feature_analysis", "stored_bytes_sha256"),
    )
    before = {f"{a}_{b}": bindings[a][b] for a, b in keys}
    bindings["dataset_manifest"]["canonical_json_sha256"] = sha256_text(canonical_json(manifest))
    bindings["dataset_manifest"]["stored_bytes_sha256"] = sha256_bytes(manifest_path)
    bindings["feature_analysis"]["stored_bytes_sha256"] = sha256_bytes(analysis_path)
    after = {f"{a}_{b}": bindings[a][b] for a, b in keys}
    return {"before": before, "after": after}


def write_policy(path: Path, policy: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def seal_bindings(policy: dict[str, Any], revision: Mapping[str, Any]) -> None:
    subobjects = policy["bindings"]["canonical_subobjects"]
    subobjects["coldstart_revision_sha256"] = sha256_text(canonical_json(dict(revision)))
    subobjects["selected_policy_sha256_unchanged"] = (
        sha256_text(canonical_json(policy["selected_policy"]))
        == subobjects["selected_policy_sha256"]
    )


def manifest_split_citation(
    name: str,
    published: Mapping[str, Any],
    cap_row: Mapping[str, Any] | None = None,
    cap: int = ESTIMATION_CAP,
) -> dict[str, Any]:
    """A split cited from the manifest alone, for a run that HALTED early.

    Counting published rows is a split audit, which step 2 requires; it is not
    an evaluation and it opens nothing.
    """

    return {
        "ref": (
            "artifacts/recall-map/relevance/dataset-manifest.json" f"#coldstart.splits.{name}"
        ),
        "items": published["items"],
        "consumed": published["consumed"],
        "rate": published["rate"],
        "cold_candidates": published["cold_candidates"],
        "cold_candidate_share": published["cold_candidate_share"],
        "cold_consumed": published["cold_consumed"],
        "cold_base_rate": published["cold_rate"],
        "family_unscoreable": published["family_unscoreable"],
        "membership_digest": published["membership_digest"],
        "concentration": published["concentration"],
        "estimation": (
            None
            if cap_row is None
            else {
                "estimation_cap": cap,
                "namespace": CAP_SUBSAMPLE_NAMESPACE,
                "capped_items": cap_row["items"],
                "capped_cold_candidates": cap_row["cold_candidates"],
                "capped_cold_candidate_share": cap_row["cold_candidate_share"],
                "items_retained_share": cap_row["items_retained_share"],
                "components_under_the_cap": cap_row["components"],
                "effective_components_capped": cap_row["effective_components"],
                "clears_the_whole_split_effective_component_floor": (
                    float(cap_row["effective_components"])
                    >= EFFECTIVE_COMPONENT_FLOOR_WHOLE_SPLIT
                ),
                "cited_from_the_published_cap_table_no_row_was_read": True,
            }
        ),
        "cited_from_the_manifest_only": (
            "the run HALTED before this partition was materialized, so these are the "
            "published audit counts and no quantity here was computed from its rows"
        ),
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def command_run(args: argparse.Namespace) -> int:
    prereg = json.loads(Path(args.prereg).read_text(encoding="utf-8"))
    plan_sha256 = verify_prereg_seal(prereg)

    if "=" not in args.source:
        raise ColdStartFitError("--source must be LOGICAL_NAME=SQLITE_PATH")
    _name, raw_path = args.source.split("=", 1)

    policy_path = Path(args.policy)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    block = manifest["coldstart"]

    population = load_population(Path(raw_path).expanduser())
    splits_match = confirm_splits_match_manifest(population, manifest)
    splits = SealedSplits(_rows=population["rows"])

    def seal_to_disk(payload: Mapping[str, Any], digest: str) -> dict[str, Any]:
        """``seal.fail_closed``: the digest is ON DISK before the holdout opens.

        Written into ``policy.json`` itself rather than a sidecar, so the
        artifact that will carry the verdict is the artifact that carried the
        commitment.  If this run dies between here and the verdict, what is left
        on disk is a parameter block with no holdout result -- which is the
        honest record of what happened, and fails every wireability check.
        """

        provisional = {
            "id": REVISION_ID,
            "status": "parameters_sealed_holdout_not_yet_opened",
            "wireable": False,
            "prereg": {
                "path": "artifacts/recall-map/relevance/coldstart-prereg-v3.json",
                "plan_sha256": plan_sha256,
            },
            "parameter_sha256": digest,
            "sealed_parameters": dict(payload),
            "holdout_evaluations": 0,
            "field_subtree_accessed": False,
            "post_deployment_holdout_accessed": False,
        }
        staged = json.loads(json.dumps(policy))
        staged["coldstart_revision"] = provisional
        write_policy(policy_path, staged)
        return {
            "written_to": str(policy_path.relative_to(REPO_ROOT)),
            "key": "coldstart_revision.parameter_sha256",
            "parameter_sha256": digest,
            "stored_bytes_sha256": sha256_bytes(policy_path),
            "before_the_holdout_was_opened": True,
        }

    outcome = run_decision_procedure(
        splits, population, prereg, policy, block, plan_sha256, seal_to_disk
    )
    pending = outcome.get("pending") or {}
    published_digest = pending.get("digest", "")
    reproduction = (
        compute_reproduction(population, policy, published_digest, splits_match)
        if published_digest
        else {
            "splits_match_manifest": dict(splits_match),
            "byte_identical": False,
            "not_applicable": "the run HALTED before any parameter was fitted",
        }
    )
    if pending.get("holdout"):
        outcome = finalize_verdict(outcome, reproduction)

    cap = int(pending.get("estimation_cap") or ESTIMATION_CAP)
    cap_table = block["enlargement"]["cap_feasibility"]["table"][str(cap)]
    split_reports = {
        FIT_SPLIT: split_reporting(
            FIT_SPLIT,
            population["rows"][FIT_SPLIT],
            block["splits"][FIT_SPLIT],
            pending.get("train_arms"),
            cap,
        ),
        VERIFY_SPLIT: split_reporting(
            VERIFY_SPLIT,
            population["rows"][VERIFY_SPLIT],
            block["splits"][VERIFY_SPLIT],
            pending.get("eval_arms"),
            cap,
        ),
        SEALED_SPLIT: (
            pending["holdout"]["split"]
            if pending.get("holdout")
            else manifest_split_citation(
                SEALED_SPLIT, block["splits"][SEALED_SPLIT], cap_table[SEALED_SPLIT], cap
            )
        ),
    }
    residual_risk = uniform_draw_residual_risk(population, block)
    republished = republish_manifest_estimation(
        Path(args.manifest), Path(args.analysis), cap, split_reports
    )
    revision = build_revision_block(
        outcome,
        population,
        reproduction,
        prereg,
        block,
        plan_sha256,
        residual_risk,
        splits,
        split_reports,
        policy["selected_policy"]["id"],
    )
    revision["dataset"]["estimation_population_republished_to_the_manifest"] = republished

    policy["coldstart_revision"] = revision
    policy["bindings"]["rebound_by_coldstart_fit"] = rebind(
        policy, Path(args.manifest), Path(args.analysis)
    )
    seal_bindings(policy, revision)
    write_policy(policy_path, policy)

    print(
        f"verdict={revision['verdict']} wireable={revision['wireable']} "
        f"holdout_evaluations={revision['holdout_evaluations']} "
        f"holdout_cold_lift={revision['generalization_threshold']['holdout_result']} "
        f"holdout_cold_admitted={revision['holdout_cold_admitted']} "
        f"estimation_cap={cap} "
        f"failed_steps={','.join(revision['failed_steps']) or 'none'} "
        f"parameter_sha256={published_digest[:16] or 'none'}"
    )
    return 0


def command_verify(args: argparse.Namespace) -> int:
    """Re-derive the published parameters and re-check the seal claims."""

    prereg = json.loads(Path(args.prereg).read_text(encoding="utf-8"))
    verify_prereg_seal(prereg)
    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    revision = policy.get("coldstart_revision")
    if not isinstance(revision, Mapping):
        raise ColdStartFitError("policy carries no coldstart_revision block")

    problems: list[str] = []
    # Stated as the INVARIANT rather than as a literal policy id: the wire
    # sibling promotes this revision into ``selected_policy`` and moves
    # ``recall_map.RELEVANCE_POLICY_ID`` in the same commit, so a hard-coded id
    # here would go red the moment that lands while saying nothing about what
    # actually matters -- that the revision is not already the in-force policy.
    if revision["id"] == policy["selected_policy"]["id"]:
        problems.append("revision id was not bumped off the in-force policy")
    if revision["holdout_evaluations"] > 1:
        problems.append("the sealed partition was evaluated more than once")
    if revision["refit_on_holdout"] is not False:
        problems.append("refit_on_holdout is not false")
    if revision["sealed_partition_reads_observed"] != revision["holdout_evaluations"]:
        problems.append("observed sealed reads disagree with the published count")
    digest = sha256_text(canonical_json(dict(revision)))
    subobjects = policy["bindings"]["canonical_subobjects"]
    if digest != subobjects.get("coldstart_revision_sha256"):
        problems.append(f"coldstart_revision digest mismatch: {digest}")
    if sha256_text(canonical_json(policy["selected_policy"])) != subobjects[
        "selected_policy_sha256"
    ]:
        problems.append("selected_policy digest does not match its binding")
    if subobjects.get("selected_policy_sha256_unchanged") is not True:
        problems.append("selected_policy_sha256_unchanged is not true")
    if revision["wireable"] and revision["verdict"] != "pass":
        problems.append("a non-passing revision is offered to the wire")

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if policy["bindings"]["dataset_manifest"]["canonical_json_sha256"] != sha256_text(
        canonical_json(manifest)
    ):
        problems.append("dataset_manifest binding is stale")

    candidate = revision.get("fitted_candidate")
    reproduced = False
    if args.source and candidate is None:
        print(
            "SKIP: the run HALTED before any parameter was fitted, so there is nothing to "
            "re-derive. The reproduction check is not applicable, not passed.",
            file=sys.stderr,
        )
    elif args.source:
        if "=" not in args.source:
            raise ColdStartFitError("--source must be LOGICAL_NAME=SQLITE_PATH")
        _name, raw_path = args.source.split("=", 1)
        population = load_population(Path(raw_path).expanduser())
        splits_match = confirm_splits_match_manifest(population, manifest)
        published = candidate["parameter_sha256"]
        report = compute_reproduction(population, policy, published, splits_match)
        reproduced = True
        if not report["train_only_reproduces_published"]:
            problems.append("train-only refit does not reproduce the published parameters")
        if not report["withheld_holdout_reproduces_published"]:
            problems.append("withheld-holdout refit does not reproduce the published parameters")
        print(
            "train-only refit: "
            + ("REPRODUCES" if report["train_only_reproduces_published"] else "DIFFERS")
            + " | withheld-holdout refit: "
            + ("REPRODUCES" if report["withheld_holdout_reproduces_published"] else "DIFFERS")
        )

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    print(
        f"OK: plan seal verified, verdict={revision['verdict']}, the sealed partition was "
        f"materialized {revision['holdout_evaluations']} time(s), selected_policy is "
        "byte-identical and its binding holds, and the revision digest matches"
        + (
            ", and the published parameters re-derive from the train split alone"
            if reproduced
            else ". No parameter reproduction was checked: "
            + (
                "this run fitted none"
                if candidate is None
                else "pass --source to re-derive them"
            )
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="execute the sealed nine-step decision procedure")
    run.add_argument("--source", required=True, metavar="NAME=PATH")
    run.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    run.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)

    verify = sub.add_parser("verify", help="re-derive the published parameters from train alone")
    verify.add_argument("--source", metavar="NAME=PATH", default=None)
    verify.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    verify.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return command_run(args)
    if args.command == "verify":
        return command_verify(args)
    raise AssertionError(args.command)  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
