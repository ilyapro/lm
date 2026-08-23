#!/usr/bin/env python3
"""Render the expanded historical recall-map relevance report.

This renderer deliberately lives outside the hash-bound historical evaluator.
It consumes only the aggregate dataset manifest and feature analysis; in
particular, it never opens a source snapshot, the tracked Markdown, or the
candidate-holdout directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts/recall-map/relevance/dataset-manifest.json"
DEFAULT_ANALYSIS = ROOT / "artifacts/recall-map/relevance/feature-analysis.json"

COHORTS = (
    ("organic_train", "Organic train", "sole fitting population"),
    ("organic_eval", "Organic evaluation", "disjoint evaluation"),
    ("observed_map_eval", "Observed-map evaluation", "transfer evaluation only"),
    ("secondary_anchor", "Secondary anchor", "sensitivity diagnostic only"),
)
PRIMARY_EVALUATION_COHORTS = COHORTS[:3]
SCORE_FEATURES = ("score", "bm25_score", "vector_score", "graph_score", "trigger_score")
BALLAST_CLASSES = (
    "file_chunk_envelope",
    "strategy_stagnation",
    "supervision_journal",
)

TRANSFER_INTERPRETATIONS = {
    "node_age_log_days": "freshness/age direction does not transfer",
    "level_trace": "level effect reverses on map evaluation",
    "level_concept": "direction already reverses within organic data",
    "level_schema": "same-sign association only; not by itself deployment evidence",
    "prior_matured_log_count": "same-sign association only",
    "prior_nonconsumed_log_count": "does not support the hypothesized negative penalty",
    "prior_nonconsumption_streak_log": "univariate direction does not support a negative penalty",
    "prior_consumption_rate": "same-sign association only",
}

OMISSION_REASONS = {
    "form_file_chunk_envelope": "snapshot-only form classifier output; not recorded at delivery; ballast sensitivity diagnostic only",
    "form_strategy_stagnation": "snapshot-only form classifier output; not recorded at delivery; ballast sensitivity diagnostic only",
    "form_supervision_journal": "snapshot-only form classifier output; not recorded at delivery; ballast sensitivity diagnostic only",
    "form_machine_ballast": "snapshot-only union of form classifiers; not recorded at delivery; ballast sensitivity diagnostic only",
    "content_log_chars": "current-snapshot content size; historical value not versioned",
    "content_log_lines": "current-snapshot content size; historical value not versioned",
    "content_json_envelope": "current-snapshot content form; historical value not versioned",
    "content_code_fence": "current-snapshot content form; historical value not versioned",
    "provenance_log_key_count": "current-snapshot provenance shape; historical provenance not versioned",
    "provenance_log_source_trace_count": "current-snapshot provenance shape; historical provenance not versioned",
    "context_log_key_count": "current-snapshot context shape; historical context not versioned",
    "context_log_nested_count": "current-snapshot context shape; historical context not versioned",
    "context_log_sequence_count": "current-snapshot context shape; historical context not versioned",
    "context_log_scalar_count": "current-snapshot context shape; historical context not versioned",
    "cascade_anchor": "historical cascade stage not recorded at delivery and not reconstructed; unavailable in every arm",
    "score": "available for organic results but absent from every observed-map payload; no transfer cell",
    "bm25_score": "available for organic results but absent from every observed-map payload; no transfer cell",
    "vector_score": "available for organic results but absent from every observed-map payload; no transfer cell",
    "graph_score": "available for organic results but absent from every observed-map payload; no transfer cell",
    "trigger_score": "available for organic results but absent from every observed-map payload; no transfer cell",
}

REJECTION_ROWS = {
    "node ids": ("Node identifier", "identity leakage and forbidden lookup behavior"),
    "labels": ("Outcome label", "target leakage"),
    "task names": ("Task metadata", "semantic identity leakage"),
    "hosts": ("Host", "environment identity leakage"),
    "source identity": ("Source identity", "source lookup leakage"),
    "cache keys": ("Cache identity", "grouping identity; used only to enforce split disjointness"),
    "sessions": ("Session identity", "grouping identity; used only to enforce split disjointness"),
    "current access_count": (
        "Current `access_count`",
        "future-mutated after delivery and not reconstructible as of delivery",
    ),
    "current usefulness_score": (
        "Current `usefulness_score`",
        "future-mutated after delivery and not reconstructible as of delivery",
    ),
    "current last_accessed": (
        "Current `last_accessed`",
        "future-mutated after delivery and not reconstructible as of delivery",
    ),
    "updated_at-derived usage": (
        "`updated_at`-derived usage",
        "future-mutated proxy and not reconstructible as of delivery",
    ),
}


class ReportError(ValueError):
    """Raised when aggregate inputs cannot safely drive the report."""


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stored_json_sha256(payload: Mapping[str, Any]) -> str:
    return _sha256_text(canonical_json(payload) + "\n")


def _assert_not_holdout(path: Path) -> None:
    parts = tuple(part.casefold() for part in path.resolve().parts)
    prohibited = ("artifacts", "recall-map", "relevance", "field")
    if any(parts[index : index + len(prohibited)] == prohibited for index in range(len(parts))):
        raise ReportError(f"candidate-holdout access is prohibited: {path}")


def _read_object(path: Path) -> dict[str, Any]:
    _assert_not_holdout(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReportError(f"cannot read aggregate JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"aggregate JSON must be an object: {path}")
    return value


def _validate_inputs(manifest: Mapping[str, Any], analysis: Mapping[str, Any]) -> None:
    if manifest.get("artifact") != "recall-map-relevance-dataset-manifest":
        raise ReportError("unexpected dataset-manifest artifact type")
    if analysis.get("artifact") != "recall-map-relevance-feature-analysis":
        raise ReportError("unexpected feature-analysis artifact type")
    if manifest.get("schema_version") != 1 or analysis.get("schema_version") != 1:
        raise ReportError("unsupported relevance-evidence schema version")
    if manifest.get("as_of") != analysis.get("as_of"):
        raise ReportError("manifest and analysis as_of values differ")
    expected_manifest_sha256 = _sha256_text(canonical_json(manifest))
    if analysis.get("dataset_manifest_sha256") != expected_manifest_sha256:
        raise ReportError("analysis does not bind the supplied canonical manifest")
    leakage = analysis.get("leakage_audit")
    if not isinstance(leakage, Mapping) or leakage.get("candidate_holdout_accessed") is not False:
        raise ReportError("candidate-holdout non-access assertion is missing")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != 1 or not isinstance(sources[0], Mapping):
        raise ReportError("expanded report requires exactly one aggregate source receipt")


def _manifest_cohort(manifest: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key == "organic_train":
        value = manifest["train"]
    elif key == "organic_eval":
        value = manifest["eval"]["organic"]
    elif key == "observed_map_eval":
        value = manifest["eval"]["observed_map"]
    elif key == "secondary_anchor":
        value = manifest["secondary_anchor"]
    else:  # pragma: no cover - all callers use the fixed schema
        raise KeyError(key)
    if not isinstance(value, Mapping):
        raise ReportError(f"malformed cohort: {key}")
    return value


def _integer(value: Any) -> str:
    return f"{int(value):,}"


def _word_number(value: Any) -> str:
    number = int(value)
    words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
    return words[number] if 0 <= number < len(words) else _integer(number)


def _fixed(value: Any, places: int) -> str:
    return f"{float(value):.{places}f}"


def _signed(value: Any, places: int) -> str:
    number = float(value)
    sign = "+" if number >= 0 else "−"
    return f"{sign}{abs(number):.{places}f}"


def _association_cell(summary: Mapping[str, Any]) -> str:
    consumed = summary.get("mean_consumed")
    not_consumed = summary.get("mean_not_consumed")
    difference = summary.get("difference")
    if consumed is None or not_consumed is None or difference is None:
        return f"unavailable, {_integer(summary.get('available', 0))}/{_integer(summary.get('missing', 0))}"
    if float(consumed) == float(not_consumed) == float(difference) == 0.0:
        return "0 / 0; 0"
    return f"{_fixed(consumed, 8)} / {_fixed(not_consumed, 8)}; {_signed(difference, 8)}"


def _yes_no(value: Any, *, emphasize_failure: bool = False) -> str:
    answer = "Yes" if bool(value) else "No"
    return f"**{answer}**" if emphasize_failure and not bool(value) else answer


def _component_count(cohort: Mapping[str, Any]) -> int:
    components = cohort.get("components", [])
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
        raise ReportError("cohort components must be a sequence")
    return len(components)


def _content_form(
    analysis: Mapping[str, Any], cohort: str, form: str
) -> Mapping[str, Any]:
    value = analysis["content_forms"][cohort].get(form, {})
    if not isinstance(value, Mapping):
        raise ReportError(f"malformed content-form summary: {cohort}/{form}")
    return value


def _append_header(lines: list[str], title: str) -> None:
    lines.extend((f"## {title}", ""))


def render_report(manifest: Mapping[str, Any], analysis: Mapping[str, Any]) -> str:
    """Return the deterministic expanded report for two aggregate objects."""

    _validate_inputs(manifest, analysis)
    model = analysis["primary_model"]
    transfer = analysis["transfer_verdict"]
    model_evaluations = model["evaluations"]
    primary_features = analysis["feature_policy"]["primary_model_features"]
    map_evaluation = model_evaluations["observed_map_eval"]
    map_cohort = _manifest_cohort(manifest, "observed_map_eval")
    minimums = transfer["cohort_minimums"]
    floor_pass = bool(transfer["floor_pass"])
    minimums_pass = bool(transfer["minimums_pass"])
    minimum_phrase = "did pass every" if minimums_pass else "did not pass every"
    floor_phrase = "passed" if floor_pass else "did not pass"

    lines = [
        "# Historical recall-map relevance evidence",
        "",
        f"This is a deterministic, aggregate-only rendering of the frozen dataset manifest and feature analysis pinned at `{manifest['as_of']}`. The candidate holdout was neither inspected nor created.",
        "",
        f"**Result:** the organic-train-fitted model selected {_integer(map_evaluation['items'])} of {_integer(map_evaluation['total_items'])} observed-map evaluation items and consumed {_integer(map_evaluation['consumed'])} of those {_integer(map_evaluation['items'])}, a rate of `{_fixed(map_evaluation['rate'], 6)}`. The unchanged transfer floor of `{_fixed(transfer['threshold'], 3)}` therefore **{floor_phrase}**. The selected transfer cohort {minimum_phrase} unchanged minimum: {_integer(map_evaluation['events'])} events versus {_integer(minimums['events'])} required, {_integer(map_evaluation['items'])} items versus {_integer(minimums['items'])} required, and {_integer(map_evaluation['transport_sessions'])} transport sessions versus {_integer(minimums['transport_sessions'])} required. The overall transfer verdict is **{transfer['verdict']}**, with no evaluation refit.",
        "",
        "This result is evidence about historical transfer, not a candidate deployment verdict. "
        + (
            "The frozen model passed its historical transfer floor."
            if floor_pass
            else "No feature is established for deployment by a model that failed the frozen transfer floor."
        ),
        "",
    ]

    _append_header(lines, "Outcome, cohort construction, and exact denominators")
    protocol = manifest["protocol"]
    organic_window = protocol["organic_window"]
    lines.extend(
        (
            f"The endpoint is non-anchor-stratum node-id consumption within {_integer(protocol['horizon_hours'])} hours. Its unit is one `(delivering event, delivered item)` pair, and its rate is consumed items divided by items. The evaluator verified the sealed preregistration and effect-tool hashes before reusing the frozen protocol, consumer index, item construction, anchor stratification, and node-id scoring behavior. Organic items use the frozen window `[{organic_window[0]}, {organic_window[1]})`, an organic head cut of {_integer(protocol['organic_head_cut'])}, and caps of {_integer(protocol['organic_cap'])} items for each arm.",
            "",
            f"The split is event-component based, not item based. Events are joined when they share a source-qualified cache identity, transport identity, or session identity. This produced {_integer(manifest['split']['components'])} connected components. Every component touching any observed map delivery—including a delivery with no scorable primary item—was forced to evaluation; {_integer(len(manifest['split']['forced_eval_components']))} components were forced this way. Each remaining organic component was assigned deterministically with the sealed seed and an {_fixed(float(manifest['split']['organic_train_fraction']) * 100, 0)}% train fraction. The final assignment contains {_integer(_component_count(_manifest_cohort(manifest, 'organic_train')))} train components and {_integer(len(manifest['eval']['components']))} evaluation components, with no component shared between train and evaluation. No identity value is published or used as a model feature.",
            "",
            f"The primary fitting cohort contains only pre-feature, opportunity-bearing organic items in the primary stratum. The organic evaluation cohort contains the remaining disjoint primary-stratum organic items. Every already-observed pre-candidate map item is transfer evaluation only; the model fit contains exactly {_word_number(analysis['leakage_audit']['map_rows_in_fit'])} map rows. Component counts within the two evaluation arms need not add to the {_integer(len(manifest['eval']['components']))} evaluation components because arms may occur in the same connected component and some bridge events have no scorable primary item.",
            "",
            "| Cohort | Role | Components represented | Events | Items | Consumed | Not consumed | Rate | Transport sessions |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for key, label, role in COHORTS:
        cohort = _manifest_cohort(manifest, key)
        lines.append(
            f"| {label} | {role} | {_integer(_component_count(cohort))} | {_integer(cohort['events'])} | {_integer(cohort['items'])} | {_integer(cohort['consumed'])} | {_integer(int(cohort['items']) - int(cohort['consumed']))} | {_fixed(cohort['rate'], 6)} | {_integer(cohort['transport_sessions'])} |"
        )
    lines.extend(
        (
            "",
            "The secondary-anchor rows are outside the primary endpoint. Their events can overlap primary-stratum events, so they must not be added to the primary cohort denominators.",
            "",
        )
    )

    _append_header(lines, "Frozen sources and hashes")
    source = manifest["sources"][0]
    database_parts = [part for part in source["parts"] if part.get("kind") == "database"]
    if len(database_parts) != 1:
        raise ReportError("source receipt requires exactly one database part")
    database = database_parts[0]
    sqlite = source["sqlite"]
    sealed = manifest["sealed_inputs"]
    lines.extend(
        (
            f"The single logical source is the redacted snapshot `{source['redacted_path']}`. The database part is {_integer(database['bytes'])} bytes and has SHA-256 `{database['sha256']}`. Its canonical parts-receipt digest is `{source['snapshot_sha256']}`; its schema digest is `{sqlite['schema_sha256']}`. The pinned pre-candidate slice contains {_integer(sqlite['pre_candidate_event_rows'])} event rows, from `{sqlite['pre_candidate_event_min']}` through `{sqlite['pre_candidate_event_max']}`, across {_integer(sqlite['schema_objects'])} schema objects.",
            "",
            "| Bound input or artifact | SHA-256 | Meaning |",
            "|---|---|---|",
            f"| Historical evaluator | `{manifest['evaluator_sha256']}` | code bytes used to produce and verify the evidence |",
            f"| Frozen effect tool | `{sealed['effect_tool']['sha256']}` | reused {_integer(protocol['horizon_hours'])}-hour consumption implementation |",
            f"| Frozen preregistration | `{sealed['prereg']['sha256']}` | sealed protocol input |",
            f"| Frozen baseline | `{sealed['baseline']['sha256']}` | supplies the frozen organic baseline rate |",
            f"| Frozen plan binding | `{sealed['plan_sha256']}` | protocol plan digest |",
            f"| Dataset manifest, canonical JSON | `{analysis['dataset_manifest_sha256']}` | digest bound by the feature analysis |",
            f"| Dataset manifest, stored bytes | `{_stored_json_sha256(manifest)}` | includes the terminal newline |",
            f"| Feature analysis, stored bytes | `{_stored_json_sha256(analysis)}` | machine-readable source for this report |",
            f"| Primary feature schema | `{model['feature_schema_sha256']}` | ordered {_word_number(len(primary_features))}-feature model surface |",
            "",
        )
    )

    _append_header(lines, "Decision-time boundary and leakage matrix")
    availability = analysis["availability"]
    train_avail = availability["organic_train"]
    organic_avail = availability["organic_eval"]
    map_avail = availability["observed_map_eval"]
    anchor_avail = availability["secondary_anchor"]

    def reason_count(summary: Mapping[str, Any], reason: str) -> int:
        return int(summary.get("reasons", {}).get(reason, 0))

    lines.extend(
        (
            f"Decision-time evidence is restricted to immutable node creation fields, the recorded delivery envelope, and earlier delivery outcomes whose complete {_integer(protocol['horizon_hours'])}-hour windows had already matured. Content, provenance, and context shape are current-snapshot diagnostics only. They were not fitted, because their historical values were not versioned. Unrecorded historical values were left unavailable rather than reconstructed from future state.",
            "",
            "| Evidence surface | Organic train | Organic evaluation | Observed-map evaluation | Secondary anchor | Leakage disposition |",
            "|---|---:|---:|---:|---:|---|",
            f"| Immutable node age | {_integer(train_avail['node_age']['items'])}/{_integer(train_avail['node_age']['items'])} available | {_integer(organic_avail['node_age']['items'])}/{_integer(organic_avail['node_age']['items'])} | {_integer(map_avail['node_age']['items'])}/{_integer(map_avail['node_age']['items'])} | {_integer(anchor_avail['node_age']['items'])}/{_integer(anchor_avail['node_age']['items'])} | decision-time; derived only from immutable `created_at` |",
            f"| Recorded or immutable level | {_integer(reason_count(train_avail['level'], 'available_from_recorded_organic_result'))} recorded | {_integer(reason_count(organic_avail['level'], 'available_from_recorded_organic_result'))} recorded | {_integer(reason_count(map_avail['level'], 'available_from_immutable_node_level'))} immutable | {_integer(reason_count(anchor_avail['level'], 'available_from_recorded_organic_result'))} recorded + {_integer(reason_count(anchor_avail['level'], 'available_from_immutable_node_level'))} immutable | decision-time |",
            f"| Strictly matured past non-consumption | {_integer(train_avail['past_nonconsumption']['items'])}/{_integer(train_avail['past_nonconsumption']['items'])} | {_integer(organic_avail['past_nonconsumption']['items'])}/{_integer(organic_avail['past_nonconsumption']['items'])} | {_integer(map_avail['past_nonconsumption']['items'])}/{_integer(map_avail['past_nonconsumption']['items'])} | {_integer(anchor_avail['past_nonconsumption']['items'])}/{_integer(anchor_avail['past_nonconsumption']['items'])} | decision-time; only outcomes ending no later than the current delivery instant |",
            f"| Recorded delivery scores | {_integer(reason_count(train_avail['recorded_delivery_scores'], 'available_from_recorded_organic_result'))}/{_integer(train_avail['recorded_delivery_scores']['items'])} | {_integer(reason_count(organic_avail['recorded_delivery_scores'], 'available_from_recorded_organic_result'))}/{_integer(organic_avail['recorded_delivery_scores']['items'])} | {_integer(0)}/{_integer(map_avail['recorded_delivery_scores']['items'])} | {_integer(reason_count(anchor_avail['recorded_delivery_scores'], 'available_from_recorded_organic_result'))}/{_integer(anchor_avail['recorded_delivery_scores']['items'])} | analysis-only; missing for every map item, so not transferable |",
            f"| Content form and size | {_integer(train_avail['snapshot_content_form']['items'])} snapshot-only | {_integer(organic_avail['snapshot_content_form']['items'])} snapshot-only | {_integer(map_avail['snapshot_content_form']['items'])} snapshot-only | {_integer(anchor_avail['snapshot_content_form']['items'])} snapshot-only | sensitivity only; not recorded at delivery |",
            f"| Historical context and provenance | 0 reconstructed; {_integer(train_avail['historical_context_provenance']['items'])} snapshot-only | 0; {_integer(organic_avail['historical_context_provenance']['items'])} snapshot-only | 0; {_integer(map_avail['historical_context_provenance']['items'])} snapshot-only | 0; {_integer(anchor_avail['historical_context_provenance']['items'])} snapshot-only | not versioned; excluded from fitting |",
            f"| Context shape | {_integer(train_avail['snapshot_context_shape']['items'])} snapshot-only | {_integer(organic_avail['snapshot_context_shape']['items'])} snapshot-only | {_integer(map_avail['snapshot_context_shape']['items'])} snapshot-only | {_integer(anchor_avail['snapshot_context_shape']['items'])} snapshot-only | sensitivity only; not recorded at delivery |",
            f"| Historical cascade stage | 0/{_integer(train_avail['cascade_stage']['items'])} | 0/{_integer(organic_avail['cascade_stage']['items'])} | 0/{_integer(map_avail['cascade_stage']['items'])} | 0/{_integer(anchor_avail['cascade_stage']['items'])} | not recorded; not reconstructed |",
            "| Current mutable node statistics | not read | not read | not read | not read | current `access_count`, `usefulness_score`, and `last_accessed`, plus `updated_at`-derived usage, are rejected |",
            "| Identity and grouping fields | split construction only | split construction only | split construction only | split construction only | identifiers, task metadata, hosts, source identity, cache identity, and session identity are forbidden model features and are not published |",
            "| Outcome label | target only | target only | target only | target only | never a predictor |",
            "| Observed-map rows | 0 in fit | evaluation only | evaluation only | not in primary fit | no fit contamination and no evaluation refit |",
            "| Candidate holdout | not accessed | not accessed | not accessed | not accessed | no holdout observations exist in this evidence |",
            "",
            f"The machine-readable leakage audit additionally records an empty mutable-column read set, `future_mutated_node_stats_read=false`, and the exact history cut `{analysis['leakage_audit']['past_history_cut']}`.",
            "",
        )
    )

    _append_header(lines, "Decision-time associations")
    denominators = [
        f"{_integer(_manifest_cohort(manifest, key)['consumed'])}/{_integer(int(_manifest_cohort(manifest, key)['items']) - int(_manifest_cohort(manifest, key)['consumed']))}"
        for key, _label, _role in COHORTS
    ]
    lines.extend(
        (
            f"Each association below is univariate: `C / N; Δ` means the feature mean among consumed items, the mean among non-consumed items, and `C − N`. A positive sign is association, not causation. For these {_word_number(len(primary_features))} features there is no missingness, so the exact consumed/non-consumed denominators are {denominators[0]} for organic train, {denominators[1]} for organic evaluation, {denominators[2]} for observed-map evaluation, and {denominators[3]} for the secondary-anchor sensitivity cohort.",
            "",
            "| Decision-time feature | Organic train C / N; Δ | Organic eval C / N; Δ | Observed-map eval C / N; Δ | Secondary anchor C / N; Δ |",
            "|---|---|---|---|---|",
        )
    )
    for feature in primary_features:
        association = analysis["associations"][feature]
        cells = [_association_cell(association[key]) for key, _label, _role in COHORTS]
        lines.append(f"| `{feature}` | {' | '.join(cells)} |")
    lines.extend(
        (
            "",
            "Transferability requires the same non-zero direction in organic train, organic evaluation, and observed-map evaluation.",
            "",
            "| Feature | Train direction | Organic-eval direction | Map-eval direction | Same sign? | Interpretation |",
            "|---|---|---|---|---|---|",
        )
    )
    for feature in primary_features:
        evidence = analysis["transferability"][feature]
        directions = evidence["directions"]
        lines.append(
            f"| `{feature}` | {directions['organic_train']} | {directions['organic_eval']} | {directions['observed_map_eval']} | {_yes_no(evidence['same_sign'])} | {TRANSFER_INTERPRETATIONS[feature]} |"
        )
    lines.extend(
        (
            "",
            "The age, trace-level, and concept-level signals must not be described as transferring. The four past-history count/rate associations and schema level have the same sign, but the frozen combined policy still fails the required transfer rate.",
            "",
        )
    )

    _append_header(lines, "Frozen primary model and transfer result")
    lines.extend(
        (
            f"The primary model is deterministic L2 logistic regression, fitted once on all {_integer(model['fit_items'])} organic-train items and their {_integer(model['fit_consumed'])} positive outcomes. It ran {_integer(model['iterations'])} iterations with learning rate `{model['learning_rate']}`, L2 value `{model['l2']}`, intercept `{_fixed(model['intercept'], 10)}`, organic-train mean imputation, and no missingness indicator. Its probability threshold, `{model['threshold']}`, was fitted only on train by maximum F1 subject to selecting at least 20% of train. It was then applied without refitting.",
            "",
            "Coefficients are on standardized features and are adjusted model weights; they need not have the same sign as a feature's univariate association.",
            "",
            "| Feature | Train mean | Train scale | Standardized coefficient |",
            "|---|---:|---:|---:|",
        )
    )
    for feature in primary_features:
        standardization = model["standardization"][feature]
        lines.append(
            f"| `{feature}` | {_fixed(standardization['mean'], 10)} | {_fixed(standardization['scale'], 10)} | {_signed(model['coefficients'][feature], 10)} |"
        )
    lines.extend(
        (
            "",
            "| Applied cohort | Total items | Selected items | Selection rate | Selected events | Selected transport sessions | Selected consumed | Selected rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for key, label, _role in PRIMARY_EVALUATION_COHORTS:
        evaluation = model_evaluations[key]
        lines.append(
            f"| {label} | {_integer(evaluation['total_items'])} | {_integer(evaluation['items'])} | {_fixed(evaluation['selection_rate'], 6)} | {_integer(evaluation['events'])} | {_integer(evaluation['transport_sessions'])} | {_integer(evaluation['consumed'])} | {_fixed(evaluation['rate'], 6)} |"
        )
    derivation = transfer["threshold_derivation"]
    lines.extend(
        (
            "",
            f"The unchanged decision floor is `{_fixed(transfer['threshold'], 3)}`, derived from the frozen organic rate `{derivation['frozen_organic_rate']}` times the sealed relative floor `{derivation['sealed_relative_floor']}`, with the preregistered rounded floor retained. For observed-map transfer:",
            "",
            "| Gate | Observed | Required | Passed? |",
            "|---|---:|---:|---|",
            f"| Consumption rate | {_fixed(map_evaluation['rate'], 6)} | at least {_fixed(transfer['threshold'], 3)} | {_yes_no(floor_pass, emphasize_failure=True)} |",
            f"| Events | {_integer(map_evaluation['events'])} | at least {_integer(minimums['events'])} | {_yes_no(int(map_evaluation['events']) >= int(minimums['events']))} |",
            f"| Items | {_integer(map_evaluation['items'])} | at least {_integer(minimums['items'])} | {_yes_no(int(map_evaluation['items']) >= int(minimums['items']))} |",
            f"| Transport sessions | {_integer(map_evaluation['transport_sessions'])} | at least {_integer(minimums['transport_sessions'])} | {_yes_no(int(map_evaluation['transport_sessions']) >= int(minimums['transport_sessions']))} |",
            "",
            f"Thus the cohort minimums {'passed' if minimums_pass else 'failed'}, the unchanged `{_fixed(transfer['threshold'], 3)}` threshold {'passed' if floor_pass else 'failed'}, and the combined verdict is **{transfer['verdict']}**. The unfiltered observed-map rate was {_integer(map_cohort['consumed'])}/{_integer(map_cohort['items'])} = `{_fixed(map_cohort['rate'], 6)}`; selection {'raised' if float(map_evaluation['rate']) > float(map_cohort['rate']) else 'did not raise'} the descriptive rate but {'met' if floor_pass else 'did not meet'} the sealed floor.",
            "",
        )
    )

    _append_header(lines, "Snapshot-only ballast diagnostic")
    lines.extend(
        (
            "The form classifier is the emitter-invariant `form/schema/provenance` classifier. Its real-corpus output is a current-snapshot prevalence diagnostic only and never enters fitting. The real corpus has no independent item-level form ground truth, so real-corpus false-positive and false-negative rates are unavailable; the counts below are classifier outputs, not a claim of perfect real-corpus accuracy.",
            "",
            "| Cohort | Eligible items (consumed) | File-chunk items (consumed) | Strategy-stagnation items (consumed) | Supervision-journal items (consumed) | All classified ballast | Ballast prevalence | Ballast consumed rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    ballast_summaries: dict[str, tuple[int, int]] = {}
    for key, label, _role in COHORTS:
        cohort = _manifest_cohort(manifest, key)
        eligible = _content_form(analysis, key, "eligible")
        forms = [_content_form(analysis, key, form) for form in BALLAST_CLASSES]
        ballast_items = sum(int(form.get("items", 0)) for form in forms)
        ballast_consumed = sum(int(form.get("consumed", 0)) for form in forms)
        ballast_summaries[key] = (ballast_items, ballast_consumed)
        form_cells = [
            f"{_integer(form.get('items', 0))} ({_integer(form.get('consumed', 0))})"
            for form in forms
        ]
        consumed_rate = ballast_consumed / ballast_items if ballast_items else 0.0
        lines.append(
            f"| {label} | {_integer(eligible.get('items', 0))} ({_integer(eligible.get('consumed', 0))}) | {' | '.join(form_cells)} | {_integer(ballast_items)}/{_integer(cohort['items'])} | {_fixed(100 * ballast_items / int(cohort['items']), 4)}% | {_integer(ballast_consumed)}/{_integer(ballast_items)} = {_fixed(consumed_rate, 6)} |"
        )
    observed_ballast, observed_ballast_consumed = ballast_summaries["observed_map_eval"]
    organic_file_chunk = _content_form(analysis, "organic_train", "file_chunk_envelope")
    organic_eval_file_chunk = _content_form(analysis, "organic_eval", "file_chunk_envelope")
    organic_file_chunk_items = int(organic_file_chunk.get("items", 0)) + int(
        organic_eval_file_chunk.get("items", 0)
    )
    organic_file_chunk_consumed = int(organic_file_chunk.get("consumed", 0)) + int(
        organic_eval_file_chunk.get("consumed", 0)
    )
    lines.extend(
        (
            "",
            f"The observed-map arm contains {_integer(observed_ballast)} classified ballast items and {'none was' if observed_ballast_consumed == 0 else _integer(observed_ballast_consumed) + ' were'} consumed. Organic file-chunk items, however, were {'all' if organic_file_chunk_items == organic_file_chunk_consumed else 'not all'} counted consumed under the frozen outcome construction. That arm-specific reversal is why the real-corpus form signal is reported as sensitivity evidence and is not smuggled into the decision-time model.",
            "",
            "### Seeded randomized controls",
            "",
        )
    )
    synthetic = analysis["form_classification_audit"]["synthetic"]
    lines.extend(
        (
            f"The fixed seed is `{synthetic['seed']}`. There are {_integer(synthetic['variants_per_ballast_class'])} independently randomized positive variants for each of the {_word_number(len(BALLAST_CLASSES))} ballast classes and {_integer(synthetic['expected_eligible'])} user-authored near-miss negative controls: {_integer(synthetic['cases'])} cases total, with {_integer(synthetic['expected_ballast'])} expected ballast and {_integer(synthetic['expected_eligible'])} expected eligible. Control text is not published.",
            "",
            "| Expected class | Predicted eligible | Predicted file chunk | Predicted strategy stagnation | Predicted supervision journal | Row total |",
            "|---|---:|---:|---:|---:|---:|",
        )
    )
    confusion = synthetic["confusion"]
    control_rows = (
        ("eligible", "Eligible near miss"),
        ("file_chunk_envelope", "File chunk"),
        ("strategy_stagnation", "Strategy stagnation"),
        ("supervision_journal", "Supervision journal"),
    )
    control_columns = ("eligible", *BALLAST_CLASSES)
    for expected, label in control_rows:
        row = confusion.get(expected, {})
        values = [int(row.get(predicted, 0)) for predicted in control_columns]
        lines.append(f"| {label} | {' | '.join(_integer(value) for value in values)} | {_integer(sum(values))} |")
    column_totals = [
        sum(int(confusion.get(expected, {}).get(predicted, 0)) for expected, _label in control_rows)
        for predicted in control_columns
    ]
    lines.extend(
        (
            f"| Column total | {' | '.join(_integer(value) for value in column_totals)} | {_integer(sum(column_totals))} |",
            "",
            f"The synthetic controls have {_integer(synthetic['true_positives'])} true positives, {_integer(synthetic['true_negatives'])} true negatives, {_integer(synthetic['false_positives'])} false positives, and {_integer(synthetic['false_negatives'])} false negatives. This validates the fixed randomized control set; it does not supply missing real-corpus ground truth.",
            "",
        )
    )

    _append_header(lines, "Snapshot-only content, provenance, and context associations")
    lines.extend(
        (
            "These features were computed from the pinned current snapshot. They are not decision-time evidence, were excluded from fitting, and must not be used to claim a historical causal or transferable effect. All values are present in all four cohorts, so their exact consumed/non-consumed denominators are the cohort denominators stated above. Notation remains `C / N; Δ`.",
            "",
            "| Snapshot-only feature | Organic train C / N; Δ | Organic eval C / N; Δ | Observed-map eval C / N; Δ | Secondary anchor C / N; Δ |",
            "|---|---|---|---|---|",
        )
    )
    analysis_only = analysis["feature_policy"]["analysis_only_features"]
    snapshot_features = [feature for feature in analysis_only if feature not in {"cascade_anchor", *SCORE_FEATURES}]
    for feature in snapshot_features:
        association = analysis["associations"][feature]
        cells = [_association_cell(association[key]) for key, _label, _role in COHORTS]
        lines.append(f"| `{feature}` | {' | '.join(cells)} |")
    lines.append("")

    _append_header(lines, "Recorded-score associations and unavailable cascade stage")
    score_available = analysis["associations"][SCORE_FEATURES[0]]["secondary_anchor"]
    lines.extend(
        (
            f"Recorded delivery scores are historical for organic delivery results, but no residual score was recorded in any of the {_integer(map_cohort['items'])} observed-map payloads. They are therefore analysis-only and cannot establish transfer. For organic train the score denominators are {denominators[0].replace('/', ' consumed and ', 1)} non-consumed; for organic evaluation, {denominators[1].replace('/', ' and ', 1)}. In the secondary-anchor diagnostic, scores are available for {_integer(score_available['available'])} items—{_integer(score_available['consumed'])} consumed and {_integer(score_available['not_consumed'])} non-consumed—and missing for {_integer(score_available['missing'])}. Observed-map availability is {_integer(analysis['associations'][SCORE_FEATURES[0]]['observed_map_eval']['available'])} and missingness is {_integer(analysis['associations'][SCORE_FEATURES[0]]['observed_map_eval']['missing'])}.",
            "",
            "| Recorded score | Organic train C / N; Δ | Organic eval C / N; Δ | Observed-map eval | Secondary anchor C / N; Δ |",
            "|---|---|---|---|---|",
        )
    )
    for feature in SCORE_FEATURES:
        association = analysis["associations"][feature]
        cells = [_association_cell(association[key]) for key, _label, _role in COHORTS]
        lines.append(f"| `{feature}` | {' | '.join(cells)} |")
    cascade = analysis["associations"]["cascade_anchor"]
    lines.extend(
        (
            "",
            f"`cascade_anchor` is unavailable for all {_integer(cascade['organic_train']['missing'])} organic-train, {_integer(cascade['organic_eval']['missing'])} organic-evaluation, {_integer(cascade['observed_map_eval']['missing'])} observed-map-evaluation, and {_integer(cascade['secondary_anchor']['missing'])} secondary-anchor items. Historical cascade stage was not recorded at delivery and was not reconstructed.",
            "",
        )
    )

    _append_header(lines, "Complete omission and rejection ledger")
    lines.extend(
        (
            f"The {_word_number(len(primary_features))} decision-time features listed in the model table are the entire frozen primary model surface. Every other measured feature is omitted from fitting as follows.",
            "",
            "| Omitted feature | Exact reason |",
            "|---|---|",
        )
    )
    for feature in analysis_only:
        lines.append(f"| `{feature}` | {OMISSION_REASONS[feature]} |")
    nontransferring = [feature for feature in primary_features if not analysis["transferability"][feature]["same_sign"]]
    transferring = [feature for feature in primary_features if analysis["transferability"][feature]["same_sign"]]
    nontransfer_text = ", ".join(f"`{feature}`" for feature in nontransferring[:-1])
    if len(nontransferring) > 1:
        nontransfer_text += f", and `{nontransferring[-1]}`"
    elif nontransferring:
        nontransfer_text = f"`{nontransferring[0]}`"
    lines.extend(
        (
            "",
            f"{_word_number(len(nontransferring)).capitalize()} fitted decision-time features are also omitted from any claim of transferable direction: {nontransfer_text}, because their univariate association signs do not agree across train and both evaluation arms. The remaining {_word_number(len(transferring))} same-sign features are not declared deployable because the combined frozen model failed transfer.",
            "",
            "The following values are rejected as predictors rather than treated as missing features:",
            "",
            "| Rejected value | Reason |",
            "|---|---|",
        )
    )
    for feature in analysis["feature_policy"]["forbidden_as_features"]:
        label, reason = REJECTION_ROWS[feature]
        lines.append(f"| {label} | {reason} |")
    lines.append("")

    _append_header(lines, "Aggregate-only publication audit")
    privacy = analysis["privacy_audit"]
    lines.append(
        f"The published artifacts contain no item-level rows or corpus text. The privacy audit compared {_integer(privacy['raw_strings_compared'])} raw strings against {_integer(privacy['published_string_cells_compared'])} published string cells and found {_word_number(privacy['exact_raw_string_matches'])} exact raw-string matches. No actual identifier, query, task value, cache identity, session identity, raw content, or candidate-holdout observation is included here."
    )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--out", type=Path, help="write Markdown here instead of stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = _read_object(args.manifest)
        analysis = _read_object(args.analysis)
        rendered = render_report(manifest, analysis)
        if args.out is None:
            sys.stdout.write(rendered)
        else:
            _assert_not_holdout(args.out)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered, encoding="utf-8")
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
