#!/usr/bin/env python3
"""A/B the live credit rule over recorded recall history, with a cutoff holdout.

Arms (``living_memory.replay.CREDIT_RULES``), all replaying the same event
stream through the same weight-update arithmetic and differing only in which
results earn credit:

* ``proportional`` — the incumbent live rule: every result of a consumed
  event is reinforced, rank-decayed.
* ``grounded`` — only results the consuming trace grounded; the rest neutral.
* ``grounded_negative`` — grounded results reinforced, ungrounded ones blamed
  at ``feedback.UNGROUNDED_NEGATIVE_FACTOR``.

Generalization design. Weight updates stop at ``--cutoff``, so holdout events
are ranked under weights each arm froze before it ever saw them; the holdout
slice is the decision surface and the train slice is reported only for
contrast.

Falsifiability control. The grounded arms gate on grounding, and the primary
``grounded`` label also derives from grounding — so a win under that label
alone could be definitional rather than real. Every arm is therefore also
scored under labels that use no grounding at all (``reconsumed``:
later re-consumption by other traces; ``usefulness``: accrued node
usefulness). A grounded arm that wins under the grounded label while
regressing under both independent labels is reported as NOT clearing the bar.

Usage::

    python3 scripts/credit_rule_ab.py \
        --db ~/.cache/living-memory-harness/snapshot.sqlite3 \
        --cutoff 2026-06-10T00:00:00Z \
        --report artifacts/grounding/credit-ab.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from living_memory.config import MemoryConfig  # noqa: E402
from living_memory.feedback import UNGROUNDED_NEGATIVE_FACTOR  # noqa: E402
from living_memory.replay import (  # noqa: E402
    CREDIT_RULES,
    LabelConfig,
    WeightTrajectory,
    _weights_dict,
    apply_grounding,
    apply_labels,
    build_label_data,
    load_live_weights,
    load_replay_events,
    normalize_cutoff,
    open_readonly,
    run_scheme,
    scope_buckets,
    snapshot_evidence,
)

ARMS = ("proportional", "grounded", "grounded_negative")
INCUMBENT = "proportional"
#: Labels that do not derive from content grounding; the grounded arms must
#: not regress on these for a win under the grounded label to count.
CONTROL_LABELS = ("reconsumed", "usefulness")
PRIMARY_LABEL = "grounded"
METRIC_KEYS = ("hit@1", "hit@5", "hit@10", "mrr")
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260818
#: A learned weight vector that has been driven into a corner is a defect the
#: aggregate metrics can hide, because the labels available here are all
#: hub-node biased and a degenerate vector can score well on them. Flag any
#: scope whose graph channel exceeds this share.
GRAPH_DEGENERACY_LIMIT = 0.5


def _paired_bootstrap(
    arm: list[tuple[str, float, float]],
    incumbent: list[tuple[str, float, float]],
    *,
    index: int,
) -> dict[str, Any]:
    """95% CI for the per-event mean delta (arm - incumbent), paired by event.

    Both arms rank the same events, so the samples pair exactly; resampling
    events (not results) respects that unit of independence.
    """

    base = {event_id: value[index - 1] for event_id, *value in incumbent}
    deltas = [
        value[index - 1] - base[event_id]
        for event_id, *value in arm
        if event_id in base
    ]
    if not deltas:
        return {"n": 0}
    rng = random.Random(BOOTSTRAP_SEED)
    size = len(deltas)
    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        means.append(statistics.fmean(rng.choices(deltas, k=size)))
    means.sort()
    low = means[int(0.025 * BOOTSTRAP_RESAMPLES)]
    high = means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]
    return {
        "n": size,
        "mean_delta": round(statistics.fmean(deltas), 6),
        "ci95_low": round(low, 6),
        "ci95_high": round(high, 6),
        # "Distinguishable" means the CI excludes zero: only then is the gap
        # evidence rather than resampling noise.
        "distinguishable": low > 0.0 or high < 0.0,
    }


def _weight_sanity(weights: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Detect learned vectors driven into a floor/ceiling corner."""

    graph_heavy = sorted(
        scope
        for scope, value in weights.items()
        if value["graph"] > GRAPH_DEGENERACY_LIMIT
    )
    return {
        "scopes": len(weights),
        "graph_over_limit": len(graph_heavy),
        "graph_over_limit_scopes": graph_heavy,
        "graph_limit": GRAPH_DEGENERACY_LIMIT,
        "max_graph": round(max((v["graph"] for v in weights.values()), default=0.0), 6),
        "mean_graph": round(
            statistics.fmean([v["graph"] for v in weights.values()]) if weights else 0.0, 6
        ),
    }


def run(db_path: str, cutoff_raw: str, *, min_scope_events: int) -> dict[str, Any]:
    cutoff = normalize_cutoff(cutoff_raw)
    connection = open_readonly(db_path)
    try:
        events, load_stats = load_replay_events(connection)
        label_data = build_label_data(connection, events)
        live_weights = load_live_weights(connection)
        evidence = snapshot_evidence(connection)
        buckets = scope_buckets(events, min_scope_events)
        memory_config = MemoryConfig()
        learning_rates = {
            scope: weights.learning_rate for scope, weights in live_weights.items()
        }
        # Grounding is label-independent: graded once, reused by every arm.
        grounding = apply_grounding(events, label_data)

        results: dict[str, Any] = {}
        final_weights: dict[str, Any] = {}
        for label_protocol in (PRIMARY_LABEL, *CONTROL_LABELS):
            label_summary = apply_labels(
                events, label_data, LabelConfig(protocol=label_protocol)
            )
            per_arm: dict[str, Any] = {}
            holdout_samples: dict[str, list[tuple[str, float, float]]] = {}
            for arm in ARMS:
                trajectory = WeightTrajectory(
                    memory_config,
                    evidence=evidence,
                    snapshot_learning_rates=learning_rates,
                )
                report = run_scheme(
                    f"replayed_{arm}",
                    events,
                    buckets,
                    cutoff,
                    trajectory=trajectory,
                    credit_rule=CREDIT_RULES[arm],
                )
                blocks = report.to_dict()
                per_arm[arm] = {
                    split: {key: blocks[split][key] for key in METRIC_KEYS}
                    | {
                        "events": blocks[split]["events"],
                        "events_with_useful": blocks[split]["events_with_useful"],
                    }
                    for split in ("overall", "train", "holdout")
                    if split in blocks
                }
                per_arm[arm]["weight_updates"] = sum(trajectory.update_counts.values())
                holdout_samples[arm] = list(report.splits["holdout"].per_event)
                if label_protocol == PRIMARY_LABEL:
                    final_weights[arm] = {
                        scope: _weights_dict(weights)
                        for scope, weights in sorted(trajectory.weights.items())
                    }
            for arm in ARMS:
                if arm == INCUMBENT:
                    continue
                per_arm[arm]["paired_vs_incumbent"] = {
                    "mrr": _paired_bootstrap(
                        holdout_samples[arm], holdout_samples[INCUMBENT], index=1
                    ),
                    "hit@5": _paired_bootstrap(
                        holdout_samples[arm], holdout_samples[INCUMBENT], index=2
                    ),
                }
            results[label_protocol] = {
                "label": label_summary["config"],
                "discrimination": label_summary["discrimination"],
                "events_with_useful": label_summary["events_with_useful"],
                "useful_results": label_summary["useful_results"],
                "arms": per_arm,
            }
    finally:
        connection.close()

    sanity = {arm: _weight_sanity(final_weights[arm]) for arm in ARMS}
    verdict = _decide(results, sanity)
    return {
        "db_path": str(db_path),
        "cutoff": cutoff,
        "corpus": load_stats,
        "grounding": grounding,
        "ungrounded_negative_factor": UNGROUNDED_NEGATIVE_FACTOR,
        "incumbent": INCUMBENT,
        "arms": list(ARMS),
        "primary_label": PRIMARY_LABEL,
        "control_labels": list(CONTROL_LABELS),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "results": results,
        "final_weights": final_weights,
        "weight_sanity": sanity,
        "verdict": verdict,
    }


def _decide(results: dict[str, Any], sanity: dict[str, Any]) -> dict[str, Any]:
    """Pick the arm that improves the primary label without regressing elsewhere.

    Three conditions, applied in order:

    1. **No degenerate weights.** An arm whose learned vectors are driven into
       a corner is disqualified outright. Every label available here is
       hub-node biased, so a degenerate vector can post good aggregate numbers
       for the wrong reason; this check does not go through the labels at all.
    2. **No significant regression under any label.** A holdout delta counts
       as a regression only when the paired bootstrap CI excludes zero — an
       arm indistinguishable from the incumbent has not regressed.
    3. **A significant gain under the primary label.** Ties fall back to the
       incumbent: no change without evidence.
    """

    verdict: dict[str, Any] = {"per_arm": {}}
    for arm in ARMS:
        if arm == INCUMBENT:
            continue
        checks = {}
        for label_protocol, block in results.items():
            base = block["arms"][INCUMBENT]["holdout"]
            arm_block = block["arms"][arm]["holdout"]
            paired = block["arms"][arm]["paired_vs_incumbent"]
            checks[label_protocol] = {
                metric: {
                    "incumbent": base[metric],
                    "arm": arm_block[metric],
                    "delta": round(arm_block[metric] - base[metric], 6),
                    "ci95": [paired[metric]["ci95_low"], paired[metric]["ci95_high"]],
                    "distinguishable": paired[metric]["distinguishable"],
                    "regression": paired[metric]["distinguishable"]
                    and paired[metric]["mean_delta"] < 0.0,
                    "gain": paired[metric]["distinguishable"]
                    and paired[metric]["mean_delta"] > 0.0,
                }
                for metric in ("hit@5", "mrr")
            }
        degenerate = sanity[arm]["graph_over_limit"] > 0
        regressions = [
            f"{label}.{metric}"
            for label, block in checks.items()
            for metric, item in block.items()
            if item["regression"]
        ]
        gains = [
            metric for metric, item in checks[PRIMARY_LABEL].items() if item["gain"]
        ]
        verdict["per_arm"][arm] = {
            "checks": checks,
            "weights_degenerate": degenerate,
            "significant_regressions": regressions,
            "primary_gains": gains,
            "pass": not degenerate and not regressions and bool(gains),
        }
    passing = [arm for arm, block in verdict["per_arm"].items() if block["pass"]]
    verdict["winner"] = (
        max(
            passing,
            key=lambda arm: verdict["per_arm"][arm]["checks"][PRIMARY_LABEL]["mrr"]["delta"],
        )
        if passing
        else INCUMBENT
    )
    return verdict


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Credit-rule A/B: grounded vs reinforce-everything",
        "",
        f"DB: `{report['db_path']}` | cutoff `{report['cutoff']}` | "
        f"{report['corpus']['usable_events']} usable events",
        "",
        f"- Grounding at min_containment {report['grounding']['min_containment']}: "
        f"{report['grounding']['grounded_results']} of "
        f"{report['grounding']['graded_results']} consumed results grounded "
        f"({report['grounding']['grounded_share']:.4f}); per-event vs whole-corpus "
        f"IDF agreement {report['grounding']['corpus_idf_agreement']:.4f}",
        f"- Incumbent `{report['incumbent']}` reinforces **every** result; the "
        f"grounded arms reinforce that {report['grounding']['grounded_share']:.1%} "
        "share only",
        f"- `grounded_negative` blames ungrounded results at "
        f"{report['ungrounded_negative_factor']} of the positive signal",
        "",
        "## Holdout metrics by label and arm",
        "",
        "| label | arm | events w/useful | hit@1 | hit@5 | hit@10 | MRR | weight updates |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label_protocol, block in report["results"].items():
        for arm in report["arms"]:
            holdout = block["arms"][arm]["holdout"]
            lines.append(
                f"| {label_protocol} | {arm} | {holdout['events_with_useful']} "
                f"| {holdout['hit@1']:.4f} | {holdout['hit@5']:.4f} "
                f"| {holdout['hit@10']:.4f} | {holdout['mrr']:.4f} "
                f"| {block['arms'][arm]['weight_updates']} |"
            )
    lines += ["", "## Learned-weight sanity (primary label, final trajectory)", ""]
    lines.append("| arm | scopes | mean graph | max graph | scopes with graph > 0.5 |")
    lines.append("|---|---|---|---|---|")
    for arm in report["arms"]:
        sanity = report["weight_sanity"][arm]
        lines.append(
            f"| {arm} | {sanity['scopes']} | {sanity['mean_graph']:.4f} "
            f"| {sanity['max_graph']:.4f} | {sanity['graph_over_limit']} "
            f"{sanity['graph_over_limit_scopes'] or ''} |"
        )
    lines += ["", "## Verdict", ""]
    lines.append(
        f"Winner: **{report['verdict']['winner']}** "
        f"(incumbent `{report['incumbent']}`)."
    )
    lines.append("")
    lines.append(
        "Deltas are holdout arm-minus-incumbent with a paired bootstrap 95% CI "
        f"({report['bootstrap']['resamples']} resamples over events, seed "
        f"{report['bootstrap']['seed']}). A gap only counts when its CI excludes "
        "zero."
    )
    lines.append("")
    lines.append("| arm | label | metric | incumbent | arm | delta | CI95 | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm, block in report["verdict"]["per_arm"].items():
        for label_protocol, checks in block["checks"].items():
            for metric, item in checks.items():
                if item["regression"]:
                    tag = "REGRESSION"
                elif item["gain"]:
                    tag = "gain"
                else:
                    tag = "indistinguishable"
                lines.append(
                    f"| {arm} | {label_protocol} | {metric} | {item['incumbent']:.6f} "
                    f"| {item['arm']:.6f} | {item['delta']:+.6f} "
                    f"| [{item['ci95'][0]:+.6f}, {item['ci95'][1]:+.6f}] | {tag} |"
                )
    lines.append("")
    for arm, block in report["verdict"]["per_arm"].items():
        lines.append(
            f"- `{arm}`: degenerate weights = {block['weights_degenerate']}, "
            f"significant regressions = {block['significant_regressions'] or 'none'}, "
            f"significant primary gains = {block['primary_gains'] or 'none'} "
            f"-> {'ADOPT' if block['pass'] else 'reject'}"
        )
    lines += [
        "",
        "`grounded` and the primary label share the content-grounding notion, so "
        "the `reconsumed` and `usefulness` rows are the falsifiability control: "
        "they label usefulness without grounding. Both controls are known to be "
        "hub-node biased (the replay module documents `reconsumed` as nearly "
        "vacuous and `usefulness` as rank-circular), which is why a weight-sanity "
        "check that does not go through any label is applied first.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--markdown")
    parser.add_argument("--min-scope-events", type=int, default=50)
    args = parser.parse_args(argv)

    report = run(args.db, args.cutoff, min_scope_events=args.min_scope_events)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = Path(args.markdown) if args.markdown else report_path.with_suffix(".md")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"winner={report['verdict']['winner']} -> {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
