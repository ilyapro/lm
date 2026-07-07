# Recall-replay baseline

Generated: 2026-07-07T20:23:08Z | DB: `/tmp/lm_backfill_copy.sqlite3`

## Corpus and split

- Events: 47011 total, 46991 usable (skipped: 20 empty, 0 incomplete results)
- Labeled (consumed) events: 7600 | span 2026-05-15T07:25:32Z .. 2026-07-07T19:39:43Z
- Cutoff `2026-06-10T00:00:00Z`: train 38016 events (5376 labeled) / holdout 8975 events (2224 labeled, 19.1% of usable events, 29.3% of labeled)

## Labels

- Protocol: `grounded` {'protocol': 'grounded', 'min_containment': 0.25, 'reconsume_min_traces': 2, 'usefulness_threshold': 0.8}
- 9911 useful of 59924 labeled results; 3612 of 7600 labeled events have at least one useful result
- Discrimination over 7579 multi-result events: strict subset 46.8%, all-useful 0.9%, none-useful 52.4%, mean useful share 0.16

## Metrics by scheme
| scheme | split | events | w/useful | hit@1 | hit@5 | hit@10 | MRR | graph-unique | zeroed hit@5 | zeroed MRR | drop@1 | drop@5 | drop@max | drop-all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recorded | overall | 7600 | 3612 | 0.328 | 0.859 | 0.993 | 0.539 | 0.008 | 0.859 | 0.539 | 0.000 | 0.000 | 0.001 | 0.001 |
| recorded | train | 5376 | 2075 | 0.265 | 0.822 | 0.989 | 0.484 | 0.005 | 0.822 | 0.484 | 0.000 | 0.000 | 0.002 | 0.002 |
| recorded | holdout | 2224 | 1537 | 0.413 | 0.909 | 0.998 | 0.613 | 0.010 | 0.909 | 0.613 | 0.000 | 0.000 | 0.000 | 0.000 |
| live_weights | overall | 7600 | 3612 | 0.491 | 0.923 | 0.996 | 0.669 | 0.008 | 0.922 | 0.669 | 0.008 | 0.012 | 0.001 | 0.001 |
| live_weights | train | 5376 | 2075 | 0.461 | 0.904 | 0.995 | 0.643 | 0.005 | 0.903 | 0.644 | 0.007 | 0.008 | 0.002 | 0.002 |
| live_weights | holdout | 2224 | 1537 | 0.533 | 0.949 | 0.997 | 0.704 | 0.010 | 0.948 | 0.703 | 0.009 | 0.017 | 0.000 | 0.000 |
| replayed_winner_take_all | overall | 7600 | 3612 | 0.488 | 0.925 | 0.997 | 0.666 | 0.008 | 0.924 | 0.666 | 0.006 | 0.013 | 0.001 | 0.001 |
| replayed_winner_take_all | train | 5376 | 2075 | 0.454 | 0.907 | 0.996 | 0.637 | 0.005 | 0.906 | 0.638 | 0.003 | 0.010 | 0.002 | 0.002 |
| replayed_winner_take_all | holdout | 2224 | 1537 | 0.533 | 0.949 | 0.997 | 0.704 | 0.010 | 0.948 | 0.703 | 0.009 | 0.017 | 0.000 | 0.000 |
| replayed_proportional | overall | 7600 | 3612 | 0.487 | 0.927 | 0.996 | 0.665 | 0.008 | 0.926 | 0.665 | 0.008 | 0.011 | 0.001 | 0.001 |
| replayed_proportional | train | 5376 | 2075 | 0.450 | 0.911 | 0.996 | 0.635 | 0.005 | 0.911 | 0.637 | 0.003 | 0.010 | 0.002 | 0.002 |
| replayed_proportional | holdout | 2224 | 1537 | 0.537 | 0.949 | 0.997 | 0.706 | 0.010 | 0.947 | 0.704 | 0.013 | 0.011 | 0.000 | 0.000 |

### Per-scope (overall split)

| scheme | scope | events | w/useful | hit@5 | MRR | graph-unique | drop@5 |
|---|---|---|---|---|---|---|---|
| recorded | _other | 86 | 41 | 0.951 | 0.723 | 0.000 | 0.000 |
| recorded | global | 115 | 79 | 0.975 | 0.844 | 0.000 | 0.000 |
| recorded | project:ae | 1252 | 384 | 0.818 | 0.518 | 0.009 | 0.000 |
| recorded | project:lm | 265 | 122 | 0.943 | 0.680 | 0.000 | 0.000 |
| recorded | project:mm | 63 | 21 | 0.762 | 0.604 | 0.000 | 0.000 |
| recorded | project:octopus | 1546 | 779 | 0.854 | 0.481 | 0.008 | 0.001 |
| recorded | project:online | 2318 | 760 | 0.784 | 0.469 | 0.001 | 0.000 |
| recorded | project:x | 1955 | 1426 | 0.898 | 0.578 | 0.011 | 0.000 |
| live_weights | _other | 86 | 41 | 0.976 | 0.741 | 0.000 | 0.009 |
| live_weights | global | 115 | 79 | 0.962 | 0.857 | 0.000 | 0.018 |
| live_weights | project:ae | 1252 | 384 | 0.885 | 0.607 | 0.009 | 0.004 |
| live_weights | project:lm | 265 | 122 | 0.959 | 0.778 | 0.000 | 0.005 |
| live_weights | project:mm | 63 | 21 | 0.905 | 0.915 | 0.000 | 0.027 |
| live_weights | project:octopus | 1546 | 779 | 0.900 | 0.582 | 0.008 | 0.014 |
| live_weights | project:online | 2318 | 760 | 0.878 | 0.636 | 0.001 | 0.004 |
| live_weights | project:x | 1955 | 1426 | 0.964 | 0.725 | 0.011 | 0.016 |
| replayed_winner_take_all | _other | 86 | 41 | 0.976 | 0.720 | 0.000 | 0.038 |
| replayed_winner_take_all | global | 115 | 79 | 0.962 | 0.867 | 0.000 | 0.022 |
| replayed_winner_take_all | project:ae | 1252 | 384 | 0.896 | 0.607 | 0.009 | 0.008 |
| replayed_winner_take_all | project:lm | 265 | 122 | 0.967 | 0.755 | 0.000 | 0.005 |
| replayed_winner_take_all | project:mm | 63 | 21 | 0.905 | 0.916 | 0.000 | 0.027 |
| replayed_winner_take_all | project:octopus | 1546 | 779 | 0.905 | 0.584 | 0.008 | 0.015 |
| replayed_winner_take_all | project:online | 2318 | 760 | 0.875 | 0.623 | 0.001 | 0.004 |
| replayed_winner_take_all | project:x | 1955 | 1426 | 0.964 | 0.724 | 0.011 | 0.016 |
| replayed_proportional | _other | 86 | 41 | 0.976 | 0.720 | 0.000 | 0.038 |
| replayed_proportional | global | 115 | 79 | 0.962 | 0.874 | 0.000 | 0.022 |
| replayed_proportional | project:ae | 1252 | 384 | 0.891 | 0.605 | 0.009 | 0.009 |
| replayed_proportional | project:lm | 265 | 122 | 0.959 | 0.752 | 0.000 | 0.005 |
| replayed_proportional | project:mm | 63 | 21 | 0.905 | 0.916 | 0.000 | 0.027 |
| replayed_proportional | project:octopus | 1546 | 779 | 0.919 | 0.578 | 0.008 | 0.014 |
| replayed_proportional | project:online | 2318 | 760 | 0.874 | 0.621 | 0.001 | 0.002 |
| replayed_proportional | project:x | 1955 | 1426 | 0.965 | 0.727 | 0.011 | 0.011 |

## Label sensitivity (live_weights scheme, overall)

| protocol | strict-subset | all-useful | none-useful | events w/useful | hit@5 | MRR |
|---|---|---|---|---|---|---|
| grounded | 46.8% | 0.9% | 52.4% | 3612 | 0.923 | 0.669 |
| reconsumed | 36.2% | 63.1% | 0.7% | 7543 | 0.995 | 0.929 |
| usefulness | 78.3% | 16.6% | 5.1% | 7198 | 0.947 | 0.660 |
| grounded_or_reconsumed | 27.1% | 72.6% | 0.2% | 7578 | 0.997 | 0.956 |

## Labeling protocol

A recall event is **labeled** when a later `memory_remember` consumed it
(`feedback_trace_id` is set). Within a labeled event, a result is
**confirmed useful** under the primary `grounded` protocol when the
IDF-weighted share of the result node's content tokens that also appear in
the consuming trace's content (containment) is at least `min_containment`.
IDF is computed over the corpus of result-node and consuming-trace contents,
so boilerplate tokens contribute little and identifiers/paths dominate.

Why not the recorded linkage? `feedback.apply_pending_recall_feedback` links
**every** result of a consumed event to the consuming trace
(provenance `recalled_nodes`/`source_traces`, `related` edges), so
"linked by the consuming trace" marks all results useful and every ranking
metric becomes vacuous. Grounding discriminates within the result list.
Alternative protocols (`reconsumed`, `usefulness`,
`grounded_or_reconsumed`) are reported in the label-sensitivity section:
re-consumption is nearly vacuous on this corpus (a few thousand hub nodes
recirculate across most events), and long-run usefulness is
rank-circular (it accrues via the same rank-decayed reinforcement loop the
harness audits) with a saturated distribution (median 1.0).

## Metric definitions

* `hit@k` / `mrr` — over labeled events with at least one confirmed-useful
  result: whether/where the first useful result appears in the re-ranked
  order of the recorded candidates. A useful result that a scheme ranks to
  zero score counts as a miss.
* `graph_unique_share` — share of confirmed-useful results whose recorded
  evidence is graph-only (`graph_score > 0`, `bm25 = vector = 0`).
  Scheme-independent (recorded evidence), repeated per scheme for
  consumer convenience.
* `graph_zeroed.*` — counterfactual re-rank of the same candidates with
  `graph_score` forced to 0 (which also disables the in-rank graph rescue):
  `dropped_topK` is the share of confirmed-useful results in the scheme's
  top-K that leave the top-K; `dropped_entirely` is the share of ranked
  useful results that drop to zero score; `hit@5`/`mrr` are re-computed on
  the zeroed ranking. `dropped_*` can sit below `graph_unique_share`:
  schema nodes whose only method evidence is graph still survive zeroing
  through the trigger-score override (on this corpus most graph-only
  confirmed-useful results are exactly such schema nodes).

## Limitations

* **Candidate selection bias**: recorded results are only the
  top-`max_results` under the *old* live weights + rescue ranking. Replay
  can re-order or drop recorded candidates but can never surface candidates
  the old ranking excluded, so absolute metric levels are optimistic and
  graph-value estimates are lower bounds relative to a full re-retrieval.
* **Neutral node stats**: re-ranked schemes hold node-level feedback
  multipliers (confidence/usefulness/access, supersedes corrections) at
  neutral because event-time node stats are not recorded. The `recorded`
  scheme preserves the historical order including those multipliers.
* **Grounding is textual**: results used conceptually without shared
  identifiers are missed (~half of labeled events have no grounded result;
  they are excluded from hit/MRR denominators). Consuming traces deleted
  since (`missing_consuming_traces`) cannot be grounded.
* **Trajectory approximations**: initial weights are the configured family
  defaults; per-scope learning rates come from the snapshot weights table
  (the live adaptive tuner's final state); floor evidence gates are
  evaluated against the snapshot, not historically; explicit
  feedback/teach events are not in the corpus and are not replayed.

