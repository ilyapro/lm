# Recall-replay baseline

Generated: 2026-07-07T14:34:20Z | DB: `/tmp/lm_replay_snapshot.sqlite3`

## Corpus and split

- Events: 46821 total, 46801 usable (skipped: 20 empty, 0 incomplete results)
- Labeled (consumed) events: 7589 | span 2026-05-15T07:25:32Z .. 2026-07-07T14:11:01Z
- Cutoff `2026-06-10T00:00:00Z`: train 38016 events (5376 labeled) / holdout 8785 events (2213 labeled, 18.8% of usable events, 29.2% of labeled)

## Labels

- Protocol: `grounded` {'protocol': 'grounded', 'min_containment': 0.25, 'reconsume_min_traces': 2, 'usefulness_threshold': 0.8}
- 9888 useful of 59847 labeled results; 3604 of 7589 labeled events have at least one useful result
- Discrimination over 7568 multi-result events: strict subset 46.7%, all-useful 0.9%, none-useful 52.4%, mean useful share 0.16

## Metrics by scheme
| scheme | split | events | w/useful | hit@1 | hit@5 | hit@10 | MRR | graph-unique | zeroed hit@5 | zeroed MRR | drop@1 | drop@5 | drop@max | drop-all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recorded | overall | 7589 | 3604 | 0.327 | 0.859 | 0.993 | 0.538 | 0.008 | 0.858 | 0.538 | 0.000 | 0.000 | 0.001 | 0.001 |
| recorded | train | 5376 | 2075 | 0.265 | 0.822 | 0.989 | 0.484 | 0.005 | 0.822 | 0.484 | 0.000 | 0.000 | 0.002 | 0.002 |
| recorded | holdout | 2213 | 1529 | 0.412 | 0.908 | 0.998 | 0.612 | 0.010 | 0.908 | 0.612 | 0.000 | 0.000 | 0.000 | 0.000 |
| live_weights | overall | 7589 | 3604 | 0.500 | 0.924 | 0.997 | 0.675 | 0.008 | 0.925 | 0.675 | 0.006 | 0.006 | 0.001 | 0.001 |
| live_weights | train | 5376 | 2075 | 0.477 | 0.907 | 0.998 | 0.652 | 0.005 | 0.907 | 0.652 | 0.007 | 0.007 | 0.002 | 0.002 |
| live_weights | holdout | 2213 | 1529 | 0.531 | 0.948 | 0.997 | 0.706 | 0.010 | 0.949 | 0.706 | 0.004 | 0.005 | 0.000 | 0.000 |
| floor_defaults | overall | 7589 | 3604 | 0.489 | 0.930 | 0.997 | 0.668 | 0.008 | 0.929 | 0.669 | 0.010 | 0.014 | 0.001 | 0.001 |
| floor_defaults | train | 5376 | 2075 | 0.450 | 0.914 | 0.997 | 0.636 | 0.005 | 0.913 | 0.637 | 0.008 | 0.012 | 0.002 | 0.002 |
| floor_defaults | holdout | 2213 | 1529 | 0.542 | 0.952 | 0.997 | 0.712 | 0.010 | 0.951 | 0.712 | 0.013 | 0.016 | 0.000 | 0.000 |
| uniform | overall | 7589 | 3604 | 0.491 | 0.930 | 0.996 | 0.668 | 0.008 | 0.932 | 0.670 | 0.018 | 0.024 | 0.001 | 0.001 |
| uniform | train | 5376 | 2075 | 0.457 | 0.918 | 0.996 | 0.639 | 0.005 | 0.917 | 0.642 | 0.013 | 0.023 | 0.002 | 0.002 |
| uniform | holdout | 2213 | 1529 | 0.536 | 0.947 | 0.995 | 0.706 | 0.010 | 0.952 | 0.709 | 0.024 | 0.026 | 0.000 | 0.000 |
| replayed_winner_take_all | overall | 7589 | 3604 | 0.497 | 0.925 | 0.997 | 0.672 | 0.008 | 0.926 | 0.672 | 0.003 | 0.006 | 0.001 | 0.001 |
| replayed_winner_take_all | train | 5376 | 2075 | 0.471 | 0.908 | 0.998 | 0.647 | 0.005 | 0.908 | 0.647 | 0.002 | 0.008 | 0.002 | 0.002 |
| replayed_winner_take_all | holdout | 2213 | 1529 | 0.531 | 0.948 | 0.997 | 0.706 | 0.010 | 0.949 | 0.706 | 0.004 | 0.005 | 0.000 | 0.000 |

### Per-scope (overall split)

| scheme | scope | events | w/useful | hit@5 | MRR | graph-unique | drop@5 |
|---|---|---|---|---|---|---|---|
| recorded | _other | 86 | 41 | 0.951 | 0.723 | 0.000 | 0.000 |
| recorded | global | 115 | 79 | 0.975 | 0.844 | 0.000 | 0.000 |
| recorded | project:ae | 1252 | 384 | 0.818 | 0.518 | 0.009 | 0.000 |
| recorded | project:lm | 254 | 114 | 0.939 | 0.673 | 0.000 | 0.000 |
| recorded | project:mm | 63 | 21 | 0.762 | 0.604 | 0.000 | 0.000 |
| recorded | project:octopus | 1546 | 779 | 0.854 | 0.481 | 0.008 | 0.001 |
| recorded | project:online | 2318 | 760 | 0.784 | 0.469 | 0.001 | 0.000 |
| recorded | project:x | 1955 | 1426 | 0.898 | 0.578 | 0.011 | 0.000 |
| live_weights | _other | 86 | 41 | 0.976 | 0.728 | 0.000 | 0.009 |
| live_weights | global | 115 | 79 | 0.975 | 0.847 | 0.000 | 0.018 |
| live_weights | project:ae | 1252 | 384 | 0.859 | 0.576 | 0.009 | 0.002 |
| live_weights | project:lm | 254 | 114 | 0.965 | 0.787 | 0.000 | 0.005 |
| live_weights | project:mm | 63 | 21 | 0.952 | 0.933 | 0.000 | 0.000 |
| live_weights | project:octopus | 1546 | 779 | 0.915 | 0.597 | 0.008 | 0.010 |
| live_weights | project:online | 2318 | 760 | 0.862 | 0.650 | 0.001 | 0.001 |
| live_weights | project:x | 1955 | 1426 | 0.972 | 0.734 | 0.011 | 0.005 |
| floor_defaults | _other | 86 | 41 | 0.976 | 0.707 | 0.000 | 0.019 |
| floor_defaults | global | 115 | 79 | 0.975 | 0.861 | 0.000 | 0.036 |
| floor_defaults | project:ae | 1252 | 384 | 0.875 | 0.555 | 0.009 | 0.017 |
| floor_defaults | project:lm | 254 | 114 | 0.965 | 0.745 | 0.000 | 0.000 |
| floor_defaults | project:mm | 63 | 21 | 0.952 | 0.910 | 0.000 | 0.026 |
| floor_defaults | project:octopus | 1546 | 779 | 0.935 | 0.598 | 0.008 | 0.016 |
| floor_defaults | project:online | 2318 | 760 | 0.868 | 0.625 | 0.001 | 0.008 |
| floor_defaults | project:x | 1955 | 1426 | 0.968 | 0.739 | 0.011 | 0.014 |
| uniform | _other | 86 | 41 | 0.976 | 0.714 | 0.000 | 0.019 |
| uniform | global | 115 | 79 | 0.975 | 0.855 | 0.000 | 0.054 |
| uniform | project:ae | 1252 | 384 | 0.885 | 0.571 | 0.009 | 0.036 |
| uniform | project:lm | 254 | 114 | 0.965 | 0.757 | 0.000 | 0.009 |
| uniform | project:mm | 63 | 21 | 0.952 | 0.910 | 0.000 | 0.026 |
| uniform | project:octopus | 1546 | 779 | 0.928 | 0.596 | 0.008 | 0.029 |
| uniform | project:online | 2318 | 760 | 0.875 | 0.629 | 0.001 | 0.018 |
| uniform | project:x | 1955 | 1426 | 0.966 | 0.731 | 0.011 | 0.021 |
| replayed_winner_take_all | _other | 86 | 41 | 0.976 | 0.707 | 0.000 | 0.028 |
| replayed_winner_take_all | global | 115 | 79 | 0.975 | 0.858 | 0.000 | 0.023 |
| replayed_winner_take_all | project:ae | 1252 | 384 | 0.862 | 0.579 | 0.009 | 0.004 |
| replayed_winner_take_all | project:lm | 254 | 114 | 0.974 | 0.766 | 0.000 | 0.005 |
| replayed_winner_take_all | project:mm | 63 | 21 | 0.952 | 0.933 | 0.000 | 0.000 |
| replayed_winner_take_all | project:octopus | 1546 | 779 | 0.920 | 0.599 | 0.008 | 0.010 |
| replayed_winner_take_all | project:online | 2318 | 760 | 0.858 | 0.636 | 0.001 | 0.001 |
| replayed_winner_take_all | project:x | 1955 | 1426 | 0.972 | 0.734 | 0.011 | 0.005 |

## Label sensitivity (live_weights scheme, overall)

| protocol | strict-subset | all-useful | none-useful | events w/useful | hit@5 | MRR |
|---|---|---|---|---|---|---|
| grounded | 46.7% | 0.9% | 52.4% | 3604 | 0.924 | 0.675 |
| reconsumed | 36.3% | 63.1% | 0.7% | 7535 | 0.994 | 0.918 |
| usefulness | 78.3% | 16.6% | 5.2% | 7187 | 0.928 | 0.620 |
| grounded_or_reconsumed | 27.2% | 72.6% | 0.2% | 7568 | 0.996 | 0.950 |

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

