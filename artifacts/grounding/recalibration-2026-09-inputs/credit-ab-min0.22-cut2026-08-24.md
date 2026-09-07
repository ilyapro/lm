# Credit-rule A/B: grounded vs reinforce-everything

DB: `/home/sfx/.cache/living-memory-harness/usage-signal/sfx-2026-09-07.sqlite3` | cutoff `2026-08-24T00:00:00Z` | 62474 usable events

- Grounding at min_containment 0.22: 12295 of 83504 consumed results grounded (0.1472); per-event vs whole-corpus IDF agreement 0.9558
- Incumbent `proportional` reinforces **every** result; the grounded arms reinforce that 14.7% share only
- `grounded_negative` blames ungrounded results at 0.25 of the positive signal

## Holdout metrics by label and arm

| label | arm | events w/useful | hit@1 | hit@5 | hit@10 | MRR | weight updates |
|---|---|---|---|---|---|---|---|
| grounded | proportional | 352 | 0.5170 | 0.9460 | 1.0000 | 0.6983 | 76516 |
| grounded | grounded | 352 | 0.5114 | 0.9460 | 1.0000 | 0.6964 | 11842 |
| grounded | grounded_negative | 352 | 0.4432 | 0.9290 | 1.0000 | 0.6476 | 76516 |
| reconsumed | proportional | 1093 | 0.6240 | 0.9735 | 0.9991 | 0.7674 | 76516 |
| reconsumed | grounded | 1093 | 0.6304 | 0.9762 | 0.9991 | 0.7708 | 11842 |
| reconsumed | grounded_negative | 1093 | 0.6386 | 0.9698 | 0.9991 | 0.7793 | 76516 |
| usefulness | proportional | 1087 | 0.2447 | 0.9163 | 0.9936 | 0.4990 | 76516 |
| usefulness | grounded | 1087 | 0.2392 | 0.9098 | 0.9936 | 0.4914 | 11842 |
| usefulness | grounded_negative | 1087 | 0.2971 | 0.9310 | 0.9936 | 0.5386 | 76516 |

## Learned-weight sanity (primary label, final trajectory)

| arm | scopes | mean graph | max graph | scopes with graph > 0.5 |
|---|---|---|---|---|
| proportional | 66 | 0.0497 | 0.2685 | 0  |
| grounded | 42 | 0.0338 | 0.1364 | 0  |
| grounded_negative | 66 | 0.0714 | 0.7500 | 4 ['global', 'project:lm', 'project:mm', 'project:online'] |

## Verdict

Winner: **proportional** (incumbent `proportional`).

Deltas are holdout arm-minus-incumbent with a paired bootstrap 95% CI (2000 resamples over events, seed 20260818). A gap only counts when its CI excludes zero.

| arm | label | metric | incumbent | arm | delta | CI95 | verdict |
|---|---|---|---|---|---|---|---|
| grounded | grounded | hit@5 | 0.946023 | 0.946023 | +0.000000 | [-0.008523, +0.008523] | indistinguishable |
| grounded | grounded | mrr | 0.698275 | 0.696449 | -0.001826 | [-0.012547, +0.008022] | indistinguishable |
| grounded | reconsumed | hit@5 | 0.973468 | 0.976212 | +0.002744 | [-0.003660, +0.009149] | indistinguishable |
| grounded | reconsumed | mrr | 0.767372 | 0.770811 | +0.003439 | [-0.003620, +0.010659] | indistinguishable |
| grounded | usefulness | hit@5 | 0.916283 | 0.909844 | -0.006439 | [-0.013799, +0.000920] | indistinguishable |
| grounded | usefulness | mrr | 0.498994 | 0.491425 | -0.007569 | [-0.013810, -0.001276] | REGRESSION |
| grounded_negative | grounded | hit@5 | 0.946023 | 0.928977 | -0.017046 | [-0.031250, -0.005682] | REGRESSION |
| grounded_negative | grounded | mrr | 0.698275 | 0.647643 | -0.050632 | [-0.073268, -0.029498] | REGRESSION |
| grounded_negative | reconsumed | hit@5 | 0.973468 | 0.969808 | -0.003660 | [-0.010064, +0.002745] | indistinguishable |
| grounded_negative | reconsumed | mrr | 0.767372 | 0.779280 | +0.011908 | [+0.001507, +0.022818] | gain |
| grounded_negative | usefulness | hit@5 | 0.916283 | 0.931003 | +0.014720 | [+0.004600, +0.025759] | gain |
| grounded_negative | usefulness | mrr | 0.498994 | 0.538590 | +0.039596 | [+0.028742, +0.050874] | gain |

- `grounded`: degenerate weights = False, significant regressions = ['usefulness.mrr'], significant primary gains = none -> reject
- `grounded_negative`: degenerate weights = True, significant regressions = ['grounded.hit@5', 'grounded.mrr'], significant primary gains = none -> reject

`grounded` and the primary label share the content-grounding notion, so the `reconsumed` and `usefulness` rows are the falsifiability control: they label usefulness without grounding. Both controls are known to be hub-node biased (the replay module documents `reconsumed` as nearly vacuous and `usefulness` as rank-circular), which is why a weight-sanity check that does not go through any label is applied first.
