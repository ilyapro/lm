# Credit-rule A/B: grounded vs reinforce-everything

DB: `/home/sfx/.cache/living-memory-harness/usage-signal/sfx-2026-09-07.sqlite3` | cutoff `2026-08-24T00:00:00Z` | 62474 usable events

- Grounding at min_containment 0.25: 9267 of 83504 consumed results grounded (0.1110); per-event vs whole-corpus IDF agreement 0.9616
- Incumbent `proportional` reinforces **every** result; the grounded arms reinforce that 11.1% share only
- `grounded_negative` blames ungrounded results at 0.25 of the positive signal

## Holdout metrics by label and arm

| label | arm | events w/useful | hit@1 | hit@5 | hit@10 | MRR | weight updates |
|---|---|---|---|---|---|---|---|
| grounded | proportional | 274 | 0.5037 | 0.9380 | 1.0000 | 0.6851 | 76516 |
| grounded | grounded | 274 | 0.5000 | 0.9489 | 1.0000 | 0.6866 | 8984 |
| grounded | grounded_negative | 274 | 0.3942 | 0.9197 | 1.0000 | 0.6108 | 76516 |
| reconsumed | proportional | 1093 | 0.6240 | 0.9735 | 0.9991 | 0.7674 | 76516 |
| reconsumed | grounded | 1093 | 0.6258 | 0.9762 | 0.9991 | 0.7683 | 8984 |
| reconsumed | grounded_negative | 1093 | 0.6642 | 0.9716 | 0.9991 | 0.7945 | 76516 |
| usefulness | proportional | 1087 | 0.2447 | 0.9163 | 0.9936 | 0.4990 | 76516 |
| usefulness | grounded | 1087 | 0.2392 | 0.9108 | 0.9936 | 0.4911 | 8984 |
| usefulness | grounded_negative | 1087 | 0.3395 | 0.9356 | 0.9936 | 0.5681 | 76516 |

## Learned-weight sanity (primary label, final trajectory)

| arm | scopes | mean graph | max graph | scopes with graph > 0.5 |
|---|---|---|---|---|
| proportional | 66 | 0.0497 | 0.2685 | 0  |
| grounded | 40 | 0.0314 | 0.1059 | 0  |
| grounded_negative | 66 | 0.0816 | 0.7500 | 5 ['global', 'project:ae', 'project:lm', 'project:mm', 'project:online'] |

## Verdict

Winner: **proportional** (incumbent `proportional`).

Deltas are holdout arm-minus-incumbent with a paired bootstrap 95% CI (2000 resamples over events, seed 20260818). A gap only counts when its CI excludes zero.

| arm | label | metric | incumbent | arm | delta | CI95 | verdict |
|---|---|---|---|---|---|---|---|
| grounded | grounded | hit@5 | 0.937956 | 0.948905 | +0.010949 | [+0.000000, +0.025547] | indistinguishable |
| grounded | grounded | mrr | 0.685089 | 0.686553 | +0.001464 | [-0.012530, +0.014942] | indistinguishable |
| grounded | reconsumed | hit@5 | 0.973468 | 0.976212 | +0.002744 | [-0.002745, +0.009149] | indistinguishable |
| grounded | reconsumed | mrr | 0.767372 | 0.768325 | +0.000953 | [-0.006054, +0.007989] | indistinguishable |
| grounded | usefulness | hit@5 | 0.916283 | 0.910764 | -0.005519 | [-0.012879, +0.001840] | indistinguishable |
| grounded | usefulness | mrr | 0.498994 | 0.491101 | -0.007893 | [-0.013796, -0.002062] | REGRESSION |
| grounded_negative | grounded | hit@5 | 0.937956 | 0.919708 | -0.018248 | [-0.040146, +0.003650] | indistinguishable |
| grounded_negative | grounded | mrr | 0.685089 | 0.610801 | -0.074288 | [-0.103185, -0.048219] | REGRESSION |
| grounded_negative | reconsumed | hit@5 | 0.973468 | 0.971638 | -0.001830 | [-0.009149, +0.005489] | indistinguishable |
| grounded_negative | reconsumed | mrr | 0.767372 | 0.794489 | +0.027117 | [+0.015991, +0.038768] | gain |
| grounded_negative | usefulness | hit@5 | 0.916283 | 0.935603 | +0.019320 | [+0.009200, +0.030359] | gain |
| grounded_negative | usefulness | mrr | 0.498994 | 0.568094 | +0.069100 | [+0.057374, +0.081596] | gain |

- `grounded`: degenerate weights = False, significant regressions = ['usefulness.mrr'], significant primary gains = none -> reject
- `grounded_negative`: degenerate weights = True, significant regressions = ['grounded.mrr'], significant primary gains = none -> reject

`grounded` and the primary label share the content-grounding notion, so the `reconsumed` and `usefulness` rows are the falsifiability control: they label usefulness without grounding. Both controls are known to be hub-node biased (the replay module documents `reconsumed` as nearly vacuous and `usefulness` as rank-circular), which is why a weight-sanity check that does not go through any label is applied first.
