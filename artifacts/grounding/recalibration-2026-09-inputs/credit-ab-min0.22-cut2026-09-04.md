# Credit-rule A/B: grounded vs reinforce-everything

DB: `/home/sfx/.cache/living-memory-harness/usage-signal/sfx-2026-09-07.sqlite3` | cutoff `2026-09-04T00:00:00Z` | 62474 usable events

- Grounding at min_containment 0.22: 12295 of 83504 consumed results grounded (0.1472); per-event vs whole-corpus IDF agreement 0.9558
- Incumbent `proportional` reinforces **every** result; the grounded arms reinforce that 14.7% share only
- `grounded_negative` blames ungrounded results at 0.25 of the positive signal

## Holdout metrics by label and arm

| label | arm | events w/useful | hit@1 | hit@5 | hit@10 | MRR | weight updates |
|---|---|---|---|---|---|---|---|
| grounded | proportional | 123 | 0.4878 | 0.9024 | 1.0000 | 0.6612 | 80738 |
| grounded | grounded | 123 | 0.5366 | 0.9106 | 1.0000 | 0.6994 | 12131 |
| grounded | grounded_negative | 123 | 0.4228 | 0.8780 | 1.0000 | 0.6240 | 80738 |
| reconsumed | proportional | 327 | 0.5046 | 0.9297 | 1.0000 | 0.6795 | 80738 |
| reconsumed | grounded | 327 | 0.4954 | 0.9358 | 1.0000 | 0.6729 | 12131 |
| reconsumed | grounded_negative | 327 | 0.4495 | 0.9205 | 1.0000 | 0.6494 | 80738 |
| usefulness | proportional | 390 | 0.2154 | 0.8744 | 1.0000 | 0.4683 | 80738 |
| usefulness | grounded | 390 | 0.1667 | 0.8744 | 1.0000 | 0.4255 | 12131 |
| usefulness | grounded_negative | 390 | 0.2282 | 0.8974 | 1.0000 | 0.4853 | 80738 |

## Learned-weight sanity (primary label, final trajectory)

| arm | scopes | mean graph | max graph | scopes with graph > 0.5 |
|---|---|---|---|---|
| proportional | 67 | 0.0536 | 0.5463 | 1 ['project:ae'] |
| grounded | 43 | 0.0354 | 0.1364 | 0  |
| grounded_negative | 67 | 0.0763 | 0.7500 | 4 ['global', 'project:lm', 'project:mm', 'project:online'] |

## Verdict

Winner: **proportional** (incumbent `proportional`).

Deltas are holdout arm-minus-incumbent with a paired bootstrap 95% CI (2000 resamples over events, seed 20260818). A gap only counts when its CI excludes zero.

| arm | label | metric | incumbent | arm | delta | CI95 | verdict |
|---|---|---|---|---|---|---|---|
| grounded | grounded | hit@5 | 0.902439 | 0.910569 | +0.008130 | [+0.000000, +0.024390] | indistinguishable |
| grounded | grounded | mrr | 0.661150 | 0.699361 | +0.038211 | [+0.005556, +0.074255] | gain |
| grounded | reconsumed | hit@5 | 0.929664 | 0.935780 | +0.006116 | [-0.012232, +0.024465] | indistinguishable |
| grounded | reconsumed | mrr | 0.679495 | 0.672861 | -0.006634 | [-0.027829, +0.013379] | indistinguishable |
| grounded | usefulness | hit@5 | 0.874359 | 0.874359 | +0.000000 | [-0.015385, +0.015385] | indistinguishable |
| grounded | usefulness | mrr | 0.468346 | 0.425507 | -0.042839 | [-0.060192, -0.024890] | REGRESSION |
| grounded_negative | grounded | hit@5 | 0.902439 | 0.878049 | -0.024390 | [-0.056911, +0.008130] | indistinguishable |
| grounded_negative | grounded | mrr | 0.661150 | 0.623955 | -0.037195 | [-0.088008, +0.011179] | indistinguishable |
| grounded_negative | reconsumed | hit@5 | 0.929664 | 0.920489 | -0.009175 | [-0.027523, +0.009174] | indistinguishable |
| grounded_negative | reconsumed | mrr | 0.679495 | 0.649368 | -0.030127 | [-0.061109, -0.003025] | REGRESSION |
| grounded_negative | usefulness | hit@5 | 0.874359 | 0.897436 | +0.023077 | [+0.000000, +0.046154] | indistinguishable |
| grounded_negative | usefulness | mrr | 0.468346 | 0.485251 | +0.016905 | [-0.006396, +0.041430] | indistinguishable |

- `grounded`: degenerate weights = False, significant regressions = ['usefulness.mrr'], significant primary gains = ['mrr'] -> reject
- `grounded_negative`: degenerate weights = True, significant regressions = ['reconsumed.mrr'], significant primary gains = none -> reject

`grounded` and the primary label share the content-grounding notion, so the `reconsumed` and `usefulness` rows are the falsifiability control: they label usefulness without grounding. Both controls are known to be hub-node biased (the replay module documents `reconsumed` as nearly vacuous and `usefulness` as rank-circular), which is why a weight-sanity check that does not go through any label is applied first.
