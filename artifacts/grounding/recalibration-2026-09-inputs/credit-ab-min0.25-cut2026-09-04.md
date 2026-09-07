# Credit-rule A/B: grounded vs reinforce-everything

DB: `/home/sfx/.cache/living-memory-harness/usage-signal/sfx-2026-09-07.sqlite3` | cutoff `2026-09-04T00:00:00Z` | 62474 usable events

- Grounding at min_containment 0.25: 9267 of 83504 consumed results grounded (0.1110); per-event vs whole-corpus IDF agreement 0.9616
- Incumbent `proportional` reinforces **every** result; the grounded arms reinforce that 11.1% share only
- `grounded_negative` blames ungrounded results at 0.25 of the positive signal

## Holdout metrics by label and arm

| label | arm | events w/useful | hit@1 | hit@5 | hit@10 | MRR | weight updates |
|---|---|---|---|---|---|---|---|
| grounded | proportional | 99 | 0.4646 | 0.8687 | 1.0000 | 0.6268 | 80738 |
| grounded | grounded | 99 | 0.4949 | 0.8788 | 1.0000 | 0.6538 | 9159 |
| grounded | grounded_negative | 99 | 0.3535 | 0.8384 | 1.0000 | 0.5478 | 80738 |
| reconsumed | proportional | 327 | 0.5046 | 0.9297 | 1.0000 | 0.6795 | 80738 |
| reconsumed | grounded | 327 | 0.4954 | 0.9358 | 1.0000 | 0.6725 | 9159 |
| reconsumed | grounded_negative | 327 | 0.4893 | 0.9235 | 1.0000 | 0.6730 | 80738 |
| usefulness | proportional | 390 | 0.2154 | 0.8744 | 1.0000 | 0.4683 | 80738 |
| usefulness | grounded | 390 | 0.1667 | 0.8667 | 1.0000 | 0.4250 | 9159 |
| usefulness | grounded_negative | 390 | 0.2949 | 0.9051 | 1.0000 | 0.5324 | 80738 |

## Learned-weight sanity (primary label, final trajectory)

| arm | scopes | mean graph | max graph | scopes with graph > 0.5 |
|---|---|---|---|---|
| proportional | 67 | 0.0536 | 0.5463 | 1 ['project:ae'] |
| grounded | 41 | 0.0329 | 0.1132 | 0  |
| grounded_negative | 67 | 0.0879 | 0.7500 | 5 ['global', 'project:ae', 'project:lm', 'project:mm', 'project:online'] |

## Verdict

Winner: **proportional** (incumbent `proportional`).

Deltas are holdout arm-minus-incumbent with a paired bootstrap 95% CI (2000 resamples over events, seed 20260818). A gap only counts when its CI excludes zero.

| arm | label | metric | incumbent | arm | delta | CI95 | verdict |
|---|---|---|---|---|---|---|---|
| grounded | grounded | hit@5 | 0.868687 | 0.878788 | +0.010101 | [+0.000000, +0.030303] | indistinguishable |
| grounded | grounded | mrr | 0.626756 | 0.653764 | +0.027008 | [-0.004040, +0.061111] | indistinguishable |
| grounded | reconsumed | hit@5 | 0.929664 | 0.935780 | +0.006116 | [-0.012232, +0.024465] | indistinguishable |
| grounded | reconsumed | mrr | 0.679495 | 0.672453 | -0.007042 | [-0.028005, +0.012980] | indistinguishable |
| grounded | usefulness | hit@5 | 0.874359 | 0.866667 | -0.007692 | [-0.023077, +0.007692] | indistinguishable |
| grounded | usefulness | mrr | 0.468346 | 0.425037 | -0.043309 | [-0.060675, -0.025641] | REGRESSION |
| grounded_negative | grounded | hit@5 | 0.868687 | 0.838384 | -0.030303 | [-0.080808, +0.010101] | indistinguishable |
| grounded_negative | grounded | mrr | 0.626756 | 0.547787 | -0.078969 | [-0.127525, -0.038324] | REGRESSION |
| grounded_negative | reconsumed | hit@5 | 0.929664 | 0.923547 | -0.006117 | [-0.024465, +0.012232] | indistinguishable |
| grounded_negative | reconsumed | mrr | 0.679495 | 0.672966 | -0.006529 | [-0.034893, +0.019873] | indistinguishable |
| grounded_negative | usefulness | hit@5 | 0.874359 | 0.905128 | +0.030769 | [+0.007692, +0.053846] | gain |
| grounded_negative | usefulness | mrr | 0.468346 | 0.532406 | +0.064060 | [+0.044085, +0.084783] | gain |

- `grounded`: degenerate weights = False, significant regressions = ['usefulness.mrr'], significant primary gains = none -> reject
- `grounded_negative`: degenerate weights = True, significant regressions = ['grounded.mrr'], significant primary gains = none -> reject

`grounded` and the primary label share the content-grounding notion, so the `reconsumed` and `usefulness` rows are the falsifiability control: they label usefulness without grounding. Both controls are known to be hub-node biased (the replay module documents `reconsumed` as nearly vacuous and `usefulness` as rank-circular), which is why a weight-sanity check that does not go through any label is applied first.
