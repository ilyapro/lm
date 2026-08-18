# Credit-rule A/B: grounded vs reinforce-everything

DB: `/home/sfx/.cache/living-memory-harness/snapshot.sqlite3` | cutoff `2026-06-10T00:00:00Z` | 54757 usable events

- Grounding at min_containment 0.25: 8698 of 72023 consumed results grounded (0.1208); per-event vs whole-corpus IDF agreement 0.9616
- Incumbent `proportional` reinforces **every** result; the grounded arms reinforce that 12.1% share only
- `grounded_negative` blames ungrounded results at 0.25 of the positive signal

## Holdout metrics by label and arm

| label | arm | events w/useful | hit@1 | hit@5 | hit@10 | MRR | weight updates |
|---|---|---|---|---|---|---|---|
| grounded | proportional | 2015 | 0.5211 | 0.9300 | 0.9945 | 0.6921 | 42348 |
| grounded | grounded | 2015 | 0.5325 | 0.9300 | 0.9940 | 0.7000 | 3842 |
| grounded | grounded_negative | 2015 | 0.5161 | 0.9280 | 0.9935 | 0.6870 | 42348 |
| reconsumed | proportional | 3638 | 0.7526 | 0.9827 | 0.9934 | 0.8489 | 42348 |
| reconsumed | grounded | 3638 | 0.7479 | 0.9813 | 0.9934 | 0.8464 | 3842 |
| reconsumed | grounded_negative | 3638 | 0.7688 | 0.9832 | 0.9934 | 0.8600 | 42348 |
| usefulness | proportional | 3381 | 0.2694 | 0.8814 | 0.9799 | 0.5020 | 42348 |
| usefulness | grounded | 3381 | 0.2650 | 0.8802 | 0.9799 | 0.5000 | 3842 |
| usefulness | grounded_negative | 3381 | 0.2996 | 0.8974 | 0.9805 | 0.5317 | 42348 |

## Learned-weight sanity (primary label, final trajectory)

| arm | scopes | mean graph | max graph | scopes with graph > 0.5 |
|---|---|---|---|---|
| proportional | 40 | 0.0509 | 0.2010 | 0  |
| grounded | 28 | 0.0195 | 0.1041 | 0  |
| grounded_negative | 40 | 0.0677 | 0.7500 | 3 ['project:ae', 'project:mm', 'project:online'] |

## Verdict

Winner: **grounded** (incumbent `proportional`).

Deltas are holdout arm-minus-incumbent with a paired bootstrap 95% CI (2000 resamples over events, seed 20260818). A gap only counts when its CI excludes zero.

| arm | label | metric | incumbent | arm | delta | CI95 | verdict |
|---|---|---|---|---|---|---|---|
| grounded | grounded | hit@5 | 0.930025 | 0.930025 | +0.000000 | [-0.001985, +0.002481] | indistinguishable |
| grounded | grounded | mrr | 0.692065 | 0.699981 | +0.007916 | [+0.003371, +0.012607] | gain |
| grounded | reconsumed | hit@5 | 0.982683 | 0.981308 | -0.001375 | [-0.003024, +0.000000] | indistinguishable |
| grounded | reconsumed | mrr | 0.848874 | 0.846360 | -0.002514 | [-0.005021, +0.000046] | indistinguishable |
| grounded | usefulness | hit@5 | 0.881396 | 0.880213 | -0.001183 | [-0.004141, +0.001775] | indistinguishable |
| grounded | usefulness | mrr | 0.502025 | 0.500038 | -0.001987 | [-0.004888, +0.000986] | indistinguishable |
| grounded_negative | grounded | hit@5 | 0.930025 | 0.928040 | -0.001985 | [-0.008933, +0.005459] | indistinguishable |
| grounded_negative | grounded | mrr | 0.692065 | 0.687009 | -0.005056 | [-0.012378, +0.001761] | indistinguishable |
| grounded_negative | reconsumed | hit@5 | 0.982683 | 0.983233 | +0.000550 | [-0.001924, +0.003024] | indistinguishable |
| grounded_negative | reconsumed | mrr | 0.848874 | 0.859992 | +0.011118 | [+0.007296, +0.014989] | gain |
| grounded_negative | usefulness | hit@5 | 0.881396 | 0.897368 | +0.015972 | [+0.010944, +0.021295] | gain |
| grounded_negative | usefulness | mrr | 0.502025 | 0.531737 | +0.029712 | [+0.024110, +0.035808] | gain |

- `grounded`: degenerate weights = False, significant regressions = none, significant primary gains = ['mrr'] -> ADOPT
- `grounded_negative`: degenerate weights = True, significant regressions = none, significant primary gains = none -> reject

`grounded` and the primary label share the content-grounding notion, so the `reconsumed` and `usefulness` rows are the falsifiability control: they label usefulness without grounding. Both controls are known to be hub-node biased (the replay module documents `reconsumed` as nearly vacuous and `usefulness` as rank-circular), which is why a weight-sanity check that does not go through any label is applied first.
