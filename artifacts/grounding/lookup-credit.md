# Lookup credit: `grounded_or_lookup` vs incumbent `grounded` (sfx snapshot 2026-09-07, cutoff 2026-09-04)

DB: `/home/sfx/.cache/living-memory-harness/usage-signal/sfx-2026-09-07.sqlite3` | cutoff `2026-09-04T00:00:00Z` | 62474 usable events | lookup window 86400 s same-transport

## Criterion (stated before the numbers)

LM_LOOKUP_CREDIT_POLICY=delivered stays the default if the grounded_or_lookup arm shows no significant regression against the grounded incumbent under any label (grounded, reconsumed, usefulness; holdout hit@5 and MRR, paired bootstrap 95% CI must exclude zero to count) and its learned weights are non-degenerate (no scope with graph share above 0.5). A significant primary gain is NOT required: the case for lookup credit is coverage (closures that credit anything), which the A/B cannot reward because holdout events are never reinforced in replay. The script's own winner field additionally demands a significant primary gain; where it disagrees, both are printed and the coverage criterion decides.

## Coverage before/after (closures that credit at least one node)

Source: `scripts/usage_signal_replay.py` over the closure corpora at the checkout's DEFAULT_MIN_CONTAINMENT = 0.25. "Before" is grounded-only credit (the live `LM_RECALL_CREDIT_POLICY=grounded` gate); "after" is grounded-or-looked-up. A pair is a (delivered result, closing trace) pair; the overlap column is the pairs both signals fire on, which the live credit ledger the live-path change adds (`recall_credit_ledger`, one row per (event, node)) credits once, so lookup credit adds exactly the lookup-only pairs.

| host | closures | before (grounded) | after (grounded or lookup) | lookup closures | lookup pairs | overlap pairs (dedup removes) | lookup-only pairs |
|---|---|---|---|---|---|---|---|
| sfx | 421 | 60 (14.2%) | 139 (33.0%) | 102 | 168/2959 | 22 | 146 |
| alt | 423 | 100 (23.6%) | 184 (43.5%) | 119 | 227/3847 | 36 | 191 |
| pooled | 844 | 160 (19.0%) | 323 (38.3%) | 221 | 395/6806 | 58 | 337 |

The JSON artifact's top-level `before` and `after` blocks mirror this table, one entry per host (sfx, alt, pooled) derived from its `coverage` block without recomputation: `before` is grounded-only credit at min_containment 0.25 on the master tokenizer (the live gate at the time), `after` is grounded or same-transport lookup within 86400 s, deduplicated per (event, node).

## Holdout A/B

- Grounding at min_containment 0.25: 9036 of 83504 consumed results grounded (0.1082) over the whole event history; per-event vs whole-corpus IDF agreement 0.9634
- Incumbent `grounded` reinforces only grounded results; `grounded_or_lookup` additionally reinforces results a same-transport `memory_lookup` fetched within 86400 s of delivery and not after the cutoff
- Lookup ledger: 548 rows in 296 lookup events, first at 2026-08-23T22:55:05Z, last at 2026-09-06T21:50:54Z. Train side (follows not after the cutoff): 145 results followed across all events (8 also grounded, 137 lookup-only); within consumed events, the only ones a credit rule reinforces, 50 events carry a follow and 70 results are followed, 62 of them lookup-only; 173 results whose only in-window follow fell after the cutoff were withheld from the trajectory

| label | arm | events w/useful | hit@1 | hit@5 | hit@10 | MRR | weight updates |
|---|---|---|---|---|---|---|---|
| grounded | grounded | 78 | 0.5000 | 0.8718 | 1.0000 | 0.6544 | 8949 |
| grounded | grounded_or_lookup | 78 | 0.5128 | 0.8718 | 1.0000 | 0.6609 | 9011 |
| reconsumed | grounded | 327 | 0.4893 | 0.9358 | 1.0000 | 0.6694 | 8949 |
| reconsumed | grounded_or_lookup | 327 | 0.4832 | 0.9358 | 1.0000 | 0.6656 | 9011 |
| usefulness | grounded | 390 | 0.1667 | 0.8692 | 1.0000 | 0.4249 | 8949 |
| usefulness | grounded_or_lookup | 390 | 0.1692 | 0.8692 | 1.0000 | 0.4256 | 9011 |

### Learned-weight sanity (primary label, final trajectory)

| arm | scopes | mean graph | max graph | scopes with graph > 0.5 |
|---|---|---|---|---|
| grounded | 41 | 0.0323 | 0.1178 | 0  |
| grounded_or_lookup | 41 | 0.0328 | 0.1107 | 0  |

### Paired deltas (holdout, arm minus incumbent)

Paired bootstrap 95% CI (2000 resamples over events, seed 20260818). A gap only counts when its CI excludes zero.

| arm | label | metric | incumbent | arm | delta | CI95 | verdict |
|---|---|---|---|---|---|---|---|
| grounded_or_lookup | grounded | hit@5 | 0.871795 | 0.871795 | +0.000000 | [+0.000000, +0.000000] | indistinguishable |
| grounded_or_lookup | grounded | mrr | 0.654441 | 0.660852 | +0.006411 | [+0.000000, +0.019231] | indistinguishable |
| grounded_or_lookup | reconsumed | hit@5 | 0.935780 | 0.935780 | +0.000000 | [+0.000000, +0.000000] | indistinguishable |
| grounded_or_lookup | reconsumed | mrr | 0.669395 | 0.665572 | -0.003823 | [-0.010194, +0.001529] | indistinguishable |
| grounded_or_lookup | usefulness | hit@5 | 0.869231 | 0.869231 | +0.000000 | [+0.000000, +0.000000] | indistinguishable |
| grounded_or_lookup | usefulness | mrr | 0.424908 | 0.425635 | +0.000727 | [-0.001068, +0.003632] | indistinguishable |

- `grounded_or_lookup`: degenerate weights = False, significant regressions = none, significant primary gains = none
- Script winner field: **grounded** (it requires a significant primary gain, which the criterion above does not); criterion above: **PASS**

## Verdict

Keep LM_LOOKUP_CREDIT_POLICY=delivered as the default: on the sfx holdout the grounded_or_lookup arm is non-degenerate and shows no significant regression under any label (grounded MRR delta +0.006411 [+0.000000, +0.019231], hit@5 +0.000000 [+0.000000, +0.000000]), while lookup credit lifts closure coverage from 160/844 to 323/844 pooled (60/421 to 139/421 on sfx).

The script's winner stays `grounded` because its rule 3 demands a significant gain under the primary label and the lookup arm is indistinguishable there. That is expected, not a contradiction: on this holdout, lookup credit changes the trajectory through the train-side follows only, and the primary metric is computed on holdout events whose own follows are never replayed. The decision the criterion asks is "does the extra credit hurt", and it does not.

## Holdout is small

The holdout holds 421 labeled (closed) events after the cutoff, of which 102 carry a same-transport lookup follow (168 results). Under the grounded label only 78 holdout events have any useful result, so the primary-label CIs are wide. Lookups exist only since 2026-08-23T22:55:05Z (the ledger's first row), so train-side lookup credit covers 12 days: 50 closed events and 70 results, 62 of them lookup-only (8 were already grounded), against 8949 grounded weight updates over the whole history. The arm therefore differs from the incumbent by 62 weight updates in total. A null result here means "too small to hurt measurably", not "proven harmless at scale"; re-run once the ledger spans months.

## Timing limitation

The replay credits at closure time: `grounded_or_lookup` applies the lookup credit at the event's reinforcement instant (`feedback_applied_at`, when the closing trace lands), not at lookup time as the live `apply_lookup_credit` path does. Unclosed events (no `feedback_trace_id`) are never reinforced in replay even though the live path credits their lookups, so the A/B measures lookup credit on the closed-event stream only and understates the live signal's volume. The order of credit within a closed event is also collapsed: live, a lookup that precedes the closure claims the ledger row first and the grounded pass skips it; in replay both collapse to one proportional credit per result, which is the same weight update.

## Reproduce

```
python3 scripts/usage_signal_replay.py --corpus ~/.cache/living-memory-harness/usage-signal/sfx-closures.jsonl --report /tmp/lc-sfx.json
python3 scripts/usage_signal_replay.py --corpus ~/.cache/living-memory-harness/usage-signal/alt-closures.jsonl --report /tmp/lc-alt.json
python3 scripts/usage_signal_replay.py --corpus ~/.cache/living-memory-harness/usage-signal/sfx-closures.jsonl --corpus ~/.cache/living-memory-harness/usage-signal/alt-closures.jsonl --report /tmp/lc-pooled.json
timeout 5400 python3 scripts/credit_rule_ab.py --db ~/.cache/living-memory-harness/usage-signal/sfx-2026-09-07.sqlite3 --cutoff 2026-09-04T00:00:00Z --incumbent grounded --arms grounded,grounded_or_lookup --report /tmp/lc-ab.json
bash scripts/test.sh tests/test_replay_harness.py -q
```

Snapshots and corpora are opened read-only (`replay.open_readonly`); the artifact JSON carries the full A/B report under `ab`, the per-host coverage under `coverage`, and the holdout counts under `holdout`.
