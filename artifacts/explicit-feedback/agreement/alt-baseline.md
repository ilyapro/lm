# Explicit-feedback agreement check — alt

Store: `/tmp/lm-alt-agreement-20260927.sqlite3` (read-only), since `2026-09-07`, generated 2026-09-27T11:13:54Z.
Script: `scripts/explicit_feedback_agreement.py`. Placebo method: LM memory 01M3H76JBHEZKJMBPWFVN70SB2 (twin = undelivered node created before the event, |cos(twin,T) − cos(n,T)| ≤ 0.01, mean-pooled chunk vectors; grounding `ground_token_sets` on delivered + twin, threshold 0.22).
A/B exclusion: whole transport sessions with an A/B marker (scope project:target|repo|x, tree-context-ab/fixture/project=target in node context, or A/B-like recall query).

## Counts

| item | value |
|---|---|
| events | 3170 |
| closed_events | 2754 |
| closed_scored | 2752 |
| delivered_without_twin | 359 |
| closed_without_trace_vector | 2 |
| excluded_ab_events | 1505 |
| ab_sessions | 1027 |
| delivered_pairs | 13344 |

## Placebo reference rates (grounded vs same-cosine twin)

| slice | pairs | grounded | twin | excess | excess 95% CI | grounded −q | twin −q | excess −q | excess −q 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| r1 | 2593 | 13.9% | 2.6% | 11.3% | [+10.1, +12.7] pp | 5.4% | 1.7% | 3.7% | [+2.8, +4.6] pp |
| top3 | 7151 | 11.1% | 2.2% | 8.9% | [+8.2, +9.7] pp | 4.5% | 1.3% | 3.2% | [+2.7, +3.7] pp |

`−q`: query tokens removed from the closing trace before grading.

Lookup rate (same transport, after delivery, or ledger basis lookup): r1 2.9%, top-3 4.7%. No placebo exists for lookup (a twin is never on the card).

r1 by cos(r1, closing trace):

| cos bin | pairs | grounded | twin | excess |
|---|---|---|---|---|
| <0.5 | 391 | 0.5% | 0.0% | 0.5% |
| 0.5-0.7 | 1621 | 8.1% | 0.3% | 7.8% |
| >=0.7 | 581 | 39.1% | 10.7% | 28.4% |

## Reference by agent type

| agent | events | r1 pairs | r1 grounded | r1 twin | r1 excess | top-3 pairs | top-3 grounded | top-3 twin | top-3 excess | r1 lookup |
|---|---|---|---|---|---|---|---|---|---|---|
| claude | 34 | 34 | 32.4% | 0.0% | 32.4% | 101 | 18.8% | 1.0% | 17.8% | 8.8% |
| codex | 144 | 126 | 4.8% | 0.0% | 4.8% | 347 | 4.9% | 0.9% | 4.0% | 17.4% |
| other | 2729 | 2372 | 14.2% | 2.8% | 11.5% | 6524 | 11.4% | 2.3% | 9.1% | 2.1% |
| unknown | 263 | 61 | 9.8% | 1.6% | 8.2% | 179 | 7.8% | 2.2% | 5.6% | 2.7% |

Agent type: recall_events.agent / ambient agent, else the majority `context.agent` of nodes written on the same transport session; `unknown` = nothing recorded.

## Marks

**No marks.** table recall_feedback_marks is absent.

## Falsifier: better than random

> better_than_random: explicit `used` marks agree with the independent check better than random iff ALL hold: (a) at least 30 accepted `used` marks on at least 10 in-window recall events; (b) within-event permutation test: the number of `used`-marked nodes that carry evidence (grounded in the closing trace at containment >= 0.22 OR looked up on the same transport after delivery) is compared with 2000 draws in which each event's `used` marks are re-assigned uniformly at random to the same number of that event's delivered nodes; one-sided p = (1 + #draws >= observed) / (1 + 2000) must be < 0.05; (c) placebo-subtracted evidence excess of `used` marks, mean over marks of [evidence(n) - grounded(twin(n))], has an event-cluster bootstrap 95% CI (1000 rounds) whose lower bound is > 0. If (a) fails the verdict is `insufficient`; otherwise failing (b) or (c) is `fail` - the falsifier fires and marks are not connected to reinforcement.

**Verdict: `insufficient`** (used marks: 0).
