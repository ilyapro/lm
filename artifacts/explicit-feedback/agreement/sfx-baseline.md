# Explicit-feedback agreement check — sfx

Store: `/home/sfx/.local/share/living-memory/global.sqlite3` (read-only), since `2026-09-07`, generated 2026-09-27T11:14:29Z.
Script: `scripts/explicit_feedback_agreement.py`. Placebo method: LM memory 01M3H76JBHEZKJMBPWFVN70SB2 (twin = undelivered node created before the event, |cos(twin,T) − cos(n,T)| ≤ 0.01, mean-pooled chunk vectors; grounding `ground_token_sets` on delivered + twin, threshold 0.22).
A/B exclusion: whole transport sessions with an A/B marker (scope project:target|repo|x, tree-context-ab/fixture/project=target in node context, or A/B-like recall query).

## Counts

| item | value |
|---|---|
| events | 7031 |
| closed_events | 5096 |
| closed_scored | 5071 |
| delivered_without_twin | 588 |
| closed_without_trace_vector | 25 |
| excluded_ab_events | 5466 |
| ab_sessions | 2815 |
| delivered_pairs | 35890 |

## Placebo reference rates (grounded vs same-cosine twin)

| slice | pairs | grounded | twin | excess | excess 95% CI | grounded −q | twin −q | excess −q | excess −q 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| r1 | 4841 | 8.7% | 2.3% | 6.4% | [+5.6, +7.2] pp | 3.9% | 1.4% | 2.5% | [+1.9, +3.0] pp |
| top3 | 14023 | 7.1% | 1.6% | 5.5% | [+5.1, +6.0] pp | 3.2% | 1.0% | 2.1% | [+1.8, +2.5] pp |

`−q`: query tokens removed from the closing trace before grading.

Lookup rate (same transport, after delivery, or ledger basis lookup): r1 2.6%, top-3 5.4%. No placebo exists for lookup (a twin is never on the card).

r1 by cos(r1, closing trace):

| cos bin | pairs | grounded | twin | excess |
|---|---|---|---|---|
| <0.5 | 1065 | 0.3% | 0.1% | 0.2% |
| 0.5-0.7 | 2781 | 5.5% | 0.7% | 4.7% |
| >=0.7 | 995 | 26.9% | 9.1% | 17.8% |

## Reference by agent type

| agent | events | r1 pairs | r1 grounded | r1 twin | r1 excess | top-3 pairs | top-3 grounded | top-3 twin | top-3 excess | r1 lookup |
|---|---|---|---|---|---|---|---|---|---|---|
| claude | 531 | 492 | 12.8% | 3.3% | 9.6% | 1481 | 10.6% | 1.6% | 9.0% | 1.1% |
| codex | 263 | 152 | 4.6% | 0.7% | 3.9% | 430 | 4.7% | 0.7% | 4.0% | 1.9% |
| opencode | 11 | 11 | 0.0% | 0.0% | 0.0% | 32 | 0.0% | 0.0% | 0.0% | 18.2% |
| other | 4793 | 3607 | 8.1% | 2.2% | 5.9% | 10321 | 6.5% | 1.6% | 4.9% | 2.8% |
| unknown | 1433 | 579 | 10.5% | 2.8% | 7.8% | 1759 | 8.6% | 1.6% | 7.0% | 2.4% |

Agent type: recall_events.agent / ambient agent, else the majority `context.agent` of nodes written on the same transport session; `unknown` = nothing recorded.

## Marks

**No marks.** table recall_feedback_marks is absent.

## Falsifier: better than random

> better_than_random: explicit `used` marks agree with the independent check better than random iff ALL hold: (a) at least 30 accepted `used` marks on at least 10 in-window recall events; (b) within-event permutation test: the number of `used`-marked nodes that carry evidence (grounded in the closing trace at containment >= 0.22 OR looked up on the same transport after delivery) is compared with 2000 draws in which each event's `used` marks are re-assigned uniformly at random to the same number of that event's delivered nodes; one-sided p = (1 + #draws >= observed) / (1 + 2000) must be < 0.05; (c) placebo-subtracted evidence excess of `used` marks, mean over marks of [evidence(n) - grounded(twin(n))], has an event-cluster bootstrap 95% CI (1000 rounds) whose lower bound is > 0. If (a) fails the verdict is `insufficient`; otherwise failing (b) or (c) is `fail` - the falsifier fires and marks are not connected to reinforcement.

**Verdict: `insufficient`** (used marks: 0).
