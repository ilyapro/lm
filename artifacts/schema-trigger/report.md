# Report: `LM_RECALL_SCHEMA_TRIGGER=name` (goal schema-ranks-by-meaning)

Lines and data are in `prereg.md`, which was committed before the holdout run.
Numbers come from `scripts/schema_trigger_replay.py run|names` on snapshots
taken 2026-09-29 07:13Z (sfx sha256 `3027dbb3…`, alt `2b8fa04f…`). The files
are `dev-*.json`, `holdout-*.json` and `names-*.json`. Query texts are not
committed. sfx and alt are reported separately and never merged.

"Marked schemas" means accepted `recall_feedback_marks` on `level='schema'`
nodes. "Delivered" means delivered in a full slot by the replayed arm.
"Trigger-found" means the field delivered that schema with `trigger` in
`methods`. Schema text is the content chars after `delivery.shape_recall_results`,
with the full node chars in brackets.

## Holdout

| | sfx legacy | sfx name | alt legacy | alt name |
|---|---|---|---|---|
| events replayed | 335 | 335 | 146 | 146 |
| irrelevant marked schemas delivered (of marked) | 75 / 164 | **10** / 164 | 7 / 7 | **2** / 7 |
| … trigger-found | 73 / 159 | **8** / 159 | 5 / 5 | **0** / 5 |
| used marked schemas delivered (of marked) | 35 / 52 | 16 / 52 (**46%** kept) | – (0 marked) | – |
| … trigger-found | 33 / 49 | 14 / 49 (42%) | – | – |
| irrelevant share among delivered marked schemas | 68.2% | **38.5%** | 100% (7) | 100% (2) |
| schema slots delivered | 198 | 61 | 9 | 4 |
| schema at rank 1 | 81 | 16 | 5 | 0 |
| schema text, chars | 541,155 (935,550) | **92,333** (174,824) | 6,354 (8,022) | **1,307** (2,975) |
| all text, chars | 1,843,367 | 1,491,879 | 414,039 | 412,779 |
| recall latency mean / p50 / p90, s | 1.47 / 0.61 / 3.71 | **1.35** / 0.62 / 3.59 | 2.56 / 2.38 / 4.77 | **2.51** / 2.51 / 4.46 |
| rank 1 changed | | 70 | | 5 |

## Dev (design data, before the holdout start)

| | sfx legacy | sfx name | alt legacy | alt name |
|---|---|---|---|---|
| events | 150 (sample, seed 7) | | 330 | |
| irrelevant marked schemas delivered | 19 / 76 | 0 / 76 | 7 / 14 | 2 / 14 |
| used marked schemas delivered | 11 / 20 | 6 / 20 (55%) | 11 / 13 | 4 / 13 (36%) |
| irrelevant share among delivered marked schemas | 63.3% | 0% | 38.9% | 33.3% |
| schema text, chars | 68,763 | 24,924 | 23,061 | 8,410 |
| latency mean, s | 2.45 | 2.57 | 1.39 | 1.40 |

## Procedure-name queries (L5, fresh snapshot, schema's own scope, top 5)

| | sfx legacy | sfx name | alt legacy | alt name |
|---|---|---|---|---|
| queries | 150 | 150 | 50 (every schema with a trigger) | 50 |
| schema at rank 1 | 103 | **128** | 47 | **48** |
| … when legacy saw meaning agreement (vector ≥ 0.55 or bm25 ≥ 0.2) | 72 / 109 | **108 / 109** | 46 / 47 | **47 / 47** |
| latency mean, s | 1.37 | 1.10 | 2.36 | 2.35 |

On sfx, 42 names that legacy did not put at rank 1 are now at rank 1. The
legacy failures came from several trigger schemas that all got the 1.8× boost
competing for the top. 17 names that legacy put at rank 1 are gone. All of
them are generic names with no meaning agreement: `reopen lesson` ×11 (vector
0.51, demoted by the query's own irrelevant marks), `verify pass` ×2,
`task outcome`, `gate bug lesson`, `implementation fix` (vector 0),
`mr review`, and `p7b v2 fresh evidence` (vector 0.53, bm25 0, gate 0.26).
The single agreement miss is `implementation repair`, with vector 0 and bm25
exactly 0.2 and a gate score of 0.10. Its schema passes the goal's reference
rule only through the bm25 = 0.2 boundary.

## Verdicts (prereg lines)

| line | sfx | alt |
|---|---|---|
| L1 irrelevant trigger-found ≤ 0.5× and share lower | **PASS** (73 → 8; 68.2% → 38.5%) | not judged (7 < 20 delivered marked); 5 → 0 |
| L2 used kept ≥ 0.5× | **FAIL** (35 → 16 = 0.457) | not judged (0 used marked) |
| L3 schema text lower | **PASS** (−83%) | not judged; −79% |
| L4 latency mean not higher | **PASS** (1.47 → 1.35 s) | not judged; 2.56 → 2.51 s |
| L5 name queries | **FAIL** on the agreement clause (108/109); rank 1 128 ≥ 103 passes | **PASS** (48 ≥ 47; 47/47) |

**By the pre-registered rule the valve is not recommended on sfx.** L2 misses
by 4 points and L5 by one query. On alt the holdout is too small to judge
L1–L4. The code is in and the valve is unset on both hosts, so enabling it is
the operator's call, and the trade-off is below.

## What the trade-off is

The irrelevant schema slots are cut by 87% (sfx) and the schema text by
80–83%. About half of the useful schemas go with them, and on alt dev 7 of 11.
The loss is not a defect of the name rule. It is the verdict of the existing
quality gate. The main case on sfx is the AE supervisor loop
(`ae supervisor current monitoring sfx alt active goals …`). It marks trigger
schemas such as `dashboard goal api` as used. Those schemas have vector ≈ 0.55
and bm25 0, so on the channel scale they score about 0.3, below
`LM_RECALL_MIN_SCORE=0.35`. Older traces with a similar vector and feedback
multipliers take the slots instead. The goal's reference rule
"vector ≥ 0.55 or bm25 ≥ 0.2" kept 53% of used at 57% precision. `name` keeps
46% at 62% precision (16 used / 10 irrelevant) on the sfx holdout. Keeping more
of them means changing the gate, which is a non-goal here.

## Remaining special cases of the trigger

| | legacy | `name` |
|---|---|---|
| constant score `0.95 + 0.05·overlap` | yes | no |
| `1.8×` final multiplier | yes | no |
| own gate scale (`trigger_gate_score ≥ 0.5`) | yes | no |
| trigger finds candidates (word overlap ≥ 0.5) | yes | no |
| cross-scope admission by trigger | yes | inert (no trigger score while ranking) |
| name pin: query token set = trigger token set, passes the general gate → rank 1 | – | yes (the one case left) |

The valve name, the value and the condition for removing it together with the
legacy path are in `docs/recall-schema-trigger.md`.
