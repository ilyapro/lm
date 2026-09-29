# Report 3: one score for order and gate. `LM_RECALL_SCHEMA_TRIGGER=name` meets the bar on sfx

The rules come from `prereg-2.md`, with the data status of `prereg-3.md`. Both
were committed before this change (b070c08) and before any replay of the fresh
holdout. Numbers are from `scripts/schema_trigger_replay.py run|names` on sfx
snapshot 2026-09-29T16:29:01Z (sha256 `4b6380d3…`), code 9006a95. The two
judged replays are written locally to `v2/original-sfx.json` and
`v2/new-sfx.json`, which the acceptance check reads. Everything else goes to
`v3/`. Replay outputs of this attempt are private and are not committed. No
query text is committed.

## What changed

Under the valve only, `rank_candidates` scores every candidate exactly as the
quality gate does (`score_gate.gate_score`). The channels are blended with the
fixed `REFERENCE_WEIGHTS`, not the store's learned per-scope weights. The ranker
also drops the 1.2x prior for nodes that correct another, because the gate
never applies it. The order and the gate now read one score, for every node,
whatever its level, origin or project. The correction-dominance pass still
puts each correction above the node it corrects. The named-schema repair of
5804d83 (bm25 1.0 for an exact name, first if it passes the gate) stays. Two
changed lines in `rank_candidates`, one new line. A redundant early return in
`_named_schemas_first` is removed. `src/` against master before this goal
(9034741): +20 / −21. Without the valve `reference` is `None`, the per-scope
weights and the correction prior apply as before, and `_named_schemas_first`
does not run. That path is byte for byte the old code.

## Why (design data only)

The earlier losses of type "slot" were schemas that passed the gate but had
more than `max_results` passing results ranked above them. Dumping the ranked
list showed where the two criteria disagree (seen holdout, `project:lm` event,
max 8). Traces with only vector 0.57 have gate 0.38 but ranker 0.87 under
`project:lm`'s vector-heavy weights. The used procedure `lm recall map curtail`
has vector 0.52 and graph 0.32. Its gate is 0.46, the ranker put it at 0.67,
rank 23. The first candidate changed only the weights. Its real replay gave
10/21, while the offline re-sort had predicted 12. The gap was the correction
prior: corrections at gate 0.42 still ranked at 0.51, above the procedure.
With the prior dropped too, the real deliveries equal the re-sort exactly (both
data sets, every event).

## Data sizes (sfx; alt at the end)

| | original corpus | seen holdout (07–15Z) | fresh holdout (15:00–16:26Z) |
|---|---|---|---|
| events | 335 | 176 | 53 |
| marked used / irrelevant procedure-events | 39 / 98 | 61 / 169 | 16 / 71 |
| … delivered by `legacy` | 31 / 93 | 21 / 32 | 3 / 5 |

## Field replay (sfx, `legacy` recomputed on the same snapshot)

| | original legacy | original name | seen legacy | seen name | fresh legacy | fresh name |
|---|---|---|---|---|---|---|
| used proc delivered | 31 | **20 (64.5%)** | 21 | **12 (57.1%)** | 3 | 2 |
| irrelevant proc delivered | 93 | **24** | 32 | **5** | 5 | 0 |
| irrelevant share (proc) | 75.0% | **54.5%** | 60.4% | **29.4%** | 62.5% | 0% |
| nodes: irrelevant / used delivered | 76 / 27 | 11 / 11 | 32 / 21 | 5 / 10 | 5 / 3 | 0 / 2 |
| schema text, chars | 442,252 | **96,266** | 185,755 | **26,204** | 46,367 | 7,508 |
| schema slots | 180 | 67 | 66 | 27 | 11 | 4 |
| latency mean / p90, s | 1.18 / 3.02 | **1.15** / 3.16 | 2.04 / 3.57 | **1.86** / 3.56 | 3.79 / 3.95 | 2.97 / 3.75 |

For reference, 5804d83 (`name` before this change) on the same snapshot
delivers used 19 / irrelevant 23 (original) and 10 / 6 (seen).

## Names (L5; top 5, the schema's own scope)

| | dev legacy | dev name | holdout legacy | holdout name |
|---|---|---|---|---|
| queries | 149 | 149 | 333 | 333 |
| procedure at rank 1 | 98 | **142** | 201 | **321** |
| … of the meaning-agreement queries | 45 / 80 | **80 / 80** | 119 / 218 | **218 / 218** |

One dev name is at rank 1 under 5804d83 on the 15:21Z snapshot and is not
delivered now (`tree decomposition`, `project:octopus`, no meaning agreement).
The cause is not the code. The schema is named (bm25 1.0), but on the 16:29Z
snapshot it carries a demotion of 0.1, which appeared after 15:21Z. Its gate
comes to 0.08, and any `name` version refuses it. One agreement query of the
holdout left the set between the two snapshots (219 → 218).

## Negative cases (N)

Delivered lists that differ between 5804d83 and this change: original corpus
159 / 335 (3 of them exact-name queries), seen holdout 47 / 176. The order
change is not schema-specific, so it moves traces too. Marked non-schema
nodes delivered:

| | legacy | 5804d83 | this change |
|---|---|---|---|
| original: used / irrelevant | 105 / 310 | 107 / 314 | 105 / 269 |
| seen: used / irrelevant | 59 / 156 | 59 / 158 | 57 / 137 |

Irrelevant non-schema deliveries fall by 14% (original) and 13% (seen). Used
ones fall by 2 on each set: legacy level on the original corpus, 2 below it on
the seen holdout. A named procedure is still refused when the query's own
irrelevant marks or a hub demotion take it under the gate (the case above).
Containing a procedure's name as a phrase is still not naming it.

## Where a useful procedure is still lost (sfx, legacy delivered it)

| stage | original (12) | seen (9) |
|---|---|---|
| gate, retrieval or admission: not in the top 80 by gate score, or gate < 0.35 | 9 + 1 | 6 + 1 |
| slot: passes, but `max_results` results score higher on the same gate score | 2 | 2 |

On the original corpus `name` also delivers one used procedure-event that
`legacy` does not (`dashboard goal api supervision`), hence 31 − 12 + 1 = 20.

These are the procedures of `report-2.md` again (`gate bug lesson`,
`dashboard agent chats`, `augment repair transaction`, `ae goal creation
style`, …). None has meaning evidence beyond shared trigger words. The slot
losses are now outranked on the gate's own score, so they are the same verdict
as for any other node.

## Verdicts (sfx)

| line | original corpus | seen holdout | fresh holdout |
|---|---|---|---|
| L1 irrelevant proc ≤ 0.5× and share lower | **PASS** (93 → 24; 75.0% → 54.5%) | **PASS** (32 → 5; 60.4% → 29.4%) | reported (5 → 0; too small) |
| L2 used proc ≥ 0.5× | **PASS** (31 → 20 = 0.645) | **PASS** (21 → 12 = 0.571) | reported (3 → 2; too small) |
| L3 schema text lower | **PASS** (−78%) | **PASS** (−86%) | reported (−84%) |
| L4 latency mean ≤ legacy + 5% | **PASS** (1.15 vs 1.18) | **PASS** (1.86 vs 2.04) | reported (2.97 vs 3.79) |
| L5 names | **PASS** dev 142 ≥ 98, 80/80 | **PASS** holdout 321 ≥ 201, 218/218 | – |

**By the prereg-2 rule the bar is met on sfx.** Every judged line passes on
both data sets. The acceptance check (`v2/{original,new}-sfx.json`, L1–L4)
passes. The fresh holdout is below the judging floor. It points the same way,
but it is not evidence of a general win. The seen holdout was design data for
this change: the cause was found in its ranked lists.

## alt

Snapshot 2026-09-29T17:04:47Z (sha256 `10d899ad…`), streamed from alt to
sfx. Nothing from sfx goes to alt.

| | original legacy | original name | 07–15Z legacy | 07–15Z name |
|---|---|---|---|---|
| events | 146 | 146 | 17 | 17 |
| used / irrelevant proc marked | 0 / 7 | | 0 / 2 | |
| irrelevant proc delivered | 7 | 2 | 0 | 0 |
| schema text, chars (slots) | 6,737 (11) | 3,948 (8) | 0 (0) | 1,095 (3) |
| latency mean, s | 0.20 | 0.17 | 0.51 | 0.25 |

L1–L4 are below the judging floor and are reported only. In the 17-event
window `name` delivers three schemas that no one marked, where `legacy`
delivered none. 5804d83 had delivered none there as well. This is the order
change: the schemas pass the gate, and the per-scope weights no longer keep
them low. L5 names dev (47 agreement queries, judged): rank 1 48 against
legacy 47, and 47 / 47 agreement queries at rank 1. **PASS.** Names holdout
on alt is empty: all 50 alt schemas were in dev.

## Addendum (17:24Z): the two judged files are regenerated and no longer tracked

The first landing of this candidate stopped before acceptance. The two
regenerated files had been hidden from git with `skip-worktree`, and the
harness refuses such an index entry. The execution directory was then
replaced, and the regenerated files went with it.

`v2/original-sfx.json` and `v2/new-sfx.json` are now untracked and listed in
`.gitignore` (101a487). The versions committed by 5804d83 (snapshot 15:21Z,
`name` before this change, L2 10 / 21) stay in history. Their numbers are in
`report-2.md`, which is not changed.

Both files were regenerated with the same command, on the same snapshot
(`4b6380d3…`) and the same code: `src/`, `scripts/` and `tests/` have not
changed since 9006a95. Every count of both arms equals the replay reported
above, on both data sets. Only the latency differs, because the host was less
loaded:

| latency mean / p90, s | legacy | name |
|---|---|---|
| original corpus | 0.31 / 0.57 | **0.29** / 0.54 |
| seen holdout | 0.42 / 0.78 | **0.35** / 0.60 |

L4 passes in both runs. The acceptance check passes on the regenerated files.
Nothing else was replayed again, because the candidate is the same: L5, the
fresh holdout, alt and N are the measurements above.

One new check was run, for the path without the valve. The same two windows
were replayed on the code before this goal (9034741), with this branch's
replay scripts and the same snapshot. The delivered lists of the `legacy` arm
are identical to this branch's in 335 of 335 events (original corpus) and 176
of 176 (seen holdout).

Later field traffic was counted in the live store (read-only), not replayed.
From 16:26Z to 17:24Z there are 32 events with 7 used and 42 irrelevant marks
on procedures. That is smaller than the fresh holdout (16 / 71 marked), it is
below the judging floor, and it includes the recalls of this goal's own
attempts. So the evidence on independent data is still the fresh holdout and
alt, both reported only.
