# Independent holdout: protocol, frozen before the first run

Written and committed before `holdout.py` was ever run. Its criteria, selector,
seed and exclusions are not changed after the run; any later change needs a new
protocol file and a new seed.

## Why a new set

The seed-20260930 replay sets (40 schema topics, 23 prose recipes) were inspected
while repairing the code. The two recipes lost there were read in full, and so were the
teach corrections of 10 groups. Those sets are regression/design data from now
on. They are not evidence of generalisation.

## Source

A fresh read-only SQLite backup of the live store, taken on 2026-09-30 before
any pass of the new code ran on it (483 active procedural schemas). It lives
only in the node's scratch directory. Nothing from it enters git except
aggregate counts.

## Population and exclusions

- Unit: one active procedural schema with a readable trigger (the replay's
  `readable`), i.e. one group, as the replaced mechanism left it.
- Excluded, mechanically, before sampling:
  1. every group of the seed-20260930 sets (the schema topics and the groups of
     the prose-recipe targets), recomputed with `replay.eval_sets`;
  2. every group whose key was displayed during the repair session: the first
     25 groups by teach-head count in the order `heads_order` reproduces (this
     includes the 10 groups whose teach corrections were read in full and both
     lost recipes).
- Sample: `random.Random(20261001)`, shuffled, first 40 groups.

## Items

For each sampled group, an item is each record that the legacy schema carried
(its id is in the schema's `source_traces`) and that is still current (active,
not superseded). Items are labelled for evaluation only. Production reads none
of this:

- **instruction/correction**: the text opens as a procedure/recipe
  (`replay.RECIPE_HEAD`), or the record is a teach correction (it is the source of
  a `supersedes` edge);
- **case record**: every other item.

## Measurement

The query is `how to <trigger>` in the schema's scope, `max_results=10`. It runs
through the MCP tool functions with the operator's delivery environment and
unchanged live valves. Every non-full delivered schema is re-fetched with
`memory_lookup`, and so is any non-full delivery of an item. An item is
*obtained* when its full text is in the delivered or re-fetched text. `before`
is the backup as it is. `after` is a separate copy after two ordinary
procedural passes of the candidate code.

## Criteria (fixed now)

- H1: new losses of instruction/correction items (obtained before, not after)
  = **0**.
- H2: census after the passes: unconfirmed schema steps = **0**, counted over
  every active procedural schema, orphans included. Unconfirmed steps delivered
  on the holdout queries = **0**.
- H3: delivered+forced-lookup characters after ≤ before, summed over the
  holdout queries.

Reported next to the criteria, without deciding pass or fail: the baseline
misses (items not obtained even before), the losses of case records (these are
removed on purpose), and instruction/correction items obtained after but not
before.

## Holdout 1 outcome (recorded without changing the criteria above)

The seeded 40 groups contain **no** instruction/correction item: 289 items, all
labelled case records. H1 = 0 there is vacuous and is not claimed as evidence.
H2 and H3 were measured and are reported in the report.

# Holdout 2: protocol, frozen before its first run

Holdout 2 was added after holdout 1 came out vacuous. Before freezing it, only the
size of the population was looked at (315 groups remain after the exclusions
above; 33 of them hold 52 instruction/correction items). No outcome was
looked at.

- Population: same source, same exclusions, same items and labels as holdout
  1.
- Unit: **every** remaining group that holds at least one instruction/correction
  item. This is a census of the stratum, not a sample, so no seed. None of these
  groups is in holdout 1.
- Measurement: as in holdout 1.
- Criteria: H1 (new losses of instruction/correction items = 0), H2 and H3, as
  above.
- `python3 holdout.py <copy> <work> stratum` runs it. The default run is
  holdout 1, unchanged.

# Holdout 3: protocol, frozen before its first run (supervisor #1333)

Holdout 2 was examined while diagnosing the name carrier, so it is regression
data now, like the design set. Holdout 3 is drawn from what nobody has looked
at. Before freezing it, nothing about its population was looked at, not even its size.

- Source: the same pristine backup (taken before any pass of new code ran).
- Excluded groups: every group of the design sets, the 25 displayed groups,
  and every group that holds an item of holdout 1 or holdout 2.
- Unit: one current (active, not superseded) record labelled
  *instruction/correction* by the same evaluation-only rule (the text opens as a
  procedure/recipe, or the record is a teach correction), carrying a readable
  procedure name. At most one per group. It does not matter whether a legacy schema
  carried it: the set also checks that the new name channel costs nothing to
  records that never needed a schema.
- Sample: `random.Random(20261002)` over eligible records sorted by id,
  first 60 with distinct groups.
- Query: `how to <the record's own normalized procedure name>` in its scope,
  `max_results=10`, operator delivery environment, live valves unchanged.
  Non-full deliveries of schemas and of the item are re-fetched with
  `memory_lookup`.
- `before`: code at `24e9009` on an untouched copy. `after`: the candidate on a
  separate copy after two ordinary procedural passes.
- Criteria: H1: new losses (obtained before, not after) = **0**. H3:
  delivered+lookup characters after ≤ before. Reported without a pass/fail
  verdict: gains and baseline misses.
- Collateral check, reported only: 60 distinct real recall queries from the
  copy's `recall_events` (`random.Random(20261003)`). Share of queries whose
  delivered top 10 is unchanged, before vs after.
- `holdout3.py` runs it (commands in its docstring).

## Holdout 3 outcome (recorded without changing the criteria above)

The one independent run measured the candidate whose production code is the
tree of commit `1ebec0f` (the same `src/` as `211af7b`). Result: 39 items, 38
obtained before and 38 after, 0 new losses, 0 gains, 1 baseline miss. Characters
fell from 970 955 to 395 635. After seeing this, the retrieval blend was changed
again (commits `ff76267`..`d49ca46`) because of losses on the design set. A later run of
holdout 3 on that final code is **regression data**, not a second independent
holdout. No unseen set remains: the live store gained one instruction-labelled
record since the snapshot.

# Holdout 4: the independent alt corpus (supervisor #1339), frozen before its first run

Design, holdout 2 and holdout 3 were all examined while repairing, so they are
regression data. Holdout 4 comes from a store nobody examined here: a
read-only online-backup snapshot of the **alt** host's live store, copied
alt→sfx only (nothing went sfx→alt). Its source has 50 live procedural schemas
and 10 271 live traces (counted before freezing; no item, text or outcome was
looked at).

- Candidate: frozen in the commit that pins this section's candidate line
  below, before any run. After the run, no code change is attributed to this
  set; any later code gets this set as regression data only.
- Independence: every node whose id **or** exact content (sha256 of the
  stripped text) also exists anywhere in the sfx pristine copy — the store the
  design set and holdouts 1–3 came from — is excluded. The excluded count is
  reported.
- Family A, schema topics (the queries that obtained items before): every live
  procedural schema with a readable trigger. Query `how to <trigger>` in its
  scope. Items: every current (active, not superseded) source whose whole text
  that schema delivered. Each item is labelled *instruction/correction* (the
  text opens as a procedure/recipe, or it is a teach correction) or *case*.
- Family B, own name: every current instruction/correction record with a
  readable procedure name, queried by `how to <its normalized name>`.
- All eligible items, no sampling, no seed.
- Measurement: `max_results=10`, alt's own operator environment (its
  `~/.config/living-memory/env` without secrets and DB path; it differs from
  sfx only in `LM_RECALL_MIN_SCORE=0.30`). *Obtained* = the item's whole text is
  in a delivered result or in a lookup the reader path forces: a cut schema,
  group carrier or wanted record is re-fetched, and the evidence ids a
  delivered group carrier lists are looked up (`replay.read`). All those
  characters count.
- `before`: code `24e9009` on an untouched copy. `after`: the candidate on a
  separate copy after two ordinary procedural passes per scope.
- Criteria. H1: new losses of instruction/correction items (obtained before,
  not after) = **0** in each family. H2: unconfirmed schema steps after = 0 in
  the census and on every delivered schema. Reported without a verdict: case
  items, gains and baseline misses (separately), characters before/after.
  If fewer than 10 instruction/correction items survive the exclusions, the
  set is reported as too small to support an independence claim.
- `holdout4.py` runs it (commands in its docstring). One run.

Candidate pinned for holdout 4 (and for the final regression runs), before
any of them ran: commit `e8913bb`, `src/` tree `b59d654e`, oracle tree
`artifacts/procedure-case-separation` `3862218c`. Every run imports that code
from an immutable checkout of `e8913bb`; `before` imports `24e9009`.
