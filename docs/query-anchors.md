# Query anchors: evaluation protocol and what it can prove

The decisions that fix how query anchors are *scored*, and the measured ceiling
that says which of the goal's numeric targets are reachable at all.

Every number here comes from `artifacts/anchors/estimate.json` /
`estimate.md`, produced read-only by `scripts/anchor_grounding_estimate.py`
over a `sqlite3`-backup-API snapshot of the live database taken 2026-08-18
(`sha256 5f82312e…`, 16,717 nodes, 54,814 recall events, 9,108 consumed), with
the real `paraphrase-multilingual-MiniLM-L12-v2` model. The anchor-free
baseline is a real `retrieval_harness run` over the same snapshot and the
shipped goldset: **hit@1 0.188 / hit@5 0.577 / MRR 0.354**, reproducing the
recorded phase-1 numbers exactly.

Reproduce:

```bash
python3 -m living_memory.retrieval_harness snapshot \
    --source-db ~/.local/share/living-memory/global.sqlite3 \
    --snapshot-out /tmp/anchor-est/snap.sqlite3
python3 -m living_memory.retrieval_harness run \
    --snapshot /tmp/anchor-est/snap.sqlite3 \
    --goldset artifacts/harness/goldset.jsonl \
    --report /tmp/anchor-est/harness-baseline.json
python3 scripts/anchor_grounding_estimate.py \
    --snapshot /tmp/anchor-est/snap.sqlite3 \
    --goldset artifacts/harness/goldset.jsonl \
    --baseline-report /tmp/anchor-est/harness-baseline.json \
    --ceiling-cutoff 2026-06-10T00:00:00Z --ceiling-cutoff 2026-07-15T00:00:00Z \
    --out-json artifacts/anchors/estimate.json --out-md artifacts/anchors/estimate.md
```

## 1. What the history yields

Grading all 9,108 consumed events with the live grounding verdict
(`grounding.ground_results`, per-event IDF, `min_containment` 0.25) confirms
the goal's 600-event estimate:

| figure | prior estimate | measured, full history |
| --- | --- | --- |
| grounded events | 41.3% | **40.6%** (3,682 of 9,073 gradeable) |
| anchors after exact-fingerprint dedup | ~3,760 | **3,450** |
| edges | ~8,550 | **8,335** |
| edges per anchor | 2.28 | **2.42** |
| edges into decayed targets | 4.6% | **5.7%** |

Three things the sample could not show, and the store should act on:

- **Exact repeats barely exist.** Only **57 of 3,450** anchors are reinforced
  by more than one grounded event (1.07 events per anchor). The "operator
  repeats himself" premise does *not* show up as identical query strings; it
  can only show up as near-duplicates, which makes the retrieval-time match —
  not write-time dedup — the load-bearing part of the design.
- **No fingerprint ever spans two scopes** (0 of 3,450), so
  `recall_fingerprint(query, requested_scope)` is safe as the anchor key and
  "an anchor never crosses scope" is free.
- **Decayed targets concentrate**: 5.7% overall but **15.1% in
  `project:online`** and 12.3% in `project:lm`. A backfill that writes edges to
  decayed nodes would point one scope's anchors at the past far more than the
  headline number suggests. Skip decayed targets at write time.

## 2. Near-duplicate dedup threshold: 0.995, plus an identifier guard

Cosine between every same-scope pair of the 3,450 anchor queries (1,482,384
pairs). The median anchor already has a same-scope neighbour at **0.774**, so
any threshold near that merges everything.

| threshold | pairs ≥ t | of those, with *different* identifiers | anchors absorbed by transitive merge |
| --- | --- | --- | --- |
| 0.90 | 664 | 241 (36%) | 299 (8.7%) |
| 0.95 | 230 | 127 (**55%**) | 105 (3.0%) |
| 0.97 | 94 | 56 (**60%**) | 55 (1.6%) |
| 0.99 | 7 | 1 (14%) | 7 (0.2%) |

An *identifier* is a token containing a digit — `EZ-13871`, `9806`, `NW-7`.
Above 0.95 the **majority** of merge candidates differ in exactly that: they
are the same *class* of situation and a different ticket. Two failure modes are
visible in the sampled pairs:

- `task_pattern:a7c7530c40cc59c4` ↔ `task_pattern:5b492dddc439419d` = **0.979**.
  Opaque hex lookups carry no signal for the model, so it maps all of them next
  to each other. Merging these merges unrelated lookups.
- `Изучи https://jira.2gis.ru/browse/EZ-13512 …` ↔ `… /EZ-13560 …` = **0.990**.
  A long templated query differing only in the ticket still clears 0.99.

**Decision.** Write-time near-duplicate merging uses **cosine ≥ 0.995 AND
identical identifier-token sets**; below that, a similar query creates its own
anchor. This costs almost nothing — at 0.99 only 7 pairs (0.2% of anchors)
would have merged at all, so exact-fingerprint dedup plus reinforcement already
does the work. Near-duplicate similarity belongs at *retrieval* time (matching
an incoming query to an anchor), where a wrong match costs one bad seed, not at
*write* time, where it destroys an anchor permanently.

## 3. The cutoff: `2026-07-15T00:00:00Z`

The anchor window is the events strictly before the cutoff. Later is better for
anchors and worse for the event-derived eval set, so the choice is measured,
not argued. Anchor-side reachability of the goldset, grounded edges only:

| cutoff | anchors | edges | Cyrillic in window | cross_lingual oracle / @10 | role_query oracle / @10 |
| --- | --- | --- | --- | --- | --- |
| 2026-06-10 | 1,685 | 3,523 | 178 (3.3%) | 0.184 / 0.026 | 0.111 / 0.083 |
| 2026-07-01 | 2,938 | 7,296 | 216 (2.9%) | 0.474 / 0.079 | 0.139 / 0.083 |
| **2026-07-15** | **3,178** | **7,879** | **391 (4.9%)** | **0.579 / 0.105** | **0.167 / 0.111** |
| 2026-08-01 | 3,271 | 8,051 | 553 (6.7%) | 0.579 / 0.105 | 0.167 / 0.111 |
| 2026-08-10 | 3,331 | 8,159 | 687 (8.0%) | 0.579 / 0.105 | 0.167 / 0.083 |

`oracle` = some eligible anchor holds an edge to a relevant node (the
structural ceiling, ignoring the match). `@10` = one of the ten nearest anchors
does (what the mechanism can actually see).

**Reachability saturates at 2026-07-15** — every later cutoff buys zero for the
two strata that carry the goal's targets, while the regenerated
`content_grounded` stratum keeps shrinking (269 items at 2026-07-15, 172 at
2026-08-01). 2026-07-15 is therefore the measured optimum, and it is also the
first cutoff whose window contains a non-trivial share of the Russian-jargon
traffic the whole premise depends on (391 queries vs 178 at 2026-06-10).

## 4. The goldset: regenerate `content_grounded`, freeze the curated strata

**Regenerate at the cutoff.** The shipped `artifacts/harness/goldset.jsonl` was
built at cutoff 2026-06-10 (a seeded 160-of-1,786 sample; the build command is
recorded in `artifacts/harness/seed-queries.json`). Its `content_grounded`
items come from events *after* 2026-06-10, so at a 2026-07-15 cutoff **141 of
its 160 items sit inside the anchor training window**. Measured, that is not a
theoretical risk: those items score **@1 1.000** because the anchor is their own
recall event. Regenerating at 2026-07-15 yields **269** items, all post-cutoff.

**The anchor-free baseline MUST be recomputed on the regenerated goldset.**
The existing 0.577 / 0.354 belong to the 234-item shipped goldset. Comparing an
anchored run on a 343-item goldset against them is invalid — different items,
different denominator. Arm A and arm B must read the identical file.

**Freeze the 74 curated items.** `cross_lingual` relevance is resolved as the
paraphrase query's own top-k (`retrieval_harness.py:1568`,
`resolution: "paraphrase_top_k"`), so it moves whenever retrieval moves.
Measured: rebuilding on a snapshot **one day newer** changed the relevant node
of **14 of 38 cross_lingual items (37%)**; `role_query`, whose ids are curated,
changed 0 of 36. Two consequences:

1. The scored run must use one frozen copy of these 74 items for both arms.
   Re-resolving them on an anchor-carrying snapshot would let anchors define
   their own ground truth.
2. `cross_lingual` is an *agreement* metric — jargon query versus paraphrase
   query under one retrieval configuration — not ground truth. Its absolute
   value should never be read as "how often the right node was found".

Build order, which is what actually enforces this:

```bash
# 1. freeze the snapshot (anchor-free), 2. build the goldset on it, 3. only then backfill anchors
python3 -m living_memory.retrieval_harness build-goldset \
    --snapshot SNAP --cutoff 2026-07-15T00:00:00Z --seed 20260817 \
    --seed-queries artifacts/harness/seed-queries.json --out GOLDSET --build-report BUILD
# then splice the 74 curated items from artifacts/harness/goldset.jsonl over the rebuilt ones
```

## 5. The leak rule

Absolute, and what makes the run evidence rather than theatre:

1. **Anchors come only from events with `created_at` strictly before the
   cutoff.** `scripts/backfill_query_anchors.py --until 2026-07-15T00:00:00Z`,
   into a copy of the snapshot. Never the live database.
2. **Scored queries come only from at or after the cutoff** — enforced by
   `build-goldset --cutoff`, which selects `created_at >= cutoff`.
3. **A timestamp does not bind the curated strata.** 74 of 234 items
   (`cross_lingual` 38, `role_query` 36) carry `source_event_id: null`, so rule
   2 binds only `content_grounded`. For them the leak test is the *fingerprint*
   test below, and it must be run: the role_query item `EZ-13871 ревью задачи`
   *is* a recorded consumed query (events on 2026-08-17). It is post-cutoff
   today at every candidate cutoff, and it would silently become a leaked item
   at any cutoff after 2026-08-17.
4. **Report the fingerprint-disjoint holdout separately from exact repeats.**
   An item is an exact repeat when `recall_fingerprint(query, scope)` is
   already a pre-cutoff anchor. At cutoff 2026-07-15 on the shipped goldset
   that split is 131 exact repeats (@1 **1.000**) versus 103
   fingerprint-disjoint items (@1 **0.049**) — the same metric, off by a factor
   of twenty. A headline number that mixes them measures memorization. After
   regeneration the exact-repeat set should be empty for `content_grounded`;
   report it anyway, as the check that it is.
5. **Both arms on the same snapshot and the same goldset file**, differing only
   in whether the anchor tables are populated.

## 6. Which targets are structurally reachable

`structural` assumes every anchor-reachable miss becomes a hit and nothing
regresses — a bound no implementation can beat. `matched` replaces the oracle
with the measured top-10 nearest-anchor match, which is what the entry design
can see. Both at cutoff 2026-07-15, grounded edges, over the anchor-free
baseline run.

| target (goal) | baseline | structural ceiling | matched ceiling | verdict |
| --- | --- | --- | --- | --- |
| `cross_lingual` hit@5 ≥ 0.210 (double 0.105) | 0.105 | 0.605 | **0.184** | **not reachable** as specified |
| `role_query` hit@1 > 0.111 | 0.111 | 0.250 | 0.194 | reachable, ≤ 3 items |
| overall hit@5 ≥ 0.577 | 0.577 | — | — | **restate as a delta** |
| overall MRR ≥ 0.354 | 0.354 | — | — | **restate as a delta** |

- **`cross_lingual` doubling: no.** The edges exist for 22 of 38 items (oracle
  0.579), but the nearest-anchor match surfaces them for 4 (0.105). Of the 34
  items the baseline misses at 5, anchors can rescue 19 in principle and **3**
  at the measured match rate — 0.105 → 0.184, short of 0.210. And 0.184 is
  itself optimistic: it only asks whether the anchor is in the top ten, not
  whether the injected seed then ranks into the final top five. The honest
  expectation is **+2 to +3 items of 38**.
- **`role_query` hit@1: yes, barely.** 5 of 32 missed items are anchor-reachable
  at all; 3 at the measured match rate. 0.111 → 0.194 is an increase, so the
  target as literally written ("grows from 0.111") is met — but by three items.
- **The overall floors are not raise targets**, they are no-regression floors,
  and on a regenerated goldset the literal numbers 0.577 / 0.354 do not exist.
  Restate as: *overall hit@5 and MRR with anchors are not below the anchor-free
  arm on the same goldset*, with the arm-A numbers recorded.
- **Statistical power is the real limit.** `cross_lingual` has 38 items and
  `role_query` 36. "Doubling 0.105" means moving 4 items; the measured
  achievable move is 3. A 3-item swing on 38 is not distinguishable from noise.
  `seed-queries.json` records that 60+ cross_lingual pairs were authored and
  filtered down to 38 — **enlarging the curated strata is a precondition for
  these targets to be evidence at all**, and it is cheap compared to another
  scored run whose result cannot be read.

## 7. Reproduction of the parent probe, and the cost of the grounding filter

The optimistic probe (anchors = all pre-2026-06-10 consumed fingerprints, edges
= every recorded result, no grounding filter, scope-gated) reproduces exactly:

| stratum | probe @1 / @3 / @10 | reproduced unfiltered | grounding-filtered |
| --- | --- | --- | --- |
| content_grounded | 0.031 / 0.056 / 0.113 | **0.031 / 0.056 / 0.113** | 0.044 / 0.081 / 0.119 |
| cross_lingual | 0.000 / 0.079 / 0.158 | **0.000 / 0.079 / 0.158** | 0.000 / 0.000 / 0.026 |
| role_query | 0.083 / 0.111 / 0.139 | **0.083 / 0.111 / 0.139** | 0.028 / 0.083 / 0.083 |

(The probe's 4,589 anchors vs 4,580 here are the same set: 9 consumed events
recorded zero results and cannot carry an edge.)

The grounding filter strips **7.2× the edges** and 2.2× the anchors at cutoff
2026-07-15 (56,507 → 7,879 edges; 6,904 → 3,178 anchors) — more than the 4–5×
the goal assumed. It costs `cross_lingual` the most (@10 0.158 → 0.026 at
2026-06-10) and *helps* `content_grounded` at @1/@3, which is what a precision
filter should do. The filter stays: unfiltered edges are the 87.9% vacuous
reinforcement the previous goal removed from the live loop, and re-admitting
them offline would cement exactly that noise into the graph.

## 8. The three structural reasons the numbers are what they are

1. **The training window barely contains the traffic the premise is about.**
   Before 2026-06-10 the history is 26 days and **3.3% Cyrillic** (178 of
   5,385); after it, **16.3%** (606 of 3,723). "The operator repeats himself in
   his own jargon" is a property of *recent* traffic. Moving the cutoff to
   2026-07-15 raises in-window Cyrillic to 391 queries and moves
   `cross_lingual` oracle reach from 0.184 to 0.579 — the single largest lever
   found, and the reason the cutoff is not 2026-06-10.
2. **`cross_lingual` scores agreement, not truth, and its target moves.**
   Relevance is the paraphrase's own top-k (`retrieval_harness.py:1568`).
   Measured instability: 14 of 38 labels changed across one day of new memory.
   Freezing the items makes the A/B valid; nothing makes the metric ground
   truth.
3. **74 of 234 items carry no timestamp** (`source_event_id: null`), so a
   per-item temporal cutoff binds only the 160 `content_grounded` items. The
   fingerprint-disjointness check (§5.4) is not a nicety — it is the *only*
   leak test that applies to the two strata carrying the goal's targets.

## 9. What this means for the goal

The mechanism is worth building: 3,450 anchors and 8,335 grounded edges exist,
the scope key is safe, and at the right cutoff the edges reach a relevant node
for 58% of `cross_lingual` items. What is *not* supported is the acceptance
contract as written. Before the scored run, the goal's numeric targets should be
restated as:

- `cross_lingual` hit@5: **≥ 0.158** (0.105 + 2 items) on the frozen curated
  set, reported with the oracle 0.579 alongside, so the gap between "edge
  exists" and "match finds it" is visible and is the next measurement's target.
- `role_query` hit@1: **> 0.111**, with the item count stated, not the rate
  alone.
- overall hit@5 / MRR: **not below the anchor-free arm on the same goldset**.
- and the curated strata grown beyond 38/36 before any of these is read as
  evidence.
