# Query anchors: the graph's entry from query space — result

The graph had no entry from query space: BFS could only be seeded by nodes that
bm25, the vector channel or a schema trigger had already found. This work turns
every grounded recall consumption into an **anchor node** — the operator's own
question, embedded with the same encoder as content, in the consuming event's
scope, with edges to exactly the nodes that consumption grounded on — and lets
an incoming query match those anchors and hand their edges to the existing
graph channel as seeds.

**The mechanism is built, tested, backfilled onto the live database and
evaluated leak-free. It works, and it does not yet do what it was built for.**
On a temporal holdout it helps where the query repeats a question the anchor
corpus was trained on, and nowhere else: on the 342 fingerprint-disjoint items
it improved 0 and regressed 2. Three of the acceptance bars are **not met** —
the two target strata and the fresh-node gate — and each is published below next
to the ceiling that bounds it, with the structural cause named. **No constant,
weight, floor or cap was changed to move a number.**

Records: [`artifacts/anchors/eval.md`](artifacts/anchors/eval.md) (the scored
run), [`deploy-log.md`](artifacts/anchors/deploy-log.md) (the live backfill),
[`calibration.md`](artifacts/anchors/calibration.md) (the match entry),
[`latency.json`](artifacts/anchors/latency.json),
[`floor-fix.json`](artifacts/anchors/floor-fix.json),
[`live-verify.json`](artifacts/anchors/live-verify.json).

---

## 1. Acceptance against the original goal

| # | What the goal demands | Bar | Measured | Verdict |
|---|---|---|---|---|
| 1a | Temporal holdout, anchors built **only** from pre-cutoff events | 0 leaks | newest anchor **and** newest edge write `2026-07-14T13:57:33Z`; 0 of 3,071 anchors and 0 of 7,510 edges at/after the 2026-07-15 cutoff | **PASS** |
| 1b | `cross_lingual` hit@5 at least doubles from 0.105 | ≥ 0.210 | **0.1053 → 0.1053 (+0.0000)** — ceiling 0.1053, *below the bar* — §4 | **NOT MET** |
| 1c | `role_query` hit@1 rises above 0.111 | > 0.111 | **0.1111 → 0.1111 (+0.0000)** — reachability at the shipped entry **0.0000** — §4 | **NOT MET** |
| 1d | Overall hit@5 / MRR not below 0.577 / 0.354 | ≥ | **0.5812 / 0.3596** with anchors vs 0.5769 / 0.3540 without | **PASS** |
| 2 | Fresh-node visibility slice not worse than the anchor-free arm | ≥ | 57 items: hit@5 **0.6842 → 0.6667**, MRR 0.3775 → 0.3765; 0 better, **1 worse**, 56 unchanged — §6 | **NOT MET** |
| 3 | Regression test that fails on the old code (jargon query, anchor-only node) | test | `test_jargon_query_reaches_anchor_only_node` — rank **8 → 5**, guarded by `without_matches_reference` so "fails on the old code" cannot go vacuous — §2 | **PASS** |
| 4 | Edge-migration test (supersede → replacement) and anchor-dedup test | tests | 5 migration tests + 3 dedup tests, named in §2 | **PASS** |
| 5 | p50 `memory_recall` not worse than +5 ms; scan and hop figures published | ≤ +5 ms | anchor scan **0.560 ms** p50; paired **+3.777 ms**, traffic-weighted **+2.907 ms**; pooled **+7.093 ms exceeds** — §8 | **PASS** (paired / traffic-weighted); pooled reported as exceeding |
| 6 | Retro backfill on the live DB vs the ~3.7k / ~8.5k estimate; verify clean | counts | **3,337 anchors / 7,942 edges** (−9.8% / −6.6%); verify exit 0, every gate **0** — §9 | **PASS** |
| 7a | Existing `tests/` green | all | **971 passed, 83 skipped** (`bash scripts/test.sh`, exit 0) | **PASS** |
| 7b | Deploy by runbook with restart and WRITE smoke; `alt` untouched | done | **NOT DONE — operator step after merge.** The code is unmerged and `/home/sfx/p/lm` is read-only from the execution worktree; the runbook is staged and the restart is argued mandatory — §10. `alt` untouched. | **PENDING** |
| 8a | Trigger-channel contribution after anchors + numeric boost recommendation | numbers | trigger-unique **0 of 22** on `role_query`, **0 of 257** overall; removing ×1.8 costs **−0.0046 MRR**; recommendation: **do not retire on these numbers** — §7 | **PASS** |
| 8b | Goldset anchor coverage | stated | **107 of 343 (31.2%)**; per stratum and nearest-anchor cosine distribution — §4 | **PASS** |
| 8c | Goldset regeneration from post-grounding traffic named as the next step | named | §11.3 | **PASS** |

Cold start, which the goal required to be honest rather than measured:
`test_cold_start_ranking_is_byte_identical_to_pre_anchor_code` and
`test_unresembled_query_class_ranks_identically_to_pre_anchor_code` pin that a
query class with no anchor falls back to the existing channels **byte-identically**,
and `test_empty_anchor_corpus_runs_no_anchor_scan` pins that it costs nothing.

---

## 2. What shipped

| Goal item | Where | Pinned by |
|---|---|---|
| Anchor **is** a graph node: query text, embedding via the same encoder path, the **event's** scope, edges to the content-grounded consumption nodes | `src/living_memory/query_anchors.py`, schema **v7** by additive self-guarding DDL | `test_query_anchors.py` (39 tests) |
| Dedup: identical and near-identical questions reinforce one anchor | fingerprint `(scope, whitespace-collapsed query)` + cosine **0.995**, two stages cheapest-first | `test_exact_repeat_reinforces_the_same_anchor_by_fingerprint`, `test_near_identical_query_deduplicates_at_the_cosine_threshold`, `test_dedup_threshold_keeps_same_jargon_different_object_apart` |
| Entry into retrieval: anchor vector scan → matched anchor's edges become graph **seeds**, scored through the existing `graph_score` — **not** a fifth weighted channel | `src/living_memory/retrieval.py` | `test_anchor_only_node_arrives_through_the_graph_channel` (asserts the target mints no trigger score), `test_anchor_seed_opens_one_hop`, `test_anchor_seeds_are_bounded_by_the_graph_seed_limit` |
| Live path: every content-grounded consumption creates or reinforces an anchor; unreinforced anchors fall to the standard decay | `src/living_memory/feedback.py` (`apply_pending_recall_feedback`) | `test_anchor_live_path.py` (11 tests): grounded subset only, event scope not trace scope, no anchor under `policy=all`, one batched vector per consumption, a failed anchor never costs the trace its credit |
| Retro backfill, resumable, backup-gated, with `verify` | `scripts/backfill_query_anchors.py`, runbook `docs/query-anchors-migration.md` | `test_backfill_query_anchors.py` (26 tests) incl. SIGKILL-mid-run resume equivalence and refusal to run without a valid backup |
| Edge migration: an anchor edge follows `supersedes` and era eviction, so it never leads into the past | `query_anchors.resolve_replacement` + write hook + repair sweep | `test_teach_supersede_moves_the_anchor_edge_to_the_replacement`, `test_supersedes_chain_leaves_edges_on_the_latest_node`, `test_era_exclusions_move_edges_for_the_reasons_that_name_a_replacement`, `test_migration_refuses_self_loops_cycles_and_dead_destinations`, `test_sweep_repairs_edges_written_before_the_hook_and_is_idempotent` |

Two defects found by the first scored run were fixed and re-measured:

- **The anchor demoted the node it seeded.** The graph-weight floor is a step
  function at `graph_score > 0`, so an anchor giving a strong lexical candidate a
  *small* graph score moved a quarter of its blend off the evidence carrying it —
  54 results lost score, median 8.1%. The pre-anchor walk now runs first and
  alone, the anchor walk scores separately, and each touched candidate keeps the
  better of the two blends. Across all 343 holdout items **no candidate now
  scores lower with anchors on**. `test_an_anchor_seed_never_lowers_a_candidates_score`
  fails on the reverted code, and `test_the_fixture_still_provokes_the_demotion_on_the_pre_fix_code`
  stops that assertion from going vacuous.
- **The match floor was uncalibrated.** `ANCHOR_MATCH_COSINE_THRESHOLD` shipped at
  0.80 as a documented default and fired on 3.8% of held-out queries. Calibrated
  to **0.60** on a nested temporal split — anchors before 2026-06-10, calibration
  queries from [2026-06-10, 2026-07-15) — proven disjoint from every scored item
  (0 collisions on fingerprint / normalized query / event id, max cosine 0.9258),
  with the publisher refusing to write if that guard trips.

---

## 3. The scored run

| what | value |
|---|---|
| snapshot | sha256 `5201b10682fcc0e5…` — the anchor-free base plus the anchor tables, nothing else |
| cutoff | **2026-07-15T00:00:00Z** |
| anchor corpus | **3,071 anchors / 7,510 edges**, all live, 2.445 per anchor |
| holdout goldset | sha256 `816c365c…`, 343 items (content_grounded 269 regenerated at the cutoff; cross_lingual 38 + role_query 36 spliced in **frozen**) |
| frozen goldset | `artifacts/harness/goldset.jsonl`, sha256 `af40cb0d…`, 234 items |
| match entry | `ANCHOR_MATCH_COSINE_THRESHOLD = 0.60`, `ANCHOR_MATCH_LIMIT = 5` — applied unchanged from calibration |
| arms | one snapshot, `--no-anchors` the only difference; live-agreement 1.000, 0 divergences, in both arms |

The live database was never written by the scored run; every read happened on a
working copy.

### Holdout, 343 items, `without -> with`

| bucket | items | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| **overall** | 343 | 0.2507 → 0.2507 | 0.6122 → 0.6122 | 0.7493 → 0.7493 | 0.4035 → **0.4040** |
| content_grounded | 269 | 0.3048 → 0.3048 | 0.7063 → 0.7063 | 0.8476 → 0.8476 | 0.4708 → 0.4714 |
| **cross_lingual** | 38 | 0.0000 → 0.0000 | **0.1053 → 0.1053** | 0.1842 → 0.1842 | 0.0448 → 0.0448 |
| **role_query** | 36 | **0.1111 → 0.1111** | 0.4444 → 0.4444 | 0.6111 → 0.6111 | 0.2797 → 0.2797 |
| tail | 26 | 0.0385 → 0.0385 | 0.3846 → 0.3846 | 0.5385 → 0.5385 | 0.2039 → 0.2039 |

Paired: **1 improved, 2 regressed, 340 unchanged.** Mean MRR delta +0.000481,
95% bootstrap CI [−0.001472, +0.003397]; mean hit@5 delta 0.000000, CI
[−0.008746, +0.008746]. `role_query`: **0 of 36 ranked lists changed at all.**

### Frozen goldset, 234 items — the required floor, met

| bucket | items | hit@1 | hit@5 | MRR |
|---|---|---|---|---|
| **overall, without anchors** | 234 | 0.1880 | **0.5769** | **0.3540** |
| **overall, with anchors** | 234 | 0.1923 | **0.5812** | **0.3596** |
| exact repeat | 129 | 0.2558 → 0.2636 | 0.7597 → 0.7674 | 0.4535 → 0.4636 |
| fingerprint-disjoint | 105 | 0.1048 → 0.1048 | 0.3524 → 0.3524 | 0.2317 → 0.2317 |

Paired: **3 improved, 0 regressed, 231 unchanged.** The anchors-off arm
reproduces the phase-1 numbers exactly, so the comparison is valid. Re-run
independently during final verification on the same three shas: **0.5812 /
0.3596**, identical.

All 3 improved items are exact repeats and the 105 fingerprint-disjoint items
are byte-identical between arms. This goldset moves because 129 of its 234 items
are exact repeats; the holdout does not, because it has 1.

### The two rank regressions are displacement, not demotion

| item | relevant rank | cause |
|---|---|---|
| `content_grounded-88849ff6a5bc` | 2 → 3 | `01KT9N1N…` 0.9061 → 1.0664 (gained graph from an anchor seed), rank 6 → 2. No candidate lost score. |
| `content_grounded-b7206fc167d1` | 5 → 7 | `01KT41WX…` 0.7868 → 1.1327, rank 8 → 2, plus a newcomer at rank 3. No candidate lost score. |

*"An anchor never lowers a candidate's score"* is an invariant a fix can
guarantee, and it holds. *"An anchor never lowers a relevant node's rank"* is
not, once the mechanism may promote anything. These two items measure **edge
quality**, not scoring.

---

## 4. The two bars that are not met, with the ceilings that bound them

### `cross_lingual` hit@5 ≥ 0.210 — measured **0.1053**, unchanged

| quantity | value |
|---|---|
| bar | **0.210** |
| measured, both arms | **0.1053** |
| ceiling stated by the goal (no-floor top-10) | **0.184** |
| ceiling measured on this snapshot (no-floor top-10) | **0.1053** |
| oracle — any in-scope anchor, no k, no floor | 0.6316 |

**The ceiling is below the bar in both accountings.** The goal's own figure
(0.184) is under 0.210, and re-measuring it here puts it lower still at 0.1053 —
the share of items for which an anchor holding a live edge to a relevant node is
among the 10 nearest by cosine with **no floor at all**. The bar was not
reachable by this mechanism on this corpus before the run started.

**Structural cause: cosine cannot retrieve the anchor that exists.** The oracle
says the corpus does hold a qualifying anchor for 24 of 38 items (63%), but it
is never among the 10 nearest — a Russian paraphrase and its English source
query sit 0.35–0.66 apart in `paraphrase-multilingual-MiniLM-L12-v2` (measured
max over the whole stratum: **0.663**). The goal's premise — the operator's
queries share one language and jargon, so query↔query beats query↔content —
holds *within* a language; this stratum is cross-language pairs by construction,
the one case the premise excludes.

### `role_query` hit@1 > 0.111 — measured **0.1111**, unchanged

| quantity | value |
|---|---|
| bar | **> 0.111** |
| measured, both arms | **0.1111** |
| ceiling stated by the goal | **0.194** ("structurally reachable") |
| anchor reachability at the shipped entry (top-5, ≥ 0.60) | **0.0000** |
| ceiling measured on this snapshot (no-floor top-10) | **0.1111** |
| oracle — any in-scope anchor | 0.1667 |

**The "structurally reachable" assessment does not survive measurement.**
Anchors matched 7 of 36 `role_query` items — up from 1 — and changed **0
rankings**, because reachability of a relevant node through an anchor edge at the
shipped entry is **0.0000**. Unfloored at top-10 it is 0.1111, and the oracle
over the entire corpus is 0.1667: the anchor corpus barely contains the answer
for this stratum at all.

### What now bounds both — anchor edge yield, not the match floor

The floor was genuinely binding and no longer is. Coverage, holdout goldset:

| stratum | items | matched @0.80 (previous) | matched @0.60 (this run) | share |
|---|---|---|---|---|
| content_grounded | 269 | 12 | **96** | 35.7% |
| **cross_lingual** | 38 | 0 | **4** | 10.5% |
| **role_query** | 36 | 1 | **7** | 19.4% |
| all | 343 | 13 (3.8%) | **107** | **31.2%** |

Live match rate over 384 distinct holdout queries went **0.52% → 37.76%**. And
quality did not follow:

| stratum | matched | of which an anchor edge reaches a relevant node | yield |
|---|---|---|---|
| content_grounded | 96 | 39 | 40.6% |
| **cross_lingual** | **4** | **0** | **0.0%** |
| **role_query** | **7** | **0** | **0.0%** |
| all | 107 | 39 | 36.4% |

**Zero of the 11 matched items on the two target strata has an anchor edge into a
relevant node.** The anchor fires, brings its targets, and none of them is an
answer. No threshold, limit or weight can change that — the edges point
elsewhere. This is the honest replacement for the previous run's "the 0.80 floor
is the constraint".

Nearest in-scope anchor cosine per eval query, unfloored:

| stratum | min | p50 | p90 | max | ≥ 0.60 |
|---|---|---|---|---|---|
| content_grounded | 0.341 | 0.558 | 0.710 | 1.000 | 96 |
| **cross_lingual** | 0.351 | 0.455 | 0.589 | **0.663** | 4 |
| role_query | 0.354 | 0.529 | 0.697 | 0.912 | 7 |

Scope is not the constraint: every eval query has an eligible anchor pool.

Two further limits belong on the record. `cross_lingual` scores *agreement*, not
truth — its relevance labels are the paraphrase query's own top-k, and 14 of 38
moved across one day of new memory when rebuilt; freezing them makes the A/B
valid, nothing makes them ground truth. And at 38 and 36 items the smallest
resolvable move is 0.026 / 0.028 while the effect being hunted is 2–4 items: the
run has no power to resolve what it was asked to resolve.

---

## 5. The generalization claim lives only on the fingerprint-disjoint subset

| bucket | items | hit@1 | hit@5 | MRR |
|---|---|---|---|---|
| exact repeat | **1** | 0.0000 → 0.0000 | **0.0000 → 1.0000** | 0.1111 → **0.5000** |
| **fingerprint-disjoint** | **342** | 0.2515 → 0.2515 | **0.6140 → 0.6111** | **0.4044 → 0.4037** |
| content_grounded : disjoint | 268 | 0.3060 → 0.3060 | 0.7090 → 0.7052 | 0.4721 → 0.4713 |
| cross_lingual : disjoint | 38 | 0.0000 → 0.0000 | 0.1053 → 0.1053 | 0.0448 → 0.0448 |
| role_query : disjoint | 36 | 0.1111 → 0.1111 | 0.4444 → 0.4444 | 0.2797 → 0.2797 |

**The entire aggregate gain is the single exact-repeat item.** On the 342
fingerprint-disjoint items anchors improved **0 and regressed 2**: −0.0029 hit@5,
−0.0007 MRR. The same shape appears on the frozen goldset, where all 3 gains are
exact repeats and the 105 disjoint items are identical between arms.

Stated plainly: **on this evidence the mechanism helps where the query is a
repeat of a query it was trained on, and nowhere else.** That is weaker than the
claim the goal set out to establish, and it is the claim the disjoint subset
supports.

---

## 6. Fresh-node visibility — the gate fired

The lower match floor took this slice from **6 items to 57** (83 fresh nodes), so
the "rich get richer" pressure it guards against became observable.

| arm | items | fresh nodes | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|---|
| without anchors | 57 | 83 | 0.1754 | **0.6842** | 0.8070 | **0.3775** |
| **with anchors** | 57 | 83 | 0.1754 | **0.6667** | 0.8070 | **0.3765** |

Per item, first fresh-node rank: **0 better, 1 worse, 56 unchanged.** The one
that moved is `content_grounded-b7206fc167d1`, rank **5 → 7** — across the hit@5
boundary, which is the entire −0.0175. Cause confirmed directly: two anchor
targets (`01KT41WX…`, `01KT3QKN…`) were promoted above the fresh node, and **no
candidate lost score**. This is the anchor preferring what it already knows over
something new — precisely the contour the slice exists to detect.

The criterion is "not worse than the anchors-off run", so **it is not met.** Its
magnitude is one item of 57, the mechanism-level guarantee still holds in
`tests/test_anchor_visibility_slice.py`, and on the frozen goldset the same slice
is 12 items and identical in both arms. But the slice is reporting a real effect
for the first time; at 6 items it could not have seen this, and any change that
raises coverage further must re-read it.

---

## 7. The trigger channel after anchors

Measured on `role_query`. Anchors matched 7 of 36 items and changed 0 rankings,
so these numbers are **identical in both arms and on both goldsets**.

- trigger present in the first-relevant hit: **6 of 22 (27.3%)**; graph: 18 of 22 (81.8%)
- **trigger ∩ graph = 5**; trigger without graph = 1
- **trigger-unique = 0 of 22.** Overall across 343 items: trigger-unique **0 of 257**
- **anchor ∩ trigger overlap: 3 items** — half the trigger-carrying hits are also
  anchor-matched, but since anchors changed no `role_query` ranking the overlap is
  latent, not competitive. It is the number to re-read once edge yield improves,
  because that is when the two would begin duplicating work.
- **164 of 433 active schemas (37.9%)** carry triggers of ≥ 5 distinct tokens
  (reproducing the goal's ~39%), and the channel needs ≥ 50% token overlap, so a
  5-token trigger needs 3 of its tokens in the query.

### Recommendation on the 0.95 × 1.8 boost — do not retire it on these numbers

| configuration | hit@1 | hit@5 | MRR |
|---|---|---|---|
| as shipped | 0.1111 | 0.4444 | 0.2797 |
| **`SCHEMA_TRIGGER_BOOST` ×1.8 removed** | 0.1111 | 0.4444 | **0.2751 (−0.0046)** |
| every trigger-only result dropped (upper bound on removing the channel) | 0.1111 | **0.4722 (+0.0278)** | 0.2820 (+0.0023) |

**This reverses the previous run's recommendation.** That run measured the ×1.8
removal at exactly 0 cost on all three metrics and recommended retiring it. On
this build the cost is 0 items at hit@1 and hit@5 but **−0.0046 MRR**: the
multiplier does lift a relevant result within the top-5 on at least one item.
Retiring it is *nearly* free, not free. If it is retired, set
`SCHEMA_TRIGGER_BOOST = 1.0`, re-run both arms on both goldsets, expect −0.0046
MRR on `role_query` and nothing else, and treat any other change as evidence the
counterfactual missed something.

Bounds: the `0.95 + 0.05 × overlap` **base** cannot be undone from recorded
scores (it enters through `base_score = max(weighted_sum, trigger_score)` and the
report does not carry `weighted_sum`), and n = 36, where −0.0046 MRR is one item
moving one rank.

**Do not remove the channel.** Its unique contribution to first-relevant hits is
0, but that measures ranking of what retrieval returned; the trigger channel is
also a *candidate producer*, and 1 of 22 `role_query` hits (`graph+trigger`)
reached the list with no bm25 and no vector evidence at all.

---

## 8. Cost

Re-measured on this build with **`scripts/anchor_latency_bench.py`** (landed in
the repo so the figure is reproducible): one store with `anchor_seeding`
toggled, arms alternated every iteration, paired per-query medians,
`log_access=False`/`log_event=False`, 13 iterations and 2 warmups per query,
against the real 3,071-anchor / 7,510-edge population.

| figure | value | budget |
|---|---|---|
| anchor scan, p50 / p95 / max | **0.560 / 1.036 / 1.126 ms** | — |
| anchor scan, no anchor matched | 0.436 ms p50 | — |
| anchor scan, anchor matched | 0.688 ms p50 | — |
| **paired `memory_recall` p50 delta, all strata** | **+3.777 ms** | +5 ms → **PASS** |
| paired p50, unmatched holdout | +1.111 ms | +5 ms |
| paired p50, matched holdout | +5.867 ms | — |
| paired p50, repeat worst case | +5.210 ms (p95 +22.95, max +36.02) | — |
| **traffic-weighted p50 delta** (37.76% matched) | **+2.907 ms** | +5 ms → **PASS** |
| **pooled p50 delta** | **+7.093 ms** | +5 ms → **EXCEEDS** |
| pooled p95 delta | −3.573 ms | — |

**The budget holds on the paired and traffic-weighted measures; the pooled
measure exceeds it.** That is reported rather than hidden. The pooled row's query
mix is deliberately adversarial — 26 of 42 measured queries carry a matching
anchor, because the `repeat` stratum is every-query-has-an-anchor by
construction, against 37.76% of real distinct holdout queries. The paired figure
answers "what did anchors cost this query"; the traffic-weighted figure is what
an operator would see.

The hop is not overhead: a matched anchor opens a BFS from seeds no other channel
produced, and anchor seeds compete for the same `GRAPH_SEED_LIMIT` (50) slots as
every other seed, so root count is unchanged.

---

## 9. The retro backfill on the live database

Run 2026-08-18 against `~/.local/share/living-memory/global.sqlite3` following
`docs/query-anchors-migration.md` steps 0–2 and 6. The server stayed **up**
throughout (`boot_id d291137f…`); no code was deployed and nothing was restarted.

| quantity | value |
|---|---|
| consumed recall events in range | 9,122 (9,087 processed; 35 have no consuming trace left) |
| grounded consumptions | **3,682 — 40.5%** (the goal's 600-event sample estimated 41.3%) |
| **anchors written** | **3,337** (324 reinforcements, 101 merged by cosine) |
| **anchor edges written** | **7,942** rows from 8,436 writes — **2.38** per anchor |
| vs the goal's estimate (~3,700 / ~8,500) | **−9.8% / −6.6%** |
| edge-migration sweep | 2,111 targets examined, 0 moved |
| `SQLITE_BUSY` retries under the live server | 0 |
| database growth | +8.6 MB (anchors 6.54 MB, edges 0.87 MB) |
| wall time | 65.9 s including the backup |

The shortfall is fully accounted, not residual: 101 questions with distinct
`(scope, fingerprint)` identities but near-identical vectors merged into existing
anchors, 21 grounded events had no live target left, and on the edge side
8,699 grounded targets − 263 decayed with no heir − 494 that accumulated weight on
an existing `(anchor, target)` pair = 7,942, confirmed by `sum(hits) = 8,436`.

**`verify` exits 0 with every gate zero:** `edges_to_missing_nodes: 0`,
`edges_to_superseded_nodes: 0`, `anchors_outside_source_scope: 0`,
`anchors_without_edges: 0`, `anchors_with_unusable_embedding: 0` (all 384-dim),
`duplicate_identities: 0`.

**The node graph was not touched, checked from outside the process:** SHA-256
over ordered projections of `nodes`, `connections` and `recall_events`, compared
between the pre-backfill backup and the live database afterwards — identical on
all three (16,734 / 143,340 / 54,824 rows, delta 0), `nodes.updated_at` moved on
0 rows. The backup was taken through the sqlite backup API, not `cp`: the live
file carried a **503 MB WAL** at the time, which a copy would have missed.

---

## 10. Deployment — what remains an operator step

**Not performed, by contract.** The code is unmerged and the shared checkout
`/home/sfx/p/lm` is read-only from the execution worktree, so the deploy, the
restart and the WRITE smoke test are staged in
[`artifacts/anchors/deploy-log.md`](artifacts/anchors/deploy-log.md) to run from
merged master. Until they do, **the anchors sit in the database unread** — the
running process holds the old modules.

The restart is **mandatory, not advisable**, on two independent grounds:

1. `MemoryStore` resolves `_anchor_tables_cache` once per process
   (`storage.py:478-487`). The server booted *before* the anchor tables existed,
   so it has cached `False`: every anchor read degrades to "no anchors" and every
   anchor write raises. Only a new process changes that.
2. The deploy also changes retrieval behaviour — `ANCHOR_MATCH_COSINE_THRESHOLD`
   0.80 → 0.60 and the graph-floor fix are module-level and process-resident.
   Measured effect of the threshold alone: live match rate 0.52% → 37.8%.

The smoke test **must be a write** (`memory_remember` / `memory_teach`): a store
that cached a pre-v7 shape answers reads with "no anchors" and raises on writes,
so a read-only recall smoke passes while writes are broken. A catch-up backfill
pass follows the restart, because consumptions between the backfill and the
restart were handled by code that wrote no anchors. Rollback is documented and
cheap: the old code ignores both tables entirely.

**`alt` was not touched and is not part of this procedure.** Nothing automated
connects to, deploys to, restarts or backfills that host.

---

## 11. What the measurer does next

1. **Fix edge yield, not the threshold.** 0 of 11 matched `cross_lingual` /
   `role_query` items has an anchor edge into a relevant node. The next
   intervention is on *which nodes an anchor links to* — today the grounded
   subset of the event's results, 2.4 targets per anchor — and on how many
   grounded consumptions an anchor needs before it may seed. Not on the match
   entry, which is calibrated and demonstrably not binding.
2. **Decide what `cross_lingual` is testing.** Oracle 0.632 vs top-10-no-floor
   0.105 is a matcher-recall gap, not a corpus gap. Either the stratum's
   cross-language framing is the wrong test for a query↔query mechanism, or
   anchor matching needs a path that is not raw cosine in this encoder. Decide
   before spending another scored run on it.
3. **Regenerate the goldset from post-grounding traffic.** The live grounded
   write path has run since 2026-08-18 ~18:00; every query it has anchored since
   is traffic the operator asked *after* grounding became the credit signal, and
   none of it is in this run's training window. Rebuild `content_grounded` at a
   cutoff inside that window and **recompute the anchor-free baseline on the new
   file** — the 0.5769 / 0.3540 pair belongs to the 234-item shipped goldset and
   to nothing else.
4. **Grow `cross_lingual` and `role_query` past 38 / 36.** `seed-queries.json`
   records 60+ authored cross_lingual pairs filtered to 38. At 38 items the
   smallest resolvable move is 0.026 and the effect being hunted is 2–4 items —
   cheaper than another scored run whose result cannot be read.
5. **Keep the fresh-node slice as a gate and re-read it after any coverage
   change** (§6).

---

## 12. Boundaries observed

- **No constant, weight, floor or cap was changed to move a number.**
  `feedback_weighted_score`, the channel weights, `minimum_graph` (0.25 / 0.75),
  bm25/FTS, `tokenize`/`_SYNONYMS` and the embedding model are untouched. The
  match entry moved 0.80 → 0.60 once, on a calibration set proven disjoint from
  every scored item, and was applied unchanged downstream.
- **The schema trigger channel was not removed** — its contribution after anchors
  is measured (§7) and the recommendation is numeric and negative.
- **Schema changes are additive, self-guarding DDL** (v6 → v7); a v6 database
  migrates leaving `nodes` and `connections` DDL byte-identical, and anchor reads
  degrade to empty without the v7 tables.
- **Nothing was deployed, no service restarted, and the only write to the live
  database was the sanctioned retro backfill** (§9), whose non-anchor tables were
  verified untouched from outside the process.
- Snapshot experiments went through the sqlite backup API; a copy without `-wal`
  would have been invalid.
