# vector-recall-chunking-harness — result

Raising the quality of the Living Memory vector recall channel: chunked embeddings
instead of a 128-token truncation, float32 BLOB storage instead of JSON, and a
re-calibration of `retrieval_weights` — all gated by an offline end-to-end harness
on real query history.

**Outcome: the phase-1 gate passes on all nine conditions; phase 2 kept the
incumbent weights because the challenger failed its holdout bar. The live database
has NOT been migrated** (see [Live database state](#live-database-state-not-migrated)).

Every number below is quoted with the artifact it comes from. Nothing here was
re-derived or re-run by this node except the test suite and one read-only query
against the live database to establish its current state.

---

## 1. Acceptance at a glance

| Acceptance item | Bar | Measured | Source | Verdict |
|---|---|---|---|---|
| Harness reproducible, baseline frozen | — | 3 runs, `metrics` block byte-identical (3613 B, `e942ed41c9a2dc30…`) | `artifacts/harness/baseline.md` | PASS |
| Live-agreement sanity | agreement on a query sample | 20 of 234 sampled, agreement_rate **1.000**, no divergences (1.0 in all four gate arms too) | `artifacts/harness/baseline.json` → `agreement`; `phase1-gate.json` → `quality.live_agreement` | PASS |
| Goldset hit@5 ≥ baseline | ≥ 0.526 | **0.577** (+0.051) | `artifacts/harness/phase1-gate.json` | PASS |
| Goldset MRR ≥ baseline | ≥ 0.324 | **0.354** (+0.030) | same | PASS |
| Tail subset rises | clear rise over 0.077 | hit@5 **0.385** (2 → 10 of 26), MRR 0.046 → **0.204** | same | PASS |
| Regression test fails on old code | must fail | `2 failed, 1 passed` at `8b93e8d`; `3 passed` at `b267569` | `artifacts/harness/phase1-gate.md` §Regression test | PASS |
| Stored embedding bytes | < 134.6 MB | **92,974,080 B (93.0 MB)** for 3.65× the vectors | `phase1-gate.json` → `performance.embedding_storage` | PASS |
| Cold vector-channel init | < 200 ms | **104.84 ms** median for all 60,530 vectors | `phase1-gate.json` → `performance.cold_vector_init` | PASS |
| `memory_recall` p50 | ≤ baseline + 5 ms | **184.38 ms** vs 192.08 ms — **−7.69 ms**, i.e. faster | `phase1-gate.json` → `performance.recall_latency` | PASS |
| Existing tests green | no new failures | **839 passed, 83 skipped, 0 failed, 0 errors**, rc=0 | `bash scripts/check.sh` at `72eb6c8`, this node, 2026-08-18 | PASS |
| New tests (chunker, BLOB roundtrip, max-pool, write-path invalidation) | present and passing | included in the 839 | `tests/test_chunking.py`, `tests/test_chunk_store.py`, `tests/test_retrieval_chunk_maxpool.py`, `tests/test_backfill_chunk_embeddings.py`, `tests/test_chunk_tail_regression.py` | PASS |
| Weights re-calibrated with rationale | choice justified on holdout | **incumbent kept**: challenger holdout hit@5 0.6934 vs 0.7005, MRR 0.3872 vs 0.4031 | `artifacts/replay/post-chunking-ab.json` | PASS (no change adopted) |
| result.md carries handoff material | channel splits + schema inventory + OOV jargon | §7 below | this file | PASS |

The one estimate that missed: planning predicted ~69 MB of BLOBs from ~45k chunks
(3.53 per node). The real cut produces **60,530 chunks (4.71 per node)** and
**93.0 MB**. The bar was `< 134.6 MB`, so the gate passes, but the ~69 MB
orientation figure was wrong by 35% and is stated here as wrong rather than rounded away.

---

## 2. Baseline — phase 0

Source: `artifacts/harness/baseline.json`, rendered in `artifacts/harness/baseline.md`.
Generated 2026-08-17T18:45:50Z at commit `44fc0a0`, harness v1, embedding backend
`auto` (real `paraphrase-multilingual-MiniLM-L12-v2`, never the hash backend).

- Snapshot `~/.cache/living-memory-harness/snapshot.sqlite3`, sha256
  `d46f1b845adc3d80…`, captured 2026-08-17T17:59:08Z via the SQLite backup API:
  12,862 active nodes, 16,683 nodes, 142,906 connections, 54,777 recall events.
- Goldset `artifacts/harness/goldset.jsonl`, sha256 `af40cb0da26fa78c…`,
  **234 items of which 26 are tail**; cutoff `2026-06-10T00:00:00Z`, seed 0
  (goldset build seed `20260817`).

### Overall and per stratum

| bucket | items | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| overall | 234 | 0.175 | 0.526 | 0.688 | 0.324 |
| stratum:content_grounded | 160 | 0.231 | 0.694 | 0.900 | 0.422 |
| stratum:cross_lingual | 38 | 0.000 | 0.105 | 0.184 | 0.057 |
| stratum:role_query | 36 | 0.111 | 0.222 | 0.278 | 0.167 |
| **tail** | 26 | 0.038 | 0.077 | 0.077 | 0.046 |

### Per channel

| channel | items | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| bm25 | 234 | 0.158 | **0.573** | 0.688 | 0.321 |
| graph | 234 | 0.154 | 0.521 | 0.688 | 0.306 |
| trigger | 234 | 0.162 | 0.526 | 0.688 | 0.315 |
| vector | 234 | **0.175** | 0.538 | 0.684 | **0.333** |

Channel attribution over the 161 items that returned a relevant node at all:
vector 149 (0.925), bm25 103 (0.640), trigger 32 (0.199), graph 31 (0.193). A
result found by two channels counts once for each.

A per-channel bucket re-orders the *same* returned list by one channel's score, so
it bounds that channel's **ranking** power, not its recall. `hit@10` here is a
candidate-coverage ceiling, not a ranking metric: 224 of 234 items return exactly
10 results, so 0.688 = 161/234 is simply "a relevant node was collected at all".

### Live-agreement sanity

20 of the 234 items sampled (seed 0, top-5) against an independently constructed
`MemoryStore` + `MemoryRecallService` pair: **agreement_rate 1.000, no
divergences**. The harness run exits non-zero below 1.0 unless a divergence note
explains it.

### Reproducibility

Three runs over the same snapshot and goldset. The `metrics` block, compared as
raw bytes sliced out of the written JSON (not re-serialized), is 3613 B hashing to
`e942ed41c9a2dc30…` in all three. `agreement`, `definitions` and the 1,152,331-byte
`runs` block are byte-identical too — the per-result channel scores of all 234
queries reproduce exactly, not merely the aggregates. Only `provenance.generated_at`
(and `provenance.cutoff` in run 3) differs.

---

## 3. Post-chunking — the phase-1 gate

Source: `artifacts/harness/phase1-gate.json`, rendered in
`artifacts/harness/phase1-gate.md`. Phase-1 commit `b267569`, pre-change commit
`8b93e8d`. **Verdict: PASS on all nine conditions.**

The comparison is run on the **baseline's own bytes** — the same frozen snapshot,
copied through the backup API and then chunked — so that code change is not
confounded with live-database drift. The control arm (`8b93e8d` on that chunked
snapshot) reproduces the committed baseline's `metrics` block **byte-for-byte**
(3613 B, `e942ed41c9a2dc30…`): adding 60,530 chunk rows changes nothing the
pre-change code sees. Every delta below is attributable to the vector channel.

| bucket | items | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| overall | 234 | 0.175 → **0.188** (+0.013) | 0.526 → **0.577** (+0.051) | 0.688 → **0.726** (+0.038) | 0.324 → **0.354** (+0.030) |
| stratum:content_grounded | 160 | 0.231 → 0.250 (+0.019) | 0.694 → 0.719 (+0.025) | 0.900 → 0.881 (**−0.019**) | 0.422 → 0.444 (+0.022) |
| stratum:cross_lingual | 38 | 0.000 → 0.000 | 0.105 → 0.105 (**0.000**) | 0.184 → 0.184 | 0.057 → 0.045 (**−0.013**) |
| stratum:role_query | 36 | 0.111 → 0.111 | 0.222 → **0.444** (+0.222) | 0.278 → 0.611 (+0.333) | 0.167 → 0.280 (+0.113) |
| **tail** | 26 | 0.038 → 0.038 | 0.077 → **0.385** (+0.308) | 0.077 → **0.538** (+0.462) | 0.046 → **0.204** (+0.158) |

Per-channel buckets:

| channel | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|
| bm25 | 0.158 → 0.179 (+0.021) | 0.573 → 0.568 (**−0.004**) | 0.688 → 0.726 | 0.321 → 0.342 (+0.021) |
| graph | 0.154 → 0.201 (+0.047) | 0.521 → 0.560 (+0.038) | 0.688 → 0.726 | 0.306 → 0.353 (+0.048) |
| trigger | 0.162 → 0.175 (+0.013) | 0.526 → 0.577 (+0.051) | 0.688 → 0.726 | 0.315 → 0.345 (+0.030) |
| vector | 0.175 → 0.167 (**−0.009**) | 0.538 → 0.585 (+0.047) | 0.684 → 0.726 | 0.333 → 0.342 (+0.010) |

### The tail rise, explicitly

**tail hit@5 0.077 → 0.385** — from **2 of 26** items to **10 of 26**, a ×5.0 rise.
**tail MRR 0.046 → 0.204** (×4.4). **tail hit@10 0.077 → 0.538** (2 → 14 of 26).

The hit@10 line is the mechanism, not a bonus metric. At baseline, tail hit@10
*equalled* hit@5: in 24 of the 26 tail items the relevant node never entered the
returned candidate list at all, so no re-ranking could ever have saved it — this
was a candidate-collection failure. After chunking, 14 of 26 tail items collect a
relevant node, because the post-128-token content is now in a vector that can match.
That is precisely the failure this phase existed to remove. Across the whole
goldset, items with a relevant node anywhere rise **161 → 170 of 234**.

`role_query` doubles its hit@5 (8 → 16 of 36) because most tail items live there.

### Item movement

| stratum | items | better | worse | unchanged | newly found | lost entirely |
|---|---|---|---|---|---|---|
| content_grounded | 160 | 43 | 25 | 92 | 3 | 6 |
| cross_lingual | 37 | 1 | 6 | 30 | 1 | 1 |
| cross_lingual/tail | 1 | 0 | 0 | 1 | 0 | 0 |
| role_query | 11 | 0 | 1 | 10 | 0 | 0 |
| role_query/tail | 25 | 14 | 1 | 10 | 13 | 1 |
| **total** | 234 | **58** | 33 | 143 | 17 | 8 |

### What got worse (four sub-threshold regressions, none of them a gate condition)

1. **`cross_lingual` MRR 0.057 → 0.045.** hit@1/5/10 all unchanged; six items
   shift by one to three ranks, one newly found at rank 10. The stratum's real
   problem — a Russian jargon query scores cosine 0.20–0.39 against an English
   node versus 0.75–0.83 for normal lexicon — is a vocabulary problem, untouched
   by windowing, as predicted.
2. **`content_grounded` hit@10 0.900 → 0.881.** Three items had their relevant
   node at rank 7–10 and lost it out of a 10-result list to newly stronger
   candidates. Overall hit@10 still rises.
3. **bm25 bucket hit@5 −0.004 (one item), vector bucket hit@1 −0.009 (two items).**
   Buckets re-order a list that itself changed, so these move for two reasons at
   once. Both buckets' MRR rises.
4. **One tail item lost its hit** — `role_query-a486aa8b5f70`, rank 1 → not
   returned. A 4,951-char `level:schema` node cut into 20 windows pays
   `0.031 · log2(20) = 0.135` of length-bias correction
   (`LENGTH_BIAS_LOG2_COEFFICIENT`, `src/living_memory/retrieval.py:1333`) while
   its best window gains only ~0.085 over its head: vector score 0.385 → 0.335,
   rank 1 → 2 at `max_results=50`, out of the list at the goldset's
   `max_results=10`. A real cost inside a clear net win, in the same stratum the
   change doubles.

### Drift check and post-drop equivalence

A fresh backup-API snapshot (2026-08-17T20:10:12Z, 12,871 active nodes) was
chunked and **both** revisions re-run on it: overall 0.526 → 0.577 and MRR
0.326 → 0.354, tail 0.077 → 0.385 — identical to the frozen arm to three decimals.
Two hours of live writes move the pre-change arm by less than 0.005 on every
headline metric. The win is not an artefact of which snapshot it was measured on.

Dropping `nodes.embedding` was rehearsed on the fresh chunked snapshot and the
harness re-run with chunk BLOBs as the only vectors in the database: hit@1 0.188,
hit@5 0.577, MRR 0.354, tail hit@5 0.385 — **identical**. Phase 2 and anything
downstream can work on a chunk-only database.

---

## 4. Regression test — fails on the old code, passes now

`tests/test_chunk_tail_regression.py`, recorded in `artifacts/harness/phase1-gate.md`
§"Regression test" and `phase1-gate.json` → `regression_test`.

**What it asserts.** It builds a node whose first 128 model tokens are meeting
logistics and whose answer — a TLS certificate expiry on an internal registry
mirror — starts at character 1,160 of 1,871, around token 246 of 431, and asks for
it **in Russian** so the FTS channel (which indexes the whole node, tail included)
cannot answer: `bm25_score` is asserted to be exactly 0, and the wording avoids
every entry of the Russian synonym table. Eight plausible English neighbours fill
the top 5 when the tail node is invisible. Three assertions:

- the tail node must be in the **top 5** (house rule);
- the delivered `vector_score` must be meaningfully above the **whole-node cosine
  recomputed at test time** — scale-free, and on a single-vector channel that
  ratio is exactly 1.0 by construction, so it cannot pass on the old code;
- a **fixture-validity** check re-ranks the same results by that whole-node cosine
  and requires the node to fall outside the top 5, so the corpus cannot drift into
  one the old channel would also have solved. The premise (where the answer sits
  relative to the encoder window) is computed with the model's own tokenizer, not
  asserted in prose.

**Both outcomes, recorded:**

| revision | result |
|---|---|
| pre-change `8b93e8d` (`git archive` into a clean tree, test file copied in unchanged) | **`2 failed, 1 passed in 9.90s`** — `tail node ranked 9 of 9 (vector_score 0.2208, whole-node cosine 0.2208)`; `assert 9 <= 5` |
| phase-1 `b267569` | **`3 passed in 9.51s`** — same query delivers `vector_score` 0.5393 against the same 0.2208 whole-node cosine, i.e. **2.44×** |

The failing line is the single-vector channel caught in the act: on the old code
the delivered score *equals* the whole-node cosine to four decimals. The premise
test passes on both revisions, as it must.

---

## 5. Storage, cold init, latency, test suite

### 5.1 Stored embedding bytes

| | vectors | bytes | bytes/vector |
|---|---|---|---|
| before — `nodes.embedding` JSON (frozen snapshot) | 16,606 | **134,788,051** (134.8 MB) | 8,116.8 |
| after — `node_chunk_embeddings` float32 BLOB | **60,530** | **92,974,080** (93.0 MB) | 1,536.0 |

- **Bar: < 134.6 MB → PASS at 93.0 MB** — 69.0% of the JSON payload for **3.65×**
  the vectors, i.e. 31% less storage.
- **Chunk count 60,530; chunks per node 4.71** over 12,862 active nodes; the
  largest single node produces **485** chunks. 1,536 B/vector is exactly 384
  float32 with no framing; every row records `dimensions = [384]`.
- Three "before" figures exist and they are not the same number, so all three are
  stated: the goal's bar cites **134,603,140 B / 134.6 MB over 16,583 rows**
  (live DB, 2026-08-17); the gate measured **134,788,051 B over 16,606 rows** on
  the frozen snapshot; the live column read `mode=ro` by this node today
  (2026-08-18) holds **134,900,593 B over 16,620 rows**. The column grows as the
  server writes; the gate's ratio applies to all three.
  Sources: `phase1-gate.json` → `performance.embedding_storage`,
  `docs/chunk-migration.md` §Measured numbers.
- Database file over the whole migration (`performance.database_file_size`):
  532,840,448 → 664,449,024 after backfill → **499,453,952** after
  `drop-embedding-column --vacuum`, i.e. **33.4 MB smaller than before the
  migration started** while carrying 3.65× as many vectors.

### 5.2 Cold vector-channel init

Method: fresh `MemoryRecallService` over an already-open store, corpus load called
directly, no encoder involved; 5 repetitions, fresh service each time.

| | vectors | median | min |
|---|---|---|---|
| after — chunk BLOB matrix, all scopes | 60,530 | **104.84 ms** | 100.99 ms |
| before — JSON column, all scopes (same box, `8b93e8d`) | 12,829 | 700.70 ms | 698.59 ms |

**Bar: < 200 ms → PASS at 104.84 ms** — 6.7× faster while reading 4.7× as many
vectors. Largest single scope after: `project:x` 34.66 ms for 13,861 vectors; a
real recall pays only its plan's scopes.

The goal's bar quoted **817 ms (74 ms fetch + 743 ms `json.loads` over 12,806
rows)** from an earlier measurement. The same operation measures **700.70 ms** on
this box at the pre-change revision, so the gate compares against what the box
actually does. Either way the after-figure is inside the 200 ms bar.

### 5.3 `memory_recall` p50

Method: one process, one store, one service; encoder warmed and one untimed recall
first; then **all 234 goldset queries in `query_id` order × 3 repetitions**, with
`log_access` and `log_event` left on, same snapshot for both arms.
**n = 702 per arm.**

| | p50 | p90 | p99 | mean | min | max | p50 per repetition |
|---|---|---|---|---|---|---|---|
| after (`b267569`) | **184.38** | 644.33 | 855.81 | 292.40 | 58.51 | 1428.86 | 183.0 / 182.9 / 186.4 |
| before (`8b93e8d`) | **192.08** | 662.14 | 912.74 | 300.40 | 52.21 | 1510.90 | 191.3 / 190.0 / 195.4 |

**Bar: ≤ baseline + 5 ms → PASS with margin.** p50 is **−7.69 ms**: the change is
a small improvement, not a cost inside the allowance, and p90/p99/mean move the
same way. The three per-repetition p50s agree within 3.4 ms, so this is not
repetition noise. Encoding the query alone costs 9.78 ms at p50 over the same 702
measurements — the vector channel is a minority of a recall either way, which is
why an 8 ms gap and not a 600 ms one appears here despite the 596 ms cold-init
difference (the pre-change code cached parsed vectors per service instance, so
only its first recall paid the JSON parse).

### 5.4 Test suite

| point | result |
|---|---|
| recorded HEAD baseline, commit `5996314` | **640 passed, 1 failed, 81 errors** — all 82 in `tests/test_replacement_holdout_packet.py` |
| phase-1 gate, commit `b267569` | 788 passed, 82 skipped, 0 failed, 0 errors (two runs, 64.78 s / 64.03 s) |
| **tree tip `72eb6c8`, measured by this node 2026-08-18** | **839 passed, 83 skipped, 0 failed, 0 errors**, `bash scripts/check.sh` rc=0, 63.7 s |

`tests/test_replacement_holdout_packet.py` was red at the recorded baseline for a
pre-existing `path_missing` fixture breakage that has nothing to do with this goal:
its fixture needs a machine-local snapshot that no longer exists and cannot be
recovered. **This tree did not fix those tests and did not make any of them pass.**
One correction to the contract's phrasing, because the honest version differs: the
file *was* touched — the `shared-cause-repair-c1fc3045` sibling (commit `a015d5d`)
added a guard that **skips** those 82 when the snapshot is absent, so they are now
skips rather than a failure and an error storm. Run alone the file reports
`6 passed, 82 skipped` (rc=0). No test in it was rewritten to pass, and the
underlying breakage is still outside this goal's scope.

Net: **+199 passing tests** over the recorded baseline, no new failures, no new
errors.

---

## 6. Retrieval weights — phase 2

Source: `artifacts/replay/post-chunking-ab.json`, rendered in
`artifacts/replay/post-chunking-ab.md`. Commit `0c39c10`.

### Decision: **incumbent weights kept. `src/living_memory/config.py` is unchanged.**

Because recorded per-result scores belong to the pre-chunking distribution, every
number comes from queries **re-run end-to-end through the current code**
(`retrieval_harness.run_goldset` → `MemoryRecallService.memory_recall`) against the
chunk-backfilled snapshot; the regenerated scores are then fed to the *unmodified*
replay A/B machinery. Three disjoint time slices over 3,659 content-grounded items
(C1 = 2026-06-01, C2 = 2026-06-20, on the recall event's own `created_at`):

| slice | events | span |
|---|---|---|
| train | 1,168 | 2026-05-15 … 2026-05-31 |
| eval | 1,369 | 2026-06-01 … 2026-06-19 |
| holdout | **1,122** (bar: ≥ 1000) | 2026-06-20 … 2026-08-17 |

Disjointness is verified as sets in `tests/test_retrieval_weight_recalibration.py`.

**Selection rule, fixed before holdout was read:** finalists = incumbent + config
defaults + uniform + 3 train-fitted trajectories + top-5 explicit and top-5
config-default grid candidates on train; winner = highest **eval** hit@5; adopted
only if on **holdout** its hit@5 ≥ incumbent's AND its MRR ≥ incumbent's. Holdout
scored exactly once.

| stage | winner | incumbent |
|---|---|---|
| train (258 schemes) | `config/explicit_*_g0.15` family at hit@5 0.4084 | 0.3810 — ranked **222 of 258** |
| eval (selection) | **`explicit_b0.75_v0.10_g0.15`** hit@5 **0.5661**, MRR 0.2899 | hit@5 0.5435, MRR 0.2910 |
| **holdout (decides)** | 0.6934 / 0.3872 | **0.7005 / 0.4031** |

| condition | incumbent | eval winner | delta | verdict |
|---|---|---|---|---|
| holdout hit@5 | 0.7005 | 0.6934 | **−0.0071** | FAIL |
| holdout MRR | 0.4031 | 0.3872 | **−0.0159** | FAIL |

The challenger failed both, so **the incumbent stands** — exactly the overfitting
the three-way split exists to catch. Per query over the 1,122 holdout events: the
challenger puts a relevant node in the top-5 where the incumbent does not on **34**
events, the incumbent does so where the challenger does not on **42**; both succeed
on 744, both fail on 302. At rank level the challenger is better on 127, worse on
**216**, unchanged on 779. Two-sided sign tests: **p = 0.4222** on the hit@5
disagreements (indistinguishable) and **p = 2.0e-06** on rank movement (the
challenger is measurably the weaker ranking).

The incumbent is the per-scope learned weights stored in the snapshot's
`retrieval_weights` table, resolved through the live fallback chain:

| scope | bm25 | vector | graph |
|---|---|---|---|
| `global` | 0.1506 | 0.7994 | 0.0500 |
| `project:ae` | 0.1478 | 0.7395 | 0.1128 |
| `project:lm` | 0.1000 | 0.8500 | 0.0500 |
| `project:octopus` | 0.1043 | 0.7757 | 0.1201 |
| `project:online` | 0.1562 | 0.7365 | 0.1073 |
| `project:x` | 0.1000 | 0.8493 | 0.0507 |

**The goal's own premise was refuted by measurement.** Over the 11,964 (event, node)
pairs present both in the recorded row and in the end-to-end re-run, `vector_score`
moved on 79.1% — but **downward**: 2,512 up (21.0%), 6,950 down, 2,502 unchanged;
mean 0.5275 → 0.4935, p50 0.5782 → 0.5503. Max-pool alone would only raise a score,
but the shipped channel subtracts `0.031 · log2(chunk_count)`, which costs an
8-chunk node 0.093 and a 20-chunk node 0.134 — more than max-pool gains for most
nodes. The *conclusion* survives (the distribution the live weights were fitted
against no longer exists, so re-ranking recorded scores would have measured
nothing); only the expected sign of the correction was wrong.

`DEFAULT_RETRIEVAL_POLICY_FLOORS`, `MemoryStore.update_retrieval_weights`,
`MemoryStore.apply_retrieval_weight_floors` and `feedback._method_signals` are
untouched and pinned by `tests/test_retrieval_weight_recalibration.py`.

**Operator re-seed runbook:** `artifacts/replay/post-chunking-ab.md` §"Operator
runbook — re-seeding live per-scope weights (NOT executed)" — six steps, not run.
It matters because config defaults seed only *new* scopes
(`MemoryStore._seed_retrieval_weights` uses `ON CONFLICT(scope) DO NOTHING`), so a
config change alone reaches nothing that exists today; re-seeding must go through
`set_retrieval_weights` + `apply_retrieval_weight_floors`, never raw SQL, which
would bypass the floor pass. Live evidence that these rows are alive: of the 48
scopes in the live table, **3** (`global`, `project:living-memory`, `project:x`)
have drifted from the snapshot's values through ordinary implicit feedback since
the snapshot was frozen. Nothing in this work wrote them.

---

## 7. Handoff for the next goals

### 7.1 Per-channel baseline splits (input to any future retrieval goal)

The tables in §2 and §3 are the frozen instrument: `artifacts/harness/baseline.json`
(pre-change) and `artifacts/harness/phase1-gate.json` (post-change), both over the
frozen 234-item goldset, both reproducible byte-for-byte. Channel attribution moved
with chunking: vector 0.925 → **0.947** of items with a relevant result, graph
0.193 → **0.288** (a max-pooled candidate set feeds the traversal more seeds — part
of graph's apparent strength is borrowed from the vector channel), bm25 0.640 →
0.618, trigger 0.199 → 0.188.

**The blend is no longer the bottleneck.** All 16 finalists in the phase-2 A/B land
inside a 0.009 band of holdout hit@5 (0.6916–0.7005) with the incumbent at the top
of it. The remaining headroom is in *what enters the candidate set*: on holdout,
336 of 1,122 holdout events put no relevant node in the top-5 even under the
incumbent, and the frozen goldset's `cross_lingual` stratum sits at hit@5
**0.1053** against **0.7188** for recorded content-grounded traffic.

### 7.2 `level:schema` inventory with triggers and firing statistics

`artifacts/handoff/schema-inventory.md` (+ `.json`), generated read-only by
`scripts/handoff_inventory.py v1.1.0` with `--as-of 2026-08-17T15:45:00Z`;
`--verify` enforces the determinism contract mechanically.

The rule being measured (`src/living_memory/retrieval.py:437-465`):
`overlap = |tokenize(query) ∩ tokenize(trigger)| / |tokenize(trigger)|`, fires at
`overlap >= 0.5`, scoring `0.95 + 0.05 · overlap`. It only *adds a candidate*;
delivery still depends on the final ranking and `max_results`.

Top findings:

- **433 active schemas**, all carrying `context.trigger` (365 with `task_pattern`,
  315 with `procedure_id`), but only **274 distinct trigger texts** — 159 schemas
  repeat a trigger that already exists. Scope split: `project:x` 188,
  `project:octopus` 94, `project:ae` 89, `project:online` 47, `global` 8,
  `project:lm` 4, `project:gas-stations-ui` 2, `project:online-sitemaps` 1.
- **106 of 433 schemas were never delivered**, and the split is the actionable
  part: **51 did match the trigger rule and still never reached a caller** (they
  need less competition), while **55 never matched any recorded query at all**
  (they need a better trigger).
- **Only 67.7% of trigger matches reach the caller**: 86,330 simulated matches
  against 58,462 actually delivered, with **50,961 matches occurring in events
  where more schemas fired than `max_results` could return**. The channel is
  over-subscribed, not under-firing.
- **Hair triggers: 104 schemas carry 1–2 trigger tokens.** At `overlap >= 0.5` a
  two-token trigger fires on ONE shared token. `reopen lesson` alone is the trigger
  of **64 schemas across 6 scopes**; `active goal supervision` of 40;
  `dashboard goal api supervision` of 14. 19 trigger texts are shared by more than
  one schema and account for **178 of 433** schemas — duplicates fire together by
  construction and then compete for the same slots.
- **4,495 colliding trigger pairs** involving **329 of 433 schemas**, in 40
  connected components; the largest cluster is 129 schemas spanning four scopes,
  20 groups have byte-identical trigger token sets.

→ Follow-up goal: **`when_to_use` trigger authoring.** The data says the work is
de-duplication and specificity (kill the 64-way `reopen lesson`, widen the 104 hair
triggers, give the 55 never-matching schemas real triggers), plus a
crowding/eviction policy for the 50,961 over-subscribed matches — not "add more
triggers".

### 7.3 Ranked Cyrillic OOV jargon dictionary

`artifacts/handoff/oov-jargon.md` (+ `.json`, all tokens with occurrences, distinct
queries/events/sessions/scopes, `corpus_df_active`, `fts_df_active`, OOV flag and a
`canonicalization_proposal` block). Same generator, same read-only slice.

Population — note the count, because the goal statement quotes a different one:
**2,931 distinct Cyrillic query tokens (length ≥ 3) before the stoplist, 2,877
after** it removes 54 (23,944 occurrences). The goal's "2936 distinct tokens
observed" does not appear anywhere in the artifact; `summary.distinct_tokens` is
**2877** and `summary.distinct_tokens_before_stoplist` is **2931**. Of those:

- **444 strictly out-of-vocabulary** (absent from every active node's content AND
  from the active-node FTS index), 2,411 occurrences;
- **241 cross-language bridge candidates** (a latin form exists in the corpus with
  ≥ 2× the Cyrillic token's coverage), 3,284 occurrences;
- evidence tiers: **strong** = 247 latin-exact + 11 curated; **medium** = 208
  latin-skeleton + 59 cyrillic-stem; **weak** = 348 cyrillic-editdist + 17
  latin-editdist + 3 cyrillic-prefix.

Highest-leverage bridges (`gap` = df(latin form) / (1 + df(cyrillic token))):

| token | occ | df_active | → canonical | method | gap |
|---|---|---|---|---|---|
| реквест | 155 | 2 | request | curated | 245.7× |
| октопуса | 208 | 8 | octopus | latin-exact | 209.0× |
| туллтипов | 229 | 4 | tooltip | curated | 141.4× |
| мердж | 165 | 8 | merge | curated | 114.6× |
| аттач | 24 | 0 (OOV) | attachment | curated | 63.0× |
| интент | 104 | 5 | intent | latin-exact | 27.5× |
| эксперимент | 246 | 33 | experiment | latin-exact | 15.4× |
| архитектуру / архитектуре | 148 / 204 | 1 / 7 | architecture | latin-skeleton | 151.0× / 37.8× |

The exact class the root goal named is present and canonicalized: **поревьювь → review**
(14 occ, curated), **апрув → approve** (4 occ, curated), plus `мердж` and `реквест`
above. Highest raw `rank_score` overall is **проанализируй** (161 occurrences,
`corpus_df_active` 0, and the generator proposes no canonical form for it) — a bare
command verb; the obvious reading is that it needs query-side stripping rather than
a synonym, but that is an inference, not something the artifact measures.

Two cautions carried from the artifact: **every entry is a PROPOSAL** — nothing in
`tokenize()`/`_SYNONYMS` was touched — and three high-frequency tokens
(`прорамми`, `обсуж`, `недост`) are **truncated queries, not jargon**; they should
be excluded from a synonym table rather than mapped.

→ Follow-up goal: **jargon / synonym dictionary.** The target metric already exists
and is frozen: `cross_lingual` hit@5 **0.105** on the goldset (flat across
chunking, MRR even −0.013), against 0.694 → 0.719 for `content_grounded`. Relevant
node returned *anywhere* in only 7 of 38 items — a candidate-collection failure
again, so a synonym expansion on the bm25/FTS side is the lever, and the harness
measures it end-to-end without any new instrument.

### 7.4 Embedding-model replacement, decided on harness data

The harness (`python3 -m living_memory.retrieval_harness`), the frozen goldset and
the two frozen reports are the decision instrument for that goal: a candidate model
is accepted or rejected on the same 234 items, the same five buckets and the same
byte-identical reproducibility, against `baseline.json` (pre-chunking) and
`phase1-gate.json` (current). The 512-token window variant was already rejected on
data (§8), which is the shape that goal should follow.

---

## 8. Known limits and what was deliberately NOT done

- **`max_seq_length` stays at 128.** The cheap "raise it to 512" option was
  measured and rejected before this tree started: on 8 nodes > 1,500 chars queried
  by a tail fragment, mean cos **0.479 at 512 versus 0.489 at 128** — adding text
  to one vector dilutes the signal rather than enriching it — and encoding is ~5×
  more expensive (18 vs 90 texts/s CPU). Chunking scored 0.954 on the same
  microtest. `chunking.py` therefore keeps `MODEL_MAX_SEQ_LENGTH = 128`,
  `DEFAULT_OVERLAP_TOKENS = 32`.
- **bm25/FTS, `tokenize()`/`_SYNONYMS` and the trigger channel are untouched.**
  They are the next goals (§7.2, §7.3); the inventories are read-only proposals.
- **No external vector index.** Pure numpy + sqlite; no FAISS, no sqlite-vec, no
  service.
- **Feedback mechanics untouched** — confidence/usefulness/access, `supersedes`,
  weight floors and the weight-learning loop are all unchanged and pinned by tests.
- **`cross_lingual` was not improved and was never going to be** by this phase:
  hit@5 flat at 0.105, MRR −0.013. It is a vocabulary problem.
- **One tail item regressed** (§3, regression 4) because of the length-bias
  correction, and the correction's coefficient (0.031) is a measured constant that
  taxes long `level:schema` nodes hardest — 20 windows cost 0.135. If a future goal
  judges it too aggressive, that constant belongs to the vector channel, not to the
  weight blend.
- **The A/B measures the blend, not recall.** Each query's candidate set is
  whatever the live learned weights actually retrieved; no scheme can be credited
  for documents no scheme retrieved.
- **Goldset relevance is the goldset's judgement**, frozen at build time:
  `content_grounded` inherits replay's IDF-containment label, `cross_lingual`
  inherits the paraphrase query's own top-k (so it measures jargon-vs-paraphrase
  agreement, not absolute truth), `role_query` is curated against active
  `level:schema` triggers. A relevant node that retrieval never returns counts as a
  **miss**, not as an unevaluated item.
- **`tests/test_replacement_holdout_packet.py` is still not really green** — its 82
  tests are skipped, not passing, for a machine-local snapshot that cannot be
  recovered (§5.4).

---

## 9. Migration runbook and live database state

**Runbook: [`docs/chunk-migration.md`](docs/chunk-migration.md)** — additive first,
destructive last, server up throughout except a restart:
0 preflight (read-only dry run) → 1 backup (backup API; `cp` is *not* a backup —
the live file carries a ~159 MB WAL) → 2 backfill with the server up → 3 deploy →
4 restart → 5 second backfill pass to catch the deploy window → 6 verify → 7 drop
the JSON column behind its gate. Tool:
`scripts/backfill_chunk_embeddings.py` (`backfill` / `verify` /
`drop-embedding-column`), resumable and idempotent, refusing to start without a
WAL-correct backup or without the real encoder.

Rehearsed costs from the runbook's §Measured numbers: backup 0.77–0.80 s; full
backfill **12 m 12 s** (82.7 chunks/s) producing exactly the 60,530 chunks and
92,974,080 bytes the dry run predicted; file growth +125.3 MiB during the
migration; budget ~700 MB free disk.

**The gate on the last step:** `drop-embedding-column` must not run until the
deployed code reads *chunks*. A premature drop does not crash — it silently turns
recall into bm25 + graph + trigger with an empty vector channel, which is worse.

### Live database state (NOT migrated)

Read `mode=ro` by this node on 2026-08-18 from
`~/.local/share/living-memory/global.sqlite3`:

| property | value |
|---|---|
| `metadata.schema_version` | **5** (code is at `SCHEMA_VERSION = 6`) |
| `node_chunk_embeddings` table | **absent** |
| `nodes.embedding` column | **present** — 16,620 rows, 134,900,593 B |
| active nodes / recall events | 12,876 / 54,790 |
| file + WAL | 532,918,272 B + 158,961,992 B |

So, plainly: **the production backfill has not been run, and `nodes.embedding` has
not been dropped.** Every measurement in this document was taken on backup-API
snapshots; the live database was opened read-only for counting only, and was never
opened through `MemoryStore` (whose constructor migrates and writes). The migration
is rehearsed end-to-end — including the drop and the post-drop harness run — but
running it against production is an operator step that this tree deliberately did
not take.

---

## 10. Artifact index

| artifact | what it holds |
|---|---|
| `artifacts/harness/goldset.jsonl` | frozen 234-item goldset, sha256 `af40cb0da26fa78c…` |
| `artifacts/harness/baseline.json` / `.md` | phase-0 baseline: overall/per-stratum/per-channel/tail, live agreement, determinism proof |
| `artifacts/harness/phase1-gate.json` / `.md` | nine gate conditions, four harness arms, storage/cold-init/p50, regression-test transcripts |
| `artifacts/replay/post-chunking-ab.json` / `.md` | three-slice weight A/B, holdout verdict, operator re-seed runbook |
| `artifacts/handoff/schema-inventory.json` / `.md` | 433 active schemas, triggers, firing and collision statistics |
| `artifacts/handoff/oov-jargon.json` / `.md` | 2,877 ranked Cyrillic query tokens with canonicalization proposals |
| `docs/chunk-migration.md` | the operator runbook and its measured costs |
| `src/living_memory/{chunking,storage,retrieval}.py`, `scripts/backfill_chunk_embeddings.py`, `scripts/handoff_inventory.py`, `scripts/replay_post_chunking.py` | the implementation |

Reproduction commands for each report are inside that report; the harness runs only
against a backup-API snapshot and is bit-deterministic.

---

<sub>This file replaces a stale `result.md` that documented the unrelated
`fix-dedup-noise` goal (schema-v3 content-fingerprint write-path dedup). That work
is in git history and its numbers are not affected by anything here. Its four
deferred follow-ups are carried forward so they are not lost with the document:
one-shot retroactive dedup of the ~625 duplicate excess in live `project:ae`;
path/kind-based supersede for changed files; the AE caller round-trip in
`cmd_bootstrap_project`; health-surface enrichment for the dedup metrics. None of
them belongs to this tree.</sub>
