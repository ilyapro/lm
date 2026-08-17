# Phase-1 gate: chunked embeddings, float32 BLOB storage, max-pool vector channel

Generated 2026-08-17T20:55:00Z | phase-1 commit `b267569` | pre-change commit `8b93e8d` | harness v1

**Verdict: PASS** — all nine gate conditions hold, measured not assumed.

The frozen goldset (234 items, 26 tail) moves overall hit@5 0.526 → **0.577** and MRR 0.324 → **0.354**, while the tail subset — the 26 items whose answer lies past the node's first 128 model tokens, the whole reason this phase exists — goes from 2/26 to **10/26** at hit@5 (0.077 → **0.385**) and from 0.046 to **0.204** at MRR. Stored embedding bytes fall from 134.8 MB of JSON to **93.0 MB** of float32 BLOBs while holding 3.65× as many vectors; cold vector init drops 701 ms → **105 ms**; `memory_recall` p50 is **184.4 ms** against 192.1 ms before, i.e. faster rather than within the +5 ms allowance.

| gate | requirement | measured | verdict |
|---|---|---|---|
| overall hit@5 | ≥ 0.526 | 0.577 (+0.051) | PASS |
| overall MRR | ≥ 0.324 | 0.354 (+0.030) | PASS |
| tail hit@5 | clear rise over 0.077 | 0.385 — 2 → 10 of 26 items | PASS |
| tail MRR | clear rise over 0.046 | 0.204 (+0.158, ×4.4) | PASS |
| stored embedding bytes | < 134.6 MB | 93.0 MB (60,530 chunks) | PASS |
| cold vector init | < 200 ms | 104.84 ms for the whole 60,530-vector corpus | PASS |
| recall p50 | ≤ pre-change + 5 ms | 184.38 ms vs 192.08 ms (-7.69 ms, n=702 per arm) | PASS |
| test suite | no new failures vs 640 passed / 1 failed / 81 errors at `5996314` | 788 passed / 0 failed / 0 errors / 82 skipped | PASS |
| regression test | must fail on the pre-change code | `2 failed, 1 passed in 9.90s` on `8b93e8d`, `3 passed in 9.51s` here | PASS |

## What was compared with what

The published baseline was measured on a specific 532 MB snapshot. Comparing against a *fresh* snapshot alone would confound the code change with two hours of live-database drift, so the primary gate uses **the baseline's own bytes** — the same frozen snapshot, copied through the SQLite backup API and then chunked — and a fresh snapshot is used separately as a drift check. Four full harness runs, 234 queries each:

| arm | code | snapshot | hit@1 | hit@5 | MRR | tail hit@5 |
|---|---|---|---|---|---|---|
| control | `8b93e8d` | frozen + chunks | 0.175 | 0.526 | 0.324 | 0.077 |
| **gate** | `b267569` | frozen + chunks | 0.188 | 0.577 | 0.354 | 0.385 |
| drift, before | `8b93e8d` | fresh + chunks | 0.179 | 0.526 | 0.326 | 0.077 |
| drift, after | `b267569` | fresh + chunks | 0.188 | 0.577 | 0.354 | 0.385 |

The control arm is the load-bearing one. Its `metrics` block, sliced out of the written JSON as **raw bytes** (from `\n  "metrics"` to `\n  "provenance"`), is 3613 bytes hashing to `e942ed41c9a2dc30…` — **byte-identical to the committed baseline's**. Adding 60,530 chunk rows to the database changes nothing the pre-change code sees, and the harness reproduces the frozen baseline exactly on this box today. Every delta below is therefore attributable to the vector channel and to nothing else — not to drift, not to the backfill, not to harness noise.

Snapshots, all taken through the SQLite backup API (never `cp`, which would lose the WAL):

| snapshot | captured | bytes | active nodes | sha256 | role |
|---|---|---|---|---|---|
| `snapshot.sqlite3` | 2026-08-17T17:59:08Z | 532,647,936 | 12,862 | `d46f1b845adc3d80…` | the baseline's bytes, re-verified unchanged before and after every run |
| `phase1/frozen-chunked.sqlite3` | + backfill | 664,096,768 | 12,862 | `cc1675fba54e5d53…` | **primary gate snapshot** |
| `phase1/fresh.sqlite3` | 2026-08-17T20:10:12Z | 532,840,448 | 12,871 | `752ff825a6038d6b…` | fresh snapshot of the live database, taken for this gate |
| `phase1/fresh-chunked.sqlite3` | + backfill | 664,449,024 | 12,871 | — | drift check, then the column-drop rehearsal |

The live database at `/home/sfx/.local/share/living-memory/global.sqlite3` was opened read-only (file:...?mode=ro) for counting only and is still at schema v5 with no chunk table: **no backfill and no column drop was run against it**. The frozen `goldset.jsonl` (`af40cb0da26fa78c…`) and `baseline.json` (`a708a3b7885712da…`) were read only; `git status` shows them unmodified.

Live-agreement sanity ran in all four arms and came back 1.0 over 20 sampled items each: an independently constructed store + service pair returns the same top-5 as the runner.

## Quality gate

| bucket | items | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| overall | 234 | 0.175 → **0.188** (+0.013) | 0.526 → **0.577** (+0.051) | 0.688 → **0.726** (+0.038) | 0.324 → **0.354** (+0.030) |
| stratum:content_grounded | 160 | 0.231 → **0.250** (+0.019) | 0.694 → **0.719** (+0.025) | 0.900 → **0.881** (-0.019) | 0.422 → **0.444** (+0.022) |
| stratum:cross_lingual | 38 | 0.000 → **0.000** (+0.000) | 0.105 → **0.105** (+0.000) | 0.184 → **0.184** (+0.000) | 0.057 → **0.045** (-0.013) |
| stratum:role_query | 36 | 0.111 → **0.111** (+0.000) | 0.222 → **0.444** (+0.222) | 0.278 → **0.611** (+0.333) | 0.167 → **0.280** (+0.113) |
| **tail** | 26 | 0.038 → **0.038** (+0.000) | 0.077 → **0.385** (+0.308) | 0.077 → **0.538** (+0.462) | 0.046 → **0.204** (+0.158) |

Per-channel buckets re-order the same returned list by one channel's score (definition in `baseline.md`), so they bound a channel's ranking power, not its recall:

| channel | items | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| bm25 | 234 | 0.158 → **0.179** (+0.021) | 0.573 → **0.568** (-0.004) | 0.688 → **0.726** (+0.038) | 0.321 → **0.342** (+0.021) |
| graph | 234 | 0.154 → **0.201** (+0.047) | 0.521 → **0.560** (+0.038) | 0.688 → **0.726** (+0.038) | 0.306 → **0.353** (+0.048) |
| trigger | 234 | 0.162 → **0.175** (+0.013) | 0.526 → **0.577** (+0.051) | 0.688 → **0.726** (+0.038) | 0.315 → **0.345** (+0.030) |
| vector | 234 | 0.175 → **0.167** (-0.009) | 0.538 → **0.585** (+0.047) | 0.684 → **0.726** (+0.043) | 0.333 → **0.342** (+0.010) |

### The tail subset, which is the point

- hit@5 0.077 → **0.385** (2 → 10 of 26 items, ×5.0)
- MRR 0.046 → **0.204** (×4.4)
- hit@10 0.077 → **0.538** (2 → 14 of 26)

The hit@10 line is the mechanism, not a bonus metric. The baseline noted that on the tail hit@10 *equalled* hit@5: in 24 of 26 tail items the relevant node never entered the returned candidate list at all, so no re-ranking could have saved it. After chunking, 14 of 26 tail items collect a relevant node — the post-128-token content is now in a vector that can match, which is exactly the failure this phase set out to remove. Across the whole goldset the count of items with a relevant node anywhere rises 161 → 170 of 234.

`role_query` — procedural schemas asked for by role, e.g. "поревьювь <KEY>" — doubles its hit@5 (8 → 16 of 36 items) because most tail items live in that stratum. `cross_lingual` is flat at hit@5 and slightly down at MRR, which is the expected shape: it is a jargon-vocabulary problem (поревьювь / мердж-реквест / апрув), not a window problem, and it belongs to a later goal. Details below.

### Where every item moved

Rank of the first relevant node, baseline versus phase 1, per item. `newly found` and `lost entirely` are subsets of `better` and `worse` — an item whose relevant node was never returned before and is now returned counts in both. Tail items are split out of their stratum, so `role_query` + `role_query/tail` is the 36-item stratum.

| stratum | items | better | worse | unchanged | newly found | lost entirely |
|---|---|---|---|---|---|---|
| content_grounded | 160 | 43 | 25 | 92 | 3 | 6 |
| cross_lingual | 37 | 1 | 6 | 30 | 1 | 1 |
| cross_lingual/tail | 1 | 0 | 0 | 1 | 0 | 0 |
| role_query | 11 | 0 | 1 | 10 | 0 | 0 |
| role_query/tail | 25 | 14 | 1 | 10 | 13 | 1 |
| **total** | 234 | **58** | 33 | 143 | 17 | 8 |

### What got worse (nothing here is papered over)

Four sub-threshold regressions sit inside the aggregate win. None of them is a gate condition, and each is stated with its mechanism rather than as a rounding note.

**1. `cross_lingual` MRR 0.057 → 0.045 (-0.013).** hit@1, hit@5 and hit@10 are all unchanged (0.000 / 0.105 / 0.184), so no item won or lost a top-5 place; six items shifted by one to three ranks (2→3, 2→3, 4→5, 6→8, 6→9, 10→out) and one was newly found at rank 10. The jargon queries reach their answer through bm25 or graph, and the nodes that displace them by one rank are long nodes whose max-pool now scores higher. The stratum's own problem — a Russian jargon query has cosine 0.20–0.39 to an English node against 0.75–0.83 for normal lexicon — is untouched by windowing, as predicted.

**2. `content_grounded` hit@10 0.900 → 0.881** while its hit@5 and MRR both rise. Three of these items had their relevant node at rank 7–10 and lost it out of a 10-result list to newly stronger candidates. Overall hit@10 still rises (0.688 → 0.726) because the tail gains more than this stratum's fringe loses.

**3. The bm25 bucket's hit@5 0.573 → 0.568 (one item of 234) and the vector bucket's hit@1 0.175 → 0.167 (two items).** A bucket re-orders whatever retrieval returned, and the returned lists themselves changed, so these two numbers move for two reasons at once and cannot be read as "bm25 got worse". Both buckets' MRR rises (+0.021 bm25, +0.010 vector).

**4. One tail item lost its hit: `role_query-a486aa8b5f70`, rank 1 → not returned.** This one deserves the full mechanism, because it was one of only two tail hits the baseline had. Its relevant node is a 4,951-character `level:schema` node cut into 20 windows. Re-running that single query with `max_results=50` on both revisions isolates what happened:

| | rank | score | bm25 | vector | graph |
|---|---|---|---|---|---|
| pre-change | 1 | 0.5938 | 0.500 | 0.385 | 0.570 |
| phase 1 | 2 | 0.5519 | 0.500 | 0.335 | 0.570 |

Its lexical and graph evidence is untouched; its vector score falls 0.385 → 0.335. A 20-window node pays 0.0313·log2(20) = 0.135 of length-bias correction and its best window only gains about 0.085 over its head, so the correction costs it 0.050 net — and at the goldset item's `max_results=10` that is enough to fall out of the list, because the other candidates rose. The correction is the measured, deliberate cure for max-over-k inflation (`retrieval.py` module docstring), and the same stratum it costs here it doubles overall (role_query hit@5 0.222 → 0.444), so this is a real cost inside a clear net win, not a bug. If the parent judges the length coefficient too aggressive for long schema nodes, that constant and its measurement belong to the `retrieval-maxpool` sibling; the channel-blend response belongs to phase 2's `weights-recalibration`.

## Drift check on a fresh snapshot of the live database

A fresh backup-API snapshot (2026-08-17T20:10:12Z, 12,871 active nodes and 54,785 recall events against the frozen snapshot's 12,862 and 54,777) was chunked the same way and **both** revisions were re-run on it, so the drift arm is a paired comparison rather than a number floating against an older baseline:

| bucket | pre-change hit@5 → phase 1 | pre-change MRR → phase 1 |
|---|---|---|
| overall | 0.526 → **0.577** | 0.326 → **0.354** |
| stratum:content_grounded | 0.694 → **0.719** | 0.422 → **0.444** |
| stratum:cross_lingual | 0.105 → **0.105** | 0.071 → **0.045** |
| stratum:role_query | 0.222 → **0.444** | 0.167 → **0.280** |
| tail | 0.077 → **0.385** | 0.046 → **0.204** |

Two hours of live writes move the pre-change arm by less than 0.005 on every headline metric (overall hit@1 0.1752 → 0.1795, hit@5 unchanged at 0.5256, MRR 0.3238 → 0.3259) and leave the phase-1 arm identical to three decimals on all of them. The win is not an artefact of which snapshot it was measured on.

## End state: after `nodes.embedding` is dropped

The migration's last step removes the legacy JSON column. Rehearsed on the fresh chunked snapshot (`drop-embedding-column --yes --vacuum`), and then the harness re-run on the result, where chunk BLOBs are the *only* vectors in the database:

- hit@1 0.188, hit@5 **0.577**, MRR **0.354**, tail hit@5 0.385 — identical to the pre-drop run: true
- file 664,449,024 → 499,453,952 bytes, i.e. 33.4 MB **smaller than before the migration started** (532,840,448 bytes) while carrying 3.65× as many vectors

## Performance and size gate

### Stored embedding bytes

| | vectors | bytes | bytes per vector |
|---|---|---|---|
| `nodes.embedding` JSON | 16,606 | 134,788,051 (134.8 MB) | 8,116.8 |
| `node_chunk_embeddings` float32 | 60,530 | 92,974,080 (93.0 MB) | 1,536.0 |

69.0% of the JSON payload for 3.65× the vectors — 31% less, against a gate of < 134.6 MB. 1,536 bytes per vector is exactly 384 float32s with no framing, and every row records dimensions [384]. The planning estimate of ~69 MB assumed ~45k chunks; the real cut produces 60,530 (4.71 per node, 485 in the largest node), so the total lands at 93.0 MB rather than 69 MB — still a third below the column it replaces. The live database's JSON column is 134,844,299 bytes today, so the same ratio applies there.

### Cold initialisation of the vector channel

Method: fresh MemoryRecallService over an already-open store, corpus load called directly, no encoder involved; 5 repetitions, fresh service each time.

| scope | phase 1: chunk BLOB matrix | pre-change: JSON column |
|---|---|---|
| (all scopes) | 104.84 ms median (100.99 min), 60,530 vectors | 700.70 ms median (698.59 min), 12,829 vectors |
| global | 11.45 ms median (10.98 min), 5,237 vectors | 44.35 ms median (44.17 min), 803 vectors |
| project:ae | 25.55 ms median (24.72 min), 12,146 vectors | 108.06 ms median (106.04 min), 1,945 vectors |
| project:lm | 8.69 ms median (8.34 min), 3,577 vectors | 53.82 ms median (53.25 min), 977 vectors |
| project:octopus | 27.23 ms median (27.16 min), 13,001 vectors | 158.24 ms median (155.48 min), 2,877 vectors |
| project:online | 27.29 ms median (26.04 min), 11,477 vectors | 161.10 ms median (160.46 min), 2,959 vectors |
| project:x | 34.66 ms median (33.92 min), 13,861 vectors | 169.59 ms median (167.02 min), 3,058 vectors |

The whole corpus loads in **104.84 ms** against the 700.70 ms the JSON column costs on this box for the same nodes — 6.7× faster while reading 4.7× as many vectors, and comfortably inside the 200 ms gate. (The gate quoted 817 ms from an earlier measurement; the same operation measures 701 ms here, so the comparison is stated against what this box actually does rather than against the remembered figure.) A real recall pays only its plan's scopes: the largest single scope is 34.66 ms.

### `memory_recall` p50

Method: one process, one store, one service; encoder warmed and one untimed recall first; then every goldset query in query_id order, 3 repetitions, log_access and log_event left on; same snapshot for both arms. n = **702** per arm (234 queries × 3 repetitions).

| | p50 | p90 | p99 | mean | min | max | p50 per repetition |
|---|---|---|---|---|---|---|---|
| phase 1 | **184.38** | 644.33 | 855.81 | 292.40 | 58.51 | 1428.86 | 183.0, 182.9, 186.4 |
| pre-change | **192.08** | 662.14 | 912.74 | 300.40 | 52.21 | 1510.90 | 191.3, 190.0, 195.4 |

p50 is -7.69 ms — the change is a small improvement, not a cost inside the +5 ms allowance, and p90/p99/mean move the same way. The three per-repetition p50s agree within 3.4 ms, so the difference is not repetition noise. For scale, encoding the query alone costs 9.78 ms at p50 (same 702 measurements): the vector channel is a minority of a recall either way, which is why an 8 ms gap and not a 600 ms one shows up here even though cold init differs by 596 ms — the pre-change code cached parsed vectors per service instance, so only its first recall paid the JSON parse.

## Test suite

`bash scripts/check.sh` at `b267569` with the new regression test in place, run twice: **788 passed, 82 skipped, 0 failed, 0 errors**, rc=0, in 64.8 s and 64.0 s.

The recorded HEAD baseline at `5996314` was 640 passed / 1 failed / 81 errors, with all 82 problems in `tests/test_replacement_holdout_packet.py`. Those same 82 are now **skips** in that same file — `bash scripts/test.sh tests/test_replacement_holdout_packet.py` alone reports `6 passed, 82 skipped`, the machine-local snapshot its fixture needs being absent — and the suite has grown by 148 passing tests since that commit: 145 from this phase's siblings and 3 from this node. **No new failures and no new errors**; that file was not touched.

## Regression test: fails on the old code, passes on the new

`tests/test_chunk_tail_regression.py` builds a node whose first 128 model tokens are meeting logistics and whose answer — a TLS certificate expiry on an internal registry mirror — starts at character 1,160 of 1,871, around token 246 of 431. It asks in Russian, so the FTS channel (which indexes the whole node, tail included) cannot answer: `bm25_score` is asserted to be exactly 0, and the wording avoids every entry of the Russian synonym table that would expand into an English token. Eight plausible English neighbours fill the top 5 when the tail node is invisible.

### On this revision (`b267569`)

```text
$ bash scripts/test.sh tests/test_chunk_tail_regression.py -v
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/sfx/p/lm/.worktrees/_node_exec_vector-recall-chunking-harness_phase1-gate
configfile: pyproject.toml
plugins: anyio-4.13.0
collected 3 items

tests/test_chunk_tail_regression.py ...                                  [100%]

============================== 3 passed in 9.51s ===============================
```

### On the pre-change revision (`8b93e8d`)

The parent of the max-pool commit, checked out into a clean tree with `git archive`, with this test file copied in unchanged:

```text
$ git archive 8b93e8d | tar -x -C /tmp/lm-prechange
$ cp tests/test_chunk_tail_regression.py /tmp/lm-prechange/tests/
$ cd /tmp/lm-prechange && PYTHONPATH=/tmp/lm-prechange/src:<worktree>/.cache/python-deps \
    LIVING_MEMORY_EMBEDDING_BACKEND=hash python3 -m pytest tests/test_chunk_tail_regression.py -v
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.3, pluggy-1.6.0
rootdir: /tmp/lm-prechange
configfile: pyproject.toml
plugins: anyio-4.13.0
collected 3 items

tests/test_chunk_tail_regression.py .FF                                  [100%]
...
E       AssertionError: tail node ranked 9 of 9 (vector_score 0.2208, whole-node cosine 0.2208)
E       assert 9 <= 5
E       AssertionError: vector_score 0.2208 is not meaningfully above the whole-node cosine 0.2208: the tail is still invisible
E       assert 0.2208152711391449 >= (1.5 * 0.22081530091261353)
FAILED tests/test_chunk_tail_regression.py::test_tail_grounded_node_is_recalled_in_the_top_five
FAILED tests/test_chunk_tail_regression.py::test_the_vector_channel_is_what_found_it
========================= 2 failed, 1 passed in 9.90s ==========================
```

The premise test passes on both revisions, as it must — it only measures where the answer sits relative to the encoder's window, which no code change moves. The two behavioural tests fail on the old code and pass here. Note the second failure line: on the pre-change revision the delivered `vector_score` **equals the whole-node cosine to four decimals** (0.2208 = 0.2208), which is the single-vector channel caught in the act; on this revision it is 0.5393 against the same 0.2208 whole-node cosine, i.e. 2.44×.

### Why it is a real test and not a fixture that happens to pass

- The **top-5 assertion** is the house rule, and it fails on the old code at rank 9 of 9 — last place, not a near miss.
- The **mechanism assertion** compares the delivered score against a whole-node vector recomputed at test time (the encoder truncates at 128 tokens, so one vector of the whole node *is* what the old code stored). It is scale-free, so it survives model or corpus drift, and it cannot pass on a single-vector channel by construction — there the ratio is exactly 1.0. Recomputing rather than reading `nodes.embedding` also keeps the test working after an operator drops that column.
- A **fixture-validity assertion** re-ranks the same returned results by that whole-node cosine and requires the node to fall outside the top 5, so the corpus cannot silently drift into one the old channel would also have solved.
- The **premise** is measured, not asserted in prose: every answer-bearing term (`certificate`, `certbot`, `registry`, `mirror`, `expired`) must first occur at or after the character where the encoder's 126-content-token window ends, computed with the model's own tokenizer.
- It runs the model in a **child process**. Clearing `LIVING_MEMORY_EMBEDDING_BACKEND` in-process was tried first and broke two unrelated tests — `embeddings._MODEL_CACHE` then handed the real model to a test that mocks `sentence_transformers`, and `torch` stayed in `sys.modules` for a test asserting it was never imported. The child returns measurements as JSON; every assertion runs in the parent, which imports nothing from `living_memory`. Runtime is ~9.5 s for all three tests, and it skips loudly if the model is not in the local cache.

## Reproduction

```bash
# 0. snapshots, via the backup API (never cp: the live file carries a ~159 MB WAL)
python3 -m living_memory.retrieval_harness snapshot \
    --source-db ~/.local/share/living-memory/global.sqlite3 \
    --snapshot-out ~/.cache/living-memory-harness/phase1/fresh.sqlite3
python3 -m living_memory.retrieval_harness snapshot \
    --source-db ~/.cache/living-memory-harness/snapshot.sqlite3 \
    --snapshot-out ~/.cache/living-memory-harness/phase1/frozen-chunked.sqlite3

# 1. chunk the copies -- never the live database
python3 scripts/backfill_chunk_embeddings.py backfill \
    --db ~/.cache/living-memory-harness/phase1/frozen-chunked.sqlite3 \
    --existing-backup ~/.cache/living-memory-harness/snapshot.sqlite3 \
    --json /tmp/gate/frozen-chunked-backfill.json --progress-interval 60

# 2. one harness run per arm (PYTHONPATH=src, LIVING_MEMORY_EMBEDDING_BACKEND unset)
python3 -m living_memory.retrieval_harness run \
    --snapshot ~/.cache/living-memory-harness/phase1/frozen-chunked.sqlite3 \
    --goldset artifacts/harness/goldset.jsonl --agreement-sample 20 \
    --cutoff 2026-06-10T00:00:00Z \
    --report /tmp/gate/run-new-frozen-chunked.json --markdown /tmp/gate/run-new-frozen-chunked.md

# 3. the same command from a git archive of 8b93e8d gives the control arm
# 4. size and coverage
python3 scripts/backfill_chunk_embeddings.py verify \
    --db ~/.cache/living-memory-harness/phase1/fresh-chunked.sqlite3 --json /tmp/gate/verify.json
# 5. suite and regression test
bash scripts/check.sh
bash scripts/test.sh tests/test_chunk_tail_regression.py -v
```

Cold-init and latency were measured by two throwaway scripts whose method is stated in full above and in `phase1-gate.json` (`performance.cold_vector_init.method`, `performance.recall_latency.method`): both call the corpus-load method directly / drive `MemoryRecallService.memory_recall` over the goldset in `query_id` order, and both were run identically against each revision's `src/` on the same snapshot.

## Handoff notes for phase 2 (weights recalibration)

- **The vector-score distribution moved, as expected.** Channel attribution over items with a relevant result: vector participates in 161/170 = 0.947 of them (baseline 149/161 = 0.925), and graph's share rises 0.193 → 0.288 because a max-pooled candidate set feeds the traversal more seeds. The learned weights were calibrated against the old distribution.
- **The graph bucket gained the most of any channel** (hit@1 +0.047, MRR +0.048), entirely as a consequence of better seeds. Any A/B over channel weights should hold that in mind: part of graph's apparent strength is borrowed from the vector channel.
- **Long `level:schema` nodes are the population the length-bias correction taxes hardest** — see regression 4 above, where 20 windows cost 0.135 of cosine. Phase 2 changes weights, not that coefficient, but the two interact through `STRONG_VECTOR_MATCH` and the schema trigger boost.
- **`cross_lingual` remains the largest untouched gap** (hit@5 0.105, relevant node returned anywhere in 7 of 38 items). Chunking was never going to fix it; the ranked Cyrillic OOV dictionary in `artifacts/handoff/oov-jargon.md` is the input for that goal.
- **Post-drop equivalence is proven** (section above), so phase 2 can work on a chunk-only database without keeping the JSON column alive as a fallback.

