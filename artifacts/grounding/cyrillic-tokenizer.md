# Cyrillic tokenizer: before/after (snapshots 2026-09-07, closures since 2026-09-04)

Change under test: Snowball-style Russian stemming for all-Cyrillic tokens in `living_memory.embeddings.tokenize` (`_stem`), stems of the 69 Cyrillic synonym keys mapped to their canonical token, and FTS5 prefix terms (`"стем"*`) for the Russian content words of a BM25 query (`retrieval._expanded_query` -> `storage._fts_query`). Switch: `LM_TOKENIZE_CYRILLIC_STEM` (`on` default, `off` = previous behaviour). Before = `off`, after = `on`, same checkout.

## Summary

- `off` reproduces the baseline artifact (`usage-signal-baseline.json`) exactly: yes.
- lat/lat pair grounding unchanged at every threshold and host: yes (byte-identity of non-Cyrillic tokenization, pinned by `tests/test_tokenize_cyrillic.py`).
- Pooled at 0.25: cyr-any pairs 201/5500 (3.6%) -> 378/5500 (6.9%); lat/lat 67/1306 (5.1%) -> 67/1306 (5.1%); closures grounded 160/844 (19.0%) -> 240/844 (28.4%); pair S/N 13.235 -> 14.553; closure S/N 3.428 -> 2.766.
- Retrieval harness (real encoder, 3659 goldset items, 227 Russian queries), branch minus master: overall hit@5 -0.0003, MRR -0.0002; russian_query hit@5 -0.0044, MRR -0.0030; non_russian_query hit@5 +0.0000, MRR +0.0000; non-Russian rankings identical on every item; `off` identical to master on every item: yes. BM25 latency per query with prefix terms: p50 22.3 -> 23.0 ms, p95 37.0 -> 38.3 ms.

## Usage-signal replay: grounding before vs after

Pairs = result/trace pairs; `cyr-any` = Cyrillic on either side, `lat/lat` = neither. Relatedness by encoder cosine (related ≥ 0.5, unrelated ≤ 0.3). S/N = grounded share among related over grounded share among unrelated.

### pooled

| thr | cyr-any before | cyr-any after | lat/lat before | lat/lat after | unrelated pairs before | after | pair S/N before | after | closures before | after | closure S/N before | after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 2486/5500 (45.2%) | 3147/5500 (57.2%) | 652/1306 (49.9%) | 652/1306 (49.9%) | 55/393 (14.0%) | 81/393 (20.6%) | 4.467 | 3.56 | 716/844 (84.8%) | 744/844 (88.1%) | 1.234 | 1.142 |
| 0.15 | 1186/5500 (21.6%) | 1810/5500 (32.9%) | 287/1306 (22.0%) | 287/1306 (22.0%) | 22/393 (5.6%) | 31/393 (7.9%) | 5.864 | 5.819 | 505/844 (59.8%) | 582/844 (69.0%) | 1.588 | 1.377 |
| 0.20 | 504/5500 (9.2%) | 870/5500 (15.8%) | 126/1306 (9.7%) | 126/1306 (9.7%) | 7/393 (1.8%) | 10/393 (2.5%) | 8.331 | 9.236 | 304/844 (36.0%) | 405/844 (48.0%) | 2.353 | 2.199 |
| 0.25 | 201/5500 (3.6%) | 378/5500 (6.9%) | 67/1306 (5.1%) | 67/1306 (5.1%) | 2/393 (0.5%) | 3/393 (0.8%) | 13.235 | 14.553 | 160/844 (19.0%) | 240/844 (28.4%) | 3.428 | 2.766 |

### alt

| thr | cyr-any before | cyr-any after | lat/lat before | lat/lat after | unrelated pairs before | after | pair S/N before | after | closures before | after | closure S/N before | after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 2119/3816 (55.5%) | 2632/3816 (69.0%) | 20/31 (64.5%) | 20/31 (64.5%) | 22/140 (15.7%) | 45/140 (32.1%) | 4.351 | 2.519 | 399/423 (94.3%) | 411/423 (97.2%) | 1.139 | 1.051 |
| 0.15 | 1045/3816 (27.4%) | 1569/3816 (41.1%) | 18/31 (58.1%) | 18/31 (58.1%) | 10/140 (7.1%) | 17/140 (12.1%) | 5.01 | 4.322 | 317/423 (74.9%) | 365/423 (86.3%) | 1.6 | 1.255 |
| 0.20 | 448/3816 (11.7%) | 760/3816 (19.9%) | 12/31 (38.7%) | 12/31 (38.7%) | 6/140 (4.3%) | 9/140 (6.4%) | 3.758 | 4.177 | 203/423 (48.0%) | 277/423 (65.5%) | 1.805 | 1.707 |
| 0.25 | 181/3816 (4.7%) | 337/3816 (8.8%) | 7/31 (22.6%) | 7/31 (22.6%) | 2/140 (1.4%) | 3/140 (2.1%) | 4.657 | 5.692 | 100/423 (23.6%) | 165/423 (39.0%) | 2.317 | 2.111 |

### sfx

| thr | cyr-any before | cyr-any after | lat/lat before | lat/lat after | unrelated pairs before | after | pair S/N before | after | closures before | after | closure S/N before | after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 367/1684 (21.8%) | 515/1684 (30.6%) | 632/1275 (49.6%) | 632/1275 (49.6%) | 33/253 (13.0%) | 36/253 (14.2%) | 3.704 | 3.859 | 317/421 (75.3%) | 333/421 (79.1%) | 1.334 | 1.243 |
| 0.15 | 141/1684 (8.4%) | 241/1684 (14.3%) | 269/1275 (21.1%) | 269/1275 (21.1%) | 12/253 (4.7%) | 14/253 (5.5%) | 5.424 | 5.418 | 188/421 (44.7%) | 217/421 (51.5%) | 1.455 | 1.492 |
| 0.20 | 56/1684 (3.3%) | 110/1684 (6.5%) | 114/1275 (8.9%) | 114/1275 (8.9%) | 1/253 (0.4%) | 1/253 (0.4%) | 29.3 | 38.05 | 101/421 (24.0%) | 128/421 (30.4%) | 4.256 | 3.913 |
| 0.25 | 20/1684 (1.2%) | 41/1684 (2.4%) | 60/1275 (4.7%) | 60/1275 (4.7%) | 0/253 (0.0%) | 0/253 (0.0%) | inf | inf | 60/421 (14.2%) | 75/421 (17.8%) | 8.651 | 5.149 |

## BM25 latency with prefix terms (sfx snapshot)

Snapshot `sfx-2026-09-07.sqlite3`, 400 distinct recorded Russian queries since 2026-08-01, 3 rounds per arm, arms measured off/on/off/on after a warm-up; one `search_content` call per scope of the recorded event's plan, per-scope limit as `_collect_bm25` uses it.

| arm | per query p50 | p95 | p99 | max | per call p50 | p95 | prefix terms/query (mean, max) | FTS terms/query (mean) |
|---|---|---|---|---|---|---|---|---|
| off | 22.2 ms | 37.1 ms | 41.9 ms | 60.2 ms | 11.6 ms | 19.1 ms | 0.0, 0 | 42.9 |
| on | 22.9 ms | 37.6 ms | 42.6 ms | 69.0 ms | 12.0 ms | 19.5 ms | 8.092, 20 | 42.62 |
| off | 22.3 ms | 37.0 ms | 42.3 ms | 54.3 ms | 11.7 ms | 19.1 ms | 0.0, 0 | 42.9 |
| on | 23.0 ms | 39.0 ms | 43.1 ms | 55.3 ms | 11.9 ms | 19.8 ms | 8.092, 20 | 42.62 |

Mean over the two runs of each arm, per query: off p50 22.3 ms / p95 37.0 ms, on p50 23.0 ms / p95 38.3 ms.

### Prefix breadth by stem length (sfx vocabulary, Cyrillic goldset queries)

Stems of the Russian content words of the Cyrillic goldset queries (460 distinct), each matched as an FTS5 prefix against the sfx index vocabulary; shipped floor `_RU_MIN_PREFIX = 4`.

| stem length | stems | vocab terms matched (median / mean) | doc postings (median / mean) |
|---|---|---|---|
| 3 | 39 | 19 / 27.5 | 212 / 329.3 |
| 4 | 71 | 14 / 19.9 | 150 / 260.9 |
| 5 | 80 | 8 / 11.3 | 100 / 163.3 |
| 6+ | 270 | 7 / 8.3 | 47 / 88.1 |

## Retrieval non-regression (retrieval harness, real encoder)

Goldset: 3659 items, 227 with a Cyrillic query. Real encoder, frozen snapshot `frozen-chunked.sqlite3` (sha256 d46f1b845adc…), live-agreement rate master 1.0, branch_on 1.0, branch_off 1.0, branch_on_prefix3 1.0.

| arm | bucket | items | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|---|
| master | overall | 3659 | 0.184 | 0.598 | 0.764 | 0.356 |
| master | russian_query | 227 | 0.339 | 0.749 | 0.885 | 0.514 |
| master | non_russian_query | 3432 | 0.174 | 0.588 | 0.755 | 0.345 |
| branch_on | overall | 3659 | 0.184 | 0.597 | 0.764 | 0.356 |
| branch_on | russian_query | 227 | 0.339 | 0.745 | 0.885 | 0.511 |
| branch_on | non_russian_query | 3432 | 0.174 | 0.588 | 0.755 | 0.345 |
| branch_off | overall | 3659 | 0.184 | 0.598 | 0.764 | 0.356 |
| branch_off | russian_query | 227 | 0.339 | 0.749 | 0.885 | 0.514 |
| branch_off | non_russian_query | 3432 | 0.174 | 0.588 | 0.755 | 0.345 |
| branch_on_prefix3 | overall | 3659 | 0.184 | 0.597 | 0.764 | 0.355 |
| branch_on_prefix3 | russian_query | 227 | 0.335 | 0.745 | 0.885 | 0.509 |
| branch_on_prefix3 | non_russian_query | 3432 | 0.174 | 0.588 | 0.755 | 0.345 |

branch_on minus master: overall: hit@1 +0.0000, hit@5 -0.0003, hit@10 +0.0000, mrr -0.0002; russian_query: hit@1 +0.0000, hit@5 -0.0044, hit@10 +0.0000, mrr -0.0030; non_russian_query: hit@1 +0.0000, hit@5 +0.0000, hit@10 +0.0000, mrr +0.0000.
branch_off minus master: overall: hit@1 +0.0000, hit@5 +0.0000, hit@10 +0.0000, mrr +0.0000; russian_query: hit@1 +0.0000, hit@5 +0.0000, hit@10 +0.0000, mrr +0.0000; non_russian_query: hit@1 +0.0000, hit@5 +0.0000, hit@10 +0.0000, mrr +0.0000.
branch_on_prefix3 minus master: overall: hit@1 -0.0002, hit@5 -0.0003, hit@10 +0.0000, mrr -0.0004; russian_query: hit@1 -0.0044, hit@5 -0.0044, hit@10 +0.0000, mrr -0.0052; non_russian_query: hit@1 +0.0000, hit@5 +0.0000, hit@10 +0.0000, mrr +0.0000.
Russian-query items, branch_on vs master: ranking changed on 85, first relevant node higher on 5, lower on 8; hit@10 unchanged. The remaining losses are queries whose Russian words are common (`проблема`, `задача`, `статус`) next to a rare identifier: the prefix term brings in documents dense in that word and the identifier match slips a rank or two.
Branch with `LM_TOKENIZE_CYRILLIC_STEM=off` returns the same ranked ids as master on every item: yes.

## What does not need reindexing, and what shifts

- **chunks** (`chunking.py`): windows are cut with the encoder's own tokenizer (`tokenizer.json` of the sentence-transformers snapshot); the module imports `DEFAULT_EMBEDDING_MODEL`, `_HASH_BACKENDS` and `_resolve_local_model_source` from `embeddings` and never `tokenize`. Chunk vectors come from the encoder. Unaffected.
- **query anchors**: stored as encoder vectors (`storage.iter_anchor_embedding_rows`, `retrieval._collect_anchor_seeds` matches on the query embedding). Unaffected.
- **`nodes_fts`**: the `CREATE VIRTUAL TABLE ... tokenize = 'unicode61'` block and its triggers are byte-identical to master (checked with `git show master:src/living_memory/storage.py`); the index keeps surface forms and the branch only changes the MATCH expression. No reindex.
- **recall-map c-TF-IDF labels** (`recall_map._terms`): candidates come from the map's own `_TERM_RE` and are looked up unstemmed in `nodes_fts_vocab`; `tokenize` is consulted only as a stop-word verdict. Over the sfx vocabulary (26163 Russian terms of 3+ letters) that verdict changes for 29 terms, all of them forms whose stem is a stop word (какая, какие, каким, каких, какого, какое, какой, каком, какому, пода, подам, подами, ...); these stop being label candidates. Labels are computed at recall time, nothing stored.
- **live consumers whose overlap values shift** (both tokenize both sides at call time, nothing stored):
  - `storage._recall_event_text_similarity` (pending-event matching, threshold 0.55): over the 844 corpus closures, cyr traces n=590: mean 0.6852 -> 0.7066, above threshold 415 -> 435, verdict flips 20; lat traces n=250: mean 0.6011 -> 0.6011, above threshold 151 -> 151, verdict flips 0; mixed traces n=4: mean 0.7403 -> 0.776, above threshold 3 -> 3, verdict flips 0.
  - `retrieval._collect_schema_triggers` (overlap threshold 0.5): 462 sfx schemas with a trigger × 300 recorded Russian queries: pairs firing 531 -> 531 (only on: 0, only off: 0).
- **derived stem synonyms** (29 entries): авторизац→authentication, аутентификац→authentication, баз→database, дан→data, депл→deployment, депло→deployment, зависим→require, задержк→slow, конфигурац→configuration, кэш→cache, медлен→slow, миграц→schema, настройк→configuration, нуж→require, нужн→require, окружен→environment, отсутств→missing, отсутствова→missing, ошибк→failure, пада→failure, паден→failure, причин→cause, проблем→failure, продакш→production, развертыван→deployment, сбо→failure, схем→schema, треб→require, упа→failure. `потому` is skipped (its stem `пот` is also the stem of `потом`).

## Known limits of the light stemmer

- Zero-ending genitive plurals with a fleeting vowel stay unstemmed (`ошибок` ≠ `ошибк`).
- Loanwords in `-ой` take the adjective path (`деплой` → `депл`, `деплоя` → `депло`); both are listed synonym forms and their prefix terms overlap, so BM25 still matches.
- A stem shorter than three letters is rejected in favour of the surface form (`этого` stays `этого`).
- Prefix terms need a four-letter stem. The three-letter floor was measured first (arm `branch_on_prefix3` above, where supplied): `рев*`, `цел*`, `дан*`, `мок*` and the like matched a median of 19 vocabulary terms each and moved relevant nodes down on more Russian goldset items than up, so the floor was raised; words with a shorter stem keep matching exactly as before.
