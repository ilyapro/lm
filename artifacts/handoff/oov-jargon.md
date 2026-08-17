# Handoff inventory: Cyrillic query jargon and its corpus coverage

Read-only evidence for the *jargon dictionary* goal. Every canonicalization below is a **PROPOSAL**: this run did not touch `tokenize()`, `_SYNONYMS`, the bm25/FTS path or any source module.

## Provenance

- generator: `scripts/handoff_inventory.py` v1.1.0
- generated_at / slice cutoff (`--as-of`): `2026-08-17T15:45:00Z` (slice frozen: True)
- database: `/home/sfx/.local/share/living-memory/global.sqlite3` (532647936 bytes, mtime 2026-08-17T16:41:10Z)
- access: sqlite3.connect(file:<path>?mode=ro, uri=True) + PRAGMA query_only=1 + one deferred read transaction pinning a single WAL snapshot; MemoryStore is never imported (its constructor migrates and writes)
- git commit: `254539ead919e77005c615c43db1ebf3695037a7` (branch `vector-recall-chunking-harness--handoff-inventory`, worktree dirty: True) -- rebase-fragile, see below
- generator sha256: `a3ef0412cc3c8b20cc22be7c498c2ad8a9caee645a455a5c175916ff7e3cae6d` (rebase-invariant anchor; `sha256sum scripts/handoff_inventory.py`)
- runtime: python 3.12.3, sqlite 3.45.1
- row counts: corpus_distinct_surface_tokens = 50093, corpus_nodes_active = 12843, corpus_nodes_all = 16664, first_event_at = 2026-05-14T20:05:23Z, last_event_at = 2026-08-17T15:34:52Z, nodes_active = 12858, nodes_in_slice = 16664, nodes_total = 16679, recall_events_in_slice = 54759, recall_events_scanned = 54759, recall_events_total = 54772, result_rows_scanned = 367151, schema_nodes_active_in_slice = 433

### Exact query slices

```sql
-- fts_document_frequency
SELECT count(*) FROM nodes_fts JOIN nodes n ON n.rowid = nodes_fts.rowid WHERE nodes_fts MATCH :token AND n.decayed = 0 AND n.created_at <= :as_of;
-- node_content_for_corpus_df
SELECT content, decayed FROM nodes WHERE created_at <= :as_of ORDER BY id;
-- recall_events
SELECT id, query, scope, resolved_scopes, session_id, transport_session_id, results, created_at, max_results FROM recall_events WHERE created_at <= :as_of ORDER BY id;
-- schemas
SELECT id, scope, context, content, created_at, timestamp, access_count, usefulness_score, confidence, unique_agents, last_accessed FROM nodes WHERE decayed = 0 AND level = 'schema' AND created_at <= :as_of ORDER BY id;
```

### Determinism contract

- frozen by `--as-of`: Every value in the analysis payload EXCEPT the fields listed in current_state_at_read. Re-running with the recorded --as-of against the same database must reproduce them byte-for-byte.
- current state at read (NOT reproducible, tolerated by `--verify`): `schemas[].access_count`, `schemas[].last_accessed`, `schemas[].usefulness_score`, `schemas[].confidence`, `schemas[].unique_agents`
- why: nodes stores only the latest value of these counters (no per-access history), so --as-of cannot reconstruct their value at the cutoff. They advance whenever the live MCP server serves a recall that touches the node.
- decay: corpus_df_active / fts_df_active / schema membership also depend on decayed = 0, which is current state -- but decay is a rare batch event, not continuous drift, so --verify deliberately reports it as frozen drift. That is the intended signal: the corpus moved under the artifact, regenerate it. The per-node counters above are excluded only because they advance on every served recall, which would make the check fire constantly and mean nothing.
- generator binding: provenance.git.generator_sha256 is checked as frozen, so --verify fails if the committed artifacts were produced by a different generator than the committed scripts/handoff_inventory.py.

### Caveats

- Determinism has THREE layers, not two. (1) The derived analysis -- token set, occurrence counts, event-slice statistics, trigger firing, collisions, canonicalization proposals -- is frozen by --as-of and is byte-identical across re-runs. (2) A handful of per-node columns are CURRENT STATE AT READ, not as-of state, because SQLite stores only their latest value and no history: they move whenever the live server serves a recall. They are enumerated field-by-field in `provenance.determinism.current_state_at_read`. (3) The PROVENANCE block deliberately describes the live file at read time, so database bytes/mtime, the unsliced *_total counts and the git dirty flag move too. Run `python3 scripts/handoff_inventory.py --verify` to check this contract mechanically: it fails if any layer-1 value drifted.
- decayed = 0 is CURRENT state, not as-of state: the node set can shift if a schema decays after this run, while the event slice cannot.
- recall_events.results records only DELIVERED results (max_results cap), so a trigger that matched but lost its slot leaves no trace there -- that is what the simulated_trigger_matches column measures.
- The live database is written concurrently by the MCP server; the read transaction isolates one snapshot, and --as-of freezes the slice.

Reproduce: `python3 scripts/handoff_inventory.py --db /home/sfx/.local/share/living-memory/global.sqlite3 --as-of 2026-08-17T15:45:00Z`
Verify instead of rewrite: `python3 scripts/handoff_inventory.py --verify`

## Method

- extraction: surface tokens of recall_events.query under the shipped normalization (embeddings.tokenize minus stemming/synonyms, which are latin-only and therefore identity on Cyrillic)
- token filter: `^[а-я]+$ after yo-folding, length >= 3`
- OOV definition: corpus_df_active == 0 AND fts_df_active == 0 -- absent from the content of every active node and unreachable through the bm25 FTS path (storage.py:1208-1227 joins nodes_fts with n.decayed = 0)
- ranking: `rank_score = occurrences * 1/(1 + corpus_df_active)` -- a token nobody stored keeps its full query frequency, a token stored in N active nodes keeps `1/(1+N)` of it
- proposals computed when: `occurrences >= 2 and corpus_df_active <= 60`

### Stoplist (explicit)

Shipped Russian stop words from `living_memory.embeddings._RUSSIAN_STOP_WORDS` with length >= 3 (15 entries):

> без, для, его, если, или, как, под, при, так, уже, что, эти, это, этот, эту

Plus this script's explicit `EXTRA_CYRILLIC_STOPWORDS` (75 entries), function words that survive the shipped list but carry no retrieval signal:

> был, была, были, было, быть, вот, все, всех, вы, где, да, даже, два, две, ему, есть, еще, ещё, здесь, или, их, йот, которая, которые, который, кто, куда, мне, мной, может, можно, мы, надо, нам, нас, него, нее, нет, ним, них, ничего, ну, нужно, они, оно, очень, потом, потому, почему, просто, раз, себя, сейчас, тебя, тем, теперь, тех, того, тоже, той, только, том, тот, тут, чем, через, чтобы, эта, этим, этих, этого, этом, этому, является, являеться

## Population

- distinct Cyrillic query tokens (length >= 3) before the stoplist: 2931; the stoplist removes 54 of them (2417 occurrences)
- distinct Cyrillic query tokens after the stoplist: **2877** over 23944 occurrences
- strictly out-of-vocabulary (absent from every active node AND from the active-node FTS index): **444** tokens, 2411 occurrences
- cross-language bridge candidates (a latin form exists in the corpus with at least 2x the Cyrillic token's coverage): **241** tokens, 3284 occurrences

Corpus coverage bands (`corpus_df_active` = active nodes whose content contains the token):

| corpus_df_active | tokens |
|---|---|
| 0 | 444 |
| 1 | 330 |
| 2-5 | 699 |
| 6-20 | 869 |
| 21-100 | 498 |
| >100 | 37 |

Proposal methods and their evidence tier (`strong` = curated or an exact transliteration hit in the corpus; `medium` = identical consonant skeleton; `weak` = bounded edit distance or a prefix completion -- JSON-only material, not quoted as a bridge below):

| method | tier | tokens |
|---|---|---|
| in-corpus | none | 965 |
| not-evaluated | none | 875 |
| cyrillic-editdist | weak | 348 |
| latin-exact | strong | 247 |
| latin-skeleton | medium | 208 |
| none | none | 144 |
| cyrillic-stem | medium | 59 |
| latin-editdist | weak | 17 |
| curated | strong | 11 |
| cyrillic-prefix | weak | 3 |

- session attribution: 44275 of 54759 events carry neither `session_id` nor `transport_session_id`, so `distinct_sessions` is a lower bound
- content scan vs FTS index: 58 of 2877 tokens disagree between `corpus_df_active` and `fts_df_active` (content scan counts active nodes whose text contains the token under this script's normalization; fts_df_active is the real bm25 path (fts5 unicode61). Both are measured over the SAME as-of-sliced active node set, so the remaining disagreements are tokenizer differences only -- not a node-set difference.)

## Top 40 by `rank_score` (frequency x OOV-ness)

This is the list to cut a threshold on.

| token | occ | queries | sessions | scopes | df_active | fts_df | OOV | rank_score | PROPOSAL | method | tier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| проанализируй | 161 | 68 | 49 | 3 | 0 | 0 | yes | 161.00 | -- | none | none |
| выработать | 148 | 2 | 4 | 1 | 0 | 0 | yes | 148.00 | работать | cyrillic-editdist | weak |
| кото | 148 | 2 | 4 | 1 | 0 | 0 | yes | 148.00 | кто | cyrillic-editdist | weak |
| подкрутить | 148 | 2 | 4 | 1 | 0 | 0 | yes | 148.00 | -- | none | none |
| промежу | 125 | 2 | 0 | 1 | 0 | 0 | yes | 125.00 | preimage | latin-skeleton | medium |
| архитектуру | 148 | 2 | 4 | 1 | 1 | 1 | no | 74.00 | architecture | latin-skeleton | medium |
| соответсвующий | 73 | 31 | 13 | 2 | 0 | 0 | yes | 73.00 | соответствующий | cyrillic-editdist | weak |
| реквест | 155 | 67 | 46 | 7 | 2 | 2 | no | 51.67 | request | curated | strong |
| туллтипов | 229 | 4 | 0 | 1 | 4 | 4 | no | 45.80 | tooltip | curated | strong |
| подключаем | 40 | 4 | 32 | 2 | 0 | 0 | yes | 40.00 | подключен | cyrillic-editdist | weak |
| возможностей | 149 | 3 | 5 | 2 | 3 | 3 | no | 37.25 | -- | in-corpus | none |
| идею | 148 | 2 | 4 | 1 | 3 | 3 | no | 37.00 | -- | in-corpus | none |
| предела | 148 | 2 | 4 | 1 | 3 | 3 | no | 37.00 | -- | in-corpus | none |
| ресурсы | 172 | 74 | 50 | 4 | 4 | 4 | no | 34.40 | -- | in-corpus | none |
| прораммировать | 133 | 5 | 0 | 1 | 3 | 3 | no | 33.25 | -- | in-corpus | none |
| заводим | 32 | 2 | 32 | 1 | 0 | 0 | yes | 32.00 | заводить | cyrillic-editdist | weak |
| развить | 332 | 17 | 4 | 2 | 10 | 10 | no | 30.18 | -- | in-corpus | none |
| диза | 28 | 2 | 28 | 1 | 0 | 0 | yes | 28.00 | дифа | cyrillic-editdist | weak |
| смотрели | 28 | 2 | 28 | 1 | 0 | 0 | yes | 28.00 | смотреть | cyrillic-editdist | weak |
| улучшениями | 28 | 2 | 28 | 1 | 0 | 0 | yes | 28.00 | улучшения | cyrillic-editdist | weak |
| связаннные | 81 | 33 | 13 | 2 | 2 | 2 | no | 27.00 | связанные | cyrillic-editdist | weak |
| перевести | 27 | 16 | 25 | 2 | 0 | 0 | yes | 27.00 | перенести | cyrillic-editdist | weak |
| умении | 183 | 14 | 0 | 1 | 6 | 6 | no | 26.14 | -- | in-corpus | none |
| архитектуре | 204 | 18 | 0 | 1 | 7 | 7 | no | 25.50 | architecture | latin-skeleton | medium |
| общаться | 203 | 17 | 0 | 1 | 7 | 7 | no | 25.38 | -- | in-corpus | none |
| конфигурацию | 150 | 4 | 4 | 2 | 5 | 5 | no | 25.00 | -- | in-corpus | none |
| аттач | 24 | 2 | 0 | 1 | 0 | 0 | yes | 24.00 | attachment | curated | strong |
| галочку | 24 | 2 | 0 | 1 | 0 | 0 | yes | 24.00 | галочка | cyrillic-editdist | weak |
| октопуса | 208 | 22 | 0 | 1 | 8 | 8 | no | 23.11 | octopus | latin-exact | strong |
| испо | 20 | 2 | 0 | 1 | 0 | 0 | yes | 20.00 | по | cyrillic-editdist | weak |
| копотом | 20 | 2 | 0 | 1 | 0 | 0 | yes | 20.00 | потом | cyrillic-editdist | weak |
| полноценной | 20 | 2 | 0 | 1 | 0 | 0 | yes | 20.00 | полноценный | cyrillic-editdist | weak |
| мердж | 165 | 77 | 52 | 8 | 8 | 7 | no | 18.33 | merge | curated | strong |
| прорамми | 18 | 1 | 0 | 1 | 0 | 0 | yes | 18.00 | прораммировать | cyrillic-prefix | weak |
| прочитай | 18 | 7 | 6 | 2 | 0 | 0 | yes | 18.00 | перечитай | cyrillic-editdist | weak |
| интент | 104 | 38 | 90 | 4 | 5 | 5 | no | 17.33 | intent | latin-exact | strong |
| бэкэнд | 34 | 4 | 34 | 1 | 1 | 1 | no | 17.00 | backend | latin-skeleton | medium |
| аттестационный | 17 | 9 | 5 | 1 | 0 | 0 | yes | 17.00 | attestation | latin-editdist | weak |
| изучи | 183 | 82 | 56 | 6 | 10 | 10 | no | 16.64 | -- | in-corpus | none |
| добавгть | 16 | 2 | 16 | 1 | 0 | 0 | yes | 16.00 | добавить | cyrillic-editdist | weak |

## Top 40 actionable candidates (strong/medium evidence, ranked the same way)

The same ranking restricted to tokens that actually have a defensible canonical form. `in-corpus` tokens are dropped here: bm25 already reaches them, so they need no synonym.

| token | occ | df_active | OOV | rank_score | PROPOSAL | method | gap |
|---|---|---|---|---|---|---|---|
| промежу | 125 | 0 | yes | 125.00 | preimage | latin-skeleton | 8.0x |
| архитектуру | 148 | 1 | no | 74.00 | architecture | latin-skeleton | 151.0x |
| реквест | 155 | 2 | no | 51.67 | request | curated | 245.7x |
| туллтипов | 229 | 4 | no | 45.80 | tooltip | curated | 141.4x |
| архитектуре | 204 | 7 | no | 25.50 | architecture | latin-skeleton | 37.8x |
| аттач | 24 | 0 | yes | 24.00 | attachment | curated | 63.0x |
| октопуса | 208 | 8 | no | 23.11 | octopus | latin-exact | 209.0x |
| мердж | 165 | 8 | no | 18.33 | merge | curated | 114.6x |
| интент | 104 | 5 | no | 17.33 | intent | latin-exact | 27.5x |
| бэкэнд | 34 | 1 | no | 17.00 | backend | latin-skeleton | 414.5x |
| всплесков | 12 | 0 | yes | 12.00 | всплеск | cyrillic-stem | 7.0x |
| глубокую | 21 | 1 | no | 10.50 | glibc | latin-skeleton | 3.0x |
| текстуру | 21 | 1 | no | 10.50 | textarea | latin-skeleton | 6.0x |
| доказательное | 10 | 0 | yes | 10.00 | доказательно | cyrillic-stem | 3.0x |
| минимал | 10 | 0 | yes | 10.00 | minimal | latin-exact | 87.0x |
| поп | 10 | 0 | yes | 10.00 | pop | latin-exact | 27.0x |
| шаринга | 10 | 0 | yes | 10.00 | sharing | curated | 41.0x |
| зеленого | 34 | 3 | no | 8.50 | sealing | latin-skeleton | 11.2x |
| городом | 8 | 0 | yes | 8.00 | город | cyrillic-stem | 16.0x |
| матрице | 8 | 0 | yes | 8.00 | metric | latin-skeleton | 193.0x |
| медленно | 8 | 0 | yes | 8.00 | medallion | latin-skeleton | 32.0x |
| репозитори | 8 | 0 | yes | 8.00 | repository | latin-skeleton | 89.0x |
| смысла | 8 | 0 | yes | 8.00 | смысл | cyrillic-stem | 3.0x |
| толь | 8 | 0 | yes | 8.00 | tol | latin-exact | 14.0x |
| эксперимент | 246 | 33 | no | 7.24 | experiment | latin-exact | 15.4x |
| релоада | 7 | 0 | yes | 7.00 | reload | latin-exact | 207.0x |
| проведи | 59 | 8 | no | 6.56 | proved | latin-exact | 6.2x |
| тултипы | 37 | 5 | no | 6.17 | tooltip | curated | 117.8x |
| заправок | 43 | 6 | no | 6.14 | zapravki | latin-skeleton | 1.9x |
| вызови | 6 | 0 | yes | 6.00 | вызов | cyrillic-stem | 41.0x |
| касаются | 6 | 0 | yes | 6.00 | costs | latin-skeleton | 24.0x |
| контекстой | 6 | 0 | yes | 6.00 | context | latin-exact | 931.0x |
| промеж | 6 | 0 | yes | 6.00 | preimage | latin-skeleton | 8.0x |
| брифе | 16 | 2 | no | 5.33 | бриф | cyrillic-stem | 4.7x |
| заправки | 16 | 2 | no | 5.33 | zapravki | latin-exact | 4.3x |
| менеджер | 16 | 2 | no | 5.33 | manager | latin-skeleton | 17.7x |
| смотри | 16 | 2 | no | 5.33 | symmetry | latin-skeleton | 2.3x |
| сбол | 10 | 1 | no | 5.00 | sbol | latin-exact | 5.5x |
| репозитория | 134 | 26 | no | 4.96 | repository | latin-skeleton | 3.3x |
| бензину | 28 | 5 | no | 4.67 | benzin | latin-exact | 54.7x |

## Cross-language bridge candidates (strong tier only)

The class the root goal named: a Russian-spelled borrowing whose English form is what the corpus actually stores. `gap` = `df(proposed latin form) / (1 + df(cyrillic token))`. Restricted to curated and exact-transliteration evidence -- weaker matches live in the JSON.

| token | occ | df_active | PROPOSAL | method | gap |
|---|---|---|---|---|---|
| эксперимент | 246 | 33 | experiment | latin-exact | 15.4x |
| туллтипов | 229 | 4 | tooltip | curated | 141.4x |
| октопуса | 208 | 8 | octopus | latin-exact | 209.0x |
| бензин | 167 | 54 | benzin | latin-exact | 6.0x |
| мердж | 165 | 8 | merge | curated | 114.6x |
| реквест | 155 | 2 | request | curated | 245.7x |
| контекст | 147 | 57 | context | latin-exact | 16.1x |
| интент | 104 | 5 | intent | latin-exact | 27.5x |
| тултипа | 65 | 52 | tooltip | curated | 13.3x |
| проведи | 59 | 8 | proved | latin-exact | 6.2x |
| оператор | 44 | 44 | operator | latin-exact | 19.9x |
| сервер | 40 | 60 | server | latin-exact | 14.7x |
| тултипы | 37 | 5 | tooltip | curated | 117.8x |
| контент | 33 | 34 | content | latin-exact | 28.2x |
| бсс | 30 | 10 | bss | latin-exact | 47.8x |
| маркер | 29 | 46 | marker | latin-exact | 12.0x |
| панели | 29 | 19 | panel | latin-exact | 10.4x |
| бензину | 28 | 5 | benzin | latin-exact | 54.7x |
| контракту | 28 | 22 | contract | latin-exact | 98.6x |
| тултипам | 28 | 8 | tooltip | latin-exact | 78.6x |
| мини | 25 | 38 | mini | latin-exact | 3.5x |
| аттач | 24 | 0 | attachment | curated | 63.0x |
| рубрики | 24 | 33 | rubric | latin-exact | 5.6x |
| статусы | 23 | 32 | status | latin-exact | 63.0x |
| эксперимента | 23 | 9 | experiment | latin-exact | 52.4x |
| агентов | 22 | 13 | agent | latin-exact | 55.1x |
| маркера | 22 | 35 | marker | latin-exact | 15.7x |
| статусов | 20 | 13 | status | latin-exact | 148.6x |
| элементов | 20 | 12 | element | latin-exact | 8.8x |
| метрики | 19 | 14 | metric | latin-exact | 12.9x |
| модель | 19 | 24 | model | latin-exact | 70.9x |
| чат | 19 | 24 | chat | latin-exact | 6.8x |
| конфиг | 18 | 34 | config | latin-exact | 47.3x |
| ассетов | 17 | 10 | asset | latin-exact | 2.3x |
| лист | 17 | 19 | list | latin-exact | 36.2x |
| проблема | 17 | 9 | problem | latin-exact | 3.6x |
| сео | 17 | 6 | seo | latin-exact | 21.1x |
| заправки | 16 | 2 | zapravki | latin-exact | 4.3x |
| контракта | 16 | 37 | contract | latin-exact | 59.7x |
| материалы | 16 | 4 | material | latin-exact | 18.0x |

## Truncated / misspelled query tokens

Tokens whose only in-corpus match is a word they are a strict prefix of (queries cut mid-word) or a near-identical spelling. These are *not* jargon and should probably be excluded from a synonym table rather than mapped.

| token | occ | df_active | nearest in-corpus word | method |
|---|---|---|---|---|
| прорамми | 18 | 0 | прораммировать | cyrillic-prefix |
| обсуж | 10 | 0 | обсуждения | cyrillic-prefix |
| недост | 8 | 0 | недоступен | cyrillic-prefix |

## Full dictionary

`artifacts/handoff/oov-jargon.json` carries all 2877 tokens sorted by `rank_score`, each with occurrences, distinct queries/events/sessions/scopes, top scopes, `corpus_df_active`, `corpus_df_all_nodes`, `fts_df_active`, the OOV flag and the `canonicalization_proposal` block (canonical, method, evidence, alternatives, bridge_gap_ratio).

**EVERY canonicalization entry is a PROPOSAL. This script does not edit tokenize()/_SYNONYMS, the bm25/FTS path or the trigger channel.**

