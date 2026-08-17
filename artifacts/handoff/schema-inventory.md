# Handoff inventory: active `level:schema` nodes and their triggers

Read-only evidence for the *when_to_use triggers* goal. Nothing here was written back to the database and no source module was modified.

## Provenance

- generator: `scripts/handoff_inventory.py` v1.1.0
- generated_at / slice cutoff (`--as-of`): `2026-08-17T15:45:00Z` (slice frozen: True)
- database: `/home/sfx/.local/share/living-memory/global.sqlite3` (532647936 bytes, mtime 2026-08-17T16:41:10Z)
- access: sqlite3.connect(file:<path>?mode=ro, uri=True) + PRAGMA query_only=1 + one deferred read transaction pinning a single WAL snapshot; MemoryStore is never imported (its constructor migrates and writes)
- git commit: `254539ead919e77005c615c43db1ebf3695037a7` (branch `vector-recall-chunking-harness--handoff-inventory`, worktree dirty: True) -- rebase-fragile, see below
- generator sha256: `a3ef0412cc3c8b20cc22be7c498c2ad8a9caee645a455a5c175916ff7e3cae6d` (rebase-invariant anchor; `sha256sum scripts/handoff_inventory.py`)
- runtime: python 3.12.3, sqlite 3.45.1
- row counts: first_event_at = 2026-05-14T20:05:23Z, last_event_at = 2026-08-17T15:34:52Z, nodes_active = 12858, nodes_in_slice = 16664, nodes_total = 16679, recall_events_in_slice = 54759, recall_events_scanned = 54759, recall_events_total = 54772, result_rows_scanned = 367151, schema_nodes_active_in_slice = 433

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

## The rule being measured

- source: `src/living_memory/retrieval.py:437-465`
- `overlap = |tokenize(query) & tokenize(trigger)| / |tokenize(trigger)|`
- fires when `overlap >= 0.5`, scoring `0.95 + 0.05 * overlap`
- the channel only *adds a candidate*; whether that candidate is delivered depends on the final ranking and on `max_results`.

## Population

- active schema nodes in slice: **433** (all 433 carry `context.trigger`, 433 tokenize to a non-empty trigger set)
- with `task_pattern`: 365; with `procedure_id`: 315
- distinct trigger texts: **274** (distinct token sets: 273) -- 159 schemas repeat a trigger that already exists
- content size: min 110, median 2074.0, max 132819 chars

### Scope distribution

| scope | schemas |
|---|---|
| project:x | 188 |
| project:octopus | 94 |
| project:ae | 89 |
| project:online | 47 |
| global | 8 |
| project:lm | 4 |
| project:gas-stations-ui | 2 |
| project:online-sitemaps | 1 |

### Trigger token-count histogram

| trigger tokens | schemas |
|---|---|
| 1 | 1 |
| 2 | 103 |
| 3 | 88 |
| 4 | 77 |
| 5 | 50 |
| 6 | 40 |
| 7 | 39 |
| 8 | 8 |
| 9 | 12 |
| 10 | 7 |
| 11 | 4 |
| 12 | 2 |
| 18 | 1 |
| 19 | 1 |

## Firing: what actually surfaced

- schema deliveries recorded in `recall_events.results`: **68037**
- of those, trigger-channel hits (`trigger_score > 0`): **58462**; delivered by other channels only: **9575**
- schemas ever delivered: **327** / 433; **never delivered: 106**
- never trigger-fired: 117; delivered only via bm25/vector/graph: 11

Channel mix over delivered schema results (a result can carry several channels):

| channel | delivered schema results |
|---|---|
| trigger | 58462 |
| vector | 56355 |
| bm25 | 51439 |
| graph | 12935 |

## Firing: what the trigger rule *would* have matched

replay of the live trigger rule (retrieval.py:443-465) over every recorded query: scope-filtered by recall_events.resolved_scopes and restricted to events at or after the schema's created_at.

- simulated trigger matches: **86330** vs **58462** actually delivered (67.7% of matches reached the caller)
- schemas that would match at least once: 369; never matched: **64**
- schemas that matched but were never delivered: **51**
- matches in events where more schemas fired than `max_results` could return: **50961**

Simultaneous trigger matches per event (how crowded the channel gets):

| schemas matching one query | events |
|---|---|
| 1 | 4414 |
| 2 | 2650 |
| 3 | 1124 |
| 4 | 3489 |
| 5 | 1119 |
| 6 | 1724 |
| 7 | 804 |
| 8 | 100 |
| 9 | 28 |
| 10 | 210 |
| 11 | 107 |
| 12 | 442 |
| 13 | 96 |
| 14 | 164 |
| 15 | 8 |
| 16 | 10 |
| 17 | 17 |
| 18 | 103 |
| 19 | 62 |
| 20 | 179 |
| 21 | 8 |
| 22 | 13 |
| 23 | 6 |
| 24 | 15 |
| 25 | 13 |
| 26 | 137 |
| 27 | 92 |
| 28 | 4 |
| 29 | 2 |
| 30 | 140 |
| 38 | 3 |
| 39 | 102 |
| 40 | 23 |
| 41 | 14 |
| 42 | 2 |

## Top 40 firers

| id | scope | trigger | delivered | trigger hits | simulated matches | delivered share |
|---|---|---|---|---|---|---|
| 01KRVVE5XZQ5JGHBAT1WGD820Z | project:ae | reopen lesson | 5656 | 5592 | 5607 | 99.7% |
| 01KS57467DM9E5S861BSPQD19A | project:ae | reopen lesson | 5132 | 4911 | 5011 | 98.0% |
| 01KSB3GDM0J3ZB24QSRA97RRWJ | project:ae | reopen lesson | 4614 | 4592 | 4646 | 98.8% |
| 01KSB3GDMAKA684W129G3EPGBY | project:ae | reopen lesson | 4481 | 4475 | 4646 | 96.3% |
| 01KRXTWDXJ28QHFVS7G68NB3J3 | project:octopus | reopen lesson | 2430 | 2382 | 2382 | 100.0% |
| 01KRXTWDWQ6AAF57VH1HGR8KDN | project:octopus | reopen lesson | 2414 | 2381 | 2382 | 100.0% |
| 01KRXTWDXNQJXZBQG6VTWMZ673 | project:octopus | reopen lesson | 2399 | 2380 | 2382 | 99.9% |
| 01KRXTWDXPCQNSDZH1VVEZFF8D | project:octopus | reopen lesson | 2387 | 2382 | 2382 | 100.0% |
| 01KRXTWDXKAPNF6K9SZHC7NJXV | project:octopus | reopen lesson | 2383 | 2380 | 2382 | 99.9% |
| 01KS0JK8AXF663CKZB1QYC434S | project:online | reopen lesson | 2230 | 1918 | 1924 | 99.7% |
| 01KRWVEF08GR0365E296ZDNDWH | project:ae | verify pass | 1455 | 1297 | 1297 | 100.0% |
| 01KRWVEEZZJ0Y6FR1N5CKPNCWR | project:ae | verify pass | 1433 | 1297 | 1297 | 100.0% |
| 01KSX9EN813VPYZFRTF0SDFZF6 | project:ae | reopen lesson | 1417 | 1412 | 1541 | 91.6% |
| 01KS7AWDACPYWXHQ1JZ5GZSZ58 | project:online | reopen lesson | 1304 | 1293 | 1825 | 70.8% |
| 01KTS6WQMERXDVPXCD1Z34Y702 | project:x | depth4 plastic state residency | 1273 | 126 | 126 | 100.0% |
| 01KV47JDFGY0B2ACYFW6AM3VA6 | project:x | the ceil expanded curriculum training v2 v2 training run | 1019 | 89 | 89 | 100.0% |
| 01KRVVE5XREYC63HETRYX60F3A | project:ae | stub agent healthy tree dispatch | 880 | 0 | 0 | n/a |
| 01KTSB8N0BN0GMK48N8DB5H03C | project:online | reopen lesson | 862 | 840 | 840 | 100.0% |
| 01KTSB8MZND4ZSVB90P2KMB6MY | project:online | reopen lesson | 822 | 810 | 840 | 96.4% |
| 01KVC1S15SMG3WC1G4WDAB25CR | project:x | reopen lesson | 807 | 801 | 801 | 100.0% |
| 01KT3J5PVW3DXDWWPXZH81XX7C | project:online | reopen lesson | 792 | 786 | 1362 | 57.7% |
| 01KV5DJDYPP51KZCCZMZ4Z888Y | project:online | reopen lesson | 766 | 754 | 792 | 95.2% |
| 01KRVVE5XNXZW1XY9K84XCPNE9 | project:ae | implementation note | 719 | 15 | 16 | 93.8% |
| 01KRW4J9MEJ8M1MPCNKABB05XH | project:octopus | final verification | 706 | 661 | 663 | 99.7% |
| 01KVYS3SMGS83R2SKX7ZWTBDJN | project:online | reopen lesson | 649 | 642 | 657 | 97.7% |
| 01KRWVEF00JXWVY4N0T6J9CN84 | project:ae | verification result | 616 | 456 | 459 | 99.3% |
| 01KRXY78CSP7W5CJMC6GX21EBC | project:octopus | gate repair | 614 | 602 | 663 | 90.8% |
| 01KRXY78CFNNE6RP4T7STP82T1 | project:octopus | tree decomposition | 591 | 508 | 511 | 99.4% |
| 01KRY9XWFT4KYC8J652TS7CX50 | global | octopus gemini flash breakthrough decomposition v1 | 544 | 122 | 123 | 99.2% |
| 01KRXTWDWPWEHKCYEYFHFDF6MT | project:octopus | goal execution | 495 | 486 | 498 | 97.6% |
| 01KTSKWMRG1BKBVJPNHVPGKS8E | project:ae | dashboard goal api | 445 | 112 | 113 | 99.1% |
| 01KT7QN1HD0XG3P4EB4RZ330XM | project:ae | dashboard goal api supervision | 432 | 174 | 174 | 100.0% |
| 01KS5746895WYVF6AHVJCFF42A | project:octopus | ocpa generative action substrate v1 decomposition | 409 | 71 | 71 | 100.0% |
| 01KSB3GDMJX4CJKGPEG93RYPWE | project:ae | active goal supervision repair | 388 | 129 | 131 | 98.5% |
| 01KRVVE5XP69G159YJ94JTM020 | project:ae | task outcome | 375 | 249 | 249 | 100.0% |
| 01KRVVF4H7HACZRQH5E6AXR422 | project:octopus | acceptance repair | 374 | 341 | 341 | 100.0% |
| 01KRVVE5XQ9CJ7JPVPWFW9102F | project:ae | task outcome | 339 | 249 | 249 | 100.0% |
| 01KRVVF4H8V2NJWT2PDF2QWJ1X | project:octopus | tree decomposition critique | 331 | 230 | 232 | 99.1% |
| 01KRVVE5XMB03WJN1E8XGKRY12 | project:ae | tree goal direct execution | 320 | 153 | 168 | 91.1% |
| 01KTNGMYMRWNSBNRWG7R03BJ6Y | project:online | reopen lesson | 283 | 257 | 891 | 28.8% |

## Never delivered (106 schemas)

Split by cause -- the actionable distinction for the trigger goal: **51** of these DID match the trigger rule and still never reached a caller (they need less competition), while **55** never matched any recorded query at all (they need a better trigger).

| id | scope | trigger | trigger tokens | simulated matches | eligible events |
|---|---|---|---|---|---|
| 01KVYS3SM9R8B1SXMEVZWFCTYD | project:online | reopen lesson | 2 | 657 | 2874 |
| 01KVYS3SMDH4A5Z5J5V882HPY3 | project:online | reopen lesson | 2 | 657 | 2874 |
| 01KVYS3SMN67XK6SPSHNQ9321B | project:online | reopen lesson | 2 | 657 | 2874 |
| 01KX5CXCRXZ4P84RTT6G09R40E | project:online | reopen lesson | 2 | 492 | 2221 |
| 01KX5CXCSHKJBFYAK905VY3JPH | project:online | reopen lesson | 2 | 492 | 2221 |
| 01KX5CXCV19PV7XVMD7SY0DQV4 | project:online | reopen lesson | 2 | 492 | 2221 |
| 01KX5CXCVJ1C4E6NFACFGXEMD2 | project:online | reopen lesson | 2 | 492 | 2221 |
| 01KXDD5HGJM4D4P43T0CC82HMX | project:online | reopen lesson | 2 | 359 | 1839 |
| 01KXQKC13VEE38PYHH5H2BSRV1 | project:online | reopen lesson | 2 | 279 | 1530 |
| 01KXQKC1664RXD7J4NF4KA2253 | project:online | reopen lesson | 2 | 279 | 1530 |
| 01KXQKC16H85527EZ7W674ZJYZ | project:online | reopen lesson | 2 | 279 | 1530 |
| 01KZDN9CAGE14TE5SMR5SGW0A0 | project:online | reopen lesson | 2 | 139 | 921 |
| 01KZDN9CBEGS3V494AH17PK3MF | project:online | reopen lesson | 2 | 139 | 921 |
| 01KZDN9CBKBC8PYZKWX7KN0RMD | project:online | reopen lesson | 2 | 139 | 921 |
| 01KZDN9CBT96Q09V0CC20BPMS5 | project:online | reopen lesson | 2 | 139 | 921 |
| 01KZDN9CC1C1D3ZTYNZEQFX4W1 | project:online | reopen lesson | 2 | 139 | 921 |
| 01KZDN9CCTFQ9F64A79QMDA8CK | project:online | reopen lesson | 2 | 139 | 921 |
| 01KZDN9CD45WAW2B5SF7GNZQ87 | project:online | reopen lesson | 2 | 139 | 921 |
| 01KZDN9CDBTGYEFZMAZCE5YTCT | project:online | reopen lesson | 2 | 139 | 921 |
| 01KTSKWMXKBSC1MX818AA1E93X | project:ae | dashboard goal api supervision | 4 | 113 | 4134 |
| 01KT7C5RN928Z1F5HB3NGA5HQA | project:octopus | active goal supervision | 3 | 67 | 288 |
| 01KT7C5RQHF7PYEJEHJEFJYXNT | project:octopus | active goal supervision | 3 | 67 | 288 |
| 01KT7C5RQX52K26VT5QQ7ZK9AD | project:octopus | active goal supervision | 3 | 67 | 288 |
| 01KT7QN1GG021T8G2VVWY4F39K | project:ae | active goal supervision | 3 | 39 | 4816 |
| 01KT7QN1GN8XQ7NDG73ETDFNQW | project:ae | active goal supervision | 3 | 39 | 4816 |
| 01KT7QN1GY5CQXZ364GTGRM0A4 | project:ae | active goal supervision | 3 | 39 | 4816 |
| 01KT7QN1HV4VER8WZJETQZ59HG | project:ae | active goal supervision | 3 | 39 | 4816 |
| 01KT7QN1J9SWGPJM7JWHQW8WM3 | project:ae | active goal supervision | 3 | 39 | 4816 |
| 01KT7QN1JRN5H6NJDFS3A89W6Q | project:ae | active goal supervision | 3 | 39 | 4816 |
| 01KT7QN1JXVAVTWJPMCJ35ZEX7 | project:ae | active goal supervision | 3 | 39 | 4816 |
| 01KT87354TB1QE9AD4RSHJDQ1R | project:octopus | active goal supervision | 3 | 27 | 123 |
| 01KT873561Z5ZY1NE8R6VJBRPK | project:octopus | active goal supervision | 3 | 27 | 123 |
| 01KT87356FB2Y11PHB2TRJ1D4X | project:octopus | active goal supervision | 3 | 27 | 123 |
| 01KT87356XXPSRV55P0WN4GHMY | project:octopus | active goal supervision | 3 | 27 | 123 |
| 01KT8735878B6RR9DJAQXVRBF0 | project:octopus | active goal supervision | 3 | 27 | 123 |
| 01KSB3GDG6D8R8BGNCV5769Y2V | project:octopus | tree decomposition feature | 3 | 17 | 10135 |
| 01KZXPK816P943M7P92HVHXSZJ | project:online | reopen lesson | 2 | 15 | 119 |
| 01KT7C5RSC0QTHTE4NNDRHEQSF | project:octopus | supervisor scan | 2 | 12 | 288 |
| 01KTSKWMY9YNFE94T6D8VXKEZF | project:ae | active goal supervision | 3 | 5 | 4134 |
| 01KSQ3355MC4VJ6Q72PWXSV5P1 | project:octopus | corrected tree decomposition | 3 | 4 | 6827 |

_66 further never-delivered schemas are in the JSON under `summary.never_delivered_ids`._

## Trigger duplication

19 trigger texts are shared by more than one schema; they account for 178 of 433 schemas. Duplicates fire together by construction and then compete for the same result slots.

| trigger | schemas | scopes |
|---|---|---|
| reopen lesson | 64 | project:ae:12, project:gas-stations-ui:2, project:octopus:7, project:online:41, project:online-sitemaps:1, project:x:1 |
| active goal supervision | 40 | project:ae:17, project:octopus:23 |
| dashboard goal api supervision | 14 | project:ae:14 |
| tree decomposition repair | 8 | project:octopus:1, project:x:7 |
| tree decomposition verification | 7 | project:ae:1, project:octopus:6 |
| task outcome | 6 | project:ae:6 |
| tree decomposition critique | 6 | project:octopus:6 |
| verify pass | 6 | project:ae:6 |
| ae dashboard supervision 20260604 octopus scan | 4 | project:ae:4 |
| octopus supervision | 3 | project:ae:1, project:octopus:2 |
| stagnation detection | 3 | project:ae:3 |
| tree decomposition | 3 | project:octopus:3 |
| acceptance repair | 2 | project:octopus:2 |
| active goal supervision repair | 2 | project:ae:2 |
| anomaly detection | 2 | project:ae:2 |
| expanded curriculum training v2 replacement | 2 | project:x:2 |
| jira ticket review tree goal | 2 | global:1, project:online:1 |
| ocpa generative action substrate v1 decomposition | 2 | global:1, project:octopus:1 |
| supervisor scan | 2 | project:octopus:2 |

## Trigger-token collisions

Collision rule: `shared >= 0.5*|A| and shared >= 0.5*|B| -- the shared tokens alone fire both schemas under retrieval.py:461`.

- colliding pairs: **4495**
- schemas involved in at least one collision: **329** / 433
- collision clusters (connected components): 40
- groups with byte-identical trigger token sets: 20

Largest clusters:

| size | scopes | example triggers |
|---|---|---|
| 129 | project:ae, project:lm, project:octopus, project:x | acceptance repair / active goal supervision / active goal supervision repair / active goal supervision scan |
| 64 | project:ae, project:gas-stations-ui, project:octopus, project:online, project:online-sitemaps, project:x | reopen lesson |
| 22 | project:x | nw7 r2 fixed final arm execution blocked / nw7 r2 fixed final six arm execution runtime blocked / nw7 r2 fixed final train eval v3 parent repair / nw7 r2 fixed final train eval v3 planning |
| 11 | project:x | p15c a03 datasets / p15c a03 protocol reset / p15c a03 train unscored / p15c a04 datasets generation |
| 6 | project:ae | verify pass |
| 6 | project:x | the ceil nw5 train scoreblind arms / the ceil nw6 train scoreblind arms nw6 scoreblind budget replan / the ceil nw6 train scoreblind arms nw6 stock frontier arm reduced stock frontier reduced training run / the ceil nw6 train scoreblind arms nw6 stock frontier arm reduced stock frontier reduced training run stock factorized bptt throughput repair stock factorized source repair v2 stock contract preservation audit |
| 5 | project:ae | anomaly detection / stagnation detection |
| 5 | project:x | expanded curriculum training v2 replacement / expanded curriculum v2 train measure decomposition / the ceil expanded curriculum training v2 fresh evidence / the ceil expanded curriculum training v2 v2 training run |
| 4 | project:ae | ae dashboard supervision 20260604 octopus scan |
| 4 | project:x | runtime bottleneck elimination x postfix frontier rerun selection / runtime bottleneck elimination x precision policy frontier refresh mixed precision learning gates / runtime bottleneck elimination x precision policy frontier refresh v2 / runtime bottleneck elimination x precision policy frontier refresh v2 mixed precision learning gates |
| 4 | project:x | p15c post a05 ordered candidate subtree / the ceil p15c a05 protocol freeze / the ceil p15c a05 protocol freeze a05 candidate contract freeze / the ceil p15c post a05 research expansion |
| 4 | project:x | nw5 dual path forward cpu / nw5 dual path runtime / the ceil nw5 dual path runtime dual path state surface / the ceil nw5 dual path runtime v2 |
| 3 | global, project:octopus | ocpa generative action substrate v1 decomposition / ocpa generative action substrate v1 restart decomposition |
| 3 | global, project:octopus | goal smarter simpler faster decomposition / goal smarter simpler faster decomposition 2026 05 26 / goal smarter simpler faster useful throughput decomposition 2026 05 27 |
| 3 | project:x | post v2 return curve inplace refresh / the ceil post v2 return curve in place / the ceil return curve falsifier import |
| 3 | project:x | data scaling v2 fresh successor protocol / data scaling v2 post v5 replacement / the ceil fresh scoreblind data scaling protocol reset |
| 3 | project:x | p7b frontier repair or escalation / p7b frontier repair or escalation parent verify / the ceil p7b frontier repair or escalation p7b scoreblind battery evidence |
| 3 | project:x | p15c second wave methods prereg decomposition / p15c second wave preregistration binding / p15c second wave scoreblind execution chain decomposition |
| 3 | project:x | nw6 stock bounded throughput smoke / nw6 stock factorized throughput repair decomposition / stock factorized source throughput projection benchmark |
| 3 | project:x | nw7 parent repair after failed successor selection / nw7 successor v1 repair after no eligible successor / nw7 successor v2 eval selection no eligible |
| 3 | project:ae | dashboard agent chats / dashboard agent chats persistent runtime / dashboard chat auto title |
| 2 | global, project:octopus | octopus gemini flash breakthrough decomposition v1 / octopus goal rise ocpa efe gemini flash decomposition 2026 05 19 |
| 2 | project:octopus | supervisor scan |
| 2 | project:x | production efficiency frontier x mixed precision training / production efficiency frontier x mixed precision training mixed precision training gates |
| 2 | project:x | matrix runner carry probes / matrix runner direct expansion |
| 2 | project:x | depth4 plastic init residency design / depth4 plastic state residency |
| 2 | project:x | merge gate smoke timeout attribution / smoke openmp merge gate |
| 2 | project:x | per role recast execution / per role recast execution parent repair |
| 2 | project:x | cuda plastic mixed dtype / plastic mixed dtype kernels mixed plastic equivalence tests |
| 2 | project:x | recurrent gemm mixed role dispatch / recurrent gemm mixed role dispatch parent repair |
| 2 | project:x | fresh v3 fixed epoch16 training / fresh v3 fixed epoch16 training v2 acceptance |
| 2 | project:x | fresh v2 scoreblind evidence replacement / p7b v2 fresh evidence |
| 2 | project:x | fresh scoreblind successor repair / p15c fresh scoreblind successor search |
| 2 | project:x | the ceil fresh scoreblind successor successor temporal producer repair curriculum temporal contrast generator / the ceil fresh scoreblind successor successor temporal producer repair trainer retention consistency |
| 2 | project:x | frontier v2 checkpoint seal / frontier v2 raw rerun |
| 2 | project:x | the ceil p15c scoreblind registry policy reset a05 invalid history ledger / the ceil p15c scoreblind registry policy reset post a05 ledger exhaustion init |
| 2 | project:x | p15c a05 train checkpoint provenance / p15c a05 train score once decomposition |
| 2 | project:x | stock factorized rank scratch source repair v2 / stock factorized source repair v2 parent verification |
| 2 | project:x | return curve saturation closure planning / the ceil return curve saturation closure parent verification |
| 2 | global, project:online | jira ticket review tree goal |

## Hair triggers (104 schemas with 1-2 trigger tokens)

With `overlap >= 0.5`, a two-token trigger fires on ONE shared token and a one-token trigger fires on any query containing that token.

| id | scope | trigger | tokens | simulated matches | trigger hits |
|---|---|---|---|---|---|
| 01KRVVE5XZQ5JGHBAT1WGD820Z | project:ae | reopen lesson | lesson, reopen | 5607 | 5592 |
| 01KS57467DM9E5S861BSPQD19A | project:ae | reopen lesson | lesson, reopen | 5011 | 4911 |
| 01KSB3GDM0J3ZB24QSRA97RRWJ | project:ae | reopen lesson | lesson, reopen | 4646 | 4592 |
| 01KSB3GDMAKA684W129G3EPGBY | project:ae | reopen lesson | lesson, reopen | 4646 | 4475 |
| 01KRXTWDWQ6AAF57VH1HGR8KDN | project:octopus | reopen lesson | lesson, reopen | 2382 | 2381 |
| 01KRXTWDXJ28QHFVS7G68NB3J3 | project:octopus | reopen lesson | lesson, reopen | 2382 | 2382 |
| 01KRXTWDXKAPNF6K9SZHC7NJXV | project:octopus | reopen lesson | lesson, reopen | 2382 | 2380 |
| 01KRXTWDXNQJXZBQG6VTWMZ673 | project:octopus | reopen lesson | lesson, reopen | 2382 | 2380 |
| 01KRXTWDXPCQNSDZH1VVEZFF8D | project:octopus | reopen lesson | lesson, reopen | 2382 | 2382 |
| 01KS914B1GPHQSB3A54M1Z18KV | project:octopus | reopen lesson | lesson, reopen | 2041 | 33 |
| 01KS0JK8AXF663CKZB1QYC434S | project:online | reopen lesson | lesson, reopen | 1924 | 1918 |
| 01KS7AWDACPYWXHQ1JZ5GZSZ58 | project:online | reopen lesson | lesson, reopen | 1825 | 1293 |
| 01KSX9EN813VPYZFRTF0SDFZF6 | project:ae | reopen lesson | lesson, reopen | 1541 | 1412 |
| 01KT3J5PVW3DXDWWPXZH81XX7C | project:online | reopen lesson | lesson, reopen | 1362 | 786 |
| 01KRWVEEZZJ0Y6FR1N5CKPNCWR | project:ae | verify pass | pas, verify | 1297 | 1297 |
| 01KRWVEF08GR0365E296ZDNDWH | project:ae | verify pass | pas, verify | 1297 | 1297 |
| 01KWQ5YJCBF7C2TAQFDCNNV80F | project:ae | reopen lesson | lesson, reopen | 1002 | 217 |
| 01KWQ5YJCYJ6QHXHVT4WX8TR8N | project:ae | reopen lesson | lesson, reopen | 1002 | 54 |
| 01KTCH9T0TJJV17H9K3KB6WSV1 | project:online | reopen lesson | lesson, reopen | 947 | 205 |
| 01KTNGMYMNYS1RB8EKX5HHBDAB | project:online | reopen lesson | lesson, reopen | 891 | 189 |
| 01KTNGMYMRWNSBNRWG7R03BJ6Y | project:online | reopen lesson | lesson, reopen | 891 | 257 |
| 01KY5HQ9FH3HNAEPHMKRYYSRQ8 | project:ae | reopen lesson | lesson, reopen | 847 | 11 |
| 01KY5HQ9FT41STY773NT9JT4VS | project:ae | reopen lesson | lesson, reopen | 847 | 13 |
| 01KY5HQ9G26J8WTVQD69ENWCB7 | project:ae | reopen lesson | lesson, reopen | 847 | 41 |
| 01KTSB8MZND4ZSVB90P2KMB6MY | project:online | reopen lesson | lesson, reopen | 840 | 810 |
| 01KTSB8N0BN0GMK48N8DB5H03C | project:online | reopen lesson | lesson, reopen | 840 | 840 |
| 01KTSB8N2E6HA4CSF2DXBQYK80 | project:online | reopen lesson | lesson, reopen | 840 | 17 |
| 01KTSB8N2ZJB8CBS46M65B1TE1 | project:online | reopen lesson | lesson, reopen | 840 | 9 |
| 01KTSB8N3197SD4V5BHHWDAZSR | project:online | reopen lesson | lesson, reopen | 840 | 71 |
| 01KVC1S15SMG3WC1G4WDAB25CR | project:x | reopen lesson | lesson, reopen | 801 | 801 |
| 01KV5DJDYPP51KZCCZMZ4Z888Y | project:online | reopen lesson | lesson, reopen | 792 | 754 |
| 01KV5DJE2J3Y5X08K0PFGHA0NA | project:online | reopen lesson | lesson, reopen | 792 | 9 |
| 01KVB3XD0G35N712AGWVDYYJAG | project:online | reopen lesson | lesson, reopen | 720 | 20 |
| 01KRW4J9MEJ8M1MPCNKABB05XH | project:octopus | final verification | final, verification | 663 | 661 |
| 01KRXY78CSP7W5CJMC6GX21EBC | project:octopus | gate repair | gate, repair | 663 | 602 |
| 01KVYS3SM9R8B1SXMEVZWFCTYD | project:online | reopen lesson | lesson, reopen | 657 | 0 |
| 01KVYS3SMDH4A5Z5J5V882HPY3 | project:online | reopen lesson | lesson, reopen | 657 | 0 |
| 01KVYS3SMGS83R2SKX7ZWTBDJN | project:online | reopen lesson | lesson, reopen | 657 | 642 |
| 01KVYS3SMN67XK6SPSHNQ9321B | project:online | reopen lesson | lesson, reopen | 657 | 0 |
| 01KVYS3SMQ8EEAQKD0GH9Z1BJ9 | project:online | reopen lesson | lesson, reopen | 657 | 2 |

## Full inventory

`artifacts/handoff/schema-inventory.json` carries every active schema with `id, scope, task_pattern, procedure_id, trigger, trigger_tokens, content_chars, created_at, access_count, usefulness_score` and the full `firing` block (delivered/trigger/other-channel counts, channel mix, best rank, distinct sessions/scopes/queries, eligible events, simulated matches).

