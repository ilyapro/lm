# Recall effect by day — sfx (2026-09-07..2026-09-27)

- DB: `/home/sfx/.local/share/living-memory/global.sqlite3` opened as `file:/home/sfx/.local/share/living-memory/global.sqlite3?mode=ro`; generated 2026-09-27T11:18:24Z.
- Used = recall_credit_ledger row for (event, node) with basis in ['grounded', 'lookup']. Shares are over events with ≥1 delivered node.
- Ledger bases present: grounded, lookup; explicit credit present: False; recall_feedback_marks present: False.
- A/B exclusion per transport session. Receipt ids scanned: 1604 (matched sessions in window: 925). Fixture task families from harness corpus manifests: development-battery, development-intake, development-ledger, development-packaging, development-transfer, development-water, ev-calibration, ev-flood, ev-labels, ev-parcel, ev-recovery, ev-source-malformed, ev-source-missing, ev-source-unreadable, ev-stock, ev-timetable.

## Window total (kept traffic)

Events 12497, kept 9893, excluded 2604. rank-1 used 7.7%, top-3 used 19.1%, useful nodes 9.3% (4686/50275).

## Daily metric (kept traffic)

| day | events | excl | kept | closed | never | r1 used | top3 used | useful nodes | grounded r1/top3/nodes | lookup r1/top3/nodes | explicit r1/top3/nodes | legacy-filter r1 / useful |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 2026-09-07 | 1226 | 0 | 1226 | 973 | 253 | 5.6% | 12.8% | 8.3% | 1.9% / 3.7% / 1.3% | 3.8% / 9.3% / 7.0% | 0.0% / 0.0% / 0.0% | 5.8% / 8.6% |
| 2026-09-08 | 715 | 0 | 715 | 551 | 164 | 6.3% | 17.2% | 9.4% | 4.9% / 9.2% / 2.8% | 1.4% / 8.7% / 6.6% | 0.0% / 0.0% / 0.0% | 6.2% / 9.4% |
| 2026-09-09 | 812 | 0 | 812 | 704 | 108 | 8.4% | 17.7% | 9.2% | 6.7% / 11.5% / 4.1% | 1.7% / 7.1% / 5.0% | 0.0% / 0.0% / 0.0% | 8.7% / 9.3% |
| 2026-09-10 | 204 | 0 | 204 | 180 | 24 | 19.6% | 44.6% | 19.4% | 16.7% / 24.0% / 7.9% | 2.9% / 26.0% / 11.5% | 0.0% / 0.0% / 0.0% | 19.4% / 20.5% |
| 2026-09-11 | 46 | 0 | 46 | 40 | 6 | 23.9% | 37.0% | 20.4% | 21.7% / 26.1% / 13.2% | 2.2% / 17.4% / 7.3% | 0.0% / 0.0% / 0.0% | 25.0% / 20.8% |
| 2026-09-12 | 277 | 0 | 277 | 168 | 109 | 15.9% | 29.2% | 16.0% | 14.1% / 16.6% / 5.7% | 1.8% / 18.1% / 10.3% | 0.0% / 0.0% / 0.0% | 15.2% / 16.2% |
| 2026-09-13 | 196 | 0 | 196 | 189 | 7 | 12.8% | 40.8% | 14.8% | 10.7% / 17.3% / 4.8% | 2.0% / 28.1% / 10.0% | 0.0% / 0.0% / 0.0% | 13.5% / 14.9% |
| 2026-09-14 | 1118 | 0 | 1118 | 938 | 180 | 6.6% | 16.8% | 8.1% | 4.3% / 8.0% / 2.6% | 2.3% / 9.5% / 5.5% | 0.0% / 0.0% / 0.0% | 6.7% / 8.2% |
| 2026-09-15 | 276 | 0 | 276 | 231 | 45 | 6.2% | 14.5% | 7.4% | 5.1% / 9.4% / 3.5% | 1.1% / 5.8% / 3.8% | 0.0% / 0.0% / 0.0% | 8.3% / 9.0% |
| 2026-09-16 | 1005 | 720 | 285 | 189 | 96 | 15.1% | 30.2% | 9.8% | 12.3% / 19.7% / 7.0% | 2.8% / 10.5% / 2.7% | 0.0% / 0.0% / 0.0% | 20.6% / 9.8% |
| 2026-09-17 | 911 | 673 | 238 | 147 | 91 | 17.2% | 33.6% | 11.8% | 14.3% / 23.9% / 6.4% | 2.9% / 11.8% / 5.4% | 0.0% / 0.0% / 0.0% | 29.0% / 15.4% |
| 2026-09-18 | 232 | 0 | 232 | 190 | 42 | 16.8% | 30.6% | 12.9% | 14.7% / 19.8% / 8.0% | 2.2% / 12.5% / 4.9% | 0.0% / 0.0% / 0.0% | 18.4% / 13.5% |
| 2026-09-19 | 1002 | 382 | 620 | 491 | 129 | 6.9% | 20.2% | 9.9% | 5.8% / 9.8% / 3.1% | 1.1% / 11.1% / 6.8% | 0.0% / 0.0% / 0.0% | 7.6% / 11.0% |
| 2026-09-20 | 803 | 262 | 541 | 383 | 158 | 7.6% | 17.0% | 5.8% | 5.0% / 7.6% / 1.8% | 2.6% / 10.7% / 4.0% | 0.0% / 0.0% / 0.0% | 8.5% / 8.5% |
| 2026-09-21 | 819 | 0 | 819 | 548 | 271 | 7.4% | 16.1% | 5.9% | 6.5% / 10.1% / 2.8% | 1.0% / 6.8% / 3.1% | 0.0% / 0.0% / 0.0% | 8.3% / 6.6% |
| 2026-09-22 | 259 | 3 | 256 | 181 | 75 | 9.8% | 22.7% | 8.2% | 9.4% / 16.0% / 4.5% | 0.4% / 9.0% / 3.7% | 0.0% / 0.0% / 0.0% | 11.2% / 10.8% |
| 2026-09-23 | 850 | 564 | 286 | 220 | 66 | 5.6% | 16.1% | 7.1% | 3.9% / 6.6% / 1.6% | 1.8% / 10.1% / 5.5% | 0.0% / 0.0% / 0.0% | 38.2% / 42.3% |
| 2026-09-24 | 452 | 0 | 452 | 367 | 85 | 4.0% | 18.8% | 9.1% | 3.5% / 8.0% / 2.8% | 0.4% / 11.3% / 6.2% | 0.0% / 0.0% / 0.0% | 4.0% / 9.3% |
| 2026-09-25 | 414 | 0 | 414 | 322 | 92 | 5.1% | 16.2% | 7.7% | 4.8% / 11.3% / 4.2% | 0.2% / 6.0% / 3.5% | 0.0% / 0.0% / 0.0% | 5.2% / 8.1% |
| 2026-09-26 | 607 | 0 | 607 | 406 | 201 | 2.1% | 15.5% | 11.3% | 1.8% / 4.0% / 1.4% | 0.3% / 11.5% / 9.9% | 0.0% / 0.0% / 0.0% | 2.4% / 12.1% |
| 2026-09-27 | 273 | 0 | 273 | 191 | 82 | 2.2% | 13.6% | 9.6% | 2.2% / 2.9% / 1.0% | 0.0% / 10.6% / 8.6% | 0.0% / 0.0% / 0.0% | 2.1% / 10.8% |

## Closed vs never-closed (kept traffic)

| day | closed n | closed r1 | closed top3 | closed useful | never n | never r1 | never top3 | never useful |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-09-07 | 973 | 4.8% | 13.3% | 8.8% | 253 | 8.7% | 11.1% | 5.6% |
| 2026-09-08 | 551 | 8.0% | 20.9% | 10.5% | 164 | 0.6% | 4.9% | 4.5% |
| 2026-09-09 | 704 | 9.4% | 19.6% | 9.9% | 108 | 1.8% | 5.6% | 2.8% |
| 2026-09-10 | 180 | 21.7% | 46.1% | 18.7% | 24 | 4.2% | 33.3% | 27.5% |
| 2026-09-11 | 40 | 27.5% | 42.5% | 22.6% | 6 | 0.0% | 0.0% | 0.0% |
| 2026-09-12 | 168 | 26.2% | 47.6% | 17.5% | 109 | 0.0% | 0.9% | 2.2% |
| 2026-09-13 | 189 | 13.2% | 42.3% | 15.2% | 7 | 0.0% | 0.0% | 2.2% |
| 2026-09-14 | 938 | 7.2% | 19.2% | 8.9% | 180 | 3.3% | 4.4% | 3.1% |
| 2026-09-15 | 231 | 6.5% | 16.4% | 7.7% | 45 | 4.4% | 4.4% | 4.8% |
| 2026-09-16 | 189 | 20.1% | 37.0% | 14.4% | 96 | 5.2% | 16.7% | 3.3% |
| 2026-09-17 | 147 | 25.2% | 44.9% | 13.8% | 91 | 4.4% | 15.4% | 8.2% |
| 2026-09-18 | 190 | 20.5% | 37.4% | 15.2% | 42 | 0.0% | 0.0% | 0.5% |
| 2026-09-19 | 491 | 7.9% | 23.0% | 11.4% | 129 | 3.1% | 9.3% | 4.6% |
| 2026-09-20 | 383 | 10.4% | 22.7% | 8.1% | 158 | 0.6% | 3.2% | 1.1% |
| 2026-09-21 | 548 | 10.2% | 21.9% | 8.9% | 271 | 1.8% | 4.4% | 1.0% |
| 2026-09-22 | 181 | 13.8% | 29.8% | 9.0% | 75 | 0.0% | 5.3% | 4.7% |
| 2026-09-23 | 220 | 7.3% | 20.0% | 8.5% | 66 | 0.0% | 3.0% | 1.0% |
| 2026-09-24 | 367 | 4.6% | 19.6% | 9.1% | 85 | 1.2% | 15.3% | 9.2% |
| 2026-09-25 | 322 | 6.5% | 20.5% | 8.3% | 92 | 0.0% | 1.1% | 2.0% |
| 2026-09-26 | 406 | 3.0% | 16.3% | 10.4% | 201 | 0.5% | 13.9% | 13.1% |
| 2026-09-27 | 191 | 3.1% | 14.1% | 10.8% | 82 | 0.0% | 12.2% | 6.9% |

## Leak check 2026-09-23

Events on 2026-09-23 excluded by the new filter that the legacy filter (scope project:target|repo|x + ledger/billing/tree-context/fixture/kit acceptance) let through: **250** in 213 sessions; by reason {"fixture_provenance": 228, "receipt": 250}.

| filter | kept events | r1 used | top3 used | useful nodes |
|---|---:|---:|---:|---:|
| legacy (scope/keywords) | 492 | 38.2% | 53.2% | 42.3% |
| this script (session evidence) | 286 | 5.6% | 16.1% | 7.1% |

- `9a0c21b35a75` global: ev-calibration reuse retired quadratic calibration fit out/answer.json decision.json
- `abde33b31aee` global: evaluation labels migration L- three digits normalize inputs/task.json answer.json decision.json
- `6559923169c5` global: ev-labels inventory label migration L- three digits normalize duplicates out/answer.json decision.js
- `2ecf2f9b80c1` project:mm: artifact_provenance transfer-prepare working bytes/mode differ from pinned commit tree acceptance.js
- `a0e8e81f94f3` project:mm: artifact provenance transfer-prepare working bytes mode differ from pinned commit tree acceptance.js
- `0ae84976b65e` global: decision json posted canonical_inputs repair replaced_identity
- `ffe31500b5d7` global: ev-recovery shipment posting repair out/answer.json out/decision.json S-81
- `f5db1194a359` global: ev-recovery repair shipment posting verification parent node out/answer.json out/decision.json

## Exclusions per day

| day | excluded events | sessions | by reason (events; a session may carry several) | legacy would drop | legacy leak | legacy over-exclude |
|---|---:|---:|---|---:|---:|---:|
| 2026-09-07 | 0 | 0 | {} | 43 | 0 | 43 |
| 2026-09-08 | 0 | 0 | {} | 26 | 0 | 26 |
| 2026-09-09 | 0 | 0 | {} | 73 | 0 | 73 |
| 2026-09-10 | 0 | 0 | {} | 18 | 0 | 18 |
| 2026-09-11 | 0 | 0 | {} | 2 | 0 | 2 |
| 2026-09-12 | 0 | 0 | {} | 7 | 0 | 7 |
| 2026-09-13 | 0 | 0 | {} | 11 | 0 | 11 |
| 2026-09-14 | 0 | 0 | {} | 55 | 0 | 55 |
| 2026-09-15 | 0 | 0 | {} | 96 | 0 | 96 |
| 2026-09-16 | 720 | 423 | {"fixture_provenance": 151, "fixture_query_family": 58, "fixture_scope": 15, "fixture_task": 273, "synthetic_scope": 223} | 438 | 331 | 49 |
| 2026-09-17 | 673 | 529 | {"fixture_provenance": 287, "fixture_query_family": 65, "fixture_scope": 9, "fixture_task": 231, "receipt": 124} | 518 | 217 | 62 |
| 2026-09-18 | 0 | 0 | {} | 26 | 0 | 26 |
| 2026-09-19 | 382 | 220 | {"fixture_provenance": 91, "receipt": 382} | 437 | 28 | 83 |
| 2026-09-20 | 262 | 130 | {"fixture_provenance": 128, "receipt": 262} | 297 | 54 | 89 |
| 2026-09-21 | 0 | 0 | {} | 146 | 0 | 146 |
| 2026-09-22 | 3 | 3 | {"fixture_query_family": 3} | 45 | 3 | 45 |
| 2026-09-23 | 564 | 480 | {"fixture_provenance": 380, "fixture_query_family": 1, "fixture_scope": 1, "receipt": 562} | 358 | 250 | 44 |
| 2026-09-24 | 0 | 0 | {} | 29 | 0 | 29 |
| 2026-09-25 | 0 | 0 | {} | 28 | 0 | 28 |
| 2026-09-26 | 0 | 0 | {} | 60 | 0 | 60 |
| 2026-09-27 | 0 | 0 | {} | 37 | 0 | 37 |

### Excluded sessions (up to 15 per day, leaked-by-legacy first; full list in the JSON)

**2026-09-16** — 423 sessions

- `6c1b3b4233ab` 24 ev (legacy caught 0), project:_chat_inject_live_e2e_319480_1789588040_orig,project:_chat_inject_live_e2e_319480_1789588046_long_plus,project:_chat_inject_live_e2e_319480_1789588053_head_aligned,project:_chat_inject_live_e2e_319480_1789588060_head_aligned_tight — synthetic_scope: 01M2NW4XDR2PJ4PHHJ35AEWR57 project:_chat_inject_live_e2e_319480_1789588040_orig — “The user's explicit instruction to use the Read tool one file at a time, without”
- `7402359a8011` 22 ev (legacy caught 4), project:_chat_inject_refs4_265035_current_asis,project:_chat_inject_refs4_265035_long_plus — synthetic_scope: 01M2NVWQCS4TC99FBMRYZ9DV7F project:_chat_inject_refs4_265035_current_asis — “The user's explicit instruction to use the Read tool one file at a time, without”
- `ccdd0a6ee41a` 22 ev (legacy caught 4), project:_chat_inject_refs4_244379_long,project:_chat_inject_refs4_244379_long_plus — synthetic_scope: 01M2NVVAKDSQ6SVKXYGE78SN93 project:_chat_inject_refs4_244379_long — “The user's explicit instruction to use the Read tool one file at a time, without”
- `a964fce7da9b` 21 ev (legacy caught 0), project:_chat_inject_opt_4076258_current,project:_chat_inject_opt_4076258_full,project:_chat_inject_opt_4076258_full_tight — synthetic_scope: 01M2NVF0RC237JQFXFV46MA7FQ project:_chat_inject_opt_4076258_current — “The user's explicit instruction to use the Read tool one file at a time, without”
- `61e47d30f450` 20 ev (legacy caught 4), project:_chat_inject_refs3_170206_long_refs,project:_chat_inject_refs3_170206_short_refs — synthetic_scope: 01M2NVRPPRNVHWNJ59GPQA4KYR project:_chat_inject_refs3_170206_short_refs — “The user's explicit instruction to use the Read tool one file at a time, without”
- `1a85a1af1ee2` 18 ev (legacy caught 2), project:_chat_inject_refs2_74901_refs4_distinct,project:_chat_inject_refs2_74901_refs4_repeat — synthetic_scope: 01M2NVNR6EGS327TY21HYDPSC6 project:_chat_inject_refs2_74901_refs4_repeat — “The user's explicit instruction to use the Read tool one file at a time, without”
- `20a34dbf029f` 18 ev (legacy caught 2), project:_chat_inject_live_e2e_329480_1789588091_orig,project:_chat_inject_live_e2e_329480_1789588100_head_aligned_tight — synthetic_scope: 01M2NW6FPJ80MKN5MCQJZSS2DY project:_chat_inject_live_e2e_329480_1789588091_orig — “Walk the sequential-read file chain in this working directory: start with start.”
- `f742fcd291ed` 14 ev (legacy caught 0), project:_chat_inject_refs_7985_baseline_1ref,project:_chat_inject_refs_7985_refs4 — synthetic_scope: 01M2NVKPZ88FERM8H43CV91QFF project:_chat_inject_refs_7985_baseline_1ref — “The user's explicit instruction to use the Read tool one file at a time, without”
- `209e56d4faf1` 5 ev (legacy caught 4), global,project:repo,project:target — fixture_scope: 01M2P1T9PHQJ41FM4NFFXYG3SE project:target within 30 min of proven harness traffic — “ev-recovery repair replace failed child post with repost shipment posting journa”
- `c9654cc12e2b` 5 ev (legacy caught 0), project:_chat_inject_calib_3361120 — synthetic_scope: 01M2NTM3Z2EDS5JXY50DFR7QD4 project:_chat_inject_calib_3361120 — “Walk the sequential-read file chain in this working directory: start with start.”
- `0c0c9ad6a90e` 3 ev (legacy caught 0), project:_chat_inject_live_e2e_2139628_1789584992 — synthetic_scope: 01M2NS832K1BDGEVXT565PB2P4 project:_chat_inject_live_e2e_2139628_1789584992 — “instructs using the Read tool for this step rather than Bash, and that explicit ”
- `241ba63e1757` 3 ev (legacy caught 0), project:_chat_inject_live_e2e_846228_1789582817 — synthetic_scope: 01M2NQ5STXHRZN13TH4SH3D8YN project:_chat_inject_live_e2e_846228_1789582817 — “The task explicitly instructs me to walk the chain using the Read tool, not shel”
- `32bcfa39dddb` 3 ev (legacy caught 0), project:_chat_inject_live_e2e_2733231_1789585718 — synthetic_scope: 01M2NSY9FTBHM3DRKRWWFN0CJE project:_chat_inject_live_e2e_2733231_1789585718 — “The task explicitly wants me to use the Read tool for this chain, overriding the”
- `45e8e66bbbaf` 3 ev (legacy caught 0), project:_chat_inject_live_e2e_296497_1789587927 — synthetic_scope: 01M2NW1PFV7NX7BA6Q3AHDCZ6E project:_chat_inject_live_e2e_296497_1789587927 — “The task instructions specifically call for using the Read tool to walk the chai”
- `48a1301d2055` 3 ev (legacy caught 0), project:_chat_inject_live_e2e_342770_1789588150 — synthetic_scope: 01M2NW8GV6FN2WNMFQCEHW4HGY project:_chat_inject_live_e2e_342770_1789588150 — “between the general bypass-permissions guidance to use Bash for reading files an”
- … 408 more

**2026-09-17** — 529 sessions

- `15720144db2f` 4 ev (legacy caught 3), project:mm,project:repo,project:target — fixture_provenance: 01M2Q428H57KM81WJQC18NG2DJ context.run=evaluation_recovery--claude_stream-json--live--r1--live-a — “ev-recovery repair transaction replace failed child post with repost shipment jo”
- `0f270314f66d` 3 ev (legacy caught 2), project:mm,project:repo — fixture_provenance: 01M2RJDNXY7PJ1C5ZS9T2K3TVN context.fixture; receipt: tree-context-baseline-evidence/bundles/evaluation-live/runs/evaluation_recovery--claude_stream-json--live--r2--live-b/capture-memory-obh5h3vh — “ev-recovery shipment repost decision.json acceptance.json provenance gate”
- `3075acd35600` 3 ev (legacy caught 2), global,project:ae,project:target — fixture_provenance: 01M2Q462W0015FCM56MN07KHYJ content:AE fixture — “ev-timetable planner scan wash seal schedule window gap operator requests”
- `4651fcf535b9` 3 ev (legacy caught 1), global,project:mm,project:repo — fixture_provenance: 01M2Q1MVTE0KQBKSFGAPGY5BGW content:AE fixture — “ev-recovery repair replace failed child post with repost shipment journal S-81”
- `52f07f053417` 3 ev (legacy caught 0), global — fixture_provenance: 01M2Q4M7X7SV35SYSARCBXSZ5Q context.fixture — “development-battery assembly repair pair-v2 battery pairing quarantine”
- `55b731986c0a` 3 ev (legacy caught 0), global — fixture_provenance: 01M2RJD83N491F4QTNMAX31SE4 context.fixture; receipt: tree-context-baseline-evidence/bundles/evaluation-live/runs/evaluation_timetable--claude_stream-json--live--r1--live-a/raw/a2/capture-memory-aatjlgf3 — “ev-timetable planner schedule scan wash seal window gap operator requests”
- `73c5fd76c9a0` 3 ev (legacy caught 2), global,project:repo,project:x — fixture_provenance: 01M2PZP078EPQH9RG8QEE0GDGD content:AE fixture — “ev-recovery repost shipment posting journal S-81 replay”
- `8570dfd9b9ec` 3 ev (legacy caught 1), project:mm,project:target — fixture_provenance: 01M2QX72JC4HV03BXQ7QRBFQ5T context.run=development_collection-transfer--claude_stream-json--live--r — “collection transfer dispatch outputs/transfer.json artifact provenance immutable”
- `9dfe6c2b4cec` 3 ev (legacy caught 2), global,project:ae,project:target — fixture_scope: 01M2Q465AAXVVRPSG62X128JM0 project:target within 30 min of proven harness traffic — “ev-timetable planner schedule scan wash seal window gap operator requests”
- `aae22a7ffd2a` 3 ev (legacy caught 0), global — fixture_provenance: 01M2PZJW9ZV41SQ14CKV2ZEYNZ context.run=development_reservoir-allocation--claude_stream-json--live-- — “reservoir allocation outputs/allocation.json emergency water dispatch stations”
- `f41268e85f47` 3 ev (legacy caught 2), global,project:repo,project:x — fixture_provenance: 01M2Q11GNDM242W173H5DSVN7A context.run=evaluation_labels--claude_stream-json--live--r3--live-c — “inventory label migration L- three digits normalize inputs/task.json out/answer.”
- `fbbd28da8416` 3 ev (legacy caught 1), global,project:mm,project:x — fixture_provenance: 01M2RHF1NG571EWKTRBK2AVKFG context.fixture; receipt: tree-context-baseline-evidence/bundles/evaluation-live/runs/evaluation_calibration--claude_stream-json--live--r3--live-c/capture-memory-bgdcw37s — “ev-calibration reuse retired quadratic calibration out/decision.json artifact_pr”
- `00db876d4b24` 2 ev (legacy caught 1), global,project:target — fixture_provenance: 01M2PZEQ04BVZTFBNAVWRVP3P3 context.run=evaluation_recovery--claude_stream-json--live--r3--live-c — “ev-recovery repair shipment posting journal S-81 repost child”
- `04011c0f6cab` 2 ev (legacy caught 0), global — receipt: tree-context-baseline-evidence/bundles/development-live/runs/development_battery-repair--claude_stream-json--live--r3--live-c/capture-memory-6lq8oc0k — “battery pairing repair assembly/pair replacement pair-v2 quarantine partial-pair”
- `0810d802223f` 2 ev (legacy caught 1), global,project:repo — fixture_query_family: 2 queries name development-water — “reservoir allocation development-water outputs/allocation.json stations calibrat”
- … 514 more

**2026-09-19** — 220 sessions

- `7b1b49064887` 6 ev (legacy caught 5), project:target,session:development-packaging-routing-20260920 — receipt: tree-context-baseline-evidence/generation-20260919T215122Z/bundles/development-live/runs/development_kit-acceptance--codex_exec--live--r3--repaired-baseline-3/raw/a1/capture-memory-mrftgm62 — “development-packaging routing accepted kits only constraints source identities f”
- `7d3346644b43` 3 ev (legacy caught 2), project:mm,project:target — fixture_provenance: 01M2XXCW3S06AJR6KNDR3H040W context.run=development_collection-transfer--claude_alt_backend--live--r; receipt: tree-context-baseline-evidence/generation-20260919T215122Z/bundles/development-live/runs/development_collection-transfer--claude_alt_backend--live--r1--repaired-baseline-1/capture-memory-61iwxy34 — “collection transfer repair artifact provenance transfer.json transport-policy im”
- `fa042a4fc9fb` 3 ev (legacy caught 2), global,project:repo — fixture_provenance: 01M2XGZY3W6DJKTJW7CMPY7P22 content:AE fixture; receipt: tree-context-baseline-evidence/generation-20260919T184821Z/bundles/evaluation-live/runs/evaluation_recovery--claude_stream-json--live--r3--repaired-baseline-3/capture-memory-5vdh_5ve — “ev-recovery repair failed child post repost shipment journal S-81 augment transa”
- `1132a8402d6f` 2 ev (legacy caught 1), global,project:target — fixture_provenance: 01M2XWJPDCVW9WGCANTQETH2WV context.project=target; receipt: tree-context-baseline-evidence/generation-20260919T215122Z/bundles/development-live/runs/development_kit-acceptance--claude_alt_backend--live--r1--repaired-baseline-1/capture-memory-j69x2ai1 — “kits packaging acceptance candidates/packing.json tube_shortage coolant_mass acc”
- `4599347db555` 2 ev (legacy caught 1), global,project:target — receipt: tree-context-baseline-evidence/generation-20260919T215122Z/bundles/evaluation-live/runs/evaluation_recovery--claude_alt_backend--live--r2--repaired-baseline-2/raw/a3/capture-memory-0z_257uy — “shipment posting idempotent repair repost S-81 journal posted receipts”
- `47027ad530e4` 2 ev (legacy caught 1), global,project:target — fixture_provenance: 01M2XGP0RHTJ501PJNARADW79C context.fixture; receipt: tree-context-baseline-evidence/generation-20260919T184821Z/bundles/evaluation-live/runs/evaluation_labels--claude_alt_backend--live--r2--repaired-baseline-2/capture-memory-nep0msft — “normalize task.json labels L-001 trim lowercase sum duplicates sort answer.json ”
- `681381da8ef4` 2 ev (legacy caught 0), global,project:mm — receipt: tree-context-baseline-evidence/generation-20260919T184821Z/bundles/development-live/runs/development_battery-repair--claude_stream-json--live--r1--repaired-baseline-1/raw/a1/capture-memory-ydhz241i — “battery pairing repair assembly pair-v2 quarantine augment transaction”
- `6d1b18f1e294` 2 ev (legacy caught 0), global — receipt: tree-context-baseline-evidence/generation-20260919T184821Z/bundles/evaluation-live/runs/evaluation_timetable--claude_stream-json--live--r2--repaired-baseline-2/raw/a2/capture-memory-9d_tu3ut — “planner timetable scan wash seal window gap operator requests”
- `b98e943cfc84` 2 ev (legacy caught 0), global — fixture_provenance: 01M2XGHKX04CV7V0BQQP9W7K3N context.fixture; receipt: tree-context-baseline-evidence/generation-20260919T184821Z/bundles/evaluation-live/runs/evaluation_labels--claude_stream-json--live--r2--repaired-baseline-2/raw/a2/capture-memory-1wspvpx0 — “inventory label migration L- three digits normalize task.json answer.json decisi”
- `d053f732deed` 2 ev (legacy caught 1), global,project:target — fixture_provenance: 01M2XX49GB1NPW4671SJMH8MX8 context.run=development_collection-transfer--claude_alt_backend--live--r; receipt: tree-context-baseline-evidence/generation-20260919T215122Z/bundles/development-live/runs/development_collection-transfer--claude_alt_backend--live--r2--repaired-baseline-2/capture-memory-jluiqxye — “collection transfer sealing progress.json durable item identities transport-poli”
- `dee139a898f8` 2 ev (legacy caught 1), project:mm,project:target — receipt: tree-context-baseline-evidence/generation-20260919T215122Z/bundles/evaluation-live/runs/evaluation_recovery--claude_alt_backend--live--r2--repaired-baseline-2/raw/a3/capture-memory-0z_257uy — “ev-recovery shipment posting repair augment repost decision.json replaced_identi”
- `e6a7df20b99c` 2 ev (legacy caught 1), global,project:target — receipt: tree-context-baseline-evidence/generation-20260919T184821Z/bundles/development-live/runs/development_kit-acceptance--claude_stream-json--live--r1--repaired-baseline-1/capture-memory-cdyt0k1z — “development-packaging routing accepted kits only D1 constraints source identitie”
- `153061d3a6ce` 1 ev (legacy caught 0), global — fixture_provenance: 01M2XH234B2Q2ZQ50G8CE8CWT0 context.fixture; receipt: tree-context-baseline-evidence/generation-20260919T184821Z/bundles/development-live/runs/development_kit-acceptance--claude_alt_backend--live--r2--repaired-baseline-2/capture-memory-tvg5zo42 — “development-packaging routing accepted kits constraints source identities file o”
- `1fd601999b5d` 1 ev (legacy caught 0), global — receipt: tree-context-baseline-evidence/generation-20260919T184821Z/bundles/development-live/runs/development_collection-transfer--claude_stream-json--live--r2--repaired-baseline-2/raw/a1/capture-memory-vijciua0 — “collection transfer dispatch progress.json sealed_items durable identities trans”
- `313cdd40d773` 1 ev (legacy caught 0), global — fixture_provenance: 01M2XGHXHEW1GRYHBMBYBCE40A content:AE fixture; receipt: tree-context-baseline-evidence/generation-20260919T184821Z/bundles/evaluation-live/runs/evaluation_labels--claude_stream-json--live--r3--repaired-baseline-3/raw/a2/capture-memory-p0pbp2r7 — “label migration L- three digits normalize inputs/task.json out/answer.json decis”
- … 205 more

**2026-09-20** — 130 sessions

- `042cf23a998a` 4 ev (legacy caught 2), global,project:target — receipt: tree-context-baseline-evidence/generation-20260920T130402Z/bundles/evaluation-live/runs/evaluation_timetable--claude_stream-json--live--r3--repaired-baseline-3/raw/a2/capture-memory-7pw12_g8 — “planner timetable scan wash seal schedule operator requests window gap”
- `b9584875557f` 3 ev (legacy caught 2), global,project:repo — fixture_provenance: 01M2ZHDR6ECV54MH5EBC963WJB context.run=evaluation_recovery--claude_stream-json--live--r2--repaired-; receipt: tree-context-baseline-evidence/generation-20260920T130402Z/bundles/evaluation-live/runs/evaluation_recovery--claude_stream-json--live--r2--repaired-baseline-2/capture-memory-k026wgi8 — “ev-recovery repair interrupted shipment posting repost journal S-81”
- `06bf39a40730` 2 ev (legacy caught 1), global,project:repo — fixture_provenance: 01M2ZHCZ9ZPZF45C57147V7YM7 content:AE fixture; receipt: tree-context-baseline-evidence/generation-20260920T130402Z/bundles/evaluation-live/runs/evaluation_recovery--claude_stream-json--live--r1--repaired-baseline-1/raw/a3/capture-memory-e9y5iwl9 — “ev-recovery repair shipment posting journal S-81 repost augment transaction”
- `1823ed959178` 2 ev (legacy caught 0), global — fixture_provenance: 01M2ZHZHX8EHAR59XGXQ8VX4PT context.project=target; receipt: tree-context-baseline-evidence/generation-20260920T130402Z/bundles/development-live/runs/development_collection-transfer--claude_stream-json--live--r2--repaired-baseline-2/raw/a1/capture-memory-fq0o5zx8 — “collection transfer dispatch progress.json sealed_items transfer.json packing”
- `210ab1708103` 2 ev (legacy caught 0), global,project:mm — fixture_provenance: 01M2ZJ3BC1F9G8N2GZMXMCB19V context.fixture; receipt: tree-context-baseline-evidence/generation-20260920T130402Z/bundles/development-live/runs/development_collection-transfer--claude_stream-json--live--r2--repaired-baseline-2/capture-memory-fq0o5zx8 — “collection transfer dispatch outputs/transfer.json packing deferred east closed ”
- `416b8e8d54a6` 2 ev (legacy caught 1), global,project:x — fixture_provenance: 01M2ZJA6384FKQQEST12ZK70ZJ context.run=evaluation_timetable--claude_stream-json--live--r1--repaired; receipt: tree-context-baseline-evidence/generation-20260920T130402Z/bundles/evaluation-live/runs/evaluation_timetable--claude_stream-json--live--r1--repaired-baseline-1/capture-memory-4wkspr30 — “planner timetable scan wash seal operator requests window gap decision.json”
- `542c9cd1f22f` 2 ev (legacy caught 1), project:mm,project:repo — fixture_provenance: 01M2ZHFRRHVYS2GK2NEXQX8R6K context.fixture; receipt: tree-context-baseline-evidence/generation-20260920T130402Z/bundles/evaluation-live/runs/evaluation_recovery--claude_stream-json--live--r3--repaired-baseline-3/capture-memory-p7stwwo3 — “ev-recovery repost shipment posting journal replay S-81 answer.json decision.jso”
- `741d3e0ed873` 2 ev (legacy caught 0), global — fixture_provenance: 01M2ZHYE9ZH2FQAXPM3YRRGY3G context.run=development_collection-transfer--claude_stream-json--live--r; receipt: tree-context-baseline-evidence/generation-20260920T130402Z/bundles/development-live/runs/development_collection-transfer--claude_stream-json--live--r1--repaired-baseline-1/raw/a1/capture-memory-40s55jvj — “collection transfer dispatch outputs/transfer.json progress sealed_items policy_”
- `7f298506017c` 2 ev (legacy caught 1), global,project:target — receipt: tree-context-baseline-evidence/generation-20260920T130402Z/bundles/development-live/runs/development_kit-acceptance--claude_stream-json--live--r1--repaired-baseline-1/capture-memory-dnl3issw — “development-packaging routing accepted kits route only”
- `823c17e76927` 2 ev (legacy caught 0), global — receipt: tree-context-baseline-evidence/generation-20260920T130402Z/bundles/development-live/runs/development_battery-repair--claude_stream-json--live--r3--repaired-baseline-3/raw/a1/capture-memory-widkxl3g — “battery pairing repair assembly pair-v2 quarantine duplicated cell B2”
- `851a7582679a` 2 ev (legacy caught 0), global,project:mm — fixture_provenance: 01M2ZGXXD3H7X422YC4JERCMZJ context.fixture; receipt: tree-context-baseline-evidence/generation-20260920T130402Z/bundles/evaluation-live/runs/evaluation_calibration--claude_stream-json--live--r3--repaired-baseline-3/capture-memory-2_61pb5q — “ev-calibration reuse retired quadratic calibration out/answer.json out/decision.”
- `8eff0539e508` 2 ev (legacy caught 0), global,project:mm — receipt: tree-context-baseline-evidence/generation-20260920T130402Z/bundles/development-live/runs/development_battery-repair--claude_stream-json--live--r1--repaired-baseline-1/raw/a1/capture-memory-2h229unf — “development-battery assembly repair pair-v2 quarantine augment transaction”
- `8fc277be0df0` 2 ev (legacy caught 0), global — receipt: tree-context-baseline-evidence/generation-20260920T130402Z/bundles/development-live/runs/development_battery-repair--claude_stream-json--live--r1--repaired-baseline-1/capture-memory-2h229unf — “battery pairing repair assembly pair-v2 quarantine augment transaction”
- `a744459e0bbe` 2 ev (legacy caught 0), global,project:mm — fixture_provenance: 01M2ZJ4B08G7H4P55NKYS60JYQ context.fixture; receipt: tree-context-baseline-evidence/generation-20260920T130402Z/bundles/development-live/runs/development_collection-transfer--claude_stream-json--live--r1--repaired-baseline-1/capture-memory-40s55jvj — “collection transfer dispatch transfer.json packing deferred east closed capacity”
- `cca91c52e6e8` 2 ev (legacy caught 0), global,project:mm — fixture_provenance: 01M2ZHRH84E9SX9EFQTSY8T5CK context.fixture; receipt: tree-context-baseline-evidence/generation-20260920T130402Z/bundles/evaluation-live/runs/evaluation_recovery--claude_stream-json--live--r3--repaired-baseline-3/capture-memory-p7stwwo3 — “ev-recovery interrupted shipment posting replay journal S-81 repair decision.jso”
- … 115 more

**2026-09-22** — 3 sessions

- `055a4d8e5368` 1 ev (legacy caught 0), global — fixture_query_family: 1 queries name development-battery — “development-battery assembly repair pair-v2 augment transaction battery pairing”
- `5b69b41ab402` 1 ev (legacy caught 0), global — fixture_query_family: 1 queries name development-battery — “development-battery assembly repair pair-v2 quarantine pairing transaction”
- `b1a74d932364` 1 ev (legacy caught 0), global — fixture_query_family: 1 queries name development-battery — “development-battery assembly repair pair-v2 quarantine pairing transaction”

**2026-09-23** — 480 sessions

- `1952488de1c4` 3 ev (legacy caught 1), global,project:x — fixture_provenance: 01M37PNZ8M65Z8D1JT37JT0CC2 context.fixture; receipt: tree-context-ab/live-20260923T174224Z/baseline/runs/evaluation_timetable--claude_stream-json--live--r1--r1/capture-memory-nbdz85kn — “planner timetable scan wash seal operator requests window gap”
- `53e056b58845` 3 ev (legacy caught 0), global — fixture_provenance: 01M37PTGWZ2KH8PYNXX5CGVDVS context.run=tree-context-ab-live-20260923T174224Z--candidate--r1--r1; receipt: tree-context-ab/live-20260923T174224Z/candidate/runs/development_battery-repair--claude_stream-json--live--r1--r1/raw/a1/capture-memory-3egxewk1 — “development-battery assembly pair repair pairing subtree pair-v2”
- `55a38ab62eed` 3 ev (legacy caught 2), global,project:repo — fixture_provenance: 01M37F30R3KR228G6A3GJ9BS83 context.fixture; receipt: tree-context-ab/live-20260923T142502Z/baseline/runs/evaluation_recovery--claude_stream-json--live--r3--r3/capture-memory-1407_z1g — “ev-recovery repost shipment posting journal replan decomposition”
- `657e724ae598` 3 ev (legacy caught 0), global,project:mm — receipt: tree-context-ab/live-20260923T122922Z/baseline/runs/development_battery-repair--claude_stream-json--live--r1--r1/capture-memory-n5qd1ei_ — “battery pairing repair augment assembly/pair replacement pair-v2 quarantine”
- `eb19973db575` 3 ev (legacy caught 2), global,project:repo — fixture_provenance: 01M36XTYTB0BB15CPGA1KRZB0H context.run=tree-context-ab/live-20260923T095325Z/candidate/r2--r2; receipt: tree-context-ab/live-20260923T095325Z/candidate/runs/evaluation_recovery--claude_stream-json--live--r2--r2/raw/a3/capture-memory-io5cnzrv — “ev-recovery repair failed child post repost shipment journal S-81”
- `f6fae0b66769` 3 ev (legacy caught 1), global — fixture_provenance: 01M37EK9ZYTZQN3J0P6CPD4MKJ context.fixture; receipt: tree-context-ab/live-20260923T142502Z/candidate/runs/development_kit-acceptance--claude_stream-json--live--r2--r2/capture-memory-i02xxxbs — “development-packaging routing accepted kits route only”
- `046317be4e66` 2 ev (legacy caught 0), global — fixture_provenance: 01M371APFXFJTH7XFBWAAT4D7H context.run=tree-context-ab-live-20260923T095325Z--candidate--r3--r3; receipt: tree-context-ab/live-20260923T095325Z/candidate/runs/development_battery-repair--claude_stream-json--live--r3--r3/capture-memory-wceavuwq — “development-battery assembly pair-v2 pairs.json verification”
- `0ae84976b65e` 2 ev (legacy caught 1), global,project:repo — fixture_provenance: 01M36VBHGRVP1SJ8M23X9RVMVX context.fixture; receipt: tree-context-ab/live-20260923T095325Z/candidate/runs/evaluation_recovery--claude_stream-json--live--r1--r1/capture-memory-xzyq4hl8 — “ev-recovery repost shipment posting journal replay answer.json decision.json”
- `1a80432115f4` 2 ev (legacy caught 0), global — fixture_provenance: 01M36YB8HDVVVRWJZDVGHHMTDQ context.run=tree-context-ab-live-20260923T095325Z--candidate--r2--r2; receipt: tree-context-ab/live-20260923T095325Z/candidate/runs/development_battery-repair--claude_stream-json--live--r2--r2/raw/a1/capture-memory-beqjaekf — “augment repair mode development-battery assembly pair-v2 replacement node”
- `1a98b0934bb2` 2 ev (legacy caught 0), global — fixture_provenance: 01M374EHSN8G6QGQTZB3BEVNWG context.run=evaluation_timetable--claude_stream-json--live--r1--r1; receipt: tree-context-ab/live-20260923T122922Z/candidate/runs/evaluation_timetable--claude_stream-json--live--r1--r1/raw/a2/capture-memory-af8008ux — “planner timetable scan wash seal window gap operator requests”
- `1d60a19b86d5` 2 ev (legacy caught 0), global — fixture_provenance: 01M374CGM3NC8S1KK594Y5KVY9 context.run=evaluation_timetable--claude_stream-json--live--r1--r1; receipt: tree-context-ab/live-20260923T122922Z/baseline/runs/evaluation_timetable--claude_stream-json--live--r1--r1/capture-memory-t1dgha0_ — “planner timetable schedule scan wash seal window gap operator requests”
- `20c6e9d11759` 2 ev (legacy caught 0), global — fixture_provenance: 01M376JZZWSX03P5YBFGSC2R02 context.run=tree-context-ab-live-20260923T122922Z--candidate--r2--r2; receipt: tree-context-ab/live-20260923T122922Z/candidate/runs/development_battery-repair--claude_stream-json--live--r2--r2/raw/a1/capture-memory-n2xiiepq — “augment repair mode battery pairing assembly pair-v2 replacement node”
- `23be22abc04e` 2 ev (legacy caught 1), global — fixture_provenance: 01M36WQB94SYKYVKRTBKT4HT02 context.fixture; receipt: tree-context-ab/live-20260923T095325Z/baseline/runs/development_kit-acceptance--claude_stream-json--live--r1--r1/capture-memory-iqtb9qyt — “development-packaging routing accepted kits route only”
- `2ecf2f9b80c1` 2 ev (legacy caught 1), global,project:mm — fixture_provenance: 01M36V2KZKK4FQH9B4NBMA84NK context.fixture; receipt: tree-context-ab/live-20260923T095325Z/baseline/runs/evaluation_calibration--claude_stream-json--live--r1--r1/capture-memory-q792nl3d — “ev-calibration reuse retired quadratic calibration children_ledger artifact prov”
- `31fe89d03bfb` 2 ev (legacy caught 0), global,project:mm — fixture_provenance: 01M370KEW3K37YVMXRXQNMRXSC context.fixture; receipt: tree-context-ab/live-20260923T095325Z/baseline/runs/evaluation_recovery--claude_stream-json--live--r3--r3/capture-memory-o1fl1d8m — “ev-recovery shipment posting repair decision.json answer.json”
- … 465 more

## Legacy over-exclusion sample (kept by the new filter)

- 2026-09-07 `8f3b2b4bd223` project:lm: lookup credit usage signal recall_lookup_events apply_pending_recall_feedback ledger
- 2026-09-07 `efb1ca962520` project:lm: lookup credit ledger recall_credit_ledger apply_lookup_credit memory_lookup usage signal decompositi
- 2026-09-08 `b7e95a945a8a` project:x: Sber auth2 external_credentials provider_client_id sandbox production identity isolation state secur
- 2026-09-08 `e7279761eeb9` project:repo: EZ-13929 code review merge request Jira repo review procedure
- 2026-09-09 `0eab15c6c0de` project:ae: OpenCode MCP loading-probe gate failure secrets diff size fixtures compact evidence native instructi
- 2026-09-09 `06bf35a12f3c` project:ae: GigaCode deterministic HTTP tests owned sockets TLS fixture CONNECT lifecycle leak source evidence b
- 2026-09-10 `9b9ce44d54f1` project:x: gigacode facade public cli status catalog exit codes options bounds
- 2026-09-10 `e9bca07b175c` project:x: facade evidence.cjs recorder pattern source inventory hashing frozenExecutionInput false pinned obse
- 2026-09-11 `1859096a3b29` project:ae: ae tree node context loss between parent and child: verification focus, children ledger summary trun
- 2026-09-11 `66a4e846c69b` project:x: tools/gigacode facade createOpenCodeAcquisition status catalog exports
- 2026-09-12 `8fb4bf7c30a6` project:x: augment transaction planner repair moves prerequisite sibling widen contract selftest fixture
- 2026-09-12 `978df445decb` project:mm: augment transaction selftest goal root selftest fixture expected output grammar SUBGOALS AUGMENT_OPS
- 2026-09-13 `6c2004f04080` project:ae: test_node_restart_drain.sh scheduler fixture make_library sourceable fixture model no provider
- 2026-09-13 `0183f6d93d0d` project:ae: tests/test_dashboard_restart_ownership.js test_node_restart_ownership.sh fixture flock helper pitfal
- 2026-09-14 `b81c15c45ac8` project:ae: chat-live scenario-cleanup-verdict context.close cleanupAcceptable foreign_sources proc-scan dashboa
- 2026-09-14 `3b97f007ff26` project:x: artifact_evidence.json "pre-execution" "replan" missing required_artifacts refresh baseline
- 2026-09-15 `04e2247c9284` project:ae: ae prompt bloat archived done children ledger unbounded compact continuation repeated schema
- 2026-09-15 `04e2247c9284` project:x: node_read_context sibling context empty full node_build_prompt NODE_ID environment exports
- 2026-09-16 `0968372a729a` project:ae: tree-context evidence validator commit worktree pause checkpoint
- 2026-09-16 `ca1bc83b76cd` project:ae: tree-context development-cases merge gate failure secrets operator_request diff size
- 2026-09-17 `d1b0e9fdbd31` project:x: tree_context_bench validation evidence _providers.py alias provider midturn_injection validate_matri
- 2026-09-17 `d1b0e9fdbd31` project:ae: tree-context-bench evidence validation alias provider claude_alt backend injectable
- 2026-09-18 `e41c3126e663` project:x: tree-context-system-redesign rebaseline freeze conflict resolution
- 2026-09-18 `698d79cd328f` project:ae: tree_context_bench rebaseline freeze blocking conflicts sidecar harness rewrite selftest foreign che
- 2026-09-19 `6c447a6f35c9` project:x: replan refused lifecycle_owned done child attempt phase exiting never exited; relisting carried-over
- 2026-09-19 `92d72d75b7d7` project:ae: ae commit message convention node.sh selftest fixture commit style
- 2026-09-20 `d039e846aa71` project:ae: "authenticity-repair" isolated inventory fixture SOURCE_PATHS
- 2026-09-20 `a2a035000f1a` project:ae: tree-context-system-redesign context-evidence repair-large-zstd-authentication 86b785724 transfer-bl
- 2026-09-21 `f95dda22cfc3` project:x: observer publication source drift core exercises state session bridge
- 2026-09-21 `f95dda22cfc3` project:repo: observer-capture-record-growth compatible-evidence publication dependencies candidate report indepen
- 2026-09-22 `aa4fd51a4c4d` project:x: Prior context delivery is uncertain automatic replay is blocked reconcile consumption evidence chat_
- 2026-09-22 `aa4fd51a4c4d` project:x: opencode Code Mode OPENCODE_EXPERIMENTAL_CODE_MODE harness fit plugin tools context7 binary strings 
- 2026-09-23 `3d9108dfc9c3` project:x: tree-context-system-redesign augment operator context-evidence closed as history live A/B comparison
- 2026-09-23 `3d9108dfc9c3` project:x: augment tree-context-system-redesign live A/B decomposition bench proxy recorder codex {} claude fai
- 2026-09-24 `5155fe9b1538` project:x: module_colocation classifier false dependency executable imports shell python heredoc
- 2026-09-24 `2fc68fc155f1` project:x: executable-import-evidence module_colocation docstring supervisor September 24
- 2026-09-25 `40f001539544` project:x: augment prompt render_current_tree complete sources parent entry lines budget TOTAL_MAX
- 2026-09-25 `f55be322403b` project:x: augment-tree-sources-by-rule render_sources Live Tree prompt size TOTAL_MAX node.sh sources lines pe
- 2026-09-26 `54f2a205c464` project:x: claude quota tier check recipe: how to read remaining quota / floor before creating goal; bounded-ex
- 2026-09-26 `d8707f4ba105` project:x: lifecycle_owned apply_stage written set path_spec.md node_write_child_context_files root_goal.md exe
- 2026-09-27 `f9fb9ecc981d` project:x: supervisor direction project-state-journal restartable step durable accepted operation no new layers
- 2026-09-27 `b6b77a0a3ee3` project:x: repair-resumes-existing-tree b7357067e supervisor 469 grant 390 object-successors biome planner
