# Query-anchor yield and reachability ceiling

Read-only measurement over snapshot `/tmp/anchor-est/snap.sqlite3` (sha256 `5f82312ec3566f99…`, 16717 nodes, 54814 recall events), embedding backend `sentence-transformers`, grounding threshold 0.25.

## 1. What retro grounded labeling yields

- consumed events: **9108** (2026-05-15T05:04:39Z .. 2026-08-18T12:01:36Z)
- gradeable (consuming trace still present): **9073**; unresolvable: 35
- grounded events (>= 1 grounded result): **3682** = **40.6%** of gradeable
- anchors after exact-fingerprint dedup: **3450**
- edges: **8335**, **2.415942** per anchor
- edges into already-decayed targets: **476** = **5.7%**
- anchors reinforced by more than one event: 57 (1.067246 events per anchor)

### Against the goal's prior estimate (600-event sample)

| figure | prior estimate | measured (full history) | delta |
| --- | --- | --- | --- |
| grounded_share | 0.413 | 0.405819 | -0.007181 |
| anchors | 3760 | 3450 | -310 |
| edges | 8550 | 8335 | -215 |
| edges_per_anchor | 2.28 | 2.415942 | 0.135942 |
| decayed_edge_share | 0.046 | 0.057109 | 0.011109 |

### Per scope (top scopes by consumed events)

| scope | consumed | grounded | grounded share | anchors | edges | per anchor | decayed edges |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `project:online` | 2993 | 820 | 27.4% | 678 | 1208 | 1.781711 | 15.1% |
| `project:x` | 2007 | 1375 | 68.5% | 1354 | 4069 | 3.00517 | 3.1% |
| `project:ae` | 1585 | 432 | 27.3% | 413 | 716 | 1.733656 | 5.2% |
| `project:octopus` | 1551 | 728 | 46.9% | 687 | 1641 | 2.388646 | 3.6% |
| `project:lm` | 453 | 165 | 36.4% | 157 | 285 | 1.815287 | 12.3% |
| `global` | 239 | 88 | 36.8% | 87 | 259 | 2.977011 | 8.5% |
| `project:mm` | 67 | 17 | 25.4% | 17 | 34 | 2.0 | 8.8% |
| `project:gas-stations-ui` | 46 | 8 | 17.4% | 8 | 11 | 1.375 | 18.2% |
| `project:living-memory` | 18 | 3 | 18.8% | 3 | 6 | 2.0 | 0.0% |
| `project:id-2gis-com` | 15 | 1 | 6.7% | 1 | 3 | 3.0 | 0.0% |
| `project:/root/p/octopus` | 12 | 2 | 100.0% | 2 | 6 | 3.0 | 0.0% |
| `project:natural-flow/downstream-route-propagation/_critique/review-packet-route-triple` | 12 | 3 | 25.0% | 3 | 4 | 1.333333 | 0.0% |

### Split at candidate cutoffs (before = anchor training window)

| cutoff | before: events | anchors | edges | per anchor | cyrillic | span days | after: events | anchors | cyrillic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-10 | 5385 | 1685 | 3523 | 2.090801 | 178 (3.3%) | 25.78 | 3723 | 1767 | 606 (16.3%) |
| 2026-07-01 | 7387 | 2938 | 7296 | 2.483322 | 216 (2.9%) | 46.69 | 1721 | 514 | 568 (33.0%) |
| 2026-07-15 | 7908 | 3178 | 7879 | 2.479232 | 391 (4.9%) | 60.46 | 1200 | 273 | 393 (32.8%) |
| 2026-08-01 | 8225 | 3271 | 8051 | 2.461327 | 553 (6.7%) | 77.63 | 883 | 180 | 231 (26.2%) |
| 2026-08-10 | 8545 | 3331 | 8159 | 2.449415 | 687 (8.0%) | 86.73 | 563 | 120 | 97 (17.2%) |

## 2. Near-duplicate anchor similarity (fixes the dedup threshold)

3450 anchors across 37 scopes, 1482384 same-scope pairs (401259 sampled for percentiles).

- pair similarity: {"min": -0.202, "p50": 0.3187, "p90": 0.4888, "p99": 0.6846, "p99.9": 0.8925, "max": 0.9913, "mean": 0.3234}
- nearest neighbour per anchor: {"min": 0.2339, "p50": 0.7743, "p90": 0.918, "p99": 0.9856, "p99.9": 0.9946, "max": 0.9963, "mean": 0.7679}

| threshold | pairs >= t | with differing identifiers | collision share | clusters after transitive merge | anchors absorbed |
| --- | --- | --- | --- | --- | --- |
| 0.80 | 2803 | 1108 | 39.5% | 2405 | 1045 (30.3%) |
| 0.85 | 1412 | 458 | 32.4% | 2916 | 534 (15.5%) |
| 0.90 | 664 | 241 | 36.3% | 3151 | 299 (8.7%) |
| 0.92 | 450 | 184 | 40.9% | 3231 | 219 (6.3%) |
| 0.95 | 230 | 127 | 55.2% | 3345 | 105 (3.0%) |
| 0.97 | 94 | 56 | 59.6% | 3395 | 55 (1.6%) |
| 0.99 | 7 | 1 | 14.3% | 3443 | 7 (0.2%) |

**Band 0.99-1.01**

- 0.9963 [same ids] `packages/desktop/src/entities/entity/ui/catalogTooltip.tsx packages/desktop/src/entities/tooltip/entity/ui/entityTooltipExperimentView.tsx edit conventions EZ-1` ↔ `packages/desktop/src/entities/tooltip/entity/ui/entityTooltipExperimentView.tsx packages/desktop/src/entities/entity/ui/catalogTooltip.tsx edit conventions EZ-1`
- 0.9935 [same ids] `EZ-13771 tooltip rating stars value fill experiment` ↔ `EZ-13771 tooltip experiment rating value stars fill`
- 0.992 [same ids] `Close the NW-10 mechanism CROSS at source level: prove-or-make gradient_neural_memory_v1 with inner_update_steps>=2 trainable UNDER multistep credit window>=2 —` ↔ `clarification: Close the NW-10 mechanism CROSS at source level: prove-or-make gradient_neural_memory_v1 with inner_update_steps>=2 trainable UNDER multistep cre`
- 0.9915 [same ids] `data_training_returns return_curve post-v2 rungs cognitive_battery validator fail-closed epsilon diagnostic non-credit` ↔ `data_training_returns return_curve post-v2 rungs cognitive_battery validator fail closed epsilon diagnostic non-credit`
- 0.9913 [same ids] `EZ-13771 tooltip experiment requirements matrix horizontal photo Confluence Figma BSS` ↔ `EZ-13771 tooltip experiment requirements matrix horizontal photo BSS Figma Confluence`
- 0.9904 [DIFFERENT ids] `clarification: Изучи https://jira.2gis.ru/browse/EZ-13512 и связаннные ресурсы. Проанализируй соответсвующий мердж-реквест (код репозитория` ↔ `clarification: Изучи https://jira.2gis.ru/browse/EZ-13560 и связаннные ресурсы. Проанализируй соответсвующий мердж-реквест (код репозитория`

**Band 0.97-0.99**

- 0.979 [DIFFERENT ids] `task_pattern:a7c7530c40cc59c4` ↔ `task_pattern:5b492dddc439419d`
- 0.9734 [DIFFERENT ids] `task_pattern:a94ab916b4894a87` ↔ `task_pattern:15f30f014b37762b`
- 0.9719 [DIFFERENT ids] `task_pattern:2a52b7ddc5baad6e` ↔ `task_pattern:a7f7372b228ef272`
- 0.9716 [DIFFERENT ids] `task_pattern:a7c7530c40cc59c4` ↔ `task_pattern:7bdd9349a6d3fc15`
- 0.9703 [same ids] `clarification: Measure repaired production-selected ABI v2 checkpoint/resume and write tracked selected evidence.` ↔ `Measure repaired production-selected ABI v2 checkpoint/resume and write tracked selected evidence.`
- 0.9702 [same ids] `Cherry-pick reachable commit `284f1cd60067932e0e8338f16253d298d9f91819`, or apply an equivalent batched EFE planner implementation, into this live verify branch` ↔ `Cherry-pick reachable commit 284f1cd60067932e0e8338f16253d298d9f91819, or apply the equivalent batched EFE planner implementation, into this _verify branch incl`

**Band 0.95-0.97**

- 0.9672 [same ids] `a08 runtime checkpoint smoke bounded manifest non-holdout trainer proof runtime_checkpoint_test make smoke failure holdout scoring artifacts optimizer steps sou` ↔ `a08 runtime checkpoint smoke bounded manifest non-holdout trainer runtime_checkpoint_test make smoke failure source manifest optimizer step expectations`
- 0.9613 [DIFFERENT ids] `task_pattern:5b492dddc439419d` ↔ `task_pattern:be4cd034f0360800`
- 0.9608 [same ids] `dashboard basic auth password change UI DASHBOARD_PASSWORD ae.env Authentication section` ↔ `dashboard basic auth password change DASHBOARD_PASSWORD ae.env authentication section`
- 0.9596 [DIFFERENT ids] `task_pattern:be4cd034f0360800` ↔ `task_pattern:a7f7372b228ef272`
- 0.9571 [DIFFERENT ids] `task_pattern:ee4db943bf9941f6` ↔ `task_pattern:be4cd034f0360800`
- 0.9519 [same ids] `rebuild postopt source snapshot verification DAG-owned preopt baseline efficiency gate canonical training run` ↔ `rebuild postopt source snapshot DAG-owned preopt baseline efficiency gate verification canonical training run artifacts capability`

**Band 0.90-0.95**

- 0.9496 [same ids] `bilingual evaluation harness Octopus artifacts bilingual_en_ru_status.md malformed acceptance command previous attempt` ↔ `bilingual evaluation harness artifacts/bilingual_en_ru_status.md previous attempt acceptance syntax error Octopus`
- 0.9403 [same ids] `strategy watchdog stagnation detection tree goals` ↔ `strategy watchdog stagnation detection per-tree`
- 0.9293 [DIFFERENT ids] `EZ-13870-dashboard-redesign reopen_lesson dashboard header redesign searchbar rubrics weather design mismatch` ↔ `reopen_lesson EZ-13870 dashboard redesign header searchbar rubrics weather design fidelity`
- 0.9195 [same ids] `octopus goals dashboard api active goals stuck failed supervisor cron root cause ae works on` ↔ `Octopus current goals dashboard API stuck failed root cause supervisor cron ae octopus active goals`
- 0.9075 [DIFFERENT ids] `task_pattern:ee4db943bf9941f6` ↔ `level:schema task_pattern:64b0ff4518c9e958`
- 0.9073 [same ids] `Octopus capability audit roadmap bilingual Russian English tool use programming dialogue self-modification prior audit` ↔ `Octopus capability audit roadmap bilingual tool-use programming dialogue self-modification`

**Band 0.85-0.90**

- 0.8893 [same ids] `octopus dashboard goals active stuck failed operator request cron supervisor ae development context` ↔ `octopus dashboard goals active stuck failed operator request pause ae supervisor cron root cause`
- 0.8835 [same ids] `Octopus blinded review packet restricted key reviewer-facing packets provenance candidate identity blinding validate_review_blinding build_blinded_review_packet` ↔ `validate-blinded-public-restricted-split decomposition critique reviewer packets restricted identity key Octopus parity review packets blinding row join validat`
- 0.8806 [same ids] `octopus active goals dashboard API stuck failed paused operator request root cause supervisor` ↔ `octopus goals dashboard stuck failed paused operator request cron supervisor ae active goals root cause`
- 0.8654 [DIFFERENT ids] `jira_ticket_review_tree_goal procedure: how to review an EZ ticket review tree goal` ↔ `jira ticket review tree goal procedure: how to review a 2gis Jira ticket (MR diff, verification, comment)`
- 0.8571 [DIFFERENT ids] `data_scaling_v2 raw predictions v2_battery_manifest digest-bound records cognitive battery trained checkpoint validation protocol` ↔ `data_scaling_v2 fresh-scoreblind successor replacement battery raw records manifest checkpoint freeze shortcut fields cognitive battery validation`
- 0.8512 [same ids] `octopus dashboard api goals operator requests tree goal stuck failed pause unpause create goal AE keepup supervisor` ↔ `AE working on Octopus goals dashboard API goal orchestration stuck failed paused root cause supervisor`

## 3. Reachability ceiling

### Cutoff 2026-06-10T00:00:00Z

**grounded** — anchors 1685, edges 3523

| stratum | items | after cutoff | with eligible anchor | @1 | @3 | @10 | oracle (any anchor) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| content_grounded | 160 | 160 | 160 | 0.044 | 0.081 | 0.119 | 0.150 |
| cross_lingual | 38 | 38 | 38 | 0.000 | 0.000 | 0.026 | 0.184 |
| role_query | 36 | 36 | 36 | 0.028 | 0.083 | 0.083 | 0.111 |
| overall | 234 | 234 | 234 | 0.034 | 0.068 | 0.098 | 0.150 |

**unfiltered** — anchors 4580, edges 37518

| stratum | items | after cutoff | with eligible anchor | @1 | @3 | @10 | oracle (any anchor) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| content_grounded | 160 | 160 | 160 | 0.031 | 0.056 | 0.113 | 0.181 |
| cross_lingual | 38 | 38 | 38 | 0.000 | 0.079 | 0.158 | 0.395 |
| role_query | 36 | 36 | 36 | 0.083 | 0.111 | 0.139 | 0.250 |
| overall | 234 | 234 | 234 | 0.034 | 0.068 | 0.124 | 0.226 |

**Headroom over the anchor-free baseline run (grounded arm)**

`structural_ceiling` assumes every anchor-reachable miss converts to a hit and nothing regresses; `matched_ceiling` uses the measured top-10 anchor match.

| stratum | depth | baseline | missed | oracle-rescuable | top10-rescuable | structural ceiling | matched ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- |
| content_grounded | @1 | 0.250 | 120 | 19 | 14 | 0.369 | 0.338 |
| content_grounded | @5 | 0.719 | 45 | 8 | 5 | 0.769 | 0.750 |
| cross_lingual | @1 | 0.000 | 38 | 7 | 1 | 0.184 | 0.026 |
| cross_lingual | @5 | 0.105 | 34 | 6 | 1 | 0.263 | 0.132 |
| role_query | @1 | 0.111 | 32 | 3 | 2 | 0.194 | 0.167 |
| role_query | @5 | 0.444 | 20 | 1 | 1 | 0.472 | 0.472 |
| overall | @1 | 0.188 | 190 | 29 | 17 | 0.312 | 0.261 |
| overall | @5 | 0.577 | 99 | 15 | 7 | 0.641 | 0.607 |

**Exact-repeat vs fingerprint-disjoint holdout (grounded arm)**

| subset | items | @1 | @3 | @10 | oracle |
| --- | --- | --- | --- | --- | --- |
| exact_repeat | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| fingerprint_disjoint | 233 | 0.030 | 0.064 | 0.094 | 0.146 |

### Cutoff 2026-07-01T00:00:00Z

**grounded** — anchors 2938, edges 7296

| stratum | items | after cutoff | with eligible anchor | @1 | @3 | @10 | oracle (any anchor) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| content_grounded | 160 | 46 | 160 | 0.700 | 0.762 | 0.781 | 0.819 |
| ↳ content_grounded leak-free only | 46 | 46 | 46 | 0.130 | 0.239 | 0.283 | 0.391 |
| cross_lingual | 38 | 38 | 38 | 0.000 | 0.000 | 0.079 | 0.474 |
| role_query | 36 | 36 | 36 | 0.000 | 0.056 | 0.083 | 0.139 |
| overall | 234 | 120 | 234 | 0.479 | 0.530 | 0.560 | 0.658 |

> **114 of 234 items predate this cutoff**: their own source events are inside the anchor training window, so the un-suffixed rows above are leak-contaminated and are shown only to size the leak. Read the `leak-free only` rows, or regenerate the goldset at this cutoff.

**unfiltered** — anchors 6432, edges 52233

| stratum | items | after cutoff | with eligible anchor | @1 | @3 | @10 | oracle (any anchor) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| content_grounded | 160 | 46 | 160 | 0.756 | 0.787 | 0.831 | 0.863 |
| ↳ content_grounded leak-free only | 46 | 46 | 46 | 0.152 | 0.261 | 0.413 | 0.522 |
| cross_lingual | 38 | 38 | 38 | 0.000 | 0.105 | 0.316 | 0.711 |
| role_query | 36 | 36 | 36 | 0.083 | 0.083 | 0.139 | 0.333 |
| overall | 234 | 120 | 234 | 0.530 | 0.568 | 0.641 | 0.756 |

> **114 of 234 items predate this cutoff**: their own source events are inside the anchor training window, so the un-suffixed rows above are leak-contaminated and are shown only to size the leak. Read the `leak-free only` rows, or regenerate the goldset at this cutoff.

**Headroom over the anchor-free baseline run (grounded arm)**

`structural_ceiling` assumes every anchor-reachable miss converts to a hit and nothing regresses; `matched_ceiling` uses the measured top-10 anchor match.

| stratum | depth | baseline | missed | oracle-rescuable | top10-rescuable | structural ceiling | matched ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- |
| content_grounded | @1 | 0.250 | 120 | 99 | 93 | 0.869 | 0.831 |
| content_grounded | @5 | 0.719 | 45 | 36 | 32 | 0.944 | 0.919 |
| cross_lingual | @1 | 0.000 | 38 | 18 | 3 | 0.474 | 0.079 |
| cross_lingual | @5 | 0.105 | 34 | 15 | 2 | 0.500 | 0.158 |
| role_query | @1 | 0.111 | 32 | 4 | 2 | 0.222 | 0.167 |
| role_query | @5 | 0.444 | 20 | 2 | 1 | 0.500 | 0.472 |
| overall | @1 | 0.188 | 190 | 121 | 98 | 0.705 | 0.607 |
| overall | @5 | 0.577 | 99 | 53 | 35 | 0.803 | 0.726 |

**Exact-repeat vs fingerprint-disjoint holdout (grounded arm)**

| subset | items | @1 | @3 | @10 | oracle |
| --- | --- | --- | --- | --- | --- |
| exact_repeat | 105 | 1.000 | 1.000 | 1.000 | 1.000 |
| fingerprint_disjoint | 129 | 0.054 | 0.147 | 0.202 | 0.380 |

### Cutoff 2026-07-15T00:00:00Z

**grounded** — anchors 3178, edges 7879

| stratum | items | after cutoff | with eligible anchor | @1 | @3 | @10 | oracle (any anchor) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| content_grounded | 160 | 19 | 160 | 0.844 | 0.881 | 0.900 | 0.925 |
| ↳ content_grounded leak-free only | 19 | 19 | 19 | 0.158 | 0.263 | 0.316 | 0.421 |
| cross_lingual | 38 | 38 | 38 | 0.026 | 0.026 | 0.105 | 0.579 |
| role_query | 36 | 36 | 36 | 0.000 | 0.056 | 0.111 | 0.167 |
| overall | 234 | 93 | 234 | 0.581 | 0.615 | 0.650 | 0.752 |

> **141 of 234 items predate this cutoff**: their own source events are inside the anchor training window, so the un-suffixed rows above are leak-contaminated and are shown only to size the leak. Read the `leak-free only` rows, or regenerate the goldset at this cutoff.

**unfiltered** — anchors 6904, edges 56507

| stratum | items | after cutoff | with eligible anchor | @1 | @3 | @10 | oracle (any anchor) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| content_grounded | 160 | 19 | 160 | 0.906 | 0.919 | 0.944 | 0.956 |
| ↳ content_grounded leak-free only | 19 | 19 | 19 | 0.211 | 0.316 | 0.526 | 0.632 |
| cross_lingual | 38 | 38 | 38 | 0.053 | 0.158 | 0.368 | 0.842 |
| role_query | 36 | 36 | 36 | 0.083 | 0.083 | 0.194 | 0.472 |
| overall | 234 | 93 | 234 | 0.641 | 0.667 | 0.735 | 0.863 |

> **141 of 234 items predate this cutoff**: their own source events are inside the anchor training window, so the un-suffixed rows above are leak-contaminated and are shown only to size the leak. Read the `leak-free only` rows, or regenerate the goldset at this cutoff.

**Headroom over the anchor-free baseline run (grounded arm)**

`structural_ceiling` assumes every anchor-reachable miss converts to a hit and nothing regresses; `matched_ceiling` uses the measured top-10 anchor match.

| stratum | depth | baseline | missed | oracle-rescuable | top10-rescuable | structural ceiling | matched ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- |
| content_grounded | @1 | 0.250 | 120 | 111 | 107 | 0.944 | 0.919 |
| content_grounded | @5 | 0.719 | 45 | 40 | 36 | 0.969 | 0.944 |
| cross_lingual | @1 | 0.000 | 38 | 22 | 4 | 0.579 | 0.105 |
| cross_lingual | @5 | 0.105 | 34 | 19 | 3 | 0.605 | 0.184 |
| role_query | @1 | 0.111 | 32 | 5 | 3 | 0.250 | 0.194 |
| role_query | @5 | 0.444 | 20 | 2 | 1 | 0.500 | 0.472 |
| overall | @1 | 0.188 | 190 | 138 | 114 | 0.778 | 0.675 |
| overall | @5 | 0.577 | 99 | 61 | 40 | 0.838 | 0.748 |

**Exact-repeat vs fingerprint-disjoint holdout (grounded arm)**

| subset | items | @1 | @3 | @10 | oracle |
| --- | --- | --- | --- | --- | --- |
| exact_repeat | 131 | 1.000 | 1.000 | 1.000 | 1.000 |
| fingerprint_disjoint | 103 | 0.049 | 0.126 | 0.204 | 0.437 |

### Cutoff 2026-08-01T00:00:00Z

**grounded** — anchors 3271, edges 8051

| stratum | items | after cutoff | with eligible anchor | @1 | @3 | @10 | oracle (any anchor) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| content_grounded | 160 | 8 | 160 | 0.894 | 0.925 | 0.938 | 0.963 |
| ↳ content_grounded leak-free only | 8 | 8 | 8 | 0.000 | 0.125 | 0.125 | 0.375 |
| cross_lingual | 38 | 38 | 38 | 0.026 | 0.053 | 0.105 | 0.579 |
| role_query | 36 | 36 | 36 | 0.000 | 0.056 | 0.111 | 0.167 |
| overall | 234 | 82 | 234 | 0.615 | 0.650 | 0.675 | 0.778 |

> **152 of 234 items predate this cutoff**: their own source events are inside the anchor training window, so the un-suffixed rows above are leak-contaminated and are shown only to size the leak. Read the `leak-free only` rows, or regenerate the goldset at this cutoff.

**unfiltered** — anchors 7174, edges 58943

| stratum | items | after cutoff | with eligible anchor | @1 | @3 | @10 | oracle (any anchor) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| content_grounded | 160 | 8 | 160 | 0.963 | 0.969 | 0.981 | 0.988 |
| ↳ content_grounded leak-free only | 8 | 8 | 8 | 0.250 | 0.375 | 0.625 | 0.750 |
| cross_lingual | 38 | 38 | 38 | 0.079 | 0.211 | 0.395 | 0.842 |
| role_query | 36 | 36 | 36 | 0.083 | 0.083 | 0.222 | 0.583 |
| overall | 234 | 82 | 234 | 0.684 | 0.709 | 0.769 | 0.902 |

> **152 of 234 items predate this cutoff**: their own source events are inside the anchor training window, so the un-suffixed rows above are leak-contaminated and are shown only to size the leak. Read the `leak-free only` rows, or regenerate the goldset at this cutoff.

**Headroom over the anchor-free baseline run (grounded arm)**

`structural_ceiling` assumes every anchor-reachable miss converts to a hit and nothing regresses; `matched_ceiling` uses the measured top-10 anchor match.

| stratum | depth | baseline | missed | oracle-rescuable | top10-rescuable | structural ceiling | matched ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- |
| content_grounded | @1 | 0.250 | 120 | 116 | 112 | 0.975 | 0.950 |
| content_grounded | @5 | 0.719 | 45 | 43 | 39 | 0.988 | 0.963 |
| cross_lingual | @1 | 0.000 | 38 | 22 | 4 | 0.579 | 0.105 |
| cross_lingual | @5 | 0.105 | 34 | 19 | 3 | 0.605 | 0.184 |
| role_query | @1 | 0.111 | 32 | 5 | 3 | 0.250 | 0.194 |
| role_query | @5 | 0.444 | 20 | 2 | 1 | 0.500 | 0.472 |
| overall | @1 | 0.188 | 190 | 143 | 119 | 0.799 | 0.697 |
| overall | @5 | 0.577 | 99 | 64 | 43 | 0.850 | 0.761 |

**Exact-repeat vs fingerprint-disjoint holdout (grounded arm)**

| subset | items | @1 | @3 | @10 | oracle |
| --- | --- | --- | --- | --- | --- |
| exact_repeat | 142 | 1.000 | 1.000 | 1.000 | 1.000 |
| fingerprint_disjoint | 92 | 0.022 | 0.109 | 0.174 | 0.435 |

### Cutoff 2026-08-10T00:00:00Z

**grounded** — anchors 3331, edges 8159

| stratum | items | after cutoff | with eligible anchor | @1 | @3 | @10 | oracle (any anchor) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| content_grounded | 160 | 4 | 160 | 0.906 | 0.938 | 0.950 | 0.963 |
| ↳ content_grounded leak-free only | 4 | 4 | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| cross_lingual | 38 | 38 | 38 | 0.026 | 0.053 | 0.105 | 0.579 |
| role_query | 36 | 36 | 36 | 0.000 | 0.056 | 0.083 | 0.167 |
| overall | 234 | 78 | 234 | 0.624 | 0.658 | 0.679 | 0.778 |

> **156 of 234 items predate this cutoff**: their own source events are inside the anchor training window, so the un-suffixed rows above are leak-contaminated and are shown only to size the leak. Read the `leak-free only` rows, or regenerate the goldset at this cutoff.

**unfiltered** — anchors 7489, edges 60690

| stratum | items | after cutoff | with eligible anchor | @1 | @3 | @10 | oracle (any anchor) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| content_grounded | 160 | 4 | 160 | 0.975 | 0.975 | 0.981 | 0.988 |
| ↳ content_grounded leak-free only | 4 | 4 | 4 | 0.000 | 0.000 | 0.250 | 0.500 |
| cross_lingual | 38 | 38 | 38 | 0.105 | 0.237 | 0.421 | 0.842 |
| role_query | 36 | 36 | 36 | 0.083 | 0.111 | 0.278 | 0.611 |
| overall | 234 | 78 | 234 | 0.697 | 0.722 | 0.782 | 0.906 |

> **156 of 234 items predate this cutoff**: their own source events are inside the anchor training window, so the un-suffixed rows above are leak-contaminated and are shown only to size the leak. Read the `leak-free only` rows, or regenerate the goldset at this cutoff.

**Headroom over the anchor-free baseline run (grounded arm)**

`structural_ceiling` assumes every anchor-reachable miss converts to a hit and nothing regresses; `matched_ceiling` uses the measured top-10 anchor match.

| stratum | depth | baseline | missed | oracle-rescuable | top10-rescuable | structural ceiling | matched ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- |
| content_grounded | @1 | 0.250 | 120 | 116 | 114 | 0.975 | 0.963 |
| content_grounded | @5 | 0.719 | 45 | 43 | 41 | 0.988 | 0.975 |
| cross_lingual | @1 | 0.000 | 38 | 22 | 4 | 0.579 | 0.105 |
| cross_lingual | @5 | 0.105 | 34 | 19 | 3 | 0.605 | 0.184 |
| role_query | @1 | 0.111 | 32 | 5 | 2 | 0.250 | 0.167 |
| role_query | @5 | 0.444 | 20 | 2 | 1 | 0.500 | 0.472 |
| overall | @1 | 0.188 | 190 | 143 | 120 | 0.799 | 0.701 |
| overall | @5 | 0.577 | 99 | 64 | 45 | 0.850 | 0.769 |

**Exact-repeat vs fingerprint-disjoint holdout (grounded arm)**

| subset | items | @1 | @3 | @10 | oracle |
| --- | --- | --- | --- | --- | --- |
| exact_repeat | 146 | 0.993 | 0.993 | 0.993 | 0.993 |
| fingerprint_disjoint | 88 | 0.011 | 0.102 | 0.159 | 0.420 |

## 4. Parent probe reproduction

| stratum | probe @1 | probe @3 | probe @10 | reproduced @1 | @3 | @10 | grounded @1 | @3 | @10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| content_grounded | 0.031 | 0.056 | 0.113 | 0.031 | 0.056 | 0.113 | 0.044 | 0.081 | 0.119 |
| cross_lingual | 0.000 | 0.079 | 0.158 | 0.000 | 0.079 | 0.158 | 0.000 | 0.000 | 0.026 |
| role_query | 0.083 | 0.111 | 0.139 | 0.083 | 0.111 | 0.139 | 0.028 | 0.083 | 0.083 |

## 5. Goldset label stability under rebuild

`content_grounded` labels come from recorded events; `cross_lingual` labels are the paraphrase query's own top-k on the snapshot being built from, so they move when retrieval moves.

| rebuild | items | content_grounded | cross_lingual changed | role_query changed |
| --- | --- | --- | --- | --- |
| `goldset-2026-06-10.jsonl` | 1867 | 1793 | 14 / 38 (36.8%) | 0 / 36 (0.0%) |
| `goldset-2026-07-15.jsonl` | 343 | 269 | 14 / 38 (36.8%) | 0 / 36 (0.0%) |
| `goldset-2026-08-01.jsonl` | 246 | 172 | 14 / 38 (36.8%) | 0 / 36 (0.0%) |

## 6. Goldset structure

234 items: content_grounded 160, cross_lingual 38, role_query 36

- items with a source event timestamp (a per-item cutoff can bind them): **160**
- items with `source_event_id: null` (no cutoff binds them): **74** cross_lingual 38, role_query 36

