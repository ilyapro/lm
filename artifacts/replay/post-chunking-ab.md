# Post-chunking retrieval-weight A/B

Generated 2026-08-17T22:44:52Z | commit `0c39c104b6baf7b1ea4cdeaf63bf6b8a1c4ac0ff` | report v1

**Decision: KEEP THE INCUMBENT.** `explicit_b0.75_v0.10_g0.15` won on eval but did not clear the holdout bar (hit@5 0.6934 vs 0.7005, MRR 0.3872 vs 0.4031), so the incumbent stands.

## What was measured, and why not with `replay` alone

Max-pool over chunk embeddings changed `vector_score` itself, so every per-result score recorded in `recall_events.results` belongs to the pre-chunking distribution. Re-ranking those rows would have measured nothing about the current system. Every number below therefore comes from **queries re-run end-to-end through the current code** (`retrieval_harness.run_goldset` -> `MemoryRecallService.memory_recall`) against a chunk-backfilled snapshot; the regenerated per-result `bm25/vector/graph/trigger` scores are then fed to the *unmodified* replay A/B machinery (`replay.run_scheme`, `static_resolver`, `explicit_weights`, `floor_default_weights`, `WeightTrajectory`) through `retrieval_harness.to_replay_event`.

What this can and cannot show: the candidate *set* of each query is whatever the live learned weights actually retrieved, so the A/B measures **the blend**, not recall of documents no scheme retrieved. That is the correct scope for a weight recalibration and the same scope every prior `artifacts/replay/` report used.

## Corpus and the three disjoint time slices

`/home/sfx/.cache/living-memory-harness/recalib/goldset-recalibration.jsonl` — 3659 content-grounded items built by `retrieval_harness.build_content_grounded` (same IDF-containment label, same active-node filter as the frozen goldset) over the full labeled history, uncapped. The frozen `artifacts/harness/goldset.jsonl` (sha256 `af40cb0da26fa78c…`) is read as frozen input and scored separately at the end.

C1 = `2026-06-01T00:00:00Z` C2 = `2026-06-20T00:00:00Z`; train: created_at <= C1; eval: C1 < created_at <= C2; holdout: created_at > C2, on the recall event's own created_at.

| slice | events | first event | last event | query-id digest |
|---|---|---|---|---|
| train | 1168 | 2026-05-15T07:25:32Z | 2026-05-31T19:35:01Z | `4051c6190aeb191f…` |
| eval | 1369 | 2026-06-01T07:52:50Z | 2026-06-19T23:43:32Z | `f80008f1f491c95d…` |
| holdout | 1122 | 2026-06-20T00:09:25Z | 2026-08-17T16:36:15Z | `774cc56a87e10671…` |

The slices are disjoint by construction and verified as sets in `tests/test_retrieval_weight_recalibration.py`: the published `query_ids` lists share no element, each is non-empty, and holdout carries 1122 labeled events (bar: >= 1000).

## Incumbent

The incumbent is `incumbent_live_weights` — the per-scope weights **as learned and stored in the snapshot's `retrieval_weights` table**, resolved through the live fallback chain (`replay.static_resolver`: exact scope, then family, then `default`). Config defaults only ever seed *new* scopes, so the incumbent for every scope that already exists is its learned row, not the config value.

| scope | bm25 | vector | graph |
|---|---|---|---|
| `global` | 0.1506 | 0.7994 | 0.0500 |
| `project:ae` | 0.1478 | 0.7395 | 0.1128 |
| `project:lm` | 0.1000 | 0.8500 | 0.0500 |
| `project:octopus` | 0.1043 | 0.7757 | 0.1201 |
| `project:online` | 0.1562 | 0.7365 | 0.1073 |
| `project:x` | 0.1000 | 0.8493 | 0.0507 |

## Stage 1 — TRAIN: fitting and grid search

258 schemes scored on the 1168-event train slice: the fixed schemes, 126 explicit weight triples and 126 config-default triples over the same simplex grid (step 0.05, graph <= 0.3), plus 3 weight trajectories replayed with `replay.WeightTrajectory` over the train events only.

Train leaders (top 10 by hit@5):

| scheme | events | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| `config_b0.70_v0.15_g0.15` | 1168 | 0.0942 | 0.4084 | 0.5659 | 0.2218 |
| `config_b0.75_v0.10_g0.15` | 1168 | 0.0942 | 0.4084 | 0.5659 | 0.2218 |
| `config_b0.80_v0.05_g0.15` | 1168 | 0.0942 | 0.4084 | 0.5659 | 0.2218 |
| `config_b0.85_v0.00_g0.15` | 1168 | 0.0942 | 0.4084 | 0.5659 | 0.2218 |
| `explicit_b0.75_v0.10_g0.15` | 1168 | 0.0942 | 0.4084 | 0.5659 | 0.2217 |
| `explicit_b0.70_v0.15_g0.15` | 1168 | 0.0942 | 0.4084 | 0.5659 | 0.2217 |
| `config_b0.65_v0.20_g0.15` | 1168 | 0.0942 | 0.4084 | 0.5659 | 0.2215 |
| `explicit_b0.65_v0.20_g0.15` | 1168 | 0.0942 | 0.4084 | 0.5659 | 0.2215 |
| `config_b0.75_v0.15_g0.10` | 1168 | 0.0925 | 0.4084 | 0.5659 | 0.2206 |
| `config_b0.80_v0.10_g0.10` | 1168 | 0.0925 | 0.4084 | 0.5659 | 0.2206 |
| `incumbent_live_weights` | 1168 | 0.1045 | 0.3810 | 0.5651 | 0.2222 |

The last row is the incumbent, ranked **222 of 258** on train. Worth stating plainly this early, because it is the shape of the whole result: the incumbent looks poor on the two slices used to search and select, and best of every finalist on the slice that decides.

## Stage 2 — EVAL: selection

**Selection rule, fixed before the holdout slice was read:**

> Finalists are the incumbent live weights, the current config defaults, uniform, the three train-fitted weight trajectories, and the top-5 explicit and top-5 config-default grid candidates ranked on TRAIN. Among the finalists, the winner is the scheme with the highest EVAL hit@5; ties are broken by higher EVAL MRR, then by preferring the incumbent, then by scheme name. The winner is adopted only if, on HOLDOUT, its hit@5 >= the incumbent's hit@5 AND its MRR >= the incumbent's MRR. If it clears neither or only one of those, the incumbent weights are kept. HOLDOUT is scored exactly once, in a single final pass after the winner is locked by this rule.

| scheme | events | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| `explicit_b0.75_v0.10_g0.15` | 1369 | 0.1074 | 0.5661 | 0.7757 | 0.2899 |
| `explicit_b0.80_v0.05_g0.15` | 1369 | 0.1066 | 0.5661 | 0.7757 | 0.2893 |
| `config_b0.65_v0.20_g0.15` | 1369 | 0.1074 | 0.5654 | 0.7757 | 0.2897 |
| `explicit_b0.65_v0.20_g0.15` | 1369 | 0.1074 | 0.5654 | 0.7757 | 0.2897 |
| `config_b0.70_v0.15_g0.15` | 1369 | 0.1074 | 0.5639 | 0.7757 | 0.2893 |
| `config_b0.75_v0.10_g0.15` | 1369 | 0.1074 | 0.5639 | 0.7757 | 0.2893 |
| `config_b0.80_v0.05_g0.15` | 1369 | 0.1074 | 0.5639 | 0.7757 | 0.2893 |
| `config_b0.85_v0.00_g0.15` | 1369 | 0.1074 | 0.5639 | 0.7757 | 0.2893 |
| `explicit_b0.70_v0.15_g0.15` | 1369 | 0.1074 | 0.5639 | 0.7757 | 0.2893 |
| `explicit_b0.75_v0.15_g0.10` | 1369 | 0.1066 | 0.5632 | 0.7757 | 0.2891 |
| `config_defaults_current` | 1369 | 0.1074 | 0.5610 | 0.7750 | 0.2899 |
| `trajectory_proportional_from_config` | 1369 | 0.1169 | 0.5573 | 0.7772 | 0.2957 |
| `uniform` | 1369 | 0.1125 | 0.5522 | 0.7765 | 0.2905 |
| `trajectory_proportional_from_live` | 1369 | 0.1125 | 0.5478 | 0.7757 | 0.2908 |
| `incumbent_live_weights` | 1369 | 0.1132 | 0.5435 | 0.7765 | 0.2910 |
| `trajectory_winner_take_all_from_live` | 1369 | 0.1140 | 0.5420 | 0.7765 | 0.2910 |

Winner on eval: **`explicit_b0.75_v0.10_g0.15`** — one explicit triple applied to every scope: (0.75, 0.1, 0.15)

## Stage 3 — HOLDOUT: scored once, after the winner was locked

**Generalization bar:** the winner is adopted only if its holdout hit@5 >= the incumbent's holdout hit@5 AND its holdout MRR >= the incumbent's holdout MRR.

| scheme | events | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| `incumbent_live_weights` | 1122 | 0.1916 | 0.7005 | 0.8966 | 0.4031 |
| `trajectory_winner_take_all_from_live` | 1122 | 0.1952 | 0.6996 | 0.8966 | 0.4054 |
| `config_defaults_current` | 1122 | 0.1711 | 0.6996 | 0.8948 | 0.3907 |
| `trajectory_proportional_from_live` | 1122 | 0.1881 | 0.6979 | 0.8966 | 0.4004 |
| `explicit_b0.75_v0.15_g0.10` | 1122 | 0.1702 | 0.6952 | 0.8948 | 0.3888 |
| `uniform` | 1122 | 0.1916 | 0.6943 | 0.8948 | 0.3965 |
| `trajectory_proportional_from_config` | 1122 | 0.1800 | 0.6943 | 0.8966 | 0.3946 |
| `config_b0.65_v0.20_g0.15` | 1122 | 0.1720 | 0.6943 | 0.8948 | 0.3890 |
| `explicit_b0.65_v0.20_g0.15` | 1122 | 0.1720 | 0.6943 | 0.8948 | 0.3890 |
| `config_b0.70_v0.15_g0.15` | 1122 | 0.1738 | 0.6934 | 0.8948 | 0.3897 |
| `config_b0.75_v0.10_g0.15` | 1122 | 0.1738 | 0.6934 | 0.8948 | 0.3897 |
| `config_b0.80_v0.05_g0.15` | 1122 | 0.1738 | 0.6934 | 0.8948 | 0.3897 |
| `config_b0.85_v0.00_g0.15` | 1122 | 0.1738 | 0.6934 | 0.8948 | 0.3897 |
| `explicit_b0.70_v0.15_g0.15` | 1122 | 0.1738 | 0.6934 | 0.8948 | 0.3897 |
| `explicit_b0.75_v0.10_g0.15` | 1122 | 0.1702 | 0.6934 | 0.8948 | 0.3872 |
| `explicit_b0.80_v0.05_g0.15` | 1122 | 0.1693 | 0.6916 | 0.8948 | 0.3864 |

| condition | incumbent | winner | delta | verdict |
|---|---|---|---|---|
| holdout hit@5 | 0.7005 | 0.6934 | -0.0071 | FAIL |
| holdout mrr | 0.4031 | 0.3872 | -0.0159 | FAIL |

`explicit_b0.75_v0.10_g0.15` led on eval but failed the holdout bar on hit@5 and mrr, which is exactly the overfitting the three-way split exists to catch. The incumbent weights stand and no configuration value changes.

### How many queries actually moved

Aggregate deltas hide the count, so here it is per query over the same 1122 holdout events. `explicit_b0.75_v0.10_g0.15` puts a relevant node in the top-5 where the incumbent does not on **34** events; the incumbent does so where the challenger does not on **42**; both succeed on 744 and both fail on 302. At the rank level the challenger is better on 127 events, worse on 216, unchanged on 779. Exact two-sided sign tests: p=0.4222 on the hit@5 disagreements, p=2.0e-06 on rank movement.

Read together those two tests say something sharper than the aggregate table. The hit@5 gap alone is small enough to be noise — 34 against 42 events is not a distinguishable difference. The rank-level comparison is not noise: on the 343 holdout events where the two schemes disagree at all, the challenger is worse on roughly two of every three (p ~ 2e-06). The eval winner is not merely unproven on holdout, it is measurably the weaker ranking of the two.

### Why the eval optimum does not survive: the slices are not identically distributed

The grid leaders on train and eval are bm25-heavy, the holdout leader is the vector-heavy incumbent. That is a property of the data, not of the search: the per-channel score distribution of the returned results shifts across the slices.

| slice | results | mean bm25 | mean vector | mean graph | vector p90 | graph nonzero |
|---|---|---|---|---|---|---|
| train | 12142 | 0.1341 | 0.4910 | 0.0666 | 0.6931 | 0.2318 |
| eval | 14135 | 0.1153 | 0.4910 | 0.0775 | 0.6887 | 0.2770 |
| holdout | 11498 | 0.1326 | 0.5341 | 0.0645 | 0.7054 | 0.2177 |

### The premise, measured rather than assumed — and it does not hold as stated

Over the 11964 (recall event, node) pairs that appear both in the recorded row and in the end-to-end re-run of the same query, the vector score moved on 79.1% of them — but **downward**, not upward: 2512 up (21.0%), 6950 down, 2502 unchanged; mean 0.5275 -> **0.4935** (-0.0339), p50 0.5782 -> 0.5503, p90 0.7262 -> 0.6949.

The premise this recalibration was authored on — *max-pool makes `vector_score` stochastically >= the old single-vector score* — is therefore **false for the code that actually shipped**. Max-pool alone would indeed only raise a node's score (a max over windows dominates the first window), but the shipped vector channel subtracts a length-bias correction of `LENGTH_BIAS_LOG2_COEFFICIENT * log2(chunk_count)` with the coefficient measured at 0.031 (`retrieval.py`, `_pooled_chunk_similarities`), which costs an 8-chunk node 0.093 and a 20-chunk node 0.134 — more than max-pool gains for most nodes. A second, smaller contribution is that some nodes' content changed between the event being recorded and this snapshot, so their vectors differ for reasons unrelated to chunking.

The conclusion of the goal's premise survives even though its direction does not: the distribution the live weights were fitted against no longer exists, which is why re-ranking recorded scores would have measured nothing and why every number here comes from regenerated ones. What changes is the expected *sign* of the correction — there was no reason to expect the blend to need less vector weight, and the measurement says it needs none of that adjustment either.

### Per-scope breakdown on holdout

| scope | scheme | events | hit@5 | MRR |
|---|---|---|---|---|
| `_other` | `config_b0.65_v0.20_g0.15` | 26 | 0.5000 | 0.3129 |
| `_other` | `config_b0.70_v0.15_g0.15` | 26 | 0.5385 | 0.3206 |
| `_other` | `config_b0.75_v0.10_g0.15` | 26 | 0.5385 | 0.3206 |
| `_other` | `config_b0.80_v0.05_g0.15` | 26 | 0.5385 | 0.3206 |
| `_other` | `config_b0.85_v0.00_g0.15` | 26 | 0.5385 | 0.3206 |
| `_other` | `config_defaults_current` | 26 | 0.4615 | 0.3175 |
| `_other` | `explicit_b0.65_v0.20_g0.15` | 26 | 0.5000 | 0.3129 |
| `_other` | `explicit_b0.70_v0.15_g0.15` | 26 | 0.5385 | 0.3206 |
| `_other` | `explicit_b0.75_v0.10_g0.15` | 26 | 0.5385 | 0.3206 |
| `_other` | `explicit_b0.75_v0.15_g0.10` | 26 | 0.5385 | 0.3206 |
| `_other` | `explicit_b0.80_v0.05_g0.15` | 26 | 0.5385 | 0.3212 |
| `_other` | `incumbent_live_weights` | 26 | 0.5000 | 0.3158 |
| `_other` | `trajectory_proportional_from_config` | 26 | 0.4615 | 0.3171 |
| `_other` | `trajectory_proportional_from_live` | 26 | 0.5000 | 0.2965 |
| `_other` | `trajectory_winner_take_all_from_live` | 26 | 0.5000 | 0.3158 |
| `_other` | `uniform` | 26 | 0.5385 | 0.3279 |
| `global` | `config_b0.65_v0.20_g0.15` | 31 | 0.8710 | 0.4019 |
| `global` | `config_b0.70_v0.15_g0.15` | 31 | 0.8710 | 0.4019 |
| `global` | `config_b0.75_v0.10_g0.15` | 31 | 0.8710 | 0.4019 |
| `global` | `config_b0.80_v0.05_g0.15` | 31 | 0.8710 | 0.4019 |
| `global` | `config_b0.85_v0.00_g0.15` | 31 | 0.8710 | 0.4019 |
| `global` | `config_defaults_current` | 31 | 0.8710 | 0.4530 |
| `global` | `explicit_b0.65_v0.20_g0.15` | 31 | 0.8710 | 0.4019 |
| `global` | `explicit_b0.70_v0.15_g0.15` | 31 | 0.8710 | 0.4008 |
| `global` | `explicit_b0.75_v0.10_g0.15` | 31 | 0.8710 | 0.4008 |
| `global` | `explicit_b0.75_v0.15_g0.10` | 31 | 0.8710 | 0.4008 |
| `global` | `explicit_b0.80_v0.05_g0.15` | 31 | 0.8710 | 0.3992 |
| `global` | `incumbent_live_weights` | 31 | 0.7742 | 0.4571 |
| `global` | `trajectory_proportional_from_config` | 31 | 0.7742 | 0.4410 |
| `global` | `trajectory_proportional_from_live` | 31 | 0.7742 | 0.4463 |
| `global` | `trajectory_winner_take_all_from_live` | 31 | 0.8065 | 0.5030 |
| `global` | `uniform` | 31 | 0.9032 | 0.4710 |
| `project:ae` | `config_b0.65_v0.20_g0.15` | 174 | 0.6839 | 0.3922 |
| `project:ae` | `config_b0.70_v0.15_g0.15` | 174 | 0.6839 | 0.3949 |
| `project:ae` | `config_b0.75_v0.10_g0.15` | 174 | 0.6839 | 0.3949 |
| `project:ae` | `config_b0.80_v0.05_g0.15` | 174 | 0.6839 | 0.3949 |
| `project:ae` | `config_b0.85_v0.00_g0.15` | 174 | 0.6839 | 0.3949 |
| `project:ae` | `config_defaults_current` | 174 | 0.7011 | 0.3967 |
| `project:ae` | `explicit_b0.65_v0.20_g0.15` | 174 | 0.6839 | 0.3922 |
| `project:ae` | `explicit_b0.70_v0.15_g0.15` | 174 | 0.6839 | 0.3949 |
| `project:ae` | `explicit_b0.75_v0.10_g0.15` | 174 | 0.6782 | 0.3940 |
| `project:ae` | `explicit_b0.75_v0.15_g0.10` | 174 | 0.6897 | 0.3953 |
| `project:ae` | `explicit_b0.80_v0.05_g0.15` | 174 | 0.6782 | 0.3937 |
| `project:ae` | `incumbent_live_weights` | 174 | 0.6897 | 0.4119 |
| `project:ae` | `trajectory_proportional_from_config` | 174 | 0.6839 | 0.4101 |
| `project:ae` | `trajectory_proportional_from_live` | 174 | 0.6839 | 0.4101 |
| `project:ae` | `trajectory_winner_take_all_from_live` | 174 | 0.6782 | 0.4179 |
| `project:ae` | `uniform` | 174 | 0.6897 | 0.4117 |
| `project:lm` | `config_b0.65_v0.20_g0.15` | 73 | 0.7397 | 0.4823 |
| `project:lm` | `config_b0.70_v0.15_g0.15` | 73 | 0.7397 | 0.4816 |
| `project:lm` | `config_b0.75_v0.10_g0.15` | 73 | 0.7397 | 0.4816 |
| `project:lm` | `config_b0.80_v0.05_g0.15` | 73 | 0.7397 | 0.4816 |
| `project:lm` | `config_b0.85_v0.00_g0.15` | 73 | 0.7397 | 0.4816 |
| `project:lm` | `config_defaults_current` | 73 | 0.7123 | 0.4808 |
| `project:lm` | `explicit_b0.65_v0.20_g0.15` | 73 | 0.7397 | 0.4823 |
| `project:lm` | `explicit_b0.70_v0.15_g0.15` | 73 | 0.7397 | 0.4816 |
| `project:lm` | `explicit_b0.75_v0.10_g0.15` | 73 | 0.7397 | 0.4786 |
| `project:lm` | `explicit_b0.75_v0.15_g0.10` | 73 | 0.7397 | 0.4816 |
| `project:lm` | `explicit_b0.80_v0.05_g0.15` | 73 | 0.7397 | 0.4784 |
| `project:lm` | `incumbent_live_weights` | 73 | 0.7123 | 0.4759 |
| `project:lm` | `trajectory_proportional_from_config` | 73 | 0.7123 | 0.4690 |
| `project:lm` | `trajectory_proportional_from_live` | 73 | 0.7123 | 0.4690 |
| `project:lm` | `trajectory_winner_take_all_from_live` | 73 | 0.7123 | 0.4759 |
| `project:lm` | `uniform` | 73 | 0.7397 | 0.4659 |
| `project:octopus` | `config_b0.65_v0.20_g0.15` | 1 | 0.0000 | 0.0000 |
| `project:octopus` | `config_b0.70_v0.15_g0.15` | 1 | 0.0000 | 0.0000 |
| `project:octopus` | `config_b0.75_v0.10_g0.15` | 1 | 0.0000 | 0.0000 |
| `project:octopus` | `config_b0.80_v0.05_g0.15` | 1 | 0.0000 | 0.0000 |
| `project:octopus` | `config_b0.85_v0.00_g0.15` | 1 | 0.0000 | 0.0000 |
| `project:octopus` | `config_defaults_current` | 1 | 0.0000 | 0.0000 |
| `project:octopus` | `explicit_b0.65_v0.20_g0.15` | 1 | 0.0000 | 0.0000 |
| `project:octopus` | `explicit_b0.70_v0.15_g0.15` | 1 | 0.0000 | 0.0000 |
| `project:octopus` | `explicit_b0.75_v0.10_g0.15` | 1 | 0.0000 | 0.0000 |
| `project:octopus` | `explicit_b0.75_v0.15_g0.10` | 1 | 0.0000 | 0.0000 |
| `project:octopus` | `explicit_b0.80_v0.05_g0.15` | 1 | 0.0000 | 0.0000 |
| `project:octopus` | `incumbent_live_weights` | 1 | 0.0000 | 0.0000 |
| `project:octopus` | `trajectory_proportional_from_config` | 1 | 0.0000 | 0.0000 |
| `project:octopus` | `trajectory_proportional_from_live` | 1 | 0.0000 | 0.0000 |
| `project:octopus` | `trajectory_winner_take_all_from_live` | 1 | 0.0000 | 0.0000 |
| `project:octopus` | `uniform` | 1 | 0.0000 | 0.0000 |
| `project:online` | `config_b0.65_v0.20_g0.15` | 219 | 0.6484 | 0.3676 |
| `project:online` | `config_b0.70_v0.15_g0.15` | 219 | 0.6393 | 0.3645 |
| `project:online` | `config_b0.75_v0.10_g0.15` | 219 | 0.6393 | 0.3645 |
| `project:online` | `config_b0.80_v0.05_g0.15` | 219 | 0.6393 | 0.3645 |
| `project:online` | `config_b0.85_v0.00_g0.15` | 219 | 0.6393 | 0.3645 |
| `project:online` | `config_defaults_current` | 219 | 0.6667 | 0.3631 |
| `project:online` | `explicit_b0.65_v0.20_g0.15` | 219 | 0.6484 | 0.3676 |
| `project:online` | `explicit_b0.70_v0.15_g0.15` | 219 | 0.6393 | 0.3645 |
| `project:online` | `explicit_b0.75_v0.10_g0.15` | 219 | 0.6438 | 0.3596 |
| `project:online` | `explicit_b0.75_v0.15_g0.10` | 219 | 0.6438 | 0.3600 |
| `project:online` | `explicit_b0.80_v0.05_g0.15` | 219 | 0.6393 | 0.3591 |
| `project:online` | `incumbent_live_weights` | 219 | 0.6758 | 0.3758 |
| `project:online` | `trajectory_proportional_from_config` | 219 | 0.6667 | 0.3697 |
| `project:online` | `trajectory_proportional_from_live` | 219 | 0.6667 | 0.3697 |
| `project:online` | `trajectory_winner_take_all_from_live` | 219 | 0.6758 | 0.3766 |
| `project:online` | `uniform` | 219 | 0.6347 | 0.3634 |
| `project:x` | `config_b0.65_v0.20_g0.15` | 598 | 0.7090 | 0.3878 |
| `project:x` | `config_b0.70_v0.15_g0.15` | 598 | 0.7090 | 0.3892 |
| `project:x` | `config_b0.75_v0.10_g0.15` | 598 | 0.7090 | 0.3892 |
| `project:x` | `config_b0.80_v0.05_g0.15` | 598 | 0.7090 | 0.3892 |
| `project:x` | `config_b0.85_v0.00_g0.15` | 598 | 0.7090 | 0.3892 |
| `project:x` | `config_defaults_current` | 598 | 0.7124 | 0.3886 |
| `project:x` | `explicit_b0.65_v0.20_g0.15` | 598 | 0.7090 | 0.3878 |
| `project:x` | `explicit_b0.70_v0.15_g0.15` | 598 | 0.7090 | 0.3892 |
| `project:x` | `explicit_b0.75_v0.10_g0.15` | 598 | 0.7090 | 0.3869 |
| `project:x` | `explicit_b0.75_v0.15_g0.10` | 598 | 0.7090 | 0.3890 |
| `project:x` | `explicit_b0.80_v0.05_g0.15` | 598 | 0.7074 | 0.3859 |
| `project:x` | `incumbent_live_weights` | 598 | 0.7174 | 0.4032 |
| `project:x` | `trajectory_proportional_from_config` | 598 | 0.7124 | 0.3918 |
| `project:x` | `trajectory_proportional_from_live` | 598 | 0.7174 | 0.4032 |
| `project:x` | `trajectory_winner_take_all_from_live` | 598 | 0.7174 | 0.4032 |
| `project:x` | `uniform` | 598 | 0.7090 | 0.3955 |

## Frozen goldset (234 items), scored in the same final pass

The frozen phase-0 goldset is not time-sliceable (its `cross_lingual` and `role_query` items are curated, not recorded events), so it is not part of the selection. It is reported as an independent sanity block on the same locked schemes. `live_order` is the actual end-to-end ranking the current code produced, measured by `retrieval_harness.compute_metrics`.

| scheme | events | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|
| `live_order` | 234 | 0.1880 | 0.5769 | 0.7265 | 0.3539 |
| `explicit_b0.75_v0.15_g0.10` | 234 | 0.1538 | 0.5598 | 0.7222 | 0.3178 |
| `explicit_b0.80_v0.05_g0.15` | 234 | 0.1538 | 0.5598 | 0.7222 | 0.3171 |
| `uniform` | 234 | 0.1496 | 0.5556 | 0.7222 | 0.3173 |
| `config_b0.70_v0.15_g0.15` | 234 | 0.1538 | 0.5556 | 0.7222 | 0.3172 |
| `config_b0.75_v0.10_g0.15` | 234 | 0.1538 | 0.5556 | 0.7222 | 0.3172 |
| `config_b0.80_v0.05_g0.15` | 234 | 0.1538 | 0.5556 | 0.7222 | 0.3172 |
| `config_b0.85_v0.00_g0.15` | 234 | 0.1538 | 0.5556 | 0.7222 | 0.3172 |
| `explicit_b0.70_v0.15_g0.15` | 234 | 0.1538 | 0.5556 | 0.7222 | 0.3172 |
| `explicit_b0.75_v0.10_g0.15` | 234 | 0.1538 | 0.5556 | 0.7222 | 0.3166 |
| `trajectory_proportional_from_config` | 234 | 0.1624 | 0.5513 | 0.7222 | 0.3274 |
| `config_defaults_current` | 234 | 0.1496 | 0.5513 | 0.7222 | 0.3154 |
| `config_b0.65_v0.20_g0.15` | 234 | 0.1538 | 0.5470 | 0.7222 | 0.3173 |
| `explicit_b0.65_v0.20_g0.15` | 234 | 0.1538 | 0.5470 | 0.7222 | 0.3173 |
| `trajectory_proportional_from_live` | 234 | 0.1538 | 0.5385 | 0.7222 | 0.3215 |
| `trajectory_winner_take_all_from_live` | 234 | 0.1538 | 0.5342 | 0.7222 | 0.3202 |
| `incumbent_live_weights` | 234 | 0.1581 | 0.5299 | 0.7222 | 0.3218 |

The incumbent sits at the bottom of this table, and that is not a contradiction of the holdout result — it is why this block is a sanity check and not a selector. 234 items is roughly a fifth of the holdout slice, so single-item moves are worth 0.004 here; two of its three strata (`cross_lingual`, `role_query`) are curated jargon and role queries that the phase-1 gate already showed behave unlike recorded traffic; and its `content_grounded` items are drawn from after 2026-06-10, i.e. they straddle eval and holdout rather than forming an independent sample. The load-bearing line is `live_order`: the ranking the current code actually produced end-to-end (hit@1 0.1880, hit@5 0.5769, MRR 0.3539) reproduces `artifacts/harness/phase1-gate.md`'s published 0.188 / 0.577 / 0.354 to every digit it printed, which is the check that this run's retrieval path is the same one phase 1 gated.

## Configuration outcome

**`src/living_memory/config.py` is unchanged.** `explicit_b0.75_v0.10_g0.15` led on eval but failed the holdout bar on hit@5 and mrr, which is exactly the overfitting the three-way split exists to catch. The incumbent weights stand and no configuration value changes.

`DEFAULT_RETRIEVAL_POLICY_FLOORS`, `MemoryStore.update_retrieval_weights`, `MemoryStore.apply_retrieval_weight_floors` and `feedback._method_signals` are untouched, and pinned by `tests/test_retrieval_weight_recalibration.py`.

What this hands to the next goal: the channel blend is not the bottleneck any more. All 16 finalists land inside a 0.009 band of holdout hit@5 (0.6916-0.7005), and the incumbent is already at the top of it — re-mixing channels has almost no leverage left over what retrieval returns. The remaining headroom is in what enters the candidate set at all: on holdout, 336 of 1122 events put no relevant node in the top-5 even under the incumbent, and the frozen goldset's `cross_lingual` stratum is still at hit@5 0.1053 against 0.7188 for recorded traffic. That is a vocabulary problem — jargon and `when_to_use` triggers — not a weighting one.

## Operator runbook — re-seeding live per-scope weights (NOT executed)

Config defaults seed only *new* scopes (`MemoryStore._seed_retrieval_weights` uses `ON CONFLICT(scope) DO NOTHING`). Every scope that already has a row keeps its learned weights forever, so a config change alone reaches nothing that exists today. This run did **not** mutate the live `retrieval_weights` table and does not recommend mutating it now; the procedure below is written down so that a future goal that *does* decide to re-seed has a rehearsed one.

1. Stop the MCP server or accept that the change takes effect on its next `get_retrieval_weights` call; the table is read per ranking pass, so an update is picked up without a restart but mid-flight requests may straddle it.
2. Back up first, with the WAL: `python3 -m living_memory.retrieval_harness snapshot --source-db ~/.local/share/living-memory/global.sqlite3 --snapshot-out ~/.cache/living-memory-harness/pre-reseed.sqlite3`. A plain `cp` of the `.sqlite3` file without its `-wal` is an incomplete copy.
3. Decide the target set explicitly. Re-seeding *all* scopes discards months of per-scope learning; the defensible subset is scopes whose `updated_at` is older than the chunking migration and whose weights were therefore fitted to the pre-chunking vector distribution.
4. Apply through the API, never with raw SQL: for each target scope call `MemoryStore.set_retrieval_weights(scope, bm25=..., vector=..., graph=...)` followed by `MemoryStore.apply_retrieval_weight_floors(scope, weights)` so the floors in `DEFAULT_RETRIEVAL_POLICY_FLOORS` are enforced exactly as the learning loop enforces them. Raw `UPDATE retrieval_weights` bypasses the floor pass and can leave a scope below `vector_min`/`bm25_min`.
5. Verify with `memory_health`: `retrieval_skew.scopes_at_risk` must be empty and `floors_in_effect` must match the configured floors. Then re-run this script against a fresh snapshot and confirm the incumbent row now matches the intended weights.
6. Rollback is a restore of the step-2 snapshot, or a second `set_retrieval_weights` call with the values recorded in this report's `incumbent.weights_by_scope` block.

## Reproducing this report

```
python3 scripts/replay_post_chunking.py --snapshot /home/sfx/.cache/living-memory-harness/phase1/frozen-chunked.sqlite3 --c1 2026-06-01T00:00:00Z --c2 2026-06-20T00:00:00Z --observe-live-db /home/sfx/.local/share/living-memory/global.sqlite3 --report /home/sfx/p/lm/.worktrees/_node_exec_vector-recall-chunking-harness_weights-recalibration/artifacts/replay/post-chunking-ab.json --markdown /home/sfx/p/lm/.worktrees/_node_exec_vector-recall-chunking-harness_weights-recalibration/artifacts/replay/post-chunking-ab.md
```

Snapshot `/home/sfx/.cache/living-memory-harness/phase1/frozen-chunked.sqlite3` (sha256 `cc1675fba54e5d53…`, 664096768 bytes, 12862 active nodes, 54777 recall events), embedding backend `auto`. The live database was never opened through `MemoryStore` and was never opened for writing.

One honesty note that a bare "nothing was mutated" would hide. The MCP server kept serving throughout this work, and its ordinary implicit-feedback loop keeps learning: of the 48 scopes in the live `retrieval_weights` table, **3** have moved away from the snapshot's values since it was frozen. Nothing here wrote them — the file was opened `file:...?mode=ro` for this reading only — but the rows are not frozen either, which is the strongest possible argument for the runbook above: these are learned rows, and a config default never reaches them.

| scope | snapshot bm25/vector/graph | live bm25/vector/graph | L1 | live updated_at |
|---|---|---|---|---|
| `global` | 0.1506 / 0.7994 / 0.0500 | 0.1693 / 0.7807 / 0.0500 | 0.0373 | 2026-08-17T21:52:07Z |
| `project:living-memory` | 0.2855 / 0.6645 / 0.0500 | 0.3022 / 0.6478 / 0.0500 | 0.0334 | 2026-08-17T21:02:02Z |
| `project:x` | 0.1000 / 0.8493 / 0.0507 | 0.1000 / 0.8500 / 0.0500 | 0.0015 | 2026-08-17T21:52:07Z |

