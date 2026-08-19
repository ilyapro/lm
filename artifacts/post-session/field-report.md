# Post-session extraction — sealed-holdout field report

Batch `field-20260819`, generated 2026-08-19T14:14:48Z. **Verdict: INCONCLUSIVE** (gate PASS, mode extracted_vs_control).

Measured on the already-accumulated transcripts before scaling, because "written" is not "used". Pre-registration `7361f0f10c72d15e…` was frozen before any holdout number existed. The sealed holdout digest `e7042c4dd49e913c…` could not be re-derived (AE archival renames path-derived ae_node_result keys and the CLI's rolling window pruned the oldest claude sliver — see `field-report.json#seal`); every measured session instead carries an individual sealed-membership proof: stable UUID key, started before the sealing instant, singleton identity group, own-key bucket in the sealed holdout buckets, and named by no development artifact.

## The four pre-registered numbers

| # | Metric | Value | Bar | Pass |
|---|--------|-------|-----|------|
| 1 | extracted vs organic consumption | 0.2222 vs 0.117 (ratio 1.8991) | >= 0.5x organic (0.0585) | True |
| 2 | absolute extracted rate | 0.2222 | >= 0.05 (protocol floor) | True |
| 3 | anti-dump audit precision (holdout) | 0.6667 (95% CI [0.3542, 0.8794]) | >= 0.70 | False |
| 4 | eval->holdout precision degradation | -1.6668 | <= 0.25 | True |

Legacy scalar floor: the goal text's 0.15 is the deprecated retrospective-protocol floor; against it the extracted rate meets 0.15, but the baseline forbids reading the scalar and the organic control itself scores 0.117 under this protocol

## Cohorts under one protocol

| Cohort | Nodes | Consumption | 95% CI | Grounded |
|--------|------:|------------:|--------|---------:|
| extracted (holdout) | 9 | 0.2222 | [0.0632, 0.5474] | 1.0 (n=2) |
| extracted (eval) | 8 | 0.0 | [0.0, 0.3244] | None (n=0) |
| organic control | 949 | 0.117 | [0.098, 0.139] | 0.1591 (n=88) |

Consumption degradation eval->holdout (diagnostic): None.

## Live writes

None. verdict INCONCLUSIVE: the pre-registered bar was not cleared; the extractor signals and stops instead of writing

## What would have to change

- cohort: 9 seeded holdout traces < minimum 20. At the observed yield (9 accepted ops per 300 sessions) a decision-grade cohort needs ~667 eligible sessions; the whole sealed holdout holds 381 today. The corpus must accumulate more finished sessions; lowering the minimum is only admissible in a fresh pre-registration for a new batch.
- audit: holdout anti-dump precision 0.6667 (6/9) is below 0.7; 7/9 was required. Raising it means tightening the extractor's gate, which is extractor iteration: allowed against train/eval only, then re-measured on a NEW sealed batch - this batch's holdout sessions and numbers are burned.
- invariant: no bar, floor, replay set or audit rule may be tuned to these numbers; any change re-runs the full protocol on a fresh sealed batch under a new pre-registration.

## Protocol

- Snapshot `04e6276876453f71…` (16793 nodes, 54892 recall events), captured 2026-08-19T11:22:47Z, cut at as-of 2026-08-19T00:00:00Z.
- Replay set: every recorded recall event in [2026-08-15T00:00:00Z, 2026-08-19T00:00:00Z] — the arm pinned by the baseline's replay_set_rule.
- Extracted traces seeded with created_at 2026-08-14T23:59:59Z; cohorts selected by context marker under the identical window, tool, as-of, containment and horizon.
- Full pre-registration, per-phase digests and untracked-state hashes: see `field-report.json`.
