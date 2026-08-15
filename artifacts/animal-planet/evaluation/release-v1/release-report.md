# Animal-planet default-off release v1

Status: **PASS with P6 deferred**. P2-P5 pass; automatic-only compaction is not claimed by this release.

## P2 — payload and retention

| Evidence | Median after/before | Median ratio | P90 after/before | P90 ratio | Minimum retention |
|---|---:|---:|---:|---:|---:|
| Override-free eval | 14394/25345 | 0.5679226672 | 17455/30176 | 0.5784398197 | 1.0 |
| Preserved holdout aggregate | 13707/23956 | 0.5721739856 | 17519/29595 | 0.5919581010 | 1.0 |

Eval-to-holdout reduction gaps are 0.425132 pp (median) and 1.351828 pp (p90).

## P3 — scope precision

| Evidence | Cross-scope admission reduction | Same-scope retention |
|---|---:|---:|
| Override-free eval | 0.5583756345 | 0.9788025288 |
| Preserved holdout aggregate | 0.5984604368 | 0.9571535022 |

## P4 and P5 — correction and era safety

The correction suite passed 26 tests and the era-safety suite passed 45 tests. Eval replay reports 0 violations. Recorded-candidate replay exercises zero correction/superseded pairs in eval and preserved holdout; the 26-test live/property/SQLite focused suite is the substantive ordering proof.

## P6 — deferred

Status: `deferred_pending_confirmatory_v3`. `LM_RECALL_REPEAT_GATING` and `LM_RECALL_REPEAT_DROP_TRAILING_STUBS` are both false with overrides absent; the override-free eval replay gated zero events. The legacy path is class-blind and experimental. Historical negative evidence remains byte-identical.

## Evidence boundary

`eval-compare.json` is the calculator's stdout byte-for-byte. The release manifest hash-binds the implementation, effective configuration, original and replacement packet manifests, retired v2 design, latency/context-cost evidence, and all seven historical aggregates. No holdout calculator or semantic case reader was invoked.
