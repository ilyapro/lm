# Typed-edge backfill audit

Generated: 2026-07-07T20:03:36Z | mode: `apply` | DB: `/tmp/lm_bf.sqlite3`

## Guarantees

- **Additive-only**: no existing edge is deleted, retyped, or reweighted.
- **Idempotent**: a second run inserts 0 new rows (all pairs already connected).
- **Rollback**: `DELETE FROM connections WHERE json_extract(metadata,'$.basis')='provenance_derivation';` (marker `metadata.basis = "provenance_derivation"`, a namespace no other edge uses).

## Totals

- New edge rows inserted: **10139**
- New rows by type: related=10126, caused=4, supersedes=9
- Net-new connectivity (pairs with no prior edge, any type/direction): **9098** — the non-redundant graph gain (§4).
- Rows derived by more than one rule (collapsed): 45
- R5a `derived_from` annotations on existing schema→trace edges: 5228

## Edge type counts

| type | before | after |
|---|---|---|
| related | 99878 | 110004 |
| caused | 0 | 4 |
| contradicts | 3066 | 3066 |
| supersedes | 239 | 248 |
| requires | 0 | 0 |
| total | 103183 | 113322 |

## Typed / provenance share

| metric | before | after |
|---|---|---|
| strict-typed | 3305 (0.0320) | 3318 (0.0293) |
| provenance-derived | 3305 (0.0320) | 18672 (0.1648) |

Target (artifacts/discovery/typed-edge-rules.md §4): strict-typed ~0.030 (3,305 -> 3,318 / 111,689), provenance-derived ~0.157 (with R5a annotations applied).

## Per-rule counts

| rule | type | derived | new rows | net-new pairs | existing skipped |
|---|---|---|---|---|---|
| R1c | supersedes | 9 | 9 | 8 | 0 |
| R2a | caused | 4 | 4 | 2 | 0 |
| R4a | related | 207 | 202 | 48 | 5 |
| R4b | related | 10089 | 9857 | 8942 | 232 |
| R6 | related | 199 | 112 | 106 | 87 |
| R5a | related/derived_from (annotation, no new rows) | 5228 | 5228 | - | - |

> Per-rule `derived` counts a pair under each contributing rule; the physical row total is `totals.new_rows` (collapses in `totals.multi_rule_rows`). `net-new pairs` (pairs with no prior edge of any type in either direction) is the §4 non-redundancy projection: `new rows` is larger because exact `(source,target,type)` novelty also types pairs already linked in reverse/another type, as the write path does.

