# fix-dedup-noise — result

## Outcome
DIRECT_EXECUTION. Implemented the schema-v3 content-fingerprint write-path
dedup defined in `artifacts/ANALYSIS.md` §4 and `artifacts/discovery/dedup-noise.md`.
Identical-content trace writes in the same `(level='trace', scope)` now collapse
to one active trace plus an append-only chain of decayed older copies linked by
`supersedes` connections with `metadata.kind="duplicate_content"`. Raw history is
preserved (no `DELETE`, no `content` mutation); decayed rows remain reachable
via `get_node`, `include_decayed=True` list calls, and the supersedes edges.

## Changes
- `src/living_memory/storage.py`
  - Bumped `SCHEMA_VERSION` 2 → 3.
  - Added `content_fingerprint TEXT` column to `nodes` CREATE TABLE.
  - Added `idx_nodes_dedup` partial index on `(level, scope, content_fingerprint) WHERE decayed = 0 AND content_fingerprint IS NOT NULL`.
  - Added `_migrate_pre_v3_schema()` (`ALTER TABLE ADD COLUMN` if column missing).
  - Added `_backfill_missing_content_fingerprints()` (one-shot SHA-256 backfill of NULL fingerprints, idempotent).
  - In `_insert_node`: compute fingerprint once; for `level == "trace"`, look up older active duplicates by `(level, scope, content_fingerprint)`, INSERT the new node carrying the fingerprint, then for each duplicate insert a `supersedes` edge (`metadata.kind="duplicate_content"`, `metadata.fingerprint=hex`) and UPDATE the older row to `decayed=1, decay_reason='duplicate_content'`. All inside the existing per-`create_node` transaction.
  - Concept/schema nodes are intentionally NOT deduped (`memory_consolidate` and `memory_teach` own their own merge semantics).
- `tests/test_dedup_supersede.py` (new, 11 contract tests):
  - same-scope identical content supersedes older trace
  - supersedes edge carries `kind="duplicate_content"` + `fingerprint`
  - cross-scope identical content is NOT deduped
  - cross-level identical content is NOT deduped
  - `[file-chunk]`, `[file-summary]`, `[project-overview]` shaped content dedup correctly
  - 10 repeated inserts yield exactly 1 active + 9 decayed + 9 supersedes edges
  - dedup preserves original content + `include_decayed=True` history + reachability of supersedes chain (history-preservation case)
  - dedup excludes already-decayed traces (does not re-decay or attach to them)
  - `memory_recall` returns at most one result for content that was inserted 5 times
- `tests/test_storage.py`:
  - Added `content_fingerprint` to expected column set; added `idx_nodes_dedup` index assertions.
  - Added `test_schema_version_is_three_after_initialize`.
  - Added `test_schema_v2_database_migrates_to_v3_with_backfill` (50-row v2 fixture migrates to v3 with backfilled fingerprints).
  - Added `test_repeated_initialize_does_not_rewrite_existing_fingerprints` (backfill idempotency).
- `tests/test_resources_prompts.py`:
  - Adjusted `test_memory_health_reports_activity_dedup_and_staleness` to insert three distinct contents (post-fix the previous identical-content setup would collapse) and assert the steady-state `duplicate_density == 0.0`.
  - Added `test_memory_health_dedup_collapses_identical_writes` to verify the active/decayed split after dedup and that the metric reads `0%` on the active row set.
- `tests/test_consolidation.py`:
  - `test_consolidation_clusters_cross_language_traces_into_one_concept` now tags each insert with the incident index so byte-identical content does not collapse during the cross-language clustering fixture.
- `tests/test_procedural_schemas.py`, `tests/test_schema_distillation.py`:
  - `_append_procedure_traces` helpers prefix the content with the procedure id so two procedure runs in the same scope produce distinct content.
  - `test_nodes_table_has_no_new_columns` expected set includes `content_fingerprint`.

## Verification
- `npm run check` → 182 passed, 3 failed.
  - The 3 failures are `tests/test_instructions_imperative.py::{test_required_keywords_present, test_must_not_present, test_size_within_mcp_budget}`. Verified pre-existing on the baseline branch tip via `git stash && pytest && git stash pop` — unrelated to dedup; they predate this branch.
- `tests/test_dedup_supersede.py` (11) and `tests/test_storage.py` (7 incl. new schema-v3 cases) both pass.
- `tests/test_acceptance_contract.py` (11) passes; cold-start recall budget remains satisfied.
- Write-path micro-benchmark (artifacts/dedup_latency_microbench.json, 500 inserts × 3 repeats per round, fresh tmp DB):
  - Unique-insert per-call mean: 0.244 ms (target ≤ 1.0 ms; 1.07× of master 0.227 ms; within 1.20× budget).
  - Duplicate-insert per-call mean: 0.402 ms (target ≤ 2.0 ms; 1.59× of master 0.253 ms, absolute well under target).
  - Read path is unchanged (no edits to retrieval.py); canonical read-path before/after lives with the `verify-and-record` sibling.

## Append-only invariants preserved
1. No `DELETE FROM nodes`.
2. `update_node` continues to refuse `content` changes on traces (`storage.py:344-345`).
3. Decayed-by-dedup rows are reachable via `get_node(old_id)` and via `list_nodes(..., include_decayed=True)`.
4. `supersedes` edges record the audit trail; backfill of `content_fingerprint` reads `content` only.

## Deferred follow-ups (out of fix-node scope)
- One-shot retroactive dedup of the existing 625 duplicate excess in live `project:ae` (operator-triggered).
- Path/kind-based supersede for changed files (different problem; AE-side cleaner).
- AE caller round-trip optimisation in `cmd_bootstrap_project`.
- Health-surface enrichment for new dedup metrics (owned by `fix-health-audit`).

## Known caveat
The pre-existing `test_instructions_imperative.py` failures (3) are baseline-branch-tip
state, not introduced by this fix. They block a fully-green `npm run check`. They
should be addressed in a separate workstream owned by whoever maintains the
server instructions text.
