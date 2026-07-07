# Typed-edge corpus backfill — operator runbook

`src/living_memory/edge_backfill.py` applies the provenance-derived go-rules
(`edge_derivation.py`: **R1c** content-correction `supersedes`, **R2a**
root-cause `caused`, **R4a** same-commit `related`, **R4b** shared-files
`related`, **R6** content-reference `related`, and the optional **R5a**
`derived_from` annotation) to the **existing** corpus, so historical nodes gain
the same typed edges the write path already gives new traces.

Guarantees (all test-enforced in `tests/test_edge_backfill.py`):

- **Additive-only** — never deletes, retypes, or reweights an existing edge.
  It inserts brand-new `(source_id, target_id, type)` rows only; a pair already
  connected with the same type is skipped, not merged. (`--annotate-derived-from`
  adds only additive metadata keys to existing schema→trace edges; weight/type
  unchanged.)
- **Idempotent** — a second run inserts 0 new rows and 0 new annotations.
- **Audited** — per-rule counts, samples, before/after edge and typed/provenance
  shares in `audit.json` / `audit.md`.

## Scope boundary — live application is an operator action

**Applying to the live `/home/sfx/.local/share/living-memory/global.sqlite3` is
out of scope for tree automation and is an operator action.** Every automated
run in this tree wrote only a read-only snapshot **copy**; the live DB was never
written. The commands below are for the operator to run deliberately.

## 0. Preview — always safe (dry-run is the default)

```bash
cd <repo> && export PYTHONPATH="src:.cache/python-deps" LIVING_MEMORY_EMBEDDING_BACKEND=hash
python3 -m living_memory.edge_backfill derive-edges \
  --db /home/sfx/.local/share/living-memory/global.sqlite3 \
  --report /tmp/derive-edges-preview.json
```

Dry-run reads the live DB and writes only the audit JSON/MD — the `connections`
table is untouched. Inspect `totals.new_rows`, the per-rule table, and
`typed_share.after` before applying.

## 1. Back up, then apply — the exact one command

```bash
# (a) Consistent backup via the sqlite backup API (safe with a live WAL):
python3 -c "import sqlite3; s=sqlite3.connect('file:/home/sfx/.local/share/living-memory/global.sqlite3?mode=ro',uri=True); d=sqlite3.connect('/home/sfx/.local/share/living-memory/global.pre-backfill.sqlite3'); s.backup(d)"

# (b) Apply the full go-rule set (edges + R5a annotations):
cd <repo> && export PYTHONPATH="src:.cache/python-deps" LIVING_MEMORY_EMBEDDING_BACKEND=hash
python3 -m living_memory.edge_backfill derive-edges \
  --db /home/sfx/.local/share/living-memory/global.sqlite3 \
  --apply --annotate-derived-from \
  --report /tmp/derive-edges-live.json
```

The write path keeps typing new traces regardless; the backfill can run while
the server is up (it opens its own connection with `busy_timeout=5000` on the
shared WAL), though a quiet window is preferable.

**Conservative alternative** — row edges only, which keeps every existing edge
byte-identical and makes the rollback a single `DELETE`: omit
`--annotate-derived-from`. R5a annotations are then left to the normal
consolidation pass (`consolidation._connect_schema_to_traces` already stamps
`derived_from` on schema→trace edges it touches).

## 2. Rollback — precise

```sql
-- (a) Remove every backfill-inserted row. The marker basis='provenance_derivation'
--     is a namespace no rule or hand edge uses, so this deletes exactly what the
--     backfill added and nothing else (verified: 0 such rows pre-backfill).
DELETE FROM connections WHERE json_extract(metadata,'$.basis')='provenance_derivation';

-- (b) Only if --annotate-derived-from was used, strip the R5a annotation.
--     Optional: it changes no weight/type, and the next consolidation re-adds it.
UPDATE connections
   SET metadata = json_remove(metadata, '$.kind', '$.rule')
 WHERE type='related'
   AND json_extract(metadata,'$.rule')='R5a'
   AND json_extract(metadata,'$.basis')='procedural';
```

Simplest full restore: stop the server and swap `global.pre-backfill.sqlite3`
back into place.

## 3. Old server code reads the DB fine after backfill

- **Schema unchanged.** Opening the DB runs only `CREATE INDEX IF NOT EXISTS`
  for `idx_nodes_commit_prefix` and `idx_nodes_files_present` (additive, already
  shipped with `typed-edge-engine-v2`). No table, column, or constraint change;
  no destructive migration.
- **No unknown connection types minted.** The backfill emits only `supersedes`,
  `caused`, and `related` — all already in
  `CHECK(type IN ('related','caused','contradicts','supersedes','requires'))`.
  `caused` becomes non-empty for the first time; its traversal support already
  shipped (`retrieval._traversal` causal-mode factors 0.25/1.0), so causal-mode
  recall reads it correctly. Code that only walks `related`/`supersedes` simply
  ignores `caused` rows.
- **Traversal marker is invisible to ranking.** `retrieval._traversal` keys its
  kind-specific factors on `metadata.kind` (`derived_from`, `content_reference`)
  only. The rollback marker rides in `metadata.basis`, which traversal never
  reads; the rules' own `same_commit`/`shared_files` bases are preserved under
  `rule_basis`.

## 4. Validation evidence (live-snapshot copy, this tree)

Snapshot: 2026-07-08 copy of the live DB (`sqlite3` backup API, read-only
source). Full audit: `audit.json` / `audit.md`.

### Edge inventory and typed/provenance share

| metric | before | after | target (typed-edge-rules.md §4) |
|---|---|---|---|
| total edges | 103,183 | 113,322 | ~111,689 |
| new rows inserted | — | 10,139 (all marked `provenance_derivation`) | — |
| net-new connectivity (pairs w/ no prior edge) | — | 9,098 | ~9,025 |
| `supersedes` | 239 | 248 (+9 R1c) | +9 |
| `caused` | 0 | 4 (R2a) — causal channel filled | +4 |
| R5a `derived_from` annotations | 0 | 5,228 | 5,228 |
| **strict-typed share** | 3,305 = 3.20% | 3,318 = **2.93%** | ~3.0% |
| **provenance-derived share** | 3,305 = 3.20% | 18,672 = **16.48%** | ~15.7% |

Per-rule net-new pairs match the mining projection closely (R1c 8/§4 8, R2a
2/2, R4a 48/47, R4b 8,942/8,863, R6 106/105) — the derivation faithfully
reproduces `mine_typed_edges.py`. Raw `new rows` exceed §4's non-redundancy
projection because exact `(source,target,type)` novelty (the write path's and
this backfill's dedup rule) also types pairs already linked in the reverse
direction or by another type; those add provenance typing without new
node-to-node reach.

### Replay no-regression gate — PASS

Harness: `python3 -m living_memory.replay --cutoff 2026-06-10T00:00:00Z
--schemes live_weights,replayed_proportional --credit-rule proportional
--supersedes snapshot` on the same copy, before vs after backfill
(`--supersedes snapshot` is the only replay config sensitive to backfilled
edges — the 9 new R1c `supersedes` pairs; `related`/`caused`/R5a edges do not
enter the replay ranking). Holdout (1,537 events with useful results):

| scheme (holdout) | metric | before | after | after/before |
|---|---|---|---|---|
| live_weights | hit@5 | 0.948601 | 0.947951 | 0.9993× |
| live_weights | MRR | 0.703785 | 0.703243 | 0.9992× |
| replayed_proportional | hit@5 | 0.949252 | 0.948601 | 0.9993× |
| replayed_proportional | MRR | 0.705807 | 0.705265 | 0.9992× |

All ≥ 0.99× (gate). Against the committed baseline
(`artifacts/replay/baseline.json`, `live_weights` holdout hit@5 0.947678 / MRR
0.706212) the backfilled snapshot is also ≥ 0.99× (hit@5 1.0003×, MRR 0.9958×).
`graph_unique_share` is unchanged (0.010349) **by construction**: the harness
re-ranks the per-result method scores recorded at event time, so new-edge graph
reach is not observable here (documented candidate-selection-bias limitation).
The graph-value gain is the +9.8% non-redundant connectivity projected in §4
(9,098 net-new pairs at ~5.6% co-retrieval redundancy), realized at recall time
by `retrieval` graph expansion, not in recorded-candidate replay.
