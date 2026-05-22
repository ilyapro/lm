# Health Audit Metric Contract

Authoritative metric and output-schema contract for the `fix-health-audit`
subtree (`scope:memory-quality-root-fixes/fix-health-audit`). Consumed by the
sibling children `audit-engine`, `memory-health-surface`, `fixture-tests`, and
`sample-output`. Numbers cited as "baseline" come from
`artifacts/baseline.md` against the frozen snapshot
`/tmp/lm-baseline-replay.sqlite3` (MD5 `58fc12fd47c5f71e5b8867ee71590668`).

## 0. Invariants

1. **Surface is the existing `resources.memory_health`.** The audit metrics are
   additive sections on the same JSON object returned today by
   `src/living_memory/resources.py:323-459`. No new MCP tool, no new HTTP route,
   no new resource URI, no daemon.
2. **Default invocation stays cheap.** A `memory_health` call without new
   optional arguments must remain in the same latency budget as today
   (`after/before ≤ 1.25`, absolute increase ≤ 25 ms — `ANALYSIS.md` §6.9).
3. **Read-only.** All metric SQL is read-only. No `INSERT`, `UPDATE`, `DELETE`,
   `CREATE`, or state-mutating `PRAGMA`. The audit engine must accept a
   `MemoryStore` that may have been opened from a read-only URI
   (`file:...?mode=ro`) and must not assume a writable connection.
4. **Append-only / no schema change.** Live SQLite schema is not altered by the
   audit. The existing `nodes`, `recall_events`, and `connections` tables are
   sufficient; the storage-side schema bump for `content_fingerprint` is owned
   by `fix-dedup-noise`, not by this fix.
5. **No latency side effects.** The hot-recall latency probe must use the
   production retrieval helper with `log_access=False` and `log_event=False`,
   so no `recall_events` row is inserted and no `nodes.access_count` /
   `nodes.last_accessed` columns are touched. Default `memory_health` does
   **not** run the probe.
6. **Existing fields stay byte-compatible.** The keys already returned by
   `memory_health` (`scope`, `window_hours`, `generated_at`, `counts`,
   `activity`, `dedup`, `staleness`, `retrieval_policy`) keep their shape and
   value semantics. New keys are added at the top level only.

## 1. Top-level output schema

The audit extension turns the existing `memory_health` JSON into:

```jsonc
{
  // --- existing (unchanged) ---
  "scope": "all" | "<normalized scope>",
  "window_hours": <int>,
  "generated_at": "<ISO 8601 Z>",
  "counts": { ... },
  "activity": { ... },
  "dedup": { ... },              // §2.1 duplicate density
  "staleness": { ... },
  "retrieval_policy": { ... },

  // --- new additive audit sections ---
  "feedback":      { ... },       // §2.2 feedback_applied ratio
  "access":        { ... },       // §2.3 never-accessed ratio
  "scope_hygiene": { ... },       // §2.4 scope leakage candidates
  "storage":       { ... },       // §2.5 DB size
  "latency":       null | { ... },// §2.6 hot recall latency baseline (opt-in)
  "instructions":  null | { ... } // §3 supplemental (opt-in)
}
```

All audit values are JSON-serializable scalars / lists / dicts of scalars.
Floats are computed as `1.0 * num / NULLIF(denom, 0)` and may be `None` (JSON
`null`) only when the denominator is zero. Counts are non-negative integers.

The six metrics named in the parent SPEC (`fix-health-audit` P1) map to the
fields below:

| # | Metric                       | Section                        |
|---|------------------------------|--------------------------------|
| 1 | duplicate density            | `dedup.duplicate_density`      |
| 2 | feedback_applied ratio       | `feedback.feedback_applied_ratio` |
| 3 | scope leakage candidates     | `scope_hygiene.*`              |
| 4 | never-accessed ratio         | `access.active_nodes.*`, `access.active_traces.*` |
| 5 | DB size                      | `storage.db_size_bytes`        |
| 6 | hot recall latency baseline  | `latency.summary.median_ms` (opt-in) |

## 2. Required metric formulas

Every formula below is sourced from `artifacts/baseline.md` (§"SQL Metrics" /
§"Hot Recall Latency") and `artifacts/discovery/health-observability.md`
(§"Baseline-Aligned Metric Definitions"). The audit engine **must** compute
these from DB contents at call time; baseline figures are reproduction targets,
not hardcoded outputs.

### 2.1 Duplicate density (`dedup`, already implemented)

Active-trace exact-content collisions.

```sql
SELECT
  COUNT(*)                                                           AS total_traces,
  COUNT(DISTINCT content)                                            AS distinct_contents,
  COUNT(*) - COUNT(DISTINCT content)                                 AS duplicate_excess,
  1.0 * (COUNT(*) - COUNT(DISTINCT content)) / NULLIF(COUNT(*), 0)   AS duplicate_density
FROM nodes
WHERE level = 'trace' AND decayed = 0
  [AND scope = :scope]   -- when memory_health(scope=...) is set
```

Output keys (existing): `total_traces`, `distinct_contents`, `duplicate_excess`,
`duplicate_density`. Baseline (scope=None): `5029 / 4377 / 652 / 0.129648`.

### 2.2 Feedback applied ratio (`feedback`, new)

Per-scope filter rule matches the existing `activity.recall_total` rule
(`resources.py:341-359`): when `scope` is set, filter `recall_events.scope = ?`
**only** (do not OR with `requested_scope`).

```sql
SELECT
  COUNT(*)                                                                   AS recall_events,
  SUM(CASE WHEN feedback_applied = 1 THEN 1 ELSE 0 END)                      AS feedback_applied,
  COUNT(*) - SUM(CASE WHEN feedback_applied = 1 THEN 1 ELSE 0 END)           AS feedback_missing,
  1.0 * SUM(CASE WHEN feedback_applied = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS feedback_applied_ratio
FROM recall_events
  [WHERE scope = :scope]
```

Output keys: `recall_events`, `feedback_applied`, `feedback_missing`,
`feedback_applied_ratio` (JSON `null` if `recall_events == 0`). Baseline
(scope=None): `5114 / 1060 / 4054 / 0.207274`.

### 2.3 Never-accessed ratio (`access`, new)

Two aggregates: all active nodes, and active traces only. `access_count = 0`
counts traces / nodes that were inserted but never returned by a recall.

```sql
-- active nodes
SELECT
  COUNT(*)                                                                          AS total,
  SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END)                                 AS never_accessed,
  1.0 * SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0)     AS never_accessed_ratio
FROM nodes
WHERE decayed = 0
  [AND scope = :scope]

-- active traces (subset)
SELECT ... FROM nodes
WHERE decayed = 0 AND level = 'trace'
  [AND scope = :scope]
```

Output shape:

```json
"access": {
  "active_nodes":  {"total": <int>, "never_accessed": <int>, "never_accessed_ratio": <float|null>},
  "active_traces": {"total": <int>, "never_accessed": <int>, "never_accessed_ratio": <float|null>}
}
```

Baseline (scope=None): `active_nodes 3710 / 5128 / 0.723479`,
`active_traces 3681 / 5029 / 0.731955`.

### 2.4 Scope leakage candidates (`scope_hygiene`, new)

This is an audit predicate, not a write-side fix. It is **not** filtered by
`memory_health(scope=...)`; it always runs against the full DB so that
project-scoped audit calls do not silently miss other-project leakage.

**Verbatim audit predicate** (must be used identically across `audit-engine`,
`fixture-tests`, and `sample-output`; sourced from
`artifacts/baseline.md` §"Scope Leakage Candidates" and
`artifacts/discovery/baseline-raw.sql:109-262`):

```sql
scope = 'rise' OR scope LIKE 'rise/%' OR scope LIKE 'scope:rise%'
  OR scope = 'breakthrough' OR scope LIKE 'breakthrough/%'
  OR scope = 'ocpa-generative-action-substrate-v1'
  OR scope LIKE 'ocpa-generative-action-substrate-v1/%'
  OR scope = 'project:rise' OR scope LIKE 'project:rise/%'
  OR scope = 'project:breakthrough' OR scope LIKE 'project:breakthrough/%'
  OR scope = 'project:ocpa-generative-action-substrate-v1'
  OR scope LIKE 'project:ocpa-generative-action-substrate-v1/%'
```

Apply the same predicate against three columns: `nodes.scope`,
`recall_events.scope`, and `recall_events.requested_scope`. Candidate scopes
are the **union** of those three column sets.

Required output keys (counts only — detail list is bounded, see below):

```json
"scope_hygiene": {
  "candidate_scopes":                                <int>,  // |UNION over the three columns|
  "active_candidate_nodes":                          <int>,  // nodes matching predicate AND decayed = 0
  "active_candidate_traces":                         <int>,  // ... AND level = 'trace'
  "active_candidate_traces_never_accessed":          <int>,  // ... AND access_count = 0
  "active_candidate_trace_never_accessed_ratio":     <float|null>,
  "candidate_scopes_with_zero_recall_events":        <int>,  // candidate scopes where neither recall_events.scope nor .requested_scope appears
  "top_candidate_scopes": [                                 // bounded detail list, ordered deterministically
    {
      "scope":                              <str>,
      "active_traces":                      <int>,
      "active_candidate_traces_never_accessed": <int>,
      "recall_events_as_scope":             <int>,
      "recall_events_as_requested_scope":   <int>
    },
    ...
  ]
}
```

- `top_candidate_scopes` list is sized to a small bounded cap (default `20`,
  configurable downstream). Ordering is deterministic:
  `(active_traces DESC, recall_events_as_scope ASC, scope ASC)` — high-volume
  bare-scope rows surface first; ties break by ASCII scope.
- Baseline (against snapshot): `candidate_scopes = 54`,
  `active_candidate_traces = 111`,
  `active_candidate_traces_never_accessed = 111`,
  `active_candidate_trace_never_accessed_ratio = 1.0`,
  `candidate_scopes_with_zero_recall_events = 36`.

### 2.5 DB size (`storage`, new)

Two independent measures so the audit reports both filesystem size (what an
operator sees on disk) and SQLite-internal size (portable across filesystems
and `:memory:` stores).

```python
db_path     = store.db_path                                          # may be ":memory:"
db_size     = db_path.stat().st_size if str(db_path) != ":memory:" else None
page_count  = conn.execute("PRAGMA page_count").fetchone()[0]
page_size   = conn.execute("PRAGMA page_size").fetchone()[0]
page_bytes  = page_count * page_size
```

Output shape:

```json
"storage": {
  "db_path":        <str>,            // string form of store.db_path
  "db_size_bytes":  <int|null>,       // null when path == ":memory:"
  "page_count":     <int>,
  "page_size":      <int>,
  "page_bytes":     <int>             // page_count * page_size
}
```

`PRAGMA page_count` and `PRAGMA page_size` are read-only queries safe under the
read-only URI. Baseline (snapshot): `db_size_bytes = 100339712`.

### 2.6 Hot recall latency baseline (`latency`, new, opt-in)

Opt-in via a new optional argument to `memory_health` (e.g.
`latency_samples: int = 0`). When `latency_samples <= 0`, the engine emits
`"latency": null` and runs no recall. When `latency_samples > 0`, the engine
runs **one warmup** plus `latency_samples` measured invocations of the
production retrieval helper exactly as the baseline captures it:

```python
from living_memory.retrieval import memory_recall
memory_recall(
    store,
    query,                       # default constant; see below
    scope=normalized_scope,      # passes through memory_health(scope=...)
    ambient_context={"benchmark": "memory_health.latency_probe"},
    max_results=1,
    depth=1,
    log_access=False,            # MUST be False — read-only safety
    log_event=False,             # MUST be False — no recall_events insertion
)
```

Constraints:

- `log_access=False` and `log_event=False` are **mandatory** — they are the
  only safe-against-read-only-DB combination. The audit must not call any
  retrieval path that bypasses these flags.
- Default query when not overridden:
  `"Living Memory hot recall latency baseline duplicate density feedback applied"`.
  This matches the baseline query (`baseline.md:228`) so the probe is
  comparable across runs.
- `max_results = 1`, `depth = 1` (matches baseline; deeper expansion is a
  separate experiment owned by `recall-speed-usefulness`, not by health).
- Warmup wall-time is reported but excluded from min/median/mean/max.
- Wall time is measured with `time.perf_counter()`; values are milliseconds
  rounded to 3 decimals.
- Latency probe is **only** invoked when `latency_samples > 0`; default
  `memory_health` calls never invoke it (defends ANALYSIS.md §6.9).

Output shape (when `latency_samples > 0`):

```json
"latency": {
  "query":         <str>,
  "scope":         <str>,                // resolved scope passed to memory_recall
  "max_results":   1,
  "depth":         1,
  "log_access":    false,
  "log_event":     false,
  "samples":       <int>,                // == latency_samples
  "warmup_ms":     <float>,
  "samples_ms":    [<float>, ...],       // length == samples
  "summary": {
    "min_ms":      <float>,
    "median_ms":   <float>,
    "mean_ms":     <float>,
    "max_ms":      <float>
  }
}
```

Baseline (snapshot, 5 measured samples after 1 warmup, scope `project:lm`):
`median_ms ≈ 91.869`, `min_ms ≈ 91.322`, `max_ms ≈ 92.756`. Live MCP path (out
of scope here, recorded for context) is ≈ 65-70 ms.

## 3. Supplemental section: `instructions` (out of the six, opt-in)

`fix-health-audit` may additionally report instruction-text facts when the MCP
wrapper passes `instructions_text` (per `ANALYSIS.md` §6.2). This section is
**not** one of the six required metrics; it is recorded here so consumers of
this contract do not invent a clashing shape.

```json
"instructions": null | {
  "default_scope":     <str>,
  "length_chars":      <int>,    // len(instructions_text)
  "length_bytes":      <int>,    // len(instructions_text.encode("utf-8"))
  "must_count":        <int>,    // literal "MUST" occurrences (case-sensitive)
  "must_not_count":    <int>,    // literal "MUST NOT" occurrences (case-sensitive)
  "before_count":      <int>,    // literal "BEFORE"
  "after_count":       <int>,    // literal "AFTER"
  "contract_command":  "PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_instructions_imperative.py -p no:cacheprovider"
}
```

The contract command is reported as a string. The audit **must not** invoke
pytest from inside `memory_health` (ANALYSIS.md §6.4 R5).

## 4. Scope-filter behavior summary

| Section          | Honors `memory_health(scope=...)` | Reason |
|------------------|-----------------------------------|--------|
| `dedup`          | Yes (existing)                    | Scope-local duplicate density is the established meaning. |
| `feedback`       | Yes — `recall_events.scope = ?` only | Matches existing `activity` rule (`resources.py:341-359`). |
| `access`         | Yes — `nodes.scope = ?`            | Consistent with `dedup`. |
| `scope_hygiene`  | **No** — always global             | Audit predicate is cross-scope by design; project-scoped audit calls must still see leakage. |
| `storage`        | No                                 | DB-wide measurement. |
| `latency`        | Yes — passes through to `memory_recall(scope=...)` | Lets a scoped audit measure scoped recall cost. |
| `instructions`   | No                                 | Server-instruction text is global. |

## 5. Read-only safety constraints

- `audit-engine` must accept either:
  1. an in-process `MemoryStore` instance, or
  2. a path string / `Path` that it opens under
     `sqlite3.connect(f"file:{path}?mode=ro", uri=True)` for path-based reads.
- No `attach`, `detach`, or `PRAGMA writable_schema = ON`.
- All non-`PRAGMA page_count` / `PRAGMA page_size` SQL goes through the
  existing `store.connection.execute` / `store.connection.executemany` paths.
- `latency` section is the only one that exercises the live retrieval pipeline;
  it is gated by the `log_access=False, log_event=False` invariant in §2.6 so
  the read-only mode is preserved end-to-end.
- The audit is mechanically testable: opening the connection under
  `mode=ro` and calling `memory_health(..., latency_samples=N)` against the
  frozen snapshot must succeed and must not alter the snapshot MD5.

## 6. Latency micro-check (binding for `fix-health-audit`)

Per `ANALYSIS.md` §6.9 / §9:

| Probe                                                  | Threshold |
|--------------------------------------------------------|-----------|
| `memory_health(scope=None, top_stale=5)` default call  | `after/before ≤ 1.25` AND absolute increase ≤ 25 ms |
| Default `memory_health` must call `memory_recall`?     | **No** — opt-in only |
| `memory_health(scope=None, top_stale=5, latency_samples=5)` | Includes the cost of N+1 recall calls; reported separately, no threshold |

`audit-engine` provides a benchmark snippet that the verification node can
re-run; thresholds are evaluated against
`artifacts/latency_before.json` / `artifacts/latency_after.json`.

## 7. Acceptance against the baseline

`fixture-tests` and `sample-output` must verify:

1. **Field completeness.** Every key in §1 / §2 / §3 (where opt-in is enabled)
   is present with the documented type. Absence of any required key is a test
   failure.
2. **Numeric agreement on the snapshot.** When run against the frozen snapshot
   `/tmp/lm-baseline-replay.sqlite3` (MD5 `58fc12fd47c5f71e5b8867ee71590668`)
   with `scope=None`:
   - `dedup.duplicate_density` matches `0.129648` to ≤ 1e-3
   - `feedback.feedback_applied_ratio` matches `0.207274` to ≤ 1e-3
   - `access.active_nodes.never_accessed_ratio` matches `0.723479` to ≤ 1e-3
   - `access.active_traces.never_accessed_ratio` matches `0.731955` to ≤ 1e-3
   - `scope_hygiene.candidate_scopes == 54`
   - `scope_hygiene.active_candidate_traces == 111`
   - `scope_hygiene.candidate_scopes_with_zero_recall_events == 36`
   - `storage.db_size_bytes == 100339712`
3. **Hardcoded-lookup defence.** `fixture-tests` must use a small synthetic DB
   with values **distinct** from the live baseline (per parent `_critique` of
   `fix-health-audit`). The synthetic fixture's expected ratios are derived
   from inserted counts at test time, not from a stored answer key. Tests fail
   if the audit engine returns the snapshot values regardless of fixture
   contents.
4. **Read-only mechanical check.** A fixture test opens the snapshot under
   `mode=ro` and asserts the snapshot MD5 is unchanged after a call to
   `memory_health(..., latency_samples=3)`.
5. **Determinism.** Calling `memory_health` twice against the same fixture in
   the same process yields byte-identical JSON for every non-latency, non-time
   field (i.e. excluding `generated_at`, `latency.samples_ms`,
   `latency.summary.*`, and `latency.warmup_ms`).

## 8. Out of scope for this fix

These are explicitly **not** part of `fix-health-audit` per ANALYSIS.md §6 and
§11.4; downstream children must not implement them here:

- Registering `recall_events_summary` / `connections_summary` as standalone
  MCP resources (deferred per §11.4).
- Any change to `memory_status`, HTTP `/health`, `/admin/info`,
  `/admin/restart`, or instruction-text content.
- Any schema change to `nodes`, `recall_events`, or `connections`.
- Any write-side fix for scope hygiene, dedup, or feedback linkage (owned by
  sibling fix nodes).
- Historical migration / backfill of bare-scope rows or duplicates.
- Live-DB latency probes (owned by `verify-and-record`).

## 9. Source-of-truth citations

- `artifacts/baseline.md` §"SQL Metrics", §"Hot Recall Latency"
- `artifacts/baseline.md:174-213` — scope leakage predicate values
- `artifacts/discovery/baseline-raw.sql:109-262` — full leakage candidate SQL
- `artifacts/discovery/health-observability.md` §"Baseline-Aligned Metric
  Definitions", §"Chosen Extension Surface", §"Rejected Alternatives"
- `artifacts/ANALYSIS.md` §6 (health and observability), §9 (per-fix latency
  micro-check), §10 (numeric targets)
- `src/living_memory/resources.py:323-459` — existing `memory_health`
- `src/living_memory/resources.py:462-504` — existing `recall_events_summary`
  (template for `feedback` section SQL)
- `src/living_memory/retrieval.py:108-174`, `:485-504` — production
  `memory_recall` helper used for the opt-in latency probe
- `src/living_memory/storage.py:63` — `MemoryStore.db_path`
