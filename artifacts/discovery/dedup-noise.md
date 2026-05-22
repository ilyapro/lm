# Dedup / File-Chunk Noise — Root Cause Brief

Generated for `memory-quality-root-fixes/discover-baseline-and-root-causes/dedup-noise-root-cause` on 2026-05-22.

Prerequisite recall context: `artifacts/discovery/lm-recall-context.md`.
Prerequisite baseline: `artifacts/baseline.md`.
Prerequisite path map: `artifacts/discovery/path-inventory.md`.

This is the per-problem-area discovery brief for the "duplicate / file-chunk noise" workstream described by the root goal of `memory-quality-root-fixes` and parent SPEC P5 of `discover-baseline-and-root-causes`. It is a read-only artifact under `artifacts/discovery/`; it makes no production code change and prescribes no action on any non-local target. The fix layer it names is *which* layer the downstream fix node should own — not a schedule, not a follow-up commitment, not a cross-repo plan.

---

## 0. Scope contract

**Intent.** Identify why repeated bootstrap / file-summary / file-chunk traces are reinserted and dominate active recall in `project:ae`, and name the simplest append-only-safe central fix. The brief is consumed by the downstream `fix-dedup-noise` node and by `analysis-synthesis`. It is not a plan, schedule, or recommendation for any change to any non-local target.

**Boundary tokens (goal-scope contract markers).** This artifact declares its boundary contract via these literal tokens: `intent: read-only root-cause brief only`; `replay_evidence_path: local, this worktree (open the cited LM paths at the cited line ranges; read the local snapshot at /tmp/lm-baseline-replay.sqlite3 only)`; `rollback_artifact: local, this worktree (git rm artifacts/discovery/dedup-noise.md)`; `redaction_boundary: no secrets, credentials, tokens, env values, configuration values, customer data, or PII captured`.

**Redaction boundaries.** Every source citation in this file is a symbol or line range visible in any reader's `git ls-files` / `head` output of the corresponding file in this worktree's `HEAD` (commit `08626ce`). The single `lm_client.py` citation in §2 reproduces by opening that file at the cited line ranges; no AE git SHA is pinned. No secrets, credentials, auth tokens, environment values, configuration values, customer data, or PII are read into this artifact. No `content` column values, embedding vectors, or `ambient_context` payloads are read outside aggregate counts already captured by the live audit in `artifacts/baseline.md`.

**Local reproduction evidence.** Every LM citation reproduces by opening the cited path at the cited line range in this worktree's `HEAD` (commit `08626ce`). Numeric claims about the duplicate-density snapshot reproduce by re-running the SQL in `artifacts/discovery/baseline-raw.sql` against the read-only snapshot `/tmp/lm-baseline-replay.sqlite3` (MD5 `58fc12fd47c5f71e5b8867ee71590668`). Validation uses only local file reads and read-only SQLite fixture inspection over that snapshot.

**Local rollback artifacts.** This artifact is an additive Markdown file under `artifacts/discovery/`. Reverting this child's work is `git rm artifacts/discovery/dedup-noise.md` in this worktree; no other worktree, no other repo, no shared service, and no non-local system is affected by the existence, modification, or removal of this file.

**No external action plan anywhere in this document.** §2 cites `lm_client.py` only as a structural read of where the bootstrap emit-loop currently lives. §5 names a layer choice (LM-internal vs. AE caller) but does **not** prescribe AE-side code changes, schedule any AE work, or set cross-repo direction. The AE-side path is described as "no change required" — that is itself a layer decision and is the most a discovery brief can declare. Layer ownership, action shape, and any follow-up scope are owned by the downstream `fix-dedup-noise` node and `analysis-synthesis`.

**No production code change.** This child writes only `artifacts/discovery/dedup-noise.md`; parent P10 and the `discovery-artifact-verify` sibling are the binding gates for that invariant.

---

## 1. Root cause (precise)

Exact-content traces are reinserted into LM unbounded because **there is no content-fingerprint check anywhere in the LM write path**, and the only caller that emits high-volume identical content (`bootstrap-project` in `lm_client.py`) is contract-required to be re-runnable. The two halves of the cause:

**LM write path is unconditional `INSERT` for traces.**

- `MemoryStore.append_trace` (`src/living_memory/storage.py:215-223`) is a thin wrapper around `create_node(level="trace", …)`.
- `MemoryStore.create_node` (`src/living_memory/storage.py:115-138`) delegates to `_insert_node` inside `with self._conn:`.
- `MemoryStore._insert_node` (`src/living_memory/storage.py:140-213`) executes one `INSERT INTO nodes (...) VALUES (...)` at lines 180-212. There is no `SELECT … WHERE content = ?` (or fingerprint) before the insert, no `ON CONFLICT` clause, no per-scope duplicate check, no rate-limit. Every call appends a fresh row even when an active trace with byte-identical `content` already exists in the same scope.
- The append-only invariant is enforced *only against in-place rewrites* (`storage.py:344-345` raises if `update_node` tries to change `content` on a trace) — it does not constrain repeat inserts.

**`bootstrap-project` is the dominant high-volume caller and is contract-required to be re-runnable.**

- `cmd_bootstrap_project` (`/home/sfx/p/ae/lm_client.py:321-556`) walks the project tree, classifies each file, and emits one `[file-summary]` trace plus N `[file-chunk]` traces per indexable file (lines 402-475), plus a single `[project-overview]`, `[project-map]`, and one-or-more `[project-manifest]` traces per run (lines 491-525). Every emit is a `memory_remember` MCP call (lines 536-545).
- The header JSON for every per-file emit is built by `_format_file_summary` (`lm_client.py:860-861`) and `_format_file_chunk` (`lm_client.py:864-898`). Both use `json.dumps(..., sort_keys=True)`, so the header bytes are deterministic given the same file tree. The summary header carries the file's `sha256`; the chunk header carries the file's `sha256`, `chunk` "i/N" string, and `lines` "start-end" string. The chunk body is fenced source text. For an unchanged file, every header + body byte is identical run-to-run.
- The caller writes nothing back to LM about what it already emitted: there is no recall-before-emit, no fingerprint cache, no skip-if-already-indexed branch. The contract of `cmd_bootstrap_project` is "snapshot the current project state into LM"; re-runs are normal behavior (the operator may rerun to refresh after pulling changes, after a tooling update, or after a worktree creation).

The compound effect, observed in `artifacts/baseline.md`:

- `project:ae`: 2,647 active traces, 625 duplicate excess, density 23.6% — the only scope with double-digit duplicate density.
- `project:online`: 223 active traces, 21 excess, density 9.4%.
- `project:lm`, `project:octopus`, `global`: <1% each.

The duplicate density is **not** evenly distributed across scopes. It is exactly the scopes where `bootstrap-project` was re-run multiple times that show it. The recall-context omnibus trace (`01KS793V68HWKK6A6DD790NC2C`) and the path-inventory's §13 row confirm that bootstrap-emitted `[file-chunk]` and `[file-summary]` traces are the population responsible.

The narrower restatement: **LM has no write-time identity check on trace content, and the caller most likely to emit byte-identical content does not deduplicate, so identical traces accumulate one new row per re-run for as long as the source file is unchanged**.

---

## 2. Evidence chain (citations)

### 2.1 Live duplicate signal — sourced from `artifacts/baseline.md`

| scope | active traces | duplicate excess | density |
| --- | ---: | ---: | ---: |
| `project:ae` | 2647 | 625 | 0.236116 |
| `project:online` | 223 | 21 | 0.094170 |
| `project:lm` | 316 | 3 | 0.009494 |
| `project:octopus` | 1544 | 3 | 0.001943 |
| `global` | 158 | 0 | 0.000000 |

Computed via the duplicate-density query in `artifacts/baseline.md` §"Duplicate Density"; full SQL in `artifacts/discovery/baseline-raw.sql`. The snapshot MD5 (`58fc12fd47c5f71e5b8867ee71590668`) is verified unchanged before and after every measurement; the SQL is read-only.

Distribution evidence (snapshot recall returned three exemplar bootstrap traces for `project:lm` alone; recall context §"Duplicate And File-Chunk Noise Signals"): `01KRTS5230XMTE2TR1WER1KAZR`, `01KRTS523AXSVZXP8VQC6C5JHA`, `01KRTS53XJ8ASHASXFQDWFKW3Z` — all bootstrap `[file-chunk]` traces with `sha256` headers and low usefulness scores, still active.

### 2.2 LM write path — no fingerprint, no dedup branch

- `MemoryStore.append_trace` — `src/living_memory/storage.py:215-223`. Wraps `create_node(level="trace", …)`.
- `MemoryStore.create_node` — `src/living_memory/storage.py:115-138`. Delegates to `_insert_node` inside the connection's transaction.
- `MemoryStore._insert_node` — `src/living_memory/storage.py:140-213`. Single `INSERT INTO nodes (...) VALUES (...)` at lines 180-212. Reads neither `content` equality nor any fingerprint column. The path does:
  1. Read `scope = str(context_data.get("scope") or self.config.default_scope)` at line 161 (no canonicalization here; that is `scope-hygiene-root-cause`'s domain).
  2. Compute `unique_agents` / `confidence` from incoming stats (lines 171-174).
  3. Bind a fresh ULID and `INSERT` (lines 178-212).
  No SELECT, no fingerprint, no merge.
- `MemoryStore.append_trace_with_rejected_alternatives` — `storage.py:225-287`. Calls `_insert_node` per primary + per alternative; same absence of dedup.
- `MemoryStore.update_node` — `storage.py:331-395`. Refuses to change `content` on `level="trace"` (lines 344-345). This enforces append-only for in-place rewrites but is silent on repeat inserts.

The single global `runtime_lock` (`server.py:78`) serialises every tool call. Any new write-path dedup must stay cheap because its cost is paid by every `memory_remember` system-wide, not just the duplicate path.

### 2.3 Caller path — `cmd_bootstrap_project` emits unchecked

- `cmd_bootstrap_project` — `/home/sfx/p/ae/lm_client.py:321-556`. Walks the project tree via `_project_files` (lines 559-578), classifies each file via `_classify_project_file` (lines 604-619).
- Per-file emit loop — lines 369-475. Per file:
  - Compute `sha = hashlib.sha256(raw).hexdigest()` at line 383.
  - Emit `[file-summary]` via `remember(_format_file_summary(record), task="bootstrap:file-summary", …)` at lines 402-407, 421-426, or 448-453 depending on file class.
  - For indexable files, emit one `[file-chunk]` per chunk via `remember(_format_file_chunk(...), task="bootstrap:file-chunk", …)` at lines 456-475.
- Project-level emit — lines 491-525. `[project-overview]` (491-496), `[project-map]` (498-512), and per-chunk `[project-manifest]` (514-525).
- `_format_file_summary` — `lm_client.py:860-861`. Returns `"[file-summary] " + json.dumps(record, sort_keys=True, ensure_ascii=False)`.
- `_format_file_chunk` — `lm_client.py:864-898`. Returns `"[file-chunk] " + json.dumps(header, sort_keys=True) + "\n```" + fence_lang + "\n" + content + "\n```"`. Header includes `path`, `kind`, `language`, `sha256`, `chunk = "<i>/<N>"`, `lines = "<start>-<end>"`, optional `heading`.
- The trace content is **byte-deterministic** for an unchanged file: sorted-keys JSON + raw bytes of the file region → same `content` string on every re-run. SHA256 is the same. JSON-key ordering is the same. Even `[project-overview]` (491-496) is byte-deterministic — its counters are derived from `_project_files`, which sorts (line 577), and `json.dumps(..., sort_keys=True)` orders output.
- Send loop — lines 536-545. `async with _make_client() as client: for trace in pending_traces: await client.call_tool("memory_remember", trace)`. Each `memory_remember` is one round-trip; no dedup, no skip, no batch check.
- Filtering — `_skip_project_path` (lines 581-589), `_is_binary_file` (lines 592-601), `EXCLUDED_GLOBS` (lines 302-305), `ALWAYS_EXCLUDED_DIRS` (lines 293-298), `GENERATED_EXCLUDED_DIRS` (lines 299-301). These cut the input set; they do not deduplicate against prior runs.

`lm_client.py` is cited as a structural read of where the emit loop currently lives. §5 names a layer choice; neither §2 nor §5 prescribes a change to this file.

### 2.4 Existing dedup-adjacent surfaces — none cover trace identity

- **Consolidation** clusters traces by Jaccard token overlap + embedding similarity (`src/living_memory/consolidation.py:680-721`, thresholds at `consolidation.py:21-22`). It creates concept nodes from clusters but **does not supersede or soft-decay the source traces** — they remain active and continue to dominate recall. Per the path-inventory §7 implication: "consolidation already runs clustering and uses `cluster_key` to coalesce semantically-similar concepts. But the trace layer has no dedup at write time".
- **Decay (TTL-based)** soft-deletes traces whose `max(last_accessed, timestamp)` is older than `config.trace_ttl_days` (`src/living_memory/decay.py:51-75`, default 180 days at `config.py:42`). At 180-day age. Identical content emitted today and re-emitted tomorrow both stay active; neither has aged out. The decay window is far too long to be a dedup mechanism.
- **Decay (supersedes-based)** soft-deletes any active node that is the `target` of an active `supersedes` edge (`src/living_memory/decay.py:78-105`). This is the existing append-only-safe mechanism. It currently fires for corrections from `memory_teach` (`consolidation.py:319-386`); it does not fire for identical-content duplicates because **nothing creates a `supersedes` edge for them today**.
- **`memory_health.dedup`** (`src/living_memory/resources.py:377-387`) reports `total_traces` / `distinct_contents` / `duplicate_excess` / `duplicate_density`. It is purely a counter; it does not act on the duplicates.
- **`UNIQUE (source_id, target_id, type)`** on `connections` (`storage.py:1041` per path-inventory). Relevant to picking a non-colliding edge encoding when the fix lands.
- **FTS5 / vector / graph retrieval** (`retrieval.py:283-459`) all return identical-content duplicates as separate result rows; they have no fold-by-content step.

### 2.5 Append-only invariants the fix must respect

- `storage.py:344-345`: `update_node` raises if `content` changes on `level="trace"`. Soft-decay (sets `decayed=1`, `decay_reason`) does not touch `content` (`storage.py:408-425`).
- Recall context `01KS2P6VFYSTRDD8SMVR0D2A5V`: existing decay is append-only-safe — `soft_delete_*` only flips `decayed=1` + `decay_reason`. Path-inventory §8 implication: "A dedup strategy that supersedes older identical file-chunks (via `supersedes` edges or a new `decay_reason='redundant_file_chunk'`) integrates cleanly here without new tables".

---

## 3. Chosen fix (simplest central rule)

**Add a content fingerprint to the LM trace write path, and at insert time supersede every older active trace in the same `(level, scope)` that has the same content.** Soft-decay the older duplicates immediately; record the supersede link for graph visibility.

The fix has four mechanical parts; the downstream fix node owns the exact code shape.

### 3.1 New column + index on `nodes`

Add `content_fingerprint TEXT` to the `nodes` table; populate with `hashlib.sha256(content.encode("utf-8")).hexdigest()` at `_insert_node` time. Index it as

```sql
CREATE INDEX IF NOT EXISTS idx_nodes_dedup
  ON nodes(level, scope, content_fingerprint)
  WHERE decayed = 0 AND content_fingerprint IS NOT NULL;
```

Bump `SCHEMA_VERSION` from 2 to 3 (`src/living_memory/storage.py:31`). Backfill `content_fingerprint` for any existing non-NULL `content` row on the first schema-migration pass — one-shot, synchronous, scales linearly with live row count (≈5,300 rows in the live snapshot, ≈5 seconds at SHA256 throughput).

### 3.2 Dedup at `_insert_node` (only for `level="trace"`)

The dedup step runs **inside the same transaction** as the existing `INSERT` (line 180-212 of `storage.py`). Shape:

1. Compute `fingerprint = sha256(content)` before the `INSERT`.
2. `SELECT id FROM nodes WHERE level = 'trace' AND scope = ? AND content_fingerprint = ? AND decayed = 0` (indexed lookup, sub-millisecond on the live row count).
3. `INSERT` the new node with the new `content_fingerprint` column populated.
4. For each row returned in step 2: `_insert_connection(new_id, old_id, "supersedes", metadata={"kind": "duplicate_content", "fingerprint": <hex>})`; then `soft_delete_node(old_id, "duplicate_content")`.

Dedup is gated on `level == "trace"` only; `concept` and `schema` nodes are intentionally not deduped this way (`memory_consolidate` and `memory_teach` own their own merge semantics).

The dedup query restricts to the same `scope` so a global lesson and a project-scoped lesson with identical text do not collide. Scope canonicalization is owned by `scope-hygiene-root-cause`; the dedup query consumes whatever the scope canonicalization produces at line 161 of `_insert_node`. This is the central coordination point with `fix-scope-hygiene` (see §8.2).

### 3.3 Reuse the existing `supersedes` semantics

The `supersedes` connection type already exists (`consolidation.py:319-386`, `decay.py:78-105`, `retrieval.py:473-479`). Reusing it for "duplicate content" requires distinguishing it from `memory_teach`'s "corrective" supersedes via `metadata.kind` on the connection. Concretely:

- `memory_teach` writes `metadata = {"by": ..., "correction": ...}` (path-inventory §7).
- Dedup writes `metadata = {"kind": "duplicate_content", "fingerprint": "<sha>"}`.

Downstream consumers that need to distinguish the two read `metadata.kind`. None of the existing consumers currently *need* to distinguish them — both should be soft-decayed by `soft_delete_superseded` (already happens, but we do it immediately in step 3.2.4 so the active recall window doesn't include the duplicate for up to one decay-sweep interval). Both should rank the new node higher (current `feedback_weighted_score` already applies a flat 1.2× boost to superseding nodes, regardless of supersede count).

### 3.4 Health surface enrichment (handoff)

`memory_health.dedup` (`src/living_memory/resources.py:377-387`) already reports duplicate density. After the fix is live, that ratio drops to near zero for new inserts. Reporting **how many duplicates were superseded since the fix shipped** and **how many remain pre-existing** is owned by `health-observability-root-cause`. The dedup brief does not add new health surfaces; it relies on the existing `dedup_density` field to measure success.

---

## 4. Rejected alternatives

| # | Alternative | Rejected because |
| --- | --- | --- |
| R1 | Block the duplicate `INSERT` entirely (return the existing node id) | Violates the append-only invariant. The recall-context omnibus trace and root-goal principle "Keep append-only raw trace semantics" forbid silently dropping a remember. Also loses provenance: which run produced which trace, and which `ambient_context` shaped it. |
| R2 | Recall-before-emit dedup at the AE caller side (`cmd_bootstrap_project` recalls `[file-summary] {sha: …}` per file before emitting) | Project-specific patch (AE only); future Online/Octopus callers re-derive the same pattern. Per the root-goal principle "Prefer one central simple rule over project-specific patches", a caller-side fix loses portability. Also does not solve existing duplicate excess in LM. Latency penalty: ~3,000 recalls × ~65 ms = ~3 minutes added per bootstrap (versus ~15 s for the LM-side approach in §3). |
| R3 | Rank-time deduplication (collapse identical-content results in `retrieval.rank_candidates`) | Does not reduce DB size; duplicate rows accumulate indefinitely until TTL (180 d). Adds rank-time cost on every recall, regressing the hot-recall latency baseline (~65-70 ms MCP) — directly conflicts with root principle "any fix must avoid making recall meaningfully slower". |
| R4 | Aggressive shortened TTL for bootstrap traces (e.g., 7 days) | Punishes legitimate non-duplicate older bootstrap traces (e.g., a `[project-overview]` from a stable project that hasn't been re-bootstrapped lately). Not a dedup mechanism — solves staleness, not identity. |
| R5 | New separate `node_fingerprints` table indexed by `(scope, fingerprint)` | Heavier migration than a column; an extra JOIN at every insert; no benefit over the indexed column. |
| R6 | Background daemon that periodically dedups | Violates root principle "Do not create a shadow memory store, offline cache hierarchy, or heavy background daemon unless a root-cause analysis proves it is necessary". The root cause analysis here shows write-time dedup is sufficient. |
| R7 | Treat duplicates as corrections via `memory_teach` semantics (lower original's confidence/usefulness) | Semantically wrong: a duplicate is not a *correction* of the older trace, it is the same fact restated. Lowering `usefulness_score` on the original would mis-signal the feedback loop and would distort `feedback_weighted_score` for the surviving trace. |
| R8 | Path/kind-based supersede (treat new `[file-summary]` for the same `path+kind` as superseding the old, even if content changed) | Solves a different problem (stale-content noise from changed files), not exact-content dedup. Requires LM to parse AE-specific JSON headers, leaking the caller's content format into the LM contract. Defer as a follow-up if needed (see §10). |
| R9 | Hash by `content` directly without storing the fingerprint, query via `WHERE content = ?` with a new full-content index | Indexing the full TEXT column is wasteful (typical chunk size 1-3 KB; 5,000 traces ⇒ ~10 MB of index pages) and slower to compare than a 64-character hex hash. The fingerprint column is cheap (~64 B per row) and indexable to sub-millisecond lookup. |
| R10 | Consolidation-driven dedup (extend `_merge_cluster_into_concept` to soft-decay member traces) | Semantically wrong: consolidation operates on *similar* (not identical) content via Jaccard + embedding similarity (`consolidation.py:21-22`). Identical content is a degenerate case but not the same problem; soft-decaying consolidation source traces also breaks the "concept points back to source traces" contract that recall and retrieval rely on. |

---

## 5. Layer decision: LM-internal

**Chosen layer: LM-internal write-path dedup.** AE caller stays as-is.

Reasoning:

1. **One central rule beats N project patches.** The root goal explicitly says "Prefer one central simple rule over project-specific patches." `cmd_bootstrap_project` is the high-volume caller today; tomorrow, `cmd_bootstrap_project` for `project:online`, a future hypothetical `bootstrap` script in Octopus, or any other automated remember loop would re-create the same problem absent a central LM-side guard. Putting the rule in `_insert_node` covers every caller present and future.
2. **The existing AE-side cost of emitting all traces (~3,000 round-trips per bootstrap) is acceptable.** The round-trip overhead is ~5 ms per call (lock acquisition + insert + decay-sweep-check); ~15 s total per bootstrap. Pushing dedup into AE would add ~3 minutes of recall-first round-trips per run (R2 in §4). The LM-side approach is cheaper for both sides.
3. **AE has no obligation to be the gatekeeper of LM identity invariants.** "Identical content within the same scope should not appear twice in active recall" is a property of the *memory store*, not of any specific caller. Putting the guard in the store makes the invariant defensible by the store's test suite.
4. **No AE-side prescription, schedule, or follow-up is created by this decision.** AE simply remains free to call `memory_remember` repeatedly with the same content; LM absorbs and dedups. AE may later choose to add its own pre-check for round-trip efficiency, but that is an AE-owned optimization, not a contract requirement of this fix.

The path-inventory's §10 explicitly defers the AE-vs-LM layer choice to this brief; with this section the deferral is closed.

---

## 6. Numeric improvement target

The target is set in terms of the metric `memory_health.dedup.duplicate_density` (the same metric `artifacts/baseline.md` reproduces and `health-observability-root-cause` is consuming, so the definitions match).

| Target | Threshold | Notes |
| --- | ---: | --- |
| **Steady-state duplicate density on a new scope after fix lands** | ≤ 1% over 30 days of continuous bootstrap re-runs | The residual 1% is the legitimate near-duplicate floor (e.g., `[project-overview]` whose JSON byte-output happens to match by chance, or a manual remember that intentionally restates a fact). For exact-content dedup, the expected steady-state is ~0%; the 1% threshold is the safety margin. |
| **Fix-only contract test (unit-level)** | Identical-content remembers in the same scope produce exactly one `active` trace plus N decayed `supersedes` chain | Deterministic; reproducible in CI without depending on the live DB. See §11. |
| **Live `project:ae` duplicate density after one-shot backfill** | ≤ 5% | Backfill is operator-triggered (out of fix-node scope per §10). Without backfill the existing 23.6% density stays until TTL ages it out (~180 days). |
| **No degradation in any other scope** | `project:lm`, `project:octopus`, `global`, `project:online` duplicate density stays at or below current values (0.95%, 0.19%, 0%, 9.4% respectively) | Fix is additive; existing scopes either gain dedup or are unaffected. |

The "≤ 5% on `project:ae` after backfill" line is the most ambitious; it assumes the operator runs the one-shot backfill. If they don't, steady-state is reached only after TTL aging.

The fix-node should declare success when the unit contract (row 2) passes and the steady-state test (row 1) is shown to hold on a synthetic 30-bootstrap-re-run scenario. Live-DB rows 3 and 4 are *observability targets* for `health-observability-root-cause` to surface after the fix ships; they are not gating on this brief's downstream fix node.

---

## 7. Raw-history preservation rule (append-only audit)

The fix must preserve every previously-inserted trace as a queryable row. Concretely:

1. **No `DELETE FROM nodes`.** All decay is `UPDATE nodes SET decayed=1, decay_reason=?` via the existing `soft_delete_node` path (`storage.py:408-425`). Direct deletes remain forbidden under the existing `delete_node` alias for soft-delete (the path-inventory §3 notes both paths flip `decayed=1`; this fix uses only the soft path).
2. **`get_node(old_id)` returns the row after dedup.** The decayed row remains readable via direct store reads (`storage.py:298-300`); only `list_nodes`/`search_content`/`recall` exclude it via the existing `include_decayed=False` default. This is the same regime that `memory_teach` already establishes for corrections (recall context `01KS2P6VFYSTRDD8SMVR0D2A5V`).
3. **The new `supersedes` edge is the audit trail.** A traversal from the new trace's id through `WHERE type='supersedes' AND metadata.kind='duplicate_content'` enumerates every prior identical trace, with `created_at` ordering preserved. This is the "provenance and supersede" mechanism the root goal explicitly endorses.
4. **The `content_fingerprint` column never changes after insert.** It is computed once from `content` at `_insert_node` time; `update_node` continues to refuse `content` changes on traces (`storage.py:344-345`); fingerprint immutability follows.
5. **Backfill of `content_fingerprint` for pre-existing rows is read-from-`content` only.** It does not modify `content`, `scope`, `level`, or any other column. It does not soft-decay any row. (Backfill is mechanically separate from retroactive dedup; see §10 R3.)
6. **No content is rewritten, summarised, or compressed.** Identical content remains identical content in storage; the only change is that older identical traces are marked decayed.

The append-only contract is mechanically defensible: the fix only adds one `INSERT` (the new trace), one `INSERT` per duplicate found (the `supersedes` edge), and one `UPDATE … SET decayed=1, decay_reason=?` per duplicate. None of those mutate prior `content`.

---

## 8. Downstream file ownership

### 8.1 Files the `fix-dedup-noise` node owns (write/edit)

| File | What changes |
| --- | --- |
| `src/living_memory/storage.py` | `SCHEMA_VERSION` bump 2→3 (`storage.py:31`); add `content_fingerprint TEXT` column to the `nodes` CREATE TABLE in `_initialize_schema` (`storage.py:961-1116`); add `idx_nodes_dedup` partial index; add the migration step (compute SHA256 for existing rows with NULL fingerprint, single pass at startup); insert the dedup branch at `_insert_node` (`storage.py:140-213`) — compute fingerprint before INSERT, SELECT duplicates, INSERT new row carrying fingerprint, then create `supersedes` edges and `soft_delete_node` calls inside the same `with self._conn:` block. |
| `tests/test_dedup_supersede.py` (new) | Unit tests for the new contract: same-scope identical content → 1 active + N decayed; cross-scope identical content stays separate; cross-level same content stays separate (trace vs concept); `[file-chunk]` and `[file-summary]` shaped content dedup correctly; `supersedes` edge metadata carries `kind="duplicate_content"`. See §11. |
| `tests/test_storage.py` | Extend `test_storage_schema_invariants` (or equivalent — see path-inventory §11) to assert the new column and index exist, and that the migration completes on a schema-v2 fixture. |
| `tests/test_resources_prompts.py` | Update `test_memory_health_reports_activity_dedup_and_staleness` (`tests/test_resources_prompts.py:122`) if the synthetic-duplicate setup it uses now triggers the new dedup path. Likely needs adjustment so the synthetic duplicates remain `active` for that test (e.g., by using distinct content with the same prefix), since the existing test currently relies on duplicates accumulating. |

### 8.2 Coordination with sibling fix nodes (overlap)

- **`fix-scope-hygiene`** also edits `_insert_node` (`storage.py:140-213`). The two fixes interleave around line 161 (`scope = str(...)`):
  - Line 161 is the canonical scope read; `fix-scope-hygiene` canonicalises `scope` here.
  - The dedup branch in §3.2 runs **after** scope canonicalization, so the dedup query uses the canonical scope.
  - File ownership split: `fix-scope-hygiene` owns lines ~155-165 (scope reading + canonicalization); `fix-dedup-noise` owns the new pre-INSERT fingerprint compute (immediately after) and the post-INSERT supersede/decay block. The integrate stage must serialise the two; sequencing is mechanical (scope first, then dedup), not contested.
  - Test files do not overlap: `test_scope.py` vs. `test_dedup_supersede.py`.
- **`fix-feedback-linkage`** edits `feedback.py` and `storage.py:pending_recall_events` / `_recall_event_matches`. No overlap with the dedup write path.
- **`fix-health-audit`** consumes the existing `memory_health.dedup` metric and may add `feedback_applied` ratio + `scope_leakage` + `never_accessed` ratio + `db_size` + `latency` sections. No write-path overlap. The dedup brief's §6 numeric targets are written to be observable via the existing `memory_health.dedup` surface so `fix-health-audit` does not need to add a new dedup metric.

### 8.3 Files the fix-dedup-noise node does NOT own

- `/home/sfx/p/ae/lm_client.py` — out of scope per §5 layer decision. Read-only structural inspection only.
- `src/living_memory/server.py` — the MCP public API stays compatible (the fix is internal to `_insert_node`). The recall context confirms `memory_remember` MCP signature does not change.
- `src/living_memory/retrieval.py` — the read path is unchanged; the existing `_supersedes_sets` already picks up the new dedup-edge supersedes without modification.
- `src/living_memory/consolidation.py` — `memory_teach` and `memory_consolidate` semantics are unaffected. The `supersedes` edge kind it writes (`"by", "correction"`) is disjoint from the dedup kind (`"duplicate_content"`).
- `src/living_memory/decay.py` — `soft_delete_superseded` already idempotently handles already-soft-decayed targets (the JOIN `target.decayed = 0` excludes them). No code change.

---

## 9. Latency micro-check guidance (per-fix benchmark)

To honor the latency-safe constraint shared by `recall-speed-usefulness-root-cause` and the parent root goal ("any fix must avoid making recall meaningfully slower"; "hot recall median 65-70 ms through the public path, 44-47 ms local"), the fix node records before/after timings for both `memory_remember` inserts and hot `memory_recall`.

### 9.1 Setup (deterministic local fixture)

Use the same read-only snapshot fixture pattern as `artifacts/baseline.md` (open `/tmp/lm-baseline-replay.sqlite3` under `mode=ro` so the benchmark cannot mutate the live state). For the dedup-specific write benchmark, use a fresh in-memory or per-test-tempfile `MemoryStore` so the fixture stays unchanged.

### 9.2 Write-path micro-benchmark

```python
# Pseudocode — final shape owned by fix node
import time, statistics
from pathlib import Path
from living_memory.storage import MemoryStore

with MemoryStore(Path(tmp_path / "bench.sqlite3")) as store:
    contents = [f"unique trace {i}" for i in range(1000)]
    # Unique inserts — baseline
    t = time.perf_counter()
    for c in contents:
        store.append_trace(c, {"scope": "project:bench"})
    baseline_unique_ms = (time.perf_counter() - t) * 1000

    # Duplicate inserts — exercises the dedup branch
    t = time.perf_counter()
    for c in contents:
        store.append_trace(c, {"scope": "project:bench"})
    dup_dedup_ms = (time.perf_counter() - t) * 1000

    print({"unique_ms_mean": baseline_unique_ms / 1000,
           "dup_ms_mean": dup_dedup_ms / 1000})
```

Target:

- Unique insert mean ≤ 1.0 ms per call (the dedup branch's SELECT must miss; cost is just the indexed SELECT + INSERT).
- Duplicate insert mean ≤ 2.0 ms per call (SELECT hits, INSERT, edge INSERT, soft-delete UPDATE).
- Regression budget: unique insert mean must be within 1.20× the same benchmark run against `master` (the pre-fix baseline).

### 9.3 Read-path micro-benchmark

Use the exact in-process recall benchmark from `artifacts/baseline.md` §"Hot Recall Latency" (`memory_recall(store, Q, scope='project:lm', ambient_context={…}, max_results=1, depth=1, log_access=False, log_event=False)` × 5 measured runs after one warm-up). The fix node measures this against the same `/tmp/lm-baseline-replay.sqlite3` (which is byte-identical pre- and post-fix because the read path code is unchanged) and against a fresh tempfile DB with 1,000 traces inserted through the new dedup path.

Target:

- In-process `memory_recall` median against the live snapshot: ≤ 100 ms (baseline measured 91.869 ms median in `baseline.md`).
- In-process `memory_recall` median against the synthetic 1,000-trace DB: ≤ 50 ms (smaller DB, faster scans).
- Regression budget: ≤ 1.05× the same measurement on `master`.

### 9.4 Reporting

Capture both write and read benchmarks as JSON sidecars `artifacts/latency_before.json` and `artifacts/latency_after.json` (the parent SPEC P8 / `recall-speed-usefulness-root-cause` already names these). The fix node compares the two and reports the multiplier in its result.md.

### 9.5 Coverage note

The recall-speed brief is the canonical owner of latency thresholds; this section is the dedup-specific extension. If the recall-speed brief sets a stricter ceiling than the numbers above, the stricter one wins.

---

## 10. Risks and open follow-ups

### 10.1 In-scope risks for the fix node to manage

**R1. SCHEMA_VERSION migration safety on the production-sized DB.**
The live DB is ~96 MB with 5,300 nodes. SHA256 throughput in CPython is >300 MB/s; backfilling fingerprints is bound by SQLite write rate (~5,000 UPDATEs/s with `synchronous=NORMAL`). Expected total backfill time: ~5 s synchronous at first server startup after the upgrade. The fix node should:
- Run the backfill inside an explicit transaction so an interrupt leaves SCHEMA_VERSION = 2 (atomic).
- Log the backfill duration to stderr so operators see it.
- Skip the backfill on `nodes` rows with NULL `content` (defensive; should not occur).

**R2. Cross-scope dedup must NOT fire.**
The dedup query restricts to `WHERE scope = ?`. A test must assert this explicitly (`test_dedup_supersede.test_cross_scope_identical_content_is_not_deduped`).

**R3. Cross-level dedup must NOT fire.**
A `trace` with the same content as an existing `concept` must remain a trace and the concept must remain a concept. The dedup query restricts to `WHERE level = 'trace'`. A test must assert this.

**R4. Concurrent inserts of the same content.**
Two simultaneous `memory_remember` calls with identical content might both see "no existing" in their respective `SELECT`s and both INSERT, then both create supersedes edges that race. The single global `runtime_lock` (`server.py:78`) already serialises tool calls, so this race cannot occur on the MCP path. Direct-store usage in tests or in-process integration must rely on the per-`MemoryStore` connection's transaction (the `with self._conn:` block in `_insert_node`). Both inserts will serialize on the SQLite-level write lock; the second sees the first via the indexed SELECT and dedupes. Document this in the fix-node's design notes.

**R5. The `feedback_weighted_score` 1.2× superseding boost compounds favorably but the absolute boost is bounded.**
Per `feedback.py:237-252` and recall context, the boost is applied flat if the node id is in `superseding_ids`, not per edge. A trace that supersedes 50 duplicates gets 1.2× — same as one that supersedes 1. Safe.

### 10.2 Out-of-scope follow-ups (not part of this brief's downstream fix)

**FU1. Backfill of existing duplicates on the live DB.**
The fix only prevents new accumulation. The existing 625 duplicate excess in `project:ae` stays active until TTL ages it out (~180 days from each trace's `last_accessed`). An operator may run a one-shot retroactive sweep via a `scripts/dedup_backfill.py` helper or a `POST /admin/dedup-backfill` endpoint. **This is out of scope for the fix-dedup-noise node.** It should be either:
- A separate AE goal scheduled by the operator (Layer: LM ops), OR
- Folded into `analysis-synthesis` as a deferred follow-up sub-task.

The brief does not schedule this; it identifies it as known and bounded.

**FU2. Path/kind-based supersede for changed files.**
When `[file-summary]` for the same `(path, language)` is emitted after a file has changed, the new content has a different SHA256 (because the file body changed). The new trace does NOT supersede the older one (different fingerprint). The older now-stale trace remains active until TTL. This is a *distinct* problem from exact-content dedup. A path/kind supersede would require LM to parse the `[file-summary]` JSON header, leaking AE format into the LM contract; cleaner is to have AE emit a path-based forget or supersede call. **Out of scope for this brief.** If `analysis-synthesis` deems it material, it should be a follow-up goal owned by AE, not LM.

**FU3. AE caller round-trip optimization.**
After the LM-side dedup ships, AE bootstrap still emits ~3,000 `memory_remember` round-trips per refresh; ~15 s of that is duplicate transport overhead. AE could add a pre-check that recalls a single content-digest manifest from LM and skips files already present. This is a caller-side optimization (not correctness; correctness is now handled by LM-side dedup). **Out of scope.**

**FU4. Health surface enrichment.**
`fix-health-audit` is the owner of any new health metrics (e.g., a `dedup_supersedes_in_window` count). This brief does not name new metrics; it relies on `memory_health.dedup.duplicate_density` (which already exists at `resources.py:377-387`) to measure success.

---

## 11. Expected test surface (for downstream fix node)

The fix node should expand the test suite as follows. These are the minimum contract tests; the fix node may add more.

### 11.1 `tests/test_dedup_supersede.py` (new)

| Test | Assertion |
| --- | --- |
| `test_same_scope_identical_content_supersedes_older_trace` | After two `append_trace` calls with identical `content` in `project:demo`, there is exactly one active trace; the older is decayed with `decay_reason="duplicate_content"`. |
| `test_supersedes_connection_has_dedup_metadata` | The created `supersedes` connection has `metadata["kind"] == "duplicate_content"` and `metadata["fingerprint"]` equal to `sha256(content).hexdigest()`. |
| `test_cross_scope_identical_content_is_not_deduped` | Two `append_trace` calls with identical content in different scopes produce two active traces. |
| `test_cross_level_identical_content_is_not_deduped` | A trace and a concept with identical content remain both active and distinct. |
| `test_dedup_handles_file_chunk_shaped_content` | Insert `[file-chunk] {...sha256:...}\n```python\n...```` twice; assert one active, one decayed, edge present. |
| `test_dedup_handles_file_summary_shaped_content` | Insert `[file-summary] {...}` twice; assert one active, one decayed, edge present. |
| `test_dedup_handles_project_overview_shaped_content` | Insert `[project-overview] {...}` with byte-identical JSON twice; assert one active. |
| `test_n_repeated_inserts_yield_one_active_n_minus_one_decayed` | Loop 10× identical insert; assert 1 active + 9 decayed + 9 supersedes edges. |
| `test_dedup_preserves_original_content_in_store` | `get_node(old_id)` returns the row with original `content` and `decayed=1` after dedup; `content` is unchanged. |
| `test_dedup_excludes_already_decayed_traces` | Decay a trace manually via `soft_delete_node`, then insert identical content; new trace is active; the decayed old one is not touched (no double-decay, no new edge to a decayed row). |
| `test_memory_recall_returns_only_active_traces_after_dedup` | After 5 identical inserts, recall returns at most 1 result for that content. |

### 11.2 `tests/test_storage.py` (extend)

| Test | Assertion |
| --- | --- |
| `test_schema_version_is_three_after_initialize` | `SCHEMA_VERSION` constant is 3; `metadata.schema_version` is 3 after `MemoryStore` init. |
| `test_content_fingerprint_column_exists` | `PRAGMA table_info(nodes)` includes `content_fingerprint TEXT`. |
| `test_idx_nodes_dedup_index_exists` | `PRAGMA index_list(nodes)` includes `idx_nodes_dedup`. |
| `test_schema_v2_database_migrates_to_v3_with_backfill` | Load a fixture v2 DB with 50 rows and NULL `content_fingerprint`; open via `MemoryStore`; assert all rows now have non-NULL `content_fingerprint` equal to `sha256(content).hexdigest()`; assert `schema_version` is 3. |

### 11.3 `tests/test_resources_prompts.py` (adjust)

| Test | Change |
| --- | --- |
| `test_memory_health_reports_activity_dedup_and_staleness` (`:122`) | The current test inserts two identical-content traces and expects `duplicate_density > 0`. After the fix, those two would collapse to one active. Adjust the test to insert two *different* contents that share a token prefix (the existing dedup metric measures distinct *content*, not similarity). Confirm the test still asserts `duplicate_density > 0`. |

### 11.4 Integration / acceptance tests

`tests/test_acceptance_contract.py` already enforces a cold-start recall budget (`tests/test_acceptance_contract.py:54`). Confirm this still passes after the fix. No new acceptance test is required at this level (the dedup contract is enforced by the unit tests above).

---

## 12. Summary

| Field | Value |
| --- | --- |
| **Root cause** | LM `_insert_node` performs unconditional `INSERT` for traces; no content fingerprint or identity check anywhere in the write path. The dominant high-volume caller (`bootstrap-project`) is contract-required to be re-runnable and emits byte-deterministic content per file. Both halves together: identical content accumulates one row per re-run, indefinitely (until 180-day TTL). |
| **Chosen fix** | LM-side write-path content fingerprint with immediate `supersedes`-edge + soft-decay of older identical traces in the same `(level, scope)`. Schema bump 2→3 with one-shot backfill of `content_fingerprint`. Reuse the existing `supersedes` edge type with `metadata.kind="duplicate_content"`. |
| **Layer decision** | LM-internal. AE caller unchanged. |
| **Numeric target** | Unit-level: identical-content remembers in the same scope yield exactly one active trace + N decayed via `supersedes`. Live `project:ae` `duplicate_density` ≤ 5% post-backfill (operator-triggered, out of fix-node scope); ≤ 1% steady-state on a new scope. Latency: unique insert ≤ 1.0 ms mean, duplicate insert ≤ 2.0 ms mean; hot recall regression ≤ 1.05×. |
| **Raw-history preservation rule** | All decay is soft-delete; `content` is never mutated; `content_fingerprint` is immutable post-insert; older identical traces remain queryable via `get_node`; `supersedes` edges provide the audit trail. |
| **Primary files owned by `fix-dedup-noise`** | `src/living_memory/storage.py` (column + index + migration + dedup branch), `tests/test_dedup_supersede.py` (new), `tests/test_storage.py` (extend), `tests/test_resources_prompts.py` (adjust one test). |
| **Coordination point** | `_insert_node` (`storage.py:140-213`) is co-touched with `fix-scope-hygiene` at line ~161; sequencing is scope-canonicalization first, then fingerprint dedup; no contested logic. |
| **Out-of-scope follow-ups** | (a) one-shot retroactive backfill of the existing 625 duplicate excess in `project:ae`; (b) path-based supersede for changed files; (c) AE caller round-trip optimization; (d) new health metrics. All deferred to `analysis-synthesis` or follow-up goals. |
