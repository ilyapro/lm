# ANALYSIS — `memory-quality-root-fixes` Discovery Synthesis

Generated for `memory-quality-root-fixes/discover-baseline-and-root-causes/analysis-synthesis` on 2026-05-22.

Integrates the five per-problem briefs into one downstream-ready document so the sibling fix nodes (`fix-scope-hygiene`, `fix-dedup-noise`, `fix-feedback-linkage`, `fix-health-audit`) can execute without re-discovering fundamentals.

Inputs consumed:

- `artifacts/baseline.md` — frozen audit metrics + reproduction commands.
- `artifacts/discovery/scope-hygiene.md` — bare-scope leakage root cause.
- `artifacts/discovery/dedup-noise.md` — duplicate / file-chunk write-path root cause.
- `artifacts/discovery/feedback-linkage.md` — `feedback_applied` ratio root cause.
- `artifacts/discovery/health-observability.md` — additive `memory_health` extension shape.
- `artifacts/discovery/recall-speed-usefulness.md` — retrieval latency-safe constraints + benchmark spec.

The discovery siblings additionally relied on `artifacts/discovery/path-inventory.md` and `artifacts/discovery/lm-recall-context.md`; this synthesis treats those as already-cited reference material rather than restating them.

---

## 0. Scope contract

**Intent.** This artifact integrates the five per-problem root-cause briefs into a single downstream-ready file owned by `artifacts/ANALYSIS.md`. The intent is bounded to (a) restating each brief's root cause with file:line citations, (b) recording the chosen simplest central fix and rejected alternatives for each, (c) recording the AE-vs-LM layer decision for each, (d) recording downstream file ownership and the integration ordering between fix nodes, (e) recording numeric improvement targets, fixture/live-DB policy, and per-fix latency micro-check guidance. The intent excludes any modification of production code, any database write, any contact with a running LM server, and any cross-repository change.

**Boundary tokens (goal-scope contract markers).** This artifact declares its boundary contract via these literal tokens: `intent: read-only synthesis only`; `replay_evidence_path: local, this worktree (open the cited LM source paths at the cited line ranges; open the five sibling briefs under artifacts/discovery/ at the cited section anchors)`; `rollback_artifact: local, this worktree (git rm artifacts/ANALYSIS.md)`; `redaction_boundary: no secrets, credentials, tokens, env values, configuration values, customer data, or PII captured`.

**Local replay evidence.** Every citation in this file reproduces by opening the cited LM source path (under `src/living_memory/` or `tests/`) at the cited line range in this worktree's `HEAD` (commit `08626ce`). Numeric figures reproduce from `artifacts/baseline.md` and `artifacts/discovery/baseline-raw.metrics.json` against the read-only snapshot `/tmp/lm-baseline-replay.sqlite3` (MD5 `58fc12fd47c5f71e5b8867ee71590668`). No network access, no remote endpoint, and no non-local execution is required. local replay_evidence is fully self-contained inside this worktree.

**Local rollback artifacts.** Reverting this child's work is `git rm artifacts/ANALYSIS.md` in this worktree; no other worktree, no other repository, no shared service, and no non-local system is affected. local rollback_artifact = this single Markdown file.

**Redaction.** Every citation is a function name, file path, line range, schema column, MCP tool name, scope identifier already present in `artifacts/baseline.md`, or metric ratio derived from public-shaped aggregates. No secrets, credentials, tokens, env values, configuration values, customer data, or PII are read into this artifact. The redaction_boundary is enforced by the input contract: this synthesis aggregates already-redacted sibling briefs and adds no new field that crosses the boundary.

**No external action plan.** The synthesis names which LM-internal layer owns each fix and which file ranges each downstream fix node should touch. AE-side mentions are structural (where a caller currently emits a particular shape) and never prescriptive (AE is not asked to change). Any cross-repository follow-up surfaced by an individual brief is recorded in §11 as deliberately deferred, not scheduled.

**No production code change.** This child writes only `artifacts/ANALYSIS.md`; parent P10 and the `discovery-artifact-verify` sibling are the binding gates for that invariant.

---

## 1. Executive synthesis

### 1.1 Problem map

| # | Area | Root cause (one line) | Layer | Fix shape | Brief |
| --- | --- | --- | --- | --- | --- |
| 1 | Scope hygiene | Write path bypasses `normalize_scope` at `src/living_memory/storage.py:161`; read path always normalises. | LM-internal | Canonicalise `scope` inside `_insert_node` before INSERT. | `scope-hygiene.md` |
| 2 | Duplicate / file-chunk noise | `_insert_node` performs unconditional INSERT with no content-fingerprint check; `bootstrap-project` re-emits byte-deterministic traces per run. | LM-internal | Add `content_fingerprint` column + index; supersede + soft-decay older identical traces in same `(level, scope)`. | `dedup-noise.md` |
| 3 | Feedback linkage | `apply_pending_recall_feedback` defaults `limit=1` (`src/living_memory/feedback.py:135-140`); matcher in `_recall_event_matches` is scope-only unless both sides set `session_id` (`src/living_memory/storage.py:1321-1334`); AE recall side propagates no metadata. | LM-internal (primary) | Context-aware matcher (task/session strong, agent weak); raise limit to a small cap for strong matches; tighten cross-scope fallback. | `feedback-linkage.md` |
| 4 | Health / observability | `memory_health` (`src/living_memory/resources.py:323-459`) reports activity/dedup/staleness only; feedback ratio, leakage, never-accessed, DB size, instructions, and latency baseline live elsewhere or nowhere. | LM-internal | Extend `memory_health` with additive SQL-aggregate sections; opt-in latency probe; keep `memory_status` and HTTP `/health` unchanged. | `health-observability.md` |
| 5 | Recall speed / usefulness | Graph traversal at `src/living_memory/retrieval.py:401-459` issues one `list_connections` per candidate and dominates hot-path latency; BM25/vector scans scale with active rows including duplicates. | Constraint, not a standalone fix | Suppress noisy candidates before retrieval via fixes 1+2; never widen retrieval to alias scopes; enforce per-fix paired benchmarks. | `recall-speed-usefulness.md` |

### 1.2 The single integrating insight

Four of the five problem areas concentrate on **one chokepoint**: `src/living_memory/storage.py:_insert_node` (`storage.py:140-213`) and the recall-feedback write path around `storage.py:1321-1334` / `feedback.py:135-140`. The fifth (health) is an observability-only extension of `resources.py:memory_health`. The retrieval module (`retrieval.py:108-174`) is **only a constraint** — it bounds latency-acceptable shapes for the other four fixes; it does not get an independent code change in this round.

This consolidation is significant for sequencing: `fix-scope-hygiene` and `fix-dedup-noise` both co-touch `_insert_node` and must integrate in a defined order (scope canonicalisation first, then fingerprint dedup). `fix-feedback-linkage` touches a disjoint surface (`feedback.py`, `storage._recall_event_matches`, `server.py` instructions block). `fix-health-audit` touches a disjoint surface (`resources.py` aggregations + new MCP wrapper signature). See §8 for the integration DAG.

### 1.3 Live-baseline anchor

Every numeric target in §10 and every regression budget in §9 is anchored to the frozen snapshot at `/tmp/lm-baseline-replay.sqlite3` (MD5 `58fc12fd47c5f71e5b8867ee71590668`) and the metric definitions in `artifacts/baseline.md`. The baseline figures referenced throughout this synthesis are:

- 5,029 active traces; 4,377 distinct contents; duplicate density 0.1296 (`baseline.md:93-109`).
- 5,114 recall events; 1,060 feedback applied; ratio 0.2073 (`baseline.md:121-144`).
- 3,681 / 5,029 active traces never accessed (0.7320); 3,710 / 5,128 active nodes never accessed (0.7235) (`baseline.md:146-172`).
- 54 candidate scope-leakage scopes; 111 active candidate traces; 111 / 111 never accessed (`baseline.md:174-213`).
- Snapshot main-file size 100,339,712 bytes / 95.69 MiB (`baseline.md:40-43`).
- Hot in-process recall median 91.869 ms over 5 samples, `project:lm`, `max_results=1`, `depth=1` (`baseline.md:215-271`). Prior live hot baseline from Living Memory recall context: 65–70 ms (deeper in-process number reflects the cold-helper path; see §9).

---

## 2. Cross-cutting decisions

These decisions apply to every downstream fix node and are extracted into one place so the four fix nodes do not re-derive them.

### 2.1 AE-vs-LM layer (per problem)

| Problem | Layer chosen | Reason in one line | AE caller change required? |
| --- | --- | --- | --- |
| Scope hygiene | LM-internal | Single chokepoint at `_insert_node` covers every present and future producer of node rows. | No |
| Dedup / file-chunk noise | LM-internal | "Identical content within a scope should not appear twice in active recall" is a memory-store invariant, not a caller invariant. | No |
| Feedback linkage | LM-internal (primary) | Server owns matcher correctness and false-positive prevention; AE recall-side metadata propagation is a useful follow-up but does not gate this fix. | Optional follow-up only (see §11) |
| Health / observability | LM-internal | Storage and helper functions already retain every needed field; the gap is omitted aggregations in `memory_health`. | No |
| Recall speed / usefulness | Constraint | No independent fix in this round; latency is protected by the per-fix micro-check in §9 and by suppressing noisy candidates via fixes 1+2. | No |

The unifying rationale is the root-goal principle "Prefer one central simple rule over project-specific patches". Caller-side enforcement (AE-only) would leave Online, Octopus, and future producers free to repeat the bug; LM-internal enforcement closes every entry point at once. Where a caller-side optimisation would compound benefit (e.g. AE recall-side ambient metadata enrichment for fix #3, or AE pre-bootstrap recall-before-emit for fix #2), it is recorded in §11 as a deferred follow-up the integration owner can sequence independently.

### 2.2 Fixture vs live-DB policy

Authoritative rule for every fix node:

- **Unit tests** use a fresh tmp-path SQLite store per pytest function (`with MemoryStore(tmp_path / "...sqlite3") as store: ...`). The pattern is already established in `tests/test_scope.py:9, :32, :66, :86`, `tests/test_storage.py`, and `tests/test_recall_feedback_loop.py`. None touches the live DB.
- **Read-path latency micro-checks** open the frozen snapshot `/tmp/lm-baseline-replay.sqlite3` under `mode=ro` (snapshot URI `file:/tmp/lm-baseline-replay.sqlite3?mode=ro`). The snapshot's MD5 is verified before and after every measurement. Because every fix node runs in its own worktree but shares this snapshot file, no fix node may mutate it; read-only URI mechanically enforces the invariant.
- **Write-enabled micro-checks** copy the snapshot to a temp file (`tempfile.TemporaryDirectory` + `shutil.copy2`) and operate on the copy. The live DB at `/home/sfx/.local/share/living-memory/global.sqlite3` is never opened with any write capability by any fix node.
- **Integration / live-DB observation** (e.g. `health-audit` reporting the live `duplicate_density` after the fix ships) is owned by the `verify-and-record` parent-tree node, not by any individual fix node. Fix nodes do not write to the live DB during their own iteration.

This rule defuses parent `_critique` observation 7 ("shared live DB state across parallel fix-* nodes"): no fix node writes to the live DB; the snapshot is read-only; therefore audit-metric snapshots taken before/after each fix would all read the same frozen state regardless of whether earlier fixes have landed.

### 2.3 Append-only and supersedes invariants

The four invariants every fix node must preserve:

1. **No `DELETE FROM nodes`.** All decay is `UPDATE nodes SET decayed=1, decay_reason=?` via `soft_delete_node` (`storage.py:408-425`). Direct deletes remain forbidden.
2. **`content` is immutable on traces.** `update_node` raises if `content` changes on `level="trace"` (`storage.py:344-345`). The new `content_fingerprint` column inherits this immutability (computed once at insert).
3. **`supersedes` is the audit edge.** Both `memory_teach` corrections (`consolidation.py:319-386`) and the new dedup branch reuse the existing `supersedes` connection type, disambiguated via `metadata.kind` (`"by"`/`"correction"` vs. `"duplicate_content"`).
4. **Public MCP API is compatible unless documented.** `memory_remember`, `memory_recall`, `memory_teach`, `memory_consolidate`, `memory_status`, and existing `memory_health` signatures stay back-compatible. Health fix may add an **optional** `latency_samples` / `instructions_text` argument with a default that preserves current behaviour.

### 2.4 Gate considerations (downstream fix nodes)

Every fix node's owned artifacts will pass through the same `goal_scope_contract.sh` gate that screens the discovery artifacts. The four literal snake_case tokens (`intent`, `replay_evidence`, `rollback_artifact`, `redaction_boundary`) and the "no same-line keyword pair" rule are documented in trace `01KS7C74T7D01299QCDWZMRD4K` (project:lm). Fix nodes should mirror the §0 boundary block from the sibling discovery briefs and rerun `bash -c 'source /home/sfx/p/ae/goal_scope_contract.sh; goal_scope_contract_check_changed_files <path>'` against each new artifact before submitting for merge.

---

## 3. Problem 1 — Scope hygiene (bare `rise/*`, `breakthrough/*`, `ocpa-generative-action-substrate-v1/*` leakage)

### 3.1 Root cause

There is exactly one causal step: **the LM write path does not call `normalize_scope` before persisting the `scope` column.**

- Read path: `memory_recall` runs `ScopeResolver.resolve` (`src/living_memory/scope.py:54-69`) which always passes the requested or ambient scope through `normalize_scope` (`scope.py:97-113`). `normalize_scope` rewrites bare strings (no `:`) to `project:<value>` at `scope.py:113`.
- Write path: `MemoryStore._insert_node` reads `scope` directly from `context_data["scope"] or self.config.default_scope` at `src/living_memory/storage.py:161` and persists it **as-is** at `storage.py:195`. No call to `normalize_scope` between dictionary read and SQL `INSERT`.

Octopus tree-decomposition goal-node identifiers (`rise/_critique`, `breakthrough/gemini-flash-contract`, `ocpa-generative-action-substrate-v1/...`) are passed by callers as `context["scope"]`. Because the write side bypasses canonicalisation, the bare strings persist; because the read side always canonicalises, the same agent recalling under `project:rise/_critique` finds nothing.

Ruled-out alternatives (from `scope-hygiene.md` §2.3):

- `_ambient_scope` cwd derivation (`scope.py:185-204`) cannot be the cause — `memory_remember` does not accept `ambient_context`; only `memory_recall` does (`server.py:556-581`). Cwd-derived `_node_exec_*` scopes therefore appear only in `recall_events.requested_scope`, never as a stored trace `scope`.
- `infer_project_scope` (`scope.py:122-143`) is a read-only path; cannot affect what gets written.
- MCP `memory_remember` framing (`server.py:475-515`) is a thin pass-through; no transformation.
- `record_recall_event` (`storage.py:601-648`) is symmetric with the read path; not a contributor.

### 3.2 Chosen simplest central fix

Add `scope = normalize_scope(...)` inside `MemoryStore._insert_node` so every node insertion canonicalises the `scope` column before SQL.

Mechanical shape (the fix node executes the change; this brief does not):

- Import `from living_memory.scope import normalize_scope` near `storage.py:16`. `scope.py` imports only stdlib + `pathlib.Path`; no circular-import risk.
- Replace `storage.py:161`:
  ```python
  scope = str(context_data.get("scope") or self.config.default_scope)
  ```
  with:
  ```python
  scope = normalize_scope(str(context_data.get("scope") or self.config.default_scope))
  ```
- Persist the canonical scope back into `context_data["scope"]` so the JSON `context` column matches the SQL `scope` column.

Why this shape:

1. `_insert_node` is the only producer of new rows in the `nodes` table. One line covers every level (trace, concept, schema) and every internal caller (`append_trace`, `append_trace_with_rejected_alternatives`, concept creation in `consolidation.py`, schema materialisation).
2. `normalize_scope` is idempotent for already-canonical inputs (`"project:lm"` → `"project:lm"`, `"global"` → `"global"`, `"session:abc"` → `"session:abc"`).
3. Public MCP signature is unchanged.
4. Append-only invariant unaffected (rewrite happens before INSERT; existing rows untouched).
5. No new column, no new table, no schema migration; `SCHEMA_VERSION = 2` (`storage.py:31`) does not change for this fix alone.
6. Constant-time per write; no measurable read-path impact.

### 3.3 Rejected alternatives

| # | Alternative | Reason |
| --- | --- | --- |
| R1 | Rewrite every AE caller to pass `project:<goal-id>` | Scatters enforcement; future internal LM callers can re-introduce the bug. |
| R2 | Scope-rewrite table (`rise → project:octopus`, `breakthrough → project:octopus`, ...) | Embeds Octopus-specific knowledge; collapses goal-tree path signal; needs maintenance per new project. |
| R3 | Enforce at MCP server boundary (`server.py:476-499`) | Misses non-MCP producers: `consolidation._merge_cluster_into_concept`, `_materialize_procedural_schemas`, cross-scope promotion, direct `MemoryStore.append_trace` users. |
| R4 | Worktree-aware cwd canonicalisation (combined with this fix) | Orthogonal concern, affects only the read-side `requested_scope` derivation; deferred to §11. |
| R5 | One-time SQL migration of historical bare-scope rows | Migration is a separate policy choice (does the team want `rise/_critique` → `project:rise/_critique`, `→ project:octopus`, `→ project:octopus/rise/_critique`, or no migration?). Deferred to §11. |
| R6 | Reject any non-canonical scope at write time (no rewrite, only error) | Louder than necessary; existing valid callers that pass bare strings would break entirely. Canonicalisation is a strict refinement of current semantics. |
| R7 | Periodic background re-scoping of orphan traces | Mutates rows post-hoc (violates append-only intent); O(N) per sweep with no upside over the O(1) per-write rule. |

### 3.4 AE-vs-LM layer

**LM-internal.** Fix lives entirely in `src/living_memory/storage.py` and (transitively) imports `normalize_scope` from `src/living_memory/scope.py`. AE callers are unchanged. A future AE refactor, new MCP client, `bootstrap-project` rerun, or direct `MemoryStore.append_trace` user gets the rule for free.

### 3.5 Downstream file ownership (owned by `fix-scope-hygiene`)

| Component | File / range | Action shape |
| --- | --- | --- |
| Write canonicalisation guard | `src/living_memory/storage.py:16` (import) and `storage.py:161-164` (scope read + rewrite) | Import + normalise + sync `context_data["scope"]`. |
| Test: bare scope is canonicalised | new test in `tests/test_scope.py` (after `:103`) or `tests/test_storage.py` (after `:180`) | Append-trace with `context={"scope": "rise/_critique"}`; assert stored `scope` is `"project:rise/_critique"`. |
| Test: idempotency for canonical input | new test in `tests/test_scope.py` | Append-trace with `"project:lm"`; assert stored is `"project:lm"`. |
| Test: alias rewrite | new test in `tests/test_scope.py` | `"workspace:lm"` → stored `"project:lm"`; `"repo:lm"` → `"project:lm"`. |
| Test: unknown prefix raises | new test in `tests/test_storage.py` | `"foo:bar"` raises `ValueError("unsupported scope prefix: foo")`. |
| Test: empty scope falls back | new test in `tests/test_storage.py` | Empty `context["scope"]`; default-scope behaviour preserved. |

### 3.6 Numeric target

- **Bare-scope failure mode**: new bare-scope rows after the fix = 0 in any audit window. The pre-existing 111 candidate traces and 36 zero-recall scopes do not retroactively shrink (a migration is deferred per §11) — they age out via 180-day TTL.
- **Error-prefix failure mode**: `memory_remember` raises `ValueError("unsupported scope prefix: <prefix>")` for unknown prefixes. Recommended posture: fail loud, because silent fallbacks are how the original asymmetry survived.

### 3.7 Fixture / live-DB policy

Per §2.2: unit tests use per-test tmp-path stores; latency micro-check uses the frozen snapshot read-only.

### 3.8 Latency micro-check

The fix touches only the write path; read latency is unchanged. The fix node runs:

- The canonical hot-service matrix from `recall-speed-usefulness.md` §"Deterministic micro-benchmark policy" against the frozen snapshot before and after the change.
- Assert: median delta ≤ 5 ms and after/before ratio ≤ 1.05 for `project:lm`, `project:ae`, `project:octopus`.
- A focused fixture comparison: explicit `scope="project:octopus"` versus `ambient_context={"cwd": "/root/p/octopus/.worktrees/_node_exec_rise"}` after canonicalisation. Canonicalised ambient path should resolve to the same plan and not be more than 5 ms or 5% slower in a 7-sample median.

---

## 4. Problem 2 — Duplicate / file-chunk noise

### 4.1 Root cause

Two interacting causes:

**(a) LM write path is unconditional `INSERT` for traces.** `MemoryStore.append_trace` (`storage.py:215-223`) → `MemoryStore.create_node` (`storage.py:115-138`) → `MemoryStore._insert_node` (`storage.py:140-213`) executes one `INSERT INTO nodes (...) VALUES (...)` at lines 180-212 with no `SELECT … WHERE content = ?`, no `ON CONFLICT`, no fingerprint, no per-scope identity check. Every `memory_remember` appends a fresh row even when an active trace with byte-identical `content` already exists in the same scope. The append-only invariant at `storage.py:344-345` constrains in-place rewrites, not repeat inserts.

**(b) `cmd_bootstrap_project` is the dominant high-volume caller and is contract-required to be re-runnable.** `/home/sfx/p/ae/lm_client.py:321-556` walks the project tree and emits one `[file-summary]` + N `[file-chunk]` traces per indexable file, plus `[project-overview]`, `[project-map]`, and `[project-manifest]` per run. The header JSON for every per-file emit is `json.dumps(..., sort_keys=True)` and the chunk body is the raw fenced source text; the per-trace `content` string is byte-deterministic given the same file tree. Re-runs are normal operator behaviour.

Live duplicate signal (from `baseline.md:117-119`):

| scope | active traces | duplicate excess | density |
| --- | ---: | ---: | ---: |
| `project:ae` | 2647 | 625 | 0.236116 |
| `project:online` | 223 | 21 | 0.094170 |
| `project:lm` | 316 | 3 | 0.009494 |
| `project:octopus` | 1544 | 3 | 0.001943 |
| `global` | 158 | 0 | 0.000000 |

Duplicate density concentrates exactly in scopes where `bootstrap-project` was re-run multiple times.

### 4.2 Chosen simplest central fix

Add a content fingerprint to the LM trace write path, and at insert time supersede every older active trace in the same `(level, scope)` that has the same content. Soft-decay the older duplicates immediately; record the supersede link for graph visibility.

Mechanical parts:

1. **Schema change.** Add `content_fingerprint TEXT` to `nodes` (`storage.py:_initialize_schema` at `:961-1116`); add partial index `idx_nodes_dedup ON nodes(level, scope, content_fingerprint) WHERE decayed = 0 AND content_fingerprint IS NOT NULL`. Bump `SCHEMA_VERSION` 2 → 3 (`storage.py:31`). One-shot synchronous backfill of `content_fingerprint` for existing rows at first server startup post-upgrade.
2. **Dedup at `_insert_node` for `level="trace"` only.** Inside the same `with self._conn:` transaction: compute `fingerprint = sha256(content.encode("utf-8")).hexdigest()`; `SELECT id FROM nodes WHERE level='trace' AND scope=? AND content_fingerprint=? AND decayed=0`; INSERT the new node with `content_fingerprint`; for each matching old id, `_insert_connection(new_id, old_id, "supersedes", metadata={"kind": "duplicate_content", "fingerprint": ...})` then `soft_delete_node(old_id, "duplicate_content")`.
3. **Reuse `supersedes` semantics** disambiguated via `metadata.kind` (corrections vs. dedup). Existing retrieval consumers (`retrieval._supersedes_sets()` at `retrieval.py:473-479`, decay's `soft_delete_superseded`) do not require code change to absorb the new edges.
4. **No new health surface** — the existing `memory_health.dedup` (`resources.py:377-387`) already measures the metric used to verify success.

### 4.3 Rejected alternatives

| # | Alternative | Reason |
| --- | --- | --- |
| R1 | Block the duplicate INSERT entirely (return existing id) | Violates append-only; loses provenance of which run produced which trace. |
| R2 | Recall-before-emit dedup at AE caller side | Project-specific patch; ~3,000 recalls × ~65 ms = ~3 minutes added per bootstrap vs ~15 s for the LM-side approach. Future Online/Octopus callers re-derive the same pattern. |
| R3 | Rank-time dedup (collapse identical-content results in `rank_candidates`) | Does not reduce DB size; adds rank-time cost on every recall; conflicts with "any fix must avoid making recall meaningfully slower". |
| R4 | Aggressive shortened TTL for bootstrap traces | Punishes legitimate non-duplicate older traces; solves staleness, not identity. |
| R5 | New separate `node_fingerprints` table | Heavier migration; extra JOIN per insert; no benefit. |
| R6 | Background daemon for periodic dedup | Forbidden by root principle "no shadow store, no heavy background daemon unless proven necessary". Write-time dedup is sufficient. |
| R7 | Treat duplicates as `memory_teach`-style corrections | Semantically wrong: a duplicate is not a *correction*; lowering original's `usefulness_score` would distort feedback weighting. |
| R8 | Path/kind-based supersede for changed files | Solves a different problem (stale-content), requires LM to parse AE-specific JSON header; deferred to §11. |
| R9 | Hash by `content` directly with a TEXT index | Indexing full TEXT is wasteful (~10 MB index pages for 5,000 1–3 KB rows); fingerprint column is ~64 B/row, indexable to sub-ms. |
| R10 | Consolidation-driven dedup (extend `_merge_cluster_into_concept`) | Semantically wrong: consolidation operates on *similar* content via Jaccard + embedding; identical content is a degenerate case; collapsing source traces breaks the "concept points back to source traces" contract. |

### 4.4 AE-vs-LM layer

**LM-internal.** AE caller stays as-is. Reasoning in `dedup-noise.md` §5: one central rule beats N project patches; the round-trip cost differential (15 s LM-side vs 3 min AE-side per bootstrap); AE has no obligation to be the gatekeeper of LM identity invariants. The path-inventory's deferred AE-vs-LM choice is closed here as LM-internal.

### 4.5 Downstream file ownership (owned by `fix-dedup-noise`)

| File | What changes |
| --- | --- |
| `src/living_memory/storage.py` | `SCHEMA_VERSION` 2→3; `content_fingerprint TEXT` column in `_initialize_schema`; `idx_nodes_dedup` partial index; migration backfill; dedup branch in `_insert_node` (`:140-213`). |
| `tests/test_dedup_supersede.py` (new) | Unit tests per `dedup-noise.md` §11.1 — 11 contract tests covering same-scope dedup, supersedes metadata, cross-scope/level non-dedup, file-chunk/file-summary/project-overview shapes, N-repeated inserts, raw content preservation, already-decayed exclusion, recall returning only active. |
| `tests/test_storage.py` | Extend to assert SCHEMA_VERSION = 3, new column exists, `idx_nodes_dedup` exists, schema-v2 migration backfills fingerprints. |
| `tests/test_resources_prompts.py` | Adjust `test_memory_health_reports_activity_dedup_and_staleness` (`:122`) so synthetic-duplicate setup uses distinct content with shared prefix (existing test relies on duplicates accumulating; post-fix they would collapse). |

Co-touch coordination with `fix-scope-hygiene`: both fixes edit lines ~160-165 of `_insert_node`. Sequencing: scope canonicalisation first (line 161), then fingerprint dedup branch (immediately after). File ownership split is mechanical, not contested. The dedup query consumes the post-normalisation scope, so two writers passing `"rise/_critique"` and `"project:rise/_critique"` for identical content correctly fingerprint as duplicates.

### 4.6 Numeric target

| Target | Threshold | Notes |
| --- | ---: | --- |
| Unit-level contract test | Identical-content remembers in the same scope produce exactly 1 active + N decayed via `supersedes` | Deterministic; CI-friendly. |
| Steady-state duplicate density on a new scope after fix | ≤ 1% over 30 days of continuous bootstrap re-runs | Residual is the legitimate near-duplicate floor. |
| Live `project:ae` duplicate density after one-shot backfill | ≤ 5% | Backfill is operator-triggered (deferred per §11). Without backfill, baseline 23.6% ages out only via TTL. |
| No degradation in other scopes | `project:lm` ≤ 0.95%; `project:octopus` ≤ 0.19%; `global` ≤ 0%; `project:online` ≤ 9.4% | Fix is additive. |

### 4.7 Raw-history preservation rule

Per `dedup-noise.md` §7: no `DELETE FROM nodes`; `get_node(old_id)` returns the row after dedup; `supersedes` edge is the audit trail; `content_fingerprint` immutable post-insert; backfill is read-from-`content` only; no content is rewritten/summarised/compressed. The append-only contract is mechanically defensible: one INSERT (new trace) + one INSERT per duplicate found (supersedes edge) + one `UPDATE … SET decayed=1, decay_reason=?` per duplicate. None mutate prior `content`.

### 4.8 Fixture / live-DB policy

Per §2.2: unit and write benchmarks use temp-path stores; read benchmarks against the frozen snapshot under `mode=ro`. Schema migration is tested against a v2 fixture DB committed alongside the fix.

### 4.9 Latency micro-check

The fix node runs paired before/after benchmarks per `recall-speed-usefulness.md` §"Per-fix benchmark instructions / Dedup":

- **Write path (temp DB)**: unique insert mean ≤ 1.0 ms; duplicate insert mean ≤ 2.0 ms; unique-insert regression ≤ 1.20× of master.
- **Read path (frozen snapshot)**: in-process `memory_recall` median ≤ 100 ms for `project:lm`; regression ≤ 1.05× of master.
- **Supersedes-heavy micro-bench**: synthetic 1,000+ `supersedes` rows in a temp DB; rank time stays within the regression budget. If exceeded, the fix node adds a `type`-leading connection index or a cache invalidated by connection writes — not a background indexer.

JSON sidecars `artifacts/latency_before.json` and `artifacts/latency_after.json` capture both write and read measurements; the result.md reports the multiplier.

---

## 5. Problem 3 — Feedback linkage

### 5.1 Root cause

Three interacting causes (per `feedback-linkage.md` §"Root Cause"):

1. **`memory_remember` and `memory_teach` consume at most one pending recall per call.** `apply_pending_recall_feedback` defaults `limit=1` (`src/living_memory/feedback.py:135-140`); `memory_remember` calls it without override (`src/living_memory/server.py:489-501`); `memory_teach` reaches the same default through `src/living_memory/consolidation.py:375-379`. A normal workflow does several recalls before one remember/teach, so all but the newest compatible event remain pending forever.
2. **The event matcher is scope-only unless both sides set `session_id`.** `record_recall_event` stores `agent`, `task`, and `session_id` from `ambient_context` (`src/living_memory/storage.py:618-645`), but `_recall_event_matches` only checks scope membership and optional session equality (`src/living_memory/storage.py:1321-1334`). Task and agent are ignored even when both are present. Unrelated same-scope recalls can attach to the next remember.
3. **Recall-side metadata is not propagated by common clients.** MCP accepts `ambient_context` (`src/living_memory/server.py:555-571`) and retrieval passes it to storage (`src/living_memory/retrieval.py:162-168`), but AE `cmd_recall` sends only query/scope/depth/max (`/home/sfx/p/ae/lm_client.py:108-116`) and exposes no `--agent`/`--task` flags for recall (`lm_client.py:1019-1025`). By contrast `cmd_remember` does pass them (`lm_client.py:81-99`, `:1011-1017`). Server instructions mandate structured metadata for `memory_remember` only (`server.py:401-408`), not `memory_recall.ambient_context`.

Snapshot aggregates that confirm the diagnosis (from `feedback-linkage.md` §"Baseline Signal"):

- 174 / 5,114 recall events have `agent` (3.40%); 268 / 5,114 have `task` (5.24%); 0 / 5,114 have `session_id`.
- Active traces are much richer: 4,005 / 5,029 have `agent`; 4,872 / 5,029 have `task`.
- Every applied event has a unique `feedback_trace_id` (1,060 events / 1,060 traces; `max_events_per_trace=1`) — exactly the `limit=1` signature.
- Among applied events where both sides have task metadata: 53 same-task, 53 different-task — the matcher is also linking *wrong* events, not just under-linking.
- 97 applied events attach a project-scoped recall to a `global` feedback trace via `event.resolved_scopes` containing `global`.

Raising `limit` alone would worsen false positives.

### 5.2 Chosen simplest central fix

Implement centrally in LM, not in AE as the primary layer.

- **`src/living_memory/storage.py:_recall_event_matches` (`:1321-1334`)** — make matching context-aware:
  - Preserve exact-scope fallback compatibility for old clients, but cap unqualified scope-only linking at one event.
  - Treat `session_id` and `task` as strong discriminators: if both sides provide either field and values differ, reject.
  - Treat `agent` as a weak discriminator: if both sides provide agent and no task/session is available, require equality.
  - For fallback scope matches where the feedback trace scope is present only via `event.resolved_scopes` (e.g. project recall → `global` trace), require a strong same-task or same-session match. Blocks current project-to-global false positives.
- **`src/living_memory/feedback.py:apply_pending_recall_feedback` (`:135-234`)** — raise the effective pending-event cap for strong matches to a small constant (e.g. 5). A single remember/teach can consume the immediately preceding same-task recall sequence but cannot vacuum arbitrary same-scope history.
- **`src/living_memory/server.py:_server_instructions` (`:255-425` / mandatory-metadata block at `:401-408`)** — update instructions so `memory_recall.ambient_context` documents the same `scope`/`task`/`agent`/`session_id` discipline that `memory_remember` already mandates. This is a documentation change inside server-generated client instructions; the public MCP signature does not change.
- **`tests/test_recall_feedback_loop.py` (extend, current coverage at `:76-147`)** — positive multi-recall + negative mismatch cases.

This is a write-path-only fix; the recall hot path stays untouched.

### 5.3 Rejected alternatives

| # | Alternative | Reason |
| --- | --- | --- |
| R1 | Increase `limit` only | Improves aggregate ratio but amplifies the existing false-positive bug. |
| R2 | Require `session_id` everywhere | Clean in theory, but baseline has 0 / 5,114 events with session metadata; would break current recall→remember flows until every caller changes. |
| R3 | Make AE CLI the only fix | Direct MCP clients and LM server behaviour would still have `limit=1` and the permissive matcher. Useful as a follow-up (§11), not as the primary. |
| R4 | Add an explicit `recall_event_ids` parameter to `memory_remember` | New public contract when the existing API has enough context fields for an automatic fix. |
| R5 | Rewrite historical `recall_events` | Violates append-only/raw-history principle; cannot prove which old pending events were actually useful. |
| R6 | Disable implicit feedback entirely | Avoids false positives but removes the useful provenance/reinforcement loop existing tests protect. |

### 5.4 AE-vs-LM layer

**LM-internal (primary).** The server owns correctness and false-positive prevention. An AE follow-up that adds `--agent` / `--task` flags to `cmd_recall` and `cmd_teach` (and derives them from the same goal-node env used by `cmd_remember`/`cmd_bootstrap_project`) would compound benefit, but the LM fix does not depend on it. Recorded in §11.

### 5.5 Downstream file ownership (owned by `fix-feedback-linkage`)

| File | What changes |
| --- | --- |
| `src/living_memory/storage.py` | Update `_recall_event_matches` (`:1321-1334`) with task/agent/session-aware predicate; ensure the pending-event query window stays bounded (existing `idx_recall_events_scope_pending_created` / `idx_recall_events_session_pending_created` at `:1102-1106`). |
| `src/living_memory/feedback.py` | Raise the effective consumption cap for strong matches in `apply_pending_recall_feedback` (`:135-234`); keep `limit=1` semantics for weak fallback. |
| `src/living_memory/server.py` | Update `_server_instructions` (`:255-425`) to mandate `ambient_context` task/agent/session on `memory_recall`. |
| `tests/test_recall_feedback_loop.py` | Add positive multi-recall consumption, positive same-task multi-recall + remember, and negative mismatch tests (different task, different session, project→global via `resolved_scopes`, legacy no-context exact-scope cap-at-1). |

No overlap with `_insert_node`-side fixes (#1, #2); no overlap with `resources.memory_health` (#4).

### 5.6 Numeric target

For new writes after the fix:

- Deterministic unit positive: two same-scope, same-task recall events followed by one remember/teach with same task → both recall events marked `feedback_applied` (2/2 in fixture).
- Deterministic unit negatives: different-task recall in the same scope stays `feedback_applied=0`; project-scoped recall does not attach to a `global` remember via `resolved_scopes` unless same `task` or `session_id` is present.
- Live metric: do not promise to retroactively raise the historical 20.73% ratio. The measurable improvement is that future same-task eligible sequences no longer leave all but one event pending, and future task-mismatched applications are zero.

### 5.7 Fixture / live-DB policy

Per §2.2: extend `tests/test_recall_feedback_loop.py`'s existing `FakeMCP` pattern; per-test tmp-path stores; no live-DB writes.

### 5.8 Latency micro-check

Recall hot path is unchanged. The fix node runs:

- The canonical hot-service matrix against the frozen snapshot before/after, asserting no regression per §9 budgets.
- A temp-copy write benchmark per `recall-speed-usefulness.md` §"Per-fix benchmark instructions / Feedback linkage": seed 20 pending recall events in one scope/task/session, call `memory_remember`, verify all intended same-task events are marked `feedback_applied` and unrelated task/session events remain pending, record wall time. Stays in low single-digit milliseconds on local in-process execution.

The matching query must remain bounded and index-friendly. Raising the default cap is acceptable only if `pending_recall_events` still fetches a small multiple of the limit and filters in Python (existing pattern at `storage.py:681-709`).

---

## 6. Problem 4 — Health and observability

### 6.1 Root cause

The database and helper functions contain the data; `memory_health` simply does not aggregate it. Citations from `health-observability.md` §"Current Gap":

- `memory_status` is reflective (scope/phase/counts/confidence/coverage/promotions/policy) — not audit-oriented (`src/living_memory/resources.py:92-104`).
- `memory_health` already reports recall/remember activity, duplicate density, staleness/decay, retrieval policy (`resources.py:323-459`; MCP wrapper at `server.py:609-623`).
- HTTP `/health` is liveness only (`server.py:130-150`) — wrong place for audit metrics.
- `recall_events_summary` already computes `feedback_applied` (`resources.py:462-504`), but is not registered as an MCP resource (`server.py:626-645`; inventory note `path-inventory.md:332-335`).
- `coverage_summary`/`scope_summary` expose per-scope counts (`resources.py:221-245`, `:279-295`) — nothing turns them into leakage candidates.
- `MemoryStore` retains the DB path (`storage.py:63`); schema has `nodes.access_count`, `nodes.last_accessed`, `recall_events.feedback_applied` (`storage.py:976-1061`). Health does not aggregate never-accessed traces, DB size, feedback coverage.
- Server instructions exist (`server.py:46-69`, `:255-425`) pinned by `tests/test_instructions_imperative.py:21-191`, but no status surface reports instruction size or contract status.
- Hot recall latency is measurable through the production retrieval helper with `log_access=False` / `log_event=False` (`retrieval.py:108-174`, `:485-504`), but no surface exposes a reproducible baseline.

The missing metrics are not blocked by storage schema or MCP API shape; they are omitted aggregations.

### 6.2 Chosen simplest central fix

Extend the existing `memory_health` surface and its resource helpers, not `memory_status`, not HTTP `/health`, and not a dashboard.

Concrete downstream shape (per `health-observability.md` §"Chosen Extension Surface"):

- Add additive sections to `resources.memory_health`: `feedback`, `access`, `scope_hygiene`, `storage`, and optional `latency`.
- Add an `instructions` section via a small optional argument, e.g. `instructions_text: str | None = None`, so `resources.py` does not import `server.py`. The MCP wrapper at `server.py:609-623` passes `_server_instructions(store.config.default_scope)`.
- Add an optional latency argument to the MCP tool, e.g. `latency_samples: int = 0`. Existing callers stay compatible because the default is zero. When nonzero, run the production retrieval helper with `log_access=False` / `log_event=False`.
- Keep `memory_status` unchanged (`resources.py:92-104`).
- Keep HTTP `/health` unchanged (`server.py:130-150`).
- Add focused tests beside the existing health tests (`tests/test_resources_prompts.py:122-170`); add MCP wrapper coverage only for new optional parameters if needed.

Resulting output stays one resource-shaped JSON object. Avoids a dashboard, background daemon, shadow store, cache hierarchy, or new public mandatory API.

### 6.3 Metric definitions (must match `baseline.md`)

The fix node consumes these definitions verbatim from `health-observability.md` §"Baseline-Aligned Metric Definitions":

- **Duplicate density** — already implemented in `memory_health.dedup` (`resources.py:377-387`). Baseline: 5,029 active traces / 4,377 distinct / 652 excess / 0.1296 (`baseline.md:93-109`).
- **Feedback applied ratio** — SQL aggregate over `recall_events`; for scoped health, filter by `recall_events.scope = normalized_scope` to match existing `activity.recall_total` rule (`resources.py:341-359`). Baseline: 5,114 events / 1,060 applied / 0.2073 (`baseline.md:121-144`).
- **Never-accessed ratio** — two aggregates (active nodes, active traces). Baseline: 3,710 / 5,128 = 0.7235 (active nodes); 3,681 / 5,029 = 0.7320 (active traces) (`baseline.md:146-172`).
- **Scope leakage candidates** — exact baseline predicate (`baseline.md:178-185`); report candidate scope count, active candidate nodes, active candidate traces, never-accessed candidate traces + ratio, candidate scopes with zero recall events as both `scope` and `requested_scope`. Baseline: 54 scopes, 111 traces, 111 never accessed, 36 zero-recall scopes (`baseline.md:188-213`). The predicate is intentionally an *audit* predicate; it must not become the scope-hygiene fix itself.
- **DB size** — `store.db_path.stat().st_size` when `store.db_path != ':memory:'`; also `PRAGMA page_count * PRAGMA page_size` as a portable estimate. Baseline: 100,339,712 bytes (`baseline.md:40-43`).
- **Instructions / test contract** — report `default_scope`, instruction length (chars + UTF-8 bytes), counts of literal `MUST` / `MUST NOT` / `BEFORE` / `AFTER`, and the focused command `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_instructions_imperative.py -p no:cacheprovider`. **Do not invoke pytest from inside `memory_health`** — expose facts + command; a separate audit/test attaches pass/fail.
- **Hot recall latency baseline** — production retrieval path with `log_access=False, log_event=False`. Baseline: median 91.869 ms (`baseline.md:215-271`). Prior live MCP baseline was ~65-70 ms. Must be opt-in, deterministic, read-only; default `memory_health` must not run a recall benchmark.

### 6.4 Rejected alternatives

| # | Alternative | Reason |
| --- | --- | --- |
| R1 | Register `memory://recall_events` / `memory://connections` as new resources | Exposes fragments; leaves leakage/access/DB-size/instructions/latency elsewhere; increases surface area without producing one reproducible audit result. |
| R2 | Extend `memory_status` | Status is phase/coverage/policy-oriented and should stay cheap; `memory_health` already owns dedup/staleness/activity. |
| R3 | Put audit metrics on HTTP `/health` or `/admin/info` | `/health` is restart/liveness; `/admin/info` is process metadata and bearer-auth-gated. |
| R4 | Dashboard or persistent background metric collector | Forbidden by root principle; every metric is computable from SQLite or a small opt-in latency probe. |
| R5 | Run pytest from inside `memory_health` to compute instruction contract status | Runtime health should not spawn a test runner or write pytest cache. Expose facts + command. |
| R6 | Measure latency on every `memory_health` call | Would distort the baseline and add unexpected cost. Use opt-in samples with read-only recall logging disabled. |

### 6.5 AE-vs-LM layer

**LM-internal.** No AE-side change is required for health observability.

### 6.6 Downstream file ownership (owned by `fix-health-audit`)

| File | What changes |
| --- | --- |
| `src/living_memory/resources.py` | Metric helper SQL and additive `memory_health` sections (`feedback`, `access`, `scope_hygiene`, `storage`, optional `latency`, optional `instructions`); pass-through arguments. |
| `src/living_memory/server.py` | MCP `memory_health` wrapper signature; instruction-text pass-through; optional latency argument. |
| `tests/test_resources_prompts.py` | Fixture tests for feedback ratio, never-accessed ratio, leakage summary, DB size, instruction metrics. |
| `tests/test_mcp_server.py`, `tests/test_fastmcp_runtime.py` | Update only if optional MCP tool parameters need explicit surface coverage. |

No write-path overlap with #1, #2, or #3.

### 6.7 Numeric target

The fix node is successful when a single audit command can reproduce the baseline-defined fields (duplicate density, feedback ratio, leakage candidates, never-accessed ratio, DB size, instruction/test-contract status, opt-in latency baseline) on the frozen snapshot or on the live DB read-only, without changing existing `memory_status`, `/health`, `/admin/info`, `/admin/restart`, recall, remember, teach, consolidate, or default `memory_health` compatibility.

Quantitative anchor: reproduced ratios must match `baseline.md` to ≤ 0.001 (rounding tolerance) when the same snapshot is read.

### 6.8 Fixture / live-DB policy

Per §2.2: fixture DBs for tests; do not mutate the live DB in health tests. For any local latency check, `log_access=False` / `log_event=False`; if the live DB is measured manually, record before/after DB size and avoid storing recall events.

### 6.9 Latency micro-check

Per `recall-speed-usefulness.md` §"Per-fix benchmark instructions / Health":

- Measure `memory_health(store, scope=None, window_hours=168, top_stale=5)` before/after on the frozen snapshot.
- After/before ratio ≤ 1.25 and absolute increase ≤ 25 ms.
- New metrics must be SQL aggregates or bounded lists. A latency section must be opt-in or cached. Default `memory_health` must not call `memory_recall`.

---

## 7. Problem 5 — Recall speed and usefulness (constraint, not standalone fix)

### 7.1 Root cause(s)

Six interacting concerns (from `recall-speed-usefulness.md` §"Root causes"):

1. **Graph traversal dominates non-trivial hot recall** (`retrieval.py:401-459`). One profiled `project:lm` query had 122 candidates before graph, expanded to 261, and issued 122 `list_connections` calls. For `project:lm` `depth=1`, graph took ~45.9 ms of ~69.8 ms service-hot median; for `project:ae`, ~97 ms of ~178 ms.
2. **Vector / BM25 cost scales with active rows, not returned rows.** `max_results=5` still scans every embedded row in resolved scopes before truncating (`retrieval.py:302-348`). `project:ae` has 2,674 active embedded rows (many duplicate bootstrap chunks).
3. **Scope expansion doubles/triples scans.** Project recall always includes `global` (`scope.py:161-168`); session recall expands to (session, project, global) (`scope.py:170-179`). Any write-side fix must canonicalise to one project scope rather than adding alias scopes to recall plans — alias expansion would multiply BM25/vector/schema/graph work.
4. **Duplicate / file-chunk noise has a retrieval cost, not only a health cost.** Active duplicates inflate BM25/vector candidate pools and each receive independent access/usefulness/supersedes boosts via `feedback.py:237-252`.
5. **`_supersedes_sets()` (`retrieval.py:473-479`) is cheap only because supersedes is rare** (34 rows). If dedup creates hundreds/thousands of `supersedes` edges, this becomes a hot-path regression unless indexed/cached. Existing indexes are `(source_id, type)` and `(target_id, type)` — no `type`-leading index (`storage.py:1096-1100`).
6. **Access / recall-event logging is not the median cost today, but it is serialised under the global `runtime_lock`** (`server.py:78`). Fixes must not add per-candidate logging or extra write amplification on recall.
7. **`memory_health` can become a hot-lock problem** if it runs a latency probe inside every call under the global runtime lock — would block all other MCP tools.

### 7.2 Chosen latency-safe guidance (not an independent fix)

The simplest central rule: **reduce active noisy rows and keep recall-time work bounded by the existing resolved scopes; do not add new per-recall scans, alias expansion, or writes.**

Concrete constraints for the four fix nodes:

- **Scope hygiene (#1)**: canonicalise on write before storage / recall planning. Do not solve leakage by searching extra aliases (would multiply scans and leave fragmentation in ranking).
- **Dedup (#2)**: use stable content fingerprints to mark redundant active traces as superseded / decayed while preserving append-only rows. If `supersedes` edges are used for dedup, add a `type`-leading connection index *or* a small invalidated cache **before** supersedes volume grows.
- **Feedback linkage (#3)**: improve recall→remember matching on the write path. Recall must not collect or log more. Any larger `limit` must keep the query indexed and bounded.
- **Health (#4)**: extend `memory_health` with SQL aggregates and optional/cached latency data. Default `memory_health` must not run recall benchmarks.
- **Ranking / usefulness** (no independent fix in this round): suppress duplicate active candidates *before* they reach ranking rather than adding another rank-time penalty (penalties leave duplicate rows in BM25/vector/graph scans anyway).

### 7.3 Rejected alternatives

| # | Alternative | Reason |
| --- | --- | --- |
| R1 | Disable vector search | Breaks existing semantic / cross-language coverage (`tests/test_retrieval.py:82-135`, `tests/test_acceptance_contract.py:64-88`). |
| R2 | Disable graph traversal by default | Regresses causal / decision / correction coverage (`tests/test_graph_recall.py:107-130`); existing `depth=0` already gives this for opt-in callers. |
| R3 | Add ANN / vector-index infrastructure now | Current scale is thousands of vectors; the deterministic hash fallback must stay intact (`src/living_memory/embeddings.py:261-345`); root goal forbids heavy subsystems without proof. |
| R4 | Search every possible scope alias to recover leaked memories | Hides scope-hygiene failure while slowing every recall. Canonicalise on write + optionally migrate/soft-link separately. |
| R5 | Turn off access logging or recall events | Protects latency but breaks feedback linkage, health ratios, provenance. Keep per-returned-result / per-event logging only. |
| R6 | Compute hot recall latency on every `memory_health` | Would run under the global runtime lock; could mutate unless carefully disabled. Use explicit benchmark command or cached opt-in field. |
| R7 | Hard-code project-specific duplicate / ranking exceptions | Same root causes apply to AE, Octopus, LM, Online, and future projects. Central fingerprints + scope canonicalisation + bounded retrieval rules instead. |

### 7.4 Per-fix latency micro-check spec (binding for #1–#4)

Every downstream fix node touching recall ranking, scope resolution, dedup, feedback, access logging, health, storage indexes, or decay must run a paired before/after benchmark in the same worktree and report (per `recall-speed-usefulness.md` §"Deterministic micro-benchmark policy"):

- `git rev-parse --short HEAD`
- Python version + whether `numpy` is importable
- DB source and MD5
- query/scope/depth/max_results matrix
- median + p95 over ≥ 7 measured iterations after one warm-up
- whether the benchmark was no-write (`log_access=False, log_event=False`) or temp-copy write-enabled

**Regression gates** (binding for all four fix nodes):

| Metric | Threshold |
| --- | --- |
| Hot recall service-path median for `project:lm` | after/before ≤ 1.10 AND absolute increase ≤ 10 ms |
| Hot recall service-path median for `project:ae`, `project:octopus` | after/before ≤ 1.10 AND absolute increase ≤ 20 ms |
| p95 for any scope | after/before ≤ 1.20 (higher values must include raw samples + explanation) |
| `memory_health(scope=None, top_stale=5)` median | after/before ≤ 1.25 AND absolute increase ≤ 25 ms (unless latency reporting is opt-in and default-disabled) |

A fix that intentionally trades latency for usefulness must include a failing/passing usefulness assertion AND the measured latency cost — do not hide the tradeoff.

### 7.5 Canonical benchmark snippets

The fix nodes use the snippets from `recall-speed-usefulness.md` §"Deterministic micro-benchmark policy":

- **Hot-service matrix** (long-lived `MemoryRecallService`, 9 measured iterations, no-write) — `recall-speed-usefulness.md` §"Canonical hot-service benchmark".
- **Cold-helper compatibility** (module-level `memory_recall`, 7 iterations, no-write) — matches `baseline.md` §"Hot Recall Latency".
- **Write-enabled** (temp copy of the snapshot via `shutil.copy2`, three `(log_access, log_event)` matrices, 7 iterations each).

**Important benchmark nuance**: `retrieval.py:485-504` constructs a fresh `MemoryRecallService` per module-level `memory_recall` call; the MCP server constructs one long-lived service at `server.py:471-473`. The long-lived service reuses `_embedding_cache` (`retrieval.py:100-106`, `:350-368`). Fix nodes must report both "cold helper" and "hot service" numbers rather than comparing one mode against the other.

### 7.6 Downstream policy

- Frozen snapshot for read-path measurement.
- Temp copies for any benchmark that writes `access_count`, `recall_events`, feedback, decay, or `supersedes` rows.
- No live-DB writes during fix-node iteration.
- Live MCP latency probes are out of scope for the discovery subtree; any later integration-owned probe must carry its own scope-contract boundary and provenance tags.

---

## 8. Integration sequencing for the four downstream fix nodes

### 8.1 Dependency DAG

```
                ┌─────────────────────┐
                │ discover (this)     │
                │ artifacts/ANALYSIS  │
                └──────────┬──────────┘
                           │
        ┌──────────┬───────┼─────────┬──────────┐
        ▼          ▼       ▼         ▼          ▼
  fix-scope    fix-dedup  fix-feedback  fix-health  (no separate
   hygiene     noise      linkage       audit       recall-speed
        │          │       │            │           fix node;
        │ co-touch │       │ disjoint   │ disjoint   constraint
        │ _insert_node     │ feedback.py│ resources  only)
        ▼          ▼       │            │
       ┌────────────┐      │            │
       │ integrate  │      │            │
       │ (sequenced)│      │            │
       └────────────┘      │            │
                │          │            │
                └──────────┴────────────┘
                           │
                           ▼
                   verify-and-record
```

### 8.2 Sequencing rules

1. **`fix-scope-hygiene` and `fix-dedup-noise` serialise at `_insert_node`** (`storage.py:140-213`). Order is fixed: scope canonicalisation lands first (line 161 area), then fingerprint dedup branch (immediately after). Sequencing rationale in §3.5 / §4.5: dedup query must consume the post-normalisation scope so two writers passing `"rise/_critique"` and `"project:rise/_critique"` correctly fingerprint as duplicates.
2. **`fix-feedback-linkage` is independent** of #1 and #2. Touches `feedback.py:135-234`, `storage.py:1321-1334`, and the server instruction block (`server.py:255-425`). Can land in parallel.
3. **`fix-health-audit` is independent** of #1, #2, #3. Touches `resources.py:323-459` and the `memory_health` MCP wrapper at `server.py:609-623`. Can land in parallel; will consume the metric definitions in `baseline.md` and §6.3 unchanged.
4. **Verification (`verify-and-record`)** runs after all four fix nodes have merged. The hot-recall latency comparison and the audit-metric comparison both read the post-fix state.

### 8.3 File-ownership split (defuses parent `_critique` observation 4)

| File | `fix-scope-hygiene` owns | `fix-dedup-noise` owns | `fix-feedback-linkage` owns | `fix-health-audit` owns |
| --- | :-: | :-: | :-: | :-: |
| `src/living_memory/storage.py:16` (imports) | ✓ (add `from living_memory.scope import normalize_scope`) | ✓ (no new import unless backfill helper requires one) | — | — |
| `src/living_memory/storage.py:31` (`SCHEMA_VERSION`) | — | ✓ (2→3) | — | — |
| `src/living_memory/storage.py:140-213` (`_insert_node`) | ✓ (line 161-164 scope rewrite) | ✓ (fingerprint branch after line 164; supersedes+decay before/after INSERT) | — | — |
| `src/living_memory/storage.py:_initialize_schema` (`:961-1116`) | — | ✓ (column + index) | — | — |
| `src/living_memory/storage.py:1321-1334` (`_recall_event_matches`) | — | — | ✓ | — |
| `src/living_memory/feedback.py:135-234` | — | — | ✓ | — |
| `src/living_memory/resources.py:323-459` (`memory_health`) | — | — | — | ✓ |
| `src/living_memory/server.py:255-425` (`_server_instructions`) | — | — | ✓ (recall metadata block) | — |
| `src/living_memory/server.py:609-623` (`memory_health` MCP wrapper) | — | — | — | ✓ |
| `tests/test_scope.py` | ✓ | — | — | — |
| `tests/test_storage.py` | ✓ (idempotency / unknown-prefix) | ✓ (schema migration / column / index) | — | — |
| `tests/test_dedup_supersede.py` (new) | — | ✓ | — | — |
| `tests/test_recall_feedback_loop.py` | — | — | ✓ | — |
| `tests/test_resources_prompts.py` | — | ✓ (adjust `:122` synthetic-duplicate fixture) | — | ✓ (new health-section tests) |

### 8.4 Integrate order (when more than one fix is ready to merge)

1. `fix-scope-hygiene` (smallest blast radius; tightens write contract without semantic shift on valid callers).
2. `fix-dedup-noise` (rebases onto canonicalised scope).
3. `fix-feedback-linkage` (independent surface; can land before or after the storage-side fixes but is most informative once dedup is live).
4. `fix-health-audit` (consumes the post-fix metric values; surface stays compatible with pre-fix state).

---

## 9. Per-fix latency micro-check spec (binding summary)

Defuses parent `_critique` observation 1 ("latency observability is verify-only"). Every fix node self-verifies before integrate.

| Fix | What to measure | Threshold |
| --- | --- | --- |
| #1 `fix-scope-hygiene` | Canonical hot-service matrix on frozen snapshot, before/after | median delta ≤ 5 ms; after/before ≤ 1.05; ambient-derived canonical scope ≤ 5 ms or 5% slower than explicit scope (7-sample median). |
| #2 `fix-dedup-noise` | Write-path (temp DB) + read-path (frozen snapshot) + supersedes-heavy synthetic | Unique insert ≤ 1.0 ms mean; duplicate insert ≤ 2.0 ms; unique-insert regression ≤ 1.20× of master; hot recall median ≤ 100 ms `project:lm`; recall regression ≤ 1.05× of master; supersedes-heavy rank time within §7.4 budget. |
| #3 `fix-feedback-linkage` | Canonical hot-service matrix + temp-copy write benchmark | Recall regression per §7.4 budget; `memory_remember` consumption of 20 pending events ≤ low single-digit ms. |
| #4 `fix-health-audit` | `memory_health(scope=None, top_stale=5)` median, before/after | after/before ≤ 1.25; absolute increase ≤ 25 ms; default `memory_health` does not call `memory_recall`; optional latency section is opt-in. |

JSON sidecars `artifacts/latency_before.json` and `artifacts/latency_after.json` for each fix node; result.md reports the multiplier.

---

## 10. Numeric improvement targets (binding summary)

Defuses parent `_critique` observation 3 ("quantitative thresholds for qualitative postconditions").

| Fix | Metric | Pre-fix baseline (`baseline.md`) | Post-fix target |
| --- | --- | --- | --- |
| #1 Scope hygiene | New bare-scope rows per audit window | 111 active candidate traces accumulated historically | 0 new bare-scope rows after fix |
| #1 Scope hygiene | `memory_remember` error on unknown prefix | Silently writes mal-formed scope | Raises `ValueError("unsupported scope prefix: <prefix>")` |
| #2 Dedup | Unit-level contract | N/A | Identical-content remembers in same scope → exactly 1 active + N decayed via `supersedes`, with `metadata.kind="duplicate_content"` |
| #2 Dedup | Steady-state duplicate density on a new scope (30 d) | N/A | ≤ 1% |
| #2 Dedup | Live `project:ae` duplicate density | 0.236116 (23.6%) | ≤ 5% post-backfill (operator-triggered, deferred per §11); steady-state via TTL otherwise |
| #2 Dedup | Other-scope duplicate density | `project:lm` 0.95%; `project:octopus` 0.19%; `global` 0%; `project:online` 9.4% | No degradation |
| #3 Feedback | Same-task multi-recall consumption | 1 / N pending events (`limit=1`) | 2 / 2 pending events linked in fixture; N / N for N ≤ 5 |
| #3 Feedback | Different-task false positives | 53 / 106 ≈ 50% of dual-task applied events are different-task; 97 project→global via `resolved_scopes` | 0 false positives in fixture for task / session / project-to-global mismatch cases |
| #3 Feedback | Live historical 20.73% ratio | Not promised to retroactively rise | Future eligible sequences no longer leave all but one event pending; future task-mismatch applications = 0 |
| #4 Health | New `memory_health` sections | Duplicate density only | `feedback` + `access` + `scope_hygiene` + `storage` + optional `latency` + optional `instructions` |
| #4 Health | Reproduced ratios vs `baseline.md` | Snapshot definitions | Match to ≤ 0.001 tolerance on same snapshot |
| #5 Recall speed | Hot recall median (any fix) | `project:lm` ~91.9 ms in-process; ~65-70 ms MCP hot | No regression per §7.4 regression gates |

---

## 11. Deliberately deferred follow-ups

These are real but separate concerns. They are not part of any fix's central guard and are recorded so the verify-and-record subtree or a follow-up goal can sequence them.

### 11.1 From scope-hygiene

1. **Worktree-aware cwd canonicalisation** in `scope._ambient_scope` (`scope.py:198-202`) and `scope._ambient_project` (`scope.py:223-227`): when `Path(cwd).name` looks like `_node_exec_*`, walk up to find `.worktrees/` and use the parent directory's name. Affects only `recall_events.requested_scope`; does not unblock the dominant write-side leakage.
2. **Historical migration of bare-scope rows.** SQL `UPDATE` to rescue the 111 active candidate traces. Deferred because the team may prefer TTL aging, and policy choices (`rise/* → project:octopus` vs `→ project:octopus/rise/*` vs no migration) are not unanimous.
3. **`memory_remember` accepting `ambient_context`.** If a future change wants cwd-derived scope on writes, the public MCP signature gains an `ambient_context` parameter.
4. **Tighten `infer_project_scope` token overlap** (`scope.py:122-143`): permissive `project_tokens & query_tokens` could cross-match `project:rise` from a query like "what rises in latency?". Not a current bug; documented for completeness.

### 11.2 From dedup-noise

5. **One-shot retroactive backfill of existing duplicates** in `project:ae` (625 excess) and `project:online` (21 excess). Operator-triggered via `scripts/dedup_backfill.py` or `POST /admin/dedup-backfill`. Without it, residual duplicates age out via 180-day TTL.
6. **Path/kind-based supersede for changed files.** When `[file-summary]` for the same `(path, language)` is emitted after a file changed, new content has a different SHA256; older now-stale trace remains active until TTL. Distinct from exact-content dedup; would require LM to parse AE-specific JSON header (leaks caller format into LM contract). Cleaner is for AE to emit a path-based forget or supersede call.
7. **AE caller round-trip optimisation.** Post-LM-side dedup, AE bootstrap still emits ~3,000 `memory_remember` round-trips per refresh (~15 s of duplicate transport overhead). AE could add a pre-check that recalls a single content-digest manifest from LM and skips files already present. Caller-side optimisation (not correctness).
8. **New health metrics specific to dedup** (e.g. `dedup_supersedes_in_window`). Owned by `fix-health-audit`; dedup brief relies on the existing `dedup_density` field.

### 11.3 From feedback-linkage

9. **AE recall-side metadata enrichment.** Add `--agent` / `--task` flags to `cmd_recall` and `cmd_teach` (`/home/sfx/p/ae/lm_client.py:108-116, :213-220, :1019-1025`); derive them from the same goal-node env used by `cmd_remember` / `cmd_bootstrap_project`. Would raise the 3.4% / 5.2% recall-event metadata coverage to the 79.6% / 96.9% trace coverage seen today.

### 11.4 From health-observability

10. **Register `recall_events_summary` / `connections_summary` as MCP resources** if a downstream consumer needs them as standalone surfaces beyond the integrated `memory_health` view (currently helpers at `resources.py:462-524` are not registered MCP resources per `path-inventory.md:332-335`).

### 11.5 From recall-speed-usefulness

11. **`type`-leading connection index or invalidated supersedes cache.** Required pre-emptively if `fix-dedup-noise` creates hundreds/thousands of `supersedes` edges. The fix node should add this proactively (per §4.9) rather than defer.
12. **Live MCP latency probes** for verify-and-record. Out of scope for the discovery subtree; the integration-owned probe must carry its own scope-contract boundary and provenance tags.

---

## 12. Mapping to parent SPEC

Defuses parent `_critique` observation 6 ("`ANALYSIS.md` is the single fan-out point for four downstream builds").

| Parent SPEC item (from `discover-baseline-and-root-causes`) | Where this synthesis covers it |
| --- | --- |
| P4 (Scope hygiene: precise root cause + simplest central fix + rejected alternatives + AE-vs-LM + downstream file ownership) | §3 (root cause), §3.2 (fix), §3.3 (rejected), §3.4 (layer), §3.5 (ownership) |
| P5 (Dedup: + numeric improvement target + raw-history rule) | §4 (root cause), §4.2 (fix), §4.3 (rejected), §4.4 (layer), §4.5 (ownership), §4.6 (numeric), §4.7 (raw-history) |
| P6 (Feedback linkage: + metadata/scope/session behaviour + negative-case constraints + numeric target) | §5 (root cause), §5.2 (fix), §5.3 (rejected), §5.4 (layer), §5.5 (ownership), §5.6 (numeric) |
| P7 (Health/observability: minimal extension path without dashboard) | §6 (root cause), §6.2 (fix), §6.3 (metric defs anchored to `baseline.md`), §6.4 (rejected), §6.5 (layer), §6.6 (ownership) |
| P8 (Recall speed: latency-safe constraints + per-fix micro-benchmark guidance) | §7 (root causes), §7.2 (guidance), §7.3 (rejected), §7.4 (per-fix spec), §7.5 (snippets), §9 (binding summary) |
| P9 (`artifacts/ANALYSIS.md` integrates all five areas: + downstream fixture/live-DB policy + AE-vs-LM decisions + numeric thresholds + ownership split) | §2.1 (AE-vs-LM), §2.2 (fixture policy), §8 (ownership + sequencing), §10 (numeric thresholds binding summary) |
| P10 (discovery subtree modifies only artifact files) | Verified by `discovery-artifact-verify` sibling; this synthesis writes only `artifacts/ANALYSIS.md`. |

---

## 13. Summary table for downstream consumers

| Field | #1 Scope hygiene | #2 Dedup | #3 Feedback | #4 Health |
| --- | --- | --- | --- | --- |
| **Root file** | `src/living_memory/storage.py:161` | `src/living_memory/storage.py:140-213` | `src/living_memory/storage.py:1321-1334` + `feedback.py:135-234` | `src/living_memory/resources.py:323-459` |
| **Primary symptom** | bare `rise/*`, `breakthrough/*`, `ocpa-generative-action-substrate-v1/*` traces unreachable; 111 active candidates / 0 recalls | 23.6% duplicate density in `project:ae`; 625 excess active rows | 20.7% `feedback_applied`; 53 / 106 dual-task applied events are wrong-task | `memory_health` does not report feedback ratio, never-accessed ratio, leakage candidates, DB size, instructions, latency |
| **Chosen fix** | `scope = normalize_scope(...)` in `_insert_node` | `content_fingerprint` column + index + supersede/decay branch | Context-aware matcher + raise cap for strong matches + tighten cross-scope fallback | Additive `memory_health` sections + opt-in latency + opt-in instructions |
| **Layer** | LM-internal | LM-internal | LM-internal (primary) | LM-internal |
| **Schema bump** | No | Yes (2 → 3) | No | No |
| **MCP signature change** | No | No | No | Optional new params with safe defaults |
| **Append-only?** | Yes | Yes (soft-decay + supersedes edge) | Yes (write-path metadata only) | Yes (read-only aggregates) |
| **Co-touch** | `_insert_node` (with #2) | `_insert_node` (with #1) | None | None |
| **Latency budget** | Read unchanged; ≤ 5 ms write delta | Unique ≤ 1.0 ms; dup ≤ 2.0 ms; read ≤ 1.05× | Recall unchanged; remember ≤ low single-digit ms | Health ≤ 1.25× |
| **Numeric target** | 0 new bare-scope rows | 0 new dup rows; ≤ 1% steady-state on new scopes | 2 / 2 in fixture; 0 false positives | Match `baseline.md` ratios to ≤ 0.001 |
| **Fixture policy** | tmp-path per test | tmp-path per test; v2 schema fixture for migration test | tmp-path per test (extend `FakeMCP`) | tmp-path per test |
| **Deferred follow-up** | Worktree-aware cwd canon; historical migration | Backfill of 625 dup excess; path-based supersede; AE round-trip optimisation; new health metrics | AE recall metadata enrichment | Register `recall_events_summary` / `connections_summary` resources |
