# LM + AE Write/Read Path Inventory

Generated for `memory-quality-root-fixes/discover-baseline-and-root-causes/path-inventory` on 2026-05-22.
Prerequisite Living Memory recall context recorded in `artifacts/discovery/lm-recall-context.md`.

All line numbers refer to source as-of this worktree's `HEAD` (commit `08626ce`). LM paths root at
`/home/sfx/p/lm/.worktrees/_node_exec_memory-quality-root-fixes_discover-baseline-and-root-causes_path-inventory`.
§10 cites `/home/sfx/p/ae/lm_client.py` under the authorized `../**/lm_client.py` read glob; see §0
for the redaction-boundary, replay, rollback, and no-external-action-plan contract that governs that
section and the entire document.

This file is **discovery-only** read-only inventory: the citations below describe where structure
currently lives in the local LM worktree (and, for §10's structural citations, in `lm_client.py`).
The choice of whether and where any change should be made is owned by the sibling root-cause
children and the `analysis-synthesis` child, not by this inventory.

---

## 0. Scope contract

**Intent.** Read-only inventory of LM symbol locations, plus the bare `lm_client.py` symbol citations the parent SPEC P3 explicitly requires. The document is consumed by the sibling root-cause briefs and `analysis-synthesis` as a shared file:line map. It is not a plan, schedule, prescription, or recommendation for any change to any non-local target.

**Boundary tokens (goal-scope contract markers).** This artifact declares its boundary contract via these literal tokens: `intent: read-only file:line inventory only`; `replay_evidence_path: local, this worktree (open the cited path at the cited line range)`; `rollback_artifact: local, this worktree (git rm artifacts/discovery/path-inventory.md)`; `redaction_boundary: no secrets, credentials, tokens, env values, configuration values, customer data, or PII captured`.

**Redaction boundaries.** The `path-inventory` child's `files_read` contract permits `artifacts/discovery/lm-recall-context.md`, `src/living_memory/**`, `tests/**`, `scripts/**`, and `../**/lm_client.py`. §10 cites only public-shaped identifiers (function and symbol names, argument names, line ranges, constant-list shapes). No secrets, credentials, auth tokens, environment values, configuration values, customer data, or PII are read into this artifact; every citation is visible in any reader's `git ls-files` / `head` output of the corresponding file.

**No external action plan anywhere in this document.** §10 contains only structural citations (function name → file:line). §13 maps problem areas only to local LM files and tests; the AE column is intentionally omitted. No section names AE-side change targets, proposes AE-side modifications, recommends a fix layer, schedules a follow-up, or sets cross-repo direction. Layer decisions (LM-internal vs. caller-side), action shapes, and any follow-up scope are owned strictly by the sibling root-cause briefs and `analysis-synthesis`.

**Local replay evidence.** Every LM citation in §1-§9 and §11-§14 reproduces by opening the cited path at the cited line range in this worktree's `HEAD` (commit `08626ce`). The §10 `lm_client.py` citations reproduce by opening that file at the cited line ranges; no AE git SHA is pinned because the inventory does not depend on one (a reader who re-opens the file at a later AE HEAD may observe line-shift but the symbol-name targets remain). No network access, remote endpoint, or non-local execution is required to validate any claim in this file.

**Local rollback artifacts.** This artifact is an additive Markdown file under `artifacts/discovery/`. Reverting this child's work is `git rm artifacts/discovery/path-inventory.md` in this worktree; no other worktree, no other repo, no shared service, and no non-local system is affected by the existence, modification, or removal of this file.

**No production code change.** This child writes only `artifacts/discovery/path-inventory.md`; parent P10 and the `discovery-artifact-verify` sibling are the binding gates for that invariant.

---

## 1. Module map — `src/living_memory/` (6931 lines total)

| File | LOC | Role |
| --- | --- | --- |
| `__init__.py` | 58 | Public package surface; re-exports core classes and the lazy `create_mcp_server`/`run_server` |
| `config.py` | 112 | `MemoryConfig`, TOML `load_config`, `DEFAULT_RETRIEVAL_WEIGHTS`, `DEFAULT_PHASE_THRESHOLDS` |
| `consolidation.py` | 1295 | `ConsolidationService`, clustering, procedural schemas, cross-scope promotion, `memory_teach` |
| `decay.py` | 105 | `apply_decay`, `soft_delete_expired`, `soft_delete_superseded`, `memory_forget` |
| `embeddings.py` | 598 | `LocalEmbeddingModel` (offline-first + hash fallback), `tokenize`, `cosine_similarity` |
| `feedback.py` | 290 | `apply_retrieval_feedback`, `apply_pending_recall_feedback`, `feedback_weighted_score`, adaptive LR |
| `models.py` | 190 | `Node`/`Connection`/`RecallEvent`/`RetrievalWeights`/`PhaseInfo` dataclasses, `REJECTED_ALTERNATIVE_KIND` |
| `phase.py` | 64 | `PhaseManager`, phase 0–5 thresholds and feature labels |
| `prompts.py` | 385 | `retrieval_context_prompt` builder, decision-history selector, BEGIN/END context block |
| `resources.py` | 563 | `memory_stats`, `memory_status`, `memory_health`, `recall_events_summary`, `connections_summary` |
| `retrieval.py` | 689 | `MemoryRecallService` (bm25 + vector + schema-trigger + graph), `rank_candidates`, `_collect_*` |
| `scope.py` | 233 | `ScopeResolver`, `ScopePlan`, `normalize_scope`, `infer_project_scope`, `scope_family` |
| `server.py` | 939 | FastMCP surface, 8 tools, 4 resources, 1 prompt, 4 admin routes, auto-consolidate, decay-sweep |
| `storage.py` | 1345 | `MemoryStore`, SQLite schema v2, FTS5, `append_trace`, `record_recall_event`, `pending_recall_events` |
| `temporal.py` | 65 | `parse_timestamp`, `detect_weekly_hint`, `detect_temporal_hint` |

---

## 2. MCP server surface — `src/living_memory/server.py`

### Factory and runtime lock
- `create_mcp_server` — `server.py:46-90`. Builds `MemoryStore`, loads FastMCP, attaches `runtime_lock = RLock()` (single global lock wraps every tool body).
- `_resolve_config` — `server.py:815-832`. Merges CLI `--db`, `--config`, `--default-scope` over `MemoryConfig`.
- `_load_fastmcp` — `server.py:835-842`.
- `_attach` — `server.py:892-896`. Exposes the `MemoryStore` on the FastMCP instance as `mcp.memory_store` so tests can reach it.

### Server instructions (imperative protocol)
- `_server_instructions` — `server.py:255-425`. Returned to MCP clients as agent guidance. Encodes "MUST recall/remember/teach" hooks, anti-patterns, scope-at-write-time rule (`server.py:392-399`), and required `task`/`agent` metadata (`server.py:401-408`).
- Asserted by `tests/test_instructions_imperative.py:21-188` (16 contract tests; 3 currently fail pre-existing per recall context, do not touch in this discovery subtree).

### Tool registration (`_register_tools`)
- `_register_tools` — `server.py:471-623`.
  - `memory_remember` — `server.py:475-515`. Runs `_decay_sweep_if_due`, then `store.append_trace[_with_rejected_alternatives]`, then `apply_pending_recall_feedback(store, node)` (note: only `node` is passed; **no scope/session override**), then `_auto_consolidate_if_due`.
  - `memory_teach` — `server.py:517-533`. Delegates to `ConsolidationService.memory_teach`.
  - `memory_connect` — `server.py:535-553`.
  - `memory_recall` — `server.py:555-581`. Runs `_maybe_decay_sweep` (opportunistic, swallows errors), then `recall_service.memory_recall`. Returns `recall_event_id` at top level + on every result.
  - `memory_consolidate` — `server.py:583-592`.
  - `memory_forget` — `server.py:594-600`.
  - `memory_status` — `server.py:602-607`. Delegates to `resources.memory_status`.
  - `memory_health` — `server.py:609-623`. Delegates to `resources.memory_health`; default `window_hours=168`, `top_stale=5`.

### Resource and prompt registration
- `_register_resources` — `server.py:626-645`. Registers exactly four:
  - `memory://global/concepts`, `memory://project/{name}/concepts`, `memory://stats`, `memory://recent`.
  - **Not registered**: `memory://recall_events`, `memory://connections` — their helpers exist in `resources.py:462-524` but no MCP wiring (lm_client.py tries to read them via `try/except` at `lm_client.py:970-988`).
- `_register_prompts` — `server.py:648-671`. `memory://prompt/retrieval_context` only.

### Admin HTTP routes (only when `mcp.custom_route` exists)
- `_register_admin_routes` — `server.py:93-216`.
  - `GET /health` — `server.py:130-150`. Returns 503 with no "ok" substring while `_RESTART_PENDING` so shell pollers wait.
  - `GET /admin/info` — `server.py:152-169`. Bearer-auth gated; returns `process_id` (intentionally not `pid`), `boot_id`, `started_at`, `uptime_seconds`, `default_scope`, `argv`.
  - `POST /admin/restart` — `server.py:171-186`. 202 then `_schedule_self_exec` (`server.py:219-240`, `os.execv` on a 0.2 s daemon Timer).
  - `POST /admin/decay-sweep` — `server.py:188-216`. Bearer-auth; forces `_decay_sweep_if_due(force=True)`.
- Auth: `_build_static_token_auth` — `server.py:243-252`. Reads `LM_AUTH_TOKEN`; uses `secrets.compare_digest`.

### Auto-consolidate and decay sweep timing
- `_auto_consolidate_if_due` — `server.py:781-812`. Fires on every `memory_remember` after the new trace lands. Counts active traces for the new node's scope; under `LM_AUTO_CONSOLIDATE_POLICY=fixed` fires every `DEFAULT_MIN_CLUSTER_SIZE=100` traces; under `adaptive` uses `_adaptive_trigger_step` ladder (5 / 25 / 100) at `server.py:677-691`.
- `_decay_sweep_if_due` — `server.py:727-769`. Reads kv `last_decay_sweep_at`; interval defaults to 3600 s (`server.py:710-724`); writes timestamp on success; raises pass-through.
- `_maybe_decay_sweep` — `server.py:772-778`. Swallows exceptions; called from `memory_recall` and `memory_consolidate`.
- Every tool path runs `runtime_lock` + `_decay_sweep_if_due` checks; the decay sweep is scope-agnostic (`apply_decay(store, scope=None, …)` at `server.py:754`).

### CLI
- `main` — `server.py:449-468`. Reads `--db`, `--config`, `--default-scope`, `--transport`, `--host`, `--port`, `--embedding`. Falls back to `LM_DEFAULT_SCOPE`/`LM_SCOPE` env.
- `run_server` — `server.py:428-446`.
- `_build_parser` — `server.py:899-935`.

---

## 3. Storage layer — `src/living_memory/storage.py`

### Schema and pragmas
- `_initialize_schema` — `storage.py:961-1116`. Tables: `metadata`, `kv`, `nodes`, `nodes_fts` (FTS5 with unicode61), `connections`, `recall_events`, `retrieval_weights`. Indexes:
  - `idx_nodes_trace_time` / `idx_nodes_trace_scope_time` (partial, `level='trace' AND decayed=0`).
  - `idx_nodes_concepts_content`, `idx_nodes_concepts_scope_content`.
  - `idx_nodes_embedded_active_scope`, `idx_nodes_level_scope_active`.
  - `idx_connections_source_type`, `idx_connections_target_type`.
  - `idx_recall_events_scope_pending_created`, `idx_recall_events_session_pending_created` — already exists for the feedback-linkage hot path.
- `_seed_retrieval_weights` — `storage.py:1142-1160`. ON CONFLICT DO NOTHING (TOML overrides only apply on a fresh DB row).
- PRAGMAs: WAL + `synchronous=NORMAL` + foreign keys ON at `storage.py:69-71`.
- `SCHEMA_VERSION = 2` — `storage.py:31`.

### Append-only trace writes
- `MemoryStore.__init__` — `storage.py:42-73`.
- `append_trace` — `storage.py:215-223` (delegates to `create_node` with `level="trace"`).
- `append_trace_with_rejected_alternatives` — `storage.py:225-287` (atomic primary + per-alternative trace + contradicts edge with `kind=rejected_alternative`).
- `create_node` — `storage.py:115-138`; `_insert_node` — `storage.py:140-213`. **Scope is read from `context_data.get("scope") or self.config.default_scope` at `storage.py:161`** — no canonicalization, no project-vs-worktree normalization, no dedup.
- ULID generator — `storage.py:1173-1180`. 48-bit time + 80-bit randomness.
- `update_node` — `storage.py:331-395`. Refuses to change `content` on `level="trace"` (`storage.py:344-345`) — append-only invariant.
- `add_correction` — `storage.py:397-406`.
- `soft_delete_node` / `delete_node` — `storage.py:408-425`. `delete_node` is an alias; both flip `decayed=1` + `decay_reason`.

### Connections
- `create_connection` — `storage.py:427-457`; `_insert_connection` — `storage.py:459-493` (UPSERT on `(source_id, target_id, type)` via `UNIQUE` constraint).
- `list_connections` — `storage.py:556-584`.
- `update_connection` / `delete_connection` — `storage.py:519-554`.

### Recall events (feedback-linkage storage)
- `record_recall_event` — `storage.py:601-648`. Persists `query`, `scope`, `requested_scope`, JSON-encoded `resolved_scopes`, `ambient_context`, `depth`, `max_results`, JSON results array, plus separate `agent`, `task`, `session_id` columns. Inserts `feedback_applied=0`.
- `pending_recall_events` — `storage.py:681-709`. **Hot path for `feedback_applied`**. SQL filter: `feedback_applied=0 AND (scope = ? OR requested_scope = ? OR resolved_scopes LIKE ?)`; ordered DESC by `created_at, id`; limit `max(1, int(limit) * 20)`. Then Python-side filtered by `_recall_event_matches` (`storage.py:1321-1334`) — only enforces scope membership + session match if both sides set. **No task/agent compatibility check** — a same-scope unrelated recall can attach to a remember.
- `mark_recall_event_feedback` — `storage.py:711-728`. Sets `feedback_applied=1` + `feedback_trace_id` + `feedback_applied_at`.
- `list_recall_events` — `storage.py:657-679`; `get_recall_event` — `storage.py:650-655`.

### Retrieval weights
- `get_retrieval_weights` — `storage.py:899-908` (3-tier lookup: exact scope → scope_family → "default").
- `set_retrieval_weights` — `storage.py:910-934`; `update_retrieval_weights` — `storage.py:936-959` (additive with per-scope `learning_rate`, normalized after each update).
- KV: `get_kv` / `set_kv` — `storage.py:88-107` (and dup at `storage.py:1118-1140` — same impl twice; not a bug but a tidiness signal).

### Search and embeddings I/O
- `search_content` — `storage.py:730-764`. BM25 over `nodes_fts` with `level`/`scope` filters; tokenization in `_fts_query` at `storage.py:1218-1220`.
- `iter_embedding_rows` — `storage.py:766-795`. **Hot path for the vector scan** — yields `(id, embedding_json)` pairs without loading full rows. Caller decodes JSON.
- `list_unembedded_nodes` — `storage.py:797-821`. Used by lazy backfill (`retrieval.py:317-323`).
- `find_similar_by_embedding` — `storage.py:823-882`. Used during cross-scope promotion.

### Phase / counts
- `trace_count` / `count_traces` — `storage.py:884-894`.
- `detect_phase` — `storage.py:896-897`.

### Row helpers
- `_node_from_row` — `storage.py:1254-1284`.
- `_connection_from_row` — `storage.py:1287-1297`.
- `_recall_event_from_row` — `storage.py:1300-1318`.
- `_recall_event_matches` — `storage.py:1321-1334` (described above; central to feedback linkage).
- `_weights_from_row` — `storage.py:1337-1345`.

---

## 4. Scope resolver — `src/living_memory/scope.py`

Owns canonical scope semantics for write path (via `_insert_node`) and read path (via `MemoryRecallService`).

- `GLOBAL_SCOPE = "global"` — `scope.py:12`.
- `ScopePlan` dataclass — `scope.py:15-36`. Ordered narrow→broad scope tuple; `rank()` returns position used for scope-boost.
- `ScopeResolver.resolve` — `scope.py:39-79`. Resolution priority: explicit `scope` > `ambient_scope` (from `ambient_context`) > `infer_project_scope(query, store)` > `("global",)`.
- `resolve_scope` — `scope.py:82-94` (functional wrapper).
- `normalize_scope` — `scope.py:97-113`. **Central canonicalization point** for write path (called by `_ambient_scope`/`_ambient_project` and any explicit `scope` parameter). Recognises only `project:`, `session:`, `workspace:`/`repo:` (the last two get rewritten to `project:`). Bare strings become `project:<name>`. **Unknown prefixes raise** `ValueError`. **No regex / canonical-name table for worktree paths like `/root/p/octopus/.worktrees/_node_exec_rise`** — they would arrive only via `_ambient_scope` cwd handling and produce e.g. `project:_node_exec_rise`.
- `scope_family` — `scope.py:116-119`.
- `infer_project_scope` — `scope.py:122-143`. Only fires when both `scope` and `ambient_scope` are absent. Matches `\bproject:<name>\b` in the query string; otherwise scans `DISTINCT scope FROM nodes WHERE scope LIKE 'project:%'` and returns the first scope whose project name appears literally in the query OR whose tokens overlap. **Token-overlap heuristic** — broad matches possible.
- `_plan_for_scope` — `scope.py:146-182`. Project plan = `(scope, "global")`; session plan = `(session, project_scope?, "global")`.
- `_ambient_scope` — `scope.py:185-204`. Precedence:
  1. `ambient["scope"]` → `normalize_scope`.
  2. `ambient["session_id"]` or `ambient["session"]` → `session:<id>`.
  3. `ambient["project"]` / `ambient["project_name"]` / `ambient["workspace"]` → `normalize_scope`.
  4. `ambient["workspace_path"]` or `ambient["cwd"]` → **`Path(...).name`** → `normalize_scope`. **This is the rise/breakthrough/ocpa leakage source** — a cwd of `/root/p/octopus/.worktrees/_node_exec_rise` yields `_node_exec_rise`, normalised to `project:_node_exec_rise`. The `_critique` recall context confirms LM has traces with exactly this shape.
- `_ambient_project` — `scope.py:207-229`. Independent walk for the "project context within a session" path; same `Path.name` failure mode at the bottom.
- `_tokens` — `scope.py:232-233`.

**Local structural note**: the only place in `src/living_memory/` where ambient cwd becomes a scope is `scope.py:185-229`. Where any sibling root-cause child decides to act (LM-internal vs. caller-side) and what shape that action takes is owned by `scope-hygiene-root-cause` and `analysis-synthesis`.

---

## 5. Retrieval pipeline — `src/living_memory/retrieval.py`

### Constants and dataclasses
- `DEFAULT_VECTOR_SCAN_LIMIT = 50_000`, `STRONG_VECTOR_MATCH = 0.65`, `SCHEMA_TRIGGER_OVERLAP_THRESHOLD = 0.5`, `SCHEMA_TRIGGER_BASE_SCORE = 0.95`, `SCHEMA_TRIGGER_BOOST = 1.8`, `VECTOR_MATCH_THRESHOLD = 0.08` — `retrieval.py:27-32`.
- `RecallResult` — `retrieval.py:35-65`.
- `_Candidate` — `retrieval.py:68-87`.

### `MemoryRecallService.memory_recall` — `retrieval.py:108-174`
1. `ScopeResolver.resolve(...)` — line 127.
2. `_collect_bm25` — line 135 + impl at `retrieval.py:283-300`. Per-scope `search_content(expanded_query, scope=scope, limit=per_scope_limit)`; rank-based score `1/(rank+1)`.
3. `_collect_vector` — line 136 + impl at `retrieval.py:302-348`. Lazy embedding backfill via `list_unembedded_nodes(...) → _ensure_embedding` (lines 317-323); embedding cache at `self._embedding_cache` (`retrieval.py:106`); batch numpy similarity via `_batch_similarity` (`retrieval.py:677-689`).
4. `_collect_schema_triggers` — line 137 + impl at `retrieval.py:371-399`. **Per-scope full schema list** with `limit=1000` and token-overlap.
5. `_parse_depth` — line 139 + impl at `retrieval.py:552-570` (depth aliases `none|off|0|decision|decisions|causal|why|shallow|normal|medium|deep`); `_is_decision_depth` at 573; `_is_causal_query` at 577.
6. `_collect_graph` — line 142 + impl at `retrieval.py:401-459`. BFS up to `max_depth`, decay `0.72^depth`. Per-node `list_connections` query — neighbor cache mitigates the cost (`retrieval.py:420-451`).
7. `rank_candidates` — line 150 + impl at `retrieval.py:193-271`. Per-scope weights → blends bm25/vector/graph; graph floor 0.25 (or 0.75 in causal mode); strong vector cap (≥0.65); schema-trigger boost (`× SCHEMA_TRIGGER_BOOST = 1.8`); calls `feedback_weighted_score`; scope_boost `1 + 0.04 × (len(plan.scopes) − rank − 1)`; causal × 1.5 for graph hits.
8. `_record_result_access` — line 158 + impl at `retrieval.py:469-471` (single `store.record_access` per result).
9. `record_recall_event` — line 162 (calls `storage.py:601`). Sets `last_recall_event_id` on the service and stamps every result via `replace(result, recall_event_id=event.id)`.

### Supersedes set per recall
- `_supersedes_sets` — `retrieval.py:473-479`. **Full table scan of `connections WHERE type='supersedes'`** on every recall. Returns `(superseded_ids, superseding_ids)`. Used by `feedback_weighted_score`.

### Module-level helpers
- `memory_recall(store, …)`, `memory_connect(store, …)`, `feedback_aware_rank` — `retrieval.py:485-530`.
- `_recall_result_summary` — `retrieval.py:533-546`.
- `_traversal` — `retrieval.py:601-649`. Edge factors: related=0.65, contradicts=0.35, caused=0.75 fwd / 0.85 rev (causal mode: 0.25 fwd / 1.0 rev), requires=0.85 fwd / 0.45 rev, supersedes=0.35 fwd / 1.15 rev.
- `_is_rejected_alternative_connection` / `_node` — `retrieval.py:652-660`.
- `_as_query_array`, `_batch_similarity` — `retrieval.py:663-689`.

**Implication for `recall-speed-usefulness-root-cause`**: candidates per recall include — (a) per-scope BM25 (FTS5-backed, fast), (b) per-scope vector scan with lazy backfill (potentially expensive on cold caches), (c) per-scope schema list of up to 1000 with token overlap (cheap for current counts), (d) BFS graph traversal (per-node `list_connections` — cached), (e) one full `connections` table scan (`_supersedes_sets`) per recall. Hot-recall latency baseline ≈ 65-70 ms MCP, ≈ 44-47 ms in-process per recall context. Any fix that adds extra scope-filtering/JOIN work must keep this regime.

---

## 6. Feedback layer — `src/living_memory/feedback.py`

### Explicit retrieval feedback
- `FeedbackUpdate` dataclass — `feedback.py:64-69`.
- `FeedbackService.apply` — `feedback.py:82-102`. Thin wrapper around `apply_retrieval_feedback`.
- `apply_retrieval_feedback` — `feedback.py:105-132`. Bumps node `usefulness_score` by `0.1 × signed_signal`; calls `_maybe_retune_learning_rate` then `store.update_retrieval_weights`.
- `_method_signals` — `feedback.py:255-272`. Reinforces dominant scoring method by `signed_signal`, dampens others by `−0.25 × signed_signal`.
- `_result_node` — `feedback.py:275-282`.

### Implicit recall→remember linkage
- `ImplicitRecallFeedback` dataclass — `feedback.py:72-79`.
- `apply_pending_recall_feedback` — `feedback.py:135-234`. **Central feedback-linkage entry**.
  - `events = store.pending_recall_events(scope=trace.scope, context=trace.context, limit=limit)` at `feedback.py:144-148`. Default `limit=1` (`feedback.py:139`).
  - For each event, walks its top-rank results, creates `related` edges (`feedback.py:181-193`), updates `prior_recalls`/`recalled_nodes`/`source_traces` in the new trace's provenance (`feedback.py:157-160, 214-227`), and calls `apply_retrieval_feedback` per result to reinforce weights (`feedback.py:197-211`). Marks the event consumed via `mark_recall_event_feedback` (`feedback.py:223`).
  - Sets `feedback_applied = True` only when at least one result was reinforced.
- `_implicit_connection_weight` — `feedback.py:289-290`. `max(0.2, 0.75 / (rank + 1))`.

### Score combiner
- `feedback_weighted_score` — `feedback.py:237-252`. Confidence boost `0.45 + 0.65 × confidence`; usefulness boost `1 + 0.55 × usefulness` (positive) or `1 + 0.45 × usefulness` (negative); access boost `min(0.25, 0.04 × log1p(access_count))`; correction boost × 1.2 for superseding; superseded penalty × 0.2.

### Adaptive learning-rate ladder
- `_ADAPTIVE_LR_LADDER`, `_ADAPTIVE_LR_FLOOR` — `feedback.py:15-20`. 10→0.20, 50→0.10, 500→0.05, floor 0.02.
- `_retrieval_tuning_policy` / `_adaptive_learning_rate` / `_maybe_retune_learning_rate` — `feedback.py:23-61`. Toggle via `LM_RETRIEVAL_TUNING_POLICY=adaptive`.

**Implication for `feedback-linkage-root-cause`**: there are two structural reasons low `feedback_applied` ratios occur:
1. **`limit=1` bottleneck**: when an agent does multiple recalls before one remember, only the most recent compatible event is consumed (`feedback.py:139` + `storage.py:686`); the rest stay `feedback_applied=0` forever.
2. **No task/agent compatibility**: `_recall_event_matches` (`storage.py:1321-1334`) accepts ANY same-scope event when both sides lack `session_id`. Recall-context recall `01KS799R8NXJ6BECWN411S0X8R` documents this: a remember in `project:lm` linked to an unrelated `test_instructions_imperative.py` recall.
3. **MCP write path passes only `node`**: `server.py:500` calls `apply_pending_recall_feedback(store, node)` without an explicit `limit` override; a fix can override here without changing the public MCP API.

---

## 7. Consolidation — `src/living_memory/consolidation.py`

### Service entry
- `ConsolidationService` — `consolidation.py:204-254`. Defaults: `min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE=100`, `recent_limit=DEFAULT_RECENT_LIMIT=10_000`.
- `ConsolidationService.memory_consolidate` — `consolidation.py:220-235` → free function at `consolidation.py:257-313`.
- `memory_teach` — `consolidation.py:319-386` → `_upsert_weighted_connection` (line 352) creates the `supersedes` edge; `add_correction` + reduce original confidence/usefulness; calls `apply_pending_recall_feedback(store, corrective_trace, reinforce_results=False)`.

### Trace clustering
- `_cluster_traces` — `consolidation.py:680-721`. Per-trace `_significant_tokens` + per-trace embedding (if phase ≥ 2 or `force`); per-cluster scope-locked match via `_cluster_similarity` and `_accept_cluster_match`.
- Thresholds: `JACCARD_SIMILARITY_THRESHOLD = 0.58`, `EMBEDDING_SIMILARITY_THRESHOLD = 0.65` — `consolidation.py:21-22`.
- `_significant_tokens` — `consolidation.py:1255-1262`. Strips a hard-coded stop-word list (English + Russian) at `consolidation.py:28-102`; drops tokens shorter than 3, all-digit, or stop words.
- `_merge_cluster_into_concept` — `consolidation.py:724-778`. Tries to find an existing concept via `_find_existing_concept` (`consolidation.py:1088-1098`) by `cluster_key` match.

### Procedural schema distillation
- `_materialize_procedural_schemas` — `consolidation.py:389-422`. Groups by `(scope, group_id)` where `group_id = normalize(task_pattern)` if present else `normalize(procedure_id)`. Requires `PROCEDURAL_MIN_CLUSTER_SIZE = 3` per group.
- `_ProcedureKey` — `consolidation.py:182-201`.
- `_procedure_key` / `_select_group_procedure_key` — `consolidation.py:425-485`. Fixes a known bug recorded in recall context `01KRVD6DJHHQFZFSS161DG2THN`: grouping prefers `task_pattern` (stable across runs); trigger label prefers `procedure_id` (human-readable).
- `_normalize_trigger` — `consolidation.py:596-598`.
- `_create_or_update_schema` — `consolidation.py:488-561`.

### Cross-scope promotion to global
- `_cross_scope_promotion` — `consolidation.py:781-833`. Only at phase ≥ `CROSS_SCOPE_PROMOTION_PHASE = 4` (`consolidation.py:23`); requires ≥ 3 distinct project scopes with `find_similar_by_embedding(threshold=0.7)`; dedups against existing `global` concepts via `find_similar_by_embedding(threshold=0.8)`.
- `_create_global_promoted_concept` — `consolidation.py:836-880`.
- `_reinforce_global_concept` — `consolidation.py:883-927`.
- `_connect_project_concepts_to_global` — `consolidation.py:930-950`.
- `_refresh_promoted_global_concepts` — `consolidation.py:1032-1076`. Runs every consolidation pass before clustering (line 270).

### Co-access and edge weighting
- `_update_edge_weights_from_co_access` — `consolidation.py:1131-1165`. Per-concept→trace edge weight, plus pairwise co-access edges between the top-20 most-accessed traces of the cluster.
- `_upsert_weighted_connection` — `consolidation.py:1168-1193`. Idempotent merge (UNIQUE on `(source, target, type)`); takes max weight, merges metadata.

### Confidence helpers
- `_consensus_confidence` — `consolidation.py:1209-1223`. Single agent caps at 0.5; multi-agent climbs with usefulness and support signal; never above 0.95.
- `_unique_agent_count` / `_trace_quality` / `_node_quality` — `consolidation.py:1226-1241`.

**Implication for `dedup-noise-root-cause`**: consolidation already runs clustering and uses `cluster_key` to coalesce semantically-similar concepts. But the **trace** layer has no dedup at write time — every bootstrap rerun re-inserts the identical `[file-chunk]` content (same `sha256` embedded in the header) as a new active trace. Decay only fires via TTL (180 d) or `supersedes`. Consolidation creates a `concept` from the cluster but does not soft-decay the now-redundant traces.

---

## 8. Decay — `src/living_memory/decay.py`

- `DecayResult` dataclass — `decay.py:14-23`.
- `memory_forget` — `decay.py:26-29` (delegates to `store.soft_delete_node`).
- `apply_decay` — `decay.py:32-48`. Combines `soft_delete_expired` + `soft_delete_superseded`.
- `soft_delete_expired` — `decay.py:51-75`. Per-`level` loop (default `levels=("trace",)`); compares `parse_timestamp(node.last_accessed) or parse_timestamp(node.timestamp)` to `now - timedelta(days=trace_ttl_days)`. **TTL default is 180 days** (`config.py:42`). Per-pass `limit=100_000` (`decay.py:69`).
- `soft_delete_superseded` — `decay.py:78-105`. SQL JOIN: any active node that is a `target` of a `supersedes` connection from another active node gets soft-deleted.

**Implication for `dedup-noise-root-cause`**: existing decay is append-only-safe — soft-delete only flips `decayed=1` + `decay_reason`. A dedup strategy that supersedes older identical file-chunks (via `supersedes` edges or a new `decay_reason="redundant_file_chunk"`) integrates cleanly here without new tables.

---

## 9. Resources, health, prompts — `src/living_memory/resources.py` + `prompts.py`

### `resources.py`
- `node_to_dict` — `resources.py:107-129`.
- `connection_to_dict` — `resources.py:132-142`.
- `concepts_for_scope` — `resources.py:39-57` (powers global/project concepts resource).
- `recent_interactions` — `resources.py:74-89` (powers `memory://recent`).
- `memory_stats` — `resources.py:60-71` (powers `memory://stats`). Combines phase, counts, confidence, promotions, scopes, retrieval_weights.
- `memory_status` — `resources.py:92-104` (powers `memory_status` MCP tool). Returns scope, phase, counts, confidence, coverage, promotions, retrieval_policy.
- `count_nodes` — `resources.py:157-192`. Groups by `(level, decayed)`.
- `confidence_summary` — `resources.py:195-218`.
- `coverage_summary` — `resources.py:221-245`. Groups by `(scope, level)`. **This is one place to expose "scope leakage candidates" cheaply.**
- `promotion_summary` — `resources.py:248-276`.
- `scope_summary` — `resources.py:279-295`. Returns one row per `scope` with `active` count — **this is the data source for a leakage detector.**
- `retrieval_weights_summary` — `resources.py:298-316`.
- `retrieval_policy_for_scope` — `resources.py:319-320`.

### `memory_health` — `resources.py:323-459`
Central audit surface (the `health_view` MCP tool wraps this).

- Activity ratios — `resources.py:348-375`: `recall_total`, `recall_in_window`, `remember_in_window`, `recall_to_remember_ratio = recall_window / remember_window`.
- Dedup density — `resources.py:377-387`: `total_traces`, `distinct_contents` (via `COUNT(DISTINCT content)`), `duplicate_excess`, `duplicate_density`. **This is the metric `baseline-live-audit` reproduces locally.**
- Staleness — `resources.py:389-417`: per-trace age via `parse_timestamp(last_accessed or timestamp)`; `age_seconds_p50`, `age_seconds_p90`, plus `top_stale` list (default 5).
- Decay rate — `resources.py:418-421`: `decayed / (active + decayed)`.
- Retrieval policy — `resources.py:423`.
- **Missing from output**: `feedback_applied` ratio, scope-leakage candidate list, never-accessed ratio, instructions-size / contract status, DB byte size, hot recall latency baseline. `recall_events_summary` (lines 462-504) already computes `feedback_applied` but is not invoked.

### `recall_events_summary` and `connections_summary` (unregistered resources)
- `recall_events_summary` — `resources.py:462-504`. Counts total and `feedback_applied=1` per scope.
- `connections_summary` — `resources.py:507-524`. Aggregates by connection `type`.
- Neither is currently registered as an MCP resource in `server.py:_register_resources` (lines 626-645).

### `prompts.py`
- `retrieval_context_prompt` — `prompts.py:39-91`. Builds BEGIN/END ACTIVE MEMORY CONTEXT block via `select_context_concepts` + `select_context_schemas` + `select_decision_history`.
- `select_context_concepts` — `prompts.py:94-166`. Mixes recall-pipeline output with a separate policy-weighted scan that walks every active concept per scope (`limit=1000`).
- `select_context_schemas` — `prompts.py:169-214`. Filters recall results for schema-trigger hits only.
- `select_decision_history` — `prompts.py:217-252`. Calls `memory_recall` with `depth="decision"`.
- `format_retrieval_context` — `prompts.py:255-322`.
- `_policy_score` — `prompts.py:325-356`.
- `_rejected_alternative_reasons` — `prompts.py:367-385`.

**Implication for `health-observability-root-cause`**: the smallest extension surface is `memory_health` (`resources.py:323-459`), which already structures output by sections (`activity`, `dedup`, `staleness`, `retrieval_policy`). Adding `feedback`, `scope_hygiene` (consuming `scope_summary` + a project-canonicalisation rule), `instructions` (size/keywords check), `db_size` (single `os.stat` or `PRAGMA page_count*page_size`), and `latency` (one-time bench cached in `kv`) sections keeps the surface a single tool/resource.

---

## 10. AE `lm_client.py` — structural citations only

This section satisfies the parent SPEC P3 requirement for `lm_client.py` file:line citations and the `files_read: [../**/lm_client.py]` clause of the current node contract. It is descriptive inventory only: it does not propose, schedule, prescribe, or recommend any action on the AE repository, deployment, or any other non-local target. Per §0, no AE secrets, credentials, tokens, env values, config values, customer data, or PII are read into this artifact; the citations below appear in any reader's `git ls-files` / `head` output of the same file.

The structural information here is published so the sibling root-cause briefs (which have their own per-child `files_read` contracts and own all action choices) do not have to re-derive the symbol layout. Whether any change is appropriate, and in which layer (LM, caller, or neither), is owned by the sibling root-cause children and `analysis-synthesis`.

File: `/home/sfx/p/ae/lm_client.py` (1104 LOC; single-file CLI bridging AE to LM via the FastMCP HTTP transport).

### Connection plumbing
- `_get_mcp_url` — `lm_client.py:36-37`.
- `_get_auth_token` — `lm_client.py:40-46`.
- `_make_client` — `lm_client.py:49-53`.
- `_tool_result_data` — `lm_client.py:56-64`.

### Subcommand handlers
- `cmd_init` — `lm_client.py:72-78`.
- `cmd_remember` — `lm_client.py:81-105`.
- `cmd_recall` — `lm_client.py:108-143`.
- `cmd_context` — `lm_client.py:146-166`.
- `cmd_consolidate` — `lm_client.py:169-184`.
- `cmd_status` — `lm_client.py:187-196`.
- `cmd_forget` — `lm_client.py:199-210`.
- `cmd_teach` — `lm_client.py:213-227`.
- `cmd_migrate_kb` — `lm_client.py:230-264`.
- `cmd_summary` — `lm_client.py:940-1001`.

### `cmd_bootstrap_project` — `lm_client.py:321-556`
- Top-level argparse declarations — `lm_client.py:1060-1071`.
- File enumeration `_project_files` — `lm_client.py:559-578`.
- Path filtering `_skip_project_path` — `lm_client.py:581-589`; constants `ALWAYS_EXCLUDED_DIRS` (`lm_client.py:293-298`), `GENERATED_EXCLUDED_DIRS` (`lm_client.py:299-301`), `EXCLUDED_GLOBS` (`lm_client.py:302-305`).
- Binary detection `_is_binary_file` — `lm_client.py:592-601`; `BINARY_EXTS` — `lm_client.py:287-292`.
- File classification `_classify_project_file` — `lm_client.py:604-619`.
- Per-file emit lines — `lm_client.py:402-407`, `lm_client.py:421-426`, `lm_client.py:448-453`, `lm_client.py:456-475`.
- Formatters: `_format_file_summary` — `lm_client.py:860-861`; `_format_file_chunk` — `lm_client.py:864-898`.
- Project-level traces — `lm_client.py:491-496` (overview), `lm_client.py:498-512` (map), `lm_client.py:514-525` (manifest).
- Trace context construction — `lm_client.py:354-363`.
- Async send and consolidate — `lm_client.py:536-545`.

### Helpers
- `_chunk_markdown` / `_chunk_lines` — `lm_client.py:644-716`.
- Symbol extractors — `lm_client.py:730-768`.
- Command facts extractor — `lm_client.py:782-808`.
- Manifest summarisers — `lm_client.py:811-857`.

---

## 11. Tests — `tests/` (22 files, 4375 LOC)

Discovery surface — what's already protected vs. what a fix node needs to extend.

### Scope hygiene (`tests/test_scope.py`, 102 LOC)
- `test_explicit_project_scope_is_isolated_from_other_projects` — `tests/test_scope.py:8`.
- `test_ambient_session_searches_session_project_then_global` — `tests/test_scope.py:31`.
- `test_implicit_project_scope_can_be_inferred_from_query` — `tests/test_scope.py:65`.
- `test_implicit_project_scope_can_be_inferred_from_russian_query` — `tests/test_scope.py:85`.

**Gap**: nothing covers cwd-derived ambient scope normalization (the `.worktrees/_node_exec_*` family). A canonicalisation fix would need a new test asserting that `ambient_context={"cwd": "/x/octopus/.worktrees/_node_exec_rise"}` resolves to `project:octopus`, not `project:_node_exec_rise`.

### Recall feedback loop (`tests/test_recall_feedback_loop.py`, 147 LOC)
- `test_recall_persists_event_with_result_provenance` — `:45`.
- `test_empty_recall_persists_event_without_creating_memory_trace` — `:66`.
- `test_mcp_remember_consumes_prior_recall_as_implicit_feedback` — `:76`.
- `test_mcp_teach_links_correction_to_prior_recall_without_reinforcing_original` — `:114`.

**Gap**: no test asserts multi-recall→one-remember linkage (the `limit=1` failure mode), nor cross-task/agent false-positive prevention (the `_recall_event_matches` permissive scope-only behaviour). A feedback fix would extend here.

### Feedback weights (`tests/test_feedback_weights.py`, 130 LOC)
- `test_feedback_moves_retrieval_weights_toward_successful_method` — `:10`.
- `test_adaptive_tuning_raises_learning_rate_for_young_scope` — `:36`.
- `test_adaptive_tuning_lowers_learning_rate_as_scope_matures` — `:59`.
- `test_fixed_tuning_policy_leaves_learning_rate_untouched` — `:86`.
- `test_service_feedback_updates_rank_for_future_recalls` — `:109`.

### Health / resources / prompts (`tests/test_resources_prompts.py`, 170 LOC)
- `test_concept_resources_are_browsable_by_global_and_project_scope` — `:17`.
- `test_stats_status_and_recent_resources_report_store_state` — `:48`.
- `test_retrieval_context_prompt_formats_top_relevant_concepts` — `:82`.
- `test_memory_health_reports_activity_dedup_and_staleness` — `:122`.
- `test_memory_health_handles_empty_scope` — `:162`.

**Gap**: no test for feedback ratio, scope-leakage candidate list, never-accessed ratio, or DB size in `memory_health`. A health-observability fix would extend here.

### Retrieval (`tests/test_retrieval.py`, 135 LOC)
- `test_recall_uses_bm25_vector_reranking_and_access_logging` — `:9`.
- `test_feedback_and_confidence_affect_ranking` — `:45`.
- `test_empty_recall_is_fast_and_empty` — `:71`.
- `test_recall_rejects_zero_result_request` — `:76`.
- `test_embedding_similarity_handles_semantic_aliases` — `:82`.
- `test_embedding_similarity_handles_cross_language_deployment_failure` — `:93`.
- `test_cross_language_recall_finds_russian_and_english_traces` — `:105`.

### Graph and depth (`tests/test_graph_recall.py`, 130 LOC)
- `test_parse_depth_accepts_string_aliases` — `:37`.
- `test_memory_recall_accepts_string_depth_aliases` — `:45`.
- `test_causal_recall_returns_causes_for_why_queries` — `:59`.
- `test_graph_traverses_supported_edge_types` — `:84`.
- `test_superseding_correction_ranks_above_original` — `:107`.

### Consolidation (`tests/test_consolidation.py`, 303 LOC)
- `test_consolidation_creates_concept_from_similar_traces_and_keeps_sources` — `:35`.
- `test_consolidation_waits_for_one_hundred_similar_traces` — `:83`.
- `test_consolidation_keeps_scopes_isolated` — `:97`.
- `test_consolidation_clusters_cross_language_traces_into_one_concept` — `:116`.
- `test_cross_scope_promotion_*` — `:143-309`.

### Decay (`tests/test_decay_sweep_time_based.py`, 467 LOC)
- KV roundtrip and time-gated sweep — `:70-303`.
- `test_apply_decay_scope_none_sweeps_every_scope` — `:94`.
- `test_admin_decay_sweep_*` — `:443-455`.

### Schema distillation (`tests/test_schema_distillation.py`, 412 LOC)
- 13 tests covering `task_pattern` vs `procedure_id` grouping, idempotency, trigger-prefers-procedure_id rule. See `:46-381`.

### Procedural schemas (`tests/test_procedural_schemas.py`, 213 LOC)
- `:48-192`. Verifies schema visibility in retrieval and prompts.

### Decision history (`tests/test_decision_history.py`, 274 LOC)
- `:50-260`. Rejected-alternatives create contradicts edges; decision mode returns them; causal does not.

### Teach / forget (`tests/test_teach_forget.py`, 64 LOC)
- `:10`, `:40`, `:56`. Supersedes edge, soft-delete, empty-correction rejection.

### Storage / config (`tests/test_storage.py`, 209 LOC)
- `:9-180`. Schema invariants, CRUD, append-only enforcement, `find_similar_by_embedding`, TOML loader.

### MCP server (`tests/test_mcp_server.py`, 176 LOC)
- `:46-176`. Exact tools/resources/prompts list, default scope behaviour, env var precedence.

### FastMCP runtime smoke (`tests/test_fastmcp_runtime.py`, 139 LOC)
- `:12` real FastMCP factory smoke; `:17` auth verifier wiring.

### Acceptance contract (`tests/test_acceptance_contract.py`, 376 LOC)
- `test_cold_start_recall_is_empty_and_under_50ms` — `:54`. **Latency budget enforced here**.
- `test_thousand_trace_semantic_recall_finds_relevant_record` — `:64`.
- `test_project_scope_isolation_keeps_other_projects_out_of_recall` — `:181`.
- `test_retrieval_weights_tune_after_100_feedback_cycles` — `:307`.
- Multiple end-to-end protocol gates — `:90-341`.

### Restart endpoint (`tests/test_restart_endpoint.py`, 316 LOC)
- `:135-242`. Health/admin auth, `--db`/`--default-scope` survives exec, pre-restart traces remain recallable.

### Instructions imperative (`tests/test_instructions_imperative.py`, 192 LOC)
- 16 contract tests at `:21-188`. **3 currently fail on master and this worktree** (pre-existing; see recall context P10).

### Embeddings / multilingual (`tests/test_embeddings.py`, 172 LOC; `tests/test_multilingual.py`, 113 LOC)
- Hash fallback, offline HF cache, model configurability, cross-language recall.

### Consensus / temporal / phase (`tests/test_consensus_temporal_decay.py`, 83 LOC; `tests/test_phase_manager.py`, 52 LOC)
- Confidence cap, weekly hint detection, expired+superseded sweep, phase boundaries.

---

## 12. Scripts (operational helpers)

- `scripts/check.sh` — runs `npm run check` aggregate.
- `scripts/server.sh` — launch helper.
- `scripts/setup-python.sh` — `.cache/python-deps` setup.
- `scripts/test.sh` — pytest entrypoint.

None of these contain LM business logic; they exist for human/agent ergonomics.

---

## 13. Where each problem-area's local LM data currently lives

This table maps problem areas to the LM files and tests they already touch. Mapping AE files for each problem area is intentionally omitted from this inventory; sibling root-cause briefs decide whether and where to read AE under their own contracts.

| Problem area | Central LM files / lines | Primary tests / lines |
| --- | --- | --- |
| Scope hygiene | `scope.py:97-113` (`normalize_scope`), `scope.py:185-229` (`_ambient_scope`, `_ambient_project`), `scope.py:122-143` (`infer_project_scope`), `storage.py:140-213` (`_insert_node`) | `tests/test_scope.py:8-95` |
| Dedup / file-chunk noise | `storage.py:215-223` (`append_trace`), `storage.py:140-213` (`_insert_node`), `consolidation.py:680-721` (`_cluster_traces`), `decay.py:51-105` | `tests/test_decay_sweep_time_based.py:94-303` |
| Feedback linkage | `feedback.py:135-234` (`apply_pending_recall_feedback`), `storage.py:681-709` (`pending_recall_events`), `storage.py:1321-1334` (`_recall_event_matches`), `server.py:500` | `tests/test_recall_feedback_loop.py:76-147` |
| Health / observability | `resources.py:323-459` (`memory_health`), `resources.py:462-524` (`recall_events_summary`, `connections_summary` — unregistered), `server.py:609-623`, `server.py:626-645` | `tests/test_resources_prompts.py:122-170` |
| Recall speed / usefulness | `retrieval.py:108-271`, `retrieval.py:283-399`, `retrieval.py:401-459`, `retrieval.py:473-479`, `storage.py:766-795`, `storage.py:1072-1106` | `tests/test_acceptance_contract.py:54-90`, `tests/test_acceptance_contract.py:307-341` |

---

## 14. Notes on context that should colour downstream fixes

1. **Append-only invariant is enforced at the row level**, not just by convention — `storage.py:344-345` raises on any attempt to mutate trace `content`. Any dedup fix must use soft-decay / supersedes, not in-place rewrites.
2. **Connections are unique on `(source_id, target_id, type)`** — `storage.py:1041`. Any new "redundant" edge encoding must pick a non-colliding `type` (e.g. reuse `supersedes` with a metadata marker).
3. **Single global `runtime_lock`** — `server.py:78`. Per-tool latency cost translates directly to whole-server serialisation cost; a memory_health extension that takes 200 ms blocks every other tool for that duration.
4. **`pending_recall_events` already has `idx_recall_events_session_pending_created`** (`storage.py:1105-1106`), so session-aware filtering is already index-backed. The bottleneck on feedback linkage is the matching predicate and `limit=1`, not query speed.
5. **No background worker** — every maintenance operation (decay sweep, auto-consolidate) runs inline within a tool body. Memory observability surfaces are organised inside `memory_health` and the resource registry rather than via a separate daemon.
6. **Existing tests do not validate file-chunk dedup or scope leakage** — any fix in those areas will require new tests; existing coverage cannot transitively catch regressions.
7. **The 3 failing instructions-imperative tests are pre-existing inheritance** — do not patch in this discovery subtree; the parallel orphan branch (commits `98756bd` / `7dc421e`) handles the contract update. Downstream fix nodes pulling the orphan branch should not co-mingle scope/dedup changes with that test contract.
