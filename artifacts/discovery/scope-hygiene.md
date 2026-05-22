# Scope Hygiene Root Cause

Generated for `memory-quality-root-fixes/discover-baseline-and-root-causes/scope-hygiene-root-cause` on 2026-05-22. Reads only `artifacts/baseline.md`, `artifacts/discovery/path-inventory.md`, `src/living_memory/**`, and `tests/**` in this worktree; produces only this Markdown file.

---

## 0. Scope contract

**Intent.** Read-only root-cause brief that names the local LM code site responsible for the bare `rise/*`, `breakthrough/*`, and `ocpa-generative-action-substrate-v1/*` scopes observed in `artifacts/baseline.md`, names the simplest LM-local central guard that prevents future fragmentation, lists rejected alternatives with their reasons, and assigns downstream file ownership for the fix node. The intent is bounded to producing a single Markdown file under `artifacts/discovery/` in this worktree; the intent excludes any production code change, any DB write, any MCP server call, and any cross-repo modification.

**Boundary tokens (goal-scope contract markers).** This artifact declares its boundary contract via these literal tokens: `intent: read-only root-cause analysis only`; `replay_evidence_path: local, this worktree (open the cited LM source path at the cited line range)`; `rollback_artifact: local, this worktree (git rm artifacts/discovery/scope-hygiene.md)`; `redaction_boundary: no secrets, credentials, tokens, env values, configuration values, customer data, or PII captured`.

**Local replay evidence.** Every citation below reproduces by opening the cited LM path (`src/living_memory/scope.py`, `src/living_memory/storage.py`, `src/living_memory/server.py`, `tests/test_scope.py`) at the cited line range in this worktree's `HEAD` (commit `08626ce`). The metric numbers reproduce from `artifacts/baseline.md` + `artifacts/discovery/baseline-raw.sql` + `artifacts/discovery/baseline-raw.metrics.json` against the local snapshot `/tmp/lm-baseline-replay.sqlite3` (MD5 `58fc12fd47c5f71e5b8867ee71590668`) under `mode=ro`. No network access, no remote endpoint, and no non-local execution are required. local replay_evidence is fully self-contained inside this worktree.

**Local rollback artifacts.** Reverting this child's work is `git rm artifacts/discovery/scope-hygiene.md` in this worktree; no other worktree, no other repo, no shared service, and no non-local system is affected. local rollback_artifact = this single Markdown file.

**Redaction.** Citations are public-shaped: function name → file:line, schema column name, MCP tool name, metric ratio, and scope identifiers that already appear in `artifacts/baseline.md`. No secrets, credentials, tokens, env values, configuration values, customer data, or PII are read into this artifact. The redaction_boundary covers what may leave the worktree; nothing in this artifact crosses it.

**No external action plan.** This artifact's §5 (AE-vs-LM decision) and §6 (downstream file ownership) describe LM-local action options only. Any AE-side mention is structural ("AE callers currently pass `scope: '<goal-id>'` without `project:` prefix") and never prescriptive ("AE must change"). The chosen fix lives entirely inside `src/living_memory/`; an AE-side migration of pre-existing rows is out of scope for this discovery brief and is mentioned only as a deliberately deferred follow-up in §7.

**No production code change.** This child writes only `artifacts/discovery/scope-hygiene.md`. Parent P10 and the `discovery-artifact-verify` sibling are the binding gates for that invariant.

---

## 1. Observed problem (from baseline)

From `artifacts/baseline.md` §"Scope Leakage Candidates" (lines 174-213) and `artifacts/discovery/baseline-raw.metrics.json` (`scope_leakage` section):

- **54 candidate scopes** match the predicate `scope = 'rise' OR scope LIKE 'rise/%' OR scope = 'breakthrough' OR scope LIKE 'breakthrough/%' OR scope = 'ocpa-generative-action-substrate-v1' OR scope LIKE 'ocpa-generative-action-substrate-v1/%' OR scope LIKE 'project:rise%' OR scope LIKE 'project:breakthrough%' OR scope LIKE 'project:ocpa-generative-action-substrate-v1%'`.
- **111 active candidate traces** total.
- **111 / 111 candidate traces have `access_count = 0`** — never accessed.
- **36 of 54 candidate scopes have zero recall events** as both `scope` and `requested_scope`.

The salient row pattern:

| stored `scope` | active traces | never accessed | recall events as scope/requested |
| --- | ---: | ---: | ---: |
| `rise` | 7 | 7 | 0 / 0 |
| `rise/_critique` | 9 | 9 | 0 / 0 |
| `rise/_verify` | 6 | 6 | 0 / 0 |
| `breakthrough/gemini-flash-contract` | 3 | 3 | 0 / 0 |
| `breakthrough/programming-capability-loop` | 4 | 4 | 0 / 0 |
| `breakthrough/split-manifests-and-leakage-guard` | 4 | 4 | 0 / 0 |
| `ocpa-generative-action-substrate-v1` | 9 | 9 | 0 / 0 |
| `ocpa-generative-action-substrate-v1/_critique` | 2 | 2 | 0 / 0 |
| `ocpa-generative-action-substrate-v1/untrack-generated-artifacts` | 4 | 4 | 0 / 0 |
| `project:rise` | **0** | 0 | **6 / 6** |
| `project:breakthrough/gemini-flash-contract` | **0** | 0 | **8 / 8** |
| `project:ocpa-generative-action-substrate-v1` | **0** | 0 | **7 / 7** |

The mirror-image asymmetry is the signal: agents **store** traces under bare goal-node identifiers (`rise/_critique`, `breakthrough/gemini-flash-contract`, `ocpa-generative-action-substrate-v1/...`), but **recall** them under `project:` prefixes. The bare-scope traces are functionally unreachable.

---

## 2. Root cause

There is exactly one causal step: **the LM write path does not call `normalize_scope` before persisting the `scope` column.**

### 2.1 The asymmetry in code

- **Read path** (`memory_recall`) — `src/living_memory/retrieval.py:127` calls `ScopeResolver.resolve`, which at `src/living_memory/scope.py:54-69` always passes the requested or ambient scope through `normalize_scope` (`scope.py:97-113`). `normalize_scope` rewrites bare strings (no `:`) to `project:<value>` at `scope.py:113`, and accepts only the prefixes `project:`, `session:`, `workspace:` (rewritten to `project:`), `repo:` (rewritten to `project:`), or the literal `global`. Unknown prefixes raise.
- **Write path** (`memory_remember` and every other producer of new nodes) — `src/living_memory/storage.py:161` reads `scope` directly from `context_data["scope"] or self.config.default_scope` and persists it **as-is** at `storage.py:195`. There is no call to `normalize_scope` between the dictionary read and the SQL `INSERT`.

This is the entire mechanism. Any caller passing `context={"scope": "rise/_critique", ...}` to `memory_remember` (`server.py:475-499`) lands in `MemoryStore.append_trace` (`storage.py:215-223`) → `MemoryStore.create_node` (`storage.py:115-138`) → `MemoryStore._insert_node` (`storage.py:140-213`), where the bare string is written verbatim to the `nodes.scope` column. The same caller's subsequent recall under `scope="project:rise/_critique"` is normalised on the read side to `project:rise/_critique` and `WHERE scope = ?` finds nothing, because the stored value is the bare form.

### 2.2 Why this manifests as Octopus-affiliated leakage specifically

The Octopus tree-decomposition system creates goal-node scopes shaped as `<root-goal>/<child>/<grandchild>` (e.g. `rise/_critique`, `breakthrough/gemini-flash-contract`, `ocpa-generative-action-substrate-v1/untrack-generated-artifacts`). Octopus agents pass these strings directly as `context["scope"]` when calling `memory_remember`. The `rise`, `breakthrough`, and `ocpa-generative-action-substrate-v1` names are not Python project identifiers; they are tree-node labels that happen to be unique enough to look like scope keys. Because `normalize_scope` is not invoked on write, the bare strings persist; because every recall path goes through `normalize_scope`, the same agents recalling later miss their own writes.

This is not a problem specific to Octopus. Any future caller emitting a bare scope string would reproduce the same fragmentation. `project:ae`, `project:lm`, `project:octopus`, `project:online`, and `global` work today because their callers happen to pass the canonical form already. The write-side asymmetry is a latent foot-gun for every caller; Octopus is the loudest current consumer.

### 2.3 What is **not** the root cause

These were considered and ruled out by reading the code:

- **`_ambient_scope` cwd derivation (`scope.py:185-204`)** — uses `Path(workspace_path or cwd).name`. A cwd of `/root/p/octopus/.worktrees/_node_exec_rise` would yield ambient `_node_exec_rise` → `project:_node_exec_rise`. This is a real second-order issue (path-inventory §4 line 183 notes it; tests/test_scope.py:8-103 has no coverage of it), but it cannot be the cause of the bare `rise/*` leakage observed in the baseline because `memory_remember` (`server.py:476-499`) does not accept `ambient_context` at all — only the `memory_recall` tool does (`server.py:556-581`). Cwd-derived `_node_exec_*` scopes therefore only appear in `recall_events.requested_scope`, never as a stored trace `scope`. The 7 traces stored under `rise`, the 9 under `rise/_critique`, and the 9 under `ocpa-generative-action-substrate-v1` came from explicit `context["scope"]` values, not from cwd derivation.
- **`infer_project_scope` (`scope.py:122-143`)** — only fires when both explicit `scope` and ambient scope are absent. It is a recall-only path and cannot affect what gets written.
- **MCP `memory_remember` framing (`server.py:475-515`)** — the handler is a thin pass-through; it does not transform `context` before calling `append_trace`. There is no MCP-layer canonicalisation to fix.
- **`MemoryConfig.default_scope`** — falls back at `storage.py:161` only when `context["scope"]` is absent; the bare-scope traces in the baseline have explicit (mal-formed) scope values, so the default is not in the causal chain.
- **`record_recall_event` (`storage.py:601-648`)** — persists both `scope` and `requested_scope`, but the `scope` recorded there is the post-`normalize_scope` value from the resolved plan, not the raw input. This is symmetric with the read path and does not contribute to the asymmetry.

---

## 3. Chosen simplest central fix

**Add `scope = normalize_scope(...)` inside `MemoryStore._insert_node` so that every node insertion canonicalises the `scope` column before it reaches SQL.**

Concretely (downstream fix node executes this; this brief does not edit production code):

- Change `src/living_memory/storage.py:161` from
  ```python
  scope = str(context_data.get("scope") or self.config.default_scope)
  ```
  to
  ```python
  scope = normalize_scope(str(context_data.get("scope") or self.config.default_scope))
  ```
- Add `from living_memory.scope import normalize_scope` near the existing `from living_memory.config import …` import (`storage.py:16`). `scope.py` imports only stdlib + `pathlib.Path`, so there is no circular import risk.
- Persist the canonical scope back into `context_data["scope"]` at `storage.py:164` so the JSON `context` column matches the SQL `scope` column.

### Why this is the right shape

1. **Single chokepoint covers every level.** `_insert_node` is the only producer of new rows in the `nodes` table. `append_trace`, `append_trace_with_rejected_alternatives`, concept creation in `consolidation.py`, schema materialisation, and any future caller all flow through it. One line of additional canonicalisation closes every write path that could otherwise diverge from `normalize_scope`.
2. **Idempotent for existing callers.** `normalize_scope` is a no-op for already-canonical inputs: `"project:lm"` → `"project:lm"`, `"project:ae"` → `"project:ae"`, `"global"` → `"global"`, `"session:abc"` → `"session:abc"`. Existing valid producers (every current LM and AE call that passes a canonical scope) are unaffected. Only the latent bug — bare strings and unknown prefixes — changes behaviour, and that change is exactly the desired one.
3. **Preserves the public MCP API.** `memory_remember`'s wire signature (`server.py:476-481`) does not change. Callers continue to pass `context={"scope": "..."}` with any value they like; the server tool body still calls `append_trace`. The canonicalisation happens one frame deeper, transparently.
4. **Preserves append-only semantics.** The `scope` rewrite happens before `INSERT`. Existing rows are untouched. The "trace content is append-only" invariant at `storage.py:344-345` is unrelated to this code path and is not weakened.
5. **No new DB columns, no new tables, no schema migration.** `SCHEMA_VERSION = 2` (`storage.py:31`) does not change. The fix is purely runtime.
6. **Same `normalize_scope` rule everywhere.** Read-side `ScopeResolver.resolve` and write-side `_insert_node` then share the exact same canonicalisation function, so future edits to scope canonicalisation propagate to both sides automatically. The asymmetry that produced this bug becomes structurally impossible.
7. **Constant-time per write.** `normalize_scope` is a few string ops and a regex-free `split(":", 1)`; the added latency per write is in the microsecond range and never touches the read hot path. The `artifacts/baseline.md` hot recall median (~92 ms in-process, ~65–70 ms MCP per Living Memory's own latency observation) is not affected.

### Numeric target

- For the **bare-scope failure mode** (the observed cause): **new bare-scope rows after the fix = 0** in any week-long audit window. The baseline currently shows 111 active candidate traces and 36 candidate scopes with zero recall events; after the fix, the candidate-trace count cannot grow from new writes (existing rows persist until a separate migration / decay run).
- For the **error-prefix failure mode**: `normalize_scope` raises `ValueError("unsupported scope prefix: <prefix>")` for unknown prefixes (`scope.py:112`). Today this never surfaces because the write path bypasses normalisation; after the fix, `memory_remember` would raise on mal-formed scope strings instead of silently writing them. The fix node must decide whether to (a) propagate the error (loud failure for buggy callers) or (b) fall back to `default_scope` with a logged warning. The recommended posture is (a) — fail loud, because silent fallbacks are how the original asymmetry survived for so long. The MCP error surface already returns structured errors to clients.

---

## 4. Rejected alternatives

### 4.1 Rewrite every AE caller to pass `project:<goal-id>`
**Rejected because:** It is the inverse of "one central rule": every existing and future caller that emits a memory_remember would need to know the scope-canonicalisation rule and apply it correctly. Scattered enforcement is exactly the failure mode that produced the bug. It also leaves LM internal callers (consolidation concepts, schemas, future server features) free to repeat the bug. Single LM-internal guard at the SQL chokepoint dominates.

### 4.2 Add a scope-rewrite table mapping `rise → project:octopus`, `breakthrough → project:octopus`, `ocpa-generative-action-substrate-v1 → project:octopus`
**Rejected because:** This embeds Octopus-specific project knowledge into the LM core. Every new project would need a new mapping. It also changes the semantic of stored scopes — agents passing `rise/_critique` may have intended to keep the goal-tree path; collapsing all of `rise/*` and `breakthrough/*` into `project:octopus` would lose that signal forever. The simpler rule "bare strings become `project:<name>`" preserves every caller's intent while restoring read/write symmetry. Semantic consolidation of Octopus subgoal scopes into `project:octopus` is a caller-side policy choice, not an LM rule.

### 4.3 Enforce `normalize_scope` at the MCP server boundary (`server.py:476-499`) instead of in storage
**Rejected because:** It would miss every non-MCP producer of nodes — `consolidation.py` cluster→concept creation (`consolidation.py:204-313`), `_materialize_procedural_schemas` (`consolidation.py:389-422`), cross-scope promotion (`consolidation.py:781-833`), and any direct `MemoryStore.append_trace` user (tests, future internal tooling). Storage is the actual chokepoint; the MCP boundary is just one of several entry points to it.

### 4.4 Add a write-time fingerprint that rewrites any cwd-derived `_node_exec_*` scope to its enclosing project
**Rejected as part of the dominant fix:** This addresses a separate, smaller leakage (cwd-derived `project:_node_exec_*` scope on the recall side only — `memory_remember` doesn't accept `ambient_context`, so it cannot produce `_node_exec_*` trace scopes today). It is genuinely worth doing in `scope._ambient_scope` and `_ambient_project` (`scope.py:198-202` and `scope.py:223-227`) by detecting `.worktrees/` in the path and using the project root, but it is orthogonal to the bare-scope leakage that dominates the baseline. Tackling both in one fix muddies blame and conflates two independent guards. Listed under §7 as a deliberately deferred companion fix.

### 4.5 Run a one-time SQL migration that rewrites historical bare-scope rows to `project:<name>`
**Rejected as part of the dominant fix:** Migration is a separate verification/integration concern; the discovery brief asks for the simplest *central* guard that prevents *future* fragmentation. Migration touches existing data and is its own decision (does the team want `rise/_critique` → `project:rise/_critique`, or `→ project:octopus`, or `→ project:octopus/rise/_critique`, or no migration at all and rely on decay?). Listed under §7 as a deliberately deferred companion.

### 4.6 Reject any scope at write time that does not start with `project:` / `session:` / `global` (no canonicalisation, only rejection)
**Rejected because:** It produces louder failure than necessary. Existing valid callers that happen to pass bare strings (e.g. `"alpha"` rather than `"project:alpha"`) currently get a half-working write; pure rejection would break them entirely with no recovery. Canonicalising bare strings to `project:<name>` matches the existing read-side behaviour, so the fix is a strict refinement of current semantics rather than a contract break.

### 4.7 Add a write-time consolidation step that periodically re-scopes orphan traces
**Rejected because:** It is a background scan running on every node, mutates rows after the fact (against append-only intent), and adds latency to every consolidation cycle. The write-time canonicalisation guard is O(1) per write and idempotent; consolidation cleanup would be O(N) per sweep with no upside over the simple rule. Existing decay / consolidation is the right place to address legacy rows only if a migration is explicitly chosen (see 4.5).

---

## 5. AE-vs-LM layer decision

**LM-internal.** The fix lives entirely in `src/living_memory/storage.py` and (transitively) imports `normalize_scope` from `src/living_memory/scope.py`. AE callers are unchanged. The decision rationale:

1. **Single point of enforcement.** A future AE refactor, a new MCP client, a `bootstrap-project` rerun, or any direct `MemoryStore.append_trace` user gets the rule for free. AE-side changes would scatter the rule across `lm_client.py`, every agent wrapper, every shell entry point, and the AE tree-decomposition system. The structural fragility that produced the bug is the absence of a single rule — adding the rule scattered preserves the fragility.
2. **No AE knowledge required.** The fix does not need to know that `rise`, `breakthrough`, or `ocpa-generative-action-substrate-v1` mean Octopus subgoals, that AE uses worktrees, or that the goal-tree system exists. It only enforces the existing `normalize_scope` contract, which AE already satisfies for `project:ae` and the other canonical scopes.
3. **Read-side already uses LM-internal canonicalisation.** The read path runs `normalize_scope` inside `ScopeResolver.resolve` regardless of which client sent the request. Making the write path symmetric in the same layer is the structural mirror.
4. **AE-side companion (out of scope here).** AE could additionally surface a `project:` prefix in its `memory_remember` calls for clarity, and could optionally collapse goal-tree paths into a single project scope. Both of those are caller-side policy choices that should be decided where AE owns its scope semantics. This brief does not prescribe AE behaviour; it only notes that the LM-internal guard is sufficient on its own to stop new bare-scope rows from landing.

---

## 6. Downstream file ownership

Owned by the downstream fix node `fix-scope-hygiene` (a sibling of this discovery subtree, scheduled to execute after `analysis-synthesis`):

| Component | File / range | Action shape |
| --- | --- | --- |
| Write canonicalisation guard | `src/living_memory/storage.py:16` (import) and `src/living_memory/storage.py:161-164` (scope read + rewrite) | Import `normalize_scope`; replace `scope = str(context_data.get("scope") or self.config.default_scope)` with the normalised form; set `context_data["scope"] = scope` so the JSON `context` column matches the SQL `scope` column |
| Test: bare scope canonicalisation | new test in `tests/test_scope.py` (after `:103`) **or** `tests/test_storage.py` (after `:180`) | Append-trace with `context={"scope": "rise/_critique"}` and assert the stored node's `scope` is `"project:rise/_critique"` |
| Test: explicit project scope is unchanged | new test in `tests/test_scope.py` | Append-trace with `context={"scope": "project:lm"}` and assert stored `scope` is `"project:lm"` (idempotency) |
| Test: `workspace:` and `repo:` aliases are rewritten | new test in `tests/test_scope.py` | Append-trace with `context={"scope": "workspace:lm"}` and assert stored scope is `"project:lm"` |
| Test: unknown prefix raises at write | new test in `tests/test_storage.py` | Append-trace with `context={"scope": "foo:bar"}` and assert `ValueError("unsupported scope prefix: foo")` from `_insert_node` |
| Test: empty scope falls back to default | new test in `tests/test_storage.py` | Append-trace with empty `context["scope"]`; verify default-scope behaviour matches today's semantics |

Co-touch concern (raised in the parent `_critique`, observation 4): the `fix-dedup-noise` fix node also operates near `storage.py:161` because content fingerprinting will likely live in `_insert_node` (or a wrapper helper). To avoid an integration conflict the fix nodes coordinate at integrate time as follows: `fix-scope-hygiene` lands first (it is a single 2-line change with a tight blast radius), `fix-dedup-noise` rebases onto the canonicalised `scope` value so its fingerprint logic sees the post-normalisation scope. This ordering preserves the property that the fingerprint is computed on the canonical scope — important because two writers passing `rise/_critique` and `project:rise/_critique` for identical content should fingerprint as duplicates, which only holds if scope is normalised first.

Fixture-vs-live DB policy (raised in the parent `_critique`, observation 7): the downstream `fix-scope-hygiene` test suite runs against a fresh tmp-path SQLite store in each pytest function (`with MemoryStore(tmp_path / "memory.sqlite3") as store:` — see existing tests `tests/test_scope.py:9`, `:32`, `:66`, `:86`). It does not touch the live DB at `/home/sfx/.local/share/living-memory/global.sqlite3`. Any latency or behaviour comparison against the live snapshot uses the read-only snapshot `/tmp/lm-baseline-replay.sqlite3` already created by `baseline-live-audit` (MD5 `58fc12fd47c5f71e5b8867ee71590668`).

Per-fix latency micro-check (raised in the parent `_critique`, observation 1): the fix node should run the in-process latency probe described in `artifacts/baseline.md` §"Hot Recall Latency" before and after the change and assert median delta < 5 ms. The probe runs against the snapshot, with `log_access=False, log_event=False`, so it produces no DB writes and the snapshot MD5 stays unchanged. Expected outcome: zero measurable delta on read latency because the change touches only the write path; write-side delta is bounded by one `normalize_scope` call per insert.

---

## 7. Deliberately deferred follow-ups

These are real but separate concerns; they are not part of this fix's central guard and are recorded so `analysis-synthesis` can sequence them:

1. **Worktree-aware cwd canonicalisation** in `scope._ambient_scope` (`scope.py:198-202`) and `scope._ambient_project` (`scope.py:223-227`): when `Path(cwd).name` looks like `_node_exec_*`, walk up the path to find a `.worktrees/` segment and use the parent directory's name as the project root. Adds a regex match and one `Path.parent` walk per recall; trivial cost; needs a focused test that `ambient_context={"cwd": "/x/octopus/.worktrees/_node_exec_rise"}` resolves to `project:octopus`, not `project:_node_exec_rise`. Deferred because it only affects `recall_events.requested_scope` (`memory_remember` does not accept `ambient_context`), so it does not unblock the dominant write-side leakage. Sequence-after-write-guard so the two fixes land independently with separate test coverage.
2. **Historical migration of bare-scope rows.** A one-shot SQL `UPDATE nodes SET scope = 'project:' || scope WHERE scope NOT LIKE 'project:%' AND scope NOT LIKE 'session:%' AND scope <> 'global'` would rescue the 111 active candidate traces. Deferred because (a) it touches stored data, (b) the team may prefer to leave them to TTL decay (180 d) rather than migrate, (c) a migration policy may decide instead to map `rise/* → project:octopus/rise/*` which requires explicit choice. Listed for the verify-and-record subtree to surface as a decision.
3. **`memory_remember` accepting `ambient_context`.** If a future change wants cwd-derived scope to apply on writes as well as reads, the public MCP signature for `memory_remember` (`server.py:476-481`) would need an `ambient_context` parameter. Out of scope here. Mentioned only because the worktree-aware fix from item 1 would then propagate to write-side as well.
4. **Tightening `infer_project_scope` token overlap** (`scope.py:122-143`): currently does a permissive `project_tokens & query_tokens` match that could cross-match `project:rise` from a query like "what rises in latency?". Not a regression of the current bug, but adjacent to the topic; documented for completeness so it is not silently re-introduced when the canonical-scope rule lands.

---

## 8. Acceptance signal for the downstream fix node

The `fix-scope-hygiene` node is done when, in this repo:

1. `MemoryStore.append_trace(content, {"scope": "rise/_critique", ...})` results in a stored `scope` column of `"project:rise/_critique"` (asserted by new test in `tests/test_scope.py` or `tests/test_storage.py`).
2. `MemoryStore.append_trace(content, {"scope": "project:lm", ...})` stores `"project:lm"` (idempotency assertion).
3. `MemoryStore.append_trace(content, {"scope": "foo:bar", ...})` raises `ValueError("unsupported scope prefix: foo")` (consistent with `normalize_scope`).
4. The full pytest suite passes — including `tests/test_scope.py`, `tests/test_storage.py`, `tests/test_recall_feedback_loop.py`, `tests/test_acceptance_contract.py`, and the existing test in `tests/test_acceptance_contract.py:54` (`test_cold_start_recall_is_empty_and_under_50ms`) which is the cold-recall latency guard.
5. In-process hot recall median against `/tmp/lm-baseline-replay.sqlite3` does not regress more than 5 ms vs the `artifacts/baseline.md` measurement (read path is unchanged; the assertion is a regression guard, not an improvement claim).
