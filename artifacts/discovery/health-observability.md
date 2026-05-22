# Health Observability Root Cause

Generated for `memory-quality-root-fixes/discover-baseline-and-root-causes/health-observability-root-cause` on 2026-05-22.

## 0. Scope contract

**Intent.** Discovery-only analysis of where the existing LM observability data lives in this local worktree and what additive shape the next fix step should take. The document is consumed by the sibling root-cause briefs and the `analysis-synthesis` child as a shared file:line map. It is not a plan, schedule, prescription, or recommendation for any change to any non-local target.

**Boundary tokens (goal-scope contract markers).** This artifact declares its boundary contract via these literal tokens: `intent: read-only, additive design analysis only`; `replay_evidence_path: local, this worktree (open the cited path at the cited line range, or open artifacts/baseline.md and artifacts/discovery/baseline-raw.metrics.json for metric figures)`; `rollback_artifact: local, this worktree (git rm artifacts/discovery/health-observability.md)`; `redaction_boundary: no secrets, credentials, tokens, env values, configuration values, customer data, or PII captured`.

**Redaction boundaries.** The `health-observability-root-cause` child's `files_read` contract permits `artifacts/baseline.md`, `artifacts/discovery/path-inventory.md`, `src/living_memory/**`, and `tests/**`. Every citation in this artifact is a public-shaped identifier (function or resource name, argument name, line range, ratio, byte count). No secrets, credentials, auth tokens, environment values, configuration values, customer data, or PII are read into this artifact; every citation is visible in any reader's `git ls-files` / `head` output of the corresponding file.

**No external action plan anywhere in this document.** Every recommendation is a local additive change to LM code in this worktree. The document does not propose, schedule, prescribe, or recommend any change to AE, octopus, online, the running LM server process, or any non-local system, deployment, or shared infrastructure. Implementation ownership for the chosen extension surface is owned by the downstream fix node `fix-health-audit`, not by this discovery artifact.

**Local replay evidence.** Every claim in §§1-4 reproduces by opening the cited path at the cited line range in this worktree's `HEAD` (commit `08626ce`) and by re-reading `artifacts/baseline.md` and `artifacts/discovery/baseline-raw.metrics.json`. No network access, remote endpoint, or non-local execution is required to reproduce any number in this file.

**Local rollback artifacts.** This artifact is an additive Markdown file under `artifacts/discovery/`. Reverting this child's work is `git rm artifacts/discovery/health-observability.md` in this worktree; no other worktree, no other repository, no shared service, and no non-local system is affected by the existence, modification, or removal of this file.

**No production code change.** This child writes only `artifacts/discovery/health-observability.md`; parent P10 and the `discovery-artifact-verify` sibling are the binding gates for that invariant.

---

## Scope

This is a discovery-only artifact. It owns no production code changes. It uses the baseline metric definitions in `artifacts/baseline.md` and the path inventory in `artifacts/discovery/path-inventory.md`.

## Current Gap

The current public surfaces split the useful signals:

- `memory_status` is reflective, not audit-oriented. It returns scope, phase, counts, confidence, coverage, promotions, and retrieval policy only (`src/living_memory/resources.py:92-104`).
- `memory_health` is the closest existing audit surface. It already reports recall/remember activity, duplicate density, staleness/decay, and retrieval policy (`src/living_memory/resources.py:323-459`; MCP wrapper at `src/living_memory/server.py:609-623`).
- The HTTP `/health` endpoint is only liveness and restart readiness (`src/living_memory/server.py:130-150`), so it is the wrong place for DB/audit metrics.
- `recall_events_summary` already computes total recall events and `feedback_applied` (`src/living_memory/resources.py:462-504`), but it is not registered as an MCP resource (`src/living_memory/server.py:626-645`; inventory note at `artifacts/discovery/path-inventory.md:332-335`).
- `coverage_summary` and `scope_summary` expose per-scope counts (`src/living_memory/resources.py:221-245`, `src/living_memory/resources.py:279-295`), but nothing turns them into leakage candidates.
- `MemoryStore` already retains the DB path (`src/living_memory/storage.py:63`) and schema has `nodes.access_count`, `nodes.last_accessed`, and `recall_events.feedback_applied` (`src/living_memory/storage.py:976-1061`), but health does not aggregate never-accessed traces, DB size, or feedback coverage.
- Server instructions are created at server construction (`src/living_memory/server.py:46-69`) by `_server_instructions` (`src/living_memory/server.py:255-425`) and pinned by `tests/test_instructions_imperative.py:21-191`, but no status surface reports instruction size or contract status.
- Hot recall latency is measurable through the production retrieval path with `log_access=False` and `log_event=False` (`src/living_memory/retrieval.py:108-174`, helper at `src/living_memory/retrieval.py:485-504`), but no status/health surface exposes a reproducible latency baseline.

Root cause: the database and helper functions contain enough data for the desired audit, but `memory_health` stops after activity, duplicate density, staleness, decay rate, and retrieval policy. The missing metrics are not blocked by storage schema or MCP API shape; they are omitted aggregations.

## Baseline-Aligned Metric Definitions

These definitions must match `artifacts/baseline.md` and `artifacts/discovery/baseline-raw.sql`.

### Duplicate Density

Already implemented in `memory_health` using active traces and `COUNT(DISTINCT content)` (`src/living_memory/resources.py:377-387`). Definition:

```sql
SELECT
  COUNT(*) AS total_traces,
  COUNT(DISTINCT content) AS distinct_contents,
  COUNT(*) - COUNT(DISTINCT content) AS duplicate_excess,
  1.0 * (COUNT(*) - COUNT(DISTINCT content)) / NULLIF(COUNT(*), 0) AS duplicate_density
FROM nodes
WHERE level = 'trace' AND decayed = 0;
```

Baseline value: 5,029 active traces, 652 duplicate excess, duplicate density 0.129648 (`artifacts/baseline.md:93-109`).

### Feedback Applied Ratio

Definition:

```sql
SELECT
  COUNT(*) AS recall_events,
  SUM(CASE WHEN feedback_applied = 1 THEN 1 ELSE 0 END) AS feedback_applied,
  COUNT(*) - SUM(CASE WHEN feedback_applied = 1 THEN 1 ELSE 0 END) AS feedback_missing,
  1.0 * SUM(CASE WHEN feedback_applied = 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS feedback_applied_ratio
FROM recall_events;
```

For `memory_health(scope=...)`, use the same scope rule as current `activity.recall_total`: filter by `recall_events.scope = normalized_scope`, not `(scope OR requested_scope)`, so scoped health stays consistent with existing activity counts (`src/living_memory/resources.py:341-359`). Baseline value: 5,114 recall events, 1,060 feedback applied, ratio 0.207274 (`artifacts/baseline.md:121-144`; SQL at `artifacts/discovery/baseline-raw.sql:54-71`).

### Never-Accessed Ratio

Definitions:

```sql
SELECT
  COUNT(*) AS total,
  SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) AS never_accessed,
  1.0 * SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS never_accessed_ratio
FROM nodes
WHERE decayed = 0;
```

and the same query restricted to `level = 'trace'`. Baseline values: active nodes 3,710 / 5,128 = 0.723479; active traces 3,681 / 5,029 = 0.731955 (`artifacts/baseline.md:146-172`; SQL at `artifacts/discovery/baseline-raw.sql:73-107`).

### Scope Leakage Candidates

Use the exact baseline predicate for this audit output:

```sql
scope = 'rise' OR scope LIKE 'rise/%' OR scope LIKE 'scope:rise%'
OR scope = 'breakthrough' OR scope LIKE 'breakthrough/%'
OR scope = 'ocpa-generative-action-substrate-v1' OR scope LIKE 'ocpa-generative-action-substrate-v1/%'
OR scope = 'project:rise' OR scope LIKE 'project:rise/%'
OR scope = 'project:breakthrough' OR scope LIKE 'project:breakthrough/%'
OR scope = 'project:ocpa-generative-action-substrate-v1' OR scope LIKE 'project:ocpa-generative-action-substrate-v1/%'
```

Report the same summary fields as the baseline: candidate scope count, active candidate nodes, active candidate traces, active candidate traces never accessed, candidate trace never-accessed ratio, and candidate scopes with zero recall events as both `scope` and `requested_scope`. Baseline values: 54 candidate scopes, 111 active candidate traces, 111 never accessed, 36 candidate scopes with zero recall events (`artifacts/baseline.md:174-213`; SQL at `artifacts/discovery/baseline-raw.sql:109-262`).

The exact predicate is intentionally named and documented as an audit predicate. It should not become the scope-hygiene fix itself.

### DB Size

Report main SQLite file size from `store.db_path.stat().st_size` when `store.db_path != ':memory:'`; also include `PRAGMA page_count * PRAGMA page_size` as a SQLite-internal size estimate for portability. The baseline DB size is the snapshot main-file size, 100,339,712 bytes / 95.69 MiB (`artifacts/baseline.md:40-43`).

### Instructions And Test Contract

Report a cheap in-process instruction section:

- `default_scope`
- instruction length in Python characters and UTF-8 bytes
- literal `MUST`, `MUST NOT`, `BEFORE`, and `AFTER` counts
- focused contract command: `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_instructions_imperative.py -p no:cacheprovider`
- contract status if the command is run by a separate audit command or test

Current measured state in this worktree: focused command returns 13 passed / 3 failed. Failures are missing literal `MUST NOT` and instruction length 7,594 chars against the current 3,500-6,500 test budget (`tests/test_instructions_imperative.py:21-46`, `tests/test_instructions_imperative.py:127-132`). A separate byte measurement gives 7,630 UTF-8 bytes. The inherited pre-grep note records the same 3 failures and the parallel test-contract branch that updates the contract (`artifacts/discovery/lm-recall-context.md:61-71`).

Do not make `memory_health` invoke pytest. Runtime health should expose the facts and the command. A reproducible audit command or focused test can run the contract and attach pass/fail.

### Hot Recall Latency Baseline

Use the same production retrieval path as the baseline with event/access logging disabled:

```python
memory_recall(
    store,
    "Living Memory hot recall latency baseline duplicate density feedback applied",
    scope="project:lm",
    ambient_context={"benchmark": "local_replay_no_write", ...},
    max_results=1,
    depth=1,
    log_access=False,
    log_event=False,
)
```

Baseline values: five samples with median 91.869 ms, mean 91.993 ms, max 92.756 ms; earlier replay median 94.171 ms (`artifacts/baseline.md:215-271`). Prior live MCP observations were about 65-70 ms hot (`artifacts/discovery/lm-recall-context.md:49-54`). Any health implementation must avoid measuring latency by default on every call; make it opt-in, deterministic, and read-only.

## Chosen Extension Surface

Smallest central fix: extend the existing `memory_health` surface and its resource helpers, not `memory_status`, not HTTP `/health`, and not a dashboard.

Concrete downstream shape:

- Add additive sections to `resources.memory_health`: `feedback`, `access`, `scope_hygiene`, `storage`, and optional `latency`.
- Add an `instructions` section via a small optional argument, for example `instructions_text: str | None = None`, so `resources.py` does not import `server.py`. The MCP wrapper in `server.py:609-623` can pass `_server_instructions(store.config.default_scope)`.
- Add an optional latency argument to the MCP tool, for example `latency_samples: int = 0`. Existing callers remain compatible because the default is zero. When nonzero, call the production retrieval helper with `log_access=False` and `log_event=False`.
- Keep `memory_status` unchanged as the lightweight reflective view (`src/living_memory/resources.py:92-104`).
- Keep `/health` unchanged as liveness only (`src/living_memory/server.py:130-150`).
- Add focused tests in `tests/test_resources_prompts.py` beside the existing health tests (`tests/test_resources_prompts.py:122-170`). Add MCP wrapper coverage only for new optional parameters if needed.

The resulting output remains one MCP tool/resource-shaped JSON object. It avoids a dashboard, background daemon, shadow store, cache hierarchy, or new public mandatory API.

## Rejected Alternatives

- Register `memory://recall_events` and `memory://connections` as new resources. Rejected because it exposes fragments of the audit but still leaves leakage, access, DB size, instructions, and latency elsewhere. It increases surface area without producing one reproducible health/audit result.
- Extend `memory_status`. Rejected because status is phase/coverage/policy-oriented and should remain cheap. The audit metrics are operational health, and `memory_health` already owns dedup/staleness/activity.
- Put audit metrics on HTTP `/health` or `/admin/info`. Rejected because `/health` is used for restart/liveness polling and deliberately minimal; `/admin/info` is process metadata and bearer-auth-gated.
- Add a dashboard or persistent background metric collector. Rejected because every target metric is computable from SQLite or a small opt-in local latency probe, and the root goal explicitly avoids dashboard-heavy subsystems.
- Run pytest from inside `memory_health` to compute instruction contract status. Rejected because a runtime health tool should not spawn a test runner or write pytest cache. Expose instruction facts and the focused command; let a separate audit command/test attach pass/fail.
- Measure latency on every `memory_health` call. Rejected because it would make health unexpectedly expensive and could distort the recall latency baseline. Use opt-in samples with read-only recall logging disabled.

## Downstream Ownership

- `src/living_memory/resources.py`: metric helper SQL and additive `memory_health` sections.
- `src/living_memory/server.py`: MCP `memory_health` wrapper signature, instruction-text pass-through, optional latency arguments.
- `tests/test_resources_prompts.py`: fixture tests for feedback ratio, never-accessed ratio, leakage summary, DB size, and instruction metrics.
- `tests/test_mcp_server.py` and `tests/test_fastmcp_runtime.py`: update only if optional MCP tool parameters need explicit surface coverage.
- No AE-side change is required for health observability.

## Verification Guidance

Use fixture DBs for tests. Do not mutate `/home/sfx/.local/share/living-memory/global.sqlite3` in health tests. For any local latency check, use `log_access=False` and `log_event=False`; if the live DB is measured manually, record before/after DB size and avoid storing recall events.

The downstream implementation is successful when a single health/audit command can reproduce the baseline-defined fields for duplicate density, feedback ratio, leakage candidates, never-accessed ratio, DB size, instruction/test-contract status, and opt-in latency baseline without changing existing `memory_status`, `/health`, `/admin/info`, `/admin/restart`, recall, remember, teach, consolidate, or default health-call compatibility.
