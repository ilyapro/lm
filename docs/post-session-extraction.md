# Post-session extraction — stage contract and the AE integration point

The post-session extraction stage is an **offline consumer of finished session
transcripts** and an ordinary Living Memory client. One CLI drives it end to
end:

```
scripts/post_session_extract.py --transcript <path> [--source claude|codex|gigacode|deepseek|ae_chat|ae_node_result] \
    [--write] [--out <report.json>]
```

Flow: transcript → `SessionRecord` → proposed `memory_remember` ops
(`postsession.insights`) + `memory_teach` ops (`postsession.corrections`) +
`memory_attest` evidence (`postsession.evidence`) → **consumption gate** →
attributed, budget-bounded, idempotent writes over the live MCP interface →
run report. The gate is checked *before any session is read*: the
pre-registered bar in `artifacts/post-session/usage-baseline.json` names the
runner as its enforcer, and a failed, missing or stale counterfactual verdict
means exit 3 with zero writes. Whether the stage may be scaled at all was
decided by the sealed-holdout field measurement in
`artifacts/post-session/field-report.json` (driver:
`scripts/postsession_field_run.py`).

This document specifies **when and how the stage gets invoked after a session
ends** — the integration point with AE — precisely enough that the hook can be
installed by a separate goal without touching this stage's core. Nothing in
this document (or in the goal that produced it) modifies anything under
`~/p/ae`; every AE fact below was verified read-only on 2026-08-19.

## 1. The zero-edit seam that already exists in AE

AE already ships a per-project lifecycle hook mechanism, and the LM project's
hook directory for goal completion **exists and is empty**:

```
/home/sfx/p/ae/projects/lm/hooks/goal_completed/     ← present, empty
```

`hooks.sh:81 goal_emit_hook <event> <goal_id>` runs every executable
`<hooks_dir>/<event>/*.sh` (where `<hooks_dir>` defaults to
`<ae_dir>/projects/<project>/hooks`), so installing the integration is
*dropping one executable file into that directory* — zero edits to AE source.
Semantics, verified in `hooks.sh`:

- **Environment given to the hook**: `AE_HOOK_EVENT`, `AE_HOOK_PROJECT`,
  `AE_HOOK_GOAL_ID`, `AE_HOOK_GOAL_BRANCH`, `AE_HOOK_BASE_BRANCH`,
  `AE_HOOK_COMMIT_SHA`, `AE_HOOK_TIMESTAMP`, `AE_HOOK_DEPTH`; the working
  directory is the project root.
- **Async by default**, capped by `timeout --kill-after=5
  ${AE_HOOK_ASYNC_MAX_SEC:-300}` (hooks.sh:192). A sibling `<hook>.toml` with
  `wait_for_completion = true` makes it synchronous with `timeout_sec`
  (default 60, hooks.sh:69).
- **Failures never propagate**: hook stdout/stderr goes to a per-invocation
  log under the project's `hook_logs` dir; `goal_emit_hook` returns 0
  regardless (hooks.sh:24).
- **Recursion refused** at `AE_HOOK_DEPTH >= 2` (hooks.sh:86), so a hook that
  itself completes goals cannot loop.
- **Fired once per goal**: `goal.sh:2098` guards with a
  `hook_fired_goal_completed` sentinel plus a lock directory before calling
  `goal_emit_hook goal_completed` (goal.sh:2106).

## 2. The seam's limitation, stated plainly

The **only** call site of the `goal_completed` hook is `goal.sh:2088
goal_record_reached()`, reached from `node.sh:1379 node_record_reached()`
**only when the finishing node is a root goal** (`GOAL_PATH` contains no `/`).
**Sub-node completion fires no hook today.** A deep goal tree can run for many
hours, finishing dozens of sessions, before the root completes and the single
`goal_completed` fires. The alternatives, if per-session latency matters:

- **Telemetry tail**: every node completion — root or not — already emits
  `tree_node_completed` / `tree_node_failed` into
  `projects/<project>/state/metrics.jsonl` (`events.sh:168 emit_event`; emit
  sites in `node.sh` at 3372, 9874, 10044, 10381, 10756, 11019). An external
  watcher (`tail -F`, `inotify`, or a timer) can react per node without any
  AE change at all. Payloads carry the goal path and outcome, not transcript
  paths — the watcher maps goal → transcripts itself (see §3).
- **The closest "session finished, whatever the outcome" point** inside AE is
  `node.sh:10965 _node_cleanup_on_exit()`, installed as the EXIT trap
  (node.sh:10975): it runs on success, failure, and interruption alike — but
  wiring anything there **requires editing node.sh**, which is exactly what
  the hook goal may decide to propose, separately.
- **Dashboard chats have no hook at all**: a chat turn is finalized by
  `dashboard/lib/chat_runtime.js:702 finalizeTurn()`, which appends a trailing
  `{"type":"done", "exit_code":…, "duration_ms":…}` record to
  `state/chats/<chatId>/transcript.jsonl`. Polling for that trailing record is
  the only zero-edit trigger for chat sessions.

## 3. The contract the future hook must satisfy

The hook (installed by a **separate goal**, as
`/home/sfx/p/ae/projects/lm/hooks/goal_completed/<name>.sh`) must:

**Invocation form.** Call the stage once per finished transcript:

```
cd ~/p/lm && python3 scripts/post_session_extract.py \
    --transcript <path> --source <source> \
    --write \
    --max-ops-per-session 3 --max-total-ops 24 --max-judge-calls 12 \
    --out "$HOME/.local/state/living-memory/postsession/reports/<ts>-<name>.json"
```

**Inputs.** Either a transcript path with its `--source`, or a corpus
`--session <key>` when the index is fresh. Mapping AE's env to transcripts:
the node's own artifacts live under
`projects/$AE_HOOK_PROJECT/state/goals/$AE_HOOK_GOAL_ID/` (node results =
source `ae_node_result`); the CLI sessions a node spawned are found through
the transcript roots recorded in `artifacts/post-session/corpus.json` (claude:
`~/.claude/projects/<cwd-slug>/<uuid>.jsonl`, codex:
`~/.codex/sessions/…`, chats: `state/chats/<id>/transcript.jsonl`). Prefer
transcripts whose mtime falls inside the goal's run window; the stage itself
is idempotent, so over-approximating the set is safe and cheap (already-done
transcripts produce zero writes).

**Exit codes.** `0` = ran to completion (including "nothing to write");
`3` = the consumption gate did not authorize the run — **nothing was
written**, and the hook must treat this as a signal to stop scheduling runs,
not retry; `2` = usage/environment error; `1` = ran, but at least one accepted
op failed to write (safe to re-run: writes are idempotent). Because AE
swallows hook failures, the hook should record non-zero outcomes in its own
log/queue if anyone needs to notice them.

**Idempotency guarantee.** Re-invoking the stage on the same transcript writes
nothing twice, regardless of how many times the hook fires: every executed op
is keyed `(transcript sha256, op fingerprint)` in the ledger at
`~/.local/state/living-memory/postsession/ledger.jsonl`, and even with the
ledger lost the write path re-checks the live server (extraction identity by
transcript sha + source span, IDF-containment near-duplicate check;
`memory_attest` is idempotent per (event, evidence) on the server's own
ledger). AE adds its own once-per-goal sentinel (§1). Concurrent hook firings
for different transcripts are safe; for the same transcript they are safe but
wasteful.

**Budget flags.** `--max-ops-per-session` (default 3), `--max-total-ops` (24),
`--max-attests-per-session` (8), `--max-total-attests` (64),
`--max-judge-calls` (12 per session), `--max-bytes` (12 MB per transcript).
The judge is `claude -p` (sandboxed, tool-less; `--model sonnet`); expect
~1–3 min and ~$0.4 per session, so the hook must either run **async** (the
default; raise `AE_HOOK_ASYNC_MAX_SEC` beyond 300 s for multi-transcript
batches) or — recommended — **enqueue-and-exit**: the hook writes one job line
(transcript path + source) to a spool under
`~/.local/state/living-memory/postsession/queue/` and returns immediately; a
timer-driven drainer invokes the stage. The gate is checked before any judge
spend, and a gate halt costs nothing.

**Consumption-gate freshness.** `--gate-verdict` /
`--gate-control` default to the tracked counterfactual reports; a verdict
older than `--max-verdict-age-days` (14) halts with exit 3. Keeping the
verdict fresh (re-running `scripts/counterfactual_consumption.py`) is an
operator duty, deliberately outside the hook.

**Where the run report lands.** Wherever `--out` points. The report contains
redacted op previews (up to 240 chars each), so it must land **outside any
tracked tree** — recommended:
`~/.local/state/living-memory/postsession/reports/`. Exit codes and counts are
also printed to stderr, which AE captures into the per-invocation hook log.

**Privacy boundary.** No transcript content in tracked artifacts, ever. The
hook must not commit reports, op dumps, cassettes, or logs into a git tree;
tracked artifacts of this stage (`usage-baseline.json`, `field-report.json`,
`corpus.json`) carry aggregates, digests and node ids only, and are guarded by
`check_privacy` / `assert_no_content` at write time. The corpus *index*
(absolute paths under `$HOME`) stays gitignored.

## 4. What was measured before allowing any of this to scale

The root goal's binding lesson — "written" is not "used" — was honored by
measuring, on the already-accumulated transcripts, whether traces this stage
extracts are actually *consumed* by later recall traffic, before any scaling
decision. The full pre-registered protocol, the four numbers (extracted vs
organic consumption, protocol-floor check, adversarial anti-dump precision,
eval→holdout degradation), the verdict, and any live writes with their exact
`memory_forget` reversal commands are published in
`artifacts/post-session/field-report.{json,md}`. If that verdict is anything
but PASS (FAIL or INCONCLUSIVE), the hook goal should not be started until the
report's "What would have to change" section is addressed and a fresh
measurement on a new sealed batch passes.

Batch `field-20260819` returned **INCONCLUSIVE**: consumption cleared both
bars (extracted 0.2222 vs organic 0.117 — 1.9x, against a 0.5x bar), but the
extractor's anti-dump gate is so selective (9 accepted ops per 300 holdout
sessions) that the cohort stayed below the pre-registered 20-trace minimum,
and holdout audit precision was 0.6667 against the 0.70 bar. Zero live writes
were performed. The hook goal is therefore **not unblocked yet**; the corpus
must accumulate more finished sessions first (see the report).
