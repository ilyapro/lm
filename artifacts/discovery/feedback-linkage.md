# Feedback Linkage Root Cause

Generated for `memory-quality-root-fixes/discover-baseline-and-root-causes/feedback-linkage-root-cause` on 2026-05-22.

## Boundary Contract

intent: discovery-only analysis of recall-event feedback linkage. This artifact records root cause, citations, chosen downstream fix, rejected alternatives, numeric target, metadata behavior, and negative-case constraints. It does not modify production code.

local replay_evidence: open the cited files at the cited line numbers in this worktree, re-run the temp-DB reproductions below, and re-run the read-only SQL aggregates against the snapshot described in `artifacts/baseline.md`.

local rollback_artifact: this Markdown file only; rollback is `git rm artifacts/discovery/feedback-linkage.md` in this worktree.

redaction_boundary: no raw trace contents, secrets, credentials, tokens, or API keys are included. Aggregate counts, file paths, function names, and test-fixture recall ids are included.

## Baseline Signal

The frozen baseline in `artifacts/baseline.md` reports 5,114 recall events, 1,060 with `feedback_applied=1`, and 4,054 still pending: ratio 0.207274 (20.73%). Scope ratios vary widely: `project:lm` is 54.79%, `project:online` 28.62%, `project:octopus` 27.16%, `project:ae` 14.47%, and `global` 5.83%.

Additional read-only snapshot aggregates:

- Recall-event metadata is sparse: 174 / 5,114 have `agent` (3.40%), 268 / 5,114 have `task` (5.24%), and 0 / 5,114 have `session_id`.
- Active trace metadata is much richer: 4,005 / 5,029 active traces have `agent` (79.64%) and 4,872 / 5,029 have `task` (96.88%).
- Every applied event has a unique `feedback_trace_id`: 1,060 feedback traces for 1,060 applied events; `max_events_per_trace=1`. That is exactly the `limit=1` signature.
- Among applied events where both the recall event and feedback trace have task metadata, 53 are same-task and 53 are different-task. The current matcher is therefore not just under-linking; it is also linking wrong events.
- 97 applied events attach a project-scoped recall to a `global` feedback trace because project recall plans include `global` in `resolved_scopes`.

## Root Cause

There are three interacting causes.

1. `memory_remember` and `memory_teach` consume at most one pending recall. `apply_pending_recall_feedback` defaults `limit=1` at `src/living_memory/feedback.py:135-140`; `memory_remember` calls it without an override at `src/living_memory/server.py:489-501`; `memory_teach` reaches the same default through `src/living_memory/consolidation.py:375-379`. A normal workflow often does several recalls before one remember/teach, so all but the newest compatible event remain pending forever.

2. The event matcher is scope-only unless both sides set `session_id`. `record_recall_event` stores `agent`, `task`, and `session_id` from `ambient_context` at `src/living_memory/storage.py:618-645`, but `_recall_event_matches` only checks scope membership and optional session equality at `src/living_memory/storage.py:1321-1334`. It ignores task and agent even when both are present. This allows unrelated same-scope recalls to attach to the next remember.

3. Recall-side metadata is not propagated by common clients. The MCP tool accepts `ambient_context` at `src/living_memory/server.py:555-571`, and retrieval passes it to storage at `src/living_memory/retrieval.py:162-168`. But AE `cmd_recall` sends only query/scope/depth/max at `/home/sfx/p/ae/lm_client.py:108-116`, and its parser exposes no `--agent` or `--task` flags for recall at `/home/sfx/p/ae/lm_client.py:1019-1025`. By contrast, AE `cmd_remember` has `--agent` and `--task` and passes them in context at `/home/sfx/p/ae/lm_client.py:81-99` and `/home/sfx/p/ae/lm_client.py:1011-1017`. LM server instructions also only make `memory_remember` structured metadata mandatory at `src/living_memory/server.py:401-408`; they do not state the matching requirement for `memory_recall.ambient_context`.

The result is a low applied ratio from under-consumption plus bad applied examples from permissive matching. Raising the limit alone would increase false positives.

## Temp-DB Reproductions

Command used:

```bash
PYTHONPATH=src python3 - <<'PY'
# Create temp MCP servers, issue two same-task recalls before one remember,
# then issue a different-task recall before one same-scope remember.
# See shell history for the full snippet; it uses tests/test_recall_feedback_loop.py's FakeMCP shape.
PY
```

Observed behavior:

```text
multi_recall_applied [
  ('01KS7CFK2SE2NFDN4WJTAJTNX1', False, None),
  ('01KS7CFK2SSX9P1G67WZHWPMXC', True, '01KS7CFK2TG1N9RXWEK3E8C8K4')
]
cross_task_applied True 01KS7CFK3EFR71XQXQMX1DEF54
```

The first reproduction proves `limit=1` leaves a same-task recall pending. The second proves a different-task recall links when scope matches and no session is present.

## Chosen Fix

Implement the fix centrally in LM, not in AE as the primary layer.

Downstream ownership:

- `src/living_memory/storage.py`: make pending-recall matching context-aware.
- `src/living_memory/feedback.py`: consume multiple strong context matches per write.
- `src/living_memory/server.py`: update instructions so recall calls include the same `scope`, `task`, `agent`, and `session_id` ambient context that later remember/teach calls use.
- `tests/test_recall_feedback_loop.py`: add positive multi-recall and negative mismatch coverage.

Recommended matching rule:

- Preserve public MCP API compatibility.
- Keep exact-scope fallback compatibility for old clients, but cap unqualified scope-only linking at one event.
- Treat `session_id` and `task` as strong discriminators: if both sides provide either field and values differ, reject.
- Treat `agent` as a weak discriminator: if both sides provide agent and no task/session is available, require equality; do not reject a same-task flow only because different agents participated.
- For fallback scope matches where the feedback trace scope is present only through `event.resolved_scopes` (for example, a `project:*` recall linking to a `global` trace), require a strong same-task or same-session match. This blocks current project-to-global false positives while still allowing deliberate same-task universal lessons.
- Raise the effective pending-event cap for strong matches to a small constant, e.g. 5. A single remember/teach should be able to consume the immediately preceding same-task recall sequence, but it should not vacuum arbitrary same-scope history.

This is a small write-path fix. It does not change recall ranking, does not rewrite historical rows, and does not require a public API migration.

## Numeric Target

For new writes after the fix:

- Deterministic unit target: two same-scope, same-task recall events followed by one remember/teach with the same task apply feedback to both recall events (`2/2`, 100% for the eligible fixture).
- Negative deterministic target: a different-task recall in the same scope remains `feedback_applied=0` after an unrelated remember (`0` false applications in the fixture).
- Negative deterministic target: a project-scoped recall does not attach to a `global` remember via `resolved_scopes` unless same `task` or `session_id` is present.
- Live metric target: do not promise to retroactively raise the historical 20.73% ratio. The measurable improvement is that future same-task eligible recall sequences no longer leave all but one event pending, and future task-mismatched applications are zero.

## Rejected Alternatives

- Increase `limit` only: improves the aggregate ratio but makes the existing false-positive bug worse by linking several same-scope unrelated recalls.
- Require `session_id` everywhere: clean in theory, but baseline has 0 / 5,114 recall events with session metadata, so it would break current recall->remember flows until every caller changes.
- Make AE CLI the only fix: useful follow-up, but incomplete. Direct MCP clients and LM server behavior would still have the `limit=1` and permissive matcher bugs.
- Add an explicit `recall_event_ids` public parameter to `memory_remember`: precise, but it is a new public contract when the current API already has enough context fields for an automatic fix.
- Rewrite historical `recall_events`: violates the append-only/raw-history principle and cannot prove which old pending events were actually useful.
- Disable implicit feedback entirely: avoids false positives but removes the useful provenance and reinforcement loop that existing tests protect.

## Client Gap

AE recall-side context propagation is a real caller gap. `/home/sfx/p/ae/lm_client.py:108-116` does not pass `ambient_context`; `/home/sfx/p/ae/lm_client.py:1019-1025` does not expose recall `--agent` or `--task`; `/home/sfx/p/ae/lm_client.py:213-220` also sends teach without context. That explains why recall events are much less annotated than traces.

This branch should still fix LM first because the server owns correctness and false-positive prevention. An AE follow-up can add recall/teach context flags or derive them from the same goal-node environment used by remember/bootstrap, but the downstream LM fix should not depend on that cross-repo work.

## Tests To Add

Extend `tests/test_recall_feedback_loop.py`, which currently only covers one recall -> one remember and one recall -> one teach at `tests/test_recall_feedback_loop.py:76-147`.

Add tests for:

- `memory_remember` consumes multiple pending recalls when `scope`, `task`, and `agent`/`session_id` are compatible.
- `memory_teach` does the same without reinforcing the original recalled result.
- Different `task` in the same scope does not link.
- Different `session_id` in the same scope does not link.
- A global write does not consume a project recall merely because `global` appears in `resolved_scopes`.
- Legacy no-task/no-session exact-scope behavior still links at most one pending event for backward compatibility.

## Latency And DB Policy

This fix is write-path only. `memory_recall` hot latency should not change. The pending query already uses `idx_recall_events_scope_pending_created` and `idx_recall_events_session_pending_created` at `src/living_memory/storage.py:1102-1106`; the downstream fix should keep the candidate window bounded.

Micro-check for downstream fix nodes: in a temp DB, seed 20 pending recall events in one scope, then time one `memory_remember` with matching task context. Confirm it consumes at most the configured cap and stays in low single-digit milliseconds on local in-process execution. Do not use the live DB for mutation tests.
