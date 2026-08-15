# Final retrieval signal and context-cost report

Overall status: **FAIL**.

The strict P2 and P3 generalization gates, correction dominance, era safety, and shadow latency all pass. The sole authorized replacement P6 evaluation fails both required capability gates. Its aggregate is preserved unchanged; the authorization is consumed, and policy forbids tuning, case-level diagnosis, or a retry.

## Strict gate results

| Gate | Status | Observed | Required |
|---|---|---:|---:|
| Payload generalization | PASS | Eval median/p90 `0.543252` / `0.495360`; holdout `0.517323` / `0.497176`; retention `1.0`; gaps `0.425132` / `1.351828` pp | Ratios `<=0.60`; retention `>=0.95`; gaps `<=5` pp |
| Cross-scope generalization | PASS | Eval reduction/retention `0.558376` / `0.978803`; holdout `0.598460` / `0.957154` | Reduction `>=0.50`; retention `>=0.95` |
| Unseen-in-dev automatic recall | **FAIL** | 16 repeated families, 113 events, 21 gated events; reduction `0.040904` | Reduction `>=0.50` |
| Organic recall preservation | **FAIL** | Payload delta `-33.039%`; content-access delta `-45.5378%` | Each within `-5%` to `+5%` |
| Correction dominance | PASS | 26 focused tests passed, 0 failed; 0 replay violations | Zero violations and passing regressions |
| Era safety | PASS | 45 focused tests passed, 0 failed | Passing mixed-era and ordinary-promotion regressions |
| Shadow latency | PASS | p50 `+0.019807`; p95 `-0.075293` | Both degradation ratios `<=0.10` |

P2 and P3 therefore pass. P6 fails. Because final P7 evidence must include a passing P6 replacement result, the combined P7/final status is also fail.

## One-shot execution record

- Evaluated commit: `ee042335dbcd763054819def24e7e487de5d8f11`
- Evaluator SHA-256: `18a162f7923af10705acf67b4655f76808ebd111dfa80f23bb10c59c26fa534a`
- Replacement manifest SHA-256: `fd36c972b049c296acbd2537f9af8f1db8d7db725f188c5dd90d27ab9aa3ab83`
- Captured aggregate SHA-256: `fc1739fac4843e989cf33cd7ca707f49cd9bcaa3d3f21bda08a63894f41781ba`
- Replacement semantic reads: **1**
- Original-holdout semantic reruns: **0**
- Raw shadow reruns: **0**
- Evaluator exit status: `0`
- Stdout was redirected directly to a prechecked-absent same-directory temporary file, validated as aggregate JSON, and atomically installed unchanged.

The exact evaluator command and SHA-256 bindings for every aggregate, packet input, evaluator input, production module, and prerequisite test are recorded in `final-report.json`.

## Frozen configuration and preflight

Delivery used the recorded defaults: 1200-character snippets, 160-character context/provenance values, session deduplication, the `[0,1000,700,500,300,200]` ladder, full-node diet, sparse entries, and stats compaction. Repeat gating used `min_unlinked=5`, `max_link_rate=0.2`, `min_sessions=2`, `probe_every=25`, and trailing-stub removal. The hash embedding backend was pinned and all delivery/repeat overrides were explicitly unset.

Before launch:

- The worktree was clean, all three owned outputs and the temporary capture path were absent, and every authoritative hash matched.
- The fixed keyless verifier returned aggregate pass with zero semantic reads.
- The retrieval scope suite passed 52 tests under both hash and cached/default backends.
- The receipt bridge passed all 27 focused tests and its named manifest-only regression.
- Native dev/eval automatic-recall reductions were `0.760254` / `0.769469`, with `0%` organic deltas.
- Eval cross-scope reduction/retention was `0.558376` / `0.978803`, with zero correction violations.

No production code, evaluator, policy, threshold, corpus, manifest, or configuration changed after launch. No case was inspected and no private raw I/O is committed.

## Evidence limits

- The preserved original-holdout payload comparison contains 48 transcript-matched events and uses the frozen cited field payload as its denominator.
- The correction replay has zero candidate pairs; its focused regression suite is the substantive dominance evidence.
- Shadow latency covers 35 heterogeneous measured calls per version.
- The replacement result is aggregate-only. The sealed policy forbids case-level diagnosis, tuning, and any second semantic read, so the P6 miss must be escalated as observed.
