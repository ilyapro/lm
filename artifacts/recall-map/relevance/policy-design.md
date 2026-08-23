# Recall-map relevance policy — frozen design

**Status: frozen for implementation.** The machine-readable source of truth is
[`policy.json`](policy.json), stored-byte SHA-256
`1f065b9e0e86b415ba8f7ea94649c9590b8e1f36aa1f312eb12d9afe6f9afe2b`.
This design was closed before candidate implementation and before any candidate holdout
was created or read.

## Decision

Freeze `directional-zsum-r1`, not the eight-feature logistic model reported as primary in
the evidence artifact. The primary model is a useful rejected baseline: fitted on organic
train and applied without refitting, it reaches only `54/395 = 0.136709` on observed-map
eval and therefore cannot authorize implementation.

The selected rule uses all and only the five decision-time features whose univariate
direction is positive in organic train, disjoint organic eval, and observed-map eval. It
gives them equal positive weight after organic-train standardization. Equal weights are
deliberate: coefficient magnitudes were not searched against eval, and no same-sign
feature was cherry-picked after seeing the transfer result. The cutoff is the train top
quintile, using the evidence evaluator's existing `0.20` minimum selection fraction and
including all ties. Neither the cutoff nor any numeric parameter is fitted to eval.

Applied unchanged after the narrow mandatory ballast eligibility filter:

| cohort | before ballast | eligible | selected | consumed | rate | events | transports |
|---|---:|---:|---:|---:|---:|---:|---:|
| organic train | 3,102 | 3,065 | 627 | 454 | 0.724083 | 250 | 239 |
| organic eval | 1,389 | 1,381 | 216 | 132 | 0.611111 | 97 | 70 |
| observed-map eval | 686 | 652 | **37** | **9** | **0.243243** | **37** | **33** |

The observed-map result clears the unchanged `0.233` floor by `0.010243` and the unchanged
minimums of 20 items, 20 events, and 10 transport sessions. The margin is intentionally
reported rather than rounded away: implementation must reproduce the frozen arithmetic,
not approximate or “improve” it.

## Exact member rule

For each residual node at map-build instant `t`, compute the vector in this exact order:

1. `level_schema`: one iff the node level is `schema`, otherwise zero.
2. `prior_matured_log_count`: `log1p(M)`.
3. `prior_nonconsumed_log_count`: `log1p(M - C)`.
4. `prior_nonconsumption_streak_log`: `log1p(K)`.
5. `prior_consumption_rate`: `C / M`, or zero when `M = 0`.

`M` counts prior deliveries of the same node created before `t` whose complete 24-hour
outcome window ends at or before `t`; `C` is the subset consumed under the frozen node-id
outcome; `K` is the consecutive nonconsumed tail ordered by outcome end and delivery
instant. Thus every history value is known before this decision. The current delivery,
unmatured windows, and later outcomes are invisible.

Let `mu` and `sigma` be the following organic-train values:

| feature | `mu` | `sigma` | weight |
|---|---:|---:|---:|
| `level_schema` | 0.26434558349451964 | 0.4409841221421261 | +1 |
| `prior_matured_log_count` | 2.856678070667311 | 2.5772848149641323 | +1 |
| `prior_nonconsumed_log_count` | 2.789348366105738 | 2.5560016163553305 | +1 |
| `prior_nonconsumption_streak_log` | 1.5799800264635717 | 1.765967625291225 | +1 |
| `prior_consumption_rate` | 0.06059739660863959 | 0.11674776461860277 | +1 |

The score is exactly `S = sum((x_i - mu_i) / sigma_i)`. Admit when
`S >= 3.8708378402511`. The cutoff is the score at descending train rank
`ceil(0.20 * 3102) = 621`; 37 train items tie at the cutoff and all ties are included.
A genuinely unavailable selected value is replaced by its frozen train mean without a
missingness feature. A node with no prior history has known zeros, not missing values.

The two nonconsumption terms are **positive**, not penalties. Their direction is positive
in all three evidence cohorts. This is best interpreted as repeated opportunity/exposure,
not as a causal claim that ignoring a node makes it useful. A negative “dead cluster”
prior is forbidden until new train/eval evidence supports that direction.

## Recency and rejected signals

There is no freshness boost, age penalty, time decay, or memory cleanup in this policy.
`node_age_log_days` is positive in organic train and organic eval but negative in
observed-map eval. Adding either sign would violate the transfer rule. `level_trace` and
`level_concept` also reverse direction and are omitted.

Recorded `score`, `bm25_score`, `vector_score`, `graph_score`, and `trigger_score` are
absent from all 686 observed-map eval items. Historical cascade stage is unavailable.
Current content/context/provenance shape was not versioned. Current `access_count`,
`usefulness_score`, `last_accessed`, and `updated_at`-derived usage are post-outcome mutable
state. None may enter eligibility, score, ordering, tie-breaks, or enablement.

## Ballast eligibility

Ballast is a separate structural eligibility invariant, never a score feature. It runs
before `MAX_POOL_NODES`, is fail-open on malformed or near-miss input, and uses no emitter,
host, project, task, or observed node list. The classifier order is fixed:

- `fc`: content starts exactly with `[file-chunk]`, followed by a space/tab and a valid
  first-line JSON envelope with nonempty `path` and `kind`, string `language`, lowercase
  64-hex `sha256`, valid positive `part/total`, and valid positive `first/last` lines.
- `ss`: the whole content matches the bounded `Strategy stagnation detected on …` machine
  form and at most four `attempts|strategy|window|reason:` lines.
- `sj`: a schema marker in `context` or `provenance` under
  `kind|type|lesson_kind|record_kind` normalizes exactly to one of the four journal kinds,
  or the first line is exactly the supervision/monitoring bracket marker. `topic` is not a
  marker, so prose discussing journals remains eligible.

The fixed randomized audit has 96/96 positives and 10/10 near-miss negatives correct. In
observed-map eval, all 34 classified ballast items were unconsumed. Organic file chunks
reverse direction under the frozen outcome construction, which is why form is not
smuggled into fitting or score. Its authority here is the narrow P2 machine-form exclusion
contract, not a claim that snapshot form is a general predictor.

No physical node deletion or decay is authorized. Any later cleanup remains a separate,
reversible goal.

## Pool, clusters, and deterministic ties

The implementation order is mechanical:

1. Inspect every residual entry. Missing node/identity is `iv`; repeated identity after
   the first valid occurrence is `du`. Identity performs bookkeeping only.
2. Apply `fc`, `ss`, and `sj` before any pool cap.
3. Compute `S`; reject below threshold as `lr`.
4. Stable-sort survivors by descending `S`, then original residual ordinal. Everything
   beyond 200 is `pc`.
5. Run the existing four-stage clustering and label gate over the admitted pool.
6. Rank clusters by descending maximum member `S`, descending mean member `S` (computed
   with `math.fsum` in original residual order), then ascending best original residual
   ordinal. Cluster membership partitions the pool, so that final ordinal is unique and
   makes the order total without an identity, stage, or label tie-break.

Cluster size, stage, labels, retrieval score, and reciprocal-rank mass are not relevance
features. Keep the existing stage-specific medoid algorithm, but its candidates are only
admitted members, so every delivered medoid clears the evaluated member rule. Any
otherwise redundant node-id final tie in that algorithm must be replaced by the already
unique original residual ordinal. Existing cluster fields and their meanings do not
change.

Node IDs and identity hashes are forbidden as scoring inputs or tie-breaks. The existing
wire medoid ID, duplicate detection, and later outcome measurement are the only allowed
node-identity uses. Task/cache/session/source identities may retain their existing cache
partition and evidence-split roles, but cannot affect this policy's eligibility, score,
threshold, ordering, tie-breaks, or enablement. There is no host check, environment gate,
feature flag, project lookup, skill gate, or literal observed-label table.

## Cache contract

Selection and history scores are recomputed on every build, on both cold and cached paths.
The structure cache may reuse only cluster templates and must recount them against the
freshly filtered and relevance-ordered pool. Bind reusable structures to selected-policy
digest `3acad3d92db2538bf3096ab99d2c4d337ea3c8aebddc226646527b4d277660ea`
and clear pre-policy in-process entries at deployment.

Do not cache low-score verdicts. If a score memo is introduced only to satisfy latency, it
must bind the policy digest, recall-event revision, 24-hour horizon, and next maturation
instant, invalidating at whichever boundary comes first. Identical residual, node values,
matured history, build instant, and corpus state must produce byte-identical selection and
ordering on cold and cached paths.

## Additive wire accounting

Add one top-level sibling, `sel`, to the response payload and therefore to persisted
`recall_events.recall_map`. Existing fields remain unchanged. `sel` has compact version
`r1` and fixed reason-vector order:

```text
[iv, du, fc, ss, sj, lr, pc]
```

Its non-optional core is `v`, raw inspected count `n`, admitted count `e`, fixed count
vector `x`, and omitted-sample count `o`. The invariant is
`n == e + sum(x)` and existing `pool == e`. Optional `q` carries at most two deterministic
`[code, content_gist]` examples of at most 24 characters and never a node ID. Counts are
unconditional; samples disappear first under budget. The exact shape, sampling order, and
reason meanings are in `policy.json`.

The new block participates in the existing payload fitter. The caps stay recall payload
`<=700`, instructions `<=150`, and AE render `<=400`. Budget pressure removes `q`, then
tail clusters down to the existing breadth floor, then existing forensic names; it never
removes `sel` core counts or changes existing unconditional counts. Journal-only maps
remain persisted and do not count as offers.

## Hash bindings and holdout boundary

| bound object | SHA-256 |
|---|---|
| dataset manifest stored bytes | `54ee3bcc9d77b99b7b2c057db3e45b747c92a18d14ccec69eb2903dfe7cbf308` |
| dataset manifest canonical JSON | `af9bb44b41fefa2a0582212c0e909afbf41c7784a5014c05f47343ffcb291e5c` |
| feature analysis stored bytes | `d17711a3bd9afe24aaf3f1fff5e07b53437988d4cf19ac4abfb527a904621f8b` |
| selected policy canonical JSON | `3acad3d92db2538bf3096ab99d2c4d337ea3c8aebddc226646527b4d277660ea` |
| expected wire additions canonical JSON | `7848191b20dce5de844ef7668246b7757aed47cf2d0af5150f52a1c8bc64a0d5` |
| holdout boundary canonical JSON | `ae9321d0d58cc32e467bbc8c0a76fa1e9b3acb1718361b9f2802e2435b551a1f` |

The holdout population does not exist yet. Its semantic boundary is frozen now: only
candidate map events and AE journals strictly after the future sealed candidate deployment
instant; no source-qualified event, transport session, session, or cache identity shared
with train or eval. The deployment node must bind the candidate commit, the selected-policy
digest, host configurations, instant, and journal roots before reading any outcome. It then
uses the unchanged tools without refitting. This freeze did not inspect or create
`artifacts/recall-map/relevance/field`.

The frozen instruments remain byte-identical:

| instrument | SHA-256 |
|---|---|
| `artifacts/recall-map/prereg.json` | `9eb6a170459bb68138a972b4ba76b76abb044cbb00024cdb04e38e038a85b1b0` |
| `scripts/recall_map_effect.py` | `e92aceeafb27ae736378f1b1866886cf8446b1cf17c1269a7cd6ad61a039eeea` |
| `scripts/recall_map_latency_bench.py` | `f9f238f31a1e795a929ec8e97eb1d2090e8474beac0db4a07a8d3ea3767b8550` |
| `~/p/ae/docs/agent-stream-prereg.md` | `68a3d4a597539a0a00d65d3dd792f3e2e6d734db263c0f79f086b0edbe98d9cb` |
| `~/p/ae/tools/lm_workload/stream_injection_metrics.py` | `da9923fe37ad3c51c0c91fe6f3bb7e80083bcb17add9c0b983a6ced25ce7235d` |

## Mechanical acceptance for the implementation node

Implementation is conforming only if it reproduces the exact formula and threshold,
accounts every residual entry with the fixed vector, filters ballast before the 200-node
cap, produces identical cold/cached ordering, preserves additive consumers and all three
budgets, and reruns the same-snapshot latency benchmark with warm pooled p95 overhead
`<=0.20`. Any parameter change requires a new policy artifact and new pre-holdout freeze;
the fresh holdout may confirm or reject this policy but may never tune it.
