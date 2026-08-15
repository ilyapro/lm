# Animal-planet final evaluation

Status: **evidence gap**. Every observed performance, retention, latency, protocol, era-safety, and compatibility gate passed. The frozen evaluator did not emit the required auto-recall slice for holdout fingerprints absent from dev, so that one claim is not marked pass and the responsible evaluator area is escalated below. No holdout-driven tuning or source edit was made.

## Sealed holdout

The packet was frozen at commit `5fd03fc66246d833d90285d8b456e6f3a317ca8b`. Tracked history contained no prior semantic evaluation artifact. This node made the one authorized semantic read at final merged HEAD `9b70f98883764dcaf5c7071c38d78ea7264a16be`:

```text
python3 scripts/ap_baseline.py compare --split holdout --metrics payload,cross_scope,correction_dominance,auto_recall
```

It exited 0 in 4.02 seconds. The read is recorded in Living Memory trace `01KZYCFYYZ7731R2YPNTR42Z0Y`. No holdout case was inspected, the comparison was not rerun, and no code or threshold was changed after seeing the result. The holdout hash remains `d83a3b9a19c8237d0ad31f9434c2822ec8875fb5753cf94b63d59a6413788b3c`.

## Frozen baseline

The field-cited and reproducible survivor baselines differ where 114 historical `/root`-session tool results are no longer readable. This is the packet’s frozen `d3-w3-payload-population` discrepancy, not a post-hoc adjustment.

| Measure | Field cited | Reproduced packet |
| --- | ---: | ---: |
| Organic feedback linkage | 42.4% | 44.0994% |
| Automatic feedback linkage | 17.6% | 17.5926% |
| Payload median chars | 26,496 | 26,792 |
| Payload p90 chars | 35,237 | 30,614 |
| Payload population | 278 | 164 |

`feedback_applied` remains a lower-bound proxy for observed use, not ground-truth relevance.

## Recorded replay: payload

Reductions below compare current delivery with the recorded behavior on the same split and matched-event population. The ratios in the last two columns compare current output directly with the field-cited 26,496/35,237-character baseline.

| Split | n | Median before → after | Reduction | p90 before → after | Reduction | After / cited median | After / cited p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 185 | 24,702 → 14,211 | 42.47% | 30,780 → 17,962 | 41.64% | 0.536× | 0.510× |
| eval | 133 | 25,345 → 14,394 | 43.21% | 30,176 → 17,455 | 42.16% | 0.543× | 0.495× |
| holdout | 48 | 23,956 → 13,707 | 42.78% | 29,595 → 17,519 | 40.80% | 0.517× | 0.497× |

Holdout exceeds the required 25% absolute reduction. Its reduction differs from eval by only 0.43 percentage points at median and 1.35 points at p90, both within the 10-point generalization budget. Top-result content retention and useful-feedback content retention are 100% on dev, eval, and holdout.

## Recorded replay: scope, automatic recall, and corrections

Same-scope retention is `(after.results − after.cross_scope_results) / (before.results − before.cross_scope_results)`. It is a scope-label proxy, not an independent relevance judgment.

| Split | Cross-scope before → after | Admission reduction | Same-scope retention | Repeated-auto compaction | Organic char delta | Organic access delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 1,310 → 597 | 54.43% | 97.63% | 76.03% | 0.00% | 0.00% |
| eval | 788 → 348 | 55.84% | 97.88% | 76.95% | 0.00% | 0.00% |
| holdout | 5,586 → 2,243 | 59.85% | 95.72% | 59.09% | −1.91% | −0.57% |

The holdout cross-scope gate passes the required ≥40% reduction at ≥95% retention. Split-wide repeated automatic traffic also exceeds 30% compaction, while organic payload and access remain inside the ±5% safety budget.

The correction replay reports zero violations on all splits. Known-superseded deliveries fall 111→2 on dev, 61→1 on eval, and 27→1 on holdout; superseded top-1 events fall to zero. Because no replay event contains both endpoints of a supersedes pair, the actual ordering proof comes from the focused correction/property/SQLite tests: 26 passed.

## End-to-end real-workflow shadow

The local live database was copied with WAL/SHM to `/tmp`, checkpointed and integrity-checked on the copy, then selected through a `mode=ro`/`query_only` connection. Its 492,367,872-byte pristine snapshot hash was `4d6648f9e3c33e8ffaa7bf15620bd26a18bdf9f7ddf5b97641aa8c082e9aaa67` before and after replay. Raw databases, requests, responses, and logs remain untracked.

Eleven recent organic workflow sessions across `project:ae`, `project:lm`, and `project:online` supplied 46 requests. Each historical transport ID was removed and each workflow ran over a fresh stdio connection, preventing historical session-dedup contamination. The first request of each workflow warmed both versions and established real within-session history; 35 subsequent calls per version were measured. Both servers used the cached production `paraphrase-multilingual-MiniLM-L12-v2` backend offline.

The actual path was `scripts/server.sh → living_memory.server → FastMCP stdio → tools/call memory_recall`, with call order alternated between merge-base `3b2f578` and HEAD.

| Measure | Old | New | Delta | Gate |
| --- | ---: | ---: | ---: | ---: |
| Tool-result content chars, total | 1,168,651 | 670,816 | −42.60% | — |
| Tool-result content chars, median | 24,427 | 14,384 | −41.11% | — |
| Full JSON-RPC wire chars, total | 2,267,900 | 1,298,214 | −42.76% | — |
| Latency p50 | 134.646 ms | 137.313 ms | +1.98% | ≤10%: pass |
| Latency p95 | 482.637 ms | 446.298 ms | −7.53% | ≤10%: pass |

All per-project latency slices also stay within 10%; the largest degradation is `project:online` p95 at +4.55%. Full aggregates and fidelity notes are in `shadow-latency.json`.

## Final merged-state verification

| Evidence | Command | Result |
| --- | --- | ---: |
| Imperative protocol integrity | `npm test -- tests/test_instructions_imperative.py` | 24 passed |
| Superseded-vs-correction invariants | `npm test -- tests/test_retrieval_correction_dominance.py tests/test_dedup_supersede.py` | 26 passed |
| Mixed-era consolidation safety | `npm test -- tests/test_consolidation.py tests/test_schema_distillation.py tests/test_consensus_temporal_decay.py` | 45 passed |
| Repo-global backward compatibility | `npm run check` | 551 passed in 38.82s |

The era suite covers the audited mixed-era no-schema outcome, repair of an existing mixed-era schema, cross-era supersedes splitting, taught-trace exclusion, and non-conflicting promotion. The full suite is green on the merged source state.

## Evidence gap and escalation

`auto_recall.repeat_gating` in the frozen compare output aggregates all repeated fingerprints in a split. It does not emit dev/holdout fingerprint overlap or a holdout aggregate restricted to fingerprints absent from dev. Therefore the observed 59.09% holdout compaction cannot honestly be labeled “unseen in dev,” even though 4,500 holdout events come from predeclared local project scopes absent from dev.

**OUT_OF_SCOPE_ESCALATION:** responsible area: `scripts/ap_baseline.py`, specifically the `family_repeat_gating` / `auto_recall` compare output. Before a future sealed evaluation, it should preload dev automatic fingerprints and emit a privacy-safe unseen-in-dev aggregate: population, ungated chars, gated chars, reduction ratio, and organic safety deltas. This node owns only `artifacts/animal-planet/evaluation/**`, and the authorized holdout read has already occurred. Fixing the evaluator now and rereading this holdout would violate the seal; use a newly authorized or formally re-sealed workload instead.

No observed threshold failed. Overall status remains `evidence_gap`, not `pass`, solely because this required generalization slice is unavailable.

## Fidelity and privacy limits

- Recorded-candidate replay re-mixes historical candidate scores; it does not regenerate BM25/vector/graph candidates.
- Node stats and supersedes edges are snapshot-time rather than event-time.
- Surrogate content preserves lengths and equality classes, but snippet boundaries and causal-query markers are approximate.
- Holdout’s historical before/after payload population is 48 transcript-matched events; all 4,741 events have current replay output but no historical payload comparator.
- The shadow is a 35-call heterogeneous real-workflow distribution, not a repeated-query microbenchmark; schema migration/model startup were excluded symmetrically from call latency.
- Only aggregate JSON and Markdown are tracked. No raw live database, snapshot, transcript, holdout output, request, response, or replay log is committed.
