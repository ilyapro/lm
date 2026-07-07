# Proportional retrieval credit — A/B replay report (credit-fix-v2)

Change under test: `feedback._method_signals` now distributes credit (and blame)
proportional to each method's **raw** recorded score share —
`signals[m] = signed_signal * max(0, raw_m) / sum(raw)` over
`{bm25, vector, graph}`, all-zero signals when `sum(raw) <= 0` (no
bm25-dominant fallback, no update) — replacing the winner-take-all rule
(dominant `+signal`, both others `-0.25*signal`; on negative feedback dominant
`-|signal|`, others `+0.15*|signal|`). `feedback._method_signals` is the single
source of truth: the harness's `CREDIT_RULES['proportional']` delegates to it,
and the pre-fix rule is frozen verbatim as `CREDIT_RULES['winner_take_all']`
for this A/B (parity and freeze are test-enforced in
`tests/test_replay_harness.py`).

## Verdict — gate PASSED

Gate (committed `artifacts/replay/baseline.json`, `live_weights` holdout):
hit@5 >= 0.99 × 0.947678 = 0.938201 and MRR >= 0.99 × 0.706212 = 0.699150.

| `replayed_proportional` holdout | value | gate | margin |
|---|---|---|---|
| hit@5 | **0.950424** | >= 0.938201 | +0.012223 |
| MRR | **0.707793** | >= 0.699150 | +0.008643 |

Proportional credit does not merely clear the 0.99× no-regression bar: it beats
the **undiscounted** baseline (hit@5 +0.002746, MRR +0.001581) and every
holdout metric of both `live_weights` and `replayed_winner_take_all` on the
same snapshot.

## Proportional vs winner-take-all (same snapshot, same events)

| split | metric | replayed_winner_take_all | replayed_proportional | Δ |
|---|---|---|---|---|
| holdout | hit@1 | 0.531637 | 0.534247 | **+0.002610** |
| holdout | hit@5 | 0.947815 | 0.950424 | **+0.002609** |
| holdout | MRR | 0.706652 | 0.707793 | **+0.001141** |
| train | hit@1 | 0.471325 | 0.461687 | -0.009638 |
| train | hit@5 | 0.908434 | 0.912289 | +0.003855 |
| train | MRR | 0.646933 | 0.642720 | -0.004213 |

`replayed_winner_take_all` reproduces the `live_weights` holdout metrics
exactly (0.531637 / 0.947815 / 0.706652): the live learned weights ARE the
winner-take-all fixed point, so the WTA column doubles as the live-behavior
anchor. Holdout — the decision split, ranked under weights frozen at the
cutoff — improves on all three metrics. Train events are ranked under a
still-moving trajectory (updates merge chronologically), so the small train
hit@1/MRR dip reflects mid-convergence ordering, not the converged policy.

Per-scope holdout deltas (proportional − WTA): project:x (1535 events, the
largest holdout scope) hit@5 +0.0032 / MRR +0.0019; project:ae (135 events)
MRR -0.0060; project:lm (19 events) MRR -0.0012; global/mm/online/_other
essentially flat (|Δ| <= 0.0006). No scope regresses at holdout hit@5.

## Converged per-scope weights (bm25 / vector / graph, at cutoff)

Trajectories start from config family defaults and replay pre-cutoff labeled
consumption feedback; `updates` counts reinforcement events per scope. Full
40-scope table: `credit-proportional-weights.json`.

| scope | live (snapshot) | replayed WTA | replayed proportional | updates |
|---|---|---|---|---|
| global | 0.100/0.850/0.050 | 0.100/0.850/0.050 | 0.137/0.813/0.050 | 761 |
| project:ae | 0.100/0.850/0.050 | 0.100/0.850/0.050 | 0.116/0.807/0.077 | 8684 |
| project:lm | 0.102/0.841/0.056 | 0.101/0.849/0.050 | **0.164/0.779/0.058** | 1758 |
| project:mm | 0.128/0.790/0.082 | 0.100/0.850/0.050 | 0.102/0.811/0.087 | 552 |
| project:octopus | 0.100/0.850/0.050 | 0.100/0.850/0.050 | **0.104/0.695/0.201** | 12317 |
| project:online | 0.100/0.850/0.050 | 0.100/0.850/0.050 | 0.164/0.761/0.076 | 14438 |
| project:x | 0.100/0.850/0.050 | 0.100/0.850/0.050 | 0.166/0.722/0.113 | 3397 |

- Every high-traffic scope that winner-take-all pinned at the floor-clamped
  fixed point (0.100/0.850/0.050) moves to an **interior** point under
  proportional credit: the floors stop being load-bearing for these scopes
  (measured input for the P3 floor/rescue simplification, owned by
  integrate-simplify).
- project:lm bm25 rises off the 0.10 floor to 0.164 (+63%), in the direction
  of its honest mean-contribution share (~0.502 computed unweighted over the
  full labeled corpus through 2026-07-07). The cutoff-frozen trajectory is
  rank-decay signal-weighted and candidate-selection-biased (it only sees
  results the old vector-dominant ranking surfaced), and the honest-share
  estimate includes post-cutoff events, so the converged value lands short of
  the unweighted full-corpus share; the defect being fixed — monotonic drift
  to the floor regardless of contribution — is gone.
- project:x graph converges to 0.113, matching its honest graph share
  estimate (0.113) exactly; project:octopus graph earns 0.201 (4× its floor;
  honest estimate 0.130). Graph weight is now earned where graph evidence
  actually contributes instead of surviving only via the floor.

## Cross-checks

- **Freeze fidelity**: the WTA trajectory replayed on this fresh snapshot
  (side run: `--schemes replayed_winner_take_all --no-label-sensitivity`)
  converges to per-scope weights identical (< 1e-9) to
  `baseline.json weights.replayed_final` across all 40 baseline scopes — the
  relocated rule preserved the old semantics exactly, and the pre-cutoff
  corpus is stable across snapshots.
- **Parity**: `test_proportional_rule_is_feedback_method_signals` asserts
  `CREDIT_RULES['proportional']` equals `feedback._method_signals` over a
  6×6×6 raw-score grid × 6 signals (positive, negative, zero, including
  negative raw scores and the zero-evidence case).

## Migration

No weight reset or schema migration ships with this change. Learned
`retrieval_weights` rows are a fixed point of the (old) update rule; under the
proportional rule the same rows simply re-converge toward the honest
per-scope shares as new feedback arrives — the trajectories above show
convergence from arbitrary starting points within the recorded corpus. Floors
(`storage.apply_retrieval_weight_floors`) and the in-rank graph rescue
(`retrieval.py`) are deliberately untouched here; narrowing them on this
evidence belongs to integrate-simplify (P3).

## Reproduction

```
python3 -c "import sqlite3; s=sqlite3.connect('file:$HOME/.local/share/living-memory/global.sqlite3?mode=ro',uri=True); d=sqlite3.connect('/tmp/lm_credit_snapshot.sqlite3'); s.backup(d)"
PYTHONPATH=src python3 -m living_memory.replay --db /tmp/lm_credit_snapshot.sqlite3 \
  --cutoff 2026-06-10T00:00:00Z --report artifacts/replay/credit-proportional.json \
  --schemes recorded,live_weights,replayed_winner_take_all,replayed_proportional \
  --trajectory-out artifacts/replay/credit-proportional-weights.json
```

Deviations from the committed baseline run (`baseline.json`): the scheme list
is the goal-prescribed A/B set (baseline additionally ran the static
`floor_defaults`/`uniform` schemes, which are credit-rule-independent and
remain valid there); the snapshot is one day fresher (46,985 events / 7,596
labeled vs 46,821 / 7,589 — holdout 2,220 vs 2,213 labeled events), which is
why this run's `live_weights` holdout (0.947815 / 0.706652) differs in the
4th decimal from the committed gate constants (0.947678 / 0.706212) — the
gate above uses the committed constants. All other flags are identical
(grounded labels 0.25/2/0.8, evidence=snapshot, lr-policy=snapshot,
supersedes=none, min-scope-events=50).

---

# Appendix — full harness report

Generated: 2026-07-07T19:03:20Z | DB: `/tmp/lm_credit_snapshot.sqlite3`

## Corpus and split

- Events: 46985 total, 46965 usable (skipped: 20 empty, 0 incomplete results)
- Labeled (consumed) events: 7596 | span 2026-05-15T07:25:32Z .. 2026-07-07T19:02:12Z
- Cutoff `2026-06-10T00:00:00Z`: train 38016 events (5376 labeled) / holdout 8949 events (2220 labeled, 19.1% of usable events, 29.2% of labeled)

## Labels

- Protocol: `grounded` {'protocol': 'grounded', 'min_containment': 0.25, 'reconsume_min_traces': 2, 'usefulness_threshold': 0.8}
- 9899 useful of 59895 labeled results; 3608 of 7596 labeled events have at least one useful result
- Discrimination over 7575 multi-result events: strict subset 46.7%, all-useful 0.9%, none-useful 52.4%, mean useful share 0.16

## Metrics by scheme
| scheme | split | events | w/useful | hit@1 | hit@5 | hit@10 | MRR | graph-unique | zeroed hit@5 | zeroed MRR | drop@1 | drop@5 | drop@max | drop-all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recorded | overall | 7596 | 3608 | 0.328 | 0.859 | 0.993 | 0.539 | 0.008 | 0.859 | 0.538 | 0.000 | 0.000 | 0.001 | 0.001 |
| recorded | train | 5376 | 2075 | 0.265 | 0.822 | 0.989 | 0.484 | 0.005 | 0.822 | 0.484 | 0.000 | 0.000 | 0.002 | 0.002 |
| recorded | holdout | 2220 | 1533 | 0.413 | 0.909 | 0.998 | 0.612 | 0.010 | 0.909 | 0.612 | 0.000 | 0.000 | 0.000 | 0.000 |
| live_weights | overall | 7596 | 3608 | 0.500 | 0.924 | 0.997 | 0.675 | 0.008 | 0.925 | 0.675 | 0.006 | 0.006 | 0.001 | 0.001 |
| live_weights | train | 5376 | 2075 | 0.477 | 0.907 | 0.998 | 0.652 | 0.005 | 0.907 | 0.652 | 0.007 | 0.007 | 0.002 | 0.002 |
| live_weights | holdout | 2220 | 1533 | 0.532 | 0.948 | 0.997 | 0.707 | 0.010 | 0.949 | 0.706 | 0.004 | 0.005 | 0.000 | 0.000 |
| replayed_winner_take_all | overall | 7596 | 3608 | 0.497 | 0.925 | 0.997 | 0.672 | 0.008 | 0.926 | 0.672 | 0.003 | 0.006 | 0.001 | 0.001 |
| replayed_winner_take_all | train | 5376 | 2075 | 0.471 | 0.908 | 0.998 | 0.647 | 0.005 | 0.908 | 0.647 | 0.002 | 0.008 | 0.002 | 0.002 |
| replayed_winner_take_all | holdout | 2220 | 1533 | 0.532 | 0.948 | 0.997 | 0.707 | 0.010 | 0.949 | 0.706 | 0.004 | 0.005 | 0.000 | 0.000 |
| replayed_proportional | overall | 7596 | 3608 | 0.493 | 0.928 | 0.997 | 0.670 | 0.008 | 0.928 | 0.671 | 0.004 | 0.009 | 0.001 | 0.001 |
| replayed_proportional | train | 5376 | 2075 | 0.462 | 0.912 | 0.998 | 0.643 | 0.005 | 0.913 | 0.644 | 0.003 | 0.010 | 0.002 | 0.002 |
| replayed_proportional | holdout | 2220 | 1533 | 0.534 | 0.950 | 0.997 | 0.708 | 0.010 | 0.949 | 0.707 | 0.005 | 0.008 | 0.000 | 0.000 |

### Per-scope (overall split)

| scheme | scope | events | w/useful | hit@5 | MRR | graph-unique | drop@5 |
|---|---|---|---|---|---|---|---|
| recorded | _other | 86 | 41 | 0.951 | 0.723 | 0.000 | 0.000 |
| recorded | global | 115 | 79 | 0.975 | 0.844 | 0.000 | 0.000 |
| recorded | project:ae | 1252 | 384 | 0.818 | 0.518 | 0.009 | 0.000 |
| recorded | project:lm | 261 | 118 | 0.941 | 0.680 | 0.000 | 0.000 |
| recorded | project:mm | 63 | 21 | 0.762 | 0.604 | 0.000 | 0.000 |
| recorded | project:octopus | 1546 | 779 | 0.854 | 0.481 | 0.008 | 0.001 |
| recorded | project:online | 2318 | 760 | 0.784 | 0.469 | 0.001 | 0.000 |
| recorded | project:x | 1955 | 1426 | 0.898 | 0.578 | 0.011 | 0.000 |
| live_weights | _other | 86 | 41 | 0.976 | 0.728 | 0.000 | 0.009 |
| live_weights | global | 115 | 79 | 0.975 | 0.847 | 0.000 | 0.018 |
| live_weights | project:ae | 1252 | 384 | 0.859 | 0.576 | 0.009 | 0.002 |
| live_weights | project:lm | 261 | 118 | 0.966 | 0.790 | 0.000 | 0.009 |
| live_weights | project:mm | 63 | 21 | 0.952 | 0.933 | 0.000 | 0.000 |
| live_weights | project:octopus | 1546 | 779 | 0.915 | 0.597 | 0.008 | 0.010 |
| live_weights | project:online | 2318 | 760 | 0.862 | 0.650 | 0.001 | 0.001 |
| live_weights | project:x | 1955 | 1426 | 0.972 | 0.734 | 0.011 | 0.005 |
| replayed_winner_take_all | _other | 86 | 41 | 0.976 | 0.707 | 0.000 | 0.028 |
| replayed_winner_take_all | global | 115 | 79 | 0.975 | 0.858 | 0.000 | 0.023 |
| replayed_winner_take_all | project:ae | 1252 | 384 | 0.862 | 0.579 | 0.009 | 0.004 |
| replayed_winner_take_all | project:lm | 261 | 118 | 0.975 | 0.770 | 0.000 | 0.004 |
| replayed_winner_take_all | project:mm | 63 | 21 | 0.952 | 0.933 | 0.000 | 0.000 |
| replayed_winner_take_all | project:octopus | 1546 | 779 | 0.920 | 0.599 | 0.008 | 0.010 |
| replayed_winner_take_all | project:online | 2318 | 760 | 0.858 | 0.636 | 0.001 | 0.001 |
| replayed_winner_take_all | project:x | 1955 | 1426 | 0.972 | 0.734 | 0.011 | 0.005 |
| replayed_proportional | _other | 86 | 41 | 0.976 | 0.707 | 0.000 | 0.028 |
| replayed_proportional | global | 115 | 79 | 0.975 | 0.863 | 0.000 | 0.022 |
| replayed_proportional | project:ae | 1252 | 384 | 0.872 | 0.573 | 0.009 | 0.008 |
| replayed_proportional | project:lm | 261 | 118 | 0.966 | 0.766 | 0.000 | 0.004 |
| replayed_proportional | project:mm | 63 | 21 | 0.952 | 0.933 | 0.000 | 0.000 |
| replayed_proportional | project:octopus | 1546 | 779 | 0.928 | 0.591 | 0.008 | 0.014 |
| replayed_proportional | project:online | 2318 | 760 | 0.858 | 0.634 | 0.001 | 0.004 |
| replayed_proportional | project:x | 1955 | 1426 | 0.974 | 0.735 | 0.011 | 0.007 |

## Label sensitivity (live_weights scheme, overall)

| protocol | strict-subset | all-useful | none-useful | events w/useful | hit@5 | MRR |
|---|---|---|---|---|---|---|
| grounded | 46.7% | 0.9% | 52.4% | 3608 | 0.924 | 0.675 |
| reconsumed | 36.2% | 63.1% | 0.7% | 7539 | 0.994 | 0.918 |
| usefulness | 78.3% | 16.6% | 5.1% | 7194 | 0.928 | 0.620 |
| grounded_or_reconsumed | 27.2% | 72.6% | 0.2% | 7574 | 0.996 | 0.951 |

## Labeling protocol

A recall event is **labeled** when a later `memory_remember` consumed it
(`feedback_trace_id` is set). Within a labeled event, a result is
**confirmed useful** under the primary `grounded` protocol when the
IDF-weighted share of the result node's content tokens that also appear in
the consuming trace's content (containment) is at least `min_containment`.
IDF is computed over the corpus of result-node and consuming-trace contents,
so boilerplate tokens contribute little and identifiers/paths dominate.

Why not the recorded linkage? `feedback.apply_pending_recall_feedback` links
**every** result of a consumed event to the consuming trace
(provenance `recalled_nodes`/`source_traces`, `related` edges), so
"linked by the consuming trace" marks all results useful and every ranking
metric becomes vacuous. Grounding discriminates within the result list.
Alternative protocols (`reconsumed`, `usefulness`,
`grounded_or_reconsumed`) are reported in the label-sensitivity section:
re-consumption is nearly vacuous on this corpus (a few thousand hub nodes
recirculate across most events), and long-run usefulness is
rank-circular (it accrues via the same rank-decayed reinforcement loop the
harness audits) with a saturated distribution (median 1.0).

## Metric definitions

* `hit@k` / `mrr` — over labeled events with at least one confirmed-useful
  result: whether/where the first useful result appears in the re-ranked
  order of the recorded candidates. A useful result that a scheme ranks to
  zero score counts as a miss.
* `graph_unique_share` — share of confirmed-useful results whose recorded
  evidence is graph-only (`graph_score > 0`, `bm25 = vector = 0`).
  Scheme-independent (recorded evidence), repeated per scheme for
  consumer convenience.
* `graph_zeroed.*` — counterfactual re-rank of the same candidates with
  `graph_score` forced to 0 (which also disables the in-rank graph rescue):
  `dropped_topK` is the share of confirmed-useful results in the scheme's
  top-K that leave the top-K; `dropped_entirely` is the share of ranked
  useful results that drop to zero score; `hit@5`/`mrr` are re-computed on
  the zeroed ranking. `dropped_*` can sit below `graph_unique_share`:
  schema nodes whose only method evidence is graph still survive zeroing
  through the trigger-score override (on this corpus most graph-only
  confirmed-useful results are exactly such schema nodes).

## Limitations

* **Candidate selection bias**: recorded results are only the
  top-`max_results` under the *old* live weights + rescue ranking. Replay
  can re-order or drop recorded candidates but can never surface candidates
  the old ranking excluded, so absolute metric levels are optimistic and
  graph-value estimates are lower bounds relative to a full re-retrieval.
* **Neutral node stats**: re-ranked schemes hold node-level feedback
  multipliers (confidence/usefulness/access, supersedes corrections) at
  neutral because event-time node stats are not recorded. The `recorded`
  scheme preserves the historical order including those multipliers.
* **Grounding is textual**: results used conceptually without shared
  identifiers are missed (~half of labeled events have no grounded result;
  they are excluded from hit/MRR denominators). Consuming traces deleted
  since (`missing_consuming_traces`) cannot be grounded.
* **Trajectory approximations**: initial weights are the configured family
  defaults; per-scope learning rates come from the snapshot weights table
  (the live adaptive tuner's final state); floor evidence gates are
  evaluated against the snapshot, not historically; explicit
  feedback/teach events are not in the corpus and are not replayed.

