# Integrate & simplify — final report

Node `scope:retrieval-credit-assignment-graph-value / integrate-simplify`
(postcondition targets **P3, P5, P6**). Machine-checkable companion:
`artifacts/replay/final-report.json`.

**Verdict: all gates PASS.** Proportional credit is a net improvement over
winner-take-all; the typed-edge backfill is replay-neutral (its value is
non-redundant graph reach, unobservable in recorded-score replay by
construction); both floor/rescue band-aids are **retained with measured
justification** — neither is removed silently, and neither survives
undocumented.

Baseline gate anchor (`artifacts/replay/baseline.json`, `live_weights`
holdout, cutoff `2026-06-10T00:00:00Z`): **hit@5 0.947678 / MRR 0.706212**.
The 0.99× gate is therefore **hit@5 ≥ 0.938201, MRR ≥ 0.699150**.

---

## P5 — Combined holdout replay

All replay reports in `artifacts/replay/` share the cutoff
`2026-06-10T00:00:00Z` (asserted programmatically; the lone non-report,
`credit-proportional-weights.json`, is a weight-trajectory dump with no
cutoff field). The three headline configs are replayed on **one 47,011-event
corpus** (holdout = 2,224 labeled events, 1,537 with a confirmed-useful
result), `--credit-rule proportional --supersedes snapshot`, through the real
`rank_candidates` path:

| config | snapshot | holdout hit@5 | holdout MRR | vs baseline | gate |
|---|---|---|---|---|---|
| **A** baseline winner-take-all | non-backfilled | 0.948601 | 0.703801 | 1.0010× / 0.9966× | PASS |
| **B** new-credit (proportional) | non-backfilled | 0.949252 | 0.705807 | 1.0017× / 0.9994× | PASS |
| **C** new-credit + typed-edge backfill | backfilled | 0.948601 | 0.705265 | 1.0010× / 0.9987× | PASS |
| live_weights | non-backfilled | 0.948601 | 0.703785 | 1.0010× / 0.9966× | PASS |
| live_weights | backfilled | 0.947951 | 0.703243 | 1.0003× / 0.9958× | PASS |

Committed child references (each on its own snapshot, for provenance):
`baseline.json` live_weights holdout 0.947678 / 0.706212;
`credit-proportional.json` `replayed_proportional` holdout 0.950424 /
0.707793.

**Reading it.** *B > A*: proportional credit beats winner-take-all on the
same corpus (overall holdout hit@5 +0.000651, MRR +0.002006). *C ≈ B*: the
backfill's replay delta is tiny **by construction** — recorded-score replay
re-ranks the per-result bm25/vector/graph scores captured at event time, so
only the 9 new R1c `supersedes` pairs enter ranking; the 10,126 new `related`
edges, 4 `caused` edges, and 5,228 R5a annotations never touch the recorded
candidate set.

Per-scope holdout (honest, including the wobbles):

| scope (holdout events) | A wta | B prop | C prop+backfill |
|---|---|---|---|
| project:lm (23) | 0.947368 / 0.867168 | 0.947368 / **0.874269** | 0.947368 / 0.874269 |
| project:x (1535) | 0.962963 / 0.704902 | **0.964573 / 0.708350** | 0.964573 / 0.708350 |
| project:online (482) | 0.870370 / 0.630713 | 0.864198 / 0.626524 | 0.864198 / 0.626524 |
| global (32) | 0.904762 / 0.850000 | 0.904762 / 0.850529 | 0.857143 / 0.810847 |

Proportional credit lifts the focus scopes (project:lm MRR +0.0071,
project:x both channels up); project:online dips slightly; global (a
32-event scope) dips under the backfill from `supersedes` demotion of
superseded nodes — the intended direction, on a small sample. Every
config's **overall** holdout clears the gate.

### Graph-unique contribution

`graph_unique_share` = **0.010349** on the current corpus, identical across
A/B/C and essentially equal to the committed baseline's 0.010391 (the hair of
difference is corpus drift, 46,821 → 47,011 events). This metric is
**scheme-invariant** in recorded-score replay — it is a property of the
recorded per-result evidence (graph-only = `graph_score>0, bm25=vector=0`),
so it *cannot rise* in replay no matter the ranking change (the documented
candidate-selection-bias limitation). It is **not decreased**. The realized
graph-value gain lives outside recorded-score replay: the backfill adds
**9,098 net-new non-redundant edges** (pairs with no prior edge in either
direction) and lifts provenance-derived edge share from 3.20% → **16.48%**,
exercised at live recall time by graph expansion.

---

## P3 — Band-aid simplification (evidence-gated)

Sweep substrate: backfilled snapshot `/tmp/lm_bf.sqlite3`, proportional
credit, `--supersedes snapshot`, holdout split. Reference **R0** = current
code (rescue + floors intact): live_weights 0.947951 / 0.703243,
replayed_proportional 0.948601 / 0.705265.

### Rescue (`retrieval.py:224-232`) — **KEEP**

| variant | live_weights h5/MRR | replayed_proportional h5/MRR |
|---|---|---|
| R0 rescue on | 0.947951 / 0.703243 | 0.948601 / 0.705265 |
| E1 causal-only | 0.946649 / 0.700532 | 0.946649 / 0.701717 |
| E2 full removal | 0.946649 / 0.700532 | 0.946649 / 0.701717 |

E1 and E2 are **identical**: in the holdout the causal 0.75 arm changes no
useful-result ranking, so the rescue's entire measurable value is the
non-causal 0.25 arm. Both variants **regress** vs R0 (live_weights MRR
0.703243 → 0.700532; replayed_proportional MRR 0.705265 → 0.701717). So the
rescue still carries positive holdout value **even under honest credit +
typed edges** — retained on evidence, not merely on test-compatibility.
Removing/narrowing it would also break the tests that pin it —
`test_replay_harness.py::test_rerank_uses_real_ranking_path` and
`::test_recorded_scheme_metrics_hand_computed` (non-causal 0.25 survival),
and the causal rank-1 tests in `test_graph_recall.py` /
`test_acceptance_contract.py` — which gate (3) forbids weakening.
**Double-disqualified: replay regression + existing-test breakage.**

### Per-scope floors (`config.py:42-46` + `storage.apply_retrieval_weight_floors`) — **KEEP**

| variant | live_weights h5/MRR | replayed_proportional h5/MRR |
|---|---|---|
| R0 floors current | 0.947951 / 0.703243 | 0.948601 / 0.705265 |
| F1 graph_min→0 | 0.947951 / 0.703243 | 0.948601 / **0.705874** |
| F2 wide rails | 0.947951 / 0.703243 | **0.949902 / 0.705821** |

Narrowing the rails is replay-**neutral-to-slightly-positive** — there is no
holdout regression, because the proportional-credit mean-share fixed point
already does the real work *within* the current rails, so the floors are no
longer binding on the fixed point. But the *specific* floor values are pinned
by exact-value assertions in **nine** existing test files
(`test_retrieval_bm25_floor`, `test_retrieval_weight_maintenance`,
`test_feedback_weights`, `test_health_retrieval_skew`,
`test_retrieval_semantic_recall`, `test_retrieval_lexical_recall`,
`test_retrieval_feedback_amplification`, `test_graph_recall`,
`test_replay_harness`). Changing the defaults would rewrite those assertions
— forbidden by gate (3). And those tests encode properties (exact-identifier
recall survival, semantic-recall floor, graph floor) that the grounded
holdout metric does not fully exercise, so the floors are retained as narrow
guardrails, **not** as compensation for a starved signal.

**Operator path (no automatic rewrites):** narrowing the persisted rails, if
desired, is an audited operator action via
`maintenance.reseat_retrieval_floors` (dry-run is the default; apply writes
only rows whose before ≠ after, one per-scope `set_retrieval_weights` each).
This tree performs **no** automatic weight rewrites.

---

## P6 — Safety invariants

### MCP surface — unchanged
`tests/test_mcp_server.py` + `tests/test_fastmcp_runtime.py` → **8 passed**.
The whole-tree `server.py` diff vs master is one import plus a
`try/except derive_edges_for_new_trace(store, node)` inside the existing
`memory_remember` write path — **no** tool registration, name, or signature
change.

### Migration safety — no weight reset / bulk-rewrite
The only SQL writers to `retrieval_weights` are the per-scope UPSERT
`set_retrieval_weights` (`storage.py:1074`, `INSERT … ON CONFLICT(scope) DO
UPDATE`) and the idempotent `_seed_retrieval_weights` (`storage.py:1467`,
`… DO NOTHING`). No `DELETE` / `REPLACE` / bulk `UPDATE` / `TRUNCATE` exists
anywhere in `src/`. **This tree added none of these paths**: `feedback.py`
changed only the `_method_signals` arithmetic (winner-take-all → proportional
shares, still feeding the pre-existing per-scope `update_retrieval_weights`);
`storage.py` added only two additive partial indexes; `config.py` and
`maintenance.py` are untouched.

### Latency — after ≤ 1.10× before
Methodology `artifacts/discovery/profile_recall.py` (real SentenceTransformer
backend, as live; 2 warmups + 9 iters; median(total) = p50), on one
`sqlite backup` copy of the live DB, before = master `@394ad4c`, after = this
tree `@2d375a6`:

| pass | scope | before p50 | after p50 | ratio |
|---|---|---|---|---|
| no-write | global | 109.11 | 106.82 | 0.979 |
| no-write | project:ae | 157.32 | 151.79 | 0.965 |
| no-write | project:lm | 71.51 | 74.32 | 1.039 |
| no-write | project:octopus d0 | 125.36 | 117.14 | 0.934 |
| no-write | project:octopus d1 | 188.80 | 191.96 | 1.017 |
| write | global | 112.12 | 107.87 | 0.962 |
| write | project:ae | 154.78 | 154.27 | 0.997 |
| write | project:lm | 79.26 | 77.41 | 0.977 |
| write | project:octopus d0 | 121.65 | 123.22 | 1.013 |
| write | project:octopus d1 | 199.07 | 197.86 | 0.994 |

Worst per-scope ratio **1.039×** (project:lm, the smallest scope — within
run-to-run noise; its write-pass ratio is 0.977×); write-pass aggregate p50
121.65 → 123.22 ms (1.013×). All ≤ 1.10×. The tree's recall-path change is
two `metadata.get('kind')`-keyed `elif` branches in `_traversal`; the two new
indexes serve the write path only.

---

## Full suite — green, nothing weakened
`npm run check` → **332 passed in 26.78s** (0 failed, 0 skipped, 0 xfailed).
Among existing test files only `tests/test_feedback_weights.py` was modified
(the single `-1` is its import line being expanded with `_method_signals`;
everything else is five *added* tests) — **no existing assertion weakened**.
The only skip marker in the tree diff is an inactive
`@pytest.mark.skipif(not BASELINE_PATH.exists())` in the new
`tests/test_replay_harness.py`; `baseline.json` exists, so it does not fire
(the suite reports 0 skipped).

## Reproduce
```bash
export PYTHONPATH=src LIVING_MEMORY_EMBEDDING_BACKEND=hash
# Combined holdout (same corpus; non-backfilled twin + backfilled snapshot):
python3 -m living_memory.replay --db /tmp/lm_backfill_copy.sqlite3 \
  --cutoff 2026-06-10T00:00:00Z --credit-rule proportional --supersedes snapshot \
  --schemes recorded,live_weights,replayed_winner_take_all,replayed_proportional \
  --report artifacts/replay/combined-holdout-nonbackfilled.json
python3 -m living_memory.replay --db /tmp/lm_bf.sqlite3 \
  --cutoff 2026-06-10T00:00:00Z --credit-rule proportional --supersedes snapshot \
  --schemes recorded,live_weights,replayed_winner_take_all,replayed_proportional \
  --report artifacts/replay/combined-holdout-backfilled.json
# Latency (real backend), before on master worktree, after on this tree:
unset LIVING_MEMORY_EMBEDDING_BACKEND
PYTHONPATH=/tmp/lm-master/src python3 artifacts/discovery/profile_recall.py /tmp/lm-profile-copy.sqlite3
PYTHONPATH=src           python3 artifacts/discovery/profile_recall.py /tmp/lm-profile-copy.sqlite3
```
