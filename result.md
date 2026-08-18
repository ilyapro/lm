# feedback-credit-grounding — result

Content grounding moves from the offline replay harness into the live credit
assignment loop: a recall result is reinforced only when the consuming trace
actually used it. Linkage, provenance and graph edges stay exhaustive — only
*reinforcement* is gated.

**Outcome: shipped with `LM_RECALL_CREDIT_POLICY=grounded` as the default.**
On the recorded history 87.9% of every reinforcement the live loop has ever
applied went to results the consuming trace never used; that share now earns
nothing. The A/B arm that additionally *penalised* unused results
(`grounded_negative`) was measured and **rejected** — it drives learned weights
into degenerate corners (graph 0.75 on three scopes).

Every number below is quoted with the artifact it came from and the command
that produced it. No experiment touched the live database; all measurements
run on `~/.cache/living-memory-harness/snapshot.sqlite3` (SQLite backup-API
snapshot, sha256 `d46f1b845adc3d80…`, 54,777 recall events,
2026-05-14 → 2026-08-17) or on the frozen phase-1 harness snapshot.

---

## 1. Acceptance at a glance

| Acceptance item | Bar | Measured | Source | Verdict |
|---|---|---|---|---|
| Regression test fails on old code | must fail | `assert [0,1,2,3,4,5,6,7] == [3]` — all 8 results reinforced under the pre-grounding default, 1 under the new one; 3 tests fail | §2, `tests/test_grounded_credit_assignment.py` | PASS |
| Grounding logic shared, not duplicated | one implementation | `replay._containment` and `feedback.apply_pending_recall_feedback` both call `living_memory.grounding`; pinned by source-inspection tests | §3, `tests/test_grounding.py` | PASS |
| `min_containment` fixed with numbers | value + rationale | **0.25** (replay default kept); precision 0.978 / recall 0.768 vs the corpus-IDF reference, F1 optimum 0.225 deliberately not taken | §4, `artifacts/grounding/calibration.{json,md}` | PASS |
| Replay A/B, cutoff holdout, grounded ≥ incumbent | ≥ incumbent | holdout hit@5 **0.9300 = 0.9300**, MRR **0.7000 vs 0.6921** (+0.0079, bootstrap CI [+0.0034, +0.0126]) | §5, `artifacts/grounding/credit-ab.{json,md}` | PASS |
| Ungrounded fate decided by A/B | pick an arm | **neutral** (`grounded`); `grounded_negative` rejected on weight degeneracy + no significant primary gain | §5 | PASS |
| E2E harness hit@5 / MRR ≥ phase1 | ≥ 0.577 / 0.354 | **0.576923 / 0.353929** — delta **0.000000** on every metric, agreement 1.000 | §6, `artifacts/grounding/e2e-gate.json` | PASS |
| `memory_remember` p50 ≤ +10 ms | ≤ +10 ms | **−4.20 ms** (real model, n=150) and **−4.38 ms** (hash, n=400): faster, not slower | §7, `artifacts/grounding/remember-latency.json` | PASS |
| Existing tests green | all | **867 passed, 83 skipped** (was 839/83; +28 new). Includes `test_transport_feedback_closure_e2e` and `test_retrieval_weight_recalibration` | §8 | PASS |
| Feedback-closure tests not broken | unchanged behaviour | closure/linkage assertions all hold; three tests updated because they asserted *reinforcement* as a proxy for *linkage* | §8 | PASS |
| Handoff for "query anchors in the graph" | written | §10, incl. retro-labelling volume and quality estimate | §10 | PASS |

---

## 2. The defect, quantified

`apply_pending_recall_feedback` walked `for rank, result in enumerate(event.results)`
and gave **every** result a positive signal of `max(0.2, 1/(rank+1))` into both
node usefulness and `update_retrieval_weights`. On the snapshot:

| Quantity | Value |
|---|---|
| Recall events recorded | 54,777 (54,757 usable) |
| Events consumed by a later `memory_remember` | 9,069 (16.6%) |
| Result rows in those events (all reinforced by the old rule) | 72,023 |
| Result rows the consuming trace actually grounded | **8,698 (12.08%)** |
| Reinforcements that were noise | **63,325 (87.9%)** |
| Consumed events where *nothing* was grounded | 5,388 of 9,069 (59.4%) |

The repo already documented this: `replay.py`'s module docstring calls
"linked by the consuming trace" *vacuous as a usefulness label*. That
judgement is now enforced in the mechanism, not only in the measurement.

**Regression test (home rule: must fail on old code).**
`tests/test_grounded_credit_assignment.py` seeds 8 nodes sharing a two-token
anchor, delivers all 8 in one recall event, and consumes it with a trace that
quotes exactly one node's body. Reproduced by flipping
`DEFAULT_RECALL_CREDIT_POLICY` to `"all"` (the pre-grounding rule, still
reachable):

```
assert gained == [USED_INDEX]
E  assert [0, 1, 2, 3, 4, 5, ...] == [3]
E    Left contains 7 more items, first extra item: 1
3 failed: test_only_the_grounded_result_is_reinforced,
          test_grounded_policy_leaves_ungrounded_weights_untouched,
          test_policy_surface_is_pinned
```

The test asserts both arms in one run, so it also fails if the two policies
ever collapse into one. A fixture control (`test_fixture_grounds_exactly_one_of_the_eight_bodies`)
pins that the used node scores > 0.6 containment and the other seven < 0.15,
so the headline test cannot pass for a threshold-luck reason.

---

## 3. What changed

**New `src/living_memory/grounding.py`** — the single implementation:
`token_set`, `build_idf`, `containment`, `ground_token_sets` (pre-tokenized
entry, used by replay), `ground_results` (text entry, used by the live path).

**`src/living_memory/feedback.py`** — `apply_pending_recall_feedback` resolves
each event's results once, grades the whole event with one shared IDF index,
then reinforces only the grounded ones. Policies via `LM_RECALL_CREDIT_POLICY`:
`grounded` (default), `grounded_negative`, `all` (operator fallback to the old
rule without a code change). New: `UNGROUNDED_NEGATIVE_FACTOR = 0.25`,
`ungrounded_negative_signals()`, and `ImplicitRecallFeedback.grounded_node_ids`.

**`src/living_memory/replay.py`** — `_containment` and `build_label_data`
delegate to the shared module (the local IDF arithmetic is deleted, not
copied). `ReplayResult` gains `grounded`/`containment`, set by the new
`apply_grounding()` which replays the *live* per-event view. New credit rules
`grounded` and `grounded_negative`; `CreditRule` may now return `None` meaning
"no update at all", which `WeightTrajectory.reinforce_event` honours.
`MetricAccumulator` records per-event `(id, reciprocal_rank, hit@5)` so paired
significance tests are possible.

**Untouched, as required:** `feedback_weighted_score`,
`FEEDBACK_MULTIPLIER_CAP`, the weight floors, `retrieval.py`, and the DB
schema. `memory_teach` still passes `reinforce_results=False`, which now also
skips grading entirely.

**Deliberately not changed: linkage.** `recalled_nodes`, `source_traces`,
`prior_recalls`, and the rank-weighted `related` edges still cover every
delivered result. `test_linkage_still_covers_every_delivered_result` pins all
four.

---

## 4. Calibration: why `min_containment` stays 0.25

The two callers cannot share a corpus. Replay owns all 8,623 labeled documents
and builds one global IDF; the live write path holds only the consuming trace
and that event's result nodes, so it builds a per-event IDF. Same arithmetic,
different corpus — so the divergence was measured rather than assumed.

`python3 scripts/grounding_calibration.py --db ~/.cache/living-memory-harness/snapshot.sqlite3 --report artifacts/grounding/calibration.json`
over **72,023 result/trace pairs / 9,035 consumed events**:

- Pearson correlation, live vs corpus containment: **0.9844**
- Reference (corpus IDF ≥ 0.25): 11,073 positives (15.37% of pairs)

| live threshold | positives | precision | recall | F1 | agreement |
|---|---|---|---|---|---|
| 0.200 | 13,685 | 0.7837 | 0.9686 | 0.8664 | 0.9541 |
| 0.225 | 10,849 | 0.9149 | 0.8964 | **0.9056** | **0.9713** |
| **0.250** | **8,698** | **0.9776** | 0.7679 | 0.8601 | 0.9616 |
| 0.275 | 6,976 | 0.9923 | 0.6251 | 0.7670 | 0.9416 |

**Adopted: 0.25, the replay default — not recalibrated.** F1 peaks at 0.225,
but F1 is the wrong objective for a credit rule: a false positive is precisely
the defect being fixed (noise re-entering the learned signal), while a false
negative merely withholds a signal that recurs at the next consumption. 0.25 is
the precision-favouring point (0.978 precision, 195 false positives against
11,073 reference positives). Both columns are proxies for "the agent used
this"; neither is ground truth, and the sweep is reported as agreement, not
accuracy.

---

## 5. Replay A/B: which credit rule

`python3 scripts/credit_rule_ab.py --db ~/.cache/living-memory-harness/snapshot.sqlite3 --cutoff 2026-06-10T00:00:00Z --report artifacts/grounding/credit-ab.json`

Three arms replay the identical event stream through the identical weight
arithmetic, differing only in who earns credit. Weight updates stop at the
cutoff, so the **3,693 holdout events (2,015 with a useful result)** are ranked
under weights frozen before the arm ever saw them.

### Holdout, primary (`grounded`) label

| arm | hit@1 | hit@5 | MRR | weight updates |
|---|---|---|---|---|
| `proportional` (incumbent) | 0.5211 | 0.9300 | 0.6921 | 42,348 |
| **`grounded`** | **0.5325** | **0.9300** | **0.7000** | **3,842** |
| `grounded_negative` | 0.5161 | 0.9280 | 0.6870 | 42,348 |

The headline is the last column: `grounded` matches hit@5 and beats MRR using
**9.1% of the updates**. The discarded 91% was not carrying ranking signal.

### Falsifiability control

`grounded` gates on grounding and the primary label derives from grounding, so
a win there could be definitional. Every arm is therefore also scored under two
labels that use no grounding at all, with a paired bootstrap (2,000 resamples
over events, seed 20260818) — a gap counts only when its CI excludes zero:

| arm | label | metric | delta | CI95 | verdict |
|---|---|---|---|---|---|
| `grounded` | grounded | MRR | +0.007916 | [+0.003371, +0.012607] | **gain** |
| `grounded` | grounded | hit@5 | +0.000000 | [−0.001985, +0.002481] | indistinguishable |
| `grounded` | reconsumed | MRR | −0.002514 | [−0.005021, +0.000046] | indistinguishable |
| `grounded` | reconsumed | hit@5 | −0.001375 | [−0.003024, +0.000000] | indistinguishable |
| `grounded` | usefulness | MRR | −0.001987 | [−0.004888, +0.000986] | indistinguishable |
| `grounded` | usefulness | hit@5 | −0.001183 | [−0.004141, +0.001775] | indistinguishable |

`grounded` shows a significant gain under the primary label and **no
significant regression under either independent label**.

### Why `grounded_negative` was rejected

It posts the *largest* control-label gains (reconsumed MRR +0.0111, usefulness
MRR +0.0297, both CI-significant) yet is rejected, because a check that goes
through no label at all disqualifies it first:

| arm | scopes | mean graph | max graph | scopes with graph > 0.5 |
|---|---|---|---|---|
| `proportional` | 40 | 0.0509 | 0.2010 | 0 |
| `grounded` | 28 | 0.0195 | 0.1041 | 0 |
| `grounded_negative` | 40 | 0.0677 | **0.7500** | **3** — `project:ae`, `project:mm`, `project:online` |

Blaming 88% of deliveries crushes bm25 and vector into their floors until
normalisation hands the residue to graph: `project:online` lands at
0.100/0.150/**0.750**. Both control labels are documented in `replay.py` as
hub-node biased (`reconsumed` "nearly vacuous", `usefulness` "rank-circular"),
and graph-heavy weights surface hub nodes — so those gains are an artifact of
the labels, not quality. `grounded_negative` also shows **no** significant gain
under the primary label. Ungrounded results are therefore **neutral**.

---

## 6. End-to-end harness on the frozen goldset

```
python3 -m living_memory.retrieval_harness run \
  --snapshot ~/.cache/living-memory-harness/phase1/frozen-chunked.sqlite3 \
  --goldset artifacts/harness/goldset.jsonl --agreement-sample 20 \
  --cutoff 2026-06-10T00:00:00Z --report /tmp/gate/run-incumbent.json
```

| metric | phase1 | this branch | delta |
|---|---|---|---|
| overall hit@5 | 0.576923 | **0.576923** | 0.000000 |
| overall MRR | 0.353929 | **0.353929** | 0.000000 |
| overall hit@1 | 0.188034 | 0.188034 | 0.000000 |
| tail hit@5 (26 items) | 0.384615 | 0.384615 | 0.000000 |
| live agreement | 1.0 | 1.0 | — |

Exact, and **exact by construction**: the change is write-path only, so a
retrieval-only harness over a frozen snapshot must be unchanged. This is the
"not worse" gate, not evidence of improvement — as the goal states, the change
cleans signal *accumulation going forward* and already-accrued weights stay put.

### Negative result worth recording: weight transfer

To get evidence that is *not* trivially unchanged, each arm's replayed final
weights were written into a copy of the frozen snapshot and the same 234-query
goldset re-run through the full pipeline:

| weights in the snapshot | hit@1 | hit@5 | MRR |
|---|---|---|---|
| live (unchanged, = phase1) | 0.188 | 0.577 | 0.354 |
| replayed `proportional` | 0.201 | 0.581 | **0.359** |
| replayed `grounded` | 0.171 | 0.581 | **0.336** |

**Equal hit@5, but `grounded` is 0.023 MRR *below* `proportional` here.** This
does not reverse the decision, and the reason matters:

- The replay trajectory starts from *config defaults* and re-learns from
  scratch. With 3,842 updates instead of 42,348 the `grounded` arm simply has
  not travelled as far from the bm25-heavy defaults (`global` 0.412/0.538/0.050
  vs `proportional` 0.137/0.813/0.050); it also leaves 12 scopes unmaterialised.
  This measures **convergence speed from a cold start**, not signal honesty.
- Production never cold-starts. Live `global` is already 0.151/0.799/0.050 —
  essentially the `proportional` arm's endpoint, because that rule ran for
  three months. Grounding changes only *future* updates from that converged
  point.
- The goldset's own `relevant_node_ids` are grounded results of events recorded
  under the live vector-dominant ranking, so it inherits candidate-selection
  bias toward what vector-dominant retrieval finds.

**What to watch after merge:** grounded credit delivers ~9% of the previous
update volume, so per-scope weights will adapt more slowly. If that proves too
slow, the lever is the per-scope `learning_rate` (deliberately not touched
here — this goal is about the label, not the scoring), not a return to vacuous
credit. Nothing in the learning rate, floors, or `feedback_weighted_score` was
changed.

---

## 7. Write-path cost

`python3 scripts/remember_latency_bench.py --snapshot ~/.cache/living-memory-harness/snapshot.sqlite3 --report artifacts/grounding/remember-latency.json`

One identical seeded workload, run twice, changing only the policy. Node
contents are sampled from the real snapshot (p50 965 B, p90 3,708 B, max
64 KB) so tokenization sees production-sized documents; 10 results delivered
per recall.

| backend | policy | n | p50 | p90 | mean | grounded/event | events with ≥1 grounded |
|---|---|---|---|---|---|---|---|
| real model | `all` | 150 | 11.883 ms | 38.579 ms | 64.53 ms | — | — |
| real model | `grounded` | 150 | **7.686 ms** | 30.536 ms | 58.27 ms | 1.94 / 10 | 133/150 |
| hash | `all` | 400 | 12.483 ms | 33.772 ms | 88.46 ms | — | — |
| hash | `grounded` | 400 | **8.108 ms** | 28.339 ms | 78.86 ms | 2.64 / 10 | 384/400 |

**p50 delta −4.20 ms (real model) / −4.38 ms (hash) against a +10 ms budget.**
Grounding is a net *saving*: tokenizing one trace plus ten nodes costs less
than the nine `apply_retrieval_feedback` calls it avoids, each of which is a
node UPDATE plus a `retrieval_weights` UPDATE. The benchmark counts what it
grades (1.94–2.64 grounded per event, 89–96% of events with at least one), so
the saving is not the artifact of a gate that grounds nothing.

---

## 8. Tests

`npm run check` → **867 passed, 83 skipped in 60.25s** (before this change:
839 passed, 83 skipped). All 83 skips are pre-existing and environmental
(82 in `test_replacement_holdout_packet.py` needing `LM_AP_SNAPSHOT`/`LM_AP_STAGING`
restored copies; 1 in `test_retrieval_weight_recalibration.py` guarded on the
report having kept the incumbent). Named acceptance targets pass:
`test_transport_feedback_closure_e2e`, `test_retrieval_weight_recalibration`.

**New (28 tests, 2 files).** `tests/test_grounding.py` pins the IDF formula,
containment edge cases, directionality, that rare tokens outweigh boilerplate,
that the text entry point delegates to the token entry point, and — by source
inspection — that neither caller re-derives the arithmetic.
`tests/test_grounded_credit_assignment.py` is the falsifiable regression suite
described in §2, plus exhaustive-linkage, the `grounded_negative` arm, the
policy surface, and the `memory_teach` no-reinforcement path.

**Three existing tests updated**, each because it asserted *reinforcement* as an
incidental proxy for *linkage*:

- `test_delivery_diet_e2e.py::test_closure_links_recall_to_remember_despite_stub_delivery`
  and `test_recall_gating.py::test_feedback_link_restores_full_delivery` both
  use closure content that shares **zero tokens** with the seeded results (an
  explicit control, asserted in-file). Both already assert the real closure at
  the DB level (`event.feedback_applied`, `feedback_trace_id`,
  `fingerprint.linked_count`). The line
  `implicit_feedback["feedback_applied"] is True` was replaced by a *stronger*
  deliberate pair: linkage lands (`linked_node_ids`) **and** reinforcement is
  correctly withheld (`feedback_applied is False`) — which now fails if credit
  is ever re-broadened.
- `test_feedback_delivery_concentration.py` drives the entrenchment loop it
  measures through `memory_remember`, with follow-up notes that shared no
  vocabulary with the corpus — under grounding the loop would close on nothing
  and the simulation would test an empty system. The note now echoes exactly
  the four corpus tokens that appear in **no** query (`caused`,
  `misconfiguration`, `mitigation`, `documented`), so it grounds whichever
  paraphrase was delivered at an identical 0.2961 containment, keeping arms
  symmetric and the corpus unpolluted. **All original calibrated assertion
  margins were kept unchanged and still pass** on both seeds (7, 17), including
  `legacy.max_corpus_usefulness == 1.0`, `top1_share >= 0.6`,
  `capture_rate >= 0.15`.

---

## 9. Constraints honoured

- **Schema unchanged.** No migration, no new column or table;
  `recall_events.results` already held everything.
- **Live DB never mutated.** Every experiment ran on backup-API snapshots
  (`~/.cache/living-memory-harness/snapshot.sqlite3` and the phase-1 frozen
  copy; weight-transfer copies made with `sqlite3.Connection.backup`, never a
  bare file copy — the source carries a WAL).
- **Scoring untouched:** `feedback_weighted_score`, `FEEDBACK_MULTIPLIER_CAP`,
  per-scope floors, `retrieval.py`.
- **No growth promised or claimed.** The e2e gate is "not worse" and is met
  exactly (delta 0.000000).

---

## 10. Handoff: "query anchors in the graph"

**From when do grounded consumptions accumulate.** From this commit forward,
on every `memory_remember` that consumes a pending recall event. The grounded
subset is exposed in-process as
`ImplicitRecallFeedback.grounded_node_ids` (deliberately **not** added to the
MCP response, so the tool surface is unchanged). Nothing is persisted that
distinguishes a grounded consumption from any other — the `related` edge
metadata still records only `basis`, `recall_event_id`, `recall_query`, `rank`.
**The anchors goal will need to persist the grounded flag**, e.g. as an extra
key in that edge metadata (no schema change needed — the column is JSON).
Expected rate: ~12% of results of consumed events, and consumed events are
16.6% of all recalls.

**Retrospective labelling of history is viable and worth doing.** The history
is complete (`recall_events.results` + `feedback_trace_id`), so the same
grounding can grade the past. Measured on the snapshot with the shipped
per-event view at 0.25:

| Quantity | Value |
|---|---|
| Recall events total / consumed | 54,777 / 9,069 (16.6%) |
| Consumed events with ≥1 grounded result | 3,681 (40.6% of consumed) |
| **Distinct (query → node) anchor pairs** | **8,334** |
| Distinct nodes receiving ≥1 grounded consumption | 2,283 |
| Coverage of the active corpus (12,862 nodes) | **17.8%** |
| Distinct anchoring queries | 3,441 |
| Mean grounded results per consumed event | 0.959 |
| Span | 2026-05-15 → 2026-08-17 |
| Most-anchored node | 120 grounded consumptions |

**Quality assessment of a retro pass.** Good enough to build edges from, with
three caveats to design around:

1. *Precision is high, recall is not.* 0.978 precision against the corpus-IDF
   reference, but 0.768 recall — retro anchors will be sparse and
   conservative. That is the right failure direction for graph edges (a wrong
   anchor is permanent noise; a missing one is filled by the next
   consumption), but do not read absence of an anchor as evidence against
   relevance.
2. *Textual grounding misses conceptual use.* 59.4% of consumed events ground
   nothing — results used without shared identifiers are invisible. Anchors
   will over-represent identifier-heavy domains (paths, error strings, ticket
   ids) and under-represent discussion-shaped traces.
3. *Hub skew.* One node carries 120 grounded consumptions; anchor weight must
   be normalised or the graph will simply re-encode the existing hub
   concentration. `feedback.USEFULNESS_GAIN_ACCESS_DAMPING` exists for the same
   reason on the usefulness channel and is worth mirroring.

Retro labelling would use the *offline* whole-corpus view, which marks 11,073
results (15.4%) rather than the live 8,698 (12.1%); the two agree on 96.2% of
pairs. Pick one deliberately and record which — mixing them would make edge
provenance ambiguous. **Not implemented here**, as instructed.

---

## 11. Reproduction

```bash
# 1. calibration (fast, ~1 min)
python3 scripts/grounding_calibration.py \
  --db ~/.cache/living-memory-harness/snapshot.sqlite3 \
  --report artifacts/grounding/calibration.json

# 2. credit-rule A/B (~21 s)
python3 scripts/credit_rule_ab.py \
  --db ~/.cache/living-memory-harness/snapshot.sqlite3 \
  --cutoff 2026-06-10T00:00:00Z \
  --report artifacts/grounding/credit-ab.json

# 3. write-path latency (~6 min; needs .cache/python-deps on PYTHONPATH)
PYTHONPATH=src:.cache/python-deps python3 scripts/remember_latency_bench.py \
  --snapshot ~/.cache/living-memory-harness/snapshot.sqlite3 \
  --report artifacts/grounding/remember-latency.json

# 4. end-to-end harness (real model; LIVING_MEMORY_EMBEDDING_BACKEND must be unset)
env -u LIVING_MEMORY_EMBEDDING_BACKEND PYTHONPATH=src:.cache/python-deps \
  python3 -m living_memory.retrieval_harness run \
  --snapshot ~/.cache/living-memory-harness/phase1/frozen-chunked.sqlite3 \
  --goldset artifacts/harness/goldset.jsonl --agreement-sample 20 \
  --cutoff 2026-06-10T00:00:00Z --report /tmp/gate/run-incumbent.json

# 5. suite
npm run check

# 6. falsify the regression test: flip DEFAULT_RECALL_CREDIT_POLICY to "all"
#    in src/living_memory/feedback.py, then
bash scripts/test.sh tests/test_grounded_credit_assignment.py   # -> 3 failed
```

Committed artifacts: `artifacts/grounding/calibration.{json,md}`,
`artifacts/grounding/credit-ab.{json,md}`,
`artifacts/grounding/remember-latency.json`,
`artifacts/grounding/e2e-gate.json` (all three harness runs plus the gate
verdict; goldset sha256 `af40cb0da26fa78c…`, identical to the one pinned in
`artifacts/harness/phase1-gate.json`), and
`artifacts/grounding/e2e-live-weights.md`.
