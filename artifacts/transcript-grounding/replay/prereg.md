# Phase-4 replay A/B — pre-registration

Goal `transcript-grounding`, phase 4. This document fixes the decision rule
**before any grounding number exists**. It was written in parallel with the
phase-1 join measurement, deliberately, so that no threshold below can have
been fitted to a number its author had seen.

**Precedence anchor.** Parent branch `transcript-grounding` HEAD at writing
time: `f8ae65ad83bfb217c3a2d3d86bd23bb50efb1b35`
(`2026-08-29T14:30:44+07:00`, "Write the turnkey alt-field operator handoff
pack"). Everything this document pins is decided as of that commit. At that
commit `artifacts/transcript-grounding/` contained `join/` and `ledger/` only:
no `grade/` results, no replay numbers, and this node read neither `join/` nor
`grade/`.

**Immutability.** The committed bytes of this file are the contract. Its
sha256 is recorded in `prereg.sha256` beside it. Phase 4 must embed that
sha256 in its verdict artifact and assert equality against the committed file.
Editing this file after its commit does not amend the pre-registration — it
voids it (§8).

**What this document does not do.** It designs no harness and runs nothing.
The phase-4 node owns `scripts/transcript_grounding_replay.py`; everything
here is expressible over quantities `src/living_memory/replay.py` and
`scripts/credit_rule_ab.py` already compute, plus the phase-3 ledger columns.

---

## 1. The question

Does reinforcement that additionally consumes transcript-grounding verdicts
retain confirmed-useful results better than reinforcement without them, on
recorded history, without making the delivered head noisier?

A `positive` answer opens the phase-5 valve for the operator to consider. Any
other answer keeps it off.

---

## 2. Universe, arms, and what differs between them

### 2.1 Field and method version

One run covers exactly one **field** `F ∈ {'local', 'alt'}` — the
`transcript_grounding_verdicts.field` value — and exactly one
**`method_version` `V`**. Fields are never pooled (the one-host blind-spot
lesson; `src/living_memory/transcript_ledger.py:51`). The decision field is
`local`; see §7.4 for how an `alt` replication is treated.

`V` is the single `method_version` the phase-2 grade report shipped. If the
ledger holds rows under more than one `method_version` for field `F`, the
operator must name `V` in the verdict artifact **before any metric is
computed**; a run that selects `V` after seeing any metric is invalid (§8).

### 2.2 Universe `U`

`U` = every usable recorded recall event (`replay.load_replay_events`,
`replay.py:173`) whose id appears in the phase-1 join match table for field
`F`, and whose joined transcript session key resolves to an identity group in
the post-session corpus (§4).

Events with no join carry no bucket. They are **excluded from both arms** —
from reinforcement and from evaluation alike. This is deliberate: an unjoined
event may still belong to an eval or holdout session whose join merely failed,
so admitting it would open exactly the leakage channel the group split closes.
The exclusion is symmetric across arms and therefore cannot bias the contrast.
Its cost: absolute metric levels in this run are **not** comparable to
`artifacts/replay/` or `artifacts/grounding/credit-ab.*`; only the A-vs-B
contrast inside this run is.

Within `U`:

* a **loud** event has `feedback_trace_id` non-NULL — a `memory_remember`
  consumed it (`ReplayEvent.labeled`, `replay.py:134`);
* a **silent** event has `feedback_trace_id` NULL — the ~80% share this whole
  goal exists to reach.

### 2.3 Arm A — reinforcement WITHOUT transcript-grounding verdicts

The incumbent live rule, replayed over train-bucket events only:

* Only **loud** train-bucket events produce updates, ordered by
  `replay.reinforcement_order` (`replay.py:847`).
* Per result at 0-based rank `r`: signal `max(0.2, 1/(r+1))`
  (`replay.py:840`), credit rule `replay.CREDIT_RULES['grounded']`
  (`replay.py:730`) — proportional credit for results the **consuming trace**
  grounded, neutral otherwise. This is the live default
  (`feedback.DEFAULT_RECALL_CREDIT_POLICY = 'grounded'`,
  `src/living_memory/feedback.py:49`).
* `result.grounded` is set by `replay.apply_grounding` (`replay.py:334`) at
  `grounding.DEFAULT_MIN_CONTAINMENT`, i.e. against the consuming trace — the
  ledger is not read at all in this arm.
* Weight arithmetic: `replay.WeightTrajectory` (`replay.py:771`), updates
  written to `event.scope`, floors via
  `MemoryStore.apply_retrieval_weight_floors`, `evidence='snapshot'`,
  `lr_policy='snapshot'` — the settings `scripts/credit_rule_ab.py:141`
  already uses.

### 2.4 Arm B — reinforcement WITH transcript-grounding verdicts

Arm A, plus one strictly additive class of updates:

* For each **silent** train-bucket event `e` and each delivered result
  `(e.id, node_id)` that has a ledger row with `field = F`,
  `method_version = V` and admitted by the strictness candidate `S` (§5.1),
  apply one update: same signal `max(0.2, 1/(r+1))` at the delivery's recorded
  0-based rank `r`, same `replay.proportional_credit` (`replay.py:719`), same
  `event.scope`.
* Silent deliveries with no ledger row, or with a row `S` does not admit, are
  **neutral** — no reinforcement, no blame. This mirrors the live `grounded`
  policy and keeps arm B a superset of arm A's updates.
* Loud events are handled identically in both arms; the ledger is never
  consulted for them. The treatment is therefore exactly "the silent share
  earns credit", isolated.
* Ordering: the extra updates enter the same stream keyed by the ledger's
  `delivered_at`; loud updates keep their `feedback_applied_at or created_at`
  key. Within one key, loud updates precede silent ones, then ascending
  `node_id`, so the stream is a total order and the run is byte-reproducible.

Both arms start from `MemoryConfig().retrieval_weights` and are trained on the
**train buckets only** (§4). Neither arm ever reinforces from an eval or
holdout event — including those events' own loud consumption. Eval and holdout
events are ranked under weights each arm froze before it saw them.

### 2.5 What is held identical

Same `U`, same event order, same labels, same candidate sets, same ranking
path (`MemoryRecallService.rank_candidates` via `replay.rerank_event`), same
`supersedes` setting, same seeds. The arms differ in **one** thing: the
presence of the silent-event updates of §2.4. Any other difference invalidates
the run (§8).

### 2.6 Scope boundary, stated honestly

The replay harness replays the **retrieval-weight** channel. It does not
replay query anchors (`feedback._reinforce_query_anchors`) or node-level
usefulness. Phase 4 therefore decides the weight channel only; a positive
verdict licenses the valve for the reinforcement the harness actually
measured, and phase 5 must not read it as evidence about anchors.

---

## 3. Metrics

Both metrics are evaluated on **loud events in the split under test** —
events an independent `memory_remember` consumed. The label comes from the
consuming trace; the treatment signal comes from the transcript remainder of
*silent* events. Different documents, disjoint event sets: a win here cannot
be definitional.

Label: `replay.LabelConfig(protocol='grounded')` at
`grounding.DEFAULT_MIN_CONTAINMENT` (`replay.py:406`), applied by
`replay.apply_labels`. It is arm-independent, so both arms score the identical
event set and every comparison below is exactly paired.

Let `E_s` = the loud events of split `s` with at least one confirmed-useful
result (`MetricAccumulator.events_with_useful`, `replay.py:963`), and
`n_s = |E_s|`. For event `e`, let `rr_e` be the reciprocal rank of the first
confirmed-useful result in that arm's re-ranked order, `0.0` if none is ranked
— the value already stored as element 1 of `MetricAccumulator.per_event`
(`replay.py:945`, `replay.py:973`).

### 3.1 Primary — retention of confirmed-useful results

```
R(arm, s) = mrr(arm, s) = (1/n_s) * Σ_{e ∈ E_s} rr_e
```

i.e. `MetricAccumulator.mrr_sum / events_with_useful`, the value emitted as
`mrr` (`replay.py:1013`). Retention of a confirmed-useful result is its
reciprocal rank in the delivery order the agent actually receives; `R` is that
retention averaged over evaluated events. Per-event paired samples already
exist, so `credit_rule_ab._paired_bootstrap(..., index=1)`
(`scripts/credit_rule_ab.py:82`) applies unchanged.

```
ΔR = R(B, s) − R(A, s)
```

### 3.2 Guard — head noise

```
h1_e   = 1 if rr_e ≥ 1.0 − 1e-9 else 0          (rr_e = 1 ⟺ first_rank = 1)
N(arm, s) = 1 − hit@1(arm, s) = 1 − (1/n_s) * Σ_{e ∈ E_s} h1_e
```

`hit@1` is `MetricAccumulator.hits[1] / events_with_useful` (`replay.py:1010`);
`h1_e` is derived exactly from the stored `rr_e`, so no new harness quantity
is required.

```
ΔN = N(B, s) − N(A, s)          (equivalently −Δhit@1)
```

`N` is the share of evaluated events whose **top-ranked delivered result is
not confirmed useful** — junk in the one slot the agent reads first. It is the
right guard because it is the failure the primary can hide: mean rank can rise
while useful results shuffle from 3 to 2 in many events and non-useful ones
take the head in others. The valve must not buy retention with head noise.

### 3.3 Reported, never decisive

`hit@5`, `hit@10`, `graph_unique_share`, `graph_zeroed.*`, per-scope
breakdowns, the `reconsumed` and `usefulness` control labels
(`credit_rule_ab.py:70`), update counts, and final weight vectors are all
reported for both arms and both splits. None of them may become decisive
after the fact. Promoting any of them to a decision input is a post-hoc
amendment (§8).

### 3.4 Uncertainty

Paired bootstrap over **events** (the unit of independence), exactly
`credit_rule_ab._paired_bootstrap`: `BOOTSTRAP_RESAMPLES = 2000`,
`BOOTSTRAP_SEED = 20260818`, `random.Random(seed)` + `rng.choices`, percentile
interval over the sorted resample means. Fixed indices into the sorted
2000-element mean array (0-based), pinned so no rounding convention is left
open:

| level | low index | high index |
|---|---|---|
| 95% | `50` | `1949` |
| 97.5% | `25` | `1974` |

The 97.5% level is the Bonferroni adjustment for the two strictness
candidates of §5.1 and is used at the eval stage only. The holdout stage tests
one candidate at 95%.

Point estimates alone never decide anything: every test below is a point
estimate **and** an interval condition.

---

## 4. Split — identity groups, disjoint by construction

For a recall event in `U`, let `session_key` be the transcript session key the
phase-1 join assigned it. Its **`split_key`** is the identity-group root of
that session: the union-find root over session keys and the declared
`id:<uuid>` identifiers, rooted at the lexicographically smallest token —
`living_memory.postsession.corpus._Union` (`corpus.py:186`) as applied by
`corpus.assign_splits` (`corpus.py:563`). In practice this is the `split_key`
field of the session's row in the post-session corpus index; a run may read
that index (read-only — the sealed `artifacts/post-session/` tree is never
written) or recompute the same root over the joined sessions. If a joined
`session_key` has no identity group, the event leaves `U` (§2.2).

```
bucket = int(sha256(split_key).hexdigest(), 16) % 10      # corpus.bucket_for, corpus.py:161
0-5 -> train      6-7 -> eval      8-9 -> holdout          # corpus.split_for_bucket, corpus.py:167
```

Because the bucket is a property of the identity **group**, every duplicate
recording of one conversation lands in the same split. Splits are disjoint by
construction, and the run must assert it: the three sets of `split_key` values
are pairwise disjoint, and no event id appears in two splits. Failure to
assert, or a failed assertion, invalidates the run (§8).

Roles, fixed:

* **train (0-5)** — the only events that produce reinforcement, in both arms.
* **eval (6-7)** — where the decision is made and the strictness candidate is
  selected.
* **holdout (8-9)** — confirmation only, consulted once, after the eval
  decision is committed.

---

## 5. Decision rule

### 5.1 The one degree of freedom, decided on eval

Two pre-registered strictness candidates for which ledger rows earn treatment
credit in arm B:

* **S0 (as shipped)** — `grounded = 1`.
* **S1 (strict)** — `grounded = 1 AND containment ≥ q75`, where `q75` is the
  75th percentile of `containment` over ledger rows with `grounded = 1`,
  `field = F`, `method_version = V`, restricted to deliveries of **train-bucket
  events only**. Nearest-rank definition, pinned: sort those `containment`
  values ascending into a 1-based array of length `m`; `q75` is the element at
  index `ceil(0.75 * m)`. If `m = 0`, S1 is undefined and the run is
  `inconclusive` (§7.3).

`q75` is computed from train buckets alone, so the strictness parameter never
touches eval or holdout data. Arm A is unaffected by the candidate; arm B has
one trajectory per candidate (`B0`, `B1`).

No other parameter is selectable. Thresholds, margins, minimum `n`, metric
definitions, seeds and the split rule are fixed by this document and identical
at both stages.

### 5.2 Preconditions (checked before any metric is read)

| # | Precondition | On failure |
|---|---|---|
| P1 | `n_eval ≥ 200` and, at the holdout stage, `n_holdout ≥ 200` | `inconclusive` |
| P2 | Treatment materially present: arm B's total update count ≥ `1.2 ×` arm A's, for the candidate under test | `inconclusive` |
| P3 | Split disjointness asserted and holding (§4) | invalid (§8) |
| P4 | Exactly one `field` and one `method_version` in every ledger read | invalid (§8) |
| P5 | No degenerate learned weights in **either** arm: `credit_rule_ab._weight_sanity(...)['graph_over_limit'] == 0`, i.e. no scope with `graph > GRAPH_DEGENERACY_LIMIT = 0.5` (`credit_rule_ab.py:79`, `:121`); and `max_graph(B) ≤ max_graph(A) + 0.05` | `negative` |

P5 is checked first among the metric-adjacent checks and goes through no
label: every label available here is hub-node biased, so a cornered weight
vector can post good aggregate numbers for the wrong reason.

### 5.3 The test, applied identically at both stages

For split `s` and candidate `S`, with `Δ` = arm B (under `S`) minus arm A:

```
EFFECT(s)     :  ΔR ≥ +0.010                     AND  ci_low(ΔR) > 0
NON_INFERIOR(s):  ci_high(ΔN) ≤ +0.005
PASS(s)       :  P5 holds  AND  EFFECT(s)  AND  NON_INFERIOR(s)
```

* **Minimum retention effect `+0.010` absolute MRR.** The prior credit-rule
  A/B on this corpus moved holdout MRR by ~`0.002` between rules that were
  genuinely different; `0.010` is five times that floor. A goal claiming a
  ~5× increase in the learning signal's event coverage that cannot clear five
  times the previously observed rule-change effect has not earned a live
  valve.
* **Non-inferiority margin `+0.005` absolute on `N`.** Half the minimum
  retention effect, so any admissible run trades at least 2:1 in favour of
  retention over head noise. The test is one-sided on the upper interval
  bound: `N` may fall freely, and being indistinguishable from arm A passes.
* **Minimum `n = 200`** paired events per decision split. Below it the paired
  bootstrap cannot resolve a `0.010` effect and the honest answer is
  `inconclusive`, not `negative`.

Absolute (not relative) thresholds are used on purpose: `R` and `N` are
bounded ratios, and an absolute margin cannot be inflated by a small
denominator.

---

## 6. Stage protocol

**Stage 1 — eval (buckets 6-7).** Train both arms on train buckets. Evaluate
`PASS(eval)` for `B0` and `B1` at the **97.5%** interval level (§3.4).

* If neither candidate passes → verdict `negative` (or `inconclusive` if a
  precondition failed). **The holdout is not computed at all** — not a count,
  not a metric — so it stays unspent for a later, differently-designed
  attempt.
* If exactly one passes, it is selected. If both pass, select the larger
  `ΔR` point estimate; exact tie → `S0`.

The eval decision — selected candidate, both `ΔR`/`ΔN` point estimates and
intervals, `n_eval`, precondition results, and the sha256 of this file — is
written to the verdict artifact and **committed** before anything else
happens.

**Stage 2 — holdout (buckets 8-9).** Only reachable from a committed eval
PASS. No retraining, no re-selection: the same two trajectories from stage 1,
the selected candidate only, `PASS(holdout)` at the **95%** level.

* `PASS(holdout)` → verdict `positive`.
* Otherwise → verdict `negative`. The holdout is decisive; an eval win it does
  not confirm is not a win.

The verdict artifact records both stages' numbers, so the eval-only and the
confirmed states are distinguishable forever after.

---

## 7. Stop rule

### 7.1 The goal's falsifier, mapped

> Если replay A/B не показывает улучшения удержания подтверждённо-полезных
> результатов при неухудшении шума — клапан не включается, вердикт negative.

Formally: **if `PASS(eval)` fails for both candidates, or `PASS(holdout)`
fails for the selected one, the verdict is `negative`.** Phase 5 then commits
its negative-stop note citing these numbers, the env valve is not shipped in
an enabled state, and the live path is untouched.

### 7.2 Positive

`positive` requires `PASS(eval)` **and** `PASS(holdout)`. It licenses phase 5
to ship the valve **default-off**, with reinforcement matching §2.4 exactly
(same signal, same credit rule, same neutrality for ungrounded and unledgered
deliveries, same `field`/`method_version` discipline, same strictness
candidate). It does not license enabling it: that is the operator's call on
these numbers, and no deploy or restart is part of phase 4 or 5.

### 7.3 Inconclusive

A precondition failure (P1, P2, or `m = 0` in S1 when S1 is the only surviving
candidate) yields `inconclusive`. For the valve, `inconclusive` behaves
exactly like `negative` — it stays off. It is recorded distinctly because it
licenses a future, better-powered attempt, whereas `negative` records a
measured refutation.

### 7.4 Fields

The decision field is `local`. If an `alt`-field ledger exists, the same rule
is executed on it as an independent replication and reported separately:

* local `positive`, no alt run → `positive` (unreplicated; noted as such).
* local `positive`, alt `positive` → `positive` (replicated).
* local `positive`, alt `negative` → overall `inconclusive`; the valve stays
  off pending an operator decision. The one-host blind spot is a lesson this
  goal already paid for.
* local `negative` → `negative`, whatever alt says.

---

## 8. What makes a run INVALID

An invalid run produces no verdict at all. It may not be reported as
`negative`, `inconclusive`, or `positive`; it must be discarded and, if
repeated, repeated under a **new** pre-registration with a new sha256.

1. **Holdout consulted before the eval decision is committed.** Any holdout
   number, count, or metric computed, printed, or stored before the commit
   carrying the eval decision block. Enforcement: the holdout stage refuses to
   run unless the eval decision file is present in `HEAD`, and any holdout
   number appearing in the same commit as the eval decision is itself the
   violation.
2. **Post-hoc threshold amendment.** Any change to this file after its
   committing commit — thresholds, margins, minimum `n`, metric formulas,
   split rule, arm definitions, candidate set, seeds, interval levels or stage
   order. Detected by `prereg.sha256` mismatch against the committed
   `prereg.md`; the verdict artifact must carry the sha256 and assert
   equality.
3. **Metric substitution.** Deciding on any quantity of §3.3, on a different
   `k`, on a different label protocol, or on a subset of scopes.
4. **Broken pairing.** Arm A and arm B not evaluated over the identical event
   set, or per-event samples joined by anything but `event_id`.
5. **Arms differing in more than §2.4.** Different seeds, orders, labels,
   `evidence`/`lr_policy`/`supersedes` settings, candidate sets, or harness
   versions between arms.
6. **Split violation.** A bucket assigned by anything but §4's function, a
   `split_key` appearing in two splits, an event evaluated in a split other
   than its own, or reinforcement drawn from a non-train bucket.
7. **Ledger drift mid-run.** The ledger snapshot's row count and content
   digest for `(field = F, method_version = V)` must be recorded at stage 1
   and re-asserted equal at stage 2. Rows imported between the stages
   invalidate.
8. **Mixed field or method_version** inside one comparison, or a `V` chosen
   after any metric was seen (§2.1). Also: selecting the strictness candidate
   on anything but the eval numbers, or recomputing `q75` on eval/holdout
   data.
9. **Live-store contamination.** Any connection to a live store opened other
   than read-only, any write to the live path, or a deploy/restart performed
   as part of the run.
10. **Unreproducible run.** Numbers that a re-execution at the recorded input
    digests does not reproduce bit-for-bit.

---

## 9. Required shape of the verdict artifact

`artifacts/transcript-grounding/replay/verdict.json` must carry at least:

* `prereg_sha256` and an assertion that it equals the sha256 of the committed
  `prereg.md`;
* `field`, `method_version`, ledger row count and content digest for
  `(F, V)`, recorded at both stages;
* `universe`: event counts in `U` and per split, `n_train` / `n_eval` /
  `n_holdout`, and the split disjointness assertion result;
* per stage and per candidate: `R(A)`, `R(B)`, `ΔR`, its interval and level;
  `N(A)`, `N(B)`, `ΔN`, its interval and level; update counts per arm;
  `_weight_sanity` for both arms;
* the selected candidate and `q75` (with `m`);
* preconditions P1-P5 with their outcomes;
* `verdict ∈ {positive, negative, inconclusive}` and the literal rule
  evaluation that produced it;
* the eval-decision commit sha, for the §8.1 ordering check.

`report.md` renders the same content for a human. Neither file may contain a
threshold that differs from this document.

---

## 10. Known limitations (recorded now, not after the numbers)

* **Candidate-selection bias.** Replay re-orders recorded results; it cannot
  surface candidates the historical ranking excluded. Absolute levels are
  optimistic for both arms; the paired contrast is the claim.
* **Coverage-restricted universe.** Excluding unjoined events (§2.2) buys
  leakage safety at the price of sample size, and shifts the population toward
  sessions whose transcripts survived and joined.
* **Textual grounding on both sides.** Both the label and the treatment signal
  are IDF-containment proxies. Conceptual use without shared identifiers is
  invisible to both. Neither is ground truth for "the agent used this".
* **Trajectory approximations** inherited from `replay.py`: configured family
  defaults as initial weights, snapshot learning rates, snapshot floor
  evidence gates, no explicit `memory_teach` feedback in the corpus.
* **Weight channel only** (§2.6): anchors and node usefulness are not
  replayed and are not decided here.

None of these limitations may be invoked after the fact to reinterpret a
`negative` verdict as anything else.
