# Cold-start selector — pre-registered design, v2

Machine-readable plan: `artifacts/recall-map/relevance/coldstart-prereg-v2.json`,
sealed under `plan_sha256`
`0af8124b743ee2031d93bd71b35b99df65fe9d7b05068fe7baa836a1f5f24de6`.

It supersedes `coldstart-prereg.json`, `plan_sha256`
`d17082c4d3b321a71cdb1399163c352728511ef4dd25c685fbf10de2d40b5df3`, whose own
`amendment_policy` licenses exactly one repair for a shape change: "a new
pre-registration under a new plan_sha256, sealed before the next number is
read." This is that file. v1 is not edited and stays on disk with its recorded
negative result, because a superseded plan is evidence, not a draft.

Nothing in this document is a measurement. Every population figure quoted here
is a **v1** figure, copied from artifacts committed before this node started.
The enlarged population the sibling `coldstart-population-widen` is building did
not exist in this worktree at sealing, and no number from it appears anywhere.

---

## What v1 got right, and the one thing it got wrong

v1 produced an honest negative result, and it is worth being precise about what
it established, because a v2 that re-registers the same wall is worse than no v2
at all.

**The family works.** All four core members of `recorded-delivery-scores` held
their registered signs on the fit split *and* on a component-disjoint
verification split, whole-tail *and* on the cold subpopulation measured
separately. `policy.json#coldstart_revision.findings.the_family_does_discriminate_among_cold_candidates`
records cold lift **1.9185** on train (binomial p 1.70761e-05) and **1.8312** on
eval (binomial p 0.0325543, cold selected precision 0.667) — both above the
pre-registered generalization threshold of 1.5. The candidate-time features
carry real cold-start signal. This plan does not weaken them, does not add a
member, and does not re-sign one.

**The imputation fix does not work.** This is the fact that disciplines
everything below. `policy.json#coldstart_revision.history_arm_sensitivity` ran
the counterfactual in which a cold row's history block contributes `0` instead
of `-3.6135` — mean substitution repaired, in isolation, with the member set,
the signed weights, the fitted transforms and the threshold construction all
identical. The result was **unchanged**: still `feasible_threshold_count: 0`,
cold admission moved by *one row* out of 1233, reaching the cold floor 0.05
still demanded 0.5567 of the split overall against a ceiling of 0.30, and the
threshold at the ceiling still left cold admission at 0.0341.

The reason is recorded in the same block: on a complete ledger
`storage.matured_recall_history` returns `known(matured=0)`, not `unavailable`.
The cold population is overwhelmingly on the *available-but-empty* arm, so a
repair to the *unavailable* arm touches almost none of it.

An honest v2 therefore may not be built on the claim that replacing the
train-mean placeholder repairs cold start. That claim has been measured and
refuted. The placeholder is still replaced here — leaving a known-wrong constant
in a scorer being rewritten would be indefensible, and it is wrong on the live
path — but the file says plainly, at
`feasibility_argument.what_the_neutral_rule_does_not_do`, that it is not what
carries the repair.

**So what was wrong was the shape.** v1 froze "the legacy five keep their
existing positions and arithmetic" and only *appended* the family. That single
decision is the whole failure.

---

## The defect, in arithmetic

Two scales that cannot meet, both published in v1's own findings.

| quantity | value |
| --- | --- |
| history arm, cold row (available, empty) | −3.6135 |
| history arm, warm median | +2.0103 |
| history arm, warm p95 | +5.1953 |
| **cold-to-warm-median handicap** | **5.6238** |
| family arm on cold rows, median | −0.4501 |
| family arm on cold rows, p95 | +0.4607 |
| **family's median-to-p95 range** | **0.9108** |
| **handicap ÷ family range** | **6.17×** |

A cold row must make up 5.62 of history z using an arm whose entire ordinary
dynamic range is 0.91. No threshold can arbitrate between two quantities that
differ by a factor of six: the ordering of the pooled population *is* the
ordering by history, up to a rounding error contributed by the family. Cold rows
sit at the bottom as one solid block, so a descending threshold admits nearly
every warm row before it admits any material number of cold ones — and the two
admitted fractions move as one quantity instead of two. That is the welding, and
it is why `feasible_threshold_count` was 0.

Note *why* the family's range is so narrow, because it is not weakness. The
published cold family_z distribution is p95 **+0.4607** against p99 **+9.46**.
Winsorize-then-standardize divides by a standard deviation that heavy-tailed
retrieval scores set from the tail, not the bulk, so ninety-five percent of the
cold population is compressed into a band under one z wide while the top
percentile is twenty times wider. The family's *rank* information was intact —
that is why its lift was 1.83 — while its *arithmetic magnitude* was a rounding
error. v1 measured real discrimination and then had no shape that could spend
it.

v1 said the same thing in its own words:

> The family is not too weak at the top: its best cold row reaches +10.16, well
> past the +4.4703 the prereg registered as the target. It is too weak in the
> MIDDLE: the median cold row sits at −0.45 and the 95th percentile at +0.46, so
> only 41 of 1233 cold rows (3.3%) reach the schema target and 37 (3.0%) reach
> the trace target.

---

## The shape

    score = N + F

Two arms, one threshold.

**`N`, the node-intrinsic block** — `N = (L + H) / 2`, range `[−0.5, +0.5]`.

- `L` is the frozen `level_is_schema` feature, mapped to a centred rank-uniform
  scale fitted on train.
- `H` is the frozen four-term history z-sum — the *unchanged* arithmetic of
  `_history_features`, including its `round(..., 8)` — mapped to the same kind of
  scale, fitted over rows where the block is readable.

`rank_uniform_centered` is strictly monotone, so within the readable-history
subpopulation the revision's history ordering is **bit-for-bit the frozen
policy's**. Not one pair of rows is re-ranked by history. Only the arm's
arithmetic scale changes.

The two arms are **averaged, not summed**. Summing is exactly how the frozen
policy let a warm row accumulate +5.68 to +5.78: each of four history terms
contributes its own z and the total grows with the term count. A bound that
grows with the feature count is not a bound. Averaging fixes total
node-intrinsic influence at one half-unit however many node-intrinsic features
exist, so the feasibility argument below survives future feature work rather
than holding only for today's feature set.

**`F`, the candidate-time family arm** — the four core members at their
registered signs, each on a centred rank-uniform scale, summed, and the
composite mapped to a **rank-normal** scale clipped to `[−3, +3]`.

`graph_score` is now permanently **excluded**, not conditional. v1 registered its
exclusion in advance, ran the registered admission test, and the test confirmed
the flip on the real label (−0.002927 raw, −0.026056 transformed). That question
is answered; re-opening it would be running a test until it gives a different
answer, in the direction that *adds* a feature.

Winsorization is dropped. Rank statistics are invariant to every monotone
transform, so clipping a member at p1/p99 cannot change any rank it feeds.
Keeping it would be inert ceremony and two more fitted constants per member.

### Why the two arms get different scales

Node-intrinsic evidence is calibrated to a **bounded** scale; candidate-time
evidence to an **unbounded** one.

A cold node cannot acquire delivery history at candidate time and cannot change
its level. Whatever those arms say about it, it cannot answer. An arm a row
cannot answer must be able to **reorder** rows but must not be able to put a row
**out of reach** — that is the arithmetic form of "history informs, history does
not gate", and breaking the delivery → history → selection → delivery loop is
the root goal's stated purpose. The family is the opposite kind of evidence:
what *this* retrieval found about this node for *this* query, which the row owns
and can differ on. It is allowed to reach as far as the evidence warrants.

This is a pre-registered **scale** choice, fixed before any number from the
population was read, derived from the principle above. It is not a coefficient
and it is not fitted. It is stated plainly in the sealed file rather than buried,
because a reviewer is entitled to read it as a weight in disguise and judge the
justification on its merits.

### Nothing is fitted on labels

Every calibration is a rank transform of the fitting split, which reads no
consumption label. The threshold is selected by a label-free rule (below). This
is strictly stronger than v1, which permitted the scalar threshold to be fitted
on labels.

---

## Why the bands are jointly satisfiable

v1's fatal flaw was not a wrong belief — it was an *unchecked* one. Nobody
computed whether its bands were reachable before sealing, so an arithmetically
incapable shape survived until step 6, after the dataset, the extractor and the
fit had all been built. This section is that computation, published before
sealing, so it can be refuted on the page instead of by spending a holdout.

### Step 1 — an exact identity

On a midrank rank-uniform centred scale, if the lowest-valued rows form a tie
block of mass *q*, the block's value is exactly `q/2 − 0.5` and the mean value of
everything above it is exactly `q/2`. **The gap is exactly ½, for every *q*.**

*Proof.* The block occupies positions 1…qn, so its midrank fraction is
`(0 + qn/2)/n = q/2`. Midranks preserve the sum of ranks, so the remaining
(1−q)n rows have mean rank fraction `(qn + (1−q)n/2)/n = (1+q)/2`, whatever their
internal tie structure. Centring subtracts ½ from both; the difference is
`(1+q)/2 − q/2 = ½`. ∎

Every cold row with a readable ledger has M = C = K = 0, hence an identical
composite of −3.6135, hence a single tie block at the bottom of the history arm.
So the identity applies directly, and after averaging into `N` the history arm's
cold-to-warm-mean contribution is exactly **0.25**. The level arm's gap is
likewise exactly ½, contributing at most **0.25**. Worst-case total
node-intrinsic deficit: **0.5** against the warm mean, **1.0** between the
extremes of `N`.

None of that depends on the cold share, the schema share, the population size,
or any number the sibling will publish.

### Step 2 — what a cold row now has to do

Because `F` is rank-normal by construction, the family percentile a cold row
must reach to tie a mean-placed warm row is a fixed, computable number:

| cold row's node-intrinsic deficit | family z needed | share of the family distribution that qualifies |
| --- | --- | --- |
| 0.25 (level-neutral) | +0.25 | **40.1%** |
| 0.50 (adverse level) | +0.50 | **30.9%** |
| 1.00 (absolute worst case) | +1.00 | **15.9%** |

Against v1, where a cold row needed family z **+2.2026** (schema) or **+4.4703**
(trace/concept) and only **3.3%** and **3.0%** of cold rows reached it. That is a
**12.1×** and **10.3×** enlargement of the admissible cold pool.

The family is not touched. Not one member is added, not one sign changed, not one
cold row's family *rank* moved. What moves is the bar — from "+2.2026 on a scale
whose p95 is +0.4607" to "+0.25 to +0.50 on a scale whose p69 is +0.50" — purely
because the two quantities are finally measured in the same units.

### Step 3 — the frontier, swept

A null model: `F` exactly N(0,1) pooled (enforced by the transform, not assumed
of the data), split into cold and warm at a mean separation δ; `H` rank-uniform
with the cold block at `q/2 − 0.5`; `L` the two-point image of a binary at schema
share *p*; `N = (L+H)/2`; score `= N + F`. The model assumes **no** relationship
between any arm and the consumption label — the correct conservative choice,
since an admitted-*fraction* band is about volume, not discrimination.

A normal model would have been worthless applied to v1, whose family composite
was nowhere near normal (p95 +0.4607, p99 +9.46); it would have declared v1
feasible and been wrong. Here the normality is *created by the transform*. That
is the difference between a model of the population and a model of the
arithmetic, and only the second can be trusted before the population is read.

Swept over 60 cells — *q* ∈ {0.10, 0.15, 0.2177, 0.30, 0.40}, δ ∈ {0, 0.25,
0.50}, *p* ∈ {0.10, 0.2643}, and both a neutral and an adverse level
association:

- **Feasible in 60 of 60 cells** at the registered cold floor 0.05.
- Narrowest joint threshold window over the whole grid: **0.3478** score units —
  not a knife edge.
- At v1's measured cold share with δ = 0.25: cold admitted **0.1762** at the
  0.30 overall ceiling.

| | cold admitted fraction at the 0.30 overall ceiling |
| --- | --- |
| v1, as registered | 0.0300 |
| v1, mean-substitution sensitivity arm | 0.0341 |
| **v2, modelled** | **0.1762** |

That is **5.9×** v1's registered arm and **3.5×** headroom over the floor it had
to clear. v1 was short by a factor of 1.47 with zero feasible thresholds.

The model is thirty lines of arithmetic, reads no dataset, and is fully
specified in the sealed file. Anyone may recompute every number in it.

### The corner where this plan fails, named in advance

The grid clears the *derived fit floor* of 0.10 (see below) in **59 of 60**
cells. The exception is *q* = 0.10 **and** δ = 0.50 **and** schema share 0.2643
**and** an adverse level association, all at once: a thin cold population, a
half-sigma family deficit for cold rows, and every cold row non-schema. There the
threshold rule finds nothing and the plan halts at step 4 with
`feasible_threshold_count: 0` — a second negative result, at the cost of **no
holdout read**.

δ is the one quantity the model cannot bound: v1 published the cold family_z
distribution but never the pooled one, so the cold-versus-warm shift is not
recoverable from published numbers. If the true δ exceeds half a standard
deviation this plan fails, and the finding will be that cold candidates genuinely
carry weaker candidate-time evidence — a real result about the population, not
another shape defect. Forcing admission in that case would be fiat, and this plan
declines to build a mechanism that could.

---

## The missing-value rule, and why it is not a cohort branch

One rule, keyed on whether an arm is **readable in the record being scored**,
applied per arm, with the same formula running for every row.

- **Node-intrinsic arms.** If `L` or `H` cannot be read, that arm contributes its
  calibrated scale's neutral value `0.0`, and `N = (L + H)/2` is formed as usual.
  The divisor stays 2.
- **Family arm.** If any admitted member cannot be read as a finite float, the
  row is `family_unscoreable` and is **inadmissible**. Fail closed. This is v1's
  rule, carried unchanged.

The asymmetry is not convenience. History unreadability is a real, live,
structural property of the population this capability exists to admit; a node the
ledger cannot speak about must be scored *neutrally*, not condemned. Family
unreadability is not live-possible at all — `retrieval.py:345-357` declares
`score` as a required float and the four components as `float = 0.0` — and arises
only from a historical extractor that did not persist them (all 686 observed-map
items, 338 of 4061 secondary-anchor items). Imputing a neutral there would let
**extractor identity** supply evidence, which is source identity and is already
banned. A row with no candidate-time evidence has, by definition, no
candidate-time evidence to be admitted on.

### The distinction the ban actually draws

v1 bans any test of `matured == 0`: no per-cohort, per-coldness, per-scope or
per-level branch, no separate threshold, no bonus. That ban is carried in full.

A cohort branch asks **which population** a row belongs to and scores it
differently. A missing-value rule asks whether a **field of this record** can be
read and runs the same formula either way. The code already contains exactly such
a rule, today, at `src/living_memory/recall_map.py:1190`: `_history_features`
tests `history is None`, `available is not True`, non-int counts, and the count
invariants, and returns a substitute when any fails. Nobody has ever called that
a cohort branch. It is merely **mis-calibrated** — the substitute is a foreign
cohort's raw means, costing a history-absent row about 2.0 z against the realized
warm median of +2.0103. This plan replaces the substituted *value* and keeps the
trigger condition character-for-character.

And the rule fires on **readability, never on emptiness**. A row with a readable
ledger reporting M = C = K = 0 — which is what a cold row on a complete ledger
*is* — does **not** receive the neutral. It receives the real midrank of the
−3.6135 block, at the bottom of the arm. Nothing about it is forgiven. The rule
strictly separates "no information" from "information saying zero", so it cannot
be a disguised cold-start bonus: the cold population overwhelmingly takes the
not-forgiven path. That is also precisely why this plan does not rest on it.

### Answering v1's own objection

v1 argued that omitting a z-term from an additive standardized sum is
arithmetically identical to substituting that member's mean, so "a scorer cannot
both be additive and treat an unreadable member as neutral without imputing the
mean."

That is correct, and it does not apply. v1's mistake was concluding that mean
substitution is therefore always the *same act*. What matters is **whose** mean,
measured on **which** population. The banned substitution imputes
`RELEVANCE_FEATURE_MEANS[1..4]` — a foreign warm cohort's raw means — onto a
cold-inclusive population, where its effect is not neutral at all: it places the
row roughly 2.0 z below the realized warm median of the population actually being
scored. The registered substitution imputes the arm's **own median on the fitting
split**, by construction, because the scale is centred there.

Both are imputation. No additive scorer can remove it. The difference between
them is the entire parent goal: *replacing* the train-mean placeholder is not the
same as removing imputation. This plan makes the imputed value honest instead of
pretending it is absent.

---

## What is not moving

| number | value | why |
| --- | --- | --- |
| overall admitted band | `[0.10, 0.30]` | v1's, exactly |
| cold admitted band | `[0.05, 0.25]` | v1's, exactly |
| generalization threshold | `holdout_cold_lift ≥ 1.5` | v1's, exactly |
| minimum cold candidates per split | `400` | v1's, exactly |
| minimum cold admitted on holdout | `40` | v1's, exactly |
| binomial alpha | `0.05` | v1's, exactly |
| holdout reads | `1` | v1's, exactly |

The bands are the ones v1's shape proved it could not satisfy jointly. Relaxing
either would convert a shape failure into a bookkeeping victory and make this
file worthless. The claim here is falsifiable *precisely because* the bands did
not move: v1 got overall 0.7542 at the cold floor and cold 0.0300 at the overall
ceiling, and v2 must get both inside the band at one threshold or record a second
negative result.

The split minimum stays at 400 though v1 missed it by 27 on the holdout.
Lowering a bar after missing it is the p-hacking these files exist to prevent;
the bar stays and the *population* moves — that is the sibling's job. A **larger**
minimum was considered: 800 per split would make the 0.05 cold floor and the
40-admitted minimum exactly consistent, which is a genuine power argument. It is
rejected because the sibling's postcondition targets 400, so registering 800
would author a halt at step 1 by construction — this node breaking a sibling
rather than raising a bar.

### The derived fit floor, 0.10

v1 registered a cold floor of 0.05, a split minimum of 400 and a holdout minimum
of 40 admitted cold rows, and never checked them against each other.
`0.05 × 400 = 20`, half of 40. The three were not jointly satisfiable at the
registered split minimum and nothing in v1 said so.

The fix needs no published minimum to change. The **threshold-selection rule**
targets a cold admitted fraction of `40/400 = 0.10` on train, so a holdout
sitting exactly at the split minimum can still deliver the 40 rows the binomial
guard needs. The cold *verdict* band stays `[0.05, 0.25]` — a split admitting
0.06 of its cold rows passes. The fit merely aims higher so the registered floors
are mutually reachable.

This is a small thing next to the shape failure, but it is the same species of
error: registering constraints without computing whether they can hold at once.

### The threshold rule

Sweep distinct train scores descending; take the **highest** threshold at which
overall admitted ∈ [0.10, 0.30] **and** cold admitted ∈ [0.10, 0.25] on train.
The highest qualifying threshold is the smallest admission consistent with the
floors — the conservative end, the one that least risks turning the map into
noise. If no such threshold exists: **halt**, `feasible_threshold_count: 0`,
recorded negative result, holdout never opened.

Both quantities are ratios of counts of rows above a score. Neither reads a
consumption outcome, and coldness is a property of the ledger, not of the label.

---

## The anti-fiat guards

v1 relied on the shape ban to stop a bias correction passing as learning: no test
of `matured == 0`, therefore no fiat. That ban is retained, but it is not what
does the work here, because this plan *deliberately* improves a cold row's
arithmetic position. The honest question a reviewer will ask is: did the selector
**learn which cold nodes are worth delivering**, or did it merely stop penalising
all of them equally? A shape ban cannot answer that. Only a control can.

**(a) Generalization.** `holdout_cold_lift ≥ 1.5`. v1's number, carried exactly.
v1's eval measurement of 1.8312 is neither a reason to raise it nor to lower it;
a threshold that moves after a related number is seen is not a threshold.

**(b) Family ablation.** The *same registered shape* with the family arm replaced
by its neutral value: `S = N + 0.0`. Same calibration tables, same rows, same
fail-closed rules — `family_unscoreable` rows stay inadmissible in both, so the
two share a denominator and differ only in whether `F` contributes its real value
or zero. Its threshold is fitted **on train** to admit at least as many cold rows
as the registered shape, with ties broken toward the *higher* threshold — fewer
admitted rows, higher ablation precision, and therefore *harder* for the
registered shape to beat. The conservative direction is chosen on purpose. Both
thresholds enter the sealed parameter digest before the holdout is opened.

The registered shape must **strictly exceed** the ablation on holdout cold lift
*and* on holdout cold AUC. Lift is measured at a threshold, so an ablation
landing at a very different volume can post a flattering or punishing lift for
reasons having nothing to do with the family; AUC is threshold-free and
volume-free and cannot be distorted that way. Requiring both strictly is harder
than requiring lift alone — added, not substituted.

This ablation is not a strawman. On the cold subpopulation `S = N` is nearly
constant, because every cold row with a readable ledger shares one history atom,
so its only remaining discrimination among cold rows is the level arm. The
ablation therefore measures exactly whether the family beats **"prefer schema"** —
a real and tempting degenerate policy. More to the point, it *is* the
recalibration fix with no family in it. If recalibration alone carried the
result, the ablation would match the registered shape and this guard would fail.

**(c) Warm non-degradation.** On `coldstart_eval` warm rows (`matured ≥ 1`),
score with the frozen scorer exactly as it stands today (threshold
3.8708378402511) and let *K* be the count it admits; then rank the same rows by
the revised score, take the top *K*, and require the revised precision to be at
least the frozen precision. Matching on *K* compares the two **orderings** of the
same rows; comparing at each policy's own threshold would measure the thresholds
instead. If *K* < 20 the comparison is recorded as indeterminate and the binding
condition becomes revised warm lift ≥ 1.0 — registered in advance, with a trigger
itself measured before the holdout opens, so an undefined statistic cannot later
be read as a pass.

The root goal requires that the map does not start returning noise. A revision
that admits cold nodes by degrading the ordering of warm ones has moved the
problem, not solved it.

**(d) Admitted-fraction bands.** Both bands on train, eval and holdout,
independently. Checked on train and eval at step 6: if either fails there the
holdout is **never opened**. This is the step v1 failed, and failing it here again
costs no holdout.

### One pass, and no second read

`coldstart_holdout` is materialized **exactly once**. Both scores, both admitted
sets, both fractions, both lifts, both AUCs, the binomial p-value, the
by-level cold admitted counts and the unreadable counts all come from that one
materialization. Both shapes read the same sealed tables and differ by one term,
so scoring both is one extra addition per row.

**An ablation that needs a second read is a second evaluation and is
forbidden** — as is a second read to recompute a forgotten statistic, to try
another threshold, or to check a hypothesis the first pass suggested. If the fit
discovers afterwards that it needed a quantity it did not compute, that quantity
is unavailable and the verdict is decided without it. The guard designed to
protect the verdict must not be what destroys it.

---

## What counts as PASS

All of:

1. every split ≥ 400 cold candidates, pairwise component-disjoint, assignment
   blind to every label, feature and coldness statistic;
2. every core family member holds its registered sign on train, eval and the
   secondary anchor — whole-tail **and** cold, measured separately;
3. `feasible_threshold_count ≥ 1` on train, with both bands holding on train and
   eval;
4. warm non-degradation on eval;
5. `holdout_cold_lift ≥ 1.5`;
6. strictly beats the family ablation on both holdout cold lift and holdout cold
   AUC;
7. ≥ 40 admitted cold holdout rows, and > 0;
8. exact binomial p ≤ 0.05;
9. both bands on the holdout;
10. the withheld-holdout reproduction returning a byte-identical parameter
    digest.

Anything else is a recorded FAIL or NEGATIVE RESULT, and neither may be repaired
by widening the family, adding a branch, re-drawing a split, moving a band, or
re-reading the holdout.

A second negative result under a shape with a *published* feasibility argument
would be worth as much as the first, because it would refute that argument on the
page and say something new about the population. What is not legitimate is a
third pre-registration that finally moves a band. The sealed
`amendment_policy` adds a clause v1 did not have: the bands, the 1.5, the 400 and
the 40 may not be moved **downward** by any future plan. Those four numbers have
now survived two plans and one recorded negative result. A plan that needs them
lower is not a better plan; it is the p-hacking these files exist to prevent,
arriving by instalments.

---

## Direction checks, and one estimator change

The direction constraint is carried from v1's amended text in full: legacy
features keep the original constraint; every family member must show its
**registered sign** in `coldstart_train`, `coldstart_eval` and `secondary_anchor`;
and every member must show it in the **cold subpopulation** of each, measured
separately, with ≥ 200 cold rows and ≥ 40 consumed cold rows or the check is
INDETERMINATE — never passed.

One thing changes: the statistic is taken on the rank-uniform values *this*
scorer consumes, not on v1's winsorized-and-standardized values. The mean
difference of rank-uniform values between two groups is an affine function of the
Mann-Whitney U, so the check becomes exactly "does this member rank consumed rows
above non-consumed rows" — invariant to every monotone re-expression. v1's
raw-scale mean difference was vulnerable to the same heavy-tail distortion that
produced the p95 0.4607 / p99 9.46 spike and killed its scorer; a handful of
extreme rows could set the sign. The registered **signs** are unchanged; only the
estimator is made robust, in the direction that is harder to pass by accident.

---

## Disclosed

- **The holdout is not virgin with respect to direction.**
  `feature-analysis.json#associations` publishes this family's aggregate
  directions, and `policy.json#coldstart_revision` now publishes v1's cold lifts
  on the v1 splits. This author has read all of it. Those are cohort-level
  statistics, never item rows, so they fix no row's membership; the directions
  they inform are pre-registered *here*, before any fit on the enlarged
  population. And v1's holdout was **never read** — `holdout_evaluations: 0`,
  `holdout_cold_admitted: null` — so no cold-start holdout number has ever
  existed for anyone to have seen.
- **Balancing on component size is permitted and leaks a little.** The
  assignment must be computable from seed, opaque component digest and item count
  alone; no label, feature or coldness statistic may enter it. But size correlates
  with coldness, and v1's two candidate rules differ by 334 cold rows on the
  holdout. So the fit must itself recompute the v1 seed-uniform draw over the
  published component list and publish its per-split cold counts beside the
  shipped assignment's, making the difference auditable. It needs no extra read to
  do this.
- **Arm independence is assumed by the model.** A node delivered many times may
  also rank higher for the queries that reach it. Positive correlation narrows the
  joint window; the mitigation is that the modelled window is 0.3478 wide at its
  narrowest, and that step 4 measures `feasible_threshold_count` empirically
  before the holdout is touched. The model justifies sealing this shape; it does
  not replace the measurement.
- **The calibration tables are a new wireable surface.** The frozen policy ships
  ten floats; this one ships up to 1024 breakpoints per calibrated quantity, with
  a published maximum interpolation error of 0.001953125 on `u`. Atoms — values
  whose multiplicity is ≥ 0.5% of the fitting rows, which the available-but-empty
  history composite is by construction — are published as exact pairs and matched
  by equality before any interpolation, so the tie-block identity holds exactly
  and not approximately. The published table **is** the definition, so fit, eval,
  holdout and the live scorer can be checked against one another for bit-identical
  output.
- **The holdout is historical, not prospective.** It is a partition of
  already-persisted state — disjoint by component, never refit on, and the
  strongest available evidence for this subtree, but not a field trial. The
  population `policy.json#holdout_boundary` reserves is forbidden input and output
  for every node here, and its literal path is deliberately not spelled in either
  artifact: `scripts/recall_map_relevance_eval.py:64` rejects those tokens as raw
  substrings before resolving any path, so an artifact naming one would be refused
  by every command in the evaluator that reads it.
