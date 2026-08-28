# Cold-start selector pre-registration v3 — the argument

Companion to `artifacts/recall-map/relevance/coldstart-prereg-v3.json`
(`plan_sha256` `be85f81c0b581741f498632952f012de56a8aedaa475fcdb73006da542a6213b`).
This document is prose only. Every number in it is copied from
`dataset-manifest.json#coldstart` or `policy.json#coldstart_revision`, both committed
before this plan was written. Nothing here was computed from a database.

---

## 1. What halted, and what this plan is for

`coldstart-prereg-v2.json` (`plan_sha256` `0af8124b743ee2031d93bd71b35b99df65fe9d7b05068fe7baa836a1f5f24de6`) halted at
`decision_procedure` step 3, `calibrate_and_check_registered_signs_on_train`, because
a registered sign failed. Two of its four registered directions inverted on the
enlarged `coldstart_train`:

- `result_score`, registered **positive**, read **-0.051024**
  on the cold subpopulation — clause (c).
- `bm25_score`, registered **negative**, read **+0.004903**
  on the whole tail — clause (b).

v2's own step 9 and `amendment_policy` forbade the node that saw those numbers from
repairing them in place. They asked for exactly one of two things: a re-registered family
whose directions are measured on the enlarged population *before* they are registered, or
an explicit finding that no direction-stable signal exists. This plan takes the first and
writes the second into its registration rule as a reachable outcome, so that a successor
who re-measures on a different population can arrive at the finding mechanically rather
than by argument.

## 2. The diagnosis: the signs did not move because the family lost its signal

They moved because the enlargement destroyed independence. Under the sealed identity rule
— connected components over the source-qualified cache key and the session ids — months of
pre-August work share one cache key and collapse into one component. `effective_components`
fell train 12.05 → 2.78,
eval 115.96 → 4.18,
holdout 116.16 → 34.91.
Train's largest single component holds 7384 of
12344 items — 0.5982 of the split.

A consumed-minus-non-consumed mean difference over 12344 rows that are
effectively 2.78 independent units is not a statistic
about a population. It is a statistic about one cache key. So is an admitted fraction, so is a
lift, so is an AUC — and so, above all, is a binomial *p*-value.

v2 applied its component discipline to the **split assignment** and not to the **estimator**.
That gap is the whole of what v3 closes.

## 3. The estimator: an intra-split cap

### 3.1 Why a cap and not component-equalized weights

The manifest publishes both options. Component-equalizing (`w_i = 1/(C·n_c)`) drops no rows
and pays in variance: Kish effective sample size falls to
2649.53 on train,
1727.46 on eval and
1629.48 on the holdout, and on the cold
subpopulation alone to 494.35,
258.06 and
272.8.

The cap is registered instead, for four reasons stated before any sign was read:

1. **The binomial.** The registered guard is an *exact* one-sided binomial on integer counts.
   Under fractional weights there are no integer counts and no exact test — one would have to
   invent an effective-*n* and round, which is a fitted quantity smuggled into a guard. Under
   a cap the counts are counts of real rows.
2. **The integer minima.** Four registered numbers are counts of rows: 400 cold candidates
   per split, 200 cold and 40 consumed-cold rows for a direction check, 40 admitted cold rows
   on the holdout, and *K* ≥ 20 for guard (c). Under weights each would have to be re-read as
   a weighted count — a change to a protected number by reinterpretation.
3. **Where the weight goes.** Equalizing multiplies a one-row component by *C* and divides the
   7384-row component by
   7384. It buys independence by moving
   almost all the mass onto the least-observed units.
4. **Disjointness survives either way.** The guarantee comes from assigning *whole* components
   to splits. Capping usage inside the split a component was already assigned to puts no cache
   key and no session id into a second split. The manifest says so at
   `enlargement.concentration_cost.an_intra_split_cap_does_not_weaken_disjointness`.

Component-equalized weighting is not discarded: it stays a mandatory published arm on every
verdict-carrying statistic, beside the capped one and the uncapped one. It carries no verdict.

### 3.2 Which cap, and the floor that picks it

Two numbers are registered together, and the order matters: the **floor** first, then the
largest cap that meets it.

`effective_components = 1 / Σ_c s_c²` (inverse Simpson over component shares). The identity
that makes the floor derivable rather than invented is elementary: if the largest share is
*s* then `Σ_c s_c² ≥ s²`, so `effective_components ≤ 1/s²`, and therefore
`effective_components ≥ E` implies `s ≤ 1/√E`.

The manifest already registers `split.one_cluster_bar.limit = 0.25` — no component may hold
more than a quarter of a split. On the effective-count scale that bar is exactly **E ≥ 16**,
and that is what the **cold-subpopulation floor** is set to: the cold cohort must satisfy on
its own the bar the whole split already had to satisfy.

**E ≥ 100** implies `s ≤ 0.10`, a strict tightening of the same bar, and that is the
**whole-split floor**. It is set higher there because the whole split is what the binomial
guard, the bands and the lift are computed on, and sixteen units cannot carry a *p*-value at
α = 0.05. One hundred leaves a factor of five.

Effective components per split, per cap, from `enlargement.cap_feasibility.table`:

| cap | train | eval | holdout | train items kept | eval items kept | holdout items kept | meets floor 100 |
|---|---:|---:|---:|---:|---:|---:|:--:|
| 25 | 212.55 | 148.4 | 140.8 | 3761 | 2429 | 2344 | YES |
| 50 | 169.48 | 126.62 | 113.51 | 4517 | 2667 | 2682 | YES |
| 100 | 134.32 | 104.32 | 76.0 | 4986 | 2812 | 3122 | no |
| 200 | 114.49 | 80.15 | 44.2 | 5160 | 2912 | 3791 | no |
| 400 | 81.49 | 42.89 | 34.91 | 5360 | 3112 | 4115 | no |
| *uncapped* | 2.78 | 4.18 | 34.91 | 12344 | 5285 | 4115 | no |

The floor of 100 is met at caps 25 and 50 and missed at 100 (holdout 76.0),
200 and 400. **The largest feasible cap is 50**, and the selection is unique: substituting any
floor in {77 … 113} returns the same cap. Nothing about the selection touches a sign, a label
or an outcome.

At cap 50 every split still clears the 400-cold minimum —
train 1035,
eval 558,
holdout 489 — so the protected split minimum is
untouched by the choice.

The cap's price is named rather than hidden: it keeps
0.366 of train,
0.505 of eval and
0.652 of the holdout. Those rows are not
evidence lost; they are evidence that was being counted many times over. But the headroom it
costs is real — see §7.

### 3.3 What the cap touches, and what it does not

It touches **every statistic that carries a verdict**: directions, admitted fractions,
`feasible_threshold_count`, the threshold and `t_ablation` themselves, lift, AUC, the admitted
cold count against the 40-row minimum, the binomial and its base rate, the warm precisions and
*K*, and the effective-component counts.

It does **not** touch the calibration tables, which stay fitted on all of `coldstart_train`
uncapped; nor the score; nor the split assignment; nor the live selector, which estimates
nothing. It is drawn once per cohort and held fixed for every statistic on that cohort —
re-drawing it per statistic would reintroduce as a degree of freedom exactly what it exists to
remove.

## 4. The directions, and how they were registered

### 4.1 The rule, stated before it was applied

The registered sign is read off the **whole** published component-aware arm set — the five caps
and the component-equalized arm — never off one arm. Change the estimation cap to any other
grid point and every registered sign is unchanged.

- **Rung 1.** All six arms agree in sign on *both* readings, whole tail and cold → register it.
- **Rung 2.** The six are unanimous on exactly one of the two readings → register that reading's
  sign. **This is the tie-break rule**, stated before any eval number exists.
- **Rung 3.** Neither reading unanimous, or both unanimous and disagreeing → the member has no
  registrable direction. Since the family is closed and no member may be dropped, the plan
  terminates with the explicit finding that the candidate-time family carries no direction-stable
  cold-start signal over this store's full history.

Unanimity rather than a majority, for two reasons. The caps *nest* — the survivors at 25 are a
subset of those at 50 and so on — so the five cap arms are a monotone sequence, not five
independent readings, and a 3-of-5 majority would be decided by where in the sequence the sign
turns. And a rule that always returns a sign cannot produce the negative finding the HALT asked
to be reachable.

### 4.2 The measurement, and the outcome

All four tables below are copied field-for-field from
`dataset-manifest.json#coldstart.train_direction_accounting`. The six component-aware arms are
above the rule; the three italic rows are published for audit and register nothing.

**`result_score` — registered POSITIVE, rung 2**

| arm | whole tail | cold | both match `positive`? |
|---|---:|---:|:--:|
| cap 25 | +0.048144 | -0.007383 | NO |
| cap 50 | +0.042548 | +0.006125 | yes |
| cap 100 | +0.032610 | +0.010187 | yes |
| cap 200 | +0.030927 | +0.013390 | yes |
| cap 400 | +0.027699 | +0.003702 | yes |
| component-equalized | +0.055918 | -0.037101 | NO |
| *uncapped (v2 arm)* | +0.026820 | -0.051024 | NO |
| *era pre-Aug* | +0.058894 | +0.008425 | yes |
| *era Aug* | +0.019879 | +0.013799 | yes |

**`trigger_score` — registered POSITIVE, rung 1**

| arm | whole tail | cold | both match `positive`? |
|---|---:|---:|:--:|
| cap 25 | +0.063560 | +0.008498 | yes |
| cap 50 | +0.054013 | +0.006267 | yes |
| cap 100 | +0.044194 | +0.004869 | yes |
| cap 200 | +0.042155 | +0.004227 | yes |
| cap 400 | +0.040346 | +0.003937 | yes |
| component-equalized | +0.059551 | +0.013709 | yes |
| *uncapped (v2 arm)* | +0.019100 | +0.010233 | yes |
| *era pre-Aug* | +0.008730 | +0.010165 | yes |
| *era Aug* | +0.056568 | +0.003252 | yes |

**`bm25_score` — registered NEGATIVE, rung 1**

| arm | whole tail | cold | both match `negative`? |
|---|---:|---:|:--:|
| cap 25 | -0.017924 | -0.072749 | yes |
| cap 50 | -0.012712 | -0.075771 | yes |
| cap 100 | -0.016996 | -0.076764 | yes |
| cap 200 | -0.016794 | -0.070165 | yes |
| cap 400 | -0.015346 | -0.068724 | yes |
| component-equalized | -0.028178 | -0.019443 | yes |
| *uncapped (v2 arm)* | +0.004903 | -0.011408 | NO |
| *era pre-Aug* | +0.004313 | -0.028642 | NO |
| *era Aug* | -0.022356 | -0.071024 | yes |

**`vector_score` — registered NEGATIVE, rung 1**

| arm | whole tail | cold | both match `negative`? |
|---|---:|---:|:--:|
| cap 25 | -0.053633 | -0.113273 | yes |
| cap 50 | -0.048609 | -0.104734 | yes |
| cap 100 | -0.055185 | -0.107386 | yes |
| cap 200 | -0.057486 | -0.105803 | yes |
| cap 400 | -0.056211 | -0.107631 | yes |
| component-equalized | -0.059596 | -0.119531 | yes |
| *uncapped (v2 arm)* | -0.029671 | -0.096195 | yes |
| *era pre-Aug* | +0.008223 | -0.057157 | NO |
| *era Aug* | -0.078777 | -0.102274 | yes |

### 4.3 Reading the tables

**`bm25_score` is the member the goal named as the tie-break case** — and under the uncapped
arm it is: cold -0.011408, whole tail
+0.004903, two readings of opposite sign.
Under every one of the six component-aware arms the split disappears and both readings are
negative. bm25 registers at rung 1 and the tie-break never fires for it. That is the single
cleanest piece of evidence that the HALT was an estimator artifact — and it is evidence, not a
licence: bm25 still has to hold its sign on cohorts this plan has not read.

**`result_score` is the member the tie-break actually applies to.** Its whole-tail sign is
positive in all nine arms. Its cold sign is positive at caps 50, 100, 200 and 400 and negative
at cap 25 (-0.007383) and under
component-equalizing (-0.037101). Rung 2
registers it positive from the whole-tail unanimity. **This is the member most likely to kill
the plan**, and that is written down here before the number exists.

**At the registered cap of 50 all four signs match their own train readings on both readings.**
Six readings elsewhere in the table contradict the sign registered for their member; all six are
listed in the JSON at `direction_constraint.expected_train_readings`, registered in advance so
none can later be presented as a surprise or as a reason to re-argue a sign.

**The registered signs are, member for member, v2's signs.** That is an *outcome* of the rule,
not an input to it. v3's composite is therefore arithmetically identical to v2's, and this plan
does not claim a new family. It claims that the estimator under which v2's family failed was not
measuring the population, and that under an estimator that is, the same four signs are what the
measurement supports.

## 5. Why the train check is no longer a test

**The directions in this plan were taken from `coldstart_train`. A check of those directions on
`coldstart_train` is a tautology.** Running it as a gate and reporting that it passed would be
precisely the self-confirmation these files exist to prevent.

So step 3 does something else. It computes the same readings on train in all three arms and per
era, publishes them, and checks that they **reproduce** the values registered in the plan to
within 1e-6. That check *is* falsifiable and *is* worth running — a mismatch means the
population, the extractor or the u-axis moved between the sealing of the manifest and the run,
and every number in the plan would then be describing a different dataset. It is a reproduction
check on the pipeline, not a test of the family, and the JSON does not permit it to be reported
as the latter.

The binding, falsifiable direction test moves to **`coldstart_eval` and `secondary_anchor`,
whole tail and cold subpopulation evaluated separately**, at step 5, after the parameter digest
is sealed and before the holdout is opened. Neither cohort has ever had a direction reading
computed on it for this family — the manifest states so at
`train_direction_accounting.why_only_train`, and v2's run halted at step 3 and never reached
its step 5.

The `cold_subpopulation_clause` minima are carried unchanged at 200 cold rows and 40 consumed
cold rows, now counted on the capped set, and INDETERMINATE still **blocks** rather than passes.

## 6. Eras: the second finding, and why it is the same finding

The finding: the pre-August slice consumes at
0.632563 inside `coldstart_train`
and 0.148703 inside
`coldstart_eval`. A calibration table fitted on train is being applied to a materially different
regime.

Look at what those two slices are:

| split | era | items | components | effective components | largest share | consumption rate |
|---|---|---:|---:|---:|---:|---:|
| train | pre-Aug | 8826 | 107 | 1.43 | 0.836619 | 0.632563 |
| train | Aug | 3518 | 174 | 75.09 | 0.039795 | 0.444571 |
| eval | pre-Aug | 1002 | 1 | 1.0 | 1.0 | 0.148703 |
| eval | Aug | 4283 | 199 | 7.24 | 0.366799 | 0.373803 |
| holdout | pre-Aug | 2476 | 45 | 13.35 | 0.134895 | 0.668417 |
| holdout | Aug | 1639 | 143 | 104.35 | 0.018304 | 0.473459 |

**`coldstart_eval`'s entire pre-August slice is ONE component** —
1 component,
effective components 1.0, largest share 1.0,
1002 items. Its rate of
0.148703 is one cluster's rate, not an
era's rate. Train's pre-August slice is nominally
107 components
but 1.43
effective, with one component holding
0.8366
of its items.

**So the era gap and the independence collapse are the same fact.** Read as an era effect,
0.632563 against
0.148703 is a comparison of two
individual cache keys — and a difference between two units is exactly what an estimator with
2.78 and 4.18
effective units cannot distinguish from a population effect.

There is an arithmetic coincidence the fit is required to settle: eval's largest component
overall holds 2573 items; its pre-August slice
is one component of 1002; its
August-era largest component holds
1571.
Those two sum to exactly the first. That is consistent with a single component straddling the era
boundary — in which case eval's whole era difference is a *within-component* difference and the
era cut on eval is not a cut at all. The plan does not assert it; component identities are opaque
digests. It requires the fit to publish whether they are the same component.

**What the cap does to it, exactly.** Eval's pre-August slice is one component, so at cap 50 it
contributes at most 50 rows to eval's capped set of 2667 —
at most 0.0187 of it, against an uncapped share of
0.1896. Train's
largest component contributes at most 50 of 4517, at most
0.0111, against 0.5982.
Both are exact bounds, not estimates: one component cannot exceed the cap. The regime whose rate
differed is bounded to under two per cent of either capped estimation set.

**What is not threatened by the non-exchangeability, and why.** Each calibration table is a
*monotone* step function applied by lookup: if raw *a* > raw *b* then *u(a)* ≥ *u(b)* on every
cohort, whatever the era mix. A foreign mix changes which part of the table is used, never the
ordering. The direction statistic, the family arm and the AUC are all rank quantities computed
*within* one cohort, so none is exposed to the mix. And `holdout_cold_lift` is normalised by the
cold base rate **of the same holdout split**, so a shifted mix moves numerator and denominator
together — which is why 1.5 survives this finding without moving and without needing to.

**What is threatened is the threshold**, a train-fitted scalar applied to a differently composed
cohort. That exposure is already caught by a guard carried unchanged from v2: guard (d) checks
the admitted-fraction bands on train *and eval* at step 6, and if either is out of band **the
holdout is never opened**. v3 adds two things to it — the bands are now judged on the capped set,
and the per-era decomposition of every verdict-carrying statistic is mandatory, so a pooled
in-band number cannot hide one era at 0.02 and another at 0.45.

One new binding clause, the **Simpson guard**: if a member's pooled capped eval reading matches
its registered sign while *every measurable era contradicts it*, that member fails. Pooled
agreement that no stratum supports is not evidence of a direction. It is registered as binding
and **expected to be vacuous on eval** — eval's pre-August era contributes at most 50 capped rows,
far below the 200-row measurability bar — and that expectation is registered so a vacuous result
cannot be reported as a passed one.

What the plan deliberately does *not* do about eras: no era reweighting (a second estimator
choice made after seeing the eras differ, reweighting toward a stratum that is one component);
no per-era band (there is no feasibility argument for one, and a band invented without one is the
error v2 documented in v1); no re-drawn split (step 9 forbids it and the plan makes no exception
for itself).

## 7. What did not move, and how this plan can fail

**Carried forward unchanged, none of them lowered:** overall admitted band [0.10, 0.30]; cold band
[0.05, 0.25]; generalization threshold `holdout_cold_lift ≥ 1.5`; split minimum of 400 cold
candidates; holdout minimum of 40 admitted cold rows; `derived_fit_floor` 0.10; binomial α = 0.05;
`holdout_evaluations` exactly 1; all four anti-fiat guards; the closed four-member family;
`no_cold_branch`; one threshold only; and the permanent exclusion of `graph_score`, which may not
be re-tested.

Two of these are now evaluated on a strictly smaller row set — the 400-cold minimum and the
40-admitted-cold minimum are counted on the capped set — which can only make them harder to clear.

**The one new minimum** is the effective-component floor, 100 whole-split and 16 cold. Adding a
minimum is a tightening and is permitted. It is also the condition v2 lacked: v2's step 2 checked
the 400-cold minimum, disjointness and the blindness invariant, all three of which *passed*, and
said nothing about how many independent units a split contained. Under this floor v2 would have
halted at step 2 — train at 2.78,
eval at 4.18 — before a
single direction was read, and therefore before any sign could be seen and re-argued.

**How this plan dies, written down in advance:**

- `result_score`'s cold sign fails on `coldstart_eval` at step 5. It reached rung 2, not rung 1;
  its cold sign is not unanimous across the component-aware arms. This is the likeliest failure.
- Any core member fails on `coldstart_eval` or `secondary_anchor`, whole tail or cold, or is
  INDETERMINATE for want of cold rows.
- Guard (d) or guard (c) fails on eval at step 6; the holdout is never opened.
- `feasible_threshold_count` on the capped train split is 0; the holdout is never opened.
- A cold effective-component floor of 16 is missed. It is registered **blind** — the manifest does
  not publish cold effective components per cap, and this node could not compute them without
  reading eval and holdout rows. A halt on it is pre-registered, not a surprise.
- **Power, on the tightest bar in the plan.** At cap 50 the holdout carries 489 capped cold
  candidates, so 40 admitted needs a cold admitted fraction of 0.081800 against a train
  selection target of 0.10 — about 18 per cent of slack. Uncapped the
  same slack was about 44 per cent. That is the price of the floor: a
  floor of 100 forbids cap 100, which would have left 552
  capped cold candidates. If the plan fails only here, that is a power failure and not a refutation
  of the family — and it is still a FAIL, and the minimum still does not move.

**One honest caveat about the floor itself.** Inverse Simpson measures concentration, not
independence. Capping a component at 50 rows leaves those 50 rows correlated with each other.
"Effective components ≥ 100" should be read as "no cluster holds more than a tenth of the estimate",
which is what the identity actually gives. The binomial *p* computed on it is still an
approximation — a far better one than a *p* over
2.78 units, and
still an approximation.

## 8. `coldstart_eval` is not virgin, and the holdout is

The holdout has never been read by anyone. `holdout_evaluations` is 0 under v1 and under v2, and
`policy.json#coldstart_revision.findings.the_shape_was_never_tested.holdout_still_virgin` records
that no cold-start holdout number has ever existed. The whole verdict rests there, exactly once.

`coldstart_eval` has been read once. v2's halted run published its frontier and its admission at
the threshold its rule would have returned, because v2's own `threshold_rule.published` required
`feasible_threshold_count` on eval whatever the verdict. What exists on the record:
`feasible_threshold_count` 582, and at threshold
0.87186366 an admitted fraction of
0.195459, cold admitted fraction
0.12945, cold lift
1.512, cold AUC
0.63297, binomial *p*
0.000246842.

Since v3's composite is arithmetically identical to v2's, those numbers were computed under,
effectively, this plan's shape. **Two of this plan's gates are therefore partially foreknown** —
guard (d) on eval, whose fractions at that threshold were both inside the registered bands, and
the eval cold lift. The threshold under v3 is selected from *capped* fractions and will not in
general equal 0.87186366, so the foreknowledge is
approximate rather than exact. It is foreknowledge all the same, and a reader should discount
those two gates accordingly.

**What is not discounted is the eval direction test** — the test this plan moves onto eval and
rests its evidentiary weight on. No direction reading on `coldstart_eval` has ever been computed,
by v1, by v2, or by the sibling that published the independence accounting.

## 9. What was read to write this

`coldstart-prereg-v2.json` in full; aggregate published fields of `dataset-manifest.json`
(`coldstart.train_direction_accounting`, `coldstart.enlargement.*`, `coldstart.splits.*.concentration`
and `.stratified.by_era`, `coldstart.split`, `coldstart.protocol`, `secondary_anchor` totals); and
`policy.json#coldstart_revision` in full, including its eval diagnostic.

No item row of any split was read. No database was queried. No file inside the fail-closed subtree
`policy.json#holdout_boundary` reserves was touched, and that directory's literal path is
deliberately not spelled anywhere in these two artifacts, because
`scripts/recall_map_relevance_eval.py:64` `HOLDOUT_MARKERS` rejects it as a raw substring before
resolving any path.

The directions registered here were **measured by a different node** —
`coldstart-population-independence` — and published in the manifest before this node read them.
What this node contributed is an estimator choice and a mechanical rule over the published arm set.
That is the whole of the discretion exercised, and both are stated in a form that can be
re-executed against the manifest without re-running anything.

The corresponding hazard is named rather than denied: a plan written after seeing which signs
failed could choose an estimator that unfails them. The two defences are on the page. The
estimation cap is selected by a floor rule that never touches a sign and returns the same cap under
any floor in a wide band. And the registration ladder reads all six component-aware arms
*unanimously* rather than any one of them, so no choice of cap can set a sign.
