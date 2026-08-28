# Cold-start selector — pre-registered design

**Status: registered before any fit.** The machine-readable source of truth is
[`coldstart-prereg.json`](coldstart-prereg.json), `plan_sha256`
`d17082c4d3b321a71cdb1399163c352728511ef4dd25c685fbf10de2d40b5df3` over the
fields `plan_sha256_over` names, including itself.

This file was written by a node that owns only these two artifacts. It fitted
nothing and read no dataset the sibling `coldstart-dataset` node produces — that
node was running in parallel and its output did not exist here at registration.
Whoever states the generalization threshold must not have seen a fitted result,
and that separation is the point rather than an accident of scheduling.

## The defect, in arithmetic

`src/living_memory/recall_map.py:1190` `_history_features` returns
`RELEVANCE_FEATURE_MEANS[1..4]` verbatim when history is unavailable. Four of the
five z-terms are then identically zero and `relevance_score` collapses to the
level term alone, against `RELEVANCE_THRESHOLD = 3.8708378402511`:

| row | history z-terms | schema | trace / concept |
|---|---|---:|---:|
| history unavailable | `0, 0, 0, 0` (means substituted) | **+1.6682** | **−0.5994** |
| history available, `M=C=K=0` | `−1.1084, −1.0913, −0.8947, −0.519` | **−1.9452** | **−4.2129** |
| delivered 1000×, never consumed | `+1.5722, +1.6117, +3.0175, −0.519` | +7.3505 | +5.0829 |

A node with *available but empty* history scores worse than one with no history
at all, because four zero-valued features sit far below their train means.
Admission needs `log1p(M) ≈ 6.9` — roughly a thousand matured windows. That is
the delivery loop the root goal describes: delivery grows history, history clears
the selector, the selector delivers again.

No better signal fed into the existing five can repair this.
`follow-signal/census.json#frozen_scorer_probe` proves the K term is
*monotonically increasing* — with `M=20, C=0` the score climbs from `0.427201` to
`2.151198` as the unfollowed tail lengthens from 0 to 20. A new feature family is
the only shape that works.

## 1. The family: recorded delivery scores

`scripts/recall_map_relevance_eval.py:91` already reserves the name
`recorded_delivery_scores` as an analysis-only family. This registration promotes
it to a fitted one, overturning
`policy.json#selected_policy.rejected_features.recorded_delivery_scores`
("absent from all 686 observed-map eval items").

That rejection was right for a policy whose transfer evidence *was* the
observed-map arm. It is the wrong rejection for cold start.
`src/living_memory/retrieval.py:345` `RecallResult` is a frozen dataclass whose
`score`, `bm25_score`, `vector_score`, `graph_score` and `trigger_score` are
`float` fields with `0.0` defaults. **At live map-build time every residual entry
carries all five, always.** The family is absent from one historical arm, not
from the decision. Rejecting a live-always-present family because one extractor
did not persist it is letting a recording gap choose a production feature set.

The values are already in hand and thrown away: `recall_map.py:1930`
`_select_pool` enumerates `results` and reads only `.node`.

Registered members, with the association evidence copied from
`feature-analysis.json#associations` (organic train / organic eval / secondary
anchor):

| member | direction | weight | train | eval | anchor |
|---|---|---:|---:|---:|---:|
| `result_score` (`score`) | positive | +1 | +0.47069 | +0.27967 | +0.08752 |
| `trigger_score` | positive | +1 | +0.25200 | +0.16743 | +0.04680 |
| `bm25_score` | **negative** | −1 | −0.05069 | −0.07457 | −0.01995 |
| `vector_score` | **negative** | −1 | −0.08142 | −0.09368 | −0.01689 |
| `graph_score` | positive | +1 | +0.02243 | +0.02628 | +0.02082 | *conditional, excluded by default* |

Two things are deliberate here.

**bm25 and vector enter negative.** They are consistently negative in all three
cohorts that record them, which inverts the naive reading that a better lexical
or embedding match is a better candidate. A same-signed negative direction is
transferable evidence — it is simply not a positive one, which is why the
direction constraint had to stop saying "positive" (§3). Under signed unit
weights the family reduces to a channel-composition contrast: *a high blended
score earned by trigger and graph rather than by surface match.* That population
is already named in the codebase — `replay.py:114` identifies `graph_score > 0`
with `bm25_score == 0` and `vector_score == 0` as a distinct kind of hit.

**`graph_score` is excluded by default.** Its whole-tail direction is positive in
all three cohorts, but an approximate probe of the cold subpopulation alone
(2026-08-01..08-22, ranks 3..8, n=3610 cold of 19746) found it *flipping* to
−0.011 against +0.017 whole-tail. That probe used a proxy consumption label and
no component split, so it is go/no-go evidence only and cannot reject the feature
by itself. It is enough to move the burden of proof: this is the one member whose
whole-population sign is not safe on the very cohort the revision would admit. So
exclusion is the *registered default*, admission requires a positive cold
association on the fit split alone, and falling back to the four-member core is a
pre-registered outcome rather than a post-hoc retreat. `graph_score` may not be
re-admitted with a negative sign — a sign flip found in the data is a finding to
report, not a feature to fit.

Rejected candidates and why: `access_count` and `usefulness_score` stay banned as
mutable post-outcome stats (`policy.json#selected_policy.bans`); `methods` is a
thresholded discretization of components already in the family; `path_length` and
`scope` have never had a direction measured, so none can honestly be registered;
residual rank would encode where the extractor cut the tail
(`protocol.organic_head_cut = 3`) rather than the candidate; `node_age` reverses
sign between cohorts and stays rejected.

### Why four members and not two

A heavily delivered row carries about **+5.68** of history z that a cold row
structurally cannot obtain. For a cold row to clear the threshold the family must
supply **+2.2026** (schema) or **+4.4703** (trace/concept). Four signed unit-weight
members can plausibly reach that on a genuinely strong candidate — top-of-
distribution score, trigger matched, zero bm25, zero vector accumulate positive z
on all four terms at once. A two-member family could not reach +4.47, so trace and
concept nodes would stay categorically inadmissible while cold start was declared
repaired for schema nodes only: the same collapse one level down.

## 2. The missing-value rule

| cohort | items | with recorded scores |
|---|---:|---:|
| organic train | 3102 | 3102 |
| organic eval | 1389 | 1389 |
| secondary anchor | 4061 | 3723 |
| observed-map eval | 686 | **0** |

Missingness has two causes and collapsing them is how a recording gap gets scored
as evidence.

**Cause A — live unreadable.** Cannot occur for a well-formed `RecallResult`; the
rule is a fail-closed guard against a malformed residual. Such a row is
**inadmissible**, carrying the existing `lr` code. No reason code is added,
reordered or spent: `SELECTION_REASON_CODES` is frozen and positional, and a row
whose evidence cannot be read is below every threshold, so `lr` already states the
truth.

**Cause B — historically unrecorded.** The 686 observed-map items and 338
secondary-anchor items. These are excluded from the family's fit, direction checks
and threshold calibration, counted and reported as `family_unscoreable` rather
than silently dropped, kept in their cohort for the legacy-feature checks and kept
in every admitted-fraction denominator. When the revised policy is *applied* to
such a row, the row is inadmissible, as in cause A.

**Mean substitution is prohibited under either cause** — and there is no weaker
version available. Omitting a z-term from an additive standardized sum is
*arithmetically identical* to substituting that member's mean, because the omitted
contribution is `(mean − mean)/scale = 0`. A scorer cannot both be additive and
treat an unreadable member as neutral without imputing the mean. So the only
fail-closed rule an additive scorer admits is that an unscoreable row is
inadmissible, and the consequences are registered below rather than discovered
later.

No missingness indicator either. It could not fit on anything real: live
missingness is structurally impossible, so the indicator would train entirely on
*which historical extractor wrote the row* — source identity, already banned — and
it would correlate perfectly with the observed-map arm, making it a cohort label.

### Does the observed-map arm remain a transfer cohort for this family?

**No.** Zero of 686 items record any delivery score; the arm cannot measure this
family's direction, magnitude or threshold. Keeping it would mean verifying a
claim against data that cannot express it.

It is retained in three other roles: direction cohort for the legacy five (at full
original force), an admitted-fraction ceiling cohort, and a reported cohort.

Its **expected admitted count under the revision is 0**, because every row is
`family_unscoreable` and therefore inadmissible. This is registered here, before
any fit, so it can later be presented neither as a surprise nor as a failure. It
follows that the frozen policy's observed-map transfer PASS — 37 items, 9
consumed, `0.243243` against the `0.233` floor — **is not inherited by the
revision and must not be claimed for it.** The revision's transfer evidence is the
three-way cold split, and nothing else.

## 3. The direction-constraint amendment

Today `selected_policy.fit.train_direction_constraint` reads: *"each selected
feature has positive consumed-minus-nonconsumed association in organic train,
organic eval, and observed-map eval."*

No candidate-time score feature can satisfy that as written. Applied to this
family the constraint is not strict, it is **undefined** — the observed-map arm
records no scores, so there is no association to be positive or negative. It also
assumes every selected feature is positive, which two registered members are not.

The amended constraint is per family, checked on the transformed values the scorer
consumes, and makes three changes:

1. **Signed, not positive.** Each member must show its *registered sign*, so
   `bm25_score` and `vector_score` are held to negative and a flip fails them.
2. **Third cohort replaced** — for this family only — by `secondary_anchor`.
3. **Cold clause added**, which the original constraint never had.

| family | cohorts |
|---|---|
| legacy `level_schema` + four history terms | organic train, organic eval, observed-map eval — **unchanged** |
| `recorded_delivery_scores` | organic train, organic eval, **secondary anchor** |

`secondary_anchor` earns the slot: it records the scores (3723/4061 = 91.7%
against 0/686), it already carries measured same-signed directions for all five
fields, and it is a genuine distribution shift — the *anchor* stratum rather than
the primary tail, with systematically weaker associations (+0.088 against +0.471
on train), which is what a transfer cohort should look like rather than a
re-slice.

Its weakness is stated rather than glossed: **it is not session-independent.** Its
component list overlaps train's (e.g. `0022e83e0b…` appears in both) because the
anchor stratum and the primary tail are different ranks of the same delivering
events. It tests transfer across *retrieval stratum*, not across sessions. It is a
direction check; it is not the generalization evidence. Session-level independence
comes from the three-way component split in §4, which is where the generalization
claim actually rests.

**The cold clause is the real teeth.** Each member's registered sign must also
hold on the cold subpopulation of each cohort, measured separately (minimum 200
cold rows and 40 consumed cold rows, else the check is INDETERMINATE — never
"passed", and an indeterminate cold check on a core member blocks the revision).
A whole-population sign is a statement about the cohort the frozen policy already
admits; it is not evidence about the cohort this revision would newly admit, and
the graph_score probe shows the two can differ. Without this clause the amendment
would be strictly *weaker* than what it replaces — trading a cohort that cannot
measure the family for one that can, and asking nothing new. With it, it is
strictly stronger on the axis that matters.

Nothing is silently failed: observed-map eval keeps full original force on the
features it can measure, and the fit artifact must record its admitted count, the
reason `family_unscoreable_fail_closed`, a citation of this registration as where
the expectation was set in advance, and the statement that the `0.243243` result
is not inherited.

## 4. The split protocol

Same identity rule as the existing work —
`connected_components(source-qualified cache key, transport session, session)`,
implemented at `recall_map_relevance_eval.py:702` — with event-level component
membership, because an event carrying no scorable item can still be the cache or
session bridge between two events.

Three pairwise component-disjoint partitions: `coldstart_train` (0.60),
`coldstart_eval` (0.20), `coldstart_holdout` (0.20), assigned by

```
draw = int(sha256(seed + "\0" + component_id).hexdigest()[:16], 16) / float(16**16)
```

under a **new** seed `recall-map-coldstart-components-v1`. The seed must differ
from `recall-map-relevance-components-v1`: reusing it would reproduce the existing
boundary, making the "holdout" a re-slice of the old eval — a set already consumed
by the frozen policy's transfer verification. Any component containing an
observed-map delivery event is forced to `coldstart_eval`, carrying over the
existing protection and keeping the holdout free of an arm that is
`family_unscoreable` by construction.

### Naming

`recall_map_relevance_eval.py:64` `HOLDOUT_MARKERS` rejects three literal
substrings *before* resolving any path, matching both raw text and the resolved
lowercased path. Those tokens are reserved for the post-deployment field
population at `policy.json#holdout_boundary`, whose status is
`semantic_boundary_frozen_population_not_created` — a different thing entirely
(candidate-only events created strictly after a sealed deployment instant) that
this subtree may neither create nor read. `coldstart_holdout` is a historical
partition of already-persisted state, lives in `dataset-manifest.json#coldstart`,
and touches none of it. Verified against the guard itself: the three new paths are
accepted and the reserved ones still rejected.

### How the holdout is sealed, and why "not refit on" is checkable

Each split publishes an opaque `membership_digest` (sha256 over canonical JSON of
the sorted per-item opaque keys) and a `component_digest`, never item rows —
preserving `dataset-manifest.json#privacy.item_level_rows = false`. Order is
fail-closed: dataset writes and commits the digests → fit reads `coldstart_train`
and writes the parameter digest to disk → only then may eval be read → only then
may the holdout be read, exactly once.

The primary check converts an unfalsifiable claim into a deterministic equality:

> **Re-run the fit with `coldstart_holdout` physically absent from the input. The
> fitted parameter digest — clip points, means, scales, member set, threshold —
> must be byte-identical to the full run's.**

If any holdout row influenced any parameter, the digests differ. Supporting
checks: the holdout `membership_digest` cited by the policy must equal the
manifest's (a re-draw changes it); the commit introducing that digest must be an
ancestor of the policy commit (`git merge-base --is-ancestor`), so the seal
predates the fit in recorded history rather than in prose; the three component
lists must be pairwise disjoint.

### Residual contamination, stated

`feature-analysis.json#associations` already publishes this family's aggregate
directions over cohorts that jointly cover the population the splits are drawn
from. The holdout is therefore **not virgin with respect to the family's
direction**. That leak is inert here: those are cohort-level mean differences,
never item rows, so they fix no row's membership — and the directions they inform
are registered in this file *before* any fit and before the splits exist.
Committing to a direction in advance is what a pre-registration is *for*; the leak
would be choosing it after seeing the holdout, which this file makes impossible.
The holdout is virgin with respect to every quantity the verdict depends on: clip
points, means, scales, member set, threshold, admitted fraction, lift. The claim
is "directions pre-registered, magnitudes and threshold held out" — not a fully
blind holdout, which it is not.

## 5. Cold candidates in every split

A **cold candidate** is one whose strictly-matured prior delivery count at the
map-build instant is zero (`matured == 0`). That covers both arms of the defect —
history unavailable *and* history available-but-empty — because both are rows with
no delivery history, both are arithmetically incapable of admission today, and
both are what the live residual is predominantly made of. Splitting them would
create a cohort defined by a storage detail rather than by the node's situation.

`minimum_cold_candidates_per_split = 400`. Every split must contain cold
candidates, because a cohort of history-bearing rows is exactly the leakage being
repaired — that was the failure of the frozen fit, whose cohort was ledger
organic-tail items that all had history. 400 is anchored on the one cold
measurement in hand: the probe found 3610 cold rows in a narrow three-week,
rank-3..8 slice of 19746, so a 20% partition of even that narrow slice yields
about 722. If a split cannot reach it the fit **halts** — it may not proceed on a
smaller split or rebalance the fractions to manufacture one.

Coldness is a reporting and split-audit partition only. It must never be a
feature, a branch, a tie-break or an eligibility test.

## 6. The generalization threshold

> **`generalization_threshold = 1.5`**, on
> `holdout_cold_lift` = (consumed among admitted cold holdout candidates ÷ admitted
> cold holdout candidates) ÷ (consumed cold holdout candidates ÷ cold holdout
> candidates), compared `>=`.

A lift rather than an absolute rate, because the cold subpopulation's base rate
has never been measured with a real label. Any absolute rate registered now would
be a guess that is either vacuous — below the base rate, passed by admitting
everything — or unreachable. **A lift cannot be gamed by volume: admitting every
cold candidate scores exactly 1.00 by construction, and so does admitting at
random.** It states the capability directly: the selector *discriminates* among
cold candidates.

The number is anchored on the frozen policy's own transfer performance,
recomputed from `policy.json#transfer_verification`:

| cohort | precision | base rate | lift |
|---|---:|---:|---:|
| organic train | 0.724083 | 0.505158 | 1.4334 |
| **organic eval** (disjoint) | 0.611111 | 0.378690 | **1.6138** |
| observed-map eval | 0.243243 | 0.083090 | 2.9275 |

1.50 sits just under the 1.6138 the frozen policy achieves on an independent arm:
the revision must discriminate among cold candidates nearly as well as the frozen
policy does among warm ones, with a small allowance for cold rows carrying no
history signal at all.

Companion conditions. Holdout cold admission must be **strictly greater than
zero** (parent contract) — subsumed by a registered floor of **40 admitted cold
rows**, because a bar of one row is decided by luck. And a one-sided exact
binomial guard, `P(X ≥ observed | n = admitted cold, p = cold base rate) ≤ 0.05`,
so the lift cannot be met by luck on a handful of rows.

At α = 0.05 the two conditions divide labour cleanly instead of one dominating.
Once about 100 cold rows are admitted, lift ≥ 1.50 binds at every plausible base
rate. Below that the binomial binds and demands a proportionally larger effect — at
the floor of 40 rows it requires lift 2.50 at a 0.05 base rate, 2.00 at 0.10, 1.83
at 0.15, 1.62 at 0.20. α = 0.01 was considered and **rejected as
over-constraining**: at a 0.05 base rate it would demand 427 admitted cold holdout
rows before lift 1.50 could clear it, which the cold ceiling could forbid outright.

## 7. The noise guard

The live selector admits 1.51–1.89% of candidates against a policy calibrated for
20%. A revision that "fixes" that by admitting everything is not a fix — it is the
map turning into noise, the one thing the root goal says the repair must not do.
Without a stated ceiling nothing distinguishes a selector that learned which cold
nodes are worth delivering from one that stopped selecting.

| bound | value | scope |
|---|---:|---|
| overall admitted fraction | **≤ 0.30** | every split |
| **cold** admitted fraction | **≤ 0.25** | every split |
| cold admitted fraction | ≥ 0.05 | train |
| overall admitted fraction | ≥ 0.10 | train |

Denominator is candidates eligible after the unchanged ballast filter. The cold
ceiling is the tighter of the two because that is where the flooding risk sits:
the live residual is predominantly cold, so an unbounded cold rate would dominate
the pool by itself. The cold floor exists because cold admission today is
arithmetically zero — it is what stops the fit from satisfying the lift by
admitting almost no cold rows at all.

The frozen policy's `minimum_train_selection_fraction = 0.20` is deliberately
**not** carried over. It was calibrated against a warm-only fit cohort of 3102
delivered items; the cold-inclusive candidate population is a different and much
larger denominator, so reusing the number would be a category error dressed as
continuity. 0.10 preserves the intent against the denominator that now applies.

For reference, the frozen policy's admitted fractions were 0.2046 (organic train),
0.1564 (organic eval) and 0.0567 (observed-map eval).

## The scorer's shape, and the branch that is forbidden

A single additive standardized sum, exactly as `_directional_zsum` does today. The
legacy five keep their positions and arithmetic; the family is **appended**. Every
member enters at unit magnitude with its registered sign — no coefficient is
fitted, preserving the frozen policy's rule that no coefficient was optimized on
labels. Members are winsorized at train 1st/99th percentiles (monotone and
sign-preserving, bounding any single heavy-tailed row), then standardized on train
mean and scale, then `round(…, 8)` to reproduce threshold ties. No log transform:
the registered directions were measured as mean differences on raw values, and a
log would leave them describing a quantity the scorer does not use.

**Only the scalar threshold may be fitted on labels**, on `coldstart_train` only.
`3.8708378402511` is not inherited — it was the score at descending rank 621 of
3102 warm items under a five-feature scorer, and is meaningless against a wider
vector on a cold-inclusive population.

**No cold branch.** A separate threshold or bonus for `matured == 0` would make
admission a function of a membership test rather than of evidence — the flag-gate
shape banned at `bans.forbidden_enablement`, and exactly what the capability
contract means by a lookup table: it would "fix" cold start without learning
anything about *which* cold nodes are worth delivering. If a single threshold
cannot satisfy the cold band and the overall ceiling at once, **that is a
pre-registered negative result to be recorded as such** — not a problem to be
solved by adding a branch.

The revision must not disturb: the frozen positional order of
`SELECTION_REASON_CODES`, the append-only `POOL_GATE_REASON_CODES`,
`_SelectionLedger.freeze` trimming trailing zeros so an unarmed server still emits
exactly seven counts in `sel.x`, the accounting equation
`sel.n == sel.e + sum(sel.x)`, and `RELEVANCE_POLICY_ID` moving together with
`RELEVANCE_POLICY_DIGEST`.

## What counts as PASS

All of: `holdout_cold_lift ≥ 1.5`; admitted cold holdout count ≥ 40 and > 0;
binomial `p ≤ 0.05`; overall admitted fraction ≤ 0.30 and cold ≤ 0.25 on every
split; and the withheld-holdout re-run reproducing a byte-identical parameter
digest. Any other outcome is a recorded FAIL or negative result, and none of them
may be repaired by widening the family, adding a branch, re-drawing a split, or
re-reading the holdout.

A negative result is a legitimate outcome. If a single additive threshold cannot
satisfy both bands, or the registered signs do not survive the cold subpopulation,
the correct output names which condition failed and by how much. That is a finding
about the feature family, and it is worth more than a passing number obtained by
amending this file after seeing it — which
[`coldstart-prereg.json#amendment_policy`](coldstart-prereg.json) forbids: no
direction, sign, member, cohort, seed, fraction, minimum, threshold, bound or
alpha may be changed by amendment, only by a new registration sealed before the
next number is read.
