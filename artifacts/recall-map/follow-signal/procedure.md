# Follow-signal weights: what the field says, and how an operator turns it into thresholds

Companion to `census.json`, produced by `scripts/recall_map_follow_signal_census.py`.
Regenerate both with:

```
python3 scripts/recall_map_follow_signal_census.py \
  --corpus 'field:live-global=~/.local/share/living-memory/global.sqlite3' \
  --build-synthetic /tmp/census-synth.sqlite3 \
  --out artifacts/recall-map/follow-signal/census.json
```

**No default is set here.** This file is a decision procedure, not a decision.
Neither the usefulness admission valve nor the demotion valve has a default in
code or in config, and this artifact does not propose one be added. The same
rule that governed `LM_DRAIN_NEAR_DUP_SUPERSEDES` governs these: the operator
turns a valve on, from numbers, deliberately.

---

## 1. The headline, stated before anything is inferred from it

The field arm is the live store the running LM server writes to
(`~/.local/share/living-memory/global.sqlite3`), read `mode=ro`, never opened
through `MemoryStore`, never migrated; server and dashboard untouched.

| | field (`live-global`) |
|---|---|
| matured delivery windows | **185 357** |
| **known** windows (`lookup_consumed IS NOT NULL`) | **0** |
| **NULL** (unobservable) windows | **185 357 — 100 %** |
| wall clock the lookup signal has existed *in this database* | **0 hours** |
| recall events | 57 194, from 2026-05-14 to 2026-08-23 (~101 days) |

The `recall_lookup_events` table is **absent** from the live store and its
ledger is still at format 1, with no `lookup_consumed` column. The lookup
signal exists in this branch's code and in its tests; it has never run against
a live database, because deploying it means restarting the LM server, which
this goal's boundary forbids.

So the question this node was asked — *how strongly should a lookup count
against a re-delivery* — **cannot be answered from field data yet, and this
artifact does not answer it.** Zero known windows is not a small sample; it is
no sample. Section 4 fixes the procedure that will answer it, in advance of the
data, which is the only remaining way to keep the answer honest.

The second arm in `census.json` is labelled `synthetic`. It is a nine-window
hand-built scenario whose only job is to prove the classifier discriminates all
four classes and keeps NULL separate (it yields 3 `lookup_followed`, 1
`ask_follow`, 1 `redelivered_only`, 2 `nothing`, 2 NULL). **Those counts are
not evidence of anything about the corpus** — the scenario was constructed to
contain one of each — and no threshold below is drawn from them.

### Statistical floor, pre-registered

A rate is called *decided* here when its 95 % Wilson interval has a half-width
of ≤ 0.10. That needs **97 known windows** in the worst case (`p ≈ 0.5`), or
**35** if the rate turns out to be near 0.1. Both numbers are in
`census.json → arms[].volume.power`. Until the known-window count clears the
applicable one, every rule below returns *under-powered* and both valves stay
off.

---

## 2. What the field arm **does** establish today

These four findings need no lookup window; they come from the ledger and the
frozen scorer as they stand, and each is reproducible from `census.json`.

### 2.1 The ledger's re-delivery bit and the query-echo probe are uncorrelated

Over the 1 812 matured **map-medoid** windows (the only ones carrying a label a
later query could echo):

| ask-follow correlation | fires on | P(ask ∣ re-delivered) | P(ask ∣ not re-delivered) | lift |
|---|---|---|---|---|
| `any_transport` | 76.1 % of windows | 0.791 | 0.758 | **1.04** |
| `same_transport` | 12.4 % of windows | 0.386 | 0.100 | **3.85** |

Read both, and neither in isolation:

- The `any_transport` lift of **1.04** is the quantitative form of the
  diagnosis that opened this work. Knowing a window was re-delivered tells you
  essentially nothing about whether anybody asked about it again. But this
  reading is also saturated: with ~570 recalls a day and a two-token label
  needing one shared token, a global echo probe fires on three windows in four,
  so it discriminates nothing in either direction.
- The `same_transport` lift of **3.85** looks like re-delivery carrying real
  signal — and is confounded, in the obvious direction. A window is
  re-delivered *by a later recall*, and it is ask-followed *by a later recall*;
  restricting both to the same transport session means both are largely
  measuring "the session kept going". The lift is an upper bound on
  re-delivery's worth, not an estimate of it.

Neither width gives a usable exogenous ground truth. That is the argument for
the id-fetch signal existing at all, and it is why the weighting question is
deferred to real lookup windows rather than settled from this table.

### 2.2 The frozen K term cannot demote anything

`census.json → frozen_scorer_probe`, with `matured = 20, consumed = 0`:

| trailing_nonconsumed | 0 | 1 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| relevance (schema) | 0.427 | 0.820 | 1.212 | 1.442 | 1.785 | **2.151** |

The K feature is monotonically **increasing**: a row delivered twenty times and
consumed none of them scores *higher* the longer it has gone unconsumed. The
sign was learned by the `directional-zsum-r1` fit, not chosen, and the frozen
evaluator is out of scope to change.

**Consequence for the demotion valve, and it is a design constraint, not a
remark:** demotion cannot be obtained by feeding a truer signal into
`trailing_nonconsumed`. It has to be a gate applied *outside* the zsum, after
scoring, with its own selection-journal reason. Any design that routes the new
signal into K makes hubs stickier, not less sticky.

### 2.3 "Sticky" names two different node populations

Ranking by ledger re-delivery count and ranking by map-offer count produce
almost disjoint top tens:

| cohort (top 10) | windows | re-delivered | map-medoid windows |
|---|---|---|---|
| by re-delivery count | 5 558 | 1 770 | **17** |
| by map-offer count | 770 | 121 | **162** |

The first cohort is `project:online` traces with `usefulness_score = 1.0` — the
ledger's *organic tail* limb (result ranks 3–8), which the recall map does not
control and the pool gates do not govern. Only 1 812 of 185 357 windows (1.0 %)
are map medoids at all.

**Consequence:** the valves must be tuned on the map-offer cohort. A threshold
fitted to the corpus-wide re-delivery distribution would be fitted to traffic
the valves cannot touch.

### 2.4 The named hub survives on the re-delivery ratio, not on `C/M = 1, K = 0`

`01KSB3GDM0J3ZB24QSRA97RRWJ` (`project:ae`, level `schema`,
`usefulness_score = 0.0777`), as the ledger actually holds it:

| M | C | K | C/M | relevance | admission threshold |
|---|---|---|---|---|---|
| 251 | 92 | 4 | 0.367 | **6.237** | 3.871 |

Its score decomposes as `consumed_ratio 2.620` + `level_is_schema 1.668` +
`log1p_matured 1.037` + `log1p_unconsumed 0.894` + `log1p_K 0.017`.

Two corrections to the premise this work started from, both worth carrying
forward:

1. The intent's "these nodes ride at `consumed = 1, K = 0` forever" is not
   literally what the ledger says — `C/M` is 0.367 and `K` is 4. The
   *direction* is right: the endogenous re-delivery ratio is the single
   largest term in the score (42 % of it), and with `level_is_schema` the two
   carry 69 %. But a valve written against a literal `C/M = 1` would miss this
   node entirely.
2. The "4 616 re-deliveries" in the diagnosis is `access_count`, equal to the
   number of recalls that *returned* the node
   (`recall_history_result_nodes`). The ledger records only result ranks 3–8
   plus map medoids, so its window count for the same node is **251**, of which
   only **3** are map offers. Both numbers are in `census.json →
   top_sticky_rows[].node.returned_by_recalls` and `.matured_history.matured`;
   quoting either alone misstates the problem by an order of magnitude.

---

## 3. Reading the census

Each arm reports the same shape twice-over: `distribution.overall` (every
matured window), `distribution.map_medoid_windows_only` (the recall map's own
offers), and `distribution.top_sticky_cohorts.{by_redelivery_count,
by_map_offer_count}`. Inside each:

- `known` — windows with an observed lookup outcome. This is the partition the
  valve thresholds get drawn from, and it is empty in the field today.
- `null` — windows whose lookup outcome is unobservable. Their re-delivery and
  ask-follow arms are still counted, because those *are* observed; only the
  lookup arm is missing. Never fold these into a `known` denominator.
- `by_ask_correlation` — the class split under each correlation width. The
  class priority is `lookup_followed` → `ask_follow` → `redelivered_only` →
  `nothing`, so a window that is both looked up and re-delivered is classed as
  looked up while `lookup_by_redelivery` keeps both bits.
- `ask_follow_probe.applicable_windows` — the ask-follow denominator. Windows
  without a label are `not_applicable`, never a probe that found nothing.

`diagnostics.echo_agreement` re-answers a 200-call sample of the probe the slow
way, through `recall_map._echoes` itself over every query in range, and reports
the disagreement count. It is 0 in the committed run. If it is ever non-zero,
the inverted index has drifted from the module and no number in the arm may be
used.

---

## 4. The decision procedure

Each valve gets: an estimator, a rule fixed *before* the data exists, a
stop condition, and a falsifier. Run section 5's checklist first; if it fails,
the answer is "leave both valves off" and no further reading is needed.

### 4.1 Lookup-versus-re-delivery weighting — decide this one first

The other two depend on it. The estimand is how much a re-delivery is worth as
evidence that a row was genuinely used, taking a lookup as the unit of
exogenous evidence.

**Estimator.** From `distribution.map_medoid_windows_only.known
.lookup_by_redelivery`, over map-medoid windows only:

```
LR = P(lookup_followed | re-delivered) / P(lookup_followed | not re-delivered)
   = [n11 / (n11+n01)] / [n10 / (n10+n00)]
```

where `nSR` is `signal_S__redelivered_R`. Report the 95 % Wilson interval on
each arm; `LR` is decided only when both arms are decided.

**Pre-registered rule**, `w` being the weight a re-delivery carries relative to
a lookup in the demotion rule of §4.2:

| `LR` 95 % interval | reading | `w` |
|---|---|---|
| interval contains 1.0 | re-delivery carries no evidence of use | **`w = 0`** — re-delivery neither demotes nor protects |
| interval entirely above 1.0 | re-delivery is a noisy but real proxy | **`w = clamp(1 − 1/LR_low, 0, 0.5)`** |
| interval entirely below 1.0 | re-delivery *anti*-predicts use | `w = 0`, and open a separate goal: the ranker is actively re-delivering what nobody reads |

`w` is capped at 0.5 by construction, and that cap is the intent's own
constraint rather than a statistical one: re-delivery **softens** the demotion
and must never veto it, because a signal a hub satisfies trivially would make
hubs immortal by definition. Even `LR = 20` buys `w = 0.5`, never a veto.

**Stop condition.** ≥ 97 known map-medoid windows, or ≥ 35 if both arms'
point estimates are under 0.2 — whichever bound applies, checked against
`volume.power`, not by eye.

**Falsifier.** If `n11 + n01` (re-delivered known windows) is under 20, `LR`'s
denominator is unstable no matter how large the total: report under-powered and
keep waiting, even if the overall known count has cleared 97.

### 4.2 Demotion N — how many known windows without exogenous follow demote a row

Per §2.2 this is a gate outside the frozen zsum, not a re-tuning of K.

**Definition.** A known window counts as *exogenously followed* when
`lookup_consumed = 1`, or when the ask-follow probe fires under the
`same_transport` width. NULL windows are **skipped** — they neither extend nor
break a run. A re-delivery does not clear the run; it discounts it, by `w` from
§4.1: a run of `n` windows of which `r` were re-delivered counts as
`n − w·r` effective windows against the row.

**Estimator.** Over rows that eventually *did* show an exogenous follow, take
the distribution of the longest preceding run of known-and-unfollowed windows
— the "how long does a genuinely useful row go quiet" distribution. Call its
95th percentile `q95`.

**Pre-registered rule.** `N = max(3, ceil(q95) + 1)`.

The floor of 3 is `CURTAIL_STREAK`'s own reasoning, borrowed deliberately: two
is inside the noise of one distracted session, and there is no case for
demoting a row faster than the channel-level curtail collapses a whole map.
The `q95 + 1` term is what makes the rule falsifiable — it is set so that at
most 5 % of rows that were about to be useful get demoted first, and if `q95`
comes back large enough that `N` exceeds the number of windows a typical row
ever accrues, the honest conclusion is that per-row demotion cannot work on
this corpus and only the channel-level curtail can.

**Stop condition.** At least 30 rows with an observed exogenous follow *and* at
least one preceding known window, so `q95` rests on something. Fewer than 30:
under-powered, valve stays off.

**Falsifier.** If the demotion set computed at `N` on held-out windows contains
any row that shows an exogenous follow in the *next* window, `N` is too small
and the measurement is repeated one horizon later. Pre-register the held-out
split before computing it: last 20 % of windows by `delivered_at`.

### 4.3 Usefulness admission threshold

`usefulness_score` does not enter map cluster selection at all today, which is
how a node with a verdict of 0.0777 stays a medoid.

**Estimator.** Bucket the *map pool* rows by `usefulness_score` decile and, per
decile, compute the exogenous-follow rate over known windows with its Wilson
interval.

**Pre-registered rule.** The threshold is the **highest** decile boundary `t`
satisfying both:

1. every decile strictly below `t` has an exogenous-follow rate whose 95 %
   upper bound is below the corpus-wide rate's 95 % lower bound — the excluded
   band is measurably worse, not merely lower; and
2. excluding everything below `t` removes ≤ 20 % of the current pool,
   measured on the same corpus.

If no `t` satisfies both, **there is no threshold** and the valve stays off.
That is a permitted and expected outcome: `usefulness_score` may simply not be
predictive of follow, and the second condition exists so a valve cannot be
justified by emptying the pool.

**Stop condition.** ≥ 35 known windows *per decile* considered — not 35 across
the corpus. With ten deciles this is the most data-hungry of the three rules
and will be the last to become answerable; coarsening to quintiles or terciles
is allowed, and must be declared in the artifact before the rates are read.

**Falsifier.** Excluding the band must not measurably reduce map coverage:
re-run `scripts/recall_map_effect.py` with the valve on and off and require the
covered-cluster count to stay inside its pre-registered band. A usefulness gate
that improves follow rates by delivering fewer maps has proved nothing.

---

## 5. Before either valve is turned on

A checklist, all of which is machine-checkable from a fresh census:

1. **The signal is deployed.** `arms[].schema.lookup_table_present == true` and
   `ledger_format_version == 2` on the live store. Today: **false / 1**. This
   needs an LM server restart, which is the operator's call and outside this
   goal.
2. **Windows have accrued.** `volume.known_windows` clears the bound in
   `volume.power` for the specific rule being decided (§4.1: ≥ 97 known
   map-medoid windows; §4.2: ≥ 30 rows with a follow; §4.3: ≥ 35 per bucket).
3. **The accrual window is long enough to contain ordinary variation.**
   `volume.lookup_signal.observed_days_to_last_activity ≥ 14`. Seven days of
   one project's work is not the corpus's behaviour; the map's stickiest
   clusters rotate on a multi-day cycle.
4. **The probe still agrees with the module.**
   `diagnostics.echo_agreement.disagreements == 0`.
5. **The frozen surfaces are unmoved.** `RELEVANCE_POLICY_DIGEST` unchanged,
   `frozen_scorer_probe.admission_threshold` still 3.8708378402511, and the
   M/C/K triple byte-identical to the pre-lookup golden. If any moved, every
   number above is void and the census is re-run before anything is decided.
6. **The rule was written down before the numbers were read.** Sections 4.1–4.3
   are that record. If a threshold ends up chosen outside these rules, say so
   in the artifact and say why — a documented deviation is recoverable, an
   undocumented one is not.

### What must be **re-measured**, not merely re-checked

- §4.1's `LR` after any change to the ranker's schema-trigger boost: the boost
  is what makes re-delivery endogenous, so retuning it changes the estimand
  itself.
- §4.3's decile rates after any `usefulness_score` recomputation or decay
  sweep — the buckets are not stable across a rescoring.
- All three after the `curtail-key-alignment` change lands. Aligning curtail on
  `task_pattern` will start collapsing per-turn-task channels that have never
  collapsed, which removes deliveries from the corpus and changes the window
  population these rates are computed over.

---

## 6. Scope note

This artifact and its script set no default, register no environment variable
name, and change nothing in `src/`. The env-valve names belong to the
`pool-gates-env-valves` node; deliberately, none is mentioned here, so that
naming them cannot be mistaken for enabling them. With both valves unset the
map output stays byte-identical to the pre-change build, which is that node's
postcondition, not this one's.

---

## 7. Amendment, 2026-08-24: what §4 could not be evaluated on, and what replaced it

Written by the `pool-gate-threshold-and-charge` node **before any pool-candidate
distribution or follow rate was read.** The commit that adds this section adds
no measurement; `artifacts/recall-map/pool-quality/gate-decision.json` and
`scripts/recall_map_pool_usefulness_census.py` land after it, and the git order
is the record. §6's scope note no longer holds for this section alone: this one
names the env valves, because deciding them is the amending node's whole job.

### 7.1 The signal deployed, and that is the only checklist item that moved

§5.1 has flipped. Commit `178c2e3` shipped to both hosts on 2026-08-24 and
`recall_delivery_history` is at `format_version = 2` with the tri-state
`lookup_consumed` column; `recall_lookup_events` exists. Measured on
`~/.local/share/living-memory/global.sqlite3`, opened `mode=ro`, at
**2026-08-24T12:52:26Z**:

| quantity | value |
|---|---|
| `recall_delivery_history` rows | 187 887 |
| rows with `lookup_consumed IS NOT NULL` (*known*) | **723** |
| known rows that are **matured** (`outcome_end <= now`) | **0** |
| earliest known window's `outcome_end` | **2026-08-24T12:55:07Z** — 2 min 41 s in the *future* |
| `recall_lookup_events` rows | 14 |
| map-medoid share of matured ledger windows (`census.json`) | 1 812 / 185 357 = **1.0 %** |

The middle two rows are the whole difference between §4 and this amendment. The
column began being written the morning of the measurement, and a window matures
24 hours after it opens, so the *first* window that can ever carry an id-fetch
verdict closes minutes after this reading. Zero matured known windows is not a
thin sample that a careful estimator can still squeeze; it is an empty one, and
every rate §4 names divides by it.

§5.3 fails for the same reason and independently:
`observed_days_to_last_activity` for the lookup signal is under one day against
a required 14. Even if maturity had been reached, the accrual window would be a
single morning of one project's work.

### 7.2 The three estimators, and which of them died

| rule | stop condition | today | evaluable? |
|---|---|---|---|
| §4.1 `LR` (lookup vs re-delivery weight) | ≥ 97 known map-medoid windows, or ≥ 35 if both arms under 0.2 | **0** matured known windows, of which map-medoid **0** | **no** |
| §4.2 demotion `N` | ≥ 30 rows with an observed exogenous follow *and* ≥ 1 preceding known window | **0** such rows — an exogenous follow requires a matured known window | **no** |
| §4.3 condition 1 (decile follow rates) | ≥ 35 known windows **per decile** | **0** per decile; 0 across all ten | **no** |
| §4.3 condition 2 (≤ 20 % of pool removed) | none — it is a budget, not a rate | measurable from the candidate population alone | **yes, and binding** |
| §4.3 falsifier (coverage must not fall) | none | measurable by paired replay, see §7.6 | **yes, and binding** |

§4.3 condition 1 does not merely lack power; it is *decidably false for every
`t`*. The condition asks that each decile below `t` have a follow rate whose
95 % Wilson upper bound sits below the corpus rate's 95 % lower bound. On
`n = 0` the Wilson interval is the whole unit interval, so the comparison is
`1.0 < 0.0` in every cell of the table, for every candidate boundary. Applying
§4.3 as written therefore already returns *no threshold, valve stays off* —
which is a permitted outcome, and is the outcome this amendment must be able to
beat honestly or accept.

§4.1's coarsening escape hatch does not apply anywhere. §4.3's stop condition
permits collapsing deciles to quintiles or terciles and requires the choice be
declared before rates are read. **Declared: no coarsening.** Zero known windows
divided by three buckets is still zero, so coarsening buys no power and would
only make the table look less empty than it is. The decile table is computed and
published at ten buckets.

### 7.3 The demotion valve is not licensed under any substitute, and stays off

§4.2's estimand is a percentile of an outcome distribution — the longest run of
known-and-unfollowed windows preceding a row's *first exogenous follow*. There
is no non-outcome form of that quantity: `q95` is not a property of the corpus,
it is a property of rows that were followed, and today no row has been followed
because no window has matured. `N = max(3, ceil(q95) + 1)` cannot be evaluated,
approximated, or bounded from the candidate distribution.

`LM_MAP_POOL_DEMOTION_GATE` and `LM_MAP_POOL_DEMOTE_AFTER` therefore stay unset
on both hosts regardless of what §7.4 decides about the usefulness floor. No
substitute is offered for §4.2 because inventing one would be inventing the
data. This is recorded as a numeric decline: **0 rows with an observed exogenous
follow, against a pre-registered floor of 30.**

### 7.4 The substitute for §4.3 condition 1, declared before the numbers

Condition 1's job is to establish that the excluded band is **measurably
worse**, not merely lower. No outcome evidence exists, so the substitute cannot
make that claim and does not try to. It replaces one outcome condition with
three non-outcome ones, all of which must hold at a candidate boundary `t` for
the valve to be charged.

Write `P` for the replayed pool-candidate population of §7.5.

- **S1 — the threshold is not inert.** `t` is a decile boundary of `P` with
  `t > 0.0`, and the band `{score < t}` is non-empty on `P`. The code compares
  strictly (`_below_usefulness`, `recall_map.py:1338-1343`), so a boundary of
  exactly `0.0` removes nothing; charging an inert valve would put a number in
  the environment of two production hosts while changing no behaviour, and
  would make the seven-wide `sel` contract report a gate that never fires.

- **S2 — the budget, inherited unchanged from §4.3 condition 2.** Excluding
  everything below `t` removes ≤ 20 % of the pool. §4.3 does not say which
  denominator "the pool" means, and the two available readings differ by two
  orders of magnitude, so both are declared here and **both must pass**:
  - *admitted* — the members `_pool` returns after the cap, i.e. what reaches
    clustering and what `RecallMap.pool` counts. This is the reading the code's
    own vocabulary supports and the one "emptying the pool" describes.
  - *candidates* — the rows that reach the floor at all: post-`iv`, post-`du`,
    post-ballast, pre-history. This is the population the deciles are taken
    over, so a budget stated against it is the one that binds the table.

- **S3 — cold-start non-regression, the substitute proper.** Among the
  candidates `t` removes, the share that are **cold** — no matured delivery
  history at all — must not exceed the cold share of the candidates `t`
  retains.

  S3 exists because `usefulness_score` is not an independent verdict on a node.
  It is a delivery-history feature. `nodes.usefulness_score` is
  `NOT NULL DEFAULT 0.0` (`storage.py:110`) and moves only through
  `feedback.apply_retrieval_feedback`, which adds `0.1 · signal` to a node
  *that a recall returned and a caller fed back on* (`feedback.py:282-283`).
  A node nobody has retrieved is therefore at exactly the schema default, and
  70.8 % of the corpus (12 375 / 17 472 nodes) sits there. A floor above zero
  is, mechanically, a floor on "has this been delivered and rewarded before" —
  the same rich-get-richer selection that features 1–4 of `relevance_score`
  already impose and that the root goal was opened to break. S3 is the cheapest
  test that distinguishes a floor which removes junk from a floor which removes
  the cold, and it is the only part of condition 1's intent that survives
  without outcomes: it cannot show the excluded band is worse, but it can show
  the excluded band is not simply *younger*.

**Decision rule.** Take the candidate boundaries in descending order. The
threshold is the highest `t` satisfying S1, S2 and S3 together and passing the
§7.6 falsifier. If no `t` satisfies all of them, **there is no threshold, the
valve stays off, and the exact failing condition is recorded with its numbers**
in `gate-decision.json`. Declining is the pre-registered default and needs no
further justification than a failing cell.

This substitution is a documented deviation in §5.6's sense. Stated plainly:
**S1–S3 are weaker than condition 1 and do not prove the excluded band is less
useful.** A valve charged on them is charged on a distributional argument and a
harm check, not on evidence of benefit, and it must be revisited under §4.3 as
written once ≥ 35 matured known windows per decile exist. Any charge made here
carries that expiry in `gate-decision.json`.

### 7.5 The candidate population is replayed, not proxied

The distribution `P` is produced by `scripts/recall_map_pool_usefulness_census.py`,
which replays real `recall_events` queries against a **frozen snapshot** of the
live store through the same `MemoryRecallService` the server uses, takes
`last_residual`, and runs `RecallMapBuilder._pool`'s own classification pass over
it. The rows bucketed are the rows the floor would actually be asked about: the
survivors of the identity, duplicate and ballast checks, read at the point
`recall_map.py:1949` reaches the floor, before any history is fetched.

Two cheaper populations are excluded by name, because both are wrong and both
are wrong in a direction that would license a threshold this data does not
support:

- **the corpus-wide distribution** — 70.8 % of 17 472 nodes are exactly 0.0 and
  `p90` is 0.116. A floor read off it deletes nearly the whole corpus.
- **the recall-returned distribution** — `p20 = 0.0777`, 19.81 % below it. These
  are the nodes recall *delivered*, which is the opposite selection from the
  residual the map is built out of.

### 7.6 The falsifier, in the form that can actually run

§4.3 names `scripts/recall_map_effect.py` and requires the covered-cluster count
to stay inside a pre-registered band. That script cannot execute this falsifier
as written, and saying so is part of keeping the pre-registration honest: it
reads persisted `recall_events.recall_map` payloads out of history
(`map_items`, `recall_map_effect.py:719-758`) and never constructs a map. An env
valve that has never run leaves both of its arms byte-identical, so running it
"with the valve on and off" produces the same file twice and falsifies nothing.

**Declared executable form**, same intent, same quantity: the census script
replays the *same* query set against the *same* frozen snapshot twice, once with
`LM_MAP_POOL_USEFULNESS_GATE` unset and once with it charged at `t`, and reports
`covered` — `RecallMap.covered`, the cluster-coverage count the payload
carries — summed over the replay, together with the count of non-empty maps.

**Pre-registered band, one-sided:** `covered_on >= covered_off` and
`maps_with_clusters_on >= maps_with_clusters_off`. The band is one-sided because
the falsifier's own sentence is one-sided — "a usefulness gate that improves
follow rates by delivering fewer maps has proved nothing" — and because the
metric this valve exists to move is already at 46.3 % of recalls with no
clusters at all (1 305 / 2 818 persisted payloads, measured 2026-08-24). There
is no headroom below.

`scripts/recall_map_effect.py --verify-prereg` is still run, for what it does
establish: that the sealed plan and its baseline are unmoved, i.e. §5.5.

### 7.7 Result: both valves decline, and the decisive condition is a pre-registered one

Measured after §7.1–§7.6 were committed (`c069034`). Replay: 500 distinct
`recall_events` queries against a frozen snapshot of the live store
(`snapshot_sha256 c70875d9…`, 17 474 nodes, 57 781 recall events), `as-of`
2026-08-24T13:00:00Z, production recall cut `max_results = 5`.

**The replay is the real population, and it proves it.** Every one of the 498
queries with a residual asserted the mirrored classification pass against
`_pool`'s own `SelectionAccounting`; 498 checks, 0 failures. The replayed
admitted share is **5 069 / 339 707 = 1.49 %** against the live
`sum(sel.e)/sum(sel.n) = 1.5086 %` — the two agree to within a thousandth,
which is the corroboration that matters, because the admitted share is the
quantity the floor acts on.

**P is half zero.** Of 339 707 candidates, **185 643 (54.65 %) score exactly
0.0**, so decile boundaries 1 through 5 all land on 0.0 and a floor there is
inert. The only non-inert boundaries are `0.02`, `0.045`, `0.1133`, `0.52`.

| decile | range | n | cold | admitted | known windows |
|---|---|---|---|---|---|
| 1 | [-0.25, 0) | 100 | 60 | 0 | **0** |
| 2–5 | [0, 0] | 0 | 0 | 0 | **0** |
| 6 | [0, 0.02) | 194 482 | 149 988 | 908 | **0** |
| 7 | [0.02, 0.045) | 41 998 | 4 869 | 221 | **0** |
| 8 | [0.045, 0.1133) | 35 128 | 3 919 | 281 | **0** |
| 9 | [0.1133, 0.52) | 33 940 | 528 | 670 | **0** |
| 10 | [0.52, 4.0] | 34 059 | 327 | 2 989 | **0** |

The `known windows` column is the point. It is zero in every bucket, against
§4.3's stop condition of 35 *per decile*, and it is carried in the table rather
than omitted so the table shows its own emptiness.

**The budget, both denominators.** Removing everything below `t`:

| `t` | of 339 707 candidates | of 5 069 admitted | cold share removed | cold share retained |
|---|---|---|---|---|
| 0.02 | 194 582 = **57.28 %** | 908 = **17.91 %** | **77.11 %** | 6.64 % |
| 0.045 | 236 580 = **69.64 %** | 1 129 = **22.27 %** | 65.48 % | 4.63 % |
| 0.1133 | 271 708 = **79.98 %** | 1 410 = **27.82 %** | 58.46 % | 1.26 % |
| 0.52 | 305 648 = **89.97 %** | 2 080 = **41.03 %** | 52.14 % | 0.96 % |

Stated rather than buried: under the *narrow* reading of "the pool" — admitted
members only, which is probably what §4.3's own estimator meant — `t = 0.02`
**passes** condition 2 at 17.91 %. It is the single survivor of any denominator
reading, and it is therefore the only threshold on which anything else could
still matter.

**The pre-registered falsifier kills it.** Same snapshot, same 500 queries,
valve off and then charged at `t = 0.02`:

| | off | on | Δ |
|---|---|---|---|
| covered clusters | 1 056 | **743** | **−313 (−29.64 %)** |
| maps with clusters | 195 | **180** | −15 |
| admitted pool rows | 5 069 | 4 161 | −908 |

The band declared in §7.6 is `covered_on >= covered_off`. It does not hold, and
it does not hold by nearly a third. **The decline therefore rests on a
pre-registered condition that this amendment never touched**, not on the
substitute — which is the outcome an honest amendment should hope for, because
it means the weakened rule was never load-bearing.

**S3 rejects it independently, and says why the valve is the wrong shape.** At
`t = 0.02` the floor removes a population that is **77.11 % cold** and keeps one
that is **6.64 % cold**. `usefulness_score` is a delivery-history feature
(`feedback.py:282-283`, default `0.0` at `storage.py:110`), so a floor on it is
a floor on "has this been delivered and rewarded before". Charging it would add
a *sixth* history term to a selector whose cold-start defect is that four of its
five terms are already history. The gate is not mis-tuned; at this stage of the
corpus it is pointed the wrong way.

**Verdict.**

| valve | env | decision | failing condition |
|---|---|---|---|
| usefulness admission | `LM_MAP_POOL_USEFULNESS_GATE`, `LM_MAP_POOL_MIN_USEFULNESS` | **unset, not charged** | §4.3 falsifier: covered 1 056 → 743 at the only budget-surviving `t` |
| unfollowed-window demotion | `LM_MAP_POOL_DEMOTION_GATE`, `LM_MAP_POOL_DEMOTE_AFTER` | **unset, not charged** | §4.2 stop condition: 0 rows with an observed exogenous follow, floor 30 |

No environment changed, so **no host was restarted and there are no restart
instants to record.** Both hosts were read on 2026-08-24 to make the decline a
verified state rather than an assumption: neither `~/.config/living-memory/env`
(local) nor `/home/user/.config/living-memory/env` (alt) contains any of the
four variables, and the two files carry the same valve set otherwise
(`LM_DRAIN_NEAR_DUP_SUPERSEDES=1`, `LM_RECALL_NEAR_DUP_COSINE=0.97`). The
hosts are symmetric, and stay symmetric.

`scripts/recall_map_effect.py --verify-prereg --as-of 2026-08-19T11:00:00Z`
returns **OK** on all nine checks, `plan_sha256 d17b2e65…` reproducing against
the sealed plan: §5.5's frozen surfaces are unmoved, so none of the above is
void.

**What this does not say.** It does not say `usefulness_score` is a worthless
signal — only that no threshold on it is licensed today, and that both
conditions still capable of rejecting one do reject it. It does not say the map
is healthy; the replay reproduces the defect exactly (1.49 % of candidates
admitted, 303 of 498 maps empty). It says the fix is not on this lever. Removing
candidates from a pool that is already admitting 1.5 % cannot raise coverage,
and measured here, it lowers it by 29.6 %.

Re-open under §4.3 as written when ≥ 35 matured known windows exist per decile
over map-pool rows — months away at the 1.0 % map-medoid share of the ledger —
or, sooner and more usefully, after `selector-cold-start-recalibration` changes
`P` itself, at which point every cold share above is a re-measurement and not a
re-check.
