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
