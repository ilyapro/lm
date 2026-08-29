# transcript-grounding — goal-level two-field closure

**Combined verdict: `negative-stop` on both fields, and therefore on the goal.**

Machine-readable form: `two-field-closure.json` beside this file. Every number
below was recomputed from the tracked artifacts and, for the alt field,
re-derived directly against the alt store read-only — not copied from upstream
summary fields.

---

## 1. What this closes

The goal's boundary section pins the measurement fields as «оба стора
(локальный и alt) — фазы 1–2 гоняются на каждом хосте отдельно, урок
однохостовых слепых пятен учтён». Phases 1–2 have now been run separately on
both hosts. The obligation is discharged by measurement, not by argument.

It had stayed open because all five phase children returned `done` while each
one individually deferred the alt field.

---

## 2. Both fields, side by side

| | local | alt |
|---|---|---|
| phase-1 coverage | 6798 / 59074 = **0.115076** | 4426 / 16098 = **0.274941** |
| 60% falsifier | **fail** | **fail** |
| echo ceiling | 0.126468 (7471 ids) | 0.273512 (4403 ids) |
| transcript retention | partial (~446 sessions rolled off disk) | **full** (missing_transcripts = 0) |
| consumed share, store-wide | 0.171734 | 0.151348 |
| consumed share, matched | 0.647249 | 0.539313 |
| enrichment | **3.77x** | **3.563397x** |
| dominant tier | event_id_echo, 93.6% of matches | event_id_echo, 96.2% of matches |
| a neutral tier exists? | yes — transport_session, 0.182 ≈ base rate | **no** — transport_session 0.414201 vs base 0.151348 |
| phase-2 | negative-stop | negative-stop |
| phase-4 / 5 | negative-stop / not-built | no-run-can-be-formed / unchanged |

Both fields fail the pre-registered falsifier independently, on separate hosts,
at their own structural ceilings. Each is already at the maximum any join could
reach on its corpus, so neither number is a runner defect and neither is
tunable.

---

## 3. The two falsifiers, stated honestly

**Falsifier 1** («покрытие corpus join < 60% → цель останавливается») — **fired
on both fields**. This is the actual root cause.

**Falsifier 2** («replay A/B не показывает улучшения → клапан не включается,
вердикт negative») — its **antecedent was never evaluated**: no replay A/B ran
on either field, so "does not show improvement" was never observed. Claiming it
fired on its own terms would assert a measured refutation that was never taken.
Its operative consequent — the valve stays off — holds anyway, through
`prereg.md` §7.2: only a `positive` verdict licenses the valve, and no verdict
of any kind exists.

---

## 4. The finding that outlives the goal

**The failure belongs to the join mechanism, not to either host's disk
retention — and not to the idea being tested.**

The local fail was originally diagnosed as structural to local disk retention,
i.e. a host property. The alt field refutes that: alt retains transcripts across
its entire event history and still ceilings at 27.4%. Retention modulates the
number (12.6% → 27.4%); it does not cause the failure.

The mechanism: the only tier that finds transcripts at scale, `event_id_echo`,
matches an event because its `recall_event_id` was written into the transcript —
and the dominant reason an id lands there is the `memory_remember` provenance
that **consumes** it. The join is closer to a consumption detector than to a
neutral link. Hence ~3.5x consumption enrichment among matched events on *both*
fields, holding inside every comparable calendar month on both.

Its sharpest form, measured on alt: `delivery_record_index` — the anchor the
grading method requires, because grading must begin strictly *after* the
delivery instant to avoid the LM tool-result echo trap — is present on exactly
the 4257 `event_id_echo` rows and NULL on all 169 `transport_session` rows.
Every event the prescribed method could grade comes from the consumption-detector
tier; every event from the non-detector tier is ungradeable for want of a
delivery instant. The bias is not merely large, it is structurally inseparable
from the method — so it could not have been mitigated by grading only the clean
tier.

### What that does to the goal's value claim

The goal promised the learning signal would grow «кратно (~×5 по покрытию
событий)». Granting *every* silent matched event a grounded verdict — an upper
bound no real threshold would reach:

* **alt**: 0.151348 → 0.277978 event coverage, **×1.84**
* **local**: 0.171734 → 0.212327 event coverage, **×1.24**

against a ~×5 claim. The ~×5 figure presumed a join reaching most silent
recalls; neither field's join does. (Derived here from recorded counts, not
pre-registered; the local baseline is back-derived from the recorded share and
is approximate to the unit.)

---

## 5. Phase states at closure

* **1 — corpus join**: done on both fields; both fail the falsifier.
* **2 — grounding grade**: stopped on both. No grader exists
  (`scripts/transcript_grounding_grade.py` is absent), no threshold was
  calibrated, no noise floor was measured, no `method_version` was minted.
* **3 — additive ledger**: code exists, deployed nowhere, fed nothing.
  `transcript_grounding_verdicts` is absent from **both** stores, verified
  read-only on each.
* **4 — replay A/B**: no run can be formed on either field. `method_version V`
  and strictness candidate `S` are both unbindable, and binding either by
  invention would mislabel invented semantics as pre-registered.
  `replay/verdict.json` is deliberately **absent**, and its absence remains
  load-bearing as the phase-5 gate.
* **5 — valve**: not built. Live path untouched.

`prereg.md` is byte-identical to its seal
(`1542f93682784cbf13a6c4ff873346f8bb114a0679bdfe8605571f8a67c9a927`); §8.2 is
not triggered, and this closure grants no licence to relax any threshold in it.

---

## 6. Reopening

Not another host, and not better join tiers — a third field would reproduce the
same ceiling for the same reason.

Reopening requires **prospective instrumentation** that records the delivery
instant for recalls independently of whether a `memory_remember` later consumes
them, so the joinable population stops being selected by the very outcome the
signal is meant to predict. Phase 1 would then be re-measured, not re-argued.

Nothing measured here tests whether transcript-grounded silent-recall
reinforcement helps. The blocker sits upstream of the hypothesis.
