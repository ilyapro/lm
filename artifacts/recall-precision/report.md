# Recall precision valves: holdout report

Pre-registered in `prereg.md` (commits 86f0e7b and a1d0255, before this run). One frozen run per host
on the holdout segment, all events, production env. Numbers are per host and never merged. Every
effect is against the replayed baseline arm on the same events.

## sfx

- holdout events 167, cutoff 2026-09-28T15:23:18.000000Z, train-defined hubs 115

### Verdicts

| id | condition | value | verdict |
|---|---|---|---|
| G1 | gate irrelevant_full_cut >= 25% | 36.9% | **PASS** |
| G2 | gate used_full_lost <= 12% | 2.5% | **PASS** |
| G3 | gate rank1_removed == 0 | 0 | **PASS** |
| H1 | hub hub_top3_cut >= 80% | 35.9% | **FAIL** |
| H2 | hub used_full_lost <= 3% | 1.3% | **PASS** |
| D1 | fc080_mw1 census share m>0.9 < 30% (holdout window) | 23.1% | **PASS** |
| S1 | dedup dup_schema_slots == 0 | 0 | **PASS** |
| Q1-gate | gate new-entrant excess >= baseline delivered-set excess | new_entrant_excess 0.0%, graded 16, baseline_delivered_excess 4.2% | **FAIL** |
| Q1-hub | hub new-entrant excess >= baseline delivered-set excess | new_entrant_excess -4.8%, graded 21, baseline_delivered_excess 4.2% | **FAIL** |
| Q1-combined | combined new-entrant excess >= baseline delivered-set excess | new_entrant_excess 0.0%, graded 60, baseline_delivered_excess 4.2% | **FAIL** |

### Arms

| arm | full/ev | irr full | irr cut | used full | used lost | rank1 removed | hub top3 | hub cut | dup schema | chars/ev | chars saved | new entrants graded | entrant excess |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_a | 5.0299 | 236 | — | 79 | — | — | 39 | — | 36 | 12656.5868 | — | 334 | 0.0419 |
| combined | 2.9281 | 127 | 46.2% | 63 | 20.2% | 5 | 11 | 71.8% | 0 | 9672.7425 | 23.6% | 60 | 0.0 |
| gate | 2.9641 | 149 | 36.9% | 77 | 2.5% | 0 | 23 | 41.0% | 36 | 9741.9102 | 23.0% | 16 | 0.0 |
| gate_alt_T | 3.5988 | 176 | 25.4% | 77 | 2.5% | 0 | 24 | 38.5% | 36 | 10675.1198 | 15.7% | 9 | 0.0 |
| baseline_b | 5.0299 | 236 | — | 79 | — | — | 39 | — | 36 | 12656.5868 | — | 334 | 0.0419 |
| dedup | 5.0299 | 226 | 4.2% | 69 | 12.7% | 0 | 39 | 0.0% | 0 | 12649.1737 | 0.1% | 31 | -0.0323 |
| demote | 5.0299 | 231 | 2.1% | 75 | 5.1% | 0 | 38 | 2.6% | 36 | 12655.3892 | 0.0% | 7 | 0.1429 |
| hub | 5.0299 | 221 | 6.4% | 78 | 1.3% | 5 | 25 | 35.9% | 35 | 12677.9581 | -0.2% | 21 | -0.0476 |

`baseline_a` is the baseline for gate, gate_alt_T and combined, and `baseline_b` for hub, demote and dedup.
The two are the same replay in separate processes. For a baseline row, the entrant columns show the quality of the whole delivered set.

### By recorded rank (irr cut / used lost vs baseline; n = baseline full slots)

| arm | 1 | 2 | 3 | 4-5 | 6-10 |
|---|---|---|---|---|---|
| combined | 11.9% (n=42) / 3.6% (n=28) | 31.8% (n=44) / 9.5% (n=21) | 43.9% (n=41) / 35.3% (n=17) | 68.5% (n=73) / 33.3% (n=9) | 61.1% (n=36) / 100.0% (n=4) |
| gate | 0.0% (n=42) / 0.0% (n=28) | 27.3% (n=44) / 4.8% (n=21) | 36.6% (n=41) / 11.8% (n=17) | 56.2% (n=73) / -11.1% (n=9) | 52.8% (n=36) / 0.0% (n=4) |
| gate_alt_T | 0.0% (n=42) / 0.0% (n=28) | 22.7% (n=44) / 4.8% (n=21) | 29.3% (n=41) / 5.9% (n=17) | 35.6% (n=73) / 0.0% (n=9) | 33.3% (n=36) / 0.0% (n=4) |
| dedup | 2.4% (n=42) / 0.0% (n=28) | 11.4% (n=44) / 0.0% (n=21) | 2.4% (n=41) / 17.6% (n=17) | 4.1% (n=73) / 33.3% (n=9) | 0.0% (n=36) / 100.0% (n=4) |
| demote | 0.0% (n=42) / 0.0% (n=28) | 0.0% (n=44) / 4.8% (n=21) | 0.0% (n=41) / 11.8% (n=17) | 4.1% (n=73) / 11.1% (n=9) | 5.6% (n=36) / 0.0% (n=4) |
| hub | 2.4% (n=42) / 3.6% (n=28) | 4.5% (n=44) / 0.0% (n=21) | 9.8% (n=41) / 0.0% (n=17) | 9.6% (n=73) / 0.0% (n=9) | 2.8% (n=36) / 0.0% (n=4) |

### Demotion strength census (applicable event × marked-node cases)

| window | events | cases | default m>0.9 | fc080_mw1 m>0.9 | fc080_mw1 m<=0.75 | fc080_mw1 median |
|---|---|---|---|---|---|---|
| trainval | 741 | 2630 | 70.2% | 24.6% | 47.5% | 0.768 |
| holdout | 305 | 1213 | 76.9% | 23.1% | 45.3% | 0.7713 |

## alt

- holdout events 111, cutoff 2026-09-28T14:00:58.000000Z, train-defined hubs 18

### Verdicts

| id | condition | value | verdict |
|---|---|---|---|
| G1 | gate irrelevant_full_cut >= 25% | 22.6% | **FAIL** |
| G2 | gate used_full_lost <= 12% | 6.9% | **PASS** |
| G3 | gate rank1_removed == 0 | 0 | **PASS** |
| H3 | hub cut / used loss (reported) | hub_top3_cut 50.0%, used_full_lost -2.3% | **REPORT** |
| D2 | fc080_mw1 census share m>0.9 (reported) | 24.2% | **REPORT** |
| S1 | dedup dup_schema_slots == 0 | 0 | **PASS** |
| Q1-gate | gate new-entrant excess >= baseline delivered-set excess | new_entrant_excess 6.7%, graded 15, baseline_delivered_excess 5.2% | **PASS** |
| Q1-hub | hub new-entrant excess >= baseline delivered-set excess | new_entrant_excess 16.7%, graded 18, baseline_delivered_excess 5.2% | **PASS** |
| Q1-combined | combined new-entrant excess >= baseline delivered-set excess | new_entrant_excess 10.8%, graded 37, baseline_delivered_excess 5.2% | **PASS** |

### Arms

| arm | full/ev | irr full | irr cut | used full | used lost | rank1 removed | hub top3 | hub cut | dup schema | chars/ev | chars saved | new entrants graded | entrant excess |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_a | 5.7477 | 301 | — | 87 | — | — | 4 | — | 0 | 9079.0991 | — | 594 | 0.0522 |
| combined | 4.7568 | 224 | 25.6% | 80 | 8.1% | 0 | 0 | 100.0% | 0 | 7923.5405 | 12.7% | 37 | 0.1081 |
| gate | 4.7748 | 233 | 22.6% | 81 | 6.9% | 0 | 4 | 0.0% | 0 | 7972.1802 | 12.2% | 15 | 0.0667 |
| gate_alt_T | 3.8288 | 176 | 41.5% | 77 | 11.5% | 0 | 4 | 0.0% | 0 | 6632.3784 | 27.0% | 33 | 0.0606 |
| baseline_b | 5.7477 | 301 | — | 87 | — | — | 4 | — | 0 | 9079.0991 | — | 594 | 0.0522 |
| dedup | 5.7477 | 301 | 0.0% | 87 | 0.0% | 0 | 4 | 0.0% | 0 | 9079.0991 | 0.0% | 0 | None |
| demote | 5.7477 | 299 | 0.7% | 86 | 1.1% | 0 | 2 | 50.0% | 0 | 9057.5315 | 0.2% | 7 | 0.1429 |
| hub | 5.7477 | 293 | 2.7% | 89 | -2.3% | 0 | 2 | 50.0% | 0 | 9072.2613 | 0.1% | 18 | 0.1667 |

`baseline_a` is the baseline for gate, gate_alt_T and combined, and `baseline_b` for hub, demote and dedup.
The two are the same replay in separate processes. For a baseline row, the entrant columns show the quality of the whole delivered set.

### By recorded rank (irr cut / used lost vs baseline; n = baseline full slots)

| arm | 1 | 2 | 3 | 4-5 | 6-10 |
|---|---|---|---|---|---|
| combined | -4.3% (n=23) / 3.3% (n=30) | 0.0% (n=43) / 4.2% (n=24) | 10.2% (n=49) / 7.7% (n=13) | 34.6% (n=101) / 20.0% (n=15) | 44.7% (n=85) / 20.0% (n=5) |
| gate | 0.0% (n=23) / 3.3% (n=30) | 2.3% (n=43) / 4.2% (n=24) | 8.2% (n=49) / 7.7% (n=13) | 31.7% (n=101) / 13.3% (n=15) | 36.5% (n=85) / 20.0% (n=5) |
| gate_alt_T | 0.0% (n=23) / 3.3% (n=30) | 4.7% (n=43) / 4.2% (n=24) | 34.7% (n=49) / 15.4% (n=13) | 52.5% (n=101) / 26.7% (n=15) | 62.4% (n=85) / 40.0% (n=5) |
| dedup | 0.0% (n=23) / 0.0% (n=30) | 0.0% (n=43) / 0.0% (n=24) | 0.0% (n=49) / 0.0% (n=13) | 0.0% (n=101) / 0.0% (n=15) | 0.0% (n=85) / 0.0% (n=5) |
| demote | 0.0% (n=23) / 0.0% (n=30) | 0.0% (n=43) / 0.0% (n=24) | 0.0% (n=49) / 0.0% (n=13) | 1.0% (n=101) / 6.7% (n=15) | 1.2% (n=85) / 0.0% (n=5) |
| hub | -4.3% (n=23) / 0.0% (n=30) | 0.0% (n=43) / 0.0% (n=24) | 2.0% (n=49) / 0.0% (n=13) | 2.0% (n=101) / -13.3% (n=15) | 7.1% (n=85) / 0.0% (n=5) |

### Demotion strength census (applicable event × marked-node cases)

| window | events | cases | default m>0.9 | fc080_mw1 m>0.9 | fc080_mw1 m<=0.75 | fc080_mw1 median |
|---|---|---|---|---|---|---|
| trainval | 428 | 903 | 89.5% | 39.4% | 23.5% | 0.861 |
| holdout | 128 | 583 | 91.1% | 24.2% | 36.4% | 0.8165 |

## Reading the verdicts

Per the prereg rule, a valve is recommended for a host only if every line
that applies to it on that host passes.

**Gate (`LM_RECALL_MIN_SCORE`, drop).**

- sfx, T=0.35: the gate removes 36.9% of irrelevant full slots, loses 2.5% of
  used full slots (2 of 79), never removes rank 1, and cuts context by 23%
  (12.7k → 9.7k JSON chars per answer). G1–G3 pass. Q1 fails on its point
  estimate: the gate's 16 new entrants have excess 0.0 against 0.042 for the
  baseline's delivered set. At the baseline rate, 16 entrants would be
  expected to hold about 0.7 grounded hits, so this line cannot tell the arm
  from the baseline at this n. It is still a registered FAIL, and the
  recommendation follows it.
- alt, T=0.30: the gate removes 22.6% of irrelevant full slots, under the 25%
  line (G1 FAIL), and loses 6.9% of used slots. T=0.35, run as a reported arm
  only, would give 41.5% / 11.5%. That value was not the registered one: on eval
  it lost 12.2% of used slots.
- Drop against stub. At equal used loss on eval, drop saved 3.2× (sfx) and
  2.6× (alt) more context than stub. A stub entry still costs ~300–400 JSON
  chars, and drop rarely refills the slot: sfx holdout full slots per answer
  fell from 5.03 to 2.96.
- By rank, the gate cuts little in ranks 2–3 and most below them (sfx: 27% at
  rank 2, 37% at rank 3, 56% at ranks 4–5). Used-marked slots are sparse below
  rank 3 (n ≤ 9 per bucket on sfx), so used loss there has wide error. This
  is the position bias of the marks. The rank-1 bucket uses *recorded* rank.
  On alt, 1 of 30 recorded-rank-1 used slots is lost because the baseline
  replay itself puts a different node first. G3 is measured against the
  replayed rank 1 and is 0 on both hosts.

**Hub (`LM_HUB_SUPPRESSION_FACTOR=0.1`).**

- sfx: the valve cuts 35.9% of top-3 hub slots (39 → 25), under the 80% line
  (H1 FAIL). Used loss is 1.3% (H2 pass). Q1 fails as well: the 21 new
  entrants have excess −0.048 against the baseline's 0.042, again at a small n. The cause was found on eval, before
  the holdout run: the valve lifts a hub on any credit row after its first
  complaint, and `lookup` counts as credit. Of the 115 train-defined hubs, 39
  are lifted this way (all 39 by `lookup`, 13 also by `grounded`), so only 75
  are ever demoted. The factor does not matter: 0.1 and 0.03 give identical
  lists. The same point was made about marks in LM
  01M3MXJWTPZ2H2HJ3ZSPKZXGTE: lookup means "read", not "useful". Lifting only
  on `used`/explicit/`grounded` is the change to measure next. It is not in
  this goal's scope.
- alt: 4 → 2 hub top-3 slots, with no used loss. The effect is real but tiny,
  as expected: alt has 18 train hubs.

**Demotion strength (`LM_QUERY_IRRELEVANCE_FULL_COSINE=0.80`,
`LM_QUERY_IRRELEVANCE_MARK_WEIGHT=1.0`).**

- sfx holdout-window census: the share of applicable cases with multiplier
  > 0.9 falls from 76.9% to 23.1%, and the median multiplier from ~0.94 to
  0.77. D1 passes. On alt the share falls from 91.1% to 24.2%.
- On the live path the effect on delivery is small, because demotion only
  reorders what is already a candidate. sfx: irrelevant full slots −2.1% (5),
  used full slots −5.1% (4 of 79). alt: −0.7% / −1.1%.

**Schema dedup (`LM_RECALL_SCHEMA_DEDUP=1`).**

- sfx: duplicate schema slots 36 → 0 (S1 pass). The arm also loses 12.7% of
  used full slots (10 of 79). This is a property of the labels, not a lost
  procedure. In the sfx holdout, 24 `used` marks sit on schema nodes. For 15 of
  them a same-title twin is in the same recorded list, and for 11 that twin
  ranks earlier, so dedup keeps the twin and drops the marked copy. The
  procedure title is still delivered, under the other node id.
- alt: no duplicates, so the arm is the identity. S1 passes trivially.

**Combined.** sfx: irrelevant −46.2%, used −20.3%, hub top-3 −71.8%, context
−24%. The used loss is roughly the sum of dedup's twin artefact and the
gate's and demotion's losses. alt: irrelevant −25.6%, used −8.0%, hub top-3
4 → 0. The combined arm has no pass line of its own, only Q1: new-entrant
excess 0.0 on sfx at n=60 (fail) and 0.108 on alt at n=37 (pass).

**Caveats.**

- The holdout is the last 25% inside the frozen snapshot (sfx 167 events,
  alt 111). Events after the snapshot were not replayed.
- The counterfactual store keeps pre-cutoff rows updated after the cutoff,
  and today's retrieval weights. The baseline replay therefore differs from
  the recorded delivery (see `smoke/README.md`), which is why every number
  here is taken against the replayed baseline.
- Labels are accepted explicit marks. Nodes created at or after the cutoff
  are left out of the labelled metrics.
