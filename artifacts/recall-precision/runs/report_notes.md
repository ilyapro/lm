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
