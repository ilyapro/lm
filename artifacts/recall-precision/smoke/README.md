# Recall precision live-path replay

Per host, never merged. Labels: accepted explicit marks on the replayed event; marks are position-biased, see the rank breakdown.

## sfx — eval

- snapshot `074ee970243d`, cutoff 2026-09-28T09:21:40.000000Z, events 50/166 (sample=50, seed=7), train-defined hubs 115
- stripped at/after cutoff: {"connections": 4465, "query_anchor_edges": 2, "query_anchor_edges_of_late_anchors": 461, "query_anchors": 329, "query_irrelevance": 10, "query_irrelevance_of_late_anchors": 1129, "recall_credit_ledger": 158, "recall_explicit_credit": 318, "recall_feedback_marks": 1579, "recall_lookup_events": 250}
- pre-cutoff rows updated after cutoff (kept): {"connections": 6389, "query_anchor_edges": 23, "query_anchors": 3, "query_irrelevance": 5}
- runtime: total 30.4 s (prepare 4.36, replay 25.37, score 0.67); 0.5073 s per event×arm

| arm | full/ev | stub/ev | irr marked | irr full | irr top3 | used marked | used lost | rank1=recorded | hub top3 share | dup schema | entrants/ev | entrant excess | future filtered | labels excl. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 4.8800 | 0.0000 | 116 | 92 | 54 | 42 | 7 | 0.8200 | 0.1957 | 15 | 0.7800 | 0.0435 | 0 | 1 |

By recorded rank (irr removed share / used lost share):

| arm | 1 | 2 | 3 | 4-5 | 6-10 | 11+ |
|---|---|---|---|---|---|---|
| baseline | 0.0000 (n=15) / 0.0526 (n=19) | 0.0417 (n=24) / 0.0000 (n=6) | 0.2500 (n=20) / 0.2000 (n=5) | 0.2353 (n=34) / 0.1429 (n=7) | 0.4348 (n=23) / 0.8000 (n=5) | — |

## alt — eval

- snapshot `a887efc397ef`, cutoff 2026-09-28T07:13:51.000000Z, events 50/110 (sample=50, seed=7), train-defined hubs 18
- stripped at/after cutoff: {"connections": 3643, "query_anchor_edges": 0, "query_anchor_edges_of_late_anchors": 469, "query_anchors": 231, "query_irrelevance": 0, "query_irrelevance_of_late_anchors": 650, "recall_credit_ledger": 117, "recall_explicit_credit": 356, "recall_feedback_marks": 1100, "recall_lookup_events": 95}
- pre-cutoff rows updated after cutoff (kept): {"connections": 4127, "query_anchor_edges": 4, "query_anchors": 0, "query_irrelevance": 0}
- runtime: total 23.66 s (prepare 2.62, replay 20.28, score 0.77); 0.4055 s per event×arm

| arm | full/ev | stub/ev | irr marked | irr full | irr top3 | used marked | used lost | rank1=recorded | hub top3 share | dup schema | entrants/ev | entrant excess | future filtered | labels excl. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 3.8200 | 0.0000 | 80 | 59 | 33 | 58 | 21 | 0.7800 | 0.0315 | 0 | 1.3000 | 0.0156 | 0 | 49 |

By recorded rank (irr removed share / used lost share):

| arm | 1 | 2 | 3 | 4-5 | 6-10 | 11+ |
|---|---|---|---|---|---|---|
| baseline | 0.2500 (n=8) / 0.2000 (n=20) | 0.3571 (n=14) / 0.4737 (n=19) | 0.2143 (n=14) / 0.5000 (n=10) | 0.3077 (n=26) / 0.3333 (n=9) | 0.1667 (n=18) / — (n=0) | — |

## Smoke notes

- Command (per host, run 2026-09-29 on sfx, default embedding backend = production MiniLM):
  `python3 scripts/recall_precision_replay.py run --host {sfx,alt} --segment eval --sample 50 --out artifacts/recall-precision/smoke/{host}.json`,
  then `render`. Wall-clock: sfx 30.4 s, alt 23.7 s for 50 events × 1 arm (≈0.4–0.5 s per event×arm warm;
  the very first cold run on sfx took 2.05 s per event×arm while the model and chunk index loaded).
- The baseline replay does not reproduce the recorded lists exactly (rank 1 = recorded in 82% sfx / 78% alt;
  alt baseline already loses 21 of 58 used-marked slots vs the recorded delivery): the counterfactual store has
  post-cutoff anchors/edges/marks/credit stripped and retrieval weights, node stats and pre-cutoff rows updated
  after the cutoff (counted above) are as of the snapshot. Arm effects must therefore be read **against the
  replayed baseline arm** (`vs_baseline` block), not against the recorded delivery.
- Candidates created after the replayed event are dropped before ranking (`future_candidates_hidden` in the JSON),
  so `future filtered` is 0.
- `labels excl.` = marked nodes created at or after the cutoff, left out of labelled metrics (alt: 49 of 138
  marks on the sample — alt writes many nodes during the eval window).
