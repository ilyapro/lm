# Prereg: `LM_RECALL_SCHEMA_TRIGGER=name` on the holdout (goal schema-ranks-by-meaning)

Committed before any holdout run. The design was chosen on the dev segment and
on name queries against the old (2026-09-28 21:46Z) sfx snapshot; the holdout
below was not replayed before this file was committed.

## Data

- Snapshots taken 2026-09-29 07:13Z with the SQLite backup API: sfx from the
  live DB opened `mode=ro`, alt streamed over ssh into sfx (nothing written on
  alt, no sfx data sent to alt). sha256 in each run JSON.
- Events: `recall_precision_replay.window_events` (marks era from
  2026-09-27T14:45Z, non-empty, not A/B).
- **dev** = events before the recall-precision holdout start
  (`artifacts/recall-precision/split.json`: sfx 2026-09-28T15:23:18Z,
  alt 2026-09-28T14:00:58Z); counterfactual cutoff at that split's eval start.
- **holdout** = events from that holdout start up to 2026-09-29T07:00:00Z
  (this goal's own recalls start after it); counterfactual cutoff at the
  holdout start: marks, credits, anchors, edges learned from then on are
  stripped before replay, so no holdout mark shapes the ranking it is scored on.
- Labels: latest accepted `recall_feedback_marks` row per (event, node), only
  `level='schema'` nodes created before the cutoff.

## Arms (per host, never merged)

- `legacy`: the host's field env (sfx gate 0.35, alt 0.30, drop, hub 0.1,
  dedup, demotion 0.80/1.0, near-dup 0.97, credit policy).
- `name`: the same plus `LM_RECALL_SCHEMA_TRIGGER=name`.

## Lines (holdout, each host)

- **L1 (no raise without meaning).** Irrelevant-marked schemas the field found
  by trigger that are still delivered in full: `name` ≤ 0.5 × `legacy`; and the
  irrelevant share among delivered marked schemas is lower under `name`.
- **L2 (useful schemas kept).** Used-marked schemas delivered in full:
  `name` ≥ 0.5 × `legacy` (the goal's reference rule "vector ≥ 0.55 or
  bm25 ≥ 0.2" kept 53% of used).
- **L3 (text).** Schema content chars delivered (after `delivery` shaping):
  `name` < `legacy`.
- **L4 (latency).** Mean `memory_recall` latency: `name` ≤ `legacy`.
- **L5 (name queries).** On the fresh snapshot, 150 schemas with a trigger
  (seed 7), query = trigger, scope = schema scope, field env: rank-1 count
  under `name` ≥ `legacy`; among queries where `legacy` shows meaning agreement
  for the schema (vector ≥ 0.55 or bm25 ≥ 0.2), `name` puts it at rank 1 in
  every case.

alt: a line is evaluated only if `legacy` delivers ≥ 20 marked schemas in the
holdout; otherwise it is reported, not judged.

## Rule

The valve is recommended for a host when every evaluated line passes there.
