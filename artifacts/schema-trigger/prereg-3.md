# Prereg 3: one relevance score for order and gate (goal useful-schema-survives-ranking, reopened)

Committed before the code change of the reopened goal and before any replay of
the fresh holdout below. The measurement rules of `prereg-2.md` (procedure-events,
duplicates, arms, lines L1–L5 and N, the 20-event judging floor) stand unchanged.
The bar is unchanged too: L2 ≥ 0.5 × `legacy` on **both** sfx data sets of
`prereg-2.md`, together with L1, L3, L4 and L5.

## Status of the data

- The **original corpus** and the 07:00–15:00Z window (`new` in `prereg-2.md`)
  have both been looked at: `report-2.md` lists the lost procedures of each.
  They are **design data** now. The 07:00–15:00Z window is called **seen
  holdout** from here on. The acceptance check still reads it from
  `v2/new-sfx.json`, and it still has to pass there.
- **fresh holdout** = window events in `[2026-09-29T15:00:00Z, 2026-09-29T16:26:00Z)`,
  counterfactual cutoff 15:00Z. 16:26Z is when this attempt's own recall traffic began.
  Snapshot: sfx 2026-09-29T16:29:01Z, sha256 `4b6380d3…`. Size before replay:
  53 events, 16 used / 71 irrelevant marked procedure-events (raw). It carries
  the earlier attempt's own recalls from 15:00Z to about 16:05Z, as field
  traffic. With fewer than 20 used procedure-events it can only be **reported**
  for L2. L1 is judged only if `legacy` delivers ≥ 20 irrelevant
  procedure-events. It is not a win on its own.
- Every replay of this attempt, the original corpus and the seen holdout
  included, runs on the 16:29Z snapshot. `legacy` is recomputed there, and the
  ratios are taken against that recomputed `legacy`.

## Cause (found on design data only)

The ranked list around a lost used procedure was dumped with both scores,
the ranker's and the gate's. In the "slot" losses of `report-2.md`, the schema
**passes** the gate: gate 0.39–0.46, above the threshold 0.35. It still ranks
below `max_results` other passing results. Those results score higher on the
ranker but lower on the gate. The ranker blends the channels with the store's
learned per-scope weights. The gate re-blends the same channel scores with the
fixed `score_gate.REFERENCE_WEIGHTS`. In `project:lm`, for example, the learned
weights put traces with only vector 0.57 (gate 0.38) at ranker 0.87. A schema
with vector 0.52 and graph 0.32 (gate 0.46) lands at 0.67, 23rd. So two
criteria of relevance decide one delivery. One of them depends on the
project the node lives in.

## Hypothesis H1 (the candidate)

Under `LM_RECALL_SCHEMA_TRIGGER=name` only, `rank_candidates` blends every
candidate with `REFERENCE_WEIGHTS` in place of the per-scope weights. The
order and the gate then read the same score. It is the same for every node,
whatever its level, origin or project. Nothing is schema-specific. Without
the valve nothing changes.

Offline simulation on the captured name-mode ranked lists (16:29Z snapshot,
reproducing the `name` arm of `v2` exactly) predicts:

| | seen holdout | original corpus |
|---|---|---|
| used proc delivered, `name` now → H1 (legacy in v2) | 10 → 12 (21) | 19 → 20 (31) |
| irrelevant proc delivered | 6 → 5 (32) | 23 → 24 (93) |
| non-schema used / irrelevant marked delivered | 59/158 → 57/136 | 107/314 → 105/268 |

The non-schema counts are a **negative check**. They are reported, not judged:
the order change also moves traces.

## Plan of measurement

1. `schema_trigger_replay.py run` for the original corpus and the seen
   holdout: `legacy` and `name` on the H1 code. Output goes to
   `v2/original-sfx.json` and `v2/new-sfx.json`. The acceptance check reads
   these. The replay outputs stay local and are not committed (private).
2. `run` for the fresh holdout (reported).
3. `names` for names dev and names holdout (L5).
4. N: the list of delivered-list differences between `name`@5804d83 and H1
   on the original corpus, and the all-level used / irrelevant counts of both.
