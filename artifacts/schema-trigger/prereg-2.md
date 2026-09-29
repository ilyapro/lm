# Prereg 2: `LM_RECALL_SCHEMA_TRIGGER=name` back to the quality bar (goal useful-schema-survives-ranking)

Committed before any code change of this goal and before any replay of the new
holdout below. It fixes the measurement rules. The design is chosen only on the
original corpus: the `dev` and `holdout` segments of `prereg.md` and the 150
L5 names sampled there.

## Data

- Snapshots taken 2026-09-29 15:21Z with the SQLite backup API
  (`recall_precision_replay.py snapshot`): sfx from the live DB `mode=ro`
  (sha256 `5a3b1f8a…`), alt streamed over ssh (sha256 `3ddc4867…`). Nothing
  from sfx goes to alt, and no query text is committed.
- **original corpus** = the `holdout` segment of `prereg.md` (recall-precision
  holdout start .. 2026-09-29T07:00Z, counterfactual cutoff at that start),
  replayed on the new snapshot. It was the design data for this goal.
- **new holdout** = window events (`recall_precision_replay.window_events`)
  in `[2026-09-29T07:00:00Z, 2026-09-29T15:00:00Z)`. The counterfactual
  cutoff is 07:00Z, so no mark from the new window shapes the ranking it is
  scored on. Size before replay: sfx 176 events (61 used / 169 irrelevant
  schema marks, raw), alt 17 events (0 used / 2 irrelevant).
- **names dev** = the schemas of the 150-query L5 sample in
  `names-{sfx,alt}.json` that are still active. **names holdout** = every other
  active schema with a trigger in the new snapshot, all of them, disjoint from
  names dev by schema id.

## Duplicates

The supervisor marks several copies of one procedure in the same answer. A
procedure is the schema title (`schema_dedup.schema_title`, `Procedure: <trigger>`).
**Procedure-events** are distinct `(event, title)` pairs among the marked
schemas of one mark (used or irrelevant) created before the cutoff. A
procedure-event is **delivered** when any schema with that title is delivered
in full, whether or not that copy was the one marked. Node-level counts (every
marked copy counted, as in `report.md`) are reported alongside and not judged.

## Arms (per host, never merged)

- `legacy`: the host field env of `schema_trigger_replay.FIELD_ENV`.
- `name`: the same plus `LM_RECALL_SCHEMA_TRIGGER=name`, on this goal's code.
- `name@9034741` (reference only): `name` on the code before this goal, for the
  original corpus and names dev. It shows what this goal changed.

## Lines (each host, each data set: original corpus and new holdout separately)

- **L1**: irrelevant procedure-events delivered: `name` ≤ 0.5 × `legacy`; and
  the irrelevant share among delivered marked procedure-events is lower under
  `name`.
- **L2**: used procedure-events delivered: `name` ≥ 0.5 × `legacy`.
- **L3**: schema content chars delivered, `name` < `legacy`.
- **L4**: mean recall latency, `name` ≤ `legacy` + 5% (the noise between the
  two arms of one process on this machine).
- **L5** (names dev and names holdout separately): query = trigger, scope =
  schema scope, field env, top 5. Rank-1 count `name` ≥ `legacy`, and among
  queries where `legacy` shows meaning agreement (vector ≥ 0.55 or bm25 ≥ 0.2)
  `name` puts the procedure at rank 1 in every case.
- **N (negative)**: on the original corpus, field events whose delivered list
  differs between `name@9034741` and `name` are counted and listed by kind
  (exact-name query or not). The goal's change targets exact-name queries.
  Any other difference is reported and explained.

alt: a line is judged only if `legacy` delivers ≥ 20 marked procedure-events
of that mark (L1/L2), or ≥ 20 agreement queries (L5). Otherwise it is reported
and not judged. With 17 events, the new alt holdout is expected to be reported only.

## Rule

The goal meets the bar on sfx when every judged line passes on both the
original corpus and the new holdout. A pass on a data set with fewer than 20
marked procedure-events of the judged mark is reported as "too small", not
as a win.
