# Recall: a schema is ranked by meaning, not by trigger words

Goal `schema-ranks-by-meaning`. Code: `src/living_memory/retrieval.py`
(`schema_trigger_by_name`, `MemoryRecallService._named_schemas_first`),
`src/living_memory/score_gate.py` (`trigger_gate_score`),
`src/living_memory/storage.py` (`MemoryStore.schema_triggers`).
Measurement: `scripts/schema_trigger_replay.py`, `artifacts/schema-trigger/`.

## Why

Before this change, a schema whose `context.trigger` shared at least half its
words with the query got three things of its own, whatever the query meant:

1. a near-constant base score, `0.95 + 0.05 * overlap`;
2. a `1.8x` multiplier on the final score;
3. its own scale in the quality gate (`trigger_gate_score` ≥ 0.5), which a
   clean word match always passes.

On the field marks (sfx, from 2026-09-27) these schemas were 72% `irrelevant`,
no better than the rest of the delivery, yet they took the top slots and
shipped their full text. The overlap geometry did not separate useful from
useless; agreement of the meaning channels (vector, bm25) did.

## Valve

`LM_RECALL_SCHEMA_TRIGGER=name` (case-insensitive). Any other value, or unset,
uses the ordinary trigger channel. Ordinary recall keeps the half-of-trigger
overlap requirement, including for a short instruction embedded in a compound
question. A trigger adds a candidate and can admit it across scopes. Its rank
contribution now follows the content's vector similarity, with the existing
graph weight providing a small discovery floor. The unconditional 1.8x rank
multiplier is gone. The separate trigger gate still keeps applicable instructions
available when their content has little lexical or vector support; feedback,
irrelevance demotion, and supersedes still apply. No scope is removed from
search. The `name` valve remains disabled by default.

Under `name`:

- The trigger finds nothing and scores nothing. A schema enters the recall
  only through bm25/vector/graph, is scored by the same blend as every other
  node, and passes the quality gate (`LM_RECALL_MIN_SCORE`) on that score.
  There is no trigger scale in the gate.
- Every node, schema or not, is ranked on the gate's own score
  (`score_gate.gate_score`). The weights are the fixed `REFERENCE_WEIGHTS`,
  not the store's learned per-scope weights, and there is no 1.2x prior for
  corrections. Order and gate therefore read one score, the same for every
  node whatever its origin or project. A result that passes the gate is no
  longer outranked by one that the gate scores lower. Before this, a useful
  procedure at gate 0.46 lost its slot to traces at gate 0.38 that
  `project:lm`'s vector-heavy weights lifted. Corrections still outrank what
  they correct (`_enforce_correction_dominance`).
- A query that **is** the procedure's name (its token set equals the trigger's
  token set, so `lm recall map curtail`, `lm_recall_map_curtail` and
  `LM recall-map curtail` all name the same schema) is that schema's best
  lexical match: the whole query is its title, so its bm25 is 1.0 (the
  channel's rank 1) instead of wherever word frequencies over longer texts put
  it. It moves to rank 1 if it passes the quality gate on that score; the
  query's demotion, feedback and the threshold still apply. Rank 1 always
  ships in full.
  With the gate off it moves to rank 1 unconditionally. Only schemas the
  meaning channels already ranked can be named; a named schema is marked with
  `trigger_score = 1.0` and `"trigger"` in `methods` for the wire.
- The trigger scan reads two columns (`id`, `context.trigger`) instead of up
  to 1000 whole schema nodes per scope, and only after ranking.

What is left of the trigger once the valve is the only path: the name match,
ordered by the same gate as everything else. The constant score, the
multiplier and the gate scale are gone.

## Ordinary recall after the applicability repair

The ordinary trigger path remains enabled. Trigger collection still requires
half of a saved trigger's terms, including for a short instruction within a
compound question. Ranking now also records how much of the *query* those
matched terms cover after tokenization removes procedural filler. A carrier's
trigger contributes through the existing blend only in proportion to the
square of that coverage when it cites procedural source records; an arbitrary
concept with only a trigger gets no new rank claim. Its vector, bm25, graph,
scope, feedback and correction evidence keep their existing roles. The score gate uses the same
coverage for nonbinding carriers. Binding schemas keep their trigger
availability scale, so an applicable short instruction can survive a
compound query even without a strong content vector. The `name` valve stays
disabled by default.

The square makes an unrelated single-label match weak in a compound query,
while a preserved legacy carrier whose trigger describes the whole query can
still reach the result slots and offer its full current group evidence through
`content_ref`. This does not treat case records as binding steps. See
`artifacts/recall-applicability/candidate-v3.md` for the synthetic retention
checks and the frozen development comparison.

## Measurement

`artifacts/schema-trigger/prereg.md` (lines, committed before the holdout run),
`artifacts/schema-trigger/report.md` (numbers and verdicts per host),
`holdout-{sfx,alt}.json`, `names-{sfx,alt}.json`. Goal
`useful-schema-survives-ranking`: `prereg-2.md` (committed before the change),
`report-2.md`, `v2/` — counts per procedure (duplicates of one procedure
counted once), a new holdout, names holdout, and where each lost useful
procedure is lost. Reopened: `prereg-3.md` (committed before the change),
`report-3.md` — one score for order and gate, a fresh holdout.

## Rollout

Per the project's order: the code lands with the valve unset (no behaviour
change); the operator adds the line to the host's `EnvironmentFile` and
restarts. Holdout verdict (report): on sfx the irrelevant, text and latency
lines pass but the pre-registered "keep ≥ 50% of used schemas" line fails at
46% (and one of 109 agreeing name queries fails the gate), so the rule does not
recommend it; alt's holdout is too small to judge. After the bm25-by-name
change (report-2): every agreeing name query is at rank 1 on sfx (dev 80/80,
holdout 219/219) and alt (47/47); per procedure the original corpus keeps 61%
of used, but the new sfx holdout keeps 10 of 21 (47.6%), one short of the
line. After order and gate were put on one score (report-3), every judged
line passes on sfx, on both data sets: used procedures 20/31 and 12/21,
irrelevant 93 → 24 and 32 → 5, schema text −78% / −86%, latency not higher,
names 80/80 and 218/218. On alt, names 47/47 pass, and the field sets are too
small to judge. The value, for the operator to switch on per host:

```
LM_RECALL_SCHEMA_TRIGGER=name
```

## Removing the valve and the legacy path

Remove once, on each host that runs `name`, at least 7 days of field marks
after enabling show both of these on `recall_feedback_marks` joined to
`recall_events.results`:

- the `irrelevant` share of delivered schemas is below the legacy share
  (72% on sfx before the change), and
- no regression in named lookups: every event whose query is a schema's name
  and whose schema was ranked has that schema at rank 1.

Then delete, in one commit: `SCHEMA_TRIGGER_OVERLAP_THRESHOLD`,
`SCHEMA_TRIGGER_BASE_SCORE`, `SCHEMA_TRIGGER_BOOST`,
`MemoryRecallService._collect_schema_triggers`, the two `trigger_score`
branches in `_blend_candidate_score`, the `trigger_score` branch of
`_cross_scope_admissible`, `score_gate.trigger_gate_score` with
`SCHEMA_TRIGGER_GATE_MIN`/`SCHEMA_TRIGGER_SCALE`, `SCHEMA_TRIGGER_ENV` and
`schema_trigger_by_name` (call `_named_schemas_first` unconditionally), and the
legacy trigger tests in `tests/test_recall_score_gate.py` and
`tests/test_retrieval_cross_scope_gate.py`.
