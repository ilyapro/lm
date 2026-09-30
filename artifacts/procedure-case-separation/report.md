# Procedures without case history

Base: merge-base `24e9009`. The change is in `src/living_memory/consolidation.py`
(procedural stage of ordinary consolidation plus `memory_teach`). The regressions
are in `tests/test_procedure_case_separation.py`. The replay is
`artifacts/procedure-case-separation/replay.py`, run against a private read-only
copy. This report contains only aggregate numbers, with no ids, texts, scopes or
project names.

## Causal mechanism

`_materialize_procedural_schemas` grouped every trace carrying a
`task_pattern`/`procedure_id`. `_procedure_steps` then turned **every** member into
a step: `_step_description` fell back to the whole `task: content`, and
`_format_schema_content` numbered those steps. A shared group key only shows
that the traces share a topic. In the live corpus copy, 5219 of the 5229 traces
that carry such a key declare no step at all (no `step_order`/`step`/
`step_description`/`step_content`). They are decision records, rejected
alternatives, execution reports, traps and whole prose recipes. As a result, the
histories of independent tasks, from many projects and eras, were delivered as a
binding numbered instruction list (up to 87 518 characters in one schema).

The unmarked records are not all case histories, though. Some are whole
conditional recipes written in one record (supervisor #1325). The fix therefore
does not claim that unmarked means useless. It only says that an unmarked record
is not a *step of a group procedure*. Such a record stays an ordinary trace that
recall and lookup deliver as it is.

A second, independent defect was verified: `memory_teach` copied the original's
whole context into the corrective trace, `step_description` included. The step
would therefore re-render the corrected-away text after a teach.

## Change (one root landing, existing model only)

- A group yields a schema only from members whose writer declared a step. The
  existing floor `PROCEDURAL_MIN_CLUSTER_SIZE` now counts declared steps
  instead of topic members. The threshold value, the era assessment, the group
  identity and the schema format are unchanged.
- A later record at the same declared position restates that step (the earlier
  text stays in its own trace). Steps declared by text alone follow, in write
  order. The `task:` prefix fallback is removed.
- A group that reaches the floor with records but not with declared steps soft-deletes
  (`soft_delete_node`, reason `procedural: no declared steps`) the active
  uncorrected schema that earlier passes built for that group. The node, its
  content, its provenance and its edges are kept. Lookup returns it with
  `decayed=true`. This happens in the ordinary pass: no cleanup job, no flag and
  no manual DB edit.
- `memory_teach` no longer inherits `step_description`/`step_content`. The
  correction keeps the step position and replaces the text.
- `_find_existing_schema` takes the group id (the only field it used), and the
  parallel `keys_by_group` bookkeeping is gone.

Not changed: ranking, thresholds, triggers/name valve, delivery shaping,
models/tiers/providers, operator settings, the remember/recall paths (no new
synchronous model calls), and the storage schema.

## Oracle (fixed before the implementation: commit `6b7cc2d`)

For each group the oracle states the exact ordered step texts that an active schema
must carry. An empty list means no active schema may exist. The dev groups are:
ordered, conditional with an important last step written out of order,
independent case history, a decision-log group whose records share one
`step_description`, one instruction restated at one position, a taught step plus a
later restatement, and two different group keys that share one trigger. After
supervisor #1325, a whole conditional **prose recipe without step metadata**,
later corrected by teach, was added. Its oracle is ordinary consolidation, then
a recall phrased the way a user asks, then the lookup the delivery points to.
It must yield the current recipe complete, with every condition and the last
step.

Negative controls (all asserted to fail the oracle):
- The legacy concatenation fails on every group except the two fully declared
  ones it already handled (these are marked `legacy_ok`).
- A schema without its last conditional step fails, and so does a schema with
  every step cut to 60 characters.
- Dropping every unmarked instruction fails the recipe oracle.

Fixture numbers (design data; see the next section):

| set | groups | expected steps | extra steps, legacy | extra steps, new | step chars legacy → new |
|---|---|---|---|---|---|
| dev + shared trigger | 8 | 15 | 18 | 0 | 2182 → 719 |
| "holdout" fixture | 3 | 6 | 5 | 0 | 531 → 261 |

All expected steps, conditions, orders and applicable corrections are kept
(exact equality). Lifecycle checks in the same module cover the following:
- consolidate → recall → lookup;
- a second pass creates nothing and changes nothing;
- a new case adds no step, and a new declared step extends the *same* schema id;
- teach on a described step replaces its text;
- a pre-existing oversized case-history schema is retired by an ordinary pass,
  while its sources stay active, identical and recallable;
- there is never more than one active schema per group key.

**Honesty about the holdout.** The fixture "holdout" groups (string `step`,
`step_content`, other case shapes) live next to the implementation and ran during
development. They are **design/regression data**, not evidence of
generalisation. The only check drawn after the candidate was fixed is the
replay below, and nothing was tuned on it. The replay script gained one extra
metric (lookup by procedure name) after its first run. The code under test and
the evaluation sets did not change.

## Replay on the private copy (local, after the candidate was fixed)

Method: the copy was taken read-only with the SQLite backup API from the live
DB. `before` is the copy as it is, the state the replaced mechanism produced.
`after` is a second copy after the procedural stage of an ordinary pass ran
twice per scope (15 scopes) with the new code. The evaluation sets were drawn
with seed 20260930 from the pristine copy:
- (a) 40 readable triggers of active procedural schemas, queried as "how to
  <trigger>";
- (b) 23 active traces that carry a group key and no step metadata and whose
  text opens as a procedure/recipe (an evaluation-only selector, not used in
  production), queried as "how to <their procedure name>".

Recall ran through the MCP tool functions with the operator's delivery
environment. "Forced lookup" re-fetches every delivered schema that arrived
non-full (a binding schema must be read whole) and the target recipe when it
arrived as a snippet.

| census | before | after |
|---|---|---|
| active procedural schemas | 483 | 3 |
| schema characters | 1 816 902 | 5 522 |
| largest schema | 87 518 | 3 339 |
| steps | 2 844 | 7 |
| unconfirmed steps (text not from a declared step of its sources) | 2 842 (99.9 %) | 3 |
| same, in groups an ordinary pass visits | 2 839 | 0 |
| max active schemas per group | 1 | 1 |
| triggers shared by several groups | 19 | 0 |

Pass 1 retired 480 schemas and updated 2 in place. Pass 2 created 0 and retired 0:
it rewrote the same 2 schemas in place and left the census identical, so it is
idempotent.

| recall | schema topics before | after | prose recipes before | after |
|---|---|---|---|---|
| queries | 40 | 40 | 23 | 23 |
| delivered + forced lookup chars, total | 1 584 853 | 455 540 | 552 604 | 247 123 |
| same, median per query | 31 912 | 11 134 | 21 202 | 12 267 |
| forced lookup chars | 574 624 | 0 | 250 073 | 24 986 |
| schemas delivered | 133 | 3 | 34 | 1 |
| unconfirmed steps delivered | 1 281 | 6 | 210 | 0 |
| same, from groups an ordinary pass visits | 1 275 | 0 | 210 | 0 |
| recipe obtained whole by recall(+lookup) | – | – | 14 | 12 |
| of which from the recipe's own trace | – | – | 12 | 12 |
| recipe returned by lookup on its procedure name | – | – | 23 | 23 |
| recall latency median, ms | 228 | 201 | 229 | 215 |

Procedural stage latency on a fresh copy (3 consecutive passes, seconds): old
3.48 / 2.69 / 2.55, new 1.97 / 0.34 / 0.32. The first new pass includes the 480
retirements.

### What the replay does *not* show, and the regressions it does show

- **Two recipes are no longer obtained by recall, and the change does not
  hide them.** In 2 of the 23 sampled prose recipes, the recipe text reached
  the user before only as one "step" inside a large case-history schema of the
  same group (`lost_carrier.schema_only = 2`, `own_trace = 0`). The recipe's own
  trace is kept out of recall identically before and after by a live ranking
  valve that acts on that one trace. The replay's diagnostic turns each valve off
  in its own process only, and then the own trace is recalled: in one case the
  hub valve (`LM_HUB_SUPPRESSION_FACTOR`: agents marked the record irrelevant
  for at least three distinct questions and never credited it afterwards), in
  the other the score gate (`LM_RECALL_MIN_SCORE`). One of the two is a
  still-current self-contained recipe. The other is a note that a step no longer
  applies. Both stay whole and reachable by `memory_lookup` on their procedure
  name (23/23 before and after). The case-history schema delivered them past
  their own demotion only because it bundled them with unrelated histories.
  Getting that back would mean keeping the defect or retuning live valves, and
  the goal forbids both.
- **One legacy schema remains**: 2 073 characters and 3 unconfirmed steps. Its
  group has no active member left, so an ordinary pass never visits that group
  and does not retire it. Retiring such orphans needs a sweep over schemas
  rather than groups. That was left out to keep the production size from growing.
- The 6 unconfirmed steps still delivered for schema topics all come from that
  orphan schema.
- Fewer schemas is not claimed as a benefit in itself. The benefit claimed is
  0 unconfirmed steps from active groups, exact preservation of the declared
  procedures, and less delivered+lookup volume at unchanged recall latency.
- Queries are derived from procedure names, a proxy for natural phrasing. The
  sample is 40 + 23 from one corpus.

### Holdout threshold (P5), as measured

The holdout is the seeded replay sample above (40 schema topics and 23 prose
recipes, drawn from the copy after the candidate was fixed). The development
sets are the fixture groups. There is no train set. The threshold and results:

- Unconfirmed steps delivered from groups that an ordinary pass visits: **0**
  (before: 1 275 for schema topics, 210 for recipes). The 6 that remain come
  from the one orphan schema whose group has no active member.
- No recipe is lost to recall through its own record. Recipes obtained from their own
  trace number 12 before and 12 after (same queries, same valves). Every sampled
  recipe is returned whole by lookup on its procedure name (23 → 23). Only
  losses of the case-history carrier remain (2), each tied to a live valve as
  described above. They count as defect removal, not as a regression the change
  introduced.

The copy was taken again on the day of this report. The census and the recall
counts equal the first run. Latency medians differ by host load (±30 ms).

## Fixture and replay vs live effect

Everything above comes from synthetic fixtures or a local replay on a private copy.
Nothing here was measured on the live server. The live effect appears only after
the code is delivered through the normal live-code path and an ordinary
consolidation pass runs there. Landing this change is not evidence of benefit.

## Tests whose oracle was replaced

- `tests/test_mcp_server.py::test_memory_remember_signature_unchanged_and_consolidate_report_stays_full`
  and `tests/test_edge_derivation.py::test_consolidation_annotates_existing_schema_edges_without_new_rows`
  built their schema from three undeclared records ("rehearsal N", "run N").
  That is exactly the case-history promotion that was removed. Their fixtures now
  declare steps. The contracts they pin (full node dicts in the consolidate
  report; R5a edge annotation) are unchanged.
- No procedural, incremental, dedup, era or recall contract was weakened. The
  target suites (`test_procedure_case_separation`, `test_schema_distillation`,
  `test_procedural_schemas`, `test_consolidation_no_copies`) pass.

## Gate and size

- Full suite (`scripts/test.sh tests`, hash embedding backend): exit 0, no
  failures, some skips (at the commit that fixed the edge-derivation fixture;
  later commits touch only this artifact directory).
- Production against merge-base: `src/living_memory/consolidation.py`
  +39 / −41 lines (net −2). Nothing was compacted, no existing explanation was
  removed, and nothing moved outside `src/`. The replaced mechanism removed is the
  whole-trace step fallback, the unordered sentinel and the parallel key
  bookkeeping.
