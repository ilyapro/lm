# Procedures without case history

Base: merge-base `24e9009`. The rollback `8289d37` reverted the first landing
`7903a82` on master; this branch replays that landing and builds on it.
**Final candidate: production `c25a6cd`** (later commits touch only protocol,
test comments and artifacts). Production code:
`src/living_memory/{consolidation,schema_dedup,storage,retrieval,score_gate}.py`.
Oracle: `tests/test_procedure_case_separation.py`. Replay: frozen
`fulltext-regression-protocol.md`, runner `fulltext_run.py`, reader
`replay.read`; aggregates in `fulltext-regression-result.json`. All replays ran
on private read-only copies; this report holds aggregate numbers only (no ids,
texts, scopes or project names).

## Status in one paragraph

A procedural group without writer-declared steps no longer yields a binding
schema: its node becomes a nonbinding **group carrier** that holds each current
record **whole** as evidence (never as steps), in place, with the same id,
edges and trigger. Recall's live dedup stage identifies schemas and carriers
by title, as the base did for schemas. On fresh copies of the pristine sfx
snapshot, with a reader that sees only the query and the real output, the
final candidate has **0 new losses** of full current instructions/corrections
on the design set, holdout 2 and holdout 3 (baseline misses counted apart),
**0 unconfirmed binding steps** in the census (orphans included) and in
deliveries, one node per group and an idempotent second pass. The price,
against `24e9009`: stored procedural text **+33%** (1.82M → 2.42M chars),
read text (recall + forced lookups, all sets) **+11%**, production lines
**+111/−105, net +6** — over the original net ≤ 0 budget. Only 1 of 10
required independent holdout items is eligible. The operator accepted this
candidate with both limits disclosed; the measured sets establish regressions
on examined data, not independent validation or a general benefit.

## Causal mechanism

`_materialize_procedural_schemas` grouped every trace carrying a
`task_pattern`/`procedure_id`, and `_procedure_steps` turned **every** member
into a step: `_step_description` fell back to the whole `task: content`, and
`_format_schema_content` numbered them. In the sfx corpus copy 5219 of 5229
keyed traces declare no step (decisions, rejected alternatives, reports, prose
recipes); histories of independent tasks were delivered as a binding numbered
list of up to 87 518 characters (2842 of 2844 census steps unconfirmed).
`memory_teach` also copied `step_description` into the correction, so the old
text re-rendered.

Such a schema was also the only **search surface** of some current records:
its trigger held the group's name and its body held every record whole, so
recall reached records that the hub valve or the score gate keeps out on their
own. Everything tried before the final candidate lost that surface in one way
or another (history below): retiring the schema; a carrier holding
600-character leads (it forced lookups and ranked the wrong same-title carrier
first); group-identity dedup (ten same-trigger carriers crowded out records the
original delivered directly); and a pass that relabels the node. Searchable
content and the authority to prescribe are separate properties: the fix keeps
the first and removes the second.

## Change (final candidate)

Consolidation (one ordinary path; no cleanup job, flag, index, model call):
- Only writer-declared steps (`step_order`/`step`/`step_description`/
  `step_content`) make a binding schema; one declared position holds its
  latest restatement; the whole-trace fallback is gone; teach drops inherited
  step text.
- Every procedural group has at most one live node. With ≥3 declared steps it
  is a `schema` of those steps; otherwise a `concept` carrier: context keeps
  `trigger`/`procedure_key` and no `procedure`; content
  `Evidence: <trigger> (case records, not steps)` and `- <id>: <record>` per
  current record (superseded and decayed left out), each record **whole**.
- A group's node changes form **in place** (`update_node(level=...)`). A legacy
  case-history schema, visited or orphan, becomes its group's carrier; one with
  no current record left is retired; a later duplicate node is retired; an
  orphan never adopts another group's records.
- A group's node keeps the trigger it is found by while a current record still
  carries it (the pass used to relabel with the records' majority
  procedure_id).

Recall: `schema_dedup` collapses by title (base rule), now also for carriers
(`Evidence: <trigger> ...`), so same-trigger carriers share one slot as the
legacy schemas did. The existing trigger rule, blend and gate scale apply to
any non-trace node the trigger channel found. Storage: `update_node` takes
`level`; `schema_triggers` lists schemas and carriers.

Not changed: live valves (`LM_HUB_SUPPRESSION_FACTOR`, `LM_RECALL_MIN_SCORE`,
`LM_RECALL_SCHEMA_TRIGGER`, `LM_RECALL_SCHEMA_DEDUP` — on/off as set), weights,
thresholds, constants, `max_results`, models/tiers/providers, operator
settings, DDL. No live DB edited; no code delivered to a live host.

## Oracles

`tests/test_procedure_case_separation.py` (20 tests) and
`tests/test_recall_schema_dedup.py`. The test reader (`obtained_texts`) is the
replay's policy: every delivery, re-fetched whole when cut; no target id, no
expansion of the ids a carrier lists.
- declared procedure, conditional last step, corrections; negative controls
  (legacy concatenation, cut last step, truncated steps, dropped instructions)
  fail as required;
- orphan repair, no adoption of another group's records, duplicate retirement,
  in-place identity (mutant rebuilding under a new id is caught), taught recipe
  arriving whole through its converted schema, older-era record kept;
- carrier loss: dropping or retiring the carrier loses the recipe (mutants);
  **new** `test_carrier_loss_mutant_holding_record_leads_loses_a_long_recipe` —
  a carrier with 600-character leads loses a longer recipe;
- **new** crowding counterexample of the group-identity loss:
  `test_same_trigger_carriers_do_not_crowd_out_a_record_recall_delivers_directly`
  (ten same-trigger groups and a lone recipe) with its group-identity mutant;
- **new** relabel regression `test_group_node_keeps_the_trigger_it_is_found_by`
  with its mutant.
The three new positive oracles fail on `e3cb245`.

Replaced oracles, with reason: the group-slot tests of `e3cb245` (one slot per
group) are replaced by the crowding oracle, because that rule is what crowded
records out; the dedup unit tests are the base title tests plus a carrier
title case. The retired-state test was replaced because that state was never
created: the store only soft-deletes (`decayed` + `decay_reason`; no
`DELETE FROM nodes`), so a schema retired by any first-landing pass would still
carry `procedural: no declared steps`, and the complete read-only backups of
both live stores taken after the revert hold **no** node with any
`procedural` decay reason.

## End-to-end results (final `c25a6cd`)

Protocol frozen before the run (`09c92c5`). Original = `24e9009` on untouched
fresh copies of the pristine snapshot; candidate = one fresh copy, two ordinary
procedural passes, census, one non-matching warm-up recall per searched scope,
then one copy per set. Same queries (`how to <name>`, `max_results=10`,
operator environment and live valves), same items. Reader (both sides): every
delivered result, re-fetched whole by id when delivered cut; nothing else.
All delivered and forced lookup response characters, with metadata, count.
Under this reader the original obtains exactly the items it obtained under the
old one (recipes 14 = 14 itemwise, holdout-2 items 181 = 181); only its lookup
volume differs. **These sets were examined during repair: regression data, not
independent validation.**

| | original `24e9009` | **final `c25a6cd`** |
|---|---|---|
| design recipes obtained whole, of 23 | 14 | **17** |
| — itemwise kept / **new loss** / gained / baseline miss | – | 14 / **0** / 3 / 6 |
| — obtained from their own record | 12 | 12 |
| H2 instructions/corrections kept / **new loss** / gained / miss | – | 42 / **0** / 0 / 10 |
| H2 case records kept / lost / gained / miss | – | 139 / 0 / 0 / 77 |
| H3 kept / **new loss** / gained / miss | – | 15 / **0** / 0 / 24 |
| unconfirmed steps delivered (design topics / H2) | 1 281 / 594 | **0 / 0** |
| schemas delivered (recipes / topics / H3) | 34 / 133 / 28 | 1 / 1 / 1 |
| read chars, design recipes (of which forced) | 960 384 (656 458) | 1 282 279 (692 984) |
| read chars, design topics (of which forced) | 2 084 169 (1 077 816) | 2 381 834 (1 100 353) |
| read chars, H2 (of which forced) | 1 771 029 (1 067 096) | 1 820 820 (982 136) |
| read chars, H3 | 1 325 887 | 1 344 877 |
| **read chars, all sets** | **6 141 469** | **6 829 810 (+11.2%)** |
| recall median ms, recipes / topics | 1 447 / 1 541 | 567 / 410 |
| H3 collateral queries, same set / same order, of 60 | – | 47 / 42 |

Census after two ordinary passes: 2 binding schemas (4 declared steps), 489
carriers, **0 unconfirmed binding steps** (orphans included), **max 1 active
node per group**, second pass creates 0 nodes (first: 9 created, 481 legacy
schemas converted in place). Stored procedural nodes: 483 → 491; content
**1 816 902 → 2 419 575 chars (+602 673, +33%)**; chunk rows 7 080 → 9 757
(10.9 → 15.0 MB of vectors). Mechanism of the growth: a carrier holds every
current record of its group whole, while a legacy schema held the era-current
subset of its last visit; 9 groups gained a carrier; each line carries its id.
No record text is duplicated across carriers (one node per group), and no
bucket-wide or peer reading is involved.

First run of this protocol, on `7cb9436` (full text + title dedup, before the
relabel fix): design 14 / 0 / 3 / 6, H3 15 / 0, but **H2 40 kept / 2 new
instruction losses** (and 9 case records). Both losses belonged to one group:
the pass had relabeled its converted schema with the records' majority
procedure_id, and the query matched the old trigger; the original delivered
that schema first. `c25a6cd` keeps the label; the fix was measured once more
under the unchanged protocol (amendment `325f8a7`).

### Holdout 6 — independent set: not available

`holdout6-protocol.md` was frozen before new snapshots were taken. Read-only
online backups of the live sfx store and of alt (copied alt→sfx only),
filtered against every examined snapshot, leave **1** eligible
instruction/correction item (sfx 1, alt 0) of the required **10**: a deficit
of 9 (`holdout6-result.json`). Not re-sampled for this candidate; the
candidate was not measured on it; **no independent generalization claim**.

## Latency, volume, lines

- Procedural stage, two ordinary passes on a fresh sfx copy: 7.0 / 3.8 s
  (5.8 / 3.7 s in the first run; `e3cb245` 4.4 / 1.7 s). The first recall per
  scope after conversion re-embeds the rewritten carriers in recall's existing
  drain: 349 s for all searched scopes alone, 516 s with a concurrent gate.
  Recall medians fall (table) because one carrier per title replaces many
  large schemas in the answer.
- Volume: above. Reading +11.2% over all sets (design recipes +33.5%, topics
  +14.3%, H2 +2.8%, H3 +1.4%); storage +33%. For comparison under the old
  reader (not the same reader, shown only for scale): `47ec513` read 16× the
  original on design topics, `e3cb245` 9.7×.
- Lines against `24e9009` (`git diff --numstat 24e9009 -- src`):
  `consolidation.py` +99/−96, `schema_dedup.py` +4/−3, `storage.py` +5/−3,
  `retrieval.py` +2/−2, `score_gate.py` +1/−1: **+111/−105, net +6**. Removed:
  the whole-trace step fallback, `keys_by_group` bookkeeping, the separate
  `_find_existing_schema` scan. Added: the carrier form and in-place level move
  (+2 storage), the carrier title (+1), keeping the trigger label (+3 in
  consolidation). Nothing unrelated was removed or compacted to reach a count.

## Accepted limits and evidence gaps

1. **Line budget: net +6 against ≤ 0, accepted by the operator.** The remaining additions are the
   carrier's in-place level move, the carrier title in dedup and the label
   keeping; the replaced mechanisms are already deleted. Meeting ≤ 0 would
   need compaction or removal of unrelated code, which the goal forbids. The
   operator explicitly accepted this measured exception on 2026-09-30.
2. **Insufficient independent holdout** (1 of 10 items). Every end-to-end
   number is fixture or regression replay on examined sfx data. The operator
   accepted completion with this evidence gap; independence and general gain
   are not claimed.
3. The storage (+33%) and reading (+11%) growth is the price of keeping every
   current record searchable and whole in its group's carrier; it is reported,
   an explicit part of the accepted tradeoff.
4. The query-only reader re-fetches every cut delivery (for both sides); a
   reader that skips irrelevant cut results would read less on both sides.
5. `test_same_trigger_carriers_do_not_crowd_out_a_record_recall_delivers_directly`
   failed once in about ten local runs with the real embedding model (not
   reproduced in 8 later runs; the gate uses the hash backend and passed).

## Fixture and replay vs live effect

Everything above is fixture or local replay. Nothing was measured on a live
server; landing is not evidence of benefit. Live servers run master (`8289d37`)
and are not touched by this branch.

## Gate

One ordinary full project gate on the final pin (production `c25a6cd`, tests
at `ca1fc85`): `scripts/test.sh tests -p no:warnings -rf -q` — exit 0, 3307
passed, 84 skipped, 0 failed (counted from the progress output; about 5 min).
The required targeted command `python3 -m pytest -q
tests/test_procedure_case_separation.py tests/test_schema_distillation.py
tests/test_procedural_schemas.py tests/test_consolidation_no_copies.py` exits
0. The gate supports the regression result; it does not supply an independent
holdout or prove a live effect.

## History: earlier candidates (superseded; old reader unless noted)

The old reader also re-fetched a cut wanted record (target id) and looked up
every evidence id a carrier listed; its volumes are not comparable with the
table above. Itemwise new losses of instructions/corrections:

| candidate | idea | design recipes | H2 | H3 | design-topic read chars (original 1.58M) | net lines |
|---|---|---|---|---|---|---|
| `7903a82` first landing | retire case-history schemas | 2 | 22 | – | 0.45M | – |
| `e8913bb` | carrier with 600-char leads, era-filtered | 2 | 2 | – | 7.69M | – |
| `4f8a6d2` | + title dedup for carriers | 0 | 2 | – | – | – |
| `47ec513` | + peer carrier ids (bucket reading) | 0 | 0 | 0 | 25.60M | +58 |
| `e3cb245` | group-identity dedup, no peers | 1 | 2 | 1 | 15.36M | 0 |
| `7cb9436` (new reader) | full-text carrier + title dedup | 0 | 2 | 0 | 2.38M (orig. 2.08M) | +3 |
| **`c25a6cd`** (new reader) | + keep trigger label | **0** | **0** | **0** | 2.38M (orig. 2.08M) | +6 |

Holdout 4 (alt corpus, frozen protocol, independent for `e8913bb` only): 0 new
instruction losses of 21 + 42 items; `holdout4-result.json`. Holdout 5 (alt,
new snapshot) had 0 eligible items (`holdout5-result.json`). Earlier protocols
and aggregates are kept unchanged: `holdout-protocol.md`,
`final-regression-*`, `group-identity-regression-*`.
