# Recall: same-procedure schema duplicates take one slot

Stage of goal `recall-precision` (P4). Code: `src/living_memory/schema_dedup.py`,
called from `MemoryRecallService.memory_recall` on the full ranked list,
**before** the quality gate and the `max_results` cut.

## Valve

`LM_RECALL_SCHEMA_DEDUP=1` (also `true`/`yes`/`on`) turns it on. If it is unset or
has any other value, `collapse_schema_duplicates` is the identity and `memory_recall`
output does not change.

When the valve is on, the stage walks the ranked list in order and does this:

- It keeps the highest-ranked schema of each title.
- It drops every lower-ranked schema with the same title. Because the cut happens
  before `max_results`, the next ranked results fill the freed slots.
- Trace and concept nodes are never collapsed. A schema with no title (bare
  `Procedure`, empty first line) is never collapsed either.
- Rank 1 is always kept, because the first schema of any title is kept.

The collapsed duplicates are not delivered. They stay in the store unchanged, and
`memory_lookup` still reaches them by id.

## Definition of "same title"

**title = first line of the schema's content, with whitespace collapsed and casefolded.**
Consolidation writes that line as `Procedure: <trigger>`
(`consolidation._format_schema_content`), and the agent sees it as the heading of
the result.

### Measurement (sfx, 2026-09-29)

The data is a snapshot of `~/.local/share/living-memory/global.sqlite3` taken with
`retrieval_harness.backup_database`, so the live DB was opened only with `mode=ro`.
The events are `recall_events.results`, i.e. the answers as delivered. An event
counts when its answer holds at least 2 schemas with the same key. "Extra slots"
counts the slots held by the 2nd and later copies, which the stage would free.
"Group slots" counts every slot of a duplicated group.

Window from `2026-09-27T14:45Z` (start of the `recall_feedback_marks` window):
1054 events, 4669 slots.

| candidate key | events | extra slots | group slots |
|---|---|---|---|
| **first content line** | **9.1% (96)** | **6.4%** | 8.8% |
| `context.trigger` | 9.1% (96) | 6.4% | 8.8% |
| first line + scope | 8.6% (91) | 6.3% | 8.6% |
| `context.procedure_id` | 5.0% (53) | 3.5% | 4.9% |
| `context.procedure_key` | 4.9% (52) | 2.9% | 4.0% |
| `context.task_pattern` | 4.2% (44) | 2.5% | 3.4% |
| `context.name` | 0 | 0 | 0 |

Only the first-line (= trigger) title reproduces the goal's ~10% of events / ~7% of
slots. The keys in `context` miss about half of the duplicates, because many schemas
lack `procedure_id` or carry a per-goal `procedure_key` or `task_pattern` under the
same heading. Across all history (76150 events), the first line gives 17.4% of
events and 10.1% extra slots.

alt, same window, read-only (`ssh alt`, `mode=ro`): 562 events, 2362 slots. The first
line gives 0.2% (1 event) and 0.0% extra slots, which matches the goal's "alt: 0".

### Does it merge different procedures?

- All 893 schema nodes on sfx have a first line equal to
  `"Procedure: " + context.trigger`, so the two keys are the same key.
- The store has 308 distinct titles, and 37 of them have several nodes. **No title
  maps to more than one `procedure_id`**, after normalizing `-`/`_` (e.g.
  `octopus_supervision` / `octopus-supervision`). Some nodes have no `procedure_id`.
  Where they share a title with nodes that have one, the title comes from the same
  trigger.
- The window has 301 duplicate pairs. 296 are in the same scope, and 126 share a
  `procedure_key`. The steps differ often (median token Jaccard 0.23), because the
  duplicates are separate consolidations of the same procedure at different times.
  The main groups are `dashboard goal api supervision` (134 pairs),
  `active goal supervision` (82), `reopen lesson` (41) and `verify pass` (20). They
  are the same procedure under one heading, so the agent sees near-identical entries.
  After the collapse, the other copies are still reachable through `memory_lookup`
  and through later recalls where the kept copy ranks lower.

Scope is deliberately not part of the key. It changes little (8.6% vs 9.1%), and
two schemas with the same heading take up the answer the same way whatever their
scope.

Whether turning the valve on improves usefulness is measured on the live path by
`precision-measure-handoff`, not here.
