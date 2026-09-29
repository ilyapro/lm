# Report 2: `LM_RECALL_SCHEMA_TRIGGER=name` back to the quality bar (goal useful-schema-survives-ranking)

The rules are in `prereg-2.md`, committed (543224a) before the code change and
before any replay of the new holdout. The numbers come from
`scripts/schema_trigger_replay.py run|names` on the 2026-09-29 15:21Z snapshots
(sfx `5a3b1f8a…`, alt `3ddc4867…`); the files are in `v2/`. sfx and alt are
reported separately and never merged. No query text is committed.

## What changed

There is one line of behaviour, and only under the valve. In
`MemoryRecallService._named_schemas_first`, a schema that the query names
(the query's token set equals the schema's trigger) is scored with bm25 1.0
before the gate checks it. The whole query is the schema's title, so it is
the lexical channel's rank 1. Before this change, bm25 counted the query words
across the corpus, and a 1.9k-char schema called `implementation repair`
ranked 5th for the query `implementation repair` (bm25 0.2, vector 0, gate
0.10). The named schema still goes through the same `passes_gate` on the same
reference blend. The threshold, the query's own demotion, feedback and the
superseded penalty all still apply. There is no bypass and no new scale.
Removing a redundant filter (the loop only walks ranked results anyway)
keeps `src/` at +12 / −12 lines. With the valve unset, behaviour is
unchanged: legacy delivered lists are identical, 335/335.

Rejected: dropping the reorder and relying on bm25 1.0 inside the ranking
blend alone. On names dev, agreement rank-1 fell to 58/80, because per-scope
weights let strong-vector traces outrank the named schema.

## Data sizes

| | sfx | alt |
|---|---|---|
| original corpus events (prereg `holdout` segment) | 335 | 146 |
| … marked used / irrelevant procedure-events | 39 / 98 | 0 / 7 |
| new holdout events (07:00Z–15:00Z) | 176 | 17 |
| … marked used / irrelevant procedure-events | 61 / 169 | 0 / 2 |
| names dev (prereg L5 sample still active) | 149 | 50 |
| names holdout (every other schema with a trigger) | 333 | 0 (all 50 alt schemas were in dev) |

## Field replay

"proc" = procedure-events (distinct `(event, title)`; delivered when any copy
of that title is delivered in full). "nodes" = every marked copy counted, as
in `report.md`.

| | sfx original legacy | sfx original name | sfx new legacy | sfx new name |
|---|---|---|---|---|
| irrelevant proc delivered (of marked) | 93 / 98 | **23** | 32 / 169 | **6** |
| used proc delivered (of marked) | 31 / 39 | **19 (61%)** | 21 / 61 | **10 (47.6%)** |
| irrelevant share among delivered marked proc | 75.0% | **54.8%** | 60.4% | **37.5%** |
| nodes: irrelevant / used delivered | 76 / 27 | 10 / 11 (41%) | 32 / 21 | 6 / 8 (38%) |
| schema text, chars (full node chars) | 442,508 (916,132) | **33,283** (191,909) | 185,755 (259,071) | **25,945** (68,295) |
| schema slots | 181 | 54 | 66 | 25 |
| latency mean / p90, s | 1.75 / 4.22 | 1.72 / 4.07 | 3.06 / 5.07 | 3.03 / 4.95 |

alt original: irrelevant proc delivered 7 → 2 (of 7), no used marks, schema
text 6,737 → 1,690, latency 3.79 → 3.43 s. alt new: 17 events, no schema
delivered by either arm. Both are below the 20-proc threshold and are
reported, not judged.

The reference `name@9034741` (the code before this goal):

- On the original corpus, all 335 delivered lists are identical to `name`.
  No field query there is an exact name that the old gate refused.
- On the new holdout, 175/176 are identical. The one difference is the
  exact-name query `reopen_lesson`: the new code delivers the procedure,
  which the field marked used. `name@9034741` keeps 9/21 used proc (43%),
  `name` keeps 10/21. Irrelevant proc delivered is 6 in both.

## Names (L5; top 5, the schema's own scope, field env)

| | sfx dev legacy | sfx dev name | sfx holdout legacy | sfx holdout name | alt dev legacy | alt dev name |
|---|---|---|---|---|---|---|
| queries | 149 | 149 | 333 | 333 | 50 | 50 |
| procedure at rank 1 | 98 | **143** | 200 | **322** | 47 | **48** |
| … where legacy saw meaning agreement | 45 / 80 | **80 / 80** | 120 / 219 | **219 / 219** | 46 / 47 | **47 / 47** |
| latency mean, s | 3.47 | 3.47 | 1.74 | 1.73 | 4.57 | 4.25 |

On the same snapshot, `name@9034741` has names dev at 119 rank 1 and 75/80
agreement. The 5 agreement misses were `active goal supervision` ×4 (vector
0.595, bm25 0, gate 0.29) and `implementation repair` (gate 0.10). The new
code also puts at rank 1 explicit names without meaning agreement, most of
which legacy had at rank 1 as well: `reopen lesson` ×10, `task outcome`,
`mr review`, `implementation fix`, `p7b v2 fresh evidence`, and (legacy rank 2)
`gate bug lesson`, `augment operator transaction`, `dashboard goal supervision`.
The query is the procedure's name. A demotion from that query's own irrelevant marks
would still take the procedure out.

## Verdicts

| line | sfx original corpus | sfx new holdout | alt |
|---|---|---|---|
| L1 irrelevant proc ≤ 0.5× and share lower | **PASS** (93 → 23; 75.0% → 54.8%) | **PASS** (32 → 6; 60.4% → 37.5%) | not judged (7, 2) |
| L2 used proc kept ≥ 0.5× | **PASS** (31 → 19 = 0.61) | **FAIL** (21 → 10 = 0.476) | not judged (0 used) |
| L3 schema text lower | **PASS** (−92%) | **PASS** (−86%) | not judged; −75% |
| L4 latency mean ≤ legacy + 5% | **PASS** | **PASS** | not judged; lower |
| L5 names rank 1 ≥ legacy, every agreement query at rank 1 | **PASS** dev 80/80, holdout 219/219 | – | **PASS** dev 47/47; holdout empty |
| N differences vs `name@9034741` | 0 of 335 | 1 of 176, exact-name query, +1 used | – |

**By the prereg-2 rule the bar is not met on sfx.** L2 fails on the new
holdout by one procedure-event: 10 of 21 kept, and 11 are needed. L5 now passes
everywhere it is judged. The earlier L2 figure of 46% was mostly duplicate
counting: the supervisor marks several copies of one procedure, and dedup
delivers one. On the original corpus that figure is 61% per procedure.

## Where a useful procedure is lost (sfx, legacy delivered it, `name` did not)

`v2/lost-used-sfx.json`, per procedure-event, at the best copy's stage:

| stage | original (12) | new (11) | what it is |
|---|---|---|---|
| gate (score below 0.35) | 8 | 5 | No meaning evidence: vector 0 (not in the vector top-k), bm25 rank ≥ 7, gate 0.03–0.22. `gate bug lesson` ×4, `dashboard agent chats` ×2, `augment repair transaction`, `chats message audio playback`, `ae goal creation style` ×4, `ae stream agents thinking map`. Legacy delivered them only because the query shared trigger words. |
| slot (passes the gate without demotion, ranked below `max_results` passing results) | 1 | 3 | Ranks 28 (gate 0.367); 4, 11 and 23 (gate 0.39–0.46) behind passing traces. Legacy lifted schemas with the 1.8× boost. |
| admission (candidate dropped by the cross-scope rule) | 0 | 2 | One project:ae event. Two `global` schemas with vector 0.57–0.60 fall below 0.9 × the best in-scope vector. Legacy admitted any trigger-found schema across scopes. |
| dedup (this copy collapsed, the kept copy failed the gate) | 1 | 1 | `dashboard goal api supervision`: vector 0.55–0.64 and bm25 ≤ 0.1, so the collapsed copy is at gate ≈ 0.25 and would fail too. |
| retrieval (no channel found it) | 2 | 0 | `dashboard goal api supervision`, `augment repair transaction`: only the legacy trigger found them. |

Every remaining loss is the equal-criterion verdict of the existing
retrieval, gate and cross-scope admission, applied to a schema as to any
other node. Getting them back means giving schemas an origin-dependent
advantage again (the boost, trigger admission across scopes, a scale of their
own), or changing the shared gate or admission thresholds, which also changes
behaviour without the valve. Neither is in scope. Negative case: containing
a procedure name as a phrase in a longer query is not a name either. At
08:00Z on the new holdout, a supervisor query contained `active goal
supervision`, and the field marked that procedure irrelevant.
