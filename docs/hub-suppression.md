# Global hub suppression

Goal: recall-precision, item P2. Code: `src/living_memory/hubs.py`. Tests:
`tests/test_hub_suppression.py`.

## What it does

Some nodes get delivered for many unrelated questions and are marked
`irrelevant` each time. On the held-out half of the sfx marks, the rule
"irrelevant from 3 or more distinct queries, never `used`" flags 99 nodes. They
hold 13% of top-3 slots and received 3 `used` marks against 247 `irrelevant`.
Query-relative demotion (`docs/query-irrelevance.md`) can't remove such a node,
because each new question starts at full strength. Hub suppression gives these
nodes one global multiplier that applies to every query.

A node is a **hub** when both of these hold:

1. It has accepted `irrelevant` marks (`recall_feedback_marks.accepted = 1`)
   from at least `LM_HUB_MIN_QUERIES` **distinct questions**.
2. It has **no positive evidence**. Positive evidence is either of:
   - an accepted `used` mark at any time;
   - a `recall_credit_ledger` row (`grounded` or `lookup`) or a
     `recall_explicit_credit` row for the node with `credited_at` at or after
     the node's first accepted irrelevant mark.

A single piece of positive evidence lifts the demotion. Credit that is older
than every complaint does not protect the node. This is deliberate: a node that
was useful once and later turned into noise still counts as a hub.

`MemoryRecallService._collect_hub_demotions` returns `{node_id: factor}`. The
result is merged with the query-relative demotions, and when both apply to a
node the smaller multiplier wins (`retrieval._merge_demotions`). The merged
map is applied in `rank_candidates`, so the hub's final score is multiplied by
the factor and the freed slot goes to the next candidate.

### Distinct questions

Each mark is keyed by the query anchor of the recall event it was passed on.
The key is resolved with the identity `query_anchors.resolve_query_anchor`
uses, trying each step in order:

1. The exact `(normalize_scope(scope), recall_fingerprint(query, scope))` row
   in `query_anchors`.
2. Otherwise, the `anchor_id` of the `query_irrelevance` row whose
   `last_event_id` is that event. This is the cosine near-duplicate anchor that
   the credit write path picked.
3. Otherwise, the whitespace-collapsed query text in its scope.

As a result, repeats of one question count once however many times it is
marked, and the same text in two scopes counts as two questions. This matches
the anchor channel's own rule.

**Anchors against raw queries (sfx copy, 2026-09-29, all accepted marks
through 2026-09-28T21:42Z):**

| min distinct | hubs by anchors | hubs by raw queries |
|---:|---:|---:|
| 2 | 304 | 304 |
| 3 | 146 | 146 |
| 4 | 73 | 73 |
| 5 | 40 | 40 |

Only 1 of the 1,012 irrelevant-marked nodes has fewer distinct anchors than
distinct raw queries, and that node sits below every threshold in the table.
On this data the anchor identity changes no hub verdict. Raw-query counting
would give the same set; anchors are used only because they match the
query-relative path's definition of "the same question".

At `N = 3`, 256 nodes reach 3 distinct questions. 56 of them are lifted by a
`used` mark and 54 more only by credit after their first complaint, which
leaves **146 hubs** on the full data. The 99 in the goal text come from the
held-out half only. The hub set that goes into the measurement is defined on
train with `hub_counts(conn, before=<split>)`.

## Valves

| env | meaning | default |
|---|---|---|
| `LM_HUB_SUPPRESSION_FACTOR` | Multiplier for hubs, a float in `[0, 1)`. | unset = **off** |
| `LM_HUB_MIN_QUERIES` | Distinct questions needed to be a hub, an integer >= 1. | `3` |

When `LM_HUB_SUPPRESSION_FACTOR` is unset, blank, unparsable, non-finite or
outside `[0, 1)` (so `1.0` also counts as off), `hub_demotions` returns `{}`
without reading the store, and the ranking is byte-identical. The test
`test_off_is_identity_on_ranking` checks this. An invalid
`LM_HUB_MIN_QUERIES` falls back to 3. The factor value to switch on is chosen
by the precision measurement (`artifacts/recall-precision/`), not here.

## Absent tables

Every read checks `sqlite_master` first:

- no `recall_feedback_marks`: no hubs;
- no `recall_events`: the event id stands in for the question;
- no `query_anchors` / `query_irrelevance`: that identity step is skipped;
- no credit tables: that evidence is absent.

A `sqlite3.Error` inside `hub_demotions` returns `{}`, and the service wrapper
catches everything else. A read-only snapshot (`file:...?mode=ro`) works,
because nothing here writes to the store.

## Offline reuse

- `hub_counts(conn, *, min_queries=3, before=None) -> list[HubCount]` is a pure
  function of a `sqlite3.Connection`. It returns one row per irrelevant-marked
  node: `distinct_anchors`, `distinct_queries`, `irrelevant_marks`,
  `first_irrelevant_at`, `used_marks`, `credits_after_first` and the `hub`
  verdict. `before` drops every mark and credit row at or after that timestamp,
  which defines hubs on a train split.
- `hub_ids(conn, ...) -> {node_id: distinct_anchors}` returns the hubs only.

## Cache and cost

The inputs are append-only: nothing in LM deletes from the marks, credit or
anchor tables. The cache key is `MAX(rowid)` of `recall_feedback_marks`,
`recall_credit_ledger`, `recall_explicit_credit` and `query_anchors`, which
costs four index lookups. Marks and credit grow on almost every recall, so a
changed key does not by itself trigger a recount:

- **New rows only:** the path reads just the rows appended since the last
  call. A `used` mark, or credit at or after the hub's first complaint, lifts
  the hub **immediately**.
- **Full recount:** this is the only way a node *becomes* a hub. It runs when
  new irrelevant marks arrived since the last count and that count is older
  than `HUB_REFRESH_SECONDS` (60 s). A new hub therefore appears within about
  a minute. Replays open fresh stores, so they always count in full.
- The cache is stored on the `MemoryStore` object and dies with it.

**Overhead, sfx store copy** (made with `retrieval_harness.backup_database`
from the live DB opened `mode=ro`; 146 hubs; WSL2 laptop):

| path | p50 | p95 |
|---|---:|---:|
| valve off | 0 (env read only) | — |
| cache hit (inputs unchanged) | 0.037 ms | 0.082 ms |
| rows appended (lift check) | 0.25 ms | 0.51 ms |
| full recount (≤ once / 60 s) | 38.9 ms | 53.4 ms |

End to end, 40 recent sfx queries through `MemoryRecallService.memory_recall`
(hash embedding backend, warm cache) took 207 ms p50 with the valve off and
209 ms p50 with it on. The difference is inside run-to-run noise. The same run
showed hub slots in the top 5 falling from 20/200 to 2/200 at factor 0.1. This
is an in-sample illustration only. The holdout evidence is the job of the
precision measurement.

## Why these choices

- **Distinct questions, not mark count.** An agent that repeats one question
  and marks the same node every time is giving query-relative evidence, and
  `query_irrelevance` already handles that. Only breadth across questions
  shows that a node is noise everywhere.
- **`used` ever, credit only after the first complaint.** An explicit `used`
  is a direct claim of usefulness, so it wins. Grounded and lookup credit are
  weaker, implicit signals, and a node that was consumed months ago can still
  have turned into noise since.
- **Multiplier, not removal.** A demoted hub stays in the candidate pool and
  can still win on a query where it scores far above the rest. With factor
  `0` it effectively leaves the answer while staying reachable through
  `memory_lookup`.
