# Global hub suppression

Code: `src/living_memory/hubs.py`. Tests:
`tests/test_hub_suppression.py`.

## What it does

Some nodes get delivered for many unrelated questions and are marked
`irrelevant` each time. Query-relative demotion (`docs/query-irrelevance.md`) can't remove such a node,
because each new question starts at full strength. Hub suppression gives these
nodes one global multiplier that applies to every query.

A node is a **hub** when both of these hold:

1. It has accepted `irrelevant` marks (`recall_feedback_marks.accepted = 1`)
   from at least `LM_HUB_MIN_QUERIES` **distinct questions**.
2. It has **no positive evidence**. Positive evidence is either of:
   - an accepted `used` mark at any time;
   - a `recall_credit_ledger` row with `basis='grounded'` or a
     `recall_explicit_credit` row for the node with `credited_at` at or after
     the node's first accepted irrelevant mark.

A single piece of positive evidence lifts the demotion. Credit that is older
than every complaint does not protect the node. This is deliberate: a node that
was useful once and later turned into noise still counts as a hub.
`basis='lookup'` records a read, so it stays in the ledger without lifting a
hub. Lookup of any node remains available.

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
2. Otherwise, the whitespace-collapsed query text in its scope, or the event id
   when its query is unavailable.

As a result, repeats of one question count once however many times it is
marked, and the same text in two scopes counts as two questions. This matches
the anchor channel's own rule. Historical similarity-based anchor associations
do not merge distinct questions for this count.

## Valves

| env | meaning | default |
|---|---|---|
| `LM_HUB_SUPPRESSION_FACTOR` | Multiplier for hubs, a float in `[0, 1)`. | unset = **off** |
| `LM_HUB_MIN_QUERIES` | Distinct questions needed to be a hub, an integer >= 1. | `3` |

When `LM_HUB_SUPPRESSION_FACTOR` is unset, blank, unparsable, non-finite or
outside `[0, 1)` (so `1.0` also counts as off), `hub_demotions` returns `{}`
without reading the store, and the ranking is byte-identical. The test
`test_off_is_identity_on_ranking` checks this. An invalid
`LM_HUB_MIN_QUERIES` falls back to 3.

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
  which defines hubs on a train split. Anchor identities and near-anchor
  associations remain current; `before` does not filter their creation times.
- `hub_ids(conn, ...) -> {node_id: distinct_anchors}` returns the hubs only.

## Cache and cost

The inputs are append-only: nothing in LM deletes from the marks, credit or
anchor tables. The cache key is `MAX(rowid)` of `recall_feedback_marks`,
`recall_credit_ledger`, `recall_explicit_credit` and `query_anchors`, which
costs four index lookups. Marks and credit grow on almost every recall, so a
changed key does not by itself trigger a recount:

- **New rows only:** the path reads just the rows appended since the last
  call. A `used` mark, or grounded or explicit credit at or after the hub's
  first complaint, lifts the hub **immediately**. Lookup rows advance the cache
  revision but leave the hub suppressed.
- **Full recount:** this is the only way a node *becomes* a hub. It runs when
  new irrelevant marks arrived since the last count and that count is older
  than `HUB_REFRESH_SECONDS` (60 s). A new hub therefore appears within about
  a minute. Replays open fresh stores, so they always count in full.
- The cache is stored on the `MemoryStore` object and dies with it.

Each full recount reads `(scope, fingerprint, id)` from `query_anchors` in one
SQL statement and resolves every event against that observation. It reads no
anchor vectors or metadata. The identity map is local to the count and uses
memory proportional to the number of anchors; it is rebuilt at each full
recount. Exact attribution therefore adds one SQL read regardless of the
number of historical events. Absent anchors or events skip that read.

Event identity uses the normalized effective scope and a recomputed fingerprint.
The stored recall-request fingerprint can refer to a different requested scope.
The event map shares normalization within the call; near-anchor fallback remains
specific to both event and node. New anchor identities are visible at the next
full recount under the same refresh rule.

## Why these choices

- **Distinct questions, not mark count.** An agent that repeats one question
  and marks the same node every time is giving query-relative evidence, and
  `query_irrelevance` already handles that. Only breadth across questions
  shows that a node is noise everywhere.
- **`used` ever, grounded or explicit credit only after the first complaint.**
  An explicit `used` is a direct claim of usefulness, so it wins. Grounded
  credit indicates actual use; explicit credit is intentional positive feedback.
  A lookup only indicates reading, which can happen while checking an irrelevant
  result. Earlier positive credit does not protect a node that later turns noisy.
- **Multiplier, not removal.** A demoted hub stays in the candidate pool and
  can still win on a query where it scores far above the rest. With factor
  `0` it effectively leaves the answer while staying reachable through
  `memory_lookup`.
