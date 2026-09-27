# Query-relative irrelevance demotion

Part of goal explicit-recall-feedback (P3). Code: `src/living_memory/irrelevance.py`;
hook `feedback._apply_query_irrelevance`; read path
`MemoryRecallService._collect_query_demotions` in `retrieval.py`. Tests:
`tests/test_query_irrelevance.py`. Explicit-mark interface: `docs/explicit-feedback.md`.

## What an irrelevant mark means

"This delivered node was not relevant to THIS query." The mark does not say the
node is wrong (`memory_teach` covers that) or useless, because it can be right
for another question. So an irrelevant mark never changes a node's
`usefulness_score`, its `confidence`, or the per-scope retrieval weights. It
only records an association between the query and the node, and retrieval reads
that association back for matching queries only.

## When it applies

| `LM_EXPLICIT_FEEDBACK_POLICY` | irrelevant mark writes a row | retrieval demotes |
|---|---|---|
| `off` | no | no |
| `audit` (default) | no (audited and link hygiene only) | no |
| `credit` | yes | yes |

Retrieval reads the demotion rows only under `credit`. Switching the valve back
to `audit` or `off` therefore restores the undemoted ranking immediately,
without touching the store: this is the rollback. Rows written earlier stay
behind for audit.

## Query side: the anchor

The event's query resolves to its query anchor
(`query_anchors.resolve_query_anchor`) through the same identity a grounded
consumption uses: fingerprint over (query, scope) first, then in-scope cosine
of at least `ANCHOR_DEDUP_COSINE_THRESHOLD` (0.95). Unlike `upsert_anchor`, the
resolver neither reinforces the anchor nor writes an edge, because a negative
signal must not count as a use. If no anchor exists, one is created with no
edges, so it seeds nothing into the graph channel.

At retrieval, the incoming query vector is matched with `match_anchors` using
the anchor channel's own floor and limit (`ANCHOR_MATCH_COSINE_THRESHOLD` =
0.60, `ANCHOR_MATCH_LIMIT` = 5), served from the same cached anchor vectors.
"Same question" therefore means here exactly what it means to the anchor
channel. The match runs whatever the graph depth or `anchor_seeding` setting
is: demotion is a score multiplier, not a graph seed.

## Storage: one additive table (restart requirement)

The association cannot go into an existing table:

- `query_anchor_edges` is positive evidence. Its weight saturates upward, and
  every edge becomes a graph seed, so a negative row there would pull the node
  in.
- `connections` links a node to a node and has a CHECK on `type`.

Hence one new table:

```sql
CREATE TABLE IF NOT EXISTS query_irrelevance (
    anchor_id TEXT NOT NULL, node_id TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.0, marks INTEGER NOT NULL DEFAULT 0,
    cancels INTEGER NOT NULL DEFAULT 0, last_event_id TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (anchor_id, node_id))
```

The table is additive and created lazily: the first accepted irrelevant mark
under `credit` creates it. No migration step is needed, and a store that never
sees that valve never gets the table. The code change itself still needs a
deploy and a **restart of the LM server** on each host (an operator decision).
Every read tolerates a missing table and missing anchor tables (a pre-v7 file,
or a snapshot opened `file:...?mode=ro`), which then simply has no demotions.
Writes are no-ops without anchor tables.

## Strength, accumulation, saturation

- One accepted mark adds `IRRELEVANCE_MARK_WEIGHT` = 0.5 to the (anchor, node)
  row. The weight saturates at 1.0, so two marks reach full strength. Marks
  from later events count too if their query resolves to the same anchor.
  `marks` keeps counting past saturation.
- The multiplier on the node's final score for the incoming query is

  `m = 1 - (1 - F) * weight * closeness`, with `closeness = (cos - 0.60) / (1 - 0.60)`

  where `cos` is the incoming query's similarity to the anchor and `F` is
  `LM_QUERY_IRRELEVANCE_FACTOR`. The exact query has closeness 1. A paraphrase
  just over the match floor barely moves. When several matched anchors mark
  the same node, the strongest claim wins (no product). The multiplier is
  applied inside `rank_candidates` before sorting, so the correction-dominance
  pass still sees the demoted order.
- The multiplier is bounded: `F <= m <= 1`. A demoted node is never removed
  from the candidate set, and it still ranks by its full score for any query
  that matches no marking anchor.

## Default `LM_QUERY_IRRELEVANCE_FACTOR = 0.5`

- **1.0 disables** demotion. The rows are still written, only the read has no
  effect. Values outside [0, 1] or unparsable values fall back to 0.5.
- **Why not stronger (0.0–0.3).** Explicit marks are unvalidated. The
  agreement check (`scripts/explicit_feedback_agreement.py`) has no live marks
  to score yet, and the grounded signal they are compared with is itself about
  one-third noise (LM 01M3H76JBHEZKJMBPWFVN70SB2). A wrong mark must stay
  cheap to recover from. At 0.5, one mark costs ×0.75 on the exact query and a
  saturated mark costs ×0.5: the node sinks under peers of similar score but
  stays above clearly weaker ones.
- **Why not weaker (0.8–0.9).** On live traffic, rank-1 is used in only 3–6% of
  recalls and useful nodes are 9–13% of those delivered, so scores near the
  top are tightly bunched and differences are small. At ×0.9 a mark would
  rarely change the order at all. The agent would pay for the mark and see no
  effect.
- **Retune by measurement.** This is a documented default, not a measured
  constant. After `credit` is enabled, re-run the daily effect metric
  (`scripts/recall_effect_daily.py`) and the agreement check per host. Move
  `F` towards 1.0 if demoted nodes later get grounded or lookup credit for the
  same anchor (visible as `cancels` > 0 in `query_irrelevance`).

## Cancellation: positive credit cancels (not weakens)

Grounded, lookup and explicit `used` credit all reinforce the query anchor
through `feedback._reinforce_query_anchors`. That path calls
`irrelevance.cancel_query_irrelevance` for the credited targets on the same
anchor: the weight goes to **0** and `cancels` is incremented (the row is kept
for audit). A new irrelevant mark starts accumulating again from zero.

Why cancel rather than weaken: observed use of a node for this question
(lexical grounding in the closing trace, a lookup, or an explicit `used`) is
stronger evidence than an earlier claim that it did not belong. A partial
subtraction would leave a node that has demonstrably answered the query still
demoted for it. Cancellation also works outside the explicit-feedback valve:
grounded and lookup credit cancel under any policy. That only matters if rows
exist, and only `credit` writes them.

## Lifetime

- Rows do not follow `supersedes`. A correction is new content and starts
  without demotion.
- A decayed anchor (`ANCHOR_TTL_DAYS` = 90 without a match) stops matching,
  so its demotions lapse with it. A later match revives both.

## Cost

With no live row (every store today), the read path costs one `sqlite_master`
probe plus one `LIMIT 1` query per recall under `credit`, and nothing
otherwise. With live rows, it adds one `match_anchors` pass over the cached
vectors (in-scope anchors only, about 1–3 ms, see `retrieval.py`'s anchor
latency notes) and one indexed `IN (...)` lookup over at most 5 anchors.
