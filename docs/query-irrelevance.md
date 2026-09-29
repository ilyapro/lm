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

## Strength valves (goal recall-precision, P3)

Measured on live traffic, the default curve rarely acts: typical anchor
cosines are 0.65–0.70, so `closeness = (cos - 0.60) / 0.40` is about
0.15–0.25, and one mark (weight 0.5) at F = 0.5 gives m of about 0.92–0.96.
Two read-side valves, both read in `irrelevance.py`, steepen the curve. When
unset, or set to an invalid value, `query_demotions` gives byte-identical
output (pinned by `tests/test_query_irrelevance_strength.py` against the
original arithmetic):

| Valve | Meaning | Unset |
|---|---|---|
| `LM_QUERY_IRRELEVANCE_FULL_COSINE` | cosine at which closeness reaches 1.0: `closeness = (cos - 0.60) / (full - 0.60)`, clipped to [0, 1]. Valid in (0.60, 1]. | `full = 1.0` (the /0.40 span) |
| `LM_QUERY_IRRELEVANCE_MARK_WEIGHT` | read-side worth of one mark. A stored weight `w` counts as `w / 0.5` marks and reads as `min(1, marks * mark_weight)`. Valid in (0, 1]. | stored weight as is |

The write side is unchanged. A mark still stores +0.5, and no row is
rewritten or migrated, so unsetting a valve is an instant rollback (a restart
is needed, because env values are read by the running process). The bound
`F <= m` and the match floor 0.60 stay the same. The valves move how much of
the range between 1 and `F` a match uses. They do not change which
(query, node) pairs are demoted.

### Census: sfx, 2026-09-29

Script: `scripts/query_demotion_strength_census.py` (read-only: `mode=ro`, or
`--snapshot` via the SQLite backup API). It is run on a backup-API copy of the
sfx store, with data up to 2026-09-28T21:42Z. Events run from the first
irrelevance row (2026-09-27T14:55Z). Applicable case, following the definition
in LM 01M3MXJWTPZ2H2HJ3ZSPKZXGTE: event × node, where the node has a
`query_irrelevance` row created before the event, on an anchor that existed
at the event, is in the event's resolved scopes, and is among the top-5
matches at cos > 0.60. The query vector is the production encoder
(paraphrase-multilingual-MiniLM-L12-v2), and F = 0.5 throughout. Row weight
at event time is rebuilt from `created_at`/`updated_at`/`marks`. That is exact
except for 28 of 2,559 rows (23 cancelled, 3 with more than two marks, 2
cancelled and then marked again), and those are handled conservatively.

1,047 events, 3,843 applicable cases, 434 of them on a delivered node:

| Setting | FULL_COSINE | MARK_WEIGHT | m > 0.9 | m <= 0.75 | median m | delivered: m > 0.9 | delivered: m <= 0.75 |
|---|---|---|---|---|---|---|---|
| default | — | — | 72.3% | 3.8% | 0.942 | 59.7% | 4.2% |
| fc080 | 0.80 | — | 43.9% | 16.7% | 0.884 | 33.6% | 25.4% |
| fc075 | 0.75 | — | 33.7% | 30.2% | 0.845 | 25.4% | 42.6% |
| fc070 | 0.70 | — | 24.0% | 47.2% | 0.768 | 18.4% | 61.1% |
| mw1 | — | 1.0 | 44.3% | 16.4% | 0.885 | 34.1% | 25.1% |
| fc080_mw1 | 0.80 | 1.0 | 24.1% | 46.8% | 0.770 | 18.4% | 60.4% |
| fc075_mw1 | 0.75 | 1.0 | 18.9% | 58.5% | 0.693 | 13.4% | 69.1% |
| fc070_mw1 | 0.70 | 1.0 | 12.8% | 70.6% | 0.540 | 9.0% | 77.9% |

The default row reproduces the earlier measurement (69% above 0.9 on the
window up to 2026-09-28). A cosine span alone needs `full <= 0.70` to bring
the share above 0.9 under the 30% line (P3). Combining it with
`MARK_WEIGHT=1.0` gets there at 0.80, because one mark then counts at full
strength. Candidates that meet the line: **fc080_mw1** (the mildest, median
m 0.77), fc070, and fc075_mw1. These are strength numbers only. Whether a
stronger demotion improves the delivered list (and what replaces the demoted
node) is for the live-path replay of the measurement sibling
(`artifacts/recall-precision/`), which makes the final choice. alt was not
censused here: its snapshot is taken by the replay harness sibling and stored
on sfx. Run the same script on it with `--host alt` (the earlier measurement
put 91% of alt cases above 0.9).

Reproduce:

```
PYTHONPATH=src python3 scripts/query_demotion_strength_census.py \
    --db ~/.local/share/living-memory/global.sqlite3 --snapshot "$TMPDIR/sfx.sqlite3" \
    --host sfx --setting default: --setting fc080:full=0.80 --setting fc075:full=0.75 \
    --setting fc070:full=0.70 --setting mw1:mw=1.0 --setting fc080_mw1:full=0.80,mw=1.0 \
    --setting fc075_mw1:full=0.75,mw=1.0 --setting fc070_mw1:full=0.70,mw=1.0
```

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
