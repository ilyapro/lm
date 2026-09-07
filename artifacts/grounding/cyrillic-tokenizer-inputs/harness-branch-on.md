# End-to-end retrieval-harness baseline

Generated: 2026-09-07T05:50:00Z | snapshot `/home/sfx/.cache/living-memory-harness/phase1/frozen-chunked.sqlite3` | commit `d54670d6db0fd80790eb5d265df8b3ef1ed6bc34`

## Snapshot and goldset

- Source DB: `/home/sfx/.cache/living-memory-harness/snapshot.sqlite3`
- Snapshot sha256 `d46f1b845adc3d806ec5cb7f593438f723bdf2b7c491c0f47aa92e4ef20bd75d` captured 2026-08-17T20:10:14Z
- Snapshot rows: active_nodes 12862, connections 142906, nodes 16683, recall_events 54777
- recall_events span 2026-05-14T20:05:23Z .. 2026-08-17T17:58:08Z | cutoff `None` | seed 0
- Goldset `/home/sfx/.cache/living-memory-harness/recalib/goldset-recalibration.jsonl` sha256 `1aabe424c0dec64086a44ded20a96d6daaca0dffb40a9940a52bdf5add4366d1` with 3659 items (0 tail)
- Embedding backend `auto` model `paraphrase-multilingual-MiniLM-L12-v2` | harness v2

## Metrics

| bucket | items | w/relevant | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|---|
| overall | 3659 | 3659 | 0.184 | 0.597 | 0.764 | 0.356 |
| stratum:content_grounded | 3659 | 3659 | 0.184 | 0.597 | 0.764 | 0.356 |
| tail | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| fresh_node_visibility | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |

### Per channel

| channel | items | w/relevant | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|---|
| bm25 | 3659 | 3659 | 0.181 | 0.598 | 0.765 | 0.359 |
| graph | 3659 | 3659 | 0.154 | 0.556 | 0.763 | 0.321 |
| trigger | 3659 | 3659 | 0.190 | 0.600 | 0.764 | 0.359 |
| vector | 3659 | 3659 | 0.221 | 0.621 | 0.763 | 0.386 |

### Channel attribution (2808 items with a relevant result)

| method | first-relevant results | share | unique to it | unique share |
|---|---|---|---|---|
| bm25 | 2109 | 0.751 | 0 | 0.000 |
| vector | 2659 | 0.947 | 484 | 0.172 |
| graph | 468 | 0.167 | 0 | 0.000 |
| trigger | 414 | 0.147 | 0 | 0.000 |

- `content_grounded` (2808 items with a relevant result), method sets: `bm25+graph+trigger` 63, `bm25+graph+trigger+vector` 92, `bm25+graph+vector` 117, `bm25+trigger` 11, `bm25+trigger+vector` 119, `bm25+vector` 1707, `graph+trigger` 75, `graph+trigger+vector` 35, `graph+vector` 86, `trigger+vector` 19, `vector` 484

## Live-agreement sanity

- Sampled 20 of 3659 items (seed 0, top-5): agreement_rate 1.000
- No divergences.

## Definitions

### Tail rule

```text
Per relevant node of a goldset item:

1. grounding tokens := tokenize(node.content) & tokenize(justifying_text),
   where `tokenize` is `living_memory.embeddings.tokenize` and the justifying
   text is the consuming trace's content for the `content_grounded` stratum
   and the goldset query itself for `cross_lingual` and `role_query`.
2. visible span := encode node.content with the real model tokenizer using
   truncation=True, max_length=128, return_offsets_mapping=True; drop every
   token whose get_special_tokens_mask() entry is 1; visible_char_end is the
   maximum end offset of what remains. That is the character prefix the
   embedding vector actually sees.
3. The node is TAIL iff its grounding-token set is non-empty AND every
   occurrence of every grounding token in node.content starts at character
   offset >= visible_char_end. An empty grounding-token set (the
   cross_lingual case by construction) is NOT tail, and the reason is
   recorded in the item's provenance.

Item level: an item is TAIL iff at least one relevant node has a non-empty
grounding-token set and every such node is TAIL. A single-relevant-node item
therefore reduces to the per-node rule exactly; an item that keeps any
head-grounded relevant node is not tail, because that node is still reachable
through the prefix the vector sees.
```

### Per-channel buckets and channel attribution

```text
A per-channel bucket re-orders the SAME returned result list by that single
channel score descending, ties broken by the live rank, and computes
hit@1/hit@5/hit@10/MRR over that order. No result is added or removed, so the
bucket measures how well one channel alone would have ranked what retrieval
actually returned. `channel_attribution` is a separate view: for the first
relevant result in the LIVE order it counts each method present in that
result's `.methods`, so a result found by two channels counts once for each.
```

### Live agreement

```text
For a `random.Random(seed)` sample of goldset query ids (drawn over the sorted
id list), a second `MemoryStore` + `MemoryRecallService` pair is constructed
independently of the runner's code path on the same working copy, and its
top-5 node ids are compared with the runner's top-5. `agreement_rate` is the
share of sampled items that match exactly; every mismatch is listed in full.
The run exits non-zero when the rate is below 1.0 unless `--divergence-note`
explains it.
```

### Anchor holdout, coverage and fresh-node visibility

```text
Per goldset item, computed from the snapshot alone so both arms of an
anchors-on/anchors-off comparison annotate identically:

1. The query is embedded with the run's own encoder and matched against the
   live anchor vectors of the item's resolved `ScopePlan` through
   `query_anchors.match_anchors`, at the shipped floor
   (ANCHOR_MATCH_COSINE_THRESHOLD, limit ANCHOR_MATCH_LIMIT) -- the same call,
   scope gate and floor `retrieval._collect_anchor_seeds` uses. The annotation
   therefore states what retrieval could see, not what a laxer probe finds.
2. An item is EXACT_REPEAT iff `storage.recall_fingerprint(query, scope)`
   already names a live anchor in one of those scopes, and
   FINGERPRINT_DISJOINT otherwise. Both subsets are reported apart, never
   merged into a headline: an exact repeat retrieving its own past answer is a
   lookup table, and the generalization claim lives only in the disjoint
   subset, where a new query must reach the right node by resembling a
   *different* past query.
3. `anchor_as_of` := the newest `updated_at` over the matched anchors -- the
   moment the anchor corpus last learned anything about this query. The
   backfill stamps it with the source event's own `created_at`, so it is a
   training-set timestamp and not a run timestamp.
4. A relevant node is FRESH iff its `created_at` is strictly after
   `anchor_as_of` and it is not itself one of the matched anchors' live edge
   targets. An item enters the FRESH-NODE-VISIBILITY slice iff it matched an
   anchor, that anchor holds at least one live edge (it points at an older
   node), and the item has at least one fresh relevant node. The slice scores
   ONLY those fresh nodes: it asks whether the node the anchor cannot know
   about is still reachable once the anchor's older targets are seeded into
   the graph channel. This is the gate against "the rich get richer"; anchors
   must not make it worse than the anchor-free arm.
```

## Limitations

- Relevance is the goldset's judgement, frozen at build time: `content_grounded` items inherit replay's IDF-containment label over the consuming trace, `cross_lingual` items inherit the paraphrase query's own top-k (so they measure jargon-vs-paraphrase agreement, not absolute truth), and `role_query` items are curated against active `level:schema` triggers.
- A relevant node that retrieval never returns counts as a **miss**, not as an unevaluated item: it enters the denominator with no hit and no reciprocal rank. The run therefore measures ranking and candidate collection together and cannot tell one failure from the other.
- Per-channel buckets re-order only what retrieval already returned, so they bound a channel's ranking power, not its recall.
