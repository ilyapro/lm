# End-to-end retrieval-harness baseline

Generated: 2026-08-18T10:38:37Z | snapshot `/home/sfx/.cache/living-memory-harness/phase1/frozen-chunked.sqlite3` | commit `fcd88d20043609cc3c53a3d89431d46e4790fae0`

## Snapshot and goldset

- Source DB: `/home/sfx/.cache/living-memory-harness/snapshot.sqlite3`
- Snapshot sha256 `d46f1b845adc3d806ec5cb7f593438f723bdf2b7c491c0f47aa92e4ef20bd75d` captured 2026-08-17T20:10:14Z
- Snapshot rows: active_nodes 12862, connections 142906, nodes 16683, recall_events 54777
- recall_events span 2026-05-14T20:05:23Z .. 2026-08-17T17:58:08Z | cutoff `2026-06-10T00:00:00Z` | seed 0
- Goldset `artifacts/harness/goldset.jsonl` sha256 `af40cb0da26fa78c5c477747357beec518053b01a2daa7e875338acc47c95612` with 234 items (26 tail)
- Embedding backend `auto` model `paraphrase-multilingual-MiniLM-L12-v2` | harness v1

## Metrics

| bucket | items | w/relevant | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|---|
| overall | 234 | 234 | 0.188 | 0.577 | 0.726 | 0.354 |
| stratum:content_grounded | 160 | 160 | 0.250 | 0.719 | 0.881 | 0.444 |
| stratum:cross_lingual | 38 | 38 | 0.000 | 0.105 | 0.184 | 0.045 |
| stratum:role_query | 36 | 36 | 0.111 | 0.444 | 0.611 | 0.280 |
| tail | 26 | 26 | 0.038 | 0.385 | 0.538 | 0.204 |

### Per channel

| channel | items | w/relevant | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|---|
| bm25 | 234 | 234 | 0.179 | 0.568 | 0.726 | 0.342 |
| graph | 234 | 234 | 0.201 | 0.560 | 0.726 | 0.353 |
| trigger | 234 | 234 | 0.175 | 0.577 | 0.726 | 0.345 |
| vector | 234 | 234 | 0.167 | 0.585 | 0.726 | 0.342 |

### Channel attribution (170 items with a relevant result)

| method | first-relevant results | share |
|---|---|---|
| bm25 | 105 | 0.618 |
| vector | 161 | 0.947 |
| graph | 49 | 0.288 |
| trigger | 32 | 0.188 |

## Live-agreement sanity

- Sampled 20 of 234 items (seed 0, top-5): agreement_rate 1.000
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

## Limitations

- Relevance is the goldset's judgement, frozen at build time: `content_grounded` items inherit replay's IDF-containment label over the consuming trace, `cross_lingual` items inherit the paraphrase query's own top-k (so they measure jargon-vs-paraphrase agreement, not absolute truth), and `role_query` items are curated against active `level:schema` triggers.
- A relevant node that retrieval never returns counts as a **miss**, not as an unevaluated item: it enters the denominator with no hit and no reciprocal rank. The run therefore measures ranking and candidate collection together and cannot tell one failure from the other.
- Per-channel buckets re-order only what retrieval already returned, so they bound a channel's ranking power, not its recall.
