# End-to-end retrieval-harness baseline

Generated: 2026-08-17T18:45:50Z | snapshot `/home/sfx/.cache/living-memory-harness/snapshot.sqlite3` | commit `44fc0a0ba658516c90c2c7fee24c457348dd68ef`

## Snapshot and goldset

- Source DB: `/home/sfx/.local/share/living-memory/global.sqlite3`
- Snapshot sha256 `d46f1b845adc3d806ec5cb7f593438f723bdf2b7c491c0f47aa92e4ef20bd75d` captured 2026-08-17T17:59:08Z
- Snapshot rows: active_nodes 12862, connections 142906, nodes 16683, recall_events 54777
- recall_events span 2026-05-14T20:05:23Z .. 2026-08-17T17:58:08Z | cutoff `2026-06-10T00:00:00Z` | seed 0
- Goldset `artifacts/harness/goldset.jsonl` sha256 `af40cb0da26fa78c5c477747357beec518053b01a2daa7e875338acc47c95612` with 234 items (26 tail)
- Embedding backend `auto` model `paraphrase-multilingual-MiniLM-L12-v2` | harness v1

## Metrics

| bucket | items | w/relevant | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|---|
| overall | 234 | 234 | 0.175 | 0.526 | 0.688 | 0.324 |
| stratum:content_grounded | 160 | 160 | 0.231 | 0.694 | 0.900 | 0.422 |
| stratum:cross_lingual | 38 | 38 | 0.000 | 0.105 | 0.184 | 0.057 |
| stratum:role_query | 36 | 36 | 0.111 | 0.222 | 0.278 | 0.167 |
| tail | 26 | 26 | 0.038 | 0.077 | 0.077 | 0.046 |

### Per channel

| channel | items | w/relevant | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|---|
| bm25 | 234 | 234 | 0.158 | 0.573 | 0.688 | 0.321 |
| graph | 234 | 234 | 0.154 | 0.521 | 0.688 | 0.306 |
| trigger | 234 | 234 | 0.162 | 0.526 | 0.688 | 0.315 |
| vector | 234 | 234 | 0.175 | 0.538 | 0.684 | 0.333 |

### Channel attribution (161 items with a relevant result)

| method | first-relevant results | share |
|---|---|---|
| bm25 | 103 | 0.640 |
| vector | 149 | 0.925 |
| graph | 31 | 0.193 |
| trigger | 32 | 0.199 |

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

<!-- ==== everything above this marker is verbatim `render_markdown` output; everything below
     is appended by the baseline-report node and is not regenerated by the harness ==== -->

## Reproduction

```bash
python3 -m living_memory.retrieval_harness run \
  --snapshot ~/.cache/living-memory-harness/snapshot.sqlite3 \
  --goldset artifacts/harness/goldset.jsonl \
  --agreement-sample 20 \
  --cutoff 2026-06-10T00:00:00Z \
  --report artifacts/harness/baseline.json \
  --markdown artifacts/harness/baseline.md
```

Run with `PYTHONPATH=src` and with `LIVING_MEMORY_EMBEDDING_BACKEND` **unset** — the
baseline uses the real sentence-transformers backend (`embedding_backend: auto`), never
the hash backend.

Two seeds appear in the provenance chain and they are not the same number:
`provenance.seed = 0` is this run's live-agreement sample seed, while the goldset was
built with seed `20260817` (recorded per item under `provenance.build.seed`).

### Snapshot integrity, verified not assumed

The frozen snapshot was already present, so it was verified rather than re-created:
`sha256sum` of `~/.cache/living-memory-harness/snapshot.sqlite3` is
`d46f1b845adc3d806ec5cb7f593438f723bdf2b7c491c0f47aa92e4ef20bd75d`, which matches both
its sidecar manifest and the `provenance.build.snapshot_sha256` recorded in every one of
the 234 goldset items. **There is no snapshot drift**, and all 433 distinct
`relevant_node_ids` in the goldset resolve to active (`decayed = 0`) nodes in that
snapshot — 433 of 433, checked over a `mode=ro` connection. The snapshot hashed
identically again after all three runs: the harness works on a temporary copy and leaves
the frozen bytes untouched. The live DB at
`~/.local/share/living-memory/global.sqlite3` was never opened by this node.

## Determinism proof

The harness was invoked three times over the same snapshot and the same goldset. The
`metrics` block was compared as **raw bytes sliced out of the written JSON files** (from
`\n  "metrics"` up to `\n  "provenance"`, the neighbouring key under `sort_keys=True`),
not by re-serializing parsed objects — re-serialization would hide an ordering bug in the
writer.

| run | `--cutoff` | report path | `metrics` bytes | sha256 of the `metrics` block |
|---|---|---|---|---|
| 1 | unset | `artifacts/harness/baseline.json` (superseded) | 3613 | `e942ed41c9a2dc3098ca28735d48443f71b61d29dd2e15837282af34f926d007` |
| 2 | unset | `/tmp/baseline-run2.json` | 3613 | `e942ed41c9a2dc3098ca28735d48443f71b61d29dd2e15837282af34f926d007` |
| 3 | `2026-06-10T00:00:00Z` | `artifacts/harness/baseline.json` (final) | 3613 | `e942ed41c9a2dc3098ca28735d48443f71b61d29dd2e15837282af34f926d007` |

All three are byte-identical. So are the `agreement` (1454 B,
`11c24ebf5897f1c8…`), `definitions` (2449 B, `38af1575ed8732c9…`) and `runs`
(1152331 B, `ebbb39b514c48187…`) blocks — the per-result channel scores of all 234
queries reproduce exactly, not merely the aggregates. The **only** field that differs
between runs is `provenance.generated_at`, plus `provenance.cutoff` in run 3, which is
recorded in provenance and does not enter retrieval. Run 3 is therefore proof of
something slightly stronger than a repeat: changing a provenance-only flag leaves the
metrics bit-for-bit unchanged.

This is the phase-1 gate's anchor: any metric movement after chunking is attributable to
the change, not to harness noise.

## Notes for the downstream phase-1 gate

### hit@10 is a recall ceiling here, not a ranking metric

224 of the 234 items return exactly 10 results (the rest return 7–30). Re-ordering a list
of ≤10 items cannot change which of them fall inside the top 10, so **hit@10 is identical
at 0.688 for the live order and for the bm25, graph and trigger buckets** — it is
measuring "was the relevant node collected at all", and 0.688 = 161/234 is exactly the
count of items where a relevant node appears anywhere in the returned list. Vector's
0.684 (160/234) differs by a single item, one of the ten with `max_results > 10`. Read
hit@1 / hit@5 / MRR for ranking quality; read hit@10 as candidate coverage.

### Where the vector channel stands

| | hit@1 | hit@5 | MRR | present in first-relevant result |
|---|---|---|---|---|
| vector | **0.175** (best) | 0.538 | **0.333** (best) | **149/161 = 0.925** |
| bm25 | 0.158 | **0.573** (best) | 0.321 | 103/161 = 0.640 |
| trigger | 0.162 | 0.526 | 0.315 | 32/161 = 0.199 |
| graph | 0.154 | 0.521 | 0.306 | 31/161 = 0.193 |

Vector is the dominant channel by participation — it is a contributing method in 92.5% of
the items where retrieval surfaced a relevant node — and it already wins hit@1 and MRR.
But bm25 beats it at hit@5 (0.573 vs 0.538), i.e. lexical overlap currently rescues
mid-ranked results that the truncated embedding does not. Because vector participates in
nearly everything, a change to it moves the overall number more than a change to any
other channel; that cuts both ways, and is why the byte-identical determinism above
matters.

### The tail subset is the number chunking must move

26 items are tail (25 `role_query`, 1 `cross_lingual`) — computed mechanically by the
tail rule restated verbatim above, never hand-picked.

- tail hit@1 **0.038**, hit@5 **0.077**, hit@10 **0.077**, MRR **0.046**
- against overall hit@5 0.526 — the tail is ~7× worse

The decisive detail is that **hit@10 equals hit@5 on the tail**: in 24 of the 26 tail
items the relevant node is not merely ranked badly, it never enters the returned
candidate list at all (the 2 that do land at ranks 1 and 5). This is a
candidate-collection failure, not a ranking failure. No amount of weight re-calibration —
phase 2 — can recover a node that retrieval never proposes; only putting the post-128-token
content into a vector that can match, i.e. phase-1 chunking, can. Expect the gate to move
tail hit@5 up from 0.077 (2/26) while overall hit@5 holds at or above 0.526.

For completeness, the other weak stratum is `cross_lingual` (hit@5 0.105, relevant node
returned anywhere in only 7 of 38 items). That is the measured jargon class
(поревьювь / мердж-реквест / апрув). Chunking addresses it only where it overlaps the
tail (1 item); the rest is a vocabulary problem and is explicitly out of scope for
phase 1.
