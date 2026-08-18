# Query anchors: the leak-free temporal-holdout run (final)

Two arms of the real retrieval path over **one snapshot, one goldset, one code
path**, differing only in whether the graph channel's entry from query space is
switched on. Anchors were built strictly from events **before** the cutoff; the
scored queries come strictly from **at or after** it.

This run re-scores the same sha-pinned inputs as the previous one, with two
changes landed by sibling nodes in between:

* the **graph-floor fix** (`881c125`) — an anchor activation can no longer lower
  a candidate's score;
* the **calibrated match entry** (`ed0c61d`) — `ANCHOR_MATCH_COSINE_THRESHOLD`
  0.80 → **0.60**, `ANCHOR_MATCH_LIMIT` 5, chosen on a temporally disjoint
  calibration set that never saw this goldset.

Machine-readable companions: `eval-with-anchors.json`, `eval-without-anchors.json`
(same schema, `provenance.anchor_seeding` is the one bit that differs),
`latency.json`.

---

## 1. Setup, and the exact bytes

| what | value |
| --- | --- |
| scored snapshot (anchored) | `/tmp/anchor-eval/snap-anchored.sqlite3`, sha256 `5201b10682fcc0e5…` — a sqlite-backup-API copy of the anchor-free base snapshot (`5f82312ec3566f99…`) plus the anchor tables, nothing else |
| cutoff | **2026-07-15T00:00:00Z** |
| anchor corpus | **3,071 anchors / 7,510 edges**, all live, 2.445 edges per anchor |
| holdout goldset | `/tmp/anchor-eval/goldset-holdout-20260715.jsonl`, sha256 `816c365cbfa88d47…`, **343 items** (content_grounded 269, cross_lingual 38, role_query 36) |
| frozen goldset | `artifacts/harness/goldset.jsonl`, sha256 `af40cb0da26fa78c…`, **234 items** |
| match entry | `ANCHOR_MATCH_COSINE_THRESHOLD = 0.60`, `ANCHOR_MATCH_LIMIT = 5` |
| encoder | `paraphrase-multilingual-MiniLM-L12-v2` |
| live-agreement | 1.000 on 20 sampled items, 0 divergences, **in both arms** |

Both arms report the identical `snapshot_sha256` and `goldset_sha256` in their
`provenance` block. The live database was never written; every read happened on
a working copy.

### Reproduce

```bash
# inputs are sha-pinned; assert before scoring
sha256sum /tmp/anchor-eval/snap-anchored.sqlite3   # 5201b106…
sha256sum /tmp/anchor-eval/goldset-holdout-20260715.jsonl  # 816c365c…
sha256sum artifacts/harness/goldset.jsonl          # af40cb0d…

# if /tmp was cleared, rebuild first:
python3 -m living_memory.retrieval_harness snapshot \
    --source-db /tmp/anchor-est/snap.sqlite3 \
    --snapshot-out /tmp/anchor-eval/snap-anchored.sqlite3
python3 scripts/backfill_query_anchors.py backfill \
    --db /tmp/anchor-eval/snap-anchored.sqlite3 \
    --backup /tmp/anchor-eval/backup-pre-backfill.sqlite3 \
    --until 2026-07-15T00:00:00Z --json /tmp/anchor-eval/backfill.json
python3 scripts/backfill_query_anchors.py verify \
    --db /tmp/anchor-eval/snap-anchored.sqlite3 --json /tmp/anchor-eval/verify.json

# the four arms (--no-anchors is the ONLY difference between arms)
for g in /tmp/anchor-eval/goldset-holdout-20260715.jsonl artifacts/harness/goldset.jsonl; do
  for flag in "" --no-anchors; do
    python3 -m living_memory.retrieval_harness run \
        --snapshot /tmp/anchor-eval/snap-anchored.sqlite3 --goldset "$g" \
        --cutoff 2026-07-15T00:00:00Z --seed 20260818 --report <out.json> $flag
  done
done

# cost
python3 scripts/anchor_latency_bench.py \
    --snapshot /tmp/anchor-eval/snap-anchored.sqlite3 \
    --cutoff 2026-07-15T00:00:00Z --out artifacts/anchors/latency.json
```

---

## 2. Leak audit (the run is evidence only if this passes)

Read directly off the scored snapshot:

| check | result |
| --- | --- |
| newest anchor `first_seen` / `created_at` / `updated_at` | **2026-07-14T13:57:33Z** — strictly before the cutoff |
| newest anchor `last_matched_at` | 2026-07-14T13:57:33Z |
| newest anchor edge `created_at` | 2026-07-14T13:57:33Z |
| anchors with `first_seen >= cutoff` | **0 of 3,071** |
| anchors with `created_at >= cutoff` | **0 of 3,071** |
| anchors with `updated_at >= cutoff` | **0 of 3,071** |
| anchor edges with `created_at >= cutoff` | **0 of 7,510** |
| anchor edges with `updated_at >= cutoff` | **0 of 7,510** |
| `content_grounded` source events | all 269 at/after the cutoff |
| items carrying no timestamp | 74 (cross_lingual 38, role_query 36) — bound by the fingerprint test instead |
| curated items that are exact repeats of a pre-cutoff anchor | **0 of 74** |

The 74 curated items were **frozen**, not rebuilt. Both arms read the identical
file, so anchors cannot define their own ground truth.

---

## 3. Holdout results — 343 items, both arms

`without -> with`.

| bucket | items | hit@1 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| **overall** | 343 | 0.2507 → 0.2507 | 0.6122 → 0.6122 | 0.7493 → 0.7493 | 0.4035 → **0.4040** (+0.0005) |
| content_grounded | 269 | 0.3048 → 0.3048 | 0.7063 → 0.7063 | 0.8476 → 0.8476 | 0.4708 → 0.4714 |
| **cross_lingual** | 38 | 0.0000 → 0.0000 | **0.1053 → 0.1053 (+0.0000)** | 0.1842 → 0.1842 | 0.0448 → 0.0448 |
| **role_query** | 36 | **0.1111 → 0.1111 (+0.0000)** | 0.4444 → 0.4444 | 0.6111 → 0.6111 | 0.2797 → 0.2797 |
| tail | 26 | 0.0385 → 0.0385 | 0.3846 → 0.3846 | 0.5385 → 0.5385 | 0.2039 → 0.2039 |

Paired, per item: **1 improved, 2 regressed, 340 unchanged.**
Mean MRR delta +0.000481, 95% bootstrap CI **[−0.001472, +0.003397]**; mean
hit@5 delta +0.000000, CI [−0.008746, +0.008746]. Nine items' ranked lists
changed at all. `role_query`: **0 of 36 lists changed**. `cross_lingual`:
1 of 38 changed, and not across a metric boundary.

### The monotonicity fix holds, and the regressions are displacement

Across all 343 items, in both regressions and everywhere else, **no candidate
scored lower with anchors on than with anchors off**. The invariant the
graph-floor fix was built to restore is intact; the previous run's "54 lost
score, median 8.1% relative loss" is gone.

The two rank regressions are a different thing — a *non-relevant* anchor target
gaining score and overtaking the relevant node:

| item | relevant rank off → on | cause |
| --- | --- | --- |
| `content_grounded-88849ff6a5bc` | 2 → 3 | `01KT9N1N1J24DNX7Q2MGAT8FTT` 0.9061 → 1.0664 (gained graph from an anchor seed), rank 6 → 2. No candidate lost score; no newcomer. |
| `content_grounded-b7206fc167d1` | 5 → 7 | `01KT41WXXK2NTXJ6W8CMJY2828` 0.7868 → 1.1327, rank 8 → 2, **plus** a newcomer `01KT3QKNDD484XYYK2BEPVW34K` admitted at rank 3. No candidate lost score. |

This distinction matters and is worth stating plainly: *"an anchor never lowers
a candidate's score"* is an invariant a fix can guarantee, and it holds.
*"an anchor never lowers a relevant node's rank"* is not, and cannot be, once
the mechanism is allowed to promote anything at all. What these two items
measure is **edge quality** — the anchor promoted the wrong node — not a
scoring defect.

### Fingerprint-disjoint holdout (the only subset that can carry a claim)

| bucket | items | hit@1 | hit@5 | MRR |
| --- | --- | --- | --- | --- |
| exact repeat | **1** | 0.0000 → 0.0000 | **0.0000 → 1.0000** | 0.1111 → **0.5000** |
| **fingerprint-disjoint** | **342** | 0.2515 → 0.2515 | **0.6140 → 0.6111** | **0.4044 → 0.4037** |
| content_grounded : disjoint | 268 | 0.3060 → 0.3060 | 0.7090 → 0.7052 | 0.4721 → 0.4713 |
| cross_lingual : disjoint | 38 | 0.0000 → 0.0000 | 0.1053 → 0.1053 | 0.0448 → 0.0448 |
| role_query : disjoint | 36 | 0.1111 → 0.1111 | 0.4444 → 0.4444 | 0.2797 → 0.2797 |

**The entire aggregate gain is the single exact-repeat item.** On the 342-item
fingerprint-disjoint subset — the only one that can carry a generalization
claim — anchors improved **0 items and regressed 2**, for −0.0029 hit@5 and
−0.0007 MRR. The mechanism currently helps where the query is a repeat of a
query it was trained on, and nowhere else.

### Fresh-node visibility — a live gate now, and it fires

The lower match floor took this slice from **6 items to 57** (83 fresh nodes),
so the "rich get richer" pressure it guards against is now actually observable
rather than a formality.

| arm | items | fresh nodes | hit@1 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| without anchors | 57 | 83 | 0.1754 | **0.6842** | 0.8070 | **0.3775** |
| **with anchors** | 57 | 83 | 0.1754 | **0.6667** | 0.8070 | **0.3765** |

Per item, first fresh-node rank: **0 better, 1 worse, 56 unchanged.** The one
that moved is `content_grounded-b7206fc167d1`, rank **5 → 7** — across the
hit@5 boundary, which is the whole −0.0175. The cause is crowding, confirmed
directly: two anchor targets (`01KT41WXXK2NTXJ6W8CMJY2828`,
`01KT3QKNDD484XYYK2BEPVW34K`) were promoted above the fresh node, and **no
candidate lost score**. This is the anchor preferring what it already knows over
something new — exactly the contour the slice exists to detect.

On the frozen goldset the same slice is 12 items and **identical in both arms**
(hit@5 0.4167, MRR 0.3153).

---

## 4. The frozen shipped goldset — the required floor

`artifacts/harness/goldset.jsonl` (234 items), same snapshot, both arms.

| bucket | items | hit@1 | hit@5 | MRR |
| --- | --- | --- | --- | --- |
| **overall, without anchors** | 234 | 0.1880 | **0.5769** | **0.3540** |
| **overall, with anchors** | 234 | **0.1923** | **0.5812** | **0.3596** |
| content_grounded | 160 | 0.2500 → 0.2562 | 0.7188 → 0.7250 | 0.4441 → 0.4523 |
| cross_lingual | 38 | 0.0000 → 0.0000 | 0.1053 → 0.1053 | 0.0448 → 0.0448 |
| role_query | 36 | 0.1111 → 0.1111 | 0.4444 → 0.4444 | 0.2797 → 0.2797 |
| exact repeat | 129 | 0.2558 → 0.2636 | 0.7597 → 0.7674 | 0.4535 → 0.4636 |
| fingerprint-disjoint | 105 | 0.1048 → 0.1048 | 0.3524 → 0.3524 | 0.2317 → 0.2317 |

The anchor-free arm reproduces the phase-1 numbers **exactly** (0.5769 / 0.3540).
The anchors-on arm is now **above** them: **0.5812 ≥ 0.5769** and
**0.3596 ≥ 0.3540**. Paired: **3 improved, 0 regressed, 231 unchanged.**

The previous run's −0.0043 / −0.0054 regression is fixed, and the mechanism that
caused it is gone rather than masked. Note where the gain sits: all 3 improved
items are exact repeats, and the fingerprint-disjoint 105 are **byte-identical**
between arms. 129 of 234 items on this goldset are exact repeats and
content_grounded coverage is 91.25%, which is why this goldset moves at all
while the holdout does not.

---

## 5. What bounds the two target strata — it is no longer the floor

### Coverage went up 8×; quality did not follow

| stratum | items | matched @0.80 (previous) | matched @0.60 (this run) | share |
| --- | --- | --- | --- | --- |
| content_grounded | 269 | 12 | **96** | 35.7% |
| **cross_lingual** | 38 | 0 | **4** | 10.5% |
| **role_query** | 36 | 1 | **7** | 19.4% |
| all | 343 | 13 (3.8%) | **107** | **31.2%** |

On live traffic the same change took the match rate from **0.52% to 37.76%** of
384 distinct holdout queries (`latency.json`). The floor was genuinely binding,
and it is no longer binding.

Nearest in-scope anchor cosine per eval query, **unfloored**:

| stratum | min | p50 | p90 | max | ≥ 0.60 |
| --- | --- | --- | --- | --- | --- |
| content_grounded | 0.341 | 0.558 | 0.710 | 1.000 | 96 |
| **cross_lingual** | 0.351 | 0.455 | 0.589 | **0.663** | **4** |
| role_query | 0.354 | 0.529 | 0.697 | 0.912 | 7 |

### The new bound: anchor edge yield

Of the items that now match, how many have a relevant node among the matched
anchors' **edge targets** — i.e. how often does what the anchor found actually
carry the answer?

| stratum | matched | matched **and** an anchor edge reaches a relevant node | yield |
| --- | --- | --- | --- |
| content_grounded | 96 | 39 | 40.6% |
| **cross_lingual** | **4** | **0** | **0.0%** |
| **role_query** | **7** | **0** | **0.0%** |
| all | 107 | 39 | 36.4% |

**Zero of the 11 matched cross_lingual and role_query items has an anchor edge
into a relevant node.** That is why those two strata are byte-identical between
arms despite matching 11 times: the anchor fires, brings its targets, and none
of them is an answer. No threshold, limit or weight can change this — the edges
point elsewhere.

### Reachability ceilings, measured on this snapshot

Share of items for which an anchor holding a **live edge to a relevant node**
is reachable at all — the hard bound on what the entry could ever contribute:

| stratum | at the shipped entry (top-5, ≥ 0.60) | top-10 nearest, **no floor** | oracle: **any** in-scope anchor |
| --- | --- | --- | --- |
| content_grounded | 0.1450 | 0.3086 | 0.4647 |
| **cross_lingual** | **0.0000** | **0.1053** | 0.6316 |
| **role_query** | **0.0000** | **0.1111** | 0.1667 |

The oracle column is the interesting one. For `cross_lingual` the corpus *does*
hold a qualifying anchor for 24 of 38 items (63%) — but never among the 10
nearest by cosine, because a Russian paraphrase and its English source query are
0.35–0.66 apart in this encoder. The anchor exists; the matcher cannot find it.
For `role_query` the corpus barely holds one at all: 6 of 36, oracle included.

### Structural causes, ranked by what they cost

1. **Anchor edge yield on the two target strata is 0.0%.** 11 matches, 0 with an
   edge to a relevant node. This is now the binding constraint and it replaces
   the match floor as the answer.
2. **For `cross_lingual`, cosine cannot retrieve the anchor that exists.** Oracle
   0.632 vs top-10-no-floor 0.105 — an 0.53 gap that is entirely matcher recall.
   The premise "the operator's queries are written in one language and jargon,
   so query↔query beats query↔content" holds *within* a language; these goldset
   items are cross-language pairs by construction, which is the one case the
   premise excludes.
3. **`cross_lingual` scores agreement, not truth.** Its relevance is the
   paraphrase query's own top-k, and 14 of 38 labels moved across one day of new
   memory when rebuilt. Freezing them makes the A/B valid; nothing makes the
   metric ground truth.
4. **The curated strata are 38 and 36 items.** The smallest resolvable move is
   1/38 = 0.026 and 1/36 = 0.028; the effects being hunted are 2–4 items. The run
   has no power to resolve what it was asked to resolve.

---

## 6. The trigger channel after anchors

Measured on `role_query`. Anchors matched 7 of 36 items and changed **0**
rankings, so these tables are **identical in both arms and on both goldsets**:

| method set of the first relevant result | items |
| --- | --- |
| bm25+graph+vector | 12 |
| bm25+graph+trigger+vector | 3 |
| bm25+vector | 3 |
| bm25+trigger+vector | 1 |
| graph+trigger | 1 |
| graph+trigger+vector | 1 |
| graph+vector | 1 |
| **total with a relevant hit** | **22 of 36** |

- trigger present in **6 of 22 (27.3%)**; graph present in **18 of 22 (81.8%)**
- **trigger ∩ graph = 5**; trigger without graph = 1
- **trigger-unique = 0 of 22.** Overall across all 343 items: trigger-unique
  **0 of 257** in both arms. Anchors did change *attribution* without changing
  order: the first-relevant result carries a graph score on **89** items with
  anchors vs **62** without (+27), and vector-unique falls from **36 to 27**
  (the 9 of those 27 that had been vector-only now also carry graph). The
  ranked lists behind these counts are unchanged — a candidate acquiring an
  anchor activation is what moved, not its position.
- **anchor ∩ trigger overlap: 3 items.** Three of the six trigger-carrying hits
  are also anchor-matched — but since anchors changed no role_query ranking, the
  overlap is currently latent, not competitive. It is the number to re-read after
  edge yield improves, because that is when the two would start duplicating work.

### Numeric recommendation on the 0.95 × 1.8 boost — **do not retire it on these numbers**

Counterfactuals computed from the recorded per-result scores, `role_query`, n = 36,
identical in all four arms:

| configuration | hit@1 | hit@5 | MRR |
| --- | --- | --- | --- |
| as shipped | 0.1111 | 0.4444 | 0.2797 |
| **`SCHEMA_TRIGGER_BOOST` ×1.8 removed** (divide every trigger-carrying schema result's final score by 1.8, re-rank) | 0.1111 | 0.4444 | **0.2751 (−0.0046)** |
| every trigger-only result dropped (upper bound on removing the channel) | 0.1111 | **0.4722 (+0.0278)** | 0.2820 (+0.0023) |

**This reverses the previous run's recommendation.** That run measured the ×1.8
removal at exactly 0 cost on all three metrics and recommended retiring it. On
this build the cost is **0 items at hit@1 and hit@5 but −0.0046 MRR** — the
multiplier does reorder a relevant result upward within the top-5 on at least one
item. The honest reading is: retiring ×1.8 is *nearly* free, not free, and the
evidence no longer supports doing it blind.

Two bounds, unchanged:

1. The `0.95 + 0.05 × overlap` **base** cannot be undone from recorded scores — it
   enters through `base_score = max(weighted_sum, trigger_score)` before the
   feedback multiplier, and the report does not carry `weighted_sum`. This covers
   the ×1.8 half only.
2. n = 36. A −0.0046 MRR result on 36 items is one item moving one rank.

If the boost is retired, it should be `SCHEMA_TRIGGER_BOOST = 1.0` plus a re-run
of both arms on both goldsets, expecting **−0.0046 MRR on role_query and nothing
else** — and any other change treated as evidence the counterfactual missed
something.

**Do not remove the channel.** Its unique contribution to first-relevant hits is
0, but that measures ranking of what retrieval returned; the trigger channel is
also a *candidate producer*, and 1 of 22 role_query hits (`graph+trigger`)
reached the list with no bm25 and no vector evidence at all.

On this snapshot **164 of 433 active schemas (37.9%) carry triggers of ≥ 5
distinct tokens**, and the channel needs ≥ 50% token overlap to fire, so a
5-token trigger needs 3 of its tokens present in a short query.

---

## 7. Cost

Re-measured on this build, because the match rate changed by 70× and the
previous +0.738 ms figure no longer describes it. Benchmark:
`scripts/anchor_latency_bench.py` (landed in the repo this time; the previous
script lived in /tmp and is gone). Full record: `latency.json`.

| figure | value | budget |
| --- | --- | --- |
| anchor scan, p50 / p95 / max | **0.560 / 1.036 / 1.126 ms** | — |
| anchor scan, no anchor matched | 0.436 ms p50 | — |
| anchor scan, anchor matched | 0.688 ms p50 | — |
| **paired `memory_recall` p50 delta, all strata** | **+3.777 ms** | **+5 ms → PASS** |
| paired p50 delta, unmatched holdout | +1.111 ms | +5 ms |
| paired p50 delta, matched holdout | +5.867 ms | — |
| paired p50 delta, repeat worst case | +5.210 ms (p95 +22.95, max +36.02) | — |
| **traffic-weighted p50 delta** (37.76% matched) | **+2.907 ms** | **+5 ms → PASS** |
| **pooled p50 delta** | **+7.093 ms** | **+5 ms → EXCEEDS** |
| pooled p95 delta | −3.573 ms | — |

**The pooled row exceeds the budget and that is reported, not hidden.** Its
query mix is deliberately adversarial: 26 of the 42 measured queries carry a
matching anchor (the `repeat` stratum is every-query-has-an-anchor by
construction), against **37.76%** of real distinct holdout queries. Pooling a
median across that mix over-weights the expensive case. The paired figure
(+3.777 ms) is what answers "what did anchors cost this query", and the
traffic-weighted figure (+2.907 ms) is what an operator would see. Both are
inside +5 ms; the pooled number is an upper bound on an unrepresentative mix.

The hop is not overhead: a matched anchor opens a BFS from seeds no other
channel produced. Root count is unchanged — anchor seeds compete for the same
`GRAPH_SEED_LIMIT` (50) slots as every other seed.

Match-rate diagnostic, 384 distinct holdout queries: **145 cleared the floor
(37.76%)**; best-anchor cosine p50 0.564, p90 0.737, p95 0.794, p99 0.856.

---

## 8. What the measurer does next

**Regenerate the goldset from post-grounding traffic**, and fix edge yield —
in that order of evidence, opposite order of effect.

1. **Edge yield is the thing to work on, not the threshold.** 0 of 11 matched
   cross_lingual/role_query items has an anchor edge into a relevant node. The
   next intervention is on *which nodes an anchor links to* (top-3 recorded
   results is the current rule) and on how many consumptions an anchor needs
   before it is allowed to seed — not on the match entry, which is now
   calibrated and demonstrably not binding.
2. **For `cross_lingual`, the matcher is the gap, not the corpus** — oracle
   0.632 vs top-10-no-floor 0.105. Either the goldset's cross-language framing
   is the wrong test for a query↔query mechanism, or anchor matching needs a
   path that is not raw cosine in this encoder. Decide which before spending
   another scored run on it.
3. **Regenerate `content_grounded` from post-grounding traffic.** The live
   grounded write path has been running since 2026-08-18 ~18:00; every query it
   has anchored since is traffic the operator actually asked *after* grounding
   became the credit signal, and none of it is in this run's training window.
   Recompute the anchor-free baseline on the new file — the 0.5769 / 0.3540 pair
   belongs to the 234-item shipped goldset and to nothing else.
4. **Grow `cross_lingual` and `role_query` past 38 / 36.** `seed-queries.json`
   records that 60+ cross_lingual pairs were authored and filtered to 38. At 38
   items the smallest resolvable move is 0.026 and the effect being hunted is
   2–4 items. This is cheaper than another scored run whose result cannot be read.
5. **Keep the fresh-node slice as a gate.** It fired for the first time this run
   (1 of 57 items, rank 5 → 7). At 6 items it could not have. Any change that
   raises coverage further must re-read it.
