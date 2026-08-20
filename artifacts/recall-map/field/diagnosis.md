# Recall-map field diagnosis — why the live maps say `public (73)` and carry no gist

**Measured 2026-08-20 from the first field deployment of the mid-session inject channel.**
Read-only throughout: nothing was written on host `alt`, nothing under `~/p/ae` was touched.

Every number below is computed over the **train and eval splits only**. The holdout
split is listed in full in [`manifest.json`](manifest.json) so it can be replayed, and
is neither analysed nor quoted here — not one holdout label, medoid, query or count
appears in this document.

---

## 0. What was read, and how it was joined

| source | what it proves | provenance |
|---|---|---|
| 18 `stream_inject/*.jsonl` journals under `~/p/ae/projects/game/state/goals/**` | what the *agent* saw, and the consumption evidence no server table holds | copied read-only off `alt`; a recursive sweep finds 18 journals, not the 14 the intent cites — a shallow glob misses the `archive/unlisted-children-*/children/**` tree |
| `recall_events.recall_map` on `alt`'s living-memory store | what the *server actually built* — the only place the missing medoid `example` can be confirmed rather than inferred from a renderer | `/home/user/.local/share/living-memory/global.sqlite3` (located from the `living-memory-server --db` argv), streamed with its WAL to this workstation and frozen via the sqlite backup API |
| the corpus itself (nodes, `query_anchors`, `nodes_fts_vocab`) | stage attribution and the document frequencies behind the label statistics | same frozen snapshot, sha256 `12932fd8…f42a`, 2 573 nodes |
| local live-e2e journals + this workstation's store | the contrastive control (§6) | `/tmp/ae-stream-live-e2e.*/journal.jsonl`; the e2e ran against the local store, not `alt` |

The join is exact: every probe that reached the server (`outcome != dedup`) matched a
persisted `recall_events` row on the query string — **81 of 81** in train+eval, 106 of
106 overall. The extractor is `scripts/recall_map_field_extract.py`; it imports
`recall_map`'s own functions rather than reimplementing them, so an attribution error
here would be a bug in that module, not a judgement call.

### Train+eval channel accounting (14 sessions, 14 journals)

| stage of the channel | count |
|---|---:|
| probes journalled | 142 |
| suppressed client-side before any recall (`dedup`) | 61 |
| recalls that reached the server | 81 |
| **server built and persisted a map with clusters** | **77** |
| server returned the curtailed marker instead | 4 |
| clusters in those maps | 294 |
| **clusters carrying a medoid `example` (the gist)** | **0** |
| **injects the agent actually saw** | **29** |
| consumptions | 1 |

The gap between 77 delivered maps and 29 injects is not noise. It is §5, and it is the
most expensive consequence of the label defect.

### Splits

`manifest.json` carries 119 replayable cases split **by cache key** — `(scope, normalized
task)`, the builder's own structure-cache key — because two cases under one key are two
looks at one cached structure and straddling a key would leak exactly what the holdout
exists to withhold.

| split | field cases (`project:game`, host `alt`) | contrastive controls (`project:ae`, local) |
|---|---:|---:|
| train | 49 (4 keys) | 13 |
| eval | 32 (3 keys) | 0 |
| holdout | 25 (3 keys) | 0 |

The controls are train by construction — §6 quotes them, and a case the diagnosis reads
cannot also be held out. They are a different corpus on a different snapshot and must
never be pooled with the field cases in a metric. The splits are disjoint in cases and in
cache keys; they are **not** disjoint in corpus, since all field keys read one
`project:game` store. That is the point: a rule fitted on train meets unseen tasks,
unseen pools and an unseen label set on holdout, over a corpus it has already seen.

---

## 1. Every delivered label, attributed to its cascade stage

Attribution is not inferred from the shape of a label. Each case was **replayed through
the real builder** on the frozen snapshot — the recall re-run, `last_residual` fed to
`RecallMapBuilder` — and where a replay arm reproduces a delivered label, that arm's own
`MapCluster.stage` is the answer. Two arms, because the field maps were not all built the
same way: `cold` is a fresh builder per case (the full four-stage cascade), `warm` shares
one builder per cache key in the order the field ran them (the structure-cache path).

| attribution route | clusters |
|---|---:|
| replayed cold — the builder's own `stage` | 189 |
| replayed warm — the builder's own `stage` | 79 |
| fallback: medoid precedence, corroborated (label reconstructs exactly, or the medoid carries no structural key, no context path and no live anchor edge, so precedence leaves only stage 4) | 19 |
| fallback: medoid precedence, **flagged uncertain** (label does not reconstruct) | 7 |
| **total delivered clusters** | **294** |

The 7 uncertain ones are six `server` and one `showcase`. In each the medoid carries a
structural key whose normalization is nothing like the delivered label, which is the
signature of the `_recount` drift described in §2 — so stage 2 is the likely owner of all
seven, but the manifest keeps them marked (`stage_source: "medoid_precedence"`,
`label_match: "mismatch"`) rather than rounding them into the total.

### The result

| stage | clusters | share |
|---|---:|---:|
| 2 — **path collapse** | **267** | **90.8 %** |
| 4 — chunk-embedding + c-TF-IDF | 20 | 6.8 % |
| 1 — structural context keys | 7 | 2.4 % |
| 3 — query anchors | 0 | 0 % |

**Nine out of ten delivered field clusters are stage-2 clusters, and stage 2 names a
cluster after one directory segment.** Stage 3 never delivered a single cluster in the
field window. The per-inject table is in the appendix; every one of the 29 injects is
attributed there.

---

## 2. Why stage 2 wins, and why its labels are one word

Three facts compose into the observed degeneracy.

**(a) The pool is path-dominated.** Classifying all 15 926 pool members across the 81
replayed train+eval pools by the cascade's own precedence:

| what the node offers the cascade | members | share |
|---|---:|---:|
| a file path in context, no structural key → **stage 2** | 7 055 | 44.3 % |
| a structural key (`procedure_id`/`lesson_kind`/`type`/`topic`/`task_pattern`) → stage 1 | 5 595 | 35.1 % |
| neither, but a chunk vector → stage 4 | 2 499 | 15.7 % |
| a live query anchor → stage 3 | 777 | 4.9 % |

**(b) `_subsystem` collapses a whole tree into one word.** `_subsystem`
(recall_map.py:641) takes a single directory segment; this repository's paths are
`mmo/public/assets/…`, `mmo/test/…`, `mmo/server/…`, so 44 % of the pool concentrates
into a handful of buckets named `public`, `test`, `mmo`, `server`. `normalize_key` then
lowercases the segment and hands it over as the label, unchanged.

**(c) `_finish` sorts by size first.** `_finish` (recall_map.py:1529-1537) orders
`-len(members)` before stage before label. The stage-2 buckets are by construction the
biggest thing in the pool — median stage-2 cluster count 25, max 89, against a median
of 19 for the stage-4 clusters — so they take the top slots on every single call, and
`MAX_CLUSTERS`/the 700-char budget leaves nothing behind them.

### A fourth fact, previously unrecorded: the structure cache drains stage 1 into stage 2

`_matches` (recall_map.py:1502-1511) re-derives *path* membership from
`_node_subsystem` alone. It never asks whether stage 1 would have claimed the node
first — a precondition that only holds in a fresh cascade, where stage 1 already ran.
On a cache hit that precondition is gone: the cached template set holds only the
structural keys the *cached* pool happened to show, so a node whose `lesson_kind` is new
to that key falls through to a path template and is renamed after a directory.

Measured directly on real field pools — build cold on a key's first residual, then
`_recount` that key's second residual through the resulting templates:

* **7 of 7** cache keys with ≥ 2 recalls drift;
* **239 pool members** that carry a structural key of their own land in a path cluster.
  `lesson_kind: instrument-scope-pitfall` and `lesson_kind: production-equivalence-gap`
  both end up inside `public`.

Corroboration in the delivered payloads themselves: **62 of 294** delivered clusters are
path clusters whose *own medoid* carries a structural key — impossible in a cold build.
And the warm arm reproduces the delivered label sequence in **42 of 77** maps against
**9 of 77** for the cold arm, i.e. the field maps were predominantly cache-served.

This matters for the polish design: making stage 2 build a multi-word label fixes the
name, but the nodes that had a *good* name and lost it are recovered only by making
`_matches` respect stage precedence.

---

## 3. Frequency table of degenerate labels

`built` counts clusters in persisted payloads; `rendered` counts label slots the agent
actually saw across the 29 injects (ae renders the top 3). Document frequencies come
from the same `nodes_fts_vocab` index the c-TF-IDF stage reads, so "generic" is measured
against exactly the corpus recall searches — not against a word list.

FTS corpus for the document frequencies below: **N = 2573** indexed documents (host `alt`, snapshot of 2026-08-20T07:35Z).

| label | built | rendered | terms | max doc-frequency of a term | stage(s) |
|---|---:|---:|---:|---|---|
| `public` | 77 | 14 | 1 | `public` 531/2573 = 20.6% | path ×77 |
| `test` | 74 | 14 | 1 | `test` 1452/2573 = 56.4% | path ×74 |
| `mmo` | 62 | 14 | 1 | `mmo` 1439/2573 = 55.9% | path ×62 |
| `server` | 55 | 13 | 1 | `server` 596/2573 = 23.2% | path ×49, structural ×6 |
| `aether scene file` | 4 | 2 | 3 | `aether` 1154/2573 = 44.9% | embedding ×4 |
| `tools` | 3 | 1 | 1 | `tools` 168/2573 = 6.5% | path ×3 |
| `showcase` | 2 | 0 | 1 | `showcase` 200/2573 = 7.8% | structural ×1, path ×1 |
| `webgl view planet` | 2 | 1 | 3 | `planet` 548/2573 = 21.3% | embedding ×2 |
| `gpu facade фасада` | 2 | 1 | 3 | `gpu` 237/2573 = 9.2% | embedding ×2 |
| `aether hand pose` | 2 | 1 | 3 | `aether` 1154/2573 = 44.9% | embedding ×2 |
| `kills cls pbr` | 1 | 0 | 3 | `pbr` 238/2573 = 9.2% | embedding ×1 |
| `planet animal species` | 1 | 1 | 3 | `planet` 548/2573 = 21.3% | embedding ×1 |
| `planet outcome species` | 1 | 1 | 3 | `planet` 548/2573 = 21.3% | embedding ×1 |
| `webgl outcome species` | 1 | 0 | 3 | `species` 330/2573 = 12.8% | embedding ×1 |
| `master aether file` | 1 | 1 | 3 | `aether` 1154/2573 = 44.9% | embedding ×1 |
| `server цели login` | 1 | 1 | 3 | `server` 596/2573 = 23.2% | embedding ×1 |
| `обновление незнакомые resource` | 1 | 0 | 3 | `resource` 31/2573 = 1.2% | embedding ×1 |
| `hand lines aether` | 1 | 1 | 3 | `aether` 1154/2573 = 44.9% | embedding ×1 |
| `master git merge` | 1 | 0 | 3 | `git` 389/2573 = 15.1% | embedding ×1 |
| `camera` | 1 | 1 | 1 | `camera` 246/2573 = 9.6% | path ×1 |
| `light scene sky` | 1 | 0 | 3 | `scene` 427/2573 = 16.6% | embedding ×1 |

* **274 of 294** built clusters (93.2 %) carry a single content term.
* **268 of 294** (91.2 %) carry one of exactly four words: `public`, `test`, `mmo`,
  `server`.
* **57 of 67** rendered label slots (85.1 %) were a single word.
* Only **21 distinct labels** exist across the entire train+eval field window.

Two cautions for whoever pins the label gate. First, term count and document frequency
are *independent* signals here: `tools` (6.5 %), `camera` (9.6 %) and `showcase` (7.8 %)
are rare in the corpus and still useless as invitations, while `aether` appears in 44.9 %
of documents and sits inside three-word labels that read fine. Second, multi-word does
not imply meaningful — the contrastive run's top label is `dir dev null` (§6), which is
shell noise wearing three words.

---

## 4. The gist: the `_fit` squeeze-before-drop defect, confirmed and re-scoped

**Prior measurement.** LM trace `01M0DM85KWV2A3DJXQMVDACJMR` (2026-08-19, project:lm)
recorded that `RecallMapBuilder._fit` squeezes medoid examples to zero *before* dropping
clusters and never reflows them afterwards, leaving budget unused.

**Verdict against the field payloads: confirmed, and the field shows it is worse than a
reflow bug.**

`_fit` (recall_map.py:1552-1589) runs `while example_chars > 0`. Once a shave step
computes `example_chars = 0`, the loop body rebuilds the clusters at zero and the loop
test then fails — so the rebuild at zero is *never* checked against the budget. Control
falls to the drop loop, clusters are dropped from the tail, and no code path ever
restores an example. Both halves of that are visible in the data.

**Half one — the pure reflow loss.** Every delivered map ran the drop loop. All 77 came
out at 3 or 4 clusters where `MAX_CLUSTERS` is 6, and `clusters + more` — the number of
populated groups the cascade produced — is at least 8 on every map (max 119), so `kept`
was 6 every time and the missing two-to-three clusters were taken by the budget, not by
the cap. After the drops:

* median leftover headroom **48 chars**, maximum **203**, **4 748 chars in total** across
  the 77 maps — budget that bought nothing;
* **14 of 77** maps came out at 3 clusters with room for a **19–54 char** example each
  (median 35), and shipped a zero-length one.

**Half two — and this is the part the prior trace could not see.** Rebuilding each
persisted payload with the builder's own `_payload_size`, varying only the example
length, gives the exact room the 700-char budget affords at each breadth:

| clusters kept | maps | largest example that still fits (min / median / max) | maps with ≥ 40 chars | with ≥ 60 |
|---|---:|---|---:|---:|
| 4 | 63 | 0 / 0 / 0 | 0 | 0 |
| 3 | 77 | 19 / 51 / 55 | 63 | 0 |
| 2 | 77 | 120 / 120 / 120 | 77 | 77 |
| 1 | 77 | 120 / 120 / 120 | 77 | 77 |

One serialized cluster costs 88 characters of JSON keys before any content, so at four
clusters the 700-char budget is exhausted by scaffolding alone: **not one character of
gist fits at breadth 4, on any of the 63 maps that delivered four clusters.**

So the fix cannot be "reflow after dropping" alone — that recovers a gist for the
14 three-cluster maps and nothing for the other 63. **A gist floor has to buy its room
with breadth: roughly one dropped cluster per ~50 characters of gist**, and the advertised
`MEDOID_EXAMPLE_CHARS = 120` is only reachable at two clusters. The order `_fit` uses
today is exactly inverted against the channel's purpose, and the drop it already performs
is the currency the gist should have been spending.

**Not a renderer problem.** The ae probe reads `medoid.example` into `ref.gist`
(`agent_stream_probe.js:255`) and renders `label (count): gist -> memory_lookup …`
whenever it is non-empty, squeezing it away only under its own 400-char budget
(`renderMiniMap`, :267-290). The field never gave it anything to squeeze: the example is
absent in the *persisted payload*, before any renderer sees it — **zero** across 294 field
clusters and 33 contrastive clusters (§6).

---

## 5. What the degenerate labels actually cost: the channel starves after two injects

The server delivered 77 maps with clusters; the agent saw 29. The 39 probes that carried
a real, non-curtailed, cluster-bearing map and were still journalled `miss` are fully
accounted for: ae's probe rejects a cluster whose ask-hint or medoid node id was already
delivered in that session (`dashboard/lib/agent_stream_probe.js:787`, `_isNovel` →
`deliveredHints`). Testing that explanation case by case:

> **39 of 39** `miss`-with-a-delivered-map probes offered *only* clusters whose ask-hint
> or medoid had already been delivered earlier in the same session. No residual.

Because 91 % of clusters carry one of four labels, a whole session only ever offers
**5–6 distinct ask-hints**. The consequence is the same shape in 13 of the 14 sessions:

```
I I m m m m m m m m m      I = inject the agent saw
                            m = map built and persisted, never shown (not novel)
```

Two injects, then a long run of maps nobody sees — while the server keeps paying full
map-construction cost on every probe. It is not ae's inject cap: that cap is 3 and only
one session ever reached it. The overlap half of the novelty rule never fires on map
clusters either — `overlapRatio` returns 0 below `MIN_TOKENS_FOR_OVERLAP = 4` tokens, and
a label + ask-hint set of one to three tokens never reaches it — so the exact ask-hint
dedup is the only novelty gate that ever bites.

**The second force, and the label defect feeds it too.** ae also disables the channel
locally after two consecutive unconsumed injections (`localCurtailStreak: 2`). That fired
in **9 of the 14** train+eval sessions, and in all 9 it is the last non-`dedup` event in
the journal — the channel is off from then on. The two forces compose rather than
compete: novelty exhaustion produces the run of invisible maps, and the local curtail is
only *evaluated* when a probe finally passes novelty, so exhaustion postpones the verdict
and then the verdict lands. Either way the map gets about two shots per session, and it
spends both on `public` and `test`.

**Label variety is the channel's supply of novelty, not decoration.** A cascade that
offers four distinct labels over a whole corpus caps the channel at ~2 useful injects per
session however well those four labels read — and then hands the curtail rule a
zero-consumption streak that the labels themselves manufactured.

---

## 6. Contrastive: the live-e2e run that was consumed

Same server code, same commit, a different corpus — this workstation's store, scope
`project:ae`/`global`, task `stream-agents-thinking-map-live-e2e/*`. 13 recalls, 11
delivered maps, 33 clusters.

The named case, `01M0EZVRS93TNWJE3RMQ4NHSBH` at 2026-08-20T07:06:13Z, rendered:

```
[auto] memory also holds, near what you are doing now:
- dir pid dev (50) -> memory_lookup 01KRRPZ1VD39G8R84BR6SB6B7Q
- cleanup commit invariant (8) -> memory_lookup 01KSK6EZQ7J9QZ9Z37VHNXTKRH
- implementation note prompt (7) -> memory_lookup 01KZW6FV9AK2TMEJ4MQ5VVH63M
Automatic — ignore if not useful.
```

| | field (train+eval) | live-e2e control |
|---|---|---|
| delivered clusters | 294 | 33 |
| distinct labels | 21 | 17 |
| single-content-term labels | 274 (93.2 %) | 7 (21.2 %) |
| stage mix | path 90.8 %, embedding 6.8 %, structural 2.4 % | structural 30.3 %, embedding 48.5 %, path 21.2 % |
| pool composition | path 44.3 %, structural 35.1 % | structural 52.2 %, path 14.0 % |
| clusters with a gist | **0** | **0** |
| injects → consumptions | 29 → 1 | 3 → 2 |

`cleanup commit invariant` and `implementation note prompt` are **stage-1** labels:
`lesson_kind` values the extraction stage wrote, handed through `normalize_key`
unchanged. They read well for exactly the reason the field labels read badly — that
corpus reaches stage 1 for 52 % of its pool, where this one reaches stage 2 for 44 %.
The difference is corpus shape meeting a cascade that names stage 2 in one word; it is
not a different code path.

### What this control does *not* establish

Honesty about the contrast matters more than the contrast:

* **The gist is missing here too.** All 33 control clusters have a zero-length example,
  and the named case had only 16 chars of headroom at three clusters. Whatever produced
  the consumption, it was not the gist.
* **Which cluster was clicked does not support "the rich label invited the click."** In
  all three consumptions ever recorded — two here, one in the field — the agent looked up
  the **first-listed** cluster's medoid. Here that first cluster is `dir pid dev`, which
  is shell noise. The rich labels sat in positions 2 and 3 and were not taken.
* **The one field consumption came from a degenerate map.** It was `public / test /
  server`, and the agent looked up `public`'s medoid at distance 1. Degenerate labels are
  not inert; they are merely rare to reach.
* **n = 3.** Two consumptions out of three injects against one out of 29 is a suggestive
  rate difference on a sample far too small to carry a verdict, and the two arms differ in
  corpus, scope and task as well as in label quality.

What the control *does* establish is that the cascade can produce multi-word, meaningful,
differentiating labels with no code change at all when the pool reaches stage 1 — so the
target for stage 2 is a known-attainable shape, not an invention. And it establishes that
position 1 is where the channel's whole attention goes, which is an argument for the
usefulness-ordering work independent of anything about labels.

---

## 7. What this diagnosis licenses

Established, on train+eval field material, with executable provenance:

1. Nine of ten delivered field labels come from **stage 2**, which names a cluster after
   one directory segment; stage 3 delivered nothing at all.
2. **93.2 %** of delivered labels are a single content term; four words account for
   **91.2 %** of all delivered clusters and the corpus offers only 21 distinct labels.
3. The `_fit` gist defect is **confirmed** on real payloads — zero examples in 294 field
   and 33 control clusters, 4 748 chars of budget left unspent — and is **larger than a
   reflow bug**: at four clusters the 700-char budget affords zero gist, so a floor has
   to trade breadth for it at roughly one cluster per 50 chars.
4. Degenerate labels **starve the channel**: 39 of 39 invisible maps are explained by
   ask-hint repetition, 13 of 14 sessions stop at exactly two injects, and in 9 of 14 the
   local curtail then switches the channel off for good on a zero-consumption streak the
   label vocabulary manufactured.
5. `_recount`/`_matches` **drains stage 1 into stage 2** on cache hits — 7 of 7 keys,
   239 members — so richly-named nodes are being renamed after directories.

Limits that must travel with these numbers:

* The frozen snapshot is ~30 minutes younger than the field run; the replay reproduces
  the ranked head on **66 of 81** cases. Aggregates over *persisted payloads* (§1 label
  set, §3 frequencies, §4 budget arithmetic) do not depend on the replay at all; only the
  stage attribution and the pool composition do, and 26 of 294 attributions fall back to
  medoid precedence with 7 flagged uncertain.
* Everything here is one project (`project:game`), one afternoon, one agent
  (`ae:tree_node`, claude). The holdout split exists precisely because a rule fitted to
  these 21 labels could be a lookup table rather than a rule.
* Consumption evidence is n = 1 in the field and n = 2 in the control. No consumption
  threshold should be pinned from it.

---

## Appendix — every delivered inject, attributed

All 29 injects the agent saw in the train+eval window. `gist` is the medoid example
length in characters, per rendered cluster, straight out of the persisted payload.

| # | when (UTC) | split | task | seq | rendered label (count) → stage | gist |
|---|---|---|---|---|---|---|
| 1 | 04:40:34 | eval | `quality/clips-coverage/idle-rest-clips` | 1 | `public` (56) → **path**<br>`aether scene file` (25) → **embedding**<br>`test` (24) → **path** | 0; 0; 0 |
| 2 | 04:40:34 | train | `quality/clips-coverage/registry-refresh` | 1 | `aether scene file` (51) → **embedding**<br>`public` (43) → **path**<br>`test` (22) → **path** | 0; 0; 0 |
| 3 | 04:40:47 | train | `quality/clips-coverage/registry-refresh` | 2 | `server` (20) → **path**<br>`mmo` (11) → **path** | 0; 0 |
| 4 | 04:40:59 | eval | `quality/clips-coverage/idle-rest-clips` | 2 | `server` (22) → **path**<br>`mmo` (10) → **path** | 0; 0 |
| 5 | 04:53:32 | train | `quality/clips-coverage/registry-refresh` | 1 | `public` (53) → **path**<br>`planet outcome species` (20) → **embedding**<br>`test` (17) → **path** | 0; 0; 0 |
| 6 | 04:53:57 | eval | `quality/clips-coverage/idle-rest-clips` | 1 | `public` (52) → **path**<br>`server` (21) → **path**<br>`test` (18) → **path** | 0; 0; 0 |
| 7 | 04:55:01 | eval | `quality/clips-coverage/idle-rest-clips` | 2 | `mmo` (15) → **path** | 0 |
| 8 | 04:56:15 | train | `quality/clips-coverage/registry-refresh` | 2 | `mmo` (24) → **path**<br>`server` (13) → **path** | 0; 0 |
| 9 | 05:05:36 | train | `quality/clips-coverage/registry-refresh` | 1 | `public` (53) → **path**<br>`webgl view planet` (26) → **embedding**<br>`server` (12) → **path** | 0; 0; 0 |
| 10 | 05:07:34 | eval | `quality/clips-coverage/idle-rest-clips` | 1 | `public` (55) → **path**<br>`test` (28) → **path**<br>`server` (17) → **structural** | 0; 0; 0 |
| 11 | 05:08:13 | train | `quality/clips-coverage/registry-refresh` | 2 | `test` (29) → **path**<br>`mmo` (4) → **path** | 0; 0 |
| 12 | 05:09:52 | eval | `quality/clips-coverage/idle-rest-clips` | 2 | `mmo` (21) → **path** | 0 |
| 13 | 05:23:25 | eval | `quality/clips-coverage/idle-rest-clips` | 1 | `planet animal species` (37) → **embedding**<br>`public` (34) → **path**<br>`server` (17) → **path** | 0; 0; 0 |
| 14 | 05:24:50 | eval | `quality/clips-coverage/idle-rest-clips` | 2 | `test` (42) → **path**<br>`mmo` (11) → **path** | 0; 0 |
| 15 | 05:54:33 | train | `quality/clips-coverage/clips-restore` | 1 | `public` (27) → **path**<br>`master aether file` (20) → **embedding**<br>`test` (18) → **path** | 0; 0; 0 |
| 16 | 05:55:05 | train | `quality/clips-coverage/clips-restore` | 2 | `server` (10) → **path**<br>`mmo` (6) → **path** | 0; 0 |
| 17 | 06:02:27 | train | `quality/clips-coverage/clips-restore` | 1 | `public` (46) → **path**<br>`server цели login` (18) → **embedding**<br>`test` (16) → **path** | 0; 0; 0 |
| 18 | 06:03:12 | train | `quality/clips-coverage/clips-restore` | 2 | `server` (15) → **path**<br>`mmo` (12) → **path** | 0; 0 |
| 19 | 06:20:53 | eval | `quality/clips-coverage/rest-clips-land` | 1 | `public` (45) → **path**<br>`hand lines aether` (25) → **embedding**<br>`server` (18) → **path** | 0; 0; 0 |
| 20 | 06:21:24 | eval | `quality/clips-coverage/rest-clips-land` | 2 | `test` (42) → **path**<br>`mmo` (15) → **path** | 0; 0 |
| 21 | 06:45:07 | train | `quality/meter-honesty` | 1 | `public` (73) → **path**<br>`test` (37) → **path**<br>`aether hand pose` (6) → **embedding** | 0; 0; 0 |
| 22 | 06:45:46 | train | `quality/meter-honesty` | 2 | `mmo` (8) → **path** | 0 |
| 23 | 07:24:32 | eval | `quality/footslip-rest/gait-reach` | 1 | `public` (63) → **path**<br>`gpu facade фасада` (18) → **embedding**<br>`test` (11) → **path** | 0; 0; 0 |
| 24 | 07:24:32 | train | `quality/footslip-rest/proc-footlock` | 1 | `public` (55) → **path**<br>`test` (23) → **path**<br>`server` (11) → **structural** | 0; 0; 0 |
| 25 | 07:24:55 | eval | `quality/footslip-rest/gait-reach` | 2 | `mmo` (15) → **path**<br>`server` (10) → **path** | 0; 0 |
| 26 | 07:25:08 | train | `quality/footslip-rest/proc-footlock` | 2 | `mmo` (10) → **path** | 0 |
| 27 | 07:25:31 | train | `quality/meter-honesty` | 1 | `public` (56) → **path**<br>`test` (26) → **path**<br>`mmo` (11) → **path** | 0; 0; 0 |
| 28 | 07:25:52 | train | `quality/footslip-rest/proc-footlock` | 3 | `camera` (10) → **path**<br>`tools` (7) → **path** | 0; 0 |
| 29 | 07:28:02 | train | `quality/meter-honesty` | 2 | `server` (10) → **path** | 0 |
Inject #24 is the one field consumption: 18.4 seconds after the probe, the agent called
`memory_lookup` on `public`'s medoid `01M0ER54TB54ZESC2YMGDVQRC4`, one tool call later
(`distance: 1`). The map it came from was `public / test / server` — the degenerate set.

---

## Reproducing this

```bash
python3 scripts/recall_map_field_extract.py \
  --streams <copy of the stream_inject tree> \
  --field-snapshot ~/.cache/living-memory-field/recall-map-field-alt-20260820.sqlite3 \
  --e2e-snapshot  ~/.cache/living-memory-field/recall-map-e2e-local-20260820.sqlite3 \
  --e2e-journals '/tmp/ae-stream-live-e2e.*/journal.jsonl' \
  --out-manifest artifacts/recall-map/field/manifest.json \
  --out-analysis artifacts/recall-map/field/analysis.json
```

`analysis.json` holds every aggregate quoted here, per split; `manifest.json` holds the
per-case material, including the holdout cases this document does not touch. Snapshot
paths, sha256s and row counts are pinned in both.
