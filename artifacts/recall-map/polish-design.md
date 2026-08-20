# Recall-map polish — pinned design

**Status: pinned.** Three sequential builder edits implement this document and may not
drift from it: `label-quality-gate` (sections *label-gate*, *label-construction*,
*filter-journaling*), `gist-floor` (section *gist-floor*), `usefulness-order` (section
*usefulness-order*). Every number below is derived from the **train and eval** splits of
[`field/manifest.json`](field/manifest.json) and [`field/analysis.json`](field/analysis.json),
read through [`field/diagnosis.md`](field/diagnosis.md). **No holdout case was opened, replayed,
or quoted.** Every figure in this document was recomputed here, from the artifacts, against the
real module — nothing is carried over on trust.

Source of truth for behaviour: `src/living_memory/recall_map.py` (1746 lines at the time of
writing). Consumers: `src/living_memory/instructions_map.py` and the ae probe
`dashboard/lib/agent_stream_probe.js` (read-only, never edited — see *filter-journaling*).

---

## 0. Budgets, restated

These are the three inviolable caps. They are restated here because every section below
spends against them, and because the arithmetic in *gist-floor* is only meaningful next
to them.

| budget | constant | cap | who enforces |
|---|---|---:|---|
| recall-response payload | `MAX_RESPONSE_CHARS` | **700** | `RecallMapBuilder._fit` via `_payload_size` |
| instructions line | `MAX_INSTRUCTIONS_CHARS` | **150** | `RecallMap.render_compact`; `instructions_map.MAX_LINE_CHARS` |
| ae inject render | (ae-side `renderMiniMap`) | **400** | ae, unchanged by this design |

Two more caps are load-bearing and unchanged: `MAX_LABEL_CHARS = 40` and
`MAX_CLUSTERS = 6`.

**Verified end-to-end at the worst case** (2 clusters, labels at the full 40-char cap,
40-char gist, journal block present carrying counts and `names_omitted`), by constructing a
real `RecallMap` and calling the real renderers:

| surface | measured | cap | headroom |
|---|---:|---:|---:|
| `to_dict()` payload | **680** | 700 | 20 |
| `render_compact()` | **110** | 150 | 40 |
| `instructions_map.compose_map_section` line | **89** | 150 | 61 |
| `instructions_map` whole section | **138** | 240 | 102 |
| ae render at gist 40 | **356** | 400 | 44 |
| ae render at gist 56 | **388** | 400 | 12 |

The ae row is the one to read twice. At two clusters and 40-char labels the ae channel
saturates at a gist of roughly **66 chars**; above that `renderMiniMap` squeezes on its own
budget. So the server's gist floor of 40 and ae's effective ceiling of ~56–66 bracket a
narrow usable band, and `MEDOID_EXAMPLE_CHARS = 120` is aspirational rather than reachable
in this corpus. The floor is what matters; the cap stays 120 so a two-cluster map over a
cheap-label corpus can still fill it.

---

## 1. `label-gate`

### The predicate

A label is **deliverable** when its content terms are, collectively, rare enough in the
corpus that the label distinguishes something. Rarity is measured against the same FTS
document-frequency index `_ranked_terms` already reads, so "generic" means generic *to the
corpus recall searches*, never to a word list.

Let `N = store.fts_document_count()` and `df(t) = store.term_document_frequencies([t])[t]`.

```
ic(t)      = log((N + 1) / (1 + df(t)))
S3(label)  = sum of the LABEL_GATE_TOP_TERMS largest ic(t)
             over the distinct content terms of `label`
             (terms via recall_map._terms — stop-words already dropped)
```

New module constants:

```python
LABEL_GATE_TOP_TERMS = 3
LABEL_GATE_MIN_IC = 4.0
```

**A cluster is delivered only when `S3(label) >= LABEL_GATE_MIN_IC`.**

`_terms` is used verbatim and unstemmed, because the df index is `unicode61` and does not
stem; a stemmed term reports `df = 0` and scores as maximally rare, which would turn the
gate inside out.

### Why 4.0, and why top-3 rather than a plain sum

Computed over **every label the field ever delivered** (train+eval, 294 clusters, 21
distinct labels, `N = 2573`):

| population | labels | clusters | S3 range |
|---|---:|---:|---|
| single content term | 7 | **274** | **0.57 … 2.72** |
| multi-word | 14 | **20** | **4.89 … 15.53** |

```
 0.57  test          2.55  showcase        6.65  webgl view planet
 0.58  mmo           2.72  tools           7.17  server цели login
 1.46  server       ──────── gap ────────  9.24  kills cls pbr
 1.58  public        4.89  aether scene file      11.14  gpu facade фасада
 2.34  camera        5.15  planet animal species  15.53  обновление незнакомые resource
```

The interval `(2.72, 4.89)` is **empty**. A floor at 4.0 sits inside it with ≈1.3 nats of
margin below and ≈0.9 above, gates exactly the 274-cluster single-term population, and
keeps all 20 multi-word clusters. It is not fitted to a boundary case; it is placed in a
void.

**Top-3, not the full sum, is what makes the floor mean anything.** The corpus's four house
words — `public`, `server`, `mmo`, `test` — have individual `ic` of 1.577, 1.461, 0.581,
0.572. Their *plain sum* is **4.191** and would clear a floor of 4.0; their *top-3* is
**3.619** and does not. Capping the sum at three terms is what stops a label from buying
its way past the gate by concatenating house vocabulary.

The floor also encodes the content-term-count half of the predicate implicitly, which is
better than stating it twice: at 4.0 a **solo** term survives only if
`df <= (N+1)/e^4 - 1 = 46`, i.e. **1.79 %** of documents. A one-word label is not banned by
rule — it is admitted exactly when one word genuinely is that rare.

Reproduce: `analysis.json $.train_eval_field.label_statistics` carries `df` per label term.

### Degradation

`_ranked_terms` already degrades to plain term frequency when the store cannot answer the
df queries (old schema, no indexed documents). The gate must degrade too, and must never
be the reason a map goes empty:

* when `N <= 1`, or `term_document_frequencies` / `fts_document_count` raises
  `AttributeError` or `sqlite3.OperationalError`, **fall back to the purely structural
  predicate `len(distinct content terms of label) >= 2`**.

A store with no corpus statistics is a store where "generic" is undefined, and a gate that
withholds everything on a missing index is worse than one that admits a weak label.

### Withheld, not renamed

A gated cluster is **removed from delivery entirely**. It is *not*:

* relabelled to `noise`, `other`, `misc`, or any catch-all;
* merged into a residual bucket;
* appended to `clusters` with a marker field.

Two reasons, both hard. First, a catch-all row is a row the agent cannot act on, and after
*gist-floor* the map has only **two** rows to spend — a wasted one costs half the channel.
Second, anything that reaches the `clusters` list is a delivered item to the curtail probe
(`_delivered_items`) and to ae's `_isNovel`; a noise row would consume novelty and suppress
curtail on a cluster nobody could ever follow. Withheld clusters go to the journal (section
*filter-journaling*) and nowhere else.

If every cluster is gated, `_finish` returns a **journal-only map**: `clusters` is empty
and `filtered` still carries the withheld count and budgeted names. This is required by
the no-silent-caps rule — returning `None` here would make a persisted event whose gate
rejected everything indistinguishable from an event whose residual was empty. `build`
calls `_note_delivery(offered=False)` for the journal-only shape, so it reaches the
response and `recall_events.recall_map` without growing the curtail offer count. The
instructions and ae renderers already tolerate empty `clusters` because curtailed maps
use the same empty-list shape.

### Order of operations

Enrichment first, gate second — **enrichment is the rescue, the gate is the net**:

```
_cluster -> per-stage label construction (§2) -> _disambiguate
         -> LABEL GATE (withhold + journal)
         -> _finish: usefulness order (§4) -> _fit: breadth/gist (§3)
```

The gate runs after `_disambiguate` because `_disambiguate` can *lengthen* a label with a
distinguishing term, which can only raise `S3`. Gating before it would withhold clusters
that were about to be rescued.

---

## 2. `label-construction`

Target shape for every stage: **multi-word, distinguishing, readable without context,
within `MAX_LABEL_CHARS = 40`.** The diagnosis established this is attainable with no new
machinery — the live-e2e control produced `cleanup commit invariant` and `implementation
note prompt` from the same code, because that corpus reached stage 1. This corpus reaches
stage 2 for 44.3 % of its pool, and stage 2 names a cluster after one directory segment.

### The invariant that governs all four stages

> **A cluster's `ask_hint` may carry at most `ASK_HINT_MAX_TOKENS = 2` tokens on the path
> and structural stages, and must retain the head token it would have had before
> enrichment.**

This is not a style rule. It is the entire reason the polish does not break the curtail
rule, and it is proved in section *curtail-interplay*. It requires a change to
`_disambiguate` (recall_map.py:963): today `hint_followed_label` makes the hint track the
enriched label for every non-anchor stage, which would drag the hint to three tokens.
**The hint must follow the label's head only.**

`MAX_ASK_HINT_CHARS = 80` is unchanged and no longer binding — the token cap binds first.

### Stage 2 — path (90.8 % of delivered field clusters)

`_stage_path` (recall_map.py:1025-1045) currently sets
`label = ask_hint = normalize_key(subsystem)`. Replace with:

```
head   = normalize_key(subsystem)
extra  = first CTFIDF_LABEL_TERMS - 1 terms of _ranked_terms(bucket)
         that are not already tokens of `head`
label  = _shorten(" ".join([head, *extra]), MAX_LABEL_CHARS)
hint   = " ".join([head, *extra[:1]])          # <= 2 tokens, head retained
```

`_ranked_terms` is the existing c-TF-IDF ranker (recall_map.py:1194) — reused, not
reimplemented, so path labels and embedding labels are distinguishing by the same measure
against the same index.

This is what rescues the gate's 274 clusters rather than withholding them. `public` has
`ic = 1.577`; two bucket terms at `df = 100` contribute `ic = 3.24` each, giving
`S3 = 8.06` against a floor of 4.0. Enrichment clears the gate with a wide margin for any
plausible pair of distinguishing terms, which is the intended relationship between the two
sections: **the gate should almost never fire after enrichment, and when it does it is
reporting a cluster with nothing to say.**

**Cost, and the reason `latency-bench` is a sibling node.** `_ranked_terms` is today called
only by stage 4. Calling it per path bucket adds up to `CTFIDF_TERM_BUDGET = 40` indexed
seeks plus one count per bucket. Field pools carry a handful of path buckets, so this is
single-digit multiples of an existing cost — but it is a new cost on the hot path and the
before/after bench is mandatory, not advisory. `fts_document_count()` must be memoized per
`build()` call rather than asked once per bucket.

### Stage 1 — structural, generic values enriched

`_stage_structural` (recall_map.py:994-1023) hands `normalize_key(raw)` through unchanged.
Field evidence that this is not always enough: **6 of the 55 `server` clusters were
structural**, not path — a structural key whose value is one house word is exactly as
undeliverable as a directory segment.

Rule: build `readable` as today (including the existing `_looks_unreadable` →
`_phrase_from_content` branch), then **if `S3(readable) < LABEL_GATE_MIN_IC`, enrich it by
the stage-2 recipe** — append distinguishing terms from the bucket, keep the hint at ≤2
tokens with the original value as head. A structural value that already passes is left
exactly alone: `cleanup commit invariant` needs no help, and rewriting a key the corpus
actually stores would throw away the one label form the control run proved works.

### Stage 3 — anchor, checked only

`_stage_anchor` labels with a past grounded query verbatim. Two rules, both conservative:

* the label is **checked** against the gate like any other, and withheld if it fails;
* the `ask_hint` is **never rewritten** — not to enrich it, and not to shorten it to 2
  tokens. It is a question that already worked, and `_disambiguate` already exempts it.

The `ASK_HINT_MAX_TOKENS` cap therefore does **not** apply to stage 3. This is safe for the
curtail rule because anchor hints are already long today: the rule preserves *historical*
`_echoes` behaviour, and stage 3's behaviour is unchanged by this design. It is also
untested in the field — **stage 3 delivered 0 clusters in the entire field window** — so
this branch carries no field evidence either way and must not be tuned on the strength of
any.

### Stage 4 — embedding, checked only

`_ctfidf_label` already emits `CTFIDF_LABEL_TERMS = 3` terms and all 20 field embedding
labels pass the gate (S3 4.89–15.53). No construction change; gate check only.

One honest caveat carried from the diagnosis: multi-word does not imply meaningful. The
control run's top label was `dir dev null` — shell noise wearing three words, S3 comfortably
above the floor. The gate bounds *genericness*, not *meaning*. Nothing in this design claims
otherwise, and no threshold here should be moved to try.

### Parent closure: `_recount` preserves stage precedence

The diagnosis (§2) established that `_matches` (recall_map.py:1502-1511) re-derives path
membership without asking whether stage 1 would have claimed the node first, draining
stage-1 nodes into stage-2 clusters on every cache hit — **7 of 7 keys, 239 pool members**.

This is real and the parent verification closes it: a cached path template first rejects
any node `_structural_key` would have claimed, re-establishing the precondition stage 2 gets
for free in a cold cascade. If the cached template set has no exact structural signature for
such a node, `_recount` refuses the cache and performs a cold rebuild, so the node is neither
renamed after a directory nor silently left uncovered. The structure cache therefore keeps
stable shapes only while they still respect the cascade's structural-before-path invariant.

---

## 3. `gist-floor`

### The defect, in one line

`_fit` (recall_map.py:1552-1589) runs `while example_chars > 0`; once a shave computes
`example_chars = 0` the loop rebuilds at zero and the loop *test* then fails, so the rebuild
at zero is never checked against the budget. Control falls to the drop loop, clusters are
dropped, and **no code path ever restores an example.** Result in the field: **0 medoid
examples across 294 delivered clusters**, with 4 748 chars of budget left unspent.

### The arithmetic that makes "reflow after dropping" insufficient

Recomputed here with the real `_payload_size` over the real persisted payloads, varying only
example length (77 train+eval maps):

| breadth | maps | largest example that fits (min / median / max) | ≥40 chars | ≥60 |
|---:|---:|---|---:|---:|
| 4 | 63 | 0 / 0 / 0 | 0 | 0 |
| 3 | 77 | 19 / 51 / 55 | 63 | 0 |
| 2 | 77 | 120 / 120 / 120 | 77 | 77 |
| 1 | 77 | 120 / 120 / 120 | 77 | 77 |

One serialized cluster costs 88 chars of JSON scaffolding before content and **156 chars in
practice** with a short label — because `plan_item` restates both the label and the ask-hint
(`"on touching {label} - recall '{hint}' ({count})"`). Measured cluster cost by label width,
with a short hint: 6 chars → 156, 17 → 178, 24 → 192, 30 → 204, 40 → **224**.

So at breadth 4 the 700-char budget is exhausted by scaffolding alone, and reflow alone
recovers a gist for 14 maps and nothing for the other 63. **The gist has to buy its room
with breadth.**

### The rule

New module constants:

```python
MIN_MEDOID_EXAMPLE_CHARS = 40   # the gist floor
MIN_CLUSTERS = 2                # the breadth floor
```

`_fit` is restructured to this order. Each step re-measures with `_payload_size`; nothing is
estimated:

1. **Widest example that fits at the current breadth.** Search `[0, MEDOID_EXAMPLE_CHARS]`
   for the largest example length whose payload is `<= MAX_RESPONSE_CHARS`.
2. **If that is `>= MIN_MEDOID_EXAMPLE_CHARS`, done.**
3. **Otherwise trade breadth for gist**, while `len(kept) > MIN_CLUSTERS`: drop the tail
   cluster — which the usefulness sort has already made the least relevant — increment
   `dropped`, record its label and count in the journal, and **return to step 1**. Returning
   to step 1 *is* the reflow fix: the example budget is recomputed from scratch after every
   drop, never inherited from the pre-drop state.
4. **At the breadth floor, the journal names give way** (section *filter-journaling*): drop
   one name, return to step 1. Counts never give way.
5. **Gist never dies.** If breadth is at the floor and no journal names remain but the
   fixed cluster fields still cannot afford the gist floor, deliver no cluster. Move the
   remaining groups to `filtered.dropped` and persist a journal-only map instead. This is
   the rare long-hint case: silence plus an explicit reason is cheaper than an invitation
   whose informative part was squeezed away. The breadth floor is never broken to save
   the gist.

The floor is a floor on *what the budget affords*, not on what the node has: when
`len(_collapse(medoid.content)) < MIN_MEDOID_EXAMPLE_CHARS`, the whole content is the gist
and the floor is satisfied. Field medoid content is never short — min 230 chars, median
1 301, and **0 of 294** below 40 — so this branch is defensive, not routine.

`MIN_CLUSTERS` must be clamped to the number of populated groups: a pool yielding one
cluster delivers one cluster.

### Verified

Simulating the algorithm above across all 77 train+eval maps, with the journal block
present and journal names shortened to 24 chars:

| label width | breadth | gist chars (min / median / max) | ≥ floor | journal names kept | status |
|---|---|---|---:|---|---|
| 24 chars (realistic enriched) | 2 on 77/77 | **40 / 53 / 57** | **77/77** | 2 on 73, 1 on 4 | ok on 77 |
| 40 chars (`MAX_LABEL_CHARS`) | 2 on 77/77 | **40 / 41 / 62** | **77/77** | 1 on 39, 0 on 38 | ok on 77 |

The floor holds on every map at both widths, and it holds because breadth gives way. The
honest consequence, stated plainly: **on this corpus the map goes from four shallow rows to
two rich ones.** `MAX_CLUSTERS = 6` becomes unreachable here. That is the trade the channel
was designed to want — the diagnosis showed all three recorded consumptions took position 1
— but it is a real reduction in breadth and the eval must measure it, not assume it.

### Why the breadth floor is 2 and not 1

Measured on the field's own historical queries (section *curtail-interplay* explains the
machinery):

| breadth cap | clusters delivered | echo fires | deliveries judged "followed" |
|---:|---:|---:|---:|
| all (baseline) | 294 | 688 | **67** of 77 |
| 3 | 231 | 494 | 65 |
| **2** | 154 | 293 | **64** |
| 1 | 77 | 148 | **57** |

Breadth 2 preserves 64 of 67 followed-deliveries (−4.5 %); breadth 1 loses 10 of 67
(−15 %) because the second cluster is carrying real consumption signal. **2 is the floor.**

---

## 4. `usefulness-order`

### The defect

`_finish` (recall_map.py:1529-1538) sorts by `(-len(members), stage, label)`. Stage-2 path
buckets are by construction the biggest thing in a path-dominated pool — median 25 members
against 19 for stage 4 — so they take the top slots on every call, and the caps leave
nothing behind them. Size is not relevance; it is an artifact of how the corpus stores
paths.

### The key

The residual is *already ranked by the recall's own scoring for this query*. Each `_Member`
carries `rank` (0 = best). Order by reciprocal-rank mass:

```python
score(group) = sum(1.0 / (1 + m.rank)
                   for m in sorted(group.members, key=lambda m: m.rank))

sort key = (-score, best_rank, -len(members), _STAGE_ORDER[stage], label)
           where best_rank = min(m.rank for m in group.members)
```

Reciprocal rank rather than best-rank-alone, because best-rank-alone lets a singleton
holding rank 0 outrank a fifty-member cluster whose best is rank 1 — one node deciding the
whole map. Reciprocal rank rewards both height and mass: a lone rank-0 member scores 1.0,
and a fifty-member cluster spanning ranks 20–70 scores ≈1.22.

Mean rank was considered and rejected: it *punishes* mass, so a large cluster with a
strong head sorts below a tight mid-ranked one, which reintroduces the same
size-vs-relevance confusion from the opposite direction.

### Determinism

Total, and total by construction:

* the sum is taken over members in **ascending rank order**, a fixed permutation for a given
  pool, so the float sum is bit-reproducible across runs;
* ties fall through `best_rank`, then `-len(members)`, then `_STAGE_ORDER`, and finally
  `label` — which `_disambiguate` has already made **unique across the map**. The chain
  cannot end in a tie, so no ordering depends on list order or hash seeding.

### Interaction with the structure cache

No cache change, and none is needed. `_ClusterTemplate` (recall_map.py:414-427) stores
`stage`, `signature`, `label`, `ask_hint`, `member_ids`, `medoid_id` — **it stores no
order**. `_recount` returns groups built from *current-pool* members, so every `m.rank` is
this call's rank, and `_finish` re-sorts on every call.

The resulting contract, which the sibling node must assert:

* **structure is cached** — labels, signatures and medoid choice are stable across calls
  under one key;
* **order is not cached** — it is a per-call refresh over the current residual;
* the same pool through a warm builder and a cold builder yields the **same order**;
* a cache hit against a *different* residual may legitimately reorder, and that is the
  feature, not drift.

### Measured effect

Cluster membership is not persisted, but `manifest.json` carries `replay.pool_node_ids` (the
ranked pool) and each cluster's `medoid_node_id` — and for stages 1–3 the medoid **is** the
best-ranked member (recall_map.py:1633-1634). So `best_rank` is directly recoverable for
274 of 294 field clusters. Over the 70 train+eval maps with a recoverable pool and ≥2
clusters:

* **62 maps (88.6 %) reorder** under the new key;
* **31 maps (44.3 %) change their lead cluster** — 25 path→path, 4 path→embedding,
  1 path→structural, 1 embedding→path.

Medoid rank in the ranked residual, by stage: path median 12 (min 0, max 199), structural
median 18, embedding median 37. The stage that was winning on size sits mid-pack on
relevance.

Since every consumption ever recorded — two in the control, one in the field — took the
**first-listed** cluster, changing the lead slot on 44 % of maps is the largest single
behavioural change in this design. The `replay-eval` sibling measures it on eval and
holdout; nothing here claims an outcome from it.

---

## 5. `filter-journaling`

### Shape

One **additive** top-level key on the payload. Emitted only when something was actually
filtered, so an unfiltered map is byte-identical to today's.

```json
"filtered": {
  "withheld": 274,
  "dropped": 3,
  "names": [["w", "public", 73], ["d", "webgl view planet", 12]],
  "names_omitted": 2
}
```

| field | meaning | may give way? |
|---|---|---|
| `withheld` | clusters the **label gate** refused (§1) | **never** |
| `dropped` | clusters the **breadth/gist trade** dropped (§3) | **never** |
| `names` | `["w"\|"d", label, count]` triples, newest-dropped first | yes, under budget |
| `names_omitted` | names not shown; present only when `> 0` | **never** (it is the anti-silence field) |

New constants:

```python
FILTER_JOURNAL_NAMES = 4         # most name triples carried
FILTER_JOURNAL_LABEL_CHARS = 24  # names shortened harder than delivered labels
```

Names are shortened to 24 rather than 40 for the same reason `instructions_map` shortens to
28: a journal name is a forensic breadcrumb, not an invitation, and it competes directly
with the gist for the same 700 chars.

### The no-silent-caps discipline

The rule is *no silent* caps, not *no* caps. The counts are the honesty guarantee and are
unconditional — priced at **+40 chars** (+58 with `names_omitted`), always affordable: the
worst-case payload carrying counts and no names measures **680** of 700. The names are
convenience and are budget-dependent, and whenever the budget takes one, `names_omitted`
says so. Nothing is ever dropped without a number attached.

Pricing, measured against the payload: counts only **+40**; counts + 2 flat names **+76**;
counts + 3 tagged triples **+98**. At the absolute worst case — 2 clusters, both labels at
the full 40-char cap, 40-char gist — even a single 24-char tagged name overflows (**726**
> 700), which is exactly why names are shortened *and* give way ahead of the gist. Across
the 77 train+eval maps the §3 simulation keeps **2 names on 73 maps** at realistic 24-char
labels, and **1 name on 39 / 0 on 38** at the 40-char cap — degrading exactly as designed,
with the count of what was dropped always on the wire.

### `more` is not touched

`more` keeps its exact current meaning — budget-dropped clusters — and `covered` keeps
counting only delivered clusters. Withheld clusters are **not** added to `more`. Two
consequences, both wanted: `instructions_map._clusters_of`'s `held_back` boolean
(`more > 0 or covered < pool`) keeps working unchanged and stays `True`, and a forensic
reader can tell "the filter was harsh" (`withheld`) from "the budget was tight"
(`dropped`) without joining two fields. `filtered.dropped` deliberately mirrors `more` so
the block is self-contained and greppable; the duplication costs ~14 chars and is priced in
above.

### Consumer check

Verified by reading each consumer, not by assumption. **All three tolerate the addition with
no change**, and the one hard rule that makes this true is stated after the table.

| consumer | what it reads | effect of `filtered` |
|---|---|---|
| `instructions_map._clusters_of` (:251-268) | `clusters`, `more`, `covered`, `pool` | unknown key, ignored by `Mapping.get`; `held_back` unchanged |
| `instructions_map._merge` (:271-316) | per-cluster `label`, `count` | iterates `clusters` only; never sees it |
| `recall_map._delivered_items` (:701-732) | `clusters[:MAX_CLUSTERS]`, then `medoid.node_id`, `label`, `ask_hint` | requires `clusters` to be a `list`/`tuple`; `filtered` is a sibling key, never reached |
| ae `agent_stream_probe.js` | `clusters[].label`/`count`/`ask_hint`/`medoid.example`/`medoid.node_id` | unknown property, ignored; **no ae edit needed** |

> **Hard rule: withheld and dropped clusters must never be appended to the `clusters`
> list**, and no cluster entry gains a "this was filtered" flag.

`_delivered_items` is deliberately paranoid about shape and treats an unparseable payload as
*unjudgeable* rather than unconsumed — but it is not paranoid about *extra entries*. A
journal row inside `clusters` would be read as a delivered item by the curtail probe and as
a novelty-consuming hint by ae's `_isNovel`, corrupting both. The additive-sibling-key shape
is what keeps that impossible rather than merely unlikely.

The sibling node must add a regression test asserting that
`_delivered_items(payload_with_filtered) == _delivered_items(payload_without_filtered)` and
that `compose_map_section` renders identically with and without the block.

---

## 6. `curtail-interplay` — longer labels must not retune the curtail rule

### The mechanism

`_was_followed` (recall_map.py:1381-1405) judges a delivered map consumed if a medoid was
accessed after delivery **or** if a later query echoes a cluster's label **or** its ask-hint:

```python
_echoes(phrasing, query) := |phrasing & query| >= CURTAIL_QUERY_OVERLAP * |phrasing|   # 0.5
```

Because the shared count is an integer, the requirement is `ceil(n/2)` for a phrasing of `n`
tokens:

| tokens in phrasing | shared tokens required |
|---:|---:|
| 1 | 1 |
| **2** | **1** |
| 3 | 2 |
| 4 | 2 |

**One and two tokens require the same evidence. Three does not.** That single step is the
whole risk: longer labels make `_echoes` stricter, fewer deliveries are judged followed,
more keys collapse under `CURTAIL_STREAK` — a silent retune of a pre-registered rule,
arriving as a side effect of a cosmetic change.

### Measured on historical field queries

Replayed with the real `token_set` and the real `CURTAIL_QUERY_OVERLAP` over every
(delivery, later-query) pair under a shared cache key — **2 524 pairs across 77 delivered
maps**, train+eval only. Enrichment terms are modelled as tokens *no historical query
contains*, i.e. the worst case for matching:

| arm | echo fires | deliveries judged "followed" |
|---|---:|---:|
| baseline, labels as shipped | **688** | **67** of 77 |
| label → 3 tokens, **ask-hint also → 3** | **0** | **0** |
| label → 3 tokens, **ask-hint held at ≤2** | **688** | **67** |
| label → 3 tokens, ask-hint → 2 | 688 | 67 |

Letting the ask-hint follow the label to three tokens destroys **every** consumption signal
in the field window and would collapse every key. Holding it at ≤2 tokens is
**bit-identical to baseline**.

The guarantee is analytic, not statistical, which is why it is safe to pin: the enriched
hint's token set is a **superset** of the old one, and `ceil(2/2) = ceil(1/2) = 1`, so a
query that cleared the old hint clears the new one **by construction**. The 688/688 result
is a check on the reasoning, not the reason.

Delivered-label token counts today: **274 clusters at 1 token, 20 at 3**. So the invariant
binds on exactly the 274 clusters enrichment touches.

### What does not move

* `CURTAIL_QUERY_OVERLAP = 0.5` — unchanged.
* `CURTAIL_STREAK = 3` — unchanged.
* `CURTAIL_HISTORY_LIMIT = 24` — unchanged.
* `MAX_LABEL_CHARS = 40`, `MAX_ASK_HINT_CHARS = 80`, `MAX_CLUSTERS = 6`,
  `MEDOID_EXAMPLE_CHARS = 120`, `CACHE_MIN_COVERAGE`, `EMBEDDING_CLUSTER_COSINE` —
  unchanged.
* Nothing in `artifacts/recall-map/prereg.json` or `~/p/ae/docs/agent-stream-prereg.md`
  is touched. **No pre-registered threshold moves in this design.**

New constants are all additive: `LABEL_GATE_TOP_TERMS`, `LABEL_GATE_MIN_IC`,
`ASK_HINT_MAX_TOKENS`, `MIN_MEDOID_EXAMPLE_CHARS`, `MIN_CLUSTERS`, `FILTER_JOURNAL_NAMES`,
`FILTER_JOURNAL_LABEL_CHARS`.

### The uncomfortable half, stated rather than buried

All **688** echo fires come from clusters the gate would withhold: `mmo` 342, `test` 180,
`public` 146, `server` 11, `tools` 9. Against **one** real consumption in the entire field
window.

These are house-vocabulary false positives. `CURTAIL_QUERY_OVERLAP`'s own docstring says
*"a query that merely shares the corpus's house vocabulary with a cluster does not clear
it"* — and with a one-token label, that is precisely what it does. The curtail rule has been
suppressed by echoes the degenerate labels manufactured.

So richer labels would *correctly* tighten consumption detection toward the module's stated
intent. This design **declines to take that**, deliberately:

* tightening it is a change to a pre-registered rule and must be pre-registered and measured
  in the field on its own evidence, not smuggled in as a side effect of a label change;
* the field's consumption evidence is `n = 1`, which cannot support moving a threshold in
  either direction;
* holding the ask-hint short keeps the two changes **orthogonal**, so the polish can be
  judged on label and gist quality without the curtail rate moving underneath it.

The second-order effect of *breadth* is bounded separately and is small: at the breadth
floor of 2, 64 of 67 followed-deliveries survive (§3). Combined worst case for the whole
design — enrichment with terms no query ever contains, plus breadth 2 — is **64 of 67
(−4.5 %)**, and that residual comes from breadth, not from labels.

---

## 7. What the builder nodes must not do

* Do not enumerate observed labels. No blacklist of `{public, test, mmo, server}`, no
  stop-word list beyond the shared `embeddings.tokenize` policy `_terms` already uses. The
  gate is a corpus statistic so that it transfers to a corpus whose house vocabulary is
  different words — that is the whole reason a word list was forbidden.
* Do not move any constant listed under *What does not move*.
* Do not touch `~/p/ae`. If the server emits a gist and honest counts, the ae side needs no
  edit — verified in the consumer table.
* Do not read, replay, or quote a holdout case. `manifest.json` marks the split on every
  case; iterate `split in ("train", "eval")` and assert it.
* Do not change `more` or `covered` semantics, and do not put anything into `clusters` that
  was not delivered.
* Do not tune `LABEL_GATE_MIN_IC` against eval or holdout results. It is placed in an empty
  interval on train+eval and is pinned here; if it proves wrong, that is a finding to
  report, not a number to slide.
