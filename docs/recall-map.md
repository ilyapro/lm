# The recall map — what *else* memory holds, as a plan

A recall returns its top results and throws the rest of the ranked pool away.
The recall map is a compact summary of that discarded residual: a handful of
labelled clusters, each with a count, a medoid example, a phrasing to ask with,
and a line the agent can paste into its own todo plan. It turns the one recall a
session reliably makes — the start-of-session ritual — into a **plan for the
mid-session ones**.

This document covers the whole feature: what the map is, the three surfaces it
is delivered on, the deterministic labeling cascade and the invariant that keeps
an LLM off the recall path, the cache and its invalidation, the curtail rule
that collapses a map nobody consumes, and the pre-registered effect gate. The
last section is a **contract**, not a description: it specifies how an
orchestrator (AE) must pass task identity so the map is stable across the child
sessions it spawns. That section changes nothing in `~/p/ae`; it is the
specification future AE work implements.

## 0. Status at a glance

| Surface / mechanism | State | Where |
| --- | --- | --- |
| Residual pool exposed | shipped (`d72b1de`) | `retrieval.py:614` |
| FTS document-frequency API | shipped (`8e46d2e`) | `storage.py:1970`, `:2015` |
| Map module (cascade, cache) | shipped (`94406c7`) | `recall_map.py` |
| Additive `recall_map` in the recall response | shipped (`a2a89c6`) | `server.py:1128-1146` |
| Delivery persisted + read API | shipped (`a2a89c6`) | `storage.py:1416`, `:1436` |
| Latency gate | shipped, measured | `artifacts/recall-map/latency-before-after.json` |
| Server-instructions section | shipped (`a0d8cab`) | §3.2, `instructions_map.py` |
| Curtail rule (delivered ≠ used) | shipped (`e16693c`) | §6, `recall_map.py` |
| Effect-gate pre-registration | sealed (`6417f57`) | §7, `artifacts/recall-map/` |
| Pool gates (usefulness floor, unfollowed-window demotion) | shipped, **both valves off — no default in code or config** | §11, `recall_map._pool` |
| AE task-context contract | **design, this document** | §9 |

Everything above is described from the code as merged on `a0d8cab`. The one
section still marked *design* is the AE task-context contract (§9): it binds
future orchestrator work outside this repository, not code here.

## 1. Why the map exists

The motivating number is a field measurement, not a hunch. Over 30 goal-tree
sessions on the deployment host (2026-08-19), `memory_recall` was a **startup
ritual**: median call position **0.02** of session length, and exactly **1 of 57
calls** happened mid-session. Every recall trigger in the protocol was bound to
an action boundary, and a linear `--print` pass crosses its biggest boundary
once — at the start. The measurement is recorded at
`src/living_memory/postsession/insights.py:5-7` and pinned as protocol rationale
in `tests/test_instructions_imperative.py:421-438`.

The protocol half of the response already shipped: the mid-work trigger — *"You
MUST recall MID-WORK, at every new turn of thought"* — is law in
`_RECALL_DESCRIPTION` (commit `7bc05ee`). A law is necessary and not sufficient:
an agent told to recall mid-work still has to invent *what to ask*. The map is
the mechanism half. It answers the question the agent has not asked yet, using
the pool the recall already paid to rank.

## 2. What the map is

### 2.1 The residual pool

`MemoryRecallService.memory_recall` ranks the full candidate pool and cuts it at
`max_results`. The tail — `ranked[max_results:]` — is published as
`service.last_residual` (`retrieval.py:614`), reset to `[]` at method entry
(`:561`) so it is always the tail of the *current* call. Exposing it changed no
ranking, no delivery, no event: the map describes the remainder, it never
reorders the top.

On the live corpus the residual is large. In the latency benchmark (24 real
replayed queries) the **median residual was 479 rows** behind a `max_results=5`
delivery. That is the material the map summarizes.

### 2.2 A cluster

Each cluster (`recall_map.py:294`, `MapCluster`) carries:

| Field | Meaning |
| --- | --- |
| `label` | Short name of the pile, ≤40 chars (`MAX_LABEL_CHARS`) |
| `count` | How many residual members it accounts for |
| `medoid.node_id` | The member that speaks for the cluster — feed it to `memory_lookup` |
| `medoid.example` | A quote from that member, ≤120 chars, **budget-squeezable to nothing** (§8.1) |
| `ask_hint` | How to phrase the follow-up recall, ≤80 chars |
| `plan_item` | `on touching <label> - recall '<ask_hint>' (<count>)` |

The payload adds `pool` (residual members considered), `covered` (members the
shown clusters account for) and `more` (clusters the caps dropped). `more` is
never silent: a map that truncated without saying so would read as *"this is
everything memory holds"*, which is the one thing it must never say.

### 2.3 Caps

All caps are module constants (`recall_map.py:136-184`), because two delivery
surfaces budget against them and must not each invent their own number.

| Constant | Value | Binds |
| --- | --- | --- |
| `MAX_CLUSTERS` | 6 | Past this a menu reads as a wall |
| `MAX_RESPONSE_CHARS` | 700 | Compact JSON of `to_dict()` |
| `MAX_INSTRUCTIONS_CHARS` | 150 | `render_compact()`, the instructions line |
| `MEDOID_EXAMPLE_CHARS` | 120 | Longest example before the squeeze |
| `MAX_LABEL_CHARS` / `MAX_ASK_HINT_CHARS` | 40 / 80 | Label and hint width |
| `MAX_POOL_NODES` | 200 | Residual members read per call |
| `EMBEDDING_CLUSTER_COSINE` | 0.55 | Stage-4 admission threshold |
| `CACHE_MIN_COVERAGE` | 0.5 | Cached structure must still cover half the pool |
| `MAX_CACHE_ENTRIES` | 32 | Distinct `(scope, task)` structures kept per process |

The two size caps bind in different regimes. One serialized cluster costs 88
characters of JSON keys before any content, so six clusters spend 528 of the 700
on punctuation: `MAX_CLUSTERS` is the ceiling for short labels, and
`MAX_RESPONSE_CHARS` is the binding constraint for anything richer.

### 2.4 A real example

Built from a `sqlite3`-backup snapshot of the live database (project `lm`,
residual 443 rows behind a `max_results=5` recall for *"how is the recall map
delivered and cached"*):

```json
{
 "clusters": [
  {"label": "recall memory feedback", "count": 27,
   "medoid": {"node_id": "01KZYJFS26HV0NGHAEXFZP42T3"},
   "ask_hint": "recall memory feedback",
   "plan_item": "on touching recall memory feedback - recall 'recall memory feedback' (27)"},
  {"label": "implementation note recall", "count": 10,
   "medoid": {"node_id": "01KRV9EM82F9TE5RVQNPRPHV5X"},
   "ask_hint": "implementation note recall",
   "plan_item": "on touching implementation note recall - recall 'implementation note recall' (10)"}
 ],
 "pool": 200, "covered": 37, "more": 101
}
```

Produced by running one real recall against the snapshot and mapping its
residual — `MemoryRecallService.memory_recall(q, scope='project:lm',
max_results=5, log_access=False, log_event=False)` followed by
`RecallMapBuilder(store).build(svc.last_residual, scope='project:lm',
task_pattern='lm/recall-map')` — the same two calls the server makes
(`server.py:1128-1146`), on a `sqlite3` backup-API copy so the live file is never
opened for writing. The §8.1 table below comes from the same pool.

`pool` is 200, not 443, because `MAX_POOL_NODES` caps what the map reads.
`more: 101` says a hundred further clusters exist and were not shown. The medoid
examples are empty — that is the budget squeeze of §8.1, and the reason the
`node_id` always survives: *"show me one"* stays a single `memory_lookup` away.

The same map rendered for the instructions channel is 78 characters:

```
memory also holds: recall memory feedback(27) · implementation note recall(10)
```

## 3. The three delivery surfaces

In descending priority, because they have very different reach.

### 3.1 Priority 1 — additive `recall_map` in the recall response (shipped)

The handler builds the map after shaping the response and attaches the builder's
output **verbatim** as one more top-level key, beside `auto_decay` — the
additive-key precedent already established (`server.py:1128-1146`):

```python
residual = recall_service.last_residual
if residual and _recall_map_enabled():
    built = recall_map_builder.build(
        residual,
        scope=scope,
        task=_ambient_text(ambient_context, "task"),
        task_pattern=_ambient_text(ambient_context, "task_pattern"),
    )
    if built is not None:
        response["recall_map"] = built.to_dict()
        if recall_service.last_recall_event_id is not None:
            store.attach_recall_map(recall_service.last_recall_event_id, response["recall_map"])
```

Invariants, each pinned by `tests/test_recall_map_delivery.py`:

- **Additive only.** `results` and every other envelope field are byte-identical
  with the map on and off (`test_results_and_envelope_byte_identical_with_the_map_on_and_off`).
  The delivery diet remains in force; the map decorates, it never re-sorts.
- **The server adds no logic.** Whatever the builder returns lands on the wire
  and in the column unchanged
  (`test_the_builders_output_is_attached_and_recorded_verbatim`).
- **Absent, never empty.** No residual → no key. A residual the builder declines
  to describe (`build` returns `None`) → no key, not an empty map.
- **One valve.** `LM_RECALL_MAP=0` removes the key outright and restores a
  byte-identical pre-map response (`server.py:1446`). It is default-*on*, in the
  shape of the `LM_DELIVERY_*` valves.

The map is also **recorded at delivery time**. `recall_events.recall_map`
(schema DDL `storage.py:3484-3517`, additive migration `_migrate_recall_map_column`
at `:3803`) holds the compact JSON of what the agent was actually shown, written
by `attach_recall_map` (`:1416`) — one statement, no read-back, `KeyError` on an
unknown event, because an unattributable delivery is exactly the failure the
column exists to prevent. `recent_recall_map_history` (`:1436`) is the read side
for the three downstream consumers: the instructions channel, the curtail rule,
and the effect gate. It filters on `transport_session_id` / `scope` / `task`,
each an equality probe on an indexed column.

The `memory_recall` docstring tells the agent what the key is
(`server.py:1032-1037`); `_RECALL_DESCRIPTION` itself stays map-silent, because
the tool description is register-tested and the map needs no advert there.

### 3.2 Priority 2 — a session-personalized section in server instructions (shipped)

MCP server instructions are the one channel a client delivers *before the agent
acts at all*. They are also the most tightly budgeted: clients clip at **2048
chars**, and before this work `_server_instructions('global')` measured 2042 of
them — six characters of headroom.

How the shipped surface honours that contract:

- **Compression-funded.** The section was paid for by tightening non-pinned
  connective prose only: the empty-history text now measures **1756 chars** for
  scope `global`, with every phrase `tests/test_instructions_imperative.py` pins
  (the three laws, the bootstrap directive, the mid-work trigger wording, the
  anti-pattern names) surviving byte-identical, and the budget still 2048. Zero
  protocol displacement is a tested property, not an intention: the
  empty-history text is a **byte-exact prefix** of the populated one, and the
  worst section the composer can emit (`MAX_SECTION_CHARS = 240`,
  `instructions_map.py:76`) fits for every default scope up to 56 chars — the
  text ends with the scope name, so total length is `1992 + len(scope)` at the
  maximum, and the contract test parameterizes scope lengths up to 45.
- **No LLM at initialize.** Same invariant as the recall path (§4.1).
- **Signal source.** At `initialize` there is no scope and no ambient context —
  only `clientInfo` and persisted history. The section — headed `## Memory
  nearby` — is composed by `src/living_memory/instructions_map.py` from
  `recent_recall_map_history()` over `recall_events`: what this deployment's
  recent recalls actually mapped. The line mirrors `RecallMap.render_compact()`
  (`recall_map.py:357`) — same `label(count)` shape, same tail-first dropping,
  the mirror pinned in `tests/test_instructions_map.py` — and every label runs
  a screen-scrub-rescreen sanitizer, because labels are user data flowing into
  a register-contract-tested string: one hostile label would otherwise fail the
  imperative suite. Anything dropped turns on a trailing `· …` so a shortened
  map never reads as the whole picture.
- **FastMCP mechanics.** `instructions` is a static `str`, not a callable. The
  server composes it at boot (`_instructions_with_map`, `server.py:569`) and
  `_InstructionsRefresh` (`server.py:595`) re-composes it after recall and
  remember tool calls; streamable-HTTP re-snapshots once per session run, so a
  refresh reaches the *next* session's initialize, while stdio freezes at
  process start — the map reflects state at the last refresh. `LM_RECALL_MAP=0`
  strips the section too, so one valve rolls back both server-owned surfaces.

### 3.3 Priority 3 — the todo-plan-transferable `plan_item` (shipped in the format)

```
on touching <label> - recall '<ask_hint>' (<count>)
```

Constructed at `recall_map.py:1606`; the whole list is available as
`RecallMap.plan_items()`.

This format exists because of a structural fact about how these agents run: a
linear `--print` pass has **no self-observation surface except its own todo
list**. It cannot re-read its context, it cannot notice that a thought turned;
what it *can* do is carry an item forward. So the map's third surface is not a
channel the server owns at all — it is a shape the agent copies:

> - on touching postsession - recall 'postsession extraction gate' (12)
> - on touching contract correction - recall 'contract correction' (3)

Each line is a **conditional trigger with the question pre-written**. "On
touching X" binds to work the agent is going to do anyway; `'<ask_hint>'` is the
query string to send, and at cascade stage 3 that string is a query that already
grounded once (§4.2). The count is there so the agent can rank: a pile of 27 is
a different proposition from a pile of 2.

## 4. The labeling cascade

### 4.1 The invariant: no LLM on the recall path

**No model call happens while building a map, and none may be added.** The map
is built inside the live recall latency budget; a model call there would cost
more than the recall it decorates, and the same rule holds at `initialize`.

LLM enrichment of the labeling substrate is not forbidden — it is *relocated*.
It happens offline, in consolidation and in the post-session extraction stage,
which already writes full structural context keys (`procedure_id`,
`lesson_kind`, `type`, `topic`, `task_pattern`). Every node those stages label
is a node stage 1 of the cascade claims for free, so map quality improves over
time without the recall path ever paying for a token.

### 4.2 Four stages, each claiming what it can

`RecallMapBuilder._cluster` (`recall_map.py:917`) runs the stages in order, each
over what the previous one left.

| # | Stage | Groups by | Label from | Ask-hint quality |
| --- | --- | --- | --- | --- |
| 1 | `structural` (`:726`) | first present of `procedure_id`, `lesson_kind`, `type`, `topic`, `task_pattern` | the key itself, normalized | constructed |
| 2 | `path` (`:757`) | subsystem the node's context paths collapse to | the subsystem name | constructed |
| 3 | `anchor` (`:779`) | the live query anchor whose edges cover the node | the anchor's query text | **evidence** — a query that already grounded |
| 4 | `embedding` (`:835`) | greedy leader clustering over chunk vectors, cosine ≥ 0.55 | c-TF-IDF terms | constructed |

Notes that matter:

- **Stage 1** normalizes with `normalize_key` (`:344`), a deliberate four-character
  mirror of `consolidation._normalize_trigger` — mirrored rather than imported so
  the module stays free of the consolidation stack, and pinned against the
  original in `tests/test_recall_map.py`. When the raw key is opaque —
  `task_pattern` is routinely a content hash — `_looks_unreadable` (`:358`)
  detects it and the label comes from the group's medoid content instead. A hash
  makes a fine grouping key and a useless label.
- **Stage 2** reads *context*, never content. A path in prose is a mention; a
  path in `context['files']` is a claim about what the node is about. It starts
  reading after the last container segment (`src`, `lib`, `tests`, …), which is
  what turns `src/living_memory/postsession/report.py` into `postsession` while
  `src/living_memory/storage.py` stays `living_memory`.
- **Stage 3** is the only stage whose ask-hint is *evidence rather than
  construction*: a query anchor is a past grounded query, so the label doubles as
  a proven way to ask for the cluster. One indexed edge lookup per unclaimed node
  (`idx_query_anchor_edges_target`); a node covered by several anchors goes to
  the heaviest edge, ties to the lower anchor id.
- **Stage 4** labels by c-TF-IDF (`_ranked_terms`, `:926`): `tf * log(N / (1 + df))`,
  with the cluster standing in for the document and `df`/`N` read from the FTS
  vocab index (`term_document_frequencies` `storage.py:1970`, `fts_document_count`
  `:2015`) — so rarity is measured against exactly the corpus recall searches.
  The numerator is deliberately unsmoothed: under `log(1 + N/(1+df))` a term
  present in *every* document still scores 0.69, and three occurrences inside a
  cluster turn that into a winning term — a corpus of memory-server notes gets
  labelled "memory server". Plain `log(N/(1+df))` sends a universal term
  negative, so it cannot win however often the cluster repeats it. Terms are
  looked up **unstemmed**, because the `unicode61` tokenizer that indexes
  `nodes_fts` does not stem and a stemmed term would report `df = 0` and score as
  maximally rare. A store predating the vocab index degrades to plain term
  frequency rather than to nothing.

### 4.3 Determinism

Identical input yields a byte-identical map. Every sort carries an explicit tie
break; the greedy pass walks the pool in rank order; stage precedence is fixed,
so a node with both a `procedure_id` and a file list is a structural cluster
member, never a path one. Stage 4 uses *leader* rather than centroid clustering
precisely so membership cannot depend on arrival order beyond rank, and admission
is *strictly* better than the incumbent so an exact tie keeps the earlier seed.

### 4.4 Uniqueness, ordering, medoid

- **`_disambiguate`** (`:665`) makes every displayed label unique. Two rows
  reading the same text are a defect whatever produced them: the reader cannot
  act differently on them. The first group to claim a label keeps it; later ones
  extend theirs with their own next-most-distinguishing term, falling back to a
  numeric suffix only if no term is left. Colliding groups are **not merged** —
  they were separated by evidence, and a label collision is a failure to *name*
  the difference, not proof it is absent. Uniqueness is decided on the label
  already cut to 40 chars, so two anchor queries sharing a long prefix cannot
  pass this pass and collide again on the wire.
- **Ordering** (`_finish`, `:1077`): largest first (the biggest pile is the
  likeliest next question), then by stage (a label the corpus actually stores
  outranks one derived from term statistics), then by label to make the order
  total.
- **Medoid** (`_medoid`, `:1176`): a cached choice wins whenever that member is
  still in the pool; otherwise, with vectors in play (stage 4 only), the member
  of maximum mean similarity to the rest; everywhere else the highest-ranked
  member, because rank is the only evidence of centrality those stages hold.

## 5. The `(scope, task_pattern)` cache

A task that asks twice should see the same shape twice, or the map is furniture
rather than a plan.

### 5.1 The key

```python
cache_key(scope, task=…, task_pattern=…) -> f"{normalize_scope(scope)}|{normalize_key(label)}"
```

`recall_map.py:1657`. **`task_pattern` wins when both are present**: it names the
recurring *class* of work, which is what should share a structure across
sessions, while `task` is usually one session's phrasing of it. Both are folded
by `normalize_key`, so `postsession-extraction` and `Postsession Extraction` are
one key. With neither, the key degrades to the scope alone — still stable,
merely coarser (§9.5).

The scope half is the tool's explicit `scope` argument when given, else the
dominant scope of the pool itself (`_dominant_scope`, `:1235`).

This one key now keys three things, and that is the point: the structure cache,
the `_CurtailMemo` gate, and — through `recall_events.task_pattern` — the
delivery-history read behind the curtail rule (§6, §9.3). While the last of them
was keyed on `task` alone, a client whose `task` changed every turn had a map
that was stable and a history that was empty.

### 5.2 What is cached, and what is not

The cache stores the cluster **structure** — stage, signature, label, ask-hint,
member ids, and the medoid the full build chose. Counts, membership and medoid
selection are refit against the pool in hand on every call (`_recount`, `:1017`).
Labels and ask-hints survive until the corpus moves.

Refitting is asymmetric by necessity: stages 1 and 2 re-derive membership exactly
(their signatures are pure functions of a node's context), so a node the cached
pool never saw still lands in the right cluster; stages 3 and 4 match by
remembered membership, because a cached embedding cluster has no rule to offer
beyond the ids it was built from. `CACHE_MIN_COVERAGE = 0.5` guards that
asymmetry: when the cached templates no longer cover half the current pool,
`_recount` returns `None` and the caller rebuilds. Stability is worth having only
while it is still stability *about this pool*.

The medoid id is part of the cached structure rather than of the counts, because
stage 4 picks it with vectors the cache path deliberately does not re-read —
without remembering it, the same pool would get one medoid cold and another warm.

### 5.3 Invalidation by revision, never by clock

`_corpus_revision` (`:973`) mirrors `retrieval`'s two-tier probe exactly:

1. A free `(connection.total_changes, PRAGMA data_version)` probe answers *"has
   anything been written at all"*. Most recalls move this with their own
   bookkeeping — `record_access` writes on every call.
2. Only when it moves does tier two ask the tables: `(COUNT(*), MAX(id))` over
   `node_chunk_embeddings` plus `query_anchor_revision()`. Chunk rows are only
   inserted or deleted, never updated in place, so a delete moves the count and
   an insert raises the ULID maximum.

**There are no timers.** A cache that expired on a clock would hand the same task
a different plan for no reason the agent can see.

`MAX_CACHE_ENTRIES = 32` bounds the process-local cache; entries are evicted in
insertion order, and a rebuild re-inserts. The cache lives on the builder
instance, one per server process — it is not shared across processes and does not
survive a restart. Cross-session stability comes from the key being stable, not
from the cache being durable: the same key over an unchanged corpus rebuilds to
the same structure.

## 6. The curtail rule — delivered ≠ used (shipped)

The preprompt-push lesson is binding: a channel that is occupied but not consumed
is worse than an empty one, because it costs budget and teaches the reader to
skim. A map nobody follows must **signal and collapse**, not keep occupying the
response.

The rule is evaluated inside the builder — the server already attaches builder
output verbatim and omits the key on `None`, so no `server.py` change is needed.

- **Consumption primitive** (cheap, live, deterministic, no LLM): a delivered map
  item counts as consumed when a *later* recall or lookup matched it — three
  probes, none of them redundant:
  1. the later `recall_events.query` tokens overlap the item's `label`/`ask_hint`
     above a fixed threshold (reusing the tokenizer of the grounding/embeddings
     rails);
  2. the item's `medoid.node_id` was accessed after the delivery
     (`nodes.last_accessed`, which retrieval stamps on every result it delivers);
  3. a `memory_lookup` id-fetched the item's `medoid.node_id` **inside the
     delivery's frozen outcome window** — read off `recall_delivery_history
     .lookup_consumed` rather than recomputed, so the curtail rule and the M/C/K
     triple cannot end up with two definitions of one delivery's window.

  Probe 3 is the only *exogenous* one. Probes 1 and 2 both reduce to "a later
  recall returned this node again", which is the ranker's own echo: the
  schema-trigger boost mixes hub schemas into nearly every recall, so a hub
  satisfies them forever with no reader anywhere. They stay regardless — in
  curtail every ambiguity resolves against collapsing, so a weak probe is
  evidence, not noise. `lookup_consumed IS NULL` (a window that closed before
  this database recorded any lookup) is *not* evidence: reading unknown as
  followed would switch the rule off for the entire pre-feature corpus.

  Deliveries are read through `recent_recall_map_history()` — indexed reads,
  bounded limit, no new writes — **keyed on `task_pattern` with a `task`
  fallback**, i.e. on the same label `cache_key` uses. See §9.3: before the
  `task_pattern` column this read could only ask for `task`, and against a
  per-turn `task` the window was empty every turn and the streak never
  accumulated at all.
- **Curtail rule**: for a given cache key, if the last **`CURTAIL_STREAK` = 3**
  deliveries each produced zero consumed items, the builder returns a collapsed
  form — `{"clusters": [], "pool": N, "covered": 0, "curtailed": true,
  "streak": S}`, ~60 chars against the map's 700, rendering as the empty string
  in the instructions channel — instead of running the label cascade at all.
  Any consumption resets the streak. The query-overlap threshold is
  `CURTAIL_QUERY_OVERLAP` = 0.5 under the grounding tokenizer
  (`grounding.token_set`); the medoid-access probe reads `nodes.last_accessed`,
  which retrieval stamps on every result it delivers; the lookup probe reads
  `recall_delivery_history.lookup_consumed = 1`, whose window is the frozen
  `RECALL_DELIVERY_HISTORY_HORIZON_HOURS` = 24; history reads are bounded by
  `CURTAIL_HISTORY_LIMIT` = 24.
- **Decaying retry** (`LM_MAP_CURTAIL_DECAY`, default off): with the valve
  unset a collapsed key retries only when its unread offers slide out of the
  24-delivery window — 3 maps in 25, first retry 22 deliveries after the
  collapse. The injection-throttle prereg (П2,
  `~/p/ae/artifacts/injection-throttle/prereg-draft.md`) measured that cadence
  strangling the channel on exactly the quiet armored day the cold lane first
  filled the maps (curtailed share 0.024 → 0.562), so armed, the retry decays:
  after `N` unaccepted offers the key skips `min(2**(N-1),
  CURTAIL_DECAY_SKIP_MAX = 8)` deliveries as markers and then offers a full
  map again — 4 at the entry threshold, 8 from the fourth offer on, the same
  ~12% duty cycle at the plateau but a retry in 4–8 deliveries instead of 22.
  The entry threshold, the marker payload, the probes and the reset are
  byte-identical either way; the valve reads only the `lead` the streak walk
  already counts (the offerless run at the head of the window) and can shorten
  a silence, never start or lengthen one. It binds the cold lane exactly as
  the collapse it relaxes: a reprieved key delivers cold clusters on the same
  delivery a warm map would return.
- **Pool-aware re-offer** (same valve). The first field read of the decay
  (`~/p/ae/artifacts/injection-throttle/read-2026-09-02/READ.md`) found the
  retry firing into nothing: a served run rebuilt the map over a pool with
  nothing deliverable in it, the empty map read as one more offerless
  delivery, the next build retried again, and every delivery after the run
  became an empty *uncollapsed* map — curtailed share ≤ 0.01 with the empty
  share at the storm level (M1 0.77–0.88). So armed, a served run looks
  before it offers. The first marker after an offer persists a digest of the
  key's pool — sorted admitted member ids, the `(member, anchor)` pairs stage
  3 would partition by, the cold-lane candidates the lane would rank — as
  `pd` on the marker it was writing anyway. A served run rebuilds the pool
  (already done on every build) and compares: unchanged, it writes the same
  digest into one more marker and the run starts over from that marker — not
  an offer, so `offers` and the run's width do not move; moved, it builds the
  full map, and if that comes back with nothing deliverable in it, the wire is
  a marker carrying the new digest rather than an empty map. On the empty-pool
  path (cold lane armed) the same rule runs over the lane's eligible
  candidates, and the digest rides the empty map, which never carried
  `curtailed` and still does not. The digest is read back by the same walk
  that reads the streak, from the same window, so a restart or a second
  process sees the baseline; a window written before this rule (no `pd`)
  gets the frozen schedule up to its first look. With the valve unset the
  digest is neither computed nor written, and the marker is byte-identical
  (`test_without_the_valve_the_pool_is_never_looked_at`). Cost of a look: the
  anchor lookups stage 3 would do and one cold-lane ledger probe, once per
  run rather than a full cascade per delivery. The census reads `curtailed`,
  so extended markers count as curtailed, never as empty uncollapsed maps.
- **Default-on, no env flag.** The protocol channels' machinery ban
  (`test_protocol_channels_advertise_no_non_default_machinery`) forbids
  default-off machinery. A curtail rule that has to be switched on is not a
  curtail rule. This is a claim about the *curtail decision*, and it is pinned
  as one: `test_the_rule_needs_no_flag` enumerates every env name the module
  reads, checks each is a registered operator valve, and checks none of them
  appears anywhere in the curtail probes. The decay valve above is the one
  registered exception at the decision itself and keeps the guarantee — it can
  only shorten the silence. The pool gates are off by default for the
  opposite reason — their thresholds are the operator's, not this codebase's —
  and they are advertised in no protocol channel.

The marker is the point. A map that quietly stops appearing is indistinguishable
from a bug; a map that says *"three deliveries, nothing followed"* is a
measurement.

One implementation fact worth knowing: `recent_recall_map_history(scope=…)` is
not the cheap read its name suggests — the query plan walks the whole scope
partition (~8.5 ms at 21k events, growing with the partition).
`idx_recall_events_scope_pattern_created` narrows the pattern-keyed form of that
question to a MULTI-INDEX OR over two bounded `(scope, task_pattern)` ranges —
the pattern itself and the NULL-pattern legacy rows — instead of the whole
partition; the task-keyed form is unchanged and the gate stays either way. The
probe is therefore still
gated by `_CurtailMemo` (`recall_map.py:469`) — keyed by `cache_key` on the same
`(scope, task_pattern or task)` label as the structure cache, or a per-turn
client would seed a fresh memo every turn and pay on every delivery the read the
gate exists to avoid — which counts this
builder's *own* offers per key and pays for a history read only once that count
could cross the threshold. The gate can delay a collapse or buy a needless
read; it can never collapse a map by itself — every collapse is decided by an
actual read. A confirmed collapse is deliberately *not* memoized, so
consumption reopens the channel on the very next recall; that is affordable
because a collapsed build skips the cascade it replaces. Measured cost of the
probe: about +1 ms of map-stage p50, pooled warm p95 overhead ratio 0.0224
against the same pre-registered 0.20 budget (§8).

## 7. The effect gate (sealed)

The gate was **sealed before any post-deployment number was read**, copying
the sealed-holdout discipline the post-session work already uses
(`scripts/postsession_field_run.py`; `artifacts/post-session/field-report.json`
`preregistration` block, with `plan_sha256` frozen over the canonical protocol
fields, a `bar_source` pointing at the artifact that owns the bar, and amendments
allowed only to non-outcome-affecting fields).

Artifacts and where they live:

| Artifact | Contents |
| --- | --- |
| `artifacts/recall-map/prereg.json` | The frozen consumption rule (fixed, do-not-tune style of `usage_metric.CONSUMED_RULE`), the organic-baseline arm definition, the positional-metric protocol, the verdict rule and floors, and `plan_sha256` over the canonical protocol JSON |
| `artifacts/recall-map/baseline.json` | The organic arm and the positional baseline computed **now**, in the pre-feature window, stamped with `as_of` and `plan_sha256` |
| `scripts/recall_map_effect.py` | Read-only measurement tool (`replay.open_readonly`, `mode=ro`), mandatory `--as-of`, both arms through the same code path, delivered maps read from `recall_events.recall_map`, aggregates only — every output must pass `usage_metric.check_privacy`; plus `--verify-prereg`, which recomputes `plan_sha256` and exits nonzero on any mismatch |
| `artifacts/recall-map/latency-before-after.json` | The latency gate, already captured (§8) |

Two endpoints:

1. **Consumption** — did a later recall/lookup follow a delivered map item —
   against a concurrently measured **organic baseline**: consumption of
   comparable residual-pool items in the pre-feature window, where no map was
   delivered. Same tool, byte-identical protocol fields, pinned `as_of` and
   window.
2. **Position** — the fractional position of recall calls within a session, via
   the post-session corpus stats (`scripts/postsession_corpus.py --stats`). The
   pre-feature baseline is the measurement of §1: **median 0.02, 1 of 57 calls
   mid-session**. The map's claim is that the mid-session share rises.

**Anchor-circularity control, pre-registered:** stage-3 labels *are* past query
phrasings, so a later matching query may reflect what the agent would have asked
anyway. The sealed plan reports stage-3-labelled items as a **separate stratum**
next to the primary endpoint. Without that control, the map's best labels are
also its most self-confirming ones.

Seal status: `plan_sha256 =
d17b2e65a20df83968886d307f7f9792491d02cb0ebb406ddeb45b1392e7ad25`, `as_of =
2026-08-19T11:00:00Z`; `--verify-prereg` recomputes the hash and re-checks the
baseline pair, and passes at that `--as-of`. The organic arm and the positional
baseline are captured in `baseline.json` over the pre-feature window; the map
arm is recorded there as *not yet measured* rather than omitted — that run
belongs to the field, by the same tool, after deployment. Amendments live
outside the hashed fields; the channel has been used once, to attribute a
published prior, so the hash is byte-identical to the blind seal.

## 8. Cost, and what it actually costs

The latency gate is pre-registered in the artifact itself: **warm p95 overhead
ratio ≤ 0.2**. Measured over 24 real replayed queries × 21 iterations on a
frozen snapshot (12,893 active nodes, 60,636 chunk embeddings, 3,071 query
anchors; arms rotated every iteration so no arm keeps another's warm page cache):

| Arm | p50 | p95 | map stage p50 | map stage p95 |
| --- | --- | --- | --- | --- |
| map off | 100.0 ms | 707.5 ms | — | — |
| map on, warm builder | 112.4 ms | 707.9 ms | 4.9 ms | 10.3 ms |
| map on, cold builder | 189.8 ms | 754.0 ms | 68.4 ms | 160.9 ms |

Warm p95 overhead ratio **0.0006** — the gate passes with room to spare, and the
p95 is dominated by recall itself, not by the map. The p50 tells the honest
story: a cold build costs ~90 ms of median wall clock and a warm one ~14 ms. The
cache is not a nicety; it is what makes this affordable. In the benchmark every
mapped query hit the warm cache (`warm_cache_hit_share: 1.0`), and every one of
the 24 queries produced a map (median 3 clusters).

Reproduce:

```bash
python3 scripts/recall_map_latency_bench.py \
    --snapshot /tmp/anchor-eval/snap-anchored.sqlite3 \
    --out artifacts/recall-map/latency-before-after.json \
    --queries 24 --iters 21 --warmup 2 --max-results 5 --seed 20260820
```

### 8.1 Known limitation: the medoid example is squeezed first, and not reflowed

`_fit` (`recall_map.py:1552`) squeezes medoid examples down before dropping
clusters, and once breadth-dropping starts it does not reflow the examples. On
the live corpus this is not a corner case — it is the normal outcome. Measured on
the snapshot, identical pool (project `lm`, residual 443):

| `max_clusters` | clusters shown | payload chars | example lengths |
| --- | --- | --- | --- |
| 6 | 2 | 505 | 0, 0 |
| 3 | 2 | 505 | 0, 0 |
| 2 | 2 | **699** | 84, 84 |

So a map that had to drop any cluster ships with **no examples and ~195 chars of
its budget unused**. The advertised "label, count, medoid example" degrades to
"label, count" exactly when the residual is richest. The `medoid.node_id` always
survives, so the example is one `memory_lookup` away — but a reader budgeting on
the documented shape should expect `example` to be absent, and `MapMedoid.to_dict`
omits the key entirely when empty rather than emitting `""`.

### 8.2 Other bounded behaviours

- **`MAX_POOL_NODES = 200`.** With a median residual near 480, the map routinely
  describes the top 200 of the tail and the rest is invisible. `pool` reports what
  was considered, not what existed.
- **The cache is process-local.** A restart rebuilds; a second server process has
  its own cache. Stability across sessions is a property of the *key*, not of the
  cache's lifetime.

## 9. The AE task-context contract

> **Design only.** This section specifies what an orchestrator must send. It
> changes nothing in `~/p/ae`, and it deliberately uses **only signals the server
> already accepts** — no new tool parameters, no schema change, no server work.
> It is the contract future AE work implements.

### 9.1 The signals the server already accepts

Every field below travels in `memory_recall.ambient_context` (and in
`memory_remember.context` / `memory_teach.context` under the same names).

| Field | Where it lands | Consumed by |
| --- | --- | --- |
| `task` | `recall_events.task` **column** (`storage.py:1376`) | Map cache key (fallback); `recent_recall_map_history(task=…)`; feedback closure |
| `task_pattern` | `recall_events.task_pattern` **column** — and the ambient JSON, unchanged | Map cache key (**preferred**, `recall_map.py:1657`); `recent_recall_map_history(task_pattern=…)` and the curtail rule; `memory_lookup(task_pattern=…)`; consolidation grouping (`consolidation.py:662-669`) |
| `session_id` | `recall_events.session_id` column (`:1377`) — **and scope resolution** | Pending-recall matching; `_ambient_session` (see §9.4) |
| `agent` | `recall_events.agent` column (`:1375`) | Attribution, closure precedence |
| `scope` | `recall_events.scope` / `requested_scope` | Everything |
| `transport_session_id` | `recall_events.transport_session_id` column | Session dedup, closure — **server-stamped; clients MUST NOT set it** |

`transport_session_id` is derived server-side from the MCP transport
(`server.py:392-433`) and stamped into the context only if the caller did not
supply it — an explicit value always wins, which is exactly why a client must not
supply one. Setting it forges the correlation identity that session dedup and
feedback closure depend on, and there is no legitimate client-side reason to: the
transport already has the answer.

### 9.2 What the cache keys on — and what AE sends today

The map cache keys on **normalized `task_pattern`, falling back to `task`**
(`recall_map.py:1657`), folded by `normalize_key`: `_`, `-` and `/` become
spaces, then lowercase.

Measured on the live database, 2026-08-20, read-only (54,917 recall events):

| Signal | Events carrying it |
| --- | --- |
| `task` | 5,415 (9.9%) |
| `agent` | 5,206 (9.5%) |
| `session_id` | 4,889 (8.9%) |
| `transport_session_id` (server-stamped) | 7,896 (14.4%) |
| **`task_pattern`** | **281 (0.5%)** |

So today the map key is effectively `task`, and `task` is not the shape the cache
wants. AE's `lm_client.py` documents `task` as *"goal path (`<tree>/<node>`) or
task id"*, and in practice **1,107 of the 5,415 attributed events carry an
absolute filesystem path**:

```
/home/sfx/p/ae/projects/lm/state/goals/recall-map/children/ae-contract-doc
```

328 distinct such strings, collapsing to only **187 distinct tree-goals**. Fed to
the cache that becomes:

```
project:lm|home sfx p ae projects lm state goals recall map children ae contract doc
```

Two defects in one key: it embeds the **host filesystem path** (so the same goal
run on another host, or from a `/tmp` test harness, is a different task), and it
descends to the **child node** (so each of a tree-goal's eight children is a
different task). The map's whole promise — one task, one structure — is defeated
by an identifier that is neither stable nor shared.

### 9.3 The contract

On every Living Memory tool call made by AE, or by an agent session AE spawns,
send:

```json
{
  "ambient_context": {
    "scope": "project:lm",
    "task_pattern": "lm/recall-map",
    "task": "recall-map/ae-contract-doc",
    "agent": "ae:tree_node",
    "session_id": "<invocation id>"
  }
}
```

1. **`task_pattern` — the stable slug, one per tree-goal. This is the load-bearing
   field.** Recommended format `<project>/<root-goal-id>`: the AE goal id is
   already a stable kebab slug (`projects/<project>/state/goals/<goal-id>/`), it
   is host-independent, it is identical for every child node and every retry, and
   it is exactly the identity a map should be stable across. It must **not**
   contain a filesystem path, a worktree name, a timestamp, a PID, or a child-node
   id. Keep it short: it is normalized to words, and a 70-word key is a key nobody
   can read in a log.
2. **`task` — the specific node identity, kept.** Use the logical goal path
   `<root-goal-id>/<child-id>[/<grandchild-id>]`, **not** the absolute state
   directory. It stays useful: it is a real column, it drives feedback closure and
   `recent_recall_map_history(task=…)`, and it is the cache fallback when a
   caller has no pattern.
3. **`agent` — `ae:<subsystem>`**, the vocabulary AE already documents
   (`ae:tree_node`, `ae:prompt_engine`, `ae:gate`, `ae:node_clarification_preflight`).
   The map does not read it; the effect gate's arm attribution does.
4. **`session_id` — the invocation id — only together with an explicit `scope`.**
   See §9.4.
5. **`transport_session_id` — never set it.** §9.1.

Both `task` and `task_pattern` are still required, not either/or, but the reason
has changed and the change is worth stating plainly, because the old reason was
a documented debt and this section is where it was recorded.

**Debt paid: `task_pattern` is a column.** It used to live in
`recall_events.ambient_context` JSON only, which meant
`recent_recall_map_history` — the read API behind the instructions channel, the
curtail rule and the effect gate — could filter on `scope`, `task` and
`transport_session_id` and on nothing else. The map keyed its cache on the
pattern; the delivery history could only be grouped by the task. For a stable
`task` that was merely coarse. For the client this contract is written for it
was fatal: an AE chat sends `task=chat:<id>/turn-<N>`, unique every turn by
§9.3(2), so **every turn read an empty delivery history, every turn was the
first delivery, and `CURTAIL_STREAK` could never be reached** — the map could
not be curtailed in exactly the channel §9.5 says curtailment matters most for.

`recall_events` now carries a `task_pattern` column, populated on write from the
same ambient context `task` is lifted from, backfilled from the ambient JSON of
every row that ever carried one, indexed by
`idx_recall_events_scope_pattern_created`, and read by
`recent_recall_map_history(task_pattern=…)`. A row that carries no pattern —
every row written before the column, and every row from a client that sends
none — falls back to its `task`, which is the only identity it has. The
fallback is deliberately not a wildcard: a pattern-less row is matched only when
its `task` matches too (`recall_map._delivery_history`).

So the two fields are read by different consumers still, and both remain
required:

- **`task_pattern`** now keys *both* the cache and the delivery history, which
  is what makes a streak accumulate across turns.
- **`task`** remains a real column driving feedback closure and
  `recent_recall_map_history(task=…)`, remains the cache fallback for a caller
  with no pattern, and is what a pattern-less legacy row is recognized by.

A client that sends only `task_pattern` now gets a stable map *and* a filterable
delivery history. A client that sends only `task` is unchanged in every respect.

### 9.4 The `session_id` trap

Scope resolution derives the requested scope from `session_id` when no explicit
scope is given: `_ambient_session` (`scope.py`) returns `session:<id>`, and
`recall_events.scope` stores the *requested* scope (`retrieval.py:622`). The
search itself covers the whole store — the session and ambient project only rank
first — so results are not lost. What is lost is every downstream filter:
each session writes its events under a scope value no other session shares.

Measured: the live database holds **53 recall events across 18 distinct
`session:*` scopes, and exactly 0 nodes in any session scope**. Those events are
invisible to `recent_recall_map_history(scope='project:lm')`.

**Therefore: whenever AE sets `session_id`, it must also set an explicit `scope`
(the `scope` tool argument, or `ambient_context.scope`).** An explicit scope
always wins over the ambient derivation (`ScopeResolver.resolve`).

### 9.5 What AE gains

- **A stable map structure across child sessions.** All eight children of the
  `recall-map` tree hit `project:lm|lm recall map`, so the first child's recall
  builds the structure and every sibling sees the same labels, in the same order,
  until the corpus itself moves. Cross-session stability comes from the key, not
  from a shared cache (§5.3) — which is why the key must be the thing that is
  shared.
- **Cache economy.** `MAX_CACHE_ENTRIES = 32` per process. Keyed on the
  tree-goal, the live corpus's 1,726 distinct task strings become 187 patterns,
  and a tree in flight occupies exactly one entry. Keyed on `task`, one tree's
  children evict each other.
- **Instructions personalization.** The §3.2 channel composes from persisted
  history; with a real scope and a real `task` column it can say *what this
  deployment's recent work mapped* instead of averaging over everything.
- **A curtail rule that means something.** §6 counts consumption per cache key. A
  key that changes every node makes every delivery the first delivery, and the
  streak never reaches K — the map can never be curtailed, which is the failure
  mode the rule exists to prevent. Since the `task_pattern` column landed the
  server holds up its half: send the pattern and the streak accumulates across
  turns and across child sessions. Send only a per-turn `task` and it still
  cannot, because there is then nothing that says the turns are one channel.
- **An effect gate that can stratify.** Arm assignment and the anchor-circularity
  stratum both need to group events by the work they belong to.
- **Better offline consolidation, free.** `task_pattern` is consolidation's
  preferred grouping key (`consolidation.py:656-669`) — the same slug that
  stabilizes the map also groups traces into procedural schemas, which feeds
  cascade stage 1 back with better structural keys.

### 9.6 What happens when the fields are absent

Nothing breaks. The degradation is defined and monotonic:

| Sent | Cache key | Consequence |
| --- | --- | --- |
| `task_pattern` (+ `task`) | `project:lm\|lm recall map` | Intended: one structure per tree-goal, filterable history |
| `task` only | `project:lm\|recall map ae contract doc` | One structure per node; siblings do not share; 32-entry cache churns |
| neither | `project:lm\|` | **Scope-level map**: every task in the scope shares one structure. Still stable and still correct — the map describes the residual it is given — merely coarser, and the labels drift as unrelated work moves through the scope |
| no `scope` either | dominant scope of the pool | The map still names a scope; the recall event may record `global` or `session:<id>` |

The scope-level map is a legitimate operating point, not a failure: an
unattributed client — a human at a CLI, an agent with no orchestrator — gets a
map of what its scope holds. The contract exists to buy *sharpness*, not
correctness.

### 9.7 Conformance check

From the Living Memory side, over any window of live events:

```sql
-- share of recent recalls carrying a usable map key
SELECT
  COUNT(*)                                                        AS events,
  SUM(task IS NOT NULL AND task <> '')                            AS with_task,
  SUM(task LIKE '/%')                                             AS absolute_path_task,   -- must be 0
  SUM(task_pattern IS NOT NULL AND task_pattern <> '')            AS with_task_pattern,
  SUM(scope LIKE 'session:%')                                     AS session_scoped        -- must be 0
FROM recall_events
WHERE created_at >= :since AND agent LIKE 'ae:%';
```

`task_pattern` is read from its own column now, not out of the ambient JSON. The
two agree by construction on anything this build wrote, and the backfill made
them agree on everything older that carried a string value; a row where they
differ is a row whose ambient context held something other than a string under
that key.

Conformant AE traffic has `with_task_pattern = events`, `absolute_path_task = 0`
and `session_scoped = 0`. Open the database read-only (`mode=ro`) — the live file
belongs to the running server.

Baseline reading, `:since = '2026-08-01'`, live database 2026-08-20:

```
events 1121 | with_task 1121 | absolute_path_task 1106 | with_task_pattern 0 | session_scoped 0
```

That is the gap this contract closes: AE already attributes every call, already
avoids the `session_id` trap, and fails both remaining clauses — no
`task_pattern` at all, and a task that is an absolute host path 99% of the time.

## 10. Reference map

| Concern | File |
| --- | --- |
| Residual pool | `src/living_memory/retrieval.py:537`, `:561`, `:614` |
| Map module | `src/living_memory/recall_map.py` |
| Response wiring, valve, ambient reads | `src/living_memory/server.py:1128-1146`, `:1446`, `:1457` |
| Transport identity stamping | `src/living_memory/server.py:392-433` |
| Instructions section (composer, boot, refresh) | `src/living_memory/instructions_map.py`; `src/living_memory/server.py:435`, `:569`, `:595` |
| Curtail rule | `src/living_memory/recall_map.py:202-222`, `:469` |
| Curtail key alignment (`task_pattern` column, read, fallback) | `src/living_memory/storage.py` `_migrate_recall_events_task_pattern_column`, `_backfill_recall_events_task_pattern`, `recent_recall_map_history`; `recall_map._delivery_history` |
| Curtail lookup probe (exogenous follow signal) | `recall_map._medoid_lookups`, `_was_followed`; ledger column `recall_delivery_history.lookup_consumed` |
| Pool gates behind env valves (§11) | `recall_map.pool_usefulness_floor_from_env`, `pool_demotion_windows_from_env`, `_below_usefulness`, `_unfollowed_run`, `_demoted`, `RecallMapBuilder._pool`; codes `POOL_GATE_REASON_CODES` |
| Effect gate (sealed) | `scripts/recall_map_effect.py`; `artifacts/recall-map/prereg.json`, `baseline.json` |
| Delivery persistence + history | `src/living_memory/storage.py:1416`, `:1436`, DDL `:3484-3517`, migration `:3803` |
| Ambient → column lift (`agent`, `task`, `task_pattern`, `session_id`) | `src/living_memory/storage.py:1374-1378` |
| FTS document frequencies | `src/living_memory/storage.py:1970`, `:2015`; vocab DDL `:3443` |
| Scope resolution | `src/living_memory/scope.py` (`ScopeResolver`) |
| Tests | `tests/test_recall_map.py`, `tests/test_recall_map_delivery.py`, `tests/test_recall_map_pool.py`, `tests/test_recall_map_curtail.py`, `tests/test_fts_vocab.py`, `tests/test_instructions_map.py`, `tests/test_instructions_refresh.py`, `tests/test_instructions_imperative.py` |
| Latency artifact | `artifacts/recall-map/latency-before-after.json` |
| Bench script | `scripts/recall_map_latency_bench.py` |

## 11. Pool gates behind env valves (both off in code and in config)

`relevance_score` (§2.1, `recall_map.py`) reads exactly two things: whether the
node is a schema, and the matured M/C/K triple of its delivery history. That is
the frozen `directional-zsum-r1` policy and it is not being touched here —
`RELEVANCE_FEATURE_MEANS`, `RELEVANCE_FEATURE_SCALES` and `RELEVANCE_THRESHOLD`
stay byte-for-byte what the sealed evaluator fitted.

Two things it therefore cannot see:

1. **The usefulness verdict.** A node with `usefulness_score = 0.078` is not
   distinguishable from one at `0.9`; it stays a medoid indefinitely.
2. **Whether anybody ever followed the row.** The K tail (`trailing_nonconsumed`)
   was designed to demote unfollowed rows, but its notion of "followed" is
   re-delivery — a later recall returning the node again. The schema-trigger
   boost (1.8) mixes hub schemas into nearly every recall, so a hub carries
   `consumed == matured`, `K = 0` forever. One live example carried 4 616
   re-deliveries and never sank.

Both signals now enter, and they enter **only** as admission rules applied
around the frozen selector inside `RecallMapBuilder._pool`. Neither is on by
default, in code or in config. That is deliberate and follows the
`LM_DRAIN_NEAR_DUP_SUPERSEDES` precedent (`retrieval.py`): the numbers come from
the field census of known-window outcomes, not from an argument in a review, so
the operator arms them from measured values.

### 11.1 The valves

| Gate | Valve (`1`/`true`/`yes`/`on`) | Value |
| --- | --- | --- |
| (a) usefulness floor | `LM_MAP_POOL_USEFULNESS_GATE` | `LM_MAP_POOL_MIN_USEFULNESS` |
| (b) unfollowed-window demotion | `LM_MAP_POOL_DEMOTION_GATE` | `LM_MAP_POOL_DEMOTE_AFTER` |

Two variables per gate, mirroring the drain's `SUPERSEDES` + `COSINE` pair, with
one difference that matters: **the value has no shipped default.** The drain
ships a measured cosine and the flag decides whether to use it; here the number
*is* the operator's decision, so a valve turned on without one is inert. There
is no way to enable either gate without saying what number enables it, and
nothing in this repository or its config sets any of the four.

With both valves unset the builder's behaviour is byte-identical to the build
that predates them — not "equivalent", byte-identical, including the persisted
selection journal. That is pinned by
`tests/test_recall_map_pool.py::test_both_valves_unset_is_byte_identical_to_the_pre_gate_build`
against `PRE_GATE_GOLDEN`, a payload captured by running the fixture scenario
against `recall_map.py` at commit `e579fb4` — the parent of the commit that
added the gates. The golden is a literal in the test file, with the recapture
command beside it, because a golden the suite can regenerate is a golden that
re-blesses whatever the code does today.

### 11.2 Gate (a): the usefulness floor

A candidate whose `usefulness_score` is **strictly below** the floor is excluded
from the pool before its history is read, so a rejected row costs no ledger row
either. It leaves its own trace in the selection journal under reason code
**`uf`**.

`nodes.usefulness_score` is `NOT NULL DEFAULT 0.0`, so a never-scored row is a
genuine zero and is judged like one — the census the floor comes from counted
those zeroes. The "unknown" case the code guards is a *result object* from
another layer with no such attribute; that is read as no verdict, never as a
bad one.

### 11.3 Gate (b): demotion after N known unfollowed windows

A row is demoted — excluded from the pool, reason code **`nf`** — when it has
been delivered into **N consecutive known windows that nobody followed**, where
N is `LM_MAP_POOL_DEMOTE_AFTER`.

- **Known windows only.** The run is `MaturedRecallHistory.lookup_trailing_absent`
  (§6, `recall_delivery_history.lookup_consumed`): newest-first windows with no
  id-fetch, stopping at the first followed one, with `NULL` windows **skipped**
  rather than counted. `NULL` means "this window closed before the database
  recorded lookups at all", which is not evidence that nobody followed it. A
  corpus that has never recorded a lookup demotes nothing.
- **Exogenous follows only.** Two probes hold a row: the id-fetch (from the
  ledger) and the ask-echo — a later query under this key covering
  `CURTAIL_QUERY_OVERLAP` of the cluster's label or ask-hint, the same probe
  `_was_followed` uses, harvested from the curtail read the build already paid
  for. The third curtail probe, the medoid's `last_accessed`, is deliberately
  **not** consulted: any recall that returns the node bumps it, so a gate that
  accepted it could never sink a hub, which is the entire job.
- **Re-delivery softens, never vetoes.** The row's re-delivery share
  (`consumed / matured`) shortens the effective run by
  `DEMOTION_REDELIVERY_WEIGHT` = 0.5 of itself. A row nothing ever re-delivered
  sinks after N windows; a row re-delivered on every single window needs 2N;
  everything else lands in between. The weight is strictly inside `(0, 1)`
  because `0` is no softening and `1` is the veto that makes hubs immortal —
  and *which* value inside the interval is not a threshold to tune, since the
  comparison is against the operator's N, which absorbs any fixed scaling.

Ordering inside `_pool` is load-bearing: the demotion runs **after** the frozen
`RELEVANCE_THRESHOLD` has spoken, so `nf` counts exactly the rows the gate newly
removed and `lr` keeps reporting exactly what it always reported.

### 11.4 Cost, and what does not become a per-row query

Neither gate adds a query, and neither turns a batched read into a per-row one:

- the usefulness score is already on the `Node` the residual carries;
- the window aggregates come from the same bounded, batched
  `matured_recall_history` call `_pool` already makes (batches of
  `MAX_RECALL_HISTORY_CANDIDATES` = 200);
- the ask-echo set is a by-product of `_read_curtailment`, computed from rows
  and tokens already in hand. Arming gate (b) moves that read *before* the
  pool instead of after it — same read, same `_CurtailMemo` gate, carried
  forward so the build never asks twice.

The ask-echo evidence is therefore exactly as stale as the memo curtail already
trusts, and scoped to this cache key. Both limitations point the same way — less
ask evidence means more demotion — and the mitigation is that the operator's N
is measured on the same evidence the gate reads.

Measured with `scripts/recall_map_latency_bench.py` over
`/tmp/anchor-eval/snap-anchored.sqlite3` (20 replayed real queries, 7 timed
iterations per arm, median residual 512), against the pre-registered
`BUDGET_P95_OVERHEAD_RATIO` = 0.20:

| Run | Pooled warm p95 overhead ratio | Within budget | Warm paired Δ median |
| --- | --- | --- | --- |
| both valves unset | **0.1771** | yes | 74.6 ms |
| both valves armed (`0.25`, `N=3`) | **0.1915** | yes | 63.5 ms |

Read the second row for what it is. That snapshot predates the lookup recorder,
so every window in it is `NULL` and gate (b) demoted nothing — the run bounds
the *cost of arming* the gates (the reordered curtail read, the per-candidate
arithmetic), not their effect. The two paired deltas straddle each other because
the gates remove rows the cascade would otherwise have had to label; the honest
reading is that arming them is inside the noise of this bench, not that it is
free.

### 11.5 The selection journal

`SELECTION_REASON_CODES` — the seven-wide positional `x` vector in the additive
`sel` payload — is unchanged. `POOL_GATE_REASON_CODES = ("uf", "nf")` appends
after it, and `_SelectionLedger.freeze` trims trailing zeros back to the frozen
width, so:

- with both valves unset, `x` is exactly seven entries, as before;
- a longer `x` is itself the statement that a gate fired;
- indices 0–6 keep meaning what they meant to every existing reader.

Both gate codes are **sampleable**: like `lr` and `pc` they report a policy
decision that is otherwise invisible, and unlike any other code their threshold
is something an operator typed. A journal that says "the floor dropped 14"
without ever naming one of them gives that operator no way to tell a floor that
is working from a floor set one decimal place too high. The gist is
identity-free and capped at `SELECTION_SAMPLE_GIST_CHARS` = 24, as everywhere
else.
