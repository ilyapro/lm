# The recall map

A recall map summarizes useful knowledge left in the current recall's residual
pool. Its clusters supply labels, counts, representative node ids and queries
for follow-up recalls. The main recall still owns ranking, result delivery,
corrections and scope resolution. The map is an optional addition to that answer.

## 1. Recall and delivery

`MemoryRecallService.last_residual` holds the current call's ranked remainder
after result selection. It is reset on every call. `RecallMapBuilder` selects
and groups members of that residual without changing the main results.

The server constructs the map after shaping the required response. It records
the complete map with `attach_recall_map` on the same recall event before adding
`recall_map` to the response. The persisted JSON is the payload delivered to the
caller with `view="full"`. The default knowledge view omits only diagnostic
selection accounting (`sel`); navigation, filtered labels and limitations remain
identical. A missing residual or a builder returning `None` produces no map key.
A measured pool with no eligible members may produce an empty map.

Map construction and attribution share a nonfatal boundary: an optional-map
failure leaves the required recall response usable. Errors in the required
recall path retain their normal behavior. `LM_RECALL_MAP=0` disables the optional
map. It does not change the main results or their feedback protocol.

MCP initialization serves static instructions: identity, recall/remember/teach
laws, the tool bootstrap and default scope. Initialization has no current-work
identity, so persisted maps are not injected into that text. Recall and remember
calls do not rewrite the instructions received by subsequent clients.

`instructions_map.py` provides a pure compact formatter and an explicitly
filtered history formatter for callers that already know which work they are
describing. They do not constitute an automatic initialization channel.

## 2. Payload and bounds

Each cluster carries:

| Field | Meaning |
| --- | --- |
| `label` | A short, distinct name for the group |
| `count` | Residual members represented by the group |
| `medoid.node_id` | A representative node that `memory_lookup` can fetch |
| `medoid.example` | A budget-dependent excerpt; omitted when squeezed away |
| `ask_hint` | A query for a follow-up recall |
| `plan_item` | `on touching <label> - recall '<ask_hint>' (<count>)` |

An agent can carry `plan_item` into its own plan. It is a conditional reminder
to recall relevant knowledge when that part of the work becomes current. The
server does not create or manage the agent's plan.

The map adds `pool`, `covered` and, when needed, `more`. These describe the
considered pool, represented members and omitted clusters; they do not claim
to describe every memory in the store. `filtered` records withheld labels and
budget exclusions. The additive `sel` journal records candidate selection and
identity-free exclusion samples. It is stored with the complete map and exposed
through `view="full"` for diagnostics.

| Constant | Value | Purpose |
| --- | --- | --- |
| `MAX_CLUSTERS` | 6 | Maximum displayed groups |
| `MAX_RESPONSE_CHARS` | 700 | Compact JSON budget |
| `MAX_INSTRUCTIONS_CHARS` | 150 | Compact formatter's line budget |
| `MEDOID_EXAMPLE_CHARS` | 120 | Maximum representative excerpt |
| `MAX_LABEL_CHARS` / `MAX_ASK_HINT_CHARS` | 40 / 80 | Label and query width |
| `MAX_POOL_NODES` | 200 | Admitted residual members |
| `EMBEDDING_CLUSTER_COSINE` | 0.55 | Stage-4 grouping threshold |
| `CACHE_MIN_COVERAGE` | 0.5 | Minimum cached-template coverage |
| `MAX_CACHE_ENTRIES` | 32 | Structures retained per builder |

The response fitter searches for the widest common excerpt after each change
in displayed breadth. It drops tail clusters and optional journal names to
preserve a useful gist. If the minimum useful presentation cannot fit, it
retains selection accounting without clusters in the stored map and diagnostic
view. `memory_lookup` remains the
full-content path. A fully skipped curtail decision omits `pool` and `sel`:
neither was computed. A measured empty pool includes `pool: 0`; its selection
accounting remains available in the stored map and diagnostic view.

## 3. Current-work identity

The same work identity governs structure caching, the curtail memo, persisted
delivery history and attribution of follow-up activity. It consists of the
resolved recall scope plus the following identity:

| Available context | Identity |
| --- | --- |
| Nonempty `task_pattern` | That explicit work label |
| Otherwise, nonempty `task` | That explicit work label |
| Otherwise, a transport session | That transport |
| None of these | Unknown work |

Explicit labels are compared as exact strings, not normalized into words.
Punctuation and case can distinguish independent work. `task_pattern` takes
precedence when both fields are present. A label supplied through either field
identifies the same work when it is the effective label. A different pattern
does not inherit a task's history merely because both calls also carry the
same `task`.

An explicit work label can continue across connections and process restarts.
Sharing a label intentionally shares that work's delivery history and curtail
state. Independent consumers should not be assigned one shared label simply
because their tasks concern the same technical topic.

Without an explicit label, the server derives `transport_session_id` from the
actual MCP request context. Clients do not need to invent a task or session id.
The transport is a fallback identity, not a knowledge-search restriction.
When no identity is available, the builder constructs the current map without
using the work cache, curtail memo or another caller's delivery history.

The map scope is the scope resolved for the recall event. It is not inferred
from whichever project happens to dominate the residual. This keeps the cache
and persisted evidence aligned while preserving the normal recall search plan.
Omitting an explicit search scope continues to permit reusable knowledge from
other projects; work identity does not narrow that search.

## 4. Selection and labeling

`QueryRelevance` shares the recall quality gate's evidence with the map. It
keeps the full ranked set, scope plan, causal mode, query demotions and correction
edges for this call. With `LM_RECALL_MIN_SCORE` configured, candidates must pass
that same threshold or the existing trigger-availability rule. A present
correction can inherit its original's relevance through the ranked correction
chain. Every rank uses the same qualification.
Without a configured threshold, positive channel evidence or a qualifying
trigger suffices; residual membership alone does not.

The evidence is evaluated lazily inside the optional-map failure boundary when
the main gate is disabled. Map construction never changes the main selection,
its ranking, scope resolution, decay or correction validity. Explicit id lookup
remains available for full knowledge, including entries the map does not offer.

All candidates use the same query qualification regardless of their age or
exposure history. A never-shown or rarely used entry needs no separate quota.
Query anchors, positive feedback and history-derived graph weights can aid
retrieval and its order; they do not qualify a map suggestion. The shared
qualifier uses query-term coverage of current source text and transient source
graph evidence. Reciprocal FTS rank remains discovery/order information and
does not qualify map content or seed source activation. The qualifier retains source confidence,
correction and query-relative/hub restrictions on both channel and trigger
scales. The net usefulness accumulator does not enter either scale. No map-specific corpus delivery count,
consumed ratio or nonuse streak admits or orders a candidate.
The optional stored-usefulness floor rejects values below
`LM_MAP_POOL_MIN_USEFULNESS` only when `LM_MAP_POOL_USEFULNESS_GATE` is enabled.
Malformed or unavailable usefulness supplies no exclusion evidence.

After identity, ballast and query checks, the bounded pool shares capacity
round-robin among structural and path groups, with an additional bucket for
members awaiting anchor/vector grouping. Groups and members enter in retrieval
order. This keeps a large known topic from taking every slot before a relevant
minority topic is considered. Anchor and vector work runs only on the admitted
pool. Unstructured members share one capacity bucket; the later cascade can
separate them into multiple anchor or embedding clusters.

Repeated map nonuse belongs to the current-work curtail rule in §6. The selection
journal retains positional reasons `iv`, `du`, `fc`, `ss`, `sj`, `lr`, `pc`;
`lr` means the candidate lacks this query's qualifying evidence. Optional `uf`
records the usefulness floor, and `nf` is reserved for stored journal readers.
Trailing empty positions are omitted.

No model call occurs while building a map. The builder uses structural context,
query anchors, stored chunk vectors and FTS document frequencies. Offline
consolidation and extraction can improve those inputs without adding inference
to the live map path.

The labeling cascade claims members in order:

| Stage | Grouping | Label and follow-up query |
| --- | --- | --- |
| Structural | First available `procedure_id`, `lesson_kind`, `type`, `topic`, `task_pattern` | Context value; opaque values use representative content |
| Path | Subsystem inferred from context paths | Subsystem name enriched with distinguishing terms |
| Anchor | A grounded query anchor covering the node | The anchor's query, a previously grounded way to ask |
| Embedding | Greedy leader grouping over stored chunk vectors | c-TF-IDF terms |

Context-key normalization is a labeling operation and does not normalize work
identity. Path grouping reads context paths, not incidental path mentions in
prose. Anchor assignment prefers the heaviest covering edge with an explicit
tie-break. Embedding labels use `tf * log(N / (1 + df))`; terms are unstemmed
to match the FTS vocabulary. Universal corpus terms cannot win merely by
repetition. Structural and path enrichment and the label-information gate use
the same document-frequency evidence.

Identical input and policy produce deterministic groups. Every sort has a
tie-break and each stage receives only members unclaimed by earlier stages.
Label disambiguation preserves distinct groups and makes their displayed names
unique after truncation. Display order follows each cluster's best original retrieval ordinal. Cluster
size and corpus exposure do not determine its position. Embedding groups choose the
most central member; other groups use the best original residual ordinal.
Representatives and presentation order are recomputed from the current pool.

## 5. Structure cache

The process-local cache stores cluster stage, signature, label, query and member
ids. Every offered map refreshes counts, membership and representatives against
the current pool. Structural and path templates can classify newly arriving
members; anchor and embedding templates retain known membership. Unmatched
members cause a rebuild. Embedding membership is checked against fresh vectors
and the current ranked pool before cached labels are reused. Scores, order and
representative identity do not survive as cached decisions.

Corpus revision probes use `connection.total_changes` and `PRAGMA data_version`
before checking chunk and query-anchor revisions. Corpus changes invalidate dependent structure. There is no timer-based expiry. Up to 32 work
entries are retained per builder, with insertion-order eviction.

The cache is not persisted or shared between processes. A cold process rebuilds
from the current residual. Persisted delivery history still identifies the same
explicit work, so restarting does not erase its nonuse history. Unknown work
creates no reusable entry.

## 6. Curtailment and follow evidence

For identified work, three unaccepted map offers (`CURTAIL_STREAK = 3`) cause
the builder to return a compact marker:

```json
{"clusters": [], "covered": 0, "curtailed": true, "streak": 3}
```

Offers and markers are read from that work's bounded recent delivery window
(`CURTAIL_HISTORY_LIMIT = 24`). A map is followed when later activity belonging
to the same work supplies one of these signals:

- A recall query echoes the cluster label or ask-hint under the shared tokenizer
  and `CURTAIL_QUERY_OVERLAP = 0.5`.
- A recall's recorded results contain the offered representative node.
- An id-fetch lookup names that node within the delivery's frozen outcome window.

Recalls without maps can supply the first two signals. Only map-bearing events
contribute offers and markers to the curtail cadence. Global node access
timestamps and globally aggregated lookup-consumed bits do not establish that
this consumer used a map.

Lookup evidence comes from existing lookup events. Its transport's latest
preceding recall identifies the work at the lookup time. An explicit label
therefore supports a resumed connection after that connection recalls under
the label. Looking up a node after switching to different work does not consume
the previous work's map. If timestamp ties leave different work identities
possible, the lookup is not attributed to one of them.

The lookup window uses the existing delivery ledger's boundaries, including the
24-hour `RECALL_DELIVERY_HISTORY_HORIZON_HOURS`. This attribution requires no
new persistent registry, counter table or migration. The shared corpus ledger
continues to serve its measurement and attribution consumers.

Unreadable history and unknown work do not justify curtailment. The memo can
delay a history read while the current builder's known offers remain below the
threshold, but a memo estimate cannot itself collapse a map. A confirmed
collapse is rechecked so new follow evidence can reopen the channel.

Without decay, offers eventually leave the 24-delivery history window and allow
a retry. With `LM_MAP_CURTAIL_DECAY` enabled, the skip allowance is
`min(2**(N-1), 8)` for N unaccepted offers. Pool-aware retries compare a digest
of query-qualified admitted members and their anchor assignments. An
unchanged pool extends the skip run; a changed pool can produce a full map.
Empty retries persist their digest too. Intermediate skips perform no pool
construction, query qualification or selection accounting.

The digest and markers live in the same persisted history, so cold continuation
retains retry state. Neither cadence changes the main recall or blocks access
to knowledge through an explicit recall or lookup.

## 7. Client context and feedback

`memory_recall` keeps optional ambient `task`, `task_pattern`, `agent` and
session fields. Both task fields are not required: either can name current
work, and ordinary MCP callers can rely on server-derived transport identity.
For continuation, an orchestrator can reuse one explicit label for the same
work across its connections. A per-turn `task` can accompany a stable
`task_pattern`; only the pattern determines map identity in that case.

`agent` remains descriptive attribution. `session_id` participates in ordinary
scope preferences and is distinct from `transport_session_id`; it is not a
substitute for the map's effective work identity. Choose search scope according
to the knowledge needed, not to isolate map history.

The public `used` and `irrelevant` feedback contract remains transport-based:
accepted marks refer to prior results delivered on that transport. Lookup credit,
feedback deduplication, corrections and supersedes behavior retain their own
contracts. Changing a map's explicit work label neither fabricates a previous
delivery nor authorizes feedback for another transport. `content_ref` and
`memory_lookup` continue to provide complete content.

## 8. Evidence and costs

The implementation retains bounded process-local structure and curtail memos,
existing recall/lookup events and existing delivery-window data. It adds no
background controller or client registration. Context-specific evidence reads
replace global consumption assumptions; bootstrap needs no map-history read.
The usefulness floor and corpus ranking are independent of current-work
curtailment.

Relevant evidence and tools:

| Artifact or tool | Purpose |
| --- | --- |
| `artifacts/recall-map/prereg.json` | Sealed consumption, organic-baseline and positional measurement protocol |
| `artifacts/recall-map/baseline.json` | Measurements tied to their recorded `as_of` and plan hash |
| `scripts/recall_map_effect.py` | Read-only aggregate measurement with mandatory `--as-of` and privacy checks |
| `artifacts/recall-map/latency-before-after.json` | Recorded latency measurements and their configuration |
| `scripts/recall_map_latency_bench.py` | Snapshot replay for map-stage latency |

The effect protocol separates query-anchor labels because a later matching
query may be something the caller would have asked without a map. Consumption
and within-session recall position are separate endpoints. Recorded benchmark
results apply to their snapshots and configurations; local correctness checks
do not establish fleet-wide consumption, token or model gains.

## 9. Source and checks

| Concern | Location |
| --- | --- |
| Residual and main recall | `src/living_memory/retrieval.py` |
| Selection, grouping, cache and curtailment | `src/living_memory/recall_map.py` |
| Tool delivery, transport context and static instructions | `src/living_memory/server.py` |
| Persisted events and delivery windows | `src/living_memory/storage.py` |
| Scope resolution | `src/living_memory/scope.py` |
| Explicit feedback and lookup credit | `src/living_memory/feedback.py` |
| Compact formatting | `src/living_memory/instructions_map.py` |

Targeted tests cover map delivery, current-work isolation, cold continuation,
curtail cadence, unknown context, mapless follow evidence and optional-map
failures. `test_instructions_refresh.py` checks that persisted maps and tool
calls cannot personalize a different client's bootstrap.
`test_instructions_imperative.py` pins protocol content and client budgets;
`test_instructions_map.py` tests the pure formatter independently.
