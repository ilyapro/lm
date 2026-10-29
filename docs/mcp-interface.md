# Living Memory MCP Interface

The MCP server registers nine core tools, exposes a deliberately smaller
discovery surface, and also provides four browsable resources and one prompt.
Create it in Python with
`living_memory.server:create_mcp_server`, run it from the repository with
`npm run server -- ./living_memory.sqlite3`, or run
`python -m living_memory.server` after installing the package dependencies.

## Tool visibility contract

With no operator exposure setting, real FastMCP `tools/list` advertises exactly
these four agent tools:

* `memory_recall`
* `memory_remember`
* `memory_teach`
* `memory_lookup`

The served server instructions name the same four tools in their bootstrap
directive ("load the Living Memory tools NOW"), so a session whose client
defers tool schemas is never told to load a tool it cannot discover.

The other five core tools are maintenance tools:

* `memory_consolidate`
* `memory_forget`
* `memory_connect`
* `memory_status`
* `memory_health`

They are always registered and directly callable by name through real FastMCP,
including the in-memory, stdio, HTTP, and HTTPS/TLS transports. By default they
are merely omitted from `tools/list`; the discovery filter does not disable the
handlers or remove them from FastMCP lookup. This is not an authorization
boundary; configured transport authentication applies as usual.

Set the literal environment value `LM_EXPOSE_OPERATOR_TOOLS=1` to include all
five maintenance tools in discovery, for all nine core tools. Python embedders
can pass `expose_operator_tools=True` or `False` to `create_mcp_server`. The
constructor has precedence: an explicit boolean is authoritative even when the
environment says the opposite; `None` (the default) consults the environment.

The optional attestation tool is controlled independently. It is not registered
by default. `LM_EXPOSE_ATTEST=1` or `expose_attest=True` registers and advertises
`memory_attest`, while explicit `expose_attest=False` overrides the environment.
Thus operator-only exposure lists the nine core tools without attestation,
attestation-only exposure lists the four agent tools plus `memory_attest`, and
enabling both lists ten tools.

## Transport and TLS

The same configured tool, resource, and prompt surface is reachable over every
transport.
`python -m living_memory.server` defaults to stdio; `--transport http` (or
`--transport sse`) with `--host`/`--port` serves the `/mcp` endpoint plus the
admin routes over the network. By default those transports use plain `http://`.

Set both `--tls-cert`/`LM_TLS_CERT` (PEM certificate) and
`--tls-key`/`LM_TLS_KEY` (PEM private key) to serve the same endpoints over
`https://`; the CLI flags take precedence over the environment variables. The
pair is all-or-nothing — supply both to enable TLS, or neither to keep the
unchanged plain-HTTP default, and supplying only one stops startup. Bearer
authentication via `LM_AUTH_TOKEN` is unchanged and keeps gating the admin
routes over either scheme. See the [README](../README.md#http-transport-and-tls)
for a runnable example.

That bearer token can be rotated at runtime: `POST /admin/token`, authorized by
the current token, accepts `{"token": "<new>"}` and thereafter accepts only the
new token on both the admin routes and `/mcp`. The rotation is persisted to the
SQLite state file (so it survives a restart and outranks the `LM_AUTH_TOKEN`
seed on the next start) and the token value never appears in the response,
logs, or any tracked file.

## Correlation identity and feedback closure

LM uses FastMCP 3 with negotiated HTTP sessions. Keep recall, lookup, and
feedback in one MCP client session: attribution and duplicate delivery depend
on a stable transport session identity across requests.

Retrieval-policy learning depends on recall events being *closed* — consumed
as implicit feedback by a later `memory_remember`/`memory_teach`. Clients
rarely send the explicit `agent`/`task`/`session_id` identity that used to be
the only reliable closing signal, so the server derives one itself: every
`memory_recall`, `memory_remember`, and `memory_teach` request is stamped with
a reserved `transport_session_id` context key taken from the MCP transport
session (the streamable-HTTP session id; a stable per-connection UUID on
stdio, SSE, and in-memory transports).

* The stamp lands in `memory_recall.ambient_context`,
  `memory_remember.context`, and `memory_teach.context`; an explicit caller
  value for the key always wins verbatim.
* Tool schemas are unchanged and the key is deliberately *not* `session_id`:
  scope resolution derives a `session:<id>` scope from `session_id`, and the
  transport stamp never perturbs scope resolution.
* Pending-recall matching requires a shared identity. Conflicting explicit
  `session_id`/`task` values reject attribution. An equal session or task
  permits continuation after reconnect within the recall's scope, even when
  agent or transport changes. Otherwise transport ids must be equal and
  agent values compatible. Equal transport also spans the divergence between
  scope-less recalls (`global`) and the configured default ingest scope.
* Similar text and equal agent names never establish that a recall was shown
  to the current work. Missing or one-sided identity alone cannot close an
  event. Without a compatible event, remember still saves the new knowledge,
  without implicit recall provenance, credit or query anchors.
* Outside a request context (direct tool-function calls, embedding servers
  without FastMCP) nothing is stamped. Supply a stable `task` or `session_id`
  on recall and subsequent writes for attribution, including across reconnects.
  `memory_teach` uses only the current call's identity; it preserves the
  source's subject context, but does not inherit its author/session/task.

Explicit `used`/`irrelevant` marks have a stricter reconnect contract: retain a
caller-unique `session_id` in recall's `ambient_context` and writes' `context`,
with compatible task, agent and scope. A shared task alone does not identify
the recipient of an earlier delivery. Without a session saved on both sides,
old-transport marks are `not_delivered`; identity cannot be inferred after the
fact. Same-transport marks and transport-only lookup credit retain their
contracts. See [explicit feedback](explicit-feedback.md#resolution).

The effect is that an identity-less client gets its recall events closed by
its own subsequent remembers — and never by another connection's — with no
client-side changes. `memory_health.feedback_closure` (below) makes the
closure rate observable. End-to-end behaviour over real TLS streamable-HTTP,
stdio, and in-memory transports is pinned by
`tests/test_transport_feedback_closure_e2e.py`, and
`scripts/verify_live_db_migration.py` verifies the additive `recall_events`
schema migration against a copy of a live database.

## Optional decay outcome

`memory_remember`, `memory_recall` and `memory_consolidate` share the
[`auto_decay` contract](../README.md#optional-decay-maintenance): `null` means
skipped, `swept: true` with counts and the completion timestamp means success,
and `{"swept": null, "error": "..."}` reports nonfatal maintenance failure
with unknown completeness. Failure is also logged and may follow durable decay;
it does not replace the primary tool result. Only completed maintenance advances
the interval marker. The next ordinary call can retry, without a cooldown.

## Tools

### `memory_remember`

Core operation: ingest.

Input:

```json
{
  "content": "string",
  "context": {
    "scope": "global | project:<name> | session:<id>",
    "agent": "string",
    "task": "string",
    "timestamp": "ISO-8601",
    "procedure_id": "optional procedural pattern id",
    "step_order": 1,
    "step_description": "optional explicit step text"
  },
  "feedback": {
    "confidence": 0.5,
    "usefulness_score": 0.0,
    "unique_agents": 1
  },
  "alternatives_considered": [
    {
      "approach": "hand-write files under projects/<name>/state/goals/",
      "rejected_because": "misses canonical marker; goals_list() ignores"
    }
  ]
}
```

The `context.procedure_id` (or `context.task_pattern`) field is opt-in:
agents that want to teach a repeatable procedure tag related traces with the
same identifier. Once three or more traces share an id, `memory_consolidate`
materializes a `level='schema'` node carrying a normalized `trigger` and an
ordered `procedure` list of step descriptions.

Output is a compact, size-bounded write confirmation — the writer already
holds the content it just stored, so nothing bulky is echoed back:

```json
{
  "node": {
    "id": "ulid",
    "level": "trace",
    "scope": "project:alpha",
    "created_at": "ISO-8601",
    "prior_recall_count": 2,
    "source_trace_count": 0
  },
  "implicit_feedback": {
    "recall_event_ids": ["recall event id"],
    "feedback_applied": true
  },
  "auto_consolidation": null,
  "auto_decay": null
}
```

`node` carries identity and counters instead of the full
[node payload](#node-payload-contract); fetch complete content and validity
with `memory_lookup(node_id=...)`, one context field with `context_key`, or
the complete stored node with `view="full"`.

`memory_remember` prepares and stores current search chunks for its new trace
and rejected alternatives before returning. `memory_teach` does the same for
its corrective trace. This removes their dependence on unrelated lazy-index
backlog; ranking, scope, validity and delivery still determine recall results.
It does not guarantee semantic completeness. Encoding uses the existing model
and fallback policy and adds work to the write, including all windows of long
content. Normalized correction and rejected-alternative text, and unchanged
remember text, supply immutable preparation inputs before the shared store lock
and SQLite write transaction. Encoder loading, inference and fallback have a
separate guard shared by model instances; preparation releases it before store
apply. The writer accepts prepared chunks only for exactly matching stored text,
using the existing replacement savepoint. Current scope, validity and pending
feedback are resolved under the store lock, not captured in index preparation.
Other tools and SQLite writers that do not need the busy encoder can proceed
during preparation. Calls that themselves encode queries or feedback anchors
still share that encoder; query preparation and recall maintenance retain their
existing boundaries. New knowledge and its successful index become visible only
at the required commit.

Search windows stay within the paragraph IDs supplied by
`source_text.passage_spans`, shared with derivation and delivery. Fitting
paragraphs stay whole; long paragraphs retain bounded overlapping token windows
with conservative passage cuts. Adjacent paragraphs are never combined merely
to fill a vector window. This can increase the number of vectors and storage;
it is not a general speed or recall guarantee. Each window retains its global
source token and character coordinates. Delivery uses that source address,
not the window as an independent assertion. After the mandatory source, it
prefers complete adjacent paragraphs from that same source before distant
lexical matches. It tries both neighbours together, then smaller whole
paragraphs first if both cannot fit. If adjacent source
context cannot fit, the excerpt says `[Incomplete source context]`; if the
mandatory passage itself cannot fit, it says `[Incomplete passage]`. Other
omissions use `[Context omitted]`. Neither adjacency nor these textual boundaries
prove semantic completeness. Conditions can still be omitted; `content_ref`
and exact lookup retain access to the complete source.

The splitter applies when chunks are written or replaced. Opening a database
does not reindex existing chunks: freshness is based on the source content
fingerprint, not the splitter version. Existing addressed chunks with unchanged
content keep their boundaries and scores until ordinary content reindexing.
Missing or content-stale chunks use the existing bounded recall repair. The
bounded refresh of a delivered legacy snippet without source coordinates also
uses the current splitter; it does not refresh already addressed chunks just
because packing changed. Deployment requires neither a full drain nor a
synchronous corpus reindex.

Knowledge, correction relationships and required recall feedback commit
together. Each new trace's derived index has its own savepoint. An encoding or
index replacement failure leaves that trace durable with no partial chunk set
and adds `indexing: {"status": "deferred", "node_ids": [...]}` to the response.
This confirms the knowledge write, not a completed index; exact lookup and
lexical search remain available. Existing bounded recall maintenance can repair
missing chunks on a later call. It also continues repairing old and edited
nodes. Required source, validity, knowledge and feedback failures still raise;
this boundary does not recover from failures of the enclosing transaction.
Whitespace-only content has no search windows.

When the
scope reaches a consolidation boundary, the pass is *scheduled*, not run:
the write answers at write time and `auto_consolidation` reports
`{"status": "scheduled", "scope": ..., "trace_count": N, "coalesced": bool}`
(`coalesced` when the request joined a pass already queued or running for
that scope). `memory_teach` reports the same key for its corrective trace.
One background worker per server runs the passes, one scope at a time; a
pass takes the runtime lock one short step at a time (one encode batch, one
cluster merge), so recall, lookup and other clients' writes are served while
it runs. Due requests arriving during a scope's pass merge into a single
follow-up pass. A failed pass is retried after a back-off, and the set of
scopes whose pass has not completed is kept in kv
(`auto_consolidation_pending`), so a pass that failed or was cut short by a
restart is resumed on the next server start. The finished pass's summary —
id lists plus counters (`concepts_created`, `concepts_updated`,
`concepts_promoted`, `schemas_created`, `schemas_updated`, `decayed`,
`clusters_considered`, `traces_considered`) — is kept by the scheduler; only
a direct `memory_consolidate` call, which stays synchronous, returns the
full result with node dicts. If a compatible prior
`memory_recall` is pending, the new trace records that recall under
`provenance.prior_recalls` and all shown ids in `provenance.recalled_nodes`
(and trace ids in `source_traces`). `implicit_feedback` lists the consumed
event ids and whether numeric feedback was applied. This history is accessible
through `memory_lookup(view="full")`; it creates no `related` connections.
Requests arriving over an MCP
transport are stamped with the reserved `transport_session_id` context key
(explicit values win) so the pending-recall link works without any explicit
identity — see
[Correlation identity and feedback closure](#correlation-identity-and-feedback-closure).

When `alternatives_considered` is supplied, ingest also creates one ordinary
trace node per rejected alternative and connects each rejected trace to the
primary trace with a `contradicts` edge whose metadata includes
`kind: "rejected_alternative"` and the non-empty rejection `reason`. The
response adds `rejected_alternatives: ["node id", ...]`. Rejected alternatives
reuse the primary trace scope and task, set
`context.is_rejected_alternative: true`, and start with confidence no greater
than half of the primary trace confidence. Omitting the field preserves the
previous response shape.

### `memory_teach`

Core operation: ingest correction.

Input:

```json
{
  "trace_id": "node id",
  "correction": "corrected content",
  "confidence": 0.9,
  "context": { "agent": "teacher" }
}
```

The tool appends a corrective trace, creates a `supersedes` edge from the
corrective trace to the original, and records the correction in provenance. If
the correction follows a compatible recall, the corrective trace records that
recall in provenance without reinforcing the incorrect original fact.

`supersedes` denotes an explicit correction with a source, independently of
whether its replacement can currently be delivered. Ordinary remembered prose,
quotations, negations and context labels do not establish supersession. Content
ULID references use `related(kind=content_reference)`; correction writes use
`memory_teach` or an explicit `memory_connect` with type `supersedes`. Exact-copy bookkeeping uses
`related(kind=duplicate_content)` and does not mark a fact corrected. Relation
readers exclude known historical duplicate bookkeeping (`kind=duplicate_content`),
drain heuristics (`kind=drain_near_duplicate`), and the offline heuristic producer
(`by=lm_collapse_near_dups.py`, `reason=near-duplicate collapse`), and inferred
text corrections (`rule=R1c` or `kind=content_correction`) from correction
semantics. A nonempty string in the relation's `correction` field records the
explicit teach contribution and remains authoritative in mixed metadata.
Unknown supersedes metadata keeps its meaning. Historical rows and
stored lifecycle flags remain intact; this is not a corpus restoration pass.


### `memory_connect`

Core operation: ingest graph relation.

Input:

```json
{
  "id_a": "source node id",
  "id_b": "target node id",
  "relation_type": "related | caused | contradicts | supersedes | requires",
  "weight": 1.0,
  "metadata": {}
}
```

The returned connection is stored in the SQLite adjacency table.

### `memory_recall`

Core operation: retrieve.

Input:

```json
{
  "query": "natural language query",
  "scope": "project:alpha",
  "depth": "causal | decision",
  "max_results": 5,
  "view": "knowledge",
  "ambient_context": {
    "session_id": "s1",
    "project": "alpha",
    "cwd": "/workspace/alpha"
  }
}
```

`max_results` defaults to 5. Pass a larger value when the task needs more
breadth. `view` accepts `"knowledge"` (default) or `"full"`; it changes
serialization only, preserving retrieval, ranking, and feedback events.

Scope: an explicit `scope` (the argument or `ambient_context.scope`)
restricts the search to it — a `session:<id>` scope also sees the ambient
project — and `global`. Without one the whole store is searched, and the
ambient `session_id`, project (`project`/`workspace`/`cwd`) or the configured
default project only rank first; the plan records this as a trailing `*`
(`global > *`). Nothing is inferred from the query text. Retrieval combines
BM25, local vector, and graph scores, reranks by feedback/confidence/access,
logs access for returned nodes, and persists a recall event containing the
query, scope plan, result IDs, and component scores. Causal queries such as
"why did B happen" traverse semantic `caused` and `requires` edges. Work context,
chronology and problem/fix labels do not establish causality. Ordinary and causal
recall exclude historical `caused` rows with `rule=R2a` or
`kind=failure_resolution`, unless `basis=explicit` records an explicit contribution.
An explicit `memory_connect` replaces the row's metadata; without such a surviving
basis, an old mixed row cannot be distinguished from inferred-only history. Queries whose
tokens cover a saved `context.trigger` can discover an instruction; content
support and query coverage contribute to ranking. Main content and optional
map qualification share current source evidence. Access, net usefulness and
query anchors retain their discovery/ranking roles without qualifying unchanged
content. Low source confidence, corrections and confirmed query-relative/hub
restrictions remain effective on both channel and trigger qualification scales.
See [recall qualification](recall-score-gate.md).
Decision recall (`depth: "decision"`) returns matching primary traces plus
their rejected alternatives by following only `contradicts` edges with
`metadata.kind == "rejected_alternative"`. Shallow and numeric graph depths
filter rejected-alternative traces from results.

Output includes a top-level `recall_event_id` so later provenance can refer
to the exact retrieval interaction. The
persisted event also stores the transport-derived `transport_session_id`
stamped into `ambient_context` (explicit values win), which later
remembers/teaches from the same connection use to close the event as
feedback.

#### Result views

The default `view="knowledge"` returns each result directly:

```json
{
  "id": "ulid",
  "level": "trace",
  "content": "The current finding or a marked excerpt.",
  "scope": "project:alpha",
  "provenance": {"corrections": []},
  "decayed": false,
  "decay_reason": null,
  "delivery": "full",
  "content_ref": {"node_id": "ulid"}
}
```

`provenance` retains corrections and `supersedes` when present. Correction
identity and links survive; oversized correction text may be shortened, with
original details available through `memory_lookup(..., view="full")`.
`content_ref` is present on every knowledge result, including complete
content, so metadata remains addressable. Use `id` in the next call's `used`
or `irrelevant` feedback. Content excerpts and duplicate previews retain
`delivery`, `full_content_chars`, and any `duplicate_of` reference.

`view="full"` returns diagnostic ranked entries: `node`, `score`,
`bm25_score`, `vector_score`, `graph_score`, `trigger_score`, `scope_rank`,
`methods`, and delivery information. Zero-valued score keys stay present and
floats retain six-decimal precision under the default sparse renderer.
This view includes compacted context, statistics, and provenance; it does
not disable snippets or duplicate handling. For original metadata, request
exact-id `memory_lookup(..., context_key="field")` or `view="full"`.
Context-filter lookup continues to return complete matching nodes.

Both views preserve the response envelope, including `recall_event_id`,
`recall_map`, feedback summaries, decay summaries, warnings, and limitations
when emitted. Within `recall_map`, diagnostic selection accounting (`sel`,
including its exclusion samples) appears only in `view="full"`; the complete
map remains stored on the recall event. Every other map field, including
navigation, filtered labels and limitations, is identical between views.
A content delivery class of `full` means the content is
complete; it is independent of the selected result view.

#### Delivery shaping

Ranked results are shaped before serialization so repeated and oversized
content is not delivered again and again. Every result carries a `delivery`
class:

* `full` — complete `content`.
* `snippet` — selected passages from content longer than this bearer's ladder
  budget. A passage cut to fit ends with `…`.
* `session_duplicate` — this version's complete content or the exact excerpt
  now offered was included in an earlier response on this transport session;
  `content` is a one-line preview (~160 chars).
* `twin_duplicate` — byte-identical content with the same scope and recorded
  source identity as a higher-ranked result. One twin bears the content;
  the others carry a preview and `content_ref.duplicate_of`.

Similarity helps retrieve and rank candidates. It does not establish that two
claims are identical, release a result slot as a duplicate, or create a
correction. Different assertions can compete for the bounded top-k without
being labeled repeats. Exact duplicate selection uses the same content/source
identity as delivery; independent derivations and different sources remain
separate even when their text matches.

Content budgets come from the snippet ladder, indexed by content-bearer
position — stubs don't consume ladder slots. The default ladder
`2400,1000,700,500,300,200` bounds ordinary content at each position
(positions past the end reuse the last budget). A verified semantic source address
selects all overlapping conservative passages together; otherwise lexical relevance
selects a passage within its source record. The same budget decision retains that
anchor before adding complete enclosing paragraphs and immediate neighbours.
Further optional matches are included as whole paragraphs within their source
record, only when the group fits the remaining budget. Each selected run preserves
its source text and evidence label; evidence carriers also retain their heading.
Optional context cannot displace the
found passage. Arbitrary vector-window boundaries never become claim boundaries.

`[Context omitted]` marks unselected source passages. If the mandatory source cannot
fit, only its beginning is delivered, with `…` and `[Incomplete passage]`. A budget
too small for the heading, label and notice falls back to a prefix with `…`.
`content_ref` always permits byte-complete lookup. These textual boundaries are
heuristics: abbreviations and dependencies between sentences or paragraphs can
require more context. Use the full original when omitted context or a missing tail
could qualify the statement; snippets do not guarantee semantic completeness.
Recipe `level="schema"` content is delivered complete instead of snippeted.
This preserves its steps and conditions; the storage level grants no authority.

In the diagnostic `view="full"`, metadata in every delivery class is
compacted on the wire:

* `provenance.prior_recalls` is summarized to `{"count": n}`, and any other
  oversized provenance value compacts per key exactly like context values.
  `corrections` are exempt from wholesale collapse: each correction keeps
  its full key set with short values (who/when/ids) verbatim and only
  oversized texts truncated — the supersedes signal always survives
  delivery.
* Oversized `context` values are compacted per key: strings are truncated at
  a clean boundary with a trailing `…`, lists and objects collapse to
  `{"count": n, "chars": m}` (a procedural schema's `context.procedure`
  would otherwise re-ship the node's whole content alongside a one-line
  stub).
* `stats` drop bookkeeping and default-valued fields (`last_accessed`, null
  `temporal_hint`, default `confidence`/`unique_agents`, zero counts) and
  round `usefulness_score`.
* Sparse entries drop the per-result `recall_event_id` copy (the envelope
  carries it once), empty `path`, null `agent`/`task`/`decay_reason`,
  `decayed: false`, and `timestamp`/`updated_at` equal to `created_at`.
  Score fields always stay present — zeros included — with floats rounded
  to 6 decimals, so score consumers keep a predictable schema.

Every removal is wire-only: `memory_lookup(node_id=...)` returns full content
and validity; `memory_lookup(..., view="full")` returns the complete stored
node. Non-full content deliveries in either view carry a `content_ref` with the
`node_id` to pass to that lookup (plus `full_content_chars`, and
`duplicate_of` on twin stubs; diagnostic results with
`LM_DELIVERY_SPARSE=0` also include the literal `fetch` call string):

```json
{
  "node_id": "ulid",
  "full_content_chars": 5120,
  "duplicate_of": "bearer ulid — twin stubs"
}
```

Knowledge results always carry at least `{"node_id": "ulid"}`. Diagnostic
results carry that minimal reference when complete content was delivered
but context or provenance was compacted.

Session dedup is keyed by the transport-derived `transport_session_id` and
reads shaped-content evidence from the 200 most recent recall events before
retrieval. Each selected result retains its ranking/access/feedback history;
after constructing the final response, the handler adds a `delivery` record
with `kind` and, for full/snippet bodies, `content_sha256` (source text) and
`delivered_sha256` (response text). Stubs need only their kind; a dropped result
records `kind: "omitted"`. Both response views use the same evidence. Only `full` and `snippet` records prove content delivery:
withheld text, duplicate previews, and legacy events without these records
cannot suppress a later full result. Selection recognizes previously offered
versions from these same records; shaping also checks the offered excerpt.

Changed source content is eligible again, including in-place derived-node
updates. A different query or snippet budget can offer a different excerpt,
which is eligible unless this version's complete content was already included.
A schema recipe is always offered complete. Access timestamps and other
bookkeeping do not define a content version. Expired evidence permits delivery
again; no corpus rewrite or separate persistent cache is needed.

A new transport session has independent history. Without a transport session
id, session dedup is skipped; within-response dedup and snippeting still apply.
The experimental fingerprint repeat control reads the same content records
within its own bounded window and remains disabled by default.

Recording response content is an optional atomic event update after response
construction. Read or write failure logs a warning and permits re-delivery,
without cancelling the recall or its selection/access/feedback history. These
records describe formed responses, not transport acknowledgments: a failed
connection can lose a response after its record was saved. Exact-id lookup
remains the full-content access path; there is no network exactly-once promise.

Env knobs — every diet lever has its own rollback valve:

* `LM_DELIVERY_SNIPPET_LADDER` — per-bearer-position content budgets,
  comma-separated `full` (deliver complete content) or char counts (default
  `2400,1000,700,500,300,200`; positions past the end reuse the last entry;
  `off`/`uniform` fall back to the uniform legacy budget below; malformed
  values fall back to the default ladder).
* `LM_DELIVERY_SNIPPET_CHARS` — uniform max content chars delivered inline
  (default 1200; `0` disables snippeting). Setting it explicitly while the
  ladder is unset selects the uniform legacy mode, so the pre-ladder
  contract survives unchanged.
* `LM_DELIVERY_SESSION_DEDUP` — session-dedup rollback valve (default on;
  `0`/`false`/`no`/`off` disable).
* `LM_DELIVERY_CONTEXT_VALUE_CHARS` — max chars a single `context` value may
  occupy in diagnostic results (default 160; `0` disables context compaction).
* `LM_DELIVERY_FULL_NODE_DIET` — default on; `0` restores byte-untouched
  provenance/context on `full` deliveries (rollback valve).
* `LM_DELIVERY_PROVENANCE_VALUE_CHARS` — max chars a single provenance value
  may occupy on dieted results (default 160; `0` limits provenance shaping
  to the legacy `prior_recalls` summarization).
* `LM_DELIVERY_STATS_COMPACTION` — default on; `0` restores complete `stats`
  dicts (rollback valve).
* `LM_DELIVERY_SPARSE` — default on; `0` restores null/duplicate entry
  fields, the `content_ref.fetch` string, and full float precision
  (rollback valve).

These metadata settings govern `view="full"`; the knowledge view keeps its
focused fields independently. Storage and exact-id full lookup are unchanged.

`tests/test_delivery_diet_e2e.py` pins the wire-level behaviour — dedup,
degradation, complete schema recipes, ladder snippet re-fetch, diagnostic
context compaction, and closure invariance — over a
real MCP client; `tests/test_delivery_shaping.py` pins each lever and its
valve at the unit level.

#### Historical repeat-recall gating result

Ordinary recall, delivery and feedback do not maintain repeat-gating state.
The historical experiment gated a recall whose `(query, requested_scope)`
fingerprint kept being delivered without earning feedback links. The
hypothesis was that gating such repeats would compact repeated *automatic*
recall while leaving organic, agent-triggered recall intact. **That hypothesis
was measured on sealed data and refuted.** It is not deferred, not pending,
and not awaiting a better threshold — it failed, and the failure is recorded
here so nobody pays to rediscover it.

##### The two measured failures

The sealed one-shot evaluation, whose report was generated
`2026-08-14T03:05:29Z`, ran the gate explicitly enabled (`min_unlinked=5`,
`max_link_rate=0.2`, `min_sessions=2`, `probe_every=25`, trailing-stub removal
on) against the sealed replacement holdout corpus, and failed both of its
required gates:

| Gate | Required | Observed |
|---|---:|---:|
| Unseen-in-dev repeated-automatic character reduction | `>= 0.50` | **`0.040904`** — 16 repeated fingerprint families, 113 events, only 21 gated |
| Organic recall preservation | each delta within `-5%` … `+5%` | **payload `-33.039%`, content-access `-45.5378%`** |

Sealed source: `artifacts/animal-planet/evaluation/final-report.json` and its
markdown rendering `final-report.md`, which carry these numbers, their
thresholds, and the SHA-256 of every input. Evaluated commit
`ee042335dbcd763054819def24e7e487de5d8f11`; evaluator `scripts/ap_baseline.py`
SHA-256 `18a162f7923af10705acf67b4655f76808ebd111dfa80f23bb10c59c26fa534a`;
replacement holdout manifest SHA-256
`fd36c972b049c296acbd2537f9af8f1db8d7db725f188c5dd90d27ab9aa3ab83`. The
authorization was one-shot and is consumed: the sealed policy forbids tuning,
case-level diagnosis and any retry, so these numbers are the final word on
this design rather than a work-in-progress reading.

##### Why it is not a tuning problem

In-sample the mechanism looked excellent. On the dev and eval splits its own
thresholds were derived from, the identical configuration reduced repeated
automatic payload by `0.760254` (dev) and `0.769469` (eval) with organic deltas
of exactly `0%` (`evidence.preflight.native_auto_recall` in the sealed report). On
fingerprints unseen in dev, that reduction collapsed to `0.040904` — roughly
one nineteenth of the in-sample figure — while organic recall lost a third of
its payload and nearly half of its content access. Even measured split-wide
across all 18 repeated families (238 events) the holdout reduction was only
`0.212469`, still less than half the required floor. The apparent gain was
fitted to the split it was tuned on; it did not generalize.

The cause is structural rather than numeric. The fingerprint is class-blind:
`(query, requested_scope)` cannot tell an automatic or preprompt recall from
one the agent deliberately issued. A server-side gate therefore cannot compact
the first class without stripping the second — which is precisely the
asymmetry the two failing gates report. No threshold reachable from this
signal separates the classes, because the signal does not contain the
distinction.

##### Conclusion: this deduplication belongs on the calling side

The caller knows which class a recall belongs to, already holds its own
session history, and dedupes there today — demonstrably, and without touching
organic recall. That is where repeat suppression belongs. It is not a
server-side policy, and it will not become one by re-tuning this gate.

##### Offline source and evidence

The evaluated source is revision
`ee042335dbcd763054819def24e7e487de5d8f11` in the source repository retained at
`/home/sfx/p/lm`, retained by the local Git tag
`archive/server-repeat-gating-evaluation`. Its calculator has the SHA-256
recorded above. Read the
unchanged evidence at `artifacts/animal-planet/evaluation/final-report.json`
and `final-report.md`; the source revision also contains the experiment's
implementation and tests. No production compatibility layer is required.

For an isolated offline source inspection on the source host:

```sh
history_dir=$(mktemp -d)
git -C /home/sfx/p/lm archive ee042335dbcd763054819def24e7e487de5d8f11 \
  src scripts/ap_baseline.py pyproject.toml | tar -x -C "$history_dir"
sha256sum "$history_dir/scripts/ap_baseline.py"
python3 "$history_dir/scripts/ap_baseline.py" --help
```

This extracts source only and does not run the sealed holdout. Its one-shot
permission remains consumed; source availability grants no permission to
repeat or tune that evaluation. The current `scripts/ap_baseline.py` retains
baseline report/verification, ordinary delivery metrics and recorded-candidate
ranking comparisons. It does not evaluate repeat gating.

Existing `recall_events.fingerprint`/`gated` columns and `recall_fingerprints`
rows remain untouched when a database is opened. New events do not populate
them, and fresh databases do not create them. The independent
`recall_fingerprint()` query identity remains in use by query anchors and hubs.
Session content receipts, exact-content delivery shaping and explicit
mandatory/credit feedback operate independently of this historical experiment.

### `memory_attest`

Core operation: retrieve (retroactive grounded credit). The offline write path
for a finished session whose recall was never closed by an in-session
`memory_remember`. Full rationale, closure semantics and the field measurement
are in [docs/post-session-attestation.md](post-session-attestation.md).

Input:

```json
{
  "recall_event_id": "recall event id the server returned during the session",
  "evidence": [
    "@@ -1,4 +1,6 @@\n-old line\n+new line",
    "$ python3 -m pytest -q\n1312 passed"
  ],
  "context": { "agent": "extractor", "source_session_key": "claude:<uuid>" },
  "trace_id": "optional node id to close the event against"
}
```

**The client submits evidence; the server decides.** The server loads *that
event's own* result nodes from its own database and recomputes containment with
`grounding.ground_results(..., min_containment=RECALL_CREDIT_MIN_CONTAINMENT)` —
the same gate the live credit path uses (0.22 since the September 2026
recalibration, `artifacts/grounding/recalibration-2026-09.md`;
`LM_GROUNDING_MIN_CONTAINMENT` overrides it for a process). A `grounded`,
`containment` or
`useful` verdict asserted anywhere in the payload is never read. Grounding runs
**per evidence item**, and each node keeps its maximum across items.

Evidence must be a list of strings, each lifted verbatim from the session's own
artifacts (diff hunks, command output). Bounds, all enforced rather than
applied — an over-cap submission is rejected, never silently truncated:

| Cap | Value | Meaning |
| --- | ---: | --- |
| `EVIDENCE_MAX_ITEMS` | 32 | items per attestation |
| `EVIDENCE_MAX_ITEM_CHARS` | 600 | characters per canonical item |
| `EVIDENCE_MAX_TOTAL_CHARS` | 19200 | characters across all items |

Items are canonicalized (CRLF normalized, trailing whitespace stripped, leading
and trailing blank lines removed) and joined with `\x1e` to form
`evidence_sha256`. `(recall_event_id, evidence_sha256)` is the idempotency key.

Output:

```json
{
  "attestation_id": "ledger row id",
  "recall_event_id": "the event graded",
  "scope": "scope of the event, which is the scope credit lands under",
  "evidence_sha256": "digest over the canonical items",
  "evidence_items": 12,
  "evidence_chars": 6104,
  "min_containment": 0.22,
  "results": [
    { "node_id": "…", "rank": 0, "containment": 0.41,
      "grounded": true, "evidence_index": 3 }
  ],
  "grounded_node_ids": ["…"],
  "credited": true,
  "anchor_ids": ["query anchor reinforced for this event's question"],
  "closed": false,
  "closed_by_attestation": false,
  "feedback_trace_id": null,
  "replay": false
}
```

* `results` carries one entry per result the server could resolve, with the
  containment it recomputed and the `evidence_index` of the item that produced
  it. A result whose node has since decayed or been forgotten is skipped.
* `credited` means grounded results earned `feedback.apply_retrieval_feedback`
  under the event's scope with the live `max(0.2, 1 / (rank + 1))` rank decay.
  `anchor_ids` is written over the grounded subset only.
* `closed` / `closed_by_attestation` / `feedback_trace_id` describe event
  closure. Without `trace_id` credit is applied and the event stays open, which
  keeps it available to the live pending-consumption path on purpose. With a
  `trace_id` that resolves to a real node the event is closed exactly once;
  an unresolvable `trace_id` is an error, not a downgrade to credit-only.
* `replay: true` means this `(recall_event_id, evidence_sha256)` was already
  recorded: the stored verdict is returned verbatim and **nothing is applied**.
  Every other field then describes the recorded attestation, not this call.

`memory_attest` is deliberately not one of the four default-visible,
protocol-bearing tools `scripts/check_deployed_protocol.py` compares
byte-for-byte: it is an offline extractor's tool, not part of the in-session
`memory_recall`/`memory_remember`/`memory_teach`/`memory_lookup` surface every
connected client pays for on every request.

### `memory_lookup`

Core operation: retrieve (exact match).

Input:

```json
{
  "scope": "project:alpha",
  "level": "trace | concept | schema",
  "lesson_kind": "optional context.type filter",
  "procedure_id": "optional procedural pattern id",
  "task_pattern": "optional task pattern",
  "node_id": "optional node id",
  "node_ids": ["optional", "node", "ids"],
  "view": "knowledge | full (id fetch only; default knowledge)",
  "context_key": "optional single context key (id fetch only)"
}
```

Returns active nodes whose context fields match exactly, without ranked
recall, and does not record a recall event.

`node_id`/`node_ids` switch the tool to direct id fetch — the re-fetch path
that recall stub and snippet `content_ref`s advertise. By default, id fetches
return full, unchanged `content`, `id`, `level`, `scope`, `decayed`,
`decay_reason`, and `provenance.corrections` (with correction links and bounded
long text) in request order under `results`. They omit unrelated `context`,
stats, and provenance history. `context_key="procedure"` adds only that
original field under `context`; an absent key yields `{}`. `view="full"`
returns the complete stored node, and cannot be combined with `context_key`.
Invalid selectors return an `error` without recording a lookup event. Id
fetches list unknown ids under `missing`, ignore `level`, and include decayed
nodes. A `scope` passed alongside ids verifies instead of filtering:
mismatching nodes stay in `results` and are reported under
`scope_mismatches`. Combining ids with context filters is rejected with an
`error`.

An id fetch is also a usage signal. The request is recorded in
`recall_lookup_events` (every id asked for, resolved or not; no access counter
moves), and when a `memory_recall` on the same transport session delivered
one of those ids within `LM_LOOKUP_CREDIT_WINDOW_SECONDS` (24 h), that node is
credited as if the closing trace had grounded it — usefulness, the event
scope's retrieval weights, one query-anchor edge — with the delivered rank's
signal. `recall_credit_ledger` holds one row per (recall event, node), so a
lookup and a later grounded closure credit the pair once, whichever comes
first; ids the recall never delivered, lookups from another transport, and
lookups outside the window credit nothing. `LM_LOOKUP_CREDIT_POLICY=off`
keeps the record and skips the credit. The credit is derived from the lookup
and never fails it. Measured in `artifacts/grounding/lookup-credit.md`.

### `memory_consolidate`

Core operation: consolidate.

Input:

```json
{
  "scope": "project:alpha",
  "force": false
}
```

The tool clusters similar recent traces, creates or updates concept nodes,
records source trace IDs, computes consensus confidence and weekly temporal
hints, updates related-edge weights, and applies decay. Concept content is a
deterministic extractive digest of the cluster. It selects complete source
sentences within a 1200-character budget, using token novelty to rank sentences
and exact equality to skip repeats. Distinct conditions are never replaced by
novel-token fragments. A source that cannot fit remains available by its
recorded id; the budget does not guarantee that every claim appears in the
digest. The direct `memory_consolidate` response reports created/updated
nodes as full node dicts (the automatic pass a write schedules keeps only an
id-only summary). A pass encodes through the model only traces it has not
seen: whole-content trace vectors are cached in `consolidation_embeddings`
keyed by content fingerprint and encoder. Procedural traces
(those tagged with `context.procedure_id` or `context.task_pattern`) are
grouped by normalized trigger and materialized as `level='schema'` nodes once
three traces share the pattern. The schema's `context` stores `procedure_key`,
`trigger` (normalized pattern), the original opt-in field, and `procedure`
(ordered step descriptions).

### `memory_forget`

Core operation: consolidate decay.

Input:

```json
{
  "id": "node id",
  "reason": "operator requested"
}
```

The node is soft-deleted by setting `decayed` and `decay_reason`; the stored
record remains available to direct store reads.

### `memory_status`

Core operation: reflect.

Input:

```json
{
  "scope": "project:alpha"
}
```

Output reports counts, confidence summary, coverage, and active
retrieval policy.

### `memory_health`

Core operation: reflect (operational metrics).

Input:

```json
{
  "scope": "project:alpha",
  "window_hours": 168,
  "top_stale": 5
}
```

Output is a metrics report with `activity`, `counts`, `dedup`, `staleness`,
`retrieval_policy`, `retrieval_skew`, `feedback`, `feedback_closure`,
`access`, `storage`, `latency`, and `instructions` blocks.

`staleness.age_seconds_p50`, `age_seconds_p90`, and `top_stale` use the same
age for active traces in the selected scope: time since `last_accessed`, falling
back to `timestamp` when the access date is absent or empty. Missing or
unparseable reference dates are omitted; an invalid nonempty access date does
not fall back to creation time. Future reference dates have age zero.
`top_stale` lists at most the requested number of traces, oldest by that age
first; zero disables the list. This diagnostic does not record accesses or
change decay TTL or retrieval eligibility.

`retrieval_skew.floors_in_effect` lists the configured family floors, including
an empty mapping when none are configured. Each `scopes_at_risk` row separates
normalized `raw` weights, `applicable_floors` for that scope's current evidence,
and `effective_after_floor` from the same calculation used by feedback updates
and floor maintenance. Unavailable vector/graph minima are zero; the BM25
ceiling is 1 when no positive nonlexical minimum applies. The lexical BM25
minimum remains active. An unconfigured family has `applicable_floors: {}`.
`floor_violation` compares only applicable constraints; `flags` describe raw
weight skew against the reported thresholds regardless of policy or evidence.
Health observes these values without persisting weights or applying feedback.

`feedback_closure` is the first-class view of the implicit feedback loop: the
windowed share of recall events that a later ingest consumed, partitioned by
the identity each event carried:

```json
{
  "window_hours": 168,
  "recall_events_in_window": 1268,
  "feedback_applied_in_window": 975,
  "closure_ratio": 0.769,
  "identity_coverage": {
    "explicit": {"events": 120, "closed": 95, "closure_ratio": 0.792},
    "transport_only": {"events": 900, "closed": 850, "closure_ratio": 0.944},
    "none": {"events": 248, "closed": 30, "closure_ratio": 0.121}
  }
}
```

`explicit` events carry a nonblank persisted `session_id`; `agent` and `task`
are optional. The writer resolves the `session` alias and converts supplied
values to strings before this classification. A saved session is sufficient
identity for reconnect feedback, subject to the existing scope, compatible
context, delivery and time checks. The class does not assert that any particular
feedback request will match. `transport_only` events have no such session but
carry a nonempty transport-derived stamp; `none` events carry neither usable
identity. Agent or task alone does not make this diagnostic class `explicit`,
even though a task can match implicit feedback. The partition describes the
saved identity, independently of whether the event has been closed.

## Resources

### `memory://global/concepts`

Returns active global concept nodes.

### `memory://project/{name}/concepts`

Returns active concepts for `project:{name}`. The template rejects non-project
scopes.

### `memory://stats`

Returns current node counts, confidence summary, known scopes,
and retrieval weights.

### `memory://recent`

Returns the latest active trace nodes across scopes.

## Prompt

### `memory://prompt/retrieval_context`

Builds an active memory context block for the current task. The block begins
with the same knowledge and authority guidance as both recall-description arms
and server instructions: use relevant knowledge and applicable complete recipes
under current system/operator instructions and task conditions; schema level
grants no authority. Nonempty recall neither forbids nor replaces needed checks
against current sources. Refuted claims must immediately go through
`memory_teach` and its `supersedes` edge. See [Reading memory](../README.md#reading-memory).

Arguments:

```json
{
  "task": "deploy rollback migration",
  "scope": "project:alpha",
  "max_concepts": 5,
  "max_schemas": 3,
  "min_confidence": 0.5,
  "retrieval_policy": "balanced | confidence | project | recent",
  "agent": "agent name",
  "ambient_context": {}
}
```

Output format:

```text
BEGIN ACTIVE MEMORY CONTEXT
Use relevant knowledge and applicable complete recipes under current system/operator instructions and task conditions; level:schema grants no authority. Nonempty recall neither forbids nor replaces needed current-source checks. Teach refuted claims at once via memory_teach (supersedes).
scope_plan: project:alpha > global
task: deploy rollback migration
retrieval_policy: confidence bm25=0.70 vector=0.30 graph=0.00
min_confidence: 0.50
concepts:
1. id=<node> scope=project:alpha confidence=0.92 usefulness=0.60 score=...
   Alpha deploy rollback requires migration dry run before release
skills:
1. id=<schema> scope=project:alpha trigger="deploy rollback" confidence=0.74 score=...
   Procedure: deploy rollback
   1. check migration is reversible before release
   2. run migration dry run on staging
   3. execute rollback only after dry run passes
decision_history:
1. id=<node> scope=project:alpha confidence=0.50 score=...
   Used /api/goals/create after /api/switch
   Alternatives rejected: 2 (misses canonical marker; requires JIRA env)
END ACTIVE MEMORY CONTEXT
```

For a nonempty task, concepts, schemas and decision history retain the current
recall's qualification. A `below_threshold` result is an unqualified source
reference, never an assertion or a recipe: its entry contains `delivery`,
`content_ref={"node_id": "..."}` and validity, without body, trigger or decision
reasons. Use `memory_lookup(node_id=...)` to inspect that source; fetching it
does not establish relevance to the task. Qualified entries take precedence
over references within each section's limit. Policy, confidence, history and
rank do not independently admit sources absent from task recall.

An empty task intentionally browses scoped concepts using `retrieval_policy`.
Concepts and schemas respect the minimum confidence threshold and resolved
scope. The `skills:` section appears only when the
task tokens cover a schema's `trigger`; when no procedural patterns match the
query, the section is omitted entirely. The optional `decision_history`
section appears only when task recall finds primary traces with rejected
alternatives.

The direct Python selectors return `ContextConcept`, `ContextSchema` and
`DecisionHistory` entries with the same `withheld` field as `RecallResult`.
Their source nodes remain available for identity and validity; consumers must
respect `withheld` before using source content. Withheld decision entries have
no derived reasons. `format_retrieval_context` renders these entries as the
explicit references described above.

## Node Payload Contract

Tool and resource responses serialize nodes with this shape:

```json
{
  "id": "ulid",
  "level": "trace | concept | schema",
  "content": "string",
  "scope": "project:alpha",
  "agent": "agent-a",
  "task": "incident",
  "timestamp": "2026-05-14T00:00:00Z",
  "context": {},
  "stats": {
    "access_count": 0,
    "last_accessed": null,
    "usefulness_score": 0.0,
    "confidence": 0.5,
    "unique_agents": 1,
    "temporal_hint": null
  },
  "provenance": {
    "source_traces": [],
    "corrections": [],
    "prior_recalls": [],
    "recalled_nodes": []
  },
  "decayed": false,
  "decay_reason": null,
  "created_at": "ISO-8601",
  "updated_at": "ISO-8601"
}
```

Embeddings are intentionally omitted from MCP read payloads to keep context
small.
