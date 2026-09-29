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
* Pending-recall matching uses identity precedence: explicit
  `session_id`/`task` matches and mismatches always decide first; equal
  transport ids link strongly — including across the divergence between
  scope-less recalls (whose requested scope is `global`) and scope-less traces
  (which land on the configured default scope); differing transport ids
  reject; with a stamp on only one side, the legacy text-similarity fallback
  applies, and only within the exact scope.
* Outside a request context (direct tool-function calls, embedding servers
  without FastMCP) nothing is stamped and behaviour degrades to the legacy
  rules. Explicit `agent`/`task`/`session_id` remain recommended: they link
  work across reconnects, beyond one transport session.

The effect is that an identity-less client gets its recall events closed by
its own subsequent remembers — and never by another connection's — with no
client-side changes. `memory_health.feedback_closure` (below) makes the
closure rate observable. End-to-end behaviour over real TLS streamable-HTTP,
stdio, and in-memory transports is pinned by
`tests/test_transport_feedback_closure_e2e.py`, and
`scripts/verify_live_db_migration.py` verifies the additive `recall_events`
schema migration against a copy of a live database.

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
    "linked_node_count": 3,
    "source_trace_count": 0
  },
  "implicit_feedback": {
    "recall_event_ids": ["recall event id"],
    "linked_node_ids": ["node id"],
    "feedback_applied": true
  },
  "auto_consolidation": null,
  "auto_decay": null
}
```

`node` carries identity and counters instead of the full
[node payload](#node-payload-contract); fetch the complete node — content,
context, provenance bodies — with `memory_lookup(node_id=...)`. When the
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
`provenance.prior_recalls`, links to recalled nodes, and `implicit_feedback`
lists the consumed event and node ids. Requests arriving over an MCP
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
  "ambient_context": {
    "session_id": "s1",
    "project": "alpha",
    "cwd": "/workspace/alpha"
  }
}
```

`max_results` defaults to 5: first deliveries carry full content, so the
tighter default keeps responses lean — pass a larger value when the task
needs more breadth.

Scope: an explicit `scope` (the argument or `ambient_context.scope`)
restricts the search to it — a `session:<id>` scope also sees the ambient
project — and `global`. Without one the whole store is searched, and the
ambient `session_id`, project (`project`/`workspace`/`cwd`) or the configured
default project only rank first; the plan records this as a trailing `*`
(`global > *`). Nothing is inferred from the query text. Retrieval combines
BM25, local vector, and graph scores, reranks by feedback/confidence/access,
logs access for returned nodes, and persists a recall event containing the
query, scope plan, result IDs, and component scores. Causal queries such as
"why did B happen" traverse `caused` and `requires` edges. Queries whose
tokens cover a `level='schema'` node's `context.trigger` apply an explicit
trigger-match boost so the schema outranks its connected source traces.
Decision recall (`depth: "decision"`) returns matching primary traces plus
their rejected alternatives by following only `contradicts` edges with
`metadata.kind == "rejected_alternative"`. Shallow and numeric graph depths
filter rejected-alternative traces from results.

Output includes a `recall_event_id` at the top level and on each returned
result so later provenance can refer to the exact retrieval interaction. The
persisted event also stores the transport-derived `transport_session_id`
stamped into `ambient_context` (explicit values win), which later
remembers/teaches from the same connection use to close the event as
feedback.

#### Delivery shaping

Ranked results are shaped before serialization so repeated and oversized
content is not delivered again and again. Every result carries a `delivery`
class:

* `full` — complete `content`.
* `snippet` — content longer than this bearer's ladder budget, truncated
  inline at a clean boundary with a trailing `…`.
* `session_duplicate` — the node's full content was already delivered on
  this transport session; `content` is a one-line preview (~160 chars).
* `twin_duplicate` — the content is byte-identical to a higher-ranked result
  in the same response (typically a legacy concept and its verbatim source
  trace): exactly one twin bears the content, the rest are stubs.
* `near_duplicate` — the content repeats the *meaning* of a higher-ranked
  result in the same response (cosine over mean-pooled chunk vectors above
  `LM_RECALL_NEAR_DUP_COSINE`, default 0.95): the higher-ranked result bears
  the content, this one is a stub whose `content_ref.duplicate_of` names the
  bearer. Byte equality finds almost none of these — on the live corpus 4
  active nodes were byte-identical to another while 550 had a neighbour above
  0.95 — so this is the class that keeps five recall slots holding five facts
  rather than one fact five ways. Two guards bound it: a candidate more than
  `LM_RECALL_NEAR_DUP_LENGTH_RATIO` (default 0.2 = 20%) longer than its bearer
  is never collapsed (that is "the same fact plus a new detail"), and a node
  with no chunk vectors is never collapsed at all. `LM_RECALL_NEAR_DUP_COSINE=0`
  restores byte-only dedup without a revert.

Content budgets come from the snippet ladder, indexed by content-bearer
position — stubs don't consume ladder slots. The default ladder
`full,1000,700,500,300,200` ships the **top-ranked bearer complete,
however long** (positions past the end reuse the last budget), so the best
match never loses content to the diet; lower-ranked bearers arrive
progressively trimmed and re-fetchable.

Beyond content, every delivery class is dieted on the wire:

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

Every removal is wire-only: `memory_lookup(node_id=...)` returns the stored
node byte-complete. Non-full results carry a `content_ref` with the
`node_id` to pass to that lookup (plus `full_content_chars`, and
`duplicate_of` on twin stubs; with `LM_DELIVERY_SPARSE=0` also the literal
`fetch` call string):

```json
{
  "node_id": "ulid",
  "full_content_chars": 5120,
  "duplicate_of": "bearer ulid — twin stubs only"
}
```

A `full` delivery whose provenance or context lost anything to the diet
carries a minimal `{"node_id": "ulid"}` hint marking that the lookup
returns strictly more than was delivered.

Session dedup is keyed by the transport-derived `transport_session_id` and
consults the node ids recorded in that session's recent recall events (a
bounded window of the 200 most recent) *before* the current call's event is
written, so a response never stubs itself. A new connection is a new
transport session and receives full content again. Without a transport
session id (direct in-process calls, servers embedded without a request
context) session dedup degrades to legacy behaviour — every recall delivers
full content every time; twin dedup and snippeting still apply, being
per-response and deterministic. The persisted recall event always records
the unshaped result ids: delivery shaping changes the wire response only and
never perturbs feedback closure.

Env knobs — every diet lever has its own rollback valve:

* `LM_DELIVERY_SNIPPET_LADDER` — per-bearer-position content budgets,
  comma-separated `full` (deliver complete content) or char counts (default
  `full,1000,700,500,300,200`; positions past the end reuse the last entry;
  `off`/`uniform` fall back to the uniform legacy budget below; malformed
  values fall back to the default ladder).
* `LM_DELIVERY_SNIPPET_CHARS` — uniform max content chars delivered inline
  (default 1200; `0` disables snippeting). Setting it explicitly while the
  ladder is unset selects the uniform legacy mode, so the pre-ladder
  contract survives unchanged.
* `LM_DELIVERY_SESSION_DEDUP` — session-dedup rollback valve (default on;
  `0`/`false`/`no`/`off` disable).
* `LM_DELIVERY_CONTEXT_VALUE_CHARS` — max chars a single `context` value may
  occupy on dieted results (default 160; `0` disables context compaction;
  the pre-ladder default was 240).
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

Setting `LM_DELIVERY_SNIPPET_LADDER=off LM_DELIVERY_FULL_NODE_DIET=0
LM_DELIVERY_PROVENANCE_VALUE_CHARS=0 LM_DELIVERY_STATS_COMPACTION=0
LM_DELIVERY_SPARSE=0 LM_DELIVERY_CONTEXT_VALUE_CHARS=240` restores the
pre-ladder delivery renderer byte-for-byte.

`tests/test_delivery_diet_e2e.py` pins the wire-level behaviour — dedup,
degradation, the top-bearer complete-content guarantee, ladder snippet
re-fetch, full-node context compaction, and closure invariance — over a
real MCP client; `tests/test_delivery_shaping.py` pins each lever and its
valve at the unit level.

#### Repeat-recall gating — a refuted hypothesis, kept only as an instrument

The separate repeat path gates a recall whose `(query, requested_scope)`
fingerprint keeps being delivered without earning feedback links. The
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

##### Why the mechanism is nevertheless kept

It stays in the tree as an instrument, not as a feature.
`scripts/ap_baseline.py` — the frozen calculator that produced the refutation
above, and whose SHA-256 is recorded in `final-report.json` — depends on all
five moving parts of the mechanism. It imports `FingerprintGatePolicy`,
`should_gate_fingerprint` and `recall_fingerprint` from
`living_memory.storage`, and reaches the remaining two through the store API,
as `store.get_recall_fingerprint_stats(...)` and
`store.mark_recall_event_gated(...)`. Deleting any of the five would delete
the instrument that reproduces its own refutation and leave the sealed report
unverifiable. Removing a negative result's evidence is strictly worse than
carrying a dormant code path.

The schema-v5 columns stay for the same reason and for backward
compatibility. `recall_events.fingerprint`, `recall_events.gated` and the
`recall_fingerprints` aggregate are written by recall accounting that runs
unconditionally inside `record_recall_event`, independent of both valves below.
Accounting is not gating: stamping a fingerprint costs one hash and changes no
response.

##### Both valves, strictly opt-in

* `LM_RECALL_REPEAT_GATING` enables history-based fingerprint gating.
* `LM_RECALL_REPEAT_DROP_TRAILING_STUBS` permits a gated response's trailing
  run of `session_duplicate`/`twin_duplicate`/`near_duplicate` stubs to be
  removed; setting it alone neither enables gating nor changes an ungated
  response.

`FingerprintGatePolicy` defaults both fields to `False`, so a clean
environment resolves both off and the repeat path never runs. For either
valve, only a whitespace-trimmed, case-insensitive `1`, `true`, `yes`, or `on`
enables it; unset, empty, known-false (`0`, `false`, `no`, `off`), and
malformed values are off. Consequently a fully repeated response can become
empty only when both valves are explicitly enabled and the fingerprint gate
fires. With gating alone the ranked response retains re-fetchable stubs; with
trailing-drop alone that valve removes nothing from a nonempty ranking. The
remaining `LM_RECALL_REPEAT_MIN_UNLINKED`, `LM_RECALL_REPEAT_MAX_LINK_RATE`,
`LM_RECALL_REPEAT_MIN_SESSIONS` and `LM_RECALL_REPEAT_PROBE_EVERY` knobs only
shift thresholds inside an already-enabled gate; they cannot enable it.

##### Never advertised in the protocol channel

This section is the only place the repeat path is documented, and that is
deliberate. It MUST NOT be mentioned in any protocol-bearing text — tool
descriptions, server instructions, or the `memory://prompt/retrieval_context`
prompt. Those channels are paid for by every connected client on every
request and are hard-capped by clients (1024 characters per function
description for OpenAI-compatible clients); an agent following the protocol
gains nothing from a refuted, default-off mechanism it will never enable, and
every character spent on it is taken from the protocol itself. Re-adding it
there is a regression, not a documentation improvement.

This refuted class-blind gate is distinct from the class-agnostic
`LM_DELIVERY_*` diet described above. The diet intentionally shapes every
caller class the same way, preserves direct access through `memory_lookup`,
passed its generalization gates, and remains enabled by default; its
documented rollback valves do not opt the repeat path in.

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
  "node_ids": ["optional", "node", "ids"]
}
```

Returns active nodes whose context fields match exactly, without ranked
recall, and does not record a recall event.

`node_id`/`node_ids` switch the tool to direct id fetch — the re-fetch path
that recall stub and snippet `content_ref`s advertise. Id fetches return
complete node dicts in request order under `results`, list unknown ids under
`missing`, ignore `level`, and return decayed nodes with their `decayed`
flag set. A `scope` passed alongside ids verifies instead of filtering:
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
deterministic extractive digest of the cluster — the strongest source's lead
plus the most distinctive sentence from each remaining source, capped at
1200 chars — not a verbatim copy of one trace: whenever the cluster holds
two or more distinct contents, the digest is byte-distinct from every source
trace. The direct `memory_consolidate` response reports created/updated
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

Output reports phase, counts, confidence summary, coverage, and active
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
`access`, `scope_hygiene`, `storage`, `latency`, and `instructions` blocks.
`feedback_closure` is the first-class view of the implicit feedback loop: the
windowed share of recall events that a later ingest consumed, partitioned by
the identity each event carried:

```json
{
  "window_hours": 168,
  "recall_events_in_window": 1268,
  "feedback_applied_in_window": 219,
  "closure_ratio": 0.173,
  "identity_coverage": {
    "explicit": {"events": 120, "closed": 95, "closure_ratio": 0.792},
    "transport_only": {"events": 900, "closed": 850, "closure_ratio": 0.944},
    "none": {"events": 248, "closed": 30, "closure_ratio": 0.121}
  }
}
```

`explicit` events carry `agent`+`task`+`session_id`, `transport_only` events
carry only the transport-derived stamp, and `none` events carry neither
(legacy rows and non-request calls). A rising `transport_only` share with a
high closure ratio is the observable effect of server-side identity
derivation; regressions show up as events sliding back into `none` or the
ratio dropping.

## Resources

### `memory://global/concepts`

Returns active global concept nodes.

### `memory://project/{name}/concepts`

Returns active concepts for `project:{name}`. The template rejects non-project
scopes.

### `memory://stats`

Returns system health: phase, node counts, confidence summary, known scopes,
and retrieval weights.

### `memory://recent`

Returns the latest active trace nodes across scopes.

## Prompt

### `memory://prompt/retrieval_context`

Builds an active memory context block for the current task.

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

The prompt only includes active concepts allowed by the resolved scope and the
minimum confidence threshold. The `skills:` section appears only when the
task tokens cover a schema's `trigger`; when no procedural patterns match the
query, the section is omitted entirely. The optional `decision_history`
section appears only when task recall finds primary traces with rejected
alternatives.

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
