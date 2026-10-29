# Explicit recall feedback: `used` / `irrelevant`

Explicit marks state which results helped with their recall query and which
were off-topic. They supplement grounded consumption and lookup credit without
asserting a relation between the shown knowledge and a later written trace.

## Tool fields

`memory_recall`, `memory_remember` and `memory_teach` each take two optional
fields:

| field | type | meaning |
|---|---|---|
| `used` | `list[str] \| None` | node ids from this caller's earlier recall that the agent used |
| `irrelevant` | `list[str] \| None` | node ids from an earlier recall that did not fit *that recall's query* |

Both default to `None`. When both are omitted, the response has no
`feedback_marks` key and no mark row is written.

### Why these three tools and not `memory_lookup`

- `memory_recall`: the next recall is the one call an agent almost always
  makes after reading a result. It also carries marks for recalls that are
  never closed by a write.
- `memory_remember` / `memory_teach`: the closing writes. A mark carried here
  is processed **before** grounded feedback of the same call, so explicit
  usage can claim numeric credit before inferred consumption.
- `memory_lookup` does **not** get the fields. A lookup of a delivered id
  already earns numeric lookup credit through `apply_lookup_credit`; it does
  not confirm relevance for that question. The agent can mark a previously
  fetched node `used` or `irrelevant` on a later recall or write. Keeping
  lookup's schema unchanged keeps the re-fetch path small.

### Response

When either field was passed, the response carries a compact summary:

```json
"feedback_marks": {"accepted": 2, "dropped": 1, "by_reason": {"not_delivered": 1}}
```

Under `LM_EXPLICIT_FEEDBACK_POLICY=off` the summary is
`{"accepted": 0, "dropped": 0, "by_reason": {}, "ignored": true}`.

## Resolution

Implemented in `feedback.resolve_explicit_marks`. A marked id is **accepted**
only if an eligible `recall_events` row delivered it. The current transport's
deliveries remain eligible regardless of context or scope. For a delivery on
another transport, both calls must declare the same nonblank `session_id`
(`session` is also supported). Both-present `task` or `agent` conflicts reject
that match. The current scope must equal the event's `scope` or
`requested_scope`, unless the saved read was broad (its scope plan contains
`*`). Breadth removes only this scope comparison: identity, delivered-id
membership and the bounded window still apply. Named `resolved_scopes` entries
alone never authorize attribution. Agent names, query text and a shared task
alone never establish explicit delivery attribution across transports.

To preserve feedback over reconnect, choose a session unique to the logical
caller and retain it in `memory_recall.ambient_context` and in
`memory_remember.context` / `memory_teach.context`. Independent callers need
distinct sessions even when working on the same task. Read broadly by omitting
both top-level `scope` and `ambient_context.scope`; ambient project is only a
preference. The saved broad read's preferred scope, often `session:<id>`, does
not restrict the new knowledge's destination. For example, recall with
`ambient_context={"session_id":"caller-run", "project":"demo"}`, then remember
or teach after reconnect with
`context={"session_id":"caller-run", "scope":"project:demo"}` and the marks.
No auxiliary recall is needed.

An explicit read filter still restricts retrieval; top-level `scope` takes
precedence over `ambient_context.scope`. For such a saved read, the current
call's scope resolution must match its `scope` or `requested_scope`, even if
its search included other scopes as fallbacks. Teach uses the current call's
context, never the corrected source's identity. Implicit pending-recall closure
keeps its separate matching contract. No new tool argument or caller registry
is required.

Without that durable identity, a previous transport's delivery is not
attributable. Supplying identity only after reconnect cannot recover it. Do not
copy an old `transport_session_id`, replay writes/marks or widen the window to
compensate. A call without transport still receives `no_transport`.

The event is the newest delivering event that is no later than the mark and
within `LM_LOOKUP_CREDIT_WINDOW_SECONDS` (default 24 h), among the newest 200
eligible caller events. Storage's `recall_deliveries` supplies one selection
to explicit marks, lookup credit and missing-anchor preparation. Lookup omits
caller context and remains transport-only. Indexed transport and session
probes share one cap before id/time filtering; events sort by
`created_at DESC, rowid DESC`, and time eligibility uses parsed UTC seconds
with inclusive window boundaries. Duplicate results use their first position.
Closed events remain eligible; the credit ledger decides once-only credit.

Each preparation, guarded revalidation and application reads current facts.
Only matching result JSON and event/query/scope/claim facts reach preparation;
application loads full events only for selected deliveries it needs. The selector
checks each candidate's time before reading its membership and stops once all
requested ids have a proven delivery. Older unnecessary result bodies do not
affect that selection. Malformed JSON in a still-needed eligible event and
required storage errors propagate; non-array bodies and non-object elements
provide no delivery evidence. A missing eligible result remains `not_delivered`.
A `memory_recall`
resolves its marks *before* it records its own event, so a mark can never name
the delivery that carries it.

A dropped mark gets one reason, checked in this order:

| reason | when |
|---|---|
| `empty` | not a non-blank string |
| `duplicate` | repeated within the same list (the first occurrence stands) |
| `conflict` | named in both `used` and `irrelevant` (every occurrence is dropped) |
| `no_transport` | the call has no transport identity, so there is no session to join |
| `not_delivered` | no eligible caller event in the window delivered it (missing/conflicting identity, foreign scope, unknown id, too old) |

## Audit table `recall_feedback_marks`

Every mark is written, accepted or dropped, under the `audit` and `credit`
policies. Additive: `CREATE TABLE IF NOT EXISTS` on every open, no foreign
keys, no `SCHEMA_VERSION` bump.

| column | notes |
|---|---|
| `id` | INTEGER PRIMARY KEY |
| `recall_event_id` | the delivering event; `''` when dropped |
| `node_id` | as passed (stripped); `''` for an `empty` mark |
| `mark` | `CHECK (mark IN ('used','irrelevant'))` |
| `accepted` | 0/1 |
| `reject_reason` | NULL when accepted |
| `via_tool` | `memory_recall` / `memory_remember` / `memory_teach` |
| `source_id` | the trace id for remember/teach, the *new* recall event id for recall |
| `transport_session_id` | NULL without transport |
| `agent` | `context.agent` / `ambient_context.agent` |
| `rank` | 0-based rank in the delivering event; NULL when dropped |
| `marked_at` | storage clock |

Index `idx_recall_feedback_marks_event(recall_event_id, mark, accepted)`
supports inspection of accepted and dropped marks for one recall event.

`memory_recall` resolves marks before its own event exists, then writes them
after committing the required formed response, with that new event as
`source_id`. Required passage/correction reads or shaping failures leave no
current consumption event or marks. Earlier eligible events remain available;
optional map or delivery-receipt failure does not remove their eligibility.
`memory_teach` writes marks before its corrective trace and fills `source_id`
with `MemoryStore.set_feedback_marks_source` in the same required transaction.
A teach that fails on its arguments (unknown `trace_id`, empty correction)
records nothing.

## Explicit credit basis: side table, not a rebuild

`recall_credit_ledger` has `CHECK (basis IN ('grounded','lookup'))`, and
SQLite cannot alter a CHECK (01M3H8BR0GHPTMTPYECRB0WAT8). **Choice: a side
table** `recall_explicit_credit(recall_event_id, node_id, source_id,
credited_at, UNIQUE(recall_event_id, node_id))`.

- A rebuild (copy/drop/rename) would rewrite the live ledger on the first
  open after deploy. That is not additive, a code rollback cannot undo it,
  and it changes a DDL string that the byte-identity tests pin.
- The side table keeps every existing DDL string byte-identical, and a
  restart creates it in the same way it created the ledger.
- Dedup across tables: `MemoryStore.claim_recall_credit` inserts into one
  table with a single `INSERT OR IGNORE … SELECT … WHERE NOT EXISTS (SELECT 1
  FROM <other table> …)`. The existence check and the claim are one
  statement, and so one transaction. A pair is claimed at most once across
  `grounded | lookup | explicit`, whichever arrives first.
  `credited_node_ids` returns the union of both tables.
- Consequence for readers: a query of `recall_credit_ledger` alone sees
  grounded and lookup credit only, the same as before this change. Explicit
  credit has to be read from `recall_explicit_credit`.

Explicit credit, applied only under `credit`: `source_id` is
`mark:<recall_feedback_marks.id>`. It uses the grounded assignment:
`apply_retrieval_feedback` with the event's recorded per-channel scores,
`useful=True`, `signal = max(0.2, 1/(rank+1)) × LM_EXPLICIT_CREDIT_WEIGHT`,
`scope = event.scope`, and then one anchor per event with edges to the
marked nodes only, the same as lookup credit. With weight 0 the pair is
claimed and nothing moves.
An accepted `used` mark also cancels query-relative irrelevance for that
question under `credit`, even if a prior lookup claimed the pair's numeric
credit. Repeated marks remain in the audit table but add no numeric credit or
repeat cancellation of an already cleared row.

## Valves

| env | values | default | unknown value |
|---|---|---|---|
| `LM_EXPLICIT_FEEDBACK_POLICY` | `audit`, `credit`, `off` | `audit` | default |
| `LM_EXPLICIT_CREDIT_WEIGHT` | float ≥ 0 | `1.0` | default (also NaN/negative) |
| `LM_EXPLICIT_FEEDBACK_PROMPT` | `optional`, `mandatory` | `optional` | default |

- **audit**: records and audits marks. There is no
  reinforcement, **no ledger or side-table row is claimed**, and grounded
  and lookup credit behave exactly as without marks.
- **credit**: audit, plus explicit credit for accepted `used` marks, plus
  query-relative demotion for accepted `irrelevant` marks. Each event's query
  preparation is shared across its positive and negative effects.
- **off**: the fields are accepted and ignored. Nothing is recorded.

## Irrelevance

`feedback._apply_query_feedback` passes the prepared anchor to
`irrelevance.record_query_irrelevance`, which records a query-anchor-relative
demotion (see docs/query-irrelevance.md). It is called only under `credit`, with
that event's accepted irrelevant ids, deduplicated and in mark order. An
irrelevant mark must never lower a node's global `usefulness_score`. It
means "not for this query", not "wrong". "Wrong" is `memory_teach`.

## Delivery history and knowledge relations

Compatible closing writes preserve all shown results in `recalled_nodes`,
`source_traces` and `prior_recalls`, independently of credit and explicit marks.
`memory_lookup(view="full")` exposes that provenance. Recall events retain the
original query, result ids, ranking scores and attribution.

Neither delivery, `used` for that query, nor a lookup asserts a `related` edge
from the closing trace. `memory_teach` records the same history without applying
grounded reinforcement; its explicit `supersedes` and validity remain separate.
Grounded query anchors still connect each query to its supported results.

Semantic graph readers exclude stored `related` rows whose current
`basis=implicit_recall_feedback` represents only recall history. This applies
before adjacency limits and to graph-evidence checks in live retrieval and
replay. Raw rows remain inspectable with `MemoryStore.get_connection` or SQL;
no corpus rewrite is required. A content-reference or derivation `kind` merged
into such a row establishes its own relation, and a replacement basis is read
as its current evidence. Fresh semantic contributions replace purely technical
metadata and weight instead of inheriting them. Explicit `related`, `caused`,
`requires` and correction relations retain their semantics.

## Prompt arms (`LM_EXPLICIT_FEEDBACK_PROMPT`)

The arm is read once, at server start. Both arms share `_RECALL_DESCRIPTION`:
recall before acting and mid-work, use knowledge under current instructions and
task conditions, verify changeable claims when needed, and immediately teach
refutations. A schema's type grants no authority; complete recipes and
`content_ref`/`memory_lookup` access remain available.

- **optional** (default): feedback field descriptions request ids used and ids
  that did not fit their recall query.
- **mandatory**: `_RECALL_DESCRIPTION_MANDATORY` adds: "You MUST mark each
  recall's results on your next call: used ids as used, off-topic ids as
  irrelevant." Feedback fields on recall, remember and teach say "REQUIRED
  after a recall: …". Remember and teach descriptions are shared by both arms.

The client budget is 1024 characters per tool description. The shared recall
text is 872 characters; the mandatory arm is 974. Tests check both lengths and
the causal debugging hint, action triggers, knowledge boundary and lookup path.
The deployed-protocol checker selects the expected arm from the local service's
env file, or from `--prompt-arm` for an explicit target. Serving the other arm
is configuration drift. See `docs/deployment.md` for its invocation.

## Rollout notes

The two new tables and the index are created by the first open after a
restart. That is an additive migration, which makes it an operator decision.
No existing DDL string changes, and `schema_version` stays 8. Rollback is
the old code: the new tables are ignored. The AE contract assessment is in
`artifacts/explicit-feedback/core/ae-contract-check.md`.
