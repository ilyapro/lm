# Explicit recall feedback: `used` / `irrelevant`

Goal `explicit-recall-feedback`, node `explicit-marks-core`. This document is
the interface contract that the sibling nodes build against
(irrelevance-query-demotion, marks-agreement-check, link-policy-replay,
optional-vs-mandatory-experiment, rollout-handoff).

## Why

Grounded credit is a rare lexical signal, and about a third of what it fires
on is noise (LM 01M3H76JBHEZKJMBPWFVN70SB2). 26% of recalls are never closed
at all (01M3H77WY4ZNR3XNA62Y2YEVMR). Implicit feedback also linked the closing
trace to *every* delivered node, irrelevant ones included
(01M3H7GVW9Y3DWRZXRZ6CE85NK). An explicit mark is the agent stating directly
which results it used and which were off-topic.

## Tool fields

`memory_recall`, `memory_remember` and `memory_teach` each take two optional
fields:

| field | type | meaning |
|---|---|---|
| `used` | `list[str] \| None` | node ids from an earlier recall on this transport session that the agent used |
| `irrelevant` | `list[str] \| None` | node ids from an earlier recall that did not fit *that recall's query* |

Both default to `None`. A call that omits both behaves exactly as it did
before: the response has no `feedback_marks` key and no row is written. No
tool was added.

### Why these three tools and not `memory_lookup`

- `memory_recall`: the next recall is the one call an agent almost always
  makes after reading a result. It is also the only carrier for the 26% of
  recalls that are never closed by a write.
- `memory_remember` / `memory_teach`: the closing writes. A mark carried here
  is processed **before** the implicit feedback of the same call, so an
  `irrelevant` mark keeps its edge out from the start.
- `memory_lookup` does **not** get the fields. A lookup of a delivered id is
  already a "used" signal (lookup credit, `apply_lookup_credit`), so a `used`
  field there would duplicate it. An `irrelevant` field there would be
  self-contradictory: the agent would be fetching the full node it calls
  irrelevant. Keeping lookup's schema unchanged also keeps the re-fetch path,
  which the recall description advertises, as small as possible.

### Response

When either field was passed, the response carries a compact summary:

```json
"feedback_marks": {"accepted": 2, "dropped": 1, "by_reason": {"not_delivered": 1}}
```

Under `LM_EXPLICIT_FEEDBACK_POLICY=off` the summary is
`{"accepted": 0, "dropped": 0, "by_reason": {}, "ignored": true}`.

## Resolution

Implemented in `feedback.resolve_explicit_marks`. A marked id is **accepted**
only if a recall event with the same `transport_session_id` delivered it. The
event is the newest delivering event that is no later than the mark and
within `LM_LOOKUP_CREDIT_WINDOW_SECONDS` (default 24 h), among the newest 200
events of the session. This is the join `apply_lookup_credit` does. A
`memory_recall` resolves its marks *before* it records its own event, so a
mark can never name the delivery that carries it.

A dropped mark gets one reason, checked in this order:

| reason | when |
|---|---|
| `empty` | not a non-blank string |
| `duplicate` | repeated within the same list (the first occurrence stands) |
| `conflict` | named in both `used` and `irrelevant` (every occurrence is dropped) |
| `no_transport` | the call has no transport identity, so there is no session to join |
| `not_delivered` | no same-session event in the window delivered it (foreign session, unknown id, too old) |

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
serves the per-event hygiene read that runs on every closing write.

`memory_teach` and `memory_recall` write their marks before their carrier
exists: the corrective trace in one case, the new recall event in the other.
They fill `source_id` in once the carrier has been written
(`MemoryStore.set_feedback_marks_source`). A teach that is about to fail on
its own arguments (unknown `trace_id`, empty correction) records nothing.

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

## Valves

| env | values | default | unknown value |
|---|---|---|---|
| `LM_EXPLICIT_FEEDBACK_POLICY` | `audit`, `credit`, `off` | `audit` | default |
| `LM_EXPLICIT_CREDIT_WEIGHT` | float ≥ 0 | `1.0` | default (also NaN/negative) |
| `LM_IMPLICIT_LINK_POLICY` | `all`, `credited` | `all` | default |
| `LM_EXPLICIT_FEEDBACK_PROMPT` | `optional`, `mandatory` | `optional` | default |

- **audit**: records and audits marks, and link hygiene applies. There is no
  reinforcement, **no ledger or side-table row is claimed**, and grounded
  and lookup credit behave exactly as without marks.
- **credit**: audit, plus explicit credit for accepted `used` marks, plus
  `_apply_query_irrelevance(store, event, node_ids)` once per event for
  accepted `irrelevant` marks.
- **off**: the fields are accepted and ignored. Nothing is recorded, and
  there is no hygiene: this is exactly the behaviour before the feature.

## Irrelevance

`feedback._apply_query_irrelevance(store, event, node_ids)` delegates to
`src/living_memory/irrelevance.py`, which records a query-anchor-relative
demotion (see docs/query-irrelevance.md). It is called only under `credit`, with
that event's accepted irrelevant ids, deduplicated and in mark order. An
irrelevant mark must never lower a node's global `usefulness_score`. It
means "not for this query", not "wrong". "Wrong" is `memory_teach`.

## Link hygiene

Link hygiene applies under every policy except `off`:

- `apply_pending_recall_feedback` creates no `related` edge from the closing
  trace to a node that has an accepted `irrelevant` mark for that event.
  Provenance (`recalled_nodes`, `source_traces`, `prior_recalls`) still
  records everything shown. The node's grounded credit, if its content was
  grounded, is unaffected.
- Marks on the same `memory_remember` / `memory_teach` call are processed
  before implicit feedback.
- If a later call marks a node irrelevant after the event has closed, the
  closing trace's `related` edge to the node is **deleted**. This happens
  only if the edge's metadata is `basis=implicit_recall_feedback` and
  `recall_event_id=<that event>`. The `(source, target, type)` key is unique,
  so an edge that other evidence has rewritten is left alone.

`LM_IMPLICIT_LINK_POLICY=credited` links only results credited for that event
under any basis: grounded in this call, or earlier lookup or explicit credit.
`all` is the default and is today's behaviour minus irrelevant marks.
Sibling link-policy-replay measures which one to recommend. The default is
unchanged. Note: `memory_teach` runs implicit feedback with
`reinforce_results=False`, so under `credited` a teach links only results
that were credited before it.

## Prompt arms (`LM_EXPLICIT_FEEDBACK_PROMPT`)

The arm is read once, at server start. The experiment switches arms only by
restarting its sandbox server with this env var.

- **optional** (default): every tool description is byte-identical to
  before. The hint is carried only by the field schemas: `used` = "Optional:
  ids of results from your earlier recalls this session that you used.",
  `irrelevant` = "Optional: ids of earlier recall results that did not fit
  their query."
- **mandatory**: `memory_recall` gets `_RECALL_DESCRIPTION_MANDATORY`, which
  ends in the binding sentence "You MUST mark each recall's results on your
  next call: used ids as used, off-topic ids as irrelevant." It was fitted by
  rewording, not appending:
  - the mid-work law no longer repeats the opener's "at every new turn of
    thought";
  - "searching," and "— measuring" were dropped from the world trigger,
    whose class "pulling knowledge from the world" and the world-before-memory
    clause still cover both;
  - "choosing a design or approach" became "choosing an approach";
  - "Query: identifiers and what you need; re-ask as it moves." became
    "Query by identifiers; re-ask.";
  - the content_ref sentence was shortened.

  "Depth 'causal' when debugging." stays in the mandatory arm too: it was
  dropped in the first fit and restored by the two world-trigger and design
  cuts above (tests/test_recall_description_causal.py).

  Every pinned law phrase stays (tests/test_explicit_feedback.py,
  tests/test_instructions_imperative.py). The field schemas on all three
  tools say "REQUIRED after a recall: …". The remember and teach
  descriptions are unchanged: `memory_remember` is at 1022/1024 and its
  every clause is pinned, and the recall sentence already binds "your next
  call".

Character cost (measured on the real FastMCP `list_tools`):

| arm | recall desc | remember desc | teach desc | field-schema delta per tool (JSON chars) | total extra vs. pre-feature |
|---|---|---|---|---|---|
| pre-feature | 1015 | 1022 | 614 | 0 | 0 |
| optional | 1015 (+0) | 1022 (+0) | 614 (+0) | +399 | +1197 (schema only) |
| mandatory | 1020 (+5) | 1022 (+0) | 614 (+0) | +419 | +1262 (+5 desc, +1257 schema) |

The mandatory binding sentence costs 101 description characters, funded by
the rewording. The field schemas are paid in both arms. The mandatory arm
diverges from the constants that `scripts/check_deployed_protocol.py`
compares, so that check reports drift on a host running it. That is expected
for a sandbox experiment and wrong for production.

## Rollout notes

The two new tables and the index are created by the first open after a
restart. That is an additive migration, which makes it an operator decision.
No existing DDL string changes, and `schema_version` stays 8. Rollback is
the old code: the new tables are ignored. The AE contract assessment is in
`artifacts/explicit-feedback/core/ae-contract-check.md`.
