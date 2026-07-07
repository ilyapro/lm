# Living Memory MCP Interface

The MCP server exposes four core operations through nine tools, four
browsable resources, and one prompt. Create it in Python with
`living_memory.server:create_mcp_server`, run it from the repository with
`npm run server -- ./living_memory.sqlite3`, or run
`python -m living_memory.server` after installing the package dependencies.

## Transport and TLS

The same tool, resource, and prompt surface is reachable over every transport.
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
  scope-less recalls (which plan the `global` scope) and scope-less traces
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

Output includes the stored node and an `auto_consolidation` summary when the
scope reaches a consolidation boundary. If a compatible prior `memory_recall`
is pending, the new trace also records that recall under
`provenance.prior_recalls`, links to recalled nodes, and returns an
`implicit_feedback` summary. Requests arriving over an MCP transport are
stamped with the reserved `transport_session_id` context key (explicit values
win) so the pending-recall link works without any explicit identity — see
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
  "max_results": 10,
  "ambient_context": {
    "session_id": "s1",
    "project": "alpha",
    "cwd": "/workspace/alpha"
  }
}
```

Retrieval searches `session -> project -> global` when applicable, combines
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

### `memory_lookup`

Core operation: retrieve (exact match).

Input:

```json
{
  "scope": "project:alpha",
  "level": "trace | concept | schema",
  "lesson_kind": "optional context.type filter",
  "procedure_id": "optional procedural pattern id",
  "task_pattern": "optional task pattern"
}
```

Returns active nodes whose context fields match exactly, without ranked
recall, and does not record a recall event.

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
hints, updates related-edge weights, and applies decay. Procedural traces
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
