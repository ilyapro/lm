# Living Memory MCP Interface

The MCP server exposes four core operations through seven tools, four
browsable resources, and one prompt. Create it in Python with
`living_memory.server:create_mcp_server`, run it from the repository with
`npm run server -- ./living_memory.sqlite3`, or run
`python -m living_memory.server` after installing the package dependencies.

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
  }
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
`implicit_feedback` summary.

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
  "depth": "causal",
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

Output includes a `recall_event_id` at the top level and on each returned
result so later provenance can refer to the exact retrieval interaction.

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
grouped by id and materialized as `level='schema'` nodes once three traces
share the id. The schema's `context` stores `procedure_id`, `trigger`
(normalized pattern), and `procedure` (ordered step descriptions).

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
END ACTIVE MEMORY CONTEXT
```

The prompt only includes active concepts allowed by the resolved scope and the
minimum confidence threshold. The `skills:` section appears only when the
task tokens cover a schema's `trigger`; when no procedural patterns match the
query, the section is omitted entirely.

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
