# Living Memory Architecture

Living Memory is a local MCP memory server backed by one SQLite file. It stores
raw interactions, consolidated concepts, and future schemas in the same table,
then uses level-specific indexes and retrieval policy weights to read the data
efficiently.

## Runtime Shape

The server is a Python package under `src/living_memory`.

| Layer | Modules | Responsibility |
| --- | --- | --- |
| MCP surface | `server.py`, `delivery.py` | Registers nine core tools, conditionally registers the independently controlled attestation tool, and filters only `tools/list`; also registers four resources and the retrieval-context prompt, stamps transport-derived correlation identity into recall/remember/teach, and shapes recall responses (session/twin dedup, snippets, context compaction) before serialization. |
| Storage | `storage.py`, `models.py` | Owns SQLite schema, uniform node CRUD, connections, FTS5, and retrieval weights. |
| Retrieval | `retrieval.py`, `scope.py`, `embeddings.py`, `feedback.py` | Resolves scope, searches FTS5, computes multilingual embeddings, traverses graph edges, reranks, logs access, stores recall events, and tunes weights. |
| Learning loop | `consolidation.py`, `decay.py`, `temporal.py` | Clusters similar traces into concepts, computes consensus and temporal hints, updates edge weights, records corrections, and soft-deletes expired or superseded records. |
| Offline credit | `attestation.py`, `postsession/` | Grades a finished session's recall events against evidence lifted from that session's own artifacts and applies the credit the live loop missed. See [post-session-attestation.md](post-session-attestation.md). |
| Read models | `resources.py`, `prompts.py` | Produces browsable resource payloads and formatted active memory context. |
| Configuration | `config.py` | Loads TOML settings and supplies defaults. |

### Tool discovery versus call reachability

The nine core handlers are always registered. FastMCP `tools/list` advertises
exactly `memory_recall`, `memory_remember`, `memory_teach`, and `memory_lookup`
by default. It filters the five maintenance handlers — `memory_consolidate`,
`memory_forget`, `memory_connect`, `memory_status`, and `memory_health` — out of
discovery without removing them from FastMCP lookup. They therefore remain
directly callable by name over the in-memory, stdio, HTTP, and HTTPS/TLS
transports. The filter is not an authorization boundary; transport
authentication still governs calls.

`LM_EXPOSE_OPERATOR_TOOLS=1` includes the five maintenance tools in
`tools/list`, producing the complete nine-core-tool inventory. The equivalent
Python constructor option is
`create_mcp_server(expose_operator_tools=True)`. An explicit constructor
`True` or `False` wins over the environment; only `None` (including omission)
consults it.

`memory_attest` is separate from those nine core handlers and is not registered
by default. `LM_EXPOSE_ATTEST=1` or `expose_attest=True` registers and
advertises it, while explicit `expose_attest=False` overrides the environment.
The attestation and operator controls are independent: either can be enabled
without changing the other surface.

## Data Flow

1. `memory_remember` appends a trace through `MemoryStore.append_trace`.
2. The trace is indexed by SQLite FTS5 immediately. Embeddings are computed lazily
   on first semantic recall using a configured sentence-transformers model only
   when it resolves to a local path or existing local cache. If no local model is
   available, recall uses a deterministic Unicode-aware hash fallback without
   attempting network access.
3. Every 100 active traces in a scope, `memory_remember` triggers consolidation.
4. `memory_recall` resolves the search scope, queries FTS5, scans local
   embeddings, optionally traverses graph edges, reranks by feedback and
   confidence, records access on returned nodes, and stores a `recall_events`
   row containing the query, resolved scopes, method scores, and result IDs.
   When the call arrives over an MCP transport, the server first stamps a
   transport-derived `transport_session_id` into the ambient context (an
   explicit caller value always wins), and the row keeps that identity in its
   own column.
5. The ranked results then pass through the pure delivery-shaping stage
   (`delivery.shape_recall_results`) on their way to the wire: a result
   byte-identical to a higher-ranked one becomes a `twin_duplicate` stub, a
   node id already delivered on the same transport session becomes a
   `session_duplicate` stub (the delivered-id set is read from the session's
   recent recall events *before* this call's row is written, so a response
   never stubs itself), and content over the snippet limit is truncated
   inline. Stubs and snippets keep the full node key set, compact oversized
   `context` values (strings truncate at a clean boundary; lists/objects —
   e.g. a procedural schema's `context.procedure`, which duplicates the whole
   content — collapse to `{"count", "chars"}`), and carry a `content_ref`
   pointing at `memory_lookup(node_id=...)`, which restores content and
   context complete. Shaping changes the serialized response only — the
   recorded event keeps the unshaped result IDs, and without a transport
   session id the session-dedup stage is skipped entirely (legacy full
   delivery).
6. The next compatible `memory_remember` consumes the pending recall event,
   records it in the new trace provenance, creates `related` edges to recalled
   nodes, and applies implicit positive feedback to the recalled results and
   retrieval weights. Compatibility follows identity precedence: explicit
   `session_id`/`task` matches and mismatches always decide first, equal
   transport session ids link strongly — including across the divergence
   between scope-less recalls (which plan `global`) and scope-less traces
   (which land on the configured default scope) — and the legacy
   text-similarity fallbacks apply only within the exact scope. Positive
   usefulness reinforcement carries diminishing returns: each increment is
   scaled by the node's remaining usefulness headroom (floored, never
   zeroed) and divided by a logarithm of its access count, so entrenched,
   frequently delivered nodes re-earn rank far more slowly than fresh ones
   and the delivery → reinforcement → delivery loop stops concentrating
   deliveries on a handful of saturated nodes. Explicit negative feedback
   (corrections) is exempt and always applies at full strength.
7. `memory_teach` appends a corrective trace and creates a `supersedes` edge
   from the correction to the original.
8. When independently enabled for the offline stage, `memory_attest` closes the
   same loop for events that step 6 never reached —
   ~80% of them, because recall is a session-opening ritual and
   `memory_remember` a session-closing one. An offline extractor submits the
   `recall_event_id` plus evidence lifted verbatim from that session's
   artifacts (diff hunks, command output — never agent prose, never a
   `memory_*` payload); the server loads *that event's own* result nodes from
   its own database and recomputes containment with the same 0.25 gate the
   live path uses, so a verdict asserted by the client is never read. Grounded
   results earn the same `feedback.apply_retrieval_feedback` call with the same
   `max(0.2, 1 / (rank + 1))` decay under the *event's* scope, plus anchor
   reinforcement over the grounded subset only. Idempotent per
   `(recall_event_id, evidence_sha256)`. Credit does not close the event unless
   a resolving `trace_id` is passed, so an attested event stays available to
   step 6.
   `attestation.py` is the **only** entry point for that offline credit, and it
   deliberately funnels through the running server rather than opening the
   database itself: `MemoryStore.__init__` (storage.py) migrates and writes
   whatever file it opens, and the live server holds that file, so an offline
   process opening `~/.local/share/living-memory/global.sqlite3` directly is a
   schema write behind the server's back — not a read. Every offline writer
   goes over MCP.
9. `memory_consolidate` clusters recent active traces, creates or updates
   concept nodes, computes consensus confidence and weekly temporal hints,
   refreshes related-edge weights, and applies decay. Concept content is a
   deterministic extractive digest — the strongest source's lead sentences
   plus the most distinctive sentence of each remaining source in
   cluster-centrality order, capped at 1200 chars — rather than a verbatim
   copy of one trace: for clusters holding two or more distinct contents the
   digest is byte-distinct from every source, and it is idempotent across
   passes (an unchanged cluster re-digests to the same bytes, and the
   duplicate-concept guard merges byte-identical twins instead of creating
   copies). It additionally groups
   traces by normalized `context.procedure_id` (or `context.task_pattern`) and
   materializes one `level='schema'` node per group of three or more
   procedural traces, storing the normalized trigger and ordered procedure
   steps in `context`.
10. `memory_status`, resources, and the retrieval-context prompt read the same
    store without requiring external services.

## SQLite Schema

The database is a single SQLite file. No external database, queue, hosted model,
or LLM is required.

### `nodes`

Uniform storage for traces, concepts, and schemas.

| Column | Purpose |
| --- | --- |
| `id` | ULID primary key. |
| `level` | `trace`, `concept`, or `schema`. |
| `content` | Natural-language memory content. |
| `embedding` | JSON vector, filled lazily. |
| `scope` | `global`, `project:<name>`, or `session:<id>`. |
| `agent`, `task`, `context`, `timestamp` | Interaction context. |
| `decayed`, `decay_reason` | Soft-delete state. |
| `access_count`, `last_accessed`, `usefulness_score`, `confidence`, `unique_agents`, `temporal_hint` | Ranking and learning statistics. |
| `source_traces`, `corrections`, `provenance` | Concept lineage and correction history. |
| `created_at`, `updated_at` | Store timestamps. |

Raw trace content is append-only. Consolidation creates concept nodes and
connections; it does not rewrite source traces.

Schema nodes (`level='schema'`) materialize repeating procedural patterns:
when three or more traces share a normalized `context.procedure_id` (or
`context.task_pattern`), consolidation emits one schema node whose `context`
carries the normalized `procedure_key`, normalized `trigger`, the original
opt-in field, and an ordered `procedure` list of step descriptions sourced
from the contributing traces. No new columns are introduced; all procedural
metadata lives in the existing JSON `context` column.

### `connections`

Adjacency table for graph recall.

| Column | Purpose |
| --- | --- |
| `source_id`, `target_id` | Connected node IDs. |
| `type` | `related`, `caused`, `contradicts`, `supersedes`, or `requires`. |
| `weight` | Traversal strength. |
| `metadata` | JSON metadata about how the edge was created. |

### `recall_events`

Durable provenance for retrieve interactions. Each row stores the recall query,
requested and resolved scopes, ambient agent/session context, result IDs,
component scores, and whether a later ingest trace consumed the event as
feedback. Consumed events point back to the ingest trace through
`feedback_trace_id`.

`transport_session_id` (schema v4) carries the transport-derived session
identity so feedback closure works for clients that send no explicit
`agent`/`task`/`session_id`. Databases created before v4 are upgraded by an
idempotent additive `ALTER TABLE ... ADD COLUMN` migration that preserves
every row and feedback flag; `scripts/verify_live_db_migration.py` proves this
against a backup-API copy of a live database. `memory_health` reports the
windowed closure ratio per identity class (`explicit`, `transport_only`,
`none`) under `feedback_closure`.

The same column also feeds recall delivery shaping: an idempotent
`idx_recall_events_transport_created` index supports
`MemoryStore.delivered_node_ids`, the bounded per-session query (node ids in
the 200 most recent events of one transport session) that session dedup
consults before delivering full content twice on the same connection.

`fingerprint` and `gated` (schema v5) are recall *accounting*, not a delivery
policy. `record_recall_event` unconditionally stamps every event with
`recall_fingerprint(query, requested_scope)` and rolls it into the
`recall_fingerprints` aggregate below; nothing about the response changes.
Pre-v5 databases are upgraded by the same idempotent additive
`ALTER TABLE ... ADD COLUMN` migration style, then `recall_fingerprints` is
rebuilt once from the stamped events.

### `recall_fingerprints`

Per-fingerprint delivered-versus-linked signal, maintained online by
`record_recall_event` and `mark_recall_event_feedback`: delivery and link
counts, deliveries since the last link, transport-session spread, and
first/last timestamps. It answers "how often was this exact request served
without ever earning feedback?" and is the substrate the repeat-recall gate
was built on.

That gate is a refuted hypothesis and is default-off — it failed both of its
thresholds on sealed data (see *Repeat-recall gating* in
`docs/mcp-interface.md`). This table nevertheless stays populated: it is
written by default-on accounting rather than by the gate, it is the measured
signal the refutation rests on, and `scripts/ap_baseline.py` reads it to
reproduce that result. Do not drop it as gate leftovers.

### `recall_attestations`

Idempotency ledger for the offline credit path (schema v8, strictly additive —
one new table touching no existing one). One row is one *graded submission* of
session-artifact evidence against one recall event: the digest, the item and
character counts, the containment the server recomputed per result, the grounded
node ids, the anchors reinforced, and whether the event was closed.

`UNIQUE(recall_event_id, evidence_sha256)` is the whole point. The row is
claimed *before* any credit is applied, so a crash mid-apply leaves the key
taken and the retry replays instead of double-crediting. See
[post-session-attestation.md](post-session-attestation.md) for what that
idempotency does and does not cover.

### `nodes_fts`

SQLite FTS5 virtual table that indexes node content for BM25 retrieval.

### `retrieval_weights`

Self-tuning policy table. Defaults are seeded for `default`, `global`,
`project`, and `session` scope families. Feedback cycles update the three
weights: `bm25`, `vector`, and `graph`.

## Indexing Strategy

Storage is uniform, but retrieval is stratified with partial indexes:

- active traces by timestamp;
- active traces by scope and timestamp;
- active concepts and schemas by content;
- active concepts and schemas by scope and content;
- active nodes by level and scope;
- graph edges by source/type and target/type.

SQLite's query planner chooses the relevant index for the requested level and
scope while the application keeps one CRUD path for all node levels.

## Automatic Phases

`PhaseManager` maps trace count to capability phase:

| Phase | Trace count | Capability |
| --- | ---: | --- |
| 0 | 0 | Empty local memory. |
| 1 | 1+ | FTS5 recall. |
| 2 | 100+ | Lazy embeddings and pattern detection. |
| 3 | 10,000+ | Concept graph activation. |
| 4 | 100,000+ | Hierarchical routing boundary. |
| 5 | 1,000,000+ | Federation-ready policy boundary. |

Each phase adds behavior while preserving the one-file SQLite store.

## Configuration

Defaults work without a config file. A TOML file can override storage,
the embedding model, phase thresholds, and retrieval weights.

The default embedding backend is offline-only. `embedding_model` can point to a
local sentence-transformers directory or a model already present in the local
cache. Missing models fall back to hash embeddings; model downloads require the
explicit environment override `LIVING_MEMORY_EMBEDDING_BACKEND=online`.

```toml
[storage]
db_path = "living_memory.sqlite3"
default_scope = "global"
trace_ttl_days = 180
embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"

[phases]
0 = 0
1 = 1
2 = 100
3 = 10000
4 = 100000
5 = 1000000

[retrieval_weights.project]
bm25 = 0.7
vector = 0.3
graph = 0.0
learning_rate = 0.05
```

## Local Commands

```sh
npm run setup:python
npm run check
npm test
python -m pytest tests/test_acceptance_contract.py
npm run server -- --help
npm run server -- ./living_memory.sqlite3
npm run server -- --config ./memory.toml --transport stdio
```

`npm run check` first installs the package test/runtime dependencies into
`.cache/python-deps`, then delegates to the Python test suite through
`scripts/check.sh` and `scripts/test.sh`. This check path includes a real
FastMCP server smoke test that verifies the documented runtime can instantiate
the four-tool default discovery surface and exercise all nine registered core
tools locally, including the five maintenance tools hidden from discovery.
