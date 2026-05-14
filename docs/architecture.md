# Living Memory Architecture

Living Memory is a local MCP memory server backed by one SQLite file. It stores
raw interactions, consolidated concepts, and future schemas in the same table,
then uses level-specific indexes and retrieval policy weights to read the data
efficiently.

## Runtime Shape

The server is a Python package under `src/living_memory`.

| Layer | Modules | Responsibility |
| --- | --- | --- |
| MCP surface | `server.py` | Registers the seven tools, four resources, and retrieval-context prompt. |
| Storage | `storage.py`, `models.py` | Owns SQLite schema, uniform node CRUD, connections, FTS5, and retrieval weights. |
| Retrieval | `retrieval.py`, `scope.py`, `embeddings.py`, `feedback.py` | Resolves scope, searches FTS5, computes multilingual embeddings, traverses graph edges, reranks, logs access, stores recall events, and tunes weights. |
| Learning loop | `consolidation.py`, `decay.py`, `temporal.py` | Clusters similar traces into concepts, computes consensus and temporal hints, updates edge weights, records corrections, and soft-deletes expired or superseded records. |
| Read models | `resources.py`, `prompts.py` | Produces browsable resource payloads and formatted active memory context. |
| Configuration | `config.py` | Loads TOML settings and supplies defaults. |

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
5. The next compatible `memory_remember` consumes the pending recall event,
   records it in the new trace provenance, creates `related` edges to recalled
   nodes, and applies implicit positive feedback to the recalled results and
   retrieval weights.
6. `memory_teach` appends a corrective trace and creates a `supersedes` edge
   from the correction to the original.
7. `memory_consolidate` clusters recent active traces, creates or updates
   concept nodes, computes consensus confidence and weekly temporal hints,
   refreshes related-edge weights, and applies decay.
8. `memory_status`, resources, and the retrieval-context prompt read the same
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
the server surface and exercise the seven registered tools locally.
