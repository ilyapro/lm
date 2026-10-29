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
2. The trace is indexed by SQLite FTS5 immediately. MCP writes prepare their own
   search chunks from immutable source text before the shared store lock and
   apply them to the exact stored content in the knowledge transaction. A
   derived-index failure returns an explicit deferred receipt; required knowledge
   and feedback failures still raise. Missing and stale indices use bounded
   recall repair. Search windows stay within the paragraph IDs of `source_text.passage_spans`, shared with
   derivation and delivery; long paragraphs retain bounded overlapping windows.
   Existing content-current chunks are not reindexed merely by opening the
   database. Embeddings use the configured sentence-transformers model only
   when it resolves locally; otherwise the deterministic Unicode-aware hash
   fallback avoids network access. See [write indexing](mcp-interface.md).
3. Every 100 active traces in a scope, `memory_remember` triggers consolidation.
4. `memory_recall` resolves the search scope, queries FTS5, scans local
   embeddings, optionally traverses graph edges, reranks by feedback and
   confidence, and reads the selected passage coordinates. Its short consumption
   transaction records access on returned nodes and stores a `recall_events`
   row containing the query, resolved scopes, method scores, and result IDs.
   Required correction reads and response shaping complete before this transaction
   commits. A failure rolls back access and event/history together, leaving no
   new event for later explicit or implicit feedback. Query/feedback encoding,
   index preparation and ranking precede the transaction. The direct Python API
   keeps independent `log_access`/`log_event` flags; disabling both opens no
   consumption transaction.
   When the call arrives over an MCP transport, the server first stamps a
   transport-derived `transport_session_id` into the ambient context (an
   explicit caller value always wins), and the row keeps that identity in its
   own column.
5. The ranked results then pass through the pure delivery-shaping stage
   (`delivery.shape_recall_results`) on their way to the wire: a result
   byte-identical to a higher-ranked one with the same scope and recorded source
   identity becomes a `twin_duplicate` stub, a
   current version whose complete content or offered excerpt was included in
   an earlier session response becomes a `session_duplicate` stub, and
   oversized ordinary content becomes a bounded excerpt selected at paragraph
   and conservative sentence-end hints within each evidence source. Common
   ambiguous boundary hints stay inside a candidate; this heuristic does not parse arbitrary
   abbreviations or preserve all context across sentences. All passages overlapped
   by a verified semantic source address are budgeted together before optional
   context. Complete adjacent paragraphs from the same source precede distant
   lexical context, which is added only as whole paragraphs within each source
   record. Missing adjacent context is explicitly marked incomplete;
   other omitted context is marked, and an oversized anchor contributes an
   explicitly incomplete source prefix. Arbitrary character offsets and soft line wraps do not
   create independent candidates. Binding schemas
   remain complete instead of being snippeted. The bounded event history is
   read before retrieval, so a response never stubs itself.
   The default `view="knowledge"` serializes identity, content, validity,
   corrections, delivery class, and `content_ref` directly in each result.
   `view="full"` includes the ranked `node` wrapper, all score fields, and
   compacted metadata for diagnostic consumers. Both views point to
   `memory_lookup(node_id=...)` for complete content and current validity;
   `context_key` retrieves one original context field and lookup's
   `view="full"` restores the complete stored node. After shaping, an optional
   atomic event update records each selected result's delivery class and, for
   full/snippet bodies, source/body hashes. Selection IDs and scores retain their access and
   feedback roles. Missing delivery evidence permits re-delivery, including
   for legacy events. Without a transport session id, session dedup is skipped.
   Optional map, delivery-receipt and passage-maintenance failures do not undo
   the required response or its event. Feedback eligibility does not require an
   optional delivery receipt. A formed response is not proof of successful
   network transport: an interruption after commit has an unknown delivery
   outcome and does not authorize automatic replay.
6. The next compatible `memory_remember` consumes the pending recall event,
   records the complete delivery history in the new trace provenance, and
   applies content-grounded credit to recalled results and retrieval weights.
   Delivery history and credit for an earlier query create no knowledge
   relation to the new trace. Compatibility follows identity precedence: explicit
   conflicting `session_id`/`task` values reject attribution; an equal
   session or task permits continuation across reconnects within the recall's
   scope. Otherwise an equal transport id and compatible agent are required,
   including across scope-less recall/ingest default-scope divergence.
   Similar text, an agent name, or missing identity alone supplies no causal
   link. A trace without a compatible recall is still saved, without recall
   provenance, credit or anchors. Corrections use the current call's identity,
   while retaining the corrected source's subject and step location. Positive
   usefulness reinforcement carries diminishing returns: each increment is
   scaled by the node's remaining usefulness headroom (floored, never
   zeroed) and divided by a logarithm of its access count, so entrenched,
   frequently delivered nodes re-earn rank far more slowly than fresh ones
   and the delivery → reinforcement → delivery loop stops concentrating
   deliveries on a handful of saturated nodes. Explicit negative feedback
   (corrections) is exempt and always applies at full strength. Reinforcement
   is gated on *usage*, not on delivery: a result is credited when the
   consuming trace grounds its content (IDF containment at
   `DEFAULT_MIN_CONTAINMENT`), or when the agent id-fetched it with
   `memory_lookup` on the same transport session within 24 h of the delivery
   (`feedback.apply_lookup_credit`, run at lookup time). Either signal claims
   the (event, node) row in `recall_credit_ledger` before crediting, so the
   pair is credited once whichever comes first; linkage and event closure stay
   with the closing trace. `LM_RECALL_CREDIT_POLICY` and
   `LM_LOOKUP_CREDIT_POLICY` switch the two gates (see Configuration).
7. `memory_teach` appends a corrective trace and creates a `supersedes` edge
   from the correction to the original.
8. When independently enabled for the offline stage, `memory_attest` closes the
   same loop for events that step 6 never reached —
   ~80% of them, because recall is a session-opening ritual and
   `memory_remember` a session-closing one. An offline extractor submits the
   `recall_event_id` plus evidence lifted verbatim from that session's
   artifacts (diff hunks, command output — never agent prose, never a
   `memory_*` payload); the server loads *that event's own* result nodes from
   its own database and recomputes containment with the same gate the live
   path uses (`RECALL_CREDIT_MIN_CONTAINMENT`, 0.22 since the September 2026
   recalibration in `artifacts/grounding/recalibration-2026-09.md`), so a
   verdict asserted by the client is never read. Grounded
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
   deterministic extractive digest of complete source sentences, capped at
   1200 characters. Token novelty ranks sentences; only exact sentence repeats
   are omitted as duplicates. A source that cannot fit remains addressable by
   its recorded id. Source membership and group identify a derived owner;
   similarity and equal digest text do not merge independent derivations.
   After project-source writes, promotion refresh selects owner identities
   through the same bounded global concept identity/provenance projection used
   by promotion summaries. The timestamp/id cap precedes materializer admission;
   only selected owners need full reads, with current ownership and source
   validity rechecked in the existing writer transaction.
   It additionally groups
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
| `created_at`, `updated_at` | Row creation and last write timestamps, exposed by resources. |

The storage writer owns edge update policy. `create_connection` replaces weight
and metadata by default. Consolidation, teach and typed derivation pass
`merge=True`: weight is the maximum of current and incoming values, clamped to
[0, 1] and rounded to six decimals; incoming metadata keys overwrite existing
keys shallowly, including nulls and whole nested objects. A new correction
contribution replaces known technical provenance and its weight on the same
pair, so obsolete duplicate markers cannot veto that explicit correction.
Identity, direction, type and creation time stay fixed. The read and write share a writer transaction.
An unchanged contribution preserves the row, including `updated_at`; that field
is not a contribution counter or a last-observed event. Replacement writes still
refresh it. Every genuine correction write, including an unchanged contribution, moves
newly attached query anchors inside the same transaction. Query-anchor edges
retain their separate additive weight/hits policy. Typed backfill inserts absent
edges and can replace technical related rows with source relations; existing
genuine relations retain their fields. Derivation and backfill treat all content
ULID mentions as references, independent of correction wording or context labels.
Only explicit correction writes establish supersession.

Causal relations require an explicit connection. Shared `task` or `session_id`,
nearby timestamps and problem/fix labels do not create `caused` edges.
The shared Python/SQL policy excludes historical `caused` rows marked `rule=R2a`
or `kind=failure_resolution` from ordinary and causal graph traversal, ranking,
source qualification and offline semantic inventories. A caller-supplied
`basis=explicit` survives merged inferred markers. `memory_connect` replaces
metadata by default, so a later explicit connection clears an inferred-only row;
a semantic merge contribution also replaces that row's metadata and weight.
An older unmarked explicit connection followed by inferred metadata may be
indistinguishable from an inferred-only row. Stored weight, timestamps and
work context cannot reconstruct the missing explicit basis. History is retained
without a corpus rewrite. Other relation types keep their existing semantics.

Exact copies use `related` edges with `kind=duplicate_content`, retaining their
rows and source identities separately from correction validity. Anchor and
source-lineage traversal follows such bookkeeping only after checking current
bytes, scope, level and recorded source identity. The shared `relations` policy
excludes known historical heuristic, duplicate and inferred text supersedes rows
(`rule=R1c` or `kind=content_correction`) from semantic adjacency, correction
ranking/lookup, decay and replay. A nonempty string in the relation metadata's
`correction` field preserves an explicit teach contribution even alongside
inferred markers. Unknown correction metadata remains valid. No startup corpus rewrite or inferred restoration of
old decay/usage/anchor state is performed.


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
`MemoryStore.recall_delivery_history`, which reads content receipts from the
200 most recent events of a transport session. The delivery shaper compares
these receipts against current source content and the proposed excerpt;
selected IDs alone do not prove delivery.

Existing schema-v5 `fingerprint`/`gated` columns and the `recall_fingerprints`
table are preserved as historical data. Opening a database, recording a recall
and applying feedback neither populate nor rebuild that experimental state.
Fresh databases do not create it. Query anchors and hubs independently use
`recall_fingerprint(query, scope)` for stable query identity. The refuted
repeat-gating result and its offline source address are documented in
[mcp-interface.md](mcp-interface.md#historical-repeat-recall-gating-result).

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

SQLite FTS5 virtual table that indexes node content for BM25 retrieval. Its
tokenizer is `unicode61` (Unicode case folding, Latin diacritics removed, no
stemming, `ё` and `е` distinct), so the index holds surface forms only and is
never rebuilt for a change in the Python tokenizer.

The query side is where normalization lives. `retrieval._expanded_query` sends
the raw query text plus `embeddings.tokenize(query)`: lowercase, `ё` folded to
`е`, camelCase and `_-/.` split, stop words dropped, a synonym table
(`база`/`базы`/`базе` → `database`, `migration` → `schema`), a light suffix
stemmer for ASCII tokens and a Snowball-style Russian stemmer for all-Cyrillic
tokens. Because a Russian stem such as `миграц` is not an index token, the
expansion also carries one FTS5 prefix term per Russian content word whose
stem has at least four letters (`миграц*`), which `storage._fts_query` emits
as `"миграц"*` while every other
term stays an exact quoted term. Identifiers, paths, hashes and mixed-script
tokens are never stemmed. `LM_TOKENIZE_CYRILLIC_STEM=off` disables the Russian
stemmer and the prefix terms together and restores the surface-form behaviour;
the setting is read once per process (`embeddings.reset_cyrillic_stem_cache`
re-reads it). The same `tokenize` feeds IDF containment in `grounding.py` and
schema-trigger overlap, so those see one token per Russian lemma too;
recall-map labels keep their own unstemmed term
extraction (`recall_map._terms`) because they are looked up in
`nodes_fts_vocab`.

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

## Retrieval and consolidation

Recall collects BM25 and available vector candidates, with graph traversal at
the requested depth. Its effective scope policy weights those channels; index
readiness and actual evidence determine the results. Status and stats report
stored counts, confidence, coverage, promotions and retrieval policy rather
than inferring capabilities from corpus size.

Consolidation uses the configured local encoder for the selected active traces,
reusing whole-content vectors cached by source content and encoder identity.
Missing vectors are encoded in batches of eight under the encoder lock;
encoding, encoder identity waits and clustering run outside the shared-store
guard. The existing merge floor, similarity thresholds, scope
and procedure grouping, current-source checks and atomic derivation writers
determine admission. `force` expands the trace selection to an explicitly
requested cluster floor when needed; it does not waive evidence requirements.

New cross-scope promotion requires similar, uncorrected, unsuperseded active
concepts from at least three distinct project scopes in the same procedure
group. The pass searches existing representative vectors only after it
materializes project concepts. It pages through existing concepts in batches
of 500 within the insertion boundary observed at preparation start. Each
candidate is scored against the pass's compatible seeds outside the guard;
only the best match per project scope and seed is retained. A full page does
not exclude later scopes. It does not encode other projects or repair their
indexes. Each writer rechecks current source validity, grouping and similarity
inside its transaction. Missing representatives cannot authorize promotion.
Existing global derivations still refresh from their recorded current sources
when there is insufficient evidence to create a new global concept.

Derived search chunks encode the exact output digest independently of its
clustering representative. The shared cluster and promotion writers accept
prepared content vectors. A missing or changed output returns detached inputs
through the caller's preparation boundary: its ordinary transaction rolls back,
the caller releases the store guard, prepares the content, and retries against
current owners and sources. Cluster admission is checked again, including
competing owners and group compatibility. Source corrections and explicit forget
retain source-first derivative refresh in the same transaction; native teach
also rolls back and reapplies its accompanying feedback together. A caller-owned
outer transaction receives an unresolved preparation requirement without
entering the encoder. Encoding and chunk-write failures remain required failures.
Unchanged content with current chunks avoids re-encoding. Procedural outputs and
refreshes without a representative retain their existing lazy invalidation.

Preparation reads existing project representatives once per pass and compares
each candidate with materialized concepts in the same group. Total work grows
with those inputs; guarded reads and retained per-scope matches are bounded
independently of unrelated trace counts. Trace encoding pays only for cache
misses. These operations use the existing consolidation scheduler;
they add no background cycle or whole-memory migration.

## Configuration

Defaults work without a config file. A TOML file can override storage,
the embedding model and retrieval weights.

The default embedding backend is offline-only. `embedding_model` can point to a
local sentence-transformers directory or a model already present in the local
cache. Missing models fall back to hash embeddings; model downloads require the
explicit environment override `LIVING_MEMORY_EMBEDDING_BACKEND=online`.

Lexical normalization has one switch of its own: `LM_TOKENIZE_CYRILLIC_STEM`
(`on` by default; `off`, `0`, `false`, `no` or `disabled` turn the Russian
stemmer and the BM25 prefix terms off, see [`nodes_fts`](#nodes_fts)). The
grounding gate has one too: `LM_GROUNDING_MIN_CONTAINMENT`, a decimal in
`(0, 1]` that replaces `grounding.DEFAULT_MIN_CONTAINMENT` (0.22, calibrated in
`artifacts/grounding/recalibration-2026-09.md`) for the process; it is read
once at import, so it must be in the server's environment before start, and a
value that is not a finite number in that range is logged and ignored. The
lookup usage signal has two: `LM_LOOKUP_CREDIT_POLICY` (`delivered` by
default — a same-transport `memory_lookup` id-fetch of a node a recall
delivered within the window is credited exactly as a grounded result, once per
(event, node) across both signals through `recall_credit_ledger`; `off` keeps
recording the lookup event and credits nothing; an unknown value falls back
to the default, like `LM_RECALL_CREDIT_POLICY`) and
`LM_LOOKUP_CREDIT_WINDOW_SECONDS` (how far back a lookup may reach for its
delivering event, 86400 by default; unparsable or non-positive falls back).
Measurements: `artifacts/grounding/lookup-credit.md` and the integrated
`artifacts/grounding/usage-signal-after.md`.

```toml
[storage]
db_path = "living_memory.sqlite3"
default_scope = "global"
trace_ttl_days = 180
embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"

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
