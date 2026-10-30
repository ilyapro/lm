# Living Memory

Living Memory is a local MCP memory server that turns agent interactions into
append-only traces, consolidated concepts, and future schemas. It uses one
SQLite file for storage, FTS5 for lexical recall, multilingual embeddings for
semantic recall, graph edges for causal traversal, and feedback to tune retrieval
weights over time. By default semantic recall uses the multilingual
`paraphrase-multilingual-MiniLM-L12-v2` sentence-transformers model only when it
is already available as a local path or local cache, with a dependency-free hash
fallback. The default runtime never downloads model weights.

## What Is Included

- Uniform node storage in `src/living_memory/storage.py`.
- Scope-aware retrieval in `src/living_memory/retrieval.py`.
- Consolidation, correction ingestion, and soft decay in
  `src/living_memory/consolidation.py` and `src/living_memory/decay.py`.
- FastMCP tool/resource/prompt registration in `src/living_memory/server.py`.
- End-to-end acceptance tests in `tests/test_acceptance_contract.py`.
- Developer docs in `docs/architecture.md` and `docs/mcp-interface.md`.

## Recall and feedback

`memory_recall` commits selected-node access and its durable event only after
required passage reads, correction references and response shaping succeed.
A failure leaves no new consumption event for later explicit or implicit
feedback. Marks resolve against earlier eligible events before the current
event exists. Optional recall maps, formed-delivery receipts and index
maintenance can fail without cancelling the required answer or its attribution.
Forming and saving an answer does not prove network delivery; an interrupted
transport has an unknown outcome and does not authorize automatic replay.

Search and model preparation precede the short consumption transaction. Direct
Python callers retain independent `log_access` and `log_event` controls;
disabling both opens no consumption transaction.

Explicit `used` / `irrelevant` feedback goes on the next naturally needed
`memory_recall`, `memory_remember` or `memory_teach` call. A necessary
`memory_lookup` can come first; it has no mark fields and leaves feedback
pending. Do not add a recall or fabricate a write solely for marks. Mark actual
use and query-relative irrelevance; relevant unused results need neither mark.
The optional arm requests marks and the mandatory arm requires them, using the
shared protocol producer in `server.py`. Caller/session attribution and the
once-only credit shared by lookup, explicit use and grounded feedback retain
their ordinary boundaries.

## Layout

- `src/living_memory/` - server, store, retrieval, consolidation, resources,
  prompts, and configuration code.
- `tests/` - automated tests.
- `docs/` - project-facing documentation.
- `scripts/` - local automation used by package scripts and development workflows.

## Commands

```sh
npm run setup:python
npm run check
npm test
python -m pytest tests/test_acceptance_contract.py
npm run server -- --help
npm run server -- ./living_memory.sqlite3
```

`npm run check` installs the Python test/runtime dependencies into the ignored
`.cache/python-deps` directory and runs the Python test suite through
`scripts/check.sh`. The suite includes a real FastMCP integration smoke test
that instantiates the server, verifies the four-tool default discovery surface,
and exercises all nine registered core tools locally.

`scripts/setup-python.sh`, `scripts/test.sh`, and `scripts/server.sh` use
`.cache/python-deps` by default. Setup reuses a cache when its `pyproject.toml`
fingerprint and target preparation format match and `fastmcp` and `pytest`
import successfully. Preparation installs into an empty temporary directory,
checks `fastmcp`, `pytest`, and `sentence_transformers`, then publishes the
complete target with its readiness marker. Failed preparation leaves no ready
marker; updates replace the whole dependency tree, including package metadata.
Compatible reuse is read-only and does not install or load the encoder. A worktree
without a cache can borrow a compatible main-checkout cache through a symlink.
If a borrowed cache needs installation, setup replaces only the local symlink
with a checkout-owned directory; the borrowed cache and its marker stay intact.

Set `LIVING_MEMORY_DEPS_DIR` to select a different writable dependency directory.
This explicit override also authorizes installation through a symlink at that
path. Setup, test/server scripts, and checkout imports use the same override;
an unset or empty value selects the default directory.

## Retrieval floor maintenance

Preview persisted retrieval-weight changes with the maintenance CLI:

```sh
python -m living_memory.maintenance reseat-floors --db ./living_memory.sqlite3
python -m living_memory.maintenance reseat-floors --config ./config.toml --dry-run
python -m living_memory.maintenance reseat-floors --config ./config.toml --apply
```

The default and `--dry-run` open an existing database with SQLite `mode=ro`.
They do not create a database or parent directories, initialize or migrate the
schema, backfill fields, seed weights, or write data. A missing, unreadable or
incompatible database fails with a diagnostic on stderr and exit status 2,
without a successful audit on stdout. The existing schema must support the
floor audit's weight and evidence queries; preview never repairs it.

Both modes use the same storage floor calculation, including scope-family
configuration and vector/graph evidence. `--config` supplies the TOML policy
and database path; `--db` overrides that path. Successful JSON output includes
`mode`, `applied`, `affected_count`, and changed scopes with configured `floor`
and `before`/`after` weights. `--apply` opens the ordinary writable store,
including its initialization, migrations and backfills, and persists the
audited weight changes. It recalculates from the data available at execution.

Preview uses ordinary SQLite reads, including committed WAL data. SQLite may
create or use `-wal`/`-shm` sidecars for an existing WAL database, so this is a
schema/data read-only guarantee, not byte-for-byte filesystem immutability.
SQLite's [read-only WAL requirements](https://sqlite.org/wal.html#read_only_databases)
still apply. Each CLI invocation loads the installed source; updating this
command does not require restarting an LM server.

## Typed-edge backfill

Preview derived relations in an existing database, or explicitly apply them to
a disposable snapshot:

```sh
python -m living_memory.edge_backfill derive-edges --db ./living_memory.sqlite3
python -m living_memory.edge_backfill derive-edges --db ./living_memory.sqlite3 --dry-run
python -m living_memory.edge_backfill derive-edges --db ./snapshot.sqlite3 --apply
```

Default and `--dry-run` choose the existing SQLite `mode=ro` reader before
opening the database. They do not create files or parent directories, initialize
or migrate the schema, backfill fingerprints, seed weights, or write database
data. Missing paths, invalid SQLite files and schemas that cannot support the
backfill queries fail with a diagnostic on stderr and exit status 2, without a
successful report. The operation closes its connection on success or failure.
Normal SQLite WAL visibility and sidecar requirements apply; preview is not a
frozen snapshot or a promise of no filesystem bookkeeping.

The audit counts semantic relations using the shared relation policy. Preview
projects both new rows and semantic replacements of technical rows; on unchanged
input these match apply's totals and typed shares. Existing semantic relations
keep their weights and metadata. `--annotate-derived-from` optionally adds R5a
metadata without changing the existing edge's weight or type. Reapplying is a
no-op. The reported rollback SQL removes only newly inserted marked rows;
semantic replacements and R5a annotations remain.

`--report audit.json` writes JSON and a matching Markdown report; `--md` selects
the Markdown path when `--report` is used. Without `--report`, JSON goes to
stdout. `--apply` retains ordinary writable-store initialization and writes the
derived changes in one transaction. Applying to live `global.sqlite3` remains
operator-only. CLI source updates require no LM service restart.

Disposable checks: `python -m pytest tests/test_edge_backfill.py`.

## Post-session insight evaluation

`scripts/postsession_insights.py` evaluates the gate and proposes insight
operations from an explicitly supplied corpus index. It never executes the
proposed operations. Calibration reads `train` and `eval`; extraction reads
`eval`. The reserved split stays outside this evaluator.

```sh
python3 scripts/postsession_insights.py --run \
  --index /path/to/corpus-index.jsonl --manifest /path/to/corpus.json \
  --db /path/to/memory.sqlite3 --out /path/to/private/report.json
```

`--db` names an existing Living Memory database. Extraction opens it with
SQLite `mode=ro`, without schema initialization, migrations, fingerprint
backfills or seeding. No preliminary copy is required. Normal SQLite WAL
visibility and sidecar requirements apply; this is not a frozen snapshot.
The extraction operation closes its reader after all workers finish, both on
success and on failure, before report writing.

The helper is a context manager: use `with build_deduper(path) as deduper`
and finish all workers inside that context. `StoreMemoryIndex` borrows the
store and propagates search errors. The helper checks the search schema even
when no proposals are produced. Missing paths, invalid SQLite files and
unsupported search schemas fail explicitly; they are never created or repaired
and do not produce a successful evaluation report. SQLite failures return
CLI exit code 2.

Without `--db`, dedup against existing memory is explicitly unavailable.
When extraction outcomes are present, the report records
`dedup.index_available: false` and unavailable verdicts. `--calibrate` alone
never opens memory, even if a `--db` argument is supplied. `--report` combines
calibration and extraction and opens memory only for extraction.

Only equal content, scope and recorded extraction source (transcript hash and
source span) suppress a proposal. Related or independently sourced claims
remain proposals. Candidate retrieval stays bounded; judge selection, budgets,
split restrictions and report fields retain their ordinary meaning.

Disposable checks run without a real corpus or external models:

```sh
python3 -m pytest tests/test_postsession_insights_cli.py
```

These checks cover reader lifetime, unchanged schema/data including a NULL
fingerprint, missing and unsupported databases, concurrent verdicts against a
synthetic index, fake-judge reporting and WAL visibility. They do not measure
production frequency, copy costs, fleet savings or judge quality.

## Reading memory

`memory_recall(query=...)` returns knowledge directly in `results`: identity,
content, delivery class, re-fetch reference, and validity/corrections. Read
`results[i].content` and use its `id` for feedback. When an excerpt or duplicate
preview is insufficient, pass `content_ref.node_id` to `memory_lookup` for
complete content and current validity. Schema recipes retain complete content;
duplicate previews still provide `content_ref` access to the whole recipe.

Both recall views and exact-id lookup include `superseded` and `superseded_by`
on each result (beside `node` in diagnostic recall). `superseded: true` means
an existing correction relation targets this node, independently of
`provenance.corrections`, decay, and whether the correction qualified for the
answer. `superseded_by` lists direct correction ids for `memory_lookup`: only
existing nodes admitted by the request's scope plan are linked. Decayed
corrections remain addressable history with their own decay flags. An empty
list never cancels known supersession; missing or out-of-scope corrections
supply neither links nor text. Without a requested scope, links can cross scopes.

Semantic similarity retrieves and ranks knowledge; it does not establish that
claims are identical or corrected. Duplicate selection and delivery require
exact content with matching scope and recorded source identity. A different
condition, prohibition, source or derivation remains independent. The bounded
top-k may still omit a candidate by rank; its omission is not a duplicate claim.

Ordinary remembered text and context labels are data, including correction
wording, negations and quotations. ULID mentions can create content references;
supersession requires `memory_teach` or an explicit typed connection. Shared
work context, nearby timestamps and problem/fix labels do not establish
causality. Inferred-only `caused` history (`rule=R2a` or
`kind=failure_resolution`) is excluded from graph evidence; explicit causal
connections remain usable. A caller's `basis=explicit` survives merged inferred
markers. A later explicit connection replaces inferred metadata; historical
unmarked explicit writes overwritten by inferred markers cannot be reconstructed
from timestamps or work context. The stored rows remain available for inspection.

Exact-copy lifecycle bookkeeping uses `related(kind=duplicate_content)` rather
than correction semantics. Known historical `supersedes` rows from the drain
(`kind=drain_near_duplicate`), offline cosine collapse
(`by=lm_collapse_near_dups.py`, `reason=near-duplicate collapse`) and technical
copies (`kind=duplicate_content`), and inferred text corrections
(`rule=R1c` or `kind=content_correction`) confer no correction validity, rank
priority, semantic graph traversal or decay authority. An explicit nonempty
`correction` text in relation metadata retains its meaning even when merged
with inferred metadata. Unknown supersedes metadata retains its meaning. Exact
copies can redirect anchors and source lineage only after checking their
current content and source identity. Historical rows and stored decay flags
remain intact; this policy does not reconstruct prior lifecycle side effects.

New traces from `memory_remember`, its rejected alternatives, and
`memory_teach` prepare search chunks from their exact normalized source text
before taking the shared store lock or opening a SQLite write transaction.
Remember preserves its input text unchanged. Encoder loading, inference and
fallback have their own guard, released before store apply. The writer checks
exact source identity and commits knowledge, successful chunks and required
feedback together, using current scope and validity. A derived-index failure
returns `indexing: {"status": "deferred", "node_ids": [...]}` while preserving
the knowledge write; required failures still raise and roll it back. Reads and
independent writes that do not need the busy encoder can proceed during
preparation. Query encoding, including feedback anchors, still shares the
encoder. This provides neither a general latency guarantee nor exactly-once
writes after response loss.

`memory_recall` also prepares its stripped query and immutable query vector
before taking the shared store lock, using the same serialized encoder. An
independent client's exact lookup can finish while this query encoding waits.
Retrieval and anchors reuse that vector; scope, source validity, ranking,
feedback and delivery history are read under the lock after preparation.
Direct `MemoryRecallService.memory_recall` callers can still pass query text.
This preparation is per call, with no query cache.

`retrieval_context` runs separate concept, schema and decision-history recalls
through one `MemoryRecallService`, sharing compatible query and corpus
preparation while retaining each selector's limits, depth, ranking and
qualification. Every recall checks current corpus revisions, scope and validity.
The Python builder creates a service per operation by default; it and standalone
selectors also accept `recall_service` for the same store. Native prompts reuse
their persistent preparation service and its pending-error cursor, preparing
queries and bounded pending chunks before the runtime lock. Native selectors
use `repair_pending=False`; standalone callers retain repair by default.

The shared tokenizer reuses up to 4096 canonical token strings per process,
keyed by token and effective Cyrillic stemming policy. Tokens longer than 128
characters normalize in full without retention. Evidence and default modes
share normalization while keeping their own numeric and stop-word filters,
token order and repetitions. `reset_cyrillic_stem_cache()` clears both the
policy reading and token results. Knowledge bodies and output lists are not
cached.

Feedback from lookup,
explicit marks and grounded consumption uses an existing anchor's exact
scope/fingerprint identity and stored vector without encoding the query.
Absent identities are prepared outside the store lock and before the required
remember/teach transaction. The guarded operation reads current deliveries,
grounding, claims and anchor identities again, keeping explicit marks before
implicit closure. One further preparation can cover concurrent new inputs or
a deleted anchor; continuing changes reject required feedback before knowledge
or feedback writes, without consuming feedback. Exact lookup tolerates failed
credit preparation, including continued input changes or anchor-read errors:
it reads current source and validity under the store lock and skips credit with
a bounded diagnostic. Source/validity read errors still fail the lookup. Optional
usage writes are never retried after failure. A failed encoder batch leaves
existing-anchor effects available. Query vectors live only within the operation.

Pending indexing selects at most 16 oldest eligible sources after due decay and before recall
selection. Whole-source and complete chunk vectors are prepared without the
store lock or a SQLite transaction. The writer rechecks content, scope,
liveness, representative and index identity before applying each complete
replacement. Concurrent changes win; independent failed repairs stay pending
while other selected sources and current knowledge remain available. Ranking,
response shaping, event/map/delivery attribution and shared-connection SQL stay
under the store guard. These boundaries establish independent read availability
during preparation, not a general production latency or speedup guarantee.

Exact-id lookup preserves the complete original content. `context_key` adds
one original context field; `view="full"` returns all stored node fields plus
the same relation-derived validity. Knowledge reads project content, identity,
stored validity and the selected context value directly from storage, without
materializing a full node or decoding unrelated metadata in Python. Relation
validity and optional usage credit use fresh reads under the runtime guard.
An explicit lookup scope restricts
discovered correction links using recall's scope hierarchy. Explicitly requested
ids still return regardless of scope, with differences in `scope_mismatches`.
Neither supersession nor its absence establishes that a correction or original
is correct. Chains and cycles expose direct links without chasing a final
replacement; read each node's own validity. Links do not count as delivered
results, lookups, or confirmed use for feedback attribution.

Consolidation uses the configured encoder and cached source vectors regardless
of the total trace count. Same-scope clusters still require their merge floor,
similarity and current-source checks. Global promotion requires valid similar
concepts from at least three distinct project scopes in the same procedure
group. Candidate reads use pages of 500 and share each vector across the pass's
seeds, retaining the best match per scope; later scopes remain searchable.
Scoring runs outside the shared-store guard and each atomic writer rechecks
current evidence. Other projects are not re-embedded. Existing derivations
continue to refresh from their recorded sources. First construction and later
writes resolve each source through the same correction and validity checks.
Promotion counts the union of current original authors through recorded
derivation inputs; a materializer is not an additional author. Unavailable or
unidentified legacy inputs retain the maximum declared support, never a sum of
possibly overlapping counts. Scope diversity still contributes its confidence
bonus, subject to storage's confidence cap of 0.5 for at most one author.
A correction replaces its explicit source; time gaps and cluster membership
cannot displace independent inputs. Replacing a recorded derivative explicitly
displaces its cited inputs when they occur in the same candidate. Original
records and correction history remain available through exact lookup.
Status and stats expose
counts, confidence, promotions and effective retrieval policy; retrieval uses
BM25, ready vector evidence and requested graph traversal without a corpus
maturity stage.

Consolidation digests and bounded excerpts share conservative source passage
boundaries: paragraph breaks and sentence-ending punctuation with a capitalized
continuation. Periods also require a lowercase alphabetic terminal word longer
than three letters; soft line wraps and semicolons stay inside a passage.
Digests select complete passages within
their existing budget, preserving the text and order inside each passage.
Exact repeats may share a slot; similarity only ranks candidates. Useful
complementary passages remain eligible. A passage too large for a digest stays
in the recorded source, available through provenance and exact-id lookup.
These rules apply to new derived writes and ordinary source-based refreshes,
including consolidation, correction and promotion. Reading or opening a store
does not rewrite existing derived content; there is no bulk migration.

An explicit `memory_connect(new, old, "supersedes")` from a live, distinct
replacement refreshes recorded derivatives atomically with the relation and
anchor moves, using the same materializers as teach and forget. A derivative
without a known materializer becomes decayed and remains available by exact
lookup. Required refresh or preparation failures roll back the whole change.
Connect returns the connection, creates no corrective trace, and preserves the
existing endpoints, scopes and provenance. Inferred-only and bookkeeping
relations do not apply corrections.

Teach, forget and explicit connect discover consumers from a projection of recorded writer and
source fields before loading full nodes. Admission uses the same materializer
policy as refresh, including legacy cluster records; explicit `derived_from`
edges supply inputs only for nodes without a recognized writer. Ordinary recall
citations do not require their bodies, context or full provenance to be decoded.
Discovery crosses scopes and follows correction and duplicate predecessors;
required source failures still abort the atomic change. Preparation and the
authoritative writer each discover current dependencies. This uses the existing
citation query and primary-key body reads, without a separate dependency index.

Evidence passages retain their source labels and heading. Semantic retrieval
carries the winning chunk's verified source address into delivery. All conservative
passages overlapping that address are budgeted together before optional enclosing
paragraphs and adjacent context. Further optional matches retain their whole
paragraph within the source record or are omitted when that group cannot fit.
Without a usable address, lexical source-record and passage relevance selects
the anchor. Optional context
cannot displace the anchor, and the vector window itself is never presented as an
independent claim. Delivery does not embed or search again.

`[Context omitted]` marks selection that leaves out source passages. If the anchor
itself exceeds the budget, delivery provides its source prefix with `…` and
`[Incomplete passage]`; very small budgets may fit only a prefix and `…`, including
when the evidence heading and label cannot fit. These are textual heuristics,
not a guarantee of semantic completeness. Conditions in omitted sentences or the
missing tail may matter; use the full original via `content_ref` to resolve them.

Chunk writers save verified character coordinates alongside their vectors.
Existing chunks without coordinates retain lexical excerpt selection for the
current call. After recording its response, feedback, map and delivery facts,
MCP recall can refresh one selected legacy snippet's
chunk index, limited to 32,768 source characters and 32 old and new chunks. It
uses the current encoder and verified source coordinates; the next ordinary
recall scores that new index. No old score is assigned to a replacement chunk.
Source identity reads and the short recheck/apply use the shared store lock;
tokenization and encoding use immutable input outside that lock and the SQLite
transaction. Independent exact lookups can proceed while this recall waits for
its repair. Concurrent source, scope, decay or generation changes cancel the write. Full deliveries,
duplicate/withheld results, current indices and oversized records do not incur
refresh encoding. Failed or unsupported refreshes preserve lexical fallback and
full lookup. An indexing exception suppresses refresh for that immutable scored
chunk row in this service instance; an in-progress attempt reserves that same
row until completion. Other sources and replacement rows remain
eligible. Repeated reads of the same winning row do no further refresh work.
Different winners can attempt at most 32 refreshes per unchanged eligible index,
with at most one attempted source per read and 32 chunks encoded per attempt.
Metadata-only updates and opening a database do not refresh these
chunks; missing or stale chunks retain their existing indexing path.

Before retrieval, recall attempts at most 16 sources with missing or stale
chunk indexes. Each service keeps a cyclic timestamp/id position for up to 128
recent search-scope sets, shared across MCP clients. Selection reads current
pending identities, advances past failed sources and wraps so they can retry.
Each body is decoded separately after selection; malformed unrelated pending
input is deferred with a bounded diagnostic and stays eligible for retry.
Strict maintenance lists, pending discovery and storage read errors, and errors
decoding knowledge required by retrieval remain explicit.
Different scope sets have independent positions. Restarting the service or
evicting an inactive position starts that scope set at its oldest pending source;
progress is not persisted. Source and index revalidation, atomic replacement and
per-source failure isolation protect current knowledge during each attempt.

Recall before acting and at every new turn of thought, including mid-work.
Use relevant knowledge and applicable complete recipes under current system
and operator instructions and the actual task conditions. Those instructions
define permissions and restrictions; neither a nonempty result nor
`level:schema` creates authority. A schema preserves a useful recipe as a whole;
its storage level does not establish that the procedure applies now.

A nonempty recall neither forbids nor replaces necessary checks of changeable
claims against current sources. For example, after recalling a historical
service overview, read the current overview or code when the task requires it.
Use what remains relevant and immediately correct a refuted claim through
`memory_teach(trace_id=..., correction=...)`, which creates the `supersedes`
edge. An applicable exact recipe retains its steps and conditions; access to
that knowledge is independent of permission to act.

Remember only verified knowledge expensive to re-derive. Use `used` for results
you used and `irrelevant` only for results off-topic for their recall query;
a relevant historical result is not off-topic merely because it needs checking.
The mandatory feedback arm requires those marks on the next call. Both arms,
server instructions and `memory://prompt/retrieval_context` share the same
knowledge and authority contract.

Implicit recall attribution requires compatible shared `task`, `session_id`,
or MCP transport identity. A conflicting task or session prevents attribution;
an equal task or session can continue work after reconnect within the recall's
scope. Equal transport can also bridge scope-less recall/ingest defaults.
Similar text and equal agent names alone never establish that a recall was
shown to this work. Without a compatible recall, new knowledge is still saved
without implicit provenance, credit or query anchors. `memory_teach` takes
identity from the current call, preserving the source's subject and step
location without inheriting its author/session/task.

Explicit `used` / `irrelevant` feedback can follow a reconnect when both calls
retain the same nonblank, caller-unique `session_id` in recall's
`ambient_context` and writes' `context`; both-present task or agent conflicts
reject attribution. To read broadly, omit both `scope` and
`ambient_context.scope`: ambient `project` is only a preference. Such a saved
read can receive marks on a write to any knowledge destination, for example
`context.scope='project:demo'`. A restricted read requires the current scope to
match its saved scope. Named read fallbacks alone do not authorize feedback.
The marked id must belong to the saved delivery within the bounded time/event
window; repeating a mark cannot earn credit twice. Missing session identity
cannot be inferred after reconnect. Same-transport marks retain their contract;
lookup credit remains transport-only.

Delivery selection caps the newest caller events before filtering by time or
requested id. It reads eligible event membership in order and stops when every
requested id has a proven delivery. Identity and membership share one read
snapshot. An older unnecessary malformed result body
does not block that selection. A malformed body still needed to resolve a mark,
or a required storage failure, remains an error. Non-array bodies and non-object
result elements cannot prove delivery; absent eligible membership remains
`not_delivered`. Selection itself writes nothing.

Request `memory_recall(..., view="full")` to inspect ranked entries with their
`node`, scores, compacted metadata, and recall-map selection accounting (`sel`).
The knowledge view keeps the map's navigation, filtered labels and limitations;
the complete map remains stored on the recall event. For original metadata, use
exact-id
`memory_lookup(..., context_key="field")` or `view="full"`. The
[interface contract](docs/mcp-interface.md#memory_recall) describes delivery,
feedback, and warnings.

## Tool discovery and operator access

By default, FastMCP `tools/list` advertises exactly the four agent tools:
`memory_recall`, `memory_remember`, `memory_teach`, and `memory_lookup`. The five
maintenance tools — `memory_consolidate`, `memory_forget`, `memory_connect`,
`memory_status`, and `memory_health` — remain registered and directly callable
by name over the in-memory, stdio, HTTP, and HTTPS/TLS transports; they are only
hidden from discovery. This is a tool-list diet, not an authorization boundary;
configured transport authentication still applies to every call.

Set the literal environment value `LM_EXPOSE_OPERATOR_TOOLS=1` before starting
the server to advertise all nine core tools. Python embedders can instead pass
`expose_operator_tools=True` or `False` to `create_mcp_server`; an explicit
constructor value takes precedence over the environment, while `None` (the
default) falls back to it.

Attestation has an independent, default-off control. `LM_EXPOSE_ATTEST=1` or
`create_mcp_server(expose_attest=True)` registers and advertises
`memory_attest`; an explicit `expose_attest=False` overrides its environment
variable. Neither attestation control changes maintenance-tool visibility, and
the operator-tool control does not enable attestation.

FastMCP and sentence-transformers are declared runtime dependencies. When
sentence-transformers is absent in a source checkout or a model is unavailable
locally, Living Memory uses its deterministic multilingual hash fallback.

Embedding model downloads are opt-in. Set `LIVING_MEMORY_EMBEDDING_BACKEND=online`
only in environments where network-backed sentence-transformers loading is
allowed.

## Health report

`memory_health(scope=..., window_hours=168, top_stale=5)` reports activity,
staleness and feedback closure without recording accesses. For active traces
in the selected scope, age percentiles and `staleness.top_stale` both use time
since `last_accessed`, or `timestamp` when the access date is absent or empty.
Missing or invalid reference dates are omitted, and future dates have age zero.
An invalid nonempty access date does not fall back to creation time.
`top_stale` returns at most the requested number, oldest by this age first;
zero disables the list. This diagnostic leaves decay TTL and retrieval unchanged.

`feedback_closure.identity_coverage` partitions events in the selected scope
and creation-time window. `explicit` means a nonblank persisted `session_id`,
including the writer's `session` alias and string conversion; agent and task
are optional. `transport_only` means no such session but a nonempty transport
stamp, and `none` means neither usable identity. A class describes saved identity,
not a guarantee that a feedback request matches: reconnect still requires the
existing compatible scope/context, delivery and time checks. Each class reports
event and closed-event counts and their ratio; closure means
`feedback_applied = 1`, regardless of when feedback was applied.

## Optional decay maintenance

By default, a live concept or schema guards its `source_traces` from TTL decay.
With the operator's `LM_DECAY_CONCEPT_COVERED` opt-in, sources unread since
coverage can retire. Materializers keep each source's continuous coverage start
in `provenance.source_coverage`: repeats preserve it, new or replacement IDs
start at inclusion, and removed IDs lose their entries. Missing or invalid
starts remain unknown across maintenance; owner creation and `consolidated_at`
do not establish source age. Across live owners, unknown age takes precedence,
otherwise the earliest start wins. A recorded read after that start (or with
unknown age) prevents coverage retirement; the opt-in TTL fallback still applies.

`memory_recall`, `memory_remember` and `memory_consolidate` each check the
scope-agnostic decay interval once and include `auto_decay` in their response:

- `null`: skipped because the interval is disabled or a completed sweep is recent.
- `{"swept": true, ...}`: completed sweep with `decayed_count`, `expired_count`,
  `superseded_count`, `duration_ms` and the saved `last_decay_sweep_at` timestamp.
- `{"swept": null, "error": "..."}`: optional maintenance failed; completeness
  is unknown. Changes may already be durable. The server logs a warning with
  the exception, while the primary operation retains its own result.

`LM_DECAY_SWEEP_INTERVAL_SEC` defaults to 3600; `0` disables the automatic
sweep. A failed attempt is not recorded as a completed sweep and adds no retry
delay: the next ordinary call checks the same due condition. Recall checks decay
before selecting pending index repairs. Forced `POST /admin/decay-sweep` bypasses
the interval and returns HTTP 500 on failure. Errors in the requested
consolidation itself also remain errors.

## HTTP transport and TLS

The server speaks the stdio transport by default. Pass `--transport http`
(or `--transport sse`) with `--host`/`--port` to serve the `/mcp` endpoint and
the admin routes (`/health`, `/admin/info`, `/admin/token`, `/admin/restart`,
`/admin/decay-sweep`) over the network.

Those transports serve plain `http://` unless TLS is configured. To serve
`https://` instead, supply a PEM certificate and its matching private key:

```sh
npm run server -- --transport http --host 0.0.0.0 --port 8000 \
  --tls-cert ./tls/server.crt --tls-key ./tls/server.key
```

| CLI flag | Environment variable | Purpose |
| --- | --- | --- |
| `--tls-cert PATH` | `LM_TLS_CERT` | PEM certificate file. |
| `--tls-key PATH` | `LM_TLS_KEY` | PEM private key for that certificate. |

The CLI flag takes precedence when both it and the environment variable are
set. TLS is opt-in, and the certificate and key must be supplied together:
provide both to serve HTTPS, or omit both to keep the default plain-HTTP
behavior unchanged. Supplying only one is a configuration error and the server
exits without starting. The stdio transport ignores these flags. Clients must
trust the certificate, so a self-signed certificate (handy for local testing)
requires the client to be pointed at the same cert or told to skip
verification.

Bearer authentication composes with TLS rather than being replaced by it. When
`LM_AUTH_TOKEN` is set, the admin routes still reject missing or invalid tokens
(HTTP 401) over both plain HTTP and HTTPS — and over HTTPS the token is no
longer sent in cleartext on the wire.

`POST /admin/token` rotates that bearer token in place. A request authorized by
the *current* token supplies the replacement as JSON (`{"token": "<new>"}`);
the server then accepts only the new token on both the admin routes and the
`/mcp` surface, with no restart required. The new value is persisted to the
SQLite state file, so it survives a restart and takes precedence over
`LM_AUTH_TOKEN` on the next start — `LM_AUTH_TOKEN` only seeds the token before
the first rotation. The token value never appears in the response, logs, or any
tracked file. (Send the rotation over HTTPS so the new token is not exposed on
the wire.)

## Documentation

- [Architecture](docs/architecture.md)
- [MCP Interface](docs/mcp-interface.md)
- [Deployment](docs/deployment.md) - how each host is updated, restarted and
  verified, and how to tell that one is behind `master`.
  `scripts/check_deployed_protocol.py` answers that last question executably:
  it compares the protocol texts a running host serves against this checkout
  and exits non-zero with a diff when they differ.
