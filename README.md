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

Exact-copy lifecycle bookkeeping uses `related(kind=duplicate_content)` rather
than correction semantics. Known historical `supersedes` rows from the drain
(`kind=drain_near_duplicate`), offline cosine collapse
(`by=lm_collapse_near_dups.py`, `reason=near-duplicate collapse`) and technical
copies (`kind=duplicate_content`) confer no correction validity, rank priority,
semantic graph traversal or decay authority. Real corrections, including
`memory_teach`, and unknown supersedes metadata retain their meaning. Exact
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
This preparation is per call, with no query cache. Pending/legacy indexing and
feedback encoding can still run under the store lock; this is no guarantee of
availability throughout recall or of a general production speedup.

Exact-id lookup preserves the complete original content. `context_key` adds
one original context field; `view="full"` returns all stored node fields plus
the same relation-derived validity. An explicit lookup scope restricts
discovered correction links using recall's scope hierarchy. Explicitly requested
ids still return regardless of scope, with differences in `scope_mismatches`.
Neither supersession nor its absence establishes that a correction or original
is correct. Chains and cycles expose direct links without chasing a final
replacement; read each node's own validity. Links do not count as delivered
results, lookups, or confirmed use for feedback attribution.

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
current call. After shaping, MCP recall can refresh one selected legacy snippet's
chunk index, limited to 32,768 source characters and 32 old and new chunks. It
uses the current encoder and verified source coordinates; the next ordinary
recall scores that new index. No old score is assigned to a replacement chunk.
Concurrent source or generation changes cancel the write. Full deliveries,
duplicate/withheld results, current indices and oversized records do not incur
refresh encoding. Failed or unsupported refreshes preserve lexical fallback and
full lookup. An indexing exception suppresses refresh for that immutable scored
chunk row in this service instance; other sources and replacement rows remain
eligible. Repeated reads of the same winning row do no further refresh work.
Different winners can attempt at most 32 refreshes per unchanged eligible index,
with at most one attempted source per read and 32 chunks encoded per attempt.
Metadata-only updates and opening a database do not refresh these
chunks; missing or stale chunks retain their existing indexing path.

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
location without inheriting its author/session/task. Explicit `used` and
`irrelevant` marks retain their same-transport delivery contract.

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
