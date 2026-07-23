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
that instantiates the server and exercises the nine registered tools locally.

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
