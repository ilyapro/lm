# memory_recall Latency Profile — Validation & Bottleneck Localization

Goal: `optimization/profile-latency` (P1). Profile `memory_recall` to (a) validate it
is slow and (b) locate the bottleneck. Discovery node — read-only; evidence below is
reproducible against a copy of the live DB.

Date: 2026-06-02. Commit: `git rev-parse --short HEAD` on branch `optimization--profile-latency`.

## Verdict

**Recall is genuinely slow, and there are TWO distinct, independently-measured bottlenecks:**

1. **Steady-state median (p50≈200ms) is dominated by GRAPH TRAVERSAL** — 64–84% of every
   hot recall (135–195ms). The mechanism is an **N+1 query fan-out**: the depth-1 BFS seeds
   from *all* pre-graph candidates and issues **one `store.get_node` SQL round-trip per
   neighbor reached** — ~1,969 single-row queries for a heavy scope. This regressed ~3–4×
   vs the 2026-05-22 baseline because the `related` graph grew 5× (8,407 → 43,920 edges,
   with super-hubs up to degree 2,209).
2. **The p99 tail (≈4,116ms) is the COLD embedding-model load** — the first recall after each
   server start lazily loads `paraphrase-multilingual-MiniLM-L12-v2`, measured at **3,811ms**.

Both map almost exactly onto the live `memory://latency` percentiles (see Mapping).

## Method

- DB: online backup of the live store `LM_DB_PATH=/home/sfx/.local/share/living-memory/global.sqlite3`
  → `/tmp/lm-profile-copy.sqlite3` (246 MB; `sqlite3 .backup`, safe while the server runs).
  Live DB was 246 MB main **+ 69 MB un-checkpointed WAL** at capture time.
- Backend: **real** SentenceTransformer (`auto`), exactly as the live server runs — the model
  IS cached at `~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2`.
  (The prior 2026-05-22 artifact forced `LIVING_MEMORY_EMBEDDING_BACKEND=hash`, which is why it
  never saw the embedding cost or the cold-load tail.)
- Harness: `artifacts/discovery/profile_recall.py` — wraps each internal phase
  (`_collect_bm25/_collect_vector/_collect_schema_triggers/_collect_graph/rank_candidates/
  _supersedes_sets/_record_result_access`, `embedder.embed`, `record_recall_event`) with
  `perf_counter`; 2 warmups + 9 measured iterations; reports median & p95; runs both no-write
  and write-enabled passes.
- Env: Python 3.12.3, numpy 2.4.3, sentence-transformers 5.5.0.

## DB growth since the last baseline (2026-05-22 → 2026-06-02)

| metric | 2026-05-22 baseline | now | change |
| --- | ---: | ---: | --- |
| DB size (main) | 95.7 MiB | 246 MB (+66 MB WAL) | ~2.6× |
| active nodes | 5,128 | 5,269 | ~flat |
| `related` connections | 8,407 | **43,920** | **5.2×** |
| `contradicts` / `supersedes` | 411 / 34 | 1,294 / 119 | 3.1× / 3.5× |
| recall_events | ~5,114 | 30,534 | ~6× |
| active embedded rows (octopus / ae / online / global / lm) | — | 2,401 / 1,206 / 815 / 460 / 310 | — |

The graph blew up while node count stayed flat → **edge density per node exploded**, which is
exactly what makes graph traversal the new dominant cost.

## Evidence 1 — cold vs hot embedding (the p99)

```
COLD embed (incl. model load): 3811.4 ms     # first recall after server start
HOT  embed (real model/query): 9.10 ms median
HASH embed (prior benchmark) : 0.071 ms       # 128× cheaper, lower quality
```

## Evidence 2 — per-phase decomposition (hot, 9 iters, median ms)

| scope (depth=1) | total | **graph** | bm25 | vector_scan | embed | schema | rank | record_acc+evt | graph % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| project:lm     | 186 | **135** | 18 | 16 | 8 | 0.7 | 5 | ~1.7 | **72%** |
| project:octopus| 285 | **181** | 23 | 36 | 8 | 16 | 11 | ~1.7 | **64%** |
| project:ae     | 287 | **195** | 31 | 33 | 9 | 6 | 9 | ~1.7 | **68%** |
| global         | 214 | **179** | 10 | 9 | 9 | 0.8 | 4 | ~1.7 | **84%** |
| octopus, depth=0 | **97** | 0 | 30 | 36 | 8 | 16 | 5 | ~2.3 | — |

- Graph p95 (isolated) is 268–316 ms — graph is also the **variance** driver.
- **depth=0 removes graph entirely → recall drops from 285ms to 97ms (−66%)** on octopus.
- Write path (`record_access` ~1.5ms, `record_event` ~0.2ms) is **negligible** — not the bottleneck.
- `_supersedes_sets` is cheap today (1.5–2.3ms, only 119 rows) but is a near-full scan that would
  regress if a dedup fix adds many `supersedes` edges (no `type`-leading index exists).

## Evidence 3 — graph is an N+1 `get_node` fan-out (octopus depth=1, 303ms recall)

```
list_connections : 365 calls,  40.3 ms  (0.110 ms/call — indexed, fine)
get_node         : 1969 calls, 158.6 ms (0.081 ms/call)   <-- dominant
graph DB time    = 198.9 ms of 303.1 ms = 66%
```

Candidate fan-out: pre-graph **365 → post-graph 1,550** candidates for 10 returned results.
`related`-edge degree: mean 11.7, p95 44, **max 2,209** (single super-hub
`01KSF1JQXD8XE7DHJ6NAVZGXS4`; next 992). The BFS (`retrieval.py:401-459`) seeds from every
candidate and, for each reached neighbor not already a candidate, calls `store.get_node`
one-at-a-time (`retrieval.py:450`). Connection lookups are indexed
(`idx_connections_source_type`, `idx_connections_target_type`) and cheap; the cost is the
**volume of per-neighbor single-row round-trips** plus **unbounded hub fan-out**.

## Mapping to live `memory://latency`

```
memory_recall: count 23, p50 200ms, p95 986ms, p99 4116ms   (memory_status p50 19ms)
```

| live percentile | explanation | profiled evidence |
| --- | --- | --- |
| p50 ≈ 200ms | steady-state recall, graph-dominated | profiled medians 186–287ms across scopes |
| p95 ≈ 986ms | graph p95 (268–316ms isolated) amplified by the shared `runtime_lock` + 66 MB WAL on live | graph is the variance driver |
| p99 ≈ 4116ms | cold model load on first recall after restart | measured 3,811ms cold embed |

## Levers for the downstream `optimize-latency` node (evidence-backed, not prescriptive)

1. **Kill the graph N+1**: batch neighbor hydration (`get_node ... WHERE id IN (...)`) and/or
   one `list_connections` per BFS frontier instead of per node → removes ~159ms of the ~199ms graph cost.
2. **Bound graph fan-out**: seed the BFS only from the top-K pre-graph candidates (not all ~365),
   cap neighbors per node, and/or skip/penalize super-hub nodes (degree ≫ p95=44). Most of the
   1,550 expanded candidates never reach top-10.
3. **Pre-warm / share the embedder**: load the SentenceTransformer at server start (or warm on
   first connect) so no user-facing recall eats the 3.8s cold load → kills the p99 tail.
   The MCP server already keeps one long-lived service (`server.py:471-473`); only the *first*
   `embed` pays the load.
4. (Secondary) Give `connections` a `type`-leading index or cache `_supersedes_sets` before any
   dedup work multiplies `supersedes` rows.

Disabling graph entirely is **not** recommended — it regresses causal/decision/correction recall
(`tests/test_graph_recall.py`, decision-history tests); the win is in *bounding* it, not removing it.

## Reproduce

```bash
# 1. fresh copy of the live DB (safe while server runs)
python3 -c "import sqlite3;s=sqlite3.connect('file:/home/sfx/.local/share/living-memory/global.sqlite3?mode=ro',uri=True);d=sqlite3.connect('/tmp/lm-profile-copy.sqlite3');s.backup(d);s.close();d.close()"
# 2. per-phase profile, real backend (as live):
PYTHONPATH=src python3 artifacts/discovery/profile_recall.py /tmp/lm-profile-copy.sqlite3
# 3. cold vs hot embed:
PYTHONPATH=src python3 -c "import time;from living_memory.embeddings import LocalEmbeddingModel as M;m=M();t=time.perf_counter();m.embed('x');print('cold',(time.perf_counter()-t)*1000);t=time.perf_counter();m.embed('y');print('hot',(time.perf_counter()-t)*1000)"
```
