# Retrieval Weight Flow Inventory

Scope: discovery note for `retrieval-feedback-stability/reproduce-and-inventory`.
This artifact maps how retrieval weights are configured, read, mutated, surfaced,
and amplified through implicit recall feedback. It does not select the fix.

## Focused Reproduction

New regression: `tests/test_retrieval_feedback_amplification.py`.

- `tests/test_retrieval_feedback_amplification.py:46-52` marks the current
  behavior as `xfail(strict=True)`, so the test must fail internally on this
  branch and will XPASS once the mechanism is fixed unless the marker is removed.
- `tests/test_retrieval_feedback_amplification.py:57-59` disables adaptive
  retuning for a stable fixed-rate reproduction and creates the MCP server.
- `tests/test_retrieval_feedback_amplification.py:72-77` seeds one project trace
  through public `memory_remember` and confirms the project-family vector
  starting point is 0.3.
- `tests/test_retrieval_feedback_amplification.py:79-101` performs four cycles
  of five public `memory_recall` calls, then one public `memory_remember` that
  consumes exactly five pending recall events via implicit feedback.
- `tests/test_retrieval_feedback_amplification.py:103-110` verifies exact
  lexical recall still returns a BM25 top hit after the feedback pressure.
- `tests/test_retrieval_feedback_amplification.py:112-114` asserts the intended
  invariant: project BM25 should stay at or below 0.85 and vector should stay at
  or above 0.15. Current code collapses to BM25=1.0/vector=0.0, so this is the
  failing assertion under the strict xfail.

With fixed tuning, the project-family default starts at 0.7/0.3/0.0. Each top
BM25 result reinforced as useful applies `bm25_signal=1.0` and
`vector_signal=-0.25`; at learning_rate 0.05 this is raw +0.05 to BM25 and
-0.0125 to vector before normalization. Twenty reinforced recall events are
enough to clamp vector to zero.

## Configuration And Models

- `src/living_memory/config.py:13-18` defines the static
  `RetrievalWeightConfig` shape: BM25, vector, graph, learning_rate.
- `src/living_memory/config.py:21-26` defines default rows. Important defaults:
  `default=1.0/0.0/0.0`, `project=0.7/0.3/0.0`, `global=0.4/0.4/0.2`,
  `session=0.8/0.2/0.0`.
- `src/living_memory/config.py:38-47` puts retrieval weights into
  `MemoryConfig`.
- `src/living_memory/config.py:74-85` overlays TOML `retrieval_weights` entries
  on top of defaults without any policy bounds.
- `src/living_memory/models.py:93-142` defines `RecallEvent`; the `results`
  payload carries per-result score components that later become feedback input.
- `src/living_memory/models.py:145-167` defines `RetrievalWeights.normalized()`.
  If the raw total is zero, it returns pure BM25 `1.0/0.0/0.0`; otherwise it
  normalizes whatever non-negative values storage produced.

## Storage Source Of Truth

- `src/living_memory/storage.py:51-82` initializes the SQLite schema and seeds
  retrieval weights on store construction.
- `src/living_memory/storage.py:1121-1128` creates the `retrieval_weights`
  table. There are no DB constraints enforcing nonzero semantic or graph
  capacity.
- `src/living_memory/storage.py:1246-1264` seeds configured weights with
  `ON CONFLICT(scope) DO NOTHING`; existing rows are not overwritten by config
  defaults.
- `src/living_memory/storage.py:955-964` reads weights by checking the exact
  scope, then the scope family from `_scope_family`, then `default`.
- `src/living_memory/storage.py:966-990` upserts explicit retrieval weights for
  a scope.
- `src/living_memory/storage.py:992-1015` is the persisted mutation choke point:
  it adds `learning_rate * signal` per component, clamps each component with
  `max(0.0, ...)`, normalizes, then persists via `set_retrieval_weights`.
  This is where vector/graph can become exactly zero.
- `src/living_memory/storage.py:1297-1300` implements the storage-local scope
  family fallback used by `get_retrieval_weights`.

## Recall Event And Pending Feedback Storage

- `src/living_memory/storage.py:649-696` records each recall event, including
  query, scope, resolved scopes, ambient context, and serialized result score
  components.
- `src/living_memory/storage.py:729-765` returns recent pending recall events.
  It scans more rows than the requested limit, filters by match strength, allows
  at most one weak fallback, and stops at `limit`.
- `src/living_memory/storage.py:1434-1468` determines whether a pending event is
  compatible with a new trace. Same session or same task is immediately strong;
  same agent plus text match is also strong; text-only matches are weak.
- `src/living_memory/storage.py:1471-1477` computes the text overlap used by
  content-aware pending-event matching.

## Retrieval Scoring And Weight Reads

- `src/living_memory/scope.py:42-79` resolves a query into a `ScopePlan`.
- `src/living_memory/scope.py:97-119` normalizes scopes and derives scope
  families; projects search `(project:<name>, global)`, sessions search
  `(session:<id>, project, global)`.
- `src/living_memory/scope.py:146-182` builds the project/session/global plan
  that determines which scopes can read and later persist feedback.
- `src/living_memory/retrieval.py:108-174` is the public recall implementation:
  resolve scope, collect BM25/vector/schema/graph candidates, rank, log access,
  and persist a recall event.
- `src/living_memory/retrieval.py:283-300` collects FTS/BM25 candidates. It
  assigns synthetic BM25 score `1.0 / (rank + 1)`, so the top lexical result
  often has `bm25_score=1.0`.
- `src/living_memory/retrieval.py:302-348` collects vector candidates and
  writes lazy embeddings before scanning. Vector scores are cosine similarities
  above `VECTOR_MATCH_THRESHOLD`.
- `src/living_memory/retrieval.py:401-459` collects graph candidates by BFS
  from existing BM25/vector seeds.
- `src/living_memory/retrieval.py:210-228` reads normalized retrieval weights
  for each candidate node scope and computes the weighted base score.
- `src/living_memory/retrieval.py:214-223` applies a runtime graph floor when a
  candidate has graph evidence, preserving graph influence at ranking time even
  if persisted graph weight is low.
- `src/living_memory/retrieval.py:229-230` preserves strong vector matches by
  taking `max(base_score, candidate.vector_score)` when vector similarity is
  high.
- `src/living_memory/retrieval.py:243-244` adds a causal-mode graph boost after
  feedback weighting.
- `src/living_memory/retrieval.py:533-546` serializes recall result summaries,
  including BM25/vector/graph/trigger scores, into recall events. These fields
  feed later implicit feedback.

## Feedback Mutation Path

- `src/living_memory/feedback.py:15-21` defines adaptive learning-rate constants
  and `_DEFAULT_PENDING_RECALL_LIMIT = 5`.
- `src/living_memory/feedback.py:24-62` optionally retunes the learning rate
  when `LM_RETRIEVAL_TUNING_POLICY=adaptive`; fixed mode leaves the stored
  learning rate unchanged.
- `src/living_memory/feedback.py:106-133` applies explicit or synthetic
  retrieval feedback: update node usefulness, choose a target scope, compute
  method signals, and call `store.update_retrieval_weights`.
- `src/living_memory/feedback.py:257-274` converts one result into component
  signals. Useful feedback rewards the dominant component and applies
  `-0.25 * signal` to the other components. For a top exact result,
  `bm25_score=1.0` usually makes BM25 dominant.
- `src/living_memory/feedback.py:136-150` fetches pending recall events for a
  new trace, passing trace scope, context, content, and the default limit of 5.
- `src/living_memory/feedback.py:166-225` iterates every compatible pending
  event and every result in each event, creates implicit related edges, applies
  synthetic useful feedback, then marks the event as consumed.
- `src/living_memory/feedback.py:199-213` constructs the synthetic feedback
  result from the stored recall-event score components and calls
  `apply_retrieval_feedback` with `scope=event.scope`.
- `src/living_memory/feedback.py:206-212` scales the synthetic signal by rank:
  rank 1 receives 1.0, rank 2 receives 0.5, and lower ranks bottom out at 0.2.
  In the reproduction, `max_results=1` makes every consumed event a full-strength
  rank-1 reinforcement.

## MCP Public API Surface

- `src/living_memory/server.py:443-487` registers `memory_remember`. The write
  path appends the trace and then calls `apply_pending_recall_feedback` before
  returning `implicit_feedback` in the response.
- `src/living_memory/server.py:527-553` registers `memory_recall`, which calls
  `MemoryRecallService.memory_recall` and returns the recall event id and
  result score components.
- `src/living_memory/server.py:581-595` registers `memory_health`, which
  delegates to the health resource.
- `src/living_memory/server.py:598-617` registers resources. Only concepts,
  stats, and recent interactions are exposed as resources today; recall events
  and connection summaries are helper functions, not resources.

## Resource, Prompt, And Health Read Surfaces

- `src/living_memory/resources.py:70-81` exposes `retrieval_weights_summary` in
  `memory_stats`.
- `src/living_memory/resources.py:102-114` exposes the normalized
  `retrieval_policy` for a scope in `memory_status`.
- `src/living_memory/resources.py:308-326` lists raw rows from
  `retrieval_weights` without effective/floored weights or skew flags.
- `src/living_memory/resources.py:329-330` exposes only normalized effective
  weights for a single scope via `retrieval_policy_for_scope`.
- `src/living_memory/resources.py:333-495` builds `memory_health`; it includes
  activity, dedup, staleness, the current retrieval policy, and audit sections,
  but no retrieval-feedback skew diagnosis today.
- `src/living_memory/resources.py:498-540` has a recall-events summary helper
  with feedback-applied counts, but it is not registered by `server.py`.
- `src/living_memory/prompts.py:39-91` builds active memory context and delegates
  selection to recall/context helpers.
- `src/living_memory/prompts.py:112-120` uses `MemoryRecallService` for concept
  selection, so the prompt path reads the same retrieval weights as recall.
- `src/living_memory/prompts.py:133-153` separately scores fallback concepts
  with `store.get_retrieval_weights(item_scope).normalized()`.
- `src/living_memory/prompts.py:266-277` prints the normalized retrieval policy
  in the generated context block.
- `src/living_memory/prompts.py:325-356` uses the normalized BM25/vector/graph
  weights inside `_policy_score`.
- `src/living_memory/health_audit.py:127-150` reports recall-event feedback
  application ratios, but not the direction of feedback pressure or component
  skew.
- `src/living_memory/health_audit.py:492-508` provides read-only retrieval
  weight fallback for audit stores when DB rows are absent.

## Discovery Conclusions For The Design Child

- The persisted mutation source of truth is
  `src/living_memory/storage.py:992-1015`, reached from
  `src/living_memory/feedback.py:127-132`.
- The 06f6422-style amplification path is public and end to end:
  `memory_recall` persists score components, `memory_remember` consumes up to
  five compatible pending events, and each stored top result can call
  `apply_retrieval_feedback`.
- Ranking-time graph boosts already exist at
  `src/living_memory/retrieval.py:214-223` and `src/living_memory/retrieval.py:243-244`.
  A source-of-truth fix should preserve these boosts.
- Existing resource/health surfaces show raw or normalized weights but do not
  identify scopes near BM25 monoculture, near-zero vector, raw-vs-effective
  differences, or recent component pressure.
