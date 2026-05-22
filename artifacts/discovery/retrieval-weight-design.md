# Retrieval Weight Design Decision

Scope: design selection for `retrieval-feedback-stability/design-decision`.
This artifact is the gate between `reproduce-and-inventory` (done) and the
parallel build children (`implement-feedback-fix`, `health-skew-visibility`)
plus the final `integrate-and-verify` child. It is evidence-driven (cites
file:line locations from `artifacts/discovery/retrieval-weight-flow.md` and
the reproduction test at `tests/test_retrieval_feedback_amplification.py`)
and resolves every blocking item raised in the parent SPEC critique.

## Evidence Summary

Live persisted retrieval weights (normalized, from parent SPEC text):

| scope          | bm25  | vector | graph |
|----------------|------:|-------:|------:|
| project:online | 0.995 |  0.005 | 0.000 |
| project:lm     | 0.970 |  0.030 | 0.000 |
| global         | 0.949 |  0.017 | 0.000 |

Mechanism, traced through the public API (see
`artifacts/discovery/retrieval-weight-flow.md`):

1. `memory_recall` (`src/living_memory/server.py:527-553`) persists per-result
   score components into a recall event
   (`src/living_memory/storage.py:649-696`, `src/living_memory/retrieval.py:533-546`).
   BM25 candidates carry a synthetic `bm25_score=1.0` for rank 1
   (`src/living_memory/retrieval.py:283-300`, line 298).
2. `memory_remember` (`src/living_memory/server.py:443-487`) calls
   `apply_pending_recall_feedback` (`src/living_memory/feedback.py:136-225`).
   Since `06f6422`, the default pending-event limit is
   `_DEFAULT_PENDING_RECALL_LIMIT = 5`
   (`src/living_memory/feedback.py:21,140`), so one write can consume up to
   five compatible recall events.
3. For every result in every consumed event, the loop builds a synthetic
   feedback result and calls `apply_retrieval_feedback`
   (`src/living_memory/feedback.py:199-213`).
4. `_method_signals` (`src/living_memory/feedback.py:257-274`) picks the
   dominant component (BM25 in lexical hits) and emits
   `signals = {dominant: +signal, others: -0.25*signal}` for useful feedback.
5. `update_retrieval_weights` (`src/living_memory/storage.py:992-1015`)
   applies `max(0.0, current + learning_rate * signal)` per component,
   normalizes, and persists via `set_retrieval_weights`. Once vector hits
   0.0, subsequent BM25-dominant useful feedback keeps it pinned at 0.0
   (`max(0, 0 + 0.05 * -0.25) = 0`) and `normalized()` returns pure BM25
   (`src/living_memory/models.py:156-167`).

The reproduction at `tests/test_retrieval_feedback_amplification.py:46-114`
uses the public MCP API to drive four cycles of 5 recalls + 1 remember on
`project:collapse`, consumes 5 pending recall events per write, and ends
asserting `after.bm25 <= 0.85` and `after.vector >= 0.15`. Current code
fails both; the file is marked `xfail(strict=True)` and the strict mark
flips to XPASS after the fix.

## Selected Design

**Design A — Per-family policy floors at the persisted mutation site, with
write-time auto-recovery and an explicit audited reseat path.**

The persisted-weight source of truth is
`MemoryStore.update_retrieval_weights`
(`src/living_memory/storage.py:992-1015`). Today it clamps each component
with `max(0.0, …)` and then `normalized()`. The fix replaces the
unbounded simplex with per-family policy bounds (named in
`RetrievalPolicyFloors`), applied to the *normalized* tuple after the
existing math. Deficient channels are lifted by donating mass from BM25
first; if BM25 has no surplus, mass comes from any other channel above
its own floor. This preserves the BM25-collapse fix while still making
the policy invariant true for non-BM25-dominant edge cases.

### Floor enforcement algorithm (normalized space)

```text
# pseudocode for src/living_memory/storage.py:update_retrieval_weights
raw_b = max(0.0, current.bm25   + lr * bm25_signal)
raw_v = max(0.0, current.vector + lr * vector_signal)
raw_g = max(0.0, current.graph  + lr * graph_signal)
n     = RetrievalWeights(raw_b, raw_v, raw_g).normalized()   # n.bm25 + n.vector + n.graph = 1.0

floors = config.retrieval_policy_floors.for_family(_scope_family(scope))
weights = {"bm25": n.bm25, "vector": n.vector, "graph": n.graph}
minimum = {"bm25": 0.0, "vector": floors.vector_min, "graph": floors.graph_min}

def lift(target):
    deficit = max(0.0, minimum[target] - weights[target])
    # Prefer to take from BM25. If the edge case is graph/vector-dominant,
    # donate from any remaining channel above its own floor.
    for donor in ("bm25", "graph", "vector"):
        if donor == target or deficit <= 1e-12:
            continue
        available = max(0.0, weights[donor] - minimum[donor])
        taken = min(deficit, available)
        weights[donor] -= taken
        weights[target] += taken
        deficit -= taken

lift("vector")
lift("graph")

if weights["bm25"] > floors.bm25_max:
    surplus = weights["bm25"] - floors.bm25_max
    weights["bm25"] -= surplus
    weights["vector"] += surplus

return RetrievalWeights(scope, weights["bm25"], weights["vector"], weights["graph"], ...).set()
```

Because every configured floor set satisfies
`bm25_max + vector_min + graph_min <= 1.0`, the transfer always has
enough surplus inside the normalized tuple. The common collapse path
uses BM25 as the only donor; the fallback donor loop is just to keep the
invariant true if future feedback makes graph or vector the dominant
channel while another secondary channel is below its floor.

### Per-family floors (with evidence)

Defaults remain at the values defined in
`src/living_memory/config.py:21-26`. Floors are roughly half of each
family's default vector share, leaving feedback room to move weights but
preventing semantic recall from being disabled.

| family   | default (b/v/g)       | floor: bm25_max | vector_min | graph_min | rationale                                                            |
|----------|-----------------------|----------------:|-----------:|----------:|----------------------------------------------------------------------|
| project  | 0.70 / 0.30 / 0.00    |            0.85 |       0.15 |      0.00 | matches parent SPEC: project bm25<=0.85, vector>=0.15. Graph default is 0, no graph-recall regression expected. |
| global   | 0.40 / 0.40 / 0.20    |            0.75 |       0.20 |      0.05 | respects parent SPEC vector>=0.20 and keeps a small global graph floor because global's default graph share is 0.20. BM25 can still learn far above the 0.40 default; exact lexical tests cover the precision risk. Ranking-time graph clamp at `retrieval.py:215` (`minimum_graph=0.25` / 0.75 causal) still dominates when graph evidence exists. |
| session  | 0.80 / 0.20 / 0.00    |            0.90 |       0.10 |      0.00 | matches parent SPEC: session bm25<=0.90, vector>=0.10.               |
| default  | 1.00 / 0.00 / 0.00    |             — |        — |       — | no floor enforced. `default` is the BM25-only baseline used as final fallback by `_scope_family` (`src/living_memory/storage.py:955-964`) when neither the exact scope nor its family is in `retrieval_weights`. Forcing vector mass on `default` would change semantics for scopes that genuinely have no embeddings. |

Floor values are **policy**, not heuristics: they are stored in
`DEFAULT_RETRIEVAL_POLICY_FLOORS` in `src/living_memory/config.py` and
overridable via TOML `[retrieval_policy_floors.<family>]`.

The "embeddings exist" condition in the parent SPEC ("vector recall when
embeddings exist") is handled by the system, not the floor: if a scope
has no embedded nodes, `_collect_vector`
(`src/living_memory/retrieval.py:302-348`) returns no candidates and the
non-zero vector weight is multiplied by zero. The floor remains universal
because it is harmless on empty-vector scopes and avoids per-scope
branching in the hot path. The reproduction test covers the embedded
case; an additional balanced-no-false-positive check in
`tests/test_health_retrieval_skew.py` exercises the unembedded case via a
balanced scope fixture.

### Recovery strategy for already-collapsed prod weights

Two layers, both required to satisfy parent non-goal "Do not silently
mutate production SQLite weights as the only fix. Any maintenance/reset
path must be explicit, tested, and audited":

1. **Primary — auto-recovery on next feedback event.** The next call to
   `update_retrieval_weights(scope=…)` for any collapsed scope writes a
   floored tuple. For `project:online` (0.995/0.005/0.000), the next
   useful BM25-dominant event normalizes to roughly the same skew, then
   the floor lifts vector to 0.15 (taking 0.145 from BM25). This is the
   *exact* path the regression test exercises — the test ends in a
   floored state. There is no global state change, no batch migration:
   each scope heals organically as it is used.

2. **Secondary — explicit, audited reseat path.** A new method
   `MemoryStore.reseat_retrieval_floors() -> list[ScopeFloorChange]` walks
   `retrieval_weights`, computes the floor-corrected tuple for each row,
   writes back only the rows that violate their family's floor, and
   returns a structured audit list (scope, before, after, floor). The
   method is invoked by a new admin CLI subcommand
   `python -m living_memory.maintenance reseat-floors [--dry-run | --apply]`
   that prints the audit list. It is **not** called from any constructor
   or hot path. This satisfies "explicit, tested, audited":
   - **Explicit:** the caller invokes the subcommand; the store does not
     mutate on its own at boot.
   - **Tested:** a new test in `tests/test_storage.py` (extension; the
     `implement-feedback-fix` child owns it) seeds a deliberately-collapsed
     `project:fixture` row (bm25=1.0, vector=0.0, graph=0.0), runs
     `reseat_retrieval_floors()`, and asserts the returned audit list plus
     the persisted tuple match the project-family floor.
   - **Audited:** the returned `ScopeFloorChange` rows are printed in
     before/after form by the subcommand and can be redirected to a log
     for ops review.

The regression test covers (1) end-to-end through the public MCP API; the
storage test covers (2) as an explicit isolated operation. Operators can
choose to apply (2) immediately to heal `project:online`/`project:lm`/
`global` or leave them to heal organically via (1) on their next write.

### Defense-in-depth at the read site

The existing ranking-time graph floor at
`src/living_memory/retrieval.py:214-223` and the STRONG_VECTOR_MATCH
override at `src/living_memory/retrieval.py:229-230` remain unchanged.
They are not load-bearing for the fix — the persisted floor is — but
they protect graph and high-similarity vector evidence even when
persisted weights briefly dip toward their floor between feedback events.
The build child must not weaken or extend these blocks.

The causal graph boost at `src/living_memory/retrieval.py:243-244` (1.5×
multiplier) is also preserved verbatim.

## Rejected Alternatives

### B — Non-destructive useful-feedback signal math (zero penalty on losers)

Change `_method_signals` (`src/living_memory/feedback.py:257-274`) so
useful feedback emits `{dominant: +signal, others: 0.0}` instead of
`{dominant: +signal, others: -0.25*signal}`. The signal stops actively
pushing vector/graph toward zero.

**Rejected because:**
- Does not establish a lower bound; vector can still drift to zero via
  any `useful=False` feedback or via floating-point dilution under
  repeated normalization.
- Does not auto-recover already-collapsed scopes
  (`project:online` at 0.005 stays at 0.005 until positive vector signals
  arrive — which require vector results to win, which require non-zero
  vector weight, a circular precondition once near zero).
- Loses the deliberate "teach the system which method won" semantic
  (`apply_retrieval_feedback` docstring at
  `src/living_memory/feedback.py:114`: "nudge weights toward the winning
  method"). Zeroing the loser penalty also weakens legitimate signal.

B is partially complementary to A — design A could be combined with a
softer `-0.10 * signal` loser penalty — but the floor in A makes the
exact loser-penalty constant non-load-bearing. We keep `_method_signals`
unchanged.

### C — Score calibration before signal derivation

Normalize `bm25_score`/`vector_score`/`graph_score` against per-scope
baseline distributions (e.g. z-score over a rolling window) before
selecting the dominant method. The synthetic `1.0 / (rank + 1)` BM25
score at `src/living_memory/retrieval.py:298` would no longer
automatically dominate.

**Rejected because:**
- Requires new infrastructure: per-scope rolling-distribution storage and
  bootstrap state (how do you calibrate the first 100 recalls?).
- Discovery-dependent: would need additional measurement on
  `project:online`/`project:lm`/`global` to choose distribution
  parameters per family, contradicting the parent's "simplest fundamental
  fix … prefer small universal changes over per-project tuning."
- Solves a less precise root cause: the score components themselves are
  fine (BM25=1.0 *is* the strongest rank-1 BM25 signal). The defect is
  the persistence math, not the score derivation. A fix at the
  derivation layer adds machinery without removing the failure mode.

### D — Ranking-time-only clamp on read

Apply the per-family floor only in `MemoryRecallService._weighted_results`
(`src/living_memory/retrieval.py:200-271`) when reading
`store.get_retrieval_weights(node.scope).normalized()`, leaving persisted
state untouched.

**Rejected explicitly by parent non-goal:** "do not solve this only with
a cosmetic ranking-time clamp if persisted feedback continues to collapse
channels." Persisted state would still drift to 1.0/0.0/0.0; the
`memory_health` retrieval-policy field at
`src/living_memory/resources.py:472` would keep reporting the collapsed
weights, the audit history would still show monoculture, and any
future call site that reads raw weights would see the broken value. The
read-site clamp is retained only as the pre-existing defense-in-depth at
`retrieval.py:214-223` for graph; we do not extend it to vector.

## Health-Skew Metric Surface

Owner: `health-skew-visibility` child.

A new `retrieval_skew` block is added to the `memory_health` report
returned by `src/living_memory/resources.py:memory_health` (currently
ends at line 495). The block is additive — no existing field changes —
and the block is built by a standalone helper
`retrieval_skew_metrics(store, *, scope=None)` in
`src/living_memory/resources.py`, so the `health-skew-visibility` child
does not need ownership of unrelated modules.

### Block shape (JSON-equivalent)

```json
"retrieval_skew": {
  "thresholds": {
    "bm25_monoculture": 0.90,
    "near_zero_vector": 0.10,
    "near_zero_graph":  0.05
  },
  "floors_in_effect": {
    "project": {"bm25_max": 0.85, "vector_min": 0.15, "graph_min": 0.00},
    "global":  {"bm25_max": 0.75, "vector_min": 0.20, "graph_min": 0.05},
    "session": {"bm25_max": 0.90, "vector_min": 0.10, "graph_min": 0.00}
  },
  "scopes_at_risk": [
    {
      "scope": "project:online",
      "family": "project",
      "raw": {"bm25": 0.995, "vector": 0.005, "graph": 0.000},
      "effective_after_floor": {"bm25": 0.850, "vector": 0.150, "graph": 0.000},
      "flags": ["bm25_monoculture", "near_zero_vector"],
      "floor_violation": {
        "vector": {"current": 0.005, "floor": 0.15}
      }
    }
  ],
  "recent_feedback_pressure": {
    "window_hours": 24,
    "consumed_recall_events": 47,
    "dominant_method_counts": {"bm25": 42, "vector": 4, "graph": 1},
    "bm25_dominance_ratio": 0.894
  }
}
```

### Field definitions

- `thresholds`: configured via
  `MemoryConfig.retrieval_skew_thresholds` (new field; see config.py
  partitioning below). Threshold flags describe severe skew; policy
  `floor_violation` entries are computed separately so the health
  surface can report a raw row that violates a floor even if it does not
  cross a severe-skew threshold.
- `floors_in_effect`: echo of `MemoryConfig.retrieval_policy_floors` so
  operators can confirm config without reading the source.
- `scopes_at_risk`: rows from `retrieval_weights` whose normalized
  values trip any threshold flag or violate a policy floor. Each row
  carries both the raw normalized triple and the floor-corrected preview
  (`effective_after_floor`); there is no separate read-site effective
  policy in design A, so this preview is the maintenance/audit answer
  rather than a hidden runtime clamp. `near_zero_graph` is evaluated only
  for families with a configured `graph_min > 0`, so ordinary project
  and session rows with graph default 0.0 are not false positives.
- `recent_feedback_pressure`: derived from `recall_events` (last
  `window_hours`) by parsing the stored `results` JSON payload written by
  `src/living_memory/storage.py:649-696`.
  The dominant method per consumed result is counted, and
  `bm25_dominance_ratio = bm25 / sum(counts)` exposes the pressure
  driving the floor enforcement. Computed only when feasible —
  `null`/omitted under the same condition the existing
  `feedback_metrics` helper uses for empty windows.

Skew thresholds (0.90, 0.10, 0.05) are configurable. The defaults fire
on `project:online` (0.995 > 0.90 and 0.005 < 0.10) and on `global`
(0.949 > 0.90, 0.017 < 0.10, and graph below the configured global
graph floor), but do not fire on a balanced project at 0.65/0.30/0.05.

### Test surface (owned by `tests/test_health_retrieval_skew.py`)

- **Detection:** seed `project:lopsided` at bm25=0.97/vector=0.02/graph=0.01,
  then expect `memory_health` to flag it with
  `["bm25_monoculture", "near_zero_vector"]` and a floor_violation entry
  for vector.
- **No false positive on balanced scope:** seed `project:balanced` at
  0.65/0.30/0.05, expect no entry in `scopes_at_risk` and no flags.
- **Floors echo:** verify `floors_in_effect` reflects the configured
  `RetrievalPolicyFloors`.
- **Empty-window pressure tolerates absence:** with no recall events in
  the window, `recent_feedback_pressure.bm25_dominance_ratio` is `null`
  and the block does not crash.

## Config.py Partitioning (resolves critique item 1)

Both build children edit `src/living_memory/config.py`. To avoid silent
collision, the file is partitioned by named symbols. Neither child may
edit symbols owned by the other.

### Owned by `implement-feedback-fix`

```python
@dataclass(frozen=True, slots=True)
class RetrievalPolicyFloors:
    bm25_max: float
    vector_min: float
    graph_min: float

DEFAULT_RETRIEVAL_POLICY_FLOORS: dict[str, RetrievalPolicyFloors] = {
    "project": RetrievalPolicyFloors(bm25_max=0.85, vector_min=0.15, graph_min=0.00),
    "global":  RetrievalPolicyFloors(bm25_max=0.75, vector_min=0.20, graph_min=0.05),
    "session": RetrievalPolicyFloors(bm25_max=0.90, vector_min=0.10, graph_min=0.00),
}
```

Plus `MemoryConfig.retrieval_policy_floors: dict[str, RetrievalPolicyFloors]`
(new field, defaulted to the constant) and a `_table(raw,
"retrieval_policy_floors")` overlay in `load_config`.

### Owned by `health-skew-visibility`

```python
@dataclass(frozen=True, slots=True)
class RetrievalSkewThresholds:
    bm25_monoculture: float
    near_zero_vector: float
    near_zero_graph: float

DEFAULT_RETRIEVAL_SKEW_THRESHOLDS = RetrievalSkewThresholds(
    bm25_monoculture=0.90,
    near_zero_vector=0.10,
    near_zero_graph=0.05,
)
```

Plus `MemoryConfig.retrieval_skew_thresholds: RetrievalSkewThresholds`
(new field, defaulted to the constant) and a `_table(raw,
"retrieval_skew_thresholds")` overlay in `load_config`.

The two children can land in either order because their symbols are
disjoint and neither dataclass references the other. `MemoryConfig`
ordering of the two new fields is a textual convention only; the
`integrate-and-verify` child will assert both are present.

## Test File Paths (resolves critique item 3)

The reproduction artifact already landed the regression test at
`tests/test_retrieval_feedback_amplification.py` (repo-root `tests/`).
The decomposition's earlier references to
`src/living_memory/tests/test_retrieval_feedback_amplification.py` are
**incorrect** — there is no `src/living_memory/tests/` directory.

All behaviour and health tests for this subtree live under repo-root
`tests/`:

- `tests/test_retrieval_feedback_amplification.py` — existing; flips
  from xfail-strict to passing after the floor is implemented.
- `tests/test_retrieval_lexical_recall.py` — **new**, owned by
  `implement-feedback-fix`.
- `tests/test_retrieval_semantic_recall.py` — **new**, owned by
  `implement-feedback-fix`.
- `tests/test_graph_recall.py` — existing, **extended** by
  `implement-feedback-fix` with a graph-floor preservation assertion.
- `tests/test_health_retrieval_skew.py` — **new**, owned by
  `health-skew-visibility`.
- `tests/test_storage.py` — existing, **extended** by
  `implement-feedback-fix` with the `reseat_retrieval_floors` audit test.

## Behaviour Tests `integrate-and-verify` Will Assert (resolves critique item 4)

Exactly four behaviour test files, plus one health test, plus the
existing `npm run check` umbrella:

1. `tests/test_retrieval_feedback_amplification.py` — amplification
   collapse cannot disable vector for `project:collapse`; the
   xfail-strict mark is removed and the assertions at
   `tests/test_retrieval_feedback_amplification.py:112-114` pass.
2. `tests/test_retrieval_lexical_recall.py` (new) — exact lexical
   queries — file paths, ticket keys, hashes, command tokens — still
   surface their BM25 top hit with `bm25_score == 1.0` after a floor is
   in effect. Confirms the per-family `bm25_max` cap of 0.85 does **not**
   regress lexical precision (the cap reduces a normalized weight, not
   the underlying BM25 score; ranking still places the lexical winner
   first).
3. `tests/test_retrieval_semantic_recall.py` (new) — a paraphrase /
   synonym query that has no lexical overlap still returns the
   semantically matching trace, both from a freshly-seeded scope (floor
   not yet applied) and from a previously-collapsed scope after one
   feedback event has triggered floor recovery.
4. `tests/test_graph_recall.py` (existing, extended) — causal and depth
   graph recall continues to work with the new persisted floors, and
   the ranking-time graph floor at
   `src/living_memory/retrieval.py:214-223` is not removed or weakened.

The health test:

5. `tests/test_health_retrieval_skew.py` (new) — detection, no-false-positive,
   floors echo, empty-window pressure tolerance (described above).

`integrate-and-verify` runs `npm run check` (the umbrella) plus an
explicit `pytest tests/test_retrieval_feedback_amplification.py
tests/test_retrieval_lexical_recall.py tests/test_retrieval_semantic_recall.py
tests/test_graph_recall.py tests/test_health_retrieval_skew.py -q` to keep
the gate explicit and not lose any of the five behind a global suite
selector. No existing test may be weakened to satisfy the new behaviour;
the integration child greps for `xfail`, `pytest.skip`, and assertion
deletions on the listed files and `tests/test_feedback_weights.py` /
`tests/test_recall_feedback_loop.py`.

## Invariants the Build Children Must Preserve

- The persisted-mutation source of truth remains
  `MemoryStore.update_retrieval_weights`; no parallel persistence path
  may be introduced.
- `RetrievalWeights.normalized()` (`src/living_memory/models.py:156-167`)
  is unchanged; the floor is applied **after** the existing normalize
  step, by `update_retrieval_weights` itself.
- `apply_retrieval_feedback` and `_method_signals` semantics (reward the
  dominant component, penalize the others by `-0.25 * signal`) are
  unchanged. The fix is a **lower-bound constraint on persistence**,
  not a change in signal derivation.
- The seeding path at `src/living_memory/storage.py:1246-1264`
  (`ON CONFLICT DO NOTHING`) is unchanged. Defaults still do not
  overwrite existing rows.
- Existing graph runtime boosts at
  `src/living_memory/retrieval.py:214-223` and
  `src/living_memory/retrieval.py:243-244`, and the strong-vector
  override at `src/living_memory/retrieval.py:229-230`, are unchanged.
- No new constructor-time mutation of `retrieval_weights`. The reseat
  path is opt-in and invoked only by the new admin subcommand.

## Open Items Deferred to Build

These are intentionally not specified here; they are implementation
choices the build child resolves:

- Exact dataclass file ordering inside `src/living_memory/config.py`
  (top-of-file vs. bottom-of-file). The partition is by symbol, not by
  line range.
- The internal helper signature of `reseat_retrieval_floors` (returning
  `list[ScopeFloorChange]` is suggested; the return-type structure is
  free as long as the admin subcommand can render a before/after audit
  table).
- Exact helper factoring inside `src/living_memory/resources.py` for
  `retrieval_skew_metrics`. It should stay in that file unless the
  health child explicitly expands ownership, because the current sibling
  interface owns `resources.py`, `config.py`, and the health test only.
