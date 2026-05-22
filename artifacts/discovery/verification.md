# Discovery Artifact Verification

Verification record for the `discover-baseline-and-root-causes` subtree
covering the parent SPEC P1–P10. Read-only audit; no production code
touched here.

## 0. Scope contract

This file is an artifact-only audit and stores no executable steps.

- intent: confirm artifact presence, required sections, and
  artifacts-only blast radius for the discovery subtree.
- Local replay_evidence_path: artifacts/discovery/verification.md
  (this file). All checks are reproducible against the local git tree
  and on-disk artifacts in this worktree.
- Local rollback_artifact: none required; the verification adds one
  new artifact and reverts nothing.
- redaction_boundary: this file references only repository file paths,
  git commit shas, and section names already present in shared
  artifacts; no secret material is included.

## 1. Verification approach (local-only)

All commands run inside the worktree
`/home/sfx/p/lm/.worktrees/_node_exec_memory-quality-root-fixes_discover-baseline-and-root-causes_discovery-artifact-verify`
against its own git index and filesystem. No external lookups, no
network fetches.

The merge base used for "subtree-introduced changes" is
`git merge-base HEAD master` → `cef85cf6d6f4aa0b25edb276f15a1c82dda6436a`.
Anything on this branch after that commit is owned by this subtree.

## 2. Artifact presence

All required deliverables from the parent SPEC are present in the
worktree with non-zero size.

| Target | Path | Lines | Bytes |
|---|---|---|---|
| P2  | `artifacts/baseline.md` | 285 | 15530 |
| P2  | `artifacts/discovery/baseline-raw.sql` | 281 | 12091 |
| P2  | `artifacts/discovery/baseline-raw.metrics.json` | 148 | 10430 |
| P1  | `artifacts/discovery/lm-recall-context.md` | 71 | 8943 |
| P3  | `artifacts/discovery/path-inventory.md` | 535 | 46719 |
| P4  | `artifacts/discovery/scope-hygiene.md` | 194 | 27546 |
| P5  | `artifacts/discovery/dedup-noise.md` | 417 | 45920 |
| P6  | `artifacts/discovery/feedback-linkage.md` | 126 | 10571 |
| P7  | `artifacts/discovery/health-observability.md` | 182 | 15473 |
| P8  | `artifacts/discovery/recall-speed-usefulness.md` | 318 | 24294 |
| P9  | `artifacts/ANALYSIS.md` | 710 | 71656 |
| P10 | `artifacts/discovery/verification.md` | (this) | (this) |

No required artifact is missing. No artifact is a placeholder/stub:
every file's section grep returned a substantive multi-section outline.

## 3. Required-section coverage by SPEC item

Section checks were done with header greps (`^#{1,3} `) on each file.

### P1 — `artifacts/discovery/lm-recall-context.md`

Contains: `Recall Events`, `Useful Facts` →
{`Task And Tool Surface`, `Scope Leakage Signals`,
`Duplicate And File-Chunk Noise Signals`, `Feedback Linkage Signals`,
`Health, Observability, And Latency Signals`},
`Noise To Treat Carefully`, `Inherited Gate State (Pre-Grep Note)`.

Covers the parent contract's required recall queries (scope leakage,
rise/breakthrough/ocpa-generative-action-substrate-v1, file-chunk
dedup, feedback_applied recall_event linkage, memory_health metrics)
and records the recall_event ids that downstream children depend on.

### P2 — `artifacts/baseline.md` + raw artifacts

`baseline.md` sections include `External Action Boundaries`
(`Intent`, `Local replay_evidence`, `Local rollback_artifact`,
`redaction_boundary`), `Source And Read-Only Fixture`,
`memory_health` (in-process), `SQL Metrics` →
{`Duplicate Density`, `feedback_applied Ratio`, `Never-Accessed Ratio`,
`Scope Leakage Candidates`}, `Hot Recall Latency`,
`Reproduction Order`.

Raw companions present: `baseline-raw.sql` (281 lines) and
`baseline-raw.metrics.json` (148 lines). The SPEC's required
metrics (duplicate density, feedback_applied ratio, scope leakage
candidates incl. `rise/*`, `breakthrough/*`,
`ocpa-generative-action-substrate-v1`, never-accessed ratio, DB
size, hot recall latency baseline) all map to dedicated sections
in `baseline.md`.

### P3 — `artifacts/discovery/path-inventory.md`

Thirteen top-level sections cover MCP server surface, storage,
scope resolver, retrieval, feedback, consolidation, decay,
resources/health/prompts, AE `lm_client.py`, tests, scripts,
per-problem data location, and downstream-colour notes. Spot-check
of file:line citations confirms specific anchors
(e.g. `scope.py:97-113`, `scope.py:185-229`, `storage.py:215-223`,
`feedback.py:135-234`, `resources.py:323-459`,
`lm_client.py:321-556`, `retrieval.py:473-479`,
`retrieval.py:302-348`+`371-399`). All required surfaces in the
parent SPEC are represented.

### P4 — `artifacts/discovery/scope-hygiene.md`

Sections: `Scope contract`, `Observed problem (from baseline)`,
`Root cause` (with `The asymmetry in code`,
`Why this manifests as Octopus-affiliated leakage specifically`,
`What is not the root cause`), `Chosen simplest central fix`
(with `Why this is the right shape`, `Numeric target`),
`Rejected alternatives` (seven sub-options),
`AE-vs-LM layer decision`, `Downstream file ownership`,
`Deliberately deferred follow-ups`,
`Acceptance signal for the downstream fix node`.

Covers parent contract: root cause, chosen central fix, rejected
alternatives, AE-vs-LM layer decision, downstream file ownership.

### P5 — `artifacts/discovery/dedup-noise.md`

Sections include `Root cause (precise)`, `Evidence chain (citations)`
(live duplicate signal, write path, caller path, dedup-adjacent
surfaces, append-only invariants), `Chosen fix (simplest central
rule)` (new column + index, dedup at `_insert_node`, reuse
`supersedes` semantics, health surface enrichment),
`Rejected alternatives`, `Layer decision: LM-internal`,
`Numeric improvement target`, `Raw-history preservation rule`,
`Downstream file ownership`, `Latency micro-check guidance`,
`Risks and open follow-ups`, `Expected test surface`,
`Summary`.

Covers parent contract: root cause + citations, append-only-safe
dedup/supersede strategy, rejected alternatives, numeric target,
raw-history preservation rule, downstream file ownership.

### P6 — `artifacts/discovery/feedback-linkage.md`

Sections: `Boundary Contract`, `Baseline Signal`, `Root Cause`,
`Temp-DB Reproductions`, `Chosen Fix`, `Numeric Target`,
`Rejected Alternatives`, `Client Gap`, `Tests To Add`,
`Latency And DB Policy`.

Covers parent contract: root cause + citations, simplest robust
recall→remember/teach linkage fix, rejected alternatives, numeric
improvement target, metadata/scope/session behavior, negative-case
constraints (covered under `Client Gap` and `Rejected Alternatives`).

### P7 — `artifacts/discovery/health-observability.md`

Sections: `Scope contract`, `Scope`, `Current Gap`,
`Baseline-Aligned Metric Definitions` (`Duplicate Density`,
`Feedback Applied Ratio`, `Never-Accessed Ratio`,
`Scope Leakage Candidates`, `DB Size`,
`Instructions And Test Contract`, `Hot Recall Latency Baseline`),
`Chosen Extension Surface`, `Rejected Alternatives`,
`Downstream Ownership`, `Verification Guidance`.

Covers parent contract: root cause, smallest extension surface,
rejected alternatives, metric definitions that match those in
`baseline.md`.

### P8 — `artifacts/discovery/recall-speed-usefulness.md`

Sections: `Scope contract`, `Baseline facts`,
`Current retrieval path`, `Root causes`,
`Chosen latency-safe guidance`, `Rejected alternatives`,
`Deterministic micro-benchmark policy`,
`Per-fix benchmark instructions` (scope hygiene, dedup/file-chunk
noise, feedback linkage, health/observability, retrieval/ranking/
index changes), `Downstream policy`.

Covers parent contract: root causes + citations, latency-safe
guidance, rejected alternatives, deterministic per-fix benchmark
instructions for each of the four downstream fix nodes.

### P9 — `artifacts/ANALYSIS.md`

Sections (13 top-level): `Scope contract`, `Executive synthesis`
(problem map, single integrating insight, live-baseline anchor),
`Cross-cutting decisions` (AE-vs-LM layer, fixture vs live-DB
policy, append-only invariants, gate considerations),
`Problem 1 — Scope hygiene`, `Problem 2 — Duplicate / file-chunk
noise`, `Problem 3 — Feedback linkage`,
`Problem 4 — Health and observability`,
`Problem 5 — Recall speed and usefulness`,
`Integration sequencing for the four downstream fix nodes`
(DAG, sequencing rules, file-ownership split, integrate order),
`Per-fix latency micro-check spec`,
`Numeric improvement targets`, `Deliberately deferred follow-ups`,
`Mapping to parent SPEC`, `Summary table for downstream consumers`.

Each per-problem section includes Root cause, Chosen simplest central
fix, Rejected alternatives, AE-vs-LM layer, Downstream file ownership,
Numeric target, Fixture/live-DB policy, and Latency micro-check.
This is sufficient for the four downstream fix nodes to execute
without re-discovering fundamentals.

## 4. P10 — discovery subtree blast radius

P10 requires that this discovery subtree modifies only artifact files
and makes no production code changes.

### 4.1 Subtree commits (master..HEAD)

```
2907db6  artifacts/ANALYSIS.md
a4c2571  artifacts/discovery/health-observability.md
724a893  artifacts/discovery/dedup-noise.md
7a55bd4  artifacts/discovery/recall-speed-usefulness.md
e17d7ca  artifacts/discovery/scope-hygiene.md
e74eab6  artifacts/discovery/feedback-linkage.md
378ef1c  artifacts/baseline.md
         artifacts/discovery/baseline-raw.metrics.json
         artifacts/discovery/baseline-raw.sql
2a29578  artifacts/discovery/path-inventory.md
08626ce  artifacts/discovery/lm-recall-context.md
```

Every one of the nine branch commits touches a path under
`artifacts/`. Reproduce:

```
git log --name-only master..HEAD
git log --oneline master..HEAD -- ':!artifacts'
```

The second command returns no output, confirming no subtree commit
modified anything outside `artifacts/`.

### 4.2 Worktree dirty-file note (per parent goal contract)

`git status --porcelain` returns zero lines. The worktree had no
unrelated dirty files at the start of this verification node and
has none now. Nothing was reverted; nothing needed to be.

### 4.3 Why `git diff --stat master..HEAD` shows non-artifact rows

`git diff --stat master..HEAD` lists these non-artifact rows:

```
final_verdict.json                     |  71 ---
living_memory_trace_project_lm.json    |  14 -
result.md                              |  96 ----
tests/test_instructions_imperative.py  |  78 +--
src/living_memory/server.py            |  10 -    (only when comparing past merge base)
```

These are NOT changes introduced by this subtree. They reflect
master moving forward after the branch diverged at `cef85cf`:

- `final_verdict.json`, `living_memory_trace_project_lm.json`,
  `result.md` exist on master but not on HEAD because they were
  added on master after the branch's merge base
  (`git ls-tree --name-only master -- <file>` returns the file;
  `git ls-tree --name-only HEAD -- <file>` returns nothing;
  `git log --oneline master..HEAD -- <file>` returns nothing,
  confirming the branch never touched them).
- `tests/test_instructions_imperative.py` was edited on master in
  commit `7dc421e` ("Fix the LM test contract for MCP server
  instructions without changing the instructions themselves"),
  which is on master but not on this branch
  (`git log --oneline master..HEAD -- tests/test_instructions_imperative.py`
  returns nothing). This is the parallel orphan fix recorded in
  Living Memory trace `01KS79ZBKFMKWXZJYSVK3CWV4C` — the discovery
  subtree honoured P10 by declining to patch that test from a
  discovery node and instead recording the inheritance in
  `artifacts/discovery/lm-recall-context.md` §`Inherited Gate State`.
- `src/living_memory/server.py` only shows a 10-line delta when
  the diff window extends past the merge base (e.g., `HEAD~10..HEAD`
  reaches commits `cef85cf`, `c0d2c60`, `b86d0fa`, `3ff930d` which
  edited instructions on master before the branch existed).
  `git log --oneline master..HEAD -- src/living_memory/server.py`
  returns no output, confirming the branch made zero edits to
  `server.py`.

P10 holds. The four downstream fix nodes are responsible for the
master→branch contract sync (the test contract update and the
deletions of `final_verdict.json` / `living_memory_trace_project_lm.json`
/ `result.md` will land naturally when each fix branch rebases on
or merges current master), not this discovery subtree.

## 5. SPEC mapping (verified)

| Parent SPEC | Artifact | Verification |
|---|---|---|
| P1  | `artifacts/discovery/lm-recall-context.md` | §3 P1 |
| P2  | `artifacts/baseline.md` + raw companions | §3 P2 |
| P3  | `artifacts/discovery/path-inventory.md` | §3 P3 |
| P4  | `artifacts/discovery/scope-hygiene.md` | §3 P4 |
| P5  | `artifacts/discovery/dedup-noise.md` | §3 P5 |
| P6  | `artifacts/discovery/feedback-linkage.md` | §3 P6 |
| P7  | `artifacts/discovery/health-observability.md` | §3 P7 |
| P8  | `artifacts/discovery/recall-speed-usefulness.md` | §3 P8 |
| P9  | `artifacts/ANALYSIS.md` | §3 P9 |
| P10 | (no production diffs on branch) | §4 |

## 6. Reproducibility commands (local, shell-only)

The audit in this file can be re-run from inside the worktree with
these read-only commands:

```
git merge-base HEAD master
git log --name-only master..HEAD
git log --oneline master..HEAD -- ':!artifacts'   # must print nothing
git status --porcelain                            # must print nothing
ls -la artifacts artifacts/discovery
for f in artifacts/baseline.md artifacts/ANALYSIS.md \
         artifacts/discovery/lm-recall-context.md \
         artifacts/discovery/path-inventory.md \
         artifacts/discovery/scope-hygiene.md \
         artifacts/discovery/dedup-noise.md \
         artifacts/discovery/feedback-linkage.md \
         artifacts/discovery/health-observability.md \
         artifacts/discovery/recall-speed-usefulness.md; do
  test -s "$f" && echo "OK  $f" || echo "MISS $f"
  grep -cE '^#{1,3} ' "$f"
done
```

All steps execute against the local filesystem and git index; no
external endpoint is contacted; no network credential is read; no
secret_redaction surface is touched.

## 7. Result

All ten parent SPEC postconditions are satisfied. Discovery subtree
is artifact-only. The worktree carried no unrelated dirty files;
nothing needed to be set aside or reverted. Downstream fix nodes
have what they need in `artifacts/ANALYSIS.md` and the per-problem
briefs under `artifacts/discovery/`.
