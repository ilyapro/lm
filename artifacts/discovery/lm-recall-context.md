# Living Memory Recall Context

Generated for `memory-quality-root-fixes/discover-baseline-and-root-causes/memory-context-prelude` on 2026-05-22 before any code grep, source inspection, or SQLite audit on this branch.

## Recall Events

| Topic | Scope | Recall event id |
| --- | --- | --- |
| Omnibus task context: scope leakage, file chunk dedup, feedback linkage, health metrics | `project:lm` | `01KS793V68HWKK6A6DD790NC2C` |
| Scope leakage: `rise/*`, `breakthrough/*`, `ocpa-generative-action-substrate-v1` | `project:lm` | `01KS79441RDC76KAPCBHSZ889C` |
| Scope leakage: `rise/*`, `breakthrough/*`, `ocpa-generative-action-substrate-v1` | `global` | `01KS794EM94H52Y9NWSAGDB05Y` |
| File chunk dedup/bootstrap noise | `project:lm` | `01KS794KBKC8YTQMHJFAZYEGMZ` |
| File chunk dedup/bootstrap noise | `global` | `01KS794PPBY34J7P7MWVA0D1Y2` |
| `feedback_applied` recall_event linkage | `project:lm` | `01KS794TKSBYB4ZSJEA4WJ8V6H` |
| `feedback_applied` recall_event linkage | `global` | `01KS794Y70VPC0C6VC0KNTV56R` |
| `memory_health` metrics and latency baseline | `project:lm` | `01KS795C4V1N4HSZEE0AAKNSZZ` |
| `memory_health` metrics and latency baseline | `global` | `01KS795GHJ7MGCWW1XJ6FDTKBQ` |
| Artifact write-path recall for this file | `project:lm` | `01KS795TWCD4Y39WT4YBJFSAYM` |

## Useful Facts

### Task And Tool Surface

- `01KS787AWG159K8N2V8FGG503V`: the `memory-quality-root-fixes` AE/API goal was created for `project:lm` on 2026-05-22 to address LM scope leakage, duplicate/file-index noise, feedback linkage, health observability, and recall speed/usefulness with simple central mechanisms.
- `01KRV6BBY3MEGXXWRMV7RM61VD`: MCP tools include `memory_remember`, `memory_recall`, `memory_teach`, `memory_consolidate`, `memory_status`, and `memory_health`; `memory_recall` returns a top-level `recall_event_id` and each result repeats it.
- `01KS2P62CJAS3A9D4SJBBE5FHB`: server instructions are the canonical imperative protocol for recall-before-edit/state-writing and remember-after-discovery. Tool handlers are wrapped by a single runtime lock.

### Scope Leakage Signals

- `01KRY9XWFT4KYC8J652TS7CX50`: a global schema named `octopus gemini flash breakthrough decomposition v1` was consolidated from Octopus breakthrough worktree traces. Its provenance points at `/root/p/octopus/.worktrees/_node_exec_breakthrough`, not `project:octopus`.
- `01KS544V04ASECX6JWD2HHWJFV` and `01KS544TZSPBCDJZ5916Z3MK9J`: global schemas for `ocpa generative action substrate v1` and its restart were consolidated from Octopus worktree goal traces.
- `01KS1JM69A6RNPX70YWDQFJ97W`: a global trace for `/root/p/octopus/.worktrees/_node_exec_rise` shows prior recall scope `project:/root/p/octopus/.worktrees/_node_exec_rise`.
- `01KS4F7FT6SRGBPDC60950KMF3` and `01KS4FVBJSX17WTE9ZP434CCV0`: global traces describe repairs/decompositions for `ocpa-generative-action-substrate-v1`; they are semantically Octopus work but stored in `global`.
- Downstream implication: inspect scope normalization/inference for raw worktree-derived scopes such as `project:/root/p/octopus/.worktrees/...` and how consolidation promotes or stores those facts outside `project:octopus`.

### Duplicate And File-Chunk Noise Signals

- `01KRTS5230XMTE2TR1WER1KAZR`, `01KRTS523AXSVZXP8VQC6C5JHA`, and `01KRTS53XJ8ASHASXFQDWFKW3Z`: active `project:lm` bootstrap `[file-chunk]` traces exist for `src/living_memory/resources.py`, `src/living_memory/retrieval.py`, and `tests/test_resources_prompts.py`. They include path/chunk/sha256 metadata and low usefulness, but still appear in recalls.
- `01KS2P6VFYSTRDD8SMVR0D2A5V`: existing decay support is append-only-safe: expired traces and nodes targeted by active `supersedes` edges are soft-decayed; `memory_teach` creates a `supersedes` edge and reduces original confidence/usefulness.
- Downstream implication: inspect bootstrap ingestion and append paths for whether they reinsert identical file chunks despite stable `sha256`, then prefer a soft-decay/supersede or active-ranking filter over destructive deletion.

### Feedback Linkage Signals

- `01KRV67RRKPBRBV9QD0KN9G4KG`: `apply_pending_recall_feedback` runs on `memory_remember`; it queries recent unconsumed `recall_events` with `feedback_applied=0`, scope compatibility, optional session compatibility, and `limit=1`; it creates related edges, stores `prior_recalls`/`recalled_nodes`/`source_traces`, reinforces retrieval results, and marks the event consumed.
- `01KS2P71VZR1CMPFVZPSHPG99W`: feedback has two loops: explicit ranking via `feedback_weighted_score`, and implicit recall-feedback on `memory_remember`/`memory_teach`. `memory_remember` reinforces retrieval weights; `memory_teach` links without reinforcement.
- `01KS7994YXH7W2CS67PCKH32SX` / `01KS799R8NXJ6BECWN411S0X8R`: remembering this artifact exposed a concrete linkage hazard. The first remember used `scope=project:lm`, `task=memory-quality-root-fixes/discover-baseline-and-root-causes/memory-context-prelude`, and `agent=codex`, but implicit feedback linked it to unrelated recall_event `01KS796ZDK1QWJJXW7RD460KD0` with query `test_instructions_imperative.py MUST NOT semantic negative policy size budget`. A second remember of that observation then linked to this file's artifact recall_event `01KS795TWCD4Y39WT4YBJFSAYM`. This supports inspecting same-scope cross-task contamination when no stronger session/task/agent compatibility is enforced.
- Downstream implication: inspect `pending_recall_events` SQL and context matching for task/session/agent/scope behavior. The `limit=1` detail is a concrete suspect for low `feedback_applied` coverage when agents make multiple recalls before one remember.

### Health, Observability, And Latency Signals

- `01KRV5ZNZFZTYWST6RQBMQF68S`: `memory_health` currently reports recall/remember activity ratio, duplicate density, staleness, and retrieval policy.
- `01KRTS5230XMTE2TR1WER1KAZR`: `recall_events_summary` can count total events and `feedback_applied`, but the earlier tool-surface recall says `memory://recall_events` and `memory://connections` helpers exist without being registered as resources.
- `01KS2WDRPJWRZQPQH2T7RE4FKS`: prior latency benchmark on 2026-05-20: MCP hot `memory_status` about 21 ms; MCP `memory_recall` median about 66.5 ms for max_results=1 and 67.5 ms for max_results=5; local service path medians about 44.19 ms, 46.62 ms, and 46.64 ms for max_results 1/5/20; causal max_results=5 median about 13.89 ms; cold/status outliers of 462.2 ms and 1501.8 ms were observed.
- Downstream implication: health work can extend existing `memory_health`/CLI/resource surfaces rather than introduce a new dashboard subsystem, and speed checks should preserve the hot MCP recall baseline around 65-70 ms unless variance is explained.

## Noise To Treat Carefully

- Scope-leakage recalls returned `level=schema` Octopus decomposition procedures. Their relevance here is the storage/provenance evidence, not that LM discovery must adopt those Octopus task procedures.
- Global health/metrics queries also returned AE supervision audit traces because of broad `health`/`audit` terms. They are not LM health metric definitions.

## Inherited Gate State (Pre-Grep Note)

Confirmed on this worktree before any discovery edits: `npm run check` / `PYTHONPATH=src python -m pytest` shows `154 passed` plus `3 failed` exclusively in `tests/test_instructions_imperative.py`:

- `test_required_keywords_present`: `MUST NOT` keyword missing from `_server_instructions(...)` in `src/living_memory/server.py:255`.
- `test_must_not_present`: literal `MUST NOT` count is 0; only `DO NOT` / `What NOT to store` appear.
- `test_size_within_mcp_budget`: instructions size is 7594 bytes against a 3500-6500 bytes contract.

These failures exist identically on `master` and are pre-existing inheritance, not caused by the `memory-quality-root-fixes` subtree. A parallel orphan branch (commits `98756bd` / `7dc421e`, task `tree-node-merge-instructions-contract-tests-1592907`) updates the test contract — it raises the budget to 9000 bytes (UTF-8), drops `MUST NOT` from required keywords, and replaces `test_must_not_present` with a semantic `test_negative_policy_is_hard_and_concrete`. That branch deliberately changes only `tests/test_instructions_imperative.py`, never the instructions themselves.

Downstream implication for this discovery subtree: per parent P10 (`modifies only artifact files and makes no production code changes`) and the `discovery-artifact-verify` sibling (`subtree stayed discovery-only`), do not patch `src/living_memory/server.py` or `tests/test_instructions_imperative.py` from within these discovery children. The 3-test inheritance must be tolerated until the parallel branch lands or the root coordinates an explicit synchronization. When the four fix-* nodes later need a green gate, they should pull in the parallel test contract fix as a single non-discovery step rather than re-deriving it.
