# Preregistration: optional vs mandatory explicit recall feedback

Status: **preregistered 2026-09-27, before any arm was run.** No experiment
result exists when this file is committed; `result.md` next to it must not exist
at that commit. The sibling that runs the experiment
(`optional-vs-mandatory-experiment`) follows this file exactly. It does not
edit it. Anything it does differently goes into `result.md` under
**Deviations**, with the reason.

Companion file: `tasks.jsonl` (24 rows, sha256
`1a6dfde2a5dd7765e697735f7b99c6e6a25406068253b2fea979a02a844f832f`), the fixed
task battery. Repository base when it was frozen: `c11ddfa`.

---

## 0. What is being tested, and what is not already known

Hypothesis. When a recall delivers a node, an agent can say directly whether
the node was used (`used`) or was off-topic for the query (`irrelevant`). These
direct marks agree with independent, placebo-subtracted evidence of use better
than chance. There is a second question: does making the marks **mandatory** in
the tool description raise coverage without turning marking into a ritual?

Why this is still open:

- `memory_attest` failed, but that failure is **not** evidence against direct
  marks. `memory_attest` never took an agent's usefulness verdict. The server
  graded the submitted evidence strings with the same lexical containment gate
  (0.25 in August), so 0/21 and 1/6 measure a failure of that checker, not the
  truthfulness of direct used/irrelevant judgements. "27/27" meant that all
  observed attestations were same-session. It did not mean 27 of 27 sessions
  complied. "21%" counts sessions whose recalls were never closed
  (`feedback_trace_id IS NULL`), not a count of `memory_remember` calls
  (Living Memory 01M3H84X8BBYA7BGMTZ9052Z9Y, correcting review
  01M3H7S4W8QZGMRRKG720PTNW7; `docs/post-session-attestation.md`, "the client
  submits evidence, the server decides"). What still holds: attest is off by
  default (`LM_EXPOSE_ATTEST`), and the risk of ritual calls is real. That
  second point is why ritual detection below is a first-class metric.
- The existing live signal is weak. Grounded credit on rank 1 is 12.8% against
  3.7% for a placebo twin, and 7.7% against 2.4% once the query words are
  removed (01M3H76JBHEZKJMBPWFVN70SB2). 26% of recalls are never closed, and a
  recall that memory fully answered earns no credit, because the remember
  policy forbids a trace when nothing new was learned
  (01M3H77WY4ZNR3XNA62Y2YEVMR). So the independent check below is itself
  noisy. It is used only as a *relative* yardstick (marked vs unmarked vs
  twin), never as ground truth.
- The transcript-grounding channel was refuted (01M16PNM28H3EMPCFZ4WD1D59K) and
  is **not** used as the independent check. Run transcripts are used only for
  the ritual detector R_noclose (§6.3), which is secondary.

## 1. Arms and the fixed server configuration

The **only** difference between arms is the environment variable
`LM_EXPLICIT_FEEDBACK_PROMPT` on the sandbox server:

| arm | `LM_EXPLICIT_FEEDBACK_PROMPT` | meaning |
|---|---|---|
| `optional` | `optional` | `used` / `irrelevant` are optional id lists on `memory_recall`, `memory_remember` and `memory_teach`, with a short hint |
| `mandatory` | `mandatory` | same fields, with binding feedback wording in the tool description |

The wording of both arms is whatever sibling `explicit-marks-core` merged, at
the commit the experiment runs from. The runner does not tune it. Before the
first arm run, the runner records for each arm the full `tools/list` payload,
its sha256, and the per-tool description character counts. Every run must see
the payload sha of its arm (§9).

Held constant in both arms:

- `LM_EXPLICIT_FEEDBACK_POLICY=audit`. Marks are recorded in
  `recall_feedback_marks` (recall_event_id, node_id, mark, accepted,
  reject_reason, via_tool, source_id, transport_session_id, agent, rank,
  marked_at). They do not reinforce anything and they do not claim ledger rows.
  Implicit grounded/lookup credit runs as usual and is part of the evidence.
- Retrieval knobs as on the live hosts. They are identical on sfx and alt in
  `~/.config/living-memory/env` as of 2026-09-27:
  `LM_DEFAULT_SCOPE=global LM_AUTO_CONSOLIDATE_POLICY=adaptive
  LM_RETRIEVAL_TUNING_POLICY=adaptive LM_DRAIN_NEAR_DUP_SUPERSEDES=1
  LM_RECALL_NEAR_DUP_COSINE=0.97 LM_MAP_POOL_COLD_QUOTA_GATE=1
  LM_MAP_POOL_COLD_SLOTS=2 LM_MAP_CURTAIL_DECAY=1`. The runner re-reads both
  env files at run time. If a knob differs, it uses each host's own knobs for
  that host's store and records this.
- Overrides: `LM_DECAY_SWEEP_INTERVAL_SEC=0`, so no TTL sweep mutates the
  snapshot mid-run. `LM_AUTH_TOKEN=<fresh random 32-byte hex per server>`.
  `LM_DB_PATH` and `LM_URL` are **unset**. The embedding backend is left at
  the live default (`auto`); `hash` is forbidden because it would disable real
  vector retrieval.
- `LM_IMPLICIT_LINK_POLICY` and `LM_EXPLICIT_CREDIT_WEIGHT` stay at their code
  defaults. They do not matter under `audit` for marks.

## 2. Sandbox, isolation, and the "never touch live" rule

Server code: `src/living_memory` from the experiment worktree, started as

```bash
env -i HOME="$HOME" PATH="$PATH" LANG=C.UTF-8 \
  PYTHONPATH="$WT/src:$WT/.cache/python-deps" \
  LM_EXPLICIT_FEEDBACK_PROMPT="$ARM" LM_EXPLICIT_FEEDBACK_POLICY=audit \
  LM_DECAY_SWEEP_INTERVAL_SEC=0 LM_AUTH_TOKEN="$TOKEN" <knobs from §1> \
  python3 -m living_memory.server --transport http --host 127.0.0.1 \
    --port "$PORT" --db "$RUN/store.sqlite3" --default-scope global
```

`$PORT` is a free port in 18800–18999. **Never 8765.** Readiness means an
authenticated MCP `tools/list` succeeds within 120 s.

Snapshots. Take these once, before any arm, and record their sha256:

- **sfx**: `living_memory.retrieval_harness.backup_database(
  "/home/sfx/.local/share/living-memory/global.sqlite3",
  "$EXP/snap/sfx.raw.sqlite3")`. The source is opened `file:...?mode=ro`.
- **alt**: on alt, over ssh, the same `backup_database` from
  `/home/user/.local/share/living-memory/global.sqlite3` to
  `~/efx_exp/alt.raw.sqlite3`, then `scp` to `$EXP/snap/alt.raw.sqlite3`.
  Only the copy leaves alt. Nothing on alt is written except `~/efx_exp/`.
  All arms run on the sfx host, so alt's live server is never contacted over
  MCP. If alt cannot be reached or the copy fails, the experiment is sfx-only
  and the alt column reads `not run: <reason>`. It is never filled from sfx.

Prepared base per store. Copy the raw snapshot to `$EXP/snap/<store>.base.sqlite3`.
Open it **only that copy** with `MemoryStore`, which applies additive
migrations. Then call `store.soft_delete_node(id, "efx-ablation")` for every id
in `ablate_node_ids` of that store's class-B rows. Record the ablated count and
the missing ids. The base sha256 is recorded.

Per run: `cp --reflink=auto <store>.base.sqlite3 $RUN/store.sqlite3`, then
start a dedicated server on it. Every run therefore starts from the identical
prepared state: remembers and marks from one run never reach another run, arm
or agent. After the run, export these rows as JSONL to
`artifacts/explicit-feedback/experiment/results/<run_id>/` and delete the store
copy: `recall_events`, `recall_feedback_marks`, `recall_credit_ledger`,
`recall_lookup_events`, `connections`, plus the nodes created after the server
started.

Isolation from other Living Memory servers:

1. Agents are launched under `env -i` with an allowlist (below). The shell's
   `LM_AUTH_TOKEN`, `LM_URL` and `LM_DB_PATH`, which point at the **live**
   server, are not inherited.
2. The agent's MCP configuration names exactly one server, `living-memory`,
   pointing at the sandbox port.
3. Claude runs have no command-running or web tool (`--tools "Read,Grep,Glob"`).
   Codex runs in `-s read-only`, whose Linux sandbox disables network for
   model-run commands. So neither agent can open an HTTP connection to
   127.0.0.1:8765 itself.
4. A per-run check (§9, V2) confirms that the agent's startup record lists only
   the sandbox server and only sandbox tools.
5. A post-hoc leak check (§9, X1) runs read-only against both live stores.

### 2.1 Agent command templates

Fixed per run: `$RUN` (per-run dir), `$RUN/cwd` (fresh empty dir), `$PORT`,
`$TOKEN`, `$PROMPT` (the task's `prompt` field verbatim), and the per-run
timeout of 420 s.

`$RUN/mcp.json`:

```json
{"mcpServers":{"living-memory":{"type":"http","url":"http://127.0.0.1:$PORT/mcp",
  "headers":{"Authorization":"Bearer $TOKEN"}}}}
```

**Claude Code (headless):**

```bash
cd "$RUN/cwd" && env -i HOME="$HOME" PATH="$PATH" USER="$USER" LANG=C.UTF-8 TERM=dumb \
  ENABLE_CLAUDEAI_MCP_SERVERS=false \
  timeout 420 claude -p "$PROMPT" \
    --model claude-opus-5-5 \
    --strict-mcp-config --mcp-config "$RUN/mcp.json" \
    --tools "Read,Grep,Glob" \
    --allowedTools "mcp__living-memory__memory_recall,mcp__living-memory__memory_remember,mcp__living-memory__memory_teach,mcp__living-memory__memory_lookup,Read,Grep,Glob" \
    --permission-mode dontAsk \
    --no-session-persistence \
    --output-format stream-json --verbose \
  > "$RUN/claude.stream.jsonl" 2> "$RUN/claude.stderr"
```

User settings stay loaded, so the user's global CLAUDE.md applies in both arms.
It tells the agent to use Living Memory actively, which matches the live fleet
and is constant across arms.

**Codex (headless):**

```bash
env -i HOME="$HOME" PATH="$PATH" USER="$USER" LANG=C.UTF-8 LM_SANDBOX_TOKEN="$TOKEN" \
  timeout 420 codex exec \
    --ignore-user-config --ephemeral --skip-git-repo-check \
    -s read-only -C "$RUN/cwd" \
    -c 'mcp_servers.living-memory.url="http://127.0.0.1:'"$PORT"'/mcp"' \
    -c 'mcp_servers.living-memory.bearer_token_env_var="LM_SANDBOX_TOKEN"' \
    -c 'mcp_servers.living-memory.default_tools_approval_mode="approve"' \
    --json -o "$RUN/codex.last.txt" \
    "$PROMPT" \
  > "$RUN/codex.events.jsonl" 2> "$RUN/codex.stderr"
```

`--ignore-user-config` drops every MCP server in `~/.codex/config.toml`,
including the live `living-memory`. Authentication still comes from
`~/.codex/auth.json`. The model is the codex default; the runner records it
from the event stream. `~/.codex/AGENTS.md` (same text as the global
CLAUDE.md) applies in both arms.

**"Codex usable" rule.** Codex is usable when a preflight
(§8, step P3) against a throwaway copy satisfies all of the following: exit
code 0; one `memory_recall` call through server `living-memory` succeeded
(an `mcp_tool_call` item in the `--json` stream); and the stderr MCP transport
lines name no URL or command other than the sandbox, all within 120 s. Codex
0.156.1 emits no MCP startup event in `--json`; this was checked 2026-09-27
with a dummy URL, where every rmcp error in stderr named only that URL. If codex is not usable, codex cells are dropped
and reported as `codex: not usable (<error>)`, and the Claude cells run
unchanged. This is not a deviation.

## 3. Task battery

`tasks.jsonl` holds 24 tasks: **12 per store** (sfx, alt), 8 of class
`answer_in_memory` (A) and 4 of class `answer_not_in_memory` (B) per store.
Every task comes from one real historical recall event on that store. The task
prompt carries that event's query verbatim, framed as "a real earlier session
needed to know this; find it" (template in the `prompt` field). The prompt says
nothing about marks or feedback.

Selection rule. The script is in Appendix A. It is deterministic and read-only
(`?mode=ro`), and identical on both hosts (sha256 `34254432695ab739406ac466ca52b9980763886a7dc6806d0b785fcd17ad1c41`).

1. Pool: `recall_events` with `gated = 0` and `created_at` in
   [2026-09-07, 2026-09-27) UTC.
2. A/B exclusion at **session** level. Drop every transport session that has
   any event with scope `project:target` or `project:repo`, or with a query
   matching `ledger|billing|tree-context|fixture|kit acceptance`: 1619 sessions
   on sfx and 229 on alt. This filter is stricter than the per-event filter
   that leaked on 23.09.
3. Drop queries shorter than 25 or longer than 300 characters. Drop queries
   that mention the experiment's own subject (`explicit|irrelevan|used/|
   recall_feedback|feedback_marks|preregist`), queries that look like secrets,
   and events with fewer than 3 delivered nodes. De-duplicate on the first 60
   normalized characters.
4. Class A: the event has at least one `recall_credit_ledger` row (grounded or
   lookup). `gold_node_ids` are the credited delivered nodes that are not
   decayed and have at least 120 characters.
5. Class B: the event was closed (`feedback_trace_id` set) and earned **no**
   credit, i.e. its delivery did not answer it and the session had to write
   new knowledge. The closing trace T (at least 200 characters) is **ablated**
   from the prepared base: soft-deleted together with nodes sharing its
   content fingerprint, nodes whose `source_traces` include T, and nodes that
   supersede T (`ablate_node_ids`). The store then keeps the topically related
   material that did not answer the question, and not the answer itself.
6. Known-irrelevant material (`known_irrelevant_node_ids`) means delivered
   nodes of the source event that were not credited and whose IDF containment
   against the gold (A) or against T (B) is below 0.05
   (`living_memory.grounding.ground_results`). Class B tasks are the "memory
   has only non-answering material" condition. The known-irrelevant list is a
   weak label, used only in secondary metrics.
7. Strata, in hash order `sha256("explicit-feedback-prereg-2026-09-27:" +
   event_id)`. Class A takes 3 from LM scopes (`project:living-memory`,
   `project:lm`), 2 from `project:ae` and 3 from other scopes. Class B takes
   1, 1 and 2. A short stratum is filled from the rest of the same class in
   hash order. alt has no eligible LM-scope events, so its LM quota was filled
   from `project:ae` and `project:game`.

Realized battery:

- sfx: scopes lm×4, living-memory×1, ae×3, goal-tree×1, 2gis-seaf-web×2,
  online×2.
- alt: ae×3, game×9.

Snapshot drift. The experiment snapshot is newer than the battery. If a gold
node is decayed or missing in the snapshot, the task still runs and is flagged
`gold_missing`. Its runs count for compliance and ritual, but not for the gold
secondary metric.

## 4. Design, randomization, sample size, budget

- Cells: store ∈ {sfx, alt} × agent ∈ {claude, codex}, up to 4 cells. Every
  task runs **once per arm** in its store's cells, so both arms get the same
  tasks and the design is paired. That makes 12 tasks × 2 arms = 24 runs per
  cell and at most 96 runs in total.
- **Pairs are launched together.** A pair is (task, agent) in both arms, each
  on its own fresh copy and server. The two runs start within 5 s of each
  other, so API-side drift hits both arms equally. The first launched arm
  alternates by `launch_position` (`first_launched_arm`: odd → optional first).
- Launch queue, interleaved so that truncation stays balanced: for
  `launch_position` p = 1..12, for each cell in the order
  [sfx-claude, alt-claude, sfx-codex, alt-codex], enqueue the pair (task of
  that store at position p, agent). At most **3 pairs** (6 runs, 6 servers) run
  at the same time.
- Per-run timeout: 420 s. Readiness is ≤120 s. At about 5 min per pair
  including setup and 3 concurrent pairs, 48 pairs take about 80 min.
- **Wall-clock budget: 120 min** from the first arm launch. It covers arm runs
  only; preflight and snapshotting come before it.
- **Token budget:** at most 40 M total tokens across all runs, counting input
  including cache reads and cache creation, plus output. At most 2 M output
  tokens in total. At most 150 k output tokens per run. Claude is counted from
  the stream-json `result.usage`, codex from the `--json` usage events.
- Honest power note: expect about 2–4 recall calls per run, so roughly 50–100
  recall events per (store, arm), agents pooled. That resolves the 10% and 50%
  thresholds coarsely, but not small agreement effects. The minimum-n rules
  in §6 turn "too few marks" into **inconclusive**, not into a pass.

## 5. Units and definitions

- **Recall event**: a row in the run's `recall_events`, i.e. one
  `memory_recall` call by the agent. A run is one transport session.
- **Delivered pair**: (event, node) for each node in that event's `results`.
  Rank is 0-based, as in `recall_feedback_marks.rank`.
- **Accepted mark**: a `recall_feedback_marks` row with `accepted = 1`.
  Rejected rows (foreign or unknown ids) are reported as `reject_rate`, per
  arm, and never counted as marks.
- **Closing trace** of event e: the first `memory_remember` or `memory_teach`
  node written later in the same run (the live pending-feedback rule).
- **G(e,n)**: 1 if node n grounds against e's closing trace at containment
  ≥ 0.22 (`living_memory.grounding`, current production threshold), else 0.
  It is 0 when e has no closing trace.
- **L(e,n)**: 1 if the same run `memory_lookup`-ed n after e (the lookup
  ledger), else 0.
- **Twin(e,n)**: a node that was *not* delivered in e and was created before
  e, with |cos(twin, closing) − cos(n, closing)| ≤ 0.01. Cosines use
  mean-pooled `node_chunk_embeddings`, per the method in
  01M3H76JBHEZKJMBPWFVN70SB2. The twin is picked deterministically: the
  smallest `sha256(e.id + n.id + twin.id)` among candidates. If there is no
  candidate, the pair gets no twin and is dropped from twin-subtracted
  statistics, and this is counted.
- **Evidence score** s(e,n) = max(G(e,n), L(e,n)) − G(e,Twin(e,n)). This is the
  placebo-subtracted grounded excess plus lookup. Lookup has no placebo term,
  because an undelivered twin cannot be looked up from the delivery.
- **Evaluable event**: an event with a closing trace **or** at least one
  lookup after it. Agreement statistics use evaluable events only.

## 6. Metrics

Every metric is computed **per store** (sfx and alt never merged) × **per arm**.
Agents are pooled in the primary line and broken down by agent type in a
secondary line.

### 6.1 Coverage (compliance)

**C = share of recall events that received ≥ 1 accepted mark** (used or
irrelevant) from any later call in the same run. The denominator is all recall
events of valid runs. Secondary: the same with a "markable" denominator
(events followed by at least one later `memory_*` call); used-only and
irrelevant-only shares; marks per marked event.

### 6.2 Agreement with the independent check (primary)

Computed by `scripts/explicit_feedback_agreement.py` (sibling
`marks-agreement-check`). If that script lacks a piece, the runner computes it
in `scripts/explicit_feedback_experiment.py` exactly as defined here and says
so in `result.md`.

**Statistic D_used** is evaluated over evaluable events that have ≥ 1 accepted
`used` mark:

  D_used = mean s(e,n) over `used`-marked pairs − mean s(e,n) over delivered
  pairs in the same events that carry no mark.

**"Better than random" test.** A within-event permutation test. The null is a
marker that picks the same number of `used` nodes per event uniformly at
random among that event's delivered nodes.

- Shuffle the `used` labels among each event's delivered nodes, keeping each
  event's count.
- Recompute D_used, 10 000 times, with a seeded PRNG (seed = first 8 hex
  digits of the `tasks.jsonl` sha256 as an int).
- One-sided p = (1 + #{D_perm ≥ D_obs}) / 10 001.
- **Better than random** ⇔ D_used > 0 **and** p < 0.05 **and** the cell has
  ≥ 20 evaluable `used` marks from ≥ 8 distinct events.
- If the count condition fails, the cell is **inconclusive**. For every
  decision below, inconclusive is treated as "not better than random".

Also report a 95% bootstrap CI of D_used, resampling events and 10 000
replicates with the same seed. It is used by the weight rule in §7.

Secondary agreement, reported but not decision-bearing:

- D_irr = mean s(unmarked) − mean s(`irrelevant`-marked), with the same
  permutation test.
- Gold hit rate: share of `used` marks on class-A gold nodes among events
  where a gold node was delivered.
- `used` mark rate on class-B events vs class-A events. Class B has no answer
  in memory, so a high `used` rate there is a warning sign.
- Share of known-irrelevant nodes, when delivered, that were marked `used`
  vs `irrelevant`.

### 6.3 Ritual

Flags on accepted `used` marks:

- **R_all ("everything delivered = used")**: the event has ≥ 3 delivered nodes
  and its `used` set equals its delivered set. All of that event's `used`
  marks are flagged.
- **R_rank1 ("always only rank 1")**: in a run with ≥ 2 events carrying
  `used` marks, every such event's `used` set is exactly {rank 0}. All those
  marks are flagged. A single rank-1 mark in a run is not flagged, because it
  cannot be told apart from a legitimate one.
- **R_noclose ("marks without closing work")**, secondary: a `used` mark whose
  node grounds (containment ≥ 0.22) against nothing the run produced after the
  event. That means the closing trace, later remember/teach content, the final
  answer text, and the arguments of non-memory tool calls. Transcripts serve
  this detector only.

**Ritual share** RS = share of accepted `used` marks flagged by R_all ∪
R_rank1. This follows the goal's falsifier text. R_noclose and the union of
all three are reported separately. Per-detector shares and the per-agent split
are reported as well.

### 6.4 Description token cost

- Primary: the characters and tokens of each arm's `tools/list` payload
  (descriptions plus input schemas of the four agent-visible tools), and the
  mandatory − optional difference. Tokens use `tiktoken` `o200k_base` if
  installed, else chars/4, and the result says which.
- Secondary: the per-pair difference in the input tokens of the first Claude
  request (`input_tokens + cache_creation_input_tokens + cache_read_input_tokens`
  of the first assistant turn), median over pairs.
- Also report `memory_recall`'s description length against the 1024-character
  budget pinned by `tests/test_instructions_imperative.py`.

### 6.5 Descriptive only, not decision-bearing

- Recall uptake: recall calls per run, and runs with 0 recalls, per arm.
- Answer outcome: whether the final answer names ≥ 1 gold-node identifier (A),
  or says "not found" (B).
- Run duration and tokens per run, per arm.

## 7. Thresholds, falsifiers and decision table (fixed before the run)

For each store, compute:

- **AG_o / AG_m**: D_used is better than random (§6.2) in the optional /
  mandatory arm.
- **RIT_m**: RS > 0.50 in the mandatory arm, with ≥ 20 accepted `used` marks.
  With fewer marks it is inconclusive. Treat that as RIT_m = false for
  rejection, but the mandatory arm then cannot be *recommended* either (row
  M0).
- **POOR_o**: C < 0.10 in the optional arm.

Falsifiers. Each stated before the run; each applied per store.

- **F1**: if AG is false in **both** arms, direct `used` marks show no
  connection to reinforcement on that store. They are **not** connected to
  reinforcement, and the negative result is recorded.
- **F2**: if RIT_m holds, the **mandatory arm is rejected**.
- **F3**: if POOR_o holds, the optional channel is poor. Marks may be
  connected only together with the mandatory arm, never as optional alone.

Decision table. Apply the first matching row per store.

| row | condition (per store) | `LM_EXPLICIT_FEEDBACK_PROMPT` | `LM_EXPLICIT_FEEDBACK_POLICY` | `LM_EXPLICIT_CREDIT_WEIGHT` |
|---|---|---|---|---|
| F1 | AG_o false **and** AG_m false | `optional` (cheapest description; keeps collecting) | `audit` | `0` |
| M0 | mandatory not evaluable (RIT_m inconclusive **and** AG_m inconclusive) **and** (POOR_o **or** AG_o false) | `optional` | `audit` | `0` |
| R1 | RIT_m **and** (POOR_o **or** AG_o false) | `optional` | `audit` | `0` |
| R2 | RIT_m **and** AG_o **and** not POOR_o | `optional` | `credit` | W(optional) |
| P1 | POOR_o **and** AG_m **and** not RIT_m | `mandatory` | `credit` | W(mandatory) |
| P2 | POOR_o **and** (AG_m false **or** RIT_m) | `optional` | `audit` | `0` |
| O1 | AG_o **and** not POOR_o **and** AG_m **and** not RIT_m **and** C_m ≥ 2·C_o **and** C_m − C_o ≥ 0.15 | `mandatory` | `credit` | W(mandatory) |
| O2 | AG_o **and** not POOR_o (all other cases) | `optional` | `credit` | W(optional) |
| O3 | AG_o false **and** AG_m **and** not RIT_m | `mandatory` | `credit` | W(mandatory) |

Weight rule. W(arm) is set from the lower bound (lo) of the 95% bootstrap CI
of D_used in that arm:

- lo ≥ 0.10 → `1.0` (parity with a grounded credit);
- 0 < lo < 0.10 → `0.5`;
- lo ≤ 0 → `0.25`, flagged "weak". This can only happen when p < 0.05 but the
  CI touches 0.

If the per-agent D_used is ≤ 0 for one agent type with ≥ 10 evaluable `used`
marks, the weight is halved and the result flags it for the operator, because
the valve is global and cannot be set per agent. The mapping assumes
explicit-marks-core's semantics: 1.0 is parity with a grounded credit, and 0
means no effect. If the merged semantics differ, the runner maps these
three levels onto them and records the mapping.

Combining the stores. The recommendation is **per host**, because each host
has its own store and its own valve. If the two stores' rows differ and an
operator wants a single fleet setting, use the more conservative of the two.
From least to most conservative, the order is `audit` < `credit`, and under
credit, `mandatory` over `optional`, then a lower weight. Results from
different stores are never pooled into one number.

The recommendation is an input to the operator. Enabling `credit` on a live
host is a restart/deploy decision that belongs to the operator (see the
rollout handoff).

## 8. Order of operations

- **P1.** Verify the base: `git log` shows this file and `tasks.jsonl`
  unchanged since their preregistration commit, and `result.md` absent.
  Record `git rev-parse HEAD` of the experiment worktree.
- **P2.** Take the snapshots (§2) and build the prepared bases. Record the
  hashes and ablation counts.
- **P3.** Preflight on a throwaway copy, excluded from results. For each arm,
  start a server, then:
  - Record `tools/list` (payload sha, character counts).
  - Assert that the optional and mandatory payloads differ.
  - Run one Claude and one codex smoke call: the prompt "Call memory_recall
    with query 'sandbox preflight' and report how many results you got."
  - Apply the codex-usable rule.
  - Measure server readiness time.

  If readiness exceeds 120 s, or 6 concurrent servers do not fit in memory,
  use **fallback F-cell**. Each (store, arm, agent) cell gets one server and one
  store copy, and its tasks run sequentially in `launch_position` order.
  Pairing across arms is kept by running the two arms' cells concurrently.
  Record the switch as a preregistered contingency, not a deviation. Its known
  cost is that within-cell carry-over of remembers becomes possible; it is
  reported.
- **P4.** Run the launch queue (§4) under the stopping rules (§9).
- **P5.** Leak check X1. Export results, compute §6, apply §7, and write
  `result.md`.

## 9. Stopping rules and validity

Stopping, for the whole experiment:

- **S1**: no new pair launches after 105 min from the first arm launch. Runs
  still in flight at 120 min are killed and marked `incomplete`.
- **S2**: the token budget (§4) is checked after each pair. If it is exceeded,
  no new pair launches and in-flight pairs finish.
- **S3**: after 3 consecutive pairs fail with provider errors (rate limit,
  auth, 5xx), stop launching for that agent. The other agent continues.
- **S4**: an immediate abort, which also invalidates everything, when a live
  touch is detected (X1, or V2 showing a non-sandbox LM server).
- No interim look at marks, agreement or ritual numbers before the queue ends.
  Only budget, validity and liveness are monitored. Nothing is rerun because
  its result looked wrong.

A **run** is invalid, excluded and counted per cell, when any of these hold:

- **V1**: the agent exits non-zero or times out before its first tool call,
  or hits an auth or quota error. One retry is allowed on a fresh copy, then
  the run is invalid.
- **V2**: the agent's startup record (Claude stream-json `system/init`
  `mcp_servers` and tools) lists any MCP server other than the sandbox
  `living-memory`, or that server is not `connected`. For codex, which has no
  startup event, V2 fires when an MCP tool-call item names a server other than
  `living-memory`, or when stderr shows an MCP transport to any URL or command
  other than the sandbox. A 2026-09-27 template check confirmed that Claude's
  init with `ENABLE_CLAUDEAI_MCP_SERVERS=false` and `--strict-mcp-config`
  listed only `living-memory`, with tools `Glob, Grep, Read`.
- **V3**: the sandbox server crashed or became unreachable mid-run.
- **V4**: the arm payload sha in the run differs from the arm's recorded sha.
- **V5**: the run ended `incomplete` (S1).

If one run of a pair is invalid, the pair is dropped from the paired
comparisons (the token-cost delta) and both runs are excluded from every
metric, which keeps the arms on identical task sets. A valid run with 0 recall
calls is valid. It adds 0 events and shows up in recall uptake.

The **experiment** (or one store's part of it) is invalid when any of these
hold:

- **X1 (live touch)**: after the queue ends, the runner scans each live store
  **read-only** (`?mode=ro`). It looks for `recall_events` created during the
  experiment window whose `query` exactly equals any sandbox recall query from
  that window. If there is ≥ 1 match that the live traffic cannot explain
  (same query text by a non-experiment agent is checked by transport session
  and agent), the experiment is invalid. The live DB files are never opened
  read-write, and no process started by the runner connects to port 8765.
- **X2**: the optional and mandatory payloads are identical, meaning the arm
  valve is not wired.
- **X3**: more than 25% of runs in a cell are invalid. That cell is invalid. If
  both Claude cells of a store are invalid, that store's result is invalid.
- **X4**: the worktree code differs from the recorded HEAD during the run, or
  this file or `tasks.jsonl` changed after preregistration.
- **X5**: fewer than 6 valid pairs in a store's Claude cell. That store's
  result is invalid (too small to read).

An invalid store yields no recommendation, only the row `invalid: <reason>`.
Rerunning is allowed only as a new, separately preregistered attempt.

## 10. Outputs the runner must produce

`artifacts/explicit-feedback/experiment/result.md` contains:

- Code HEAD, snapshot and base hashes, and the per-arm payload sha and cost.
- Valid and invalid runs per cell.
- Per store × arm: C, D_used with p, CI and n, D_irr, RS and each detector,
  and the secondary metrics, each also by agent.
- F1–F3 outcomes, the matched decision-table row, and the recommended valve
  triple per host.
- Deviations.

`results/` holds per-run exports and a machine-readable summary.

---

## Appendix A — battery selection script (as run, both hosts)

Run 2026-09-27 read-only. The sfx command was
`PYTHONPATH=<worktree>/src python3 select_battery.py sfx /home/sfx/.local/share/living-memory/global.sqlite3`.
The alt command was the same script with `~/.local/share/lm-venv/bin/python`
over `/home/user/.local/share/living-memory/global.sqlite3`, with the
worktree's `src/living_memory` copied to alt. The pools left after
selection were:

- sfx: A:lm 0, A:ae 1093, A:other 659, B:lm 28, B:ae 1398, B:other 1581.
- alt: A:lm 0, A:ae 159, A:other 493, B:lm 0, B:ae 237, B:other 743.

`tasks.jsonl` is the two outputs concatenated. Three fields are added:
`prompt` (the template shown in each row); `order_key` =
`sha256("explicit-feedback-prereg-2026-09-27:order:" + task_id)`; and
`launch_position` / `first_launched_arm`, which come from order_key rank
within the store, odd → optional first.

```python
"""Deterministic task-battery selection for the explicit-feedback experiment.

Read-only: the store is opened as file:...?mode=ro. Usage:
    PYTHONPATH=<repo>/src python3 select_battery.py <store_label> <sqlite_path>
Prints JSONL task rows to stdout.
"""
import hashlib, json, re, sqlite3, sys

from living_memory.grounding import ground_results

SALT = "explicit-feedback-prereg-2026-09-27"
WINDOW = ("2026-09-07T00:00:00Z", "2026-09-27T00:00:00Z")
AB_SCOPES = {"project:target", "project:repo"}
AB_QUERY = re.compile(r"ledger|billing|tree-context|fixture|kit acceptance", re.I)
SELF_REF = re.compile(r"explicit|irrelevan|used/|recall_feedback|feedback_marks|preregist", re.I)
SECRETISH = re.compile(r"(glpat-|sk-|figd_|Bearer |[0-9a-f]{40,})")
LM_SCOPES = {"project:living-memory", "project:lm"}
QUOTA = {  # class -> [(stratum, n)]
    "A": [("lm", 3), ("ae", 2), ("other", 3)],
    "B": [("lm", 1), ("ae", 1), ("other", 2)],
}
IRRELEVANT_MAX_CONTAINMENT = 0.05


def stratum(scope):
    if scope in LM_SCOPES:
        return "lm"
    if scope == "project:ae":
        return "ae"
    return "other"


def key(event_id):
    return hashlib.sha256(f"{SALT}:{event_id}".encode()).hexdigest()


def main(label, path):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    events = c.execute(
        "select id, query, scope, agent, task, results, transport_session_id,"
        " feedback_trace_id, created_at from recall_events"
        " where created_at >= ? and created_at < ? and gated = 0",
        WINDOW,
    ).fetchall()
    tainted = {
        e["transport_session_id"]
        for e in events
        if e["transport_session_id"]
        and (e["scope"] in AB_SCOPES or AB_QUERY.search(e["query"] or ""))
    }
    credit = {}
    for r in c.execute("select recall_event_id, node_id, basis from recall_credit_ledger"):
        credit.setdefault(r[0], {})[r[1]] = r[2]

    def node(nid):
        return c.execute(
            "select id, content, level, scope, decayed, created_at, content_fingerprint"
            " from nodes where id = ?",
            (nid,),
        ).fetchone()

    seen_queries = set()
    pools = {("A", s): [] for s in ("lm", "ae", "other")}
    pools.update({("B", s): [] for s in ("lm", "ae", "other")})
    for e in sorted(events, key=lambda e: key(e["id"])):
        q = (e["query"] or "").strip()
        if e["transport_session_id"] in tainted or e["scope"] in AB_SCOPES:
            continue
        if not (25 <= len(q) <= 300) or SELF_REF.search(q) or SECRETISH.search(q):
            continue
        norm = re.sub(r"\W+", " ", q.lower())[:60]
        if norm in seen_queries:
            continue
        results = json.loads(e["results"] or "[]")
        delivered = [r["node_id"] for r in results if r.get("node_id")]
        if len(delivered) < 3:
            continue
        credited = credit.get(e["id"], {})
        if credited:
            gold = [n for n in delivered if n in credited]
            gold_rows = [node(n) for n in gold]
            gold_rows = [g for g in gold_rows if g and not g["decayed"] and len(g["content"]) >= 120]
            if not gold_rows:
                continue
            cls, ref_text = "A", "\n".join(g["content"] for g in gold_rows)
            ablate = []
        else:
            if not e["feedback_trace_id"]:
                continue
            t = node(e["feedback_trace_id"])
            if not t or t["decayed"] or len(t["content"]) < 200:
                continue
            cls, ref_text, gold_rows = "B", t["content"], []
            ablate = {t["id"]}
            if t["content_fingerprint"]:
                ablate |= {
                    r[0]
                    for r in c.execute(
                        "select id from nodes where content_fingerprint = ?",
                        (t["content_fingerprint"],),
                    )
                }
            ablate |= {
                r[0]
                for r in c.execute(
                    "select id from nodes where source_traces like ?", (f"%{t['id']}%",)
                )
            }
            ablate |= {
                r[0]
                for r in c.execute(
                    "select source_id from connections where target_id = ? and type = 'supersedes'",
                    (t["id"],),
                )
            }
            ablate = sorted(ablate)
        pool = pools[(cls, stratum(e["scope"]))]
        contents = {}
        for n in delivered:
            row = node(n)
            if row and n not in {g["id"] for g in gold_rows}:
                contents[n] = row["content"]
        graded = ground_results(ref_text, contents) if contents else {}
        irrelevant = sorted(
            n
            for n, g in graded.items()
            if g.containment < IRRELEVANT_MAX_CONTAINMENT and n not in credited
        )
        seen_queries.add(norm)
        pool.append(
            {
                "store": label,
                "class": "answer_in_memory" if cls == "A" else "answer_not_in_memory",
                "stratum": stratum(e["scope"]),
                "source_recall_event_id": e["id"],
                "source_created_at": e["created_at"],
                "scope": e["scope"],
                "source_agent": e["agent"],
                "source_task": e["task"],
                "query": q,
                "gold_node_ids": sorted(g["id"] for g in gold_rows),
                "gold_basis": sorted({credited[g["id"]] for g in gold_rows}),
                "closing_trace_id": e["feedback_trace_id"],
                "ablate_node_ids": ablate,
                "known_irrelevant_node_ids": irrelevant,
                "source_delivered_node_ids": delivered,
            }
        )
    out = []
    for cls, quotas in QUOTA.items():
        shortfall = 0
        for s, n in quotas:
            pool = pools[(cls, s)]
            out.extend(pool[:n])
            shortfall += max(0, n - len(pool))
            del pool[:n]
        # fill any stratum shortfall from the remaining candidates of the same class, hash order
        rest = sorted(
            (t for s in ("lm", "ae", "other") for t in pools[(cls, s)]),
            key=lambda t: key(t["source_recall_event_id"]),
        )
        out.extend(rest[:shortfall])
    for i, t in enumerate(out, 1):
        t["task_id"] = f"{label}-{'A' if t['class'] == 'answer_in_memory' else 'B'}{i:02d}"
    for t in out:
        print(json.dumps({"task_id": t.pop("task_id"), **t}, ensure_ascii=False, sort_keys=False))
    print(
        json.dumps({"_pool_sizes": {f"{k[0]}:{k[1]}": len(v) for k, v in pools.items()}, "_tainted_sessions": len(tainted)}),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```
