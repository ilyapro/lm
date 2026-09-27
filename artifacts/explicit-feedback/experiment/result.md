# Result: optional vs mandatory explicit recall feedback

Run 2026-09-27 per `preregistration.md` (commit 5ae6873, unchanged; `tasks.jsonl`
sha256 `1a6dfde2…f832f` verified). Runner: `scripts/explicit_feedback_experiment.py`
(subcommands `verify-base → prepare → preflight → run → leakcheck → analyze`).
Machine-readable numbers: `results/summary.json`. Per-run exports:
`results/runs/<task>__<agent>__<arm>/` (sandbox rows as JSONL, observations,
marks, parsed run record, the agreement script's own per-run report). No tokens or
secrets are stored. Sandbox tokens were fresh per server, and `mcp.json` was
deleted after each run.

## Verdict

On **both stores**, the preregistered decision table matches row **P1**:
the optional channel is poor (F3 fires) and the mandatory arm agrees with the
independent check better than random without ritual. The recommended valve
triple per host:

| host | row | `LM_EXPLICIT_FEEDBACK_PROMPT` | `LM_EXPLICIT_FEEDBACK_POLICY` | `LM_EXPLICIT_CREDIT_WEIGHT` |
|---|---|---|---|---|
| sfx | P1 | `mandatory` | `credit` | `1.0` (CI lo 0.30 ≥ 0.10) |
| alt | P1 | `mandatory` | `credit` | `1.0` (CI lo 0.40 ≥ 0.10) |

The two hosts agree, so the fleet setting is the same triple.

**Operator caveat (not preregistered, does not change the row).** The
agreement is carried **entirely by the lookup channel**. The grounded
(placebo-subtracted) channel contributed **zero** evaluable `used` pairs: 0
closed events in 3 of 4 store×arm cells, and in sfx-mandatory the 10 events
closed by 2 codex remembers had no chunk vectors in the sandbox DB, so they
were not scored. In the mandatory arm, 47/55 (alt) and 23/28 (sfx) looked-up
pairs were marked `used`, against 7/125 and 0/105 looked-up unmarked pairs.
So "better than random" here means "agents mark as used what they fetched in
full", which is consistent, but it is not proof that a `used` mark *without* a
lookup is true. Those are the marks where explicit credit adds anything,
because a looked-up pair is already claimed by lookup credit (ledger dedup):
88/135 used marks on alt and 70/93 on sfx. The prereg recommends `credit`
at weight 1.0. A cautious operator could start at the P1 prompt with
`credit` and a lower weight, or at `mandatory` + `audit` until live grounded
evidence for non-lookup `used` marks accumulates. That choice belongs to the
operator. The runner does not make it.

## Setup actually executed

- Code: server `src/living_memory` at `1e57a98` (explicit-marks-core merged);
  runner commit `f5b667e` (HEAD during the run; X4 held).
- Snapshots (`backup_database`, source `file:…?mode=ro`, 2026-09-27 11:55 UTC):
  sfx raw `41e9f11f…a016a`; alt (backup on alt, then scp; only `~/efx_exp/`
  written on alt) raw `ec2d5029…079a6`.
- Prepared bases: sfx `b127e853…48f5a` (31 nodes ablated, 0 missing), alt
  `3cc1b244…6bff51` (40 ablated, 0 missing). No gold node is decayed or missing
  (`gold_missing` = none).
- Knobs re-read from both hosts' env files, identical (§1 list). Overrides:
  `LM_EXPLICIT_FEEDBACK_POLICY=audit`, `LM_DECAY_SWEEP_INTERVAL_SEC=0`, fresh
  `LM_AUTH_TOKEN`. `LM_DB_PATH`/`LM_URL` unset, embedding backend default.
  Ports 18800+, never 8765.
- Server readiness: 6–9 s (no F-cell fallback). 3 pairs (6 servers) ran
  concurrently.
- Agents: Claude Code 2.1.283 `claude-opus-5-5` (template §2.1); codex-cli
  0.156.1 (default model, not reported in the `--json` stream). Both passed
  preflight: exit 0, a `memory_recall` succeeded, and V2 was clean. Claude's init
  listed only `living-memory` (connected) plus `Glob, Grep, Read` and the 4
  sandbox tools. Codex stderr named no foreign URL.
- Queue: 48 pairs = 96 runs, 12:01:57–12:32:14 UTC (30 min of the 120-min
  budget). Tokens 16.10 M total (budget 40 M), output 0.127 M (budget 2 M).
  No stopping rule fired.

### Validity

| cell | runs | invalid | valid pairs | retries |
|---|---|---|---|---|
| sfx-claude | 24 | 0 | 12 | 0 |
| alt-claude | 24 | 0 | 12 | 0 |
| sfx-codex | 24 | 0 | 12 | 0 |
| alt-codex | 24 | 0 | 12 | 0 |

X1 (live touch), `results/leakcheck.json`: 0 exact sandbox-query matches in
either live store over the run window (309 distinct sandbox queries). The live
sfx `recall_events` count went 75012 (11:55) → 75015 (12:32). The 3 new events
in the window are unrelated operator/supervisor queries with no sandbox query
text. No live row has the preflight query `sandbox preflight`. The live alt
count was 43521, with its last event at 11:18, before the window. X2: payloads
differ. X3/X5: no cell invalid, 12 valid Claude pairs per store.

## Description cost (§6.4)

| arm | `tools/list` sha256 | 4-tool payload chars | tokens (tiktoken o200k_base) | recall desc chars (budget 1024) |
|---|---|---|---|---|
| optional | `8afc5d05…61d6cbe` | 6810 | 1702 | 1015 |
| mandatory | `3722e0e9…324d292c` | 6865 | 1724 | 1010 |

The mandatory arm costs **+55 chars / +22 tokens** in the payload. The median
per-pair difference in the first Claude request's input is **+40 tokens** on
both stores. Codex does not report first-request usage.

## Metrics per store × arm (agents pooled; §6)

| | sfx optional | sfx mandatory | alt optional | alt mandatory |
|---|---|---|---|---|
| recall events (runs) | 98 (24) | 101 (24) | 104 (24) | 99 (24) |
| **C** (≥1 accepted mark) | **0.071** | **0.822** | **0.096** | **0.768** |
| C, markable denominator | 0.090 | 0.988 | 0.119 | 0.962 |
| used-only / irrelevant-only / both | 0.061 / 0.010 / 0 | 0.010 / 0.347 / 0.465 | 0.067 / 0.010 / 0.019 | 0.020 / 0.192 / 0.556 |
| marks per marked event | 1.86 | 6.33 | 2.20 | 5.95 |
| accepted used / irrelevant | 10 / 3 | 95 / 432 | 15 / 7 | 136 / 317 |
| reject rate | 0.071 | 0.000 | 0.083 | 0.004 |
| evaluable events | 28 | 29 | 29 | 34 |
| **D_used** (n marks / events) | 0.256 (6/3) | **0.433** (52/23) | 0.576 (9/5) | **0.508** (87/31) |
| p (10 000 perm.) | 0.250 | 0.0001 | 0.0019 | 0.0001 |
| 95% bootstrap CI | [−0.33, 0.50] | [0.30, 0.57] | [0.30, 0.80] | [0.40, 0.61] |
| AG (better than random) | inconclusive → false | **true** | inconclusive → false | **true** |
| D_irr (n) | — (0) | 0.398 (139), p 0.0001 | 0.381 (7), inconcl. | 0.476 (111), p 0.0001 |
| **RS** = R_all ∪ R_rank1 share | 0.000 (0/10) | **0.032** (3/93) | 0.133 (2/15) | **0.074** (10/135) |
| R_all / R_rank1 marks | 0 / 0 | 3 / 0 | 0 / 2 | 10 / 0 |
| R_noclose share (secondary) | 0.40 | 0.35 | 0.00 | 0.24 |
| gold hit rate (used on gold / used in gold-delivered events) | 1/2 | 7/19 | 1/8 | 10/30 |
| used rate class A vs class B (per delivered) | 0.015 / 0.015 | 0.157 / 0.133 | 0.032 / 0.013 | 0.261 / 0.199 |
| known-irrelevant delivered → used / irrelevant | 31 → 0 / 0 | 28 → 2 / 20 | 24 → 0 / 2 | 18 → 3 / 8 |
| recalls per run; 0-recall runs | 4.08; 0 | 4.21; 0 | 4.33; 0 | 4.12; 0 |
| descriptive D_used grounded-only (closed events, no lookup) | no marks | 0.0 (14/6, unscored traces) | no marks | no marks |

### By agent type (secondary)

| store | agent | C opt → mand | D_used mand (n, p) | RS mand | R_noclose mand | tokens/run opt → mand | s/run opt → mand |
|---|---|---|---|---|---|---|---|
| sfx | claude | 0.158 → 0.721 | 0.626 (25, 0.0001) | 0.000 | 0.33 | 97.8k → 121.2k | 41 → 49 |
| sfx | codex | 0.017 → 0.897 | 0.259 (27, 0.0002) | 0.056 | 0.37 | 211.0k → 266.4k | 73 → 98 |
| alt | claude | 0.281 → 0.625 | 0.497 (36, 0.0001) | 0.000 | 0.11 | 92.0k → 83.7k | 34 → 37 |
| alt | codex | 0.014 → 0.836 | 0.529 (51, 0.0001) | 0.114 | 0.31 | 216.1k → 253.1k | 70 → 91 |

No agent type has D_used ≤ 0 with ≥ 10 marks, so the weight is not halved.
Claude marks voluntarily far more than codex under `optional` (0.16/0.28 vs
0.01–0.02). Codex complies almost only when the marks are mandatory. In
the mandatory arm, codex carries all R_all ritual (10 marks on alt, 3 on sfx).

Answer outcome (descriptive, final answer names a gold id for A / says "not
found" for B), opt → mand: sfx-claude A 3/8 → 4/8, B 1/4 → 2/4; sfx-codex A 1/8
→ 0/8, B 1/4 → 2/4; alt-claude A 0/8 → 0/8, B 2/4 → 2/4; alt-codex A 0/8 →
1/8, B 3/4 → 2/4. The mandatory arm costs about +15–30% tokens and time per
run, except alt-claude.

## Falsifiers (§7), per store

| | sfx | alt |
|---|---|---|
| AG_o | false (inconclusive: 6 used marks / 3 events) | false (inconclusive: 9 / 5; D 0.58, p 0.002) |
| AG_m | **true** (D 0.433, p 1e-4, 52/23) | **true** (D 0.508, p 1e-4, 87/31) |
| RIT_m (RS > 0.50, ≥ 20 used) | false (RS 0.032, 93 used) | false (RS 0.074, 135 used) |
| POOR_o (C_o < 0.10) | **true** (0.071) | **true** (0.096) |
| **F1** no agreement in either arm | not triggered | not triggered |
| **F2** mandatory rejected for ritual | not triggered | not triggered |
| **F3** optional channel poor → connect only with the mandatory arm | **triggered** | **triggered** |
| first matching row | **P1** | **P1** |
| W(mandatory) from CI lo | 0.30 → 1.0 | 0.40 → 1.0 |

Weight mapping: explicit-marks-core's `LM_EXPLICIT_CREDIT_WEIGHT` multiplies the
grounded-style signal. 1.0 is parity with a grounded credit, and 0 means the pair
is claimed with no effect. The prereg's levels map 1:1.

## Deviations

1. **PYTHONPATH.** The worktree has no `.cache/python-deps`, so the servers ran
   with `PYTHONPATH=$WT/src` and the user site-packages
   (`~/.local/lib/python3.12`, the same deps the live server uses).
2. **kv token guard.** `prepare` deletes `kv.auth_token` from each base,
   because a rotated live token in kv would override the fresh sandbox token
   (server.py). Neither snapshot had that row, so this was a no-op.
3. **Twin rule.** The twin comes from `explicit_feedback_agreement.pick_twin`,
   a seeded uniform draw keyed by `sha256(event:node)`, not the smallest
   `sha256(e+n+twin)`. Both are deterministic and within ±0.01. It had no
   effect: no `used` pair had a scored closing trace.
4. **Closing traces without vectors.** The 10 sfx-mandatory events closed by
   codex remembers had no `node_chunk_embeddings` row for the trace in the
   sandbox DB, so `observe()` left them unscored (`closed_without_trace_vector`
   = 10). The prereg did not anticipate this. G = 0 was used, so those events
   were evaluable only via lookup.
5. **Agreement script vs prereg statistic.** `explicit_feedback_agreement.py`
   was run on every sandbox DB (`results/runs/*/agreement_script.json`, 96
   reports). Its thresholds (30 marks / 10 events, 2000 permutations) and its
   A/B session filter differ from §6.2, and per-run reports are too small to
   read. The decision-bearing D_used / D_irr / RS were computed in the runner
   exactly per §6.2–6.3, reusing the script's `load_store`/`observe` for G, L
   and twin G, as §6.2 allows.
6. **Interim look.** At about 12:05, while checking liveness, the runner printed
   the accepted-mark row counts of the first 6 runs. Nothing was changed,
   stopped or rerun as a result. No agreement or ritual number was computed
   before the queue ended.
7. **Markable denominator.** "A later `memory_*` call" is approximated as a
   later recall event, a later lookup, or a later node written in the run.
8. **Store copy lifetime.** Each run's store copy was deleted after export
   *and* scoring, because scoring needs the vectors. It was not deleted right
   after export. Copies were plain `cp`, since reflink is unsupported.
9. **Codex details.** The codex model is not in the `--json` stream (recorded
   as null). Codex has no first-request usage, so the token-delta secondary is
   Claude-only.
10. **R_noclose reference text.** The final answer, the arguments of non-memory
    tool calls, and all nodes the run wrote after the event. Intermediate
    assistant text was not included.

## Reproduce

```bash
python3 scripts/explicit_feedback_experiment.py verify-base
python3 scripts/explicit_feedback_experiment.py snapshot     # sfx local + alt over ssh, mode=ro
python3 scripts/explicit_feedback_experiment.py prepare
python3 scripts/explicit_feedback_experiment.py preflight
python3 scripts/explicit_feedback_experiment.py run          # 120-min budget, ~30 min observed
python3 scripts/explicit_feedback_experiment.py leakcheck
python3 scripts/explicit_feedback_experiment.py analyze      # -> results/summary.json
```

Per the prereg, a rerun is a new, separately preregistered attempt.
