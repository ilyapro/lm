# Recall precision valves: rollout handoff (sfx, alt)

This is for the operator. Nothing here has been deployed, restarted or put
into any host's env. Every valve below is **off by default**, and turning one
on is the operator's decision. The numbers come from the pre-registered
live-path measurement:

- `artifacts/recall-precision/prereg.md`: arms, pass lines and frozen values,
  committed before the holdout run
- `artifacts/recall-precision/report.md`: per-host verdicts
- `artifacts/recall-precision/results.json`: the underlying numbers

sfx and alt are measured separately, and the numbers are never merged.

## Recommendation per valve

The rule was fixed in the prereg: a valve is recommended for a host only if
every pre-registered line that applies to it on that host passes on the
holdout.

| valve (value measured) | sfx | alt |
|---|---|---|
| Quality gate `LM_RECALL_MIN_SCORE` + `LM_RECALL_GATE_FORM=drop` (sfx 0.35, alt 0.30) | **do not enable.** Irrelevant full slots −36.9%, used −2.5%, rank 1 never removed, context −23%. But the new-entrant grounding line fails: excess 0.0 at n=16 against 0.042 for baseline (Q1). | **do not enable.** Irrelevant full slots only −22.6% against the 25% line (G1 FAIL); used −6.9%. |
| Hub suppression `LM_HUB_SUPPRESSION_FACTOR=0.1` | **do not enable.** Hub top-3 slots cut 35.9% against the 80% line (H1 FAIL); used −1.3%. Cause: `lookup` credit lifts 39 of 115 hubs. | **do not enable.** Allowed by the rule, but the effect is 4 → 2 hub top-3 slots in 111 answers, too small to matter. |
| Demotion strength `LM_QUERY_IRRELEVANCE_FULL_COSINE=0.80` + `LM_QUERY_IRRELEVANCE_MARK_WEIGHT=1.0` | **enable.** Share of applicable cases with multiplier > 0.9 drops from 76.9% to 23.1% (D1, line < 30%). Effect on delivery is small: irrelevant −2.1%, used −5.1% (4 of 79). | **may enable** (reported only). > 0.9 share 91.1% → 24.2%; irrelevant −0.7%, used −1.1%. |
| Schema dedup `LM_RECALL_SCHEMA_DEDUP=1` | **enable.** Duplicate schema slots 36 → 0 (S1). The 12.7% used loss is on twins: 11 of the 24 used-marked schemas have a same-title twin ranked earlier, and dedup keeps that twin. | **may enable.** No duplicates on alt, so it has no effect. |

Two points are left open on purpose:

1. **The gate on sfx.** Every line on used, irrelevant and rank 1 passes, and
   the only failure is the small-n entrant line. If more holdout data is
   wanted, a re-run after a week of marks is the natural next step
   (commands below).
2. **Hubs.** The valve would need to lift only on `used`, explicit or
   `grounded` credit, not on `lookup`. That is a code change for a follow-up
   goal.

## Exact env lines

These go into the host's `EnvironmentFile`. Keep every line already there,
in particular `LM_EXPLICIT_FEEDBACK_POLICY=credit`,
`LM_EXPLICIT_FEEDBACK_PROMPT=mandatory` and `LM_IMPLICIT_LINK_POLICY=credited`.
Demotion is read only under `LM_EXPLICIT_FEEDBACK_POLICY=credit`.

**sfx**, file `~/.config/living-memory/env` (recommended set):

```
LM_QUERY_IRRELEVANCE_FULL_COSINE=0.80
LM_QUERY_IRRELEVANCE_MARK_WEIGHT=1.0
LM_RECALL_SCHEMA_DEDUP=1
```

**alt**, file `/home/user/.config/living-memory/env` (optional set; both
lines were measured and neither line is required):

```
LM_QUERY_IRRELEVANCE_FULL_COSINE=0.80
LM_QUERY_IRRELEVANCE_MARK_WEIGHT=1.0
LM_RECALL_SCHEMA_DEDUP=1
```

If the operator overrides a "do not enable" call, these are the lines that
were measured. Do not use other values without re-measuring.

```
# sfx
LM_RECALL_MIN_SCORE=0.35
LM_RECALL_GATE_FORM=drop
LM_HUB_SUPPRESSION_FACTOR=0.1
# alt
LM_RECALL_MIN_SCORE=0.30
LM_RECALL_GATE_FORM=drop
LM_HUB_SUPPRESSION_FACTOR=0.1
```

`LM_HUB_MIN_QUERIES` stays unset (default 3).

## Prerequisite

The recall-precision code has landed on lm `master`:

```
git -C /home/sfx/p/lm log master --grep 'AE-Task: tree-node-merge-recall-precision'
```

The deferred deploy plan is LM 01M3MY2FBG4NGAFNPEESF00SX3. Its watcher is
`~/.local/share/ae-watchers/watch-recall-precision.sh`. With every valve
unset, the landed code behaves exactly as before, so updating the code alone
is safe. Env lines are a separate decision.

## sfx: editable install, restart only

`/home/sfx/.local/bin/living-memory-server` runs straight from
`/home/sfx/p/lm` (editable install, `direct_url.json` →
`file:///home/sfx/p/lm`). There is no install step.

1. **Back up the DB before the restart**, using the SQLite backup API with
   the live file opened read-only:

   ```
   mkdir -p ~/.local/share/living-memory/backups
   PYTHONPATH=/home/sfx/p/lm/src python3 -c "from living_memory.retrieval_harness import backup_database as b; \
   b('/home/sfx/.local/share/living-memory/global.sqlite3', \
     '/home/sfx/.local/share/living-memory/backups/global-$(date -u +%Y%m%dT%H%MZ)-pre-recall-precision.sqlite3')"
   ```

   About 1.1 GB. Check `df -h ~` first.
2. Append the chosen lines to `~/.config/living-memory/env`.
3. `systemctl --user restart living-memory.service`
4. Verify:
   - `p=$(systemctl --user show -p MainPID --value living-memory.service); tr '\0' '\n' </proc/$p/environ | grep -E '^LM_(QUERY_IRRELEVANCE|RECALL_|HUB_)'`
     shows exactly the chosen lines.
   - `ps -o lstart= -p $p` is later than the merge commit.
   - Port 8765 answers MCP `initialize`, and a `memory_recall` returns results.

## alt: bundle → ff-merge → reinstall → restart

alt runs a non-editable venv at `~/.local/share/lm-venv`. The repo is at
`~/p/lm`, at `622df40` on 2026-09-29. The procedure is the same as LM
01M1HJ6007QMRKRJTV6YRCQT17 and 01M3MY2FBG4NGAFNPEESF00SX3.

1. On sfx:

   ```
   ALT_HEAD=$(ssh alt 'git -C ~/p/lm rev-parse HEAD')
   git -C /home/sfx/p/lm bundle create /tmp/lm-recall-precision.bundle $ALT_HEAD..master
   scp /tmp/lm-recall-precision.bundle alt:/tmp/
   ```

2. On alt:

   ```
   cd ~/p/lm && git fetch /tmp/lm-recall-precision.bundle master && git merge --ff-only FETCH_HEAD
   cd / && ~/.local/bin/uv pip install --python ~/.local/share/lm-venv/bin/python3 --no-deps --force-reinstall ~/p/lm
   ```

   Check that the installed `RECORD` lists `living_memory/score_gate.py`,
   `living_memory/hubs.py` and `living_memory/schema_dedup.py`.
3. **Back up the DB on alt before the restart.** On 2026-09-29, `df -h ~`
   showed 235 GB free (74%). Memory said ~99% earlier, so check again. If
   space is short, back up to sfx instead, the way the harness streams it
   (`python3 scripts/recall_precision_replay.py snapshot --host alt --ssh alt
   --remote-db /home/user/.local/share/living-memory/global.sqlite3 --dir <dir on sfx>`).
   Local form:

   ```
   mkdir -p ~/.local/share/living-memory/backups
   ~/.local/share/lm-venv/bin/python3 -c "from living_memory.retrieval_harness import backup_database as b; \
   b('/home/user/.local/share/living-memory/global.sqlite3', \
     '/home/user/.local/share/living-memory/backups/global-$(date -u +%Y%m%dT%H%MZ)-pre-recall-precision.sqlite3')"
   ```

4. Append the chosen lines, if any, to `/home/user/.config/living-memory/env`.
5. `systemctl --user restart living-memory.service`
6. Verify as on sfx: env in `/proc/<MainPID>/environ`, start time,
   MCP `initialize`, one `memory_recall`.

## Rollback

All four valves are read-side. They change ranking and delivery only and
write nothing to the DB. Hub state is an in-process cache. To roll back:

1. Delete the valve's line from the host's env file. An unset valve is the
   pre-change behaviour, byte for byte.
2. `systemctl --user restart living-memory.service`. Env is read at process
   start.
3. Confirm through `/proc/<MainPID>/environ` that the line is gone.

The DB backup is only for an unrelated failure during the restart. The
valves do not need it.

## Re-measuring

The snapshots and split are frozen, and their sha256 values are in
`artifacts/recall-precision/split.json`. To re-measure on fresh data, take new
snapshots and a new split with `scripts/recall_precision_replay.py`
`snapshot`/`split`. Then run the driver on each host separately, with thread
limits because sfx is shared:

```
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10 \
  python3 artifacts/recall-precision/runs/driver.py --host sfx --segment holdout \
  --arm gate:LM_RECALL_MIN_SCORE=0.35,LM_RECALL_GATE_FORM=drop \
  --workdir /tmp/rp --out /tmp/rp/holdout-sfx.json
```

The driver sets the production LM_* env itself. The demotion census is
`scripts/query_demotion_strength_census.py --db <snapshot> --since <holdout start>`.
