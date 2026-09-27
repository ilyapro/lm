# Rollout handoff: explicit `used` / `irrelevant` recall feedback

Goal `explicit-recall-feedback`, node `rollout-handoff`, 2026-09-27.
Audience: the operator. **Nothing has been deployed.** Every deploy, restart
and valve change below is an operator decision.

**Incident while this handoff was being written (read this).** At
2026-09-27 12:47:38Z, a botched self-test of a snippet from this file
executed the §3 Phase C block on sfx. It appended
`LM_EXPLICIT_CREDIT_WEIGHT=1.0` to `~/.config/living-memory/env` and
**restarted the live sfx `living-memory.service`** (pid 392728 → 1742847).
The env line was removed right away (the file is back to 378 bytes, mode
600). The restarted process runs the same master `c11ddfa` code, which
never reads `LM_EXPLICIT_*`, so behaviour is unchanged. The one effect was
a restart blip that dropped open MCP sessions on sfx. alt was not touched.
LM trace 01M3HEHN0TNHEXB2608H10N7ST.

State at writing time, checked read-only:

| | sfx | alt |
|---|---|---|
| LM checkout | `/home/sfx/p/lm`, master `c11ddfa` | `~/p/lm` (= `/home/user/p/lm`), master `c11ddfa` |
| install | editable (`__editable__.living_memory-0.1.0.pth` → `/home/sfx/p/lm/src`) | wheel in `~/.local/share/lm-venv` (`uv pip install`, no pip in the venv) |
| unit | `systemctl --user` `living-memory.service`, `EnvironmentFile=~/.config/living-memory/env`, `127.0.0.1:8765` | same unit and env path, `ExecStart=~/.local/share/lm-venv/bin/living-memory-server` |
| store | `~/.local/share/living-memory/global.sqlite3` (1.05 GB + WAL) | same path (0.52 GB + WAL). Disk is 94% full, 58 GB free |
| new tables | absent (`check_rollout_schema.py --expect absent` → ok) | absent (same check, run over ssh → ok) |
| explicit-feedback env vars | none set | none set |

The rollout commit is master after this goal's branch merges. It carries 7
commits on top of `c11ddfa` (explicit-marks-core `1e57a98`, irrelevance
`fdeaf55`, link-policy replay `227b16b`, effect metric `5edd738`, experiment
`b5ee7b6`, prereg `5ae6873`, agreement check `36cef9c`), plus this
handoff. Only `1e57a98` and `fdeaf55` touch `src/`. Below, `ROLLOUT=<sha of master
after the merge>`.

## 1. What changes

Sources: `docs/explicit-feedback.md` and `docs/query-irrelevance.md`.

**API.** All changes are optional fields.
- `memory_recall`, `memory_remember` and `memory_teach` take optional `used`
  and `irrelevant` lists of node ids (`default: null`, not in `required`). A
  call without them behaves exactly as before, and its response gets no new
  key. `memory_lookup` is unchanged, because a lookup already counts as a
  "used" signal. No tool is added.
- A mark is accepted only if a recall on the **same transport session**
  delivered that id within `LM_LOOKUP_CREDIT_WINDOW_SECONDS` (24 h). Anything
  else is dropped with a reason (`empty`, `duplicate`, `conflict`,
  `no_transport`, `not_delivered`). When the fields are passed, the response
  carries `feedback_marks: {accepted, dropped, by_reason}`.

**Storage.** Additive. `schema_version` stays 8, and no existing DDL string changes.
- `recall_feedback_marks`: an audit row for every mark, accepted or dropped,
  plus the index `idx_recall_feedback_marks_event`. Created by `CREATE TABLE
  IF NOT EXISTS` on the first open, i.e. **at restart**.
- `recall_explicit_credit(recall_event_id, node_id, source_id, credited_at,
  UNIQUE(event,node))`: the explicit credit basis. It is a **side table,
  not a rebuild of `recall_credit_ledger`**. The ledger's `CHECK (basis IN
  ('grounded','lookup'))` cannot be altered in SQLite, and a rebuild would
  rewrite the live ledger in a way that a code rollback cannot undo. A single
  `INSERT OR IGNORE … WHERE NOT EXISTS (other table)` statement keeps the
  claim once per (event, node) across `grounded | lookup | explicit`.
  Created at restart. **Readers note:** `recall_credit_ledger` alone still
  shows only grounded and lookup credit. Explicit credit is in the side table.
- `query_irrelevance(anchor_id, node_id, weight, marks, cancels, …)`: this
  table is **lazy**. The first accepted `irrelevant` mark under
  `LM_EXPLICIT_FEEDBACK_POLICY=credit` creates it. Its absence after the
  restart is expected.

This is an additive migration: it needs a restart, and that is the operator's
decision. It was rehearsed on a copy of the live sfx store with the rollout
code (§2.0).

**Behaviour, by valve** (§3 has the recommended values):
- `audit` (code default): marks are recorded and audited, and link hygiene
  applies. Implicit feedback creates no `related` edge from a closing trace
  to a node with an accepted `irrelevant` mark for that event, and a later
  irrelevant mark deletes that edge. Nothing is reinforced and no credit row
  is claimed.
- `credit`: `audit`, plus explicit credit for accepted `used` marks, using
  the grounded assignment with `signal = max(0.2, 1/(rank+1)) ×
  LM_EXPLICIT_CREDIT_WEIGHT` plus an anchor. Accepted `irrelevant` marks add
  **query-relative** demotion. The node's final score is multiplied by `m =
  1 − (1 − F)·weight·closeness` only for queries that match the marking query's
  anchor (cos ≥ 0.60). Global `usefulness_score`, confidence and retrieval
  weights never change. Positive credit on the same anchor cancels the row.
  Switching back from `credit` turns demotion off at once without touching
  the store.
- `off`: the fields are accepted and ignored. This is exactly the
  pre-feature behaviour, with no hygiene either.
- `LM_IMPLICIT_LINK_POLICY=credited`: implicit feedback links the closing
  trace only to results credited for that event, on any basis. The code
  default `all` is the current behaviour minus irrelevant-marked nodes.

## 2. Per-host steps

Use a neutral working directory (for example `cd ~`) for every Python command.
Both repo roots contain a shadow `living_memory/` package. A script run from
the repo root imports it instead of `src/` (LM 01M11T34DF9RD776EWMA2ME6CV).
The live server is not affected.

Helper scripts committed with this handoff:
- `artifacts/explicit-feedback/handoff/check_rollout_schema.py`: stdlib only,
  opens the store `file:…?mode=ro`, and checks `schema_version` = 8, the two new
  tables, the index, the 12 mark columns and the unchanged ledger CHECK.
  `--expect absent` checks the pre-rollout state.
- `artifacts/explicit-feedback/handoff/smoke_used_mark.py`: needs `fastmcp`.
  It checks the 4 tools and the optional fields on all three tools, runs a
  recall, then marks the top result `used` plus one never-delivered id on the
  next call. It expects `{"accepted": 1, "dropped": 1, "by_reason":
  {"not_delivered": 1}}`. By default the mark rides on a second recall, and
  `--remember TEXT` carries it on a remember instead. **It writes** recall
  events, mark rows and, with `--remember`, a trace.
- `scripts/verify_live_db_migration.py`: a copy-based migration check. It copies
  the store with the backup API from a `mode=ro` source and opens the copy
  with the new code. **Known false red:** `no_legacy_row_stamped` fails on any
  store migrated after 2026-08-24 (it was written for the transport-column
  migration; LM 01M1XB259YR4MJB3C690V0DN0R). Every other check must be `true`.

### 2.0 Rehearsal already done (sfx copy, no live write)

`artifacts/explicit-feedback/handoff/rehearsal/`:
- `verify.json`: `verify_live_db_migration.py` on a backup-API copy of the
  live sfx store (26,125 nodes, 75,021 recall events, 290,284 connections).
  Copy took 2.1 s, migrate-open 0.06 s. All checks `true` except the known
  `no_legacy_row_stamped`. Counts and integrity were preserved, and
  `schema_version` went 8 → 8.
- `check_rollout_schema.py` on the migrated copy: ok. Both tables and the index
  are present, the mark columns are complete and the ledger CHECK is unchanged.
- A sandbox server (rollout code, `LM_EXPLICIT_FEEDBACK_POLICY=audit`, port
  18899, fresh token) on that copy: `smoke-recall.json` and
  `smoke-remember.json` are both `ok: true`. Accepted marks carry `rank` 0 and
  `source_id`. The foreign id was dropped `not_delivered`. `recall_explicit_credit`
  stayed at 0 rows and `recall_credit_ledger` stayed at 10,831, so audit claims
  nothing.
- `scripts/check_deployed_protocol.py` against that sandbox: all 4 tool
  descriptions matched. It reported DRIFT only in the data-dependent `## Memory
  nearby` block of the server instructions, which is expected (LM
  01M3HE9H77T4HJ1XFNE3KZJDZH). Under `LM_EXPLICIT_FEEDBACK_PROMPT=mandatory`,
  the `memory_recall` description also differs by design.

### 2.1 sfx

```bash
cd ~
ROLLOUT=$(git -C /home/sfx/p/lm rev-parse master)      # after the goal merge
git -C /home/sfx/p/lm log --oneline -1 "$ROLLOUT"
git -C /home/sfx/p/lm status --short                   # expect clean

# 1. Rehearse on a fresh copy (optional; already done at b5ee7b6, see 2.0)
python3 /home/sfx/p/lm/scripts/verify_live_db_migration.py \
  ~/.local/share/living-memory/global.sqlite3 --workdir /tmp/efx-verify \
  --json /tmp/efx-verify/verify.json >/dev/null
python3 -c "import json;c=json.load(open('/tmp/efx-verify/verify.json'))['checks'];print({k:v for k,v in c.items() if not v})"
#   expect only {'no_legacy_row_stamped': False}
rm -rf /tmp/efx-verify

# 2. Pre-restart backup (backup API, source read-only; safe while the server runs)
D=$(date -u +%Y%m%d)
python3 - "$D" <<'EOF'
import sqlite3, sys, os
src = os.path.expanduser("~/.local/share/living-memory/global.sqlite3")
dst = os.path.expanduser(f"~/.local/share/living-memory/global.pre-explicit-feedback-{sys.argv[1]}.sqlite3")
r = sqlite3.connect(f"file:{src}?mode=ro", uri=True); w = sqlite3.connect(dst)
r.backup(w); w.close(); r.close(); print(dst, os.path.getsize(dst))
EOF
cp ~/.config/living-memory/env ~/.config/living-memory/env.pre-explicit-feedback.bak

# 3. Pre-state check
python3 /home/sfx/p/lm/artifacts/explicit-feedback/handoff/check_rollout_schema.py \
  ~/.local/share/living-memory/global.sqlite3 --expect absent

# 4. Valves for phase A (section 3): append to the env file
cat >> ~/.config/living-memory/env <<'EOF'
LM_EXPLICIT_FEEDBACK_POLICY=audit
LM_EXPLICIT_FEEDBACK_PROMPT=mandatory
LM_IMPLICIT_LINK_POLICY=credited
EOF

# 5. Restart. The editable install serves /home/sfx/p/lm/src, so the restart is the deploy.
#    SIGTERM was once ignored for 90 s before SIGKILL (LM 01M3C9HCVDQVC8NYNCMA6FJQPP); wait for it.
#    The TLS bridge on 8766 is a separate process: leave it alone.
systemctl --user restart living-memory.service
date -u +%FT%TZ                                         # record as T0_sfx
systemctl --user show living-memory.service -p MainPID -p ActiveEnterTimestamp
tr '\0' '\n' < /proc/$(systemctl --user show -p MainPID --value living-memory.service)/environ \
  | grep -E '^LM_(EXPLICIT|IMPLICIT|QUERY_IRR)'         # confirm the valves are live

# 6. Verify the schema read-only
python3 /home/sfx/p/lm/artifacts/explicit-feedback/handoff/check_rollout_schema.py \
  ~/.local/share/living-memory/global.sqlite3            # expect "ok": true

# 7. Smoke with a used mark (writes 2 recall events + 2 mark rows; add --remember "<deploy note>" to carry it on a remember)
python3 /home/sfx/p/lm/artifacts/explicit-feedback/handoff/smoke_used_mark.py \
  --env-file ~/.config/living-memory/env                 # expect "ok": true
python3 /home/sfx/p/lm/scripts/check_deployed_protocol.py --env-file ~/.config/living-memory/env
#   expected drift: "## Memory nearby" block, and the memory_recall description under PROMPT=mandatory
```

### 2.2 alt (`ssh alt`)

This follows the LM deploy recipe `alt_ae_lm_update_restart` (LM traces
01M1XB259YR4MJB3C690V0DN0R, deploy 180e186 on 2026-09-07;
01M0CHKXET25TAVDS6WY3D7C8Y; 01M1J51TBYYWW0EAHF19SNBH77): transfer a bundle,
fast-forward on alt, run `uv pip install --no-deps --force-reinstall` into
`lm-venv`, then restart. alt's `origin` is GitHub, but the proven route is
the bundle, so do not rely on a push. alt's system `python3` has no
fastmcp or sentence-transformers, so run LM scripts with
`~/.local/share/lm-venv/bin/python3`. The stdlib-only schema check runs with
either interpreter.

```bash
# on sfx
ROLLOUT=$(git -C /home/sfx/p/lm rev-parse master)
ALT_HEAD=$(ssh alt 'git -C ~/p/lm rev-parse HEAD')      # c11ddfa… at writing time
git -C /home/sfx/p/lm merge-base --is-ancestor "$ALT_HEAD" "$ROLLOUT" && echo ff-ok
git -C /home/sfx/p/lm bundle create /tmp/lm-rollout.bundle "$ALT_HEAD..master"
scp /tmp/lm-rollout.bundle alt:/tmp/lm-rollout.bundle

# fast-forward alt's checkout (tracked tree must be clean)
ssh alt "cd ~/p/lm && git status --short --untracked-files=no && \
  git bundle verify /tmp/lm-rollout.bundle && \
  git fetch /tmp/lm-rollout.bundle master:refs/lm-rollout/incoming && \
  git merge --ff-only refs/lm-rollout/incoming && git update-ref -d refs/lm-rollout/incoming && \
  git rev-parse HEAD && rm -f /tmp/lm-rollout.bundle"
rm -f /tmp/lm-rollout.bundle
#   alt HEAD must equal $ROLLOUT

# 1. Rehearse on a copy (optional; writes only /tmp/efx-verify on alt)
ssh alt 'cd ~ && ~/.local/share/lm-venv/bin/python3 ~/p/lm/scripts/verify_live_db_migration.py \
  ~/.local/share/living-memory/global.sqlite3 --workdir /tmp/efx-verify --json /tmp/efx-verify/verify.json >/dev/null; \
  python3 -c "import json;c=json.load(open(\"/tmp/efx-verify/verify.json\"))[\"checks\"];print({k:v for k,v in c.items() if not v})"; \
  rm -rf /tmp/efx-verify'
#   expect only {'no_legacy_row_stamped': False}
#   Caution: until the uv install below, lm-venv still holds the OLD code. Run this after step 3
#   to rehearse the new code, or with PYTHONPATH=~/p/lm/src before it.

# 2. Pre-restart backup + env backup + pre-state check
ssh alt 'cd ~ && D=$(date -u +%Y%m%d) && python3 - "$D" <<EOF
import sqlite3, sys, os
src = os.path.expanduser("~/.local/share/living-memory/global.sqlite3")
dst = os.path.expanduser(f"~/.local/share/living-memory/global.pre-explicit-feedback-{sys.argv[1]}.sqlite3")
r = sqlite3.connect(f"file:{src}?mode=ro", uri=True); w = sqlite3.connect(dst)
r.backup(w); w.close(); r.close(); print(dst, os.path.getsize(dst))
EOF
cp ~/.config/living-memory/env ~/.config/living-memory/env.pre-explicit-feedback.bak
python3 ~/p/lm/artifacts/explicit-feedback/handoff/check_rollout_schema.py \
  ~/.local/share/living-memory/global.sqlite3 --expect absent'

# 3. Install the rollout code into lm-venv
ssh alt '~/.local/bin/uv pip install --python ~/.local/share/lm-venv/bin/python3 --no-deps --force-reinstall ~/p/lm && \
  grep -c "living_memory/irrelevance.py" ~/.local/share/lm-venv/lib/python3.12/site-packages/living_memory-0.1.0.dist-info/RECORD'
#   expect 1. Check RECORD, not diff: a diff of site-packages against src is a false signal (LM 01M1J51TBYYWW0EAHF19SNBH77)

# 4. Valves for phase A (section 3)
ssh alt 'cat >> ~/.config/living-memory/env <<EOF
LM_EXPLICIT_FEEDBACK_POLICY=audit
LM_EXPLICIT_FEEDBACK_PROMPT=mandatory
LM_IMPLICIT_LINK_POLICY=credited
EOF'

# 5. Restart and record T0_alt (alt's local clock is CEST; record UTC)
ssh alt 'systemctl --user restart living-memory.service; date -u +%FT%TZ; \
  systemctl --user show living-memory.service -p MainPID -p ActiveEnterTimestamp; \
  tr "\0" "\n" < /proc/$(systemctl --user show -p MainPID --value living-memory.service)/environ | grep -E "^LM_(EXPLICIT|IMPLICIT|QUERY_IRR)"'

# 6. Verify the schema read-only
ssh alt 'cd ~ && python3 ~/p/lm/artifacts/explicit-feedback/handoff/check_rollout_schema.py \
  ~/.local/share/living-memory/global.sqlite3'           # expect "ok": true

# 7. Smoke with a used mark
ssh alt 'cd ~ && ~/.local/share/lm-venv/bin/python3 ~/p/lm/artifacts/explicit-feedback/handoff/smoke_used_mark.py \
  --env-file ~/.config/living-memory/env'                # expect "ok": true
```

Restart side effects on both hosts: open MCP sessions reconnect with a new
transport session. Ids delivered before the restart cannot be marked
afterwards, so they are dropped as `not_delivered`. That is expected for the
first hour and is visible in `by_reason`. Clients pick up the mandatory
`memory_recall` description in new sessions.

## 3. Valve settings and phases

| env | phase A (at restart) | phase C (after a passing gate) | source |
|---|---|---|---|
| `LM_EXPLICIT_FEEDBACK_PROMPT` | `mandatory` | `mandatory` | `artifacts/explicit-feedback/experiment/result.md`: row **P1** on both stores. The optional arm is poor (F3 fires: C = 0.071 sfx / 0.096 alt < 0.10). Mandatory reaches C = 0.822 / 0.768 with D_used 0.433 [0.30, 0.57] / 0.508 [0.40, 0.61], p = 1e-4, and ritual share 0.032 / 0.074 (≤ 0.50). Cost: +55 chars / +22 tokens in `tools/list`, about +15–30% tokens per run. |
| `LM_EXPLICIT_FEEDBACK_POLICY` | `audit` | `credit` | The same result gives P1 → `credit`. The root goal requires the switch to be gated by live numbers, and the result's own caveat applies: all agreement came from the lookup channel. The grounded channel had 0 evaluable `used` pairs, so non-lookup `used` marks, the only ones where explicit credit adds anything, are unvalidated. Hence phase A collects in `audit` first. |
| `LM_EXPLICIT_CREDIT_WEIGHT` | (unused under audit; leave unset = 1.0) | `1.0` if the live gate's lower CI bound is ≥ 0.10, `0.5` if between 0 and 0.10 | `result.md` W(mandatory): the CI lower bound is 0.30 on sfx and 0.40 on alt, so both map to 1.0. The prereg weight rule (`preregistration.md` §7) is applied to the live gate's CI. |
| `LM_IMPLICIT_LINK_POLICY` | `credited` | `credited` | `artifacts/explicit-feedback/link-policy/report.md`: the preregistered rule R0–R3 passes on both hosts. Δmrr is +0.0024 (sfx) / +0.0014 (alt), and Δhit@3 is +0.0020 / +0.0058, i.e. no loss. Uncredited graph hits per event fall by 22% / 37%, and replay p50 latency by 9% / 8%. This valve is independent of marks, so it is set at restart. |
| `LM_QUERY_IRRELEVANCE_FACTOR` | (inactive under audit; leave unset = 0.5) | `0.5` (default) | `docs/query-irrelevance.md`: a documented default, not a measured constant. One mark gives ×0.75 on the exact query, and a saturated mark gives ×0.5. Move it towards 1.0 if `query_irrelevance.cancels > 0` becomes common, i.e. if demoted nodes later earn credit for the same anchor. `1.0` disables demotion. |

Caveats that belong with these numbers:
- **Link policy only acts on new closes.** The replay's `credited` arm also
  removed 89,478 (sfx) / 40,879 (alt) historical implicit edges that lacked
  credit. The live valve does not delete existing edges, so the measured
  graph-noise reduction arrives gradually as new closes replace history.
  Pruning history would be a separate live-store write, which is not part
  of this rollout and would need its own decision and backup.
- The mandatory arm makes `scripts/check_deployed_protocol.py` report drift on
  the `memory_recall` description. That is intended here. The check's
  constants encode the optional arm.

### Phase A: collect in audit (≥ 7 full UTC days per host)

Set up by §2. Marks are recorded, link hygiene applies, and nothing is
reinforced. Mid-phase sanity check, read-only, per host:

```bash
python3 /home/sfx/p/lm/artifacts/explicit-feedback/handoff/check_rollout_schema.py \
  ~/.local/share/living-memory/global.sqlite3 | grep -A8 marks_by_mark_accepted
```

### Phase B: gate read, each host separately, never merged

```bash
cd ~
# sfx (live store, mode=ro)
PYTHONPATH=/home/sfx/p/lm/src python3 /home/sfx/p/lm/scripts/explicit_feedback_agreement.py \
  --db ~/.local/share/living-memory/global.sqlite3 --host-label sfx --since "<T0_sfx date>" \
  --json /tmp/efx-gate-sfx.json --md /tmp/efx-gate-sfx.md
# alt: snapshot on alt with the backup API (read-only source), copy here, score the copy
ssh alt 'python3 - <<EOF
import sqlite3
r = sqlite3.connect("file:/home/user/.local/share/living-memory/global.sqlite3?mode=ro", uri=True)
w = sqlite3.connect("/tmp/lm-alt-gate.sqlite3"); r.backup(w); w.close(); r.close()
EOF'
scp alt:/tmp/lm-alt-gate.sqlite3 /tmp/lm-alt-gate.sqlite3 && ssh alt rm -f /tmp/lm-alt-gate.sqlite3
PYTHONPATH=/home/sfx/p/lm/src python3 /home/sfx/p/lm/scripts/explicit_feedback_agreement.py \
  --db /tmp/lm-alt-gate.sqlite3 --host-label alt --since "<T0_alt date>" \
  --json /tmp/efx-gate-alt.json --md /tmp/efx-gate-alt.md
python3 -c "
import json,sys
for h in ('sfx','alt'):
    r=json.load(open(f'/tmp/efx-gate-{h}.json')); f=r['falsifier']
    print(h, f.get('better_than_random'), f.get('used_marks'), f.get('used_events'), f.get('placebo_subtracted_evidence_excess_ci95'),
          {k:v.get('share') for k,v in r.get('ritual',{}).items()})"
```

Switch a host to credit (phase C) only if **all** of the following hold for
that host:
1. `falsifier.better_than_random == "pass"`. That requires at least 30 used
   marks on at least 10 events, permutation p < 0.05 and a placebo-subtracted
   CI lower bound > 0. `insufficient` means extend phase A. `fail` means stay
   on `audit`: the falsifier fired, so record a negative result.
2. No ritual detector is `flagged`, i.e. no share is above 0.50
   (`all_delivered_marked_used`, `rank1_only_marking`, `no_closing_work`,
   `session_duplicate_marked`). The per-agent table has no agent type with a
   flag and ≥ 10 used marks.
3. Compliance ≥ 0.10: in the effect metric,
   `days[].marks.events_with_accepted_mark_share` (§5) is at least 0.10.
   This is the F3 line.
4. The operator's caveat is addressed. In the report's "Marks" table (JSON
   `classes.used.grounded`), read the `used` row's `grounded excess`: grounded
   rate minus twin rate, which does not depend on lookup. If it is ≤ 0 on
   `n` ≥ 30 while `evidence` is carried by `lookup`, the pass rests on lookups
   alone. Use weight `0.5` instead of `1.0`, or stay on `audit`. The script
   has no lookup/non-lookup split, and this check was not preregistered. It
   is advisory.

The weight comes from the passing CI lower bound, using the prereg rule
(≥ 0.10 → 1.0; 0 to 0.10 → 0.5). Commit the two gate reports as
`artifacts/explicit-feedback/agreement/{sfx,alt}-gate.{md,json}`.

### Phase C: optional switch to credit, per host

```bash
# sfx (alt: the same edit via ssh, then its restart)
sed -i 's/^LM_EXPLICIT_FEEDBACK_POLICY=audit$/LM_EXPLICIT_FEEDBACK_POLICY=credit/' ~/.config/living-memory/env
echo 'LM_EXPLICIT_CREDIT_WEIGHT=1.0' >> ~/.config/living-memory/env   # or 0.5 per the gate
systemctl --user restart living-memory.service; date -u +%FT%TZ      # record T1_<host>
```

The env file is read only at start, so every valve change needs a restart.
After T1, a `used` mark claims a `recall_explicit_credit` row. Check that
`check_rollout_schema.py` shows the `recall_explicit_credit` count rising.

## 4. AE contract impact

Source: `artifacts/explicit-feedback/core/ae-contract-check.md`. **No AE
change is required.**
- `/home/sfx/p/ae/tests/test_mcp_distribution_contract.sh` pins only the four LM tool
  **names** (Copilot allowlist `memory_lookup, memory_recall, memory_remember,
  memory_teach`, `deferTools: never`). No tool is added or renamed.
- `/home/sfx/p/ae/tests/e2e/opencode-node/harness/stub/fake_lm.js` declares no schema. It
  answers every `tools/call` with `{"results": []}` and ignores arguments.
- The new parameters are optional on the real FastMCP schema
  (`tests/test_explicit_feedback.py::test_real_fastmcp_schema_carries_optional_fields`,
  plus the smoke test's `optional_used_irrelevant_fields` check).
- Optional future work, not a compatibility fix: AE protocol text that tells
  its agents to send marks, or a `fake_lm.js` that echoes `feedback_marks`.
  Under `PROMPT=mandatory` the instruction already reaches every agent
  through the `memory_recall` description.

## 5. After measurement

Procedure: `artifacts/explicit-feedback/effect-metric/after-procedure.md`,
script `scripts/recall_effect_daily.py`, per host with the same defaults and
exclusion layers as the baseline.
- `T0_<host>` = the UTC restart timestamps recorded in §2.1 step 5 and §2.2
  step 5. `SINCE` = the first full UTC day after T0. The window is ≥ 7 full days.
- If phase C happens at `T1_<host>`, report `[SINCE, T1)` (audit) and `[T1,
  UNTIL]` (credit) as separate windows.
- Outputs: `artifacts/explicit-feedback/effect-metric/{sfx,alt}-after.{md,json}`.
  Store the per-host after numbers in LM, scope `project:living-memory`.

Before numbers (kept live traffic, all events; from
`artifacts/explicit-feedback/effect-metric/README.md` and `{sfx,alt}-before.md`):

| host | window | events | rank-1 used | top-3 used | useful nodes | never-closed |
|---|---|---:|---:|---:|---:|---:|
| sfx | 07–13.09 | 3476 | 8.7% | 19.9% | 11.2% | 19.3% |
| sfx | 14–20.09 | 3310 | 9.0% | 20.6% | 8.9% | 22.4% |
| sfx | 21–27.09 | 3107 | 5.1% | 16.7% | 8.0% | 28.1% |
| sfx | 07–27.09 total | 9893 (2604 A/B excluded) | 7.7% | 19.1% | 9.3% | 23.1% |
| alt | 07–11.09 | 3499 | 11.4% | 22.1% | 11.1% | 13.0% |
| alt | 23–27.09 | 1176 | 13.0% | 25.3% | 12.9% | 5.0% |
| alt | 07–27.09 total | 4675 (0 excluded) | 11.8% | 22.9% | 11.6% | 11.0% |

alt has no events between 12.09 and 22.09. Week-to-week spread under
`audit`, where no move is expected: sfx r1 5–9%, useful 8–11%; alt r1
11–13%, useful 11–13%. Under `audit`, `per_basis.explicit` must stay zero.
Compare `marked_used`, `used_or_marked` and `never_closed.marked_used` to see
whether marks reach the never-closed events that grounded credit cannot
reach.

Agreement reference rates (placebo, before any mark; from
`artifacts/explicit-feedback/agreement/README.md`), grounded vs same-cosine
twin:

| host | events | r1 grounded / twin / excess | top-3 grounded / twin / excess |
|---|---:|---|---|
| sfx | 7027 | 8.7% / 2.3% / +6.4 pp | 7.1% / 1.6% / +5.5 pp |
| alt | 3170 | 13.9% / 2.6% / +11.3 pp | 11.1% / 2.2% / +8.9 pp |

## 6. Rollback

Pick the lightest level that fixes the problem. Each level needs a restart,
because the env file is read at start.

1. **Stop reinforcement only** (from phase C): `LM_EXPLICIT_FEEDBACK_POLICY=audit`,
   then restart. Query-irrelevance demotion stops at once, because reads are
   gated on `credit`. Explicit credit already applied (usefulness, anchors,
   `recall_explicit_credit` rows) stays. It is bounded by the weight and
   decays like any credit.
2. **Back to pre-feature behaviour, new code kept:**
   `LM_EXPLICIT_FEEDBACK_POLICY=off`, `LM_EXPLICIT_FEEDBACK_PROMPT=optional`
   (tool descriptions byte-identical to before), `LM_IMPLICIT_LINK_POLICY=all`.
   Or simply restore the env file:
   `cp ~/.config/living-memory/env.pre-explicit-feedback.bak ~/.config/living-memory/env`.
   Then restart. The new tables stay and are not read.
3. **Code rollback:**
   - sfx: `git -C /home/sfx/p/lm revert --no-edit -m 1 <rollout merge sha>`,
     then `systemctl --user restart living-memory.service`. The editable
     install picks up the tree.
   - alt: `ssh alt 'git -C ~/p/lm worktree add /tmp/lm-rollback c11ddfa && ~/.local/bin/uv pip install --python ~/.local/share/lm-venv/bin/python3 --no-deps --force-reinstall /tmp/lm-rollback && systemctl --user restart living-memory.service && git -C ~/p/lm worktree remove /tmp/lm-rollback'`.
   - Old code ignores `recall_feedback_marks`, `recall_explicit_credit` and
     `query_irrelevance`. No DDL was changed, so no schema undo is needed.
     Old code also ignores the new env vars, but remove them anyway (step 2).
4. **Store restore** (last resort, data loss): only if the store is damaged.
   Stop the service, move `global.sqlite3{,-wal,-shm}` aside, copy
   `global.pre-explicit-feedback-<D>.sqlite3` to `global.sqlite3`, then
   start. Every write since the backup is lost. Levels 1–3 never need this.

After any rollback, record in LM (`project:living-memory`) which level was
used, why, and the UTC time. The after-report window must end there.
