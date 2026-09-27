# "After" report: procedure (operator-gated)

The after report can only be produced after the explicit-feedback server change has been rolled out. The rollout adds `recall_feedback_marks`, the explicit credit basis and the valves, and it needs a restart, which is the operator's decision. Produce the report per host, with the same script, the same defaults and the same exclusion layers as the baseline. Never merge the two hosts.

## 0. Fix the dates

- `T0_<host>`: the UTC timestamp of that host's LM restart onto the rollout commit (from the rollout handoff, `artifacts/explicit-feedback/handoff/rollout.md`).
- `SINCE`: the first full UTC day after `T0_<host>`.
- `UNTIL`: the last full UTC day before the report is run. Use a window of at least 7 full days, the same length as the baseline weeks it is compared against.
- If `LM_EXPLICIT_FEEDBACK_POLICY` is switched from `audit` to `credit` at some later time `T1`, report `[SINCE, T1)` and `[T1, UNTIL]` as separate windows.

## 1. sfx

```bash
cd /home/sfx/p/lm            # or a worktree on the rollout commit
SINCE=YYYY-MM-DD UNTIL=YYYY-MM-DD
python3 scripts/recall_effect_daily.py \
  --db ~/.local/share/living-memory/global.sqlite3 \
  --host-label sfx --since "$SINCE" --until "$UNTIL" \
  --receipts-root /home/sfx/p/ae/artifacts \
  --json artifacts/explicit-feedback/effect-metric/sfx-after.json \
  --md   artifacts/explicit-feedback/effect-metric/sfx-after.md
```

The live DB is opened as `file:...?mode=ro` only. To work from a frozen copy instead, add `--snapshot-to /tmp/lm-sfx-after.sqlite3`. That copies the DB with `living_memory.retrieval_harness.backup_database` and reads the copy.

## 2. alt

Take the snapshot on alt with the SQLite backup API, reading the source read-only. Never open alt's live DB for writing.

```bash
ssh alt 'python3 - <<EOF
import sqlite3
r = sqlite3.connect("file:/home/user/.local/share/living-memory/global.sqlite3?mode=ro", uri=True)
w = sqlite3.connect("/tmp/lm-alt-after.sqlite3")
r.backup(w); w.close(); r.close()
EOF'
scp alt:/tmp/lm-alt-after.sqlite3 /tmp/lm-alt-after.sqlite3
ssh alt rm -f /tmp/lm-alt-after.sqlite3
sha256sum /tmp/lm-alt-after.sqlite3          # record it in the report README

# alt harness receipts (was 0 capture dirs at baseline); pass them if any appear
ssh alt 'find ~/p/ae/artifacts -type d -name "capture-memory-*" | wc -l'
#   if > 0: rsync -a --include="*/" --include="*.headers.json" --include="corpus_manifest.json" \
#             --exclude="*" alt:p/ae/artifacts/ /tmp/alt-ae-receipts/  and add --receipts-root /tmp/alt-ae-receipts

python3 scripts/recall_effect_daily.py \
  --db /tmp/lm-alt-after.sqlite3 \
  --host-label alt --since "$SINCE" --until "$UNTIL" \
  --receipts-root /home/sfx/p/ae/artifacts \
  --json artifacts/explicit-feedback/effect-metric/alt-after.json \
  --md   artifacts/explicit-feedback/effect-metric/alt-after.md
```

The sfx `--receipts-root` is still passed for alt: it supplies the fixture goal families from the harness corpus manifests. Its session ids cannot collide with alt's.

## 3. What to compare (per host, before vs after)

The before files are `<host>-before.json`. Compare the same-length weekly windows.

1. `days[].metrics.used`: `rank1_used`, `top3_used`, `useful_node_share` (grounded+lookup, the like-for-like primary). A rise here under `credit` is the reinforcement effect. Under `audit` it should not move beyond the baseline week-to-week spread (sfx 5–9% r1, 8–11% useful; alt 11–13% r1, 11–13% useful).
2. `days[].metrics.per_basis.explicit`: explicit credit. It stays zero under `audit`, which records marks without claiming ledger rows.
3. `days[].metrics.marked_used` and `used_or_marked`, plus `days[].marks`: mark volume (`events_with_accepted_mark_share`, i.e. compliance), accepted vs rejected marks, and `via_tool`. `used_or_marked` is the "used" signal including agent marks.
4. `days[].never_closed.marked_used`: whether marks reach the never-closed events (sfx 23%, alt 11% of events in the baseline) that grounded credit cannot reach.
5. `exclusion[]`: the per-day A/B exclusion breakdown. The explicit-feedback experiment runs in a SANDBOX server and must produce **zero** events in the live stores. Any sandbox, harness or fixture traffic that does show up in a live store is listed there with its reason and removed from the metric.

Agreement of marks with placebo-subtracted grounded evidence is a separate metric (`scripts/explicit_feedback_agreement.py`). This report only measures the effect.

## 4. Commit

```bash
git add artifacts/explicit-feedback/effect-metric/{sfx,alt}-after.{md,json}
git commit -m "Explicit feedback effect: after report <SINCE>..<UNTIL> (sfx, alt)"
```

Store the per-host after numbers in Living Memory, scope `project:living-memory`.
