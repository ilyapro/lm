# confirmatory-holdout-v4 accrual campaign — operator runbook

This document tells you how to arm the v4 accrual campaign, and what you are
committing to when you do. Read all of it before you run anything.

Nothing in this repository arms the campaign. `scripts/v4_cadence.py` renders a
cadence unit and checks it; it has no install, enable, or start command. The
commands in [Arming](#4-arming) are the only ones that begin the operation, and
they are yours to issue.

---

## 1. What you are committing to

The campaign runs **29 daily slots**. Their instants are frozen in
`artifacts/animal-planet/evaluation/confirmatory-holdout-v4/POLICY.md` and
cannot be moved, extended, shortened, or re-anchored by anything observed later:

| | |
| --- | --- |
| Slot 0 opens | **2026-08-17T00:00:00Z** |
| Slot 0 closes | **2026-08-17T06:00:00Z** |
| Slot *i* opens | slot 0 + *i* × 86400 seconds, for *i* = 0 … 28 |
| Grace window | 21600 seconds, half-open `[opens, opens + 21600)` |
| Final selection slot | **2026-09-14T00:00:00Z** |
| Absolute horizon | **2026-09-14T06:00:00Z** |

Print the full derived schedule at any time:

```sh
python3 -B scripts/v4_cadence.py schedule
```

### The cost of a missed slot

**Missing any due slot ends the campaign terminally.** If a slot's window closes
without a validator-valid receipt, the driver records an append-only `missed`
marker and the namespace enters terminal `schedule-integrity-failure`. From that
point no later probe or seal is authorized.

There is **no catch-up, no backfill, no interpolation, no replacement snapshot,
no late launch, and no discretionary skip.** A missed slot cannot be re-run the
next hour, the next day, or at the end. A correction requires parent escalation
and, if authorized at all, an entirely new namespace — the v4 identifiers,
reader IDs, and sealing authority are permanently consumed.

This applies to slot 0 exactly as it applies to slot 28. If you arm the campaign
late, or the machine is asleep at 2026-08-17T00:00:00Z, or the work directory is
not writable at that moment, the whole v4 namespace is finished before it opened
a single source.

Treat arming as a one-way door. If you are not certain the host will be awake,
powered, and reachable for 29 consecutive days, do not arm.

---

## 2. Why the cadence has this shape

`POLICY.md` states that "every slot is a distinct invocation. No process may
sleep, poll, or remain alive waiting across slots, and an orchestration retry is
not a new cadence event."

So there is deliberately no supervisor daemon. An external timer starts a fresh,
short-lived process for each slot, and that process exits when its slot is done:

```
/usr/bin/python3 -B /path/to/lm/scripts/ap_confirmatory_slot_v4.py --work-dir <WORK_DIR> --slot-index <N> --campaign-module <CAMPAIGN_MODULE>
```

One invocation, one slot, then exit. The driver reads the real system clock and
has no flag that can tell it what time it is: an invocation before its slot
opens is refused before any source is touched, and an invocation after the grace
window closes records the terminal missed-slot marker.

---

## 3. Before you arm

### 3.1 Choose a work directory outside the repository namespace

`--work-dir` is **required** and must resolve **outside**
`artifacts/animal-planet/evaluation/confirmatory-holdout-v4/`. That namespace is
byte-locked; both the renderer and the driver refuse a work root inside it.

It must also live outside any disposable checkout. The append-only ledger has to
survive 29 days of merges, branch deletions, and worktree cleanup, so do not
point it at a `.worktrees/` path or anything else you might remove. A durable
location under your home directory or `/var/lib` is appropriate.

```sh
mkdir -p "$HOME/v4-campaign"
```

### 3.2 Provide the campaign module

`--campaign-module` is the operator-private module that supplies alias locators,
database observations, and the runtime observer. It is deliberately not tracked
in this repository: no tracked file may name a source. Place it outside the repo
as well.

### 3.3 Render your own cadence unit

`scripts/v4-cadence/` holds an **example** render whose paths point at
`/nonexistent/EXAMPLE-*`. Never arm it. Render your own:

```sh
python3 -B scripts/v4_cadence.py render \
  --out-dir "$HOME/v4-cadence" \
  --repo-root "$(pwd)" \
  --work-dir "$HOME/v4-campaign" \
  --campaign-module "$HOME/v4-campaign-private/campaign.py" \
  --cron-timezone "$(cat /etc/timezone)"
```

This writes a systemd user timer/service pair under `$HOME/v4-cadence/systemd/`
and an equivalent crontab fallback at `$HOME/v4-cadence/cron/`. Rendering
installs nothing.

### 3.4 Verify and preflight

```sh
python3 -B scripts/v4_cadence.py verify --unit-dir "$HOME/v4-cadence"
python3 -B scripts/v4_cadence.py preflight --unit-dir "$HOME/v4-cadence"
```

`verify` re-derives the schedule from the frozen constants and proves the units
match it. `preflight` additionally checks this host: that the interpreter, the
driver, the campaign module, and the work directory all exist; that the crontab
was rendered for the timezone the cron daemon actually runs in; and that slot 0
has not already opened.

**Both must report `"ok": true` before you continue.** They exit non-zero
otherwise.

---

## 4. Arming

### 4.1 systemd user timer (preferred)

```sh
install -d "$HOME/.config/systemd/user"
install -m 0644 "$HOME/v4-cadence/systemd/"ap-confirmatory-v4-slot@*.{service,timer} \
  "$HOME/.config/systemd/user/"

# A user manager is stopped when your last session ends. Without lingering, the
# timers do not run while you are logged out and every slot in between is lost.
loginctl enable-linger "$USER"

systemctl --user daemon-reload
systemctl enable --now --user ap-confirmatory-v4-slot@{0..28}.timer
```

That last command is the one that arms the campaign.

Confirm all 29 firings are scheduled, and that the first is 2026-08-17T00:00:00Z:

```sh
systemctl --user list-timers --all 'ap-confirmatory-v4-slot@*'
```

### 4.2 crontab fallback

Only if the host has no systemd user manager. It is strictly worse: cron
discards each firing's output unless an MTA is configured, whereas the systemd
form records every firing in the journal.

Two cron limitations are already handled by the render, and you must not undo
them by hand. Debian cron has no per-crontab timezone — `CRON_TZ` is ignored,
see `crontab(5)` LIMITATIONS — so each entry is written in the cron daemon's own
local time; on a `+07` host, midnight UTC is written as `0 7`. And cron has no
year field, so every entry carries an explicit UTC guard that lets it run in
exactly one hour of exactly one day. **Re-render if the host timezone changes.**

Installing a crontab **replaces your entire existing crontab**. Back it up
first: `crontab(1)`'s list flag (`-l`) prints the current table — redirect that
to a file, then append the rendered entries to a copy of it, and install the
merged file:

```sh
crontab "$HOME/v4-cadence/cron/ap-confirmatory-v4.crontab"
```

---

## 5. During the campaign

Do not restart, re-run, or "retry" a slot. An orchestration retry is not a
cadence event, and the units refuse a manual start for exactly that reason
(`RefuseManualStart=yes`). The first validator-valid resolution of a slot is
immutable.

Watch, but do not intervene:

```sh
systemctl --user list-timers --all 'ap-confirmatory-v4-slot@*'
journalctl --user -u 'ap-confirmatory-v4-slot@*' --since today
ls "$HOME/v4-campaign/ledger"
```

Each invocation prints one JSON summary and exits. Exit statuses:

| Status | Meaning |
| --- | --- |
| 0 | the slot resolved, or handed off to the sealer |
| 1 | integrity failure |
| 2 | usage error |
| 3 | the work directory resolved inside the frozen namespace |
| 4 | terminal schedule-integrity failure — the campaign is over |
| 5 | launched before the slot opened; nothing was touched |
| 6 | the slot already had a validator-valid resolution |

If you ever see status 4, the campaign has ended terminally. Do not re-arm, do
not render a replacement schedule, and do not start a new namespace on your own
authority: escalate to the parent goal.

Keep the machine awake. A suspended or powered-off host at a slot instant is a
missed slot, and `Persistent=false` means systemd will **not** replay it on
resume — by design, because replaying it would be backfill.

### Keep the serving runtime frozen

A missed slot is not the only way to lose the campaign. The floors must all be
met inside **one attested runtime segment** — the runtime pinned by
`control-watermark.json`, whose serving build, sanitized configuration, and
effective flag values were observed when the boundary instant was taken. Two
kinds of runtime change during the 29 days have consequences, and they are not
the same consequence.

**A serving-build or service-tuple change closes the segment and destroys the
counts.** If the attested build's commit, tree, or module hashes change, or the
observed service tuple changes for any other reason, the frozen POLICY applies:
*"Captured snapshots and provisional counts are destroyed and no result from
that segment may contribute later"*, and in the successor segment *"Counts start
at zero"*. Nothing is unioned or carried forward. Restarting the Living Memory
service onto a different build on day 20 does not cost you a slot — it costs you
all twenty days of accrual.

The successor segment is not free either. It requires a fresh attestation whose
ceremony *"must start within the half-open 30 seconds after the closure and
finish before the next due slot"*, and *"A late, missing, failed, or invalid
attempt is terminal runtime-attestation failure with no alternate boundary."*
The driver dispatches that ceremony through your campaign module's
`attest_successor_segment`; if your module cannot complete it inside that
window, the campaign is over.

**Setting either legacy repeat control to true ends the campaign outright.**
`LM_RECALL_REPEAT_GATING` and `LM_RECALL_REPEAT_DROP_TRAILING_STUBS` must both
observe as literally `false` on every attested service. A true value is not a
"change" that closes the segment — it makes the observation *malformed*, and
*"An unavailable or malformed observation is terminal, never a closure."* There
is no re-attestation path back. Do not experiment with those two flags on any
serving instance while the campaign is armed, on either host.

So, for the whole 29 days: no upgrades, no redeploys, no service replacement, no
configuration edits, and no flag experiments on the attested runtimes. If you
have a Living Memory change you want serving, land it **before** you arm, or
wait until after 2026-09-14T06:00:00Z.

---

## 6. Disarming

Before slot 0 opens, disarming costs nothing:

```sh
systemctl disable --now --user ap-confirmatory-v4-slot@{0..28}.timer
```

After slot 0 opens, disarming ends the campaign terminally, exactly as a missed
slot does. There is no pause.

Once the campaign has finished — at the absolute horizon 2026-09-14T06:00:00Z,
or earlier on a terminal outcome — disarm as above, and if you used the crontab
fallback, remove its entries: cron has no year field, so the guarded entries
would otherwise sit dormant in your table for another year.

---

## 7. Related files

| Path | What it is |
| --- | --- |
| `scripts/v4_cadence.py` | renders and verifies the cadence unit; never arms it |
| `scripts/v4-cadence/` | example render, not for arming |
| `scripts/ap_confirmatory_slot_v4.py` | the one-shot-per-slot driver each firing invokes |
| `scripts/ap_confirmatory_snapshot_v4.py` | the read-only source snapshot launcher |
| `artifacts/animal-planet/evaluation/confirmatory-holdout-v4/POLICY.md` | the frozen protocol these instants come from |
