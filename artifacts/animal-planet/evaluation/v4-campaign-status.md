# confirmatory-holdout-v4 campaign status

Status: **rendered, not armed, not started**.

The v4 accrual campaign has **executed zero slots**. It has opened zero sources,
consumed zero authorities, produced zero probe receipts, written zero ledger
entries, and evaluated no floor. The machinery to run it now exists and is
hash-bound into the packet manifest; the cadence unit is rendered and verified
and deliberately **not** activated. Arming is an operator decision that has not
been taken.

This record replaces the unreachable seal-or-floor-gap contract, which required
either a sealed packet or a measured floor gap. Neither can exist: no slot has
run, so there is nothing to seal and nothing to measure. The machine-readable
form is [`v4-campaign-status.json`](v4-campaign-status.json), and
`tests/test_v4_campaign_status.py` falsifies it against disk.

## What is true, and what is simply not observed

These are two different kinds of statement and this record keeps them apart.

**Action counts are 0** — the campaign never took these actions, and the absence
of any ledger settles it:

| Action | Count |
|---|---:|
| Slots executed | 0 |
| Sources opened | 0 |
| Authorities consumed | 0 |
| Probe receipts produced | 0 |
| Ledger entries written | 0 |
| Floor evaluations performed | 0 |
| Snapshots captured | 0 |
| Packet publication attempts | 0 |
| Semantic reads | 0 |

**Measurements are `not observed`** — never `0`. A quantity the campaign never
measured has no zero-valued reading; reporting one would be a fabricated
measurement. Every floor count, the readiness status, the probe receipt status,
the floor gap, the snapshot-set digest, the segment identity, and the packet
seal state are all recorded as the sentinel string `not observed`.

The preregistered floor thresholds *are* reproduced — from the frozen
`analysis-plan.json` `readiness` block — but as **requirements, not readings**.
They live in their own `preregistered_floor_thresholds` block, and every one of
them is paired with a `not observed` counterpart. The holdout floor requires 30
unseen-in-dev repeated automatic families, 150 such events, 30 distinct
components containing a qualifying family, 200 organic events, 30 distinct
source-qualified organic sessions, 30 distinct components containing an organic
event, and at least 2 distinct requested project scopes. The shadow floor
requires 30 genuine non-synthetic real workflows, 100 unique replayable logical
calls, 30 distinct components containing a counted workflow, and 2 project
scopes. **Whether any of these is met is unknown.** The plan's initial readiness
status is `unprobed`, and unprobed is what it remains.

There is **no ledger on disk**. No work directory has been created; the ledger
would live at `<work-dir>/ledger/NNNNNNNN.json` with markers at
`<work-dir>/markers/`, in an operator-chosen location outside this repository.
The test asserts this claim against an actual scan, so a ledger appearing later
without this record changing falsifies the record rather than being absorbed by
it.

## The schedule you would be committing to

| | |
| --- | --- |
| Slot 0 opens | **2026-08-17T00:00:00Z** |
| Slot 0 closes | **2026-08-17T06:00:00Z** |
| Slot *i* opens | slot 0 + *i* × 86400 seconds, for *i* = 0 … 28 |
| Grace window | 21600 seconds, half-open `[opens, opens + 21600)` |
| Slots | **29** |
| Final selection slot | **2026-09-14T00:00:00Z** |
| Absolute horizon | **2026-09-14T06:00:00Z** |

The allocation is calendar-only and source-blind: it is not derived from source
availability or from anything observed later.

### A missed slot 0 ends the v4 namespace terminally

**Missing any due slot ends the campaign terminally**, and this applies to slot
0 exactly as it applies to slot 28. If a slot's window closes without a
validator-valid receipt, the driver records an append-only `missed` marker and
the namespace enters terminal `schedule-integrity-failure` (driver exit status
`4`). From that point no later probe or seal is authorized.

There is **no catch-up, no backfill, no interpolation, no replacement snapshot,
no late launch, and no discretionary skip.** If the campaign is armed late, or
the host is asleep at 2026-08-17T00:00:00Z, or the work directory is not
writable at that moment, the whole v4 namespace is finished before it opened a
single source. A correction requires parent escalation and, if authorized at
all, an entirely new namespace — the v4 identifiers, reader IDs, and sealing
authority are permanently consumed.

Treat arming as a one-way door. If you are not certain the host will be awake,
powered, and reachable for 29 consecutive days, do not arm.

### The serving runtime must stay frozen too

All floors must be met inside **one attested runtime segment**, so the schedule
is not the only commitment. A change to the attested serving build or service
tuple **closes the segment**: the frozen POLICY destroys the captured snapshots
and provisional counts, no result from that segment may contribute later, and
the successor segment starts its counts at zero — an upgrade on day 20 costs all
twenty days, not one slot. The successor attestation must then start within 30
seconds of the closure and finish before the next due slot, or the campaign ends
in terminal runtime-attestation failure.

Worse, setting either `LM_RECALL_REPEAT_GATING` or
`LM_RECALL_REPEAT_DROP_TRAILING_STUBS` to a true value on an attested service is
**not** a closure at all. Both must observe as literally `false`; a true value
makes the observation malformed, and a malformed observation is terminal with no
re-attestation path. For the whole 29 days: no upgrades, no redeploys, no
configuration edits, and no flag experiments on either attested runtime.

## Machinery that now exists

All of it is hash-bound into the packet manifest under the binding key
`runtime_observer_aggregate_probe_and_accrual_ledger_with_tests`, by path,
sha256, and bytes. The independent verifier settles the whole tool set against
disk, so an unbound or drifted tool fails closed as `pin`.

| Path | Role | SHA-256 |
|---|---|---|
| `scripts/ap_confirmatory_snapshot_v4.py` | read-only source snapshot launcher | `de5214d4c6c0a979c59c6376754f0e90fccf7d8eae7ec2ee6973339c61427a43` |
| `scripts/ap_confirmatory_slot_v4.py` | one-shot-per-slot accrual driver | `bcc99f9ae646775e1be5e2fdc8aa852943dcf8e29e2067044e4f0572cdb2d418` |
| `scripts/v4_cadence.py` | cadence renderer and verifier; never arms | `0511c2bcdad960259d253931b065e650f0a6fa88fe9130f8b152e2eacd81c70b` |
| `scripts/ap_confirmatory_probe_v4.py` | frozen aggregate probe | `c6b308b11126416617d09cb0e555367137269a3f7afc7602cd538c18eff664d6` |
| `scripts/ap_confirmatory_accrual_v4.py` | append-only accrual ledger | `aa918a42ff5f6b31dc621f168341cb357a08720a225c77450f5053c79cd18765` |
| `scripts/ap_confirmatory_runtime_v4.py` | runtime observer | `b6026afd3188fc64f6a05a92f8d7918cf0a8881a16cb2e5371ebf3cb4425fb6f` |
| `scripts/ap_confirmatory_seal_v4.py` | no-overwrite seal launcher | `0fa22d190fcc815053820f51f16af16511666e1d3e571a0d4b0347fa2c746082` |
| `tests/test_ap_confirmatory_snapshot_v4.py` | snapshot launcher suite | `a2cfd3ba9b36e4b9f2a8414342178078fb82c676bdf83e89f7fef7ae01d7b183` |
| `tests/test_ap_confirmatory_slot_v4.py` | slot driver suite | `1d7cdf525fbcc69d4d3c853ac674457bdac5109da1a4ed328d3f171fe60d1ced` |
| `tests/test_v4_cadence.py` | cadence renderer suite | `0845456875357cd6d79872b4b5bab97c6924fd11bcc5a0ed90dac5af465a2f98` |

The packet builder and independent verifier that carry those bindings are
`artifacts/animal-planet/evaluation/confirmatory-holdout-v4/recipe/build.py`
(`e2932b3a…`) and `recipe/verify.py` (`475f8bc8…`).

That enumeration is closed over a **name-shape** discovery rule — any
`scripts/ap_confirmatory_*_v4.py`, `scripts/v4_*.py`,
`tests/test_ap_confirmatory_*_v4.py` or `tests/test_v4_*.py` counts as a campaign
tool and must be bound, or the builder fails closed with
`campaign_tool_enumeration_not_closed`. That net is deliberately conservative,
and it is aimed at tooling that touches evidence. This record's test touches
none: it reads tracked artifacts, scans for absence, and runs the release
validator. It is therefore named `tests/test_campaign_status_v4.py`, outside the
campaign-tool shape, rather than being enumerated as a campaign tool it is not.

### The cadence unit is rendered, not activated

29 systemd timer units, 1 service template, and 1 crontab fallback are rendered
under `scripts/v4-cadence/`. That render is an **example** whose paths point at
`/nonexistent/EXAMPLE-*`; it is not armable and must never be armed. Each firing
spawns one fresh short-lived process for one slot — no process sleeps or polls
across slots. The service sets `RefuseManualStart=yes`, because an orchestration
retry is not a cadence event, and the timers set `Persistent=false`, because
replaying a missed firing on resume would be backfill.

## Operator arming procedure

Reproduced from [`docs/v4-campaign-runbook.md`](../../../docs/v4-campaign-runbook.md)
(sha256 `f1280d015ab5950840bd4190d73d976e06230d09ab5d8b49307247d681f8c654`).
Nothing in this repository arms the campaign; these commands are the operator's
to issue. Read the runbook in full first.

1. **Choose a durable work directory outside the frozen namespace.** `--work-dir`
   is required and must resolve outside
   `artifacts/animal-planet/evaluation/confirmatory-holdout-v4/`, and outside any
   disposable checkout — the append-only ledger must survive 29 days of merges,
   branch deletions, and worktree cleanup.

   ```sh
   mkdir -p "$HOME/v4-campaign"
   ```

2. **Provide the operator-private campaign module.** `--campaign-module` supplies
   alias locators, database observations, and the runtime observer. It is
   deliberately untracked: no tracked file may name a source.

3. **Render your own cadence unit.** Rendering installs nothing.

   ```sh
   python3 -B scripts/v4_cadence.py render \
     --out-dir "$HOME/v4-cadence" \
     --repo-root "$(pwd)" \
     --work-dir "$HOME/v4-campaign" \
     --campaign-module "$HOME/v4-campaign-private/campaign.py" \
     --cron-timezone "$(cat /etc/timezone)"
   ```

4. **Verify and preflight.** Both must report `"ok": true`; they exit non-zero
   otherwise.

   ```sh
   python3 -B scripts/v4_cadence.py verify --unit-dir "$HOME/v4-cadence"
   python3 -B scripts/v4_cadence.py preflight --unit-dir "$HOME/v4-cadence"
   ```

5. **Install the units.**

   ```sh
   install -d "$HOME/.config/systemd/user"
   install -m 0644 "$HOME/v4-cadence/systemd/"ap-confirmatory-v4-slot@*.{service,timer} \
     "$HOME/.config/systemd/user/"
   ```

6. **Enable lingering.** Without it the timers do not run while you are logged
   out and every slot in between is lost.

   ```sh
   loginctl enable-linger "$USER"
   ```

7. **Arm.** This is the command that begins the operation.

   ```sh
   systemctl --user daemon-reload
   systemctl enable --now --user ap-confirmatory-v4-slot@{0..28}.timer
   ```

8. **Confirm** all 29 firings are scheduled and the first is 2026-08-17T00:00:00Z.

   ```sh
   systemctl --user list-timers --all 'ap-confirmatory-v4-slot@*'
   ```

The crontab fallback (`crontab "$HOME/v4-cadence/cron/ap-confirmatory-v4.crontab"`)
is only for hosts with no systemd user manager. Installing a crontab replaces
your entire existing crontab — back it up with `crontab -l` and install a merged
file. Debian cron ignores `CRON_TZ`, so entries are written in the cron daemon's
local time and must be re-rendered if the host timezone changes; and because
cron has no year field, each entry carries an explicit UTC guard.

Driver exit statuses: `0` resolved or handed off to the sealer · `1` integrity
failure · `2` usage error · `3` work directory inside the frozen namespace ·
`4` terminal schedule-integrity failure, the campaign is over · `5` launched
before the slot opened, nothing touched · `6` the slot already had a
validator-valid resolution.

Before slot 0 opens, disarming costs nothing. After slot 0 opens, disarming ends
the campaign terminally, exactly as a missed slot does. There is no pause.

## P6 and P7 disposition

**Unconfirmed. Default-off release retained. No promotion.**

The automatic-versus-agent repair is **unimplemented and unconfirmed**. No
evidence exists that would confirm it, and none is claimed here.
`LM_RECALL_REPEAT_GATING` and `LM_RECALL_REPEAT_DROP_TRAILING_STUBS` both remain
**default-off**. There is no operational promotion.

This is exactly the action the frozen plan preregisters for this situation —
`promotion.negative_invalid_inconclusive_timeout_or_missing_gate_action`:
*"retain default-off release and escalate; no operational promotion."* The plan
also fixes that the default-off release remains shippable, that promotion would
require all confirmation gates to pass conjunctively, and that the serving
accrual build is not a promotable candidate.

The sole authorized replacement P6 evaluation was consumed and no retry is
permitted, which is why the confirmatory campaign exists at all — and why its
non-execution leaves P6/P7 where they are rather than moving them.

*(P6/P7 here are the parent goal targets recorded in `final-report.json`
`overall.parent_targets`, not the path SPEC item numbering of this scope.)*

## What IS confirmed on held-out evidence

The deferral above is scoped to the **automatic lane only**. It is not a failure
of the whole goal. These outcomes are confirmed on held-out evidence and are
unaffected by the campaign's non-execution:

| Outcome | Observed | Gate |
|---|---|---|
| Payload reduction, original holdout | median ratio `0.517323`, p90 ratio `0.497176` | PASS |
| Top-result retention | `1.0` | PASS |
| Useful-feedback retention | `1.0` | PASS |
| Cross-scope admission reduction, original holdout | `0.598460` | PASS |
| Same-scope retention, original holdout | `0.957154` | PASS |
| Correction-dominance violations | `0` (26 focused tests passed, 0 failed) | PASS |
| Era safety | 45 focused tests passed, 0 failed | PASS |
| Shadow latency | p50 `+0.019807`, p95 `-0.075293` (threshold `<=0.10`) | PASS |

Recorded fidelity limits still apply: the preserved payload comparison rests on
48 transcript-matched original-holdout events; correction replay contains zero
candidate pairs, so the focused regression suite is the substantive dominance
evidence; and shadow latency is a heterogeneous 35-call workflow sample per
version.

## Control arm

The hash-pinned gating-off control arm is intact and unmodified.
`python3 -B scripts/ap_release.py validate` exits `0`, hash-binding all 21
`src/living_memory/*.py` modules. Nothing under `src/living_memory/` was
modified to produce this record.

## Scope of this record

This record is **aggregate-only**. It claims no floor result, no readiness
status, and no packet seal. It does not activate the cadence unit and does not
execute a slot. It lives outside the frozen `confirmatory-holdout-v4/`
namespace, which it does not modify.
