# Cold-quota live wire — operator runbook (arm, verify, read, disarm, rollback)

Binding plan: `artifacts/recall-map/pool-quality/cold-quota-prereg.json`,
`plan_sha256 f6d5593a53b88288abb8b3924df3d27fdc2b8e7a57610e874211a9e84becf9a8`,
sealed `2026-08-24T19:57:56Z` (commit 91f85b1). Nothing in this runbook may
amend that plan; where they disagree, the prereg wins and the window is invalid.

## Deploy record (as executed)

| | local | alt (31.207.47.90) |
|---|---|---|
| repo | `/home/sfx/p/lm` (detached HEAD) | `/home/user/p/lm` (detached HEAD) |
| deployed sha | `b029e6bb132817b1498638e8a8eb1af195ecc159` | `b029e6bb132817b1498638e8a8eb1af195ecc159` |
| env file | `~/.config/living-memory/env` | `/home/user/.config/living-memory/env` |
| valve lines | `LM_MAP_POOL_COLD_QUOTA_GATE=1`, `LM_MAP_POOL_COLD_SLOTS=2` | same two lines |
| restart instant | 2026-08-25 04:16:58 +07 = **2026-08-24T21:16:58Z** | 2026-08-24 23:18:23 CEST = **2026-08-24T21:18:23Z** |
| env verified in live pid | pid 2545062 `/proc/<pid>/environ` after port 8765 up | pid 1665501 `/proc/<pid>/environ` after port 8765 up |
| forbidden valves absent | yes (no USEFULNESS/DEMOTION valves; near-dup pair unchanged: `LM_RECALL_NEAR_DUP_COSINE=0.97`, `LM_DRAIN_NEAR_DUP_SUPERSEDES=1`) | yes (same) |

The local `living-memory-tls` unit is an HTTPS:8766→HTTP:8765 bridge with no
`EnvironmentFile`; it restarted at the same instant and needs no valve lines.

## T0 and the registered window

- **T0 = 2026-08-24T21:18:23Z** — the later of the two restart instants
  (prereg `window.T0`), strictly after the seal instant 19:57:56Z.
- Accrual: payloads with `created_at` in **(T0, 2026-08-27T21:18:23Z]** —
  exactly the census `trailing_72h` window at `--as-of 2026-08-27T21:18:23Z`.
  The 3 local payloads in [21:16:58Z, T0] are pre-window by the census's own
  strict-`>` cutoff and are read by no metric.
- Census read (binding, LOCAL store only):

      PYTHONPATH=src python3 scripts/recall_map_quality_census.py \
          --as-of 2026-08-27T21:18:23Z \
          --out artifacts/recall-map/pool-quality/after.json.census

  with the committed salt `recall-map-pool-quality/live-map-quality-census/r1`
  (a different salt un-groups the label-set digests; the repeat metrics stop
  being comparable). Alt's store gets the symmetric census run **observationally
  only** — it is not a metric source.
- Ledger read / verdict instant: **not before 2026-08-28T21:18:23Z** (T0+96h;
  a window opened at T0+72h matures 24h later).
- Volume floor: ≥ 500 maps in accrual. Below the floor: exactly one extension
  to (T0, T0+144h] (census `--as-of 2026-08-30T21:18:23Z`, ledger read not
  before 2026-08-31T21:18:23Z). Still under the floor after the extension:
  close as `negative-unevaluable`, run the disarm branch.
- Validity: ≥ 98% of sel-carrying payloads inside accrual must carry the `c`
  key; asymmetric valve state, an unrecorded restart on either host, any
  change to the six frozen env valves, or any code deploy touching
  `recall_map.py`, `storage.py` delivery paths or the census script during the
  window invalidates it (recorded in after.json, never graded; fresh T0 after
  the defect is fixed).

## Arm (the procedure, for re-arming from scratch)

1. Confirm the target sha on both hosts. Local: the live checkout
   `/home/sfx/p/lm` at the parent-branch state. Alt (editable install — no
   reinstall): `git bundle create /tmp/lm.bundle <old>..<new>` → `scp` →
   `git -C /home/user/p/lm fetch /tmp/lm.bundle <new>` → `git merge --ff-only <new>`.
2. Append exactly two lines to each host's env file:
   `LM_MAP_POOL_COLD_QUOTA_GATE=1` and `LM_MAP_POOL_COLD_SLOTS=2`
   (paired valves: gate without a parsable number in {1,2} is inert).
3. `systemctl --user restart living-memory` on each host (unit is up in
   ~5–10 s; the model load precedes the port). Local TLS bridge needs nothing.
4. Record both restart instants (`systemctl --user show living-memory
   -p ActiveEnterTimestamp`). **T0 = the later one**; it must postdate the
   prereg seal.

## Verify (all checks executed 2026-08-24T21:2xZ–21:45Z, all green)

1. Port first, then env: wait for 8765
   (`ss -tlnp | grep 8765`), take the **port-owner pid** and only then read
   `/proc/<pid>/environ` — `pgrep` right after restart can catch the fork
   wrapper whose environ is still empty. Expect both valve lines; expect no
   `LM_MAP_POOL_USEFULNESS_GATE` / `LM_MAP_POOL_MIN_USEFULNESS` /
   `LM_MAP_POOL_DEMOTION_GATE` / `LM_MAP_POOL_DEMOTE_AFTER`.
2. Wire form: a live recall's persisted payload
   (`recall_events.recall_map`) carries `sel.c = [g, k]` (present even as
   `[0, 0]` — armed-and-idle and unarmed must not read the same), `x` still at
   the frozen width 7, and each appended cold cluster marked `"cold": 1`.
   Verified live: local event `01M0TV0W6VF4A94WMY9E4RQX5X` (21:32:30Z),
   `sel.c=[215,2]`, `n=479 e=0`, 2 cold clusters; alt event
   `01M0TTPM4RH9A6M75BAH0Z60AQ` (21:26:54Z), `c=[569,2]`, 2 cold clusters;
   alt empty-map events carry `c=[0,0]`.
3. Ledger landing (G5): both cold medoids of the local event have
   `recall_delivery_history` rows with `delivery_event_id` = that event,
   `delivered_at` 21:32:30Z, `outcome_end` +24h — the lane writes through the
   untouched existing path.
4. Armed share since T0 on the local store: 8/8 sel payloads carry `c`
   (validity floor is 98% over the full accrual window).

## Hold (mid-window, every check-in)

Read-only; no registered metric may alter plan, window, bars or arming
(prereg `falsifier.no_peeking`). Check on BOTH hosts:
- `ActiveEnterTimestamp` unchanged (any restart must be recorded; an
  unrecorded one invalidates);
- env file byte-identical; live pid environ still armed;
- repo HEAD still `b029e6b`; no deploys touching `recall_map.py` /
  `storage.py` delivery paths / census script;
- armed-share spot check on the local store.

## Extension record (2026-08-28T21:21Z — written BEFORE any metric read)

At the first read attempt (2026-08-28T21:21:34Z, past the T0+96h floor) the
accrual window (T0, 2026-08-27T21:18:23Z] held **277 maps < 500** — the
volume floor is unmet and the ONE registered extension fires mechanically
(prereg `window.volume_floor`): accrual is now **(T0, 2026-08-30T21:18:23Z]**
(T0+144h), census `--as-of 2026-08-30T21:18:23Z`, ledger read **not before
2026-08-31T21:18:23Z** (T0+168h). No M-metric was computed at this attempt;
only the always-permitted validity/volume counters were read.

Pinned interpretation (recorded pre-read; the registered texts compose
literally): the M/N census metrics are DEFINED on `trailing_72h` of the after
census (prereg `map_metric_acceptance` names trailing_72h explicitly), so
under the extension they read the LAST 72h of the extended accrual at
`--as-of` T0+144h; the 500-map volume floor governs the FULL accrual
(T0, T0+144h], per `window.volume_floor`'s own words. N2/N3/armed-share and
the warm observable read the full accrual, as registered.

Host-restart reconstruction (recorded per `window.invalidating_events`): the
LOCAL machine rebooted 2026-08-27 ~13:44–13:51Z (journalctl: service starts
13:44:47Z / 13:48:25Z / 13:51:05Z across three systemd user-manager
generations; current pid 273 since 13:51:05Z). The env file
(md5 c5954e983ab7372facb645e2af368bcc), HEAD (b029e6b) and live-pid environ
are unchanged; the service auto-restarts from the same EnvironmentFile, and
arming continuity is witnessed on the wire itself: **277/277** accrual sel
payloads carry `c` (armed_share 1.0, zero c-less payloads). The restart is
therefore recorded-with-evidence, symmetric, and configuration-preserving;
window validity continues to rest on the registered armed-share floor. Alt:
zero restarts, same pid 1665501 since T0, env md5
2179970672a21c41d9928720def19d77, HEAD b029e6b, environ armed.

## Read (at or after the ledger-read instant)

1. Run the census (binding, local store, `mode=ro`) with `--as-of
   2026-08-27T21:18:23Z` and the committed salt; run the same census on alt
   observationally.
2. Compute the registered decision inputs, each beside its bar
   (baseline = `baseline.json` `windows.trailing_72h`, as_of pin
   2026-08-24T12:43:56Z — like window against like window, never `all`):
   - M1 uncurtailed empty share ≤ 0.194 (baseline 0.293881)
   - M2 repeat_share ≤ 0.5 (baseline 0.551881) AND top1_share ≤ 0.2 (baseline 0.270239), over non-empty maps, salt r1
   - M3 all-maps empty share strictly < 0.531517
   - N1 curtailed share of maps fires above 0.46
   - N2 cold follow floor: fires if cold rate < 0.5 × warm rate over matured
     known-outcome medoid windows opened in accrual, arms split by the
     cluster `cold` marker; unevaluable below 50 known matured cold windows
     (recorded as such; neither passes nor fires)
   - N3 delivered cold clusters / all delivered clusters fires above 0.7
3. Verdict (prereg `falsifier.decision_rule`): PASS iff M1 ∧ M2 ∧ M3 ∧ ¬N1 ∧
   ¬N3 ∧ (N2 unevaluable ∨ ¬N2); anything else FAIL. No third grade, no
   re-read.
4. Write `artifacts/recall-map/pool-quality/after.json` with keys
   {verdict, window, baseline, metrics, warm_precision_observable, deploy};
   publish the warm-precision observable (non-binding) with both rates, both
   window counts and the NULL-outcome share.

## Disarm (falsifier FAIL, negative-unevaluable, or invalidation cleanup)

1. Remove exactly the two valve lines from both env files.
2. `systemctl --user restart living-memory` on both hosts, port-first pid
   check as above.
3. Confirm on the wire: post-restart persisted payloads carry **no** `c` key
   and `sel.x` is back at the frozen width 7 with no cold markers — absence
   of the lane is encoded exactly one way.
4. Record in after.json: verdict `negative` with the exact failing conditions
   and their numbers; the registered answer is "cold admission needs a
   different signal source". A clean negative completes the experiment.

## Rollback (code)

During the window a code deploy invalidates — do not roll code to save a
window; record the invalidation instead. Outside the window: check out the
prior sha in the live checkout (local) / `git -C /home/user/p/lm reset
--hard <sha>` (alt, editable install), restart, verify the wire form matches
the sha's expectation. Valves-off behavior is byte-identical to pre-lane code
(invariance I1), so disarming alone is always a sufficient behavioral
rollback; code rollback is hygiene, not urgency.

## Emergency

An operator disarm (outage, privacy incident, operator order) is always
allowed, at any instant, and yields verdict `aborted-unevaluated` — never a
pass, never a negative.

## Close record (2026-08-28T21:55:16Z)

The window was closed by operator order via exactly that emergency path: the
extension (fired 21:21:34Z on 277 < 500) was not confirmed; the outcome is
**aborted-unevaluated** and no registered metric was ever read (no-peeking
held from T0 to close). `after.json` records the outcome.

Documented deviation, on the same order: the cleanup this outcome normally
prescribes (disarm) was NOT executed — **arming is retained on both hosts**
(GATE=1, SLOTS=2, code b029e6b, symmetry stays a directive). The operator's
basis is an unregistered directed read over the ALT store outside this
goal's contour (mini-prereg `~/p/ae/artifacts/injection-throttle/
alt-census-prereg.md`, sha256 a979d473…, artifacts beside it); those numbers
are grounds of the operator decision, not a verdict of this experiment, and
are not window metrics. Verified at close: both live pids' environ armed,
both HEADs b029e6b, env md5s unchanged. No env edits, no restarts, no disarm
were performed. Future work (operator's fixation): silencer softening and
transcript-grounding move to separate goals; measurement surfaces move to
alt — no further window registration on the local store is planned.
