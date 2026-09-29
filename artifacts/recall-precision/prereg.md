# Recall precision valves: pre-registration

Written and committed before any holdout run. Nothing in this file may be edited
after the first holdout run. The only allowed addition is the "Frozen values"
section at the end. It is filled from train/eval calibration and committed
before the holdout run.

## Data

- Snapshots are the frozen backup-API copies from the replay harness, stored
  on sfx under `/home/sfx/p/ae/artifacts/recall-precision/snapshots/{sfx,alt}.sqlite3`.
  Their sha256 values are pinned in `artifacts/recall-precision/split.json`
  (sfx `074ee970243d…`, alt `a887efc397ef…`).
- The split comes from `split.json`: per host, recall events from
  2026-09-27T14:45Z, ordered by time. Train is the first 50%, eval the next
  25%, holdout the last 25%.
  - sfx: 332 / 166 / 167 events; holdout from 2026-09-28T15:23:18Z.
  - alt: 220 / 110 / 111 events; holdout from 2026-09-28T14:00:58Z.
- Events after the snapshot are not in the frozen snapshot. The holdout is the
  last 25% inside it, and it is replayed in full, with no sampling.
- Hubs for the hub metric are defined on train marks only, by the harness
  `train_hubs` rule: at least 3 distinct query texts marked the node
  `irrelevant` and it has 0 `used` marks. The valve itself decides on
  whatever marks the counterfactual store holds, which are the marks before
  the cutoff.
- Replay: `scripts/recall_precision_replay.py` on the live path
  (`MemoryRecallService.memory_recall`, `log_access=False, log_event=False`).
  It runs on a counterfactual copy of each snapshot, with marks, anchors,
  edges, credit and lookups at or after the segment cutoff removed.
  `transport_session_id` is stripped, and candidates created after the event
  are hidden.
- Process env for every arm is the production env of both hosts' LM servers,
  read from `/proc/<pid>/environ` on 2026-09-29. It is identical on sfx and
  alt except for paths and the token:
  `LM_EXPLICIT_FEEDBACK_POLICY=credit`, `LM_EXPLICIT_FEEDBACK_PROMPT=mandatory`,
  `LM_IMPLICIT_LINK_POLICY=credited`, `LM_AUTO_CONSOLIDATE_POLICY=adaptive`,
  `LM_RETRIEVAL_TUNING_POLICY=adaptive`, `LM_RECALL_NEAR_DUP_COSINE=0.97`,
  `LM_DRAIN_NEAR_DUP_SUPERSEDES=1`, `LM_MAP_POOL_COLD_QUOTA_GATE=1`,
  `LM_MAP_POOL_COLD_SLOTS=2`, `LM_MAP_CURTAIL_DECAY=1`, `LM_DEFAULT_SCOPE=global`.
  The policy `credit` matters: query-relative demotion is read only under
  it. The arm env is applied on top of this; `baseline` sets no new valve.
- Driver: `artifacts/recall-precision/runs/driver.py`. It uses the harness
  functions and adds the context-chars, rank-1 and entrant-quality metrics
  defined below.

## Arms

Each arm is a set of env valves on top of the production env.

| arm family | env | candidate values (calibrated on eval) |
|---|---|---|
| `baseline` | none | — |
| `gate_drop_T` | `LM_RECALL_MIN_SCORE=T`, `LM_RECALL_GATE_FORM=drop` | T ∈ {0.25, 0.30, 0.35, 0.40, 0.45, 0.50} |
| `gate_stub_T` | `LM_RECALL_MIN_SCORE=T`, `LM_RECALL_GATE_FORM=stub` | T ∈ {0.30, 0.35, 0.40, 0.45, 0.50} |
| `hub_F` | `LM_HUB_SUPPRESSION_FACTOR=F` (`LM_HUB_MIN_QUERIES` default 3) | F ∈ {0.3, 0.1, 0.03} |
| `demote_S` | `LM_QUERY_IRRELEVANCE_FULL_COSINE`, `LM_QUERY_IRRELEVANCE_MARK_WEIGHT` | S ∈ {fc080_mw1 (0.80, 1.0), fc075_mw1 (0.75, 1.0), fc070 (0.70, —)} |
| `dedup` | `LM_RECALL_SCHEMA_DEDUP=1` | — |
| `combined` | chosen gate + chosen hub + chosen demotion + dedup | — |

## Calibration rules (train/eval only; applied per host)

1. **Gate value and form.** For each form, take the largest T whose eval
   `used_full_lost` vs baseline is ≤ 8%, a 4-point margin under the 12% line.
   Form: compare the two forms at their chosen T by context chars saved per
   event (JSON chars of `delivery.shape_recall_results` output, fresh session)
   divided by used-marked slots lost (+1). Pick the higher. On a tie within 10%,
   pick `stub`, because it keeps the cut result's id in the answer.
   If one T satisfies both hosts, use it for both; otherwise each host gets
   its own T. If no T in the grid meets the 8% line on eval, the frozen arm
   uses the smallest T in the grid, and the handoff says "do not enable" for
   that host unless the holdout passes.
2. **Hub factor.** Take the largest F (mildest) that gives an eval
   `hub_top3_cut` ≥ 90% with `used_full_lost` ≤ 2% on sfx. If none does, take
   the F with the largest cut. The same F is used on alt.
3. **Demotion setting.** Take the mildest census setting whose sfx >0.9 share
   is < 25%, computed by `scripts/query_demotion_strength_census.py` on events
   before the holdout start. Its eval replay arm must have `used_full_lost` ≤ 5%
   on each host. Mildness order: fc080_mw1 < fc075_mw1 < fc070 by median
   multiplier, taken from the census table.
4. **Combined.** Chosen gate (per host) + chosen hub + chosen demotion + dedup.

## Metrics (per host, never merged; every arm is compared with the replayed baseline arm on the same events)

- `irrelevant_full_cut` = (baseline full slots on nodes marked
  `irrelevant` − arm full slots on them) / baseline full slots on them. A
  stub is not a full slot.
- `used_full_lost` = (baseline full slots on `used`-marked nodes − arm full
  slots on them) / baseline full slots on them.
- `rank1_removed` = events where the baseline's rank-1 node is not delivered
  in full at rank 1 by a gate-only arm. The gate does not reorder, so for
  gate-only arms this is exactly the "rank 1 never removed" check.
- `hub_top3_cut` = (baseline top-3 full slots held by train-defined hubs −
  arm's) / baseline's. Hub-arm used loss = `used_full_lost`.
- `dup_schema_slots` = full schema slots per answer beyond the first with the
  same normalised title.
- Demotion strength: the census share of applicable (event × marked node)
  cases with multiplier > 0.9.
- Entrant quality, the independent signal: grounding in the event's closing
  trace minus a cosine-matched twin's grounding (`explicit_feedback_agreement`
  method, containment 0.5). It is computed over (a) every full slot the
  baseline delivers and (b) the arm's full slots that are not in the
  baseline's full list for the same event ("new entrants").
  `excess = mean(grounded − twin_grounded)`.
- Context: JSON chars per event of the shaped answer, and chars saved vs
  baseline.
- Rank breakdown: for every arm, `irrelevant_full_cut` and `used_full_lost`
  by recorded rank bucket (1, 2, 3, 4-5, 6-10).

## Pass lines (holdout, each host separately)

| id | condition | host |
|---|---|---|
| G1 | frozen gate arm `irrelevant_full_cut` ≥ 25% | sfx, alt |
| G2 | frozen gate arm `used_full_lost` ≤ 12% | sfx, alt |
| G3 | frozen gate arm `rank1_removed` = 0 | sfx, alt |
| H1 | frozen hub arm `hub_top3_cut` ≥ 80% | sfx |
| H2 | frozen hub arm used-marked full slots lost ≤ 3% of baseline used full slots | sfx |
| H3 | alt hub effect reported (cut, used loss), no pass line (alt has ~1% hub slots) | alt |
| D1 | chosen demotion setting: census share with m > 0.9 < 30% on sfx holdout-window events | sfx |
| D2 | same census on alt, reported only | alt |
| S1 | dedup arm `dup_schema_slots` = 0 | sfx, alt |
| Q1 | new-entrant excess of the frozen gate, hub and combined arms ≥ the baseline delivered-set excess (point estimate). If an arm has < 10 graded entrants, it is reported as "insufficient n", not a pass. | sfx, alt |

A condition that fails is reported as FAIL with its number. No condition is
re-run with other values. The handoff recommendation for a valve on a host is
"enable" only if every line that applies to that valve on that host passes.
Otherwise it is "do not enable", with the failing numbers.

## Position bias of marks

Agents mark mostly what they see at the top. In the smoke eval sample, 60–70%
of marks were in ranks 1–5, and unmarked slots are not labelled. The
measurement handles this as follows:

1. Every effect is a difference against the replayed baseline arm on the same
   events, never against the recorded delivery. A bias that affects both arms
   the same way cancels.
2. Unmarked slots are not counted as irrelevant or used. The quality of
   what fills a freed slot is judged by the grounding-minus-twin signal (Q1),
   which does not depend on list position.
3. Every cut and loss is also broken down by recorded rank bucket. A gate
   that only cuts low ranks, where marks are sparse, shows up as small
   denominators in the 4-5 and 6-10 buckets and is read that way.
4. Rank 1 is excluded from the gate by construction and checked (G3).

## Frozen values

Filled from eval calibration on 2026-09-29, before any holdout run. The eval
outputs are `runs/eval-{sfx,alt}-{gate,other}.json` and
`runs/census-{sfx,alt}-trainval.json`. They cover all eval events (sfx 166,
alt 110) and use the production env.

**Gate.** Rule 1 is read per host: "if one T satisfies both hosts" means the
per-host largest T is the same on both. It is not, so each host gets its own
value.

| host | form | T | eval irr cut | eval used lost | chars saved/ev (drop / stub) | used slots lost (drop / stub) |
|---|---|---|---|---|---|---|
| sfx | drop | 0.35 | 34.4% | 6.8% | 2662 / 823 | 7 / 7 |
| alt | drop | 0.30 | 10.6% | 6.1% | 433 / 167 | 5 / 5 |

At T=0.35, alt eval used loss is 12.2%, which is over the 8% calibration line.
`drop` beats `stub` on both hosts. A stub entry still costs ~300–400 JSON
chars, and at these thresholds `drop` rarely refills the slot: sfx drop
delivers 2.98 full slots per event against 2.88 full + 1.97 stubs for stub.
Rank 1 was never removed on eval (0 events in every gate arm).

Also run on holdout and reported without pass lines: the other host's T
(`gate_alt_T`: sfx 0.30, alt 0.35).

**Hub.** No F reaches a 90% cut on sfx eval. F = 0.3 / 0.1 / 0.03 gives
27.3% / 30.3% / 30.3% cut, all with 0 used loss. Rule 2 takes the largest cut,
and of the tied values the mildest: **F = 0.1**. Diagnosis: the valve counts
lookup credit after the first complaint as positive evidence (P2). Of the 115
train-defined sfx hubs, 39 are lifted this way (all 39 by `lookup`, 13 also by
`grounded`), so only 75 are suppressed. H1 is therefore expected to fail. It is
still run as registered.

**Demotion.** fc080_mw1 (`LM_QUERY_IRRELEVANCE_FULL_COSINE=0.80`,
`LM_QUERY_IRRELEVANCE_MARK_WEIGHT=1.0`). The sfx train+eval census gives a
>0.9 share of 24.6% (default 70.2%, fc075_mw1 20.0%, fc070 24.5%). Eval used
loss is 0.0% on sfx and on alt. The alt census gives 39.4% (default 89.5%).

**Dedup.** `LM_RECALL_SCHEMA_DEDUP=1`. On sfx eval it takes dup schema slots
from 41 to 0.

**Holdout arms (frozen)**, env on top of the production env:

| arm | sfx | alt |
|---|---|---|
| baseline | — | — |
| gate | `LM_RECALL_MIN_SCORE=0.35,LM_RECALL_GATE_FORM=drop` | `LM_RECALL_MIN_SCORE=0.30,LM_RECALL_GATE_FORM=drop` |
| gate_alt_T (reported) | `LM_RECALL_MIN_SCORE=0.30,LM_RECALL_GATE_FORM=drop` | `LM_RECALL_MIN_SCORE=0.35,LM_RECALL_GATE_FORM=drop` |
| hub | `LM_HUB_SUPPRESSION_FACTOR=0.1` | same |
| demote | `LM_QUERY_IRRELEVANCE_FULL_COSINE=0.80,LM_QUERY_IRRELEVANCE_MARK_WEIGHT=1.0` | same |
| dedup | `LM_RECALL_SCHEMA_DEDUP=1` | same |
| combined | gate + hub + demote + dedup | gate + hub + demote + dedup |

D1/D2 census on holdout-window events (`--since` holdout start), settings
`default` and `fc080_mw1`.
