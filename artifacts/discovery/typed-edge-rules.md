# Typed-edge derivation rules — provenance mining on the live Living Memory corpus

Discovery artifact for goal `retrieval-credit-assignment-graph-value/provenance-mining`.
Produced 2026-07-07 from a consistent read-only snapshot of the live DB.

## 0. Scope contract — local-only, read-only, no external action plan

- **Intent**: design (not apply) the typed-edge derivation rule set that lets the
  LM graph channel carry non-redundant signal. This file prescribes changes only
  for this repository's downstream tree nodes (`typed-edge-engine`,
  `typed-edge-backfill`); it prescribes nothing for other repositories or live
  systems.
- **Read boundaries**: the live DB `/home/sfx/.local/share/living-memory/global.sqlite3`
  was accessed exclusively through the sqlite backup API from a connection opened
  with `file:...?mode=ro` (plus `PRAGMA query_only=ON` in the miner). No writes
  were made to the live DB or its WAL. Quoted trace excerpts below are the
  operator's own local memory content; no credentials/PII appear in the quotes.
- **Local replay**: `python3 scripts/mine_typed_edges.py --db <path>` (defaults to
  the live path, opened read-only; use a snapshot copy for stable counts:
  `python3 -c "import sqlite3; s=sqlite3.connect('file:/home/sfx/.local/share/living-memory/global.sqlite3?mode=ro',uri=True); d=sqlite3.connect('/tmp/lm_snapshot.sqlite3'); s.backup(d)"`).
  All numbers below come from the 2026-07-07 ~21:09 UTC+7 snapshot,
  seed=42, sample_size=20, max_file_fanout=20, max_commit_group=10,
  max_proc_group=12.
- **Local rollback**: `git rm artifacts/discovery/typed-edge-rules.md scripts/mine_typed_edges.py`.
- **Redaction**: nothing to redact; the corpus is single-operator local data and
  excerpts are truncated to ≤360 chars.

## 1. Method and metrics

`scripts/mine_typed_edges.py` loads all nodes (14,142; 10,936 active), all
connections (102,661), and all recall_events (46,819), then evaluates each
candidate rule and reports:

- **derivable count** — unique undirected pairs the rule emits, and the
  **both-active** subset. Graph expansion skips decayed neighbors
  (`src/living_memory/retrieval.py:525`), so only both-active pairs add
  traversal reach. (Exception: a `supersedes` edge to a decayed target still
  feeds the superseding-source rank boost via `rank_candidates`
  `_supersedes_sets`, `retrieval.py:211,249-254`.)
- **new-information fraction** — share of both-active pairs *not already
  connected by any edge of any type in either direction* (existing unique
  pairs: 100,795; both-active: 92,192).
- **co-retrieval redundancy** — share of both-active pairs that already
  co-occurred in at least one recorded `recall_events` result list (134,803
  distinct co-retrieved pairs). This directly measures "bm25/vector already
  surface these two together"; lower = more non-redundant graph signal.
- **sampled precision** — seeded random sample (all pairs when ≤20, else 20)
  manually inspected; every verdict recorded below.

Baseline edge inventory (snapshot): `related` 99,356 (implicit_recall_feedback
53,936 / cluster 21,108 / co_access 18,341 / procedural 5,938 / other 33),
`contradicts` 3,066 (100% rejected-alternative kind), `supersedes` 239 (teach
184 / duplicate_content 53 / other 2). **`caused` and `requires` do not exist
(0 edges each), and causal-mode traversal follows only those two types
(`retrieval.py:688`) — the causal graph channel is literally empty today.**
Typed share now: 3,305 / 102,661 = **3.22%**.

Two structural facts shape everything below:

1. **Decay eats supersedes targets.** 238 of 239 supersedes targets are
   decayed (teach drops the original's confidence to ≤0.25 and usefulness to
   ≤ -0.25, `consolidation.py:361-374`, and the decay sweep then removes it).
   Any correction-lineage rule has near-zero *backfill* yield; its value is at
   *write time*, when the target is still active.
2. **Lessons are written self-contained.** Failure, root cause, and fix are
   usually one trace (`implementation_bug` lesson_kind: 364 active traces).
   Separate cause→effect trace pairs are rare, so typed `caused`/`supersedes`
   volume is intrinsically bounded by authoring style, not by rule cleverness.

## 2. Rule-by-rule results

### R1a `supersedes_transitive` — NO-GO

Close supersedes chains: A⇒B, B⇒C ⟹ A⇒C.

- Derivable: 323 directed pairs; **both-active: 1**; new-info 1.0; co-retrieval 0.
- Sampled precision (the entire active yield, 1 pair):

| # | pair | verdict | reason |
|---|------|---------|--------|
| 1 | `01KTC4SFTZ…` ⇒ `01KTC45M9C…` | NO | two *complementary* corrections of the same procedure (ticket-field vs noCommitOk); A does not invalidate C |

- Verdict: **no-go**. Yield ≈ 0 because chain middles decay, and the one
  surviving pair shows the semantic flaw: teach chains correct *different
  aspects*, so transitivity does not preserve supersession.

### R1b `correction_lineage` (heuristic target) — NO-GO

Correction-marked remember-traces (context.type ∈ {correction,
memory_correction, decomposition_correction} or content prefix `Correction`)
⇒ supersedes a recalled node. Two variants measured:

- Loose (targets = same-task recalled ∪ top-3 of latest prior recall):
  72 directed, 56 both-active, new-info 0.0 (all already related-edged via
  implicit recall feedback), co-retrieval 0.70. Sampled precision **0/20** —
  every sampled target was *not* the corrected node (corrections correct one
  specific trace; recall/task overlap does not identify it).
- Tight (same-scope same-task recalled only, ≤2 targets): 14 directed,
  7 both-active. Sampled precision **1/7**:

| # | pair | verdict | reason |
|---|------|---------|--------|
| 1 | `01KS0CSJCV…`⇒`01KS0BFHM8…` | NO | rerender correction vs unrelated diff-check note |
| 2 | `01KSG93S07…`⇒`01KS0CSJCV…` | NO | searchId correction vs rerender-memoization correction (different aspects) |
| 3 | `01KSG93S07…`⇒`01KS0N3J9Y…` | NO | searchId correction vs jest test-discovery correction |
| 4 | `01KSQQK095…`⇒`01KS0CSJCV…` | NO | decomposition correction vs rerender correction |
| 5 | `01KSQQK095…`⇒`01KS0N3J9Y…` | NO | decomposition correction vs jest correction |
| 6 | `01KTBWKZZZ…`⇒`01KSHN0QSP…` | YES | photo-flex correction explicitly supersedes the older photo-flex fix trace |
| 7 | `01KVAWWM5Q…`⇒`01KVAWAZ12…` | NO | typo correction targets the client-payload trace, not this bot-detector trace |

- Verdict: **no-go for both backfill and write path**. Inferring the corrected
  target heuristically is wrong ~6/7 times. Consequence for the system (not a
  rule): corrections must carry an explicit target — `memory_teach(trace_id)`
  (already creates the supersedes edge) or a named ULID in content (→ R1c).
  The server instructions already push teach-on-correction; this measurement
  is the quantitative reason why.

### R1c `content_ulid_supersedes` — GO (write-path primary, small backfill)

Trace content contains an adjacent correction/update phrase naming a node
ULID: regex
`(?i)(?:\b(?:correction|update)s?(?:/\w+)?[-\s]+(?:to|of|for)\b|\b(?:corrects|supersedes)\b)[^\n]{0,60}?\b01[0-9A-HJKMNP-TV-Z]{24}\b`
(fallback: correction-marked context + any content ULID)
⇒ `supersedes` named node.

- Derivable: 15 unique pairs; **both-active: 9**; new-info 0.889; co-retrieval **0.0**.
- A looser 80-char any-correction-word window yielded 12 active pairs at
  8–9/12; the adjacent form above drops exactly the pairs whose trigger word
  referred to a third node. Final sampled precision **9/9**:

| # | pair | verdict | reason |
|---|------|---------|--------|
| 1 | `01KSTFQ7FK…`⇒`01KSTDBPRG…` | YES | "Supersedes-progress of design node 01KSTDBPRG…" (design → implemented) |
| 2 | `01KSTWE81W…`⇒`01KSTPXFGZ…` | YES | "supersedes the branch-only note" (branch-complete → merged to main) |
| 3 | `01KSWYKDD6…`⇒`01KSWY4SHV…` | YES | "UPDATE/correction to node … (point 5) … is FIXED" |
| 4 | `01KSX21KBN…`⇒`01KSX0DZQG…` | YES | "UPDATE to nodes [[lm-bm25-floor-fix-landed]] (01KSX0DZ…)" landed → live |
| 5 | `01KTDZKEXC…`⇒`01KTDYR1CH…` | YES | completion status supersedes the in-progress finding for the same node |
| 6 | `01KTRQYJYP…`⇒`01KTRQDVN2…` | YES | "UPDATE (supersedes 'goal left in failed/active state…' from node …)" |
| 7 | `01KV30KW63…`⇒`01KV2W32M4…` | YES | "corrects/supersedes the leading hypothesis in 01KV2W32…" |
| 8 | `01KVBNCFN1…`⇒`01KVBMHJJM…` | YES | "CORRECTION to trace 01KVBMHJ… /var/lock, NOT /tmp" |
| 9 | `01KWQHH5EN…`⇒`01KWQG16AF…` | YES | "correction/extension of yesterday's child-level note (01KWQG16…)" |

- Direction: correction ⇒ corrected (matches teach). Mapping: `supersedes`,
  weight 0.9, metadata `{kind: "content_correction", rule: "R1c", matched: <phrase>}`.
- Traversal: existing supersedes factors (forward 0.35 / backward 1.15,
  `retrieval.py:714-715`) are exactly right — a query matching the stale node
  boosts the correction 1.15×, and the stale node takes the superseded rank
  penalty, which is correct for corrected/obsolete status content.
- Why the backfill yield is small: 4 more pairs name decayed targets — same
  decay dynamic as R1a/teach. **The rule's main value is at write time**
  (memory_remember hook), where ~44 `Correction*`-prefixed traces accrued over
  7 weeks (~1/day) and targets are still active.
- Verdict: **go**.

### R2a `root_cause_caused_resolution` — GO (small, high-precision; fills the empty causal channel)

Within a (scope, context.task) or (scope, session_id) group, RA-nodes
excluded: a solution-labeled trace (type ∈ {failure_resolution, working_fix,
fix, gate_failure_resolution, supervision_repair} or lesson_kind ∈
{*decomposition*repair*, working_fix}) written **0–60 min after** a
problem trace typed root_cause/root_cause_and_debugging_insights (or outcome
structural_fail/reopened, solution-labeled excluded), nearest problem only ⇒
**`caused`, direction root_cause → resolution**.

- Evidence for the tight design: the loose variant (any problem label, any
  gap ≤ same task) gave 38 pairs at sampled precision **4/20** — leaks were
  positive-polarity `verification_finding`, self-contained
  `implementation_bug` lessons, and the recurring "supervise active AE goals"
  umbrella task pairing unrelated incidents. All 4 true pairs sat ≤4 minutes
  apart; a 6-hour window admitted one false pair at 287 minutes (4/5). The
  60-minute window keeps exactly the true pairs.
- Derivable: **4 both-active pairs**; new-info 0.5; co-retrieval 0.75 (small-n;
  the value here is the causal typing, not connectivity).
- Sampled precision (entire yield) **4/4**:

| # | pair | verdict | reason |
|---|------|---------|--------|
| 1 | `01KS541R1N…`→`01KS544TQH…` | YES | dead-pid node_status root cause → "Recover interrupted tree node status" fix, 2 min |
| 2 | `01KS667G03…`→`01KS66H6AR…` | YES | bounded-exec zero-output-stall root cause → its universal fix, 4 min, matching slugs |
| 3 | `01KS793CVQ…`→`01KS79403M…` | YES | malformed acceptance payload root cause → "Reject malformed acceptance payloads" fix, 30 s |
| 4 | `01KS79WMZX…`→`01KS79X696…` | YES | stale npm-test-failures root cause → baseline-downgrade gate fix, 2 min |
- Mapping: `caused`, weight 0.9, metadata `{kind: "failure_resolution", rule: "R2a", gap_seconds, group_key}`.
- Direction semantics: cause precedes consequence. Backward traversal from the
  fix finds its cause (factor 0.85 normal / 1.0 causal mode); forward from a
  matched root-cause text surfaces the fix (0.75); the pair becomes reachable
  in **causal mode**, which follows only caused/requires and today has zero
  edges. `supersedes` was rejected: causal mode skips it, and the superseded
  rank penalty would wrongly punish still-valid root-cause knowledge.
- Traversal: keep existing caused factors (0.75/0.85; causal mode 0.25/1.0,
  `retrieval.py:707-711`). No new factor needed.
- Verdict: **go** — 4 backfill edges plus write-path accrual; supervision-style
  sessions (root_cause trace then fix trace minutes later) recur.

### R2b `root_cause_caused` (separate failure-evidence traces) — NO-GO

root_cause-typed ⇒ `caused` ⇒ failure-evidence-typed (verification_finding /
failure / error / regression / structural_fail / reopened) in the same task.

- Derivable: **0 pairs** on the whole corpus. Root-cause analyses and the
  failures they explain are written as one self-contained trace here; separate
  failure-event traces co-occurring with root_cause traces in one task do not
  exist. Verdict: **no-go** (nothing to derive; revisit only if authoring
  style changes).

### R3 `procedure_requires` — NO-GO

Consecutive timestamp-ordered, RA-excluded traces within one
(scope, procedure_id), group size 2–12, ≥2 distinct timestamps ⇒ later
`requires` earlier.

- Derivable: 250 both-active pairs; new-info 0.81; co-retrieval 0.18.
- Sampled precision as `requires`: **0/20**:

| # | pair | verdict | reason |
|---|------|---------|--------|
| 1 | `01KWVC47FW…`→`01KWV7NSP2…` | NO | two sibling lessons of one procedure run, not steps |
| 2 | `01KTC44HVE…`→`01KTC35P6V…` | NO | complementary corrections of one procedure |
| 3 | `01KRZ8Q497…`→`01KRZ8PZQG…` | NO | unrelated tree completions batch-written 5 s apart |
| 4 | `01KRVX29J4…`→`01KRVTTN7C…` | NO | next-day re-verification of the same feature |
| 5 | `01KRVVXZEH…`→`01KRVTYRWD…` | NO | sequential project work; "requires" overclaims |
| 6 | `01KS1RYQQT…`→`01KS0T0Q08…` | NO | different meta-anomaly detections |
| 7 | `01KRVW4XSH…`→`01KRVSHW2M…` | NO | acceptance repairs of different nodes |
| 8 | `01KS0W5P86…`→`01KRXY5R3Y…` | NO | different tree fixes a day apart |
| 9 | `01KV3T4R3G…`→`01KV3PQTTT…` | NO | sequential parent repairs (supersedes-flavored, not requires) |
| 10 | `01KS0QG95Y…`→`01KRY0P5VQ…` | NO | recurring anomaly log entries |
| 11 | `01KSEX351W…`→`01KRWQX4GP…` | NO | verification results of different tasks, days apart |
| 12 | `01KRXMW638…`→`01KRXH6SQW…` | NO | sequential gate repairs; recurrence, not prerequisite |
| 13 | `01KTQDRHKQ…`→`01KTQBTDMM…` | NO | re-execution continuation — real relation, wrong type |
| 14 | `01KRVSHW2M…`→`01KRVSB4Y9…` | NO | sibling repairs of one node 6 min apart |
| 15 | `01KT81M2M7…`→`01KT73Y391…` | NO | supervisor repairs of different goals |
| 16 | `01KS7SPZPE…`→`01KS7SP4YP…` | NO | verification completions 28 s apart, unrelated archives |
| 17 | `01KS93B25T…`→`01KS8PVG4G…` | NO | unrelated implementation completions |
| 18 | `01KRVZPX73…`→`01KRVW4XSH…` | NO | acceptance repairs of different nodes |
| 19 | `01KWY4K9SG…`→`01KTV00V1K…` | NO | reopen fixes for different tickets a month apart |
| 20 | `01KW0Y9Y4E…`→`01KW0R3AF1…` | NO | replacement decisions for different nodes |

- Root cause of the failure: **`procedure_id` groups recurring *instances* of
  a procedure, not ordered steps** — only 4 of 3,255 procedure_id nodes carry
  a `step`/`step_order` key, small groups are completion+rejected-alternative
  write batches sharing one timestamp, and large groups (reopen_lesson n=716,
  active_goal_supervision n=217) are routine logs. Timestamp order encodes
  recurrence, not dependency. The same-procedure relation itself is already
  represented by the procedural schema hub edges (schema→trace, 5,938).
- Verdict: **no-go**. `requires` stays empty until real step metadata exists
  (write-path opportunity: populate `step_order` when procedures are recorded
  as actual step sequences — out of scope here).

### R4a `same_commit` — GO

Active nodes whose `context.commit` shares a 7-char prefix, group size 2–10,
minus both-rejected-alternative pairs ⇒ undirected `related`.

- Derivable: 206 both-active pairs; new-info 0.228 (≈47 unconnected pairs);
  co-retrieval **0.053**.
- Sampled precision **20/20** (seed-42 sample of the pre-RA-exclusion pool;
  5 sampled pairs were both-RA pairs since excluded from the rule — the
  exclusion is motivated by RA rank suppression in `rank_candidates`
  (`retrieval.py:216-217`), not by precision):

| # | pair | verdict | reason |
|---|------|---------|--------|
| 1 | `01KSPGATZM…`↔`01KSPGATZN…` | YES | repair trace + its rejected alternative, one commit |
| 2 | `01KRVT81QR…`↔`01KRVT81QR…` | YES | two RAs of one change (both-RA class, now excluded) |
| 3 | `01KWQCW8T5…`↔`01KWQCW8T7…` | YES | two RAs of one change (excluded class) |
| 4 | `01KS7GK72E…`↔`01KS7H4NPF…` | YES | integration verification + verification *failure* of the same landed commit — cross-batch |
| 5 | `01KSM0JGJ4…`↔`01KSM0JGJ4…` | YES | RA + the commit's description |
| 6 | `01KRX87KVN…`↔`01KRX87KVZ…` | YES | two RAs (excluded class) |
| 7 | `01KRVT7D55…`↔`01KRVT81QQ…` | YES | RA + implementation trace, cross-batch same commit |
| 8 | `01KRVT5339…`↔`01KRVT5339…` | YES | fix + its RA |
| 9 | `01KRXW2XCQ…`↔`01KRXW2XCQ…` | YES | RA + repair description |
| 10 | `01KS7GCP9R…`↔`01KS7H4NPF…` | YES | integration outcome + verification failure, same commit |
| 11 | `01KS0JJG44…`↔`01KS0JJG44…` | YES | two RAs (excluded class) |
| 12 | `01KRVVRA9P…`↔`01KRVVRA9Q…` | YES | fix + RA |
| 13 | `01KSPGATZN…`↔`01KSPGATZN…` | YES | two RAs (excluded class) |
| 14 | `01KS79403M…`↔`01KS797HD5…` | YES | fix + follow-up live-state repair applying it, 90 s apart |
| 15 | `01KTCPNNT0…`↔`01KTCPNNT1…` | YES | two RAs (excluded class) |
| 16 | `01KSNR68SC…`↔`01KSNR68SQ…` | YES | observation + RA |
| 17 | `01KRWTZKV6…`↔`01KRWTZKV6…` | YES | RA + implementation |
| 18 | `01KSP2JVSV…`↔`01KSP2JVT8…` | YES | repair + RA |
| 19 | `01KSKHC461…`↔`01KSKHC462…` | YES | cleanup note + RA |
| 20 | `01KRX98K08…`↔`01KRX98K09…` | YES | lesson + RA |

- Mapping: `related` (no inherent direction — "typed vs better-basis related?"
  answer: **better-basis related**), weight 0.9,
  metadata `{basis: "same_commit", commit: <prefix>, rule: "R4a"}`.
- Traversal: standard related factor 0.65; the higher weight (0.9 vs implicit
  edges' rank-scaled weights) carries the strength.
- Verdict: **go**. Small but the 5% co-retrieval redundancy is the best of all
  rules — same-commit pairs describe *different aspects of one change*
  (impl vs verification-failure vs gotcha) that bm25/vector do not co-surface.

### R4b `shared_files` — GO (the volume rule)

Active nodes citing the same `context.files` entry, per-file fanout ≤ 20
(popular files like whole-subsystem hubs excluded as topic-dilute), minus
both-RA pairs ⇒ undirected `related`.

- Derivable: **9,992 both-active pairs**; new-info **0.887** (≈8,863
  unconnected pairs); co-retrieval **0.056**.
- Sampled precision **20/20** (2 weak):

| # | pair | verdict | reason |
|---|------|---------|--------|
| 1 | `01KT3NQ6X4…`↔`01KT442C36…` | YES | design-contract freeze + figma evidence bundle (2 shared files) |
| 2 | `01KVYS4MNB…`↔`01KW57RRX4…` | YES (weak) | eval-harness plan + battery execution, 1 shared script |
| 3 | `01KS7SZFXA…`↔`01KS7V1J3T…` | YES | stale-floors note + xfail gate repair on the same test file |
| 4 | `01KVA7ZFDC…`↔`01KVCJ2N4M…` | YES | sequential evidence freezes over the same package (3 files) |
| 5 | `01KVQ48NHV…`↔`01KVQ4G33T…` | YES | reopen lesson + api-contract note on the same doc policy |
| 6 | `01KSQZG63R…`↔`01KWCFEVH7…` | YES | EZ-13771 tooltip gate + EZ-14067 rollout repair — cross-ticket, same component |
| 7 | `01KTXXWQH5…`↔`01KW6WE0WD…` | YES (weak) | fp8 design + NW-9 plastic core; same file, distant topics |
| 8 | `01KT2EQPQJ…`↔`01KTV3P0VY…` | YES | richDashboard deprecation lesson + its later restore repair — cross-time |
| 9 | `01KSQNXTPF…`↔`01KSQPEN3N…` | YES | provider-ui completion + api-types child note |
| 10 | `01KTGWWSTF…`↔`01KTYCW2VF…` | YES | precision-policy leaf + fp8 policy gate, same policy surface |
| 11 | `01KTC6SW82…`↔`01KTC7X6WR…` | YES | discovery + validation over the same formatters |
| 12 | `01KT3PA4BW…`↔`01KT91RN3T…` | YES | jira re-fetch + figma surface inventory, same contract doc |
| 13 | `01KVRACJ8M…`↔`01KVTGYQN1…` | YES | hint-copy correction + reopen fix, same spec file |
| 14 | `01KV2AYXQ1…`↔`01KVAVDXXB…` | YES | return-curve tooling + verification that uses that validator |
| 15 | `01KT1H3S7J…`↔`01KVQNN5BB…` | YES | header analytics rerun + dashboard reopen fix, same dispatch.ts |
| 16 | `01KTBAQ17W…`↔`01KTMACDTR…` | YES | GPU-utilization plan + loss.cpp RA, same backend effort |
| 17 | `01KS7F306D…`↔`01KTTNNEE5…` | YES | LM decomposition correction + floors adequacy audit — cross-task, same feedback.py |
| 18 | `01KT1XPPMX…`↔`01KT20C9JN…` | YES | mobile.tsx note + parent-retry RA, same repair session |
| 19 | `01KS0QMN4W…`↔`01KVAYNZ8K…` | YES | EZ-13771 jest verification + EZ-13962 TZ=UTC gotcha — cross-ticket, same spec file |
| 20 | `01KTACB63S…`↔`01KTQGQHDS…` | YES | scheduler impl + CUDA-test RA, same header |

- Mapping: `related`, weight `min(0.95, 0.5 + 0.15 × n_shared_files)` (1 file
  → 0.65 = baseline related, multi-file → up to 0.95), metadata
  `{basis: "shared_files", files: [...], rule: "R4b"}`. Undirected.
- Traversal: standard related factor 0.65; weight carries overlap strength.
  The weak tail (1 shared file, topically distant) lands at combined
  0.65×0.65 ≈ 0.42, appropriately below implicit-edge strength.
- Verdict: **go**. 8.9k new active pairs at 5.6% co-retrieval redundancy is
  the dominant non-redundancy gain: same-file knowledge (conventions, gotchas,
  regressions) that lexical/vector retrieval does not connect across tickets
  and weeks.

### R5a `derived_from` (schema → source trace) — GO as kind-annotation; concepts NO-GO

Schema-level nodes → their `source_traces` members (provenance ground truth;
edges already exist as `related`/procedural with source=schema, created in
`consolidation.py:657-677`).

- Derivable: 5,228 directed pairs (5,223 both-active); new-info 0.0 (all
  already edged — this is a **retype/annotate** rule, not new connectivity);
  co-retrieval 0.047.
- Sampled precision **20/20** for schemas (samples: critique/acceptance-repair/
  reopen-lesson/supervision/task-outcome schemas → their instances; coherent
  even at 121–185 sources because schema grouping is by exact
  task_pattern/procedure_id field, not similarity):

| # | pair | verdict | # | pair | verdict |
|---|------|---------|---|------|---------|
| 1 | `01KT4E4TDK…`→`01KS3BMRCP…` | YES | 11 | `01KS1B6J2G…`→`01KRXW2XCQ…` | YES |
| 2 | `01KW5CAA8G…`→`01KV8X4KAK…` | YES | 12 | `01KSQZF18J…`→`01KS1N7QBX…` | YES |
| 3 | `01KW15GV6Z…`→`01KV8QJGV0…` | YES | 13 | `01KS1Q20ES…`→`01KRVXDWR9…` | YES |
| 4 | `01KT7QN1JK…`→`01KT8DDB57…` | YES | 14 | `01KT8735JA…`→`01KT6MPEBF…` | YES |
| 5 | `01KVFRV26V…`→`01KVC7NYH6…` | YES | 15 | `01KT2968E8…`→`01KSZG5YD2…` | YES |
| 6 | `01KSFNCKPF…`→`01KRVSHW2M…` | YES | 16 | `01KTSB8N0B…`→`01KT8XF63M…` | YES |
| 7 | `01KS15ABR1…`→`01KRXW2XCQ…` | YES | 17 | `01KWQ5YJZS…`→`01KRVX6EHJ…` | YES |
| 8 | `01KRXY78CS…`→`01KT29C011…` | YES | 18 | `01KT7QN1JR…`→`01KT6PZ9VS…` | YES |
| 9 | `01KSQJ5GBC…`→`01KSEVEJ64…` | YES | 19 | `01KSB3Y9D2…`→`01KRVRA4FD…` | YES |
| 10 | `01KT4E4T2X…`→`01KRY9E10R…` | YES | 20 | `01KT7QN1M6…`→`01KT7ZM443…` | YES |

- **Concepts are excluded** (a rule variant including concept→source_traces was
  measured: 26,279 directed pairs): sampled concept pairs scored **0/16** —
  similarity clustering produced degenerate mega-clusters (93–2,209 sources;
  e.g. concepts with 357/506/478 sources whose representative content is a
  single completion trace, unrelated to arbitrary members). This is an
  upstream consolidation-quality issue, out of scope here; do not type those
  edges until cluster quality is fixed.
- Mapping: **no new connection type** (a new type would require migrating the
  `connections` CHECK constraint). Follow the `rejected_alternative`
  precedent (`_is_rejected_alternative_connection`, `retrieval.py:725-729`):
  keep type `related`, set metadata `{kind: "derived_from", rule: "R5a"}` on
  the existing schema→trace procedural edges (metadata merge via the existing
  `_upsert_weighted_connection` semantics — weight max-merge, metadata merge;
  no rows deleted or retyped).
- Traversal factor proposal (`_traversal`, `retrieval.py:674-722`, new kind
  branch before the generic `related` case): **forward (schema→trace) 0.5,
  backward (trace→schema) 0.95**. Rationale: when a query matches a concrete
  instance, surface the distilled procedure strongly (it aggregates all
  instances and already gets trigger boosts only on exact trigger matches);
  drilling from schema to exemplars is useful but weaker. Excluded from
  causal mode and decision mode like plain related.
- Verdict: **go** (annotation + asymmetric traversal; 5,223 edges become
  directional-typed without any new rows). Must pass the replay-harness
  no-regression gate since it changes ranking shape on existing edges.

### R5b `trace_informed_by` — NO-GO

Trace-level `source_traces` (nodes recalled just before the trace was
written, `feedback.py:195-196`) ⇒ directed related.

- Derivable: 41,129 directed (39,229 both-active); **new-info 0.0** (these are
  exactly the implicit_recall_feedback edges); co-retrieval 0.245.
- Sampled quality as generic related: 10 YES / 3 weak-YES / 2 partial / 5 NO
  (≈0.50–0.65) — the known mixed quality of recall-echo edges (cross-project
  and topic-drift noise: e.g. an EZ-13771 verification linked to an octopus
  decomposition because both were recalled in one session).
- Verdict: **no-go**. Adds zero new pairs; retyping 39k mixed-precision edges
  would launder recall echo as provenance. Leave as basis-tagged related.

### R6 `content_ulid_reference` — GO

Trace content names another node's ULID without correction phrasing (R1c
pairs excluded) ⇒ directed `related` (referrer → referee).

- Derivable: 209 directed pairs (197 both-active); new-info 0.533 (≈105 new
  pairs); co-retrieval 0.44.
- Sampled precision **20/20** — explicit citations are deliberate provenance:
  re-execution referencing the prior run, verification referencing the
  completed sub-goal, incident recurrence referencing the original incident,
  plans referencing design nodes, PASS verification referencing the earlier
  FAIL:

| # | pair | verdict | # | pair | verdict |
|---|------|---------|---|------|---------|
| 1 | `01KS79PE21…`→`01KS79KQ52…` | YES | 11 | `01KS8BQXDA…`→`01KS8B7FZG…` | YES |
| 2 | `01KS8XFW2G…`→`01KS8XAP7H…` | YES | 12 | `01KS9M4MX9…`→`01KS9KWDAF…` | YES |
| 3 | `01KTA9EKQE…`→`01KTA5CTYY…` | YES | 13 | `01KS8BQXDA…`→`01KS8AK8YZ…` | YES |
| 4 | `01KVBNCFN1…`→`01KVBMHJJM…` | YES* | 14 | `01KS9FTT75…`→`01KS9EAC6F…` | YES |
| 5 | `01KSB4N0VV…`→`01KSB3T2WF…` | YES | 15 | `01KVHR825M…`→`01KV0CDJ1S…` | YES |
| 6 | `01KTYS2XNK…`→`01KTYN42A9…` | YES | 16 | `01KTA5CTYY…`→`01KTA4K8SC…` | YES |
| 7 | `01KT0ZYHCV…`→`01KSETVHGH…` | YES | 17 | `01KT68DSXY…`→`01KT3H45YP…` | YES |
| 8 | `01KSQHC9ND…`→`01KSQB8W8Q…` | YES | 18 | `01KVQ0YAPS…`→`01KTXT09EY…` | YES |
| 9 | `01KTYCZDF2…`→`01KTY2595Y…` | YES | 19 | `01KTPPPRRH…`→`01KTJ0WSCY…` | YES |
| 10 | `01KSWYKDD6…`→`01KSWY4SHV…` | YES* | 20 | `01KSTE3FFQ…`→`01KSTDBPRG…` | YES |

  (*) #4/#10 were correction-shaped and are claimed by R1c in the final rule
  set (sampled before the R1c/R6 split was finalized); their verdicts carry over.
- Mapping: `related`, weight 0.9, metadata `{kind: "content_reference", rule: "R6"}`,
  direction referrer → referee stored as source → target.
- Traversal factor proposal (kind branch): **forward (referrer→referee) 0.75,
  backward (referee→referrer) 0.9** — when an old node matches, prefer the
  newer trace that explicitly built on it; the referee is context for the
  referrer at slightly lower strength. (Plain 0.65 symmetric is an acceptable
  fallback if the engine wants fewer knobs; the weight already exceeds
  implicit-edge weights.)
- Verdict: **go**.

## 3. Go/no-go summary

| Rule | Type / kind | Direction | Both-active pairs | New-pair fraction | Co-retrieval redundancy | Sampled precision | Verdict |
|------|-------------|-----------|-------------------|-------------------|--------------------------|-------------------|---------|
| R1a transitive supersedes | supersedes | newest→oldest | 1 | 1.0 | 0.0 | 0/1 | **no-go** |
| R1b correction (heuristic target) | supersedes | correction→recalled | 7 | 0.0 | 0.43 | 1/7 (loose: 0/20) | **no-go** |
| R1c correction names ULID | supersedes, kind=content_correction | correction→named node | 9 | 0.89 | 0.0 | 9/9 | **go** (write-path primary) |
| R2a root-cause→resolution | caused, kind=failure_resolution | root_cause→fix | 4 | 0.5 | 0.75 | 4/4 (loose: 4/20 → tightened) | **go** |
| R2b cause→failure-evidence | caused | cause→effect | 0 | — | — | — | **no-go** (no raw material) |
| R3 procedure chains | requires | later→earlier | 250 | 0.81 | 0.18 | 0/20 | **no-go** |
| R4a same commit | related, basis=same_commit | undirected | 206 | 0.23 | 0.053 | 20/20 | **go** |
| R4b shared files (fanout≤20) | related, basis=shared_files | undirected | 9,992 | 0.89 | 0.056 | 20/20 | **go** |
| R5a schema derived-from | related, kind=derived_from (annotation) | schema→trace | 5,223 | 0.0 | 0.047 | 20/20 (concepts 0/16 → excluded) | **go** (annotate + factors) |
| R5b trace informed-by | related | trace→recalled | 39,229 | 0.0 | 0.25 | ~0.5–0.65 | **no-go** |
| R6 content ULID reference | related, kind=content_reference | referrer→referee | 197 | 0.53 | 0.44 | 20/20 | **go** |

## 4. Backfill projection and expected impact on graph non-redundancy

New edges from the go-set on this snapshot (additive; related-type rules add
edges only on previously unconnected pairs; typed rules may add a typed edge
alongside an existing related edge since `UNIQUE(source_id,target_id,type)`
permits it; R5a adds no rows, only metadata annotation on 5,223 existing
procedural edges):

| Source | New edges | Of which previously unconnected pairs |
|--------|-----------|----------------------------------------|
| R1c supersedes | 9 | 8 |
| R2a caused | 4 | 2 |
| R4a related/same_commit | 47 | 47 |
| R4b related/shared_files | 8,863 | 8,863 |
| R6 related/content_reference | 105 | 105 |
| **Total** | **≈9,028** | **≈9,025** |

- **Edges**: 102,661 → ≈111,689.
- **Strictly-typed share** (caused+contradicts+supersedes+requires):
  3,305 → 3,318 = **3.0%** (from 3.22%). The honest headline: this corpus's
  authoring style (self-contained lessons, decaying teach targets) bounds
  strictly-typed volume; chasing a large typed share via heuristics would
  mean shipping the 0/20-precision rules this mining rejected.
- **Provenance-derived share** (typed types + kind/basis ∈ {derived_from,
  content_correction, failure_resolution, content_reference, same_commit,
  shared_files}): 3,305 → **≈17,556 / 111,689 = 15.7%** (≈4.9× today's 3.22%),
  vs 84% remaining co-retrieval-echo bases (implicit_recall_feedback, cluster,
  co_access).
- **Non-redundant connectivity**: unique both-active connected pairs
  92,192 → ≈101,217 (**+9.8%**), and ~94–95% of the added pairs were *never
  co-retrieved* in 46.8k recorded recall events (volume-weighted co-retrieval
  redundancy of the go-set ≈ 0.056) — versus 24.5% redundancy for the
  implicit-feedback edges that dominate the graph today. This is the direct
  mechanism by which the graph channel earns non-redundant rank: it can now
  surface same-file/same-commit/lineage neighbors that bm25/vector do not
  co-surface for the query.
- **Causal channel**: 0 → 4 `caused` edges from backfill plus ongoing
  write-path accrual (R2a pattern recurs in supervision sessions; R1c-style
  corrections accrue ~1/day). Causal-mode traversal (`retrieval.py:688`)
  stops being a no-op.
- **Latency**: +8.8% edges, concentrated on file/commit-bearing nodes
  (≤ ~8 extra edges per affected node on average; per-file fanout cap 20
  bounds the worst case). Traversal work is per-neighbor constant-time factor
  lookup; the kind branches add dict lookups only. No per-query heavy work —
  compatible with the p50 ~180 ms recall budget.

## 5. Input contract for `typed-edge-engine` (the go-rules, final parameters)

Implement exactly these five rules as pure functions (new module
`edge_derivation.py`), applied at write time in `server.py` hooks and reused
by the backfill CLI. All emitted edges carry
`metadata: {rule, basis|kind, evidence…}` as specified per rule above;
weights as specified. Never create an edge to a decayed node; never create
both-RA pairs (R4a/R4b); upsert semantics = existing
`_upsert_weighted_connection` (weight max-merge, metadata merge).

1. **R1c** on `memory_remember`: adjacent correction-phrase + ULID regex (see
   §R1c) over new trace content → `supersedes` each named existing node,
   weight 0.9, kind=content_correction. (Backfill: 9 edges.)
2. **R2a** on `memory_remember`: if the new trace is solution-labeled, find
   the nearest root_cause-labeled (or structural_fail/reopened-outcome,
   non-solution, non-RA) trace in the same (scope, task) or (scope,
   session_id) written 0–3600 s earlier → `caused` root_cause→new trace,
   weight 0.9, kind=failure_resolution. (Backfill: 4 edges.)
3. **R4a** on `memory_remember` (and backfill): same 7-char commit prefix,
   group ≤10 → `related` basis=same_commit, weight 0.9.
4. **R4b** on `memory_remember` (and backfill): shared `context.files`
   entries with fanout ≤ 20 → `related` basis=shared_files,
   weight `min(0.95, 0.5 + 0.15×n_shared)`.
5. **R5a** at consolidation/backfill: annotate schema→source-trace edges with
   kind=derived_from (no new rows).
6. **R6** on `memory_remember` (and backfill): non-correction content ULID
   references → `related` kind=content_reference, weight 0.9, referrer→referee.

`_traversal` additions (kind branches before the generic `related` arm,
mirroring the `rejected_alternative` precedent — no schema migration, MCP API
unchanged):

| Edge | forward factor | backward factor | notes |
|------|----------------|-----------------|-------|
| related + kind=derived_from (schema→trace) | 0.5 | 0.95 | instance ↑ distilled schema |
| related + kind=content_reference (referrer→referee) | 0.75 | 0.9 | old node ↑ newer citing trace |
| related + basis same_commit / shared_files | 0.65 (unchanged) | 0.65 | strength carried in weight |
| caused kind=failure_resolution | existing 0.75 / causal 0.25 | existing 0.85 / causal 1.0 | fills causal mode |
| supersedes kind=content_correction | existing 0.35 | existing 1.15 | superseded penalty correct here |

Both traversal-factor changes (derived_from, content_reference) alter ranking
shape on existing/near-existing edges and must clear the replay-harness
no-regression gate (hit@5 / MRR ≥ 0.99× baseline) before shipping; the edge
*additions* are strictly additive and low-risk but go through the same gate.

## 6. Cross-cutting findings (recorded for the parent goal)

1. **Decay is the enemy of correction lineage.** Teach and content-corrections
   tank the target's stats; the decay sweep then removes it; traversal skips
   decayed neighbors (`retrieval.py:525`). Supersedes edges therefore rarely
   contribute traversal signal — their surviving effect is the
   superseding-source rank boost. Any future "correction lineage expansion"
   must run at write time, not backfill time.
2. **Heuristic correction-target inference is unsafe** (R1b: 1/7, 0/20).
   Corrections need explicit targets: `memory_teach(trace_id)` or a named
   ULID (R1c: 9/9). This is a measured argument for the existing
   teach-on-correction guidance, and for agents citing node ULIDs in
   follow-up traces (R6: 20/20, and the citation habit visibly grew over the
   corpus period).
3. **`procedure_id` ≠ step chain** on this corpus (0/20 as `requires`;
   4/3,255 nodes carry step metadata). `requires` stays empty until real step
   metadata is written.
4. **Concept clusters are degenerate at the top end** (up to 2,209 sources;
   concept→member "derived_from" scored 0/16). Schema grouping (exact field
   match) is sound (20/20). Cluster-quality repair is a separate upstream
   issue; typed edges must not be built on those clusters as-is.
5. **The corpus's non-redundant signal lives in cross-batch, cross-ticket,
   cross-time structure fields** (files, commit, explicit citations), not in
   co-recall — exactly the axis on which the current 97%-generic edge
   inventory is weakest (co-retrieval redundancy: implicit edges 0.245 vs
   go-set 0.056).

## 7. Reproduction

```
python3 scripts/mine_typed_edges.py \
  --db /tmp/lm_snapshot.sqlite3 \
  --report /tmp/typed_edge_report.json \
  --samples /tmp/typed_edge_samples.md \
  --seed 42 --sample-size 20
```

The script is stdlib-only, opens the DB with `mode=ro` + `PRAGMA query_only`,
performs no writes anywhere, and is deterministic for a given (db, seed).
