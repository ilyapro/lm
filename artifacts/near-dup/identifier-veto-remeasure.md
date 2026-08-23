# The identifier veto, measured on the delivery layer: off vs on

Produced 2026-08-23T18:06Z and 18:27Z by two runs of
`scripts/recall_dup_slot_measure.py` at commit `0dc0e28f8557` — the first commit
in which the veto compares identifiers by **token equality** rather than
substring containment. The machine artifacts are
[`dup-slot-measurement-veto-off.json`](dup-slot-measurement-veto-off.json) and
[`dup-slot-measurement-veto-on.json`](dup-slot-measurement-veto-on.json); every
number below is read out of those two files, and the commands that produce and
compare them are at the end.

The question is not whether the veto blocks identifier-losing collapses — it is
built to. The question is what it costs: the veto also blocks honest collapses,
because an honest repeat may quote an identifier its bearer does not. That price
is stated here in counts and shares, not assumed. This revision adds a second
question, because the presence rule changed underneath the previous numbers:
**how much of that price is new.**

## This is a controlled contrast, not an indicative one

The previous edition of this file was measured at commit `a00bcace50ec`, under
the substring rule. Re-measuring it would normally mean a fresh snapshot and
therefore a fresh corpus, which would make every comparison against those numbers
cross-snapshot — indicative only, since the live database moves continuously.

That is not what happened here, and the reason is worth recording. The snapshot
and the warmed base copy of the `a00bcace50ec` runs were still on disk in the
workdir those runs actually used, `/tmp/lm-dupslot-veto`, and both hash to
exactly what the committed artifacts record:

| file | sha256 | recorded in the `a00bcace50ec` artifacts |
| --- | --- | --- |
| `/tmp/lm-dupslot-veto/snapshot.sqlite3` | `717ab6b8442730a95a3bd01991e90fba9b477ac8b7bcdf6d258b52e3f065f208` | identical |
| `/tmp/lm-dupslot-veto/base.sqlite3` | `c2120763a6c3bc5ae9ef198a2201c00e9b3897e988e41a708e25b2cc32c21286` | identical |

So both arms were re-run on those same bytes with `--reuse-snapshot
--reuse-base`, and the substring-rule numbers and the token-rule numbers below
describe **one corpus, one frozen selection and one ranking** — differing only in
the presence rule inside `near_dup.identifiers_absent_from`. That is a
one-variable contrast, and it is what lets this file answer "what did tightening
the rule cost" with a measurement rather than an estimate.

(The belief that the snapshot was gone came from inspecting `/tmp/lm-dupslot`,
which is the workdir of the *earlier*, pre-veto baseline run. The veto runs never
used it. Hash the workdir the artifact names.)

Two consequences are load-bearing below:

* the **veto-OFF arm's collapse set cannot depend on the rule at all**, because
  `build_duplicate_map` never enters the veto branch when the valve is off — so
  re-running it is a determinism check, and it passed exactly (52 pairs, 201
  collapsed slots, duplicate-slot share equal to six decimals, across two
  different commits);
* the **veto-ON arm can be differenced directly** against the committed
  substring-rule arm, which is the price question stated as an experiment.

## The experiment is one variable

| | veto OFF (`LM_NEAR_DUP_IDENTIFIER_VETO=0`) | veto ON (unset → shipped default) |
| --- | --- | --- |
| snapshot sha256 | `717ab6b8442730a95a3bd01991e90fba9b477ac8b7bcdf6d258b52e3f065f208` | same |
| warmed base sha256 | `c2120763a6c3bc5ae9ef198a2201c00e9b3897e988e41a708e25b2cc32c21286` | same |
| selection digest | `19877a4383b2e2d20fda2967ca3615fa3c806443ec1a6f72b38e95403bf2e070` | same |
| queries replayed per arm | 2,500 (2,000 traffic-weighted + 500 distinct-query) | same |
| ranking identical across arms | true | true |
| corpus constant through both arms | true | true |
| valve observed inside each arm | `{before_rollback: false, after_shipped_default: false}` | `{before_rollback: true, after_shipped_default: true}` |

Both runs opened the snapshot with `--reuse-snapshot` and the warmed base with
`--reuse-base`, so the two runs read the identical corpus **bytes**, not merely
the identical corpus counts. The live database was never opened by either run;
the live MCP server and the dashboard were never signalled, stopped or
restarted; nothing was written to the live or alt database.
`LM_DRAIN_NEAR_DUP_SUPERSEDES` was not set in either run and is enabled nowhere
in this repository.

Each run reports the veto as observed by `near_dup.identifier_veto_enabled()`
*inside* each arm's environment, and refuses to write an artifact if what ran
disagrees with what was asked for. The rollback arm (`LM_RECALL_NEAR_DUP_COSINE=0`)
is byte-identical between the two runs — at cosine 0 the map is empty before the
veto branch is reached — which is why it serves as the shared baseline both runs
are read against.

## Side by side

Traffic-weighted stratum (2,000 replayed real requests, 13,254 delivered slots);
the distinct-query stratum (500 requests, 3,808 slots) is in brackets.

| | rollback (no collapse) | veto OFF | veto ON |
| --- | ---: | ---: | ---: |
| duplicate-slot share | 1.947% [2.600%] | **0.822%** [1.234%] | **1.509%** [1.812%] |
| occupied repeat slots | 258 [99] | 109 [47] | 200 [69] |
| freed repeat slots | 28 [17] | 177 [69] | 86 [47] |
| relative reduction vs rollback | — | 57.8% [52.5%] | 22.5% [30.3%] |
| answers with a repeat in a slot | 186 = 9.3% [12.4%] | 103 = 5.1% [7.8%] | 149 = 7.4% [10.2%] |
| collapsed slots (both strata) | 0 | 201 | 88 |
| distinct collapsed pairs | 0 | 52 | 22 |
| chars replaced by a stub | 0 | 446,220 | 289,699 |
| length-guard near-miss collapses | 0 | 26 | 2 |

The repeat ground truth (286 repeat slots in the traffic stratum, 116 in the
distinct one) is computed once from the ranked answers and is the same for both
runs, so the arms differ only in what they *deliver* for a repeat. The length
guard vetoed 156 collapses in both arms and no repeat was left uncollapsed for
want of a vector in either.

### Identifier-losing pairs, under every definition in play

| definition | veto OFF | veto ON |
| --- | ---: | ---: |
| harness metric — `identifiers_lost` ≥ 3 (token ≥ 5 chars **carrying a digit**) | 5 pairs / 8 slots | **0 pairs / 0 slots** |
| veto's own, shipped — `near_dup.identifiers_absent_from`, **token equality** | 30 pairs / 113 slots | **0 pairs / 0 slots** |
| the veto's *historical* rule — same extraction, **substring containment** | 30 pairs / 113 slots | 0 pairs / 0 slots |

The first row is kept unchanged, and deliberately: it is the instrument the
frozen baseline `dup-slot-measurement-honest-bearer.json` was counted with, so
it is what makes this run comparable to that one. It requires a digit inside the
token and is therefore structurally blind to the `layer-fauna` / `layer-actors`
class that motivated the veto. The second row is the definition the map actually
enforces, imported from `living_memory.near_dup` rather than re-implemented, so
"zero" means zero by the shipped rule. The third row is measured by
`identifiers_absent_by_substring` in the harness, which shares the imported
extraction and differs from the shipped rule *only* in the presence test — it
exists to answer the next section and is reachable from nothing in `src/`.

Under the veto's own definition the residue with the veto on is **zero**: no
collapsed pair in the veto-on run hides an identifier its bearer lacks, so there
is no residual class to explain pair by pair.

## What the rule change itself did: token equality vs substring containment

Scored over the veto-OFF arm's 52 collapsed pairs / 201 slots — the honest place
to ask, because that set is the same whichever rule is in force.

| | pairs losing ≥1 identifier | slots | identifier tokens judged absent |
| --- | ---: | ---: | ---: |
| substring containment (historical) | 30 | 113 | 153 |
| token equality (shipped) | 30 | 113 | 167 |
| **difference** | **0** | **0** | **+14** |

Pairs the substring rule caught and the token rule does not: **0**, as it must
be — an extracted token is always a substring of the text it came from, so a
token the bearer *names* is also a token the bearer *contains*. The token rule is
strictly stricter, and the artifact asserts that rather than assuming it
(`presence_rule_comparison.token_rule_is_strictly_stricter`).

**So on the delivery layer, the tighter rule costs nothing new.** The 14 extra
identifier tokens it sees are spread over 3 pairs, and all 3 were already vetoed
by the substring rule for other identifiers, so not one collapse verdict changes.
The direct check confirms it end to end: the veto-ON arm under token equality
collapses **the same 22 pairs / 88 slots** as the committed veto-ON arm under
substring containment, on the same snapshot bytes — identical pair set, identical
duplicate-slot share. Of the price quoted in the next section, **0.000 pp is new.**

The class the token rule newly sees is nonetheless real, and it is exactly the
one the goal names — a bearer that spells a longer name containing the
candidate's:

| pair (cos) | tokens only token-equality sees | already vetoed by substring? |
| --- | --- | ---: |
| `01KS998AZ5…` → `01KS96N74Z…` (0.9616) | `README.md`, `status.md`, `284f1cd`, `f106824a`, `dialogue/planner` | yes |
| `01KVDG0BK9…` → `01KVDHWC4W…` (0.9517) | `…/a08-runtime-checkpoint-smoke`, `_node_exec_the-ceil_…_a08-runtime-checkpoint-smoke`, `train_eval`, `runtime_checkpoint_test`, `broadened_search/score_ledger.json` | yes |
| `01KS1V4DD5…` → `01KS1VAHKB…` (0.9556) | `training_run`, `validate_ocpa_training_run`, `validate_holdout_access_log`, `validate_command_surface` | yes |

`README.md` sits inside a bearer's longer path; `train_eval` and `training_run`
sit inside `train_eval_*` and `validate_ocpa_training_run`; the worktree name
contains the node name. This is the same shape as the goal's honest example — a
bearer spelling `src/living_memory/near_dup.py` no longer covers a candidate
spelling the bare `near_dup.py`, and that pair now vetoes. On *this* corpus the
class only ever appears alongside identifiers containment already caught, so it
changes no verdict; on the pre-hygiene backup it is what let two distinct
goal-tree nodes collapse at cosine 0.99350 and 0.99064, which is why the rule was
tightened. The delivery layer is where the tightening is free; the drain band is
where it pays.

### The price that is real regardless of the rule

The veto is not free even though the tightening was. Against the rollback
baseline the collapse saves 1.125 pp of duplicate-slot share with the veto off
(1.947% → 0.822%) and 0.438 pp with it on (1.947% → 1.509%), so **the veto hands
back 0.687 pp, 61% of the collapse's slot gain** (traffic stratum; 1.366 pp →
0.788 pp, 42% handed back, in the distinct stratum). In characters, the collapse
moves 21,990 chars off repeat slots with the veto off and 13,278 with it on
(traffic stratum). What the veto keeps is the part of the gain that was buying
nothing but a hidden fact.

### What the veto blocked: 30 pairs, read one by one

The veto-on collapsed pairs are a strict subset of the veto-off ones: **30 pairs
/ 113 slots** stopped collapsing, **0 pairs appeared** that the veto-off run did
not have, and **0 blocked pairs lacked a veto reason**. In slot terms the veto
blocked 113 of the 201 collapses the unvetoed map made — **56.2%** of the
collapsed slots, or 0.66% of all 17,062 delivered slots in the two strata.

This blocked set is *identical* to the one the substring-rule runs produced —
verified pair for pair, not assumed — so the hand reading recorded at
`a00bcace50ec` is a reading of these same 30 pairs and carries over unchanged:

* **By the frozen baseline's own two flags** — the harness identifier flag (≥3
  digit-bearing identifiers lost) and the lexical flag (token Jaccard < 0.60) —
  22 of the 30 blocked pairs (**92 slots, 45.8% of the veto-off collapsed slots**)
  would have looked honest: neither flag fires on them. That is the honest-loss
  upper bound, and it is large.
* **By reading all 30 pairs**, the upper bound does not survive. Every one of the
  22 unflagged pairs is two different facts:
  * corrections naming *different* traces — `Correction for … trace
    01KS1B30E6BSWV44EECRX18NCF` vs `… trace 01KS1B35X482R3Z7YDAVC65NZT`, identical
    to the last comma otherwise (cos 0.9986, Jaccard 0.95);
  * corrections about *different* original task_patterns — `ed1cc80ceec1d16b`
    ('verification_critique') vs `7bdd9349a6d3fc15` ('critique') (cos 0.9941);
  * `[file-chunk]`/`[file-summary]` nodes for *different versions of the same file*
    — `node.sh` at sha `301c5c42…` vs `526e7e1a…`, `goal.sh` at 124,027 bytes /
    3,247 lines vs 123,942 / 3,246 (cos 0.9995–0.9999, Jaccard 0.94–0.99);
  * outcome lines for *different goal nodes and different tickets* — `EZ-13512` vs
    `EZ-13560`; `…/final-verification/land-fig8-closed-state-on-verification-branch`
    vs `…-cherry-pick`; `…/tooltip-implementation/_verify` vs
    `…/tooltip-experiment-view-ui-actions/_verify`.
  These are exactly the template-dominated pairs the root goal describes: the
  template carries the vector, the one token that carries the fact is a rounding
  error in it, and the lexical flag cannot see it either.
* **The debatable residue is 4 pairs / 10 slots** (5.0% of the veto-off collapsed
  slots), and this run identifies them mechanically rather than by eye: they are
  the only blocked pairs whose *entire* lost-identifier set is an elapsed time —
  `201.9s`, `178.3s`, `296.2s`, `358.1s`, at cosine 0.9850–0.9992. They are
  different runs of the same goal, so calling them different facts is defensible,
  but an agent reading recall arguably loses nothing when they collapse. This is
  the honest cost of the veto on this corpus: **at most 4 pairs / 10 slots, and
  zero pairs that are unambiguously the same fact twice.**

The other 8 blocked pairs were already flagged by the baseline's own instruments
(different reopen lessons `__prompt_format_2062600_decision` vs
`__prompt_format_3480223_decision`; different supervisor scans, PIDs and clock
times; different worktrees `…a08-runtime-checkpoint-smoke` vs
`…-smoke-bounded-manifest`), and read the same way.

### The ≥0.99 band, as context for the drain

Of the 52 pairs the unvetoed map collapsed, **13 pairs (50 slots) sit at cosine
≥ 0.99 — and 12 of those 13 hide an identifier under the veto's token
definition.** All 13 are `trace`/`trace`, and exactly 1 survives with the veto on.
The band the drain would automate is therefore not clean by threshold alone on
this corpus; the identifier veto is what makes it clean. (This is the delivery
path's pair population, not the drain's — the drain additionally requires same
level, same scope and an older bearer — so it is evidence for the drain's
residual risk, not a substitute for `drain-simulation.json`.)

## Drift from the 2026-08-23 baseline artifact

The within-run comparison above is the claim; the frozen baseline
(`dup-slot-measurement-honest-bearer.json`) is context, and the corpus moved
between them. This remeasure runs on the same snapshot as the `a00bcace50ec`
edition, so this table is unchanged from it.

| | baseline (honest-bearer) | this remeasure |
| --- | ---: | ---: |
| snapshot latest event | 2026-08-23T13:45:23Z | 2026-08-23T15:50:12Z |
| recall_events / active nodes | 56,605 / 13,456 | 56,687 / 13,340 |
| repeat slots, traffic stratum | 609 | 286 |
| occupied repeat slots, rollback | 317 (2.392%) | 258 (1.947%) |
| collapsed slots, unvetoed | 250 | 201 |
| distinct collapsed pairs, unvetoed | 77 | 52 |
| identifier-losing pairs / slots (harness metric) | 11 / 54 | 5 / 8 |

The frozen selection replays the same 2,500 recorded requests, but 116 active
nodes fewer answer them: the near-duplicate hygiene pass and the corpus ceiling
landed on the live database in between (commit `ce93bae`). Fewer near-duplicates
in the corpus means fewer repeats reach an answer at all — the repeat ground truth
halved — so every absolute count here is smaller than in the baseline artifact.
The baseline's 11 pairs / 54 slots and this run's 5 pairs / 8 slots are the same
metric over different corpora, and neither is a regression against the other.
What *is* comparable, because it is measured inside one snapshot, is 30 → 0 and
5 → 0.

## The decision, per consumer

The numbers confirm the default the map already ships, at the tighter rule.
Nothing under `src/living_memory/**` was edited by this node, and no escalation
is required.

### 1. Recall delivery — veto **ON** by default

`server._near_duplicate_map` passes `identifier_veto=identifier_veto_enabled()`,
which is on unless the valve says otherwise. **Re-affirmed, unchanged, and now
for a stronger reason than before:** the rule change that motivated this
remeasure moves none of these numbers, so the decision does not rest on a
re-argued trade-off.

* It removes 30 pairs / 113 slots of identifier-losing collapses — 56.2% of every
  collapse the unvetoed map made on real traffic — and takes the veto-definition
  residue to exactly 0.
* Its cost is at most 4 pairs / 10 slots of arguably-honest collapse (5.0% of the
  collapsed slots), and 0 pairs that are unambiguously one fact twice.
* Duplicate-slot share still improves on rollback: 1.947% → 1.509% traffic-weighted
  (2.600% → 1.812% distinct), i.e. 39% of the unvetoed gain is retained while the
  part that was hiding facts is given back.
* Tightening containment into token equality added **0 pairs, 0 slots and 0.000 pp**
  here, so nobody is paying at the delivery layer for a fix the drain band needed.
* The asymmetry decides the doubt: a false veto costs one uncollapsed stub in one
  answer; a false pass costs a fact the agent never learns it had.

Rollback for an operator who disagrees: `LM_NEAR_DUP_IDENTIFIER_VETO=0` reproduces
the veto-off column exactly — that column *is* a measurement of the rollback path,
not a projection of it.

### 2. The standalone hygiene script — veto **ON** by default

`scripts/lm_collapse_near_dups.py` collapses nodes in a database: it writes
supersedes edges, and the collapsed node stops being what recall returns. A wrong
collapse there is not a stub with a `content_ref` an agent can follow — it is a
fact removed from the answer set until someone reverts it. The delivery-layer
numbers are the lower bound on how often that would happen: 30 of 52 pairs on this
corpus carried an identifier the bearer lacked, 12 of them above cosine 0.99, where
a hygiene run is most confident. Same valve, same default, same rollback string;
the veto is duplicated into the script rather than imported, because the script's
portability contract forbids importing `living_memory`, and the sibling node
`standalone-mirror-veto` owns that mirror and the test that proves the two agree
verdict-for-verdict. This node did not edit that file.

The token rule matters more here than at delivery, and the reason is visible in
the table above: a hygiene pass runs over the whole corpus rather than over one
answer's slots, so the suffixed-node-name class that never changed a verdict in
these 52 pairs is a class it will meet.

### 3. The recall-path drain — veto **ON** by default, and the drain itself stays OFF

`retrieval._supersede_drained_near_dups` already reads the same
`identifier_veto_enabled()` through the same `build_duplicate_map` call, so "the
veto is on" means one thing in all three consumers. It must stay on there most of
all: the drain's threshold is 0.99, and the ≥0.99 band measured here is 12
identifier-losing pairs out of 13 — a template-dominated `trace`/`trace` pair that
differs only in an id clears 0.99 comfortably, and the veto is the only thing that
stops it. This is also the consumer the token rule was tightened for: the two
pairs that escaped the band under containment were suffixed goal-tree node names.

`LM_DRAIN_NEAR_DUP_SUPERSEDES` remains **off**, in code and in configuration. This
node did not enable it and does not recommend enabling it on the strength of these
numbers: the delivery layer is measured here, the drain's own would-collapse set is
`drain-simulation.json`, and the operator's line — when that report is read and its
≥0.99 band is clean — is:

```bash
LM_DRAIN_NEAR_DUP_SUPERSEDES=1   # operator-only; not set anywhere in this repo
```

## Reproducing this

```bash
# both arms drive the SAME snapshot and the SAME warmed base copy
python3 scripts/recall_dup_slot_measure.py \
    --live ~/.local/share/living-memory/global.sqlite3 \
    --workdir /tmp/lm-dupslot-veto \
    --selection artifacts/near-dup/dup-slot-measurement.json \
    --identifier-veto off --reuse-snapshot --reuse-base \
    --json artifacts/near-dup/dup-slot-measurement-veto-off.json

python3 scripts/recall_dup_slot_measure.py \
    --live ~/.local/share/living-memory/global.sqlite3 \
    --workdir /tmp/lm-dupslot-veto \
    --selection artifacts/near-dup/dup-slot-measurement.json \
    --identifier-veto on --reuse-snapshot --reuse-base \
    --json artifacts/near-dup/dup-slot-measurement-veto-on.json
```

The cross-run assertions, the blocked-pair split and the presence-rule comparison
are computed from the two committed artifacts alone — each lists every pair it
collapsed under `spot_check.collapsed_pairs`, scored under both presence rules:

```python
import json
off = json.load(open("artifacts/near-dup/dup-slot-measurement-veto-off.json"))
on  = json.load(open("artifacts/near-dup/dup-slot-measurement-veto-on.json"))

assert off["snapshot"]["sha256"] == on["snapshot"]["sha256"]                       # one snapshot
assert off["corpus_stability"]["base_sha256"] == on["corpus_stability"]["base_sha256"]
assert off["selection"]["digest"] == on["selection"]["digest"]                     # one frozen selection
assert not any(off["identifier_veto"]["observed_in_arms"].values())                # the valve really was off
assert all(on["identifier_veto"]["observed_in_arms"].values())                     # and really was on
assert on["spot_check"]["veto_identifier_loss"]["pairs_losing_any"] == 0           # zero residue

A = {(p["collapsed"], p["bearer"]): p for p in off["spot_check"]["collapsed_pairs"]}
B = {(p["collapsed"], p["bearer"]) for p in on["spot_check"]["collapsed_pairs"]}
blocked = [A[k] for k in A if k not in B]
honest  = [r for r in blocked if r["identifiers_lost"] < 3 and r["token_jaccard"] >= 0.60]
print(len(blocked), sum(r["slots"] for r in blocked))   # 30 113  -- blocked by the veto
print(len(honest),  sum(r["slots"] for r in honest))    # 22  92  -- honest-loss UPPER bound

# what tightening the presence rule cost, on the arm whose collapse set does not
# depend on the rule
cmp = off["spot_check"]["presence_rule_comparison"]
print(cmp["substring_rule"])            # {'pairs_losing_any': 30, 'slots_losing_any': 113}
print(cmp["token_rule"])                # {'pairs_losing_any': 30, 'slots_losing_any': 113}
print(cmp["newly_seen_by_token_rule"]["pairs"])       # 0  -- no verdict changes here
print(cmp["seen_only_by_substring_rule"]["pairs"])    # 0  -- token equality is strictly stricter
print(sum(p["veto_identifiers_lost"] - p["substring_identifiers_lost"]
          for p in off["spot_check"]["collapsed_pairs"]))   # 14 extra tokens, 3 pairs
```
