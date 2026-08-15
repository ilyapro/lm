# Confirmatory holdout v3

Status: **preregistered, source-unprobed, accruable, and unsealed**.

This namespace freezes the confirmatory evidence protocol before any v3 source
availability probe, packet construction, production repair, or semantic
evaluation. It contains no source rows, availability counts, case records,
identity tokens, outcomes, packet, or evaluation result.

The immutable protocol documents are:

- `POLICY.md`: authority, schedule, runtime-segment, selection, privacy,
  sealing, and reader rules.
- `analysis-plan.json`: machine-readable pins, predicates, domains, floors,
  state transitions, gates, uncertainty, and receipt allowlists.
- `README.md`: this status and operator boundary.

`tests/test_confirmatory_evidence_protocol_v3.py` pins the exact bytes and
semantics of all three documents. They may not be amended in place. A protocol
correction requires a new parent-authorized namespace; observed accrual belongs
only in append-only `segments/` and `probes/` artifacts defined by this
protocol.

## Frozen calendar

The first selection slot is `2026-08-17T00:00:00Z`. Slots repeat every 86,400
seconds through slot 28 at `2026-09-14T00:00:00Z`. Each slot has the half-open
execution window `[slot_at, slot_at + 21,600 seconds)`. The absolute horizon is
`2026-09-14T06:00:00Z`; it never moves when a runtime segment changes.

Every valid slot selects the complete source-qualified event population with
`created_at` strictly after both the fixed release-control boundary and the
active segment boundary, and no later than the scheduled slot instant. Launch
or capture time never changes that upper bound. A valid below-floor receipt
before the final slot is nonterminal. A provisional first pass is never exposed
for discretionary action: in the same invocation a final runtime check must
survive, a durable one-shot seal-consumption marker is created, the sealer is
spawned behind a closed gate, and the staged ready receipt/ledger pair is
validated over inherited IPC and atomically exposed before the gate opens. A proven
runtime change produces only a count-free closure resolution for that slot.
A grace window that closes without any validator-valid resolution is terminal
schedule-integrity failure: later evidence could not prove that it came from
the first passing scheduled population. If the final valid resolution is below
floor or segment-closed, horizon expiry is terminal insufficient evidence and
requires parent escalation.

## Runtime boundary

The initial segment is bound to the validator-valid release control watermark
at SHA-256
`a9e6e6a4dbd6178345e11e951e973a1af9c062fb50f8dd7bd51528d44217a86f`.
For protocol compatibility its observed runtime-provenance lower bound
`2026-08-14T15:03:46.793603Z` is named `release_effective_at`; this is not a
claim that the replay-control candidate was deployed. The serving runtime
supplies event population only. Replay control remains commit
`46a9951842512333b0896370056d07a9e1c25bdf`, tree
`c1606671b9d13bed78c21b7d2f9a4bb75a3d1c1c`.

The watermark service set is unaliased; it does not map array indexes to
`local` or `alt`. A future row-blind observer must privately prove the fixed
alias-authority/service/database bijection and publish only its allowlisted
hash result. Any change in the complete admitted service set, that binding, boot, serving
build or implementation identities, sanitized configuration, or either
effective legacy repeat control closes the entire active segment. A successor
requires an append-only attestation and the plan's deterministic max-of-launcher-
and-source-clocks boundary ceremony. It has one attempt, must start within 30
seconds after a slot closure, must finish before the next due slot, and fails
terminally rather than choosing another boundary. Counts, snapshots, and
provisional results never cross segments; the calendar and absolute horizon
remain unchanged. There is no inter-slot polling; every change is rejected by
the next fixed-slot precheck before rows can count.

## Authority boundary

The fatal v2 attempt and all v2 authorities remain unusable. The unchanged
repair design stays byte-pinned at SHA-256
`76d34d33b7be57230ef0988662060fe70eea229f9b07f2cd5b8a2354600a79f5`.
V3 supersedes the retired v2 evidence protocol: its readiness lifecycle,
protocol/plan pins, packet and reader registry, domains and seeds,
replayability-based floor and measurement eligibility, and production seed
validation, receipt/evaluator transport, and the reference-arm binding. It does
not change repair invariants R1--R15, defaults, storage, delivery, metric
meanings or arithmetic operators, numeric gates, or production behavior. V3's
input-only replayability rule intentionally changes the admissible cohort and
therefore its denominator; the reference is the pinned default-off replay
control, never the serving accrual build.

Scheduled probes are repeatable aggregate-only mechanical operations and
consume no packet or semantic-reader authority. Packet sealing has one
no-overwrite launch. After a seal commit strictly precedes the repair subtree,
the only semantic readers are `confirmatory-shadow-v3-eval` and
`confirmatory-holdout-v3-eval`, each with one launch and one forward pass.
The frozen implementation stack, development calibration, and public
evaluation are connected by exact typed receipts; development and public each
have one no-overwrite pre-reader launch-attempt marker and a 30-second start
deadline. After development passes, every terminal public-eval outcome
unconditionally advances to shadow. Shadow runs under embargo with a fixed watchdog; every
terminal shadow outcome unconditionally advances to holdout on the identical
frozen stack. Controller-synthesized launch-integrity, launch-failure, invalid,
inconclusive, timeout, crash, and forced-termination receipts prevent a failed
reader from blocking the sequence. Each next attempt marker is a durable
30-second controller handoff. Exactly both embedded terminal receipts release
together through one private, fsynced envelope and one atomic no-replace
visibility operation; neither receipt is independently visible first.

Static verification may read the public metadata and policy files pinned by the
analysis plan. It must not open either consumed corpus, invoke a consumed
packet verifier, inspect a private source, or run a v3 semantic reader.
