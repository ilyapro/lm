# Confirmatory holdout v4

Status: **preregistered, source-unprobed, accruable, and unsealed**.

This namespace freezes the confirmatory evidence protocol before any v4 source
availability probe, packet construction, production repair, or semantic
evaluation. It contains no source rows, availability counts, case records,
identity tokens, outcomes, packet, or evaluation result.

The immutable protocol documents are:

- `POLICY.md`: authority, schedule, runtime-segment, bootstrap, selection,
  privacy, sealing, and reader rules.
- `analysis-plan.json`: machine-readable pins, predicates, domains, floors,
  state transitions, gates, uncertainty, and receipt allowlists.
- `README.md`: this status and operator boundary.

`tests/test_confirmatory_evidence_protocol_v4.py` pins the exact bytes and
semantics of all three documents. They may not be amended in place. A protocol
correction requires a new parent-authorized namespace; observed accrual belongs
only in append-only `segments/` and `probes/` artifacts defined by this
protocol.

## Why v4 exists

V3 was frozen with one unreachable transition. Its policy and plan authorized a
count-free slot-0 `active-services-state-change` closure before the initial
source-binding attestation existed, but its only closure schema required both
`prior_source_binding_core_sha256_and_bytes` and
`observed_source_binding_core_sha256_and_bytes`. The initial segment is
attested by the release watermark alone, so it has a prior service tuple and no
prior source binding at all. A strict validator therefore had nothing valid to
emit on that branch, and inventing a value to fill the field would have been a
provenance claim about a binding that never happened.

V4 is a self-contained successor namespace rather than an overlay or a patch.
It preserves the v3 runtime boundary, the exact 29-slot calendar, the grace
window, the absolute horizon, the numeric floors, the privacy rules, the
default-off replay control, the sealing sequence, and the one-shot reader
authorities. It replaces exactly one thing: the unreachable bootstrap
transition. Every v3 document stays byte-identical and non-authoritative,
hash-pinned here as a retired predecessor.

## Frozen calendar

The first selection slot is `2026-08-17T00:00:00Z`. Slots repeat every 86,400
seconds through slot 28 at `2026-09-14T00:00:00Z`. Each slot has the half-open
execution window `[slot_at, slot_at + 21,600 seconds)`. The absolute horizon is
`2026-09-14T06:00:00Z`; it never moves when a runtime segment changes, and no
later knowledge of source availability may re-anchor, extend, or shorten it.

Every valid slot selects the complete source-qualified event population with
`created_at` strictly after both the fixed release-control boundary and the
active segment boundary, and no later than the scheduled slot instant. Launch
or capture time never changes that upper bound. A valid below-floor receipt
before the final slot is nonterminal. A provisional first pass is never exposed
for discretionary action: in the same invocation a final runtime check must
survive, a durable one-shot seal-consumption marker is created, the sealer is
spawned behind a closed gate, and the staged ready receipt/ledger pair is
validated over inherited IPC and atomically exposed before the gate opens. A
proven runtime change produces only a count-free closure resolution for that
slot. A grace window that closes without any validator-valid resolution is
terminal schedule-integrity failure: later evidence could not prove that it
came from the first passing scheduled population. If the final valid resolution
is below floor or segment-closed, horizon expiry is terminal insufficient
evidence and requires parent escalation.

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

## Two honest bootstrap branches

Slot 0 runs the sole row-blind initial ceremony and resolves into exactly one
of two mutually exclusive branches, selected only by whether the complete
stable observed active-services digest equals the pinned watermark digest.

When they match, the ceremony publishes the ordinary `source-binding-attestation`
and the same invocation continues into its ordinary probe resolution.

When they differ, the sole valid output is the distinct count-free
`initial-runtime-change-closure`. It binds the watermark prior tuple, the
complete stable observed tuple, the recomputable observed source-binding core,
the initial attempt marker, the runtime observer, the analysis plan, the
accrual-ledger predecessor, and the fixed slot-0 schedule instants. Its sole
mismatch kind is `active-services-state-change`. The schema declares **no**
prior-source-binding member at any nesting depth: not a null, not a sentinel,
not a placeholder digest, not a copy of the observed core. Absence is the exact
representation of a value that never existed, and any receipt that carries such
a member is invalid under the recursive unknown-member rule.

The pre-binding closure reads no source row, computes no key, captures no
snapshot, carries no count, and constructs no active segment. It consumes slot
0, retains a typed pre-binding predecessor, clears every latch and source
authority, sets the next probe slot to 1, and continues only through the
ordinary successor ceremony. The ordinary two-core `slot-segment-closed`
resolution stays exactly as it was and remains the sole closure once a valid
binding exists; neither schema is a permitted spelling of the other.

## Authority boundary

The fatal v2 attempt, the retired v3 protocol, and all of their authorities
remain unusable. Every v4 domain, source alias ID, sealing authority, and
semantic reader is a fresh identity; nothing is inherited, aliased, or renamed
from a retired lineage, and a retired artifact can never validate as v4
evidence. The unchanged repair design stays byte-pinned at SHA-256
`76d34d33b7be57230ef0988662060fe70eea229f9b07f2cd5b8a2354600a79f5`.
V4 supersedes the retired evidence protocols: readiness lifecycle,
protocol/plan pins, packet and reader registry, domains and seeds,
replayability-based floor and measurement eligibility, and production seed
validation, receipt/evaluator transport, and the reference-arm binding. It does
not change repair invariants R1--R15, defaults, storage, delivery, metric
meanings or arithmetic operators, numeric gates, or production behavior. The
input-only replayability rule intentionally changes the admissible cohort and
therefore its denominator; the reference is the pinned default-off replay
control, never the serving accrual build.

Scheduled probes are repeatable aggregate-only mechanical operations and
consume no packet or semantic-reader authority. Packet sealing has one
no-overwrite launch. After a seal commit strictly precedes the repair subtree,
the only semantic readers are `confirmatory-shadow-v4-eval` and
`confirmatory-holdout-v4-eval`, each with one launch and one forward pass.
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
packet verifier, inspect a private source, or run a v4 semantic reader.
