# Confirmatory holdout v4 preregistration and access policy

This policy governs only `confirmatory-holdout-v4`. It is a source-blind
preregistration, not a source-readiness result, packet, implementation grant,
or evaluation. Before this freeze no candidate private row, availability
aggregate, identity, source snapshot, sealed case, or outcome was queried or
used, and no runtime observer, availability probe, or scheduled slot was ever
executed under this namespace or its retired predecessor. Neither consumed
corpus nor its verifier was opened. The admitted inputs are only the public
hash-pinned metadata and frozen protocol/design files listed in
`analysis-plan.json`, plus the governing goal and Living Memory decision
context as instructions rather than candidate evidence.

`POLICY.md`, `README.md`, and `analysis-plan.json` become immutable together at
the protocol commit. Their exact byte hashes are locked by
`tests/test_confirmatory_evidence_protocol_v4.py`. Observed evidence may be
appended only in the later paths and schemas declared here. A correction,
changed calendar, changed floor, changed partition, new source, failed seal, or
semantic-reader failure does not authorize an in-place edit or retry; it
requires parent escalation and, if authorized, a new namespace.

## 1. Frozen lineage and authority replacement

The v2 scanner launch ended `fatal-before-readiness`. Its one launch is
consumed; it produced no validator-valid readiness receipt, packet, reader
grant, or repair grant, and its deleted snapshots cannot be reconstructed.

The v3 protocol is retired as `retired-unreachable-initial-bootstrap`. It froze
an internally contradictory initial bootstrap: its policy and plan authorized a
count-free slot-0 `active-services-state-change` closure before any initial
source-binding attestation existed, while its only closure schema required both
`prior_source_binding_core_sha256_and_bytes` and
`observed_source_binding_core_sha256_and_bytes`. For the unbound initial
segment no prior source-binding core is defined, so that authorized transition
had no validator-valid representation. V3 never observed a runtime, opened a
source, produced a receipt, accrued a slot, sealed a packet, or granted a
reader. Its three documents and its protocol test stay byte-identical at the
hashes and sizes pinned in `analysis-plan.json`; they are preserved as
immutable, non-authoritative evidence of that defect and are not edited,
patched, or re-frozen.

The following identifiers are permanently unusable:

- consumed historical readers `shadow-eval` and `replacement-holdout-eval`;
- void v2 reservations `confirmatory-shadow-v2-eval` and
  `confirmatory-holdout-v2-eval`;
- void v3 reservations `confirmatory-shadow-v3-eval` and
  `confirmatory-holdout-v3-eval`;
- v2 source alias IDs `confirmatory-local-v2-ro` and
  `confirmatory-alt-v2-ro`;
- v3 source alias IDs `confirmatory-local-v3-ro` and
  `confirmatory-alt-v3-ro`;
- the v3 sealing authority `confirmatory-holdout-v3-seal`;
- every v2 and v3 scan, publication, packet, retry, reseal, partition,
  arm-order, and bootstrap authority or domain, including every domain string
  under the retired prefixes `confirmatory-holdout-v2/` and
  `confirmatory-holdout-v3/`.

Mentioning or mechanically verifying a retired identifier never revives it.
Every v4 domain, source alias ID, sealing authority, and semantic reader is a
fresh identity declared in `analysis-plan.json`; none is inherited, aliased, or
renamed from a retired lineage.

The production repair design remains exactly the existing file with SHA-256
`76d34d33b7be57230ef0988662060fe70eea229f9b07f2cd5b8a2354600a79f5`.
Its production invariants R1--R15, defaults, state machine, storage and delivery
design, metric meanings and arithmetic operators, numeric point gates, and
implementation ownership are unchanged. Its evidence protocol is explicitly
superseded by the exact v4 declarations: the protocol and plan pins, one-shot
readiness attempt and evidence order, packet namespace, reader IDs, domains and
seeds, admissible replayable floor and measured cohorts and therefore their
denominators, production-shape seed validation, receipt/evaluator transport
literals, and the same-implementation reference arm. The v4 reference is the
pinned default-off replay control. This changes no production repair behavior.
The unchanged repair becomes eligible for implementation only when a
validator-valid v4 packet manifest has been introduced by a seal commit that is
a strict Git ancestor of every repair or semantic-evaluator commit. No v2 or v3
result can satisfy that precondition.

The release control is the exact `control-watermark.json` at SHA-256
`a9e6e6a4dbd6178345e11e951e973a1af9c062fb50f8dd7bd51528d44217a86f`,
10,553 bytes. The protocol name `release_effective_at` is an exact alias for
that artifact's `selection.lower_bound_exclusive_at`,
`2026-08-14T15:03:46.793603Z`. It denotes an observed runtime-provenance
boundary, not candidate deployment. The serving build contributes only the
event population. The gating-off replay control is exactly commit
`46a9951842512333b0896370056d07a9e1c25bdf`, tree
`c1606671b9d13bed78c21b7d2f9a4bb75a3d1c1c`; it never becomes a claim about
what was serving during accrual.

## 2. Fixed UTC calendar, grace, missed slots, and horizon

Calendar values were chosen without source counts or availability estimates.
They are not power estimates, they may not be adapted to observed accrual, and
no later observation of source availability, readiness, or outcome may move,
extend, shorten, or re-anchor them:

```text
anchor_at                  = 2026-08-17T00:00:00Z
cadence_seconds            = 86400
slot indices               = integers 0 through 28 inclusive
slot_at(i)                 = anchor_at + i * 86400 seconds
grace_seconds              = 21600
valid execution window     = [slot_at(i), slot_at(i) + 21600 seconds)
final_selection_slot_at    = 2026-09-14T00:00:00Z
absolute_horizon_expires_at= 2026-09-14T06:00:00Z
```

Every slot is a distinct invocation. No process may sleep, poll, or remain
alive waiting across slots, and an orchestration retry is not a new cadence
event. An invocation before `slot_at` rejects before source access. Within the
half-open grace window, a technical execution may repeat only when the launcher
proves it failed before key creation and before any source open. Once either
source is opened, no recapture or new keyed attempt is allowed. A mechanical
validator retry may use only the already latched exact snapshot/keyed-result
set and may not reopen a source. A crash, invalid output, or structural failure
after source open is terminal unless the sole valid output is a count-free
runtime-segment closure backed by a proven tuple or binding mismatch. Observer
unavailability is never evidence of change and cannot close a segment. No count
or status may escape a failed worker. This ordinary pre-source retry rule never
applies to the one-shot initial or successor runtime/source-binding ceremony;
once its attempt marker exists, any failure is terminal. A probe whose progress
cannot be proven after its attempt marker is conservatively terminalized by the
fixed watchdog, never treated as an unopened retry. The
first validator-valid slot resolution is immutable and forbids every
replacement or second resolution. The launch, pre-runtime observation, both
snapshot captures, post-runtime observation, validation, and any ready-to-seal
handoff must all complete inside the window. The selected upper bound is always
`slot_at`, never launch, observation, snapshot, validation, or wall-clock time.

If the window closes without a validator-valid receipt, an append-only
`missed` marker is recorded without source access. That state is terminal
`schedule-integrity-failure`: no later probe or seal is authorized, because the
unobserved population might already have passed and a later result could not
prove first-pass stopping. There is no late launch, backfill, catch-up,
interpolation, replacement snapshot, or discretionary skip.

A valid below-floor receipt at slots 0 through 27 is `continue`, not
`insufficient`, and grants no authority. A valid passing result is the first in
the uninterrupted active segment. It never becomes externally visible as a
discretionary latch. From ledger head H, the same supervisor performs the final
runtime check, creates the no-overwrite seal marker and marker-ledger entry M,
and spawns the sealer behind a closed inherited gate. The child cannot read,
build, or publish. The supervisor stages the exact ready resolution and its
deterministic ledger entry R privately, sends both to the child over inherited
IPC for validation, exposes them together with one no-replace atomic directory
rename, and only then opens the gate. Every step has one original 30-second
handoff deadline; failure or EOF is terminal. All later probes are forbidden
and the exact retained snapshots flow directly to that child.
If the final check instead proves a runtime change, the supervisor publishes
only a count-free closure for that slot; provisional counts never
escape, all prior segment counts are void, and a successor may start at the
next fixed slot after re-attestation. This outcome-blind closure occupies the
slot, so it is not a gap or a substitute snapshot. Once seal authority is
consumed, failure is terminal and cannot reset into accrual.

If slot 28 has a valid below-floor or closed-segment resolution, no later
source read is permitted.
At `2026-09-14T06:00:00Z` the ledger must append terminal
`insufficient-evidence` and escalate to the parent. A final missed slot is a
schedule-integrity failure rather than fabricated insufficiency. Segment
closure, re-attestation, delays, failures, deployments, and restarts never
shift a slot or extend the absolute horizon.

## 3. Complete runtime segments

The initial segment is the complete two-service runtime tuple and stable
pre/post digest in the pinned control watermark. That artifact deliberately
stores a canonically sorted, **unaliased** service array: its index order does
not assign `local` or `alt`. Before any source-row access, the runtime observer
must privately prove a bijection from the two frozen alias authorities to the
exact unordered watermark service set and to two distinct live database
instances. It publishes only the strict source-binding attestation declared in
the plan: alias IDs, privacy-safe database-identity hashes, the unordered
watermark-services digest, and pass metadata. No raw locator, identity, or
alias-to-array-index claim persists.

### 3.1 The two initial bootstrap branches

The initial segment is attested by the watermark alone and therefore has a
prior service tuple but **no** prior source binding. The plan models that
asymmetry directly instead of hiding it. The initial binding has one row-blind
ceremony in the slot-0 invocation after its durable attempt markers and before
HMAC creation, snapshot open, or source row access, and it resolves into
exactly one of two mutually exclusive branches. The branch selector is only the
equality of the observed complete stable active-services digest and the pinned
watermark digest; no operator, availability, count, or outcome may influence
it.

**Same-tuple initial binding.** When the complete stable observed tuple equals
the watermark tuple, the ceremony publishes the ordinary
`source-binding-attestation` and the same slot-0 invocation continues into its
ordinary probe resolution without a second attempt marker.

**Changed-tuple pre-binding closure.** When the complete stable observed tuple
differs from the watermark tuple, the sole valid output is the distinct
count-free `initial-runtime-change-closure`. That receipt is a separate typed
schema, not the ordinary closure with a hole in it. It binds the watermark
prior active-services tuple, the complete stable observed active-services
tuple, the recomputable observed source-binding core, the sole initial
source-binding attempt marker, the runtime observer, the analysis plan, the
immediate accrual-ledger predecessor, and the fixed slot-0 schedule instants.
Its `mismatch_kinds` is exactly the single-element array
`active-services-state-change`, because a source-binding change cannot be
proved against a binding that never existed.

The schema declares **no** prior-source-binding member at any nesting depth.
Absence is the honest representation: the value does not exist, so the receipt
does not carry a field for it. Under the recursive unknown-member rule any
receipt that does carry such a member is invalid, whatever it holds -- an
object, a JSON null, an empty string, a zero, a placeholder digest, or a copy
of the observed core. A validator may not synthesize, infer, backfill, or
default that value, and it may not accept the ordinary two-core closure here by
supplying one. Symmetrically, the ordinary two-core `slot-segment-closed`
resolution remains the strict schema for every closure after a valid binding
exists, and it is invalid before one exists. Neither schema is a permitted
spelling of the other, and neither may be reused in the other's slot state.

The pre-binding closure reads no source row, computes no key, captures no
snapshot, and carries no count. It consumes slot 0 as that slot's immutable
resolution, retains a typed pre-binding predecessor rather than a constructed
active segment, clears every latch and source authority, grants nothing, and
sets the next probe slot to 1. Its only continuation is the ordinary successor
ceremony below. Malformed, unstable, partial, or unavailable tuple or binding
evidence, worker failure, or an invalid receipt is terminal runtime-attestation
failure rather than a closure; total absence of the invocation resolves only as
the ordinary missed slot.

### 3.2 Tuple, binding, and successor ceremony

An active runtime tuple consists of the exact unaliased service set. Each
service contributes its privacy-safe service identity hash; boot identity hash
and observed boot instant; serving commit and tree; every implementation
component name, SHA-256, and byte size; sanitized effective-configuration
schema, encoding, SHA-256, and byte size; and both effective legacy repeat
controls. Both controls must be boolean `false`. A separate private binding
joins each fixed alias authority to one service and one database identity using
the exact domain, canonical input, and pre/post derivation in the plan. Alias
swaps, additions, removals, mirrors, duplicate database instances, or an
unstable binding are fatal. The replay-control commit/tree remains a separate,
population-independent role.

Every scheduled probe observes and canonicalizes the entire unaliased service
set and private alias/database binding before and after read-only snapshot
capture. Both must equal their open segment attestations. Reordering cannot
hide a mismatch. A change to any member, including deployment of the replay
candidate, closes the whole segment. During a slot it emits only the count-free
closure. There is no inter-slot polling or discretionary
closure: process-lifetime tuple and configured-database identities are
immutable, boot IDs are nonreusable, and the next fixed-slot precheck detects a
replacement before rows can count. Captured snapshots and provisional counts
are destroyed and no result from that segment may contribute later. An ordinary
post-binding closure stores the retained privacy-safe prior and observed
tuple/core objects; its `mismatch_kinds` is the sorted, unique set containing
exactly `active-services-state-change`, `source-binding-change`, or both as
mechanically proved. An unavailable or malformed observation is terminal, never
a closure.

A successor segment requires a new append-only attestation using the frozen v4
schema, identity derivation, and domains plus a predecessor link to the closure
receipt, which may be either the pre-binding initial closure or an ordinary
post-binding closure. Exactly one attempt is authorized; its ceremony must
start within the half-open 30 seconds after the closure and finish before the
next due slot. Record launcher pre-time; observe `local` then `alt` complete
tuples, bindings, and source clocks; take a fresh `boundary_launcher_at` after
both return; set `boundary_at` to the exact maximum of that fresh launcher time
and both source clocks; do not begin the post phase before `boundary_at +
250 ms`; observe `local` then `alt` again; and record launcher post-time.
Complete pre/post state and binding must match, all pre instants are at or
before the boundary, every post clock exceeds it, boot instants precede the
corresponding pre observation, and the envelope is under 300 seconds.
`lower_bound_exclusive_at` equals that exact `boundary_at`; no operator may
choose another instant. A late, missing, failed, or invalid attempt is terminal
runtime-attestation failure with no alternate boundary. The release watermark is
never overwritten. Counts start at zero and selection remains strictly after
fixed `release_effective_at`. There is no cross-segment union,
carry-forward, latch, snapshot, component, family, session, workflow, or scope
count, and nothing crosses the bootstrap break. Closed-segment receipts remain
historical and grant nothing.

## 4. Sources, snapshots, and complete outcome-blind selection

The complete source alias set is exactly `local` and `alt`, with fresh alias
IDs `confirmatory-local-v4-ro` and `confirmatory-alt-v4-ro`. Raw locators stay
operator-private and untracked. Each alias must resolve to its frozen distinct
authority and database instance. A valid slot creates one immutable SQLite
snapshot per alias and opens it only with `mode=ro`, `immutable=1`,
`cache=private`, `PRAGMA query_only=ON`, and memory-only temporary storage.
Equal snapshot digests, reused sources, alias swaps, or a missing alias are
fatal. Receipts expose only alias-bound SHA-256 and byte size, never paths.

For a valid slot in an active segment, the selected population is the
source-qualified union of the complete recall-event contents present in either
of the two exact alias snapshots and satisfying all three timestamp clauses:

```text
created_at > release_effective_at
created_at > active_segment.lower_bound_exclusive_at
created_at <= scheduled slot_at
```

The lower bounds are strict and the upper bound is inclusive. The initial
segment boundary equals `release_effective_at`; a successor boundary can only
narrow the active interval. There is no source, scope, replayability,
family-size, session, workflow, feedback, success, error, payload, latency,
access, result, content, candidate, baseline, or other outcome filter. Floors
never select, truncate, or stop inclusion. An accepted timestamp is exactly
`YYYY-MM-DDTHH:MM:SS[.fraction]Z`, with one to six fractional digits when
present, a valid Gregorian date, seconds 00--59, and no offset spelling.
Selection and ordering compare parsed integer UTC microseconds, never text;
receipts emit exactly six fractional digits. Canonical order is parsed time,
source-alias UTF-8 byte order, then event-id UTF-8 byte order.

Source alias, a nonempty UTF-8 event identifier, and a valid UTC `created_at`
are structural. A missing or invalid structural field, a duplicate
source-qualified event key, or a timestamp outside the representable UTC
contract is fatal rather than droppable. Missing non-selection metadata keeps
the event selected and partitioned but can make it ineligible for floors and
measurement. Every selected event is accounted for in the packet manifest;
selected-but-nonreplayable records remain outcome-free inventory records and
cannot silently disappear.

The strict fixed lower bound is later than the original manifest window and
the replacement manifest's strict upper bound
`created_at < 2026-08-13T20:16:51Z`. This proves event disjointness from both
consumed packets using their public manifests. Neither corpus nor verifier may
be opened to repeat that proof.

## 5. Identity, replayability, components, and partitions

Each aggregate-probe attempt that reaches keyed computation receives its own
fresh ephemeral 256-bit HMAC-SHA256 key; the one-shot sealer receives a
different fresh key. A key never crosses attempts or roles. Within one process
it covers the complete frozen-dev automatic reference and every selected
automatic candidate identity. Equality classes and counts are key-invariant;
an HMAC collision is terminal. The v4 identity message is the fresh domain
`confirmatory-holdout-v4/identity/v1`, a NUL byte, then exactly:

```python
(" ".join(query.split()) + "\n" + requested_scope).encode("utf-8")
```

The public de-identified `corpus/dev.jsonl` is not an identity bridge. The raw
operator-resolved frozen alt recall-event export remains pinned at SHA-256
`45dbfdb389324a8702ba1a84b28be028991f23dcfe1d7ab9d771928c37c2cc7a`,
4,162,914 bytes, with its original split rule. It is opened read-only only
inside the same isolated keyed process. If unavailable or mismatched, the slot
is a fatal integrity failure; no surrogate or alternate reference may replace
it.

Automatic means persisted `agent IS NULL`; organic means `agent IS NOT NULL`.
Every non-`None` value, including an empty string, zero, or false, is organic.
A repeated automatic family is source-qualified and must have at least three
floor-eligible replayable automatic events across at least two distinct
nonempty source-qualified transport sessions. `unseen_in_dev` means the family
token is absent from the complete frozen-dev automatic reference. Selected
nonreplayable events may share the equality class and component but never
establish family size, session spread, or any floor.

Replayability is a frozen input-only predicate, not an outcome filter. A
floor-eligible event has a nonempty query and every persisted effective input
listed in the plan. The packet reconstructs a canonical behaviorally equivalent
call: an explicit de-identified persisted requested scope, the persisted
resolved positive `max_results`, and the persisted normalized accepted `depth`,
independent of whether the historical wire call omitted a default or used an
equivalent surface spelling. The hash-pinned input-only resolver must recompute
the complete persisted de-identified scope plan and provenance exactly before
either replay arm. Requested scope must be exactly `global`, nonempty
`project:*`, or nonempty `session:*`. Unknown context keys make the call
unconditionally nonreplayable; no later implementation proof or case-specific
exception is allowed. Caller values are de-identified with their persisted
nullness, scalar type, and equality preserved. The evaluator injects
`ambient_context.agent` from the de-identified persisted agent exactly: every
non-`None` value, including empty string, zero, or false, remains non-`None` and
equal in the caller envelope. Persisted task/session provenance must likewise
agree wherever present. A mismatch is nonreplayable and is fatal at preseal if
the case was floor-counted.

Replay also requires a valid created-at and event label plus a hash-bound,
de-identified, outcome-independent projection of **every** allowed source-state
row in the alias snapshot. The projection is never selected from recorded
results and can represent every node, edge, retrieval weight, and resolver input
required by any selected call. The same production-seed validator is consumed
byte-identically by the probe, sealer, and evaluator seed builder. It requires a
nonempty retrieval-weights table and recognizes policy keys `default`,
`project`, `global`, and `session`; only nonempty `project:*` and `session:*`
learned keys map to their corresponding kind. Empty suffixes and every other
key reject; it never applies the node-scope grammar to policy keys. The packet
verifier also proves that raw normalized-identity equality holds if and only if
surrogate normalized-identity equality within each source, preventing de-id
from splitting or merging repeated families. Workflow counting
additionally requires a nonempty source-qualified transport session.
Feedback, success, error, result, access, payload, latency, and either replay
arm's behavior are forbidden replayability inputs. Every event, call, session,
scope, workflow, family witness, and component witness contributing to any
floor must be backed by at least one such replayable case. A selected event
that fails replayability remains selected and partitioned, contributes to no
floor or measured cohort, and is reported only through aggregate counts.
Missing family identity creates no identity edge or automatic contribution;
missing transport session creates no session/workflow edge or session/workflow
contribution. Any independently valid other edge remains, otherwise the event
is a selected singleton.

Connected components cover every selected event. Edges join all automatic
events in a source-qualified identity equality class, whether replayable or
qualifying, and all events sharing a nonempty source-qualified transport
session/workflow key. Missing both kinds of edge yields a singleton. The
representative is the lexicographically smallest bytes
`source_alias + NUL + event_id`. Partitioning uses the fresh v4 domain:

```text
digest = SHA256(b"confirmatory-holdout-v4/partition/v1\0" + representative)
bucket = int.from_bytes(digest[0:4], "big") % 100
bucket 0..49  -> holdout
bucket 50..99 -> shadow
```

Every selected event has exactly one partition; intersection is empty and
union is complete. No identity class, organic session, or workflow crosses the
split. Input order and every outcome field are irrelevant. Collisions fail
closed. Key searching, rebalancing, stratification, truncation, reassignment,
or favorable retry is forbidden.

## 6. Unchanged floors and first-pass stopping

Every readiness condition is conjunctive and is evaluated on the complete
fixed partition of the slot population. All numeric power floors are unchanged
from v2 and v3; replayability remains mandatory for every counted event or
call.

Holdout requires:

- at least 30 unseen-in-dev repeated-automatic families containing at least
  150 replayable automatic events across at least 30 components containing a
  qualifying unseen family;
- at least 200 replayable organic events spanning at least 30 distinct
  nonempty source-qualified organic sessions and at least 30 components
  containing a counted organic event;
- at least two distinct requested `project:` scopes represented by counted
  replayable holdout events.

Shadow requires:

- at least 30 genuine nonsynthetic production workflows;
- at least 100 unique replayable logical calls;
- at least 30 components containing a counted workflow; and
- at least two distinct requested `project:` scopes represented by counted
  replayable shadow calls.

A genuine workflow uses the single source-blind proxy: it originates in a
declared production snapshot, has a source-qualified nonempty transport
session, contains at least one selected replayable `agent IS NOT NULL` anchor,
and contains a replayable call. No content or undeclared fixture classifier is
allowed. One logical recall event is one call; two replay arms form a pair, not
two calls. Post-run evidence still requires at least 100 complete measured
pairs across at least 30 workflows, with at least one complete pair in each
counted workflow. No count pools across partitions or segments.

The unchanged 30-component independence minimum and design references remain
recorded in the plan. They are not claims of iid observations. The unchanged
10,000-replicate whole-component bootstrap reports dependence-sensitive
uncertainty after evaluation. Neither availability nor a confidence interval
can lower or rescue a floor or point gate.

A probe computes only the complete deterministic partition and aggregate floor
counts in an isolated process. It emits no corpus, candidate plan, equality
token, component map, per-source count, case, or outcome. An intermediate
valid below-floor receipt is historical nonterminal evidence. The first valid
all-floor pass is the mandatory seal trigger; no operator may ignore it,
inspect cases first, request a newer population, or choose between passing
slots.

## 7. One-shot packet sealing and ancestry

The v4 sealing authority is exactly `confirmatory-holdout-v4-seal`, with one
process launch and one publication attempt. There is no externally observable
`ready` latch waiting for an operator. In the same supervisor invocation that
computes a provisional first pass, and still within the slot grace, a final
complete runtime and alias/database-binding check must pass; the supervisor
then creates and fsyncs the no-overwrite consumption marker and its ledger
entry **before** spawning the sealer behind a closed inherited gate. It stages
the ready receipt plus deterministic ledger entry privately; the blocked child
validates both over inherited IPC; one atomic no-replace directory rename makes
both visible together; only then does the gate open. The original handoff
deadline is 30 seconds. A runtime mismatch instead creates only the count-free
slot closure, consumes no seal authority, and reveals no provisional count.
Creation of the consumption marker is irreversible: marker-ledger, spawn,
staging, IPC, visibility, gate, EOF, crash, collision, validation, key, or
timeout failure is terminal. An existing valid marker is consumed authority,
never permission to spawn a replacement child.
After a marker-ledger write or fsync error, the supervisor reads only the
unique expected entry: an exact valid intended M is authoritative and resumes
that same handoff, proven absence terminalizes from H, and invalid or partial
bytes cause integrity escalation without an alternate ledger branch.

The sealer consumes only the exact immutable snapshot set handed off from that
first passing slot. That set is the domain-separated canonical identity of the
exact `local` and `alt` snapshot hashes and byte sizes and must match the ready
receipt, seal marker, post-build observation, terminal receipt, and manifest.
It independently recomputes complete selection, input-only
replayability, components, partitions, and aggregate counts. Any mismatch is
terminal. Publication is no-overwrite, content first, canonical manifest last.
There is one fresh HMAC key, one independent de-identification salt, and one
hard process/destructor boundary. After building content but before publishing
the manifest, the sealer performs another complete runtime and binding
observation. It must equal the handoff and active attestation. A change or
unavailable observation after authority consumption is terminal seal failure,
not a return to accrual. A sealed state requires a present hash-valid canonical
manifest with `frozen: true`, a passing first-ready resolution already bound to
the consumption marker and process launch, a runtime segment proven unchanged
through the post-build observation, a passing keyed preseal result with zero
mismatches, and zero semantic reads for both fresh readers. Consumption is
recorded only in separate hash-bound receipts, never written back into the
immutable manifest.

The sealer has a 3,600-second watchdog and 30-second forced-termination grace.
Every sealed or failed outcome must become a validator-valid terminal receipt
and `seal-terminal` ledger entry; inability to make that evidence durable is
itself terminal integrity failure. Timeout or failure consumes the one-shot
authority and can never return to accrual or reseal.

The manifest-introduction commit must be a strict Git ancestor of every repair,
semantic evaluator, calculator, repair configuration, or threshold-registry
commit. Timestamps are insufficient. A diff from this protocol freeze through
the seal must exclude production repair and semantic-evaluator changes. The
serving accrual build is never substituted for the hash-pinned gating-off
replay control.

## 8. Fresh one-shot semantic readers

The complete future semantic-reader set is exactly:

- `confirmatory-shadow-v4-eval`, for the shadow partition; and
- `confirmatory-holdout-v4-eval`, for the holdout partition.

There are no aliases, delegates, wildcards, debug readers, fallback readers,
or transferable capabilities, and no retired reader identity may be revived,
renamed, or delegated into this set. Mechanical runtime observers, probes,
ledgers, builders, verifiers, sealers, hashers, and privacy checks are not
semantic readers and cannot emit case meaning. Each semantic reader has at most
one process launch and one forward semantic pass. Its durable O_EXCL
launch-attempt marker is both the one-shot authority consumption and the next
controller handoff; a separate consumption marker must fsync before process
spawn or source open. A valid existing attempt marker is recovered without
relaunch. A marker, consumption, validation, pre-open, spawn, partial-read,
crash, invalid, inconclusive, timeout, or forced-termination failure produces
the exact typed aggregate terminal status in the plan; an attempt-marker
failure uses the predecessor terminal artifact and a null marker rather than
fabricating one. Every evaluation process has a 3,600-second controller
watchdog followed by a 30-second forced-termination grace. Original watchdog
and handoff deadlines do not reset during controller recovery.

All synthetic, dev, public-eval, integrity, privacy, implementation,
configuration, threshold, calculator, and evaluator checks finish before
either reader. One exact implementation-stack-freeze receipt anchors the
clock. Development calibration and public evaluation each consume one exact
phase through a durable O_EXCL pre-reader launch-attempt marker before spawn;
their 30-second start deadlines and terminal receipts form a nonresetting
chain. Development-calibration failure is the sole pre-reader stop.
After development passes, public evaluation gets one terminal attempt; every
public outcome--pass, fail, crash, invalid, inconclusive, or timeout--requires
the shadow attempt within 30 seconds. One clean implementation tree and complete measurement stack
are frozen. Shadow writes its terminal receipt into an access-controlled
embargo channel. Only the isolated shadow evaluator and controller may create
or hash it; no external actor, repair process, selector, dashboard, or holdout
path may learn its aggregate before joint release. Every shadow terminal
outcome, including launch-integrity or launch-failure, requires the holdout
attempt on the identical frozen stack within 30 seconds, independent of both
public and shadow outcomes. A holdout attempt-marker failure itself becomes a
typed holdout launch-integrity terminal receipt rather than skipping joint
release. After holdout termination, the controller writes and fsyncs one
private same-filesystem envelope embedding both exact terminal receipt objects
and their hashes, then performs one atomic no-replace visibility operation
within the original 30-second deadline. Neither private receipt is separately
visible. A partial, duplicate, invalid, or late joint release is terminal
integrity failure. No reader may tune, select, repair, rerun, or influence the
other.

## 9. Measurement, isolation, and uncertainty

The gating-off control is the pinned release commit/tree; the candidate is the
future post-seal repair commit/tree. They necessarily have distinct hash-bound
arm implementations, while using the same evaluator, configuration,
environment, ordered calls, resolver, and seed bytes. The serving accrual build
is neither arm. Both arms execute the same canonical replayable calls from
separate byte-identical, hash-bound pre-open state clones for each partition
and source. Seed construction starts a fresh database and copies only the v4
plan's de-identified `nodes`, `connections`, and `retrieval_weights` column
allowlists. `recall_events`, automatic signal/history, delivery history,
sessions, caches, pending feedback, outcome tables, and temporary tables start
empty. Every unclassified mutable table or required extra field is fatal.
Production-seeded structural validation must recognize the retrieval-policy
keys `default`, `project`, `global`, and `session`; it must never apply a memory
node-scope predicate to policy keys or accept an empty synthetic table as
production proof.

The v4 arm-order domain predetermines one dispatch bit from the private
source-qualified event key before mappings are destroyed. The sealed case
stores only that bit. Both arm responses finish before either arm advances.
State, history, process, and caches never cross source aliases, arms,
partitions, or readers. `content_ref` checks occur only after primary calls in
fresh isolated read-only resolver clones and do not affect primary payload or
latency.

Full-sample point estimates are authoritative. The unchanged conjunctive gates
are: unseen repeated-automatic character reduction at least 0.50; automatic
content access at least 0.95; organic payload and access deltas each within
inclusive +/-0.05; zero differing agent-triggered result bytes; every eligible
novel node content-bearing; and shadow p50 and p95 latency degradation each at
most 0.10. Missing metrics, zero denominators, a zero novel-node population,
incomplete pairs, or instrumentation gaps fail closed. There is no trimming,
cohort replacement, metric substitution, or rounding before comparison.

The bootstrap remains 10,000 whole-component resamples with two-sided 95%
nearest-rank intervals and the SHA-256 counter sampler. V4 uses only its fresh
bootstrap domain and namespace-derived seeds in the plan; the retired v2 and v3
seed values are not reusable. Intervals are supplementary: they cannot replace,
average away, or rescue a failed full-sample point estimate.

The existing default-off release remains shippable. Promotion is authorized
only if every frozen confirmation gate passes conjunctively. Any negative,
invalid, incomplete, inconclusive, timed-out, or missing gate retains the
default-off release and requires escalation; the serving accrual build is never
a promotable substitute.

## 10. Privacy, receipts, and hash binding

Raw SQLite rows, transcripts, queries, content, paths, identifiers, identities,
tokens, equality maps, case records, family membership, per-family sizes, and
case-level outcomes must not enter prompts, version control, stdout, stderr,
diagnostics, probe receipts, closure receipts, or final aggregate reports. Raw
private I/O stays inside isolated processes and is destroyed after its
authorized boundary. No HMAC key, key commitment, token, sample, plaintext or
normalized identity, unkeyed fingerprint, salt, or bijective map persists.

The source binding, probe attempt/failure, probe resolution, segment
attestation, initial-runtime-change closure, ordinary segment closure,
missed-slot, horizon, ledger, seal consumption/terminal, implementation-stack
freeze, pre-reader launch, development/public terminal, reader launch/terminal,
and joint-release schemas are exact typed allowlists in `analysis-plan.json`;
extra fields, duplicate keys, free text, unknown enums, symlinks, nested
extras, mistyped scalars, or noncanonical encoding fail. Probe outputs are
limited to namespace/schema/status, fixed schedule and boundaries, segment and
predecessor hashes, alias-bound snapshot hashes and sizes, total selected and
replayable counts, the fixed aggregate floor counts, and tool/plan hashes.
There are no individual, family, scope-spelling, workflow, or per-source
counts. Closure records contain content-addressed privacy-safe runtime/core
objects and the mechanically complete mismatch-kind set, not cases.
Final reports are aggregate only.

A sealed corpus may contain only freshly de-identified outcome-free case and
inventory records plus random packet-local equality labels required for replay
and clustering. Labels are unrelated to HMAC output and stay inside the sealed
boundary. De-identification preserves only exact character length, whitespace
positions, and within-build equality. Keys and maps travel through anonymous
descriptors; descriptor closure, descendant process-group termination, and
observed keyed-process exit precede manifest construction.

Every segment, probe, packet, and evaluation artifact is hash- and byte-bound
to the exact inputs listed in the plan. Append-only receipts form a predecessor
hash chain. The packet manifest binds the release watermark, active segment,
slot, exact snapshots, protocol documents, frozen public manifests and split,
raw dev identity reference, unchanged repair design, retired v2 and v3
documents, probe/ledger/builder/verifier/de-identifier implementations,
complete corpus and seed states, and the preseal receipt. Evaluation receipts
additionally bind the packet, clean repair commit/tree, replay control,
evaluator, calculator, configuration, thresholds, dependency/environment lock,
runtime settings, arm schedule, state/resolver hashes, reader ID, and
partition. Mutable names, path-only references, timestamps without content
hashes, and serving-build substitution are not evidence bindings.

## 11. Cross-protocol replay safety

A retired v2 or v3 artifact can never validate as v4 evidence, and a v4
artifact can never validate as retired evidence. The separation is structural,
not conventional: `schema_version` is `4` and the namespace constant is
`confirmatory-holdout-v4`; every active domain separator carries the
`confirmatory-holdout-v4/` prefix, so every keyed, ledger, partition, segment,
slot, snapshot, ready-core, identity, arm-order, and bootstrap preimage differs
from its retired counterpart; the fresh alias IDs enter the source-binding-core
preimage, so no retired binding core can equal a v4 core; the initial segment
identity is recomputed under the v4 segment domain and differs from the retired
v3 value; and the ledger genesis binds this plan's own SHA-256. The public
hash-pinned inputs shared with the retired protocols -- the release watermark
and manifest, the frozen original and replacement manifests, the split
declaration, the de-identified development corpus, and the unchanged repair
design -- are read identically by each protocol. Sharing an immutable public
input is not sharing an authority, a domain, a receipt, or a packet.
