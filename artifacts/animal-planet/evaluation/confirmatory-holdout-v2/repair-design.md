# Automatic-versus-agent recall repair design

Status: **source-blind design freeze**
Design baseline commit: `c950e1d1288818aa776fd11392b1b2d6d01d64eb`
Design baseline tree: `2699969c568930a5112d4c55d997513b61037e82`
Scope: parent requirements P2, P6, and P7 only

This document freezes the repair before any repair implementation, semantic
evaluator, or confirmatory source read. It is a design and ownership contract,
not evidence that the repair passes. The selected design separates protocol
provenance from benchmark labels, preserves the legacy mixed fingerprint view,
adds an automatic-only signal and access-history lane, and replaces omitted
ranked repeats with compact, directly resolvable references.

Implementation is conditional on the ordering gate in the frozen
`confirmatory-holdout-v2` policy. If the one-shot readiness result is
`insufficient`, no leaf below may start under this namespace. If it is `ready`,
both partitions must be published and sealed at a commit that is a strict Git
ancestor of every repair or evaluator commit.

## 1. Evidence boundary and input-hash allowlist

The parent goal and P2/P6/P7 invariants are normative instructions. Every
repository input admitted to the design is listed below by exact SHA-256 at the
baseline commit. There are no path globs, mutable `HEAD` references, or
unhashed substitutes.

| Role | Repository path | SHA-256 |
|---|---|---|
| Frozen development input | `artifacts/animal-planet/corpus/dev.jsonl` | `605e4f3fdc91817788a40d0dbbabffb16b86ef98ad28020c20379ef249cc9a36` |
| Frozen protocol | `artifacts/animal-planet/evaluation/confirmatory-holdout-v2/POLICY.md` | `096263ddf554dc014c8cd971e6129d47300bd1c0b9710da90a5749ff186a1afb` |
| Frozen analysis plan | `artifacts/animal-planet/evaluation/confirmatory-holdout-v2/analysis-plan.json` | `5f5f050330b97c8a35b0e76bd31cbf1b9dd73f4d1a2baa0175817c8201a0c653` |
| Production static code | `src/living_memory/server.py` | `1144444bb4058c6644d4812dd31e7c1b6331957c60d2082e3500315baabf3055` |
| Production static code | `src/living_memory/storage.py` | `7b82093d41ba04c15250390a0501d279f6c6787f67280a3170b830d245264eaa` |
| Production static code | `src/living_memory/delivery.py` | `b3507a0c336fe04b3e25eaffba50c300d3994225b15d972fd138efa434b7a990` |
| Public model contract | `src/living_memory/models.py` | `682dad7894d7d2d9d89655325454ee45fcc195d5b8dc6d9f38a9f30e17c0867b` |
| Recall event call path | `src/living_memory/retrieval.py` | `6e3929b7f2a7d19ba2a1f4f954e1c4a16a941d641df493e1df1a4d2f10547282` |
| Re-fetch rendering | `src/living_memory/resources.py` | `8b3bcada5f3b97233b90afd5b90ac4979ee41cee6e8d25d867886bfa4070ed20` |
| Requested-scope identity | `src/living_memory/scope.py` | `513044c8e9f15343c40bb8da736d356d69cd5a67967e6dcd14690d8184a56c39` |
| Historical calculator, static only | `scripts/ap_baseline.py` | `18a162f7923af10705acf67b4655f76808ebd111dfa80f23bb10c59c26fa534a` |
| Public delivery contract | `docs/mcp-interface.md` | `d1734b3bcd85c23fb2ba06542d0c2de095bcfb1c30fea735c91b768c44ed28c3` |
| Generated/E2E properties | `tests/test_recall_gating.py` | `26cc14b8a53e6fb0f535fd0b14dbb186e59e46a9215040900616287328a58911` |
| Generated/storage properties | `tests/test_storage.py` | `5780b31a6820cae4b7cae51a57f675860c9483625ec9d280c411029697a86c13` |
| Pure delivery properties | `tests/test_delivery_shaping.py` | `6b510612efc10d8bbc584a4e70cf070d42a33ea99cfed9faac43bd0138217990` |
| Wire delivery properties | `tests/test_delivery_diet_e2e.py` | `d170219527e0e5ec7eb1f80ee67ef699135506a4fc48e466db045ff6123eb760` |
| MCP boundary properties | `tests/test_mcp_server.py` | `2fabc11113d43214c60ebf0f24603ad4fdadd1cebab7c853d78558ada3e76192` |
| Evaluator/property interfaces, static scan only | `tests/test_ap_baseline_unseen.py` | `da3ae484965f7138d08fbe35af4b03a1b0c89cb14fcce798922461ab0572f120` |
| Protocol invariants | `tests/test_confirmatory_evidence_protocol.py` | `d2fbe803c45771e40e4d0f2e99e62269735b854daf5ada166d9092bc5b7bca8d` |
| Seeded-property precedent | `tests/test_retrieval_correction_dominance.py` | `6fa1957935ffdd84e4c25ae95fb117ba6940621fcdb450362bc3baa903410ff9` |

Living Memory was used for navigation and prior architectural context, but an
unhashed memory, report summary, or metric is not an admitted design input.
Every factual claim below is reproducible from the allowlist. In particular,
no value incidentally present in historical memory is accepted as evaluation
evidence.

Explicitly excluded from this design and from all pre-freeze commands are:

- `artifacts/animal-planet/corpus/eval.jsonl`;
- every original, replacement, or confirmatory holdout corpus and case record;
- `artifacts/animal-planet/evaluation/replacement-auto-recall.json` and the
  recovered replacement `final-report.json` / `final-report.md`;
- original-holdout and consumed-shadow result artifacts;
- private transcripts, SQLite rows, source aggregates, paths, identities, and
  raw or de-identified case records outside the allowed dev corpus; and
- execution of `tests/test_ap_baseline_unseen.py`, and any real-eval expected
  value or consumed-packet fixture embedded there. Static ranges of this
  hash-pinned file were inspected only to identify synthetic evaluator
  interfaces and parity gaps; no embedded real-evidence value is admitted.

No prohibited repository artifact was opened to produce this document. The
frozen dev corpus is used only in aggregate. No query, content surrogate,
event identity, family identity, or case row is reproduced here.

## 2. Static diagnosis

At the baseline commit, the failure follows directly from the call graph:

1. `server.py:867-908` resolves and evaluates a repeat fingerprint whenever
   repeat gating is enabled. It never checks `ambient_context.agent`.
2. `storage.py:860-953` stamps every recall with the same `(normalized query,
   requested scope)` fingerprint and updates one `recall_fingerprints` row.
   Agent-triggered and automatic events therefore warm, reset, and populate
   one another's gate and history.
3. `storage.py:918-929` treats each missing transport ID as another session.
   Enough sessionless direct calls can therefore satisfy a purported
   cross-session gate.
4. The defaults are `min_unlinked=5`, `min_sessions=2`, and `probe_every=25`.
   `should_gate_fingerprint` makes offset zero a probe, so the first otherwise
   eligible event is full and default compaction begins only after several
   warmups. A three-event family cannot reliably exercise the mechanism.
5. `server.py:917-940` turns fingerprint-history matches into
   `session_duplicate` stubs and removes the trailing all-stub suffix. A fully
   repeated ranked result set can become an empty wire response even though
   the recall event retains its raw ranked IDs.
6. `delivery.py:226-280` renders each stub as a dieted node plus preview, so a
   retained stub remains relatively expensive. It also chooses the first
   byte-identical node as the twin bearer before repeat status is known; a
   repeated higher-ranked twin can deprive a novel lower-ranked twin of the
   bearer role.
7. The decision read and later event insert are separate operations protected
   only by one process-local `RLock`. Two server processes can share the WAL
   database, so the warmup/probe decision is not a serializable database
   operation.
8. `recall_events.results` is written before delivery shaping. It records
   ranked candidates, not proof that every candidate remained on the wire.
   Consequently legacy gated rows cannot safely establish delivered-access
   history.
9. The historical calculator mirrors the mixed, class-blind decision and
   records replay events with transport provenance only. A repaired evaluator
   would see all replayed events as automatic unless it carries the frozen
   caller's agent nullness through the production predicate.

The allowed dev split contains 826 events: 732 agent-null and 94 agent-present.
There are 353 exact automatic `(normalized query, requested scope)` families;
14 families, covering 352 events, meet the frozen minimum of at least three
events and at least two distinct nonempty transport sessions. Those aggregates
show that dev covers short and long repeats. It does not contain a qualifying
cross-class collision. Only 7 linked events across 2 qualifying families carry
feedback, without a controlled reset/collision boundary, so generated
properties—not dev anecdotes—are load-bearing for those behaviors.

## 3. Frozen invariants

The implementation and evaluator must satisfy all of the following. These are
conjunctive; a payload win cannot compensate for an access or provenance
failure.

**R1 — Production provenance only.** A request is agent-triggered when
`ambient_context` contains an `agent` value that is not `None`. Any non-`None`
value, including an empty string, zero, or false, is a bypass because normalized
persistence is SQL `agent IS NOT NULL`. Missing `agent` and explicit
`agent: null` are automatic.
No benchmark `class`, template, skill, scope, query text, result-node agent, or
new caller-controlled gate flag may participate.

**R2 — Agent bypass.** Agent-triggered requests skip every automatic-only
stats read, session-history read, decision, gated mark, and reference
transformation. Ordinary retrieval, event persistence, delivery diet, twin
dedup, and same-transport session dedup remain unchanged.

**R3 — Byte identity.** With time, IDs, initial state, call order, and ordinary
delivery settings paired, every agent-triggered raw MCP tool-result content
byte is identical with automatic repair enabled versus
`LM_RECALL_REPEAT_GATING=0`. Comparing parsed or key-sorted JSON is
insufficient.

**R4 — One rendered warmup.** An automatic fingerprint with no server-validated
ordinary response since its last automatic feedback link is never compacted.
One ordinary response with at least one ranked result, complete ordered
coverage, and a nonempty transport session establishes the warmup. Its later
exact repeat in a different observed nonempty transport session may compact;
runtime need not wait for a third event. The evaluation cohort still requires
at least three events and two nonempty sessions. A zero-result, sessionless,
failed-before-finalization, agent, unknown, or stale-epoch event never warms.

**R5 — Real cross-session evidence.** The current request must have a nonempty
transport session. Every node selected for reference must have appeared in a
server-validated automatic response for this exact fingerprint, after the last
automatic feedback link, in a different nonempty transport session. `NULL`,
empty, unfinalized, pre-reset, agent-only, and repeated same-session stamps
never create cross-session or per-node reference evidence.

**R6 — Feedback reset.** The first `feedback_applied: 0 -> 1` transition on an
automatic event advances the automatic feedback epoch and resets the unlinked
streak, warmup, finalized sessions, per-node reference history, and probe
phase. The next decision serialized after that reset is ordinary. Re-marking
the same event never credits twice. Feedback on agent or unknown-provenance
events changes only the legacy mixed view and never automatic state. A late
finalizer from the old epoch may certify its event row but cannot re-arm any
current-epoch state.

**R7 — Periodic probes.** The first core-eligible repeat is compact, not a
probe. Under a positive interval, no more than `probe_every - 1` reference
reservations occur between successfully finalized probes. A due probe is
latched before commit; a failed or unfinalized probe does not clear it, and a
late parallel probe cannot erase newer reservations. `probe_every=0` disables
probes; positive values below two are invalid. Agent, novel-only, and
ineligible same/missing-session events do not advance the probe phase.

**R8 — Complete first and probe responses.** First deliveries, post-feedback
warmups, and probes contain every ranked node inline through the existing
full/snippet diet. They may retain ordinary sparse fields and snippet limits,
but automatic/session/twin stubs and references cannot replace an individual
ranked node. Thus every baseline-defined first-event node is content-bearing.
The agent path alone remains exactly the ordinary gating-off shaping path.

**R9 — Every ranked result survives.** Automatic compaction never removes a
ranked entry. Response `count`, order, node IDs, rank score fields, and
re-fetchability match the unshaped ranked result list, with score values and
rounding preserved exactly as the ordinary renderer would expose them.

**R10 — Novel nodes remain content-bearing.** A result node absent from the
eligible automatic history for this fingerprint and feedback epoch is forced
through an inline full/snippet entry. It bypasses ordinary session and twin
stub classification for that ID, even when its bytes equal another novel or
repeated node. Agent history, pre-reset history, the current session, and
another fingerprint never make it an automatic-repeat reference. Every novel
opaque node—not merely one bearer per byte-equivalence class—retains inline
content.

**R11 — References preserve access.** A repeated node uses the compact schema
in section 6 and its `content_ref.node_id` resolves through normal
`memory_lookup` to the byte-complete stored node. Resolver failure is lost
access, never a retained result.

**R12 — Legacy compatibility.** Existing `recall_events.fingerprint`,
`recall_events.gated`, `recall_fingerprints`, their public storage methods,
mixed-class updates, and `RecallEvent.to_dict()` remain unchanged. The repair
is additive. No stored query, result, feedback, or legacy aggregate is deleted
or reinterpreted.

**R13 — No retrieval change.** The repair does not alter scope resolution,
ranking, correction dominance, access logging, feedback matching, or
`memory_lookup` semantics. It shapes delivery after ranking and records only
the provenance needed to make that shaping safe.

**R14 — Default behavior is the measured behavior.** Candidate measurements
run with every `LM_RECALL_*` and `LM_DELIVERY_*` override absent. The reference
arm differs only by `LM_RECALL_REPEAT_GATING=0`. Effective defaults are emitted
and hash-bound; inherited operator environment is not evidence.

**R15 — No evidence-directed rule.** There is no query/fingerprint allowlist,
answer table, project special case, benchmark-class branch, skill gate, or
packet-specific production path. Exact identity, persisted signal, caller
provenance, and successful access history are the only decision inputs.

## 4. Additive storage and migration design

Raise `SCHEMA_VERSION` from 5 to 6. Keep the v5 structures and online writes
intact, then add the following storage-only state.

### 4.1 Event columns

Add to `recall_events`:

```sql
repeat_eligibility INTEGER NOT NULL DEFAULT -1
    CHECK (repeat_eligibility IN (-1, 0, 1));
automatic_access_complete INTEGER NOT NULL DEFAULT 0
    CHECK (automatic_access_complete IN (0, 1));
automatic_delivery_mode TEXT NOT NULL DEFAULT 'unknown'
    CHECK (automatic_delivery_mode IN
           ('unknown', 'bypass', 'ordinary', 'reference', 'probe'));
automatic_feedback_epoch INTEGER NOT NULL DEFAULT 0
    CHECK (automatic_feedback_epoch >= 0);
automatic_probe_generation INTEGER NOT NULL DEFAULT 0
    CHECK (automatic_probe_generation >= 0);
automatic_reference_node_ids TEXT NOT NULL DEFAULT '[]'
    CHECK (json_valid(automatic_reference_node_ids) = 1
           AND json_type(automatic_reference_node_ids) = 'array');
automatic_inline_node_ids TEXT NOT NULL DEFAULT '[]'
    CHECK (json_valid(automatic_inline_node_ids) = 1
           AND json_type(automatic_inline_node_ids) = 'array');
automatic_finalized_projection TEXT NOT NULL DEFAULT '[]'
    CHECK (json_valid(automatic_finalized_projection) = 1
           AND json_type(automatic_finalized_projection) = 'array');
```

The values of `repeat_eligibility` are:

- `1`: persisted agent is `NULL`; automatic-only accounting is eligible;
- `0`: persisted agent is non-`NULL`; protocol provenance bypasses repair; and
- `-1`: transitional/imported/old-writer state not yet reconciled. Unknown
  fails closed: it cannot gate or enter automatic history.

Fresh writes derive this value inside storage from the same normalized
`agent` value written to `recall_events.agent`. Callers cannot supply it.
None of these columns is added to `RecallEvent` or `RecallEvent.to_dict()`, so
the public event and MCP response shapes do not change.

`automatic_delivery_mode`, epoch, generation, and the canonical ordered JSON
reference/forced-inline ID lists are an immutable decision snapshot written
with the event. `bypass` is the agent, omitted-policy/direct-call, or master-
valve-off path and invokes the pre-repair renderer.
`gated=1` means only a fresh automatic `reference` decision with a nonempty
reference list; ordinary, probe, bypass, and unknown decisions retain
`gated=0`. `get_recall_repeat_decision(event_id)`
loads this snapshot rather than re-reading mutable history after commit.

`automatic_access_complete` means that the MCP handler constructed and
validated a complete ordered result and committed the finalizer. It is an
access-offered boundary, not a network acknowledgement: later FastMCP
serialization, disconnect, or client consumption is unobservable. A failure
before finalization leaves zero. This same observable definition is used in
storage, tests, and the evaluator.

### 4.2 Automatic-only aggregate

Add the following tables. The legacy-named count/timestamp fields have the
same SQL declarations as `recall_fingerprints`, but they are a filtered
automatic-only signal over persisted events. The session count is the exact
count of access-complete nonempty sessions in the current feedback epoch.

```sql
CREATE TABLE automatic_recall_fingerprints (
    fingerprint TEXT PRIMARY KEY,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    delivery_count INTEGER NOT NULL DEFAULT 0,
    linked_count INTEGER NOT NULL DEFAULT 0,
    deliveries_since_link INTEGER NOT NULL DEFAULT 0,
    last_linked_at TEXT,
    transport_session_count INTEGER NOT NULL DEFAULT 0,
    last_transport_session_id TEXT,
    updated_at TEXT NOT NULL,
    full_access_since_link INTEGER NOT NULL DEFAULT 0
        CHECK (full_access_since_link IN (0, 1)),
    feedback_epoch INTEGER NOT NULL DEFAULT 0 CHECK (feedback_epoch >= 0),
    compact_reservations_since_probe INTEGER NOT NULL DEFAULT 0
        CHECK (compact_reservations_since_probe >= 0),
    probe_due INTEGER NOT NULL DEFAULT 0 CHECK (probe_due IN (0, 1)),
    probe_generation INTEGER NOT NULL DEFAULT 0 CHECK (probe_generation >= 0)
);

CREATE TABLE automatic_recall_fingerprint_sessions (
    fingerprint TEXT NOT NULL
        REFERENCES automatic_recall_fingerprints(fingerprint) ON DELETE CASCADE,
    feedback_epoch INTEGER NOT NULL CHECK (feedback_epoch >= 0),
    transport_session_id TEXT NOT NULL CHECK (length(transport_session_id) > 0),
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    PRIMARY KEY (fingerprint, feedback_epoch, transport_session_id)
);
```

Only the first successful finalization of an automatic event with at least one
result and a nonempty session inserts or updates the session table. Raw
attempts, zero-result events, and stale epochs do not. For every aggregate,
`transport_session_count` equals `COUNT(*)` for its current `feedback_epoch`,
and `last_transport_session_id` is the most recently finalized such session;
feedback resets both aggregate fields while leaving prior-epoch rows inert.

Add an index serving automatic access history:

```sql
CREATE INDEX idx_recall_events_auto_access_fingerprint_created
ON recall_events(
    fingerprint, automatic_feedback_epoch, transport_session_id, created_at DESC
)
WHERE repeat_eligibility = 1 AND automatic_access_complete = 1;
```

For a decision, `history_events=200` means the most recent 200
access-complete automatic events for the exact fingerprint and current
feedback epoch whose nonempty session differs from the current session,
ordered by `(created_at DESC, rowid DESC)`. It does not mean the last 200 raw
attempts followed by filtering. The union of their stored raw node IDs is the
only eligible reference history. Novelty uses the same 200-event rule without
the current-session exclusion; forgetting an older node only causes extra
inline content, never a false reference.

The internal APIs are separate and explicitly named:

- `get_automatic_recall_fingerprint_stats(fingerprint)`;
- `automatic_fingerprint_accessible_node_ids(fingerprint, feedback_epoch, *,
  exclude_transport_session_id=None, max_events=200)`;
- `get_recall_repeat_decision(event_id)`; and
- `finalize_recall_delivery(event_id, ordered_entry_projection)`.

`get_recall_repeat_decision` returns a private immutable value with exactly
`event_id`, `repeat_eligibility`, `mode`, `feedback_epoch`,
`probe_generation`, ordered `reference_node_ids`, and ordered
`forced_inline_node_ids`. It is defined in `storage.py`, not the public models
module. Every ID list is a duplicate-free subsequence of the event's raw ranked
IDs.

The history getter always ignores sessionless events. With no exclusion it
serves novelty detection across the current epoch; with the current nonempty
session supplied it serves only different-session reference evidence.

The existing mixed getters and `fingerprint_delivered_node_ids` keep their
current semantics. New server/evaluator code must not call them for automatic
eligibility.

### 4.3 Online writes and concurrency

`record_recall_event` gains an optional internal `AutomaticRepeatPolicy`.
Its return type remains `RecallEvent`. Only the MCP server supplies a policy;
an omitted policy keeps existing service/module callers ordinary and ungated,
and their unfinalized rows cannot warm access history. For the MCP path the
method performs, under one `BEGIN IMMEDIATE` transaction:

1. derive eligibility solely from the persisted agent nullness;
2. choose `bypass` without automatic reads for agent/unknown provenance or an
   omitted/disabled policy;
3. for enabled automatic provenance, read the prior automatic signal,
   current-epoch finalized sessions, all-session access history for novelty,
   and eligible different-session history for references;
4. persist as forced-inline every ranked ID absent from the current-epoch
   all-session history; compute core eligibility, then intersect the ranked IDs
   with different-session history in rank order;
5. if the intersection is empty, choose `ordinary` and do not reserve probe
   cadence; otherwise choose `reference` or latch/select `probe` as section 5
   specifies;
6. insert the event with the immutable mode, epoch, generation, and ordered
   reference/forced-inline snapshots, and update the legacy mixed aggregate
   exactly as v5;
7. update only the automatic filtered signal counts for eligibility `1`—never
   the finalized session/access evidence—and commit.

This removes the current server pre-read / later insert race across HTTP and
HTTPS processes. The current event cannot see itself as access history because
its `automatic_access_complete` is still zero. Two concurrent repeats may
conservatively receive extra inline content while an earlier response is
unfinalized, but neither may use uncommitted or incomplete access evidence.
Automatic signal `delivery_count` remains an event-attempt count for migration
compatibility; it cannot authorize references without separately finalized
warmup, session, and per-node evidence. Compact cadence counts decision-time
reservations, so failed references can only cause an earlier conservative
probe, never too many references.

Finalization is a second `BEGIN IMMEDIATE` transaction because shaping cannot
hold a SQLite write lock. Before calling it, server integration validates the
ordered entry projection `(node_id, delivery, content_ref_node_id,
content_ref_kind, inline_content_present)`: IDs exactly equal the stored raw
ranked IDs; each planned reference appears at the same rank with
`content_ref_node_id == node_id` and `content_ref_kind == automatic_repeat`;
the reference-ID subsequence exactly equals the persisted snapshot; every
forced-inline ID is `full` or `snippet` with inline content; and every
remaining entry is inline, a valid in-response twin, or a resolvable ordinary
session reference. Storage accepts this canonical projection directly,
repeats every derivable ID/tag/forced-inline check, and stores its canonical
JSON in `automatic_finalized_projection`.

In that transaction storage loads the immutable eligibility/mode/epochs and
performs `automatic_access_complete: 0 -> 1` with a conditional update.
Aggregate/session effects occur only when that row changes and the stored
feedback epoch still equals the aggregate epoch. Any valid nonempty automatic
response with a nonempty stored session may enter current-epoch session/node
history, but only an `ordinary` or `probe` finalization with nonempty raw
results and a nonempty stored session sets `full_access_since_link=1`.
Reference and sessionless events never create warmup. A matching probe clears
`probe_due`, resets reservations, and increments
`probe_generation` only if both its stored generation and the aggregate's
current due generation match; the first parallel finalizer wins. A stale
feedback/probe-generation finalizer may certify only its own event row. An
identical repeated finalization returns the recorded state without effects; a
byte-different projection errors. Agent/unknown/bypass finalization is rejected
as an integration bug. Later transport failure is outside this observable
boundary.

### 4.4 Feedback

On the existing atomic `feedback_applied=0` conditional update:

- always preserve the legacy mixed credit/reset;
- additionally credit and reset the automatic row only when
  `repeat_eligibility=1`;
- increment `feedback_epoch` and `probe_generation`, set automatic
  `deliveries_since_link=0`, `full_access_since_link=0`, current session count
  to zero, last session to `NULL`, reservations to zero, and `probe_due=0`;
- never change automatic state for eligibility `0` or `-1`.

The legacy behavior that a second mark may replace `feedback_trace_id` without
crediting the aggregate again remains unchanged. Epoch comparison makes
feedback-versus-finalization serializable: whichever `BEGIN IMMEDIATE`
transaction commits first defines the order, and an older decision can never
restore reset state.

### 4.5 v5-to-v6 migration and downgrade reconciliation

An initial metadata read may be diagnostic but cannot decide reconciliation,
because another v6 process may finish migration before this process acquires
the writer lock. The executable order is:

1. run the existing idempotent base-through-v5 prerequisites and the v5
   fingerprint stamping/rebuild logic when its own NULL guard requires it,
   without otherwise touching the mixed table or final metadata version;
2. start one explicit `BEGIN IMMEDIATE`; reject and roll back if the locked
   metadata version is greater than 6 rather than stamping a future schema
   down;
3. while holding that writer transaction, re-inspect `PRAGMA table_info` before
   each v6 column addition and execute each required `ALTER TABLE`,
   `CREATE TABLE`, and `CREATE INDEX` with individual `execute` calls—never
   `executescript`, whose implicit boundary would release serialization;
4. re-read the metadata version and `-1` existence under the same lock; if the
   locked current version is already 6 and no `-1` exists, commit the structural
   no-op/additions without any data rebuild;
5. otherwise classify every event from persisted SQL `agent IS NULL`, never
   ambient JSON or a benchmark label, and rebuild only the automatic filtered
   signal;
6. replay deliveries at `created_at` and links at
   `max(feedback_applied_at, created_at)`, ordered by
   `(action_time, delivery-before-link, rowid)`, assigning feedback epochs and
   retaining the v5 clamp/tie rules;
7. set every migrated event to access-incomplete with mode `unknown`, empty
   reference/inline IDs and finalized projection, and zero generation; start
   finalized sessions, warmup, and probe state cold—legacy `gated=0` proves
   event persistence, not completion
   of the later shaping/handler path, while legacy gated rows may have lost a
   suffix;
8. write schema version 6 as the final statement of that data transaction and
   commit.

The migration must not delete or rebuild `recall_fingerprints` unless the
pre-existing v5 NULL-fingerprint migration independently requires it. Tests
capture the projection onto every pre-v6 column plus all mixed rows/aggregates
before opening v6 and require canonical logical equality afterward. Reopening
v6 with no reconciliation trigger performs no automatic-lane data mutation;
tests compare canonical ordered `SELECT` output, not SQLite/WAL file bytes.

The `-1` default lets an older binary insert without pretending it maintained
new state. Current v5 also stamps metadata back to 5, so a later v6 open fully
reconciles even when the old writer only applied feedback to an already
classified row. Reconciliation deliberately resets access/session/probe state
cold because old-writer render order is unknowable. A crash before the data
commit leaves version below 6 and/or `-1` rows and reruns safely. Simultaneous
v5/v6 writers are unsupported; deployment must quiesce the old binary during
migration. Unknown rows never gate. No bidirectional write guarantee is
claimed beyond preserving all v5 projections and recovering conservatively on
the next v6 open.

A two-connection startup barrier synchronizes both stores immediately before
the first v6 column guard, makes both observe v5, then proves serialized DDL,
exactly one cold rebuild, and no erasure of a fresh post-v6 event, decision, or
finalization.

## 5. Automatic policy and state machine

The new server path uses an `AutomaticRepeatPolicy`; the old
`FingerprintGatePolicy`, `should_gate_fingerprint`, mixed table, and imports
remain available for compatibility but are no longer the MCP eligibility
source.

The fixed default policy is:

```text
enabled          = true
min_unlinked     = 1
max_link_rate    = 0.20
min_sessions     = 2
probe_every      = 25
history_events   = 200
```

The existing `LM_RECALL_REPEAT_GATING`, `MIN_UNLINKED`, `MAX_LINK_RATE`,
`MIN_SESSIONS`, and `PROBE_EVERY` names remain operational rollback/tuning
knobs. The semantic defaults above are frozen; dev calibration is pass/fail,
not a threshold search. `LM_RECALL_REPEAT_DROP_TRAILING_STUBS` is deprecated
for the new path and cannot remove automatic references. The single safe
rollback is `LM_RECALL_REPEAT_GATING=0`, which restores gating-off shaping
while automatic-only signal counts continue to accumulate. Disabled calls add
no access evidence; prior current-epoch finalized evidence remains available
after re-enable unless feedback reset it. Policy construction
rejects negative values and rejects `probe_every=1`; probes are disabled only
by zero and otherwise have an interval of at least two.

The decision, evaluated against state before the current event is inserted,
is equivalent to:

```text
agent_bypass = ambient.agent is not None
automatic    = not agent_bypass

if agent_bypass or policy is omitted or not policy.enabled:
    bypass: use the pre-repair renderer and do no automatic history read

all_session_history =
    raw node IDs from recent access-complete automatic events
    in this feedback epoch, in any nonempty session

novel_inline_ids = ranked node IDs absent from all_session_history

prospective_nonempty_sessions =
    prior current-epoch finalized-session count
    + 1 if current transport is nonempty and unseen in this epoch

cross_session_history =
    raw node IDs from the last history_events access-complete automatic events
    in this feedback epoch whose nonempty transport differs from current

low_signal =
    linked_count == 0
    or laplace_smoothed_link_rate <= max_link_rate

core_eligible =
    automatic
    and current transport is nonempty
    and cross_session_history is nonempty
    and prospective_nonempty_sessions >= min_sessions
    and deliveries_since_link >= min_unlinked
    and full_access_since_link == 1
    and low_signal

if not core_eligible:
    ordinary delivery; reference_ids = [];
    forced_inline_ids = novel_inline_ids
else:
    reference_ids = ranked node IDs intersecting cross_session_history,
                    retaining rank order
    if reference_ids is empty:
        ordinary delivery; forced_inline_ids = novel_inline_ids;
        do not mutate probe cadence
    elif probe_every > 0 and
         (probe_due or compact_reservations_since_probe >= probe_every - 1):
        latch probe_due = 1; persist current probe_generation;
        probe delivery with reference_ids = [] and every ranked ID forced inline
    else:
        reference delivery; persist ordered reference_ids;
        forced_inline_ids = every ranked ID not in reference_ids;
        increment compact_reservations_since_probe atomically
```

The zero-link clause is deliberate: after one successful unlinked delivery,
the smoothed rate is `1/3`, so applying the old `0.20` ceiling unconditionally
would silently reintroduce multiple warmups. Once a link exists, Laplace
smoothing remains the conservative long-run usefulness signal. A feedback
reset independently guarantees at least the next full-path response.

This rule may compact event two: a finalized nonempty automatic response in
session `s1` is sufficient for an exact repeat in `s2` to reference the
overlapping nodes. The three-event/two-session minimum belongs only to
generalization measurement: a qualifying short family then contains one
warmup and two repeat opportunities instead of measuring three warmups. The
persisted decision snapshot, rather than a post-commit history query, is the
sole shaping input.

## 6. Delivery shaping contract

Add an internal automatic-repeat reference renderer, but reuse the existing
wire value `delivery: "session_duplicate"`. This avoids expanding a public
string enum that strict clients may already validate. The compact variant is
used only for IDs in a persisted `reference` decision; evaluator and storage
identify it from that decision snapshot plus its ID-only node/reference shape,
not by treating every ordinary session duplicate as an automatic repeat.
Ordinary session duplicates keep their existing preview shape byte-for-byte.
The documentation must define a tagged union: `content_ref.kind` absent means
the unchanged legacy preview variant, while `kind: "automatic_repeat"`
requires `node.keys() == {"id"}` plus rank fields, `delivery`, and
`content_ref`. A strict compatibility fixture parses both variants and proves
ordinary session-duplicate bytes unchanged. This tagged compact variant is an
intentional additive automatic-path union member; the master valve restores
the old variant-only response, and agent responses never emit the new member.

With default sparse delivery, the exact entry shape is:

```json
{
  "node": {"id": "01..."},
  "score": 0.812345,
  "bm25_score": 1.0,
  "vector_score": 0.456789,
  "graph_score": 0.0,
  "trigger_score": 0.0,
  "scope_rank": 0,
  "methods": ["bm25", "vector"],
  "path": ["optional", "graph", "path"],
  "delivery": "session_duplicate",
  "content_ref": {
    "node_id": "01...",
    "kind": "automatic_repeat",
    "full_content_chars": 5120
  }
}
```

`path` is omitted only when the existing sparse rule would omit an empty path.
All overall/component scores, `scope_rank`, and methods retain the same
rounding and values as ordinary shaping. With sparse delivery disabled, the
compact entry **must** restore the per-entry event ID, full-precision score
fields, path including `[]`, and `content_ref.fetch` exactly as the existing
valve requires; `content_ref.kind` remains `automatic_repeat` and the node
remains exactly `{"id": ...}`. Sparse on/off properties pin both schemas. The
result list itself supplies rank/order—no new rank field is synthesized.

Reference entries contain no node content preview, context, provenance,
stats, timestamps, agent, task, level, scope, or decay metadata. Those fields
are available from the advertised re-fetch path. Repeating the ID under
`node.id` and `content_ref.node_id` is intentional: the first preserves the
ranked-result identity shape, and the second is the existing lookup contract.
The additive `content_ref.kind` is the union discriminator; ordinary
`session_duplicate` entries do not gain it and retain their preview bytes.

The shaping algorithm is:

1. Iterate ranked results in their original order.
2. If a node ID is in the decision's ordered reference snapshot, emit the
   compact `session_duplicate` reference. Do not register its bytes as a twin
   bearer and do not consume a snippet-ladder bearer position.
3. If a node ID is in the decision's forced-inline snapshot, emit its ordinary
   full/snippet entry while bypassing session/twin stub classification for that
   ID. Two byte-identical novel IDs each receive inline content.
4. Send every remaining ID through the unchanged ordinary renderer. A probe's
   forced-inline snapshot contains every ranked ID; a first or post-feedback
   response also contains every ID because its epoch history is empty.
5. Never pop, filter, coalesce, or reorder entries. Each repeated node keeps
   its own ID, scores, and `content_ref`; byte-identical repeated nodes do not
   collapse onto one another.
6. Verify response IDs equal stored raw result IDs in order before marking the
   event access-complete.

`memory_lookup(node_ids=[...])` already returns complete nodes in request
order. Unit and MCP tests call it without a scope restriction, batch every
reference ID, require the exact ID order and stored content bytes, and require
each `content_ref.node_id` to equal its entry's `node.id`. A valid global node
in a project recall is not a scope-mismatch failure. A mere
`content_ref` key without successful resolution never counts as access in the
confirmatory evaluator.

No existing `LM_DELIVERY_*` default changes. Every agent, direct-call, and
master-valve-off response calls the old shaping function with no automatic
reference or forced-inline IDs, which is the construction that makes R3
falsifiable. Enabled automatic ordinary/probe responses add only the explicit
inline guarantees above; score sparsity, node diet, and snippet sizes remain
at their existing defaults.

## 7. Generated/property contract

Create deterministic synthetic properties with seed `20260814`. All families
use synthetic queries absent from every replay corpus, contain at least three
events, span at least two declared transport sessions unless the property is a
negative boundary, and report non-vacuity counters. A second run with the same
seed must produce the same case digest and decisions.

### 7.1 Class-label flip invariance

For every generated event envelope, independently replace, flip, or remove
synthetic `class`, `template_id`, and similarly named metadata while holding
agent nullness and all production inputs fixed. Raw responses, stored
eligibility, automatic stats/history, decisions, and aggregate metrics must be
identical. Include malicious `class=automatic` with non-`None` agent and
`class=organic` with `agent=None`.

### 7.2 Repeated-agent byte identity

Drive at least three exact agent-triggered calls over at least two transports,
alone and interleaved with automatic calls of the same fingerprint. Run paired
fresh cloned stores with frozen clock, ULID sequence, decay, and call order.
Repeat with agent values `"named"`, `""`, `0`, and `false`; nullness, never
truthiness, controls the bypass.
Compare the actual MCP tool-result content bytes after every agent call under
defaults versus the master valve off. Require zero differences, no agent
`gated` rows, and no mutation of automatic-only counts, sessions, history,
feedback state, or probe phase.

### 7.3 Cross-class collision permutations

Exhaust all 24 orderings of three automatic events plus one agent event that
share exact query, requested scope, and ranked nodes, while automatic events
span at least two nonempty sessions. The automatic projection must equal a run
with the agent event removed. This projection is narrowly the automatic-only
eligibility, aggregate, epoch, finalized-session/history, probe, and decision
state over an injected fixed ranked-result stream; it does not assert that a
real agent recall cannot legitimately change node access statistics and later
ranking. The first automatic response is ordinary even when agent-only mixed
history precedes it; a later cross-session automatic repeat may reference;
every agent response is byte-identical to gating off.

Repeat the permutation family with feedback attached once to an automatic
event and once to the agent event. Only the former resets automatic state.
Also generate missing, empty, same-session, alternating-session, and
source-qualified collision boundaries. Agent sessions never satisfy the
automatic session floor.

### 7.4 Family boundary and warmup

Negative families are `2 automatic / 2 sessions`, `3 automatic / 1 session`,
and `3 automatic / only 1 nonempty session`. The measurement cohort rejects
all three. The exact positive boundary is `3 automatic / 2 distinct nonempty
sessions`. Runtime behavior for a positive `[s1, s2, s1]` family is ordinary,
reference, reference under the fixed no-link/no-probe-fast-path policy. Add
zero-result-first, failed-finalizer-first, missing-session-first, and
post-feedback `[s3, s3, s4]` variants: none may compact until a nonempty
server-finalized warmup is followed by a different nonempty session.

### 7.5 Ordered reference and novelty

Generate repeated–novel–repeated and novel–repeated–novel rank layouts,
including byte-identical content across a repeated and novel node and two
distinct byte-identical novel IDs in the first event. Repeat with a node seen
only in the current session inside a mixed reference response, and only under
another fingerprint. Assert under sparse delivery both on and off:

- response count and node-ID order equal ranking input;
- every score channel, `scope_rank`, methods, and nonempty path survive;
- every repeated entry has the exact compact schema and resolves;
- the documented strict duplicate parser accepts both unchanged legacy
  preview entries and the tagged compact ID/reference variant;
- every novel entry contains nonempty inline content through the ordinary
  path and is never classified from agent/other-fingerprint history; and
- both byte-identical novel IDs are independently inline, not twin stubs; and
- an all-repeat nonempty retrieval never becomes an empty response.

### 7.6 Feedback and probes

Generate `ordinary -> reference -> automatic feedback -> ordinary ->
reference`, with policy values chosen only to isolate the state transition.
Re-mark feedback and prove no double credit. Generate beyond two complete
probe periods; the first eligible repeat is compact, due probes are full-path,
failed/unfinalized probes remain due, and agent interleavings do not move probe
positions. Use barriers for feedback-versus-ordinary-finalization and two
same-generation late probe finalizers; stale finalizers may certify only their
event and must not restore warmup or erase newer reservations.

### 7.7 Migration and concurrency

Seed a v5 fixture with one colliding fingerprint, null/non-null agents,
nonempty/missing sessions, automatic and agent feedback, gated/ungated events,
equal timestamps, and overlapping result IDs. Assert:

- every pre-v6 row projection, fingerprint, gated bit, public event key, and
  mixed aggregate is logically identical after migration;
- automatic counts and exact sessions use only `agent IS NULL`;
- legacy events backfill automatic signal/feedback epochs but establish no
  access, finalized-session, warmup, or probe evidence;
- links replay in the frozen delivery-before-link tie order;
- repeated opens have identical canonical logical rows and do no data rebuild;
- query plans use the new partial history index.

Use two independent `MemoryStore` connections and a barrier to prove no lost
aggregate/session/probe updates and one serializable decision order. Include
feedback-versus-finalization, parallel due probes, a v5-feedback-only downgrade
round trip, and a direct v4-to-v6 fixture. Inject an exception before insert
commit and during finalization: the former rolls back event plus both
aggregates; the latter leaves access incomplete and cannot prime a later
compact response. A conflicting second finalization must fail, while an exact
repeat is a no-op.

## 8. Evaluator parity and default-env calibration

The historical `scripts/ap_baseline.py` and its consumed reports remain
unchanged. Build a versioned repair evaluator instead of changing the meaning
of historical evidence. It has separate `synthetic`, `dev`, `public-eval`,
`confirmatory-shadow`, and `confirmatory-holdout` modes. Synthetic self-checks
may run while building D. Dev runs once only after the D code/config commit and
E's stack/dev-input manifests are frozen. `public-eval` and both semantic
readers are later post-freeze falsification runs and cannot feed a code,
default, threshold, environment, or evaluator change.

The versioned evaluator must:

1. import the production provenance predicate, policy decision, storage APIs,
   and reference renderer rather than duplicate them;
2. carry the replay envelope's pre-candidate agent nullness into
   `ambient_context.agent`, while never exposing raw caller values;
3. derive automatic/agent cohorts from that nullness, not from `class`;
4. treat class/template fields as inert and pass the class-flip property;
5. require at least three automatic events and at least two distinct nonempty
   source-qualified transport sessions for a measured repeat family—`None`
   never counts—and emit non-vacuity family/event/session counts;
6. preserve canonical event order and enforce the per-event barrier before
   either arm advances; confirmatory modes consume the sealed one-bit arm-order
   field without recomputing destroyed identities, while dev/public-eval use
   the evaluator's hash-pinned even/odd canonical-index alternation;
7. measure decoded MCP content with the exact frozen
   `json.dumps(value, ensure_ascii=False)` rule;
8. count one event-node access opportunity per gating-off ranked node;
9. resolve each unique opaque candidate `content_ref` exactly once, only after
   all timed primary calls, through normal `memory_lookup` in the frozen fresh
   isolated read-only resolver clone with scope omitted, and count access only
   when the returned ID/order matches and `content_ref.node_id == node.id`; and
10. emit only aggregate counts, sums, ratios, mismatch counts, latency
    quantiles, hashes, and the frozen uncertainty fields. No identity, family,
    case, query, content, path, or per-source outcome may escape.

Both arms begin from byte-identical fresh clones whose seed manifest requires
empty recall events, automatic/mixed signal history, delivery/session history,
caches, and pending feedback. Timed calls use `monotonic_ns`, no source-case
warmups, and the frozen nearest-rank quantiles
`index=max(0, ceil(p*n)-1)`. Primary recall latency excludes the deferred
resolver exactly as preregistered. A separate aggregate diagnostic reports
unique resolver call count, total serialized resolver characters, and
nearest-rank p50/p95 resolver latency, plus the hypothetical recall-plus-fetch
character total; it is never folded into or substituted for a frozen point
gate.

Raw-byte comparison performs no normalization. Before either arm starts, the
evaluator injects the same evaluator-only, hash-bound UTC clock, ULID sequence,
and decay schedule through production dependency seams, then consumes them in
the same canonical call order. The schedule digest and exhaustion counts are
receipt fields and a paired synthetic fixture proves equal event IDs/timestamps
while intentionally different schedules produce byte mismatches.
`monotonic_ns` remains the independent real timing clock and is never frozen.

The public-eval P2 adapter preserves the historical populations instead of
silently changing denominators: payload median/p90 use the same
transcript-matched events with recorded serialized characters; top-result
retention uses the same nonempty matched events; useful-feedback retention uses
the same nonempty matched events with `feedback_applied`. Candidate access may
be inline, an in-response twin, prior-session inline access, or a reference
that passes the isolated resolver. Public eval is the only new run that can
measure P2 useful-feedback retention; the outcome-blind confirmatory packet
does not contain feedback outcomes.

For P6, character reduction and access use every qualifying repeated automatic
event in the selected cohort, organic/agent membership is frozen from
pre-candidate agent nullness, and the novel denominator is exactly the
analysis-plan definition: in canonical gating-off family order, a node is
novel on its first appearance and every first-event node is novel. Candidate
inline content must be nonempty for every such opportunity. Dev calls this an
all-qualifying training diagnostic. Public eval and confirmatory holdout use
their frozen unseen-from-complete-dev equality mechanism; no identity is
emitted or made available to production.

Uncertainty parity is exact and supplementary. For each automatic, organic,
and shadow metric group, run 10,000 whole-connected-component resamples in
opaque component-label UTF-8 byte order, preserve drawn-component
multiplicity and all paired observations, and recompute the corresponding
full-sample formula. Use SHA-256 counter sampler domain
`confirmatory-holdout-v2/bootstrap/v1`, the frozen uint32 seeds `908356138`,
`3546189773`, and `1236355769`, and the plan's big-endian seed/replicate/draw
encoding. Each replicate draws `component_count` indices as
`int.from_bytes(SHA256(domain || NUL || uint32be(seed) ||
uint32be(replicate) || uint32be(draw))[0:8], "big") % component_count`.
The two-sided nearest-rank interval uses sorted zero-based indices 249 and
9749. Any invalid replicate makes that interval null and reports its count; no
interval or bootstrap mean can rescue or replace a full-sample point gate.
Every uncertainty object contains exactly
`full_sample_point_estimate`, `ci_lower`, `ci_upper`, `component_count`,
`observation_count`, `seed`, `resamples`, and `invalid_replicate_count`.
A synthetic fixture pins sampler indices, multiplicity, all three seeds,
invalid-replicate behavior, interval indices, and output keys.

For the allowed dev replay, opaque cross-build unseen status is unavailable by
construction. Dev measures all qualifying repeat families as a training
diagnostic and must not call them unseen. Synthetic tests, not dev, establish
collision and feedback behavior.

### 8.1 Runtime input allowlists and stack manifest

Freeze four distinct canonical manifests: `dev-inputs.json`,
`eval-inputs.json`, `shadow-inputs.json`, and `holdout-inputs.json`. Each has
`schema`, `run_kind`, a `files` array of unique
`{role, repo_relative_path, sha256, bytes}`, and a `values` array of unique
`{role, canonical_value, sha256}`. Paths must be regular, non-symlinked,
inside-root files; role/path duplicates, missing/extra inputs, hash/size
mismatch, or undeclared evaluator-mediated semantic input opens fail before
semantic parsing. Interpreter, stdlib, shared-library, and OS runtime opens are
outside that file allowlist and are instead bound by the executable/version,
dependency lock, container/runtime digest, cache policy, and positive
environment manifest; the evaluator cannot treat them as corpus inputs.

A shared artifact-only `repair-stack-manifest.json` is created immediately
after the clean A–D code/config commit. It binds that strict parent Git commit
and tree and recursively enumerates every production, evaluator, resolver,
test, documentation, dependency-lock, and runtime file in that parent; it does
not attempt to include or hash its own later commit. A separate
`effective-config.json`, already inside the bound A–D parent, records the exact
automatic policy, delivery defaults, metric thresholds, environment allowlist,
and tool versions. Every run manifest hashes the stack manifest itself. These
two levels—not later report commits—define the clean stack without a
self-referential commit/tree hash.

Each run manifest admits this design, stack/config manifests, exactly one
corpus or sealed partition, its seed-state/construction manifest, the analysis
plan/bootstrap definition, and its evaluator entry point. Dev admits only the
section-1 dev hash. Eval admits only the frozen public eval. Shadow and holdout
each admit exactly their own sealed partition and reserved reader. The receipt
records the generated seed, reference-arm, candidate-arm, resolver-clone, and
process/cache initialization hashes and byte sizes before calls. Opening any
unlisted evaluator-mediated semantic input path is fatal, not a warning or
skipped case.

### 8.2 Default environment

Run each arm in a fresh process from a positive environment allowlist. Fix and
record locale, timezone, Python hash seed, executable/version, dependency lock,
and cache configuration. Do not inherit any variable beginning `LM_RECALL_` or
`LM_DELIVERY_`.

- Candidate: no repeat or delivery override is set.
- Reference: identical environment plus only
  `LM_RECALL_REPEAT_GATING=0`.

The receipt records the effective automatic policy, delivery ladder/diet,
history horizon, and a hash of the canonical environment. A test injects
hostile parent `LM_*` values and proves they do not reach either child except
the one reference-arm valve.

### 8.3 Dev pass/fail calibration

Calibration is validation of the fixed defaults, not a parameter sweep. Over
the complete frozen dev split and generated families, require:

- qualifying repeated-automatic character reduction at least `0.50`;
- automatic content access at least `0.95`;
- automatic first/probe/novel content-bearing fraction `1.00`;
- agent-triggered raw-byte mismatch count `0`;
- agent/organic payload and access deltas each within `[-0.05, +0.05]`;
- P2 overall median and p90 serialized-character ratios at most `0.60`, with
  top-result and useful-feedback access retention at least `0.95`; and
- default-env dev p50 and p95 primary-recall latency degradation each at most
  `0.10`; the binding P7 latency gate remains the later real shadow `<=0.10`
  requirement.

A failed dev gate stops the stack. It does not authorize threshold selection,
eval access, a class branch, or a query exception; a changed design requires a
new source-blind version and another review.

### 8.4 Post-freeze public eval and evidence release

After D freezes code/config and E creates the parent-binding stack manifest,
run dev without changing either. After dev passes, freeze only the
eval/shadow/holdout input manifests under their already-fixed schema. Execute
public eval once before either reserved semantic reader. It emits the frozen P2
and P6 eval aggregates and cannot authorize a repair. Once post-freeze
evaluation starts, both confirmatory readers still launch in their
preregistered order regardless of public-eval or shadow outcome. The final
report combines the public-eval aggregate, both v2 receipts, and hash references
to the two preserved consumed aggregate measurements without reopening any
sealed corpus. Point-gate failures remain versioned negative evidence.

`confirmatory-holdout-v2` freezes only P6/P7 cohorts and omits feedback/outcome
inputs; it must not be extended with P2 fields. Therefore current-code P2 is
bound only on frozen public eval. The preserved consumed P2 aggregates remain
historical context tied to their recorded implementations, not a fresh
holdout. Fresh P2 holdout retention and the P2 eval-to-holdout reduction gap
remain an explicit evidence gap unless the parent separately preregisters and
seals an outcome-safe P2 workload before implementation. This design neither
fabricates that evidence nor changes the immutable v2 policy.

## 9. Disjoint future implementation leaves

These are the only repair leaves authorized by this design. Their ownership is
non-overlapping. All implementation leaves have the sealed-packet commit as a
precondition; that external precondition is omitted from sibling dependency
lists for readability.

### Leaf A — `automatic-recall-storage-v6`

- Phase: build; expected mode: execute.
- Owns: `src/living_memory/storage.py`, `tests/test_storage.py`.
- Reads: `src/living_memory/models.py`, this design, frozen protocol.
- Depends on: none.
- Delivers: v6 additive columns/tables/index, migration/backfill,
  `AutomaticRepeatPolicy` and immutable decision type, automatic-only
  signal/state/history APIs, atomic decision insert, epoch/generation-safe
  finalization, feedback isolation, and compatibility/concurrency properties.
- Must not change `RecallEvent`, public MCP output, delivery rendering, or any
  evaluator/artifact.
- Focused acceptance:
  `timeout 300 bash scripts/test.sh tests/test_storage.py -q`.

### Leaf B — `automatic-reference-delivery`

- Phase: build; expected mode: execute.
- Owns: `src/living_memory/delivery.py`, `tests/test_delivery_shaping.py`.
- Reads: this design and `src/living_memory/resources.py`.
- Depends on: none.
- Delivers: the compact automatic variant of existing `session_duplicate`,
  exact schema, explicit forced-inline IDs, separation of references from
  novel bearers, ordering/count/score preservation, and pure deterministic
  properties. With neither reference nor forced-inline IDs, old shaping output
  must be byte-identical.
- Must not read storage, classify provenance, or alter existing delivery/env
  defaults.
- Focused acceptance:
  `timeout 300 bash scripts/test.sh tests/test_delivery_shaping.py -q`.

### Leaf C — `automatic-recall-server-integration`

- Phase: integrate; expected mode: execute.
- Owns: `src/living_memory/server.py`, `src/living_memory/retrieval.py`,
  `docs/mcp-interface.md`, `tests/test_recall_gating.py`,
  `tests/test_mcp_server.py`, `tests/test_delivery_diet_e2e.py`, and new
  `tests/test_automatic_recall_properties.py`.
- Reads: Leaf A and B interfaces plus `src/living_memory/scope.py`.
- Depends on: `automatic-recall-storage-v6`,
  `automatic-reference-delivery`.
- Dependency reason: it consumes Leaf A's `AutomaticRepeatPolicy`, immutable
  decision/finalization APIs, and Leaf B's reference/forced-inline renderer;
  no shared file forces artificial ordering between A and B.
- Delivers: protocol predicate at the MCP boundary, no automatic reads on the
  agent path, atomic decision consumption, history snapshot, response
  validation/finalization, no suffix removal, public documentation, real MCP
  byte-identity and generated family properties.
- Focused acceptance:
  `timeout 300 bash scripts/test.sh tests/test_recall_gating.py tests/test_mcp_server.py tests/test_delivery_diet_e2e.py tests/test_automatic_recall_properties.py -q`.

### Leaf D — `automatic-recall-evaluator-parity`

- Phase: build; expected mode: execute.
- Owns: new `scripts/ap_recall_repair.py`, new
  `tests/test_ap_recall_repair.py`,
  and
  `artifacts/animal-planet/evaluation/automatic-organic-repair-v1/effective-config.json`.
- Reads: frozen packet interfaces, this design, production interfaces from
  Leaves A–C, and frozen dev. It does not modify or semantically run the
  historical evaluator/test containing consumed evidence.
- Depends on: `automatic-recall-server-integration`.
- Dependency reason: the evaluator imports the final provenance, decision,
  renderer, and resolver interfaces instead of forecasting them.
- Delivers: synthetic-tested dev/public-eval/confirmatory modes, run-manifest
  validation, class-flip invariance, actual provenance replay, nonempty-session
  family qualification, paired bytes/characters, isolated resolver access,
  aggregate-only privacy output, the fixed config registry, and a clean A–D
  code/config commit that Leaf E's stack manifest can bind as a strict parent.
- Focused acceptance:
  `timeout 300 bash scripts/test.sh tests/test_ap_recall_repair.py -q` followed
  by the script's synthetic/bootstrap self-check and mechanical validation of
  the effective config. No eval or sealed partition is opened.

### Leaf E — `automatic-recall-dev-calibration`

- Phase: integrate; expected mode: execute.
- Owns only
  `artifacts/animal-planet/evaluation/automatic-organic-repair-v1/repair-stack-manifest.json`,
  `artifacts/animal-planet/evaluation/automatic-organic-repair-v1/dev-inputs.json`
  and
  `artifacts/animal-planet/evaluation/automatic-organic-repair-v1/dev-calibration.json`.
- Reads: frozen dev plus the clean implementation/evaluator stack.
- Depends on: `automatic-recall-evaluator-parity`.
- Dependency reason: the report hash-binds and executes the finished
  versioned evaluator and candidate interfaces.
- Delivers: an artifact-only manifest binding the strict parent A–D commit/tree,
  then one default-env aggregate dev run and generated-property receipt against
  the fixed thresholds in section 8.3. It may not change code, configuration,
  thresholds, or the evaluator.
- Acceptance: versioned evaluator validation of all three tracked JSON artifacts,
  exact input hashes, zero forbidden fields, and exactly one recorded dev
  launch. A valid negative receipt completes the leaf but blocks Leaf F; it
  never triggers an automatic retry or changed defaults.

### Leaf F — `automatic-recall-post-freeze-eval`

- Phase: integrate; expected mode: execute.
- Owns only versioned run manifests and aggregate outputs:
  `artifacts/animal-planet/evaluation/automatic-organic-repair-v1/eval-inputs.json`,
  `artifacts/animal-planet/evaluation/automatic-organic-repair-v1/public-eval-aggregate.json`,
  `artifacts/animal-planet/evaluation/automatic-organic-repair-v1/shadow-inputs.json`,
  `artifacts/animal-planet/evaluation/automatic-organic-repair-v1/shadow-aggregate.json`,
  `artifacts/animal-planet/evaluation/automatic-organic-repair-v1/holdout-inputs.json`,
  `artifacts/animal-planet/evaluation/automatic-organic-repair-v1/holdout-aggregate.json`,
  `artifacts/animal-planet/evaluation/automatic-organic-repair-v1/final-evidence-manifest.json`,
  `artifacts/animal-planet/evaluation/automatic-organic-repair-v1/final-report.json`,
  and
  `artifacts/animal-planet/evaluation/automatic-organic-repair-v1/final-report.md`.
- Reads: frozen public eval for its first run, then exactly one sealed partition
  per reserved reader; every run uses only its own manifest and the common
  clean hash-bound stack.
- Depends on: `automatic-recall-dev-calibration` and the external clean-stack
  freeze commit.
- Dependency reason: semantic readers may launch only after code, defaults,
  evaluator, allowlist, resolver, and dev calibration are immutable.
- Delivers: one frozen public-eval run, one embargoed
  `confirmatory-shadow-v2-eval` launch, then one
  `confirmatory-holdout-v2-eval` launch on the identical stack, joint release,
  and a report binding both preserved consumed aggregates by hash without
  reopening their sealed sources. It never overwrites historical reports.
- Acceptance: mechanical schema/hash/privacy validation, exact reader IDs and
  launch counts, component-bootstrap fields when a reader completed, and
  preservation of every pass, fail, crash, or insufficient receipt. A valid
  negative result completes this one-shot evidence leaf; parent-level review
  evaluates P2/P6/P7 gates and runs the repository gate. A semantic launch is
  consumed on partial read, crash, invalid output, or failure and is never
  retried.

## 10. Freeze and evaluation order

The complete sequence is fixed:

1. Commit this source-blind design and the synthetic readiness scanner.
2. Run the one aggregate-only readiness attempt.
3. If insufficient, publish only the allowed insufficient receipt and stop.
4. If ready, publish both complete deterministic partitions and seal the
   manifest last.
5. Verify the seal commit is a strict ancestor of Leaves A–F.
6. Build A and B in parallel; integrate C; build D, then freeze the clean A–D
   code/config commit.
7. E creates the artifact-only stack manifest binding that strict parent, then
   runs once under `dev-inputs.json`. If any dev gate fails, preserve the
   negative receipt and stop without opening eval or either reader.
8. If dev passes, freeze `eval-inputs.json`, `shadow-inputs.json`, and
   `holdout-inputs.json`. No stack file, value, or parameter changes afterward.
9. Run frozen public eval once. Its outcome cannot alter the stack or whether
   the two already-authorized readers launch.
10. Launch the shadow reader once and embargo its aggregate result.
11. Launch the holdout reader once regardless of public-eval or shadow outcome.
12. Release both reader receipts together, create the versioned final evidence
    manifest/report, then let the parent evaluate gates and repository health.

No eval, holdout, shadow, or private workflow observation may feed a repair,
threshold, environment, or evaluator change. If any post-freeze gate fails,
the result remains a versioned failure artifact.

## 11. Rejected alternatives

**Benchmark `class` gating.** Rejected because production does not own that
label and a class flip could change behavior without changing the request.
Agent nullness is already persisted protocol provenance.

**Keep the mixed aggregate and add an agent `if` only in the server.** Rejected
because earlier agent events would still prime automatic node history and
agent feedback would still reset automatic signal. It also leaves migration
and evaluator parity wrong.

**Include class in the fingerprint.** Rejected because it breaks stable stored
identity, makes a benchmark label production-visible, and does not protect an
agent caller that lacks the benchmark field.

**Drop all repeated results or keep preview-heavy stubs.** Rejected because
omission destroys order, score, ID, and direct access, while current stubs are
too large to make short families economical. Compact references preserve the
choice to fetch.

**Count raw `recall_events.results` as delivered history.** Rejected because
legacy shaping could remove those results after persistence. Only finalized
new responses prove access; migration backfills signal but starts access cold.

**Use more warmups or the current offset-zero probe.** Rejected because the
minimum qualifying family has only three events. One warmup plus later
cross-session references is the generic exact-repeat signal required by the
parent; probes begin only after compaction has actually occurred.

**Disable ordinary agent session dedup.** Rejected because P6 asks for
byte identity with gating off, not a new agent delivery regime. Existing
same-transport diet behavior remains on both arms.

**Tune after eval/holdout.** Rejected by the evidence contract. Fixed defaults
are validated on generated properties and frozen dev; later measurements are
one-shot falsification only.

## 12. Completion map

- **P2:** agent/bypass delivery diet and rollback behavior are unchanged;
  automatic inline guarantees and references are measured together rather
  than assumed to save bytes. Dev and frozen public eval use the exact
  historical transcript-matched payload, top-result, and useful-feedback
  denominators. Final review hash-binds preserved holdout aggregates only as
  historical context and explicitly reports that current-code P2 holdout and
  eval-to-holdout generalization remain unproven absent a separate presealed
  parent protocol.
- **P6:** R1–R15, the v6 lane, compact schema, property matrix, and evaluator
  parity make provenance protection, unseen-family economics, access,
  agent-byte identity, feedback reset, probes, and novelty independently
  falsifiable.
- **P7:** the input allowlists, strict packet-before-code order, disjoint file
  ownership, clean-stack freeze, one-shot readers, aggregate-only reports,
  source/implementation hashes, resolver isolation, default environment, and
  latency gates preserve the final evidence contract.

The design is implementation-ready only as this whole contract. Relaxing
agent byte identity, admitting a class label, using mixed history, dropping a
ranked entry, treating an unresolved reference as access, changing fixed
defaults after dev, or opening future evidence before the clean freeze is a
new design—not an implementation detail.
