# Confirmatory holdout v2 preregistration and access policy

This policy governs only the `confirmatory-holdout-v2` namespace. It is a
source-blind preregistration, not a readiness result, packet, or evaluation.
At this revision no candidate source or availability aggregate was queried or
used, no private row was inspected, neither consumed sealed corpus was opened,
and no repair implementation was begun. The only repository evidence files
inspected were the four frozen public metadata or policy documents hash-pinned
in `analysis-plan.json`; the governing goal and Living Memory decision context
are instructions, not candidate evidence.

`POLICY.md`, `README.md`, and `analysis-plan.json` become immutable together at
the protocol commit. Their byte hashes are locked by
`tests/test_confirmatory_evidence_protocol.py`. A correction, changed floor,
new source, changed split, or failed ceremony requires a new parent-authorized
namespace; v2 MUST NOT be amended, retried, regenerated, or resealed.

## 1. Retired and reserved readers

The original packet's `shadow-eval` authority (exercised by the historical
`holdout-shadow-eval` goal node) and `replacement-holdout-eval` are consumed and
permanently retired. They receive no access through this policy.

The complete set of future semantic-reader identifiers is exactly:

- `confirmatory-shadow-v2-eval`, for the shadow partition only; and
- `confirmatory-holdout-v2-eval`, for the holdout partition only.

There are no aliases, delegates, wildcard readers, fallback readers, debugging
readers, or transferable capabilities. A readiness scanner, publisher,
builder, verifier, launcher, and mechanical integrity checker is not a semantic
reader. Each reserved reader has at most one process launch and one forward
semantic pass. Launch consumes its authority even if it reads only part of its
partition, crashes, produces invalid output, or is inconclusive. Consumption
is recorded only in separate aggregate evidence; a sealed manifest remains an
immutable pre-evaluation attestation.

All synthetic, development, public-eval, integrity, privacy, implementation,
configuration, threshold, calculator, and evaluator checks MUST finish before
either semantic reader launches. The shadow reader launches first and deposits
an aggregate receipt into an embargoed channel. No person or repair process may
see it before the holdout reader terminates. The holdout reader then launches
against the identical frozen implementation and measurement stack regardless
of shadow outcome. Both aggregate receipts are released together. Neither
reader may tune, select, repair, rerun, or influence the other.

## 2. Sources, capture, and complete selection

The only candidate source aliases are `local` and `alt`. Their locators remain
operator-private and untracked. `local` and `alt` MUST resolve to distinct
declared authorities and distinct database instances; reusing, mirroring, or
double-counting one source under both aliases is a fatal integrity error, as is
an equal snapshot digest. Each alias resolves once to a source used read-only to
create an immutable SQLite snapshot. The snapshots MUST be opened with
`mode=ro`, `immutable=1`, `cache=private`, `PRAGMA query_only=ON`, and
memory-only temporary storage. The receipt exposes aliases, SHA-256 digests,
and byte sizes, never raw paths.

At readiness-process launch, before either source is opened, the trusted
launcher records its current UTC instant as the capture watermark. The
watermark MUST NOT be future-dated, and capture of both snapshots MUST begin
within 60 seconds without sleeping or waiting for accrual; otherwise the sole
attempt fails. The complete eligible population is the source-qualified union
of every recall event present in either of those two exact snapshots satisfying:

```text
created_at > 2026-08-13T20:16:51Z
created_at <= capture_watermark
```

The lower boundary is strict. There is no source, scope, family-size, session,
workflow, completeness, success, payload, latency, access, feedback, result,
or other outcome filter. Automatic means `agent IS NULL`; organic means
`agent IS NOT NULL`. Floors determine readiness but never inclusion. A missing
or invalid source alias, nonempty event identifier, or UTC `created_at` is a
fatal structural integrity failure, never a drop. Missing non-selection
metadata keeps a structurally valid event selected but prevents it from
helping the affected floor: missing query/scope creates no family edge, and a
missing session creates no session/workflow edge. Duplicate source-qualified
event keys are likewise fatal rather than a reason to deduplicate. The
canonical order is `created_at ASC`, then source alias byte order, then event
identifier byte order.

The strict lower bound is later than the consumed replacement packet's strict
upper bound and later than the original packet window. That manifest-bound
temporal proof establishes disjointness without opening either sealed corpus.
Their verifiers MUST NOT be invoked during preregistration or readiness because
they can traverse consumed data.

## 3. Identity and deterministic partitions

One fresh ephemeral 256-bit HMAC-SHA256 key covers the complete frozen-dev
automatic reference and every candidate automatic identity. Its identity
message is exactly:

```python
(" ".join(query.split()) + "\n" + requested_scope).encode("utf-8")
```

The independently de-identified public `corpus/dev.jsonl` is not an identity
source: its query surrogates cannot establish equality with a new build. The
complete dev identity reference is instead the operator-resolved, read-only raw
alt recall-event export already pinned by the original manifest at SHA-256
`45dbfdb389324a8702ba1a84b28be028991f23dcfe1d7ab9d771928c37c2cc7a`.
Inside the same isolated HMAC process, the original frozen split rule selects
the automatic dev population and the one fresh key covers both reference and
candidate messages. No raw field or token leaves that process. If this exact
raw reference is unavailable or hash-mismatched, readiness fails; the public
surrogate corpus or another source MUST NOT substitute for it.

A repeated-automatic family contains automatic events from the same source
alias with the same equality token, at least three events, and at least two distinct nonempty
source-qualified `transport_session_id` values. `unseen_in_dev` means its token
is absent from the complete frozen-dev automatic reference pinned in the
original manifest. Organic events never contribute to an automatic family.

Partitioning operates on connected components of selected events. Events are
connected when they share any source-qualified automatic identity equality
class, whether or not it later meets the repeated-family floor, or the same nonempty source-qualified
transport session/workflow key. Family qualification affects counts, never
edges. Thus no automatic identity class, organic session, or real workflow can
cross partitions. An event with neither edge is its own component. The
component representative is the
lexicographically smallest byte string `source_alias + NUL + event_id` in that
component. Compute:

```text
digest = SHA256(b"confirmatory-holdout-v2/partition/v1\0" + representative)
bucket = int.from_bytes(digest[0:4], "big") % 100
bucket 0..49  -> holdout
bucket 50..99 -> shadow
```

Every eligible event belongs to exactly one component and one partition;
partition intersection is empty and union equals the complete selected
population. Source qualification makes equal raw identifiers from `local` and
`alt` distinct. Input order and all outcome fields are irrelevant. HMAC or
identifier collisions fail closed. No searching for a favorable key,
rebalancing, truncation, reassignment, stratification, or alternate bucket rule
is permitted when a partition misses a floor.

## 4. A-priori readiness and stopping rule

These floors come from independence and measurement precision requirements,
not current availability. Every pre-seal readiness condition is conjunctive and
is evaluated after the fixed partition over the complete snapshot population:

- Holdout has at least 30 unseen-in-dev repeated-automatic families covering
  at least 150 events, distributed across at least 30 distinct partition
  components that contain a qualifying unseen family.
- Holdout has at least 200 organic events spanning at least 30 distinct
  nonempty source-qualified transport sessions and at least 30 distinct
  partition components that contain organic events.
- Holdout covers at least two distinct requested scopes beginning `project:`.
- Shadow has at least 30 genuine, nonsynthetic real workflows and at least 100
  unique replayable logical calls, distributed across at least 30 distinct
  partition components that contain a counted workflow.
- Shadow also covers at least two distinct requested `project:` scopes.

A counted workflow uses one source-blind proxy and no content classifier: a row
originates in `recall_events` of a `local` or `alt` production snapshot, has a
source-qualified nonempty `transport_session_id`, its workflow contains at
least one selected `agent IS NOT NULL` anchor, and it has at least one replayable
logical call. No undeclared fixture/synthetic heuristic is allowed. One logical
recall event counts as one call; its baseline and candidate observations are a
pair, not two calls. The frozen replay-input schema requires a nonempty query,
a string requested scope, nullable caller fields, a nonempty transport session
for workflow counting, and a hash-bound initial LM state containing every
referenced node. Caller nullness and equality are stored only as opaque
packet-local labels in the sealed de-identified replay envelope; event and
requested-scope values likewise become opaque labels, with only scope-kind
prefix and equality retained. Raw caller values are destroyed before sealing. Results, feedback, latency, and candidate
behavior are not replayability inputs. After implementation, evidence
sufficiency separately requires at least 100 complete measured pairs and at
least one such pair in each of at least 30 counted shadow workflows. Partial
instrumentation never counts as measured and makes evaluation insufficient or
failed rather than shrinking the denominator. No family, session, workflow,
call, or scope is pooled across partitions, and a scope spelling cannot count
twice.

Thirty connected components per evidence stream is the conservative
independence minimum for the fixed cluster bootstrap. A family, session, or
workflow alone is not called independent because the component graph can link
them transitively. At the event floors, 150 automatic events provide five
events per family on average and 7.5 expected misses at the 0.95 access gate;
the iid reference standard error is 0.0178 (95% normal half-width 0.0349).
Two hundred organic events provide 10 expected misses and an iid reference
half-width 0.0302 at 0.95. One hundred shadow calls provide five observations
in the upper five-percent latency tail. These are a-priori design checks, not
claims that observations inside a component are iid; whole-component
resampling reports the actual uncertainty.

The frozen readiness scanner gets exactly one launch. Its launch consumes the
attempt, and the two snapshot hashes from that attempt are the only permitted
packet-build sources. If any floor or integrity check fails, it emits an
aggregate `insufficient` receipt, publishes no packet, and permanently ends v2.
It MUST NOT rescan after accrual, recapture, supplement, widen, lower a floor,
change the partition, or stop at the first qualifying N. A later attempt needs
a new protocol namespace.

If and only if every floor passes, the complete holdout and shadow partitions
are published in one no-overwrite, manifest-last ceremony. Publication gets
one attempt. The namespace is sealed only when its canonical manifest is
present, hash-valid, says `frozen: true`, records readiness status `ready` and a
passing preseal receipt, and records zero semantic reads separately for both reserved
readers. Reader consumption is never written back into that manifest. Any
partial publication, validation failure, collision, key failure, or crash
abandons v2; it cannot be repaired or resealed.

## 5. Packet-before-implementation order

The mandatory order is:

1. Freeze this source-blind protocol commit.
2. Freeze a source-blind repair design and synthetic-tested aggregate-only
   readiness tooling. Neither may implement the production repair or semantic
   evaluator.
3. Run the one readiness attempt against the two read-only aliases.
4. If ready, publish and seal both deterministic partitions, manifest last.
5. Only after the packet-seal commit exists may repair-specific production,
   evaluator, calculator, configuration, or threshold changes begin.
6. Freeze one clean implementation tree and one evaluator/configuration stack.
7. Run the two reserved readers once under the embargo rule in section 1.
8. Release hash-bound aggregate evidence and run the unchanged parent gate.

The packet-seal commit MUST be a strict Git ancestor of every repair and
evaluator implementation commit. Timestamps alone are insufficient. A diff
from protocol freeze through packet seal MUST show no production repair or
semantic evaluator change. The final receipts bind both readers to the same
packet manifest, code tree, configuration, thresholds, calculator, evaluator,
dependency lock, and runtime settings by hash.

## 6. Measurement and uncertainty

Baseline and candidate execute the same ordered logical calls with paired
inputs from separate byte-identical, hash-bound pre-open clones for each
partition and source alias. The packet-state builder starts a fresh database,
copies only the `analysis-plan.json` allowlisted de-identified columns of
`nodes`, `connections`, and `retrieval_weights`, and copies no `recall_events`.
Every other source table or column is excluded; needing one is a fatal protocol
mismatch. Recall events, automatic fingerprint or signal history, delivery
history, sessions, caches, pending feedback, feedback/outcome tables, and
temporary tables all start empty. An unclassified mutable table fails preseal.
Each implementation then performs only its normal hash-bound migration. Their
databases, history, sessions, caches, and processes remain isolated; shadow and
holdout also share no mutable state. Initial-state, clone, migration, and
configuration hashes are bound in both receipts.

Within canonical event order, a domain-separated event hash predetermines
whether gating-off or candidate dispatches first, and both finish before either
arm advances. Before raw mappings are destroyed, the builder stores only that
precomputed one-bit dispatch order in each sealed case; the future reader never
needs the raw event key or its destroyed mapping. This interleaves paired
latency observations without changing either arm's internal order or using
outcomes. Any `content_ref` is checked only after primary calls finish, through
normal `memory_lookup` in a fresh isolated read-only resolver clone. Resolver
calls share no state, cache, or process with the primary run, and their payload
and latency do not enter recall metrics.

Full-sample point estimates are authoritative. The unchanged parent
P6/P7 point gates remain conjunctive: unseen repeated-automatic character
reduction at least 0.50; automatic content access at least 0.95; organic
payload and access deltas each within +/-0.05; agent-triggered delivery
byte-identical to gating-off behavior; every eligible novel node
content-bearing; and p50 and p95 shadow latency degradation each at most 0.10.
A missing metric, zero denominator, zero eligible novel-node population,
incomplete pair, or instrumentation gap fails closed. No alternate metric or
cohort may be substituted. Aggregate ratios use the same complete paired units
and the preregistered formulas in `analysis-plan.json`; there is no trimming or
per-partition metric redefinition.

The analysis plan freezes 10,000 whole-component resamples, separate
namespace-derived seeds for the automatic, organic, and shadow component
streams, a SHA-256 counter sampler, and two-sided 95% nearest-rank intervals.
Every replicate recomputes the exact full-sample estimator over the sampled
component multiset. Every result reports the full-sample point estimate,
interval, component and observation counts, seed, replicate count, and
invalid-replicate count.
Bootstrap means and intervals are supplementary only: they never replace,
average away, rescue, or weaken a parent point-estimate failure. An invalid
replicate count is reported and makes that interval null; it does not alter the
full-sample parent point-gate result.

## 7. Privacy and hash binding

Raw private rows, transcripts, queries, content, paths, identifiers, identities,
tokens, case records, and case-level outcomes MUST NOT enter prompts, version
control, stdout, stderr, diagnostics, or receipts. A future sealed corpus may
contain only freshly de-identified case records and query/content surrogates,
plus fresh opaque packet-local event/node/family/scope/agent/task/session/workflow
labels needed for replay, equality, and clustering; those values never appear
in prompts, logs, readiness output, or aggregate reports. Family membership,
per-family sizes, and de-identified case records remain confined to that sealed
access boundary.
The readiness output allowlist is limited to namespace/schema,
ready-or-insufficient status, fixed bounds, alias-bound snapshot hashes and byte
sizes, the total selected-event count, the fixed aggregate floor counts, and
protocol and candidate-plan hashes. There is no per-source outcome breakdown.

No HMAC key, key commitment, HMAC token or sample, plaintext or normalized
identity, unkeyed query fingerprint, or private equality map persists. Opaque
packet-local equality labels are random surrogates, not HMAC output. A separate
fresh ephemeral 256-bit salt performs length/whitespace-preserving
de-identification. The salt and bijective original-to-surrogate map never
persist. Keys and maps travel only through anonymous descriptors. Descriptor
closure, descendant termination, and keyed process-group exit are the hard
destruction boundary; only then may the launcher construct and install the
manifest.

The packet manifest hash-binds both exact source snapshots, capture watermark,
protocol documents, original manifest/splits/dev reference, retired replacement
manifest/policy, readiness scanner, builder, verifier, de-identifier, corpus
partitions, and preseal aggregate receipt by SHA-256 and byte size. Later
evaluation receipts additionally bind the immutable packet manifest, clean
repair Git commit and tree, evaluator, calculator, configuration, thresholds,
dependency/environment lock, and runtime settings. Mutable names such as
`HEAD`, paths without byte hashes, and timestamps are not evidence bindings.

Mechanical hashing, byte-size, schema, count, disjointness, and privacy checks
may be repeated only when they operate on the new v2 packet and emit no case
meaning. They grant no semantic authority. Raw private I/O remains outside the
repository and is destroyed after the authorized process boundary.
