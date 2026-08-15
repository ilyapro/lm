# Confirmatory holdout v2

Status: **preregistered, source-unprobed, and unsealed**.

This namespace freezes the confirmatory evidence design before any automatic
recall repair or semantic evaluator implementation. It contains no source
rows, source-availability counts, case data, tokens, outcomes, packet, or
evaluation result.

The immutable documents are:

- `POLICY.md`: authority, selection, privacy, lifecycle, and no-retry rules.
- `analysis-plan.json`: machine-readable predicates, partitions, floors,
  point gates, bootstrap, receipt allowlists, and hash-binding requirements.
- `README.md`: namespace status, lifecycle summary, and verification boundary.

`tests/test_confirmatory_evidence_protocol.py` pins both semantics and the
exact bytes of all three namespace documents. Future work may add a readiness
receipt and, only after a passing one-shot readiness decision, new sealed
packet artifacts. It may not edit these documents. Any correction or failed
readiness/publication ceremony requires a new versioned namespace.

## Fixed lifecycle

1. This protocol is committed.
2. A source-blind repair design and synthetic-only readiness scanner are
   frozen; production repair and semantic evaluator code remain untouched.
3. The scanner makes one aggregate-only attempt against the externally mapped
   read-only aliases `local` and `alt`.
4. An insufficient result closes v2 without a packet or retry. A ready result
   permits one no-overwrite, manifest-last publication from those exact
   hash-bound snapshots.
5. Both holdout and shadow partitions must be sealed before repair or evaluator
   implementation begins.
6. After a single implementation/configuration freeze,
   `confirmatory-shadow-v2-eval` and `confirmatory-holdout-v2-eval` each receive
   one partition-specific semantic pass. Their aggregate results stay embargoed
   until both readers terminate.

The consumed `shadow-eval` and `replacement-holdout-eval` authorities remain
retired. Mechanical readiness or integrity tooling is not a semantic reader.

## What readiness means

The scanner includes every event in the one captured source pair with
`created_at > 2026-08-13T20:16:51Z` and at or before the pre-open capture
watermark fixed to the readiness-launch UTC instant. It applies the frozen,
outcome-independent connected-component split
without truncation or rebalancing. Pre-seal readiness requires all holdout and
shadow component/event/session/workflow/replayable-call/scope floors in
`analysis-plan.json`; the floor values were not chosen from current
availability. The separate post-run evidence floor still requires at least 100
complete paired calls across at least 30 measured workflows.

Cross-build `unseen_in_dev` equality is computed only inside the isolated keyed
process from the original manifest-pinned raw dev identity reference. The
independently de-identified public dev corpus is never used as an identity
bridge and neither consumed sealed corpus is opened.

The only tracked readiness information may be alias-bound hashes, fixed bounds,
the total selected-event count, aggregate floor counts, plan hashes, and
`ready` or `insufficient`. This README must never be updated with observed
counts or results.

## Verification boundary

Focused static tests may read these public documents. They must not open either
consumed corpus or invoke its verifier. Later v2 tooling may mechanically check
only a newly published v2 packet. Semantic access belongs exclusively to the
two reserved readers and is consumed by process launch, including a partial or
failed launch.
