# Corpus usage policy

This directory is the privacy-safe de-identified replay corpus of the
animal-planet audit packet: `dev.jsonl`, `eval.jsonl`, `holdout.jsonl`,
`splits.json`. Splits were declared in advance by a hash rule recorded in
`splits.json` — never by inspecting event content or outcomes.

## holdout.jsonl is SEALED

- Build-phase children of the audit-replay-packet node and the four packet
  siblings (extract-stage, build-corpus, curate-failures, baseline-calculator,
  freeze-manifest) may touch `holdout.jsonl` **only mechanically**: hashing,
  format/invariant validation, and generic whole-file metric aggregation
  inside `scripts/ap_baseline.py verify`.
- **No per-case inspection** of holdout events or nodes: no reading individual
  records, no debugging against specific holdout cases, no example-picking.
- **No tuning on holdout**: no threshold, weight, heuristic, or prompt change
  may be evaluated against `holdout.jsonl` during development; `compare`
  workflows run on `dev` and `eval` only.
- The **first semantic read** of holdout content belongs to the
  holdout-shadow-eval node, after the packet is frozen.
- The failures-curation child assigns cases from `dev`/`eval` only; every
  failure-case event id must be resolvable in `dev.jsonl` or `eval.jsonl` and
  absent from `holdout.jsonl`.

Holdout generalization coverage: besides the ~15% hash slice of animal-planet
(alt) events, `holdout.jsonl` carries all exported local-DB workloads from
three non-animal-planet project scopes (`project:octopus`, `project:online`,
`project:x`) — these appear in no other split.

## Privacy

Every private text field in these files is a length/whitespace/equality-class
preserving surrogate (see `../recipe/02-transform.md`); the HMAC salt was
ephemeral and never recorded. Do not attempt re-identification. The private
originals live only in the untracked staging area and must never be committed
or pasted into prompts, per the packet's hard rules.

Byte-level identity of the corpus is guaranteed by these frozen tracked files
(hash-pinned in the packet manifest), not by re-running the transform: a
rebuild draws a fresh salt and produces different surrogate letters with
identical lengths, structure, equality classes and therefore identical
metrics.
