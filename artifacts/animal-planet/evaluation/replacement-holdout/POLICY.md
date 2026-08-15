# Replacement holdout sealing and access policy

This policy applies only to the `replacement-holdout` namespace. It grants no
semantic evaluation access to the original animal-planet holdout and no
authority to modify either packet. Mechanical verification of the original is
required only as described below.

## 1. Current and sealed states

At the documentation-only revision that creates this policy, `corpus/` and
`manifest.json` are intentionally absent. The namespace is unsealed whenever
`manifest.json` is absent, even if corpus files are temporarily present during
publication. No evaluation or semantic access is permitted while unsealed.

The publisher MUST install `corpus/holdout.jsonl` and
`corpus/dev-fingerprint-index.json` before it installs `manifest.json`. It MUST
install every path without overwrite and MUST install the manifest last. A
namespace becomes sealed only when the canonical manifest is present,
mechanically valid, says `frozen: true` and `semantic_reads: 0`, carries a
passing keyed-preseal receipt, and hash-pins every source, implementation,
policy, and data artifact. Once sealed, every pinned byte is immutable; no
append, repair, replacement, regeneration, reseal, or in-place consumption
marker is allowed.

## 2. Population contract

The sealed population MUST contain every event satisfying the strict predicate

```text
created_at > 2026-08-12T23:13:24Z
created_at < 2026-08-13T20:16:51Z
requested_scope in {project:ae, project:online}
```

and no other event. Both time bounds are exclusive. There is no outcome
filtering. Automatic means `agent IS NULL`; organic means `agent IS NOT NULL`.
The sealed aggregates MUST be exactly 769 events: 332 automatic and 437
organic, with 502 requested in `project:ae` and 267 requested in
`project:online`.

A repeated-automatic family MUST contain only automatic events sharing one
opaque identity token, at least three events, and at least two distinct
`transport_session_id` values. The packet MUST contain exactly 18 such
families covering 238 events. `unseen_in_dev` means the family token is absent
from the same-key frozen-dev automatic index; exactly 16 families covering 113
events MUST be unseen. Organic events MUST NOT be counted in these families.

## 3. Identity and privacy

One fresh ephemeral 256-bit HMAC-SHA256 key MUST cover the frozen-dev automatic
identities and all replacement events. The identity message MUST be exactly

```python
(" ".join(query.split()) + "\n" + requested_scope).encode("utf-8")
```

and the stored token MUST be the full lowercase hexadecimal digest. The key
MUST travel to the keyed verifier only through an inherited anonymous
descriptor. The build MUST persist no HMAC key, key commitment, illustrative
token sample, plaintext or normalized identity, plaintext query, unkeyed
digest, or unkeyed query fingerprint. Only complete opaque keyed tokens needed
for equality membership may be present in the sealed data and dev index.

All private text MUST be freshly de-identified with a separate fresh ephemeral
256-bit salt. The salt and the private original-to-surrogate map MUST NOT be
persisted. Surrogates preserve Python character length and space, tab, newline,
and carriage-return positions; all other characters are lowercase `a`-`z`.
Equality classes are bijective within one build. Private raw text MUST NOT be
printed, committed, placed in the manifest, or exposed in diagnostics.

The manifest MUST pin the predeclared metadata, recall-event export, immutable
SQLite snapshot, original frozen manifest, original split declaration, and
de-identification implementation by SHA-256 and byte size (and source row count
where declared). It MUST also pin the exact README, policy, builder, verifier,
corpus, and dev index. The keyed verifier MUST rehash the complete original
packet and prove that its 6,076 event identifiers are disjoint from the 769
replacement identifiers. The original packet remains read-only.

## 4. Key-lifetime requirement

For a sealed packet, `frozen: true` has a process-lifetime meaning, not merely a
logical or buffer-state meaning. Before the launcher constructs or publishes
the manifest:

1. The keyed verifier MUST have closed its anonymous key descriptor, cleared
   its mutable key buffer, closed its receipt descriptor, and terminated.
2. The keyed worker MUST have waited for the verifier, cleared its mutable HMAC
   key and de-identification-salt buffers, emitted only aggregate status, and
   terminated.
3. The launcher MUST have observed worker status-descriptor EOF, killed the
   keyed worker process group to eliminate descendants, and waited for the
   worker. Every descendant or anonymous descriptor that could hold or carry
   key material MUST therefore have terminated or closed.

Mutable-buffer clearing is required defense-in-depth, but it is not the hard
destruction boundary: runtimes can retain transient immutable copies. Process
exit is the hard destruction boundary; descendant termination and descriptor
closure make that boundary complete for the keyed process group.
Manifest construction and manifest-last publication MUST occur only after that
boundary. No key-derived commitment or diagnostic may cross it; only the
allowlisted aggregate keyed-preseal receipt with `semantic_reads: 0` may do so.

## 5. First and only semantic read

`replacement-holdout-eval` is the **first and only authorized semantic
reader** of the sealed replacement packet. Its authority is one-shot:

- Before opening any replacement case, it MUST finish all synthetic,
  development, and evaluation preflight and freeze the evaluated code,
  configuration, thresholds, and measurement path.
- It MAY make exactly one semantic pass over a mechanically verified sealed
  packet and MUST atomically produce aggregate-only evidence bound to the
  packet manifest and evaluated implementation.
- It MUST NOT use replacement results to tune, select, repair, or rerun code,
  thresholds, prompts, gates, or policy.
- It MUST NOT inspect, display, log, export, or retain an individual case,
  case-level outcome, query surrogate, identity token, or family membership.
- It MUST NOT make or authorize a second semantic read. A crash, partial read,
  validation failure after semantic access begins, or inconclusive result does
  not restore the authorization; the gap MUST be reported without reopening
  the packet.
- It MUST NOT mutate the corpus or immutable manifest to record consumption.
  The manifest remains the pre-evaluation attestation `semantic_reads: 0`;
  one-shot consumption is recorded only in separate hash-bound aggregate
  evidence.

Hashing, byte-size checks, schema checks, count checks, and privacy scans by the
fixed keyless verifier are mechanical reads. They confer no semantic-read
authority and MUST NOT emit or expose case meaning. No actor, stage, evaluator,
or debugging workflow other than the single reader named above is authorized
for semantic access.
