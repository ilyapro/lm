# Animal-planet P6 replacement holdout

This directory is the separately namespaced, privacy-safe replacement temporal
supplement for the P6 evaluation. It does not replace or amend the original
animal-planet packet, and it grants no semantic evaluation access to it.

## Status and seal marker

At the documentation-only revision that establishes this contract,
`corpus/` and `manifest.json` are intentionally absent. These two public
documents were prepared source-blind from the pinned runner, independent
verifier, and original public manifest; no private staging file, SQLite
snapshot, corpus record, or source-backed fixture was opened to prepare them.

Operational status is determined only by the canonical manifest:

- The `replacement-holdout` namespace is **unsealed** while
  `manifest.json` is absent, including the short publication interval in which
  `corpus/` may have been installed but the manifest has not.
- Publication installs the corpus and frozen-dev index first and installs
  `manifest.json` last, without overwriting an existing path. The manifest is
  the seal commit marker.
- The namespace is sealed only when the last-installed manifest has
  `frozen: true`, `semantic_reads: 0`, a passing keyed-preseal receipt, and
  valid hashes and byte sizes for every declared data and packet file.
- Once sealed, the manifest and every file it pins are immutable. Appending,
  replacing, regenerating, or repairing any sealed byte creates a different
  packet and is forbidden in this namespace.

The historical documentation-only status above does not supersede the seal
test: after a future publication, manifest presence and successful mechanical
verification are authoritative.

## Predeclared selection

A valid seal contains every recall event from the pinned immutable SQLite main
snapshot that satisfies all of the following, and no event outside them:

```text
created_at > 2026-08-12T23:13:24Z
created_at < 2026-08-13T20:16:51Z
requested_scope in {project:ae, project:online}
```

Both time bounds are exclusive. Selection is ordered by `created_at ASC, id
ASC`. It has **no outcome filtering**: feedback, success or failure, result
content, event class, and later usefulness do not admit or remove an otherwise
matching event.

The class predicate is exact:

- `automatic` means `agent IS NULL`.
- `organic` means `agent IS NOT NULL`.

These are the mechanically required aggregate counts for a sealed packet:

| Population | Required aggregate |
| --- | ---: |
| Supplement events | 769 |
| Automatic events | 332 |
| Organic events | 437 |
| Requested scope `project:ae` | 502 |
| Requested scope `project:online` | 267 |
| Directly referenced nodes | 619 |
| Total included nodes | 804 |
| Frozen-dev events | 826 |
| Frozen-dev automatic events | 732 |
| Frozen-dev organic events | 94 |
| Frozen-dev unique automatic identities | 353 |
| Repeated-automatic families / events | 18 / 238 |
| Repeated-automatic families / events unseen in dev | 16 / 113 |

The counts are publication invariants. Their presence in this document does
not claim that an unpublished corpus exists.

## Shared opaque identity

One fresh ephemeral 32-byte HMAC key is used for both the frozen-dev automatic
index and all supplement events. The production identity is the tuple
`(collapsed query, requested_scope)`. Its exact normalization and message are:

```python
collapsed_query = " ".join(query.split())
message = (collapsed_query + "\n" + requested_scope).encode("utf-8")
fingerprint_token = HMAC_SHA256(identity_key, message).hexdigest()
```

Thus Python whitespace splitting strips leading and trailing whitespace and
collapses every internal whitespace run to one ASCII space; it does not
otherwise case-fold or rewrite the query. The requested scope is part of the
identity after a literal newline. The stored token is the full lowercase
hexadecimal HMAC-SHA256 digest.

The same key is applied mechanically to all 732 frozen-dev automatic events
and all 769 supplement events. The independent keyed verifier recomputes both
populations with that same key, proves equality membership and joint collision
freedom, and accepts only the sorted set of 353 unique frozen-dev automatic
tokens. The persisted values are opaque keyed equality labels, not portable
query fingerprints.

A repeated family is defined only over automatic supplement events. It is one
`fingerprint_token` with at least three events and at least two distinct
`transport_session_id` values. A repeated family is `unseen_in_dev` exactly
when its token is absent from the same-key frozen-dev automatic index. Organic
events never form repeated-automatic families, although their event identities
are still keyed and de-identified as part of the complete supplement.

No key, key commitment, illustrative token sample, normalized or plaintext
identity, plaintext query, unkeyed digest, or other unkeyed query fingerprint
is persisted or printed. Only the complete opaque tokens required by the
corpus and equality-only dev index may be sealed.

## Fresh de-identification

Every build uses a fresh independent ephemeral 256-bit surrogate salt; it is
not the identity HMAC key and is never persisted. Every private text value is
freshly transformed for this packet. Surrogates preserve exact Python
character length and the positions of spaces, tabs, newlines, and carriage
returns; every other character becomes a lowercase `a`-`z` character. Across
one build, equal originals map to equal surrogates and distinct originals map
to distinct surrogates. Neither the salt nor the private original-to-surrogate
mapping survives the keyed worker.

Only allowlisted structural values needed for replay and validation—such as
opaque record identifiers, timestamps, scopes, numeric fields, booleans, and
typed-edge structure—remain literal. There is no transcript capture for this
source: `transcript_matched` is false and `transcript_serialized_chars` is
null. The independent verifier scans the candidate for raw private text, key
material, encoded key material, unkeyed identity digests, and invalid
surrogates before it can issue a passing receipt.

## Immutable provenance and code pins

The future sealed manifest must bind these predeclared source inputs exactly:

| Input | SHA-256 | Bytes / rows |
| --- | --- | ---: |
| Staging metadata | `025e734f84f452711e86a31d1639f2ee9833213e7ec94ced69b95d315ebac855` | 25,382 bytes |
| Staging recall-event export | `45dbfdb389324a8702ba1a84b28be028991f23dcfe1d7ab9d771928c37c2cc7a` | 4,162,914 bytes / 1,576 rows |
| Immutable SQLite main snapshot | `4d6648f9e3c33e8ffaa7bf15620bd26a18bdf9f7ddf5b97641aa8c082e9aaa67` | 492,367,872 bytes |
| Original packet manifest | `3f1a6a87a4d34e411f06e50161f9203afc3c4992dfc973d4f544453e8bbfde48` | 34,934 bytes |
| Original split declaration | `66f04caa41e0302fd7b8582f54197a396a2bb8483ddd543e564740a8e4c3a7b7` | 4,040 bytes |
| De-identification implementation | `503de08cea164660c2f1c3b3675ad082f70109d8f4360194bb1e02c7423bd6e3` | 8,216 bytes |

The runner checks these values before work and again after independent keyed
verification. The original manifest must say `frozen: true`; its nested split
and de-identification pins and every original packet file are independently
rehash-verified. The replacement's 769 event identifiers must have an empty
intersection with all 6,076 event identifiers in that hash-verified original
packet. The original packet is read-only and is never modified.

At freeze time, the manifest also pins SHA-256 and byte size for the exact
reviewed `recipe/build.py`, the independently supplied and sealed
`recipe/verify.py`, this `README.md`, `POLICY.md`, `corpus/holdout.jsonl`, and
`corpus/dev-fingerprint-index.json`. Builder and verifier pins in the
`implementation` section must agree with their entries in `packet_files`;
corpus metadata must agree with `files`. This binds sources, code, public
policy, derived data, and the original-packet provenance in one immutable
manifest without embedding private source content.

## Key destruction and the meaning of `frozen: true`

The identity key exists only inside a forked keyed worker. For freeze, that
worker passes exactly 32 key bytes to the independently hash-pinned verifier
through an inherited anonymous pipe; the key is never placed in argv, the
environment, a regular file, a manifest field, or diagnostic output. The
unkeyed launcher never receives it. Verifier stdout and stderr are discarded,
and only an allowlisted aggregate receipt can return.

The lifecycle is part of the seal contract:

1. The keyed worker builds the candidate with fresh identity and surrogate
   secrets. The verifier closes its key descriptor after reading exactly 32
   bytes, mechanically recomputes the candidate, emits only the aggregate
   keyed-preseal receipt with `semantic_reads: 0`, clears its mutable key
   buffer, closes the receipt descriptor, and terminates.
2. The worker waits for that verifier, rechecks candidate and source pins,
   clears its mutable identity-key and surrogate-salt buffers, sends only
   aggregate status, and terminates. Its status descriptor remains open until
   process exit so EOF is a lifecycle boundary, not an in-process assertion.
3. After EOF, the launcher kills the worker process group to eliminate any
   surviving descendant and waits for the worker process. All anonymous key
   descriptors are therefore closed or owned only by processes that have
   terminated before control returns to manifest construction.
4. Only the unkeyed launcher then rehashes the candidate, constructs the
   `frozen: true` manifest, stages the two data files, and installs the
   manifest last.

Accordingly, `frozen: true` attests that the keyed worker, keyed verifier, all
of their descendants, and every anonymous descriptor capable of carrying the
identity key have terminated or closed **before manifest construction or
publication**. Clearing mutable buffers is defense-in-depth. It is not the
hard destruction proof, because a runtime may have made transient immutable
copies. Process exit is the hard destruction boundary; process-group
termination and descriptor closure ensure that it covers descendants and key
channels as well.

## Semantic-read boundary

Construction, keyed verification, keyless hash/schema verification, and seal
publication are mechanical operations and must leave `semantic_reads: 0`.
That immutable manifest value records the packet's state immediately before
evaluation; it is not edited after the one-shot evaluation.

`replacement-holdout-eval` is the first and only authorized semantic reader.
It may consume a successfully sealed, hash-verified packet exactly once and
may emit aggregate-only evidence bound to that manifest. It may not tune code
or thresholds, inspect or disclose individual cases, emit case-level
diagnostics, or read the packet a second time. Preflight must use synthetic,
development, or evaluation inputs without opening this packet. A failed or
partial semantic attempt consumes the single authorization and must be
reported as an evidence gap, not retried.

See `POLICY.md` for the normative access rules.
