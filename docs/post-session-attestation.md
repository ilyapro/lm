# Grounded-usage attestation

`memory_attest` is the write path that lets a **finished** session's recall
events earn grounded credit afterwards, from the session's own artifacts. It
exists because the live credit loop only closes when the same agent, in the same
session, writes a `memory_remember` that grounds what it recalled — measured at
~16–20% of recalls on the live database, and 0% for AE-initiated ones. Recall is
a session-opening ritual (median position 0.02 of a session), `memory_remember`
a session-closing one (median 0.96); the work in between, where the evidence of
use actually lives, produced no signal at all.

The tool surface is in [mcp-interface.md](mcp-interface.md#memory_attest); the
implementation is `src/living_memory/attestation.py`, the offline evidence
assembler is `src/living_memory/postsession/evidence.py`, and the field check is
`scripts/attest_eval.py` → `artifacts/post-session/attestation-eval.json`.

## The rule that makes it honest: the client submits evidence, the server decides

A client sends a `recall_event_id` and a list of evidence strings lifted
verbatim from the session's artifacts. It does **not** send a verdict. The
server loads *that event's own* result nodes from its own database and
recomputes containment itself with
`grounding.ground_results(..., min_containment=RECALL_CREDIT_MIN_CONTAINMENT)` —
the same 0.25 gate, the same arithmetic, the same module as the live path.

A `grounded: true` in a client payload is ignored. Not filtered — *never read*:
`attestation.py` does not contain the key names `grounded`, `containment` or
`useful` as inputs anywhere, so there is nothing to spoof and nothing that could
regress into being trusted. The client's `context` is recorded for audit and
influences no decision.

This is what keeps applicability from being optimized by the party that benefits
from it. An extractor that wants a higher credit rate cannot assert one; it can
only submit more of the session's real artifacts, and if those artifacts do not
contain what the recall delivered, the rate does not move. The whole
post-session stage is graded on a metric it cannot write to.

## Why grounding runs per item, with a maximum, and not over one concatenation

Containment is the IDF-weighted share of a **node's** tokens that appear in the
graded document. The 0.25 threshold was calibrated
(`artifacts/grounding/calibration.md`, 72,023 result/trace pairs) with the
consuming `memory_remember` trace as that document — a trace-sized text.

Concatenating a whole session into one document and grading against it would
leave the calibrated regime entirely: a long enough document eventually contains
every common token, so nodes the session never touched clear 0.25 by accident.
So each evidence item is graded separately and each node keeps its **maximum**
across items, with `evidence_index` recording which item produced it. That is
also strictly more conservative than grading the union of the items: the union
can only ever contain more of a node's tokens than its largest single member.

Per-item grading makes the *item size* load-bearing, and the field check
measured what that costs rather than assuming a value (see
[Bounds, re-fitted on train](#bounds-re-fitted-on-the-train-split) below).

## Why agent prose is excluded from evidence

`postsession/evidence.py` draws evidence from exactly two places — diff hunks
from `SessionRecord.file_mutations`, and command-plus-output pairs from
non-memory `SessionRecord.tool_calls`. Two things are excluded on purpose, and
the exclusions are the point of the path:

**`Turn.text` — assistant and user prose.** An agent can restate a recalled
memory verbatim in its own sentences. If prose counted, a recall would ground on
the agent's paraphrase of itself, which rebuilds exactly the self-marking
circularity this path exists to escape: the goal is *a usage signal that does
not depend on the reading agent marking its own recall*. A recalled fact that
reaches a diff hunk or a command output has demonstrably influenced an artifact;
one that only reaches the agent's own sentences has not.

**`memory_*` tool payloads.** A `memory_recall` result *contains the recalled
node text*. Submitting it would ground every node on itself — containment 1.0
for free, for every result the event delivered. The call arguments are excluded
with the results, because a query is the agent's prose too.

Tool arguments other than the command itself are excluded for the same reason:
a sub-agent `Task` prompt, a `TodoWrite` list or a `WebFetch` question would
smuggle prose back in through a side door.

## Closure: credit without closure by default

`recall_events` has `feedback_trace_id TEXT REFERENCES nodes(id)`. Closing an
event therefore requires pointing at a **real node**.

* **Without `trace_id`** (the default): grounded results earn credit and anchor
  reinforcement, and `recall_events.feedback_applied` stays 0. Inventing a
  synthetic node to satisfy the foreign key would pollute `nodes` with
  non-knowledge and inflate `recall_fingerprints.linked_count` for events that
  produced no trace at all — corrupting the very delivered-versus-linked measure
  this work exists to make honest.
* **With a `trace_id` that resolves** — in practice the trace the same
  extraction wrote for that session — `MemoryStore.mark_recall_event_feedback`
  runs, so the 0 → 1 flip credits `linked_count` exactly once. On an
  already-closed event it is never called, because that path would take the
  legacy overwrite branch and clobber an existing `feedback_trace_id`.
* **An unresolvable `trace_id` is an error**, not a silent downgrade to
  credit-without-closure. A caller that believes it closed an event must not be
  told it did.

### The consequence, which is deliberate

An attested-but-unclosed event **stays available to the live pending-consumption
path**. `MemoryStore.pending_recall_events` selects on `feedback_applied = 0`,
so a later in-session `memory_remember` can still consume it and close it
normally. That is the intended behaviour: attestation adds a usage signal, it
does not consume the event on the live loop's behalf. Attesting an
already-closed event likewise still applies credit — evidence and a consuming
trace are two different observations of use, and the guard against repeated
attestation is the ledger, not `feedback_applied`.

## The ledger key, and what "idempotent" does and does not cover

The `recall_attestations` table (schema v8) holds
`UNIQUE(recall_event_id, evidence_sha256)`. `evidence_sha256` is SHA-256 over
the **canonical** items joined with `\x1e` (ASCII RECORD SEPARATOR — not a
newline, because items contain newlines and a newline join would let two
different item splits hash the same). Canonicalization normalizes CRLF, strips
trailing whitespace per line, and drops leading/trailing blank lines.

**What idempotency covers.** Re-running the extractor over the same transcript
re-derives byte-identical evidence (`evidence.py` is deterministic by
construction) and therefore the same digest, so the repeat returns the recorded
verdict with `replay: true` and applies nothing — no second credit, no second
anchor reinforcement, no second `linked_count` flip. The claim-then-complete
sequence means a concurrent duplicate loses the key and replays rather than
double-crediting.

**What it does not cover.** The key is over *bytes*, not over *meaning*:

* a session re-extracted after its transcript grew, or with different assembler
  caps, produces a different digest and is graded and credited again. The same
  underlying use can therefore be counted more than once if a caller varies the
  evidence it submits;
* it says nothing across events — the same evidence attested against two
  different `recall_event_id`s is two independent attestations, which is correct
  (they are different events) but is not deduplication of *use*;
* it is not a bound on total credit. Idempotency stops a replay, not a
  determined caller submitting many distinct evidence sets for one event;
* **changing the canonicalization or the separator re-opens every key ever
  written** — a client resubmitting identical artifacts afterwards would credit
  them a second time. Those constants are a versioned API, not an
  implementation detail. Changing the *caps* does not re-open old keys but does
  change the bytes a fresh assembly produces, which is the same effect for any
  session re-extracted afterwards.

## Field measurement

`scripts/attest_eval.py` answers the falsifiable question: does the
recomputation actually separate a session's own artifacts from someone else's?

It reads **eval-split sessions only** from the sealed corpus, rebuilds each
session with `postsession.transcripts.load_transcript`, assembles its evidence
with `postsession.evidence`, reconstructs a **scratch** SQLite database from the
transcripts (one node per delivered node, one `recall_events` row per recall
with the same rank order and per-channel scores), starts a **throwaway**
`python -m living_memory.server` on a free port with a fresh `LM_AUTH_TOKEN`
over that scratch file, and drives every grading through `memory_attest` over
MCP. It never opens `~/.local/share/living-memory/global.sqlite3` and never
contacts the live unit on 127.0.0.1:8765 — `MemoryStore.__init__` migrates and
writes whatever file it opens, so an offline process opening the live database
is a schema write behind the running server's back.

Two arms, same events, same grader, same evidence documents — only the pairing
differs:

* **TRUE**: every session's evidence against its own recall events;
* **SHUFFLED**: the same events against another session's evidence, under a
  cyclic derangement whose offset is drawn from SHA-256 over the sorted session
  keys (reproducible without `random`) and chosen so no session is paired with a
  twin recording of itself — twins share a corpus `split_key`, and pairing them
  would be a same-session grading wearing a different name.

Only deliveries with `content_truncated == False` become nodes. A snippet has a
*smaller* token set than the node it stands for, and containment divides by the
node's own mass, so grading against a snippet makes the gate **easier** to clear
and would inflate the result in the flattering direction. Truncated deliveries
keep their rank slot in the replayed results, so the surviving results' rank
decay is unchanged.

### Results (held-out eval split)

Full report: `artifacts/post-session/attestation-eval.json`, corpus manifest
`artifacts/post-session/corpus.json` (sha256 `625eb97f…`, sealed eval digest
`51fd1409…`), commit `8ff02ee` with `worktree_dirty: true`.

The commit is the *parent* of the change that carries the re-fitted caps, so on
its own it would attribute these numbers to code that never ran. What identifies
the measured code is `code.measured_source_sha256` in the report — SHA-256 over
`attestation.py`, `grounding.py`, `postsession/evidence.py` and
`attest_eval.py` as they were when the run happened. Re-running after any edit
to those four files changes the digests; re-running with them unchanged
reproduces the numbers exactly.

| | TRUE pairing | SHUFFLED pairing |
| --- | ---: | ---: |
| sessions | 264 | 264 |
| (evidence, event) pairs | 758 | 758 |
| grounded pairs | 178 | 18 |
| **grounded rate** | **0.2348** | **0.0237** |
| sessions with ≥1 grounded event | 114 | 14 |
| per-pair max containment, median | 0.185 | 0.109 |
| per-pair max containment, p95 | 0.414 | 0.210 |
| per-pair max containment, max | 1.000 | 0.453 |

Separation **9.89×**. Every bar clause holds:

| clause | required | measured |
| --- | --- | ---: |
| `shuffled.grounded_rate` | ≤ 0.05 | 0.0237 |
| `eval.grounded_rate` | ≥ 3 × shuffled | 9.89 × |
| `eval.sessions` | ≥ 30 | 264 |
| `eval.sessions_with_grounded_event` | ≥ 3 | 114 |
| `holdout_sessions_read` | 0 | 0 |

Coverage of the split: 748 sealed eval sessions read, 264 usable (a session is
usable when the transcript kept a `recall_event_id`, at least one delivered node
with untruncated content, and at least one evidence item). 8,484 deliveries
seen, 6,951 gradable, **1,533 excluded as truncated**. 997 distinct nodes and
758 recall events were reconstructed into the scratch database; 1,516
attestations were driven over MCP with zero failures and zero replays.

`holdout_sessions_read: 0` is accounting, not an assertion. Every transcript
open goes through one funnel that looks the session key up in the *recovered
sealed membership* and counts the open under that split before touching the
file; a key outside the split under measurement raises. The recovery is exact:
the index that lists session keys is gitignored and rebuilt from disk, so it
drifts as new sessions accumulate, and the script searches the rows written
after the seal for the subset whose removal reproduces each sealed digest
byte-for-byte. It found one for all three splits (eval 748 + 7 new, holdout
837 + 6, train 2368 + 7), which proves the drift is purely additive — **no
session moved split** — so "not in the sealed holdout" is a fact about the seal
and not about a mutable label. If that reconciliation ever fails the script
aborts rather than measuring.

### Bounds, re-fitted on the train split

The first measurement, at the assembler's original caps (64 items × 4000 chars,
64000 total), came out at **eval 0.699 / shuffled 0.226** — a 3.1× separation
but a shuffled rate 4.5× over the bar. The mechanism was crediting noise, and
the reason was the item size: containment divides by the node's own token mass,
so a large enough evidence item covers a quarter of any node's tokens by
coincidence, whoever wrote it.

Measured on train (`memory_remember`/`memory_teach` content over 2,368 train
sessions, n = 1,876 writes), the trace-sized document the 0.25 threshold was
calibrated on is **572 chars at the median**, 889 at p75, 2,281 at p95. The
original 4000-char item cap sat *above the p95* — seven times the median — so
every item was graded outside the calibrated regime.

Sweep over the caps, **train split only**, 1,057 usable sessions / 2,496 pairs,
with the grading arithmetic cross-checked against `attestation._grade` itself:

| max_items | max_item_chars | max_total_chars | train TRUE | train SHUFFLED | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4000 | 64000 | 0.6639 | 0.2090 | 3.18 |
| 32 | 800 | 25600 | 0.2740 | 0.0319 | 8.59 |
| 64 | 600 | 38400 | 0.2564 | 0.0281 | 9.12 |
| 48 | 600 | 28800 | 0.2392 | 0.0243 | 9.83 |
| **64** | **600** | **19200** | **0.2292** | **0.0231** | **9.93** |
| 32 | 600 | 19200 | 0.2123 | 0.0201 | 10.54 |
| 24 | 600 | 14400 | 0.1923 | 0.0193 | 9.96 |
| 32 | 500 | 16000 | 0.1807 | 0.0176 | 10.25 |

Adopted, in `attestation.py`:

* `EVIDENCE_MAX_ITEM_CHARS: 4000 → 600` — one item is **one trace-sized
  document**, the scale the threshold was fitted on;
* `EVIDENCE_MAX_TOTAL_CHARS: 64000 → 19200` — 32 trace-sized documents. Because
  each node keeps its *maximum* across items, every extra item is another
  independent chance of clearing 0.25 by coincidence, so the total is the bound
  that decides how many chances there are. At the old item size 64000 could
  never bind at all;
* `EVIDENCE_MAX_ITEMS: 64` — **unchanged**. A session made of many small hunks
  should still be able to send them all; the total is what bounds the noise.

Nothing else moved. `ATTESTATION_MIN_CONTAINMENT` and
`RECALL_CREDIT_MIN_CONTAINMENT` stay at 0.25: that value is calibrated in
`artifacts/grounding/calibration.md` and moving it to make a number look better
is exactly the "optimize the applicability metric" failure the design forbids.
The grounding rule, the credit arithmetic and the tool contract are unchanged;
only the size of the document handed to the calibrated rule was corrected.

The tuning generalized, on both the sweep's own arithmetic and the real harness:

| | TRUE | SHUFFLED | ratio | sessions | pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| train, offline sweep | 0.2292 | 0.0231 | 9.93 | 1057 | 2496 |
| train, through the server (`attestation-eval-train.json`) | 0.2396 | 0.0236 | 10.14 | 827 | 2496 |
| **eval, held out (`attestation-eval.json`)** | **0.2348** | **0.0237** | **9.89** | 264 | 758 |

The held-out arms land within 2% of the split the caps were fitted on. That is
the evidence that 600 / 19200 is a property of the document scale the threshold
was calibrated at, not a value fitted to the train sessions' noise. (The two
train rows disagree on the session count because the harness only counts a
session once it also produced a non-empty evidence bundle; both cover the same
2,496 pairs.)

### What would have to be tightened if the shuffled rate drifts toward the bar

The shuffled rate is a property of the corpus as much as of the code: it rises
when cross-session evidence starts to *look like* the recalled nodes. That
happens for real reasons — more sessions in one repository, a codebase whose
vocabulary the memory nodes describe closely, an agent that pastes large
generated files into diffs. Re-run `scripts/attest_eval.py` after any corpus
re-seal, and if the shuffled rate approaches 0.05, tighten in this order:

1. **`EVIDENCE_MAX_TOTAL_CHARS`** first. It is the number of independent chances
   at the per-item maximum, and the sweep shows it moves the shuffled rate
   faster than the true rate (19200 → 16000 cost 0.11 of true rate but 0.13 of
   shuffled).
2. **`EVIDENCE_MAX_ITEM_CHARS`** next, toward the *median* trace size rather
   than below it. Below ~400 chars both arms collapse together (at 1 item × 400
   the separation is 1.7×): an item too small to contain a whole thought
   grounds nothing, and the true rate falls faster than the noise.
3. **A temporal filter on the evidence**, which is currently a deliberate
   non-goal in `evidence.py`: every event of a session is offered the same
   bundle, including artifacts produced *before* that recall happened. Filtering
   each event's evidence to what followed it is a real tightening — a node
   cannot have influenced a diff written before it was delivered — and it is the
   next lever with headroom. It costs one digest per event instead of one per
   session, which weakens the idempotency key's reuse across a session's events;
   that is the trade to weigh.

Three things are **not** available as responses. Lowering
`ATTESTATION_MIN_CONTAINMENT` would make credit easier exactly when the noise
rose. Raising it above the calibrated 0.25 to suppress a symptom is the same
mistake in the other direction and needs a recalibration run, not an edit.
Grading against snippet-truncated deliveries would flatter the number by
shrinking the denominator. If none of the permitted levers reach the bar, the
right outcome is to say the mechanism does not discriminate on that corpus —
not to move the bar.

## Reproducing

```sh
# rebuild the session index this check reads. BOTH outputs are throwaway:
# artifacts/post-session/corpus.json is the seal and a read-only consumer must
# not rewrite it, and artifacts/post-session/corpus-index.jsonl is the seal's
# own slot — an index parked there asserts a seal that has genuinely moved as
# new sessions accumulate, which tests/test_postsession_corpus.py rightly
# fails on. So the field check keeps its own copy under the ignored .cache/.
python3 scripts/postsession_corpus.py --build \
    --manifest /tmp/corpus-rebuild.json \
    --index .cache/post-session/corpus-index.jsonl

# held-out measurement (writes artifacts/post-session/attestation-eval.json,
# exits non-zero if the bar fails)
python3 scripts/attest_eval.py

# the split the caps were fitted on
python3 scripts/attest_eval.py --split train \
    --out artifacts/post-session/attestation-eval-train.json
```

The relationship between that rebuilt index and the seal is *proved*, not
assumed: the reconciliation described above reproduces each tracked split
digest byte-for-byte from the rebuilt rows, and the run aborts if it cannot.

`--split holdout` is not offered by the CLI, and the reader would refuse the
open even if it were.

One consequence worth stating: because the rebuilt index lives outside the
seal's slot, `tests/test_postsession_corpus.py::
test_tracked_manifest_matches_its_index_when_one_is_present` *skips* (`no local
index; run scripts/postsession_corpus.py --build`). That test checks the seal by
equality, which cannot hold once the machine has accumulated sessions; the
reconciliation above checks the same seal by reproduction, which can. The
reconciliation is the stronger statement — it recovers the sealed membership
rather than asserting the totals still match — but it lives in this script, not
in the suite, so it is worth knowing which of the two actually ran.

## Verification of this change

* `python3 -m pytest -q` — 1354 collected, **1270 passed, 84 skipped, 0
  failed** (exit 0). The 12 `tests/test_usage_attestation.py` fixtures that
  carried 673–1019-char artifacts were trimmed to fit the new 600-char graded
  unit, preserving what each one grounds (`EVICTION_DIFF` still grounds the
  redis node at 0.758 and nothing else; `ALEMBIC_OUTPUT` still grounds the
  alembic node at 0.758; the negative control still peaks at 0.079). The
  control's size guard is now expressed against `EVIDENCE_MAX_ITEM_CHARS`
  instead of a literal, so a future re-fit cannot quietly shrink it into a toy.
* `python3 scripts/check_deployed_protocol.py --port <spare> --token <scratch>`
  against a throwaway server started from this worktree over a scratch database
  — `OK: … serves this checkout's protocol texts (5 checked: memory_consolidate,
  memory_recall, memory_remember, memory_teach, server instructions)`, exit 0.
  The four protocol-bearing descriptions are untouched by this work;
  `memory_attest`'s own description changed only where it quotes the caps,
  which the checker does not compare and clients would otherwise read as a lie.
