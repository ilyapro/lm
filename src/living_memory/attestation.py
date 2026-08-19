"""Grounded-usage attestation: retroactive credit for a finished session.

The live credit loop only closes when the *same* agent, in the *same* session,
writes a ``memory_remember`` that grounds what it recalled. Field measurement
(2026-08-19) puts that at ~16-20% of recalls, and 0% for AE-initiated ones:
recall is a session-opening ritual, remember a session-closing one, and the
work in between is where the evidence of use actually lives. This module is the
write path that lets a *finished* session's recall events earn that credit
afterwards, from the session's own artifacts.

**The client submits evidence, the server decides.** A client sends a
``recall_event_id`` plus evidence text lifted from the session's artifacts
(diff hunks, command output). The server loads *that event's own* result nodes
from its own database and recomputes containment itself. Any ``grounded``,
``useful``, ``containment`` or similar verdict a client puts in its payload is
never read — this module does not contain the key names, so there is nothing to
spoof. That is the whole honesty guarantee: applicability cannot be optimized
by the party that benefits from it, only evidenced.

Grounding runs **per evidence item**, taking each node's maximum containment
across items and recording which item produced it. This is a deliberate
departure from grading one concatenated document, and it is the difference
between a measurement and a formality. The 0.25 threshold was calibrated
(``artifacts/grounding/calibration.md``) on trace-sized documents; a single
concatenation of a whole session would clear 0.25 by accident for nodes the
session never touched, purely because a long document eventually contains every
common token. Per-item maximum is also strictly more conservative than grounding
against the union of the items: the union can only ever contain more of a node's
tokens than its largest single member does.

*How much* text one item may carry is therefore load-bearing, and the field
check measured it instead of assuming it. Grading each session's events against
**another** session's artifacts — same grader, same documents, wrong pairing —
credited 20.9% of pairs at the first draft's 4000-char item cap: 4000 chars is
seven times the median ``memory_remember`` trace, and a long enough document
contains every common token whoever wrote it. Holding one item to one
trace-sized document (600 chars) and one submission to 32 of them takes that
cross-session rate to 2.3% while the same-session rate stays at 22.9% — a
tenfold separation instead of a threefold one. See
``docs/post-session-attestation.md``.

Credit mirrors the live path exactly — same
:func:`feedback.apply_retrieval_feedback` call, same
``max(0.2, 1 / (rank + 1))`` rank decay, same
:func:`feedback._reinforce_query_anchors` for the anchor write, all under the
**event's** scope rather than the attesting client's. Nothing about the
arithmetic is reimplemented here.

Two live-path behaviours are deliberately *not* mirrored:

* ``LM_RECALL_CREDIT_POLICY`` is ignored; attestation is always
  grounded-positive-only. ``all`` would reinforce everything the event
  delivered, which is precisely the vacuous signal the grounding work removed
  (87.9% of reinforcements went to results the trace never used).
  ``grounded_negative`` would blame nodes for not appearing in evidence that is
  *bounded and redacted* by construction — absence from a capped evidence
  sample is not evidence of absence, so the blame would be unfounded.
* There is no session-identity gate. The offline extractor connects with a new
  transport session id by construction, so any same-session check would break
  the feature outright. The attesting agent/session/source session is recorded
  in the ledger for audit; the guarantee comes from server-side recomputation,
  not from provenance.

**Closure defaults to credit without closure.** With no ``trace_id``, credit and
anchors are applied and ``recall_events.feedback_applied`` stays 0. The column
``feedback_trace_id TEXT REFERENCES nodes(id)`` means closing an event requires
pointing at a real node, and inventing a synthetic node to satisfy that foreign
key would pollute ``nodes`` with non-knowledge and inflate
``recall_fingerprints.linked_count`` for events that produced no trace at all —
corrupting the very delivered-vs-linked measure this work exists to make
honest. With a ``trace_id`` that resolves, :meth:`MemoryStore.
mark_recall_event_feedback` runs so the 0 -> 1 flip credits ``linked_count``
exactly once; on an already-closed event it is never called, because that would
take the legacy overwrite branch and clobber an existing ``feedback_trace_id``.
An unresolvable ``trace_id`` is an error, not a silent downgrade to
credit-without-closure: a caller that believes it closed an event must not be
told it did.

Attesting an already-closed event still applies credit. Evidence and a
consuming trace are different observations of use, and the guard against
repeated attestation is the ledger — ``UNIQUE(recall_event_id,
evidence_sha256)`` — not ``feedback_applied``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
import hashlib

from living_memory import feedback
from living_memory.grounding import ground_results
from living_memory.models import RecallEvent
from living_memory.storage import MemoryStore, RecallAttestation

# ---------------------------------------------------------------------------
# Canonicalization: a published contract
#
# The offline client imports these to build and pre-check its submission, so
# the digest it computes locally is the digest the server records. Changing any
# of them changes what ``evidence_sha256`` means, which re-opens every ledger
# key ever written: a client that resubmits identical artifacts after a change
# would credit them a second time. Treat as versioned API.
# ---------------------------------------------------------------------------

#: Most evidence items one attestation may carry. Unchanged by the field check:
#: a session made of many small hunks should be able to send them all, and the
#: total below is what actually bounds the noise.
EVIDENCE_MAX_ITEMS = 64
#: Most characters one canonical evidence item may carry — **one trace-sized
#: document**, because that is the scale the 0.25 threshold was calibrated on.
#: Measured over the corpus's train split (``scripts/attest_eval.py``,
#: ``artifacts/post-session/attestation-eval.json``): ``memory_remember`` /
#: ``memory_teach`` content is 572 chars at the median over 1876 writes, 889 at
#: p75, 2281 at p95. The first draft's 4000 sat above that p95 — seven times the
#: median — so every item was graded outside the calibrated regime, and the
#: shuffled cross-session arm of the field check credited 20.9% of pairs. At 600
#: it credits 2.3%.
EVIDENCE_MAX_ITEM_CHARS = 600
#: Most characters one attestation may carry across all its items: 32 trace-sized
#: documents. Grounding takes each node's *maximum* over the items, so every
#: extra item is another independent chance to clear 0.25 by coincidence, and the
#: total — not the item count — is the bound that decides how many chances there
#: are. Was 64000, which at the old item size could never bind.
EVIDENCE_MAX_TOTAL_CHARS = 19200

#: Containment gate, held equal to the live credit path's. An attestation must
#: not be easier to earn than a same-session grounded consumption.
ATTESTATION_MIN_CONTAINMENT = feedback.RECALL_CREDIT_MIN_CONTAINMENT

#: Digest separator: ASCII RECORD SEPARATOR. Not a newline, because evidence
#: items *contain* newlines and a newline join would let two different item
#: splits hash the same.
EVIDENCE_SEPARATOR = "\x1e"


class AttestationError(ValueError):
    """A submitted attestation cannot be graded as sent.

    Raised for every client-payload problem: an over-cap submission, evidence
    with nothing gradable in it, an unknown ``recall_event_id``, an
    unresolvable ``trace_id``. Always a rejection, never a silent downgrade —
    a client must never be able to believe the server graded more, or closed
    more, than it did.
    """


def canonical_evidence(items: Iterable[Any]) -> tuple[tuple[str, ...], str]:
    """Canonical evidence items plus their SHA-256 digest.

    Per item: coerce to ``str``, normalise newlines to ``\\n``, strip trailing
    whitespace from each line, strip leading and trailing blank lines. Items
    that canonicalise to nothing are dropped — they carry no evidence, and
    keeping them would make the digest depend on invisible whitespace.

    Caps are enforced rather than applied: an over-cap submission raises
    :class:`AttestationError` naming the cap it broke. Truncating instead would
    let a client believe it submitted more than the server graded, which is
    exactly the kind of quiet gap this path exists to close. The item count is
    checked against what was *sent*, before empties are dropped; sizes are
    checked against the canonical text, so no one is rejected over trailing
    whitespace alone.
    """

    if isinstance(items, str | bytes):
        raise AttestationError("evidence must be a list of strings, not a single string")
    raw = list(items)
    if len(raw) > EVIDENCE_MAX_ITEMS:
        raise AttestationError(
            f"evidence carries {len(raw)} items, over EVIDENCE_MAX_ITEMS={EVIDENCE_MAX_ITEMS}"
        )
    canonical: list[str] = []
    for index, item in enumerate(raw):
        text = _canonical_item(item)
        if not text:
            continue
        if len(text) > EVIDENCE_MAX_ITEM_CHARS:
            raise AttestationError(
                f"evidence item {index} is {len(text)} chars, over "
                f"EVIDENCE_MAX_ITEM_CHARS={EVIDENCE_MAX_ITEM_CHARS}"
            )
        canonical.append(text)
    total = sum(len(text) for text in canonical)
    if total > EVIDENCE_MAX_TOTAL_CHARS:
        raise AttestationError(
            f"evidence totals {total} chars, over "
            f"EVIDENCE_MAX_TOTAL_CHARS={EVIDENCE_MAX_TOTAL_CHARS}"
        )
    digest = hashlib.sha256(
        EVIDENCE_SEPARATOR.join(canonical).encode("utf-8")
    ).hexdigest()
    return tuple(canonical), digest


def _canonical_item(item: Any) -> str:
    text = item if isinstance(item, str) else str(item)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Graded:
    """One of the event's own results, as the server graded it."""

    node_id: str
    rank: int
    containment: float
    grounded: bool
    evidence_index: int
    #: The result payload the event recorded — the per-channel raw scores
    #: credit attribution needs. Server-side data; never client input.
    payload: dict[str, Any]

    def as_result(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "rank": self.rank,
            "containment": self.containment,
            "grounded": self.grounded,
            "evidence_index": self.evidence_index,
        }


def _grade(
    items: Sequence[str], resolved: Sequence[tuple[int, dict[str, Any], Any]]
) -> list[_Graded]:
    """Per-item containment, kept at each node's maximum across items.

    One :func:`grounding.ground_results` call per evidence item, over all of
    the event's resolved results at once, so the IDF index of each call is the
    item plus the nodes being graded — the same shape the live path builds from
    a consuming trace plus that event's results. ``evidence_index`` names the
    item that produced the reported containment; ties keep the lowest index,
    and a node no item touched reports index 0 with containment 0.0.
    """

    contents = {node.id: node.content for _rank, _payload, node in resolved}
    best: dict[str, tuple[float, int]] = {node_id: (0.0, 0) for node_id in contents}
    for index, item in enumerate(items):
        for node_id, verdict in ground_results(
            item, contents, min_containment=ATTESTATION_MIN_CONTAINMENT
        ).items():
            if verdict.containment > best[node_id][0]:
                best[node_id] = (verdict.containment, index)
    graded: list[_Graded] = []
    for rank, payload, node in resolved:
        value, index = best[node.id]
        graded.append(
            _Graded(
                node_id=node.id,
                rank=rank,
                containment=value,
                grounded=value >= ATTESTATION_MIN_CONTAINMENT,
                evidence_index=index,
                payload=payload,
            )
        )
    return graded


# ---------------------------------------------------------------------------
# The write path
# ---------------------------------------------------------------------------


def attest_recall_usage(
    store: MemoryStore,
    recall_event_id: str,
    evidence: Sequence[str],
    *,
    context: Mapping[str, Any] | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Grade one recall event against submitted evidence and credit what it used.

    Returns the verdict dict the ``memory_attest`` tool serves. See the module
    docstring for the rules; the order below is load-bearing:

    1. canonicalise and digest the evidence — a rejected submission must cost
       no reads and no writes;
    2. load the event and check the ledger — a repeat replays before anything
       is resolved;
    3. resolve the event's results *from the store* and recompute containment;
    4. claim the ledger key, then apply credit, anchors and (optionally)
       closure, then record what was applied.

    Results whose node has since decayed or been forgotten are skipped
    silently: a node disappearing between recall and attestation is ordinary,
    and it is not evidence about the rest of the event.
    """

    items, digest = canonical_evidence(evidence)
    if not items:
        raise AttestationError("evidence carries no gradable item")

    event = store.get_recall_event(str(recall_event_id))
    if event is None:
        raise AttestationError(f"unknown recall_event_id {str(recall_event_id)!r}")

    recorded = store.find_recall_attestation(event.id, digest)
    if recorded is not None:
        return _verdict(recorded, event, replay=True)

    trace_node = None
    if trace_id is not None:
        trace_node = store.get_node(str(trace_id))
        if trace_node is None:
            raise AttestationError(
                f"trace_id {str(trace_id)!r} resolves to no node; an event can only "
                "be closed against a real one"
            )

    resolved: list[tuple[int, dict[str, Any], Any]] = []
    for rank, result in enumerate(event.results):
        node_id = str(result.get("node_id") or "")
        if not node_id:
            continue
        node = store.get_node(node_id)
        if node is None:
            continue
        resolved.append((rank, dict(result), node))

    graded = _grade(items, resolved)
    grounded_ids: list[str] = []
    for entry in graded:
        if entry.grounded and entry.node_id not in grounded_ids:
            grounded_ids.append(entry.node_id)

    claimed = store.claim_recall_attestation(
        recall_event_id=event.id,
        evidence_sha256=digest,
        evidence_items=len(items),
        evidence_chars=sum(len(item) for item in items),
        min_containment=ATTESTATION_MIN_CONTAINMENT,
        containments=[entry.as_result() for entry in graded],
        grounded_node_ids=grounded_ids,
        context=context,
    )
    if claimed is None:
        # Another writer took the key between the read above and here. It owns
        # the credit; this call replays its verdict and applies nothing.
        recorded = store.find_recall_attestation(event.id, digest)
        if recorded is None:  # pragma: no cover - only a concurrent delete
            raise AttestationError("attestation ledger key vanished mid-write")
        return _verdict(recorded, event, replay=True)

    credited = False
    for entry in graded:
        if not entry.grounded:
            continue
        feedback.apply_retrieval_feedback(
            store,
            SimpleNamespace(
                node_id=entry.node_id,
                bm25_score=float(entry.payload.get("bm25_score", 0.0) or 0.0),
                vector_score=float(entry.payload.get("vector_score", 0.0) or 0.0),
                graph_score=float(entry.payload.get("graph_score", 0.0) or 0.0),
            ),
            useful=True,
            signal=max(0.2, 1.0 / (entry.rank + 1)),
            scope=event.scope,
        )
        credited = True

    # Edges to the grounded subset only, and no grounded result means no
    # anchor: an anchor with no honest target would answer its question with
    # whatever the event happened to deliver.
    anchor_ids = (
        feedback._reinforce_query_anchors(store, [(event, tuple(grounded_ids))])
        if grounded_ids
        else []
    )

    closed = bool(event.feedback_applied)
    closed_by_attestation = False
    feedback_trace_id = event.feedback_trace_id
    if trace_node is not None and not closed:
        closure = store.mark_recall_event_feedback(event.id, trace_node.id)
        closed = True
        closed_by_attestation = True
        feedback_trace_id = closure.feedback_trace_id

    record = store.complete_recall_attestation(
        claimed.id,
        credited=credited,
        closed=closed,
        closed_by_attestation=closed_by_attestation,
        anchor_ids=anchor_ids,
        feedback_trace_id=feedback_trace_id,
    )
    return _verdict(record, event, replay=False)


def _verdict(
    record: RecallAttestation, event: RecallEvent, *, replay: bool
) -> dict[str, Any]:
    """The wire shape, built from the ledger row in both cases.

    A replay reports the recorded verdict verbatim — including ``credited`` and
    ``closed_by_attestation``, which describe what the *recorded* attestation
    did, not what this call did. ``replay: true`` is the field that says this
    call applied nothing.
    """

    return {
        "attestation_id": record.id,
        "recall_event_id": record.recall_event_id,
        "scope": event.scope,
        "evidence_sha256": record.evidence_sha256,
        "evidence_items": record.evidence_items,
        "evidence_chars": record.evidence_chars,
        "min_containment": record.min_containment,
        "results": [dict(entry) for entry in record.containments],
        "grounded_node_ids": list(record.grounded_node_ids),
        "credited": record.credited,
        "anchor_ids": list(record.anchor_ids),
        "closed": record.closed,
        "closed_by_attestation": record.closed_by_attestation,
        "feedback_trace_id": record.feedback_trace_id,
        "replay": replay,
    }
