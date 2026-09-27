"""Feedback updates for retrieval ranking and learned weights."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import logging
import math
import os
from types import SimpleNamespace
from typing import Any

from living_memory.grounding import DEFAULT_MIN_CONTAINMENT, ground_results
from living_memory.irrelevance import cancel_query_irrelevance, record_query_irrelevance
from living_memory.models import Node, RecallEvent, RetrievalWeights
from living_memory import storage as storage_module
from living_memory.query_anchors import upsert_anchor
from living_memory.storage import ChunkEmbedder, MemoryStore
from living_memory.temporal import parse_timestamp

_LOG = logging.getLogger(__name__)


# Credit assignment for implicit consumption feedback.
#
# When a memory_remember consumes a pending recall event, which of that
# event's results actually earned the reinforcement? The historical answer
# was "all of them": the loop below walked every result and handed each a
# rank-decayed positive signal into both node usefulness and the per-scope
# retrieval weights. That made the signal vacuous — an agent that used one
# result of ten reinforced all ten, and nine nerfed deliveries entered the
# learned weights as evidence *for* the channel that surfaced them.
#
# Credit now follows content grounding (living_memory.grounding): a result is
# reinforced only when the IDF-weighted share of its content tokens appearing
# in the consuming trace reaches ``min_containment``. This is the same measure
# the offline replay harness has used as its confirmed-useful label since the
# credit-assignment work, promoted from measurement to mechanism.
#
# Policies (``LM_RECALL_CREDIT_POLICY``):
#
# * ``grounded`` — reinforce grounded results, leave ungrounded ones neutral.
# * ``grounded_negative`` — additionally blame ungrounded results at
#   ``UNGROUNDED_NEGATIVE_FACTOR`` of the positive signal.
# * ``all`` — the pre-grounding rule, reinforce every delivered result. Kept
#   reachable so the regression test can exhibit the old behaviour and so an
#   operator can fall back without a code change.
#
# The default is the arm chosen by the replay A/B over the recorded history;
# see artifacts/grounding/credit-ab.md for the numbers.
#
# A second usage signal, the lookup (goal feedback-usage-signal). Grounding
# sees only what the closing trace *quotes*, and on the 2026-09-07 snapshots
# that is 19.0% of closures (sfx 14.2%): traces grew threefold, the remember
# protocol forbids copies, so the agent writes a new adjacent fact and the
# delivered node it built on goes uncredited. A ``memory_lookup`` id-fetch of
# a delivered node in the same transport session is the other honest "I used
# this" the server can see — naming an exact ULID is something only a reader
# who was handed it does — and it names a delivered node in 26.2% of closures,
# overlapping the grounded set in only 58 of 395 pairs. ``apply_lookup_credit``
# turns it into credit with the *same* assignment as a grounded result: the
# event's recorded per-channel scores, ``signal = max(0.2, 1/(rank+1))``,
# ``scope = event.scope``, and one anchor edge to the looked-up node only.
#
# The two signals share one ledger (``recall_credit_ledger`` in storage):
# whichever arrives first for a (event, node) pair claims the row and applies
# the credit; the other finds the row taken and applies nothing, so credit is
# at most once per pair. Linkage — related edges, provenance, closing the
# event — stays with the grounded path exactly as before; the ledger gates
# reinforcement only. A lookup never touches a node that was not requested
# and never credits an event that did not deliver the node, and with no
# transport identity on either side there is no join and no credit.
#
# Policies (``LM_LOOKUP_CREDIT_POLICY``):
#
# * ``delivered`` — credit as above.
# * ``off`` — record the lookup event only, exactly the pre-ledger behaviour.
#
# ``LM_LOOKUP_CREDIT_WINDOW_SECONDS`` bounds how far back a lookup can reach
# for the delivering event (default 24 h, the delivery-history horizon); an
# unknown policy or an unparsable window falls back to the default, like
# ``LM_RECALL_CREDIT_POLICY``. See artifacts/grounding/lookup-credit.md for
# the replay numbers.
RECALL_CREDIT_POLICIES: tuple[str, ...] = ("grounded", "grounded_negative", "all")
DEFAULT_RECALL_CREDIT_POLICY = "grounded"

LOOKUP_CREDIT_POLICIES: tuple[str, ...] = ("delivered", "off")
DEFAULT_LOOKUP_CREDIT_POLICY = "delivered"
DEFAULT_LOOKUP_CREDIT_WINDOW_SECONDS = 86400
#: Candidate events per lookup: the newest this many events of the transport
#: session, the same horizon ``MemoryStore.delivered_node_ids`` bounds the
#: session-delivery dedup with.
_LOOKUP_CREDIT_EVENT_LIMIT = 200

# Explicit recall feedback (goal explicit-recall-feedback).
#
# Grounding and lookup are both inferred; an explicit mark is the agent
# saying it. memory_recall, memory_remember and memory_teach accept optional
# ``used`` and ``irrelevant`` lists of node ids from earlier recalls on the
# same transport session. A mark is accepted only for an id that a recall
# event on that transport delivered — the newest delivering event within
# the lookup-credit window and event limit, exactly the join
# ``apply_lookup_credit`` does — and every mark, accepted or dropped, is
# written to the ``recall_feedback_marks`` audit table. Contract and the
# reasoning behind each choice: docs/explicit-feedback.md.
#
# Policies (``LM_EXPLICIT_FEEDBACK_POLICY``):
#
# * ``audit`` (default) — record and audit marks; no reinforcement, no
#   credit claimed, so grounded and lookup credit are untouched. Irrelevant
#   marks still drive link hygiene.
# * ``credit`` — additionally turn each accepted ``used`` mark into explicit
#   credit (basis ``explicit``, once per (event, node) across all three
#   bases) with the grounded assignment scaled by
#   ``LM_EXPLICIT_CREDIT_WEIGHT``, and hand accepted ``irrelevant`` marks to
#   ``_apply_query_irrelevance``.
# * ``off`` — accept the fields, ignore them, record nothing, no hygiene.
#
# An unknown policy or an unparsable weight falls back to the default, like
# ``LM_RECALL_CREDIT_POLICY``.
#
# ``LM_IMPLICIT_LINK_POLICY`` decides which delivered results the closing
# trace gets a ``related`` edge to: ``all`` (default, today's behaviour minus
# irrelevant-marked results) or ``credited`` (only results credited for that
# event under any basis).
EXPLICIT_FEEDBACK_POLICIES: tuple[str, ...] = ("audit", "credit", "off")
DEFAULT_EXPLICIT_FEEDBACK_POLICY = "audit"
DEFAULT_EXPLICIT_CREDIT_WEIGHT = 1.0
IMPLICIT_LINK_POLICIES: tuple[str, ...] = ("all", "credited")
DEFAULT_IMPLICIT_LINK_POLICY = "all"
EXPLICIT_MARK_KINDS: tuple[str, ...] = ("used", "irrelevant")
#: Why a mark was dropped, in precedence order.
EXPLICIT_MARK_REJECT_REASONS: tuple[str, ...] = (
    "empty",
    "duplicate",
    "conflict",
    "no_transport",
    "not_delivered",
)

# Blame multiplier for a delivered-but-ungrounded result under the
# ``grounded_negative`` policy, relative to the positive signal a grounded
# result at the same rank would earn.
UNGROUNDED_NEGATIVE_FACTOR = 0.25

# Containment threshold for the live grounding gate: the shared
# ``grounding.DEFAULT_MIN_CONTAINMENT`` (0.22 since the September 2026
# recalibration on the closed recall events of the 2026-09-07 snapshots,
# artifacts/grounding/recalibration-2026-09.md; ``LM_GROUNDING_MIN_CONTAINMENT``
# overrides it at import). The June value 0.25 was the point where per-event
# and whole-corpus IDF agreed on 96.2% of 72,023 recorded pairs
# (scripts/grounding_calibration.py); the recalibration replaced that
# agreement argument with a signal/noise one against encoder relatedness.
RECALL_CREDIT_MIN_CONTAINMENT = DEFAULT_MIN_CONTAINMENT


# Query anchors on the live path.
#
# The same grounding verdict that decides credit also decides what the graph
# learns about the *question*. A grounded consumption is the only moment where
# both halves of "this query was answered by these nodes" are in hand at once,
# so that is where the anchor is written: the consumed event's query becomes an
# anchor in the consumed event's scope, with edges to exactly the nodes the
# trace grounded on (living_memory.query_anchors owns dedup, edge accumulation,
# and replacement-following; this module only decides *when* and *with what*).
#
# Four properties are load-bearing and each one is pinned by
# tests/test_anchor_live_path.py:
#
# * Scope is ``event.scope``, never ``trace.scope``. The two diverge whenever a
#   transport-session match closes feedback across the recall/ingest scope
#   divergence, and an anchor written under the trace's scope would answer a
#   question that was never asked there.
# * Edges point at the grounded subset only. Reinforcing the whole delivered
#   set is the defect the grounding work removed from credit assignment -- 87.9%
#   of reinforcements (63,325 of 72,023 recorded pairs) went to results the
#   trace never used -- and routing it back in through anchor edges would
#   rebuild it in the graph, where it would then be *retrieved*.
# * No grounded result, no anchor. Under ``LM_RECALL_CREDIT_POLICY=all`` no
#   verdict is computed at all, so that policy writes no anchors rather than
#   anchoring everything delivered; likewise ``reinforce_results=False``.
#   Falling back to the pre-grounding credit rule must not poison the graph.
# * Cost is one short vector per grounded event, batched into a single call to
#   the encoder the store already holds. Grounding is not recomputed and the
#   result contents are never re-embedded.
#
# An anchor is derived data. A failure to write one is logged and swallowed:
# the trace, its provenance, and its credit are the user's write and must not
# be lost to a stale encoder or a pre-v7 database.

_ADAPTIVE_LR_LADDER: tuple[tuple[int, float], ...] = (
    (10, 0.20),
    (50, 0.10),
    (500, 0.05),
)
_ADAPTIVE_LR_FLOOR = 0.02
_DEFAULT_PENDING_RECALL_LIMIT = 5

# Ranking feedback multipliers. Confidence is neutral at the storage-default
# base (0.5): a fresh, unvalidated node must rank on retrieval relevance
# alone rather than start with an implicit penalty, while confidence above or
# below the base still moves the boost monotonically around 1.0.
_BASE_CONFIDENCE = 0.5
_CONFIDENCE_SLOPE = 0.65
# Upper bound for the compounded confidence x usefulness x access multiplier.
# Uncapped, entrenched nodes (up to 1.325 x 1.55 x 1.25 ~= 2.57) bury fresh
# exact matches: under learned vector-dominant weights (bm25 0.10 / vector
# 0.85) a just-written trace with an exact lexical hit scores ~0.48 while a
# weak 0.38 vector match scores ~0.32 — any cap above ~1.45 lets the
# entrenched node win, so 1.4 keeps fresh writes recallable with margin while
# still rewarding validated, useful, frequently accessed knowledge.
FEEDBACK_MULTIPLIER_CAP = 1.4

# Diminishing returns for positive usefulness reinforcement.
#
# The cap above bounds how hard feedback can boost a score, but not how fast
# a node accumulates the usefulness feeding that boost. Every delivered
# recall result is implicitly reinforced by the next compatible remember
# (apply_pending_recall_feedback), so delivery itself breeds rank advantage:
# boosted nodes get delivered more, deliveries reinforce them further, and a
# handful of saturated nodes end up collecting most deliveries (measured
# 2026-07-31: ~76% of deliveries went to a few usefulness=1.0 nodes). To
# damp that loop at its source, a positive usefulness increment is scaled by
#
#     max(HEADROOM_FLOOR, 1 - usefulness) / (1 + ACCESS_DAMPING * ln(1 + accesses))
#
# so a node near saturation, or one that has already been delivered many
# times, needs disproportionately more fresh evidence per unit of further
# boost than an unproven node (an entrenched node at usefulness 1.0 with
# ~600 accesses gains ~30x slower than a fresh one). The headroom factor is
# floored, never zeroed, so usefulness 1.0 stays exactly reachable via the
# clamp; a node corrected into negative usefulness recovers with full
# headroom (factor 1.0), though the access divisor still applies — a
# heavily-delivered node re-earns its boost slowly no matter where it
# starts. Explicit negative feedback is exempt: corrections always apply at
# full strength. The cap's semantics are unchanged — it still bounds the
# compounded boost at 1.4 protecting fresh exact-match writes; these knobs
# only slow how fast entrenchment approaches that ceiling. Neutral values
# (floor 1.0, damping 0.0) reproduce the legacy flat 0.1 * signal increment
# exactly; tests/test_feedback_delivery_concentration.py runs a closed-loop
# simulation under both settings and pins the concentration reduction.
USEFULNESS_GAIN_HEADROOM_FLOOR = 0.25
USEFULNESS_GAIN_ACCESS_DAMPING = 1.0


def _retrieval_tuning_policy() -> str:
    return os.environ.get("LM_RETRIEVAL_TUNING_POLICY", "fixed").strip().lower()


def _recall_credit_policy() -> str:
    """Active credit policy; an unknown value falls back to the default."""

    policy = os.environ.get("LM_RECALL_CREDIT_POLICY", "").strip().lower()
    if policy in RECALL_CREDIT_POLICIES:
        return policy
    return DEFAULT_RECALL_CREDIT_POLICY


def _lookup_credit_policy() -> str:
    """Active lookup-credit policy; an unknown value falls back to the default."""

    policy = os.environ.get("LM_LOOKUP_CREDIT_POLICY", "").strip().lower()
    if policy in LOOKUP_CREDIT_POLICIES:
        return policy
    return DEFAULT_LOOKUP_CREDIT_POLICY


def _lookup_credit_window() -> timedelta:
    """How far back a lookup may reach for its delivering event.

    ``LM_LOOKUP_CREDIT_WINDOW_SECONDS``; anything unparsable or not positive
    falls back to the default rather than widening or closing the window.
    """

    raw = os.environ.get("LM_LOOKUP_CREDIT_WINDOW_SECONDS", "").strip()
    seconds = DEFAULT_LOOKUP_CREDIT_WINDOW_SECONDS
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 0
        if parsed > 0:
            seconds = parsed
    return timedelta(seconds=seconds)


def _explicit_feedback_policy() -> str:
    """Active explicit-feedback policy; an unknown value falls back to the default."""

    policy = os.environ.get("LM_EXPLICIT_FEEDBACK_POLICY", "").strip().lower()
    if policy in EXPLICIT_FEEDBACK_POLICIES:
        return policy
    return DEFAULT_EXPLICIT_FEEDBACK_POLICY


def _explicit_credit_weight() -> float:
    """Multiplier on the grounded signal for explicit credit.

    ``LM_EXPLICIT_CREDIT_WEIGHT``; anything unparsable, negative or not
    finite falls back to the default. Zero is honoured: the pair is claimed
    and nothing moves.
    """

    raw = os.environ.get("LM_EXPLICIT_CREDIT_WEIGHT", "").strip()
    if not raw:
        return DEFAULT_EXPLICIT_CREDIT_WEIGHT
    try:
        parsed = float(raw)
    except ValueError:
        return DEFAULT_EXPLICIT_CREDIT_WEIGHT
    if not math.isfinite(parsed) or parsed < 0.0:
        return DEFAULT_EXPLICIT_CREDIT_WEIGHT
    return parsed


def _implicit_link_policy() -> str:
    """Active implicit-link policy; an unknown value falls back to the default."""

    policy = os.environ.get("LM_IMPLICIT_LINK_POLICY", "").strip().lower()
    if policy in IMPLICIT_LINK_POLICIES:
        return policy
    return DEFAULT_IMPLICIT_LINK_POLICY


def ungrounded_negative_signals(result: Any, signal: float) -> dict[str, float]:
    """Damped per-channel blame for a delivered result the trace never used.

    Same proportional attribution as a positive signal, scaled down by
    ``UNGROUNDED_NEGATIVE_FACTOR`` and inverted. Shared with the replay
    harness's ``grounded_negative`` credit rule so the A/B arm and the live
    policy cannot drift apart.
    """

    return _method_signals(result, -UNGROUNDED_NEGATIVE_FACTOR * abs(float(signal)))


def _adaptive_learning_rate(trace_count: int) -> float:
    """Effective learning rate for a scope of the given size.

    Young scopes converge fast on a few feedback signals; mature scopes
    move slowly to avoid oscillation around a settled policy.
    """

    for ceiling, rate in _ADAPTIVE_LR_LADDER:
        if trace_count < ceiling:
            return rate
    return _ADAPTIVE_LR_FLOOR


def _maybe_retune_learning_rate(store: MemoryStore, scope: str) -> None:
    if _retrieval_tuning_policy() != "adaptive":
        return
    row = store.connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM nodes
        WHERE level = 'trace' AND scope = ? AND decayed = 0
        """,
        (scope,),
    ).fetchone()
    target_lr = _adaptive_learning_rate(int(row["count"]) if row else 0)
    current = store.get_retrieval_weights(scope)
    if abs(current.learning_rate - target_lr) < 1e-6:
        return
    store.set_retrieval_weights(
        scope,
        bm25=current.bm25,
        vector=current.vector,
        graph=current.graph,
        learning_rate=target_lr,
    )


@dataclass(frozen=True, slots=True)
class FeedbackUpdate:
    """Result of applying feedback to a recall result."""

    node: Node | None
    weights: RetrievalWeights


@dataclass(frozen=True, slots=True)
class ImplicitRecallFeedback:
    """Result of connecting a new ingest trace to prior recall events."""

    trace: Node
    events: list[RecallEvent]
    linked_node_ids: list[str]
    feedback_applied: bool
    #: Results this consumption reinforced: those whose content the consuming
    #: trace actually grounded (every delivered result under policy ``all``),
    #: minus pairs whose credit a same-session lookup or explicit mark had
    #: already claimed. A subset of ``linked_node_ids`` except for a result
    #: the agent marked irrelevant for its event, which is credited on its
    #: grounding but never linked. Empty when reinforcement was skipped.
    grounded_node_ids: list[str] = field(default_factory=list)
    #: Query anchors created or reinforced by this consumption, deduplicated —
    #: one per consumed event that grounded at least one result. Empty when
    #: nothing grounded, and also when the anchor write failed, which is
    #: logged and never raised.
    anchor_ids: list[str] = field(default_factory=list)


class FeedbackService:
    """Apply explicit or implicit feedback to nodes and retrieval weights."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def apply(
        self,
        result: Any,
        *,
        useful: bool = True,
        signal: float = 1.0,
        scope: str | None = None,
    ) -> FeedbackUpdate:
        return apply_retrieval_feedback(
            self.store,
            result,
            useful=useful,
            signal=signal,
            scope=scope,
        )


def apply_retrieval_feedback(
    store: MemoryStore,
    result: Any,
    *,
    useful: bool = True,
    signal: float = 1.0,
    scope: str | None = None,
) -> FeedbackUpdate:
    """Update node usefulness and nudge weights by each method's evidence share."""

    magnitude = max(0.0, min(abs(float(signal)), 5.0))
    signed_signal = magnitude if useful else -magnitude

    node = _result_node(result, store)
    if node is not None:
        increment = 0.1 * signed_signal
        if signed_signal > 0.0:
            increment *= _positive_reinforcement_gain(node)
        updated_usefulness = max(-1.0, min(1.0, node.usefulness_score + increment))
        node = store.update_node(node.id, stats={"usefulness_score": updated_usefulness})

    target_scope = scope or (node.scope if node is not None else "global")
    _maybe_retune_learning_rate(store, target_scope)
    method_signals = _method_signals(result, signed_signal)
    weights = store.update_retrieval_weights(
        target_scope,
        bm25_signal=method_signals["bm25"],
        vector_signal=method_signals["vector"],
        graph_signal=method_signals["graph"],
    )
    return FeedbackUpdate(node=node, weights=weights)


def apply_pending_recall_feedback(
    store: MemoryStore,
    trace: Node,
    *,
    limit: int = _DEFAULT_PENDING_RECALL_LIMIT,
    reinforce_results: bool = True,
) -> ImplicitRecallFeedback:
    """Attach recent recall provenance to a new trace and reinforce what it used.

    Provenance is exhaustive: every resolvable result of every consumed event
    lands in ``recalled_nodes``/``source_traces``, because provenance must
    record what was *shown*. The rank-weighted ``related`` edge is not
    quite: a result with an accepted explicit ``irrelevant`` mark for that
    event gets no edge (link hygiene, every ``LM_EXPLICIT_FEEDBACK_POLICY``
    but ``off``), and under ``LM_IMPLICIT_LINK_POLICY=credited`` only results
    credited for the event under any basis get one. Reinforcement is gated
    further: under the grounding policies only results the trace
    demonstrably used move node usefulness and retrieval weights. See the
    credit-assignment note at the top of this module.

    The same grounded subset also becomes a query anchor per consumed event —
    the graph's entry from query space. See the anchor note above it.
    """

    events = store.pending_recall_events(
        scope=trace.scope,
        context=trace.context,
        content=trace.content,
        limit=limit,
    )
    if not events:
        return ImplicitRecallFeedback(
            trace=trace,
            events=[],
            linked_node_ids=[],
            feedback_applied=False,
        )

    provenance = dict(trace.provenance)
    prior_recalls = list(provenance.get("prior_recalls", []))
    recalled_nodes = list(provenance.get("recalled_nodes", []))
    source_traces = list(trace.source_traces)
    linked_node_ids: list[str] = []
    grounded_node_ids: list[str] = []
    anchor_work: list[tuple[RecallEvent, tuple[str, ...]]] = []
    feedback_applied = False
    policy = _recall_credit_policy()
    link_policy = _implicit_link_policy()
    link_hygiene = _explicit_feedback_policy() != "off"

    for event in events:
        event_node_ids: list[str] = []
        # Per event, not accumulated across the batch: an anchor's edges may
        # only carry the results *its own* query earned.
        event_grounded: list[str] = []
        # Pairs already credited under any basis (a same-session lookup, an
        # explicit ``used`` mark). Read once per event, and only where it can
        # matter: under ``grounded_negative`` an ungrounded-but-credited
        # result is not blamed, because the ledger says it was used; under
        # link policy ``credited`` it decides which results get an edge.
        # Grounded results claim their own row below.
        already_credited: set[str] = (
            store.credited_node_ids(event.id)
            if (reinforce_results and policy == "grounded_negative")
            or link_policy == "credited"
            else set()
        )
        # Results the agent explicitly marked irrelevant to this event's
        # query get no edge from the closing trace (link hygiene; see
        # apply_explicit_marks). Provenance still records them as shown.
        irrelevant = store.irrelevant_marked_node_ids(event.id) if link_hygiene else set()
        # Resolve every result once: linkage needs the node, and grounding
        # needs its content. One get_node per result, as before.
        resolved: list[tuple[int, dict[str, Any], Node]] = []
        for rank, result in enumerate(event.results):
            node_id = str(result.get("node_id") or "")
            if not node_id or node_id == trace.id:
                continue
            node = store.get_node(node_id)
            if node is None:
                continue
            resolved.append((rank, result, node))

        # Grade the whole event at once: the IDF index is shared by its
        # results, and an ungraded event costs no tokenization at all.
        verdicts = {}
        if reinforce_results and policy != "all" and resolved:
            verdicts = ground_results(
                trace.content,
                {node.id: node.content for _rank, _result, node in resolved},
                min_containment=RECALL_CREDIT_MIN_CONTAINMENT,
            )

        for rank, result, node in resolved:
            node_id = node.id
            event_node_ids.append(node_id)
            if node_id not in recalled_nodes:
                recalled_nodes.append(node_id)
            if node.level == "trace" and node_id not in source_traces:
                source_traces.append(node_id)

            claimed = False
            if reinforce_results:
                verdict = verdicts.get(node_id)
                content_grounded = verdict is not None and verdict.grounded
                grounded = policy == "all" or content_grounded
                apply = True
                if grounded:
                    # Claim the pair before crediting it. False means a
                    # lookup or explicit mark already spent this credit: no
                    # second reinforcement, no second anchor edge.
                    claimed = store.claim_recall_credit(
                        event.id, node_id, basis="grounded", source_id=trace.id
                    )
                    apply = claimed
                    if claimed:
                        if content_grounded:
                            event_grounded.append(node_id)
                        if node_id not in grounded_node_ids:
                            grounded_node_ids.append(node_id)
                elif policy != "grounded_negative" or node_id in already_credited:
                    apply = False

                if apply:
                    synthetic_result = SimpleNamespace(
                        node_id=node_id,
                        bm25_score=float(result.get("bm25_score", 0.0) or 0.0),
                        vector_score=float(result.get("vector_score", 0.0) or 0.0),
                        graph_score=float(result.get("graph_score", 0.0) or 0.0),
                    )
                    signal = max(0.2, 1.0 / (rank + 1))
                    apply_retrieval_feedback(
                        store,
                        synthetic_result,
                        useful=grounded,
                        signal=signal if grounded else UNGROUNDED_NEGATIVE_FACTOR * signal,
                        scope=event.scope,
                    )
                    feedback_applied = True

            if node_id in irrelevant:
                continue
            if link_policy == "credited" and not (claimed or node_id in already_credited):
                continue
            weight = _implicit_connection_weight(rank)
            store.create_connection(
                trace.id,
                node_id,
                "related",
                weight=weight,
                metadata={
                    "basis": "implicit_recall_feedback",
                    "recall_event_id": event.id,
                    "recall_query": event.query,
                    "rank": rank + 1,
                },
            )
            if node_id not in linked_node_ids:
                linked_node_ids.append(node_id)

        if event_grounded:
            anchor_work.append((event, tuple(event_grounded)))

        prior_recalls.append(
            {
                "id": event.id,
                "query": event.query,
                "scope": event.scope,
                "timestamp": event.created_at,
                "result_ids": event_node_ids,
            }
        )
        store.mark_recall_event_feedback(event.id, trace.id)

    provenance["prior_recalls"] = prior_recalls
    provenance["recalled_nodes"] = recalled_nodes
    provenance["source_traces"] = source_traces
    updated = store.update_node(trace.id, provenance=provenance)
    # After the trace is durable, so a derived write can never cost the write
    # it was derived from.
    anchor_ids = _reinforce_query_anchors(store, anchor_work)
    return ImplicitRecallFeedback(
        trace=updated,
        events=events,
        linked_node_ids=linked_node_ids,
        feedback_applied=feedback_applied,
        grounded_node_ids=grounded_node_ids,
        anchor_ids=anchor_ids,
    )


@dataclass(frozen=True, slots=True)
class LookupCredit:
    """What one ``memory_lookup`` id-fetch was credited for."""

    lookup_event_id: str
    #: ``(recall_event_id, node_id)`` pairs this lookup claimed and reinforced,
    #: in request order. Empty under policy ``off``, without a transport
    #: identity, and when every requested node was undelivered, out of the
    #: window, or already credited.
    credited: list[tuple[str, str]] = field(default_factory=list)
    #: Query anchors created or reinforced, deduplicated; empty when nothing
    #: was credited or the anchor write failed (logged, never raised).
    anchor_ids: list[str] = field(default_factory=list)


def apply_lookup_credit(
    store: MemoryStore,
    lookup_event_id: str,
    node_ids: Iterable[str],
    transport_session_id: str | None,
    occurred_at: str | datetime,
) -> LookupCredit:
    """Credit a same-session id-fetch of delivered nodes as usage. Never raises.

    For each requested node id, the delivering event is the *newest*
    ``recall_events`` row stamped with the same ``transport_session_id`` whose
    ``created_at`` is not later than the lookup and not older than the window
    (``_lookup_credit_window``) and whose recorded results name that node.
    Instants are compared as parsed datetimes at second precision — recall
    events are stamped to the second, lookups carry microseconds, and a
    lookup in the same second as the delivery is after it, not before.

    A hit claims the ledger row (``basis='lookup'``); a pair already credited
    by an earlier grounding or lookup is skipped. A new claim is credited
    exactly as a grounded result is: ``apply_retrieval_feedback`` with the
    event's recorded per-channel scores for that result, ``useful=True``,
    ``signal = max(0.2, 1/(rank+1))`` at the result's delivered rank, in the
    event's scope; then one anchor per credited event with edges to the
    looked-up nodes only.

    Policy ``off`` or a missing transport identity returns before any read.
    Any failure is logged and swallowed, as an anchor write's is: a credit is
    derived from the lookup and must never fail it. Because the row is
    claimed before the credit is applied, a failure between the two leaves
    the pair claimed and uncredited — under-crediting, the conservative
    direction (the attestation ledger's rule).
    """

    outcome = LookupCredit(lookup_event_id=str(lookup_event_id))
    if _lookup_credit_policy() == "off" or not transport_session_id:
        return outcome
    requested: list[str] = []
    for candidate in node_ids:
        text = str(candidate or "").strip()
        if text and text not in requested:
            requested.append(text)
    if not requested:
        return outcome

    try:
        lookup_at = _lookup_instant(occurred_at)
        if lookup_at is None:
            _LOG.warning(
                "lookup credit skipped for %s: unparsable instant %r",
                lookup_event_id,
                occurred_at,
            )
            return outcome
        earliest = lookup_at - _lookup_credit_window()

        # Newest first, so the first event naming a node is its newest
        # delivery; each node is credited against at most one event.
        hits: list[tuple[RecallEvent, int, dict[str, Any], str]] = []
        pending = list(requested)
        for event in store.recall_events_for_transport(
            str(transport_session_id), limit=_LOOKUP_CREDIT_EVENT_LIMIT
        ):
            if not pending:
                break
            created_at = _lookup_instant(event.created_at)
            if created_at is None or created_at > lookup_at or created_at < earliest:
                continue
            for rank, result in enumerate(event.results):
                node_id = str(result.get("node_id") or "")
                if node_id in pending:
                    pending.remove(node_id)
                    hits.append((event, rank, result, node_id))

        anchor_work: dict[str, tuple[RecallEvent, list[str]]] = {}
        for event, rank, result, node_id in hits:
            if not store.claim_recall_credit(
                event.id, node_id, basis="lookup", source_id=str(lookup_event_id)
            ):
                continue
            apply_retrieval_feedback(
                store,
                SimpleNamespace(
                    node_id=node_id,
                    bm25_score=float(result.get("bm25_score", 0.0) or 0.0),
                    vector_score=float(result.get("vector_score", 0.0) or 0.0),
                    graph_score=float(result.get("graph_score", 0.0) or 0.0),
                ),
                useful=True,
                signal=max(0.2, 1.0 / (rank + 1)),
                scope=event.scope,
            )
            outcome.credited.append((event.id, node_id))
            anchor_work.setdefault(event.id, (event, []))[1].append(node_id)

        if anchor_work:
            outcome.anchor_ids.extend(
                _reinforce_query_anchors(
                    store,
                    [(event, tuple(targets)) for event, targets in anchor_work.values()],
                )
            )
    except Exception:
        _LOG.warning(
            "lookup credit failed for lookup event %s (%d requested id(s))",
            lookup_event_id,
            len(requested),
            exc_info=True,
        )
    return outcome


@dataclass(slots=True)
class ExplicitMark:
    """One ``used``/``irrelevant`` mark after resolution against deliveries."""

    node_id: str
    mark: str
    accepted: bool
    reject_reason: str | None = None
    #: The delivering event (newest same-transport event within the window
    #: naming the node); None for a dropped mark.
    event: RecallEvent | None = None
    #: 0-based rank of the node in ``event.results``.
    rank: int | None = None
    result: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExplicitMarks:
    """What one call's ``used``/``irrelevant`` fields did."""

    policy: str
    marks: list[ExplicitMark] = field(default_factory=list)
    #: ``recall_feedback_marks`` row ids, parallel to ``marks``.
    mark_ids: list[int] = field(default_factory=list)
    #: ``(recall_event_id, node_id)`` pairs explicit credit was claimed for.
    credited: list[tuple[str, str]] = field(default_factory=list)
    anchor_ids: list[str] = field(default_factory=list)
    #: ``related`` edges removed by link hygiene for later irrelevant marks.
    unlinked: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """The compact ``feedback_marks`` block of a tool response."""

        by_reason: dict[str, int] = {}
        for mark in self.marks:
            if not mark.accepted and mark.reject_reason:
                by_reason[mark.reject_reason] = by_reason.get(mark.reject_reason, 0) + 1
        accepted = sum(1 for mark in self.marks if mark.accepted)
        summary: dict[str, Any] = {
            "accepted": accepted,
            "dropped": len(self.marks) - accepted,
            "by_reason": by_reason,
        }
        if self.policy == "off":
            summary["ignored"] = True
        return summary


def has_explicit_marks(used: Sequence[Any] | None, irrelevant: Sequence[Any] | None) -> bool:
    """Whether a call passed either field at all (an empty list counts)."""

    return used is not None or irrelevant is not None


def resolve_explicit_marks(
    store: MemoryStore,
    used: Sequence[Any] | None,
    irrelevant: Sequence[Any] | None,
    transport_session_id: str | None,
    occurred_at: str | datetime | None = None,
) -> list[ExplicitMark]:
    """Accept or drop each marked id against this transport's deliveries.

    Read-only. Reasons, in precedence order: ``empty`` (not a non-blank
    string), ``duplicate`` (repeated within its list; the first occurrence
    stands), ``conflict`` (named in both lists; every occurrence dropped),
    ``no_transport`` (the call carries no transport identity, so there is no
    session to join), ``not_delivered`` (no recall event of this transport
    within the lookup-credit window and event limit delivered it). An
    accepted mark resolves to the *newest* delivering event, the
    ``apply_lookup_credit`` join.
    """

    raw: list[tuple[str, Any]] = [("used", item) for item in (used or [])]
    raw.extend(("irrelevant", item) for item in (irrelevant or []))
    texts: list[tuple[str, str]] = [
        (kind, item.strip() if isinstance(item, str) else "") for kind, item in raw
    ]
    used_ids = {text for kind, text in texts if kind == "used" and text}
    irrelevant_ids = {text for kind, text in texts if kind == "irrelevant" and text}
    conflicts = used_ids & irrelevant_ids

    marks: list[ExplicitMark] = []
    seen: set[tuple[str, str]] = set()
    for kind, text in texts:
        reason: str | None = None
        if not text:
            reason = "empty"
        elif (kind, text) in seen:
            reason = "duplicate"
        elif text in conflicts:
            reason = "conflict"
        elif not transport_session_id:
            reason = "no_transport"
        if text:
            seen.add((kind, text))
        marks.append(
            ExplicitMark(node_id=text, mark=kind, accepted=False, reject_reason=reason)
        )

    pending = {mark.node_id for mark in marks if mark.reject_reason is None}
    hits: dict[str, tuple[RecallEvent, int, dict[str, Any]]] = {}
    if pending:
        instant = _lookup_instant(
            storage_module._utc_now() if occurred_at is None else occurred_at
        )
        if instant is not None:
            earliest = instant - _lookup_credit_window()
            for event in store.recall_events_for_transport(
                str(transport_session_id), limit=_LOOKUP_CREDIT_EVENT_LIMIT
            ):
                if not pending:
                    break
                created_at = _lookup_instant(event.created_at)
                if created_at is None or created_at > instant or created_at < earliest:
                    continue
                for rank, result in enumerate(event.results):
                    node_id = str(result.get("node_id") or "")
                    if node_id in pending:
                        pending.discard(node_id)
                        hits[node_id] = (event, rank, result)

    for mark in marks:
        if mark.reject_reason is not None:
            continue
        hit = hits.get(mark.node_id)
        if hit is None:
            mark.reject_reason = "not_delivered"
            continue
        mark.accepted = True
        mark.event, mark.rank, mark.result = hit[0], hit[1], dict(hit[2])
    return marks


def apply_explicit_marks(
    store: MemoryStore,
    used: Sequence[Any] | None,
    irrelevant: Sequence[Any] | None,
    *,
    via_tool: str,
    source_id: str,
    transport_session_id: str | None,
    agent: str | None = None,
    resolved: list[ExplicitMark] | None = None,
) -> ExplicitMarks:
    """Record, and under ``credit`` apply, one call's explicit marks. Never raises.

    Order: resolve (unless ``resolved`` is passed — memory_recall resolves
    before recording its own event, so its marks can only name *earlier*
    deliveries), audit every mark, unlink, credit. Unlinking is the
    irrelevant-link hygiene for the later-call case: when the delivering
    event is already closed and its closing trace carries an
    ``implicit_recall_feedback`` edge for this event to the marked node, the
    edge is deleted. Under ``credit`` each accepted ``used`` mark claims
    basis ``explicit`` (``source_id`` ``mark:<audit row id>``) and, when new,
    is reinforced exactly as a grounded result at its delivered rank with
    the signal scaled by ``LM_EXPLICIT_CREDIT_WEIGHT``, plus one anchor edge
    per event to the marked nodes; accepted ``irrelevant`` marks go to
    ``_apply_query_irrelevance`` per event. Global usefulness never moves
    down for an irrelevant mark.

    A failure is logged and swallowed: marks are derived from the call and
    must never fail the write or recall that carried them.
    """

    policy = _explicit_feedback_policy()
    outcome = ExplicitMarks(policy=policy)
    if policy == "off":
        return outcome
    try:
        marks = (
            resolved
            if resolved is not None
            else resolve_explicit_marks(store, used, irrelevant, transport_session_id)
        )
        outcome.marks = list(marks)
        if not marks:
            return outcome
        outcome.mark_ids = store.record_feedback_marks(
            [
                {
                    "recall_event_id": mark.event.id if mark.event is not None else "",
                    "node_id": mark.node_id,
                    "mark": mark.mark,
                    "accepted": mark.accepted,
                    "reject_reason": mark.reject_reason,
                    "via_tool": via_tool,
                    "source_id": source_id,
                    "transport_session_id": transport_session_id,
                    "agent": agent,
                    "rank": mark.rank,
                }
                for mark in marks
            ]
        )

        irrelevant_by_event: dict[str, tuple[RecallEvent, list[str]]] = {}
        for mark in marks:
            if mark.accepted and mark.mark == "irrelevant" and mark.event is not None:
                entry = irrelevant_by_event.setdefault(mark.event.id, (mark.event, []))
                if mark.node_id not in entry[1]:
                    entry[1].append(mark.node_id)
        for event, node_ids in irrelevant_by_event.values():
            outcome.unlinked.extend(_unlink_irrelevant(store, event, node_ids))

        if policy != "credit":
            return outcome

        weight = _explicit_credit_weight()
        anchor_work: dict[str, tuple[RecallEvent, list[str]]] = {}
        for mark, mark_id in zip(marks, outcome.mark_ids, strict=True):
            if not (mark.accepted and mark.mark == "used" and mark.event is not None):
                continue
            event = mark.event
            if not store.claim_recall_credit(
                event.id, mark.node_id, basis="explicit", source_id=f"mark:{mark_id}"
            ):
                continue
            outcome.credited.append((event.id, mark.node_id))
            rank = int(mark.rank or 0)
            signal = max(0.2, 1.0 / (rank + 1)) * weight
            if signal > 0.0:
                apply_retrieval_feedback(
                    store,
                    SimpleNamespace(
                        node_id=mark.node_id,
                        bm25_score=float(mark.result.get("bm25_score", 0.0) or 0.0),
                        vector_score=float(mark.result.get("vector_score", 0.0) or 0.0),
                        graph_score=float(mark.result.get("graph_score", 0.0) or 0.0),
                    ),
                    useful=True,
                    signal=signal,
                    scope=event.scope,
                )
                anchor_work.setdefault(event.id, (event, []))[1].append(mark.node_id)
        if anchor_work:
            outcome.anchor_ids.extend(
                _reinforce_query_anchors(
                    store,
                    [(event, tuple(targets)) for event, targets in anchor_work.values()],
                )
            )
        for event, node_ids in irrelevant_by_event.values():
            _apply_query_irrelevance(store, event, node_ids)
    except Exception:
        _LOG.warning(
            "explicit feedback marks failed for %s %s", via_tool, source_id, exc_info=True
        )
    return outcome


def _unlink_irrelevant(
    store: MemoryStore, event: RecallEvent, node_ids: Sequence[str]
) -> list[tuple[str, str]]:
    """Delete the closing trace's implicit edges for this event to ``node_ids``.

    Only an edge whose metadata says it is the ``implicit_recall_feedback``
    edge *of this event* is removed: the (source, target, type) key is unique,
    so an edge a later event or a typed-edge derivation rewrote is somebody
    else's evidence and stays. Re-read the event, since the caller's copy may
    predate its closure.
    """

    current = store.get_recall_event(event.id)
    trace_id = current.feedback_trace_id if current is not None else None
    if not trace_id:
        return []
    removed: list[tuple[str, str]] = []
    for node_id in node_ids:
        for connection in store.list_connections(
            source_id=trace_id, target_id=node_id, relation_type="related"
        ):
            metadata = connection.metadata or {}
            if (
                metadata.get("basis") == "implicit_recall_feedback"
                and metadata.get("recall_event_id") == event.id
            ):
                store.delete_connection(connection.id)
                removed.append((trace_id, node_id))
    return removed


def _apply_query_irrelevance(
    store: MemoryStore, event: RecallEvent, node_ids: Sequence[str]
) -> None:
    """Demote ``node_ids`` for ``event``'s query only. Never raises.

    Called under ``LM_EXPLICIT_FEEDBACK_POLICY=credit`` once per recall event
    with that event's accepted ``irrelevant`` marks (deduplicated, in mark
    order). Records one (query anchor, node) irrelevance row per node via
    ``living_memory.irrelevance``; retrieval demotes the node only for queries
    matching that anchor. It never lowers a node's global
    ``usefulness_score``, confidence or retrieval weights: an irrelevant mark
    says "not for this query", not "wrong". See docs/query-irrelevance.md.
    """

    try:
        record_query_irrelevance(store, event, node_ids, _anchor_embedder(store))
    except Exception:
        _LOG.warning(
            "query irrelevance write failed for event %s", event.id, exc_info=True
        )


def _lookup_instant(value: str | datetime | Any) -> datetime | None:
    """A UTC instant at second precision for the lookup-credit join."""

    if isinstance(value, datetime):
        parsed = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        parsed = parsed.astimezone(UTC)
    else:
        parsed = parse_timestamp(value)
    if parsed is None:
        return None
    return parsed.replace(microsecond=0)


def _reinforce_query_anchors(
    store: MemoryStore, work: Sequence[tuple[RecallEvent, tuple[str, ...]]]
) -> list[str]:
    """Create or reinforce one anchor per grounded consumption. Never raises.

    Every query is embedded in one batched call, before any anchor is written,
    so a consuming ``memory_remember`` pays one encoder round trip no matter
    how many pending events it closed. A blank query is skipped rather than
    anchored: its fingerprint would be the same for every blank query in the
    scope, so it would collect edges from unrelated consumptions.
    """

    items = [(event, targets) for event, targets in work if event.query.strip()]
    if not items:
        return []
    anchor_ids: list[str] = []
    try:
        embedder = _anchor_embedder(store)
        if embedder is None:
            return []
        vectors = embedder([event.query for event, _targets in items])
        for (event, targets), vector in zip(items, vectors, strict=True):
            if not vector:
                continue
            outcome = upsert_anchor(store, event.query, event.scope, vector, targets)
            # Positive credit for this question cancels an earlier
            # irrelevant mark on the same (anchor, node); see
            # living_memory.irrelevance.
            cancel_query_irrelevance(
                store,
                outcome.anchor.id,
                [*targets, *(edge.target_id for edge in outcome.edges)],
            )
            if outcome.anchor.id not in anchor_ids:
                anchor_ids.append(outcome.anchor.id)
    except Exception:
        _LOG.warning(
            "query anchor write failed for %d grounded consumption(s)",
            len(items),
            exc_info=True,
        )
    return anchor_ids


def _anchor_embedder(store: MemoryStore) -> ChunkEmbedder | None:
    """The encoder that puts an anchor vector in the same space as node chunks.

    Deliberately the store's *chunk* encoder rather than a model of this
    module's own. Retrieval matches an incoming query vector against anchor
    vectors and node chunks against the same query, so all three have to come
    out of one model; and the store already holds that model, lent to it by
    ``MemoryRecallService._share_embedder_with_store`` precisely so a second
    resident copy of the same weights never gets loaded.

    Returns ``None`` for a store face that has no such encoder — the
    ranking-only shim ``replay`` passes in — which writes no anchors, as an
    offline scoring run should not.
    """

    resolve = getattr(store, "_resolve_chunk_embedder", None)
    if resolve is None:
        return None
    return resolve()


def feedback_weighted_score(
    node: Node,
    base_score: float,
    *,
    superseded: bool = False,
    superseding: bool = False,
) -> float:
    """Apply confidence, usefulness, access, and correction feedback to a score."""

    confidence = max(0.0, min(node.confidence, 1.0))
    confidence_boost = 1.0 + _CONFIDENCE_SLOPE * (confidence - _BASE_CONFIDENCE)
    usefulness = max(-1.0, min(node.usefulness_score, 1.0))
    usefulness_boost = 1.0 + (0.55 * usefulness if usefulness >= 0.0 else 0.45 * usefulness)
    access_boost = 1.0 + min(0.25, 0.04 * _log1p(node.access_count))
    # Cap covers the compounded positive feedback only; correction signals
    # (superseding boost, superseded penalty) stay deliberate and uncapped.
    feedback_multiplier = min(
        confidence_boost * usefulness_boost * access_boost,
        FEEDBACK_MULTIPLIER_CAP,
    )
    correction_boost = 1.2 if superseding else 1.0
    superseded_penalty = 0.2 if superseded else 1.0
    return base_score * feedback_multiplier * correction_boost * superseded_penalty


def _positive_reinforcement_gain(node: Node) -> float:
    """Diminishing-returns factor for a positive usefulness increment.

    See the rationale at USEFULNESS_GAIN_HEADROOM_FLOOR: headroom shrinks
    gains as usefulness approaches saturation (floored so the 1.0 ceiling
    stays reachable; negative usefulness gets full headroom), and log-access
    crowding makes frequently delivered nodes need more evidence per unit of
    boost regardless of where their usefulness sits. Knobs are read at call
    time so the legacy rule is recoverable by setting them to their neutral
    values (floor 1.0, damping 0.0).
    """

    headroom = 1.0 - min(max(node.usefulness_score, 0.0), 1.0)
    saturation = max(USEFULNESS_GAIN_HEADROOM_FLOOR, headroom)
    crowding = 1.0 + USEFULNESS_GAIN_ACCESS_DAMPING * _log1p(node.access_count)
    return saturation / crowding


def _method_signals(result: Any, signed_signal: float) -> dict[str, float]:
    """Per-method credit proportional to each method's raw score share.

    Positive feedback rewards every method by its share of the recorded raw
    evidence for the reinforced result; negative feedback assigns blame by
    the same shares. Shares come from raw scores, not weight-multiplied
    contributions, so the update is independent of the current weights (a
    weight-share rule would compound its own bias). With no positive raw
    evidence there is nothing to attribute: every signal is zero and the
    weights stay put. The replay harness's ``proportional`` credit rule
    delegates here — this function is the single source of truth.
    """

    components = {
        "bm25": max(0.0, float(getattr(result, "bm25_score", 0.0) or 0.0)),
        "vector": max(0.0, float(getattr(result, "vector_score", 0.0) or 0.0)),
        "graph": max(0.0, float(getattr(result, "graph_score", 0.0) or 0.0)),
    }
    total = sum(components.values())
    if total <= 0.0:
        return {method: 0.0 for method in components}
    return {method: signed_signal * value / total for method, value in components.items()}


def _result_node(result: Any, store: MemoryStore) -> Node | None:
    node = getattr(result, "node", None)
    if isinstance(node, Node):
        return store.get_node(node.id) or node
    node_id = getattr(result, "node_id", None) or getattr(result, "id", None)
    if node_id:
        return store.get_node(str(node_id))
    return None


def _log1p(value: int) -> float:
    return math.log1p(max(0, value))


def _implicit_connection_weight(rank: int) -> float:
    return max(0.2, 0.75 / (rank + 1))
