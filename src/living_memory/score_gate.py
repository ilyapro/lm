"""Quality gate on recall results: weak results do not take a full slot.

See docs/recall-score-gate.md. ``apply_score_gate``
turns the ranked list into ``(delivered, residual)``. ``delivered`` is what the
recall returns (at most ``max_results``); ``residual`` feeds ``recall_map`` and
keeps every ranked result that was not delivered, so a cut result stays
reachable.

Valves, read per call:

- ``LM_RECALL_MIN_SCORE`` -- the threshold on :func:`gate_score`. Unset, empty,
  unparsable, non-positive or NaN: the gate is off and the function is the
  plain ``max_results`` cut, byte for byte.
- ``LM_RECALL_GATE_FORM`` -- ``drop`` (default) or ``stub``. ``drop`` walks
  ``ranked`` and prioritizes results that pass, filling spare slots with
  ``below_threshold`` source references; ``stub``
  keeps the plain cut and marks the failing results inside it
  ``withheld="below_threshold"`` (shaped by ``delivery`` as a content-less
  stub with a ``content_ref``). Anything else reads as ``drop``.

Every rank needs the same evidence to deliver content. References keep a
bounded route to found sources without treating them as qualified answers.

The gate score is NOT the ranker's score. The ranker blends channels with the
store's per-scope retrieval weights, which move with every ~100 credits; a
threshold on that number would mean something different next week. The gate
re-blends the result's current source channels with the fixed
:data:`REFERENCE_WEIGHTS`, through the ranker's own blend function. The
strong-vector floor, graph-weight floor, scope boost, source confidence restrictions,
superseded penalty, causal boost, and per-node ``demotions`` apply here. The
ranker's correction boost does not enter an individual gate score; a present
correction instead qualifies when its original passes the gate.
Transient ``source_graph_score`` excludes query anchors and historical edge
reinforcement. Access and the entire net usefulness accumulator stay outside both
qualification scales; positive confidence never lifts them. Historical signals
retain their ranking and discovery roles.

Trigger-found instructions retain a separate availability scale based on
trigger overlap, query coverage and negative restrictions:
:func:`trigger_gate_score` against
:data:`SCHEMA_TRIGGER_GATE_MIN`. Their channel-scale score uses the same
content-supported trigger contribution as the ranker. A result passes when
either scale passes. Under
``LM_RECALL_SCHEMA_TRIGGER=name`` (``retrieval.SCHEMA_TRIGGER_ENV``) there is
no trigger scale: every result, schema or not, is gated on :func:`gate_score`.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

from living_memory.feedback import feedback_weighted_score
from living_memory.models import RetrievalWeights

MIN_SCORE_ENV = "LM_RECALL_MIN_SCORE"
GATE_FORM_ENV = "LM_RECALL_GATE_FORM"

GATE_FORM_DROP = "drop"
GATE_FORM_STUB = "stub"

#: ``RecallResult.withheld`` (and the delivery class) of a stubbed result.
WITHHELD_BELOW_THRESHOLD = "below_threshold"

#: Fixed channel weights of the gate score. The seeded weights of the
#: ``global`` family (config.DEFAULT_RETRIEVAL_WEIGHTS): the one default that
#: gives every channel a say, so no channel's evidence reads as zero. Frozen
#: here on purpose -- the store's weights are a moving average and must not
#: move the threshold.
REFERENCE_WEIGHTS = RetrievalWeights(scope="score_gate", bm25=0.4, vector=0.4, graph=0.2)

#: A trigger-found schema passes on its own scale when
#: ``trigger_gate_score >= SCHEMA_TRIGGER_GATE_MIN``. The trigger scale is
#: normalised so that a clean trigger hit (base score, neutral feedback, no
#: demotion) reads 1.0; the trigger already cleared the overlap threshold to
#: exist at all, so only negative evidence -- a demotion, low confidence,
#: a correction -- that at least halves it takes the schema out.
SCHEMA_TRIGGER_GATE_MIN = 0.5
#: ``retrieval.SCHEMA_TRIGGER_BASE_SCORE`` (not imported: retrieval imports
#: this module). The trigger score of a hit at the overlap threshold is a hair
#: above it; normalising by it puts a clean hit at ~1.0.
SCHEMA_TRIGGER_SCALE = 0.95


def min_score_from_env() -> float | None:
    """Read ``LM_RECALL_MIN_SCORE``; ``None`` means the gate is off."""

    raw = os.environ.get(MIN_SCORE_ENV, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value) or value <= 0.0:
        return None
    return value


def gate_form_from_env() -> str:
    """Read ``LM_RECALL_GATE_FORM``: ``stub`` or (default, and any other value) ``drop``."""

    raw = os.environ.get(GATE_FORM_ENV, "").strip().lower()
    return GATE_FORM_STUB if raw == GATE_FORM_STUB else GATE_FORM_DROP


def gate_score(
    result: Any,
    demotions: Mapping[str, float] | None = None,
    *,
    plan: Any = None,
    causal_mode: bool = False,
) -> float:
    """Current source evidence of one ``RecallResult`` on the channel scale.

    Callable offline on a result carrying source evidence and recall demotions.
    A historical effective graph score alone is not source evidence.
    Without ``plan`` the scope boost is 1.0 (the boost needs the resolved
    scope list). Ordinary trigger evidence receives the same content-supported
    contribution as in ranking. See :func:`trigger_gate_score` for the
    separate instruction-availability scale.
    """

    # Imported lazily: retrieval imports this module at load time.
    from living_memory.retrieval import _Candidate, _blend_candidate_score, schema_trigger_by_name

    candidate = _Candidate(
        node=result.node,
        bm25_score=result.bm25_score,
        vector_score=result.vector_score,
        graph_score=getattr(result, "source_graph_score", 0.0),
        trigger_score=0.0 if schema_trigger_by_name() else result.trigger_score,
        trigger_coverage=getattr(result, "trigger_coverage", 0.0),
    )
    score = _blend_candidate_score(
        candidate,
        candidate.graph_score,
        REFERENCE_WEIGHTS,
        plan=plan if plan is not None else _NO_PLAN,
        causal_mode=causal_mode,
        superseded=bool(result.superseded),
        # Whether the node corrects another is not on the result; the 1.2x
        # correction boost is left out, which only makes the gate stricter
        # on corrections by that factor.
        superseding=False,
        qualification=True,
    )
    return score * _demotion(result, demotions)


def trigger_gate_score(
    result: Any,
    demotions: Mapping[str, float] | None = None,
) -> float | None:
    """Score of a trigger-found schema on the trigger scale; ``None`` otherwise.

    ``trigger_score / SCHEMA_TRIGGER_SCALE`` times source restrictions and demotion.
    Case-evidence carriers also need query coverage; a schema recipe retains
    its short-instruction availability when its trigger is one clause of a
    longer question. Scope and causal boosts are left out.
    """

    # Imported lazily: retrieval imports this module at load time.
    from living_memory.retrieval import schema_trigger_by_name

    node = result.node
    if node.level == "trace" or result.trigger_score <= 0.0:
        return None
    if schema_trigger_by_name():
        # ``name`` mode: a named schema earns its slot on the channel scale
        # like any other result; there is no trigger scale.
        return None
    return (
        feedback_weighted_score(
            node,
            result.trigger_score / SCHEMA_TRIGGER_SCALE,
            superseded=bool(result.superseded),
            qualification=True,
        )
        * (
            1.0 if node.level == "schema"
            else max(0.0, getattr(result, "trigger_coverage", 0.0)) ** 2
        )
        * _demotion(result, demotions)
    )


def passes_gate(
    result: Any,
    threshold: float,
    demotions: Mapping[str, float] | None = None,
    *,
    plan: Any = None,
    causal_mode: bool = False,
) -> bool:
    """Whether ``result`` earns a full slot at ``threshold``."""

    trigger = trigger_gate_score(result, demotions)
    if trigger is not None and trigger >= SCHEMA_TRIGGER_GATE_MIN:
        return True
    return gate_score(result, demotions, plan=plan, causal_mode=causal_mode) >= threshold


@dataclass
class QueryRelevance:
    """One recall's relevance evidence, shared by selection and its optional map.

    Keep the full ranked set so a correction can inherit evidence even when
    its original is outside the residual. Evaluation is lazy: with the main
    gate disabled, only the optional map needs it. Without a configured
    threshold, positive channel evidence or a qualifying trigger is enough.
    """

    ranked: Sequence[Any]
    plan: Any = None
    demotions: Mapping[str, float] | None = None
    causal_mode: bool = False
    corrections_by_superseded: Mapping[str, Sequence[str]] | None = None
    threshold: float | None = field(default_factory=min_score_from_env)
    _eligible: frozenset[str] | None = field(default=None, init=False, repr=False)

    @cached_property
    def _present_edges(self) -> dict[str, tuple[str, ...]]:
        positions = {
            result.node.id: index for index, result in enumerate(self.ranked)
        }
        edges: dict[str, tuple[str, ...]] = {}
        for result in self.ranked:
            original_id = result.node.id
            corrections = (
                self.corrections_by_superseded.get(original_id, ())
                if self.corrections_by_superseded is not None
                else getattr(result, "correction_ids", ())
            )
            present = tuple(
                correction_id for correction_id in corrections
                if correction_id in positions
                and positions[correction_id] < positions[original_id]
            )
            if present:
                edges[original_id] = present
        return edges

    def _passes(self, result: Any) -> bool:
        if self.threshold is not None:
            return passes_gate(
                result, self.threshold, self.demotions,
                plan=self.plan, causal_mode=self.causal_mode,
            )
        trigger = trigger_gate_score(result, self.demotions)
        return (
            trigger is not None and trigger >= SCHEMA_TRIGGER_GATE_MIN
        ) or gate_score(
            result, self.demotions, plan=self.plan, causal_mode=self.causal_mode,
        ) > 0.0

    def passes(self, result: Any) -> bool:
        # A plain main cut need not evaluate its unused tail. Correction
        # chains require the full set, as they do in main selection.
        if self._present_edges:
            return result.node.id in self.eligible_ids()
        return self._passes(result)

    def eligible_ids(self) -> frozenset[str]:
        if self._eligible is not None:
            return self._eligible
        eligible = {
            result.node.id for result in self.ranked if self._passes(result)
        }
        for result in reversed(self.ranked):
            if result.node.id in eligible:
                eligible.update(self._present_edges.get(result.node.id, ()))
        self._eligible = frozenset(eligible)
        return self._eligible


def apply_score_gate(
    ranked: Sequence[Any],
    max_results: int,
    *,
    plan: Any = None,
    demotions: Mapping[str, float] | None = None,
    causal_mode: bool = False,
    corrections_by_superseded: Mapping[str, Sequence[str]] | None = None,
    relevance: QueryRelevance | None = None,
) -> tuple[list[Any], list[Any]]:
    """Select results, letting present corrections inherit their original's relevance."""

    threshold = relevance.threshold if relevance is not None else min_score_from_env()
    if threshold is None or max_results <= 0 or not ranked:
        return list(ranked[:max_results]), list(ranked[max_results:])

    evidence = relevance or QueryRelevance(
        ranked, plan=plan, demotions=demotions, causal_mode=causal_mode,
        corrections_by_superseded=corrections_by_superseded or {}, threshold=threshold,
    )
    passes = evidence.passes

    if gate_form_from_env() == GATE_FORM_STUB:
        delivered = [
            result
            if passes(result)
            else replace(result, withheld=WITHHELD_BELOW_THRESHOLD)
            for result in ranked[:max_results]
        ]
        return delivered, list(ranked[max_results:])

    delivered: list[Any] = []
    residual: list[Any] = []
    for result in ranked:
        if len(delivered) < max_results and passes(result):
            delivered.append(result)
        else:
            residual.append(result)
    # If capacity remains, every residual failed the gate: the scan above
    # exhausted ranked without finding enough passing results. References
    # use the same slots and withheld shaping as the explicit stub form.
    spare = max_results - len(delivered)
    if spare:
        delivered.extend(
            replace(result, withheld=WITHHELD_BELOW_THRESHOLD)
            for result in residual[:spare]
        )
        residual = residual[spare:]
    return delivered, residual


def _demotion(result: Any, demotions: Mapping[str, float] | None) -> float:
    if not demotions:
        return 1.0
    return float(demotions.get(result.node.id, 1.0))


class _NoPlan:
    """Stand-in plan for offline scoring: no scope list, so no scope boost."""

    def boost_steps(self, scope: str) -> int:
        return 0


_NO_PLAN = _NoPlan()
