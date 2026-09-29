"""Quality gate on recall results: weak results do not take a full slot.

Seam for goal recall-precision (docs/recall-score-gate.md). ``apply_score_gate``
turns the ranked list into ``(delivered, residual)``. ``delivered`` is what the
recall returns (at most ``max_results``); ``residual`` feeds ``recall_map`` and
keeps every ranked result that was not delivered, so a cut result stays
reachable.

Valves, read per call:

- ``LM_RECALL_MIN_SCORE`` -- the threshold on :func:`gate_score`. Unset, empty,
  unparsable, non-positive or NaN: the gate is off and the function is the
  plain ``max_results`` cut, byte for byte.
- ``LM_RECALL_GATE_FORM`` -- ``drop`` (default) or ``stub``. ``drop`` walks
  ``ranked`` and delivers up to ``max_results`` results that pass; ``stub``
  keeps the plain cut and marks the failing results inside it
  ``withheld="below_threshold"`` (shaped by ``delivery`` as a content-less
  stub with a ``content_ref``). Anything else reads as ``drop``.

Rank 1 is always delivered in full, whatever its score.

The gate score is NOT the ranker's score. The ranker blends channels with the
store's per-scope retrieval weights, which move with every ~100 credits; a
threshold on that number would mean something different next week. The gate
re-blends the result's own channel scores (bm25/vector/graph) with the fixed
:data:`REFERENCE_WEIGHTS`, through the ranker's own blend function, so every
multiplier the ranker applies (strong-vector floor, graph-weight floor, scope
boost, node feedback multiplier, superseded penalty, causal boost) applies here
too, and then the per-node multiplier from ``demotions``.

Schema results found by a trigger live on another scale (trigger score
0.95 + 0.05 * overlap, then a 1.8x boost) and are gated on their own
constant: :func:`trigger_gate_score` against :data:`SCHEMA_TRIGGER_GATE_MIN`.
Such a result passes when either scale passes. Under
``LM_RECALL_SCHEMA_TRIGGER=name`` (``retrieval.SCHEMA_TRIGGER_ENV``) there is
no trigger scale: every result, schema or not, is gated on :func:`gate_score`.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
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
#: exist at all, so only negative evidence -- a demotion, negative usefulness,
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
    """Weight-independent score of one ``RecallResult`` on the channel scale.

    Callable offline on a stored result plus the demotions of its recall.
    Without ``plan`` the scope boost is 1.0 (the boost needs the resolved
    scope list). The trigger channel does not enter here; see
    :func:`trigger_gate_score`.
    """

    # Imported lazily: retrieval imports this module at load time.
    from living_memory.retrieval import _Candidate, _blend_candidate_score

    candidate = _Candidate(
        node=result.node,
        bm25_score=result.bm25_score,
        vector_score=result.vector_score,
        graph_score=result.graph_score,
    )
    score = _blend_candidate_score(
        candidate,
        result.graph_score,
        REFERENCE_WEIGHTS,
        plan=plan if plan is not None else _NO_PLAN,
        causal_mode=causal_mode,
        superseded=bool(result.superseded),
        # Whether the node corrects another is not on the result; the 1.2x
        # correction boost is left out, which only makes the gate stricter
        # on corrections by that factor.
        superseding=False,
    )
    return score * _demotion(result, demotions)


def trigger_gate_score(
    result: Any,
    demotions: Mapping[str, float] | None = None,
) -> float | None:
    """Score of a trigger-found schema on the trigger scale; ``None`` otherwise.

    ``trigger_score / SCHEMA_TRIGGER_SCALE`` times the node feedback
    multiplier (superseded penalty included) and the demotion. The scope and
    causal boosts are left out: a trigger match does not depend on either.
    """

    # Imported lazily: retrieval imports this module at load time.
    from living_memory.retrieval import schema_trigger_by_name

    node = result.node
    if node.level != "schema" or result.trigger_score <= 0.0:
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


def apply_score_gate(
    ranked: Sequence[Any],
    max_results: int,
    *,
    plan: Any = None,
    demotions: Mapping[str, float] | None = None,
    causal_mode: bool = False,
) -> tuple[list[Any], list[Any]]:
    """Return ``(delivered, residual)``; the plain cut while the valve is off."""

    threshold = min_score_from_env()
    if threshold is None or max_results <= 0 or not ranked:
        return list(ranked[:max_results]), list(ranked[max_results:])

    def passes(result: Any) -> bool:
        return passes_gate(
            result, threshold, demotions, plan=plan, causal_mode=causal_mode
        )

    if gate_form_from_env() == GATE_FORM_STUB:
        delivered = [
            result
            if index == 0 or passes(result)
            else replace(result, withheld=WITHHELD_BELOW_THRESHOLD)
            for index, result in enumerate(ranked[:max_results])
        ]
        return delivered, list(ranked[max_results:])

    delivered = [ranked[0]]
    residual: list[Any] = []
    for result in ranked[1:]:
        if len(delivered) < max_results and passes(result):
            delivered.append(result)
        else:
            residual.append(result)
    return delivered, residual


def _demotion(result: Any, demotions: Mapping[str, float] | None) -> float:
    if not demotions:
        return 1.0
    return float(demotions.get(result.node.id, 1.0))


class _NoPlan:
    """Stand-in plan for offline scoring: no scope list, so no scope boost."""

    scopes: tuple[str, ...] = ()

    def rank(self, scope: str) -> int:
        return 0


_NO_PLAN = _NoPlan()
