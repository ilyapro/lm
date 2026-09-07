"""Content grounding: did the consuming trace actually use a recall result?

Single source of truth for the IDF-containment usefulness signal. Two callers
share this module and must never re-implement the arithmetic:

* :mod:`living_memory.feedback` — the *live* credit-assignment loop. When a
  ``memory_remember`` consumes a pending recall event, only the results whose
  content is grounded in the consuming trace earn reinforcement.
* :mod:`living_memory.replay` — the *offline* harness, where the same measure
  is the ``grounded`` confirmed-useful label.

A result is **grounded** in a trace when the IDF-weighted share of the result
node's content tokens that also occur in the trace's content (its
*containment*) reaches ``min_containment``. IDF suppresses boilerplate and
lets identifiers, paths and error strings carry the decision, so "the agent
wrote about this node" is measured rather than assumed.

Corpus for the IDF differs by caller and that difference is deliberate:

* Replay owns the whole labeled corpus (~8.6k documents on the 2026-08-17
  snapshot) and builds one global index.
* The live loop only holds the consuming trace and the results of the event
  being consumed, so it builds a per-event index from exactly those documents.

The threshold. ``CALIBRATED_MIN_CONTAINMENT`` is 0.22, adopted in September
2026 (``scripts/grounding_recalibration.py``,
``artifacts/grounding/recalibration-2026-09.md``) on the closed recall events
of the 2026-09-07 sfx and alt snapshots under the Cyrillic-stemming tokenizer.
The criterion, fixed before the sweep, uses the production encoder's cosine
between the delivered node and the closing trace as a reference the lexical
measure shares nothing with: adopt the lowest threshold at which, pooled and
on each host and at two relatedness cuts (0.5/0.3 and 0.6/0.4), at most 5% of
*unrelated* result/trace pairs ground and *related* pairs ground at least five
times as often. 0.15 grounds 8-17% of unrelated pairs, 0.20 and 0.21 still
exceed 5% on alt, 0.22 is the first value that clears everywhere; it lifts
closures that reinforce at least one node from 28.4% to 40.5% pooled (sfx
17.8% -> 24.7%, alt 39.0% -> 56.3%). The June value 0.25 was chosen on the
2026-08-17 snapshot for the agreement between this per-event view and the
replay's whole-corpus view (96.2% of 72,023 pairs,
``scripts/grounding_calibration.py``, ``artifacts/grounding/calibration.md``);
that agreement is re-reported at the new value in the recalibration artifact.

``LM_GROUNDING_MIN_CONTAINMENT`` overrides the constant for one process. It is
read once, at import, because every consumer binds ``DEFAULT_MIN_CONTAINMENT``
by name at its own import; a value that is not a finite number in ``(0, 1]``
is logged and ignored.
"""

from __future__ import annotations

import logging
import math
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from living_memory.embeddings import tokenize

_LOG = logging.getLogger(__name__)

#: Environment variable that overrides the shipped threshold at import time.
MIN_CONTAINMENT_ENV_VAR = "LM_GROUNDING_MIN_CONTAINMENT"

#: The calibrated threshold; see ``artifacts/grounding/recalibration-2026-09.md``.
CALIBRATED_MIN_CONTAINMENT = 0.22


def resolve_min_containment(
    raw: str | None, *, fallback: float = CALIBRATED_MIN_CONTAINMENT
) -> float:
    """Threshold from an environment value, or ``fallback`` when it is unusable.

    Accepts a decimal in ``(0, 1]``. Unset or blank means "not overridden";
    anything that does not parse as a finite number in that range is logged and
    ignored, because a credit gate that silently opened to 0 or closed to 1
    on a typo would look exactly like the signal changing.
    """

    if raw is None:
        return fallback
    text = raw.strip()
    if not text:
        return fallback
    try:
        value = float(text)
    except ValueError:
        _LOG.warning(
            "%s=%r is not a number; using %s", MIN_CONTAINMENT_ENV_VAR, raw, fallback
        )
        return fallback
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        _LOG.warning(
            "%s=%r is outside (0, 1]; using %s", MIN_CONTAINMENT_ENV_VAR, raw, fallback
        )
        return fallback
    return value


#: Containment at or above which a result counts as used by the trace. Read
#: once, at import: every caller (``feedback``, ``replay``, ``attestation``)
#: binds this name at its own import, so the override must be in the process
#: environment before ``living_memory`` is first imported.
DEFAULT_MIN_CONTAINMENT = resolve_min_containment(os.environ.get(MIN_CONTAINMENT_ENV_VAR))


def token_set(text: str) -> frozenset[str]:
    """Content tokens of one document under the production tokenizer."""

    return frozenset(tokenize(text))


def build_idf(token_sets: Iterable[Iterable[str]]) -> dict[str, float]:
    """Inverse document frequency over a corpus of already-tokenized documents.

    ``log(1 + documents / document_frequency)``: a token in every document
    still carries a small positive weight, a token unique to one document
    carries the most. Documents are counted as passed, so a caller that wants
    a node and a trace weighted once each must not pass either twice.
    """

    frequency: Counter[str] = Counter()
    documents = 0
    for tokens in token_sets:
        frequency.update(set(tokens))
        documents += 1
    corpus_size = max(1, documents)
    return {token: math.log(1.0 + corpus_size / count) for token, count in frequency.items()}


def containment(
    node_tokens: Iterable[str] | None,
    trace_tokens: Iterable[str] | None,
    idf: Mapping[str, float],
) -> float:
    """IDF-weighted share of the node's tokens present in the trace.

    Returns 0.0 when either side is empty or the node carries no IDF mass —
    an unmeasurable pair is never grounded.
    """

    if not node_tokens or not trace_tokens:
        return 0.0
    node_set = node_tokens if isinstance(node_tokens, frozenset | set) else frozenset(node_tokens)
    trace_set = (
        trace_tokens if isinstance(trace_tokens, frozenset | set) else frozenset(trace_tokens)
    )
    total = sum(idf.get(token, 0.0) for token in node_set)
    if total <= 0.0:
        return 0.0
    shared = sum(idf.get(token, 0.0) for token in node_set & trace_set)
    return shared / total


@dataclass(frozen=True, slots=True)
class Grounding:
    """Per-result grounding verdict with the number that produced it."""

    node_id: str
    containment: float
    grounded: bool


def ground_token_sets(
    trace_tokens: Iterable[str],
    result_tokens: Mapping[str, frozenset[str]],
    *,
    min_containment: float = DEFAULT_MIN_CONTAINMENT,
) -> dict[str, Grounding]:
    """Grade one consumed event's already-tokenized results against its trace.

    The IDF index is built from exactly the documents at hand — the consuming
    trace plus the result nodes of this event — which is all the live write
    path holds when it must decide. This is the arithmetic both callers share:
    :func:`ground_results` tokenizes for the live path first, and the replay
    harness passes its pre-tokenized corpus documents straight in.
    """

    if not result_tokens:
        return {}
    trace_set = frozenset(trace_tokens)
    idf = build_idf([trace_set, *result_tokens.values()])
    graded: dict[str, Grounding] = {}
    for node_id, tokens in result_tokens.items():
        value = containment(tokens, trace_set, idf)
        graded[node_id] = Grounding(
            node_id=node_id,
            containment=value,
            grounded=value >= min_containment,
        )
    return graded


def ground_results(
    trace_content: str,
    result_contents: Mapping[str, str],
    *,
    min_containment: float = DEFAULT_MIN_CONTAINMENT,
) -> dict[str, Grounding]:
    """Grade one consumed event's results, tokenizing raw content first.

    Live entry point. Empty result sets return an empty mapping without
    tokenizing the trace, so a ``memory_remember`` that consumes nothing pays
    nothing.
    """

    if not result_contents:
        return {}
    return ground_token_sets(
        token_set(trace_content),
        {node_id: token_set(content) for node_id, content in result_contents.items()},
        min_containment=min_containment,
    )
