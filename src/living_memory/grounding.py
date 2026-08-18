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

Measured on the 2026-08-17 snapshot (72,023 result/trace pairs over 9,069
consumed events, ``scripts/grounding_calibration.py``): per-event IDF
containment correlates with global-corpus IDF containment at Pearson 0.984,
and the two agree on the >= 0.25 grounded/not-grounded decision for 96.2% of
pairs. The shared default threshold therefore stays at the replay value; see
``artifacts/grounding/calibration.json`` for the threshold sweep.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from living_memory.embeddings import tokenize

#: Containment at or above which a result counts as used by the trace.
DEFAULT_MIN_CONTAINMENT = 0.25


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
