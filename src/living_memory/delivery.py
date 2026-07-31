"""Pure recall-delivery shaping: dedup and snippet ranked recall results.

``shape_recall_results`` maps ranked :class:`~living_memory.retrieval.RecallResult`
objects to the response dicts served by ``memory_recall``, keeping the exact
per-result key set of the legacy renderer (``node``/``score``/``bm25_score``/
``vector_score``/``graph_score``/``trigger_score``/``scope_rank``/``methods``/
``path``/``recall_event_id``) and adding a ``delivery`` class per result.

Rules, applied in this order per ranked result:

1. Twin dedup — among results whose ``node.content`` are byte-identical (e.g.
   a concept and its verbatim source trace), the highest-ranked result is the
   content-bearer; the rest become ``delivery: "twin_duplicate"`` stubs.
2. Session dedup — when enabled and the node id was already delivered on this
   transport session, the result becomes a ``delivery: "session_duplicate"``
   stub. Only content-bearers are considered: a twin of a session-duplicate
   stays a twin stub, never re-classified or promoted to full.
3. Snippeting — a would-be-full result whose content exceeds
   ``snippet_max_chars`` (>0) becomes ``delivery: "snippet"`` with content
   truncated at a clean boundary plus an ellipsis marker.
4. Otherwise ``delivery: "full"`` with the node dict untouched.

Backward compatibility: stubs and snippets preserve every key of the full node
dict (``resources.node_to_dict``). ``content`` is never absent — stubs carry a
one-line preview (~160 chars), snippets the truncated text. Non-full results
gain a ``content_ref`` hint for re-fetching the full node, their bulky
``provenance.prior_recalls`` list is summarized to a count (the ``provenance``
key itself always remains), and oversized ``context`` values are compacted per
key — strings truncated at a clean boundary, lists/objects replaced by
``{"count": n, "chars": m}`` (a procedural schema's ``context.procedure``
would otherwise re-ship the node's whole content alongside a 31-char stub).
Full deliveries and the ``memory_lookup`` re-fetch path stay byte-untouched.

Env knobs (read by the server wiring, not by the pure function):

- ``LM_DELIVERY_SNIPPET_CHARS`` — max chars delivered inline (default 1200,
  ``0`` disables snippeting).
- ``LM_DELIVERY_SESSION_DEDUP`` — default on; ``"0"`` disables (rollback valve).
- ``LM_DELIVERY_CONTEXT_VALUE_CHARS`` — max chars a single context value may
  occupy on non-full results (default 240, ``0`` disables context compaction).
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .resources import node_to_dict

if TYPE_CHECKING:
    from .retrieval import RecallResult

DELIVERY_FULL = "full"
DELIVERY_SNIPPET = "snippet"
DELIVERY_SESSION_DUPLICATE = "session_duplicate"
DELIVERY_TWIN_DUPLICATE = "twin_duplicate"

SNIPPET_CHARS_ENV = "LM_DELIVERY_SNIPPET_CHARS"
SESSION_DEDUP_ENV = "LM_DELIVERY_SESSION_DEDUP"
CONTEXT_VALUE_CHARS_ENV = "LM_DELIVERY_CONTEXT_VALUE_CHARS"

DEFAULT_SNIPPET_MAX_CHARS = 1200
# 240 keeps typical files/related lists verbatim while collapsing multi-KB
# procedure lists to a counter.
DEFAULT_CONTEXT_VALUE_MAX_CHARS = 240
PREVIEW_MAX_CHARS = 160
ELLIPSIS = "…"

_DISABLED_FLAGS = frozenset({"0", "false", "no", "off"})
_CLEAN_BOUNDARIES = ("\n\n", "\n", ". ", " ")


def snippet_max_chars_from_env() -> int:
    """Read ``LM_DELIVERY_SNIPPET_CHARS`` (0 disables snippeting, invalid -> default)."""

    return _chars_from_env(SNIPPET_CHARS_ENV, DEFAULT_SNIPPET_MAX_CHARS)


def context_value_max_chars_from_env() -> int:
    """Read ``LM_DELIVERY_CONTEXT_VALUE_CHARS`` (0 disables compaction, invalid -> default)."""

    return _chars_from_env(CONTEXT_VALUE_CHARS_ENV, DEFAULT_CONTEXT_VALUE_MAX_CHARS)


def _chars_from_env(env_var: str, default: int) -> int:
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(0, value)


def session_dedup_enabled_from_env() -> bool:
    """Read ``LM_DELIVERY_SESSION_DEDUP`` (default on; "0"/"false"/"no"/"off" disable)."""

    raw = os.environ.get(SESSION_DEDUP_ENV, "").strip().lower()
    if not raw:
        return True
    return raw not in _DISABLED_FLAGS


def shape_recall_results(
    results: Sequence[RecallResult],
    *,
    already_delivered_ids: set[str],
    snippet_max_chars: int,
    context_value_max_chars: int,
    session_dedup: bool,
) -> list[dict[str, Any]]:
    """Render ranked recall results, deduplicating and snippeting delivered content.

    Non-full results also compact each ``context`` value longer than
    ``context_value_max_chars`` (>0): strings are truncated at a clean
    boundary, lists/objects summarized to ``{"count", "chars"}``.

    Pure: never mutates ``results``, their nodes, or ``already_delivered_ids``;
    identical inputs produce identical output. Output order matches input order.
    """

    bearer_by_content: dict[str, str] = {}
    shaped: list[dict[str, Any]] = []
    for result in results:
        node_dict = node_to_dict(result.node)
        node_id = node_dict["id"]
        content = node_dict["content"]
        full_chars = len(content)

        duplicate_of: str | None = None
        if content:  # empty content carries nothing worth deduplicating
            bearer_id = bearer_by_content.get(content)
            if bearer_id is None:
                bearer_by_content[content] = node_id
            else:
                duplicate_of = bearer_id

        if duplicate_of is not None:
            delivery = DELIVERY_TWIN_DUPLICATE
            node_dict["content"] = _one_line_preview(content)
        elif session_dedup and node_id in already_delivered_ids:
            delivery = DELIVERY_SESSION_DUPLICATE
            node_dict["content"] = _one_line_preview(content)
        elif 0 < snippet_max_chars < full_chars:
            delivery = DELIVERY_SNIPPET
            node_dict["content"] = _truncate_at_boundary(content, snippet_max_chars)
        else:
            delivery = DELIVERY_FULL

        if delivery != DELIVERY_FULL:
            node_dict["provenance"] = _summarize_provenance(node_dict["provenance"])
            node_dict["context"] = _summarize_context(node_dict["context"], context_value_max_chars)

        entry: dict[str, Any] = {
            "node": node_dict,
            "score": result.score,
            "bm25_score": result.bm25_score,
            "vector_score": result.vector_score,
            "graph_score": result.graph_score,
            "trigger_score": result.trigger_score,
            "scope_rank": result.scope_rank,
            "methods": list(result.methods),
            "path": list(result.path),
            "recall_event_id": result.recall_event_id,
            "delivery": delivery,
        }
        if delivery != DELIVERY_FULL:
            entry["content_ref"] = _content_ref(node_id, full_chars, duplicate_of=duplicate_of)
        shaped.append(entry)
    return shaped


def _content_ref(
    node_id: str, full_content_chars: int, *, duplicate_of: str | None
) -> dict[str, Any]:
    ref: dict[str, Any] = {
        "node_id": node_id,
        "fetch": f'memory_lookup(node_id="{node_id}")',
        "full_content_chars": full_content_chars,
    }
    if duplicate_of is not None:
        ref["duplicate_of"] = duplicate_of
    return ref


def _summarize_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    prior = provenance.get("prior_recalls")
    if isinstance(prior, list) and prior:
        return {**provenance, "prior_recalls": {"count": len(prior)}}
    return provenance


def _summarize_context(context: dict[str, Any], max_value_chars: int) -> dict[str, Any]:
    if max_value_chars <= 0:  # rollback valve: deliver context unshaped
        return context
    return {key: _summarize_context_value(value, max_value_chars) for key, value in context.items()}


def _summarize_context_value(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return _truncate_at_boundary(value, max_chars)
    if isinstance(value, (list, dict)):
        chars = _json_chars(value)
        if chars <= max_chars:
            return value
        return {"count": len(value), "chars": chars}
    return value


def _json_chars(value: Any) -> int:
    # ensure_ascii=False measures the UTF-8 wire text; \u-escapes would count
    # non-ASCII context (e.g. Cyrillic) ~6x over its delivered size.
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _one_line_preview(content: str, max_chars: int = PREVIEW_MAX_CHARS) -> str:
    first_line = content.strip().split("\n", 1)[0].strip()
    if len(first_line) <= max_chars:
        return first_line
    return _truncate_at_boundary(first_line, max_chars)


def _truncate_at_boundary(text: str, max_chars: int) -> str:
    budget = max(0, max_chars - len(ELLIPSIS))
    head = text[:budget]
    floor = budget // 2
    for boundary in _CLEAN_BOUNDARIES:
        cut = head.rfind(boundary)
        if cut >= floor:
            head = head[:cut]
            break
    return head.rstrip() + ELLIPSIS
