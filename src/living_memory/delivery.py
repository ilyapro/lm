"""Pure recall-delivery shaping: dedup, snippet, and diet ranked recall results.

``shape_recall_results`` maps ranked :class:`~living_memory.retrieval.RecallResult`
objects to the response dicts served by ``memory_recall``, keeping the
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
3. Near-duplicate dedup — a result that repeats the *meaning* of a
   higher-ranked result of the same answer becomes a
   ``delivery: "near_duplicate"`` stub, exactly as a byte twin does. Byte
   equality finds almost none of these: on the live corpus (2026-08-23) 4
   active nodes were byte-identical to another while 550 had a neighbour at
   cosine >= 0.95. The ``duplicate_of`` map arrives READY-MADE — the caller
   builds it with ``near_dup.mean_pooled_vectors``/``build_duplicate_map`` —
   because this module computes no vectors and reads no database; that is
   what keeps the function pure. Checked last of the three stub rules, so
   the two older classes keep their exact meaning: a node both already
   delivered this session and a paraphrase of a higher-ranked one stays
   ``session_duplicate``.
4. Snippeting — the content budget of a would-be-full result comes from the
   snippet ladder indexed by its content-bearer position (stubs of every
   class don't consume ladder slots): the first bearer has a finite 2400-char
   budget, lower-ranked bearers get descending budgets, and oversized content
   is selected in task-relevant passages. Binding schemas ship complete. With
   the ladder disabled (``snippet_ladder=None``) every bearer shares the
   uniform ``snippet_max_chars`` budget (``0`` disables snippeting) — the
   legacy contract.
5. Otherwise ``delivery: "full"`` with complete ``content``.

Before all five, a result the score gate marked ``withheld="below_threshold"``
(``score_gate``, ``LM_RECALL_GATE_FORM=stub``) becomes a
``delivery: "below_threshold"`` stub: empty ``content`` (the key stays), a
``content_ref`` naming the node for ``memory_lookup``, no ladder slot, and it
never becomes a twin bearer -- its text was not delivered, so a later twin
must not point at it. Without the gate no result carries ``withheld`` and
nothing below changes.

``near_duplicate`` joins ``session_duplicate``/``twin_duplicate`` in the
trailing stub run a gated recall may drop (``server.py``). It belongs there
for a stronger reason than either: a near-duplicate's text ships in full
under its bearer, higher up in the very same response, so dropping the run
removes nothing the agent is not already holding — and the drop stays doubly
opt-in behind ``LM_RECALL_REPEAT_GATING`` and
``LM_RECALL_REPEAT_DROP_TRAILING_STUBS``.

Diet, applied per delivered entry (each lever has its own rollback valve):

* Stubs and snippets preserve every key of the full node dict
  (``resources.node_to_dict``); ``content`` is never absent — stubs carry a
  one-line preview (~160 chars), snippets the truncated text; non-full
  results always gain a ``content_ref`` for re-fetching the full node.
* Provenance is summarized on every delivery class: ``prior_recalls``
  collapses to ``{"count": n}`` and any other oversized provenance value is
  compacted per key exactly like context values. ``corrections`` keep one
  structured entry per correction with every key present (``by``/``timestamp``
  and other short values verbatim) and only oversized texts truncated — the
  supersedes signal (that, when, and by whom a node was corrected, plus the
  correction gist) always survives delivery.
* Oversized ``context`` values are compacted per key — strings truncated at
  a clean boundary, lists/objects replaced by ``{"count": n, "chars": m}``
  (a procedural schema's ``context.procedure`` would otherwise re-ship the
  node's whole content alongside a 31-char stub).
* Full deliveries whose provenance or context lost anything to the diet gain
  a minimal ``content_ref`` hint (``{"node_id": ...}``) marking that
  ``memory_lookup`` returns strictly more than was delivered.
* ``stats`` drop bookkeeping and default-valued fields (``last_accessed``,
  null ``temporal_hint``, default ``confidence``/``unique_agents``, zero
  counts) and round ``usefulness_score``.
* Sparse entries keep every score field present (zeros included, floats
  rounded to 6 decimals — wire consumers see a predictable schema with
  fidelity far inside any sane tolerance) and drop the per-entry
  ``recall_event_id`` copy (the envelope carries it once), empty ``path``,
  null ``agent``/``task``/``decay_reason``, ``decayed: false``,
  ``timestamp``/``updated_at`` equal to ``created_at``, and the
  ``content_ref.fetch`` boilerplate.

The ``memory_lookup`` re-fetch path stays byte-complete: every removal above
is wire-only and reversible per entry via the advertised lookup.

Env knobs (read by the server wiring, not by the pure function):

- ``LM_DELIVERY_SNIPPET_LADDER`` — per-bearer-position content budgets,
  comma-separated ``full`` or char counts (default ``2400,1000,700,500,300,200``;
  positions past the end reuse the last entry; ``off`` falls back to the
  uniform legacy budget below).
- ``LM_DELIVERY_SNIPPET_CHARS`` — uniform max chars delivered inline
  (default 1200, ``0`` disables snippeting). Setting it explicitly while the
  ladder is unset selects the uniform legacy mode.
- ``LM_DELIVERY_SESSION_DEDUP`` — default on; ``"0"`` disables (rollback valve).
- ``LM_DELIVERY_CONTEXT_VALUE_CHARS`` — max chars a single context value may
  occupy on dieted results (default 160, ``0`` disables context compaction).
- ``LM_DELIVERY_FULL_NODE_DIET`` — default on; ``"0"`` restores byte-untouched
  provenance/context on ``delivery: "full"`` entries (rollback valve).
- ``LM_DELIVERY_PROVENANCE_VALUE_CHARS`` — max chars a single provenance value
  may occupy on dieted results (default 160, ``0`` limits provenance shaping
  to the legacy ``prior_recalls`` summarization).
- ``LM_DELIVERY_STATS_COMPACTION`` — default on; ``"0"`` restores complete
  ``stats`` dicts (rollback valve).
- ``LM_DELIVERY_SPARSE`` — default on; ``"0"`` restores zero/null/duplicate
  entry fields and full float precision (rollback valve).
- ``LM_RECALL_NEAR_DUP_COSINE`` — cosine a result must exceed against a
  higher-ranked result of the same answer to collapse into it (default 0.95).
  ``"0"`` restores byte-only dedup byte-for-byte: the call site builds no map
  at all, so not one chunk vector is read (rollback valve). A value above 1.0
  is unreachable and therefore also disables the collapse.
- ``LM_RECALL_NEAR_DUP_LENGTH_RATIO`` — the length guard (default 0.2): a
  candidate longer than its bearer by more than this fraction is never
  collapsed. That case is "the same fact plus a new detail", and the detail
  has to reach the agent.

The two near-dup knobs are ``LM_RECALL_*`` rather than ``LM_DELIVERY_*``
because what they gate — pooling a node's chunk vectors — happens at the
recall call site; this module only receives the finished map.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .resources import node_to_dict

if TYPE_CHECKING:
    from .retrieval import RecallResult

DELIVERY_FULL = "full"
DELIVERY_SNIPPET = "snippet"
DELIVERY_SESSION_DUPLICATE = "session_duplicate"
DELIVERY_TWIN_DUPLICATE = "twin_duplicate"
DELIVERY_NEAR_DUPLICATE = "near_duplicate"
# Same string as score_gate.WITHHELD_BELOW_THRESHOLD, the withheld reason.
DELIVERY_BELOW_THRESHOLD = "below_threshold"

SNIPPET_CHARS_ENV = "LM_DELIVERY_SNIPPET_CHARS"
SESSION_DEDUP_ENV = "LM_DELIVERY_SESSION_DEDUP"
CONTEXT_VALUE_CHARS_ENV = "LM_DELIVERY_CONTEXT_VALUE_CHARS"
SNIPPET_LADDER_ENV = "LM_DELIVERY_SNIPPET_LADDER"
FULL_NODE_DIET_ENV = "LM_DELIVERY_FULL_NODE_DIET"
PROVENANCE_VALUE_CHARS_ENV = "LM_DELIVERY_PROVENANCE_VALUE_CHARS"
STATS_COMPACTION_ENV = "LM_DELIVERY_STATS_COMPACTION"
SPARSE_ENV = "LM_DELIVERY_SPARSE"
NEAR_DUP_COSINE_ENV = "LM_RECALL_NEAR_DUP_COSINE"
NEAR_DUP_LENGTH_RATIO_ENV = "LM_RECALL_NEAR_DUP_LENGTH_RATIO"

DEFAULT_SNIPPET_MAX_CHARS = 1200
# 160 keeps short files/tags lists verbatim while collapsing longer lists and
# multi-KB procedure dumps to counters (the pre-ladder default 240 is
# restorable via LM_DELIVERY_CONTEXT_VALUE_CHARS).
DEFAULT_CONTEXT_VALUE_MAX_CHARS = 160
# 160 keeps short provenance values (strategy markers, small id lists)
# verbatim while collapsing recalled_nodes/source_traces id dumps to counters.
DEFAULT_PROVENANCE_VALUE_MAX_CHARS = 160
# Budget 0 delivers complete content at that ladder position.
LADDER_COMPLETE = 0
DEFAULT_SNIPPET_LADDER = (2400, 1000, 700, 500, 300, 200)
PREVIEW_MAX_CHARS = 160
# 0.95 is where the live corpus stops holding distinct facts: the 0.85-0.95
# band is different facts said in similar words (median max-cosine between
# active traces is 0.829, and a concept sits at 0.82-0.95 from its own
# sources), while 550 of 12,874 nodes have a neighbour above 0.95.
DEFAULT_NEAR_DUP_COSINE = 0.95
# An excess fraction, not a factor: 0.2 protects any candidate more than 20%
# longer than its bearer from being collapsed.
DEFAULT_NEAR_DUP_LENGTH_RATIO = 0.2
ELLIPSIS = "…"

_DISABLED_FLAGS = frozenset({"0", "false", "no", "off"})
_LADDER_OFF_FLAGS = frozenset({"off", "false", "no", "none", "uniform"})
_CLEAN_BOUNDARIES = ("\n\n", "\n", ". ", " ")
_SPARSE_SCORE_KEYS = ("score", "bm25_score", "vector_score", "graph_score", "trigger_score")
_SPARSE_NULL_NODE_KEYS = ("agent", "task", "decay_reason")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_EVIDENCE_RECORD = re.compile(r"(?m)^- ([A-Za-z0-9]{12,}): ")


def snippet_max_chars_from_env() -> int:
    """Read ``LM_DELIVERY_SNIPPET_CHARS`` (0 disables snippeting, invalid -> default)."""

    return _chars_from_env(SNIPPET_CHARS_ENV, DEFAULT_SNIPPET_MAX_CHARS)


def context_value_max_chars_from_env() -> int:
    """Read ``LM_DELIVERY_CONTEXT_VALUE_CHARS`` (0 disables compaction, invalid -> default)."""

    return _chars_from_env(CONTEXT_VALUE_CHARS_ENV, DEFAULT_CONTEXT_VALUE_MAX_CHARS)


def provenance_value_max_chars_from_env() -> int:
    """Read ``LM_DELIVERY_PROVENANCE_VALUE_CHARS`` (0 disables, invalid -> default)."""

    return _chars_from_env(PROVENANCE_VALUE_CHARS_ENV, DEFAULT_PROVENANCE_VALUE_MAX_CHARS)


def snippet_ladder_from_env() -> tuple[int, ...] | None:
    """Read ``LM_DELIVERY_SNIPPET_LADDER`` into per-bearer-position budgets.

    Unset ladder: the default ladder — unless ``LM_DELIVERY_SNIPPET_CHARS`` is
    explicitly set, which selects the uniform legacy mode (``None``) so the
    pre-ladder contract (including ``0`` disables snippeting) survives.
    ``off``/``uniform``/``none``/``false``/``no`` force the uniform mode.
    Entries are ``full``/``complete`` (deliver complete content) or char
    budgets; any malformed entry falls back to the default ladder.
    """

    raw = os.environ.get(SNIPPET_LADDER_ENV, "").strip().lower()
    if not raw:
        if os.environ.get(SNIPPET_CHARS_ENV, "").strip():
            return None
        return DEFAULT_SNIPPET_LADDER
    if raw in _LADDER_OFF_FLAGS:
        return None
    entries: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part in ("full", "complete"):
            entries.append(LADDER_COMPLETE)
            continue
        try:
            value = int(part)
        except ValueError:
            return DEFAULT_SNIPPET_LADDER
        if value < 0:
            return DEFAULT_SNIPPET_LADDER
        entries.append(value)
    return tuple(entries) if entries else DEFAULT_SNIPPET_LADDER


def near_dup_cosine_from_env() -> float:
    """Read ``LM_RECALL_NEAR_DUP_COSINE`` (0 disables collapse, invalid -> default).

    The threshold the *caller* applies when it builds the duplicate map; this
    module never reads it, so the pure function keeps taking its map as an
    argument. Zero (or any negative) means "no near-dup map at all", which is
    the rollback path to byte-only dedup.
    """

    return _float_from_env(NEAR_DUP_COSINE_ENV, DEFAULT_NEAR_DUP_COSINE)


def near_dup_length_ratio_from_env() -> float:
    """Read ``LM_RECALL_NEAR_DUP_LENGTH_RATIO`` (default 0.2, invalid -> default).

    The excess fraction above which a candidate is too long to be a repeat of
    its bearer. ``0`` protects anything longer at all; the value is floored at
    0 because "collapse candidates that are shorter than the ratio allows" is
    not something this guard can mean.
    """

    return _float_from_env(NEAR_DUP_LENGTH_RATIO_ENV, DEFAULT_NEAR_DUP_LENGTH_RATIO)


def _chars_from_env(env_var: str, default: int) -> int:
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(0, value)


def _float_from_env(env_var: str, default: float) -> float:
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value != value:  # NaN compares false against every threshold
        return default
    return max(0.0, value)


def _enabled_from_env(env_var: str) -> bool:
    raw = os.environ.get(env_var, "").strip().lower()
    if not raw:
        return True
    return raw not in _DISABLED_FLAGS


def session_dedup_enabled_from_env() -> bool:
    """Read ``LM_DELIVERY_SESSION_DEDUP`` (default on; "0"/"false"/"no"/"off" disable)."""

    return _enabled_from_env(SESSION_DEDUP_ENV)


def full_node_diet_enabled_from_env() -> bool:
    """Read ``LM_DELIVERY_FULL_NODE_DIET`` (default on; "0"/"false"/"no"/"off" disable)."""

    return _enabled_from_env(FULL_NODE_DIET_ENV)


def stats_compaction_enabled_from_env() -> bool:
    """Read ``LM_DELIVERY_STATS_COMPACTION`` (default on; "0"/"false"/"no"/"off" disable)."""

    return _enabled_from_env(STATS_COMPACTION_ENV)


def sparse_entries_enabled_from_env() -> bool:
    """Read ``LM_DELIVERY_SPARSE`` (default on; "0"/"false"/"no"/"off" disable)."""

    return _enabled_from_env(SPARSE_ENV)


def shape_recall_results(
    results: Sequence[RecallResult],
    *,
    already_delivered_ids: set[str],
    snippet_max_chars: int,
    context_value_max_chars: int,
    session_dedup: bool,
    snippet_ladder: Sequence[int] | None = DEFAULT_SNIPPET_LADDER,
    full_node_diet: bool = True,
    provenance_value_max_chars: int = DEFAULT_PROVENANCE_VALUE_MAX_CHARS,
    stats_compaction: bool = True,
    sparse_entries: bool = True,
    duplicate_of: Mapping[str, str] | None = None,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """Render ranked recall results, deduplicating, snippeting, and dieting.

    New-lever defaults match production defaults; passing
    ``snippet_ladder=None, full_node_diet=False, provenance_value_max_chars=0,
    stats_compaction=False, sparse_entries=False`` restores the legacy
    (pre-diet-extension) renderer byte-for-byte.

    ``duplicate_of`` is the finished near-duplicate map — ``{duplicate node id:
    bearer node id}`` — normally produced by
    ``near_dup.build_duplicate_map`` over these same results in this same
    order. ``None`` (the default) is byte-only dedup, unchanged. Bearers are
    taken as given: the map's builder guarantees every value is a root, so
    nothing here chases a chain, and a bearer that is itself a stub of another
    class is left alone — the entry's ``content_ref`` is the honest re-fetch
    either way. Reading the map is the only thing this function does with it;
    the vectors, the threshold, and the database read that produced it all
    happen at the call site, which is what keeps this function pure.

    Pure: never mutates ``results``, their nodes, ``already_delivered_ids``, or
    ``duplicate_of``; identical inputs produce identical output. Output order
    matches input order.
    """

    bearer_by_content: dict[str, str] = {}
    # A withheld result's text is not delivered, so it bears nothing.
    withheld_ids = {
        result.node.id for result in results if getattr(result, "withheld", None)
    }
    shaped: list[dict[str, Any]] = []
    bearer_position = 0
    for result in results:
        node_dict = node_to_dict(result.node)
        node_id = node_dict["id"]
        content = node_dict["content"]
        full_chars = len(content)

        withheld = getattr(result, "withheld", None)
        twin_of: str | None = None
        if withheld:
            pass  # content not delivered: neither a twin nor a bearer
        elif content:  # empty content carries nothing worth deduplicating
            bearer_id = bearer_by_content.get(content)
            if bearer_id is None:
                bearer_by_content[content] = node_id
            else:
                twin_of = bearer_id
        near_dup_of = duplicate_of.get(node_id) if duplicate_of else None
        if near_dup_of in withheld_ids:
            near_dup_of = None

        # The bearer named on a stub's content_ref, if this result is one.
        stub_bearer: str | None = None
        if withheld:
            delivery = DELIVERY_BELOW_THRESHOLD
            node_dict["content"] = ""
        elif twin_of is not None:
            delivery = DELIVERY_TWIN_DUPLICATE
            stub_bearer = twin_of
            node_dict["content"] = _one_line_preview(content)
        elif session_dedup and node_id in already_delivered_ids:
            delivery = DELIVERY_SESSION_DUPLICATE
            node_dict["content"] = _one_line_preview(content)
        elif near_dup_of is not None:
            delivery = DELIVERY_NEAR_DUPLICATE
            stub_bearer = near_dup_of
            node_dict["content"] = _one_line_preview(content)
        else:
            budget = _bearer_budget(bearer_position, snippet_ladder, snippet_max_chars)
            bearer_position += 1
            # A schema is binding in its entirety. Evidence carriers are only
            # records of cases, so their long bodies can be selected by task.
            if result.node.level != "schema" and LADDER_COMPLETE < budget < full_chars:
                delivery = DELIVERY_SNIPPET
                node_dict["content"] = _relevant_excerpt(content, budget, query)
            else:
                delivery = DELIVERY_FULL

        dieted_full = False
        if delivery != DELIVERY_FULL:
            node_dict["provenance"] = _compact_provenance(
                node_dict["provenance"], provenance_value_max_chars
            )
            node_dict["context"] = _summarize_context(node_dict["context"], context_value_max_chars)
        elif full_node_diet:
            provenance = _compact_provenance(node_dict["provenance"], provenance_value_max_chars)
            context = _summarize_context(node_dict["context"], context_value_max_chars)
            dieted_full = (
                provenance != node_dict["provenance"] or context != node_dict["context"]
            )
            node_dict["provenance"] = provenance
            node_dict["context"] = context
        if stats_compaction:
            node_dict["stats"] = _trim_stats(node_dict["stats"])

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
            entry["content_ref"] = _content_ref(node_id, full_chars, duplicate_of=stub_bearer)
        elif dieted_full:
            # Content is complete; the hint marks that lookup returns strictly
            # more (unsummarized provenance/context) than was delivered.
            entry["content_ref"] = {"node_id": node_id, "fetch": _fetch_call(node_id)}
        if sparse_entries:
            _sparsify_entry(entry)
        shaped.append(entry)
    return shaped


def _relevant_excerpt(content: str, budget: int, query: str | None) -> str:
    """Select bounded source-labelled passages without changing the stored node.

    Case carriers use consolidation's ``- <source id>: <record>`` format. A
    selected passage retains both its source id and the evidence-only heading.
    Other long text uses paragraphs. Ties retain original order; no match
    falls back to the beginning and the content_ref exposes the full text.
    """

    if not query:
        return _truncate_at_boundary(content, budget)
    record_scores: list[tuple[int, int]] = []
    if content.startswith("Evidence: ") and (matches := list(_EVIDENCE_RECORD.finditer(content))):
        heading = content[: matches[0].start()].strip()
        units = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            source_id = match.group(1)
            body = content[match.end() : end].strip()
            for passage in _passages(body):
                units.append((f"- {source_id}: ", passage, index))
    else:
        heading = ""
        units = [("", passage, index) for index, passage in enumerate(_passages(content))]
    if not units:
        return _truncate_at_boundary(content, budget)

    query_words = [word.casefold() for word in _WORD.findall(query or "") if len(word) >= 3]
    words = set(query_words)
    query_pairs = set(zip(query_words, query_words[1:]))

    if heading:
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            record_words = [word.casefold() for word in _WORD.findall(content[match.end() : end])]
            record_scores.append((
                len(words.intersection(record_words)),
                len(query_pairs.intersection(zip(record_words, record_words[1:]))),
            ))

    def relevance(index: int) -> tuple[int, int, int, int, int]:
        passage_words = [word.casefold() for word in _WORD.findall(units[index][1])]
        matched = words.intersection(passage_words)
        pairs = set(zip(passage_words, passage_words[1:]))
        # A source record may establish the topic in one passage and state its
        # applicable constraint in the next. Rank by the record's relationship
        # to the task before ranking fragments of that same record.
        record_score = record_scores[units[index][2]] if record_scores else (0, 0)
        return (
            *record_score,
            len(matched),
            len(query_pairs & pairs),
            -index,
        )

    scored = sorted(
        range(len(units)),
        key=relevance,
        reverse=True,
    )
    remaining = budget - len(heading) - (1 if heading else 0)
    picked: dict[int, str] = {}
    for index in scored:
        prefix, passage, _ = units[index]
        allowance = remaining - len(prefix) - (1 if picked else 0)
        if allowance < 80:
            break
        excerpt = passage if len(passage) <= allowance else _truncate_at_boundary(passage, allowance)
        picked[index] = prefix + excerpt
        remaining -= len(picked[index]) + (1 if len(picked) > 1 else 0)
        if remaining < 80:
            break
    selected = "\n".join(picked[index] for index in sorted(picked))
    return f"{heading}\n{selected}" if heading else selected


def _passages(text: str) -> list[str]:
    """Split long records at readable boundaries, keeping short records whole."""

    chunks = re.split(r"\n\s*\n", text)
    passages: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) > 900:
            for line in chunk.splitlines():
                line = line.strip()
                passages.extend(line[start : start + 700] for start in range(0, len(line), 700))
        elif chunk:
            passages.append(chunk)
    return passages


def _bearer_budget(
    position: int, snippet_ladder: Sequence[int] | None, snippet_max_chars: int
) -> int:
    if snippet_ladder is None or not snippet_ladder:
        return snippet_max_chars
    return snippet_ladder[min(position, len(snippet_ladder) - 1)]


def _fetch_call(node_id: str) -> str:
    return f'memory_lookup(node_id="{node_id}")'


def _content_ref(
    node_id: str, full_content_chars: int, *, duplicate_of: str | None
) -> dict[str, Any]:
    ref: dict[str, Any] = {
        "node_id": node_id,
        "fetch": _fetch_call(node_id),
        "full_content_chars": full_content_chars,
    }
    if duplicate_of is not None:
        ref["duplicate_of"] = duplicate_of
    return ref


def _compact_provenance(provenance: dict[str, Any], max_value_chars: int) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in provenance.items():
        if key == "prior_recalls" and isinstance(value, list) and value:
            compacted[key] = {"count": len(value)}
        elif max_value_chars <= 0:  # rollback valve: legacy prior_recalls only
            compacted[key] = value
        elif key == "corrections" and isinstance(value, list):
            compacted[key] = _compact_corrections(value, max_value_chars)
        else:
            compacted[key] = _summarize_context_value(value, max_value_chars)
    return compacted


def _compact_corrections(corrections: list[Any], max_value_chars: int) -> list[Any]:
    """Bound correction texts while keeping the supersedes signal structured.

    One entry per correction survives with its full key set — short values
    (corrector, timestamps, ids) verbatim — and only oversized texts are
    truncated; nested lists/objects compact like context values.
    """

    compacted: list[Any] = []
    for entry in corrections:
        if isinstance(entry, dict):
            compacted.append(
                {key: _summarize_context_value(value, max_value_chars) for key, value in entry.items()}
            )
        else:
            compacted.append(_summarize_context_value(entry, max_value_chars))
    return compacted


def _trim_stats(stats: dict[str, Any]) -> dict[str, Any]:
    trimmed: dict[str, Any] = {}
    for key, value in stats.items():
        if key == "last_accessed":  # bookkeeping timestamp, never actionable
            continue
        if key == "temporal_hint" and value is None:
            continue
        if key == "confidence" and value == 0.5:  # storage default
            continue
        if key == "unique_agents" and (value or 0) <= 1:
            continue
        if key == "access_count" and not value:
            continue
        if key == "usefulness_score":
            if not value:
                continue
            trimmed[key] = _round_score(value)
            continue
        trimmed[key] = value
    return trimmed


def _sparsify_entry(entry: dict[str, Any]) -> None:
    """Drop null/duplicate fields; mutates only dicts built by shaping.

    Score fields stay present — zeros included — so wire consumers (ranking
    tests, agents parsing component scores) never need existence checks;
    only their float noise is trimmed.
    """

    for key in _SPARSE_SCORE_KEYS:
        entry[key] = _round_score(entry[key])
    if not entry["path"]:
        del entry["path"]
    del entry["recall_event_id"]  # the envelope carries it once
    node_dict = entry["node"]
    for key in _SPARSE_NULL_NODE_KEYS:
        if node_dict[key] is None:
            del node_dict[key]
    if node_dict["decayed"] is False:
        del node_dict["decayed"]
    created_at = node_dict["created_at"]
    if node_dict["timestamp"] == created_at:
        del node_dict["timestamp"]
    if node_dict["updated_at"] == created_at:
        del node_dict["updated_at"]
    ref = entry.get("content_ref")
    if ref is not None:
        ref.pop("fetch", None)


def _round_score(value: Any) -> Any:
    # 6 decimals: ±5e-7 fidelity, far inside the 1e-5 tolerances score
    # consumers use, while still cutting 17-digit float tails.
    if isinstance(value, float):
        return round(value, 6)
    return value


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
    # non-ASCII context (e.g. Cyrillic) ~6x its delivered size.
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
