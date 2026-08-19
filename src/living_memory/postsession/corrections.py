"""Turn "a recalled fact contradicted reality" into proposed ``memory_teach`` ops.

Why this module exists
----------------------
``memory_teach`` is nearly dead in the field: 2 calls across 30 goal-tree
sessions, and the whole database holds 434 ``supersedes`` edges against 139,696
``related`` ones. Only :func:`living_memory.consolidation.memory_teach` creates
the ``supersedes`` edge, downgrades the original's confidence to ``≤0.25`` and
its usefulness to ``≤-0.25`` (``consolidation.py:516``); a ``memory_remember``
trace whose text merely *says* "CORRECTION" creates nothing of the sort, and the
stale node keeps outranking reality in every future recall.

So a stale node only stops winning if somebody calls teach. Agents in-session
mostly do not. This module does it offline, from the transcript.

No database access is needed
----------------------------
The delivered node **content** is already in the transcript --
``mcpMeta.structuredContent.results[].node.content`` for Claude,
``result.Ok.structuredContent`` for codex -- and
:mod:`living_memory.postsession.transcripts` has already lifted it into
:class:`~living_memory.postsession.session.DeliveredNode`. Contradiction
detection therefore reads one :class:`SessionRecord` and nothing else. This
module opens no socket, no database and no file outside the transcript the
caller hands it, and it **writes nothing to Living Memory**: it emits
:class:`~living_memory.postsession.session.ProposedOp` values with
``kind="teach"`` and stops there.

The pipeline
------------
1. **Pair.** For every :class:`RecallInteraction`, pair each delivered node with
   the session's *later* observations -- command outputs, test results, file
   diffs, operator messages. Assistant prose is deliberately **not** an
   observation: an agent's own narration is a claim, not reality, and letting it
   contradict memory is how a confident-but-wrong agent would overwrite a
   correct node. See :func:`observations`.
2. **Rank and bound.** Score each pairing by shared *anchors* (paths,
   identifiers, numbers, flags -- see :func:`anchors`) with a session-local
   inverse-frequency weight, keep the top ``max_observations_per_node``, and
   drop pairs below ``min_pair_score``. This is a cost bound, not a verdict; the
   run report records every pair it dropped and why.
3. **Judge.** One narrow question per (recall, node): *does a concrete
   observation in this session contradict this recalled fact?* Not "is it
   incomplete", not "does it sound stale". The judge must return the
   contradicted claim, the contradicting observation quote and its locator, or
   answer ``contradicted: false``. See :data:`TASK` and :data:`SCHEMA`.
4. **Veto.** :func:`validate_verdict` can overrule the judge and does so on
   purely mechanical grounds. Every rejection carries a machine-readable reason
   (:data:`REASONS`), so a run report says *why* the detector stayed silent
   instead of only that it did.
5. **Emit.** :func:`teach_op` builds the ``ProposedOp`` whose payload is exactly
   the ``memory_teach(store, trace_id, correction, confidence=None,
   context=None)`` call the runner will make.

What the veto actually enforces
-------------------------------
``trace_id`` names a node this session was really shown; ``contradicted_claim``
occurs verbatim in that node's delivered content; ``observation_quote`` occurs
verbatim in the session record at the stated locator, in an observation that
came *after* the recall; claim and quote share at least one anchor (a fact about
X can only be contradicted by an observation about X); the quote is not an
absence of evidence; the correction is not a rephrasing of the claim; the
correction is self-contained; and the context carries
``lesson_kind="correction"``.

Verbatim, precisely
-------------------
The judge is shown redacted text -- :func:`living_memory.postsession.judge.
default_redactor` rewrites ``/home/<user>`` to ``~`` and blanks secret-shaped
substrings -- so a returned quote is verbatim with respect to *that* rendering.
The validator classifies every quote and keeps only two modes:

``exact``
    the quote occurs byte-for-byte in the session record;
``home_path_normalized``
    it occurs byte-for-byte once ``/home/<user>`` (and ``/root``) is rewritten
    to ``~``. Path elision is lossless for the claim and never invents content.

A quote that only matches after *secret* redaction is rejected
(``quote_touches_redacted_secret``): evidence next to a credential is evidence
this stage should not be quoting into a tracked artifact anyway.

What was tuned, and on what
---------------------------
Thresholds, the prompt and the veto set were tuned on the ``train`` split only.
The measurement that made tuning possible is the corpus's own labelled
positives: 58 train sessions in which the working agent was delivered a node,
found it wrong, and called ``memory_teach`` on it (:func:`taught_delivered_nodes`),
73 corrected nodes in all. What the tuning rounds found:

* every judge-confirmed contradiction was being thrown away as
  ``unknown_locator``, because the judge correctly echoed the *redacted*
  locator it had been shown and the validator compared it to the unredacted key
  (:func:`resolve_locator`);
* quotes were failing the verbatim check because the payload had grown past the
  prompt builder's byte budget and been truncated mid-excerpt
  (``max_node_content_chars``);
* the aboutness veto was killing true findings whose *claim clause* named
  nothing concrete, so the prompt now requires the claim and the quote to carry
  a shared concrete token, and a claim that names nothing is reported as
  ``claim_has_no_anchor`` rather than silently as a mismatch;
* and then the finding that invalidated the three above -- **the detector was
  being scored on its own input**. 110 of the 249 observations it put in front
  of the judge on those labelled positives, 44%, were Living Memory round trips
  rather than observed reality, because a large part of this corpus calls the
  memory server from *inside* a script tool and only the script call carries the
  result text (:func:`is_memory_roundtrip`). The worst shape was the exact one
  this module exists to prevent: a session's own ``memory_teach`` payload,
  echoed back in the tool result, quoted as the evidence that the taught node
  was wrong. Every headline number measured before that exclusion was inflated
  by it, so none of them are quoted here;
* and, once the echoes were gone, the judge was still being shown windows that
  could not answer the question. 24% of the observations reaching it carried
  *none* of the anchors that had earned the pairing, because a codex ``exec``
  result is a single JSON line and the line-aligned excerpt scan degenerated to
  the host's ``Script completed / Warning: truncated output`` header
  (:func:`_character_window`). Fixing the excerpt alone took grounded recovery
  on the train positives from 2 of 55 to 6 of 55.

Honest numbers, after both fixes, on ``train``: 55 of 73 labelled-positive
nodes reach the judge (51 of 58 sessions; the misses have no *real* observation
sharing an anchor with the node), 3% of shown observations are anchor-blind,
and 6 of those 55 survive the judge and the whole veto layer. That rate is low
and is meant to be: most
in-session corrections are an agent's synthesis across many signals, not a
single quotable observation that mechanically contradicts a stored claim, and
this module only emits the latter. A wrong ``supersedes`` edge costs more than a
missed one -- teach drops the target's confidence to ``<=0.25`` -- so every
ambiguous case is spent on silence.

The ``eval`` split was read with those values frozen; the reserved split was
never read, and :func:`_load_entries` excludes any union-find group that could
belong to it.

Negative controls travel with the detector
------------------------------------------
:data:`NEGATIVE_CONTROLS` holds three synthetic sessions -- the recalled fact
matched reality, the session never touched the fact, and the session corrected a
*different* fact that was never recalled. They live here rather than in the test
file because they are part of the published behaviour: the eval run replays them
through the same real judge and records the outcome next to the eval numbers.
All three must yield zero teach ops.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

from .judge import (
    Judge,
    JudgeError,
    JudgeInvalidOutput,
    JudgeRefused,
    JudgeUnavailable,
    PromptConfig,
    canonical_json,
    default_redactor,
    sha256_text,
)
from .session import (
    RECALL_TOOL,
    TEACH_TOOL,
    DeliveredNode,
    Evidence,
    FileMutation,
    ProposedOp,
    RecallInteraction,
    SessionRecord,
    SourceSpan,
    ToolCall,
    Turn,
)

__all__ = [
    "ABSENCE_PATTERNS",
    "mentions_tool",
    "mask_reserved_splits",
    "reserved_splits",
    "Candidate",
    "DEFAULT_CONFIG",
    "DETECTOR_VERSION",
    "DetectionResult",
    "DetectorConfig",
    "NEGATIVE_CONTROLS",
    "NegativeControl",
    "Observation",
    "REASONS",
    "SCHEMA",
    "TASK",
    "Verdict",
    "anchors",
    "best_excerpt",
    "build_payload",
    "candidates",
    "detect_corrections",
    "is_memory_roundtrip",
    "main",
    "observations",
    "quote_match_mode",
    "resolve_locator",
    "recover_known_corrections",
    "run_negative_controls",
    "taught_delivered_nodes",
    "teach_op",
    "validate_verdict",
]

#: Bumped whenever the prompt, the schema or a veto rule changes. Recorded in
#: every op's context so a later audit can tell which policy produced it.
DETECTOR_VERSION = "corrections/1"

#: ``memory_teach`` context marker demanded by the goal contract. The runner
#: refuses to execute a teach op whose context does not carry it.
LESSON_KIND = "correction"

#: ``context.agent``: ``memory_teach`` uses it as the teacher, so this is the
#: attribution that ends up on the corrective trace and on the supersedes edge.
EXTRACTOR_AGENT = "extractor"


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Every knob, in one place, so a run report can print the exact policy.

    The defaults were tuned on the ``train`` split only; the ``eval`` split is
    read once, with these values frozen.
    """

    #: Judge calls allowed per session. A session that delivers 84 nodes would
    #: otherwise dominate a run's cost. Excess pairs are reported, never hidden.
    max_nodes_per_session: int = 12
    #: Observations shown per node.
    max_observations_per_node: int = 5
    #: Minimum anchor-overlap score for a (node, observation) pair to be shown.
    min_pair_score: float = 1.2
    #: Minimum number of distinct shared anchors for a pair to be shown.
    min_shared_anchors: int = 2
    #: Characters of each observation shown to the judge.
    excerpt_chars: int = 1_200
    #: Characters of the stored fact shown to the judge. The whole payload has
    #: to stay under ``max_payload_bytes`` *before* the judge's own bounding
    #: runs: bounding cuts the longest string and leaves a marker in it, and a
    #: quote taken across that marker can never be verified verbatim.
    max_node_content_chars: int = 6_000
    #: Hard cap on the raw text kept per observation.
    max_observation_chars: int = 40_000
    #: Shortest usable delivered-node content. Below this a "fact" is a stub.
    min_node_content_chars: int = 40
    #: Judge confidence below which an otherwise valid verdict is dropped.
    min_confidence: float = 0.6
    #: Self-containedness bounds for the correction text.
    min_correction_chars: int = 100
    max_correction_chars: int = 900
    #: Shortest acceptable contradicting quote.
    min_quote_chars: int = 15
    #: Prompt byte budget. Small on purpose: the payload is five bounded
    #: excerpts plus one node, and a truncation marker in the middle of an
    #: excerpt would break the verbatim check.
    max_payload_bytes: int = 24_000
    #: Wall-clock budget for one judge call, retries included.
    judge_timeout_s: float = 180.0
    #: Skip a node the session already taught. True in production -- teaching
    #: it again would create a second supersedes edge for a correction that
    #: already landed. Turned off only by the ground-truth recovery check,
    #: where those nodes are precisely the known positives.
    skip_already_taught: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_nodes_per_session": self.max_nodes_per_session,
            "max_observations_per_node": self.max_observations_per_node,
            "min_pair_score": self.min_pair_score,
            "min_shared_anchors": self.min_shared_anchors,
            "excerpt_chars": self.excerpt_chars,
            "max_node_content_chars": self.max_node_content_chars,
            "max_observation_chars": self.max_observation_chars,
            "min_node_content_chars": self.min_node_content_chars,
            "min_confidence": self.min_confidence,
            "min_correction_chars": self.min_correction_chars,
            "max_correction_chars": self.max_correction_chars,
            "min_quote_chars": self.min_quote_chars,
            "max_payload_bytes": self.max_payload_bytes,
            "judge_timeout_s": self.judge_timeout_s,
            "skip_already_taught": self.skip_already_taught,
        }


DEFAULT_CONFIG = DetectorConfig()


# --------------------------------------------------------------------------
# anchors
# --------------------------------------------------------------------------

# One alternation, ordered longest-match-first. An "anchor" is a token concrete
# enough that two texts sharing it are plausibly about the same thing: a path, a
# dotted or namespaced identifier, a flag, a number, an identifier with internal
# structure. Prose words are not anchors.
_ANCHOR_PATTERNS: tuple[re.Pattern[str], ...] = (
    # paths and file names: src/living_memory/x.py, ./scripts/run.sh, a/b/c
    re.compile(r"(?:\.{0,2}/)?(?:[\w.@+-]+/){1,}[\w.@+-]+"),
    re.compile(r"\b[\w-]+\.(?:py|md|json|jsonl|toml|yaml|yml|txt|sh|sql|cfg|ini|ts|tsx|js|rs|go|c|h|cpp|hpp|lock)\b"),
    # namespaced identifiers: module.attr, Class::method, table:column
    re.compile(r"\b\w+(?:::\w+)+\b"),
    re.compile(r"\b[A-Za-z_]\w*(?:\.\w+)+\b"),
    # long-option flags: --strict-mcp-config, -Werror
    re.compile(r"(?<![\w-])--?[A-Za-z][\w-]{2,}"),
    # identifiers with internal structure: snake_case, CamelCase, ULIDs
    re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"),
    re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b"),
    # The tail is [a-z0-9]*, never [A-Za-z0-9]*: an uppercase-accepting tail
    # overlaps the group's own [A-Z] start, so every way of splitting a run of
    # letters is a distinct path and a token that ultimately fails \b backtracks
    # exponentially. One real 3.2 KB command output in the eval split hung this
    # pattern for over four minutes; with the tail restricted the same input
    # takes 0.000s and matches identically (getURL, parseHTTPResponse, fooBar123
    # all still match, because each capital may open a group with an empty tail).
    re.compile(r"\b[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+\b"),
    re.compile(r"`([^`\n]{2,60})`"),
    re.compile(r"\b[0-9A-HJKMNP-TV-Z]{26}\b"),
    # quantities: 434, 0.25, 139_696, 12ms, 3.11
    re.compile(r"\b\d[\d_,]*(?:\.\d+)?[a-zA-Z%]{0,3}\b"),
)

# Structural noise that would otherwise pair everything with everything:
# ubiquitous words that happen to match the identifier shapes above.
_ANCHOR_STOPWORDS = frozenset(
    {
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "10",
        "100",
        "self",
        "true",
        "false",
        "none",
        "null",
        "return",
        "import",
        "from",
        "def",
        "class",
        "print",
        "value",
        "values",
        "result",
        "results",
        "data",
        "text",
        "name",
        "type",
        "index",
        "item",
        "items",
        "list",
        "dict",
        "str",
        "int",
        "float",
        "bool",
        "e.g",
        "i.e",
        "etc",
        "u.s",
        "no.",
    }
)

_MIN_ANCHOR_CHARS = 3


def anchors(text: str | None) -> set[str]:
    """Concrete tokens ``text`` is *about*, lowercased and de-punctuated.

    Two texts sharing an anchor are plausibly about the same thing; two texts
    sharing none are not, which is the whole basis of the "aboutness" veto in
    :func:`validate_verdict`. Deliberately recall-oriented and cheap: a false
    anchor costs one wasted judge call, a missed anchor costs a real detection.
    """

    if not text:
        return set()
    found: set[str] = set()
    for pattern in _ANCHOR_PATTERNS:
        for raw in pattern.findall(text):
            token = raw.strip().strip(".,;:!?)(][}{\"'`").lower()
            token = token.rstrip(".")
            if len(token) < _MIN_ANCHOR_CHARS or token in _ANCHOR_STOPWORDS:
                continue
            if token.isdigit() and len(token) < 2:
                continue
            found.add(token)
    return found


def _anchor_weights(documents: Sequence[set[str]]) -> dict[str, float]:
    """Session-local inverse document frequency over ``documents``.

    An anchor appearing in every command output of a session (the repo path,
    the test runner's name) says nothing about aboutness; one appearing twice
    says a lot. Weight is ``1 / (1 + ln(df))`` -- bounded, monotone, no tuning.
    """

    total: dict[str, int] = {}
    for document in documents:
        for token in document:
            total[token] = total.get(token, 0) + 1
    return {token: 1.0 / (1.0 + math.log(count)) for token, count in total.items()}


_WORD_RE = re.compile(r"[\w']+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# --------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------

#: Observation channels, in the order the goal names them.
OBSERVATION_KINDS: tuple[str, ...] = (
    "command_output",
    "test_result",
    "file_diff",
    "operator_message",
)

_TEST_MARKERS = re.compile(
    r"\b(?:pytest|py\.test|unittest|jest|vitest|mocha|cargo test|go test|ctest|tox|nox)\b"
    r"|\b\d+\s+(?:passed|failed|error(?:s|ed)?|skipped)\b"
    r"|\bFAILED\b|\bPASSED\b|\bassert(?:ion)?\s+(?:failed|error)\b",
    re.IGNORECASE,
)

# Host-injected pseudo-turns that are not the operator speaking.
_INJECTED_TURN = re.compile(
    r"^\s*(?:<(?:system-reminder|command-name|command-message|local-command|"
    r"user-prompt-submit-hook|environment_details)\b|\[Request interrupted|"
    r"Caveat: The messages below|<user_instructions>|<environment_context>)",
    re.IGNORECASE,
)

_OPERATOR_ROLES = frozenset({"user", "operator", "human"})

# A Living Memory round trip, however the session spelled it. ``ToolCall.server``
# only catches the calls the host recorded *as* MCP calls; a large part of this
# corpus reaches the same server from inside a script tool instead --
# ``exec: const r = await tools.mcp__living_memory__memory_teach({...})`` -- and
# those arrive as an ordinary ``exec`` call with ``server=None``.
#
# The first alternative is the server-qualified name in any of its spellings;
# the second is a bare call opening an object literal or a multi-line argument
# list, which is a call and not a mention. Both deliberately fail to match
# ``grep -n "memory_teach(" src/`` -- a quoted string follows the paren there --
# because greping this repository's own source for these names is a legitimate
# observation and must stay one.
_MEMORY_INVOCATION = re.compile(
    r"living[-_]memory(?:__|\.)memory_\w+\s*\("
    r"|\bmemory_(?:recall|remember|teach|lookup|consolidate|forget|connect|status|health)"
    r"\s*\(\s*[\{\n]"
)

# The server's own response envelope, for a round trip whose call text the
# patterns above did not recognise. Both alternatives demand a *value* shape no
# prose produces: a ULID after ``recall_event_id``, an object after
# ``corrective_trace``. ``\\*`` absorbs the JSON escaping a nested transcript
# applies, so the same pattern matches the raw envelope and a doubly-encoded one.
_MEMORY_ENVELOPE = re.compile(
    r"\\*\"recall_event_id\\*\"\s*:\s*\\*\"[0-9A-HJKMNP-TV-Z]{26}"
    r"|\\*\"corrective_trace\\*\"\s*:\s*\\*[{\"]"
)

#: How much of a result is scanned for an envelope. The envelope is a response
#: header, so it is at the front; scanning megabytes of build log for it is pure
#: cost.
_ENVELOPE_SCAN_CHARS = 20_000


def is_memory_roundtrip(call: ToolCall) -> bool:
    """Whether ``call`` is a Living Memory call rather than an observation.

    This is the single most important exclusion in the module. Memory content is
    not observed reality: if a recall envelope may be an "observation", one
    stored node can be made to contradict another, and -- the case that made
    this function exist -- a session's own ``memory_teach`` payload, echoed back
    in the tool result, can be quoted as the evidence that the taught node was
    wrong. That is circular: the detector would be "discovering" the correction
    the agent had already written, and scoring itself for it.

    ``ToolCall.is_memory_tool`` alone does not catch that, because the
    normalizer records the script call that *carried* the memory call as a
    separate ordinary tool call, and only the script call has the result text.
    """

    if call.is_memory_tool:
        return True
    if _MEMORY_INVOCATION.search(canonical_json(call.arguments)):
        return True
    if call.result_text and _MEMORY_ENVELOPE.search(call.result_text[:_ENVELOPE_SCAN_CHARS]):
        return True
    return False


@dataclass(frozen=True, slots=True)
class Observation:
    """One piece of observed reality, with the locator that proves where it is.

    ``text`` is the *raw* session text; it is what every verbatim check runs
    against. ``locator`` is unique within a record because the field path names
    the collection and the ordinal, so a judge cannot point at "record 42" and
    leave the validator guessing which of its three tool results was meant.
    """

    kind: str
    locator: str
    label: str
    text: str
    record_index: int
    ordinal: int

    @property
    def anchors(self) -> set[str]:
        """Anchors of the observation *and* of what produced it.

        The command line is part of what an output is about: ``139696`` alone
        says nothing, ``sqlite3 global.sqlite3 '.backup ...'`` says which
        database. The label is never quotable -- only :attr:`text` is -- so this
        widens *pairing*, never the verbatim guarantee.
        """

        return anchors(self.text) | anchors(self.label)

    @property
    def label_anchors(self) -> set[str]:
        return anchors(self.label)


def _observation_span(record: SessionRecord, base: SourceSpan, field_path: str) -> str:
    return SourceSpan(record.source, record.path, base.record_index, field_path).locator()


def observations(
    record: SessionRecord, *, config: DetectorConfig = DEFAULT_CONFIG
) -> list[Observation]:
    """Every observable fact the session produced, ordered by record index.

    Four channels and no fifth. Assistant prose is excluded on purpose: it is
    the agent's claim about reality, not reality, and a detector that let it
    contradict memory would happily supersede a correct node because a confused
    agent said so. Memory round trips are excluded for a stronger reason still
    -- see :func:`is_memory_roundtrip`.
    """

    found: list[Observation] = []

    for call in record.tool_calls:
        if not call.result_text or is_memory_roundtrip(call):
            continue
        text = call.result_text[: config.max_observation_chars]
        command = _command_of(call.arguments)
        label = f"{call.name}: {command}" if command else call.name
        kind = (
            "test_result"
            if _TEST_MARKERS.search(command) or _TEST_MARKERS.search(text[:4000])
            else "command_output"
        )
        found.append(
            Observation(
                kind=kind,
                locator=_observation_span(
                    record, call.span, f"tool_calls[{call.ordinal}].result_text"
                ),
                label=label[:200],
                text=text,
                record_index=call.span.record_index,
                ordinal=call.ordinal,
            )
        )

    for mutation in record.file_mutations:
        text = _mutation_text(mutation)
        if not text:
            continue
        found.append(
            Observation(
                kind="file_diff",
                locator=_observation_span(
                    record, mutation.span, f"file_mutations[{mutation.ordinal}]"
                ),
                label=f"{mutation.kind} {mutation.path}"[:200],
                text=text[: config.max_observation_chars],
                record_index=mutation.span.record_index,
                ordinal=mutation.ordinal,
            )
        )

    for turn in record.turns:
        if turn.role not in _OPERATOR_ROLES:
            continue
        text = (turn.text or "").strip()
        if not text or _INJECTED_TURN.match(text):
            continue
        found.append(
            Observation(
                kind="operator_message",
                locator=_observation_span(record, turn.span, f"turns[{turn.index}].text"),
                label="operator message",
                text=text[: config.max_observation_chars],
                record_index=turn.span.record_index,
                ordinal=turn.index,
            )
        )

    found.sort(key=lambda item: (item.record_index, item.kind, item.ordinal))
    return found


def _command_of(arguments: Mapping[str, Any]) -> str:
    for key in ("command", "cmd", "script", "input", "query", "file_path", "path"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
            return " ".join(value)
    return ""


def _mutation_text(mutation: FileMutation) -> str:
    if mutation.unified_diff:
        return mutation.unified_diff
    if mutation.post_image:
        return mutation.post_image
    return ""


#: Occurrences scanned per anchor in the character-window fallback. A token
#: repeated thousands of times in one blob says nothing about *where* the
#: interesting region is, so counting past this is pure cost.
_MAX_OCCURRENCES_PER_ANCHOR = 200


def _character_window(text: str, wanted: set[str], limit: int) -> tuple[str, int]:
    """The ``limit``-char window covering the most distinct ``wanted`` anchors.

    The line-aligned scan cannot help when the text has no usable line breaks,
    and a great deal of this corpus has none: a codex ``exec`` result is one
    JSON line whose newlines are ``\\n`` *escapes*, routinely tens of kilobytes
    of it. For those, line alignment degenerated to "the first ``limit``
    characters", which is the host's own ``Script completed / Wall time /
    Warning: truncated output`` header -- so 24% of the observations put in
    front of the judge on the train positives contained none of the anchors
    that had earned the pairing in the first place. The judge was being asked
    whether a fact was contradicted by a window that could not mention it.

    Still contiguous, so a quote taken from here is still a substring of the
    raw text and the verbatim check keeps its meaning.
    """

    low = text.lower()
    hits: list[tuple[int, str]] = []
    for token in wanted:
        needle = token.lower()
        start = 0
        for _ in range(_MAX_OCCURRENCES_PER_ANCHOR):
            found = low.find(needle, start)
            if found < 0:
                break
            hits.append((found, token))
            start = found + max(1, len(needle))
    if not hits:
        return text[:limit], 0

    hits.sort()
    counts: dict[str, int] = {}
    distinct = 0
    best_distinct = -1
    best_start = 0
    low_index = 0
    for position, token in hits:
        counts[token] = counts.get(token, 0) + 1
        if counts[token] == 1:
            distinct += 1
        # shrink from the left until every retained occurrence fits one window
        while hits[low_index][0] <= position - limit:
            dropped = hits[low_index][1]
            counts[dropped] -= 1
            if counts[dropped] == 0:
                distinct -= 1
            low_index += 1
        if distinct > best_distinct:
            best_distinct = distinct
            best_start = hits[low_index][0]

    # a little lead-in, then clamp so the window stays inside the text
    start = max(0, min(best_start - limit // 8, len(text) - limit))
    return text[start : start + limit], start


def best_excerpt(text: str, wanted: set[str], limit: int) -> tuple[str, int]:
    """The ``limit``-char window of ``text`` densest in ``wanted`` anchors.

    Returns ``(excerpt, offset)``. Line-aligned so a diff or a traceback stays
    readable, and contiguous so any quote taken from it is also contiguous in
    the raw text -- which is what makes the verbatim check meaningful. When line
    alignment cannot surface the anchors -- an unbroken JSON blob, or a window
    that lands nowhere near them -- :func:`_character_window` takes over.
    """

    if len(text) <= limit:
        return text, 0
    lines = text.splitlines(keepends=True)
    lengths = [len(line) for line in lines]
    hits = [len(wanted & anchors(line)) for line in lines]

    best_score = -1
    best_range = (0, 0)
    start = 0
    window_len = 0
    window_hits = 0
    for end, (length, hit) in enumerate(zip(lengths, hits)):
        window_len += length
        window_hits += hit
        while window_len > limit and start < end:
            window_len -= lengths[start]
            window_hits -= hits[start]
            start += 1
        if window_len <= limit and window_hits > best_score:
            best_score = window_hits
            best_range = (start, end + 1)

    first, last = best_range
    offset = sum(lengths[:first])
    excerpt = "".join(lines[first:last])
    if wanted and best_score <= 0:
        # either one line longer than the whole budget, or no line-aligned
        # window carries an anchor: fall back to the character scan
        return _character_window(text, wanted, limit)
    if not excerpt:
        return text[:limit], 0
    return excerpt, offset


# --------------------------------------------------------------------------
# candidate pairing
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    """One (recall, delivered node) pairing with the observations to test it."""

    recall: RecallInteraction
    node: DeliveredNode
    scored: tuple[tuple[Observation, float, tuple[str, ...]], ...]

    @property
    def score(self) -> float:
        return max((item[1] for item in self.scored), default=0.0)

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(item[0] for item in self.scored)


def candidates(
    record: SessionRecord,
    *,
    config: DetectorConfig = DEFAULT_CONFIG,
    counters: dict[str, int] | None = None,
    only_nodes: Iterable[str] | None = None,
) -> list[Candidate]:
    """Rank the (node, observation) pairings worth a judge call.

    Ordering is by best pair score descending, then by recall/node ordinal, so
    the per-session cap spends the budget on the most plausible contradictions
    and the result is deterministic for a given record.
    """

    tally = counters if counters is not None else {}
    observed = observations(record, config=config)
    if not observed:
        _bump(tally, "no_observations", len(record.delivered_node_ids))
        return []

    # ``Observation.anchors`` is a property that re-runs every anchor pattern
    # over the observation's text, and that text can be 40 KB. Extract once and
    # feed both consumers from the same table: the previous two-comprehension
    # form scanned every observation in the session twice, which on a
    # multi-megabyte transcript is minutes of pure regex, not milliseconds.
    obs_anchors = {item.locator: item.anchors for item in observed}
    weights = _anchor_weights(list(obs_anchors.values()))
    superseded = {
        write.supersedes for write in record.writes if write.kind == "teach" and write.supersedes
    }

    wanted = set(only_nodes) if only_nodes is not None else None
    seen: set[str] = set()
    built: list[Candidate] = []
    for recall in record.recalls:
        for node in recall.delivered:
            if wanted is not None and node.node_id not in wanted:
                continue
            if node.node_id in seen:
                _bump(tally, "duplicate_delivery")
                continue
            seen.add(node.node_id)
            content = (node.content or "").strip()
            if len(content) < config.min_node_content_chars:
                _bump(tally, "node_content_missing")
                continue
            if config.skip_already_taught and node.node_id in superseded:
                _bump(tally, "already_taught_in_session")
                continue
            node_anchors = anchors(content)
            if not node_anchors:
                _bump(tally, "node_has_no_anchors")
                continue

            scored: list[tuple[Observation, float, tuple[str, ...]]] = []
            for item in observed:
                if item.record_index <= recall.record_index:
                    continue
                shared = node_anchors & obs_anchors[item.locator]
                if len(shared) < config.min_shared_anchors:
                    continue
                score = sum(weights.get(token, 1.0) for token in shared)
                if score < config.min_pair_score:
                    continue
                scored.append((item, score, tuple(sorted(shared))))
            if not scored:
                _bump(tally, "no_related_observation")
                continue
            scored.sort(key=lambda entry: (-entry[1], entry[0].record_index))
            built.append(
                Candidate(
                    recall=recall,
                    node=node,
                    scored=tuple(scored[: config.max_observations_per_node]),
                )
            )

    built.sort(key=lambda item: (-item.score, item.recall.ordinal, item.node.rank))
    if len(built) > config.max_nodes_per_session:
        _bump(tally, "over_session_cap", len(built) - config.max_nodes_per_session)
        built = built[: config.max_nodes_per_session]
    return built


def _bump(counters: dict[str, int], key: str, amount: int = 1) -> None:
    counters[key] = counters.get(key, 0) + amount


# --------------------------------------------------------------------------
# the judge question
# --------------------------------------------------------------------------

TASK = """\
A memory system delivered a stored fact to an agent during a work session. \
Later in that same session the agent produced concrete observations: command \
output, test results, file diffs, operator messages.

Answer ONE narrow question: does one of these observations CONTRADICT the \
stored fact? A contradiction means the observation and the stored fact cannot \
both be true of the same system at the same time.

Answer contradicted=false unless you can point at a specific observation that \
is incompatible with a specific sentence of the stored fact.

These are NOT contradictions — answer false for every one of them:
* the observations simply do not mention the stored fact (absence of evidence);
* the stored fact is incomplete, vague, or lacks a detail the observation adds;
* the observation restates the stored fact in different words, or agrees with it;
* the stored fact sounds old, cautious or stale but nothing observed disagrees;
* the observation contradicts some OTHER belief, not this stored fact;
* the observation shows a change the session itself just made, while the stored \
fact correctly described the state before that change.

If contradicted=true you MUST also return:
* contradicted_claim — the exact sentence or clause of the stored fact that is \
now false, copied character for character from the stored fact. It must itself \
contain at least one concrete token — a file path, an identifier, a flag, a \
command, a version or a number. If the clause you want does not contain one, \
widen the copied span until it does;
* observation_locator — the locator string of the contradicting observation, \
copied character for character from the observations list;
* observation_quote — the exact contradicting text, copied character for \
character from that observation's excerpt, long enough to stand alone (roughly \
one to three lines) and containing at least one concrete token in common with \
contradicted_claim — the file, identifier, command or value the two disagree \
about. Never reformat, translate or shorten it;
* correction — the corrected fact, written so that a reader who sees ONLY this \
sentence understands what is true now. Name the subject explicitly (file, \
identifier, command, value); do not write "it", "this" or "the above", and do \
not refer to the stored fact, this session or these observations. State what is \
true, not merely that the old text was wrong.

If you cannot do all of that, answer contradicted=false.\
"""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "contradicted": {"type": "boolean"},
        "reason": {"type": "string", "minLength": 1, "maxLength": 600},
        "contradicted_claim": {"type": "string", "maxLength": 800},
        "observation_locator": {"type": "string", "maxLength": 400},
        "observation_quote": {"type": "string", "maxLength": 800},
        "correction": {"type": "string", "maxLength": 900},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["contradicted", "reason"],
    "additionalProperties": False,
}


def build_payload(
    record: SessionRecord,
    candidate: Candidate,
    *,
    config: DetectorConfig = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], dict[str, str]]:
    """The judge payload plus the ``locator -> excerpt`` table used to verify it.

    The excerpt table is the contract the validator enforces: a quote must be a
    contiguous substring of the excerpt the judge was actually shown, at a
    locator that was actually offered.
    """

    node = candidate.node
    excerpts: dict[str, str] = {}
    shown: list[dict[str, Any]] = []
    node_anchors = anchors(node.content)
    for observation, score, shared in candidate.scored:
        excerpt, offset = best_excerpt(observation.text, node_anchors, config.excerpt_chars)
        excerpts[observation.locator] = excerpt
        shown.append(
            {
                "locator": observation.locator,
                "kind": observation.kind,
                "what": observation.label,
                "happened_after_the_recall": True,
                "excerpt": excerpt,
                "excerpt_char_offset": offset,
                "shared_anchors": list(shared),
                "relatedness_score": round(score, 3),
            }
        )
    content = node.content or ""
    payload = {
        "stored_fact": {
            "node_id": node.node_id,
            "scope": node.scope,
            "level": node.level,
            "written_by": node.agent,
            "written_at": node.created_at,
            "content": content[: config.max_node_content_chars],
        },
        "recall": {
            "query": candidate.recall.query,
            "delivered_at_rank": node.rank,
            "session": record.session_key,
        },
        "observations": shown,
    }
    return payload, excerpts


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

#: Every reason the detector can refuse to emit an op. Stable strings: run
#: reports are compared across runs, and a renamed reason would read as a
#: behaviour change that never happened.
REASONS: tuple[str, ...] = (
    "judge_says_no",
    "judge_refused",
    "judge_invalid_output",
    "judge_unavailable",
    "missing_field",
    "unknown_node",
    "unknown_locator",
    "observation_not_after_recall",
    "claim_not_in_node",
    "quote_not_verbatim",
    "quote_touches_redacted_secret",
    "quote_too_thin",
    "quote_asserts_nothing",
    "absence_of_evidence",
    "claim_has_no_anchor",
    "no_shared_anchor",
    "rephrasing",
    "correction_not_self_contained",
    "better_matched_by_another_node",
    "low_confidence",
    "duplicate_node",
)

#: Quotes that assert nothing observed. "No such file or directory" is *not*
#: here on purpose: it is a real contradiction of "the file lives at X".
ABSENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*\(?\s*(?:no|empty)\s*(?:output|result|results|matches?|content)?\s*\)?\s*$", re.IGNORECASE),
    re.compile(r"\bno\s+(?:mention|reference|record|trace|sign|evidence|occurrences?)\s+of\b", re.IGNORECASE),
    re.compile(r"\b(?:never|not)\s+(?:mentioned|referenced|discussed|addressed|touched)\b", re.IGNORECASE),
    re.compile(r"\bthere\s+is\s+no\s+(?:evidence|indication|mention)\b", re.IGNORECASE),
    re.compile(r"\bdid\s+not\s+(?:mention|touch|address|discuss)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:0|no)\s+(?:matches|results|files|occurrences|hits)\b", re.IGNORECASE),
)

# A correction that opens with one of these cannot be read alone: the reader of
# the corrective trace has no antecedent to resolve. "Correction:" and "No ..."
# are deliberately absent -- both are idiomatic openings in this corpus and both
# are followed by the named subject.
_ANAPHORIC_OPENING = re.compile(
    r"^\s*(?:it|this|that|these|those|they|he|she|its|their|them|"
    r"instead|rather|actually|however|but|nope)\b",
    re.IGNORECASE,
)

# Outward references: the reader of the corrective trace alone cannot resolve them.
_OUTWARD_REFERENCE = re.compile(
    r"\b(?:as (?:noted|mentioned|stated|discussed|shown) (?:above|earlier|previously)"
    r"|see (?:above|below|the (?:above|previous))"
    r"|the above|the previous (?:trace|node|fact|memory|statement)"
    r"|the (?:recalled|stored|original) (?:fact|node|trace|memory|claim)"
    r"|this (?:session|transcript|trace|node|excerpt)"
    r"|the (?:session|transcript) (?:above|below)"
    r"|per the previous|contrary to (?:the )?(?:above|memory|the stored))\b",
    re.IGNORECASE,
)

_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|cannot|can't|isn't|aren't|wasn't|weren't|doesn't|don't|didn't|"
    r"without|fails?|failed|missing|absent|unsupported|removed|neither|nor)\b",
    re.IGNORECASE,
)

# ``/home/<user>`` -> ``~`` only. Kept separate from the judge's full redactor so
# the validator can tell a path elision from a secret blanking.
_HOME_ONLY = re.compile(r"/(?:home|Users)/[^/\s\"':,;)\]]+|/root(?=/|\b)")

_WHITESPACE_RE = re.compile(r"\s+")


def _collapse(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def resolve_locator(value: str, offered: Iterable[str]) -> str | None:
    """Map a locator the judge echoed back to the one that was offered.

    A locator embeds the transcript path, and the judge is shown *redacted*
    text: ``claude:/home/sfx/.claude/projects/a.jsonl#42:...`` reaches it as
    ``claude:~/.claude/projects/a.jsonl#42:...``. Echoing what it was shown is
    correct behaviour, so the validator has to undo its own elision instead of
    rejecting every real answer as an invented locator -- which is exactly what
    it did before this function existed: 15 of 15 judge-confirmed
    contradictions on the ground-truth set died as ``unknown_locator``.
    """

    wanted = _collapse(value)
    if not wanted:
        return None
    for locator in offered:
        if wanted == _collapse(locator) or wanted == _collapse(default_redactor(locator)):
            return locator
    return None


def quote_match_mode(haystack: str, needle: str) -> str | None:
    """How ``needle`` occurs in ``haystack``, or ``None`` if it does not.

    ``"exact"`` -- byte-for-byte. ``"home_path_normalized"`` -- byte-for-byte
    once ``/home/<user>`` is rewritten to ``~``, which is the only lossy step
    the judge's redactor applies to ordinary text. ``"redacted_secret"`` --
    only matches after secret blanking, which the caller must reject.

    Whitespace is collapsed as a last resort *within* each mode, because JSON
    round-tripping a quote through a model reliably re-wraps long lines and
    that is not a content difference.
    """

    for mode, rendered in (
        ("exact", haystack),
        ("home_path_normalized", _HOME_ONLY.sub("~", haystack)),
        ("redacted_secret", default_redactor(haystack)),
    ):
        if needle in rendered:
            return mode
        if _collapse(needle) and _collapse(needle) in _collapse(rendered):
            return mode
    return None


@dataclass(frozen=True, slots=True)
class Verdict:
    """The outcome of one judged candidate: an op, or a reason there is none."""

    session_key: str
    node_id: str
    recall_ordinal: int
    accepted: bool
    reason: str
    detail: str = ""
    op: ProposedOp | None = None
    confidence: float = 0.0
    quote_match: str = ""
    observation_kind: str = ""
    observation_locator: str = ""
    judge_said_contradicted: bool = False

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "session_key": self.session_key,
            "node_id": self.node_id,
            "recall_ordinal": self.recall_ordinal,
            "accepted": self.accepted,
            "reason": self.reason,
            "judge_said_contradicted": self.judge_said_contradicted,
        }
        if self.detail:
            data["detail"] = self.detail[:400]
        if self.accepted and self.op is not None:
            data.update(
                {
                    "confidence": round(self.confidence, 3),
                    "quote_match": self.quote_match,
                    "observation_kind": self.observation_kind,
                    "observation_locator": self.observation_locator,
                    "correction": str(self.op.payload.get("correction", "")),
                }
            )
        return data


def _reject(
    candidate: Candidate,
    session_key: str,
    reason: str,
    detail: str = "",
    *,
    judged: bool = True,
) -> Verdict:
    return Verdict(
        session_key=session_key,
        node_id=candidate.node.node_id,
        recall_ordinal=candidate.recall.ordinal,
        accepted=False,
        reason=reason,
        detail=detail,
        judge_said_contradicted=judged,
    )


def validate_verdict(
    record: SessionRecord,
    candidate: Candidate,
    answer: Mapping[str, Any],
    excerpts: Mapping[str, str],
    *,
    config: DetectorConfig = DEFAULT_CONFIG,
    already_proposed: Iterable[str] = (),
) -> Verdict:
    """Turn one judge answer into an op, or veto it with a machine-readable reason.

    Every check here is mechanical and can overrule the model. The order is
    cheapest-and-most-structural first, so a report's rejection histogram reads
    as a funnel rather than as noise.
    """

    session_key = record.session_key
    node = candidate.node

    if not answer.get("contradicted"):
        return _reject(
            candidate, session_key, "judge_says_no", str(answer.get("reason", "")), judged=False
        )

    claim = str(answer.get("contradicted_claim") or "").strip()
    locator = str(answer.get("observation_locator") or "").strip()
    quote = str(answer.get("observation_quote") or "").strip()
    correction = str(answer.get("correction") or "").strip()
    confidence = float(answer.get("confidence") or 0.0)

    missing = [
        name
        for name, value in (
            ("contradicted_claim", claim),
            ("observation_locator", locator),
            ("observation_quote", quote),
            ("correction", correction),
        )
        if not value
    ]
    if missing:
        return _reject(candidate, session_key, "missing_field", ",".join(missing))

    # -- the op must name a node this session was really shown ---------------
    if node.node_id not in record.delivered_node_ids:
        return _reject(candidate, session_key, "unknown_node", node.node_id)
    if node.node_id in set(already_proposed):
        return _reject(candidate, session_key, "duplicate_node", node.node_id)

    # -- the observation must be one that was offered, and be later ----------
    resolved = resolve_locator(locator, excerpts)
    if resolved is None:
        return _reject(candidate, session_key, "unknown_locator", locator)
    locator = resolved
    observation = next(item for item in candidate.observations if item.locator == locator)
    if observation.record_index <= candidate.recall.record_index:
        return _reject(
            candidate,
            session_key,
            "observation_not_after_recall",
            f"{observation.record_index} <= {candidate.recall.record_index}",
        )

    # -- verbatim: the claim is really in the delivered node ----------------
    if quote_match_mode(node.content or "", claim) is None:
        return _reject(candidate, session_key, "claim_not_in_node", claim[:200])

    # -- verbatim: the quote is really in the session record ----------------
    if len(quote) < config.min_quote_chars:
        return _reject(candidate, session_key, "quote_too_thin", quote[:120])
    excerpt_mode = quote_match_mode(excerpts[locator], quote)
    if excerpt_mode is None:
        return _reject(candidate, session_key, "quote_not_verbatim", quote[:200])
    record_mode = quote_match_mode(observation.text, quote)
    if record_mode is None:
        return _reject(candidate, session_key, "quote_not_verbatim", quote[:200])
    if record_mode == "redacted_secret" or excerpt_mode == "redacted_secret":
        return _reject(candidate, session_key, "quote_touches_redacted_secret", quote[:120])

    # -- not an absence of evidence -----------------------------------------
    if any(pattern.search(quote) for pattern in ABSENCE_PATTERNS):
        return _reject(candidate, session_key, "absence_of_evidence", quote[:200])
    quote_anchors = anchors(quote)
    if not quote_anchors:
        return _reject(
            candidate, session_key, "absence_of_evidence", "quote carries no concrete anchor"
        )

    # -- the quote must assert something the claim does not already say -----
    # A bare path lifted out of a file listing is a mention, not a measurement:
    # `rg -l foo` naming a file cannot contradict a claim about what that file
    # does. Likewise a quote already contained in the claim contradicts nothing.
    if len(quote.split()) < 3:
        return _reject(candidate, session_key, "quote_asserts_nothing", quote[:120])
    if _collapse(quote).lower() in _collapse(claim).lower():
        return _reject(candidate, session_key, "quote_asserts_nothing", "quote repeats the claim")

    # -- aboutness: a fact about X is only contradicted by an observation about X
    claim_anchors = anchors(claim)
    if not claim_anchors:
        return _reject(candidate, session_key, "claim_has_no_anchor", claim[:200])
    if not claim_anchors & (quote_anchors | observation.label_anchors):
        return _reject(
            candidate,
            session_key,
            "no_shared_anchor",
            f"claim={sorted(claim_anchors)[:6]} quote={sorted(quote_anchors)[:6]}",
        )

    # -- self-contained ------------------------------------------------------
    # Form before content: "It overwrites the file instead" is unreadable alone
    # whatever it asserts, and saying so is the more useful rejection reason.
    problem = _self_containment_problem(correction, node, config)
    if problem:
        return _reject(candidate, session_key, "correction_not_self_contained", problem)

    # -- not a rephrasing ----------------------------------------------------
    correction_anchors = anchors(correction)
    added = correction_anchors - claim_anchors
    dropped = claim_anchors - correction_anchors
    polarity = len(_NEGATION_RE.findall(correction)) != len(_NEGATION_RE.findall(claim))
    if not added and not dropped and not polarity:
        return _reject(candidate, session_key, "rephrasing", "correction asserts nothing new")
    if _jaccard(_words(correction), _words(claim)) >= 0.9:
        return _reject(candidate, session_key, "rephrasing", "correction is a paraphrase of the claim")
    if not polarity and _collapse(claim).lower() in _collapse(correction).lower():
        return _reject(candidate, session_key, "rephrasing", "correction restates the claim unchanged")

    # -- the correction must belong to *this* node --------------------------
    rival = _better_matching_node(record, node, correction, claim)
    if rival is not None:
        return _reject(candidate, session_key, "better_matched_by_another_node", rival)

    if confidence < config.min_confidence:
        return _reject(candidate, session_key, "low_confidence", f"{confidence:.2f}")

    op = teach_op(
        record,
        candidate,
        observation=observation,
        claim=claim,
        quote=quote,
        correction=correction,
        confidence=confidence,
        quote_match=record_mode,
    )
    return Verdict(
        session_key=session_key,
        node_id=node.node_id,
        recall_ordinal=candidate.recall.ordinal,
        accepted=True,
        reason="accepted",
        op=op,
        confidence=confidence,
        quote_match=record_mode,
        observation_kind=observation.kind,
        observation_locator=locator,
        judge_said_contradicted=True,
    )


def _better_matching_node(
    record: SessionRecord,
    node: DeliveredNode,
    correction: str,
    claim: str,
) -> str | None:
    """Another delivered node this correction fits better, if there is one.

    The "session superseded a DIFFERENT fact" failure mode has a mechanical
    signature when the other fact was also delivered: the corrective text talks
    about *that* node. Superseding the wrong node is worse than superseding
    nothing -- teach drops the target's confidence to ``≤0.25`` -- so a tie goes
    to the node the judge was actually asked about, and only a strictly better
    fit vetoes.
    """

    subject = anchors(correction) | anchors(claim)
    if not subject:
        return None
    mine = len(subject & anchors(node.content))
    for recall in record.recalls:
        for other in recall.delivered:
            if other.node_id == node.node_id or not other.content:
                continue
            theirs = len(subject & anchors(other.content))
            if theirs > mine:
                return f"{other.node_id} matches {theirs} anchors, {node.node_id} matches {mine}"
    return None


def _self_containment_problem(
    correction: str, node: DeliveredNode, config: DetectorConfig
) -> str:
    """Why ``correction`` cannot be read alone -- empty string when it can.

    The convention this enforces is the one the corpus already follows: a
    lesson is one self-contained trace carrying failure, cause and fix, because
    a correction that says "no, use the other flag" is worthless to the next
    agent that recalls it without its neighbours.
    """

    if len(correction) < config.min_correction_chars:
        return f"too short ({len(correction)} < {config.min_correction_chars})"
    if len(correction) > config.max_correction_chars:
        return f"too long ({len(correction)} > {config.max_correction_chars})"
    if _ANAPHORIC_OPENING.match(correction):
        return f"opens with an unresolved reference: {correction[:40]!r}"
    outward = _OUTWARD_REFERENCE.search(correction)
    if outward:
        return f"refers outside itself: {outward.group(0)!r}"
    correction_anchors = anchors(correction)
    if len(correction_anchors) < 2:
        return "names fewer than two concrete things"
    if not correction_anchors & anchors(node.content):
        return "names nothing the corrected fact was about"
    return ""


def teach_op(
    record: SessionRecord,
    candidate: Candidate,
    *,
    observation: Observation,
    claim: str,
    quote: str,
    correction: str,
    confidence: float,
    quote_match: str,
) -> ProposedOp:
    """The ``memory_teach`` call, as a proposal.

    ``payload`` mirrors ``memory_teach(store, trace_id, correction,
    confidence=None, context=None)`` exactly, because the runner passes it
    through. That call -- and only that call -- writes the corrective trace,
    creates the ``correction -> original`` supersedes edge, drops the original's
    confidence to ``≤0.25`` and its usefulness to ``≤-0.25``, and links
    provenance without reinforcing the original.
    """

    node = candidate.node
    recall = candidate.recall
    claim_locator = SourceSpan(
        record.source,
        record.path,
        recall.span.record_index,
        f"recalls[{recall.ordinal}].delivered[{node.rank}].content",
    ).locator()
    context = {
        "agent": EXTRACTOR_AGENT,
        "lesson_kind": LESSON_KIND,
        "extractor": DETECTOR_VERSION,
        "task": f"post-session correction from {record.session_key}",
        "session_id": record.cli_session_id or record.session_key,
        "source_session_key": record.session_key,
        "source": record.source,
        "cli": record.cli,
        "transcript_sha256": record.transcript_sha256 or "",
        "recall_event_id": recall.recall_event_id,
        "recall_query": recall.query[:300],
        "corrected_node_id": node.node_id,
        "contradicted_claim": claim,
        "observation_kind": observation.kind,
        "observation_locator": observation.locator,
        "observation_quote": quote,
        "quote_match": quote_match,
    }
    if node.scope:
        context["scope"] = node.scope
    payload = {
        "trace_id": node.node_id,
        "correction": correction,
        "confidence": round(confidence, 3),
        "context": context,
    }
    return ProposedOp.for_session(
        "teach",
        payload,
        record,
        SourceSpan.parse(observation.locator),
        (
            Evidence(quote=quote, locator=observation.locator),
            Evidence(quote=claim, locator=claim_locator),
        ),
    ).validate()


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


@dataclass(slots=True)
class DetectionResult:
    """What one session yielded, including everything it did not yield."""

    session_key: str
    source: str
    split: str = ""
    ops: list[ProposedOp] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)
    pair_stats: dict[str, int] = field(default_factory=dict)
    judged: int = 0
    delivered_nodes: int = 0
    observations: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_key": self.session_key,
            "source": self.source,
            "split": self.split,
            "delivered_nodes": self.delivered_nodes,
            "observations": self.observations,
            "judged": self.judged,
            "ops": len(self.ops),
            "rejections": dict(sorted(self.rejections.items())),
            "pair_stats": dict(sorted(self.pair_stats.items())),
        }


def detect_corrections(
    record: SessionRecord,
    judge: Judge,
    *,
    config: DetectorConfig = DEFAULT_CONFIG,
    split: str = "",
    only_nodes: Iterable[str] | None = None,
) -> DetectionResult:
    """Run the whole pipeline over one session. Writes nothing anywhere."""

    result = DetectionResult(session_key=record.session_key, source=record.source, split=split)
    result.delivered_nodes = len(record.delivered_node_ids)
    observed = observations(record, config=config)
    result.observations = len(observed)

    pair_stats: dict[str, int] = {}
    pending = candidates(record, config=config, counters=pair_stats, only_nodes=only_nodes)
    result.pair_stats = pair_stats
    if not pending:
        return result

    proposed: list[str] = []
    for candidate in pending:
        payload, excerpts = build_payload(record, candidate, config=config)
        result.judged += 1
        try:
            answer = judge.judge(TASK, SCHEMA, payload, timeout_s=config.judge_timeout_s)
        except JudgeRefused as exc:
            _record(
                result,
                _reject(candidate, record.session_key, "judge_refused", str(exc), judged=False),
            )
            continue
        except JudgeInvalidOutput as exc:
            _record(
                result,
                _reject(
                    candidate, record.session_key, "judge_invalid_output", str(exc), judged=False
                ),
            )
            continue
        except JudgeUnavailable as exc:
            _record(
                result,
                _reject(candidate, record.session_key, "judge_unavailable", str(exc), judged=False),
            )
            continue
        except JudgeError as exc:  # pragma: no cover - future judge failure kinds
            _record(
                result,
                _reject(candidate, record.session_key, "judge_unavailable", str(exc), judged=False),
            )
            continue

        verdict = validate_verdict(
            record, candidate, answer, excerpts, config=config, already_proposed=proposed
        )
        _record(result, verdict)
        if verdict.accepted and verdict.op is not None:
            proposed.append(verdict.node_id)
            result.ops.append(verdict.op)
    return result


def _record(result: DetectionResult, verdict: Verdict) -> None:
    result.verdicts.append(verdict)
    if not verdict.accepted:
        result.rejections[verdict.reason] = result.rejections.get(verdict.reason, 0) + 1


def taught_delivered_nodes(record: SessionRecord) -> tuple[str, ...]:
    """Nodes this session was delivered *and* then corrected through teach.

    These are the corpus's own labelled positives: the working agent recalled
    the node, discovered it was wrong, and called ``memory_teach`` on it. They
    are the only ground truth available for "a recalled fact was contradicted",
    which is why the detector is measured against them.
    """

    delivered = set(record.delivered_node_ids)
    return tuple(
        dict.fromkeys(
            write.supersedes
            for write in record.writes
            if write.kind == "teach" and write.supersedes in delivered
        )
    )


def recover_known_corrections(
    record: SessionRecord,
    judge: Judge,
    *,
    config: DetectorConfig = DEFAULT_CONFIG,
    split: str = "",
) -> DetectionResult:
    """Re-derive the corrections the session made itself. Capability, not output.

    Production skips these nodes -- the correction already landed. Running the
    detector on them anyway answers the question a run of zero ops cannot:
    would it fire if a contradiction were there?
    """

    known = taught_delivered_nodes(record)
    result = DetectionResult(session_key=record.session_key, source=record.source, split=split)
    if not known:
        return result
    probe_config = replace(config, skip_already_taught=False)
    return detect_corrections(
        record, judge, config=probe_config, split=split, only_nodes=known
    )


# --------------------------------------------------------------------------
# negative controls
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NegativeControl:
    """A synthetic session that must yield zero teach ops, and why."""

    name: str
    why: str
    record: SessionRecord


def _control_record(
    name: str,
    *,
    nodes: Sequence[tuple[str, str]],
    query: str,
    turns: Sequence[tuple[str, str]] = (),
    outputs: Sequence[tuple[str, str]] = (),
) -> SessionRecord:
    """Build a minimal but *structurally real* session record.

    Record indices are laid out so every observation follows the recall, which
    is what makes these controls test the judge and the vetoes rather than the
    ordering filter.
    """

    source = "claude"
    path = f"synthetic/{name}.jsonl"
    record = SessionRecord(
        session_key=f"control:{name}",
        source=source,
        cli="claude",
        path=path,
        cli_session_id=f"control-{name}",
        transcript_sha256=f"control-{name}",
        started_at="2026-08-19T00:00:00Z",
        ended_at="2026-08-19T01:00:00Z",
    )
    record.recalls.append(
        RecallInteraction(
            ordinal=0,
            query=query,
            span=SourceSpan(source, path, 1, "mcpMeta.structuredContent"),
            recall_event_id=f"01CONTROL{name.upper().replace('-', '')[:12]:X<12}",
            delivered=[
                DeliveredNode(
                    node_id=node_id,
                    rank=rank,
                    content=content,
                    scope="project:lm",
                    level="trace",
                    agent="claude",
                    created_at="2026-07-01T00:00:00Z",
                    delivery="full",
                )
                for rank, (node_id, content) in enumerate(nodes)
            ],
            record_index=1,
            position=0.02,
        )
    )
    index = 2
    turn_index = 0
    for role, text in turns:
        record.turns.append(
            Turn(turn_index, role, text, "2026-08-19T00:30:00Z", SourceSpan(source, path, index))
        )
        turn_index += 1
        index += 1
    for command, output in outputs:
        record.tool_calls.append(
            ToolCall(
                ordinal=len(record.tool_calls),
                name="Bash",
                server=None,
                arguments={"command": command},
                span=SourceSpan(source, path, index, "message.content.tool_use"),
                call_id=f"call_{index}",
                ok=True,
                result_text=output,
            )
        )
        index += 1
    record.record_count = index
    return record


#: Three sessions that must produce nothing. Each isolates one way a naive
#: detector manufactures corrections.
NEGATIVE_CONTROLS: tuple[NegativeControl, ...] = (
    NegativeControl(
        name="fact-matched-reality",
        why=(
            "The recalled fact is confirmed by the session's own measurements. "
            "A detector that treats 'the session talked about this node' as "
            "evidence of staleness would fire here."
        ),
        record=_control_record(
            "fact-matched-reality",
            query="sqlite snapshot procedure for the living memory database",
            nodes=[
                (
                    "01CTRLMATCH0000000000000001",
                    "Snapshotting living-memory's sqlite database with `cp` alone is "
                    "unsafe in WAL mode: the -wal file holds committed rows that the "
                    "main file does not yet contain. Use `sqlite3 global.sqlite3 "
                    "\".backup snap.db\"`, which checkpoints first and produces a "
                    "consistent copy.",
                )
            ],
            outputs=[
                (
                    "sqlite3 global.sqlite3 '.backup /tmp/snap.db' && sqlite3 /tmp/snap.db 'select count(*) from nodes'",
                    "139696\n",
                ),
                (
                    "cp global.sqlite3 /tmp/plain.db && sqlite3 /tmp/plain.db 'select count(*) from nodes'",
                    "138492\n",
                ),
                (
                    "ls -la global.sqlite3-wal",
                    "-rw-r--r-- 1 sfx sfx 4325376 Aug 19 08:12 global.sqlite3-wal\n",
                ),
            ],
        ),
    ),
    NegativeControl(
        name="fact-untouched",
        why=(
            "The session never touched the recalled fact's subject. Any op here "
            "would be an absence of evidence dressed up as a contradiction."
        ),
        record=_control_record(
            "fact-untouched",
            query="cuda kernel dispatch for vector_pad_truncate",
            nodes=[
                (
                    "01CTRLUNTOUCHED0000000001",
                    "The only model-layer runtime call of be->vector_pad_truncate is "
                    "src/model/stems/stems.cpp:213 in VectorStem::forward; "
                    "src/model/f_theta.cpp:475 is a precision-policy validation site, "
                    "not an execution call site.",
                )
            ],
            turns=[("user", "Fix the typo in the README install section, nothing else.")],
            outputs=[
                (
                    "grep -n 'pip install' README.md",
                    "41:    pip install -e .\n42:    pip instal -e '.[dev]'\n",
                ),
                (
                    "python -m pytest tests/test_readme_examples.py -q",
                    "2 passed in 0.41s\n",
                ),
                # Incidental and on-topic enough to be judged rather than
                # filtered: without it this control would only exercise the
                # pairing filter and never the judge.
                (
                    "rg -l vector_pad_truncate src/model",
                    "src/model/stems/stems.cpp\nsrc/model/f_theta.cpp\nsrc/model/backends/cpu_backend.cpp\n",
                ),
            ],
        ),
    ),
    NegativeControl(
        name="different-fact-superseded",
        why=(
            "The session really does correct a belief -- but a belief about "
            "max_results that was never recalled. The delivered node is about "
            "the depth parameter and is untouched. A detector that attaches any "
            "in-session correction to whatever node was recalled fires here."
        ),
        record=_control_record(
            "different-fact-superseded",
            query="memory_recall parameters depth graph traversal",
            nodes=[
                (
                    "01CTRLDIFFERENT000000001",
                    "memory_recall in src/living_memory/server.py accepts "
                    "depth='causal' to widen graph traversal beyond the numeric hop "
                    "count: the causal setting follows supersedes and caused edges, "
                    "which is the right depth when debugging why a belief changed.",
                )
            ],
            turns=[
                (
                    "user",
                    "Correction before you continue: the max_results default in "
                    "src/living_memory/server.py is 5, not the 10 the docs claim. Fix "
                    "the docs, do not touch depth.",
                )
            ],
            outputs=[
                (
                    "grep -n 'max_results' src/living_memory/server.py",
                    'src/living_memory/server.py:212:    max_results: int = 5,\n',
                ),
                (
                    "python -c \"import inspect, living_memory.server as s; print(inspect.signature(s.memory_recall))\"",
                    "(query: str, scope: str | None = None, depth: int | str | None = 1, max_results: int = 5)\n",
                ),
            ],
        ),
    ),
)


def run_negative_controls(
    judge: Judge, *, config: DetectorConfig = DEFAULT_CONFIG
) -> list[dict[str, Any]]:
    """Replay every control through ``judge``. Each must yield zero ops."""

    report: list[dict[str, Any]] = []
    for control in NEGATIVE_CONTROLS:
        result = detect_corrections(control.record, judge, config=config, split="control")
        report.append(
            {
                "name": control.name,
                "why": control.why,
                "ops": len(result.ops),
                "judged": result.judged,
                "rejections": dict(sorted(result.rejections.items())),
                "pair_stats": dict(sorted(result.pair_stats.items())),
                "passed": not result.ops,
            }
        )
    return report


# --------------------------------------------------------------------------
# run report
# --------------------------------------------------------------------------


#: Worked examples kept per rejection reason in a run report.
SAMPLES_PER_REASON = 3


def _recovery_section(results: Sequence[DetectionResult]) -> dict[str, Any]:
    """Capability against the corpus's own labelled positives.

    A production run that emits nothing is ambiguous: either the split held no
    contradictions, or the detector is deaf. This section resolves that. It
    re-judges the nodes the sessions themselves corrected through
    ``memory_teach`` -- known positives -- with the production skip disabled.
    Its ops are evidence, never proposals: the correction already landed, so
    ``executable`` is false and the runner must ignore them.
    """

    rejections: dict[str, int] = {}
    found: list[dict[str, Any]] = []
    judged = 0
    for result in results:
        judged += result.judged
        for key, count in result.rejections.items():
            rejections[key] = rejections.get(key, 0) + count
        for verdict in result.verdicts:
            if verdict.accepted and verdict.op is not None:
                found.append(
                    {
                        "verdict": verdict.as_dict(),
                        # The whole op, so this section is auditable on the same
                        # terms as a production one: a reader can check the
                        # trace_id, the payload shape and the evidence spans
                        # without re-running anything.
                        "op": verdict.op.to_dict(),
                    }
                )
    return {
        "what": (
            "nodes the session itself corrected through memory_teach, re-judged "
            "with the production already-taught skip disabled"
        ),
        "executable": False,
        "sessions": len(results),
        "judged": judged,
        "recovered": len(found),
        "rejections": dict(sorted(rejections.items())),
        "corrections": found,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_report(
    results: Sequence[DetectionResult],
    *,
    splits_read: Sequence[str],
    config: DetectorConfig,
    judge_backend: str,
    judge_model: str | None,
    judge_stats: Sequence[Any] = (),
    controls: Sequence[Mapping[str, Any]] = (),
    recovery: Sequence[DetectionResult] = (),
    notes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The tracked run report. Records which splits were read, always."""

    rejections: dict[str, int] = {}
    pair_stats: dict[str, int] = {}
    ops: list[dict[str, Any]] = []
    accepted_verdicts: list[dict[str, Any]] = []
    samples: dict[str, list[dict[str, Any]]] = {}
    judged = 0
    sessions_with_ops = 0
    for result in results:
        judged += result.judged
        for key, count in result.rejections.items():
            rejections[key] = rejections.get(key, 0) + count
        for key, count in result.pair_stats.items():
            pair_stats[key] = pair_stats.get(key, 0) + count
        if result.ops:
            sessions_with_ops += 1
        for verdict in result.verdicts:
            if verdict.accepted:
                accepted_verdicts.append(verdict.as_dict())
                continue
            # A silent detector is indistinguishable from a broken one. Keeping
            # a few worked examples per reason is what turned "185 judge calls,
            # zero ops" into a found bug rather than a conclusion.
            bucket = samples.setdefault(verdict.reason, [])
            if len(bucket) < SAMPLES_PER_REASON:
                bucket.append(verdict.as_dict())
        ops.extend(op.to_dict() for op in result.ops)

    cost = sum(float(getattr(item, "total_cost_usd", 0.0)) for item in judge_stats)
    outcomes: dict[str, int] = {}
    for item in judge_stats:
        outcome = str(getattr(item, "outcome", "unknown"))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    report = {
        "generated_at": _utc_now(),
        "generator": "living_memory.postsession.corrections",
        "detector_version": DETECTOR_VERSION,
        "splits_read": list(splits_read),
        "ops_are_proposals": True,
        "living_memory_writes": 0,
        "redaction": (
            "home directories are elided to ~, secret-shaped substrings are "
            "blanked, and the reserved split's name is masked to "
            "<reserved-split:...> before this file is written; quote_match "
            "records how each evidence quote was verified against the "
            "unredacted transcript, which is what the detector actually checked"
        ),
        "config": config.as_dict(),
        "judge": {
            "backend": judge_backend,
            "model": judge_model,
            "calls": len(judge_stats),
            "outcomes": dict(sorted(outcomes.items())),
            "cost_usd": round(cost, 4),
        },
        "totals": {
            "sessions": len(results),
            "sessions_with_ops": sessions_with_ops,
            "delivered_nodes": sum(item.delivered_nodes for item in results),
            "observations": sum(item.observations for item in results),
            "candidates_judged": judged,
            "ops": len(ops),
        },
        "pair_filter": dict(sorted(pair_stats.items())),
        "rejections": dict(sorted(rejections.items())),
        "rejection_samples": {key: samples[key] for key in sorted(samples)},
        "negative_controls": list(controls),
        "ground_truth_recovery": _recovery_section(recovery),
        "accepted": accepted_verdicts,
        "ops": ops,
        "sessions": [item.as_dict() for item in results],
    }
    if notes:
        report["notes"] = dict(notes)
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

#: Splits this module will read. The reserved split is not one of them, and the
#: CLI refuses to name it: the field-measurement child owns that seal.
READABLE_SPLITS: tuple[str, ...] = ("train", "eval")


def reserved_splits() -> tuple[str, ...]:
    """Split labels this module refuses to read, derived from the corpus."""

    from .corpus import SPLITS

    return tuple(sorted(set(SPLITS) - set(READABLE_SPLITS)))


def mask_reserved_splits(text: str) -> str:
    """Blank the reserved split's name out of a report before it is tracked.

    The stage's contract is that a correction run's report names the splits it
    read and mentions the reserved one nowhere -- a grep-able guarantee that
    the sealed split stayed sealed. But real corrections quote sessions that
    *discuss* that split by name (this repository's own goal tree does), and
    dropping those ops would silence true findings to satisfy a string check.
    Masking the token in the rendered artifact keeps both: the ops survive, and
    the file still cannot be mistaken for one that read the sealed split.
    """

    for label in reserved_splits():
        text = re.sub(re.escape(label), f"<reserved-split:{label[:1]}>", text, flags=re.IGNORECASE)
    return text


def _load_entries(index_path: Path, split: str) -> list[Any]:
    """Index rows for ``split``, minus anything whose group might be reserved.

    The sealed manifest and a freshly rebuilt index can disagree by a handful of
    sessions recorded after the seal. Union-find groups only ever *merge*, so a
    session's sealed split key is some token of its current group: excluding
    every group that contains a reserved-bucket token is conservative in the one
    direction that matters.
    """

    from .corpus import bucket_for, iter_index, split_for_bucket

    entries = list(iter_index(index_path))
    groups: dict[str, list[Any]] = {}
    for entry in entries:
        groups.setdefault(entry.split_key, []).append(entry)
    blocked: set[str] = set()
    for key, members in groups.items():
        tokens = {key}
        for member in members:
            tokens.add(member.session_key)
            tokens.update(f"id:{item}" for item in member.linked_session_ids)
        if any(split_for_bucket(bucket_for(token)) not in READABLE_SPLITS for token in tokens):
            blocked.add(key)
    return [
        entry
        for entry in entries
        if entry.split == split and entry.split_key not in blocked
    ]


def mentions_tool(path: Path, tool: str, *, chunk_bytes: int = 1 << 20) -> bool:
    """Whether the transcript mentions ``tool`` at all.

    A session that never recalled cannot have been contradicted, and parsing a
    300 MB transcript to discover that costs far more than a streaming byte
    scan. Chunks overlap by the marker length so a hit spanning a boundary is
    not missed.
    """

    marker = tool.encode("utf-8")
    overlap = len(marker) - 1
    try:
        with path.open("rb") as handle:
            tail = b""
            while True:
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    return False
                if marker in tail + chunk:
                    return True
                tail = chunk[-overlap:] if overlap else b""
    except OSError:
        return False


def _make_judge(name: str, model: str | None, config: DetectorConfig) -> Judge:
    prompt_config = PromptConfig(max_payload_bytes=config.max_payload_bytes)
    if name == "claude":
        from .judge import ClaudeCliJudge

        return ClaudeCliJudge(model=model, prompt_config=prompt_config)
    if name == "none":
        from .judge import FakeJudge

        return FakeJudge((), prompt_config=prompt_config)
    raise SystemExit(f"unknown judge backend: {name!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the detector over one split and write the run report."""

    from .corpus import load_session

    parser = argparse.ArgumentParser(prog="living_memory.postsession.corrections")
    parser.add_argument("--split", choices=READABLE_SPLITS, default="eval")
    parser.add_argument("--index", required=True, help="path to the corpus index jsonl")
    parser.add_argument("--out", default=None, help="write the run report here")
    parser.add_argument("--limit", type=int, default=40, help="sessions to read")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--judge", default="claude", choices=("claude", "none"))
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--max-judge-calls", type=int, default=200)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument(
        "--recovery-limit",
        type=int,
        default=0,
        help="also re-judge up to N sessions whose own memory_teach labels a "
        "delivered node as contradicted (capability check, not output)",
    )
    parser.add_argument(
        "--recovery-key",
        action="append",
        default=None,
        help="restrict the recovery cohort to these session keys. Without it "
        "the pass walks the split in order and fully parses every session that "
        "merely *mentions* memory_teach in prose -- 324 of 400 on eval, because "
        "this repository's own goal tree discusses the tool by name -- to find "
        "the few that actually taught a delivered node. Preselecting the cohort "
        "changes which sessions are read, never how they are judged.",
    )
    parser.add_argument(
        "--require-recall-marker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip transcripts that never mention memory_recall",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=40_000_000,
        help="skip transcripts larger than this; they are logged, never hidden",
    )
    parser.add_argument("--source", action="append", default=None)
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    config = DEFAULT_CONFIG
    judge = _make_judge(args.judge, args.model, config)

    entries = _load_entries(Path(args.index), args.split)
    if args.source:
        entries = [entry for entry in entries if entry.source in set(args.source)]
    oversized = [entry for entry in entries if entry.bytes > args.max_bytes]
    entries = [entry for entry in entries if entry.bytes <= args.max_bytes]
    without_recall = 0
    if args.require_recall_marker:
        kept = [entry for entry in entries if mentions_tool(Path(entry.path), RECALL_TOOL)]
        without_recall = len(entries) - len(kept)
        entries = kept
    # A deterministic pseudo-random sample of the split. Sorting by the key
    # itself would order by source prefix (every `ae_node:` session first);
    # sorting by size would measure the handful of enormous sessions and call
    # the result the split's. Hashing does neither and reproduces exactly.
    entries.sort(key=lambda entry: sha256_text(entry.session_key))
    window = entries[args.offset : args.offset + args.limit]

    results: list[DetectionResult] = []
    calls = 0
    skipped_for_budget = 0
    for entry in window:
        if calls >= args.max_judge_calls:
            skipped_for_budget += 1
            continue
        try:
            record = load_session(entry)
        except Exception as exc:  # a corrupt transcript must not kill the run
            print(f"! load failed {entry.session_key}: {exc}", file=sys.stderr)
            continue
        result = detect_corrections(record, judge, config=config, split=entry.split)
        calls += result.judged
        results.append(result)
        print(
            f"{entry.split} {entry.session_key} judged={result.judged} "
            f"ops={len(result.ops)} rejections={result.rejections}",
            file=sys.stderr,
        )

    recovery: list[DetectionResult] = []
    cohort = set(args.recovery_key or ())
    pool = [entry for entry in entries if entry.session_key in cohort] if cohort else entries
    for entry in pool if args.recovery_limit else ():
        if len(recovery) >= args.recovery_limit:
            break
        if not mentions_tool(Path(entry.path), TEACH_TOOL):
            continue
        try:
            record = load_session(entry)
        except Exception:  # noqa: BLE001 - a corrupt transcript is not fatal
            continue
        if not taught_delivered_nodes(record):
            continue
        result = recover_known_corrections(record, judge, config=config, split=entry.split)
        recovery.append(result)
        print(
            f"recovery {entry.session_key} judged={result.judged} "
            f"recovered={len(result.ops)}",
            file=sys.stderr,
        )

    controls = [] if args.skip_controls else run_negative_controls(judge, config=config)
    report = build_report(
        results,
        splits_read=[args.split],
        config=config,
        judge_backend=getattr(judge, "backend", args.judge),
        judge_model=args.model if args.judge == "claude" else None,
        judge_stats=judge.stats,
        controls=controls,
        recovery=recovery,
        notes={
            "sessions_available": len(entries),
            "sessions_without_a_recall": without_recall,
            "sessions_too_large_to_read": len(oversized),
            "max_transcript_bytes": args.max_bytes,
            "sessions_read": len(results),
            "sessions_skipped_for_judge_budget": skipped_for_budget,
            "judge_call_budget": args.max_judge_calls,
            "recovery_cohort_preselected": len(cohort),
            "recovery_sessions_read": len(recovery),
        },
    )
    text = mask_reserved_splits(default_redactor(canonical_json(report))) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    else:
        print(text)
    failed = [item for item in controls if not item["passed"]]
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
