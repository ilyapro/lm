"""Offline evidence assembly: what a finished session can *prove* it used.

``memory_attest`` (see :mod:`living_memory.attestation`) grades one recall event
by recomputing containment between the event's own delivered nodes and the
evidence a client submits. This module builds that evidence from one
:class:`~living_memory.postsession.session.SessionRecord`: a bounded, redacted,
deterministically ordered list of item strings, each carrying the
:class:`~living_memory.postsession.session.SourceSpan` it was lifted from, so a
runner can emit a ``ProposedOp(kind="attest")`` whose payload is
``{"recall_event_id": ..., "evidence": [...]}`` with real
:class:`~living_memory.postsession.session.Evidence` provenance.

Nothing here writes to Living Memory, opens a database, or makes a network
call. It reads a ``SessionRecord`` that is already in memory and returns data.

Artifacts, not prose
--------------------
Evidence is drawn from exactly two places:

* ``SessionRecord.file_mutations`` -- the hunks of ``unified_diff``, one item
  per hunk rather than one per file; plus the changed region of ``post_image``
  for whole-file writes that carry no diff;
* ``SessionRecord.tool_calls`` -- the command actually run paired with its
  ``result_text``, for non-memory tools only.

Two things are excluded on purpose, and the exclusions are the point of the
whole path:

``Turn.text`` (assistant and user prose)
    An agent can restate a recalled memory verbatim in its own sentences. If
    prose counted, a recall would ground on the agent's paraphrase of itself,
    which rebuilds exactly the self-marking circularity this path exists to
    escape -- the goal is "a usage signal that does not depend on the reading
    agent marking its own recall". A recalled fact that reaches a diff hunk or
    a command output has demonstrably influenced an artifact; one that only
    reaches the agent's own sentences has not.

``memory_*`` tool payloads (``ToolCall.is_memory_tool``)
    A ``memory_recall`` result *contains the recalled node text*. Submitting it
    would ground every node on itself: containment 1.0 for free, for every
    result the event delivered. The call arguments are excluded with the
    results, because a query is the agent's prose too.

That leaves items whose content the session had to *produce* -- a patch it
applied, a command it ran and the output that command printed.

Determinism
-----------
The same ``SessionRecord`` must always yield byte-identical items, because the
server's idempotency key is ``(recall_event_id, evidence_sha256)``: a
non-deterministic assembler would re-credit the same session on every rerun of
the extraction stage under a fresh digest. So:

* the ordering is fixed (see :func:`assemble_evidence`) and depends only on
  ordinals already recorded in the ``SessionRecord``;
* the truncation rule is fixed: an over-long text is *split* on line boundaries
  into cap-sized parts rather than cut, so the surplus competes for the budget
  as further items instead of being discarded (the server grounds each item
  independently, so many hunk-sized items give a more conservative and more
  informative verdict than a few giant ones);
* every item is already canonical under
  :func:`living_memory.attestation.canonical_evidence`, so the digest computed
  here is the digest the server computes. :func:`assemble_evidence` asserts
  that: a drift in the server's canonicalisation raises rather than silently
  changing every ledger key.

Bounds
------
``EVIDENCE_MAX_ITEMS``, ``EVIDENCE_MAX_ITEM_CHARS`` and
``EVIDENCE_MAX_TOTAL_CHARS`` are imported from
:mod:`living_memory.attestation` rather than restated, so the client's caps
cannot drift from the server's. Caller overrides are clamped *down* to them: a
client may submit less than the server accepts, never more.

Those caps are not a matter of taste: they set how much text one grading may
draw on, and the per-item maximum turns every extra item into another
independent chance of clearing 0.25 by coincidence. ``scripts/attest_eval.py``
measured that directly on real transcripts and the caps were re-fitted on the
corpus's train split as a result -- see ``docs/post-session-attestation.md``.
Because the splitter below turns an over-long text into cap-sized *parts*
rather than discarding the surplus, a tighter item cap does not lose content
outright; it makes each graded unit trace-sized, which is the regime the
threshold was calibrated in.

When the caps bind, the assembler reports what it left behind -- counts and
characters by category in :meth:`EvidenceBundle.summary` -- because a silent
truncation reads as "we submitted everything" when it did not.

Redaction
---------
Every item passes through :func:`living_memory.postsession.judge.default_redactor`
*before* bounding, so the caps describe what is actually sent, and so a secret
can never be split across two items and survive. No second set of secret
patterns is defined here; the redactor is the one place they live.

Deliberate non-goal: no temporal filter
---------------------------------------
Every recall event of a session is offered the *same* bundle, including
artifacts produced before that recall happened. Filtering evidence to what
followed a given recall would be a real tightening, but it is not what makes
this honest -- the server's containment gate is -- and it would give each event
a different digest for the same session. It is left to the field-check child,
which can measure whether it changes the cross-session discrimination at all.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any, TypeVar
import difflib
import re

from living_memory.attestation import (
    EVIDENCE_MAX_ITEM_CHARS,
    EVIDENCE_MAX_ITEMS,
    EVIDENCE_MAX_TOTAL_CHARS,
    canonical_evidence,
)

from .judge import Redactor, default_redactor
from .session import (
    Evidence,
    FileMutation,
    ProposedOp,
    SessionRecord,
    SourceSpan,
    ToolCall,
)

__all__ = [
    "CATEGORIES",
    "COMMAND",
    "COMMAND_ARG_KEYS",
    "DIFF_HUNK",
    "EVIDENCE_MAX_ITEMS",
    "EVIDENCE_MAX_ITEM_CHARS",
    "EVIDENCE_MAX_TOTAL_CHARS",
    "EvidenceBundle",
    "EvidenceError",
    "EvidenceItem",
    "FILE_WRITE",
    "assemble_evidence",
    "command_text",
    "proposed_attestations",
    "split_diff_hunks",
    "synthesize_diff",
]

#: A hunk of a recorded ``unified_diff``.
DIFF_HUNK = "diff_hunk"
#: The changed region of a whole-file write that carried no diff.
FILE_WRITE = "file_write"
#: A command the session ran, plus what it printed.
COMMAND = "command"

#: Every category an item can carry, in the order the summary reports them.
CATEGORIES: tuple[str, ...] = (DIFF_HUNK, FILE_WRITE, COMMAND)

#: Argument names that carry *the command actually run*, in lookup order.
#: ``command`` is claude's ``Bash``/``bash``; ``cmd`` is codex's
#: ``exec_command``. A call with none of them contributes nothing: rendering
#: arbitrary tool arguments would smuggle prose back in through the side door
#: (a sub-agent ``Task`` prompt, a ``TodoWrite`` list, a ``WebFetch`` question),
#: and the whole exclusion above would be for nothing. The AE chat recorder's
#: ``_input_summary`` is deliberately not on this list: it is the host's
#: *truncated* JSON echo of the call, and recovering a command from a cut-off
#: JSON string is guesswork -- those calls are reported as
#: ``tool_calls_without_command`` instead.
COMMAND_ARG_KEYS: tuple[str, ...] = ("command", "cmd", "shell_command", "script")

#: Context lines for a diff synthesised from ``pre_image``/``post_image``.
SYNTHETIC_DIFF_CONTEXT = 3

#: Marks the command line inside a command item, so the grader can tell the
#: invocation from its output. One ASCII prompt, not a sentence.
COMMAND_PREFIX = "$ "

#: A hunk header: ``@@ -a,b +c,d @@``, and the ``@@@`` form ``git diff -c``
#: emits for a merge. Deliberately narrower than "starts with @@", so a
#: clipped diff cannot turn a body line into a splitter.
_HUNK_HEADER_RE = re.compile(r"^@@+ [-+]\d")

_T = TypeVar("_T")


class EvidenceError(ValueError):
    """The assembler cannot produce evidence the server would accept as sent."""


# --------------------------------------------------------------------------
# canonical text
# --------------------------------------------------------------------------


def _canonical(text: str) -> str:
    """Normalise exactly as :func:`attestation.canonical_evidence` does.

    Kept as a private mirror rather than imported, because the server's helper
    is private; :func:`assemble_evidence` re-checks every finished item against
    the public :func:`canonical_evidence`, so a divergence fails loudly instead
    of silently changing ``evidence_sha256``.

    Every step only removes characters, so ``len(_canonical(x)) <= len(x)``:
    an item that fits a cap here still fits it after the server canonicalises.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _split_to_limit(text: str, limit: int) -> list[str]:
    """Split canonical ``text`` into canonical parts of at most ``limit`` chars.

    Splits on line boundaries; a single line longer than the cap (a minified
    bundle, a base64 blob) is cut at the cap, which is the only place a line is
    ever broken mid-way. Parts that canonicalise to nothing are dropped.
    """

    if limit <= 0:
        return []
    parts: list[str] = []
    current: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal current, size
        if current:
            parts.append("\n".join(current))
            current = []
            size = 0

    for line in text.split("\n"):
        while len(line) > limit:
            flush()
            parts.append(line[:limit])
            line = line[limit:]
        cost = len(line) + (1 if current else 0)
        if size + cost > limit:
            flush()
            cost = len(line)
        current.append(line)
        size += cost
    flush()
    return [part for part in (_canonical(part) for part in parts) if part]


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def split_diff_hunks(unified_diff: str) -> list[str]:
    """One string per ``@@`` hunk, each carrying its own file header.

    The ``--- a/x`` / ``+++ b/x`` pair a diff declares is repeated on every
    hunk taken from it, so each item is self-describing: the server grades
    items independently and has no way to look at its neighbours. The header
    lines are the artifact's own text, not something synthesised here.

    A hunk ends at the next structural marker -- another ``@@`` header or a
    ``diff --git`` line -- and only *then* at a ``--- ``/``+++ `` file header.
    The order matters both ways round: a *deleted* line whose content begins
    with ``--`` renders as ``---`` and would end the hunk early, so the
    ``@@ -a,b +c,d @@`` line counts are used to tell that case from a real file
    header; and a transcript clips oversized diffs, so a header whose counts no
    longer match its body must not silently swallow the next hunk or drop the
    rest of its own. Counts disambiguate, markers split.
    """

    lines = unified_diff.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    hunks: list[str] = []
    header: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _HUNK_HEADER_RE.match(line):
            body, index = _read_hunk(lines, index)
            hunks.append("\n".join(header + body))
            continue
        if line.startswith("diff --git "):
            header = [line]
        elif line.startswith("--- "):
            header = [line]
        elif line.startswith("+++ "):
            header = header + [line] if header else [line]
        index += 1
    return [text for text in (_canonical(hunk) for hunk in hunks) if text]


def _read_hunk(lines: Sequence[str], start: int) -> tuple[list[str], int]:
    """Return the hunk starting at ``lines[start]`` and the index after it."""

    head = lines[start]
    old_left, new_left = _hunk_counts(head)
    counted = old_left is not None and new_left is not None
    body = [head]
    index = start + 1
    while index < len(lines):
        line = lines[index]
        if _HUNK_HEADER_RE.match(line) or line.startswith("diff --git "):
            break
        spent = counted and old_left <= 0 and new_left <= 0  # type: ignore[operator]
        if line.startswith(("--- ", "+++ ")) and (not counted or spent):
            break
        if counted:
            if line.startswith("\\"):  # "\ No newline at end of file"
                pass
            elif line.startswith("-"):
                old_left = max(0, old_left - 1)  # type: ignore[operator]
            elif line.startswith("+"):
                new_left = max(0, new_left - 1)  # type: ignore[operator]
            else:  # context, including the empty string for a blank line
                old_left = max(0, old_left - 1)  # type: ignore[operator]
                new_left = max(0, new_left - 1)  # type: ignore[operator]
        body.append(line)
        index += 1
    return body, index


def _hunk_counts(head: str) -> tuple[int | None, int | None]:
    """``(old_lines, new_lines)`` from an ``@@ -a,b +c,d @@`` header."""

    try:
        ranges = head.split("@@")[1].split()
        old, new = ranges[0], ranges[1]
        old_count = int(old.split(",")[1]) if "," in old else 1
        new_count = int(new.split(",")[1]) if "," in new else 1
    except (IndexError, ValueError):
        return None, None
    return old_count, new_count


def synthesize_diff(pre_image: str, post_image: str, path: str) -> str | None:
    """A unified diff of a whole-file write the host recorded without one.

    This is the "changed region of ``post_image``" the evidence policy asks
    for: submitting the whole post-image would spend the entire budget on lines
    the session never touched.
    """

    diff = difflib.unified_diff(
        pre_image.splitlines(),
        post_image.splitlines(),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
        n=SYNTHETIC_DIFF_CONTEXT,
    )
    text = "\n".join(diff)
    return text or None


def command_text(call: ToolCall) -> str | None:
    """The command a tool call ran, or ``None`` when it ran no command.

    A list value is joined with spaces -- codex records ``["bash", "-lc", ...]``
    -- so the item reads as the line that was executed.
    """

    for key in COMMAND_ARG_KEYS:
        if key not in call.arguments:
            continue
        value = call.arguments[key]
        if isinstance(value, str):
            text = value.strip()
        elif isinstance(value, (list, tuple)):
            text = " ".join(str(part) for part in value).strip()
        else:
            continue
        if text:
            return text
    return None


def _mutation_text(mutation: FileMutation) -> tuple[str, list[str]] | None:
    """``(category, texts)`` for one file mutation, or ``None`` when it carries
    no content to quote.

    Order of preference: the recorded diff, then a diff synthesised against the
    pre-image, then the post-image itself (a create has no unchanged part, so
    all of it is the changed region).
    """

    if mutation.unified_diff:
        hunks = split_diff_hunks(mutation.unified_diff)
        if hunks:
            return DIFF_HUNK, hunks
    if mutation.post_image:
        if mutation.pre_image:
            synthetic = synthesize_diff(
                mutation.pre_image, mutation.post_image, mutation.path
            )
            if synthetic:
                hunks = split_diff_hunks(synthetic)
                if hunks:
                    return FILE_WRITE, hunks
        body = _canonical(mutation.post_image)
        if body:
            return FILE_WRITE, [body]
    return None


# --------------------------------------------------------------------------
# the bundle
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One evidence string plus where in the transcript it came from."""

    text: str
    span: SourceSpan
    category: str
    #: Human-readable origin -- the mutated path, or ``tool#ordinal``.
    origin: str

    def locator(self) -> str:
        return self.span.locator()

    def as_evidence(self) -> Evidence:
        """The provenance record a :class:`ProposedOp` carries."""

        return Evidence(quote=self.text, locator=self.span.locator())


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """The ordered, bounded, redacted evidence one session offers.

    ``evidence_sha256`` is what the server will compute over :attr:`texts`, so
    a caller can check the ledger key before submitting anything.
    """

    session_key: str
    items: tuple[EvidenceItem, ...]
    evidence_sha256: str
    #: Items each category would have contributed with unlimited caps.
    considered: Mapping[str, int]
    #: Items each category actually contributed.
    kept: Mapping[str, int]
    #: ``considered - kept`` per category: what the caps left behind.
    dropped: Mapping[str, int]
    #: Characters in those dropped items.
    dropped_chars: int
    #: Content excluded by *policy*, never by the caps (prose, memory tools).
    excluded: Mapping[str, int]
    #: The caps in force, after clamping.
    caps: Mapping[str, int]
    #: Which caps actually bound: ``max_items`` and/or ``max_total_chars``.
    caps_hit: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)

    @property
    def texts(self) -> tuple[str, ...]:
        """The evidence list to submit, in submission order."""

        return tuple(item.text for item in self.items)

    @property
    def spans(self) -> tuple[SourceSpan, ...]:
        return tuple(item.span for item in self.items)

    @property
    def chars(self) -> int:
        return sum(len(item.text) for item in self.items)

    def evidence(self) -> tuple[Evidence, ...]:
        return tuple(item.as_evidence() for item in self.items)

    def summary(self) -> dict[str, Any]:
        """Counts only -- safe to log or persist beside a tracked artifact.

        Carries the drop report: what the caps left behind, by category, and
        what policy excluded before the caps ever applied.
        """

        return {
            "session_key": self.session_key,
            "items": len(self.items),
            "chars": self.chars,
            "evidence_sha256": self.evidence_sha256,
            "considered": dict(self.considered),
            "kept": dict(self.kept),
            "dropped": dict(self.dropped),
            "dropped_chars": self.dropped_chars,
            "excluded": dict(self.excluded),
            "caps": dict(self.caps),
            "caps_hit": list(self.caps_hit),
        }


@dataclass(frozen=True, slots=True)
class _Candidate:
    text: str
    span: SourceSpan
    category: str
    origin: str


def _round_robin(groups: Sequence[Sequence[_T]]) -> list[_T]:
    """Breadth first: every group's first element, then every group's second."""

    out: list[_T] = []
    for depth in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if depth < len(group):
                out.append(group[depth])
    return out


def _sub_span(span: SourceSpan, suffix: str) -> SourceSpan:
    field = f"{span.field}.{suffix}" if span.field else suffix
    return SourceSpan(span.source, span.path, span.record_index, field)


def _parts(
    text: str,
    span: SourceSpan,
    category: str,
    origin: str,
    suffix: str,
    limit: int,
) -> list[_Candidate]:
    """Cap-sized candidates for one extracted text, with per-part locators."""

    pieces = _split_to_limit(text, limit)
    if len(pieces) == 1:
        return [_Candidate(pieces[0], _sub_span(span, suffix), category, origin)]
    return [
        _Candidate(piece, _sub_span(span, f"{suffix}.part[{i}]"), category, origin)
        for i, piece in enumerate(pieces)
    ]


def assemble_evidence(
    record: SessionRecord,
    *,
    redactor: Redactor = default_redactor,
    max_items: int = EVIDENCE_MAX_ITEMS,
    max_item_chars: int = EVIDENCE_MAX_ITEM_CHARS,
    max_total_chars: int = EVIDENCE_MAX_TOTAL_CHARS,
) -> EvidenceBundle:
    """Build the evidence bundle for ``record``.

    The pipeline, in this order, is the contract:

    1. **extract** -- diff hunks per file mutation, in mutation ordinal order;
       command plus result per non-memory tool call, in tool ordinal order.
       Prose and memory-tool payloads never enter (see the module docstring);
    2. **redact** -- ``redactor`` runs over each extracted text *whole*, before
       it is split, so a secret cannot survive by straddling a boundary;
    3. **split** -- each text becomes one or more canonical parts of at most
       ``max_item_chars``;
    4. **order** -- mutations round-robin (every mutation's first part, then
       every mutation's second) interleaved one-for-one with commands
       round-robin, mutations first. Breadth before depth, so one 400-hunk
       refactor cannot spend the whole budget before the first command is seen,
       and one chatty ``pytest`` run cannot bury the diffs;
    5. **dedupe** -- byte-identical items keep their first occurrence only;
    6. **bound** -- accept in order while the item count and the total fit.
       An item too large for the *remaining* total is skipped and the scan
       continues, so a small item behind a large one still gets in.

    Caller caps are clamped down to the server's; passing a larger one is not
    an error, it simply has no effect.
    """

    max_items = max(0, min(int(max_items), EVIDENCE_MAX_ITEMS))
    max_item_chars = max(0, min(int(max_item_chars), EVIDENCE_MAX_ITEM_CHARS))
    max_total_chars = max(0, min(int(max_total_chars), EVIDENCE_MAX_TOTAL_CHARS))

    excluded = {
        "turns": len(record.turns),
        "memory_tool_calls": 0,
        "tool_calls_without_command": 0,
        "mutations_without_content": 0,
        "duplicate_items": 0,
    }

    mutation_groups: list[list[_Candidate]] = []
    for mutation in record.file_mutations:
        extracted = _mutation_text(mutation)
        if extracted is None:
            excluded["mutations_without_content"] += 1
            continue
        category, texts = extracted
        group: list[_Candidate] = []
        for hunk_index, text in enumerate(texts):
            group.extend(
                _parts(
                    redactor(text),
                    mutation.span,
                    category,
                    mutation.path,
                    f"mutation[{mutation.ordinal}].hunk[{hunk_index}]",
                    max_item_chars,
                )
            )
        if group:
            mutation_groups.append(group)

    command_groups: list[list[_Candidate]] = []
    for call in record.tool_calls:
        if call.is_memory_tool:
            excluded["memory_tool_calls"] += 1
            continue
        command = command_text(call)
        if not command:
            excluded["tool_calls_without_command"] += 1
            continue
        body = f"{COMMAND_PREFIX}{command}"
        if call.result_text and call.result_text.strip():
            body = f"{body}\n{call.result_text}"
        group = _parts(
            redactor(body),
            call.span,
            COMMAND,
            f"{call.name or 'tool'}#{call.ordinal}",
            f"tool[{call.ordinal}].command",
            max_item_chars,
        )
        if group:
            command_groups.append(group)

    ordered: list[_Candidate] = []
    seen: set[str] = set()
    for candidate in _interleave(
        _round_robin(mutation_groups), _round_robin(command_groups)
    ):
        if candidate.text in seen:
            excluded["duplicate_items"] += 1
            continue
        seen.add(candidate.text)
        ordered.append(candidate)

    considered = {name: 0 for name in CATEGORIES}
    for candidate in ordered:
        considered[candidate.category] += 1

    kept = {name: 0 for name in CATEGORIES}
    items: list[EvidenceItem] = []
    dropped_chars = 0
    caps_hit: list[str] = []
    total = 0
    for candidate in ordered:
        if len(items) >= max_items:
            if "max_items" not in caps_hit:
                caps_hit.append("max_items")
            dropped_chars += len(candidate.text)
            continue
        if total + len(candidate.text) > max_total_chars:
            if "max_total_chars" not in caps_hit:
                caps_hit.append("max_total_chars")
            dropped_chars += len(candidate.text)
            continue
        items.append(
            EvidenceItem(
                text=candidate.text,
                span=candidate.span,
                category=candidate.category,
                origin=candidate.origin,
            )
        )
        kept[candidate.category] += 1
        total += len(candidate.text)

    texts = tuple(item.text for item in items)
    canonical, digest = canonical_evidence(texts)
    if canonical != texts:  # pragma: no cover - only reachable on server drift
        raise EvidenceError(
            "assembled items are not canonical for the server: "
            "living_memory.attestation canonicalisation has drifted from "
            "postsession.evidence._canonical, and every evidence_sha256 this "
            "assembler produced would change"
        )

    return EvidenceBundle(
        session_key=record.session_key,
        items=tuple(items),
        evidence_sha256=digest,
        considered=considered,
        kept=kept,
        dropped={name: considered[name] - kept[name] for name in CATEGORIES},
        dropped_chars=dropped_chars,
        excluded=excluded,
        caps={
            "max_items": max_items,
            "max_item_chars": max_item_chars,
            "max_total_chars": max_total_chars,
        },
        caps_hit=tuple(caps_hit),
    )


def _interleave(
    primary: Sequence[_T], secondary: Sequence[_T]
) -> list[_T]:
    """One from ``primary``, one from ``secondary``, then whatever remains."""

    out: list[_T] = []
    for left, right in zip_longest(primary, secondary):
        if left is not None:
            out.append(left)
        if right is not None:
            out.append(right)
    return out


def proposed_attestations(
    record: SessionRecord,
    *,
    bundle: EvidenceBundle | None = None,
    recall_event_ids: Iterable[str] | None = None,
    **kwargs: Any,
) -> list[ProposedOp]:
    """One validated ``attest`` op per recall event the session recorded.

    Every event of a session is graded against the same bundle -- the evidence
    is the session's artifacts, and which of them a given event's nodes appear
    in is precisely what the *server* decides. An event whose id the transcript
    never captured cannot be credited and is skipped; a session with no
    evidence yields no ops, because an op with no evidence is invalid by
    :meth:`ProposedOp.validate`.

    ``kwargs`` are forwarded to :func:`assemble_evidence` when ``bundle`` is
    not supplied.
    """

    if bundle is None:
        bundle = assemble_evidence(record, **kwargs)
    if not bundle:
        return []
    wanted = set(recall_event_ids) if recall_event_ids is not None else None
    evidence = bundle.evidence()
    payload_evidence = list(bundle.texts)
    ops: list[ProposedOp] = []
    for recall in record.recalls:
        event_id = recall.recall_event_id
        if not event_id or (wanted is not None and event_id not in wanted):
            continue
        ops.append(
            ProposedOp.for_session(
                "attest",
                {"recall_event_id": event_id, "evidence": list(payload_evidence)},
                record,
                recall.span,
                evidence,
            ).validate()
        )
    return ops
