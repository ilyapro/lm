"""Deterministic candidate mining over a finished session. No LLM here.

Why this stage exists at all
---------------------------
The judge is expensive and credulous: hand it a whole transcript and it will
happily summarise the session, which is precisely the ``done-journal-dump``
failure mode the stage is built to avoid. So the transcript is never handed to
the judge. Instead this module finds the *few* places in a session where an
expensive-to-re-derive fact could plausibly live, and hands the judge only
those spans.

Five classes are mined, and only five. Each one is a shape whose value survives
the session, i.e. a fact that a later reader cannot cheaply recover from the
diff or from ``git log``:

``resolved_failure``
    a command failed, files changed, the same command then went green. The
    durable fact is *what the failure actually was and what fixed it*.
``measurement``
    a command printed numbers and one of those exact numbers then shows up in a
    later assistant turn or in an added diff line -- so the number *drove a
    decision* rather than scrolling past.
``refuted_hypothesis``
    somebody stated an expectation and a later observation about the same
    identifiers contradicted it.
``external_contract``
    a probe of something outside this repository (a CLI flag, an HTTP endpoint,
    another tool's error vocabulary) whose reply documents behaviour.
``operator_correction``
    the human said "no, not like that", and the work changed afterwards.

Every candidate carries its evidence as *verbatim* quotes with a
:class:`~living_memory.postsession.session.SourceSpan` locator. That is what
keeps the rest of the pipeline honest: :mod:`living_memory.postsession.gate`
refuses any proposed fact whose hard identifiers are not present in these
quotes, so a judge that invents a path or a number is caught structurally
rather than by taste.

The text primitives at the bottom of this module (identifier extraction,
sentence splitting, verbatim windowing) are shared with ``gate.py`` on purpose:
"which identifiers does this text contain" must mean exactly the same thing to
the miner that produces evidence and to the gate that checks a fact against it,
or the grounding check would compare two different vocabularies.

Nothing here reads the Living Memory database, calls the server, or writes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .session import (
    Evidence,
    FileMutation,
    SessionRecord,
    SourceSpan,
    ToolCall,
    Turn,
)

__all__ = [
    "CANDIDATE_KINDS",
    "Candidate",
    "MiningConfig",
    "DEFAULT_MINING",
    "HARD_IDENTIFIER_KINDS",
    "IDENTIFIER_PATTERNS",
    "identifiers",
    "hard_identifiers",
    "sentences",
    "paragraphs",
    "quote_window",
    "tool_command",
    "mine",
    "mine_session",
    "mining_stats",
]

#: The only candidate classes this stage will ever propose from. Adding a class
#: means adding a miner *and* the evidence it quotes -- never a looser filter on
#: an existing one.
CANDIDATE_KINDS: tuple[str, ...] = (
    "resolved_failure",
    "measurement",
    "refuted_hypothesis",
    "external_contract",
    "operator_correction",
)


# --------------------------------------------------------------------------
# Shared text primitives (also imported by gate.py -- see module docstring)
# --------------------------------------------------------------------------

#: Named identifier extractors. ``hard`` kinds are the ones a fact may not
#: invent: a path, a flag, a dotted symbol, an error class, a distinctive
#: number, a ULID or a hex hash either appears in the mined evidence or the
#: fact claiming it is rejected. ``soft`` kinds (screaming-case constants,
#: backticked words) are informative but too easy to produce by accident to
#: carry a veto.
IDENTIFIER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("ulid", r"\b[0-9A-HJKMNP-TV-Z]{26}\b"),
    ("path", r"(?<![\w.@/-])(?:~?/)?(?:[\w.@+-]+/){1,}[\w.@+-]+"),
    (
        "file",
        r"(?<![\w./-])[\w.@+-]+\.(?:py|pyi|js|mjs|cjs|ts|tsx|jsx|json|jsonl|md|sh|bash|zsh"
        r"|sql|sqlite3?|ya?ml|toml|cfg|ini|conf|txt|csv|rs|go|java|kt|c|h|cc|cpp|rb|php"
        r"|html?|css|scss|lock|env|xml|proto|tf|Dockerfile)\b",
    ),
    ("flag", r"(?<![\w-])--[A-Za-z][\w-]*"),
    ("symbol", r"\b[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+\b|\b[A-Za-z_]\w{2,}\(\)"),
    ("errclass", r"\b[A-Z][A-Za-z0-9]*(?:Error|Exception|Warning|Failure|Fault)\b"),
    (
        "number",
        r"\b\d+(?:[.,]\d+)?\s?(?:%|ms|µs|us|ns|sec|s|min|MB|KB|GB|TB|B|kb|mb|gb"
        r"|req/s|rps|qps|px|pt|em|rem)\b|\b\d+\.\d+\b|\b\d{3,}\b",
    ),
    ("hash", r"\b(?![0-9]+\b)[0-9a-f]{7,40}\b"),
    ("const", r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"),
    ("quoted", r"`([^`\n]{2,80})`"),
)

#: Kinds whose members must be grounded in evidence. See the note above.
HARD_IDENTIFIER_KINDS: frozenset[str] = frozenset(
    {"ulid", "path", "file", "flag", "symbol", "errclass", "number", "hash"}
)

_COMPILED_IDENTIFIERS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern)) for name, pattern in IDENTIFIER_PATTERNS
)

#: Tokens that match an identifier pattern but say nothing. Kept deliberately
#: tiny: this is a stop-list for *noise*, never a place to encode expectations
#: about which insight a session should have produced.
_IDENTIFIER_NOISE: frozenset[str] = frozenset(
    {
        "e.g.",
        "i.e.",
        "etc.",
        "self.",
        "0.0",
        "1.0",
        "2.0",
        "0.5",
        "and/or",
        "n/a",
        "N/A",
        "TODO",
        "NOTE",
        "true/false",
    }
)


def identifiers(text: str, *, kinds: Iterable[str] | None = None) -> dict[str, tuple[str, ...]]:
    """Concrete tokens ``text`` names, grouped by kind and de-duplicated.

    Order inside a kind is first-appearance order, so the result is stable and
    a caller can quote the first hit without re-scanning.
    """

    wanted = set(kinds) if kinds is not None else None
    found: dict[str, dict[str, None]] = {}
    for name, pattern in _COMPILED_IDENTIFIERS:
        if wanted is not None and name not in wanted:
            continue
        bucket = found.setdefault(name, {})
        for match in pattern.finditer(text):
            token = (match.group(1) if match.lastindex else match.group(0)).strip()
            if len(token) < 2 or token in _IDENTIFIER_NOISE:
                continue
            bucket.setdefault(token, None)
    return {name: tuple(items) for name, items in found.items() if items}


def _anchor_identifiers(text: str) -> set[str]:
    """Hard identifiers minus bare numbers -- see :data:`_ANCHOR_KINDS`."""

    grouped = identifiers(text, kinds=_ANCHOR_KINDS)
    return {token for tokens in grouped.values() for token in tokens}


def hard_identifiers(text: str) -> tuple[str, ...]:
    """Flattened :data:`HARD_IDENTIFIER_KINDS` tokens, first-appearance order."""

    grouped = identifiers(text, kinds=HARD_IDENTIFIER_KINDS)
    seen: dict[str, None] = {}
    for name in HARD_IDENTIFIER_KINDS:
        for token in grouped.get(name, ()):
            seen.setdefault(token, None)
    return tuple(seen)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+(?=[A-ZА-ЯЁ`\"'(\[]|\d)|\n+")


def sentences(text: str) -> tuple[str, ...]:
    """Split into sentence-ish units, keeping list items as separate units."""

    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text or "")]
    return tuple(part for part in parts if part)


def paragraphs(text: str) -> tuple[str, ...]:
    """Blank-line separated blocks. Used by the gate's one-fact-per-trace rule."""

    parts = [part.strip() for part in re.split(r"\n\s*\n", text or "")]
    return tuple(part for part in parts if part)


def quote_window(
    text: str,
    start: int,
    end: int,
    *,
    before: int = 240,
    after: int = 480,
) -> str:
    """A verbatim slice of ``text`` around ``[start, end)``, snapped to lines.

    Verbatim is the whole point -- the gate later checks that the identifiers a
    proposed fact claims occur in these bytes, so the window may trim but must
    never paraphrase, re-order or normalise.
    """

    low = max(0, start - before)
    high = min(len(text), end + after)
    # Snap *inward* to whole lines: forward to the first line start at or
    # after ``low``, backward to the last line end at or before ``high``.
    # Snapping the other way would either re-add the context the caller asked
    # to drop or -- the bug this replaced -- cut the window at the first
    # newline after the match, discarding all of ``after``.
    newline = text.find("\n", low, start)
    if newline != -1:
        low = newline + 1
    newline = text.rfind("\n", end, high)
    if newline > end:
        high = newline
    return text[low:high].strip()


_COMMAND_KEYS = ("command", "cmd", "script", "shell_command", "code")
_PATH_KEYS = ("file_path", "path", "filename", "target_file", "notebook_path")


def tool_command(call: ToolCall) -> str:
    """The shell command (or file target) a tool call carries, as one string.

    Hosts disagree: Claude's ``Bash`` puts a string under ``command``, codex's
    shell puts an argv list under the same key, and file tools carry a path
    instead. All three collapse to one string here because every consumer only
    ever asks "is this the same work as that other call".
    """

    arguments = call.arguments or {}
    for key in _COMMAND_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (list, tuple)) and value:
            return " ".join(str(item) for item in value).strip()
    for key in _PATH_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return f"{call.name} {value.strip()}"
    return ""


_COMMAND_NOISE = re.compile(r"[\"'`]|\s+")
_COMMAND_WORD = re.compile(r"[A-Za-z0-9_./-]{2,}")


def _command_tokens(command: str) -> frozenset[str]:
    return frozenset(_COMMAND_WORD.findall(_COMMAND_NOISE.sub(" ", command)))


def _same_work(left: str, right: str, *, min_jaccard: float) -> bool:
    """Are two commands the same piece of work re-run?

    Jaccard over command word tokens rather than string equality: a re-run
    routinely gains a ``-x`` or drops a ``| head``, and requiring byte equality
    would miss most real red-green pairs.
    """

    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens, right_tokens = _command_tokens(left), _command_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) >= min_jaccard


# --------------------------------------------------------------------------
# Marker vocabularies
# --------------------------------------------------------------------------

#: A tool result that says the work failed. ``ok is False`` is honoured too,
#: but only Claude records it, so the text markers carry the other five sources.
FAILURE_MARKERS = re.compile(
    r"Traceback \(most recent call last\)"
    r"|^E\s{3}\w"
    r"|\b(?:AssertionError|ModuleNotFoundError|ImportError|SyntaxError|TypeError"
    r"|ValueError|KeyError|AttributeError|RuntimeError|OSError|IOError"
    r"|IndexError|NameError|PermissionError|TimeoutError)\b"
    r"|\berror\[E\d+\]"
    r"|^\s*(?:FAILED|FAIL|ERROR)\b"
    r"|\b\d+ failed\b"
    r"|\bTests?:\s+\d+ failed"
    r"|\bnpm ERR!"
    r"|\bfatal:\s"
    r"|\bpanic:\s"
    r"|\bSegmentation fault\b"
    r"|\bcommand not found\b"
    r"|\bNo such file or directory\b"
    r"|\bPermission denied\b"
    r"|\bExit code [1-9]\d*\b"
    r"|\bexited with (?:code )?[1-9]\d*\b"
    r"|\bexit status [1-9]\d*\b",
    re.MULTILINE,
)

#: A tool result that says the same work now succeeds.
SUCCESS_MARKERS = re.compile(
    r"\b\d+ passed\b"
    r"|\bpassed\b(?!\s*[:=]\s*0)"
    r"|\bOK\b"
    r"|\ball tests? pass"
    r"|\b0 failed\b"
    r"|\bno errors?\b"
    r"|\bSuccess(?:fully)?\b"
    r"|\bExit code 0\b"
    r"|\bbuild succeeded\b",
    re.IGNORECASE,
)

#: Reply text that documents somebody else's contract rather than this repo's.
CONTRACT_MARKERS = re.compile(
    r"^\s*(?:Usage|usage|USAGE|Options|OPTIONS|Commands|Flags|Synopsis)\s*:"
    r"|\bunknown (?:option|flag|argument|command)\b"
    r"|\bunrecognized (?:option|arguments?)\b"
    r"|\berror: unexpected argument\b"
    r"|\binvalid choice\b"
    r"|\bno such option\b"
    r"|\bis not a recognized\b"
    r"|\bHTTP/\d(?:\.\d)? [45]\d\d\b"
    r"|\b(?:401|403|404|409|422|429|500|502|503) (?:Unauthorized|Forbidden|Not Found"
    r"|Conflict|Unprocessable|Too Many Requests|Internal Server Error|Bad Gateway"
    r"|Service Unavailable)\b"
    r"|\"error\"\s*:"
    r"|\bdeprecated\b.*\buse\b"
    r"|\brequires? (?:the )?(?:--|-)\w",
    re.MULTILINE,
)

#: Commands that go and ask something outside this checkout how it behaves.
PROBE_COMMANDS = re.compile(
    r"\b(?:curl|wget|https?_proxy|gh\s+api|gh\s+\w+|glab|aws|az|gcloud|kubectl|docker"
    r"|psql|mysql|redis-cli|openstack|systemctl|journalctl|dpkg|apt-get|apt|brew"
    r"|npm\s+(?:view|info|ls|outdated)|pip\s+(?:show|index|download)|cargo\s+search"
    r"|man\s+\w+)\b"
    r"|\s--help\b|\s--version\b|\s-h\s*$",
)

#: Commands that only read this checkout. A ``Usage:`` block printed by ``sed``
#: is a local file quoting itself, not an external contract -- the miner would
#: otherwise fire on every ``sed -n '1,80p' scripts/foo.py`` in the corpus.
LOCAL_READERS = frozenset(
    {
        "sed", "cat", "head", "tail", "less", "more", "bat", "grep", "rg", "ag",
        "awk", "find", "ls", "wc", "tree", "diff", "git", "python", "python3",
        "node", "jq", "echo", "printf", "cd", "pwd", "Read", "Glob", "Grep",
    }
)

#: Commands whose whole purpose is producing a number.
MEASURE_COMMANDS = re.compile(
    r"\b(?:pytest|jest|vitest|mocha|tox|nose|ctest|phpunit|rspec"
    r"|go\s+test|cargo\s+(?:test|bench)|npm\s+(?:test|run\s+test)|yarn\s+test"
    r"|wc|du|df|nproc|free|uptime|hyperfine|wrk|ab|siege|k6|locust"
    r"|time|/usr/bin/time|perf|valgrind|coverage|nyc|lcov"
    r"|benchmark|bench|--durations?|--stats|--count|--coverage"
    r"|count\(|COUNT\(|SELECT\s+count|EXPLAIN\s+ANALYZE)\b",
    re.IGNORECASE,
)

#: A number carrying a unit, a percentage or real precision -- the shape that
#: means "we measured something" rather than "a digit occurred".
UNIT_NUMBER = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:%|ms|µs|us|ns|sec|s|min|MB|KB|GB|TB|kb|mb|gb"
    r"|req/s|rps|qps)\b|\b\d+\.\d{2,}\b"
)

#: File modes, years and epoch stamps look like measurements and never are.
_NUMBER_NOISE_LITERALS = frozenset({"100644", "100755", "120000", "040000", "160000"})

#: An expectation stated before the observation that will test it.
HYPOTHESIS_MARKERS = re.compile(
    r"\b(?:I (?:expect|assume|believe|suspect)|we (?:expect|assume)|expected to"
    r"|should (?:be|return|contain|work|fail|pass|have)|ought to|presumably"
    r"|my (?:guess|hypothesis|assumption)|the hypothesis|probably (?:is|the)"
    r"|looks like it|must be (?:the|a|an)"
    r"|ожида\w+|предполага\w+|должн[оаы]\b|скорее всего|видимо|наверн\w+)\b",
    re.IGNORECASE,
)

#: An observation that says the expectation was wrong. Deliberately narrow:
#: ``however``/``instead``/bare "error" words fire in every second paragraph of
#: a working transcript and drown the class they are supposed to find.
REFUTATION_MARKERS = re.compile(
    r"\b(?:actually(?: it| the| there)?|in fact|turns out|turned out"
    r"|contrary to|not the case|was wrong|is wrong|proved wrong|disproved"
    r"|does ?n[o']t actually|did ?n[o']t actually|never actually"
    r"|contradicts|refutes|the opposite"
    r"|на самом деле|оказалось|оказался|оказалась|вопреки|опроверг\w*"
    r"|не соответствует действительности|предположение неверно)\b",
    re.IGNORECASE,
)

#: Identifier kinds strong enough to link a hypothesis to its refutation. A
#: bare number is not: two paragraphs both mentioning ``2026`` share nothing,
#: and on the first train pass that alone produced every false pairing.
_ANCHOR_KINDS: frozenset[str] = frozenset(
    {"ulid", "path", "file", "flag", "symbol", "errclass", "hash", "const", "quoted"}
)

#: A human telling the agent it got it wrong. Operator turns only.
CORRECTION_MARKERS = re.compile(
    r"^\s*(?:no|nope|wrong|stop)\b[,.!\s]"
    r"|\b(?:that'?s (?:not|wrong)|not what I (?:said|asked|meant)|I said\b"
    r"|don'?t do that|do not do that|you (?:misunderstood|got it wrong|were wrong)"
    r"|read it again|why did you|revert that|undo that|that is incorrect"
    r"|no need to|I meant\b)\b"
    r"|^\s*(?:нет|не так|неправильно|неверно|стоп)\b[,.!\s]"
    r"|\b(?:не надо|не нужно|я же (?:сказал|просил|говорил)|я просил|я имел в виду"
    r"|ты (?:не понял|неправильно|ошиб\w+)|верни как было|откати|это не то"
    r"|перечитай|почему ты|зачем ты|разве не|не то\b)\b",
    re.IGNORECASE | re.MULTILINE,
)

#: Numbers worth calling a measurement: a unit, a decimal, or >= 3 digits.
_MEASURED_NUMBER = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:%|ms|µs|us|ns|sec|s|min|MB|KB|GB|TB|kb|mb|gb"
    r"|req/s|rps|qps|px)\b|\b\d+\.\d{2,}\b|\b\d{3,}\b"
)


# --------------------------------------------------------------------------
# Configuration and the candidate record
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MiningConfig:
    """Miner thresholds. Tuned on the ``train`` split only.

    Every field is a *recall* knob: mining is allowed to be generous because
    the judge and the gate downstream are what decide. Making these stricter
    trades away candidates the gate could have accepted; making them looser
    only costs judge calls.
    """

    #: Command-token Jaccard at which a later run counts as the same work.
    rerun_min_jaccard: float = 0.6
    #: How far ahead (in transcript records) a miner looks for the other half.
    lookahead_records: int = 400
    #: Maximum candidates emitted per kind per session.
    max_per_kind: int = 6
    #: Maximum candidates emitted per session across all kinds.
    max_per_session: int = 20
    #: Characters of context kept on either side of a matched marker.
    quote_before: int = 240
    quote_after: int = 480
    #: A measured number must be re-used downstream to prove it drove anything.
    require_number_reuse: bool = True
    #: Shared identifiers required to link a hypothesis to its refutation.
    hypothesis_min_shared: int = 1
    #: Refutations arrive close to the claim they break; a wide window here
    #: pairs unrelated paragraphs that merely share a file name.
    refutation_lookahead: int = 150


DEFAULT_MINING = MiningConfig()


@dataclass(frozen=True, slots=True)
class Candidate:
    """One mined span the judge may be asked about.

    ``evidence`` is verbatim and is the *only* thing the judge sees of the
    session, which is what bounds the payload and what makes hallucination
    detectable: the gate re-checks the emitted fact against exactly these
    quotes.
    """

    kind: str
    session_key: str
    ordinal: int
    span: SourceSpan
    evidence: tuple[Evidence, ...]
    identifiers: tuple[str, ...] = ()
    signals: dict[str, Any] = field(default_factory=dict)
    at: str | None = None
    position: float = 0.0

    @property
    def evidence_text(self) -> str:
        """All quotes joined -- the corpus the gate grounds a fact against."""

        return "\n".join(item.quote for item in self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "session_key": self.session_key,
            "ordinal": self.ordinal,
            "span": self.span.locator(),
            "evidence": [
                {"quote": item.quote, "locator": item.locator} for item in self.evidence
            ],
            "identifiers": list(self.identifiers),
            "signals": dict(self.signals),
            "at": self.at,
            "position": self.position,
        }


# --------------------------------------------------------------------------
# Ordering helpers
# --------------------------------------------------------------------------


def _at(item: ToolCall | Turn | FileMutation) -> int:
    """Transcript record ordinal -- the one clock every source shares."""

    return item.span.record_index


def _position(record: SessionRecord, index: int) -> float:
    return round(index / max(1, record.record_count - 1), 4) if record.record_count else 0.0


def _evidence(span: SourceSpan, quote: str) -> Evidence:
    return Evidence(quote=quote, locator=span.locator())


def _result_span(call: ToolCall) -> SourceSpan:
    base = call.span
    return SourceSpan(base.source, base.path, base.record_index, base.field or "result")


def _turn_span(turn: Turn) -> SourceSpan:
    base = turn.span
    return SourceSpan(base.source, base.path, base.record_index, base.field or "text")


# --------------------------------------------------------------------------
# The five miners
# --------------------------------------------------------------------------


def _mine_resolved_failures(
    record: SessionRecord, config: MiningConfig
) -> Iterator[Candidate]:
    """Red -> edit -> green over the same command.

    All three legs are required. A failure with no fix is an open problem, not
    a lesson; a fix with no green re-run is an unverified plan; and an edit
    between them is what distinguishes "we fixed it" from "we ran it twice".
    """

    calls = [call for call in record.tool_calls if not call.is_memory_tool]
    mutation_points = sorted(_at(item) for item in record.file_mutations)
    for index, call in enumerate(calls):
        text = call.result_text or ""
        failure = FAILURE_MARKERS.search(text)
        if failure is None and call.ok is not False:
            continue
        command = tool_command(call)
        if not command:
            continue
        start = _at(call)
        for later in calls[index + 1 :]:
            gap = _at(later) - start
            if gap <= 0:
                continue
            if gap > config.lookahead_records:
                break
            if not _same_work(command, tool_command(later), min_jaccard=config.rerun_min_jaccard):
                continue
            later_text = later.result_text or ""
            if FAILURE_MARKERS.search(later_text) or later.ok is False:
                continue
            if not (SUCCESS_MARKERS.search(later_text) or later.ok is True):
                continue
            fixed_at = [point for point in mutation_points if start < point < _at(later)]
            if not fixed_at:
                continue
            failure_quote = (
                quote_window(
                    text,
                    failure.start(),
                    failure.end(),
                    before=config.quote_before,
                    after=config.quote_after,
                )
                if failure is not None
                else text[: config.quote_after].strip()
            )
            changed = tuple(
                mutation.path
                for mutation in record.file_mutations
                if start < _at(mutation) < _at(later)
            )
            evidence = (
                _evidence(call.span, f"$ {command}"),
                _evidence(_result_span(call), failure_quote),
                _evidence(later.span, f"$ {tool_command(later)}"),
                _evidence(
                    _result_span(later),
                    later_text[: config.quote_after].strip() or "(no output)",
                ),
            )
            yield Candidate(
                kind="resolved_failure",
                session_key=record.session_key,
                ordinal=0,
                span=_result_span(call),
                evidence=evidence,
                identifiers=hard_identifiers(failure_quote),
                signals={
                    "command": command,
                    "failure_marker": failure.group(0).strip() if failure else "exit_status",
                    "changed_paths": list(dict.fromkeys(changed))[:12],
                    "green_record_index": _at(later),
                },
                at=call.at,
                position=_position(record, start),
            )
            break


def _noise_number(token: str, text: str) -> bool:
    """Numbers that look measured and are not.

    Three families, all seen firing on the train split: git file modes
    (``100644``), calendar years (``2026`` matched 8 times in one ``git show``),
    and epoch stamps. A token repeated all over its own output is boilerplate
    too -- a real measurement is quoted once, a mode line is quoted per file.
    """

    bare = token.strip().rstrip("%").strip()
    if bare in _NUMBER_NOISE_LITERALS:
        return True
    if bare.isdigit():
        value = int(bare)
        if len(bare) == 4 and 1900 <= value <= 2100:
            return True
        if 10 <= len(bare) <= 13:
            return True
    return text.count(token) > 3


def _mine_measurements(record: SessionRecord, config: MiningConfig) -> Iterator[Candidate]:
    """Numbers a command printed that a later decision then quoted back.

    Two requirements, and both are load-bearing:

    *the command measured something* -- either it is a measuring command
    (``pytest``, ``wc``, ``hyperfine``, ``SELECT count``) or the number carries
    a unit. Without this the miner fires on every ``git show`` in the corpus,
    which is what the first train pass did: 539 candidates, mostly file modes.

    *the number was re-used* -- it reappears verbatim in a later assistant turn
    or in an added diff line. Any command prints numbers; only a number a
    decision quoted back was actually used, and only then is the measurement
    worth a durable trace.
    """

    later_texts: list[tuple[int, str, SourceSpan]] = [
        (_at(turn), turn.text or "", _turn_span(turn))
        for turn in record.turns
        if turn.role == "assistant"
    ]
    for mutation in record.file_mutations:
        added = "\n".join(
            line[1:]
            for line in (mutation.unified_diff or "").splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        if added:
            later_texts.append((_at(mutation), added, mutation.span))
    later_texts.sort()

    for call in record.tool_calls:
        if call.is_memory_tool:
            continue
        text = call.result_text or ""
        command = tool_command(call)
        if not text or not command:
            continue
        measuring = MEASURE_COMMANDS.search(command) is not None
        command_numbers = set(_MEASURED_NUMBER.findall(command))
        matches = [
            match
            for match in _MEASURED_NUMBER.finditer(text)
            if match.group(0) not in command_numbers
            and not _noise_number(match.group(0), text)
            and (measuring or UNIT_NUMBER.fullmatch(match.group(0)) is not None)
        ]
        if not matches:
            continue
        start = _at(call)
        reused: tuple[str, SourceSpan, str] | None = None
        for match in matches:
            token = match.group(0)
            if len(token.strip()) < 3:
                continue
            for index, later_text, span in later_texts:
                if index <= start or index - start > config.lookahead_records:
                    continue
                position = later_text.find(token)
                if position == -1:
                    continue
                reused = (
                    token,
                    span,
                    quote_window(
                        later_text,
                        position,
                        position + len(token),
                        before=config.quote_before,
                        after=config.quote_before,
                    ),
                )
                break
            if reused is not None:
                break
        if reused is None and config.require_number_reuse:
            continue
        token, reuse_span, reuse_quote = reused or ("", call.span, "")
        anchor = next(
            (match for match in matches if match.group(0) == token), matches[0]
        )
        result_quote = quote_window(
            text,
            anchor.start(),
            anchor.end(),
            before=config.quote_before,
            after=config.quote_after,
        )
        evidence = [
            _evidence(call.span, f"$ {command}" if command else call.name),
            _evidence(_result_span(call), result_quote),
        ]
        if reuse_quote:
            evidence.append(_evidence(reuse_span, reuse_quote))
        yield Candidate(
            kind="measurement",
            session_key=record.session_key,
            ordinal=0,
            span=_result_span(call),
            evidence=tuple(evidence),
            identifiers=hard_identifiers(result_quote),
            signals={
                "command": command,
                "reused_number": token,
                "measuring_command": measuring,
                "numbers": [match.group(0) for match in matches[:8]],
            },
            at=call.at,
            position=_position(record, start),
        )


def _mine_refuted_hypotheses(
    record: SessionRecord, config: MiningConfig
) -> Iterator[Candidate]:
    """A stated expectation that a later observation about the same thing broke."""

    observations: list[tuple[int, str, SourceSpan]] = [
        (_at(turn), turn.text or "", _turn_span(turn))
        for turn in record.turns
        if turn.role in ("assistant", "user")
    ]
    observations.extend(
        (_at(call), call.result_text or "", _result_span(call))
        for call in record.tool_calls
        if call.result_text and not call.is_memory_tool
    )
    observations.sort()

    for turn in record.turns:
        if turn.role != "assistant":
            continue
        text = turn.text or ""
        hit = HYPOTHESIS_MARKERS.search(text)
        if hit is None:
            continue
        claim = quote_window(
            text, hit.start(), hit.end(), before=config.quote_before, after=config.quote_after
        )
        claim_ids = _anchor_identifiers(claim)
        if not claim_ids:
            continue
        start = _at(turn)
        paired = False
        for index, observed, span in observations:
            if paired:
                break
            if index <= start or index - start > config.refutation_lookahead:
                continue
            # Every refutation match, not just the first: a long tool result
            # routinely says "actually" once in passing before it says the
            # thing that breaks the claim.
            for refutation in REFUTATION_MARKERS.finditer(observed):
                window = quote_window(
                    observed,
                    refutation.start(),
                    refutation.end(),
                    before=config.quote_before,
                    after=config.quote_after,
                )
                shared = claim_ids & _anchor_identifiers(window)
                if len(shared) < config.hypothesis_min_shared:
                    continue
                yield Candidate(
                    kind="refuted_hypothesis",
                    session_key=record.session_key,
                    ordinal=0,
                    span=_turn_span(turn),
                    evidence=(_evidence(_turn_span(turn), claim), _evidence(span, window)),
                    identifiers=tuple(sorted(shared)),
                    signals={
                        "hypothesis_marker": hit.group(0).strip(),
                        "refutation_marker": refutation.group(0).strip(),
                        "shared_identifiers": sorted(shared)[:8],
                    },
                    at=turn.at,
                    position=_position(record, start),
                )
                paired = True
                break


def _repo_owned(token: str, record: SessionRecord) -> bool:
    """Is this token something the session's own checkout defines?"""

    if not token:
        return False
    repo = record.repo or record.cwd or ""
    if repo and token.startswith(repo):
        return True
    for path in record.mutated_paths:
        if token == path or path.endswith(f"/{token}") or token.endswith(f"/{path}"):
            return True
    return False


_SEGMENT_SPLIT = re.compile(r"[;|]|&&|\|\|")


def _probe_subject(command: str, at: int) -> str:
    """First word of the pipeline segment the probe matched in.

    ``cd repo && cargo build --help`` probes ``cargo``, not ``cd``; taking the
    first word of the whole command line made the train pass credit ``sed``
    with every external contract it found.
    """

    start = 0
    for match in _SEGMENT_SPLIT.finditer(command):
        if match.end() > at:
            break
        start = match.end()
    words = command[start:].split()
    for word in words:
        if "=" in word and not word.startswith("-"):
            continue  # leading VAR=value assignments
        return word.strip("()`\"'")
    return ""


def _mine_external_contracts(
    record: SessionRecord, config: MiningConfig
) -> Iterator[Candidate]:
    """Behaviour of somebody else's CLI/API, learned by asking it.

    Both halves are required: the command has to *ask* something outside this
    checkout (``--help``, ``curl``, ``gh api``, ``kubectl``) and the reply has
    to *document* a contract (a usage block, ``unknown option``, an HTTP 4xx).
    The subject must not be a local reader either -- ``sed -n '1,80p'`` on a
    repository file that happens to contain the word ``Usage:`` is that file
    quoting itself, and a later reader re-derives it by opening the file.
    """

    for call in record.tool_calls:
        if call.is_memory_tool:
            continue
        command = tool_command(call)
        text = call.result_text or ""
        if not command or not text:
            continue
        probe = PROBE_COMMANDS.search(command)
        documented = CONTRACT_MARKERS.search(text)
        if probe is None or documented is None:
            continue
        subject = _probe_subject(command, probe.start())
        if not subject or subject in LOCAL_READERS or _repo_owned(subject, record):
            continue
        window = quote_window(
            text,
            documented.start(),
            documented.end(),
            before=config.quote_before,
            after=config.quote_after,
        )
        yield Candidate(
            kind="external_contract",
            session_key=record.session_key,
            ordinal=0,
            span=_result_span(call),
            evidence=(
                _evidence(call.span, f"$ {command}" if command else call.name),
                _evidence(_result_span(call), window),
            ),
            identifiers=hard_identifiers(window),
            signals={
                "command": command,
                "subject": subject,
                "probe": bool(probe),
                "contract_marker": documented.group(0).strip()[:80],
            },
            at=call.at,
            position=_position(record, _at(call)),
        )


def _mine_operator_corrections(
    record: SessionRecord, config: MiningConfig
) -> Iterator[Candidate]:
    """The human said the agent had it wrong, and the work changed after.

    Requiring a *later* file mutation or tool call is what keeps a rhetorical
    "no" out: a correction nobody acted on taught the session nothing.
    """

    after_points = sorted(
        [_at(item) for item in record.file_mutations]
        + [_at(call) for call in record.tool_calls if not call.is_memory_tool]
    )
    turns = record.turns
    for position_index, turn in enumerate(turns):
        if turn.role != "user":
            continue
        text = turn.text or ""
        hit = CORRECTION_MARKERS.search(text)
        if hit is None:
            continue
        prior = next(
            (item for item in reversed(turns[:position_index]) if item.role == "assistant"),
            None,
        )
        if prior is None:
            continue
        start = _at(turn)
        if not any(start < point <= start + config.lookahead_records for point in after_points):
            continue
        correction = quote_window(
            text, hit.start(), hit.end(), before=config.quote_before, after=config.quote_after
        )
        prior_text = prior.text or ""
        claim = prior_text[-config.quote_after :].strip()
        yield Candidate(
            kind="operator_correction",
            session_key=record.session_key,
            ordinal=0,
            span=_turn_span(turn),
            evidence=(
                _evidence(_turn_span(prior), claim or "(empty assistant turn)"),
                _evidence(_turn_span(turn), correction),
            ),
            identifiers=hard_identifiers(f"{claim}\n{correction}"),
            signals={
                "correction_marker": hit.group(0).strip()[:80],
                "prior_turn_index": prior.index,
            },
            at=turn.at,
            position=_position(record, start),
        )


_MINERS: tuple[tuple[str, Any], ...] = (
    ("resolved_failure", _mine_resolved_failures),
    ("measurement", _mine_measurements),
    ("refuted_hypothesis", _mine_refuted_hypotheses),
    ("external_contract", _mine_external_contracts),
    ("operator_correction", _mine_operator_corrections),
)


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def mine(record: SessionRecord, *, config: MiningConfig = DEFAULT_MINING) -> list[Candidate]:
    """Every candidate in one session, in transcript order with fixed ordinals.

    Candidates are capped per kind before the global cap so a session with a
    thousand red-green pairs cannot crowd out its single refuted hypothesis --
    the rarer classes are exactly the expensive ones.
    """

    collected: list[Candidate] = []
    for _kind, miner in _MINERS:
        taken = 0
        for candidate in miner(record, config):
            collected.append(candidate)
            taken += 1
            if taken >= config.max_per_kind:
                break
    collected.sort(key=lambda item: (item.span.record_index, item.kind, item.span.field))
    collected = _dedupe_spans(collected)[: config.max_per_session]
    return [
        Candidate(
            kind=item.kind,
            session_key=item.session_key,
            ordinal=index,
            span=item.span,
            evidence=item.evidence,
            identifiers=item.identifiers,
            signals=item.signals,
            at=item.at,
            position=item.position,
        )
        for index, item in enumerate(collected)
    ]


def _dedupe_spans(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Keep one candidate per (kind, locator): two miners often see one event."""

    seen: dict[tuple[str, str], Candidate] = {}
    for candidate in candidates:
        seen.setdefault((candidate.kind, candidate.span.locator()), candidate)
    return list(seen.values())


def mine_session(
    record: SessionRecord, *, config: MiningConfig = DEFAULT_MINING
) -> list[Candidate]:
    """Alias kept for readability at call sites that already have a record."""

    return mine(record, config=config)


def mining_stats(candidates: Iterable[Candidate]) -> dict[str, Any]:
    """Counts by kind plus evidence volume -- what the run report publishes."""

    by_kind: dict[str, int] = {kind: 0 for kind in CANDIDATE_KINDS}
    quotes = 0
    characters = 0
    for candidate in candidates:
        by_kind[candidate.kind] = by_kind.get(candidate.kind, 0) + 1
        quotes += len(candidate.evidence)
        characters += len(candidate.evidence_text)
    total = sum(by_kind.values())
    return {
        "candidates": total,
        "by_kind": by_kind,
        "evidence_quotes": quotes,
        "evidence_chars": characters,
        "mean_evidence_chars": round(characters / total, 1) if total else 0.0,
    }


def evidence_index(candidates: Iterable[Candidate]) -> Mapping[int, Candidate]:
    """Ordinal -> candidate, for joining judge answers back to their span."""

    return {candidate.ordinal: candidate for candidate in candidates}
