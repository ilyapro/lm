"""Operator evidence within a user-role turn, in source-text coordinates."""

from collections.abc import Iterator
import hashlib
import json
import re

from .session import Turn


_CURRENT = "[current input] Addressed messages delivered into this turn:\n\n"
_DIRECT = "[operator] Mid-turn message from the operator, delivered live into this turn:\n\n"
_DIRECT_END = (
    "\n\nThis is a message, not a restatement of your prompt: it neither repeats nor replaces "
    "your goal, ownership limits or output contract. If it reports that one of them was "
    "amended, the amended version is authoritative over the one in your context. "
    "Weigh it, apply what is relevant, and continue the turn."
)
_SECTION = "\n\nCurrent addressed input ("
_END = "\n\nPreserve each source's author and recipient."
_CURRENT_CLOSE = " Inspect an UNKNOWN external outcome before any retry."
_BOUNDARY = re.compile(f"{re.escape(_SECTION)}|{re.escape(_END)}")
_OPERATOR_SECTION = re.compile(r"\n\nCurrent addressed input \([^\n]+; operator\):\n\n")


# loop_state joins these sections after the assignment (or the resume preface).
# Every dispatch has the reply address before the input block and the result
# contract after it. Neither the assignment nor a peer body escapes headings.
_LOOP_ADDRESS = re.compile(r"\nYour reply address is `[^`\n]+`\.\n")
_LOOP_RESULT = re.compile(r"\n\n## (?:Loop work result|Diagnostic result)\nCurrent loop owner: ([^\n]+)\n")
_LOOP_OPERATOR = re.compile(r"\n\n## Current operator input \((?P<source>/[^\n]+)\)\n\n")
_LOOP_BOUNDARY = re.compile(
    r"\n\n## (?:Current (?:operator input|peer input|input \(authorship unknown\)) \("
    r"|Current input unavailable\n|(?:Loop work result|Diagnostic result)\nCurrent loop owner: )"
)


def _sections(
    text: str, start: int, header_pattern: re.Pattern, boundaries: re.Pattern, *, owner: str | None = None
) -> Iterator[tuple[int, str]]:
    """Walk from an established input boundary. Only a bound body permits advancing."""
    while header := header_pattern.match(text, start):
        start = header.end()
        direct = owner is not None and header["source"].endswith("/operator_inject.md")
        if direct or not text.startswith("<!-- ae-augment ", start):
            if owner is not None and not direct:
                return
            boundary = boundaries.search(text, start)
            if boundary:
                yield start, text[start:boundary.start()]
            return

        end = text.find("\n", start)
        if end < 0 or text[end - 4:end] != " -->":
            return
        try:
            metadata = json.loads(text[start + len("<!-- ae-augment "):end - 4])
        except (ValueError, RecursionError):
            return
        if not (
            isinstance(metadata, dict)
            and metadata.get("author") == "operator"
            and metadata.get("operator_authenticated") is True
            and metadata.get("source") == "dashboard"
            and (owner is None or metadata.get("owner") == owner)
        ):
            return
        start = end + 1
        # An operator can quote a complete section header. The producer's
        # digest binds the actual end without treating quoted labels as facts.
        digest = hashlib.sha256()
        previous = start
        for boundary in boundaries.finditer(text, start):
            end = boundary.start()
            digest.update(text[previous:end].encode("utf-8", errors="surrogatepass"))
            if digest.hexdigest() == metadata.get("text_sha256"):
                yield start, text[start:end]
                start = end
                break
            previous = end
        else:
            return


def _loop_passages(text: str, address: re.Match, result: re.Match) -> Iterator[tuple[int, str]]:
    if (address.end() > result.start() or _LOOP_ADDRESS.search(text, address.end())
            or _LOOP_RESULT.search(text, result.end())):
        return
    try:
        owner = json.loads(result[1])
    except (ValueError, RecursionError):
        return
    if not isinstance(owner, str) or not owner:
        return
    start = address.end()
    if text.startswith("\n\n## Addressed operator requests and answers\n", start):
        # Request fields are JSON-escaped on physical lines in this section.
        start = text.find("\n\n", start + 2)
    if start >= 0:
        yield from _sections(text, start, _LOOP_OPERATOR, _LOOP_BOUNDARY, owner=owner)


def operator_passages(turn: Turn) -> Iterator[tuple[int, str]]:
    """Yield verbatim passages. A section body can imitate every later header."""
    if turn.role not in {"user", "operator", "human"}:
        return
    text = turn.text or ""
    if text.startswith(_DIRECT) and text.endswith(_DIRECT_END):
        yield len(_DIRECT), text[len(_DIRECT):-len(_DIRECT_END)]
        return
    # A tree assignment and its rendered guidance can both contain literal
    # source headings. The flattened prompt doesn't delimit the assignment.
    if (text.startswith("# Tree Goal Node\n\n## Your Goal\n")
            and "\n## Execution Context\n\n### Execution Worktree\n" in text):
        return
    address = _LOOP_ADDRESS.search(text)
    result = _LOOP_RESULT.search(text)
    live = text.startswith(_CURRENT)
    if not (live and text.endswith(_CURRENT_CLOSE)):
        if address and result:
            # A complete live frame can be quoted in the assignment. Walk
            # only from the renderer's unique address to its result contract.
            if not live:
                yield from _loop_passages(text, address, result)
            return
        if (address or result) and "## Current loop assignment\n" in text:
            return
    if not text.startswith(_CURRENT):
        if not text.startswith(_DIRECT) and not text.lstrip().startswith("[current input]"):
            yield 0, text
        return

    start = len(_CURRENT)
    if text.startswith(_DIRECT, start):
        start += len(_DIRECT)
        end = text.find(_DIRECT_END, start)
        boundary = _BOUNDARY.search(text, start)
        if end < 0 or (boundary and boundary.start() < end):
            return
        yield start, text[start:end]
        start = end + len(_DIRECT_END)

    yield from _sections(text, start, _OPERATOR_SECTION, _BOUNDARY)
