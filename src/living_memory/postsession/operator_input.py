"""Operator evidence within a user-role turn, in source-text coordinates."""

from collections.abc import Iterator
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
_BOUNDARY = re.compile(f"{re.escape(_SECTION)}|{re.escape(_END)}")
_OPERATOR_SECTION = re.compile(r"Current addressed input \([^\n]+; operator\):\n\n")


def operator_passages(turn: Turn) -> Iterator[tuple[int, str]]:
    """Yield verbatim passages. A section body can imitate every later header."""
    if turn.role not in {"user", "operator", "human"}:
        return
    text = turn.text or ""
    if text.startswith(_DIRECT):
        if text.endswith(_DIRECT_END):
            yield len(_DIRECT), text[len(_DIRECT):-len(_DIRECT_END)]
        return
    if not text.lstrip().startswith("[current input]"):
        yield 0, text
        return
    if not text.startswith(_CURRENT):
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

    # AE places an optional direct message first, then appends unescaped
    # addressed sources. Never search forward for an operator section after
    # reaching peer, unknown or unrecognised input.
    while text.startswith(_SECTION, start):
        header = _OPERATOR_SECTION.match(text, start + 2)
        if not header:
            return
        start = header.end()
        if text.startswith("<!-- ae-augment ", start):
            end = text.find("\n", start)
            if end < 0 or not text.startswith(" -->", end - 4):
                return
            start = end + 1
        boundary = _BOUNDARY.search(text, start)
        if not boundary:
            return
        yield start, text[start:boundary.start()]
        start = boundary.start()
