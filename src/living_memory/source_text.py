"""Conservative source boundaries shared by chunking, derivation and delivery."""

from __future__ import annotations

import re


def passage_spans(text: str) -> list[tuple[int, int, int]]:
    """Keep ambiguous punctuation inside a candidate, without an abbreviation list.

    Soft newlines and semicolons stay inside a passage. Blank lines separate
    paragraphs. A capitalized continuation suggests a new sentence. Periods
    additionally need a lowercase alphabetic terminal word longer than three letters: short,
    dotted, capitalized and numeric tokens can be abbreviations or references.
    This is a textual heuristic, not a parser; word-shaped abbreviations remain
    ambiguous. Conservative merging can leave late text in an oversized passage,
    which a bounded consumer may omit or explicitly mark as incomplete.
    """

    passages = []
    paragraph_start = 0
    breaks = [match.span() for match in re.finditer(r"\n\s*\n", text)]
    for paragraph_index, (paragraph_end, next_start) in enumerate([*breaks, (len(text), len(text))]):
        paragraph = text[paragraph_start:paragraph_end]
        def append(start: int, end: int) -> None:
            segment = paragraph[start:end]
            left = start + len(segment) - len(segment.lstrip())
            right = end - len(segment) + len(segment.rstrip())
            if left < right:
                passages.append((paragraph_start + left, paragraph_start + right, paragraph_index))

        start = 0
        for match in re.finditer(r"(?<!\S)(\S+)([.!?])\s+(?=\S)", paragraph):
            word, punctuation = match.groups()
            if not paragraph[match.end()].isupper():
                continue
            if punctuation == "." and not (word.isalpha() and word.islower() and len(word) > 3):
                continue
            append(start, match.start(2) + 1)
            start = match.end()
        append(start, len(paragraph))
        paragraph_start = next_start
    return passages
