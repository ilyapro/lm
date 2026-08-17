"""Deterministic, boundary-aware text chunking for embedding windows.

Why this module exists
---------------------
The recall encoder is ``paraphrase-multilingual-MiniLM-L12-v2`` configured with
``max_seq_length = 128``. Everything past token 128 of a node is invisible to
its vector: measured on the live database on 2026-08-17, 67% of active nodes are
longer than that (median 235 tokens), and
``cos(vector(full node), vector(first 128 tokens)) == 1.000``. Raising the limit
to 512 was measured and rejected (mean cos 0.479 versus 0.489, and 5x slower
encoding); splitting a node into overlapping ~128-token windows scored 0.954 on
the same probe. This module is only the splitter: it never embeds, stores, or
retrieves anything.

Token accounting
----------------
``max_tokens`` is the encoder's *whole* sequence budget, so ``special_tokens``
(2: ``<s>`` and ``</s>``) is subtracted from it to obtain the content window.
With the defaults every chunk carries at most 126 content tokens and therefore
survives ``max_seq_length = 128`` untruncated — leaving those two slots unspent
is what keeps the tail of the last chunk of a node visible.

Tokenizers
----------
Token counts must match what the encoder actually sees, so the preferred counter
is the model's own tokenizer, loaded from the ``tokenizer.json`` of the local
sentence-transformers snapshot through the pure-Rust ``tokenizers`` package — no
torch and no ``sentence_transformers`` import. When that is unavailable (the test
suite runs with ``LIVING_MEMORY_EMBEDDING_BACKEND=hash`` and deliberately avoids
loading torch, see ``scripts/test.sh``) the module falls back to
:class:`HeuristicTokenizer`, a documented character-rate approximation.

Chunking is deterministic for a given (text, arguments, tokenizer). Switching
tokenizer backends changes token counts and therefore chunk boundaries, so a
pipeline must use the same backend when it backfills and when it queries.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
from typing import Protocol

# Package-internal reuse, on purpose: the chunker must agree with the encoder
# about where the model lives on disk and about what counts as "no real encoder
# is in play", so both facts come from embeddings.py instead of being restated.
from living_memory.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    _HASH_BACKENDS as _EMBEDDING_HASH_BACKENDS,
    _resolve_local_model_source,
)


MODEL_MAX_SEQ_LENGTH = 128
"""``max_seq_length`` the encoder is configured with (verified live)."""

ENCODER_SPECIAL_TOKENS = 2
"""Sequence slots the encoder spends on ``<s>``/``</s>`` around the content."""

DEFAULT_MAX_TOKENS = MODEL_MAX_SEQ_LENGTH
DEFAULT_OVERLAP_TOKENS = 32

TOKENIZER_ENV_VAR = "LIVING_MEMORY_CHUNK_TOKENIZER"

_LOG = logging.getLogger(__name__)
_WARNED: set[str] = set()

_HEURISTIC_BACKENDS = frozenset({"heuristic", "fallback", "hash", "none"}) | frozenset(
    _EMBEDDING_HASH_BACKENDS
)
_MODEL_BACKENDS = frozenset({"model", "real", "encoder", "sentence-transformers"})

# A cut prefers structure, but not at any price: a paragraph or sentence break is
# only taken when it fills at least this much of the window, so respecting
# structure can never waste more than half a window.
_MIN_FILL_RATIO = 0.5

# How many tokens a hard cut may give up to land on a word start instead of in
# the middle of a word — roughly the length of one long word in subwords.
_HARD_CUT_WORD_SLACK = 6

# Paragraph break: a blank line (two or more newlines, tolerating indentation).
_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*(?:\n[ \t]*)+")

# Sentence break: a run of sentence-final punctuation, optional closing quotes or
# brackets, then whitespace — or a single line break, which ends a line-shaped
# unit such as a bullet in a list. Russian and English share this punctuation;
# the pattern is Unicode-wide and never assumes ASCII.
_SENTENCE_BREAK_RE = re.compile(r"[.!?…‽⁇⁈⁉]+[\"'»”’)\]]*\s+|\n")

_TOKENIZER_CACHE: dict[tuple[str, str], "Tokenizer"] = {}

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_OVERLAP_TOKENS",
    "ENCODER_SPECIAL_TOKENS",
    "HeuristicTokenizer",
    "MODEL_MAX_SEQ_LENGTH",
    "ModelTokenizer",
    "TOKENIZER_ENV_VAR",
    "TextChunk",
    "TokenSpan",
    "Tokenizer",
    "chunk_text",
    "count_tokens",
    "get_tokenizer",
    "reset_tokenizer_cache",
]


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """Half-open character range ``[start, end)`` covered by one token."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One embedding window cut out of a larger text.

    ``token_start``/``token_end`` are half-open indices into the token stream of
    the *whole* input, so consecutive chunks overlap by
    ``previous.token_end - current.token_start`` tokens and their union covers
    every token of the input. ``text`` is exactly
    ``source[char_start:char_end]``.
    """

    text: str
    chunk_index: int
    token_start: int
    token_end: int
    char_start: int
    char_end: int

    @property
    def token_count(self) -> int:
        """Number of tokens in this window."""

        return self.token_end - self.token_start


class Tokenizer(Protocol):
    """Minimal tokenizer contract the chunker needs: named, span-producing."""

    @property
    def name(self) -> str:
        """Stable identifier of the tokenization used, for provenance."""

    def token_spans(self, text: str) -> list[TokenSpan]:
        """Return non-empty, non-overlapping, ascending token spans."""


class HeuristicTokenizer:
    """Torch-free approximation of the model's SentencePiece tokenizer.

    Words are split into fixed-size character pieces and each punctuation
    character counts as one token. The rates below were calibrated against the
    real ``paraphrase-multilingual-MiniLM-L12-v2`` tokenizer on this repository's
    mixed Russian/English corpus (README, docs, git log, result.md, 2000-char
    slices): this estimate came out at 1.02-1.09x the real token count, i.e.
    slightly conservative, which is the safe direction — windows end up a little
    shorter than the encoder could take, never longer.
    """

    #: Characters per token inside a run of ASCII letters, digits, underscores.
    latin_chars_per_token = 4
    #: Characters per token in every other script (Cyrillic averages ~3).
    other_chars_per_token = 3

    _UNIT_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

    @property
    def name(self) -> str:
        return "heuristic"

    def token_spans(self, text: str) -> list[TokenSpan]:
        spans: list[TokenSpan] = []
        for match in self._UNIT_RE.finditer(text):
            unit = match.group()
            start = match.start()
            if not (unit[0].isalnum() or unit[0] == "_"):
                spans.append(TokenSpan(start, match.end()))
                continue
            size = (
                self.latin_chars_per_token
                if _is_ascii_word(unit)
                else self.other_chars_per_token
            )
            for offset in range(0, len(unit), size):
                spans.append(
                    TokenSpan(start + offset, start + min(offset + size, len(unit)))
                )
        return spans


class ModelTokenizer:
    """The encoder's own tokenizer, wrapped to yield character spans."""

    __slots__ = ("_tokenizer", "_name")

    def __init__(self, tokenizer: object, model_name: str) -> None:
        self._tokenizer = tokenizer
        self._name = f"model:{model_name}"

    @property
    def name(self) -> str:
        return self._name

    def token_spans(self, text: str) -> list[TokenSpan]:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
        spans: list[TokenSpan] = []
        cursor = 0
        for start, end in encoding.offsets:
            # Normalizers can emit empty or out-of-order offsets; force the
            # stream to stay non-empty and ascending so spans stay usable.
            start = max(int(start), cursor)
            end = int(end)
            if end <= start:
                continue
            spans.append(TokenSpan(start, end))
            cursor = end
        return spans


def get_tokenizer(
    model_name: str = DEFAULT_EMBEDDING_MODEL, *, backend: str | None = None
) -> Tokenizer:
    """Return the tokenizer to count with, caching one instance per resolution.

    Resolution order: an explicit ``backend`` argument, then
    ``LIVING_MEMORY_CHUNK_TOKENIZER``, then — when the configured embedding
    backend is a hash backend, so no real encoder is in play — the heuristic
    tokenizer, and otherwise the model tokenizer with a heuristic fallback.
    """

    resolved = _effective_backend(backend)
    key = (model_name, resolved)
    cached = _TOKENIZER_CACHE.get(key)
    if cached is None:
        cached = _build_tokenizer(model_name, resolved)
        _TOKENIZER_CACHE[key] = cached
    return cached


def reset_tokenizer_cache() -> None:
    """Drop cached tokenizers so environment changes take effect."""

    _TOKENIZER_CACHE.clear()


def count_tokens(text: str, *, tokenizer: Tokenizer | None = None) -> int:
    """Return how many content tokens the encoder would see for ``text``."""

    active = tokenizer if tokenizer is not None else get_tokenizer()
    return len(active.token_spans(text))


def chunk_text(
    text: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    special_tokens: int = ENCODER_SPECIAL_TOKENS,
    tokenizer: Tokenizer | None = None,
) -> list[TextChunk]:
    """Split ``text`` into overlapping, boundary-aware embedding windows.

    ``max_tokens`` is the encoder's whole sequence budget and ``special_tokens``
    is the part of it the encoder spends on ``<s>``/``</s>``, so each chunk holds
    at most ``max_tokens - special_tokens`` content tokens. Consecutive chunks
    overlap by at most ``overlap_tokens`` tokens.

    Cuts prefer a paragraph break, then a sentence break, and fall back to a hard
    token cut only when no such boundary fills at least half the window — which
    is what happens inside a single sentence longer than one window. A hard cut
    still lands on a word start when one is within a few tokens. Every chunk
    except the last one therefore holds between half a window and a full window
    of tokens. A text that fits in one window comes back as exactly one chunk
    whose ``text`` is the entire input, trailing whitespace included.

    A text with no tokens at all (empty or whitespace-only) yields no chunks:
    there is nothing to embed. Chunk text carries no trailing whitespace, so with
    ``overlap_tokens == 0`` the whitespace at a seam belongs to no chunk; the
    token stream itself is always covered completely.

    Raises:
        ValueError: on arguments that would produce empty windows or a
            zero-advance loop (``overlap_tokens`` at or above the content
            window).
    """

    window = _validated_window(max_tokens, overlap_tokens, special_tokens)
    active = tokenizer if tokenizer is not None else get_tokenizer()
    tokens = active.token_spans(text)
    if not tokens:
        return []

    breaks = _break_indices(text, tokens)
    spans = _pack(len(tokens), breaks, window, overlap_tokens)

    chunks: list[TextChunk] = []
    for index, (token_start, token_end) in enumerate(spans):
        char_start = 0 if token_start == 0 else tokens[token_start].start
        char_end = len(text) if token_end == len(tokens) else tokens[token_end - 1].end
        chunks.append(
            TextChunk(
                text=text[char_start:char_end],
                chunk_index=index,
                token_start=token_start,
                token_end=token_end,
                char_start=char_start,
                char_end=char_end,
            )
        )
    return chunks


@dataclass(frozen=True, slots=True)
class _Breaks:
    """Token indices at which a new structural unit starts.

    ``sentence`` includes every ``paragraph`` index, and ``word`` includes every
    token that opens a word, so the three form a strongest-to-weakest ladder the
    cut and overlap logic can walk down.
    """

    paragraph: tuple[int, ...]
    sentence: tuple[int, ...]
    word: tuple[int, ...]


def _validated_window(max_tokens: int, overlap_tokens: int, special_tokens: int) -> int:
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
    if special_tokens < 0:
        raise ValueError(f"special_tokens must be >= 0, got {special_tokens}")
    window = max_tokens - special_tokens
    if window < 1:
        raise ValueError(
            "max_tokens must exceed special_tokens to leave a content window, "
            f"got max_tokens={max_tokens} special_tokens={special_tokens}"
        )
    if overlap_tokens < 0:
        raise ValueError(f"overlap_tokens must be >= 0, got {overlap_tokens}")
    if overlap_tokens >= window:
        raise ValueError(
            "overlap_tokens must be smaller than the content window "
            f"(max_tokens - special_tokens = {window}) or windows would never "
            f"advance, got overlap_tokens={overlap_tokens}"
        )
    return window


def _break_indices(text: str, tokens: Sequence[TokenSpan]) -> _Breaks:
    starts = [span.start for span in tokens]
    limit = len(tokens)

    def token_index(char_offset: int) -> int:
        return bisect_left(starts, char_offset)

    def collect(pattern: re.Pattern[str]) -> set[int]:
        found: set[int] = set()
        for match in pattern.finditer(text):
            index = token_index(match.end())
            if 0 < index < limit:
                found.add(index)
        return found

    paragraph = collect(_PARAGRAPH_BREAK_RE)
    sentence = collect(_SENTENCE_BREAK_RE) | paragraph
    word = tuple(
        index
        for index in range(1, limit)
        if text[starts[index] - 1 : starts[index]].isspace()
    )
    return _Breaks(
        paragraph=tuple(sorted(paragraph)),
        sentence=tuple(sorted(sentence)),
        word=word,
    )


def _pack(
    token_count: int, breaks: _Breaks, window: int, overlap_tokens: int
) -> list[tuple[int, int]]:
    min_fill = max(1, int(window * _MIN_FILL_RATIO))
    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < token_count:
        limit = cursor + window
        if limit >= token_count:
            spans.append((cursor, token_count))
            break
        cut = _choose_cut(cursor, limit, min_fill, breaks)
        spans.append((cursor, cut))
        advanced = _next_cursor(cursor, cut, overlap_tokens, breaks)
        if advanced <= cursor or len(spans) > token_count:
            # Unreachable while the arguments are validated; a loud failure beats
            # a hang or an unbounded chunk list if that ever stops holding.
            raise RuntimeError(
                "chunking made no progress: "
                f"cursor={cursor} cut={cut} next={advanced} window={window} "
                f"overlap={overlap_tokens}"
            )
        cursor = advanced
    return spans


def _choose_cut(cursor: int, limit: int, min_fill: int, breaks: _Breaks) -> int:
    lowest = cursor + min_fill
    if lowest <= limit:
        for candidates in (breaks.paragraph, breaks.sentence):
            cut = _largest_in_range(candidates, lowest, limit)
            if cut is not None:
                return cut

    # Hard cut inside one long sentence. Pull back to the nearest word start
    # within a small slack so the window does not end mid-word: a window whose
    # both edges are word-aligned re-tokenizes to exactly the tokens it was
    # measured with, which is what keeps it inside max_seq_length.
    slack = max(1, min(_HARD_CUT_WORD_SLACK, (limit - cursor) // 4))
    snapped = _largest_in_range(breaks.word, max(limit - slack, lowest), limit)
    if snapped is not None:
        return snapped
    return limit


def _next_cursor(cursor: int, cut: int, overlap_tokens: int, breaks: _Breaks) -> int:
    if overlap_tokens <= 0:
        return cut
    target = max(cut - overlap_tokens, cursor + 1)
    # Start the next window on a sentence boundary when one sits in the front half
    # of the overlap, else at least on a word start, so the overlap stays close to
    # what was asked for and no window opens mid-word.
    upper = cut - max(1, overlap_tokens // 2)
    if upper >= target:
        for candidates in (breaks.sentence, breaks.word):
            snapped = _smallest_in_range(candidates, target, upper)
            if snapped is not None and snapped > cursor:
                return snapped
    return target


def _largest_in_range(values: Sequence[int], low: int, high: int) -> int | None:
    index = bisect_right(values, high) - 1
    if index >= 0 and values[index] >= low:
        return values[index]
    return None


def _smallest_in_range(values: Sequence[int], low: int, high: int) -> int | None:
    index = bisect_left(values, low)
    if index < len(values) and values[index] <= high:
        return values[index]
    return None


def _is_ascii_word(unit: str) -> bool:
    return all(
        ("a" <= char <= "z") or ("A" <= char <= "Z") or char.isdigit() or char == "_"
        for char in unit
    )


def _effective_backend(backend: str | None) -> str:
    configured = (backend or os.environ.get(TOKENIZER_ENV_VAR) or "").strip().lower()
    if configured:
        return configured
    embedding_backend = (
        os.environ.get("LIVING_MEMORY_EMBEDDING_BACKEND") or ""
    ).strip().lower()
    if embedding_backend in _EMBEDDING_HASH_BACKENDS:
        return "heuristic"
    return "auto"


def _build_tokenizer(model_name: str, resolved: str) -> Tokenizer:
    if resolved in _HEURISTIC_BACKENDS:
        return HeuristicTokenizer()
    tokenizer = _load_model_tokenizer(model_name)
    if tokenizer is not None:
        return tokenizer
    if resolved in _MODEL_BACKENDS:
        _warn_once(
            f"model:{model_name}",
            f"{TOKENIZER_ENV_VAR}={resolved} asked for the model tokenizer of "
            f"{model_name!r}, which is not available locally",
        )
    return HeuristicTokenizer()


def _load_model_tokenizer(model_name: str) -> Tokenizer | None:
    source = _resolve_local_model_source(model_name)
    if source is None:
        return None
    tokenizer_file = _find_tokenizer_file(Path(source))
    if tokenizer_file is None:
        return None

    try:
        from tokenizers import Tokenizer as HuggingFaceTokenizer
    except ImportError:
        _warn_once("tokenizers-missing", "the 'tokenizers' package is not installed")
        return None

    try:
        backing = HuggingFaceTokenizer.from_file(str(tokenizer_file))
    except Exception as exc:  # pragma: no cover - depends on the local snapshot
        _warn_once(f"load:{tokenizer_file}", f"could not load {tokenizer_file}: {exc}")
        return None

    # This model's tokenizer.json ships truncation (max_length 128) and padding
    # baked in. Left on, every text longer than the window would report exactly
    # 128 tokens, the chunker would see it as fitting in one window, and the tail
    # would stay invisible — the very bug chunking removes. Disable both.
    for disable in ("no_truncation", "no_padding"):
        method = getattr(backing, disable, None)
        if callable(method):
            method()
    return ModelTokenizer(backing, model_name)


def _find_tokenizer_file(root: Path) -> Path | None:
    if root.is_file():
        return root
    candidates = [root / "tokenizer.json", *sorted(root.glob("*/tokenizer.json"))]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    _LOG.warning("%s; chunking with the heuristic token counter instead", message)
