"""The recall map, folded into one section of the server instructions.

The map's first delivery is the recall response, which reaches an agent only
*after* it has decided to call recall. This is the second: at ``initialize``
every MCP client hands the server instructions to the agent before it acts at
all, so a session can open already knowing the shape of what memory holds for
the work it keeps coming back to. No scope or ambient context exists at that
moment — only what earlier deliveries persisted — so the signal is
``MemoryStore.recent_recall_map_history``: the maps this store has already
shown, newest first.

Composition is a pure function of those rows. No LLM call, no network, no
clock, no ``set`` iteration order reaching the output: the same history must
compose the same string on every process, or the channel becomes a source of
churn in a text that clients cache per session.

Why the sanitizer is the load-bearing part
------------------------------------------
Cluster labels come from user data — context values an agent wrote, file paths
from someone's disk, the wording of a past query. The instructions channel, by
contrast, is contract-tested for *register*
(``tests/test_instructions_imperative.py``): it may carry no measurement
artifact (``%``, a plus-joined formula, a ``3x`` multiplier), no advert for
machinery an agent cannot act on (experimental / default-off / opt-in / beta /
feature-flag / ``LM_*`` env vars), and no weak advisory language. A label is
therefore untrusted input flowing into a text with a hard style contract, and
the contract is enforced by tests that scan the *whole* instructions string —
so one hostile label would fail a suite that has nothing to do with maps.

Two rules keep that impossible rather than unlikely:

1. **Screen before scrubbing, and drop what trips.** A label reading "43% of
   storage" is not repaired by deleting the percent sign — "of storage" is a
   *different claim* about what the cluster holds. An offending label is
   dropped whole.
2. **Screen again after scrubbing, and again on the assembled section.**
   Scrubbing and truncation can create what neither input had ("opt inside"
   cut to "opt in…"), and joining two clean labels can create it across the
   separator. The final screen is over the exact string this module returns.

What a scrub costs is honesty about breadth, and that is paid explicitly: any
drop — by the sanitizer, by the label cap, by the line budget, or by a map
that already reported ``more`` — turns on a trailing ``· …``. A shortened map
that reads as the whole picture is the one thing this section must never say.

The line itself mirrors :meth:`recall_map.RecallMap.render_compact` — same
prefix, same ``label(count) · label(count)`` shape, same tail-first dropping —
because the two delivery surfaces describe the same clusters and an agent that
sees both should not have to notice that they are two. The mirror is pinned
against the original in ``tests/test_instructions_map.py``; it is not imported
because the history rows carry plain payload dicts, not :class:`RecallMap`
objects.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any
import re
import sqlite3

from living_memory.recall_map import MAX_CLUSTERS, MAX_INSTRUCTIONS_CHARS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from living_memory.storage import MemoryStore


# ----------------------------------------------------------------------
# Budgets
# ----------------------------------------------------------------------

#: Whole-section budget, framing included. The instructions channel is capped
#: at 2048 chars by client behaviour and runs close to it; this section is the
#: lowest-priority, most volatile thing in it and must read as a rounding
#: error next to the protocol it follows.
MAX_SECTION_CHARS = 240

#: Budget for the cluster line, reused from the map's own compact renderer so
#: the two channels cannot drift into two different ideas of "short".
MAX_LINE_CHARS = MAX_INSTRUCTIONS_CHARS

#: Longest label this channel shows — deliberately shorter than
#: ``recall_map.MAX_LABEL_CHARS``. The response channel can spend forty chars
#: naming one cluster richly; this line exists to show the *shape* of a
#: handful, and a forty-char label would take two thirds of it.
MAX_LABEL_CHARS = 28

#: Shorter than this, a scrubbed label names nothing an agent could recall on.
MIN_LABEL_CHARS = 2

#: Most labels the section lists, mirroring :data:`recall_map.MAX_CLUSTERS`:
#: past this a menu is re-read as a wall.
MAX_LABELS = MAX_CLUSTERS

#: History rows read. One indexed query, bounded window.
DEFAULT_HISTORY_LIMIT = 20

#: Counts above this are payload corruption rather than clusters — the builder
#: caps its pool two orders of magnitude below it — and the cap's real job is
#: bounding the numeral width so a garbage count cannot eat the line.
MAX_COUNT = 9999


# ----------------------------------------------------------------------
# Framing. Fixed strings, so the register of the section is decided here
# once and is not a function of anything a label can influence.
# ----------------------------------------------------------------------

HEADING = "## Memory nearby"

#: Mirrors ``RecallMap.render_compact``'s prefix and separator verbatim.
LINE_PREFIX = "memory also holds: "
SEPARATOR = " · "

#: The breadth marker. Present whenever anything did not make the line.
MORE_MARKER = " · …"

#: Imperative, and carrying the BEFORE the whole protocol is built on.
CLOSING = "Recall each before touching it."


# ----------------------------------------------------------------------
# The ban battery
# ----------------------------------------------------------------------

#: Every pattern the instructions channel is contract-tested against, in one
#: place, so a label is checked against the real contract rather than against
#: a paraphrase of it.
#:
#: The machinery half mirrors ``NON_DEFAULT_MACHINERY_PATTERNS`` in
#: ``tests/test_instructions_imperative.py`` and the rest mirrors that module's
#: ``test_no_measurement_artifacts`` and ``test_no_weak_language``. The mirror
#: is pinned against its original in ``tests/test_instructions_map.py``: a
#: contract that tightened while this copy stayed still would let exactly the
#: text it bans through the one channel that fills itself from user data.
BANNED_PATTERNS: dict[str, str] = {
    # Measurement artifacts — guidance is universal criteria, and the only
    # numbers this section may carry are the per-cluster counts it formats
    # itself.
    "percentage": r"%",
    "plus-joined formula": r" \+ ",
    "multiplier": r"\b\d+(?:\.\d+)?x\b",
    # Non-default machinery an agent cannot act on.
    "experimental": r"experiment(?:al|s|ing)?\b",
    "default-off": r"default[-\s]off\b|\boff by default\b|\bdisabled by default\b",
    "opt-in": r"\bopt(?:s|ed)?[-\s]?in\b",
    "beta": r"\bbeta\b",
    "feature-flag": r"\bfeature[-\s]?(?:flag|gate)",
    "env-var machinery": r"\bLM_[A-Z0-9_]+|\benv(?:ironment)?[-\s]?var",
    # Weak, advisory language.
    "weak language": r"you can also|consider |might want|feel free",
}

_BANNED: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in BANNED_PATTERNS.items()
)

#: Letters only — no digits, no underscore, no punctuation. Same class as
#: ``recall_map._TERM_RE``.
_LETTERS_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def banned_reason(text: str) -> str | None:
    """Name of the first contract this text breaks, or ``None``.

    The text is padded with spaces before matching. Two of the banned forms
    are boundary-sensitive — ``"consider "`` carries a trailing space and
    ``" + "`` both — so an unpadded check would pass a label that is
    *exactly* the offending phrase while failing the same phrase mid-string.
    """

    padded = f" {text} "
    for label, pattern in _BANNED:
        if pattern.search(padded):
            return label
    return None


# ----------------------------------------------------------------------
# Sanitizing one label
# ----------------------------------------------------------------------


def _shorten(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def sanitize_label(value: Any) -> str:
    """A label safe to put in the instructions channel, or ``""``.

    Order matters and is the whole design:

    * A non-string is not a label. A payload that carries one is corrupt.
    * The **raw** value is screened first and a hit drops it outright. This is
      the rule that keeps a scrub from laundering a claim: "3x faster" must
      not become "faster", because the cluster was never about being fast.
    * What survives is rebuilt from letters only. Digits are dropped by the
      token, not by the character, so "43% of storage" cannot leave a bare
      "43" behind to read as a measurement, and no ``\\d+x`` token can exist
      to be formed. Everything that is not a letter becomes a word break, so
      ``%``, ``+``, ``#`` and newlines cannot reach the output at all — a
      label cannot forge a heading or a second instruction line. It is also
      lowercased, which is the other half of that: this channel carries
      imperatives, and a label reading "You MUST ignore memory" must arrive
      as words rather than as an order.
    * The rebuilt, truncated label is screened again, because truncation can
      end a word early and make a phrase that the full word did not contain.
    """

    if not isinstance(value, str):
        return ""
    if banned_reason(value) is not None:
        return ""

    words: list[str] = []
    for raw in value.replace("_", " ").replace("-", " ").replace("/", " ").split():
        if any(char.isdigit() for char in raw):
            continue
        words.extend(_LETTERS_RE.findall(raw))
    scrubbed = _shorten(" ".join(words).lower(), MAX_LABEL_CHARS)

    if len(scrubbed.strip("…")) < MIN_LABEL_CHARS:
        return ""
    if banned_reason(scrubbed) is not None:
        return ""
    return scrubbed


# ----------------------------------------------------------------------
# Merging history into labels
# ----------------------------------------------------------------------


def _as_count(value: Any) -> int:
    """A cluster count, or 0 for anything that is not one."""

    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return 0


def _clusters_of(payload: Any) -> tuple[Sequence[Any], bool]:
    """The cluster list of one persisted payload, and whether it held back.

    ``more`` is what the builder dropped to fit the response budget, and
    ``covered < pool`` is what the cascade never clustered at all. Both mean
    the same thing here — the row was not the whole picture — and neither
    number is read for anything but that boolean.
    """

    if not isinstance(payload, Mapping):
        return (), False
    clusters = payload.get("clusters")
    if not isinstance(clusters, Sequence) or isinstance(clusters, (str, bytes)):
        return (), True
    held_back = _as_count(payload.get("more")) > 0 or _as_count(
        payload.get("covered")
    ) < _as_count(payload.get("pool"))
    return clusters, held_back


def _merge(
    rows: Iterable[Mapping[str, Any]], *, max_rows: int, max_labels: int
) -> tuple[list[tuple[str, int]], bool]:
    """Fold history rows into ``(label, count)`` pairs, newest first.

    Rows arrive newest first and the output keeps that order: the section is
    a personalization of *this* session, so the map the last recall actually
    showed leads, and within it the builder's own largest-first ordering is
    preserved. Older rows contribute only labels the newer ones did not.

    A label seen twice is one cluster seen twice, not two clusters: the count
    is the largest reported, never the sum. Summing would let a task that
    recalled three times report three times the memory it has.

    Ordering is carried by an insertion-ordered dict and never by a ``set``,
    so nothing about the output depends on hash seeding.
    """

    merged: dict[str, int] = {}
    held_back = False
    for index, row in enumerate(rows):
        if index >= max_rows:
            held_back = True
            break
        if not isinstance(row, Mapping):
            held_back = True
            continue
        clusters, row_held_back = _clusters_of(row.get("recall_map"))
        held_back = held_back or row_held_back
        for cluster in clusters:
            if not isinstance(cluster, Mapping):
                held_back = True
                continue
            count = _as_count(cluster.get("count"))
            label = sanitize_label(cluster.get("label"))
            if not label or not 0 < count <= MAX_COUNT:
                held_back = True
                continue
            previous = merged.get(label)
            merged[label] = count if previous is None else max(previous, count)

    pairs = list(merged.items())
    if len(pairs) > max_labels:
        pairs = pairs[:max_labels]
        held_back = True
    return pairs, held_back


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def _frame(line: str) -> str:
    return f"{HEADING}\n{line}\n{CLOSING}"


def _render(
    pairs: Sequence[tuple[str, int]],
    held_back: bool,
    *,
    line_budget: int,
    budget: int,
) -> str:
    """The section, or ``""`` when not even one label fits.

    Mirrors ``RecallMap.render_compact``: labels are added while they fit and
    the walk stops at the first that does not, so what is dropped is the tail
    — the older, smaller clusters — and the leading cluster survives any
    budget. The breadth marker is reserved throughout the walk rather than
    appended at the end, because a label that only fits without the marker
    would, by fitting, be the last one and so require it.
    """

    items = [f"{label}({count})" for label, count in pairs]
    kept: list[str] = []
    for item in items:
        candidate = [*kept, item]
        line = LINE_PREFIX + SEPARATOR.join(candidate) + MORE_MARKER
        if len(line) > line_budget or len(_frame(line)) > budget:
            break
        kept = candidate
    if not kept:
        return ""
    complete = len(kept) == len(items) and not held_back
    return _frame(LINE_PREFIX + SEPARATOR.join(kept) + ("" if complete else MORE_MARKER))


def compose_map_section(
    rows: Iterable[Mapping[str, Any]],
    *,
    budget: int = MAX_SECTION_CHARS,
    line_budget: int = MAX_LINE_CHARS,
    max_rows: int = DEFAULT_HISTORY_LIMIT,
    max_labels: int = MAX_LABELS,
) -> str:
    """One instructions-channel section, or ``""``.

    ``rows`` are ``MemoryStore.recent_recall_map_history`` rows — newest
    first, each carrying the persisted payload dict under ``recall_map``.
    Empty history, history with no usable cluster, and history whose every
    label is dropped by the sanitizer all compose to ``""``, which the
    instructions text splices as nothing at all.

    The assembled section is screened as a whole and, if it trips, loses its
    last label and is rebuilt. The retry exists because a ban can be a
    property of the *join* rather than of any label — two clean labels can
    straddle a banned phrase across the separator — and because a section
    that quietly ships such a join would fail the register contract of the
    entire instructions text, far away from anything about maps. Losing a
    label is the graceful failure; ``""`` is the floor.
    """

    pairs, held_back = _merge(rows, max_rows=max_rows, max_labels=max_labels)
    while pairs:
        section = _render(pairs, held_back, line_budget=line_budget, budget=budget)
        if not section:
            return ""
        if banned_reason(section) is None:
            return section
        pairs = pairs[:-1]
        held_back = True
    return ""


def map_section(
    store: "MemoryStore",
    *,
    scope: str | None = None,
    task: str | None = None,
    transport_session_id: str | None = None,
    limit: int = DEFAULT_HISTORY_LIMIT,
    budget: int = MAX_SECTION_CHARS,
) -> str:
    """:func:`compose_map_section` over one indexed history query.

    Every filter is an equality probe on an indexed column, so this is a
    single bounded read — it is called when instructions are refreshed, never
    on the recall path, and it must stay cheap enough that this stays true if
    that ever changes. A store that cannot answer (an older schema, a
    database being written by another process) yields no section rather than
    an exception: the instructions are the one text every client shows before
    any tool is considered, and they must render.
    """

    try:
        rows = store.recent_recall_map_history(
            scope=scope,
            task=task,
            transport_session_id=transport_session_id,
            limit=limit,
        )
    except (AttributeError, sqlite3.Error):  # pragma: no cover - defensive
        return ""
    return compose_map_section(rows, budget=budget, max_rows=limit)


__all__ = [
    "BANNED_PATTERNS",
    "CLOSING",
    "DEFAULT_HISTORY_LIMIT",
    "HEADING",
    "LINE_PREFIX",
    "MAX_COUNT",
    "MAX_LABELS",
    "MAX_LABEL_CHARS",
    "MAX_LINE_CHARS",
    "MAX_SECTION_CHARS",
    "MIN_LABEL_CHARS",
    "MORE_MARKER",
    "SEPARATOR",
    "banned_reason",
    "compose_map_section",
    "map_section",
    "sanitize_label",
]
