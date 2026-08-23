"""Node-vs-node near-duplicate math: mean-pooled vectors and a duplicate map.

Byte-exact dedup does not find semantic repeats. Measured on the live corpus
(12,874 active traces, 384-d, 2026-08-23) there were **4** byte-identical
active nodes, while 550 nodes (4.3%) had a neighbour at cosine >= 0.95 and 203
at >= 0.99. The repeats that fill an agent's recall slots live at the paraphrase
level, which only a vector comparison can see.

This module is the shared arithmetic for the two places that need that
comparison -- the recall delivery path, which turns a repeat into a stub, and
the recall-path drain, which turns a verbatim repeat into a supersedes edge. It
therefore imports *nothing* from ``delivery``, ``retrieval`` or ``server``: it
is the leaf both of them stand on, and a dependency in the other direction
would make the pair unmergeable independently.

Mean-pool, not max-pool
-----------------------
``retrieval.py`` max-pools a node's chunk cosines on purpose: it asks "does any
window of this node answer the *query*", and one strong window is a real
answer. Here the question is different -- "are these two *nodes* the same
fact" -- and max-pool answers it wrong, because two long, mostly-different
notes that happen to share one boilerplate window would max-pool to ~1.0. The
mean is the node's own direction, so a shared aside cannot carry a pair over
the threshold on its own.

Vector width
------------
A node whose chunk rows disagree on ``dimensions`` is dropped rather than
pooled. Widths do not mix: as ``retrieval._ChunkBlock`` puts it, vectors of
different widths are points in different spaces. A database caught mid-re-embed
holds both, and averaging across them would mint a vector that means nothing
while looking perfectly usable. Dropping the node costs one missed collapse;
pooling across widths costs a wrong one.

Absent, never zero
------------------
A node with no chunk rows is absent from :func:`mean_pooled_vectors`, and a
node absent from the map is never collapsed and never becomes a bearer by
similarity. The alternative -- a zero vector -- is worse than useless: zero
vectors are exactly equal to each other, so every unvectorized node would
collapse into every other one.

The identifier veto
-------------------
Two texts that differ in an identifier are different facts at *any* cosine, and
no threshold separates them from the honest repeats. Measured: on the alt
corpus «узел layer-fauna СДЕЛАН» and «узел layer-actors СДЕЛАН» score 0.9547,
and locally 33 collapsed slots joined different facts spread across 0.93-0.98 --
inside and above the band the shipped 0.95 admits. The template dominates the
vector; the one token that carries the fact is a rounding error in it. So the
map *vetoes* such a pair rather than re-ranking it: a candidate carrying an
identifier its resolved bearer's text does not carry is never collapsed.

"Does not carry" means the token is absent from the bearer's own extracted
tokens -- exact string equality, never substring containment. That distinction
is not academic, it was measured. Goal-tree node names in this corpus are built
by *suffixing*, so ``…/checkpoint-selected-profile-v2`` is literally a substring
of ``…/checkpoint-selected-profile-v2-repaired``. Under containment the shorter
node's whole path "occurs in" the longer node's text, the identifier diff comes
back empty, and two distinct tree nodes -- each with its own recorded OUTCOME
fail -- collapse as one fact. The drain simulation on the pre-hygiene backup
caught exactly that at cosine 0.99350 (``01KTRR6WHFXFW16M691Q1N98E1`` ->
``01KTRJCB5FQ7ZT2WKD02TQ20B3``, project:x) and again at 0.99064, where
``…/stock-contract-preservation-audit`` went under ``…-audit-reintegrate``. The
token rule closes a numeric class with them: ``0.99`` is a substring of
``0.99350`` but not a token of it.

It is not softened into "containment aligned on ``/``" -- allowing a path *tail*
to count -- because that is the same bug one level down: ``a/x.py`` and
``b/x.py`` defeat it. The price of the strict rule is the case containment was
written for: a bearer spelling ``src/living_memory/near_dup.py`` no longer
covers a candidate spelling the bare ``near_dup.py``, and that pair now vetoes.

The veto is one-directional on purpose. The *bearer's* own extra identifiers
still reach the agent -- it keeps its full text -- and it is the candidate's
that would disappear behind a stub. And it is deliberately trigger-happy: a
false veto costs one uncollapsed stub, a false pass costs a hidden fact, so
anything ambiguous is an identifier (see :func:`extract_identifiers`).

``LM_NEAR_DUP_IDENTIFIER_VETO`` is the valve, read by
:func:`identifier_veto_enabled` and honoured identically by both consumers.
Default on: doubt resolves against collapsing. ``0``/``off``/``false``/``no``
restores the previous map byte-for-byte -- the veto branch is not entered at
all, so no text is even read.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import os
import re
from typing import Any, Protocol

try:
    import numpy as _np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - numpy is a normal runtime dep
    _np = None  # type: ignore[assignment]


__all__ = [
    "IDENTIFIER_VETO_ENV",
    "ChunkVectorSource",
    "DuplicateCandidate",
    "build_duplicate_map",
    "cosine",
    "extract_identifiers",
    "identifier_veto_enabled",
    "identifiers_absent_from",
    "mean_pooled_vectors",
]

#: The one valve for the identifier veto, honoured identically by every
#: consumer -- delivery collapse and drain supersedes -- because they all read
#: it through :func:`identifier_veto_enabled`. Default on.
IDENTIFIER_VETO_ENV = "LM_NEAR_DUP_IDENTIFIER_VETO"

#: Everything else -- unset, ``1``, ``on``, or a typo -- leaves the veto on.
#: An unreadable value must not silently disable a guard whose whole purpose is
#: to fire when the evidence is unclear.
_VETO_OFF_FLAGS = frozenset({"0", "false", "no", "off"})

#: A maximal run of identifier-shaped characters. ``\w`` is Unicode, so
#: Cyrillic slugs are tokens too; ``/`` and ``~`` may also *start* one, which is
#: what keeps ``/home/sfx`` and ``~/.local/share`` whole. Quotes, brackets,
#: commas and whitespace are absent from the class and so are the boundaries.
_IDENTIFIER_TOKEN_RE = re.compile(r"[\w~/][\w./:~-]*", re.UNICODE)
#: Trailing punctuation ends a sentence, not a name: ``near_dup.py.`` and
#: ``project:`` lose their tail. Leading ``/`` and ``~`` are kept -- they are
#: the path -- while a leading ``.``/``:``/``-`` is trimmed like the tail.
_TOKEN_TRAILING_TRIM = "./:-~"
_TOKEN_LEADING_TRIM = ".:-"
#: One character is never a name; two can be (``v2``, ``id``-shaped slugs).
_MIN_IDENTIFIER_CHARS = 2
#: The length below which a token needs actual structure -- a separator, a dot,
#: a ULID or digest shape -- rather than merely a digit or a case transition.
#: Four is where a bare number stops reading as prose: ``2026`` is a year,
#: ``3`` in "measured at 3 units" is arithmetic.
_MIN_WEAK_IDENTIFIER_CHARS = 4
#: Any one of these makes a token an identifier outright. ``-`` is the class
#: the goal turns on (``layer-fauna``); ``_`` never occurs in prose; ``/`` is a
#: path; ``:`` is a scope or a URL.
_SLUG_SEPARATORS = ("-", "_", "/", ":")
#: 26 characters of Crockford base32 (no I, L, O or U): a ULID.
_ULID_RE = re.compile(r"[0-9A-HJKMNP-TV-Z]{26}")
#: Seven hex characters is a short git id; digests are longer. The digit
#: requirement lives at the call site, so hex-shaped words never match.
_HEX_DIGEST_RE = re.compile(r"[0-9a-fA-F]{7,}")


class ChunkVectorSource(Protocol):
    """The one thing :func:`mean_pooled_vectors` needs from a store.

    ``MemoryStore.list_node_chunks`` satisfies it. Stated structurally so this
    module does not have to know about connections, migrations or scopes --
    and so a test can hand it rows without a database.
    """

    def list_node_chunks(self, node_id: str) -> Sequence[Any]:
        """Return one node's chunk rows, each with ``dimensions`` and ``embedding``."""


@dataclass(frozen=True, slots=True)
class DuplicateCandidate:
    """One node offered to :func:`build_duplicate_map`, in rank order.

    ``content_length`` is the length of the text the agent would otherwise
    receive -- ``len(node.content)`` -- and it is what the length guard reads:
    the difference between "the same fact said twice" and "the same fact plus a
    new detail".

    ``content`` is that text itself, and the identifier veto is the only thing
    that reads it. It is optional so the pre-veto ``(node_id, length)`` form
    still constructs, but optional is not free: with the veto on, a candidate
    whose text the caller did not supply is NOT collapsed. A veto that cannot
    see the text cannot be enforced, and an unenforceable guard must fail
    closed rather than quietly pass everything.
    """

    node_id: str
    content_length: int
    content: str | None = None


def mean_pooled_vectors(
    store: ChunkVectorSource,
    node_ids: Iterable[str],
) -> dict[str, list[float]]:
    """Return one L2-normalized mean-pooled vector per node, from its chunks.

    Reads ``node_chunk_embeddings`` through ``list_node_chunks`` -- there is no
    ``nodes.embedding`` column to read instead -- and averages the node's chunk
    *directions*. Nodes are compared to each other here, so the pooling is the
    mean and not ``retrieval.py``'s deliberate max; see the module docstring.

    A node is ABSENT from the result, rather than present with a zero vector,
    when it has no chunk rows, when its rows disagree on ``dimensions``, or
    when its chunks cancel out to no direction at all. Absent means "never
    collapsed" downstream, which is the safe default; a zero vector would mean
    "identical to every other unvectorized node", which is the unsafe one.

    Ids are de-duplicated and each node is read once; the result carries only
    the ids that produced a usable vector, so ``len(result)`` may be smaller
    than ``len(node_ids)`` and the caller must treat a missing id as "unknown",
    not as "dissimilar".
    """

    pooled: dict[str, list[float]] = {}
    for node_id in dict.fromkeys(str(node_id) for node_id in node_ids):
        chunks = store.list_node_chunks(node_id)
        if not chunks:
            continue
        widths = {int(chunk.dimensions) for chunk in chunks}
        if len(widths) != 1:
            # Mid-re-embed, or a fixture: different spaces, not one node.
            continue
        width = widths.pop()
        if width <= 0:
            continue
        vectors = [list(chunk.embedding) for chunk in chunks]
        if any(len(vector) != width for vector in vectors):
            # The row's own ``dimensions`` is the authority on its shape; a
            # decoded vector that disagrees with it is a corrupt row, not a
            # narrower one.
            continue
        vector = _mean_pool(vectors)
        if vector is None:
            continue
        pooled[node_id] = vector
    return pooled


def build_duplicate_map(
    items: Iterable[DuplicateCandidate | tuple[str, int] | tuple[str, int, str | None]],
    vectors: Mapping[str, Sequence[float]],
    *,
    cosine_threshold: float,
    min_length_ratio: float,
    identifier_veto: bool = True,
) -> dict[str, str]:
    """Map duplicate node id -> bearer node id over ``items`` in rank order.

    Pure: reads ``items`` and ``vectors``, mutates neither, and the same inputs
    always produce the same map. Both consumers build the map here and hand the
    finished thing to a renderer, so the renderer stays pure too.

    ``items`` arrive in rank order -- best first -- and rank is what decides
    who bears. A node is a duplicate of the **highest-ranked earlier** node it
    exceeds ``cosine_threshold`` against, not of the most similar one: the
    top-ranked result is what the agent should read, so that is what must keep
    its full text. Bearers are always roots -- a duplicate's bearer is resolved
    through the map before it is recorded -- so no value in the returned map is
    also a key, and a consumer never has to chase a chain to find the node that
    actually carries the text.

    Every recorded pair clears the threshold against the bearer it *names*. When
    the node a candidate matched has itself become a stub, the root must clear
    the bar too; if it does not, the scan moves on rather than recording a
    similarity nobody measured. Cosine is not transitive, and the map is a claim
    about the text the agent is handed, not about the chain that led to it.

    ``cosine_threshold <= 0`` returns an empty map. That is the rollback path:
    the caller's env var set to 0 restores the pre-change, byte-only behaviour
    without a revert.

    The length guard. A candidate whose ``content_length`` exceeds its bearer's
    by more than ``min_length_ratio`` (a fraction: ``0.2`` protects anything
    more than 20% longer) is NEVER collapsed. That is the "same fact plus a new
    detail" case, and the detail has to reach the agent. The guard vetoes the
    collapse outright rather than looking for some other, longer bearer: it
    fires precisely when the top match is *missing* text the candidate has, and
    a lower-ranked node happening to be longer is no evidence that it has that
    same text. A candidate shorter than its bearer is unaffected -- the guard
    is one-directional by design. Negative ratios are floored at 0, since
    "protect candidates that are shorter" is not a thing this guard means.

    A node absent from ``vectors`` is skipped entirely: never collapsed, and
    never a bearer by similarity. It can still be the bearer of nodes that
    matched it byte-exactly elsewhere -- that is a different mechanism -- but
    it takes no part in this one, because there is nothing to compare it with.

    The identifier veto (``identifier_veto``, on by default; the callers pass
    :func:`identifier_veto_enabled`). A candidate carrying an identifier token
    -- a ULID, a path, a node or branch name, a slug, a dotted symbol, a digest
    -- that is not a token of its RESOLVED ROOT bearer's text is not collapsed.
    Token equality, not substring containment: the suffixed node names in this
    corpus made containment collapse distinct tree nodes, which
    :func:`identifiers_absent_from` documents with the measured pairs.
    Those two texts are different facts however close their vectors are; see
    the module docstring for the measurement that made this a veto rather than
    a threshold. Like the length guard it vetoes outright rather than hunting
    for a bearer that happens to quote the token, and for the same reason: it
    fires precisely when the top match is *missing* something the candidate
    has, and a lower-ranked node is no evidence that it holds that thing. And
    like the length guard it is one-directional -- identifiers the bearer has
    and the candidate lacks ship to the agent anyway, inside the bearer's own
    text.

    ``identifier_veto=False`` reproduces the pre-veto map byte-for-byte: the
    branch is not entered, so ``content`` is never read and it makes no
    difference whether the caller supplied any. That is the valve's rollback
    path, and it is why this stays a pure function of its arguments -- the env
    var is read by the callers, once, not in here.
    """

    if cosine_threshold <= 0.0:
        return {}
    length_ratio = max(0.0, float(min_length_ratio))
    threshold = float(cosine_threshold)

    duplicate_of: dict[str, str] = {}
    length_by_id: dict[str, int] = {}
    # Every node's identifier tokens, extracted once, when the node is offered
    # as a candidate -- the same tokens it later answers with as a BEARER, so
    # no text is tokenized twice and the inner scan does no regex work at all.
    # The drain offers ~12,900 arrivals and the scan is O(n^2) over pairs;
    # tokenizing the bearer per pair would put a regex inside that square.
    # A node with no text is absent from this map, which the veto reads as
    # "unenforceable" and refuses.
    identifiers_by_id: dict[str, frozenset[str]] = {}
    vector_by_id: dict[str, Sequence[float]] = {}
    # Earlier vector-bearing nodes, in rank order: the scan below walks this
    # best-first and stops at the first match, which is what makes the bearer
    # the highest-ranked one rather than the most similar one.
    ranked: list[tuple[str, Sequence[float]]] = []

    for item in items:
        candidate = _as_candidate(item)
        node_id = candidate.node_id
        if node_id in length_by_id:
            # The same node twice in one ranking: keep the higher-ranked
            # occurrence and ignore the repeat, so nothing can be its own
            # bearer.
            continue
        vector = vectors.get(node_id)
        if vector is None:
            continue
        length_by_id[node_id] = candidate.content_length
        vector_by_id[node_id] = vector
        # Extracted once per candidate rather than once per pair: the tokens
        # are a property of the candidate's own text, and only the "does the
        # bearer say it too" half varies down the scan. ``None`` means the
        # caller supplied no text, which the veto treats as unenforceable.
        candidate_identifiers: tuple[str, ...] | None = None
        if identifier_veto and candidate.content is not None:
            candidate_identifiers = extract_identifiers(candidate.content)
            # The set every later candidate compares against when this node is
            # their bearer. Same tokens, same extraction, one pass.
            identifiers_by_id[node_id] = frozenset(candidate_identifiers)
        for earlier_id, earlier_vector in ranked:
            if cosine(vector, earlier_vector) <= threshold:
                continue
            bearer_id = _resolve_bearer(duplicate_of, earlier_id)
            if bearer_id != earlier_id and cosine(
                vector, vector_by_id[bearer_id]
            ) <= threshold:
                # The match was against a node that is itself a stub by now, and
                # the root that actually ships is NOT above the bar for this
                # candidate. Cosine is not transitive, so chaining the pair
                # anyway would record a repeat the caller never measured: the
                # agent gets a content_ref pointing at text that does not carry
                # this fact. Measured cost of doing it anyway, over 2,000
                # replayed live recalls (artifacts/near-dup/dup-slot-measurement.md):
                # 8 collapsed slots named a bearer as far down as cosine 0.904 --
                # inside the 0.85-0.95 band where different facts live. Keep
                # scanning: a lower-ranked earlier node may still be a root this
                # candidate genuinely repeats.
                continue
            if _is_materially_longer(
                candidate.content_length, length_by_id[bearer_id], length_ratio
            ):
                break
            if identifier_veto and _identifiers_would_be_lost(
                candidate_identifiers, identifiers_by_id.get(bearer_id)
            ):
                # The candidate names something the text that would ship does
                # not. Whatever the cosine says, those are two facts.
                break
            duplicate_of[node_id] = bearer_id
            break
        ranked.append((node_id, vector))
    return duplicate_of


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two vectors; 0.0 when their widths differ.

    Works with and without numpy and returns the same number either way: both
    paths compute ``dot / sqrt(|left|^2 * |right|^2)`` with the same grouping,
    so they differ only in summation order (~1e-16 on a 384-d vector), far
    below anything a 0.95 threshold resolves.

    Differing widths score 0.0 rather than comparing the shared prefix.
    ``embeddings.cosine_similarity`` zips non-strictly and would silently score
    the overlap; here a width mismatch means two different embedding spaces,
    and the only honest answer about their similarity is "none measurable".
    """

    if left is None or right is None:
        return 0.0
    if len(left) == 0 or len(left) != len(right):
        return 0.0
    if _np is not None:
        left_arr = _np.asarray(left, dtype=_np.float64)
        right_arr = _np.asarray(right, dtype=_np.float64)
        dot = float(left_arr @ right_arr)
        square = float(left_arr @ left_arr) * float(right_arr @ right_arr)
    else:
        dot = 0.0
        left_square = 0.0
        right_square = 0.0
        for left_value, right_value in zip(left, right, strict=True):
            left_float = float(left_value)
            right_float = float(right_value)
            dot += left_float * right_float
            left_square += left_float * left_float
            right_square += right_float * right_float
        square = left_square * right_square
    if square <= 0.0:
        return 0.0
    return dot / math.sqrt(square)


def identifier_veto_enabled() -> bool:
    """Read ``LM_NEAR_DUP_IDENTIFIER_VETO``; on unless explicitly turned off.

    One reader for one env var, imported by both consumers, so "the veto is on"
    cannot mean two different things in the delivery path and the drain. Only
    ``0``/``false``/``no``/``off`` (any case, surrounding blank ignored) turn it
    off; unset and unrecognized both leave it ON, because the guard exists for
    the cases where the evidence is unclear and a mistyped valve is one of them.
    """

    return os.environ.get(IDENTIFIER_VETO_ENV, "").strip().lower() not in _VETO_OFF_FLAGS


def extract_identifiers(text: str) -> tuple[str, ...]:
    """Identifier-shaped tokens in ``text``, de-duplicated, in first-seen order.

    Deliberately broader than the measurement harness's identifier regex, which
    requires a DIGIT in the token (``scripts/recall_dup_slot_measure.py``) and
    therefore cannot see ``layer-fauna`` -- the exact class that caused the
    false collapses this veto exists for. Digits are optional here.

    A token is the maximal run of identifier-shaped characters (word characters
    plus ``. / : - ~``), trimmed of the punctuation that ends a sentence rather
    than a name. It counts as an identifier when any of these holds:

    * it contains ``-``, ``_``, ``/`` or ``:`` -- hyphen and underscore slugs
      (``layer-fauna``, ``near-dup-identifier-veto``, ``build_duplicate_map``),
      file and repo paths (``src/living_memory/near_dup.py``, ``~/.local/share``),
      scoped names (``project:lm``) and URLs. Hyphen slugs are the reason the
      rule cannot be narrower: ``layer-fauna`` and ``layer-actors`` are two
      lowercase words joined by a hyphen and nothing but the hyphen rule
      distinguishes them from prose, so Russian ``что-то`` and English
      ``well-known`` are read as identifiers too. That is the trade the goal
      asks for: a false veto costs one uncollapsed stub, a false pass costs a
      hidden fact;
    * it is dotted with at least one part of two characters or more -- module
      and symbol paths (``living_memory.near_dup``), filenames (``near_dup.py``),
      measured numbers (``0.9547``). The two-character part is what keeps
      ``e.g``, ``i.e`` and ``т.е`` out;
    * it is a 26-character Crockford id (a ULID);
    * it is seven or more hex characters including a digit -- commit ids and
      digests (``19877a4383b2e2d2``). The digit is what keeps hex-shaped English
      words out;
    * it is four characters or more and carries a digit -- years, ticket ids,
      versions, PIDs. Shorter is left alone: ``3`` in "measured at 3 units" is
      prose arithmetic, not a name;
    * it is four characters or more with a lowercase-to-uppercase transition --
      ``MemoryStore``, ``recallMap``.

    What is deliberately NOT an identifier: a bare word, however long
    (``understanding``), an all-caps word (``СДЕЛАН``, ``IMPORTANT`` -- emphasis
    is not a name), and a number of three characters or fewer.
    """

    seen: dict[str, None] = {}
    for match in _IDENTIFIER_TOKEN_RE.finditer(text):
        token = match.group().rstrip(_TOKEN_TRAILING_TRIM).lstrip(_TOKEN_LEADING_TRIM)
        if token and _is_identifier(token):
            seen.setdefault(token, None)
    return tuple(seen)


def identifiers_absent_from(text: str, bearer_text: str) -> tuple[str, ...]:
    """``text``'s identifiers that are not *tokens* of ``bearer_text``.

    Token equality, not substring containment, and case-sensitive. Both sides
    go through :func:`extract_identifiers` and the comparison is string
    equality between extracted tokens: a token counts as present only when the
    bearer names it as a token of its own, never because it happens to sit
    inside a longer one.

    The rule was containment until the drain simulation on the pre-hygiene
    backup showed what containment lets through. Node names here are built by
    suffixing, so ``…/checkpoint-selected-profile-v2`` is a substring of
    ``…/checkpoint-selected-profile-v2-repaired``; with the longer-named node
    bearing, the candidate's own path "occurred in" the bearer's text, the diff
    came back empty, and two distinct tree nodes carrying two distinct recorded
    failures collapsed into one at cosine 0.99350 (project:x,
    ``01KTRR6WHFXFW16M691Q1N98E1`` -> ``01KTRJCB5FQ7ZT2WKD02TQ20B3``). The same
    shape put ``…/stock-contract-preservation-audit`` under
    ``…-audit-reintegrate`` at 0.99064, and a numeric variant nobody had looked
    for came with it: ``0.99`` is a substring of ``0.99350`` but not a token.

    The price is exactly the case containment was written for: a bearer
    spelling ``src/living_memory/near_dup.py`` no longer covers a candidate
    spelling the bare ``near_dup.py``, so that pair vetoes now. Paid on
    purpose -- a false veto costs one uncollapsed stub, a false pass costs a
    hidden fact -- and deliberately not softened into "containment aligned on
    ``/``", which ``a/x.py`` against ``b/x.py`` defeats the same way.

    Case-sensitive because these are paths, env vars and ids, where case is
    meaning, and because the ambiguous direction is the one that vetoes.

    Non-empty return means "do not collapse". Order and de-duplication follow
    :func:`extract_identifiers`, so the result is also readable as a report of
    what a collapse would cost.
    """

    bearer_identifiers = frozenset(extract_identifiers(bearer_text))
    return tuple(
        token
        for token in extract_identifiers(text)
        if token not in bearer_identifiers
    )


def _identifiers_would_be_lost(
    candidate_identifiers: tuple[str, ...] | None,
    bearer_identifiers: frozenset[str] | None,
) -> bool:
    """Whether collapsing would hide an identifier the bearer's tokens lack.

    Takes the bearer's *extracted token set*, not its raw text, so the decision
    is the same string equality :func:`identifiers_absent_from` reports and the
    bearer's text is tokenized once per node rather than once per pair.

    ``None`` on either side is a veto, not a pass: the first means the caller
    supplied no candidate text, the second no bearer text, and in both cases
    the check the valve promises cannot be performed. Silently collapsing
    instead would make the veto unenforceable exactly where a caller forgot to
    wire it -- the one failure mode a guard like this must not have.
    """

    if candidate_identifiers is None or bearer_identifiers is None:
        return True
    return any(token not in bearer_identifiers for token in candidate_identifiers)


def _is_identifier(token: str) -> bool:
    """One trimmed token against the classes documented on ``extract_identifiers``."""

    if len(token) < _MIN_IDENTIFIER_CHARS:
        return False
    if any(separator in token for separator in _SLUG_SEPARATORS):
        return True
    if "." in token and _is_dotted_path(token):
        return True
    if _ULID_RE.fullmatch(token):
        return True
    has_digit = any(character.isdigit() for character in token)
    if has_digit and _HEX_DIGEST_RE.fullmatch(token):
        return True
    if len(token) < _MIN_WEAK_IDENTIFIER_CHARS:
        return False
    return has_digit or _has_case_transition(token)


def _is_dotted_path(token: str) -> bool:
    """A dotted token with two or more parts, one of them two characters or more.

    ``0.95`` and ``near_dup.py`` qualify; ``e.g``, ``i.e``, ``т.е`` and ``1.5``
    do not. The parts rule is the whole guard here: abbreviations are dots
    between single characters, names are not.
    """

    parts = [part for part in token.split(".") if part]
    return len(parts) >= 2 and any(len(part) >= 2 for part in parts)


def _has_case_transition(token: str) -> bool:
    """Whether a lowercase character is immediately followed by an uppercase one."""

    return any(
        left.islower() and right.isupper()
        for left, right in zip(token, token[1:], strict=False)
    )


def _mean_pool(vectors: Sequence[Sequence[float]]) -> list[float] | None:
    """Average the chunks' directions and re-normalize; None if they cancel.

    Each chunk is normalized before it is added, so a chunk contributes a
    direction and not a magnitude. The writer stores unit vectors, but a
    fixture or a half-migrated row need not, and one unnormalized long chunk
    would otherwise drag the node's vector onto itself. A zero chunk has no
    direction and so adds nothing.

    The sum is divided by the chunk count to make the intermediate an actual
    mean; the final normalization cancels that factor, so the returned
    direction is identical either way and the division is there for the
    reader, not the arithmetic.
    """

    if not vectors:
        return None
    if _np is not None:
        matrix = _np.asarray(vectors, dtype=_np.float64)
        norms = _np.sqrt(_np.einsum("ij,ij->i", matrix, matrix))
        # A zero row divided by 1.0 stays zero, which is exactly the "adds
        # nothing" the pure-Python branch gets by skipping it.
        norms[norms == 0.0] = 1.0
        mean = (matrix / norms[:, None]).sum(axis=0) / len(vectors)
        return _unit(mean.tolist())
    total = [0.0] * len(vectors[0])
    for vector in vectors:
        norm = math.sqrt(sum(float(value) * float(value) for value in vector))
        if norm <= 0.0:
            continue
        for index, value in enumerate(vector):
            total[index] += float(value) / norm
    return _unit([value / len(vectors) for value in total])


def _unit(vector: Sequence[float]) -> list[float] | None:
    """L2-normalize, or None for a vector with no direction to normalize."""

    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm <= 0.0:
        return None
    return [float(value) / norm for value in vector]


def _as_candidate(
    item: DuplicateCandidate | tuple[str, int] | tuple[str, int, str | None],
) -> DuplicateCandidate:
    """Accept a :class:`DuplicateCandidate` or a plain ``(node_id, length)`` tuple.

    The tuple form is there so a caller holding recall results or nodes can
    write ``(result.node.id, len(result.node.content))`` inline instead of
    importing a dataclass to say the same two things; a third element supplies
    the text the identifier veto reads. Without it the candidate has no text,
    and with the veto on that means it is not collapsed.
    """

    if isinstance(item, DuplicateCandidate):
        return item
    node_id, content_length, *rest = item
    content = rest[0] if rest and rest[0] is not None else None
    return DuplicateCandidate(
        node_id=str(node_id),
        content_length=int(content_length),
        content=None if content is None else str(content),
    )


def _resolve_bearer(duplicate_of: Mapping[str, str], node_id: str) -> str:
    """Follow ``node_id`` to the node that actually carries the text.

    Values recorded in the map are already roots, so this walks at most one
    step; the loop (and its visited set) is the guarantee that a caller can
    never be handed a bearer that is itself a duplicate, whatever future edits
    do to the recording side.
    """

    bearer = node_id
    visited: set[str] = set()
    while bearer in duplicate_of and bearer not in visited:
        visited.add(bearer)
        bearer = duplicate_of[bearer]
    return bearer


def _is_materially_longer(
    candidate_length: int,
    bearer_length: int,
    length_ratio: float,
) -> bool:
    """Whether the candidate exceeds its bearer's length by more than the ratio.

    An empty bearer is exceeded by any content at all: there is no length to
    take a ratio against, and a bearer carrying nothing cannot be carrying the
    candidate's detail either.
    """

    if candidate_length <= bearer_length:
        return False
    if bearer_length <= 0:
        return True
    return candidate_length > bearer_length * (1.0 + length_ratio)
