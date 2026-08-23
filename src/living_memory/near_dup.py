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
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Protocol

try:
    import numpy as _np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - numpy is a normal runtime dep
    _np = None  # type: ignore[assignment]


__all__ = [
    "ChunkVectorSource",
    "DuplicateCandidate",
    "build_duplicate_map",
    "cosine",
    "mean_pooled_vectors",
]


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
    receive -- ``len(node.content)`` -- and it is the only thing besides the
    vector the map needs: the length guard is the difference between "the same
    fact said twice" and "the same fact plus a new detail".
    """

    node_id: str
    content_length: int


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
    items: Iterable[DuplicateCandidate | tuple[str, int]],
    vectors: Mapping[str, Sequence[float]],
    *,
    cosine_threshold: float,
    min_length_ratio: float,
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
    """

    if cosine_threshold <= 0.0:
        return {}
    length_ratio = max(0.0, float(min_length_ratio))
    threshold = float(cosine_threshold)

    duplicate_of: dict[str, str] = {}
    length_by_id: dict[str, int] = {}
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


def _as_candidate(item: DuplicateCandidate | tuple[str, int]) -> DuplicateCandidate:
    """Accept a :class:`DuplicateCandidate` or a plain ``(node_id, length)`` pair.

    The pair form is there so a caller holding recall results or nodes can
    write ``(result.node.id, len(result.node.content))`` inline instead of
    importing a dataclass to say the same two things.
    """

    if isinstance(item, DuplicateCandidate):
        return item
    node_id, content_length = item
    return DuplicateCandidate(node_id=str(node_id), content_length=int(content_length))


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
