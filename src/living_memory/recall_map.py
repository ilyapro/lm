"""Deterministic maps of the recall's residual knowledge.

The map supplements selected results with structural, path, query-anchor and
embedding clusters. Labels are enriched and screened against corpus document
frequencies, then fitted to the response budget. Query qualification is shared with retrieval; map delivery leaves the main
results unchanged.

Structure and delivery cadence belong to one current-work consumer: scope plus
an exact explicit task_pattern or task, otherwise the server's transport id.
Without that identity each call builds its own map and carries no non-use
judgment. Membership, counts and medoids refresh from the current residual;
labels and ask-hints stay stable until the corpus changes.

Three unconsumed offers curtail the same work's repeated noise. Persisted map
history supplies the cadence and survives process restart. Follow evidence is
read from that work's recall events (including mapless recalls) and id lookups
attributed to the current work on their transport. Global node access and
corpus consumption statistics do not identify a current consumer.

A bounded memo avoids history reads before a collapse is possible; it cannot
collapse a map without a fresh read. Missing or truncated evidence permits an
offer. The ordinary finite history window permits retries. The optional decay
cadence compares a persisted pool digest before rebuilding an unchanged offer.
No model call, new history table or background process is needed.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import log, sqrt
from typing import TYPE_CHECKING, Any
import hashlib
import json
import os
import re
import sqlite3

from living_memory.embeddings import cosine_similarity, tokenize
from living_memory.grounding import token_set
from living_memory.models import Node
from living_memory.scope import GLOBAL_SCOPE, normalize_scope
from living_memory.score_gate import QueryRelevance
from living_memory.storage import (
    CHUNK_EMBEDDING_TABLE,
    MemoryStore,
    RecallWork,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from living_memory.retrieval import RecallResult


# ----------------------------------------------------------------------
# Caps. All of them are module constants because two sibling delivery
# surfaces (the recall response and the server instructions) budget against
# them and must not each invent their own number.
# ----------------------------------------------------------------------

#: Most clusters a map may carry. Six is the point past which a "plan" stops
#: being one: the agent has to choose what to ask next, and a list longer than
#: this is re-read as a wall rather than as a menu.
MAX_CLUSTERS = 6

#: Budget for :meth:`RecallMap.to_dict` serialized compactly. The recall
#: response is on a delivery diet; the map is additive and must stay a
#: rounding error next to the results it decorates.
#:
#: The two caps bind in different regimes and it is worth knowing which. One
#: serialized cluster costs 88 characters of JSON keys before a single
#: character of content, so six of them spend 528 of this budget on
#: punctuation: :data:`MAX_CLUSTERS` is the ceiling for short labels, and for
#: anything richer *this* is the binding constraint. Examples narrow to their
#: informative floor, then tail clusters give way down to the breadth floor,
#: then journal names yield before gist does (:meth:`RecallMapBuilder._fit`). A
#: dropped cluster is counted in the payload's ``more`` field rather than
#: vanishing, because a map that silently truncates reads as "this is
#: everything memory holds", which is the one thing it must never say.
MAX_RESPONSE_CHARS = 700

#: Budget for :meth:`RecallMap.render_compact`, the single-string form the
#: server-instructions channel embeds. Instructions are capped at 2048 chars
#: by contract test and currently run within single digits of that ceiling, so
#: the map's whole footprint there is this line.
MAX_INSTRUCTIONS_CHARS = 150

#: Longest medoid example a map carries before the response budget starts
#: taking it apart. See :meth:`RecallMapBuilder._fit`.
MEDOID_EXAMPLE_CHARS = 120

#: Shortest informative gist the response budget should buy for every medoid.
#:
#: This is a floor on what the source text can supply: a ten-character memory
#: contributes all ten characters and satisfies the rule.  On the measured
#: field corpus every medoid was longer than the floor, so the ordinary case
#: is exactly forty or more characters in every delivered row.
MIN_MEDOID_EXAMPLE_CHARS = 40

#: Breadth that gist fitting may not cross.  Two clusters preserved 64 of 67
#: historically followed deliveries; one preserved only 57.  The value is
#: clamped to the number of clusters the configured builder actually kept, so
#: a one-cluster pool (or ``max_clusters=1``) still yields a map.
MIN_CLUSTERS = 2

MAX_LABEL_CHARS = 40
MAX_ASK_HINT_CHARS = 80

#: Only admitted members pay for the full anchor/vector labeling cascade.
MAX_POOL_NODES = 200

#: Compact exclusion vector order frozen in the additive ``sel`` contract.
#: Do not alphabetize it: both persistence and the evaluator interpret counts
#: positionally.
SELECTION_REASON_CODES: tuple[str, ...] = (
    "iv",
    "du",
    "fc",
    "ss",
    "sj",
    "lr",
    "pc",
)
#: Reason codes the two env-gated pool gates add *after* the frozen vector.
#:
#: Appended, never interleaved: a reader that knows the seven above keeps
#: reading them at the same indices, which is the whole content of the
#: positional contract. The other half of keeping that promise is
#: :meth:`_SelectionLedger.freeze`, which trims trailing zeros back to the
#: frozen width — so a server with both valves unset emits exactly the
#: seven-wide ``x`` it emitted before these codes existed, and a longer vector
#: is itself the statement that a gate fired.
POOL_GATE_REASON_CODES: tuple[str, ...] = (
    #: Below the usefulness floor of :data:`POOL_USEFULNESS_FLOOR_ENV`.
    "uf",
    #: Reserved positional code for readers of stored selection journals.
    "nf",
)
SELECTION_LEDGER_REASON_CODES: tuple[str, ...] = (
    *SELECTION_REASON_CODES,
    *POOL_GATE_REASON_CODES,
)
SELECTION_SAMPLE_GIST_CHARS = 24
SELECTION_SAMPLE_LIMIT = 2
#: Both gate codes are sampleable, and that is a decision rather than a
#: default. The three ballast codes are sampleable because an operator reading
#: the journal needs to see *what* a machine-shaped rule swallowed; ``lr`` and
#: ``pc`` because a frozen score and a cap are invisible otherwise. The gate
#: codes are the strongest case of the same argument: their thresholds are the
#: one part of this module an operator sets by hand from field numbers, and a
#: journal that says "the floor dropped 14" without ever naming one of them
#: gives the operator no way to tell a floor that is working from a floor set
#: one decimal place too high. The gist is already identity-free and capped at
#: :data:`SELECTION_SAMPLE_GIST_CHARS`, so admitting them costs the payload
#: nothing it does not already spend on the codes beside them — and costs it
#: only when a valve is on, since with both unset neither code ever fires.
_SAMPLEABLE_REASONS = frozenset({"fc", "ss", "sj", "lr", "pc", "uf", "nf"})

# ----------------------------------------------------------------------
# Pool gates behind env valves
# ----------------------------------------------------------------------
#
# The optional stored-usefulness floor is independent of map consumption.
# Both an enabled valve and an explicit threshold are required.

#: ``1``/``true``/``yes``/``on`` enables a gate. Anything
#: else — unset, empty, ``0``, a typo — is off, because a misread flag here
#: silently removes rows from the one channel that says what memory holds.
_POOL_GATE_ON_FLAGS = frozenset({"1", "true", "yes", "on"})

#: Valve for gate (a): drop a candidate whose ``usefulness_score`` sits below
#: the floor. The score is already on the node the residual carries, so this
#: gate costs one attribute read per candidate and no query at all.
POOL_USEFULNESS_GATE_ENV = "LM_MAP_POOL_USEFULNESS_GATE"
#: The floor itself. No default: see the block comment above.
POOL_USEFULNESS_FLOOR_ENV = "LM_MAP_POOL_MIN_USEFULNESS"

#: Cosine at or above which greedy leader clustering admits a node to an
#: existing cluster. A fallback stage's threshold, deliberately loose: whatever
#: reaches stage 4 has no structural signal at all, and a tight cut would
#: return one singleton per node — a map that says nothing, expensively.
EMBEDDING_CLUSTER_COSINE = 0.55

#: Distinct terms per cluster that reach the document-frequency index. Each is
#: one indexed seek; the top of a cluster's term-frequency list is where the
#: label lives anyway.
CTFIDF_TERM_BUDGET = 40

#: Terms in a c-TF-IDF label.
CTFIDF_LABEL_TERMS = 3

#: Distinct label terms the delivery gate scores. Capping the sum at three is
#: what makes :data:`LABEL_GATE_MIN_IC` mean anything: without it a label buys
#: its way past the floor by concatenating house vocabulary. Measured on the
#: field corpus, the four house words ``public``/``server``/``mmo``/``test``
#: sum to 4.191 across all four and to 3.619 across their best three — the cap
#: is the difference between a floor of 4.0 that admits them and one that does
#: not.
LABEL_GATE_TOP_TERMS = 3

#: Information content, in nats, a label's best :data:`LABEL_GATE_TOP_TERMS`
#: terms must carry for the cluster to be delivered at all.
#:
#: ``ic(t) = log((N + 1) / (1 + df(t)))`` against the same FTS
#: document-frequency index :meth:`RecallMapBuilder._ranked_terms` reads, so
#: "generic" means generic *to the corpus recall searches* and the rule
#: transfers to a corpus whose house vocabulary is different words. That is the
#: whole reason this is a statistic and not a word list: an enumerated
#: blacklist of the labels a field window happened to show would be a lookup
#: table wearing a threshold.
#:
#: Placed in a void rather than fitted to a boundary. Over every label the
#: field ever delivered (294 clusters, 21 distinct labels, ``N = 2573``) the
#: single-content-term population scored 0.57–2.72 and the multi-word one
#: 4.89–15.53; the interval ``(2.72, 4.89)`` is empty and 4.0 sits inside it.
#:
#: The floor encodes the term-count half of the predicate implicitly, which is
#: better than stating it twice: a *solo* term clears 4.0 only when it appears
#: in ``(N + 1) / e**4 - 1`` documents — 1.79 % of the corpus. A one-word label
#: is not banned by rule, it is admitted exactly when one word genuinely is
#: that rare.
LABEL_GATE_MIN_IC = 4.0

#: Tokens an enriched ask-hint may carry on the path and structural stages.
#:
#: Not a style rule — the whole reason label enrichment does not silently
#: retune :data:`CURTAIL_QUERY_OVERLAP`. ``_echoes`` needs ``ceil(n/2)`` shared
#: tokens for a phrasing of ``n``, and ``ceil(1/2) == ceil(2/2) == 1``: a hint
#: widened from one token to two asks a later query for exactly the evidence it
#: asked for before, while a three-token hint would ask for two. Replayed over
#: 2 524 (delivery, later-query) pairs in the field window, holding the hint at
#: two tokens is bit-identical to baseline (688 echo fires, 67 of 77 deliveries
#: judged followed) and letting it follow a three-token label destroys every
#: one of them.
#:
#: The cap governs *enrichment*; it never shortens a hint that was already
#: longer, because dropping a token breaks the same containment guarantee it
#: exists to protect. Stage 3 is exempt outright — an anchor's hint is a
#: question that already worked and is never rewritten.
ASK_HINT_MAX_TOKENS = 2

#: Name triples the filter journal carries. Four, because the counts beside
#: them are the honesty guarantee and the names are the convenience: a reader
#: who needs more than four examples of what was filtered is reading the
#: persisted payloads, not the live map.
FILTER_JOURNAL_NAMES = 4

#: Journal names are shortened harder than delivered labels, for the same
#: reason ``instructions_map`` shortens to 28: a journal name is a forensic
#: breadcrumb, not an invitation, and it competes for the same
#: :data:`MAX_RESPONSE_CHARS` as the medoid examples that are.
FILTER_JOURNAL_LABEL_CHARS = 24

#: Share of the current pool a cached structure must still cover to be reused.
#: Below it the cached labels describe a pool that has moved on, and stability
#: would be preserved at the cost of describing nothing.
CACHE_MIN_COVERAGE = 0.5

#: Distinct ``(scope, task)`` keys kept. One session touches a handful.
MAX_CACHE_ENTRIES = 32

#: Consecutive unconsumed deliveries under one key before the map collapses.
#: Three, because two is inside the noise of a single distracted session and
#: four spends a fourth full map to learn what the third already said. The
#: streak counts *offers* — deliveries that actually carried clusters — so the
#: collapsed markers a curtailed key keeps emitting can never push it up.
CURTAIL_STREAK = 3

#: Deliveries the curtail probe reads back — a *window*, not a history, and
#: the width of it is what sets the retry.
#:
#: A collapse is remembered only for as long as the offers that caused it stay
#: inside this window. The collapsed markers that follow them push them out,
#: and when the last of them goes the key offers full maps again: with
#: :data:`CURTAIL_STREAK` of 3, a key that is never consumed spends 3
#: deliveries of every 25 on a map and the other 22 on a ~60-character marker.
#: That 12% duty cycle is deliberate. A window of exactly the streak would
#: forget its own reason on the very next delivery and never collapse at all;
#: an unbounded one would make the first three unread maps a life sentence,
#: and a task's corpus — and the agent working it — both move.
CURTAIL_HISTORY_LIMIT = 24

#: Share of a cluster's label (or ask-hint) tokens a later query must carry for
#: that cluster to count as asked about. Half: a two-token label needs both, a
#: four-token one needs two, and a query that merely shares the corpus's house
#: vocabulary with a cluster does not clear it.
CURTAIL_QUERY_OVERLAP = 0.5

#: Valve for the decaying curtail retry, drain-valve spelling like the pool
#: gates. Off — unset, empty, ``0``, a typo — is the frozen cadence above,
#: byte for byte. On, a collapsed key retries after
#: ``min(2**(offers-1), CURTAIL_DECAY_SKIP_MAX)`` collapsed deliveries instead
#: of waiting for the window to expire its evidence. Registered in
#: ``~/p/ae/artifacts/injection-throttle/prereg-draft.md`` (П2), off a field
#: number: the quiet armored day on the alt store that first filled the maps
#: also pushed their curtailed share 0.024 → 0.562.
CURTAIL_DECAY_GATE_ENV = "LM_MAP_CURTAIL_DECAY"

#: Ceiling of the decay's skip run, from the registered rule
#: ``min(2**(N-1), 8)``: at the entry threshold of 3 the first run is 4, one
#: more unaccepted retry doubles it to 8, and there it stays. At the plateau a
#: key that is never followed spends 1 delivery in 9 on a map — the same
#: budget as the frozen 3-in-25 — but its first retry lands 4 deliveries
#: after the collapse instead of 22.
CURTAIL_DECAY_SKIP_MAX = 8

#: Hex characters of the pool digest a decay-armed marker persists (``pd``).
#: 64 bits: the comparison is one key's window against itself, and the marker
#: has ~640 of its 700 characters to spare.
CURTAIL_POOL_DIGEST_CHARS = 16

#: Context fields consulted by stage 1, in precedence order.
STRUCTURAL_FIELDS: tuple[str, ...] = (
    "procedure_id",
    "lesson_kind",
    "type",
    "topic",
    "task_pattern",
)

#: Context fields stage 2 reads paths out of.
_PATH_FIELDS: tuple[str, ...] = (
    "files",
    "file",
    "paths",
    "path",
    "files_seen",
    "files_owned",
    "files_read",
    "files_evidence",
)

#: Directory segments that name a *container*, not a subsystem. Stage 2 starts
#: reading after the last one it sees, which is what turns
#: ``/home/u/p/lm/src/living_memory/postsession/x.py`` into ``postsession``
#: rather than into ``u``.
_PATH_ROOT_SEGMENTS = frozenset(
    {"src", "lib", "app", "pkg", "internal", "cmd", "source", "sources", "tests", "test"}
)

STAGE_STRUCTURAL = "structural"
STAGE_PATH = "path"
STAGE_ANCHOR = "anchor"
STAGE_EMBEDDING = "embedding"
#: Precedence, and the secondary sort key for equal-sized clusters: a cluster
#: whose label is a key the corpus actually stores outranks one whose label was
#: derived from term statistics.
_STAGE_ORDER: dict[str, int] = {
    STAGE_STRUCTURAL: 0,
    STAGE_PATH: 1,
    STAGE_ANCHOR: 2,
    STAGE_EMBEDDING: 3,
}

#: Which filter took a cluster off the wire. One character each, because they
#: ride in the payload beside the names they qualify, and the distinction they
#: carry is the one a forensic reader actually needs: ``w`` says the label gate
#: refused to deliver the cluster at all, ``d`` says the response budget could
#: not afford it. "The filter is too harsh" and "the budget is tight" are
#: different findings and must not be read off one number.
FILTER_TAG_WITHHELD = "w"
FILTER_TAG_DROPPED = "d"

_WHITESPACE_RE = re.compile(r"\s+")
_TERM_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_HEX_DIGITS = frozenset("0123456789abcdef")
_VOWELS = frozenset("aeiouyаеёиоуыэюя")
_STRATEGY_STAGNATION_RE = re.compile(
    r"^Strategy stagnation detected on [^\r\n]{1,160}(?:\r?\n"
    r"(?:attempts|strategy|window|reason):[^\r\n]{1,160}){0,4}\s*$"
)
_FILE_CHUNK_INDEX_RE = re.compile(r"^[1-9][0-9]*/[1-9][0-9]*$")
_FILE_CHUNK_LINES_RE = re.compile(r"^[1-9][0-9]*-[1-9][0-9]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JOURNAL_KINDS = frozenset(
    {
        "supervision_journal",
        "monitoring_journal",
        "execution_journal",
        "goal_tree_journal",
    }
)
_JOURNAL_METADATA_KEYS = ("kind", "type", "lesson_kind", "record_kind")


# ----------------------------------------------------------------------
# The map
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MapMedoid:
    """The member that speaks for a cluster, plus a short quote from it."""

    node_id: str
    example: str

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"node_id": self.node_id}
        if self.example:
            payload["example"] = self.example
        return payload


@dataclass(frozen=True, slots=True)
class MapCluster:
    """One row of the map: a label, how much of it there is, and how to ask."""

    label: str
    count: int
    medoid: MapMedoid
    #: How to phrase a recall for this cluster. At stage 3 this is a past
    #: grounded query verbatim — not a guess at good wording, but wording that
    #: already worked once.
    ask_hint: str
    #: Transferable into the agent's own todo plan, which on a linear
    #: ``--print`` pass is its only surface of self-observation.
    plan_item: str
    stage: str
    member_ids: tuple[str, ...]
    def to_dict(self) -> dict[str, Any]:
        payload = {
            "label": self.label,
            "count": self.count,
            "medoid": self.medoid.to_dict(),
            "ask_hint": self.ask_hint,
            "plan_item": self.plan_item,
        }
        return payload


@dataclass(frozen=True, slots=True)
class FilteredCluster:
    """One cluster the map did not deliver, reduced to what a reader can use.

    A name and a size, tagged with which filter took it. Deliberately *not* a
    :class:`MapCluster`: a filtered cluster has no medoid, no ask-hint and no
    plan item, because there is nothing here for the agent to act on — this is
    forensics, and it lives in its own payload key precisely so that no
    consumer can mistake it for something on offer.
    """

    #: :data:`FILTER_TAG_WITHHELD` or :data:`FILTER_TAG_DROPPED`.
    tag: str
    label: str
    count: int

    def to_list(self) -> list[Any]:
        """The wire form: a triple, not an object, to save the JSON keys."""

        return [self.tag, self.label, self.count]


@dataclass(frozen=True, slots=True)
class SelectionSample:
    """One identity-free example of a member-level exclusion.

    ``ordinal`` is retained only while fitting the additive selection block;
    it is never serialized.  It makes the choice independently auditable
    without turning a node id into policy input or wire content.
    """

    reason: str
    gist: str
    ordinal: int

    def to_list(self) -> list[str]:
        return [self.reason, self.gist]


@dataclass(frozen=True, slots=True)
class SelectionAccounting:
    """Complete candidate-selection accounting for one inspected residual.

    The response fitter added by the delivery integration can vary only how
    many of ``samples`` it buys.  ``inspected``, ``admitted`` and ``excluded``
    are unconditional, and ``to_dict`` recomputes ``o`` from the chosen sample
    count so dropping examples can never falsify the accounting equation.

    ``excluded`` is the frozen :data:`SELECTION_REASON_CODES` vector,
    optionally extended by the :data:`POOL_GATE_REASON_CODES` an env valve
    turned on.  The extension is *canonically trimmed*: a trailing zero past
    the frozen width is rejected rather than accepted-and-ignored, so there is
    exactly one encoding of "no gate fired" and it is byte-identical to what a
    build with no gates at all produced.  That is the invariant, not a
    convention — an accounting that could spell the same fact two ways would
    make the byte-identity claim unfalsifiable.
    """

    inspected: int
    admitted: int
    excluded: tuple[int, ...]
    samples: tuple[SelectionSample, ...] = ()
    sampleable: int = 0

    def __post_init__(self) -> None:
        frozen_width = len(SELECTION_REASON_CODES)
        width = len(self.excluded)
        if not frozen_width <= width <= len(SELECTION_LEDGER_REASON_CODES):
            raise ValueError("selection exclusion vector has the wrong width")
        if width > frozen_width and not self.excluded[-1]:
            raise ValueError("selection exclusion vector is not canonically trimmed")
        if self.inspected < 0 or self.admitted < 0 or any(
            count < 0 for count in self.excluded
        ):
            raise ValueError("selection counts must be non-negative")
        if self.inspected != self.admitted + sum(self.excluded):
            raise ValueError("selection accounting does not cover the residual")
        if self.sampleable < len(self.samples):
            raise ValueError("selection samples exceed the sampleable population")
        sampleable_cap = sum(
            count
            for count, reason in zip(
                self.excluded, SELECTION_LEDGER_REASON_CODES, strict=False
            )
            if reason in _SAMPLEABLE_REASONS
        )
        if self.sampleable > sampleable_cap:
            raise ValueError("selection sampleable count exceeds exclusions")

        reason_order = {
            reason: index
            for index, reason in enumerate(SELECTION_LEDGER_REASON_CODES)
        }
        prior = -1
        seen: set[str] = set()
        for sample in self.samples:
            if sample.reason not in _SAMPLEABLE_REASONS:
                raise ValueError("selection sample has a non-sampleable reason")
            current = reason_order[sample.reason]
            if current >= width or self.excluded[current] <= 0:
                raise ValueError("selection sample has no matching exclusion")
            if current <= prior or sample.reason in seen:
                raise ValueError("selection samples are not in fixed reason order")
            if not sample.gist or len(sample.gist) > SELECTION_SAMPLE_GIST_CHARS:
                raise ValueError("selection sample gist is empty or over budget")
            if sample.ordinal < 0:
                raise ValueError("selection sample ordinal must be non-negative")
            prior = current
            seen.add(sample.reason)

    def to_dict(self, *, sample_limit: int = SELECTION_SAMPLE_LIMIT) -> dict[str, Any]:
        """Return the additive ``sel`` shape with a budgeted ``q``."""

        limit = min(SELECTION_SAMPLE_LIMIT, max(0, int(sample_limit)))
        chosen = self.samples[:limit]
        payload: dict[str, Any] = {
            "v": "r1",
            "n": self.inspected,
            "e": self.admitted,
            "x": list(self.excluded),
            "o": self.sampleable - len(chosen),
        }
        if chosen:
            payload["q"] = [sample.to_list() for sample in chosen]
        return payload


@dataclass(frozen=True, slots=True)
class RecallMap:
    """A capped, ordered set of clusters over one recall's residual pool."""

    scope: str
    #: The cache key this map was built or served under.
    key: str
    clusters: tuple[MapCluster, ...]
    #: Members considered, or None when curtailment skipped pool construction.
    pool_size: int | None
    #: Members the clusters actually account for.
    covered: int
    #: Clusters the caps dropped. Reported, never silent.
    dropped: int = 0
    #: Clusters the label gate refused (:meth:`RecallMapBuilder._deliverable`).
    #: Deliberately *not* folded into ``dropped``: a harsh filter and a tight
    #: budget are different findings, and the field cannot tell them apart
    #: after the fact if the map reports one number for both.
    withheld: int = 0
    #: Names of what the filter removed, most reportable first, already cut to
    #: what :data:`MAX_RESPONSE_CHARS` afforded.
    filtered: tuple[FilteredCluster, ...] = ()
    #: Names the budget did not afford. The anti-silence field: the counts
    #: above are unconditional, so the map never drops a cluster without a
    #: number attached, and this says how many of them went unnamed.
    filtered_omitted: int = 0
    #: Set when this key's last :data:`CURTAIL_STREAK` maps went unconsumed.
    #: A curtailed map carries no clusters: it has stopped describing the pool
    #: and started reporting that describing it was not worth the channel.
    curtailed: bool = False
    #: Consecutive unconsumed deliveries behind the collapse. *The* signal —
    #: the number is what tells an operator (and the effect gate) that the map
    #: went quiet and how long it has been quiet for.
    streak: int = 0
    #: Member eligibility and relevance selection, computed before clustering.
    #: None means selection was not computed. It travels with every inspected
    #: residual so no later layer reconstructs counts from the capped pool.
    selection: SelectionAccounting | None = None
    #: Optional ``q`` examples that survived the response budget. When
    #: selection is computed, its core counts always survive fitting.
    selection_sample_limit: int = SELECTION_SAMPLE_LIMIT
    #: Digest of the pool this delivery was decided over (``pd``), present on
    #: exactly the deliveries a decay-armed key uses as a baseline: the first
    #: marker after an offer, and a served skip run that rebuilt the pool and
    #: found nothing new in it. ``None`` — the valve-unset value — emits
    #: nothing. Skips between digest baselines do not inspect the pool.
    pool_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The response-shaped form, budgeted by :data:`MAX_RESPONSE_CHARS`."""

        return _payload(
            self.clusters,
            self.pool_size,
            self.dropped,
            curtailed=self.curtailed,
            streak=self.streak,
            filtered=_filter_block(
                self.withheld, self.dropped, self.filtered, self.filtered_omitted
            ),
            selection=self.selection,
            selection_sample_limit=self.selection_sample_limit,
            pool_digest=self.pool_digest,
        )

    def plan_items(self) -> list[str]:
        return [cluster.plan_item for cluster in self.clusters]

    def render_compact(self) -> str:
        """One line, at most :data:`MAX_INSTRUCTIONS_CHARS` chars.

        Labels and counts only: the instructions channel has room for the
        *shape* of what memory holds, and the phrasings live one recall away.
        Clusters are dropped from the tail until the line fits, so the first —
        largest — cluster survives any budget.

        A curtailed map carries no clusters and so renders as the empty string,
        which is the entire point of the collapse: the channel it was spending
        is handed back rather than spent on a line nobody read.
        """

        prefix = "memory also holds: "
        parts: list[str] = []
        for cluster in self.clusters:
            candidate = [*parts, f"{cluster.label}({cluster.count})"]
            if len(prefix + " · ".join(candidate)) > MAX_INSTRUCTIONS_CHARS:
                break
            parts = candidate
        if not parts:
            return ""
        return prefix + " · ".join(parts)


# ----------------------------------------------------------------------
# Internal working shapes
# ----------------------------------------------------------------------


@dataclass(slots=True)
class _Member:
    """One admitted entry with its frozen score and original residual ordinal."""

    rank: int
    node: Node


@dataclass(slots=True)
class _AnchorAssignment:
    """One build's observed anchors and winning anchor for each covered node."""

    by_node: dict[str, str]
    anchors: dict[str, Any]


@dataclass(slots=True)
class _SelectionLedger:
    """Mutable accounting while the complete residual is classified."""

    inspected: int
    excluded: Counter[str] = field(default_factory=Counter)
    first_samples: dict[str, SelectionSample] = field(default_factory=dict)
    sampleable: int = 0

    def exclude(self, reason: str, *, ordinal: int, node: Any = None) -> None:
        self.excluded[reason] += 1
        if reason not in _SAMPLEABLE_REASONS or node is None:
            return
        gist = _selection_gist(node)
        if not gist:
            return
        self.sampleable += 1
        self.first_samples.setdefault(
            reason, SelectionSample(reason=reason, gist=gist, ordinal=ordinal)
        )

    def freeze(self, admitted: int) -> SelectionAccounting:
        """Close the ledger into the canonical, trimmed accounting.

        The exclusion vector is built over every code this module knows and
        then cut back to :data:`SELECTION_REASON_CODES` while its tail is
        zero. With both pool valves unset that cut always reaches the frozen
        width, so the payload is the one this module emitted before the gate
        codes existed — down to the length of ``x``.
        """

        samples = tuple(
            self.first_samples[reason]
            for reason in SELECTION_LEDGER_REASON_CODES
            if reason in self.first_samples
        )
        counts = [self.excluded[reason] for reason in SELECTION_LEDGER_REASON_CODES]
        while len(counts) > len(SELECTION_REASON_CODES) and not counts[-1]:
            counts.pop()
        return SelectionAccounting(
            inspected=self.inspected,
            admitted=admitted,
            excluded=tuple(counts),
            samples=samples,
            sampleable=self.sampleable,
        )


@dataclass(slots=True)
class _Group:
    """A cluster under construction, before labels are trimmed and capped."""

    stage: str
    #: Membership rule, and the cache's identity for this cluster. See
    #: :meth:`RecallMapBuilder._recount`.
    signature: tuple[Any, ...]
    label: str
    ask_hint: str
    members: list[_Member]
    #: Set only by stage 4, which is the only stage holding vectors.
    vectors: dict[str, list[float]] | None = None


@dataclass(slots=True)
class _FilterJournal:
    """What the filter removed on one build, before the budget names it.

    Two populations, kept apart all the way to the wire. They answer different
    questions — ``withheld`` says the gate found nothing worth saying,
    ``dropped`` says the channel had no room for something that was — and the
    no-silent-caps rule is only satisfied if a field reader can tell which.

    Order is the report order: the gate's refusals first, largest first, then
    the budget's drops newest-first. Both halves therefore lead with the
    cluster whose removal cost the reader most, which is what survives when
    :data:`FILTER_JOURNAL_NAMES` takes the tail.
    """

    withheld: list[FilteredCluster] = field(default_factory=list)
    dropped: list[FilteredCluster] = field(default_factory=list)
    #: Names are forensic convenience, not the honesty guarantee.  ``_fit``
    #: lowers this only after breadth reaches its floor and before gist does.
    name_limit: int = FILTER_JOURNAL_NAMES

    def entries(self) -> list[FilteredCluster]:
        return [*self.withheld, *self.dropped]

    def names(self) -> tuple[FilteredCluster, ...]:
        return tuple(self.entries()[: self.name_limit])

    def omitted(self) -> int:
        return max(0, len(self.entries()) - len(self.names()))

    def drop_name(self) -> bool:
        """Give one currently visible name to the gist budget."""

        visible = len(self.names())
        if visible <= 0:
            return False
        self.name_limit = visible - 1
        return True

    def block(self) -> dict[str, Any] | None:
        return _filter_block(
            len(self.withheld), len(self.dropped), self.names(), self.omitted()
        )


@dataclass(frozen=True, slots=True)
class _ClusterTemplate:
    """A cached cluster: its identity and its labels, without its counts."""

    stage: str
    signature: tuple[Any, ...]
    label: str
    ask_hint: str
    member_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class _CachedStructure:
    """One cache entry: templates plus the corpus revision that validates them."""

    revision: tuple[Any, ...]
    #: Empty only for a freshly selected empty pool.  This is a revision-bound
    #: negative structure result, not a wildcard: a later non-empty pool must
    #: rebuild rather than treating it as a reusable structure.
    templates: tuple[_ClusterTemplate, ...]


@dataclass(frozen=True, slots=True)
class _DeliveredItem:
    """One cluster of an already-delivered map, reduced to what proves use.

    Read back out of ``recall_events.recall_map``, so the fields are whatever
    :meth:`MapCluster.to_dict` wrote and nothing else: the two phrasings a
    later query can echo, and the node id a later delivery can reach.
    """

    medoid_id: str
    label: frozenset[str]
    ask_hint: frozenset[str]


@dataclass(frozen=True, slots=True)
class _Curtailment:
    """How long this key's map has been delivered without being followed."""

    #: Consecutive most-recent deliveries with nothing consumed, collapsed
    #: markers included. Reported to the agent and to the effect gate.
    streak: int
    #: Of those, the ones that actually offered clusters. Only an offer can go
    #: unaccepted, so only an offer may push the map towards silence.
    offers: int
    #: The newest deliveries of the streak that offered nothing — how far the
    #: key is into its current run of markers, zeroed the moment a map goes
    #: out. Only :attr:`retry_due` reads it; a verdict served from the memo
    #: carries the default, which is safe because the memo never serves a
    #: collapse and only a collapse can be retried.
    lead: int = 0
    #: The pool digest the newest lead row carries, when one does: the
    #: baseline a decay-armed retry compares the rebuilt pool against. Written
    #: by the first marker after an offer and by every retry that found
    #: nothing new (:meth:`RecallMapBuilder._reoffer`), read back off the same
    #: window the streak is, so it survives a restart and is shared by every
    #: process serving the key. ``None`` on a window written with the valve
    #: unset — which is every window until the valve is armed — and on one
    #: whose digest rows have all slid out.
    pool_digest: str | None = None
    #: Lead rows newer than the one carrying :attr:`pool_digest`. Counting the
    #: digest row itself, this is how much of the skip run the key has served
    #: since it last looked at its pool.
    since_digest: int = 0

    @property
    def collapse(self) -> bool:
        return self.offers >= CURTAIL_STREAK

    @property
    def retry_due(self) -> bool:
        """Whether a decay-armed key has served its skip run.

        Pure arithmetic on the verdict: the valve itself is read at the one
        collapse decision (:meth:`RecallMapBuilder._reoffer`), never here or
        below, so the probe path stays env-free and the unarmed decision is
        bit-identical to the frozen rule. The run is the registered
        ``min(2**(N-1), 8)`` with ``N`` the unaccepted offers — 4 at the
        entry threshold, 8 from the fourth on — and it is measured against a
        real read, because every collapse verdict comes off one.

        The run is counted from the last time the key looked at its pool: the
        digest row when the window carries one (the first marker after an
        offer, or a retry that found nothing new), the head of the lead when
        it carries none, which is every window the frozen code wrote and so
        the frozen schedule exactly.
        """

        if not self.collapse:
            return False
        served = self.lead if self.pool_digest is None else self.since_digest + 1
        return served >= min(2 ** (self.offers - 1), CURTAIL_DECAY_SKIP_MAX)


@dataclass(slots=True)
class _CurtailMemo:
    """A read verdict, plus what this builder has delivered since reading it.

    The history read is the one genuinely expensive thing on this path and it
    is expensive for a structural reason that no caller can fix: no index
    covers ``(scope, task, recall_map IS NOT NULL)`` in time order, so SQLite
    walks the whole scope partition and sorts — measured at 8.5 ms on a
    21k-event scope, on a recall whose median is 100 ms, *and rising with the
    partition*.

    So the read is asked only when its answer could matter. A collapse needs
    :data:`CURTAIL_STREAK` unaccepted offers; this builder knows how many
    offers it has made since it last read, and while the last read plus that
    count is still under the threshold, no history the read could return would
    collapse anything. Counting only forward, never back, is what makes the
    shortcut safe in the one direction that matters: another process's
    deliveries are invisible to it, so it can *under*-count and skip a
    collapse, and consumption is invisible to it, so it can *over*-count and
    ask for a read that turns out unnecessary. It can never collapse a map on
    its own word — every collapse in this module is decided by a read.

    A *confirmed* collapse is deliberately not held. The estimate stays at the
    threshold, so every collapsed delivery reads again — which is exactly what
    lets consumption reopen the channel on the very next recall instead of
    some deliveries later. It is affordable because a collapsed build skips
    the whole label cascade it replaces: 8.5 ms of read against the ~12 ms of
    clustering it no longer does. Measured over a replayed server loop on that
    21k-event scope: a key that is being followed spends 3.2 ms per delivery
    on this probe and answers two builds in three from memory, and a key that
    is not gets *faster* than it was before the rule existed.
    """

    verdict: _Curtailment
    #: Deliveries this builder has made under the key since the read, that
    #: carried clusters — the ones capable of going unaccepted.
    offers: int = 0
    #: All of them, collapsed markers included.
    total: int = 0
    def estimate(self) -> _Curtailment:
        return _Curtailment(
            streak=self.verdict.streak + self.total,
            offers=self.verdict.offers + self.offers,
        )


# ----------------------------------------------------------------------
# Normalization, mirrored deliberately
# ----------------------------------------------------------------------


def normalize_key(value: str) -> str:
    """Fold a slug-shaped identifier into readable words.

    Mirrors ``consolidation._normalize_trigger`` (consolidation.py:820) rather
    than importing it: this module's whole point is to be cheap and free of the
    consolidation stack, and the transform is four characters of policy. The
    mirror is pinned against the original in ``tests/test_recall_map.py`` so it
    cannot drift silently.
    """

    cleaned = str(value).replace("_", " ").replace("-", " ").replace("/", " ")
    return _WHITESPACE_RE.sub(" ", cleaned).strip().lower()


def _looks_unreadable(value: str) -> bool:
    """Whether a raw structural key is a hash rather than a name.

    ``task_pattern`` is routinely a content hash, and a hash makes a fine
    grouping key and a useless label. Three signals, all conservative: a long
    pure-hex run, a string that is half digits, and a long unpronounceable run
    with no vowel in it. Anything containing a space has already survived
    normalization as words and is left alone.
    """

    if not value:
        return True
    if " " in value:
        return False
    lowered = value.lower()
    if len(lowered) < 8:
        return False
    if len(lowered) >= 12 and all(char in _HEX_DIGITS for char in lowered):
        return True
    digits = sum(1 for char in lowered if char.isdigit())
    if digits * 2 >= len(lowered):
        return True
    return len(lowered) >= 16 and not any(char in _VOWELS for char in lowered)


def _collapse(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(text)).strip()


def _shorten(text: str, limit: int) -> str:
    collapsed = _collapse(text)
    if limit <= 0:
        return ""
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _utc_now() -> str:
    """One canonical map-build instant, captured before any history read."""

    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _is_file_chunk_envelope(envelope: Any) -> bool:
    """The evaluator's strict file-chunk schema, with no provenance lookup."""

    if not isinstance(envelope, dict):
        return False
    chunk = envelope.get("chunk")
    lines = envelope.get("lines")
    if not isinstance(chunk, str) or _FILE_CHUNK_INDEX_RE.fullmatch(chunk) is None:
        return False
    if not isinstance(lines, str) or _FILE_CHUNK_LINES_RE.fullmatch(lines) is None:
        return False
    part, total = (int(value) for value in chunk.split("/"))
    first_line, last_line = (int(value) for value in lines.split("-"))
    return (
        part <= total
        and first_line <= last_line
        and isinstance(envelope.get("path"), str)
        and bool(envelope["path"].strip())
        and isinstance(envelope.get("kind"), str)
        and bool(envelope["kind"].strip())
        and isinstance(envelope.get("language"), str)
        and isinstance(envelope.get("sha256"), str)
        and _SHA256_RE.fullmatch(envelope["sha256"]) is not None
    )


def _ballast_reason(
    content: Any,
    context: Any = None,
    provenance: Any = None,
) -> str | None:
    """Return ``fc``/``ss``/``sj`` for an exact machine envelope.

    Every parsing and shape failure is eligible.  In particular, marker words
    embedded in prose and journal names under ``topic`` are not classifiers.
    This function reads only the candidate's own form/schema/provenance; it has
    no emitter, host, project, task, label or identity surface.
    """

    if not isinstance(content, str):
        return None

    if content.startswith("[file-chunk]"):
        suffix = content[len("[file-chunk]") :]
        separator = suffix[:1]
        header = ""
        if separator in {" ", "\t"}:
            lines = suffix.lstrip(" \t").splitlines()
            header = lines[0].strip() if lines else ""
        try:
            envelope = json.loads(header)
        except (TypeError, ValueError):
            envelope = None
        if _is_file_chunk_envelope(envelope):
            return "fc"

    if _STRATEGY_STAGNATION_RE.fullmatch(content):
        return "ss"

    kind_values: set[str] = set()
    for mapping in (context, provenance):
        if not isinstance(mapping, Mapping):
            continue
        for key in _JOURNAL_METADATA_KEYS:
            try:
                value = mapping.get(key)
            except (AttributeError, TypeError, ValueError):
                continue
            kind_values.add(str(value or "").strip().lower().replace("-", "_"))
    first_line = content.splitlines()[:1]
    marker = (first_line[0] if first_line else "").strip().lower().replace("-", "_")
    if kind_values & _JOURNAL_KINDS or marker in {
        "[supervision_journal]",
        "[monitoring_journal]",
    }:
        return "sj"
    return None


def _gate_flag(name: str) -> bool:
    """Whether an operator turned this gate on, drain-valve spelling."""

    return os.environ.get(name, "").strip().lower() in _POOL_GATE_ON_FLAGS


def pool_usefulness_floor_from_env() -> float | None:
    """The usefulness floor, or ``None`` when gate (a) is not armed.

    Armed means both halves: ``LM_MAP_POOL_USEFULNESS_GATE`` on *and*
    ``LM_MAP_POOL_MIN_USEFULNESS`` holding a number. Unlike the drain's
    cosine there is no default to fall back to — the value comes off a field
    measurement, and inventing one here would be this module deciding a
    production threshold it has no evidence for. A valve on with no number,
    or with a number that does not parse, therefore leaves the pool exactly as
    it was: the operator asked for a floor and did not say where, and the
    honest answer to that is no floor rather than a guess.
    """

    if not _gate_flag(POOL_USEFULNESS_GATE_ENV):
        return None
    raw = os.environ.get(POOL_USEFULNESS_FLOOR_ENV, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value != value:  # NaN compares false against everything, gate included
        return None
    return value


def curtail_decay_from_env() -> bool:
    """Whether current work retries with exponential, pool-aware skips."""

    return _gate_flag(CURTAIL_DECAY_GATE_ENV)


def _below_usefulness(node: Any, floor: float) -> bool:
    """Whether this candidate's usefulness verdict sits under the floor.

    Read off the node the residual already carries, so no query joins the hot
    path for it. A node whose score is missing or not a real number is *not*
    below the floor: absence of a verdict is not a low verdict, and this gate
    only ever removes rows, so the unknown case has to resolve towards keeping
    them.

    That case is about the *object*, not the corpus. ``nodes.usefulness_score``
    is ``NOT NULL DEFAULT 0.0``, so a node nobody has ever scored arrives here
    as a genuine ``0.0`` and is judged against the floor like any other number
    — which is right, because the floor is chosen from a census of that same
    distribution, zeroes included. The guard is for the duck-typed faces this
    module reads through everywhere else: a result object from another layer
    that has no such attribute must not be silently read as worthless.
    """

    value = getattr(node, "usefulness_score", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    score = float(value)
    if score != score:
        return False
    return score < floor


def _selection_gist(node: Any) -> str:
    """An identity-free, compact content example for selection forensics."""

    content = _collapse(getattr(node, "content", ""))
    node_id = getattr(node, "id", None)
    if isinstance(node_id, str) and node_id:
        content = _collapse(content.replace(node_id, ""))
    return _shorten(content, SELECTION_SAMPLE_GIST_CHARS)


def _terms(text: str) -> list[str]:
    """Content terms of ``text``, unstemmed, in order.

    Unstemmed on purpose: these strings are looked up in the FTS
    document-frequency index, whose ``unicode61`` tokenizer does not stem, so a
    stemmed term would report a document frequency of 0 and score as maximally
    rare. Stop-word policy is still the shared one — a raw token is kept when
    ``embeddings.tokenize`` finds anything in it — so this does not fork a
    second stop list.
    """

    kept: list[str] = []
    decided: dict[str, bool] = {}
    for raw in _TERM_RE.findall(str(text).lower()):
        if len(raw) < 3:
            continue
        verdict = decided.get(raw)
        if verdict is None:
            verdict = bool(tokenize(raw))
            decided[raw] = verdict
        if verdict:
            kept.append(raw)
    return kept


def _phrase_from_content(content: str, *, words: int = 3) -> str:
    """A short readable phrase from a node's own content."""

    terms = _terms(content)[:words]
    return " ".join(terms) if terms else _shorten(content, MAX_LABEL_CHARS)


def _followed_hint(stage: str, label: str, previous: str) -> str:
    """The ask-hint a disambiguated label leaves behind.

    On stages 1 and 2 the hint follows the label's *head* rather than the whole
    of it, capped at :data:`ASK_HINT_MAX_TOKENS`. Disambiguation appends a
    distinguishing term to a label, and a hint that followed it there would
    reach three tokens — which is where ``_echoes`` starts demanding two shared
    tokens instead of one and quietly tightens a pre-registered rule.

    The cap is never allowed to *shorten* a hint, and the containment check is
    what makes that a guarantee rather than an intention: the widened hint must
    still carry every token the old one had, under the same tokenizer
    ``_echoes`` uses, or the old hint is kept unchanged. A query that cleared
    the old phrasing therefore clears the new one by construction.

    Stage 4's hint is left following its label exactly as before — a
    c-TF-IDF label is already three tokens, so a fourth changes nothing about
    what ``_echoes`` demands — and stage 3's is never rewritten at all.
    """

    if stage not in (STAGE_PATH, STAGE_STRUCTURAL):
        return label
    keep = max(ASK_HINT_MAX_TOKENS, len(previous.split()))
    candidate = " ".join(label.split()[:keep])
    if not candidate or not token_set(previous) <= token_set(candidate):
        return previous
    return candidate


# ----------------------------------------------------------------------
# Stage helpers
# ----------------------------------------------------------------------


def _structural_key(node: Node) -> tuple[str, str] | None:
    """``(field, raw value)`` of the first structural key this node carries."""

    context = node.context if isinstance(node.context, Mapping) else {}
    for key_field in STRUCTURAL_FIELDS:
        value = context.get(key_field)
        if isinstance(value, str) and value.strip():
            return (key_field, value.strip())
    return None


def _iter_path_strings(node: Node) -> Iterable[str]:
    """Path-like strings this node's context carries.

    Context is read, content is not. A path in prose is a mention; a path in
    ``context['files']`` is a claim about what the node is *about*, which is the
    only thing a subsystem label may be built from. Summarized context values
    (the ``{"count": n, "chars": m}`` shape the delivery diet writes over long
    lists) hold no paths and are skipped by the string check.
    """

    context = node.context if isinstance(node.context, Mapping) else {}
    for path_field in _PATH_FIELDS:
        value = context.get(path_field)
        if isinstance(value, str):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    yield item


def _subsystem(path: str) -> str | None:
    """Collapse one path to the subsystem that owns it.

    Directory segments only — the file name is the leaf, never the subsystem —
    starting after the last container segment (:data:`_PATH_ROOT_SEGMENTS`) and
    taking the deeper of the first two that remain. That is what makes
    ``src/living_memory/postsession/report.py`` read as ``postsession`` while
    ``src/living_memory/storage.py`` stays ``living_memory``: the rule descends
    exactly as far as the path affords. A path with a container segment and
    nothing after it (``tests/test_map.py``) keeps the container, which for
    ``tests`` and ``docs`` is the right subsystem anyway.
    """

    cleaned = str(path).split(":", 1)[0].strip().replace("\\", "/")
    if "/" not in cleaned:
        return None
    parts = [part for part in cleaned.split("/") if part not in ("", ".", "..")]
    directories = parts[:-1]
    if not directories:
        return None
    start = 0
    for index, segment in enumerate(directories):
        if segment.lower() in _PATH_ROOT_SEGMENTS:
            start = index + 1
    tail = directories[start:]
    if not tail:
        return directories[-1]
    return tail[1] if len(tail) >= 2 else tail[0]


def _node_subsystem(node: Node) -> str | None:
    """The subsystem a node belongs to, when its paths agree on one.

    A node touching several subsystems is assigned to the one it names most
    often, ties broken lexicographically, so the answer does not depend on the
    order a context list happened to be written in.
    """

    counted: Counter[str] = Counter()
    for candidate in _iter_path_strings(node):
        subsystem = _subsystem(candidate)
        if subsystem:
            counted[subsystem] += 1
    if not counted:
        return None
    return min(counted.items(), key=lambda item: (-item[1], item[0]))[0]


def _normalize_vector(values: Sequence[float]) -> list[float] | None:
    norm = sqrt(sum(float(value) * float(value) for value in values))
    if norm <= 0.0:
        return None
    return [float(value) / norm for value in values]


# ----------------------------------------------------------------------
# Consumption: was a delivered map followed?
# ----------------------------------------------------------------------


def _delivered_items(payload: Any) -> list[_DeliveredItem]:
    """The consumable items of one persisted map payload.

    Defensive to the point of paranoia about shape, because this reads a JSON
    blob written by whatever version of this module was running when the
    delivery happened — including versions that do not exist yet. Anything it
    cannot parse contributes no items, which makes that delivery unjudgeable
    rather than unconsumed: a payload this code cannot read is not evidence
    that nobody read it.
    """

    if not isinstance(payload, Mapping):
        return []
    clusters = payload.get("clusters")
    if not isinstance(clusters, (list, tuple)):
        return []
    items: list[_DeliveredItem] = []
    for entry in clusters[:MAX_CLUSTERS]:
        if not isinstance(entry, Mapping):
            continue
        medoid = entry.get("medoid")
        node_id = ""
        if isinstance(medoid, Mapping):
            node_id = str(medoid.get("node_id") or "")
        items.append(
            _DeliveredItem(
                medoid_id=node_id,
                label=token_set(str(entry.get("label") or "")),
                ask_hint=token_set(str(entry.get("ask_hint") or "")),
            )
        )
    return items


def _echoes(phrasing: frozenset[str], query: frozenset[str]) -> bool:
    """Whether ``query`` carries enough of ``phrasing`` to count as asking it.

    Containment of the *phrasing* in the query, not the reverse: a plan item
    says "recall X", and the recall that follows it is free to say more than X
    — it usually does. Weighting the shared tokens by rarity, as
    :mod:`living_memory.grounding` does for content, would need a corpus this
    path has no budget to build; a label is a handful of tokens the cascade
    already chose for being distinguishing, so unweighted containment over the
    shared tokenizer is the honest measure at this size.
    """

    if not phrasing or not query:
        return False
    shared = len(phrasing & query)
    return shared >= CURTAIL_QUERY_OVERLAP * len(phrasing)


# ----------------------------------------------------------------------
# Cheap revision probes, mirroring retrieval's
# ----------------------------------------------------------------------


def _data_version(connection: sqlite3.Connection) -> int:
    """``PRAGMA data_version``: bumped when another connection commits."""

    row = connection.execute("PRAGMA data_version").fetchone()
    return int(row[0]) if row is not None else 0


def _chunk_table_revision(connection: sqlite3.Connection) -> tuple[Any, ...]:
    """``(row count, greatest chunk id)``, or ``()`` on a pre-v6 database.

    A delete moves the count and a new maximum id moves the second component.
    The build's write probe gates this scan; it does not import the ranker or
    its embedding model.
    """

    try:
        counted = connection.execute(
            f"SELECT COUNT(*) FROM {CHUNK_EMBEDDING_TABLE}"
        ).fetchone()
        greatest = connection.execute(
            f"SELECT MAX(id) FROM {CHUNK_EMBEDDING_TABLE}"
        ).fetchone()
    except sqlite3.OperationalError:
        return ()
    return (int(counted[0]), greatest[0])


# ----------------------------------------------------------------------
# The builder
# ----------------------------------------------------------------------


class RecallMapBuilder:
    """Builds recall maps, and remembers their shape between calls.

    One instance per service, like ``MemoryRecallService``'s own caches: the
    stability guarantee is "same task, same structure", which only means
    anything if the thing holding it outlives a single call.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        max_clusters: int = MAX_CLUSTERS,
        max_pool_nodes: int = MAX_POOL_NODES,
        cluster_cosine: float = EMBEDDING_CLUSTER_COSINE,
    ) -> None:
        self.store = store
        self.max_clusters = max(1, int(max_clusters))
        self.max_pool_nodes = max(1, int(max_pool_nodes))
        self.cluster_cosine = float(cluster_cosine)
        #: Whether the last :meth:`build` served its structure from cache.
        #: Read by the latency gate and by the stability tests.
        self.last_cache_hit: bool = False
        #: What the last :meth:`build` found about this key's unused streak,
        #: including the builds that did *not* collapse — the response only
        #: carries the number once it has become a decision, and the runs up to
        #: that point are what a latency gate or a test needs to see.
        self.last_curtailment: _Curtailment = _Curtailment(streak=0, offers=0)
        #: Whether the last :meth:`build` paid for a history read, or answered
        #: the curtail question from what it had already delivered itself.
        self.last_curtail_read: bool = False
        self._cache: dict[str, _CachedStructure] = {}
        self._curtail_memo: dict[str, _CurtailMemo] = {}
        self._write_probe: tuple[int, int] | None = None
        self._revision: tuple[Any, ...] | None = None
        #: Corpus statistics for the build in flight, reset by :meth:`build`.
        #: Label enrichment asks the document-frequency index once per bucket
        #: and the gate once per label, which without a memo would be one
        #: ``COUNT(*)`` per question; the corpus cannot move under a single
        #: build, so one read answers all of them.
        self._fts_total: int | None = None
        self._term_df: dict[str, int] = {}
        self._df_available = True

    # -- public API ----------------------------------------------------

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def build(
        self,
        results: Sequence["RecallResult"],
        *,
        scope: str | None = None,
        task: str | None = None,
        task_pattern: str | None = None,
        transport_session_id: str | None = None,
        relevance: QueryRelevance | None = None,
    ) -> RecallMap | None:
        """Cluster one residual pool, or return None when there is nothing to say.

        Work identity keys both structure and curtailment. Explicit task_pattern
        takes precedence over task and supports continuation across transports.
        Otherwise transport_session_id identifies the consumer. Absent context
        bypasses both caches. Scope must match the recorded recall's scope.

        The shared relevance evidence includes the full ranking and correction
        edges. The pool is built only for an offer, initial digest or due retry;
        skipped calls carry no pool/selection counts."""

        self.last_cache_hit = False
        self._reset_corpus_memo()
        if not results:
            return None

        map_scope = (
            normalize_scope(scope) if scope else _dominant_residual_scope(results)
        )
        work = RecallWork.resolve(
            map_scope, task=task, task_pattern=task_pattern,
            transport_session_id=transport_session_id,
        )
        key = work.key if work else ""
        curtailment = self._curtailment(work)
        self.last_curtailment = curtailment
        pool: list[_Member] | None = None
        selection: SelectionAccounting | None = None
        anchor_assignment: _AnchorAssignment | None = None

        def ensure_pool() -> list[_Member]:
            nonlocal pool, selection
            if pool is None:
                pool, selection = self._pool(
                    results,
                    relevance=relevance,
                )
            return pool

        def digest() -> str:
            nonlocal anchor_assignment
            members = ensure_pool()
            if anchor_assignment is None:
                anchor_assignment = self._anchor_assignment(members)
            return self._pool_digest(
                members, assignment=anchor_assignment.by_node,
            )

        offer, pool_digest = self._reoffer(curtailment, digest)
        if not offer and pool is None:
            self._note_delivery(work, offered=False)
            return self._collapsed(
                map_scope, key, pool_size=None, streak=curtailment.streak,
                selection=None, pool_digest=None,
            )

        pool = ensure_pool()
        assert selection is not None
        if not pool:
            # A non-empty residual that policy filtered completely is evidence,
            # not the same event as an empty residual.  It offered no cluster
            # and therefore cannot advance curtailment.  Its empty structure
            # is still a useful negative cache result, but only for a fresh
            # empty selection under this exact corpus/policy revision.  Keep
            # this branch explicit: `_recount((), pool)` deliberately rejects
            # so an empty entry can never hide a later eligible pool.
            revision = self._corpus_revision()
            cached = self._cache.get(key)
            if (
                cached is not None
                and cached.revision == revision
                and not cached.templates
            ):
                self.last_cache_hit = True
            else:
                self._remember(key, revision, [])
            self.last_curtailment = _Curtailment(streak=0, offers=0)
            built = RecallMap(
                scope=map_scope,
                key=key,
                clusters=(),
                pool_size=0,
                covered=0,
                selection=selection,
                selection_sample_limit=_selection_sample_limit(
                    (), 0, 0, selection=selection
                ),
            )
            if not built.clusters:
                built = replace(built, pool_digest=pool_digest)
            self._note_delivery(
                work, offered=bool(built.clusters)
            )
            return built

        if not offer:
            self._note_delivery(work, offered=False)
            return self._collapsed(
                map_scope,
                key,
                pool_size=len(pool),
                streak=curtailment.streak,
                selection=selection,
                pool_digest=pool_digest,
            )

        revision = self._corpus_revision()
        cached = self._cache.get(key)
        groups: list[_Group] | None = None
        if (
            cached is not None
            and cached.revision == revision
            and cached.templates
        ):
            groups = self._recount(cached.templates, pool)
            self.last_cache_hit = groups is not None
        if groups is None:
            groups = self._cluster(pool, anchor_assignment=anchor_assignment)
            self._remember(key, revision, groups)
        built = self._finish(
            groups,
            scope=map_scope,
            key=key,
            pool_size=len(pool),
            selection=selection,
        )
        if built is not None and pool_digest is not None and not built.clusters:
            # A reprieved key that rebuilt its map and found nothing deliverable
            # in it did not wake up. The wire says so — a marker, not an empty
            # uncollapsed map — and the marker carries the digest of the pool
            # that had nothing to say, so the next look compares against it
            # rather than re-discovering the same nothing every delivery.
            self._note_delivery(work, offered=False)
            return self._collapsed(
                map_scope,
                key,
                pool_size=len(pool),
                streak=curtailment.streak,
                selection=selection,
                pool_digest=pool_digest,
            )
        if built is not None:
            # A journal-only map reaches the server so a fully gated pool does
            # not disappear from field evidence, but it offered the agent no
            # cluster and therefore must not advance the curtail offer count.
            self._note_delivery(
                work, offered=bool(built.clusters)
            )
        return built

    # -- pool ----------------------------------------------------------

    def _pool(
        self,
        results: Sequence["RecallResult"],
        *,
        relevance: QueryRelevance | None = None,
    ) -> tuple[list[_Member], SelectionAccounting]:
        """Qualify by this query, then share bounded capacity across topics.

        The shared predicate includes retrieval's scope/causal adjustments,
        demotions and correction inheritance. History of other deliveries does
        not qualify a candidate. Structural/path buckets get a turn before a
        large bucket gets another member; within each bucket retrieval order
        is preserved. Only the bounded pool pays for anchor/vector clustering.
        """

        eligible_ids = (relevance or QueryRelevance(results)).eligible_ids()
        usefulness_floor = pool_usefulness_floor_from_env()
        ledger = _SelectionLedger(inspected=len(results))
        seen: set[str] = set()
        buckets: dict[tuple[str, ...], deque[_Member]] = {}
        for ordinal, result in enumerate(results):
            node = getattr(result, "node", None)
            node_id = getattr(node, "id", None) if node is not None else None
            if not isinstance(node_id, str) or not node_id.strip():
                ledger.exclude("iv", ordinal=ordinal)
                continue
            if node_id in seen:
                ledger.exclude("du", ordinal=ordinal)
                continue
            seen.add(node_id)
            reason = _ballast_reason(
                getattr(node, "content", None),
                getattr(node, "context", None),
                getattr(node, "provenance", None),
            )
            if reason is not None:
                ledger.exclude(reason, ordinal=ordinal, node=node)
                continue
            if usefulness_floor is not None and _below_usefulness(node, usefulness_floor):
                ledger.exclude("uf", ordinal=ordinal, node=node)
                continue
            if node_id not in eligible_ids:
                ledger.exclude("lr", ordinal=ordinal, node=node)
                continue
            structural = _structural_key(node)
            path = _node_subsystem(node) if structural is None else None
            key = (
                (STAGE_STRUCTURAL, *structural) if structural is not None
                else (STAGE_PATH, path) if path is not None
                else (STAGE_EMBEDDING,)
            )
            buckets.setdefault(key, deque()).append(_Member(rank=ordinal, node=node))

        admitted: list[_Member] = []
        active = deque(buckets.values())
        while active and len(admitted) < self.max_pool_nodes:
            bucket = active.popleft()
            admitted.append(bucket.popleft())
            if bucket:
                active.append(bucket)
        for member in sorted(
            (member for bucket in active for member in bucket), key=lambda item: item.rank,
        ):
            ledger.exclude("pc", ordinal=member.rank, node=member.node)
        admitted.sort(key=lambda member: member.rank)
        return admitted, ledger.freeze(len(admitted))

    # -- cascade -------------------------------------------------------

    def _cluster(
        self, pool: list[_Member], *,
        anchor_assignment: _AnchorAssignment | None = None,
    ) -> list[_Group]:
        """Run the four stages, each over what the previous one left."""

        groups: list[_Group] = []
        remaining = pool

        claimed, remaining = self._stage_structural(remaining)
        groups.extend(claimed)
        claimed, remaining = self._stage_path(remaining)
        groups.extend(claimed)
        claimed, remaining = self._stage_anchor(remaining, anchor_assignment)
        groups.extend(claimed)
        claimed, remaining = self._stage_embedding(remaining)
        groups.extend(claimed)
        return self._disambiguate(groups)

    def _disambiguate(self, groups: list[_Group]) -> list[_Group]:
        """Make every label unique, because the map is a menu.

        Two rows reading the same text are a defect whatever produced them —
        the reader cannot act differently on them, and a repeated label is the
        degenerate map this cascade exists to avoid. Collisions are not
        hypothetical: two c-TF-IDF clusters can share their top terms, and
        ``type: root-cause`` and ``lesson_kind: root_cause`` normalize to the
        same words from two different fields.

        The first group to claim a label keeps it; later ones extend theirs with
        their own next-most-distinguishing term, and fall back to a numeric
        suffix only if a group has no term left to offer. Merging the
        colliding groups instead would be simpler and wrong: they were
        separated by evidence — different keys, different vector regions — and
        a label collision is a failure to *name* that difference, not proof
        that it is absent.

        Uniqueness is enforced on the *displayed* label, already cut to
        :data:`MAX_LABEL_CHARS`. Deciding it on the full string instead would
        let two anchor queries sharing a long prefix — the house style that
        makes anchors work in the first place — pass this pass and collide
        again in the response.
        """

        seen: set[str] = set()
        for group in groups:
            # An anchor's ask-hint is a query that already worked, and it is
            # never rewritten — not even though it starts out equal to the
            # label. Everywhere else a hint that *was* the label follows it,
            # but only as far as :func:`_followed_hint` allows: on the enriched
            # stages the hint tracks the label's head, not its whole width.
            hint_followed_label = (
                group.stage != STAGE_ANCHOR and group.ask_hint == group.label
            )
            resolved = self._unique_label(group, seen)
            seen.add(resolved)
            if hint_followed_label:
                group.ask_hint = _followed_hint(group.stage, resolved, group.ask_hint)
            group.label = resolved
        return groups

    def _unique_label(self, group: _Group, seen: set[str]) -> str:
        label = _shorten(group.label, MAX_LABEL_CHARS)
        if label and label not in seen:
            return label
        for term in self._ranked_terms(group.members):
            if term in label.split():
                continue
            # Make room for the term rather than overflowing the width, so the
            # distinguishing part cannot be the part that gets cut off.
            base = _shorten(group.label, max(1, MAX_LABEL_CHARS - len(term) - 1))
            candidate = f"{base} {term}"
            if candidate not in seen:
                return candidate
        suffix = 2
        while True:
            marker = f" #{suffix}"
            candidate = f"{_shorten(group.label, MAX_LABEL_CHARS - len(marker))}{marker}"
            if candidate not in seen:
                return candidate
            suffix += 1

    def _stage_structural(
        self, members: list[_Member]
    ) -> tuple[list[_Group], list[_Member]]:
        buckets: dict[tuple[str, str], list[_Member]] = {}
        rest: list[_Member] = []
        for member in members:
            key = _structural_key(member.node)
            if key is None:
                rest.append(member)
                continue
            buckets.setdefault(key, []).append(member)

        groups: list[_Group] = []
        for (structural_field, raw), bucket in sorted(buckets.items()):
            normalized = normalize_key(raw)
            if _looks_unreadable(raw):
                # A hash groups perfectly and names nothing; the medoid does.
                readable = _phrase_from_content(bucket[0].node.content)
            else:
                readable = normalized
            label, hint = readable, readable
            if not self._deliverable(readable):
                # A structural value that is one house word is exactly as
                # undeliverable as a bare directory segment — 6 of the 55
                # ``server`` clusters in the field window were structural, not
                # path — so it gets the same rescue. A value that already
                # passes is left alone: ``cleanup commit invariant`` needs no
                # help, and rewriting a key the corpus actually stores would
                # throw away the one label form the control run proved works.
                label, hint = self._enrich(readable, bucket)
            groups.append(
                _Group(
                    stage=STAGE_STRUCTURAL,
                    signature=(STAGE_STRUCTURAL, structural_field, raw),
                    label=label,
                    ask_hint=hint,
                    members=bucket,
                )
            )
        return groups, rest

    def _stage_path(self, members: list[_Member]) -> tuple[list[_Group], list[_Member]]:
        """Collapse paths to subsystems, then name the subsystem's own content.

        The subsystem alone is a directory, and a directory is what the field
        measured as the dominant failure of this map: enrichment is
        unconditional here rather than gated on the score, because a path
        segment is never the cluster's own vocabulary — it is where the files
        happen to live.
        """

        buckets: dict[str, list[_Member]] = {}
        rest: list[_Member] = []
        for member in members:
            subsystem = _node_subsystem(member.node)
            if subsystem is None:
                rest.append(member)
                continue
            buckets.setdefault(subsystem, []).append(member)

        groups: list[_Group] = []
        for subsystem, bucket in sorted(buckets.items()):
            label, hint = self._enrich(normalize_key(subsystem), bucket)
            groups.append(
                _Group(
                    stage=STAGE_PATH,
                    signature=(STAGE_PATH, subsystem),
                    label=label,
                    ask_hint=hint,
                    members=bucket,
                )
            )
        return groups, rest

    def _anchor_assignment(
        self, members: Sequence[_Member]
    ) -> _AnchorAssignment:
        """``node_id -> anchor_id`` for every member a live anchor covers.

        Observe each reached anchor once, including missing rows. A digest
        needs the whole pool; stage 3 uses only the remaining members of this
        assignment. Without a digest, only stage 3's members need observation.
        """

        anchors: dict[str, Any] = {}
        assignment: dict[str, str] = {}
        for member in members:
            best: tuple[float, str] | None = None
            for edge in self.store.list_query_anchor_edges(target_id=member.node.id):
                if edge.anchor_id not in anchors:
                    anchors[edge.anchor_id] = self.store.get_query_anchor(edge.anchor_id)
                anchor = anchors[edge.anchor_id]
                if anchor is None or anchor.decayed or not str(anchor.query).strip():
                    continue
                weight = float(edge.weight)
                if best is None or (-weight, edge.anchor_id) < (-best[0], best[1]):
                    best = (weight, edge.anchor_id)
            if best is not None:
                assignment[member.node.id] = best[1]
        return _AnchorAssignment(assignment, anchors)

    def _stage_anchor(
        self, members: list[_Member], assignment: _AnchorAssignment | None = None,
    ) -> tuple[list[_Group], list[_Member]]:
        """Group by the live query anchor whose edges cover each node.

        Reuse the digest's observation when present; otherwise read only the
        unclaimed nodes. A node covered by several anchors goes to the heaviest
        edge, ties to the lower anchor id, independent of edge-table order.
        """

        if assignment is None:
            assignment = self._anchor_assignment(members)

        buckets: dict[str, list[_Member]] = {}
        rest: list[_Member] = []
        for member in members:
            anchor_id = assignment.by_node.get(member.node.id)
            if anchor_id is not None:
                buckets.setdefault(anchor_id, []).append(member)
            else:
                rest.append(member)

        groups: list[_Group] = []
        for anchor_id, bucket in sorted(buckets.items()):
            query = _collapse(assignment.anchors[anchor_id].query)
            groups.append(
                _Group(
                    stage=STAGE_ANCHOR,
                    signature=(STAGE_ANCHOR, anchor_id),
                    label=query,
                    # The anchor's own wording, unedited: it is not a guess at
                    # how to ask, it is a question that already worked.
                    ask_hint=query,
                    members=bucket,
                )
            )
        return groups, rest

    def _stage_embedding(
        self, members: list[_Member]
    ) -> tuple[list[_Group], list[_Member]]:
        """Greedy leader clustering over chunk vectors, c-TF-IDF labels.

        Leader rather than centroid clustering, because a moving centroid makes
        membership depend on arrival order in a way the rank order alone cannot
        justify: here a node joins the *first-seeded* cluster it is closest to,
        and the seed is by construction the highest-ranked member of it.
        """

        vectors: dict[str, list[float]] = {}
        clusterable: list[_Member] = []
        rest: list[_Member] = []
        for member in members:
            vector = self._node_vector(member.node)
            if vector is None:
                rest.append(member)
                continue
            vectors[member.node.id] = vector
            clusterable.append(member)
        if not clusterable:
            return [], rest

        leaders: list[tuple[list[float], list[_Member]]] = []
        for member in clusterable:
            vector = vectors[member.node.id]
            best_index = -1
            best_similarity = 0.0
            for index, (leader_vector, _bucket) in enumerate(leaders):
                if len(leader_vector) != len(vector):
                    continue
                similarity = cosine_similarity(leader_vector, vector)
                # Strictly better, so an exact tie keeps the earlier-seeded
                # cluster and the walk stays a function of rank order alone.
                if similarity >= self.cluster_cosine and similarity > best_similarity:
                    best_index = index
                    best_similarity = similarity
            if best_index < 0:
                leaders.append((vector, [member]))
            else:
                leaders[best_index][1].append(member)

        groups: list[_Group] = []
        for _leader_vector, bucket in leaders:
            label = self._ctfidf_label(bucket)
            groups.append(
                _Group(
                    stage=STAGE_EMBEDDING,
                    signature=(
                        STAGE_EMBEDDING,
                        tuple(sorted(member.node.id for member in bucket)),
                    ),
                    label=label,
                    ask_hint=label,
                    members=bucket,
                    vectors={member.node.id: vectors[member.node.id] for member in bucket},
                )
            )
        return groups, rest

    def _node_vector(self, node: Node) -> list[float] | None:
        """One unit vector per node: the mean of its chunk vectors.

        Chunks first, because they are what the store keeps for every node that
        has been embedded at all, and the node-level vector second, for a
        fixture or a freshly written node whose chunks have not landed yet.
        Chunks of a width other than the first one's are dropped rather than
        averaged — vectors of two widths are points in two different spaces.
        """

        rows = [
            list(chunk.embedding)
            for chunk in self.store.list_node_chunks(node.id)
            if chunk.embedding
        ]
        if not rows and node.embedding:
            rows = [list(node.embedding)]
        if not rows:
            return None
        width = len(rows[0])
        kept = [row for row in rows if len(row) == width]
        mean = [sum(row[index] for row in kept) / len(kept) for index in range(width)]
        return _normalize_vector(mean)

    def _ctfidf_label(self, bucket: list[_Member]) -> str:
        terms = self._ranked_terms(bucket)
        if not terms:
            return _phrase_from_content(bucket[0].node.content)
        return " ".join(terms[:CTFIDF_LABEL_TERMS])

    def _ranked_terms(self, bucket: list[_Member]) -> list[str]:
        """This cluster's terms, most distinguishing first.

        ``tf * log(N / (1 + df))``: c-TF-IDF, with the cluster standing in for
        the document. Document frequencies come from the FTS vocab index, so
        rarity is measured against exactly the corpus recall searches.

        The unsmoothed numerator is what makes the label readable, and it is
        worth naming why. Under ``log(1 + N / (1 + df))`` a term present in
        *every* document still scores 0.69, which three occurrences inside the
        cluster are more than enough to turn into a winning term: a corpus of
        memory-server notes gets labelled "memory server". Plain ``log(N / (1 +
        df))`` sends a universal term negative, so it cannot win however often
        the cluster repeats it, while a term the cluster shares with three
        documents out of eleven still scores 1.0. A store predating the vocab
        index, or one with no indexed documents, degrades to plain term
        frequency rather than to nothing.
        """

        counted: Counter[str] = Counter()
        for member in bucket:
            counted.update(_terms(member.node.content))
        if not counted:
            return []

        candidates = [
            term
            for term, _count in sorted(counted.items(), key=lambda item: (-item[1], item[0]))
        ][:CTFIDF_TERM_BUDGET]

        frequencies = self._document_frequencies(candidates)
        total = self._document_count()

        scored: list[tuple[float, str]] = []
        for term in candidates:
            weight = float(counted[term])
            if total > 1 and frequencies is not None:
                weight *= log(total / (1.0 + float(frequencies.get(term, 0))))
            scored.append((weight, term))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [term for _weight, term in scored]

    def _enrich(self, head: str, bucket: list[_Member]) -> tuple[str, str]:
        """A multi-word label for a bucket that only knows one word for itself.

        Stages 1 and 2 name a cluster after a thing the corpus stores — a
        structural value, a directory segment — and the field measurement that
        motivates this is what happens when that thing is a house word: 90.8 %
        of delivered clusters in the live window were stage-2 path collapses,
        and they read as ``public (73)``, ``test (37)``, ``mmo (8)``. A row
        like that is not an invitation, it is a word.

        So the head keeps its place and the cluster's own most distinguishing
        terms are appended to it, taken from :meth:`_ranked_terms` — the
        existing c-TF-IDF ranker, reused rather than reimplemented, so a path
        label and an embedding label are distinguishing by the same measure
        against the same index. Enrichment is what *rescues* those clusters
        from the gate rather than leaving it to withhold them: ``public``
        scores 1.58 alone, and two bucket terms seen in 100 documents each add
        3.24 apiece.

        The returned ask-hint is **not** the label. It is the head plus at most
        enough terms to reach :data:`ASK_HINT_MAX_TOKENS`, so a head that is
        already two words is handed back untouched — see that constant for why
        widening it further would retune the curtail rule as a side effect.
        """

        head = _collapse(head)
        head_tokens = head.split()
        taken = set(head_tokens)
        extra: list[str] = []
        for term in self._ranked_terms(bucket):
            if term in taken:
                continue
            extra.append(term)
            taken.add(term)
            if len(extra) >= CTFIDF_LABEL_TERMS - 1:
                break
        label = _shorten(" ".join([head, *extra]), MAX_LABEL_CHARS)
        room = ASK_HINT_MAX_TOKENS - len(head_tokens)
        # A head that normalized away to nothing is dropped rather than joined
        # into a leading space; ``_cluster_of`` has the last fallback for a
        # group that ends up with no name at all.
        hint = " ".join(part for part in [head, *extra[:room]] if part) if room > 0 else head
        return label, hint

    # -- the delivery gate ---------------------------------------------

    def _reset_corpus_memo(self) -> None:
        """Forget the corpus statistics of the previous build.

        Per :meth:`build`, not per builder: the memo exists to keep one build
        from asking the same index the same question once per bucket, and a
        memo that outlived the call would answer the *next* recall out of a
        corpus that has since moved.
        """

        self._fts_total = None
        self._term_df = {}
        self._df_available = True

    def _document_count(self) -> int:
        """Indexed documents (c-TF-IDF ``N``), asked at most once per build."""

        if self._fts_total is None:
            try:
                self._fts_total = int(self.store.fts_document_count())
            except (AttributeError, sqlite3.OperationalError):  # pragma: no cover - old store
                self._fts_total = 0
        return self._fts_total

    def _document_frequencies(self, terms: Sequence[str]) -> dict[str, int] | None:
        """``term -> documents containing it``, or ``None`` if unanswerable.

        ``None`` rather than an empty mapping, because the two mean opposite
        things to a rarity measure: a term the index has never seen is
        maximally rare, while an index that cannot be read says nothing about
        rarity at all. Collapsing them would make a store with no vocabulary
        table look like a store where every label is distinguishing.
        """

        wanted = [term for term in dict.fromkeys(terms) if term not in self._term_df]
        if wanted and self._df_available:
            try:
                fetched = self.store.term_document_frequencies(wanted)
            except (AttributeError, sqlite3.OperationalError):  # pragma: no cover - old store
                self._df_available = False
            else:
                for term in wanted:
                    self._term_df[term] = int(fetched.get(term, 0))
        if not self._df_available:
            return None
        return {term: self._term_df.get(term, 0) for term in terms}

    def _label_score(self, label: str) -> float | None:
        """``S3``: the information content of ``label``'s best terms, in nats.

        ``None`` when the corpus cannot answer — see :meth:`_deliverable` for
        what happens then. Terms are taken unstemmed through :func:`_terms`,
        because the document-frequency index is ``unicode61`` and does not
        stem: a stemmed term reports ``df = 0``, scores as maximally rare, and
        turns the gate inside out.
        """

        terms = list(dict.fromkeys(_terms(label)))
        if not terms:
            return 0.0
        total = self._document_count()
        frequencies = self._document_frequencies(terms)
        if total <= 1 or frequencies is None:
            return None
        scale = float(total + 1)
        scores = sorted(
            (log(scale / (1.0 + float(frequencies.get(term, 0)))) for term in terms),
            reverse=True,
        )
        return sum(scores[:LABEL_GATE_TOP_TERMS])

    def _deliverable(self, label: str) -> bool:
        """Whether a cluster carrying ``label`` may be delivered at all.

        The whole rule, and the intent behind it restated: a cluster the
        cascade could not give a distinguishing name is withheld rather than
        shipped, because silence is cheaper than a row the agent cannot act on
        — and after the response budget has taken its share the map has only a
        handful of rows to spend, so a wasted one costs a large fraction of the
        channel.

        Withheld means *removed from delivery*, never relabelled to a catch-all
        and never appended to ``clusters`` with a marker: anything inside
        ``clusters`` is a delivered item to :func:`_delivered_items` and a
        novelty-consuming hint to the ae probe, so a "noise" row would suppress
        the curtail rule on a cluster nobody could ever follow. It goes to the
        journal (:class:`_FilterJournal`) and nowhere else.

        When the corpus cannot be measured — an old schema, an index with
        nothing in it — the gate degrades to the purely structural half of the
        predicate, ``two distinct content terms``. A store with no corpus
        statistics is a store where "generic" is undefined, and a gate that
        withholds everything on a missing index is worse than one that admits a
        weak label.
        """

        score = self._label_score(label)
        if score is None:
            return len(set(_terms(label))) >= 2
        return score >= LABEL_GATE_MIN_IC

    # -- curtail -------------------------------------------------------

    def _reoffer(
        self, verdict: _Curtailment, digest: Callable[[], str]
    ) -> tuple[bool, str | None]:
        """Whether a build under ``verdict`` offers a map, and what it persists.

        ``(True, None)`` is an open key: a map, and nothing to persist.
        ``(False, None)`` is the frozen collapse — the valve unset, or an
        armed key inside its skip run — no pool construction or accounting.
        The other two shapes are the valve's. ``(False, digest)`` is a marker
        that carries the pool's digest: the first marker after an offer,
        which records the baseline, or a served skip run whose rebuilt pool
        matched it, which starts the run over instead of re-offering — not an
        offer, so the offer count and the run's width do not move. ``(True,
        digest)`` is a served skip run whose pool moved, or one with no
        baseline to compare against (a window the frozen code wrote): build
        the full map, and should it come back with nothing deliverable in it,
        persist the digest on the marker as the new baseline.

        ``digest`` constructs the pool lazily, then reads its anchors and
        assignments. Only an initial baseline or a due retry invokes
        it. The valve is read here and nowhere else on the collapse path.
        """

        if not verdict.collapse:
            return True, None
        if not curtail_decay_from_env():
            return False, None
        if not verdict.retry_due:
            return False, digest() if verdict.lead == 0 else None
        current = digest()
        if verdict.pool_digest is not None and current == verdict.pool_digest:
            return False, current
        return True, current

    def _collapsed(
        self,
        scope: str,
        key: str,
        *,
        pool_size: int | None,
        streak: int,
        selection: SelectionAccounting | None,
        pool_digest: str | None,
    ) -> RecallMap:
        """The collapsed form: shape kept, content dropped, streak carried."""

        return RecallMap(
            scope=scope,
            key=key,
            clusters=(),
            pool_size=pool_size,
            covered=0,
            curtailed=True,
            streak=streak,
            selection=selection,
            selection_sample_limit=0 if selection is None else _selection_sample_limit(
                (),
                pool_size,
                0,
                selection=selection,
                curtailed=True,
                streak=streak,
                pool_digest=pool_digest,
            ),
            pool_digest=pool_digest,
        )

    def _pool_digest(
        self,
        pool: Sequence[_Member],
        *,
        assignment: Mapping[str, str],
    ) -> str:
        """Content-free identity of qualified members and their anchor groups."""

        members = sorted(member.node.id for member in pool)
        anchors = [list(pair) for pair in sorted(assignment.items())]
        canonical = json.dumps([members, anchors], separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[
            :CURTAIL_POOL_DIGEST_CHARS
        ]

    def _curtailment(
        self,
        work: RecallWork | None,
    ) -> _Curtailment:
        """Read this work's non-use streak only when a collapse is possible.

        The memo can delay a collapse but never authorize it. Every collapsed
        call reads fresh persisted history so its own follow evidence can reopen
        the channel. Unidentified calls have no shared state."""

        if work is None:
            self.last_curtail_read = False
            return _Curtailment(streak=0, offers=0)
        key = work.key
        memo = self._curtail_memo.get(key)
        if (
            memo is not None
            and not memo.estimate().collapse
        ):
            self.last_curtail_read = False
            return memo.estimate()

        self.last_curtail_read = True
        verdict = self._read_curtailment(work)
        self._curtail_memo.pop(key, None)
        self._curtail_memo[key] = _CurtailMemo(verdict=verdict)
        while len(self._curtail_memo) > MAX_CACHE_ENTRIES:
            self._curtail_memo.pop(next(iter(self._curtail_memo)))
        return verdict

    def _note_delivery(
        self,
        work: RecallWork | None,
        *,
        offered: bool,
    ) -> None:
        """Count one map this builder just handed the server to deliver.

        ``offered`` separates a map from a collapsed marker, because only a
        map can go unaccepted. Counted here rather than inferred at read time
        so that the gate above stays a pure function of what this process
        knows it did.
        """

        memo = self._curtail_memo.get(work.key) if work else None
        if memo is None:  # pragma: no cover - the read always seeds one
            return
        memo.total += 1
        if offered:
            memo.offers += 1

    def _read_curtailment(
        self,
        work: RecallWork | None,
    ) -> _Curtailment:
        """Walk this work's map deliveries back to the first followed offer.

        Own recalls, including mapless ones, can echo an ask or return a medoid.
        Lookups are attributed by the most recent work on their transport.
        Existing delivery windows bound both probes. Missing or truncated
        observation cannot establish non-use."""

        try:
            history = self._delivery_history(work)
        except (AttributeError, sqlite3.Error):  # pragma: no cover - old store
            return _Curtailment(streak=0, offers=0)
        if not history:
            return _Curtailment(streak=0, offers=0)

        items = [_delivered_items(row.get("recall_map")) for row in history]
        assert work is not None
        try:
            follows = self.store.recall_work_followups(
                work, since=min(str(row["created_at"]) for row in history),
                medoid_ids=sorted({item.medoid_id for delivered in items
                                   for item in delivered if item.medoid_id}),
            )
        except (AttributeError, sqlite3.Error):
            return _Curtailment(streak=0, offers=0)
        if follows is None:
            return _Curtailment(streak=0, offers=0)
        for follow in follows:
            follow["tokens"] = token_set(str(follow["query"] or ""))
        windows = {
            str(row["delivery_event_id"]): (str(row["delivered_at"]), str(row["outcome_end"]))
            for row in self.store.connection.execute(
                f"SELECT delivery_event_id, delivered_at, outcome_end "
                f"FROM {RECALL_DELIVERY_HISTORY_TABLE} "
                f"WHERE delivery_event_id IN ({','.join('?' for _ in history)})",
                [row["id"] for row in history],
            )
        }

        streak = 0
        offers = 0
        lead = 0
        pool_digest: str | None = None
        since_digest = 0
        for index, row in enumerate(history):
            delivered = items[index]
            window = windows.get(str(row["id"]))
            if delivered and window is None:
                return _Curtailment(streak=0, offers=0)
            if delivered and any(
                follow["event_id"] != row["id"]
                and follow["at"] is not None
                and window[0] <= follow["at"] <= window[1]
                and any(
                    item.medoid_id in follow["node_ids"]
                    or _echoes(item.label, follow["tokens"])
                    or _echoes(item.ask_hint, follow["tokens"])
                    for item in delivered
                )
                for follow in follows
            ):
                break
            streak += 1
            if delivered:
                offers += 1
            elif not offers:
                lead += 1
                if pool_digest is None:
                    # The newest lead row carrying a digest is the baseline;
                    # the rows above it are the run served since. Parsed off
                    # the payload already in hand, and consulted by nothing
                    # on the frozen path.
                    payload = row.get("recall_map")
                    digest = (
                        payload.get("pd") if isinstance(payload, Mapping) else None
                    )
                    if isinstance(digest, str) and digest:
                        pool_digest = digest
                        since_digest = lead - 1
        return _Curtailment(
            streak=streak,
            offers=offers,
            lead=lead,
            pool_digest=pool_digest,
            since_digest=since_digest,
        )

    def _delivery_history(self, work: RecallWork | None) -> list[dict[str, Any]]:
        """Maps offered to this exact consumer, newest first."""

        if work is None:
            return []
        return [
            row for row in self.store.recent_recall_map_history(
                work=work, limit=CURTAIL_HISTORY_LIMIT,
            )
            if isinstance(row.get("recall_map"), Mapping)
        ]

    # -- cache ---------------------------------------------------------

    def _corpus_revision(self) -> tuple[Any, ...]:
        """Identity of everything the cached *structure* was derived from.

        A ``(total_changes, PRAGMA data_version)`` probe answers "has anything
        been written at all", and only when it moves does the second tier ask
        the chunk and anchor tables themselves. Most recalls move the probe
        with their own bookkeeping — ``record_access`` writes on every call —
        and move neither input. Missing anchor revisions and uncommitted
        inputs use the write probe instead of a reusable anchor identity.
        """

        connection = self.store.connection
        probe = (int(connection.total_changes), _data_version(connection))
        cached = self._revision
        if (
            cached is not None
            and probe == self._write_probe
        ):
            return cached
        anchor_revision = self.store.query_anchor_revision()
        revision = (
            _chunk_table_revision(connection),
            anchor_revision
            if anchor_revision is not None and not connection.in_transaction
            else probe,
        )
        self._write_probe = probe
        self._revision = revision
        return revision

    def _remember(
        self, key: str, revision: tuple[Any, ...], groups: list[_Group]
    ) -> None:
        if not key:
            return
        templates = tuple(
            _ClusterTemplate(
                stage=group.stage,
                signature=group.signature,
                label=group.label,
                ask_hint=group.ask_hint,
                member_ids=frozenset(member.node.id for member in group.members),
            )
            for group in groups
        )
        self._cache.pop(key, None)
        self._cache[key] = _CachedStructure(revision=revision, templates=templates)
        while len(self._cache) > MAX_CACHE_ENTRIES:
            self._cache.pop(next(iter(self._cache)))

    def _recount(
        self, templates: tuple[_ClusterTemplate, ...], pool: list[_Member]
    ) -> list[_Group] | None:
        """Refit a cached structure to the pool in hand, or refuse to.

        Stages 1 and 2 re-derive membership exactly — their signatures are pure
        functions of a node's context — so a node the cached pool never saw
        still lands in the right cluster. Stages 3 and 4 match by remembered
        membership: an anchor lookup is cheap but a *re-clustering* is not, and
        a cached embedding cluster has no rule to offer beyond the ids it was
        built from.

        That asymmetry is what :data:`CACHE_MIN_COVERAGE` guards. When the pool
        has moved far enough that the cached labels no longer describe most of
        it, this returns ``None`` and the caller rebuilds — stability is worth
        having only while it is still stability *about this pool*.
        """

        if not templates:
            return None
        ordered = sorted(
            enumerate(templates),
            key=lambda item: (_STAGE_ORDER.get(item[1].stage, 99), item[0]),
        )
        buckets: dict[int, list[_Member]] = {}
        covered = 0
        for member in pool:
            matched = False
            for index, template in ordered:
                if not self._matches(template, member.node):
                    continue
                buckets.setdefault(index, []).append(member)
                covered += 1
                matched = True
                break
            if not matched:
                # A fresh cascade may have a new structural/path/anchor
                # signature or seed a new embedding cluster. A template cache
                # has no authority to silently omit that member.
                return None
        if covered < CACHE_MIN_COVERAGE * len(pool):
            return None

        # Stage 4 is a retrieval-ordered greedy walk. Re-run that membership
        # rule from fresh vectors and fresh scores, but retain cached labels
        # only if it reproduces the current template buckets exactly. This
        # keeps the cache template-only: no score, low-relevance verdict,
        # relevance order, or medoid identity survives a build boundary.
        embedding_indexes = [
            index
            for index, template in ordered
            if template.stage == STAGE_EMBEDDING and index in buckets
        ]
        embedding_vectors: dict[frozenset[str], dict[str, list[float]]] = {}
        if embedding_indexes:
            embedding_ids = {
                member.node.id
                for index in embedding_indexes
                for member in buckets[index]
            }
            embedding_members = [
                member for member in pool if member.node.id in embedding_ids
            ]
            fresh, rest = self._stage_embedding(embedding_members)
            if rest:
                return None
            current_sets = {
                frozenset(member.node.id for member in buckets[index])
                for index in embedding_indexes
            }
            fresh_sets = {
                frozenset(member.node.id for member in group.members) for group in fresh
            }
            if fresh_sets != current_sets or len(fresh) != len(embedding_indexes):
                return None
            embedding_vectors = {
                frozenset(member.node.id for member in group.members): group.vectors or {}
                for group in fresh
            }

        groups: list[_Group] = []
        for index, template in ordered:
            members = buckets.get(index)
            if not members:
                continue
            vectors = None
            if template.stage == STAGE_EMBEDDING:
                vectors = embedding_vectors.get(
                    frozenset(member.node.id for member in members)
                )
                if not vectors:
                    return None
            groups.append(
                _Group(
                    stage=template.stage,
                    signature=template.signature,
                    label=template.label,
                    ask_hint=template.ask_hint,
                    members=members,
                    vectors=vectors,
                )
            )
        return groups

    @staticmethod
    def _matches(template: _ClusterTemplate, node: Node) -> bool:
        if template.stage == STAGE_STRUCTURAL:
            key = _structural_key(node)
            return key is not None and (STAGE_STRUCTURAL, *key) == template.signature
        if template.stage == STAGE_PATH:
            # On a cold build stage 2 only sees the residual stage 1 declined.
            # Re-establish that precondition on a cache hit; otherwise a path
            # template can rename a node carrying a new structural value after
            # its directory merely because that value had no cached template.
            if _structural_key(node) is not None:
                return False
            subsystem = _node_subsystem(node)
            return subsystem is not None and (STAGE_PATH, subsystem) == template.signature
        return node.id in template.member_ids

    # -- assembly ------------------------------------------------------

    def _finish(
        self,
        groups: list[_Group],
        *,
        scope: str,
        key: str,
        pool_size: int,
        selection: SelectionAccounting,
    ) -> RecallMap | None:
        """Gate, order, cap and budget the groups into a map.

        The gate runs first and runs here rather than inside :meth:`_cluster`,
        for two reasons. It must see the label ``_disambiguate`` settled on —
        disambiguation can *lengthen* a label with a distinguishing term, which
        can only raise its score, so gating earlier would withhold clusters
        that were about to be rescued. And it must run on the cached path too:
        the structure cache stores every group it built, gate verdict included,
        because the verdict is a corpus statistic and the corpus moves under a
        cache that is deliberately validated against something else.

        Surviving topics follow their best current retrieval ordinal. Cached
        templates contain no ordering or representative decisions; both refresh
        from this call's members.

        A pool whose every cluster is withheld yields a journal-only map.  It
        carries no delivered clusters (and therefore counts as no curtail
        offer), but it must still reach ``recall_events.recall_map``: otherwise
        the field cannot distinguish "the residual was empty" from "the label
        gate rejected everything", exactly the ambiguity the journal exists
        to remove.
        """

        populated = [group for group in groups if group.members]
        if not populated:
            return RecallMap(
                scope=scope,
                key=key,
                clusters=(),
                pool_size=pool_size,
                covered=0,
                selection=selection,
                selection_sample_limit=_selection_sample_limit(
                    (), pool_size, 0, selection=selection
                ),
            )

        # One index round trip for every label on the table, rather than one
        # per label. ``term_document_frequencies`` pays a scratch-table fold
        # per call and then seeks per token, so the batch is nearly free where
        # the calls are not — and this is the only new cost the gate puts on
        # the *cached* path, where the cascade itself no longer runs.
        self._document_frequencies(
            [term for group in populated for term in _terms(group.label)]
        )

        journal = _FilterJournal()
        deliverable: list[_Group] = []
        for group in populated:
            if self._deliverable(group.label):
                deliverable.append(group)
            else:
                journal.withheld.append(_filtered_of(FILTER_TAG_WITHHELD, group))
        journal.withheld.sort(key=lambda entry: (-entry.count, entry.label))
        if not deliverable:
            filtered = journal.block()
            return RecallMap(
                scope=scope,
                key=key,
                clusters=(),
                pool_size=pool_size,
                covered=0,
                withheld=len(journal.withheld),
                filtered=journal.names(),
                filtered_omitted=journal.omitted(),
                selection=selection,
                selection_sample_limit=_selection_sample_limit(
                    (), pool_size, 0, filtered, selection=selection
                ),
            )

        # Each topic gets one cluster. Its best retrieval ordinal orders the
        # menu without rewarding the number or exposure history of its members.
        deliverable.sort(key=lambda group: min(member.rank for member in group.members))
        kept = deliverable[: self.max_clusters]
        journal.dropped.extend(
            _filtered_of(FILTER_TAG_DROPPED, group)
            for group in deliverable[self.max_clusters :]
        )
        clusters, sample_limit = self._fit(
            kept, pool_size, journal, selection=selection
        )
        if not clusters:
            # `_fit` may decide that even the breadth floor cannot carry an
            # informative gist.  It records every remaining group as dropped;
            # persist that decision instead of turning a budget refusal into
            # the same `None` an empty residual returns.
            return RecallMap(
                scope=scope,
                key=key,
                clusters=(),
                pool_size=pool_size,
                covered=0,
                dropped=len(journal.dropped),
                withheld=len(journal.withheld),
                filtered=journal.names(),
                filtered_omitted=journal.omitted(),
                selection=selection,
                selection_sample_limit=_selection_sample_limit(
                    (),
                    pool_size,
                    len(journal.dropped),
                    journal.block(),
                    selection=selection,
                ),
            )
        return RecallMap(
            scope=scope,
            key=key,
            clusters=tuple(clusters),
            pool_size=pool_size,
            covered=sum(cluster.count for cluster in clusters),
            dropped=len(journal.dropped),
            withheld=len(journal.withheld),
            filtered=journal.names(),
            filtered_omitted=journal.omitted(),
            selection=selection,
            selection_sample_limit=sample_limit,
        )

    def _fit(
        self,
        kept: list[_Group],
        pool_size: int,
        journal: _FilterJournal,
        *,
        selection: SelectionAccounting,
    ) -> tuple[list[MapCluster], int]:
        """Fit the widest common gist, trading tail breadth before gist quality.

        Every pass searches the full ``[0, MEDOID_EXAMPLE_CHARS]`` interval
        against the *current* payload.  That restart is load-bearing: the old
        shave loop could reach zero, drop several clusters, and then return the
        zero-width examples it had inherited even though the smaller map left
        roughly two hundred characters unused.

        A gist below :data:`MIN_MEDOID_EXAMPLE_CHARS` first buys room by
        dropping the tail cluster, down to :data:`MIN_CLUSTERS`.  Each such
        trade is journaled before the next search, so both the newly affordable
        example space and the journal's own cost are remeasured.  At the
        breadth floor, journal names give way one at a time; the unconditional
        withheld/dropped counts and ``names_omitted`` never do.  Only after no
        name remains may the widest affordable gist fall below the floor. The
        mandatory ``sel`` core participates in every measurement. Optional
        ``q`` samples are added only after the existing cluster shape fits, so
        they are always the first selection detail sacrificed to the budget.
        """

        active_groups = list(kept)
        # Medoid choice does not depend on the example width.  Resolve it once
        # per group — especially important for stage 4, whose true medoid is a
        # pairwise-vector calculation — and make the binary search below a
        # cheap rewrite of the quote only.
        full = [
            self._cluster_of(group, example_chars=MEDOID_EXAMPLE_CHARS)
            for group in active_groups
        ]
        breadth_floor = min(MIN_CLUSTERS, len(full))

        def at_width(example_chars: int) -> list[MapCluster]:
            return [
                replace(
                    cluster,
                    medoid=replace(
                        cluster.medoid,
                        example=_shorten(cluster.medoid.example, example_chars),
                    ),
                )
                for cluster in full
            ]

        def floor_met(clusters: Sequence[MapCluster]) -> bool:
            return all(
                bool(cluster.medoid.example)
                and len(cluster.medoid.example)
                >= min(MIN_MEDOID_EXAMPLE_CHARS, len(source.medoid.example))
                for cluster, source in zip(clusters, full, strict=True)
            )

        def widest() -> tuple[int, list[MapCluster]]:
            """Largest shared example cap whose measured payload fits."""

            low = 0
            high = MEDOID_EXAMPLE_CHARS
            best_chars = -1
            best: list[MapCluster] = []
            while low <= high:
                example_chars = (low + high) // 2
                candidates = at_width(example_chars)
                if (
                    _payload_size(
                        candidates,
                        pool_size,
                        len(journal.dropped),
                        journal.block(),
                        selection=selection,
                        selection_sample_limit=0,
                    )
                    <= MAX_RESPONSE_CHARS
                ):
                    best_chars = example_chars
                    best = candidates
                    low = example_chars + 1
                else:
                    high = example_chars - 1
            return best_chars, best

        while full:
            example_chars, clusters = widest()
            if example_chars >= 0 and floor_met(clusters):
                return clusters, _selection_sample_limit(
                    clusters,
                    pool_size,
                    len(journal.dropped),
                    journal.block(),
                    selection=selection,
                )

            if len(full) > breadth_floor:
                journal.dropped.insert(
                    0, _filtered_of(FILTER_TAG_DROPPED, active_groups.pop())
                )
                full.pop()
                continue

            if journal.drop_name():
                continue

            # Long pre-existing anchor/structural hints can make even two
            # clusters cost more than the ordinary worst-case model.  A gist
            # below the floor is not an invitation, so fail closed: move the
            # surviving breadth into the dropped journal and let `_finish`
            # persist a journal-only map. With no delivered rows competing for
            # the budget, restore the normal forensic name allowance.
            journal.dropped[0:0] = [
                _filtered_of(FILTER_TAG_DROPPED, group) for group in active_groups
            ]
            journal.name_limit = FILTER_JOURNAL_NAMES
            return [], 0
        return [], 0

    def _cluster_of(self, group: _Group, *, example_chars: int) -> MapCluster:
        medoid = self._medoid(group)
        label = _shorten(group.label, MAX_LABEL_CHARS) or _shorten(
            _phrase_from_content(medoid.node.content), MAX_LABEL_CHARS
        )
        ask_hint = _shorten(group.ask_hint, MAX_ASK_HINT_CHARS) or label
        count = len(group.members)
        return MapCluster(
            label=label,
            count=count,
            medoid=MapMedoid(
                node_id=medoid.node.id,
                example=_shorten(medoid.node.content, example_chars),
            ),
            ask_hint=ask_hint,
            plan_item=f"on touching {label} - recall '{ask_hint}' ({count})",
            stage=group.stage,
            member_ids=tuple(member.node.id for member in group.members),
        )

    @staticmethod
    def _medoid(group: _Group) -> _Member:
        """The member that speaks for the cluster.

        With vectors in play — stage 4 only, the one stage that has them — use
        the member of maximum mean similarity to the rest, which is the medoid
        proper. Everywhere else use the best original residual ordinal. That
        ordinal is unique, so it closes every tie without an identity fallback.
        Cached embedding groups carry freshly read vectors and therefore make
        the same choice as a cold group.
        """

        vectors = group.vectors
        if not vectors or len(group.members) < 2:
            return min(group.members, key=lambda member: member.rank)
        scored: list[tuple[float, int, _Member]] = []
        for member in group.members:
            own = vectors.get(member.node.id)
            if own is None:  # pragma: no cover - members always carry a vector
                continue
            others = [
                cosine_similarity(own, vectors[other.node.id])
                for other in group.members
                if other.node.id != member.node.id and other.node.id in vectors
            ]
            mean = sum(others) / len(others) if others else 0.0
            scored.append((-mean, member.rank, member))
        if not scored:  # pragma: no cover - defensive
            return min(group.members, key=lambda member: member.rank)
        return min(scored, key=lambda item: item[:2])[2]


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def cache_key(
    scope: str, *, task: str | None = None, task_pattern: str | None = None,
    transport_session_id: str | None = None,
) -> str:
    """Current consumer's exact key, or empty when context is unavailable."""

    work = RecallWork.resolve(
        scope, task=task, task_pattern=task_pattern,
        transport_session_id=transport_session_id,
    )
    return work.key if work else ""


def _dominant_scope(pool: list[_Member]) -> str:
    counted: Counter[str] = Counter()
    for member in pool:
        scope = getattr(member.node, "scope", "") or GLOBAL_SCOPE
        counted[str(scope)] += 1
    if not counted:
        return GLOBAL_SCOPE
    return min(counted.items(), key=lambda item: (-item[1], item[0]))[0]


def _dominant_residual_scope(results: Sequence[Any]) -> str:
    """Fallback scope when a non-empty residual admits no members."""

    counted: Counter[str] = Counter()
    for result in results:
        node = getattr(result, "node", None)
        if node is None:
            continue
        scope = getattr(node, "scope", "") or GLOBAL_SCOPE
        counted[str(scope)] += 1
    if not counted:
        return GLOBAL_SCOPE
    return min(counted.items(), key=lambda item: (-item[1], item[0]))[0]


def _filtered_of(tag: str, group: _Group) -> FilteredCluster:
    """One journal entry for a group that will not be delivered."""

    return FilteredCluster(
        tag=tag,
        label=_shorten(group.label, FILTER_JOURNAL_LABEL_CHARS),
        count=len(group.members),
    )


def _filter_block(
    withheld: int,
    dropped: int,
    names: Sequence[FilteredCluster],
    omitted: int,
) -> dict[str, Any] | None:
    """The ``filtered`` payload key, or ``None`` when nothing was filtered.

    Strictly additive, and emitted only when it has something to say, so an
    unfiltered map is byte-identical to what this module shipped before the
    journal existed. Both counts are unconditional whenever the block is
    present — that is the no-silent-caps rule in its exact form: the *names*
    are convenience and may be taken by the budget, the *numbers* never are, so
    a field reader can always tell "memory had nothing to say" (``withheld``
    high) from "the channel was full" (``dropped`` high) from "the filter is
    too harsh" (both, against a large pool).

    ``dropped`` deliberately mirrors the top-level ``more``. The duplication
    costs about fourteen characters and buys a block that is self-contained
    and greppable in a persisted payload.
    """

    if withheld <= 0 and dropped <= 0:
        return None
    block: dict[str, Any] = {"withheld": withheld, "dropped": dropped}
    if names:
        block["names"] = [entry.to_list() for entry in names]
    if omitted > 0:
        block["names_omitted"] = omitted
    return block


def _payload(
    clusters: Sequence[MapCluster],
    pool_size: int | None,
    dropped: int,
    *,
    curtailed: bool = False,
    streak: int = 0,
    filtered: Mapping[str, Any] | None = None,
    selection: SelectionAccounting | None = None,
    selection_sample_limit: int = SELECTION_SAMPLE_LIMIT,
    pool_digest: str | None = None,
) -> dict[str, Any]:
    """The wire form. Curtailed maps keep the shape and drop the content.

    ``clusters`` stays present and empty rather than being omitted, so every
    consumer of a persisted map — the instructions channel, the effect gate,
    this module's own :meth:`RecallMapBuilder._curtailment` — reads one shape
    and reaches the collapse through ``curtailed`` instead of through a
    ``KeyError``. Missing ``pool`` and ``sel`` mean the build skipped that
    work, not that the corpus or candidate selection was empty.

    ``filtered`` is a *sibling* key and never an entry inside ``clusters``.
    That is the hard rule the additive shape rests on: ``_delivered_items`` is
    paranoid about the shape of a cluster but not about extra ones, so a
    journal row inside the list would be read as a delivered item by the
    curtail probe and as a novelty-consuming hint by the ae probe — corrupting
    both on a cluster nobody was ever offered.
    """

    payload: dict[str, Any] = {
        "clusters": [cluster.to_dict() for cluster in clusters],
    }
    if pool_size is not None:
        payload["pool"] = pool_size
    payload["covered"] = sum(cluster.count for cluster in clusters)
    if dropped > 0:
        payload["more"] = dropped
    if filtered:
        payload["filtered"] = dict(filtered)
    if selection is not None:
        payload["sel"] = selection.to_dict(sample_limit=selection_sample_limit)
    if curtailed:
        payload["curtailed"] = True
        payload["streak"] = streak
    if pool_digest is not None:
        # The decay valve's baseline, on the marker it was writing anyway. A
        # sibling key like ``filtered``: never inside ``clusters``, and read
        # back only by ``_read_curtailment``, which walks the same rows.
        payload["pd"] = pool_digest
    return payload


def _selection_sample_limit(
    clusters: Sequence[MapCluster],
    pool_size: int | None,
    dropped: int,
    filtered: Mapping[str, Any] | None = None,
    *,
    selection: SelectionAccounting,
    curtailed: bool = False,
    streak: int = 0,
    pool_digest: str | None = None,
) -> int:
    """Most ``q`` examples that fit without changing existing semantics."""

    maximum = min(SELECTION_SAMPLE_LIMIT, len(selection.samples))
    for sample_limit in range(maximum, -1, -1):
        payload = _payload(
            clusters,
            pool_size,
            dropped,
            curtailed=curtailed,
            streak=streak,
            filtered=filtered,
            selection=selection,
            selection_sample_limit=sample_limit,
            pool_digest=pool_digest,
        )
        if len(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        ) <= MAX_RESPONSE_CHARS:
            return sample_limit
    return 0


def _payload_size(
    clusters: Sequence[MapCluster],
    pool_size: int | None,
    dropped: int,
    filtered: Mapping[str, Any] | None = None,
    *,
    selection: SelectionAccounting | None = None,
    selection_sample_limit: int = SELECTION_SAMPLE_LIMIT,
) -> int:
    return len(
        json.dumps(
            _payload(
                clusters,
                pool_size,
                dropped,
                filtered=filtered,
                selection=selection,
                selection_sample_limit=selection_sample_limit,
                ),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


__all__ = [
    "ASK_HINT_MAX_TOKENS",
    "CACHE_MIN_COVERAGE",
    "CTFIDF_LABEL_TERMS",
    "CURTAIL_DECAY_GATE_ENV",
    "CURTAIL_DECAY_SKIP_MAX",
    "CURTAIL_HISTORY_LIMIT",
    "CURTAIL_POOL_DIGEST_CHARS",
    "CURTAIL_QUERY_OVERLAP",
    "CURTAIL_STREAK",
    "EMBEDDING_CLUSTER_COSINE",
    "FILTER_JOURNAL_LABEL_CHARS",
    "FILTER_JOURNAL_NAMES",
    "FILTER_TAG_DROPPED",
    "FILTER_TAG_WITHHELD",
    "LABEL_GATE_MIN_IC",
    "LABEL_GATE_TOP_TERMS",
    "MAX_ASK_HINT_CHARS",
    "MAX_CLUSTERS",
    "MAX_INSTRUCTIONS_CHARS",
    "MAX_LABEL_CHARS",
    "MAX_POOL_NODES",
    "MAX_RESPONSE_CHARS",
    "MEDOID_EXAMPLE_CHARS",
    "MIN_CLUSTERS",
    "MIN_MEDOID_EXAMPLE_CHARS",
    "POOL_GATE_REASON_CODES",
    "POOL_USEFULNESS_FLOOR_ENV",
    "POOL_USEFULNESS_GATE_ENV",
    "SELECTION_LEDGER_REASON_CODES",
    "SELECTION_REASON_CODES",
    "SELECTION_SAMPLE_GIST_CHARS",
    "SELECTION_SAMPLE_LIMIT",
    "STAGE_ANCHOR",
    "STAGE_EMBEDDING",
    "STAGE_PATH",
    "STAGE_STRUCTURAL",
    "STRUCTURAL_FIELDS",
    "FilteredCluster",
    "MapCluster",
    "MapMedoid",
    "RecallMap",
    "RecallMapBuilder",
    "SelectionAccounting",
    "SelectionSample",
    "cache_key",
    "curtail_decay_from_env",
    "normalize_key",
    "pool_usefulness_floor_from_env",
]
