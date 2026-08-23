"""Recall map: what *else* memory holds for this task, as a handful of clusters.

A recall returns its top results and throws the rest of the ranked pool away.
The residual is not noise — it is the answer to a question the agent has not
asked yet, and the field measurement that motivates this module says the
question never gets asked: over 57 live recalls the median call sat at position
0.02 of the session, i.e. one start-of-session ritual and nothing after. The map
turns that single reliable recall into a *plan* for the mid-session ones: each
cluster carries a label, a count, a medoid example, a phrasing to ask with, and
a todo-plan line the agent can paste into its own plan.

Deterministic by construction
-----------------------------
No LLM call happens on this path and none may be added: the map is built inside
the live recall latency budget, and a model call there would cost more than the
recall it decorates. Labels come from a four-stage cascade, each stage claiming
what it can and handing the rest down:

1. **Structural context keys** — ``procedure_id``, ``lesson_kind``, ``type``,
   ``topic``, ``task_pattern``. The extraction stage already writes these, so
   most of a healthy corpus never reaches stage 2. The label *is* the key,
   normalized the way ``consolidation._normalize_trigger`` normalizes a
   procedure id, except where the raw key is an opaque hash — then the label
   comes from the group's medoid instead.
2. **Path collapse** — nodes carrying file paths become their subsystem, so
   ``src/living_memory/postsession/report.py`` reads as ``postsession`` rather
   than as five near-identical paths.
3. **Query anchors** — an anchor is a *past grounded query* (see
   ``query_anchors``), so an anchor covering a node supplies both the label and,
   verbatim, a proven way to ask for it. This is the only stage whose ask-hint
   is evidence rather than construction.
4. **Chunk-embedding fallback** — greedy leader clustering over the chunk
   vectors already in the store, labelled by c-TF-IDF against the FTS
   document-frequency index (``term_document_frequencies`` /
   ``fts_document_count``): terms frequent inside the cluster and rare in the
   corpus.

Identical input yields a byte-identical map. Every sort carries an explicit tie
break, the greedy pass walks the pool in rank order, and stage precedence is
fixed — a node with both a ``procedure_id`` and a file list is a structural
cluster member, never a path one.

Silence beats a word nobody can act on
--------------------------------------
The label *is* the interface. The first field window of this channel measured
what happens when it degenerates: 27 delivered injections, 0 consumptions, and
labels that had collapsed into single house words — ``public (73)``, ``test
(37)``, ``mmo (8)`` — while a control run of the same code over a corpus whose
labels read ``cleanup commit invariant`` was consumed on its first pass.

Two rules follow, and they work as a pair. Stages 1 and 2 *enrich*: a cluster
named after a structural value or a directory segment gets that name plus its
own most distinguishing terms (:meth:`RecallMapBuilder._enrich`), so the label
says something the corpus does not say everywhere. Then every label, from every
stage, is *gated*: :meth:`RecallMapBuilder._deliverable` withholds a cluster
whose best terms are collectively too common to distinguish anything
(:data:`LABEL_GATE_MIN_IC`). Enrichment is the rescue and the gate is the net —
after enrichment the gate should rarely fire, and when it does it is reporting a
cluster with nothing to say.

The gate is a corpus statistic, never a word list. "Generic" means generic to
the corpus this recall searched, measured against the same document-frequency
index the c-TF-IDF labeller already reads, so the rule transfers to a corpus
whose house vocabulary is different words instead of memorizing the ones a
field window happened to show.

And nothing leaves quietly. Every cluster the gate withholds and every cluster
the response budget drops is counted — and, as far as the budget allows, named
— in the payload's additive ``filtered`` key (:func:`_filter_block`), which
persists into ``recall_events.recall_map``. Without it a field reader cannot
tell "memory had nothing to say" from "the filter was too harsh", which are
opposite findings that both look like a short map.

Stability across sessions
-------------------------
A task that asks twice should see the same shape twice, or the map is furniture
rather than a plan. :class:`RecallMapBuilder` therefore caches the cluster
*structure* per ``(scope, normalized task)`` and revalidates it with the same
cheap monotonic probes ``retrieval`` uses for its chunk and anchor caches —
``(total_changes, PRAGMA data_version)`` in front of the chunk-table and
anchor-table revisions. Counts, medoids and membership refresh against the pool
in hand on every call; labels and ask-hints survive until the corpus moves.
There are no timers: a cache that expires on a clock would hand the same task a
different plan for no reason the agent can see.

Collapse when nobody follows it
-------------------------------
Delivered is not used. The preprompt push already taught this project that a
channel keeps costing its budget long after it stopped changing behaviour, so
the map is built with its own retreat: it measures whether the maps this key
already delivered were *followed*, and when three in a row were not
(:data:`CURTAIL_STREAK`) it stops occupying the channel and ships a marker
instead — ``curtailed: true`` with the zero-consumption streak, ~60 characters
where a map would have spent 700.

Consumption is measured, never assumed, from evidence the server already
writes (:meth:`RecallMapBuilder._curtailment`). A delivered cluster counts as
consumed when either probe fires after the delivery that offered it:

* **the ask was asked** — a later recall under the same key phrased a query
  covering :data:`CURTAIL_QUERY_OVERLAP` of the cluster's label or ask-hint
  tokens, under the tokenizer the grounding rail uses, so "followed the plan
  item" is the same measurement everywhere in this codebase;
* **the example was reached** — the cluster's medoid node was accessed after
  the delivery. ``retrieval`` stamps ``nodes.last_accessed`` on every result it
  delivers, so this fires the moment a later recall actually hands the agent
  the node the map pointed at.

Both probes read tables the recall path already writes — a bounded window over
``recall_events`` via ``MemoryStore.recent_recall_map_history``, and one
primary-key batch over ``nodes``. Nothing new is written and no LLM is asked.

The window read is not free, and pretending otherwise would be the wrong kind
of quiet: no index covers "this key's deliveries that carried a map, newest
first", so SQLite walks the scope's whole partition and sorts — 8.5 ms on a
21k-event scope, measured, and rising with it. It is therefore *gated* rather
than paid per recall: :class:`_CurtailMemo` counts the offers this builder has
made since it last read, and asks for a read only once that count could reach
the threshold. The gate can delay a collapse and can ask for a read it did not
need; it can never collapse a map by itself. What is left after the gate,
isolated by running ``scripts/recall_map_latency_bench.py`` twice over one
snapshot with the probe stubbed out of the second: +1.0 ms on the map stage's
median, which leaves the whole feature's pooled p95 overhead at 0.022 against
its pre-registered budget of 0.20.

The collapse is not a death sentence. Evidence lives in a window
(:data:`CURTAIL_HISTORY_LIMIT`), so the unread maps that silenced a key
eventually slide out from under the markers that replaced them and the key
offers again — 3 deliveries in 25 for a key that is never once followed.

The rule is unconditional. There is no environment flag and no opt-in: a
retreat that has to be switched on is a retreat nobody takes, and the protocol
channels forbid machinery an agent cannot rely on being there. Every ambiguity
in the evidence resolves *against* collapsing — an unreadable history, a
timestamp tie, a key whose deliveries this store cannot see all read as "used"
— because the cost of curtailing a map that was working is larger than the cost
of one more unread map.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from math import fsum, log, log1p, sqrt
from typing import TYPE_CHECKING, Any
import json
import re
import sqlite3

from living_memory.embeddings import cosine_similarity, tokenize
from living_memory.grounding import token_set
from living_memory.models import Node
from living_memory.scope import GLOBAL_SCOPE, normalize_scope
from living_memory.storage import (
    CHUNK_EMBEDDING_TABLE,
    MAX_RECALL_HISTORY_CANDIDATES,
    MemoryStore,
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

#: Clustering cap, applied only after the complete residual has passed frozen
#: eligibility and relevance scoring.  That ordering is load-bearing: applying
#: the cap while scanning would let a machine-ballast head crowd useful members
#: out before the selector saw them.  Only the admitted 200 pay for the heavier
#: four-stage cascade and its per-node anchor/vector reads.
MAX_POOL_NODES = 200

#: Frozen member-selection policy.  These are literals rather than a runtime
#: read of ``artifacts/``: a deployed server must not depend on its source
#: checkout being present, and changing any value is a policy revision rather
#: than configuration.  The digest binds the canonical ``selected_policy``
#: object in ``artifacts/recall-map/relevance/policy.json``.
RELEVANCE_POLICY_ID = "directional-zsum-r1"
RELEVANCE_POLICY_DIGEST = (
    "3acad3d92db2538bf3096ab99d2c4d337ea3c8aebddc226646527b4d277660ea"
)
RELEVANCE_FEATURE_MEANS: tuple[float, ...] = (
    0.26434558349451964,
    2.856678070667311,
    2.789348366105738,
    1.5799800264635717,
    0.06059739660863959,
)
RELEVANCE_FEATURE_SCALES: tuple[float, ...] = (
    0.4409841221421261,
    2.5772848149641323,
    2.5560016163553305,
    1.765967625291225,
    0.11674776461860277,
)
RELEVANCE_THRESHOLD = 3.8708378402511

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
SELECTION_SAMPLE_GIST_CHARS = 24
SELECTION_SAMPLE_LIMIT = 2
_SAMPLEABLE_REASONS = frozenset({"fc", "ss", "sj", "lr", "pc"})

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
        return {
            "label": self.label,
            "count": self.count,
            "medoid": self.medoid.to_dict(),
            "ask_hint": self.ask_hint,
            "plan_item": self.plan_item,
        }


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
    """Complete frozen-policy accounting for one inspected residual.

    The response fitter added by the delivery integration can vary only how
    many of ``samples`` it buys.  ``inspected``, ``admitted`` and ``excluded``
    are unconditional, and ``to_dict`` recomputes ``o`` from the chosen sample
    count so dropping examples can never falsify the accounting equation.
    """

    inspected: int
    admitted: int
    excluded: tuple[int, ...]
    samples: tuple[SelectionSample, ...] = ()
    sampleable: int = 0

    def __post_init__(self) -> None:
        if len(self.excluded) != len(SELECTION_REASON_CODES):
            raise ValueError("selection exclusion vector has the wrong width")
        if self.inspected < 0 or self.admitted < 0 or any(
            count < 0 for count in self.excluded
        ):
            raise ValueError("selection counts must be non-negative")
        if self.inspected != self.admitted + sum(self.excluded):
            raise ValueError("selection accounting does not cover the residual")
        if self.sampleable < len(self.samples):
            raise ValueError("selection samples exceed the sampleable population")
        sampleable_cap = sum(
            self.excluded[index]
            for index, reason in enumerate(SELECTION_REASON_CODES)
            if reason in _SAMPLEABLE_REASONS
        )
        if self.sampleable > sampleable_cap:
            raise ValueError("selection sampleable count exceeds exclusions")

        reason_order = {reason: index for index, reason in enumerate(SELECTION_REASON_CODES)}
        prior = -1
        seen: set[str] = set()
        for sample in self.samples:
            if sample.reason not in _SAMPLEABLE_REASONS:
                raise ValueError("selection sample has a non-sampleable reason")
            current = reason_order[sample.reason]
            if self.excluded[current] <= 0:
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
    #: Members of the residual pool the map considered.
    pool_size: int
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
    #: It travels with every inspected residual so no later layer has to
    #: reconstruct unconditional counts from the capped pool.
    selection: SelectionAccounting | None = None
    #: Optional ``q`` examples that survived the response budget.  The core is
    #: never conditional; only these identity-free examples may be removed.
    selection_sample_limit: int = SELECTION_SAMPLE_LIMIT

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
    relevance_score: float = 0.0


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
        samples = tuple(
            self.first_samples[reason]
            for reason in SELECTION_REASON_CODES
            if reason in self.first_samples
        )
        return SelectionAccounting(
            inspected=self.inspected,
            admitted=admitted,
            excluded=tuple(self.excluded[reason] for reason in SELECTION_REASON_CODES),
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

    @property
    def collapse(self) -> bool:
        return self.offers >= CURTAIL_STREAK


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


def _history_features(history: Any) -> tuple[float, float, float, float]:
    """Frozen evaluator features, or train means when history is unavailable."""

    unavailable = (
        RELEVANCE_FEATURE_MEANS[1],
        RELEVANCE_FEATURE_MEANS[2],
        RELEVANCE_FEATURE_MEANS[3],
        RELEVANCE_FEATURE_MEANS[4],
    )
    if history is None or getattr(history, "available", False) is not True:
        return unavailable

    values = (
        getattr(history, "matured", None),
        getattr(history, "consumed", None),
        getattr(history, "trailing_nonconsumed", None),
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        return unavailable
    matured, consumed, streak = values
    if (
        matured < 0
        or consumed < 0
        or consumed > matured
        or streak < 0
        or streak > matured - consumed
    ):
        return unavailable

    # ``round(..., 8)`` is part of the unchanged evaluator's feature
    # arithmetic.  Keeping it here is necessary to reproduce threshold ties.
    return (
        round(log1p(matured), 8),
        round(log1p(matured - consumed), 8),
        round(log1p(streak), 8),
        round(consumed / matured, 8) if matured else 0.0,
    )


def _relevance_features(node: Any, history: Any) -> tuple[float, ...]:
    level = getattr(node, "level", None)
    level_schema = (
        float(level == "schema")
        if level in {"trace", "concept", "schema"}
        else RELEVANCE_FEATURE_MEANS[0]
    )
    return (level_schema, *_history_features(history))


def _directional_zsum(features: Sequence[float]) -> float:
    """Exact equal-weight directional-zsum-r1 arithmetic."""

    if len(features) != len(RELEVANCE_FEATURE_MEANS):
        raise ValueError("relevance feature vector has the wrong width")
    return sum(
        (float(value) - mean) / scale
        for value, mean, scale in zip(
            features,
            RELEVANCE_FEATURE_MEANS,
            RELEVANCE_FEATURE_SCALES,
            strict=True,
        )
    )


def relevance_score(node: Any, history: Any) -> float:
    """Score one node from only its level and strictly matured history."""

    return _directional_zsum(_relevance_features(node, history))


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

    The same two statements, and the same completeness argument, as
    ``retrieval._chunk_table_revision``: chunk rows are only inserted or
    deleted, never updated in place, so a delete moves the count and an insert
    raises the ULID maximum. Kept here rather than imported because importing
    ``retrieval`` would pull the whole ranker — and its embedding model — into
    a module that exists to stay cheap; ``tests/test_recall_map.py`` pins the
    two against each other.
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
        decision_at: str | datetime | None = None,
    ) -> RecallMap | None:
        """Cluster one residual pool. ``None`` when there is nothing to say.

        ``results`` is the ranked residual — ``MemoryRecallService.last_residual``
        — best first. Its original order supplies the frozen ordinal after
        relevance selection; the selected pool itself is ordered by score.
        ``task``/``task_pattern`` name the work the recall belongs to and, with
        ``scope``, form the cache key; without either, the key degrades to the
        scope alone, which is still stable and merely coarser.

        ``decision_at`` is captured once per call so every bounded history
        batch sees one boundary. Supplying it is useful for deterministic
        replay; production callers omit it and receive the current UTC instant.

        A key whose last :data:`CURTAIL_STREAK` maps were delivered and never
        followed gets the collapsed form instead — clusters empty, ``curtailed``
        set, the streak carried — and gets it *before* any clustering runs, so
        a channel nobody reads also stops costing what it costs to fill.
        """

        self.last_cache_hit = False
        self._reset_corpus_memo()
        if not results:
            return None

        instant = decision_at if decision_at is not None else datetime.now(UTC)
        pool, selection = self._pool(results, decision_at=instant)
        map_scope = (
            normalize_scope(scope) if scope else _dominant_residual_scope(results)
        )
        key = cache_key(map_scope, task=task, task_pattern=task_pattern)
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
            self._note_delivery(map_scope, task, offered=False)
            return RecallMap(
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

        curtailment = self._curtailment(map_scope, task)
        self.last_curtailment = curtailment
        if curtailment.collapse:
            self._note_delivery(map_scope, task, offered=False)
            return RecallMap(
                scope=map_scope,
                key=key,
                clusters=(),
                pool_size=len(pool),
                covered=0,
                curtailed=True,
                streak=curtailment.streak,
                selection=selection,
                selection_sample_limit=_selection_sample_limit(
                    (),
                    len(pool),
                    0,
                    selection=selection,
                    curtailed=True,
                    streak=curtailment.streak,
                ),
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
            groups = self._cluster(pool)
            self._remember(key, revision, groups)
        built = self._finish(
            groups,
            scope=map_scope,
            key=key,
            pool_size=len(pool),
            selection=selection,
        )
        if built is not None:
            # A journal-only map reaches the server so a fully gated pool does
            # not disappear from field evidence, but it offered the agent no
            # cluster and therefore must not advance the curtail offer count.
            self._note_delivery(map_scope, task, offered=bool(built.clusters))
        return built

    # -- pool ----------------------------------------------------------

    def _pool(
        self,
        results: Sequence["RecallResult"],
        *,
        decision_at: str | datetime,
    ) -> tuple[list[_Member], SelectionAccounting]:
        """Apply the frozen member pipeline before the pool cap.

        Classification is one ordinal pass.  History is then read only for
        first-occurrence, structurally eligible identities, in batches no
        larger than storage's frozen bound.  Every survivor is scored before
        the stable relevance sort and cap, so machine ballast and low-score
        head entries cannot crowd a useful tail member out of the 200.
        """

        ledger = _SelectionLedger(inspected=len(results))
        seen: set[str] = set()
        eligible: list[tuple[int, Node]] = []
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
            eligible.append((ordinal, node))

        histories = self._matured_history(
            [node.id for _ordinal, node in eligible], decision_at=decision_at
        )
        survivors: list[_Member] = []
        for ordinal, node in eligible:
            score = relevance_score(node, histories.get(node.id))
            if score < RELEVANCE_THRESHOLD:
                ledger.exclude("lr", ordinal=ordinal, node=node)
                continue
            survivors.append(
                _Member(rank=ordinal, node=node, relevance_score=score)
            )

        survivors.sort(key=lambda member: (-member.relevance_score, member.rank))
        admitted = survivors[: self.max_pool_nodes]
        for member in sorted(
            survivors[self.max_pool_nodes :], key=lambda item: item.rank
        ):
            ledger.exclude("pc", ordinal=member.rank, node=member.node)
        accounting = ledger.freeze(len(admitted))
        return admitted, accounting

    def _matured_history(
        self,
        node_ids: Sequence[str],
        *,
        decision_at: str | datetime,
    ) -> dict[str, Any]:
        """Read every candidate's history through fixed-size bounded batches.

        A missing method, failed batch, absent id, malformed row, truncation or
        unavailable ledger remains absent in this mapping (or carries the
        reader's unavailable object) and therefore receives frozen train means
        in :func:`relevance_score`.  A real zero-history row is present with
        ``available=True`` and M=C=K=0, preserving the cold-start distinction.
        """

        histories: dict[str, Any] = {}
        for offset in range(0, len(node_ids), MAX_RECALL_HISTORY_CANDIDATES):
            batch = node_ids[offset : offset + MAX_RECALL_HISTORY_CANDIDATES]
            try:
                fetched = self.store.matured_recall_history(batch, decision_at)
            except (AttributeError, sqlite3.Error, TypeError, ValueError):
                continue
            if not isinstance(fetched, Mapping):
                continue
            for node_id in batch:
                if node_id in fetched:
                    histories[node_id] = fetched[node_id]
        return histories

    # -- cascade -------------------------------------------------------

    def _cluster(self, pool: list[_Member]) -> list[_Group]:
        """Run the four stages, each over what the previous one left."""

        groups: list[_Group] = []
        remaining = pool

        claimed, remaining = self._stage_structural(remaining)
        groups.extend(claimed)
        claimed, remaining = self._stage_path(remaining)
        groups.extend(claimed)
        claimed, remaining = self._stage_anchor(remaining)
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

    def _stage_anchor(
        self, members: list[_Member]
    ) -> tuple[list[_Group], list[_Member]]:
        """Group by the live query anchor whose edges cover each node.

        One indexed lookup per unclaimed node — ``idx_query_anchor_edges_target``
        exists precisely for this direction — and one anchor row per distinct
        anchor reached, memoized. A node covered by several anchors goes to the
        heaviest edge, ties to the lower anchor id, so the assignment does not
        depend on the order the edge table returns.
        """

        anchors: dict[str, Any] = {}
        assignment: dict[str, str] = {}
        rest: list[_Member] = []
        for member in members:
            best: tuple[float, str] | None = None
            for edge in self.store.list_query_anchor_edges(target_id=member.node.id):
                anchor = anchors.get(edge.anchor_id)
                if anchor is None:
                    anchor = self.store.get_query_anchor(edge.anchor_id)
                    if anchor is None:
                        continue
                    anchors[edge.anchor_id] = anchor
                if anchor.decayed or not str(anchor.query).strip():
                    continue
                weight = float(edge.weight)
                if best is None or (-weight, edge.anchor_id) < (-best[0], best[1]):
                    best = (weight, edge.anchor_id)
            if best is None:
                rest.append(member)
            else:
                assignment[member.node.id] = best[1]

        buckets: dict[str, list[_Member]] = {}
        for member in members:
            anchor_id = assignment.get(member.node.id)
            if anchor_id is not None:
                buckets.setdefault(anchor_id, []).append(member)

        groups: list[_Group] = []
        for anchor_id, bucket in sorted(buckets.items()):
            query = _collapse(anchors[anchor_id].query)
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

    def _curtailment(self, scope: str, task: str | None) -> _Curtailment:
        """This key's unaccepted-offer streak, read only when it could bite.

        The gate in front of the read is :class:`_CurtailMemo`, and the whole
        argument for it is there. What matters here: a verdict returned from
        the memo is an estimate and never collapses anything, because the
        estimate is only trusted while it stays *below* the threshold.
        """

        key = cache_key(scope, task=task)
        memo = self._curtail_memo.get(key)
        if memo is not None and not memo.estimate().collapse:
            self.last_curtail_read = False
            return memo.estimate()

        self.last_curtail_read = True
        verdict = self._read_curtailment(scope, task)
        self._curtail_memo.pop(key, None)
        self._curtail_memo[key] = _CurtailMemo(verdict=verdict)
        while len(self._curtail_memo) > MAX_CACHE_ENTRIES:
            self._curtail_memo.pop(next(iter(self._curtail_memo)))
        return verdict

    def _note_delivery(self, scope: str, task: str | None, *, offered: bool) -> None:
        """Count one map this builder just handed the server to deliver.

        ``offered`` separates a map from a collapsed marker, because only a
        map can go unaccepted. Counted here rather than inferred at read time
        so that the gate above stays a pure function of what this process
        knows it did.
        """

        memo = self._curtail_memo.get(cache_key(scope, task=task))
        if memo is None:  # pragma: no cover - the read always seeds one
            return
        memo.total += 1
        if offered:
            memo.offers += 1

    def _read_curtailment(self, scope: str, task: str | None) -> _Curtailment:
        """Walk this key's deliveries back until one of them was followed.

        Newest first, stopping at the first delivery with a consumed cluster:
        what is *before* that delivery cannot make the channel look unused,
        because the channel demonstrably was used. Everything walked past is
        the streak; the offers among it are what decides the collapse.

        Two reads for the whole window, not two per delivery: the medoid
        timestamps come back in one batch and every query is tokenized once,
        because this is the expensive half of the rule and
        :class:`_CurtailMemo` exists to keep it from being asked often. Any
        store that cannot answer either read — an older schema, a ranking-only
        store face — yields an empty verdict, and an empty verdict never
        collapses anything.
        """

        try:
            history = self._delivery_history(scope, task)
        except (AttributeError, sqlite3.Error):  # pragma: no cover - old store
            return _Curtailment(streak=0, offers=0)
        if not history:
            return _Curtailment(streak=0, offers=0)

        items = [_delivered_items(row.get("recall_map")) for row in history]
        try:
            accessed = self._medoid_access(items)
        except sqlite3.Error:  # pragma: no cover - defensive
            accessed = {}
        queries = [token_set(str(row.get("query") or "")) for row in history]

        streak = 0
        offers = 0
        for index, row in enumerate(history):
            delivered = items[index]
            if delivered and self._was_followed(
                delivered,
                delivered_at=str(row.get("created_at") or ""),
                later_queries=queries[:index],
                accessed=accessed,
            ):
                break
            streak += 1
            if delivered:
                offers += 1
        return _Curtailment(streak=streak, offers=offers)

    def _delivery_history(self, scope: str, task: str | None) -> list[dict[str, Any]]:
        """This key's recent deliveries, newest first.

        The storage read filters on the two columns ``recall_events`` actually
        stores — the resolved ``scope`` and the ambient ``task`` — and the
        ``task`` re-check here is what makes a *task-less* key its own key
        rather than a bucket collecting every task in the scope.

        Two ways this can under-see, both of which end in "no history", and
        both of which are therefore safe: a ``task_pattern``-keyed map looks
        for its ``task`` (the only one of the two the table has, and the
        coarser, so it can only merge deliveries of one task and never split
        one), and a scope resolved differently from the one the map is built
        under simply matches nothing. Seeing less history can only shorten a
        streak, and a shorter streak is a map that keeps being delivered.
        """

        rows = self.store.recent_recall_map_history(
            scope=scope,
            task=task,
            limit=CURTAIL_HISTORY_LIMIT,
        )
        wanted = normalize_key(task) if task else ""
        return [
            row
            for row in rows
            if normalize_key(str(row.get("task") or "")) == wanted
            and isinstance(row.get("recall_map"), Mapping)
        ]

    def _medoid_access(
        self, items: Sequence[Sequence[_DeliveredItem]]
    ) -> dict[str, str]:
        """``node_id -> last access timestamp`` for every medoid on offer.

        One primary-key batch for the whole window — at most
        :data:`CURTAIL_HISTORY_LIMIT` times :data:`MAX_CLUSTERS` ids, well
        inside SQLite's parameter limit — and two columns of it, because the
        live path has no business deserializing whole nodes to read one
        timestamp off each.
        """

        ids = sorted(
            {item.medoid_id for delivered in items for item in delivered if item.medoid_id}
        )
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.store.connection.execute(
            f"SELECT id, last_accessed FROM nodes WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return {str(row["id"]): str(row["last_accessed"] or "") for row in rows}

    @staticmethod
    def _was_followed(
        delivered: Sequence[_DeliveredItem],
        *,
        delivered_at: str,
        later_queries: Sequence[frozenset[str]],
        accessed: Mapping[str, str],
    ) -> bool:
        """Whether any cluster of one delivered map was acted on afterwards.

        Timestamps are compared as strings because they are stored as
        second-resolution UTC ISO-8601, where lexical order *is* chronological.
        The comparison is inclusive: at one-second resolution a tie is not
        evidence of anything, and the direction this module resolves
        non-evidence in is "used".
        """

        for item in delivered:
            touched = accessed.get(item.medoid_id, "")
            if delivered_at and touched and touched >= delivered_at:
                return True
            for query in later_queries:
                if _echoes(item.label, query) or _echoes(item.ask_hint, query):
                    return True
        return False

    # -- cache ---------------------------------------------------------

    def _corpus_revision(self) -> tuple[Any, ...]:
        """Identity of everything the cached *structure* was derived from.

        Two tiers, exactly as ``retrieval._chunk_corpus_revision``: a free
        ``(total_changes, PRAGMA data_version)`` probe answers "has anything
        been written at all", and only when it moves does the second tier ask
        the chunk and anchor tables themselves. Most recalls move the probe
        with their own bookkeeping — ``record_access`` writes on every call —
        and move neither table, so the second tier is two indexed seeks that
        return the value already held.
        """

        connection = self.store.connection
        probe = (int(connection.total_changes), _data_version(connection))
        cached = self._revision
        if (
            cached is not None
            and cached[0] == RELEVANCE_POLICY_DIGEST
            and probe == self._write_probe
        ):
            return cached
        revision = (
            RELEVANCE_POLICY_DIGEST,
            _chunk_table_revision(connection),
            tuple(self.store.query_anchor_revision()),
        )
        self._write_probe = probe
        self._revision = revision
        return revision

    def _remember(
        self, key: str, revision: tuple[Any, ...], groups: list[_Group]
    ) -> None:
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

        # Stage 4 is a relevance-ordered greedy walk. Re-run that membership
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

        What survives is ordered only by the frozen member policy: descending
        maximum member score, descending mean member score, then the unique
        best original residual ordinal. The mean uses :func:`math.fsum` over
        original residual order. Size, cascade stage, label, task and identity
        are deliberately absent from the relevance order.

        Order is deliberately not part of the cached structure.  A cached
        recount supplies this call's :class:`_Member` objects and therefore
        this call's ranks and scores; this sort and medoid tie logic refresh
        relevance-derived presentation on both cold and warm paths while
        labels and signatures remain stable.

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

        def relevance_key(group: _Group) -> tuple[float, float, int]:
            members = sorted(group.members, key=lambda member: member.rank)
            scores = [member.relevance_score for member in members]
            return (
                -max(scores),
                -(fsum(scores) / len(scores)),
                members[0].rank,
            )

        deliverable.sort(key=relevance_key)
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


def cache_key(scope: str, *, task: str | None = None, task_pattern: str | None = None) -> str:
    """``(scope, normalized task or task_pattern)`` as one stable string.

    ``task_pattern`` wins when both are present: it names the recurring *class*
    of work, which is what should share a structure across sessions, while a
    ``task`` string is usually this session's phrasing of it. Both are folded by
    :func:`normalize_key`, so "postsession-extraction" and "Postsession
    Extraction" are one key.
    """

    label = task_pattern if task_pattern else task
    return f"{normalize_scope(scope)}|{normalize_key(label) if label else ''}"


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
    pool_size: int,
    dropped: int,
    *,
    curtailed: bool = False,
    streak: int = 0,
    filtered: Mapping[str, Any] | None = None,
    selection: SelectionAccounting | None = None,
    selection_sample_limit: int = SELECTION_SAMPLE_LIMIT,
) -> dict[str, Any]:
    """The wire form. Curtailed maps keep the shape and drop the content.

    ``clusters`` stays present and empty rather than being omitted, so every
    consumer of a persisted map — the instructions channel, the effect gate,
    this module's own :meth:`RecallMapBuilder._curtailment` — reads one shape
    and reaches the collapse through ``curtailed`` instead of through a
    ``KeyError``.

    ``filtered`` is a *sibling* key and never an entry inside ``clusters``.
    That is the hard rule the additive shape rests on: ``_delivered_items`` is
    paranoid about the shape of a cluster but not about extra ones, so a
    journal row inside the list would be read as a delivered item by the
    curtail probe and as a novelty-consuming hint by the ae probe — corrupting
    both on a cluster nobody was ever offered.
    """

    payload: dict[str, Any] = {
        "clusters": [cluster.to_dict() for cluster in clusters],
        "pool": pool_size,
        "covered": sum(cluster.count for cluster in clusters),
    }
    if dropped > 0:
        payload["more"] = dropped
    if filtered:
        payload["filtered"] = dict(filtered)
    if selection is not None:
        payload["sel"] = selection.to_dict(sample_limit=selection_sample_limit)
    if curtailed:
        payload["curtailed"] = True
        payload["streak"] = streak
    return payload


def _selection_sample_limit(
    clusters: Sequence[MapCluster],
    pool_size: int,
    dropped: int,
    filtered: Mapping[str, Any] | None = None,
    *,
    selection: SelectionAccounting,
    curtailed: bool = False,
    streak: int = 0,
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
        )
        if len(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        ) <= MAX_RESPONSE_CHARS:
            return sample_limit
    return 0


def _payload_size(
    clusters: Sequence[MapCluster],
    pool_size: int,
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
    "CURTAIL_HISTORY_LIMIT",
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
    "RELEVANCE_FEATURE_MEANS",
    "RELEVANCE_FEATURE_SCALES",
    "RELEVANCE_POLICY_DIGEST",
    "RELEVANCE_POLICY_ID",
    "RELEVANCE_THRESHOLD",
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
    "normalize_key",
    "relevance_score",
]
