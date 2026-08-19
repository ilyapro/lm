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
from dataclasses import dataclass
from math import log, sqrt
from typing import TYPE_CHECKING, Any
import json
import re
import sqlite3

from living_memory.embeddings import cosine_similarity, tokenize
from living_memory.grounding import token_set
from living_memory.models import Node
from living_memory.scope import GLOBAL_SCOPE, normalize_scope
from living_memory.storage import CHUNK_EMBEDDING_TABLE, MemoryStore

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
#: anything richer *this* is the binding constraint. What gives way, in order,
#: is the medoid example (:meth:`RecallMapBuilder._fit`) and only then whole
#: clusters — and a dropped cluster is counted in the payload's ``more`` field
#: rather than vanishing, because a map that silently truncates reads as "this
#: is everything memory holds", which is the one thing it must never say.
MAX_RESPONSE_CHARS = 700

#: Budget for :meth:`RecallMap.render_compact`, the single-string form the
#: server-instructions channel embeds. Instructions are capped at 2048 chars
#: by contract test and currently run within single digits of that ceiling, so
#: the map's whole footprint there is this line.
MAX_INSTRUCTIONS_CHARS = 150

#: Longest medoid example a map carries before the response budget starts
#: taking it apart. See :meth:`RecallMapBuilder._fit`.
MEDOID_EXAMPLE_CHARS = 120

MAX_LABEL_CHARS = 40
MAX_ASK_HINT_CHARS = 80

#: Residual pools are unbounded in principle; the map reads context of every
#: member and, at stage 3, issues one indexed edge lookup per unclaimed node.
#: Past this many ranked members the tail contributes nothing a six-cluster map
#: can show, so it is not paid for.
MAX_POOL_NODES = 200

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

_WHITESPACE_RE = re.compile(r"\s+")
_TERM_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_HEX_DIGITS = frozenset("0123456789abcdef")
_VOWELS = frozenset("aeiouyаеёиоуыэюя")


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
    #: Set when this key's last :data:`CURTAIL_STREAK` maps went unconsumed.
    #: A curtailed map carries no clusters: it has stopped describing the pool
    #: and started reporting that describing it was not worth the channel.
    curtailed: bool = False
    #: Consecutive unconsumed deliveries behind the collapse. *The* signal —
    #: the number is what tells an operator (and the effect gate) that the map
    #: went quiet and how long it has been quiet for.
    streak: int = 0

    def to_dict(self) -> dict[str, Any]:
        """The response-shaped form, budgeted by :data:`MAX_RESPONSE_CHARS`."""

        return _payload(
            self.clusters,
            self.pool_size,
            self.dropped,
            curtailed=self.curtailed,
            streak=self.streak,
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
    """One pool entry, carrying the rank that breaks every tie about it."""

    rank: int
    node: Node


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
    #: The medoid a cached structure already chose. Carried so that a map
    #: served from cache picks the same member a fresh build would, without
    #: re-fetching the vectors that chose it.
    preferred_medoid: str | None = None


@dataclass(frozen=True, slots=True)
class _ClusterTemplate:
    """A cached cluster: its identity and its labels, without its counts."""

    stage: str
    signature: tuple[Any, ...]
    label: str
    ask_hint: str
    member_ids: frozenset[str]
    #: The member the full build chose to speak for this cluster. Part of the
    #: structure, not of the counts: stage 4 picks it with vectors the cache
    #: path deliberately does not re-read, so without remembering it here the
    #: same pool would get one medoid cold and another warm.
    medoid_id: str


@dataclass(frozen=True, slots=True)
class _CachedStructure:
    """One cache entry: templates plus the corpus revision that validates them."""

    revision: tuple[Any, ...]
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


# ----------------------------------------------------------------------
# Stage helpers
# ----------------------------------------------------------------------


def _structural_key(node: Node) -> tuple[str, str] | None:
    """``(field, raw value)`` of the first structural key this node carries."""

    context = node.context if isinstance(node.context, Mapping) else {}
    for field in STRUCTURAL_FIELDS:
        value = context.get(field)
        if isinstance(value, str) and value.strip():
            return (field, value.strip())
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
    for field in _PATH_FIELDS:
        value = context.get(field)
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
    ) -> RecallMap | None:
        """Cluster one residual pool. ``None`` when there is nothing to say.

        ``results`` is the ranked residual — ``MemoryRecallService.last_residual``
        — best first; its order is the tie break of last resort throughout.
        ``task``/``task_pattern`` name the work the recall belongs to and, with
        ``scope``, form the cache key; without either, the key degrades to the
        scope alone, which is still stable and merely coarser.

        A key whose last :data:`CURTAIL_STREAK` maps were delivered and never
        followed gets the collapsed form instead — clusters empty, ``curtailed``
        set, the streak carried — and gets it *before* any clustering runs, so
        a channel nobody reads also stops costing what it costs to fill.
        """

        pool = self._pool(results)
        self.last_cache_hit = False
        if not pool:
            return None

        map_scope = normalize_scope(scope) if scope else _dominant_scope(pool)
        key = cache_key(map_scope, task=task, task_pattern=task_pattern)

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
            )

        revision = self._corpus_revision()

        cached = self._cache.get(key)
        groups: list[_Group] | None = None
        if cached is not None and cached.revision == revision:
            groups = self._recount(cached.templates, pool)
            self.last_cache_hit = groups is not None
        if groups is None:
            groups = self._cluster(pool)
            self._remember(key, revision, groups)
        built = self._finish(groups, scope=map_scope, key=key, pool_size=len(pool))
        if built is not None:
            # A map that reaches the server is a map that gets delivered and
            # recorded; one that came back None never happened and must not
            # count against the key that nearly made it.
            self._note_delivery(map_scope, task, offered=True)
        return built

    # -- pool ----------------------------------------------------------

    def _pool(self, results: Sequence["RecallResult"]) -> list[_Member]:
        members: list[_Member] = []
        seen: set[str] = set()
        for result in results:
            node = getattr(result, "node", None)
            if node is None or not getattr(node, "id", ""):
                continue
            if node.id in seen:
                continue
            seen.add(node.id)
            members.append(_Member(rank=len(members), node=node))
            if len(members) >= self.max_pool_nodes:
                break
        return members

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
            # label. Everywhere else the hint *is* the label, so it follows it.
            hint_followed_label = (
                group.stage != STAGE_ANCHOR and group.ask_hint == group.label
            )
            resolved = self._unique_label(group, seen)
            seen.add(resolved)
            if hint_followed_label:
                group.ask_hint = resolved
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
        for (field, raw), bucket in sorted(buckets.items()):
            normalized = normalize_key(raw)
            if _looks_unreadable(raw):
                # A hash groups perfectly and names nothing; the medoid does.
                readable = _phrase_from_content(bucket[0].node.content)
            else:
                readable = normalized
            groups.append(
                _Group(
                    stage=STAGE_STRUCTURAL,
                    signature=(STAGE_STRUCTURAL, field, raw),
                    label=readable,
                    ask_hint=readable,
                    members=bucket,
                )
            )
        return groups, rest

    def _stage_path(self, members: list[_Member]) -> tuple[list[_Group], list[_Member]]:
        buckets: dict[str, list[_Member]] = {}
        rest: list[_Member] = []
        for member in members:
            subsystem = _node_subsystem(member.node)
            if subsystem is None:
                rest.append(member)
                continue
            buckets.setdefault(subsystem, []).append(member)

        groups = [
            _Group(
                stage=STAGE_PATH,
                signature=(STAGE_PATH, subsystem),
                label=normalize_key(subsystem),
                ask_hint=normalize_key(subsystem),
                members=bucket,
            )
            for subsystem, bucket in sorted(buckets.items())
        ]
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

        try:
            frequencies = self.store.term_document_frequencies(candidates)
            total = self.store.fts_document_count()
        except (AttributeError, sqlite3.OperationalError):  # pragma: no cover - old store
            frequencies, total = {}, 0

        scored: list[tuple[float, str]] = []
        for term in candidates:
            weight = float(counted[term])
            if total > 1:
                weight *= log(total / (1.0 + float(frequencies.get(term, 0))))
            scored.append((weight, term))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [term for _weight, term in scored]

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
        if cached is not None and probe == self._write_probe:
            return cached
        revision = (
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
                medoid_id=self._medoid(group).node.id,
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
            for index, template in ordered:
                if not self._matches(template, member.node):
                    continue
                buckets.setdefault(index, []).append(member)
                covered += 1
                break
        if covered < CACHE_MIN_COVERAGE * len(pool):
            return None
        return [
            _Group(
                stage=templates[index].stage,
                signature=templates[index].signature,
                label=templates[index].label,
                ask_hint=templates[index].ask_hint,
                members=buckets[index],
                preferred_medoid=templates[index].medoid_id,
            )
            for index, _template in ordered
            if index in buckets
        ]

    @staticmethod
    def _matches(template: _ClusterTemplate, node: Node) -> bool:
        if template.stage == STAGE_STRUCTURAL:
            key = _structural_key(node)
            return key is not None and (STAGE_STRUCTURAL, *key) == template.signature
        if template.stage == STAGE_PATH:
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
    ) -> RecallMap | None:
        """Order, cap and budget the groups into a map.

        Largest first, because the map is a menu and the biggest pile is the
        likeliest next question; then by stage, so a label the corpus actually
        stores outranks one derived from term statistics; then by label, which
        makes the order total.
        """

        populated = [group for group in groups if group.members]
        if not populated:
            return None
        populated.sort(
            key=lambda group: (
                -len(group.members),
                _STAGE_ORDER.get(group.stage, 99),
                group.label,
            )
        )
        kept = populated[: self.max_clusters]
        clusters, dropped = self._fit(kept, pool_size, len(populated) - len(kept))
        if not clusters:
            return None
        return RecallMap(
            scope=scope,
            key=key,
            clusters=tuple(clusters),
            pool_size=pool_size,
            covered=sum(cluster.count for cluster in clusters),
            dropped=dropped,
        )

    def _fit(
        self, kept: list[_Group], pool_size: int, dropped: int
    ) -> tuple[list[MapCluster], int]:
        """Squeeze the map into :data:`MAX_RESPONSE_CHARS`, breadth last.

        Medoid examples give way first, and they give way *smoothly*: each pass
        shaves every example down to just under the longest one, by at least the
        per-cluster share of the overage, so a map one character over budget
        loses one character rather than losing every example it has. The shave
        strictly decreases the longest example, which is what bounds the loop.

        Only when the examples are gone entirely — the medoid ``node_id``
        survives, so "show me one" is still a ``memory_lookup`` away — does
        breadth give way, from the tail, which the sort has already made the
        smallest clusters. Every cluster lost that way is counted into
        ``dropped`` and surfaces as the payload's ``more``.
        """

        example_chars = MEDOID_EXAMPLE_CHARS
        clusters = [
            self._cluster_of(group, example_chars=example_chars) for group in kept
        ]
        while example_chars > 0:
            overage = _payload_size(clusters, pool_size, dropped) - MAX_RESPONSE_CHARS
            if overage <= 0:
                return clusters, dropped
            longest = max((len(cluster.medoid.example) for cluster in clusters), default=0)
            if longest <= 0:
                break
            example_chars = max(0, longest - max(1, overage // len(clusters)))
            clusters = [
                self._cluster_of(group, example_chars=example_chars) for group in kept
            ]

        while clusters and _payload_size(clusters, pool_size, dropped) > MAX_RESPONSE_CHARS:
            clusters = clusters[:-1]
            dropped += 1
        return clusters, dropped

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

        A cached choice wins whenever that member is still in the pool: it was
        made by a full build, with whatever evidence that build had, and
        re-deciding it here would hand the same pool one medoid cold and
        another warm.

        Otherwise, with vectors in play — stage 4 only, the one stage that has
        them — the member of maximum mean similarity to the rest, which is the
        medoid proper. Everywhere else the highest-ranked member, because rank
        is the only evidence of centrality those stages hold and inventing
        another would cost a vector fetch the fast path exists to avoid.
        """

        if group.preferred_medoid:
            for member in group.members:
                if member.node.id == group.preferred_medoid:
                    return member

        vectors = group.vectors
        if not vectors or len(group.members) < 2:
            return min(group.members, key=lambda member: (member.rank, member.node.id))
        scored: list[tuple[float, int, str, _Member]] = []
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
            scored.append((-mean, member.rank, member.node.id, member))
        if not scored:  # pragma: no cover - defensive
            return group.members[0]
        return min(scored, key=lambda item: item[:3])[3]


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


def _payload(
    clusters: Sequence[MapCluster],
    pool_size: int,
    dropped: int,
    *,
    curtailed: bool = False,
    streak: int = 0,
) -> dict[str, Any]:
    """The wire form. Curtailed maps keep the shape and drop the content.

    ``clusters`` stays present and empty rather than being omitted, so every
    consumer of a persisted map — the instructions channel, the effect gate,
    this module's own :meth:`RecallMapBuilder._curtailment` — reads one shape
    and reaches the collapse through ``curtailed`` instead of through a
    ``KeyError``.
    """

    payload: dict[str, Any] = {
        "clusters": [cluster.to_dict() for cluster in clusters],
        "pool": pool_size,
        "covered": sum(cluster.count for cluster in clusters),
    }
    if dropped > 0:
        payload["more"] = dropped
    if curtailed:
        payload["curtailed"] = True
        payload["streak"] = streak
    return payload


def _payload_size(clusters: Sequence[MapCluster], pool_size: int, dropped: int) -> int:
    return len(
        json.dumps(
            _payload(clusters, pool_size, dropped),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


__all__ = [
    "CACHE_MIN_COVERAGE",
    "CTFIDF_LABEL_TERMS",
    "CURTAIL_HISTORY_LIMIT",
    "CURTAIL_QUERY_OVERLAP",
    "CURTAIL_STREAK",
    "EMBEDDING_CLUSTER_COSINE",
    "MAX_ASK_HINT_CHARS",
    "MAX_CLUSTERS",
    "MAX_INSTRUCTIONS_CHARS",
    "MAX_LABEL_CHARS",
    "MAX_POOL_NODES",
    "MAX_RESPONSE_CHARS",
    "MEDOID_EXAMPLE_CHARS",
    "STAGE_ANCHOR",
    "STAGE_EMBEDDING",
    "STAGE_PATH",
    "STAGE_STRUCTURAL",
    "STRUCTURAL_FIELDS",
    "MapCluster",
    "MapMedoid",
    "RecallMap",
    "RecallMapBuilder",
    "cache_key",
    "normalize_key",
]
