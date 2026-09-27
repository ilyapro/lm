#!/usr/bin/env python3
"""Score explicit ``used``/``irrelevant`` recall marks against live evidence — read-only.

An explicit mark is the agent's own claim ("I used this delivered node" / "this
one was not about my query"). This script checks those claims against the two
live channels that do not come from the agent's say-so, with the chance level
of the lexical one subtracted:

* **Grounded, minus placebo.** A delivered node ``n`` of a recall event ``E``
  that was closed by a remember/teach trace ``T`` is *grounded* when
  :func:`living_memory.grounding.ground_token_sets` puts its IDF containment in
  ``T`` at or above 0.22. Lexical overlap happens by chance too, so every
  delivered node gets a **placebo twin**: a node NOT delivered in ``E``,
  created before ``E``, whose mean-pooled ``node_chunk_embeddings`` vector has
  the same cosine to ``T`` as ``n`` does (±0.01). The twin is graded in the
  same call (``docs = delivered + twin``), and *excess* = grounded rate − twin
  rate. The same is repeated with the query's tokens removed from ``T``, which
  takes out the "shared task vocabulary" channel. Method: LM memory
  01M3H76JBHEZKJMBPWFVN70SB2 (sfx 2026-09-27: r1 12.8% vs twin 3.7%; without
  query tokens 7.7% vs 2.4%).
* **Lookup.** A ``memory_lookup`` of ``n`` on the same transport session after
  ``E`` was delivered (``recall_lookup_events``), or a ``lookup`` row for
  ``(E, n)`` in ``recall_credit_ledger``. There is no placebo for this
  channel: an undelivered twin cannot be looked up from ``E``'s card.

Marks come from ``recall_feedback_marks`` (id, recall_event_id, node_id, mark
'used'|'irrelevant', accepted 0/1, reject_reason, via_tool, source_id,
transport_session_id, agent, rank 0-based, marked_at). A store without the
table reports "no marks" and still produces the placebo reference rates, which
is what a baseline run before the marks ship is for.

The falsifier test (named ``better_than_random`` in the JSON) is stated in
:data:`FALSIFIER_TEXT` and printed verbatim in every report.

Read-only by construction: the store is opened ``file:...?mode=ro`` with
``PRAGMA query_only``, never through ``MemoryStore`` (which writes whatever it
opens). Point ``--db`` at a live store or at a snapshot made with
``living_memory.retrieval_harness.backup_database``. One store per run: sfx
and alt are never merged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from living_memory.grounding import ground_token_sets, token_set  # noqa: E402

#: Grounding threshold of the placebo method (01M3H76JBHEZKJMBPWFVN70SB2).
MIN_CONTAINMENT = 0.22
#: Twin cosine band: |cos(twin, T) - cos(n, T)| <= TWIN_BAND.
TWIN_BAND = 0.01
MARKS_TABLE = "recall_feedback_marks"
AGENT_TYPES = ("claude", "codex", "opencode", "gigacode", "other", "unknown")
#: Minimum sample for the falsifier to be decided rather than "insufficient".
MIN_USED_MARKS = 30
MIN_USED_EVENTS = 10
ALPHA = 0.05
#: Share of marks (events) above which a ritual detector is flagged; the root
#: goal's mandatory-arm falsifier uses the same 50% line.
RITUAL_FLAG_SHARE = 0.5
BOOTSTRAP_ROUNDS = 1000
PERMUTATION_ROUNDS = 2000

FALSIFIER_TEXT = (
    "better_than_random: explicit `used` marks agree with the independent check "
    "better than random iff ALL hold: (a) at least 30 accepted `used` marks on "
    "at least 10 in-window recall events; (b) within-event permutation test: the "
    "number of `used`-marked nodes that carry evidence (grounded in the closing "
    "trace at containment >= 0.22 OR looked up on the same transport after "
    "delivery) is compared with 2000 draws in which each event's `used` marks "
    "are re-assigned uniformly at random to the same number of that event's "
    "delivered nodes; one-sided p = (1 + #draws >= observed) / (1 + 2000) must "
    "be < 0.05; (c) placebo-subtracted evidence excess of `used` marks, "
    "mean over marks of [evidence(n) - grounded(twin(n))], has an event-cluster "
    "bootstrap 95% CI (1000 rounds) whose lower bound is > 0. If (a) fails the "
    "verdict is `insufficient`; otherwise failing (b) or (c) is `fail` - the "
    "falsifier fires and marks are not connected to reinforcement."
)

#: Session-level A/B markers. A transport session that wrote any node carrying
#: one of these, or issued a recall scoped/queried like one, is excluded as a
#: whole: the 2026-09-23 leak came from A/B sessions whose recalls themselves
#: looked like ordinary traffic. The effect-daily-metric sibling owns the
#: canonical filter; this one is deliberately session-wide and conservative.
AB_SCOPES = frozenset({"project:target", "project:repo", "project:x"})
AB_QUERY_RE = re.compile(
    r"tree[-_ ]context|(?<!credit )(?<!lookup )\bledger\b|billing|fixture|kit acceptance|development[-_]battery"
    r"|development[-_]water|reservoir[-_]allocation",
    re.IGNORECASE,
)
AB_CONTEXT_RE = re.compile(r"tree-context-ab|\"fixture\"|\"project\": ?\"target\"", re.IGNORECASE)


# --------------------------------------------------------------------------- io


def open_readonly(path: str | Path) -> sqlite3.Connection:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def normalize_ts(value: str | None) -> str:
    """Canonical sortable UTC string; the store mixes ``...Z`` and ``....ffffffZ``."""

    if not value:
        return ""
    text = str(value).strip().replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return str(value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def agent_type(raw: str | None) -> str:
    text = (raw or "").strip().lower()
    if not text:
        return "unknown"
    for name in ("gigacode", "opencode", "codex", "claude"):
        if name in text:
            return name
    return "other"


# ------------------------------------------------------------------------ model


@dataclass
class Event:
    id: str
    created_at: str
    query: str
    scope: str
    transport: str | None
    agent_raw: str | None
    trace_id: str | None
    results: list[str]  # node ids in rank order (index = 0-based rank)


@dataclass
class Observation:
    """One delivered (event, node) pair and every signal about it."""

    event_id: str
    node_id: str
    rank: int
    agent: str
    closed: bool
    session_duplicate: bool
    lookup: bool
    cos_trace: float | None = None
    containment: float | None = None
    grounded: bool = False
    grounded_q: bool = False
    has_twin: bool = False
    twin_grounded: bool = False
    twin_grounded_q: bool = False
    mark: str | None = None

    @property
    def evidence(self) -> bool:
        return self.grounded or self.lookup

    @property
    def evidence_score(self) -> float:
        return (self.containment or 0.0) + (1.0 if self.lookup else 0.0)

    @property
    def placebo_excess(self) -> float:
        """evidence(n) - grounded(twin(n)); twin term is 0 without a closing trace."""

        twin = 1.0 if (self.closed and self.has_twin and self.twin_grounded) else 0.0
        return (1.0 if self.evidence else 0.0) - twin


@dataclass
class Mark:
    id: str
    event_id: str
    node_id: str
    mark: str
    accepted: bool
    reject_reason: str | None
    via_tool: str | None
    source_id: str | None
    transport: str | None
    agent: str | None
    rank: int | None
    marked_at: str


@dataclass
class Store:
    events: dict[str, Event]
    window_ids: list[str]
    excluded_ab: int
    ab_sessions: set[str]
    session_events: dict[str, list[tuple[str, str]]]
    session_nodes: dict[str, list[tuple[str, str]]]
    session_agent: dict[str, str]
    lookups: dict[str, list[tuple[str, str]]]
    ledger: dict[tuple[str, str], str]
    marks: list[Mark] | None
    node_created: dict[str, str] = field(default_factory=dict)


def load_store(connection: sqlite3.Connection, since: str | None) -> Store:
    since_norm = normalize_ts(since) if since else ""
    node_created: dict[str, str] = {}
    session_nodes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    session_agent_votes: dict[str, Counter[str]] = defaultdict(Counter)
    ab_sessions: set[str] = set()
    for node_id, scope, context_text, created in connection.execute(
        "SELECT id, scope, context, created_at FROM nodes"
    ):
        created_n = normalize_ts(created)
        node_created[node_id] = created_n
        try:
            context = json.loads(context_text or "{}")
        except json.JSONDecodeError:
            context = {}
        transport = context.get("transport_session_id") if isinstance(context, dict) else None
        if not transport:
            continue
        session_nodes[transport].append((created_n, node_id))
        if isinstance(context, dict) and context.get("agent"):
            session_agent_votes[transport][str(context["agent"])] += 1
        if scope in AB_SCOPES or AB_CONTEXT_RE.search(context_text or ""):
            ab_sessions.add(transport)

    events: dict[str, Event] = {}
    session_events: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in connection.execute(
        "SELECT id, created_at, query, scope, transport_session_id, agent, "
        "ambient_context, feedback_trace_id, results FROM recall_events"
    ):
        event_id, created, query, scope, transport, agent, ambient_text, trace_id, results = row
        try:
            ambient = json.loads(ambient_text or "{}")
        except json.JSONDecodeError:
            ambient = {}
        transport = transport or (ambient.get("transport_session_id") if isinstance(ambient, dict) else None)
        try:
            decoded = json.loads(results or "[]")
        except json.JSONDecodeError:
            decoded = []
        ranked = sorted(
            (item for item in decoded if isinstance(item, dict) and item.get("node_id")),
            key=lambda item: item.get("rank", 0),
        )
        agent_raw = agent or (ambient.get("agent") if isinstance(ambient, dict) else None)
        event = Event(
            id=event_id,
            created_at=normalize_ts(created),
            query=query or "",
            scope=scope or "",
            transport=transport,
            agent_raw=agent_raw,
            trace_id=trace_id,
            results=[str(item["node_id"]) for item in ranked],
        )
        events[event_id] = event
        if transport:
            session_events[transport].append((event.created_at, event_id))
            if scope in AB_SCOPES or AB_QUERY_RE.search(event.query):
                ab_sessions.add(transport)
            if agent_raw:
                session_agent_votes[transport][str(agent_raw)] += 1
    for items in session_events.values():
        items.sort()
    for items in session_nodes.values():
        items.sort()

    def is_ab(event: Event) -> bool:
        if event.transport and event.transport in ab_sessions:
            return True
        return event.scope in AB_SCOPES or bool(AB_QUERY_RE.search(event.query))

    window_ids: list[str] = []
    excluded_ab = 0
    for event in sorted(events.values(), key=lambda item: item.created_at):
        if since_norm and event.created_at < since_norm:
            continue
        if not event.results:
            continue
        if is_ab(event):
            excluded_ab += 1
            continue
        window_ids.append(event.id)

    lookups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    if _table_exists(connection, "recall_lookup_events"):
        for node_id, occurred, transport in connection.execute(
            "SELECT node_id, occurred_at, transport_session_id FROM recall_lookup_events"
        ):
            if transport:
                lookups[transport].append((normalize_ts(occurred), node_id))
    ledger: dict[tuple[str, str], str] = {}
    if _table_exists(connection, "recall_credit_ledger"):
        for event_id, node_id, basis in connection.execute(
            "SELECT recall_event_id, node_id, basis FROM recall_credit_ledger"
        ):
            ledger[(event_id, node_id)] = basis

    marks: list[Mark] | None = None
    if _table_exists(connection, MARKS_TABLE):
        marks = []
        for row in connection.execute(
            f"SELECT id, recall_event_id, node_id, mark, accepted, reject_reason, via_tool, "
            f"source_id, transport_session_id, agent, rank, marked_at FROM {MARKS_TABLE}"
        ):
            marks.append(
                Mark(
                    id=str(row[0]),
                    event_id=row[1],
                    node_id=row[2],
                    mark=row[3],
                    accepted=bool(row[4]),
                    reject_reason=row[5],
                    via_tool=row[6],
                    source_id=row[7],
                    transport=row[8],
                    agent=row[9],
                    rank=None if row[10] is None else int(row[10]),
                    marked_at=normalize_ts(row[11]),
                )
            )

    session_agent = {
        transport: votes.most_common(1)[0][0] for transport, votes in session_agent_votes.items()
    }
    return Store(
        events=events,
        window_ids=window_ids,
        excluded_ab=excluded_ab,
        ab_sessions=ab_sessions,
        session_events=session_events,
        session_nodes=session_nodes,
        session_agent=session_agent,
        lookups=lookups,
        ledger=ledger,
        marks=marks,
        node_created=node_created,
    )


# ---------------------------------------------------------------- vectors/twins


class VectorIndex:
    """Mean-pooled unit vectors of every node with usable chunk rows.

    Pooling follows ``living_memory.near_dup.mean_pooled_vectors``: each chunk
    is unit-normalized before averaging, rows disagreeing on ``dimensions``
    make the node unusable, and a missing node is "unknown", never "far".
    """

    def __init__(self, connection: sqlite3.Connection, node_created: Mapping[str, str]) -> None:
        sums: dict[str, np.ndarray] = {}
        dims: dict[str, int] = {}
        broken: set[str] = set()
        for node_id, dimensions, blob in connection.execute(
            "SELECT node_id, dimensions, embedding FROM node_chunk_embeddings ORDER BY node_id, chunk_index"
        ):
            if node_id in broken:
                continue
            if node_id in dims and dims[node_id] != dimensions:
                broken.add(node_id)
                sums.pop(node_id, None)
                continue
            raw = bytes(blob)
            if len(raw) != 4 * dimensions:
                broken.add(node_id)
                sums.pop(node_id, None)
                continue
            vector = np.frombuffer(raw, dtype="<f4").astype(np.float64)
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                continue
            dims[node_id] = dimensions
            sums[node_id] = sums.get(node_id, 0.0) + vector / norm
        width = Counter(dims[node_id] for node_id in sums).most_common(1)
        width_value = width[0][0] if width else 0
        ids: list[str] = []
        rows: list[np.ndarray] = []
        for node_id, total in sums.items():
            if dims[node_id] != width_value:
                continue
            norm = float(np.linalg.norm(total))
            if norm == 0.0:
                continue
            ids.append(node_id)
            rows.append(total / norm)
        self.ids = ids
        self.position = {node_id: index for index, node_id in enumerate(ids)}
        self.matrix = np.vstack(rows) if rows else np.zeros((0, max(1, width_value)))
        self.created = np.array([node_created.get(node_id, "") for node_id in ids], dtype="U32")

    def vector(self, node_id: str) -> np.ndarray | None:
        index = self.position.get(node_id)
        return None if index is None else self.matrix[index]


def pick_twin(
    target_cos: float,
    cos_all: np.ndarray,
    eligible: np.ndarray,
    ids: Sequence[str],
    seed_key: str,
    band: float = TWIN_BAND,
) -> str | None:
    """Uniform draw among eligible nodes whose cosine to T is within ``band``."""

    candidates = np.flatnonzero(eligible & (np.abs(cos_all - target_cos) <= band))
    if candidates.size == 0:
        return None
    seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "big")
    choice = np.random.default_rng(seed).integers(candidates.size)
    return ids[int(candidates[choice])]


# ------------------------------------------------------------------- evaluation


class Contents:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.cache: dict[str, frozenset[str] | None] = {}

    def tokens(self, node_id: str) -> frozenset[str] | None:
        if node_id not in self.cache:
            row = self.connection.execute(
                "SELECT content FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()
            self.cache[node_id] = None if row is None else token_set(row[0] or "")
        return self.cache[node_id]


def _lookup_after(store: Store, event: Event, node_id: str) -> bool:
    if store.ledger.get((event.id, node_id)) == "lookup":
        return True
    if not event.transport:
        return False
    return any(
        node == node_id and occurred >= event.created_at
        for occurred, node in store.lookups.get(event.transport, ())
    )


def _session_duplicate(store: Store, event: Event, node_id: str) -> bool:
    if not event.transport:
        return False
    for created, other_id in store.session_events.get(event.transport, ()):
        if created >= event.created_at:
            break
        if other_id != event.id and node_id in store.events[other_id].results:
            return True
    return False


def event_agent(store: Store, event: Event) -> str:
    raw = event.agent_raw or (store.session_agent.get(event.transport) if event.transport else None)
    return agent_type(raw)


def observe(
    connection: sqlite3.Connection,
    store: Store,
    *,
    min_containment: float = MIN_CONTAINMENT,
) -> tuple[list[Observation], dict[str, int]]:
    """One Observation per delivered (event, node) in the window."""

    index = VectorIndex(connection, store.node_created)
    contents = Contents(connection)
    stats: Counter[str] = Counter()
    observations: list[Observation] = []
    for event_id in store.window_ids:
        event = store.events[event_id]
        agent = event_agent(store, event)
        rows = [
            Observation(
                event_id=event.id,
                node_id=node_id,
                rank=rank,
                agent=agent,
                closed=False,
                session_duplicate=_session_duplicate(store, event, node_id),
                lookup=_lookup_after(store, event, node_id),
            )
            for rank, node_id in enumerate(event.results)
        ]
        observations.extend(rows)
        stats["events"] += 1
        if not event.trace_id:
            continue
        stats["closed_events"] += 1
        trace_tokens = contents.tokens(event.trace_id)
        trace_vector = index.vector(event.trace_id)
        if trace_tokens is None or trace_vector is None:
            stats["closed_without_trace_vector"] += 1
            continue
        stats["closed_scored"] += 1
        query_tokens = token_set(event.query)
        trace_tokens_q = trace_tokens - query_tokens
        cos_all = index.matrix @ trace_vector
        delivered = set(event.results)
        eligible = index.created < event.created_at
        for node_id in (*delivered, event.trace_id):
            position = index.position.get(node_id)
            if position is not None:
                eligible[position] = False
        docs = {node_id: contents.tokens(node_id) for node_id in event.results}
        docs = {node_id: tokens for node_id, tokens in docs.items() if tokens is not None}
        for row in rows:
            row.closed = True
            vector = index.vector(row.node_id)
            if vector is None or row.node_id not in docs:
                stats["delivered_without_vector"] += 1
                continue
            row.cos_trace = float(vector @ trace_vector)
            twin = pick_twin(row.cos_trace, cos_all, eligible, index.ids, f"{event.id}:{row.node_id}")
            graded_docs = dict(docs)
            twin_tokens = contents.tokens(twin) if twin else None
            if twin and twin_tokens is not None:
                graded_docs[f"twin::{twin}"] = twin_tokens
            full = ground_token_sets(trace_tokens, graded_docs, min_containment=min_containment)
            stripped = ground_token_sets(trace_tokens_q, graded_docs, min_containment=min_containment)
            row.containment = full[row.node_id].containment
            row.grounded = full[row.node_id].grounded
            row.grounded_q = stripped[row.node_id].grounded
            if twin and twin_tokens is not None:
                row.has_twin = True
                row.twin_grounded = full[f"twin::{twin}"].grounded
                row.twin_grounded_q = stripped[f"twin::{twin}"].grounded
            else:
                stats["delivered_without_twin"] += 1
    return observations, dict(stats)


# ------------------------------------------------------------------- statistics


def _rate(values: Iterable[bool | float]) -> float | None:
    data = [float(value) for value in values]
    return sum(data) / len(data) if data else None


def cluster_bootstrap(
    rows: Sequence[Observation],
    statistic: Any,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = 7,
) -> tuple[float | None, float | None]:
    """95% percentile CI resampling events (clusters), not rows."""

    if not rows:
        return None, None
    clusters: dict[str, list[Observation]] = defaultdict(list)
    for row in rows:
        clusters[row.event_id].append(row)
    keys = list(clusters)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(rounds):
        sample: list[Observation] = []
        for pick in rng.integers(len(keys), size=len(keys)):
            sample.extend(clusters[keys[int(pick)]])
        value = statistic(sample)
        if value is not None:
            values.append(value)
    if not values:
        return None, None
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def twin_block(rows: Sequence[Observation]) -> dict[str, Any]:
    """Grounded vs placebo-twin rates over rows that have a twin."""

    paired = [row for row in rows if row.closed and row.has_twin]

    def excess(sample: Sequence[Observation]) -> float | None:
        if not sample:
            return None
        return _rate(row.grounded for row in sample) - _rate(row.twin_grounded for row in sample)

    def excess_q(sample: Sequence[Observation]) -> float | None:
        if not sample:
            return None
        return _rate(row.grounded_q for row in sample) - _rate(row.twin_grounded_q for row in sample)

    low, high = cluster_bootstrap(paired, excess)
    low_q, high_q = cluster_bootstrap(paired, excess_q)
    return {
        "n": len(paired),
        "grounded_rate": _rate(row.grounded for row in paired),
        "twin_rate": _rate(row.twin_grounded for row in paired),
        "excess": excess(paired),
        "excess_ci95": [low, high],
        "grounded_rate_query_stripped": _rate(row.grounded_q for row in paired),
        "twin_rate_query_stripped": _rate(row.twin_grounded_q for row in paired),
        "excess_query_stripped": excess_q(paired),
        "excess_query_stripped_ci95": [low_q, high_q],
    }


def class_block(rows: Sequence[Observation]) -> dict[str, Any]:
    def mean_excess(sample: Sequence[Observation]) -> float | None:
        return _rate(row.placebo_excess for row in sample)

    low, high = cluster_bootstrap(rows, mean_excess)
    return {
        "n": len(rows),
        "events": len({row.event_id for row in rows}),
        "lookup_rate": _rate(row.lookup for row in rows),
        "evidence_rate": _rate(row.evidence for row in rows),
        "placebo_subtracted_evidence_excess": mean_excess(rows),
        "placebo_subtracted_evidence_excess_ci95": [low, high],
        "grounded": twin_block(rows),
    }


def auc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """Mann-Whitney AUC with ties counted half."""

    if not positives or not negatives:
        return None
    neg = np.sort(np.asarray(negatives, dtype=float))
    total = 0.0
    for value in positives:
        below = np.searchsorted(neg, value, side="left")
        at_or_below = np.searchsorted(neg, value, side="right")
        total += below + 0.5 * (at_or_below - below)
    return float(total / (len(positives) * len(negatives)))


def permutation_test(
    observations: Sequence[Observation],
    rounds: int = PERMUTATION_ROUNDS,
    seed: int = 11,
) -> dict[str, Any]:
    """Within-event re-assignment of `used` marks to random delivered nodes."""

    by_event: dict[str, list[Observation]] = defaultdict(list)
    for row in observations:
        by_event[row.event_id].append(row)
    strata: list[tuple[int, np.ndarray]] = []
    observed = 0
    for rows in by_event.values():
        used = sum(1 for row in rows if row.mark == "used")
        if not used:
            continue
        evidence = np.array([1 if row.evidence else 0 for row in rows])
        observed += int(sum(1 for row in rows if row.mark == "used" and row.evidence))
        strata.append((used, evidence))
    if not strata:
        return {"events": 0, "observed": 0, "expected": None, "p_value": None}
    rng = np.random.default_rng(seed)
    expected = float(sum(used * evidence.mean() for used, evidence in strata))
    at_least = 0
    for _ in range(rounds):
        total = 0
        for used, evidence in strata:
            total += int(evidence[rng.permutation(evidence.size)[:used]].sum())
        if total >= observed:
            at_least += 1
    return {
        "events": len(strata),
        "observed": observed,
        "expected": expected,
        "p_value": (1 + at_least) / (1 + rounds),
    }


def falsifier(observations: Sequence[Observation]) -> dict[str, Any]:
    used = [row for row in observations if row.mark == "used"]
    events = len({row.event_id for row in used})
    permutation = permutation_test(observations)
    block = class_block(used)
    low = block["placebo_subtracted_evidence_excess_ci95"][0]
    if len(used) < MIN_USED_MARKS or events < MIN_USED_EVENTS:
        verdict = "insufficient"
    elif (
        permutation["p_value"] is not None
        and permutation["p_value"] < ALPHA
        and low is not None
        and low > 0
    ):
        verdict = "pass"
    else:
        verdict = "fail"
    return {
        "test": FALSIFIER_TEXT,
        "used_marks": len(used),
        "used_events": events,
        "permutation": permutation,
        "placebo_subtracted_evidence_excess": block["placebo_subtracted_evidence_excess"],
        "placebo_subtracted_evidence_excess_ci95": block["placebo_subtracted_evidence_excess_ci95"],
        "better_than_random": verdict,
    }


# --------------------------------------------------------------------- rituals


def _closing_work(store: Store, mark: Mark, event: Event) -> bool:
    """Any remember/teach, lookup or other recall between the recall and the mark.

    A mark carried by ``memory_remember``/``memory_teach`` is closing work by
    itself: that write is the work. A mark carried by the next ``memory_recall``
    needs something else in between, so "recall, then recall again marking
    everything" counts as no closing work.
    """

    if (mark.via_tool or "").endswith(("remember", "teach")):
        return True
    transport = mark.transport or event.transport
    if not transport:
        return False
    start, end = event.created_at, mark.marked_at or "9999"
    carrier = mark.source_id
    for created, node_id in store.session_nodes.get(transport, ()):
        if start < created <= end and node_id != carrier:
            return True
    for occurred, _node in store.lookups.get(transport, ()):
        if start <= occurred <= end:
            return True
    for created, event_id in store.session_events.get(transport, ()):
        if start < created < end and event_id not in {event.id, carrier}:
            return True
    return False


def ritual_block(store: Store, marks: Sequence[Mark], observations: Sequence[Observation]) -> dict[str, Any]:
    by_pair = {(row.event_id, row.node_id): row for row in observations}
    marks_by_event: dict[str, list[Mark]] = defaultdict(list)
    for mark in marks:
        marks_by_event[mark.event_id].append(mark)
    all_used_events = 0
    multi_delivered_events = 0
    rank1_only = 0
    events_with_used = 0
    for event_id, items in marks_by_event.items():
        event = store.events.get(event_id)
        if event is None:
            continue
        used_nodes = {mark.node_id for mark in items if mark.mark == "used"}
        if len(event.results) >= 2:
            multi_delivered_events += 1
            if used_nodes >= set(event.results):
                all_used_events += 1
        if used_nodes:
            events_with_used += 1
            ranks = {
                mark.rank if mark.rank is not None else (event.results.index(mark.node_id) if mark.node_id in event.results else -1)
                for mark in items
                if mark.mark == "used"
            }
            if ranks == {0}:
                rank1_only += 1
    no_work = sum(
        1 for mark in marks if mark.event_id in store.events and not _closing_work(store, mark, store.events[mark.event_id])
    )
    session_dup = sum(
        1
        for mark in marks
        if (row := by_pair.get((mark.event_id, mark.node_id))) is not None and row.session_duplicate
    )

    def share(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    result = {
        "all_delivered_marked_used": {
            "events": multi_delivered_events,
            "hits": all_used_events,
            "share": share(all_used_events, multi_delivered_events),
        },
        "rank1_only_marking": {
            "events": events_with_used,
            "hits": rank1_only,
            "share": share(rank1_only, events_with_used),
        },
        "no_closing_work": {"marks": len(marks), "hits": no_work, "share": share(no_work, len(marks))},
        "session_duplicate_marked": {
            "marks": len(marks),
            "hits": session_dup,
            "share": share(session_dup, len(marks)),
        },
    }
    for detector in result.values():
        value = detector["share"]
        detector["flagged"] = value is not None and value > RITUAL_FLAG_SHARE
    return result


# ---------------------------------------------------------------------- report


def reference_rates(observations: Sequence[Observation]) -> dict[str, Any]:
    r1 = [row for row in observations if row.rank == 0]
    top3 = [row for row in observations if row.rank < 3]
    bins: dict[str, Any] = {}
    for label, low, high in (("<0.5", -2.0, 0.5), ("0.5-0.7", 0.5, 0.7), (">=0.7", 0.7, 2.0)):
        subset = [row for row in r1 if row.cos_trace is not None and low <= row.cos_trace < high]
        block = twin_block(subset)
        bins[label] = {key: block[key] for key in ("n", "grounded_rate", "twin_rate", "excess")}
    return {
        "r1": twin_block(r1),
        "top3": twin_block(top3),
        "r1_lookup_rate": _rate(row.lookup for row in r1),
        "top3_lookup_rate": _rate(row.lookup for row in top3),
        "r1_by_cos_trace": bins,
    }


def build_report(
    connection: sqlite3.Connection,
    *,
    host_label: str,
    since: str | None,
    db_label: str,
) -> dict[str, Any]:
    store = load_store(connection, since)
    observations, stats = observe(connection, store)
    report: dict[str, Any] = {
        "host": host_label,
        "db": db_label,
        "since": since,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": {
            "min_containment": MIN_CONTAINMENT,
            "twin_band": TWIN_BAND,
            "memory": "01M3H76JBHEZKJMBPWFVN70SB2",
            "ab_exclusion": "whole transport sessions with an A/B marker (scope project:target|repo|x, "
            "tree-context-ab/fixture/project=target in node context, or A/B-like recall query)",
        },
        "counts": {
            **stats,
            "excluded_ab_events": store.excluded_ab,
            "ab_sessions": len(store.ab_sessions),
            "delivered_pairs": len(observations),
        },
        "reference": reference_rates(observations),
        "reference_by_agent": {},
    }
    by_agent: dict[str, list[Observation]] = defaultdict(list)
    for row in observations:
        by_agent[row.agent].append(row)
    for name in AGENT_TYPES:
        rows = by_agent.get(name, [])
        if rows:
            reference = reference_rates(rows)
            report["reference_by_agent"][name] = {
                "events": len({row.event_id for row in rows}),
                "r1": {key: reference["r1"][key] for key in ("n", "grounded_rate", "twin_rate", "excess")},
                "top3": {key: reference["top3"][key] for key in ("n", "grounded_rate", "twin_rate", "excess")},
                "r1_lookup_rate": reference["r1_lookup_rate"],
            }

    if store.marks is None:
        report["marks"] = {"status": "no marks", "reason": f"table {MARKS_TABLE} is absent"}
        report["falsifier"] = {"test": FALSIFIER_TEXT, "better_than_random": "insufficient", "used_marks": 0}
        return report

    in_window = set(store.window_ids)
    by_pair = {(row.event_id, row.node_id): row for row in observations}
    accepted = [mark for mark in store.marks if mark.accepted]
    scored: list[Mark] = []
    for mark in accepted:
        row = by_pair.get((mark.event_id, mark.node_id))
        if mark.event_id in in_window and row is not None:
            scored.append(mark)
            # 'irrelevant' after 'used' on the same pair keeps the latest claim.
            row.mark = mark.mark
    report["marks"] = {
        "status": "marks" if store.marks else "no marks",
        "total": len(store.marks),
        "accepted": len(accepted),
        "rejected": len(store.marks) - len(accepted),
        "reject_reasons": dict(Counter(mark.reject_reason or "" for mark in store.marks if not mark.accepted)),
        "scored_in_window": len(scored),
        "outside_window_or_ab": len(accepted) - len(scored),
        "by_mark": dict(Counter(mark.mark for mark in scored)),
    }
    used = [row for row in observations if row.mark == "used"]
    irrelevant = [row for row in observations if row.mark == "irrelevant"]
    unmarked = [row for row in observations if row.mark is None]
    marked_events = {row.event_id for row in used + irrelevant}
    unmarked_in_marked = [row for row in unmarked if row.event_id in marked_events]
    report["classes"] = {
        "used": class_block(used),
        "irrelevant": class_block(irrelevant),
        "unmarked": class_block(unmarked),
        "unmarked_in_marked_events": class_block(unmarked_in_marked),
    }

    def diff(sample: Sequence[Observation]) -> float | None:
        a = _rate(row.placebo_excess for row in sample if row.mark == "used")
        b = _rate(row.placebo_excess for row in sample if row.mark == "irrelevant")
        return None if a is None or b is None else a - b

    low, high = cluster_bootstrap(used + irrelevant, diff)
    report["agreement"] = {
        "used_minus_irrelevant_excess": diff(used + irrelevant),
        "used_minus_irrelevant_excess_ci95": [low, high],
        "auc_used_vs_irrelevant": auc(
            [row.evidence_score for row in used], [row.evidence_score for row in irrelevant]
        ),
        "auc_used_vs_unmarked": auc(
            [row.evidence_score for row in used], [row.evidence_score for row in unmarked]
        ),
        "auc_score": "containment in closing trace (0 when unclosed) + 1.0 if looked up",
    }
    report["falsifier"] = falsifier(observations)
    report["ritual"] = ritual_block(store, scored, observations)
    by_agent_marks: dict[str, list[Mark]] = defaultdict(list)
    for mark in scored:
        by_agent_marks[agent_type(mark.agent) if mark.agent else by_pair[(mark.event_id, mark.node_id)].agent].append(mark)
    report["marks_by_agent"] = {}
    for name in AGENT_TYPES:
        items = by_agent_marks.get(name)
        if not items:
            continue
        pairs = {(mark.event_id, mark.node_id) for mark in items}
        rows = [row for row in observations if (row.event_id, row.node_id) in pairs]
        events = {mark.event_id for mark in items}
        agent_rows = [row for row in observations if row.event_id in events or (row.agent == name)]
        report["marks_by_agent"][name] = {
            "marks": len(items),
            "used": class_block([row for row in rows if row.mark == "used"]),
            "irrelevant": class_block([row for row in rows if row.mark == "irrelevant"]),
            "falsifier": falsifier([row for row in agent_rows if row.event_id in events])["better_than_random"],
            "ritual": {
                key: value["share"] for key, value in ritual_block(store, items, observations).items()
            },
        }
    return report


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _ci(pair: Sequence[Any]) -> str:
    if not pair or pair[0] is None:
        return "—"
    return f"[{100 * pair[0]:+.1f}, {100 * pair[1]:+.1f}] pp"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Explicit-feedback agreement check — {report['host']}",
        "",
        f"Store: `{report['db']}` (read-only), since `{report['since']}`, generated {report['generated_at']}.",
        f"Script: `scripts/explicit_feedback_agreement.py`. Placebo method: LM memory {report['method']['memory']} "
        f"(twin = undelivered node created before the event, |cos(twin,T) − cos(n,T)| ≤ {report['method']['twin_band']}, "
        f"mean-pooled chunk vectors; grounding `ground_token_sets` on delivered + twin, threshold {report['method']['min_containment']}).",
        f"A/B exclusion: {report['method']['ab_exclusion']}.",
        "",
        "## Counts",
        "",
        "| item | value |",
        "|---|---|",
    ]
    for key, value in report["counts"].items():
        lines.append(f"| {key} | {value} |")
    reference = report["reference"]
    lines += [
        "",
        "## Placebo reference rates (grounded vs same-cosine twin)",
        "",
        "| slice | pairs | grounded | twin | excess | excess 95% CI | grounded −q | twin −q | excess −q | excess −q 95% CI |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label in ("r1", "top3"):
        block = reference[label]
        lines.append(
            f"| {label} | {block['n']} | {_pct(block['grounded_rate'])} | {_pct(block['twin_rate'])} | "
            f"{_pct(block['excess'])} | {_ci(block['excess_ci95'])} | {_pct(block['grounded_rate_query_stripped'])} | "
            f"{_pct(block['twin_rate_query_stripped'])} | {_pct(block['excess_query_stripped'])} | "
            f"{_ci(block['excess_query_stripped_ci95'])} |"
        )
    lines += [
        "",
        "`−q`: query tokens removed from the closing trace before grading.",
        "",
        f"Lookup rate (same transport, after delivery, or ledger basis lookup): r1 {_pct(reference['r1_lookup_rate'])}, "
        f"top-3 {_pct(reference['top3_lookup_rate'])}. No placebo exists for lookup (a twin is never on the card).",
        "",
        "r1 by cos(r1, closing trace):",
        "",
        "| cos bin | pairs | grounded | twin | excess |",
        "|---|---|---|---|---|",
    ]
    for label, block in reference["r1_by_cos_trace"].items():
        lines.append(
            f"| {label} | {block['n']} | {_pct(block['grounded_rate'])} | {_pct(block['twin_rate'])} | {_pct(block['excess'])} |"
        )
    lines += [
        "",
        "## Reference by agent type",
        "",
        "| agent | events | r1 pairs | r1 grounded | r1 twin | r1 excess | top-3 pairs | top-3 grounded | top-3 twin | top-3 excess | r1 lookup |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, block in report["reference_by_agent"].items():
        r1, top3 = block["r1"], block["top3"]
        lines.append(
            f"| {name} | {block['events']} | {r1['n']} | {_pct(r1['grounded_rate'])} | {_pct(r1['twin_rate'])} | "
            f"{_pct(r1['excess'])} | {top3['n']} | {_pct(top3['grounded_rate'])} | {_pct(top3['twin_rate'])} | "
            f"{_pct(top3['excess'])} | {_pct(block['r1_lookup_rate'])} |"
        )
    lines += ["", "Agent type: recall_events.agent / ambient agent, else the majority `context.agent` of nodes written on the same transport session; `unknown` = nothing recorded.", ""]
    lines += ["## Marks", ""]
    marks = report["marks"]
    if marks.get("status") == "no marks":
        lines.append(f"**No marks.** {marks.get('reason', 'table present but empty')}.")
    else:
        lines.append(
            f"total {marks['total']}, accepted {marks['accepted']}, rejected {marks['rejected']} "
            f"{marks['reject_reasons']}, scored in window {marks['scored_in_window']}, by mark {marks['by_mark']}."
        )
        lines += [
            "",
            "| class | n | events | evidence | lookup | placebo-subtracted evidence excess | 95% CI | grounded | twin | grounded excess |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, block in report["classes"].items():
            grounded = block["grounded"]
            lines.append(
                f"| {name} | {block['n']} | {block['events']} | {_pct(block['evidence_rate'])} | {_pct(block['lookup_rate'])} | "
                f"{_pct(block['placebo_subtracted_evidence_excess'])} | {_ci(block['placebo_subtracted_evidence_excess_ci95'])} | "
                f"{_pct(grounded['grounded_rate'])} | {_pct(grounded['twin_rate'])} | {_pct(grounded['excess'])} |"
            )
        agreement = report["agreement"]
        lines += [
            "",
            f"used − irrelevant excess: {_pct(agreement['used_minus_irrelevant_excess'])} "
            f"{_ci(agreement['used_minus_irrelevant_excess_ci95'])}; AUC used vs irrelevant "
            f"{_fmt(agreement['auc_used_vs_irrelevant'])}, used vs unmarked {_fmt(agreement['auc_used_vs_unmarked'])} "
            f"(score: {agreement['auc_score']}).",
            "",
            "### Ritual detectors",
            "",
            "| detector | base | hits | share | flagged (> 50%) |",
            "|---|---|---|---|---|",
        ]
        for name, block in report["ritual"].items():
            base = block.get("events", block.get("marks"))
            lines.append(f"| {name} | {base} | {block['hits']} | {_pct(block['share'])} | {block['flagged']} |")
        lines += ["", "### By agent type", "", "| agent | marks | used n | used excess | irrelevant n | verdict | all-used | rank1-only | no work | session dup |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, block in report["marks_by_agent"].items():
            ritual = block["ritual"]
            lines.append(
                f"| {name} | {block['marks']} | {block['used']['n']} | {_pct(block['used']['placebo_subtracted_evidence_excess'])} | "
                f"{block['irrelevant']['n']} | {block['falsifier']} | {_pct(ritual['all_delivered_marked_used'])} | "
                f"{_pct(ritual['rank1_only_marking'])} | {_pct(ritual['no_closing_work'])} | {_pct(ritual['session_duplicate_marked'])} |"
            )
    falsifier_block = report["falsifier"]
    lines += [
        "",
        "## Falsifier: better than random",
        "",
        f"> {falsifier_block['test']}",
        "",
        f"**Verdict: `{falsifier_block['better_than_random']}`** (used marks: {falsifier_block.get('used_marks', 0)}).",
    ]
    if "permutation" in falsifier_block:
        permutation = falsifier_block["permutation"]
        lines.append(
            f"Permutation: observed {permutation['observed']} evidence-bearing used marks vs expected "
            f"{_fmt(permutation['expected'])} over {permutation['events']} events, p = {_fmt(permutation['p_value'])}; "
            f"placebo-subtracted excess {_pct(falsifier_block['placebo_subtracted_evidence_excess'])} "
            f"{_ci(falsifier_block['placebo_subtracted_evidence_excess_ci95'])}."
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True, help="live store or snapshot; opened mode=ro")
    parser.add_argument("--host-label", required=True, help="sfx | alt (never merged)")
    parser.add_argument("--since", default=None, help="ISO date/time lower bound on recall_events.created_at")
    parser.add_argument("--json", dest="json_out", default=None, help="write the JSON report here")
    parser.add_argument("--md", dest="md_out", default=None, help="write the markdown report here")
    args = parser.parse_args(argv)
    connection = open_readonly(args.db)
    try:
        report = build_report(connection, host_label=args.host_label, since=args.since, db_label=str(args.db))
    finally:
        connection.close()
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md_out).write_text(markdown, encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
