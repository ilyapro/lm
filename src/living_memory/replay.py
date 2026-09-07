"""Offline recall-replay evaluation harness over recorded ``recall_events``.

This module is the arbiter for ranking/learning changes: it re-ranks the
recorded per-result method scores (bm25/vector/graph/trigger) through the
*real* ranking code path (``MemoryRecallService.rank_candidates``) under
different weight schemes, replays the implicit-feedback reinforcement loop
(``feedback.apply_pending_recall_feedback``) through a pluggable credit rule
to produce per-scope weight trajectories, and reports hit@k / MRR /
graph-unique-contribution metrics for later-confirmed-useful results.

The harness never writes to the source database: connections are opened with
``file:...?mode=ro``. Run it against a snapshot copy of the live DB.

CLI::

    python3 -m living_memory.replay --db PATH --cutoff ISO --report OUT.json

Key design decisions (documented in the generated markdown report as well):

* Labeled events are recall events later consumed by a ``memory_remember``
  (``feedback_trace_id`` set). Because the live consumption loop links *all*
  results of a consumed event, "linked by the consuming trace" is vacuous as
  a usefulness label. The default label is therefore **content grounding**:
  a result is confirmed useful when the IDF-weighted share of its content
  tokens contained in the consuming trace's content is at least
  ``min_containment``. Alternative labels (subsequent re-consumption,
  long-run usefulness) are implemented for sensitivity analysis.
* Re-ranking builds synthetic ``_Candidate`` objects with *neutral* node
  stats (confidence 0.5, usefulness 0, access 0) so ``feedback_weighted_score``
  is a no-op multiplier: replayed rankings isolate the method-mixing policy.
  The ``recorded`` scheme preserves the historical order (which did include
  node-level feedback multipliers at event time) as an anchor.
* The credit replay mirrors ``apply_pending_recall_feedback``: per result at
  0-based rank ``r`` the reinforcement signal is ``max(0.2, 1/(r+1))``,
  applied at the *event's* scope through a pluggable credit rule — the
  ``proportional`` rule delegates to the live ``feedback._method_signals``,
  while ``winner_take_all`` keeps the pre-fix rule frozen for A/B replays,
  and ``grounded_or_lookup`` additionally credits results a same-transport
  ``memory_lookup`` fetched after delivery (``load_lookup_follows``) —
  and the exact ``update_retrieval_weights`` arithmetic including
  ``apply_retrieval_weight_floors``. Updates are ordered
  by ``feedback_applied_at`` (batch-internal order: newest event first, like
  ``pending_recall_events``); updates after ``--cutoff`` are skipped so
  holdout events are ranked under weights frozen at the cutoff.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from living_memory.config import MemoryConfig
from living_memory.feedback import _method_signals, ungrounded_negative_signals
from living_memory.grounding import (
    DEFAULT_MIN_CONTAINMENT,
    build_idf,
    containment as _idf_containment,
    ground_token_sets,
    token_set,
)
from living_memory.models import NODE_LEVELS, Node, RetrievalWeights
from living_memory.retrieval import MemoryRecallService, _Candidate, _is_decision_depth, _parse_depth
from living_memory.scope import ScopePlan
from living_memory.storage import (
    CHUNK_EMBEDDING_TABLE,
    RECALL_LOOKUP_EVENT_TABLE,
    MemoryStore,
    _scope_family,
)

HIT_KS: tuple[int, ...] = (1, 5, 10)
DEFAULT_RECONSUME_MIN_TRACES = 2
DEFAULT_USEFULNESS_THRESHOLD = 0.8
DEFAULT_MIN_SCOPE_EVENTS = 50
#: Same-transport lookup window, in seconds, after a recall event within which
#: an id-fetch of a delivered node counts as a follow. Matches the closure
#: corpus grader (``scripts/usage_signal_corpus.py``) and the baseline.
DEFAULT_LOOKUP_WINDOW_SECONDS = 24 * 3600.0
OTHER_SCOPE_BUCKET = "_other"
LABEL_PROTOCOLS = ("grounded", "reconsumed", "usefulness", "grounded_or_reconsumed")
STATIC_SCHEMES = ("recorded", "live_weights", "floor_defaults", "uniform")
REPLAYED_SCHEME_PREFIX = "replayed"


# ---------------------------------------------------------------------------
# Event loading
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReplayResult:
    """One recorded recall result with its per-method evidence scores."""

    node_id: str
    rank: int
    level: str
    scope: str
    score: float
    bm25_score: float
    vector_score: float
    graph_score: float
    trigger_score: float
    path: tuple[str, ...] = ()
    useful: bool = False
    #: Live-path verdict from ``grounding.ground_results`` (per-event IDF).
    #: Deliberately distinct from ``useful``, which is the harness label under
    #: the configured protocol and, for ``grounded``, uses the whole-corpus
    #: IDF. Credit rules read ``grounded``; metrics read ``useful``.
    grounded: bool = False
    containment: float = 0.0
    #: A same-transport ``memory_lookup`` id-fetch of this node landed after
    #: the event, within the lookup window (``load_lookup_follows``). The
    #: usage signal the ``grounded_or_lookup`` credit rule adds to grounding;
    #: never part of any label.
    looked_up: bool = False

    @property
    def graph_only(self) -> bool:
        """Reachable only via graph evidence (goal-literal definition)."""

        return self.graph_score > 0.0 and self.bm25_score == 0.0 and self.vector_score == 0.0


@dataclass(slots=True)
class ReplayEvent:
    """One recorded recall interaction, parsed for replay."""

    id: str
    created_at: str
    scope: str
    requested_scope: str
    resolved_scopes: tuple[str, ...]
    depth: str | None
    max_results: int
    query: str
    feedback_trace_id: str | None
    feedback_applied_at: str | None
    results: list[ReplayResult]
    #: MCP transport identity stamped on the event; the join key that ties a
    #: later ``memory_lookup`` to this delivery. ``None`` on events recorded
    #: before transports were stamped, which can therefore never be followed.
    transport_session_id: str | None = None

    @property
    def labeled(self) -> bool:
        return self.feedback_trace_id is not None

    @property
    def useful_ids(self) -> set[str]:
        return {result.node_id for result in self.results if result.useful}


def open_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open the source database strictly read-only."""

    connection = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _parse_result(raw: Mapping[str, Any], fallback_rank: int) -> ReplayResult | None:
    node_id = str(raw.get("node_id") or "")
    if not node_id:
        return None
    if any(key not in raw for key in ("bm25_score", "vector_score", "graph_score", "methods")):
        return None
    level = str(raw.get("level") or "trace")
    if level not in NODE_LEVELS:
        level = "trace"
    return ReplayResult(
        node_id=node_id,
        rank=int(raw.get("rank") or fallback_rank),
        level=level,
        scope=str(raw.get("scope") or "global"),
        score=float(raw.get("score") or 0.0),
        bm25_score=float(raw.get("bm25_score") or 0.0),
        vector_score=float(raw.get("vector_score") or 0.0),
        graph_score=float(raw.get("graph_score") or 0.0),
        trigger_score=float(raw.get("trigger_score") or 0.0),
        path=tuple(str(step) for step in raw.get("path") or ()),
    )


def load_replay_events(
    connection: sqlite3.Connection,
    *,
    max_events: int | None = None,
) -> tuple[list[ReplayEvent], dict[str, int]]:
    """Load events in chronological order, skipping unusable ones gracefully."""

    stats = {"total_events": 0, "empty_results": 0, "incomplete_results": 0, "usable_events": 0}
    events: list[ReplayEvent] = []
    # Snapshots taken before transports were stamped lack the column; they
    # load fine and simply never carry a lookup follow.
    columns = {
        str(row["name"]) for row in connection.execute("PRAGMA table_info(recall_events)")
    }
    transport_column = (
        "transport_session_id" if "transport_session_id" in columns else "NULL"
    )
    rows = connection.execute(
        f"""
        SELECT id, created_at, scope, requested_scope, resolved_scopes, depth,
               max_results, query, results, feedback_trace_id, feedback_applied_at,
               {transport_column} AS transport_session_id
        FROM recall_events
        ORDER BY created_at ASC, rowid ASC
        """
    )
    for row in rows:
        stats["total_events"] += 1
        try:
            raw_results = json.loads(row["results"] or "[]")
        except ValueError:
            stats["incomplete_results"] += 1
            continue
        if not raw_results:
            stats["empty_results"] += 1
            continue
        parsed = [_parse_result(raw, index + 1) for index, raw in enumerate(raw_results)]
        if any(result is None for result in parsed):
            stats["incomplete_results"] += 1
            continue
        try:
            resolved = tuple(str(scope) for scope in json.loads(row["resolved_scopes"] or "[]"))
        except ValueError:
            resolved = ()
        events.append(
            ReplayEvent(
                id=str(row["id"]),
                created_at=str(row["created_at"]),
                scope=str(row["scope"]),
                requested_scope=str(row["requested_scope"] or row["scope"]),
                resolved_scopes=resolved or (str(row["scope"]),),
                depth=row["depth"],
                max_results=int(row["max_results"]),
                query=str(row["query"]),
                feedback_trace_id=row["feedback_trace_id"],
                feedback_applied_at=row["feedback_applied_at"],
                results=[result for result in parsed if result is not None],
                transport_session_id=(
                    str(row["transport_session_id"]) or None
                    if row["transport_session_id"] is not None
                    else None
                ),
            )
        )
        stats["usable_events"] += 1
        if max_events is not None and stats["usable_events"] >= max_events:
            break
    return events, stats


def _parse_instant(raw: str) -> datetime:
    """Storage timestamp -> aware UTC datetime.

    ``recall_events.created_at`` is second precision while
    ``recall_lookup_events.occurred_at`` carries microseconds, so the two are
    only comparable as parsed datetimes, never as strings.
    """

    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def load_lookup_follows(
    connection: sqlite3.Connection,
    events: Sequence[ReplayEvent],
    *,
    window_seconds: float = DEFAULT_LOOKUP_WINDOW_SECONDS,
    not_after: str | None = None,
) -> dict[str, Any]:
    """Set ``result.looked_up`` in place from the ``recall_lookup_events`` ledger.

    A result is *followed* when a later ``memory_lookup`` id-fetch named its
    node from the same transport session as the event, strictly after the
    event's ``created_at`` and within ``window_seconds`` of it — the
    same-transport / 24 h definition ``scripts/usage_signal_corpus.py`` grades
    closures with. Events without a transport identity are never followed: the
    transport is the only key that ties a fetch to a delivery. ``not_after``
    (a storage timestamp, in practice the replay cutoff) additionally rejects
    lookups after that instant, so a post-cutoff fetch can never leak into the
    pre-cutoff weight trajectory of an event that closed before the cutoff.

    Older snapshots predate the ledger (the table arrived with the
    lookup-consumed recall-map signal); such a database yields zero follows
    and ``table_present=False`` instead of an exception.

    The ``followed_and_grounded`` / ``followed_only`` counters read
    ``result.grounded`` as currently set, so run this after
    ``apply_grounding`` for the overlap to mean anything. Counters prefixed
    ``labeled_`` cover only closed events, the ones a credit rule ever sees.
    """

    for event in events:
        for result in event.results:
            result.looked_up = False
    stats: dict[str, Any] = {
        "table_present": False,
        "window_seconds": float(window_seconds),
        "not_after": normalize_cutoff(not_after) if not_after else None,
        "rows": 0,
        "rows_with_transport": 0,
        "unparsable_rows": 0,
        "lookup_events": 0,
        "first_lookup_at": None,
        "last_lookup_at": None,
        "events_with_transport": 0,
        "events_with_follow": 0,
        "labeled_events_with_follow": 0,
        "results_followed": 0,
        "labeled_results_followed": 0,
        "followed_and_grounded": 0,
        "followed_only": 0,
        # Results whose only in-window fetch came after ``not_after``.
        "not_after_excluded_results": 0,
        # Results fetched at exactly ``created_at`` (second precision): not a
        # follow under the strict "after" rule, counted so the choice is visible.
        "same_instant_results": 0,
    }
    present = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (RECALL_LOOKUP_EVENT_TABLE,),
    ).fetchone()
    if present is None:
        return stats
    stats["table_present"] = True

    follows: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    lookup_ids: set[str] = set()
    earliest: datetime | None = None
    latest: datetime | None = None
    rows = connection.execute(
        f"SELECT lookup_event_id, node_id, occurred_at, transport_session_id "
        f"FROM {RECALL_LOOKUP_EVENT_TABLE}"
    )
    for row in rows:
        stats["rows"] += 1
        lookup_ids.add(str(row["lookup_event_id"]))
        try:
            instant = _parse_instant(str(row["occurred_at"]))
        except ValueError:
            stats["unparsable_rows"] += 1
            continue
        earliest = instant if earliest is None or instant < earliest else earliest
        latest = instant if latest is None or instant > latest else latest
        transport = row["transport_session_id"]
        if transport is None or not str(transport):
            continue
        stats["rows_with_transport"] += 1
        follows[(str(transport), str(row["node_id"]))].append(instant)
    stats["lookup_events"] = len(lookup_ids)
    stats["first_lookup_at"] = _format_instant(earliest)
    stats["last_lookup_at"] = _format_instant(latest)

    window = timedelta(seconds=float(window_seconds))
    limit = _parse_instant(not_after) if not_after else None
    for event in events:
        if not event.transport_session_id:
            continue
        stats["events_with_transport"] += 1
        try:
            created = _parse_instant(event.created_at)
        except ValueError:
            continue
        deadline = created + window
        followed_here = 0
        for result in event.results:
            instants = follows.get((event.transport_session_id, result.node_id))
            if not instants:
                continue
            in_window = False
            admitted = False
            for instant in instants:
                if instant == created:
                    stats["same_instant_results"] += 1
                    continue
                if instant < created or instant > deadline:
                    continue
                in_window = True
                if limit is None or instant <= limit:
                    admitted = True
                    break
            if in_window and not admitted:
                stats["not_after_excluded_results"] += 1
            if not admitted:
                continue
            result.looked_up = True
            followed_here += 1
            stats["results_followed"] += 1
            stats["followed_and_grounded"] += int(result.grounded)
            stats["followed_only"] += int(not result.grounded)
            if event.labeled:
                stats["labeled_results_followed"] += 1
        if followed_here:
            stats["events_with_follow"] += 1
            if event.labeled:
                stats["labeled_events_with_follow"] += 1
    return stats


def _format_instant(instant: datetime | None) -> str | None:
    if instant is None:
        return None
    return instant.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LabelConfig:
    """Parameters of the confirmed-useful labeling protocol."""

    protocol: str = "grounded"
    min_containment: float = DEFAULT_MIN_CONTAINMENT
    reconsume_min_traces: int = DEFAULT_RECONSUME_MIN_TRACES
    usefulness_threshold: float = DEFAULT_USEFULNESS_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "min_containment": self.min_containment,
            "reconsume_min_traces": self.reconsume_min_traces,
            "usefulness_threshold": self.usefulness_threshold,
        }


@dataclass(slots=True)
class LabelData:
    """Corpus-derived inputs shared by all labeling protocols."""

    token_sets: dict[str, frozenset[str]]
    idf: dict[str, float]
    usefulness: dict[str, float]
    consumers: dict[str, list[tuple[str, str]]]
    missing_traces: int = 0
    missing_nodes: int = 0


def _fetch_content_rows(
    connection: sqlite3.Connection, node_ids: Iterable[str]
) -> dict[str, tuple[str, float]]:
    found: dict[str, tuple[str, float]] = {}
    ids = list(dict.fromkeys(node_ids))
    for start in range(0, len(ids), 900):
        chunk = ids[start : start + 900]
        placeholders = ",".join("?" * len(chunk))
        rows = connection.execute(
            f"SELECT id, content, usefulness_score FROM nodes WHERE id IN ({placeholders})",
            chunk,
        )
        for row in rows:
            found[str(row["id"])] = (str(row["content"]), float(row["usefulness_score"]))
    return found


def build_label_data(
    connection: sqlite3.Connection, events: Sequence[ReplayEvent]
) -> LabelData:
    """Fetch node/trace contents and build the IDF + re-consumption indexes."""

    labeled = [event for event in events if event.labeled]
    result_ids = {result.node_id for event in labeled for result in event.results}
    trace_ids = {event.feedback_trace_id for event in labeled if event.feedback_trace_id}
    node_rows = _fetch_content_rows(connection, result_ids)
    trace_rows = _fetch_content_rows(connection, trace_ids)

    token_sets: dict[str, frozenset[str]] = {}
    usefulness: dict[str, float] = {}
    for node_id, (content, score) in node_rows.items():
        token_sets[node_id] = token_set(content)
        usefulness[node_id] = score
    for trace_id, (content, _score) in trace_rows.items():
        token_sets.setdefault(trace_id, token_set(content))

    idf = build_idf(token_sets.values())

    consumers: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for event in labeled:
        trace_id = event.feedback_trace_id or ""
        for node_id in {result.node_id for result in event.results}:
            consumers[node_id].append((event.created_at, trace_id))
    for entries in consumers.values():
        entries.sort()

    return LabelData(
        token_sets=token_sets,
        idf=idf,
        usefulness=usefulness,
        consumers=dict(consumers),
        missing_traces=len(trace_ids) - len(trace_rows),
        missing_nodes=len(result_ids) - len(node_rows),
    )


def _containment(data: LabelData, node_id: str, trace_id: str | None) -> float:
    """IDF-weighted share of the node's tokens present in the consuming trace.

    Whole-corpus IDF: the label view, which sees every document at once.
    ``apply_grounding`` computes the live path's per-event view separately.
    """

    if not trace_id:
        return 0.0
    return _idf_containment(
        data.token_sets.get(node_id), data.token_sets.get(trace_id), data.idf
    )


def apply_grounding(
    events: Sequence[ReplayEvent],
    data: LabelData,
    min_containment: float = DEFAULT_MIN_CONTAINMENT,
) -> dict[str, Any]:
    """Set ``result.grounded`` exactly as the live write path would.

    Replays ``grounding.ground_results`` per consumed event over the same
    documents ``feedback.apply_pending_recall_feedback`` holds at that moment
    (the consuming trace plus that event's result nodes), so a replayed credit
    rule gates on the live verdict rather than on the harness label. Returns
    the agreement between the two views, which is what licenses reading the
    A/B as a statement about the live rule.
    """

    graded = 0
    grounded_results = 0
    agree = 0
    for event in events:
        for result in event.results:
            result.grounded = False
            result.containment = 0.0
        if not event.labeled or not event.feedback_trace_id:
            continue
        trace_tokens = data.token_sets.get(event.feedback_trace_id)
        if trace_tokens is None:
            continue
        # The corpus token sets are already tokenized, so grade through the
        # pre-tokenized entry point rather than re-tokenizing the contents.
        result_tokens = {
            result.node_id: tokens
            for result in event.results
            if (tokens := data.token_sets.get(result.node_id)) is not None
        }
        verdicts = ground_token_sets(
            trace_tokens, result_tokens, min_containment=min_containment
        )
        for result in event.results:
            verdict = verdicts.get(result.node_id)
            if verdict is None:
                continue
            result.containment = verdict.containment
            result.grounded = verdict.grounded
            graded += 1
            grounded_results += int(result.grounded)
            corpus_grounded = (
                _containment(data, result.node_id, event.feedback_trace_id) >= min_containment
            )
            agree += int(corpus_grounded == result.grounded)
    return {
        "min_containment": min_containment,
        "graded_results": graded,
        "grounded_results": grounded_results,
        "grounded_share": _ratio(grounded_results, graded),
        "corpus_idf_agreement": _ratio(agree, graded),
    }


def _reconsumed_later(
    data: LabelData, event: ReplayEvent, node_id: str, min_traces: int
) -> bool:
    """Node appears in later consumed events from at least ``min_traces`` other traces."""

    seen: set[str] = set()
    for created_at, trace_id in data.consumers.get(node_id, ()):
        if created_at > event.created_at and trace_id != event.feedback_trace_id:
            seen.add(trace_id)
            if len(seen) >= min_traces:
                return True
    return False


def _label_result(
    data: LabelData, config: LabelConfig, event: ReplayEvent, result: ReplayResult
) -> bool:
    grounded = (
        _containment(data, result.node_id, event.feedback_trace_id) >= config.min_containment
    )
    if config.protocol == "grounded":
        return grounded
    if config.protocol == "reconsumed":
        return _reconsumed_later(data, event, result.node_id, config.reconsume_min_traces)
    if config.protocol == "usefulness":
        return data.usefulness.get(result.node_id, 0.0) >= config.usefulness_threshold
    if config.protocol == "grounded_or_reconsumed":
        return grounded or _reconsumed_later(
            data, event, result.node_id, config.reconsume_min_traces
        )
    raise ValueError(f"unknown label protocol: {config.protocol}")


def apply_labels(
    events: Sequence[ReplayEvent], data: LabelData, config: LabelConfig
) -> dict[str, Any]:
    """Set ``result.useful`` in place and return labeling + discrimination stats."""

    labeled_events = 0
    labeled_results = 0
    useful_results = 0
    events_with_useful = 0
    for event in events:
        if not event.labeled:
            for result in event.results:
                result.useful = False
            continue
        labeled_events += 1
        positives = 0
        for result in event.results:
            result.useful = _label_result(data, config, event, result)
            positives += int(result.useful)
        labeled_results += len(event.results)
        useful_results += positives
        events_with_useful += int(positives > 0)

    summary = {
        "config": config.to_dict(),
        "labeled_events": labeled_events,
        "labeled_results": labeled_results,
        "useful_results": useful_results,
        "events_with_useful": events_with_useful,
        "missing_consuming_traces": data.missing_traces,
        "missing_result_nodes": data.missing_nodes,
        "discrimination": label_discrimination(events),
    }
    return summary


def label_discrimination(events: Sequence[ReplayEvent]) -> dict[str, Any]:
    """How selective the labels are inside multi-result labeled events.

    The consumption loop links every result of a consumed event, so a
    meaningful ``strict_subset_fraction`` is what makes the ranking metrics
    non-vacuous.
    """

    multi = strict = all_useful = none_useful = 0
    share_sum = 0.0
    for event in events:
        if not event.labeled or len(event.results) < 2:
            continue
        multi += 1
        positives = sum(1 for result in event.results if result.useful)
        share_sum += positives / len(event.results)
        if positives == 0:
            none_useful += 1
        elif positives == len(event.results):
            all_useful += 1
        else:
            strict += 1
    return {
        "multi_result_events": multi,
        "strict_subset_events": strict,
        "strict_subset_fraction": _ratio(strict, multi),
        "all_useful_fraction": _ratio(all_useful, multi),
        "none_useful_fraction": _ratio(none_useful, multi),
        "mean_useful_share": _ratio(share_sum, multi),
    }


# ---------------------------------------------------------------------------
# Weight schemes and the ranking shim
# ---------------------------------------------------------------------------


class _NullEmbedder:
    """Replay never embeds; the ranking path must not ask for embeddings."""

    def embed(self, text: str) -> list[float]:  # pragma: no cover - guard only
        raise RuntimeError("replay harness must not compute embeddings")


class _ShimCursor:
    def __init__(self, rows: list[Mapping[str, str]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[Mapping[str, str]]:
        return self._rows


class _ShimConnection:
    """Serves the single supersedes query issued by ``rank_candidates``."""

    def __init__(self, supersedes_pairs: Iterable[tuple[str, str]]) -> None:
        self._rows = [
            {"source_id": source, "target_id": target} for source, target in supersedes_pairs
        ]

    def execute(self, sql: str, *params: Any) -> _ShimCursor:
        if "supersedes" not in sql:  # pragma: no cover - guard only
            raise RuntimeError(f"unexpected query through replay shim: {sql}")
        return _ShimCursor(self._rows)


class _SchemeStore:
    """Duck-typed ``MemoryStore`` face used by ``MemoryRecallService.rank_candidates``."""

    def __init__(
        self,
        resolver: Callable[[str], RetrievalWeights],
        supersedes_pairs: Iterable[tuple[str, str]] = (),
    ) -> None:
        self._resolver = resolver
        self.connection = _ShimConnection(supersedes_pairs)

    def get_retrieval_weights(self, scope: str) -> RetrievalWeights:
        return self._resolver(scope)


def make_ranking_service(
    resolver: Callable[[str], RetrievalWeights],
    supersedes_pairs: Iterable[tuple[str, str]] = (),
) -> MemoryRecallService:
    """Real ranking service over a synthetic weights source."""

    return MemoryRecallService(
        _SchemeStore(resolver, supersedes_pairs),  # type: ignore[arg-type]
        embedder=_NullEmbedder(),  # type: ignore[arg-type]
    )


def static_resolver(weights_by_scope: Mapping[str, RetrievalWeights]) -> Callable[[str], RetrievalWeights]:
    """Scope -> weights with the live fallback chain (exact, family, default)."""

    def resolve(scope: str) -> RetrievalWeights:
        for candidate in (scope, _scope_family(scope), "default"):
            weights = weights_by_scope.get(candidate)
            if weights is not None:
                return weights
        return RetrievalWeights(scope=scope, bm25=1.0, vector=0.0, graph=0.0)

    return resolve


def load_live_weights(connection: sqlite3.Connection) -> dict[str, RetrievalWeights]:
    weights: dict[str, RetrievalWeights] = {}
    for row in connection.execute("SELECT * FROM retrieval_weights"):
        weights[str(row["scope"])] = RetrievalWeights(
            scope=str(row["scope"]),
            bm25=float(row["bm25"]),
            vector=float(row["vector"]),
            graph=float(row["graph"]),
            learning_rate=float(row["learning_rate"]),
            updated_at=str(row["updated_at"]),
        )
    return weights


def load_supersedes_pairs(connection: sqlite3.Connection) -> list[tuple[str, str]]:
    rows = connection.execute("SELECT source_id, target_id FROM connections WHERE type = 'supersedes'")
    return [(str(row["source_id"]), str(row["target_id"])) for row in rows]


class _FloorShim:
    """Duck-typed ``self`` for reusing ``MemoryStore.apply_retrieval_weight_floors``."""

    _lift_retrieval_weight = staticmethod(MemoryStore._lift_retrieval_weight)

    def __init__(self, config: MemoryConfig, evidence: Callable[[str], tuple[bool, bool]]) -> None:
        self.config = config
        self._evidence = evidence

    def _has_vector_evidence(self, scope: str) -> bool:
        return self._evidence(scope)[0]

    def _has_graph_evidence(self, scope: str) -> bool:
        return self._evidence(scope)[1]


def snapshot_evidence(connection: sqlite3.Connection) -> Callable[[str], tuple[bool, bool]]:
    """Evidence gates evaluated against the snapshot (current corpus state).

    Historical gate values are not recorded; for active scopes they flip to
    True early in a scope's life, so the snapshot state is a close stand-in.
    """

    cache: dict[str, tuple[bool, bool]] = {}
    # Mirror ``MemoryStore._has_vector_evidence`` across both storage shapes:
    # pre-v6 snapshots keep a JSON ``nodes.embedding`` column, later ones (the
    # 2026-09 snapshots) dropped it in favour of ``node_chunk_embeddings``.
    node_columns = {
        str(row["name"]) for row in connection.execute("PRAGMA table_info(nodes)")
    }
    embedding_column = "embedding" in node_columns
    chunk_table = (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (CHUNK_EMBEDDING_TABLE,),
        ).fetchone()
        is not None
    )

    def has_vector(scope: str) -> bool:
        if embedding_column:
            row = connection.execute(
                "SELECT 1 FROM nodes WHERE scope = ? AND decayed = 0 "
                "AND embedding IS NOT NULL LIMIT 1",
                (scope,),
            ).fetchone()
            if row is not None:
                return True
        if not chunk_table:
            return False
        row = connection.execute(
            f"""
            SELECT 1
            FROM {CHUNK_EMBEDDING_TABLE} c
            JOIN nodes n ON n.id = c.node_id
            WHERE n.scope = ? AND n.decayed = 0
            LIMIT 1
            """,
            (scope,),
        ).fetchone()
        return row is not None

    def evidence(scope: str) -> tuple[bool, bool]:
        cached = cache.get(scope)
        if cached is not None:
            return cached
        vector_row = has_vector(scope)
        graph_row = connection.execute(
            """
            SELECT 1
            FROM connections c
            JOIN nodes source ON source.id = c.source_id
            JOIN nodes target ON target.id = c.target_id
            WHERE source.decayed = 0 AND target.decayed = 0
              AND (source.scope = ? OR target.scope = ?)
            LIMIT 1
            """,
            (scope, scope),
        ).fetchone()
        result = (vector_row, graph_row is not None)
        cache[scope] = result
        return result

    return evidence


def assumed_evidence(_scope: str) -> tuple[bool, bool]:
    return (True, True)


def floor_default_weights(
    config: MemoryConfig, evidence: Callable[[str], tuple[bool, bool]]
) -> Callable[[str], RetrievalWeights]:
    """Configured family defaults with the scope-family floors applied."""

    shim = _FloorShim(config, evidence)
    cache: dict[str, RetrievalWeights] = {}

    def resolve(scope: str) -> RetrievalWeights:
        cached = cache.get(scope)
        if cached is not None:
            return cached
        family = _scope_family(scope)
        base = config.retrieval_weights.get(family) or config.retrieval_weights["default"]
        weights = RetrievalWeights(
            scope=scope,
            bm25=base.bm25,
            vector=base.vector,
            graph=base.graph,
            learning_rate=base.learning_rate,
        )
        floored = MemoryStore.apply_retrieval_weight_floors(shim, scope, weights)  # type: ignore[arg-type]
        cache[scope] = floored
        return floored

    return resolve


def explicit_weights(bm25: float, vector: float, graph: float) -> Callable[[str], RetrievalWeights]:
    def resolve(scope: str) -> RetrievalWeights:
        return RetrievalWeights(scope=scope, bm25=bm25, vector=vector, graph=graph)

    return resolve


# ---------------------------------------------------------------------------
# Credit rules and the weight-trajectory replay
# ---------------------------------------------------------------------------


#: ``None`` means "this result earns no update at all" — the rule declined to
#: assign credit, and the live path would not have called
#: ``update_retrieval_weights`` either. Returning all-zero signals is not the
#: same thing: it still counts as an update in the trajectory.
CreditRule = Callable[[ReplayResult, float], dict[str, float] | None]


def winner_take_all_credit(result: ReplayResult, signal: float) -> dict[str, float]:
    """The pre-fix live rule, frozen verbatim for A/B replays.

    Until the proportional fix this was ``feedback._method_signals``: reward
    only the dominant method on positive feedback and penalize the other two
    by ``-0.25 * signal`` each; on negative feedback blame the dominant by
    ``-|signal|`` and compensate the others by ``+0.15 * |signal|``; treat
    zero evidence as bm25-dominant. Kept here (and only here) so replays can
    reproduce the historical weight trajectories.
    """

    components = {
        "bm25": max(0.0, result.bm25_score),
        "vector": max(0.0, result.vector_score),
        "graph": max(0.0, result.graph_score),
    }
    dominant = max(components, key=components.get)
    if components[dominant] <= 0.0:
        dominant = "bm25"

    if signal >= 0:
        signals = {method: -0.25 * signal for method in components}
        signals[dominant] = signal
    else:
        positive = abs(signal)
        signals = {method: 0.15 * positive for method in components}
        signals[dominant] = -positive
    return signals


def proportional_credit(result: ReplayResult, signal: float) -> dict[str, float]:
    """The live rule: delegate to ``feedback._method_signals`` verbatim."""

    synthetic = SimpleNamespace(
        bm25_score=result.bm25_score,
        vector_score=result.vector_score,
        graph_score=result.graph_score,
    )
    return _method_signals(synthetic, signal)


def grounded_credit(result: ReplayResult, signal: float) -> dict[str, float] | None:
    """Proportional credit, but only for results the consuming trace used.

    The candidate live rule under ``LM_RECALL_CREDIT_POLICY=grounded``:
    ungrounded results are *neutral* — no reinforcement, no blame — so a
    delivery the agent ignored neither helps nor hurts the node or the
    channel that surfaced it.
    """

    if not result.grounded:
        return None
    return proportional_credit(result, signal)


def grounded_negative_credit(result: ReplayResult, signal: float) -> dict[str, float] | None:
    """Grounded results earn credit; ungrounded ones take a damped penalty.

    The candidate live rule under ``LM_RECALL_CREDIT_POLICY=grounded_negative``:
    the other arm of the A/B, treating "delivered and not used" as weak
    evidence against the channel that surfaced it, scaled by
    ``feedback.UNGROUNDED_NEGATIVE_FACTOR``.
    """

    if result.grounded:
        return proportional_credit(result, signal)
    synthetic = SimpleNamespace(
        bm25_score=result.bm25_score,
        vector_score=result.vector_score,
        graph_score=result.graph_score,
    )
    return ungrounded_negative_signals(synthetic, signal)


def grounded_or_lookup_credit(result: ReplayResult, signal: float) -> dict[str, float] | None:
    """Proportional credit for results the trace grounded *or* the agent fetched.

    Mirrors the live policy pair ``LM_RECALL_CREDIT_POLICY=grounded`` plus
    ``LM_LOOKUP_CREDIT_POLICY=delivered``: a delivered result earns credit
    when the closing trace grounds it (``result.grounded``) or when a
    same-transport ``memory_lookup`` fetched it after delivery
    (``result.looked_up``, set by ``load_lookup_follows``); everything else
    stays neutral. One proportional credit per result whichever signal fired,
    which is the live ledger's at-most-once-per-(event, node) rule, so the
    overlap between the two signals is deduplicated by construction.

    Timing caveat: the replay applies lookup credit at the event's
    reinforcement instant (``feedback_applied_at``, when the closing trace
    lands), not at lookup time as the live path does, and events that never
    closed are never reinforced here even though the live path credits their
    lookups. The A/B therefore measures lookup credit on the closed-event
    stream only.
    """

    if not (result.grounded or result.looked_up):
        return None
    return proportional_credit(result, signal)


CREDIT_RULES: dict[str, CreditRule] = {
    "winner_take_all": winner_take_all_credit,
    "proportional": proportional_credit,
    "grounded": grounded_credit,
    "grounded_negative": grounded_negative_credit,
    "grounded_or_lookup": grounded_or_lookup_credit,
}


class WeightTrajectory:
    """In-memory mirror of the per-scope learned weights under a credit rule.

    Mirrors ``MemoryStore.update_retrieval_weights``: additive
    ``learning_rate * signal`` per channel, clamp at zero, normalize, then
    ``apply_retrieval_weight_floors`` (reused from ``MemoryStore``), written to
    the exact event scope. Only implicit consumption feedback is replayed;
    explicit ``memory_teach``/direct feedback and node usefulness updates are
    not part of the recorded corpus.
    """

    def __init__(
        self,
        config: MemoryConfig,
        *,
        evidence: Callable[[str], tuple[bool, bool]] = assumed_evidence,
        snapshot_learning_rates: Mapping[str, float] | None = None,
    ) -> None:
        self._shim = _FloorShim(config, evidence)
        self._snapshot_learning_rates = dict(snapshot_learning_rates or {})
        self.weights: dict[str, RetrievalWeights] = {
            scope: RetrievalWeights(
                scope=scope,
                bm25=weight_config.bm25,
                vector=weight_config.vector,
                graph=weight_config.graph,
                learning_rate=weight_config.learning_rate,
            )
            for scope, weight_config in config.retrieval_weights.items()
        }
        self.update_counts: Counter[str] = Counter()

    def resolve(self, scope: str) -> RetrievalWeights:
        for candidate in (scope, _scope_family(scope), "default"):
            weights = self.weights.get(candidate)
            if weights is not None:
                return weights
        return RetrievalWeights(scope=scope, bm25=1.0, vector=0.0, graph=0.0)

    def apply_signals(self, scope: str, signals: Mapping[str, float]) -> RetrievalWeights:
        current = self.resolve(scope)
        learning_rate = current.learning_rate
        if scope not in self.weights:
            learning_rate = self._snapshot_learning_rates.get(scope, learning_rate)
        updated = RetrievalWeights(
            scope=scope,
            bm25=max(0.0, current.bm25 + learning_rate * signals.get("bm25", 0.0)),
            vector=max(0.0, current.vector + learning_rate * signals.get("vector", 0.0)),
            graph=max(0.0, current.graph + learning_rate * signals.get("graph", 0.0)),
            learning_rate=learning_rate,
        ).normalized()
        floored = MemoryStore.apply_retrieval_weight_floors(self._shim, scope, updated)  # type: ignore[arg-type]
        floored = RetrievalWeights(
            scope=scope,
            bm25=floored.bm25,
            vector=floored.vector,
            graph=floored.graph,
            learning_rate=learning_rate,
        )
        self.weights[scope] = floored
        self.update_counts[scope] += 1
        return floored

    def reinforce_event(self, event: ReplayEvent, rule: CreditRule) -> None:
        """Apply the rank-decayed consumption reinforcement for one event."""

        for index, result in enumerate(event.results):
            if event.feedback_trace_id and result.node_id == event.feedback_trace_id:
                continue
            signal = max(0.2, 1.0 / (index + 1))
            signals = rule(result, signal)
            if signals is None:
                continue
            self.apply_signals(event.scope, signals)


def reinforcement_order(events: Sequence[ReplayEvent]) -> list[tuple[str, ReplayEvent]]:
    """Consumption updates ordered as the live loop applied them.

    ``pending_recall_events`` hands events to one consuming trace newest
    first, so batches sharing (applied_at, trace) replay in descending
    ``created_at`` order.
    """

    labeled = [event for event in events if event.labeled]
    # Newest-first within a batch, then a stable sort by (applied_at, trace)
    # groups batches chronologically while preserving that inner order.
    labeled.sort(key=lambda event: event.created_at, reverse=True)
    labeled.sort(
        key=lambda event: (
            event.feedback_applied_at or event.created_at,
            event.feedback_trace_id or "",
        )
    )
    return [(event.feedback_applied_at or event.created_at, event) for event in labeled]


# ---------------------------------------------------------------------------
# Re-ranking and metrics
# ---------------------------------------------------------------------------


def build_candidates(event: ReplayEvent, *, zero_graph: bool = False) -> dict[str, _Candidate]:
    """Synthetic candidates from recorded method scores with neutral node stats."""

    candidates: dict[str, _Candidate] = {}
    for result in event.results:
        node = Node(
            id=result.node_id,
            level=result.level,  # type: ignore[arg-type]
            content="",
            scope=result.scope,
            timestamp=event.created_at,
        )
        candidates[result.node_id] = _Candidate(
            node=node,
            bm25_score=result.bm25_score,
            vector_score=result.vector_score,
            graph_score=0.0 if zero_graph else result.graph_score,
            trigger_score=result.trigger_score,
            path=result.path,
        )
    return candidates


def rerank_event(
    service: MemoryRecallService, event: ReplayEvent, *, zero_graph: bool = False
) -> list[str]:
    """Rank recorded candidates through the real ``rank_candidates`` path."""

    plan = ScopePlan(
        requested_scope=event.requested_scope,
        scopes=event.resolved_scopes,
    )
    _graph_depth, causal_mode = _parse_depth(event.depth, event.query)
    decision_mode = _is_decision_depth(event.depth)
    ranked = service.rank_candidates(
        build_candidates(event, zero_graph=zero_graph),
        plan,
        causal_mode=causal_mode,
        decision_mode=decision_mode,
    )
    return [ranked_result.node.id for ranked_result in ranked]


def recorded_order(event: ReplayEvent, *, zero_graph: bool = False) -> list[str]:
    """The historical ranking; the zeroed variant drops graph-only candidates."""

    ordered = sorted(event.results, key=lambda result: result.rank)
    if not zero_graph:
        return [result.node_id for result in ordered]
    survivors = []
    for result in ordered:
        has_non_graph = (
            result.bm25_score > 0.0
            or result.vector_score > 0.0
            or (result.level == "schema" and result.trigger_score > 0.0)
        )
        if has_non_graph:
            survivors.append(result.node_id)
    return survivors


class MetricAccumulator:
    """Streaming hit@k / MRR / graph-contribution counters for one bucket."""

    def __init__(self) -> None:
        self.events = 0
        self.events_with_useful = 0
        self.hits: Counter[int] = Counter()
        # Per-event (event_id, reciprocal_rank, hit@5) for events that have at
        # least one confirmed-useful result. Aggregates alone cannot say
        # whether a 0.002 metric gap between two schemes is signal, so this
        # keeps the paired samples a significance test needs.
        self.per_event: list[tuple[str, float, float]] = []
        self.mrr_sum = 0.0
        self.useful_results = 0
        self.graph_only_useful = 0
        self.zero_hits5 = 0
        self.zero_mrr_sum = 0.0
        self.zero_in_top: Counter[str] = Counter()
        self.zero_dropped: Counter[str] = Counter()

    def add_event(self, event: ReplayEvent, order: Sequence[str], zero_order: Sequence[str]) -> None:
        useful = event.useful_ids
        self.events += 1
        self.useful_results += len(useful)
        self.graph_only_useful += sum(
            1 for result in event.results if result.useful and result.graph_only
        )
        if not useful:
            return
        self.events_with_useful += 1

        first_rank = next(
            (index + 1 for index, node_id in enumerate(order) if node_id in useful), None
        )
        if first_rank is not None:
            self.mrr_sum += 1.0 / first_rank
            for k in HIT_KS:
                if first_rank <= k:
                    self.hits[k] += 1
        self.per_event.append(
            (
                event.id,
                1.0 / first_rank if first_rank is not None else 0.0,
                1.0 if first_rank is not None and first_rank <= 5 else 0.0,
            )
        )

        zero_first = next(
            (index + 1 for index, node_id in enumerate(zero_order) if node_id in useful), None
        )
        if zero_first is not None:
            self.zero_mrr_sum += 1.0 / zero_first
            if zero_first <= 5:
                self.zero_hits5 += 1

        cuts = {"top1": 1, "top5": 5, "top_max": max(1, event.max_results)}
        zero_positions = {node_id: index + 1 for index, node_id in enumerate(zero_order)}
        for name, k in cuts.items():
            for node_id in list(order)[:k]:
                if node_id not in useful:
                    continue
                self.zero_in_top[name] += 1
                zero_rank = zero_positions.get(node_id)
                if zero_rank is None or zero_rank > k:
                    self.zero_dropped[name] += 1
        for node_id in order:
            if node_id in useful:
                self.zero_in_top["ranked"] += 1
                if node_id not in zero_positions:
                    self.zero_dropped["ranked"] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": self.events,
            "events_with_useful": self.events_with_useful,
            "useful_results": self.useful_results,
            "hit@1": _ratio(self.hits[1], self.events_with_useful),
            "hit@5": _ratio(self.hits[5], self.events_with_useful),
            "hit@10": _ratio(self.hits[10], self.events_with_useful),
            "mrr": _ratio(self.mrr_sum, self.events_with_useful),
            "graph_unique_share": _ratio(self.graph_only_useful, self.useful_results),
            "graph_zeroed": {
                "hit@5": _ratio(self.zero_hits5, self.events_with_useful),
                "mrr": _ratio(self.zero_mrr_sum, self.events_with_useful),
                "dropped_top1": _ratio(self.zero_dropped["top1"], self.zero_in_top["top1"]),
                "dropped_top5": _ratio(self.zero_dropped["top5"], self.zero_in_top["top5"]),
                "dropped_top_max": _ratio(
                    self.zero_dropped["top_max"], self.zero_in_top["top_max"]
                ),
                "dropped_entirely": _ratio(
                    self.zero_dropped["ranked"], self.zero_in_top["ranked"]
                ),
            },
        }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


@dataclass(slots=True)
class SchemeReport:
    """Nested accumulator tree: split -> (overall bucket, per-scope buckets)."""

    splits: dict[str, MetricAccumulator] = field(default_factory=dict)
    per_scope: dict[str, dict[str, MetricAccumulator]] = field(default_factory=dict)

    def add(
        self,
        splits: Iterable[str],
        scope_bucket: str,
        event: ReplayEvent,
        order: Sequence[str],
        zero_order: Sequence[str],
    ) -> None:
        for split in splits:
            self.splits.setdefault(split, MetricAccumulator()).add_event(event, order, zero_order)
            scoped = self.per_scope.setdefault(split, {})
            scoped.setdefault(scope_bucket, MetricAccumulator()).add_event(
                event, order, zero_order
            )

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for split, accumulator in self.splits.items():
            block = accumulator.to_dict()
            block["per_scope"] = {
                scope: acc.to_dict()
                for scope, acc in sorted(self.per_scope.get(split, {}).items())
            }
            output[split] = block
        return output


# ---------------------------------------------------------------------------
# Harness driver
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HarnessConfig:
    """Everything the replay run needs besides the database itself."""

    cutoff: str
    label: LabelConfig = field(default_factory=LabelConfig)
    schemes: tuple[str, ...] = STATIC_SCHEMES
    credit_rule: str = "winner_take_all"
    explicit: tuple[float, float, float] | None = None
    min_scope_events: int = DEFAULT_MIN_SCOPE_EVENTS
    evidence_mode: str = "snapshot"
    lr_policy: str = "snapshot"
    use_snapshot_supersedes: bool = False
    max_events: int | None = None
    label_sensitivity: bool = True
    trajectory_out: Path | None = None
    lookup_window_seconds: float = DEFAULT_LOOKUP_WINDOW_SECONDS


def normalize_cutoff(raw: str) -> str:
    """Normalize a CLI cutoff to the storage timestamp format (UTC, Z suffix)."""

    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def scope_buckets(
    events: Sequence[ReplayEvent], min_scope_events: int
) -> dict[str, str]:
    """Map event scope -> reported bucket (small scopes fold into ``_other``)."""

    counts = Counter(event.scope for event in events if event.labeled)
    return {
        scope: (scope if count >= min_scope_events else OTHER_SCOPE_BUCKET)
        for scope, count in counts.items()
    }


def _splits_for(event: ReplayEvent, cutoff: str) -> tuple[str, ...]:
    return ("overall", "holdout" if event.created_at > cutoff else "train")


def run_scheme(
    name: str,
    events: Sequence[ReplayEvent],
    buckets: Mapping[str, str],
    cutoff: str,
    *,
    resolver: Callable[[str], RetrievalWeights] | None = None,
    trajectory: WeightTrajectory | None = None,
    credit_rule: CreditRule | None = None,
    supersedes_pairs: Iterable[tuple[str, str]] = (),
) -> SchemeReport:
    """Replay all labeled events under one scheme, chronologically.

    For the dynamic (``replayed``) scheme, reinforcement updates are merged
    into the event stream by ``feedback_applied_at``; updates after the cutoff
    are skipped so holdout events see weights frozen at the cutoff.
    """

    report = SchemeReport()
    service: MemoryRecallService | None = None
    if name != "recorded":
        active_resolver = trajectory.resolve if trajectory is not None else resolver
        if active_resolver is None:
            raise ValueError(f"scheme {name} needs a weights resolver")
        service = make_ranking_service(active_resolver, supersedes_pairs)

    updates = reinforcement_order(events) if trajectory is not None else []
    update_index = 0

    for event in events:
        if trajectory is not None and credit_rule is not None:
            while update_index < len(updates) and updates[update_index][0] < event.created_at:
                update_time, update_event = updates[update_index]
                update_index += 1
                if update_time > cutoff:
                    continue
                trajectory.reinforce_event(update_event, credit_rule)
        if not event.labeled:
            continue
        if name == "recorded":
            order = recorded_order(event)
            zero_order = recorded_order(event, zero_graph=True)
        else:
            assert service is not None
            order = rerank_event(service, event)
            zero_order = rerank_event(service, event, zero_graph=True)
        report.add(
            _splits_for(event, cutoff),
            buckets.get(event.scope, OTHER_SCOPE_BUCKET),
            event,
            order,
            zero_order,
        )

    if trajectory is not None and credit_rule is not None:
        while update_index < len(updates):
            update_time, update_event = updates[update_index]
            update_index += 1
            if update_time <= cutoff:
                trajectory.reinforce_event(update_event, credit_rule)

    return report


def run_replay(db_path: str | Path, config: HarnessConfig) -> dict[str, Any]:
    """Execute the full harness and return the report dictionary."""

    connection = open_readonly(db_path)
    try:
        return _run_replay(connection, db_path, config)
    finally:
        connection.close()


def _run_replay(
    connection: sqlite3.Connection, db_path: str | Path, config: HarnessConfig
) -> dict[str, Any]:
    cutoff = normalize_cutoff(config.cutoff)
    events, load_stats = load_replay_events(connection, max_events=config.max_events)
    label_data = build_label_data(connection, events)
    label_summary = apply_labels(events, label_data, config.label)
    # Independent of the label: this is the live write path's own verdict,
    # and it is what the grounded credit rules gate on.
    grounding_summary = apply_grounding(
        events, label_data, config.label.min_containment
    )
    # Lookup follows are bounded by the cutoff so a post-cutoff fetch cannot
    # feed a pre-cutoff weight update; grounding first, so the overlap counts.
    lookup_summary = load_lookup_follows(
        connection, events, window_seconds=config.lookup_window_seconds, not_after=cutoff
    )

    labeled_events = [event for event in events if event.labeled]
    buckets = scope_buckets(events, config.min_scope_events)
    memory_config = MemoryConfig()
    live_weights = load_live_weights(connection)
    evidence = (
        snapshot_evidence(connection) if config.evidence_mode == "snapshot" else assumed_evidence
    )
    supersedes_pairs = (
        load_supersedes_pairs(connection) if config.use_snapshot_supersedes else ()
    )
    snapshot_learning_rates = (
        {scope: weights.learning_rate for scope, weights in live_weights.items()}
        if config.lr_policy == "snapshot"
        else {}
    )

    holdout_events = [event for event in events if event.created_at > cutoff]
    holdout_labeled = [event for event in holdout_events if event.labeled]

    metrics: dict[str, Any] = {}
    trajectory: WeightTrajectory | None = None
    scheme_names = list(config.schemes)
    for name in scheme_names:
        if name == "recorded":
            report = run_scheme(name, events, buckets, cutoff)
        elif name == "live_weights":
            report = run_scheme(
                name,
                events,
                buckets,
                cutoff,
                resolver=static_resolver(live_weights),
                supersedes_pairs=supersedes_pairs,
            )
        elif name == "floor_defaults":
            report = run_scheme(
                name,
                events,
                buckets,
                cutoff,
                resolver=floor_default_weights(memory_config, evidence),
                supersedes_pairs=supersedes_pairs,
            )
        elif name == "uniform":
            report = run_scheme(
                name,
                events,
                buckets,
                cutoff,
                resolver=explicit_weights(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
                supersedes_pairs=supersedes_pairs,
            )
        elif name == "explicit":
            if config.explicit is None:
                raise ValueError("explicit scheme requested without --explicit-weights")
            report = run_scheme(
                name,
                events,
                buckets,
                cutoff,
                resolver=explicit_weights(*config.explicit),
                supersedes_pairs=supersedes_pairs,
            )
        elif name.startswith(REPLAYED_SCHEME_PREFIX):
            rule_name = name.removeprefix(REPLAYED_SCHEME_PREFIX).lstrip("_") or config.credit_rule
            rule = CREDIT_RULES.get(rule_name)
            if rule is None:
                raise ValueError(f"unknown credit rule for scheme {name}: {rule_name}")
            trajectory = WeightTrajectory(
                memory_config,
                evidence=evidence,
                snapshot_learning_rates=snapshot_learning_rates,
            )
            report = run_scheme(
                name,
                events,
                buckets,
                cutoff,
                trajectory=trajectory,
                credit_rule=rule,
                supersedes_pairs=supersedes_pairs,
            )
        else:
            raise ValueError(f"unknown scheme: {name}")
        metrics[name] = report.to_dict()

    report_data: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "db_path": str(db_path),
        "harness": {
            "version": 1,
            "schemes": scheme_names,
            "credit_rule": config.credit_rule,
            "label": config.label.to_dict(),
            "evidence_mode": config.evidence_mode,
            "lr_policy": config.lr_policy,
            "supersedes": "snapshot" if config.use_snapshot_supersedes else "none",
            "min_scope_events": config.min_scope_events,
            "lookup_window_seconds": config.lookup_window_seconds,
        },
        "corpus": {
            **load_stats,
            "first_event_at": events[0].created_at if events else None,
            "last_event_at": events[-1].created_at if events else None,
        },
        "labeled_events": len(labeled_events),
        "cutoff": cutoff,
        "split": {
            "train": {
                "events": len(events) - len(holdout_events),
                "labeled_events": len(labeled_events) - len(holdout_labeled),
            },
            "holdout": {
                "events": len(holdout_events),
                "labeled_events": len(holdout_labeled),
                "event_share_of_usable": _ratio(len(holdout_events), len(events)),
                "labeled_share_of_labeled": _ratio(len(holdout_labeled), len(labeled_events)),
            },
        },
        "labeling": label_summary,
        "grounding": grounding_summary,
        "lookups": lookup_summary,
        "weights": {
            "live": {
                scope: _weights_dict(weights) for scope, weights in sorted(live_weights.items())
            },
            "replayed_final": {
                scope: _weights_dict(weights)
                for scope, weights in sorted(trajectory.weights.items())
            }
            if trajectory is not None
            else {},
            "replayed_update_counts": dict(sorted(trajectory.update_counts.items()))
            if trajectory is not None
            else {},
        },
        "metrics": metrics,
    }

    if config.label_sensitivity:
        report_data["label_sensitivity"] = _label_sensitivity(
            events, label_data, config, buckets, cutoff, live_weights, supersedes_pairs
        )
        # Restore the primary labels after the sensitivity passes mutated them.
        apply_labels(events, label_data, config.label)

    if config.trajectory_out is not None and trajectory is not None:
        config.trajectory_out.parent.mkdir(parents=True, exist_ok=True)
        config.trajectory_out.write_text(
            json.dumps(
                {
                    "final": {
                        scope: _weights_dict(weights)
                        for scope, weights in sorted(trajectory.weights.items())
                    },
                    "update_counts": dict(sorted(trajectory.update_counts.items())),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return report_data


def _label_sensitivity(
    events: Sequence[ReplayEvent],
    label_data: LabelData,
    config: HarnessConfig,
    buckets: Mapping[str, str],
    cutoff: str,
    live_weights: Mapping[str, RetrievalWeights],
    supersedes_pairs: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    """Discrimination + live-weights metrics for the alternative label protocols."""

    output: dict[str, Any] = {}
    for protocol in LABEL_PROTOCOLS:
        variant = LabelConfig(
            protocol=protocol,
            min_containment=config.label.min_containment,
            reconsume_min_traces=config.label.reconsume_min_traces,
            usefulness_threshold=config.label.usefulness_threshold,
        )
        summary = apply_labels(events, label_data, variant)
        report = run_scheme(
            "live_weights",
            events,
            buckets,
            cutoff,
            resolver=static_resolver(dict(live_weights)),
            supersedes_pairs=supersedes_pairs,
        )
        overall = report.to_dict().get("overall", {})
        overall.pop("per_scope", None)
        output[protocol] = {
            "discrimination": summary["discrimination"],
            "events_with_useful": summary["events_with_useful"],
            "useful_results": summary["useful_results"],
            "overall_live_weights": overall,
        }
    return output


def _weights_dict(weights: RetrievalWeights) -> dict[str, float]:
    return {
        "bm25": round(weights.bm25, 6),
        "vector": round(weights.vector, 6),
        "graph": round(weights.graph, 6),
        "learning_rate": round(weights.learning_rate, 6),
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

_PROTOCOL_NOTES = """\
## Labeling protocol

A recall event is **labeled** when a later `memory_remember` consumed it
(`feedback_trace_id` is set). Within a labeled event, a result is
**confirmed useful** under the primary `grounded` protocol when the
IDF-weighted share of the result node's content tokens that also appear in
the consuming trace's content (containment) is at least `min_containment`.
IDF is computed over the corpus of result-node and consuming-trace contents,
so boilerplate tokens contribute little and identifiers/paths dominate.

Why not the recorded linkage? `feedback.apply_pending_recall_feedback` links
**every** result of a consumed event to the consuming trace
(provenance `recalled_nodes`/`source_traces`, `related` edges), so
"linked by the consuming trace" marks all results useful and every ranking
metric becomes vacuous. Grounding discriminates within the result list.
Alternative protocols (`reconsumed`, `usefulness`,
`grounded_or_reconsumed`) are reported in the label-sensitivity section:
re-consumption is nearly vacuous on this corpus (a few thousand hub nodes
recirculate across most events), and long-run usefulness is
rank-circular (it accrues via the same rank-decayed reinforcement loop the
harness audits) with a saturated distribution (median 1.0).

## Metric definitions

* `hit@k` / `mrr` — over labeled events with at least one confirmed-useful
  result: whether/where the first useful result appears in the re-ranked
  order of the recorded candidates. A useful result that a scheme ranks to
  zero score counts as a miss.
* `graph_unique_share` — share of confirmed-useful results whose recorded
  evidence is graph-only (`graph_score > 0`, `bm25 = vector = 0`).
  Scheme-independent (recorded evidence), repeated per scheme for
  consumer convenience.
* `graph_zeroed.*` — counterfactual re-rank of the same candidates with
  `graph_score` forced to 0 (which also disables the in-rank graph rescue):
  `dropped_topK` is the share of confirmed-useful results in the scheme's
  top-K that leave the top-K; `dropped_entirely` is the share of ranked
  useful results that drop to zero score; `hit@5`/`mrr` are re-computed on
  the zeroed ranking. `dropped_*` can sit below `graph_unique_share`:
  schema nodes whose only method evidence is graph still survive zeroing
  through the trigger-score override (on this corpus most graph-only
  confirmed-useful results are exactly such schema nodes).

## Limitations

* **Candidate selection bias**: recorded results are only the
  top-`max_results` under the *old* live weights + rescue ranking. Replay
  can re-order or drop recorded candidates but can never surface candidates
  the old ranking excluded, so absolute metric levels are optimistic and
  graph-value estimates are lower bounds relative to a full re-retrieval.
* **Neutral node stats**: re-ranked schemes hold node-level feedback
  multipliers (confidence/usefulness/access, supersedes corrections) at
  neutral because event-time node stats are not recorded. The `recorded`
  scheme preserves the historical order including those multipliers.
* **Grounding is textual**: results used conceptually without shared
  identifiers are missed (~half of labeled events have no grounded result;
  they are excluded from hit/MRR denominators). Consuming traces deleted
  since (`missing_consuming_traces`) cannot be grounded.
* **Trajectory approximations**: initial weights are the configured family
  defaults; per-scope learning rates come from the snapshot weights table
  (the live adaptive tuner's final state); floor evidence gates are
  evaluated against the snapshot, not historically; explicit
  feedback/teach events are not in the corpus and are not replayed.
"""


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable companion for the JSON report."""

    lines: list[str] = []
    corpus = report["corpus"]
    split = report["split"]
    labeling = report["labeling"]
    discrimination = labeling["discrimination"]

    lines.append("# Recall-replay baseline")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']} | DB: `{report['db_path']}`")
    lines.append("")
    lines.append("## Corpus and split")
    lines.append("")
    lines.append(
        f"- Events: {corpus['total_events']} total, {corpus['usable_events']} usable "
        f"(skipped: {corpus['empty_results']} empty, "
        f"{corpus['incomplete_results']} incomplete results)"
    )
    lines.append(
        f"- Labeled (consumed) events: {report['labeled_events']} | "
        f"span {corpus['first_event_at']} .. {corpus['last_event_at']}"
    )
    lines.append(
        f"- Cutoff `{report['cutoff']}`: train {split['train']['events']} events "
        f"({split['train']['labeled_events']} labeled) / holdout "
        f"{split['holdout']['events']} events ({split['holdout']['labeled_events']} labeled, "
        f"{split['holdout']['event_share_of_usable']:.1%} of usable events, "
        f"{split['holdout']['labeled_share_of_labeled']:.1%} of labeled)"
    )
    lines.append("")
    lines.append("## Labels")
    lines.append("")
    lines.append(f"- Protocol: `{labeling['config']['protocol']}` {labeling['config']}")
    lines.append(
        f"- {labeling['useful_results']} useful of {labeling['labeled_results']} labeled results; "
        f"{labeling['events_with_useful']} of {labeling['labeled_events']} labeled events have "
        f"at least one useful result"
    )
    lines.append(
        f"- Discrimination over {discrimination['multi_result_events']} multi-result events: "
        f"strict subset {discrimination['strict_subset_fraction']:.1%}, "
        f"all-useful {discrimination['all_useful_fraction']:.1%}, "
        f"none-useful {discrimination['none_useful_fraction']:.1%}, "
        f"mean useful share {discrimination['mean_useful_share']:.2f}"
    )
    lines.append("")
    lines.append("## Metrics by scheme")
    header = (
        "| scheme | split | events | w/useful | hit@1 | hit@5 | hit@10 | MRR "
        "| graph-unique | zeroed hit@5 | zeroed MRR | drop@1 | drop@5 | drop@max | drop-all |"
    )
    divider = "|" + "---|" * 14
    lines.append(header)
    lines.append(divider)
    for scheme, blocks in report["metrics"].items():
        for split_name in ("overall", "train", "holdout"):
            block = blocks.get(split_name)
            if not block:
                continue
            zero = block["graph_zeroed"]
            lines.append(
                f"| {scheme} | {split_name} | {block['events']} | {block['events_with_useful']} "
                f"| {block['hit@1']:.3f} | {block['hit@5']:.3f} | {block['hit@10']:.3f} "
                f"| {block['mrr']:.3f} | {block['graph_unique_share']:.3f} "
                f"| {zero['hit@5']:.3f} | {zero['mrr']:.3f} | {zero['dropped_top1']:.3f} "
                f"| {zero['dropped_top5']:.3f} | {zero['dropped_top_max']:.3f} "
                f"| {zero['dropped_entirely']:.3f} |"
            )
    lines.append("")
    lines.append("### Per-scope (overall split)")
    lines.append("")
    lines.append("| scheme | scope | events | w/useful | hit@5 | MRR | graph-unique | drop@5 |")
    lines.append("|" + "---|" * 8)
    for scheme, blocks in report["metrics"].items():
        overall = blocks.get("overall", {})
        for scope, block in overall.get("per_scope", {}).items():
            zero = block["graph_zeroed"]
            lines.append(
                f"| {scheme} | {scope} | {block['events']} | {block['events_with_useful']} "
                f"| {block['hit@5']:.3f} | {block['mrr']:.3f} "
                f"| {block['graph_unique_share']:.3f} | {zero['dropped_top5']:.3f} |"
            )
    lines.append("")
    sensitivity = report.get("label_sensitivity")
    if sensitivity:
        lines.append("## Label sensitivity (live_weights scheme, overall)")
        lines.append("")
        lines.append(
            "| protocol | strict-subset | all-useful | none-useful | events w/useful | hit@5 | MRR |"
        )
        lines.append("|" + "---|" * 7)
        for protocol, data in sensitivity.items():
            disc = data["discrimination"]
            overall = data["overall_live_weights"]
            lines.append(
                f"| {protocol} | {disc['strict_subset_fraction']:.1%} "
                f"| {disc['all_useful_fraction']:.1%} | {disc['none_useful_fraction']:.1%} "
                f"| {data['events_with_useful']} | {overall.get('hit@5', 0.0):.3f} "
                f"| {overall.get('mrr', 0.0):.3f} |"
            )
        lines.append("")
    lines.append(_PROTOCOL_NOTES)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_explicit(raw: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in raw.replace(":", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected bm25,vector,graph")
    bm25, vector, graph = (float(part) for part in parts)
    return (bm25, vector, graph)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m living_memory.replay",
        description="Offline recall-replay evaluation over recorded recall_events.",
    )
    parser.add_argument("--db", required=True, help="SQLite database (opened read-only)")
    parser.add_argument("--cutoff", required=True, help="ISO timestamp splitting train/holdout")
    parser.add_argument("--report", required=True, help="output JSON report path")
    parser.add_argument("--md", help="output markdown report path (default: report with .md)")
    parser.add_argument(
        "--schemes",
        default="recorded,live_weights,floor_defaults,uniform,replayed",
        help="comma-separated schemes: recorded, live_weights, floor_defaults, uniform, "
        "explicit, replayed[_<rule>]",
    )
    parser.add_argument(
        "--explicit-weights",
        type=_parse_explicit,
        help="bm25,vector,graph for the 'explicit' scheme",
    )
    parser.add_argument(
        "--credit-rule",
        default="winner_take_all",
        choices=sorted(CREDIT_RULES),
        help="credit rule for the 'replayed' scheme",
    )
    parser.add_argument(
        "--label-protocol",
        default="grounded",
        choices=LABEL_PROTOCOLS,
        help="confirmed-useful labeling protocol",
    )
    parser.add_argument("--min-containment", type=float, default=DEFAULT_MIN_CONTAINMENT)
    parser.add_argument(
        "--lookup-window",
        type=float,
        default=DEFAULT_LOOKUP_WINDOW_SECONDS,
        help="seconds after an event within which a same-transport memory_lookup of a "
        "delivered node counts as a follow (grounded_or_lookup credit rule)",
    )
    parser.add_argument(
        "--reconsume-min-traces", type=int, default=DEFAULT_RECONSUME_MIN_TRACES
    )
    parser.add_argument(
        "--usefulness-threshold", type=float, default=DEFAULT_USEFULNESS_THRESHOLD
    )
    parser.add_argument("--min-scope-events", type=int, default=DEFAULT_MIN_SCOPE_EVENTS)
    parser.add_argument(
        "--evidence",
        default="snapshot",
        choices=("snapshot", "assume"),
        help="floor evidence gates: query the snapshot or assume both channels present",
    )
    parser.add_argument(
        "--lr-policy",
        default="snapshot",
        choices=("snapshot", "config"),
        help="learning rate for newly materialized scopes in the trajectory replay",
    )
    parser.add_argument(
        "--supersedes",
        default="none",
        choices=("none", "snapshot"),
        help="supersedes pairs for re-ranking (snapshot edges are not event-time accurate)",
    )
    parser.add_argument("--max-events", type=int, help="debug: cap the number of loaded events")
    parser.add_argument(
        "--no-label-sensitivity",
        action="store_true",
        help="skip the alternative-label sensitivity passes",
    )
    parser.add_argument("--trajectory-out", help="optional JSON dump of the replayed weights")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    schemes: list[str] = []
    for name in (part.strip() for part in args.schemes.split(",")):
        if not name:
            continue
        schemes.append(name)
    config = HarnessConfig(
        cutoff=args.cutoff,
        label=LabelConfig(
            protocol=args.label_protocol,
            min_containment=args.min_containment,
            reconsume_min_traces=args.reconsume_min_traces,
            usefulness_threshold=args.usefulness_threshold,
        ),
        schemes=tuple(schemes),
        credit_rule=args.credit_rule,
        explicit=args.explicit_weights,
        min_scope_events=args.min_scope_events,
        evidence_mode=args.evidence,
        lr_policy=args.lr_policy,
        use_snapshot_supersedes=args.supersedes == "snapshot",
        max_events=args.max_events,
        label_sensitivity=not args.no_label_sensitivity,
        trajectory_out=Path(args.trajectory_out) if args.trajectory_out else None,
        lookup_window_seconds=args.lookup_window,
    )
    report = run_replay(args.db, config)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path = Path(args.md) if args.md else report_path.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(report), encoding="utf-8")

    overall = report["metrics"].get(next(iter(report["metrics"]), ""), {}).get("overall", {})
    print(
        f"replayed {report['labeled_events']} labeled events "
        f"({overall.get('events_with_useful', 0)} with useful results) -> {report_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
