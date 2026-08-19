"""Retrospective consumption metric: were these nodes actually recalled later?

The write half of Living Memory can only be judged by the read half. A trace
that nobody ever recalls again is not memory, it is a log line — so the gate on
any writer (the post-session extractor above all) has to be a *measured*
consumption rate, not the writer's own opinion of its output.

Definition (fixed before the measurement, reproduced against the published
field baseline — do not tune):

    A node is CONSUMED when its id appears in
    ``json.loads(recall_events.results)[].node_id`` of an event whose
    ``created_at`` is **strictly greater** than the node's ``created_at``.

The cohort is **every row of ``nodes``** whose ``created_at`` falls in the
window — all levels (trace + concept + schema) and including decayed rows. That
detail is load-bearing and was measured: on the 2026-08-19 database at
``--as-of 2026-08-19T00:00:00Z`` the July window over all levels gives
1129 / 558 / 49.4% / 41.1%, which matches the published ground truth, while
restricting to ``level='trace'`` gives 1072 / 523 / 48.8% / 41.2%, which does
not. :func:`load_cohort_nodes` therefore filters by window, ``agent`` and a
``context`` key predicate and by nothing else.

``as_of`` cuts **both** ``nodes`` and ``recall_events`` on ``created_at``. It is
what makes a run reproducible: the live database keeps growing, and a recall
event recorded after a window closed can only ever *add* consumption to it.

GROUNDED variant. Raw consumption says a node was delivered again; it does not
say the reading agent used it. The grounded variant additionally requires that
the consuming event was closed by a ``memory_remember`` trace (a non-null
``feedback_trace_id``) whose content *grounds* the node, under the shared
IDF-containment rule at :data:`DEFAULT_MIN_CONTAINMENT`. The containment
arithmetic is never reimplemented here: it comes from
:func:`living_memory.replay.build_label_data` /
:func:`living_memory.replay.apply_grounding` (bulk path) and
:func:`living_memory.grounding.ground_results` (live path, re-exported as
:func:`ground_delivered_results` and used as a per-run cross-check).

The grounded variant reports its own denominator and never divides by the whole
cohort by default. Measured trap: of the 2747 recall events after 2026-08-12,
all 2747 carry non-empty ``results`` but only 494 (18%) carry a non-null
``feedback_trace_id``, and grounding is undefined for the rest. Dividing
grounded consumption by the full cohort understates it several-fold (2.4x on
the July window: 173/479 = 36.1% vs 173/1129 = 15.3%), so ``rate`` divides by
``nodes_with_closed_consumer`` and the cohort-wide figure is published
separately as ``rate_over_cohort``.

SAFETY. This module is read-only by construction. The live database is opened
only through :func:`living_memory.replay.open_readonly` (a ``mode=ro`` SQLite
URI); nothing here ever opens a writable connection or instantiates the
migrating store wrapper, which would rewrite whatever file it is pointed at.

Nothing here imports the extractor, the judge or any transcript code: the
metric must stay usable as an independent judge of them.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from living_memory.grounding import DEFAULT_MIN_CONTAINMENT, ground_results
from living_memory.replay import (
    ReplayEvent,
    ReplayResult,
    apply_grounding,
    build_label_data,
    open_readonly,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "DEFAULT_WITHIN_DAYS",
    "LIVE_PATH_SAMPLE_EVENTS",
    "NULL_AGENT_TOKENS",
    "SAFE_STRING",
    "UNATTRIBUTED_LABEL",
    "CohortMeasurement",
    "CohortNode",
    "CohortSelector",
    "ConsumptionIndex",
    "PrivacyGuardError",
    "build_selectors",
    "check_privacy",
    "ground_delivered_results",
    "load_cohort_nodes",
    "load_consumption_index",
    "measure_cohort",
    "normalize_instant",
    "parse_context_key",
    "parse_timestamp",
    "parse_window",
    "run_metric",
]

#: Where the live database lives. Callers pass their own path; this is only the
#: CLI default and is never opened writably.
DEFAULT_DB_PATH = Path("~/.local/share/living-memory/global.sqlite3")

#: "Recalled again soon" horizon, in days, for ``consumed_within_7d``.
DEFAULT_WITHIN_DAYS = 7

#: ``nodes.agent`` is NULL for a large share of rows (7463 of 16759 on the
#: 2026-08-19 database). These CLI tokens select exactly those rows; neither is
#: a real agent name in the field data, which was checked before they were
#: reserved.
NULL_AGENT_TOKENS = ("NULL", "unattributed")
UNATTRIBUTED_LABEL = "unattributed"

#: Values in a published artifact must be aggregate-only: printable ASCII,
#: short, single line. Anything else — node content, query text, a transcript
#: fragment — is rejected before the artifact is written. Same convention as
#: ``~/p/ae/tools/lm_workload/remote_usage_linkage.py``.
SAFE_STRING = re.compile(r"^[\x20-\x7e]{0,200}$")

#: How many grounded events to re-grade through the live entry point as a
#: per-run cross-check of the bulk path.
LIVE_PATH_SAMPLE_EVENTS = 50

#: Sorts after every real event ordinal, so a bisect on ``(timestamp, ordinal)``
#: lands past every event sharing the node's own timestamp — consumption is
#: defined as *strictly* later.
_AFTER_ANY_EVENT = float("inf")

_SQL_PARAM_CHUNK = 900


# ---------------------------------------------------------------------------
# Privacy guard
# ---------------------------------------------------------------------------


class PrivacyGuardError(ValueError):
    """A value bound for a published artifact is not aggregate-only."""


def check_privacy(value: Any, path: str = "$") -> None:
    """Reject anything but aggregate-safe values, recursively.

    Numbers, ``None`` and booleans pass. Strings must be short printable ASCII
    (:data:`SAFE_STRING`), which memory content, queries and transcript text
    reliably are not. Raises :class:`PrivacyGuardError` naming the offending
    path so the failure is diagnosable without printing the value itself.
    """

    if value is None or isinstance(value, bool | int | float):
        return
    if isinstance(value, str):
        if not SAFE_STRING.match(value):
            raise PrivacyGuardError(f"privacy guard: unsafe string value at {path}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            check_privacy(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not SAFE_STRING.match(key):
                raise PrivacyGuardError(f"privacy guard: unsafe key at {path}")
            check_privacy(item, f"{path}.{key}")
        return
    raise PrivacyGuardError(f"privacy guard: unexpected value type at {path}")


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def parse_timestamp(raw: Any) -> datetime | None:
    """Parse a stored ISO-8601 timestamp; ``None`` when it is unusable.

    Naive values are read as UTC, which is what the writer emits.
    """

    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def normalize_instant(raw: str) -> str:
    """Normalize a CLI instant to the exact shape stored in the database.

    Every ``created_at`` in the field database is ``%Y-%m-%dT%H:%M:%SZ``, so
    normalizing to that shape makes plain SQL string comparison a correct
    ordering. A bare ``YYYY-MM-DD`` means midnight UTC that day.
    """

    text = str(raw).strip()
    if not text:
        raise ValueError("empty instant")
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        text = f"{text}T00:00:00Z"
    parsed = parse_timestamp(text)
    if parsed is None:
        raise ValueError(f"not an ISO-8601 instant: {raw!r}")
    return parsed.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_window(raw: str) -> tuple[str, str]:
    """Parse ``LO..HI`` into a normalized half-open ``[lo, hi)`` window."""

    text = str(raw).strip()
    if ".." not in text:
        raise ValueError(f"window must be LO..HI, got {raw!r}")
    low, _, high = text.partition("..")
    start, end = normalize_instant(low), normalize_instant(high)
    if start >= end:
        raise ValueError(f"window start must precede its end: {raw!r}")
    return start, end


def parse_context_key(raw: str) -> tuple[str, str | None]:
    """Parse a ``K=V`` context predicate.

    ``K=`` (empty value) means "key present with a non-null value", which is how
    a cohort like "everything the extractor stamped a session id on" is
    selected without pinning the id itself.
    """

    text = str(raw)
    if "=" not in text:
        raise ValueError(f"context key predicate must be K=V, got {raw!r}")
    key, _, value = text.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"context key predicate has an empty key: {raw!r}")
    return key, (value if value != "" else None)


# ---------------------------------------------------------------------------
# Cohort selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CohortSelector:
    """Which rows of ``nodes`` a cohort is made of.

    Deliberately has no level and no decay knob: the cohort is every row in the
    window, which is the definition the published baseline was measured under.
    """

    label: str
    start: str
    end: str
    agent: str | None = None
    agent_is_null: bool = False
    context_key: str | None = None
    context_value: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "window_start": self.start,
            "window_end": self.end,
            "agent": UNATTRIBUTED_LABEL if self.agent_is_null else self.agent,
            "agent_is_null": self.agent_is_null,
            "context_key": self.context_key,
            "context_value": self.context_value,
        }


def build_selectors(
    window: tuple[str, str],
    *,
    agents: Sequence[str] = (),
    context_keys: Sequence[tuple[str, str | None]] = (),
) -> list[CohortSelector]:
    """Window-wide cohort first, then one refinement per agent / context key.

    The window-wide cohort is always emitted: a per-agent number is only
    readable against the population it was drawn from.
    """

    start, end = window
    selectors = [CohortSelector(label="all", start=start, end=end)]
    for agent in agents:
        if agent in NULL_AGENT_TOKENS:
            selectors.append(
                CohortSelector(
                    label=f"agent={UNATTRIBUTED_LABEL}",
                    start=start,
                    end=end,
                    agent_is_null=True,
                )
            )
        else:
            selectors.append(
                CohortSelector(label=f"agent={agent}", start=start, end=end, agent=agent)
            )
    for key, value in context_keys:
        shown = "*" if value is None else value
        selectors.append(
            CohortSelector(
                label=f"context.{key}={shown}",
                start=start,
                end=end,
                context_key=key,
                context_value=value,
            )
        )
    return selectors


@dataclass(frozen=True, slots=True)
class CohortNode:
    """One cohort row: only what the metric needs, never its content."""

    id: str
    created_at: str
    created_dt: datetime | None


def _context_matches(raw: Any, key: str, value: str | None) -> bool:
    """Top-level key predicate over the ``nodes.context`` JSON blob."""

    if not isinstance(raw, str) or not raw.strip():
        return False
    try:
        context = json.loads(raw)
    except ValueError:
        return False
    if not isinstance(context, dict) or key not in context:
        return False
    found = context[key]
    if found is None:
        return False
    if value is None:
        return True
    if isinstance(found, bool):
        return value == ("true" if found else "false")
    return str(found) == value


def load_cohort_nodes(
    connection: sqlite3.Connection,
    selector: CohortSelector,
    *,
    as_of: str | None = None,
) -> list[CohortNode]:
    """Every ``nodes`` row the selector admits — all levels, decayed included."""

    clauses = ["created_at >= ?", "created_at < ?"]
    params: list[Any] = [selector.start, selector.end]
    if as_of is not None:
        clauses.append("created_at <= ?")
        params.append(as_of)
    if selector.agent_is_null:
        clauses.append("agent IS NULL")
    elif selector.agent is not None:
        clauses.append("agent = ?")
        params.append(selector.agent)
    sql = (
        "SELECT id, created_at, context FROM nodes WHERE "
        + " AND ".join(clauses)
        + " ORDER BY created_at ASC, id ASC"
    )
    nodes: list[CohortNode] = []
    for row in connection.execute(sql, params):
        if selector.context_key is not None and not _context_matches(
            row["context"], selector.context_key, selector.context_value
        ):
            continue
        created_at = str(row["created_at"])
        nodes.append(
            CohortNode(
                id=str(row["id"]),
                created_at=created_at,
                created_dt=parse_timestamp(created_at),
            )
        )
    return nodes


# ---------------------------------------------------------------------------
# Consumption index
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ConsumptionIndex:
    """Every recorded delivery, keyed by the node that was delivered.

    ``by_node[node_id]`` is the chronologically sorted list of
    ``(event created_at, event ordinal)`` pairs in which that node appeared, so
    "the first delivery strictly after this node was written" is one bisect.
    """

    events: list[ReplayEvent]
    event_times: list[datetime | None]
    by_node: dict[str, list[tuple[str, int]]]
    stats: dict[str, int]

    def later_deliveries(self, node: CohortNode) -> list[tuple[str, int]]:
        entries = self.by_node.get(node.id)
        if not entries:
            return []
        start = bisect_right(entries, (node.created_at, _AFTER_ANY_EVENT))
        return entries[start:]


def _replay_results(raw_results: Any) -> list[ReplayResult]:
    """The delivered node ids of one event, as replay's own result records.

    Method scores are left at zero on purpose: the grounding path reads only
    ``node_id`` and the node's stored content, and parsing the recorded scores
    would make the metric drop events whose ``results`` JSON predates a scoring
    field — events the literal definition counts.
    """

    if not isinstance(raw_results, list):
        return []
    results: list[ReplayResult] = []
    for index, item in enumerate(raw_results):
        if not isinstance(item, dict):
            continue
        node_id = item.get("node_id")
        if not node_id:
            continue
        results.append(
            ReplayResult(
                node_id=str(node_id),
                rank=int(item.get("rank") or index + 1),
                level=str(item.get("level") or "trace"),
                scope=str(item.get("scope") or "global"),
                score=0.0,
                bm25_score=0.0,
                vector_score=0.0,
                graph_score=0.0,
                trigger_score=0.0,
            )
        )
    return results


def load_consumption_index(
    connection: sqlite3.Connection, *, as_of: str | None = None
) -> ConsumptionIndex:
    """Read every recall event at or before ``as_of`` into the delivery index."""

    sql = "SELECT id, created_at, results, feedback_trace_id FROM recall_events"
    params: list[Any] = []
    if as_of is not None:
        sql += " WHERE created_at <= ?"
        params.append(as_of)
    sql += " ORDER BY created_at ASC, rowid ASC"

    events: list[ReplayEvent] = []
    event_times: list[datetime | None] = []
    by_node: dict[str, list[tuple[str, int]]] = {}
    stats = {
        "events_scanned": 0,
        "events_unparsable_results": 0,
        "events_without_node_ids": 0,
        "events_with_deliveries": 0,
        "events_closed_by_trace": 0,
        "events_unparsable_created_at": 0,
        "distinct_delivered_nodes": 0,
    }

    for row in connection.execute(sql, params):
        stats["events_scanned"] += 1
        try:
            raw_results = json.loads(row["results"] or "[]")
        except ValueError:
            stats["events_unparsable_results"] += 1
            continue
        results = _replay_results(raw_results)
        if not results:
            stats["events_without_node_ids"] += 1
            continue
        created_at = str(row["created_at"])
        created_dt = parse_timestamp(created_at)
        if created_dt is None:
            stats["events_unparsable_created_at"] += 1
        trace_id = row["feedback_trace_id"]
        ordinal = len(events)
        events.append(
            ReplayEvent(
                id=str(row["id"]),
                created_at=created_at,
                scope="",
                requested_scope="",
                resolved_scopes=(),
                depth=None,
                max_results=len(results),
                query="",
                feedback_trace_id=trace_id,
                feedback_applied_at=None,
                results=results,
            )
        )
        event_times.append(created_dt)
        stats["events_with_deliveries"] += 1
        if trace_id:
            stats["events_closed_by_trace"] += 1
        for result in results:
            by_node.setdefault(result.node_id, []).append((created_at, ordinal))

    for entries in by_node.values():
        entries.sort()
    stats["distinct_delivered_nodes"] = len(by_node)
    return ConsumptionIndex(
        events=events, event_times=event_times, by_node=by_node, stats=stats
    )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CohortMeasurement:
    """Raw consumption for one cohort, plus what the grounded pass needs."""

    selector: CohortSelector
    nodes: int = 0
    consumed_later: int = 0
    consumed_within_7d: int = 0
    consuming_events: set[int] = field(default_factory=set)
    closed_consuming_events: set[int] = field(default_factory=set)
    #: node id -> its later deliveries that were closed by a remember trace
    closed_consumers: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    #: node id -> deadline for the "recalled again soon" horizon
    horizons: dict[str, datetime | None] = field(default_factory=dict)

    @property
    def nodes_with_closed_consumer(self) -> int:
        return len(self.closed_consumers)


def measure_cohort(
    index: ConsumptionIndex,
    nodes: Sequence[CohortNode],
    selector: CohortSelector,
    *,
    within_days: int = DEFAULT_WITHIN_DAYS,
) -> CohortMeasurement:
    """Raw consumption: was each node delivered by a strictly later event?"""

    measurement = CohortMeasurement(selector=selector, nodes=len(nodes))
    for node in nodes:
        later = index.later_deliveries(node)
        if not later:
            continue
        measurement.consumed_later += 1
        horizon = (
            node.created_dt + timedelta(days=within_days)
            if node.created_dt is not None
            else None
        )
        measurement.horizons[node.id] = horizon
        first_dt = index.event_times[later[0][1]]
        if horizon is not None and first_dt is not None and first_dt <= horizon:
            measurement.consumed_within_7d += 1
        closed: list[tuple[str, int]] = []
        for created_at, ordinal in later:
            measurement.consuming_events.add(ordinal)
            if index.events[ordinal].feedback_trace_id:
                measurement.closed_consuming_events.add(ordinal)
                closed.append((created_at, ordinal))
        if closed:
            measurement.closed_consumers[node.id] = closed
    return measurement


def _ratio(part: float | None, whole: float) -> float | None:
    if part is None or not whole:
        return None
    return round(part / whole, 4)


def _pct(part: float | None, whole: float) -> float | None:
    if part is None or not whole:
        return None
    return round(100.0 * part / whole, 1)


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


def ground_delivered_results(
    trace_content: str,
    node_contents: Mapping[str, str],
    *,
    min_containment: float = DEFAULT_MIN_CONTAINMENT,
) -> set[str]:
    """Ids of the delivered nodes the closing trace grounds — live entry point.

    A thin, honest wrapper over :func:`living_memory.grounding.ground_results`:
    the containment arithmetic lives there and is never duplicated. Exported
    because a caller holding raw contents (a replayed retrieval, for instance)
    needs the same verdict this module applies to recorded events.
    """

    verdicts = ground_results(
        trace_content, dict(node_contents), min_containment=min_containment
    )
    return {node_id for node_id, verdict in verdicts.items() if verdict.grounded}


def _fetch_contents(
    connection: sqlite3.Connection, ids: Iterable[str]
) -> dict[str, str]:
    unique = list(dict.fromkeys(ids))
    found: dict[str, str] = {}
    for start in range(0, len(unique), _SQL_PARAM_CHUNK):
        chunk = unique[start : start + _SQL_PARAM_CHUNK]
        placeholders = ",".join("?" * len(chunk))
        rows = connection.execute(
            f"SELECT id, content FROM nodes WHERE id IN ({placeholders})", chunk
        )
        for row in rows:
            found[str(row["id"])] = str(row["content"] or "")
    return found


def _live_path_crosscheck(
    connection: sqlite3.Connection,
    index: ConsumptionIndex,
    ordinals: Sequence[int],
    *,
    min_containment: float,
    sample_events: int,
) -> dict[str, Any]:
    """Re-grade a deterministic sample through the live entry point.

    The bulk pass grades pre-tokenized corpus documents; the live write path
    tokenizes raw content. They are meant to be the same decision, and this
    guards that assumption per run instead of trusting it.
    """

    sampled = list(ordinals)[:sample_events]
    trace_ids = [
        index.events[ordinal].feedback_trace_id or "" for ordinal in sampled
    ]
    node_ids = [
        result.node_id for ordinal in sampled for result in index.events[ordinal].results
    ]
    contents = _fetch_contents(connection, [*trace_ids, *node_ids])

    pairs = 0
    agreements = 0
    graded_events = 0
    for ordinal in sampled:
        event = index.events[ordinal]
        trace_content = contents.get(event.feedback_trace_id or "")
        if trace_content is None:
            continue
        result_contents = {
            result.node_id: contents[result.node_id]
            for result in event.results
            if result.node_id in contents
        }
        if not result_contents:
            continue
        graded_events += 1
        grounded_ids = ground_delivered_results(
            trace_content, result_contents, min_containment=min_containment
        )
        for result in event.results:
            if result.node_id not in result_contents:
                continue
            pairs += 1
            agreements += int(result.grounded == (result.node_id in grounded_ids))
    return {
        "sampled_events": graded_events,
        "sampled_pairs": pairs,
        "agreements": agreements,
        "agreement_rate": _ratio(agreements, pairs),
    }


def _grounded_block(
    measurement: CohortMeasurement,
    index: ConsumptionIndex,
    *,
    min_containment: float,
    within_days: int,
) -> dict[str, Any]:
    """Grounded consumption with the denominator it is actually measured over."""

    grounded_nodes = 0
    grounded_within = 0
    grounded_events: set[int] = set()
    for node_id, closed in measurement.closed_consumers.items():
        first_grounded: datetime | None = None
        hit = False
        for _created_at, ordinal in closed:
            event = index.events[ordinal]
            if not any(
                result.node_id == node_id and result.grounded for result in event.results
            ):
                continue
            hit = True
            grounded_events.add(ordinal)
            when = index.event_times[ordinal]
            if when is not None and (first_grounded is None or when < first_grounded):
                first_grounded = when
        if not hit:
            continue
        grounded_nodes += 1
        horizon = measurement.horizons.get(node_id)
        if horizon is not None and first_grounded is not None and first_grounded <= horizon:
            grounded_within += 1

    denominator = measurement.nodes_with_closed_consumer
    return {
        "consumed_later": grounded_nodes,
        "consumed_within_7d": grounded_within,
        "nodes_with_closed_consumer": denominator,
        "denominator": "nodes_with_closed_consumer",
        "rate": _ratio(grounded_nodes, denominator),
        "rate_pct": _pct(grounded_nodes, denominator),
        "within_7d_rate": _ratio(grounded_within, denominator),
        "rate_over_cohort": _ratio(grounded_nodes, measurement.nodes),
        "consuming_events": len(measurement.consuming_events),
        "consuming_events_closed_by_trace": len(measurement.closed_consuming_events),
        "closed_event_share": _ratio(
            len(measurement.closed_consuming_events), len(measurement.consuming_events)
        ),
        "grounded_consuming_events": len(grounded_events),
        "min_containment": min_containment,
        "within_days": within_days,
    }


def _empty_grounded_block(*, min_containment: float, within_days: int) -> dict[str, Any]:
    return {
        "consumed_later": None,
        "consumed_within_7d": None,
        "nodes_with_closed_consumer": None,
        "denominator": "nodes_with_closed_consumer",
        "rate": None,
        "rate_pct": None,
        "within_7d_rate": None,
        "rate_over_cohort": None,
        "consuming_events": None,
        "consuming_events_closed_by_trace": None,
        "closed_event_share": None,
        "grounded_consuming_events": None,
        "min_containment": min_containment,
        "within_days": within_days,
        "skipped": True,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _measurement_to_dict(
    measurement: CohortMeasurement,
    grounded: dict[str, Any],
    *,
    within_days: int,
) -> dict[str, Any]:
    return {
        "id": measurement.selector.label,
        "selector": measurement.selector.to_dict(),
        "nodes": measurement.nodes,
        "consumed_later": measurement.consumed_later,
        "consumed_within_7d": measurement.consumed_within_7d,
        "rate": _ratio(measurement.consumed_later, measurement.nodes),
        "rate_pct": _pct(measurement.consumed_later, measurement.nodes),
        "within_7d_rate": _ratio(measurement.consumed_within_7d, measurement.nodes),
        "within_7d_rate_pct": _pct(measurement.consumed_within_7d, measurement.nodes),
        "within_days": within_days,
        "grounded": grounded,
    }


def _repo_revision() -> str | None:
    root = Path(__file__).resolve().parents[3]
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = proc.stdout.strip()
    return revision if proc.returncode == 0 and revision else None


def _count(connection: sqlite3.Connection, table: str, as_of: str | None) -> int:
    if as_of is None:
        row = connection.execute(f"SELECT count(*) FROM {table}").fetchone()
    else:
        row = connection.execute(
            f"SELECT count(*) FROM {table} WHERE created_at <= ?", (as_of,)
        ).fetchone()
    return int(row[0])


CONSUMED_RULE = (
    "node.id appears in json.loads(recall_events.results)[].node_id of an event "
    "whose created_at is strictly greater than the node's created_at"
)

GROUNDED_RULE = (
    "the consuming event was closed by a memory_remember trace "
    "(feedback_trace_id not null) whose content grounds the node at "
    "IDF containment >= min_containment"
)


def run_metric(
    db_path: str | Path,
    *,
    window: str | tuple[str, str],
    as_of: str | None = None,
    agents: Sequence[str] = (),
    context_keys: Sequence[tuple[str, str | None]] = (),
    within_days: int = DEFAULT_WITHIN_DAYS,
    min_containment: float = DEFAULT_MIN_CONTAINMENT,
    grounded: bool = True,
    sample_events: int = LIVE_PATH_SAMPLE_EVENTS,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Measure retrospective consumption, read-only, and return the artifact."""

    bounds = parse_window(window) if isinstance(window, str) else (
        normalize_instant(window[0]),
        normalize_instant(window[1]),
    )
    cutoff = normalize_instant(as_of) if as_of else None
    selectors = build_selectors(bounds, agents=agents, context_keys=context_keys)
    notes: list[str] = []

    connection = open_readonly(db_path)
    try:
        index = load_consumption_index(connection, as_of=cutoff)
        measurements = [
            measure_cohort(
                index,
                load_cohort_nodes(connection, selector, as_of=cutoff),
                selector,
                within_days=within_days,
            )
            for selector in selectors
        ]

        grounding_diagnostics: dict[str, Any] | None = None
        crosscheck: dict[str, Any] | None = None
        relevant: set[int] = set()
        for measurement in measurements:
            relevant |= measurement.closed_consuming_events
        if grounded and relevant:
            # Grounding is decided per event from that event's own trace and
            # results, so restricting the pass to the events that can affect a
            # cohort changes no verdict — only the cost.
            subset = [index.events[ordinal] for ordinal in sorted(relevant)]
            label_data = build_label_data(connection, subset)
            grounding_diagnostics = apply_grounding(
                subset, label_data, min_containment
            )
            grounding_diagnostics["corpus"] = "cohort_relevant_events"
            grounding_diagnostics["corpus_events"] = len(subset)
            crosscheck = _live_path_crosscheck(
                connection,
                index,
                sorted(relevant),
                min_containment=min_containment,
                sample_events=sample_events,
            )
        elif grounded:
            notes.append(
                "no consuming event in any cohort was closed by a remember trace: "
                "the grounded variant has an empty denominator, not a zero rate"
            )

        cohorts = [
            _measurement_to_dict(
                measurement,
                _grounded_block(
                    measurement,
                    index,
                    min_containment=min_containment,
                    within_days=within_days,
                )
                if grounded
                else _empty_grounded_block(
                    min_containment=min_containment, within_days=within_days
                ),
                within_days=within_days,
            )
            for measurement in measurements
        ]

        meta = {
            "artifact": "trace_usage_linkage",
            "question": (
                "were the nodes written in this window actually delivered by later recalls"
            ),
            "db_path": str(db_path),
            "open_mode": "ro",
            "as_of": cutoff,
            "generated_at_utc": generated_at
            or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lm_revision": _repo_revision(),
            "nodes_total": _count(connection, "nodes", cutoff),
            "recall_events_total": _count(connection, "recall_events", cutoff),
            "events_with_deliveries": index.stats["events_with_deliveries"],
            "events_closed_by_trace": index.stats["events_closed_by_trace"],
            "closed_event_share": _ratio(
                index.stats["events_closed_by_trace"], index.stats["events_with_deliveries"]
            ),
            "index_stats": dict(index.stats),
        }
    finally:
        connection.close()

    if cutoff is None:
        notes.append(
            "no --as-of cutoff: this run is not reproducible, because later recall "
            "events can only add consumption to an already-closed window"
        )
    notes.append(
        "cohort is every nodes row in the window, all levels, decayed included; "
        "level='trace' only does not reproduce the published baseline"
    )
    notes.append(
        "grounded.rate divides by nodes_with_closed_consumer, not by the cohort: "
        "most recall events carry no feedback_trace_id"
    )
    notes.append(
        "dividing grounded consumption by the whole cohort understates it "
        "several-fold; that figure is published as grounded.rate_over_cohort"
    )
    notes.append(
        "consumed_later is a lower bound on usefulness: a node never delivered "
        "again may simply never have been queried for"
    )

    report: dict[str, Any] = {
        "meta": meta,
        "cohort_definition": {
            "consumed_rule": CONSUMED_RULE,
            "grounded_rule": GROUNDED_RULE,
            "window_start": bounds[0],
            "window_end": bounds[1],
            "window_bounds": "half-open [start, end)",
            "levels": "all",
            "include_decayed": True,
            "within_days": within_days,
            "min_containment": min_containment,
            "null_agent_tokens": list(NULL_AGENT_TOKENS),
            "selectors": [selector.to_dict() for selector in selectors],
        },
        "cohorts": cohorts,
        "grounding": {
            "diagnostics": grounding_diagnostics,
            "live_path_crosscheck": crosscheck,
        },
        "notes": notes,
    }
    check_privacy(report)
    return report
