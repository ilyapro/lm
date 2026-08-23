#!/usr/bin/env python3
"""Follow-signal census over matured recall delivery windows.

What this answers, and why it is a script rather than an argument
-----------------------------------------------------------------

``recall_delivery_history`` now carries two different claims about the same
24-hour window.  The three frozen bits (``transport_matched``,
``transport_consumed``, ``fallback_consumed``) all resolve to *a later recall
returned this node again* — the ranker's own echo, which a hub schema the
trigger boost mixes into nearly every recall satisfies forever without anybody
reading it.  ``lookup_consumed`` resolves to *somebody fetched this exact ULID*,
which only a reader who was handed the ULID does.

How strongly the second should count against the first is the question the
pool-gate valves need answered, and the project's rule (the same one
``LM_DRAIN_NEAR_DUP_SUPERSEDES`` was fixed under) is that it gets answered by
measurement over accrued windows, not by intuition.  This script produces the
measurement; ``artifacts/recall-map/follow-signal/procedure.md`` turns a
distribution into thresholds.  Neither sets a default: turning a valve on stays
the operator's decision.

The four classes, in the priority order they are assigned
---------------------------------------------------------

1. ``lookup_followed`` — ``lookup_consumed = 1``.  Exogenous by construction.
2. ``ask_follow`` — a later recall inside ``(delivered_at, outcome_end]`` asked
   a query that echoes the delivered cluster's ``label`` or ``ask_hint``.  The
   probe is ``recall_map._echoes`` itself, imported rather than reimplemented,
   so a disagreement with the live curtail rule would be a bug in the module
   and not a judgement call here.  Exogenous, but weaker: a query is a much
   commoner act than an id-fetch, and the overlap rule is a heuristic.
3. ``redelivered_only`` — neither of the above, and the frozen consumed bit is
   set.  This is the endogenous class the whole exercise exists to price.
4. ``nothing`` — none of the three.

Known versus NULL is the axis that decides what may be concluded at all.  A
window whose ``lookup_consumed`` is NULL closed before this database recorded
any lookup; it is *unobservable*, not *unfollowed*, and it is reported in its
own partition rather than folded into the denominator.  A database still on
ledger format 1 has no ``lookup_consumed`` column at all: every window is NULL,
which is the honest reading and is exactly what a corpus written by a server
that predates the feature should say.

What the ask-follow probe can and cannot reach
-----------------------------------------------

Only a *map medoid* window carries a label.  ``recall_delivery_history`` is the
union, per event, of the organic result tail (ranks 3..8) and the delivered
map's medoids (``storage._organic_tail_node_ids`` /
``storage._recall_map_medoid_ids``), and the organic limb has no phrasing a
later query could echo.  Those windows are counted as
``ask_follow_probe: "not_applicable"`` — never as a probe that fired and found
nothing, which would understate exogenous follow by the ratio of the two limbs.

Correlation width is reported twice, because the choice is arguable and the
gap between the two answers is the honest error bar: ``any`` counts a later
recall from any transport (mirroring the lookup correlation, which is global
by necessity — the probe that offers a card and the agent that fetches it sit
on different MCP connections), and ``transport`` counts only later recalls from
the same transport session.

Usage::

    python3 scripts/recall_map_follow_signal_census.py \\
        --corpus 'field:live-global=/home/user/.local/share/living-memory/global.sqlite3' \\
        --out artifacts/recall-map/follow-signal/census.json

Every database is opened ``mode=ro``.  This script never writes to a corpus it
reads, never opens one through :class:`~living_memory.storage.MemoryStore`
(which would migrate it), and never touches a server.  ``--build-synthetic``
is the one writing path, and it writes only to the scratch path it is given.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence
import argparse
import json
import math
import os
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from living_memory.grounding import token_set  # noqa: E402
from living_memory.recall_map import (  # noqa: E402
    CURTAIL_QUERY_OVERLAP,
    MAX_CLUSTERS,
    _delivered_items,
    _echoes,
)
from living_memory.storage import (  # noqa: E402
    MAX_RECALL_HISTORY_DELIVERIES_PER_NODE,
    RECALL_DELIVERY_HISTORY_HEAD_CUT,
    RECALL_DELIVERY_HISTORY_HORIZON_HOURS,
    RECALL_DELIVERY_HISTORY_ITEMS_PER_EVENT,
    RECALL_DELIVERY_HISTORY_STATE_TABLE,
    RECALL_DELIVERY_HISTORY_TABLE,
    RECALL_HISTORY_RESULT_TABLE,
    RECALL_LOOKUP_EVENT_TABLE,
    _RECALL_DELIVERY_HISTORY_FORMAT,
)
from living_memory.temporal import parse_timestamp  # noqa: E402

#: The cohort exemplar named in the goal: a ``project:ae`` procedure schema the
#: map has offered for months with a usefulness verdict of 0.078.  Pinned by
#: default so the artifact always carries its row, whether or not it happens to
#: rank inside the top-sticky cut of the database being read.
DEFAULT_PINNED_NODES: tuple[str, ...] = ("01KSB3GDM0J3ZB24QSRA97RRWJ",)

#: Ranked nodes reported individually in the sticky cohort.
DEFAULT_TOP_STICKY = 10

CLASS_LOOKUP = "lookup_followed"
CLASS_ASK = "ask_follow"
CLASS_REDELIVERED = "redelivered_only"
CLASS_NOTHING = "nothing"
CLASSES: tuple[str, ...] = (CLASS_LOOKUP, CLASS_ASK, CLASS_REDELIVERED, CLASS_NOTHING)

#: The two widths the ask-follow correlation is reported at.  ``any_transport``
#: mirrors the lookup correlation, which has to be global — the probe that
#: offers a card and the agent that fetches it sit on different MCP
#: connections.  What justifies that width for an id-fetch is the rarity of the
#: act, and a *query* is not rare, so the same width is not automatically
#: justified here.  ``same_transport`` is the conservative reading, and the
#: distance between the two is the artifact's honest error bar on this probe.
ASK_CORRELATIONS: tuple[str, ...] = ("any_transport", "same_transport")

#: Names for the five frozen relevance features, in ``_relevance_features``
#: order.  Names only — the means, scales and arithmetic stay where they are.
RELEVANCE_FEATURE_NAMES: tuple[str, ...] = (
    "level_is_schema",
    "log1p_matured",
    "log1p_unconsumed",
    "log1p_trailing_nonconsumed",
    "consumed_ratio",
)

#: Half-width a rate must reach before this artifact calls it decided.  Not a
#: statistical law — a pre-registration, so "under-powered" is a verdict the
#: numbers can return rather than a word an author chooses.
TARGET_HALF_WIDTH = 0.10

#: Two-sided 95% normal quantile, for the Wilson interval below.
Z_95 = 1.959963984540054

_LEDGER_INSTANT = "%Y-%m-%dT%H:%M:%S.%fZ"


# ----------------------------------------------------------------------
# Small statistics
# ----------------------------------------------------------------------


def wilson_interval(successes: int, total: int, z: float = Z_95) -> dict[str, Any]:
    """Wilson score interval, which stays inside [0, 1] at the edges.

    The normal approximation is what an under-powered artifact must not use:
    at ``0/3`` it reports ``0 +/- 0`` and invites exactly the confident
    threshold this measurement exists to refuse.  Wilson returns ``[0, 0.56]``
    for the same data, which is the honest statement.
    """

    if total <= 0:
        return {"point": None, "low": None, "high": None, "half_width": None}
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    spread = (
        z
        * math.sqrt(phat * (1.0 - phat) / total + z * z / (4 * total * total))
        / denominator
    )
    low = max(0.0, centre - spread)
    high = min(1.0, centre + spread)
    return {
        "point": round(phat, 6),
        "low": round(low, 6),
        "high": round(high, 6),
        "half_width": round((high - low) / 2.0, 6),
    }


def windows_for_half_width(
    half_width: float, p: float = 0.5, z: float = Z_95
) -> int:
    """Known windows needed before a rate near ``p`` is decided to ``half_width``.

    The normal sample-size formula, used only to say how far away a decision
    is.  ``p = 0.5`` is the worst case and therefore the default: a planner
    who does not yet know the rate should budget for the widest one.
    """

    if half_width <= 0:
        raise ValueError("half_width must be positive")
    return int(math.ceil((z * z * p * (1.0 - p)) / (half_width * half_width)))


# ----------------------------------------------------------------------
# Corpora
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusSpec:
    """One database to census, and the claim its numbers are allowed to make.

    ``provenance`` is not decoration.  ``field`` numbers may set a threshold;
    ``synthetic`` numbers may only prove the classifier discriminates, and the
    artifact says so beside every one of them.
    """

    label: str
    provenance: str
    path: Path
    note: str = ""

    @classmethod
    def parse(cls, raw: str) -> "CorpusSpec":
        """``PROVENANCE:LABEL=PATH``, with an optional ``#note`` suffix."""

        spec, _, note = raw.partition("#")
        head, sep, path = spec.partition("=")
        if not sep or not path.strip():
            raise argparse.ArgumentTypeError(
                f"expected PROVENANCE:LABEL=PATH, got {raw!r}"
            )
        provenance, _, label = head.partition(":")
        provenance = provenance.strip().lower()
        label = (label or provenance).strip()
        if provenance not in {"field", "snapshot", "synthetic"}:
            raise argparse.ArgumentTypeError(
                f"provenance must be field|snapshot|synthetic, got {provenance!r}"
            )
        return cls(
            label=label,
            provenance=provenance,
            path=Path(path.strip()).expanduser(),
            note=note.strip(),
        )


def open_readonly(path: Path) -> sqlite3.Connection:
    """Open one corpus for reading and prove to the caller that it is read-only.

    ``mode=ro`` alone is the boundary the goal draws around live databases;
    ``query_only`` is asserted on top of it so a future edit to this file
    cannot quietly acquire a write path to a database somebody is using.
    ``immutable=1`` is deliberately *not* used: a live database has a WAL, and
    ignoring it would read a torn prefix of the ledger.
    """

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = 1")
    row = connection.execute("PRAGMA query_only").fetchone()
    if row is None or not int(row[0]):  # pragma: no cover - sqlite guarantees it
        connection.close()
        raise RuntimeError(f"connection to {path} is not query-only")
    return connection


@dataclass(frozen=True)
class SchemaProbe:
    """What the ledger of one corpus is able to answer, before asking it."""

    ledger_present: bool
    lookup_column_present: bool
    lookup_table_present: bool
    format_version: int | None
    complete: bool | None
    unavailable_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ledger_present": self.ledger_present,
            "lookup_column_present": self.lookup_column_present,
            "lookup_table_present": self.lookup_table_present,
            "ledger_format_version": self.format_version,
            "ledger_format_expected": _RECALL_DELIVERY_HISTORY_FORMAT,
            "ledger_complete": self.complete,
            "ledger_unavailable_reason": self.unavailable_reason,
        }


def probe_schema(connection: sqlite3.Connection) -> SchemaProbe:
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    ledger = RECALL_DELIVERY_HISTORY_TABLE in tables
    columns: set[str] = set()
    if ledger:
        columns = {
            str(row["name"])
            for row in connection.execute(
                f"PRAGMA table_info({RECALL_DELIVERY_HISTORY_TABLE})"
            )
        }
    state = None
    if RECALL_DELIVERY_HISTORY_STATE_TABLE in tables:
        state = connection.execute(
            f"""
            SELECT format_version, complete, unavailable_reason
            FROM {RECALL_DELIVERY_HISTORY_STATE_TABLE} WHERE singleton = 1
            """
        ).fetchone()
    return SchemaProbe(
        ledger_present=ledger,
        lookup_column_present="lookup_consumed" in columns,
        lookup_table_present=RECALL_LOOKUP_EVENT_TABLE in tables,
        format_version=None if state is None else int(state["format_version"]),
        complete=None if state is None else bool(int(state["complete"])),
        unavailable_reason=(
            None if state is None else (state["unavailable_reason"] or None)
        ),
    )


# ----------------------------------------------------------------------
# Recall events: the consumer side of every window
# ----------------------------------------------------------------------


@dataclass
class EventIndex:
    """Every recall event of a corpus, time-sorted and inverted by token.

    The inverted index exists because the naive shape of the ask-follow probe
    is quadratic: a hundred thousand windows times the several hundred recalls
    inside each 24-hour horizon.  A cluster label is two to four tokens that
    the cascade chose *for being distinguishing*, so its posting lists are
    short, and the range slice of a posting list is two bisections.
    """

    ids: list[str]
    times: list[datetime]
    transports: list[str | None]
    queries: list[frozenset[str]]
    postings: dict[str, list[int]]
    maps: dict[str, dict[str, tuple[frozenset[str], frozenset[str]]]]
    map_events: int
    unparsed_maps: int

    @property
    def total(self) -> int:
        return len(self.ids)


def load_events(connection: sqlite3.Connection) -> EventIndex:
    """Read ``recall_events`` once and build everything the probe needs.

    The map payloads are decoded through :func:`recall_map._delivered_items`,
    the same reader the live curtail rule uses, so a payload shape this build
    cannot parse contributes no phrasings here for the same reason it
    contributes no evidence there.
    """

    rows = connection.execute(
        """
        SELECT id, created_at, query, transport_session_id, recall_map
        FROM recall_events
        ORDER BY created_at, rowid
        """
    ).fetchall()

    ids: list[str] = []
    times: list[datetime] = []
    transports: list[str | None] = []
    queries: list[frozenset[str]] = []
    postings: dict[str, list[int]] = defaultdict(list)
    maps: dict[str, dict[str, tuple[frozenset[str], frozenset[str]]]] = {}
    map_events = 0
    unparsed_maps = 0

    for row in rows:
        occurred = parse_timestamp(row["created_at"])
        if occurred is None:
            continue
        ordinal = len(ids)
        ids.append(str(row["id"]))
        times.append(occurred.astimezone(UTC))
        transports.append(row["transport_session_id"] or None)
        tokens = token_set(str(row["query"] or ""))
        queries.append(tokens)
        for token in tokens:
            postings[token].append(ordinal)

        raw_map = row["recall_map"]
        if raw_map is None:
            continue
        map_events += 1
        try:
            payload = json.loads(raw_map)
        except (TypeError, ValueError):
            unparsed_maps += 1
            continue
        items = _delivered_items(payload)
        phrasings = {
            item.medoid_id: (item.label, item.ask_hint)
            for item in items
            if item.medoid_id
        }
        if phrasings:
            maps[str(row["id"])] = phrasings
        elif items:
            unparsed_maps += 1

    # The events arrive ordered by ``created_at``; ties are broken by rowid,
    # which is insertion order.  Both bisections below need a non-decreasing
    # sequence and nothing stronger, so no re-sort is needed.
    return EventIndex(
        ids=ids,
        times=times,
        transports=transports,
        queries=queries,
        postings=dict(postings),
        maps=maps,
        map_events=map_events,
        unparsed_maps=unparsed_maps,
    )


def echo_matches(
    index: EventIndex, phrasing: frozenset[str], lo: int, hi: int
) -> list[int]:
    """Ordinals in ``[lo, hi)`` whose query echoes ``phrasing``.

    The threshold is ``CURTAIL_QUERY_OVERLAP * len(phrasing)`` and the
    comparison is ``>=``, copied from ``_echoes`` rather than restated: a
    two-token label needs one shared token, a four-token label needs two.
    :func:`verify_echo_agreement` pins this fast path against the module's own
    function on real data every run.
    """

    if not phrasing or hi <= lo:
        return []
    needed = CURTAIL_QUERY_OVERLAP * len(phrasing)
    counts: Counter[int] = Counter()
    for token in phrasing:
        postings = index.postings.get(token)
        if not postings:
            continue
        start = bisect_left(postings, lo)
        stop = bisect_left(postings, hi)
        for ordinal in postings[start:stop]:
            counts[ordinal] += 1
    return sorted(ordinal for ordinal, count in counts.items() if count >= needed)


def verify_echo_agreement(
    index: EventIndex, samples: Sequence[tuple[frozenset[str], int, int]]
) -> dict[str, Any]:
    """Replay a sample of probe calls through ``_echoes`` itself.

    An inverted index that answers a containment question by counting postings
    is the kind of optimisation that is right until a tokenizer change makes it
    subtly wrong.  This re-answers a bounded sample the slow way — the module's
    own predicate over every query in the range — and reports the comparison in
    the artifact, so the shortcut is falsifiable rather than trusted.
    """

    checked = 0
    disagreements = 0
    for phrasing, lo, hi in samples:
        fast = set(echo_matches(index, phrasing, lo, hi))
        slow = {
            ordinal
            for ordinal in range(lo, hi)
            if _echoes(phrasing, index.queries[ordinal])
        }
        checked += 1
        if fast != slow:
            disagreements += 1
    return {"probe_calls_replayed": checked, "disagreements": disagreements}


# ----------------------------------------------------------------------
# Window classification
# ----------------------------------------------------------------------


@dataclass
class Tally:
    """One partition of windows, counted along every axis the valves need."""

    windows: int = 0
    map_medoid_windows: int = 0
    organic_only_windows: int = 0
    lookup_followed: int = 0
    lookup_absent: int = 0
    lookup_unknown: int = 0
    redelivered: int = 0
    ask_applicable: int = 0
    ask_fired_any: int = 0
    ask_fired_transport: int = 0
    #: Class counters, one pair per ask-follow correlation width.  Both are
    #: kept because the field gap between them is a factor of six, and an
    #: artifact that published one of them would be publishing a choice.
    known_classes: dict[str, Counter[str]] = field(
        default_factory=lambda: {name: Counter() for name in ASK_CORRELATIONS}
    )
    unknown_classes: dict[str, Counter[str]] = field(
        default_factory=lambda: {name: Counter() for name in ASK_CORRELATIONS}
    )
    #: ``(lookup, redelivered)`` over known windows — the 2x2 the weighting
    #: question is literally about.
    lookup_by_redelivery: Counter[tuple[int, int]] = field(default_factory=Counter)
    #: ``(ask_follow, redelivered)`` over ask-applicable windows, per
    #: correlation width.  The only exogenous-versus-endogenous contingency a
    #: format-1 corpus can fill in at all.
    ask_by_redelivery: dict[str, Counter[tuple[int, int]]] = field(
        default_factory=lambda: {name: Counter() for name in ASK_CORRELATIONS}
    )

    def add(self, window: "Window") -> None:
        self.windows += 1
        if window.is_map_medoid:
            self.map_medoid_windows += 1
        else:
            self.organic_only_windows += 1
        if window.redelivered:
            self.redelivered += 1
        if window.lookup is True:
            self.lookup_followed += 1
        elif window.lookup is False:
            self.lookup_absent += 1
        else:
            self.lookup_unknown += 1
        if window.ask_applicable:
            self.ask_applicable += 1
            if window.ask_any:
                self.ask_fired_any += 1
            if window.ask_transport:
                self.ask_fired_transport += 1
        for correlation in ASK_CORRELATIONS:
            fired = window.ask_fired(correlation)
            if window.ask_applicable:
                self.ask_by_redelivery[correlation][
                    (int(fired), int(window.redelivered))
                ] += 1
            bucket = (
                self.unknown_classes if window.lookup is None else self.known_classes
            )
            bucket[correlation][window.window_class(correlation)] += 1
        if window.lookup is not None:
            self.lookup_by_redelivery[
                (int(window.lookup), int(window.redelivered))
            ] += 1

    @property
    def known(self) -> int:
        return self.lookup_followed + self.lookup_absent

    def to_dict(self) -> dict[str, Any]:
        known = self.known
        null = self.lookup_unknown
        null_classes = tuple(name for name in CLASSES if name != CLASS_LOOKUP)
        return {
            "windows": self.windows,
            "map_medoid_windows": self.map_medoid_windows,
            "organic_only_windows": self.organic_only_windows,
            "redelivered": self.redelivered,
            "known_windows": known,
            "null_windows": null,
            "known": {
                "total": known,
                "by_ask_correlation": {
                    correlation: {
                        "classes": {
                            name: self.known_classes[correlation].get(name, 0)
                            for name in CLASSES
                        },
                        "shares": _shares(
                            self.known_classes[correlation], known, CLASSES
                        ),
                    }
                    for correlation in ASK_CORRELATIONS
                },
                "lookup_by_redelivery": _contingency(self.lookup_by_redelivery),
                "lookup_follow_rate": wilson_interval(self.lookup_followed, known),
            },
            "null": {
                "total": null,
                "by_ask_correlation": {
                    correlation: {
                        "classes": {
                            name: self.unknown_classes[correlation].get(name, 0)
                            for name in null_classes
                        },
                        "shares": _shares(
                            self.unknown_classes[correlation], null, null_classes
                        ),
                    }
                    for correlation in ASK_CORRELATIONS
                },
            },
            "ask_follow_probe": {
                "applicable_windows": self.ask_applicable,
                "not_applicable_windows": self.windows - self.ask_applicable,
                "fired": {
                    "any_transport": self.ask_fired_any,
                    "same_transport": self.ask_fired_transport,
                },
                "rate": {
                    correlation: wilson_interval(
                        self.ask_fired_any
                        if correlation == "any_transport"
                        else self.ask_fired_transport,
                        self.ask_applicable,
                    )
                    for correlation in ASK_CORRELATIONS
                },
                "ask_by_redelivery": {
                    correlation: _contingency(self.ask_by_redelivery[correlation])
                    for correlation in ASK_CORRELATIONS
                },
                "redelivery_precision_for_ask_follow": {
                    correlation: _precision(self.ask_by_redelivery[correlation])
                    for correlation in ASK_CORRELATIONS
                },
            },
        }


def _shares(
    counts: Counter[str], total: int, names: Sequence[str]
) -> dict[str, float | None]:
    if total <= 0:
        return {name: None for name in names}
    return {name: round(counts.get(name, 0) / total, 6) for name in names}


def _contingency(counts: Counter[tuple[int, int]]) -> dict[str, int]:
    """``signal_x__redelivered_y`` cells, always all four, always explicit."""

    return {
        f"signal_{signal}__redelivered_{redelivered}": counts.get(
            (signal, redelivered), 0
        )
        for signal in (1, 0)
        for redelivered in (1, 0)
    }


def _precision(counts: Counter[tuple[int, int]]) -> dict[str, Any]:
    """How much a re-delivery is worth as a claim of exogenous follow.

    ``precision`` is ``P(signal | redelivered)``: of the windows the endogenous
    bit calls consumed, the share an exogenous probe also fires on.  ``lift`` is
    that against ``P(signal | not redelivered)``.  A lift near 1.0 is the
    quantitative form of "the re-delivery bit carries no information about
    whether anyone read it"; a lift far above 1.0 would mean re-delivery is a
    noisy but real proxy and should keep some weight.
    """

    redelivered = counts.get((1, 1), 0) + counts.get((0, 1), 0)
    not_redelivered = counts.get((1, 0), 0) + counts.get((0, 0), 0)
    with_signal = counts.get((1, 1), 0)
    without = counts.get((1, 0), 0)
    precision = wilson_interval(with_signal, redelivered)
    baseline = wilson_interval(without, not_redelivered)
    lift: float | None = None
    if precision["point"] is not None and baseline["point"]:
        lift = round(precision["point"] / baseline["point"], 6)
    return {
        "p_signal_given_redelivered": precision,
        "p_signal_given_not_redelivered": baseline,
        "lift": lift,
    }


@dataclass(frozen=True, slots=True)
class Window:
    """One matured ``(delivery_event_id, node_id)`` row, fully classified."""

    node_id: str
    delivery_event_id: str
    delivered_at: str
    outcome_end: str
    redelivered: bool
    lookup: bool | None
    is_map_medoid: bool
    ask_applicable: bool
    ask_any: bool
    ask_transport: bool

    def ask_fired(self, correlation: str) -> bool:
        return self.ask_any if correlation == "any_transport" else self.ask_transport

    def window_class(self, correlation: str = "any_transport") -> str:
        if self.lookup is True:
            return CLASS_LOOKUP
        if self.ask_fired(correlation):
            return CLASS_ASK
        if self.redelivered:
            return CLASS_REDELIVERED
        return CLASS_NOTHING


def classify_windows(
    connection: sqlite3.Connection,
    index: EventIndex,
    *,
    probe: SchemaProbe,
    decision_at: datetime,
) -> tuple[list[Window], dict[str, Any]]:
    """Every matured window of the corpus, classified once.

    Maturation is the ledger's own rule, unchanged: ``delivered_at <
    decision_at AND outcome_end <= decision_at``.  Both stored columns are
    written through ``storage._normalize_recall_history_instant``, so the
    fixed-width text compares correctly against a normalised bound in SQL and
    no reformatting is needed on the hot path.
    """

    bound = decision_at.astimezone(UTC).strftime(_LEDGER_INSTANT)
    lookup_select = (
        "lookup_consumed" if probe.lookup_column_present else "NULL AS lookup_consumed"
    )
    rows = connection.execute(
        f"""
        SELECT delivery_event_id, node_id, delivered_at, outcome_end,
               transport_session_id, transport_matched, transport_consumed,
               fallback_consumed, {lookup_select}
        FROM {RECALL_DELIVERY_HISTORY_TABLE}
        WHERE delivered_at < ? AND outcome_end <= ?
        """,
        (bound, bound),
    ).fetchall()

    # One range resolution per delivery event, shared by its up-to-six rows.
    ranges: dict[str, tuple[int, int]] = {}
    echo_cache: dict[tuple[str, frozenset[str]], tuple[bool, bool]] = {}
    probe_samples: list[tuple[frozenset[str], int, int]] = []
    windows: list[Window] = []
    unresolved_instants = 0

    for row in rows:
        event_id = str(row["delivery_event_id"])
        node_id = str(row["node_id"])
        phrasings = index.maps.get(event_id, {}).get(node_id)
        is_map_medoid = phrasings is not None
        redelivered = bool(
            row["transport_consumed"]
            if row["transport_matched"]
            else row["fallback_consumed"]
        )
        raw_lookup = row["lookup_consumed"]
        lookup = None if raw_lookup is None else bool(raw_lookup)

        ask_any = False
        ask_transport = False
        ask_applicable = False
        if phrasings is not None:
            window_range = ranges.get(event_id)
            if window_range is None:
                start = parse_timestamp(row["delivered_at"])
                end = parse_timestamp(row["outcome_end"])
                if start is None or end is None:  # pragma: no cover - normalised
                    unresolved_instants += 1
                    window_range = (0, 0)
                else:
                    window_range = (
                        bisect_right(index.times, start.astimezone(UTC)),
                        bisect_right(index.times, end.astimezone(UTC)),
                    )
                ranges[event_id] = window_range
            lo, hi = window_range
            ask_applicable = True
            transport = row["transport_session_id"] or None
            for phrasing in phrasings:
                if not phrasing:
                    continue
                cache_key = (event_id, phrasing)
                cached = echo_cache.get(cache_key)
                if cached is None:
                    matches = echo_matches(index, phrasing, lo, hi)
                    cached = (
                        bool(matches),
                        any(index.transports[i] == transport for i in matches)
                        if transport
                        else False,
                    )
                    echo_cache[cache_key] = cached
                    if len(probe_samples) < 200 and hi > lo:
                        probe_samples.append((phrasing, lo, hi))
                ask_any = ask_any or cached[0]
                ask_transport = ask_transport or cached[1]

        windows.append(
            Window(
                node_id=node_id,
                delivery_event_id=event_id,
                delivered_at=str(row["delivered_at"]),
                outcome_end=str(row["outcome_end"]),
                redelivered=redelivered,
                lookup=lookup,
                is_map_medoid=is_map_medoid,
                ask_applicable=ask_applicable,
                ask_any=ask_any,
                ask_transport=ask_transport,
            )
        )

    diagnostics = {
        "ledger_rows_scanned": len(rows),
        "delivery_events_with_ask_probe": len(ranges),
        "unresolved_instants": unresolved_instants,
        "echo_agreement": verify_echo_agreement(index, probe_samples),
    }
    return windows, diagnostics


# ----------------------------------------------------------------------
# Per-node view: M/C/K, stickiness, and the cohort
# ----------------------------------------------------------------------


def matured_triples(windows: Sequence[Window]) -> dict[str, dict[str, Any]]:
    """Recompute the ledger's own M/C/K per node, ordering rule included.

    Deliberately a reimplementation rather than a call into
    :meth:`MemoryStore.matured_recall_history`: that method needs a writable
    store, which would migrate a live database this script is forbidden to
    touch.  ``tests/test_recall_map_follow_signal_census.py`` pins the two
    against each other on a corpus where both can run, so the copy is checked
    rather than asserted.
    """

    by_node: dict[str, list[Window]] = defaultdict(list)
    for window in windows:
        by_node[window.node_id].append(window)

    triples: dict[str, dict[str, Any]] = {}
    for node_id, rows in by_node.items():
        if len(rows) > MAX_RECALL_HISTORY_DELIVERIES_PER_NODE:
            triples[node_id] = {
                "available": False,
                "unavailable_reason": "history_truncated",
                "matured": None,
                "consumed": None,
                "trailing_nonconsumed": None,
                "lookup_consumed": None,
                "lookup_known": None,
                "lookup_trailing_absent": None,
                "rows": len(rows),
            }
            continue
        ordered = sorted(
            rows,
            key=lambda w: (
                w.outcome_end,
                w.delivered_at,
                int(w.redelivered),
                w.delivery_event_id,
            ),
            reverse=True,
        )
        outcomes = [w.redelivered for w in ordered]
        trailing = 0
        for consumed in outcomes:
            if consumed:
                break
            trailing += 1
        lookups = [w.lookup for w in ordered]
        lookup_trailing = 0
        for followed in lookups:
            if followed is None:
                continue
            if followed:
                break
            lookup_trailing += 1
        triples[node_id] = {
            "available": True,
            "unavailable_reason": None,
            "matured": len(outcomes),
            "consumed": sum(outcomes),
            "trailing_nonconsumed": trailing,
            "lookup_consumed": sum(1 for followed in lookups if followed),
            "lookup_known": sum(1 for followed in lookups if followed is not None),
            "lookup_trailing_absent": lookup_trailing,
            "rows": len(rows),
        }
    return triples


def _rank(
    per_node: dict[str, Tally], key: Any
) -> list[tuple[str, Tally]]:
    """Nodes ordered by one stickiness key, ties broken by id for determinism."""

    return sorted(
        per_node.items(), key=lambda item: (*key(item[1]), item[0]), reverse=True
    )


class _NodeShim:
    """The only attribute ``_relevance_features`` reads off a node.

    Deserializing whole :class:`~living_memory.models.Node` objects to ask the
    scorer one question would make the census depend on the node table's shape
    for no gain; the frozen feature vector reads ``level`` and nothing else.
    """

    __slots__ = ("level",)

    def __init__(self, level: Any) -> None:
        self.level = level


def frozen_relevance(level: Any, triple: Mapping[str, Any]) -> dict[str, Any]:
    """What the untouched relevance evaluator sees for one node, today.

    Computed through ``recall_map.relevance_score`` on a
    :class:`MaturedRecallHistory` rebuilt from this census's own triple, so the
    artifact reports the frozen scorer's actual view rather than a paraphrase
    of it.  This is the number the pool gates would sit beside, and the reason
    the sticky rows carry it: the goal's premise is that hub schemas ride on
    ``consumed = 1, K = 0`` forever, and that premise is checkable here.
    """

    from living_memory.recall_map import (  # local: keep the module header short
        RELEVANCE_FEATURE_MEANS,
        RELEVANCE_FEATURE_SCALES,
        _relevance_features,
        relevance_score,
    )
    from living_memory.storage import MaturedRecallHistory

    if triple.get("available"):
        history = MaturedRecallHistory.known(
            matured=int(triple["matured"]),
            consumed=int(triple["consumed"]),
            trailing_nonconsumed=int(triple["trailing_nonconsumed"]),
            lookup_consumed=int(triple.get("lookup_consumed") or 0),
            lookup_known=int(triple.get("lookup_known") or 0),
            lookup_trailing_absent=int(triple.get("lookup_trailing_absent") or 0),
        )
    else:
        history = MaturedRecallHistory.unavailable(
            str(triple.get("unavailable_reason") or "history_ledger_unavailable")
        )

    consumed_ratio = None
    if triple.get("available") and triple.get("matured"):
        consumed_ratio = round(int(triple["consumed"]) / int(triple["matured"]), 6)
    features = _relevance_features(_NodeShim(level), history)
    return {
        "level": level,
        "consumed_ratio": consumed_ratio,
        "trailing_nonconsumed": triple.get("trailing_nonconsumed"),
        "score": round(relevance_score(_NodeShim(level), history), 6),
        # Per-feature z-contributions to that sum.  The premise this artifact
        # was asked to check is that hub schemas survive on ``consumed = 1,
        # K = 0``; the contributions say which term actually carries them, and
        # a claim about the wrong term would send the valves after the wrong
        # thing.
        "contributions": {
            name: round((float(value) - mean) / scale, 6)
            for name, value, mean, scale in zip(
                RELEVANCE_FEATURE_NAMES,
                features,
                RELEVANCE_FEATURE_MEANS,
                RELEVANCE_FEATURE_SCALES,
                strict=True,
            )
        },
    }


def frozen_scorer_probe() -> dict[str, Any]:
    """Which way the frozen relevance score moves as the K tail grows.

    Corpus-independent and deliberately so: it interrogates the frozen
    evaluator, not a database.  It is in the census because the demotion valve
    is specified as "N known windows without exogenous follow demotes the row",
    and whether that can be expressed *through* ``trailing_nonconsumed`` or has
    to be a gate *outside* the zsum depends entirely on this sign.  Nothing
    here changes the scorer; it only asks it.
    """

    from living_memory.recall_map import (
        RELEVANCE_THRESHOLD,
        relevance_score,
    )
    from living_memory.storage import MaturedRecallHistory

    curves: dict[str, Any] = {}
    for level in ("schema", "trace"):
        points = []
        for streak in (0, 1, 3, 5, 10, 20):
            history = MaturedRecallHistory.known(
                matured=20, consumed=0, trailing_nonconsumed=streak
            )
            points.append(
                {
                    "trailing_nonconsumed": streak,
                    "score": round(relevance_score(_NodeShim(level), history), 6),
                }
            )
        curves[level] = points
    schema_curve = [point["score"] for point in curves["schema"]]
    return {
        "question": (
            "with matured = 20 and consumed = 0, does a longer unfollowed tail "
            "lower the frozen relevance score?"
        ),
        "admission_threshold": RELEVANCE_THRESHOLD,
        "curves": curves,
        "k_tail_is_demoting": schema_curve[-1] < schema_curve[0],
        "finding": (
            "the K term is monotonically increasing: a row delivered twenty "
            "times and consumed none of them scores higher the longer it has "
            "gone unconsumed. Demotion therefore cannot be obtained by feeding "
            "a better signal into trailing_nonconsumed; it has to be a gate "
            "applied outside the frozen zsum, which is also the only shape the "
            "frozen-evaluator boundary permits."
        ),
    }


def node_metadata(
    connection: sqlite3.Connection, node_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Level, scope, usefulness and access count for the reported nodes.

    ``returned_by_recalls`` reconciles the ledger with the diagnosis that
    opened this work.  The ledger records only the organic result tail
    (``RECALL_DELIVERY_HISTORY_HEAD_CUT`` onwards, for
    ``RECALL_DELIVERY_HISTORY_ITEMS_PER_EVENT`` ranks) plus map medoids, so a
    node's window count is much smaller than the number of recalls that
    returned it — for the cohort exemplar, 251 against 4 616.  Both are
    reported, because a reader who sees only one of them will read the other's
    absence as a contradiction.
    """

    if not node_ids:
        return {}
    marks = ",".join("?" for _ in node_ids)
    meta: dict[str, dict[str, Any]] = {}
    try:
        rows = connection.execute(
            f"""
            SELECT id, level, scope, access_count, usefulness_score, decayed,
                   substr(content, 1, 80) AS head
            FROM nodes WHERE id IN ({marks})
            """,
            list(node_ids),
        ).fetchall()
    except sqlite3.Error:  # pragma: no cover - a ledger-only corpus
        rows = []
    for row in rows:
        meta[str(row["id"])] = {
            "level": row["level"],
            "scope": row["scope"],
            "access_count": row["access_count"],
            "usefulness_score": row["usefulness_score"],
            "decayed": bool(row["decayed"]) if row["decayed"] is not None else None,
            "content_head": (row["head"] or "").replace("\n", " ").strip(),
        }
    try:
        returned = connection.execute(
            f"""
            SELECT node_id, COUNT(*) AS n
            FROM {RECALL_HISTORY_RESULT_TABLE}
            WHERE node_id IN ({marks})
            GROUP BY node_id
            """,
            list(node_ids),
        ).fetchall()
    except sqlite3.Error:  # pragma: no cover - a nodes-only corpus
        returned = []
    for row in returned:
        meta.setdefault(str(row["node_id"]), {})["returned_by_recalls"] = int(row["n"])
    return meta


# ----------------------------------------------------------------------
# Volume: what the numbers are allowed to support
# ----------------------------------------------------------------------


def lookup_volume(
    connection: sqlite3.Connection,
    probe: SchemaProbe,
    decision_at: datetime,
    last_activity: datetime | None,
) -> dict[str, Any]:
    """How long the lookup signal has existed in this corpus, in wall clock.

    The single most load-bearing number in the artifact.  A distribution over
    known windows means nothing without it: three windows accrued in an hour
    and three thousand accrued over a month are the same table and completely
    different evidence.

    Two clocks, because a corpus and the calendar can disagree.  ``to_decision``
    is wall clock to the maturation bound; ``to_last_activity`` stops at the
    last thing the corpus recorded.  On a live database they are the same
    number; on a snapshot — or on a fixture whose scenario is dated — the
    second is the honest one, and quoting the first would inflate the age of
    the signal by however long the file has been sitting still.
    """

    if not probe.lookup_table_present:
        return {
            "recorded": False,
            "reason": (
                "recall_lookup_events absent: this database was written by a "
                "server that predates the lookup signal, so every window is NULL"
            ),
            "requested_pairs": 0,
            "events": 0,
            "node_ids": 0,
            "first_at": None,
            "last_at": None,
            "observed_hours_to_decision": 0.0,
            "observed_hours_to_last_activity": 0.0,
            "observed_days_to_last_activity": 0.0,
        }
    row = connection.execute(
        f"""
        SELECT COUNT(*) AS pairs,
               COUNT(DISTINCT lookup_event_id) AS events,
               COUNT(DISTINCT node_id) AS node_ids,
               MIN(occurred_at) AS first_at,
               MAX(occurred_at) AS last_at
        FROM {RECALL_LOOKUP_EVENT_TABLE}
        """
    ).fetchone()
    first = parse_timestamp(row["first_at"]) if row["first_at"] else None
    last_lookup = parse_timestamp(row["last_at"]) if row["last_at"] else None
    horizon = max(
        [moment for moment in (last_activity, last_lookup) if moment is not None],
        default=None,
    )
    to_decision = 0.0
    to_activity = 0.0
    if first is not None:
        start = first.astimezone(UTC)
        to_decision = (decision_at.astimezone(UTC) - start).total_seconds() / 3600.0
        to_activity = (
            0.0
            if horizon is None
            else max(0.0, (horizon.astimezone(UTC) - start).total_seconds() / 3600.0)
        )
    return {
        "recorded": bool(row["events"]),
        "reason": None,
        "requested_pairs": int(row["pairs"] or 0),
        "events": int(row["events"] or 0),
        "node_ids": int(row["node_ids"] or 0),
        "first_at": row["first_at"],
        "last_at": row["last_at"],
        "observed_hours_to_decision": round(to_decision, 3),
        "observed_hours_to_last_activity": round(to_activity, 3),
        "observed_days_to_last_activity": round(to_activity / 24.0, 3),
    }


def power_statement(known: int, matured: int) -> dict[str, Any]:
    """What a distribution over ``known`` windows can and cannot support."""

    needed = windows_for_half_width(TARGET_HALF_WIDTH)
    interval = wilson_interval(0, known)
    return {
        "target_half_width": TARGET_HALF_WIDTH,
        "known_windows": known,
        "matured_windows": matured,
        "known_share_of_matured": (
            round(known / matured, 6) if matured else None
        ),
        "windows_needed_worst_case": needed,
        "windows_needed_if_rate_near_0_1": windows_for_half_width(
            TARGET_HALF_WIDTH, 0.1
        ),
        "powered": known >= needed,
        "widest_interval_at_this_n": interval,
        "verdict": (
            "powered"
            if known >= needed
            else "under-powered: no threshold may be drawn from this arm"
        ),
    }


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------


def census_corpus(
    spec: CorpusSpec,
    *,
    decision_at: datetime,
    top_sticky: int = DEFAULT_TOP_STICKY,
    pinned: Sequence[str] = DEFAULT_PINNED_NODES,
) -> dict[str, Any]:
    """The whole census of one database, as the artifact records it."""

    stat = os.stat(spec.path)
    connection = open_readonly(spec.path)
    try:
        probe = probe_schema(connection)
        if not probe.ledger_present:
            return {
                "label": spec.label,
                "provenance": spec.provenance,
                "note": spec.note,
                "path": str(spec.path),
                "usable": False,
                "reason": "recall_delivery_history absent",
                "schema": probe.to_dict(),
            }
        index = load_events(connection)
        windows, diagnostics = classify_windows(
            connection, index, probe=probe, decision_at=decision_at
        )
        triples = matured_triples(windows)

        overall = Tally()
        medoid_only = Tally()
        per_node: dict[str, Tally] = defaultdict(Tally)
        windows_by_node: dict[str, list[Window]] = defaultdict(list)
        for window in windows:
            overall.add(window)
            per_node[window.node_id].add(window)
            windows_by_node[window.node_id].append(window)
            if window.is_map_medoid:
                medoid_only.add(window)

        # Two stickiness rankings, because the ledger's two limbs produce two
        # different hubs and the valves only govern one of them.  Ranking by
        # re-delivery count answers the goal's question as asked; ranking by
        # map offers answers the one the pool gates would act on, and in this
        # corpus the two lists barely intersect.
        by_redelivery = _rank(per_node, lambda t: (t.redelivered, t.windows))
        by_map_offers = _rank(
            per_node, lambda t: (t.map_medoid_windows, t.redelivered, t.windows)
        )
        cohorts = {
            "by_redelivery_count": by_redelivery,
            "by_map_offer_count": by_map_offers,
        }
        cohort_ids: list[str] = []
        for ranking in cohorts.values():
            for node_id, _ in ranking[: max(0, top_sticky)]:
                if node_id not in cohort_ids:
                    cohort_ids.append(node_id)
        for node_id in pinned:
            if node_id in per_node and node_id not in cohort_ids:
                cohort_ids.append(node_id)
        meta = node_metadata(connection, cohort_ids)

        cohort_views: dict[str, Any] = {}
        for name, ranking in cohorts.items():
            tally = Tally()
            for node_id, _ in ranking[: max(0, top_sticky)]:
                for window in windows_by_node[node_id]:
                    tally.add(window)
            cohort_views[name] = {
                "members": [node_id for node_id, _ in ranking[: max(0, top_sticky)]],
                "windows": tally.to_dict(),
            }
        ranks = {
            name: {node_id: rank for rank, (node_id, _) in enumerate(ranking, start=1)}
            for name, ranking in cohorts.items()
        }

        rows: list[dict[str, Any]] = []
        for node_id in cohort_ids:
            triple = triples.get(node_id, {})
            rows.append(
                {
                    "node_id": node_id,
                    "rank": {name: ranks[name].get(node_id) for name in cohorts},
                    "pinned": node_id in pinned,
                    "node": meta.get(node_id, {}),
                    "matured_history": triple,
                    "frozen_relevance": frozen_relevance(
                        meta.get(node_id, {}).get("level"), triple
                    ),
                    "windows": per_node[node_id].to_dict(),
                }
            )

        span = connection.execute(
            """
            SELECT MIN(created_at) AS first_at, MAX(created_at) AS last_at,
                   COUNT(*) AS events
            FROM recall_events
            """
        ).fetchone()
        truncated = [
            node_id
            for node_id, triple in triples.items()
            if not triple.get("available")
        ]

        return {
            "label": spec.label,
            "provenance": spec.provenance,
            "note": spec.note,
            "path": str(spec.path),
            "usable": True,
            "field_evidence": spec.provenance == "field",
            "read": {
                "opened": "mode=ro",
                "query_only": True,
                "file_bytes": stat.st_size,
                "file_mtime_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            },
            "schema": probe.to_dict(),
            "corpus": {
                "recall_events": int(span["events"] or 0),
                "first_recall_at": span["first_at"],
                "last_recall_at": span["last_at"],
                "events_with_map": index.map_events,
                # A map that collapsed to a journal-only marker carries no
                # clusters and therefore no medoid and no phrasing.  Reported
                # because the gap between these two counts is most of the map
                # traffic, and a reader who saw only the first would divide by
                # the wrong denominator.
                "events_with_map_clusters": len(index.maps),
                "events_with_unparsed_map": index.unparsed_maps,
                "distinct_delivered_nodes": len(per_node),
                "nodes_with_truncated_history": len(truncated),
            },
            "volume": {
                "matured_windows": overall.windows,
                "known_windows": overall.known,
                "null_windows": overall.lookup_unknown,
                "lookup_signal": lookup_volume(
                    connection,
                    probe,
                    decision_at,
                    parse_timestamp(span["last_at"]) if span["last_at"] else None,
                ),
                "power": power_statement(overall.known, overall.windows),
            },
            "distribution": {
                "overall": overall.to_dict(),
                "map_medoid_windows_only": medoid_only.to_dict(),
                "top_sticky_cohorts": cohort_views,
            },
            "top_sticky_rows": rows,
            "diagnostics": diagnostics,
        }
    finally:
        connection.close()


def build_census(
    specs: Sequence[CorpusSpec],
    *,
    decision_at: datetime,
    top_sticky: int = DEFAULT_TOP_STICKY,
    pinned: Sequence[str] = DEFAULT_PINNED_NODES,
) -> dict[str, Any]:
    arms = [
        census_corpus(
            spec, decision_at=decision_at, top_sticky=top_sticky, pinned=pinned
        )
        for spec in specs
    ]
    field_arms = [arm for arm in arms if arm.get("field_evidence")]
    field_known = sum(
        int(arm["volume"]["known_windows"]) for arm in field_arms if arm.get("usable")
    )
    return {
        "artifact": "recall-map follow-signal census",
        "generated_at": decision_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/recall_map_follow_signal_census.py",
        "decision_at": decision_at.astimezone(UTC).strftime(_LEDGER_INSTANT),
        "sets_no_defaults": True,
        "definitions": {
            "window": (
                "one (delivery_event_id, node_id) row of recall_delivery_history: "
                "the frozen 24h outcome horizon opened by one recall"
            ),
            "matured": "delivered_at < decision_at AND outcome_end <= decision_at",
            "horizon_hours": RECALL_DELIVERY_HISTORY_HORIZON_HOURS,
            "classes": {
                CLASS_LOOKUP: "lookup_consumed = 1 (exogenous id-fetch in window)",
                CLASS_ASK: (
                    "a later recall in the window echoed the delivered label or "
                    "ask_hint under recall_map._echoes"
                ),
                CLASS_REDELIVERED: (
                    "neither exogenous probe fired and the frozen consumed bit "
                    "(transport_consumed if transport_matched else "
                    "fallback_consumed) is set"
                ),
                CLASS_NOTHING: "no signal of any kind",
            },
            "class_priority": list(CLASSES),
            "known_window": "lookup_consumed IS NOT NULL",
            "null_window": (
                "lookup_consumed IS NULL: the window closed before this database "
                "recorded any lookup, so its outcome is unobservable, not absent"
            ),
            "ask_follow_applicability": (
                "only map-medoid windows carry a label; organic-tail windows are "
                "counted not_applicable rather than as a probe that found nothing"
            ),
            "curtail_query_overlap": CURTAIL_QUERY_OVERLAP,
            "max_clusters": MAX_CLUSTERS,
            "ledger_organic_ranks": [
                RECALL_DELIVERY_HISTORY_HEAD_CUT,
                RECALL_DELIVERY_HISTORY_HEAD_CUT
                + RECALL_DELIVERY_HISTORY_ITEMS_PER_EVENT
                - 1,
            ],
        },
        "frozen_scorer_probe": frozen_scorer_probe(),
        "field_known_windows": field_known,
        "arms": arms,
    }


# ----------------------------------------------------------------------
# Synthetic corpus (labelled as such wherever it is used)
# ----------------------------------------------------------------------


def build_synthetic_corpus(path: Path, *, base: datetime | None = None) -> dict[str, Any]:
    """Write a scratch corpus exercising all four classes plus NULL windows.

    Every row is produced by the production write paths — recall events are
    inserted and then replayed through
    ``MemoryStore._backfill_recall_delivery_history``, lookups through
    ``MemoryStore.record_lookup_event`` — so the ledger this yields is the one
    the server would have built.  What it is *not* is evidence: the scenario
    below was chosen to make each class occur, and a distribution over a
    scenario chosen to contain a class says nothing about how often that class
    occurs in the field.  Callers label it ``synthetic`` and the artifact
    repeats the label beside every number.
    """

    from living_memory.storage import MemoryStore  # local: heavier import

    if base is None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def stamp(offset_hours: float) -> str:
        return (base + timedelta(hours=offset_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

    followed_id = ""
    asked_id = ""
    sticky_id = ""
    ignored_id = ""
    with MemoryStore(path) as store:
        followed_id = store.append_trace(
            "the canary window drill runbook, step by step",
            {"scope": "project:census", "procedure_id": "canary-window"},
        ).id
        asked_id = store.append_trace(
            "the checkpoint drill rehearsal, step by step",
            {"scope": "project:census", "procedure_id": "checkpoint-drill"},
        ).id
        sticky_id = store.append_trace(
            "the throttle runbook that everything keeps redelivering",
            {"scope": "project:census", "procedure_id": "throttle-runbook"},
        ).id
        ignored_id = store.append_trace(
            "the quarantine ledger nobody has ever opened",
            {"scope": "project:census", "procedure_id": "quarantine-ledger"},
        ).id

    filler = [f"filler-node-{i:03d}" for i in range(RECALL_DELIVERY_HISTORY_HEAD_CUT)]

    def event(
        event_id: str,
        offset: float,
        query: str,
        medoids: Sequence[tuple[str, str]] = (),
        results: Sequence[str] = (),
        transport: str = "transport-a",
        task: str = "census-task",
    ) -> tuple[str, ...]:
        payload = None
        if medoids:
            payload = json.dumps(
                {
                    "clusters": [
                        {
                            "label": label,
                            "ask_hint": label,
                            "medoid": {"node_id": node_id},
                        }
                        for node_id, label in medoids
                    ]
                }
            )
        body = json.dumps([{"node_id": node_id} for node_id in [*filler, *results]])
        return (
            event_id,
            query,
            "project:census",
            "project:census",
            "[]",
            "{}",
            None,
            10,
            body,
            "census",
            task,
            "session-census",
            transport,
            None,
            0,
            0,
            None,
            None,
            stamp(offset),
            payload,
        )

    #: hours -> what happens.  Windows 0/1 predate the first lookup and must
    #: come back NULL; the rest are inside the observability epoch.
    rows = [
        # --- pre-epoch: unobservable no matter what the reader did ----------
        event("01CENSUSPREEPOCH0000000001", 0, "throttle runbook offer",
              medoids=[(sticky_id, "throttle runbook")], results=[sticky_id]),
        event("01CENSUSPREEPOCH0000000002", 1, "throttle runbook again",
              results=[sticky_id]),
        # --- lookup_followed ------------------------------------------------
        event("01CENSUSLOOKUP000000000001", 100, "unrelated opening question",
              medoids=[(followed_id, "canary window")], results=[followed_id]),
        # --- ask_follow: a later recall echoes the delivered label ----------
        event("01CENSUSASK000000000000001", 200, "unrelated opening question",
              medoids=[(asked_id, "checkpoint drill")], results=[asked_id]),
        event("01CENSUSASK000000000000002", 203,
              "checkpoint drill rehearsal notes", transport="transport-a"),
        # --- redelivered_only: a later recall returns it, nobody asks -------
        event("01CENSUSREDEL00000000000001", 300, "unrelated opening question",
              medoids=[(sticky_id, "throttle runbook")], results=[sticky_id]),
        event("01CENSUSREDEL00000000000002", 305, "unrelated follow-up question",
              results=[sticky_id]),
        # --- nothing ---------------------------------------------------------
        event("01CENSUSNOTHING000000000001", 400, "unrelated opening question",
              medoids=[(ignored_id, "quarantine ledger")], results=[ignored_id]),
        # --- both: lookup wins the class, contingency keeps both bits -------
        event("01CENSUSBOTH00000000000001", 500, "unrelated opening question",
              medoids=[(followed_id, "canary window")], results=[followed_id]),
        event("01CENSUSBOTH00000000000002", 505, "unrelated follow-up question",
              results=[followed_id]),
    ]

    connection = sqlite3.connect(path)
    try:
        connection.executemany(
            """
            INSERT INTO recall_events (
                id, query, scope, requested_scope, resolved_scopes,
                ambient_context, depth, max_results, results, agent, task,
                session_id, transport_session_id, fingerprint, gated,
                feedback_applied, feedback_trace_id, feedback_applied_at,
                created_at, recall_map
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        connection.commit()
    finally:
        connection.close()

    with MemoryStore(path) as store:
        # The epoch: the first recorded lookup.  Placed after the hour-1
        # window has closed, so those two windows can only be NULL.
        store.record_lookup_event(
            [followed_id],
            transport_session_id="transport-dashboard",
            occurred_at=stamp(101),
        )
        store.record_lookup_event(
            [followed_id],
            transport_session_id="transport-dashboard",
            occurred_at=stamp(506),
        )
        # Force the replay so every window is resolved by the rebuild path
        # rather than by the order this fixture happened to write in.
        store.connection.execute(
            f"UPDATE {RECALL_DELIVERY_HISTORY_STATE_TABLE} SET format_version = 0"
        )
        store.connection.commit()
    with MemoryStore(path):
        pass

    return {
        "path": str(path),
        "base": base.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "nodes": {
            "lookup_followed": followed_id,
            "ask_follow": asked_id,
            "redelivered_only": sticky_id,
            "nothing": ignored_id,
        },
        "recall_events": len(rows),
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--corpus",
        action="append",
        default=[],
        type=CorpusSpec.parse,
        metavar="PROVENANCE:LABEL=PATH",
        help="database to census; provenance is field|snapshot|synthetic",
    )
    parser.add_argument(
        "--build-synthetic",
        type=Path,
        default=None,
        help="write a labelled synthetic corpus here and census it too",
    )
    parser.add_argument(
        "--decision-at",
        default=None,
        help="maturation bound (ISO-8601 UTC); default now",
    )
    parser.add_argument("--top-sticky", type=int, default=DEFAULT_TOP_STICKY)
    parser.add_argument(
        "--pin-node",
        action="append",
        default=list(DEFAULT_PINNED_NODES),
        help="always report this node in the sticky cohort",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    decision_at = (
        datetime.now(UTC)
        if args.decision_at is None
        else (parse_timestamp(args.decision_at) or datetime.now(UTC))
    )
    specs: list[CorpusSpec] = list(args.corpus)
    if args.build_synthetic is not None:
        built = build_synthetic_corpus(args.build_synthetic)
        specs.append(
            CorpusSpec(
                label="synthetic-classifier-check",
                provenance="synthetic",
                path=Path(built["path"]),
                note=(
                    "hand-built scenario proving the classifier discriminates all "
                    "four classes and NULL; NOT field evidence and no threshold "
                    "may be drawn from it"
                ),
            )
        )
    if not specs:
        raise SystemExit("no corpus given: pass --corpus or --build-synthetic")

    report = build_census(
        specs,
        decision_at=decision_at,
        top_sticky=args.top_sticky,
        pinned=tuple(args.pin_node),
    )
    text = json.dumps(report, indent=args.indent, ensure_ascii=False, sort_keys=False)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(
            f"wrote {args.out} — "
            f"{len(report['arms'])} arm(s), "
            f"{report['field_known_windows']} known field window(s)"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
