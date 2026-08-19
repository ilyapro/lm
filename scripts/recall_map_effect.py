#!/usr/bin/env python3
"""Measure the recall-map effect gate sealed in ``artifacts/recall-map/prereg.json``.

The gate asks one question: does a delivered map item get *followed* — does a
later recall in the same session come back for it — more than a comparable item
the agent was merely shown at the bottom of a list? Two arms, one scoring
function, and a plan that was hashed before any number existed.

    PYTHONPATH=src python3 scripts/recall_map_effect.py \
        --as-of 2026-08-19T11:00:00Z \
        --out artifacts/recall-map/baseline.json

    PYTHONPATH=src python3 scripts/recall_map_effect.py \
        --verify-prereg --as-of 2026-08-19T11:00:00Z

``--as-of`` is mandatory in every mode, not optional: the live database grows
continuously and a recall event recorded after a window closed can only ever
*add* consumption to it, exactly as
:mod:`living_memory.postsession.usage_metric` documents at its lines 24-26. A
run without a pinned cutoff is not reproducible and must not be published.

READ-ONLY BY CONSTRUCTION. The database is opened only through
:func:`living_memory.replay.open_readonly` — a ``file:...?mode=ro`` URI — and
``MemoryStore`` is never instantiated anywhere in this script or its imports:
the store migrates whatever file it is pointed at, and on the live database
that would add the ``recall_events.recall_map`` column and destroy the very
pre-feature baseline this tool exists to capture. The column being *absent* is
therefore a first-class state and reads as "zero maps delivered", not a crash.

AGGREGATES ONLY. Every value that reaches the artifact passes
:func:`living_memory.postsession.usage_metric.check_privacy` (printable ASCII,
at most 200 chars, no query text, no node content, no cluster label and no
ask_hint). Internal node-id sets live under underscore-prefixed keys and are
stripped before the guard runs, following ``counterfactual._strip_internal``.

Nothing here decides anything. Every floor, window, horizon, cap, band and
verdict rule is read out of the sealed plan; the plan's own ``plan_sha256`` is
recomputed from the keys ``plan_sha256_over`` names before a single row is
read, and a mismatch aborts the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from living_memory.embeddings import tokenize  # noqa: E402
from living_memory.postsession.corpus import (  # noqa: E402
    SessionEntry,
    build_corpus,
    index_path_for,
    iter_index,
    load_session,
    read_manifest,
    recovery_stats,
    write_corpus,
)
from living_memory.postsession.judge import canonical_json  # noqa: E402
from living_memory.postsession.transcripts import Roots  # noqa: E402
from living_memory.postsession.usage_metric import (  # noqa: E402
    DEFAULT_DB_PATH,
    SAFE_STRING,
    PrivacyGuardError,
    check_privacy,
    normalize_instant,
    parse_timestamp,
    parse_window,
)
from living_memory.replay import open_readonly  # noqa: E402

#: The sealed plan. Frozen input: this tool reads it and never writes it.
DEFAULT_PREREG = REPO_ROOT / "artifacts" / "recall-map" / "prereg.json"

#: Where the captured baseline lives, for ``--verify-prereg``.
DEFAULT_BASELINE = REPO_ROOT / "artifacts" / "recall-map" / "baseline.json"

#: The corpus manifest/index this gate rebuilds for its positional arm. It must
#: never be ``artifacts/post-session/corpus.json``: that one is sealed for the
#: post-session extraction field measurement and overwriting it would break a
#: different gate. :func:`_guard_corpus_paths` enforces that, fail-closed.
DEFAULT_CORPUS_MANIFEST = Path("~/.cache/living-memory/recall-map/corpus.json")

#: The directory whose contents are sealed for the *other* gate.
SEALED_CORPUS_DIR = REPO_ROOT / "artifacts" / "post-session"

#: How this tool must be named by the plan it executes.
TOOL_NAME = "scripts/recall_map_effect.py"


class ProtocolError(SystemExit):
    """The sealed plan cannot be executed as written; name the exact field."""


# ---------------------------------------------------------------------------
# The sealed plan
# ---------------------------------------------------------------------------


def _dig(payload: Any, path: str) -> Any:
    """Fetch ``a.b.c`` out of the plan, or die naming the exact field."""

    cursor = payload
    for step in path.split("."):
        if not isinstance(cursor, dict) or step not in cursor:
            raise ProtocolError(
                f"sealed plan is not executable as written: missing protocol field "
                f"$.{path} (stopped at {step!r}); this tool refuses to reinterpret it"
            )
        cursor = cursor[step]
    return cursor


def _num(payload: Any, path: str, kind: type) -> Any:
    value = _dig(payload, path)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProtocolError(
            f"sealed plan is not executable as written: protocol field $.{path} "
            f"must be a number, got {type(value).__name__}"
        )
    return kind(value)


def _text(payload: Any, path: str) -> str:
    value = _dig(payload, path)
    if not isinstance(value, str) or not SAFE_STRING.match(value):
        raise ProtocolError(
            f"sealed plan is not executable as written: protocol field $.{path} "
            "must be a single printable-ASCII string of at most 200 chars"
        )
    return value


def recompute_plan_sha256(prereg: dict[str, Any]) -> str:
    """Re-hash exactly the keys the plan's own ``plan_sha256_over`` names.

    The construction is the one ``scripts/postsession_field_run.py``
    ``cmd_preregister`` uses: sha256 over ``judge.canonical_json`` of the
    selected keys, encoded UTF-8. ``plan_sha256_over`` names itself, so the
    choice of what is sealed is itself sealed.
    """

    keys = prereg.get("plan_sha256_over")
    if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys):
        raise ProtocolError(
            "sealed plan is not executable as written: $.plan_sha256_over must be a "
            "non-empty list of key names"
        )
    missing = [key for key in keys if key not in prereg]
    if missing:
        raise ProtocolError(
            "sealed plan is not executable as written: $.plan_sha256_over names keys "
            f"absent from the plan: {', '.join(sorted(missing))}"
        )
    body = {key: prereg[key] for key in keys}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Protocol:
    """Every knob this tool obeys, read out of the sealed plan and never chosen here."""

    plan_sha256: str
    horizon_hours: int
    min_token_overlap: float
    head_cut: int
    tokenizer: str
    endpoint: str
    stratum_rule: str
    cohort_restriction: str
    unit_of_analysis: str
    tool: str
    organic_as_of: str
    organic_window: tuple[str, str]
    organic_cap: int
    map_cap: int
    identical_fields: tuple[str, ...]
    relative_floor: float
    absolute_floor: float
    min_organic_rate: float
    min_organic_items: int
    min_map_events: int
    min_map_items: int
    min_map_transports: int
    positional_cutoff: str
    positional_band: tuple[float, float]
    positional_limit: int
    positional_min_calls: int
    positional_min_magnitude: float

    @classmethod
    def from_prereg(cls, prereg: dict[str, Any]) -> "Protocol":
        params = "consumed_rule.parameters"
        band = _dig(prereg, "positional_protocol.mid_session_band")
        if (
            not isinstance(band, list)
            or len(band) != 2
            or not all(isinstance(edge, int | float) for edge in band)
            or float(band[0]) >= float(band[1])
        ):
            raise ProtocolError(
                "sealed plan is not executable as written: protocol field "
                "$.positional_protocol.mid_session_band must be [lo, hi] with lo < hi"
            )
        window = _dig(prereg, "arms.organic.window")
        if not isinstance(window, list) or len(window) != 2:
            raise ProtocolError(
                "sealed plan is not executable as written: protocol field "
                "$.arms.organic.window must be a two-element [start, end]"
            )
        identical = _dig(prereg, "arms.protocol_fields_identical")
        if not isinstance(identical, list) or not identical:
            raise ProtocolError(
                "sealed plan is not executable as written: protocol field "
                "$.arms.protocol_fields_identical must be a non-empty list"
            )
        return cls(
            plan_sha256=str(prereg.get("plan_sha256") or ""),
            horizon_hours=_num(prereg, f"{params}.horizon_hours", int),
            min_token_overlap=_num(prereg, f"{params}.min_token_overlap", float),
            head_cut=_num(prereg, "arms.organic.head_cut", int),
            tokenizer=_text(prereg, f"{params}.tokenizer"),
            endpoint=_text(prereg, "endpoints.primary.name"),
            stratum_rule=_text(prereg, "endpoints.stratum_rule.name"),
            cohort_restriction=_text(prereg, "endpoints.cohort_restriction.rule"),
            unit_of_analysis=_text(prereg, "endpoints.unit_of_analysis"),
            tool=_text(prereg, "arms.tool"),
            organic_as_of=normalize_instant(_text(prereg, "arms.organic.as_of")),
            organic_window=parse_window(f"{window[0]}..{window[1]}"),
            organic_cap=_num(prereg, "arms.organic.items_per_event_cap", int),
            map_cap=_num(prereg, "arms.map.items_per_event_cap", int),
            identical_fields=tuple(str(name) for name in identical),
            relative_floor=_num(prereg, "verdict_rule.relative_floor", float),
            absolute_floor=_num(prereg, "verdict_rule.absolute_floor", float),
            min_organic_rate=_num(
                prereg, "verdict_rule.inconclusive_rule.min_organic_rate", float
            ),
            min_organic_items=_num(
                prereg, "verdict_rule.cohort_minimums.min_organic_items", int
            ),
            min_map_events=_num(
                prereg, "verdict_rule.cohort_minimums.min_map_arm_events", int
            ),
            min_map_items=_num(
                prereg, "verdict_rule.cohort_minimums.min_map_arm_items", int
            ),
            min_map_transports=_num(
                prereg, "verdict_rule.cohort_minimums.min_map_arm_transport_sessions", int
            ),
            positional_cutoff=normalize_instant(
                _text(prereg, "positional_protocol.cohort_pin.pre_feature_cutoff")
            ),
            positional_band=(float(band[0]), float(band[1])),
            positional_limit=_num(
                prereg, "positional_protocol.cohort_pin.session_limit", int
            ),
            positional_min_calls=_num(
                prereg, "positional_protocol.criterion.min_recall_calls_per_arm", int
            ),
            positional_min_magnitude=_num(
                prereg, "positional_protocol.criterion.minimum_magnitude_absolute", float
            ),
        )

    def identical_field_values(self, *, cap: int) -> dict[str, Any]:
        """One arm's view of ``protocol_fields_identical``, for the step-2 check.

        ``items_per_event_cap`` is stored separately per arm in the plan, so the
        comparison is a real read of two fields rather than one value echoed
        twice; the rest are single sealed values and are identical by
        construction, which is what the check is meant to confirm.
        """

        available = {
            "horizon_hours": self.horizon_hours,
            "min_token_overlap": self.min_token_overlap,
            "items_per_event_cap": cap,
            "tokenizer": self.tokenizer,
            "endpoint": self.endpoint,
            "stratum_rule": self.stratum_rule,
            "cohort_restriction": self.cohort_restriction,
            "unit_of_analysis": self.unit_of_analysis,
            "tool": self.tool,
        }
        missing = [name for name in self.identical_fields if name not in available]
        if missing:
            raise ProtocolError(
                "sealed plan is not executable as written: "
                "$.arms.protocol_fields_identical names field(s) this tool cannot "
                f"compute: {', '.join(missing)}"
            )
        return {name: available[name] for name in self.identical_fields}


def load_prereg(path: Path) -> tuple[dict[str, Any], Protocol, str]:
    """Read the sealed plan and verify its own hash before anything is measured."""

    if not path.is_file():
        raise SystemExit(f"sealed plan not found: {path}")
    try:
        prereg = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SystemExit(f"sealed plan is not valid JSON: {path}: {exc}") from exc
    if not isinstance(prereg, dict):
        raise SystemExit(f"sealed plan must be a JSON object: {path}")

    recomputed = recompute_plan_sha256(prereg)
    sealed = prereg.get("plan_sha256")
    if not isinstance(sealed, str) or recomputed != sealed:
        raise SystemExit(
            "sealed plan hash mismatch: recomputed "
            f"{recomputed} over the {len(prereg['plan_sha256_over'])} keys named in "
            f"$.plan_sha256_over, plan carries {sealed!r}; refusing to measure"
        )

    protocol = Protocol.from_prereg(prereg)
    if protocol.tool != TOOL_NAME:
        raise ProtocolError(
            f"sealed plan names $.arms.tool = {protocol.tool!r}, but this is "
            f"{TOOL_NAME}; the plan was written for a different tool"
        )
    return prereg, protocol, recomputed


# ---------------------------------------------------------------------------
# The database, read-only
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Event:
    """One ``recall_events`` row, reduced to what the metric needs.

    The query text is never kept: only its token set, which is all the sealed
    rule's overlap limb reads and the only form that cannot leak into an
    aggregates-only artifact.
    """

    id: str
    created_at: str
    transport_session_id: str | None
    scope: str | None
    task: str | None
    result_ids: tuple[str, ...]
    query_tokens: frozenset[str]
    recall_map: str | None = None


@dataclass(frozen=True, slots=True)
class Item:
    """One delivered item: a node id, plus the tokens of its label and ask_hint.

    ``tokens`` is empty for an organic tail item — it has no label at all,
    which is exactly why the primary endpoint is restricted to the node_id
    limb.
    """

    node_id: str
    tokens: frozenset[str] = frozenset()


def _has_recall_map_column(connection: sqlite3.Connection) -> bool:
    """Is the feature's column even there?

    A pre-feature database has never been opened by writable code, so
    ``storage._migrate_recall_map_column`` has never run and the column does
    not exist. That is the live database's state today and it means "zero maps
    delivered", not a failure.
    """

    return any(
        str(row[1]) == "recall_map"
        for row in connection.execute("PRAGMA table_info(recall_events)")
    )


def load_events(
    connection: sqlite3.Connection,
    *,
    start: str,
    end: str,
    with_map: bool,
) -> list[Event]:
    """Every recall event in ``[start, end]``, oldest first, read-only."""

    columns = "id, created_at, transport_session_id, scope, task, results, query"
    if with_map:
        columns += ", recall_map"
    rows = connection.execute(
        f"SELECT {columns} FROM recall_events "
        "WHERE created_at >= ? AND created_at <= ? ORDER BY created_at, id",
        (start, end),
    ).fetchall()

    events: list[Event] = []
    for row in rows:
        try:
            raw_results = json.loads(row["results"] or "[]")
        except ValueError:
            raw_results = []
        result_ids = tuple(
            str(entry["node_id"])
            for entry in raw_results
            if isinstance(entry, dict) and entry.get("node_id")
        )
        events.append(
            Event(
                id=str(row["id"]),
                created_at=str(row["created_at"]),
                transport_session_id=row["transport_session_id"] or None,
                scope=row["scope"] or None,
                task=row["task"] or None,
                result_ids=result_ids,
                query_tokens=frozenset(tokenize(str(row["query"] or ""))),
                recall_map=(row["recall_map"] if with_map else None),
            )
        )
    return events


def _shift_hours(instant: str, hours: int) -> str:
    parsed = parse_timestamp(instant)
    if parsed is None:
        raise ValueError(f"unusable stored timestamp: {instant!r}")
    return (parsed + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


class ConsumerIndex:
    """Which later events could have consumed what a given event delivered.

    The sealed rule, verbatim: a qualifying later event is a ``recall_events``
    row whose ``created_at`` is strictly greater than the delivering event's
    and at most ``horizon_hours`` later, and which shares the delivering
    event's ``transport_session_id``, **or failing that** its ``(scope, task)``
    pair with both values non-null. The fallback fires only when the transport
    limb matched nothing, which is what "failing that" means.

    Strict ``>`` on ``created_at`` is the self-exclusion: an event is never its
    own consumer and equal timestamps never qualify, so no tie-break is needed.
    """

    def __init__(self, events: list[Event], *, horizon_hours: int) -> None:
        self._horizon = horizon_hours
        self._by_transport: dict[str, list[Event]] = {}
        self._by_scope_task: dict[tuple[str, str], list[Event]] = {}
        for event in events:
            if event.transport_session_id:
                self._by_transport.setdefault(event.transport_session_id, []).append(event)
            if event.scope and event.task:
                self._by_scope_task.setdefault((event.scope, event.task), []).append(event)
        self._transport_keys = {
            key: [item.created_at for item in bucket]
            for key, bucket in self._by_transport.items()
        }
        self._scope_task_keys = {
            key: [item.created_at for item in bucket]
            for key, bucket in self._by_scope_task.items()
        }

    @staticmethod
    def _slice(
        bucket: list[Event], keys: list[str], low: str, high: str
    ) -> list[Event]:
        return bucket[bisect_right(keys, low) : bisect_right(keys, high)]

    def qualifying(self, event: Event) -> list[Event]:
        low = event.created_at
        high = _shift_hours(low, self._horizon)
        transport = event.transport_session_id
        if transport and transport in self._by_transport:
            found = self._slice(
                self._by_transport[transport], self._transport_keys[transport], low, high
            )
            if found:
                return found
        if event.scope and event.task:
            key = (event.scope, event.task)
            bucket = self._by_scope_task.get(key)
            if bucket is not None:
                return self._slice(bucket, self._scope_task_keys[key], low, high)
        return []


class AnchorStratum:
    """Re-derive the anchor stratum from a node id, read-only.

    The delivered payload carries no ``stage`` — ``MapCluster.to_dict`` emits
    label, count, medoid, ask_hint and plan_item only — so membership is
    re-derived by the rule ``recall_map._stage_anchor`` itself applies:
    every ``query_anchor_edges`` row pointing at the node, its
    ``query_anchors`` row fetched and skipped when decayed or blank-queried,
    heaviest weight winning and the lowest ``anchor_id`` breaking ties. An item
    is in the anchor stratum iff that search returns an anchor.

    The derivation needs only a node id, so it is symmetric: an organic tail
    item, which has no label at all, is stratified by exactly this code.

    ``as_of`` is offered as a *diagnostic* cut only. The sealed stratum rule
    names no cutoff, so the gating stratification reads the anchor tables as
    they stand; the cut variant is reported beside it so the difference is
    visible rather than assumed away.
    """

    def __init__(self, connection: sqlite3.Connection, *, as_of: str | None = None) -> None:
        self._connection = connection
        self._as_of = as_of
        self._anchors: dict[str, sqlite3.Row | None] = {}
        self._cache: dict[str, str | None] = {}

    def anchor_for(self, node_id: str) -> str | None:
        cached = self._cache.get(node_id, "")
        if cached != "":
            return cached  # type: ignore[return-value]
        best: tuple[float, str] | None = None
        if self._as_of is None:
            edges = self._connection.execute(
                "SELECT anchor_id, weight FROM query_anchor_edges WHERE target_id = ?",
                (node_id,),
            ).fetchall()
        else:
            edges = self._connection.execute(
                "SELECT anchor_id, weight FROM query_anchor_edges "
                "WHERE target_id = ? AND created_at <= ?",
                (node_id, self._as_of),
            ).fetchall()
        for edge in edges:
            anchor_id = str(edge["anchor_id"])
            if anchor_id not in self._anchors:
                self._anchors[anchor_id] = self._connection.execute(
                    "SELECT id, query, decayed, created_at FROM query_anchors WHERE id = ?",
                    (anchor_id,),
                ).fetchone()
            anchor = self._anchors[anchor_id]
            if anchor is None:
                continue
            if int(anchor["decayed"]) or not str(anchor["query"] or "").strip():
                continue
            if self._as_of is not None and str(anchor["created_at"]) > self._as_of:
                continue
            weight = float(edge["weight"])
            if best is None or (-weight, anchor_id) < (-best[0], best[1]):
                best = (weight, anchor_id)
        result = None if best is None else best[1]
        self._cache[node_id] = result
        return result


# ---------------------------------------------------------------------------
# The one scoring path. Both arms go through here; only item production differs.
# ---------------------------------------------------------------------------


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


@dataclass(slots=True)
class Tally:
    """Counted items and the consumed subset of them, for one stratum."""

    items: int = 0
    consumed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": self.items,
            "consumed": self.consumed,
            "rate": _ratio(self.consumed, self.items),
        }


def score_pairs(
    pairs: list[tuple[Event, Item]],
    index: ConsumerIndex,
    *,
    min_token_overlap: float,
    stratum: AnchorStratum,
    diagnostic_stratum: AnchorStratum | None = None,
) -> dict[str, Any]:
    """Apply the frozen consumed rule to (delivering event, item) pairs.

    THE shared code path: the map arm and the organic arm are both scored here,
    called twice with different item sources, so no arm can be scored by a rule
    the other was not.

    The node_id limb — an item is consumed when a qualifying later event lists
    its node id in ``json.loads(recall_events.results)[].node_id`` — is the
    primary endpoint and is symmetric across arms. The token-overlap limb —
    containment of the item's own tokens in a later query — needs a label and
    an ask_hint that only a map item has; it is tallied here and reported as
    exploratory, never as the gating rate.

    The unit is the ``(delivering_event_id, item_node_id)`` pair: a node
    delivered by ten events contributes ten items.
    """

    primary = Tally()
    anchor = Tally()
    overlap_scorable = Tally()
    either_limb = Tally()
    diagnostic_primary = Tally()
    diagnostic_anchor = Tally()
    events_scored: set[str] = set()
    transports: set[str] = set()
    consumer_cache: dict[str, tuple[frozenset[str], tuple[frozenset[str], ...]]] = {}

    for event, item in pairs:
        cached = consumer_cache.get(event.id)
        if cached is None:
            consumers = index.qualifying(event)
            delivered: set[str] = set()
            queries: list[frozenset[str]] = []
            for consumer in consumers:
                delivered.update(consumer.result_ids)
                if consumer.query_tokens:
                    queries.append(consumer.query_tokens)
            cached = (frozenset(delivered), tuple(queries))
            consumer_cache[event.id] = cached
        delivered_ids, query_token_sets = cached

        by_node = item.node_id in delivered_ids
        by_tokens = False
        if item.tokens:
            needed = min_token_overlap * len(item.tokens)
            by_tokens = any(
                len(item.tokens & tokens) >= needed for tokens in query_token_sets
            )

        events_scored.add(event.id)
        if event.transport_session_id:
            transports.add(event.transport_session_id)

        target = anchor if stratum.anchor_for(item.node_id) is not None else primary
        target.items += 1
        target.consumed += by_node

        if diagnostic_stratum is not None:
            alt = (
                diagnostic_anchor
                if diagnostic_stratum.anchor_for(item.node_id) is not None
                else diagnostic_primary
            )
            alt.items += 1
            alt.consumed += by_node

        if item.tokens:
            overlap_scorable.items += 1
            overlap_scorable.consumed += by_tokens
        either_limb.items += 1
        either_limb.consumed += by_node or by_tokens

    scored: dict[str, Any] = {
        "events_scored": len(events_scored),
        "transport_sessions": len(transports),
        "items": primary.items + anchor.items,
        "strata": {"primary": primary.to_dict(), "anchor": anchor.to_dict()},
        "primary_endpoint": primary.to_dict(),
        "secondary_anchor_stratum": anchor.to_dict(),
        "exploratory_token_overlap": (
            overlap_scorable.to_dict()
            if overlap_scorable.items
            else {
                "items": 0,
                "consumed": 0,
                "rate": None,
                "computable": False,
                "why": "items carry no label or ask_hint; the token limb is undefined here",
            }
        ),
        "diagnostic_full_consumed_rule_either_limb": either_limb.to_dict(),
    }
    if diagnostic_stratum is not None:
        scored["diagnostic_stratum_under_as_of_cut"] = {
            "primary": diagnostic_primary.to_dict(),
            "anchor": diagnostic_anchor.to_dict(),
            "why": (
                "non-gating: the sealed stratum rule names no cutoff, so this shows "
                "what an as_of-cut anchor table would have stratified instead"
            ),
        }
    return scored


# ---------------------------------------------------------------------------
# Item sources. One per arm; this is the only thing that differs between them.
# ---------------------------------------------------------------------------


def organic_items(event: Event, *, head_cut: int, cap: int) -> list[Item]:
    """The delivered tail: rank >= head_cut, first ``cap`` of them.

    Registered in the plan as a proxy, not as the residual pool: a past
    recall's residual candidates are not recoverable read-only, because
    ``recall_events.results`` holds only the delivered top-N and
    ``replay.build_candidates`` rebuilds candidates from that same row.
    """

    seen: set[str] = set()
    items: list[Item] = []
    for node_id in event.result_ids[head_cut : head_cut + cap]:
        if node_id in seen:
            continue
        seen.add(node_id)
        items.append(Item(node_id=node_id))
    return items


def map_items(event: Event, *, cap: int) -> list[Item]:
    """One item per delivered cluster, keyed by ``clusters[].medoid.node_id``.

    The payload is read straight out of ``recall_events.recall_map``:
    ``RecallEvent`` and ``_recall_event_from_row`` do not expose the column,
    and ``MemoryStore.recent_recall_map_history`` is an API on the writable
    store, which this tool must never open. Its shape is
    ``{"clusters": [{"label", "count", "medoid": {"node_id"[, "example"]},
    "ask_hint", "plan_item"}], "pool", "covered"[, "more"]}``.

    Label and ask_hint are read only to build a token set. Neither is kept and
    neither can reach the artifact.
    """

    if not event.recall_map:
        return []
    try:
        payload = json.loads(event.recall_map)
    except ValueError:
        return []
    if not isinstance(payload, dict):
        return []
    clusters = payload.get("clusters")
    if not isinstance(clusters, list):
        return []

    seen: set[str] = set()
    items: list[Item] = []
    for cluster in clusters[:cap]:
        if not isinstance(cluster, dict):
            continue
        medoid = cluster.get("medoid")
        node_id = str(medoid.get("node_id") or "") if isinstance(medoid, dict) else ""
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        tokens = frozenset(
            tokenize(str(cluster.get("label") or ""))
        ) | frozenset(tokenize(str(cluster.get("ask_hint") or "")))
        items.append(Item(node_id=node_id, tokens=tokens))
    return items


def measure_arm(
    connection: sqlite3.Connection,
    *,
    kind: str,
    gating: bool,
    as_of: str,
    window: tuple[str, str],
    protocol: Protocol,
    cap: int,
    item_source: str,
    stratum: AnchorStratum,
    diagnostic_stratum: AnchorStratum | None = None,
) -> dict[str, Any]:
    """Load one arm's cohort, produce its items, and score them shared-path.

    ``item_source`` selects the *only* step that differs between the arms.
    Everything after it — the cohort restriction to opportunity-bearing events,
    the consumed rule, the stratification, the unit of analysis — is the same
    code for all three.

    ``organic_concurrent`` is the non-gating regime-matched read: the same tail
    rule, head_cut and cap, applied to the SAME post-feature events that
    carried a map, so the map items and the tail items come from one event.
    """

    start, end = window
    map_bearing = item_source in ("map", "organic_concurrent")
    pool = load_events(connection, start=start, end=as_of, with_map=map_bearing)
    index = ConsumerIndex(pool, horizon_hours=protocol.horizon_hours)
    cohort = [
        event
        for event in pool
        if start <= event.created_at < end and (event.recall_map if map_bearing else True)
    ]

    pairs: list[tuple[Event, Item]] = []
    opportunity_bearing = 0
    for event in cohort:
        if not index.qualifying(event):
            continue
        opportunity_bearing += 1
        items = (
            map_items(event, cap=cap)
            if item_source == "map"
            else organic_items(event, head_cut=protocol.head_cut, cap=cap)
        )
        pairs.extend((event, item) for item in items)

    scored = score_pairs(
        pairs,
        index,
        min_token_overlap=protocol.min_token_overlap,
        stratum=stratum,
        diagnostic_stratum=diagnostic_stratum,
    )
    return {
        "kind": kind,
        "gating": gating,
        "measured": True,
        "item_source": item_source,
        "as_of": as_of,
        "window_start": start,
        "window_end": end,
        "head_cut": None if item_source == "map" else protocol.head_cut,
        "items_per_event_cap": cap,
        "events_in_window": len(cohort),
        "opportunity_bearing_events": opportunity_bearing,
        "observable_consumer_events": len(pool),
        **scored,
    }


# ---------------------------------------------------------------------------
# The positional endpoint
# ---------------------------------------------------------------------------


def _guard_corpus_paths(manifest: Path, index: Path) -> None:
    """Refuse to touch the corpus sealed for the post-session extraction gate."""

    for path in (manifest, index):
        try:
            path.resolve().relative_to(SEALED_CORPUS_DIR.resolve())
        except ValueError:
            continue
        raise SystemExit(
            f"refusing to write {path}: {SEALED_CORPUS_DIR} holds the corpus sealed "
            "for the post-session extraction field measurement; point "
            "--corpus-manifest / --corpus-index outside it"
        )


def ensure_corpus(
    manifest_path: Path,
    index_path: Path,
    *,
    rebuild: bool,
    log: Any,
) -> dict[str, Any]:
    """Rebuild the session index unless a usable one is already on disk.

    The tracked index is gitignored and absent, so the positional arm normally
    starts with a rebuild. A rebuild today also sweeps in the sessions of the
    tree that built this feature, which is exactly why the plan pins the cohort
    by transcript-internal ``ended_at`` and never by a corpus digest.
    """

    _guard_corpus_paths(manifest_path, index_path)
    if rebuild or not index_path.is_file() or not manifest_path.is_file():
        log(f"building session corpus -> {index_path}")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        corpus = build_corpus(Roots.discover(), index_display_path=str(index_path))
        write_corpus(corpus, manifest_path, index_path)
    manifest = read_manifest(manifest_path)
    counts = manifest.get("counts") or {}
    return {
        "manifest": _tilde(manifest_path),
        "generated_at": manifest.get("generated_at"),
        "rows": counts.get("sessions"),
        "by_split": counts.get("by_split"),
        "holdout_sha256": manifest.get("holdout_sha256"),
        "index_sha256": manifest.get("index_sha256"),
    }


def _eligible_sessions(
    index_path: Path, *, cutoff: str, side: str
) -> tuple[list[SessionEntry], dict[str, int]]:
    """Index rows in train+eval with a parseable ``ended_at`` on ``side`` of the cut.

    A row with no ``ended_at`` is excluded fail-closed: it cannot be proven
    pre-feature, and the whole point of the pin is that a rebuild run today
    must not quietly admit the sessions that built the feature.
    """

    counts = {"train_eval": 0, "no_ended_at": 0, "unparseable": 0, "eligible": 0}
    boundary = parse_timestamp(cutoff)
    eligible: list[SessionEntry] = []
    for entry in iter_index(index_path):
        if entry.split not in ("train", "eval"):
            continue
        counts["train_eval"] += 1
        if not entry.ended_at:
            counts["no_ended_at"] += 1
            continue
        ended = parse_timestamp(entry.ended_at)
        if ended is None:
            counts["unparseable"] += 1
            continue
        if (ended < boundary) if side == "pre" else (ended > boundary):
            eligible.append(entry)
    counts["eligible"] = len(eligible)
    return eligible, counts


def measure_positional(
    index_path: Path, *, cutoff: str, side: str, protocol: Protocol
) -> dict[str, Any]:
    """Mid-session share of recall calls, over the pinned session cohort.

    ``position`` is ``postsession.session.RecallInteraction.position``, filled
    by ``transcripts._finalize`` as ``record_index / max(1, record_count - 1)``
    and capped at 1.0. ``mid_session_share`` is the share of observed recall
    *calls* whose position lies inside the sealed band, as a closed interval:
    the plan's own derivation reads the lower edge as "at or past the first
    quarter" and the upper edge as keeping the closing-ritual region out.

    There is no mid-session share anywhere in the code —
    ``corpus.recovery_stats`` emits ``median_position`` and nothing else — so
    the median is taken from that aggregator and the share is computed here
    against the band the plan froze, with both denominators published beside
    it.
    """

    eligible, counts = _eligible_sessions(index_path, cutoff=cutoff, side=side)
    selected = sorted(eligible, key=lambda entry: entry.sha256)[
        : protocol.positional_limit
    ]

    records = []
    failures: dict[str, int] = {}
    for entry in selected:
        try:
            records.append(load_session(entry))
        except Exception as error:  # noqa: BLE001 - a transcript we cannot parse is a count
            name = type(error).__name__
            failures[name] = failures.get(name, 0) + 1

    stats = recovery_stats(records)
    low, high = protocol.positional_band
    positions = [call.position for record in records for call in record.recalls]
    mid = [value for value in positions if low <= value <= high]
    share = _ratio(len(mid), len(positions))
    return {
        "side": side,
        "cutoff": cutoff,
        "measured": True,
        "mid_session_band": [low, high],
        "band_bounds": "closed interval: low <= position <= high",
        "position_definition": (
            "session.RecallInteraction.position = record_index / max(1, record_count-1)"
        ),
        "aggregator": "postsession.corpus.recovery_stats",
        "session_limit": protocol.positional_limit,
        "sessions_train_eval": counts["train_eval"],
        "sessions_without_ended_at": counts["no_ended_at"],
        "sessions_unparseable_ended_at": counts["unparseable"],
        "sessions_eligible": counts["eligible"],
        "sessions_selected": len(selected),
        "sessions_parsed": len(records),
        "sessions_with_recall": sum(1 for record in records if record.recalls),
        "parse_failures": sum(failures.values()),
        "parse_failure_kinds": dict(sorted(failures.items())),
        "recall_calls": len(positions),
        "mid_session_calls": len(mid),
        "mid_session_share": share,
        "median_position": stats["median_position"],
        "write_calls": int(stats["totals"].get("writes", 0)),
        "min_recall_calls_per_arm": protocol.positional_min_calls,
        "small_n": len(positions) < protocol.positional_min_calls,
    }


# ---------------------------------------------------------------------------
# The verdict: the plan's own decision procedure, step for step
# ---------------------------------------------------------------------------


def positional_verdict(
    pre: dict[str, Any] | None, post: dict[str, Any] | None, protocol: Protocol
) -> dict[str, Any]:
    """``positional_protocol.criterion``: up, by at least the sealed magnitude."""

    out: dict[str, Any] = {
        "rule": "PASS iff post_mid_session_share - pre_mid_session_share >= magnitude",
        "direction": "up",
        "minimum_magnitude_absolute": protocol.positional_min_magnitude,
        "min_recall_calls_per_arm": protocol.positional_min_calls,
        "pre_mid_session_share": (pre or {}).get("mid_session_share"),
        "post_mid_session_share": (post or {}).get("mid_session_share"),
        "pre_recall_calls": (pre or {}).get("recall_calls"),
        "post_recall_calls": (post or {}).get("recall_calls"),
    }
    if pre is None or post is None:
        out["verdict"] = "INCONCLUSIVE"
        out["reason"] = "only one arm measured; the post arm needs a deployed map"
        return out
    if pre["small_n"] or post["small_n"]:
        out["verdict"] = "INCONCLUSIVE"
        out["reason"] = "small_n_rule: an arm observed fewer recall calls than the minimum"
        return out
    if (pre["mid_session_share"] or 0.0) > 0.90:
        out["verdict"] = "INCONCLUSIVE"
        out["reason"] = "unfalsifiable_guard: pre share above 0.90; pending new pre-registration"
        return out
    delta = round(post["mid_session_share"] - pre["mid_session_share"], 4)
    out["delta"] = delta
    out["verdict"] = (
        "PASS" if delta >= protocol.positional_min_magnitude else "FAIL"
    )
    return out


def build_verdict(
    organic: dict[str, Any],
    map_arm: dict[str, Any],
    positional: dict[str, Any],
    protocol: Protocol,
    identical_check: dict[str, Any],
) -> dict[str, Any]:
    """Steps 1-8 of ``verdict_rule.decision_procedure``, in order, fail-closed."""

    organic_primary = organic["primary_endpoint"]
    organic_rate = organic_primary["rate"]
    map_primary = map_arm.get("primary_endpoint") or {"items": 0, "consumed": 0, "rate": None}
    map_rate = map_primary["rate"]

    consumption: dict[str, Any] = {
        "organic_rate": organic_rate,
        "organic_items": organic_primary["items"],
        "map_rate": map_rate,
        "map_items": map_primary["items"],
        "relative_floor": protocol.relative_floor,
        "relative_floor_required": (
            round(protocol.relative_floor * organic_rate, 4)
            if organic_rate is not None
            else None
        ),
        "absolute_floor": protocol.absolute_floor,
        "cohort_minimums": {
            "min_organic_items": protocol.min_organic_items,
            "min_map_arm_events": protocol.min_map_events,
            "min_map_arm_items": protocol.min_map_items,
            "min_map_arm_transport_sessions": protocol.min_map_transports,
            "measured_organic_items": organic_primary["items"],
            "measured_map_events": map_arm.get("events_scored", 0),
            "measured_map_items": map_primary["items"],
            "measured_map_transport_sessions": map_arm.get("transport_sessions", 0),
        },
    }

    # Step 2: the comparison is void whatever the two rates are.
    if not identical_check["identical"]:
        consumption["verdict"] = "INCONCLUSIVE"
        consumption["reason"] = "protocol_fields_identical differ between the arms"
    # Step 3: INCONCLUSIVE judges the reference, never the map.
    elif organic_primary["items"] < protocol.min_organic_items:
        consumption["verdict"] = "INCONCLUSIVE"
        consumption["reason"] = "organic primary-stratum items below min_organic_items"
    elif organic_rate is None:
        consumption["verdict"] = "INCONCLUSIVE"
        consumption["reason"] = "organic arm produced no scorable item"
    elif organic_rate <= protocol.absolute_floor:
        consumption["verdict"] = "INCONCLUSIVE"
        consumption["reason"] = (
            "falsifiability_invariant: organic rate at or below the absolute floor; "
            "pending a NEW pre-registration, never FAIL"
        )
    elif organic_rate < protocol.min_organic_rate:
        consumption["verdict"] = "INCONCLUSIVE"
        consumption["reason"] = "organic rate below min_organic_rate; the reference cannot discriminate"
    # Step 4: a two-item map arm scoring 1.0 must not PASS.
    elif not map_arm.get("measured"):
        consumption["verdict"] = "INCONCLUSIVE"
        consumption["reason"] = "map arm not measured: no map has been delivered yet"
    elif (
        map_arm.get("events_scored", 0) < protocol.min_map_events
        or map_primary["items"] < protocol.min_map_items
        or map_arm.get("transport_sessions", 0) < protocol.min_map_transports
    ):
        consumption["verdict"] = "INCONCLUSIVE"
        consumption["reason"] = "map arm below a cohort minimum on the primary endpoint"
    else:
        # Step 5.
        clears_relative = map_rate >= protocol.relative_floor * organic_rate
        clears_absolute = map_rate >= protocol.absolute_floor
        consumption["clears_relative_floor"] = clears_relative
        consumption["clears_absolute_floor"] = clears_absolute
        consumption["verdict"] = "PASS" if clears_relative and clears_absolute else "FAIL"

    overall = (
        "PASS"
        if consumption["verdict"] == "PASS" and positional["verdict"] == "PASS"
        else ("FAIL" if "INCONCLUSIVE" not in (consumption["verdict"], positional["verdict"]) else "INCONCLUSIVE")
    )
    return {
        "bar_source": "artifacts/recall-map/prereg.json#verdict_rule",
        "consumption": consumption,
        "positional": positional,
        "overall": overall,
        "never_pass_on_inconclusive": True,
    }


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _strip_internal(payload: Any) -> None:
    """Drop internal node-id / text sets before the privacy guard runs."""

    if isinstance(payload, dict):
        for key in [key for key in payload if isinstance(key, str) and key.startswith("_")]:
            payload.pop(key)
        for value in payload.values():
            _strip_internal(value)
    elif isinstance(payload, list):
        for value in payload:
            _strip_internal(value)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tilde(path: Path | str) -> str:
    """Collapse the home directory, the way the plan writes its own paths."""

    text = str(path)
    home = str(Path.home())
    return f"~{text[len(home):]}" if text.startswith(home) else text


#: Quantities the sealing node measured, and where this tool computes the same
#: thing. Reproducing them is a cross-check between two independent readings of
#: one frozen prose protocol: the node that sealed the plan and the node that
#: executes it. A drift here means the rule was read two different ways, which
#: is exactly what a pre-registration exists to make visible.
SEALING_CHECKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cohort_events_in_window", ("events_in_window",)),
    ("observable_consumer_events_to_as_of", ("observable_consumer_events",)),
    ("opportunity_bearing_events_horizon_24h", ("opportunity_bearing_events",)),
    ("opportunity_bearing_events_tail_bearing", ("events_scored",)),
    ("organic_items_opportunity_bearing", ("items",)),
    ("organic_items_primary_stratum", ("strata", "primary", "items")),
    ("organic_items_anchor_stratum", ("strata", "anchor", "items")),
)


def check_sealing_denominators(
    prereg: dict[str, Any], organic: dict[str, Any]
) -> dict[str, Any]:
    """Re-measure the counts the sealing node published, and say so either way."""

    sealed = ((prereg.get("sealing_denominators") or {}).get("database") or {})
    compared: dict[str, Any] = {}
    drifted: list[str] = []
    for name, path in SEALING_CHECKS:
        if name not in sealed:
            continue
        measured: Any = organic
        for step in path:
            measured = (measured or {}).get(step)
        compared[name] = {"sealed": sealed[name], "measured": measured}
        if sealed[name] != measured:
            drifted.append(name)
    return {
        "why": (
            "the sealing node and this tool read one frozen protocol independently; "
            "equal counts mean they read it the same way"
        ),
        "compared": compared,
        "reproduced": not drifted,
        "drifted": sorted(drifted),
        "note": "informational, never gating: the verdict rule names no such check",
    }


def _notes(
    organic: dict[str, Any],
    map_arm: dict[str, Any],
    positional_pre: dict[str, Any] | None,
    protocol: Protocol,
) -> list[str]:
    """What a reader of this artifact has to know that the numbers do not say."""

    notes = [
        "Aggregates only: no node id, no query text, no cluster label and no ask_hint "
        "reaches this artifact; every value passes usage_metric.check_privacy.",
        "The database was opened read-only through replay.open_readonly; MemoryStore was "
        "never instantiated, so the recall_map column was not migrated into existence.",
        "The sealed stratum rule names no cutoff, so the gating stratification reads the "
        "anchor tables as they stand; the as_of-cut variant sits beside it as a diagnostic.",
    ]
    if not map_arm.get("measured"):
        notes.append(
            "Map arm NOT YET MEASURED, by design: it is recorded rather than omitted so a "
            "later field run fills it in under this same plan_sha256."
        )
    rate = organic["primary_endpoint"]["rate"]
    if rate is not None:
        notes.append(
            f"Organic primary rate {rate} is admissible (>= min_organic_rate "
            f"{protocol.min_organic_rate} and above absolute_floor "
            f"{protocol.absolute_floor}), so the relative floor "
            f"{round(protocol.relative_floor * rate, 4)} is the binding one."
        )
    if positional_pre and positional_pre.get("mid_session_share") is not None:
        share = positional_pre["mid_session_share"]
        notes.append(
            f"Pre-feature mid_session_share {share} over "
            f"{positional_pre['recall_calls']} calls in "
            f"{positional_pre['sessions_parsed']} sessions is NOT the published '1 of 57'."
        )
        notes.append(
            "The plan registers that figure as a prior only: the threshold behind the "
            "word mid-session is recorded nowhere in the code, so it is not comparable "
            "to a share under this band."
        )
        notes.append(
            f"The positional criterion stays falsifiable: a pre share of {share} is below "
            "0.90, so the sealed rise of 0.1 is arithmetically reachable."
        )
    return notes


def feature_deploy_instant(connection: sqlite3.Connection, *, as_of: str) -> str | None:
    """``created_at`` of the earliest event that ever carried a map, or None.

    ``no such column: recall_map`` is not an error here: on a database that
    pre-feature code alone has ever opened, ``storage._migrate_recall_map_column``
    has never run, so the column does not exist and the honest answer is "zero
    maps delivered".
    """

    if not _has_recall_map_column(connection):
        return None
    try:
        row = connection.execute(
            "SELECT min(created_at) FROM recall_events "
            "WHERE recall_map IS NOT NULL AND created_at <= ?",
            (as_of,),
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such column" in str(exc):
            return None
        raise
    return str(row[0]) if row and row[0] else None


def run_measurement(
    db_path: Path,
    *,
    as_of: str,
    prereg: dict[str, Any],
    protocol: Protocol,
    recomputed_sha: str,
    index_path: Path | None,
    corpus_info: dict[str, Any] | None,
    log: Any,
) -> dict[str, Any]:
    """Both gating arms, the non-gating control, the positional endpoint, the verdict."""

    connection = open_readonly(db_path)
    try:
        stratum = AnchorStratum(connection)
        stratum_cut = AnchorStratum(connection, as_of=as_of)

        log(f"organic arm: {protocol.organic_window[0]}..{protocol.organic_window[1]}")
        organic = measure_arm(
            connection,
            kind="organic",
            gating=True,
            as_of=protocol.organic_as_of,
            window=protocol.organic_window,
            protocol=protocol,
            cap=protocol.organic_cap,
            item_source="organic",
            stratum=stratum,
            diagnostic_stratum=stratum_cut,
        )

        deploy = feature_deploy_instant(connection, as_of=as_of)
        map_window_end = _shift_hours(as_of, -protocol.horizon_hours)
        if deploy is None or deploy >= map_window_end:
            reason = (
                "recall_events.recall_map is absent on this database: no writable open "
                "has ever migrated it, so no map has ever been delivered"
                if deploy is None
                else "no map-bearing event lies before the window end (as_of - horizon_hours)"
            )
            log(f"map arm: not measured ({reason})")
            map_arm: dict[str, Any] = {
                "kind": "map",
                "gating": True,
                "measured": False,
                "item_source": "map",
                "reason": reason,
                "as_of": as_of,
                "window_start": deploy,
                "window_end": map_window_end,
                "items_per_event_cap": protocol.map_cap,
                "events_in_window": 0,
                "events_scored": 0,
                "transport_sessions": 0,
                "items": 0,
                "primary_endpoint": {"items": 0, "consumed": 0, "rate": None},
                "secondary_anchor_stratum": {"items": 0, "consumed": 0, "rate": None},
                "recall_map_column_present": _has_recall_map_column(connection),
            }
            concurrent: dict[str, Any] = {
                "kind": "organic",
                "gating": False,
                "measured": False,
                "item_source": "organic_concurrent",
                "reason": reason,
            }
        else:
            log(f"map arm: {deploy}..{map_window_end}")
            map_arm = measure_arm(
                connection,
                kind="map",
                gating=True,
                as_of=as_of,
                window=(deploy, map_window_end),
                protocol=protocol,
                cap=protocol.map_cap,
                item_source="map",
                stratum=stratum,
                diagnostic_stratum=stratum_cut,
            )
            map_arm["recall_map_column_present"] = True
            concurrent = measure_arm(
                connection,
                kind="organic",
                gating=False,
                as_of=as_of,
                window=(deploy, map_window_end),
                protocol=protocol,
                cap=protocol.organic_cap,
                item_source="organic_concurrent",
                stratum=stratum,
            )
    finally:
        connection.close()

    organic_fields = protocol.identical_field_values(cap=protocol.organic_cap)
    map_fields = protocol.identical_field_values(cap=protocol.map_cap)
    mismatched = sorted(
        name for name in organic_fields if organic_fields[name] != map_fields[name]
    )
    identical_check = {
        "fields": list(protocol.identical_fields),
        "organic": organic_fields,
        "map": map_fields,
        "identical": not mismatched,
        "mismatched": mismatched,
        "on_mismatch": "INCONCLUSIVE; the comparison is void whatever the two rates are",
    }

    positional_pre: dict[str, Any] | None = None
    positional_post: dict[str, Any] | None = None
    if index_path is not None:
        log("positional arm: pre-feature cohort")
        positional_pre = measure_positional(
            index_path, cutoff=protocol.positional_cutoff, side="pre", protocol=protocol
        )
        if map_arm.get("measured") and map_arm.get("window_start"):
            log("positional arm: post-feature cohort")
            positional_post = measure_positional(
                index_path,
                cutoff=normalize_instant(str(map_arm["window_start"])),
                side="post",
                protocol=protocol,
            )

    positional = {
        "protocol": "artifacts/recall-map/prereg.json#positional_protocol",
        "joint_endpoint": (
            "measures the mid-work recall law and the map together, not the map alone"
        ),
        "published_reference": "median position 0.02; 1 of 57 mid-session; 2026-08-19",
        "published_reference_status": (
            "the prior, NOT the pre value: the pre arm is re-measured under this protocol"
        ),
        "corpus": corpus_info,
        "pre": positional_pre or {"measured": False, "reason": "positional arm not run"},
        "post": positional_post
        or {"measured": False, "reason": "no map has been deployed; nothing to measure"},
        "verdict_inputs": None,
    }
    positional["verdict_inputs"] = positional_verdict(
        positional_pre, positional_post, protocol
    )

    report = {
        "artifact": (
            "recall-map-effect-field" if map_arm.get("measured") else "recall-map-effect-baseline"
        ),
        "generated_at": _utc_now(),
        "as_of": as_of,
        "db": _tilde(db_path),
        "read_only": "living_memory.replay.open_readonly (file:...?mode=ro); MemoryStore never instantiated",
        "plan_sha256": protocol.plan_sha256,
        "preregistration": {
            "path": "artifacts/recall-map/prereg.json",
            "plan_sha256": protocol.plan_sha256,
            "plan_sha256_recomputed": recomputed_sha,
            "plan_sha256_match": recomputed_sha == protocol.plan_sha256,
            "plan_sha256_over": list(prereg.get("plan_sha256_over") or []),
            "registered_at": prereg.get("registered_at"),
            "bar_source": prereg.get("bar_source"),
        },
        "protocol": {
            "as_of": as_of,
            "as_of_is_arm_specific": (
                "the run's cutoff, and the map arm's; the organic arm's is pinned in the "
                f"plan at {protocol.organic_as_of} and does not move with this flag"
            ),
            "horizon_hours": protocol.horizon_hours,
            "min_token_overlap": protocol.min_token_overlap,
            "head_cut": protocol.head_cut,
            "items_per_event_cap": {
                "organic": protocol.organic_cap,
                "map": protocol.map_cap,
            },
            "tokenizer": protocol.tokenizer,
            "endpoint": protocol.endpoint,
            "stratum_rule": protocol.stratum_rule,
            "cohort_restriction": protocol.cohort_restriction,
            "unit_of_analysis": protocol.unit_of_analysis,
            "tool": protocol.tool,
            "consumed_limb_gating": "node_id only; the token limb is exploratory",
            "anchor_circularity": "anchor stratum EXCLUDED from the primary endpoint, reported separately",
        },
        "protocol_fields_identical_check": identical_check,
        "arms": {
            "organic": organic,
            "map": map_arm,
            "organic_concurrent": concurrent,
        },
        "positional": positional,
        "verdict": build_verdict(
            organic, map_arm, positional["verdict_inputs"], protocol, identical_check
        ),
        "sealing_denominator_check": check_sealing_denominators(prereg, organic),
        "notes": _notes(organic, map_arm, positional_pre, protocol),
    }
    _strip_internal(report)
    check_privacy(report)
    return report


# ---------------------------------------------------------------------------
# --verify-prereg: fail closed, never warn and continue
# ---------------------------------------------------------------------------


def verify_prereg(
    *,
    prereg_path: Path,
    baseline_path: Path,
    as_of: str,
) -> int:
    """Recompute the seal, then check the captured baseline against it.

    Every check is fatal. A gate that warns and continues is not a gate: the
    only outcomes here are "everything matches" and a nonzero exit naming what
    did not.
    """

    problems: list[str] = []
    checks: list[tuple[str, bool, str]] = []

    if not prereg_path.is_file():
        raise SystemExit(f"sealed plan not found: {prereg_path}")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    recomputed = recompute_plan_sha256(prereg)
    sealed = prereg.get("plan_sha256")
    ok = isinstance(sealed, str) and recomputed == sealed
    checks.append(
        (
            "plan_sha256 reproduces over $.plan_sha256_over",
            ok,
            f"recomputed {recomputed} vs sealed {sealed}",
        )
    )
    if not ok:
        problems.append("plan_sha256 does not reproduce")

    # The plan must also still be executable, not merely well hashed.
    try:
        checks.append(
            (
                "sealed plan is executable as written",
                True,
                Protocol.from_prereg(prereg).endpoint,
            )
        )
    except SystemExit as exc:
        checks.append(("sealed plan is executable as written", False, str(exc)))
        problems.append("sealed plan is not executable as written")

    exists = baseline_path.is_file() and baseline_path.stat().st_size > 0
    checks.append(("baseline artifact exists and is non-empty", exists, str(baseline_path)))
    if not exists:
        problems.append("baseline artifact missing or empty")
        _report_checks(checks, problems)
        return 1

    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        checks.append(("baseline artifact is valid JSON", False, str(exc)))
        problems.append("baseline artifact is not valid JSON")
        _report_checks(checks, problems)
        return 1
    checks.append(("baseline artifact is valid JSON", True, ""))

    try:
        check_privacy(baseline)
        checks.append(("baseline passes usage_metric.check_privacy", True, "aggregates only"))
    except PrivacyGuardError as exc:
        checks.append(("baseline passes usage_metric.check_privacy", False, str(exc)))
        problems.append("baseline violates the privacy guard")

    baseline_as_of = baseline.get("as_of")
    has_as_of = isinstance(baseline_as_of, str) and bool(baseline_as_of.strip())
    checks.append(("baseline carries an as_of", has_as_of, str(baseline_as_of)))
    if not has_as_of:
        problems.append("baseline carries no as_of")
    else:
        matches = normalize_instant(baseline_as_of) == as_of
        checks.append(
            ("baseline as_of equals --as-of", matches, f"{baseline_as_of} vs {as_of}")
        )
        if not matches:
            problems.append("baseline as_of differs from --as-of")

    baseline_sha = baseline.get("plan_sha256")
    byte_equal = isinstance(baseline_sha, str) and baseline_sha == sealed
    checks.append(
        (
            "baseline plan_sha256 is byte-equal to the sealed plan's",
            byte_equal,
            f"{baseline_sha} vs {sealed}",
        )
    )
    if not byte_equal:
        problems.append("baseline plan_sha256 differs from the sealed plan's")

    organic = ((baseline.get("arms") or {}).get("organic") or {})
    measured = bool(organic.get("measured")) and organic.get("primary_endpoint", {}).get(
        "rate"
    ) is not None
    checks.append(
        (
            "baseline carries a measured organic arm",
            measured,
            f"rate={organic.get('primary_endpoint', {}).get('rate')}",
        )
    )
    if not measured:
        problems.append("baseline carries no measured organic arm")

    _report_checks(checks, problems)
    return 1 if problems else 0


def _report_checks(checks: list[tuple[str, bool, str]], problems: list[str]) -> None:
    for label, ok, detail in checks:
        print(
            f"[{'ok' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""),
            flush=True,
        )
    if problems:
        print("verify-prereg FAILED: " + "; ".join(problems), file=sys.stderr)
    else:
        print("verify-prereg OK")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--as-of",
        help="ISO instant; cuts BOTH nodes and recall_events on created_at. MANDATORY "
        "in every mode: without it a run is not reproducible and must not be published",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH.expanduser()),
        help="database to read (opened read-only through replay.open_readonly; never written)",
    )
    parser.add_argument(
        "--prereg",
        default=str(DEFAULT_PREREG),
        help="the sealed pre-registration; frozen input, never written (default: %(default)s)",
    )
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="the captured baseline artifact, for --verify-prereg (default: %(default)s)",
    )
    parser.add_argument(
        "--verify-prereg",
        action="store_true",
        help="recompute plan_sha256 and check the baseline artifact; nonzero on ANY mismatch",
    )
    parser.add_argument("--out", metavar="PATH", help="write the JSON report here (default: stdout)")
    parser.add_argument(
        "--no-positional",
        action="store_true",
        help="skip the positional endpoint (the database arms only)",
    )
    parser.add_argument(
        "--corpus-manifest",
        default=str(DEFAULT_CORPUS_MANIFEST),
        help="session-corpus manifest for the positional arm; must be OUTSIDE "
        "artifacts/post-session, which is sealed for another gate (default: %(default)s)",
    )
    parser.add_argument(
        "--corpus-index",
        default=None,
        help="session index (default: the manifest's *-index.jsonl sibling)",
    )
    parser.add_argument(
        "--rebuild-corpus",
        action="store_true",
        help="rebuild the corpus index even when one is already on disk",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress on stderr")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.as_of or not str(args.as_of).strip():
        raise SystemExit(
            "--as-of is mandatory: the live database grows continuously and a recall "
            "event recorded after a window closed can only ADD consumption to it, so a "
            "run without a pinned cutoff is not reproducible and must not be published"
        )
    try:
        as_of = normalize_instant(args.as_of)
    except ValueError as exc:
        raise SystemExit(f"--as-of: {exc}") from exc

    prereg_path = Path(args.prereg).expanduser()
    baseline_path = Path(args.baseline).expanduser()

    if args.verify_prereg:
        return verify_prereg(
            prereg_path=prereg_path, baseline_path=baseline_path, as_of=as_of
        )

    def log(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr, flush=True)

    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        raise SystemExit(f"database not found: {db_path}")

    prereg, protocol, recomputed = load_prereg(prereg_path)
    if as_of < protocol.organic_as_of:
        raise SystemExit(
            f"--as-of {as_of} precedes the pinned organic arm's as_of "
            f"{protocol.organic_as_of}: the sealed organic window cannot be honoured "
            "under an earlier cutoff"
        )

    corpus_info: dict[str, Any] | None = None
    index_path: Path | None = None
    if not args.no_positional:
        manifest_path = Path(args.corpus_manifest).expanduser()
        index_path = (
            Path(args.corpus_index).expanduser()
            if args.corpus_index
            else index_path_for(manifest_path)
        )
        corpus_info = ensure_corpus(
            manifest_path, index_path, rebuild=args.rebuild_corpus, log=log
        )

    try:
        report = run_measurement(
            db_path,
            as_of=as_of,
            prereg=prereg,
            protocol=protocol,
            recomputed_sha=recomputed,
            index_path=index_path,
            corpus_info=corpus_info,
            log=log,
        )
    except PrivacyGuardError as exc:
        raise SystemExit(str(exc)) from exc
    except (ValueError, OSError, sqlite3.Error) as exc:
        raise SystemExit(f"measurement failed: {exc}") from exc

    # Second, independent pass: the guard runs where the report is built and
    # again here, before a single byte reaches disk.
    check_privacy(report)
    payload = json.dumps(report, indent=2, sort_keys=True)

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(payload)

    for name in ("organic", "map", "organic_concurrent"):
        arm = report["arms"][name]
        if not arm.get("measured"):
            print(f"{name:20s} NOT MEASURED ({arm.get('reason')})")
            continue
        primary = arm["primary_endpoint"]
        anchor = arm["secondary_anchor_stratum"]
        print(
            "%-20s events=%s items=%s | primary %s/%s rate=%s | anchor %s/%s rate=%s"
            % (
                name,
                arm["events_scored"],
                arm["items"],
                primary["consumed"],
                primary["items"],
                primary["rate"],
                anchor["consumed"],
                anchor["items"],
                anchor["rate"],
            )
        )
    pre = report["positional"]["pre"]
    if pre.get("measured"):
        print(
            "positional pre        mid_session_share=%s (%s of %s calls over %s sessions, %s with recall) median=%s"
            % (
                pre["mid_session_share"],
                pre["mid_session_calls"],
                pre["recall_calls"],
                pre["sessions_parsed"],
                pre["sessions_with_recall"],
                pre["median_position"]["recall"],
            )
        )
    verdict = report["verdict"]
    print(
        "verdict: consumption=%s positional=%s overall=%s (%s)"
        % (
            verdict["consumption"]["verdict"],
            verdict["positional"]["verdict"],
            verdict["overall"],
            verdict["consumption"].get("reason") or "floors applied",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
