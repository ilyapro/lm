#!/usr/bin/env python3
"""Rebuild the cold-start candidate population and publish its three splits.

The sealed evaluator (``scripts/recall_map_relevance_eval.py``) answers a
different question — it fits and transfers a model over an *observed map*
cohort — and it hardcodes an ``as_of`` pinned to a source snapshot that no
longer exists on this host.  This extractor answers the question the cold-start
recalibration actually needs: of the residual candidates the pool sees, how
many carry **no matured delivery history at all** at the instant the selector
had to decide, and are those cold rows consumed at a rate that differs from the
warm ones?

Three properties make this honest rather than a re-run with a new number:

* Nothing here re-derives the sealed evaluator's evidence.  The sealed
  artifacts are read for their protocol constants and their fail-closed
  helpers, and the cold-start result is published beside them under its own
  ``coldstart`` key.  The sealed cohorts are not recomputed, not corrected and
  not touched.
* The population, the maturation rule and the consumption rule are the ones
  the *live* server uses, not a reconstruction of them.  ``organic_items`` and
  ``ConsumerIndex`` come from the frozen effect tool, byte-checked before
  import; the outcome bits come from ``recall_delivery_history``, which is the
  same ledger ``MemoryStore.matured_recall_history`` reads to build the M/C/K
  features that ``recall_map.relevance_score`` consumes.
* Every window in scope has matured.  ``COLDSTART_AS_OF`` sits at least
  ``horizon_hours`` behind the snapshot's newest recall event, and the
  candidate window closes a further ``horizon_hours`` before that, so a
  candidate delivered at the very end of the window still had its full
  24-hour outcome window observed inside the snapshot.

The population is the whole persisted history, not a three-week slice.  The
first published split carried 373 cold candidates in its holdout against a
pre-registered minimum of 400; the bar does not move, so the data did.  Only
one of the two obvious levers turned out to exist.  Widening the window
backwards works.  Raising the per-event rank cap does not: ``rank_coverage_census``
measures, every run, that ``recall_delivery_history`` records an outcome for
0-based ranks 3..8 and for no other rank, so a deeper row carries no
consumption label and cannot enter a supervised dataset at all.  Both the
sealed ``head_cut`` and the sealed ``organic_cap`` therefore stay exactly where
they were.

Enlarging is the dangerous half, and the danger is silent.  Neither part of the
split is stable under a change of population: ``component_id`` is a digest over
the component's entire member set, so one new bridging event renames it, and
``_balanced_assignment`` is greedy against item targets computed over the whole
population, so every placement moves when the population does.  A naive re-run
over a wider window scatters components already read as train or eval into the
new holdout with every published count still looking healthy — and an unread
holdout is the only asset this work has.  So the v1 population is rebuilt first
and checked against pinned digests, each v2 component inherits the most-read v1
role among its events (train > eval > holdout), only wholly-new and pure
v1-holdout components are freshly assignable, and the emptiness of
``v2_holdout ∩ (v1_train ∪ v1_eval)`` is asserted, enforced and published.

Because widening backwards crosses corpus regimes, every per-split number is
published again cut by era and by rank band, and the enlargement's cost —
months of work collapsing into a few very large components under the unchanged
identity rule — is published beside its benefit rather than left to be
discovered.

Only aggregates and opaque component digests are written.  Identity values,
corpus text, and the candidate holdout subtree are unreachable from here: the
fail-closed path guard and the sensitive-key guard are *imported* from the
sealed evaluator rather than restated, so they cannot drift from it.
"""

from __future__ import annotations

import argparse
import bisect
import importlib.util
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "recall-map" / "relevance" / "dataset-manifest.json"
DEFAULT_ANALYSIS = REPO_ROOT / "artifacts" / "recall-map" / "relevance" / "feature-analysis.json"
SEALED_EVALUATOR = REPO_ROOT / "scripts" / "recall_map_relevance_eval.py"

# The sealed evaluator's own bytes, as bound by
# ``artifacts/recall-map/relevance/policy.json#bindings.historical_evaluator``.
# Checking it before import is what lets this script *reuse* the sealed
# fail-closed guards instead of paraphrasing them.
EXPECTED_EVALUATOR_SHA256 = "dfa25689cfa71bf79de34e8943b7b6a8c68e11f5b148cf04c52e16a2e354cb41"

# Pinned strictly behind the snapshot instant so no window in scope is still
# open.  ``run`` refuses to proceed unless the snapshot actually reaches
# COLDSTART_AS_OF + horizon_hours, which is the property this constant claims.
COLDSTART_AS_OF = "2026-08-23T00:00:00Z"

# The window this artifact first published: the sealed organic start, closing
# ``horizon_hours`` before the as-of.  Kept verbatim, because the carry-over
# rule below has to reconstruct that population exactly before it may enlarge
# it.  Nothing else in the extractor reads it.
V1_CANDIDATE_WINDOW_START = "2026-08-01T00:00:00Z"

# The enlarged window.  This instant PREDATES the first recall event that has
# ever been persisted (2026-05-14T20:05:23Z), and ``build_population`` refuses
# to run unless that is still true of the snapshot in front of it.  So this is
# not a chosen cut with a defensible-sounding date on it: it is the statement
# that no era is excluded, checked rather than asserted.  The window still
# closes ``horizon_hours`` before the as-of, which is what makes every
# candidate's outcome window observable.
CANDIDATE_WINDOW_START = "2026-05-14T00:00:00Z"

# The per-event cap stays at the sealed ``organic_cap`` — and NOT because the
# constant is sacred.  Lifting it was the obvious second lever and it is dead:
# ``recall_delivery_history`` records an outcome for 0-based ranks 3..8 and for
# nothing else.  ``rank_coverage_census`` measures that on the snapshot every
# run, and it is published in the artifact, so the claim is falsifiable rather
# than inherited.  A row at rank >= 9 therefore carries no consumption label at
# all and cannot enter a supervised dataset; widening the cap would add
# unlabelable rows, not evidence.  ``head_cut`` is likewise untouched: ranks
# 0..2 are the delivered head rather than residual candidates, and the ledger
# records nothing there either.  The whole enlargement is the window.
RANK_COVERAGE_CENSUS_DEPTH = 16

SPLIT_SEED = "recall-map-coldstart-components-v1"
TRAIN_FRACTION = 0.60
EVAL_FRACTION = 0.20
SPLIT_NAMES = ("train", "eval", "holdout")

# Most-read first.  A v2 component inherits the most-read role among the v1
# roles of the events it carries, so a component that was ever read as train
# can never fall back to eval or holdout, and one that was ever read as eval
# can never fall back to holdout.  Only a component with no v1 role at all is
# freshly assignable, which is what keeps the sealed partition sealed.
ROLE_PRECEDENCE = ("train", "eval", "holdout")

# The v1 partition, pinned.  ``membership_digest`` is the sha256 of the
# canonical JSON of the split's sorted opaque component digest list, exactly as
# ``artifacts/recall-map/relevance/dataset-manifest.json#coldstart.splits``
# published it at commit 1a964c6.  The carry-over is only sound if the v1 map
# it carries over IS the v1 map that was actually read, so the recomputation is
# checked against these constants and the run fails closed on any drift.
V1_SPLITS = {
    "train": {
        "items": 5664,
        "cold_candidates": 1233,
        "components": 195,
        "effective_components": 12.05,
        "membership_digest": (
            "9c0dcadea79e0caf1d607302f3e86c4e4a88e4de18350ac856f557f6631dbabd"
        ),
    },
    "eval": {
        "items": 1888,
        "cold_candidates": 423,
        "components": 160,
        "effective_components": 115.96,
        "membership_digest": (
            "f26f9aabe6981fb687128f6b9b8dbee5376dc3cf595e6e7ae82af82776e46bb4"
        ),
    },
    "holdout": {
        "items": 1888,
        "cold_candidates": 373,
        "components": 161,
        "effective_components": 116.16,
        "membership_digest": (
            "c528c91e3fd99485f187f9e1c7b7aac4dd93054ea865ab353b7a4c47866a6896"
        ),
    },
}

# ``coldstart-prereg.json#split_protocol.minimum_cold_candidates_per_split``.
# The bar does not move; the population does.  Mirrored here so the extractor
# fails closed on it rather than leaving the fit to discover the shortfall.
MINIMUM_COLD_CANDIDATES_PER_SPLIT = 400

# Era strata.  ``preexisting`` is precisely the v1 window; ``enlargement`` is
# everything the widening added.  Reported for every per-split number so a
# later reader can see what the enlargement mixed in instead of trusting that
# it was harmless.
ERA_PREEXISTING = "preexisting_2026_08"
ERA_ENLARGEMENT = "enlargement_pre_2026_08"
ERA_NAMES = (ERA_PREEXISTING, ERA_ENLARGEMENT)

# Rank bands.  These tile the whole labelable depth (ranks 3..8); the trailing
# band is open-ended only so a future ledger that recorded deeper would still be
# reported rather than silently folded into rank_7_8.
RANK_BANDS = (
    ("rank_3_4", 3, 4),
    ("rank_5_6", 5, 6),
    ("rank_7_8", 7, 8),
    ("rank_9_and_deeper", 9, None),
)

# No single connected component may carry more than this share of an
# evaluation set's items.  Above it the set is one correlated cluster and any
# generalization number read off it is a number about that cluster.
MAX_EVAL_COMPONENT_SHARE = 0.25

# Distinct from the sealed evaluator's component namespace on purpose: this
# extractor's event universe is not the sealed one, so a digest that collided
# would assert an identity that does not hold.
EVENT_DIGEST_NAMESPACE = "recall-map-coldstart-event-v1"
COMPONENT_DIGEST_NAMESPACE = "coldstart-component-v1"
LINK_DIGEST_NAMESPACE = "coldstart-component-link-v1"

SCORE_FIELDS = ("score", "bm25_score", "vector_score", "graph_score", "trigger_score")
LEVELS = ("trace", "concept", "schema")

# ``storage.MAX_RECALL_HISTORY_DELIVERIES_PER_NODE``.  Above this the live
# reader answers *unavailable* rather than an inexact M/C/K, so such a node
# receives the frozen train means — it is neither cold nor warm to the scorer.
# Mirrored rather than imported so this extractor never loads the writable
# store module against a live database siblings are also reading.
MAX_HISTORY_DELIVERIES_PER_NODE = 1024

# ---------------------------------------------------------------------------
# Independence accounting
# ---------------------------------------------------------------------------

# The intra-split cap grid.  A cap says: use at most this many of a component's
# rows for ESTIMATION, inside the split that component was already assigned to.
CAP_GRID = (25, 50, 100, 200, 400)

# Which rows a cap keeps.  Digest order over label-free fields only, so the
# retained subsample is an arbitrary but reproducible draw from the component
# rather than its earliest or its highest-ranked rows, and the caps NEST: the
# survivors at 25 are a subset of the survivors at 50.
CAP_SUBSAMPLE_NAMESPACE = "coldstart-component-cap-v1"

# ``coldstart-prereg-v2.json#direction_constraint.cold_subpopulation_clause``.
# Mirrored here so every arm of the accounting publishes the two cohort counts
# that decide whether its cold reading is measurable at all.
MINIMUM_COLD_ROWS_FOR_A_DIRECTION_CHECK = 200
MINIMUM_CONSUMED_COLD_ROWS_FOR_A_DIRECTION_CHECK = 40

# The four members and their registered signs, mirrored from
# ``recall_map_coldstart_fit.py:CORE_FAMILY``.  The mirror is checked against
# the fit module at build time, so the two cannot drift apart silently.
FAMILY_DIRECTIONS: tuple[tuple[str, str, float], ...] = (
    ("score", "result_score", 1.0),
    ("trigger_score", "trigger_score", 1.0),
    ("bm25_score", "bm25_score", -1.0),
    ("vector_score", "vector_score", -1.0),
)

# The step-3 numbers the v2 fit recorded its HALT on, at
# ``policy.json#coldstart_revision.decision_procedure[2].report.members``.
# The unweighted arm of this accounting must reproduce them exactly: an
# explanation of a HALT that does not reproduce the HALT's own numbers is an
# explanation of something else.  If a successor legitimately moves the
# population, it updates this pin deliberately and says so.
PUBLISHED_STEP3_RANK_UNIFORM: dict[str, dict[str, float]] = {
    "result_score": {"whole_tail": 0.02682, "cold": -0.051024},
    "trigger_score": {"whole_tail": 0.0191, "cold": 0.010233},
    "bm25_score": {"whole_tail": 0.004903, "cold": -0.011408},
    "vector_score": {"whole_tail": -0.029671, "cold": -0.096195},
}

# The split the label-bearing half of the accounting may touch, and the only
# one.  ``coldstart_eval`` and ``coldstart_holdout`` carry the falsifiable
# direction test a successor plan will register; a family-member statistic
# published on either of them here would spend it before it was registered.
DIRECTION_ACCOUNTING_SPLIT = "train"

CALIBRATION_MODULE = REPO_ROOT / "scripts" / "recall_map_coldstart_fit.py"
_CALIBRATION: ModuleType | None = None

_SEALED: ModuleType | None = None


class ColdStartError(SystemExit):
    """A pinned-input, privacy, split, or maturation invariant failed."""


def sealed() -> ModuleType:
    """Import the sealed evaluator as a frozen library, bytes checked first.

    Nothing in it is executed for its evidence; it is imported for the
    protocol bindings and for the guards that must stay identical across both
    artifacts.  A modified copy is refused before any of its Python runs.
    """

    global _SEALED
    if _SEALED is not None:
        return _SEALED
    if not SEALED_EVALUATOR.is_file():
        raise ColdStartError(f"sealed evaluator not found: {SEALED_EVALUATOR}")
    digest = _sha256_path(SEALED_EVALUATOR)
    if digest != EXPECTED_EVALUATOR_SHA256:
        raise ColdStartError(
            "sealed evaluator hash mismatch: expected "
            f"{EXPECTED_EVALUATOR_SHA256}, got {digest}"
        )
    name = "_living_memory_frozen_recall_map_relevance_eval"
    spec = importlib.util.spec_from_file_location(name, SEALED_EVALUATOR)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise ColdStartError(f"cannot import sealed evaluator: {SEALED_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:  # pragma: no cover - import plumbing
        sys.modules.pop(name, None)
        raise
    _SEALED = module
    return module


def calibration() -> ModuleType:
    """Import the sibling fit script for its rank-calibration table, by path.

    The direction accounting has to be computed on the SAME ``u`` axis the fit
    published its step-3 numbers on, or the arms below would be comparable
    only to each other and not to the HALT they exist to explain.  Reusing
    ``StepFunction`` is the only way to guarantee that; a second copy of the
    atom-plus-512-knot rule here would be a copy that can drift.

    The dependency runs one way at run time.  The fit imports this extractor
    from :func:`recall_map_coldstart_fit.extractor`, but only from inside its
    own functions, and it never calls :func:`build_coldstart_block` -- so
    loading it back here is a leaf, not a cycle.
    """

    global _CALIBRATION
    if _CALIBRATION is not None:
        return _CALIBRATION
    if not CALIBRATION_MODULE.is_file():
        raise ColdStartError(f"calibration module not found: {CALIBRATION_MODULE}")
    name = "_living_memory_coldstart_fit_calibration"
    spec = importlib.util.spec_from_file_location(name, CALIBRATION_MODULE)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise ColdStartError(f"cannot import calibration module: {CALIBRATION_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:  # pragma: no cover - import plumbing
        sys.modules.pop(name, None)
        raise
    mirrored = tuple((source, member, sign) for source, member, sign in FAMILY_DIRECTIONS)
    if tuple(module.CORE_FAMILY) != mirrored:
        raise ColdStartError(
            "FAMILY_DIRECTIONS has drifted from recall_map_coldstart_fit.CORE_FAMILY: "
            f"{tuple(module.CORE_FAMILY)!r} != {mirrored!r}"
        )
    _CALIBRATION = module
    return module


def _sha256_path(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return sealed().canonical_json(value)


def sha256_text(value: str) -> str:
    return sealed()._sha256_text(value)


def assert_not_holdout(path: Path | str) -> None:
    """Delegate to the sealed guard so the marker list cannot drift."""

    try:
        sealed()._assert_not_holdout(path)
    except SystemExit as exc:
        raise ColdStartError(str(exc)) from exc


def parse_instant(value: str) -> Any:
    return sealed()._parse_instant(value)


def shift_hours(value: str, hours: int) -> str:
    return sealed()._shift_hours(value, hours)


def log_count(value: int | float) -> float:
    return round(math.log1p(max(0.0, float(value))), 8)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Candidate:
    """One residual candidate the pool saw, at the instant it saw it.

    ``matured`` / ``matured_consumed`` / ``trailing_nonconsumed`` are the exact
    M/C/K the live selector would have read at ``decided_at``; ``cold`` is
    ``matured == 0``, which is the distinction the whole recalibration turns on.
    """

    source: str
    event_id: str
    node_id: str
    decided_at: str
    rank: int
    consumed: bool
    matured: int
    matured_consumed: int
    trailing_nonconsumed: int
    opportunity: bool
    transport_session_id: str | None = None
    component_id: str = ""
    split: str = ""
    level: str = ""
    node_age_log_days: float | None = None
    scores: dict[str, float | None] = field(default_factory=dict)

    @property
    def cold(self) -> bool:
        return self.matured == 0


@dataclass(slots=True)
class EventIdentity:
    source: str
    event_id: str
    created_at: str
    qualified_cache_key: str
    qualified_sessions: tuple[str, ...]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(str(raw or "{}"))
    except ValueError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _load_event_identities(
    source_name: str,
    connection: sqlite3.Connection,
    events: Sequence[Any],
    cache_key_fn: Any,
    *,
    as_of: str,
) -> dict[str, EventIdentity]:
    """The sealed identity rule, read at this extractor's as-of.

    Deliberately a copy of the sealed shape rather than a call into it: the
    sealed loader filters on its own hardcoded ``AS_OF``, which is earlier than
    this one, and silently dropping later events would weaken the very
    component bridges the split depends on.
    """

    columns = _table_columns(connection, "recall_events")
    wanted = [
        name
        for name in ("id", "results", "session_id", "ambient_context")
        if name in columns
    ]
    rows = connection.execute(
        f"SELECT {', '.join(wanted)} FROM recall_events "
        "WHERE created_at <= ? ORDER BY created_at, id",
        (as_of,),
    ).fetchall()
    raw_by_id = {str(row["id"]): row for row in rows}
    identities: dict[str, EventIdentity] = {}
    for event in events:
        raw = raw_by_id.get(event.id)
        if raw is None:
            continue
        ambient = _json_object(raw["ambient_context"] if "ambient_context" in columns else None)
        task_pattern = str(ambient.get("task_pattern") or "").strip() or None
        local_cache = cache_key_fn(
            event.scope or "global", task=event.task, task_pattern=task_pattern
        )
        sessions: list[str] = []
        if event.transport_session_id:
            sessions.append(f"{source_name}\0transport\0{event.transport_session_id}")
        session_id = None
        if "session_id" in columns:
            session_id = str(raw["session_id"] or "").strip() or None
        if session_id:
            sessions.append(f"{source_name}\0session\0{session_id}")
        identities[event.id] = EventIdentity(
            source=source_name,
            event_id=event.id,
            created_at=event.created_at,
            qualified_cache_key=f"{source_name}\0cache\0{local_cache}",
            qualified_sessions=tuple(sessions),
        )
    return identities


@dataclass(slots=True)
class NodeLedger:
    """One node's whole delivery history, in the live reader's own order.

    ``instants`` is the bisect key; ``outcomes`` is aligned with it.  Sorting by
    ``(delivered_at, consumed, delivery_event_id)`` ascending and reading it
    backwards reproduces the live ``ORDER BY outcome_end DESC, delivered_at
    DESC, consumed DESC, delivery_event_id DESC`` exactly, which matters
    because K is the *leading* run of that order.
    """

    instants: list[str] = field(default_factory=list)
    outcomes: list[bool] = field(default_factory=list)


def _load_delivery_ledger(
    connection: sqlite3.Connection,
) -> tuple[dict[str, NodeLedger], dict[tuple[str, str], bool], dict[str, Any]]:
    """Per-node matured-delivery history plus each delivery's own outcome.

    ``matured_recall_history`` selects rows with ``delivered_at < decision`` and
    ``outcome_end <= decision``.  This asserts the ledger's own invariant that
    ``outcome_end`` is exactly ``delivered_at + horizon_hours`` and then reduces
    that predicate to a single sorted-list bisect on ``delivered_at``, which
    selects the same rows and is what makes a 21k-candidate sweep tractable.
    """

    state = connection.execute(
        "SELECT format_version, complete, unavailable_reason "
        "FROM recall_delivery_history_state WHERE singleton = 1"
    ).fetchone()
    if state is None:
        raise ColdStartError("delivery-history ledger has no state row")
    if not int(state["complete"]):
        raise ColdStartError(
            "delivery-history ledger is incomplete: "
            f"{state['unavailable_reason'] or 'unknown'}"
        )

    staged: dict[str, list[tuple[str, bool, str]]] = defaultdict(list)
    outcome: dict[tuple[str, str], bool] = {}
    horizons: Counter[float] = Counter()
    rows = connection.execute(
        "SELECT delivery_event_id, node_id, delivered_at, outcome_end, "
        "transport_matched, transport_consumed, fallback_consumed "
        "FROM recall_delivery_history"
    )
    for row in rows:
        delivered_at = str(row["delivered_at"])
        gap = (
            parse_instant(str(row["outcome_end"])) - parse_instant(delivered_at)
        ).total_seconds() / 3600.0
        horizons[round(gap, 6)] += 1
        consumed = bool(
            row["transport_consumed"] if row["transport_matched"] else row["fallback_consumed"]
        )
        node_id = str(row["node_id"])
        event_id = str(row["delivery_event_id"])
        staged[node_id].append((delivered_at, consumed, event_id))
        outcome[(event_id, node_id)] = consumed

    by_node: dict[str, NodeLedger] = {}
    for node_id, entries in staged.items():
        entries.sort()
        by_node[node_id] = NodeLedger(
            instants=[entry[0] for entry in entries],
            outcomes=[entry[1] for entry in entries],
        )
    ledger_meta = {
        "format_version": int(state["format_version"]),
        "complete": True,
        "rows": len(outcome),
        "distinct_nodes": len(by_node),
        "outcome_end_offset_hours": sorted(horizons),
        "per_node_read_limit": MAX_HISTORY_DELIVERIES_PER_NODE,
    }
    return by_node, outcome, ledger_meta


def _matured_history_at(ledger: NodeLedger | None, cutoff: str) -> tuple[int, int, int]:
    """The exact M, C, K the live reader would return at this instant.

    ``cutoff`` is ``decision_at - horizon_hours``; because ``outcome_end`` is
    exactly ``delivered_at + horizon_hours``, ``delivered_at <= cutoff`` is the
    same predicate as the live pair ``delivered_at < decision AND outcome_end
    <= decision``.
    """

    if ledger is None:
        return 0, 0, 0
    matured = bisect.bisect_right(ledger.instants, cutoff)
    if matured == 0:
        return 0, 0, 0
    outcomes = ledger.outcomes[:matured]
    consumed = sum(outcomes)
    streak = 0
    for was_consumed in reversed(outcomes):
        if was_consumed:
            break
        streak += 1
    return matured, consumed, streak


def _extract_window(
    *,
    events: Sequence[Any],
    connection: sqlite3.Connection,
    effect: Any,
    source_name: str,
    window_start: str,
    window_end: str,
    head_cut: int,
    organic_cap: int | None,
    horizon: int,
    consumer_index: Any,
    by_node: Mapping[str, "NodeLedger"],
    outcome: Mapping[tuple[str, str], bool],
) -> tuple[list[Candidate], list[Any], int, int]:
    """Every residual candidate one window's events offered, at their instant.

    ``organic_cap`` of ``None`` means the whole recorded tail: the frozen
    ``organic_items`` is still what selects and de-duplicates, called with a cap
    equal to the event's own result count, so lifting the cap changes which
    ranks are in scope and nothing else about how an item is derived.
    """

    window_events = [event for event in events if window_start <= event.created_at < window_end]
    raw_by_id = {
        str(row["id"]): row
        for row in connection.execute(
            "SELECT id, results FROM recall_events WHERE created_at >= ? AND created_at < ?",
            (window_start, window_end),
        )
    }
    candidates: list[Candidate] = []
    unledgered = 0
    truncated = 0
    for event in window_events:
        opportunity = bool(consumer_index.qualifying(event))
        maturation_cut = shift_hours(event.created_at, -horizon)
        recorded = {
            str(entry.get("node_id")): entry
            for entry in _json_list(raw_by_id.get(event.id, {}))
            if isinstance(entry, dict) and entry.get("node_id")
        }
        cap = len(event.result_ids) if organic_cap is None else organic_cap
        for offset, item in enumerate(
            effect.organic_items(event, head_cut=head_cut, cap=cap)
        ):
            label = outcome.get((event.id, item.node_id))
            if label is None:
                unledgered += 1
                continue
            matured, consumed_prior, streak = _matured_history_at(
                by_node.get(item.node_id), maturation_cut
            )
            if matured > MAX_HISTORY_DELIVERIES_PER_NODE:
                truncated += 1
            entry = recorded.get(item.node_id, {})
            candidates.append(
                Candidate(
                    source=source_name,
                    event_id=event.id,
                    node_id=item.node_id,
                    decided_at=event.created_at,
                    rank=head_cut + offset,
                    consumed=label,
                    matured=matured,
                    matured_consumed=consumed_prior,
                    trailing_nonconsumed=streak,
                    opportunity=opportunity,
                    transport_session_id=event.transport_session_id,
                    level=str(entry.get("level") or ""),
                    scores={name: _float_or_none(entry.get(name)) for name in SCORE_FIELDS},
                )
            )
    return candidates, window_events, unledgered, truncated


def rank_coverage_census(
    events: Sequence[Any],
    outcome: Mapping[tuple[str, str], bool],
    *,
    depth: int = RANK_COVERAGE_CENSUS_DEPTH,
) -> dict[str, Any]:
    """How deep into a delivered result list the outcome ledger actually goes.

    This is what decides whether the per-event cap is a lever at all.  A rank
    with recorded results but no ledger rows is a rank that carries no
    consumption label, and an unlabelable row cannot enter a supervised
    dataset however much the population needs rows.  Counts only.
    """

    ledgered: Counter[int] = Counter()
    unledgered: Counter[int] = Counter()
    for event in events:
        for rank, node_id in enumerate(event.result_ids):
            if rank >= depth:
                break
            if (event.id, node_id) in outcome:
                ledgered[rank] += 1
            else:
                unledgered[rank] += 1
    covered = sorted(rank for rank in ledgered if ledgered[rank])
    return {
        "depth_examined": depth,
        "by_rank": {
            str(rank): {
                "results_recorded": ledgered[rank] + unledgered[rank],
                "outcome_ledgered": ledgered[rank],
            }
            for rank in range(depth)
            if ledgered[rank] + unledgered[rank]
        },
        "ledgered_ranks": covered,
        "first_ledgered_rank": covered[0] if covered else None,
        "last_ledgered_rank": covered[-1] if covered else None,
    }


def build_population(
    db_path: Path,
    *,
    source_name: str = "local",
    as_of: str = COLDSTART_AS_OF,
    window_start: str = CANDIDATE_WINDOW_START,
    v1_window_start: str = V1_CANDIDATE_WINDOW_START,
) -> dict[str, Any]:
    """Rebuild every residual candidate the pool saw inside the window.

    Two populations come back, not one.  ``candidates`` is the enlarged (v2)
    population this artifact publishes; ``v1_candidates`` is the population the
    committed manifest published, rebuilt from the same connection under the v1
    window and the v1 cap.  The v1 rebuild is not decoration: the split
    assignment is not stable under a change of population, so the only way to
    enlarge without scattering already-read components into the sealed holdout
    is to reconstruct the v1 role of every event first and carry it over.

    The connection is opened by the frozen effect tool, which opens ``mode=ro``:
    sibling nodes are reading the same store concurrently and none of this may
    take a write lock.
    """

    module = sealed()
    assert_not_holdout(db_path)
    spec = module.SourceSpec.parse(f"{source_name}={db_path}")
    bindings = module.load_frozen_bindings()
    protocol = bindings.protocol

    horizon = int(protocol.horizon_hours)
    head_cut = int(protocol.head_cut)
    organic_cap = int(protocol.organic_cap)
    map_cap = int(protocol.map_cap)
    window_end = shift_hours(as_of, -horizon)
    if not window_start < window_end:
        raise ColdStartError("candidate window is empty at this as-of")
    if not window_start <= v1_window_start:
        raise ColdStartError("enlarged window does not contain the v1 window")

    receipt = module.snapshot_receipt(spec)
    connection = bindings.effect.open_readonly(spec.path)
    try:
        connection.execute("PRAGMA query_only = ON")
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        missing = {"recall_events", "nodes", "recall_delivery_history"} - tables
        if missing:
            raise ColdStartError(
                f"source {source_name!r} lacks required table(s): {', '.join(sorted(missing))}"
            )

        newest = connection.execute("SELECT MAX(created_at) FROM recall_events").fetchone()[0]
        if newest is None:
            raise ColdStartError("source snapshot has no recall events")
        snapshot_instant = str(newest)
        # The claim COLDSTART_AS_OF makes: every 24-hour outcome window that
        # can appear in this dataset closed before the snapshot was taken.
        if snapshot_instant < shift_hours(as_of, horizon):
            raise ColdStartError(
                "snapshot does not reach as_of + horizon_hours; windows in scope "
                f"have not matured (newest event {snapshot_instant})"
            )

        has_map = bindings.effect._has_recall_map_column(connection)
        earliest = connection.execute("SELECT MIN(created_at) FROM recall_events").fetchone()[0]
        # The claim CANDIDATE_WINDOW_START makes: the enlarged window excludes
        # no era, because it opens before anything was ever recorded.  Checked
        # against the snapshot rather than asserted in a comment.
        if not window_start <= str(earliest):
            raise ColdStartError(
                "enlarged candidate window starts after the snapshot's first recall event "
                f"({earliest}); it would silently cut an era"
            )
        events = bindings.effect.load_events(
            connection, start=str(earliest), end=as_of, with_map=has_map
        )
        identities = _load_event_identities(
            source_name, connection, events, bindings.cache_key, as_of=as_of
        )
        events = [event for event in events if event.id in identities]
        consumer_index = bindings.effect.ConsumerIndex(events, horizon_hours=horizon)
        by_node, outcome, ledger_meta = _load_delivery_ledger(connection)
        if ledger_meta["outcome_end_offset_hours"] != [float(horizon)]:
            raise ColdStartError(
                "ledger outcome_end is not exactly delivered_at + horizon_hours: "
                f"{ledger_meta['outcome_end_offset_hours']}"
            )

        shared = {
            "events": events,
            "connection": connection,
            "effect": bindings.effect,
            "source_name": source_name,
            "window_end": window_end,
            "head_cut": head_cut,
            "horizon": horizon,
            "consumer_index": consumer_index,
            "by_node": by_node,
            "outcome": outcome,
        }
        candidates, window_events, unledgered, truncated = _extract_window(
            window_start=window_start, organic_cap=organic_cap, **shared
        )
        v1_candidates, v1_window_events, _, _ = _extract_window(
            window_start=v1_window_start, organic_cap=organic_cap, **shared
        )
        coverage = rank_coverage_census(window_events, outcome)
        if not candidates:
            raise ColdStartError("pinned source produced no residual candidates")
        if not v1_candidates:
            raise ColdStartError("the v1 population could not be reconstructed")
        _attach_node_age(connection, [*candidates, *v1_candidates])

        # Mirrors scripts/recall_map_relevance_eval.py:1666: a delivery of the
        # feature itself, at or after the instant the feature was deployed.
        deploy = bindings.effect.feature_deploy_instant(connection, as_of=as_of)
        observed_map_event_ids = {
            event.id
            for event in window_events
            if deploy is not None and event.recall_map and deploy <= event.created_at
        }
        return {
            "spec": spec,
            "bindings": bindings,
            "receipt": receipt,
            "connection": connection,
            "identities": identities,
            "candidates": candidates,
            "window_events": window_events,
            "v1_candidates": v1_candidates,
            "v1_window_events": v1_window_events,
            "observed_map_event_ids": observed_map_event_ids,
            "feature_deploy_instant": deploy,
            "snapshot_first_event": str(earliest),
            "consumer_index": consumer_index,
            "ledger": ledger_meta,
            "unledgered": unledgered,
            "history_read_truncated": truncated,
            "snapshot_instant": snapshot_instant,
            "rank_coverage": coverage,
            "protocol": {
                "horizon_hours": horizon,
                "organic_head_cut": head_cut,
                "organic_cap": organic_cap,
                "map_cap": map_cap,
                "candidate_window": [window_start, window_end],
                "v1_candidate_window": [v1_window_start, window_end],
            },
        }
    except BaseException:
        connection.close()
        raise


def _json_list(row: Any) -> list[Any]:
    if row is None:
        return []
    try:
        raw = row["results"]
    except (TypeError, IndexError, KeyError):
        return []
    try:
        parsed = json.loads(str(raw or "[]"))
    except ValueError:
        return []
    return list(parsed) if isinstance(parsed, list) else []


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), 8)


def _attach_node_age(connection: sqlite3.Connection, candidates: Sequence[Candidate]) -> None:
    """Age at decision time from the immutable ``created_at`` only.

    ``access_count``, ``usefulness_score``, ``last_accessed`` and ``updated_at``
    are mutated after the decision instant and are therefore never read here —
    the same rejection the sealed evaluator's feature allowlist makes.
    """

    wanted = sorted({candidate.node_id for candidate in candidates})
    rows: dict[str, sqlite3.Row] = {}
    for offset in range(0, len(wanted), 500):
        batch = wanted[offset : offset + 500]
        marks = ",".join("?" for _ in batch)
        for row in connection.execute(
            f"SELECT id, level, created_at FROM nodes WHERE id IN ({marks})", batch
        ):
            rows[str(row["id"])] = row
    for candidate in candidates:
        row = rows.get(candidate.node_id)
        if row is None:
            continue
        if not candidate.level:
            candidate.level = str(row["level"] or "")
        try:
            seconds = (
                parse_instant(candidate.decided_at) - parse_instant(str(row["created_at"]))
            ).total_seconds()
        except (TypeError, ValueError):
            continue
        if seconds >= 0:
            candidate.node_age_log_days = round(math.log1p(seconds / 86_400.0), 8)


# ---------------------------------------------------------------------------
# Three-way component split
# ---------------------------------------------------------------------------


def build_components(
    candidates: Sequence[Candidate],
    identities: Mapping[str, EventIdentity],
    *,
    component_event_ids: Iterable[str],
) -> list[str]:
    """The sealed identity rule, unchanged, stamped onto every candidate.

    Membership stays event-level, so an event with no scorable residual item
    still bridges two events that share its cache identity or either session
    identity.  Only the *assignment* of components to partitions differs from
    the sealed two-way split; what counts as one component does not.

    Returns the event -> component map.  The map, not just the component list,
    is what the carry-over rule needs: a component's digest is taken over its
    whole member set, so a single new bridging event changes the id and the
    only durable handle on "the same cluster of work" is the events inside it.
    """

    module = sealed()
    event_ids = sorted(set(component_event_ids) | {c.event_id for c in candidates})
    missing = [event_id for event_id in event_ids if event_id not in identities]
    if missing:
        raise ColdStartError("component event has no loaded identity")

    opaque_by_event = {
        event_id: sha256_text(
            f"{EVENT_DIGEST_NAMESPACE}\0{identities[event_id].source}\0{event_id}"
        )
        for event_id in event_ids
    }
    union = module._UnionFind(opaque_by_event.values())
    owners: dict[str, str] = {}
    for event_id in event_ids:
        identity = identities[event_id]
        opaque = opaque_by_event[event_id]
        for value in (identity.qualified_cache_key, *identity.qualified_sessions):
            token = sha256_text(f"{LINK_DIGEST_NAMESPACE}\0{value}")
            union.union(opaque, owners.setdefault(token, opaque))

    groups: dict[str, list[str]] = defaultdict(list)
    for opaque in opaque_by_event.values():
        groups[union.find(opaque)].append(opaque)
    component_for_root = {
        root: sha256_text(COMPONENT_DIGEST_NAMESPACE + "\0" + "\0".join(sorted(digests)))
        for root, digests in groups.items()
    }
    component_by_event = {
        event_id: component_for_root[union.find(opaque)]
        for event_id, opaque in opaque_by_event.items()
    }
    for candidate in candidates:
        candidate.component_id = component_by_event[candidate.event_id]
    return component_by_event


def _uniform_draw_assignment(components: Sequence[str], *, seed: str) -> dict[str, str]:
    """The literal three-way extension of the sealed per-component draw.

    Kept because it is the obvious rule and because it *fails* here; the
    failure is published as a diagnostic rather than quietly discarded.
    """

    placement: dict[str, str] = {}
    for component_id in sorted(components):
        draw = int(sha256_text(f"{seed}\0{component_id}")[:16], 16) / float(16**16)
        if draw < TRAIN_FRACTION:
            placement[component_id] = "train"
        elif draw < TRAIN_FRACTION + EVAL_FRACTION:
            placement[component_id] = "eval"
        else:
            placement[component_id] = "holdout"
    return placement


def _balanced_assignment(
    sizes: Mapping[str, int], *, seed: str
) -> dict[str, str]:
    """Deterministic largest-first balancing, components kept atomic.

    A per-component coin flip is only unbiased in *components*, and these
    components differ in size by three orders of magnitude, so it partitions
    items arbitrarily.  Walking components largest-first into whichever split
    is furthest below its item target is the standard fix: no component is ever
    divided, so no cache or session identity spans two splits, while the item
    fractions land on their targets and no single component can dominate an
    evaluation set.

    Ties in size are broken by a seeded hash, not by digest order, so the
    result is not an artifact of how the digests happen to sort.
    """

    total = sum(sizes.values())
    targets = {
        "train": TRAIN_FRACTION * total,
        "eval": EVAL_FRACTION * total,
        "holdout": (1.0 - TRAIN_FRACTION - EVAL_FRACTION) * total,
    }
    load = dict.fromkeys(SPLIT_NAMES, 0)
    placement: dict[str, str] = {}
    order = sorted(
        sizes, key=lambda component: (-sizes[component], sha256_text(f"{seed}\0{component}"))
    )
    for component_id in order:
        # ``max`` returns the first maximum, so SPLIT_NAMES order breaks ties.
        chosen = max(SPLIT_NAMES, key=lambda name: targets[name] - load[name])
        placement[component_id] = chosen
        load[chosen] += sizes[component_id]
    return placement


def _concentration(rows: Sequence[Candidate]) -> dict[str, Any]:
    """How many *independent* units a split really contains.

    A split whose items nearly all come from one component is one observation
    wearing a thousand hats; reporting only the component count would hide
    that.  ``effective_components`` is the inverse Simpson index over the
    component item shares.
    """

    sizes = Counter(row.component_id for row in rows)
    total = len(rows)
    if not total:
        return {"components": 0, "largest_component_items": 0}
    largest = max(sizes.values())
    simpson = sum((count / total) ** 2 for count in sizes.values())
    return {
        "components": len(sizes),
        "largest_component_items": largest,
        "largest_component_share": _ratio(largest, total),
        "effective_components": round(1.0 / simpson, 2) if simpson else None,
    }


def _membership_digest(components: Iterable[str]) -> str:
    """A split's identity: sha256 over its sorted opaque component digests."""

    return sha256_text(canonical_json(sorted(set(components))))


def reconstruct_v1_partition(
    v1_candidates: Sequence[Candidate],
    identities: Mapping[str, EventIdentity],
    *,
    component_event_ids: Iterable[str],
    seed: str = SPLIT_SEED,
) -> dict[str, Any]:
    """Rebuild the committed v1 partition and hand back its event -> role map.

    This has to happen before the enlarged population exists, and it has to be
    checked, because the whole carry-over rests on it.  Neither half of the
    split survives a change of population: ``component_id`` digests the
    component's entire member set, so one new bridging event renames it, and
    ``_balanced_assignment`` is greedy against per-split item targets computed
    over the whole population, so every placement moves when the population
    does.  A naive re-run over a wider window would therefore scatter
    components already read as train or eval into the new holdout and every
    published count would still look healthy.

    Fails closed if the rebuild does not reproduce ``V1_SPLITS`` exactly: a v1
    role that is not the v1 role that was actually read protects nothing.
    """

    opportunity = [row for row in v1_candidates if row.opportunity]
    if not opportunity:
        raise ColdStartError("the v1 population carries no opportunity-bearing candidate")
    component_by_event = build_components(
        opportunity, identities, component_event_ids=component_event_ids
    )
    sizes = Counter(row.component_id for row in opportunity)
    for component_id in set(component_by_event.values()):
        sizes.setdefault(component_id, 0)
    placement = _balanced_assignment(sizes, seed=seed)

    reproduced: dict[str, Any] = {}
    for name in SPLIT_NAMES:
        rows = [row for row in opportunity if placement[row.component_id] == name]
        members = sorted({row.component_id for row in rows})
        reproduced[name] = {
            "items": len(rows),
            "cold_candidates": sum(1 for row in rows if row.cold),
            "components": len(members),
            "effective_components": _concentration(rows).get("effective_components"),
            "membership_digest": _membership_digest(members),
        }
    drift = [name for name in SPLIT_NAMES if reproduced[name] != dict(V1_SPLITS[name])]
    if drift:
        raise ColdStartError(
            "the committed v1 partition did not reproduce, so no v1 role can be carried "
            f"over safely: {', '.join(drift)}"
        )

    role_by_event = {
        event_id: placement[component_id]
        for event_id, component_id in component_by_event.items()
    }
    return {
        "role_by_event": role_by_event,
        "receipt": {
            "checked_against": (
                "the v1 partition as committed at dataset-manifest.json#coldstart.splits, "
                "pinned in this extractor as V1_SPLITS"
            ),
            "reproduces_pinned_v1": True,
            "window": [V1_CANDIDATE_WINDOW_START, None],
            "splits": reproduced,
            "events_carrying_a_v1_role": len(role_by_event),
            "why_it_must_be_checked": (
                "component ids and balanced placements both move with the population, so a "
                "v1 role reconstructed from a drifted rebuild would authorise exactly the "
                "contamination the carry-over exists to prevent"
            ),
        },
    }


def carry_over_assignment(
    sizes: Mapping[str, int],
    component_by_event: Mapping[str, str],
    *,
    v1_role_by_event: Mapping[str, str],
    forced_eval_components: Iterable[str] = (),
    seed: str = SPLIT_SEED,
) -> dict[str, Any]:
    """Inherit first; balance only what is genuinely new.

    A v2 component inherits the MOST-READ role among the v1 roles of the events
    it contains, precedence train > eval > holdout, so a component whose work
    has been read as train can never reappear as eval or holdout and one read
    as eval can never reappear as holdout.  A component carrying an observed-map
    delivery is forced to eval regardless, which is the existing protection at
    ``scripts/recall_map_relevance_eval.py:702`` and which moves rows only
    *out* of train and holdout.  Everything else — wholly-new components and
    pure v1-holdout components — is balanced toward the item targets, and it is
    the only material that can reach the holdout at all.
    """

    v1_roles: dict[str, set[str]] = defaultdict(set)
    for event_id, role in v1_role_by_event.items():
        component_id = component_by_event.get(event_id)
        if component_id is not None:
            v1_roles[component_id].add(role)

    placement: dict[str, str] = {}
    reason: dict[str, str] = {}
    for component_id, roles in v1_roles.items():
        if component_id not in sizes:
            continue
        placement[component_id] = next(role for role in ROLE_PRECEDENCE if role in roles)
        reason[component_id] = "inherited_most_read_v1_role"
    forced = sorted(set(forced_eval_components) & set(sizes))
    for component_id in forced:
        placement[component_id] = "eval"
        reason[component_id] = "touches_observed_map_delivery"

    total = sum(sizes.values())
    if not total:
        raise ColdStartError("cannot balance an empty population")
    targets = {
        "train": TRAIN_FRACTION * total,
        "eval": EVAL_FRACTION * total,
        "holdout": (1.0 - TRAIN_FRACTION - EVAL_FRACTION) * total,
    }
    load = dict.fromkeys(SPLIT_NAMES, 0)
    for component_id, name in placement.items():
        load[name] += int(sizes[component_id])
    fresh = sorted(
        (component_id for component_id in sizes if component_id not in placement),
        key=lambda component: (-sizes[component], sha256_text(f"{seed}\0{component}")),
    )
    for component_id in fresh:
        # RELATIVE deficit, not absolute.  The published contract is a set of
        # item *fractions*, and the carry-over hands the balancer a starting
        # load it did not choose — the forced observed-map arm alone can push
        # one split past its target before a single fresh component is placed.
        # Equalising the absolute shortfall would then take the overshoot out of
        # the smallest split as hard as out of the largest, which is the wrong
        # loss for a fraction target.  ``max`` returns the first maximum, so
        # SPLIT_NAMES order still breaks exact ties.
        chosen = max(SPLIT_NAMES, key=lambda name: (targets[name] - load[name]) / targets[name])
        placement[component_id] = chosen
        reason[component_id] = "freshly_assigned"
        load[chosen] += int(sizes[component_id])
    return {
        "placement": placement,
        "reason": reason,
        "fresh_components": fresh,
        "forced_eval_components": forced,
        "inherited_components": sorted(
            component_id
            for component_id, why in reason.items()
            if why == "inherited_most_read_v1_role"
        ),
        "achieved_item_fractions": {
            name: _ratio(load[name], total) for name in SPLIT_NAMES
        },
    }


# ---------------------------------------------------------------------------
# Publishing the assignment's inputs, and re-executing it from them alone
#
# ``coldstart-prereg-v2.json#split_protocol.assignment_rule.blindness_invariant_binding``
# does not accept "the assignment is blind" as a claim.  It requires the
# manifest to publish the assignment function's NAME and its INPUTS, and a
# re-executable check that recomputes the assignment from those inputs alone
# and reproduces the published membership digests.  The v1 receipt already in
# the artifact reproduces the *v1* partition, which is a different property:
# it proves the carried-over roles are the roles that were really read, not
# that the v2 placement is a function of published, label-free inputs.
#
# So the inputs are enumerated exhaustively below, published per component in a
# compact table, and replayed by ``replay_published_assignment`` -- a function
# whose only data argument is that published table.  It cannot reach a
# candidate row, the database or the placement dict, which is the whole point:
# if it reproduces the three membership digests, then every quantity the
# assignment used is on the page, and a reader can check the list against the
# invariant's exclusions instead of trusting a sentence.
# ---------------------------------------------------------------------------


DECLARED_ASSIGNMENT_INPUTS: tuple[dict[str, str], ...] = (
    {
        "name": "seed",
        "what": (
            "the constant SPLIT_SEED = 'recall-map-coldstart-components-v1', the same seed "
            "the v1 draw was registered under"
        ),
        "why_it_carries_no_consumption_label": (
            "one fixed string for the whole population, chosen before any outcome was read; "
            "it does not vary with a component and so cannot encode one"
        ),
    },
    {
        "name": "component_digest",
        "what": (
            "the opaque component id: sha256 over the sorted opaque event digests of the "
            "component's members, used here only as identity and as the tie-break hash "
            "sha256_text(seed + NUL + component_digest)"
        ),
        "why_it_carries_no_consumption_label": (
            "a digest of an identity set, not of any outcome; nothing consumed, cold or "
            "scored enters the preimage"
        ),
    },
    {
        "name": "component_item_count",
        "what": (
            "the number of opportunity-bearing candidate rows the component contributes, "
            "published as n"
        ),
        "why_it_carries_no_consumption_label": (
            "the invariant permits balancing on item count explicitly: a count is fixed "
            "before any outcome is read"
        ),
    },
    {
        "name": "v1_partition_roles",
        "what": (
            "the set of v1 partition roles held by the events the component contains, "
            "published as r and resolved by role_precedence"
        ),
        "why_it_carries_no_consumption_label": (
            "a v1 role is _balanced_assignment(sizes, seed) over the v1 subpopulation: a "
            "deterministic function of the seed, the component digest and the item count -- "
            "exactly the three permitted inputs"
        ),
    },
    {
        "name": "observed_map_delivery_flag",
        "what": (
            "whether the component contains an event that carried a recall map at or after "
            "the feature deploy instant, published as f"
        ),
        "why_it_carries_no_consumption_label": (
            "it records that a map was DELIVERED, not that it was read, followed or useful; "
            "it is a deployment-time and delivery-time fact about the envelope"
        ),
    },
    {
        "name": "target_item_fractions_and_tie_break_order",
        "what": (
            "the constants TRAIN_FRACTION and EVAL_FRACTION, the holdout residual, and the "
            "SPLIT_NAMES order that breaks an exact tie in relative deficit"
        ),
        "why_it_carries_no_consumption_label": (
            "pre-registered protocol constants; they are the same three numbers and the same "
            "order for every component in the population"
        ),
    },
)


BLINDNESS_INVARIANT_READING: dict[str, Any] = {
    "invariant_text_says": (
        "the assignment must be computable from the seed, the opaque component digest, and "
        "the component ITEM COUNT alone"
    ),
    "this_assignment_also_reads": [
        "the set of v1 partition roles held by the events a component contains",
        "whether the component contains an observed-map delivery event",
    ],
    "verdict": "does not violate the invariant",
    "why": [
        "The named HALT trigger is narrow and specific: 'if the manifest's assignment reads any",
        "cold statistic, the fit HALTS'. The invariant's own exclusion list is a consumption label,",
        "a feature value, a score, and a coldness statistic. A v1 partition role is none of these.",
        "The decisive argument is that the v1 role is not a new information channel at all. The v1",
        "assignment was _balanced_assignment(sizes, seed), which reads the seed, the component",
        "digest and the item count and NOTHING else -- exactly the three permitted inputs. So the",
        "carry-over is a composition of two blind functions: the v1 role is a deterministic",
        "function of permitted inputs evaluated over the v1 subpopulation. Reading it reads no",
        "kind of information the invariant does not already permit.",
        "The observed-map flag is pre-registered by the plan itself at",
        "split_protocol.forced_assignment, so the plan cannot forbid its own registered rule. It is",
        "also label-free on its own terms: it says an event carried a recall map at or after the",
        "instant the effect-measuring feature was deployed. That is a deployment-time and",
        "delivery-time fact. It says nothing about whether the map was read, followed or useful.",
    ],
    "residual_leakage_is_the_one_already_disclosed": [
        "The invariant permits balancing on item count explicitly. Component size correlates with",
        "coldness, so any size-aware rule -- v1's or v2's -- moves per-split cold counts",
        "indirectly. assignment_rule.residual_risk_stated discloses exactly this and binds the fit",
        "to publish the v1 seed-uniform draw's per-split cold counts beside the shipped",
        "assignment's. Because the v1 role is itself a size-balanced placement, the carry-over",
        "inherits that same disclosed surface and introduces no second one.",
    ],
    # One sentence in the reading above, split across lines only because the
    # frozen privacy guard caps a published string at 200 printable ASCII
    # characters; every long passage in this artifact is a list for the same
    # reason and the words are unchanged.
    "what_would_have_triggered_a_halt": [
        "an assignment reading matured_recall_history, the cold flag, the consumed bit, any",
        "family member value, or any score. The published inputs contain none of these, and",
        "assignment_reproduction re-executes the assignment from the published inputs alone",
        "to prove it.",
    ],
}


ASSIGNMENT_COMPONENTS_ENCODING = (
    "one object per component, sorted by c: c is the opaque component digest, n its item "
    "count. r (the sorted v1 roles) is OMITTED when empty and f when false, so an omission "
    "is load-bearing."
)

ASSIGNMENT_COMPONENTS_ZERO_ITEM_NOTE = (
    "n = 0 is published, not dropped: such a component takes a placement and bridges "
    "identities but carries no row, so it appears in no per-split list. Hence these entries "
    "outnumber the members."
)


ASSIGNMENT_REPRODUCTION_PROCEDURE = [
    "1. Inherit. For every component whose r is non-empty, place it at the FIRST role in "
    "constants.role_precedence that appears in r (train beats eval beats holdout).",
    "2. Force. For every component whose f is true, overwrite its placement with eval. The "
    "observed-map rule outranks the carry-over and moves work only out of train and holdout.",
    "3. Total and targets. total is the sum of n over every published component; "
    "targets[name] is constants.target_item_fractions[name] * total.",
    "4. Starting load. load[name] is the sum of n over the components already placed in "
    "that split by steps 1 and 2, which is a load the balancer did not choose.",
    "5. Fresh order. The components with no placement yet are ordered by "
    "(-n, sha256_text(constants.seed + NUL + c)): largest first, the seeded digest breaking "
    "equal sizes.",
    "6. Balance. In that order each fresh component takes the split with the largest "
    "relative deficit (targets[s]-load[s])/targets[s]; then load[chosen] += n. Ties go to "
    "the first in split_names_order.",
    "7. Membership. A split's members are the sorted digests placed there whose n > 0, and "
    "its digest is sha256(canonical_json(sorted(members))). A zero-item component is "
    "published nowhere.",
]


def assignment_inputs_block(
    sizes: Mapping[str, int],
    component_by_event: Mapping[str, str],
    *,
    v1_role_by_event: Mapping[str, str],
    forced_eval_components: Iterable[str] = (),
    seed: str = SPLIT_SEED,
) -> dict[str, Any]:
    """Everything ``carry_over_assignment`` reads, published component by component.

    This is deliberately the *whole* input, not a summary of it.  A reader who
    wants to know whether the assignment touched a coldness statistic should
    not have to read this script: the table below is the complete argument list
    the placement is a function of, and ``replay_published_assignment`` proves
    the list is complete by reproducing the three membership digests from it.

    The per-component keys are one letter each because the universe is large
    and the manifest is read by humans as well as by tests; the convention is
    documented in ``components_encoding`` beside the table rather than left to
    be inferred.
    """

    roles_by_component: dict[str, set[str]] = defaultdict(set)
    for event_id, role in v1_role_by_event.items():
        component_id = component_by_event.get(event_id)
        if component_id is not None:
            roles_by_component[component_id].add(role)
    forced = set(forced_eval_components) & set(sizes)

    components: list[dict[str, Any]] = []
    for component_id in sorted(sizes):
        entry: dict[str, Any] = {"c": component_id, "n": int(sizes[component_id])}
        roles = sorted(roles_by_component.get(component_id, ()))
        if roles:
            entry["r"] = roles
        if component_id in forced:
            entry["f"] = True
        components.append(entry)

    return {
        "function": {
            "name": "v1_role_carry_over_then_size_aware_largest_first_balanced",
            "logical_location": (
                "scripts/recall_map_coldstart_dataset.py:carry_over_assignment"
            ),
            "driven_from": (
                "scripts/recall_map_coldstart_dataset.py:assign_three_way_splits"
            ),
        },
        "constants": {
            "seed": seed,
            "target_item_fractions": {
                "train": TRAIN_FRACTION,
                "eval": EVAL_FRACTION,
                "holdout": 1.0 - TRAIN_FRACTION - EVAL_FRACTION,
            },
            "target_item_fractions_are_the_divisors_used": [
                "These are the numbers the balancer divided by, not a restatement of them:",
                "TRAIN_FRACTION, EVAL_FRACTION, and the holdout as the residual 1.0 - train -",
                "eval, which on these constants evaluates to the double 0.2 bit for bit and so",
                "agrees with split.target_item_fractions exactly. Published because step 6",
                "compares RELATIVE deficits, so a divisor that drifted would move placements.",
            ],
            "split_names_order": list(SPLIT_NAMES),
            "role_precedence": list(ROLE_PRECEDENCE),
            "tie_break": [
                "Two ties are broken, and neither reads anything new. Equal-sized fresh",
                "components are ordered by sha256_text(seed + NUL + component_digest); and two",
                "splits with an exactly equal relative deficit are resolved by max() returning",
                "its first maximum, so split_names_order decides.",
            ],
        },
        "declared": [dict(entry) for entry in DECLARED_ASSIGNMENT_INPUTS],
        "components": components,
        "components_encoding": ASSIGNMENT_COMPONENTS_ENCODING,
        "components_zero_item_note": ASSIGNMENT_COMPONENTS_ZERO_ITEM_NOTE,
        "blindness_invariant_reading": dict(BLINDNESS_INVARIANT_READING),
    }


def replay_published_assignment(assignment_inputs: Mapping[str, Any]) -> dict[str, list[str]]:
    """Recompute the three v2 memberships from the published block and nothing else.

    The single argument is the artifact.  There is no candidate row, no
    database handle and no placement dict in scope, which is the property that
    makes the receipt worth anything: if this reproduces the shipped membership
    digests, then the shipped assignment used no input that is not printed in
    ``assignment_inputs``, and the blindness invariant can be checked against a
    finite list instead of against a promise.

    Returns the per-split sorted member lists, restricted to components that
    carry at least one row -- a zero-item component takes a placement but
    appears in no published per-split component list.
    """

    if not isinstance(assignment_inputs, Mapping):
        raise ColdStartError("assignment_inputs is not an object")
    constants = assignment_inputs.get("constants")
    components = assignment_inputs.get("components")
    if not isinstance(constants, Mapping) or not isinstance(components, list):
        raise ColdStartError("assignment_inputs is missing constants or components")
    seed = constants.get("seed")
    order = tuple(constants.get("split_names_order") or ())
    precedence = tuple(constants.get("role_precedence") or ())
    fractions = constants.get("target_item_fractions")
    if not isinstance(seed, str) or not order or not precedence:
        raise ColdStartError("assignment_inputs.constants is incomplete")
    if not isinstance(fractions, Mapping) or any(name not in fractions for name in order):
        raise ColdStartError("assignment_inputs.constants.target_item_fractions is incomplete")

    sizes: dict[str, int] = {}
    for entry in components:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("c"), str):
            raise ColdStartError("assignment_inputs.components carries a malformed entry")
        if entry["c"] in sizes:
            raise ColdStartError("assignment_inputs.components repeats a component digest")
        try:
            sizes[entry["c"]] = int(entry["n"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ColdStartError(
                f"component {entry['c'][:12]} publishes no usable item count"
            ) from exc

    # Step 1: inherit the most-read v1 role.
    placement: dict[str, str] = {}
    for entry in components:
        roles = entry.get("r") or ()
        if not roles:
            continue
        chosen = next((role for role in precedence if role in roles), None)
        if chosen is None:
            raise ColdStartError(
                f"component {entry['c'][:12]} carries a v1 role outside role_precedence"
            )
        placement[entry["c"]] = chosen
    # Step 2: the pre-registered observed-map rule overrides the carry-over.
    for entry in components:
        if entry.get("f"):
            placement[entry["c"]] = "eval"

    # Steps 3 and 4: targets over the whole population, and the load the
    # balancer inherited rather than chose.
    total = sum(sizes.values())
    if not total:
        raise ColdStartError("the published assignment inputs carry no items")
    targets = {name: float(fractions[name]) * total for name in order}
    if any(target <= 0.0 for target in targets.values()):
        raise ColdStartError("a published target item fraction is not positive")
    load = dict.fromkeys(order, 0)
    for component_id, name in placement.items():
        if name not in load:
            raise ColdStartError(f"placement produced the unknown split {name!r}")
        load[name] += sizes[component_id]

    # Steps 5 and 6: largest first, seeded digest breaking equal sizes, each
    # fresh component to the largest RELATIVE deficit.
    fresh = sorted(
        (component_id for component_id in sizes if component_id not in placement),
        key=lambda component: (-sizes[component], sha256_text(f"{seed}\0{component}")),
    )
    for component_id in fresh:
        chosen = max(order, key=lambda name: (targets[name] - load[name]) / targets[name])
        placement[component_id] = chosen
        load[chosen] += sizes[component_id]

    # Step 7: a component with no row is placed but published nowhere.
    return {
        name: sorted(
            component_id
            for component_id, where in placement.items()
            if where == name and sizes[component_id] > 0
        )
        for name in order
    }


def assignment_reproduction_receipt(
    assignment_inputs: Mapping[str, Any],
    published_membership_digests: Mapping[str, str],
) -> dict[str, Any]:
    """Replay the assignment, compare, and fail closed on any difference.

    The second argument is the three digests the manifest already publishes at
    ``coldstart.splits.{name}.membership_digest``; it is the thing being
    checked, not an input to the computation.  A mismatch means the shipped
    partition is not the partition the published inputs imply -- either an
    input is missing from the table or the assignment read something it did not
    declare -- and either way the receipt must not be written.
    """

    members = replay_published_assignment(assignment_inputs)
    sizes = {
        entry["c"]: int(entry["n"]) for entry in assignment_inputs.get("components", [])
    }
    digests = {name: _membership_digest(members[name]) for name in members}
    drift = sorted(
        name
        for name in digests
        if digests[name] != published_membership_digests.get(name)
    )
    if drift:
        raise ColdStartError(
            "the assignment did not reproduce from its published inputs, so the published "
            "input list is not the assignment's real input list: "
            + ", ".join(drift)
        )
    return {
        "function": "scripts/recall_map_coldstart_dataset.py:replay_published_assignment",
        "recomputed_from": "split.assignment_inputs, and nothing else",
        "checked_against": "coldstart.splits.{train,eval,holdout}.membership_digest",
        "reproduces_published_v2": True,
        "membership_digests": digests,
        "components": {name: len(members[name]) for name in members},
        "items": {
            name: sum(sizes[component_id] for component_id in members[name])
            for name in members
        },
        "inputs_used": [entry["name"] for entry in DECLARED_ASSIGNMENT_INPUTS],
        "procedure": list(ASSIGNMENT_REPRODUCTION_PROCEDURE),
        "why_the_deficit_is_relative": (
            "the published contract is a set of item FRACTIONS, and the carry-over hands the "
            "balancer a load it did not choose; equalising absolute shortfall would hit the "
            "smallest split as hard as the largest"
        ),
        "fails_closed": (
            "assignment_reproduction_receipt raises ColdStartError if any recomputed digest "
            "differs, so a run that cannot reproduce its own assignment writes no artifact"
        ),
        "re_executable_by": (
            "python3 scripts/recall_map_coldstart_dataset.py verify-assignment "
            "--manifest artifacts/recall-map/relevance/dataset-manifest.json, which opens "
            "the manifest and no other data source"
        ),
    }


def assign_three_way_splits(
    candidates: Sequence[Candidate],
    identities: Mapping[str, EventIdentity],
    *,
    component_event_ids: Iterable[str],
    v1_role_by_event: Mapping[str, str],
    observed_map_event_ids: Iterable[str] = (),
    seed: str = SPLIT_SEED,
) -> dict[str, Any]:
    """Partition the components three ways and report how the split behaves."""

    component_by_event = build_components(
        candidates, identities, component_event_ids=component_event_ids
    )
    components = sorted(set(component_by_event.values()))
    sizes = Counter(candidate.component_id for candidate in candidates)
    for component_id in components:
        sizes.setdefault(component_id, 0)

    forced_eval_components = sorted(
        {
            component_by_event[event_id]
            for event_id in observed_map_event_ids
            if event_id in component_by_event
        }
    )
    carried = carry_over_assignment(
        sizes,
        component_by_event,
        v1_role_by_event=v1_role_by_event,
        forced_eval_components=forced_eval_components,
        seed=seed,
    )
    placement = carried["placement"]
    for candidate in candidates:
        candidate.split = placement[candidate.component_id]

    # The same arguments the call above was given, published exhaustively so
    # the blindness invariant can be checked against a finite list.  Built here
    # rather than in the block assembler because ``sizes`` is the assignment's
    # actual argument and a reconstruction of it downstream would be a second
    # implementation of the thing under audit.
    assignment_inputs = assignment_inputs_block(
        sizes,
        component_by_event,
        v1_role_by_event=v1_role_by_event,
        forced_eval_components=forced_eval_components,
        seed=seed,
    )

    uniform = _uniform_draw_assignment(components, seed=seed)
    diagnostic: dict[str, Any] = {}
    for name in SPLIT_NAMES:
        rows = [row for row in candidates if uniform[row.component_id] == name]
        diagnostic[name] = {
            "items": len(rows),
            "item_share": _ratio(len(rows), len(candidates)),
            **_concentration(rows),
        }
    # What the obvious rule would have cost: how much already-read v1 work the
    # per-component draw would have deposited in the sealed holdout.
    read_events = {
        event_id
        for event_id, role in v1_role_by_event.items()
        if role in ("train", "eval") and event_id in component_by_event
    }
    read_components = {component_by_event[event_id] for event_id in read_events}
    uniform_leaked = sorted(
        component_id for component_id in read_components if uniform.get(component_id) == "holdout"
    )
    diagnostic["already_read_components_it_would_put_in_holdout"] = {
        "components": len(uniform_leaked),
        "items": sum(int(sizes[component_id]) for component_id in uniform_leaked),
    }

    # The property the sealed partition's whole value rests on, measured on the
    # published placement rather than argued from the construction.
    holdout_events = {
        event_id
        for event_id, component_id in component_by_event.items()
        if placement[component_id] == "holdout"
    }
    leaked_events = sorted(read_events & holdout_events)
    if leaked_events:
        raise ColdStartError(
            f"{len(leaked_events)} already-read v1 train/eval event(s) landed in the "
            "enlarged holdout; the carry-over failed and the sealed partition would be "
            "destroyed"
        )
    return {
        "placement": placement,
        "components": components,
        "component_by_event": component_by_event,
        "carry_over": carried,
        "assignment_inputs": assignment_inputs,
        "uniform_draw_diagnostic": diagnostic,
        "holdout_contamination": {
            "assertion": (
                "the intersection of v2 holdout event ids with the union of v1 train and "
                "v1 eval event ids is empty"
            ),
            "v1_read_events_in_scope": len(read_events),
            "v2_holdout_events": len(holdout_events),
            "intersection": len(leaked_events),
            "holds": True,
        },
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _two_proportion_z(
    consumed_a: int, total_a: int, consumed_b: int, total_b: int
) -> float | None:
    if not total_a or not total_b:
        return None
    pooled = (consumed_a + consumed_b) / (total_a + total_b)
    variance = pooled * (1.0 - pooled) * (1.0 / total_a + 1.0 / total_b)
    if variance <= 0.0:
        return None
    return round(
        (consumed_a / total_a - consumed_b / total_b) / math.sqrt(variance), 4
    )


def _family_unscoreable(rows: Sequence[Candidate]) -> int:
    """Rows this feature family cannot be fitted or calibrated on.

    ``coldstart-prereg.json#missing_value_rule.cause_b_historically_unrecorded``:
    a row reconstructed from persisted state whose delivery envelope never
    recorded the scores is excluded from the family's fit, its direction checks
    and its threshold calibration — and is *counted and reported*, never
    silently dropped.  ``SCORE_FIELDS`` is exactly the family: the four core
    members plus the conditional ``graph_score``.
    """

    return sum(
        1
        for row in rows
        if any(row.scores.get(name) is None for name in SCORE_FIELDS)
    )


def _era(row: Candidate) -> str:
    """Which side of the enlargement this row came from."""

    return ERA_PREEXISTING if row.decided_at >= V1_CANDIDATE_WINDOW_START else ERA_ENLARGEMENT


def _rank_band(rank: int) -> str:
    for label, low, high in RANK_BANDS:
        if rank >= low and (high is None or rank <= high):
            return label
    return RANK_BANDS[-1][0]


def _stratified(rows: Sequence[Candidate]) -> dict[str, Any]:
    """The same numbers again, cut by what the enlargement actually added.

    Widening backwards crosses corpus regimes and lifting the per-event cap
    reaches ranks the v1 population never contained.  Reporting only the totals
    would ask a later reader to trust that the mixture was harmless; these cuts
    let them check it instead.
    """

    by_rank: dict[str, Any] = {}
    for label, _low, _high in RANK_BANDS:
        band = [row for row in rows if _rank_band(row.rank) == label]
        if band:
            by_rank[label] = _cohort(band, include_concentration=True)
    return {
        "by_era": {
            name: _cohort(
                [row for row in rows if _era(row) == name], include_concentration=True
            )
            for name in ERA_NAMES
        },
        "by_rank_band": by_rank,
    }


def _cohort(
    rows: Sequence[Candidate],
    *,
    include_components: bool = False,
    include_concentration: bool = False,
) -> dict[str, Any]:
    cold = [row for row in rows if row.cold]
    warm = [row for row in rows if not row.cold]
    consumed = sum(1 for row in rows if row.consumed)
    cold_consumed = sum(1 for row in cold if row.consumed)
    warm_consumed = sum(1 for row in warm if row.consumed)
    summary: dict[str, Any] = {
        "items": len(rows),
        "consumed": consumed,
        "rate": _ratio(consumed, len(rows)),
        "events": len({row.event_id for row in rows}),
        "distinct_nodes": len({row.node_id for row in rows}),
        "transport_sessions": len(
            {row.transport_session_id for row in rows if row.transport_session_id}
        ),
        "cold_candidates": len(cold),
        "cold_candidate_share": _ratio(len(cold), len(rows)),
        "cold_consumed": cold_consumed,
        "cold_rate": _ratio(cold_consumed, len(cold)),
        # ``coldstart-prereg.json#split_protocol.per_split_reporting`` names this
        # one cold_base_rate; it is the same quantity as cold_rate, published
        # under the registered name so the fit can cite it without a mapping.
        "cold_base_rate": _ratio(cold_consumed, len(cold)),
        "family_unscoreable": _family_unscoreable(rows),
        "warm_candidates": len(warm),
        "warm_consumed": warm_consumed,
        "warm_rate": _ratio(warm_consumed, len(warm)),
        "cold_minus_warm_z": _two_proportion_z(
            cold_consumed, len(cold), warm_consumed, len(warm)
        ),
        "levels": {
            name: sum(1 for row in rows if row.level == name) for name in LEVELS
        },
    }
    if include_components:
        members = sorted({row.component_id for row in rows})
        summary["components"] = members
        summary["membership_digest"] = _membership_digest(members)
    if include_components or include_concentration:
        summary["concentration"] = _concentration(rows)
    return summary


def _score_availability(rows: Sequence[Candidate]) -> dict[str, dict[str, Any]]:
    """How much candidate-time evidence a cold row actually owns.

    This is the load-bearing number for the fit that follows: if the recorded
    retrieval scores were missing on cold rows, no cold-start feature family
    could be built from them at all.
    """

    cold = [row for row in rows if row.cold]
    out: dict[str, dict[str, Any]] = {}
    for name in SCORE_FIELDS:
        present = sum(1 for row in rows if row.scores.get(name) is not None)
        cold_present = sum(1 for row in cold if row.scores.get(name) is not None)
        out[name] = {
            "available": present,
            "available_share": _ratio(present, len(rows)),
            "cold_available": cold_present,
            "cold_available_share": _ratio(cold_present, len(cold)),
        }
    aged = sum(1 for row in rows if row.node_age_log_days is not None)
    out["node_age_log_days"] = {
        "available": aged,
        "available_share": _ratio(aged, len(rows)),
        "cold_available": sum(
            1 for row in cold if row.node_age_log_days is not None
        ),
        "cold_available_share": _ratio(
            sum(1 for row in cold if row.node_age_log_days is not None), len(cold)
        ),
    }
    return out


# ---------------------------------------------------------------------------
# Independence accounting
#
# The enlargement bought cold candidates and paid in independent units, and the
# manifest already published the price at enlargement.concentration_cost.  What
# it never did was connect that price to the sign failure the fit then recorded.
# These four blocks are that connection, and they are deliberately split by what
# they are allowed to touch:
#
#   * component_size_distribution, cap_feasibility and component_equalized are
#     STRUCTURAL.  They read component membership and the cold flag -- a
#     history-availability fact, already published per split -- and no
#     consumption label and no family-member value at all.  They are therefore
#     computed for all three splits.
#   * train_direction_accounting is LABEL-BEARING and reads family-member
#     values.  It is computed on coldstart_train and nothing else.
# ---------------------------------------------------------------------------


def _cap_order_key(row: Candidate) -> str:
    """Which of a component's rows a cap keeps, from label-free fields only.

    Not ``decided_at`` and not ``rank``: an earliest-first or a shallowest-first
    prefix of a component that spans months would be a systematically different
    cohort from the component itself, and the whole point of the cap is to keep
    the surviving rows representative of the unit they stand for.
    """

    return sha256_text(
        canonical_json(
            [CAP_SUBSAMPLE_NAMESPACE, row.component_id, row.source, row.event_id, row.rank]
        )
    )


def cap_survivors(rows: Sequence[Candidate], cap: int) -> list[Candidate]:
    """The rows that survive an intra-split cap of ``cap`` items per component.

    ``cap`` is applied INSIDE the split each component was already assigned to.
    No component is divided across splits by this and none can be: the split a
    row belongs to is not consulted, only how many of its component's rows have
    already been kept.
    """

    if cap <= 0:
        raise ColdStartError(f"cap must be positive, got {cap}")
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for row in rows:
        grouped[row.component_id].append(row)
    kept: list[Candidate] = []
    for component_id in sorted(grouped):
        members = sorted(grouped[component_id], key=_cap_order_key)
        kept.extend(members[:cap])
    return kept


def _equalized_weights(rows: Sequence[Candidate]) -> list[float]:
    """Equal total weight per component, shared equally inside it.

    The no-rows-dropped alternative to a cap: every row still contributes, but
    a component of 7384 rows and a component of 4 rows now count the same.
    """

    sizes = Counter(row.component_id for row in rows)
    count = len(sizes)
    if not count:
        return []
    return [1.0 / (count * sizes[row.component_id]) for row in rows]


def _effective_sample_size(weights: Sequence[float]) -> float | None:
    """Kish's effective sample size, ``(sum w)^2 / sum w^2``.

    For uniform weights this is exactly the row count, so the unweighted and
    the equalized arms are read off one scale.
    """

    total = sum(weights)
    squares = sum(weight * weight for weight in weights)
    if not squares:
        return None
    return round(total * total / squares, 2)


def component_size_distribution(rows: Sequence[Candidate]) -> dict[str, Any]:
    """Item 1: how a split's items are actually distributed over its components.

    ``effective_components`` alone says a split is worth ~3 units; it does not
    say whether that is one giant and a long tail or three equals.  The sorted
    size list does, and the parallel cold count says whether the cold
    candidates -- the rows the whole recalibration is about -- sit in the giant
    or in the tail.  Pairs are ordered by size, NOT by component digest, so
    they add no link between a component's identity and its contents beyond
    what ``splits.*.components`` already publishes.
    """

    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for row in rows:
        grouped[row.component_id].append(row)
    pairs = sorted(
        (
            [len(members), sum(1 for member in members if member.cold)]
            for members in grouped.values()
        ),
        key=lambda pair: (-pair[0], -pair[1]),
    )
    sizes = [pair[0] for pair in pairs]
    total = len(rows)
    middle = len(sizes) // 2
    return {
        "components": len(sizes),
        "items": total,
        "cold_candidates": sum(pair[1] for pair in pairs),
        "effective_components": _concentration(rows).get("effective_components"),
        "largest_component_items": sizes[0] if sizes else 0,
        "largest_component_share": _ratio(sizes[0], total) if sizes else None,
        "median_component_items": sizes[middle] if sizes else 0,
        "smallest_component_items": sizes[-1] if sizes else 0,
        "components_larger_than_cap": {
            str(cap): sum(1 for size in sizes if size > cap) for cap in CAP_GRID
        },
        "items_above_cap": {
            str(cap): sum(size - cap for size in sizes if size > cap) for cap in CAP_GRID
        },
        "size_and_cold_by_size_desc": pairs,
        "reading": [
            "Each pair is [items, cold_candidates] for one component, ordered by",
            "size. effective_components is the inverse Simpson index over the",
            "item shares -- the number of equally sized components this split is",
            "worth. Read it, not components.",
        ],
    }


def cap_feasibility(by_split: Mapping[str, Sequence[Candidate]]) -> dict[str, Any]:
    """Item 2: does a cap exist that buys independence and keeps the floor?

    Structural and label-free, so it is computed for all three splits.  It
    answers one question and states the answer rather than leaving it to be
    reconstructed: at which caps do all three splits still clear the
    pre-registered 400 cold candidates, and how much larger is the effective
    sample there.
    """

    table: dict[str, Any] = {}
    clearing: list[int] = []
    for cap in CAP_GRID:
        per_split: dict[str, Any] = {}
        for name in SPLIT_NAMES:
            kept = cap_survivors(by_split[name], cap)
            cold = sum(1 for row in kept if row.cold)
            concentration = _concentration(kept)
            per_split[name] = {
                "items": len(kept),
                "items_dropped": len(by_split[name]) - len(kept),
                "items_retained_share": _ratio(len(kept), len(by_split[name])),
                "cold_candidates": cold,
                "cold_candidate_share": _ratio(cold, len(kept)),
                "components": concentration["components"],
                "effective_components": concentration.get("effective_components"),
                "meets_the_cold_minimum": cold >= MINIMUM_COLD_CANDIDATES_PER_SPLIT,
            }
        every = all(entry["meets_the_cold_minimum"] for entry in per_split.values())
        if every:
            clearing.append(cap)
        effective = [
            entry["effective_components"]
            for entry in per_split.values()
            if entry["effective_components"] is not None
        ]
        table[str(cap)] = {
            **per_split,
            "all_splits_meet_the_cold_minimum": every,
            "smallest_effective_components": min(effective) if effective else None,
        }
    baseline = {
        name: _concentration(rows).get("effective_components")
        for name, rows in by_split.items()
    }
    return {
        "caps": list(CAP_GRID),
        "minimum_cold_candidates_per_split": MINIMUM_COLD_CANDIDATES_PER_SPLIT,
        "minimum_cold_candidates_per_split_source": (
            "coldstart-prereg.json#split_protocol.minimum_cold_candidates_per_split"
        ),
        "uncapped_effective_components": baseline,
        "rule": {
            "what_a_cap_does": [
                "Use at most CAP of a component's rows for estimation, inside the",
                "split that component was already assigned to. No component is",
                "divided across splits by this, so the disjointness guarantee is",
                "untouched -- see enlargement.concentration_cost",
                ".an_intra_split_cap_does_not_weaken_disjointness.",
            ],
            "which_rows_survive": [
                "sha256 order over [namespace, component_id, source, event_id, rank].",
                "Label-free by construction, so the draw cannot select on the",
                "consumption bit, on the cold flag, or on any family member value.",
            ],
            "caps_nest": [
                "The order does not depend on the cap, so the survivors at 25 are a",
                "subset of the survivors at 50, and so on up the grid.",
            ],
            "namespace": CAP_SUBSAMPLE_NAMESPACE,
        },
        "table": table,
        "caps_clearing_the_cold_minimum_on_every_split": clearing,
        "largest_cap_clearing_the_cold_minimum": max(clearing) if clearing else None,
        "smallest_cap_clearing_the_cold_minimum": min(clearing) if clearing else None,
        "reading": [
            "A cap costs rows -- items_dropped is the price and it is real. What it",
            "buys is effective_components: compare each cell against",
            "uncapped_effective_components for the same split.",
            "caps_clearing_the_cold_minimum_on_every_split is the feasible set. It is",
            "reported so a successor plan can register a cap on a measurement instead",
            "of choosing one and hoping the floor survives.",
        ],
    }


def component_equalized(by_split: Mapping[str, Sequence[Candidate]]) -> dict[str, Any]:
    """Item 3: the same accounting for the alternative that drops no rows.

    Under equal per-component weight every row still contributes; what changes
    is that a component of 7384 rows no longer outvotes a component of 4 by a
    factor of 1846.  ``effective_sample_size`` is Kish's, which for uniform
    weights is exactly the row count -- so the capped and the equalized arms
    are readable against the uncapped one without a conversion.
    """

    per_split: dict[str, Any] = {}
    for name in SPLIT_NAMES:
        rows = list(by_split[name])
        weights = _equalized_weights(rows)
        cold_rows = [row for row in rows if row.cold]
        cold_weights = _equalized_weights(cold_rows)
        effective = _effective_sample_size(weights)
        per_split[name] = {
            "items": len(rows),
            "components": len({row.component_id for row in rows}),
            "effective_sample_size": effective,
            "effective_sample_share_of_items": _ratio_float(effective, len(rows)),
            "cold_candidates": len(cold_rows),
            "components_carrying_cold_candidates": len(
                {row.component_id for row in cold_rows}
            ),
            "cold_effective_sample_size": _effective_sample_size(cold_weights),
            "rows_dropped": 0,
        }
    return {
        "weight_rule": [
            "w_i = 1 / (C * n_c), where C is the split's component count and n_c",
            "is the item count of row i's component. Every component therefore",
            "carries total weight 1/C, and every row inside one carries the same",
            "share of it.",
        ],
        "effective_sample_size_definition": [
            "Kish, (sum w)^2 / sum w^2. It equals the row count exactly when the",
            "weights are uniform, so this scale IS the uncapped item count's",
            "scale and the three arms can be read against each other directly.",
        ],
        "no_rows_dropped": True,
        "per_split": per_split,
        "reading": [
            "This is the no-rows-dropped alternative to a cap. It keeps every cold",
            "candidate -- so the 400-per-split floor is untouched by construction --",
            "and pays instead in variance: effective_sample_size is what the split is",
            "worth once one component can no longer outvote the rest.",
            "cold_effective_sample_size is the same figure computed over the cold",
            "subpopulation alone, which is the cohort the direction clause binds on.",
        ],
    }


def _ratio_float(numerator: float | None, denominator: float | int) -> float | None:
    if numerator is None or not denominator:
        return None
    return round(numerator / denominator, 6)


def _weighted_difference(
    rows: Sequence[Candidate],
    weights: Sequence[float],
    values: Sequence[float | None],
) -> float | None:
    """Consumed-minus-nonconsumed weighted mean difference on ``values``.

    With uniform weights this is exactly
    ``recall_map_coldstart_fit.association``'s plain mean difference, which is
    what lets the unweighted arm reproduce the published step-3 numbers.
    """

    consumed_weight = consumed_sum = other_weight = other_sum = 0.0
    for row, weight, value in zip(rows, weights, values, strict=True):
        if value is None:
            continue
        if row.consumed:
            consumed_weight += weight
            consumed_sum += weight * value
        else:
            other_weight += weight
            other_sum += weight * value
    if not consumed_weight or not other_weight:
        return None
    return consumed_sum / consumed_weight - other_sum / other_weight


def _direction_arm(
    rows: Sequence[Candidate],
    weights: Sequence[float] | None,
    u_of: Mapping[str, Any],
) -> dict[str, Any]:
    """One arm: the four members' directions on a cohort, under one weighting."""

    if weights is None:
        weights = [1.0] * len(rows)
    cold_mask = [row.cold for row in rows]
    cold_rows = [row for row, is_cold in zip(rows, cold_mask, strict=True) if is_cold]
    cold_weights = [
        weight for weight, is_cold in zip(weights, cold_mask, strict=True) if is_cold
    ]
    cold_consumed = sum(1 for row in cold_rows if row.consumed)
    measurable = (
        len(cold_rows) >= MINIMUM_COLD_ROWS_FOR_A_DIRECTION_CHECK
        and cold_consumed >= MINIMUM_CONSUMED_COLD_ROWS_FOR_A_DIRECTION_CHECK
    )
    members: dict[str, Any] = {}
    for source, member, sign in FAMILY_DIRECTIONS:
        table = u_of[source]
        values = [table(row) for row in rows]
        cold_values = [
            value for value, is_cold in zip(values, cold_mask, strict=True) if is_cold
        ]
        whole = _weighted_difference(rows, weights, values)
        cold = _weighted_difference(cold_rows, cold_weights, cold_values)
        members[member] = {
            "registered_direction": "positive" if sign > 0 else "negative",
            "whole_tail": _round6(whole),
            "cold": _round6(cold),
            "whole_tail_matches_registered": None if whole is None else whole * sign > 0,
            "cold_matches_registered": None if cold is None else cold * sign > 0,
        }
    return {
        "rows": len(rows),
        "consumed": sum(1 for row in rows if row.consumed),
        "cold_rows": len(cold_rows),
        "cold_consumed": cold_consumed,
        "cold_check_measurable": measurable,
        "effective_sample_size": _effective_sample_size(weights),
        "cold_effective_sample_size": _effective_sample_size(cold_weights),
        "components": len({row.component_id for row in rows}),
        "members": members,
    }


def _round6(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def train_direction_accounting(train_rows: Sequence[Candidate]) -> dict[str, Any]:
    """Item 4: the four members' directions on ``coldstart_train``, every way.

    This is the label-bearing half of the accounting and it is confined to one
    split on purpose.  The v2 plan's binding direction test was itself run on
    train, which is what made its failure diagnosable here at all; a successor
    plan will move that test to ``coldstart_eval`` and ``secondary_anchor``,
    and publishing a family-member statistic on eval now would spend a test
    that has not been registered yet.

    Every arm evaluates the SAME ``u`` axis: the per-member step function
    fitted once on all of train, exactly as ``recall_map_coldstart_fit
    .fit_tables`` fits it.  Refitting the table inside each cap would change
    two things at once and make the arms incomparable both to each other and
    to the step-3 numbers they exist to explain.  What varies across arms is
    only which rows the estimator counts, and how much.
    """

    module = calibration()
    rows = list(train_rows)
    shims = [{"scores": row.scores} for row in rows]

    tables: dict[str, Any] = {}
    for source, member, _sign in FAMILY_DIRECTIONS:
        fitted = [
            value
            for value in (module.member_raw(shim, source) for shim in shims)
            if value is not None
        ]
        tables[source] = module.StepFunction.fit(member, fitted)

    # Precomputed once: u(v) per row per member, keyed by the row's identity in
    # this list, so every arm below is a selection over the same values rather
    # than a re-evaluation that could disagree with another arm.
    cache: dict[str, dict[int, float | None]] = {}
    for source, _member, _sign in FAMILY_DIRECTIONS:
        table = tables[source]
        per_row: dict[int, float | None] = {}
        for index, shim in enumerate(shims):
            raw = module.member_raw(shim, source)
            per_row[id(rows[index])] = None if raw is None else table.u(raw)
        cache[source] = per_row
    u_of = {
        source: (lambda row, _map=cache[source]: _map[id(row)])
        for source, _member, _sign in FAMILY_DIRECTIONS
    }

    unweighted = _direction_arm(rows, None, u_of)
    by_cap = {
        str(cap): _direction_arm(cap_survivors(rows, cap), None, u_of) for cap in CAP_GRID
    }
    equalized = _direction_arm(rows, _equalized_weights(rows), u_of)
    by_era = {
        name: _direction_arm([row for row in rows if _era(row) == name], None, u_of)
        for name in ERA_NAMES
    }

    # Which arms agree with the sign v2 registered, per member.  Purely
    # mechanical -- a fold over the arms above, asserting nothing they do not
    # already say -- but it is the shape a successor plan actually needs, and
    # reconstructing it by eye across eleven tables is how a disagreement gets
    # missed.  It registers no direction: that is coldstart-prereg-v3's job.
    labelled_arms: list[tuple[str, Mapping[str, Any]]] = [("unweighted", unweighted)]
    labelled_arms.extend((f"cap_{cap}", by_cap[str(cap)]) for cap in CAP_GRID)
    labelled_arms.append(("component_equalized", equalized))
    labelled_arms.extend((f"era_{name}", by_era[name]) for name in ERA_NAMES)
    component_aware = [f"cap_{cap}" for cap in CAP_GRID] + ["component_equalized"]
    stability: dict[str, Any] = {}
    for _source, member, _sign in FAMILY_DIRECTIONS:
        per_arm = {
            label: {
                "whole_tail": arm["members"][member]["whole_tail_matches_registered"],
                "cold": arm["members"][member]["cold_matches_registered"],
            }
            for label, arm in labelled_arms
        }
        disagreeing_arms = sorted(
            label
            for label, verdicts in per_arm.items()
            if verdicts["whole_tail"] is False or verdicts["cold"] is False
        )
        stability[member] = {
            "registered_direction": unweighted["members"][member]["registered_direction"],
            "matches_registered_by_arm": per_arm,
            "arms_disagreeing_with_the_registered_direction": disagreeing_arms,
            "stable_across_every_arm": not disagreeing_arms,
            "stable_across_every_component_aware_arm": all(
                per_arm[label]["whole_tail"] is not False
                and per_arm[label]["cold"] is not False
                for label in component_aware
            ),
            "the_two_eras_agree": all(
                per_arm[f"era_{ERA_NAMES[0]}"][key] == per_arm[f"era_{ERA_NAMES[1]}"][key]
                for key in ("whole_tail", "cold")
            ),
        }

    published = {
        member: {
            "whole_tail": unweighted["members"][member]["whole_tail"],
            "cold": unweighted["members"][member]["cold"],
            "published_whole_tail": PUBLISHED_STEP3_RANK_UNIFORM[member]["whole_tail"],
            "published_cold": PUBLISHED_STEP3_RANK_UNIFORM[member]["cold"],
        }
        for _source, member, _sign in FAMILY_DIRECTIONS
    }
    disagreeing = sorted(
        member
        for member, entry in published.items()
        if entry["whole_tail"] != entry["published_whole_tail"]
        or entry["cold"] != entry["published_cold"]
    )
    if disagreeing:
        raise ColdStartError(
            "the unweighted direction arm does not reproduce the step-3 numbers the v2 "
            f"fit halted on, for {', '.join(disagreeing)}. Either the population moved "
            "or the u axis did; update PUBLISHED_STEP3_RANK_UNIFORM deliberately and "
            "say why, do not let this accounting claim to explain a different HALT."
        )

    return {
        "split": DIRECTION_ACCOUNTING_SPLIT,
        "splits_measured": [DIRECTION_ACCOUNTING_SPLIT],
        "why_only_train": [
            "This is the only label-bearing block in the accounting and the only one",
            "that reads a family member's value. coldstart_eval and coldstart_holdout",
            "carry the falsifiable direction test a successor plan still has to",
            "register; measuring the family on either of them here would spend that",
            "test before it exists. coldstart_holdout is additionally sealed and is",
            "not read at all.",
        ],
        "statistic": (
            "consumed-minus-nonconsumed mean difference on rank_uniform_centered "
            "values, weighted where the arm says so"
        ),
        "u_axis": {
            "definition": "(A + M/2)/n - 0.5, midrank on ties",
            "fitted_on": "all of coldstart_train, once",
            "fitted_by": "recall_map_coldstart_fit.StepFunction.fit",
            "held_fixed_across_arms": True,
            "why_held_fixed": [
                "Refitting the table inside each cap would move the value axis and",
                "the row set together, leaving the arms comparable to nothing --",
                "not to each other, and not to the published step-3 reading.",
                "What varies across arms here is only which rows the estimator",
                "counts, and how much each one counts for.",
            ],
            "fitted_rows": {
                member: tables[source].n for source, member, _sign in FAMILY_DIRECTIONS
            },
        },
        "cold_subpopulation_clause": {
            "source": (
                "coldstart-prereg-v2.json#direction_constraint.cold_subpopulation_clause"
            ),
            "minimum_cold_rows": MINIMUM_COLD_ROWS_FOR_A_DIRECTION_CHECK,
            "minimum_consumed_cold_rows": (
                MINIMUM_CONSUMED_COLD_ROWS_FOR_A_DIRECTION_CHECK
            ),
            "reading": [
                "Every arm publishes cold_rows and cold_consumed beside its numbers,",
                "so whether that arm's cold reading is measurable at all is checkable",
                "rather than assumed. An arm below the minima is INDETERMINATE, which",
                "the v2 plan makes blocking rather than passing.",
            ],
        },
        "unweighted": unweighted,
        "by_cap": by_cap,
        "component_equalized": equalized,
        "by_era": by_era,
        "direction_stability": {
            "what_this_is": [
                "A fold over the arms above: for each member, which arms agree with",
                "the direction v2 registered. It asserts nothing the arms do not",
                "already say. It registers no direction and selects no estimator --",
                "both are coldstart-prereg-v3's to do, from these numbers.",
                "A member whose entry is stable_across_every_component_aware_arm but",
                "not stable_across_every_arm failed v2 because of the pooling, not",
                "because it lacks signal. A member where the_two_eras_agree is false",
                "is a member no single pooled reading describes.",
            ],
            "component_aware_arms": component_aware,
            "by_member": stability,
        },
        "reproduces_published_step3": {
            "source": (
                "policy.json#coldstart_revision.decision_procedure[2].report.members"
            ),
            "agrees": True,
            "compared": published,
            "why_it_is_checked": [
                "The unweighted arm IS the published reading. If it did not reproduce",
                "it exactly, the arms beside it would be explaining some other run,",
                "and build_coldstart_block refuses to write a block where it does not.",
            ],
        },
        "reading": [
            "The unweighted row is the v2 step-3 reading, reproduced. Everything",
            "beside it is the same statistic with the estimator made component-aware,",
            "and the point is how little of train is left underneath: read",
            "effective_sample_size on each arm, not rows.",
            "by_era is the second cut, and it is not a weighting at all -- it asks",
            "whether the two eras even agree on a member's sign, which a single",
            "pooled number cannot show.",
            "No sign here is registered and no threshold is selected. Registering",
            "directions from this measurement is coldstart-prereg-v3's job, and it",
            "must also record that a direction taken from train makes the train",
            "direction check a tautology rather than a test.",
        ],
    }


def build_coldstart_block(extracted: Mapping[str, Any]) -> dict[str, Any]:
    candidates: list[Candidate] = list(extracted["candidates"])
    identities: Mapping[str, EventIdentity] = extracted["identities"]
    consumer_index = extracted["consumer_index"]
    bindings = extracted["bindings"]
    protocol = dict(extracted["protocol"])
    rank_coverage = dict(extracted["rank_coverage"])

    opportunity = [row for row in candidates if row.opportunity]
    if not opportunity:
        raise ColdStartError("no opportunity-bearing residual candidate survived")

    component_event_ids = {
        event.id
        for event in extracted["window_events"]
        if bool(consumer_index.qualifying(event))
    }
    # Reconstruct what was already read BEFORE building anything wider.
    v1_component_event_ids = {
        event.id
        for event in extracted["v1_window_events"]
        if bool(consumer_index.qualifying(event))
    }
    v1 = reconstruct_v1_partition(
        extracted["v1_candidates"],
        identities,
        component_event_ids=v1_component_event_ids,
    )
    v1["receipt"]["window"] = list(protocol["v1_candidate_window"])

    observed_map_event_ids = set(extracted.get("observed_map_event_ids") or ())
    split_result = assign_three_way_splits(
        opportunity,
        identities,
        component_event_ids=component_event_ids,
        v1_role_by_event=v1["role_by_event"],
        observed_map_event_ids=observed_map_event_ids,
    )
    components = split_result["components"]
    carried = split_result["carry_over"]

    by_split = {name: [row for row in opportunity if row.split == name] for name in SPLIT_NAMES}
    empty = [name for name, rows in by_split.items() if not rows]
    if empty:
        raise ColdStartError(f"split(s) came back empty: {', '.join(empty)}")

    component_sets = {
        name: set(_cohort(rows, include_components=True)["components"])
        for name, rows in by_split.items()
    }
    pairwise = {
        f"{a}_{b}": len(component_sets[a] & component_sets[b])
        for index, a in enumerate(SPLIT_NAMES)
        for b in SPLIT_NAMES[index + 1 :]
    }
    if any(pairwise.values()):
        raise ColdStartError(f"split components overlap: {pairwise}")

    splits = {name: _cohort(rows, include_components=True) for name, rows in by_split.items()}
    for name, summary in splits.items():
        rows = by_split[name]
        summary["stratified"] = _stratified(rows)
        summary["component_sizes"] = component_size_distribution(rows)
        summary["assignment_reasons"] = {
            why: sum(
                1
                for component_id in summary["components"]
                if carried["reason"].get(component_id) == why
            )
            for why in (
                "inherited_most_read_v1_role",
                "touches_observed_map_delivery",
                "freshly_assigned",
            )
        }
        # A forced component is not a sampling accident, so the
        # one-correlated-cluster bar is measured on the material that WAS drawn
        # freely.  Both numbers are published; neither is suppressed.
        forced_here = {
            component_id
            for component_id in summary["components"]
            if carried["reason"].get(component_id) == "touches_observed_map_delivery"
        }
        free_rows = [row for row in rows if row.component_id not in forced_here]
        free_sizes = Counter(row.component_id for row in free_rows)
        largest_free = max(free_sizes.values()) if free_sizes else 0
        summary["concentration"].update(
            {
                "forced_components": len(forced_here),
                "forced_component_items": len(rows) - len(free_rows),
                "largest_freely_assigned_component_items": largest_free,
                "largest_freely_assigned_component_share": _ratio(largest_free, len(rows)),
                "effective_components_excluding_forced": _concentration(free_rows).get(
                    "effective_components"
                ),
            }
        )
    zero_cold = [name for name, summary in splits.items() if not summary["cold_candidates"]]
    starved = sorted(
        name
        for name, summary in splits.items()
        if int(summary["cold_candidates"]) < MINIMUM_COLD_CANDIDATES_PER_SPLIT
    )
    if starved:
        raise ColdStartError(
            "split(s) below the pre-registered minimum of "
            f"{MINIMUM_COLD_CANDIDATES_PER_SPLIT} cold candidates: "
            + ", ".join(f"{name}={splits[name]['cold_candidates']}" for name in starved)
        )

    # The blindness-invariant receipt, computed BEFORE the block exists so a
    # run whose assignment does not reproduce from its own published inputs
    # writes nothing at all.
    assignment_inputs = split_result["assignment_inputs"]
    assignment_reproduction = assignment_reproduction_receipt(
        assignment_inputs,
        {name: splits[name]["membership_digest"] for name in SPLIT_NAMES},
    )

    cold = [row for row in opportunity if row.cold]
    warm = [row for row in opportunity if not row.cold]
    all_cold = [row for row in candidates if row.cold]

    block: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "recall-map-coldstart-candidate-dataset",
        "as_of": COLDSTART_AS_OF,
        "extractor_sha256": _sha256_path(Path(__file__).resolve()),
        "sealed_inputs": {
            "historical_evaluator": {
                "logical_path": "scripts/recall_map_relevance_eval.py",
                "sha256": EXPECTED_EVALUATOR_SHA256,
            },
            "effect_tool": {
                "logical_path": "scripts/recall_map_effect.py",
                "sha256": bindings.effect_sha256,
            },
            "prereg": {
                "logical_path": "artifacts/recall-map/prereg.json",
                "sha256": bindings.prereg_sha256,
            },
            "plan_sha256": bindings.protocol.plan_sha256,
        },
        "sources": [dict(extracted["receipt"])],
        "maturation": {
            "snapshot_newest_event": extracted["snapshot_instant"],
            "as_of_plus_horizon": shift_hours(COLDSTART_AS_OF, protocol["horizon_hours"]),
            "every_window_in_scope_matured": True,
            "sealed_snapshot_reproducible": False,
            "why_new_snapshot": [
                "policy.json#bindings.source_snapshot pins a 529993728-byte local.sqlite3",
                "no file of that size or digest survives on this host, so the sealed",
                "run cannot be reproduced byte-for-byte; a fresh receipt is recorded here",
                "and no sealed evidence is recomputed against it.",
            ],
        },
        "protocol": {
            **protocol,
            "population": (
                "organic residual results at rank >= organic_head_cut, capped at "
                "organic_cap per event: what the pool sees after the head"
            ),
            "organic_cap_unchanged": True,
            "organic_cap_is_not_a_free_lever": {
                "claim": [
                    "raising organic_cap cannot enlarge this population.",
                    "recall_delivery_history records an outcome for 0-based ranks 3..8 and",
                    "for no other rank, so a rank-9 row carries no consumption label at all",
                    "and cannot enter a supervised dataset.",
                ],
                "measured_on_the_snapshot": rank_coverage,
                "consequence": (
                    "the enlargement is the window and only the window; the cap and the head "
                    "cut both stay at the sealed constants"
                ),
            },
            "opportunity_rule": (
                "frozen ConsumerIndex: a later event within horizon_hours sharing "
                "transport session, or failing that (scope, task)"
            ),
            "cold_rule": (
                "zero matured prior delivery windows at the decision instant, i.e. "
                "matured_recall_history M == 0"
            ),
            "outcome_source": "recall_delivery_history",
            "outcome_rule": (
                "transport_consumed when transport_matched, else fallback_consumed: "
                "the rule MemoryStore.matured_recall_history reads"
            ),
            "unit_of_analysis": (
                "the (delivering event, residual candidate) pair; the rate is "
                "consumed items over items"
            ),
            "mutable_node_stats_read": [],
        },
        "ledger": dict(extracted["ledger"]),
        "split": {
            "algorithm": (
                "connected_components(source-qualified cache key, transport session, session)"
            ),
            "seed": SPLIT_SEED,
            "partitions": list(SPLIT_NAMES),
            "assignment": "v1_role_carry_over_then_size_aware_largest_first_balanced",
            # coldstart-prereg-v2.json#split_protocol.assignment_rule
            # .blindness_invariant_binding: the name and the inputs, plus a
            # re-executable check that recomputes the assignment from those
            # inputs alone.  ``verify-assignment`` runs it off the artifact.
            "assignment_inputs": assignment_inputs,
            "assignment_reproduction": assignment_reproduction,
            "target_item_fractions": {
                "train": TRAIN_FRACTION,
                "eval": EVAL_FRACTION,
                "holdout": round(1.0 - TRAIN_FRACTION - EVAL_FRACTION, 8),
            },
            "components": len(components),
            "component_universe_events": len(component_event_ids),
            "pairwise_component_disjoint": True,
            "pairwise_component_intersections": pairwise,
            "identity_rule_matches_sealed_manifest": True,
            "components_never_divided": True,
            "uniform_draw_diagnostic": {
                **split_result["uniform_draw_diagnostic"],
                "rejected": True,
                "why": [
                    "A per-component coin flip is unbiased in components, not in items,",
                    "and these components differ in size by three orders of magnitude, so it",
                    "partitions items arbitrarily. Its measured concentration on this",
                    "population is published above, beside the split it would have produced.",
                    "It now fails for a second and decisive reason: it is computed from the",
                    "component digest alone, so on an enlarged population it re-draws every",
                    "component from scratch and deposits already-read v1 train and eval work",
                    "into the holdout. The count it would have leaked is published above at",
                    "already_read_components_it_would_put_in_holdout.",
                    "The identity rule and component atomicity are unchanged; only the",
                    "assignment was replaced, and the rejected draw's own numbers are kept.",
                ],
            },
            "carry_over": {
                "rule": [
                    "A v2 component inherits the MOST-READ role among the v1 roles of the",
                    "events it contains, precedence train > eval > holdout. A component with",
                    "no v1 role at all is freshly assignable. Only wholly-new components and",
                    "pure v1-holdout components can therefore reach the holdout.",
                    "The remainder is balanced toward the item targets using only the freshly",
                    "assignable components, against targets computed over the whole enlarged",
                    "population and against the load the carry-over already placed.",
                ],
                "why": [
                    "Neither half of the split is stable under a change of population.",
                    "component_id is sha256 over the component's entire member set, so one new",
                    "bridging event renames it; and _balanced_assignment is greedy against",
                    "per-split item targets computed over the whole population, so every",
                    "placement moves when the population does. A naive re-run over the wider",
                    "window would scatter already-read components into the new holdout with",
                    "every published count still looking healthy.",
                ],
                "components_inherited": len(carried["inherited_components"]),
                "components_freshly_assigned": len(carried["fresh_components"]),
                "components_forced_to_eval": len(carried["forced_eval_components"]),
                "forced_assignment_rule": (
                    "any component containing an observed-map delivery event is forced to "
                    "eval, out of train and out of holdout"
                ),
                "forced_assignment_source": (
                    "coldstart-prereg.json#split_protocol.forced_assignment, carried over "
                    "from scripts/recall_map_relevance_eval.py:702 assign_component_splits"
                ),
                "observed_map_events_in_window": len(observed_map_event_ids),
                "feature_deploy_instant": extracted.get("feature_deploy_instant"),
                "achieved_item_fractions": carried["achieved_item_fractions"],
                "v1_reproduction": v1["receipt"],
            },
            "one_cluster_bar": {
                "limit": MAX_EVAL_COMPONENT_SHARE,
                "holdout": "largest component share over ALL its items, unconditional",
                "eval": [
                    "largest component share over its FREELY ASSIGNED items.",
                    "A component forced to eval by the observed-map rule is placed by a",
                    "pre-registered rule with no discretion, precisely so it is neither fit",
                    "on nor held out, and cannot be a sampling accident.",
                ],
                "train": "exempt: it carries no generalization claim",
                "nothing_is_suppressed": (
                    "splits.eval.concentration publishes largest_component_share and "
                    "effective_components over all items as well as the freely-assigned "
                    "figures, so the forced arm's weight is readable directly"
                ),
            },
            "holdout_contamination": split_result["holdout_contamination"],
            "minimum_cold_candidates_per_split": MINIMUM_COLD_CANDIDATES_PER_SPLIT,
            "minimum_cold_candidates_per_split_source": (
                "coldstart-prereg.json#split_protocol.minimum_cold_candidates_per_split"
            ),
            "every_split_meets_the_minimum": True,
        },
        "enlargement": {
            "what_changed": [
                "the candidate window opens at CANDIDATE_WINDOW_START instead of the v1",
                "2026-08-01T00:00:00Z, and the per-event tail cap is lifted",
            ],
            "why": (
                "the v1 holdout carried 373 cold candidates against a pre-registered "
                "minimum of 400; the bar does not move, the population does"
            ),
            "what_still_bounds_it": {
                "reading": [
                    "The window is now the whole persisted history and the per-event cap is",
                    "not a lever, so these two bounds are what is left. Neither may be",
                    "loosened here: the opportunity rule is the frozen ConsumerIndex, and",
                    "the forced observed-map arm is pre-registered.",
                ],
                "residual_items_seen": len(candidates),
                "labelable_after_the_opportunity_rule": len(opportunity),
                "forced_to_eval_before_balancing": sum(
                    1
                    for row in opportunity
                    if carried["reason"].get(row.component_id)
                    == "touches_observed_map_delivery"
                ),
            },
            "window": {
                "v1": list(protocol["v1_candidate_window"]),
                "v2": list(protocol["candidate_window"]),
                "opens_before_the_first_persisted_event": True,
                "snapshot_first_event": extracted.get("snapshot_first_event"),
                "why_that_matters": (
                    "the enlarged window predates every recall event that exists, so it "
                    "excludes no era by choice; build_population refuses to run if that "
                    "stops being true of the snapshot in front of it"
                ),
            },
            "rank_cap": {
                "v1": protocol["organic_cap"],
                "v2": protocol["organic_cap"],
                "changed": False,
                "why_not": (
                    "it is not a lever: protocol.organic_cap_is_not_a_free_lever measures "
                    "that the outcome ledger covers ranks 3..8 and nothing else, so every "
                    "row a wider cap would add is unlabelable"
                ),
            },
            "eras": {
                ERA_PREEXISTING: _cohort(
                    [row for row in opportunity if _era(row) == ERA_PREEXISTING],
                    include_concentration=True,
                ),
                ERA_ENLARGEMENT: _cohort(
                    [row for row in opportunity if _era(row) == ERA_ENLARGEMENT],
                    include_concentration=True,
                ),
            },
            "population_before": {
                name: {
                    key: V1_SPLITS[name][key]
                    for key in (
                        "items",
                        "cold_candidates",
                        "components",
                        "effective_components",
                    )
                }
                for name in SPLIT_NAMES
            },
            "population_after": {
                name: {
                    "items": splits[name]["items"],
                    "cold_candidates": splits[name]["cold_candidates"],
                    "components": splits[name]["concentration"]["components"],
                    "effective_components": splits[name]["concentration"][
                        "effective_components"
                    ],
                }
                for name in SPLIT_NAMES
            },
            "concentration_cost": {
                "read_this_before_reading_the_counts": [
                    "The enlargement did not buy cold candidates for free. Under the sealed",
                    "identity rule -- connected components over the source-qualified cache",
                    "key and the session ids, unchanged here -- the pre-August material",
                    "collapses into a few very large components, because months of work",
                    "share one cache key. So every split gained items and LOST independent",
                    "units, and effective_components is the number to read, not components.",
                    "population_before and population_after above carry both, per split.",
                    "The per-era cut at splits.*.stratified.by_era shows where it comes",
                    "from: the enlargement era is far more concentrated than the",
                    "pre-existing August window it was added to.",
                ],
                "effective_components_before": {
                    name: V1_SPLITS[name]["effective_components"] for name in SPLIT_NAMES
                },
                "effective_components_after": {
                    name: splits[name]["concentration"]["effective_components"]
                    for name in SPLIT_NAMES
                },
                "identity_rule_unchanged": True,
                "what_it_cost_the_fit": [
                    "This is where the recorded step-3 HALT comes from, and until now the",
                    "manifest published the concentration and the sign failure without",
                    "joining them. The rank-uniform direction statistic that inverted for",
                    "result_score on the cold subpopulation and for bm25_score on the whole",
                    "tail is a statistic over roughly three independent units, dominated by",
                    "one: train's largest component alone holds 7384 of 12344 items.",
                    "train_direction_accounting at the top of this block is the join --",
                    "the same four members, the same statistic, with the estimator made",
                    "component-aware and cut by era.",
                ],
                "an_intra_split_cap_does_not_weaken_disjointness": {
                    "supersedes": "not_mitigated_by_subsampling",
                    "the_earlier_claim_was": (
                        "trimming a large component would divide it, and component "
                        "atomicity is what keeps a cache or session identity out of two "
                        "splits at once"
                    ),
                    "why_that_was_wrong": [
                        "It conflated ASSIGNMENT atomicity with USAGE atomicity. The",
                        "disjointness guarantee comes from assigning each whole connected",
                        "component to exactly one split. Capping how many of that",
                        "component's rows are used for ESTIMATION, inside the split it was",
                        "already assigned to, puts no cache key and no session id into a",
                        "second split -- the component is still undivided across splits,",
                        "and split.components_never_divided and",
                        "split.pairwise_component_intersections still hold unchanged.",
                    ],
                    "what_remains_true": [
                        "A cap costs rows, and on a population this concentrated it costs",
                        "a lot of them. cap_feasibility publishes items_dropped at every",
                        "cap and whether the 400-cold floor survives it.",
                    ],
                    "the_alternative_that_costs_no_rows": [
                        "component_equalized reweights instead of trimming, and pays in",
                        "effective sample size rather than in rows.",
                    ],
                },
            },
            "cap_feasibility": cap_feasibility(by_split),
            "component_equalized": component_equalized(by_split),
        },
        "splits": splits,
        "train_direction_accounting": train_direction_accounting(by_split["train"]),
        "population": {
            "opportunity_bearing": _cohort(opportunity, include_concentration=True),
            "opportunity_bearing_stratified": _stratified(opportunity),
            "all_residual": _cohort(candidates),
            "no_opportunity": _cohort([row for row in candidates if not row.opportunity]),
            "unledgered_pairs_skipped": int(extracted["unledgered"]),
            "history_read_truncated": {
                "items": int(extracted["history_read_truncated"]),
                "per_node_read_limit": MAX_HISTORY_DELIVERIES_PER_NODE,
                "why": (
                    "above the limit the live reader answers unavailable, so these "
                    "rows score on train means rather than on their own history"
                ),
            },
        },
        "cold_candidate_finding": {
            "population": "opportunity_bearing",
            "cold_candidates": len(cold),
            "cold_candidate_share": _ratio(len(cold), len(opportunity)),
            "cold_rate": _ratio(sum(1 for row in cold if row.consumed), len(cold)),
            "warm_rate": _ratio(sum(1 for row in warm if row.consumed), len(warm)),
            "cold_minus_warm_z": _two_proportion_z(
                sum(1 for row in cold if row.consumed),
                len(cold),
                sum(1 for row in warm if row.consumed),
                len(warm),
            ),
            "every_split_has_cold_candidates": not zero_cold,
            "splits_without_cold_candidates": sorted(zero_cold),
            "cold_candidates_per_split": {
                name: splits[name]["cold_candidates"] for name in SPLIT_NAMES
            },
            "minimum_cold_candidates_per_split": MINIMUM_COLD_CANDIDATES_PER_SPLIT,
            "every_split_meets_the_minimum": True,
            "family_unscoreable": _family_unscoreable(opportunity),
            "unfiltered_cold_candidate_share": _ratio(len(all_cold), len(candidates)),
            "reading": [
                "On the labelable population the cold rows are consumed materially",
                "LESS often than the warm ones, not indistinguishably from them.",
                "The indistinguishable reading reproduces only on the unfiltered",
                "residual, where items with no consumption opportunity are scored",
                "as not-consumed and flatten both arms toward the same low rate.",
            ],
        },
        "candidate_time_availability": _score_availability(opportunity),
        "privacy": {
            "aggregate_only": True,
            "opaque_component_digests_only": True,
            "item_level_rows": False,
            "corpus_text": False,
            "candidate_only_holdout_accessed": False,
            "identity_values_published": False,
        },
    }
    return block


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_published_assignment(block: Mapping[str, Any]) -> list[str]:
    """Check the published assignment inputs, then re-execute them.

    Two distinct properties, and the second is the one the plan actually asks
    for.  Completeness: the table has to cover the whole component universe,
    account for every published per-split member and for every item in the
    population, or the "these are the inputs" claim is a partial one.
    Reproduction: replaying the table has to land on the three shipped
    membership digests, which is what turns "the assignment is blind" from an
    assertion into something a third party can falsify with the artifact alone.
    """

    module = sealed()
    problems: list[str] = []
    split = block.get("split", {})
    inputs = split.get("assignment_inputs")
    if not isinstance(inputs, Mapping) or not inputs:
        return [
            "the split publishes no assignment_inputs, so the blindness invariant is "
            "unfalsifiable from the artifact"
        ]

    function = inputs.get("function", {})
    if not isinstance(function, Mapping):
        problems.append("assignment_inputs.function is not an object")
        function = {}
    if function.get("name") != split.get("assignment"):
        problems.append("assignment_inputs.function.name is not the published assignment name")
    if function.get("logical_location") != (
        "scripts/recall_map_coldstart_dataset.py:carry_over_assignment"
    ):
        problems.append("assignment_inputs.function does not locate the assignment function")

    constants = inputs.get("constants", {})
    if not isinstance(constants, Mapping):
        problems.append("assignment_inputs.constants is not an object")
        constants = {}
    if constants.get("seed") != split.get("seed"):
        problems.append("assignment_inputs.constants.seed is not the published split seed")
    if list(constants.get("split_names_order") or []) != list(SPLIT_NAMES):
        problems.append("assignment_inputs.constants.split_names_order is not the tie-break order")
    if list(constants.get("role_precedence") or []) != list(ROLE_PRECEDENCE):
        problems.append("assignment_inputs.constants.role_precedence is not the carry-over order")
    published_fractions = split.get("target_item_fractions", {})
    declared_fractions = constants.get("target_item_fractions", {})
    if not isinstance(declared_fractions, Mapping) or any(
        not isinstance(declared_fractions.get(name), (int, float))
        or round(float(declared_fractions[name]), 8) != published_fractions.get(name)
        for name in SPLIT_NAMES
    ):
        problems.append(
            "assignment_inputs.constants.target_item_fractions does not agree with "
            "split.target_item_fractions"
        )
    if not constants.get("tie_break"):
        problems.append("assignment_inputs.constants does not describe the tie-break")

    declared = inputs.get("declared")
    expected_names = [entry["name"] for entry in DECLARED_ASSIGNMENT_INPUTS]
    if not isinstance(declared, list) or [
        entry.get("name") if isinstance(entry, Mapping) else None for entry in declared
    ] != expected_names:
        problems.append(
            "assignment_inputs.declared does not enumerate the assignment's inputs in order"
        )
    elif any(
        not str(entry.get("what") or "").strip()
        or not str(entry.get("why_it_carries_no_consumption_label") or "").strip()
        for entry in declared
    ):
        problems.append("a declared assignment input says nothing about what it is or why")
    if not str(inputs.get("components_encoding") or "").strip():
        problems.append("assignment_inputs does not document its per-component encoding")
    reading = inputs.get("blindness_invariant_reading", {})
    if not isinstance(reading, Mapping) or not reading.get("verdict"):
        problems.append("assignment_inputs publishes no reading of the blindness invariant")

    components = inputs.get("components")
    if not isinstance(components, list) or not components:
        return problems + ["assignment_inputs publishes no component table"]
    digests = [entry.get("c") if isinstance(entry, Mapping) else None for entry in components]
    if any(
        not isinstance(value, str) or module.OPAQUE_DIGEST_RE.fullmatch(value) is None
        for value in digests
    ):
        problems.append("assignment_inputs.components carries a non-opaque component digest")
    if digests != sorted(str(value) for value in digests):
        problems.append("assignment_inputs.components is not sorted by component digest")
    if len(set(digests)) != len(digests):
        problems.append("assignment_inputs.components repeats a component digest")
    if len(components) != split.get("components"):
        problems.append(
            f"assignment_inputs.components lists {len(components)} components, not the "
            f"{split.get('components')} the split published"
        )
    # Counted rather than reported one by one: a malformed table would
    # otherwise bury every other problem under 706 identical lines.
    sizes: dict[str, int] = {}
    uncounted = 0
    empty_role = 0
    false_flag = 0
    for entry in components:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("c"), str):
            continue
        try:
            sizes[entry["c"]] = int(entry["n"])
        except (KeyError, TypeError, ValueError):
            uncounted += 1
        empty_role += int("r" in entry and not entry["r"])
        false_flag += int("f" in entry and entry["f"] is not True)
    if uncounted:
        problems.append(f"{uncounted} assignment_inputs component(s) publish no item count")
    if empty_role:
        problems.append(
            f"{empty_role} assignment_inputs component(s) publish an empty r, which the "
            "encoding says is omitted rather than written"
        )
    if false_flag:
        problems.append(
            f"{false_flag} assignment_inputs component(s) publish a non-true f, which the "
            "encoding says is omitted rather than written"
        )
    population = block.get("population", {}).get("opportunity_bearing", {}).get("items")
    if sum(sizes.values()) != population:
        problems.append(
            f"assignment_inputs item counts sum to {sum(sizes.values())}, not to the "
            f"{population} items of the opportunity-bearing population"
        )
    for name in SPLIT_NAMES:
        members = block.get("splits", {}).get(name, {}).get("components", [])
        missing = [value for value in members if value not in sizes]
        empty = [value for value in members if sizes.get(value) == 0]
        if missing:
            problems.append(
                f"{len(missing)} published {name} component(s) are absent from "
                "assignment_inputs.components"
            )
        if empty:
            problems.append(
                f"{len(empty)} published {name} component(s) carry no item in "
                "assignment_inputs.components"
            )

    # The re-executable check itself.
    receipt = split.get("assignment_reproduction")
    if not isinstance(receipt, Mapping) or not receipt:
        problems.append("the split publishes no assignment_reproduction receipt")
        receipt = {}
    published_digests = {
        name: block.get("splits", {}).get(name, {}).get("membership_digest")
        for name in SPLIT_NAMES
    }
    try:
        members = replay_published_assignment(inputs)
    except ColdStartError as exc:
        return problems + [f"the published assignment inputs do not re-execute: {exc}"]
    for name in SPLIT_NAMES:
        recomputed = _membership_digest(members.get(name, []))
        if recomputed != published_digests.get(name):
            problems.append(
                f"the {name} membership does not reproduce from assignment_inputs alone, so "
                "the published input list is not the assignment's real input list"
            )
        if receipt.get("membership_digests", {}).get(name) != recomputed:
            problems.append(f"the assignment_reproduction receipt misreports the {name} digest")
        if receipt.get("components", {}).get(name) != len(members.get(name, [])):
            problems.append(
                f"the assignment_reproduction receipt misreports the {name} component count"
            )
        if receipt.get("items", {}).get(name) != sum(
            sizes.get(component_id, 0) for component_id in members.get(name, [])
        ):
            problems.append(
                f"the assignment_reproduction receipt misreports the {name} item count"
            )
    if receipt.get("reproduces_published_v2") is not True:
        problems.append("the assignment_reproduction receipt does not claim reproduction")
    if list(receipt.get("inputs_used") or []) != expected_names:
        problems.append("the assignment_reproduction receipt does not name the inputs it used")
    if not receipt.get("procedure"):
        problems.append("the assignment_reproduction receipt states no re-implementable procedure")
    return problems


def verify_coldstart_block(block: Mapping[str, Any]) -> list[str]:
    """Everything checkable without re-reading a 640 MB snapshot."""

    module = sealed()
    problems: list[str] = []
    if block.get("artifact") != "recall-map-coldstart-candidate-dataset":
        problems.append("coldstart artifact kind mismatch")
    if block.get("as_of") != COLDSTART_AS_OF:
        problems.append("coldstart as_of is not the pinned instant")
    sealed_inputs = block.get("sealed_inputs", {})
    if sealed_inputs.get("historical_evaluator", {}).get("sha256") != EXPECTED_EVALUATOR_SHA256:
        problems.append("coldstart sealed evaluator hash mismatch")

    maturation = block.get("maturation", {})
    newest = str(maturation.get("snapshot_newest_event") or "")
    if not newest or newest < str(maturation.get("as_of_plus_horizon") or "~"):
        problems.append("snapshot does not cover as_of + horizon_hours")
    if maturation.get("every_window_in_scope_matured") is not True:
        problems.append("maturation assertion missing")

    protocol = block.get("protocol", {})
    for name, expected in (
        ("horizon_hours", 24),
        ("organic_head_cut", 3),
        ("organic_cap", 6),
        ("map_cap", 6),
    ):
        if protocol.get(name) != expected:
            problems.append(f"protocol {name} is not the sealed constant {expected}")
    # The enlargement moved the window and nothing else, and says why the other
    # candidate lever is not one — measured on the snapshot, not asserted.
    lever = protocol.get("organic_cap_is_not_a_free_lever", {})
    coverage = lever.get("measured_on_the_snapshot", {}) if isinstance(lever, Mapping) else {}
    if coverage.get("first_ledgered_rank") != protocol.get("organic_head_cut"):
        problems.append("the outcome ledger does not start at the sealed head cut")
    if coverage.get("last_ledgered_rank") != 8:
        problems.append(
            "the outcome ledger no longer ends at rank 8, so the cap claim is stale"
        )
    if protocol.get("mutable_node_stats_read") != []:
        problems.append("mutable node stats entered the cold-start extraction")
    window = protocol.get("candidate_window")
    if (
        not isinstance(window, list)
        or len(window) != 2
        or window[1] != shift_hours(COLDSTART_AS_OF, -24)
    ):
        problems.append("candidate window does not close a horizon before the as-of")
    elif window[0] != CANDIDATE_WINDOW_START:
        problems.append("candidate window does not open at the enlarged window start")
    v1_window = protocol.get("v1_candidate_window")
    if not isinstance(v1_window, list) or v1_window[:1] != [V1_CANDIDATE_WINDOW_START]:
        problems.append("the v1 candidate window is not recorded as the enlargement's baseline")

    # "The enlarged window excludes no era" has to be readable off the artifact,
    # not only enforced inside the run that produced it.
    enlargement_window = block.get("enlargement", {}).get("window", {})
    first_event = enlargement_window.get("snapshot_first_event")
    if not isinstance(first_event, str) or not first_event:
        problems.append("the snapshot's first recall event is not published")
    elif not CANDIDATE_WINDOW_START <= first_event:
        problems.append(
            "the enlarged window opens after the snapshot's first recall event, so it "
            "excludes an era"
        )
    elif enlargement_window.get("opens_before_the_first_persisted_event") is not True:
        problems.append("the no-era-excluded assertion is missing or false")

    split = block.get("split", {})
    if split.get("algorithm") != (
        "connected_components(source-qualified cache key, transport session, session)"
    ):
        problems.append("coldstart split algorithm differs from the manifest identity rule")
    if list(split.get("partitions", [])) != list(SPLIT_NAMES):
        problems.append("coldstart split is not the three named partitions")
    if split.get("pairwise_component_disjoint") is not True:
        problems.append("pairwise disjointness assertion missing")
    if split.get("components_never_divided") is not True:
        problems.append("component atomicity assertion missing")

    # The carry-over receipt: the one asset this subtree still has is a holdout
    # that has never been read, and enlarging the population is exactly the move
    # that can destroy it while every count still looks healthy.
    contamination = split.get("holdout_contamination", {})
    if contamination.get("holds") is not True:
        problems.append("the holdout non-contamination assertion is missing or false")
    if contamination.get("intersection") != 0:
        problems.append(
            "v2 holdout events intersect the v1 train/eval events: the sealed "
            f"partition is contaminated ({contamination.get('intersection')} event(s))"
        )
    if not contamination.get("v1_read_events_in_scope"):
        problems.append("the holdout assertion was checked against no already-read v1 event")
    carry_over = split.get("carry_over", {})
    reproduction = carry_over.get("v1_reproduction", {})
    if reproduction.get("reproduces_pinned_v1") is not True:
        problems.append("the v1 partition reproduction receipt is missing or false")
    for name in SPLIT_NAMES:
        recorded = reproduction.get("splits", {}).get(name)
        if not isinstance(recorded, Mapping) or recorded != dict(V1_SPLITS[name]):
            problems.append(f"the reconstructed v1 {name} split does not match the pinned v1")
    if not carry_over.get("components_inherited"):
        problems.append("no component inherited a v1 role, so nothing was carried over")
    if carry_over.get("components_freshly_assigned") is None:
        problems.append("the carry-over does not report how many components were fresh")

    # The v1 receipt above reproduces the v1 partition, which is a different
    # property.  This one reproduces the SHIPPED assignment from the inputs the
    # manifest declares, which is what
    # coldstart-prereg-v2.json#split_protocol.assignment_rule
    # .blindness_invariant_binding actually demands.
    problems.extend(verify_published_assignment(block))

    minimum = split.get("minimum_cold_candidates_per_split")
    if minimum != MINIMUM_COLD_CANDIDATES_PER_SPLIT:
        problems.append(
            f"the published cold-candidate minimum is {minimum}, not the pre-registered "
            f"{MINIMUM_COLD_CANDIDATES_PER_SPLIT}"
        )
    if split.get("every_split_meets_the_minimum") is not True:
        problems.append("the cold-candidate minimum assertion is missing or false")

    splits = block.get("splits", {})
    sets: dict[str, set[str]] = {}
    for name in SPLIT_NAMES:
        summary = splits.get(name)
        if not isinstance(summary, Mapping):
            problems.append(f"split {name} is missing")
            sets[name] = set()
            continue
        members = summary.get("components", [])
        if any(
            not isinstance(value, str) or module.OPAQUE_DIGEST_RE.fullmatch(value) is None
            for value in members
        ):
            problems.append(f"split {name} contains a non-opaque component digest")
        sets[name] = set(members)
        # coldstart-prereg.json#split_protocol.per_split_reporting, plus the
        # effective-component count that says how many independent units the
        # split really holds.
        for required in (
            "items",
            "consumed",
            "rate",
            "cold_candidates",
            "cold_candidate_share",
            "cold_consumed",
            "cold_base_rate",
            "family_unscoreable",
            "membership_digest",
        ):
            if summary.get(required) is None:
                problems.append(f"split {name} does not report {required}")
        if summary.get("concentration", {}).get("effective_components") is None:
            problems.append(f"split {name} does not report effective_components")
        if not summary.get("cold_candidates"):
            problems.append(f"split {name} contains no cold candidates")
        elif int(summary["cold_candidates"]) < MINIMUM_COLD_CANDIDATES_PER_SPLIT:
            problems.append(
                f"split {name} carries {summary['cold_candidates']} cold candidates, below "
                f"the pre-registered minimum of {MINIMUM_COLD_CANDIDATES_PER_SPLIT}"
            )
        # Widening backwards crosses corpus regimes and lifting the cap reaches
        # ranks v1 never held; both cuts are published so the mixture is
        # checkable rather than something a reader has to take on trust.
        strata = summary.get("stratified", {})
        for axis in ("by_era", "by_rank_band"):
            cut = strata.get(axis)
            if not isinstance(cut, Mapping) or not cut:
                problems.append(f"split {name} is not stratified {axis}")
                continue
            for label, stratum in cut.items():
                if not isinstance(stratum, Mapping):
                    problems.append(f"split {name} stratum {axis}.{label} is malformed")
                    continue
                for required in ("items", "cold_candidates", "family_unscoreable"):
                    if stratum.get(required) is None:
                        problems.append(
                            f"split {name} stratum {axis}.{label} does not report {required}"
                        )
        for axis in ("by_era", "by_rank_band"):
            cut = strata.get(axis, {})
            if isinstance(cut, Mapping) and cut:
                total = sum(int(stratum.get("items") or 0) for stratum in cut.values())
                if total != summary.get("items"):
                    problems.append(
                        f"split {name} strata {axis} sum to {total}, not to its "
                        f"{summary.get('items')} items"
                    )
        if isinstance(strata.get("by_era"), Mapping):
            enlargement = strata["by_era"].get(ERA_ENLARGEMENT, {})
            if not enlargement.get("items"):
                problems.append(f"split {name} received nothing from the enlargement")
        # An evaluation set dominated by one component is one observation, not
        # a sample; train is exempt because it carries no generalization claim.
        # The holdout carries the generalization claim and holds no forced
        # material, so it is checked unconditionally.  Eval is checked on the
        # material that was drawn freely: a component forced there by the
        # pre-registered observed-map rule is placed by a rule with no
        # discretion, precisely so it is neither fit on nor held out, and it
        # cannot be a sampling accident.  Its size is published, not suppressed.
        concentration = summary.get("concentration", {})
        if name == "holdout":
            if concentration.get("forced_components"):
                problems.append("the holdout carries a forced observed-map component")
            share = concentration.get("largest_component_share")
            measured_on = "all"
        elif name == "eval":
            share = concentration.get("largest_freely_assigned_component_share")
            measured_on = "freely assigned"
        else:
            share = None
            measured_on = ""
        if measured_on and (
            not isinstance(share, (int, float)) or share > MAX_EVAL_COMPONENT_SHARE
        ):
            problems.append(
                f"split {name} is dominated by one component, measured over its "
                f"{measured_on} items (largest share {share}, "
                f"limit {MAX_EVAL_COMPONENT_SHARE})"
            )
        if summary.get("items") and summary.get("consumed") is not None:
            expected_rate = _ratio(int(summary["consumed"]), int(summary["items"]))
            if summary.get("rate") != expected_rate:
                problems.append(f"split {name} rate is not consumed over items")
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            if sets[left] & sets[right]:
                problems.append(f"split components overlap: {left}/{right}")
            recorded = split.get("pairwise_component_intersections", {}).get(f"{left}_{right}")
            if recorded != 0:
                problems.append(f"recorded {left}/{right} intersection is not zero")

    totals = block.get("population", {}).get("opportunity_bearing", {})
    summed = sum(int(splits.get(name, {}).get("items") or 0) for name in SPLIT_NAMES)
    if totals.get("items") != summed:
        problems.append("split item counts do not sum to the opportunity-bearing population")

    finding = block.get("cold_candidate_finding", {})
    if finding.get("every_split_has_cold_candidates") is not True:
        problems.append("cold-candidate presence assertion missing or false")

    privacy = block.get("privacy", {})
    if privacy.get("candidate_only_holdout_accessed") is not False:
        problems.append("candidate-only holdout non-access assertion missing")
    if privacy.get("aggregate_only") is not True:
        problems.append("aggregate-only assertion missing")

    problems.extend(verify_independence_accounting(block))

    leaked = sorted(set(module._walk_keys(block)) & module.SENSITIVE_ARTIFACT_KEYS)
    if leaked:
        problems.append("item-level sensitive artifact key(s): " + ", ".join(leaked))
    try:
        module.load_frozen_bindings().effect.check_privacy(block)
    except Exception as exc:  # noqa: BLE001 - guard type lives in the frozen tool
        problems.append(f"privacy guard rejected the coldstart block: {exc}")
    return problems


def verify_independence_accounting(block: Mapping[str, Any]) -> list[str]:
    """The independence accounting, re-checked from the artifact alone.

    Two different jobs in one pass.  The arithmetic checks say the published
    tables are internally consistent -- a cap cannot retain more rows than the
    split holds, a Kish size cannot exceed the row count, the sorted component
    sizes must sum to the split.  The seal check says the label-bearing half
    stayed inside its boundary: no family-member statistic may appear under
    ``coldstart_eval`` or ``coldstart_holdout``, and the accounting must not
    have grown a second split while nobody was reading.
    """

    problems: list[str] = []
    splits = block.get("splits", {})
    enlargement = block.get("enlargement", {})

    # Item 1 -- per-split component-size distributions, all three splits.
    for name in SPLIT_NAMES:
        summary = splits.get(name, {})
        distribution = summary.get("component_sizes")
        if not isinstance(distribution, Mapping):
            problems.append(f"split {name} publishes no component_sizes distribution")
            continue
        pairs = distribution.get("size_and_cold_by_size_desc") or []
        sizes = [pair[0] for pair in pairs]
        colds = [pair[1] for pair in pairs]
        if sum(sizes) != summary.get("items"):
            problems.append(f"split {name} component sizes do not sum to its item count")
        if sum(colds) != summary.get("cold_candidates"):
            problems.append(
                f"split {name} per-component cold counts do not sum to its cold candidates"
            )
        if sizes != sorted(sizes, reverse=True):
            problems.append(f"split {name} component sizes are not sorted descending")
        if any(cold > size for size, cold in pairs):
            problems.append(f"split {name} reports a component with more cold rows than rows")
        if len(sizes) != len(summary.get("components") or []):
            problems.append(f"split {name} component_sizes disagrees with its component list")
        if distribution.get("effective_components") != summary.get("concentration", {}).get(
            "effective_components"
        ):
            problems.append(
                f"split {name} component_sizes reports a different effective_components "
                "than its concentration block"
            )

    # Item 2 -- the cap grid.
    feasibility = enlargement.get("cap_feasibility")
    if not isinstance(feasibility, Mapping):
        problems.append("enlargement publishes no cap_feasibility table")
    else:
        if list(feasibility.get("caps") or []) != list(CAP_GRID):
            problems.append("cap_feasibility does not cover the declared cap grid")
        if feasibility.get("minimum_cold_candidates_per_split") != (
            MINIMUM_COLD_CANDIDATES_PER_SPLIT
        ):
            problems.append("cap_feasibility cites the wrong cold-candidate minimum")
        table = feasibility.get("table") or {}
        clearing = list(feasibility.get("caps_clearing_the_cold_minimum_on_every_split") or [])
        for cap in CAP_GRID:
            entry = table.get(str(cap))
            if not isinstance(entry, Mapping):
                problems.append(f"cap_feasibility carries no row for cap {cap}")
                continue
            for name in SPLIT_NAMES:
                cell = entry.get(name) or {}
                items = cell.get("items")
                published_items = splits.get(name, {}).get("items")
                if not isinstance(items, int) or items > (published_items or 0):
                    problems.append(f"cap {cap} on {name} retains more items than the split has")
                    continue
                expected = sum(
                    min(size, cap)
                    for size, _cold in (
                        splits.get(name, {})
                        .get("component_sizes", {})
                        .get("size_and_cold_by_size_desc")
                        or []
                    )
                )
                if items != expected:
                    problems.append(
                        f"cap {cap} on {name} retains {items} items, but the published "
                        f"component sizes imply {expected}"
                    )
                if cell.get("items_dropped") != (published_items or 0) - items:
                    problems.append(f"cap {cap} on {name} misreports items_dropped")
                cold = cell.get("cold_candidates")
                if not isinstance(cold, int) or cold > items:
                    problems.append(f"cap {cap} on {name} reports more cold rows than rows")
                elif cell.get("meets_the_cold_minimum") is not (
                    cold >= MINIMUM_COLD_CANDIDATES_PER_SPLIT
                ):
                    problems.append(f"cap {cap} on {name} misreports the cold minimum")
            every = all(
                (entry.get(name) or {}).get("meets_the_cold_minimum") is True
                for name in SPLIT_NAMES
            )
            if entry.get("all_splits_meet_the_cold_minimum") is not every:
                problems.append(f"cap {cap} misreports whether every split clears the floor")
            if every != (cap in clearing):
                problems.append(f"cap {cap} is misreported in the feasible cap set")

    # Item 3 -- the no-rows-dropped alternative.
    equalized = enlargement.get("component_equalized")
    if not isinstance(equalized, Mapping):
        problems.append("enlargement publishes no component_equalized table")
    else:
        if equalized.get("no_rows_dropped") is not True:
            problems.append("component_equalized does not claim to drop no rows")
        for name in SPLIT_NAMES:
            cell = (equalized.get("per_split") or {}).get(name) or {}
            items = splits.get(name, {}).get("items")
            if cell.get("items") != items:
                problems.append(f"component_equalized misreports {name}'s item count")
            if cell.get("cold_candidates") != splits.get(name, {}).get("cold_candidates"):
                problems.append(f"component_equalized misreports {name}'s cold candidates")
            effective = cell.get("effective_sample_size")
            if not isinstance(effective, (int, float)) or effective > (items or 0):
                problems.append(
                    f"component_equalized reports an effective sample size for {name} that "
                    "exceeds its row count, which no weighting can do"
                )
            if cell.get("rows_dropped") != 0:
                problems.append(f"component_equalized dropped rows on {name}")

    # Item 4 -- train only, and provably so.
    accounting = block.get("train_direction_accounting")
    if not isinstance(accounting, Mapping):
        problems.append("the block publishes no train_direction_accounting")
        return problems
    if accounting.get("split") != DIRECTION_ACCOUNTING_SPLIT or list(
        accounting.get("splits_measured") or []
    ) != [DIRECTION_ACCOUNTING_SPLIT]:
        problems.append(
            "train_direction_accounting names a split other than train; the label-bearing "
            "half of the accounting is train-only and the artifact must say so"
        )
    # Prose may NAME the sealed splits -- why_only_train has to say which
    # splits it is staying out of.  A cohort KEYED by one is the violation, so
    # the guard reads keys and not values.
    forbidden = sorted(set(SPLIT_NAMES) - {DIRECTION_ACCOUNTING_SPLIT})
    accounting_keys = set(_walk_mapping_keys(accounting))
    for name in forbidden:
        if name in accounting_keys:
            problems.append(
                f"train_direction_accounting carries a cohort keyed {name!r}; a "
                "family-member statistic on that split is exactly what the seal forbids"
            )
    arms: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(accounting.get("unweighted"), Mapping):
        arms.append(("unweighted", accounting["unweighted"]))
    for label, container in (("by_cap", "by_cap"), ("by_era", "by_era")):
        group = accounting.get(container) or {}
        if not isinstance(group, Mapping):
            problems.append(f"train_direction_accounting.{container} is not a table")
            continue
        arms.extend((f"{label}.{key}", value) for key, value in group.items())
    if isinstance(accounting.get("component_equalized"), Mapping):
        arms.append(("component_equalized", accounting["component_equalized"]))
    expected_caps = {str(cap) for cap in CAP_GRID}
    if set((accounting.get("by_cap") or {}).keys()) != expected_caps:
        problems.append("train_direction_accounting.by_cap does not cover the cap grid")
    if set((accounting.get("by_era") or {}).keys()) != set(ERA_NAMES):
        problems.append("train_direction_accounting.by_era does not cover both eras")
    train_items = splits.get("train", {}).get("items")
    members = {member for _source, member, _sign in FAMILY_DIRECTIONS}
    for label, arm in arms:
        if not isinstance(arm, Mapping):
            problems.append(f"train_direction_accounting arm {label} is not a table")
            continue
        rows = arm.get("rows")
        if not isinstance(rows, int) or rows > (train_items or 0):
            problems.append(f"arm {label} reports more rows than coldstart_train holds")
        if set((arm.get("members") or {}).keys()) != members:
            problems.append(f"arm {label} does not report all four registered members")
        for field_name in ("cold_rows", "cold_consumed", "cold_check_measurable"):
            if arm.get(field_name) is None:
                problems.append(
                    f"arm {label} omits {field_name}, so the cold_subpopulation_clause "
                    "minima are not checkable on it"
                )
        measurable = (
            isinstance(arm.get("cold_rows"), int)
            and isinstance(arm.get("cold_consumed"), int)
            and arm["cold_rows"] >= MINIMUM_COLD_ROWS_FOR_A_DIRECTION_CHECK
            and arm["cold_consumed"] >= MINIMUM_CONSUMED_COLD_ROWS_FOR_A_DIRECTION_CHECK
        )
        if arm.get("cold_check_measurable") is not measurable:
            problems.append(f"arm {label} misreports cold_check_measurable")
        for member, entry in (arm.get("members") or {}).items():
            for key in ("whole_tail", "cold"):
                value = entry.get(key)
                if value is None:
                    continue
                if not isinstance(value, (int, float)) or abs(value) > 1.0:
                    problems.append(
                        f"arm {label} member {member} reports {key}={value!r}, outside the "
                        "[-1, +1] a difference of rank-uniform means can occupy"
                    )
                matches = entry.get(f"{key}_matches_registered")
                sign = 1.0 if entry.get("registered_direction") == "positive" else -1.0
                if matches is not (value * sign > 0):
                    problems.append(
                        f"arm {label} member {member} misreports whether {key} matches its "
                        "registered direction"
                    )
    # The stability fold must actually be a fold over the arms it summarises,
    # not a hand-written table that agrees with them today.
    arm_labels = dict(
        [("unweighted", accounting.get("unweighted") or {})]
        + [
            (f"cap_{cap}", (accounting.get("by_cap") or {}).get(str(cap)) or {})
            for cap in CAP_GRID
        ]
        + [("component_equalized", accounting.get("component_equalized") or {})]
        + [
            (f"era_{name}", (accounting.get("by_era") or {}).get(name) or {})
            for name in ERA_NAMES
        ]
    )
    stability = (accounting.get("direction_stability") or {}).get("by_member") or {}
    if set(stability) != members:
        problems.append("direction_stability does not cover all four registered members")
    for member, entry in stability.items():
        per_arm = entry.get("matches_registered_by_arm") or {}
        if set(per_arm) != set(arm_labels):
            problems.append(f"direction_stability for {member} does not fold over every arm")
            continue
        for label, verdicts in per_arm.items():
            source_entry = (arm_labels[label].get("members") or {}).get(member) or {}
            for key in ("whole_tail", "cold"):
                if verdicts.get(key) != source_entry.get(f"{key}_matches_registered"):
                    problems.append(
                        f"direction_stability for {member} restates arm {label} {key} "
                        "differently from the arm itself"
                    )
        disagreeing = sorted(
            label
            for label, verdicts in per_arm.items()
            if verdicts.get("whole_tail") is False or verdicts.get("cold") is False
        )
        if list(entry.get("arms_disagreeing_with_the_registered_direction") or []) != disagreeing:
            problems.append(f"direction_stability misreports which arms {member} disagrees on")
        if entry.get("stable_across_every_arm") is not (not disagreeing):
            problems.append(f"direction_stability misreports overall stability for {member}")

    reproduction = accounting.get("reproduces_published_step3") or {}
    if reproduction.get("agrees") is not True:
        problems.append(
            "train_direction_accounting does not claim to reproduce the published step-3 "
            "numbers, so it is not an account of the HALT that was recorded"
        )
    for member, entry in (reproduction.get("compared") or {}).items():
        if entry.get("whole_tail") != entry.get("published_whole_tail") or entry.get(
            "cold"
        ) != entry.get("published_cold"):
            problems.append(
                f"the unweighted arm does not reproduce the published step-3 reading for "
                f"{member}"
            )
        unweighted = ((accounting.get("unweighted") or {}).get("members") or {}).get(member, {})
        if entry.get("whole_tail") != unweighted.get("whole_tail") or entry.get(
            "cold"
        ) != unweighted.get("cold"):
            problems.append(
                f"the reproduction receipt for {member} does not quote the unweighted arm "
                "it claims to be quoting"
            )
    return problems


def _walk_mapping_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_mapping_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_mapping_keys(nested)


def rebind_analysis(manifest: Mapping[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    """Repoint the feature analysis at the manifest it now accompanies.

    This is a pointer rebinding and nothing else: not one measured value inside
    ``feature-analysis.json`` is recomputed.  Re-deriving that evidence would
    mean re-running a sealed evaluator against a snapshot that no longer
    exists, which is exactly the thing that cannot be done honestly.
    """

    analysis["dataset_manifest_sha256"] = sha256_text(canonical_json(dict(manifest)))
    return analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    assert_not_holdout(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ColdStartError(f"JSON artifact must be an object: {path}")
    return payload


def _write_canonical(path: Path, payload: Mapping[str, Any]) -> None:
    assert_not_holdout(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(dict(payload)) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="rebuild the population and write the coldstart block")
    run.add_argument("--source", required=True, metavar="NAME=PATH")
    run.add_argument("--as-of", default=COLDSTART_AS_OF)
    run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    run.add_argument(
        "--no-rebind",
        action="store_true",
        help="do not repoint feature-analysis.json at the rewritten manifest",
    )

    verify = subparsers.add_parser("verify", help="verify the published coldstart block")
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    verify.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)

    # The re-executable check
    # coldstart-prereg-v2.json#split_protocol.assignment_rule
    # .blindness_invariant_binding requires.  It takes no --source and opens no
    # database: the manifest is its only data input, so a third party holding
    # the artifact can run it and see for themselves that the shipped
    # partition follows from the declared, label-free inputs.
    verify_assignment = subparsers.add_parser(
        "verify-assignment",
        help=(
            "recompute the three split memberships from coldstart.split.assignment_inputs "
            "alone and check them against the published membership digests"
        ),
    )
    verify_assignment.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        if args.as_of != COLDSTART_AS_OF:
            raise ColdStartError(f"--as-of is pinned at {COLDSTART_AS_OF}; got {args.as_of}")
        if "=" not in args.source:
            raise ColdStartError("--source must be LOGICAL_NAME=SQLITE_PATH")
        name, raw_path = args.source.split("=", 1)
        extracted = build_population(Path(raw_path).expanduser(), source_name=name)
        try:
            block = build_coldstart_block(extracted)
        finally:
            extracted["connection"].close()
        problems = verify_coldstart_block(block)
        if problems:
            raise ColdStartError("coldstart verification failed: " + "; ".join(problems))

        manifest = _read_json(args.manifest)
        manifest["coldstart"] = block
        _write_canonical(args.manifest, manifest)
        if not args.no_rebind:
            analysis = _read_json(args.analysis)
            _write_canonical(args.analysis, rebind_analysis(manifest, analysis))
        summary = block["splits"]
        print(
            "wrote coldstart splits: "
            + " ".join(
                f"{name}={summary[name]['items']}"
                f"(cold {summary[name]['cold_candidates']})"
                for name in SPLIT_NAMES
            )
            + f"; minimum {MINIMUM_COLD_CANDIDATES_PER_SPLIT}"
            + "; v1 train/eval events in the holdout: "
            + str(block["split"]["holdout_contamination"]["intersection"])
        )
        return 0

    if args.command == "verify":
        manifest = _read_json(args.manifest)
        analysis = _read_json(args.analysis)
        block = manifest.get("coldstart")
        problems: list[str] = []
        if not isinstance(block, Mapping):
            problems.append("manifest carries no coldstart block")
        else:
            problems.extend(verify_coldstart_block(block))
        expected = sha256_text(canonical_json(dict(manifest)))
        if analysis.get("dataset_manifest_sha256") != expected:
            problems.append("feature analysis does not bind the extended manifest")
        if problems:
            for problem in problems:
                print(f"FAIL: {problem}", file=sys.stderr)
            return 1
        print(
            "OK: coldstart splits are pairwise component-disjoint, every split clears "
            f"{MINIMUM_COLD_CANDIDATES_PER_SPLIT} cold candidates and is stratified by era "
            "and rank band, no already-read v1 train/eval event reaches the holdout, and "
            "the analysis binds the extended manifest"
        )
        return 0

    if args.command == "verify-assignment":
        manifest = _read_json(args.manifest)
        block = manifest.get("coldstart")
        if not isinstance(block, Mapping):
            print("FAIL: the manifest carries no coldstart block", file=sys.stderr)
            return 1
        split = block.get("split", {})
        inputs = split.get("assignment_inputs")
        if not isinstance(inputs, Mapping):
            print(
                "FAIL: coldstart.split publishes no assignment_inputs, so the assignment "
                "cannot be re-executed from the artifact",
                file=sys.stderr,
            )
            return 1
        published = {
            name: block.get("splits", {}).get(name, {}).get("membership_digest")
            for name in SPLIT_NAMES
        }
        try:
            receipt = assignment_reproduction_receipt(inputs, published)
        except ColdStartError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        stale = [
            name
            for name in SPLIT_NAMES
            if split.get("assignment_reproduction", {}).get("membership_digests", {}).get(name)
            != receipt["membership_digests"][name]
        ]
        if stale:
            print(
                "FAIL: the published assignment_reproduction receipt disagrees with a fresh "
                "replay: " + ", ".join(stale),
                file=sys.stderr,
            )
            return 1
        print(
            "OK: replayed "
            + str(len(inputs.get("components", [])))
            + f" components from {inputs['function']['name']} using only "
            + ", ".join(receipt["inputs_used"])
            + "; reproduced "
            + " ".join(
                f"{name}={receipt['components'][name]}c/{receipt['items'][name]}i"
                for name in SPLIT_NAMES
            )
            + " and all three published membership digests"
        )
        return 0
    raise AssertionError(args.command)  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
