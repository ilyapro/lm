#!/usr/bin/env python3
"""The one-shot-per-slot accrual driver for ``confirmatory-holdout-v4``.

``POLICY.md`` states that "every slot is a distinct invocation.  No process may
sleep, poll, or remain alive waiting across slots, and an orchestration retry is
not a new cadence event."  This module is the supervisor that shape demands: one
process drives exactly one slot index and then exits.  There is no loop, no
sleep, no wait, and no state carried forward in memory -- an external timer
starts a fresh interpreter for every slot.

Three rules shape every decision here.

*The calendar is the only authority on time.*  ``--slot-index`` selects a slot;
the scheduled instant and the grace deadline are re-derived from the frozen
constants through :func:`ap_confirmatory_runtime_v4.slot_times` on every run, so
nothing observable at launch can move, extend, or re-anchor a slot.  The launch
instant itself comes from :class:`DriverClock`, which reads the real system
clock and nothing else.

*The command line cannot fake an instant.*  There is deliberately no ``--now``,
no clock flag, and no environment override.  A CLI path that could name the
launch instant would let an orchestration retry masquerade as a cadence event
and would let a source open outside its authorized window -- both unrecoverable
under the frozen schedule.  Tests reach the clock only through the Python-level
``clock=`` seam on :func:`run_slot`, which no argv can address.

*A late launch is terminal, and it proves that without touching a source.*  An
invocation before ``slot_at`` rejects before the campaign boundary is consulted
at all; an invocation at or after ``slot_at + grace_seconds`` appends the
``missed-slot`` marker with ``source_open_count`` fixed at zero and enters
terminal ``schedule-integrity-failure``.  There is no late launch, backfill,
catch-up, or discretionary skip.

The driver owns schedule discipline, the durable append-only layout, and
dispatch order.  It owns no operator mapping: alias locators, database
observations, and the runtime observer all arrive through a
:class:`CampaignBoundary` supplied by an operator-private module, exactly as
:mod:`ap_confirmatory_snapshot_v4` requires, so no tracked file here can ever
name a source.

Because ``ap_confirmatory_accrual_v4`` derives its capability keys with
``os.urandom`` at import, a ``LedgerState`` cannot cross a process boundary.
The boundary therefore rebuilds the state each invocation and the driver
*proves* the rebuilt state against the durable transcript with
``validate_persisted_ledger_prefix`` before it appends anything.  For the same
reason the seal handoff is constructed in this process rather than spawned:
only the sealer child is a separate process, and only exact bytes cross to it.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# The accrual import chain writes ``scripts/__pycache__`` unless this is set
# first, which would leave a byte outside ``--work-dir``.
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
NAMESPACE_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "confirmatory-holdout-v4"
)

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Reach every upstream module through one import chain.  All four use
# ``type(x) is SomeDataclass`` identity checks, so a second importlib copy of
# the runtime would silently invalidate every wrapper handed back to them.
import ap_confirmatory_seal_v4 as seal  # noqa: E402
import ap_confirmatory_snapshot_v4 as snapshot  # noqa: E402

accrual = seal.accrual
probe = accrual.probe
runtime = accrual.runtime

if (
    runtime is not probe.runtime
    or runtime is not snapshot.runtime
    or probe is not snapshot.probe
    or probe is not accrual.probe
):
    raise SystemExit("v4 runtime module identity is not shared")

IntegrityFailure = runtime.IntegrityFailure

NAMESPACE = runtime.NAMESPACE
SCHEMA_VERSION = runtime.SCHEMA_VERSION
SOURCE_ALIASES = runtime.SOURCE_ALIASES

# --------------------------------------------------------------------------
# The durable layout, and the statuses one invocation can end in
# --------------------------------------------------------------------------

# ``<work-dir>/ledger/NNNNNNNN.json`` and ``<work-dir>/markers/``.  The marker
# directory may never be the ledger directory: ``accrual`` keeps the ledger
# directory exact-member, so a stray marker file there would invalidate the
# whole prefix.
LEDGER_DIRECTORY = "ledger"
MARKER_DIRECTORY = "markers"

DISPOSITION_PRE_SLOT = "pre-slot"
DISPOSITION_IN_WINDOW = "in-window"
DISPOSITION_AFTER_GRACE = "after-grace"

STATUS_PRE_SLOT_REFUSED = "pre-slot-refused"
STATUS_MISSED = "missed-slot"
STATUS_ALREADY_RESOLVED = "slot-already-resolved"
STATUS_BELOW_FLOOR = "below-floor"
STATUS_SEAL_HANDOFF = "seal-handoff"
STATUS_TERMINAL = "terminal"

MISSED_SLOT_STATUS = "terminal-schedule-integrity-failure"
MISSED_SLOT_RECEIPT_KIND = "missed-slot"

EXIT_SLOT_RESOLVED = 0
EXIT_INTEGRITY_FAILURE = 1
EXIT_USAGE = 2
EXIT_NAMESPACE_REFUSED = 3
EXIT_TERMINAL_SCHEDULE_INTEGRITY_FAILURE = 4
EXIT_PRE_SLOT_REFUSED = 5
EXIT_SLOT_ALREADY_RESOLVED = 6

NAMESPACE_REFUSAL_CODE = "namespace_work_dir_refused"

# ``_validate_missed_marker`` accepts the marker only from a state that still
# owes this slot a probe.  Anything else means the slot was already consumed.
MISSED_SLOT_PHASES = frozenset(
    {
        accrual.PHASE_BOOTSTRAP_PROBE,
        accrual.PHASE_AWAITING_PROBE,
        accrual.PHASE_RETRY_ALLOWED,
    }
)

# Slot 0 opens the initial segment: its probe attempt is written *first*, then
# the runtime-attestation attempt, then the source-binding attestation, which
# is what opens the ``ActiveSegment``.  The ceremony therefore lands directly
# in ``PHASE_PROBE_ACTIVE`` and the driver must not append a second attempt.
INITIAL_ATTESTATION_PHASES = frozenset(
    {
        accrual.PHASE_BOOTSTRAP_PROBE,
        accrual.PHASE_INITIAL_RUNTIME_ATTEMPT,
        accrual.PHASE_INITIAL_SOURCE_BINDING,
    }
)

# A closed segment must be re-attested before the next fixed slot.  That
# ceremony lands in ``PHASE_AWAITING_PROBE``, so this slot's attempt follows.
SUCCESSOR_ATTESTATION_PHASES = frozenset(
    {
        accrual.PHASE_SUCCESSOR_ATTEMPT,
        accrual.PHASE_SUCCESSOR_BINDING,
        accrual.PHASE_SUCCESSOR_SEGMENT,
    }
)

# Every other phase means the ledger owes something this driver does not issue
# -- a terminal receipt, a same-slot retry, or a handoff already in flight.
DRIVABLE_PHASES = (
    INITIAL_ATTESTATION_PHASES
    | SUCCESSOR_ATTESTATION_PHASES
    | {accrual.PHASE_AWAITING_PROBE}
)


class NamespaceRefusal(Exception):
    """A work root resolving inside the frozen namespace."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path


def _fail() -> None:
    raise IntegrityFailure from None


def normalized_path(path: Path | str) -> Path:
    return Path(os.path.realpath(os.fspath(path)))


def require_work_dir_outside_namespace(path: Path | str) -> Path:
    """Refuse any work root resolving inside the frozen namespace.

    This mirrors ``ap_confirmatory_seal_v4.require_output_root_outside_namespace``
    exactly, including its distinct exit status, so a namespace-resident
    ``--work-dir`` is refused before a single byte is written and can never be
    mistaken for a slot failure.  ``realpath`` resolution means a symlink
    pointing into the byte-locked namespace is refused too.
    """

    candidate = normalized_path(path)
    namespace = normalized_path(NAMESPACE_ROOT)
    if candidate == namespace or namespace in candidate.parents:
        raise NamespaceRefusal(candidate)
    return candidate


# --------------------------------------------------------------------------
# The clock -- one seam, unreachable from any command line
# --------------------------------------------------------------------------


class DriverClock:
    """The launch instant, read from the real system clock.

    There is deliberately no override parameter, no environment lookup, and no
    command-line flag anywhere above this class.  A driver that could be told
    what time it is could open a source outside its authorized window or let an
    orchestration retry present itself as a cadence event, and the frozen
    schedule treats both as unrecoverable.

    Tests substitute a subclass through the ``clock=`` keyword of
    :func:`run_slot`.  That seam exists only in Python; :func:`parse_arguments`
    exposes no argument that can reach it.
    """

    def now(self) -> datetime:
        return datetime.now(UTC)

    def canonical_now(self) -> str:
        return runtime.canonical_utc(self.now())


def _require_clock(clock: Any) -> DriverClock:
    if clock is None:
        return DriverClock()
    if not isinstance(clock, DriverClock):
        _fail()
    return clock


# --------------------------------------------------------------------------
# The frozen calendar decides what this invocation is allowed to do
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SlotLaunch:
    """One observed launch instant, placed against one frozen slot window."""

    slot_index: int
    scheduled_at: str
    grace_deadline_at: str
    launched_at: str
    disposition: str


def classify_launch(slot_index: Any, launched_at: Any) -> SlotLaunch:
    """Place a launch instant against the frozen half-open execution window.

    The bounds come from :func:`ap_confirmatory_runtime_v4.slot_times` and
    nothing else, so no observation can re-anchor them.  ``POLICY.md`` defines
    the valid execution window as ``[slot_at(i), slot_at(i) + grace_seconds)``:
    the lower bound is inclusive and the grace deadline is exclusive.
    """

    slot = runtime.slot_times(slot_index)
    observed = runtime.parse_utc(launched_at, receipt=True)
    scheduled = runtime.parse_utc(slot.scheduled_at, receipt=True)
    grace = runtime.parse_utc(slot.grace_deadline_at, receipt=True)
    if observed < scheduled:
        disposition = DISPOSITION_PRE_SLOT
    elif observed >= grace:
        disposition = DISPOSITION_AFTER_GRACE
    else:
        disposition = DISPOSITION_IN_WINDOW
    return SlotLaunch(
        slot_index=slot.slot_index,
        scheduled_at=slot.scheduled_at,
        grace_deadline_at=slot.grace_deadline_at,
        launched_at=runtime.canonical_utc(observed),
        disposition=disposition,
    )


# --------------------------------------------------------------------------
# What one invocation reports
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SlotOutcome:
    """The complete, aggregate-only account of one slot invocation."""

    status: str
    exit_code: int
    launch: SlotLaunch
    source_open_count: int
    terminal_status: str | None = None
    ledger_entry_path: Path | None = None
    seal_consumption_marker_sha256: str | None = None
    sealer_process_launched_at: str | None = None
    steps: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        """Aggregate-only stdout payload; no count and no private value."""

        return {
            "namespace": NAMESPACE,
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "slot_index": self.launch.slot_index,
            "scheduled_at": self.launch.scheduled_at,
            "grace_deadline_at": self.launch.grace_deadline_at,
            "launched_at": self.launch.launched_at,
            "disposition": self.launch.disposition,
            "source_open_count": self.source_open_count,
            "terminal_status_or_null": self.terminal_status,
            "ledger_entry_or_null": (
                None
                if self.ledger_entry_path is None
                else self.ledger_entry_path.name
            ),
            "seal_consumption_marker_sha256_or_null": (
                self.seal_consumption_marker_sha256
            ),
            "sealer_process_launched_at_or_null": self.sealer_process_launched_at,
            "steps": list(self.steps),
            "slept_or_polled": False,
            "real_packet_sealed": self.status == STATUS_SEAL_HANDOFF,
        }


# --------------------------------------------------------------------------
# The operator boundary -- everything this tracked file must never name
# --------------------------------------------------------------------------


class CampaignBoundary:
    """Everything one slot needs that a tracked file may not carry.

    ``sources.aliases`` is an ``external-untracked-operator-mapping``: alias
    locators, database identities, and the runtime observer are operator-private
    and are resolved before the slot opens.  The driver therefore takes them as
    typed calls rather than as paths on a command line, and never stores a
    default for any of them.

    An implementation is supplied by ``--campaign-module``.  Every method may
    raise :class:`IntegrityFailure`; nothing else is caught.
    """

    def open_ledger(self, ledger_directory: Path) -> Any:
        """Rebuild the durable ledger into an in-process ``LedgerState``.

        A ``LedgerState`` cannot cross a process boundary, so every invocation
        rebuilds it through the typed transitions.  Whatever this returns is
        proven against the on-disk transcript by the driver before use: the
        implementation must have persisted every entry it replayed.
        """

        raise NotImplementedError

    def runtime_observer_identity(self) -> Any:
        """The ``HashAndBytes`` identity of the runtime observer in use."""

        raise NotImplementedError

    def attest_initial_source_binding(self, state: Any, launch: SlotLaunch) -> Any:
        """Run the slot-0 initial runtime and source-binding ceremony.

        Returns the state after the ``source-binding-attestation`` entry, with
        the initial :class:`ActiveSegment` open.  The driver persists each
        appended entry; the boundary owns only the live observations.
        """

        raise NotImplementedError

    def attest_successor_segment(self, state: Any, launch: SlotLaunch) -> Any:
        """Run the successor re-attestation after a closed segment."""

        raise NotImplementedError

    def source_locators(self) -> Mapping[str, Any]:
        """The operator-private ``{alias: SourceLocator}`` mapping."""

        raise NotImplementedError

    def observe_database_identity(self, alias: str) -> Mapping[str, Any]:
        """Observe the live configured database identity for one alias."""

        raise NotImplementedError

    def observe_active_state(self) -> tuple[Any, Any]:
        """Observe ``(ServiceTuple, SourceBinding)`` for the active segment."""

        raise NotImplementedError

    def open_dev_reference(self) -> int:
        """Return an already-open read-only fd for the frozen dev reference."""

        raise NotImplementedError

    def seal_ceremony(self, request: "SealRequest") -> Any:
        """Build the seal ceremony for an all-floor pass.

        The default implementation constructs the real
        :class:`ap_confirmatory_seal_v4.SealCeremony`; the whole handoff must
        run in this process, because its capability objects are process-local.
        """

        return seal.SealCeremony(
            clock=request.clock,
            state=request.state,
            computation=request.computation,
            observe_active_state=request.observe_active_state,
            runtime_observer_identity=request.runtime_observer_identity,
            seal_root=request.seal_root,
            draft_dir=request.draft_dir,
            publish_dir=request.publish_dir,
            handoff_path=request.handoff_path,
            snapshot_paths=request.snapshot_paths,
            snapshot_fds=request.snapshot_fds,
            sealer_identity=request.sealer_identity,
            verifier_identity=request.verifier_identity,
        )


@dataclass(frozen=True, slots=True)
class SealRequest:
    """Exactly what :meth:`CampaignBoundary.seal_ceremony` is handed."""

    clock: Any
    state: Any
    computation: Any
    observe_active_state: Callable[[], tuple[Any, Any]]
    runtime_observer_identity: Mapping[str, Any]
    seal_root: Path
    draft_dir: Path
    publish_dir: Path
    handoff_path: Path
    snapshot_paths: Mapping[str, Path]
    snapshot_fds: Mapping[str, int]
    sealer_identity: Mapping[str, Any]
    verifier_identity: Mapping[str, Any]
    slot_index: int


class _SealClockAdapter(seal.SealClock):
    """Give the ceremony the driver's clock, not a second time source."""

    def __init__(self, clock: DriverClock) -> None:
        self._clock = clock

    def now(self) -> datetime:
        return self._clock.now()


# --------------------------------------------------------------------------
# Durable append-only wiring
# --------------------------------------------------------------------------


def _persist_head(ledger_directory: Path, state: Any) -> Path:
    """Append exactly the last transition, O_EXCL, never replacing anything.

    ``write_ledger_entry_exclusive`` refuses unless the directory holds the
    exact prior prefix and the target index does not yet exist, so a second
    invocation for an already-written index fails here rather than overwriting.
    """

    return accrual.write_ledger_entry_exclusive(ledger_directory, state)


def _require_state_matches_disk(ledger_directory: Path, state: Any) -> None:
    """Prove the rebuilt in-process state is exactly the durable transcript."""

    accrual.validate_persisted_ledger_prefix(ledger_directory, state)


def build_missed_slot_marker(state: Any, launch: SlotLaunch) -> bytes:
    """Build the frozen ``missed-slot`` receipt for a launch after grace.

    ``source_open_count`` is the literal integer zero, not an observation: the
    marker is recorded precisely because no source was, or ever may be, opened
    for this slot.
    """

    value = {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "receipt_kind": MISSED_SLOT_RECEIPT_KIND,
        "status": MISSED_SLOT_STATUS,
        "slot_index": launch.slot_index,
        "scheduled_at": launch.scheduled_at,
        "grace_deadline_at": launch.grace_deadline_at,
        "recorded_at": launch.launched_at,
        "source_open_count": 0,
        "previous_ledger_entry_sha256": state.head,
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
    }
    return runtime.canonical_json_bytes(value)


# --------------------------------------------------------------------------
# One slot, one invocation
# --------------------------------------------------------------------------


def _pre_slot_outcome(launch: SlotLaunch) -> SlotOutcome:
    """Refuse before the campaign boundary is consulted at all.

    ``POLICY.md``: "An invocation before ``slot_at`` rejects before source
    access."  Nothing is read, nothing is created, and no boundary method has
    been called, so the slot remains exactly as unstarted as it was.
    """

    return SlotOutcome(
        status=STATUS_PRE_SLOT_REFUSED,
        exit_code=EXIT_PRE_SLOT_REFUSED,
        launch=launch,
        source_open_count=0,
        steps=("classify-launch",),
    )


def _record_missed_slot(
    *, ledger_directory: Path, state: Any, launch: SlotLaunch
) -> SlotOutcome:
    """Append the terminal missed marker without any source access.

    ``POLICY.md``: "If the window closes without a validator-valid receipt, an
    append-only ``missed`` marker is recorded without source access.  That state
    is terminal ``schedule-integrity-failure``: no later probe or seal is
    authorized."  There is no backfill and no catch-up, so this function opens
    no source, builds no snapshot, and consults no floor.
    """

    if state.phase not in MISSED_SLOT_PHASES:
        _fail()
    raw = build_missed_slot_marker(state, launch)
    missed = accrual.append_missed_slot(state, raw)
    path = _persist_head(ledger_directory, missed)
    return SlotOutcome(
        status=STATUS_MISSED,
        exit_code=EXIT_TERMINAL_SCHEDULE_INTEGRITY_FAILURE,
        launch=launch,
        source_open_count=0,
        terminal_status=missed.terminal_status,
        ledger_entry_path=path,
        steps=("classify-launch", "prove-durable-transcript", "append-missed-slot"),
    )


def _slot_is_already_resolved(state: Any, slot_index: int) -> bool:
    """Has this slot already reached a validator-valid resolution?

    ``POLICY.md``: "The first validator-valid slot resolution is immutable and
    forbids every replacement or second resolution."  A resolution advances
    ``next_slot_index`` past its own slot, or clears it at the last slot and at
    every terminal state, so anything other than "this slot is still owed" means
    the answer is yes.
    """

    if state.terminal_status is not None:
        return True
    if state.next_slot_index is None:
        return True
    return state.next_slot_index != slot_index


def _build_probe_attempt(state: Any, launch: SlotLaunch, *, observer: Any) -> Any:
    """Build and validate this slot's ``probe-attempt`` marker."""

    active = state.active_segment
    if type(active) is not runtime.ActiveSegment:
        _fail()
    launched = runtime.parse_utc(launch.launched_at, receipt=True)
    grace = runtime.parse_utc(launch.grace_deadline_at, receipt=True)
    # ``watchdog_deadline_at == min(launched + 3600s, grace - 60s)``, exactly.
    watchdog = min(
        runtime._add_seconds(launched, 3600), runtime._add_seconds(grace, -60)
    )
    value = {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "receipt_kind": "probe-attempt",
        "slot_index": launch.slot_index,
        "scheduled_at": launch.scheduled_at,
        "grace_deadline_at": launch.grace_deadline_at,
        "attempt_ordinal": 0,
        "launched_at": launch.launched_at,
        "watchdog_deadline_at": runtime.canonical_utc(watchdog),
        "segment_id": active.segment_id,
        "previous_ledger_entry_sha256": state.head,
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "probe_sha256_and_bytes": probe._self_identity().as_dict(),
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
    }
    return probe.validate_probe_attempt_marker(
        runtime.canonical_json_bytes(value),
        contract=state.contract,
        active_segment=active,
        expected_previous_ledger_sha256=state.head,
        expected_slot_index=launch.slot_index,
        expected_attempt_ordinal=0,
    )


def _append_below_floor(
    *,
    ledger_directory: Path,
    state: Any,
    computation: Any,
    launch: SlotLaunch,
    validated_at: str,
) -> tuple[Any, Path]:
    """Append the immutable aggregate receipt for a below-floor slot.

    ``readiness.intermediate_below_floor_action``: "append immutable aggregate
    receipt, destroy snapshots, and continue to the next fixed slot".  A valid
    below-floor receipt grants no authority and is not insufficiency.
    """

    active = state.active_segment
    attempt = state.current_probe_attempt
    raw = runtime.canonical_json_bytes(
        probe.build_below_floor_resolution(
            computation,
            active_segment=active,
            launched_at=launch.launched_at,
            validated_at=validated_at,
            previous_ledger_entry_sha256=state.head,
            contract=state.contract,
            probe_identity=attempt.probe_identity,
        )
    )
    resolution = probe.validate_probe_resolution(
        raw,
        contract=state.contract,
        active_segment=active,
        expected_previous_ledger_sha256=state.head,
        expected_probe_identity=attempt.probe_identity,
        expected_snapshot_identities=computation.snapshots(),
    )
    resolved = accrual.append_below_floor_resolution(state, resolution)
    return resolved, _persist_head(ledger_directory, resolved)


def _seal_paths(work: Path) -> tuple[Path, Path, Path]:
    """The three non-ledger roots the ceremony needs, all under ``--work-dir``."""

    return work / "draft", work / "packet", work / "handoff.json"


def _require_seal_inputs_staged(work: Path) -> None:
    """Prove the seal inputs exist *before* a source is opened.

    A ready pass can occur at any slot, and the whole handoff must complete
    inside that slot's grace window -- there is no staging a packet mid-window.
    Discovering a missing packet after the probe has opened a source would be a
    structural failure after source open, which ``POLICY.md`` makes terminal, so
    the check belongs here, while the slot is still recoverable.
    """

    _draft_dir, publish_dir, _handoff_path = _seal_paths(work)
    for required in (seal.BUILDER_PATH, publish_dir / "recipe" / "verify.py"):
        if not required.is_file():
            _fail()


def run_slot(
    *,
    work_dir: Path | str,
    slot_index: Any,
    campaign: CampaignBoundary,
    clock: DriverClock | None = None,
) -> SlotOutcome:
    """Drive exactly one slot, then return.  This function never waits.

    The order below is the whole point of the module and is falsifiable step by
    step: the namespace refusal precedes every write, the frozen calendar
    precedes every clock comparison, the pre-slot refusal precedes every
    boundary call, and the missed-slot marker precedes -- and permanently
    forecloses -- every source open.
    """

    work = require_work_dir_outside_namespace(work_dir)
    driver_clock = _require_clock(clock)
    # The single reading of the real clock that places this invocation.
    launch = classify_launch(slot_index, driver_clock.canonical_now())
    if launch.disposition == DISPOSITION_PRE_SLOT:
        # Before any source open, and before the boundary is even consulted.
        return _pre_slot_outcome(launch)

    ledger_directory = work / LEDGER_DIRECTORY
    state = campaign.open_ledger(ledger_directory)
    _require_state_matches_disk(ledger_directory, state)

    if launch.disposition == DISPOSITION_AFTER_GRACE:
        return _record_missed_slot(
            ledger_directory=ledger_directory, state=state, launch=launch
        )

    if _slot_is_already_resolved(state, launch.slot_index):
        # An orchestration retry is not a new cadence event, and the first
        # validator-valid resolution is immutable.  Nothing is written.
        return SlotOutcome(
            status=STATUS_ALREADY_RESOLVED,
            exit_code=EXIT_SLOT_ALREADY_RESOLVED,
            launch=launch,
            source_open_count=0,
            terminal_status=state.terminal_status,
            steps=("classify-launch", "prove-durable-transcript"),
        )

    return _run_in_window_slot(
        work=work,
        ledger_directory=ledger_directory,
        state=state,
        launch=launch,
        campaign=campaign,
        clock=driver_clock,
    )


def _run_in_window_slot(
    *,
    work: Path,
    ledger_directory: Path,
    state: Any,
    launch: SlotLaunch,
    campaign: CampaignBoundary,
    clock: DriverClock,
) -> SlotOutcome:
    """Dispatch the attestation, the probe, and the resolution, in order."""

    steps = ["classify-launch", "prove-durable-transcript"]
    if state.phase not in DRIVABLE_PHASES:
        _fail()
    # Everything this slot will need, proven while nothing is yet committed:
    # once the attempt marker exists, a failure that cannot prove its progress
    # is conservatively terminalized rather than retried.
    _require_seal_inputs_staged(work)
    steps.append("prove-seal-inputs-staged")
    # An independent re-proof of the same window against the same frozen
    # calendar.  The launcher refuses to exist outside it.
    window = snapshot.open_slot_window(launch.slot_index, launch.launched_at)

    if state.phase in INITIAL_ATTESTATION_PHASES:
        state = campaign.attest_initial_source_binding(state, launch)
        steps.append("attest-initial-source-binding")
    elif state.phase in SUCCESSOR_ATTESTATION_PHASES:
        state = campaign.attest_successor_segment(state, launch)
        steps.append("attest-successor-segment")
    _require_state_matches_disk(ledger_directory, state)

    if state.phase == accrual.PHASE_AWAITING_PROBE:
        observer = campaign.runtime_observer_identity()
        state = accrual.append_probe_attempt(
            state, _build_probe_attempt(state, launch, observer=observer)
        )
        _persist_head(ledger_directory, state)
        steps.append("append-probe-attempt")

    if state.phase != accrual.PHASE_PROBE_ACTIVE:
        _fail()

    launcher = snapshot.SourceSnapshotLauncher(
        window=window,
        locators=campaign.source_locators(),
        observe_database_identity=campaign.observe_database_identity,
        staging_root=work / "snapshots",
    )
    dev_fd = campaign.open_dev_reference()
    try:
        launcher.mark_keyed_worker_handoff()
        computation = probe.compute_probe_aggregate(
            active_segment=state.active_segment,
            observe_active_state=campaign.observe_active_state,
            capture_snapshot=launcher.capture_snapshot,
            dev_reference_fd=dev_fd,
            slot_index=launch.slot_index,
            contract=state.contract,
        )
        steps.append("compute-probe-aggregate")

        if not computation.provisional_ready:
            _append_below_floor(
                ledger_directory=ledger_directory,
                state=state,
                computation=computation,
                launch=launch,
                validated_at=clock.canonical_now(),
            )
            steps.append("append-below-floor")
            return SlotOutcome(
                status=STATUS_BELOW_FLOOR,
                exit_code=EXIT_SLOT_RESOLVED,
                launch=launch,
                source_open_count=launcher.source_open_count,
                steps=tuple(steps),
            )

        return _run_seal_handoff(
            work=work,
            state=state,
            computation=computation,
            launch=launch,
            campaign=campaign,
            clock=clock,
            launcher=launcher,
            steps=steps,
        )
    finally:
        with contextlib.suppress(OSError):
            os.close(dev_fd)
        # ``intermediate_below_floor_action`` destroys the snapshots; a ready
        # pass has already handed its retained copies to the blocked child.
        launcher.destroy()


def _run_seal_handoff(
    *,
    work: Path,
    state: Any,
    computation: Any,
    launch: SlotLaunch,
    campaign: CampaignBoundary,
    clock: DriverClock,
    launcher: Any,
    steps: list[str],
) -> SlotOutcome:
    """Perform ``readiness.ready_action`` inside this invocation and in grace.

    "Inside the same supervisor invocation and before publishing ready: retain
    the exact snapshots, perform the final complete runtime and alias/database-
    binding check, durably create the no-overwrite seal-consumption marker, and
    spawn the one-shot sealer within grace."  ``SealCeremony`` owns that exact
    seven-step order; the driver's contribution is that it happens here, in this
    process, while the grace window is still open.
    """

    remaining = runtime.utc_microseconds(
        launch.grace_deadline_at, receipt=True
    ) - runtime.utc_microseconds(clock.canonical_now(), receipt=True)
    if remaining <= 0:
        # The whole handoff must complete inside the window; a ready pass that
        # cannot start in grace never consumes seal authority.
        _fail()

    draft_dir, publish_dir, handoff_path = _seal_paths(work)
    ceremony = campaign.seal_ceremony(
        SealRequest(
            clock=_SealClockAdapter(clock),
            state=state,
            computation=computation,
            observe_active_state=campaign.observe_active_state,
            runtime_observer_identity=campaign.runtime_observer_identity().as_dict(),
            seal_root=work,
            draft_dir=draft_dir,
            publish_dir=publish_dir,
            handoff_path=handoff_path,
            snapshot_paths=launcher.snapshot_paths(),
            snapshot_fds=launcher.snapshot_fds(),
            sealer_identity=seal.identity_of_path(seal.BUILDER_PATH),
            verifier_identity=seal.identity_of_path(
                publish_dir / "recipe" / "verify.py"
            ),
            slot_index=launch.slot_index,
        )
    )
    status, receipt = ceremony.run()
    steps.append("seal-handoff")
    marker = getattr(ceremony, "marker_identity", None)
    return SlotOutcome(
        status=STATUS_SEAL_HANDOFF if status == seal.EXIT_SEALED else STATUS_TERMINAL,
        exit_code=EXIT_SLOT_RESOLVED
        if status == seal.EXIT_SEALED
        else EXIT_INTEGRITY_FAILURE,
        launch=launch,
        source_open_count=launcher.source_open_count,
        terminal_status=receipt.get("status"),
        seal_consumption_marker_sha256=(
            None if marker is None else marker.get("sha256")
        ),
        sealer_process_launched_at=receipt.get("sealer_process_launched_at_or_null"),
        steps=tuple(steps),
    )


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def load_campaign_module(path: Path | str) -> CampaignBoundary:
    """Load the operator-private campaign boundary.

    The alias mapping is an ``external-untracked-operator-mapping``, so it can
    never live in a tracked file here and cannot be reduced to a path flag: the
    driver needs typed observations, not locations.  The module must expose
    ``build_campaign() -> CampaignBoundary``.
    """

    resolved = require_work_dir_outside_namespace(path)
    if not resolved.is_file():
        _fail()
    spec = importlib.util.spec_from_file_location(
        "ap_confirmatory_slot_v4_campaign", resolved
    )
    if spec is None or spec.loader is None:
        _fail()
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    builder = getattr(module, "build_campaign", None)
    if not callable(builder):
        _fail()
    campaign = builder()
    if not isinstance(campaign, CampaignBoundary):
        _fail()
    return campaign


def build_parser() -> argparse.ArgumentParser:
    """The whole command line, exposed so its surface stays checkable.

    Note what is absent and must stay absent: no ``--now``, no ``--launched-at``,
    no ``--clock``, no ``--date``, and no environment override.  The launch
    instant is read from the system clock inside :class:`DriverClock` and can be
    substituted only through the Python-level ``clock=`` seam on
    :func:`run_slot`, which no argv can address.
    """

    parser = argparse.ArgumentParser(
        description=(
            "confirmatory-holdout-v4 one-shot-per-slot accrual driver: "
            "drives exactly one slot and exits"
        )
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--slot-index", type=int, required=True)
    parser.add_argument("--campaign-module", type=Path, required=True)
    return parser


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = parse_arguments(argv)
    except SystemExit as request:
        # argparse exits 0 for --help and 2 for a real usage error.
        return EXIT_SLOT_RESOLVED if request.code in (0, None) else EXIT_USAGE

    try:
        work = require_work_dir_outside_namespace(arguments.work_dir)
        outcome = run_slot(
            work_dir=work,
            slot_index=arguments.slot_index,
            campaign=load_campaign_module(arguments.campaign_module),
        )
    except NamespaceRefusal as refusal:
        print(
            json.dumps(
                {
                    "code": NAMESPACE_REFUSAL_CODE,
                    "path": os.fspath(refusal.path),
                    "slot_executed": False,
                },
                sort_keys=True,
            )
        )
        return EXIT_NAMESPACE_REFUSED
    except IntegrityFailure:
        # Nothing about a private source may escape through a diagnostic.
        print(
            json.dumps(
                {
                    "namespace": NAMESPACE,
                    "schema_version": SCHEMA_VERSION,
                    "integrity_escalation": True,
                    "slot_executed": False,
                },
                sort_keys=True,
            )
        )
        return EXIT_INTEGRITY_FAILURE

    print(json.dumps(outcome.summary(), sort_keys=True))
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
