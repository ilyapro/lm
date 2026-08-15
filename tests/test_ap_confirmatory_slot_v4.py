"""Synthetic adversarial tests for the confirmatory-v4 per-slot accrual driver.

Every ledger, marker, and source in this file is built by this module inside a
temporary directory.  The campaign is preregistered source-blind and slot 0 does
not open until 2026-08-17T00:00:00Z, so no test here may drive a real slot, open
a configured production database, reach any host, or write into the frozen
namespace: any of those would consume evidence the protocol has not authorized.

The driver is falsified rather than demonstrated.  Each of the five load-bearing
claims -- pre-slot refusal before source access, terminal handling of a late
launch, one slot per process with no sleeping or polling, the immutability of
the first valid resolution, and refusal of a namespace work root -- has a test
that would fail if the property were removed.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ap_confirmatory_slot_v4.py"
NAMESPACE_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "confirmatory-holdout-v4"
)

SLOT_ZERO_AT = "2026-08-17T00:00:00.000000Z"
SLOT_ZERO_GRACE_AT = "2026-08-17T06:00:00.000000Z"
IN_WINDOW_AT = "2026-08-17T00:30:00.000000Z"
BEFORE_SLOT_ZERO_AT = "2026-08-16T23:59:59.999999Z"
SLOT_ONE_AT = "2026-08-18T00:00:00.000000Z"
SLOT_ONE_IN_WINDOW_AT = "2026-08-18T00:30:00.000000Z"
SLOT_ONE_GRACE_AT = "2026-08-18T06:00:00.000000Z"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver() -> Any:
    return _load("ap_confirmatory_slot_v4_test", SCRIPT)


def _failure(module: Any, call: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Every rejection is the shared, message-free integrity signal."""

    with pytest.raises(module.IntegrityFailure) as caught:
        call(*args, **kwargs)
    assert caught.value.args == ()
    assert str(caught.value) == ""


# ---------------------------------------------------------------------------
# Test doubles: a clock that no command line can reach, and a boundary that
# records every question the driver asks it
# ---------------------------------------------------------------------------


def _clock_class(driver: Any) -> Any:
    class _FixedClock(driver.DriverClock):
        """A launch instant injected through the Python-level seam only."""

        def __init__(self, moment: str) -> None:
            self.moment = driver.runtime.parse_utc(moment, receipt=True)
            self.reads = 0

        def now(self) -> datetime:
            self.reads += 1
            # Later reads advance, so a test cannot accidentally depend on a
            # frozen wall clock the real driver would never see.
            return self.moment + timedelta(microseconds=self.reads - 1)

    return _FixedClock


def _campaign_class(driver: Any) -> Any:
    class _Campaign(driver.CampaignBoundary):
        """Records every boundary call so a test can prove what never ran."""

        def __init__(self, ledger_state: Any = None) -> None:
            self.calls: list[str] = []
            self.ledger_state = ledger_state
            self.source_opens = 0

        def open_ledger(self, ledger_directory: Path) -> Any:
            self.calls.append("open_ledger")
            if self.ledger_state is None:
                raise AssertionError("the driver asked for a ledger it may not have")
            return self.ledger_state

        def runtime_observer_identity(self) -> Any:
            self.calls.append("runtime_observer_identity")
            return driver.runtime.hash_and_bytes(driver.seal.SYNTHETIC_OBSERVER)

        def attest_initial_source_binding(self, state: Any, launch: Any) -> Any:
            self.calls.append("attest_initial_source_binding")
            raise AssertionError("no test may run a real attestation ceremony")

        def attest_successor_segment(self, state: Any, launch: Any) -> Any:
            self.calls.append("attest_successor_segment")
            raise AssertionError("no test may run a real attestation ceremony")

        def source_locators(self) -> Any:
            self.calls.append("source_locators")
            raise AssertionError("no test may address a source")

        def observe_database_identity(self, alias: str) -> Any:
            self.calls.append(f"observe_database_identity:{alias}")
            raise AssertionError("no test may observe a live database")

        def observe_active_state(self) -> Any:
            self.calls.append("observe_active_state")
            raise AssertionError("no test may observe a live runtime")

        def open_dev_reference(self) -> int:
            self.calls.append("open_dev_reference")
            self.source_opens += 1
            raise AssertionError("no test may open the frozen dev reference")

    return _Campaign


# ---------------------------------------------------------------------------
# Synthetic ledgers.  Nothing below reads a real source.
# ---------------------------------------------------------------------------


def _ledger_directory(driver: Any, work: Path) -> Path:
    directory = work / driver.LEDGER_DIRECTORY
    directory.mkdir(mode=0o700, parents=True)
    return directory


def _genesis_ledger(driver: Any, work: Path) -> tuple[Path, Any]:
    """Entry zero only: slot 0 is still owed and no segment is open."""

    directory = _ledger_directory(driver, work)
    state = driver.seal.persist_head(directory, driver.accrual.initialize_ledger())
    assert state.phase == driver.accrual.PHASE_BOOTSTRAP_PROBE
    assert state.next_slot_index == 0
    return directory, state


def _resolved_slot_zero(driver: Any, work: Path) -> tuple[Path, Any]:
    """The frozen bootstrap order, then one valid below-floor slot-0 receipt."""

    directory = _ledger_directory(driver, work)
    observer = driver.runtime.hash_and_bytes(driver.seal.SYNTHETIC_OBSERVER)
    state = driver.seal.bootstrap_synthetic_ledger(observer, directory)
    snapshots = {
        "local": driver.runtime.hash_and_bytes(b"synthetic-local-snapshot"),
        "alt": driver.runtime.hash_and_bytes(b"synthetic-alt-snapshot"),
    }
    computation = driver.seal.synthesize_probe_computation(
        state,
        snapshot_identities=snapshots,
        counts=driver.seal.below_floor_counts(),
    )
    assert not computation.provisional_ready
    state = driver.seal.persist_head(
        directory, driver.seal.close_below_floor_slot(state, computation)
    )
    # An intermediate below-floor receipt is ``continue``, not ``insufficient``:
    # it consumes slot 0 and advances the ledger, and only the last slot ever
    # latches ``final_resolution_status``.
    assert state.entries[-1].entry_kind == "below-floor"
    assert state.entries[-1].slot_index == 0
    assert state.phase == driver.accrual.PHASE_AWAITING_PROBE
    assert state.next_slot_index == 1
    return directory, state


def _ledger_bytes(directory: Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in sorted(directory.iterdir())}


def _entry(directory: Path, index: int) -> dict[str, Any]:
    path = directory / f"{index:08d}.json"
    return json.loads(path.read_bytes().decode("utf-8"))


# ---------------------------------------------------------------------------
# The frozen calendar is the only authority on when a slot may run
# ---------------------------------------------------------------------------


def test_slot_bounds_are_derived_only_from_the_frozen_constants(driver: Any) -> None:
    """Every one of the 29 slots re-derives from anchor, cadence, and grace."""

    anchor = datetime(2026, 8, 17, tzinfo=UTC)
    assert driver.runtime.ANCHOR_AT == "2026-08-17T00:00:00Z"
    assert driver.runtime.CADENCE_SECONDS == 86_400
    assert driver.runtime.GRACE_SECONDS == 21_600
    assert (driver.runtime.FIRST_SLOT_INDEX, driver.runtime.LAST_SLOT_INDEX) == (0, 28)

    for index in range(29):
        scheduled = anchor + timedelta(seconds=index * 86_400)
        launch = driver.classify_launch(index, driver.runtime.canonical_utc(scheduled))
        assert launch.scheduled_at == driver.runtime.canonical_utc(scheduled)
        assert launch.grace_deadline_at == driver.runtime.canonical_utc(
            scheduled + timedelta(seconds=21_600)
        )
        assert launch.disposition == driver.DISPOSITION_IN_WINDOW

    # The lattice runs 2026-08-17 .. 2026-09-14 and stops there.
    assert driver.classify_launch(28, "2026-09-14T00:00:00.000000Z").scheduled_at == (
        "2026-09-14T00:00:00.000000Z"
    )
    for rejected in (-1, 29, 1_000, True, "0", 1.0, None):
        _failure(driver, driver.classify_launch, rejected, IN_WINDOW_AT)


@pytest.mark.parametrize(
    ("observed_at", "disposition"),
    (
        ("2026-08-16T00:00:00.000000Z", "pre-slot"),
        (BEFORE_SLOT_ZERO_AT, "pre-slot"),
        (SLOT_ZERO_AT, "in-window"),
        (IN_WINDOW_AT, "in-window"),
        ("2026-08-17T05:59:59.999999Z", "in-window"),
        (SLOT_ZERO_GRACE_AT, "after-grace"),
        ("2026-08-17T06:00:00.000001Z", "after-grace"),
        ("2027-01-01T00:00:00.000000Z", "after-grace"),
    ),
)
def test_the_execution_window_is_half_open_at_the_grace_deadline(
    driver: Any, observed_at: str, disposition: str
) -> None:
    """``[slot_at(i), slot_at(i) + 21600s)``: lower closed, upper open."""

    assert driver.classify_launch(0, observed_at).disposition == disposition


# ---------------------------------------------------------------------------
# Falsification 1: a pre-slot launch rejects before any source open
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("observed_at", ("2026-01-01T00:00:00.000000Z", BEFORE_SLOT_ZERO_AT))
def test_a_pre_slot_launch_is_refused_before_the_boundary_is_consulted(
    driver: Any, tmp_path: Path, observed_at: str
) -> None:
    """``POLICY.md``: "An invocation before ``slot_at`` rejects before source access."

    The refusal is proven by what did *not* happen: the campaign boundary was
    never asked for a ledger, a locator, or an observation, and the work
    directory gained nothing at all.
    """

    work = tmp_path / "work"
    work.mkdir()
    campaign = _campaign_class(driver)()
    clock = _clock_class(driver)(observed_at)

    outcome = driver.run_slot(
        work_dir=work, slot_index=0, campaign=campaign, clock=clock
    )

    assert outcome.status == driver.STATUS_PRE_SLOT_REFUSED
    assert outcome.exit_code == driver.EXIT_PRE_SLOT_REFUSED
    assert outcome.source_open_count == 0
    assert outcome.ledger_entry_path is None
    # Nothing was asked of the boundary, so nothing could have been opened.
    assert campaign.calls == []
    assert campaign.source_opens == 0
    # And nothing was created: no ledger, no markers, no staging root.
    assert list(work.iterdir()) == []


def test_the_pre_slot_refusal_holds_one_microsecond_before_the_slot(
    driver: Any, tmp_path: Path
) -> None:
    """The boundary between refusal and execution is exactly ``slot_at``."""

    work = tmp_path / "work"
    work.mkdir()
    campaign_class = _campaign_class(driver)
    clock_class = _clock_class(driver)

    refused = driver.run_slot(
        work_dir=work,
        slot_index=0,
        campaign=campaign_class(),
        clock=clock_class(BEFORE_SLOT_ZERO_AT),
    )
    assert refused.status == driver.STATUS_PRE_SLOT_REFUSED
    assert list(work.iterdir()) == []

    # One microsecond later the driver does consult the boundary, which proves
    # the refusal above was the schedule and not an unrelated failure.
    admitted = campaign_class()
    with pytest.raises(AssertionError):
        driver.run_slot(
            work_dir=work,
            slot_index=0,
            campaign=admitted,
            clock=clock_class(SLOT_ZERO_AT),
        )
    assert admitted.calls == ["open_ledger"]


def test_a_pre_slot_launch_of_a_later_slot_is_refused_while_that_slot_is_open(
    driver: Any, tmp_path: Path
) -> None:
    """Being inside slot 0 grants nothing to slot 1."""

    work = tmp_path / "work"
    work.mkdir()
    outcome = driver.run_slot(
        work_dir=work,
        slot_index=1,
        campaign=_campaign_class(driver)(),
        clock=_clock_class(driver)(IN_WINDOW_AT),
    )
    assert outcome.status == driver.STATUS_PRE_SLOT_REFUSED
    assert outcome.launch.scheduled_at == SLOT_ONE_AT
    assert list(work.iterdir()) == []


# ---------------------------------------------------------------------------
# Falsification 2: a launch at or after the grace deadline is terminal, and
# records that without any source access
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "observed_at", (SLOT_ZERO_GRACE_AT, "2026-08-17T06:00:00.000001Z", "2026-09-20T00:00:00.000000Z")
)
def test_a_late_launch_appends_the_missed_marker_without_source_access(
    driver: Any, tmp_path: Path, observed_at: str
) -> None:
    """``POLICY.md``: the missed marker "is recorded without source access".

    It is also terminal: "no later probe or seal is authorized, because the
    unobserved population might already have passed."
    """

    work = tmp_path / "work"
    directory, state = _genesis_ledger(driver, work)
    campaign = _campaign_class(driver)(state)

    outcome = driver.run_slot(
        work_dir=work,
        slot_index=0,
        campaign=campaign,
        clock=_clock_class(driver)(observed_at),
    )

    assert outcome.status == driver.STATUS_MISSED
    assert outcome.exit_code == driver.EXIT_TERMINAL_SCHEDULE_INTEGRITY_FAILURE
    assert outcome.terminal_status == "terminal-schedule-integrity-failure"
    assert outcome.source_open_count == 0
    # The ledger was rebuilt and the marker appended; nothing else was asked.
    assert campaign.calls == ["open_ledger"]
    assert campaign.source_opens == 0

    entry = _entry(directory, 1)
    assert entry["entry_kind"] == "missed-slot"
    assert entry["slot_index_or_null"] == 0
    assert outcome.ledger_entry_path == directory / "00000001.json"
    assert sorted(item.name for item in directory.iterdir()) == [
        "00000000.json",
        "00000001.json",
    ]
    # No snapshot staging root and no marker directory were ever created.
    assert sorted(item.name for item in work.iterdir()) == [driver.LEDGER_DIRECTORY]


def test_the_missed_marker_states_zero_source_opens_and_the_frozen_slot_times(
    driver: Any, tmp_path: Path
) -> None:
    """``source_open_count`` is the literal zero, never an observation."""

    work = tmp_path / "work"
    _directory, state = _genesis_ledger(driver, work)
    launch = driver.classify_launch(0, SLOT_ZERO_GRACE_AT)
    marker = json.loads(
        driver.build_missed_slot_marker(state, launch).decode("utf-8")
    )

    assert set(marker) == set(driver.accrual.MISSED_SLOT_FIELDS)
    assert marker["receipt_kind"] == "missed-slot"
    assert marker["status"] == "terminal-schedule-integrity-failure"
    assert marker["source_open_count"] == 0
    assert type(marker["source_open_count"]) is int
    assert marker["slot_index"] == 0
    assert marker["scheduled_at"] == SLOT_ZERO_AT
    assert marker["grace_deadline_at"] == SLOT_ZERO_GRACE_AT
    assert marker["recorded_at"] == SLOT_ZERO_GRACE_AT
    assert marker["previous_ledger_entry_sha256"] == state.head


def test_a_missed_slot_forecloses_every_later_invocation(
    driver: Any, tmp_path: Path
) -> None:
    """"There is no late launch, backfill, catch-up, or discretionary skip."""

    work = tmp_path / "work"
    directory, state = _genesis_ledger(driver, work)
    campaign_class = _campaign_class(driver)
    clock_class = _clock_class(driver)

    missed = driver.run_slot(
        work_dir=work,
        slot_index=0,
        campaign=campaign_class(state),
        clock=clock_class(SLOT_ZERO_GRACE_AT),
    )
    assert missed.status == driver.STATUS_MISSED

    terminal = driver.accrual.append_missed_slot(
        state, driver.build_missed_slot_marker(state, missed.launch)
    )
    assert terminal.terminal_status == "terminal-schedule-integrity-failure"
    assert terminal.next_slot_index is None
    frozen = _ledger_bytes(directory)

    # Slot 1 is a real, open window -- and it is still refused, because the
    # terminal state, not the calendar, is what forecloses it.
    later = campaign_class(terminal)
    outcome = driver.run_slot(
        work_dir=work,
        slot_index=1,
        campaign=later,
        clock=clock_class(SLOT_ONE_IN_WINDOW_AT),
    )
    assert outcome.status == driver.STATUS_ALREADY_RESOLVED
    assert outcome.exit_code == driver.EXIT_SLOT_ALREADY_RESOLVED
    assert outcome.terminal_status == "terminal-schedule-integrity-failure"
    assert later.calls == ["open_ledger"]
    assert _ledger_bytes(directory) == frozen


def test_a_late_launch_of_a_later_slot_is_missed_against_that_slot(
    driver: Any, tmp_path: Path
) -> None:
    """The marker binds the slot it missed, not the slot the clock is nearest."""

    work = tmp_path / "work"
    directory, state = _resolved_slot_zero(driver, work)
    campaign = _campaign_class(driver)(state)

    outcome = driver.run_slot(
        work_dir=work,
        slot_index=1,
        campaign=campaign,
        clock=_clock_class(driver)(SLOT_ONE_GRACE_AT),
    )

    assert outcome.status == driver.STATUS_MISSED
    assert outcome.launch.slot_index == 1
    assert outcome.launch.scheduled_at == SLOT_ONE_AT
    assert campaign.calls == ["open_ledger"]
    index = max(int(item.stem) for item in directory.iterdir())
    assert _entry(directory, index)["entry_kind"] == "missed-slot"
    assert _entry(directory, index)["slot_index_or_null"] == 1


def test_the_missed_path_refuses_a_slot_the_ledger_does_not_owe(
    driver: Any, tmp_path: Path
) -> None:
    """A marker may only close the slot the ledger is actually waiting for."""

    work = tmp_path / "work"
    _directory, state = _resolved_slot_zero(driver, work)
    # The ledger owes slot 1; a late launch naming slot 0 must not rewrite it.
    _failure(
        driver,
        driver.run_slot,
        work_dir=work,
        slot_index=0,
        campaign=_campaign_class(driver)(state),
        clock=_clock_class(driver)(SLOT_ZERO_GRACE_AT),
    )


# ---------------------------------------------------------------------------
# Falsification 3: one slot per process -- never a sleep, a poll, or a wait
# ---------------------------------------------------------------------------


def test_the_driver_contains_no_sleep_poll_or_wait_construct(driver: Any) -> None:
    """``POLICY.md:109-110`` forbids a process that stays alive across slots.

    "No process may sleep, poll, or remain alive waiting across slots, and an
    orchestration retry is not a new cadence event."  The compliant shape is an
    external timer that spawns a fresh short-lived process per slot, so this
    module must contain no waiting construct at all.
    """

    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # A loop is how a process would survive from one slot to the next.
    assert [node for node in ast.walk(tree) if isinstance(node, ast.While)] == []
    assert [node for node in ast.walk(tree) if isinstance(node, ast.AsyncFor)] == []

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(
        {"time", "select", "sched", "signal", "asyncio", "threading", "socket"}
    )

    waiting = re.compile(
        r"\b("
        r"sleep|poll|epoll|kqueue|select|monotonic|perf_counter|"
        r"wait|waitpid|join|Timer|set_alarm|alarm"
        r")\b"
    )
    offenders = sorted(
        {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and waiting.search(node.attr)
        }
        | {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and waiting.search(node.id)
        }
    )
    assert offenders == []


def test_an_invocation_never_sleeps_polls_or_waits(
    driver: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Falsify it dynamically too: every waiting primitive is made fatal."""

    import select
    import time

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the slot driver waited")

    for module, name in (
        (time, "sleep"),
        (select, "select"),
        (select, "poll"),
        (os, "wait"),
        (os, "waitpid"),
    ):
        monkeypatch.setattr(module, name, _forbidden, raising=False)

    work = tmp_path / "work"
    work.mkdir()
    refused = driver.run_slot(
        work_dir=work,
        slot_index=0,
        campaign=_campaign_class(driver)(),
        clock=_clock_class(driver)(BEFORE_SLOT_ZERO_AT),
    )
    assert refused.status == driver.STATUS_PRE_SLOT_REFUSED

    missed_work = tmp_path / "missed"
    _directory, state = _genesis_ledger(driver, missed_work)
    missed = driver.run_slot(
        work_dir=missed_work,
        slot_index=0,
        campaign=_campaign_class(driver)(state),
        clock=_clock_class(driver)(SLOT_ZERO_GRACE_AT),
    )
    assert missed.status == driver.STATUS_MISSED


def test_one_invocation_reads_the_clock_once_to_place_itself(
    driver: Any, tmp_path: Path
) -> None:
    """The launch instant is one reading, taken before anything is decided."""

    work = tmp_path / "work"
    work.mkdir()
    clock = _clock_class(driver)(BEFORE_SLOT_ZERO_AT)
    driver.run_slot(
        work_dir=work, slot_index=0, campaign=_campaign_class(driver)(), clock=clock
    )
    assert clock.reads == 1


def test_one_invocation_advances_exactly_one_slot(
    driver: Any, tmp_path: Path
) -> None:
    """A process drives one slot index and appends at most one resolution."""

    work = tmp_path / "work"
    directory, state = _genesis_ledger(driver, work)
    before = _ledger_bytes(directory)

    driver.run_slot(
        work_dir=work,
        slot_index=0,
        campaign=_campaign_class(driver)(state),
        clock=_clock_class(driver)(SLOT_ZERO_GRACE_AT),
    )

    after = _ledger_bytes(directory)
    added = sorted(set(after) - set(before))
    assert added == ["00000001.json"]
    # Every pre-existing entry is byte-identical: the ledger is append-only.
    assert {name: after[name] for name in before} == before
    assert {_entry(directory, 1)["slot_index_or_null"]} == {0}


def test_the_driver_never_derives_a_slot_index_from_the_clock(driver: Any) -> None:
    """The slot is chosen by the caller; the clock only places the launch.

    A driver that inferred "the current slot" from the wall clock would turn a
    late orchestration retry into a different, still-open cadence event.
    """

    parameters = inspect.signature(driver.run_slot).parameters
    assert parameters["slot_index"].default is inspect.Parameter.empty
    assert set(parameters) == {"work_dir", "slot_index", "campaign", "clock"}
    # ``classify_launch`` is total over the launch instant: it never picks one.
    assert list(inspect.signature(driver.classify_launch).parameters) == [
        "slot_index",
        "launched_at",
    ]


# ---------------------------------------------------------------------------
# Falsification 4: the first validator-valid resolution is immutable
# ---------------------------------------------------------------------------


def test_a_second_invocation_does_not_replace_a_valid_resolution(
    driver: Any, tmp_path: Path
) -> None:
    """``POLICY.md``: the first validator-valid slot resolution "is immutable and
    forbids every replacement or second resolution".

    The second invocation is launched squarely inside slot 0's own open window,
    so only the existing resolution -- not the schedule -- can be what stops it.
    """

    work = tmp_path / "work"
    directory, state = _resolved_slot_zero(driver, work)
    frozen = _ledger_bytes(directory)
    resolved_index = max(int(item.stem) for item in directory.iterdir())
    assert _entry(directory, resolved_index)["entry_kind"] == "below-floor"
    assert _entry(directory, resolved_index)["slot_index_or_null"] == 0

    campaign = _campaign_class(driver)(state)
    outcome = driver.run_slot(
        work_dir=work,
        slot_index=0,
        campaign=campaign,
        clock=_clock_class(driver)(IN_WINDOW_AT),
    )

    assert outcome.status == driver.STATUS_ALREADY_RESOLVED
    assert outcome.exit_code == driver.EXIT_SLOT_ALREADY_RESOLVED
    assert outcome.ledger_entry_path is None
    assert outcome.source_open_count == 0
    # It stopped before touching a source, a snapshot, or the runtime.
    assert campaign.calls == ["open_ledger"]
    assert campaign.source_opens == 0
    # And every durable byte is exactly what it was.
    assert _ledger_bytes(directory) == frozen


def test_the_ledger_writer_itself_refuses_to_replace_an_existing_index(
    driver: Any, tmp_path: Path
) -> None:
    """Defence in depth: even without the guard, no byte can be overwritten.

    ``write_ledger_entry_exclusive`` requires the directory to hold the exact
    prior prefix and creates its target with ``O_EXCL``, so a replay of the same
    transition fails rather than rewriting a resolution.
    """

    work = tmp_path / "work"
    directory, state = _resolved_slot_zero(driver, work)
    frozen = _ledger_bytes(directory)

    _failure(driver, driver._persist_head, directory, state)
    assert _ledger_bytes(directory) == frozen


def test_the_driver_proves_the_rebuilt_state_against_the_durable_transcript(
    driver: Any, tmp_path: Path
) -> None:
    """A ``LedgerState`` cannot cross a process boundary, so it is proven.

    The boundary rebuilds the state each invocation.  If what it hands back is
    not exactly the durable transcript -- a dropped entry, an extra file, a
    divergent byte -- the driver refuses before appending anything.
    """

    work = tmp_path / "work"
    directory, state = _resolved_slot_zero(driver, work)
    campaign_class = _campaign_class(driver)
    clock_class = _clock_class(driver)

    head = max(int(item.stem) for item in directory.iterdir())
    truncated = directory / f"{head:08d}.json"
    retained = truncated.read_bytes()
    truncated.unlink()

    _failure(
        driver,
        driver.run_slot,
        work_dir=work,
        slot_index=1,
        campaign=campaign_class(state),
        clock=clock_class(SLOT_ONE_IN_WINDOW_AT),
    )

    # Restore the entry, then corrupt one byte instead of removing the file.
    truncated.write_bytes(retained)
    stray = directory / "00000099.json"
    stray.write_bytes(retained)
    _failure(
        driver,
        driver.run_slot,
        work_dir=work,
        slot_index=1,
        campaign=campaign_class(state),
        clock=clock_class(SLOT_ONE_IN_WINDOW_AT),
    )


# ---------------------------------------------------------------------------
# Falsification 5: a work root inside the frozen namespace is refused
# ---------------------------------------------------------------------------


def _namespace_entries() -> set[str]:
    return {item.name for item in NAMESPACE_ROOT.iterdir()}


@pytest.mark.parametrize(
    "relative", ("", "work", "recipe", "segments/work", "recipe/../work")
)
def test_a_namespace_work_dir_is_refused_before_any_write(
    driver: Any, relative: str
) -> None:
    """The frozen namespace is byte-locked; the driver writes nothing into it.

    This mirrors ``require_output_root_outside_namespace`` at
    ``scripts/ap_confirmatory_seal_v4.py:512-524``, including its own distinct
    exit status, so the refusal can never be mistaken for a slot failure.
    """

    before = _namespace_entries()
    candidate = NAMESPACE_ROOT if not relative else NAMESPACE_ROOT / relative

    with pytest.raises(driver.NamespaceRefusal) as caught:
        driver.require_work_dir_outside_namespace(candidate)
    assert caught.value.path == driver.normalized_path(candidate)

    with pytest.raises(driver.NamespaceRefusal):
        driver.run_slot(
            work_dir=candidate,
            slot_index=0,
            campaign=_campaign_class(driver)(),
            clock=_clock_class(driver)(IN_WINDOW_AT),
        )
    assert _namespace_entries() == before


def test_a_symlink_into_the_namespace_is_refused_too(
    driver: Any, tmp_path: Path
) -> None:
    """``realpath`` resolution means an indirect route is refused as well."""

    link = tmp_path / "sneaky"
    link.symlink_to(NAMESPACE_ROOT, target_is_directory=True)
    before = _namespace_entries()
    with pytest.raises(driver.NamespaceRefusal):
        driver.run_slot(
            work_dir=link / "work",
            slot_index=0,
            campaign=_campaign_class(driver)(),
            clock=_clock_class(driver)(IN_WINDOW_AT),
        )
    assert _namespace_entries() == before


def test_a_work_dir_outside_the_namespace_is_accepted(
    driver: Any, tmp_path: Path
) -> None:
    """The refusal is about the namespace, not about work roots in general."""

    accepted = driver.require_work_dir_outside_namespace(tmp_path / "work")
    assert accepted == driver.normalized_path(tmp_path / "work")
    # A sibling of the namespace, sharing its prefix, is not inside it.
    sibling = NAMESPACE_ROOT.parent / "confirmatory-holdout-v4-elsewhere"
    assert driver.require_work_dir_outside_namespace(sibling) == (
        driver.normalized_path(sibling)
    )


def test_the_command_line_maps_a_namespace_work_dir_to_its_own_status() -> None:
    """Exit 3 and a machine-readable code, distinct from every slot failure."""

    before = _namespace_entries()
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and script
        [
            sys.executable,
            "-B",
            os.fspath(SCRIPT),
            "--work-dir",
            os.fspath(NAMESPACE_ROOT / "work"),
            "--slot-index",
            "0",
            "--campaign-module",
            os.fspath(SCRIPT),
        ],
        capture_output=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 3
    payload = json.loads(completed.stdout.decode("utf-8"))
    assert payload["code"] == "namespace_work_dir_refused"
    assert payload["slot_executed"] is False
    assert _namespace_entries() == before


# ---------------------------------------------------------------------------
# Hard constraint: no command line, and no environment, can name an instant
# ---------------------------------------------------------------------------


TIME_ARGUMENT = re.compile(
    r"now|time|clock|date|instant|when|launch|schedul|deadline|grace|slot_at|epoch|utc",
    re.IGNORECASE,
)


def test_the_argparse_surface_exposes_no_time_argument(driver: Any) -> None:
    """A CLI path that can fake the launch instant is a fabrication vector.

    It would let an orchestration retry masquerade as a cadence event and let a
    source open outside its authorized window, both unrecoverable under the
    frozen schedule.  The surface is therefore enumerated exactly.
    """

    parser = driver.build_parser()
    options = sorted(
        option for action in parser._actions for option in action.option_strings
    )
    assert options == [
        "--campaign-module",
        "--help",
        "--slot-index",
        "--work-dir",
        "-h",
    ]

    for action in parser._actions:
        assert not TIME_ARGUMENT.search(action.dest), action.dest
        for option in action.option_strings:
            assert not TIME_ARGUMENT.search(option), option
        assert not TIME_ARGUMENT.search(action.help or "")

    # ``--slot-index`` selects a slot; it can never be a time value.
    namespace = driver.parse_arguments(
        ["--work-dir", "/tmp/w", "--slot-index", "3", "--campaign-module", "/tmp/c.py"]
    )
    assert namespace.slot_index == 3
    assert sorted(vars(namespace)) == ["campaign_module", "slot_index", "work_dir"]


@pytest.mark.parametrize(
    "injected",
    (
        ["--now", IN_WINDOW_AT],
        ["--launched-at", IN_WINDOW_AT],
        ["--clock", IN_WINDOW_AT],
        ["--time", IN_WINDOW_AT],
        ["--date", "2026-08-17"],
        ["--scheduled-at", SLOT_ZERO_AT],
    ),
)
def test_no_command_line_can_name_the_launch_instant(injected: list[str]) -> None:
    """Every spelling of a clock flag is a usage error, not a hidden option."""

    completed = subprocess.run(  # noqa: S603 - fixed interpreter and script
        [
            sys.executable,
            "-B",
            os.fspath(SCRIPT),
            "--work-dir",
            "/tmp/definitely-not-used",
            "--slot-index",
            "0",
            "--campaign-module",
            os.fspath(SCRIPT),
            *injected,
        ],
        capture_output=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 2
    assert b"unrecognized arguments" in completed.stderr


def test_the_module_reads_no_environment_variable(driver: Any) -> None:
    """An environment clock override would be the same vector, spelled quietly."""

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert attributes.isdisjoint({"environ", "getenv", "putenv", "environb"})
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert names.isdisjoint({"environ", "getenv"})


def test_the_clock_seam_exists_only_in_python(driver: Any) -> None:
    """Time is injected by substituting the clock object, never by argv."""

    parameters = inspect.signature(driver.run_slot).parameters
    assert parameters["clock"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["clock"].default is None

    # The real clock takes no argument at all, so nothing can steer it.
    assert list(inspect.signature(driver.DriverClock.now).parameters) == ["self"]
    assert list(inspect.signature(driver.DriverClock.canonical_now).parameters) == [
        "self"
    ]
    assert driver.DriverClock().now().tzinfo is UTC

    # And only a real clock is accepted: an arbitrary stand-in is refused.
    class _NotAClock:
        def now(self) -> datetime:
            return datetime(2026, 8, 17, tzinfo=UTC)

    _failure(driver, driver._require_clock, _NotAClock())
    _failure(driver, driver._require_clock, "2026-08-17T00:00:00.000000Z")


def test_the_driver_names_no_source_path_host_or_transport() -> None:
    """No literal in this module can address a real database.

    The alias mapping is an ``external-untracked-operator-mapping``; the driver
    takes typed observations from a boundary precisely so that no tracked file
    here ever carries an operator mapping.
    """

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert [text for text in literals if re.match(r"/[^/]", text)] == []
    forbidden = re.compile(r"/(home|root|Users|var|etc|opt|srv|mnt|media|tmp)(/|\b)")
    assert [text for text in literals if forbidden.search(text)] == []
    assert [text for text in literals if ".sqlite3" in text] == []

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"socket", "shutil", "ssl", "http", "urllib"})


# ---------------------------------------------------------------------------
# Layout and dispatch wiring
# ---------------------------------------------------------------------------


def test_the_durable_layout_is_ledger_and_markers_under_the_work_dir(
    driver: Any, tmp_path: Path
) -> None:
    """``<work-dir>/ledger/NNNNNNNN.json`` and ``<work-dir>/markers/``.

    The two must stay distinct: ``accrual`` keeps the ledger directory
    exact-member, so a marker written beside the entries would invalidate the
    whole prefix.
    """

    assert (driver.LEDGER_DIRECTORY, driver.MARKER_DIRECTORY) == ("ledger", "markers")
    work = tmp_path / "work"
    directory, state = _genesis_ledger(driver, work)
    assert directory == work / "ledger"
    assert driver.accrual.ledger_entry_filename(0) == "00000000.json"
    assert driver.accrual.ledger_entry_filename(28) == "00000028.json"

    outcome = driver.run_slot(
        work_dir=work,
        slot_index=0,
        campaign=_campaign_class(driver)(state),
        clock=_clock_class(driver)(SLOT_ZERO_GRACE_AT),
    )
    assert outcome.ledger_entry_path.parent == work / "ledger"
    assert outcome.ledger_entry_path.name == "00000001.json"


def test_the_seal_handoff_is_constructed_in_this_process(driver: Any) -> None:
    """``readiness.ready_action`` happens inside the same supervisor invocation.

    ``SealCeremony`` must be built here rather than spawned, because ``accrual``
    derives its capability keys with ``os.urandom`` at import: ``LedgerState``,
    ``DurableSealConsumption`` and ``BlockedSpawnProof`` cannot cross a process
    boundary.  Only the sealer child is a separate process, and only exact bytes
    reach it.
    """

    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    # The driver never re-executes the sealer as a command.
    assert "subprocess" not in {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert driver.seal.SealCeremony is not None

    request = inspect.signature(driver.SealRequest).parameters
    ceremony = inspect.signature(driver.seal.SealCeremony.__init__).parameters
    # Every ceremony input is supplied by the driver's own request object.
    assert {name for name in ceremony if name != "self"} <= set(request)


def _staged_packet(work: Path) -> Path:
    """A synthetic stand-in for the packet a ready pass would seal."""

    publish = work / "packet" / "recipe"
    publish.mkdir(parents=True)
    (publish / "verify.py").write_bytes(b"# synthetic verifier placeholder\n")
    return work / "packet"


def _fake_launcher(driver: Any, work: Path) -> Any:
    class _Launcher:
        source_open_count = 2

        def snapshot_paths(self) -> dict[str, Path]:
            return {alias: work / f"{alias}.sqlite3" for alias in driver.SOURCE_ALIASES}

        def snapshot_fds(self) -> dict[str, int]:
            return {alias: 100 + index for index, alias in enumerate(driver.SOURCE_ALIASES)}

    return _Launcher()


def test_the_seal_handoff_is_dispatched_inside_this_invocation_and_in_grace(
    driver: Any, tmp_path: Path
) -> None:
    """``readiness.ready_action``, dispatched by the supervisor that owns the slot.

    The ceremony itself is proven by the seal launcher's own self-check; what
    this falsifies is the driver's half -- that the handoff is built in this
    process, rooted at ``--work-dir``, and only while grace is still open.
    """

    work = tmp_path / "work"
    directory, state = _resolved_slot_zero(driver, work)
    _staged_packet(work)
    recorded: list[Any] = []

    class _StubCeremony:
        marker_identity = {"sha256": "a" * 64, "bytes": 512}

        def __init__(self, request: Any) -> None:
            recorded.append(request)

        def run(self) -> tuple[int, dict[str, Any]]:
            return driver.seal.EXIT_SEALED, {
                "status": "sealed",
                "sealer_process_launched_at_or_null": SLOT_ONE_IN_WINDOW_AT,
            }

    class _SealingCampaign(_campaign_class(driver)):
        def seal_ceremony(self, request: Any) -> Any:
            self.calls.append("seal_ceremony")
            return _StubCeremony(request)

        def observe_active_state(self) -> Any:
            return ("services", "binding")

    campaign = _SealingCampaign(state)
    launch = driver.classify_launch(1, SLOT_ONE_IN_WINDOW_AT)
    outcome = driver._run_seal_handoff(
        work=work,
        state=state,
        computation=object(),
        launch=launch,
        campaign=campaign,
        clock=_clock_class(driver)(SLOT_ONE_IN_WINDOW_AT),
        launcher=_fake_launcher(driver, work),
        steps=["classify-launch"],
    )

    assert outcome.status == driver.STATUS_SEAL_HANDOFF
    assert outcome.exit_code == driver.EXIT_SLOT_RESOLVED
    assert outcome.seal_consumption_marker_sha256 == "a" * 64
    assert outcome.sealer_process_launched_at == SLOT_ONE_IN_WINDOW_AT
    assert "seal-handoff" in outcome.steps

    (request,) = recorded
    # Every durable root stays under --work-dir, and the ledger the ceremony
    # appends to is the one this slot has been proving all along.
    assert request.seal_root == work
    assert request.draft_dir == work / "draft"
    assert request.publish_dir == work / "packet"
    assert request.handoff_path == work / "handoff.json"
    assert request.slot_index == 1
    assert request.state is state
    assert directory == request.seal_root / driver.LEDGER_DIRECTORY
    # The ceremony reads the driver's clock, never a second time source.
    assert isinstance(request.clock, driver.seal.SealClock)
    ceremony_now = request.clock.now()
    assert driver.runtime.parse_utc(
        SLOT_ONE_AT, receipt=True
    ) <= ceremony_now < driver.runtime.parse_utc(SLOT_ONE_GRACE_AT, receipt=True)


def test_the_seal_handoff_is_refused_once_grace_has_closed(
    driver: Any, tmp_path: Path
) -> None:
    """"The launch ... and any ready-to-seal handoff must all complete inside
    the window."  A handoff that cannot start in grace never consumes authority.
    """

    work = tmp_path / "work"
    _directory, state = _resolved_slot_zero(driver, work)
    _staged_packet(work)

    class _NeverBuilt(_campaign_class(driver)):
        def seal_ceremony(self, request: Any) -> Any:
            raise AssertionError("seal authority was consumed after grace closed")

    _failure(
        driver,
        driver._run_seal_handoff,
        work=work,
        state=state,
        computation=object(),
        launch=driver.classify_launch(1, SLOT_ONE_IN_WINDOW_AT),
        campaign=_NeverBuilt(state),
        clock=_clock_class(driver)(SLOT_ONE_GRACE_AT),
        launcher=_fake_launcher(driver, work),
        steps=[],
    )


def test_missing_seal_inputs_are_proven_before_any_source_is_opened(
    driver: Any, tmp_path: Path
) -> None:
    """A ready pass can occur at any slot, so the packet must already be staged.

    ``POLICY.md`` makes a structural failure after source open terminal, so
    discovering an unstaged packet mid-window would cost the whole campaign.
    The driver therefore proves it while the slot is still recoverable -- before
    the snapshot launcher exists and before a locator is ever requested.
    """

    work = tmp_path / "work"
    _directory, state = _resolved_slot_zero(driver, work)

    class _Watchful(_campaign_class(driver)):
        pass

    campaign = _Watchful(state)
    frozen = _ledger_bytes(_directory)
    _failure(
        driver,
        driver.run_slot,
        work_dir=work,
        slot_index=1,
        campaign=campaign,
        clock=_clock_class(driver)(SLOT_ONE_IN_WINDOW_AT),
    )
    # It stopped before asking for a locator, a database identity, or the dev
    # reference -- i.e. before anything could have been opened.  Crucially it
    # also stopped before the attempt marker, so the slot stays recoverable
    # instead of being terminalized for unprovable progress.
    assert campaign.calls == ["open_ledger"]
    assert campaign.source_opens == 0
    assert not (work / "snapshots").exists()
    assert _ledger_bytes(_directory) == frozen

    # With the packet staged, the same launch gets past the check and only then
    # reaches the boundary that would address a source.
    _staged_packet(work)
    staged = _Watchful(state)
    with pytest.raises(AssertionError, match="address a source"):
        driver.run_slot(
            work_dir=work,
            slot_index=1,
            campaign=staged,
            clock=_clock_class(driver)(SLOT_ONE_IN_WINDOW_AT),
        )
    assert staged.calls[-1] == "source_locators"


def test_the_boundary_owns_every_operator_private_observation(driver: Any) -> None:
    """The driver stores no default for anything that could name a source."""

    boundary = driver.CampaignBoundary()
    for method, arguments in (
        ("open_ledger", (Path("/nowhere"),)),
        ("runtime_observer_identity", ()),
        ("source_locators", ()),
        ("observe_database_identity", ("local",)),
        ("observe_active_state", ()),
        ("open_dev_reference", ()),
    ):
        with pytest.raises(NotImplementedError):
            getattr(boundary, method)(*arguments)


def test_a_bare_command_line_is_a_usage_error_not_a_default_slot() -> None:
    """Nothing about a slot may be implicit: both selectors are required."""

    for arguments in ([], ["--slot-index", "0"], ["--work-dir", "/tmp/w"]):
        completed = subprocess.run(  # noqa: S603 - fixed interpreter and script
            [sys.executable, "-B", os.fspath(SCRIPT), *arguments],
            capture_output=True,
            check=False,
            timeout=300,
        )
        assert completed.returncode == 2
        assert b"required" in completed.stderr
