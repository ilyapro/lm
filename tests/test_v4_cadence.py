"""Tests for the confirmatory-v4 one-shot-per-slot cadence renderer and verifier.

Nothing here activates anything.  The campaign is preregistered source-blind,
slot 0 does not open until the frozen anchor, and arming consumes one-shot
authorities whose first missed slot is terminal for the whole namespace, so no
test may install a unit, enable or start a timer, replace a crontab, or reach
``systemd-run``.  Every render in this file goes to a pytest temporary
directory, and every assertion is made against file *content*.

The claims are falsified rather than demonstrated.  Each load-bearing property
-- the schedule is exactly the 29 frozen instants, each firing maps to its own
slot index, every firing lands inside the grace window, the unit spawns a fresh
process per slot with nothing sleeping or surviving it, and the instants are
derived rather than tabled -- has a test that fails if the property is removed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "v4_cadence.py"
TRACKED_RENDER = REPO_ROOT / "scripts" / "v4-cadence"
RUNBOOK = REPO_ROOT / "docs" / "v4-campaign-runbook.md"
NAMESPACE_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "confirmatory-holdout-v4"
)

# The schedule this node exists to reproduce, written out independently of the
# renderer.  If the renderer ever consults a table instead of the frozen
# constants, these literals are what it must still match.
FIRST_INSTANT = "2026-08-17T00:00:00Z"
LAST_INSTANT = "2026-09-14T00:00:00Z"
HORIZON_INSTANT = "2026-09-14T06:00:00Z"
CADENCE_SECONDS = 86_400
GRACE_SECONDS = 21_600
SLOT_COUNT = 29

EXPECTED_INSTANTS = tuple(
    (datetime(2026, 8, 17, tzinfo=UTC) + timedelta(seconds=CADENCE_SECONDS * index))
    for index in range(SLOT_COUNT)
)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cadence = _load("v4_cadence_under_test", SCRIPT)


# --------------------------------------------------------------------------
# Fixtures: every render lands in a temporary directory
# --------------------------------------------------------------------------

EXAMPLE_INPUTS = {
    "repo_root": "/nonexistent/EXAMPLE-REPO-ROOT",
    "work_dir": "/nonexistent/EXAMPLE-WORK-DIR",
    "campaign_module": "/nonexistent/EXAMPLE-PRIVATE/campaign.py",
    "cron_timezone": "UTC",
}


def make_request(**overrides: Any) -> Any:
    return cadence.build_request(**{**EXAMPLE_INPUTS, **overrides})


@pytest.fixture
def rendered(tmp_path: Path) -> Path:
    """A complete rendered set in a throwaway directory."""

    out = tmp_path / "cadence"
    cadence.write_render(out, cadence.render(make_request()))
    return out


def timer_path(unit_dir: Path, slot_index: int) -> Path:
    return unit_dir / "systemd" / f"{cadence.UNIT_PREFIX}@{slot_index}.timer"


def service_path(unit_dir: Path) -> Path:
    return unit_dir / "systemd" / f"{cadence.UNIT_PREFIX}@.service"


def crontab_path(unit_dir: Path) -> Path:
    return unit_dir / "cron" / cadence.CRON_FILE_NAME


def timer_instants(unit_dir: Path) -> dict[int, datetime]:
    """Every ``OnCalendar`` in the rendered timers, keyed by instance name."""

    found: dict[int, datetime] = {}
    for path in (unit_dir / "systemd").glob(f"{cadence.UNIT_PREFIX}@*.timer"):
        index = int(path.name[len(cadence.UNIT_PREFIX) + 1 : -len(".timer")])
        match = re.search(r"^OnCalendar=(.+)$", path.read_text(), re.MULTILINE)
        assert match is not None, f"{path.name} has no OnCalendar"
        found[index] = datetime.strptime(
            match.group(1).strip(), "%Y-%m-%d %H:%M:%S UTC"
        ).replace(tzinfo=UTC)
    return found


def cron_entries(unit_dir: Path) -> list[re.Match[str]]:
    matches = []
    for raw in crontab_path(unit_dir).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" in line.split(" ", 1)[0]:
            continue
        match = cadence.CRON_LINE_PATTERN.match(line)
        assert match is not None, f"unparsed crontab entry: {line[:80]}"
        matches.append(match)
    return matches


def mutate(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, f"{path.name} does not contain {old!r}"
    path.write_text(text.replace(old, new, 1))


# --------------------------------------------------------------------------
# The schedule is exactly the 29 frozen instants
# --------------------------------------------------------------------------


def test_rendered_schedule_is_exactly_the_twenty_nine_frozen_instants(
    rendered: Path,
) -> None:
    instants = timer_instants(rendered)
    assert len(instants) == SLOT_COUNT
    assert sorted(instants) == list(range(SLOT_COUNT))
    assert [instants[index] for index in sorted(instants)] == list(EXPECTED_INSTANTS)


def test_schedule_endpoints_match_the_stated_campaign_bounds(rendered: Path) -> None:
    instants = timer_instants(rendered)
    assert instants[0].strftime("%Y-%m-%dT%H:%M:%SZ") == FIRST_INSTANT
    assert instants[SLOT_COUNT - 1].strftime("%Y-%m-%dT%H:%M:%SZ") == LAST_INSTANT


def test_spacing_between_consecutive_firings_is_the_frozen_cadence(
    rendered: Path,
) -> None:
    instants = timer_instants(rendered)
    ordered = [instants[index] for index in sorted(instants)]
    spacings = {
        (later - earlier).total_seconds() for earlier, later in zip(ordered, ordered[1:])
    }
    assert spacings == {float(CADENCE_SECONDS)}


def test_schedule_report_states_the_frozen_horizon() -> None:
    report = cadence._schedule_report()
    assert report["slot_count"] == SLOT_COUNT
    assert report["anchor_at"] == FIRST_INSTANT
    assert report["final_selection_slot_at"] == LAST_INSTANT
    assert report["absolute_horizon_expires_at"] == HORIZON_INSTANT
    assert report["cadence_seconds"] == CADENCE_SECONDS
    assert report["grace_seconds"] == GRACE_SECONDS


# --------------------------------------------------------------------------
# Each firing maps to the correct slot index
# --------------------------------------------------------------------------


def test_each_timer_activates_the_service_instance_for_its_own_slot(
    rendered: Path,
) -> None:
    for index, instant in timer_instants(rendered).items():
        assert instant == EXPECTED_INSTANTS[index]
        text = timer_path(rendered, index).read_text()
        assert f"Unit={cadence.UNIT_PREFIX}@{index}.service" in text


def test_the_service_takes_its_slot_index_from_the_timer_instance(
    rendered: Path,
) -> None:
    """``%i`` is the instance name, so slot N's timer runs the driver on slot N."""

    exec_start = re.search(
        r"^ExecStart=(.+)$", service_path(rendered).read_text(), re.MULTILINE
    )
    assert exec_start is not None
    assert "--slot-index %i" in exec_start.group(1)
    assert cadence.EXEC_START_PATTERN.match(exec_start.group(1)) is not None


def test_every_cron_entry_pairs_its_instant_with_its_own_slot_index(
    rendered: Path,
) -> None:
    seen: dict[int, datetime] = {}
    for match in cron_entries(rendered):
        index = int(match.group("slot_index"))
        assert index not in seen, f"slot {index} has more than one crontab entry"
        fired_at = datetime(
            2026,
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            tzinfo=UTC,
        )
        seen[index] = fired_at
    assert sorted(seen) == list(range(SLOT_COUNT))
    assert [seen[index] for index in sorted(seen)] == list(EXPECTED_INSTANTS)


def test_cron_guard_pins_each_entry_to_its_own_utc_hour(rendered: Path) -> None:
    for match in cron_entries(rendered):
        index = int(match.group("slot_index"))
        assert match.group("guard") == EXPECTED_INSTANTS[index].strftime("%Y-%m-%dT%H")


def test_cron_entries_convert_into_the_daemon_timezone(tmp_path: Path) -> None:
    """A midnight-UTC slot must be written in local time, not copied verbatim.

    Debian cron ignores CRON_TZ, so on a ``+07`` host an unconverted ``0 0``
    entry would fire seven hours early, the driver would refuse a pre-slot
    launch, and slot 0 would be missed -- which is terminal.
    """

    out = tmp_path / "novosibirsk"
    request = make_request(cron_timezone="Asia/Novosibirsk")
    cadence.write_render(out, cadence.render(request))

    for match in cron_entries(out):
        index = int(match.group("slot_index"))
        assert (match.group("hour"), match.group("minute")) == ("7", "0")
        local = datetime(
            2026,
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            tzinfo=cadence.resolve_timezone("Asia/Novosibirsk"),
        )
        assert local.astimezone(UTC) == EXPECTED_INSTANTS[index]
    assert cadence.verify(out) == []


# --------------------------------------------------------------------------
# Every firing lands inside the grace window
# --------------------------------------------------------------------------


def test_systemd_worst_case_firing_lands_inside_the_grace_window(
    rendered: Path,
) -> None:
    """Scheduled instant plus the unit's own maximum deferral, still in-window."""

    for index, instant in timer_instants(rendered).items():
        text = timer_path(rendered, index).read_text()
        accuracy = cadence._duration_seconds(
            re.search(r"^AccuracySec=(.+)$", text, re.MULTILINE).group(1), 60
        )
        delay = cadence._duration_seconds(
            re.search(r"^RandomizedDelaySec=(.+)$", text, re.MULTILINE).group(1), 0
        )
        opens = EXPECTED_INSTANTS[index]
        closes = opens + timedelta(seconds=GRACE_SECONDS)
        assert instant == opens
        assert opens <= instant + timedelta(seconds=accuracy + delay) < closes


def test_cron_guard_admits_only_instants_inside_the_grace_window(
    rendered: Path,
) -> None:
    tolerance = timedelta(seconds=cadence.CRON_GUARD_TOLERANCE_SECONDS)
    assert tolerance < timedelta(seconds=GRACE_SECONDS)
    for match in cron_entries(rendered):
        opens = EXPECTED_INSTANTS[int(match.group("slot_index"))]
        assert opens + tolerance < opens + timedelta(seconds=GRACE_SECONDS)


def test_the_service_run_is_bounded_by_the_grace_window(rendered: Path) -> None:
    assert f"TimeoutStartSec={GRACE_SECONDS}" in service_path(rendered).read_text()


# --------------------------------------------------------------------------
# A fresh process per slot: nothing sleeps, restarts, or survives
# --------------------------------------------------------------------------


def test_the_service_is_a_oneshot_that_does_not_outlive_its_slot(
    rendered: Path,
) -> None:
    sections = cadence.parse_unit(service_path(rendered).read_text())
    assert cadence.unit_value(sections, "Service", "Type") == "oneshot"
    assert cadence.unit_value(sections, "Service", "RemainAfterExit") == "no"
    assert cadence.unit_value(sections, "Service", "Restart") == "no"


def test_no_remain_after_exit_anywhere_in_the_rendered_set(rendered: Path) -> None:
    for path in sorted(rendered.rglob("*")):
        if not path.is_file():
            continue
        body = path.read_text()
        assert not re.search(r"^RemainAfterExit=(?!no$)", body, re.MULTILINE)
        assert not re.search(r"^RemainAfterElapse=(?!no$)", body, re.MULTILINE)


def test_nothing_in_the_rendered_set_sleeps_polls_or_wraps_the_driver(
    rendered: Path,
) -> None:
    for path in sorted(rendered.rglob("*")):
        if not path.is_file():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            for token in cadence.FORBIDDEN_COMMAND_TOKENS:
                assert not re.search(rf"(?<![\w-]){token}(?![\w-])", line.lower()), (
                    f"{path.name} contains a waiting construct: {line[:80]}"
                )


def test_the_exec_start_is_a_bare_driver_invocation_not_a_shell(
    rendered: Path,
) -> None:
    exec_start = re.search(
        r"^ExecStart=(.+)$", service_path(rendered).read_text(), re.MULTILINE
    ).group(1)
    for metacharacter in (";", "&&", "||", "|", "$(", "`", "/bin/sh", "/bin/bash", "-c "):
        assert metacharacter not in exec_start
    assert exec_start.startswith("/")
    assert " -B " in exec_start
    assert exec_start.endswith("campaign.py")


def test_no_long_lived_daemon_is_declared(rendered: Path) -> None:
    for path in sorted((rendered / "systemd").glob("*")):
        body = path.read_text()
        assert not re.search(
            r"^Type=(simple|exec|forking|notify|notify-reload|dbus|idle)$",
            body,
            re.MULTILINE,
        )
        assert not re.search(r"^(BusName|PIDFile|WatchdogSec)=", body, re.MULTILINE)
        assert "WantedBy=default.target" not in body
        assert "WantedBy=multi-user.target" not in body


def test_no_relative_or_repeating_trigger_is_declared(rendered: Path) -> None:
    for index in range(SLOT_COUNT):
        body = timer_path(rendered, index).read_text()
        for directive in cadence.FORBIDDEN_TIMER_DIRECTIVES:
            assert f"{directive}=" not in body
        assert body.count("OnCalendar=") == 1


def test_a_missed_firing_is_never_replayed_as_backfill(rendered: Path) -> None:
    for index in range(SLOT_COUNT):
        sections = cadence.parse_unit(timer_path(rendered, index).read_text())
        assert cadence.unit_value(sections, "Timer", "Persistent") == "false"
        assert cadence.unit_value(sections, "Timer", "RemainAfterElapse") == "no"


def test_a_manual_start_is_refused_because_a_retry_is_not_a_cadence_event(
    rendered: Path,
) -> None:
    sections = cadence.parse_unit(service_path(rendered).read_text())
    assert cadence.unit_value(sections, "Unit", "RefuseManualStart") == "yes"


# --------------------------------------------------------------------------
# The instants are derived from the frozen constants, not tabled
# --------------------------------------------------------------------------


def test_moving_the_frozen_anchor_moves_every_rendered_firing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The falsification test for "derived, never tabled".

    If the renderer held a literal schedule, a changed anchor would leave the
    render untouched.  Here the whole set must shift with it.
    """

    monkeypatch.setattr(cadence.runtime, "ANCHOR_AT", "2027-03-01T00:00:00Z")
    monkeypatch.setattr(cadence.runtime, "ABSOLUTE_HORIZON_AT", "2027-03-29T06:00:00Z")

    out = tmp_path / "moved"
    cadence.write_render(out, cadence.render(make_request()))
    instants = timer_instants(out)

    assert len(instants) == SLOT_COUNT
    assert instants[0] == datetime(2027, 3, 1, tzinfo=UTC)
    assert instants[SLOT_COUNT - 1] == datetime(2027, 3, 29, tzinfo=UTC)
    assert not any(instant in EXPECTED_INSTANTS for instant in instants.values())
    assert FIRST_INSTANT not in crontab_path(out).read_text()


def test_changing_the_frozen_cadence_changes_the_rendered_spacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cadence.runtime, "CADENCE_SECONDS", 2 * CADENCE_SECONDS)
    monkeypatch.setattr(cadence.runtime, "ABSOLUTE_HORIZON_AT", "2026-10-12T06:00:00Z")

    out = tmp_path / "stretched"
    cadence.write_render(out, cadence.render(make_request()))
    instants = timer_instants(out)
    ordered = [instants[index] for index in sorted(instants)]
    spacings = {
        (later - earlier).total_seconds() for earlier, later in zip(ordered, ordered[1:])
    }
    assert spacings == {float(2 * CADENCE_SECONDS)}


def test_the_renderer_source_contains_no_schedule_literals() -> None:
    """No date, cadence, or grace literal may appear in the renderer at all."""

    source = SCRIPT.read_text()
    for literal in ("2026", "2027", "08-17", "09-14", "86400", "86_400", "21600", "21_600"):
        assert literal not in source, f"{literal!r} is tabled in {SCRIPT.name}"


def test_the_renderer_refuses_a_schedule_that_breaks_a_frozen_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A constant moved without its horizon is a refusal, not a quiet render."""

    monkeypatch.setattr(cadence.runtime, "ANCHOR_AT", "2027-03-01T00:00:00Z")
    with pytest.raises(cadence.CadenceError, match="horizon"):
        cadence.slot_firings()

    # And the frozen runtime refuses it on its own terms too.
    with pytest.raises(cadence.runtime.IntegrityFailure):
        cadence.runtime.absolute_horizon_at()


# --------------------------------------------------------------------------
# The verifier catches what a hand edit could break
# --------------------------------------------------------------------------


def test_a_freshly_rendered_set_verifies_clean(rendered: Path) -> None:
    assert cadence.verify(rendered) == []


@pytest.mark.parametrize(
    ("description", "edit", "expected"),
    [
        (
            "a moved firing",
            lambda d: mutate(
                timer_path(d, 3), "OnCalendar=2026-08-20", "OnCalendar=2026-08-21"
            ),
            "slot 3 opens",
        ),
        (
            "a firing pushed past its grace window",
            lambda d: mutate(timer_path(d, 5), "AccuracySec=1s", "AccuracySec=7h"),
            "outside the window",
        ),
        (
            "a timer wired to another slot's service",
            lambda d: mutate(
                timer_path(d, 7),
                "Unit=ap-confirmatory-v4-slot@7.service",
                "Unit=ap-confirmatory-v4-slot@8.service",
            ),
            "activates",
        ),
        (
            "backfill of a missed slot",
            lambda d: mutate(timer_path(d, 2), "Persistent=false", "Persistent=true"),
            "backfill",
        ),
        (
            "a timer that survives its own firing",
            lambda d: mutate(
                timer_path(d, 4), "RemainAfterElapse=no", "RemainAfterElapse=yes"
            ),
            "re-fire",
        ),
        (
            "a relative trigger",
            lambda d: mutate(
                timer_path(d, 6), "[Timer]", "[Timer]\nOnUnitActiveSec=86400"
            ),
            "relative trigger",
        ),
        (
            "a deleted timer",
            lambda d: timer_path(d, 11).unlink(),
            "no timer for slot indices [11]",
        ),
        (
            "a service held open after its slot",
            lambda d: mutate(
                service_path(d), "RemainAfterExit=no", "RemainAfterExit=yes"
            ),
            "outlive the slot",
        ),
        (
            "a restarting service",
            lambda d: mutate(service_path(d), "Restart=no", "Restart=on-failure"),
            "retry a cadence event",
        ),
        (
            "a long-lived service type",
            lambda d: mutate(service_path(d), "Type=oneshot", "Type=simple"),
            "one process",
        ),
        (
            "a sleeping wrapper",
            lambda d: mutate(
                service_path(d),
                "ExecStart=/usr/bin/python3 -B",
                "ExecStartPre=/bin/sleep 30\nExecStart=/usr/bin/python3 -B",
            ),
            "outside the driver",
        ),
        (
            "an unbounded run",
            lambda d: mutate(
                service_path(d),
                f"TimeoutStartSec={GRACE_SECONDS}",
                "TimeoutStartSec=infinity",
            ),
            "grace window",
        ),
        (
            "a work root inside the frozen namespace",
            lambda d: mutate(
                service_path(d),
                "--work-dir /nonexistent/EXAMPLE-WORK-DIR",
                f"--work-dir {NAMESPACE_ROOT}/work",
            ),
            "frozen namespace",
        ),
        (
            "a crontab whose timezone no longer matches its entries",
            lambda d: mutate(
                crontab_path(d),
                "# cadence-timezone: UTC",
                "# cadence-timezone: Asia/Novosibirsk",
            ),
            "but the slot opens at",
        ),
        (
            "a crontab guard that no longer pins the year",
            lambda d: mutate(crontab_path(d), '= "2026-08-19T00"', '= "2027-08-19T00"'),
            "guard",
        ),
        (
            "a crontab entry for the wrong slot",
            lambda d: mutate(crontab_path(d), "--slot-index 9 ", "--slot-index 10 "),
            "slot 10 fires at",
        ),
        (
            "a deleted crontab entry",
            lambda d: mutate(
                crontab_path(d),
                '0 0 30 8 * [ "$(date -u +\\%Y-\\%m-\\%dT\\%H)" = "2026-08-30T00" ] && exec ',
                "# removed ",
            ),
            "no entry for slot indices [13]",
        ),
    ],
)
def test_verify_rejects(
    rendered: Path, description: str, edit: Callable[[Path], None], expected: str
) -> None:
    edit(rendered)
    findings = cadence.verify(rendered)
    assert findings, f"verify accepted {description}"
    assert any(expected in finding for finding in findings), (
        f"{description}: expected {expected!r} in {findings}"
    )


def test_verify_rejects_a_missing_service(rendered: Path) -> None:
    service_path(rendered).unlink()
    assert any("missing template service" in f for f in cadence.verify(rendered))


def test_verify_rejects_a_missing_crontab(rendered: Path) -> None:
    crontab_path(rendered).unlink()
    assert any("missing crontab fallback" in f for f in cadence.verify(rendered))


# --------------------------------------------------------------------------
# Render-time refusals
# --------------------------------------------------------------------------


def test_render_refuses_a_work_root_inside_the_frozen_namespace() -> None:
    with pytest.raises(cadence.CadenceError, match="frozen namespace"):
        make_request(work_dir=os.fspath(NAMESPACE_ROOT / "work"))
    with pytest.raises(cadence.CadenceError, match="frozen namespace"):
        make_request(work_dir=os.fspath(NAMESPACE_ROOT))


def test_render_requires_absolute_whitespace_free_paths() -> None:
    with pytest.raises(cadence.CadenceError, match="absolute"):
        make_request(work_dir="relative/work")
    with pytest.raises(cadence.CadenceError, match="whitespace"):
        make_request(work_dir="/tmp/work dir")


def test_render_refuses_an_unknown_cron_timezone() -> None:
    with pytest.raises(cadence.CadenceError, match="unknown timezone"):
        make_request(cron_timezone="Mars/Olympus_Mons")


def test_render_is_deterministic(tmp_path: Path) -> None:
    first = cadence.render(make_request())
    second = cadence.render(make_request())
    assert first == second
    assert cadence.render_digest(first) == cadence.render_digest(second)


def test_render_replaces_a_previous_set_rather_than_merging(tmp_path: Path) -> None:
    out = tmp_path / "cadence"
    cadence.write_render(out, cadence.render(make_request()))
    stale = out / "systemd" / f"{cadence.UNIT_PREFIX}@99.timer"
    stale.write_text("[Timer]\nOnCalendar=2026-08-17 00:00:00 UTC\n")
    cadence.write_render(out, cadence.render(make_request()))
    assert not stale.exists()
    assert cadence.verify(out) == []


# --------------------------------------------------------------------------
# The tracked example render cannot drift from the renderer
# --------------------------------------------------------------------------


def test_the_tracked_example_matches_a_fresh_render(tmp_path: Path) -> None:
    fresh = cadence.render(
        cadence.build_request(
            repo_root=cadence.EXAMPLE_REPO_ROOT,
            work_dir=cadence.EXAMPLE_WORK_DIR,
            campaign_module=cadence.EXAMPLE_CAMPAIGN_MODULE,
            cron_timezone=cadence.EXAMPLE_CRON_TIMEZONE,
        )
    )
    for relative, text in fresh.items():
        tracked = TRACKED_RENDER / relative
        assert tracked.is_file(), f"{relative} is missing from the tracked render"
        assert tracked.read_text() == text, f"{relative} has drifted from the renderer"

    on_disk = {
        os.fspath(path.relative_to(TRACKED_RENDER))
        for path in TRACKED_RENDER.rglob("*")
        if path.is_file() and path.suffix != ".md"
    }
    assert on_disk == set(fresh)


def test_the_tracked_example_verifies_clean() -> None:
    assert cadence.verify(TRACKED_RENDER) == []


def test_the_tracked_example_cannot_be_armed_by_accident() -> None:
    """Its paths must not resolve, so preflight refuses before slot 0 ever opens."""

    for path in (
        cadence.EXAMPLE_REPO_ROOT,
        cadence.EXAMPLE_WORK_DIR,
        cadence.EXAMPLE_CAMPAIGN_MODULE,
    ):
        assert not Path(path).exists()
    findings = cadence.preflight(TRACKED_RENDER, clock=FrozenClock(BEFORE_SLOT_ZERO))
    assert any("does not exist" in finding for finding in findings)


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------

BEFORE_SLOT_ZERO = datetime(2026, 8, 16, 12, tzinfo=UTC)
AFTER_SLOT_ZERO = datetime(2026, 8, 17, 0, 0, 1, tzinfo=UTC)


class FrozenClock(cadence.PreflightClock):
    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self) -> datetime:
        return self._moment


@pytest.fixture
def armable(tmp_path: Path) -> Path:
    """A render whose paths all resolve, so preflight has nothing to complain about."""

    work = tmp_path / "work"
    work.mkdir()
    module = tmp_path / "private" / "campaign.py"
    module.parent.mkdir()
    module.write_text("")

    out = tmp_path / "cadence"
    request = cadence.build_request(
        repo_root=os.fspath(REPO_ROOT),
        work_dir=os.fspath(work),
        campaign_module=os.fspath(module),
        python=sys.executable,
        cron_timezone=cadence.host_timezone_name() or "UTC",
    )
    cadence.write_render(out, cadence.render(request))
    return out


def test_preflight_passes_on_a_resolvable_set_before_slot_zero(armable: Path) -> None:
    assert cadence.preflight(armable, clock=FrozenClock(BEFORE_SLOT_ZERO)) == []


def test_preflight_refuses_once_slot_zero_has_opened(armable: Path) -> None:
    findings = cadence.preflight(armable, clock=FrozenClock(AFTER_SLOT_ZERO))
    assert any("can no longer be armed" in finding for finding in findings)
    assert any("terminal" in finding for finding in findings)


def test_preflight_flags_a_crontab_rendered_for_another_timezone(
    armable: Path,
) -> None:
    host = cadence.host_timezone_name() or "UTC"
    other = "UTC" if host != "UTC" else "Asia/Novosibirsk"
    mutate(
        crontab_path(armable), f"# cadence-timezone: {host}", f"# cadence-timezone: {other}"
    )
    findings = cadence.preflight(armable, clock=FrozenClock(BEFORE_SLOT_ZERO))
    assert any("runs cron in" in finding for finding in findings)


def test_preflight_flags_a_missing_work_directory(armable: Path) -> None:
    mutate(service_path(armable), "--work-dir ", "--work-dir /nonexistent/gone")
    findings = cadence.preflight(armable, clock=FrozenClock(BEFORE_SLOT_ZERO))
    assert any("work directory" in finding for finding in findings)


def test_preflight_has_no_clock_flag() -> None:
    """The clock is a Python seam only, exactly as the driver keeps it."""

    parser = cadence.build_parser()
    rendered_help = parser.format_help()
    for flag in ("--now", "--clock", "--date", "--launched-at", "--slot-at"):
        assert flag not in rendered_help


# --------------------------------------------------------------------------
# This tool cannot arm anything
# --------------------------------------------------------------------------


def test_the_command_line_has_no_activation_subcommand() -> None:
    help_text = cadence.build_parser().format_help()
    for verb in ("install", "enable", "arm", "activate", "start", "deploy"):
        assert f" {verb} " not in help_text
    assert cadence.main(["render", "--help"]) == cadence.EXIT_OK


def test_the_renderer_cannot_execute_anything() -> None:
    """No subprocess surface at all: it writes files and reads files.

    With no way to spawn a process, the renderer cannot install a unit, enable a
    timer, or replace a crontab even if some future edit tried to.
    """

    source = SCRIPT.read_text()
    for forbidden in (
        "import subprocess",
        "subprocess.",
        "os.system",
        "os.exec",
        "os.spawn",
        "os.popen",
        "os.fork",
        "runpy",
    ):
        assert forbidden not in source, f"{SCRIPT.name} references {forbidden}"

    # The scheduler binaries appear only as tokens the verifier rejects, never
    # as something this module invokes.
    for binary in ("systemctl", "systemd-run", "loginctl"):
        for line in source.splitlines():
            if binary in line:
                assert line.strip().startswith(('"', "#")), (
                    f"{SCRIPT.name} mentions {binary} outside the rejection list: {line}"
                )


def test_no_artifact_contains_a_blocked_arming_pattern() -> None:
    """The acceptance contract, enforced over everything this node produces.

    Assembled at run time so the patterns are not literals in this file either.
    """

    blocked = (
        "systemctl --user " + "enable",
        "systemctl --user " + "start",
        "crontab" + " -",
    )
    artifacts = [SCRIPT, RUNBOOK, Path(__file__), *sorted(TRACKED_RENDER.rglob("*"))]
    for path in artifacts:
        if not path.is_file():
            continue
        body = path.read_text()
        for pattern in blocked:
            assert pattern not in body, f"{path.name} contains a blocked pattern"


def test_the_cli_renders_and_verifies_without_touching_the_repository(
    tmp_path: Path,
) -> None:
    """End-to-end through ``__main__``, and it must leave no byte behind."""

    out = tmp_path / "cli"
    before = {path for path in (REPO_ROOT / "scripts").rglob("*") if path.is_file()}

    render = subprocess.run(
        [
            sys.executable,
            "-B",
            os.fspath(SCRIPT),
            "render",
            "--out-dir",
            os.fspath(out),
            "--repo-root",
            EXAMPLE_INPUTS["repo_root"],
            "--work-dir",
            EXAMPLE_INPUTS["work_dir"],
            "--campaign-module",
            EXAMPLE_INPUTS["campaign_module"],
            "--cron-timezone",
            "UTC",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert render.returncode == cadence.EXIT_OK, render.stderr
    report = json.loads(render.stdout)
    assert report["ok"] is True and report["armed"] is False
    assert report["files"] == SLOT_COUNT + 2

    check = subprocess.run(
        [sys.executable, "-B", os.fspath(SCRIPT), "verify", "--unit-dir", os.fspath(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == cadence.EXIT_OK, check.stdout
    assert json.loads(check.stdout)["findings"] == []

    after = {path for path in (REPO_ROOT / "scripts").rglob("*") if path.is_file()}
    assert after == before


def test_the_cli_reports_a_failure_with_a_nonzero_status(
    rendered: Path, tmp_path: Path
) -> None:
    mutate(timer_path(rendered, 0), "OnCalendar=2026-08-17", "OnCalendar=2026-08-18")
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            os.fspath(SCRIPT),
            "verify",
            "--unit-dir",
            os.fspath(rendered),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == cadence.EXIT_FAILED
    assert json.loads(result.stdout)["ok"] is False


# --------------------------------------------------------------------------
# The runbook states what the operator is committing to
# --------------------------------------------------------------------------


def test_the_runbook_states_the_campaign_bounds_and_their_cost() -> None:
    body = RUNBOOK.read_text()
    for required in (
        FIRST_INSTANT,
        "2026-08-17T06:00:00Z",
        LAST_INSTANT,
        HORIZON_INSTANT,
        "--work-dir",
        "--slot-index",
        "ap_confirmatory_slot_v4.py",
        "loginctl enable-linger",
    ):
        assert required in body, f"the runbook does not state {required!r}"
    lowered = body.lower()
    for phrase in ("terminal", "no catch-up", "backfill", "replacement", "outside"):
        assert phrase in lowered, f"the runbook does not state {phrase!r}"


def test_the_runbook_shows_the_exact_per_slot_invocation() -> None:
    body = RUNBOOK.read_text()
    assert re.search(
        r"python3 -B \S*scripts/ap_confirmatory_slot_v4\.py .*--work-dir", body
    )
    assert "--slot-index" in body


# --------------------------------------------------------------------------
# The rendered invocation must keep matching the driver it invokes
# --------------------------------------------------------------------------

DRIVER = REPO_ROOT / "scripts" / "ap_confirmatory_slot_v4.py"


def driver_arguments() -> dict[str, bool]:
    """The driver's argparse surface, read statically from its source.

    Reading the source rather than importing keeps this test away from the
    accrual chain, which mints capability keys at import time.
    """

    import ast

    surface: dict[str, bool] = {}
    for node in ast.walk(ast.parse(DRIVER.read_text())):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute) or function.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        flag = node.args[0].value
        if not isinstance(flag, str) or not flag.startswith("--"):
            continue
        surface[flag] = any(
            keyword.arg == "required"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
    return surface


def test_the_rendered_invocation_supplies_every_flag_the_driver_requires(
    rendered: Path,
) -> None:
    """A driver flag added without a re-render would fail at the first firing.

    argparse exits 2 on a missing required argument, so the slot would be missed
    before anything ran -- and a missed slot is terminal.
    """

    surface = driver_arguments()
    required = {flag for flag, is_required in surface.items() if is_required}
    assert required, "could not read the driver's argparse surface"

    exec_start = re.search(
        r"^ExecStart=(.+)$", service_path(rendered).read_text(), re.MULTILINE
    ).group(1)
    supplied = set(re.findall(r"(--[a-z-]+)", exec_start))

    assert required <= supplied, f"the unit omits required driver flags: {required - supplied}"
    assert supplied <= set(surface), f"the unit passes unknown flags: {supplied - set(surface)}"

    for match in cron_entries(rendered):
        cron_supplied = set(re.findall(r"(--[a-z-]+)", match.group(0)))
        assert required <= cron_supplied
        assert cron_supplied <= set(surface)


def test_the_driver_still_exposes_no_clock_override() -> None:
    """The cadence is the only authority on time; the driver may not be told."""

    surface = driver_arguments()
    for flag in ("--now", "--clock", "--date", "--launched-at", "--slot-at", "--force"):
        assert flag not in surface


def test_the_runbook_documents_every_driver_exit_status() -> None:
    """An operator reading a status at 02:00 needs the table to be complete."""

    import ast

    statuses = {
        node.value.value
        for node in ast.walk(ast.parse(DRIVER.read_text()))
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, int)
        and any(
            isinstance(target, ast.Name) and target.id.startswith("EXIT_")
            for target in node.targets
        )
    }
    assert statuses, "could not read the driver's exit statuses"

    body = RUNBOOK.read_text()
    documented = {int(value) for value in re.findall(r"^\| (\d) \| ", body, re.MULTILINE)}
    assert statuses <= documented, f"the runbook omits exit statuses {statuses - documented}"
