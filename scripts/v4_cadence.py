#!/usr/bin/env python3
"""Render and verify the one-shot-per-slot cadence unit for ``confirmatory-holdout-v4``.

``POLICY.md`` forbids the obvious shape.  "Every slot is a distinct invocation.
No process may sleep, poll, or remain alive waiting across slots, and an
orchestration retry is not a new cadence event."  A supervisor that loops and
sleeps until the next slot is therefore not merely inelegant here, it is
non-compliant: it would keep one process alive across slot boundaries and turn
its own restarts into cadence.  The compliant shape is an *external* timer that
starts a fresh, short-lived interpreter once per slot and lets it exit.

This module renders that timer and then checks it.  It renders two equivalent
forms -- a systemd user timer/service pair and a crontab fallback -- and neither
is ever installed, enabled, started, or otherwise armed from here.  Arming
consumes one-shot authorities and starts an operation whose first missed slot
is terminal for the whole namespace; that decision belongs to the operator, and
``docs/v4-campaign-runbook.md`` is where it is written down.

*Every instant is derived, never tabled.*  The anchor, the cadence, the grace
window, and the slot bounds are read from :mod:`ap_confirmatory_runtime_v4` --
the same frozen constants the driver validates against -- and each firing is
computed through :func:`ap_confirmatory_runtime_v4.slot_times`.  No date literal
appears anywhere in this file, so a drifted constant cannot be masked by a
hardcoded schedule: it changes the render, and the tests notice.

*The two forms are not interchangeable in one respect worth stating loudly.*
systemd calendar specifications carry their own timezone, so a slot instant is
expressed directly as UTC.  Debian ``cron`` does not: it has no per-crontab
timezone (``CRON_TZ`` is ignored, see ``crontab(5)`` LIMITATIONS) and no year
field.  Rendering ``0 0 <day> <month> *`` for a midnight-UTC slot on a host at,
say, ``+07`` would fire seven hours *early*, the driver would correctly refuse a
pre-slot launch, and the campaign would die at slot 0 having never opened a
source.  So the crontab form converts each instant into the cron daemon's own
timezone, pins the year and hour with an explicit UTC guard, and records the
timezone it assumed in a header the verifier and the preflight check read back.

The verifier does not trust the renderer.  ``verify`` parses whatever is on
disk, re-derives the schedule from the frozen constants, and proves the units
match it -- including that each firing maps to its own slot index, that the
worst-case deferral still lands inside the grace window, and that nothing in the
unit sleeps, loops, restarts, or survives its slot.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Rendering must not leave a byte outside ``--out-dir``.
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
DRIVER_RELATIVE_PATH = Path("scripts") / "ap_confirmatory_slot_v4.py"
RUNBOOK_RELATIVE_PATH = Path("docs") / "v4-campaign-runbook.md"


def _load_runtime() -> Any:
    """Import the frozen runtime constants without importing the accrual chain.

    ``ap_confirmatory_accrual_v4`` mints process-local capability keys with
    ``os.urandom`` at import time.  A renderer has no business holding those, so
    this reaches the runtime module directly rather than through the driver.
    """

    path = HERE / "ap_confirmatory_runtime_v4.py"
    name = "ap_confirmatory_runtime_v4"

    # The driver's chain asserts that every v4 module shares one runtime object.
    # If that chain is already loaded in this interpreter, reuse its module
    # rather than executing -- and registering -- a second copy.
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None) == os.fspath(path):
        return existing

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise SystemExit(f"cannot load the frozen runtime constants from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


runtime = _load_runtime()


# --------------------------------------------------------------------------
# Names, defaults, and the one tolerance this module introduces
# --------------------------------------------------------------------------

UNIT_PREFIX = "ap-confirmatory-v4-slot"
SYSTEMD_SUBDIR = "systemd"
CRON_SUBDIR = "cron"
CRON_FILE_NAME = f"{UNIT_PREFIX.rsplit('-', 1)[0]}.crontab"

DEFAULT_PYTHON = "/usr/bin/python3"

# systemd may defer a timer by up to ``AccuracySec`` plus ``RandomizedDelaySec``.
# Both are pinned to their tightest useful values so a firing is effectively the
# scheduled instant; the verifier still proves the worst case lands in-window.
ACCURACY_SECONDS = 1
RANDOMIZED_DELAY_SECONDS = 0

# The crontab guard compares whole UTC hours, so a guarded line may execute
# anywhere inside its hour.  Minute-exact guards would turn ordinary scheduling
# skew into a permanently missed slot, which is terminal; an hour is both
# forgiving and far inside the grace window.  The verifier proves that.
CRON_GUARD_TOLERANCE_SECONDS = 3600

# The tracked example render under ``scripts/v4-cadence/``.  The paths are
# deliberately unusable: an operator who arms the example verbatim gets an
# immediate, loud failure from ``preflight`` instead of a silent miss.
EXAMPLE_REPO_ROOT = "/nonexistent/EXAMPLE-REPO-ROOT"
EXAMPLE_WORK_DIR = "/nonexistent/EXAMPLE-WORK-DIR"
EXAMPLE_CAMPAIGN_MODULE = "/nonexistent/EXAMPLE-PRIVATE/campaign.py"
EXAMPLE_CRON_TIMEZONE = "UTC"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


class CadenceError(Exception):
    """A render or verification refusal, reported with its own message."""


# --------------------------------------------------------------------------
# The schedule, derived from the frozen constants on every call
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SlotFiring:
    """One slot: its index, when it opens, and when its window closes."""

    slot_index: int
    scheduled_at: datetime
    grace_deadline_at: datetime

    @property
    def instant(self) -> str:
        return format_instant(self.scheduled_at)

    @property
    def deadline(self) -> str:
        return format_instant(self.grace_deadline_at)


def format_instant(value: datetime) -> str:
    """Second-resolution UTC, the spelling the schedule is stated in."""

    if value.tzinfo is not UTC:  # pragma: no cover - defensive
        raise CadenceError("refusing to format a non-UTC instant")
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _instant_of(slot_index: int) -> datetime:
    times = runtime.slot_times(slot_index)
    return runtime.parse_utc(times.scheduled_at, receipt=True)


def _deadline_of(slot_index: int) -> datetime:
    times = runtime.slot_times(slot_index)
    return runtime.parse_utc(times.grace_deadline_at, receipt=True)


def slot_firings() -> tuple[SlotFiring, ...]:
    """Every firing of the campaign, computed from the frozen calendar.

    This is the only source of instants in this module.  It also re-proves the
    frozen invariants it depends on, so a drifted constant fails here rather
    than silently rendering a wrong schedule.
    """

    indices = range(runtime.FIRST_SLOT_INDEX, runtime.LAST_SLOT_INDEX + 1)
    firings = tuple(
        SlotFiring(index, _instant_of(index), _deadline_of(index)) for index in indices
    )

    if len(firings) != runtime.SLOT_COUNT:
        raise CadenceError("slot count does not match the frozen constant")
    if firings[0].instant != runtime.ANCHOR_AT:
        raise CadenceError("first firing does not equal the frozen anchor")
    for previous, current in zip(firings, firings[1:]):
        spacing = current.scheduled_at - previous.scheduled_at
        if spacing != timedelta(seconds=runtime.CADENCE_SECONDS):
            raise CadenceError("firing spacing does not equal the frozen cadence")
    for firing in firings:
        window = firing.grace_deadline_at - firing.scheduled_at
        if window != timedelta(seconds=runtime.GRACE_SECONDS):
            raise CadenceError("grace window does not equal the frozen constant")
    horizon = format_instant(runtime.parse_utc(runtime.ABSOLUTE_HORIZON_AT))
    if firings[-1].deadline != horizon:
        raise CadenceError("final window does not close on the frozen horizon")
    # Let the frozen module re-prove the same identity on its own terms.
    runtime.absolute_horizon_at()
    return firings


# --------------------------------------------------------------------------
# Timezone handling for the crontab fallback
# --------------------------------------------------------------------------


def resolve_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise CadenceError(f"unknown timezone {name!r}: {error}") from error


def local_fields(moment: datetime, zone: ZoneInfo) -> tuple[int, int, int, int]:
    """Convert a UTC instant into ``(minute, hour, day, month)`` in ``zone``.

    A cron line names a wall-clock time, so a zone whose offset makes that time
    ambiguous (a repeated hour) or nonexistent (a skipped hour) cannot express
    the instant at all.  Both are refused rather than rendered into a firing
    that lands an hour away from its slot.
    """

    local = moment.astimezone(zone)
    if local.utcoffset() is None:  # pragma: no cover - defensive
        raise CadenceError("timezone produced no UTC offset")
    for fold in (0, 1):
        candidate = local.replace(fold=fold)
        if candidate.utcoffset() != local.utcoffset():
            raise CadenceError(
                f"slot at {format_instant(moment)} falls in an ambiguous local "
                f"hour in {zone.key}; render with a timezone without that "
                "transition, or use the systemd form"
            )
    round_trip = local.replace(tzinfo=zone).astimezone(UTC)
    if round_trip != moment:
        raise CadenceError(
            f"slot at {format_instant(moment)} has no unambiguous local time in "
            f"{zone.key}"
        )
    return local.minute, local.hour, local.day, local.month


def host_timezone_name() -> str | None:
    """The zone the local cron daemon would use, as best the host reports it."""

    etc = Path("/etc/timezone")
    if etc.is_file():
        name = etc.read_text(encoding="utf-8").strip()
        if name:
            return name
    localtime = Path("/etc/localtime")
    if localtime.is_symlink():
        target = os.fspath(localtime.readlink())
        marker = "/zoneinfo/"
        if marker in target:
            return target.split(marker, 1)[1]
    return None


# --------------------------------------------------------------------------
# The rendered text
# --------------------------------------------------------------------------

HEADER = f"""\
# confirmatory-holdout-v4 accrual cadence -- RENDERED, NOT ARMED.
#
# Generated by {DRIVER_RELATIVE_PATH.parent}/{Path(__file__).name}.
# Edit that renderer and re-render; never edit this file.  Check a set with:
#     python3 -B {DRIVER_RELATIVE_PATH.parent}/{Path(__file__).name} verify --unit-dir <dir>
#
# Arming is an operator decision, and the first missed slot ends the campaign
# terminally.  Read {RUNBOOK_RELATIVE_PATH} before installing anything.
"""

SERVICE_TEMPLATE = """\
{header}
[Unit]
Description=confirmatory-holdout-v4 accrual slot %i (one shot, then exit)
Documentation=file://{runbook}
# A manual start is an orchestration retry, and POLICY.md states that an
# orchestration retry is not a cadence event.  Only the timer may activate this.
RefuseManualStart=yes
RefuseManualStop=yes

[Service]
# One slot per process: oneshot forks a fresh interpreter for each firing, and
# it is not held afterwards, so nothing survives into the next slot.
Type=oneshot
RemainAfterExit=no
Restart=no
# The whole slot must complete inside its grace window.
TimeoutStartSec={grace_seconds}
WorkingDirectory={repo_root}
ExecStart={python} -B {driver} --work-dir {work_dir} --slot-index %i --campaign-module {campaign_module}
"""

TIMER_TEMPLATE = """\
{header}
[Unit]
Description=confirmatory-holdout-v4 accrual slot {slot_index} at {instant}
Documentation=file://{runbook}

[Timer]
# Slot {slot_index} of {slot_count}: the window opens at {instant}
# and closes at {deadline}.  Both are fixed by the frozen
# calendar and may not be moved, extended, shortened, or re-anchored.
OnCalendar={oncalendar}
AccuracySec={accuracy}s
RandomizedDelaySec={randomized_delay}s
FixedRandomDelay=false
# A missed firing is never replayed: catch-up would be backfill, which the
# protocol forbids outright.
Persistent=false
RemainAfterElapse=no
Unit={unit_prefix}@{slot_index}.service

[Install]
WantedBy=timers.target
"""

CRON_HEADER_TEMPLATE = """\
{header}
# Fallback for hosts without a systemd user manager.  Prefer the systemd form:
# cron discards this command's output unless an MTA is configured, whereas the
# service form records every firing in the journal.
#
# Two cron limitations shape every line below.  Debian cron has no per-crontab
# timezone -- CRON_TZ is ignored, see crontab(5) LIMITATIONS -- so each firing
# is written in the cron daemon's own local time, recorded here as
# cadence-timezone.  And cron has no year field, so each line carries an
# explicit UTC guard that lets it run in exactly one hour of exactly one day.
# Re-render if the host timezone ever changes; a stale conversion fires at the
# wrong instant and the driver will refuse it.
#
# cadence-timezone: {timezone}
SHELL=/bin/sh
PATH=/usr/bin:/bin
MAILTO=""
"""

CRON_ENTRY_TEMPLATE = """\

# slot {slot_index} of {slot_count}: {instant} .. {deadline} (local {local_time} {timezone})
{minute} {hour} {day} {month} * [ "$(date -u +\\%Y-\\%m-\\%dT\\%H)" = "{guard}" ] \
&& exec {python} -B {driver} --work-dir {work_dir} --slot-index {slot_index} \
--campaign-module {campaign_module}
"""


@dataclass(frozen=True, slots=True)
class RenderRequest:
    """Everything the operator supplies; the schedule supplies the rest."""

    repo_root: Path
    work_dir: Path
    campaign_module: Path
    python: str
    cron_timezone: str

    @property
    def driver(self) -> Path:
        return self.repo_root / DRIVER_RELATIVE_PATH

    @property
    def runbook(self) -> Path:
        return self.repo_root / RUNBOOK_RELATIVE_PATH


def _require_usable_path(value: Path | str, label: str) -> Path:
    text = os.fspath(value)
    if not text or any(character.isspace() for character in text):
        raise CadenceError(f"{label} must not be empty or contain whitespace")
    path = Path(text)
    if not path.is_absolute():
        raise CadenceError(f"{label} must be an absolute path, got {text!r}")
    return Path(os.path.normpath(text))


def build_request(
    *,
    repo_root: Path | str,
    work_dir: Path | str,
    campaign_module: Path | str,
    python: str = DEFAULT_PYTHON,
    cron_timezone: str,
) -> RenderRequest:
    """Validate operator input before a single byte is rendered.

    The work root is refused inside the frozen namespace here for the same
    reason ``ap_confirmatory_slot_v4.require_work_dir_outside_namespace``
    refuses it at run time: the namespace is byte-locked, and a unit that would
    write into it is not worth rendering.
    """

    root = _require_usable_path(repo_root, "--repo-root")
    work = _require_usable_path(work_dir, "--work-dir")
    module = _require_usable_path(campaign_module, "--campaign-module")
    interpreter = _require_usable_path(python, "--python")
    require_work_dir_outside_namespace(work)
    resolve_timezone(cron_timezone)
    return RenderRequest(
        repo_root=root,
        work_dir=work,
        campaign_module=module,
        python=os.fspath(interpreter),
        cron_timezone=cron_timezone,
    )


def require_work_dir_outside_namespace(work_dir: Path) -> Path:
    """Refuse a work root that resolves inside the frozen v4 namespace."""

    candidate = Path(os.path.normpath(os.fspath(work_dir)))
    namespace = Path(os.path.normpath(os.fspath(NAMESPACE_ROOT)))
    if candidate == namespace or namespace in candidate.parents:
        raise CadenceError(
            f"--work-dir {candidate} resolves inside the frozen namespace "
            f"{namespace}; the campaign must write outside it"
        )
    return candidate


def render_service(request: RenderRequest) -> str:
    return SERVICE_TEMPLATE.format(
        header=HEADER.rstrip("\n"),
        runbook=request.runbook,
        grace_seconds=runtime.GRACE_SECONDS,
        repo_root=request.repo_root,
        python=request.python,
        driver=request.driver,
        work_dir=request.work_dir,
        campaign_module=request.campaign_module,
    )


def render_timer(request: RenderRequest, firing: SlotFiring) -> str:
    return TIMER_TEMPLATE.format(
        header=HEADER.rstrip("\n"),
        runbook=request.runbook,
        slot_index=firing.slot_index,
        slot_count=runtime.SLOT_COUNT,
        instant=firing.instant,
        deadline=firing.deadline,
        oncalendar=firing.scheduled_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        accuracy=ACCURACY_SECONDS,
        randomized_delay=RANDOMIZED_DELAY_SECONDS,
        unit_prefix=UNIT_PREFIX,
    )


def render_crontab(request: RenderRequest, firings: Sequence[SlotFiring]) -> str:
    zone = resolve_timezone(request.cron_timezone)
    text = CRON_HEADER_TEMPLATE.format(
        header=HEADER.rstrip("\n"), timezone=request.cron_timezone
    )
    for firing in firings:
        minute, hour, day, month = local_fields(firing.scheduled_at, zone)
        text += CRON_ENTRY_TEMPLATE.format(
            slot_index=firing.slot_index,
            slot_count=runtime.SLOT_COUNT,
            instant=firing.instant,
            deadline=firing.deadline,
            local_time=f"{hour:02d}:{minute:02d}",
            timezone=request.cron_timezone,
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            guard=firing.scheduled_at.strftime("%Y-%m-%dT%H"),
            python=request.python,
            driver=request.driver,
            work_dir=request.work_dir,
            campaign_module=request.campaign_module,
        )
    return text


def render(request: RenderRequest) -> dict[str, str]:
    """The whole rendered set, as a mapping of relative path to file text."""

    firings = slot_firings()
    files = {
        f"{SYSTEMD_SUBDIR}/{UNIT_PREFIX}@.service": render_service(request),
        f"{CRON_SUBDIR}/{CRON_FILE_NAME}": render_crontab(request, firings),
    }
    for firing in firings:
        name = f"{SYSTEMD_SUBDIR}/{UNIT_PREFIX}@{firing.slot_index}.timer"
        files[name] = render_timer(request, firing)
    return files


def write_render(out_dir: Path, files: dict[str, str], *, clean: bool = True) -> None:
    """Materialise a rendered set, replacing any previous render in place."""

    for subdir in (SYSTEMD_SUBDIR, CRON_SUBDIR):
        target = out_dir / subdir
        if clean and target.is_dir():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
    for relative, text in files.items():
        (out_dir / relative).write_text(text, encoding="utf-8")


def render_digest(files: dict[str, str]) -> str:
    """A stable digest over the whole rendered set."""

    digest = hashlib.sha256()
    for relative in sorted(files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(files[relative].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


# --------------------------------------------------------------------------
# The verifier: it re-derives the schedule and distrusts the files
# --------------------------------------------------------------------------

# Anything that would keep a process alive across a slot boundary, retry a
# firing, or wrap the driver in a shell that waits.
FORBIDDEN_COMMAND_TOKENS = (
    "sleep",
    "usleep",
    "while",
    "until",
    "watch",
    "loop",
    "nohup",
    "setsid",
    "screen",
    "tmux",
    "systemd-run",
)

# Relative timers drift and repeat; only an absolute calendar may name a slot.
FORBIDDEN_TIMER_DIRECTIVES = (
    "OnBootSec",
    "OnStartupSec",
    "OnActiveSec",
    "OnUnitActiveSec",
    "OnUnitInactiveSec",
)

CRON_LINE_PATTERN = re.compile(
    r"^(?P<minute>\d{1,2}) (?P<hour>\d{1,2}) (?P<day>\d{1,2}) (?P<month>\d{1,2}) \* "
    r"\[ \"\$\(date -u \+\\%Y-\\%m-\\%dT\\%H\)\" = \"(?P<guard>\d{4}-\d{2}-\d{2}T\d{2})\" \] "
    r"&& exec (?P<python>\S+) -B (?P<driver>\S+) --work-dir (?P<work_dir>\S+) "
    r"--slot-index (?P<slot_index>\d+) --campaign-module (?P<campaign_module>\S+)$"
)

EXEC_START_PATTERN = re.compile(
    r"^(?P<python>\S+) -B (?P<driver>\S+) --work-dir (?P<work_dir>\S+) "
    r"--slot-index %i --campaign-module (?P<campaign_module>\S+)$"
)


def parse_unit(text: str) -> dict[str, list[tuple[str, str]]]:
    """A minimal systemd unit reader: sections to ordered key/value pairs."""

    sections: dict[str, list[tuple[str, str]]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, [])
            continue
        if current is None or "=" not in line:
            continue
        key, _, value = line.partition("=")
        sections[current].append((key.strip(), value.strip()))
    return sections


def unit_value(
    sections: dict[str, list[tuple[str, str]]], section: str, key: str
) -> str | None:
    values = [value for name, value in sections.get(section, []) if name == key]
    return values[-1] if values else None


def _check_command(command: str, findings: list[str], where: str) -> None:
    lowered = command.lower()
    for token in FORBIDDEN_COMMAND_TOKENS:
        if re.search(rf"(?<![\w-]){re.escape(token)}(?![\w-])", lowered):
            findings.append(f"{where}: command contains {token!r}, which may not wait")


def verify_service(
    unit_dir: Path, findings: list[str]
) -> tuple[str, str, str, str] | None:
    """Check the template service and return its invocation parts."""

    path = unit_dir / SYSTEMD_SUBDIR / f"{UNIT_PREFIX}@.service"
    if not path.is_file():
        findings.append(f"missing template service {path}")
        return None
    sections = parse_unit(path.read_text(encoding="utf-8"))
    where = path.name

    if unit_value(sections, "Service", "Type") != "oneshot":
        findings.append(f"{where}: Type must be oneshot so each slot is one process")
    remain = unit_value(sections, "Service", "RemainAfterExit")
    if remain is not None and remain.lower() not in ("no", "false", "0", "off"):
        findings.append(f"{where}: RemainAfterExit={remain} would outlive the slot")
    restart = unit_value(sections, "Service", "Restart")
    if restart is not None and restart.lower() != "no":
        findings.append(f"{where}: Restart={restart} would retry a cadence event")
    for key in ("ExecStartPre", "ExecStartPost", "ExecStop", "ExecReload"):
        if unit_value(sections, "Service", key) is not None:
            findings.append(f"{where}: {key} adds a step outside the driver")

    starts = [value for name, value in sections.get("Service", []) if name == "ExecStart"]
    if len(starts) != 1:
        findings.append(f"{where}: expected exactly one ExecStart, found {len(starts)}")
        return None
    match = EXEC_START_PATTERN.match(starts[0])
    if match is None:
        findings.append(f"{where}: ExecStart is not a plain one-slot driver invocation")
        return None
    _check_command(starts[0], findings, where)
    if not starts[0].startswith("/"):
        findings.append(f"{where}: ExecStart must name an absolute interpreter")
    if Path(match.group("driver")).name != DRIVER_RELATIVE_PATH.name:
        findings.append(f"{where}: ExecStart does not invoke {DRIVER_RELATIVE_PATH.name}")
    try:
        require_work_dir_outside_namespace(Path(match.group("work_dir")))
    except CadenceError as error:
        findings.append(f"{where}: {error}")

    timeout = unit_value(sections, "Service", "TimeoutStartSec")
    if timeout != str(runtime.GRACE_SECONDS):
        findings.append(
            f"{where}: TimeoutStartSec={timeout} does not bound the run to the "
            f"{runtime.GRACE_SECONDS}s grace window"
        )
    return (
        match.group("python"),
        match.group("driver"),
        match.group("work_dir"),
        match.group("campaign_module"),
    )


def verify_timers(unit_dir: Path, findings: list[str]) -> None:
    """Check that the timers reproduce the derived schedule exactly."""

    firings = {firing.slot_index: firing for firing in slot_firings()}
    directory = unit_dir / SYSTEMD_SUBDIR
    found = sorted(directory.glob(f"{UNIT_PREFIX}@*.timer"))
    indices: dict[int, Path] = {}
    for path in found:
        raw = path.name[len(UNIT_PREFIX) + 1 : -len(".timer")]
        if not raw.isdigit():
            findings.append(f"{path.name}: instance name is not a slot index")
            continue
        indices[int(raw)] = path

    if set(indices) != set(firings):
        missing = sorted(set(firings) - set(indices))
        extra = sorted(set(indices) - set(firings))
        if missing:
            findings.append(f"no timer for slot indices {missing}")
        if extra:
            findings.append(f"timers for unknown slot indices {extra}")

    for index in sorted(set(indices) & set(firings)):
        path = indices[index]
        firing = firings[index]
        where = path.name
        sections = parse_unit(path.read_text(encoding="utf-8"))

        calendars = [
            value for name, value in sections.get("Timer", []) if name == "OnCalendar"
        ]
        if len(calendars) != 1:
            findings.append(
                f"{where}: expected exactly one OnCalendar, found {len(calendars)}"
            )
            continue
        try:
            fired_at = datetime.strptime(calendars[0], "%Y-%m-%d %H:%M:%S UTC").replace(
                tzinfo=UTC
            )
        except ValueError:
            findings.append(f"{where}: OnCalendar={calendars[0]!r} is not an exact UTC instant")
            continue
        if fired_at != firing.scheduled_at:
            findings.append(
                f"{where}: fires at {format_instant(fired_at)}, but slot {index} "
                f"opens at {firing.instant}"
            )
            continue

        accuracy = _duration_seconds(unit_value(sections, "Timer", "AccuracySec"), 60)
        delay = _duration_seconds(unit_value(sections, "Timer", "RandomizedDelaySec"), 0)
        if accuracy is None or delay is None:
            findings.append(f"{where}: AccuracySec or RandomizedDelaySec is unreadable")
            continue
        latest = fired_at + timedelta(seconds=accuracy + delay)
        if not firing.scheduled_at <= fired_at or latest >= firing.grace_deadline_at:
            findings.append(
                f"{where}: worst-case firing {format_instant(latest)} is outside the "
                f"window [{firing.instant}, {firing.deadline})"
            )

        persistent = unit_value(sections, "Timer", "Persistent")
        if persistent is not None and persistent.lower() not in ("no", "false", "0", "off"):
            findings.append(f"{where}: Persistent={persistent} would backfill a missed slot")
        remain = unit_value(sections, "Timer", "RemainAfterElapse")
        if remain is None or remain.lower() not in ("no", "false", "0", "off"):
            findings.append(f"{where}: RemainAfterElapse must be no so the timer cannot re-fire")
        for directive in FORBIDDEN_TIMER_DIRECTIVES:
            if unit_value(sections, "Timer", directive) is not None:
                findings.append(f"{where}: {directive} is a relative trigger, not a slot")

        target = unit_value(sections, "Timer", "Unit")
        expected = f"{UNIT_PREFIX}@{index}.service"
        if target != expected:
            findings.append(f"{where}: activates {target!r}, expected {expected!r}")


def _duration_seconds(value: str | None, default: int) -> int | None:
    if value is None:
        return default
    text = value.strip().lower()
    if text.isdigit():
        return int(text)
    match = re.fullmatch(r"(\d+)(us|ms|s|sec|seconds|m|min|minutes|h|hours)", text)
    if match is None:
        return None
    scale = {
        "us": 0,
        "ms": 0,
        "s": 1,
        "sec": 1,
        "seconds": 1,
        "m": 60,
        "min": 60,
        "minutes": 60,
        "h": 3600,
        "hours": 3600,
    }[match.group(2)]
    return int(match.group(1)) * scale


def verify_crontab(unit_dir: Path, findings: list[str]) -> None:
    """Check the fallback: exact instants, correct slot, guarded to one hour."""

    path = unit_dir / CRON_SUBDIR / CRON_FILE_NAME
    if not path.is_file():
        findings.append(f"missing crontab fallback {path}")
        return
    text = path.read_text(encoding="utf-8")

    declared = re.search(r"^# cadence-timezone: (?P<zone>\S+)$", text, re.MULTILINE)
    if declared is None:
        findings.append(f"{path.name}: no cadence-timezone header to check firings against")
        return
    try:
        zone = resolve_timezone(declared.group("zone"))
    except CadenceError as error:
        findings.append(f"{path.name}: {error}")
        return

    firings = {firing.slot_index: firing for firing in slot_firings()}
    seen: dict[int, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" in line.split(" ", 5)[0]:
            continue
        match = CRON_LINE_PATTERN.match(line)
        if match is None:
            findings.append(f"{path.name}: unrecognised entry {line[:60]!r}")
            continue
        index = int(match.group("slot_index"))
        if index in seen:
            findings.append(f"{path.name}: slot {index} has more than one entry")
            continue
        seen[index] = line
        firing = firings.get(index)
        if firing is None:
            findings.append(f"{path.name}: entry for unknown slot index {index}")
            continue

        _check_command(match.group(0), findings, path.name)
        local = datetime(
            firing.scheduled_at.year,
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            tzinfo=zone,
        )
        fired_at = local.astimezone(UTC)
        if fired_at != firing.scheduled_at:
            findings.append(
                f"{path.name}: slot {index} fires at {format_instant(fired_at)} in "
                f"{declared.group('zone')}, but the slot opens at {firing.instant}"
            )
            continue
        if match.group("guard") != firing.scheduled_at.strftime("%Y-%m-%dT%H"):
            findings.append(
                f"{path.name}: slot {index} guard {match.group('guard')!r} does not "
                "pin the firing to its own UTC hour"
            )
        latest = fired_at + timedelta(seconds=CRON_GUARD_TOLERANCE_SECONDS)
        if latest >= firing.grace_deadline_at:
            findings.append(
                f"{path.name}: slot {index} guard admits {format_instant(latest)}, "
                f"outside the window [{firing.instant}, {firing.deadline})"
            )
        if Path(match.group("driver")).name != DRIVER_RELATIVE_PATH.name:
            findings.append(f"{path.name}: slot {index} does not invoke the slot driver")
        try:
            require_work_dir_outside_namespace(Path(match.group("work_dir")))
        except CadenceError as error:
            findings.append(f"{path.name}: slot {index}: {error}")

    missing = sorted(set(firings) - set(seen))
    if missing:
        findings.append(f"{path.name}: no entry for slot indices {missing}")


def verify(unit_dir: Path) -> list[str]:
    """Every check, run against whatever is on disk.  Empty means compliant."""

    findings: list[str] = []
    verify_service(unit_dir, findings)
    verify_timers(unit_dir, findings)
    verify_crontab(unit_dir, findings)
    return findings


# --------------------------------------------------------------------------
# Preflight: what the host would actually do, without doing any of it
# --------------------------------------------------------------------------


class PreflightClock:
    """The real clock, with no command-line or environment override.

    The driver refuses to be told what time it is because a fake instant could
    open a source outside its window.  Preflight only reports, but it keeps the
    same discipline so no habit of passing a clock in ever forms.  Tests
    substitute a subclass through the ``clock=`` keyword of :func:`preflight`.
    """

    def now(self) -> datetime:
        return datetime.now(UTC)


def preflight(unit_dir: Path, *, clock: PreflightClock | None = None) -> list[str]:
    """Check a rendered set against this host before the operator arms it.

    This exists because every path in a rendered unit is only a string until
    the first firing, and the first firing is the one that cannot be retried.
    """

    findings = verify(unit_dir)
    now = (clock or PreflightClock()).now()

    parts = verify_service(unit_dir, [])
    if parts is None:
        findings.append("cannot preflight paths: the service unit did not parse")
        return findings
    python, driver, work_dir, campaign_module = parts

    if not Path(python).is_file() or not os.access(python, os.X_OK):
        findings.append(f"interpreter {python} is missing or not executable")
    if not Path(driver).is_file():
        findings.append(f"slot driver {driver} does not exist on this host")
    if not Path(campaign_module).is_file():
        findings.append(f"campaign module {campaign_module} does not exist on this host")
    work = Path(work_dir)
    if not work.is_dir():
        findings.append(f"work directory {work} does not exist; create it before arming")
    elif not os.access(work, os.W_OK):
        findings.append(f"work directory {work} is not writable")

    crontab = unit_dir / CRON_SUBDIR / CRON_FILE_NAME
    if crontab.is_file():
        declared = re.search(
            r"^# cadence-timezone: (?P<zone>\S+)$",
            crontab.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        host_zone = host_timezone_name()
        if declared is not None and host_zone is not None:
            rendered_zone = declared.group("zone")
            if rendered_zone != host_zone:
                first = slot_firings()[0].scheduled_at
                try:
                    same = first.astimezone(resolve_timezone(rendered_zone)).replace(
                        tzinfo=None
                    ) == first.astimezone(resolve_timezone(host_zone)).replace(tzinfo=None)
                except CadenceError:
                    same = False
                if not same:
                    findings.append(
                        f"the crontab was rendered for {rendered_zone} but this host "
                        f"runs cron in {host_zone}; re-render with "
                        f"--cron-timezone {host_zone} or use the systemd form"
                    )

    firings = slot_firings()
    if now >= firings[0].scheduled_at:
        findings.append(
            f"slot 0 opened at {firings[0].instant} and it is already "
            f"{format_instant(now.replace(microsecond=0))}: the campaign can no "
            "longer be armed, because a missed slot is terminal and there is no "
            "backfill"
        )
    return findings


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The whole surface.  Note that nothing here arms anything.

    There is no ``install``, ``enable``, ``arm``, or ``activate`` subcommand,
    and no clock flag.  Rendering writes files; verifying reads them; preflight
    reports.  The operator issues the arming commands themselves, from the
    runbook, having read what a missed slot costs.
    """

    parser = argparse.ArgumentParser(
        description=(
            "render and verify the confirmatory-holdout-v4 one-shot-per-slot "
            "cadence unit; this tool never arms the campaign"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    renderer = subparsers.add_parser("render", help="write a cadence unit set")
    renderer.add_argument("--out-dir", type=Path, required=True)
    renderer.add_argument("--repo-root", type=Path, required=True)
    renderer.add_argument("--work-dir", type=Path, required=True)
    renderer.add_argument("--campaign-module", type=Path, required=True)
    renderer.add_argument("--python", default=DEFAULT_PYTHON)
    renderer.add_argument(
        "--cron-timezone",
        required=True,
        help="the timezone the cron daemon runs in; Debian cron ignores CRON_TZ",
    )

    checker = subparsers.add_parser("verify", help="check a rendered set")
    checker.add_argument("--unit-dir", type=Path, required=True)

    flight = subparsers.add_parser(
        "preflight", help="check a rendered set against this host before arming"
    )
    flight.add_argument("--unit-dir", type=Path, required=True)

    subparsers.add_parser("schedule", help="print the derived slot schedule")
    return parser


def _schedule_report() -> dict[str, Any]:
    firings = slot_firings()
    return {
        "anchor_at": runtime.ANCHOR_AT,
        "cadence_seconds": runtime.CADENCE_SECONDS,
        "grace_seconds": runtime.GRACE_SECONDS,
        "slot_count": runtime.SLOT_COUNT,
        "final_selection_slot_at": firings[-1].instant,
        "absolute_horizon_expires_at": firings[-1].deadline,
        "slots": [
            {
                "slot_index": firing.slot_index,
                "opens_at": firing.instant,
                "closes_at": firing.deadline,
            }
            for firing in firings
        ],
    }


def _report(findings: Iterable[str], payload: dict[str, Any]) -> int:
    problems = list(findings)
    print(json.dumps({**payload, "ok": not problems, "findings": problems}, indent=2))
    return EXIT_OK if not problems else EXIT_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as request:
        return EXIT_OK if request.code in (0, None) else EXIT_USAGE

    try:
        if arguments.command == "render":
            request = build_request(
                repo_root=arguments.repo_root,
                work_dir=arguments.work_dir,
                campaign_module=arguments.campaign_module,
                python=arguments.python,
                cron_timezone=arguments.cron_timezone,
            )
            files = render(request)
            write_render(arguments.out_dir, files)
            findings = verify(arguments.out_dir)
            return _report(
                findings,
                {
                    "command": "render",
                    "out_dir": os.fspath(arguments.out_dir),
                    "files": len(files),
                    "digest": render_digest(files),
                    "armed": False,
                },
            )

        if arguments.command == "verify":
            return _report(
                verify(arguments.unit_dir),
                {"command": "verify", "unit_dir": os.fspath(arguments.unit_dir)},
            )

        if arguments.command == "preflight":
            return _report(
                preflight(arguments.unit_dir),
                {"command": "preflight", "unit_dir": os.fspath(arguments.unit_dir)},
            )

        print(json.dumps(_schedule_report(), indent=2))
        return EXIT_OK
    except CadenceError as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2))
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
