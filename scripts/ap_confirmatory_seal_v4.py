#!/usr/bin/env python3
"""Confirmatory-holdout-v4 one-shot, no-overwrite seal launcher.

This module owns exactly what ``scripts/ap_confirmatory_accrual_v4.py``
deliberately excludes.  That module sets ``ENTRY_KINDS = FROZEN_ENTRY_KINDS -
{"seal-terminal"}`` because "the packet sealer owns ``seal-terminal``
validation and publication", so the ``seal-terminal`` receipt schema, its
validator and its ledger entry live here and nowhere else.

Two commands:

* ``self-check --work-dir DIR`` drives the complete seven-step atomic ready
  handoff and the no-overwrite publication against a synthetic fixture built
  entirely inside ``DIR``.  It seals no real packet, runs no real accrual and
  keeps every read aggregate-only.
* ``__child`` is the internal sealer entry point.  It is spawned by the
  launcher behind an inherited, still-closed start gate; it is never invoked
  by hand.

``readiness.atomic_ready_handoff.ordering_exactly`` is implemented as its
seven steps in order, and the deadline for every pre-gate stage is
``marker.consumed_at + 30`` seconds.  Authority is consumed at marker
creation, not at manifest write: an existing valid marker is consumed
authority and never permission to spawn a replacement child.

The packet builder is driven only through its fixed CLI.  Its bytes and the
verifier's bytes are hashed at runtime -- neither is embedded as a source
literal -- and they populate ``sealer_sha256_and_bytes`` and
``verifier_sha256_and_bytes_or_null`` respectively.
"""

from __future__ import annotations

import sys

# Importing the shared v4 modules must not write ``scripts/__pycache__``.
# ``self-check`` promises to touch nothing outside its work directory, and a
# stray bytecode file would break that before the first fixture is generated.
sys.dont_write_bytecode = True

import argparse
import contextlib
import ctypes
import hashlib
import hmac
import json
import os
import select
import signal
import stat
import struct
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
NAMESPACE_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "confirmatory-holdout-v4"
)
BUILDER_PATH = NAMESPACE_ROOT / "recipe" / "build.py"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# The accrual module imports the probe module, which imports the runtime
# module, all by canonical name.  Reaching the runtime through
# ``accrual.runtime`` is mandatory: all three use ``type(x) is SomeDataclass``
# identity checks, so a second importlib copy would silently invalidate every
# wrapper handed back to them.
import ap_confirmatory_accrual_v4 as accrual  # noqa: E402

probe = accrual.probe
runtime = accrual.runtime

if runtime is not probe.runtime or probe is not accrual.probe:
    raise SystemExit("v4 runtime module identity is not shared")

IntegrityFailure = runtime.IntegrityFailure

NAMESPACE = runtime.NAMESPACE
SCHEMA_VERSION = runtime.SCHEMA_VERSION
SOURCE_ALIASES = runtime.SOURCE_ALIASES

# --------------------------------------------------------------------------
# authority.seal -- the frozen one-shot grant
# --------------------------------------------------------------------------

SEAL_AUTHORITY_ID = "confirmatory-holdout-v4-seal"
MAX_PROCESS_LAUNCHES = 1
MAX_PUBLICATION_ATTEMPTS = 1
WATCHDOG_SECONDS = 3600
FORCED_TERMINATION_GRACE_SECONDS = 30
HANDOFF_DEADLINE_SECONDS = 30

if (
    SEAL_AUTHORITY_ID != accrual.SEAL_AUTHORITY_ID
    or HANDOFF_DEADLINE_SECONDS != accrual.HANDOFF_SECONDS
):  # pragma: no cover - both are frozen constants
    raise SystemExit("v4 seal authority constants disagree with accrual")

# ``readiness.atomic_ready_handoff.ordering_exactly`` in its exact order.  The
# implementation walks this list; a step is never skipped, reordered or fused.
ORDERING_EXACTLY = (
    "compute provisional ready core privately after final stable runtime "
    "check while ledger head is H",
    "O_EXCL-create and fsync seal-consumption marker binding H, provisional "
    "core digest/bytes, segment, slot, and exact snapshots",
    "append seal-consumption ledger entry whose predecessor is H; its entry "
    "hash becomes M",
    "spawn the sealer with the consumed marker and exact retained snapshots "
    "behind an inherited closed start gate; it may not read snapshots, "
    "build, or publish",
    "construct and fsync in a private same-filesystem staging directory the "
    "final ready resolution binding marker/spawn with predecessor M plus its "
    "deterministic ledger entry R, then deliver both exact bytes and R over "
    "inherited IPC while they remain invisible",
    "after child validation, expose the ready resolution and R together with "
    "one atomic no-replace directory rename; no state exposes either artifact "
    "alone",
    "open the child start gate only after atomic visibility; the child then "
    "reopens and validates the public bundle before any snapshot read, build, "
    "or publication",
)

# --------------------------------------------------------------------------
# operational_receipt_schemas.seal_terminal_receipt
# --------------------------------------------------------------------------

SEAL_TERMINAL_RECEIPT_KIND = "seal-terminal"
SEAL_TERMINAL_ENTRY_KIND = "seal-terminal"
SEAL_RECEIPT_FILENAME = "seal-receipt.json"

SEAL_TERMINAL_FIELDS = (
    "schema_version",
    "namespace",
    "receipt_kind",
    "status",
    "failure_stage_or_null",
    "failure_reason_or_null",
    "provisional_ready_core_sha256_and_bytes",
    "seal_consumption_marker_sha256_and_bytes_or_null",
    "slot_resolution_sha256_and_bytes_or_null",
    "segment_attestation_sha256_and_bytes",
    "snapshot_set_sha256_and_bytes",
    "post_build_runtime_observation_sha256_and_bytes_or_null",
    "manifest_sha256_and_bytes_or_null",
    "watchdog_deadline_at_or_null",
    "forced_termination_deadline_at_or_null",
    "completed_at",
    "previous_ledger_entry_sha256",
    "sealer_sha256_and_bytes_or_null",
    "verifier_sha256_and_bytes_or_null",
    "analysis_plan_sha256_and_bytes",
)

STATUS_SEALED = "sealed"
STATUS_TERMINAL = "terminal-seal-failure"
SEAL_TERMINAL_STATUS_ENUM = (STATUS_SEALED, STATUS_TERMINAL)

FAILURE_STAGE_ENUM = (
    None,
    "marker-create",
    "marker-ledger",
    "spawn",
    "ready-staging",
    "ready-ipc",
    "ready-visibility",
    "start-gate",
    "blocked-child",
    "build",
    "post-build-observation",
    "publication",
    "validation",
    "watchdog",
)

FAILURE_REASON_ENUM = (
    None,
    "marker-create-failure",
    "marker-collision",
    "marker-validation-failure",
    "marker-ledger-failure",
    "spawn-failure",
    "staging-failure",
    "ready-ledger-failure",
    "ipc-validation-failure",
    "atomic-visibility-failure",
    "start-gate-failure",
    "blocked-child-eof",
    "handoff-timeout",
    "build-failure",
    "runtime-change",
    "observation-unavailable",
    "publication-failure",
    "verifier-failure",
    "timeout",
    "forced-termination",
)

STAGE_REASON_MATRIX: dict[str, tuple[str, ...]] = {
    "marker-create": ("marker-create-failure", "marker-collision"),
    "marker-ledger": ("marker-validation-failure", "marker-ledger-failure"),
    "spawn": ("spawn-failure", "handoff-timeout"),
    "ready-staging": ("staging-failure", "ready-ledger-failure", "handoff-timeout"),
    "ready-ipc": ("ipc-validation-failure", "handoff-timeout"),
    "ready-visibility": ("atomic-visibility-failure", "handoff-timeout"),
    "start-gate": ("start-gate-failure", "handoff-timeout"),
    "blocked-child": ("blocked-child-eof", "handoff-timeout"),
    "build": ("build-failure",),
    "post-build-observation": ("runtime-change", "observation-unavailable"),
    "publication": ("publication-failure",),
    "validation": ("verifier-failure",),
    "watchdog": ("timeout", "forced-termination"),
}

# ``ledger_predecessor_rule``: which head a terminal receipt appends from.
# ``H`` before the marker is durable, ``M`` after it, and the "atomically
# visible ready head R" from the instant the bundle becomes visible.  The
# ranges below are the rule's own bounds; where they could overlap, the
# absolute ``branch_or_circular_reference_allowed: false`` invariant decides,
# because a chain may only ever be extended from its own durable head.
PREDECESSOR_H_STAGES = frozenset({"marker-create", "marker-ledger"})
POST_GATE_STAGES = frozenset(
    {"build", "post-build-observation", "publication", "validation", "watchdog"}
)

# Every stage that completes before the child's start gate opens.  These share
# one deadline and may carry no observation, manifest or verifier identity.
PRE_GATE_STAGES = frozenset(
    {
        "marker-create",
        "marker-ledger",
        "spawn",
        "ready-staging",
        "ready-ipc",
        "ready-visibility",
        "start-gate",
        "blocked-child",
    }
)

# ``operational_receipt_schemas.post_build_runtime_observation``
POST_BUILD_OBSERVATION_KIND = "post-build-runtime-observation"
POST_BUILD_OBSERVATION_FIELDS = (
    "schema_version",
    "namespace",
    "receipt_kind",
    "status",
    "segment_id",
    "observed_active_services_state_sha256_and_bytes",
    "observed_source_binding_core_sha256_and_bytes",
    "segment_attestation_sha256_and_bytes",
    "snapshot_set_sha256_and_bytes",
    "observed_at",
    "runtime_observer_sha256_and_bytes",
    "analysis_plan_sha256_and_bytes",
)

# ``common_validation.content_addressed_artifact_resolution`` -- the two
# observation members whose bytes must resolve in the object store.
RETAINED_OBJECT_FIELDS = (
    "observed_active_services_state_sha256_and_bytes",
    "observed_source_binding_core_sha256_and_bytes",
)

# --------------------------------------------------------------------------
# Exit statuses and refusal codes
# --------------------------------------------------------------------------

EXIT_SEALED = 0
EXIT_TERMINAL_SEAL_FAILURE = 1
EXIT_USAGE = 2
EXIT_NAMESPACE_REFUSED = 3

NAMESPACE_REFUSAL_CODE = "namespace_output_root_refused"

MAX_RECEIPT_BYTES = 1_000_000
MAX_FRAME_BYTES = 4_000_000
PRECOMMITTED_READY_OFFSET_SECONDS = 10
CHILD_BUILD_TIMEOUT_SECONDS = 900
CHILD_TERMINATION_GRACE_SECONDS = 5


class SealTerminalFailure(Exception):
    """One terminal seal failure carrying its exact stage and reason."""

    def __init__(self, stage: str, reason: str) -> None:
        super().__init__()
        if stage not in STAGE_REASON_MATRIX or reason not in STAGE_REASON_MATRIX[stage]:
            # A stage/reason pair outside the frozen matrix is itself an
            # integrity failure; it must never reach a receipt.
            raise IntegrityFailure from None
        self.stage = stage
        self.reason = reason


class NamespaceRefusal(Exception):
    """A work or publish root resolving inside the frozen namespace."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path


class ChildProtocolFailure(Exception):
    """The internal sealer child could not complete its own contract."""

    def __init__(self, code: str) -> None:
        super().__init__()
        self.code = code


def _fail() -> None:
    raise IntegrityFailure from None


# --------------------------------------------------------------------------
# Canonical JSON, identity and exact key sets
# --------------------------------------------------------------------------


def canonical_bytes(value: Any) -> bytes:
    return runtime.canonical_json_bytes(value)


def identity_of_bytes(raw: bytes) -> dict[str, Any]:
    return runtime.hash_and_bytes(raw).as_dict()


def identity_of_path(path: Path | str, *, maximum: int = 8_000_000) -> dict[str, Any]:
    """Hash a regular non-symlink file at runtime.  Never a source literal."""

    return runtime.hash_and_bytes(
        runtime.read_regular_file(path, maximum_bytes=maximum)
    ).as_dict()


def _is_int(value: Any) -> bool:
    return type(value) is int


def require_exact_keys(
    value: Any, fields: Sequence[str], label: str
) -> dict[str, Any]:
    """``unknown_or_missing_field_action`` is invalid, at every nested depth."""

    if type(value) is not dict or set(value) != set(fields):
        raise IntegrityFailure from None
    del label
    return dict(value)


def require_identity(value: Any) -> dict[str, Any]:
    """``composite_types.sha256_and_bytes``: exactly sha256 and bytes."""

    if type(value) is not dict or set(value) != {"sha256", "bytes"}:
        _fail()
    sha256 = value["sha256"]
    size = value["bytes"]
    if (
        type(sha256) is not str
        or len(sha256) != 64
        or sha256.strip("0123456789abcdef")
        or not _is_int(size)
        or size < 0
    ):
        _fail()
    return {"sha256": sha256, "bytes": size}


def require_identity_or_null(value: Any) -> dict[str, Any] | None:
    return None if value is None else require_identity(value)


def same_identity(left: Any, right: Any) -> bool:
    return require_identity(left) == require_identity(right)


def load_canonical_receipt(raw: bytes, fields: Sequence[str]) -> dict[str, Any]:
    """Parse exactly-canonical bytes with a closed key set and no duplicates."""

    if type(raw) is not bytes or len(raw) > MAX_RECEIPT_BYTES:
        _fail()
    value = runtime.load_canonical_json_bytes(raw, expected=dict)
    return require_exact_keys(value, fields, "receipt")


# --------------------------------------------------------------------------
# Durable primitives
# --------------------------------------------------------------------------

RENAME_NOREPLACE = 1 << 0

_RENAMEAT2_SYSCALL_BY_MACHINE = {
    "x86_64": 316,
    "aarch64": 276,
    "armv7l": 382,
    "armv8l": 382,
    "ppc64le": 357,
    "s390x": 347,
    "riscv64": 276,
    "loongarch64": 276,
}


def _renameat2_number() -> int:
    number = _RENAMEAT2_SYSCALL_BY_MACHINE.get(os.uname().machine)
    if number is None:
        raise SealTerminalFailure("ready-visibility", "atomic-visibility-failure")
    return int(number)


def rename_directory_no_replace(parent_fd: int, source: str, target: str) -> None:
    """Expose a fully populated directory atomically and never over a target.

    ``RENAME_NOREPLACE`` is required rather than preferred: a plain rename
    would silently replace an existing ready bundle, and
    ``publication_and_ordering.no_overwrite`` forbids that.  A kernel or libc
    without it fails closed rather than degrading to a racy existence check.
    """

    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError:
        raise SealTerminalFailure(
            "ready-visibility", "atomic-visibility-failure"
        ) from None
    call: Any
    if hasattr(libc, "renameat2"):
        call = libc.renameat2
        call.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        call.restype = ctypes.c_int
        arguments: tuple[Any, ...] = (
            parent_fd,
            os.fsencode(source),
            parent_fd,
            os.fsencode(target),
            RENAME_NOREPLACE,
        )
    elif hasattr(libc, "syscall"):
        call = libc.syscall
        call.argtypes = (
            ctypes.c_long,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        call.restype = ctypes.c_long
        arguments = (
            _renameat2_number(),
            parent_fd,
            os.fsencode(source),
            parent_fd,
            os.fsencode(target),
            RENAME_NOREPLACE,
        )
    else:
        raise SealTerminalFailure("ready-visibility", "atomic-visibility-failure")

    ctypes.set_errno(0)
    if call(*arguments) != 0:
        # EEXIST, ENOSYS and every other errno fail closed identically: the
        # bundle either became visible by this one rename or it did not.
        raise SealTerminalFailure("ready-visibility", "atomic-visibility-failure")


def fsync_directory(path: Path | str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def open_directory_fd(path: Path | str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def make_private_directory(path: Path) -> None:
    """Create one private directory that no other user can observe."""

    path.mkdir(mode=0o700)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
        _fail()
    fsync_directory(path.parent)


def normalized_path(path: Path | str) -> Path:
    return Path(os.path.realpath(os.fspath(path)))


def require_output_root_outside_namespace(path: Path | str) -> Path:
    """Refuse any root resolving inside the frozen namespace, with a code.

    The launcher seals no real packet.  A namespace-resident work or publish
    root is refused before a single byte is written, with its own distinct
    exit status so a caller cannot mistake it for a seal failure.
    """

    candidate = normalized_path(path)
    namespace = normalized_path(NAMESPACE_ROOT)
    if candidate == namespace or namespace in candidate.parents:
        raise NamespaceRefusal(candidate)
    return candidate


# --------------------------------------------------------------------------
# Clocks
# --------------------------------------------------------------------------


class SealClock:
    """Receipt time.  Deadlines are enforced against this same reading."""

    def now(self) -> datetime:  # pragma: no cover - the real path is never run
        return datetime.now(UTC)

    def canonical_now(self) -> str:
        return runtime.canonical_utc(self.now())


class SyntheticSealClock(SealClock):
    """A deterministic monotone clock anchored inside one fixed slot.

    The frozen slot lattice starts at 2026-08-17, so any exercise of this
    ceremony before that date must supply its own receipt clock.  Real elapsed
    time still bounds every wait through ``time.monotonic``; this clock only
    supplies the protocol timestamps whose arithmetic the ceremony validates.
    """

    def __init__(self, base: datetime, *, step_microseconds: int = 250_000) -> None:
        self._base = base
        self._step = timedelta(microseconds=step_microseconds)
        self._ticks = 0

    def now(self) -> datetime:
        value = self._base + self._step * self._ticks
        self._ticks += 1
        return value


# --------------------------------------------------------------------------
# seal_terminal_receipt -- construction
# --------------------------------------------------------------------------

# ``time_rule``: these three complete inside the originating slot grace rather
# than inside the handoff window, because no handoff window exists yet.
SLOT_GRACE_REASONS = frozenset(
    {"marker-create-failure", "marker-collision", "marker-validation-failure"}
)


def watchdog_deadlines(
    consumed_at: datetime | None,
) -> tuple[str | None, str | None]:
    """``watchdog_rule``: both fields exist exactly with a valid marker."""

    if consumed_at is None:
        return None, None
    watchdog = consumed_at + timedelta(seconds=WATCHDOG_SECONDS)
    forced = watchdog + timedelta(seconds=FORCED_TERMINATION_GRACE_SECONDS)
    return runtime.canonical_utc(watchdog), runtime.canonical_utc(forced)


def build_seal_terminal_receipt(
    *,
    status: str,
    failure_stage: str | None,
    failure_reason: str | None,
    provisional_ready_core: Mapping[str, Any],
    marker: Mapping[str, Any] | None,
    slot_resolution: Mapping[str, Any] | None,
    segment_attestation: Mapping[str, Any],
    snapshot_set: Mapping[str, Any],
    post_build_observation: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
    marker_consumed_at: datetime | None,
    completed_at: str,
    previous_ledger_entry_sha256: str,
    sealer: Mapping[str, Any] | None,
    verifier: Mapping[str, Any] | None,
    plan_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the exact 20 ``fields_exactly`` in their frozen spelling."""

    watchdog_at, forced_at = watchdog_deadlines(marker_consumed_at)
    return {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "receipt_kind": SEAL_TERMINAL_RECEIPT_KIND,
        "status": status,
        "failure_stage_or_null": failure_stage,
        "failure_reason_or_null": failure_reason,
        "provisional_ready_core_sha256_and_bytes": require_identity(
            provisional_ready_core
        ),
        "seal_consumption_marker_sha256_and_bytes_or_null": (
            require_identity_or_null(marker)
        ),
        "slot_resolution_sha256_and_bytes_or_null": (
            require_identity_or_null(slot_resolution)
        ),
        "segment_attestation_sha256_and_bytes": require_identity(segment_attestation),
        "snapshot_set_sha256_and_bytes": require_identity(snapshot_set),
        "post_build_runtime_observation_sha256_and_bytes_or_null": (
            require_identity_or_null(post_build_observation)
        ),
        "manifest_sha256_and_bytes_or_null": require_identity_or_null(manifest),
        "watchdog_deadline_at_or_null": watchdog_at,
        "forced_termination_deadline_at_or_null": forced_at,
        "completed_at": completed_at,
        "previous_ledger_entry_sha256": previous_ledger_entry_sha256,
        "sealer_sha256_and_bytes_or_null": require_identity_or_null(sealer),
        "verifier_sha256_and_bytes_or_null": require_identity_or_null(verifier),
        "analysis_plan_sha256_and_bytes": require_identity(plan_identity),
    }


def validate_seal_terminal_receipt(
    raw: bytes,
    *,
    slot_index: int,
    expected_previous_ledger_sha256: str,
    expected_snapshot_set: Mapping[str, Any],
    expected_plan_identity: Mapping[str, Any],
    expected_marker_consumed_at: datetime | None,
    expected_bundle_visible: bool,
) -> dict[str, Any]:
    """Validate one ``seal-terminal`` receipt against the frozen schema.

    Every rule the plan states for this receipt is checked here: the exact
    field set, both enums, ``stage_reason_matrix``,
    ``sealed_conditional_fields``, ``failure_conditional_fields``,
    ``watchdog_rule``, ``time_rule``, ``marker_identity_rule``,
    ``ledger_predecessor_rule`` and the ``cross_artifact_rule`` binding on
    ``snapshot_set_sha256_and_bytes``.
    """

    receipt = load_canonical_receipt(raw, SEAL_TERMINAL_FIELDS)

    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or receipt["namespace"] != NAMESPACE
        or receipt["receipt_kind"] != SEAL_TERMINAL_RECEIPT_KIND
    ):
        _fail()

    status = receipt["status"]
    stage = receipt["failure_stage_or_null"]
    reason = receipt["failure_reason_or_null"]
    if (
        status not in SEAL_TERMINAL_STATUS_ENUM
        or stage not in FAILURE_STAGE_ENUM
        or reason not in FAILURE_REASON_ENUM
    ):
        _fail()

    marker = require_identity_or_null(
        receipt["seal_consumption_marker_sha256_and_bytes_or_null"]
    )
    slot_resolution = require_identity_or_null(
        receipt["slot_resolution_sha256_and_bytes_or_null"]
    )
    observation = require_identity_or_null(
        receipt["post_build_runtime_observation_sha256_and_bytes_or_null"]
    )
    manifest = require_identity_or_null(receipt["manifest_sha256_and_bytes_or_null"])
    sealer = require_identity_or_null(receipt["sealer_sha256_and_bytes_or_null"])
    verifier = require_identity_or_null(receipt["verifier_sha256_and_bytes_or_null"])
    require_identity(receipt["provisional_ready_core_sha256_and_bytes"])
    require_identity(receipt["segment_attestation_sha256_and_bytes"])

    # ``cross_artifact_rule``: one snapshot set across every present artifact.
    if not same_identity(
        receipt["snapshot_set_sha256_and_bytes"], expected_snapshot_set
    ) or not same_identity(
        receipt["analysis_plan_sha256_and_bytes"], expected_plan_identity
    ):
        _fail()

    if status == STATUS_SEALED:
        # ``sealed_conditional_fields``: null failure fields, and every
        # ``*_or_null`` artifact nonnull.  A sealed packet always has a
        # visible ready bundle behind it.
        if not expected_bundle_visible:
            _fail()
        if (
            stage is not None
            or reason is not None
            or marker is None
            or slot_resolution is None
            or observation is None
            or manifest is None
            or sealer is None
            or verifier is None
        ):
            _fail()
    else:
        # ``failure_conditional_fields``
        if (
            stage is None
            or reason is None
            or reason not in STAGE_REASON_MATRIX[stage]
        ):
            _fail()
        # ``marker_identity_rule``: marker-create-failure has a null marker
        # identity; marker-collision and marker-validation-failure *may*
        # carry the identity of resolvable invalid bytes, so null is also
        # valid there; every later stage requires a nonnull marker.
        if reason == "marker-create-failure":
            if marker is not None:
                _fail()
        elif stage not in {"marker-create", "marker-ledger"} and marker is None:
            _fail()
        # The slot resolution may be null only before the ready bundle became
        # visible.  That is a fact about the bundle, not about the stage: a
        # start-gate or blocked-child failure can fall on either side of the
        # one atomic rename.
        if (slot_resolution is not None) != expected_bundle_visible:
            _fail()
        if stage in POST_GATE_STAGES and not expected_bundle_visible:
            _fail()
        if stage in PRE_GATE_STAGES and (
            observation is not None or manifest is not None or verifier is not None
        ):
            _fail()
        if stage in {"build", "post-build-observation"} and manifest is not None:
            _fail()
        # At the validation stage both the manifest and the verifier phases
        # are past, so neither may still be null.
        if stage == "validation" and (manifest is None or verifier is None):
            _fail()
        if reason in {"observation-unavailable", "runtime-change"} and (
            observation is not None
        ):
            _fail()

    # ``watchdog_rule``: both watchdog fields are null exactly when no
    # semantically valid marker exists.
    watchdog_at, forced_at = watchdog_deadlines(expected_marker_consumed_at)
    if (
        receipt["watchdog_deadline_at_or_null"] != watchdog_at
        or receipt["forced_termination_deadline_at_or_null"] != forced_at
    ):
        _fail()
    # ``ledger_predecessor_rule`` is supplied by the caller: only the ceremony
    # knows whether H, M or R is the correct head for this outcome.
    if receipt["previous_ledger_entry_sha256"] != expected_previous_ledger_sha256:
        _fail()
    if (
        type(receipt["previous_ledger_entry_sha256"]) is not str
        or len(receipt["previous_ledger_entry_sha256"]) != 64
        or receipt["previous_ledger_entry_sha256"].strip("0123456789abcdef")
    ):
        _fail()

    _validate_seal_terminal_time_rule(
        receipt,
        slot_index=slot_index,
        consumed_at=expected_marker_consumed_at,
    )
    return receipt


def _validate_seal_terminal_time_rule(
    receipt: Mapping[str, Any],
    *,
    slot_index: int,
    consumed_at: datetime | None,
) -> None:
    """``time_rule``, in its four exact cases.  No deadline resets."""

    completed = runtime.parse_utc(receipt["completed_at"], receipt=True)
    slot = runtime.slot_times(slot_index)
    scheduled = runtime.parse_utc(slot.scheduled_at, receipt=True)
    grace = runtime.parse_utc(slot.grace_deadline_at, receipt=True)
    reason = receipt["failure_reason_or_null"]
    stage = receipt["failure_stage_or_null"]

    if reason in SLOT_GRACE_REASONS:
        if not scheduled <= completed < grace:
            _fail()
        return

    if consumed_at is None:
        _fail()
        return
    handoff_deadline = consumed_at + timedelta(seconds=HANDOFF_DEADLINE_SECONDS)
    watchdog = consumed_at + timedelta(seconds=WATCHDOG_SECONDS)
    forced = watchdog + timedelta(seconds=FORCED_TERMINATION_GRACE_SECONDS)

    if stage in PRE_GATE_STAGES:
        if not consumed_at <= completed <= handoff_deadline:
            _fail()
        return
    if reason == "forced-termination":
        if completed != forced:
            _fail()
        return
    if reason == "timeout":
        if not watchdog <= completed <= forced:
            _fail()
        return
    # Sealed success and every non-timeout post-gate failure.
    if not consumed_at <= completed <= watchdog:
        _fail()


# --------------------------------------------------------------------------
# The ``seal-terminal`` ledger entry -- this module owns it
# --------------------------------------------------------------------------


def build_seal_terminal_entry(
    *,
    entry_index: int,
    slot_index: int,
    receipt_raw: bytes,
    previous_ledger_entry_sha256: str,
) -> dict[str, Any]:
    """The one entry kind ``accrual.ENTRY_KINDS`` deliberately omits.

    ``derive_ledger_entry_sha256`` is reused rather than reimplemented, so the
    entry hash is the same function the rest of the chain is built from.
    """

    if SEAL_TERMINAL_ENTRY_KIND not in accrual.FROZEN_ENTRY_KINDS:  # pragma: no cover
        _fail()
    if SEAL_TERMINAL_ENTRY_KIND in accrual.ENTRY_KINDS:  # pragma: no cover
        # The accrual leaf must keep refusing to append this kind; if it ever
        # advertises support, two owners would exist for one artifact.
        _fail()
    entry = {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "entry_index": entry_index,
        "entry_kind": SEAL_TERMINAL_ENTRY_KIND,
        # ``slot_index_or_null``: every seal artifact carries its exact slot.
        "slot_index_or_null": slot_index,
        "artifact_sha256_and_bytes": identity_of_bytes(receipt_raw),
        "previous_ledger_entry_sha256": previous_ledger_entry_sha256,
        "entry_sha256": "",
    }
    entry["entry_sha256"] = accrual.derive_ledger_entry_sha256(entry)
    if set(entry) != set(accrual.LEDGER_ENTRY_FIELDS):  # pragma: no cover
        _fail()
    return entry


# --------------------------------------------------------------------------
# Inherited IPC: length-prefixed frames on anonymous pipes
# --------------------------------------------------------------------------

_FRAME_HEADER = struct.Struct(">Q")

READY_BUNDLE_DIRECTORY = "ready"
READY_RESOLUTION_NAME = "ready-resolution.json"
READY_ENTRY_NAME = "ready-ledger-entry.json"

CHILD_ENVELOPE_KEYS = (
    "builder_path",
    "draft_dir",
    "handoff_path",
    "ledger_head_m",
    "ledger_head_r",
    "marker_identity",
    "public_bundle",
    "publish_dir",
    "ready_entry_identity",
    "ready_resolution_identity",
    "slot_index",
    "snapshot_identities",
    "snapshot_paths",
    "verifier_sha256",
)


def send_frame(
    fd: int, payload: bytes, *, timeout_seconds: float = HANDOFF_DEADLINE_SECONDS
) -> None:
    """Write one frame under a deadline; a stopped child cannot stall us."""

    if type(payload) is not bytes or len(payload) > MAX_FRAME_BYTES:
        _fail()
    view = memoryview(_FRAME_HEADER.pack(len(payload)) + payload)
    deadline = time.monotonic() + timeout_seconds
    offset = 0
    while offset < len(view):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ChildProtocolFailure("ipc_deadline")
        _, writable, _ = select.select([], [fd], [], remaining)
        if not writable:
            raise ChildProtocolFailure("ipc_deadline")
        written = os.write(fd, view[offset:])
        if written <= 0:  # pragma: no cover - blocking pipes raise instead
            raise ChildProtocolFailure("ipc_write_failed")
        offset += written


def send_json_frame(fd: int, value: Any) -> None:
    send_frame(fd, canonical_bytes(value))


def _read_exactly(fd: int, count: int, deadline: float) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise ChildProtocolFailure("ipc_deadline")
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            raise ChildProtocolFailure("ipc_deadline")
        chunk = os.read(fd, min(remaining, 1 << 20))
        if not chunk:
            raise ChildProtocolFailure("ipc_eof")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(fd: int, *, timeout_seconds: float) -> bytes:
    deadline = time.monotonic() + timeout_seconds
    header = _read_exactly(fd, _FRAME_HEADER.size, deadline)
    (length,) = _FRAME_HEADER.unpack(header)
    if length > MAX_FRAME_BYTES:
        raise ChildProtocolFailure("ipc_frame_too_large")
    return _read_exactly(fd, length, deadline)


def recv_json_frame(fd: int, *, timeout_seconds: float) -> Any:
    return runtime.load_canonical_json_bytes(
        recv_frame(fd, timeout_seconds=timeout_seconds), expected=dict
    )


def open_start_gate(fd: int) -> None:
    """Write the single byte that releases the blocked child."""

    if os.write(fd, b"\x01") != 1:
        raise SealTerminalFailure("start-gate", "start-gate-failure")


def await_start_gate(fd: int, *, timeout_seconds: float) -> None:
    """Block until the supervisor opens the gate.  EOF is never an opening."""

    payload = _read_exactly(fd, 1, time.monotonic() + timeout_seconds)
    if payload != b"\x01":
        raise ChildProtocolFailure("start_gate_invalid")


# --------------------------------------------------------------------------
# The sealer child
# --------------------------------------------------------------------------


def _child_validate_bundle_bytes(
    envelope: Mapping[str, Any],
    resolution_raw: bytes,
    entry_raw: bytes,
) -> None:
    """Reject anything that is not the exact ready resolution and its entry.

    The child never trusts the supervisor's summary of the bundle: it derives
    ``R`` itself from the entry bytes, and binds the entry to the resolution
    bytes and to ``M`` before it agrees that the handoff is coherent.
    """

    if not same_identity(
        identity_of_bytes(resolution_raw), envelope["ready_resolution_identity"]
    ) or not same_identity(
        identity_of_bytes(entry_raw), envelope["ready_entry_identity"]
    ):
        raise ChildProtocolFailure("bundle_identity_mismatch")

    entry = load_canonical_receipt(entry_raw, accrual.LEDGER_ENTRY_FIELDS)
    if (
        entry["schema_version"] != SCHEMA_VERSION
        or entry["namespace"] != NAMESPACE
        or entry["entry_kind"] != "ready"
        or entry["slot_index_or_null"] != envelope["slot_index"]
        or entry["previous_ledger_entry_sha256"] != envelope["ledger_head_m"]
        or not same_identity(
            entry["artifact_sha256_and_bytes"], identity_of_bytes(resolution_raw)
        )
        or entry["entry_sha256"] != envelope["ledger_head_r"]
        or accrual.derive_ledger_entry_sha256(entry) != envelope["ledger_head_r"]
    ):
        raise ChildProtocolFailure("bundle_entry_invalid")

    resolution = load_canonical_receipt(resolution_raw, probe.PROBE_RESOLUTION_FIELDS)
    if (
        resolution["receipt_kind"] != "slot-probe-resolution"
        or resolution["status"] != "ready"
        or resolution["slot_index"] != envelope["slot_index"]
        or resolution["previous_ledger_entry_sha256"] != envelope["ledger_head_m"]
        or not same_identity(
            resolution["seal_consumption_marker_sha256_and_bytes_or_null"],
            envelope["marker_identity"],
        )
        or resolution["sealer_process_launched_at_or_null"] is None
    ):
        raise ChildProtocolFailure("bundle_resolution_invalid")


def _child_reopen_public_bundle(
    envelope: Mapping[str, Any],
    resolution_raw: bytes,
    entry_raw: bytes,
) -> None:
    """Step seven: reopen and validate the public bundle before any read."""

    bundle = Path(envelope["public_bundle"])
    try:
        public_resolution = runtime.read_regular_file(
            bundle / READY_RESOLUTION_NAME,
            expected=runtime.hash_and_bytes(resolution_raw),
            maximum_bytes=len(resolution_raw),
        )
        public_entry = runtime.read_regular_file(
            bundle / READY_ENTRY_NAME,
            expected=runtime.hash_and_bytes(entry_raw),
            maximum_bytes=len(entry_raw),
        )
    except (IntegrityFailure, OSError):
        raise ChildProtocolFailure("public_bundle_unreadable") from None
    if not hmac.compare_digest(
        public_resolution, resolution_raw
    ) or not hmac.compare_digest(public_entry, entry_raw):
        raise ChildProtocolFailure("public_bundle_mismatch")
    # Re-derive rather than re-compare: the public entry must still hash to R.
    _child_validate_bundle_bytes(envelope, public_resolution, public_entry)


def _child_verify_retained_snapshots(
    envelope: Mapping[str, Any], snapshot_fds: Mapping[str, int]
) -> None:
    """Bind each snapshot path to the exact descriptor retained at probe time.

    The child inherits the retained read-only descriptors precisely so a path
    cannot be swapped between the probe and the build.  It reads them only
    here, after the gate.
    """

    for alias in SOURCE_ALIASES:
        fd = snapshot_fds[alias]
        os.lseek(fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        retained = {"sha256": digest.hexdigest(), "bytes": size}
        if not same_identity(retained, envelope["snapshot_identities"][alias]):
            raise ChildProtocolFailure("retained_snapshot_mismatch")
        if not same_identity(retained, identity_of_path(envelope["snapshot_paths"][alias])):
            raise ChildProtocolFailure("snapshot_path_diverged")


def _run_builder(arguments: Sequence[str], *, timeout: float) -> dict[str, Any]:
    """Drive the packet builder through its fixed CLI and nothing else."""

    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-B", os.fspath(BUILDER_PATH), *arguments],
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise ChildProtocolFailure("builder_nonzero")
    try:
        return json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ChildProtocolFailure("builder_output_invalid") from None


def run_sealer_child(
    *,
    gate_fd: int,
    inbound_fd: int,
    outbound_fd: int,
    snapshot_fds: Mapping[str, int],
) -> int:
    """The spawned sealer: blocked, then validating, then building, then last
    of all publishing.

    Nothing before ``await_start_gate`` touches a snapshot, the builder or the
    publish root.  The ordering here is the child half of
    ``ordering_exactly`` steps four and seven.
    """

    envelope = recv_json_frame(inbound_fd, timeout_seconds=HANDOFF_DEADLINE_SECONDS)
    require_exact_keys(envelope, CHILD_ENVELOPE_KEYS, "child_envelope")
    resolution_raw = recv_frame(inbound_fd, timeout_seconds=HANDOFF_DEADLINE_SECONDS)
    entry_raw = recv_frame(inbound_fd, timeout_seconds=HANDOFF_DEADLINE_SECONDS)

    # Step five, child half: validate the still-invisible bundle.
    _child_validate_bundle_bytes(envelope, resolution_raw, entry_raw)
    send_json_frame(
        outbound_fd,
        {
            "stage": "ipc-validated",
            "ready_entry_sha256": envelope["ledger_head_r"],
            "ready_resolution_sha256": envelope["ready_resolution_identity"]["sha256"],
        },
    )

    # Step seven: nothing below this line may run before the gate opens.
    await_start_gate(gate_fd, timeout_seconds=HANDOFF_DEADLINE_SECONDS)
    _child_reopen_public_bundle(envelope, resolution_raw, entry_raw)
    _child_verify_retained_snapshots(envelope, snapshot_fds)

    draft = _run_builder(
        (
            "draft",
            "--draft-dir",
            envelope["draft_dir"],
            "--handoff",
            envelope["handoff_path"],
        ),
        timeout=CHILD_BUILD_TIMEOUT_SECONDS,
    )
    if draft.get("status") != "draft" or not draft.get("floors_pass"):
        raise ChildProtocolFailure("builder_draft_invalid")
    send_json_frame(
        outbound_fd,
        {"stage": "built", "aggregate_counts": draft["aggregate_counts"]},
    )

    # The supervisor owns the post-build runtime and binding observation; the
    # child publishes only once that observation has matched the handoff.
    instruction = recv_json_frame(
        inbound_fd, timeout_seconds=float(WATCHDOG_SECONDS)
    )
    if instruction.get("stage") != "publish":
        raise ChildProtocolFailure("publish_instruction_invalid")

    frozen = _run_builder(
        (
            "freeze",
            "--draft-dir",
            envelope["draft_dir"],
            "--publish-dir",
            envelope["publish_dir"],
            "--verifier-sha256",
            envelope["verifier_sha256"],
        ),
        timeout=CHILD_BUILD_TIMEOUT_SECONDS,
    )
    if frozen.get("status") != "sealed":
        raise ChildProtocolFailure("builder_freeze_invalid")
    send_json_frame(
        outbound_fd,
        {
            "stage": "published",
            "manifest": require_identity(frozen["packet_manifest_sha256_and_bytes"]),
            "aggregate_counts": frozen["aggregate_counts"],
        },
    )
    return 0


# --------------------------------------------------------------------------
# Consumed authority: an existing valid marker is never a retry permit
# --------------------------------------------------------------------------

MARKER_SUFFIX = ".seal-consumption.json"


def validate_seal_consumption_marker_structure(raw: bytes) -> dict[str, Any]:
    """Validate an existing marker without a live ledger state.

    A second invocation cannot rebuild the process-local capabilities that
    ``accrual.validate_seal_consumption_marker`` requires, so consumed
    authority is established structurally: the exact frozen field set, the
    frozen authority id, and resolvable identities.  That is enough to prove
    the marker is consumed authority rather than a retry permit.
    """

    receipt = load_canonical_receipt(raw, accrual.SEAL_CONSUMPTION_FIELDS)
    slot_index = receipt["slot_index"]
    segment_id = receipt["segment_id"]
    previous = receipt["previous_ledger_entry_sha256"]
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or receipt["namespace"] != NAMESPACE
        or receipt["receipt_kind"] != "seal-consumption"
        or receipt["seal_authority_id"] != SEAL_AUTHORITY_ID
        or not _is_int(slot_index)
        or not 0 <= slot_index <= 28
        or type(segment_id) is not str
        or len(segment_id) != 64
        or segment_id.strip("0123456789abcdef")
        or type(previous) is not str
        or len(previous) != 64
        or previous.strip("0123456789abcdef")
    ):
        _fail()
    require_identity(receipt["provisional_ready_core_sha256_and_bytes"])
    require_identity(receipt["snapshot_set_sha256_and_bytes"])
    require_identity(receipt["sealer_sha256_and_bytes"])
    require_identity(receipt["analysis_plan_sha256_and_bytes"])
    runtime.parse_utc(receipt["consumed_at"], receipt=True)
    return receipt


def find_consumed_marker(marker_directory: Path) -> tuple[Path, bytes] | None:
    """Return the single existing marker, or None when authority is unspent."""

    try:
        names = sorted(
            item.name
            for item in marker_directory.iterdir()
            if item.name.endswith(MARKER_SUFFIX)
        )
    except FileNotFoundError:
        return None
    except OSError:
        raise SealTerminalFailure("marker-create", "marker-create-failure") from None
    if not names:
        return None
    if len(names) > 1:
        # More than one marker is a second marker by definition.
        raise SealTerminalFailure("marker-create", "marker-collision")
    path = marker_directory / names[0]
    return path, runtime.read_regular_file(path, maximum_bytes=MAX_RECEIPT_BYTES)


def build_consumed_authority_receipt(
    *,
    marker_raw: bytes,
    contract: runtime.FrozenContract,
    sealer_identity: Mapping[str, Any],
    completed_at: str,
) -> dict[str, Any]:
    """One terminal receipt for a second attempt against consumed authority.

    It is emitted, never appended.  The durable chain has already advanced
    past ``H`` to ``M``, so writing a ``seal-terminal`` entry from ``H`` would
    create exactly the branch ``ledger_predecessor_rule`` forbids.
    """

    marker = validate_seal_consumption_marker_structure(marker_raw)
    if marker["segment_id"] != runtime.INITIAL_SEGMENT_ID:
        # Only the initial segment's attestation is pinned by the plan; a
        # successor attestation cannot be resolved without its ledger.
        _fail()
    if not same_identity(marker["sealer_sha256_and_bytes"], sealer_identity):
        _fail()
    if not same_identity(
        marker["analysis_plan_sha256_and_bytes"], contract.plan_identity.as_dict()
    ):
        _fail()
    receipt = build_seal_terminal_receipt(
        status=STATUS_TERMINAL,
        failure_stage="marker-create",
        failure_reason="marker-collision",
        provisional_ready_core=marker["provisional_ready_core_sha256_and_bytes"],
        marker=identity_of_bytes(marker_raw),
        slot_resolution=None,
        segment_attestation=contract.watermark_identity.as_dict(),
        snapshot_set=marker["snapshot_set_sha256_and_bytes"],
        post_build_observation=None,
        manifest=None,
        marker_consumed_at=runtime.parse_utc(marker["consumed_at"], receipt=True),
        completed_at=completed_at,
        previous_ledger_entry_sha256=marker["previous_ledger_entry_sha256"],
        sealer=marker["sealer_sha256_and_bytes"],
        verifier=None,
        plan_identity=contract.plan_identity.as_dict(),
    )
    validate_seal_terminal_receipt(
        canonical_bytes(receipt),
        slot_index=marker["slot_index"],
        expected_previous_ledger_sha256=marker["previous_ledger_entry_sha256"],
        expected_snapshot_set=marker["snapshot_set_sha256_and_bytes"],
        expected_plan_identity=contract.plan_identity.as_dict(),
        expected_marker_consumed_at=runtime.parse_utc(
            marker["consumed_at"], receipt=True
        ),
        expected_bundle_visible=False,
    )
    return receipt


# --------------------------------------------------------------------------
# The seal ceremony
# --------------------------------------------------------------------------

STAGING_PREFIX = ".staging-ready-"


class SealCeremony:
    """One invocation of ``readiness.atomic_ready_handoff``, in order.

    The whole ceremony runs in a single process from the probe computation
    through atomic ready visibility.  It has to: ``accrual`` derives its
    capability keys with ``os.urandom`` at import, so ``LedgerState``,
    ``DurableSealConsumption`` and ``BlockedSpawnProof`` cannot cross a
    process boundary.  Only exact byte strings cross to the child.
    """

    def __init__(
        self,
        *,
        clock: SealClock,
        state: Any,
        computation: Any,
        observe_active_state: Callable[[], tuple[Any, Any]],
        runtime_observer_identity: Mapping[str, Any],
        seal_root: Path,
        draft_dir: Path,
        publish_dir: Path,
        handoff_path: Path,
        snapshot_paths: Mapping[str, Path],
        snapshot_fds: Mapping[str, int],
        sealer_identity: Mapping[str, Any],
        verifier_identity: Mapping[str, Any],
    ) -> None:
        self.clock = clock
        self.state = state
        self.computation = computation
        self.observe_active_state = observe_active_state
        self.runtime_observer_identity = require_identity(runtime_observer_identity)
        self.seal_root = seal_root
        self.ledger_directory = seal_root / "ledger"
        self.marker_directory = seal_root / "markers"
        self.objects_root = seal_root
        self.bundle_path = seal_root / READY_BUNDLE_DIRECTORY
        self.draft_dir = draft_dir
        self.publish_dir = publish_dir
        self.handoff_path = handoff_path
        self.snapshot_paths = dict(snapshot_paths)
        self.snapshot_fds = dict(snapshot_fds)
        self.sealer_identity = require_identity(sealer_identity)
        self.verifier_identity = require_identity(verifier_identity)

        # The builder deliberately permits a namespace publish root on its
        # ``freeze`` path.  This launcher never does: it seals no real packet,
        # so every root it is handed is refused inside the namespace.
        for root in (seal_root, draft_dir, publish_dir, handoff_path.parent):
            require_output_root_outside_namespace(root)

        self.contract = state.contract
        self.active_segment = state.active_segment
        self.slot_index = state.next_slot_index

        # Everything the terminal receipt may need, filled in as it is proven.
        self.head_h: str = state.head
        self.head_m: str | None = None
        self.head_r: str | None = None
        self.marker_identity: dict[str, Any] | None = None
        self.marker_consumed_at: datetime | None = None
        self.provisional_core_identity: dict[str, Any] | None = None
        self.snapshot_set_identity: dict[str, Any] | None = None
        self.slot_resolution_identity: dict[str, Any] | None = None
        self.observation_identity: dict[str, Any] | None = None
        self.manifest_identity: dict[str, Any] | None = None
        self.child: subprocess.Popen[bytes] | None = None
        self.bundle_visible = False
        self.process_launches = 0
        self.publication_attempts = 0
        self._handoff_monotonic_deadline: float | None = None
        self.aggregate_counts: dict[str, Any] | None = None
        self.steps: list[str] = []

    # -- receipt plumbing ---------------------------------------------------

    def _predecessor_for(self, stage: str | None) -> str:
        """``ledger_predecessor_rule``, resolved against the durable head.

        ``R`` is the head from the instant the bundle is atomically visible,
        so a failure after that point appends from ``R``.  Appending from
        ``M`` once ``R`` exists would create exactly the branch the rule's
        own closing clause forbids.
        """

        if self.bundle_visible:
            head = self.head_r
        elif self.head_m is not None:
            head = self.head_m
        else:
            head = self.head_h
        if head is None:
            _fail()
        if stage in PREDECESSOR_H_STAGES and head != self.head_h:
            _fail()
        if (stage is None or stage in POST_GATE_STAGES) and head != self.head_r:
            _fail()
        return head

    def _receipt(
        self, *, status: str, stage: str | None, reason: str | None
    ) -> dict[str, Any]:
        if self.provisional_core_identity is None or self.snapshot_set_identity is None:
            # No provisional core means the ceremony never reached step one and
            # no seal-terminal receipt is constructible; escalate instead.
            _fail()
        return build_seal_terminal_receipt(
            status=status,
            failure_stage=stage,
            failure_reason=reason,
            provisional_ready_core=self.provisional_core_identity,
            marker=self.marker_identity,
            slot_resolution=(
                self.slot_resolution_identity if self.bundle_visible else None
            ),
            segment_attestation=self.active_segment.attestation_identity.as_dict(),
            snapshot_set=self.snapshot_set_identity,
            post_build_observation=self.observation_identity,
            manifest=self.manifest_identity,
            marker_consumed_at=self.marker_consumed_at,
            completed_at=self.clock.canonical_now(),
            previous_ledger_entry_sha256=self._predecessor_for(stage),
            sealer=self.sealer_identity,
            verifier=self.verifier_identity if status == STATUS_SEALED else None,
            plan_identity=self.contract.plan_identity.as_dict(),
        )

    def _publish_receipt(self, receipt: Mapping[str, Any]) -> bytes:
        """Validate, durably record and append the ``seal-terminal`` entry."""

        raw = canonical_bytes(receipt)
        validate_seal_terminal_receipt(
            raw,
            slot_index=self.slot_index,
            expected_previous_ledger_sha256=receipt["previous_ledger_entry_sha256"],
            expected_snapshot_set=self.snapshot_set_identity,
            expected_plan_identity=self.contract.plan_identity.as_dict(),
            expected_marker_consumed_at=self.marker_consumed_at,
            expected_bundle_visible=self.bundle_visible,
        )
        entry_index = self._terminal_entry_index()
        if entry_index is not None:
            # ``accrual`` cannot build this entry kind, so the prefix check
            # ``write_ledger_entry_exclusive`` performs is done explicitly
            # before the same single O_EXCL write.
            # At every index this returns, ``state.entries[-1]`` is already
            # durable: either M, or R once the bundle is visible.
            accrual.validate_persisted_ledger_prefix(
                self.ledger_directory, self.state
            )
            entry = build_seal_terminal_entry(
                entry_index=entry_index,
                slot_index=self.slot_index,
                receipt_raw=raw,
                previous_ledger_entry_sha256=receipt["previous_ledger_entry_sha256"],
            )
            accrual.write_exclusive(
                self.ledger_directory / accrual.ledger_entry_filename(entry_index),
                canonical_bytes(entry),
            )
        target = self.publish_dir / SEAL_RECEIPT_FILENAME
        if target.parent.is_dir():
            # ``consumption_record_location``: a separate hash-bound aggregate
            # receipt, never written back into the immutable manifest.
            accrual.write_exclusive(target, raw)
        return raw

    def _terminal_entry_index(self) -> int | None:
        """The chain may only be extended from its own durable head.

        Before ``M`` is durable the ledger has no successor to extend, and a
        ``seal-terminal`` entry from ``H`` would branch around an existing
        ``M``; in that case the receipt is emitted and never appended.  The
        discriminator is durable visibility, not the in-memory ``head_r``:
        between appending ``R`` and exposing it, the entry file does not yet
        exist and appending would both skip an index and branch.
        """

        if self.head_m is None:
            return None
        if self.head_r is not None and not self.bundle_visible:
            return None
        return self.state.entries[-1].entry_index + 1

    # -- step one -----------------------------------------------------------

    def step_one_provisional_core(self) -> Any:
        """Compute the provisional ready core privately while the head is H."""

        self._final_stable_runtime_check(stage="marker-create")
        launched_at = runtime.canonical_utc(self.state.current_probe_attempt.launched_at)
        # ``validated_at`` is precommitted here on purpose: the eventual ready
        # receipt must be validated after the blocked sealer launch, and the
        # core that binds it is fixed before the marker exists.
        validated_at = runtime.canonical_utc(
            self.clock.now()
            + timedelta(seconds=PRECOMMITTED_READY_OFFSET_SECONDS)
        )
        provisional = accrual.prepare_provisional_ready(
            self.state,
            self.computation,
            launched_at=launched_at,
            validated_at=validated_at,
        )
        self.provisional_core_identity = provisional.core_identity.as_dict()
        self.snapshot_set_identity = provisional.snapshot_set_identity.as_dict()
        self.steps.append(ORDERING_EXACTLY[0])
        return provisional

    def _final_stable_runtime_check(self, *, stage: str) -> tuple[Any, Any]:
        """One complete runtime and binding observation against the segment."""

        try:
            services, binding = self.observe_active_state()
        except (SealTerminalFailure, IntegrityFailure):
            raise
        except Exception:
            raise SealTerminalFailure(
                stage,
                "observation-unavailable"
                if stage == "post-build-observation"
                else "marker-create-failure",
            ) from None
        try:
            runtime.validate_active_segment_observations(
                self.active_segment,
                services,
                binding,
                services,
                binding,
                contract=self.contract,
            )
        except IntegrityFailure:
            if stage == "post-build-observation":
                raise SealTerminalFailure(stage, "runtime-change") from None
            raise
        return services, binding


    # -- steps two and three ------------------------------------------------

    def steps_two_and_three_consume_authority(self, provisional: Any) -> tuple[Any, Any]:
        """Create the marker durably, then append M.  Authority is now spent.

        Consumption happens here, at marker creation -- not at manifest write.
        ``persist_seal_consumption_exclusive`` performs exactly the frozen
        order: O_EXCL-create and fsync the marker, then establish the ledger
        entry whose predecessor is H.
        """

        existing = find_consumed_marker(self.marker_directory)
        if existing is not None:
            # An existing valid marker is consumed authority, never a
            # collision authorizing a replacement child.  ``marker_identity_rule``
            # permits carrying the identity of bytes that resolve.
            self.marker_identity = identity_of_bytes(existing[1])
            raise SealTerminalFailure("marker-create", "marker-collision")

        consumed_at = runtime.canonical_utc(self.clock.now())
        # One 30-second window for every pre-gate stage, opened at the marker
        # instant itself and never reset on recovery.
        self._handoff_monotonic_deadline = (
            time.monotonic() + HANDOFF_DEADLINE_SECONDS
        )
        marker_value = {
            "schema_version": SCHEMA_VERSION,
            "namespace": NAMESPACE,
            "receipt_kind": "seal-consumption",
            "seal_authority_id": SEAL_AUTHORITY_ID,
            "slot_index": provisional.slot_index,
            "segment_id": provisional.segment_id,
            "provisional_ready_core_sha256_and_bytes": (
                provisional.core_identity.as_dict()
            ),
            "snapshot_set_sha256_and_bytes": (
                provisional.snapshot_set_identity.as_dict()
            ),
            "consumed_at": consumed_at,
            "previous_ledger_entry_sha256": self.head_h,
            "sealer_sha256_and_bytes": dict(self.sealer_identity),
            "analysis_plan_sha256_and_bytes": self.contract.plan_identity.as_dict(),
        }
        if set(marker_value) != set(accrual.SEAL_CONSUMPTION_FIELDS):  # pragma: no cover
            _fail()
        marker_raw = canonical_bytes(marker_value)

        try:
            marker = accrual.validate_seal_consumption_marker(
                marker_raw,
                state=self.state,
                provisional=provisional,
                expected_sealer_identity=self.sealer_identity,
            )
        except IntegrityFailure:
            self.marker_identity = identity_of_bytes(marker_raw)
            raise SealTerminalFailure(
                "marker-ledger", "marker-validation-failure"
            ) from None

        consumed_state = accrual.append_seal_consumption(self.state, marker)
        marker_path = self.marker_directory / accrual.seal_consumption_filename(
            marker, consumed_state.entries[-1].entry_index
        )
        try:
            durable = accrual.persist_seal_consumption_exclusive(
                consumed_state,
                marker_path=marker_path,
                ledger_directory=self.ledger_directory,
            )
        except IntegrityFailure:
            raise self._classify_persist_failure(marker_path, marker_raw) from None

        # Only now is authority spent, and only now does M exist.
        self.state = consumed_state
        self.marker_identity = marker.identity.as_dict()
        self.marker_consumed_at = marker.consumed_at
        self.head_m = consumed_state.head
        self.steps.append(ORDERING_EXACTLY[1])
        self.steps.append(ORDERING_EXACTLY[2])
        return marker, durable

    def _classify_persist_failure(
        self, marker_path: Path, marker_raw: bytes
    ) -> SealTerminalFailure:
        """Read only the unique expected entry after a write or fsync error.

        An exact valid intended marker is consumed authority; proven absence
        terminalizes from H; anything else is integrity escalation with no
        alternate ledger branch.
        """

        try:
            observed = runtime.read_regular_file(
                marker_path, maximum_bytes=MAX_RECEIPT_BYTES
            )
        except (IntegrityFailure, OSError):
            if marker_path.exists():
                # Present but unresolvable: invalid or partial bytes.
                _fail()
            return SealTerminalFailure("marker-create", "marker-create-failure")
        self.marker_identity = identity_of_bytes(observed)
        if not hmac.compare_digest(observed, marker_raw):
            return SealTerminalFailure("marker-create", "marker-collision")
        # The marker is exactly ours, so the failure was the ledger write.
        return SealTerminalFailure("marker-ledger", "marker-ledger-failure")

    # -- step four ----------------------------------------------------------

    def step_four_spawn_blocked_child(
        self, marker: Any, durable: Any
    ) -> tuple[Any, dict[str, int]]:
        """Spawn one sealer behind an inherited, still-closed start gate."""

        if self.process_launches >= MAX_PROCESS_LAUNCHES:
            # ``authority.seal.max_process_launches`` is one; a consumed
            # marker never authorizes a replacement child.
            raise SealTerminalFailure("spawn", "spawn-failure")
        self.process_launches += 1
        gate_read, gate_write = os.pipe()
        down_read, down_write = os.pipe()
        up_read, up_write = os.pipe()
        inherited = {
            "gate": gate_read,
            "inbound": down_read,
            "outbound": up_write,
            **{f"snapshot_{alias}": self.snapshot_fds[alias] for alias in SOURCE_ALIASES},
        }
        for fd in inherited.values():
            os.set_inheritable(fd, True)
        try:
            child = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                [
                    sys.executable,
                    "-B",
                    os.fspath(Path(__file__).resolve()),
                    "__child",
                    "--gate-fd",
                    str(gate_read),
                    "--inbound-fd",
                    str(down_read),
                    "--outbound-fd",
                    str(up_write),
                    "--snapshot-fd",
                    *(
                        f"{alias}={self.snapshot_fds[alias]}"
                        for alias in SOURCE_ALIASES
                    ),
                ],
                pass_fds=tuple(inherited.values()),
                close_fds=True,
                start_new_session=True,
            )
        except OSError:
            for fd in (gate_read, gate_write, down_read, down_write, up_read, up_write):
                with contextlib.suppress(OSError):
                    os.close(fd)
            raise SealTerminalFailure("spawn", "spawn-failure") from None

        for fd in (gate_read, down_read, up_write):
            with contextlib.suppress(OSError):
                os.close(fd)
        self.child = child
        channels = {"gate": gate_write, "inbound": down_write, "outbound": up_read}

        launched_at = runtime.canonical_utc(self.clock.now())
        try:
            spawned = accrual.record_blocked_spawn(
                self.state,
                marker=marker,
                durable_marker=durable,
                sealer_process_launched_at=launched_at,
            )
        except IntegrityFailure:
            for fd in channels.values():
                with contextlib.suppress(OSError):
                    os.close(fd)
            self.child = None if child.poll() is not None else child
            raise SealTerminalFailure("spawn", "spawn-failure") from None
        self.state = spawned
        self.steps.append(ORDERING_EXACTLY[3])
        return spawned, channels

    # -- step five ----------------------------------------------------------

    def step_five_stage_and_deliver(
        self, spawned: Any, channels: Mapping[str, int]
    ) -> tuple[str, bytes, bytes]:
        """Stage the ready resolution and R privately, then deliver both."""

        try:
            ready_value = accrual.build_ready_resolution(
                spawned, blocked_spawn=spawned.blocked_spawn
            )
            ready_raw = canonical_bytes(ready_value)
            resolution = probe.validate_probe_resolution(
                ready_raw,
                contract=self.contract,
                active_segment=self.active_segment,
                expected_previous_ledger_sha256=spawned.head,
                expected_probe_identity=spawned.provisional_ready.probe_identity,
                expected_snapshot_identities=dict(
                    spawned.provisional_ready.snapshot_items
                ),
                expected_seal_consumption_identity=spawned.seal_consumption.identity,
                expected_sealer_process_launched_at=runtime.canonical_utc(
                    spawned.blocked_spawn.launched_at
                ),
            )
            latched = accrual.append_ready_resolution(
                spawned, resolution, blocked_spawn=spawned.blocked_spawn
            )
        except IntegrityFailure:
            raise SealTerminalFailure(
                "ready-staging", "ready-ledger-failure"
            ) from None

        entry_raw = latched.entries[-1].raw
        head_r = latched.entries[-1].entry_sha256
        # ``cross_artifact_rule``: the ready resolution's snapshot set is the
        # same identity the marker bound.
        if not same_identity(
            accrual.derive_snapshot_set_identity(
                ready_value["aliased_source_snapshot_sha256_and_bytes"]
            ).as_dict(),
            self.snapshot_set_identity,
        ):
            _fail()

        staging = self.seal_root / f"{STAGING_PREFIX}{os.urandom(8).hex()}"
        try:
            make_private_directory(staging)
            accrual.write_exclusive(staging / READY_RESOLUTION_NAME, ready_raw)
            accrual.write_exclusive(staging / READY_ENTRY_NAME, entry_raw)
            fsync_directory(staging)
        except (IntegrityFailure, OSError):
            raise SealTerminalFailure("ready-staging", "staging-failure") from None

        envelope = {
            "builder_path": os.fspath(BUILDER_PATH),
            "draft_dir": os.fspath(self.draft_dir),
            "handoff_path": os.fspath(self.handoff_path),
            "ledger_head_m": self.head_m,
            "ledger_head_r": head_r,
            "marker_identity": dict(self.marker_identity),
            "public_bundle": os.fspath(self.bundle_path),
            "publish_dir": os.fspath(self.publish_dir),
            "ready_entry_identity": identity_of_bytes(entry_raw),
            "ready_resolution_identity": identity_of_bytes(ready_raw),
            "slot_index": self.slot_index,
            "snapshot_identities": {
                alias: identity_of_path(self.snapshot_paths[alias])
                for alias in SOURCE_ALIASES
            },
            "snapshot_paths": {
                alias: os.fspath(self.snapshot_paths[alias])
                for alias in SOURCE_ALIASES
            },
            "verifier_sha256": self.verifier_identity["sha256"],
        }
        require_exact_keys(envelope, CHILD_ENVELOPE_KEYS, "child_envelope")
        try:
            send_json_frame(channels["inbound"], envelope)
            send_frame(channels["inbound"], ready_raw)
            send_frame(channels["inbound"], entry_raw)
            acknowledgement = recv_json_frame(
                channels["outbound"],
                timeout_seconds=self._remaining_handoff_seconds(),
            )
        except ChildProtocolFailure as error:
            raise self._handoff_stage_failure(
                "ready-ipc", error, "ipc-validation-failure"
            ) from None
        except OSError:
            raise SealTerminalFailure("ready-ipc", "ipc-validation-failure") from None
        if (
            acknowledgement.get("stage") != "ipc-validated"
            or acknowledgement.get("ready_entry_sha256") != head_r
            or acknowledgement.get("ready_resolution_sha256")
            != envelope["ready_resolution_identity"]["sha256"]
        ):
            raise SealTerminalFailure("ready-ipc", "ipc-validation-failure")

        self.state = latched
        self.head_r = head_r
        self.slot_resolution_identity = identity_of_bytes(ready_raw)
        self.steps.append(ORDERING_EXACTLY[4])
        return staging.name, ready_raw, entry_raw

    def _remaining_handoff_seconds(self) -> float:
        """Real elapsed liveness bound on the shared pre-gate deadline.

        Every pre-gate stage shares one 30-second window opened at marker
        consumption, and it never resets on recovery.
        """

        if self._handoff_monotonic_deadline is None:  # pragma: no cover
            _fail()
        return max(0.0, self._handoff_monotonic_deadline - time.monotonic())

    def _handoff_stage_failure(
        self, stage: str, error: ChildProtocolFailure, default: str
    ) -> SealTerminalFailure:
        if error.code == "ipc_deadline":
            return SealTerminalFailure(stage, "handoff-timeout")
        if error.code == "ipc_eof":
            return SealTerminalFailure("blocked-child", "blocked-child-eof")
        return SealTerminalFailure(stage, default)

    # -- steps six and seven ------------------------------------------------

    def step_six_atomic_visibility(self, staging_name: str) -> None:
        """One atomic no-replace rename exposes the resolution and R together."""

        parent_fd = open_directory_fd(self.seal_root)
        try:
            rename_directory_no_replace(
                parent_fd, staging_name, READY_BUNDLE_DIRECTORY
            )
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        # The ledger's own entry index for R is written only now, from bytes
        # that are already public, so no state exposes either artifact alone.
        try:
            accrual.write_ledger_entry_exclusive(self.ledger_directory, self.state)
        except IntegrityFailure:
            raise SealTerminalFailure(
                "ready-visibility", "atomic-visibility-failure"
            ) from None
        # From here on the ready resolution and R are public together, and R
        # is the head every later receipt appends from.
        self.bundle_visible = True
        self.steps.append(ORDERING_EXACTLY[5])

    def step_seven_open_gate(self, channels: Mapping[str, int]) -> None:
        """Release the child only after the bundle is atomically visible."""

        try:
            open_start_gate(channels["gate"])
        except OSError:
            raise SealTerminalFailure("start-gate", "start-gate-failure") from None
        self.steps.append(ORDERING_EXACTLY[6])

    # -- build, observe, publish -------------------------------------------

    def await_build(self, channels: Mapping[str, int]) -> None:
        """Content is built by the child; only the build stage can fail here.

        A build that overruns its bound is a build failure, not a watchdog
        expiry: ``time_rule`` puts a ``watchdog`` completion inside
        ``[watchdog_deadline_at, forced_termination_deadline_at]``, which a
        bounded build overrun is not.
        """

        try:
            message = recv_json_frame(
                channels["outbound"], timeout_seconds=float(CHILD_BUILD_TIMEOUT_SECONDS)
            )
        except (ChildProtocolFailure, IntegrityFailure, OSError):
            raise SealTerminalFailure("build", "build-failure") from None
        if (
            type(message) is not dict
            or message.get("stage") != "built"
            or type(message.get("aggregate_counts")) is not dict
        ):
            raise SealTerminalFailure("build", "build-failure")
        self.aggregate_counts = dict(message["aggregate_counts"])

    def _retain_object(self, raw: bytes) -> dict[str, Any]:
        """Store one retained object at its content-addressed namespace path."""

        identity = runtime.hash_and_bytes(raw)
        directory = self.objects_root / "objects" / "sha256" / identity.sha256[:2]
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        target = directory / identity.sha256[2:]
        if target.exists():
            accrual.recover_exact_file(target, raw)
        else:
            accrual.write_exclusive(target, raw)
        # Prove the declared identity resolves through the frozen rule.
        resolved = runtime.read_content_addressed_object(
            self.objects_root, identity
        )
        if not hmac.compare_digest(resolved, raw):
            _fail()
        return identity.as_dict()

    def post_build_runtime_observation(self) -> bytes:
        """A second complete runtime and binding observation before publish.

        A change or an unavailable observation after authority consumption is
        terminal seal failure, never a return to accrual.
        """

        try:
            return self._post_build_runtime_observation()
        except SealTerminalFailure:
            raise
        except (IntegrityFailure, OSError):
            # After authority consumption this phase is terminal, never a
            # return to accrual.
            raise SealTerminalFailure(
                "post-build-observation", "observation-unavailable"
            ) from None

    def _post_build_runtime_observation(self) -> bytes:
        services, binding = self._final_stable_runtime_check(
            stage="post-build-observation"
        )
        observation = {
            "schema_version": SCHEMA_VERSION,
            "namespace": NAMESPACE,
            "receipt_kind": POST_BUILD_OBSERVATION_KIND,
            "status": "pass",
            "segment_id": self.active_segment.segment_id,
            "observed_active_services_state_sha256_and_bytes": self._retain_object(
                services.raw
            ),
            "observed_source_binding_core_sha256_and_bytes": self._retain_object(
                binding.core.raw
            ),
            "segment_attestation_sha256_and_bytes": (
                self.active_segment.attestation_identity.as_dict()
            ),
            "snapshot_set_sha256_and_bytes": dict(self.snapshot_set_identity),
            "observed_at": runtime.canonical_utc(self.clock.now()),
            "runtime_observer_sha256_and_bytes": dict(self.runtime_observer_identity),
            "analysis_plan_sha256_and_bytes": self.contract.plan_identity.as_dict(),
        }
        require_exact_keys(observation, POST_BUILD_OBSERVATION_FIELDS, "observation")
        # ``equality_rule``: both observed objects equal the active segment and
        # source-binding attestation byte for byte.
        if (
            observation["observed_active_services_state_sha256_and_bytes"]["sha256"]
            != self.active_segment.services.identity.sha256
            or observation["observed_source_binding_core_sha256_and_bytes"]["sha256"]
            != self.active_segment.source_binding.core.identity.sha256
        ):
            raise SealTerminalFailure("post-build-observation", "runtime-change")
        raw = canonical_bytes(observation)
        accrual.write_exclusive(self.seal_root / "post-build-observation.json", raw)
        self.observation_identity = identity_of_bytes(raw)
        return raw

    def instruct_publication(self, channels: Mapping[str, int]) -> None:
        if self.publication_attempts >= MAX_PUBLICATION_ATTEMPTS:
            # ``packet_publication_attempts`` is one and ``reseal_allowed``
            # is false.
            raise SealTerminalFailure("publication", "publication-failure")
        self.publication_attempts += 1
        try:
            send_json_frame(channels["inbound"], {"stage": "publish"})
        except (ChildProtocolFailure, OSError):
            raise SealTerminalFailure("publication", "publication-failure") from None

    def await_publication(self, channels: Mapping[str, int]) -> None:
        """Publication is no-overwrite, content first, canonical manifest last."""

        try:
            message = recv_json_frame(
                channels["outbound"], timeout_seconds=float(CHILD_BUILD_TIMEOUT_SECONDS)
            )
        except (ChildProtocolFailure, IntegrityFailure, OSError):
            # ``partial_publication_action`` is terminal-seal-failure.
            raise SealTerminalFailure("publication", "publication-failure") from None
        if (
            type(message) is not dict
            or message.get("stage") != "published"
            or type(message.get("manifest")) is not dict
            or type(message.get("aggregate_counts")) is not dict
        ):
            raise SealTerminalFailure("publication", "publication-failure")

        manifest_path = self.publish_dir / "manifest.json"
        try:
            manifest_raw = runtime.read_regular_file(
                manifest_path, maximum_bytes=4_000_000
            )
        except (IntegrityFailure, OSError):
            raise SealTerminalFailure("publication", "publication-failure") from None
        observed = identity_of_bytes(manifest_raw)
        if not same_identity(observed, message["manifest"]):
            raise SealTerminalFailure("publication", "publication-failure")

        manifest = json.loads(manifest_raw.decode("utf-8"))
        bindings = manifest.get("bindings", {})
        # ``cross_artifact_rule``: the packet manifest carries the same
        # snapshot set identity as the marker, resolution and observation.
        if not same_identity(
            bindings.get("canonical_two_alias_snapshot_set_identity"),
            self.snapshot_set_identity,
        ):
            raise SealTerminalFailure("validation", "verifier-failure")
        if (
            manifest.get("frozen") is not True
            or manifest.get("floors_pass") is not True
            or manifest.get("namespace") != NAMESPACE
            or manifest.get("publication", {}).get("manifest_last") is not True
        ):
            raise SealTerminalFailure("validation", "verifier-failure")
        self.manifest_identity = observed
        if self.aggregate_counts != dict(message["aggregate_counts"]):
            raise SealTerminalFailure("validation", "verifier-failure")

    # -- orchestration ------------------------------------------------------

    def _terminate_child(self, channels: Mapping[str, int] | None) -> None:
        """Close every channel, then destroy the child's own process group."""

        for name in ("gate", "inbound", "outbound"):
            if channels is not None and name in channels:
                with contextlib.suppress(OSError):
                    os.close(channels[name])
        child = self.child
        if child is None:
            return
        if child.poll() is None:
            with contextlib.suppress(OSError, ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=CHILD_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError, ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.wait(timeout=CHILD_TERMINATION_GRACE_SECONDS)

    def run(self) -> tuple[int, dict[str, Any]]:
        """Execute the ceremony and return its exit status and receipt."""

        channels: dict[str, int] | None = None
        try:
            provisional = self.step_one_provisional_core()
            marker, durable = self.steps_two_and_three_consume_authority(provisional)
            spawned, channels = self.step_four_spawn_blocked_child(marker, durable)
            staging_name, _, _ = self.step_five_stage_and_deliver(spawned, channels)
            self.step_six_atomic_visibility(staging_name)
            self.step_seven_open_gate(channels)

            self.await_build(channels)
            self.post_build_runtime_observation()
            self.instruct_publication(channels)
            self.await_publication(channels)

            if self.child is not None:
                try:
                    exit_status = self.child.wait(
                        timeout=CHILD_TERMINATION_GRACE_SECONDS
                    )
                except subprocess.TimeoutExpired:
                    raise SealTerminalFailure(
                        "publication", "publication-failure"
                    ) from None
                if exit_status != 0:
                    raise SealTerminalFailure(
                        "publication", "publication-failure"
                    ) from None
            receipt = self._receipt(status=STATUS_SEALED, stage=None, reason=None)
            self._publish_receipt(receipt)
            return EXIT_SEALED, receipt
        except SealTerminalFailure as failure:
            receipt = self._receipt(
                status=STATUS_TERMINAL,
                stage=failure.stage,
                reason=failure.reason,
            )
            self._publish_receipt(receipt)
            return EXIT_TERMINAL_SEAL_FAILURE, receipt
        finally:
            self._terminate_child(channels)


# --------------------------------------------------------------------------
# Synthetic fixture: a private ledger, a real probe, no real packet
# --------------------------------------------------------------------------

SYNTHETIC_SLOT_INDEX = 1
SYNTHETIC_OBSERVER = b"synthetic-seal-launcher-runtime-observer-v4"


def _synthetic_time(slot_index: int, seconds: float) -> str:
    slot = runtime.slot_times(slot_index)
    return runtime.canonical_utc(
        runtime.parse_utc(slot.scheduled_at, receipt=True)
        + timedelta(seconds=seconds)
    )


def _synthetic_authority(alias: str) -> dict[str, Any]:
    if alias == "local":
        return {
            "transport": "local",
            "kernel_uid_uint32": 1000,
            "process_user_namespace_inode_uint64": 40_001,
        }
    import base64

    return {
        "transport": "ssh",
        "ssh_host_key_algorithm": "ssh-ed25519",
        "ssh_host_key_sha256_base64": (
            base64.b64encode(bytes(range(32))).decode().rstrip("=")
        ),
        "remote_kernel_uid_uint32": 1001,
        "remote_process_user_namespace_inode_uint64": 40_002,
    }


def _synthetic_database(alias: str) -> dict[str, Any]:
    tail = 1 + (alias == "alt")
    return {
        "filesystem_uuid": f"00000000-0000-4000-8000-{tail:012d}",
        "statx_inode_uint64": 70_000 + tail,
        "statx_birthtime_ns_int64": 1_700_000_000_000_000_000 + tail,
    }


def _synthetic_binding_observation(services: Any) -> dict[str, Any]:
    observation: dict[str, Any] = {}
    for index, alias in enumerate(SOURCE_ALIASES):
        service, boot = services.service_pairs[index]
        observation[alias] = {
            "alias_id": runtime.ALIAS_IDS[alias],
            "authenticated_authority_identity": _synthetic_authority(alias),
            "service_identity_sha256": service,
            "boot_identity_sha256": boot,
            "database_instance_identity": _synthetic_database(alias),
        }
    return observation


def persist_head(ledger_directory: Path, state: Any) -> Any:
    """Durably append only the transition this state just made."""

    accrual.write_ledger_entry_exclusive(ledger_directory, state)
    return state


def bootstrap_synthetic_ledger(observer: Any, ledger_directory: Path) -> Any:
    """Replay the frozen bootstrap order into one in-memory ledger state.

    Entry zero is the control-watermark segment, then the slot-0 probe
    attempt, the initial runtime-attestation attempt and the initial
    source-binding attestation.  This is a private synthetic chain: it runs no
    real accrual and reads no source beyond the frozen public documents.
    """

    state = persist_head(ledger_directory, accrual.initialize_ledger())
    contract = state.contract
    slot = runtime.slot_times(0)
    attempt_value = {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "receipt_kind": "probe-attempt",
        "slot_index": 0,
        "scheduled_at": slot.scheduled_at,
        "grace_deadline_at": slot.grace_deadline_at,
        "attempt_ordinal": 0,
        "launched_at": _synthetic_time(0, 1),
        "watchdog_deadline_at": _synthetic_time(0, 3601),
        "segment_id": contract.initial_segment_id,
        "previous_ledger_entry_sha256": state.head,
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "probe_sha256_and_bytes": probe._self_identity().as_dict(),
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
    }
    attempt = probe.validate_probe_attempt_marker(
        canonical_bytes(attempt_value),
        contract=contract,
        active_segment=None,
        expected_segment_id=contract.initial_segment_id,
        expected_runtime_observer_identity=observer,
        expected_previous_ledger_sha256=state.head,
        expected_slot_index=0,
        expected_attempt_ordinal=0,
        expected_probe_identity=probe._self_identity(),
    )
    state = persist_head(ledger_directory, accrual.append_probe_attempt(state, attempt))

    marker_value = {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "receipt_kind": "runtime-attestation-attempt",
        "attempt_scope": "initial-source-binding",
        "segment_index": 0,
        "slot_index_or_null": 0,
        "predecessor_closure_sha256_or_null": None,
        "authorized_at": slot.scheduled_at,
        "written_at": _synthetic_time(0, 1.1),
        "start_deadline_at": slot.grace_deadline_at,
        "previous_ledger_entry_sha256": state.head,
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
    }
    attestation_attempt = runtime.validate_attestation_attempt_marker(
        canonical_bytes(marker_value),
        contract=contract,
        runtime_observer_identity=observer,
        expected_previous_ledger_sha256=state.head,
        initial_probe_launched_at=_synthetic_time(0, 1),
    )
    state = persist_head(
        ledger_directory,
        accrual.append_runtime_attestation_attempt(state, attestation_attempt),
    )

    observation = _synthetic_binding_observation(contract.initial_services)
    binding = runtime.validate_stable_source_binding_observations(
        contract.initial_services,
        observation,
        json.loads(json.dumps(observation)),
    )
    binding_value = {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "receipt_kind": "source-binding-attestation",
        "segment_index": 0,
        "attestation_attempt_marker_sha256_and_bytes": (
            attestation_attempt.identity.as_dict()
        ),
        "alias_ids_by_alias": dict(runtime.ALIAS_IDS),
        "pre_database_instance_identity_sha256_by_alias": binding.core.database_map(),
        "post_database_instance_identity_sha256_by_alias": binding.core.database_map(),
        "pre_alias_service_database_binding_sha256_by_alias": binding.core.binding_map(),
        "post_alias_service_database_binding_sha256_by_alias": binding.core.binding_map(),
        "active_services_state_sha256": contract.initial_services.identity.sha256,
        "source_binding_core_sha256_and_bytes": binding.core.identity.as_dict(),
        "status": "pass",
        "pre_observed_at": _synthetic_time(0, 1.2),
        "post_observed_at": _synthetic_time(0, 1.3),
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
        "previous_ledger_entry_sha256": state.head,
    }
    binding_attestation = runtime.validate_source_binding_attestation(
        canonical_bytes(binding_value),
        contract=contract,
        attempt=attestation_attempt,
        binding=binding,
        runtime_observer_identity=observer,
        expected_previous_ledger_sha256=state.head,
    )
    return persist_head(
        ledger_directory,
        accrual.append_source_binding_attestation(state, binding_attestation),
    )


class SyntheticRuntimeObserver:
    """The trusted launcher boundary the ceremony calls back into.

    It reports the active segment's own retained runtime tuple and source
    binding, which is what a stable synthetic runtime looks like.  Every read
    here is aggregate-only.
    """

    def __init__(self) -> None:
        self.active_segment: Any = None

    def observe(self) -> tuple[Any, Any]:
        segment = self.active_segment
        if segment is None:  # pragma: no cover - programming error only
            _fail()
        return segment.services, segment.source_binding


def floor_clearing_counts(*, margin: int = 2) -> dict[str, int]:
    """Derive an all-floor aggregate from the frozen floors themselves.

    The numbers are computed from ``HOLDOUT_FLOORS`` and ``SHADOW_FLOORS``
    rather than written down, so this fixture cannot drift into an answer key
    for one particular floor table.
    """

    families = probe.HOLDOUT_FLOORS["holdout_unseen_automatic_family_count"] + margin
    automatic = max(
        probe.HOLDOUT_FLOORS["holdout_unseen_automatic_event_count"] + margin,
        3 * families,
    )
    organic = probe.HOLDOUT_FLOORS["holdout_organic_event_count"] + margin
    organic_sessions = probe.HOLDOUT_FLOORS["holdout_organic_session_count"] + margin
    workflows = probe.SHADOW_FLOORS["shadow_real_workflow_count"] + margin
    calls = max(
        probe.SHADOW_FLOORS["shadow_replayable_logical_call_count"] + margin, workflows
    )
    replayable = automatic + organic + calls + margin
    counts = {
        "selected_event_count": replayable + margin,
        "selected_replayable_event_count": replayable,
        "selected_nonreplayable_event_count": margin,
        "holdout_unseen_automatic_family_count": families,
        "holdout_unseen_automatic_event_count": automatic,
        "holdout_unseen_automatic_component_count": families,
        "holdout_organic_event_count": organic,
        "holdout_organic_session_count": organic_sessions,
        "holdout_organic_component_count": organic_sessions,
        "holdout_project_scope_count": (
            probe.HOLDOUT_FLOORS["holdout_project_scope_count"] + margin
        ),
        "shadow_real_workflow_count": workflows,
        "shadow_replayable_logical_call_count": calls,
        "shadow_real_workflow_component_count": workflows,
        "shadow_project_scope_count": (
            probe.SHADOW_FLOORS["shadow_project_scope_count"] + margin
        ),
    }
    probe._aggregate_invariants(counts)
    if not probe._all_floors_pass(counts):  # pragma: no cover
        _fail()
    return counts


def below_floor_counts() -> dict[str, int]:
    """An empty slot: every count zero, so no floor can pass."""

    counts = {field: 0 for field in probe.AGGREGATE_FIELDS}
    probe._aggregate_invariants(counts)
    if probe._all_floors_pass(counts):  # pragma: no cover
        _fail()
    return counts


def synthesize_probe_computation(
    state: Any,
    *,
    snapshot_identities: Mapping[str, Any],
    counts: Mapping[str, int],
) -> Any:
    """Build the probe computation the ceremony consumes.

    ``probe.compute_probe_aggregate`` is deliberately unreachable here.  The
    frozen plan pins ``identity.complete_frozen_dev_reference`` to one exact
    private production export by SHA-256 and byte length, and declares
    ``unavailable_or_hash_mismatch`` a fatal slot integrity failure with "no
    surrogate or alternate reference".  A synthetic fixture therefore cannot
    drive the real probe, and must not try to: this launcher builds the
    computation through the probe's own constructor with the real fixture
    snapshot identities and a synthetic aggregate, and seals no real packet.
    """

    active = state.active_segment
    slot_index = state.next_slot_index
    slot = runtime.slot_times(slot_index)
    aggregate = {field: int(counts[field]) for field in probe.AGGREGATE_FIELDS}
    envelope = {
        "slot_index": slot_index,
        "scheduled_at": slot.scheduled_at,
        "grace_deadline_at": slot.grace_deadline_at,
        "active_segment_lower_bound_exclusive_at": active.lower_bound_exclusive_at,
        "aggregate": aggregate,
        "aliased_source_snapshot_sha256_and_bytes": {
            alias: snapshot_identities[alias].as_dict() for alias in SOURCE_ALIASES
        },
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
    }
    return probe._make_probe_computation(
        slot_index=slot_index,
        scheduled_at=slot.scheduled_at,
        grace_deadline_at=slot.grace_deadline_at,
        active_segment_lower_bound_exclusive_at=active.lower_bound_exclusive_at,
        aggregate_items=tuple((field, aggregate[field]) for field in probe.AGGREGATE_FIELDS),
        snapshot_items=tuple(
            (alias, snapshot_identities[alias]) for alias in SOURCE_ALIASES
        ),
        analysis_plan_identity=state.contract.plan_identity,
        segment_id=active.segment_id,
        segment_attestation_identity=active.attestation_identity,
        source_binding_attestation_identity=(
            active.source_binding_attestation.identity
        ),
        source_database_identity_items=tuple(
            (alias, active.source_binding.core.database_map()[alias])
            for alias in SOURCE_ALIASES
        ),
        worker_envelope_raw=canonical_bytes(envelope),
        probe_identity=probe._self_identity(),
    )


def append_probe_attempt_for_next_slot(state: Any, observer: Any) -> Any:
    slot_index = state.next_slot_index
    slot = runtime.slot_times(slot_index)
    value = {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "receipt_kind": "probe-attempt",
        "slot_index": slot_index,
        "scheduled_at": slot.scheduled_at,
        "grace_deadline_at": slot.grace_deadline_at,
        "attempt_ordinal": 0,
        "launched_at": _synthetic_time(slot_index, 1),
        "watchdog_deadline_at": _synthetic_time(slot_index, 3601),
        "segment_id": state.active_segment.segment_id,
        "previous_ledger_entry_sha256": state.head,
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "probe_sha256_and_bytes": probe._self_identity().as_dict(),
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
    }
    attempt = probe.validate_probe_attempt_marker(
        canonical_bytes(value),
        contract=state.contract,
        active_segment=state.active_segment,
        expected_previous_ledger_sha256=state.head,
        expected_slot_index=slot_index,
        expected_attempt_ordinal=0,
    )
    return accrual.append_probe_attempt(state, attempt)


def close_below_floor_slot(state: Any, computation: Any) -> Any:
    """Record the count-bearing below-floor resolution for the current slot."""

    slot_index = state.next_slot_index
    value = probe.build_below_floor_resolution(
        computation,
        active_segment=state.active_segment,
        launched_at=_synthetic_time(slot_index, 1),
        validated_at=_synthetic_time(slot_index, 5),
        previous_ledger_entry_sha256=state.head,
        contract=state.contract,
    )
    resolution = probe.validate_probe_resolution(
        canonical_bytes(value),
        contract=state.contract,
        active_segment=state.active_segment,
        expected_previous_ledger_sha256=state.head,
        expected_probe_identity=computation.probe_identity,
        expected_snapshot_identities=dict(computation.snapshot_items),
    )
    return accrual.append_below_floor_resolution(state, resolution)


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------


def _expect(condition: bool) -> None:
    if not condition:
        _fail()


def _load_builder_module() -> Any:
    """Import the builder only for its synthetic-fixture helpers.

    The packet build and the publication are driven exclusively through the
    builder's fixed CLI from the sealer child; this import supplies the
    fixture generators the builder already owns so the self-check does not
    grow a second, divergent copy of them.
    """

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ap_confirmatory_v4_packet_builder", BUILDER_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        _fail()
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.runtime is not runtime or module.accrual is not accrual:  # pragma: no cover
        raise SystemExit("v4 runtime module identity is not shared with the builder")
    return module


def _open_retained_snapshots(fixture_dir: Path) -> dict[str, int]:
    """Retain one inheritable read-only descriptor per alias for the child."""

    descriptors: dict[str, int] = {}
    for alias in SOURCE_ALIASES:
        fd = os.open(fixture_dir / f"{alias}.sqlite3", os.O_RDONLY)
        os.set_inheritable(fd, True)
        descriptors[alias] = fd
    return descriptors


def _self_check_collision(
    marker_directory: Path, contract: Any, sealer_identity: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    """Second invocation: consumed authority, never a replacement child."""

    found = find_consumed_marker(marker_directory)
    if found is None:  # pragma: no cover - caller checked
        _fail()
    _, marker_raw = found
    marker = validate_seal_consumption_marker_structure(marker_raw)
    clock = SyntheticSealClock(
        runtime.parse_utc(
            _synthetic_time(marker["slot_index"], 60), receipt=True
        )
    )
    receipt = build_consumed_authority_receipt(
        marker_raw=marker_raw,
        contract=contract,
        sealer_identity=sealer_identity,
        completed_at=clock.canonical_now(),
    )
    return EXIT_TERMINAL_SEAL_FAILURE, {
        "namespace": NAMESPACE,
        "schema_version": SCHEMA_VERSION,
        "status": receipt["status"],
        "failure_stage": receipt["failure_stage_or_null"],
        "failure_reason": receipt["failure_reason_or_null"],
        "seal_terminal_receipt": receipt,
        "handoff_steps_completed": 0,
        "real_packet_sealed": False,
        "synthetic_fixture": True,
        "seal_receipt_written": False,
    }


def run_self_check(work_dir: Path) -> tuple[int, dict[str, Any]]:
    """Drive the whole handoff and publication against a synthetic fixture.

    Everything lives inside ``work_dir``.  No real packet is sealed, no real
    accrual runs, and every read is aggregate-only.
    """

    work = require_output_root_outside_namespace(work_dir)
    seal_root = work / "seal"
    ledger_directory = seal_root / "ledger"
    marker_directory = seal_root / "markers"
    contract = runtime.load_frozen_contract()
    sealer_identity = identity_of_path(BUILDER_PATH)

    if marker_directory.is_dir():
        # Authority for this directory is already consumed.
        return _self_check_collision(marker_directory, contract, sealer_identity)

    builder = _load_builder_module()
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    fixtures = work / "fixtures"
    fixtures.mkdir(mode=0o700)
    make_private_directory(seal_root)
    make_private_directory(ledger_directory)
    make_private_directory(marker_directory)

    for alias in SOURCE_ALIASES:
        builder.write_fixture_snapshot(
            fixtures / f"{alias}.sqlite3",
            alias=alias,
            outcome_marker="base",
            reverse_rows=False,
        )
    builder.write_fixture_dev_reference(fixtures / "dev-reference.jsonl")

    observer = runtime.hash_and_bytes(SYNTHETIC_OBSERVER)
    boundary = SyntheticRuntimeObserver()
    snapshot_identities = {
        alias: runtime.hash_and_bytes(
            runtime.read_regular_file(
                fixtures / f"{alias}.sqlite3", maximum_bytes=64_000_000
            )
        )
        for alias in SOURCE_ALIASES
    }
    _expect(
        snapshot_identities["local"].sha256 != snapshot_identities["alt"].sha256
    )

    # The frozen bootstrap order, then slot 0 below-floor, then slot 1 ready.
    state = bootstrap_synthetic_ledger(observer, ledger_directory)
    boundary.active_segment = state.active_segment
    _expect(state.next_slot_index == 0)
    slot_zero = synthesize_probe_computation(
        state, snapshot_identities=snapshot_identities, counts=below_floor_counts()
    )
    _expect(not slot_zero.provisional_ready)
    state = persist_head(ledger_directory, close_below_floor_slot(state, slot_zero))
    _expect(state.next_slot_index == SYNTHETIC_SLOT_INDEX)

    state = persist_head(
        ledger_directory, append_probe_attempt_for_next_slot(state, observer)
    )
    computation = synthesize_probe_computation(
        state, snapshot_identities=snapshot_identities, counts=floor_clearing_counts()
    )
    _expect(computation.provisional_ready)

    packet = work / "packet"
    packet.mkdir(mode=0o755)
    verifier_sha256 = builder.write_fixture_publish_root(packet)
    verifier_identity = identity_of_path(packet / "recipe" / "verify.py")
    _expect(verifier_identity["sha256"] == verifier_sha256)

    handoff = builder.build_fixture_handoff(
        fixture_dir=fixtures,
        seal_launcher_path=Path(__file__).resolve(),
        slot_index=SYNTHETIC_SLOT_INDEX,
    )
    handoff_path = fixtures / "handoff.json"
    handoff_path.write_bytes(canonical_bytes(handoff))

    snapshot_fds = _open_retained_snapshots(fixtures)
    try:
        ceremony = SealCeremony(
            clock=SyntheticSealClock(
                runtime.parse_utc(
                    _synthetic_time(SYNTHETIC_SLOT_INDEX, 2), receipt=True
                )
            ),
            state=state,
            computation=computation,
            observe_active_state=boundary.observe,
            runtime_observer_identity=observer.as_dict(),
            seal_root=seal_root,
            draft_dir=work / "draft",
            publish_dir=packet,
            handoff_path=handoff_path,
            snapshot_paths={
                alias: fixtures / f"{alias}.sqlite3" for alias in SOURCE_ALIASES
            },
            snapshot_fds=snapshot_fds,
            sealer_identity=sealer_identity,
            verifier_identity=verifier_identity,
        )
        boundary.active_segment = state.active_segment
        status, receipt = ceremony.run()
    finally:
        for fd in snapshot_fds.values():
            with contextlib.suppress(OSError):
                os.close(fd)

    if status == EXIT_SEALED:
        _assert_sealed_state(ceremony, packet, seal_root, receipt)
    return status, {
        "namespace": NAMESPACE,
        "schema_version": SCHEMA_VERSION,
        "status": receipt["status"],
        "failure_stage": receipt["failure_stage_or_null"],
        "failure_reason": receipt["failure_reason_or_null"],
        "seal_terminal_receipt": receipt,
        "handoff_steps_completed": len(ceremony.steps),
        "aggregate_counts": ceremony.aggregate_counts,
        "semantic_reads": {
            "confirmatory-shadow-v4-eval": 0,
            "confirmatory-holdout-v4-eval": 0,
        },
        "real_packet_sealed": False,
        "synthetic_fixture": True,
        "seal_receipt_written": (packet / SEAL_RECEIPT_FILENAME).is_file(),
    }


def _assert_sealed_state(
    ceremony: SealCeremony,
    packet: Path,
    seal_root: Path,
    receipt: Mapping[str, Any],
) -> None:
    """Falsify the sealed claim rather than asserting it."""

    _expect(list(ceremony.steps) == list(ORDERING_EXACTLY))
    _expect(ceremony.bundle_visible is True)
    _expect(receipt["status"] == STATUS_SEALED)
    _expect(receipt["previous_ledger_entry_sha256"] == ceremony.head_r)

    bundle = seal_root / READY_BUNDLE_DIRECTORY
    _expect(bundle.is_dir())
    _expect((bundle / READY_RESOLUTION_NAME).is_file())
    _expect((bundle / READY_ENTRY_NAME).is_file())
    # No staging directory may survive a successful atomic exposure.
    _expect(not [item for item in seal_root.iterdir() if item.name.startswith(STAGING_PREFIX)])

    # Content first, canonical manifest last, and the receipt is separate.
    _expect((packet / "manifest.json").is_file())
    _expect((packet / SEAL_RECEIPT_FILENAME).is_file())
    manifest = json.loads((packet / "manifest.json").read_text(encoding="utf-8"))
    # Consumption is recorded in separate hash-bound aggregate receipts only:
    # no marker, terminal receipt or ready-resolution identity may appear in
    # the immutable manifest, and the manifest was written before the receipt.
    serialized = json.dumps(manifest, sort_keys=True)
    for digest in (
        identity_of_path(packet / SEAL_RECEIPT_FILENAME)["sha256"],
        receipt["seal_consumption_marker_sha256_and_bytes_or_null"]["sha256"],
        receipt["slot_resolution_sha256_and_bytes_or_null"]["sha256"],
        receipt["post_build_runtime_observation_sha256_and_bytes_or_null"]["sha256"],
    ):
        _expect(digest not in serialized)
    _expect("seal_consumption" not in serialized)
    _expect(
        (packet / "manifest.json").stat().st_mtime_ns
        <= (packet / SEAL_RECEIPT_FILENAME).stat().st_mtime_ns
    )

    # ``cross_artifact_rule``: one snapshot set across all five artifacts.
    snapshot_set = ceremony.snapshot_set_identity
    marker_found = find_consumed_marker(seal_root / "markers")
    _expect(marker_found is not None)
    marker = validate_seal_consumption_marker_structure(marker_found[1])
    resolution = load_canonical_receipt(
        (bundle / READY_RESOLUTION_NAME).read_bytes(), probe.PROBE_RESOLUTION_FIELDS
    )
    observation = load_canonical_receipt(
        (seal_root / "post-build-observation.json").read_bytes(),
        POST_BUILD_OBSERVATION_FIELDS,
    )
    _expect(same_identity(marker["snapshot_set_sha256_and_bytes"], snapshot_set))
    _expect(
        same_identity(
            accrual.derive_snapshot_set_identity(
                resolution["aliased_source_snapshot_sha256_and_bytes"]
            ).as_dict(),
            snapshot_set,
        )
    )
    _expect(same_identity(observation["snapshot_set_sha256_and_bytes"], snapshot_set))
    _expect(same_identity(receipt["snapshot_set_sha256_and_bytes"], snapshot_set))
    _expect(
        same_identity(
            manifest["bindings"]["canonical_two_alias_snapshot_set_identity"],
            snapshot_set,
        )
    )

    # The ledger ends at exactly one ``seal-terminal`` entry from R.
    entries = sorted((seal_root / "ledger").iterdir())
    terminal = load_canonical_receipt(
        entries[-1].read_bytes(), accrual.LEDGER_ENTRY_FIELDS
    )
    _expect(terminal["entry_kind"] == SEAL_TERMINAL_ENTRY_KIND)
    _expect(terminal["previous_ledger_entry_sha256"] == ceremony.head_r)
    _expect(
        same_identity(
            terminal["artifact_sha256_and_bytes"],
            identity_of_path(packet / SEAL_RECEIPT_FILENAME),
        )
    )
    _expect(accrual.derive_ledger_entry_sha256(terminal) == terminal["entry_sha256"])
    # ``reseal_allowed`` is false: the marker path cannot be won twice.
    _expect(find_consumed_marker(seal_root / "markers") is not None)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="confirmatory-holdout-v4 one-shot no-overwrite seal launcher"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "self-check",
        help="drive the full handoff and publication against a synthetic fixture",
    )
    check.add_argument("--work-dir", type=Path, required=True)

    child = subparsers.add_parser(
        "__child", help=argparse.SUPPRESS
    )
    child.add_argument("--gate-fd", type=int, required=True)
    child.add_argument("--inbound-fd", type=int, required=True)
    child.add_argument("--outbound-fd", type=int, required=True)
    child.add_argument("--snapshot-fd", nargs="+", required=True)

    return parser.parse_args(argv)


def _child_snapshot_fds(pairs: Sequence[str]) -> dict[str, int]:
    descriptors: dict[str, int] = {}
    for pair in pairs:
        alias, _, value = pair.partition("=")
        if alias not in SOURCE_ALIASES or not value.isdigit():
            raise ChildProtocolFailure("snapshot_fd_invalid")
        descriptors[alias] = int(value)
    if set(descriptors) != set(SOURCE_ALIASES):
        raise ChildProtocolFailure("snapshot_fd_invalid")
    return descriptors


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = parse_arguments(argv)
    except SystemExit as request:
        # argparse exits 0 for --help and 2 for a real usage error.
        return EXIT_SEALED if request.code in (0, None) else EXIT_USAGE

    if arguments.command == "__child":
        try:
            return run_sealer_child(
                gate_fd=arguments.gate_fd,
                inbound_fd=arguments.inbound_fd,
                outbound_fd=arguments.outbound_fd,
                snapshot_fds=_child_snapshot_fds(arguments.snapshot_fd),
            )
        except (ChildProtocolFailure, IntegrityFailure, OSError, subprocess.SubprocessError):
            # The child never publishes a diagnosis; the supervisor owns the
            # terminal receipt and observes this as EOF or a nonzero exit.
            return 1

    try:
        status, summary = run_self_check(arguments.work_dir)
    except SealTerminalFailure as failure:
        # A terminal failure raised outside a ceremony has no provisional core
        # to bind a receipt to -- for instance two markers contending on one
        # path.  Report the stage and reason without fabricating artifacts.
        print(
            json.dumps(
                {
                    "namespace": NAMESPACE,
                    "schema_version": SCHEMA_VERSION,
                    "status": STATUS_TERMINAL,
                    "failure_stage": failure.stage,
                    "failure_reason": failure.reason,
                    "seal_terminal_receipt": None,
                    "real_packet_sealed": False,
                },
                sort_keys=True,
            )
        )
        return EXIT_TERMINAL_SEAL_FAILURE
    except IntegrityFailure:
        # No receipt is constructible -- for example marker bytes that exist
        # but do not resolve, so no ledger head H can be recovered from them.
        print(
            json.dumps(
                {
                    "namespace": NAMESPACE,
                    "schema_version": SCHEMA_VERSION,
                    "integrity_escalation": True,
                    "seal_terminal_receipt": None,
                    "real_packet_sealed": False,
                },
                sort_keys=True,
            )
        )
        return EXIT_TERMINAL_SEAL_FAILURE
    except NamespaceRefusal as refusal:
        print(
            json.dumps(
                {
                    "code": NAMESPACE_REFUSAL_CODE,
                    "path": os.fspath(refusal.path),
                    "real_packet_sealed": False,
                },
                sort_keys=True,
            )
        )
        return EXIT_NAMESPACE_REFUSED
    print(json.dumps(summary, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
