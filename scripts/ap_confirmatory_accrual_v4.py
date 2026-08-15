#!/usr/bin/env python3
"""Canonical append-only accrual ledger for ``confirmatory-holdout-v4``.

This module is the source-blind controller between the shared runtime validator
and the aggregate-only probe.  It never opens a production source.  Inputs are
already validator-valid runtime/probe wrappers, but every wrapper is rebound to
its exact bytes, the current active segment, and the current ledger head before
it can advance the state machine.

The state transition functions are pure.  Irreversible persistence is a
separate O_EXCL primitive so callers can validate a complete transition before
creating its deterministic entry file.  A collision is always an integrity
failure; this module never overwrites, unlinks, skips, or creates an alternate
branch after a failed write.

``initial-source-binding-required`` is the one edge repaired for v4.  The
retired lineage could only leave it through a matching source binding, so a
changed observed tuple at slot 0 had no honest resolution.  Here the same phase
also accepts the distinct ``InitialRuntimeChangeClosure``: it consumes slot 0,
appends no counts and no snapshots, drops every latch and all source authority,
retains the typed :class:`~ap_confirmatory_runtime_v4.UnboundWatermarkPredecessor`
instead of a fabricated ``ActiveSegment``, sets the next probe slot to 1, and
authorizes exactly one timely fresh successor ceremony.  Nothing observed before
that break -- counts, snapshots, latches, probe attempts, or attestations --
survives it.

The seal launcher itself is intentionally out of scope.  ``record_blocked_spawn``
is the narrow trusted-supervisor boundary: it can issue the typed capability
required by ``ready`` only after a validator-valid seal marker is already the
ledger head.  A receipt-supplied timestamp, marker digest, or boolean is not a
substitute for that capability.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import stat
import sys
import threading
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence


# Import through the probe's canonical runtime module.  Both upstream modules
# deliberately use exact dataclass type identity as a wrapper-integrity check.
try:
    import ap_confirmatory_probe_v4 as probe
except ModuleNotFoundError:  # pragma: no cover - importlib callers
    import importlib.util

    _PROBE_PATH = Path(__file__).resolve().with_name(
        "ap_confirmatory_probe_v4.py"
    )
    _PROBE_SPEC = importlib.util.spec_from_file_location(
        "ap_confirmatory_probe_v4", _PROBE_PATH
    )
    if _PROBE_SPEC is None or _PROBE_SPEC.loader is None:
        raise
    probe = importlib.util.module_from_spec(_PROBE_SPEC)
    sys.modules[_PROBE_SPEC.name] = probe
    _PROBE_SPEC.loader.exec_module(probe)


runtime = probe.runtime
IntegrityFailure = runtime.IntegrityFailure

NAMESPACE = runtime.NAMESPACE
SCHEMA_VERSION = runtime.SCHEMA_VERSION
LEDGER_DOMAIN = b"confirmatory-holdout-v4/accrual-ledger/v1"
READY_CORE_DOMAIN = b"confirmatory-holdout-v4/ready-core/v1"
SNAPSHOT_SET_DOMAIN = b"confirmatory-holdout-v4/snapshot-set/v1"
SEAL_AUTHORITY_ID = "confirmatory-holdout-v4-seal"
HANDOFF_SECONDS = 30

LEDGER_ENTRY_FIELDS = (
    "schema_version",
    "namespace",
    "entry_index",
    "entry_kind",
    "slot_index_or_null",
    "artifact_sha256_and_bytes",
    "previous_ledger_entry_sha256",
    "entry_sha256",
)

FROZEN_ENTRY_KINDS = frozenset(
    {
        "segment-attestation",
        "runtime-attestation-attempt",
        "runtime-attestation-terminal",
        "source-binding-attestation",
        "probe-attempt",
        "probe-failure",
        "probe-terminal",
        "below-floor",
        "ready",
        "initial-runtime-change-closure",
        "slot-segment-closed",
        "missed-slot",
        "seal-consumption",
        "seal-terminal",
        "horizon-terminal",
    }
)

# The packet sealer owns ``seal-terminal`` validation and publication.  This
# source-blind accrual leaf deliberately stops at a visible ready resolution;
# keeping the frozen enum separate avoids falsely advertising sealer support.
ENTRY_KINDS = FROZEN_ENTRY_KINDS - {"seal-terminal"}

# The two closure entry kinds are distinct on purpose: the pre-binding closure
# is never spelled with the ordinary strict two-core kind, and the ordinary
# closure is never spelled with the pre-binding one.
INITIAL_CLOSURE_ENTRY_KIND = "initial-runtime-change-closure"
CLOSURE_ENTRY_KIND = "slot-segment-closed"
NEXT_SLOT_AFTER_INITIAL_CLOSURE = runtime.FIRST_SLOT_INDEX + 1

RUNTIME_TERMINAL_FIELDS = (
    "schema_version",
    "namespace",
    "receipt_kind",
    "status",
    "attempt_scope",
    "segment_index",
    "attestation_attempt_marker_sha256_and_bytes_or_null",
    "predecessor_closure_sha256_or_null",
    "recorded_at",
    "reason",
    "previous_ledger_entry_sha256",
    "analysis_plan_sha256_and_bytes",
)

RUNTIME_TERMINAL_REASONS = frozenset(
    {
        "attempt-marker-create-failure",
        "attempt-marker-collision",
        "attempt-start-late",
        "observation-unavailable",
        "unstable-runtime",
        "unstable-source-binding",
        "boundary-invalid",
        "next-slot-deadline",
        "worker-crash-or-eof",
        "watchdog-timeout",
        "validator-failure",
    }
)

RUNTIME_TERMINAL_NULL_MARKER_REASONS = frozenset(
    {
        "attempt-marker-create-failure",
        "attempt-marker-collision",
        "attempt-start-late",
        "worker-crash-or-eof",
        "watchdog-timeout",
    }
)

RUNTIME_TERMINAL_PRE_MARKER_ONLY_REASONS = frozenset(
    {
        "attempt-marker-create-failure",
        "attempt-marker-collision",
        "attempt-start-late",
    }
)

MISSED_SLOT_FIELDS = (
    "schema_version",
    "namespace",
    "receipt_kind",
    "status",
    "slot_index",
    "scheduled_at",
    "grace_deadline_at",
    "recorded_at",
    "source_open_count",
    "previous_ledger_entry_sha256",
    "analysis_plan_sha256_and_bytes",
)

HORIZON_TERMINAL_FIELDS = (
    "schema_version",
    "namespace",
    "receipt_kind",
    "status",
    "final_slot_index",
    "final_slot_resolution_sha256_and_bytes",
    "absolute_horizon_expires_at",
    "recorded_at",
    "reason",
    "previous_ledger_entry_sha256",
    "analysis_plan_sha256_and_bytes",
)

SEAL_CONSUMPTION_FIELDS = (
    "schema_version",
    "namespace",
    "receipt_kind",
    "seal_authority_id",
    "slot_index",
    "segment_id",
    "provisional_ready_core_sha256_and_bytes",
    "snapshot_set_sha256_and_bytes",
    "consumed_at",
    "previous_ledger_entry_sha256",
    "sealer_sha256_and_bytes",
    "analysis_plan_sha256_and_bytes",
)

PROBE_TERMINAL_FIELDS = probe.PROBE_TERMINAL_FIELDS


PHASE_BOOTSTRAP_PROBE = "bootstrap-probe-attempt-required"
PHASE_INITIAL_RUNTIME_ATTEMPT = "initial-runtime-attempt-required"
PHASE_INITIAL_SOURCE_BINDING = "initial-source-binding-required"
PHASE_AWAITING_PROBE = "awaiting-next-fixed-slot-probe"
PHASE_PROBE_ACTIVE = "probe-attempt-active"
PHASE_RETRY_ALLOWED = "same-slot-retry-authorized"
PHASE_PROBE_TERMINAL = "probe-terminal-required"
PHASE_SUCCESSOR_ATTEMPT = "successor-runtime-attempt-required"
PHASE_SUCCESSOR_BINDING = "successor-source-binding-required"
PHASE_SUCCESSOR_SEGMENT = "successor-segment-attestation-required"
PHASE_SEAL_CONSUMED = "seal-consumed-blocked-spawn-required"
PHASE_BLOCKED_SPAWN = "blocked-spawn-ready-required"
PHASE_READY_VISIBLE = "ready-visible-sealer-running"
PHASE_HORIZON = "awaiting-absolute-horizon-no-source-authority"
PHASE_TERMINAL = "terminal"

_PHASES = frozenset(
    {
        PHASE_BOOTSTRAP_PROBE,
        PHASE_INITIAL_RUNTIME_ATTEMPT,
        PHASE_INITIAL_SOURCE_BINDING,
        PHASE_AWAITING_PROBE,
        PHASE_PROBE_ACTIVE,
        PHASE_RETRY_ALLOWED,
        PHASE_PROBE_TERMINAL,
        PHASE_SUCCESSOR_ATTEMPT,
        PHASE_SUCCESSOR_BINDING,
        PHASE_SUCCESSOR_SEGMENT,
        PHASE_SEAL_CONSUMED,
        PHASE_BLOCKED_SPAWN,
        PHASE_READY_VISIBLE,
        PHASE_HORIZON,
        PHASE_TERMINAL,
    }
)


def _fail() -> None:
    raise IntegrityFailure from None


def _is_int(value: Any) -> bool:
    return type(value) is int


def _nonnegative_int(value: Any) -> int:
    if not _is_int(value) or value < 0:
        _fail()
    return value


def _slot_index(value: Any) -> int:
    index = _nonnegative_int(value)
    if index > runtime.LAST_SLOT_INDEX:
        _fail()
    return index


def _hex64(value: Any) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in runtime.HEX_LOWER for character in value)
    ):
        _fail()
    return value


def _exact_mapping(value: Any, fields: Sequence[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        _fail()
    return value


def _identity(value: Any) -> runtime.HashAndBytes:
    return (
        value
        if type(value) is runtime.HashAndBytes
        else runtime.HashAndBytes.from_value(value)
    )


def _receipt(raw: bytes, fields: Sequence[str]) -> dict[str, Any]:
    value = runtime.load_canonical_json_bytes(raw, expected=dict)
    _exact_mapping(value, fields)
    if (
        value["schema_version"] != SCHEMA_VERSION
        or type(value["schema_version"]) is not int
        or value["namespace"] != NAMESPACE
    ):
        _fail()
    return value


def _same_identity(value: Any, expected: runtime.HashAndBytes) -> None:
    if _identity(value) != expected:
        _fail()


def _same_active_segment(
    contract: runtime.FrozenContract,
    left: runtime.ActiveSegment,
    right: runtime.ActiveSegment,
) -> None:
    runtime._require_active_segment_consistent(contract, left)
    runtime._require_active_segment_consistent(contract, right)
    if (
        left.segment_index != right.segment_index
        or left.segment_id != right.segment_id
        or left.lower_bound_exclusive_at != right.lower_bound_exclusive_at
        or left.services.identity != right.services.identity
        or left.source_binding.core.identity != right.source_binding.core.identity
        or left.attestation_identity != right.attestation_identity
        or left.source_binding_attestation.identity
        != right.source_binding_attestation.identity
    ):
        _fail()


def derive_ledger_genesis(
    contract: runtime.FrozenContract | None = None,
) -> str:
    """Return the frozen domain-separated predecessor for entry zero."""

    frozen = runtime.load_frozen_contract() if contract is None else contract
    runtime._require_frozen_contract_consistent(frozen)
    return hashlib.sha256(
        LEDGER_DOMAIN + b"\0" + frozen.plan_identity.sha256.encode("ascii")
    ).hexdigest()


def derive_ledger_entry_sha256(value: Any) -> str:
    """Derive the ledger head from an exact entry-shaped mapping."""

    entry = _exact_mapping(value, LEDGER_ENTRY_FIELDS)
    previous = _hex64(entry["previous_ledger_entry_sha256"])
    core = dict(entry)
    del core["entry_sha256"]
    return hashlib.sha256(
        LEDGER_DOMAIN
        + b"\0"
        + bytes.fromhex(previous)
        + b"\0"
        + runtime.canonical_json_bytes(core)
    ).hexdigest()


def derive_snapshot_set_identity(value: Any) -> runtime.HashAndBytes:
    mapping = _exact_mapping(value, runtime.SOURCE_ALIASES)
    snapshots = {
        alias: _identity(mapping[alias]) for alias in runtime.SOURCE_ALIASES
    }
    if (
        any(item.bytes <= 0 for item in snapshots.values())
        or len({item.sha256 for item in snapshots.values()}) != 2
    ):
        _fail()
    normalized = {
        alias: snapshots[alias].as_dict() for alias in runtime.SOURCE_ALIASES
    }
    preimage = (
        SNAPSHOT_SET_DOMAIN
        + b"\0"
        + runtime.canonical_json_bytes(normalized)
    )
    return runtime.HashAndBytes(hashlib.sha256(preimage).hexdigest(), len(preimage))


def _derive_ready_core_identity(core: Any) -> runtime.HashAndBytes:
    if type(core) is not dict:
        _fail()
    expected = tuple(
        field
        for field in probe.PROBE_RESOLUTION_FIELDS
        if field
        not in {
            "resolution_id",
            "previous_ledger_entry_sha256",
            "seal_consumption_marker_sha256_and_bytes_or_null",
            "sealer_process_launched_at_or_null",
        }
    )
    _exact_mapping(core, expected)
    preimage = READY_CORE_DOMAIN + b"\0" + runtime.canonical_json_bytes(core)
    return runtime.HashAndBytes(hashlib.sha256(preimage).hexdigest(), len(preimage))


@dataclass(frozen=True, slots=True)
class ValidatedLedgerEntry:
    raw: bytes
    identity: runtime.HashAndBytes
    artifact_raw: bytes
    artifact_identity: runtime.HashAndBytes
    entry_index: int
    entry_kind: str
    slot_index: int | None
    previous_ledger_entry_sha256: str
    entry_sha256: str
    _semantic_auth: bytes = field(default=b"", compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class ProvisionalReady:
    core: dict[str, Any]
    core_identity: runtime.HashAndBytes
    snapshot_set_identity: runtime.HashAndBytes
    snapshot_items: tuple[tuple[str, runtime.HashAndBytes], ...]
    slot_index: int
    segment_id: str
    active_segment: runtime.ActiveSegment
    computation: probe.ProbeComputation
    probe_identity: runtime.HashAndBytes
    validated_at: datetime
    _origin: object


@dataclass(frozen=True, slots=True)
class ValidatedSealConsumption:
    raw: bytes
    identity: runtime.HashAndBytes
    receipt: dict[str, Any]
    slot_index: int
    segment_id: str
    consumed_at: datetime
    provisional: ProvisionalReady
    sealer_identity: runtime.HashAndBytes


_DURABLE_SEAL_ORIGIN = object()
_CAPABILITY_KEY = os.urandom(32)
_DURABLE_NONCE_LOCK = threading.Lock()
_ISSUED_DURABLE_NONCES: set[bytes] = set()
_CONSUMED_DURABLE_NONCES: set[bytes] = set()


def _issue_durable_nonce(nonce: bytes) -> None:
    with _DURABLE_NONCE_LOCK:
        if (
            type(nonce) is not bytes
            or len(nonce) != 32
            or nonce in _ISSUED_DURABLE_NONCES
            or nonce in _CONSUMED_DURABLE_NONCES
        ):
            _fail()
        _ISSUED_DURABLE_NONCES.add(nonce)


def _consume_durable_nonce(nonce: bytes) -> None:
    with _DURABLE_NONCE_LOCK:
        if nonce not in _ISSUED_DURABLE_NONCES:
            _fail()
        _ISSUED_DURABLE_NONCES.remove(nonce)
        _CONSUMED_DURABLE_NONCES.add(nonce)


@dataclass(frozen=True, slots=True, init=False)
class DurableSealConsumption:
    """Capability proving the exact marker and ledger M are durable."""

    marker_identity: runtime.HashAndBytes
    ledger_head: str
    entry_index: int
    marker_path: Path
    ledger_entry_path: Path
    _nonce: bytes
    _auth: bytes
    _origin: object

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        _fail()


_CLOSURE_BINDING_ORIGIN = object()


@dataclass(frozen=True, slots=True, init=False)
class BoundSegmentClosure:
    """A closure rebound through the full retained-object validator."""

    closure: runtime.ValidatedClosure
    active_segment: runtime.ActiveSegment
    ledger_head: str
    prior_services_raw: bytes
    observed_services_raw: bytes
    prior_source_binding_core_raw: bytes
    observed_source_binding_core_raw: bytes
    observed_binding: runtime.SourceBinding
    runtime_observer_identity: runtime.HashAndBytes
    _origin: object

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        _fail()


_INITIAL_CLOSURE_BINDING_ORIGIN = object()

# Whatever else a pre-binding closure receipt may carry, it may never carry a
# count or a snapshot set; the exact field list already forbids them, and this
# set makes the prohibition independently checkable.
_COUNT_AND_SNAPSHOT_FIELDS = frozenset(probe.AGGREGATE_FIELDS) | {
    "aliased_source_snapshot_sha256_and_bytes"
}


@dataclass(frozen=True, slots=True, init=False)
class BoundInitialRuntimeChangeClosure:
    """A pre-binding closure rebound through the full retained-object validator.

    It carries no prior source-binding core and no ``ActiveSegment`` because
    neither has ever existed at this point in the lineage.  What it retains is
    the observed evidence, the consumed slot-0 attempt, and the typed
    pre-binding predecessor.
    """

    closure: runtime.InitialRuntimeChangeClosure
    predecessor: runtime.UnboundWatermarkPredecessor
    ledger_head: str
    attempt: runtime.AttestationAttempt
    observed_services: runtime.ServiceTuple
    observed_services_raw: bytes
    observed_binding: runtime.SourceBinding
    observed_source_binding_core_raw: bytes
    runtime_observer_identity: runtime.HashAndBytes
    _origin: object

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        _fail()


_BLOCKED_SPAWN_ORIGIN = object()


@dataclass(frozen=True, slots=True, init=False)
class BlockedSpawnProof:
    """Capability issued only by the trusted supervisor transition."""

    marker_identity: runtime.HashAndBytes
    sealer_identity: runtime.HashAndBytes
    slot_index: int
    segment_id: str
    launched_at: datetime
    ledger_head: str
    _nonce: bytes
    _auth: bytes
    _origin: object

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        _fail()


@dataclass(frozen=True, slots=True)
class LedgerState:
    """Ephemeral replay state; rebuild it through transitions each invocation.

    Only canonical entry/artifact bytes are durable.  ``_history_auth`` is a
    process-local anti-substitution capability and is never serialized.
    """

    contract: runtime.FrozenContract
    entries: tuple[ValidatedLedgerEntry, ...]
    phase: str
    next_slot_index: int | None
    active_segment: runtime.ActiveSegment | None = None
    current_probe_attempt: probe.ValidatedProbeAttempt | None = None
    runtime_attempt: runtime.AttestationAttempt | None = None
    source_binding_attestation: runtime.SourceBindingAttestation | None = None
    closed_segment: runtime.ActiveSegment | None = None
    # The pre-binding predecessor lives in its own typed slot.  It is never
    # written into ``closed_segment`` and no code path builds an
    # ``ActiveSegment`` from it; a successor grows from it only through
    # ``runtime.validate_successor_segment``.
    unbound_predecessor: runtime.UnboundWatermarkPredecessor | None = None
    closure: runtime.ValidatedClosure | runtime.InitialRuntimeChangeClosure | None = None
    probe_failure: probe.ValidatedProbeFailure | None = None
    provisional_ready: ProvisionalReady | None = None
    seal_consumption: ValidatedSealConsumption | None = None
    blocked_spawn: BlockedSpawnProof | None = None
    final_resolution_identity: runtime.HashAndBytes | None = None
    final_resolution_status: str | None = None
    terminal_status: str | None = None
    _history_auth: bytes = b""

    @property
    def head(self) -> str:
        if not self.entries:
            _fail()
        return self.entries[-1].entry_sha256

    @property
    def next_entry_index(self) -> int:
        return len(self.entries)

    @property
    def source_authority(self) -> bool:
        # Production rows may be opened only behind a current durable probe
        # attempt.  Runtime/source-binding ceremonies are metadata-only.
        return self.phase == PHASE_PROBE_ACTIVE

    @property
    def closed_predecessor(
        self,
    ) -> runtime.ActiveSegment | runtime.UnboundWatermarkPredecessor | None:
        """Return the sole typed predecessor a successor may be grown from."""

        if self.closed_segment is not None and self.unbound_predecessor is not None:
            _fail()
        return (
            self.closed_segment
            if self.closed_segment is not None
            else self.unbound_predecessor
        )


_STATE_HISTORY_KEY = os.urandom(32)


def _capability_auth(domain: bytes, value: Mapping[str, Any]) -> bytes:
    return hmac.digest(
        _CAPABILITY_KEY,
        domain + b"\0" + runtime.canonical_json_bytes(dict(value)),
        hashlib.sha256,
    )


def _ledger_entry_semantic_auth(value: ValidatedLedgerEntry) -> bytes:
    return hmac.digest(
        _CAPABILITY_KEY,
        b"confirmatory-holdout-v4/validated-ledger-transition/v1\0"
        + len(value.raw).to_bytes(8, "big")
        + value.raw
        + len(value.artifact_raw).to_bytes(8, "big")
        + value.artifact_raw,
        hashlib.sha256,
    )


def _authorize_ledger_entry(
    value: ValidatedLedgerEntry,
) -> ValidatedLedgerEntry:
    return replace(value, _semantic_auth=_ledger_entry_semantic_auth(value))


def _durable_capability_auth(value: DurableSealConsumption) -> bytes:
    return _capability_auth(
        b"confirmatory-holdout-v4/durable-seal-capability/v1",
        {
            "marker": value.marker_identity.as_dict(),
            "ledger_head": value.ledger_head,
            "entry_index": value.entry_index,
            "marker_path": str(value.marker_path),
            "ledger_entry_path": str(value.ledger_entry_path),
            "nonce": value._nonce.hex(),
        },
    )


def _blocked_spawn_capability_auth(value: BlockedSpawnProof) -> bytes:
    return _capability_auth(
        b"confirmatory-holdout-v4/blocked-spawn-capability/v1",
        {
            "marker": value.marker_identity.as_dict(),
            "sealer": value.sealer_identity.as_dict(),
            "slot_index": value.slot_index,
            "segment_id": value.segment_id,
            "launched_at": runtime.canonical_utc(value.launched_at),
            "ledger_head": value.ledger_head,
            "nonce": value._nonce.hex(),
        },
    )


def _history_auth(
    entries: tuple[ValidatedLedgerEntry, ...],
) -> bytes:
    digest = hmac.new(_STATE_HISTORY_KEY, digestmod=hashlib.sha256)
    for entry in entries:
        if type(entry) is not ValidatedLedgerEntry:
            _fail()
        for raw in (entry.raw, entry.artifact_raw):
            if type(raw) is not bytes:
                _fail()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    return digest.digest()


@dataclass(frozen=True, slots=True)
class _StateProjection:
    phase: str
    next_slot_index: int | None
    terminal_status: str | None
    current_probe_attempt_identity: runtime.HashAndBytes | None
    runtime_attempt_identity: runtime.HashAndBytes | None
    source_binding_identity: runtime.HashAndBytes | None
    active_attestation_identity: runtime.HashAndBytes | None
    active_source_binding_identity: runtime.HashAndBytes | None
    closed_attestation_identity: runtime.HashAndBytes | None
    closed_source_binding_identity: runtime.HashAndBytes | None
    closure_identity: runtime.HashAndBytes | None
    pre_binding_closure: bool
    pre_binding_closure_attempt_identity: runtime.HashAndBytes | None
    probe_failure_identity: runtime.HashAndBytes | None
    seal_consumption_identity: runtime.HashAndBytes | None
    seal_probe_attempt_identity: runtime.HashAndBytes | None
    seal_active_attestation_identity: runtime.HashAndBytes | None
    seal_active_source_binding_identity: runtime.HashAndBytes | None
    final_resolution_identity: runtime.HashAndBytes | None
    final_resolution_status: str | None


_ARTIFACT_FIELDS: dict[str, Sequence[str]] = {
    "runtime-attestation-attempt": runtime.ATTEMPT_MARKER_FIELDS,
    "runtime-attestation-terminal": RUNTIME_TERMINAL_FIELDS,
    "source-binding-attestation": runtime.SOURCE_BINDING_ATTESTATION_FIELDS,
    "probe-attempt": probe.PROBE_ATTEMPT_FIELDS,
    "probe-failure": probe.PROBE_FAILURE_FIELDS,
    "probe-terminal": PROBE_TERMINAL_FIELDS,
    "below-floor": probe.PROBE_RESOLUTION_FIELDS,
    "ready": probe.PROBE_RESOLUTION_FIELDS,
    INITIAL_CLOSURE_ENTRY_KIND: runtime.INITIAL_CLOSURE_FIELDS,
    "slot-segment-closed": runtime.CLOSURE_FIELDS,
    "missed-slot": MISSED_SLOT_FIELDS,
    "seal-consumption": SEAL_CONSUMPTION_FIELDS,
    "horizon-terminal": HORIZON_TERMINAL_FIELDS,
}


def _artifact_kind_and_slot(
    raw: bytes,
    *,
    entry_kind: str,
    declared_slot: int | None,
    contract: runtime.FrozenContract,
) -> tuple[str, int | None, str]:
    """Return receipt kind, derived slot, and artifact predecessor."""

    if entry_kind == "segment-attestation" and declared_slot is None:
        identity = runtime.hash_and_bytes(raw)
        if identity == contract.watermark_identity:
            runtime.validate_control_watermark_bytes(raw)
            return "control-watermark", None, ""
        receipt = _receipt(raw, runtime.SUCCESSOR_ATTESTATION_FIELDS)
        if (
            receipt["receipt_kind"] != "runtime-segment-attestation"
            or receipt["status"] != "pass"
        ):
            _fail()
        return receipt["receipt_kind"], None, _hex64(
            receipt["previous_ledger_entry_sha256"]
        )
    fields = _ARTIFACT_FIELDS.get(entry_kind)
    if fields is None:
        _fail()
    receipt = _receipt(raw, fields)
    kind = receipt["receipt_kind"]
    expected_kind = {
        "runtime-attestation-attempt": "runtime-attestation-attempt",
        "runtime-attestation-terminal": "runtime-attestation-terminal",
        "source-binding-attestation": "source-binding-attestation",
        "probe-attempt": "probe-attempt",
        "probe-failure": "probe-failure",
        "probe-terminal": "probe-terminal",
        "below-floor": "slot-probe-resolution",
        "ready": "slot-probe-resolution",
        INITIAL_CLOSURE_ENTRY_KIND: runtime.INITIAL_CLOSURE_RECEIPT_KIND,
        "slot-segment-closed": "slot-segment-closed",
        "missed-slot": "missed-slot",
        "seal-consumption": "seal-consumption",
        "horizon-terminal": "horizon-terminal",
    }[entry_kind]
    if kind != expected_kind:
        _fail()
    status_by_kind = {
        "runtime-attestation-terminal": "terminal-runtime-attestation-failure",
        "source-binding-attestation": "pass",
        "probe-terminal": "terminal-probe-integrity-failure",
        "below-floor": "below-floor",
        "ready": "ready",
        INITIAL_CLOSURE_ENTRY_KIND: runtime.CLOSURE_STATUS,
        "slot-segment-closed": "segment-closed",
        "missed-slot": "terminal-schedule-integrity-failure",
        "horizon-terminal": "insufficient-evidence",
    }
    if entry_kind in status_by_kind and receipt.get("status") != status_by_kind[entry_kind]:
        _fail()
    if entry_kind in {"runtime-attestation-attempt", "runtime-attestation-terminal"}:
        scope = receipt["attempt_scope"]
        derived_slot = 0 if scope == "initial-source-binding" else None
    elif entry_kind == "source-binding-attestation":
        derived_slot = 0 if receipt["segment_index"] == 0 else None
    elif entry_kind == "horizon-terminal":
        derived_slot = _slot_index(receipt["final_slot_index"])
    elif entry_kind == INITIAL_CLOSURE_ENTRY_KIND:
        # The pre-binding closure exists only at slot 0 and nowhere else.
        derived_slot = _slot_index(receipt["slot_index"])
        if (
            derived_slot != runtime.FIRST_SLOT_INDEX
            or receipt["segment_index"] != 0
            or type(receipt["segment_index"]) is not int
        ):
            _fail()
    else:
        derived_slot = _slot_index(receipt["slot_index"])
    if declared_slot != derived_slot:
        _fail()
    return kind, derived_slot, _hex64(receipt["previous_ledger_entry_sha256"])


def validate_ledger_entry(
    raw: bytes,
    *,
    artifact_raw: bytes,
    contract: runtime.FrozenContract,
    expected_entry_index: int,
    expected_previous_ledger_sha256: str,
) -> ValidatedLedgerEntry:
    """Validate one entry and resolve its exact artifact bytes."""

    runtime._require_frozen_contract_consistent(contract)
    entry = _receipt(raw, LEDGER_ENTRY_FIELDS)
    index = _nonnegative_int(entry["entry_index"])
    if index != _nonnegative_int(expected_entry_index):
        _fail()
    kind = entry["entry_kind"]
    if type(kind) is not str or kind not in ENTRY_KINDS:
        _fail()
    slot_value = entry["slot_index_or_null"]
    if slot_value is not None:
        slot_value = _slot_index(slot_value)
    artifact_identity = runtime.hash_and_bytes(artifact_raw)
    _same_identity(entry["artifact_sha256_and_bytes"], artifact_identity)
    previous = _hex64(entry["previous_ledger_entry_sha256"])
    if previous != _hex64(expected_previous_ledger_sha256):
        _fail()
    expected_entry_sha = derive_ledger_entry_sha256(entry)
    if _hex64(entry["entry_sha256"]) != expected_entry_sha:
        _fail()
    _, derived_slot, artifact_previous = _artifact_kind_and_slot(
        artifact_raw,
        entry_kind=kind,
        declared_slot=slot_value,
        contract=contract,
    )
    if derived_slot != slot_value:
        _fail()
    if index == 0:
        if (
            kind != "segment-attestation"
            or slot_value is not None
            or artifact_identity != contract.watermark_identity
            or previous != derive_ledger_genesis(contract)
            or artifact_previous
        ):
            _fail()
    elif artifact_previous != previous:
        _fail()
    return ValidatedLedgerEntry(
        raw=raw,
        identity=runtime.hash_and_bytes(raw),
        artifact_raw=artifact_raw,
        artifact_identity=artifact_identity,
        entry_index=index,
        entry_kind=kind,
        slot_index=slot_value,
        previous_ledger_entry_sha256=previous,
        entry_sha256=expected_entry_sha,
    )


def _build_entry(
    *,
    contract: runtime.FrozenContract,
    entry_index: int,
    previous: str,
    entry_kind: str,
    slot_index: int | None,
    artifact_raw: bytes,
) -> ValidatedLedgerEntry:
    if entry_kind not in ENTRY_KINDS:
        _fail()
    artifact_identity = runtime.hash_and_bytes(artifact_raw)
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "entry_index": _nonnegative_int(entry_index),
        "entry_kind": entry_kind,
        "slot_index_or_null": slot_index,
        "artifact_sha256_and_bytes": artifact_identity.as_dict(),
        "previous_ledger_entry_sha256": _hex64(previous),
        "entry_sha256": "",
    }
    if tuple(value) != LEDGER_ENTRY_FIELDS:
        _fail()
    value["entry_sha256"] = derive_ledger_entry_sha256(value)
    raw = runtime.canonical_json_bytes(value)
    return validate_ledger_entry(
        raw,
        artifact_raw=artifact_raw,
        contract=contract,
        expected_entry_index=entry_index,
        expected_previous_ledger_sha256=previous,
    )


def initialize_ledger(
    contract: runtime.FrozenContract | None = None,
) -> LedgerState:
    """Create the deterministic frozen genesis state (entry zero)."""

    frozen = runtime.load_frozen_contract() if contract is None else contract
    runtime._require_frozen_contract_consistent(frozen)
    watermark_raw = runtime.read_regular_file(
        runtime.DEFAULT_WATERMARK,
        expected=frozen.watermark_identity,
        maximum_bytes=runtime.PINNED_WATERMARK_BYTES,
    )
    entry = _authorize_ledger_entry(
        _build_entry(
            contract=frozen,
            entry_index=0,
            previous=derive_ledger_genesis(frozen),
            entry_kind="segment-attestation",
            slot_index=None,
            artifact_raw=watermark_raw,
        )
    )
    entries = (entry,)
    return LedgerState(
        contract=frozen,
        entries=entries,
        phase=PHASE_BOOTSTRAP_PROBE,
        next_slot_index=0,
        _history_auth=_history_auth(entries),
    )


def _derive_state_projection(
    entries: tuple[ValidatedLedgerEntry, ...],
) -> _StateProjection:
    """Replay every state-bearing identity from the canonical transcript."""

    phase = PHASE_BOOTSTRAP_PROBE
    next_slot: int | None = 0
    terminal: str | None = None
    current_attempt_identity: runtime.HashAndBytes | None = None
    current_attempt_ordinal: int | None = None
    runtime_attempt_identity: runtime.HashAndBytes | None = None
    source_binding_identity: runtime.HashAndBytes | None = None
    active_attestation_identity: runtime.HashAndBytes | None = None
    active_source_binding_identity: runtime.HashAndBytes | None = None
    closed_attestation_identity: runtime.HashAndBytes | None = None
    closed_source_binding_identity: runtime.HashAndBytes | None = None
    closure_identity: runtime.HashAndBytes | None = None
    pre_binding_closure = False
    pre_binding_attempt_identity: runtime.HashAndBytes | None = None
    failure_identity: runtime.HashAndBytes | None = None
    seal_identity: runtime.HashAndBytes | None = None
    seal_attempt_identity: runtime.HashAndBytes | None = None
    seal_attestation_identity: runtime.HashAndBytes | None = None
    seal_source_binding_identity: runtime.HashAndBytes | None = None
    final_identity: runtime.HashAndBytes | None = None
    final_status: str | None = None
    for entry in entries[1:]:
        kind = entry.entry_kind
        receipt = runtime.load_canonical_json_bytes(
            entry.artifact_raw, expected=dict
        )
        if phase == PHASE_BOOTSTRAP_PROBE:
            if kind == "probe-attempt" and entry.slot_index == 0:
                if receipt["attempt_ordinal"] != 0:
                    _fail()
                current_attempt_identity = entry.artifact_identity
                current_attempt_ordinal = 0
                phase = PHASE_INITIAL_RUNTIME_ATTEMPT
            elif kind == "missed-slot" and entry.slot_index == 0:
                phase = PHASE_TERMINAL
                terminal = "terminal-schedule-integrity-failure"
                next_slot = None
            else:
                _fail()
        elif phase == PHASE_INITIAL_RUNTIME_ATTEMPT:
            if kind == "runtime-attestation-attempt" and entry.slot_index == 0:
                runtime_attempt_identity = entry.artifact_identity
                phase = PHASE_INITIAL_SOURCE_BINDING
            elif kind == "runtime-attestation-terminal" and entry.slot_index == 0:
                phase = PHASE_TERMINAL
                terminal = "terminal-runtime-attestation-failure"
                next_slot = None
                current_attempt_identity = None
                runtime_attempt_identity = None
            else:
                _fail()
        elif phase == PHASE_INITIAL_SOURCE_BINDING:
            if kind == "source-binding-attestation" and entry.slot_index == 0:
                if runtime_attempt_identity is None:
                    _fail()
                active_attestation_identity = entries[0].artifact_identity
                active_source_binding_identity = entry.artifact_identity
                runtime_attempt_identity = None
                phase = PHASE_PROBE_ACTIVE
            elif (
                kind == INITIAL_CLOSURE_ENTRY_KIND
                and entry.slot_index == runtime.FIRST_SLOT_INDEX
                and next_slot == runtime.FIRST_SLOT_INDEX
            ):
                # The repaired edge.  Slot 0 is consumed by a count-free,
                # snapshot-free closure; every latch dies with it and the only
                # thing that survives is the typed pre-binding predecessor,
                # replayed here as "closure present, no closed active segment".
                if runtime_attempt_identity is None:
                    _fail()
                closure_identity = entry.artifact_identity
                pre_binding_closure = True
                pre_binding_attempt_identity = runtime_attempt_identity
                runtime_attempt_identity = None
                current_attempt_identity = None
                current_attempt_ordinal = None
                closed_attestation_identity = None
                closed_source_binding_identity = None
                active_attestation_identity = None
                active_source_binding_identity = None
                next_slot = NEXT_SLOT_AFTER_INITIAL_CLOSURE
                phase = PHASE_SUCCESSOR_ATTEMPT
            elif kind == "runtime-attestation-terminal" and entry.slot_index == 0:
                phase = PHASE_TERMINAL
                terminal = "terminal-runtime-attestation-failure"
                next_slot = None
                current_attempt_identity = None
                runtime_attempt_identity = None
            else:
                _fail()
        elif phase == PHASE_AWAITING_PROBE:
            if kind == "probe-attempt" and entry.slot_index == next_slot:
                if receipt["attempt_ordinal"] != 0:
                    _fail()
                current_attempt_identity = entry.artifact_identity
                current_attempt_ordinal = 0
                phase = PHASE_PROBE_ACTIVE
            elif kind == "missed-slot" and entry.slot_index == next_slot:
                phase = PHASE_TERMINAL
                terminal = "terminal-schedule-integrity-failure"
                next_slot = None
                active_attestation_identity = None
                active_source_binding_identity = None
            else:
                _fail()
        elif phase == PHASE_PROBE_ACTIVE:
            if entry.slot_index != next_slot:
                _fail()
            if kind == "probe-failure":
                if receipt["attempt_ordinal"] != current_attempt_ordinal:
                    _fail()
                failure_identity = entry.artifact_identity
                phase = (
                    PHASE_RETRY_ALLOWED
                    if receipt["retry_authorized"] is True
                    else PHASE_PROBE_TERMINAL
                )
            elif kind == "below-floor":
                current_attempt_identity = None
                current_attempt_ordinal = None
                failure_identity = None
                if next_slot == runtime.LAST_SLOT_INDEX:
                    final_identity = entry.artifact_identity
                    final_status = "below-floor"
                    active_attestation_identity = None
                    active_source_binding_identity = None
                    phase = PHASE_HORIZON
                    next_slot = None
                else:
                    next_slot = _slot_index(next_slot) + 1
                    phase = PHASE_AWAITING_PROBE
            elif kind == "slot-segment-closed":
                current_attempt_identity = None
                current_attempt_ordinal = None
                failure_identity = None
                if next_slot == runtime.LAST_SLOT_INDEX:
                    final_identity = entry.artifact_identity
                    final_status = "segment-closed"
                    active_attestation_identity = None
                    active_source_binding_identity = None
                    phase = PHASE_HORIZON
                    next_slot = None
                else:
                    if (
                        active_attestation_identity is None
                        or active_source_binding_identity is None
                    ):
                        _fail()
                    closed_attestation_identity = active_attestation_identity
                    closed_source_binding_identity = active_source_binding_identity
                    closure_identity = entry.artifact_identity
                    active_attestation_identity = None
                    active_source_binding_identity = None
                    next_slot = _slot_index(next_slot) + 1
                    phase = PHASE_SUCCESSOR_ATTEMPT
            elif kind == "seal-consumption":
                if (
                    current_attempt_identity is None
                    or active_attestation_identity is None
                    or active_source_binding_identity is None
                ):
                    _fail()
                seal_identity = entry.artifact_identity
                seal_attempt_identity = current_attempt_identity
                seal_attestation_identity = active_attestation_identity
                seal_source_binding_identity = active_source_binding_identity
                current_attempt_identity = None
                current_attempt_ordinal = None
                active_attestation_identity = None
                active_source_binding_identity = None
                phase = PHASE_SEAL_CONSUMED
            else:
                _fail()
        elif phase == PHASE_RETRY_ALLOWED:
            if kind == "probe-attempt" and entry.slot_index == next_slot:
                if receipt["attempt_ordinal"] != current_attempt_ordinal + 1:
                    _fail()
                current_attempt_identity = entry.artifact_identity
                current_attempt_ordinal = receipt["attempt_ordinal"]
                failure_identity = None
                phase = PHASE_PROBE_ACTIVE
            elif kind == "missed-slot" and entry.slot_index == next_slot:
                phase = PHASE_TERMINAL
                terminal = "terminal-schedule-integrity-failure"
                next_slot = None
                current_attempt_identity = None
                current_attempt_ordinal = None
                failure_identity = None
                active_attestation_identity = None
                active_source_binding_identity = None
            else:
                _fail()
        elif phase == PHASE_PROBE_TERMINAL:
            if kind != "probe-terminal" or entry.slot_index != next_slot:
                _fail()
            phase = PHASE_TERMINAL
            terminal = "terminal-probe-integrity-failure"
            next_slot = None
            current_attempt_identity = None
            current_attempt_ordinal = None
            failure_identity = None
            active_attestation_identity = None
            active_source_binding_identity = None
        elif phase == PHASE_SUCCESSOR_ATTEMPT:
            if kind == "runtime-attestation-attempt" and entry.slot_index is None:
                runtime_attempt_identity = entry.artifact_identity
                phase = PHASE_SUCCESSOR_BINDING
            elif kind == "runtime-attestation-terminal" and entry.slot_index is None:
                phase = PHASE_TERMINAL
                terminal = "terminal-runtime-attestation-failure"
                next_slot = None
                closed_attestation_identity = None
                closed_source_binding_identity = None
                closure_identity = None
                pre_binding_closure = False
                pre_binding_attempt_identity = None
            else:
                _fail()
        elif phase == PHASE_SUCCESSOR_BINDING:
            if kind == "source-binding-attestation" and entry.slot_index is None:
                if runtime_attempt_identity is None:
                    _fail()
                source_binding_identity = entry.artifact_identity
                phase = PHASE_SUCCESSOR_SEGMENT
            elif kind == "runtime-attestation-terminal" and entry.slot_index is None:
                phase = PHASE_TERMINAL
                terminal = "terminal-runtime-attestation-failure"
                next_slot = None
                runtime_attempt_identity = None
                closed_attestation_identity = None
                closed_source_binding_identity = None
                closure_identity = None
                pre_binding_closure = False
                pre_binding_attempt_identity = None
            else:
                _fail()
        elif phase == PHASE_SUCCESSOR_SEGMENT:
            if kind == "segment-attestation" and entry.slot_index is None:
                if source_binding_identity is None:
                    _fail()
                active_attestation_identity = entry.artifact_identity
                active_source_binding_identity = source_binding_identity
                runtime_attempt_identity = None
                source_binding_identity = None
                closed_attestation_identity = None
                closed_source_binding_identity = None
                closure_identity = None
                pre_binding_closure = False
                pre_binding_attempt_identity = None
                phase = PHASE_AWAITING_PROBE
            elif kind == "runtime-attestation-terminal" and entry.slot_index is None:
                phase = PHASE_TERMINAL
                terminal = "terminal-runtime-attestation-failure"
                next_slot = None
                runtime_attempt_identity = None
                source_binding_identity = None
                closed_attestation_identity = None
                closed_source_binding_identity = None
                closure_identity = None
                pre_binding_closure = False
                pre_binding_attempt_identity = None
            else:
                _fail()
        elif phase == PHASE_SEAL_CONSUMED:
            if kind != "ready" or entry.slot_index != next_slot:
                _fail()
            phase = PHASE_READY_VISIBLE
            next_slot = None
            current_attempt_identity = None
            final_identity = entry.artifact_identity
            final_status = "ready"
        elif phase == PHASE_HORIZON:
            if kind != "horizon-terminal" or entry.slot_index != runtime.LAST_SLOT_INDEX:
                _fail()
            phase = PHASE_TERMINAL
            terminal = "insufficient-evidence"
        else:
            _fail()
    return _StateProjection(
        phase=phase,
        next_slot_index=next_slot,
        terminal_status=terminal,
        current_probe_attempt_identity=current_attempt_identity,
        runtime_attempt_identity=runtime_attempt_identity,
        source_binding_identity=source_binding_identity,
        active_attestation_identity=active_attestation_identity,
        active_source_binding_identity=active_source_binding_identity,
        closed_attestation_identity=closed_attestation_identity,
        closed_source_binding_identity=closed_source_binding_identity,
        closure_identity=closure_identity,
        pre_binding_closure=pre_binding_closure,
        pre_binding_closure_attempt_identity=pre_binding_attempt_identity,
        probe_failure_identity=failure_identity,
        seal_consumption_identity=seal_identity,
        seal_probe_attempt_identity=seal_attempt_identity,
        seal_active_attestation_identity=seal_attestation_identity,
        seal_active_source_binding_identity=seal_source_binding_identity,
        final_resolution_identity=final_identity,
        final_resolution_status=final_status,
    )


def _entry_for_artifact(
    entries: tuple[ValidatedLedgerEntry, ...],
    identity: runtime.HashAndBytes,
    entry_kind: str,
) -> ValidatedLedgerEntry:
    matches = tuple(
        entry
        for entry in entries
        if entry.entry_kind == entry_kind
        and entry.artifact_identity == identity
    )
    if len(matches) != 1:
        _fail()
    return matches[0]


def _require_exact_artifact(
    record: ValidatedLedgerEntry,
    raw: bytes,
    identity: runtime.HashAndBytes,
) -> None:
    if (
        type(raw) is not bytes
        or type(identity) is not runtime.HashAndBytes
        or record.artifact_identity != identity
        or not hmac.compare_digest(record.artifact_raw, raw)
    ):
        _fail()


def _require_runtime_attempt_bound(
    state: LedgerState,
    attempt: runtime.AttestationAttempt,
    expected_identity: runtime.HashAndBytes,
) -> ValidatedLedgerEntry:
    runtime._require_attempt_wrapper_consistent(attempt)
    record = _entry_for_artifact(
        state.entries, expected_identity, "runtime-attestation-attempt"
    )
    _require_exact_artifact(record, attempt.raw, attempt.identity)
    if (
        attempt.identity != expected_identity
        or attempt.analysis_plan_identity != state.contract.plan_identity
        or attempt.previous_ledger_entry_sha256
        != record.previous_ledger_entry_sha256
    ):
        _fail()
    return record


def _require_source_attestation_bound(
    state: LedgerState,
    attestation: runtime.SourceBindingAttestation,
    expected_identity: runtime.HashAndBytes,
) -> ValidatedLedgerEntry:
    runtime._require_source_attestation_wrapper_consistent(attestation)
    record = _entry_for_artifact(
        state.entries, expected_identity, "source-binding-attestation"
    )
    _require_exact_artifact(record, attestation.raw, attestation.identity)
    attempt_record = _require_runtime_attempt_bound(
        state, attestation.attempt, attestation.attempt.identity
    )
    if (
        attestation.identity != expected_identity
        or attestation.analysis_plan_identity != state.contract.plan_identity
        or attestation.previous_ledger_entry_sha256
        != record.previous_ledger_entry_sha256
        or attempt_record.entry_index >= record.entry_index
    ):
        _fail()
    return record


def closure_entry_kind(
    closure: runtime.ValidatedClosure | runtime.InitialRuntimeChangeClosure,
) -> str:
    """Map a closure wrapper to the one entry kind that may ever carry it."""

    if type(closure) is runtime.InitialRuntimeChangeClosure:
        return INITIAL_CLOSURE_ENTRY_KIND
    if type(closure) is runtime.ValidatedClosure:
        return CLOSURE_ENTRY_KIND
    return _fail()


def _require_closure_bound_to_history(
    state: LedgerState,
    closure: runtime.ValidatedClosure | runtime.InitialRuntimeChangeClosure,
    expected_identity: runtime.HashAndBytes,
) -> ValidatedLedgerEntry:
    runtime._require_closure_wrapper_consistent(state.contract, closure)
    record = _entry_for_artifact(
        state.entries, expected_identity, closure_entry_kind(closure)
    )
    _require_exact_artifact(record, closure.raw, closure.identity)
    if closure.identity != expected_identity:
        _fail()
    if type(closure) is runtime.InitialRuntimeChangeClosure:
        # The pre-binding closure is anchored to the frozen watermark
        # predecessor and to the slot-0 attempt marker it consumed, never to a
        # prior source binding: there has never been one to point at.
        runtime._require_unbound_predecessor_consistent(
            state.contract, closure.predecessor
        )
        attempt_record = _entry_for_artifact(
            state.entries,
            closure.attempt_identity,
            "runtime-attestation-attempt",
        )
        if (
            record.slot_index != runtime.FIRST_SLOT_INDEX
            or closure.slot_index != runtime.FIRST_SLOT_INDEX
            or closure.predecessor.attestation_identity
            != state.entries[0].artifact_identity
            or attempt_record.entry_index >= record.entry_index
        ):
            _fail()
    return record


def _require_unbound_predecessor_bound(
    state: LedgerState,
    predecessor: runtime.UnboundWatermarkPredecessor,
    closure: runtime.InitialRuntimeChangeClosure,
) -> ValidatedLedgerEntry:
    """Bind the typed pre-binding predecessor to entry zero and its closure."""

    if (
        type(predecessor) is not runtime.UnboundWatermarkPredecessor
        or type(closure) is not runtime.InitialRuntimeChangeClosure
    ):
        _fail()
    runtime._require_predecessor_pair_consistent(predecessor, closure)
    runtime._require_unbound_predecessor_consistent(state.contract, predecessor)
    genesis = state.entries[0]
    if (
        closure.predecessor != predecessor
        or genesis.entry_kind != "segment-attestation"
        or genesis.slot_index is not None
        or genesis.artifact_identity != predecessor.attestation_identity
        or genesis.artifact_identity != state.contract.watermark_identity
    ):
        _fail()
    return _require_closure_bound_to_history(state, closure, closure.identity)


def _predecessor_observer_identity(
    predecessor: runtime.ActiveSegment | runtime.UnboundWatermarkPredecessor,
    closure: runtime.ValidatedClosure | runtime.InitialRuntimeChangeClosure,
) -> runtime.HashAndBytes:
    """Return the observer identity a successor ceremony must reuse."""

    runtime._require_predecessor_pair_consistent(predecessor, closure)
    if type(predecessor) is runtime.UnboundWatermarkPredecessor:
        # Before any binding exists there is no source-binding attestation to
        # read an observer from; the pre-binding closure carries it instead.
        return closure.observer_identity
    return predecessor.source_binding_attestation.runtime_observer_identity


def _require_closed_predecessor_pair(
    state: LedgerState,
) -> tuple[
    runtime.ActiveSegment | runtime.UnboundWatermarkPredecessor,
    runtime.ValidatedClosure | runtime.InitialRuntimeChangeClosure,
]:
    """Return the single closed predecessor and closure the state carries."""

    predecessor = state.closed_predecessor
    closure = state.closure
    if (
        predecessor is None
        or closure is None
        or state.active_segment is not None
        or type(predecessor) not in runtime.PREDECESSOR_TYPES
        or type(closure) not in runtime.CLOSURE_TYPES
    ):
        _fail()
    runtime._require_predecessor_pair_consistent(predecessor, closure)
    runtime._require_closure_wrapper_consistent(state.contract, closure)
    if closure.segment_index != predecessor.segment_index or (
        closure.segment_id != predecessor.segment_id
    ):
        _fail()
    return predecessor, closure


def _require_active_segment_bound(
    state: LedgerState,
    active: runtime.ActiveSegment,
    expected_attestation: runtime.HashAndBytes,
    expected_source_binding: runtime.HashAndBytes,
    *,
    _seen: set[int] | None = None,
) -> None:
    runtime._require_active_segment_consistent(state.contract, active)
    seen = set() if _seen is None else _seen
    if id(active) in seen:
        _fail()
    seen.add(id(active))
    source_record = _require_source_attestation_bound(
        state,
        active.source_binding_attestation,
        expected_source_binding,
    )
    if active.attestation_identity != expected_attestation:
        _fail()
    if active.segment_index == 0:
        attestation_record = state.entries[0]
        if (
            attestation_record.entry_kind != "segment-attestation"
            or attestation_record.artifact_identity != expected_attestation
            or source_record.entry_index <= attestation_record.entry_index
        ):
            _fail()
        return
    if type(active.attestation_raw) is not bytes:
        _fail()
    # A pre-binding predecessor pairs only with the pre-binding closure, and an
    # ordinary closed segment only with the strict two-core closure.
    runtime._require_predecessor_pair_consistent(
        active.predecessor_segment, active.predecessor_closure
    )
    pre_binding = (
        type(active.predecessor_segment) is runtime.UnboundWatermarkPredecessor
    )
    expected_observer = (
        active.predecessor_closure.observer_identity
        if pre_binding
        else active.predecessor_segment.source_binding_attestation.runtime_observer_identity
    )
    if (
        active.source_binding_attestation.runtime_observer_identity
        != expected_observer
    ):
        _fail()
    attestation_record = _entry_for_artifact(
        state.entries, expected_attestation, "segment-attestation"
    )
    _require_exact_artifact(
        attestation_record, active.attestation_raw, active.attestation_identity
    )
    if pre_binding:
        # There is no prior active segment to recurse into: the chain ends at
        # the frozen watermark predecessor, which owns no source binding.
        closure_record = _require_unbound_predecessor_bound(
            state, active.predecessor_segment, active.predecessor_closure
        )
    else:
        closure_record = _require_closure_bound_to_history(
            state,
            active.predecessor_closure,
            active.predecessor_closure.identity,
        )
        _require_active_segment_bound(
            state,
            active.predecessor_segment,
            active.predecessor_segment.attestation_identity,
            active.predecessor_segment.source_binding_attestation.identity,
            _seen=seen,
        )
    if not (
        closure_record.entry_index
        < source_record.entry_index
        < attestation_record.entry_index
    ):
        _fail()


def _require_probe_attempt_bound(
    state: LedgerState,
    attempt: probe.ValidatedProbeAttempt,
    expected_identity: runtime.HashAndBytes,
    active: runtime.ActiveSegment | None,
) -> ValidatedLedgerEntry:
    probe._require_probe_attempt_wrapper_consistent(
        attempt, state.contract, active
    )
    record = _entry_for_artifact(
        state.entries, expected_identity, "probe-attempt"
    )
    _require_exact_artifact(record, attempt.raw, attempt.identity)
    if (
        attempt.identity != expected_identity
        or attempt.probe_identity != probe._self_identity()
        or attempt.analysis_plan_identity != state.contract.plan_identity
        or attempt.previous_ledger_entry_sha256
        != record.previous_ledger_entry_sha256
    ):
        _fail()
    return record


def _require_probe_failure_bound(
    state: LedgerState,
    failure: probe.ValidatedProbeFailure,
    expected_identity: runtime.HashAndBytes,
    active: runtime.ActiveSegment,
) -> None:
    probe._require_probe_failure_wrapper_consistent(
        failure, state.contract, active
    )
    record = _entry_for_artifact(
        state.entries, expected_identity, "probe-failure"
    )
    _require_exact_artifact(record, failure.raw, failure.identity)
    attempt_record = _require_probe_attempt_bound(
        state, failure.attempt, failure.attempt.identity, active
    )
    if (
        failure.identity != expected_identity
        or failure.analysis_plan_identity != state.contract.plan_identity
        or failure.previous_ledger_entry_sha256
        != record.previous_ledger_entry_sha256
        or attempt_record.entry_index >= record.entry_index
    ):
        _fail()


def _require_sealed_provisional_bound(
    state: LedgerState,
    provisional: ProvisionalReady,
    projection: _StateProjection,
) -> None:
    if (
        projection.seal_probe_attempt_identity is None
        or projection.seal_active_attestation_identity is None
        or projection.seal_active_source_binding_identity is None
    ):
        _fail()
    shadow = replace(
        state,
        next_slot_index=provisional.slot_index,
        active_segment=provisional.active_segment,
    )
    _require_provisional(shadow, provisional)
    _require_active_segment_bound(
        state,
        provisional.active_segment,
        projection.seal_active_attestation_identity,
        projection.seal_active_source_binding_identity,
    )
    attempt_record = _entry_for_artifact(
        state.entries,
        projection.seal_probe_attempt_identity,
        "probe-attempt",
    )
    attempt_receipt = _receipt(attempt_record.artifact_raw, probe.PROBE_ATTEMPT_FIELDS)
    if (
        provisional.core["slot_index"] != attempt_receipt["slot_index"]
        or provisional.core["segment_id"] != attempt_receipt["segment_id"]
        or provisional.core["launched_at"] != attempt_receipt["launched_at"]
        or _identity(provisional.core["probe_sha256_and_bytes"])
        != probe._self_identity()
        or _identity(provisional.core["segment_attestation_sha256_and_bytes"])
        != projection.seal_active_attestation_identity
        or _identity(
            provisional.core[
                "source_binding_attestation_sha256_and_bytes"
            ]
        )
        != projection.seal_active_source_binding_identity
    ):
        _fail()


def _require_seal_consumption_bound(
    state: LedgerState,
    marker: ValidatedSealConsumption,
    projection: _StateProjection,
) -> None:
    if (
        type(marker) is not ValidatedSealConsumption
        or projection.seal_consumption_identity is None
        or type(state.provisional_ready) is not ProvisionalReady
        or marker.provisional is not state.provisional_ready
    ):
        _fail()
    _require_sealed_provisional_bound(
        state, state.provisional_ready, projection
    )
    receipt = _receipt(marker.raw, SEAL_CONSUMPTION_FIELDS)
    record = _entry_for_artifact(
        state.entries,
        projection.seal_consumption_identity,
        "seal-consumption",
    )
    _require_exact_artifact(record, marker.raw, marker.identity)
    consumed = runtime.parse_utc(receipt["consumed_at"], receipt=True)
    if (
        marker.identity != projection.seal_consumption_identity
        or marker.receipt != receipt
        or marker.slot_index != state.provisional_ready.slot_index
        or marker.segment_id != state.provisional_ready.segment_id
        or marker.consumed_at != consumed
        or marker.sealer_identity
        != _identity(receipt["sealer_sha256_and_bytes"])
        or receipt["seal_authority_id"] != SEAL_AUTHORITY_ID
        or _identity(receipt["provisional_ready_core_sha256_and_bytes"])
        != state.provisional_ready.core_identity
        or _identity(receipt["snapshot_set_sha256_and_bytes"])
        != state.provisional_ready.snapshot_set_identity
        or _hex64(receipt["previous_ledger_entry_sha256"])
        != record.previous_ledger_entry_sha256
        or _identity(receipt["analysis_plan_sha256_and_bytes"])
        != state.contract.plan_identity
        or not marker.consumed_at
        <= state.provisional_ready.validated_at
        <= marker.consumed_at + timedelta(seconds=HANDOFF_SECONDS)
    ):
        _fail()


def _require_state(state: LedgerState) -> None:
    if (
        type(state) is not LedgerState
        or type(state.contract) is not runtime.FrozenContract
        or type(state.entries) is not tuple
        or not state.entries
        or state.phase not in _PHASES
        or not (
            state.next_slot_index is None
            or type(state.next_slot_index) is int
        )
    ):
        _fail()
    runtime._require_frozen_contract_consistent(state.contract)
    if (
        type(state._history_auth) is not bytes
        or not hmac.compare_digest(
            state._history_auth, _history_auth(state.entries)
        )
    ):
        _fail()
    previous = derive_ledger_genesis(state.contract)
    for index, record in enumerate(state.entries):
        if (
            type(record) is not ValidatedLedgerEntry
            or type(record._semantic_auth) is not bytes
            or not hmac.compare_digest(
                record._semantic_auth,
                _ledger_entry_semantic_auth(record),
            )
        ):
            _fail()
        rebound = validate_ledger_entry(
            record.raw,
            artifact_raw=record.artifact_raw,
            contract=state.contract,
            expected_entry_index=index,
            expected_previous_ledger_sha256=previous,
        )
        if rebound != record:
            _fail()
        previous = rebound.entry_sha256
    projection = _derive_state_projection(state.entries)
    phase_matches = state.phase == projection.phase or (
        projection.phase == PHASE_SEAL_CONSUMED
        and state.phase == PHASE_BLOCKED_SPAWN
        and type(state.blocked_spawn) is BlockedSpawnProof
    )
    if (
        not phase_matches
        or state.next_slot_index != projection.next_slot_index
        or state.terminal_status != projection.terminal_status
        or state.final_resolution_identity
        != projection.final_resolution_identity
        or state.final_resolution_status
        != projection.final_resolution_status
    ):
        _fail()
    if projection.active_attestation_identity is None:
        if state.active_segment is not None:
            _fail()
    elif (
        type(state.active_segment) is not runtime.ActiveSegment
        or projection.active_source_binding_identity is None
    ):
        _fail()
    else:
        _require_active_segment_bound(
            state,
            state.active_segment,
            projection.active_attestation_identity,
            projection.active_source_binding_identity,
        )
    if projection.current_probe_attempt_identity is None:
        if state.current_probe_attempt is not None:
            _fail()
    else:
        if type(state.current_probe_attempt) is not probe.ValidatedProbeAttempt:
            _fail()
        _require_probe_attempt_bound(
            state,
            state.current_probe_attempt,
            projection.current_probe_attempt_identity,
            state.active_segment,
        )
    if projection.runtime_attempt_identity is None:
        if state.runtime_attempt is not None:
            _fail()
    elif type(state.runtime_attempt) is not runtime.AttestationAttempt:
        _fail()
    else:
        _require_runtime_attempt_bound(
            state, state.runtime_attempt, projection.runtime_attempt_identity
        )
    if projection.source_binding_identity is None:
        if state.source_binding_attestation is not None:
            _fail()
    elif type(state.source_binding_attestation) is not runtime.SourceBindingAttestation:
        _fail()
    else:
        _require_source_attestation_bound(
            state,
            state.source_binding_attestation,
            projection.source_binding_identity,
        )
    if projection.closure_identity is None:
        if (
            state.closed_segment is not None
            or state.unbound_predecessor is not None
            or state.closure is not None
            or projection.pre_binding_closure
            or projection.pre_binding_closure_attempt_identity is not None
        ):
            _fail()
    elif projection.pre_binding_closure:
        # Pre-binding break: a typed predecessor and nothing else.  No closed
        # ActiveSegment, no closed attestation, no closed source binding.
        if (
            state.closed_segment is not None
            or type(state.unbound_predecessor)
            is not runtime.UnboundWatermarkPredecessor
            or type(state.closure) is not runtime.InitialRuntimeChangeClosure
            or projection.closed_attestation_identity is not None
            or projection.closed_source_binding_identity is not None
            or projection.pre_binding_closure_attempt_identity is None
            or state.closure.attempt_identity
            != projection.pre_binding_closure_attempt_identity
            or state.closure.identity != projection.closure_identity
        ):
            _fail()
        _require_unbound_predecessor_bound(
            state, state.unbound_predecessor, state.closure
        )
    elif (
        type(state.closed_segment) is not runtime.ActiveSegment
        or state.unbound_predecessor is not None
        or type(state.closure) is not runtime.ValidatedClosure
        or projection.closed_attestation_identity is None
        or projection.closed_source_binding_identity is None
        or projection.pre_binding_closure_attempt_identity is not None
    ):
        _fail()
    else:
        _require_active_segment_bound(
            state,
            state.closed_segment,
            projection.closed_attestation_identity,
            projection.closed_source_binding_identity,
        )
        _require_closure_bound_to_history(
            state, state.closure, projection.closure_identity
        )
    if projection.probe_failure_identity is None:
        if state.probe_failure is not None:
            _fail()
    elif (
        type(state.probe_failure) is not probe.ValidatedProbeFailure
        or type(state.active_segment) is not runtime.ActiveSegment
    ):
        _fail()
    else:
        _require_probe_failure_bound(
            state,
            state.probe_failure,
            projection.probe_failure_identity,
            state.active_segment,
        )
    if projection.seal_consumption_identity is None:
        if (
            state.provisional_ready is not None
            or state.seal_consumption is not None
            or state.blocked_spawn is not None
        ):
            _fail()
    else:
        _require_seal_consumption_bound(
            state, state.seal_consumption, projection
        )
        if state.phase == PHASE_SEAL_CONSUMED:
            if state.blocked_spawn is not None:
                _fail()
        elif state.phase in {PHASE_BLOCKED_SPAWN, PHASE_READY_VISIBLE}:
            _require_blocked_spawn(state, state.blocked_spawn)
        else:
            _fail()
    if state.next_slot_index is not None:
        _slot_index(state.next_slot_index)


def _append(
    state: LedgerState,
    *,
    entry_kind: str,
    slot_index: int | None,
    artifact_raw: bytes,
    **changes: Any,
) -> LedgerState:
    _require_state(state)
    entry = _authorize_ledger_entry(
        _build_entry(
            contract=state.contract,
            entry_index=state.next_entry_index,
            previous=state.head,
            entry_kind=entry_kind,
            slot_index=slot_index,
            artifact_raw=artifact_raw,
        )
    )
    entries = state.entries + (entry,)
    result = replace(
        state,
        entries=entries,
        _history_auth=_history_auth(entries),
        **changes,
    )
    _require_state(result)
    return result


def append_probe_attempt(
    state: LedgerState,
    attempt: probe.ValidatedProbeAttempt,
) -> LedgerState:
    """Append the only probe attempt currently authorized by the state."""

    _require_state(state)
    if type(attempt) is not probe.ValidatedProbeAttempt:
        _fail()
    if state.phase == PHASE_BOOTSTRAP_PROBE:
        expected_slot = 0
        expected_ordinal = 0
        rebound = probe.validate_probe_attempt_marker(
            attempt.raw,
            contract=state.contract,
            active_segment=None,
            expected_segment_id=state.contract.initial_segment_id,
            expected_runtime_observer_identity=attempt.runtime_observer_identity,
            expected_previous_ledger_sha256=state.head,
            expected_slot_index=expected_slot,
            expected_attempt_ordinal=expected_ordinal,
            expected_probe_identity=probe._self_identity(),
        )
        next_phase = PHASE_INITIAL_RUNTIME_ATTEMPT
    elif state.phase in {PHASE_AWAITING_PROBE, PHASE_RETRY_ALLOWED}:
        if type(state.active_segment) is not runtime.ActiveSegment:
            _fail()
        expected_slot = _slot_index(state.next_slot_index)
        expected_ordinal = (
            0
            if state.phase == PHASE_AWAITING_PROBE
            else state.probe_failure.attempt.attempt_ordinal + 1
            if type(state.probe_failure) is probe.ValidatedProbeFailure
            and state.probe_failure.retry_authorized
            else _fail()
        )
        rebound = probe.validate_probe_attempt_marker(
            attempt.raw,
            contract=state.contract,
            active_segment=state.active_segment,
            expected_previous_ledger_sha256=state.head,
            expected_slot_index=expected_slot,
            expected_attempt_ordinal=expected_ordinal,
            expected_probe_identity=probe._self_identity(),
        )
        next_phase = PHASE_PROBE_ACTIVE
    else:
        _fail()
    if rebound != attempt:
        _fail()
    if (
        state.phase == PHASE_RETRY_ALLOWED
        and type(state.probe_failure) is probe.ValidatedProbeFailure
        and rebound.launched_at < state.probe_failure.failed_at
    ):
        _fail()
    return _append(
        state,
        entry_kind="probe-attempt",
        slot_index=expected_slot,
        artifact_raw=attempt.raw,
        phase=next_phase,
        current_probe_attempt=rebound,
        probe_failure=None,
        provisional_ready=None,
        seal_consumption=None,
        blocked_spawn=None,
    )


def append_runtime_attestation_attempt(
    state: LedgerState,
    attempt: runtime.AttestationAttempt,
) -> LedgerState:
    _require_state(state)
    if type(attempt) is not runtime.AttestationAttempt:
        _fail()
    if state.phase == PHASE_INITIAL_RUNTIME_ATTEMPT:
        if type(state.current_probe_attempt) is not probe.ValidatedProbeAttempt:
            _fail()
        rebound = runtime.validate_attestation_attempt_marker(
            attempt.raw,
            contract=state.contract,
            runtime_observer_identity=(
                state.current_probe_attempt.runtime_observer_identity
            ),
            expected_previous_ledger_sha256=state.head,
            initial_probe_launched_at=runtime.canonical_utc(
                state.current_probe_attempt.launched_at
            ),
        )
        slot = 0
        next_phase = PHASE_INITIAL_SOURCE_BINDING
    elif state.phase == PHASE_SUCCESSOR_ATTEMPT:
        predecessor, closure = _require_closed_predecessor_pair(state)
        rebound = runtime.validate_attestation_attempt_marker(
            attempt.raw,
            contract=state.contract,
            runtime_observer_identity=_predecessor_observer_identity(
                predecessor, closure
            ),
            expected_previous_ledger_sha256=state.head,
            predecessor_closure=closure,
        )
        slot = None
        next_phase = PHASE_SUCCESSOR_BINDING
    else:
        _fail()
    if rebound != attempt:
        _fail()
    return _append(
        state,
        entry_kind="runtime-attestation-attempt",
        slot_index=slot,
        artifact_raw=attempt.raw,
        phase=next_phase,
        runtime_attempt=rebound,
    )


def append_source_binding_attestation(
    state: LedgerState,
    attestation: runtime.SourceBindingAttestation,
) -> LedgerState:
    _require_state(state)
    if (
        type(attestation) is not runtime.SourceBindingAttestation
        or type(state.runtime_attempt) is not runtime.AttestationAttempt
    ):
        _fail()
    if state.phase not in {
        PHASE_INITIAL_SOURCE_BINDING,
        PHASE_SUCCESSOR_BINDING,
    }:
        _fail()
    rebound = runtime.validate_source_binding_attestation(
        attestation.raw,
        contract=state.contract,
        attempt=state.runtime_attempt,
        binding=attestation.binding,
        runtime_observer_identity=state.runtime_attempt.runtime_observer_identity,
        expected_previous_ledger_sha256=state.head,
    )
    if rebound != attestation:
        _fail()
    if state.phase == PHASE_INITIAL_SOURCE_BINDING:
        active = runtime.make_initial_segment(state.contract, rebound)
        return _append(
            state,
            entry_kind="source-binding-attestation",
            slot_index=0,
            artifact_raw=attestation.raw,
            phase=PHASE_PROBE_ACTIVE,
            active_segment=active,
            runtime_attempt=None,
            source_binding_attestation=None,
        )
    return _append(
        state,
        entry_kind="source-binding-attestation",
        slot_index=None,
        artifact_raw=attestation.raw,
        phase=PHASE_SUCCESSOR_SEGMENT,
        source_binding_attestation=rebound,
    )


def append_successor_segment(
    state: LedgerState,
    active_segment: runtime.ActiveSegment,
) -> LedgerState:
    _require_state(state)
    if (
        state.phase != PHASE_SUCCESSOR_SEGMENT
        or type(active_segment) is not runtime.ActiveSegment
        or type(active_segment.attestation_raw) is not bytes
        or type(state.runtime_attempt) is not runtime.AttestationAttempt
        or type(state.source_binding_attestation)
        is not runtime.SourceBindingAttestation
    ):
        _fail()
    predecessor, closure = _require_closed_predecessor_pair(state)
    runtime._require_active_segment_consistent(state.contract, active_segment)
    if (
        active_segment.predecessor_segment is not predecessor
        or active_segment.predecessor_closure is not closure
        or active_segment.source_binding_attestation
        != state.source_binding_attestation
        or active_segment.source_binding_attestation.attempt
        != state.runtime_attempt
        or active_segment.attestation_previous_ledger_sha256 != state.head
        or active_segment.segment_index != predecessor.segment_index + 1
        or active_segment.source_binding_attestation.runtime_observer_identity
        != _predecessor_observer_identity(predecessor, closure)
    ):
        _fail()
    return _append(
        state,
        entry_kind="segment-attestation",
        slot_index=None,
        artifact_raw=active_segment.attestation_raw,
        phase=PHASE_AWAITING_PROBE,
        active_segment=active_segment,
        current_probe_attempt=None,
        runtime_attempt=None,
        source_binding_attestation=None,
        closed_segment=None,
        unbound_predecessor=None,
        closure=None,
        probe_failure=None,
        provisional_ready=None,
        seal_consumption=None,
        blocked_spawn=None,
    )


def _validate_runtime_terminal(
    state: LedgerState,
    raw: bytes,
) -> tuple[dict[str, Any], int | None]:
    receipt = _receipt(raw, RUNTIME_TERMINAL_FIELDS)
    if (
        receipt["receipt_kind"] != "runtime-attestation-terminal"
        or receipt["status"] != "terminal-runtime-attestation-failure"
        or receipt["reason"] not in RUNTIME_TERMINAL_REASONS
        or runtime.HashAndBytes.from_value(
            receipt["analysis_plan_sha256_and_bytes"]
        )
        != state.contract.plan_identity
        or _hex64(receipt["previous_ledger_entry_sha256"]) != state.head
    ):
        _fail()
    recorded = runtime.parse_utc(receipt["recorded_at"], receipt=True)
    marker_value = receipt[
        "attestation_attempt_marker_sha256_and_bytes_or_null"
    ]
    reason = receipt["reason"]
    if state.phase in {
        PHASE_INITIAL_RUNTIME_ATTEMPT,
        PHASE_INITIAL_SOURCE_BINDING,
    }:
        scope = "initial-source-binding"
        segment_index = 0
        slot = 0
        closure_sha = None
    elif state.phase in {
        PHASE_SUCCESSOR_ATTEMPT,
        PHASE_SUCCESSOR_BINDING,
        PHASE_SUCCESSOR_SEGMENT,
    }:
        predecessor, pending_closure = _require_closed_predecessor_pair(state)
        scope = "successor-segment"
        segment_index = predecessor.segment_index + 1
        slot = None
        closure_sha = pending_closure.identity.sha256
    else:
        _fail()
    supplied_closure = receipt["predecessor_closure_sha256_or_null"]
    if supplied_closure is not None:
        supplied_closure = _hex64(supplied_closure)
    if (
        receipt["attempt_scope"] != scope
        or _nonnegative_int(receipt["segment_index"]) != segment_index
        or supplied_closure != closure_sha
    ):
        _fail()
    if state.runtime_attempt is None:
        if marker_value is not None or reason not in RUNTIME_TERMINAL_NULL_MARKER_REASONS:
            _fail()
        earliest = (
            state.current_probe_attempt.launched_at
            if scope == "initial-source-binding"
            and type(state.current_probe_attempt) is probe.ValidatedProbeAttempt
            else state.closure.validated_at
            if type(state.closure) in runtime.CLOSURE_TYPES
            else _fail()
        )
        start_deadline = (
            runtime.parse_utc(
                runtime.slot_times(0).grace_deadline_at, receipt=True
            )
            if scope == "initial-source-binding"
            else state.closure.validated_at + timedelta(seconds=30)
        )
        deadline = start_deadline
    else:
        if reason in RUNTIME_TERMINAL_PRE_MARKER_ONLY_REASONS:
            _fail()
        _same_identity(marker_value, state.runtime_attempt.identity)
        earliest = state.runtime_attempt.written_at
        start_deadline = state.runtime_attempt.start_deadline_at
        deadline = (
            start_deadline
            if scope == "initial-source-binding"
            else runtime.parse_utc(
                runtime.slot_times(_slot_index(state.next_slot_index)).scheduled_at,
                receipt=True,
            )
        )
    if (
        recorded < earliest
        or recorded > deadline
        or scope == "initial-source-binding"
        and reason in {"boundary-invalid", "next-slot-deadline"}
        or reason == "attempt-start-late"
        and recorded != start_deadline
        or state.runtime_attempt is None
        and reason == "watchdog-timeout"
        and recorded != start_deadline
        or reason == "next-slot-deadline"
        and (
            scope != "successor-segment"
            or state.runtime_attempt is None
            or recorded != deadline
        )
        or scope == "successor-segment"
        and state.runtime_attempt is not None
        and recorded == deadline
        and reason != "next-slot-deadline"
    ):
        _fail()
    return receipt, slot


def append_runtime_attestation_terminal(
    state: LedgerState,
    raw: bytes,
) -> LedgerState:
    _, slot = _validate_runtime_terminal(state, raw)
    return _append(
        state,
        entry_kind="runtime-attestation-terminal",
        slot_index=slot,
        artifact_raw=raw,
        phase=PHASE_TERMINAL,
        next_slot_index=None,
        active_segment=None,
        current_probe_attempt=None,
        runtime_attempt=None,
        source_binding_attestation=None,
        closed_segment=None,
        unbound_predecessor=None,
        closure=None,
        terminal_status="terminal-runtime-attestation-failure",
    )


def append_probe_failure(
    state: LedgerState,
    failure: probe.ValidatedProbeFailure,
) -> LedgerState:
    _require_state(state)
    if (
        state.phase != PHASE_PROBE_ACTIVE
        or type(state.active_segment) is not runtime.ActiveSegment
        or type(state.current_probe_attempt) is not probe.ValidatedProbeAttempt
        or type(failure) is not probe.ValidatedProbeFailure
    ):
        _fail()
    rebound = probe.validate_probe_failure_marker(
        failure.raw,
        contract=state.contract,
        active_segment=state.active_segment,
        attempt=state.current_probe_attempt,
        expected_previous_ledger_sha256=state.head,
    )
    if rebound != failure:
        _fail()
    phase = PHASE_RETRY_ALLOWED if rebound.retry_authorized else PHASE_PROBE_TERMINAL
    return _append(
        state,
        entry_kind="probe-failure",
        slot_index=rebound.attempt.slot_index,
        artifact_raw=rebound.raw,
        phase=phase,
        probe_failure=rebound,
    )


def append_probe_terminal(
    state: LedgerState,
    raw: bytes,
) -> LedgerState:
    _require_state(state)
    if (
        state.phase != PHASE_PROBE_TERMINAL
        or type(state.active_segment) is not runtime.ActiveSegment
        or type(state.probe_failure) is not probe.ValidatedProbeFailure
    ):
        _fail()
    identity = probe.validate_probe_terminal_marker(
        raw,
        contract=state.contract,
        active_segment=state.active_segment,
        failure=state.probe_failure,
        expected_previous_ledger_sha256=state.head,
    )
    if identity != runtime.hash_and_bytes(raw):
        _fail()
    return _append(
        state,
        entry_kind="probe-terminal",
        slot_index=state.probe_failure.attempt.slot_index,
        artifact_raw=raw,
        phase=PHASE_TERMINAL,
        next_slot_index=None,
        active_segment=None,
        current_probe_attempt=None,
        runtime_attempt=None,
        source_binding_attestation=None,
        probe_failure=None,
        terminal_status="terminal-probe-integrity-failure",
    )


def _advance_after_resolution(
    state: LedgerState,
    *,
    raw: bytes,
    identity: runtime.HashAndBytes,
    status: str,
    entry_kind: str,
) -> LedgerState:
    slot = _slot_index(state.next_slot_index)
    if slot == runtime.LAST_SLOT_INDEX:
        return _append(
            state,
            entry_kind=entry_kind,
            slot_index=slot,
            artifact_raw=raw,
            phase=PHASE_HORIZON,
            next_slot_index=None,
            active_segment=None,
            current_probe_attempt=None,
            runtime_attempt=None,
            source_binding_attestation=None,
            probe_failure=None,
            provisional_ready=None,
            seal_consumption=None,
            blocked_spawn=None,
            final_resolution_identity=identity,
            final_resolution_status=status,
        )
    return _append(
        state,
        entry_kind=entry_kind,
        slot_index=slot,
        artifact_raw=raw,
        phase=PHASE_AWAITING_PROBE,
        next_slot_index=slot + 1,
        current_probe_attempt=None,
        runtime_attempt=None,
        source_binding_attestation=None,
        probe_failure=None,
        provisional_ready=None,
        seal_consumption=None,
        blocked_spawn=None,
    )


def append_below_floor_resolution(
    state: LedgerState,
    resolution: probe.ValidatedProbeResolution,
) -> LedgerState:
    _require_state(state)
    if (
        state.phase != PHASE_PROBE_ACTIVE
        or type(state.active_segment) is not runtime.ActiveSegment
        or type(state.current_probe_attempt) is not probe.ValidatedProbeAttempt
        or type(resolution) is not probe.ValidatedProbeResolution
    ):
        _fail()
    rebound = probe.validate_probe_resolution(
        resolution.raw,
        contract=state.contract,
        active_segment=state.active_segment,
        expected_previous_ledger_sha256=state.head,
        expected_probe_identity=state.current_probe_attempt.probe_identity,
    )
    receipt = runtime.load_canonical_json_bytes(resolution.raw, expected=dict)
    if (
        rebound != resolution
        or rebound.status != "below-floor"
        or rebound.slot_index != state.next_slot_index
        or receipt["launched_at"]
        != runtime.canonical_utc(state.current_probe_attempt.launched_at)
    ):
        _fail()
    return _advance_after_resolution(
        state,
        raw=resolution.raw,
        identity=rebound.identity,
        status="below-floor",
        entry_kind="below-floor",
    )


def bind_initial_runtime_change_closure(
    state: LedgerState,
    closure: runtime.InitialRuntimeChangeClosure,
    *,
    observed_services: runtime.ServiceTuple,
    observed_services_raw: bytes,
    observed_binding: runtime.SourceBinding,
    observed_source_binding_core_raw: bytes,
    runtime_observer_identity: runtime.HashAndBytes | Mapping[str, Any],
) -> BoundInitialRuntimeChangeClosure:
    """Revalidate the sole pre-binding closure against every retained object.

    The repaired ``initial-source-binding-required`` exit.  There is no prior
    source-binding core to supply and none is inferred or backfilled: the whole
    proof is the frozen contract, the one slot-0 attempt, and the complete
    stable observation that differs from the pinned watermark tuple.
    """

    _require_state(state)
    if (
        state.phase != PHASE_INITIAL_SOURCE_BINDING
        or state.active_segment is not None
        or state.closure is not None
        or state.unbound_predecessor is not None
        or state.closed_segment is not None
        or state.next_slot_index != runtime.FIRST_SLOT_INDEX
        or type(state.runtime_attempt) is not runtime.AttestationAttempt
        or type(state.current_probe_attempt) is not probe.ValidatedProbeAttempt
        or type(closure) is not runtime.InitialRuntimeChangeClosure
        or type(observed_services) is not runtime.ServiceTuple
    ):
        _fail()
    # Outcome-blind branch selection: only a complete stable observed tuple
    # that differs from the pinned watermark tuple can reach this closure.
    if (
        runtime.initial_bootstrap_branch(state.contract, observed_services)
        != runtime.CHANGED_TUPLE_BRANCH
    ):
        _fail()
    rebound = runtime.validate_initial_runtime_change_closure(
        closure.raw,
        contract=state.contract,
        attempt=state.runtime_attempt,
        observed_services=observed_services,
        observed_services_raw=observed_services_raw,
        observed_binding=observed_binding,
        observed_source_binding_core_raw=observed_source_binding_core_raw,
        runtime_observer_identity=runtime_observer_identity,
        expected_previous_ledger_sha256=state.head,
    )
    if rebound != closure:
        _fail()
    receipt = _receipt(closure.raw, runtime.INITIAL_CLOSURE_FIELDS)
    observer = _identity(runtime_observer_identity)
    if (
        closure.slot_index != state.next_slot_index
        or closure.slot_index != runtime.FIRST_SLOT_INDEX
        or closure.segment_index != 0
        or closure.segment_id != state.contract.initial_segment_id
        or closure.attempt_identity != state.runtime_attempt.identity
        or closure.observer_identity != observer
        or observer != state.runtime_attempt.runtime_observer_identity
        or observer != state.current_probe_attempt.runtime_observer_identity
        or _hex64(receipt["previous_ledger_entry_sha256"]) != state.head
        # The pre-binding ceremony continues the one slot-0 probe attempt; it
        # never authorizes a second attempt marker for the same slot.
        or receipt["launched_at"]
        != runtime.canonical_utc(state.current_probe_attempt.launched_at)
        or state.current_probe_attempt.slot_index != runtime.FIRST_SLOT_INDEX
        or _identity(receipt["attestation_sha256_and_bytes"])
        != state.contract.watermark_identity
        # No count and no snapshot may ever appear on this receipt.
        or set(receipt) & _COUNT_AND_SNAPSHOT_FIELDS
    ):
        _fail()
    bound = object.__new__(BoundInitialRuntimeChangeClosure)
    for name, value in {
        "closure": rebound,
        "predecessor": rebound.predecessor,
        "ledger_head": state.head,
        "attempt": state.runtime_attempt,
        "observed_services": observed_services,
        "observed_services_raw": observed_services_raw,
        "observed_binding": observed_binding,
        "observed_source_binding_core_raw": observed_source_binding_core_raw,
        "runtime_observer_identity": observer,
        "_origin": _INITIAL_CLOSURE_BINDING_ORIGIN,
    }.items():
        object.__setattr__(bound, name, value)
    return bound


def append_initial_runtime_change_closure(
    state: LedgerState,
    bound_closure: BoundInitialRuntimeChangeClosure,
) -> LedgerState:
    """Consume slot 0 with the count-free pre-binding closure.

    Nothing crosses this break: every latch is dropped, source authority ends,
    the next probe slot becomes 1, and the only thing retained is the typed
    pre-binding predecessor -- never a fabricated ``ActiveSegment``.
    """

    _require_state(state)
    if (
        type(bound_closure) is not BoundInitialRuntimeChangeClosure
        or bound_closure._origin is not _INITIAL_CLOSURE_BINDING_ORIGIN
        or state.phase != PHASE_INITIAL_SOURCE_BINDING
        or state.active_segment is not None
        or state.closure is not None
        or state.unbound_predecessor is not None
        or state.closed_segment is not None
        or bound_closure.ledger_head != state.head
        or bound_closure.attempt is not state.runtime_attempt
        or type(getattr(bound_closure, "observed_services", None))
        is not runtime.ServiceTuple
        or type(getattr(bound_closure, "observed_services_raw", None))
        is not bytes
        or type(getattr(bound_closure, "observed_binding", None))
        is not runtime.SourceBinding
        or type(
            getattr(bound_closure, "observed_source_binding_core_raw", None)
        )
        is not bytes
        or type(getattr(bound_closure, "runtime_observer_identity", None))
        is not runtime.HashAndBytes
        or type(getattr(bound_closure, "predecessor", None))
        is not runtime.UnboundWatermarkPredecessor
    ):
        _fail()
    closure = runtime.validate_initial_runtime_change_closure(
        bound_closure.closure.raw,
        contract=state.contract,
        attempt=state.runtime_attempt,
        observed_services=bound_closure.observed_services,
        observed_services_raw=bound_closure.observed_services_raw,
        observed_binding=bound_closure.observed_binding,
        observed_source_binding_core_raw=(
            bound_closure.observed_source_binding_core_raw
        ),
        runtime_observer_identity=bound_closure.runtime_observer_identity,
        expected_previous_ledger_sha256=state.head,
    )
    if (
        closure != bound_closure.closure
        or closure.predecessor != bound_closure.predecessor
    ):
        _fail()
    runtime._require_unbound_predecessor_consistent(
        state.contract, closure.predecessor
    )
    return _append(
        state,
        entry_kind=INITIAL_CLOSURE_ENTRY_KIND,
        slot_index=closure.slot_index,
        artifact_raw=closure.raw,
        phase=PHASE_SUCCESSOR_ATTEMPT,
        next_slot_index=NEXT_SLOT_AFTER_INITIAL_CLOSURE,
        active_segment=None,
        current_probe_attempt=None,
        runtime_attempt=None,
        source_binding_attestation=None,
        closed_segment=None,
        unbound_predecessor=closure.predecessor,
        closure=closure,
        probe_failure=None,
        provisional_ready=None,
        seal_consumption=None,
        blocked_spawn=None,
    )


def bind_segment_closure(
    state: LedgerState,
    closure: runtime.ValidatedClosure,
    *,
    prior_services_raw: bytes,
    observed_services_raw: bytes,
    prior_source_binding_core_raw: bytes,
    observed_source_binding_core_raw: bytes,
    observed_binding: runtime.SourceBinding,
    runtime_observer_identity: runtime.HashAndBytes | Mapping[str, Any],
) -> BoundSegmentClosure:
    """Revalidate a closure with every retained content-addressed object."""

    _require_state(state)
    if (
        state.phase != PHASE_PROBE_ACTIVE
        or type(state.active_segment) is not runtime.ActiveSegment
        or type(state.current_probe_attempt) is not probe.ValidatedProbeAttempt
        or type(closure) is not runtime.ValidatedClosure
    ):
        _fail()
    rebound = runtime.validate_segment_closure(
        closure.raw,
        contract=state.contract,
        active_segment=state.active_segment,
        prior_services_raw=prior_services_raw,
        observed_services_raw=observed_services_raw,
        prior_source_binding_core_raw=prior_source_binding_core_raw,
        observed_source_binding_core_raw=observed_source_binding_core_raw,
        observed_binding=observed_binding,
        runtime_observer_identity=runtime_observer_identity,
        expected_previous_ledger_sha256=state.head,
    )
    if rebound != closure:
        _fail()
    receipt = _receipt(closure.raw, runtime.CLOSURE_FIELDS)
    if (
        closure.slot_index != state.next_slot_index
        or closure.segment_index != state.active_segment.segment_index
        or closure.segment_id != state.active_segment.segment_id
        or _hex64(receipt["previous_ledger_entry_sha256"]) != state.head
        or receipt["launched_at"]
        != runtime.canonical_utc(state.current_probe_attempt.launched_at)
        or _identity(receipt["attestation_sha256_and_bytes"])
        != state.active_segment.attestation_identity
        or _identity(receipt["runtime_observer_sha256_and_bytes"])
        != state.active_segment.source_binding_attestation.runtime_observer_identity
    ):
        _fail()
    bound = object.__new__(BoundSegmentClosure)
    object.__setattr__(bound, "closure", rebound)
    object.__setattr__(bound, "active_segment", state.active_segment)
    object.__setattr__(bound, "ledger_head", state.head)
    object.__setattr__(bound, "prior_services_raw", prior_services_raw)
    object.__setattr__(bound, "observed_services_raw", observed_services_raw)
    object.__setattr__(
        bound,
        "prior_source_binding_core_raw",
        prior_source_binding_core_raw,
    )
    object.__setattr__(
        bound,
        "observed_source_binding_core_raw",
        observed_source_binding_core_raw,
    )
    object.__setattr__(bound, "observed_binding", observed_binding)
    object.__setattr__(
        bound,
        "runtime_observer_identity",
        _identity(runtime_observer_identity),
    )
    object.__setattr__(bound, "_origin", _CLOSURE_BINDING_ORIGIN)
    return bound


def append_segment_closure(
    state: LedgerState,
    bound_closure: BoundSegmentClosure,
) -> LedgerState:
    """Append a count-free closure and erase all active evidence authority."""

    _require_state(state)
    if (
        type(bound_closure) is not BoundSegmentClosure
        or bound_closure._origin is not _CLOSURE_BINDING_ORIGIN
        or state.phase != PHASE_PROBE_ACTIVE
        or type(state.active_segment) is not runtime.ActiveSegment
        or bound_closure.ledger_head != state.head
        or type(getattr(bound_closure, "prior_services_raw", None)) is not bytes
        or type(getattr(bound_closure, "observed_services_raw", None)) is not bytes
        or type(
            getattr(bound_closure, "prior_source_binding_core_raw", None)
        )
        is not bytes
        or type(
            getattr(bound_closure, "observed_source_binding_core_raw", None)
        )
        is not bytes
        or type(getattr(bound_closure, "observed_binding", None))
        is not runtime.SourceBinding
        or type(getattr(bound_closure, "runtime_observer_identity", None))
        is not runtime.HashAndBytes
    ):
        _fail()
    _same_active_segment(
        state.contract, bound_closure.active_segment, state.active_segment
    )
    closure = runtime.validate_segment_closure(
        bound_closure.closure.raw,
        contract=state.contract,
        active_segment=state.active_segment,
        prior_services_raw=bound_closure.prior_services_raw,
        observed_services_raw=bound_closure.observed_services_raw,
        prior_source_binding_core_raw=(
            bound_closure.prior_source_binding_core_raw
        ),
        observed_source_binding_core_raw=(
            bound_closure.observed_source_binding_core_raw
        ),
        observed_binding=bound_closure.observed_binding,
        runtime_observer_identity=bound_closure.runtime_observer_identity,
        expected_previous_ledger_sha256=state.head,
    )
    if closure != bound_closure.closure:
        _fail()
    slot = closure.slot_index
    if slot == runtime.LAST_SLOT_INDEX:
        return _append(
            state,
            entry_kind="slot-segment-closed",
            slot_index=slot,
            artifact_raw=closure.raw,
            phase=PHASE_HORIZON,
            next_slot_index=None,
            active_segment=None,
            current_probe_attempt=None,
            runtime_attempt=None,
            source_binding_attestation=None,
            closed_segment=None,
            unbound_predecessor=None,
            closure=None,
            probe_failure=None,
            provisional_ready=None,
            seal_consumption=None,
            blocked_spawn=None,
            final_resolution_identity=closure.identity,
            final_resolution_status="segment-closed",
        )
    return _append(
        state,
        entry_kind="slot-segment-closed",
        slot_index=slot,
        artifact_raw=closure.raw,
        phase=PHASE_SUCCESSOR_ATTEMPT,
        next_slot_index=slot + 1,
        active_segment=None,
        current_probe_attempt=None,
        runtime_attempt=None,
        source_binding_attestation=None,
        closed_segment=state.active_segment,
        unbound_predecessor=None,
        closure=closure,
        probe_failure=None,
        provisional_ready=None,
        seal_consumption=None,
        blocked_spawn=None,
    )


def _validate_missed_marker(state: LedgerState, raw: bytes) -> int:
    receipt = _receipt(raw, MISSED_SLOT_FIELDS)
    if state.phase not in {
        PHASE_BOOTSTRAP_PROBE,
        PHASE_AWAITING_PROBE,
        PHASE_RETRY_ALLOWED,
    }:
        _fail()
    slot = _slot_index(state.next_slot_index)
    expected = runtime.slot_times(slot)
    recorded = runtime.parse_utc(receipt["recorded_at"], receipt=True)
    if (
        receipt["receipt_kind"] != "missed-slot"
        or receipt["status"] != "terminal-schedule-integrity-failure"
        or receipt["slot_index"] != slot
        or receipt["scheduled_at"] != expected.scheduled_at
        or receipt["grace_deadline_at"] != expected.grace_deadline_at
        or receipt["source_open_count"] != 0
        or type(receipt["source_open_count"]) is not int
        or recorded
        < runtime.parse_utc(expected.grace_deadline_at, receipt=True)
        or _hex64(receipt["previous_ledger_entry_sha256"]) != state.head
        or _identity(receipt["analysis_plan_sha256_and_bytes"])
        != state.contract.plan_identity
    ):
        _fail()
    return slot


def append_missed_slot(state: LedgerState, raw: bytes) -> LedgerState:
    slot = _validate_missed_marker(state, raw)
    return _append(
        state,
        entry_kind="missed-slot",
        slot_index=slot,
        artifact_raw=raw,
        phase=PHASE_TERMINAL,
        next_slot_index=None,
        active_segment=None,
        current_probe_attempt=None,
        runtime_attempt=None,
        source_binding_attestation=None,
        probe_failure=None,
        terminal_status="terminal-schedule-integrity-failure",
    )


_PROVISIONAL_ORIGIN = object()


def prepare_provisional_ready(
    state: LedgerState,
    computation: probe.ProbeComputation,
    *,
    launched_at: str,
    validated_at: str,
) -> ProvisionalReady:
    """Build the private all-floor core; this is not a slot resolution."""

    _require_state(state)
    if (
        state.phase != PHASE_PROBE_ACTIVE
        or type(state.active_segment) is not runtime.ActiveSegment
        or type(state.current_probe_attempt) is not probe.ValidatedProbeAttempt
        or type(computation) is not probe.ProbeComputation
        or not computation.provisional_ready
        or computation.slot_index != state.next_slot_index
    ):
        _fail()
    probe._active_segment_recheck(state.contract, state.active_segment)
    probe._require_production_computation_binding(
        computation, state.contract, state.active_segment
    )
    slot = runtime.validate_slot_times(
        slot_index=computation.slot_index,
        scheduled_at=computation.scheduled_at,
        grace_deadline_at=computation.grace_deadline_at,
        launched_at=launched_at,
        validated_at=validated_at,
    )
    if launched_at != runtime.canonical_utc(state.current_probe_attempt.launched_at):
        _fail()
    active = state.active_segment
    core_source = active.source_binding.core
    databases = core_source.database_map()
    bindings = core_source.binding_map()
    snapshots = computation.snapshots()
    core: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "namespace": NAMESPACE,
        "receipt_kind": "slot-probe-resolution",
        "status": "ready",
        "slot_index": slot.slot_index,
        "scheduled_at": slot.scheduled_at,
        "grace_deadline_at": slot.grace_deadline_at,
        "launched_at": launched_at,
        "validated_at": validated_at,
        "release_effective_at": state.contract.release_effective_at,
        "active_segment_lower_bound_exclusive_at": active.lower_bound_exclusive_at,
        "segment_id": active.segment_id,
        "segment_attestation_sha256_and_bytes": active.attestation_identity.as_dict(),
        "source_binding_attestation_sha256_and_bytes": (
            active.source_binding_attestation.identity.as_dict()
        ),
        "pre_active_services_state_sha256": active.services.identity.sha256,
        "post_active_services_state_sha256": active.services.identity.sha256,
        "pre_database_instance_identity_sha256_by_alias": databases,
        "post_database_instance_identity_sha256_by_alias": dict(databases),
        "pre_alias_service_database_binding_sha256_by_alias": bindings,
        "post_alias_service_database_binding_sha256_by_alias": dict(bindings),
        "aliased_source_snapshot_sha256_and_bytes": {
            alias: snapshots[alias].as_dict() for alias in runtime.SOURCE_ALIASES
        },
        **computation.aggregate(),
        "runtime_observer_sha256_and_bytes": (
            active.source_binding_attestation.runtime_observer_identity.as_dict()
        ),
        "probe_sha256_and_bytes": computation.probe_identity.as_dict(),
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
    }
    identity = _derive_ready_core_identity(core)
    snapshot_identity = derive_snapshot_set_identity(
        core["aliased_source_snapshot_sha256_and_bytes"]
    )
    return ProvisionalReady(
        core=core,
        core_identity=identity,
        snapshot_set_identity=snapshot_identity,
        snapshot_items=tuple(
            (alias, snapshots[alias]) for alias in runtime.SOURCE_ALIASES
        ),
        slot_index=slot.slot_index,
        segment_id=active.segment_id,
        active_segment=active,
        computation=computation,
        probe_identity=computation.probe_identity,
        validated_at=runtime.parse_utc(validated_at, receipt=True),
        _origin=_PROVISIONAL_ORIGIN,
    )


def _require_provisional(state: LedgerState, provisional: ProvisionalReady) -> None:
    if (
        type(provisional) is not ProvisionalReady
        or provisional._origin is not _PROVISIONAL_ORIGIN
        or type(provisional.core) is not dict
        or type(provisional.core_identity) is not runtime.HashAndBytes
        or type(provisional.snapshot_set_identity) is not runtime.HashAndBytes
        or type(provisional.snapshot_items) is not tuple
        or tuple(alias for alias, _ in provisional.snapshot_items)
        != runtime.SOURCE_ALIASES
        or provisional.slot_index != state.next_slot_index
        or type(state.active_segment) is not runtime.ActiveSegment
        or type(provisional.computation) is not probe.ProbeComputation
    ):
        _fail()
    _same_active_segment(
        state.contract, provisional.active_segment, state.active_segment
    )
    probe._require_production_computation_binding(
        provisional.computation,
        state.contract,
        state.active_segment,
    )
    if (
        _derive_ready_core_identity(provisional.core)
        != provisional.core_identity
        or derive_snapshot_set_identity(
            provisional.core["aliased_source_snapshot_sha256_and_bytes"]
        )
        != provisional.snapshot_set_identity
        or provisional.segment_id != state.active_segment.segment_id
        or provisional.probe_identity
        != _identity(provisional.core["probe_sha256_and_bytes"])
        or provisional.validated_at
        != runtime.parse_utc(provisional.core["validated_at"], receipt=True)
        or {
            field: provisional.core[field]
            for field in probe.AGGREGATE_FIELDS
        }
        != provisional.computation.aggregate()
        or {
            alias: _identity(
                provisional.core[
                    "aliased_source_snapshot_sha256_and_bytes"
                ][alias]
            )
            for alias in runtime.SOURCE_ALIASES
        }
        != provisional.computation.snapshots()
        or provisional.core["slot_index"]
        != provisional.computation.slot_index
        or provisional.core["scheduled_at"]
        != provisional.computation.scheduled_at
        or provisional.core["grace_deadline_at"]
        != provisional.computation.grace_deadline_at
        or provisional.core["active_segment_lower_bound_exclusive_at"]
        != provisional.computation.active_segment_lower_bound_exclusive_at
        or dict(provisional.snapshot_items)
        != {
            alias: _identity(
                provisional.core[
                    "aliased_source_snapshot_sha256_and_bytes"
                ][alias]
            )
            for alias in runtime.SOURCE_ALIASES
        }
    ):
        _fail()
    counts = {
        field: provisional.core[field] for field in probe.AGGREGATE_FIELDS
    }
    probe._aggregate_invariants(counts)
    if not probe._all_floors_pass(counts):
        _fail()


def validate_seal_consumption_marker(
    raw: bytes,
    *,
    state: LedgerState,
    provisional: ProvisionalReady,
    expected_sealer_identity: runtime.HashAndBytes | Mapping[str, Any],
) -> ValidatedSealConsumption:
    _require_state(state)
    if state.phase != PHASE_PROBE_ACTIVE:
        _fail()
    _require_provisional(state, provisional)
    receipt = _receipt(raw, SEAL_CONSUMPTION_FIELDS)
    consumed = runtime.parse_utc(receipt["consumed_at"], receipt=True)
    sealer = _identity(expected_sealer_identity)
    slot = runtime.slot_times(provisional.slot_index)
    if (
        receipt["receipt_kind"] != "seal-consumption"
        or receipt["seal_authority_id"] != SEAL_AUTHORITY_ID
        or receipt["slot_index"] != provisional.slot_index
        or receipt["segment_id"] != provisional.segment_id
        or _identity(receipt["provisional_ready_core_sha256_and_bytes"])
        != provisional.core_identity
        or _identity(receipt["snapshot_set_sha256_and_bytes"])
        != provisional.snapshot_set_identity
        or _hex64(receipt["previous_ledger_entry_sha256"]) != state.head
        or _identity(receipt["sealer_sha256_and_bytes"]) != sealer
        or sealer.bytes <= 0
        or _identity(receipt["analysis_plan_sha256_and_bytes"])
        != state.contract.plan_identity
        or not consumed
        <= provisional.validated_at
        <= consumed + timedelta(seconds=HANDOFF_SECONDS)
        or not runtime.parse_utc(slot.scheduled_at, receipt=True)
        <= consumed
        < runtime.parse_utc(slot.grace_deadline_at, receipt=True)
        or type(state.current_probe_attempt) is not probe.ValidatedProbeAttempt
        or consumed < state.current_probe_attempt.launched_at
    ):
        _fail()
    return ValidatedSealConsumption(
        raw=raw,
        identity=runtime.hash_and_bytes(raw),
        receipt=dict(receipt),
        slot_index=provisional.slot_index,
        segment_id=provisional.segment_id,
        consumed_at=consumed,
        provisional=provisional,
        sealer_identity=sealer,
    )


def append_seal_consumption(
    state: LedgerState,
    marker: ValidatedSealConsumption,
) -> LedgerState:
    _require_state(state)
    if type(marker) is not ValidatedSealConsumption:
        _fail()
    rebound = validate_seal_consumption_marker(
        marker.raw,
        state=state,
        provisional=marker.provisional,
        expected_sealer_identity=marker.sealer_identity,
    )
    if rebound != marker:
        _fail()
    return _append(
        state,
        entry_kind="seal-consumption",
        slot_index=marker.slot_index,
        artifact_raw=marker.raw,
        phase=PHASE_SEAL_CONSUMED,
        active_segment=None,
        current_probe_attempt=None,
        runtime_attempt=None,
        source_binding_attestation=None,
        closed_segment=None,
        unbound_predecessor=None,
        closure=None,
        probe_failure=None,
        provisional_ready=marker.provisional,
        seal_consumption=rebound,
    )


def record_blocked_spawn(
    state: LedgerState,
    *,
    marker: ValidatedSealConsumption,
    durable_marker: DurableSealConsumption,
    sealer_process_launched_at: str,
) -> LedgerState:
    """Trusted supervisor event: one child is blocked behind its closed gate.

    The sealer integration must call this only after it has created the child
    with an inherited, still-closed start gate.  The returned exact-type object
    is the capability checked by the ready transition; raw receipt values alone
    cannot construct it.
    """

    _require_state(state)
    if (
        state.phase != PHASE_SEAL_CONSUMED
        or type(marker) is not ValidatedSealConsumption
        or type(state.seal_consumption) is not ValidatedSealConsumption
        or marker != state.seal_consumption
        or marker.identity != state.entries[-1].artifact_identity
        or type(durable_marker) is not DurableSealConsumption
        or durable_marker._origin is not _DURABLE_SEAL_ORIGIN
        or type(getattr(durable_marker, "_nonce", None)) is not bytes
        or type(getattr(durable_marker, "_auth", None)) is not bytes
        or not hmac.compare_digest(
            durable_marker._auth,
            _durable_capability_auth(durable_marker),
        )
        or durable_marker.marker_identity != marker.identity
        or durable_marker.ledger_head != state.head
        or durable_marker.entry_index != state.entries[-1].entry_index
    ):
        _fail()
    launched = runtime.parse_utc(sealer_process_launched_at, receipt=True)
    deadline = marker.consumed_at + timedelta(seconds=HANDOFF_SECONDS)
    grace = runtime.parse_utc(
        runtime.slot_times(marker.slot_index).grace_deadline_at, receipt=True
    )
    if not marker.consumed_at <= launched <= deadline or not launched < grace:
        _fail()
    recover_exact_file(durable_marker.marker_path, marker.raw)
    recover_exact_file(
        durable_marker.ledger_entry_path, state.entries[-1].raw
    )
    _consume_durable_nonce(durable_marker._nonce)
    proof = object.__new__(BlockedSpawnProof)
    nonce = os.urandom(32)
    for name, value in {
        "marker_identity": marker.identity,
        "sealer_identity": marker.sealer_identity,
        "slot_index": marker.slot_index,
        "segment_id": marker.segment_id,
        "launched_at": launched,
        "ledger_head": state.head,
        "_nonce": nonce,
        "_origin": _BLOCKED_SPAWN_ORIGIN,
    }.items():
        object.__setattr__(proof, name, value)
    object.__setattr__(proof, "_auth", _blocked_spawn_capability_auth(proof))
    result = replace(
        state, phase=PHASE_BLOCKED_SPAWN, blocked_spawn=proof
    )
    _require_state(result)
    _require_blocked_spawn(result, proof)
    return result


def _require_blocked_spawn(
    state: LedgerState, proof_value: BlockedSpawnProof
) -> None:
    marker_record = (
        _entry_for_artifact(
            state.entries,
            state.seal_consumption.identity,
            "seal-consumption",
        )
        if type(state.seal_consumption) is ValidatedSealConsumption
        else None
    )
    if (
        type(proof_value) is not BlockedSpawnProof
        or proof_value._origin is not _BLOCKED_SPAWN_ORIGIN
        or type(getattr(proof_value, "_nonce", None)) is not bytes
        or type(getattr(proof_value, "_auth", None)) is not bytes
        or not hmac.compare_digest(
            proof_value._auth,
            _blocked_spawn_capability_auth(proof_value),
        )
        or state.phase not in {PHASE_BLOCKED_SPAWN, PHASE_READY_VISIBLE}
        or state.blocked_spawn is not proof_value
        or type(state.seal_consumption) is not ValidatedSealConsumption
        or proof_value.marker_identity != state.seal_consumption.identity
        or proof_value.sealer_identity != state.seal_consumption.sealer_identity
        or proof_value.slot_index != state.seal_consumption.slot_index
        or proof_value.segment_id != state.seal_consumption.segment_id
        or marker_record is None
        or proof_value.ledger_head != marker_record.entry_sha256
        or not state.seal_consumption.consumed_at
        <= proof_value.launched_at
        <= state.seal_consumption.consumed_at
        + timedelta(seconds=HANDOFF_SECONDS)
    ):
        _fail()


def build_ready_resolution(
    state: LedgerState,
    *,
    blocked_spawn: BlockedSpawnProof,
) -> dict[str, Any]:
    """Construct the exact private ready receipt after marker and blocked spawn."""

    _require_state(state)
    _require_blocked_spawn(state, blocked_spawn)
    if type(state.provisional_ready) is not ProvisionalReady:
        _fail()
    _require_provisional_for_consumed_state(state, state.provisional_ready)
    values: dict[str, Any] = {}
    for field in probe.PROBE_RESOLUTION_FIELDS:
        if field == "resolution_id":
            values[field] = ""
        elif field == "previous_ledger_entry_sha256":
            values[field] = state.head
        elif field == "seal_consumption_marker_sha256_and_bytes_or_null":
            values[field] = state.seal_consumption.identity.as_dict()
        elif field == "sealer_process_launched_at_or_null":
            values[field] = runtime.canonical_utc(blocked_spawn.launched_at)
        else:
            values[field] = state.provisional_ready.core[field]
    values["resolution_id"] = runtime.derive_slot_resolution_id(values)
    raw = runtime.canonical_json_bytes(values)
    probe.validate_probe_resolution(
        raw,
        contract=state.contract,
        active_segment=state.provisional_ready.active_segment,
        expected_previous_ledger_sha256=state.head,
        expected_probe_identity=state.provisional_ready.probe_identity,
        expected_snapshot_identities=dict(state.provisional_ready.snapshot_items),
        expected_seal_consumption_identity=state.seal_consumption.identity,
        expected_sealer_process_launched_at=runtime.canonical_utc(
            blocked_spawn.launched_at
        ),
    )
    return values


def _require_provisional_for_consumed_state(
    state: LedgerState, provisional: ProvisionalReady
) -> None:
    # The active source authority is deliberately no longer exposed once M is
    # durable.  Rebind the private core against the retained exact active
    # segment in the capability rather than restoring authority in state.
    if (
        type(state.seal_consumption) is not ValidatedSealConsumption
        or state.provisional_ready is not provisional
        or state.seal_consumption.provisional is not provisional
        or _identity(
            state.seal_consumption.receipt[
                "provisional_ready_core_sha256_and_bytes"
            ]
        )
        != provisional.core_identity
        or _identity(
            state.seal_consumption.receipt[
                "snapshot_set_sha256_and_bytes"
            ]
        )
        != provisional.snapshot_set_identity
    ):
        _fail()
    shadow = replace(state, active_segment=provisional.active_segment)
    _require_provisional(shadow, provisional)


def append_ready_resolution(
    state: LedgerState,
    resolution: probe.ValidatedProbeResolution,
    *,
    blocked_spawn: BlockedSpawnProof,
) -> LedgerState:
    _require_state(state)
    _require_blocked_spawn(state, blocked_spawn)
    if (
        type(resolution) is not probe.ValidatedProbeResolution
        or type(state.provisional_ready) is not ProvisionalReady
        or type(state.seal_consumption) is not ValidatedSealConsumption
    ):
        _fail()
    expected = build_ready_resolution(state, blocked_spawn=blocked_spawn)
    expected_raw = runtime.canonical_json_bytes(expected)
    if not hmac.compare_digest(resolution.raw, expected_raw):
        _fail()
    rebound = probe.validate_probe_resolution(
        resolution.raw,
        contract=state.contract,
        active_segment=state.provisional_ready.active_segment,
        expected_previous_ledger_sha256=state.head,
        expected_probe_identity=state.provisional_ready.probe_identity,
        expected_snapshot_identities=dict(state.provisional_ready.snapshot_items),
        expected_seal_consumption_identity=state.seal_consumption.identity,
        expected_sealer_process_launched_at=runtime.canonical_utc(
            blocked_spawn.launched_at
        ),
    )
    if rebound != resolution or rebound.status != "ready":
        _fail()
    return _append(
        state,
        entry_kind="ready",
        slot_index=resolution.slot_index,
        artifact_raw=resolution.raw,
        phase=PHASE_READY_VISIBLE,
        next_slot_index=None,
        active_segment=None,
        current_probe_attempt=None,
        runtime_attempt=None,
        source_binding_attestation=None,
        probe_failure=None,
        final_resolution_identity=resolution.identity,
        final_resolution_status="ready",
    )


def _validate_horizon_marker(state: LedgerState, raw: bytes) -> None:
    if (
        state.phase != PHASE_HORIZON
        or type(state.final_resolution_identity) is not runtime.HashAndBytes
        or state.final_resolution_status
        not in {"below-floor", "segment-closed"}
    ):
        _fail()
    receipt = _receipt(raw, HORIZON_TERMINAL_FIELDS)
    recorded = runtime.parse_utc(receipt["recorded_at"], receipt=True)
    horizon = runtime.parse_utc(runtime.absolute_horizon_at(), receipt=True)
    expected_reason = (
        "final-slot-below-floor"
        if state.final_resolution_status == "below-floor"
        else "final-slot-segment-closed"
    )
    if (
        receipt["receipt_kind"] != "horizon-terminal"
        or receipt["status"] != "insufficient-evidence"
        or receipt["final_slot_index"] != runtime.LAST_SLOT_INDEX
        or type(receipt["final_slot_index"]) is not int
        or _identity(receipt["final_slot_resolution_sha256_and_bytes"])
        != state.final_resolution_identity
        or receipt["absolute_horizon_expires_at"]
        != runtime.absolute_horizon_at()
        or recorded < horizon
        or receipt["reason"] != expected_reason
        or _hex64(receipt["previous_ledger_entry_sha256"]) != state.head
        or _identity(receipt["analysis_plan_sha256_and_bytes"])
        != state.contract.plan_identity
    ):
        _fail()


def append_horizon_terminal(state: LedgerState, raw: bytes) -> LedgerState:
    _validate_horizon_marker(state, raw)
    return _append(
        state,
        entry_kind="horizon-terminal",
        slot_index=runtime.LAST_SLOT_INDEX,
        artifact_raw=raw,
        phase=PHASE_TERMINAL,
        terminal_status="insufficient-evidence",
    )


def ledger_entry_filename(entry_index: Any) -> str:
    """Return this implementation's deterministic file name (not protocol data)."""

    return f"{_nonnegative_int(entry_index):08d}.json"


def _write_all(fd: int, raw: bytes) -> None:
    view = memoryview(raw)
    offset = 0
    while offset < len(view):
        try:
            written = os.write(fd, view[offset:])
        except OSError:
            _fail()
        if written <= 0:
            _fail()
        offset += written


class _ExclusiveCreateFailure(Exception):
    """Internal write failure retaining whether this call won O_EXCL."""

    def __init__(self, created: bool) -> None:
        super().__init__()
        self.created = created


def _write_exclusive_once(
    path: Path | str, raw: bytes
) -> runtime.HashAndBytes:
    """Create once and preserve whether a later error followed creation."""

    if type(raw) is not bytes:
        _fail()
    target = Path(path)
    if not target.name or target.name in {".", ".."}:
        _fail()
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_CLOEXEC", 0
    )
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(target.parent, directory_flags)
    except OSError:
        raise _ExclusiveCreateFailure(False) from None
    fd = -1
    created = False
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(target.name, flags, 0o600, dir_fd=parent_fd)
        created = True
        _write_all(fd, raw)
        os.fchmod(fd, 0o600)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.fsync(parent_fd)
    except (IntegrityFailure, OSError):
        raise _ExclusiveCreateFailure(created) from None
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.close(parent_fd)
        except OSError:
            pass
    return runtime.hash_and_bytes(raw)


def write_exclusive(path: Path | str, raw: bytes) -> runtime.HashAndBytes:
    """Create one durable regular file without overwrite or cleanup-on-error."""

    try:
        return _write_exclusive_once(path, raw)
    except _ExclusiveCreateFailure:
        _fail()


def write_ledger_entry_exclusive(
    directory: Path | str,
    state: LedgerState,
) -> Path:
    """Persist only the last transition of a fully present exact prefix.

    Requiring the state (rather than a free-standing entry) prevents callers
    from creating index 7 in an empty directory or attaching a valid entry to
    a different on-disk prefix.  Existing prefix files are read-only checked;
    the target itself is still a single O_EXCL write.
    """

    _require_state(state)
    entry = state.entries[-1]
    root = Path(directory)
    expected_prefix_names = {
        ledger_entry_filename(predecessor.entry_index)
        for predecessor in state.entries[:-1]
    }
    try:
        observed_names = {item.name for item in root.iterdir()}
    except OSError:
        _fail()
    if observed_names != expected_prefix_names:
        _fail()
    for predecessor in state.entries[:-1]:
        recover_exact_file(
            root / ledger_entry_filename(predecessor.entry_index),
            predecessor.raw,
        )
    path = root / ledger_entry_filename(entry.entry_index)
    identity = write_exclusive(path, entry.raw)
    if identity != entry.identity:
        _fail()
    validate_persisted_ledger_prefix(root, state)
    return path


def validate_persisted_ledger_prefix(
    directory: Path | str,
    state: LedgerState,
) -> None:
    """Read-only validate the exact local file-per-entry representation."""

    _require_state(state)
    root = Path(directory)
    expected_names = {
        ledger_entry_filename(entry.entry_index) for entry in state.entries
    }
    try:
        observed_names = {item.name for item in root.iterdir()}
    except OSError:
        _fail()
    if observed_names != expected_names:
        _fail()
    for entry in state.entries:
        recover_exact_file(
            root / ledger_entry_filename(entry.entry_index), entry.raw
        )


def _write_or_recover_exact(
    path: Path | str,
    raw: bytes,
) -> runtime.HashAndBytes:
    """Attempt O_EXCL once, then permit only exact read-only recovery."""

    try:
        return _write_exclusive_once(path, raw)
    except _ExclusiveCreateFailure as error:
        if not error.created:
            _fail()
        identity = recover_exact_file(path, raw)
        _fsync_exact_file(path, raw)
        return identity


def seal_consumption_filename(
    marker: ValidatedSealConsumption, entry_index: Any
) -> str:
    if type(marker) is not ValidatedSealConsumption:
        _fail()
    # Divergent valid marker bytes from the same H must contend on one path.
    return f"{_nonnegative_int(entry_index):08d}.seal-consumption.json"


def _validate_persisted_entries(
    directory: Path | str,
    entries: tuple[ValidatedLedgerEntry, ...],
) -> None:
    root = Path(directory)
    expected_names = {
        ledger_entry_filename(entry.entry_index) for entry in entries
    }
    try:
        observed_names = {item.name for item in root.iterdir()}
    except OSError:
        _fail()
    if observed_names != expected_names:
        _fail()
    for entry in entries:
        recover_exact_file(
            root / ledger_entry_filename(entry.entry_index), entry.raw
        )


def persist_seal_consumption_exclusive(
    state: LedgerState,
    *,
    marker_path: Path | str,
    ledger_directory: Path | str,
) -> DurableSealConsumption:
    """Durably establish marker then M, with exact read-after-error recovery."""

    _require_state(state)
    if (
        state.phase != PHASE_SEAL_CONSUMED
        or type(state.seal_consumption) is not ValidatedSealConsumption
        or state.entries[-1].entry_kind != "seal-consumption"
        or state.entries[-1].artifact_identity
        != state.seal_consumption.identity
        or not hmac.compare_digest(
            state.entries[-1].artifact_raw, state.seal_consumption.raw
        )
    ):
        _fail()
    marker_target = Path(marker_path)
    ledger_root = Path(ledger_directory)
    entry_target = ledger_root / ledger_entry_filename(
        state.entries[-1].entry_index
    )
    if (
        marker_target == entry_target
        or marker_target.parent == ledger_root
        or marker_target.name
        != seal_consumption_filename(
            state.seal_consumption, state.entries[-1].entry_index
        )
    ):
        # This local representation keeps artifacts and entry files in
        # separate directories so the ledger directory can be exact-member.
        _fail()
    # Establish that M is absent before consuming marker authority.  The
    # deterministic marker name makes concurrent invocations contend on the
    # same O_EXCL target rather than creating alternate markers.
    _validate_persisted_entries(ledger_root, state.entries[:-1])
    marker_identity = _write_or_recover_exact(
        marker_target, state.seal_consumption.raw
    )
    if marker_identity != state.seal_consumption.identity:
        _fail()
    entry_identity = _write_or_recover_exact(
        entry_target, state.entries[-1].raw
    )
    if entry_identity != state.entries[-1].identity:
        _fail()
    validate_persisted_ledger_prefix(ledger_root, state)
    written_path = entry_target
    proof = object.__new__(DurableSealConsumption)
    nonce = os.urandom(32)
    for name, value in {
        "marker_identity": marker_identity,
        "ledger_head": state.head,
        "entry_index": state.entries[-1].entry_index,
        "marker_path": marker_target,
        "ledger_entry_path": written_path,
        "_nonce": nonce,
        "_origin": _DURABLE_SEAL_ORIGIN,
    }.items():
        object.__setattr__(proof, name, value)
    object.__setattr__(proof, "_auth", _durable_capability_auth(proof))
    _issue_durable_nonce(nonce)
    return proof


def recover_exact_file(
    path: Path | str,
    expected_raw: bytes,
) -> runtime.HashAndBytes:
    """Read-only recovery: accept only the exact intended existing bytes."""

    if type(expected_raw) is not bytes:
        _fail()
    expected = runtime.hash_and_bytes(expected_raw)
    raw = runtime.read_regular_file(
        path,
        expected=expected,
        maximum_bytes=expected.bytes,
    )
    if not hmac.compare_digest(raw, expected_raw):
        _fail()
    return expected


def _fsync_exact_file(path: Path | str, expected_raw: bytes) -> None:
    """Re-establish file and parent durability after an ambiguous write error."""

    target = Path(path)
    expected = runtime.hash_and_bytes(expected_raw)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_CLOEXEC", 0
    )
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    parent_fd = -1
    fd = -1
    try:
        parent_fd = os.open(target.parent, directory_flags)
        fd = os.open(target.name, file_flags, dir_fd=parent_fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected.bytes:
            _fail()
        chunks: list[bytes] = []
        remaining = expected.bytes + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if (
            len(raw) != expected.bytes
            or not hmac.compare_digest(raw, expected_raw)
        ):
            _fail()
        os.fsync(fd)
        os.fsync(parent_fd)
    except IntegrityFailure:
        raise
    except OSError:
        _fail()
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if parent_fd >= 0:
            try:
                os.close(parent_fd)
            except OSError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    # Live accrual and source/sealer orchestration are deliberately unavailable
    # from a standalone command in this atomic leaf.
    del argv
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
