"""Synthetic adversarial tests for the confirmatory-v4 accrual ledger.

All observations and artifact bytes are invented.  The suite reads only the
public frozen v4 plan/watermark and never opens either production source.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import os
import stat
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ap_confirmatory_accrual_v4.py"


@pytest.fixture(scope="module")
def accrual() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ap_confirmatory_accrual_v4_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _failure(module: Any, call: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    with pytest.raises(module.IntegrityFailure) as caught:
        call(*args, **kwargs)
    assert caught.value.args == ()
    assert str(caught.value) == ""


def _time(slot_index: int, seconds: float) -> str:
    base = datetime(2026, 8, 17, tzinfo=UTC)
    instant = base + timedelta(days=slot_index, seconds=seconds)
    return instant.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _identity(fill: str, size: int = 1) -> dict[str, Any]:
    return {"sha256": fill * 64, "bytes": size}


def _authority(alias: str) -> dict[str, Any]:
    if alias == "local":
        return {
            "transport": "local",
            "kernel_uid_uint32": 1000,
            "process_user_namespace_inode_uint64": 40_001,
        }
    fingerprint = base64.b64encode(bytes(range(32))).decode().rstrip("=")
    return {
        "transport": "ssh",
        "ssh_host_key_algorithm": "ssh-ed25519",
        "ssh_host_key_sha256_base64": fingerprint,
        "remote_kernel_uid_uint32": 1001,
        "remote_process_user_namespace_inode_uint64": 40_002,
    }


def _database(alias: str, generation: int = 0) -> dict[str, Any]:
    tail = 1 + generation * 2 + (alias == "alt")
    return {
        "filesystem_uuid": f"00000000-0000-4000-8000-{tail:012d}",
        "statx_inode_uint64": 70_000 + tail,
        "statx_birthtime_ns_int64": 1_700_000_000_000_000_000 + tail,
    }


def _binding_observation(
    services: Any, *, generation: int = 0
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, alias in enumerate(("local", "alt")):
        service, boot = services.service_pairs[index]
        result[alias] = {
            "alias_id": {
                "local": "confirmatory-local-v4-ro",
                "alt": "confirmatory-alt-v4-ro",
            }[alias],
            "authenticated_authority_identity": _authority(alias),
            "service_identity_sha256": service,
            "boot_identity_sha256": boot,
            "database_instance_identity": _database(alias, generation),
        }
    return result


def _bootstrap(
    accrual: Any, *, stop_after: str | None = None
) -> tuple[Any, Any]:
    state = accrual.initialize_ledger()
    contract = state.contract
    observer = accrual.runtime.hash_and_bytes(b"synthetic-runtime-observer-v4")
    probe_identity = accrual.probe._self_identity()
    slot = accrual.runtime.slot_times(0)
    attempt_value = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "probe-attempt",
        "slot_index": 0,
        "scheduled_at": slot.scheduled_at,
        "grace_deadline_at": slot.grace_deadline_at,
        "attempt_ordinal": 0,
        "launched_at": _time(0, 1),
        "watchdog_deadline_at": _time(0, 3601),
        "segment_id": contract.initial_segment_id,
        "previous_ledger_entry_sha256": state.head,
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "probe_sha256_and_bytes": probe_identity.as_dict(),
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
    }
    attempt = accrual.probe.validate_probe_attempt_marker(
        accrual.runtime.canonical_json_bytes(attempt_value),
        contract=contract,
        active_segment=None,
        expected_segment_id=contract.initial_segment_id,
        expected_runtime_observer_identity=observer,
        expected_previous_ledger_sha256=state.head,
        expected_slot_index=0,
        expected_attempt_ordinal=0,
        expected_probe_identity=probe_identity,
    )
    state = accrual.append_probe_attempt(state, attempt)
    if stop_after == "probe-attempt":
        return state, observer

    marker_value = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "runtime-attestation-attempt",
        "attempt_scope": "initial-source-binding",
        "segment_index": 0,
        "slot_index_or_null": 0,
        "predecessor_closure_sha256_or_null": None,
        "authorized_at": slot.scheduled_at,
        "written_at": _time(0, 1.1),
        "start_deadline_at": slot.grace_deadline_at,
        "previous_ledger_entry_sha256": state.head,
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
    }
    marker = accrual.runtime.validate_attestation_attempt_marker(
        accrual.runtime.canonical_json_bytes(marker_value),
        contract=contract,
        runtime_observer_identity=observer,
        expected_previous_ledger_sha256=state.head,
        initial_probe_launched_at=_time(0, 1),
    )
    state = accrual.append_runtime_attestation_attempt(state, marker)
    if stop_after == "runtime-attempt":
        return state, observer

    observation = _binding_observation(contract.initial_services)
    binding = accrual.runtime.validate_stable_source_binding_observations(
        contract.initial_services,
        observation,
        copy.deepcopy(observation),
    )
    binding_value = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "source-binding-attestation",
        "segment_index": 0,
        "attestation_attempt_marker_sha256_and_bytes": marker.identity.as_dict(),
        "alias_ids_by_alias": dict(accrual.runtime.ALIAS_IDS),
        "pre_database_instance_identity_sha256_by_alias": binding.core.database_map(),
        "post_database_instance_identity_sha256_by_alias": binding.core.database_map(),
        "pre_alias_service_database_binding_sha256_by_alias": binding.core.binding_map(),
        "post_alias_service_database_binding_sha256_by_alias": binding.core.binding_map(),
        "active_services_state_sha256": contract.initial_services.identity.sha256,
        "source_binding_core_sha256_and_bytes": binding.core.identity.as_dict(),
        "status": "pass",
        "pre_observed_at": _time(0, 1.2),
        "post_observed_at": _time(0, 1.3),
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
        "previous_ledger_entry_sha256": state.head,
    }
    binding_attestation = accrual.runtime.validate_source_binding_attestation(
        accrual.runtime.canonical_json_bytes(binding_value),
        contract=contract,
        attempt=marker,
        binding=binding,
        runtime_observer_identity=observer,
        expected_previous_ledger_sha256=state.head,
    )
    state = accrual.append_source_binding_attestation(state, binding_attestation)
    return state, observer


def _runtime_terminal_raw(
    accrual: Any,
    state: Any,
    *,
    reason: str,
    recorded_at: str = "2026-08-17T00:00:02.000000Z",
) -> bytes:
    attempt = state.runtime_attempt
    return accrual.runtime.canonical_json_bytes(
        {
            "schema_version": 4,
            "namespace": accrual.NAMESPACE,
            "receipt_kind": "runtime-attestation-terminal",
            "status": "terminal-runtime-attestation-failure",
            "attempt_scope": "initial-source-binding",
            "segment_index": 0,
            "attestation_attempt_marker_sha256_and_bytes_or_null": (
                None if attempt is None else attempt.identity.as_dict()
            ),
            "predecessor_closure_sha256_or_null": None,
            "recorded_at": recorded_at,
            "reason": reason,
            "previous_ledger_entry_sha256": state.head,
            "analysis_plan_sha256_and_bytes": (
                state.contract.plan_identity.as_dict()
            ),
        }
    )


def _successor_runtime_terminal_raw(
    accrual: Any,
    state: Any,
    *,
    reason: str,
    recorded_at: str,
) -> bytes:
    predecessor = state.closed_predecessor
    assert predecessor is not None and state.closure is not None
    attempt = state.runtime_attempt
    return accrual.runtime.canonical_json_bytes(
        {
            "schema_version": 4,
            "namespace": accrual.NAMESPACE,
            "receipt_kind": "runtime-attestation-terminal",
            "status": "terminal-runtime-attestation-failure",
            "attempt_scope": "successor-segment",
            "segment_index": predecessor.segment_index + 1,
            "attestation_attempt_marker_sha256_and_bytes_or_null": (
                None if attempt is None else attempt.identity.as_dict()
            ),
            "predecessor_closure_sha256_or_null": state.closure.identity.sha256,
            "recorded_at": recorded_at,
            "reason": reason,
            "previous_ledger_entry_sha256": state.head,
            "analysis_plan_sha256_and_bytes": (
                state.contract.plan_identity.as_dict()
            ),
        }
    )


def _append_attempt(
    accrual: Any,
    state: Any,
    *,
    ordinal: int = 0,
    launched_seconds: float | None = None,
) -> Any:
    assert state.active_segment is not None
    slot_index = state.next_slot_index
    slot = accrual.runtime.slot_times(slot_index)
    launch_offset = (
        1 + ordinal * 10
        if launched_seconds is None
        else launched_seconds
    )
    launched = _time(slot_index, launch_offset)
    watchdog = _time(slot_index, launch_offset + 3600)
    value = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "probe-attempt",
        "slot_index": slot_index,
        "scheduled_at": slot.scheduled_at,
        "grace_deadline_at": slot.grace_deadline_at,
        "attempt_ordinal": ordinal,
        "launched_at": launched,
        "watchdog_deadline_at": watchdog,
        "segment_id": state.active_segment.segment_id,
        "previous_ledger_entry_sha256": state.head,
        "runtime_observer_sha256_and_bytes": (
            state.active_segment.source_binding_attestation.runtime_observer_identity.as_dict()
        ),
        "probe_sha256_and_bytes": accrual.probe._self_identity().as_dict(),
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
    }
    attempt = accrual.probe.validate_probe_attempt_marker(
        accrual.runtime.canonical_json_bytes(value),
        contract=state.contract,
        active_segment=state.active_segment,
        expected_previous_ledger_sha256=state.head,
        expected_slot_index=slot_index,
        expected_attempt_ordinal=ordinal,
    )
    return accrual.append_probe_attempt(state, attempt)


def _resolution(accrual: Any, state: Any) -> Any:
    active = state.active_segment
    attempt = state.current_probe_attempt
    assert active is not None and attempt is not None
    slot = accrual.runtime.slot_times(state.next_slot_index)
    snapshots = {
        "local": _identity("a", 10 + state.next_slot_index),
        "alt": _identity("b", 20 + state.next_slot_index),
    }
    databases = active.source_binding.core.database_map()
    bindings = active.source_binding.core.binding_map()
    counts = {field: 0 for field in accrual.probe.AGGREGATE_FIELDS}
    value: dict[str, Any] = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "slot-probe-resolution",
        "status": "below-floor",
        "resolution_id": "",
        "slot_index": state.next_slot_index,
        "scheduled_at": slot.scheduled_at,
        "grace_deadline_at": slot.grace_deadline_at,
        "launched_at": accrual.runtime.canonical_utc(attempt.launched_at),
        "validated_at": _time(state.next_slot_index, 2 + attempt.attempt_ordinal * 10),
        "release_effective_at": state.contract.release_effective_at,
        "active_segment_lower_bound_exclusive_at": active.lower_bound_exclusive_at,
        "segment_id": active.segment_id,
        "segment_attestation_sha256_and_bytes": active.attestation_identity.as_dict(),
        "source_binding_attestation_sha256_and_bytes": (
            active.source_binding_attestation.identity.as_dict()
        ),
        "previous_ledger_entry_sha256": state.head,
        "pre_active_services_state_sha256": active.services.identity.sha256,
        "post_active_services_state_sha256": active.services.identity.sha256,
        "pre_database_instance_identity_sha256_by_alias": databases,
        "post_database_instance_identity_sha256_by_alias": dict(databases),
        "pre_alias_service_database_binding_sha256_by_alias": bindings,
        "post_alias_service_database_binding_sha256_by_alias": dict(bindings),
        "aliased_source_snapshot_sha256_and_bytes": snapshots,
        **counts,
        "runtime_observer_sha256_and_bytes": (
            active.source_binding_attestation.runtime_observer_identity.as_dict()
        ),
        "probe_sha256_and_bytes": attempt.probe_identity.as_dict(),
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
        "seal_consumption_marker_sha256_and_bytes_or_null": None,
        "sealer_process_launched_at_or_null": None,
    }
    value["resolution_id"] = accrual.runtime.derive_slot_resolution_id(value)
    raw = accrual.runtime.canonical_json_bytes(value)
    return accrual.probe.validate_probe_resolution(
        raw,
        contract=state.contract,
        active_segment=active,
        expected_previous_ledger_sha256=state.head,
        expected_probe_identity=attempt.probe_identity,
    )


def _probe_failure(accrual: Any, state: Any, phase: str) -> Any:
    attempt = state.current_probe_attempt
    assert attempt is not None and state.active_segment is not None
    if phase == "unknown-after-attempt":
        failed_at = accrual.runtime.canonical_utc(attempt.watchdog_deadline_at)
        source_count = None
        key_created = None
        controller = True
        failure_class = "post-source-terminal"
        retry = False
    else:
        source_count, key_created = accrual.probe.PHASE_PROGRESS[phase]
        failed_at = accrual.runtime.canonical_utc(
            attempt.launched_at + timedelta(seconds=1)
        )
        controller = False
        retry = phase == "pre-key-pre-source"
        failure_class = (
            "pre-source-retryable" if retry else "post-source-terminal"
        )
    value = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "probe-failure",
        "slot_index": attempt.slot_index,
        "attempt_ordinal": attempt.attempt_ordinal,
        "probe_attempt_marker_sha256_and_bytes": attempt.identity.as_dict(),
        "failed_at": failed_at,
        "phase_at_failure": phase,
        "failure_class": failure_class,
        "source_open_count_or_null": source_count,
        "key_created_or_null": key_created,
        "retry_authorized": retry,
        "controller_synthesized": controller,
        "previous_ledger_entry_sha256": state.head,
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
    }
    return accrual.probe.validate_probe_failure_marker(
        accrual.runtime.canonical_json_bytes(value),
        contract=state.contract,
        active_segment=state.active_segment,
        attempt=attempt,
        expected_previous_ledger_sha256=state.head,
    )


def _ready_computation(accrual: Any, state: Any) -> Any:
    active = state.active_segment
    assert active is not None
    counts = {
        "selected_event_count": 450,
        "selected_replayable_event_count": 450,
        "selected_nonreplayable_event_count": 0,
        "holdout_unseen_automatic_family_count": 30,
        "holdout_unseen_automatic_event_count": 150,
        "holdout_unseen_automatic_component_count": 30,
        "holdout_organic_event_count": 200,
        "holdout_organic_session_count": 30,
        "holdout_organic_component_count": 30,
        "holdout_project_scope_count": 2,
        "shadow_real_workflow_count": 30,
        "shadow_replayable_logical_call_count": 100,
        "shadow_real_workflow_component_count": 30,
        "shadow_project_scope_count": 2,
    }
    snapshots = {
        "local": accrual.runtime.HashAndBytes("c" * 64, 101),
        "alt": accrual.runtime.HashAndBytes("d" * 64, 202),
    }
    slot = accrual.runtime.slot_times(state.next_slot_index)
    envelope = {
        "slot_index": state.next_slot_index,
        "scheduled_at": slot.scheduled_at,
        "grace_deadline_at": slot.grace_deadline_at,
        "active_segment_lower_bound_exclusive_at": active.lower_bound_exclusive_at,
        "aggregate": counts,
        "aliased_source_snapshot_sha256_and_bytes": {
            alias: snapshots[alias].as_dict()
            for alias in accrual.runtime.SOURCE_ALIASES
        },
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
    }
    return accrual.probe._make_probe_computation(
        slot_index=state.next_slot_index,
        scheduled_at=slot.scheduled_at,
        grace_deadline_at=slot.grace_deadline_at,
        active_segment_lower_bound_exclusive_at=active.lower_bound_exclusive_at,
        aggregate_items=tuple((field, counts[field]) for field in accrual.probe.AGGREGATE_FIELDS),
        snapshot_items=tuple(
            (alias, snapshots[alias]) for alias in accrual.runtime.SOURCE_ALIASES
        ),
        analysis_plan_identity=state.contract.plan_identity,
        segment_id=active.segment_id,
        segment_attestation_identity=active.attestation_identity,
        source_binding_attestation_identity=active.source_binding_attestation.identity,
        source_database_identity_items=tuple(
            (alias, active.source_binding.core.database_map()[alias])
            for alias in accrual.runtime.SOURCE_ALIASES
        ),
        worker_envelope_raw=accrual.runtime.canonical_json_bytes(envelope),
        probe_identity=accrual.probe._self_identity(),
    )


def _bound_source_only_closure(accrual: Any, state: Any) -> tuple[Any, Any]:
    active = state.active_segment
    attempt = state.current_probe_attempt
    assert active is not None and attempt is not None
    observed = _binding_observation(active.services, generation=1)
    observed_binding = accrual.runtime.validate_stable_source_binding_observations(
        active.services, observed, copy.deepcopy(observed)
    )
    slot = accrual.runtime.slot_times(state.next_slot_index)
    value = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "slot-segment-closed",
        "status": "segment-closed",
        "resolution_id": "",
        "slot_index": state.next_slot_index,
        "scheduled_at": slot.scheduled_at,
        "grace_deadline_at": slot.grace_deadline_at,
        "launched_at": accrual.runtime.canonical_utc(attempt.launched_at),
        "validated_at": _time(state.next_slot_index, 2),
        "segment_index": active.segment_index,
        "segment_id": active.segment_id,
        "mismatch_kinds": ["source-binding-change"],
        "prior_active_services_state_sha256_and_bytes": active.services.identity.as_dict(),
        "observed_active_services_state_sha256_and_bytes": active.services.identity.as_dict(),
        "prior_source_binding_core_sha256_and_bytes": active.source_binding.core.identity.as_dict(),
        "observed_source_binding_core_sha256_and_bytes": observed_binding.core.identity.as_dict(),
        "attestation_sha256_and_bytes": active.attestation_identity.as_dict(),
        "previous_ledger_entry_sha256": state.head,
        "runtime_observer_sha256_and_bytes": (
            active.source_binding_attestation.runtime_observer_identity.as_dict()
        ),
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
    }
    value["resolution_id"] = accrual.runtime.derive_slot_resolution_id(value)
    raw = accrual.runtime.canonical_json_bytes(value)
    closure = accrual.runtime.validate_segment_closure(
        raw,
        contract=state.contract,
        active_segment=active,
        prior_services_raw=active.services.raw,
        observed_services_raw=active.services.raw,
        prior_source_binding_core_raw=active.source_binding.core.raw,
        observed_source_binding_core_raw=observed_binding.core.raw,
        observed_binding=observed_binding,
        runtime_observer_identity=(
            active.source_binding_attestation.runtime_observer_identity
        ),
        expected_previous_ledger_sha256=state.head,
    )
    bound = accrual.bind_segment_closure(
        state,
        closure,
        prior_services_raw=active.services.raw,
        observed_services_raw=active.services.raw,
        prior_source_binding_core_raw=active.source_binding.core.raw,
        observed_source_binding_core_raw=observed_binding.core.raw,
        observed_binding=observed_binding,
        runtime_observer_identity=(
            active.source_binding_attestation.runtime_observer_identity
        ),
    )
    return bound, observed_binding


def test_frozen_genesis_and_entry_zero_are_exact(accrual: Any) -> None:
    state = accrual.initialize_ledger()
    assert (
        accrual.derive_ledger_genesis(state.contract)
        == "51f49b68a5876ac08b8f01f5f7da714dd270a97fa5ad9db822fc8971d0b33378"
    )
    assert state.head == "028560e367bf008900bb3df429c9c59c917cd8b91c893ae617d8890a23d46fbb"
    # Cross-protocol replay safety: the same frozen plan under a retired ledger
    # domain yields a different genesis, so no v3 chain can be replayed here.
    assert accrual.LEDGER_DOMAIN == b"confirmatory-holdout-v4/accrual-ledger/v1"
    assert (
        hashlib.sha256(
            b"confirmatory-holdout-v3/accrual-ledger/v1"
            + b"\0"
            + state.contract.plan_identity.sha256.encode("ascii")
        ).hexdigest()
        != accrual.derive_ledger_genesis(state.contract)
    )
    entry = json.loads(state.entries[0].raw)
    assert tuple(entry) != accrual.LEDGER_ENTRY_FIELDS  # canonical sorting is expected
    assert entry["entry_index"] == 0
    assert entry["entry_kind"] == "segment-attestation"
    assert entry["slot_index_or_null"] is None
    assert state.entries[0].artifact_identity == state.contract.watermark_identity

    forged = replace(
        state.contract,
        plan_identity=accrual.runtime.HashAndBytes("0" * 64, 1),
    )
    _failure(accrual, accrual.derive_ledger_genesis, forged)


def test_chain_rejects_index_predecessor_kind_and_cached_wrapper_tampering(
    accrual: Any,
) -> None:
    state, _ = _bootstrap(accrual)
    record = state.entries[-1]
    value = json.loads(record.raw)
    for field, replacement in (
        ("entry_index", value["entry_index"] + 1),
        ("previous_ledger_entry_sha256", "0" * 64),
        ("entry_kind", "below-floor"),
        ("slot_index_or_null", None),
    ):
        changed = dict(value)
        changed[field] = replacement
        changed["entry_sha256"] = accrual.derive_ledger_entry_sha256(changed)
        _failure(
            accrual,
            accrual.validate_ledger_entry,
            accrual.runtime.canonical_json_bytes(changed),
            artifact_raw=record.artifact_raw,
            contract=state.contract,
            expected_entry_index=record.entry_index,
            expected_previous_ledger_sha256=record.previous_ledger_entry_sha256,
        )
    _failure(
        accrual,
        accrual._require_state,
        replace(
            state,
            entries=state.entries[:-1]
            + (replace(record, artifact_raw=b"not-the-attestation"),),
        ),
    )
    _failure(
        accrual,
        accrual._require_state,
        replace(
            state,
            phase=accrual.PHASE_AWAITING_PROBE,
            next_slot_index=28,
            current_probe_attempt=None,
        ),
    )
    altered_attempt = replace(
        state.current_probe_attempt,
        probe_identity=accrual.runtime.HashAndBytes("f" * 64, 999),
        launched_at=state.current_probe_attempt.launched_at
        + timedelta(microseconds=1),
    )
    _failure(
        accrual,
        accrual._require_state,
        replace(state, current_probe_attempt=altered_attempt),
    )

    original = state.active_segment.source_binding_attestation
    observation = _binding_observation(
        state.contract.initial_services, generation=9
    )
    alternate_binding = (
        accrual.runtime.validate_stable_source_binding_observations(
            state.contract.initial_services,
            observation,
            copy.deepcopy(observation),
        )
    )
    alternate_value = json.loads(original.raw)
    alternate_value.update(
        {
            "pre_database_instance_identity_sha256_by_alias": (
                alternate_binding.core.database_map()
            ),
            "post_database_instance_identity_sha256_by_alias": (
                alternate_binding.core.database_map()
            ),
            "pre_alias_service_database_binding_sha256_by_alias": (
                alternate_binding.core.binding_map()
            ),
            "post_alias_service_database_binding_sha256_by_alias": (
                alternate_binding.core.binding_map()
            ),
            "source_binding_core_sha256_and_bytes": (
                alternate_binding.core.identity.as_dict()
            ),
        }
    )
    alternate_attestation = (
        accrual.runtime.validate_source_binding_attestation(
            accrual.runtime.canonical_json_bytes(alternate_value),
            contract=state.contract,
            attempt=original.attempt,
            binding=alternate_binding,
            runtime_observer_identity=original.runtime_observer_identity,
            expected_previous_ledger_sha256=(
                original.previous_ledger_entry_sha256
            ),
        )
    )
    alternate_active = accrual.runtime.make_initial_segment(
        state.contract, alternate_attestation
    )
    _failure(
        accrual,
        accrual._require_state,
        replace(state, active_segment=alternate_active),
    )


def test_exact_bootstrap_order_and_first_resolution_immutability(accrual: Any) -> None:
    genesis = accrual.initialize_ledger()
    _failure(accrual, accrual.append_below_floor_resolution, genesis, object())
    state, _ = _bootstrap(accrual)
    assert [entry.entry_kind for entry in state.entries] == [
        "segment-attestation",
        "probe-attempt",
        "runtime-attestation-attempt",
        "source-binding-attestation",
    ]
    fake_value = json.loads(state.current_probe_attempt.raw)
    fake_probe = accrual.runtime.HashAndBytes("e" * 64, 123)
    fake_value["probe_sha256_and_bytes"] = fake_probe.as_dict()
    fake_attempt = accrual.probe.validate_probe_attempt_marker(
        accrual.runtime.canonical_json_bytes(fake_value),
        contract=genesis.contract,
        active_segment=None,
        expected_segment_id=genesis.contract.initial_segment_id,
        expected_runtime_observer_identity=(
            state.current_probe_attempt.runtime_observer_identity
        ),
        expected_previous_ledger_sha256=genesis.head,
        expected_slot_index=0,
        expected_attempt_ordinal=0,
        expected_probe_identity=fake_probe,
    )
    _failure(accrual, accrual.append_probe_attempt, genesis, fake_attempt)
    resolution = _resolution(accrual, state)
    mutated_receipt = replace(
        resolution, receipt=dict(resolution.receipt, status="ready")
    )
    _failure(
        accrual,
        accrual.append_below_floor_resolution,
        state,
        mutated_receipt,
    )
    resolved = accrual.append_below_floor_resolution(state, resolution)
    assert resolved.phase == accrual.PHASE_AWAITING_PROBE
    assert resolved.next_slot_index == 1
    assert resolved.source_authority is False
    invalid_artifact = json.loads(resolution.raw)
    invalid_artifact["analysis_plan_sha256_and_bytes"] = _identity("0", 1)
    invalid_artifact["resolution_id"] = (
        accrual.runtime.derive_slot_resolution_id(invalid_artifact)
    )
    invalid_raw = accrual.runtime.canonical_json_bytes(invalid_artifact)
    shallow_entry = accrual._build_entry(
        contract=state.contract,
        entry_index=state.next_entry_index,
        previous=state.head,
        entry_kind="below-floor",
        slot_index=0,
        artifact_raw=invalid_raw,
    )
    forged_entries = state.entries + (shallow_entry,)
    _failure(
        accrual,
        accrual._require_state,
        replace(
            resolved,
            entries=forged_entries,
            _history_auth=accrual._history_auth(forged_entries),
        ),
    )
    _failure(accrual, accrual.append_below_floor_resolution, resolved, resolution)
    _failure(accrual, accrual.append_probe_attempt, state, state.current_probe_attempt)


def test_only_proven_pre_key_pre_source_failure_authorizes_same_slot_retry(
    accrual: Any,
) -> None:
    state, _ = _bootstrap(accrual)
    failure = _probe_failure(accrual, state, "pre-key-pre-source")
    state = accrual.append_probe_failure(state, failure)
    assert state.phase == accrual.PHASE_RETRY_ALLOWED
    assert state.source_authority is False
    _failure(accrual, _append_attempt, accrual, state, ordinal=2)
    _failure(
        accrual,
        _append_attempt,
        accrual,
        state,
        ordinal=1,
        launched_seconds=1.5,
    )
    retried = _append_attempt(accrual, state, ordinal=1)
    assert retried.current_probe_attempt.attempt_ordinal == 1
    assert retried.source_authority is True
    _failure(
        accrual,
        accrual.append_probe_attempt,
        retried,
        retried.current_probe_attempt,
    )


@pytest.mark.parametrize(
    "phase",
    [
        "key-created-pre-source",
        "first-source-opened",
        "snapshots-latched",
        "keyed-count-computed",
        "atomic-seal-handoff",
        "unknown-after-attempt",
    ],
)
def test_every_nonretryable_or_unknown_failure_requires_terminal(
    accrual: Any, phase: str
) -> None:
    state, _ = _bootstrap(accrual)
    failure = _probe_failure(accrual, state, phase)
    state = accrual.append_probe_failure(state, failure)
    assert state.phase == accrual.PHASE_PROBE_TERMINAL
    _failure(accrual, _append_attempt, accrual, state, ordinal=1)
    value = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "probe-terminal",
        "status": "terminal-probe-integrity-failure",
        "slot_index": failure.attempt.slot_index,
        "attempt_ordinal": failure.attempt.attempt_ordinal,
        "probe_failure_marker_sha256_and_bytes": failure.identity.as_dict(),
        "recorded_at": accrual.runtime.canonical_utc(failure.failed_at),
        "previous_ledger_entry_sha256": state.head,
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
    }
    terminal = accrual.append_probe_terminal(
        state, accrual.runtime.canonical_json_bytes(value)
    )
    assert terminal.terminal_status == "terminal-probe-integrity-failure"
    _failure(accrual, accrual.append_missed_slot, terminal, b"{}")


def test_runtime_terminal_reasons_match_marker_progress_and_clear_authority(
    accrual: Any,
) -> None:
    before_marker, _ = _bootstrap(accrual, stop_after="probe-attempt")
    assert before_marker.phase == accrual.PHASE_INITIAL_RUNTIME_ATTEMPT
    terminal = accrual.append_runtime_attestation_terminal(
        before_marker,
        _runtime_terminal_raw(
            accrual,
            before_marker,
            reason="attempt-marker-create-failure",
        ),
    )
    assert terminal.terminal_status == "terminal-runtime-attestation-failure"
    assert terminal.current_probe_attempt is None
    assert terminal.runtime_attempt is None
    _failure(
        accrual,
        accrual.append_runtime_attestation_terminal,
        before_marker,
        _runtime_terminal_raw(
            accrual,
            before_marker,
            reason="observation-unavailable",
        ),
    )

    after_marker, _ = _bootstrap(accrual, stop_after="runtime-attempt")
    assert after_marker.phase == accrual.PHASE_INITIAL_SOURCE_BINDING
    for impossible_reason in (
        "attempt-marker-create-failure",
        "attempt-marker-collision",
        "attempt-start-late",
        "boundary-invalid",
        "next-slot-deadline",
    ):
        _failure(
            accrual,
            accrual.append_runtime_attestation_terminal,
            after_marker,
            _runtime_terminal_raw(
                accrual, after_marker, reason=impossible_reason
            ),
        )
    terminal = accrual.append_runtime_attestation_terminal(
        after_marker,
        _runtime_terminal_raw(
            accrual, after_marker, reason="observation-unavailable"
        ),
    )
    assert terminal.terminal_status == "terminal-runtime-attestation-failure"
    assert terminal.current_probe_attempt is None
    assert terminal.runtime_attempt is None
    assert terminal.source_authority is False


def test_missed_is_after_grace_source_free_and_distinct_from_insufficient(
    accrual: Any,
) -> None:
    state = accrual.initialize_ledger()
    slot = accrual.runtime.slot_times(0)
    value = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "missed-slot",
        "status": "terminal-schedule-integrity-failure",
        "slot_index": 0,
        "scheduled_at": slot.scheduled_at,
        "grace_deadline_at": slot.grace_deadline_at,
        "recorded_at": slot.grace_deadline_at,
        "source_open_count": 0,
        "previous_ledger_entry_sha256": state.head,
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
    }
    early = dict(value, recorded_at=_time(0, 21_599.999999))
    _failure(
        accrual,
        accrual.append_missed_slot,
        state,
        accrual.runtime.canonical_json_bytes(early),
    )
    terminal = accrual.append_missed_slot(
        state, accrual.runtime.canonical_json_bytes(value)
    )
    assert terminal.terminal_status == "terminal-schedule-integrity-failure"
    assert terminal.final_resolution_status is None

    retry_state, _ = _bootstrap(accrual)
    retry_state = accrual.append_probe_failure(
        retry_state,
        _probe_failure(accrual, retry_state, "pre-key-pre-source"),
    )
    retry_value = dict(
        value, previous_ledger_entry_sha256=retry_state.head
    )
    retry_terminal = accrual.append_missed_slot(
        retry_state,
        accrual.runtime.canonical_json_bytes(retry_value),
    )
    assert retry_terminal.current_probe_attempt is None
    assert retry_terminal.probe_failure is None
    assert (
        retry_terminal.terminal_status
        == "terminal-schedule-integrity-failure"
    )


def test_count_free_closure_erases_evidence_and_requires_successor(
    accrual: Any,
) -> None:
    state, _ = _bootstrap(accrual)
    bound, _ = _bound_source_only_closure(accrual, state)
    forged_capability = object.__new__(accrual.BoundSegmentClosure)
    for field, value in {
        "closure": bound.closure,
        "active_segment": state.active_segment,
        "ledger_head": state.head,
        "_origin": accrual._CLOSURE_BINDING_ORIGIN,
    }.items():
        object.__setattr__(forged_capability, field, value)
    _failure(
        accrual,
        accrual.append_segment_closure,
        state,
        forged_capability,
    )
    closed = accrual.append_segment_closure(state, bound)
    assert closed.phase == accrual.PHASE_SUCCESSOR_ATTEMPT
    assert closed.next_slot_index == 1
    assert closed.active_segment is None
    assert closed.current_probe_attempt is None
    assert closed.provisional_ready is None
    assert closed.source_authority is False
    forbidden = {
        *accrual.probe.AGGREGATE_FIELDS,
        "aliased_source_snapshot_sha256_and_bytes",
    }
    assert forbidden.isdisjoint(json.loads(closed.entries[-1].artifact_raw))
    _failure(
        accrual,
        accrual.append_probe_attempt,
        closed,
        state.current_probe_attempt,
    )
    _failure(accrual, accrual.append_segment_closure, state, bound.closure)


def test_closure_requires_one_successor_chain_before_the_next_fixed_slot(
    accrual: Any,
) -> None:
    state, _ = _bootstrap(accrual)
    bound, observed_binding = _bound_source_only_closure(accrual, state)
    state = accrual.append_segment_closure(state, bound)
    closure = state.closure
    predecessor = state.closed_segment
    assert closure is not None and predecessor is not None
    observer = predecessor.source_binding_attestation.runtime_observer_identity

    _failure(
        accrual,
        accrual.append_runtime_attestation_terminal,
        state,
        _successor_runtime_terminal_raw(
            accrual,
            state,
            reason="attempt-start-late",
            recorded_at=_time(0, 31.999999),
        ),
    )
    late_terminal = accrual.append_runtime_attestation_terminal(
        state,
        _successor_runtime_terminal_raw(
            accrual,
            state,
            reason="attempt-start-late",
            recorded_at=_time(0, 32),
        ),
    )
    assert late_terminal.terminal_status == "terminal-runtime-attestation-failure"
    assert late_terminal.closure is None
    assert late_terminal.closed_segment is None

    attempt_value = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "runtime-attestation-attempt",
        "attempt_scope": "successor-segment",
        "segment_index": 1,
        "slot_index_or_null": None,
        "predecessor_closure_sha256_or_null": closure.identity.sha256,
        "authorized_at": _time(0, 2),
        "written_at": _time(0, 2.1),
        "start_deadline_at": _time(0, 32),
        "previous_ledger_entry_sha256": state.head,
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
    }
    attempt = accrual.runtime.validate_attestation_attempt_marker(
        accrual.runtime.canonical_json_bytes(attempt_value),
        contract=state.contract,
        runtime_observer_identity=observer,
        expected_previous_ledger_sha256=state.head,
        predecessor_closure=closure,
    )
    state = accrual.append_runtime_attestation_attempt(state, attempt)
    _failure(accrual, accrual.append_runtime_attestation_attempt, state, attempt)
    ceremony_failure = accrual.append_runtime_attestation_terminal(
        state,
        _successor_runtime_terminal_raw(
            accrual,
            state,
            reason="observation-unavailable",
            recorded_at=_time(0, 33),
        ),
    )
    assert (
        ceremony_failure.terminal_status
        == "terminal-runtime-attestation-failure"
    )
    next_due = accrual.runtime.slot_times(1).scheduled_at
    deadline_terminal = accrual.append_runtime_attestation_terminal(
        state,
        _successor_runtime_terminal_raw(
            accrual,
            state,
            reason="next-slot-deadline",
            recorded_at=next_due,
        ),
    )
    assert (
        deadline_terminal.terminal_status
        == "terminal-runtime-attestation-failure"
    )
    _failure(
        accrual,
        accrual.append_runtime_attestation_terminal,
        state,
        _successor_runtime_terminal_raw(
            accrual,
            state,
            reason="observation-unavailable",
            recorded_at=next_due,
        ),
    )

    binding_value = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "source-binding-attestation",
        "segment_index": 1,
        "attestation_attempt_marker_sha256_and_bytes": attempt.identity.as_dict(),
        "alias_ids_by_alias": dict(accrual.runtime.ALIAS_IDS),
        "pre_database_instance_identity_sha256_by_alias": observed_binding.core.database_map(),
        "post_database_instance_identity_sha256_by_alias": observed_binding.core.database_map(),
        "pre_alias_service_database_binding_sha256_by_alias": observed_binding.core.binding_map(),
        "post_alias_service_database_binding_sha256_by_alias": observed_binding.core.binding_map(),
        "active_services_state_sha256": predecessor.services.identity.sha256,
        "source_binding_core_sha256_and_bytes": observed_binding.core.identity.as_dict(),
        "status": "pass",
        "pre_observed_at": _time(0, 3),
        "post_observed_at": _time(0, 3.7),
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
        "previous_ledger_entry_sha256": state.head,
    }
    binding = accrual.runtime.validate_source_binding_attestation(
        accrual.runtime.canonical_json_bytes(binding_value),
        contract=state.contract,
        attempt=attempt,
        binding=observed_binding,
        runtime_observer_identity=observer,
        expected_previous_ledger_sha256=state.head,
    )
    state = accrual.append_source_binding_attestation(state, binding)

    segment_value = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "runtime-segment-attestation",
        "status": "pass",
        "segment_index": 1,
        "segment_id": "",
        "attestation_attempt_marker_sha256_and_bytes": attempt.identity.as_dict(),
        "predecessor_closure_sha256": closure.identity.sha256,
        "lower_bound_exclusive_at": _time(0, 3.2),
        "pre_launcher_at": _time(0, 3),
        "pre_source_clock_observed_at_by_alias": {
            "local": _time(0, 3.05),
            "alt": _time(0, 3.2),
        },
        "boundary_launcher_at": _time(0, 3.1),
        "boundary_at": _time(0, 3.2),
        "post_phase_started_at": _time(0, 3.45),
        "post_source_clock_observed_at_by_alias": {
            "local": _time(0, 3.5),
            "alt": _time(0, 3.6),
        },
        "post_launcher_at": _time(0, 3.7),
        "complete_unaliased_service_tuple_sha256_and_bytes": predecessor.services.identity.as_dict(),
        "source_binding_attestation_sha256_and_bytes": binding.identity.as_dict(),
        "active_services_state_sha256": predecessor.services.identity.sha256,
        "replay_code_control_commit_and_tree": {
            "commit": accrual.runtime.REPLAY_CODE_CONTROL_COMMIT,
            "tree": accrual.runtime.REPLAY_CODE_CONTROL_TREE,
        },
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
        "previous_ledger_entry_sha256": state.head,
    }
    segment_value["segment_id"] = accrual.runtime.derive_successor_segment_id(
        segment_value
    )
    segment_raw = accrual.runtime.canonical_json_bytes(segment_value)
    services_value = accrual.runtime.load_json_bytes(
        predecessor.services.raw, expected=list
    )
    successor = accrual.runtime.validate_successor_segment(
        segment_raw,
        contract=state.contract,
        predecessor=predecessor,
        predecessor_closure=closure,
        attempt=attempt,
        source_binding_attestation=binding,
        pre_services_observation=services_value,
        post_services_observation=list(reversed(copy.deepcopy(services_value))),
        complete_services_raw=predecessor.services.raw,
        expected_previous_ledger_sha256=state.head,
    )
    state = accrual.append_successor_segment(state, successor)
    assert state.phase == accrual.PHASE_AWAITING_PROBE
    assert state.active_segment.segment_index == 1
    assert state.next_slot_index == 1
    state = _append_attempt(accrual, state)
    assert state.current_probe_attempt.slot_index == 1


def test_ready_requires_marker_then_typed_blocked_spawn_and_latches_first_pass(
    accrual: Any,
    tmp_path: Path,
) -> None:
    state, _ = _bootstrap(accrual)
    computation = _ready_computation(accrual, state)
    provisional = accrual.prepare_provisional_ready(
        state,
        computation,
        launched_at=_time(0, 1),
        validated_at=_time(0, 5),
    )
    forged_core = dict(provisional.core)
    forged_core["selected_event_count"] += 777
    forged_core["selected_nonreplayable_event_count"] += 777
    forged_provisional = replace(
        provisional,
        core=forged_core,
        core_identity=accrual._derive_ready_core_identity(forged_core),
    )
    sealer = accrual.runtime.hash_and_bytes(b"synthetic-sealer")
    marker_value = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "seal-consumption",
        "seal_authority_id": accrual.SEAL_AUTHORITY_ID,
        "slot_index": 0,
        "segment_id": state.active_segment.segment_id,
        "provisional_ready_core_sha256_and_bytes": provisional.core_identity.as_dict(),
        "snapshot_set_sha256_and_bytes": provisional.snapshot_set_identity.as_dict(),
        "consumed_at": _time(0, 3),
        "previous_ledger_entry_sha256": state.head,
        "sealer_sha256_and_bytes": sealer.as_dict(),
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
    }
    deadline_provisional = accrual.prepare_provisional_ready(
        state,
        computation,
        launched_at=_time(0, 1),
        validated_at=_time(0, 33),
    )
    deadline_marker_value = dict(
        marker_value,
        provisional_ready_core_sha256_and_bytes=(
            deadline_provisional.core_identity.as_dict()
        ),
        snapshot_set_sha256_and_bytes=(
            deadline_provisional.snapshot_set_identity.as_dict()
        ),
    )
    accrual.validate_seal_consumption_marker(
        accrual.runtime.canonical_json_bytes(deadline_marker_value),
        state=state,
        provisional=deadline_provisional,
        expected_sealer_identity=sealer,
    )
    late_provisional = accrual.prepare_provisional_ready(
        state,
        computation,
        launched_at=_time(0, 1),
        validated_at=_time(0, 33.000001),
    )
    late_marker_value = dict(
        marker_value,
        provisional_ready_core_sha256_and_bytes=(
            late_provisional.core_identity.as_dict()
        ),
        snapshot_set_sha256_and_bytes=(
            late_provisional.snapshot_set_identity.as_dict()
        ),
    )
    _failure(
        accrual,
        accrual.validate_seal_consumption_marker,
        accrual.runtime.canonical_json_bytes(late_marker_value),
        state=state,
        provisional=late_provisional,
        expected_sealer_identity=sealer,
    )
    marker = accrual.validate_seal_consumption_marker(
        accrual.runtime.canonical_json_bytes(marker_value),
        state=state,
        provisional=provisional,
        expected_sealer_identity=sealer,
    )
    forged_marker_value = dict(marker_value)
    forged_marker_value["provisional_ready_core_sha256_and_bytes"] = (
        forged_provisional.core_identity.as_dict()
    )
    _failure(
        accrual,
        accrual.validate_seal_consumption_marker,
        accrual.runtime.canonical_json_bytes(forged_marker_value),
        state=state,
        provisional=forged_provisional,
        expected_sealer_identity=sealer,
    )
    consumed = accrual.append_seal_consumption(state, marker)
    assert consumed.phase == accrual.PHASE_SEAL_CONSUMED
    assert consumed.active_segment is None
    assert consumed.current_probe_attempt is None
    altered_provisional = replace(
        consumed.provisional_ready,
        snapshot_set_identity=accrual.runtime.HashAndBytes("f" * 64, 999),
    )
    _failure(
        accrual,
        accrual._require_state,
        replace(consumed, provisional_ready=altered_provisional),
    )
    _failure(accrual, accrual.append_probe_attempt, consumed, state.current_probe_attempt)
    _failure(accrual, accrual.BlockedSpawnProof)
    _failure(accrual, accrual.DurableSealConsumption)

    divergent_value = dict(marker_value, consumed_at=_time(0, 3.5))
    divergent_marker = accrual.validate_seal_consumption_marker(
        accrual.runtime.canonical_json_bytes(divergent_value),
        state=state,
        provisional=provisional,
        expected_sealer_identity=sealer,
    )
    divergent_consumed = accrual.append_seal_consumption(
        state, divergent_marker
    )
    collision_ledger = tmp_path / "collision-ledger"
    collision_markers = tmp_path / "collision-markers"
    collision_ledger.mkdir()
    collision_markers.mkdir()
    for entry in consumed.entries[:-1]:
        accrual.write_exclusive(
            collision_ledger
            / accrual.ledger_entry_filename(entry.entry_index),
            entry.raw,
        )
    authority_path = collision_markers / accrual.seal_consumption_filename(
        marker, consumed.entries[-1].entry_index
    )
    assert authority_path.name == accrual.seal_consumption_filename(
        divergent_marker, divergent_consumed.entries[-1].entry_index
    )
    accrual.write_exclusive(authority_path, marker.raw)
    _failure(
        accrual,
        accrual.persist_seal_consumption_exclusive,
        divergent_consumed,
        marker_path=authority_path,
        ledger_directory=collision_ledger,
    )
    assert authority_path.read_bytes() == marker.raw
    assert not (
        collision_ledger
        / accrual.ledger_entry_filename(
            divergent_consumed.entries[-1].entry_index
        )
    ).exists()
    forged_spawn = object.__new__(accrual.BlockedSpawnProof)
    for field, value in {
        "marker_identity": marker.identity,
        "sealer_identity": marker.sealer_identity,
        "slot_index": marker.slot_index,
        "segment_id": marker.segment_id,
        "launched_at": accrual.runtime.parse_utc(_time(0, 4), receipt=True),
        "ledger_head": consumed.head,
        "_origin": accrual._BLOCKED_SPAWN_ORIGIN,
    }.items():
        object.__setattr__(forged_spawn, field, value)
    _failure(
        accrual,
        accrual._require_state,
        replace(
            consumed,
            phase=accrual.PHASE_BLOCKED_SPAWN,
            blocked_spawn=forged_spawn,
        ),
    )
    ledger_directory = tmp_path / "ledger"
    marker_directory = tmp_path / "markers"
    ledger_directory.mkdir()
    marker_directory.mkdir()
    for entry in consumed.entries[:-1]:
        accrual.write_exclusive(
            ledger_directory / accrual.ledger_entry_filename(entry.entry_index),
            entry.raw,
        )
    durable = accrual.persist_seal_consumption_exclusive(
        consumed,
        marker_path=marker_directory
        / accrual.seal_consumption_filename(
            marker, consumed.entries[-1].entry_index
        ),
        ledger_directory=ledger_directory,
    )
    _failure(
        accrual,
        accrual.persist_seal_consumption_exclusive,
        consumed,
        marker_path=marker_directory
        / accrual.seal_consumption_filename(
            marker, consumed.entries[-1].entry_index
        ),
        ledger_directory=ledger_directory,
    )
    accrual.validate_persisted_ledger_prefix(ledger_directory, consumed)
    spawned = accrual.record_blocked_spawn(
        consumed,
        marker=marker,
        durable_marker=durable,
        sealer_process_launched_at=_time(0, 4),
    )
    cloned_durable = object.__new__(accrual.DurableSealConsumption)
    for field in accrual.DurableSealConsumption.__dataclass_fields__:
        object.__setattr__(cloned_durable, field, getattr(durable, field))
    _failure(
        accrual,
        accrual.record_blocked_spawn,
        consumed,
        marker=marker,
        durable_marker=cloned_durable,
        sealer_process_launched_at=_time(0, 4.5),
    )
    _failure(
        accrual,
        accrual.record_blocked_spawn,
        consumed,
        marker=marker,
        durable_marker=durable,
        sealer_process_launched_at=_time(0, 5),
    )
    ready_value = accrual.build_ready_resolution(
        spawned, blocked_spawn=spawned.blocked_spawn
    )
    ready_raw = accrual.runtime.canonical_json_bytes(ready_value)
    ready = accrual.probe.validate_probe_resolution(
        ready_raw,
        contract=spawned.contract,
        active_segment=provisional.active_segment,
        expected_previous_ledger_sha256=spawned.head,
        expected_probe_identity=provisional.probe_identity,
        expected_snapshot_identities=dict(provisional.snapshot_items),
        expected_seal_consumption_identity=marker.identity,
        expected_sealer_process_launched_at=_time(0, 4),
    )
    latched = accrual.append_ready_resolution(
        spawned, ready, blocked_spawn=spawned.blocked_spawn
    )
    assert latched.phase == accrual.PHASE_READY_VISIBLE
    assert latched.final_resolution_status == "ready"
    assert latched.blocked_spawn.ledger_head == consumed.head
    _failure(accrual, accrual.append_ready_resolution, latched, ready, blocked_spawn=spawned.blocked_spawn)
    _failure(accrual, accrual.append_probe_attempt, latched, state.current_probe_attempt)


def test_slot_28_stops_source_authority_and_only_horizon_can_add_insufficient(
    accrual: Any,
) -> None:
    state, _ = _bootstrap(accrual)
    for slot_index in range(29):
        if slot_index:
            state = _append_attempt(accrual, state)
        resolution = _resolution(accrual, state)
        state = accrual.append_below_floor_resolution(state, resolution)
    assert state.phase == accrual.PHASE_HORIZON
    assert state.source_authority is False
    assert state.final_resolution_status == "below-floor"
    _failure(
        accrual,
        accrual._require_state,
        replace(
            state,
            final_resolution_identity=accrual.runtime.HashAndBytes(
                "f" * 64, 999
            ),
        ),
    )
    _failure(
        accrual,
        accrual._require_state,
        replace(state, final_resolution_status="segment-closed"),
    )
    value = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "horizon-terminal",
        "status": "insufficient-evidence",
        "final_slot_index": 28,
        "final_slot_resolution_sha256_and_bytes": state.final_resolution_identity.as_dict(),
        "absolute_horizon_expires_at": accrual.runtime.absolute_horizon_at(),
        "recorded_at": accrual.runtime.absolute_horizon_at(),
        "reason": "final-slot-below-floor",
        "previous_ledger_entry_sha256": state.head,
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
    }
    early = dict(value, recorded_at="2026-09-14T05:59:59.999999Z")
    _failure(
        accrual,
        accrual.append_horizon_terminal,
        state,
        accrual.runtime.canonical_json_bytes(early),
    )
    terminal = accrual.append_horizon_terminal(
        state, accrual.runtime.canonical_json_bytes(value)
    )
    assert terminal.terminal_status == "insufficient-evidence"
    assert terminal.entries[-1].entry_kind == "horizon-terminal"


def test_final_slot_closure_terminalizes_only_as_horizon_insufficiency(
    accrual: Any,
) -> None:
    state, _ = _bootstrap(accrual)
    for slot_index in range(28):
        if slot_index:
            state = _append_attempt(accrual, state)
        state = accrual.append_below_floor_resolution(
            state, _resolution(accrual, state)
        )
    state = _append_attempt(accrual, state)
    bound, _ = _bound_source_only_closure(accrual, state)
    state = accrual.append_segment_closure(state, bound)
    assert state.phase == accrual.PHASE_HORIZON
    assert state.final_resolution_status == "segment-closed"
    assert state.source_authority is False
    value = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "horizon-terminal",
        "status": "insufficient-evidence",
        "final_slot_index": 28,
        "final_slot_resolution_sha256_and_bytes": (
            state.final_resolution_identity.as_dict()
        ),
        "absolute_horizon_expires_at": accrual.runtime.absolute_horizon_at(),
        "recorded_at": accrual.runtime.absolute_horizon_at(),
        "reason": "final-slot-segment-closed",
        "previous_ledger_entry_sha256": state.head,
        "analysis_plan_sha256_and_bytes": (
            state.contract.plan_identity.as_dict()
        ),
    }
    terminal = accrual.append_horizon_terminal(
        state, accrual.runtime.canonical_json_bytes(value)
    )
    assert terminal.terminal_status == "insufficient-evidence"


def test_exclusive_storage_never_overwrites_and_recovery_is_read_only(
    accrual: Any, tmp_path: Path
) -> None:
    raw = b"first"
    path = tmp_path / "entry.json"
    assert accrual.write_exclusive(path, raw) == accrual.runtime.hash_and_bytes(raw)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    _failure(accrual, accrual.write_exclusive, path, b"second")
    assert path.read_bytes() == raw
    assert accrual.recover_exact_file(path, raw) == accrual.runtime.hash_and_bytes(raw)
    _failure(accrual, accrual.recover_exact_file, path, b"other")

    target = tmp_path / "target"
    target.write_bytes(b"untouched")
    link = tmp_path / "link"
    link.symlink_to(target)
    _failure(accrual, accrual.write_exclusive, link, b"replacement")
    assert target.read_bytes() == b"untouched"

    directory_collision = tmp_path / "directory-collision"
    directory_collision.mkdir()
    _failure(
        accrual,
        accrual.write_exclusive,
        directory_collision,
        b"replacement",
    )
    fifo = tmp_path / "fifo-collision"
    os.mkfifo(fifo)
    _failure(accrual, accrual.write_exclusive, fifo, b"replacement")

    state = accrual.initialize_ledger()
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    persisted = accrual.write_ledger_entry_exclusive(
        ledger_dir, state
    )
    assert persisted.name == "00000000.json"
    _failure(
        accrual,
        accrual.write_ledger_entry_exclusive,
        ledger_dir,
        state,
    )


def test_exact_recovery_requires_this_invocation_to_have_won_o_excl(
    accrual: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"intended-authority"
    path = tmp_path / "ambiguous-fsync.json"
    real_fsync = accrual.os.fsync
    calls = 0

    def fail_parent_fsync_once(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError
        real_fsync(descriptor)

    monkeypatch.setattr(accrual.os, "fsync", fail_parent_fsync_once)
    assert accrual._write_or_recover_exact(path, raw) == (
        accrual.runtime.hash_and_bytes(raw)
    )
    assert path.read_bytes() == raw
    _failure(accrual, accrual._write_or_recover_exact, path, raw)


# --------------------------------------------------------------------------
# The repaired edge: changed-tuple slot-0 closure and its successor ceremony.
# --------------------------------------------------------------------------


def _service(
    service_fill: str,
    boot_fill: str,
    *,
    boot_started_at: str = "2026-08-14T00:00:00.000000Z",
    commit_fill: str = "a",
    tree_fill: str = "b",
) -> dict[str, Any]:
    return {
        "service_identity_sha256": service_fill * 64,
        "boot_identity_sha256": boot_fill * 64,
        "boot_started_at": boot_started_at,
        "serving_build": {
            "commit": commit_fill * 40,
            "tree": tree_fill * 40,
            "implementation": {
                "src/living_memory/server.py": _identity("c", 321),
                "src/living_memory/store.py": _identity("d", 654),
            },
        },
        "sanitized_configuration": {
            "schema": "living-memory-effective-runtime-configuration-v1",
            "encoding": "canonical-json-utf8",
            "sha256": "e" * 64,
            "bytes": 42,
        },
        "effective_legacy_repeat_controls": {
            "LM_RECALL_REPEAT_GATING": False,
            "LM_RECALL_REPEAT_DROP_TRAILING_STUBS": False,
        },
    }


def _synthetic_services(accrual: Any, *, generation: int = 0) -> Any:
    """A complete two-service tuple that differs from the pinned watermark."""

    fills = "78" if generation == 0 else "9c"
    other = "56" if generation == 0 else "de"
    value = [
        _service(fills[0], fills[1], commit_fill=fills[0], tree_fill=fills[1]),
        _service(other[0], other[1], commit_fill=other[0], tree_fill=other[1]),
    ]
    return accrual.runtime.canonical_service_tuple(value)


def _stable_binding(accrual: Any, services: Any, *, generation: int = 0) -> Any:
    observation = _binding_observation(services, generation=generation)
    return accrual.runtime.validate_stable_source_binding_observations(
        services, observation, copy.deepcopy(observation)
    )


def _initial_closure_value(
    accrual: Any,
    state: Any,
    *,
    observer: Any,
    marker: Any,
    observed_services: Any,
    observed_binding: Any,
    launched_at: str | None = None,
    validated_at: str | None = None,
) -> dict[str, Any]:
    slot = accrual.runtime.slot_times(0)
    value = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "initial-runtime-change-closure",
        "status": "segment-closed",
        "resolution_id": "",
        "slot_index": 0,
        "scheduled_at": slot.scheduled_at,
        "grace_deadline_at": slot.grace_deadline_at,
        "launched_at": _time(0, 1) if launched_at is None else launched_at,
        "validated_at": _time(0, 2) if validated_at is None else validated_at,
        "segment_index": 0,
        "segment_id": state.contract.initial_segment_id,
        "mismatch_kinds": ["active-services-state-change"],
        "prior_active_services_state_sha256_and_bytes": (
            state.contract.initial_services.identity.as_dict()
        ),
        "observed_active_services_state_sha256_and_bytes": (
            observed_services.identity.as_dict()
        ),
        "observed_source_binding_core_sha256_and_bytes": (
            observed_binding.core.identity.as_dict()
        ),
        "attestation_sha256_and_bytes": state.contract.watermark_identity.as_dict(),
        "attestation_attempt_marker_sha256_and_bytes": marker.identity.as_dict(),
        "previous_ledger_entry_sha256": state.head,
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
    }
    value["resolution_id"] = accrual.runtime.derive_slot_resolution_id(value)
    return value


def _changed_tuple_material(accrual: Any) -> dict[str, Any]:
    """Reach ``initial-source-binding-required`` with a changed observed tuple."""

    state, observer = _bootstrap(accrual, stop_after="runtime-attempt")
    marker = state.runtime_attempt
    observed_services = _synthetic_services(accrual)
    observed_binding = _stable_binding(accrual, observed_services, generation=1)
    value = _initial_closure_value(
        accrual,
        state,
        observer=observer,
        marker=marker,
        observed_services=observed_services,
        observed_binding=observed_binding,
    )
    arguments = {
        "observed_services": observed_services,
        "observed_services_raw": observed_services.raw,
        "observed_binding": observed_binding,
        "observed_source_binding_core_raw": observed_binding.core.raw,
        "runtime_observer_identity": observer,
    }
    closure = accrual.runtime.validate_initial_runtime_change_closure(
        accrual.runtime.canonical_json_bytes(value),
        contract=state.contract,
        attempt=marker,
        expected_previous_ledger_sha256=state.head,
        **arguments,
    )
    bound = accrual.bind_initial_runtime_change_closure(
        state, closure, **arguments
    )
    return {
        "state": state,
        "observer": observer,
        "marker": marker,
        "observed_services": observed_services,
        "observed_binding": observed_binding,
        "closure_value": value,
        "closure": closure,
        "bound": bound,
        "arguments": arguments,
    }


def _successor_chain(
    accrual: Any, state: Any, *, observer: Any, services: Any, binding: Any
) -> Any:
    """Run the one fresh successor ceremony a closure authorizes."""

    predecessor = state.closed_predecessor
    closure = state.closure
    attempt_value = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "runtime-attestation-attempt",
        "attempt_scope": "successor-segment",
        "segment_index": predecessor.segment_index + 1,
        "slot_index_or_null": None,
        "predecessor_closure_sha256_or_null": closure.identity.sha256,
        "authorized_at": accrual.runtime.canonical_utc(closure.validated_at),
        "written_at": _time(0, 2.1),
        "start_deadline_at": _time(0, 32),
        "previous_ledger_entry_sha256": state.head,
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
    }
    attempt = accrual.runtime.validate_attestation_attempt_marker(
        accrual.runtime.canonical_json_bytes(attempt_value),
        contract=state.contract,
        runtime_observer_identity=observer,
        expected_previous_ledger_sha256=state.head,
        predecessor_closure=closure,
    )
    state = accrual.append_runtime_attestation_attempt(state, attempt)

    binding_value = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "source-binding-attestation",
        "segment_index": predecessor.segment_index + 1,
        "attestation_attempt_marker_sha256_and_bytes": attempt.identity.as_dict(),
        "alias_ids_by_alias": dict(accrual.runtime.ALIAS_IDS),
        "pre_database_instance_identity_sha256_by_alias": binding.core.database_map(),
        "post_database_instance_identity_sha256_by_alias": binding.core.database_map(),
        "pre_alias_service_database_binding_sha256_by_alias": binding.core.binding_map(),
        "post_alias_service_database_binding_sha256_by_alias": binding.core.binding_map(),
        "active_services_state_sha256": services.identity.sha256,
        "source_binding_core_sha256_and_bytes": binding.core.identity.as_dict(),
        "status": "pass",
        "pre_observed_at": _time(0, 3),
        "post_observed_at": _time(0, 3.7),
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
        "previous_ledger_entry_sha256": state.head,
    }
    binding_attestation = accrual.runtime.validate_source_binding_attestation(
        accrual.runtime.canonical_json_bytes(binding_value),
        contract=state.contract,
        attempt=attempt,
        binding=binding,
        runtime_observer_identity=observer,
        expected_previous_ledger_sha256=state.head,
    )
    state = accrual.append_source_binding_attestation(state, binding_attestation)

    segment_value = {
        "schema_version": 4,
        "namespace": accrual.NAMESPACE,
        "receipt_kind": "runtime-segment-attestation",
        "status": "pass",
        "segment_index": predecessor.segment_index + 1,
        "segment_id": "",
        "attestation_attempt_marker_sha256_and_bytes": attempt.identity.as_dict(),
        "predecessor_closure_sha256": closure.identity.sha256,
        "lower_bound_exclusive_at": _time(0, 3.2),
        "pre_launcher_at": _time(0, 3),
        "pre_source_clock_observed_at_by_alias": {
            "local": _time(0, 3.05),
            "alt": _time(0, 3.2),
        },
        "boundary_launcher_at": _time(0, 3.1),
        "boundary_at": _time(0, 3.2),
        "post_phase_started_at": _time(0, 3.45),
        "post_source_clock_observed_at_by_alias": {
            "local": _time(0, 3.5),
            "alt": _time(0, 3.6),
        },
        "post_launcher_at": _time(0, 3.7),
        "complete_unaliased_service_tuple_sha256_and_bytes": (
            services.identity.as_dict()
        ),
        "source_binding_attestation_sha256_and_bytes": (
            binding_attestation.identity.as_dict()
        ),
        "active_services_state_sha256": services.identity.sha256,
        "replay_code_control_commit_and_tree": {
            "commit": accrual.runtime.REPLAY_CODE_CONTROL_COMMIT,
            "tree": accrual.runtime.REPLAY_CODE_CONTROL_TREE,
        },
        "analysis_plan_sha256_and_bytes": state.contract.plan_identity.as_dict(),
        "previous_ledger_entry_sha256": state.head,
    }
    segment_value["segment_id"] = accrual.runtime.derive_successor_segment_id(
        segment_value
    )
    services_value = accrual.runtime.load_json_bytes(services.raw, expected=list)
    successor = accrual.runtime.validate_successor_segment(
        accrual.runtime.canonical_json_bytes(segment_value),
        contract=state.contract,
        predecessor=predecessor,
        predecessor_closure=closure,
        attempt=attempt,
        source_binding_attestation=binding_attestation,
        pre_services_observation=services_value,
        post_services_observation=list(reversed(copy.deepcopy(services_value))),
        complete_services_raw=services.raw,
        expected_previous_ledger_sha256=state.head,
    )
    return state, attempt, successor


def test_changed_tuple_closure_consumes_slot_zero_and_carries_nothing_across(
    accrual: Any,
) -> None:
    material = _changed_tuple_material(accrual)
    state = material["state"]
    bound = material["bound"]
    assert state.phase == accrual.PHASE_INITIAL_SOURCE_BINDING
    assert state.next_slot_index == 0
    # Both latches are held before the break.
    assert state.current_probe_attempt is not None
    assert state.runtime_attempt is not None
    assert state.active_segment is None

    forged = object.__new__(accrual.BoundInitialRuntimeChangeClosure)
    for field, value in {
        "closure": bound.closure,
        "predecessor": bound.predecessor,
        "ledger_head": state.head,
        "attempt": state.runtime_attempt,
        "_origin": accrual._INITIAL_CLOSURE_BINDING_ORIGIN,
    }.items():
        object.__setattr__(forged, field, value)
    _failure(accrual, accrual.append_initial_runtime_change_closure, state, forged)
    _failure(
        accrual,
        accrual.append_initial_runtime_change_closure,
        state,
        bound.closure,
    )

    closed = accrual.append_initial_runtime_change_closure(state, bound)
    entry = closed.entries[-1]
    assert entry.entry_kind == "initial-runtime-change-closure"
    assert entry.slot_index == 0
    assert closed.phase == accrual.PHASE_SUCCESSOR_ATTEMPT
    assert closed.next_slot_index == 1 == accrual.NEXT_SLOT_AFTER_INITIAL_CLOSURE

    # Every latch and all source authority die with the break.
    assert closed.source_authority is False
    for latch in (
        closed.active_segment,
        closed.current_probe_attempt,
        closed.runtime_attempt,
        closed.source_binding_attestation,
        closed.probe_failure,
        closed.provisional_ready,
        closed.seal_consumption,
        closed.blocked_spawn,
        closed.final_resolution_identity,
        closed.terminal_status,
    ):
        assert latch is None

    # The typed pre-binding predecessor is retained; no ActiveSegment is built.
    assert closed.closed_segment is None
    assert (
        type(closed.unbound_predecessor)
        is accrual.runtime.UnboundWatermarkPredecessor
    )
    assert type(closed.closure) is accrual.runtime.InitialRuntimeChangeClosure
    assert closed.closed_predecessor is closed.unbound_predecessor
    assert closed.unbound_predecessor.segment_index == 0
    assert (
        closed.unbound_predecessor.attestation_identity
        == closed.contract.watermark_identity
    )
    assert not hasattr(closed.unbound_predecessor, "source_binding")

    # No count, no snapshot, and no prior source-binding core is ever appended.
    receipt = json.loads(entry.artifact_raw)
    forbidden = {
        *accrual.probe.AGGREGATE_FIELDS,
        "aliased_source_snapshot_sha256_and_bytes",
        "prior_source_binding_core_sha256_and_bytes",
    }
    assert forbidden.isdisjoint(receipt)
    assert set(receipt) == set(accrual.runtime.INITIAL_CLOSURE_FIELDS)
    assert accrual.closure_entry_kind(closed.closure) == entry.entry_kind

    # First validator-valid slot-0 resolution wins and is immutable: no second
    # closure, no late binding, and no further slot-0 probe.
    _failure(accrual, accrual.append_initial_runtime_change_closure, closed, bound)
    _failure(
        accrual,
        accrual.append_probe_attempt,
        closed,
        state.current_probe_attempt,
    )
    _failure(accrual, accrual.append_runtime_attestation_attempt, closed, material["marker"])


def test_slot_zero_admits_exactly_one_of_binding_or_pre_binding_closure(
    accrual: Any,
) -> None:
    contract = accrual.initialize_ledger().contract
    material = _changed_tuple_material(accrual)
    state = material["state"]
    observed = material["observed_services"]
    assert (
        accrual.runtime.initial_bootstrap_branch(contract, observed)
        == accrual.runtime.CHANGED_TUPLE_BRANCH
    )
    assert (
        accrual.runtime.initial_bootstrap_branch(contract, contract.initial_services)
        == accrual.runtime.SAME_TUPLE_BRANCH
    )

    # A same-tuple observation can never be spelled as a pre-binding closure.
    watermark_binding = _stable_binding(accrual, contract.initial_services)
    _failure(
        accrual,
        accrual.bind_initial_runtime_change_closure,
        state,
        material["closure"],
        observed_services=contract.initial_services,
        observed_services_raw=contract.initial_services.raw,
        observed_binding=watermark_binding,
        observed_source_binding_core_raw=watermark_binding.core.raw,
        runtime_observer_identity=material["observer"],
    )

    # And the same-tuple branch, once taken, has no pre-binding closure exit.
    same, _ = _bootstrap(accrual)
    assert same.phase == accrual.PHASE_PROBE_ACTIVE
    assert type(same.active_segment) is accrual.runtime.ActiveSegment
    assert same.unbound_predecessor is None and same.closure is None
    assert same.source_authority is True
    _failure(
        accrual, accrual.append_initial_runtime_change_closure, same, material["bound"]
    )
    _failure(
        accrual,
        accrual.bind_initial_runtime_change_closure,
        same,
        material["closure"],
        **material["arguments"],
    )


def test_pre_binding_closure_permits_exactly_one_timely_fresh_successor(
    accrual: Any,
) -> None:
    material = _changed_tuple_material(accrual)
    state = accrual.append_initial_runtime_change_closure(
        material["state"], material["bound"]
    )
    observer = material["observer"]
    closure = state.closure

    # The ceremony must start inside the 30-second window after the closure.
    _failure(
        accrual,
        accrual.append_runtime_attestation_terminal,
        state,
        _successor_runtime_terminal_raw(
            accrual, state, reason="attempt-start-late", recorded_at=_time(0, 31.999999)
        ),
    )
    late = accrual.append_runtime_attestation_terminal(
        state,
        _successor_runtime_terminal_raw(
            accrual, state, reason="attempt-start-late", recorded_at=_time(0, 32)
        ),
    )
    assert late.terminal_status == "terminal-runtime-attestation-failure"
    assert late.closure is None and late.unbound_predecessor is None
    assert late.closed_predecessor is None

    # The slot-0 marker the closure consumed can never be the successor marker.
    _failure(
        accrual, accrual.append_runtime_attestation_attempt, state, material["marker"]
    )

    advanced, attempt, successor = _successor_chain(
        accrual,
        state,
        observer=observer,
        services=material["observed_services"],
        binding=material["observed_binding"],
    )
    assert attempt.identity != closure.attempt_identity
    state = accrual.append_successor_segment(advanced, successor)
    assert state.phase == accrual.PHASE_AWAITING_PROBE
    assert state.next_slot_index == 1
    assert state.active_segment.segment_index == 1
    assert state.closure is None
    assert state.unbound_predecessor is None
    assert state.closed_segment is None
    assert (
        type(state.active_segment.predecessor_segment)
        is accrual.runtime.UnboundWatermarkPredecessor
    )
    assert (
        type(state.active_segment.predecessor_closure)
        is accrual.runtime.InitialRuntimeChangeClosure
    )
    # Exactly one successor: the same ceremony cannot be replayed.
    _failure(accrual, accrual.append_successor_segment, state, successor)
    # Accrual resumes at slot 1, never back at the consumed slot 0.
    state = _append_attempt(accrual, state)
    assert state.current_probe_attempt.slot_index == 1


def test_pre_binding_predecessor_is_never_coerced_into_an_active_segment(
    accrual: Any,
) -> None:
    runtime = accrual.runtime
    material = _changed_tuple_material(accrual)
    state = accrual.append_initial_runtime_change_closure(
        material["state"], material["bound"]
    )
    predecessor = state.unbound_predecessor
    closure = state.closure

    # The two lineages may never be cross-paired in either direction.
    _failure(
        runtime,
        runtime._require_predecessor_pair_consistent,
        predecessor,
        object.__new__(runtime.ValidatedClosure),
    )
    same, _ = _bootstrap(accrual)
    _failure(
        runtime,
        runtime._require_predecessor_pair_consistent,
        same.active_segment,
        closure,
    )
    # The pre-binding closure is not a slot-segment-closed entry and vice versa.
    assert accrual.closure_entry_kind(closure) == accrual.INITIAL_CLOSURE_ENTRY_KIND
    _failure(accrual, accrual.closure_entry_kind, predecessor)
    _failure(
        accrual,
        accrual._entry_for_artifact,
        state.entries,
        closure.identity,
        accrual.CLOSURE_ENTRY_KIND,
    )

    # A state that files the typed predecessor as a closed ActiveSegment, or
    # keeps both, is not a reachable state.
    _failure(
        accrual,
        accrual._require_state,
        replace(state, closed_segment=same.active_segment),
    )
    _failure(
        accrual, accrual._require_state, replace(state, unbound_predecessor=None)
    )
    _failure(
        accrual,
        accrual._require_state,
        replace(state, unbound_predecessor=None, closed_segment=same.active_segment),
    )
    _failure(accrual, accrual._require_state, replace(state, closure=None))

    # A successor grown from a fabricated ActiveSegment predecessor is refused.
    advanced, _attempt, successor = _successor_chain(
        accrual,
        state,
        observer=material["observer"],
        services=material["observed_services"],
        binding=material["observed_binding"],
    )
    _failure(
        accrual,
        accrual.append_successor_segment,
        advanced,
        replace(successor, predecessor_segment=same.active_segment),
    )
    _failure(
        accrual,
        accrual.append_successor_segment,
        advanced,
        replace(successor, predecessor_closure=None),
    )

    # The retained-segment binder itself refuses every cross-paired lineage,
    # so a successor already in the ledger cannot be re-read as the other kind.
    final = accrual.append_successor_segment(advanced, successor)
    active = final.active_segment
    bound_arguments = (
        final,
        active,
        active.attestation_identity,
        active.source_binding_attestation.identity,
    )
    accrual._require_active_segment_bound(*bound_arguments)
    for forged in (
        replace(active, predecessor_closure=object.__new__(runtime.ValidatedClosure)),
        replace(active, predecessor_segment=same.active_segment),
        replace(active, predecessor_segment=None),
        replace(active, predecessor_closure=None),
    ):
        _failure(
            accrual,
            accrual._require_active_segment_bound,
            final,
            forged,
            active.attestation_identity,
            active.source_binding_attestation.identity,
        )
    # The observer a pre-binding successor must reuse comes from the closure,
    # never from a source-binding attestation that has never existed.
    assert (
        accrual._predecessor_observer_identity(predecessor, closure)
        == closure.observer_identity
    )
    _failure(
        accrual,
        accrual._predecessor_observer_identity,
        predecessor,
        object.__new__(runtime.ValidatedClosure),
    )


def test_ledger_replay_reconstructs_the_break_from_durable_entries_only(
    accrual: Any, tmp_path: Path
) -> None:
    material = _changed_tuple_material(accrual)
    state = accrual.append_initial_runtime_change_closure(
        material["state"], material["bound"]
    )
    advanced, _attempt, successor = _successor_chain(
        accrual,
        state,
        observer=material["observer"],
        services=material["observed_services"],
        binding=material["observed_binding"],
    )
    final = accrual.append_successor_segment(advanced, successor)

    for step in (state, advanced, final):
        directory = tmp_path / f"ledger-{len(step.entries)}"
        directory.mkdir()
        for entry in step.entries:
            (directory / accrual.ledger_entry_filename(entry.entry_index)).write_bytes(
                entry.raw
            )
        accrual.validate_persisted_ledger_prefix(directory, step)

    # Replaying the durable transcript alone reproduces the repaired phase.
    at_break = accrual._derive_state_projection(state.entries)
    assert at_break.phase == accrual.PHASE_SUCCESSOR_ATTEMPT
    assert at_break.next_slot_index == 1
    assert at_break.pre_binding_closure is True
    assert at_break.closure_identity == state.closure.identity
    assert at_break.pre_binding_closure_attempt_identity == (
        material["marker"].identity
    )
    # Nothing from before the break survives in the replayed projection.
    for absent in (
        at_break.closed_attestation_identity,
        at_break.closed_source_binding_identity,
        at_break.active_attestation_identity,
        at_break.active_source_binding_identity,
        at_break.current_probe_attempt_identity,
        at_break.runtime_attempt_identity,
        at_break.probe_failure_identity,
        at_break.seal_consumption_identity,
        at_break.final_resolution_identity,
        at_break.terminal_status,
    ):
        assert absent is None

    after = accrual._derive_state_projection(final.entries)
    assert after.phase == accrual.PHASE_AWAITING_PROBE
    assert after.next_slot_index == 1
    assert after.pre_binding_closure is False
    assert after.closure_identity is None
    assert after.pre_binding_closure_attempt_identity is None
    assert after.active_attestation_identity == final.active_segment.attestation_identity

    # A transcript that drops the closure entry no longer replays.
    _failure(
        accrual,
        accrual._derive_state_projection,
        state.entries[:-1] + (advanced.entries[-1],),
    )


def test_pre_binding_closure_is_a_slot_zero_ledger_artifact_only(
    accrual: Any,
) -> None:
    runtime = accrual.runtime
    material = _changed_tuple_material(accrual)
    state = material["state"]
    closure_value = material["closure_value"]
    closure_raw = material["closure"].raw

    # The frozen schema itself carries no count and no snapshot member.
    forbidden = {
        *accrual.probe.AGGREGATE_FIELDS,
        "aliased_source_snapshot_sha256_and_bytes",
        "prior_source_binding_core_sha256_and_bytes",
    }
    assert forbidden.isdisjoint(runtime.INITIAL_CLOSURE_FIELDS)
    assert accrual._COUNT_AND_SNAPSHOT_FIELDS.isdisjoint(
        runtime.INITIAL_CLOSURE_FIELDS
    )

    kind, slot, previous = accrual._artifact_kind_and_slot(
        closure_raw,
        entry_kind=accrual.INITIAL_CLOSURE_ENTRY_KIND,
        declared_slot=0,
        contract=state.contract,
    )
    assert kind == runtime.INITIAL_CLOSURE_RECEIPT_KIND
    assert slot == runtime.FIRST_SLOT_INDEX
    assert previous == state.head

    # No other slot may ever carry it, whether declared honestly or not.
    for slot_index in (1, 2, runtime.LAST_SLOT_INDEX):
        forged = dict(closure_value)
        forged["slot_index"] = slot_index
        forged["resolution_id"] = runtime.derive_slot_resolution_id(forged)
        forged_raw = runtime.canonical_json_bytes(forged)
        for declared in (slot_index, 0, None):
            _failure(
                accrual,
                accrual._artifact_kind_and_slot,
                forged_raw,
                entry_kind=accrual.INITIAL_CLOSURE_ENTRY_KIND,
                declared_slot=declared,
                contract=state.contract,
            )
        _failure(
            accrual,
            accrual._build_entry,
            contract=state.contract,
            entry_index=state.next_entry_index,
            previous=state.head,
            entry_kind=accrual.INITIAL_CLOSURE_ENTRY_KIND,
            slot_index=slot_index,
            artifact_raw=forged_raw,
        )
    # The honest closure cannot be relabelled onto a non-zero slot either.
    _failure(
        accrual,
        accrual._build_entry,
        contract=state.contract,
        entry_index=state.next_entry_index,
        previous=state.head,
        entry_kind=accrual.INITIAL_CLOSURE_ENTRY_KIND,
        slot_index=1,
        artifact_raw=closure_raw,
    )

    # A non-zero segment index, a wrong status, or the ordinary closure kind is
    # never this artifact.
    for field, value in (
        ("segment_index", 1),
        ("status", "pass"),
        ("receipt_kind", "slot-segment-closed"),
    ):
        forged = dict(closure_value)
        forged[field] = value
        forged["resolution_id"] = runtime.derive_slot_resolution_id(forged)
        _failure(
            accrual,
            accrual._artifact_kind_and_slot,
            runtime.canonical_json_bytes(forged),
            entry_kind=accrual.INITIAL_CLOSURE_ENTRY_KIND,
            declared_slot=0,
            contract=state.contract,
        )
    # And the two closure entry kinds never accept each other's bytes.
    _failure(
        accrual,
        accrual._artifact_kind_and_slot,
        closure_raw,
        entry_kind=accrual.CLOSURE_ENTRY_KIND,
        declared_slot=0,
        contract=state.contract,
    )
    closed = accrual.append_initial_runtime_change_closure(
        state, material["bound"]
    )
    _failure(
        accrual,
        accrual._artifact_kind_and_slot,
        closed.entries[-1].artifact_raw,
        entry_kind=accrual.CLOSURE_ENTRY_KIND,
        declared_slot=0,
        contract=state.contract,
    )
