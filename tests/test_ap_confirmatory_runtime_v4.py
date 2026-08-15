"""Raw-byte adversarial tests for the shared confirmatory-v4 runtime core.

Every fixture here is an invented observation.  These tests never open a Living
Memory database, contact a service, probe a source, build a packet, or invoke a
retired packet verifier.  The only repository artifacts read are the public
frozen v4 analysis plan and the release-v1 control watermark that the runtime
contract is required to pin.

The suite is organised around the one repaired edge: the initial segment is
attested by the watermark alone and has no source binding, so the two bootstrap
branches must both be reachable, must be mutually exclusive, and the pre-binding
branch must never require, construct, or accept a prior source-binding core.
"""

from __future__ import annotations

import base64
import copy
import importlib.util
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ap_confirmatory_runtime_v4.py"
FROZEN_PLAN = (
    REPO_ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "confirmatory-holdout-v4"
    / "analysis-plan.json"
)
FROZEN_WATERMARK = (
    REPO_ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "release-v1"
    / "control-watermark.json"
)


@pytest.fixture(scope="module")
def runtime() -> Any:
    spec = importlib.util.spec_from_file_location(
        "ap_confirmatory_runtime_v4_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def contract(runtime: Any) -> Any:
    return runtime.load_frozen_contract()


def _failure(runtime: Any, call: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Assert the one deliberately content-free fatal integrity signal."""

    with pytest.raises(runtime.IntegrityFailure) as caught:
        call(*args, **kwargs)
    assert caught.value.args == ()
    assert str(caught.value) == ""


def _raw(runtime: Any, value: Any) -> bytes:
    return runtime.canonical_json_bytes(value)


def _time(seconds: float) -> str:
    base = datetime(2026, 8, 17, tzinfo=UTC)
    return (base + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _identity(fill: str, size: int = 1) -> dict[str, Any]:
    return {"sha256": fill * 64, "bytes": size}


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


def _synthetic_services(
    runtime: Any, *, generation: int = 0, boot_started_at: str | None = None
) -> tuple[Any, list[dict[str, Any]]]:
    """Build a complete two-service tuple that differs from the watermark."""

    fills = "78" if generation == 0 else "9c"
    other = "56" if generation == 0 else "de"
    boot = boot_started_at or "2026-08-14T00:00:00.000000Z"
    value = [
        _service(fills[0], fills[1], commit_fill=fills[0], tree_fill=fills[1]),
        _service(
            other[0],
            other[1],
            boot_started_at=boot,
            commit_fill=other[0],
            tree_fill=other[1],
        ),
    ]
    return runtime.canonical_service_tuple(value), value


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


def _database(alias: str, *, generation: int = 0) -> dict[str, Any]:
    tail = 1 + generation * 2 + (alias == "alt")
    return {
        "filesystem_uuid": f"00000000-0000-4000-8000-{tail:012d}",
        "statx_inode_uint64": 70_000 + tail,
        "statx_birthtime_ns_int64": 1_700_000_000_000_000_000 + tail,
    }


def _binding_observation(
    runtime: Any,
    services: Any,
    *,
    generation: int = 0,
    swapped: bool = False,
) -> dict[str, Any]:
    pairs = list(services.service_pairs)
    if swapped:
        pairs.reverse()
    result: dict[str, Any] = {}
    for index, alias in enumerate(("local", "alt")):
        service_identity, boot_identity = pairs[index]
        result[alias] = {
            "alias_id": runtime.ALIAS_IDS[alias],
            "authenticated_authority_identity": _authority(alias),
            "service_identity_sha256": service_identity,
            "boot_identity_sha256": boot_identity,
            "database_instance_identity": _database(alias, generation=generation),
        }
    return result


def _stable_binding(runtime: Any, services: Any, *, generation: int = 0) -> Any:
    observation = _binding_observation(runtime, services, generation=generation)
    return runtime.validate_stable_source_binding_observations(
        services, observation, copy.deepcopy(observation)
    )


def _receipt_utc(runtime: Any, value: str) -> Any:
    return runtime.parse_utc(value, receipt=True)


def _observer(runtime: Any) -> Any:
    return runtime.hash_and_bytes(b"synthetic-runtime-observer-v4")


def _slot_zero_marker_value(runtime: Any, contract: Any, observer: Any) -> dict[str, Any]:
    return {
        "schema_version": 4,
        "namespace": "confirmatory-holdout-v4",
        "receipt_kind": "runtime-attestation-attempt",
        "attempt_scope": "initial-source-binding",
        "segment_index": 0,
        "slot_index_or_null": 0,
        "predecessor_closure_sha256_or_null": None,
        "authorized_at": runtime.slot_times(0).scheduled_at,
        "written_at": _time(0.010),
        "start_deadline_at": runtime.slot_times(0).grace_deadline_at,
        "previous_ledger_entry_sha256": "1" * 64,
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
    }


def _slot_zero_marker(runtime: Any, contract: Any, observer: Any) -> tuple[Any, dict[str, Any]]:
    value = _slot_zero_marker_value(runtime, contract, observer)
    marker = runtime.validate_attestation_attempt_marker(
        _raw(runtime, value),
        contract=contract,
        runtime_observer_identity=observer,
        expected_previous_ledger_sha256=value["previous_ledger_entry_sha256"],
        initial_probe_launched_at=_time(0),
    )
    return marker, value


def _same_tuple_material(runtime: Any, contract: Any) -> dict[str, Any]:
    """The whole ``same-tuple-initial-binding`` branch, with no closure at all."""

    observer = _observer(runtime)
    marker, marker_value = _slot_zero_marker(runtime, contract, observer)
    binding = _stable_binding(runtime, contract.initial_services)
    binding_previous = "2" * 64
    binding_value = {
        "schema_version": 4,
        "namespace": "confirmatory-holdout-v4",
        "receipt_kind": "source-binding-attestation",
        "segment_index": 0,
        "attestation_attempt_marker_sha256_and_bytes": marker.identity.as_dict(),
        "alias_ids_by_alias": dict(runtime.ALIAS_IDS),
        "pre_database_instance_identity_sha256_by_alias": binding.core.database_map(),
        "post_database_instance_identity_sha256_by_alias": binding.core.database_map(),
        "pre_alias_service_database_binding_sha256_by_alias": binding.core.binding_map(),
        "post_alias_service_database_binding_sha256_by_alias": binding.core.binding_map(),
        "active_services_state_sha256": contract.initial_services.identity.sha256,
        "source_binding_core_sha256_and_bytes": binding.core.identity.as_dict(),
        "status": "pass",
        "pre_observed_at": _time(0.020),
        "post_observed_at": _time(0.030),
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
        "previous_ledger_entry_sha256": binding_previous,
    }
    attestation = runtime.validate_source_binding_attestation(
        _raw(runtime, binding_value),
        contract=contract,
        attempt=marker,
        binding=binding,
        runtime_observer_identity=observer,
        expected_previous_ledger_sha256=binding_previous,
    )
    active = runtime.make_initial_segment(contract, attestation)
    return {
        "observer": observer,
        "marker": marker,
        "marker_value": marker_value,
        "binding": binding,
        "binding_value": binding_value,
        "binding_previous": binding_previous,
        "binding_attestation": attestation,
        "active": active,
    }


def _changed_tuple_closure_value(
    runtime: Any,
    contract: Any,
    *,
    observer: Any,
    marker: Any,
    observed_services: Any,
    observed_binding: Any,
    previous: str,
) -> dict[str, Any]:
    value = {
        "schema_version": 4,
        "namespace": "confirmatory-holdout-v4",
        "receipt_kind": "initial-runtime-change-closure",
        "status": "segment-closed",
        "resolution_id": "0" * 64,
        "slot_index": 0,
        "scheduled_at": runtime.slot_times(0).scheduled_at,
        "grace_deadline_at": runtime.slot_times(0).grace_deadline_at,
        "launched_at": _time(0),
        "validated_at": _time(2),
        "segment_index": 0,
        "segment_id": contract.initial_segment_id,
        "mismatch_kinds": ["active-services-state-change"],
        "prior_active_services_state_sha256_and_bytes": (
            contract.initial_services.identity.as_dict()
        ),
        "observed_active_services_state_sha256_and_bytes": (
            observed_services.identity.as_dict()
        ),
        "observed_source_binding_core_sha256_and_bytes": (
            observed_binding.core.identity.as_dict()
        ),
        "attestation_sha256_and_bytes": contract.watermark_identity.as_dict(),
        "attestation_attempt_marker_sha256_and_bytes": marker.identity.as_dict(),
        "previous_ledger_entry_sha256": previous,
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
    }
    value["resolution_id"] = runtime.derive_slot_resolution_id(value)
    return value


def _changed_tuple_material(runtime: Any, contract: Any) -> dict[str, Any]:
    """The whole ``changed-tuple-pre-binding-closure`` branch."""

    observer = _observer(runtime)
    marker, marker_value = _slot_zero_marker(runtime, contract, observer)
    observed_services, observed_value = _synthetic_services(runtime)
    observed_binding = _stable_binding(runtime, observed_services, generation=1)
    previous = "3" * 64
    value = _changed_tuple_closure_value(
        runtime,
        contract,
        observer=observer,
        marker=marker,
        observed_services=observed_services,
        observed_binding=observed_binding,
        previous=previous,
    )
    closure = runtime.validate_initial_runtime_change_closure(
        _raw(runtime, value),
        contract=contract,
        attempt=marker,
        observed_services=observed_services,
        observed_services_raw=observed_services.raw,
        observed_binding=observed_binding,
        observed_source_binding_core_raw=observed_binding.core.raw,
        runtime_observer_identity=observer,
        expected_previous_ledger_sha256=previous,
    )
    return {
        "observer": observer,
        "marker": marker,
        "marker_value": marker_value,
        "observed_services": observed_services,
        "observed_services_value": observed_value,
        "observed_binding": observed_binding,
        "closure_previous": previous,
        "closure_value": value,
        "closure": closure,
        "predecessor": closure.predecessor,
    }


def _validate_initial_closure(
    runtime: Any,
    contract: Any,
    material: dict[str, Any],
    value: dict[str, Any],
    **overrides: Any,
) -> Any:
    arguments = {
        "contract": contract,
        "attempt": material["marker"],
        "observed_services": material["observed_services"],
        "observed_services_raw": material["observed_services"].raw,
        "observed_binding": material["observed_binding"],
        "observed_source_binding_core_raw": material["observed_binding"].core.raw,
        "runtime_observer_identity": material["observer"],
        "expected_previous_ledger_sha256": material["closure_previous"],
    }
    arguments.update(overrides)
    return runtime.validate_initial_runtime_change_closure(
        _raw(runtime, value), **arguments
    )


def _successor_material(
    runtime: Any,
    contract: Any,
    *,
    successor_generation: int = 0,
    services_override: Any = None,
) -> dict[str, Any]:
    """A fresh successor ceremony started from the pre-binding closure."""

    base = _changed_tuple_material(runtime, contract)
    closure = base["closure"]
    observer = base["observer"]
    if services_override is None:
        services = base["observed_services"]
        services_value = base["observed_services_value"]
    else:
        services, services_value = services_override
    binding = _stable_binding(runtime, services, generation=1 + successor_generation)

    attempt_previous = "4" * 64
    attempt_value = {
        "schema_version": 4,
        "namespace": "confirmatory-holdout-v4",
        "receipt_kind": "runtime-attestation-attempt",
        "attempt_scope": "successor-segment",
        "segment_index": 1,
        "slot_index_or_null": None,
        "predecessor_closure_sha256_or_null": closure.identity.sha256,
        "authorized_at": _time(2),
        "written_at": _time(2.5),
        "start_deadline_at": _time(32),
        "previous_ledger_entry_sha256": attempt_previous,
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
    }
    attempt = runtime.validate_attestation_attempt_marker(
        _raw(runtime, attempt_value),
        contract=contract,
        runtime_observer_identity=observer,
        expected_previous_ledger_sha256=attempt_previous,
        predecessor_closure=closure,
    )

    binding_previous = "5" * 64
    binding_value = {
        "schema_version": 4,
        "namespace": "confirmatory-holdout-v4",
        "receipt_kind": "source-binding-attestation",
        "segment_index": 1,
        "attestation_attempt_marker_sha256_and_bytes": attempt.identity.as_dict(),
        "alias_ids_by_alias": dict(runtime.ALIAS_IDS),
        "pre_database_instance_identity_sha256_by_alias": binding.core.database_map(),
        "post_database_instance_identity_sha256_by_alias": binding.core.database_map(),
        "pre_alias_service_database_binding_sha256_by_alias": binding.core.binding_map(),
        "post_alias_service_database_binding_sha256_by_alias": binding.core.binding_map(),
        "active_services_state_sha256": services.identity.sha256,
        "source_binding_core_sha256_and_bytes": binding.core.identity.as_dict(),
        "status": "pass",
        "pre_observed_at": _time(3),
        "post_observed_at": _time(4),
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
        "previous_ledger_entry_sha256": binding_previous,
    }
    binding_attestation = runtime.validate_source_binding_attestation(
        _raw(runtime, binding_value),
        contract=contract,
        attempt=attempt,
        binding=binding,
        runtime_observer_identity=observer,
        expected_previous_ledger_sha256=binding_previous,
    )

    segment_previous = "6" * 64
    segment_value = {
        "schema_version": 4,
        "namespace": "confirmatory-holdout-v4",
        "receipt_kind": "runtime-segment-attestation",
        "status": "pass",
        "segment_index": 1,
        "segment_id": "0" * 64,
        "attestation_attempt_marker_sha256_and_bytes": attempt.identity.as_dict(),
        "predecessor_closure_sha256": closure.identity.sha256,
        "lower_bound_exclusive_at": _time(3.3),
        "pre_launcher_at": _time(3),
        "pre_source_clock_observed_at_by_alias": {
            "local": _time(3.1),
            "alt": _time(3.2),
        },
        "boundary_launcher_at": _time(3.3),
        "boundary_at": _time(3.3),
        "post_phase_started_at": _time(3.6),
        "post_source_clock_observed_at_by_alias": {
            "local": _time(3.7),
            "alt": _time(3.8),
        },
        "post_launcher_at": _time(4),
        "complete_unaliased_service_tuple_sha256_and_bytes": services.identity.as_dict(),
        "source_binding_attestation_sha256_and_bytes": (
            binding_attestation.identity.as_dict()
        ),
        "active_services_state_sha256": services.identity.sha256,
        "replay_code_control_commit_and_tree": {
            "commit": runtime.REPLAY_CODE_CONTROL_COMMIT,
            "tree": runtime.REPLAY_CODE_CONTROL_TREE,
        },
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
        "previous_ledger_entry_sha256": segment_previous,
    }
    segment_value["segment_id"] = runtime.derive_successor_segment_id(segment_value)
    return {
        **base,
        "successor_services": services,
        "successor_services_value": services_value,
        "successor_binding": binding,
        "attempt_value": attempt_value,
        "attempt_previous": attempt_previous,
        "successor_attempt": attempt,
        "successor_binding_value": binding_value,
        "successor_binding_attestation": binding_attestation,
        "segment_previous": segment_previous,
        "segment_value": segment_value,
    }


def _validate_successor(
    runtime: Any,
    contract: Any,
    material: dict[str, Any],
    value: dict[str, Any],
    **overrides: Any,
) -> Any:
    arguments = {
        "contract": contract,
        "predecessor": material["predecessor"],
        "predecessor_closure": material["closure"],
        "attempt": material["successor_attempt"],
        "source_binding_attestation": material["successor_binding_attestation"],
        "pre_services_observation": copy.deepcopy(
            material["successor_services_value"]
        ),
        "post_services_observation": copy.deepcopy(
            material["successor_services_value"]
        ),
        "complete_services_raw": material["successor_services"].raw,
        "expected_previous_ledger_sha256": material["segment_previous"],
    }
    arguments.update(overrides)
    return runtime.validate_successor_segment(_raw(runtime, value), **arguments)


def test_same_tuple_branch_opens_the_initial_segment_without_any_closure(
    runtime: Any, contract: Any
) -> None:
    material = _same_tuple_material(runtime, contract)
    active = material["active"]
    assert type(active) is runtime.ActiveSegment
    assert active.segment_index == 0
    assert active.segment_id == contract.initial_segment_id
    assert active.lower_bound_exclusive_at == contract.release_effective_at
    assert active.attestation_identity == contract.watermark_identity
    # Same-tuple evidence alone opens the segment: no closure, no predecessor.
    assert active.predecessor_closure is None
    assert active.predecessor_segment is None
    assert (
        runtime.initial_bootstrap_branch(contract, contract.initial_services)
        == runtime.SAME_TUPLE_BRANCH
    )


def test_changed_tuple_branch_closes_slot_zero_without_a_prior_core(
    runtime: Any, contract: Any
) -> None:
    material = _changed_tuple_material(runtime, contract)
    closure = material["closure"]
    assert type(closure) is runtime.InitialRuntimeChangeClosure
    assert closure.slot_index == 0
    assert closure.segment_index == 0
    assert closure.segment_id == contract.initial_segment_id
    assert closure.attempt_identity == material["marker"].identity
    predecessor = closure.predecessor
    assert type(predecessor) is runtime.UnboundWatermarkPredecessor
    assert predecessor.services.identity == contract.initial_services.identity
    assert predecessor.attestation_identity == contract.watermark_identity
    # The typed predecessor is not and cannot become an ActiveSegment.
    assert not isinstance(predecessor, runtime.ActiveSegment)
    assert not hasattr(predecessor, "source_binding")
    assert not hasattr(predecessor, "source_binding_attestation")
    # The wrapper exposes no observation a successor could reuse.
    assert not hasattr(closure, "observed_services")
    assert not hasattr(closure, "observed_source_binding_core")
    assert (
        runtime.initial_bootstrap_branch(contract, material["observed_services"])
        == runtime.CHANGED_TUPLE_BRANCH
    )
    receipt = json.loads(closure.raw.decode("utf-8"))
    assert "prior_source_binding_core_sha256_and_bytes" not in receipt
    assert receipt["mismatch_kinds"] == ["active-services-state-change"]


def test_bootstrap_branch_selector_is_exactly_tuple_equality(
    runtime: Any, contract: Any
) -> None:
    observed, _value = _synthetic_services(runtime)
    assert (
        runtime.initial_bootstrap_branch(contract, contract.initial_services)
        != runtime.initial_bootstrap_branch(contract, observed)
    )
    assert runtime.SAME_TUPLE_BRANCH != runtime.CHANGED_TUPLE_BRANCH
    _failure(runtime, runtime.initial_bootstrap_branch, contract, None)
    _failure(
        runtime,
        runtime.initial_bootstrap_branch,
        contract,
        replace(contract.initial_services, identity=runtime.hash_and_bytes(b"x")),
    )


def test_the_two_bootstrap_branches_are_mutually_exclusive(
    runtime: Any, contract: Any
) -> None:
    same = _same_tuple_material(runtime, contract)
    changed = _changed_tuple_material(runtime, contract)

    # A same-tuple observation can never produce the pre-binding closure.
    value = _changed_tuple_closure_value(
        runtime,
        contract,
        observer=same["observer"],
        marker=same["marker"],
        observed_services=contract.initial_services,
        observed_binding=same["binding"],
        previous="3" * 64,
    )
    _failure(
        runtime,
        runtime.validate_initial_runtime_change_closure,
        _raw(runtime, value),
        contract=contract,
        attempt=same["marker"],
        observed_services=contract.initial_services,
        observed_services_raw=contract.initial_services.raw,
        observed_binding=same["binding"],
        observed_source_binding_core_raw=same["binding"].core.raw,
        runtime_observer_identity=same["observer"],
        expected_previous_ledger_sha256="3" * 64,
    )
    _failure(
        runtime,
        runtime.validate_pre_binding_transition,
        contract.initial_services,
        contract.initial_services,
    )

    # A changed-tuple observation can never open the initial ActiveSegment.
    changed_binding_value = dict(same["binding_value"])
    changed_binding_value["active_services_state_sha256"] = changed[
        "observed_services"
    ].identity.sha256
    changed_binding_value["pre_database_instance_identity_sha256_by_alias"] = changed[
        "observed_binding"
    ].core.database_map()
    changed_binding_value["post_database_instance_identity_sha256_by_alias"] = changed[
        "observed_binding"
    ].core.database_map()
    changed_binding_value["pre_alias_service_database_binding_sha256_by_alias"] = changed[
        "observed_binding"
    ].core.binding_map()
    changed_binding_value["post_alias_service_database_binding_sha256_by_alias"] = changed[
        "observed_binding"
    ].core.binding_map()
    changed_binding_value["source_binding_core_sha256_and_bytes"] = changed[
        "observed_binding"
    ].core.identity.as_dict()
    changed_attestation = runtime.validate_source_binding_attestation(
        _raw(runtime, changed_binding_value),
        contract=contract,
        attempt=same["marker"],
        binding=changed["observed_binding"],
        runtime_observer_identity=same["observer"],
        expected_previous_ledger_sha256=same["binding_previous"],
    )
    _failure(runtime, runtime.make_initial_segment, contract, changed_attestation)


@pytest.mark.parametrize(
    "forged_prior",
    [
        None,
        "",
        0,
        False,
        "absent",
        "none",
        {"sha256": "0" * 64, "bytes": 0},
        {"sha256": "f" * 64, "bytes": 512},
        [],
        {},
    ],
    ids=[
        "null",
        "empty-string",
        "zero",
        "false",
        "sentinel-absent",
        "sentinel-none",
        "placeholder-digest",
        "fabricated-object",
        "empty-array",
        "empty-object",
    ],
)
def test_pre_binding_closure_rejects_every_spelling_of_a_prior_core(
    runtime: Any, contract: Any, forged_prior: Any
) -> None:
    material = _changed_tuple_material(runtime, contract)
    value = dict(material["closure_value"])
    value["prior_source_binding_core_sha256_and_bytes"] = forged_prior
    value["resolution_id"] = runtime.derive_slot_resolution_id(value)
    _failure(runtime, _validate_initial_closure, runtime, contract, material, value)


def test_pre_binding_closure_rejects_a_repeat_of_the_observed_core_as_prior(
    runtime: Any, contract: Any
) -> None:
    material = _changed_tuple_material(runtime, contract)
    value = dict(material["closure_value"])
    value["prior_source_binding_core_sha256_and_bytes"] = material[
        "observed_binding"
    ].core.identity.as_dict()
    value["resolution_id"] = runtime.derive_slot_resolution_id(value)
    _failure(runtime, _validate_initial_closure, runtime, contract, material, value)


def test_pre_binding_transition_takes_no_binding_argument_at_all(
    runtime: Any, contract: Any
) -> None:
    observed, _value = _synthetic_services(runtime)
    kinds = runtime.validate_pre_binding_transition(contract.initial_services, observed)
    assert kinds == ("active-services-state-change",)
    # There is no parameter through which a prior core could be supplied: the
    # signature itself refuses the argument before any validation can run.
    with pytest.raises(TypeError):
        runtime.validate_pre_binding_transition(
            contract.initial_services,
            observed,
            _stable_binding(runtime, observed, generation=1),
        )
    _failure(
        runtime,
        runtime.validate_pre_binding_transition,
        contract.initial_services,
        None,
    )


def test_pre_binding_closure_receipt_is_a_strict_allowlist(
    runtime: Any, contract: Any
) -> None:
    material = _changed_tuple_material(runtime, contract)
    assert set(material["closure_value"]) == set(runtime.INITIAL_CLOSURE_FIELDS)
    for extra in (
        "selected_event_count",
        "aliased_source_snapshot_sha256_and_bytes",
        "snapshot_set_sha256_and_bytes",
        "provisional_ready_core_sha256_and_bytes",
        "note",
        "source_locator",
    ):
        value = dict(material["closure_value"])
        value[extra] = 1 if extra.endswith("_count") else "x"
        value["resolution_id"] = runtime.derive_slot_resolution_id(value)
        _failure(runtime, _validate_initial_closure, runtime, contract, material, value)
    for missing in runtime.INITIAL_CLOSURE_FIELDS:
        value = dict(material["closure_value"])
        del value[missing]
        _failure(runtime, _validate_initial_closure, runtime, contract, material, value)


def test_pre_binding_closure_rejects_duplicate_keys_and_noncanonical_bytes(
    runtime: Any, contract: Any
) -> None:
    material = _changed_tuple_material(runtime, contract)
    canonical = _raw(runtime, material["closure_value"])
    arguments = {
        "contract": contract,
        "attempt": material["marker"],
        "observed_services": material["observed_services"],
        "observed_services_raw": material["observed_services"].raw,
        "observed_binding": material["observed_binding"],
        "observed_source_binding_core_raw": material["observed_binding"].core.raw,
        "runtime_observer_identity": material["observer"],
        "expected_previous_ledger_sha256": material["closure_previous"],
    }
    duplicate = canonical.replace(
        b'"slot_index":0', b'"slot_index":0,"slot_index":0', 1
    )
    _failure(
        runtime,
        runtime.validate_initial_runtime_change_closure,
        duplicate,
        **arguments,
    )
    pretty = json.dumps(material["closure_value"], sort_keys=True, indent=2).encode()
    _failure(
        runtime, runtime.validate_initial_runtime_change_closure, pretty, **arguments
    )
    _failure(
        runtime,
        runtime.validate_initial_runtime_change_closure,
        canonical + b"\n",
        **arguments,
    )


def test_pre_binding_closure_rejects_mismatch_kind_substitution(
    runtime: Any, contract: Any
) -> None:
    material = _changed_tuple_material(runtime, contract)
    for kinds in (
        [],
        ["source-binding-change"],
        ["active-services-state-change", "source-binding-change"],
        ["active-services-state-change", "active-services-state-change"],
        ["ACTIVE-SERVICES-STATE-CHANGE"],
        "active-services-state-change",
    ):
        value = dict(material["closure_value"])
        value["mismatch_kinds"] = kinds
        value["resolution_id"] = runtime.derive_slot_resolution_id(value)
        _failure(runtime, _validate_initial_closure, runtime, contract, material, value)


def test_pre_binding_closure_rejects_count_or_snapshot_smuggling(
    runtime: Any, contract: Any
) -> None:
    material = _changed_tuple_material(runtime, contract)
    for field in (
        "observed_active_services_state_sha256_and_bytes",
        "observed_source_binding_core_sha256_and_bytes",
    ):
        value = dict(material["closure_value"])
        value[field] = {
            **value[field],
            "selected_event_count": 7,
        }
        value["resolution_id"] = runtime.derive_slot_resolution_id(value)
        _failure(runtime, _validate_initial_closure, runtime, contract, material, value)


def test_pre_binding_closure_is_pinned_to_slot_zero_schedule(
    runtime: Any, contract: Any
) -> None:
    material = _changed_tuple_material(runtime, contract)
    slot = runtime.slot_times(0)
    assert slot.scheduled_at == "2026-08-17T00:00:00.000000Z"
    assert slot.grace_deadline_at == "2026-08-17T06:00:00.000000Z"
    drifts = (
        ("slot_index", 1),
        ("slot_index", True),
        ("slot_index", 28),
        ("scheduled_at", "2026-08-18T00:00:00.000000Z"),
        ("scheduled_at", "2026-08-17T00:00:00Z"),
        ("grace_deadline_at", "2026-08-17T06:00:00.000001Z"),
        ("grace_deadline_at", runtime.slot_times(1).grace_deadline_at),
        ("launched_at", "2026-08-16T23:59:59.999999Z"),
        ("validated_at", "2026-08-17T06:00:00.000000Z"),
        ("validated_at", _time(-1)),
        ("segment_index", 1),
        ("segment_id", "a" * 64),
    )
    for field, replacement in drifts:
        value = dict(material["closure_value"])
        value[field] = replacement
        value["resolution_id"] = runtime.derive_slot_resolution_id(value)
        _failure(runtime, _validate_initial_closure, runtime, contract, material, value)


def test_pre_binding_closure_window_is_lower_inclusive_and_upper_exclusive(
    runtime: Any, contract: Any
) -> None:
    material = _changed_tuple_material(runtime, contract)
    value = dict(material["closure_value"])
    value["launched_at"] = runtime.slot_times(0).scheduled_at
    value["validated_at"] = "2026-08-17T05:59:59.999999Z"
    value["resolution_id"] = runtime.derive_slot_resolution_id(value)
    accepted = _validate_initial_closure(runtime, contract, material, value)
    assert accepted.slot_index == 0
    value = dict(value)
    value["validated_at"] = "2026-08-17T06:00:00.000000Z"
    value["resolution_id"] = runtime.derive_slot_resolution_id(value)
    _failure(runtime, _validate_initial_closure, runtime, contract, material, value)


def test_pre_binding_closure_rejects_a_forged_resolution_or_self_identity(
    runtime: Any, contract: Any
) -> None:
    material = _changed_tuple_material(runtime, contract)
    value = dict(material["closure_value"])
    value["resolution_id"] = "b" * 64
    _failure(runtime, _validate_initial_closure, runtime, contract, material, value)

    closure = material["closure"]
    forged = replace(closure, identity=runtime.hash_and_bytes(b"forged"))
    _failure(runtime, runtime._require_closure_wrapper_consistent, contract, forged)
    _failure(
        runtime,
        runtime._require_closure_wrapper_consistent,
        contract,
        replace(closure, resolution_id="c" * 64),
    )
    _failure(
        runtime,
        runtime._require_closure_wrapper_consistent,
        contract,
        replace(closure, segment_id="d" * 64),
    )
    _failure(
        runtime,
        runtime._require_closure_wrapper_consistent,
        contract,
        replace(closure, slot_index=1),
    )
    _failure(
        runtime,
        runtime._require_closure_wrapper_consistent,
        contract,
        replace(closure, attempt_identity=runtime.hash_and_bytes(b"other")),
    )
    _failure(
        runtime,
        runtime._require_closure_wrapper_consistent,
        contract,
        replace(closure, observer_identity=runtime.hash_and_bytes(b"other")),
    )


def test_unbound_predecessor_wrapper_cannot_be_forged(
    runtime: Any, contract: Any
) -> None:
    predecessor = runtime.make_unbound_watermark_predecessor(contract)
    runtime._require_unbound_predecessor_consistent(contract, predecessor)
    observed, _value = _synthetic_services(runtime)
    for forged in (
        replace(predecessor, segment_index=1),
        replace(predecessor, segment_id="a" * 64),
        replace(predecessor, lower_bound_exclusive_at=_time(0)),
        replace(predecessor, services=observed),
        replace(predecessor, attestation_identity=runtime.hash_and_bytes(b"x")),
    ):
        _failure(
            runtime, runtime._require_unbound_predecessor_consistent, contract, forged
        )


def test_pre_binding_closure_rejects_a_forged_frozen_contract(
    runtime: Any, contract: Any
) -> None:
    material = _changed_tuple_material(runtime, contract)
    observed, _value = _synthetic_services(runtime)
    for forged in (
        replace(contract, initial_segment_id="a" * 64),
        replace(contract, release_effective_at=_time(0)),
        replace(contract, initial_services=observed),
        replace(contract, watermark_identity=runtime.hash_and_bytes(b"x")),
        replace(contract, plan_identity=runtime.hash_and_bytes(b"x")),
    ):
        _failure(
            runtime,
            _validate_initial_closure,
            runtime,
            forged,
            material,
            material["closure_value"],
        )


def test_retired_namespace_and_schema_version_cannot_replay_into_v4(
    runtime: Any, contract: Any
) -> None:
    assert runtime.NAMESPACE == "confirmatory-holdout-v4"
    assert runtime.SCHEMA_VERSION == 4
    assert runtime.RETIRED_NAMESPACES == (
        "confirmatory-holdout-v2",
        "confirmatory-holdout-v3",
    )
    material = _changed_tuple_material(runtime, contract)
    for field, replacement in (
        ("namespace", "confirmatory-holdout-v3"),
        ("namespace", "confirmatory-holdout-v2"),
        ("schema_version", 3),
        ("schema_version", 2),
        ("receipt_kind", "slot-segment-closed"),
        ("status", "pass"),
    ):
        value = dict(material["closure_value"])
        value[field] = replacement
        value["resolution_id"] = runtime.derive_slot_resolution_id(value)
        _failure(runtime, _validate_initial_closure, runtime, contract, material, value)


def test_every_active_domain_separator_is_v4_scoped(runtime: Any) -> None:
    separators = (
        runtime.RUNTIME_SEGMENT_DOMAIN,
        runtime.SOURCE_DATABASE_IDENTITY_DOMAIN,
        runtime.ALIAS_SERVICE_DATABASE_BINDING_DOMAIN,
        runtime.SOURCE_BINDING_CORE_DOMAIN,
        runtime.SLOT_RECEIPT_DOMAIN,
    )
    for separator in separators:
        assert separator.startswith(b"confirmatory-holdout-v4/")
        for retired in runtime.RETIRED_DOMAIN_PREFIXES:
            assert not separator.startswith(retired.encode("utf-8"))
    assert set(runtime.ALIAS_IDS.values()) == {
        "confirmatory-local-v4-ro",
        "confirmatory-alt-v4-ro",
    }


def test_retired_source_binding_core_bytes_are_not_v4_core_bytes(
    runtime: Any, contract: Any
) -> None:
    binding = _stable_binding(runtime, contract.initial_services)
    core_raw = binding.core.raw
    assert core_raw.startswith(b"confirmatory-holdout-v4/source-binding-core/v1\x00")
    retired = core_raw.replace(b"holdout-v4/source-binding", b"holdout-v3/source-binding", 1)
    _failure(runtime, runtime.validate_source_binding_core_bytes, retired)
    retired_alias = core_raw.replace(b"confirmatory-local-v4-ro", b"confirmatory-local-v3-ro")
    _failure(runtime, runtime.validate_source_binding_core_bytes, retired_alias)


def test_initial_segment_id_is_the_v4_value_not_the_retired_one(
    runtime: Any, contract: Any
) -> None:
    assert contract.initial_segment_id == runtime.INITIAL_SEGMENT_ID
    assert (
        runtime.derive_initial_segment_id(
            attestation_source_sha256=contract.watermark_identity.sha256,
            lower_bound_exclusive_at=contract.release_effective_at,
        )
        == runtime.INITIAL_SEGMENT_ID
    )
    assert (
        runtime.INITIAL_SEGMENT_ID
        != "b084e79959146e10397b725cae86d8edc9425cdcca21f464c9f7efac584ca53b"
    )


def test_successor_opens_from_the_typed_pre_binding_predecessor(
    runtime: Any, contract: Any
) -> None:
    material = _successor_material(runtime, contract)
    successor = _validate_successor(
        runtime, contract, material, material["segment_value"]
    )
    assert type(successor) is runtime.ActiveSegment
    assert successor.segment_index == 1
    assert successor.lower_bound_exclusive_at == _time(3.3)
    assert type(successor.predecessor_segment) is runtime.UnboundWatermarkPredecessor
    assert type(successor.predecessor_closure) is runtime.InitialRuntimeChangeClosure
    # The successor carries its own fresh ceremony, not the closure's evidence.
    assert successor.source_binding_attestation.attempt.identity != material[
        "closure"
    ].attempt_identity
    assert successor.source_binding_attestation.pre_observed_at > material[
        "closure"
    ].validated_at
    runtime._require_active_segment_consistent(contract, successor)
    assert (
        _receipt_utc(runtime, successor.lower_bound_exclusive_at)
        > _receipt_utc(runtime, contract.release_effective_at)
    )


def test_successor_uses_a_fresh_ceremony_not_the_closure_observation(
    runtime: Any, contract: Any
) -> None:
    """A runtime that changed again after the closure still attests correctly."""

    changed_again = _synthetic_services(runtime, generation=1)
    material = _successor_material(runtime, contract, services_override=changed_again)
    assert (
        material["successor_services"].identity
        != material["observed_services"].identity
    )
    successor = _validate_successor(
        runtime, contract, material, material["segment_value"]
    )
    assert successor.services.identity == material["successor_services"].identity
    receipt = json.loads(material["closure"].raw.decode("utf-8"))
    # The closure still records only what it observed; nothing was rewritten.
    assert (
        receipt["observed_active_services_state_sha256_and_bytes"]["sha256"]
        == material["observed_services"].identity.sha256
    )
    assert (
        receipt["observed_active_services_state_sha256_and_bytes"]["sha256"]
        != successor.services.identity.sha256
    )


def test_successor_rejects_reuse_of_the_consumed_slot_zero_marker(
    runtime: Any, contract: Any
) -> None:
    material = _successor_material(runtime, contract)
    _failure(
        runtime,
        _validate_successor,
        runtime,
        contract,
        material,
        material["segment_value"],
        attempt=material["marker"],
    )
    # And the marker validator itself refuses to mint a successor attempt whose
    # scope or predecessor linkage belongs to the consumed slot-0 ceremony.
    _failure(
        runtime,
        runtime.validate_attestation_attempt_marker,
        _raw(runtime, material["marker_value"]),
        contract=contract,
        runtime_observer_identity=material["observer"],
        expected_previous_ledger_sha256=material["marker_value"][
            "previous_ledger_entry_sha256"
        ],
        predecessor_closure=material["closure"],
    )


def test_successor_ceremony_must_start_strictly_after_the_closure(
    runtime: Any, contract: Any
) -> None:
    material = _successor_material(runtime, contract)
    value = dict(material["segment_value"])
    value["pre_launcher_at"] = _time(2)
    value["segment_id"] = runtime.derive_successor_segment_id(value)
    _failure(runtime, _validate_successor, runtime, contract, material, value)


def test_successor_rejects_a_tuple_equal_to_the_watermark(
    runtime: Any, contract: Any
) -> None:
    material = _successor_material(runtime, contract)
    watermark_value = json.loads(contract.initial_services.raw.decode("utf-8"))
    _failure(
        runtime,
        _validate_successor,
        runtime,
        contract,
        material,
        material["segment_value"],
        pre_services_observation=copy.deepcopy(watermark_value),
        post_services_observation=copy.deepcopy(watermark_value),
        complete_services_raw=contract.initial_services.raw,
    )


def test_successor_rejects_a_mismatched_predecessor_and_closure_pairing(
    runtime: Any, contract: Any
) -> None:
    material = _successor_material(runtime, contract)
    same = _same_tuple_material(runtime, contract)
    # An ActiveSegment may not be paired with the pre-binding closure.
    _failure(
        runtime,
        _validate_successor,
        runtime,
        contract,
        material,
        material["segment_value"],
        predecessor=same["active"],
    )
    _failure(
        runtime,
        runtime._require_predecessor_pair_consistent,
        same["active"],
        material["closure"],
    )
    _failure(
        runtime,
        runtime._require_predecessor_pair_consistent,
        material["predecessor"],
        None,
    )


def test_successor_rejects_boundary_and_envelope_drift(
    runtime: Any, contract: Any
) -> None:
    material = _successor_material(runtime, contract)
    drifts = (
        ("boundary_at", _time(3.2)),
        ("lower_bound_exclusive_at", _time(3.4)),
        ("post_phase_started_at", _time(3.54)),
        ("post_launcher_at", _time(3.3)),
        ("predecessor_closure_sha256", "a" * 64),
        ("segment_index", 2),
        ("active_services_state_sha256", "b" * 64),
    )
    for field, replacement in drifts:
        value = dict(material["segment_value"])
        value[field] = replacement
        value["segment_id"] = runtime.derive_successor_segment_id(value)
        _failure(runtime, _validate_successor, runtime, contract, material, value)
    forged = dict(material["segment_value"])
    forged["segment_id"] = "c" * 64
    _failure(runtime, _validate_successor, runtime, contract, material, forged)


def test_successor_segment_carries_nothing_forward_from_the_closed_segment(
    runtime: Any, contract: Any
) -> None:
    material = _successor_material(runtime, contract)
    successor = _validate_successor(
        runtime, contract, material, material["segment_value"]
    )
    predecessor = successor.predecessor_segment
    assert successor.services.identity != predecessor.services.identity
    assert successor.attestation_identity != predecessor.attestation_identity
    assert successor.attestation_raw is not None
    assert successor.attestation_previous_ledger_sha256 == material["segment_previous"]
    assert successor.source_binding.core.active_services_state_sha256 == (
        successor.services.identity.sha256
    )


def _post_binding_closure_value(
    runtime: Any, contract: Any, material: dict[str, Any], observed: dict[str, Any]
) -> dict[str, Any]:
    value = {
        "schema_version": 4,
        "namespace": "confirmatory-holdout-v4",
        "receipt_kind": "slot-segment-closed",
        "status": "segment-closed",
        "resolution_id": "0" * 64,
        "slot_index": 0,
        "scheduled_at": runtime.slot_times(0).scheduled_at,
        "grace_deadline_at": runtime.slot_times(0).grace_deadline_at,
        "launched_at": _time(1),
        "validated_at": _time(2),
        "segment_index": 0,
        "segment_id": material["active"].segment_id,
        "mismatch_kinds": [
            "active-services-state-change",
            "source-binding-change",
        ],
        "prior_active_services_state_sha256_and_bytes": (
            material["active"].services.identity.as_dict()
        ),
        "observed_active_services_state_sha256_and_bytes": (
            observed["services"].identity.as_dict()
        ),
        "prior_source_binding_core_sha256_and_bytes": (
            material["active"].source_binding.core.identity.as_dict()
        ),
        "observed_source_binding_core_sha256_and_bytes": (
            observed["binding"].core.identity.as_dict()
        ),
        "attestation_sha256_and_bytes": (
            material["active"].attestation_identity.as_dict()
        ),
        "previous_ledger_entry_sha256": "7" * 64,
        "runtime_observer_sha256_and_bytes": material["observer"].as_dict(),
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
    }
    value["resolution_id"] = runtime.derive_slot_resolution_id(value)
    return value


def _post_binding_material(runtime: Any, contract: Any) -> dict[str, Any]:
    material = _same_tuple_material(runtime, contract)
    services, _value = _synthetic_services(runtime)
    binding = _stable_binding(runtime, services, generation=1)
    observed = {"services": services, "binding": binding}
    value = _post_binding_closure_value(runtime, contract, material, observed)
    closure = runtime.validate_segment_closure(
        _raw(runtime, value),
        contract=contract,
        active_segment=material["active"],
        prior_services_raw=material["active"].services.raw,
        observed_services_raw=services.raw,
        prior_source_binding_core_raw=material["active"].source_binding.core.raw,
        observed_source_binding_core_raw=binding.core.raw,
        observed_binding=binding,
        runtime_observer_identity=material["observer"],
        expected_previous_ledger_sha256=value["previous_ledger_entry_sha256"],
    )
    return {**material, **observed, "closure_value": value, "closure": closure}


def test_post_binding_two_core_closure_still_works(runtime: Any, contract: Any) -> None:
    material = _post_binding_material(runtime, contract)
    closure = material["closure"]
    assert type(closure) is runtime.ValidatedClosure
    assert closure.slot_index == 0
    receipt = json.loads(closure.raw.decode("utf-8"))
    assert receipt["prior_source_binding_core_sha256_and_bytes"] == (
        material["active"].source_binding.core.identity.as_dict()
    )
    assert receipt["mismatch_kinds"] == [
        "active-services-state-change",
        "source-binding-change",
    ]


def test_the_two_closure_schemas_cannot_impersonate_each_other(
    runtime: Any, contract: Any
) -> None:
    pre = _changed_tuple_material(runtime, contract)
    post = _post_binding_material(runtime, contract)

    # The pre-binding validator rejects the post-binding two-core receipt.
    _failure(
        runtime,
        runtime.validate_initial_runtime_change_closure,
        post["closure"].raw,
        contract=contract,
        attempt=pre["marker"],
        observed_services=post["services"],
        observed_services_raw=post["services"].raw,
        observed_binding=post["binding"],
        observed_source_binding_core_raw=post["binding"].core.raw,
        runtime_observer_identity=post["observer"],
        expected_previous_ledger_sha256=post["closure_value"][
            "previous_ledger_entry_sha256"
        ],
    )
    # The post-binding validator rejects the pre-binding receipt.
    _failure(
        runtime,
        runtime.validate_segment_closure,
        pre["closure"].raw,
        contract=contract,
        active_segment=post["active"],
        prior_services_raw=post["active"].services.raw,
        observed_services_raw=pre["observed_services"].raw,
        prior_source_binding_core_raw=post["active"].source_binding.core.raw,
        observed_source_binding_core_raw=pre["observed_binding"].core.raw,
        observed_binding=pre["observed_binding"],
        runtime_observer_identity=pre["observer"],
        expected_previous_ledger_sha256=pre["closure_previous"],
    )
    # And the wrapper types do not cross-validate either.
    _failure(
        runtime,
        runtime._require_initial_closure_wrapper_consistent,
        contract,
        post["closure"],
    )


def test_post_binding_closure_requires_an_active_segment(
    runtime: Any, contract: Any
) -> None:
    pre = _changed_tuple_material(runtime, contract)
    post = _post_binding_material(runtime, contract)
    # The typed pre-binding predecessor is not an ActiveSegment and is refused.
    _failure(
        runtime,
        runtime.validate_segment_closure,
        post["closure"].raw,
        contract=contract,
        active_segment=pre["predecessor"],
        prior_services_raw=contract.initial_services.raw,
        observed_services_raw=post["services"].raw,
        prior_source_binding_core_raw=post["active"].source_binding.core.raw,
        observed_source_binding_core_raw=post["binding"].core.raw,
        observed_binding=post["binding"],
        runtime_observer_identity=post["observer"],
        expected_previous_ledger_sha256=post["closure_value"][
            "previous_ledger_entry_sha256"
        ],
    )


def test_count_guard_accepts_a_successor_grown_from_the_unbound_predecessor(
    runtime: Any, contract: Any
) -> None:
    material = _successor_material(runtime, contract)
    successor = _validate_successor(
        runtime, contract, material, material["segment_value"]
    )
    services = material["successor_services"]
    binding = material["successor_binding"]
    runtime.validate_active_segment_observations(
        successor,
        services,
        binding,
        services,
        binding,
        contract=contract,
    )
    other, _value = _synthetic_services(runtime, generation=1)
    _failure(
        runtime,
        runtime.validate_active_segment_observations,
        successor,
        other,
        _stable_binding(runtime, other, generation=3),
        services,
        binding,
        contract=contract,
    )


def test_integrity_failure_never_carries_evidence(runtime: Any, contract: Any) -> None:
    material = _changed_tuple_material(runtime, contract)
    secret = material["observed_binding"].core.raw
    value = dict(material["closure_value"])
    value["observed_source_binding_core_sha256_and_bytes"] = _identity("a", 9)
    value["resolution_id"] = runtime.derive_slot_resolution_id(value)
    with pytest.raises(runtime.IntegrityFailure) as caught:
        _validate_initial_closure(runtime, contract, material, value)
    assert caught.value.args == ()
    assert str(caught.value) == ""
    assert repr(caught.value) == "IntegrityFailure()"
    assert caught.value.__cause__ is None
    assert secret not in repr(caught.value).encode("utf-8")


def test_receipts_carry_no_locator_row_or_query_material(
    runtime: Any, contract: Any
) -> None:
    material = _successor_material(runtime, contract)
    successor = _validate_successor(
        runtime, contract, material, material["segment_value"]
    )
    blobs = [
        material["closure"].raw,
        material["marker"].raw,
        successor.source_binding_attestation.raw,
        successor.attestation_raw,
    ]
    forbidden = (
        b"/home/",
        b"sqlite",
        b".sqlite3",
        b"ssh://",
        b"SELECT",
        b"filesystem_uuid",
        b"statx_inode_uint64",
        b"kernel_uid_uint32",
        b"ssh_host_key_sha256_base64",
    )
    for blob in blobs:
        for token in forbidden:
            assert token not in blob
    # The retained private core keeps only derived digests, never raw inputs.
    core = material["observed_binding"].core.raw
    for token in (b"filesystem_uuid", b"statx_inode_uint64", b"ssh-ed25519"):
        assert token not in core


def test_frozen_plan_and_watermark_load_only_at_pinned_hashes(
    runtime: Any, tmp_path: Path
) -> None:
    assert FROZEN_PLAN.is_file() and FROZEN_WATERMARK.is_file()
    contract = runtime.load_frozen_contract()
    assert contract.plan_identity.sha256 == runtime.PINNED_PLAN_SHA256
    assert contract.plan_identity.bytes == runtime.PINNED_PLAN_BYTES
    assert contract.watermark_identity.sha256 == runtime.PINNED_WATERMARK_SHA256

    tampered_plan = tmp_path / "analysis-plan.json"
    plan = json.loads(FROZEN_PLAN.read_bytes().decode("utf-8"))
    plan["namespace"] = "confirmatory-holdout-v3"
    tampered_plan.write_bytes(json.dumps(plan).encode("utf-8"))
    _failure(runtime, runtime.load_frozen_contract, plan_path=tampered_plan)

    truncated = tmp_path / "control-watermark.json"
    truncated.write_bytes(FROZEN_WATERMARK.read_bytes()[:-1])
    _failure(runtime, runtime.load_frozen_contract, watermark_path=truncated)


def test_frozen_plan_pins_the_bootstrap_branch_schemas(runtime: Any) -> None:
    plan = json.loads(FROZEN_PLAN.read_bytes().decode("utf-8"))
    segments = plan["runtime_segments"]
    schemas = plan["operational_receipt_schemas"]
    assert segments["initial_runtime_change_closure_allowlist"] == list(
        runtime.INITIAL_CLOSURE_FIELDS
    )
    assert segments["slot_segment_closure_resolution_allowlist"] == list(
        runtime.CLOSURE_FIELDS
    )
    assert segments["segment_attestation_allowlist"] == list(
        runtime.SUCCESSOR_ATTESTATION_FIELDS
    )
    assert segments["initial_bootstrap_branches_mutually_exclusive"] is True
    initial = schemas["initial_runtime_change_closure_resolution"]
    assert initial["prior_source_binding_core_member_present"] is False
    assert initial["prior_source_binding_core_null_or_sentinel_allowed"] is False
    assert initial["requires_active_segment_object"] is False
    assert initial["fabricated_predecessor_active_segment_allowed"] is False
    assert (
        "prior_source_binding_core_sha256_and_bytes" not in initial["fields_exactly"]
    )
    assert (
        schemas["slot_segment_closure_resolution"][
            "valid_before_initial_source_binding_attestation"
        ]
        is False
    )


def test_plan_validator_rejects_a_relaxed_bootstrap_schema(
    runtime: Any, tmp_path: Path
) -> None:
    mutations = (
        ("prior_source_binding_core_member_present", True),
        ("prior_source_binding_core_null_or_sentinel_allowed", True),
        ("requires_active_segment_object", True),
        ("fabricated_predecessor_active_segment_allowed", True),
        ("valid_after_initial_source_binding_attestation", True),
        ("count_free", False),
        ("next_probe_slot_index", 0),
    )
    raw = FROZEN_PLAN.read_bytes().decode("utf-8")
    for index, (field, replacement) in enumerate(mutations):
        plan = json.loads(raw)
        plan["operational_receipt_schemas"][
            "initial_runtime_change_closure_resolution"
        ][field] = replacement
        target = tmp_path / f"plan-{index}.json"
        target.write_bytes(json.dumps(plan).encode("utf-8"))
        _failure(runtime, runtime.load_frozen_contract, plan_path=target)


def test_canonical_json_is_compact_sorted_and_duplicate_safe(runtime: Any) -> None:
    value = {"b": 1, "a": [1, {"d": 2, "c": 3}]}
    raw = runtime.canonical_json_bytes(value)
    assert raw == b'{"a":[1,{"c":3,"d":2}],"b":1}'
    assert runtime.load_canonical_json_bytes(raw, expected=dict) == value
    _failure(runtime, runtime.load_json_bytes, b'{"a":1,"a":2}')
    _failure(runtime, runtime.load_json_bytes, b'{"a":NaN}')
    _failure(runtime, runtime.load_json_bytes, b'{"a":Infinity}')
    _failure(runtime, runtime.load_canonical_json_bytes, b'{"b":1,"a":2}', expected=dict)
    _failure(runtime, runtime.load_json_bytes, b"\xff\xfe")


def test_utc_grammar_rejects_iso_aliases_and_offsets(runtime: Any) -> None:
    assert runtime.canonical_utc(
        runtime.parse_utc("2026-08-17T00:00:00Z")
    ) == "2026-08-17T00:00:00.000000Z"
    for bad in (
        "2026-08-17T00:00:00+00:00",
        "2026-08-17T00:00:00z",
        "2026-08-17 00:00:00Z",
        "2026-08-17T00:00:60Z",
        "2026-02-30T00:00:00Z",
        "2026-08-17T00:00:00.1234567Z",
        "2026-08-17T00:00:00.Z",
        20260817,
    ):
        _failure(runtime, runtime.parse_utc, bad)
    # Receipt instants demand exactly six fractional digits.
    _failure(runtime, runtime.parse_utc, "2026-08-17T00:00:00Z", receipt=True)


def test_schedule_lattice_is_exact_and_reaches_the_fixed_horizon(runtime: Any) -> None:
    assert runtime.SLOT_COUNT == 29
    assert runtime.slot_times(0).scheduled_at == "2026-08-17T00:00:00.000000Z"
    assert runtime.slot_times(28).scheduled_at == "2026-09-14T00:00:00.000000Z"
    assert runtime.absolute_horizon_at() == "2026-09-14T06:00:00.000000Z"
    _failure(runtime, runtime.slot_times, 29)
    _failure(runtime, runtime.slot_times, -1)
    _failure(runtime, runtime.slot_times, True)


def test_regular_file_reader_refuses_symlinks_and_wrong_hashes(
    runtime: Any, tmp_path: Path
) -> None:
    target = tmp_path / "real.json"
    target.write_bytes(b"{}")
    identity = runtime.hash_and_bytes(b"{}")
    assert runtime.read_regular_file(target, expected=identity) == b"{}"
    link = tmp_path / "link.json"
    link.symlink_to(target)
    _failure(runtime, runtime.read_regular_file, link)
    _failure(
        runtime,
        runtime.read_regular_file,
        target,
        expected=runtime.hash_and_bytes(b"[]"),
    )
    _failure(runtime, runtime.read_regular_file, tmp_path / ".." / "real.json")


def test_slot_zero_marker_chain_is_pinned_to_the_first_slot(
    runtime: Any, contract: Any
) -> None:
    observer = _observer(runtime)
    base = _slot_zero_marker_value(runtime, contract, observer)
    drifts = (
        ("attempt_scope", "successor-segment"),
        ("segment_index", 1),
        ("slot_index_or_null", None),
        ("slot_index_or_null", 1),
        ("predecessor_closure_sha256_or_null", "a" * 64),
        ("authorized_at", runtime.slot_times(1).scheduled_at),
        ("start_deadline_at", runtime.slot_times(1).grace_deadline_at),
        ("written_at", "2026-08-16T23:59:59.999999Z"),
    )
    for field, replacement in drifts:
        value = dict(base)
        value[field] = replacement
        _failure(
            runtime,
            runtime.validate_attestation_attempt_marker,
            _raw(runtime, value),
            contract=contract,
            runtime_observer_identity=observer,
            expected_previous_ledger_sha256=base["previous_ledger_entry_sha256"],
            initial_probe_launched_at=_time(0),
        )


def test_pre_binding_closure_requires_its_own_slot_zero_marker(
    runtime: Any, contract: Any
) -> None:
    material = _changed_tuple_material(runtime, contract)
    other_observer = runtime.hash_and_bytes(b"other-observer-v4")
    other_value = _slot_zero_marker_value(runtime, contract, other_observer)
    other_marker = runtime.validate_attestation_attempt_marker(
        _raw(runtime, other_value),
        contract=contract,
        runtime_observer_identity=other_observer,
        expected_previous_ledger_sha256=other_value["previous_ledger_entry_sha256"],
        initial_probe_launched_at=_time(0),
    )
    _failure(
        runtime,
        _validate_initial_closure,
        runtime,
        contract,
        material,
        material["closure_value"],
        attempt=other_marker,
    )
    # A marker written after the closure validated cannot have authorized it.
    late = dict(material["marker_value"])
    late["written_at"] = _time(3)
    late_marker = runtime.validate_attestation_attempt_marker(
        _raw(runtime, late),
        contract=contract,
        runtime_observer_identity=material["observer"],
        expected_previous_ledger_sha256=late["previous_ledger_entry_sha256"],
        initial_probe_launched_at=_time(0),
    )
    value = dict(material["closure_value"])
    value["attestation_attempt_marker_sha256_and_bytes"] = (
        late_marker.identity.as_dict()
    )
    value["resolution_id"] = runtime.derive_slot_resolution_id(value)
    _failure(
        runtime,
        _validate_initial_closure,
        runtime,
        contract,
        material,
        value,
        attempt=late_marker,
    )


def test_pre_binding_closure_rejects_retained_object_substitution(
    runtime: Any, contract: Any
) -> None:
    material = _changed_tuple_material(runtime, contract)
    other, _value = _synthetic_services(runtime, generation=1)
    other_binding = _stable_binding(runtime, other, generation=2)
    _failure(
        runtime,
        _validate_initial_closure,
        runtime,
        contract,
        material,
        material["closure_value"],
        observed_services_raw=other.raw,
    )
    _failure(
        runtime,
        _validate_initial_closure,
        runtime,
        contract,
        material,
        material["closure_value"],
        observed_source_binding_core_raw=other_binding.core.raw,
    )
    _failure(
        runtime,
        _validate_initial_closure,
        runtime,
        contract,
        material,
        material["closure_value"],
        observed_services=other,
    )
    _failure(
        runtime,
        _validate_initial_closure,
        runtime,
        contract,
        material,
        material["closure_value"],
        expected_previous_ledger_sha256="e" * 64,
    )


def test_pre_binding_api_has_no_active_segment_or_prior_core_surface(
    runtime: Any,
) -> None:
    import inspect

    closure_parameters = set(
        inspect.signature(runtime.validate_initial_runtime_change_closure).parameters
    )
    assert "active_segment" not in closure_parameters
    assert "prior_source_binding_core_raw" not in closure_parameters
    assert "prior_services_raw" not in closure_parameters
    assert not any("prior" in name for name in closure_parameters)

    transition_parameters = set(
        inspect.signature(runtime.validate_pre_binding_transition).parameters
    )
    assert transition_parameters == {"watermark_services", "observed_services"}

    predecessor_fields = {
        field.name
        for field in runtime.UnboundWatermarkPredecessor.__dataclass_fields__.values()
    }
    assert "source_binding" not in predecessor_fields
    assert "source_binding_attestation" not in predecessor_fields
    assert predecessor_fields == {
        "segment_index",
        "segment_id",
        "lower_bound_exclusive_at",
        "services",
        "attestation_identity",
    }
    assert runtime.UnboundWatermarkPredecessor is not runtime.ActiveSegment
    assert not issubclass(runtime.UnboundWatermarkPredecessor, runtime.ActiveSegment)
