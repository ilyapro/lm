from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = (
    ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "confirmatory-holdout-v3"
)
PLAN_PATH = NAMESPACE / "analysis-plan.json"
WATERMARK_PATH = (
    ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "release-v1"
    / "control-watermark.json"
)
RELEASE_MANIFEST_PATH = WATERMARK_PATH.with_name("release-manifest.json")
ORIGINAL_MANIFEST_PATH = ROOT / "artifacts" / "animal-planet" / "manifest.json"
REPLACEMENT_MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "replacement-holdout"
    / "manifest.json"
)
V2_NAMESPACE = (
    ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "confirmatory-holdout-v2"
)
V2_PLAN_PATH = V2_NAMESPACE / "analysis-plan.json"

DOCUMENT_HASHES = {
    "README.md": "51988680542c2a8361df3a126420abb4ba32d51b52ee588889bdfe2dbe3987c3",
    "POLICY.md": "e5976cd3d6239dbf35cbd84495308c5a61b063e3d4c08fb0a6fc6fda2ff8a551",
    "analysis-plan.json": (
        "7c8af7f53bfa6edbd8146e5593e585327aeedf2408e066d2e47c9856056570a9"
    ),
}

AUTHORIZED_INPUTS = {
    "artifacts/animal-planet/evaluation/release-v1/release-manifest.json": (
        "dd1208d4f5196460428d7049fa9e33ca74ef71be77d074c2b479876653c61768",
        16970,
    ),
    "artifacts/animal-planet/evaluation/release-v1/control-watermark.json": (
        "a9e6e6a4dbd6178345e11e951e973a1af9c062fb50f8dd7bd51528d44217a86f",
        10553,
    ),
    "artifacts/animal-planet/evaluation/confirmatory-holdout-v2/attempt-note.json": (
        "436b32d4cccdb293a6c06249d22f338328b19e49c59f45120d2da4c70f1a2087",
        1699,
    ),
    "artifacts/animal-planet/evaluation/confirmatory-holdout-v2/README.md": (
        "4455e267979284ffbe09c704de6f51069e2fd733e3092bddbbc5f2f482ac90e4",
        3581,
    ),
    "artifacts/animal-planet/evaluation/confirmatory-holdout-v2/POLICY.md": (
        "096263ddf554dc014c8cd971e6129d47300bd1c0b9710da90a5749ff186a1afb",
        19239,
    ),
    "artifacts/animal-planet/evaluation/confirmatory-holdout-v2/analysis-plan.json": (
        "5f5f050330b97c8a35b0e76bd31cbf1b9dd73f4d1a2baa0175817c8201a0c653",
        35902,
    ),
    "artifacts/animal-planet/evaluation/confirmatory-holdout-v2/repair-design.md": (
        "76d34d33b7be57230ef0988662060fe70eea229f9b07f2cd5b8a2354600a79f5",
        68799,
    ),
    "scripts/ap_confirmatory_readiness.py": (
        "ce1875bf4c442bb481cc254e3b145fb45625b483d4e98644d9365d56cfa017ba",
        54274,
    ),
    "artifacts/animal-planet/manifest.json": (
        "3f1a6a87a4d34e411f06e50161f9203afc3c4992dfc973d4f544453e8bbfde48",
        34934,
    ),
    "artifacts/animal-planet/corpus/splits.json": (
        "66f04caa41e0302fd7b8582f54197a396a2bb8483ddd543e564740a8e4c3a7b7",
        4040,
    ),
    "artifacts/animal-planet/corpus/dev.jsonl": (
        "605e4f3fdc91817788a40d0dbbabffb16b86ef98ad28020c20379ef249cc9a36",
        5638050,
    ),
    "artifacts/animal-planet/evaluation/replacement-holdout/manifest.json": (
        "fd36c972b049c296acbd2537f9af8f1db8d7db725f188c5dd90d27ab9aa3ab83",
        5537,
    ),
}

V2_FROZEN_FILES = {
    "POLICY.md": (
        V2_NAMESPACE / "POLICY.md",
        "096263ddf554dc014c8cd971e6129d47300bd1c0b9710da90a5749ff186a1afb",
        19239,
    ),
    "README.md": (
        V2_NAMESPACE / "README.md",
        "4455e267979284ffbe09c704de6f51069e2fd733e3092bddbbc5f2f482ac90e4",
        3581,
    ),
    "analysis-plan.json": (
        V2_PLAN_PATH,
        "5f5f050330b97c8a35b0e76bd31cbf1b9dd73f4d1a2baa0175817c8201a0c653",
        35902,
    ),
    "repair-design.md": (
        V2_NAMESPACE / "repair-design.md",
        "76d34d33b7be57230ef0988662060fe70eea229f9b07f2cd5b8a2354600a79f5",
        68799,
    ),
    "attempt-note.json": (
        V2_NAMESPACE / "attempt-note.json",
        "436b32d4cccdb293a6c06249d22f338328b19e49c59f45120d2da4c70f1a2087",
        1699,
    ),
    "scanner": (
        ROOT / "scripts" / "ap_confirmatory_readiness.py",
        "ce1875bf4c442bb481cc254e3b145fb45625b483d4e98644d9365d56cfa017ba",
        54274,
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
    )


def _plan() -> dict[str, Any]:
    return _load_json(PLAN_PATH)


def _utc(value: str) -> datetime:
    assert value.endswith("Z")
    parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    assert parsed.tzinfo == timezone.utc
    return parsed


def _numeric_leaves(value: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(value, dict):
        result: dict[tuple[str, ...], Any] = {}
        for key, member in value.items():
            result.update(_numeric_leaves(member, prefix + (key,)))
        return result
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {prefix: value}
    return {}


def test_v3_documents_are_nonempty_duplicate_free_and_byte_locked() -> None:
    assert set(DOCUMENT_HASHES) == {"README.md", "POLICY.md", "analysis-plan.json"}
    for name, expected_hash in DOCUMENT_HASHES.items():
        path = NAMESPACE / name
        assert path.is_file()
        assert not path.is_symlink()
        assert path.stat().st_size > 0
        assert _sha256(path) == expected_hash

    plan = _plan()
    assert plan["schema_version"] == 3
    assert plan["namespace"] == "confirmatory-holdout-v3"
    assert plan["status"] == "preregistered-source-unprobed-accruable-unsealed"
    assert plan["frozen"] is True
    assert plan["protocol_parent_commit"] == "ba2742b23ab2aa87a65118addbb1d35432e6eff3"
    assert plan["amendment_policy"].startswith("never in place;")


def test_preregistration_is_source_blind_and_uses_only_hash_pinned_inputs() -> None:
    prereg = _plan()["source_blind_preregistration"]
    assert prereg["availability_values_used"] is False
    assert prereg["calendar_derived_from_availability"] is False
    assert prereg["power_floors_derived_from_current_availability"] is False
    assert prereg["repair_implementation_started"] is False
    assert prereg["semantic_evaluator_started"] is False
    assert prereg["reads_before_freeze"] == {
        "candidate_private_rows": 0,
        "candidate_source_aggregate_queries": 0,
        "candidate_source_snapshot_captures": 0,
        "original_sealed_corpus_rows": 0,
        "replacement_sealed_corpus_rows": 0,
        "consumed_packet_verifier_invocations": 0,
        "v3_semantic_reads": 0,
    }

    declared = {
        entry["path"]: (entry["sha256"], entry["bytes"])
        for entry in prereg["allowed_inputs"]
    }
    assert declared == AUTHORIZED_INPUTS
    for relative_path, (expected_hash, expected_bytes) in AUTHORIZED_INPUTS.items():
        path = ROOT / relative_path
        assert path.is_file()
        assert path.stat().st_size == expected_bytes
        assert _sha256(path) == expected_hash


def test_release_control_watermark_is_exactly_pinned_without_deployment_claim() -> None:
    plan_control = _plan()["release_control_watermark"]
    watermark = _load_json(WATERMARK_PATH)
    assert plan_control["sha256"] == _sha256(WATERMARK_PATH)
    assert plan_control["bytes"] == WATERMARK_PATH.stat().st_size
    assert plan_control["release_effective_at"] == (
        watermark["selection"]["lower_bound_exclusive_at"]
    )
    assert plan_control["release_effective_at"] == "2026-08-14T15:03:46.793603Z"
    assert plan_control["release_effective_at_semantics"] == (
        "observed runtime-provenance boundary; not a deployment claim"
    )
    replay = plan_control["replay_code_control"]
    assert {key: replay[key] for key in ("commit", "tree", "role")} == watermark[
        "replay_code_control"
    ]
    assert replay["role"] == "replay_code_control_only"
    assert replay["measurement_arm"] == "gating-off"
    assert replay["deployment_required"] is False
    assert plan_control["release_manifest_reverse_pin"] == {
        "name": "release-manifest.json",
        "sha256": "dd1208d4f5196460428d7049fa9e33ca74ef71be77d074c2b479876653c61768",
        "bytes": 16970,
    }
    assert watermark["release_manifest"] == plan_control["release_manifest_reverse_pin"]
    assert _sha256(RELEASE_MANIFEST_PATH) == watermark["release_manifest"]["sha256"]
    assert RELEASE_MANIFEST_PATH.stat().st_size == watermark["release_manifest"]["bytes"]
    assert plan_control["identity_derivation"] == watermark["runtime_provenance"][
        "identity_derivation"
    ]


def test_initial_runtime_tuple_and_alias_bijection_are_frozen() -> None:
    plan = _plan()
    watermark = _load_json(WATERMARK_PATH)
    services = watermark["runtime_provenance"]["services"]
    assert len(services) == 2
    assert watermark["runtime_provenance"]["all_event_producing_services_attested"] is True
    sources = plan["sources"]
    assert sources["watermark_service_set_is_unaliased"] is True
    assert sources["public_alias_to_watermark_array_index_mapping"] is False
    for binding in sources["aliases"].values():
        assert "watermark_service_index" not in binding
        assert "initial_service_identity_sha256" not in binding
        assert "initial_boot_identity_sha256" not in binding

    watermark_pairs = [
        {
            "service_identity_sha256": service["service_identity_sha256"],
            "boot_identity_sha256": service["boot_identity_sha256"],
        }
        for service in services
    ]
    assert sources["initial_watermark_unordered_service_identity_pairs"] == watermark_pairs
    for service in services:
        assert len(service["serving_build"]["implementation"]) == 21
        assert all(
            set(identity) == {"bytes", "sha256"}
            for identity in service["serving_build"]["implementation"].values()
        )
        assert service["sanitized_configuration"] == {
            "bytes": 2649,
            "encoding": "canonical-json-utf8",
            "schema": "living-memory-effective-runtime-configuration-v1",
            "sha256": "36c839970c4b4f29d08e891b9193272dfbc026b664f05e19d8b79d5c3df2c3a0",
        }
        assert service["effective_legacy_repeat_controls"] == {
            "LM_RECALL_REPEAT_DROP_TRAILING_STUBS": False,
            "LM_RECALL_REPEAT_GATING": False,
        }

    observation = watermark["runtime_provenance"]["observation"]
    stable = plan["release_control_watermark"]["stable_observation"]
    assert stable["method"] == observation["method"]
    assert stable["pre_observed_at"] == observation["pre_observed_at"]
    assert stable["post_observed_at"] == observation["post_observed_at"]
    assert observation["pre_runtime_state_sha256"] == (
        observation["post_runtime_state_sha256"]
    )
    assert observation["pre_runtime_state_sha256"] == (
        stable["pre_watermark_services_state_sha256"]
    )
    assert observation["post_runtime_state_sha256"] == (
        stable["post_watermark_services_state_sha256"]
    )
    sorted_services = sorted(
        services,
        key=lambda service: (
            service["service_identity_sha256"],
            service["boot_identity_sha256"],
        ),
    )
    canonical_services = (
        json.dumps(
            sorted_services,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode()
    assert hashlib.sha256(canonical_services).hexdigest() == (
        plan["runtime_segments"]["initial_segment"][
            "watermark_services_state_sha256"
        ]
    )
    assert plan["runtime_segments"]["initial_segment"]["serving_commit"] != (
        plan["release_control_watermark"]["replay_code_control"]["commit"]
    )


def test_all_v2_files_and_authorities_remain_retired_and_byte_identical() -> None:
    for _, (path, expected_hash, expected_bytes) in V2_FROZEN_FILES.items():
        assert path.is_file()
        assert path.stat().st_size == expected_bytes
        assert _sha256(path) == expected_hash

    plan = _plan()
    retired = plan["lineage"]["retired_v2"]
    assert retired["state"] == "fatal-before-readiness"
    assert retired["scanner"]["launch_count"] == 1
    assert retired["scanner"]["launch_consumed"] is True
    assert retired["scanner"]["exit_code"] == 2
    assert retired["scanner"]["validator_valid_receipt"] is False
    assert retired["scanner"]["retry_allowed"] is False
    assert retired["authority_grants"] == []
    assert retired["all_authorities_unusable"] is True
    assert not (V2_NAMESPACE / "readiness.json").exists()
    assert not (V2_NAMESPACE / "manifest.json").exists()

    attempt = _load_json(V2_NAMESPACE / "attempt-note.json")
    assert attempt["authority"] == {
        "grants": [],
        "readiness_receipt": False,
        "packet": False,
        "reader": False,
        "repair_implementation": False,
    }
    assert attempt["attempt"]["retry_permitted"] is False

    release = _load_json(RELEASE_MANIFEST_PATH)
    pinned = {
        entry["path"]: (entry["sha256"], entry["bytes"])
        for entry in release["lineage"]["retired_confirmatory_v2"]["files"]
    }
    expected = {
        str(path.relative_to(ROOT)): (expected_hash, expected_bytes)
        for path, expected_hash, expected_bytes in V2_FROZEN_FILES.values()
    }
    assert pinned == expected


def test_public_manifest_metadata_proves_dev_pins_and_temporal_disjointness() -> None:
    plan = _plan()
    original = _load_json(ORIGINAL_MANIFEST_PATH)
    replacement = _load_json(REPLACEMENT_MANIFEST_PATH)
    dev = plan["identity"]["complete_frozen_dev_reference"]
    assert original["files"]["corpus/dev.jsonl"] == {
        "bytes": 5638050,
        "sha256": dev["public_dev_corpus"]["sha256_from_frozen_manifest"],
    }
    assert original["files"]["corpus/splits.json"] == {
        "bytes": 4040,
        "sha256": "66f04caa41e0302fd7b8582f54197a396a2bb8483ddd543e564740a8e4c3a7b7",
    }
    raw_identity = original["staging"]["exports_sha256"][
        "alt-db/recall_events.jsonl"
    ]
    assert raw_identity == {
        "bytes": dev["raw_identity_reference"]["bytes"],
        "rows": 1576,
        "sha256": dev["raw_identity_reference"]["sha256"],
    }
    upper = replacement["selection"]["predicate"]["created_at"]["lt"]
    assert upper == plan["selection"]["consumed_packet_disjointness"][
        "replacement_upper_bound_exclusive"
    ]
    assert _utc(upper) < _utc(
        plan["release_control_watermark"]["release_effective_at"]
    )
    original_ends = [original["windows"]["export"]["end"]]
    original_ends.extend(
        scope["span"][1]
        for scope in original["sources"]["local_db"]["holdout_scopes"].values()
    )
    original_max = max(original_ends, key=_utc)
    assert original_max == plan["selection"]["consumed_packet_disjointness"][
        "original_max_created_at_inclusive"
    ]
    assert _utc(original_max) < _utc(
        plan["release_control_watermark"]["release_effective_at"]
    )


def test_unchanged_repair_design_gets_only_a_fresh_post_seal_precondition() -> None:
    repair = _plan()["lineage"]["unchanged_repair_design"]
    assert repair["sha256"] == (
        "76d34d33b7be57230ef0988662060fe70eea229f9b07f2cd5b8a2354600a79f5"
    )
    assert _sha256(ROOT / repair["path"]) == repair["sha256"]
    assert repair["unchanged_parts"] == [
        "production repair invariants R1-R15",
        "defaults and rollback behavior",
        "storage and delivery design",
        "automatic-versus-agent state machine",
        "metric arithmetic and operators, metric meanings, and numeric point gates",
        "implementation ownership",
    ]
    assert len(repair["v3_evidence_protocol_supersessions"]) == 8
    supersessions = " ".join(repair["v3_evidence_protocol_supersessions"])
    for required in (
        "POLICY",
        "fixed-slot",
        "manifest-introduction",
        "reader identifiers",
        "measured-cohort",
        "retrieval_weights",
        "gating-off reference arm",
    ):
        assert required in supersessions
    freeze_commit = repair["design_freeze_commit"]
    assert freeze_commit == "fb0307f8f101cbd2c9bdbfe6a14a9e20d012d578"
    observed_tree = subprocess.check_output(
        ["git", "show", "-s", "--format=%T", freeze_commit], cwd=ROOT, text=True
    ).strip()
    assert observed_tree == repair["design_freeze_tree"]
    frozen_bytes = subprocess.check_output(
        ["git", "show", f"{freeze_commit}:{repair['path']}"], cwd=ROOT
    )
    assert len(frozen_bytes) == repair["bytes"]
    assert hashlib.sha256(frozen_bytes).hexdigest() == repair["sha256"]
    assert "strict Git ancestor" in repair["fresh_implementation_precondition"]
    assert repair["v2_can_satisfy_precondition"] is False


def test_v3_domains_and_reader_ids_are_fresh_and_exact() -> None:
    plan = _plan()
    domains = plan["domains"]
    active = {
        value
        for key, value in domains.items()
        if key.endswith("_utf8") and str(value).startswith("confirmatory-holdout-v3/")
    }
    assert active == {
        "confirmatory-holdout-v3/identity/v1",
        "confirmatory-holdout-v3/partition/v1",
        "confirmatory-holdout-v3/arm-order/v1",
        "confirmatory-holdout-v3/bootstrap/v1",
        "confirmatory-holdout-v3/segment/v1",
        "confirmatory-holdout-v3/slot-receipt/v1",
        "confirmatory-holdout-v3/accrual-ledger/v1",
        "confirmatory-holdout-v3/source-database-identity/v1",
        "confirmatory-holdout-v3/alias-service-database-binding/v1",
        "confirmatory-holdout-v3/source-binding-core/v1",
        "confirmatory-holdout-v3/snapshot-set/v1",
        "confirmatory-holdout-v3/ready-core/v1",
    }
    assert all(value.startswith("confirmatory-holdout-v3/") for value in active)
    assert not any("confirmatory-holdout-v2" in value for value in active)
    assert domains["retired_domain_prefix"] == "confirmatory-holdout-v2/"
    assert domains["retired_domains_reusable"] is False
    assert plan["identity"]["domain_separator_utf8"] == domains[
        "automatic_identity_utf8"
    ]
    assert plan["partition"]["domain_separator_utf8"] == domains["partition_utf8"]
    assert plan["measurement"]["arm_schedule"]["domain_separator_utf8"] == domains[
        "arm_order_utf8"
    ]
    assert plan["uncertainty"]["sampling"]["domain_separator_utf8"] == domains[
        "bootstrap_utf8"
    ]
    assert plan["sources"]["database_instance_identity_derivation"][
        "domain_separator_utf8"
    ] == domains["source_database_identity_utf8"]
    assert plan["sources"]["alias_service_database_binding_derivation"][
        "domain_separator_utf8"
    ] == domains["alias_service_database_binding_utf8"]
    assert plan["sources"]["source_binding_core"]["domain_separator_utf8"] == (
        domains["source_binding_core_utf8"]
    )

    authority = plan["authority"]
    retired = set(authority["retired_consumed_readers"]) | set(
        authority["retired_void_v2_readers"]
    )
    assert retired == {
        "shadow-eval",
        "replacement-holdout-eval",
        "confirmatory-shadow-v2-eval",
        "confirmatory-holdout-v2-eval",
    }
    assert authority["retired_v2_source_alias_ids"] == [
        "confirmatory-local-v2-ro",
        "confirmatory-alt-v2-ro",
    ]
    readers = authority["reserved_readers_exactly"]
    assert {reader["id"]: reader["partition"] for reader in readers} == {
        "confirmatory-shadow-v3-eval": "shadow",
        "confirmatory-holdout-v3-eval": "holdout",
    }
    assert retired.isdisjoint(reader["id"] for reader in readers)
    assert all(reader["max_process_launches"] == 1 for reader in readers)
    assert all(reader["max_semantic_passes"] == 1 for reader in readers)
    for derivation in (
        "initial_segment_id_preimage",
        "initial_segment_id_value",
        "successor_segment_id_preimage",
        "successor_segment_id_value",
        "slot_resolution_id_preimage",
        "slot_resolution_id_value",
        "ledger_genesis_preimage",
        "ledger_entry_preimage",
        "ledger_entry_value",
    ):
        assert domains[derivation]


def test_initial_segment_and_append_only_ledger_have_one_encoding() -> None:
    plan = _plan()
    domains = plan["domains"]
    initial = plan["runtime_segments"]["initial_segment"]
    segment_payload = {
        "segment_index": 0,
        "attestation_source_sha256": plan["release_control_watermark"]["sha256"],
        "lower_bound_exclusive_at": plan["release_control_watermark"][
            "release_effective_at"
        ],
    }
    compact = json.dumps(
        segment_payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    preimage = domains["runtime_segment_utf8"].encode() + b"\0" + compact
    assert hashlib.sha256(preimage).hexdigest() == initial["segment_id"]
    assert initial["segment_id"] == domains["initial_segment_id_value"]

    schemas = plan["operational_receipt_schemas"]
    ledger = schemas["ledger_entry"]
    assert "entry_index is 0" in ledger["entry_index_rule"]
    assert "predecessor.entry_index + 1" in ledger["entry_index_rule"]
    assert "domains.ledger_genesis_value" in ledger["genesis_rule"]
    assert {
        "runtime-attestation-attempt",
        "runtime-attestation-terminal",
        "slot-segment-closed",
        "seal-consumption",
        "seal-terminal",
    } <= set(ledger["entry_kind_enum"])
    assert "changed slot 0..27" in ledger["bootstrap_and_successor_order"]
    assert "slot-28 closure goes only to horizon" in ledger[
        "bootstrap_and_successor_order"
    ]
    attempt = schemas["probe_attempt_marker"]
    assert "first attempt in a slot is integer 0" in attempt["attempt_ordinal"]
    assert "prior attempt_ordinal + 1" in attempt["attempt_ordinal"]


def test_schedule_anchor_cadence_grace_and_absolute_horizon_are_exact() -> None:
    schedule = _plan()["schedule"]
    anchor = _utc(schedule["anchor_at"])
    cadence = timedelta(seconds=schedule["cadence_seconds"])
    grace = timedelta(seconds=schedule["grace_seconds"])
    assert anchor == datetime(2026, 8, 17, tzinfo=timezone.utc)
    assert cadence == timedelta(days=1)
    assert grace == timedelta(hours=6)
    assert schedule["first_slot_index"] == 0
    assert schedule["last_slot_index"] == 28
    assert schedule["slot_count"] == 29

    slots = [anchor + index * cadence for index in range(schedule["slot_count"])]
    assert slots[0] == _utc(schedule["first_slot_at"])
    assert slots[-1] == _utc(schedule["final_selection_slot_at"])
    assert slots[-1] + grace == _utc(schedule["absolute_horizon_expires_at"])
    assert schedule["window_lower_inclusive"] is True
    assert schedule["window_upper_exclusive"] is True
    assert schedule["horizon_resets_on_segment_change"] is False
    assert schedule["scheduled_upper_bound_not_actual_time"] is True

    def allowed(instant: datetime, slot: datetime) -> bool:
        return slot <= instant < slot + grace

    for vector in schedule["boundary_vectors"]:
        assert allowed(_utc(vector["instant"]), slots[vector["slot_index"]]) is (
            vector["launch_allowed"]
        )


def test_missed_slots_are_fail_closed_and_retries_cannot_become_cadence() -> None:
    schedule = _plan()["schedule"]
    assert schedule["no_sleep_or_poll_across_slots"] is True
    assert schedule["one_invocation_may_span_multiple_slots"] is False
    retry_rule = schedule["retry_without_valid_receipt"]
    assert "before key creation" in retry_rule
    assert "before any source open" in retry_rule
    assert "no recapture after either source opens" in retry_rule
    assert schedule["orchestration_retry_creates_new_slot"] is False
    assert schedule["first_validator_valid_receipt_per_slot_immutable"] is True
    assert schedule["early_invocation_action"] == "reject before source access"
    assert schedule["missed_slot_action"].startswith(
        "terminal schedule-integrity-failure"
    )
    for forbidden in (
        "backfill_allowed",
        "catch_up_allowed",
        "slot_replacement_allowed",
        "slot_skipping_allowed",
    ):
        assert schedule[forbidden] is False

    ledger = _plan()["accrual_and_stopping"]["ledger"]
    assert ledger["append_only"] is True
    assert ledger["first_validator_valid_receipt_per_slot_wins"] is True
    assert ledger["valid_receipt_replacement_allowed"] is False
    assert ledger["out_of_order_slot_allowed"] is False
    assert ledger["gap_allowed"] is False
    assert ledger[
        "attempt_and_failure_markers_are_ledger_entries_but_not_slot_resolutions"
    ] is True
    assert "before any source open" in ledger[
        "failed_execution_may_retry_inside_same_grace"
    ]
    assert ledger["post_source_open_retry_may_recapture"] is False

    failure = _plan()["operational_receipt_schemas"]["probe_failure_marker"]
    cases = [
        ("pre-key-pre-source", 0, False, True),
        ("key-created-pre-source", 0, True, False),
        ("first-source-opened", 1, True, False),
        ("snapshots-latched", 2, True, False),
        ("keyed-count-computed", 2, True, False),
    ]
    for phase, source_opens, key_created, expected in cases:
        retry = phase == "pre-key-pre-source" and source_opens == 0 and not key_created
        assert retry is expected
        assert phase in failure["phase_at_failure_enum"]


def test_runtime_change_closes_whole_segment_without_moving_calendar() -> None:
    segments = _plan()["runtime_segments"]
    assert set(segments["complete_unaliased_service_tuple"]) == {
        "service_identity_sha256",
        "boot_identity_sha256",
        "boot_started_at",
        "serving_build.commit",
        "serving_build.tree",
        "every serving_build.implementation key with sha256 and bytes",
        "sanitized_configuration.schema",
        "sanitized_configuration.encoding",
        "sanitized_configuration.sha256",
        "sanitized_configuration.bytes",
        "effective_legacy_repeat_controls.LM_RECALL_REPEAT_GATING",
        "effective_legacy_repeat_controls.LM_RECALL_REPEAT_DROP_TRAILING_STUBS",
    }
    assert segments["tuple_order_cannot_hide_change"] is True
    assert segments["change_closes_entire_segment"] is True
    assert set(segments["change_members"]) == {
        "source set",
        "alias binding",
        "service identity",
        "boot identity or boot instant",
        "serving commit or tree",
        "any implementation hash or byte size",
        "sanitized configuration identity",
        "either effective legacy repeat control",
    }
    successor = segments["successor_attestation"]
    assert successor["append_only"] is True
    assert successor["predecessor_closure_sha256_required"] is True
    assert successor["lower_bound_equals_boundary_at"] is True
    assert successor["strictly_later_lower_bound_required"] is True
    assert successor["stable_pre_post_tuple_required"] is True
    assert successor["must_complete_before_eligible_slot"] is True
    assert successor["overwrites_release_watermark"] is False
    reset = segments["on_successor"]
    assert reset["counts_reset"] is True
    assert reset["readiness_latch_reset_if_seal_authority_unconsumed"] is True
    assert reset["snapshots_carried"] is False
    assert reset["schedule_reset"] is False
    assert reset["horizon_reset"] is False
    assert segments["cross_segment_pooling_allowed"] is False
    ceremony = segments["successor_boundary_ceremony"]
    assert ceremony["pre_source_order"] == ["local", "alt"]
    assert ceremony["post_source_order"] == ["local", "alt"]
    assert ceremony["post_observation_not_before_boundary_plus_seconds"] == 0.25
    assert ceremony["maximum_envelope_seconds"] == 300
    assert ceremony["lower_bound_exclusive_at"] == "exactly boundary_at"
    assert "exact maximum" in ceremony["boundary_at"]
    assert "boundary_launcher_at" in ceremony["boundary_at"]
    assert ceremony["attempts_per_closure"] == 1
    assert ceremony["start_deadline_after_closure_seconds"] == 30
    assert segments["inter_slot_observation_or_closure_allowed"] is False
    assert segments["closure_requires_proven_tuple_or_binding_mismatch"] is True
    assert "never a closure" in segments["observation_unavailable_action"]
    assert segments["segment_closure_mismatch_kind_enum"] == [
        "active-services-state-change",
        "source-binding-change",
    ]


def test_sources_are_complete_distinct_read_only_and_slot_bound() -> None:
    sources = _plan()["sources"]
    assert sources["source_aliases_exactly"] == ["local", "alt"]
    assert set(sources["aliases"]) == {"local", "alt"}
    assert sources["alias_authorities_exactly"] == {
        "local": "local",
        "alt": "ssh-alias:alt",
    }
    for source in sources["aliases"].values():
        assert source["alias_id"].endswith("-v3-ro")
        assert source["locator"] == "external-untracked-operator-mapping"
        assert source["source_access"] == "read-only"
        assert source["snapshot_open"] == "mode=ro&immutable=1&cache=private"
        assert source["query_only"] is True
        assert source["temp_store"] == "MEMORY"
    assert sources["alias_swap_action"] == "fatal-integrity-error"
    assert sources["distinct_authorities_and_database_instances_required"] is True
    assert sources["equal_snapshot_sha256"] == "fatal-integrity-error"
    assert sources["missing_or_extra_source"] == "fatal-integrity-error"
    assert sources["same_snapshot_set_required_for_probe_and_packet"] is True
    assert sources["public_alias_to_watermark_array_index_mapping"] is False
    assert "before any source-row access" in sources["private_bijection_precondition"]
    derivation = sources["database_instance_identity_derivation"]
    assert derivation["domain_separator_utf8"].endswith(
        "/source-database-identity/v1"
    )
    assert derivation["private_input_exactly"] == [
        "filesystem_uuid",
        "statx_inode_uint64",
        "statx_birthtime_ns_int64",
    ]
    assert derivation["alias_or_service_identity_in_preimage"] is False
    assert derivation["fallback_allowed"] is False
    assert derivation["raw_inputs_persisted"] is False
    assert derivation["pre_post_equal_within_segment"] is True
    assert derivation["values_distinct_across_aliases"] is True
    assert "before any source-row or snapshot access" in derivation[
        "segment_binding_capture"
    ]
    assert "after both immutable snapshot captures" in derivation[
        "probe_binding_capture"
    ]
    binding = sources["alias_service_database_binding_derivation"]
    assert "authenticated_authority_identity" in binding["private_input_exactly"]
    core = sources["source_binding_core"]
    assert core["fields_exactly"] == [
        "alias_ids_by_alias",
        "database_instance_identity_sha256_by_alias",
        "alias_service_database_binding_sha256_by_alias",
        "active_services_state_sha256",
    ]
    assert core["map_keys_exactly"] == ["local", "alt"]
    assert core["database_digests_distinct"] is True


def test_selection_is_strict_complete_slot_bounded_and_outcome_blind() -> None:
    selection = _plan()["selection"]
    created_at = selection["predicate"]["created_at"]
    assert created_at == {
        "gt_release_effective_at": "2026-08-14T15:03:46.793603Z",
        "gt_active_segment_lower_bound_exclusive_at": True,
        "lte_scheduled_slot_at": True,
    }
    assert selection["predicate"]["scope_filter"] is None
    assert selection["predicate"]["replayability_filter"] is None
    assert selection["predicate"]["outcome_filtering"] is False
    assert selection["all_eligible_events_included"] is True
    assert selection["all_selected_events_accounted_in_packet"] is True
    assert selection["floors_are_not_inclusion_filters"] is True
    assert selection["replayability_is_not_inclusion_filter"] is True
    assert selection["no_max_n"] is True
    assert selection["no_stop_at_floor"] is True
    assert selection["duplicate_source_qualified_event_key"] == "fatal-integrity-error"
    assert "present in either exact immutable alias snapshot" in selection[
        "predicate"
    ]["population"]
    assert selection["ordering"][0] == "parsed UTC microseconds since Unix epoch ASC"
    utc_contract = selection["utc_timestamp_contract"]
    assert "1..6 fractional digits" in utc_contract["accepted_grammar"]
    assert "integer microseconds" in utc_contract["comparison"]
    assert utc_contract["canonical_receipt_encoding"].endswith(".ffffffZ")
    assert selection["missing_family_identity_action"].startswith(
        "no automatic identity edge"
    )
    assert selection["missing_transport_session_action"].startswith(
        "no session or workflow edge"
    )

    lower = _utc(created_at["gt_release_effective_at"])
    for vector in selection["initial_boundary_vectors"]:
        assert (_utc(vector["created_at"]) > lower) is vector["included"]
    for vector in selection["upper_boundary_vectors"]:
        assert (_utc(vector["created_at"]) <= _utc(vector["scheduled_at"])) is (
            vector["included"]
        )

    expected_forbidden = {
        "feedback_applied",
        "success",
        "error",
        "payload",
        "latency",
        "access",
        "results",
        "content",
        "case_outcome",
        "candidate_metric",
        "baseline_metric",
        "family_size",
        "session_presence",
        "workflow_presence",
        "replayability",
    }
    assert expected_forbidden <= set(selection["forbidden_filter_fields"])
    disjoint = selection["consumed_packet_disjointness"]
    assert _utc(disjoint["replacement_upper_bound_exclusive"]) < lower
    assert disjoint["open_consumed_corpora"] is False
    assert disjoint["invoke_consumed_verifiers"] is False


def test_replayability_is_input_only_and_required_for_every_floor_witness() -> None:
    plan = _plan()
    replay = plan["replayability"]
    assert replay["role"] == "input-only floor and measurement eligibility; never selection"
    assert set(replay["replay_input_schema"]) == {
        "event_id",
        "created_at",
        "query",
        "scope_argument",
        "resolved_requested_scope",
        "max_results_argument",
        "depth_argument",
        "ambient_context",
        "agent",
        "task",
        "session_id",
        "transport_session_id",
        "initial_state",
    }
    forbidden = " ".join(replay["forbidden_inputs"])
    for term in ("feedback", "success", "result", "latency", "candidate", "baseline"):
        assert term in forbidden
    assert replay["selected_nonreplayable_remains_selected"] is True
    assert replay["selected_nonreplayable_floor_contribution"] == 0
    assert replay["selected_nonreplayable_measurement_contribution"] == 0
    assert replay["missing_measurement_for_replayable_cohort_case"].startswith("fail;")
    assert replay["probe_builds_candidate_plan"] is False
    assert replay["probe_builds_corpus"] is False
    context = replay["replay_input_schema"]["ambient_context"]
    assert context["behaviorally_relevant_keys_exactly"] == [
        "scope",
        "session_id",
        "session",
        "project",
        "project_name",
        "workspace",
        "workspace_path",
        "cwd",
        "project_scope",
        "agent",
        "task",
        "transport_session_id",
    ]
    assert context["unknown_key_action"].startswith("unconditionally nonreplayable")
    assert "persisted requested_scope" in replay["replay_input_schema"][
        "scope_argument"
    ]
    assert "historical caller omitted" in replay["replay_input_schema"][
        "max_results_argument"
    ]
    assert "historical surface spelling" in replay["replay_input_schema"][
        "depth_argument"
    ]
    assert set(replay["requested_scope_grammar"]) == {
        "global",
        "project",
        "session",
        "anything_else",
    }
    seed_validation = replay["shared_seed_state_validation"]
    assert seed_validation["used_byte_identically_by"] == [
        "aggregate-probe",
        "packet-sealer",
        "semantic-evaluator-seed-builder",
    ]
    assert seed_validation["retrieval_weights_table_must_be_nonempty"] is True
    assert seed_validation["production_policy_keys_required"] == [
        "default",
        "project",
        "global",
        "session",
    ]
    assert seed_validation[
        "node_scope_grammar_applied_to_retrieval_weight_policy_key"
    ] is False
    consistency = replay["persisted_to_caller_consistency"]
    assert "ambient_context.agent" in consistency["agent"]
    for falsy in ("empty string", "zero", "false"):
        assert falsy in consistency["agent"]
    assert consistency["mismatch_action"].endswith(
        "fatal preseal mismatch if it was floor-counted"
    )

    readiness = plan["readiness"]
    assert readiness["all_floor_counted_events_and_calls_replayable"] is True
    assert readiness["family_qualification_uses_replayable_events_only"] is True
    assert readiness["organic_floor_uses_replayable_events_only"] is True
    assert readiness["session_component_scope_witnesses_replayable"] is True
    family = plan["identity"]["repeated_automatic_family"]
    assert family["minimum_events"] == 3
    assert family["minimum_distinct_nonempty_source_qualified_transport_sessions"] == 2
    assert family["nonreplayable_contribution"] == 0


def test_partition_rule_is_fresh_complete_and_component_closed() -> None:
    partition = _plan()["partition"]
    assert partition["domain_separator_utf8"] == "confirmatory-holdout-v3/partition/v1"
    assert partition["algorithm"] == "SHA-256"
    assert partition["modulus"] == 100
    assert partition["holdout_buckets"] == {"gte": 0, "lt": 50}
    assert partition["shadow_buckets"] == {"gte": 50, "lt": 100}

    domain = partition["domain_separator_utf8"].encode() + b"\0"
    observed_partitions: set[str] = set()
    for vector in partition["golden_vectors"]:
        representative = vector["representative_display"].replace("\\0", "\0").encode()
        digest = hashlib.sha256(domain + representative).digest()
        assert digest.hex() == vector["digest_sha256"]
        bucket = int.from_bytes(digest[:4], "big") % partition["modulus"]
        assert bucket == vector["bucket"]
        expected = "holdout" if bucket < 50 else "shadow"
        assert vector["partition"] == expected
        observed_partitions.add(expected)
    assert observed_partitions == {"holdout", "shadow"}
    assert partition["event_intersection_required"] == 0
    assert partition["event_union_equals_selected_population"] is True
    assert partition["automatic_family_cross_partition_count_required"] == 0
    assert partition["organic_session_cross_partition_count_required"] == 0
    assert partition["workflow_cross_partition_count_required"] == 0
    assert partition["outcome_field_invariant"] is True
    for adaptation in (
        "key_search_allowed",
        "rebalancing_allowed",
        "truncation_allowed",
        "stratification_allowed",
        "reassignment_allowed",
    ):
        assert partition[adaptation] is False


def test_every_numeric_power_floor_is_deep_equal_to_v2() -> None:
    v3 = _plan()
    v2 = _load_json(V2_PLAN_PATH)
    assert v3["readiness"]["holdout_floor"] == v2["readiness"]["holdout_floor"]
    assert v3["readiness"]["shadow_floor"] == v2["readiness"]["shadow_floor"]
    assert v3["readiness"]["post_run_evidence_floor"] == (
        v2["readiness"]["post_run_evidence_floor"]
    )
    assert v3["a_priori_power_floor"] == v2["a_priori_power_floor"]
    assert v3["readiness"]["decision_operator"] == "all"
    assert v3["readiness"]["scheduled_ready_floor_groups_exactly"] == [
        "holdout_floor",
        "shadow_floor",
    ]
    assert v3["readiness"]["post_run_evidence_floor_is_not_readiness_input"] is True
    assert v3["readiness"]["partition_pooling_allowed"] is False
    assert v3["readiness"]["segment_pooling_allowed"] is False
    assert v3["readiness"]["use_complete_population_not_minimum_prefix"] is True


def test_intermediate_probes_are_nonterminal_first_pass_seals_and_horizon_escalates() -> None:
    plan = _plan()
    readiness = plan["readiness"]
    assert readiness["probe_receipt_status_allowed"] == ["below-floor", "ready"]
    assert readiness["slot_resolution_status_allowed"] == [
        "below-floor",
        "ready",
        "segment-closed",
    ]
    assert readiness["intermediate_insufficient_status_allowed"] is False
    assert readiness["probe_launches_consume_authority"] is False
    assert readiness["intermediate_below_floor_action"].endswith(
        "continue to the next fixed slot"
    )
    assert "before publishing ready" in readiness["ready_action"]
    assert "no externally visible discretionary latch" in readiness["ready_action"]
    assert "publish no counts" in readiness["runtime_change_at_provisional_ready_action"]
    assert readiness["final_below_floor_action"].startswith("perform no later source read")

    stopping = plan["accrual_and_stopping"]
    assert stopping["first_scheduled_passing_slot_mandatory"] is True
    assert stopping["pass_may_be_ignored_or_deferred_for_outcome_review"] is False
    assert stopping["later_passing_slot_substitution_allowed"] is False
    assert stopping["horizon_terminal_status"] == "insufficient-evidence"
    transitions = stopping["state_machine"]

    def edges(source: str, target: str) -> list[dict[str, Any]]:
        return [
            entry
            for entry in transitions
            if entry["from"] == source and entry["to"] == target
        ]

    assert edges("awaiting-slot", "awaiting-next-fixed-slot")[0]["terminal"] is False
    assert edges("awaiting-slot", "awaiting-slot")
    assert "pre-source-retryable" in edges("awaiting-slot", "awaiting-slot")[0][
        "event"
    ]
    assert edges("awaiting-slot", "seal-marker-creation-required")
    assert edges(
        "seal-marker-creation-required",
        "seal-authority-consumed-awaiting-blocked-spawn",
    )
    assert edges(
        "seal-authority-consumed-awaiting-blocked-spawn",
        "seal-authority-consumed-awaiting-ready-ledger",
    )
    assert edges(
        "seal-authority-consumed-awaiting-ready-ledger",
        "sealer-running-ready-visible",
    )
    assert edges("sealer-running-ready-visible", "sealed")
    target_sets = {
        state: {
            entry["to"] for entry in transitions if entry["from"] == state
        }
        for state in (
            "seal-marker-creation-required",
            "seal-authority-consumed-awaiting-blocked-spawn",
            "seal-authority-consumed-awaiting-ready-ledger",
            "sealer-running-ready-visible",
        )
    }
    assert target_sets == {
        "seal-marker-creation-required": {
            "seal-authority-consumed-awaiting-blocked-spawn",
            "terminal-seal-failure-escalate",
        },
        "seal-authority-consumed-awaiting-blocked-spawn": {
            "seal-authority-consumed-awaiting-ready-ledger",
            "terminal-seal-failure-escalate",
        },
        "seal-authority-consumed-awaiting-ready-ledger": {
            "sealer-running-ready-visible",
            "terminal-seal-failure-escalate",
        },
        "sealer-running-ready-visible": {
            "sealed",
            "terminal-seal-failure-escalate",
        },
    }
    assert all(entry["to"] != "seal-required" for entry in transitions)
    assert edges(
        "awaiting-slot", "segment-closed-awaiting-immediate-re-attestation"
    )
    assert edges(
        "initial-segment-awaiting-source-binding",
        "segment-closed-awaiting-immediate-re-attestation",
    )
    assert edges(
        "segment-closed-awaiting-immediate-re-attestation",
        "awaiting-next-fixed-slot",
    )
    assert edges("awaiting-next-fixed-slot", "awaiting-slot")
    assert edges(
        "awaiting-slot", "awaiting-absolute-horizon-with-no-more-source-reads"
    )
    assert edges(
        "awaiting-absolute-horizon-with-no-more-source-reads",
        "terminal-insufficient-evidence-escalate",
    )
    assert edges("awaiting-slot", "terminal-schedule-integrity-failure-escalate")
    assert edges("awaiting-slot", "terminal-probe-integrity-failure-escalate")
    seal_terminal_edges = [
        entry
        for entry in transitions
        if entry["to"] in {"sealed", "terminal-seal-failure-escalate"}
    ]
    assert seal_terminal_edges
    assert all("seal-terminal" in entry["event"] for entry in seal_terminal_edges)
    nonterminal_targets = {
        entry["to"] for entry in transitions if not entry["terminal"]
    }
    states_with_outgoing = {entry["from"] for entry in transitions}
    assert nonterminal_targets <= states_with_outgoing


def test_packet_seal_is_one_shot_manifest_last_and_precedes_implementation() -> None:
    plan = _plan()
    seal = plan["authority"]["seal"]
    assert seal["id"] == "confirmatory-holdout-v3-seal"
    assert seal["max_process_launches"] == 1
    assert seal["max_publication_attempts"] == 1
    assert seal["consumed_at"] == (
        "durable no-overwrite consumption marker created before process "
        "spawn and before content creation"
    )
    assert seal["watchdog_seconds"] == 3600
    assert seal["forced_termination_grace_seconds"] == 30
    assert seal["timeout_action"].startswith("terminal-seal-failure")
    publication = plan["publication_and_ordering"]
    assert publication["packet_publication_attempts"] == 1
    assert publication["no_overwrite"] is True
    assert publication["content_before_manifest"] is True
    assert publication["manifest_last"] is True
    assert publication["partial_publication_action"] == "terminal-seal-failure"
    assert publication["post_build_pre_manifest_runtime_change_action"].startswith(
        "terminal-seal-failure"
    )
    assert publication["reseal_allowed"] is False
    assert publication["repair_before_both_partitions_sealed"] is False
    assert publication["semantic_evaluator_before_both_partitions_sealed"] is False
    assert publication[
        "packet_seal_commit_must_be_strict_ancestor_of_repair_and_evaluator_commits"
    ] is True
    assert publication["timestamps_alone_sufficient"] is False
    assert publication["sealed_state"]["semantic_reads"] == {
        "confirmatory-shadow-v3-eval": 0,
        "confirmatory-holdout-v3-eval": 0,
    }
    assert publication["sealed_state"][
        "active_segment_unchanged_at_post_build_pre_manifest_observation"
    ] is True
    consumption = plan["operational_receipt_schemas"]["seal_consumption_marker"]
    assert consumption["creation_precedes_process_spawn"] is True
    assert "provisional_ready_core_sha256_and_bytes" in consumption[
        "fields_exactly"
    ]
    assert "ready_resolution_id" not in consumption["fields_exactly"]
    terminal = plan["operational_receipt_schemas"]["seal_terminal_receipt"]
    assert "handoff-timeout" in terminal["failure_reason_enum"]
    for stage in (
        "spawn",
        "ready-staging",
        "ready-ipc",
        "ready-visibility",
        "start-gate",
        "blocked-child",
    ):
        assert "handoff-timeout" in terminal["stage_reason_matrix"][stage]
    assert "marker-validation-failure" in terminal["marker_identity_rule"]
    assert "ready resolution validated_at" in terminal["time_rule"]
    assert "head H" in terminal["ledger_predecessor_rule"]
    assert "continues from M" in terminal["ledger_predecessor_rule"]
    assert "ready head R" in terminal["ledger_predecessor_rule"]


def test_holdout_launch_is_unconditional_and_receipts_release_jointly() -> None:
    authority = _plan()["authority"]
    assert authority["execution_order"] == [
        "confirmatory-shadow-v3-eval",
        "confirmatory-holdout-v3-eval",
    ]
    assert authority["holdout_launch_unconditional_on_shadow_or_public_outcome"] is True
    assert authority["joint_receipt_release"] is True
    assert authority["identical_frozen_stack_for_both_readers"] is True
    assert authority["shadow_result_embargo"].startswith(
        "only the isolated shadow evaluator and controller"
    )
    assert authority["reader_watchdog_seconds"] == 3600
    assert authority["forced_termination_grace_seconds"] == 30
    assert authority["controller_handoff_deadline_seconds"] == 30
    assert authority["reader_launch_consumes_before_source_open"] is True
    assert authority["public_terminal_attempt_then_shadow_unconditional"] is True
    assert authority[
        "launcher_synthesizes_aggregate_terminal_receipt_for_missing_invalid_crashed_or_timed_out_reader"
    ] is True
    assert authority["reader_aliases_allowed"] is False
    assert authority["reader_delegation_allowed"] is False
    assert authority["reader_wildcards_allowed"] is False
    assert authority["fallback_readers_allowed"] is False


def test_reader_state_machine_watchdogs_consumption_and_joint_release_are_total() -> None:
    plan = _plan()
    authority = plan["authority"]
    transitions = authority["evaluation_state_machine"]

    def targets(source: str) -> set[str]:
        return {entry["to"] for entry in transitions if entry["from"] == source}

    assert targets("public-evaluation-terminal-attempt-required") == {
        "shadow-authority-consumed-awaiting-marker-and-spawn",
        "holdout-authority-consumed-awaiting-marker-and-spawn",
        "joint-release-required",
        "terminal-evaluation-integrity-failure-default-off",
    }
    assert targets("implementation-stack-frozen") == {
        "public-evaluation-terminal-attempt-required",
        "terminal-default-off-before-readers",
    }
    assert targets("shadow-authority-consumed-awaiting-marker-and-spawn") == {
        "shadow-terminal-required",
        "holdout-authority-consumed-awaiting-marker-and-spawn",
        "joint-release-required",
        "terminal-evaluation-integrity-failure-default-off",
    }
    assert targets("shadow-terminal-required") == {
        "holdout-authority-consumed-awaiting-marker-and-spawn",
        "joint-release-required",
        "terminal-evaluation-integrity-failure-default-off",
    }
    assert targets("holdout-terminal-required") == {
        "joint-release-required",
        "terminal-evaluation-integrity-failure-default-off",
    }
    assert targets("holdout-authority-consumed-awaiting-marker-and-spawn") == {
        "holdout-terminal-required",
        "joint-release-required",
        "terminal-evaluation-integrity-failure-default-off",
    }
    assert targets("joint-release-required") == {
        "confirmation-terminal",
        "terminal-evaluation-integrity-failure-default-off",
    }
    assert authority["reader_authority_consumption_trigger"].startswith(
        "durable O_EXCL reader-launch-attempt marker creation"
    )
    assert "without a second process launch" in authority[
        "existing_valid_attempt_marker_action"
    ]
    assert "never overwrite or relaunch" in authority[
        "invalid_or_colliding_attempt_marker_action"
    ]

    schemas = plan["operational_receipt_schemas"]
    launch = schemas["reader_launch_consumption_marker"]
    assert "before evaluator process spawn or source open" in launch["creation"]
    assert set(launch["reader_id_partition_pairs"].values()) == {
        "shadow",
        "holdout",
    }
    assert "controller_handoff_deadline_at" in launch["fields_exactly"]
    attempt = schemas["reader_launch_attempt_marker"]
    assert "controller_handoff_deadline_at" in attempt["fields_exactly"]
    assert "nonresetting deadline" in attempt["handoff_deadline_rule"]
    assert schemas["development_terminal_receipt"]["receipt_kind"] == (
        "development-calibration-terminal"
    )
    pre_reader = schemas["pre_reader_launch_attempt_marker"]
    assert pre_reader["pre_reader_phase_enum"] == [
        "development-calibration",
        "public-evaluation",
    ]
    assert "O_CREAT|O_EXCL" in pre_reader["creation"]
    assert "second process" in pre_reader["creation"]
    assert "same predecessor instant+3600 seconds" in pre_reader["time_rule"]
    assert "pre_reader_launch_attempt_marker_sha256_and_bytes_or_null" in schemas[
        "development_terminal_receipt"
    ]["fields_exactly"]
    assert "keeps the marker nonnull" in schemas["development_terminal_receipt"][
        "time_rule"
    ]
    public = schemas["public_terminal_receipt"]
    assert public["receipt_kind"] == "public-evaluation-terminal"
    assert "state_entry_at" in public["fields_exactly"]
    assert "controller_state_predecessor_sha256_and_bytes" in public[
        "fields_exactly"
    ]
    assert "pre_reader_launch_attempt_marker_sha256_and_bytes_or_null" in public[
        "fields_exactly"
    ]
    assert "state_entry_at+30 seconds" in public["time_rule"]
    assert "keeps the marker nonnull" in public["time_rule"]
    terminal = schemas["reader_terminal_receipt"]
    assert set(terminal["terminal_status_enum"]) == {
        "pass",
        "fail",
        "crash",
        "invalid-output",
        "inconclusive",
        "timeout",
        "launch-failure",
        "launch-integrity-failure",
    }
    assert "controller_handoff_deadline_at" in terminal["fields_exactly"]
    assert "never receives a fresh 30-second window" in terminal[
        "handoff_deadline_rule"
    ]
    aggregate = terminal["aggregate_measurements_schema"]
    assert set(aggregate["uncertainty_object_fields_exactly"]) == set(
        plan["uncertainty"]["required_report_fields"]
    )
    assert aggregate["resamples"] == 10000
    assert set(aggregate["metric_failure_reason_value_enum"]) == {
        None,
        "missing-metric",
        "zero-denominator",
        "incomplete-pair",
        "instrumentation-gap",
    }
    joint = schemas["joint_reader_release_manifest"]
    assert "one private same-filesystem temp envelope" in joint["publication"]
    assert "one atomic no-replace visibility operation" in joint["publication"]
    assert "never directly visible" in joint["publication"]
    assert "controller_handoff_deadline_at" in joint["time_rule"]


def test_point_gate_numbers_and_uncertainty_parameters_remain_unchanged() -> None:
    v3 = _plan()
    v2 = _load_json(V2_PLAN_PATH)
    assert _numeric_leaves(v3["measurement"]["parent_point_gates"]) == (
        _numeric_leaves(v2["measurement"]["parent_point_gates"])
    )
    normalized_v3_gates = json.loads(
        json.dumps(v3["measurement"]["parent_point_gates"])
    )
    for gate in (
        "organic_payload_delta",
        "organic_access_delta",
        "agent_triggered_delivery",
    ):
        normalized_v3_gates[gate]["cluster_unit"] = normalized_v3_gates[gate][
            "cluster_unit"
        ].replace("selected replayable organic event", "selected organic event")
    assert normalized_v3_gates == v2["measurement"]["parent_point_gates"]
    assert set(v3["measurement"]["metric_definitions"]) == set(
        v3["measurement"]["parent_point_gates"]
    )
    assert v3["measurement"]["all_parent_point_gates_conjunctive"] is True
    assert v3["measurement"]["missing_metric"] == "fail"
    assert v3["measurement"]["zero_denominator"] == "fail"
    assert v3["measurement"]["incomplete_pair"] == "fail"
    assert v3["measurement"]["alternate_metric_or_cohort_allowed"] is False
    assert v3["measurement"]["trimming_or_metric_specific_case_exclusion_allowed"] is False

    uncertainty = v3["uncertainty"]
    assert uncertainty["resamples"] == v2["uncertainty"]["resamples"] == 10000
    assert uncertainty["confidence_level"] == v2["uncertainty"]["confidence_level"] == 0.95
    assert uncertainty["interval"] == v2["uncertainty"]["interval"]
    assert uncertainty["supplementary_only"] is True
    assert uncertainty["interval_can_rescue_failed_point_gate"] is False


def test_v3_bootstrap_seeds_are_fresh_namespace_derivations() -> None:
    uncertainty = _plan()["uncertainty"]
    expected = {
        "automatic_component": ("automatic-component", 588763221),
        "organic_component": ("organic-component", 1632232291),
        "shadow_component": ("shadow-component", 2220152078),
    }
    for stream, (label, value) in expected.items():
        declared = uncertainty["seeds"][stream]
        digest = hashlib.sha256(b"confirmatory-holdout-v3\0" + label.encode()).digest()
        assert declared == {"label": label, "value": value}
        assert int.from_bytes(digest[:4], "big") == value
    v2_values = {
        entry["value"] for entry in _load_json(V2_PLAN_PATH)["uncertainty"]["seeds"].values()
    }
    assert v2_values.isdisjoint(value for _, value in expected.values())
    assert uncertainty["retired_v2_seed_values_reusable"] is False


def test_state_seed_allowlist_is_outcome_free_and_production_shaped() -> None:
    construction = _plan()["measurement"]["state_isolation"]["construction"]
    assert construction["copied_table_column_allowlist"] == {
        "nodes": [
            "packet_node_label",
            "level",
            "deidentified_content",
            "packet_scope_label",
            "created_at",
        ],
        "connections": [
            "source_packet_node_label",
            "target_packet_node_label",
            "relation_type",
            "weight",
        ],
        "retrieval_weights": [
            "packet_scope_label",
            "bm25_weight",
            "vector_weight",
            "graph_weight",
        ],
    }
    assert construction["source_retrieval_policy_keys_required"] == [
        "default",
        "project",
        "global",
        "session",
    ]
    assert "never a node memory-scope predicate" in construction[
        "source_retrieval_policy_key_validation"
    ]
    assert construction["empty_source_retrieval_weights_table_validates_production_shape"] is False
    assert construction["recall_events"].startswith("empty at seed;")
    assert construction["shared_validator_reference"].startswith(
        "replayability.shared_seed_state_validation"
    )
    forbidden_fragments = {"feedback", "result", "payload", "latency", "access", "history", "signal"}
    copied_fields = {
        field
        for fields in construction["copied_table_column_allowlist"].values()
        for field in fields
    }
    assert not any(fragment in field for fragment in forbidden_fragments for field in copied_fields)


def test_receipts_are_allowlisted_aggregate_only_and_hash_bound() -> None:
    plan = _plan()
    allowlist = set(plan["readiness"]["probe_receipt_allowlist"])
    required = {
        "status",
        "slot_index",
        "scheduled_at",
        "grace_deadline_at",
        "release_effective_at",
        "active_segment_lower_bound_exclusive_at",
        "segment_id",
        "previous_ledger_entry_sha256",
        "aliased_source_snapshot_sha256_and_bytes",
        "selected_event_count",
        "selected_replayable_event_count",
        "selected_nonreplayable_event_count",
        "holdout_unseen_automatic_family_count",
        "holdout_unseen_automatic_event_count",
        "holdout_unseen_automatic_component_count",
        "holdout_organic_event_count",
        "holdout_organic_session_count",
        "holdout_organic_component_count",
        "holdout_project_scope_count",
        "shadow_real_workflow_count",
        "shadow_replayable_logical_call_count",
        "shadow_real_workflow_component_count",
        "shadow_project_scope_count",
        "analysis_plan_sha256_and_bytes",
    }
    assert required <= allowlist
    forbidden_receipt_names = {
        "query",
        "content",
        "event_id",
        "family_membership",
        "per_source_count",
        "case_outcome",
        "candidate_plan",
    }
    assert forbidden_receipt_names.isdisjoint(allowlist)
    assert plan["readiness"][
        "individual_family_workflow_scope_spelling_or_per_source_counts_allowed"
    ] is False

    schemas = plan["operational_receipt_schemas"]
    assert {
        "common_validation",
        "source_binding_attestation",
        "runtime_segment_attestation",
        "successor_attestation_attempt_marker",
        "runtime_attestation_terminal_marker",
        "slot_segment_closure_resolution",
        "slot_probe_resolution",
        "post_build_runtime_observation",
        "probe_attempt_marker",
        "probe_failure_marker",
        "missed_slot_marker",
        "horizon_terminal_marker",
        "ledger_entry",
        "seal_consumption_marker",
        "probe_terminal_marker",
        "seal_terminal_receipt",
        "implementation_stack_freeze_receipt",
        "pre_reader_launch_attempt_marker",
        "development_terminal_receipt",
        "public_terminal_receipt",
        "reader_terminal_receipt",
        "reader_launch_attempt_marker",
        "reader_launch_consumption_marker",
        "joint_reader_release_manifest",
    } <= set(schemas)
    for name, schema in schemas.items():
        if name == "common_validation":
            continue
        fields = schema["fields_exactly"]
        assert fields
        assert len(fields) == len(set(fields))
        assert forbidden_receipt_names.isdisjoint(fields)
    common = schemas["common_validation"]
    categories = {
        name: set(fields)
        for name, fields in common["closed_field_type_rules"].items()
        if isinstance(fields, list)
    }
    all_schema_fields = {
        field
        for name, schema in schemas.items()
        if name != "common_validation"
        for field in schema["fields_exactly"]
    }
    constants = set(common["constants"])
    for name, schema in schemas.items():
        if name == "common_validation":
            continue
        for field in schema["fields_exactly"]:
            if field in constants:
                continue
            assert sum(field in members for members in categories.values()) == 1, (
                name,
                field,
            )
    assert set().union(*categories.values()) <= all_schema_fields
    assert plan["readiness"]["probe_receipt_allowlist"] == schemas[
        "slot_probe_resolution"
    ]["fields_exactly"]
    assert plan["sources"]["source_binding_attestation_allowlist"] == schemas[
        "source_binding_attestation"
    ]["fields_exactly"]
    assert plan["runtime_segments"]["segment_attestation_allowlist"] == schemas[
        "runtime_segment_attestation"
    ]["fields_exactly"]
    assert plan["runtime_segments"][
        "slot_segment_closure_resolution_allowlist"
    ] == schemas["slot_segment_closure_resolution"]["fields_exactly"]
    assert schemas["missed_slot_marker"]["source_open_count"] == 0
    assert schemas["horizon_terminal_marker"]["reason_enum"] == [
        "final-slot-below-floor",
        "final-slot-segment-closed",
    ]
    assert "probe-terminal" in schemas["ledger_entry"]["entry_kind_enum"]

    atomic = plan["readiness"]["atomic_ready_handoff"]
    assert atomic["prior_ledger_head_symbol"] == "H"
    assert len(atomic["ordering_exactly"]) == 7
    assert "predecessor is H" in atomic["ordering_exactly"][2]
    assert "private same-filesystem staging directory" in atomic[
        "ordering_exactly"
    ][4]
    assert "atomic no-replace directory rename" in atomic["ordering_exactly"][5]
    assert "start gate" in atomic["ordering_exactly"][6]
    assert atomic["branch_or_circular_reference_allowed"] is False

    privacy = plan["privacy"]
    assert privacy["all_operational_receipts_are_strict_allowlists"] is True
    assert privacy["final_reports_aggregate_only"] is True
    assert privacy["deidentification"]["persist_salt_or_map"] is False
    assert privacy["manifest_construction_after_hard_boundary"] is True
    assert privacy["private_raw_io_committed"] is False
    assert privacy["consumed_corpus_opened"] is False

    binding = plan["hash_binding"]
    assert binding["digest"] == "SHA-256"
    assert binding["byte_size_required"] is True
    assert binding["mutable_head_or_path_only_binding_allowed"] is False
    assert binding["append_only_artifact_predecessor_hash_required"] is True
    packet_bindings = set(binding["packet_manifest_must_bind"])
    assert {
        "release control watermark",
        "replay code-control commit and tree",
        "unchanged repair design",
        "active runtime-segment attestation and complete tuple",
        "active source-binding attestation and source-binding core",
        "first-ready scheduled slot and complete ledger predecessor chain",
        "local alias immutable snapshot",
        "alt alias immutable snapshot",
        "canonical exact two-alias snapshot-set identity",
        "complete selected holdout partition",
        "complete selected shadow partition",
        "keyed aggregate preseal receipt",
    } <= packet_bindings
    assert binding["both_readers_same_stack_hashes"] is True
    assert binding[
        "final_aggregate_evidence_preserves_runtime_source_implementation_configuration_latency_and_context_cost_hashes"
    ] is True


def test_protocol_text_contains_no_private_locator_and_states_critical_rules() -> None:
    tracked_text = "\n".join(
        (NAMESPACE / name).read_text(encoding="utf-8") for name in DOCUMENT_HASHES
    )
    assert "/home/" not in tracked_text
    assert "~/." not in tracked_text
    for phrase in (
        "created_at > release_effective_at",
        "provisional first pass",
        "terminal insufficient evidence",
        "confirmatory-shadow-v3-eval",
        "confirmatory-holdout-v3-eval",
        "joint",
        "76d34d33b7be57230ef0988662060fe70eea229f9b07f2cd5b8a2354600a79f5",
    ):
        assert phrase in tracked_text
