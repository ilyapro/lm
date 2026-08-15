from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = (
    ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "confirmatory-holdout-v2"
)
PLAN_PATH = NAMESPACE / "analysis-plan.json"

DOCUMENT_HASHES = {
    "POLICY.md": "096263ddf554dc014c8cd971e6129d47300bd1c0b9710da90a5749ff186a1afb",
    "README.md": "4455e267979284ffbe09c704de6f51069e2fd733e3092bddbbc5f2f482ac90e4",
    "analysis-plan.json": "5f5f050330b97c8a35b0e76bd31cbf1b9dd73f4d1a2baa0175817c8201a0c653",
}

AUTHORIZED_INPUT_HASHES = {
    "artifacts/animal-planet/manifest.json": (
        "3f1a6a87a4d34e411f06e50161f9203afc3c4992dfc973d4f544453e8bbfde48"
    ),
    "artifacts/animal-planet/corpus/splits.json": (
        "66f04caa41e0302fd7b8582f54197a396a2bb8483ddd543e564740a8e4c3a7b7"
    ),
    "artifacts/animal-planet/evaluation/replacement-holdout/POLICY.md": (
        "05d4f3d148b86a933c107341ac1157487509efe011428dc4858c18b9b1c28a90"
    ),
    "artifacts/animal-planet/evaluation/replacement-holdout/manifest.json": (
        "fd36c972b049c296acbd2537f9af8f1db8d7db725f188c5dd90d27ab9aa3ab83"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_namespace_documents_are_nonempty_valid_and_byte_locked() -> None:
    assert set(DOCUMENT_HASHES) == {"POLICY.md", "README.md", "analysis-plan.json"}
    for name, expected_hash in DOCUMENT_HASHES.items():
        path = NAMESPACE / name
        assert path.is_file()
        assert path.stat().st_size > 0
        assert _sha256(path) == expected_hash

    plan = _plan()
    assert plan["schema_version"] == 2
    assert plan["namespace"] == "confirmatory-holdout-v2"
    assert plan["status"] == "preregistered-source-unprobed-unsealed"
    assert plan["frozen"] is True
    assert plan["amendment_policy"].startswith("never;")


def test_preregistration_is_source_blind_and_binds_only_authorized_inputs() -> None:
    prereg = _plan()["source_blind_preregistration"]
    assert prereg["availability_values_used"] is False
    assert prereg["repair_implementation_started"] is False
    assert prereg["reads_before_freeze"] == {
        "candidate_private_rows": 0,
        "candidate_source_aggregate_queries": 0,
        "original_sealed_corpus_rows": 0,
        "replacement_sealed_corpus_rows": 0,
        "consumed_packet_verifier_invocations": 0,
    }

    declared = {item["path"]: item["sha256"] for item in prereg["allowed_inputs"]}
    assert declared == AUTHORIZED_INPUT_HASHES
    for relative_path, expected_hash in AUTHORIZED_INPUT_HASHES.items():
        assert _sha256(ROOT / relative_path) == expected_hash


def test_consumed_readers_are_retired_and_only_two_v2_readers_exist() -> None:
    readers = _plan()["reader_authority"]
    assert set(readers["retired_consumed"]) == {
        "shadow-eval",
        "replacement-holdout-eval",
    }

    reserved = readers["reserved_exactly"]
    assert len(readers["retired_consumed"]) == 2
    assert len(reserved) == 2
    assert {entry["id"] for entry in reserved} == {
        "confirmatory-shadow-v2-eval",
        "confirmatory-holdout-v2-eval",
    }
    assert {entry["id"]: entry["partition"] for entry in reserved} == {
        "confirmatory-shadow-v2-eval": "shadow",
        "confirmatory-holdout-v2-eval": "holdout",
    }
    assert all(entry["max_process_launches"] == 1 for entry in reserved)
    assert all(entry["max_semantic_passes"] == 1 for entry in reserved)
    assert readers["aliases_allowed"] is False
    assert readers["delegation_allowed"] is False
    assert readers["wildcards_allowed"] is False
    assert readers["fallback_readers_allowed"] is False
    assert set(readers["launch_consumes_on"]) == {
        "success",
        "partial_read",
        "crash",
        "post_open_validation_failure",
        "inconclusive_result",
    }
    assert readers["execution_order"] == [
        "confirmatory-shadow-v2-eval",
        "confirmatory-holdout-v2-eval",
    ]
    assert readers["second_launch_independent_of_first_outcome"] is True
    assert readers["identical_frozen_stack_for_both"] is True


def test_sources_are_exactly_local_and_alt_read_only_aliases() -> None:
    plan = _plan()
    sources = plan["sources"]
    assert set(sources["aliases"]) == {"local", "alt"}
    assert plan["selection"]["source_aliases_exactly"] == ["local", "alt"]
    for source in sources["aliases"].values():
        assert source["locator"] == "external-untracked-operator-mapping"
        assert source["source_access"] == "read-only"
        assert source["snapshot_open"] == "mode=ro&immutable=1&cache=private"
        assert source["query_only"] is True
        assert source["temp_store"] == "MEMORY"
    assert sources["raw_locators_in_artifacts"] is False
    assert sources["snapshot_capture_attempts"] == 1
    assert sources["alias_authorities_exactly"] == {
        "local": "local",
        "alt": "ssh-alias:alt",
    }
    assert sources["distinct_authorities_and_database_instances_required"] is True
    assert sources["same_source_under_both_aliases"] == "fatal-integrity-error"
    assert sources["equal_snapshot_sha256"] == "fatal-integrity-error"
    assert sources["capture_watermark_rule"].startswith(
        "trusted launcher current UTC instant recorded at readiness process launch"
    )
    assert sources["future_dated_watermark_allowed"] is False
    assert sources["maximum_snapshot_capture_start_delay_seconds"] == 60
    assert sources["sleep_or_wait_for_accrual_after_watermark_allowed"] is False
    assert sources["same_snapshot_pair_required_for_readiness_and_packet"] is True

    tracked_text = "\n".join(
        (NAMESPACE / name).read_text(encoding="utf-8") for name in DOCUMENT_HASHES
    )
    assert "/home/" not in tracked_text
    assert "~/." not in tracked_text


def test_selection_is_complete_strictly_post_cutoff_and_outcome_blind() -> None:
    selection = _plan()["selection"]
    predicate = selection["predicate"]
    assert predicate == {
        "created_at": {
            "gt": "2026-08-13T20:16:51Z",
            "lte": "capture_watermark_fixed_before_source_open",
        },
        "population": (
            "source-qualified union of every recall event present in either exact "
            "immutable snapshot"
        ),
        "scope_filter": None,
        "outcome_filtering": False,
    }
    assert "gte" not in predicate["created_at"]
    assert selection["classification"] == {
        "automatic": "agent IS NULL",
        "organic": "agent IS NOT NULL",
    }
    assert selection["all_eligible_events_included"] is True
    assert selection["floors_are_not_inclusion_filters"] is True
    assert selection["required_structural_fields"] == {
        "source_alias": "exactly local or alt",
        "event_id": "nonempty UTF-8 string",
        "created_at": "valid UTC instant",
    }
    assert selection["missing_or_invalid_structural_field"] == (
        "fatal-integrity-error; never drop"
    )
    assert selection["missing_nonselection_metadata_events_remain_selected"] is True
    assert selection["missing_family_identity_action"].startswith(
        "create no automatic-family edge"
    )
    assert selection["missing_session_action"].startswith(
        "create no session/workflow edge"
    )
    assert selection["no_max_n"] is True
    assert selection["no_stop_at_floor"] is True
    assert selection["duplicate_source_qualified_event_key"] == "fatal-integrity-error"

    cutoff = _utc(predicate["created_at"]["gt"])
    for vector in selection["boundary_vectors"]:
        assert (_utc(vector["created_at"]) > cutoff) is vector["included"]

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
    }
    assert expected_forbidden <= set(selection["forbidden_filter_fields"])
    assert selection["consumed_packet_disjointness"] == {
        "proof": "strict nonoverlapping manifest-bound temporal intervals",
        "open_consumed_corpora": False,
        "invoke_consumed_verifiers": False,
    }


def test_family_definition_and_dev_novelty_are_frozen() -> None:
    identity = _plan()["identity"]
    assert identity["algorithm"] == "HMAC-SHA256"
    assert identity["normalization"] == '" ".join(query.split()) + "\\n" + requested_scope'
    dev_reference = identity["complete_frozen_dev_reference"]
    assert dev_reference["public_dev_corpus"] == {
        "path": "artifacts/animal-planet/corpus/dev.jsonl",
        "sha256_from_frozen_manifest": (
            "605e4f3fdc91817788a40d0dbbabffb16b86ef98ad28020c20379ef249cc9a36"
        ),
        "identity_comparison_input": False,
        "reason": (
            "independently de-identified query surrogates cannot establish "
            "cross-build equality"
        ),
    }
    raw_reference = dev_reference["raw_identity_reference"]
    assert raw_reference["sha256"] == (
        "45dbfdb389324a8702ba1a84b28be028991f23dcfe1d7ab9d771928c37c2cc7a"
    )
    assert raw_reference["bytes"] == 4162914
    assert raw_reference["processing"].startswith("opened only inside the same isolated")
    assert raw_reference["unavailable_or_hash_mismatch"].startswith(
        "fatal readiness integrity failure"
    )
    family = identity["repeated_automatic_family"]
    assert family["population"] == "automatic events only"
    assert family["identity"].startswith("same source_alias plus equal ephemeral HMAC token")
    assert family["minimum_events"] == 3
    assert family["minimum_distinct_nonempty_source_qualified_transport_sessions"] == 2
    assert family["organic_contribution"] == 0
    assert identity["unseen_in_dev"] == (
        "family token absent from the complete frozen-dev automatic reference"
    )
    assert identity["persist_key"] is False
    assert identity["persist_key_commitment"] is False
    assert identity["persist_hmac_tokens_or_samples"] is False
    assert identity["persist_unkeyed_fingerprint"] is False


def test_partition_rule_has_golden_vectors_and_cluster_closure() -> None:
    partition = _plan()["partition"]
    assert partition["component_edges"][0].startswith(
        "shared source-qualified automatic identity equality class for every automatic event"
    )
    assert partition["domain_separator_utf8"] == "confirmatory-holdout-v2/partition/v1"
    assert partition["algorithm"] == "SHA-256"
    assert partition["bucket_integer"] == "int.from_bytes(digest[0:4], 'big')"
    assert partition["modulus"] == 100
    assert partition["holdout_buckets"] == {"gte": 0, "lt": 50}
    assert partition["shadow_buckets"] == {"gte": 50, "lt": 100}

    domain = partition["domain_separator_utf8"].encode("utf-8") + b"\0"
    for vector in partition["golden_vectors"]:
        representative = vector["representative_display"].replace("\\0", "\0").encode()
        digest = hashlib.sha256(domain + representative).digest()
        assert digest.hex() == vector["digest_sha256"]
        bucket = int.from_bytes(digest[:4], "big") % partition["modulus"]
        assert bucket == vector["bucket"]
        expected_partition = "holdout" if bucket < 50 else "shadow"
        assert vector["partition"] == expected_partition

    vectors = {v["representative_display"]: v for v in partition["golden_vectors"]}
    assert vectors["local\\0evt-a"]["digest_sha256"] != vectors["alt\\0evt-a"]["digest_sha256"]
    assert partition["event_intersection_required"] == 0
    assert partition["event_union_equals_selected_population"] is True
    assert partition["automatic_family_cross_partition_count_required"] == 0
    assert partition["organic_session_cross_partition_count_required"] == 0
    assert partition["workflow_cross_partition_count_required"] == 0
    assert partition["input_order_invariant"] is True
    assert partition["outcome_field_invariant"] is True
    assert partition["source_qualified_keys"] is True
    for forbidden_adaptation in (
        "key_search_allowed",
        "rebalancing_allowed",
        "truncation_allowed",
        "stratification_allowed",
        "reassignment_allowed",
    ):
        assert partition[forbidden_adaptation] is False

    # Synthetic transitive closure: identity A links sessions s1/s2, and s2 links
    # identity B, so all four events are one component even if B is not repeated.
    events = {
        "a1": ("identity-a", "session-1"),
        "a2": ("identity-a", "session-2"),
        "b1": ("identity-b", "session-2"),
        "b2": ("identity-b", "session-3"),
    }
    parent = {event_id: event_id for event_id in events}

    def find(event_id: str) -> str:
        while parent[event_id] != event_id:
            parent[event_id] = parent[parent[event_id]]
            event_id = parent[event_id]
        return event_id

    def union(left: str, right: str) -> None:
        parent[find(right)] = find(left)

    event_ids = list(events)
    for index, left in enumerate(event_ids):
        for right in event_ids[index + 1 :]:
            if set(events[left]) & set(events[right]):
                union(left, right)
    assert len({find(event_id) for event_id in event_ids}) == 1


def test_a_priori_floors_are_conjunctive_partition_specific_and_not_observed() -> None:
    plan = _plan()
    readiness = plan["readiness"]
    assert readiness["initial_status"] == "unprobed"
    assert readiness["receipt_status_allowed"] == ["ready", "insufficient"]
    assert readiness["decision_operator"] == "all"
    assert readiness["holdout_floor"] == {
        "unseen_in_dev_repeated_automatic_families_min": 30,
        "unseen_in_dev_repeated_automatic_events_min": 150,
        "distinct_components_containing_qualifying_unseen_automatic_family_min": 30,
        "organic_events_min": 200,
        "distinct_nonempty_source_qualified_organic_sessions_min": 30,
        "distinct_components_containing_organic_event_min": 30,
        "distinct_requested_project_scopes_min": 2,
    }
    assert readiness["shadow_floor"] == {
        "genuine_nonsynthetic_real_workflows_min": 30,
        "unique_replayable_logical_calls_min": 100,
        "distinct_components_containing_counted_workflow_min": 30,
        "distinct_requested_project_scopes_min": 2,
    }
    assert readiness["post_run_evidence_floor"] == {
        "complete_paired_measured_calls_min": 100,
        "distinct_workflows_with_at_least_one_complete_pair_min": 30,
    }
    assert readiness["partition_pooling_allowed"] is False
    assert readiness["use_complete_population_not_minimum_prefix"] is True
    assert readiness["synthetic_generated_fixture_or_manufactured_workflow_counts"] is False
    assert readiness["workflow_counts_only_if"] == (
        "contains at least one replayable logical call; final evidence requires at least "
        "one complete measured pair in every counted workflow"
    )
    assert readiness["workflow_provenance_predicate"]["organic_anchor"].startswith(
        "at least one selected event"
    )
    assert readiness["real_workflow_proxy"].endswith(
        "no content or undeclared synthetic classifier is allowed"
    )
    assert set(readiness["replay_input_schema"]) == {
        "event_id",
        "created_at",
        "query",
        "requested_scope",
        "agent",
        "task",
        "session_id",
        "transport_session_id",
        "initial_state",
    }
    assert readiness["candidate_plan"]["canonicalization"].startswith("RFC 8785")
    assert readiness["candidate_plan"]["outcome_fields_present"] is False
    assert readiness["replay_envelope_location"].startswith(
        "hash-bound de-identified case record inside the sealed partition only"
    )
    assert readiness["missing_measurement_action"].startswith("insufficient-or-fail")

    power = plan["a_priori_power_floor"]
    assert power["derived_from_current_availability"] is False
    assert power["independence_unit"] == "connected partition component"
    assert power["minimum_independent_components_per_stream"] == 30
    assert power["automatic"]["qualifying_components"] == 30
    assert power["automatic"]["events"] == 150
    assert power["organic"]["qualifying_components"] == 30
    assert power["organic"]["events"] == 200
    assert power["shadow"]["qualifying_components"] == 30
    assert power["shadow"]["measured_calls"] == 100
    assert power["iid_values_are_design_references_not_inference"] is True
    assert power["component_bootstrap_handles_family_session_workflow_dependence"] is True


def test_readiness_publication_and_semantic_access_are_one_shot() -> None:
    plan = _plan()
    readiness = plan["readiness"]
    assert readiness["scanner_launch_attempts"] == 1
    assert readiness["attempt_consumed_at"] == "process-launch-before-source-open"
    for forbidden_retry in (
        "retry_allowed",
        "rescan_allowed",
        "recapture_allowed",
        "supplement_allowed",
        "threshold_change_allowed",
        "alternate_partition_allowed",
        "version_in_place_allowed",
    ):
        assert readiness[forbidden_retry] is False
    assert readiness["insufficient_action"] == (
        "emit aggregate gap, publish no packet, begin no repair implementation, and close v2"
    )

    publication = plan["publication_and_ordering"]
    assert publication["packet_publication_attempts"] == 1
    assert publication["no_overwrite"] is True
    assert publication["content_before_manifest"] is True
    assert publication["manifest_last"] is True
    assert publication["sealed_state"] == {
        "canonical_manifest_present": True,
        "manifest_hash_valid": True,
        "frozen": True,
        "readiness_status": "ready",
        "keyed_preseal_status": "pass",
        "keyed_preseal_mismatches": 0,
        "semantic_reads": {
            "confirmatory-shadow-v2-eval": 0,
            "confirmatory-holdout-v2-eval": 0,
        },
        "consumption_record_location": "separate hash-bound aggregate receipts only",
    }
    assert publication["partial_publication_action"] == "abandon-v2"
    assert publication["reseal_allowed"] is False
    assert publication["repair_before_both_partitions_sealed"] is False
    assert publication["semantic_evaluator_before_both_partitions_sealed"] is False
    assert (
        publication[
            "packet_seal_commit_must_be_strict_ancestor_of_repair_and_evaluator_commits"
        ]
        is True
    )
    assert publication["timestamps_alone_sufficient"] is False


def test_bootstrap_seeds_clusters_and_interval_are_frozen() -> None:
    uncertainty = _plan()["uncertainty"]
    assert uncertainty["supplementary_only"] is True
    assert uncertainty["decision_basis"] == "full-sample parent point estimates"
    assert uncertainty["bootstrap_mean_is_point_estimate"] is False
    assert uncertainty["interval_can_rescue_failed_point_gate"] is False
    assert uncertainty["resamples"] == 10000
    assert uncertainty["confidence_level"] == 0.95
    assert uncertainty["interval"] == {
        "method": "two-sided percentile nearest-rank",
        "sorted_zero_based_lower_index": 249,
        "sorted_zero_based_upper_index": 9749,
    }
    sampling = uncertainty["sampling"]
    assert sampling["algorithm"] == "SHA-256 counter sampler v1"
    assert sampling["whole_component_resampling"] is True
    assert sampling["paired_observations_kept_together"] is True
    assert sampling["replicate_estimator"].startswith(
        "recompute the exact corresponding full-sample metric formula"
    )

    expected_labels = {
        "automatic_component": "automatic-component",
        "organic_component": "organic-component",
        "shadow_component": "shadow-component",
    }
    assert set(uncertainty["seeds"]) == set(expected_labels)
    for stream, label in expected_labels.items():
        declared = uncertainty["seeds"][stream]
        assert declared["label"] == label
        digest = hashlib.sha256(b"confirmatory-holdout-v2\0" + label.encode()).digest()
        assert declared["value"] == int.from_bytes(digest[:4], "big")
    assert uncertainty["invalid_replicate_action"].endswith(
        "this never changes the parent full-sample point-gate result"
    )


def test_uncertainty_cannot_weaken_or_replace_parent_point_gates() -> None:
    measurement = _plan()["measurement"]
    gates = measurement["parent_point_gates"]
    assert gates["unseen_repeated_automatic_character_reduction"] == {
        "operator": "gte",
        "value": 0.5,
        "cluster_unit": (
            "connected component containing a qualifying unseen automatic family"
        ),
    }
    assert gates["automatic_content_access"] == {
        "operator": "gte",
        "value": 0.95,
        "cluster_unit": (
            "connected component containing a qualifying unseen automatic family"
        ),
    }
    for name in ("organic_payload_delta", "organic_access_delta"):
        assert gates[name]["operator"] == "within-inclusive"
        assert gates[name]["lower"] == -0.05
        assert gates[name]["upper"] == 0.05
        assert gates[name]["cluster_unit"] == (
            "connected component containing a selected organic event"
        )
    assert gates["agent_triggered_delivery"]["operator"] == "byte-identical"
    assert gates["agent_triggered_delivery"]["reference"] == "gating-off paired delivery"
    assert gates["agent_triggered_delivery"]["maximum_violations"] == 0
    assert gates["novel_nodes_content_bearing"]["minimum_fraction"] == 1.0
    assert gates["novel_nodes_content_bearing"]["zero_eligible_population"] == "fail"
    for name in ("shadow_latency_p50_degradation", "shadow_latency_p95_degradation"):
        assert gates[name]["operator"] == "lte"
        assert gates[name]["value"] == 0.1
        assert gates[name]["cluster_unit"] == (
            "connected component containing a counted shadow workflow"
        )
    assert measurement["all_parent_point_gates_conjunctive"] is True
    assert measurement["missing_metric"] == "fail"
    assert measurement["zero_denominator"] == "fail"
    assert measurement["incomplete_pair"] == "fail"
    assert measurement["alternate_metric_or_cohort_allowed"] is False
    assert measurement["reference_behavior"].startswith("gating-off behavior")
    assert set(measurement["metric_definitions"]) == set(gates)
    assert measurement["trimming_or_metric_specific_case_exclusion_allowed"] is False
    assert set(measurement["cohorts"]) == {
        "unseen_repeated_automatic",
        "organic",
        "agent_triggered",
        "shadow",
        "novel_node_opportunity",
    }
    assert measurement["serialized_character_measure"]["serialization"] == (
        "Python json.dumps(value, ensure_ascii=False) with default separators and "
        "insertion order under the hash-bound runtime"
    )
    assert measurement["latency_measure"]["empirical_quantile"].endswith(
        "independently for p=0.50 and p=0.95"
    )
    assert measurement["state_isolation"]["cross_arm_or_cross_reader_state_reuse"] is False
    state = measurement["state_isolation"]
    assert state["state_granularity"] == "one seed state per partition and source alias"
    copied = state["construction"]["copied_table_column_allowlist"]
    assert copied == {
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
    forbidden_seed_field_fragments = {
        "feedback",
        "result",
        "payload",
        "latency",
        "access",
        "history",
        "fingerprint",
        "signal",
    }
    assert not any(
        fragment in field
        for fields in copied.values()
        for field in fields
        for fragment in forbidden_seed_field_fragments
    )
    assert state["construction"]["all_other_source_tables_or_columns"].startswith(
        "not copied"
    )
    assert state["construction"]["recall_events"] == (
        "empty at seed; no pre-cutoff or candidate recall event is copied"
    )
    assert set(state["construction"]["forced_empty_state"]) == {
        "recall_events",
        "automatic fingerprint or signal history",
        "delivery history",
        "runtime sessions",
        "request and embedding caches",
        "pending feedback",
        "feedback and outcome tables",
        "temporary tables",
    }
    assert state["reset_rule"].startswith("never reuse a clone")
    resolver = measurement["content_ref_resolver"]
    assert resolver["shares_state_cache_or_process_with_primary_calls"] is False
    assert resolver["resolver_payload_or_latency_in_primary_metrics"] is False
    assert measurement["arm_schedule"] == {
        "domain_separator_utf8": "confirmatory-holdout-v2/arm-order/v1",
        "digest_input": (
            "domain_separator_utf8 + NUL + source-qualified event key"
        ),
        "order": (
            "if SHA-256(digest_input)[0] & 1 == 0 dispatch gating-off then candidate, "
            "else candidate then gating-off"
        ),
        "publication": (
            "computed before implementation from the private source-qualified key and "
            "stored only as a one-bit dispatch-order field in the sealed case"
        ),
        "evaluator_input": (
            "the reader consumes only the sealed one-bit dispatch-order field and never "
            "needs the destroyed raw event key or mapping"
        ),
        "barrier": (
            "both paired responses complete before either arm advances to the next "
            "canonical event"
        ),
        "outcome_adaptive": False,
    }


def test_privacy_receipt_allowlist_and_hash_binding_are_explicit() -> None:
    plan = _plan()
    allowlist = set(plan["readiness"]["readiness_receipt_allowlist"])
    assert allowlist == {
        "schema_version",
        "namespace",
        "status",
        "fixed_lower_bound",
        "capture_watermark",
        "aliased_source_snapshot_sha256_and_bytes",
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
        "selected_event_count",
        "candidate_plan_sha256",
        "analysis_plan_sha256",
    }
    assert plan["readiness"]["individual_or_per_source_counts_allowed"] is False

    privacy = plan["privacy"]
    assert privacy["readiness_output_is_allowlist_only"] is True
    assert privacy["final_reports_aggregate_only"] is True
    assert privacy["sealed_corpus_only_allowance"] == [
        (
            "freshly de-identified outcome-free case records with query and content "
            "surrogates required for replay"
        ),
        (
            "fresh opaque packet-local event, node, family, scope, agent, task, session, "
            "and workflow equality labels unrelated to HMAC output"
        ),
    ]
    assert privacy["deidentification"]["persist_salt_or_map"] is False
    assert privacy["key_transport"] == "inherited anonymous descriptors only"
    assert privacy["manifest_construction_after_hard_boundary"] is True
    assert privacy["private_raw_io_committed"] is False

    binding = plan["hash_binding"]
    assert binding["digest"] == "SHA-256"
    assert binding["byte_size_required"] is True
    assert binding["mutable_head_or_path_only_binding_allowed"] is False
    packet_bindings = set(binding["packet_manifest_must_bind"])
    assert packet_bindings == {
        "local alias immutable snapshot",
        "alt alias immutable snapshot",
        "distinct source-authority attestations",
        "capture watermark",
        "POLICY.md",
        "README.md",
        "analysis-plan.json",
        "original frozen manifest",
        "original split declaration",
        "complete frozen-dev reference",
        "raw frozen-dev identity reference",
        "retired replacement manifest",
        "retired replacement policy",
        "readiness scanner and synthetic tests",
        "candidate plan",
        "builder",
        "verifier",
        "de-identification implementation",
        "holdout partition",
        "shadow partition",
        "partition/source seed states and state-construction manifest",
        "keyed aggregate preseal receipt",
    }
    evaluation_bindings = set(binding["evaluation_receipts_must_bind"])
    assert evaluation_bindings == {
        "sealed v2 packet manifest",
        "clean repair Git commit and tree",
        "semantic evaluator",
        "calculator",
        "configuration",
        "threshold registry",
        "bootstrap plan and seeds",
        "dependency and environment lock",
        "runtime settings",
        "partition initial-state and pre-execution arm-clone hashes",
        "process and cache initialization configuration",
        "content_ref resolver code, configuration, and isolated state",
        "deterministic paired-arm schedule",
        "reader identifier and partition",
    }
    assert binding["both_readers_same_stack_hashes"] is True
    assert binding["final_aggregate_evidence_preserves_source_and_implementation_hashes"] is True
