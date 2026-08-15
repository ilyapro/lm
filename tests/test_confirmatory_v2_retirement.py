from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = (
    ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "confirmatory-holdout-v2"
)
NOTE_PATH = NAMESPACE / "attempt-note.json"
PLAN_PATH = NAMESPACE / "analysis-plan.json"
SCANNER_PATH = ROOT / "scripts" / "ap_confirmatory_readiness.py"

FROZEN_COMMIT = "fb0307f8f101cbd2c9bdbfe6a14a9e20d012d578"
TRANSCRIPT_SHA256 = "092da8a1278af710fe393bc9d3eb8b0adf10d00119347d844836adc61caac9cf"
FROZEN_SHA256 = {
    "policy_sha256": (
        NAMESPACE / "POLICY.md",
        "096263ddf554dc014c8cd971e6129d47300bd1c0b9710da90a5749ff186a1afb",
    ),
    "analysis_plan_sha256": (
        PLAN_PATH,
        "5f5f050330b97c8a35b0e76bd31cbf1b9dd73f4d1a2baa0175817c8201a0c653",
    ),
    "repair_design_sha256": (
        NAMESPACE / "repair-design.md",
        "76d34d33b7be57230ef0988662060fe70eea229f9b07f2cd5b8a2354600a79f5",
    ),
    "scanner_sha256": (
        SCANNER_PATH,
        "ce1875bf4c442bb481cc254e3b145fb45625b483d4e98644d9365d56cfa017ba",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _note() -> dict:
    return json.loads(NOTE_PATH.read_text(encoding="utf-8"))


def test_fatal_attempt_note_is_exactly_the_aggregate_transcript_record() -> None:
    expected = {
        "schema_version": 1,
        "artifact_kind": "fatal-attempt-note",
        "namespace": "confirmatory-holdout-v2",
        "state": "fatal-before-readiness",
        "aggregate_only": True,
        "provenance": {
            "source_kind": "read-only-ae-result-transcript",
            "source_goal": (
                "retrieval-signal-and-context-cost/confirmatory-evidence-readiness"
            ),
            "transcript_sha256": TRANSCRIPT_SHA256,
            "reconstruction_only": True,
        },
        "frozen_binding": {
            "frozen_commit": FROZEN_COMMIT,
            "policy_sha256": FROZEN_SHA256["policy_sha256"][1],
            "analysis_plan_sha256": FROZEN_SHA256["analysis_plan_sha256"][1],
            "repair_design_sha256": FROZEN_SHA256["repair_design_sha256"][1],
            "scanner_path": "scripts/ap_confirmatory_readiness.py",
            "scanner_sha256": FROZEN_SHA256["scanner_sha256"][1],
        },
        "attempt": {
            "scanner_launch_count": 1,
            "consumed": True,
            "consumed_at": "process-launch-before-source-open",
            "launcher_aggregate_signal": "one_shot_scan=fatal",
            "exit_code": 2,
            "validator_valid_receipt_present": False,
            "ephemeral_snapshots_deleted": True,
            "retry_permitted": False,
        },
        "authority": {
            "grants": [],
            "readiness_receipt": False,
            "packet": False,
            "reader": False,
            "repair_implementation": False,
        },
        "privacy": {
            "private_sources_reopened_for_reconstruction": False,
            "private_raw_data_committed": False,
            "case_level_data_committed": False,
            "source_snapshot_hashes_committed": False,
            "underlying_sources_mutated": False,
        },
    }

    assert NOTE_PATH.name == "attempt-note.json"
    assert NOTE_PATH.is_file()
    assert NOTE_PATH.stat().st_size > 0
    assert _note() == expected
    assert "/home/" not in NOTE_PATH.read_text(encoding="utf-8")


def test_fatal_attempt_note_binds_unchanged_frozen_v2_inputs() -> None:
    binding = _note()["frozen_binding"]
    assert binding["frozen_commit"] == FROZEN_COMMIT
    for field, (path, expected_hash) in FROZEN_SHA256.items():
        assert path.is_file()
        assert _sha256(path) == expected_hash
        assert binding[field] == expected_hash


def test_fatal_note_is_not_a_readiness_receipt_or_packet_authority() -> None:
    note = _note()
    readiness_path = NAMESPACE / "readiness.json"
    manifest_path = NAMESPACE / "manifest.json"

    assert not readiness_path.exists()
    assert not readiness_path.is_symlink()
    assert not manifest_path.exists()
    assert not manifest_path.is_symlink()
    assert note["state"] not in {"ready", "insufficient"}
    assert note["attempt"]["validator_valid_receipt_present"] is False
    assert note["attempt"]["retry_permitted"] is False
    assert note["authority"]["grants"] == []
    assert all(
        value is False
        for key, value in note["authority"].items()
        if key != "grants"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCANNER_PATH),
            "validate",
            "--plan",
            str(PLAN_PATH),
            "--receipt",
            str(NOTE_PATH),
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 2
    assert completed.stdout == b""
    assert completed.stderr == b""


def test_frozen_v2_retry_and_reader_retirement_remain_unchanged() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    readiness = plan["readiness"]
    readers = plan["reader_authority"]

    assert readiness["scanner_launch_attempts"] == 1
    assert readiness["attempt_consumed_at"] == "process-launch-before-source-open"
    for key in (
        "retry_allowed",
        "rescan_allowed",
        "recapture_allowed",
        "supplement_allowed",
        "threshold_change_allowed",
        "alternate_partition_allowed",
        "version_in_place_allowed",
    ):
        assert readiness[key] is False

    assert set(readers["retired_consumed"]) == {
        "shadow-eval",
        "replacement-holdout-eval",
    }
    assert {entry["id"] for entry in readers["reserved_exactly"]} == {
        "confirmatory-shadow-v2-eval",
        "confirmatory-holdout-v2-eval",
    }
    assert _note()["authority"]["reader"] is False
