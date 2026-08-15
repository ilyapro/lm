"""Synthetic contract tests for the confirmatory-v2 readiness scanner."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import random
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ap_confirmatory_readiness.py"
FROZEN_PLAN = (
    REPO_ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "confirmatory-holdout-v2"
    / "analysis-plan.json"
)
CUTOFF = "2026-08-13T20:16:51Z"
WATERMARK = "2026-08-14T03:00:00Z"


@pytest.fixture(scope="module")
def scanner() -> Any:
    spec = importlib.util.spec_from_file_location("ap_confirmatory_readiness_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_line(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def _synthetic_plan(tmp_path: Path, dev_raw: bytes = b"") -> tuple[Path, str]:
    plan = copy.deepcopy(json.loads(FROZEN_PLAN.read_text(encoding="utf-8")))
    reference = plan["identity"]["complete_frozen_dev_reference"][
        "raw_identity_reference"
    ]
    reference["sha256"] = hashlib.sha256(dev_raw).hexdigest()
    reference["bytes"] = len(dev_raw)
    path = tmp_path / "synthetic-analysis-plan.json"
    raw = (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _bucket(alias: str, event_id: str) -> str:
    representative = f"{alias}\0{event_id}".encode()
    digest = hashlib.sha256(
        b"confirmatory-holdout-v2/partition/v1\0" + representative
    ).digest()
    return "holdout" if int.from_bytes(digest[:4], "big") % 100 < 50 else "shadow"


def _representative(prefix: str, partition: str, *, alias: str = "local") -> str:
    for nonce in range(10_000):
        event_id = f"a-{prefix}-{nonce:04d}"
        if _bucket(alias, event_id) == partition:
            return event_id
    raise AssertionError("synthetic bucket search exhausted")


def _event(
    event_id: str,
    *,
    query: Any,
    scope: Any,
    agent: Any,
    transport: Any,
    created_at: str = "2026-08-13T21:00:00Z",
    outcome: str = "PRIVATE-OUTCOME-ALPHA",
) -> dict[str, Any]:
    return {
        "id": event_id,
        "query": query,
        "requested_scope": scope,
        "agent": agent,
        "task": None,
        "session_id": None,
        "transport_session_id": transport,
        "created_at": created_at,
        "feedback_applied": 1,
        "results": outcome,
        "payload": outcome,
        "latency": 999999,
        "success": 0,
    }


def _ready_events() -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for family in range(30):
        representative = _representative(f"auto-{family:02d}", "holdout")
        for occurrence in range(5):
            event_id = representative if occurrence == 0 else f"{representative}~{occurrence}"
            events.append(
                _event(
                    event_id,
                    query=f"PRIVATE QUERY FAMILY {family}",
                    scope=f"project:{'red' if family % 2 else 'blue'}",
                    agent=None,
                    transport=f"PRIVATE-AUTO-SESSION-{family}-{occurrence % 2}",
                )
            )
    for occurrence in range(200):
        representative = _representative(f"organic-{occurrence:03d}", "holdout")
        events.append(
            _event(
                representative,
                query=f"PRIVATE ORGANIC QUERY {occurrence}",
                scope=f"project:{'red' if occurrence % 2 else 'blue'}",
                agent="PRIVATE-ORGANIC-AGENT",
                transport=f"PRIVATE-ORGANIC-SESSION-{occurrence}",
            )
        )
    for workflow in range(30):
        representative = _representative(f"workflow-{workflow:02d}", "shadow")
        calls = 4 if workflow < 10 else 3
        for call in range(calls):
            event_id = representative if call == 0 else f"{representative}~{call}"
            events.append(
                _event(
                    event_id,
                    query=f"PRIVATE SHADOW QUERY {workflow} {call}",
                    scope=f"project:{'red' if workflow % 2 else 'blue'}",
                    agent="PRIVATE-SHADOW-ANCHOR" if call == 0 else None,
                    transport=f"PRIVATE-WORKFLOW-{workflow}",
                )
            )
    assert len(events) == 450
    return events


def _make_snapshot(
    path: Path,
    events: Iterable[dict[str, Any]],
    *,
    source_marker: str,
) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            CREATE TABLE recall_events (
                id,
                query,
                requested_scope,
                agent,
                task,
                session_id,
                transport_session_id,
                created_at,
                feedback_applied,
                results,
                payload,
                latency,
                success
            );
            CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                level TEXT,
                content TEXT,
                scope TEXT,
                created_at TEXT
            );
            CREATE TABLE connections (
                source_id TEXT,
                target_id TEXT,
                type TEXT,
                weight REAL
            );
            CREATE TABLE retrieval_weights (
                scope TEXT,
                bm25 REAL,
                vector REAL,
                graph REAL
            );
            """
        )
        connection.execute(
            "INSERT INTO nodes VALUES (?,?,?,?,?)",
            (
                f"node-{source_marker}",
                "trace",
                f"PRIVATE-CONTENT-{source_marker}",
                "project:seed",
                "2026-08-13T00:00:00Z",
            ),
        )
        columns = (
            "id",
            "query",
            "requested_scope",
            "agent",
            "task",
            "session_id",
            "transport_session_id",
            "created_at",
            "feedback_applied",
            "results",
            "payload",
            "latency",
            "success",
        )
        connection.executemany(
            f"INSERT INTO recall_events ({','.join(columns)}) "
            f"VALUES ({','.join('?' for _ in columns)})",
            ([event[column] for column in columns] for event in events),
        )
        connection.commit()
    finally:
        connection.close()
    path.chmod(0o400)


def _empty_alt(path: Path) -> None:
    _make_snapshot(
        path,
        [
            _event(
                "PRIVATE-ALT-OLD-EVENT",
                query="PRIVATE ALT OLD QUERY",
                scope="project:old",
                agent=None,
                transport="PRIVATE-ALT-OLD-SESSION",
                created_at="2026-08-13T20:00:00Z",
            )
        ],
        source_marker="alt",
    )


def _scan(
    scanner: Any,
    *,
    plan: Path,
    plan_sha256: str,
    local: Path,
    alt: Path,
    dev: Path,
    watermark: str = WATERMARK,
) -> dict[str, Any]:
    descriptors = [os.open(path, os.O_RDONLY) for path in (local, alt, dev)]
    try:
        return scanner._scan_fds(
            plan_path=plan,
            local_input_fd=descriptors[0],
            alt_input_fd=descriptors[1],
            dev_input_fd=descriptors[2],
            watermark=watermark,
            expected_plan_sha256=plan_sha256,
            validate_policy=False,
            enforce_capture_age=False,
        )
    finally:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass


@pytest.fixture
def ready_sources(tmp_path: Path) -> tuple[Path, Path, Path, Path, str, list[dict[str, Any]]]:
    events = _ready_events()
    local = tmp_path / "PRIVATE-LOCAL-PATH.sqlite3"
    alt = tmp_path / "PRIVATE-ALT-PATH.sqlite3"
    dev = tmp_path / "PRIVATE-DEV-PATH.jsonl"
    _make_snapshot(local, events, source_marker="local")
    _empty_alt(alt)
    dev.write_bytes(b"")
    dev.chmod(0o400)
    plan, plan_sha = _synthetic_plan(tmp_path)
    return local, alt, dev, plan, plan_sha, events


def _safe_projection(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in receipt.items()
        if key
        not in {
            "capture_watermark",
            "aliased_source_snapshot_sha256_and_bytes",
            "analysis_plan_sha256",
        }
    }


def test_exact_floors_are_ready_deterministic_and_aggregate_only(
    scanner: Any,
    ready_sources: tuple[Path, Path, Path, Path, str, list[dict[str, Any]]],
) -> None:
    local, alt, dev, plan, plan_sha, _events = ready_sources
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (local, alt, dev)}
    first = _scan(scanner, plan=plan, plan_sha256=plan_sha, local=local, alt=alt, dev=dev)
    second = _scan(scanner, plan=plan, plan_sha256=plan_sha, local=local, alt=alt, dev=dev)

    assert tuple(first) == scanner.RECEIPT_FIELDS
    assert first["status"] == "ready"
    assert first["holdout_unseen_automatic_family_count"] == 30
    assert first["holdout_unseen_automatic_event_count"] == 150
    assert first["holdout_unseen_automatic_component_count"] == 30
    assert first["holdout_organic_event_count"] == 200
    assert first["holdout_organic_session_count"] == 200
    assert first["holdout_organic_component_count"] == 200
    assert first["holdout_project_scope_count"] == 2
    assert first["shadow_real_workflow_count"] == 30
    assert first["shadow_replayable_logical_call_count"] == 100
    assert first["shadow_real_workflow_component_count"] == 30
    assert first["shadow_project_scope_count"] == 2
    assert first["selected_event_count"] == 450
    assert _safe_projection(first) == _safe_projection(second)
    assert first["candidate_plan_sha256"] == second["candidate_plan_sha256"]
    assert set(first["aliased_source_snapshot_sha256_and_bytes"]) == {"local", "alt"}
    assert first["aliased_source_snapshot_sha256_and_bytes"]["local"]["sha256"] != (
        first["aliased_source_snapshot_sha256_and_bytes"]["alt"]["sha256"]
    )

    serialized = json.dumps(first, sort_keys=True)
    for private in (
        "PRIVATE QUERY",
        "PRIVATE-AUTO-SESSION",
        "PRIVATE-ORGANIC-AGENT",
        "PRIVATE-WORKFLOW",
        os.fspath(local),
        os.fspath(alt),
        os.fspath(dev),
    ):
        assert private not in serialized
    for path, (raw, mtime) in before.items():
        assert path.read_bytes() == raw
        assert path.stat().st_mtime_ns == mtime


def test_insertion_order_and_outcome_fields_do_not_change_plan_or_decision(
    scanner: Any,
    ready_sources: tuple[Path, Path, Path, Path, str, list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    local, alt, dev, plan, plan_sha, original_events = ready_sources
    baseline = _scan(scanner, plan=plan, plan_sha256=plan_sha, local=local, alt=alt, dev=dev)
    changed_events = copy.deepcopy(original_events)
    random.Random(9473).shuffle(changed_events)
    for index, event in enumerate(changed_events):
        event["feedback_applied"] = index % 2
        event["results"] = f"PRIVATE-CHANGED-RESULT-{index}"
        event["payload"] = "PRIVATE-CHANGED-PAYLOAD"
        event["latency"] = -index
        event["success"] = index % 3
    changed = tmp_path / "PRIVATE-CHANGED-OUTCOMES.sqlite3"
    _make_snapshot(changed, changed_events, source_marker="changed-local")
    replay = _scan(scanner, plan=plan, plan_sha256=plan_sha, local=changed, alt=alt, dev=dev)

    assert (
        baseline["aliased_source_snapshot_sha256_and_bytes"]["local"]["sha256"]
        != replay["aliased_source_snapshot_sha256_and_bytes"]["local"]["sha256"]
    )
    assert _safe_projection(baseline) == _safe_projection(replay)


def test_missing_sessions_remain_selected_but_fail_closed_as_insufficient(
    scanner: Any,
    ready_sources: tuple[Path, Path, Path, Path, str, list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    _local, alt, dev, plan, plan_sha, original_events = ready_sources
    sessionless = copy.deepcopy(original_events)
    for event in sessionless:
        event["transport_session_id"] = None
    local = tmp_path / "PRIVATE-SESSIONLESS.sqlite3"
    _make_snapshot(local, sessionless, source_marker="sessionless")
    receipt = _scan(scanner, plan=plan, plan_sha256=plan_sha, local=local, alt=alt, dev=dev)

    assert receipt["status"] == "insufficient"
    assert receipt["selected_event_count"] == 450
    assert receipt["holdout_unseen_automatic_family_count"] == 0
    assert receipt["holdout_organic_session_count"] == 0
    assert receipt["shadow_real_workflow_count"] == 0
    assert receipt["shadow_replayable_logical_call_count"] == 0


def test_empty_identity_and_unknown_scope_cannot_satisfy_replay_floors(
    scanner: Any,
    ready_sources: tuple[Path, Path, Path, Path, str, list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    _local, alt, dev, plan, plan_sha, original_events = ready_sources
    malformed = copy.deepcopy(original_events)
    for event in malformed[:150]:
        event["query"] = ""
    for event in malformed[350:]:
        event["requested_scope"] = "PRIVATE-UNKNOWN-SCOPE-KIND"
    local = tmp_path / "PRIVATE-UNREPLAYABLE.sqlite3"
    _make_snapshot(local, malformed, source_marker="unreplayable")
    receipt = _scan(scanner, plan=plan, plan_sha256=plan_sha, local=local, alt=alt, dev=dev)

    assert receipt["selected_event_count"] == 450
    assert receipt["holdout_unseen_automatic_family_count"] == 0
    assert receipt["shadow_real_workflow_count"] == 0
    assert receipt["shadow_replayable_logical_call_count"] == 0
    assert receipt["status"] == "insufficient"


def test_strict_lower_and_inclusive_watermark_selection(
    scanner: Any,
    tmp_path: Path,
) -> None:
    events = [
        _event("below", query="q", scope="project:a", agent=None, transport=None, created_at="2026-08-13T20:16:50.999999Z"),
        _event("exact-lower", query="q", scope="project:a", agent=None, transport=None, created_at=CUTOFF),
        _event("above", query="q", scope="project:a", agent=None, transport=None, created_at="2026-08-13T20:16:51.000001Z"),
        _event("exact-watermark", query="q", scope="project:a", agent=None, transport=None, created_at=WATERMARK),
        _event("future", query="q", scope="project:a", agent=None, transport=None, created_at="2026-08-14T03:00:00.000001Z"),
    ]
    local = tmp_path / "PRIVATE-BOUNDARY.sqlite3"
    alt = tmp_path / "PRIVATE-BOUNDARY-ALT.sqlite3"
    dev = tmp_path / "PRIVATE-BOUNDARY-DEV.jsonl"
    _make_snapshot(local, events, source_marker="boundary")
    _empty_alt(alt)
    dev.write_bytes(b"")
    dev.chmod(0o400)
    plan, plan_sha = _synthetic_plan(tmp_path)

    receipt = _scan(scanner, plan=plan, plan_sha256=plan_sha, local=local, alt=alt, dev=dev)
    assert receipt["selected_event_count"] == 2
    assert receipt["status"] == "insufficient"


def test_source_qualification_keeps_equal_raw_ids_and_sessions_distinct(
    scanner: Any,
    tmp_path: Path,
) -> None:
    shared = _event(
        "PRIVATE-SHARED-EVENT-ID",
        query="PRIVATE SHARED QUERY",
        scope="project:shared",
        agent=None,
        transport="PRIVATE-SHARED-SESSION",
    )
    local = tmp_path / "PRIVATE-SOURCE-QUALIFIED-LOCAL.sqlite3"
    alt = tmp_path / "PRIVATE-SOURCE-QUALIFIED-ALT.sqlite3"
    dev = tmp_path / "PRIVATE-SOURCE-QUALIFIED-DEV.jsonl"
    _make_snapshot(local, [shared], source_marker="qualified-local")
    _make_snapshot(alt, [shared], source_marker="qualified-alt")
    dev.write_bytes(b"")
    dev.chmod(0o400)
    plan, plan_sha = _synthetic_plan(tmp_path)

    receipt = _scan(scanner, plan=plan, plan_sha256=plan_sha, local=local, alt=alt, dev=dev)
    assert receipt["selected_event_count"] == 2
    assert receipt["holdout_unseen_automatic_family_count"] == 0


def test_complete_raw_dev_reference_excludes_seen_family(
    scanner: Any,
    ready_sources: tuple[Path, Path, Path, Path, str, list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    local, alt, _dev, _plan, _plan_sha, _events = ready_sources
    dev_id = next(
        f"PRIVATE-DEV-ID-{number}"
        for number in range(10_000)
        if 15
        <= int.from_bytes(
            hashlib.sha256(f"PRIVATE-DEV-ID-{number}".encode()).digest()[:8], "big"
        )
        % 100
        < 66
    )
    dev_raw = _json_line(
        {
            "event_id": dev_id,
            "query": "PRIVATE QUERY FAMILY 0",
            "requested_scope": "project:blue",
            "agent": None,
            "private_unused_outcome": "PRIVATE-DEV-OUTCOME",
        }
    )
    dev = tmp_path / "PRIVATE-SEEN-DEV.jsonl"
    dev.write_bytes(dev_raw)
    dev.chmod(0o400)
    plan, plan_sha = _synthetic_plan(tmp_path, dev_raw)
    receipt = _scan(scanner, plan=plan, plan_sha256=plan_sha, local=local, alt=alt, dev=dev)

    assert receipt["status"] == "insufficient"
    assert receipt["holdout_unseen_automatic_family_count"] == 29
    assert receipt["holdout_unseen_automatic_event_count"] == 145
    assert receipt["holdout_unseen_automatic_component_count"] == 29


def test_equal_source_or_observed_source_mutation_is_fatal(
    scanner: Any,
    ready_sources: tuple[Path, Path, Path, Path, str, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local, alt, dev, plan, plan_sha, _events = ready_sources
    with pytest.raises(scanner.IntegrityFailure):
        _scan(scanner, plan=plan, plan_sha256=plan_sha, local=local, alt=local, dev=dev)

    original = scanner._run_keyed_worker

    def mutate_after_worker(**kwargs: Any) -> dict[str, Any]:
        result = original(**kwargs)
        local.chmod(0o600)
        with local.open("ab") as stream:
            stream.write(b"PRIVATE-MUTATION")
        return result

    monkeypatch.setattr(scanner, "_run_keyed_worker", mutate_after_worker)
    with pytest.raises(scanner.IntegrityFailure):
        _scan(scanner, plan=plan, plan_sha256=plan_sha, local=local, alt=alt, dev=dev)


def test_capture_envelope_measures_capture_start_not_scan_completion(scanner: Any) -> None:
    launch = datetime.now(UTC)
    watermark = launch - timedelta(hours=2)
    scanner._validate_capture_envelope(
        watermark=watermark,
        capture_starts={
            "local": (watermark + timedelta(seconds=1)).isoformat(),
            "alt": (watermark + timedelta(seconds=60)).isoformat(),
        },
        observed_launch=launch,
        maximum_delay_seconds=60,
    )

    invalid_envelopes = [
        None,
        {"local": watermark.isoformat()},
        {
            "local": (watermark - timedelta(microseconds=1)).isoformat(),
            "alt": watermark.isoformat(),
        },
        {
            "local": watermark.isoformat(),
            "alt": (watermark + timedelta(seconds=60, microseconds=1)).isoformat(),
        },
    ]
    for envelope in invalid_envelopes:
        with pytest.raises(scanner.IntegrityFailure):
            scanner._validate_capture_envelope(
                watermark=watermark,
                capture_starts=envelope,
                observed_launch=launch,
                maximum_delay_seconds=60,
            )

    with pytest.raises(scanner.IntegrityFailure):
        scanner._validate_capture_envelope(
            watermark=launch + timedelta(microseconds=1),
            capture_starts={
                "local": launch.isoformat(),
                "alt": launch.isoformat(),
            },
            observed_launch=launch,
            maximum_delay_seconds=60,
        )


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "invalid-time", "missing-column", "dangling-state"],
)
def test_structural_integrity_failures_are_fatal(
    scanner: Any,
    tmp_path: Path,
    mutation: str,
) -> None:
    local = tmp_path / f"PRIVATE-{mutation}.sqlite3"
    alt = tmp_path / f"PRIVATE-{mutation}-alt.sqlite3"
    dev = tmp_path / f"PRIVATE-{mutation}-dev.jsonl"
    events = [
        _event("same", query="q", scope="project:a", agent=None, transport="s")
    ]
    if mutation == "duplicate":
        events.append(copy.deepcopy(events[0]))
    elif mutation == "invalid-time":
        events[0]["created_at"] = "PRIVATE-NOT-A-UTC-INSTANT"
    _make_snapshot(local, events, source_marker=mutation)
    if mutation == "missing-column":
        local.chmod(0o600)
        connection = sqlite3.connect(local)
        try:
            connection.execute("ALTER TABLE recall_events RENAME TO old_events")
            connection.execute("CREATE TABLE recall_events (id, created_at)")
            connection.commit()
        finally:
            connection.close()
        local.chmod(0o400)
    elif mutation == "dangling-state":
        local.chmod(0o600)
        connection = sqlite3.connect(local)
        try:
            connection.execute(
                "INSERT INTO connections VALUES (?,?,?,?)",
                ("missing-node", "also-missing", "related", 1.0),
            )
            connection.commit()
        finally:
            connection.close()
        local.chmod(0o400)
    _empty_alt(alt)
    dev.write_bytes(b"")
    dev.chmod(0o400)
    plan, plan_sha = _synthetic_plan(tmp_path)

    with pytest.raises(scanner.IntegrityFailure):
        _scan(scanner, plan=plan, plan_sha256=plan_sha, local=local, alt=alt, dev=dev)


def _receipt_document(scanner: Any, *, ready: bool) -> dict[str, Any]:
    counts = {
        "holdout_unseen_automatic_family_count": 30 if ready else 29,
        "holdout_unseen_automatic_event_count": 150,
        "holdout_unseen_automatic_component_count": 30 if ready else 29,
        "holdout_organic_event_count": 200,
        "holdout_organic_session_count": 30,
        "holdout_organic_component_count": 30,
        "holdout_project_scope_count": 2,
        "shadow_real_workflow_count": 30,
        "shadow_replayable_logical_call_count": 100,
        "shadow_real_workflow_component_count": 30,
        "shadow_project_scope_count": 2,
        "selected_event_count": 450,
    }
    watermark = (datetime.now(UTC) - timedelta(seconds=1)).isoformat().replace(
        "+00:00", "Z"
    )
    return {
        "schema_version": 2,
        "namespace": "confirmatory-holdout-v2",
        "status": "ready" if ready else "insufficient",
        "fixed_lower_bound": CUTOFF,
        "capture_watermark": watermark,
        "aliased_source_snapshot_sha256_and_bytes": {
            "local": {"sha256": "1" * 64, "bytes": 4096},
            "alt": {"sha256": "2" * 64, "bytes": 8192},
        },
        **counts,
        "candidate_plan_sha256": "3" * 64,
        "analysis_plan_sha256": scanner.PINNED_ANALYSIS_PLAN_SHA256,
    }


@pytest.mark.parametrize("ready", [True, False])
def test_downstream_validate_accepts_consistent_ready_and_insufficient_receipts(
    scanner: Any,
    tmp_path: Path,
    ready: bool,
) -> None:
    receipt = tmp_path / f"readiness-{ready}.json"
    receipt.write_text(
        json.dumps(_receipt_document(scanner, ready=ready), indent=2) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            os.fspath(SCRIPT),
            "validate",
            "--plan",
            os.fspath(FROZEN_PLAN),
            "--receipt",
            os.fspath(receipt),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0
    assert completed.stdout == b""
    assert completed.stderr == b""


def test_validator_rejects_privacy_fields_status_lies_and_duplicate_keys_silently(
    scanner: Any,
    tmp_path: Path,
) -> None:
    base = _receipt_document(scanner, ready=True)
    variants: list[bytes] = []
    with_private = dict(base)
    with_private["query"] = "PRIVATE-QUERY-MUST-NOT-PASS"
    variants.append(json.dumps(with_private).encode())
    wrong_status = dict(base)
    wrong_status["status"] = "insufficient"
    variants.append(json.dumps(wrong_status).encode())
    impossible_families = dict(_receipt_document(scanner, ready=False))
    impossible_families["holdout_unseen_automatic_family_count"] = 0
    impossible_families["holdout_unseen_automatic_component_count"] = 0
    variants.append(json.dumps(impossible_families).encode())
    impossible_workflows = dict(_receipt_document(scanner, ready=False))
    impossible_workflows["shadow_real_workflow_count"] = 0
    impossible_workflows["shadow_real_workflow_component_count"] = 0
    variants.append(json.dumps(impossible_workflows).encode())
    canonical = json.dumps(base)
    variants.append((canonical[:-1] + ',"status":"ready"}').encode())

    for index, raw in enumerate(variants):
        receipt = tmp_path / f"invalid-{index}.json"
        receipt.write_bytes(raw)
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                os.fspath(SCRIPT),
                "validate",
                "--plan",
                os.fspath(FROZEN_PLAN),
                "--receipt",
                os.fspath(receipt),
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        assert completed.returncode != 0
        assert completed.stdout == b""
        assert completed.stderr == b""


def test_public_scan_rejects_unpinned_dev_without_leaking_fd_inputs(
    scanner: Any,
    tmp_path: Path,
) -> None:
    local = tmp_path / "PRIVATE-CLI-LOCAL.sqlite3"
    alt = tmp_path / "PRIVATE-CLI-ALT.sqlite3"
    dev = tmp_path / "PRIVATE-CLI-DEV.jsonl"
    _make_snapshot(local, [], source_marker="cli-local")
    _make_snapshot(alt, [], source_marker="cli-alt")
    dev.write_bytes(b"PRIVATE-FAKE-DEV-REFERENCE\n")
    dev.chmod(0o400)
    descriptors = [os.open(path, os.O_RDONLY) for path in (local, alt, dev)]
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                os.fspath(SCRIPT),
                "scan",
                "--plan",
                os.fspath(FROZEN_PLAN),
                "--local-snapshot-fd",
                str(descriptors[0]),
                "--alt-snapshot-fd",
                str(descriptors[1]),
                "--dev-reference-fd",
                str(descriptors[2]),
                "--capture-watermark",
                datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "--local-capture-start",
                datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "--alt-capture-start",
                datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "--receipt",
                "-",
            ],
            cwd=REPO_ROOT,
            pass_fds=tuple(descriptors),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
    assert completed.returncode != 0
    assert json.loads(completed.stdout) == {"status": "insufficient"}
    assert completed.stderr == b""


def test_partition_vectors_and_overlap_guard(scanner: Any) -> None:
    assert scanner._partition_for_representative(b"local\0evt-a") == "holdout"
    assert scanner._partition_for_representative(b"alt\0evt-a") == "shadow"
    assert scanner._partition_for_representative(b"local\0evt-b") == "holdout"
    with pytest.raises(scanner.IntegrityFailure):
        scanner._assert_one_partition([0, 1], ["holdout", "shadow"])


def test_candidate_plan_uses_temporal_order_before_byte_tiebreakers(scanner: Any) -> None:
    later_raw_sorts_first = scanner.Event(
        alias="local",
        event_id="event-later",
        created_at="2026-08-13T21:00:00.5+00:00",
        created_instant=datetime.fromisoformat("2026-08-13T21:00:00.5+00:00"),
        query="private later",
        requested_scope="project:a",
        agent=None,
        task=None,
        session_id=None,
        transport_session_id=None,
        token=None,
        replayable=True,
    )
    earlier_raw_sorts_last = scanner.Event(
        alias="local",
        event_id="event-earlier",
        created_at="2026-08-13T21:00:00Z",
        created_instant=datetime.fromisoformat("2026-08-13T21:00:00+00:00"),
        query="private earlier",
        requested_scope="project:a",
        agent=None,
        task=None,
        session_id=None,
        transport_session_id=None,
        token=None,
        replayable=True,
    )
    events = [later_raw_sorts_first, earlier_raw_sorts_last]
    representatives = [event.key_bytes for event in events]
    partitions = [_bucket(event.alias, event.event_id) for event in events]

    records = []
    for event, representative, partition in (
        (earlier_raw_sorts_last, representatives[1], partitions[1]),
        (later_raw_sorts_first, representatives[0], partitions[0]),
    ):
        arm_bit = hashlib.sha256(
            b"confirmatory-holdout-v2/arm-order/v1\0" + event.key_bytes
        ).digest()[0] & 1
        records.append(
            {
                "source_qualified_event_key": event.key_text,
                "created_at": event.created_at,
                "connected_component_representative": representative.decode(),
                "partition": partition,
                "structural_validity": True,
                "family_floor_eligibility": False,
                "workflow_floor_eligibility": False,
                "replayability": True,
                "paired_arm_dispatch_order": arm_bit,
            }
        )
    expected = hashlib.sha256(
        json.dumps(
            records,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    assert scanner._candidate_plan_digest(
        events,
        representatives=representatives,
        partitions=partitions,
        qualifying_event_indices=set(),
        counted_workflows=set(),
    ) == expected


def test_scanner_has_no_production_or_evaluator_dependency() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "from living_memory" not in source
    assert "import living_memory" not in source
    assert "replacement-holdout/recipe" not in source
    assert "ap_baseline" not in source
    assert "corpus/" not in source
