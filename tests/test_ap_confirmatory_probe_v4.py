"""Synthetic adversarial tests for the aggregate-only confirmatory-v4 probe."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import os
import random
import re
import sqlite3
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import pytest

from living_memory.storage import MemoryStore


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ap_confirmatory_probe_v4.py"
FROZEN_PLAN = (
    REPO_ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "confirmatory-holdout-v4"
    / "analysis-plan.json"
)
RELEASE = "2026-08-14T15:03:46.793603Z"
SLOT_ZERO = "2026-08-17T00:00:00.000000Z"


@pytest.fixture(scope="module")
def probe() -> Any:
    spec = importlib.util.spec_from_file_location("ap_confirmatory_probe_v4_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _failure(probe: Any, call: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    with pytest.raises(probe.IntegrityFailure) as caught:
        call(*args, **kwargs)
    assert caught.value.args == ()
    assert str(caught.value) == ""


def _json_line(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _synthetic_plan(tmp_path: Path, dev_raw: bytes = b"") -> tuple[bytes, str]:
    plan = json.loads(FROZEN_PLAN.read_text(encoding="utf-8"))
    reference = plan["identity"]["complete_frozen_dev_reference"][
        "raw_identity_reference"
    ]
    reference["sha256"] = hashlib.sha256(dev_raw).hexdigest()
    reference["bytes"] = len(dev_raw)
    raw = (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    (tmp_path / "synthetic-plan.json").write_bytes(raw)
    return raw, hashlib.sha256(raw).hexdigest()


def _bucket(alias: str, event_id: str) -> str:
    representative = f"{alias}\0{event_id}".encode()
    digest = hashlib.sha256(
        b"confirmatory-holdout-v4/partition/v1\0" + representative
    ).digest()
    return "holdout" if int.from_bytes(digest[:4], "big") % 100 < 50 else "shadow"


def _representative(prefix: str, partition: str, *, alias: str = "local") -> str:
    for nonce in range(20_000):
        event_id = f"a-{prefix}-{nonce:05d}"
        if _bucket(alias, event_id) == partition:
            return event_id
    raise AssertionError("synthetic bucket search exhausted")


def _dev_member_id(prefix: str = "dev") -> str:
    for nonce in range(20_000):
        event_id = f"{prefix}-{nonce:05d}"
        digest = hashlib.sha256(event_id.encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") % 100
        if 15 <= bucket < 66:
            return event_id
    raise AssertionError("dev split search exhausted")


def _resolved(scope: str, ambient: dict[str, Any]) -> list[str]:
    if scope == "global":
        return ["global"]
    if scope.startswith("project:"):
        return [scope, "global"]
    project = ambient.get("project_scope") or ambient.get("project")
    values = [scope]
    if project:
        normalized = str(project)
        if not normalized.startswith("project:"):
            normalized = f"project:{normalized}"
        values.append(normalized)
    values.append("global")
    return list(dict.fromkeys(values))


def _event(
    event_id: str,
    *,
    query: Any = "query",
    scope: Any = "project:blue",
    agent: Any = None,
    transport: Any = "transport",
    created_at: str = "2026-08-16T00:00:00.000000Z",
    ambient_extra: dict[str, Any] | None = None,
    resolved_scopes: Any | None = None,
    depth: Any = "1",
    max_results: Any = 5,
    outcome: str = "PRIVATE-OUTCOME-A",
) -> dict[str, Any]:
    ambient: dict[str, Any] = {}
    if agent is not None:
        ambient["agent"] = agent
    if transport is not None:
        ambient["transport_session_id"] = transport
    ambient.update(ambient_extra or {})
    if resolved_scopes is None and type(scope) is str:
        resolved_scopes = _resolved(scope, ambient)
    return {
        "id": event_id,
        "query": query,
        "scope": scope,
        "requested_scope": scope,
        "resolved_scopes": json.dumps(resolved_scopes),
        "ambient_context": json.dumps(ambient),
        "depth": depth,
        "max_results": max_results,
        "results": json.dumps([{"private": outcome}]),
        "agent": agent,
        "task": None,
        "session_id": None,
        "transport_session_id": transport,
        "feedback_applied": 1,
        "feedback_applied_at": "2099-01-01T00:00:00Z",
        "gated": 1,
        "created_at": created_at,
    }


def _write_snapshot(
    path: Path,
    events: Iterable[dict[str, Any]],
    *,
    marker: str,
    shuffle_seed: int | None = None,
) -> None:
    # Production creates the schema and its four retrieval-policy defaults.
    with MemoryStore(path):
        pass
    rows = list(events)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute(
            "INSERT INTO kv (key,value,updated_at) VALUES (?,?,?)",
            (f"source-{marker}", marker, "2026-08-15T00:00:00Z"),
        )
        columns = (
            "id",
            "query",
            "scope",
            "requested_scope",
            "resolved_scopes",
            "ambient_context",
            "depth",
            "max_results",
            "results",
            "agent",
            "task",
            "session_id",
            "transport_session_id",
            "feedback_applied",
            "feedback_applied_at",
            "gated",
            "created_at",
        )
        connection.executemany(
            f"INSERT INTO recall_events ({','.join(columns)}) "
            f"VALUES ({','.join('?' for _ in columns)})",
            ([event[column] for column in columns] for event in rows),
        )
        connection.commit()
    finally:
        connection.close()
    path.chmod(0o400)


def _compute(
    probe: Any,
    *,
    tmp_path: Path,
    local_events: Iterable[dict[str, Any]],
    alt_events: Iterable[dict[str, Any]] = (),
    dev_raw: bytes = b"",
    slot_index: int = 0,
    lower: str = RELEASE,
    shuffle_seed: int | None = None,
) -> Any:
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan_raw, plan_sha = _synthetic_plan(tmp_path, dev_raw)
    dev = tmp_path / "dev.jsonl"
    dev.write_bytes(dev_raw)
    dev.chmod(0o400)
    local = tmp_path / "local.sqlite3"
    alt = tmp_path / "alt.sqlite3"
    _write_snapshot(local, local_events, marker="local", shuffle_seed=shuffle_seed)
    _write_snapshot(alt, alt_events, marker="alt", shuffle_seed=shuffle_seed)
    descriptors = [os.open(path, os.O_RDONLY) for path in (local, alt, dev)]
    try:
        return probe._compute_aggregate_fds(
            plan_raw=plan_raw,
            expected_plan_sha256=plan_sha,
            local_input_fd=descriptors[0],
            alt_input_fd=descriptors[1],
            dev_input_fd=descriptors[2],
            slot_index=slot_index,
            active_lower_bound_exclusive_at=lower,
        )
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def _ready_events() -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for family in range(30):
        representative = _representative(f"auto-{family:02d}", "holdout")
        for occurrence in range(5):
            event_id = representative if occurrence == 0 else f"{representative}~{occurrence}"
            events.append(
                _event(
                    event_id,
                    query=f"PRIVATE FAMILY {family}",
                    scope=f"project:{'red' if family % 2 else 'blue'}",
                    transport=f"auto-session-{family}-{occurrence % 2}",
                )
            )
    for occurrence in range(200):
        representative = _representative(f"organic-{occurrence:03d}", "holdout")
        events.append(
            _event(
                representative,
                query=f"PRIVATE ORGANIC {occurrence}",
                scope=f"project:{'red' if occurrence % 2 else 'blue'}",
                agent="organic-agent",
                transport=f"organic-session-{occurrence}",
            )
        )
    for workflow in range(30):
        representative = _representative(f"workflow-{workflow:02d}", "shadow")
        call_count = 4 if workflow < 10 else 3
        for call in range(call_count):
            event_id = representative if call == 0 else f"{representative}~{call}"
            events.append(
                _event(
                    event_id,
                    query=f"PRIVATE SHADOW {workflow} {call}",
                    scope=f"project:{'red' if workflow % 2 else 'blue'}",
                    agent="shadow-anchor" if call == 0 else None,
                    transport=f"workflow-{workflow}",
                )
            )
    assert len(events) == 450
    return events


def test_frozen_allowlists_match_the_worker_and_resolution_surface(probe: Any) -> None:
    plan = json.loads(FROZEN_PLAN.read_text(encoding="utf-8"))
    assert len(probe.AGGREGATE_FIELDS) == 14
    assert tuple(plan["readiness"]["probe_receipt_allowlist"]) == probe.PROBE_RESOLUTION_FIELDS
    assert tuple(
        plan["operational_receipt_schemas"]["slot_probe_resolution"]["fields_exactly"]
    ) == probe.PROBE_RESOLUTION_FIELDS
    forbidden = {"candidate_plan_sha256", "component_map", "identity_map", "case"}
    assert forbidden.isdisjoint(probe.AGGREGATE_FIELDS)


@pytest.mark.parametrize(
    ("representative", "partition"),
    [
        (b"local\0evt-a", "shadow"),
        (b"alt\0evt-a", "holdout"),
        (b"local\0evt-c", "shadow"),
    ],
)
def test_partition_golden_vectors(probe: Any, representative: bytes, partition: str) -> None:
    assert probe._partition_for_representative(representative) == partition


def test_partition_golden_vectors_are_the_ones_the_frozen_plan_publishes(
    probe: Any,
) -> None:
    """The hardcoded vectors above must be the plan's, not a convenient rewrite."""

    plan = json.loads(FROZEN_PLAN.read_text(encoding="utf-8"))
    published = plan["partition"]["golden_vectors"]
    assert len(published) == 3
    for vector in published:
        alias, _, event_id = vector["representative_display"].partition("\\0")
        representative = alias.encode("utf-8") + b"\0" + event_id.encode("utf-8")
        digest = hashlib.sha256(probe.PARTITION_DOMAIN + representative).digest()
        assert digest.hex() == vector["digest_sha256"]
        assert int.from_bytes(digest[:4], "big") % 100 == vector["bucket"]
        assert probe._partition_for_representative(representative) == vector["partition"]


def test_retired_partition_domain_assigns_different_partitions(probe: Any) -> None:
    """A v3 partition implementation cannot reproduce v4 assignments.

    Same representatives, same algorithm, different domain separator.  Every
    digest differs, and the published ``local\\0evt-a`` vector even lands in
    the other partition: v4 says shadow where v3 said holdout.  That is what
    stops a retired accrual from being laundered through this namespace.
    """

    retired_domain = b"confirmatory-holdout-v3/partition/v1\0"
    assert probe.PARTITION_DOMAIN != retired_domain
    representatives = (b"local\0evt-a", b"alt\0evt-a", b"local\0evt-c")
    retired_partitions = []
    for representative in representatives:
        retired_digest = hashlib.sha256(retired_domain + representative).digest()
        active_digest = hashlib.sha256(probe.PARTITION_DOMAIN + representative).digest()
        assert retired_digest != active_digest
        retired_bucket = int.from_bytes(retired_digest[:4], "big") % 100
        retired_partitions.append("holdout" if retired_bucket < 50 else "shadow")
    active_partitions = [
        probe._partition_for_representative(item) for item in representatives
    ]
    assert retired_partitions == ["holdout", "holdout", "shadow"]
    assert active_partitions == ["shadow", "holdout", "shadow"]
    assert retired_partitions != active_partitions


def test_selection_uses_both_strict_lowers_and_slot_inclusive_upper(
    probe: Any, tmp_path: Path
) -> None:
    events = [
        _event("release-exact", created_at=RELEASE),
        _event("release-plus", created_at="2026-08-14T15:03:46.793604Z"),
        _event("slot-exact", created_at=SLOT_ZERO),
        _event("slot-plus", created_at="2026-08-17T00:00:00.000001Z"),
    ]
    result = _compute(probe, tmp_path=tmp_path, local_events=events)
    assert result.aggregate()["selected_event_count"] == 2
    assert result.aggregate()["selected_replayable_event_count"] == 2


def test_successor_lower_is_strict_and_can_only_narrow_selection(
    probe: Any, tmp_path: Path
) -> None:
    lower = "2026-08-16T12:00:00.000000Z"
    events = [
        _event("before", created_at="2026-08-16T11:59:59.999999Z"),
        _event("exact", created_at=lower),
        _event("after", created_at="2026-08-16T12:00:00.000001Z"),
    ]
    result = _compute(probe, tmp_path=tmp_path, local_events=events, lower=lower)
    assert result.aggregate()["selected_event_count"] == 1


def test_requested_slot_is_the_only_upper_bound(probe: Any, tmp_path: Path) -> None:
    event = _event("after-slot-zero", created_at="2026-08-17T00:00:00.000001Z")
    first = _compute(probe, tmp_path=tmp_path / "first", local_events=[event], slot_index=0)
    second = _compute(probe, tmp_path=tmp_path / "second", local_events=[event], slot_index=1)
    assert first.aggregate()["selected_event_count"] == 0
    assert second.aggregate()["selected_event_count"] == 1
    assert first.scheduled_at == probe.runtime.slot_times(0).scheduled_at
    assert second.scheduled_at == probe.runtime.slot_times(1).scheduled_at


def test_exact_floor_population_is_private_provisional_ready_and_deterministic(
    probe: Any, tmp_path: Path
) -> None:
    events = _ready_events()
    first = _compute(
        probe, tmp_path=tmp_path / "first", local_events=events, shuffle_seed=11
    )
    second_events = copy.deepcopy(events)
    for event in second_events:
        event["results"] = json.dumps([{"private": "DIFFERENT-OUTCOME"}])
        event["feedback_applied"] = 0
        event["feedback_applied_at"] = None
        event["gated"] = 0
    second = _compute(
        probe, tmp_path=tmp_path / "second", local_events=second_events, shuffle_seed=97
    )
    assert first.aggregate() == second.aggregate()
    assert first.provisional_ready is True
    assert first.aggregate() == {
        "selected_event_count": 450,
        "selected_replayable_event_count": 450,
        "selected_nonreplayable_event_count": 0,
        "holdout_unseen_automatic_family_count": 30,
        "holdout_unseen_automatic_event_count": 150,
        "holdout_unseen_automatic_component_count": 30,
        "holdout_organic_event_count": 200,
        "holdout_organic_session_count": 200,
        "holdout_organic_component_count": 200,
        "holdout_project_scope_count": 2,
        "shadow_real_workflow_count": 30,
        "shadow_replayable_logical_call_count": 100,
        "shadow_real_workflow_component_count": 30,
        "shadow_project_scope_count": 2,
    }


@pytest.mark.parametrize("population_seed", [3, 17, 41, 89])
def test_randomized_populations_are_key_order_and_outcome_invariant(
    probe: Any, tmp_path: Path, population_seed: int
) -> None:
    generator = random.Random(population_seed)
    by_alias: dict[str, list[tuple[dict[str, Any], bool]]] = {
        "local": [],
        "alt": [],
    }
    for index in range(60):
        alias = generator.choice(list(probe.SOURCE_ALIASES))
        invalid = generator.choice(("none", "none", "unknown", "query", "plan", "max"))
        event = _event(
            f"random-{population_seed}-{index:03d}",
            query="" if invalid == "query" else f"family-{generator.randrange(9)}",
            scope=f"project:p{generator.randrange(3)}",
            agent="organic" if generator.randrange(4) == 0 else None,
            transport=(
                None
                if generator.randrange(5) == 0
                else f"session-{generator.randrange(12)}"
            ),
            ambient_extra={"unknown": index} if invalid == "unknown" else None,
            resolved_scopes=["global"] if invalid == "plan" else None,
            max_results=0 if invalid == "max" else generator.randrange(1, 15),
            outcome=f"private-{generator.randrange(1000)}",
        )
        by_alias[alias].append((event, invalid == "none"))

    first = _compute(
        probe,
        tmp_path=tmp_path / "first",
        local_events=[event for event, _ in by_alias["local"]],
        alt_events=[event for event, _ in by_alias["alt"]],
        shuffle_seed=population_seed,
    )
    mutated: dict[str, list[dict[str, Any]]] = {"local": [], "alt": []}
    for alias in probe.SOURCE_ALIASES:
        for event, _ in reversed(by_alias[alias]):
            changed = copy.deepcopy(event)
            changed["results"] = json.dumps(
                [{"private": f"mutated-{population_seed}"}]
            )
            changed["feedback_applied"] = 1 - int(changed["feedback_applied"])
            changed["gated"] = 1 - int(changed["gated"])
            mutated[alias].append(changed)
    second = _compute(
        probe,
        tmp_path=tmp_path / "second",
        local_events=mutated["local"],
        alt_events=mutated["alt"],
        shuffle_seed=population_seed + 1,
    )
    assert first.aggregate() == second.aggregate()
    assert first.aggregate()["selected_event_count"] == 60
    assert first.aggregate()["selected_replayable_event_count"] == sum(
        replayable
        for events in by_alias.values()
        for _, replayable in events
    )
    assert (
        first.aggregate()["selected_replayable_event_count"]
        + first.aggregate()["selected_nonreplayable_event_count"]
        == 60
    )


def test_nonreplayable_automatic_peer_stays_selected_and_component_linked(
    probe: Any, tmp_path: Path
) -> None:
    representative = _representative("nonreplayable-peer", "holdout")
    events = []
    for index in range(4):
        events.append(
            _event(
                representative if index == 0 else f"{representative}~{index}",
                query="same identity",
                transport=f"session-{index % 2}",
                ambient_extra={"unknown": "forbidden"} if index == 3 else None,
            )
        )
    result = _compute(probe, tmp_path=tmp_path, local_events=events)
    counts = result.aggregate()
    assert counts["selected_event_count"] == 4
    assert counts["selected_replayable_event_count"] == 3
    assert counts["selected_nonreplayable_event_count"] == 1
    assert counts["holdout_unseen_automatic_family_count"] == 1
    assert counts["holdout_unseen_automatic_event_count"] == 3
    assert counts["holdout_unseen_automatic_component_count"] == 1


def test_input_metadata_failures_do_not_filter_selected_events(
    probe: Any, tmp_path: Path
) -> None:
    base = _representative("metadata", "holdout")
    events = [
        _event(base, query="valid", agent="", transport="valid-session"),
        _event(f"{base}~1", query="", transport="empty-query"),
        _event(f"{base}~2", ambient_extra={"unknown": 1}, transport="unknown-context"),
        _event(f"{base}~3", resolved_scopes=["global"], transport="wrong-plan"),
        _event(f"{base}~4", depth=b"not-text", transport="bad-depth"),
        _event(f"{base}~5", max_results=0, transport="bad-max"),
    ]
    result = _compute(probe, tmp_path=tmp_path, local_events=events)
    counts = result.aggregate()
    assert counts["selected_event_count"] == 6
    assert counts["selected_replayable_event_count"] == 1
    assert counts["selected_nonreplayable_event_count"] == 5
    assert counts["holdout_organic_event_count"] == 1


@pytest.mark.parametrize(
    "depth", [None, "", "01", "+1", "-1", "-0", "unknown", "decision"]
)
def test_depth_replayability_matches_serving_string_normalization(
    probe: Any, depth: Any
) -> None:
    assert probe._depth_accepted(depth) is True


def test_scope_resolver_and_provenance_equivalence_are_exact(
    probe: Any, tmp_path: Path
) -> None:
    events = [
        _event("valid-session", scope="session:s", ambient_extra={"project": "p"}),
        _event("trimmed-request", scope="project:p "),
        _event("bad-ambient", scope="project:p", ambient_extra={"scope": "bad:x"}),
        _event("task-mismatch", ambient_extra={"task": "task"}),
        _event("session-mismatch", ambient_extra={"session_id": "session"}),
        _event("typed-transport-mismatch", transport=7),
    ]
    result = _compute(probe, tmp_path=tmp_path, local_events=events)
    assert result.aggregate()["selected_event_count"] == 6
    assert result.aggregate()["selected_replayable_event_count"] == 1
    assert result.aggregate()["selected_nonreplayable_event_count"] == 5


def test_shadow_workflow_requires_a_replayable_organic_anchor(
    probe: Any, tmp_path: Path
) -> None:
    representative = _representative("workflow-anchor", "shadow")
    events = [
        _event(
            representative,
            agent="organic",
            transport="workflow",
            ambient_extra={"unknown": "makes-anchor-nonreplayable"},
        ),
        _event(f"{representative}~1", agent=None, transport="workflow"),
        _event(f"{representative}~2", agent=None, transport="workflow"),
    ]
    result = _compute(probe, tmp_path=tmp_path, local_events=events)
    assert result.aggregate()["shadow_real_workflow_count"] == 0
    assert result.aggregate()["shadow_replayable_logical_call_count"] == 0


def test_identity_and_transport_edges_are_source_qualified(
    probe: Any, tmp_path: Path
) -> None:
    local_rep = _representative("qualified-local", "holdout", alias="local")
    alt_rep = _representative("qualified-alt", "holdout", alias="alt")
    local = [
        _event(local_rep, query="same", transport="same-session"),
        _event(f"{local_rep}~1", query="same", transport="other-local"),
    ]
    alt = [
        _event(alt_rep, query="same", transport="same-session"),
        _event(f"{alt_rep}~1", query="same", transport="other-alt"),
    ]
    result = _compute(probe, tmp_path=tmp_path, local_events=local, alt_events=alt)
    assert result.aggregate()["holdout_unseen_automatic_family_count"] == 0


def test_complete_raw_dev_reference_excludes_seen_family(probe: Any, tmp_path: Path) -> None:
    representative = _representative("seen-family", "holdout")
    query = "seen family identity"
    events = [
        _event(
            representative if index == 0 else f"{representative}~{index}",
            query=query,
            transport=f"session-{index % 2}",
        )
        for index in range(3)
    ]
    dev_raw = _json_line(
        {
            "event_id": _dev_member_id(),
            "query": query,
            "requested_scope": "project:blue",
            "agent": None,
            "private_outcome": "MUST-NOT-BE-READ",
        }
    )
    result = _compute(
        probe, tmp_path=tmp_path, local_events=events, dev_raw=dev_raw
    )
    assert result.aggregate()["holdout_unseen_automatic_family_count"] == 0


@pytest.mark.parametrize(
    "created_at",
    ["2026-08-16T00:00:00+00:00", "2026-08-16T00:00:00.0000000Z", "bad"],
)
def test_invalid_structural_timestamp_is_fatal_not_dropped(
    probe: Any, tmp_path: Path, created_at: str
) -> None:
    plan_raw, plan_sha = _synthetic_plan(tmp_path)
    dev = tmp_path / "dev.jsonl"
    dev.write_bytes(b"")
    local = tmp_path / "local.sqlite3"
    alt = tmp_path / "alt.sqlite3"
    _write_snapshot(local, [_event("bad-time", created_at=created_at)], marker="local")
    _write_snapshot(alt, [], marker="alt")
    fds = [os.open(path, os.O_RDONLY) for path in (local, alt, dev)]
    try:
        _failure(
            probe,
            probe._compute_aggregate_fds,
            plan_raw=plan_raw,
            expected_plan_sha256=plan_sha,
            local_input_fd=fds[0],
            alt_input_fd=fds[1],
            dev_input_fd=fds[2],
            slot_index=0,
            active_lower_bound_exclusive_at=RELEASE,
        )
    finally:
        for fd in fds:
            os.close(fd)


def test_duplicate_source_qualified_event_key_is_fatal(probe: Any, tmp_path: Path) -> None:
    plan_raw, plan_sha = _synthetic_plan(tmp_path)
    dev = tmp_path / "dev.jsonl"
    dev.write_bytes(b"")
    local = tmp_path / "local.sqlite3"
    alt = tmp_path / "alt.sqlite3"
    _write_snapshot(local, [_event("duplicate")], marker="local")
    _write_snapshot(alt, [], marker="alt")
    local.chmod(0o600)
    connection = sqlite3.connect(local)
    try:
        connection.executescript(
            """
            ALTER TABLE recall_events RENAME TO recall_events_original;
            CREATE TABLE recall_events AS SELECT * FROM recall_events_original;
            INSERT INTO recall_events SELECT * FROM recall_events_original;
            DROP TABLE recall_events_original;
            """
        )
        connection.commit()
    finally:
        connection.close()
    local.chmod(0o400)
    fds = [os.open(path, os.O_RDONLY) for path in (local, alt, dev)]
    try:
        _failure(
            probe,
            probe._compute_aggregate_fds,
            plan_raw=plan_raw,
            expected_plan_sha256=plan_sha,
            local_input_fd=fds[0],
            alt_input_fd=fds[1],
            dev_input_fd=fds[2],
            slot_index=0,
            active_lower_bound_exclusive_at=RELEASE,
        )
    finally:
        for fd in fds:
            os.close(fd)


def test_worker_result_allowlist_rejects_extra_boolean_and_bad_algebra(probe: Any) -> None:
    zero = {field: 0 for field in probe.AGGREGATE_FIELDS}
    assert probe._validate_worker_result(zero) == zero
    extra = {**zero, "case_count": 0}
    _failure(probe, probe._validate_worker_result, extra)
    boolean = dict(zero)
    boolean["selected_event_count"] = False
    _failure(probe, probe._validate_worker_result, boolean)
    mismatch = dict(zero)
    mismatch["selected_event_count"] = 1
    _failure(probe, probe._validate_worker_result, mismatch)


def _time(seconds: float) -> str:
    return (
        datetime(2026, 8, 17, tzinfo=UTC) + timedelta(seconds=seconds)
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


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


def _database(alias: str) -> dict[str, Any]:
    tail = 1 + (alias == "alt")
    return {
        "filesystem_uuid": f"00000000-0000-4000-8000-{tail:012d}",
        "statx_inode_uint64": 70_000 + tail,
        "statx_birthtime_ns_int64": 1_700_000_000_000_000_000 + tail,
    }


def _active_segment(probe: Any) -> tuple[Any, Any]:
    runtime = probe.runtime
    contract = runtime.load_frozen_contract()
    observer = runtime.hash_and_bytes(b"synthetic-runtime-observer-v4")
    marker_previous = "1" * 64
    binding_previous = "2" * 64
    marker_value = {
        "schema_version": probe.SCHEMA_VERSION,
        "namespace": probe.NAMESPACE,
        "receipt_kind": "runtime-attestation-attempt",
        "attempt_scope": "initial-source-binding",
        "segment_index": 0,
        "slot_index_or_null": 0,
        "predecessor_closure_sha256_or_null": None,
        "authorized_at": runtime.slot_times(0).scheduled_at,
        "written_at": _time(0.010),
        "start_deadline_at": runtime.slot_times(0).grace_deadline_at,
        "previous_ledger_entry_sha256": marker_previous,
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
    }
    marker = runtime.validate_attestation_attempt_marker(
        runtime.canonical_json_bytes(marker_value),
        contract=contract,
        runtime_observer_identity=observer,
        expected_previous_ledger_sha256=marker_previous,
        initial_probe_launched_at=_time(0),
    )
    pairs = list(contract.initial_services.service_pairs)
    observation: dict[str, Any] = {}
    for index, alias in enumerate(probe.SOURCE_ALIASES):
        service, boot = pairs[index]
        observation[alias] = {
            "alias_id": runtime.ALIAS_IDS[alias],
            "authenticated_authority_identity": _authority(alias),
            "service_identity_sha256": service,
            "boot_identity_sha256": boot,
            "database_instance_identity": _database(alias),
        }
    binding = runtime.validate_stable_source_binding_observations(
        contract.initial_services, observation, copy.deepcopy(observation)
    )
    binding_value = {
        "schema_version": probe.SCHEMA_VERSION,
        "namespace": probe.NAMESPACE,
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
        runtime.canonical_json_bytes(binding_value),
        contract=contract,
        attempt=marker,
        binding=binding,
        runtime_observer_identity=observer,
        expected_previous_ledger_sha256=binding_previous,
    )
    return contract, runtime.make_initial_segment(contract, attestation)


def _resolution_value(
    probe: Any,
    *,
    computation: Any,
    contract: Any,
    active: Any,
    counts: dict[str, int] | None = None,
    status: str = "below-floor",
    marker: Any = None,
    sealer_at: str | None = None,
) -> dict[str, Any]:
    aggregate = computation.aggregate() if counts is None else dict(counts)
    snapshots = computation.snapshots()
    databases = active.source_binding.core.database_map()
    bindings = active.source_binding.core.binding_map()
    value: dict[str, Any] = {
        "schema_version": probe.SCHEMA_VERSION,
        "namespace": probe.NAMESPACE,
        "receipt_kind": "slot-probe-resolution",
        "status": status,
        "resolution_id": "",
        "slot_index": computation.slot_index,
        "scheduled_at": computation.scheduled_at,
        "grace_deadline_at": computation.grace_deadline_at,
        "launched_at": _time(1),
        "validated_at": _time(2),
        "release_effective_at": contract.release_effective_at,
        "active_segment_lower_bound_exclusive_at": active.lower_bound_exclusive_at,
        "segment_id": active.segment_id,
        "segment_attestation_sha256_and_bytes": active.attestation_identity.as_dict(),
        "source_binding_attestation_sha256_and_bytes": (
            active.source_binding_attestation.identity.as_dict()
        ),
        "previous_ledger_entry_sha256": "3" * 64,
        "pre_active_services_state_sha256": active.services.identity.sha256,
        "post_active_services_state_sha256": active.services.identity.sha256,
        "pre_database_instance_identity_sha256_by_alias": databases,
        "post_database_instance_identity_sha256_by_alias": dict(databases),
        "pre_alias_service_database_binding_sha256_by_alias": bindings,
        "post_alias_service_database_binding_sha256_by_alias": dict(bindings),
        "aliased_source_snapshot_sha256_and_bytes": {
            alias: snapshots[alias].as_dict() for alias in probe.SOURCE_ALIASES
        },
        **aggregate,
        "runtime_observer_sha256_and_bytes": (
            active.source_binding_attestation.runtime_observer_identity.as_dict()
        ),
        "probe_sha256_and_bytes": probe._self_identity().as_dict(),
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
        "seal_consumption_marker_sha256_and_bytes_or_null": marker,
        "sealer_process_launched_at_or_null": sealer_at,
    }
    assert tuple(value) == probe.PROBE_RESOLUTION_FIELDS
    value["resolution_id"] = probe.runtime.derive_slot_resolution_id(value)
    return value


def _probe_attempt_fixture(probe: Any) -> tuple[Any, Any, dict[str, Any], Any]:
    contract, active = _active_segment(probe)
    value = {
        "schema_version": probe.SCHEMA_VERSION,
        "namespace": probe.NAMESPACE,
        "receipt_kind": "probe-attempt",
        "slot_index": 0,
        "scheduled_at": probe.runtime.slot_times(0).scheduled_at,
        "grace_deadline_at": probe.runtime.slot_times(0).grace_deadline_at,
        "attempt_ordinal": 0,
        "launched_at": _time(1),
        "watchdog_deadline_at": _time(3601),
        "segment_id": active.segment_id,
        "previous_ledger_entry_sha256": "4" * 64,
        "runtime_observer_sha256_and_bytes": (
            active.source_binding_attestation.runtime_observer_identity.as_dict()
        ),
        "probe_sha256_and_bytes": probe._self_identity().as_dict(),
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
    }
    validated = probe.validate_probe_attempt_marker(
        probe.runtime.canonical_json_bytes(value),
        contract=contract,
        active_segment=active,
        expected_previous_ledger_sha256="4" * 64,
        expected_slot_index=0,
        expected_attempt_ordinal=0,
    )
    return contract, active, value, validated


def test_below_floor_resolution_is_exact_canonical_and_validator_valid(
    probe: Any, tmp_path: Path
) -> None:
    computation = _compute(
        probe,
        tmp_path=tmp_path,
        local_events=[_event(_representative("below", "holdout"))],
    )
    contract, active = _active_segment(probe)
    _failure(
        probe,
        probe.build_below_floor_resolution,
        computation,
        active_segment=active,
        launched_at=_time(1),
        validated_at=_time(2),
        previous_ledger_entry_sha256="3" * 64,
        contract=contract,
    )
    receipt = _resolution_value(
        probe,
        computation=computation,
        contract=contract,
        active=active,
    )
    assert tuple(receipt) == probe.PROBE_RESOLUTION_FIELDS
    assert receipt["status"] == "below-floor"
    assert receipt["seal_consumption_marker_sha256_and_bytes_or_null"] is None
    assert receipt["sealer_process_launched_at_or_null"] is None
    raw = probe.runtime.canonical_json_bytes(receipt)
    validated = probe.validate_probe_resolution(
        raw,
        contract=contract,
        active_segment=active,
        expected_previous_ledger_sha256="3" * 64,
        expected_snapshot_identities=computation.snapshots(),
    )
    assert validated.status == "below-floor"
    assert validated.identity == probe.runtime.hash_and_bytes(raw)


def test_provisional_pass_cannot_be_serialized_by_below_floor_builder(
    probe: Any, tmp_path: Path
) -> None:
    computation = _compute(
        probe, tmp_path=tmp_path, local_events=_ready_events(), shuffle_seed=3
    )
    contract, active = _active_segment(probe)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    _failure(
        probe,
        probe.build_below_floor_resolution,
        computation,
        active_segment=active,
        launched_at=_time(1),
        validated_at=_time(2),
        previous_ledger_entry_sha256="3" * 64,
        contract=contract,
    )
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert before == after


def test_attempt_and_failure_validators_preserve_retry_boundary(probe: Any) -> None:
    contract, active, attempt_value, attempt = _probe_attempt_fixture(probe)
    failure_value = {
        "schema_version": probe.SCHEMA_VERSION,
        "namespace": probe.NAMESPACE,
        "receipt_kind": "probe-failure",
        "slot_index": 0,
        "attempt_ordinal": 0,
        "probe_attempt_marker_sha256_and_bytes": attempt.identity.as_dict(),
        "failed_at": _time(2),
        "phase_at_failure": "pre-key-pre-source",
        "failure_class": "pre-source-retryable",
        "source_open_count_or_null": 0,
        "key_created_or_null": False,
        "retry_authorized": True,
        "controller_synthesized": False,
        "previous_ledger_entry_sha256": "5" * 64,
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
    }
    failure = probe.validate_probe_failure_marker(
        probe.runtime.canonical_json_bytes(failure_value),
        contract=contract,
        active_segment=active,
        attempt=attempt,
        expected_previous_ledger_sha256="5" * 64,
    )
    assert failure.retry_authorized is True
    _failure(
        probe,
        probe.validate_probe_failure_marker,
        probe.runtime.canonical_json_bytes(failure_value),
        contract=contract,
        active_segment=active,
        attempt=attempt,
        expected_previous_ledger_sha256="4" * 64,
    )
    terminalized = dict(failure_value)
    terminalized["failure_class"] = "post-source-terminal"
    terminalized["retry_authorized"] = False
    _failure(
        probe,
        probe.validate_probe_failure_marker,
        probe.runtime.canonical_json_bytes(terminalized),
        contract=contract,
        active_segment=active,
        attempt=attempt,
        expected_previous_ledger_sha256="5" * 64,
    )


def test_initial_attempt_bootstrap_and_wrapper_rebinding(probe: Any) -> None:
    contract, active, attempt_value, attempt = _probe_attempt_fixture(probe)
    observer = active.source_binding_attestation.runtime_observer_identity
    bootstrap = probe.validate_probe_attempt_marker(
        probe.runtime.canonical_json_bytes(attempt_value),
        contract=contract,
        expected_segment_id=contract.initial_segment_id,
        expected_runtime_observer_identity=observer,
        expected_previous_ledger_sha256="4" * 64,
        expected_slot_index=0,
        expected_attempt_ordinal=0,
    )
    assert bootstrap == attempt

    retry = dict(attempt_value)
    retry["attempt_ordinal"] = 1
    _failure(
        probe,
        probe.validate_probe_attempt_marker,
        probe.runtime.canonical_json_bytes(retry),
        contract=contract,
        expected_segment_id=contract.initial_segment_id,
        expected_runtime_observer_identity=observer,
        expected_previous_ledger_sha256="4" * 64,
        expected_slot_index=0,
        expected_attempt_ordinal=1,
    )
    _failure(
        probe,
        probe.validate_probe_attempt_marker,
        probe.runtime.canonical_json_bytes(attempt_value),
        contract=contract,
        active_segment=object(),
        expected_previous_ledger_sha256="4" * 64,
        expected_slot_index=0,
        expected_attempt_ordinal=0,
    )

    overflow = dict(attempt_value)
    overflow["launched_at"] = "9999-12-31T23:59:59.999999Z"
    overflow["watchdog_deadline_at"] = "9999-12-31T23:59:59.999999Z"
    _failure(
        probe,
        probe.validate_probe_attempt_marker,
        probe.runtime.canonical_json_bytes(overflow),
        contract=contract,
        active_segment=active,
        expected_previous_ledger_sha256="4" * 64,
        expected_slot_index=0,
        expected_attempt_ordinal=0,
    )

    forged_value = dict(attempt_value)
    forged_value["segment_id"] = "b" * 64
    forged_observer = probe.runtime.HashAndBytes("a" * 64, 1)
    forged_value["runtime_observer_sha256_and_bytes"] = forged_observer.as_dict()
    forged_raw = probe.runtime.canonical_json_bytes(forged_value)
    forged_attempt = replace(
        attempt,
        raw=forged_raw,
        identity=probe.runtime.hash_and_bytes(forged_raw),
        segment_id="b" * 64,
        runtime_observer_identity=forged_observer,
    )
    failure_value = {
        "schema_version": probe.SCHEMA_VERSION,
        "namespace": probe.NAMESPACE,
        "receipt_kind": "probe-failure",
        "slot_index": 0,
        "attempt_ordinal": 0,
        "probe_attempt_marker_sha256_and_bytes": forged_attempt.identity.as_dict(),
        "failed_at": _time(2),
        "phase_at_failure": "pre-key-pre-source",
        "failure_class": "pre-source-retryable",
        "source_open_count_or_null": 0,
        "key_created_or_null": False,
        "retry_authorized": True,
        "controller_synthesized": False,
        "previous_ledger_entry_sha256": "5" * 64,
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
    }
    _failure(
        probe,
        probe.validate_probe_failure_marker,
        probe.runtime.canonical_json_bytes(failure_value),
        contract=contract,
        active_segment=active,
        attempt=forged_attempt,
        expected_previous_ledger_sha256="5" * 64,
    )
    _failure(
        probe,
        probe.validate_probe_failure_marker,
        probe.runtime.canonical_json_bytes(failure_value),
        contract=contract,
        active_segment=active,
        attempt=replace(attempt, raw=b"not-an-attempt"),
        expected_previous_ledger_sha256="5" * 64,
    )


@pytest.mark.parametrize(
    ("phase", "source_count", "key_created", "failure_class", "retry"),
    [
        ("pre-key-pre-source", 0, False, "pre-source-retryable", True),
        ("key-created-pre-source", 0, True, "post-source-terminal", False),
        ("first-source-opened", 1, True, "post-source-terminal", False),
        ("snapshots-latched", 2, True, "post-source-terminal", False),
        ("keyed-count-computed", 2, True, "post-source-terminal", False),
        ("atomic-seal-handoff", 2, True, "post-source-terminal", False),
    ],
)
def test_failure_phase_matrix_is_exact(
    probe: Any,
    phase: str,
    source_count: int,
    key_created: bool,
    failure_class: str,
    retry: bool,
) -> None:
    contract, active, _, attempt = _probe_attempt_fixture(probe)
    value = {
        "schema_version": probe.SCHEMA_VERSION,
        "namespace": probe.NAMESPACE,
        "receipt_kind": "probe-failure",
        "slot_index": 0,
        "attempt_ordinal": 0,
        "probe_attempt_marker_sha256_and_bytes": attempt.identity.as_dict(),
        "failed_at": _time(2),
        "phase_at_failure": phase,
        "failure_class": failure_class,
        "source_open_count_or_null": source_count,
        "key_created_or_null": key_created,
        "retry_authorized": retry,
        "controller_synthesized": False,
        "previous_ledger_entry_sha256": "5" * 64,
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
    }
    validated = probe.validate_probe_failure_marker(
        probe.runtime.canonical_json_bytes(value),
        contract=contract,
        active_segment=active,
        attempt=attempt,
        expected_previous_ledger_sha256="5" * 64,
    )
    assert validated.retry_authorized is retry
    malformed = dict(value)
    malformed["phase_at_failure"] = []
    _failure(
        probe,
        probe.validate_probe_failure_marker,
        probe.runtime.canonical_json_bytes(malformed),
        contract=contract,
        active_segment=active,
        attempt=attempt,
        expected_previous_ledger_sha256="5" * 64,
    )


def test_unknown_failure_and_probe_terminal_are_strict(probe: Any) -> None:
    contract, active, _, attempt = _probe_attempt_fixture(probe)
    unknown = {
        "schema_version": probe.SCHEMA_VERSION,
        "namespace": probe.NAMESPACE,
        "receipt_kind": "probe-failure",
        "slot_index": 0,
        "attempt_ordinal": 0,
        "probe_attempt_marker_sha256_and_bytes": attempt.identity.as_dict(),
        "failed_at": _time(3601),
        "phase_at_failure": "unknown-after-attempt",
        "failure_class": "post-source-terminal",
        "source_open_count_or_null": None,
        "key_created_or_null": None,
        "retry_authorized": False,
        "controller_synthesized": True,
        "previous_ledger_entry_sha256": "5" * 64,
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
    }
    failure = probe.validate_probe_failure_marker(
        probe.runtime.canonical_json_bytes(unknown),
        contract=contract,
        active_segment=active,
        attempt=attempt,
        expected_previous_ledger_sha256="5" * 64,
    )
    terminal = {
        "schema_version": probe.SCHEMA_VERSION,
        "namespace": probe.NAMESPACE,
        "receipt_kind": "probe-terminal",
        "status": "terminal-probe-integrity-failure",
        "slot_index": 0,
        "attempt_ordinal": 0,
        "probe_failure_marker_sha256_and_bytes": failure.identity.as_dict(),
        "recorded_at": _time(3602),
        "previous_ledger_entry_sha256": "6" * 64,
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
    }
    raw = probe.runtime.canonical_json_bytes(terminal)
    assert probe.validate_probe_terminal_marker(
        raw,
        contract=contract,
        active_segment=active,
        failure=failure,
        expected_previous_ledger_sha256="6" * 64,
    ) == probe.runtime.hash_and_bytes(raw)
    _failure(
        probe,
        probe.validate_probe_terminal_marker,
        raw,
        contract=contract,
        active_segment=active,
        failure=failure,
        expected_previous_ledger_sha256="5" * 64,
    )
    _failure(
        probe,
        probe.validate_probe_terminal_marker,
        raw,
        contract=contract,
        active_segment=active,
        failure=replace(failure, raw=b"not-a-failure"),
        expected_previous_ledger_sha256="6" * 64,
    )


def test_ready_requires_authenticated_marker_and_launch(
    probe: Any, tmp_path: Path
) -> None:
    computation = _compute(
        probe,
        tmp_path=tmp_path,
        local_events=[_event(_representative("ready-validator", "holdout"))],
    )
    contract, active = _active_segment(probe)
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
    marker = probe.runtime.hash_and_bytes(b"validated-seal-consumption")
    launched = _time(1.5)
    receipt = _resolution_value(
        probe,
        computation=computation,
        contract=contract,
        active=active,
        counts=counts,
        status="ready",
        marker=marker.as_dict(),
        sealer_at=launched,
    )
    raw = probe.runtime.canonical_json_bytes(receipt)
    assert probe.validate_probe_resolution(
        raw,
        contract=contract,
        active_segment=active,
        expected_previous_ledger_sha256="3" * 64,
        expected_snapshot_identities=computation.snapshots(),
        expected_seal_consumption_identity=marker,
        expected_sealer_process_launched_at=launched,
    ).status == "ready"
    for missing in ("marker", "launch"):
        kwargs = {
            "expected_seal_consumption_identity": marker,
            "expected_sealer_process_launched_at": launched,
        }
        kwargs[
            "expected_seal_consumption_identity"
            if missing == "marker"
            else "expected_sealer_process_launched_at"
        ] = None
        _failure(
            probe,
            probe.validate_probe_resolution,
            raw,
            contract=contract,
            active_segment=active,
            expected_previous_ledger_sha256="3" * 64,
            expected_snapshot_identities=computation.snapshots(),
            **kwargs,
        )
    provisional = _resolution_value(
        probe,
        computation=computation,
        contract=contract,
        active=active,
        counts=counts,
        status="ready",
    )
    _failure(
        probe,
        probe.validate_probe_resolution,
        probe.runtime.canonical_json_bytes(provisional),
        contract=contract,
        active_segment=active,
        expected_previous_ledger_sha256="3" * 64,
    )


def test_snapshot_capture_map_binds_aliases_to_attested_databases(
    probe: Any, tmp_path: Path
) -> None:
    _, active = _active_segment(probe)
    databases = active.source_binding.core.database_map()
    paths = {alias: tmp_path / f"{alias}.sqlite3" for alias in probe.SOURCE_ALIASES}
    for path in paths.values():
        path.write_bytes(b"capture")
    descriptors = {alias: os.open(path, os.O_RDONLY) for alias, path in paths.items()}
    try:
        calls: list[tuple[str, str, str]] = []

        def capture(alias: str, alias_id: str, database: str) -> int:
            calls.append((alias, alias_id, database))
            return descriptors[alias]

        assert probe._capture_snapshot_fds(active, capture) == descriptors
        assert calls == [
            (alias, probe.runtime.ALIAS_IDS[alias], databases[alias])
            for alias in probe.SOURCE_ALIASES
        ]

        def duplicate(alias: str, alias_id: str, database: str) -> int:
            del alias, alias_id, database
            return descriptors["local"]

        _failure(probe, probe._capture_snapshot_fds, active, duplicate)
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


def test_production_supervisor_owns_observe_capture_observe_order(
    probe: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, active = _active_segment(probe)
    paths = {alias: tmp_path / alias for alias in probe.SOURCE_ALIASES}
    for path in paths.values():
        path.write_bytes(b"snapshot")
    descriptors = {alias: os.open(path, os.O_RDONLY) for alias, path in paths.items()}
    dev_path = tmp_path / "dev"
    dev_path.write_bytes(b"")
    dev_fd = os.open(dev_path, os.O_RDONLY)
    order: list[str] = []

    def observe() -> tuple[Any, Any]:
        order.append("observe")
        return active.services, active.source_binding

    def capture(alias: str, alias_id: str, database: str) -> int:
        assert alias_id == probe.runtime.ALIAS_IDS[alias]
        assert database == active.source_binding.core.database_map()[alias]
        order.append(f"capture:{alias}")
        return descriptors[alias]

    def fake_core(**kwargs: Any) -> Any:
        order.append("key-created")
        kwargs["source_input_supplier"]()
        order.append("worker")
        slot = probe.runtime.slot_times(kwargs["slot_index"])
        return SimpleNamespace(
            slot_index=slot.slot_index,
            scheduled_at=slot.scheduled_at,
            grace_deadline_at=slot.grace_deadline_at,
            active_segment_lower_bound_exclusive_at=(
                kwargs["active_lower_bound_exclusive_at"]
            ),
            aggregate_items=tuple((field, 0) for field in probe.AGGREGATE_FIELDS),
            snapshot_items=tuple(
                (
                    alias,
                    probe.runtime.HashAndBytes(str(index + 1) * 64, 1),
                )
                for index, alias in enumerate(probe.SOURCE_ALIASES)
            ),
            analysis_plan_identity=contract.plan_identity,
            worker_envelope_raw=b"private-worker-envelope",
            probe_identity=probe._self_identity(),
        )

    monkeypatch.setattr(probe, "_compute_aggregate_supplier", fake_core)
    try:
        computation = probe.compute_probe_aggregate(
            active_segment=active,
            observe_active_state=observe,
            capture_snapshot=capture,
            dev_reference_fd=dev_fd,
            slot_index=0,
            contract=contract,
        )
        assert order == [
            "key-created",
            "observe",
            "capture:local",
            "capture:alt",
            "observe",
            "worker",
        ]
        assert computation.segment_id == active.segment_id
    finally:
        os.close(dev_fd)
        for descriptor in descriptors.values():
            os.close(descriptor)


def test_segment_must_exist_before_requested_slot(probe: Any) -> None:
    _, active = _active_segment(probe)
    later = replace(
        active,
        lower_bound_exclusive_at="2026-08-17T00:00:03.200000Z",
    )
    _failure(probe, probe._require_segment_slot_alignment, later, 0)
    too_many_closures = replace(active, segment_index=1)
    _failure(
        probe,
        probe._require_segment_slot_alignment,
        too_many_closures,
        0,
    )


def test_computation_cannot_be_directly_forged(probe: Any) -> None:
    _failure(probe, probe.ProbeComputation)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("selection",), 1),
        (("selection", "predicate"), []),
        (("replayability", "replay_input_schema"), "bad"),
        (("operational_receipt_schemas", "slot_probe_resolution"), None),
    ],
)
def test_malformed_nested_plan_is_silent_integrity_failure(
    probe: Any, path: tuple[str, ...], replacement: Any
) -> None:
    value = json.loads(FROZEN_PLAN.read_text(encoding="utf-8"))
    parent = value
    for member in path[:-1]:
        parent = parent[member]
    parent[path[-1]] = replacement
    raw = json.dumps(value, separators=(",", ":")).encode()
    _failure(
        probe,
        probe._load_probe_plan_bytes,
        raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    )


@pytest.mark.parametrize(
    "dev_row",
    [
        {"id": "legacy-only", "query": "q", "requested_scope": "global", "agent": None},
        {
            "event_id": "canonical",
            "id": "conflict",
            "query": "q",
            "requested_scope": "global",
            "agent": None,
        },
    ],
)
def test_dev_reference_requires_unambiguous_event_id(
    probe: Any, tmp_path: Path, dev_row: dict[str, Any]
) -> None:
    _failure(
        probe,
        _compute,
        probe,
        tmp_path=tmp_path,
        local_events=[],
        dev_raw=_json_line(dev_row),
    )


def test_wal_snapshot_and_recall_view_are_rejected(
    probe: Any, tmp_path: Path
) -> None:
    wal_path = tmp_path / "wal.sqlite3"
    with MemoryStore(wal_path):
        pass
    assert wal_path.read_bytes()[18:20] == b"\x02\x02"
    descriptor = os.open(wal_path, os.O_RDONLY)
    try:
        _failure(probe, probe._open_snapshot, descriptor)
    finally:
        os.close(descriptor)

    view_path = tmp_path / "view.sqlite3"
    _write_snapshot(view_path, [], marker="view")
    view_path.chmod(0o600)
    connection = sqlite3.connect(view_path)
    try:
        columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(recall_events)")
        ]
        connection.execute("ALTER TABLE recall_events RENAME TO recall_events_raw")
        connection.execute(
            f"CREATE VIEW recall_events AS SELECT {','.join(columns)} "
            "FROM recall_events_raw"
        )
        connection.commit()
    finally:
        connection.close()
    descriptor = os.open(view_path, os.O_RDONLY)
    try:
        _failure(probe, probe._open_snapshot, descriptor)
    finally:
        os.close(descriptor)


def test_integrity_failure_cli_is_silent_and_emits_no_false_below_floor(
    tmp_path: Path,
) -> None:
    before = set(tmp_path.iterdir())
    for arguments in (("unsupported",), ("__worker", "malformed")):
        completed = subprocess.run(
            [sys.executable, "-B", os.fspath(SCRIPT), *arguments],
            cwd=tmp_path,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        assert completed.returncode != 0
        assert completed.stdout == b""
        assert completed.stderr == b""
    assert set(tmp_path.iterdir()) == before


def test_probe_source_has_no_forbidden_builder_reader_or_artifact_path() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "ap_confirmatory_runtime_v4" in source
    assert "ap_confirmatory_readiness" not in source
    assert "from living_memory" not in source
    assert "import living_memory" not in source
    for forbidden in (
        "candidate_plan_sha256",
        "corpus/dev.jsonl",
        "semantic-evaluator",
        "arm-order/v1",
        "equality-map",
    ):
        assert forbidden not in source


# --------------------------------------------------------------------------
# v4-specific boundaries
#
# The runtime core added a typed unbound predecessor and a fresh set of domain
# separators.  The probe has to honour both: it may not run ordinary probe work
# before a source binding exists, and no retired artifact may replay into this
# namespace.  Nothing below reaches a live source.
# --------------------------------------------------------------------------


def _unbound_predecessor(probe: Any) -> tuple[Any, Any]:
    runtime = probe.runtime
    contract = runtime.load_frozen_contract()
    return contract, runtime.make_unbound_watermark_predecessor(contract)


def test_unbound_watermark_predecessor_is_never_an_active_segment(
    probe: Any,
) -> None:
    """Slot 0 before the binding exists has no segment to count against.

    The predecessor carries a segment index, a segment id, a lower bound and a
    complete service tuple, so every duck-typed field the probe reads is
    present.  Only exact type identity keeps it out, and it must be kept out of
    every entry point rather than just the first one.
    """

    contract, predecessor = _unbound_predecessor(probe)
    assert type(predecessor) is probe.runtime.UnboundWatermarkPredecessor
    assert predecessor.segment_index == 0
    assert predecessor.segment_id == contract.initial_segment_id
    assert predecessor.lower_bound_exclusive_at == contract.release_effective_at

    _failure(probe, probe._require_bound_active_segment, predecessor)
    _failure(probe, probe._active_segment_recheck, contract, predecessor)
    _failure(probe, probe._require_segment_slot_alignment, predecessor, 0)


def test_every_probe_entry_point_refuses_the_unbound_predecessor(
    probe: Any, tmp_path: Path
) -> None:
    contract, predecessor = _unbound_predecessor(probe)
    _, active = _active_segment(probe)
    computation = _compute(
        probe,
        tmp_path=tmp_path,
        local_events=[_event(_representative("unbound", "holdout"))],
    )
    raw = probe.runtime.canonical_json_bytes(
        _resolution_value(
            probe, computation=computation, contract=contract, active=active
        )
    )
    _failure(
        probe,
        probe.build_below_floor_resolution,
        computation,
        active_segment=predecessor,
        launched_at=_time(1),
        validated_at=_time(2),
        previous_ledger_entry_sha256="3" * 64,
        contract=contract,
    )
    _failure(
        probe,
        probe.validate_probe_resolution,
        raw,
        contract=contract,
        active_segment=predecessor,
        expected_previous_ledger_sha256="3" * 64,
    )
    _failure(
        probe,
        probe.compute_probe_aggregate,
        active_segment=predecessor,
        observe_active_state=lambda: (_ for _ in ()).throw(AssertionError("observed")),
        capture_snapshot=lambda *_: (_ for _ in ()).throw(AssertionError("captured")),
        dev_reference_fd=-1,
        slot_index=0,
        contract=contract,
    )


def test_probe_refuses_a_resolution_bound_to_another_segment(
    probe: Any, tmp_path: Path
) -> None:
    """Counts may not be re-presented under a segment that did not produce them.

    A closed segment's receipt keeps its bytes, but those bytes bind a segment
    id, an attestation identity, a source-binding identity and a lower bound.
    Rebinding any one of them to a different segment must fail, otherwise
    accrual could survive a runtime change.
    """

    contract, active = _active_segment(probe)
    computation = _compute(
        probe,
        tmp_path=tmp_path,
        local_events=[_event(_representative("cross", "holdout"))],
    )
    value = _resolution_value(
        probe, computation=computation, contract=contract, active=active
    )
    raw = probe.runtime.canonical_json_bytes(value)
    validated = probe.validate_probe_resolution(
        raw,
        contract=contract,
        active_segment=active,
        expected_previous_ledger_sha256="3" * 64,
    )
    assert validated.status == "below-floor"

    successor = replace(
        active,
        segment_index=1,
        segment_id="c" * 64,
        lower_bound_exclusive_at=_time(-1.0),
    )
    _failure(
        probe,
        probe.validate_probe_resolution,
        raw,
        contract=contract,
        active_segment=successor,
        expected_previous_ledger_sha256="3" * 64,
    )
    for field, replacement in (
        ("segment_id", "c" * 64),
        ("active_segment_lower_bound_exclusive_at", _time(-1.0)),
        ("pre_active_services_state_sha256", "d" * 64),
        ("post_active_services_state_sha256", "d" * 64),
    ):
        mutated = dict(value)
        mutated[field] = replacement
        mutated["resolution_id"] = ""
        mutated["resolution_id"] = probe.runtime.derive_slot_resolution_id(mutated)
        _failure(
            probe,
            probe.validate_probe_resolution,
            probe.runtime.canonical_json_bytes(mutated),
            contract=contract,
            active_segment=active,
            expected_previous_ledger_sha256="3" * 64,
        )


def test_counts_do_not_survive_into_a_differently_bound_computation(
    probe: Any, tmp_path: Path
) -> None:
    """A computation carries its own segment provenance and cannot be relabelled."""

    contract, active = _active_segment(probe)
    computation = _compute(
        probe,
        tmp_path=tmp_path,
        local_events=[_event(_representative("carry", "holdout"))],
    )
    # The FD core deliberately leaves every provenance field unbound; only the
    # production entry point rebinds them to a live segment.
    assert computation.segment_id is None
    assert computation.segment_attestation_identity is None
    assert computation.source_binding_attestation_identity is None
    assert computation.source_database_identity_items is None
    _failure(
        probe, probe._require_production_computation_binding, computation, contract, active
    )
    # Relabelling the provenance by hand is not a construction path either.
    _failure(probe, probe.ProbeComputation)


def test_probe_consumes_no_seal_or_reader_authority(probe: Any) -> None:
    """A scheduled probe is a repeatable mechanical read, not an authority draw."""

    plan = json.loads(FROZEN_PLAN.read_text(encoding="utf-8"))
    assert plan["readiness"]["probe_launches_consume_authority"] is False
    assert "aggregate-probe" in plan["authority"]["non_semantic_roles"]

    source = SCRIPT.read_text(encoding="utf-8")
    # No reserved seal or reader identity is even nameable here.
    for reserved in (
        plan["authority"]["seal"]["id"],
        *[reader["id"] for reader in plan["authority"]["reserved_readers_exactly"]],
        *plan["authority"]["retired_void_v3_readers"],
        *plan["authority"]["retired_v3_source_alias_ids"],
        plan["authority"]["retired_v3_seal_id"],
    ):
        assert reserved not in source

    # The probe can name exactly its own four receipt kinds.  Every
    # authority-bearing kind -- seal consumption, reader launch, joint release
    # -- belongs to a downstream leaf and must not be constructible here.
    # ``code`` strips the module docstring, which legitimately explains what
    # this leaf may not publish.
    code = source.split('"""', 2)[2]
    emitted = set(re.findall(r'"receipt_kind"\]?\s*[!=]?=+\s*"([a-z-]+)"', code))
    emitted |= set(re.findall(r'"receipt_kind":\s*"([a-z-]+)"', code))
    assert emitted == {
        "slot-probe-resolution",
        "probe-attempt",
        "probe-failure",
        "probe-terminal",
    }
    for authority_kind in (
        plan["operational_receipt_schemas"]["seal_consumption_marker"]["receipt_kind"],
        plan["operational_receipt_schemas"]["reader_launch_attempt_marker"][
            "receipt_kind"
        ],
        plan["operational_receipt_schemas"]["reader_launch_consumption_marker"][
            "receipt_kind"
        ],
        plan["operational_receipt_schemas"]["joint_reader_release_manifest"][
            "receipt_kind"
        ],
    ):
        assert authority_kind not in emitted
        assert authority_kind not in code


def test_below_floor_publication_holds_no_seal_or_sealer_fields(
    probe: Any, tmp_path: Path
) -> None:
    contract, active = _active_segment(probe)
    computation = _compute(
        probe,
        tmp_path=tmp_path,
        local_events=[_event(_representative("nonconsuming", "holdout"))],
    )
    value = _resolution_value(
        probe, computation=computation, contract=contract, active=active
    )
    assert value["seal_consumption_marker_sha256_and_bytes_or_null"] is None
    assert value["sealer_process_launched_at_or_null"] is None
    validated = probe.validate_probe_resolution(
        probe.runtime.canonical_json_bytes(value),
        contract=contract,
        active_segment=active,
        expected_previous_ledger_sha256="3" * 64,
    )
    assert validated.status == "below-floor"


@pytest.mark.parametrize(
    "receipt_kind",
    ["slot-probe-resolution", "probe-attempt", "probe-failure", "probe-terminal"],
)
def test_retired_schema_version_and_namespace_cannot_replay(
    probe: Any, tmp_path: Path, receipt_kind: str
) -> None:
    """Neither a v3 schema_version nor a v3 namespace validates as v4 evidence."""

    contract, active, attempt_value, attempt = _probe_attempt_fixture(probe)
    computation = _compute(
        probe,
        tmp_path=tmp_path,
        local_events=[_event(_representative("replay", "holdout"))],
    )
    if receipt_kind == "slot-probe-resolution":
        value = _resolution_value(
            probe, computation=computation, contract=contract, active=active
        )
        call: Callable[..., Any] = probe.validate_probe_resolution
        kwargs: dict[str, Any] = {
            "contract": contract,
            "active_segment": active,
            "expected_previous_ledger_sha256": "3" * 64,
        }
    elif receipt_kind == "probe-attempt":
        value = dict(attempt_value)
        call = probe.validate_probe_attempt_marker
        kwargs = {
            "contract": contract,
            "active_segment": active,
            "expected_previous_ledger_sha256": "4" * 64,
            "expected_slot_index": 0,
            "expected_attempt_ordinal": 0,
        }
    else:
        # The terminal marker is only valid over a post-source-terminal
        # failure, so the failure fixture differs per receipt kind.
        terminal_case = receipt_kind == "probe-terminal"
        failure_value = {
            "schema_version": probe.SCHEMA_VERSION,
            "namespace": probe.NAMESPACE,
            "receipt_kind": "probe-failure",
            "slot_index": 0,
            "attempt_ordinal": 0,
            "probe_attempt_marker_sha256_and_bytes": attempt.identity.as_dict(),
            "failed_at": _time(2),
            "phase_at_failure": (
                "snapshots-latched" if terminal_case else "pre-key-pre-source"
            ),
            "failure_class": (
                "post-source-terminal" if terminal_case else "pre-source-retryable"
            ),
            "source_open_count_or_null": 2 if terminal_case else 0,
            "key_created_or_null": bool(terminal_case),
            "retry_authorized": not terminal_case,
            "controller_synthesized": False,
            "previous_ledger_entry_sha256": "5" * 64,
            "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
        }
        failure = probe.validate_probe_failure_marker(
            probe.runtime.canonical_json_bytes(failure_value),
            contract=contract,
            active_segment=active,
            attempt=attempt,
            expected_previous_ledger_sha256="5" * 64,
        )
        if receipt_kind == "probe-failure":
            value = failure_value
            call = probe.validate_probe_failure_marker
            kwargs = {
                "contract": contract,
                "active_segment": active,
                "attempt": attempt,
                "expected_previous_ledger_sha256": "5" * 64,
            }
        else:
            value = {
                "schema_version": probe.SCHEMA_VERSION,
                "namespace": probe.NAMESPACE,
                "receipt_kind": "probe-terminal",
                "status": "terminal-probe-integrity-failure",
                "slot_index": 0,
                "attempt_ordinal": 0,
                "probe_failure_marker_sha256_and_bytes": failure.identity.as_dict(),
                "recorded_at": _time(3),
                "previous_ledger_entry_sha256": "6" * 64,
                "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
            }
            call = probe.validate_probe_terminal_marker
            kwargs = {
                "contract": contract,
                "active_segment": active,
                "failure": failure,
                "expected_previous_ledger_sha256": "6" * 64,
            }

    def _rebuild(mutated: dict[str, Any]) -> bytes:
        if "resolution_id" in mutated:
            mutated["resolution_id"] = ""
            mutated["resolution_id"] = probe.runtime.derive_slot_resolution_id(mutated)
        return probe.runtime.canonical_json_bytes(mutated)

    # Sanity: the unmutated receipt is accepted, so the rejections below are
    # caused by the retired constants and nothing else.
    call(_rebuild(dict(value)), **kwargs)
    for field, retired in (
        ("schema_version", 3),
        ("namespace", "confirmatory-holdout-v3"),
    ):
        mutated = dict(value)
        mutated[field] = retired
        _failure(probe, call, _rebuild(mutated), **kwargs)


def test_every_probe_domain_separator_is_v4_scoped(probe: Any) -> None:
    for domain in (probe.IDENTITY_DOMAIN, probe.PARTITION_DOMAIN):
        assert domain.startswith(b"confirmatory-holdout-v4/")
        assert domain.endswith(b"\0")
        for retired in (b"confirmatory-holdout-v2/", b"confirmatory-holdout-v3/"):
            assert retired not in domain


def test_fresh_keys_leave_the_aggregate_and_partition_unchanged(
    probe: Any, tmp_path: Path
) -> None:
    """Repeatability: a second attempt keys differently and counts identically.

    Each invocation mints its own 256-bit key inside the isolated worker, so
    equality classes are computed under a different HMAC every time.  The
    frozen counts must not notice.
    """

    events = _ready_events()
    first = _compute(probe, tmp_path=tmp_path / "a", local_events=events)
    second = _compute(probe, tmp_path=tmp_path / "b", local_events=events)
    assert first.aggregate() == second.aggregate()
    assert first.provisional_ready is True and second.provisional_ready is True
    # The worker envelopes are independent objects, and neither carries a key,
    # a token, or a per-event record.
    for envelope in (first.worker_envelope_raw, second.worker_envelope_raw):
        assert b"hmac" not in envelope.lower()
        assert b"token" not in envelope.lower()
        assert b"PRIVATE" not in envelope
