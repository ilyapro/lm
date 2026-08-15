"""Synthetic adversarial tests for the confirmatory-v4 source snapshot launcher.

Every source in this file is a SQLite database this module builds in a
temporary directory.  The campaign is preregistered source-blind, so no test
here may open a configured production database or reach any host: doing so
would consume source-blindness and invalidate the frozen protocol.  The
launcher is proven instead against the frozen probe's own capture boundary,
worker, and receipt validators.
"""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import pytest

from living_memory.storage import MemoryStore


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ap_confirmatory_snapshot_v4.py"
ACCRUAL_SCRIPT = REPO_ROOT / "scripts" / "ap_confirmatory_accrual_v4.py"
NAMESPACE_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "animal-planet"
    / "evaluation"
    / "confirmatory-holdout-v4"
)
FROZEN_PLAN = NAMESPACE_ROOT / "analysis-plan.json"

RELEASE = "2026-08-14T15:03:46.793603Z"
SLOT_ZERO_AT = "2026-08-17T00:00:00.000000Z"
SLOT_ZERO_GRACE_AT = "2026-08-17T06:00:00.000000Z"
IN_WINDOW_AT = "2026-08-17T00:30:00.000000Z"
EVENT_AT = "2026-08-16T00:00:00.000000Z"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def snapshot() -> Any:
    return _load("ap_confirmatory_snapshot_v4_test", SCRIPT)


@pytest.fixture(scope="module")
def accrual() -> Any:
    return _load("ap_confirmatory_accrual_v4_snapshot_test", ACCRUAL_SCRIPT)


def _failure(module: Any, call: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Every rejection is the shared, message-free integrity signal."""

    with pytest.raises(module.IntegrityFailure) as caught:
        call(*args, **kwargs)
    assert caught.value.args == ()
    assert str(caught.value) == ""


# ---------------------------------------------------------------------------
# Synthetic sources
# ---------------------------------------------------------------------------

EVENT_INSERT_COLUMNS = (
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


def _event(
    event_id: str,
    *,
    query: str = "synthetic query",
    scope: str = "project:blue",
    agent: str | None = None,
    transport: str = "transport-a",
    created_at: str = EVENT_AT,
) -> tuple[Any, ...]:
    ambient: dict[str, Any] = {"transport_session_id": transport}
    if agent is not None:
        ambient["agent"] = agent
    resolved = [scope, "global"] if scope != "global" else ["global"]
    values = {
        "id": event_id,
        "query": query,
        "scope": scope,
        "requested_scope": scope,
        "resolved_scopes": json.dumps(resolved),
        "ambient_context": json.dumps(ambient),
        "depth": "1",
        "max_results": 5,
        "results": json.dumps([{"private": "PRIVATE-OUTCOME"}]),
        "agent": agent,
        "task": None,
        "session_id": None,
        "transport_session_id": transport,
        "feedback_applied": 1,
        "feedback_applied_at": "2099-01-01T00:00:00Z",
        "gated": 1,
        "created_at": created_at,
    }
    return tuple(values[column] for column in EVENT_INSERT_COLUMNS)


def _events_for(marker: str) -> list[tuple[Any, ...]]:
    """A small population whose shape exercises both partitions."""

    rows = []
    for index in range(4):
        rows.append(
            _event(
                f"{marker}-auto-{index}",
                transport=f"transport-{index % 2}",
            )
        )
    for index in range(2):
        rows.append(
            _event(
                f"{marker}-organic-{index}",
                query=f"organic {marker} {index}",
                agent="claude",
                transport=f"transport-{index}",
            )
        )
    return rows


def _build_source(
    path: Path,
    marker: str,
    *,
    rows: Iterable[tuple[Any, ...]] | None = None,
    journal_mode: str = "WAL",
    drop_policy_rows: bool = False,
) -> None:
    """Create one production-shaped synthetic source database."""

    # Production creates the schema and its four retrieval-policy defaults.
    with MemoryStore(path):
        pass
    connection = sqlite3.connect(path)
    try:
        connection.execute(f"PRAGMA journal_mode={journal_mode}")
        connection.execute(
            "INSERT INTO kv (key,value,updated_at) VALUES (?,?,?)",
            (f"source-{marker}", marker, "2026-08-15T00:00:00Z"),
        )
        if drop_policy_rows:
            connection.execute("DELETE FROM retrieval_weights")
        payload = list(_events_for(marker) if rows is None else rows)
        if payload:
            connection.executemany(
                f"INSERT INTO recall_events ({','.join(EVENT_INSERT_COLUMNS)}) "
                f"VALUES ({','.join('?' for _ in EVENT_INSERT_COLUMNS)})",
                payload,
            )
        connection.commit()
    finally:
        connection.close()


def _database_input(alias: str) -> dict[str, Any]:
    """The exact private statx input the plan declares, synthesized per alias."""

    tail = 1 + (alias == "alt")
    return {
        "filesystem_uuid": f"00000000-0000-4000-8000-{tail:012d}",
        "statx_inode_uint64": 70_000 + tail,
        "statx_birthtime_ns_int64": 1_700_000_000_000_000_000 + tail,
    }


def _attested(snapshot: Any, alias: str) -> str:
    return snapshot.runtime.derive_database_instance_identity(
        _database_input(alias)
    )


class _Sources:
    """One synthetic source pair plus the launcher inputs that address it."""

    def __init__(self, snapshot: Any, root: Path) -> None:
        self.root = root
        self.snapshot = snapshot
        self.paths: dict[str, Path] = {}
        self.observations: list[str] = []
        self.identity_inputs: dict[str, dict[str, Any]] = {
            alias: _database_input(alias) for alias in snapshot.SOURCE_ALIASES
        }
        root.mkdir(parents=True, exist_ok=True)
        for alias in snapshot.SOURCE_ALIASES:
            path = root / f"synthetic-{alias}.sqlite3"
            _build_source(path, alias)
            self.paths[alias] = path

    def locators(self) -> dict[str, Any]:
        return {
            alias: self.snapshot.SourceLocator(alias, self.paths[alias])
            for alias in self.snapshot.SOURCE_ALIASES
        }

    def observe(self, alias: str) -> dict[str, Any]:
        self.observations.append(f"observe:{alias}")
        return dict(self.identity_inputs[alias])

    def launcher(self, *, window: Any = None, staging: str = "staging") -> Any:
        return self.snapshot.SourceSnapshotLauncher(
            window=window
            if window is not None
            else self.snapshot.open_slot_window(0, IN_WINDOW_AT),
            locators=self.locators(),
            observe_database_identity=self.observe,
            staging_root=self.root / staging,
        )

    def capture_all(self, launcher: Any) -> dict[str, int]:
        launcher.mark_keyed_worker_handoff()
        return {
            alias: launcher.capture_snapshot(
                alias,
                self.snapshot.ALIAS_IDS[alias],
                _attested(self.snapshot, alias),
            )
            for alias in self.snapshot.SOURCE_ALIASES
        }


@pytest.fixture
def sources(snapshot: Any, tmp_path: Path) -> Iterator[_Sources]:
    built = _Sources(snapshot, tmp_path / "sources")
    yield built


# ---------------------------------------------------------------------------
# The frozen slot window is the only thing that authorizes a source open
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("observed_at", "accepted"),
    (
        ("2026-08-16T23:59:59.999999Z", False),
        (SLOT_ZERO_AT, True),
        (IN_WINDOW_AT, True),
        ("2026-08-17T05:59:59.999999Z", True),
        (SLOT_ZERO_GRACE_AT, False),
        ("2026-08-17T06:00:00.000001Z", False),
    ),
)
def test_slot_window_is_the_frozen_half_open_grace_interval(
    snapshot: Any, observed_at: str, accepted: bool
) -> None:
    if not accepted:
        _failure(snapshot, snapshot.open_slot_window, 0, observed_at)
        return
    window = snapshot.open_slot_window(0, observed_at)
    assert window.slot_index == 0
    assert window.scheduled_at == SLOT_ZERO_AT
    assert window.grace_deadline_at == SLOT_ZERO_GRACE_AT
    assert window.observed_at == observed_at


def test_slot_window_bounds_come_from_the_frozen_schedule(snapshot: Any) -> None:
    runtime = snapshot.runtime
    for slot_index in (0, 1, runtime.LAST_SLOT_INDEX):
        slot = runtime.slot_times(slot_index)
        window = snapshot.open_slot_window(slot_index, slot.scheduled_at)
        assert (window.scheduled_at, window.grace_deadline_at) == (
            slot.scheduled_at,
            slot.grace_deadline_at,
        )
    # The campaign is exactly 29 slots; there is no slot 29 to launch into.
    for rejected in (-1, runtime.LAST_SLOT_INDEX + 1, True, "0", 1.0, None):
        _failure(snapshot, snapshot.open_slot_window, rejected, SLOT_ZERO_AT)
    for malformed in ("2026-08-17T00:00:00Z", "2026-08-17 00:00:00.000000Z", ""):
        _failure(snapshot, snapshot.open_slot_window, 0, malformed)


def test_real_clock_window_refuses_a_slot_that_has_not_opened(
    snapshot: Any,
) -> None:
    """The real-clock gate is not a formality: today is before slot 0."""

    now = datetime.now(UTC)
    assert now < snapshot.runtime.parse_utc(SLOT_ZERO_AT, receipt=True)
    _failure(snapshot, snapshot.open_slot_window_now, 0)


def test_the_real_clock_gate_exposes_no_override_surface(snapshot: Any) -> None:
    import inspect

    parameters = inspect.signature(snapshot.open_slot_window_now).parameters
    assert list(parameters) == ["slot_index"]
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "environ" not in names
    assert "getenv" not in names


def test_a_launcher_cannot_exist_without_a_proven_window(
    snapshot: Any, sources: _Sources, tmp_path: Path
) -> None:
    forged = snapshot.SlotWindow(
        slot_index=0,
        scheduled_at=SLOT_ZERO_AT,
        grace_deadline_at=SLOT_ZERO_GRACE_AT,
        observed_at="2026-08-16T00:00:00.000000Z",
    )
    for window in (None, forged, "2026-08-17T00:30:00.000000Z"):
        _failure(
            snapshot,
            snapshot.SourceSnapshotLauncher,
            window=window,
            locators=sources.locators(),
            observe_database_identity=sources.observe,
            staging_root=tmp_path / "never",
        )
    assert not (tmp_path / "never").exists()
    assert sources.observations == []


def test_a_forged_window_is_refused_before_any_source_open(
    snapshot: Any, sources: _Sources
) -> None:
    launcher = sources.launcher()
    launcher.mark_keyed_worker_handoff()
    # A frozen dataclass is still constructible by a caller, so the launcher
    # re-derives the window from the frozen schedule on every capture.
    launcher._window = snapshot.SlotWindow(
        slot_index=0,
        scheduled_at=SLOT_ZERO_AT,
        grace_deadline_at=SLOT_ZERO_GRACE_AT,
        observed_at=SLOT_ZERO_GRACE_AT,
    )
    _failure(
        snapshot,
        launcher.capture_snapshot,
        "local",
        snapshot.ALIAS_IDS["local"],
        _attested(snapshot, "local"),
    )
    assert launcher.source_open_count == 0
    assert sources.observations == []
    launcher.destroy()


# ---------------------------------------------------------------------------
# The provable pre-source failure path
# ---------------------------------------------------------------------------


def test_failure_facts_are_exactly_the_frozen_probe_phase_ledger(
    snapshot: Any,
) -> None:
    probe = snapshot.probe
    for phase, count, key_created in snapshot.LAUNCHER_PHASE_PROGRESS:
        assert probe.PHASE_PROGRESS[phase] == (count, key_created)
        facts = snapshot.failure_facts_for(count, key_created)
        assert facts.phase_at_failure == phase
        retryable = phase == "pre-key-pre-source"
        assert facts.retry_authorized is retryable
        assert facts.failure_class == (
            "pre-source-retryable" if retryable else "post-source-terminal"
        )
    # Only the one pre-key, pre-source state authorizes an in-grace retry.
    assert [
        phase
        for phase, count, key in snapshot.LAUNCHER_PHASE_PROGRESS
        if snapshot.failure_facts_for(count, key).retry_authorized
    ] == ["pre-key-pre-source"]
    for bad in ((0, 0), (1, False), (3, True), (True, True), (-1, True)):
        _failure(snapshot, snapshot.failure_facts_for, *bad)


def test_launcher_phase_alignment_guard_rejects_probe_drift(
    snapshot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    drifted = dict(snapshot.probe.PHASE_PROGRESS)
    drifted["pre-key-pre-source"] = (0, True)
    monkeypatch.setattr(snapshot.probe, "PHASE_PROGRESS", drifted)
    _failure(snapshot, snapshot._require_frozen_phase_alignment)


def test_failure_before_the_keyed_handoff_authorizes_the_in_grace_retry(
    snapshot: Any, sources: _Sources
) -> None:
    launcher = sources.launcher()
    facts = launcher.failure_facts()
    assert (facts.source_open_count, facts.key_created) == (0, False)
    assert facts.phase_at_failure == "pre-key-pre-source"
    assert facts.retry_authorized is True
    assert set(facts.marker_fields()).issubset(
        snapshot.probe.PROBE_FAILURE_FIELDS
    )
    launcher.destroy()
    # The proof survives destruction: the driver needs it precisely then.
    assert launcher.failure_facts() == facts


def test_failure_after_the_keyed_handoff_is_terminal(
    snapshot: Any, sources: _Sources
) -> None:
    launcher = sources.launcher()
    launcher.mark_keyed_worker_handoff()
    facts = launcher.failure_facts()
    assert facts.phase_at_failure == "key-created-pre-source"
    assert facts.source_open_count == 0
    assert facts.retry_authorized is False
    assert facts.failure_class == "post-source-terminal"
    # The handoff is recorded once and never after a capture has begun.
    _failure(snapshot, launcher.mark_keyed_worker_handoff)
    launcher.destroy()


def test_an_unusable_locator_fails_while_the_retry_is_still_authorized(
    snapshot: Any, sources: _Sources, tmp_path: Path
) -> None:
    """Everything checkable without a source is checked before the handoff."""

    missing = tmp_path / "absent.sqlite3"
    directory = tmp_path / "a-directory"
    directory.mkdir()
    link = tmp_path / "link.sqlite3"
    link.symlink_to(sources.paths["local"])
    for bad in (missing, directory, link, Path("relative.sqlite3")):
        locators = sources.locators()
        locators["alt"] = snapshot.SourceLocator("alt", bad)
        _failure(
            snapshot,
            snapshot.SourceSnapshotLauncher,
            window=snapshot.open_slot_window(0, IN_WINDOW_AT),
            locators=locators,
            observe_database_identity=sources.observe,
            staging_root=tmp_path / f"staging-{bad.name}",
        )
    # No launcher exists, so nothing was opened and nothing was keyed.
    facts = snapshot.failure_facts_for(0, False)
    assert facts.retry_authorized is True


def test_two_aliases_may_never_name_one_file(
    snapshot: Any, sources: _Sources, tmp_path: Path
) -> None:
    mirrored = sources.locators()
    mirrored["alt"] = snapshot.SourceLocator("alt", sources.paths["local"])
    _failure(
        snapshot,
        snapshot.SourceSnapshotLauncher,
        window=snapshot.open_slot_window(0, IN_WINDOW_AT),
        locators=mirrored,
        observe_database_identity=sources.observe,
        staging_root=tmp_path / "mirrored",
    )


def test_malformed_locator_maps_are_refused(
    snapshot: Any, sources: _Sources, tmp_path: Path
) -> None:
    base = sources.locators()
    variants: list[Any] = [
        {},
        {"local": base["local"]},
        {**base, "extra": base["local"]},
        {**base, "alt": snapshot.SourceLocator("local", sources.paths["alt"])},
        {**base, "alt": sources.paths["alt"]},
        [base["local"], base["alt"]],
    ]
    for index, locators in enumerate(variants):
        _failure(
            snapshot,
            snapshot.SourceSnapshotLauncher,
            window=snapshot.open_slot_window(0, IN_WINDOW_AT),
            locators=locators,
            observe_database_identity=sources.observe,
            staging_root=tmp_path / f"variant-{index}",
        )


def test_the_pre_source_marker_the_launcher_proves_is_validator_valid(
    snapshot: Any, sources: _Sources
) -> None:
    """Build the frozen ``probe-failure`` receipt from the launcher's facts.

    This is the whole point of the pre-source failure path: ``POLICY.md``
    authorizes an in-grace retry only for a marker that proves failure before
    key creation and before any source open, and only the frozen probe decides
    whether such a marker is valid.
    """

    probe = snapshot.probe
    runtime = snapshot.runtime
    contract = runtime.load_frozen_contract()
    observer = runtime.hash_and_bytes(b"synthetic-runtime-observer-v4")
    slot = runtime.slot_times(0)
    launched = runtime.parse_utc(slot.scheduled_at, receipt=True) + timedelta(
        seconds=30
    )
    grace = runtime.parse_utc(slot.grace_deadline_at, receipt=True)
    watchdog = min(launched + timedelta(seconds=3600), grace - timedelta(seconds=60))
    attempt_value = {
        "schema_version": probe.SCHEMA_VERSION,
        "namespace": probe.NAMESPACE,
        "receipt_kind": "probe-attempt",
        "slot_index": 0,
        "scheduled_at": slot.scheduled_at,
        "grace_deadline_at": slot.grace_deadline_at,
        "attempt_ordinal": 0,
        "launched_at": runtime.canonical_utc(launched),
        "watchdog_deadline_at": runtime.canonical_utc(watchdog),
        "segment_id": contract.initial_segment_id,
        "previous_ledger_entry_sha256": "4" * 64,
        "runtime_observer_sha256_and_bytes": observer.as_dict(),
        "probe_sha256_and_bytes": probe._self_identity().as_dict(),
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
    }
    attempt = probe.validate_probe_attempt_marker(
        runtime.canonical_json_bytes(attempt_value),
        contract=contract,
        expected_segment_id=contract.initial_segment_id,
        expected_runtime_observer_identity=observer,
        expected_previous_ledger_sha256="4" * 64,
        expected_slot_index=0,
        expected_attempt_ordinal=0,
    )

    launcher = sources.launcher()
    failure_value = {
        "schema_version": probe.SCHEMA_VERSION,
        "namespace": probe.NAMESPACE,
        "receipt_kind": "probe-failure",
        "slot_index": 0,
        "attempt_ordinal": 0,
        "probe_attempt_marker_sha256_and_bytes": attempt.identity.as_dict(),
        "failed_at": runtime.canonical_utc(launched + timedelta(seconds=5)),
        "controller_synthesized": False,
        "previous_ledger_entry_sha256": "5" * 64,
        "analysis_plan_sha256_and_bytes": contract.plan_identity.as_dict(),
        **launcher.failure_facts().marker_fields(),
    }
    validated = probe.validate_probe_failure_marker(
        runtime.canonical_json_bytes(failure_value),
        contract=contract,
        active_segment=None,
        attempt=attempt,
        expected_previous_ledger_sha256="5" * 64,
    )
    assert validated.phase_at_failure == "pre-key-pre-source"
    assert validated.failure_class == "pre-source-retryable"
    assert validated.retry_authorized is True

    # After the handoff the same construction yields a terminal marker, and a
    # terminal marker is exactly what forbids the retry.
    launcher.mark_keyed_worker_handoff()
    terminal_value = {
        **failure_value,
        **launcher.failure_facts().marker_fields(),
    }
    terminal = probe.validate_probe_failure_marker(
        runtime.canonical_json_bytes(terminal_value),
        contract=contract,
        active_segment=None,
        attempt=attempt,
        expected_previous_ledger_sha256="5" * 64,
    )
    assert terminal.retry_authorized is False
    assert terminal.failure_class == "post-source-terminal"
    launcher.destroy()


# ---------------------------------------------------------------------------
# The probe's already-open-handle contract
# ---------------------------------------------------------------------------


def test_the_handed_descriptor_is_what_the_frozen_probe_opens(
    snapshot: Any, sources: _Sources
) -> None:
    probe = snapshot.probe
    launcher = sources.launcher()
    descriptors = sources.capture_all(launcher)
    try:
        for alias, fd in descriptors.items():
            # The probe duplicates the descriptor and demands O_RDONLY on a
            # regular file...
            duplicate = probe._duplicate_readonly_regular_fd(fd)
            try:
                # ...then opens it immutable and validates the whole schema.
                connection = probe._open_snapshot(duplicate)
                try:
                    rows = connection.execute(
                        "SELECT count(*) FROM main.recall_events"
                    ).fetchone()[0]
                    assert rows == len(_events_for(alias))
                finally:
                    connection.close()
            finally:
                os.close(duplicate)
            # The launcher keeps ownership; the probe never closes this one.
            assert probe._proof_fd(fd).sha256 == (
                launcher.snapshot_identities()[alias].sha256
            )
    finally:
        launcher.destroy()


def test_the_capture_callback_matches_the_frozen_probe_boundary(
    snapshot: Any, sources: _Sources
) -> None:
    """Drive ``probe._capture_snapshot_fds`` with the launcher's own callback.

    The probe calls the boundary positionally with the exact alias, the alias
    authority id, and the attested database identity taken from the active
    segment, so this proves the signature and the identity binding together.
    """

    probe = snapshot.probe
    contract, active = _active_segment(snapshot)
    assert active.source_binding.core.database_map() == {
        alias: _attested(snapshot, alias) for alias in snapshot.SOURCE_ALIASES
    }
    launcher = sources.launcher()
    launcher.mark_keyed_worker_handoff()
    try:
        captured = probe._capture_snapshot_fds(active, launcher.capture_snapshot)
        assert captured == launcher.snapshot_fds()
        assert sources.observations == [
            "observe:local",
            "observe:alt",
            "observe:local",
            "observe:alt",
        ]
    finally:
        launcher.destroy()
    del contract


def test_the_frozen_worker_computes_an_aggregate_over_launcher_handles(
    snapshot: Any, sources: _Sources, tmp_path: Path
) -> None:
    """End to end through the real isolated ``WORKER_BOOTSTRAP`` subprocess."""

    probe = snapshot.probe
    plan_raw, plan_sha = _synthetic_plan(tmp_path)
    dev = tmp_path / "dev.jsonl"
    dev.write_bytes(b"")
    dev.chmod(0o400)
    dev_fd = os.open(dev, os.O_RDONLY)
    launcher = sources.launcher()
    launcher.mark_keyed_worker_handoff()
    try:
        computation = probe._compute_aggregate_supplier(
            plan_raw=plan_raw,
            expected_plan_sha256=plan_sha,
            source_input_supplier=lambda: {
                alias: launcher.capture_snapshot(
                    alias,
                    snapshot.ALIAS_IDS[alias],
                    _attested(snapshot, alias),
                )
                for alias in snapshot.SOURCE_ALIASES
            },
            dev_input_fd=dev_fd,
            slot_index=0,
            active_lower_bound_exclusive_at=RELEASE,
        )
        counts = computation.aggregate()
        assert counts["selected_event_count"] == 2 * len(_events_for("local"))
        assert counts["selected_event_count"] == (
            counts["selected_replayable_event_count"]
            + counts["selected_nonreplayable_event_count"]
        )
        # The worker hashed the very descriptors the launcher latched.
        assert computation.snapshots() == launcher.snapshot_identities()
    finally:
        os.close(dev_fd)
        launcher.destroy()


def test_capture_is_refused_before_the_keyed_handoff(
    snapshot: Any, sources: _Sources
) -> None:
    launcher = sources.launcher()
    _failure(
        snapshot,
        launcher.capture_snapshot,
        "local",
        snapshot.ALIAS_IDS["local"],
        _attested(snapshot, "local"),
    )
    assert launcher.source_open_count == 0
    assert sources.observations == []
    launcher.destroy()


def test_capture_enforces_exactly_one_snapshot_per_alias_in_frozen_order(
    snapshot: Any, sources: _Sources
) -> None:
    launcher = sources.launcher()
    launcher.mark_keyed_worker_handoff()
    try:
        # ``alt`` may not go first: the frozen order is local then alt.
        _failure(
            snapshot,
            launcher.capture_snapshot,
            "alt",
            snapshot.ALIAS_IDS["alt"],
            _attested(snapshot, "alt"),
        )
        for alias, alias_id, attested in (
            ("global", snapshot.ALIAS_IDS["local"], _attested(snapshot, "local")),
            ("local", snapshot.ALIAS_IDS["alt"], _attested(snapshot, "local")),
            ("local", "confirmatory-local-v3-ro", _attested(snapshot, "local")),
            ("local", snapshot.ALIAS_IDS["local"], _attested(snapshot, "alt")),
            ("local", snapshot.ALIAS_IDS["local"], "0" * 63),
            ("local", snapshot.ALIAS_IDS["local"], None),
        ):
            _failure(
                snapshot, launcher.capture_snapshot, alias, alias_id, attested
            )
        launcher.capture_snapshot(
            "local", snapshot.ALIAS_IDS["local"], _attested(snapshot, "local")
        )
        # A second ``local`` is a repeat capture, which the policy forbids.
        _failure(
            snapshot,
            launcher.capture_snapshot,
            "local",
            snapshot.ALIAS_IDS["local"],
            _attested(snapshot, "local"),
        )
        launcher.capture_snapshot(
            "alt", snapshot.ALIAS_IDS["alt"], _attested(snapshot, "alt")
        )
        assert launcher.latched is True
        # There is no third alias and no recapture once both are latched.
        _failure(
            snapshot,
            launcher.capture_snapshot,
            "local",
            snapshot.ALIAS_IDS["local"],
            _attested(snapshot, "local"),
        )
    finally:
        launcher.destroy()


# ---------------------------------------------------------------------------
# Snapshot properties
# ---------------------------------------------------------------------------


def test_a_live_wal_source_yields_a_self_contained_rollback_journal_snapshot(
    snapshot: Any, sources: _Sources
) -> None:
    """A raw byte copy of a WAL database could not satisfy ``immutable=1``."""

    writer = sqlite3.connect(sources.paths["local"])
    try:
        assert writer.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        launcher = sources.launcher()
        try:
            sources.capture_all(launcher)
            for path in launcher.snapshot_paths().values():
                header = path.read_bytes()[:100]
                assert header[:16] == b"SQLite format 3\0"
                assert (header[18], header[19]) == (1, 1)
                assert not Path(f"{path}-wal").exists()
                assert not Path(f"{path}-journal").exists()
        finally:
            launcher.destroy()
    finally:
        writer.close()


def test_capture_never_writes_to_a_source(
    snapshot: Any, sources: _Sources
) -> None:
    before = {
        alias: (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for alias, path in sources.paths.items()
    }
    launcher = sources.launcher()
    try:
        sources.capture_all(launcher)
    finally:
        launcher.destroy()
    after = {
        alias: (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for alias, path in sources.paths.items()
    }
    assert before == after


def test_snapshots_and_their_staging_root_are_private(
    snapshot: Any, sources: _Sources
) -> None:
    launcher = sources.launcher()
    try:
        sources.capture_all(launcher)
        paths = launcher.snapshot_paths()
        root = sources.root / "staging"
        assert root.stat().st_mode & 0o777 == 0o700
        for path in paths.values():
            assert path.parent == root
            assert path.stat().st_mode & 0o777 == 0o400
    finally:
        launcher.destroy()


def test_equal_snapshot_digests_are_a_fatal_integrity_error(
    snapshot: Any, sources: _Sources
) -> None:
    """``sources.equal_snapshot_sha256`` is ``fatal-integrity-error``."""

    shutil.copyfile(sources.paths["local"], sources.paths["alt"])
    launcher = sources.launcher()
    launcher.mark_keyed_worker_handoff()
    try:
        launcher.capture_snapshot(
            "local", snapshot.ALIAS_IDS["local"], _attested(snapshot, "local")
        )
        _failure(
            snapshot,
            launcher.capture_snapshot,
            "alt",
            snapshot.ALIAS_IDS["alt"],
            _attested(snapshot, "alt"),
        )
        # The rejected snapshot leaves no bytes behind to be mistaken for one.
        assert not (sources.root / "staging" / "alt.sqlite3").exists()
        assert launcher.latched is False
        _failure(snapshot, launcher.snapshot_set_identity)
    finally:
        launcher.destroy()


@pytest.mark.parametrize("defect", ("not-sqlite", "no-policy-rows", "unreadable"))
def test_an_unusable_source_fails_the_launcher_not_the_worker(
    snapshot: Any, sources: _Sources, defect: str
) -> None:
    """The launcher opens the snapshot exactly as the probe will, first."""

    target = sources.paths["alt"]
    if defect == "not-sqlite":
        target.write_bytes(b"not a database at all")
    elif defect == "no-policy-rows":
        target.unlink()
        _build_source(target, "alt", drop_policy_rows=True)
    else:
        target.unlink()

    launcher = sources.launcher() if defect != "unreadable" else None
    if launcher is None:
        # A vanished source cannot even build a launcher, so it stays retryable.
        _failure(
            snapshot,
            snapshot.SourceSnapshotLauncher,
            window=snapshot.open_slot_window(0, IN_WINDOW_AT),
            locators=sources.locators(),
            observe_database_identity=sources.observe,
            staging_root=sources.root / "unreadable",
        )
        return
    launcher.mark_keyed_worker_handoff()
    try:
        launcher.capture_snapshot(
            "local", snapshot.ALIAS_IDS["local"], _attested(snapshot, "local")
        )
        _failure(
            snapshot,
            launcher.capture_snapshot,
            "alt",
            snapshot.ALIAS_IDS["alt"],
            _attested(snapshot, "alt"),
        )
        assert not (sources.root / "staging" / "alt.sqlite3").exists()
        assert launcher.failure_facts().retry_authorized is False
    finally:
        launcher.destroy()


# ---------------------------------------------------------------------------
# The database-instance identity boundary
# ---------------------------------------------------------------------------


def test_identity_is_observed_before_the_first_open_and_after_both_captures(
    snapshot: Any, sources: _Sources
) -> None:
    """``sources.database_instance_identity_derivation.probe_binding_capture``."""

    launcher = sources.launcher()
    launcher.mark_keyed_worker_handoff()
    try:
        assert sources.observations == []
        launcher.capture_snapshot(
            "local", snapshot.ALIAS_IDS["local"], _attested(snapshot, "local")
        )
        assert sources.observations == ["observe:local", "observe:alt"]
        launcher.capture_snapshot(
            "alt", snapshot.ALIAS_IDS["alt"], _attested(snapshot, "alt")
        )
        assert sources.observations == [
            "observe:local",
            "observe:alt",
            "observe:local",
            "observe:alt",
        ]
        assert launcher.database_instance_identities() == {
            alias: _attested(snapshot, alias)
            for alias in snapshot.SOURCE_ALIASES
        }
    finally:
        launcher.destroy()


def test_an_unstable_identity_across_the_capture_is_fatal(
    snapshot: Any, sources: _Sources
) -> None:
    """``pre_post_equal_within_segment`` is not advisory."""

    launcher = sources.launcher()
    launcher.mark_keyed_worker_handoff()
    try:
        launcher.capture_snapshot(
            "local", snapshot.ALIAS_IDS["local"], _attested(snapshot, "local")
        )
        # A replaced database between the two captures must not go unnoticed.
        sources.identity_inputs["local"]["statx_inode_uint64"] = 90_001
        _failure(
            snapshot,
            launcher.capture_snapshot,
            "alt",
            snapshot.ALIAS_IDS["alt"],
            _attested(snapshot, "alt"),
        )
        assert launcher.latched is False
    finally:
        launcher.destroy()


def test_two_aliases_resolving_to_one_database_instance_are_fatal(
    snapshot: Any, sources: _Sources
) -> None:
    """``values_distinct_across_aliases`` rejects a mirror before any open."""

    sources.identity_inputs["alt"] = dict(sources.identity_inputs["local"])
    launcher = sources.launcher()
    launcher.mark_keyed_worker_handoff()
    try:
        _failure(
            snapshot,
            launcher.capture_snapshot,
            "local",
            snapshot.ALIAS_IDS["local"],
            _attested(snapshot, "local"),
        )
        assert launcher.source_open_count == 0
    finally:
        launcher.destroy()


@pytest.mark.parametrize(
    "mutation",
    (
        {"filesystem_uuid": "00000000-0000-0000-0000-000000000000"},
        {"filesystem_uuid": "not-a-uuid"},
        {"statx_inode_uint64": 0},
        {"statx_inode_uint64": True},
        {"statx_birthtime_ns_int64": 0},
        {"extra": 1},
    ),
)
def test_a_malformed_private_identity_input_is_fatal(
    snapshot: Any, sources: _Sources, mutation: dict[str, Any]
) -> None:
    sources.identity_inputs["local"].update(mutation)
    launcher = sources.launcher()
    launcher.mark_keyed_worker_handoff()
    try:
        _failure(
            snapshot,
            launcher.capture_snapshot,
            "local",
            snapshot.ALIAS_IDS["local"],
            _attested(snapshot, "local"),
        )
        assert launcher.source_open_count == 0
    finally:
        launcher.destroy()


def test_a_raising_identity_observer_is_the_shared_integrity_signal(
    snapshot: Any, sources: _Sources, tmp_path: Path
) -> None:
    def explode(alias: str) -> dict[str, Any]:
        raise RuntimeError(f"private detail about {alias}")

    launcher = snapshot.SourceSnapshotLauncher(
        window=snapshot.open_slot_window(0, IN_WINDOW_AT),
        locators=sources.locators(),
        observe_database_identity=explode,
        staging_root=tmp_path / "raising",
    )
    launcher.mark_keyed_worker_handoff()
    try:
        _failure(
            snapshot,
            launcher.capture_snapshot,
            "local",
            snapshot.ALIAS_IDS["local"],
            _attested(snapshot, "local"),
        )
        assert launcher.source_open_count == 0
    finally:
        launcher.destroy()


def test_the_raw_identity_input_is_never_retained(
    snapshot: Any, sources: _Sources
) -> None:
    """``raw_inputs_persisted`` is false: only the derived digest survives."""

    launcher = sources.launcher()
    try:
        sources.capture_all(launcher)
        private = sources.identity_inputs["local"]
        rendered = repr(vars(launcher))
        assert private["filesystem_uuid"] not in rendered
        assert str(private["statx_inode_uint64"]) not in rendered
        assert str(private["statx_birthtime_ns_int64"]) not in rendered
        assert _attested(snapshot, "local") in rendered
    finally:
        launcher.destroy()


# ---------------------------------------------------------------------------
# The snapshot-set identity
# ---------------------------------------------------------------------------


def _plan_snapshot_set_preimage(local: Any, alt: Any) -> bytes:
    """``domains.snapshot_set_preimage``, rebuilt from the frozen plan text."""

    plan = json.loads(FROZEN_PLAN.read_text(encoding="utf-8"))
    domain = plan["domains"]["snapshot_set_utf8"].encode("utf-8")
    body = json.dumps(
        {
            "local": {"sha256": local.sha256, "bytes": local.bytes},
            "alt": {"sha256": alt.sha256, "bytes": alt.bytes},
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return domain + b"\0" + body


def test_the_snapshot_set_identity_is_the_plan_preimage(
    snapshot: Any, accrual: Any, sources: _Sources
) -> None:
    launcher = sources.launcher()
    try:
        sources.capture_all(launcher)
        identities = launcher.snapshot_identities()
        preimage = _plan_snapshot_set_preimage(
            identities["local"], identities["alt"]
        )
        derived = launcher.snapshot_set_identity()
        # ``snapshot_set_value``: the digest is of the preimage and the byte
        # count is the preimage's own length, not any snapshot's length.
        assert derived.sha256 == hashlib.sha256(preimage).hexdigest()
        assert derived.bytes == len(preimage)
        assert derived.bytes != identities["local"].bytes
        # The independently written accrual derivation must agree exactly.
        assert derived == accrual.derive_snapshot_set_identity(
            {
                alias: identities[alias].as_dict()
                for alias in snapshot.SOURCE_ALIASES
            }
        )
        assert snapshot.SNAPSHOT_SET_DOMAIN.decode("utf-8") == json.loads(
            FROZEN_PLAN.read_text(encoding="utf-8")
        )["domains"]["snapshot_set_utf8"]
    finally:
        launcher.destroy()


def test_the_snapshot_set_identity_requires_two_distinct_nonempty_members(
    snapshot: Any
) -> None:
    runtime = snapshot.runtime
    local = runtime.HashAndBytes("a" * 64, 4096)
    alt = runtime.HashAndBytes("b" * 64, 8192)
    assert snapshot.derive_snapshot_set_identity({"local": local, "alt": alt})
    assert snapshot.derive_snapshot_set_identity(
        {"local": local.as_dict(), "alt": alt.as_dict()}
    ) == snapshot.derive_snapshot_set_identity({"local": local, "alt": alt})
    for rejected in (
        {"local": local, "alt": local},
        {"local": local, "alt": runtime.HashAndBytes("b" * 64, 0)},
        {"local": local},
        {"local": local, "alt": alt, "extra": alt},
        {"local": local, "remote": alt},
        {"local": local.as_dict(), "alt": {"sha256": "b" * 64}},
        {"local": local.as_dict(), "alt": {"sha256": "B" * 64, "bytes": 1}},
    ):
        _failure(snapshot, snapshot.derive_snapshot_set_identity, rejected)


def test_the_snapshot_set_is_unavailable_until_both_aliases_latch(
    snapshot: Any, sources: _Sources
) -> None:
    launcher = sources.launcher()
    launcher.mark_keyed_worker_handoff()
    try:
        _failure(snapshot, launcher.snapshot_set_identity)
        _failure(snapshot, launcher.snapshot_fds)
        _failure(snapshot, launcher.snapshot_paths)
        launcher.capture_snapshot(
            "local", snapshot.ALIAS_IDS["local"], _attested(snapshot, "local")
        )
        _failure(snapshot, launcher.snapshot_set_identity)
        launcher.capture_snapshot(
            "alt", snapshot.ALIAS_IDS["alt"], _attested(snapshot, "alt")
        )
        assert launcher.snapshot_set_identity().bytes > 0
    finally:
        launcher.destroy()


# ---------------------------------------------------------------------------
# Namespace, destruction, and source-blindness hygiene
# ---------------------------------------------------------------------------


def test_a_staging_root_inside_the_frozen_namespace_is_refused(
    snapshot: Any, sources: _Sources
) -> None:
    """That namespace is byte-locked to six allowlisted entries."""

    for candidate in (
        NAMESPACE_ROOT,
        NAMESPACE_ROOT / "staging",
        NAMESPACE_ROOT / "recipe" / "deep" / "staging",
    ):
        _failure(snapshot, snapshot.require_root_outside_namespace, candidate)
        _failure(
            snapshot,
            snapshot.SourceSnapshotLauncher,
            window=snapshot.open_slot_window(0, IN_WINDOW_AT),
            locators=sources.locators(),
            observe_database_identity=sources.observe,
            staging_root=candidate,
        )
        assert not candidate.exists() or candidate == NAMESPACE_ROOT
    # Nothing this module can do adds an entry to the closed namespace.
    assert set(entry.name for entry in NAMESPACE_ROOT.iterdir()) <= {
        "POLICY.md",
        "README.md",
        "analysis-plan.json",
        "segments",
        "probes",
        "recipe",
        "corpus",
        "manifest.json",
        "seal-receipt.json",
    }


def test_a_staging_root_must_be_absolute_and_new(
    snapshot: Any, sources: _Sources, tmp_path: Path
) -> None:
    existing = tmp_path / "already-there"
    existing.mkdir()
    for candidate in (existing, Path("relative-staging"), tmp_path / ".." / "x"):
        _failure(
            snapshot,
            snapshot.SourceSnapshotLauncher,
            window=snapshot.open_slot_window(0, IN_WINDOW_AT),
            locators=sources.locators(),
            observe_database_identity=sources.observe,
            staging_root=candidate,
        )


def test_destroy_closes_every_descriptor_and_removes_every_byte(
    snapshot: Any, sources: _Sources
) -> None:
    launcher = sources.launcher()
    descriptors = sources.capture_all(launcher)
    paths = launcher.snapshot_paths()
    root = sources.root / "staging"
    assert all(path.exists() for path in paths.values())
    launcher.destroy()
    for path in paths.values():
        assert not path.exists()
    assert not root.exists()
    for fd in descriptors.values():
        with pytest.raises(OSError):
            os.fstat(fd)
    # Idempotent, and the latched results are gone with the bytes.
    launcher.destroy()
    _failure(snapshot, launcher.snapshot_fds)
    _failure(snapshot, launcher.snapshot_set_identity)
    _failure(
        snapshot,
        launcher.capture_snapshot,
        "local",
        snapshot.ALIAS_IDS["local"],
        _attested(snapshot, "local"),
    )


def test_the_launcher_names_no_source_path_host_or_transport() -> None:
    """No literal in this module can address a real database.

    The launcher's alias locators are supplied by the caller precisely so the
    tracked file never carries an operator mapping.  An absolute path literal
    would be the first step back toward one.
    """

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    # A bare separator (``quote(..., safe="/")``) is not a path; a separator
    # followed by a segment is exactly the absolute path literal this module
    # must never contain.
    assert [text for text in literals if re.match(r"/[^/]", text)] == []
    forbidden = re.compile(
        r"/(home|root|Users|var|etc|opt|srv|mnt|media|tmp)(/|\b)"
    )
    assert [text for text in literals if forbidden.search(text)] == []
    assert [text for text in literals if ".sqlite3" in text and "/" in text] == []

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    # Nothing here may spawn a process or speak to a network peer.
    assert imported.isdisjoint({"socket", "subprocess", "shutil", "ssl", "http"})


def test_the_module_exposes_no_source_opening_command_line() -> None:
    """A path-only command line could open a source outside any window."""

    for arguments in ([], ["--local", "/does/not/matter"], ["capture", "0"]):
        completed = subprocess.run(  # noqa: S603 - fixed interpreter and script
            [sys.executable, "-B", os.fspath(SCRIPT), *arguments],
            capture_output=True,
            check=False,
            timeout=120,
        )
        assert completed.returncode == 2
        assert completed.stdout == b""
        assert completed.stderr == b""


def test_the_launcher_binds_the_frozen_alias_and_schema_constants(
    snapshot: Any
) -> None:
    runtime = snapshot.runtime
    assert snapshot.SOURCE_ALIASES == ("local", "alt")
    assert snapshot.ALIAS_IDS == {
        "local": "confirmatory-local-v4-ro",
        "alt": "confirmatory-alt-v4-ro",
    }
    assert snapshot.NAMESPACE == "confirmatory-holdout-v4"
    assert snapshot.SCHEMA_VERSION == 4
    plan = json.loads(FROZEN_PLAN.read_text(encoding="utf-8"))
    aliases = plan["sources"]["aliases"]
    assert snapshot.SNAPSHOT_OPEN_QUERY == aliases["local"]["snapshot_open"]
    assert snapshot.SNAPSHOT_OPEN_QUERY == aliases["alt"]["snapshot_open"]
    assert plan["sources"]["source_aliases_exactly"] == list(
        snapshot.SOURCE_ALIASES
    )
    # A retired namespace can never be reached from this module's domain.
    for prefix in runtime.RETIRED_DOMAIN_PREFIXES:
        assert not snapshot.SNAPSHOT_SET_DOMAIN.decode("utf-8").startswith(prefix)


# ---------------------------------------------------------------------------
# Shared synthetic protocol objects
# ---------------------------------------------------------------------------


def _synthetic_plan(tmp_path: Path, dev_raw: bytes = b"") -> tuple[bytes, str]:
    """The frozen plan with only the private dev-reference identity swapped."""

    plan = json.loads(FROZEN_PLAN.read_text(encoding="utf-8"))
    reference = plan["identity"]["complete_frozen_dev_reference"][
        "raw_identity_reference"
    ]
    reference["sha256"] = hashlib.sha256(dev_raw).hexdigest()
    reference["bytes"] = len(dev_raw)
    raw = (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return raw, hashlib.sha256(raw).hexdigest()


def _receipt_time(seconds: float) -> str:
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


def _active_segment(snapshot: Any) -> tuple[Any, Any]:
    """The synthetic initial active segment, built from frozen primitives."""

    runtime = snapshot.runtime
    probe = snapshot.probe
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
        "written_at": _receipt_time(0.010),
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
        initial_probe_launched_at=_receipt_time(0),
    )
    pairs = list(contract.initial_services.service_pairs)
    observation: dict[str, Any] = {}
    for index, alias in enumerate(snapshot.SOURCE_ALIASES):
        service, boot = pairs[index]
        observation[alias] = {
            "alias_id": runtime.ALIAS_IDS[alias],
            "authenticated_authority_identity": _authority(alias),
            "service_identity_sha256": service,
            "boot_identity_sha256": boot,
            # The same private input the launcher's observer returns, so the
            # attested identity and the observed identity must agree.
            "database_instance_identity": _database_input(alias),
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
        "pre_database_instance_identity_sha256_by_alias": (
            binding.core.database_map()
        ),
        "post_database_instance_identity_sha256_by_alias": (
            binding.core.database_map()
        ),
        "pre_alias_service_database_binding_sha256_by_alias": (
            binding.core.binding_map()
        ),
        "post_alias_service_database_binding_sha256_by_alias": (
            binding.core.binding_map()
        ),
        "active_services_state_sha256": contract.initial_services.identity.sha256,
        "source_binding_core_sha256_and_bytes": binding.core.identity.as_dict(),
        "status": "pass",
        "pre_observed_at": _receipt_time(0.020),
        "post_observed_at": _receipt_time(0.030),
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
